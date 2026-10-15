# [MODULE DOC]
# Purpose: pipeline-parallel schedules that drive forward/backward of a PipelineBlock-based model over
# microbatches.
# Main contents: PipelineEngine (abstract base with shared forward/backward helpers and validation),
# AllForwardAllBackwardPipelineEngine ("afab"), OneForwardOneBackwardPipelineEngine ("1f1b", Megatron
# PipeDream-flush style).
# How it fits: the trainer picks an engine from config and calls train_batch_iter / validate_batch_iter; the
# engine attaches a PipelineTrainBatchState to all blocks so P2P sends/recvs are queued and executed lazily.
# Import-time side effects: creates a module logger.
from abc import ABC, abstractmethod
from typing import Dict, Iterable, Optional, Union

import torch
from torch import nn as torch_nn
from torch.nn.parallel import DistributedDataParallel

from nanotron import distributed as dist
from nanotron import logging
from nanotron.distributed import ProcessGroup
from nanotron.logging import log_rank
from nanotron.optim.gradient_accumulator import GradientAccumulator
from nanotron.parallel.data_parallel.utils import ddp_trigger_sync_in_bwd
from nanotron.parallel.pipeline_parallel.context_manager import attach_pipeline_state_to_model
from nanotron.parallel.pipeline_parallel.state import PipelineTrainBatchState
from nanotron.parallel.pipeline_parallel.tensor_pointer import TensorPointer
from nanotron.utils import ContextManagers

# from nanotron.logging.timers import nanotron_timer

logger = logging.get_logger(__name__)


class PipelineEngine(ABC):
    # [CLASS DOC]
    # Purpose: abstract pipeline schedule. Key state: nb_microbatches (int or None, set by batch iterators and used
    # to normalize losses and decide DDP sync).
    # Subclasses implement train_batch_iter. String form is the class name.
    def __init__(self):
        # [FUNCTION DOC]
        # Sets nb_microbatches to None. Inputs/Outputs: None. Exceptions/Side effects: None.
        self.nb_microbatches: Optional[int] = None
        pass

    def forward(
        self,
        context: ContextManagers,
        state: PipelineTrainBatchState,
        micro_batch: Dict[str, Union[torch.Tensor, TensorPointer]],
        model: torch_nn.Module,
    ) -> Dict[str, Union[torch.Tensor, TensorPointer]]:
        # [FUNCTION DOC]
        # Behavior: increments state.nb_forwards, logs at debug, opens a new microbatch in the state, runs the model on
        # micro_batch inside context, wraps a non-dict output as {"loss": output}, divides a local loss by
        # nb_microbatches, asserts it requires grad and registers it as an activation requiring backward.
        # Inputs: context (ContextManagers), state (PipelineTrainBatchState), micro_batch (dict of tensors or
        # TensorPointers), model (nn.Module).
        # Outputs: output dict (loss is a tensor on the last stage, a TensorPointer elsewhere).
        # Exceptions: AssertionError if a local loss does not require grad; TypeError if nb_microbatches is None.
        # Side effects: mutates state; model forward queues P2P communications; logging.
        # Increment the number of backwards
        state.nb_forwards += 1
        log_rank(
            f"Forward micro batch id: {state.nb_forwards}",
            logger=logger,
            level=logging.DEBUG,
        )

        # IMPORTANT as it's basically the context manager storing all the intermediary activations
        state.new_micro_batch_forward()
        with context:
            output = model(**micro_batch)

        # We make `output` a dict
        if not isinstance(output, dict):
            output = {"loss": output}

        # We normalize our loss
        if not isinstance(output["loss"], TensorPointer):
            output["loss"] = output["loss"] / self.nb_microbatches

        # Add output as activations that require backward pass
        if not isinstance(output["loss"], TensorPointer):
            assert output["loss"].requires_grad
            state.register_activation_requiring_backward(output["loss"])
        return output

    @staticmethod
    def _get_fwd_context(model: torch_nn.Module):
        # [FUNCTION DOC]
        # HELPER: forward context for one microbatch.
        # Behavior: for a DDP model returns ContextManagers with model.no_sync (sync is triggered explicitly in backward),
        # else an empty ContextManagers.
        # Inputs: model. Outputs: ContextManagers. Exceptions/Side effects: None.
        is_ddp = isinstance(model, DistributedDataParallel)
        # We never to trigger a DDP sync in the next backward pass
        context = ContextManagers([model.no_sync()] if is_ddp else [])
        return context

    def backward(
        self, context: ContextManagers, state: PipelineTrainBatchState, grad_accumulator: Optional[GradientAccumulator]
    ):
        # [FUNCTION DOC]
        # Behavior: increments state.nb_backwards, logs, pops the last microbatch's activations requiring backward; if
        # none returns; else inside context backpropagates their sum, through grad_accumulator.backward when given.
        # Inputs: context (ContextManagers), state, grad_accumulator (or None).
        # Outputs: None. Exceptions: None explicitly raised.
        # Side effects: autograd backward (which triggers queued P2P grad sends/recvs and DDP hooks); logging.
        # KNOWN ISSUE: the debug log reports state.nb_forwards instead of state.nb_backwards as the backward micro batch id.
        # Increment the number of backwards
        state.nb_backwards += 1
        log_rank(
            f"Backward micro batch id: {state.nb_forwards}",
            logger=logger,
            level=logging.DEBUG,
        )
        # Go backward entirely
        activations = state.pop_last_activations_requiring_backward()
        if len(activations) == 0:
            return

        with context:
            if grad_accumulator is None:
                sum(activations).backward()
            else:
                grad_accumulator.backward(sum(activations))

        # TODO @nouamane: this fixes interleaved afab but makes 1f1b hang
        # with context:
        #     if grad_accumulator is None:
        #         for activation in reversed(activations): #TODO @nouamane: need to bwd only 2nd chunk
        #             activation.backward()
        #     else:
        #         for activation in reversed(activations):
        #             grad_accumulator.backward(activation)

    def _get_bwd_context(
        self,
        model: torch_nn.Module,
        nb_backwards: int,
        grad_accumulator: Optional[GradientAccumulator],
    ):
        # [FUNCTION DOC]
        # HELPER: backward context deciding gradient sync.
        # Behavior: asserts nb_microbatches is set. For DDP models: before the last microbatch adds grad_accumulator.no_sync
        # (if an accumulator exists); on the last microbatch (nb_backwards equals nb_microbatches - 1) adds
        # ddp_trigger_sync_in_bwd so DDP all-reduces. Non-DDP models get an empty context.
        # Inputs: model, nb_backwards (int, backwards done so far), grad_accumulator (or None).
        # Outputs: ContextManagers. Exceptions: AssertionError if nb_microbatches is None. Side effects: None until entered.
        assert (
            self.nb_microbatches is not None
        ), "You must call `train_batch_iter` first and set `self.nb_microbatches`"
        is_ddp = isinstance(model, DistributedDataParallel)
        context_list = []
        if is_ddp:
            if grad_accumulator is not None and nb_backwards < self.nb_microbatches - 1:
                context_list.append(grad_accumulator.no_sync())  # Prevents grad accumulator from syncing
            if nb_backwards == self.nb_microbatches - 1:
                # Triggers DDP to sync gradients in the next backward pass
                context_list.append(ddp_trigger_sync_in_bwd(model_ddp=model))
        context = ContextManagers(context_list)
        return context

    @torch.profiler.record_function("train_batch_iter")
    @abstractmethod
    def train_batch_iter(
        self,
        model: torch_nn.Module,
        pg: ProcessGroup,
        batch: Iterable[Dict[str, Union[torch.Tensor, TensorPointer]]],
        nb_microbatches: int,
        grad_accumulator: Optional[GradientAccumulator],
    ) -> Iterable[Dict[str, Union[torch.Tensor, TensorPointer]]]:
        # [FUNCTION DOC]
        # Abstract: run training forward and backward for all microbatches of a batch.
        # Inputs: model, pg (PP process group), batch (iterable of microbatch dicts), nb_microbatches (int),
        # grad_accumulator (or None). Outputs: list of per-microbatch output dicts (detached on the last stage).
        # Exceptions/Side effects: implementation-defined (P2P communication, autograd).
        """If model returns tensor, we use it as a loss to backpropagate. If model returns a dict, we assume that the key "loss" is the loss to backpropagate."""
        ...

    @torch.inference_mode()
    def validate_batch_iter(
        self,
        model: torch_nn.Module,
        batch: Iterable[Dict[str, Union[torch.Tensor, TensorPointer]]],
        nb_microbatches: int,
    ) -> Iterable[Dict[str, Union[torch.Tensor, TensorPointer]]]:
        # [FUNCTION DOC]
        # Behavior: under inference mode, sets nb_microbatches, attaches a fresh PipelineTrainBatchState, runs forward
        # for each microbatch and immediately executes queued activation sends, and collects detached outputs.
        # Inputs: model, batch (iterable of microbatch dicts), nb_microbatches (int).
        # Outputs: list of output dicts (losses normalized by nb_microbatches).
        # Exceptions: AssertionError from forward.
        # Side effects: P2P communication; mutates nb_microbatches.
        # KNOWN ISSUE: forward asserts the local loss requires grad, which can never hold under inference mode, so validation on the stage producing the loss fails unless the model returns TensorPointers there.
        # Assign a new state for the current batch
        state = PipelineTrainBatchState()  # TODO: do i need state?
        self.nb_microbatches = nb_microbatches

        outputs = []

        with attach_pipeline_state_to_model(model=model, pipeline_state=state):
            # All forward
            for micro_batch in batch:
                context = self._get_fwd_context(model=model)
                output = self.forward(context=context, state=state, micro_batch=micro_batch, model=model)
                # TODO @thomasw21: Somehow this needs to be done somewhere else to support interleaving. Somewhere right after a "stage"
                for _ in range(len(state.microbatches_activations_to_send)):
                    send_activation = state.microbatches_activations_to_send.popleft()
                    # Execute
                    send_activation()

                # We make `output` a dict
                if not isinstance(output, dict):
                    output = {"loss": output}

                # Store the loss for each microbatch
                if not isinstance(output["loss"], TensorPointer):
                    output = {k: v.detach() for k, v in output.items()}
                outputs.append(output)

        return outputs

    def __str__(self):
        # [FUNCTION DOC]
        # Returns the class name. Outputs: str.
        return self.__class__.__name__

    def __format__(self, format_spec):
        # [FUNCTION DOC]
        # Returns str of self regardless of format_spec. Inputs: format_spec. Outputs: str.
        return str(self)


class AllForwardAllBackwardPipelineEngine(PipelineEngine):
    # [CLASS DOC]
    # Purpose: GPipe-like schedule: forward every microbatch, then backward every microbatch.
    # Inheritance: PipelineEngine. String form "afab".
    def __init__(self):
        # [FUNCTION DOC]
        # Calls parent constructor. Inputs/Outputs: None.
        super().__init__()

    def __str__(self):
        # [FUNCTION DOC]
        # Returns "afab". Outputs: str.
        return "afab"

    def train_batch_iter(
        self,
        model: torch_nn.Module,
        pg: ProcessGroup,
        batch: Iterable[Dict[str, Union[torch.Tensor, TensorPointer]]],
        nb_microbatches: int,
        grad_accumulator: Optional[GradientAccumulator],
    ) -> Iterable[Dict[str, Union[torch.Tensor, TensorPointer]]]:
        # [FUNCTION DOC]
        # Behavior: creates a new state, sets nb_microbatches, attaches state. For each microbatch: forward, execute queued
        # activation sends, store output (detached if local loss). Then for each stored backward group: build backward
        # context, backward, execute queued gradient sends. Finally checks all state buffers are empty.
        # Inputs: model, pg (unused), batch, nb_microbatches, grad_accumulator.
        # Outputs: list of output dicts.
        # Exceptions: AssertionError from state checks or forward.
        # Side effects: P2P communication, autograd, DDP sync on last backward, gradient accumulation.
        # Assign a new state for the current batch
        state = PipelineTrainBatchState()
        self.nb_microbatches = nb_microbatches

        outputs = []

        with attach_pipeline_state_to_model(model=model, pipeline_state=state):
            # All forward
            for micro_batch in batch:
                context = self._get_fwd_context(model=model)
                output = self.forward(context=context, state=state, micro_batch=micro_batch, model=model)
                # TODO @thomasw21: Somehow this needs to be done somewhere else to support interleaving. Somewhere right after a "stage"
                for _ in range(len(state.microbatches_activations_to_send)):
                    send_activation = state.microbatches_activations_to_send.popleft()
                    # Execute
                    send_activation()

                # We make `output` a dict
                if not isinstance(output, dict):
                    output = {"loss": output}

                # Store the loss for each microbatch
                if not isinstance(output["loss"], TensorPointer):
                    output = {k: v.detach() for k, v in output.items()}
                outputs.append(output)

            # All backward
            for _ in range(len(state.microbatches_activations_requiring_backward)):
                context = self._get_bwd_context(
                    model=model,
                    nb_backwards=state.nb_backwards,
                    grad_accumulator=grad_accumulator,
                )
                self.backward(context=context, state=state, grad_accumulator=grad_accumulator)

                for _ in range(len(state.microbatches_grads_to_send)):
                    send_grads = state.microbatches_grads_to_send.popleft()
                    # Execute
                    send_grads()
            # Make sure that micro batches are all fully consumed
            state.check_buffers_empty()

            return outputs


class OneForwardOneBackwardPipelineEngine(PipelineEngine):
    # [CLASS DOC]
    # Purpose: 1F1B schedule (Narayanan et al. 2021): warm-up forwards, steady-state one forward plus one backward,
    # then cool-down backwards, bounding activation memory by PP size.
    # Inheritance: PipelineEngine. String form "1f1b".
    def __init__(self):
        # [FUNCTION DOC]
        # Calls parent constructor. Inputs/Outputs: None.
        super().__init__()

    def __str__(self):
        # [FUNCTION DOC]
        # Returns "1f1b". Outputs: str.
        return "1f1b"

    def train_batch_iter(
        self,
        model: torch_nn.Module,
        pg: ProcessGroup,
        batch: Iterable[Dict[str, Union[torch.Tensor, TensorPointer]]],
        nb_microbatches: int,
        grad_accumulator: Optional[GradientAccumulator],
    ) -> Iterable[Dict[str, Union[torch.Tensor, TensorPointer]]]:
        # [FUNCTION DOC]
        # Behavior: sets nb_microbatches and asserts it is at least PP size - 1. Warm-up: rank r runs PP size - r - 1
        # forwards, flushing activation sends after each. Steady state: for every remaining microbatch, forward (sends are
        # flushed lazily by the next block or backward) then one backward. Asserts the remaining backward count equals the
        # warm-up count and no activation sends/recvs remain. Cool-down: flushes pending gradient sends, then for each
        # remaining microbatch runs backward and flushes gradient sends. Checks buffers are empty.
        # Inputs: model, pg (PP process group), batch (iterable), nb_microbatches (int), grad_accumulator.
        # Outputs: list of output dicts.
        # Exceptions: AssertionError on too few microbatches or inconsistent state; StopIteration if batch is shorter than the warm-up.
        # Side effects: P2P communication, autograd, DDP/accumulator sync on the last backward.
        # KNOWN ISSUE: in the warm-up loop activation sends are flushed twice in a row; the second loop is always a no-op.
        """Check https://arxiv.org/abs/2104.04473 for diagrams for the pipeline engine"""
        self.nb_microbatches = nb_microbatches
        assert (
            self.nb_microbatches >= pg.size() - 1
        ), f"Number of microbatches ({self.nb_microbatches}) must be at least PP_SIZE-1={pg.size() - 1} when using the OneForwardOneBackwardPipelineEngine"

        state = PipelineTrainBatchState()

        outputs = []
        batch = iter(batch)

        current_pp_rank = dist.get_rank(pg)

        with attach_pipeline_state_to_model(model=model, pipeline_state=state):
            # Init
            for _ in range(pg.size() - current_pp_rank - 1):
                micro_batch = next(batch)
                context = self._get_fwd_context(model=model)
                output = self.forward(context=context, state=state, micro_batch=micro_batch, model=model)

                # TODO @thomasw21: Somehow this needs to be done somewhere else to support interleaving. Somewhere right after a "stage"
                for _ in range(len(state.microbatches_activations_to_send)):
                    send_activation = state.microbatches_activations_to_send.popleft()
                    # Execute
                    send_activation()

                # We make `output` a dict
                if not isinstance(output, dict):
                    output = {"loss": output}

                # Send tensors
                # TODO @thomasw21: Somehow this needs to be done somewhere else to support interleaving. Somewhere right after a "stage"
                for _ in range(len(state.microbatches_activations_to_send)):
                    send_activation = state.microbatches_activations_to_send.popleft()
                    # Execute
                    send_activation()

                # Store the loss for each microbatch
                if not isinstance(output["loss"], TensorPointer):
                    output = {k: v.detach() for k, v in output.items()}
                outputs.append(output)

            for micro_batch in batch:
                context = self._get_fwd_context(model=model)
                # with nanotron_timer("forward", timer_type="cuda"):
                output = self.forward(context=context, state=state, micro_batch=micro_batch, model=model)

                # We make `output` a dict
                if not isinstance(output, dict):
                    output = {"loss": output}

                # Store the loss for each microbatch
                if not isinstance(output["loss"], TensorPointer):
                    output = {k: v.detach() for k, v in output.items()}
                outputs.append(output)

                # One backward
                context = self._get_bwd_context(
                    model=model,
                    nb_backwards=state.nb_backwards,
                    grad_accumulator=grad_accumulator,
                )
                # with nanotron_timer("backward", timer_type="cuda"):
                self.backward(context=context, state=state, grad_accumulator=grad_accumulator)

            # Check figure in paper: The remain blocks are all backward and there is only `pg.size() - current_pp_rank - 1` blocks left
            assert len(state.microbatches_activations_requiring_backward) == pg.size() - current_pp_rank - 1
            # No more activation to send/recv
            assert (
                len(state.microbatches_activations_to_send) == 0
            ), f"There are activations left for me to send still: {len(state.microbatches_activations_to_send)}"
            assert (
                len(state.microbatches_activations_to_recv) == 0
            ), f"There are activations left for me to recv still: {len(state.microbatches_activations_to_recv)}"

            # Close: compute backward for the rest
            # TODO @thomasw21: Somehow this needs to be done somewhere else to support interleaving. Somewhere right after a "stage"
            for _ in range(len(state.microbatches_grads_to_send)):
                send_grads = state.microbatches_grads_to_send.popleft()
                # Execute
                send_grads()
            for _ in range(len(state.microbatches_activations_requiring_backward)):
                context = self._get_bwd_context(
                    model=model,
                    nb_backwards=state.nb_backwards,
                    grad_accumulator=grad_accumulator,
                )
                self.backward(context=context, state=state, grad_accumulator=grad_accumulator)

                # TODO @thomasw21: Somehow this needs to be done somewhere else to support interleaving. Somewhere right after a "stage"
                for _ in range(len(state.microbatches_grads_to_send)):
                    send_grads = state.microbatches_grads_to_send.popleft()
                    # Execute
                    send_grads()

            # Make sure that micro batches are all fully consumed
            state.check_buffers_empty()

        return outputs
