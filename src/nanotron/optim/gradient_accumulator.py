# [MODULE DOC]
# Purpose: gradient accumulators that keep fp32 master weights and fp32 gradient buffers for half-precision
# training, plus a DDP communication hook that accumulates into those buffers.
# Main contents: GradientAccumulator (abstract), FP32GradientAccumulator, FP32GradBucketManager (hook state),
# get_fp32_accum_hook.
# How it fits: used by the trainer and OptimizerFromGradientAccumulator; ZeRO-1 uses param offsets to shard updates.
# Import-time side effects: creates a module logger.
import dataclasses
from abc import ABC, abstractmethod
from collections import OrderedDict
from contextlib import contextmanager
from typing import Callable, Dict, Iterator, Optional, Tuple

import torch
from torch.distributed import GradBucket

import nanotron.distributed as dist
from nanotron import logging
from nanotron.parallel.parameters import NanotronParameter
from nanotron.utils import get_untyped_storage, tensor_from_untyped_storage

logger = logging.get_logger(__name__)


class GradientAccumulator(ABC):
    # [CLASS DOC]
    # Purpose: abstract interface for gradient accumulation strategies.
    # Key attributes: fp32_grads_allreduce_handle (pending async all-reduce work or None).
    # Abstract methods: __init__, backward, step, sync_gradients_across_dp, zero_grad, get_parameter_for_optimizer,
    # get_grad_buffer, state_dict, load_state_dict.
    fp32_grads_allreduce_handle: Optional[torch.futures.Future]

    @abstractmethod
    def __init__(self, named_parameters: Iterator[Tuple[str, NanotronParameter]]):
        # [FUNCTION DOC]
        # Abstract constructor. Inputs: named_parameters (iterator of name, parameter). Outputs: None.
        ...

    @abstractmethod
    def backward(self, loss: torch.Tensor):
        # [FUNCTION DOC]
        # Abstract: run backward on loss and accumulate. Inputs: loss (tensor). Outputs: implementation-defined.
        ...

    @abstractmethod
    def step(self):
        # [FUNCTION DOC]
        # Abstract: push updated master weights to model parameters. Outputs: None.
        ...

    @abstractmethod
    def sync_gradients_across_dp(self, dp_pg: dist.ProcessGroup, reduce_op: dist.ReduceOp, reduce_scatter: bool):
        # [FUNCTION DOC]
        # Abstract: reduce accumulated gradients across data-parallel ranks. Inputs: dp_pg, reduce_op, reduce_scatter
        # (bool). Outputs: None. Side effects: collective communication.
        ...

    @abstractmethod
    def zero_grad(self):
        # [FUNCTION DOC]
        # Abstract: reset accumulated gradients. Outputs: None.
        ...

    @abstractmethod
    def get_parameter_for_optimizer(self, name: str) -> NanotronParameter:
        # [FUNCTION DOC]
        # Abstract: parameter the optimizer should update for name. Inputs: name (str). Outputs: parameter tensor.
        ...

    @abstractmethod
    def get_grad_buffer(self, name: str) -> torch.Tensor:
        # [FUNCTION DOC]
        # Abstract: accumulated gradient buffer for name. Inputs: name (str). Outputs: tensor.
        ...

    @abstractmethod
    def state_dict(self) -> Dict[str, torch.Tensor]:
        # [FUNCTION DOC]
        # Abstract: accumulator state. Outputs: dict of name to tensor.
        ...

    @abstractmethod
    def load_state_dict(self, state_dict: torch.Tensor):
        # [FUNCTION DOC]
        # Abstract: restore accumulator state. Inputs: state_dict. Outputs: None.
        # KNOWN ISSUE: state_dict parameter is annotated as a tensor while implementations take a dictionary.
        ...


class FP32GradientAccumulator(GradientAccumulator):
    # [CLASS DOC]
    # Purpose: keeps fp32 copies of the parameters the optimizer updates and fp32 gradient buffers for all
    # parameters (in one contiguous CUDA buffer), accumulating half-precision grads into them.
    # Key state: fp32_grad_buffers (ordered dict name to {"half": param, "fp32_grad": fp32 view}),
    # _contiguous_fp32_grad_buffer (flat fp32 buffer backing all grad views), parameters (dict name to {"fp32": master
    # weight view into one flat buffer, "half": model param}), _is_accumulation_sync_step (when True the DDP hook
    # does the accumulation), fp32_grads_allreduce_handle, optionally param_name_to_offsets (ZeRO slice per param).
    # Invariant: fp32 weight views share stride with the half params.
    def __init__(
        self,
        named_parameters: Iterator[Tuple[str, NanotronParameter]],
        grad_buckets_named_params: Optional[Iterator[Tuple[str, NanotronParameter]]] = None,
    ):
        # [FUNCTION DOC]
        # Behavior: if grad_buckets_named_params is None, both sets are the same list. Builds grad buffers for
        # grad_buckets_named_params. Allocates one flat fp32 CUDA buffer for all named_parameters requiring grad, creates
        # views per parameter, copies half weights into them under inference mode and marks them requires_grad. Sets sync
        # flag False and the handle to None.
        # Inputs: named_parameters (params updated by this rank's optimizer), grad_buckets_named_params (params to keep
        # grad buffers for; for ZeRO-1 all model params; default None).
        # Outputs: None.
        # Exceptions: AssertionError on duplicate names, stride mismatch, or (from build_grad_buffers) fp32 params not
        # named router.weight or non-contiguous params.
        # Side effects: allocates CUDA memory.
        # KNOWN ISSUE: setting requires_grad on a tensor created inside the flat buffer within inference mode yields inference tensors; works in practice only because they are used as leaves later.
        """Create a gradient accumulator that will accumulate gradients in fp32.

        Args:
            named_parameters: The parameters that will be updated by the optimizer. In case of Zero 1, this is the parameters that will be updated in this DP rank.
            grad_buckets_named_params: The parameters to accumulate gradients for. If None it defaults to `named_parameters`. In case of Zero 1, this should be all the parameters in the model.

        Note: We use `grad_buckets_named_params` to keep grad buffers for all parameters even when Zero 1 is used. This is because we need to accumulate gradients for all parameters without having to reduce in every accumulation step.
        Note: We make a fp32 copy of parameters during initialization. Therefore parameters need to be initialized or loaded from a checkpoint before constructing this gradient accumulator
        """
        if grad_buckets_named_params is None:
            named_parameters = list(named_parameters)
            grad_buckets_named_params = named_parameters

        # Initialize grad bucket
        self.fp32_grad_buffers, self._contiguous_fp32_grad_buffer = self.build_grad_buffers(
            named_parameters=grad_buckets_named_params
        )

        # Assign big buffer for weights + grad in fp32
        segment_index = {}
        length = 0
        for name, param in named_parameters:
            if not param.requires_grad:
                continue

            start = length
            end_weight = start + param.numel()
            assert name not in segment_index
            segment_index[name] = (start, end_weight, param)
            length = end_weight

        big_flat_buffer = torch.empty(length, dtype=torch.float, device="cuda")
        self.parameters = {
            name: {
                "fp32": big_flat_buffer[start_weight:end_weight].view_as(param),
                "half": param,
            }
            for name, (start_weight, end_weight, param) in segment_index.items()
        }

        with torch.inference_mode():
            for _, elt in self.parameters.items():
                fp32_param = elt["fp32"]
                half_param = elt["half"]

                # Check that fp32 weights have the same memory representation as half precision weights
                assert fp32_param.stride() == half_param.stride()

                # Copy weights from half precision to full precision
                fp32_param.copy_(half_param)

                # Set requires_grad=True
                fp32_param.requires_grad = True

        self._is_accumulation_sync_step = False
        # We need the last allreduce handle to make sure it finishes before the optimizer step
        self.fp32_grads_allreduce_handle: Optional[torch.futures.Future] = None

    def assign_param_offsets(self, param_name_to_offsets: Dict[str, Dict[int, Tuple[int, int]]], dp_rank: int):
        # [FUNCTION DOC]
        # Behavior: stores, for every parameter that has an entry for dp_rank, the (start, end) flat slice this rank owns.
        # Inputs: param_name_to_offsets (name to dict dp_rank to (start, end)), dp_rank (int). Outputs: None.
        # Exceptions: None. Side effects: sets self.param_name_to_offsets (changes behavior of _accumulate_grad and sync).
        """To use only when you use with ZeRODistributedOptimizer"""
        self.param_name_to_offsets = {
            name: elt[dp_rank] for name, elt in param_name_to_offsets.items() if dp_rank in elt
        }

    @torch.profiler.record_function("FP32GradientAccumulator.sync_gradients_across_dp")
    def sync_gradients_across_dp(self, dp_pg: dist.ProcessGroup, reduce_op: dist.ReduceOp, reduce_scatter: bool):
        # [FUNCTION DOC]
        # Behavior: returns if dp_pg has size 1. With reduce_scatter: for params with offsets (sorted by name) does a
        # coalesced reduce-scatter where each param's flat grad is split into dp size equal chunks and this rank receives
        # its slice in place. Otherwise all-reduces the whole contiguous fp32 buffer with reduce_op.
        # Inputs: dp_pg (process group), reduce_op (ReduceOp), reduce_scatter (bool).
        # Outputs: None.
        # Exceptions: AssertionError if reduce_scatter without assign_param_offsets.
        # Side effects: collectives on dp_pg; mutates grad buffers.
        # KNOWN ISSUE: reduce_op is ignored in the reduce_scatter branch (default SUM is used).
        # KNOWN ISSUE: reduce_scatter path only covers params that have an offset on this rank, so ranks may pass different lists to the collective; also assumes numel divisible by dp size and equal to ZeRO offsets.
        if dp_pg.size() == 1:
            # They are already synced
            return

        if reduce_scatter:
            # Usually you need to run `all_reduce` in order for all gradients to be synced.
            # However when the optimizer state are sharded, you really just need to scatter to ranks that are going to run the optimizer state.
            # Effectively you replace a `all_reduce` with a `reduce_scatter` which should save an `all_gather` when using RING algorithm.
            assert hasattr(self, "param_name_to_offsets")
            named_offsets = sorted(self.param_name_to_offsets.items(), key=lambda x: x[0])
            flat_grad_buffers = [self.fp32_grad_buffers[name]["fp32_grad"].view(-1) for name, _ in named_offsets]
            dist.reduce_scatter_coalesced(
                output_tensor_list=[
                    flat_grad_buffer[start_offset:end_offset]
                    for (_, (start_offset, end_offset)), flat_grad_buffer in zip(named_offsets, flat_grad_buffers)
                ],
                input_tensor_lists=[
                    torch.split(
                        flat_grad_buffer,
                        split_size_or_sections=len(self.fp32_grad_buffers[name]["fp32_grad"].view(-1)) // dp_pg.size(),
                    )
                    for (name, _), flat_grad_buffer in zip(named_offsets, flat_grad_buffers)
                ],
                group=dp_pg,
            )
        else:
            dist.all_reduce(self._contiguous_fp32_grad_buffer, op=reduce_op, group=dp_pg)

    @staticmethod
    def build_grad_buffers(
        named_parameters: Iterator[Tuple[str, NanotronParameter]],
    ) -> Tuple[Dict[str, Dict], torch.Tensor]:
        # [FUNCTION DOC]
        # Behavior: filters params requiring grad, allocates a zeroed fp32 CUDA buffer of total numel, and carves out
        # per-parameter views by slicing its untyped storage in byte offsets, in input order.
        # Inputs: named_parameters (iterable of name, parameter).
        # Outputs: tuple (ordered dict name to {"half", "fp32_grad"}, flat fp32 buffer).
        # Exceptions: AssertionError if a param is fp32 (unless name contains router.weight) or non-contiguous.
        # Side effects: allocates CUDA memory.
        """Builds grad buffers for all model's parameters, independently of ZeRO sharding

        Args:
            named_parameters: Parameters to build buckets for. In case of Zero1, this should be all parameters.

        Note:
            In ZeRO-1, we need to accumulate grads for all parameters, because we need to allreduce all parameters' grads across DP at each sync step.
        """
        named_parameters = [(name, param) for name, param in named_parameters if param.requires_grad]

        needed_buffer_size = sum(param.numel() for _, param in named_parameters)
        # important to have grads zeroed initially (see `self._accumulate_grad`)
        contiguous_buffer_f32_gradients = torch.zeros(needed_buffer_size, dtype=torch.float, device="cuda")
        untyped_storage = get_untyped_storage(contiguous_buffer_f32_gradients)
        element_size = contiguous_buffer_f32_gradients.element_size()

        # NOTE: Although `bias` can only exist on TP=0. It shouldn't be a problem here, because we only sync across DP
        fp32_grad_buffers = OrderedDict()  # keeps order of insertion
        offset = 0
        for name, param in named_parameters:
            if not param.requires_grad:
                continue

            # MoE router weights are initialized in float32
            assert param.dtype != torch.float or "router.weight" in name, f"Expected {name} not to be float"
            assert param.is_contiguous(), f"Expected {name} to be contiguous"

            next_offset = offset + param.numel() * element_size

            fp32_grad_buffer = tensor_from_untyped_storage(
                untyped_storage=untyped_storage[offset:next_offset], dtype=torch.float
            )

            fp32_grad_buffers[name] = {
                "half": param,
                # We create sliced tensors by also slicing storage.
                # We need to specify "cuda" in order to share the same data storage, otherwise it build the tensor in "cpu" and copies over the data
                "fp32_grad": fp32_grad_buffer.view_as(param),
            }

            offset = next_offset

        return fp32_grad_buffers, contiguous_buffer_f32_gradients

    def backward(self, loss: torch.Tensor):
        # [FUNCTION DOC]
        # Behavior: calls loss.backward(), then accumulates every buffered param's half grad into fp32.
        # Inputs: loss (scalar tensor). Outputs: result of loss.backward() (None).
        # Exceptions: AssertionError from _accumulate_grad if a param has no grad.
        # Side effects: autograd, gradient mutation.
        result = loss.backward()

        for name, elt in self.fp32_grad_buffers.items():
            self._accumulate_grad(name=name, half_param=elt["half"])

        return result

    @torch.profiler.record_function("FP32GradientAccumulator._accumulate_grad")
    def _accumulate_grad(self, name: str, half_param: NanotronParameter) -> None:
        # [FUNCTION DOC]
        # HELPER: moves one parameter's half grad into its fp32 buffer.
        # Behavior: asserts grad exists; adds it into the fp32 buffer unless in sync step (where the DDP hook already did);
        # clears half grad; if the param has an fp32 master copy, sets its .grad to the fp32 buffer, or to this rank's
        # slice when ZeRO offsets exist (skipping params with no slice).
        # Inputs: name (str), half_param (parameter). Outputs: None.
        # Exceptions: AssertionError when half_param.grad is None.
        # Side effects: mutates buffers and .grad fields.
        """Accumulate grad in fp32 and set the fp32 grad to the fp32 grad buffer, so that optimizer can update fp32 weights afterwards"""
        assert half_param.grad is not None, f"Expected param {name} to have gradient."
        fp32_grad = self.get_grad_buffer(name=name)

        if self._is_accumulation_sync_step is False:
            # WARNING: We assume fp32_grad_bucket is already zeroed
            fp32_grad.add_(half_param.grad)
            # In case _is_accumulation_sync_step = True: no need to add half gradients, because it's done in the allreduce hook

        # TODO @thomasw21: Is it better to set to zero instead?
        half_param.grad = None

        # In the case an optimizer decides to set it to None, we need to re-assign previous buffer
        if name in self.parameters:
            fp32_param = self.parameters[name]["fp32"]
            if hasattr(self, "param_name_to_offsets"):
                if name not in self.param_name_to_offsets:
                    # When `name` isn't in `param_name_to_offsets` it means the slice is empty.
                    return
                start_offset, end_offset = self.param_name_to_offsets[name]
                grad = fp32_grad.view(-1)[start_offset:end_offset]
            else:
                grad = fp32_grad
            fp32_param.grad = grad

    @contextmanager
    def no_sync(self):
        # [FUNCTION DOC]
        # Behavior: context manager setting _is_accumulation_sync_step False for the duration, restoring previous value after.
        # Inputs: none. Outputs: generator yielding once.
        # Exceptions: None. Side effects: toggles flag.
        # KNOWN ISSUE: docstring says it switches the default to True, but the code only temporarily sets False; the True default comes from FP32GradBucketManager.
        """A context manager to disable gradient synchronizations across
        data-parallel ranks.

        Note: if we use `no_sync` once, that means we're in DDP mode, and we switch the default of self._is_accumulation_sync_step to True.
        """
        old_is_accumulation_sync_step = self._is_accumulation_sync_step
        self._is_accumulation_sync_step = False
        try:
            yield
        finally:
            self._is_accumulation_sync_step = old_is_accumulation_sync_step

    @torch.profiler.record_function("FP32GradientAccumulator.step")
    @torch.inference_mode()
    def step(self):
        # [FUNCTION DOC]
        # Behavior: under inference mode, copies each fp32 master weight into its half-precision model parameter.
        # Inputs: none. Outputs: None. Exceptions: None. Side effects: overwrites model weights.
        """Updates fp32 weights from fp32 grads.
        In case where OptimizerFromGradientAccumulator and gradient_accumulator_builder are using different parameters (e.g ZeRO).
        We need to update only the parameters that were updated by the optimizer.
        """
        for name in self.parameters.keys():
            fp32_param = self.parameters[name]["fp32"]
            half_param = self.parameters[name]["half"]
            # TODO @nouamane: should we use a fused kernel to copy?
            # Copy weights from full precision to half precision
            half_param.copy_(fp32_param)

    @torch.profiler.record_function("FP32GradientAccumulator.zero_grad")
    def zero_grad(self):
        # [FUNCTION DOC]
        # Behavior: clears half grads of buffered params and zeros the contiguous fp32 grad buffer.
        # Inputs: none. Outputs: None. Exceptions: None. Side effects: mutates grads and buffer.
        # Full precision gradients are reset to zero/none after the underlying `optimiser.step`, so no need to reset.
        for elt in self.fp32_grad_buffers.values():
            half_param = elt["half"]

            if half_param.grad is None:
                continue

            half_param.grad = None

        # in case where self.parameters and self.fp32_grad_buffers are not the same (e.g we want to accumulate all DPs grads, and only sync at sync step)
        self._contiguous_fp32_grad_buffer.zero_()

    def get_parameter_for_optimizer(self, name: str) -> NanotronParameter:
        # [FUNCTION DOC]
        # Returns the fp32 master weight for name. Inputs: name. Outputs: tensor. Exceptions: KeyError if unknown. Side effects: None.
        return self.parameters[name]["fp32"]

    def get_grad_buffer(self, name: str) -> torch.Tensor:
        # [FUNCTION DOC]
        # Returns the fp32 grad buffer view for name. Inputs: name. Outputs: tensor. Exceptions: KeyError if unknown. Side effects: None.
        """Returns the gradient of the parameter from the appropriate grad bucket."""
        return self.fp32_grad_buffers[name]["fp32_grad"]

    def state_dict(self) -> Dict[str, torch.Tensor]:
        # [FUNCTION DOC]
        # Returns dict of name to fp32 master weight (live tensors, not copies). Inputs: none. Exceptions: None. Side effects: None.
        # We consider `fp32` parameters as a state of the gradient accumulator
        return {name: elt["fp32"] for name, elt in self.parameters.items()}

    def load_state_dict(self, state_dict: Dict[str, torch.Tensor]):
        # [FUNCTION DOC]
        # Behavior: asserts the key sets match, then copies each tensor into the fp32 master weights in inference mode.
        # Inputs: state_dict (name to tensor). Outputs: None.
        # Exceptions: AssertionError on key mismatch. Side effects: overwrites master weights.
        assert set(state_dict.keys()) == set(self.parameters.keys())

        with torch.inference_mode():
            for name, elt in self.parameters.items():
                elt["fp32"].copy_(state_dict[name])


@dataclasses.dataclass
class FP32GradBucketManager:
    # [CLASS DOC]
    # Purpose: dataclass state object passed to the DDP fp32 accumulation hook.
    # Attributes: dp_cp_pg (data x context parallel group to reduce over), accumulator (FP32GradientAccumulator),
    # param_id_to_name (id of parameter to name).
    # Invariant: creating it switches the accumulator into sync-step mode.
    # KNOWN ISSUE: docstring lists bucket_id_to_fp32_grad_buckets_and_dependencies and param_id_to_bucket_id attributes that do not exist.
    """Manages the fp32 gradient buckets.

    Attributes:
        dp_cp_pg: The process group to allreduce gradients across.
        accumulator: The gradient accumulator which keeps the gradient buffers.
        bucket_id_to_fp32_grad_buckets_and_dependencies: A dictionary mapping bucket ids to:
            - fp32 grad bucket (torch.Tensor)
            - set of param ids that are in the bucket -> used to know when to delete the buffer
        param_id_to_bucket_id: A dictionary mapping param ids to bucket ids."""

    dp_cp_pg: dist.ProcessGroup
    accumulator: FP32GradientAccumulator
    param_id_to_name: Dict[int, str]

    def __post_init__(self):
        # [FUNCTION DOC]
        # Sets accumulator._is_accumulation_sync_step True so the hook does accumulation. Inputs/Outputs: None.
        # Exceptions: None. Side effects: mutates the accumulator.
        self.accumulator._is_accumulation_sync_step = True


def get_fp32_accum_hook(
    reduce_scatter: bool,
    reduce_op: dist.ReduceOp = dist.ReduceOp.AVG,
) -> Callable:
    # [FUNCTION DOC]
    # Behavior: returns a DDP communication hook closure configured with reduce_scatter and reduce_op.
    # Inputs: reduce_scatter (bool), reduce_op (ReduceOp, default AVG).
    # Outputs: callable hook (state, bucket) returning a future.
    # Exceptions: None at creation. Side effects: None.
    """Returns a DDP communication hook that performs gradient accumulation in fp32.

    Args:
        reduce_op: The reduction operation to perform.
    """
    # s = torch.cuda.Stream()

    def fp32_accum_hook(state: FP32GradBucketManager, bucket: GradBucket) -> torch.futures.Future[torch.Tensor]:
        # [FUNCTION DOC]
        # Behavior: called by DDP per ready bucket during backward. Adds each bucket gradient into the matching fp32
        # buffer. If group size is 1 returns a completed future with the bucket buffer. With reduce_scatter raises.
        # Otherwise launches async coalesced all-reduce of the fp32 buffers with reduce_op over dp_cp_pg, storing the
        # handle on the accumulator (not awaited). Returns a completed future of the unchanged half bucket buffer.
        # Inputs: state (FP32GradBucketManager), bucket (GradBucket).
        # Outputs: completed future of tensor.
        # Exceptions: NotImplementedError when reduce_scatter is True.
        # Side effects: mutates fp32 buffers; async collective; overwrites accumulator.fp32_grads_allreduce_handle (only the last bucket's handle is kept).
        # KNOWN ISSUE: code after the NotImplementedError is unreachable and references an undefined name dp_pg.
        # KNOWN ISSUE: only the last bucket's all-reduce handle is retained, so waiting on it may not guarantee earlier buckets finished (relies on in-order execution on the stream).
        # nonlocal s
        # DDP groups grads in GradBuckets. This hook is called throughout the bwd pass, once each bucket is ready to overlap communication with computation.
        # See https://pytorch.org/docs/stable/ddp_comm_hooks.html#what-does-a-communication-hook-operate-on for more details.
        dp_cp_pg = state.dp_cp_pg
        accumulator = state.accumulator
        param_id_to_name = state.param_id_to_name

        # Add new incoming gradient
        # with torch.cuda.stream(s):
        for param, grad in zip(bucket.parameters(), bucket.gradients()):
            name = param_id_to_name[id(param)]
            fp32_grad_buffer = accumulator.get_grad_buffer(name)
            fp32_grad_buffer.add_(grad.view_as(fp32_grad_buffer))

        # sync across dp
        if dp_cp_pg.size() == 1:
            fut = torch.futures.Future()
            fut.set_result(bucket.buffer())
            return fut

        if reduce_scatter:
            raise NotImplementedError("Not implemented")
            assert hasattr(accumulator, "param_name_to_offsets")
            grad_buffer_tensor_list = [
                accumulator.get_grad_buffer(param_id_to_name[id(param)]).view(-1) for param in bucket.parameters()
            ]
            device = grad_buffer_tensor_list[0].device
            dtype = grad_buffer_tensor_list[0].dtype
            output_tensor_list = [
                grad_buffer[slice(*accumulator.param_name_to_offsets[param_id_to_name[id(param)]])]
                if param_id_to_name[id(param)] in accumulator.param_name_to_offsets
                else torch.empty(0, dtype=dtype, device=device)
                for grad_buffer, param in zip(grad_buffer_tensor_list, bucket.parameters())
            ]
            input_tensor_lists = [
                torch.split(grad_buffer, split_size_or_sections=len(grad_buffer) // dp_pg.size())
                for grad_buffer in grad_buffer_tensor_list
            ]
            dist.reduce_scatter_coalesced(
                output_tensor_list=output_tensor_list,
                input_tensor_lists=input_tensor_lists,
                op=reduce_op,
                group=dp_cp_pg,
                async_op=True,
            )
        else:
            grad_buffer_tensor_list = [
                accumulator.get_grad_buffer(param_id_to_name[id(param)]).view(-1) for param in bucket.parameters()
            ]
            accumulator.fp32_grads_allreduce_handle = dist.all_reduce_coalesced(
                grad_buffer_tensor_list, group=dp_cp_pg, async_op=True, op=reduce_op
            )
            # we shouldn't wait for this future for the rest of the backward

        # with torch.cuda.stream(s):
        fut: torch.futures.Future[torch.Tensor] = torch.futures.Future()
        half_grad_bucket = bucket.buffer()
        fut.set_result(half_grad_bucket)
        return fut  # We don't care about the new half grad values, so we return the old ones

    return fp32_accum_hook
