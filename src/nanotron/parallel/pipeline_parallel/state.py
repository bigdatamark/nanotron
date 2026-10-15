# [MODULE DOC]
# Purpose: pipeline batch state objects that queue P2P communications (activations forward, gradients backward)
# and buffer received tensors, so pipeline engines can control when communication happens.
# Main contents: callable dataclasses SendActivation, RecvActivation, SendGrad, RecvGrad; abstract
# PipelineBatchState; PipelineTrainBatchState; PipelineEvalBatchState.
# Import-time side effects: creates a module logger.
import collections
import dataclasses
from abc import ABC, abstractmethod
from typing import List

import torch
from nanotron import distributed as dist
from nanotron import logging
from nanotron.logging import log_rank
from nanotron.parallel.pipeline_parallel.p2p import P2P

logger = logging.get_logger(__name__)


@dataclasses.dataclass
class SendActivation:
    # [CLASS DOC]
    # Purpose: deferred blocking send of one activation. Attributes: activation (tensor), to_rank (PP group rank), p2p.
    activation: torch.Tensor
    to_rank: int
    p2p: P2P

    def __call__(self):
        # [FUNCTION DOC]
        # Sends the activation with p2p.send_tensors. Outputs: None. Side effects: blocking P2P send.
        self.p2p.send_tensors([self.activation], to_rank=self.to_rank)


@dataclasses.dataclass
class RecvActivation:
    # [CLASS DOC]
    # Purpose: deferred blocking receive of one activation. Attributes: from_rank, p2p.
    from_rank: int
    p2p: P2P

    def __call__(self) -> torch.Tensor:
        # [FUNCTION DOC]
        # Receives one tensor from from_rank. Outputs: tensor. Side effects: blocking P2P receive.
        return self.p2p.recv_tensors(num_tensors=1, from_rank=self.from_rank)[0]


@dataclasses.dataclass
class SendGrad:
    # [CLASS DOC]
    # Purpose: deferred blocking send of one gradient. Attributes: grad, to_rank, p2p.
    grad: torch.Tensor
    to_rank: int
    p2p: P2P

    def __call__(self):
        # [FUNCTION DOC]
        # Sends the gradient. Outputs: None. Side effects: blocking P2P send.
        self.p2p.send_tensors([self.grad], to_rank=self.to_rank)


@dataclasses.dataclass
class RecvGrad:
    # [CLASS DOC]
    # Purpose: deferred blocking receive of one gradient. Attributes: from_rank, p2p.
    from_rank: int
    p2p: P2P

    def __call__(self) -> torch.Tensor:
        # [FUNCTION DOC]
        # Receives one tensor from from_rank. Outputs: tensor. Side effects: blocking P2P receive.
        return self.p2p.recv_tensors(num_tensors=1, from_rank=self.from_rank)[0]


class PipelineBatchState(ABC):
    # [CLASS DOC]
    # Purpose: abstract interface for queuing pipeline communication. Class attribute activations_buffer (deque of
    # received activations).
    # KNOWN ISSUE: activations_buffer is a class-level deque shared by every instance and subclass that does not override it.
    activations_buffer = collections.deque()

    @abstractmethod
    def register_activation_requiring_backward(self, activation: torch.Tensor):
        # [FUNCTION DOC]
        # Abstract: record a tensor to backpropagate from for the current microbatch. Inputs: activation.
        ...

    @abstractmethod
    def register_send_activation(self, activation: torch.Tensor, to_rank: int, p2p: P2P):
        # [FUNCTION DOC]
        # Abstract: queue an activation send. Inputs: activation, to_rank, p2p.
        ...

    @abstractmethod
    def register_recv_activation(self, from_rank: int, p2p: P2P):
        # [FUNCTION DOC]
        # Abstract: queue an activation receive. Inputs: from_rank, p2p.
        ...

    @abstractmethod
    def register_send_grad(self, grad: torch.Tensor, to_rank: int, p2p: P2P):
        # [FUNCTION DOC]
        # Abstract: queue a gradient send. Inputs: grad, to_rank, p2p.
        ...

    @abstractmethod
    def register_recv_grad(self, from_rank: int, p2p: P2P):
        # [FUNCTION DOC]
        # Abstract: queue a gradient receive. Inputs: from_rank, p2p.
        ...

    @abstractmethod
    def run_communication(self, send_only_activation: bool = False):
        # [FUNCTION DOC]
        # Abstract: execute queued communication. Inputs: send_only_activation (bool, default False).
        ...

    @abstractmethod
    def new_micro_batch_forward(self):
        # [FUNCTION DOC]
        # Abstract: start bookkeeping for a new microbatch.
        ...

    @abstractmethod
    def pop_last_activations_requiring_backward(self) -> List[torch.Tensor]:
        # [FUNCTION DOC]
        # Abstract: pop the activations of one microbatch. Outputs: list of tensors.
        ...


@dataclasses.dataclass
class PipelineTrainBatchState(PipelineBatchState):
    # [CLASS DOC]
    # Purpose: training state with queues for activation/gradient sends and receives, buffers for received
    # gradients, per-microbatch activations requiring backward (deque of deques, FIFO), and counters nb_forwards and
    # nb_backwards.
    # Inheritance: PipelineBatchState (dataclass decorator, but attributes are unannotated class attributes).
    # KNOWN ISSUE: all deques are class attributes (not dataclass fields), so every PipelineTrainBatchState instance shares the same queues; a "fresh" state per batch is not actually fresh if anything was left over.
    microbatches_activations_to_send = collections.deque()
    microbatches_activations_to_recv = collections.deque()
    microbatches_grads_to_send = collections.deque()
    microbatches_grads_to_recv = collections.deque()
    grads_buffer = collections.deque()

    # List of list, first index represent micro_batch_id, second index represent activations that needs to be popped
    microbatches_activations_requiring_backward = collections.deque()

    # Reinitialise counter
    nb_backwards = 0
    nb_forwards = 0

    def register_activation_requiring_backward(self, activation: torch.Tensor):
        # [FUNCTION DOC]
        # Appends activation to the latest microbatch's list. Inputs: activation. Outputs: None.
        # Exceptions: IndexError if no microbatch started. Side effects: mutates queues.
        # Register the activation to last microbatch
        self.microbatches_activations_requiring_backward[-1].append(activation)

    def register_send_activation(self, activation: torch.Tensor, to_rank: int, p2p: P2P):
        # [FUNCTION DOC]
        # Queues a SendActivation. Inputs: activation, to_rank, p2p. Outputs: None. Side effects: mutates queue.
        # TODO @thomasw21: We assume that each rank has a single contiguous list of blocks. This also means that we only send activations from higher ranks
        self.microbatches_activations_to_send.append(SendActivation(activation=activation, to_rank=to_rank, p2p=p2p))

    def register_recv_activation(self, from_rank: int, p2p: P2P):
        # [FUNCTION DOC]
        # Queues a RecvActivation. Inputs: from_rank, p2p. Outputs: None. Side effects: mutates queue.
        # TODO @thomasw21: We assume that each rank has a single contiguous list of blocks. This also means that we only recv activations from lower ranks
        self.microbatches_activations_to_recv.append(RecvActivation(from_rank=from_rank, p2p=p2p))

    def register_send_grad(self, grad: torch.Tensor, to_rank: int, p2p: P2P):
        # [FUNCTION DOC]
        # Queues a SendGrad. Inputs: grad, to_rank, p2p. Outputs: None. Side effects: mutates queue.
        # TODO @thomasw21: We assume that each rank has a single contiguous list of blocks. This also means that we only send gradients to lower ranks
        self.microbatches_grads_to_send.append(SendGrad(grad=grad, to_rank=to_rank, p2p=p2p))

    def register_recv_grad(self, from_rank: int, p2p: P2P):
        # [FUNCTION DOC]
        # Queues a RecvGrad. Inputs: from_rank, p2p. Outputs: None. Side effects: mutates queue.
        # TODO @thomasw21: We assume that each rank has a single contiguous list of blocks. This also means that we only recv gradients from higher ranks
        self.microbatches_grads_to_recv.append(RecvGrad(from_rank=from_rank, p2p=p2p))

    def run_communication(self, send_only_activation: bool = False):
        # [FUNCTION DOC]
        # Behavior: logs queue sizes. Executes at most one of each in order: send one activation (return here if
        # send_only_activation); receive one activation into activations_buffer (return if it does not require grad);
        # send one gradient; and if a gradient receive is queued, first keep sending queued activations while the last
        # sent activation did not require grad, then receive one gradient into grads_buffer.
        # Inputs: send_only_activation (bool, default False). Outputs: None.
        # Exceptions: UnboundLocalError (see issue). Side effects: blocking P2P communication; mutates queues and buffers.
        # KNOWN ISSUE: activation_send_requires_grad is only assigned when an activation send happened, so reaching the gradient-receive branch without a prior send raises UnboundLocalError.
        """Run communication in a specific order: send activation, recv activation, send grad, recv grad
        Only one communication is done at a time."""
        log_rank(
            f"activation_to_send: {len(self.microbatches_activations_to_send)} | "
            f"activation_to_recv: {len(self.microbatches_activations_to_recv)} | "
            f"grads_to_send: {len(self.microbatches_grads_to_send)} | "
            f"grads_to_recv: {len(self.microbatches_grads_to_recv)} | "
            f"activation_buffer: {len(self.activations_buffer)} | "
            f"grads_buffer: {len(self.grads_buffer)}",
            logger=logger,
            level=logging.DEBUG,
        )
        # Pop one send activation
        if len(self.microbatches_activations_to_send) > 0:
            send_activation = self.microbatches_activations_to_send.popleft()
            # Execute
            activation_send_requires_grad = send_activation.activation.requires_grad
            send_activation()
            if send_only_activation:
                return

        # Pop one recv activation
        if len(self.microbatches_activations_to_recv) > 0:
            recv_activation = self.microbatches_activations_to_recv.popleft()
            # Execute
            recv_activation_tensor = recv_activation()
            self.activations_buffer.append(recv_activation_tensor)
            # If somehow you receive a tensor without the need of backward, you shouldn't do cross communication
            if recv_activation_tensor.requires_grad is False:
                return

        # Pop one send gradient
        if len(self.microbatches_grads_to_send) > 0:
            send_grad = self.microbatches_grads_to_send.popleft()
            # Execute
            send_grad()

        # Pop one recv gradient
        if len(self.microbatches_grads_to_recv) > 0:
            # Send activation until `activation_send_requires_grad` is True
            while len(self.microbatches_activations_to_send) > 0 and not activation_send_requires_grad:
                send_activation = self.microbatches_activations_to_send.popleft()
                # Execute
                activation_send_requires_grad = send_activation.activation.requires_grad
                send_activation()
            recv_grad = self.microbatches_grads_to_recv.popleft()
            # Execute
            self.grads_buffer.append(recv_grad())

        # TODO @thomasw21: I need some mechanism to point to whatever is now sorted in a buffer, typically some id that would point to the correct tensor in our buffer instead of relying on the sorted list.

    def new_micro_batch_forward(self):
        # [FUNCTION DOC]
        # Appends an empty deque for a new microbatch. Outputs: None. Side effects: mutates state.
        self.microbatches_activations_requiring_backward.append(collections.deque())

    def pop_last_activations_requiring_backward(self) -> List[torch.Tensor]:
        # [FUNCTION DOC]
        # Pops and returns the oldest microbatch's activations (FIFO). Outputs: deque of tensors.
        # Exceptions: IndexError if empty. Side effects: mutates state.
        # KNOWN ISSUE: name says "last" but it pops from the left, i.e. the earliest microbatch (which is what 1F1B needs).
        return self.microbatches_activations_requiring_backward.popleft()

    def check_buffers_empty(self):
        # [FUNCTION DOC]
        # Asserts that backward, activation send/recv and grad send/recv queues are all empty.
        # Outputs: None. Exceptions: AssertionError naming the non-empty queue. Side effects: None.
        assert (
            len(self.microbatches_activations_requiring_backward) == 0
        ), f"There are still activations that require backward: {len(self.microbatches_activations_requiring_backward)}"
        assert (
            len(self.microbatches_activations_to_send) == 0
        ), f"There are activations left for me to send still: {len(self.microbatches_activations_to_send)}"
        assert (
            len(self.microbatches_activations_to_recv) == 0
        ), f"There are activations left for me to recv still: {len(self.microbatches_activations_to_recv)}"
        assert (
            len(self.microbatches_grads_to_send) == 0
        ), f"There are gradients left for me to send still: {len(self.microbatches_grads_to_send)}"
        assert (
            len(self.microbatches_grads_to_recv) == 0
        ), f"There are gradients left for me to recv still: {len(self.microbatches_grads_to_recv)}"


@dataclasses.dataclass
class PipelineEvalBatchState(PipelineBatchState):
    # [CLASS DOC]
    # Purpose: inference state with only activation send/recv queues and an activation buffer; no backward.
    # Inheritance: PipelineBatchState.
    # KNOWN ISSUE: queues are class-level deques shared between instances.
    microbatches_activations_to_send = collections.deque()
    microbatches_activations_to_recv = collections.deque()
    activations_buffer = collections.deque()

    def register_activation_requiring_backward(self, activation: torch.Tensor):
        # [FUNCTION DOC]
        # No-op. Inputs: activation. Outputs: None.
        pass

    def register_send_activation(self, activation: torch.Tensor, to_rank: int, p2p: P2P):
        # [FUNCTION DOC]
        # Behavior: queues a send, then runs communication if a receive is also queued (cross communication).
        # Inputs: activation, to_rank, p2p. Outputs: None. Side effects: possible P2P communication.
        # KNOWN ISSUE: the condition checks the receive queue twice instead of checking both send and receive queues.
        self.microbatches_activations_to_send.append(SendActivation(activation=activation, to_rank=to_rank, p2p=p2p))

        # There's a cross communication
        if len(self.microbatches_activations_to_recv) > 0 and len(self.microbatches_activations_to_recv) > 0:
            self.run_communication()

    def register_recv_activation(self, from_rank: int, p2p: P2P):
        # [FUNCTION DOC]
        # Behavior: queues a receive, then runs communication if the receive queue is non-empty (always true).
        # Inputs: from_rank, p2p. Outputs: None. Side effects: P2P communication.
        # KNOWN ISSUE: duplicated condition is always true, so every queued receive is executed immediately.
        self.microbatches_activations_to_recv.append(RecvActivation(from_rank=from_rank, p2p=p2p))

        # There's a cross communication
        if len(self.microbatches_activations_to_recv) > 0 and len(self.microbatches_activations_to_recv) > 0:
            self.run_communication()

    def register_send_grad(self, grad: torch.Tensor, to_rank: int, p2p: P2P):
        # [FUNCTION DOC]
        # Always raises NotImplementedError (no gradients in eval).
        raise NotImplementedError("You can't register a send grad in pipeline eval mode")

    def register_recv_grad(self, from_rank: int, p2p: P2P):
        # [FUNCTION DOC]
        # Always raises NotImplementedError (no gradients in eval).
        raise NotImplementedError("You can't register a recv grad in pipeline eval mode")

    def new_micro_batch_forward(self):
        # [FUNCTION DOC]
        # No-op. Outputs: None.
        pass

    def pop_last_activations_requiring_backward(self) -> List[torch.Tensor]:
        # [FUNCTION DOC]
        # No-op. Outputs: None (despite the list return annotation).
        pass

    def run_communication(self, send_only_activation: bool = False):

        # [FUNCTION DOC]
        # Behavior: pops at most one queued send and one queued receive. With neither raises. With only a receive,
        # receives into activations_buffer; with only a send, sends. With both, asserts same p2p; if both peers are higher
        # ranks than this one (assumed "lowest" rank in the cycle) sends first then receives, else receives first then sends.
        # Inputs: send_only_activation (ignored). Outputs: None.
        # Exceptions: ValueError if nothing to communicate; AssertionError if p2p objects differ.
        # Side effects: blocking P2P communication; mutates queues and buffer.
        # four cases:
        #  - you receive from higher rank and you send to higher rank
        #  - You receive from higher rank and you send to lower rank
        #  - you receive from lower rank and you send to higher rank
        #  - you receive from lower rank and you send to lower rank
        send_activation = None
        # Pop all send activation
        for _ in range(min(1, len(self.microbatches_activations_to_send))):
            send_activation = self.microbatches_activations_to_send.popleft()

        # Pop all recv activation
        recv_activation = None
        for _ in range(min(1, len(self.microbatches_activations_to_recv))):
            recv_activation = self.microbatches_activations_to_recv.popleft()

        if send_activation is None:
            if recv_activation is None:
                raise ValueError("Why the hell do we communicate when there's nothing to communicate?")
            self.activations_buffer.append(recv_activation())
        else:
            if recv_activation is None:
                send_activation()
            else:
                # Define in which order to we do it.
                # Actually we can't do any heuristics as you need global information in order to define clear ordering.
                # We make a BIG assumption that only ONE rank receives from higher rank and sends to higher rank.
                # In this case we find the "lowest" rank, send first
                # All the other ranks receive first and send after
                # Lowest rank receives.
                # If we knew who was involved in the cycle, we could just randomly choose one rank to first send then recv, however it's not clear who's involved
                p2p = send_activation.p2p
                assert p2p == recv_activation.p2p
                is_lowest = send_activation.to_rank > dist.get_rank(
                    p2p.pg
                ) and recv_activation.from_rank > dist.get_rank(p2p.pg)
                if is_lowest:
                    send_activation()
                    self.activations_buffer.append(recv_activation())
                else:
                    self.activations_buffer.append(recv_activation())
                    send_activation()

    def check_buffers_empty(self):
        # [FUNCTION DOC]
        # Asserts send/recv queues and activations_buffer are empty. Exceptions: AssertionError. Side effects: None.
        assert (
            len(self.microbatches_activations_to_send) == 0
        ), f"There are activations left for me to send still: {len(self.microbatches_activations_to_send)}"
        assert (
            len(self.microbatches_activations_to_recv) == 0
        ), f"There are activations left for me to recv still: {len(self.microbatches_activations_to_recv)}"
        assert (
            len(self.activations_buffer) == 0
        ), f"There are activations left in the buffer: {len(self.activations_buffer)}"
