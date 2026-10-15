# [MODULE DOC]
# Purpose: autograd functions that make pipeline P2P sends/receives differentiable by queuing communication in
# a PipelineBatchState instead of calling torch.distributed directly.
# Main contents: SendTensorToPipelineBuffer, SendTensorWithoutGradientToPipelineBuffer,
# send_to_pipeline_state_buffer, RecvTensorFromPipelineBuffer, recv_from_pipeline_state_buffer.
# How it fits: used by PipelineBlock.forward when a pipeline engine is attached.
# Import-time side effects: creates a module logger.
import torch
from nanotron import logging
from nanotron.parallel.pipeline_parallel.p2p import P2P
from nanotron.parallel.pipeline_parallel.state import PipelineBatchState

logger = logging.get_logger(__name__)


class SendTensorToPipelineBuffer(torch.autograd.Function):
    # [CLASS DOC]
    # Purpose: autograd function whose forward queues an activation send and returns a dummy CPU scalar that the
    # engine backpropagates from; its backward receives the gradient for that activation from the destination rank.
    # Inheritance: torch.autograd.Function.
    """Make sending tensors differentiable. The difference is here we don't use `torch.distributed` primites, but store events that's we will pop whenever we need"""

    @staticmethod
    def forward(
        ctx,
        activation: torch.Tensor,
        to_rank: int,
        p2p: P2P,
        pipeline_state: PipelineBatchState,
    ):
        # [FUNCTION DOC]
        # Behavior: asserts activation requires grad, stores p2p, to_rank and pipeline_state on ctx, registers the send,
        # returns a float CPU scalar 1 requiring grad.
        # Inputs: ctx, activation (tensor), to_rank (int, PP group rank), p2p (P2P), pipeline_state.
        # Outputs: dummy scalar tensor. Exceptions: AssertionError if activation does not require grad.
        # Side effects: queues a send in pipeline_state.
        assert activation.requires_grad
        ctx.p2p = p2p
        ctx.to_rank = to_rank
        ctx.pipeline_state = pipeline_state

        # Send tensors
        pipeline_state.register_send_activation(activation, to_rank=to_rank, p2p=p2p)

        # HACK @thomasw21: This forces the trigger to backward
        return torch.tensor(1, dtype=torch.float, device="cpu", requires_grad=True)

    @staticmethod
    def backward(ctx, grad_tensor):
        # [FUNCTION DOC]
        # Behavior: registers a gradient receive from to_rank, runs queued communication if the grads buffer is empty, pops
        # the next gradient and returns it for the activation.
        # Inputs: ctx, grad_tensor (ignored). Outputs: tuple (gradient, None, None, None).
        # Exceptions: IndexError if no gradient was received. Side effects: P2P receive.
        p2p = ctx.p2p
        to_rank = ctx.to_rank
        pipeline_state = ctx.pipeline_state

        # send a gradient and store it in buffer
        pipeline_state.register_recv_grad(from_rank=to_rank, p2p=p2p)
        if len(pipeline_state.grads_buffer) == 0:
            pipeline_state.run_communication()

        grad_tensor = pipeline_state.grads_buffer.popleft()

        return grad_tensor, None, None, None


class SendTensorWithoutGradientToPipelineBuffer(torch.autograd.Function):
    # [CLASS DOC]
    # Purpose: like SendTensorToPipelineBuffer for activations not requiring grad; uses a dummy grad-requiring input
    # so backward fires, and backward only flushes pending activation sends.
    # Inheritance: torch.autograd.Function.
    @staticmethod
    def forward(
        ctx,
        dummy_input: torch.Tensor,
        activation: torch.Tensor,
        to_rank: int,
        p2p: P2P,
        pipeline_state: PipelineBatchState,
    ):
        # [FUNCTION DOC]
        # Behavior: asserts dummy_input requires grad and activation does not, stores context, registers the send and
        # returns a dummy CPU scalar requiring grad.
        # Inputs: ctx, dummy_input, activation, to_rank, p2p, pipeline_state.
        # Outputs: dummy scalar. Exceptions: AssertionError on grad flags. Side effects: queues a send.
        assert dummy_input.requires_grad
        assert activation.requires_grad is False
        ctx.p2p = p2p
        ctx.to_rank = to_rank
        ctx.pipeline_state = pipeline_state

        # Send tensors
        pipeline_state.register_send_activation(activation, to_rank=to_rank, p2p=p2p)

        # HACK @thomasw21: This forces the trigger to backward
        return torch.tensor(1, dtype=torch.float, device="cpu", requires_grad=True)

    @staticmethod
    def backward(ctx, grad_tensor):
        # [FUNCTION DOC]
        # Behavior: runs queued communication in send-only-activation mode; returns no gradients.
        # Inputs: ctx, grad_tensor. Outputs: five Nones. Exceptions: None explicitly raised. Side effects: P2P sends.
        pipeline_state = ctx.pipeline_state

        # send only the activations
        pipeline_state.run_communication(send_only_activation=True)

        return None, None, None, None, None


def send_to_pipeline_state_buffer(tensor: torch.Tensor, to_rank: int, p2p: P2P, pipeline_state: PipelineBatchState):
    # [FUNCTION DOC]
    # Behavior: applies the grad or non-grad send function depending on tensor.requires_grad (creating a CPU dummy
    # input in the latter case) and registers the dummy result as an activation requiring backward.
    # Inputs: tensor, to_rank (int), p2p (P2P), pipeline_state. Outputs: None.
    # Exceptions: None explicitly raised. Side effects: queues communication; mutates pipeline_state.
    # This is used in order to know where to backward from.
    if tensor.requires_grad:
        result = SendTensorToPipelineBuffer.apply(tensor, to_rank, p2p, pipeline_state)
    else:
        # Trick that backward mechanism to just send the tensor.
        dummy_input = torch.empty(1, dtype=torch.float, requires_grad=True, device="cpu")
        result = SendTensorWithoutGradientToPipelineBuffer.apply(dummy_input, tensor, to_rank, p2p, pipeline_state)

    pipeline_state.register_activation_requiring_backward(result)


class RecvTensorFromPipelineBuffer(torch.autograd.Function):
    # [CLASS DOC]
    # Purpose: identity autograd function on a received activation whose backward queues sending the gradient back
    # to the source rank. Inheritance: torch.autograd.Function.
    """Make receiving tensors differentiable"""

    @staticmethod
    def forward(ctx, activation: torch.Tensor, from_rank: int, p2p: P2P, pipeline_state: PipelineBatchState):
        # [FUNCTION DOC]
        # Behavior: stores pipeline_state, p2p and from_rank on ctx and returns activation unchanged.
        # Inputs: ctx, activation, from_rank, p2p, pipeline_state. Outputs: activation. Exceptions/Side effects: None.
        ctx.pipeline_state = pipeline_state
        ctx.p2p = p2p
        ctx.from_rank = from_rank

        return activation

    @staticmethod
    def backward(ctx, grad_tensor):
        # [FUNCTION DOC]
        # Behavior: registers a gradient send of grad_tensor to from_rank; returns no input gradients.
        # Inputs: ctx, grad_tensor. Outputs: four Nones. Exceptions: None. Side effects: queues a send.
        pipeline_state = ctx.pipeline_state
        from_rank = ctx.from_rank
        p2p = ctx.p2p

        # Send tensors
        pipeline_state.register_send_grad(grad_tensor, to_rank=from_rank, p2p=p2p)

        return None, None, None, None


def recv_from_pipeline_state_buffer(from_rank: int, p2p: P2P, pipeline_state: PipelineBatchState):
    # [FUNCTION DOC]
    # Behavior: registers an activation receive from from_rank, runs communication if no activation is buffered, pops
    # the next activation and wraps it in RecvTensorFromPipelineBuffer.
    # Inputs: from_rank (int), p2p (P2P), pipeline_state. Outputs: received tensor.
    # Exceptions: IndexError if nothing was received. Side effects: P2P receive; mutates state.
    pipeline_state.register_recv_activation(from_rank=from_rank, p2p=p2p)
    if len(pipeline_state.activations_buffer) == 0:
        pipeline_state.run_communication()
    activation = pipeline_state.activations_buffer.popleft()
    return RecvTensorFromPipelineBuffer.apply(activation, from_rank, p2p, pipeline_state)
