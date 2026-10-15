# [MODULE DOC]
# Purpose: data-parallel helpers around DDP and gradient accumulators.
# Main contents: ddp_trigger_sync_in_bwd, sync_gradients_across_dp.
# Import-time side effects: None.
from contextlib import contextmanager
from typing import Optional

import torch
from nanotron import distributed as dist
from nanotron.optim.gradient_accumulator import GradientAccumulator
from torch import nn


@contextmanager
def ddp_trigger_sync_in_bwd(model_ddp):
    # [FUNCTION DOC]
    # Behavior: context manager forcing the next DDP backward to sync gradients: asserts the model is DDP, saves and
    # sets require_backward_grad_sync and require_forward_param_sync to True, calls reducer.prepare_for_backward with
    # an empty list (arming the reducer even inside no_sync), yields, then restores the flags.
    # Inputs: model_ddp (DistributedDataParallel). Outputs: generator yielding once.
    # Exceptions: AssertionError if not DDP.
    # Side effects: mutates DDP flags and reducer state (uses DDP internals).
    """Trigger the sync of the gradients in the next backward pass of DDP model."""
    assert isinstance(model_ddp, torch.nn.parallel.DistributedDataParallel)
    old_require_backward_grad_sync = model_ddp.require_backward_grad_sync
    old_require_forward_param_sync = model_ddp.require_forward_param_sync

    model_ddp.require_backward_grad_sync = True
    model_ddp.require_forward_param_sync = True
    # https://github.com/pytorch/pytorch/blob/master/torch/csrc/distributed/c10d/reducer.cpp#L1325-L1356
    model_ddp.reducer.prepare_for_backward([])
    try:
        yield
    finally:
        model_ddp.require_backward_grad_sync = old_require_backward_grad_sync
        model_ddp.require_forward_param_sync = old_require_forward_param_sync


@torch.profiler.record_function("sync_gradients_across_dp")
def sync_gradients_across_dp(
    module: nn.Module,
    dp_pg: dist.ProcessGroup,
    reduce_op: dist.ReduceOp,
    grad_accumulator: Optional[GradientAccumulator],
    **sync_options,
):
    # [FUNCTION DOC]
    # Behavior: if a grad_accumulator is given, delegates to its sync_gradients_across_dp with extra options;
    # otherwise all-reduces every parameter's grad with reduce_op over dp_pg (one blocking call per param).
    # Inputs: module (nn.Module), dp_pg (ProcessGroup), reduce_op (ReduceOp), grad_accumulator (or None),
    # sync_options (keyword options, e.g. reduce_scatter, for the accumulator path).
    # Outputs: None.
    # Exceptions: None explicitly raised; errors if a parameter grad is None in the fallback path.
    # Side effects: collectives on dp_pg; in-place grad mutation.
    # KNOWN ISSUE: fallback path ignores sync_options and fails for params whose grad is None (e.g. frozen params).
    """Sync gradients across data parallelism.

    Args:
        module: The module to sync gradients for.
        dp_pg: The data parallelism process group.
        reduce_op: The reduce operation to use.
        grad_accumulator: The gradient accumulator to use.
        sync_options: Additional options given when using `grad_accumulator`. Please look at `GradientAccumulator.sync_gradients_across_dp` for documentation
    """
    if grad_accumulator is not None:
        # This is an optimized path that
        grad_accumulator.sync_gradients_across_dp(dp_pg=dp_pg, reduce_op=reduce_op, **sync_options)
        return

    # Sync gradients
    for name, param in module.named_parameters():
        dist.all_reduce(param.grad, op=reduce_op, group=dp_pg)
