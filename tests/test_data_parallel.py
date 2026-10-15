# [MODULE DOC]
# Purpose: distributed pytest test for data parallel gradient synchronization with all-forward-all-backward accumulation using ddp_trigger_sync_in_bwd and a custom DDP communication hook.
# Main contents: test_ddp_with_afab and its per-rank body.
from contextlib import nullcontext

import pytest
import torch
from helpers.exception import assert_fail_except_rank_with
from helpers.utils import available_gpus, init_distributed, rerun_if_address_is_in_use
from nanotron import distributed as dist
from nanotron.parallel import ParallelContext
from nanotron.parallel.data_parallel.utils import ddp_trigger_sync_in_bwd
from nanotron.parallel.parameters import NanotronParameter
from nanotron.sanity_checks import assert_tensor_synced_across_pg
from torch import nn
from torch.distributed import GradBucket


@pytest.mark.skipif(available_gpus() < 2, reason="Testing test_ddp_with_afab requires at least 2 gpus")
@pytest.mark.parametrize("accumulation_steps", [1, 3])
@rerun_if_address_is_in_use()
def test_ddp_with_afab(accumulation_steps):
    # [FUNCTION DOC]
    # Behavior: runs _test_ddp_with_afab on 2 data parallel ranks (skipped with fewer than 2 GPUs).
    # Inputs: accumulation_steps, parametrized 1 or 3.
    # Outputs: None.
    # Exceptions: propagates failures.
    # Side effects: process spawning.
    init_distributed(tp=1, dp=2, pp=1)(_test_ddp_with_afab)(accumulation_steps=accumulation_steps)


def _test_ddp_with_afab(parallel_context: ParallelContext, accumulation_steps: int):
    # [FUNCTION DOC]
    # Behavior: wraps a bias-free float16 linear(3,2) on CUDA, whose weight is a NanotronParameter, in DistributedDataParallel over dp with a custom average all-reduce hook. Runs all forwards under no_sync, storing the losses; then runs the backwards in order, wrapping only the last in ddp_trigger_sync_in_bwd. After each backward checks the gradient: after the last it must be synchronized across dp; before that, the sync assertion must fail on every rank except dp rank 0 (gradients differ because inputs differ).
    # Inputs: parallel_context; accumulation_steps.
    # Outputs: None.
    # Exceptions: AssertionError on failure.
    # Side effects: GPU allocation; dp collectives; process group destruction.
    half_precision = torch.float16

    def allreduce_hook(process_group: dist.ProcessGroup, bucket: GradBucket):
        # [FUNCTION DOC]
        # HELPER: DDP communication hook.
        # Behavior: launches an asynchronous averaging all-reduce of the bucket's flat gradient buffer on the given process group (or dp if None) and returns a future resolving to the reduced buffer.
        # Inputs: process_group, optional group (the hook state); bucket, a GradBucket.
        # Outputs: torch Future of the reduced tensor.
        # Exceptions: None explicitly raised.
        # Side effects: all-reduce collective, in-place on the bucket buffer.
        # DDP groups grads in GradBuckets. This hook is called throughout the bwd pass, once each bucket is ready to overlap communication with computation.
        # See https://pytorch.org/docs/stable/ddp_comm_hooks.html#what-does-a-communication-hook-operate-on for more details.
        half_flat_bucket_buffer = bucket.buffer()
        group_to_use = process_group if process_group is not None else parallel_context.dp_pg

        return (
            dist.all_reduce(half_flat_bucket_buffer, group=group_to_use, async_op=True, op=dist.ReduceOp.AVG)
            .get_future()
            .then(lambda fut: fut.value()[0])
        )

    model_hook = nn.Linear(3, 2, bias=False, dtype=half_precision, device="cuda")

    # Create Nanotron Parameter
    model_hook.weight = NanotronParameter(model_hook.weight)

    model_ddp_hook = torch.nn.parallel.DistributedDataParallel(
        model_hook,
        process_group=parallel_context.dp_pg,
    )

    # Register DDP hook
    model_ddp_hook.register_comm_hook(state=None, hook=allreduce_hook)

    activations = []
    # All forward
    for i in range(accumulation_steps):
        input = torch.randn(5, 3, dtype=half_precision, device="cuda")

        with model_ddp_hook.no_sync():
            loss_hook = model_ddp_hook(input).sum()

        activations.append(loss_hook)

    # All backward
    for i in range(accumulation_steps):
        context = nullcontext()
        if i == accumulation_steps - 1:
            context = ddp_trigger_sync_in_bwd(model_ddp_hook)  # triggers a sync for the final backward
        loss_hook = activations[i]
        with context:
            loss_hook.backward()

        grad_hook = model_ddp_hook.module.weight.grad.clone()

        # Check that the gradients are synchronized across DP
        if i == accumulation_steps - 1:
            assert_tensor_synced_across_pg(grad_hook, parallel_context.dp_pg)
        else:
            with assert_fail_except_rank_with(AssertionError, rank_exception=0, pg=parallel_context.dp_pg):
                assert_tensor_synced_across_pg(grad_hook, parallel_context.dp_pg)

    parallel_context.destroy()
