# [MODULE DOC]
# Purpose: Runtime sanity checks run by the trainer around each training step (before/after the
# pipeline engine train_batch_iter, before/after the optimizer step) to catch desynchronized weights,
# gradients or optimizer states across data/context-parallel replicas and tied groups, plus a
# dataloader inspection helper.
# Main contents: assert_tensor_synced_across_pg, assert_fail_except_rank_with,
# before_tbi_sanity_checks, after_tbi_sanity_checks, before_optim_step_sanity_checks,
# after_optim_step_sanity_checks, check_optim_state_in_sync, sanity_check_dataloader.
# All step checks are skipped when config.general.ignore_sanity_checks is true.
# Import-time side effects: creates a module logger.
from contextlib import contextmanager
from typing import Callable, Optional

import torch
from transformers import AutoTokenizer

from nanotron import distributed as dist
from nanotron import logging, optim
from nanotron.config import Config
from nanotron.logging import get_logger, log_rank
from nanotron.models import NanotronModel
from nanotron.optim.gradient_accumulator import GradientAccumulator
from nanotron.parallel import ParallelContext
from nanotron.parallel.tied_parameters import get_tied_id_to_param

logger = get_logger(__name__)


def assert_tensor_synced_across_pg(
    tensor: torch.Tensor,
    pg: dist.ProcessGroup,
    msg: Optional[Callable[[str], str]] = None,
    reference_rank: int = 0,
):
    # [FUNCTION DOC]
    # Behavior: the reference rank broadcasts its tensor to all ranks of pg (others receive into an empty
    # buffer); every rank then compares its own tensor with the reference using torch.testing.assert_close.
    # Inputs: tensor (torch.Tensor), pg (ProcessGroup), msg (optional callable building the error message),
    # reference_rank (group rank, default 0).
    # Outputs: None.
    # Exceptions: AssertionError when tensors differ.
    # Side effects: broadcast collective on pg (must be called by all ranks of pg).
    """Assert that `tensor` is synced across `pg` with reference rank. Note that this always passes for reference rank"""
    if dist.get_rank(pg) == reference_rank:
        reference_tensor = tensor
    else:
        reference_tensor = torch.empty_like(tensor)
    dist.broadcast(
        reference_tensor,
        src=dist.get_global_rank(group=pg, group_rank=reference_rank),
        group=pg,
    )

    # TODO @nouamane: Getting Greatest absolute difference: 4.6e-10 at large scale when syncing tied weights
    torch.testing.assert_close(tensor, reference_tensor, msg=msg)


# TODO @nouamanetazi: remove this with SANITY_CHECKS
@contextmanager
def assert_fail_except_rank_with(exception_class, rank_exception, pg):
    # [FUNCTION DOC]
    # HELPER: test-style context manager asserting that every rank except one raises a given exception.
    # Behavior: runs the body; if exception_class is raised it errors when the rank is rank_exception and
    # otherwise swallows it; any other exception becomes an AssertionError; if nothing was raised on a rank
    # other than rank_exception, raises AssertionError.
    # Inputs: exception_class (exception type), rank_exception (group rank expected not to raise), pg.
    # Outputs: yields None.
    # Exceptions: AssertionError on any expectation mismatch.
    # Side effects: None.
    try:
        yield
    except exception_class:
        if rank_exception == dist.get_rank(pg):
            raise AssertionError(f"Expected rank {rank_exception} to not raise {exception_class}.")
        else:
            return

    except Exception as e:
        raise AssertionError(f"Expected {exception_class} to be raised, but got {type(e)} instead:\n{e}")
    if dist.get_rank(pg) != rank_exception:
        raise AssertionError(f"Expected {exception_class} to be raised, but no exception was raised.")


def before_tbi_sanity_checks(
    config: Config,
    parallel_context: ParallelContext,
    unwrapped_model: NanotronModel,
    grad_accumulator: GradientAccumulator,
    lr_scheduler: torch.optim.lr_scheduler.LRScheduler,
) -> None:
    # [FUNCTION DOC]
    # Behavior: unless ignored, checks that all parameters (sorted by name) are synced across the dp_cp
    # group; tied parameters are synced across their tied groups; model grads are None or exactly zero;
    # grad accumulator fp32 buffers are exactly zero; the first optimizer param group's lr equals the
    # scheduler's last lr; then calls the model's own before_tbi_sanity_checks.
    # Inputs: config (Config), parallel_context, unwrapped_model (NanotronModel), grad_accumulator
    # (optional GradientAccumulator), lr_scheduler.
    # Outputs: None.
    # Exceptions: AssertionError on any failed check.
    # Side effects: broadcast collectives on dp_cp and tied groups.
    if not config.general.ignore_sanity_checks:
        # SANITY CHECK: Check that the model params are synchronized across dp_cp
        for name, param in sorted(unwrapped_model.named_parameters(), key=lambda x: x[0]):
            assert_tensor_synced_across_pg(
                tensor=param,
                pg=parallel_context.dp_cp_pg,
                msg=lambda err: f"{name} are not synchronized across DP_CP {err}",
            )

        # SANITY CHECK: Tied weights are synchronized
        tied_params_list = sorted(
            get_tied_id_to_param(
                parameters=unwrapped_model.parameters(),
                root_module=unwrapped_model,
            ).items(),
            key=lambda x: x[0],
        )
        for (name, group_ranks), param in tied_params_list:
            group = parallel_context.world_ranks_to_pg[group_ranks]
            assert_tensor_synced_across_pg(
                tensor=param,
                pg=group,
                msg=lambda err: f"[Before train] Tied weights {name} are not synchronized. {err}",
            )

        # SANITY CHECK: Check that model grads are zeroed or None
        for name, param in unwrapped_model.named_parameters():
            if param.grad is not None:
                torch.testing.assert_close(
                    param.grad,
                    torch.zeros_like(param.grad),
                    atol=0,
                    rtol=0,
                    msg="Model half precision grads must be zeroed or None in first accumulation step.",
                )

        # SANITY CHECK: Check that the grad accumulator buffers are ready for DDP
        if grad_accumulator is not None:
            for _, elt in grad_accumulator.fp32_grad_buffers.items():
                fp32_grad_buffer = elt["fp32_grad"]
                torch.testing.assert_close(
                    fp32_grad_buffer,
                    torch.zeros_like(fp32_grad_buffer),
                    atol=0,
                    rtol=0,
                    msg="Grad accumulator buffers must be zeroed in first accumulation step.",
                )

            # TODO: add checks for memory contiguousness

        # SANITY CHECK: Check that optimizer's lr is synchronized with lr_scheduler
        for i, group in enumerate(lr_scheduler.optimizer.param_groups):
            assert (
                group["lr"] == lr_scheduler.get_last_lr()[i]
            ), f"Optimizer and LR scheduler are not in sync. Got {group['lr']} and {lr_scheduler.get_last_lr()[i]}"
            break

        # SANITY CHECK: run model specific sanity checks
        unwrapped_model.before_tbi_sanity_checks()


def after_tbi_sanity_checks(
    config: Config,
    parallel_context: ParallelContext,
    unwrapped_model: NanotronModel,
    grad_accumulator: GradientAccumulator,
) -> None:
    # [FUNCTION DOC]
    # Behavior: unless ignored, for every parameter requiring grad, resolves its (tied full) name, fetches
    # the gradient from the accumulator buffer or the param, raises if it contains NaN or Inf, logs an
    # error if it is missing; then calls the model's after_tbi_sanity_checks.
    # Inputs: config, parallel_context, unwrapped_model, grad_accumulator (optional).
    # Outputs: None.
    # Exceptions: ValueError on NaN/Inf gradients.
    # Side effects: logging.
    # KNOWN ISSUE: the None check comes after calling isnan on the gradient, so a missing gradient raises TypeError instead of being logged.
    if not config.general.ignore_sanity_checks:
        # SANITY CHECK: Check that gradient flow on the entire model
        # SANITY CHECK: Check that all parameters that required gradients, have actually a gradient
        # SANITY CHECK: Check for nan/inf
        for name, param in unwrapped_model.named_parameters():
            if not param.requires_grad:
                continue

            if param.is_tied:
                tied_info = param.get_tied_info()
                name = tied_info.get_full_name_from_module_id_to_prefix(
                    module_id_to_prefix=unwrapped_model.module_id_to_prefix
                )

            if grad_accumulator is not None:
                grad = grad_accumulator.get_grad_buffer(name=name)
            else:
                grad = param.grad

            if torch.isnan(grad).any() or torch.isinf(grad).any():
                raise ValueError(f"Gradient is nan or inf for {name}")
            if grad is None:
                log_rank(
                    f"Process rank { dist.get_rank(parallel_context.world_pg)}/{parallel_context.world_pg.size()}: {name} is missing gradient",
                    logger=logger,
                    level=logging.ERROR,
                )

        # SANITY CHECK: run model specific sanity checks
        unwrapped_model.after_tbi_sanity_checks()


def before_optim_step_sanity_checks(
    config: Config,
    parallel_context: ParallelContext,
    unwrapped_model: NanotronModel,
    grad_accumulator: GradientAccumulator,
    optimizer: optim.BaseOptimizer,
) -> None:
    # [FUNCTION DOC]
    # Behavior: unless ignored, checks tied gradients are non-None and synced across tied groups, all
    # gradients are non-None and synced across dp_cp, parameters synced across dp_cp, tied parameters
    # synced across tied groups, optimizer states synced across dp_cp, then calls the model's
    # before_optim_step_sanity_checks.
    # Inputs: config, parallel_context, unwrapped_model, grad_accumulator (optional), optimizer (BaseOptimizer).
    # Outputs: None.
    # Exceptions: AssertionError on missing or unsynced data.
    # Side effects: broadcast collectives.
    if not config.general.ignore_sanity_checks:
        # SANITY CHECK: Test tied weights gradients are synchronized
        for (name, group_ranks), param in sorted(
            get_tied_id_to_param(parameters=unwrapped_model.parameters(), root_module=unwrapped_model).items(),
            key=lambda x: x[0],
        ):
            if not param.requires_grad:
                continue

            if grad_accumulator is not None:
                grad = grad_accumulator.get_grad_buffer(name=name)
            else:
                grad = param.grad

            assert grad is not None, f"Grad is None for {name}"
            group = parallel_context.world_ranks_to_pg[group_ranks]
            assert_tensor_synced_across_pg(
                tensor=grad,
                pg=group,
                msg=lambda err: f"[Before optimizer step] Tied weights grads for {name} are not synchronized. {err}",
            )

        # SANITY CHECK: Test gradients are synchronized across DP
        for name, param in sorted(unwrapped_model.named_parameters(), key=lambda x: x[0]):
            if not param.requires_grad:
                continue

            if param.is_tied:
                tied_info = param.get_tied_info()
                name = tied_info.get_full_name_from_module_id_to_prefix(
                    module_id_to_prefix=unwrapped_model.module_id_to_prefix
                )

            if grad_accumulator is not None:
                grad = grad_accumulator.get_grad_buffer(name=name)
            else:
                grad = param.grad

            assert grad is not None, f"Grad is None for {name}"
            assert_tensor_synced_across_pg(
                tensor=grad,
                pg=parallel_context.dp_cp_pg,
                msg=lambda err: f"[Before optimizer step] weights grads for {name} are not synchronized across DP_CP. {err}",
            )

        # SANITY CHECK: Check that the model params are synchronized across dp
        for name, param in sorted(unwrapped_model.named_parameters(), key=lambda x: x[0]):
            assert_tensor_synced_across_pg(
                tensor=param,
                pg=parallel_context.dp_cp_pg,
                msg=lambda err: f"{name} are not synchronized across DP_CP {err}",
            )

        # SANITY CHECK: Tied weights are synchronized
        tied_params_list = sorted(
            get_tied_id_to_param(parameters=unwrapped_model.parameters(), root_module=unwrapped_model).items(),
            key=lambda x: x[0],
        )

        for (name, group_ranks), param in tied_params_list:
            group = parallel_context.world_ranks_to_pg[group_ranks]
            assert_tensor_synced_across_pg(
                tensor=param,
                pg=group,
                msg=lambda err: f"[Before optimizer step] Tied weights {name} are not synchronized. {err}",
            )

        # SANITY CHECK: Check that optimizer states are synchronized across DP_CP
        check_optim_state_in_sync(optimizer.state_dict(), parallel_context.dp_cp_pg)

        # SANITY CHECK: run model specific sanity checks
        unwrapped_model.before_optim_step_sanity_checks()


def after_optim_step_sanity_checks(
    config: Config,
    parallel_context: ParallelContext,
    unwrapped_model: NanotronModel,
    grad_accumulator: GradientAccumulator,
) -> None:
    # [FUNCTION DOC]
    # Behavior: unless ignored, logs an error for each parameter requiring grad that still has a gradient,
    # then calls the model's after_optim_step_sanity_checks.
    # Inputs: config, parallel_context, unwrapped_model, grad_accumulator (unused).
    # Outputs: None. Exceptions: None explicitly raised. Side effects: logging.
    if not config.general.ignore_sanity_checks:
        # SANITY CHECK: Check that gradients is cleared
        for name, param in unwrapped_model.named_parameters():
            if not param.requires_grad:
                continue

            if param.grad is not None:
                log_rank(
                    f"Process rank { dist.get_rank(parallel_context.world_pg)}/{parallel_context.world_pg.size()}: {name} still has gradient despite having ran the optimizer",
                    logger=logger,
                    level=logging.ERROR,
                )

        # SANITY CHECK: run model specific sanity checks
        unwrapped_model.after_optim_step_sanity_checks()


def check_optim_state_in_sync(optim_state_dict: dict, pg: dist.ProcessGroup):
    # [FUNCTION DOC]
    # Behavior: iterates optimizer states sorted by parameter index and asserts every state tensor except
    # "step" is synced across pg.
    # Inputs: optim_state_dict (optimizer state dict with a "state" entry), pg (ProcessGroup).
    # Outputs: None. Exceptions: AssertionError if out of sync. Side effects: broadcast collectives.
    for _, optim_state in sorted(optim_state_dict["state"].items(), key=lambda x: x[0]):
        for name, tensor in optim_state.items():
            if name == "step":
                continue
            assert_tensor_synced_across_pg(
                tensor=tensor, pg=pg, msg=lambda err: f"{name} are not synced across DP {err}"
            )


def sanity_check_dataloader(dataloader, tokenizer_path, sanity_check_dataloader_interval=None):
    # [FUNCTION DOC]
    # Behavior: no-op when the interval is None. Otherwise loads the tokenizer; on global rank 0 truncates
    # sanity_check.txt, iterates the dataloader and for every batch index divisible by the interval prints
    # the batch and decoded samples (first 400 chars) and appends full texts to the file, stopping after 10
    # inspected batches, then raises AssertionError to stop training. Other ranks wait on a barrier.
    # Inputs: dataloader (iterable of dicts containing input_ids), tokenizer_path (str),
    # sanity_check_dataloader_interval (optional int).
    # Outputs: None.
    # Exceptions: AssertionError always raised on rank 0 when enabled.
    # Side effects: file writes in the working directory, printing, tokenizer download, global barrier.
    # KNOWN ISSUE: rank 0 raises before reaching the barrier, so other ranks block on the barrier forever.
    """
    Debug function to check dataloader samples.
    Args:
        dataloader: The dataloader to check
        tokenizer_path: Path to the tokenizer
        sanity_check_dataloader_interval: Interval at which to check samples
    """
    if sanity_check_dataloader_interval is None:
        return

    NUM_BATCHES = 10
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    if dist.get_rank() == 0:
        check_step = -1

        with open("sanity_check.txt", "w") as f:
            f.write("")

        for i, batch in enumerate(dataloader):
            check_step += 1
            if i % sanity_check_dataloader_interval == 0:
                with open("sanity_check.txt", "a") as f:
                    f.write("\n\n")
                    f.write("*" * 40)
                    f.write(f"Sanity check {check_step}")
                    f.write("*" * 40)
                print(batch)

                texts = tokenizer.batch_decode(
                    batch["input_ids"], skip_special_tokens=False, clean_up_tokenization_spaces=False
                )

                for j, text in enumerate(texts):
                    print(f"\n\n>>Batch {i} || Sample {j}<<\n")
                    print(text[:400])
                    with open("sanity_check.txt", "a") as f:
                        f.write(f"\n\n>>Batch {i} || Sample {j}<<\n")
                        f.write(text)

                if i // sanity_check_dataloader_interval == NUM_BATCHES - 1:
                    break
        raise AssertionError("Sanity check complete - stopping training")

    dist.barrier()
