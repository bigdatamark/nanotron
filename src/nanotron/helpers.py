# [MODULE DOC]
# Purpose: grab-bag of training-setup helpers used by nanotron's trainer: vocab padding, TP random
# state setup, learning-rate scheduler construction, optimizer/parameter-group construction (custom
# per-parameter LR for muP/SP, weight-decay exclusion, ZeRO, FP32 grad accumulation, DDP hooks),
# profiler creation, communication benchmarks, throughput table/CSV logging, and data-stage resume
# bookkeeping.
# Main contents: _vocab_size_with_padding, init_random_states, lr_scheduler_builder,
# get_custom_weight_decay_for_named_parameters, get_custom_lr_for_named_parameters,
# merge_named_param_groups, init_optimizer_and_grad_accumulator, test_equal_dict, get_profiler,
# get_all_comps, test_all_pair_to_pair, create_table_log, get_formatted_value, create_table_output,
# write_to_csv, log_throughput, compute_remain_train_steps_of_a_data_stage_from_ckp,
# get_consumed_train_samples_of_a_data_stage_from_ckp.
# Import-time side effects: creates a module logger.
import contextlib
import csv
import gc
import math
import os
import re
import time
from datetime import datetime
from functools import partial
from math import ceil
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import torch
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.optim.lr_scheduler import LambdaLR
from torch.profiler import ProfilerActivity, profile, tensorboard_trace_handler

from nanotron import distributed as dist
from nanotron import logging
from nanotron.config import Config, DatasetStageArgs, LRSchedulerArgs, OptimizerArgs, ParallelismArgs
from nanotron.distributed import ProcessGroup
from nanotron.logging import LogItem, human_format, log_rank
from nanotron.models.base import NanotronModel
from nanotron.optim.base import BaseOptimizer, Optimizer
from nanotron.optim.gradient_accumulator import (
    FP32GradBucketManager,
    FP32GradientAccumulator,
    GradientAccumulator,
    get_fp32_accum_hook,
)
from nanotron.optim.named_optimizer import NamedOptimizer
from nanotron.optim.optimizer_from_gradient_accumulator import (
    OptimizerFromGradientAccumulator,
)
from nanotron.optim.zero import ZeroDistributedOptimizer
from nanotron.parallel import ParallelContext
from nanotron.parallel.tensor_parallel.nn import TensorParallelLinearMode
from nanotron.random import (
    RandomStates,
    get_current_random_state,
    get_synced_random_state,
)
from nanotron.scaling.parametrization import LearningRateForSP, LearningRateForSpectralMup, ParametrizationMethod
from nanotron.serialize import DataStageMetadata
from nanotron.serialize.metadata import TrainingMetadata

logger = logging.get_logger(__name__)


def _vocab_size_with_padding(orig_vocab_size: int, pg_size: int, make_vocab_size_divisible_by: int):
    # [FUNCTION DOC]
    # HELPER: pads the vocabulary for tensor-parallel embedding sharding.
    # Behavior: rounds orig_vocab_size up to a multiple of make_vocab_size_divisible_by times pg_size and
    # logs a warning on rank 0 when padding was added.
    # Inputs: orig_vocab_size, pg_size, make_vocab_size_divisible_by - ints.
    # Outputs: padded vocab size (int). Exceptions: None explicitly raised. Side effects: logging.
    """Pad vocab size so it is divisible by pg_size * make_vocab_size_divisible_by."""

    multiple = make_vocab_size_divisible_by * pg_size
    after = int(ceil(orig_vocab_size / multiple) * multiple)
    if after != orig_vocab_size:
        log_rank(
            f"[Vocab Size Padding] Padded vocab (size: {orig_vocab_size}) with {after - orig_vocab_size} dummy tokens (new size: {after})",
            logger=logger,
            level=logging.WARNING,
            rank=0,
        )
    return after


def init_random_states(parallel_config: ParallelismArgs, tp_pg: ProcessGroup):
    # [FUNCTION DOC]
    # Behavior: when parallel_config is None or TP mode is ALL_REDUCE, returns RandomStates holding a
    # "tp_synced" state equal to the current RNG state synchronized across tp_pg (so TP ranks apply
    # identical dropout etc.); for REDUCE_SCATTER (sequence parallel) returns empty RandomStates.
    # Inputs: parallel_config - ParallelismArgs or None; tp_pg - ProcessGroup.
    # Outputs: RandomStates. Exceptions: None explicitly raised.
    # Side effects: collective communication across tp_pg for the sync.
    # Get synchronized random states
    if parallel_config is None or parallel_config.tp_mode is TensorParallelLinearMode.ALL_REDUCE:
        random_states = RandomStates(
            {"tp_synced": get_synced_random_state(random_state=get_current_random_state(), pg=tp_pg)}
        )
    else:
        # NOTE: We don't need to sync across TP when using sequence parallel (REDUCE_SCATTER)
        random_states = RandomStates({})
    return random_states


def lr_scheduler_builder(optimizer: Optimizer, lr_scheduler_args: LRSchedulerArgs, total_training_steps: int):
    # [FUNCTION DOC]
    # Behavior: derives lr_decay_steps (given, or total steps minus warmup steps and minus decay start step
    # when those are set) and lr_decay_starting_step (given, else warmup steps, else 0). Builds one lambda
    # per base-optimizer param group bound to that group's initial lr and returns a LambdaLR over the base
    # optimizer.
    # Inputs: optimizer - nanotron Optimizer wrapper; lr_scheduler_args - LRSchedulerArgs;
    # total_training_steps - int.
    # Outputs: torch LambdaLR. Exceptions: AssertionError on lambda/group count mismatch (cannot happen).
    # Side effects: debug logging.
    if lr_scheduler_args.lr_decay_steps is None:
        lr_decay_steps = total_training_steps
        if lr_scheduler_args.lr_warmup_steps is not None:
            lr_decay_steps -= lr_scheduler_args.lr_warmup_steps
        if lr_scheduler_args.lr_decay_starting_step is not None:
            lr_decay_steps -= lr_scheduler_args.lr_decay_starting_step
    else:
        lr_decay_steps = lr_scheduler_args.lr_decay_steps

    if lr_scheduler_args.lr_decay_starting_step is None:
        if lr_scheduler_args.lr_warmup_steps is not None:
            lr_decay_starting_step = lr_scheduler_args.lr_warmup_steps
        else:
            lr_decay_starting_step = 0
    else:
        lr_decay_starting_step = lr_scheduler_args.lr_decay_starting_step

    def lr_lambda(current_step: int, initial_lr: float):
        # [FUNCTION DOC]
        # HELPER: per-step multiplier for LambdaLR.
        # Behavior: if warmup and decay steps are both 0, returns initial_lr unchanged. Otherwise, in order:
        # warmup (step at most warmup steps and a style set): linear ramp from 0 to initial_lr, or
        # "constant" at the global learning_rate; constant phase at initial_lr before the decay start;
        # decay phase (style set and step before start plus decay steps): cosine, linear or "1-sqrt" from
        # initial_lr to min_decay_lr; else min_decay_lr. The result is divided by initial_lr (LambdaLR multiplies).
        # Inputs: current_step - int; initial_lr - the group's base lr.
        # Outputs: float multiplier.
        # Exceptions: ValueError for unknown warmup or decay styles; TypeError if warmup style is set but
        # lr_warmup_steps is None; ZeroDivisionError if initial_lr is 0.
        # Side effects: None.
        # KNOWN ISSUE: the early return when there is no warmup or decay returns initial_lr itself rather than a multiplier of 1.0, so LambdaLR would scale lr by lr.
        # KNOWN ISSUE: constant warmup uses the global learning_rate instead of the group's initial_lr, which is wrong for groups with custom (muP) learning rates.
        """
        current_step: current training step
        initial_lr: the learning rate of a parameter group

        More info on initial_lr:
        And in standard parameterization, lr_lambda only takes a single learning rate.
        But in µTransfer, each parameter has a custom learning rate (custom_lr = lr_scheduler_args.learning_rate * scaling_factor),
        so each parameter group has a custom lr_lambda function.

        LR Scheduling function, it has from 2 up to 4 phases:
        - warmup,
        - optional: constant (if lr_decay_starting_step is set)
        - decay
        - optional: constant (if lr_decay_steps and/or lr_decay_starting_step are set)
        Warmup starts at lr=0 and ends at `lr=lr`
        Then it stays constant at lr if lr_decay_starting_step is set and larger than lr_warmup_steps
        Then it decays until `min_decay_lr` for lr_decay_steps if set, else: (total_training_steps - lr_warmup_steps or lr_decay_starting_step)
        Then it stays constant at min_decay_lr if lr_decay_starting_step is set and total_training_steps is larger)
        """
        # No warmup or decay
        if lr_scheduler_args.lr_warmup_steps == 0 and lr_decay_steps == 0:
            return initial_lr

        # Warmup phase
        elif lr_scheduler_args.lr_warmup_style is not None and current_step <= lr_scheduler_args.lr_warmup_steps:
            if lr_scheduler_args.lr_warmup_style == "linear":
                lmbda = initial_lr * current_step / max(lr_scheduler_args.lr_warmup_steps, 1)
            elif lr_scheduler_args.lr_warmup_style == "constant":
                lmbda = lr_scheduler_args.learning_rate
            else:
                raise ValueError(f"Unknown warmup style {lr_scheduler_args.lr_warmup_style}")

        # Optional constant phase at learning_rate
        elif current_step < lr_decay_starting_step:
            lmbda = initial_lr

        # Decay phase
        elif lr_scheduler_args.lr_decay_style is not None and current_step < lr_decay_starting_step + lr_decay_steps:
            if lr_scheduler_args.lr_decay_style == "cosine":
                lmbda = (
                    lr_scheduler_args.min_decay_lr
                    + (initial_lr - lr_scheduler_args.min_decay_lr)
                    * (1 + math.cos(math.pi * (current_step - lr_decay_starting_step) / lr_decay_steps))
                    / 2
                )
            elif lr_scheduler_args.lr_decay_style == "linear":
                lmbda = (
                    lr_scheduler_args.min_decay_lr
                    + (initial_lr - lr_scheduler_args.min_decay_lr)
                    * (lr_decay_steps - (current_step - lr_decay_starting_step))
                    / lr_decay_steps
                )
            elif lr_scheduler_args.lr_decay_style == "1-sqrt":
                lmbda = lr_scheduler_args.min_decay_lr + (initial_lr - lr_scheduler_args.min_decay_lr) * (
                    1 - math.sqrt((current_step - lr_decay_starting_step) / lr_decay_steps)
                )
            else:
                raise ValueError(f"Unknown decay style {lr_scheduler_args.lr_decay_style}")

        # Optional constant phase at min_decay_lr
        else:
            lmbda = lr_scheduler_args.min_decay_lr

        lmbda /= initial_lr  # Normalization for pytorch
        return lmbda

    def get_lr_lambda_for_param_group(lr: float):
        # [FUNCTION DOC]
        # HELPER: binds lr_lambda to a param group's initial lr.
        # Inputs: lr - float. Outputs: partial function of current_step. Exceptions/Side effects: None.
        return partial(lr_lambda, initial_lr=lr)

    # NOTE: get learning rate scheduler for each param group
    lr_lambdas = []
    for param_group in optimizer.get_base_optimizer().param_groups:
        lr_lambdas.append(get_lr_lambda_for_param_group(lr=param_group["lr"]))

    assert len(lr_lambdas) == len(
        optimizer.get_base_optimizer().param_groups
    ), "Custom learning rate functions dont match the number of param groups"

    log_rank(
        f"[Optimizer Building] There are total {len(lr_lambdas)} custom learning rate function for parameter groups",
        logger=logger,
        level=logging.DEBUG,
    )

    lr_scheduler = LambdaLR(optimizer.get_base_optimizer(), lr_lambda=lr_lambdas)
    return lr_scheduler


def get_custom_weight_decay_for_named_parameters(
    named_parameters: Iterable[Tuple[str, torch.Tensor]],
    model: NanotronModel,
    module_id_to_prefix: Dict[int, str],
    weight_decay: float,
    exclude_named_params: List[str],
) -> List[Dict[str, Any]]:
    # [FUNCTION DOC]
    # Behavior: creates one parameter group per named parameter with weight_decay set to 0 when its name,
    # or for tied parameters its full tied name, matches (from the start) any regex in
    # exclude_named_params, else the given weight_decay. Logs each exclusion (debug) and the total
    # (warning) on rank 0.
    # Inputs: named_parameters - iterable of (name, NanotronParameter); model - unused; module_id_to_prefix
    # - dict for resolving tied names; weight_decay - float; exclude_named_params - list of regex strings or None.
    # Outputs: list of dicts with "named_params" (single-element list) and "weight_decay".
    # Exceptions: NameError/UnboundLocalError (see known issue). Side effects: logging.
    # KNOWN ISSUE: tied_name is only assigned for tied parameters but used for every parameter when exclusions are given, so it is undefined for an untied first parameter, or stale (from a previous tied parameter) for later untied ones.
    """
    Apply weight decay to all parameters except those whose names match regex patterns in exclude_named_params.

    Args:
        named_parameters: Iterable of (name, parameter) tuples
        model: The model instance
        module_id_to_prefix: Mapping from module id to name prefix
        weight_decay: The default weight decay value to apply
        exclude_named_params: List of regex patterns to exclude parameters from weight decay

    Returns:
        List of parameter groups with appropriate weight decay values
    """
    named_param_groups_with_custom_weight_decay = []
    counter_excluded_params = 0
    for name, param in named_parameters:
        # Handle tied parameters: we exclude all tied parameters if one of them is in the exclude list
        if param.is_tied:
            tied_name = param.get_tied_info().get_full_name_from_module_id_to_prefix(
                module_id_to_prefix=module_id_to_prefix
            )

        # Determine weight decay value
        should_exclude = False
        if exclude_named_params is not None:
            should_exclude = any(
                re.match(pattern, name) or re.match(pattern, tied_name) for pattern in exclude_named_params
            )
            if should_exclude:
                counter_excluded_params += 1
                log_rank(f"Excluding parameter {name} from weight decay", logger=logger, level=logging.DEBUG, rank=0)

        current_weight_decay = 0.0 if should_exclude else weight_decay

        # Create parameter group
        named_param_groups_with_custom_weight_decay.append(
            {"named_params": [(name, param)], "weight_decay": current_weight_decay}
        )

    log_rank(
        f"[Optimizer Building] {counter_excluded_params} params excluded from weight decay",
        logger=logger,
        level=logging.WARNING,
        rank=0,
    )
    return named_param_groups_with_custom_weight_decay


def get_custom_lr_for_named_parameters(
    parametrization_method: ParametrizationMethod,
    lr: float,
    named_parameters: Iterable[Tuple[str, torch.Tensor]],
    model: NanotronModel,
) -> List[Dict[str, Any]]:
    # [FUNCTION DOC]
    # Behavior: picks LearningRateForSpectralMup or LearningRateForSP according to the parametrization,
    # builds it from the modules on this pipeline rank and the base lr, and creates one group per
    # parameter with its computed lr.
    # Inputs: parametrization_method - SPECTRAL_MUP or STANDARD; lr - float; named_parameters; model -
    # NanotronModel.
    # Outputs: list of dicts with "named_params" and "lr".
    # Exceptions: AssertionError for other methods or non-float lrs. Side effects: logging.
    """
    Get custom learning rates for parameters based on the parametrization method.

    NOTE: in some paramtrization methods, we use a global learning rate for all parameters,
    in others we use a custom learning rate for each parameter (eg: spectral µTransfer).
    """

    assert parametrization_method in [ParametrizationMethod.SPECTRAL_MUP, ParametrizationMethod.STANDARD]

    lr_mapper_cls = (
        LearningRateForSpectralMup
        if parametrization_method == ParametrizationMethod.SPECTRAL_MUP
        else LearningRateForSP
    )

    log_rank(
        f"[Optimizer Building] Using {lr_mapper_cls.__name__} as learning rate",
        logger=logger,
        level=logging.INFO,
        rank=0,
    )

    # NOTE: since in the case of pipeline parallelism, each rank only has a subset of the model
    # so we only get the parameters that are in the current rank
    learning_rate_mapper = lr_mapper_cls(names_to_modules=model.named_modules_in_pp_rank, lr=lr)

    named_param_groups_with_custom_lr = []
    for (
        name,
        param,
    ) in named_parameters:
        learning_rate = learning_rate_mapper.get_lr(name, param)
        assert isinstance(learning_rate, float), f"Expected a float, got {learning_rate} for parameter {name}"
        named_param_groups_with_custom_lr.append({"named_params": [(name, param)], "lr": learning_rate})

    log_rank(
        f"[Optimizer Building] Creating {len(named_param_groups_with_custom_lr)} param groups with custom learning rates",
        logger=logger,
        level=logging.DEBUG,
    )

    return named_param_groups_with_custom_lr


def merge_named_param_groups(
    named_param_groups_with_lr: List[Dict[str, Any]],
    named_param_groups_with_weight_decay: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    # [FUNCTION DOC]
    # Behavior: zips the lr groups and weight-decay groups (same order), asserts they reference the same
    # named params and merges each pair into a dict with named_params, lr and weight_decay.
    # Inputs: named_param_groups_with_lr, named_param_groups_with_weight_decay - lists of dicts.
    # Outputs: merged list. Exceptions: AssertionError on length or content mismatch. Side effects: None.
    assert len(named_param_groups_with_lr) == len(
        named_param_groups_with_weight_decay
    ), "Named param groups don't match in length"

    named_param_groups = []
    for group_with_lr, group_with_weight_decay in zip(
        named_param_groups_with_lr, named_param_groups_with_weight_decay
    ):
        assert group_with_lr["named_params"] == group_with_weight_decay["named_params"]
        named_param_groups.append(
            {
                "named_params": group_with_lr["named_params"],
                "lr": group_with_lr["lr"],
                "weight_decay": group_with_weight_decay["weight_decay"],
            }
        )

    return named_param_groups

def init_optimizer_and_grad_accumulator(
    parametrization_method: ParametrizationMethod,
    model: Union[nn.Module, DistributedDataParallel],
    optimizer_args: OptimizerArgs,
    parallel_context: ParallelContext,
) -> Tuple[BaseOptimizer, GradientAccumulator]:
    # [FUNCTION DOC]
    # Behavior: unwraps DDP; maps module ids to name prefixes; collects named params with correct tied
    # names; builds per-parameter groups with custom lr and weight decay. Defines a base builder producing a
    # NamedOptimizer around AdamW (lr, weight decay, eps, betas, fused from config) or SGD. If
    # accumulate_grad_in_fp32, wraps with OptimizerFromGradientAccumulator using an FP32GradientAccumulator
    # (captured into grad_accumulator). If zero_stage is above 0, wraps everything in
    # ZeroDistributedOptimizer over the DP group and sanity-checks that the first optimizer param shares
    # storage with the model param; otherwise builds directly. With ZeRO and fp32 accumulation, tells the
    # accumulator which parameter slices this DP rank owns. If the model is DDP and fp32 accumulation is on,
    # registers a comm hook (FP32GradBucketManager over the DP x CP group, averaging reduce, reduce-scatter
    # when ZeRO is used).
    # Inputs: parametrization_method; model - nn.Module or DDP; optimizer_args - OptimizerArgs;
    # parallel_context.
    # Outputs: tuple (optimizer, gradient accumulator or None).
    # Exceptions: ValueError for unsupported optimizer names; AssertionError on sanity checks or wrong
    # accumulator/optimizer types.
    # Side effects: logging; allocates fp32 gradient buffers and optimizer state; registers a DDP comm hook.
    log_rank(
        "Building Optimizer and Gradient Accumulator and Learning Rate Scheduler",
        logger=logger,
        level=logging.INFO,
        rank=0,
        is_separator=True,
    )  # noqa
    # Unwrap DDP
    unwrapped_model: NanotronModel = model.module if isinstance(model, DistributedDataParallel) else model

    module_id_to_prefix = {id(module): f"{module_name}." for module_name, module in unwrapped_model.named_modules()}
    # Fix the root_model
    module_id_to_prefix[id(unwrapped_model)] = ""

    named_parameters = list(unwrapped_model.get_named_params_with_correct_tied())

    named_param_groups_with_lr = get_custom_lr_for_named_parameters(
        parametrization_method=parametrization_method,
        named_parameters=named_parameters,
        model=unwrapped_model,
        lr=optimizer_args.learning_rate_scheduler.learning_rate,
    )
    named_param_groups_with_weight_decay = get_custom_weight_decay_for_named_parameters(
        named_parameters=named_parameters,
        model=unwrapped_model,
        module_id_to_prefix=module_id_to_prefix,
        weight_decay=optimizer_args.weight_decay,
        exclude_named_params=optimizer_args.weight_decay_exclude_named_params,
    )

    named_param_groups = merge_named_param_groups(named_param_groups_with_lr, named_param_groups_with_weight_decay)

    # Basic optimizer builder
    def basic_optimizer_builder(named_param_groups):
        # [FUNCTION DOC]
        # HELPER: builds a NamedOptimizer with AdamW or SGD from optimizer_args.
        # Inputs: named_param_groups - list of group dicts. Outputs: NamedOptimizer.
        # Exceptions: ValueError for unsupported optimizer names. Side effects: allocates optimizer state.
        optimizer = None

        if optimizer_args.optimizer_factory.name == "adamW":

            def optimizer(param_groups):
                # [FUNCTION DOC]
                # HELPER: AdamW factory closure. Inputs: param_groups. Outputs: torch AdamW with config lr, weight decay, eps, betas, fused.
                # Exceptions/Side effects: None beyond torch.
                return torch.optim.AdamW(
                    param_groups,
                    lr=optimizer_args.learning_rate_scheduler.learning_rate,
                    weight_decay=optimizer_args.weight_decay,
                    eps=optimizer_args.optimizer_factory.adam_eps,
                    betas=(optimizer_args.optimizer_factory.adam_beta1, optimizer_args.optimizer_factory.adam_beta2),
                    fused=optimizer_args.optimizer_factory.torch_adam_is_fused,
                )

        elif optimizer_args.optimizer_factory.name == "sgd":

            def optimizer(param_groups):
                # [FUNCTION DOC]
                # HELPER: SGD factory closure. Inputs: param_groups. Outputs: torch SGD with config lr and weight decay.
                # Exceptions/Side effects: None beyond torch.
                return torch.optim.SGD(
                    param_groups,
                    lr=optimizer_args.learning_rate_scheduler.learning_rate,
                    weight_decay=optimizer_args.weight_decay,
                )

        else:
            raise ValueError(f"Optimizer {optimizer_args.optimizer_factory.name} is not supported")

        return NamedOptimizer(
            named_params_or_groups=named_param_groups,
            optimizer_builder=optimizer,
        )

    optimizer_builder = basic_optimizer_builder

    # Gradient accumulator builder
    grad_accumulator: Optional[GradientAccumulator] = None
    if optimizer_args.accumulate_grad_in_fp32:

        def grad_optimizer_builder(named_param_groups):
            # [FUNCTION DOC]
            # HELPER: builds an OptimizerFromGradientAccumulator (FP32 accumulation, grad buckets over all named
            # parameters) on top of basic_optimizer_builder and stores its accumulator in the enclosing grad_accumulator.
            # Inputs: named_param_groups. Outputs: OptimizerFromGradientAccumulator.
            # Exceptions: as basic_optimizer_builder. Side effects: sets the enclosing variable; allocates fp32 buffers.
            result = OptimizerFromGradientAccumulator(
                gradient_accumulator_builder=lambda named_params: FP32GradientAccumulator(
                    named_parameters=named_params,
                    grad_buckets_named_params=named_parameters,
                ),
                named_params_or_groups=named_param_groups,
                optimizer_builder=basic_optimizer_builder,
            )

            nonlocal grad_accumulator
            grad_accumulator = result.gradient_accumulator

            return result

        optimizer_builder = grad_optimizer_builder

    if optimizer_args.zero_stage > 0:
        # Build optimizer
        optimizer = ZeroDistributedOptimizer(
            named_params_or_groups=named_param_groups,
            optimizer_builder=optimizer_builder,
            dp_pg=parallel_context.dp_pg,
        )

        # SANITY CHECK: assert that optimizer's named_params point to model's params (check only the first one)
        if (
            len(optimizer.zero_named_param_groups) > 0
            and len(optimizer.zero_named_param_groups[0]["named_params"]) > 0
        ):
            optim_model_param_name, optim_model_param = optimizer.zero_named_param_groups[0]["named_params"][0]
            if isinstance(model, DistributedDataParallel):
                optim_model_param_name = f"module.{optim_model_param_name}"
            param = model.get_parameter(optim_model_param_name)
            assert param.data_ptr() == optim_model_param.data_ptr()
    else:
        # Build optimizer
        optimizer = optimizer_builder(named_param_groups)

    if grad_accumulator is not None and optimizer_args.zero_stage > 0:
        # There's a way to only require to reduce_scatter the gradients instead of all_reducing
        # In order to do so I need to pass which segments of each parameter should be reduced on which dp rank.
        assert isinstance(optimizer, ZeroDistributedOptimizer)
        param_name_to_dp_rank_offsets = optimizer.param_name_to_dp_rank_offsets

        assert isinstance(grad_accumulator, FP32GradientAccumulator)
        grad_accumulator.assign_param_offsets(
            dp_rank=dist.get_rank(parallel_context.dp_pg),
            param_name_to_offsets=param_name_to_dp_rank_offsets,
        )

    # Register DDP hook to make fp32 grad accumulation work
    if isinstance(model, DistributedDataParallel) and grad_accumulator is not None:
        assert isinstance(grad_accumulator, FP32GradientAccumulator)
        model.register_comm_hook(
            state=FP32GradBucketManager(
                dp_cp_pg=parallel_context.dp_cp_pg,
                accumulator=grad_accumulator,
                param_id_to_name={
                    id(param): param.get_tied_info().get_full_name_from_module_id_to_prefix(
                        module_id_to_prefix=module_id_to_prefix
                    )
                    if param.is_tied # a tied param exists only once physically in memory
                    else name
                    for name, param in unwrapped_model.named_parameters()
                },
            ),
            hook=get_fp32_accum_hook(
                reduce_scatter=optimizer.inherit_from(ZeroDistributedOptimizer), reduce_op=dist.ReduceOp.AVG
            ),
        )

    return optimizer, grad_accumulator


def test_equal_dict(first: Dict, second: Dict, sub_paths: Optional[List[str]] = None) -> None:
    # [FUNCTION DOC]
    # HELPER: test utility for comparing nested state dicts.
    # Behavior: asserts identical key sets, then recursively compares sub-dicts, compares tensors exactly
    # (zero tolerance) and other values by equality, reporting the dotted path on failure.
    # Inputs: first, second - dicts; sub_paths - path prefix list (default None meaning empty).
    # Outputs: None. Exceptions: AssertionError on any mismatch. Side effects: None.
    """Raise if doesn't match."""

    if sub_paths is None:
        sub_paths = []

    first_keys = set(first.keys())
    second_keys = set(second.keys())
    assert first_keys == second_keys, f"Keys don't match.\nFirst: {first_keys}\nSecond: {second_keys}"
    for key in first_keys:
        first_elt = first[key]
        second_elt = second[key]

        if isinstance(first_elt, dict):
            assert isinstance(second_elt, dict), f"{first_elt} doesn't match {second_elt}"
            test_equal_dict(first_elt, second_elt, sub_paths=sub_paths + [str(key)])
        elif isinstance(first_elt, torch.Tensor):
            assert isinstance(second_elt, torch.Tensor), f"{first_elt} doesn't match {second_elt}"
            torch.testing.assert_close(
                first_elt,
                second_elt,
                atol=0.0,
                rtol=0.0,
                msg=lambda msg: f"tensor at {'.'.join(sub_paths + [str(key)])} don't match.\nCur: {first_elt}\nRef: {second_elt}\n{msg}",
            )
        else:
            assert (
                first_elt == second_elt
            ), f"{first_elt} doesn't match {second_elt} at key {'.'.join(sub_paths + [str(key)])}"


def get_profiler(config: Config):
    # [FUNCTION DOC]
    # Behavior: on global rank 0 with a profiler config, returns a torch profiler (CPU and CUDA, schedule
    # wait/warmup/active/repeat/skip_first, shapes, memory, stack options) exporting TensorBoard traces to
    # export path / timestamp-run if an export path is set; otherwise returns a null context manager.
    # Inputs: config - Config. Outputs: profiler or nullcontext. Exceptions: None explicitly raised.
    # Side effects: None until entered (then writes traces).
    if config.profiler is not None and dist.get_rank() == 0:
        if config.profiler.profiler_export_path is not None:
            on_trace_ready = tensorboard_trace_handler(
                config.profiler.profiler_export_path / datetime.now().strftime("%Y%m%d-%H%M%S-" + config.general.run)
            )
        else:
            on_trace_ready = None

        prof = profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            schedule=torch.profiler.schedule(
                wait=config.profiler.wait,
                warmup=config.profiler.warmup,
                active=config.profiler.active,
                repeat=config.profiler.repeat,
                skip_first=config.profiler.skip_first,
            ),
            on_trace_ready=on_trace_ready,
            record_shapes=config.profiler.record_shapes,
            profile_memory=config.profiler.profile_memory,
            with_stack=config.profiler.with_stack,
        )
    else:
        prof = contextlib.nullcontext()
    return prof


def get_all_comps(n: int) -> List[List[List[int]]]:
    # [FUNCTION DOC]
    # HELPER: builds the pairing schedule for test_all_pair_to_pair.
    # Behavior: for n a power of two, generates rounds of disjoint rank pairs covering all pairs by
    # reshaping the rank range into rows of width d (doubling d from 1), rolling odd rows by r for each
    # r below d, and transposing.
    # Inputs: n - int power of two. Outputs: nested list (rounds, pairs, 2) of ranks.
    # Exceptions: ValueError if n is not a power of two. Side effects: None.
    """Return a 3D numpy array with a series of pairs to test latency/bandwidth between:
        This basically make a square matrix from the triangle of pair-to-pair comparisons


    [[[0 1]
    [2 3]]

    [[0 2]
    [1 3]]

    [[0 3]
    [1 2]]]
    """
    # n: power of two
    if not ((n & (n - 1) == 0) and n != 0):
        # every power of 2 has exactly 1 bit set to 1 (the bit in that number's log base-2 index).
        # So when subtracting 1 from it, that bit flips to 0 and all preceding bits flip to 1.
        # That makes these 2 numbers the inverse of each other so when AND-ing them, we will get 0 as the result
        raise ValueError("n must be a power of two")

    def op(lst, d=4, r=1):
        # [FUNCTION DOC]
        # HELPER: one reshape-roll-transpose step. Inputs: lst - numpy array; d - row width (default 4); r -
        # roll amount (default 1). Outputs: flattened numpy array. Exceptions/Side effects: None (operates on a reshaped view).
        lst = lst.reshape(-1, d)
        lst[1::2] = np.roll(lst[1::2], r, axis=1)
        return lst.T.reshape(-1)

    x = np.array(list(range(n)))
    comps = []
    d = 1
    while d < n:
        for r in range(d):
            comps.append(op(x, d=d, r=r).copy())
        d *= 2
    ret = np.stack(comps)
    return ret.reshape(ret.shape[0], -1, 2).tolist()


def test_all_pair_to_pair(
    parallel_context: ParallelContext, throughput_size: int, throughput_iters: int, only_node_to_node: bool = True
):
    # [FUNCTION DOC]
    # Behavior: for each round of rank pairs from get_all_comps(world size) and each pair, all ranks hit a
    # world barrier; the two ranks of the pair (only pairs whose ranks are both multiples of 8 when
    # only_node_to_node) exchange a uint8 CUDA tensor of throughput_size bytes throughput_iters times
    # with send/recv, timing each iteration with CUDA syncs, then log the last iteration's throughput in Gbps.
    # Inputs: parallel_context; throughput_size - bytes; throughput_iters - iterations; only_node_to_node -
    # default True.
    # Outputs: None. Exceptions: None explicitly raised (NameError if throughput_iters is 0).
    # Side effects: barriers and P2P communication; CUDA allocation and cache emptying; logging.
    # KNOWN ISSUE: only the last iteration's duration is used, and the multiple-of-8 rule hard-codes 8 GPUs per node.
    """Test all pair-to-pair GPUs throughput

    Args:
        parallel_context: ParallelContext
        throughput_size: size of the tensor to send
        throughput_iters: number of warm-up iterations before testing the throughput
        only_node_to_node: if True, only test node-to-node throughput
    """
    comparisons = get_all_comps(parallel_context.world_pg.size())
    wr = dist.get_rank(parallel_context.world_pg)
    log_rank(
        f"[TEST] Testing throughput between {comparisons}",
        logger=logger,
        level=logging.WARNING,
        group=parallel_context.world_pg,
        rank=0,
    )
    for j, comp in enumerate(comparisons):
        dist.barrier(group=parallel_context.world_pg)
        for i, (a, b) in enumerate(comp):
            dist.barrier(group=parallel_context.world_pg)
            if wr not in [a, b]:
                continue
            if only_node_to_node and (a % 8 != 0 or b % 8 != 0):
                # We only check node-to-node throughput
                continue
            test_tensor = torch.zeros((int(throughput_size),), dtype=torch.uint8, device=torch.device("cuda"))
            for k in range(throughput_iters):
                pre = time.perf_counter()
                torch.cuda.synchronize()
                if wr == a:
                    dist.send(test_tensor, b, group=parallel_context.world_pg, tag=i + k)
                elif wr == b:
                    dist.recv(test_tensor, a, group=parallel_context.world_pg, tag=i + k)
                torch.cuda.synchronize()
                duration = time.perf_counter() - pre
            del test_tensor
            gc.collect()
            torch.cuda.empty_cache()
            tput = (float(throughput_size) / duration) * 8  # *8 for gigabits/second
            log_rank(
                f"[TEST] {j, i, wr} Results throughput from {a} to {b}: {tput/1e9:.4f} Gbps",
                logger=logger,
                level=logging.WARNING,
                group=parallel_context.world_pg,
                rank=None,
            )
    log_rank(
        "[TEST] All comparisons done",
        logger=logger,
        level=logging.WARNING,
        group=parallel_context.world_pg,
        rank=0,
    )


def create_table_log(
    config: Config,
    parallel_context: ParallelContext,
    model_tflops,
    hardware_tflops,
    tokens_per_sec,
    bandwidth,
    num_params,
    slurm_job_id,
):
    # [FUNCTION DOC]
    # HELPER: builds the list of LogItem rows for the benchmark table (job id, run name, nodes, batch
    # sizes, TFLOPs, tokens/s/GPU, peak memory, parallelism settings, model shape, dtype, ZeRO stage,
    # parameter counts).
    # Inputs: config; parallel_context; model_tflops; hardware_tflops; tokens_per_sec; bandwidth;
    # num_params - dict with "total" and "local"; slurm_job_id.
    # Outputs: list of LogItem. Exceptions: AttributeError if config lacks the fields. Side effects: reads CUDA memory stats.
    # KNOWN ISSUE: bandwidth is accepted but never included in the table.
    return [
        LogItem("job_id", slurm_job_id, "s"),
        LogItem("name", config.general.run, "s"),
        LogItem("nodes", math.ceil(parallel_context.world_pg.size() / torch.cuda.device_count()), "d"),
        LogItem("seq_len", config.tokens.sequence_length, "d"),
        LogItem("mbs", config.tokens.micro_batch_size, "d"),
        LogItem("batch_accum", config.tokens.batch_accumulation_per_replica, "d"),
        LogItem("gbs", config.global_batch_size, "d"),
        LogItem("mTFLOPs", model_tflops, ".2f"),
        LogItem("hTFLOPs", hardware_tflops, ".2f"),
        LogItem("tok/s/gpu", tokens_per_sec / parallel_context.world_pg.size(), ".2f"),
        LogItem("Mem Alloc (GB)", torch.cuda.max_memory_allocated() / 1024**3, ".2f"),
        LogItem("Mem Res (GB)", torch.cuda.max_memory_reserved() / 1024**3, ".2f"),
        # Important config columns
        LogItem("dp", config.parallelism.dp, "d"),
        LogItem("pp", config.parallelism.pp, "d"),
        LogItem("tp", config.parallelism.tp, "d"),
        LogItem("cp", config.parallelism.context_parallel_size, "d"),
        LogItem("ep", config.parallelism.expert_parallel_size, "d"),
        LogItem("pp_engine", str(config.parallelism.pp_engine), "s"),
        LogItem("tp_mode", config.parallelism.tp_mode, "s"),
        LogItem("tp_async_comm", str(config.parallelism.tp_linear_async_communication), "s"),
        LogItem("recompute_layer", str(config.parallelism.recompute_layer), "s"),
        LogItem("hidden_size", config.model.model_config.hidden_size, "d"),
        LogItem("hidden_act", config.model.model_config.hidden_act, "s"),
        LogItem("num_layers", config.model.model_config.num_hidden_layers, "d"),
        LogItem("num_heads", config.model.model_config.num_attention_heads, "d"),
        LogItem("num_kv_heads", config.model.model_config.num_key_value_heads, "d"),
        LogItem("max_pos", config.model.model_config.max_position_embeddings, "d"),
        LogItem("vocab_size", config.model.model_config.vocab_size, "d"),
        LogItem("tie_word_embeddings", str(config.model.model_config.tie_word_embeddings), "s"),
        LogItem("dtype", str(config.model.dtype), "s"),
        LogItem("zero_stage", config.optimizer.zero_stage, "d"),
        LogItem("ddp_bucket_cap_mb", config.model.ddp_bucket_cap_mb, "d"),
        LogItem("accumulate_grad_in_fp32", str(config.optimizer.accumulate_grad_in_fp32), "s"),
        # Params
        LogItem("Total Params", num_params["total"], "human_format"),
        LogItem("Local Params", num_params["local"], "human_format"),
    ]


def get_formatted_value(item):
    # [FUNCTION DOC]
    # HELPER: formats a LogItem value with human_format or its format spec.
    # Inputs: item - LogItem. Outputs: str. Exceptions: ValueError for incompatible format specs. Side effects: None.
    if item.log_format == "human_format":
        return human_format(item.scalar_value)
    return f"{item.scalar_value:{item.log_format}}"


def create_table_output(table_log, column_widths):
    # [FUNCTION DOC]
    # HELPER: renders LogItems as a three-line markdown-style table (header, dashes, values) with given column widths.
    # Inputs: table_log - list of LogItem; column_widths - list of ints. Outputs: str. Exceptions/Side effects: None.
    header_row = "| " + " | ".join([item.tag.ljust(width) for item, width in zip(table_log, column_widths)]) + " |"
    separator_row = "| " + " | ".join(["-" * width for width in column_widths]) + " |"
    data_row = (
        "| "
        + " | ".join([get_formatted_value(item).ljust(width) for item, width in zip(table_log, column_widths)])
        + " |"
    )
    return f"{header_row}\n{separator_row}\n{data_row}"


def write_to_csv(csv_filename, table_log, model_tflops, slurm_job_id):
    # [FUNCTION DOC]
    # Behavior: if csv_filename is empty, warns and returns. Creates the parent directory, formats header
    # and row, and tries up to 10 times (100 ms apart) to open the file in append mode with a non-blocking
    # exclusive flock; writes the header if the file is empty, appends the row, flushes and fsyncs, then
    # unlocks. Logs an error if the lock could not be acquired; any other exception is logged and swallowed.
    # Inputs: csv_filename - path; table_log - list of LogItem; model_tflops, slurm_job_id - unused.
    # Outputs: None. Exceptions: none propagate (all caught).
    # Side effects: file I/O with locking; logging; sleeps.
    # KNOWN ISSUE: model_tflops and slurm_job_id are unused, and the "Successfully wrote" message is logged in the finally block even when writing failed.
    """Write benchmark results to a CSV file with file locking using fcntl."""
    import fcntl

    try:
        # Check if csv_filename is valid
        if not csv_filename:
            logger.warning("No benchmark CSV path specified - skipping CSV output")
            return

        # Create output directory if needed
        csv_dir = os.path.dirname(csv_filename)
        if csv_dir:  # Only try to create directory if path has a directory component
            os.makedirs(csv_dir, exist_ok=True)

        # Format row data
        header = [item.tag for item in table_log]
        row = [get_formatted_value(item) for item in table_log]

        # Use fcntl for file locking
        max_attempts = 10
        attempt = 0
        log_rank(
            f"Attempting to write benchmark results to CSV file: {csv_filename}",
            logger=logger,
            level=logging.INFO,
        )
        while attempt < max_attempts:
            try:
                with open(csv_filename, mode="a+", newline="") as f:
                    # Get exclusive lock
                    fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    log_rank(
                        f"Acquired lock for CSV file: {csv_filename}",
                        logger=logger,
                        level=logging.INFO,
                    )
                    try:
                        # Check if file is empty/new
                        f.seek(0)
                        first_char = f.read(1)
                        if not first_char:  # Empty file
                            writer = csv.writer(f)
                            writer.writerow(header)

                        # Go to end of file for append
                        f.seek(0, os.SEEK_END)
                        writer = csv.writer(f)
                        writer.writerow(row)
                        f.flush()
                        os.fsync(f.fileno())
                        break
                    finally:
                        # Release lock
                        fcntl.flock(f.fileno(), fcntl.LOCK_UN)
                        log_rank(
                            f"Successfully wrote to CSV file: {csv_filename}. Releasing lock...",
                            logger=logger,
                            level=logging.INFO,
                        )
            except BlockingIOError:
                # Another process has the lock, wait and retry
                log_rank(
                    f"Another process has the lock for CSV file: {csv_filename}, waiting and retrying attempt {attempt + 1} of {max_attempts}...",
                    logger=logger,
                    level=logging.INFO,
                )
                attempt += 1
                time.sleep(0.1)  # Wait 100ms before retrying

        if attempt == max_attempts:
            logger.error(f"Failed to acquire lock for {csv_filename} after {max_attempts} attempts")

    except Exception as e:
        logger.error(f"Unexpected error writing to {csv_filename}: {str(e)}")
        return


def log_throughput(
    config: Config,
    parallel_context: ParallelContext,
    model_tflops=0,
    hardware_tflops=0,
    tokens_per_sec=0,
    bandwidth=0,
    num_params={"total": 0, "local": 0},
):
    # [FUNCTION DOC]
    # Behavior: builds the benchmark table (SLURM_JOB_ID or "N/A"), logs it on rank 0, writes it to
    # config.general.benchmark_csv_path on world rank 0, then barriers on the world group.
    # Inputs: config; parallel_context; model_tflops, hardware_tflops, tokens_per_sec, bandwidth - numbers
    # (default 0); num_params - dict (default totals of 0).
    # Outputs: None. Exceptions: as create_table_log. Side effects: logging; CSV write; world barrier.
    # KNOWN ISSUE: num_params uses a mutable dictionary default (harmless here since it is not mutated).
    slurm_job_id = os.environ.get("SLURM_JOB_ID", "N/A")

    table_log = create_table_log(
        config, parallel_context, model_tflops, hardware_tflops, tokens_per_sec, bandwidth, num_params, slurm_job_id
    )
    column_widths = [max(len(item.tag), len(get_formatted_value(item))) for item in table_log]
    table_output = create_table_output(table_log, column_widths)

    log_rank(
        table_output,
        logger=logger,
        level=logging.INFO,
        rank=0,
    )

    if dist.get_rank(parallel_context.world_pg) == 0:
        write_to_csv(config.general.benchmark_csv_path, table_log, model_tflops, slurm_job_id)
    dist.barrier(group=parallel_context.world_pg)


def compute_remain_train_steps_of_a_data_stage_from_ckp(
    stage: DatasetStageArgs, config: Config, metadata: TrainingMetadata
) -> int:
    # [FUNCTION DOC]
    # Behavior: the stage's end is the global train_steps if it is the last stage (by start step),
    # otherwise the start step of a later stage. Returns 0 if the checkpoint's last step is beyond this
    # stage's start; else returns end minus (last step if resuming, otherwise stage start) plus 1.
    # Inputs: stage - DatasetStageArgs; config - Config; metadata - TrainingMetadata.
    # Outputs: int number of remaining steps. Exceptions: None explicitly raised. Side effects: None.
    # KNOWN ISSUE: resuming in the middle of a stage (last step beyond its start but before the next stage) returns 0 instead of the remaining steps.
    # KNOWN ISSUE: the next stage is the first later stage in config order, not necessarily the nearest one if stages are unsorted.
    def is_last_stage():
        # [FUNCTION DOC]
        # HELPER: True if stage has the largest start step among config.data_stages.
        # Inputs/Outputs: bool. Exceptions/Side effects: None.
        sorted_stages = sorted(config.data_stages, key=lambda x: x.start_training_step)
        return sorted_stages[-1].start_training_step == stage.start_training_step

    def is_resume_from_training():
        # [FUNCTION DOC]
        # HELPER: True if the metadata's last train step is positive. Outputs: bool. Exceptions/Side effects: None.
        return metadata.last_train_step > 0

    if is_last_stage() is True:
        total_train_steps = config.tokens.train_steps
    else:
        next_stage = next((s for s in config.data_stages if s.start_training_step > stage.start_training_step), None)
        total_train_steps = next_stage.start_training_step

    if metadata.last_train_step > stage.start_training_step:
        # NOTE: if the last_train_step is larger than the start_training_step of the current stage,
        # it means that the training has already passed this stage
        # so there is no remaining steps
        return 0
    else:
        last_train_steps = metadata.last_train_step if is_resume_from_training() else stage.start_training_step
        return total_train_steps - last_train_steps + 1


def get_consumed_train_samples_of_a_data_stage_from_ckp(
    stage: DatasetStageArgs, metadata: TrainingMetadata
) -> Optional[int]:
    # [FUNCTION DOC]
    # Behavior: finds the metadata stage with the same start step; if none, asserts training has not
    # reached this stage yet and appends a new DataStageMetadata with 0 consumed samples to the metadata.
    # Returns that stage's consumed samples and per-dataset-folder consumed tokens.
    # Inputs: stage - DatasetStageArgs; metadata - TrainingMetadata.
    # Outputs: tuple (consumed_train_samples int, consumed_tokens_per_dataset_folder dict).
    # Exceptions: AssertionError if the stage is missing but already passed.
    # Side effects: may mutate metadata.data_stages.
    # KNOWN ISSUE: annotated as returning an optional int but returns a tuple.
    # KNOWN ISSUE: metadata.data_stages is iterated before the None check, so a None list raises TypeError first.
    start_training_step = stage.start_training_step

    # find the stage in the metadata using the start_training_step
    actual_stage = next(
        (s for s in metadata.data_stages if s.start_training_step == start_training_step),
        None,
    )
    if actual_stage is None:
        # If stage not found in metadata, ensure we haven't passed this stage's start step
        assert (
            metadata.last_train_step < stage.start_training_step
        ), f"Stage {stage.name} starting at step {stage.start_training_step} not found in checkpoint metadata, but last_train_step ({metadata.last_train_step}) >= start_training_step"
        consumed_train_samples = 0

        # Add new stage to metadata
        if metadata.data_stages is None:
            metadata.data_stages = []

        actual_stage = DataStageMetadata(
            name=stage.name,
            start_training_step=stage.start_training_step,
            consumed_train_samples=consumed_train_samples,
        )
        metadata.data_stages.append(actual_stage)

    return actual_stage.consumed_train_samples, actual_stage.consumed_tokens_per_dataset_folder
