# [MODULE DOC]
# Purpose: pytest test (currently skipped) for NanotronModel named_modules_in_pp_rank using a tiny Llama.
# Main contents: test_get_named_modules_in_pp_rank and its per-rank body.
import pytest
import torch
import torch.distributed as dist
from helpers.utils import init_distributed, rerun_if_address_is_in_use
from nanotron.config import Config, ModelArgs, RandomInit
from nanotron.parallel import ParallelContext
from nanotron.parallel.pipeline_parallel.block import PipelineBlock
from torch import nn

from tests.helpers.llama_helper import TINY_LLAMA_CONFIG, create_llama_from_config, get_llama_training_config


@pytest.mark.parametrize("tp,dp,pp", [(1, 1, 1), (2, 2, 2)])
@pytest.mark.skip
@rerun_if_address_is_in_use()
def test_get_named_modules_in_pp_rank(tp: int, dp: int, pp: int):
    # [FUNCTION DOC]
    # Behavior: builds a training Config from TINY_LLAMA_CONFIG with random init (std 1.0) and runs the body on tp*dp*pp processes.
    # Inputs: tp, dp, pp, parametrized (1,1,1) and (2,2,2).
    # Outputs: None.
    # Exceptions: AssertionError on failure (test is marked skip).
    # Side effects: process spawning.
    model_args = ModelArgs(init_method=RandomInit(std=1.0), model_config=TINY_LLAMA_CONFIG)
    config = get_llama_training_config(model_args)

    init_distributed(tp=tp, dp=dp, pp=pp)(_test_get_named_modules_in_pp_rank)(config=config)


def _test_get_named_modules_in_pp_rank(
    parallel_context: ParallelContext,
    config: Config,
):
    # [FUNCTION DOC]
    # Behavior: creates the Llama on CUDA, randomly initializes it, collects PipelineBlocks whose rank differs from the current pipeline rank, then checks that every entry of named_modules_in_pp_rank is a module not among those other-rank blocks.
    # Inputs: parallel_context; config, Config.
    # Outputs: None.
    # Exceptions: AssertionError on failure.
    # Side effects: GPU allocation, collectives during build and init.
    model = create_llama_from_config(
        model_config=config.model.model_config,
        device=torch.device("cuda"),
        parallel_context=parallel_context,
    )
    model.init_model_randomly(config=config)

    modules_that_not_in_current_pp_rank = {}
    current_pp_rank = dist.get_rank(group=parallel_context.pp_pg)
    for name, module in model.named_modules():
        if isinstance(module, PipelineBlock) and module.rank != current_pp_rank:
            modules_that_not_in_current_pp_rank[name] = module

    named_modules_in_pp_rank = model.named_modules_in_pp_rank

    for name, module in named_modules_in_pp_rank.items():
        # NOTE: if a module is in the current rank, we expect it to be an initialized module
        # not PipelineBlock
        assert isinstance(module, nn.Module)
        assert name not in modules_that_not_in_current_pp_rank
