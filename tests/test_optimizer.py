# [MODULE DOC]
# Purpose: distributed pytest test (currently skipped) for get_custom_lr_for_named_parameters under standard and spectral muP parametrizations on a tiny Llama.
# Main contents: test_get_custom_lr and _test_get_custom_lr.
from typing import Union

import pytest
import torch
from helpers.utils import init_distributed, rerun_if_address_is_in_use
from nanotron.config import ModelArgs, RandomInit, SpectralMupInit
from nanotron.helpers import get_custom_lr_for_named_parameters
from nanotron.parallel import ParallelContext
from nanotron.scaling.parametrization import ParametrizationMethod

from tests.helpers.llama_helper import TINY_LLAMA_CONFIG, create_llama_from_config, get_llama_training_config


@pytest.mark.parametrize("tp,dp,pp", [(1, 1, 1), (2, 1, 1), (1, 1, 2), (2, 1, 2)])
@pytest.mark.parametrize(
    "parametrization_method", [ParametrizationMethod.STANDARD, ParametrizationMethod.SPECTRAL_MUP]
)
@pytest.mark.skip
@rerun_if_address_is_in_use()
def test_get_custom_lr(tp: int, dp: int, pp: int, parametrization_method: ParametrizationMethod):
    # [FUNCTION DOC]
    # Behavior: chooses RandomInit(std 1.0) for STANDARD or SpectralMupInit for SPECTRAL_MUP and spawns ranks running _test_get_custom_lr with lr 1e-3.
    # Inputs: tp, dp, pp layouts; parametrization_method.
    # Outputs: None.
    # Exceptions: UnboundLocalError if an unknown method were passed (init_method undefined).
    # Side effects: process spawning.
    LR = 1e-3

    if parametrization_method == ParametrizationMethod.STANDARD:
        init_method = RandomInit(std=1.0)
    elif parametrization_method == ParametrizationMethod.SPECTRAL_MUP:
        init_method = SpectralMupInit(use_mup=True)

    init_distributed(tp=tp, dp=dp, pp=pp)(_test_get_custom_lr)(
        lr=LR,
        init_method=init_method,
        parametrization_method=parametrization_method,
    )


def _test_get_custom_lr(
    parallel_context: ParallelContext,
    lr: float,
    init_method: Union[RandomInit, SpectralMupInit],
    parametrization_method: ParametrizationMethod,
):
    # [FUNCTION DOC]
    # Behavior: builds and randomly initializes the Llama with the parametrization, collects named parameters with correct tie handling (returns early on ranks with none), computes per-parameter learning rate groups and checks one group per parameter, float learning rates, list-valued named_params, and that all learning rates equal the base one if and only if the method is STANDARD.
    # Inputs: parallel_context; lr, float; init_method; parametrization_method.
    # Outputs: None.
    # Exceptions: AssertionError on failure.
    # Side effects: GPU allocation, collectives during build.
    # KNOWN ISSUE: returns early on ranks without parameters without destroying the parallel context, and the context is never destroyed on other ranks either.
    model_args = ModelArgs(init_method=init_method, model_config=TINY_LLAMA_CONFIG)
    config = get_llama_training_config(model_args)
    llama = create_llama_from_config(
        model_config=TINY_LLAMA_CONFIG,
        device=torch.device("cuda"),
        parallel_context=parallel_context,
    )
    llama.init_model_randomly(config=config, init_method=parametrization_method)
    named_parameters = list(llama.get_named_params_with_correct_tied())

    if len(named_parameters) == 0:
        # NOTE: some pp ranks don't have any parameters
        return

    named_param_groups = get_custom_lr_for_named_parameters(
        parametrization_method=parametrization_method, lr=lr, named_parameters=named_parameters, model=llama
    )

    assert len(named_param_groups) == len(named_parameters)
    assert all(isinstance(named_param_group["lr"], float) for named_param_group in named_param_groups)
    assert all(isinstance(named_param_group["named_params"], list) for named_param_group in named_param_groups)

    is_all_lr_the_same = parametrization_method == ParametrizationMethod.STANDARD
    assert all(named_param_group["lr"] == lr for named_param_group in named_param_groups) is is_all_lr_the_same
