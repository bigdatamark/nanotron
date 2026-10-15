# [MODULE DOC]
# Purpose: pytest test checking that nanotron's parametrization-aware weight initialization (spectral muP) produces parameters with the expected standard deviations on a tiny Llama model.
# Main contents: test_parametrization (pytest entry, currently skipped) and _test_parametrization (per-rank body launched via init_distributed).
# Fits in nanotron: exercises nanotron.scaling.parametrization and the model's init_model_randomly path under tensor parallelism.
# Import-time side effects: None beyond imports of test helpers.
import math
from typing import Union

import pytest
import torch
from helpers.utils import init_distributed, rerun_if_address_is_in_use
from nanotron.config import ModelArgs, RandomInit, SpectralMupInit
from nanotron.parallel import ParallelContext
from nanotron.scaling.parametrization import ParametrizationMethod

from tests.helpers.llama_helper import TINY_LLAMA_CONFIG, create_llama_from_config, get_llama_training_config


@pytest.mark.parametrize("tp,dp,pp", [(2, 1, 1)])
@pytest.mark.parametrize("parametrization_method", [ParametrizationMethod.SPECTRAL_MUP])
@pytest.mark.skip
@rerun_if_address_is_in_use()
def test_parametrization(tp: int, dp: int, pp: int, parametrization_method: ParametrizationMethod):
    # [FUNCTION DOC]
    # Summary: Pytest entry point, parametrized over (tp, dp, pp) = (2, 1, 1) and the SPECTRAL_MUP method; marked skip and rerun on address-in-use. Chooses an init method config (RandomInit with std 1.0 for STANDARD, SpectralMupInit with use_mup enabled for SPECTRAL_MUP) and launches _test_parametrization on tp*dp*pp processes via init_distributed.
    # Inputs: tp, dp, pp - integer parallel sizes; parametrization_method - ParametrizationMethod enum value.
    # Outputs: None.
    # Exceptions: Propagates failures from the spawned workers.
    # Side effects: Spawns distributed worker processes.
    # KNOWN ISSUE: If parametrization_method is neither STANDARD nor SPECTRAL_MUP, init_method is never assigned and a NameError/UnboundLocalError would occur.
    if parametrization_method == ParametrizationMethod.STANDARD:
        init_method = RandomInit(std=1.0)
    elif parametrization_method == ParametrizationMethod.SPECTRAL_MUP:
        init_method = SpectralMupInit(use_mup=True)

    init_distributed(tp=tp, dp=dp, pp=pp)(_test_parametrization)(
        init_method=init_method,
        parametrization_method=parametrization_method,
    )


def _test_parametrization(
    parallel_context: ParallelContext,
    init_method: Union[RandomInit, SpectralMupInit],
    parametrization_method: ParametrizationMethod,
):
    # [FUNCTION DOC]
    # HELPER: per-rank body of test_parametrization.
    # Behavior: Builds ModelArgs with the given init method and TINY_LLAMA_CONFIG, derives a training config, creates a tiny Llama on CUDA, and calls init_model_randomly with the parametrization method. Builds a table mapping parameter-name substrings to expected std: 0 for layer norms, 1 for token embedding, and the spectral std (1/sqrt(fan_in) times min(1, sqrt(fan_out/fan_in))) for qkv_proj, o_proj (fan_in is the per-rank in_features times tp size), gate_up_proj and down_proj. Iterates over all named parameters except lm_head and asserts each parameter's std matches its expected value within an absolute tolerance of 0.05.
    # Inputs: parallel_context - ParallelContext; init_method - RandomInit or SpectralMupInit config; parametrization_method - ParametrizationMethod.
    # Outputs: None.
    # Exceptions: AssertionError if a parameter has no expected std or its std deviates beyond tolerance.
    # Side effects: Allocates a model on the CUDA device; consumes RNG state.
    # KNOWN ISSUE: qkv_proj expected std uses fan_out equal to intermediate_size, but the qkv projection outputs roughly hidden-size-related dimensions, not the MLP intermediate size; the expectation looks wrong (the test is skipped).
    # KNOWN ISSUE: The local variable name interdimte_size is a typo of intermediate size (cosmetic).
    def spectral_std(fan_in: int, fan_out: int):
        # [FUNCTION DOC]
        # HELPER: computes the expected spectral muP init std.
        # Behavior: Returns 1/sqrt(fan_in) multiplied by the minimum of 1 and sqrt(fan_out/fan_in), as a scalar tensor.
        # Inputs: fan_in, fan_out - integer layer dimensions.
        # Outputs: 0-dim float torch tensor.
        # Exceptions: ZeroDivisionError if fan_in is 0.
        # Side effects: None.
        return torch.tensor((1.0 / math.sqrt(fan_in)) * min(1, math.sqrt(fan_out / fan_in)))

    model_args = ModelArgs(init_method=init_method, model_config=TINY_LLAMA_CONFIG)
    config = get_llama_training_config(model_args)

    llama = create_llama_from_config(
        model_config=TINY_LLAMA_CONFIG,
        device=torch.device("cuda"),
        parallel_context=parallel_context,
    )
    llama.init_model_randomly(config=config, init_method=parametrization_method)

    hidden_size = TINY_LLAMA_CONFIG.hidden_size
    interdimte_size = TINY_LLAMA_CONFIG.intermediate_size

    o_proj_infeatures = llama.model.decoder[0].pp_block.attn.o_proj.in_features * parallel_context.tensor_parallel_size
    NAME_TO_EXPECTED_STD = {
        "input_layernorm": torch.tensor(0.0),
        "post_attention_layernorm": torch.tensor(0.0),
        "final_layer_norm": torch.tensor(0.0),
        "token_embedding": torch.tensor(1.0),
        # "lm_head": torch.tensor(1.0),
        "qkv_proj": spectral_std(fan_in=hidden_size, fan_out=interdimte_size),
        "o_proj": spectral_std(fan_in=o_proj_infeatures, fan_out=hidden_size),
        "gate_up_proj": spectral_std(fan_in=hidden_size, fan_out=interdimte_size),
        "down_proj": spectral_std(fan_in=interdimte_size, fan_out=hidden_size),
    }

    def find_expected_std(param_name):
        # [FUNCTION DOC]
        # HELPER: maps a parameter name to its expected std.
        # Behavior: Returns the value for the first key of NAME_TO_EXPECTED_STD (in insertion order) that is a substring of param_name.
        # Inputs: param_name - string parameter name.
        # Outputs: tensor expected std, or None if no key matches (implicit).
        # Exceptions: None explicitly raised.
        # Side effects: None.
        for name in NAME_TO_EXPECTED_STD:
            if name in param_name:
                return NAME_TO_EXPECTED_STD[name]

    for name, param in llama.model.named_parameters():
        if "lm_head" in name:
            continue

        expected_std = find_expected_std(name)
        assert expected_std is not None, f"Could not find expected std for {name}"
        assert torch.allclose(
            param.std().float(), expected_std, atol=0.05
        ), f"name: {name}, expected: {expected_std}, actual: {param.std()}"
