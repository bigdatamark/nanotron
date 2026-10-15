# [MODULE DOC]
# Purpose: pytest tests checking that ModelArgs restores model configs given as plain dictionaries (Llama, Qwen2, Starcoder2, nested MoE) and propagates the muP flag from the init method.
# Main contents: three test functions; no distributed setup.
import pytest
from nanotron.config import ModelArgs, RandomInit, SpectralMupInit
from nanotron.config.models_config import LlamaConfig, MoEConfig, Qwen2Config, Starcoder2Config


@pytest.mark.parametrize(
    ("model_config", "expected_type"),
    [
        ({"is_llama_config": True}, LlamaConfig),
        ({"is_qwen2_config": True}, Qwen2Config),
        ({"is_starcoder2_config": True}, Starcoder2Config),
    ],
)
def test_model_args_restores_serialized_model_config(model_config, expected_type):
    # [FUNCTION DOC]
    # Behavior: builds ModelArgs from a dictionary with a model-type marker flag and random init (std 0.02), checks the config is converted to the expected class and muP is off.
    # Inputs: model_config, dictionary; expected_type, the config class (parametrized).
    # Outputs: None.
    # Exceptions: AssertionError on failure.
    # Side effects: None.
    model_args = ModelArgs(model_config=model_config, init_method=RandomInit(std=0.02))

    assert isinstance(model_args.model_config, expected_type)
    assert model_args.model_config._is_using_mup is False


def test_model_args_restores_nested_moe_config():
    # [FUNCTION DOC]
    # Behavior: builds ModelArgs from a Qwen2 dictionary containing a nested moe_config dictionary (4 experts, top 2) and checks it becomes a MoEConfig with 4 experts.
    # Inputs: none.
    # Outputs: None.
    # Exceptions: AssertionError on failure.
    # Side effects: None.
    model_args = ModelArgs(
        model_config={"is_qwen2_config": True, "moe_config": {"num_experts": 4, "top_k": 2}},
        init_method=RandomInit(std=0.02),
    )

    assert isinstance(model_args.model_config.moe_config, MoEConfig)
    assert model_args.model_config.moe_config.num_experts == 4


def test_model_args_sets_mup_on_restored_config():
    # [FUNCTION DOC]
    # Behavior: builds ModelArgs from a Llama dictionary with SpectralMupInit(use_mup true) and checks the restored LlamaConfig reports is_using_mup true.
    # Inputs: none.
    # Outputs: None.
    # Exceptions: AssertionError on failure.
    # Side effects: None.
    model_args = ModelArgs(
        model_config={"is_llama_config": True},
        init_method=SpectralMupInit(use_mup=True),
    )

    assert isinstance(model_args.model_config, LlamaConfig)
    assert model_args.model_config.is_using_mup is True
