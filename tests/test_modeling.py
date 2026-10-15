# [MODULE DOC]
# Purpose: smoke test running a forward pass of a small Qwen2 model (1024 hidden, 2 layers, tied embeddings, flash attention 2) on one GPU; also runnable as a script.
# Main contents: test_qwen and _test_qwen.
import pytest
import torch
from helpers.utils import init_distributed, rerun_if_address_is_in_use
from nanotron.config import ModelArgs, RandomInit
from nanotron.models.qwen import Qwen2Config
from nanotron.parallel import ParallelContext
from transformers import AutoTokenizer

from tests.helpers.qwen_helper import create_qwen_from_config, get_qwen_training_config


@pytest.mark.parametrize("tp,dp,pp", [(1, 1, 1)])  # Simple test with single GPU
@rerun_if_address_is_in_use()
def test_qwen(tp: int, dp: int, pp: int):
    # [FUNCTION DOC]
    # Behavior: spawns tp*dp*pp ranks (only 1,1,1 is parametrized) running _test_qwen.
    # Inputs: tp, dp, pp.
    # Outputs: None.
    # Exceptions: propagates failures.
    # Side effects: process spawning.
    # KNOWN ISSUE: the docstring claims it tests loss masking for SFT, but the body only runs a forward pass and prints the output shape; nothing about masking is asserted.
    """Test that loss masking works correctly for Qwen SFT."""
    init_distributed(tp=tp, dp=dp, pp=pp)(_test_qwen)()


def _test_qwen(parallel_context: ParallelContext):
    # [FUNCTION DOC]
    # Behavior: builds the Qwen2 config and a training Config, creates and randomly initializes the model on CUDA, loads the Qwen/Qwen2-0.5B tokenizer (setting pad to eos), then runs a forward with random input ids and labels (batch 1, length 16, all-true masks) and prints the hidden_states shape. Destroys the context.
    # Inputs: parallel_context.
    # Outputs: None.
    # Exceptions: errors from model build or forward.
    # Side effects: tokenizer download, GPU allocation, stdout print, process group destruction.
    # KNOWN ISSUE: two tensors (positions and a mask for batch 2) are computed and discarded, and the tokenizer is never used; dead code.
    # KNOWN ISSUE: the model output is read under the key "hidden_states"; a training model normally returns a loss dictionary, so this may fail with KeyError depending on Qwen2ForTraining's outputs.
    # Setup model config
    TINY_CONFIG = Qwen2Config(
        hidden_size=1024,
        num_attention_heads=16,
        num_key_value_heads=16,
        intermediate_size=2816,
        num_hidden_layers=2,
        rms_norm_eps=1e-6,
        max_position_embeddings=32768,
        vocab_size=32000,
        rope_theta=10000.0,
        tie_word_embeddings=True,
        _attn_implementation="flash_attention_2",
    )

    # Setup model config
    model_args = ModelArgs(init_method=RandomInit(std=0.02), model_config=TINY_CONFIG)
    config = get_qwen_training_config(model_args)

    # Create model
    model = create_qwen_from_config(
        model_config=config.model.model_config,
        device=torch.device("cuda"),
        parallel_context=parallel_context,
    )
    model.init_model_randomly(config=config)

    # Create tokenizer
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B")
    tokenizer.pad_token = tokenizer.eos_token

    batch_size = 2
    seq_len = 16
    torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)
    torch.ones(batch_size, seq_len, dtype=torch.bool)

    # Forward pass
    outputs = model(
        input_ids=torch.randint(0, 100, (1, seq_len)).cuda(),
        input_mask=torch.ones(1, seq_len).bool().cuda(),
        label_ids=torch.randint(0, 100, (1, seq_len)).cuda(),
        label_mask=torch.ones(1, seq_len).cuda(),
    )
    print("Output shape:", outputs["hidden_states"].shape)

    parallel_context.destroy()


if __name__ == "__main__":
    test_qwen(1, 1, 1)
