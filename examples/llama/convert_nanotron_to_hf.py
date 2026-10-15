# [MODULE DOC]
# Purpose: Script converting a nanotron Llama (or Qwen2) checkpoint into HuggingFace LlamaForCausalLM format, optionally saving a tokenizer and running a test generation.
# Main contents: TEST_PROMPT, _handle_attention_block, _handle_gate_up_proj, convert_nt_to_hf, get_hf_config, convert_checkpoint_and_save, check_converted_model_generation, and a CLI main (--checkpoint_path, --save_path, --tokenizer_name default Llama-2-7b-chat, --config_cls LlamaConfig or Qwen2Config).
# Usage: torchrun with one process.
# Import-time side effects: None.
# KNOWN ISSUE: uses a relative import of convert_weights, so running the file directly as in the docstring fails.
"""
Converts a nanotron model to HF format
Command:
    torchrun --nproc_per_node=1 convert_nanotron_to_hf.py --checkpoint_path=nanotron-path --save_path=hf-path
"""

import json
from argparse import ArgumentParser
from pathlib import Path
from typing import Literal, Optional, Type

import torch
from nanotron.config import LlamaConfig as NanotronLlamaConfig, Qwen2Config as NanotronQwen2Config
from nanotron.config import NanotronConfigs
from nanotron.models import init_on_device_and_dtype
from nanotron.models.llama import LlamaForTraining
from transformers import AutoTokenizer, LlamaForCausalLM
from transformers import LlamaConfig as HFLlamaConfig

from .convert_weights import get_config_mapping, get_weight_mapping, load_nanotron_model

TEST_PROMPT = "What is the meaning of the word chutzpah?\nThe word chutzpah means"


def _handle_attention_block(
    qkv: torch.Tensor,
    part: Literal["q", "k", "v"],
    n_q_heads: int,
    n_kv_heads: int,
    d_qk: int,
    interleave,
) -> torch.Tensor:
    # Huggingface Llama separates the q, k, v weights (as opposed to nanotron).
    # Furthermore, in the rotary embeddings in nanotron expects interleaved pairs of even
    # and odd dimensions GPT-J style, while the huggingface implementation expects
    # the whole 1st half and then the whole 2nd half GPT-NeoX style (for more information
    # see flash_attn.layers.rotary.RotaryEmbedding).
    # This function selects the proper chunk of the bundled qkv tensor and permutation
    # to ensure correct transformation to huggingface.

    # [FUNCTION DOC]
    # HELPER: extracts q, k or v from nanotron's fused qkv_proj weight.
    # Behavior: Asserts part is q, k or v; slices rows [0, n_q*d_qk) for q, the next n_kv*d_qk for k and the rest for v; for q and k optionally undoes the interleaved rotary layout back to HF half-split layout.
    # Inputs: qkv: fused weight tensor; part: "q", "k" or "v"; n_q_heads, n_kv_heads: int; d_qk: head dimension; interleave: bool.
    # Outputs: tensor slice for the requested projection.
    # Exceptions: AssertionError on invalid part. Side effects: None.
    def interleave_weight(w: torch.Tensor):
        # [FUNCTION DOC]
        # HELPER: inverse per-head rotary permutation.
        # Behavior: Splits rows into heads of d_qk and reorders each head's interleaved pairs into first-half/second-half order, then concatenates.
        # Inputs: w: tensor [heads*d_qk, hidden]. Outputs: tensor of same shape. Exceptions: reshape errors. Side effects: None.
        w_new = []
        for head_w in w.split(d_qk):
            head_w = head_w.view(d_qk // 2, 2, -1).transpose(0, 1).reshape(d_qk, -1)
            w_new.append(head_w)
        return torch.cat(w_new)

    assert part in ["q", "k", "v"], "part must be one of [q, k, v]"

    index_end_q = n_q_heads * d_qk
    index_end_k = index_end_q + n_kv_heads * d_qk
    if part == "q":
        return interleave_weight(qkv[:index_end_q]) if interleave else qkv[:index_end_q]
    elif part == "k":
        return interleave_weight(qkv[index_end_q:index_end_k]) if interleave else qkv[index_end_q:index_end_k]
    elif part == "v":
        return qkv[index_end_k:]


def _handle_gate_up_proj(gate_up_proj: torch.Tensor, gate: bool) -> torch.Tensor:
    # The gate and up projection are bundled in nanotron.
    # This function selects the proper chunk in the bundled weights to return
    # either the gate or the up projection only.
    # [FUNCTION DOC]
    # HELPER: splits nanotron's fused gate_up_proj weight.
    # Behavior: Returns the first half of rows for the gate projection, the second half for up.
    # Inputs: gate_up_proj: tensor [2*intermediate, hidden]; gate: bool. Outputs: tensor [intermediate, hidden].
    # Exceptions: None explicitly raised. Side effects: None.
    weight_size = gate_up_proj.shape[0] // 2
    if gate:
        return gate_up_proj[:weight_size]
    else:
        return gate_up_proj[weight_size:]


def convert_nt_to_hf(
    nanotron_model: LlamaForTraining,
    hf_model: LlamaForCausalLM,
    model_config: NanotronLlamaConfig,
    interleave_qkv: bool = False,
):
    # [FUNCTION DOC]
    # Behavior: Gets the nanotron state dict and the HF-to-nanotron name mapping; for every direct parameter of every HF module looks up the nanotron tensor; for qkv takes the projection letter from the fifth name component (q_proj, k_proj or v_proj) and slices it; for gate_up chooses half by whether the module name contains "gate"; copies into the HF parameter under no_grad.
    # Inputs: nanotron_model: LlamaForTraining; hf_model: LlamaForCausalLM; model_config: nanotron config; interleave_qkv: bool, default False.
    # Outputs: None.
    # Exceptions: KeyError for unmapped HF parameters; shape errors from copy.
    # Side effects: mutates hf_model parameters in place.
    """Converts the weights from the nanotron_model to hf_model, making modifications
    in-place."""

    nanotron_model_state_dict = nanotron_model.state_dict()

    hf_to_nt = get_weight_mapping(model_config, nt_to_hf=False)
    for module_name_hf, module_hf in hf_model.named_modules():
        for param_name_hf, param_hf in module_hf.named_parameters(recurse=False):
            # Get the Nanotron parameter
            nanotron_key = hf_to_nt[f"{module_name_hf}.{param_name_hf}"]
            param = nanotron_model_state_dict[nanotron_key]

            if "qkv_proj" in nanotron_key:
                proj_name = module_name_hf.split(".")[4][0]
                param = _handle_attention_block(
                    param,
                    proj_name,
                    model_config.num_attention_heads,
                    model_config.num_key_value_heads,
                    model_config.hidden_size // model_config.num_attention_heads,
                    interleave_qkv,
                )

            elif "gate_up_proj" in nanotron_key:
                gate = "gate" in module_name_hf
                param = _handle_gate_up_proj(param, gate)

            with torch.no_grad():
                # print(f"Copying parameter {module_name_hf} ({param_hf.shape}) from {nanotron_key} ({param.shape})")
                param_hf.copy_(param)


def get_hf_config(config: NanotronLlamaConfig) -> HFLlamaConfig:
    # [FUNCTION DOC]
    # Behavior: Copies each mapped field from the nanotron config into an HF LlamaConfig.
    # Inputs: config: nanotron Llama or Qwen2 config. Outputs: HF LlamaConfig.
    # Exceptions: AttributeError if a field is missing. Side effects: None.
    """Converts a nanotron configuration to huggingface configuration."""
    attrs = {key: getattr(config, value) for key, value in get_config_mapping(nt_to_hf=False).items()}
    return HFLlamaConfig(**attrs)


def convert_checkpoint_and_save(
    checkpoint_path: Path,
    save_path: Path,
    tokenizer_name: Optional[str] = None,
    config_cls: Type[NanotronConfigs] = NanotronLlamaConfig,
):
    # [FUNCTION DOC]
    # Behavior: Reads model_config.json from checkpoint_path into config_cls, loads the nanotron model with weights, creates an empty HF LlamaForCausalLM on CUDA in bfloat16 from the converted config, optionally saves the tokenizer to save_path, copies weights and saves the HF model, then prints.
    # Inputs: checkpoint_path: Path; save_path: Path; tokenizer_name: optional HF tokenizer id; config_cls: nanotron config class (default LlamaConfig).
    # Outputs: None.
    # Exceptions: file/loading errors; KeyError from conversion.
    # Side effects: distributed initialization; GPU memory; network download for tokenizer; file writes; stdout.
    # KNOWN ISSUE: config_cls is not forwarded to load_nanotron_model, so a Qwen2Config checkpoint is loaded into LlamaForTraining.
    """Loads the nanotron checkpoint in `checkpoint_path`, creates
    a new huggingface instance, copies the weights from the nanotron checkpoint
    and saves the transformed huggingface to `save_path`."""

    # Init nanotron model.
    with open(checkpoint_path / "model_config.json", "r") as f:
        attrs = json.load(f)
        model_config = config_cls(**attrs)
    nanotron_model = load_nanotron_model(
        model_config=model_config,
        checkpoint_path=checkpoint_path,
    )
    # Init huggingface model.
    with init_on_device_and_dtype(torch.device("cuda"), torch.bfloat16):
        model_config_hf = get_hf_config(model_config)
        hf_model = LlamaForCausalLM._from_config(model_config_hf)

    # Copy weights, initialize tokenizer and save model.
    if tokenizer_name is not None:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        tokenizer.save_pretrained(save_path)
    convert_nt_to_hf(nanotron_model, hf_model, model_config)
    hf_model.save_pretrained(save_path)
    print(f"Model saved to {save_path}")


def check_converted_model_generation(save_path: Path):
    # [FUNCTION DOC]
    # Behavior: Loads tokenizer and HF model from save_path, tokenizes TEST_PROMPT on CUDA, prints the decoded input, generates up to 100 new tokens in bfloat16 and prints the decoded result.
    # Inputs: save_path: Path. Outputs: None.
    # Exceptions: loading/generation errors.
    # Side effects: GPU memory; stdout.
    """Loads a huggingface model and tokenizer from `save_path` and
    performs a dummy text generation."""

    tokenizer = AutoTokenizer.from_pretrained(save_path)
    input_ids = tokenizer(TEST_PROMPT, return_tensors="pt")["input_ids"].cuda()
    print("Inputs:", tokenizer.batch_decode(input_ids))

    model = LlamaForCausalLM.from_pretrained(save_path).cuda().bfloat16()
    out = model.generate(input_ids, max_new_tokens=100)
    print("Generation (converted): ", tokenizer.batch_decode(out))


if __name__ == "__main__":
    parser = ArgumentParser(description="Convert Nanotron weights to HF format")
    parser.add_argument("--checkpoint_path", type=Path, default="llama-7b", help="Path to the checkpoint")
    parser.add_argument("--save_path", type=Path, default="llama-7b-hf", help="Path to save the HF model")
    parser.add_argument("--tokenizer_name", type=str, default="meta-llama/Llama-2-7b-chat-hf")
    parser.add_argument("--config_cls", type=str, default="LlamaConfig", help="Config class to use for conversion (Either LlamaConfig or Qwen2Config)")
    args = parser.parse_args()

    if args.config_cls == "LlamaConfig":
        config_cls = NanotronLlamaConfig
    elif args.config_cls == "Qwen2Config":
        config_cls = NanotronQwen2Config
    else:
        raise ValueError(f"Invalid config class: {args.config_cls}. Should be one of [NanotronLlamaConfig, NanotronQwen2Config]")

    # Convert Nanotron model to HF format.
    convert_checkpoint_and_save(
        checkpoint_path=args.checkpoint_path, save_path=args.save_path, tokenizer_name=args.tokenizer_name, config_cls=config_cls
    )

    # Check if the conversion was successful by generating some text.
    if args.tokenizer_name is not None:
        check_converted_model_generation(save_path=args.save_path)
