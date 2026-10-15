# [MODULE DOC]
# Purpose: Conversion script turning a HuggingFace Qwen1.5/Qwen2-MoE checkpoint into a nanotron Qwen2 checkpoint with MoE layers (weights, metadata, tokenizer, config.yaml, model_config.json).
# Main contents: get_args (CLI), main (conversion). Module constants DEVICE (cuda; the model must be built on GPU for rotary buffers) and TORCH_DTYPE (bfloat16).
# Usage: torchrun with a single process (dp=pp=tp=1).
# Import-time side effects: logger creation; requires transformers, yaml and CUDA availability when used.
"""
torchrun --nproc-per-node 1 convert.py --nanotron-checkpoint-path nanotron_checkpoints/Qwen1.5-MoE-A2.7B --pretrained-model-name-or-path Qwen/Qwen1.5-MoE-A2.7B
"""
import argparse
import json
from dataclasses import asdict
from pathlib import Path

import torch
import yaml
from nanotron import logging
from nanotron.config import Config, GeneralArgs, LoggingArgs, ModelArgs, ParallelismArgs, TokenizerArgs
from nanotron.config.models_config import ExistingCheckpointInit, MoEConfig, Qwen2Config
from nanotron.logging import log_rank, set_ranks_logging_level
from nanotron.models import build_model
from nanotron.models.qwen import Qwen2ForTraining
from nanotron.parallel import ParallelContext
from nanotron.parallel.parameters import sanity_check
from nanotron.serialize import TrainingMetadata, save_meta, save_weights
from nanotron.serialize.metadata import DataStageMetadata
from nanotron.trainer import mark_tied_parameters
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, Qwen2MoeConfig

logger = logging.get_logger(__name__)

# NOTE: We need to initialize the model on gpu, because RotaryEmbedding
# requires its buffer to be on gpu
DEVICE = torch.device("cuda")
TORCH_DTYPE = torch.bfloat16


def get_args():
    # [FUNCTION DOC]
    # Behavior: Parses required --nanotron-checkpoint-path (output directory) and --pretrained-model-name-or-path (HF id or local path).
    # Inputs: None. Outputs: argparse Namespace. Exceptions: SystemExit on missing arguments. Side effects: reads sys.argv.
    parser = argparse.ArgumentParser()
    group = parser.add_argument_group(title="Nanotron Model")
    group.add_argument(
        "--nanotron-checkpoint-path",
        type=str,
        required=True,
        help="A path to a directory to store the converted Nanotron Checkpoint",
    )

    group = parser.add_argument_group(title="HuggingFace Model")
    group.add_argument(
        "--pretrained-model-name-or-path",
        type=str,
        required=True,
        help="A path to a directory containing model weights saved using save_pretrained() or the model id of a pretrained model hosted inside a model repo on the Hugging Face Hub",
    )

    args = parser.parse_args()

    return args


def main(args):
    # Init Nanotron Parallel Utilities
    # [FUNCTION DOC]
    # Behavior:
    # 1. Builds a single-rank ParallelContext and sets rank logging levels.
    # 2. Loads the HF causal LM in bfloat16 with flash attention 2 on CUDA and reads its Qwen2MoeConfig.
    # 3. Builds a nanotron Qwen2Config mirroring the HF config (attention bias on, non-interleaved rope, no pad token) with a MoEConfig (top-k, number of experts, expert and shared-expert intermediate sizes, router aux loss coefficient, shared expert enabled).
    # 4. Builds an empty Qwen2ForTraining on CUDA in bfloat16, marks tied parameters and runs the parameter sanity check.
    # 5. Under no_grad copies weights with shape asserts: token embedding; per layer the input norm, concatenated q/k/v weights and biases into qkv_proj, o_proj, the router (HF gate), the shared expert gate and up weights concatenated into gate_up_proj, shared expert down_proj and shared_expert_gate, every expert's gate/up/down weights transposed and stacked into merged_gate_up_proj [experts, hidden, 2*moe_intermediate] (gate first, then up) and merged_down_proj [experts, moe_intermediate, hidden], and the post-attention norm; then final norm and lm_head.
    # 6. Saves weights and training metadata (step 0, one empty data stage), the tokenizer, a config.yaml (Config with ExistingCheckpointInit pointing to the output folder) and model_config.json.
    # Inputs: args: Namespace from get_args.
    # Outputs: None.
    # Exceptions: AssertionError on any shape mismatch; errors from model download/loading and file I/O.
    # Side effects: initializes torch.distributed; GPU memory for both models; network download; writes many files under the output directory; logging.
    parallel_config = ParallelismArgs(dp=1, pp=1, tp=1)

    parallel_context = ParallelContext(
        data_parallel_size=parallel_config.dp,
        pipeline_parallel_size=parallel_config.pp,
        tensor_parallel_size=parallel_config.tp,
    )

    set_ranks_logging_level(parallel_context=parallel_context, logging_config=LoggingArgs())

    # Load Qwen-MoE HF model
    log_rank(
        f"Loading pretrained qwen moe Model: {args.pretrained_model_name_or_path}",
        logger=logger,
        level=logging.INFO,
        rank=0,
    )
    hf_model = AutoModelForCausalLM.from_pretrained(
        args.pretrained_model_name_or_path, torch_dtype=TORCH_DTYPE, attn_implementation="flash_attention_2"
    ).to(DEVICE)
    hf_config: Qwen2MoeConfig = hf_model.config

    # Set Nanotron Qwen2Config
    nanotron_config = Qwen2Config(
        bos_token_id=hf_config.bos_token_id,
        eos_token_id=hf_config.eos_token_id,
        hidden_act=hf_config.hidden_act,
        hidden_size=hf_config.hidden_size,
        initializer_range=hf_config.initializer_range,
        intermediate_size=hf_config.intermediate_size,
        is_qwen2_config=True,
        max_position_embeddings=hf_config.max_position_embeddings,
        num_attention_heads=hf_config.num_attention_heads,
        num_hidden_layers=hf_config.num_hidden_layers,
        num_key_value_heads=hf_config.num_key_value_heads,
        pad_token_id=None,
        attention_bias=True,  # qwen-moe uses attention bias
        rms_norm_eps=hf_config.rms_norm_eps,
        rope_scaling=hf_config.rope_scaling,
        rope_theta=hf_config.rope_theta,
        rope_interleaved=False,
        tie_word_embeddings=hf_config.tie_word_embeddings,
        use_cache=hf_config.use_cache,
        vocab_size=hf_config.vocab_size,
        moe_config=MoEConfig(
            top_k=hf_config.num_experts_per_tok,
            num_experts=hf_config.num_experts,
            moe_intermediate_size=hf_config.moe_intermediate_size,
            shared_expert_intermediate_size=hf_config.shared_expert_intermediate_size,
            router_aux_loss_coef=hf_config.router_aux_loss_coef,
            enable_shared_expert=True,
        ),
    )

    # Init Nanotron Qwen-MoE model
    log_rank("Init empty Nanotron Qwen Moe Model", logger=logger, level=logging.INFO, rank=0)
    nanotron_model = build_model(
        model_builder=lambda: Qwen2ForTraining(
            config=nanotron_config,
            parallel_context=parallel_context,
            parallel_config=parallel_config,
        ),
        parallel_context=parallel_context,
        dtype=TORCH_DTYPE,
        device=DEVICE,
    )

    mark_tied_parameters(model=nanotron_model, parallel_context=parallel_context, parallel_config=parallel_config)
    sanity_check(root_module=nanotron_model)

    # Copy params from HF to Nanotron
    log_rank("Copying weights from HF model to Nanotron model...", logger=logger, level=logging.INFO, rank=0)
    # Token embeddings
    log_rank("Copying Token Embeddings...", logger=logger, level=logging.INFO, rank=0)

    with torch.no_grad():
        # token embeddings
        assert (
            nanotron_model.model.token_position_embeddings.pp_block.token_embedding.weight.shape
            == hf_model.model.embed_tokens.weight.shape
        )

        nanotron_model.model.token_position_embeddings.pp_block.token_embedding.weight.copy_(
            hf_model.model.embed_tokens.weight
        )

        # Decoder layers
        for i in tqdm(
            range(nanotron_config.num_hidden_layers),
            desc="Copying Hidden Layers",
            total=nanotron_config.num_hidden_layers,
        ):
            # Input layer norm
            assert (
                hf_model.model.layers[i].input_layernorm.weight.shape
                == nanotron_model.model.decoder[i].pp_block.input_layernorm.weight.shape
            )

            nanotron_model.model.decoder[i].pp_block.input_layernorm.weight.copy_(
                hf_model.model.layers[i].input_layernorm.weight
            )

            # Self attn
            ## QKV
            tmp_qkv_proj = torch.cat(
                [
                    hf_model.model.layers[i].self_attn.q_proj.weight,
                    hf_model.model.layers[i].self_attn.k_proj.weight,
                    hf_model.model.layers[i].self_attn.v_proj.weight,
                ],
                dim=0,
            )
            assert tmp_qkv_proj.shape == nanotron_model.model.decoder[i].pp_block.attn.qkv_proj.weight.shape
            nanotron_model.model.decoder[i].pp_block.attn.qkv_proj.weight.copy_(tmp_qkv_proj)

            ## QKV bias
            tmp_qkv_bias = torch.cat(
                [
                    hf_model.model.layers[i].self_attn.q_proj.bias,
                    hf_model.model.layers[i].self_attn.k_proj.bias,
                    hf_model.model.layers[i].self_attn.v_proj.bias,
                ],
                dim=0,
            )
            assert tmp_qkv_bias.shape == nanotron_model.model.decoder[i].pp_block.attn.qkv_proj.bias.shape
            nanotron_model.model.decoder[i].pp_block.attn.qkv_proj.bias.copy_(tmp_qkv_bias)

            ## O
            assert (
                hf_model.model.layers[i].self_attn.o_proj.weight.shape
                == nanotron_model.model.decoder[i].pp_block.attn.o_proj.weight.shape
            )
            nanotron_model.model.decoder[i].pp_block.attn.o_proj.weight.copy_(
                hf_model.model.layers[i].self_attn.o_proj.weight
            )

            # MLP
            ## Router
            assert (
                hf_model.model.layers[i].mlp.gate.weight.shape
                == nanotron_model.model.decoder[i].pp_block.mlp.router.weight.shape
            )
            nanotron_model.model.decoder[i].pp_block.mlp.router.weight.copy_(hf_model.model.layers[i].mlp.gate.weight)

            ## shared expert: Gate Up Proj
            tmp_shared_expert = torch.cat(
                [
                    hf_model.model.layers[i].mlp.shared_expert.gate_proj.weight,
                    hf_model.model.layers[i].mlp.shared_expert.up_proj.weight,
                ],
                dim=0,
            )
            assert (
                tmp_shared_expert.shape
                == nanotron_model.model.decoder[i].pp_block.mlp.shared_expert.gate_up_proj.weight.shape
            )
            nanotron_model.model.decoder[i].pp_block.mlp.shared_expert.gate_up_proj.weight.copy_(tmp_shared_expert)

            ## shared expert: Down Proj
            assert (
                hf_model.model.layers[i].mlp.shared_expert.down_proj.weight.shape
                == nanotron_model.model.decoder[i].pp_block.mlp.shared_expert.down_proj.weight.shape
            )
            nanotron_model.model.decoder[i].pp_block.mlp.shared_expert.down_proj.weight.copy_(
                hf_model.model.layers[i].mlp.shared_expert.down_proj.weight
            )

            ## shared expert: Gate
            assert (
                hf_model.model.layers[i].mlp.shared_expert_gate.weight.shape
                == nanotron_model.model.decoder[i].pp_block.mlp.shared_expert_gate.weight.shape
            )
            nanotron_model.model.decoder[i].pp_block.mlp.shared_expert_gate.weight.copy_(
                hf_model.model.layers[i].mlp.shared_expert_gate.weight
            )

            ## experts:
            # concatenate all gate_up_proj and down_proj weights for experts into merged_gate_up_proj and merged_down_proj
            tmp_merged_gate_up_proj = torch.zeros(
                nanotron_config.moe_config.num_experts,
                nanotron_config.hidden_size,
                2 * nanotron_config.moe_config.moe_intermediate_size,
            )
            tmp_merged_down_proj = torch.zeros(
                nanotron_config.moe_config.num_experts,
                nanotron_config.moe_config.moe_intermediate_size,
                nanotron_config.hidden_size,
            )

            for j in range(nanotron_config.moe_config.num_experts):
                ## Gate Up Proj
                tmp_merged_gate_up_proj[j, :, : nanotron_config.moe_config.moe_intermediate_size] = (
                    hf_model.model.layers[i].mlp.experts[j].gate_proj.weight.T
                )
                tmp_merged_gate_up_proj[j, :, nanotron_config.moe_config.moe_intermediate_size :] = (
                    hf_model.model.layers[i].mlp.experts[j].up_proj.weight.T
                )

                ## Down Proj
                tmp_merged_down_proj[j] = hf_model.model.layers[i].mlp.experts[j].down_proj.weight.T

            # copy to merged_gate_up_proj and merged_down_proj
            nanotron_model.model.decoder[i].pp_block.mlp.experts.merged_gate_up_proj.copy_(tmp_merged_gate_up_proj)
            nanotron_model.model.decoder[i].pp_block.mlp.experts.merged_down_proj.copy_(tmp_merged_down_proj)

            # Post attn layer norm
            assert (
                hf_model.model.layers[i].post_attention_layernorm.weight.shape
                == nanotron_model.model.decoder[i].pp_block.post_attention_layernorm.weight.shape
            )
            nanotron_model.model.decoder[i].pp_block.post_attention_layernorm.weight.copy_(
                hf_model.model.layers[i].post_attention_layernorm.weight
            )

        # Last layer norm
        log_rank("Copying Final Layer Norm...", logger=logger, level=logging.INFO, rank=0)
        assert nanotron_model.model.final_layer_norm.pp_block.weight.shape == hf_model.model.norm.weight.shape
        nanotron_model.model.final_layer_norm.pp_block.weight.copy_(hf_model.model.norm.weight)

        # LM_Head
        log_rank("Copying LM Head...", logger=logger, level=logging.INFO, rank=0)
        assert nanotron_model.model.lm_head.pp_block.weight.shape == hf_model.lm_head.weight.shape
        nanotron_model.model.lm_head.pp_block.weight.copy_(hf_model.lm_head.weight)

    log_rank("Copied weights from HF model to Nanotron model!", logger=logger, level=logging.INFO, rank=0)
    # Store weights
    nanotron_checkpoint_path = Path(args.nanotron_checkpoint_path)
    save_weights(model=nanotron_model, parallel_context=parallel_context, root_folder=nanotron_checkpoint_path)

    # Store metadata
    log_rank("Storing Nanotron model Configs and Metadata!", logger=logger, level=logging.INFO, rank=0)
    training_metadata = TrainingMetadata(
        last_train_step=0,
        consumed_train_samples=0,
        data_stages=[DataStageMetadata(name="Empty", consumed_train_samples=0, start_training_step=0)],
    )
    save_meta(
        root_folder=nanotron_checkpoint_path, parallel_context=parallel_context, training_metadata=training_metadata
    )
    # Store Tokenizer into Nanotron Checkpoint folder
    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model_name_or_path)
    tokenizer.save_pretrained(nanotron_checkpoint_path)

    # Store Config and Model Config files
    with open(nanotron_checkpoint_path / "config.yaml", "w") as f:
        config = Config(
            general=GeneralArgs(project="Nanotron", run="Qwen2-MoE"),
            parallelism=parallel_config,
            model=ModelArgs(
                init_method=ExistingCheckpointInit(nanotron_checkpoint_path),
                model_config=nanotron_config,
            ),
            tokenizer=TokenizerArgs(tokenizer_name_or_path=args.pretrained_model_name_or_path),
        )
        log_rank("Saving config ...", logger=logger, level=logging.INFO, rank=0)
        yaml.dump(config.as_dict(), f)

    with open(nanotron_checkpoint_path / "model_config.json", "w") as f:
        log_rank("Saving model config ...", logger=logger, level=logging.INFO, rank=0)
        json.dump(asdict(nanotron_config), f)

    log_rank(
        f"Checkpoint conversion finished, check {args.nanotron_checkpoint_path}",
        logger=logger,
        level=logging.INFO,
        rank=0,
    )


if __name__ == "__main__":
    _args = get_args()
    main(_args)
