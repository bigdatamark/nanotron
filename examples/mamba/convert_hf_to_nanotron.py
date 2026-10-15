# [MODULE DOC]
# Purpose: script converting a HuggingFace transformers Mamba checkpoint (e.g. state-spaces/mamba-130m-hf)
# into a nanotron checkpoint usable by the Mamba example. Run with torchrun (one process per rank).
# Contents: load_config_hf, get_weight_from_hf, and a main block.
# Main block behavior: parses --inp_path, --out_path, --dp, --pp, --tp; builds ParallelismArgs (all-forward
# all-backward engine, ALL_REDUCE TP, no async comm) and a ParallelContext; sets per-rank log levels; reads the HF
# config and writes a MambaModelConfig (float32 dtype, RMSNorm, fused add norm, residual in fp32, ssm_cfg null)
# from a YAML template; loads the HF model on CUDA in float32 (eval mode); builds MambaForTraining with
# build_model; maps every nanotron parameter name to its HF name (embeddings, final norm, lm_head and per
# layer mixer A_log, D, in_proj, conv1d, x_proj, dt_proj, out_proj, norm); for each local nanotron parameter
# copies the HF tensor, either slice by slice using sharded local/global slice pairs for TP-sharded params or
# whole after asserting equal shapes; marks tied params; runs sanity_check; saves weights and meta
# (step 0, 0 samples); on global rank 0 writes config.yaml (MambaConfig with MambaInit and tokenizer = inp_path)
# and model_config.json.
# Side effects: network downloads, CUDA allocations, distributed init, file writes under out_path, logging.
# KNOWN ISSUE: device_map is computed but never used; the YAML template lists is_mamba_config twice and
# ignores the HF SSM hyperparameters (ssm_cfg left null, so non-default HF state/expand/conv sizes would mismatch).
# ruff: noqa: E402
"""
Converts a HF model to a Nanotron model

Command:
    torchrun --nproc_per_node=1 convert_hf_to_nanotron.py --inp_path state-spaces/mamba-130m-hf --out_path nanotron_weights
"""
import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Dict

import torch
import yaml
from config import MambaConfig, MambaInit, MambaModelConfig
from mamba import MambaForTraining
from nanotron import logging
from nanotron.config import (
    AllForwardAllBackwardPipelineEngine,
    GeneralArgs,
    LoggingArgs,
    ModelArgs,
    ParallelismArgs,
    TensorParallelLinearMode,
    TokenizerArgs,
)
from nanotron.distributed import dist
from nanotron.logging import log_rank, set_ranks_logging_level
from nanotron.models import build_model
from nanotron.parallel import ParallelContext
from nanotron.parallel.parameters import NanotronParameter, sanity_check
from nanotron.serialize import save_meta, save_weights
from nanotron.trainer import mark_tied_parameters
from tqdm import tqdm
from transformers import MambaConfig as HFMambaConfig
from transformers import MambaForCausalLM
from transformers.utils import CONFIG_NAME
from transformers.utils.hub import cached_file

logger = logging.get_logger(__name__)


def load_config_hf(model_name):
    # [FUNCTION DOC]
    # HELPER: fetches a raw HF config dictionary.
    # Behavior: resolves config.json for model_name via cached_file (no error on missing entry) and parses it as JSON.
    # Inputs: model_name (hub id or local path string). Outputs: dictionary of config values.
    # Exceptions: TypeError/OSError if the file cannot be resolved (cached_file returns None).
    # Side effects: possible network download and cache write; leaves the file handle to be closed by GC.
    # KNOWN ISSUE: unused in this script.
    resolved_archive_file = cached_file(model_name, CONFIG_NAME, _raise_exceptions_for_missing_entries=False)
    return json.load(open(resolved_archive_file))


def get_weight_from_hf(
    name: str,
    ref_module_state_dict: Dict[str, torch.Tensor],
    ref_module: MambaForCausalLM,
    nanotron_to_hf: Dict[str, str],
    get_grad: bool = False,
    param_is_tp_sharded: bool = False,
) -> torch.Tensor:
    # [FUNCTION DOC]
    # HELPER: returns the HF tensor corresponding to a nanotron parameter name.
    # Behavior: looks up the HF name in nanotron_to_hf; reads either the tensor from ref_module_state_dict
    # (get_grad False) or the gradient of the HF parameter (get_grad True). For in_proj weights it logs (rank 0) and
    # returns rows reordered by the interleave pattern (row i of the first half followed by row i of the second
    # half), because nanotron splits the x/z projection with a column-parallel view while HF chunks it in two
    # halves. Otherwise returns the tensor as is.
    # Inputs: name (nanotron parameter name); ref_module_state_dict (name to tensor dictionary); ref_module (HF
    # MambaForCausalLM); nanotron_to_hf (name mapping dictionary); get_grad (bool, default False);
    # param_is_tp_sharded (bool, default False).
    # Outputs: torch tensor (full, unsharded) or gradient tensor (may be None).
    # Exceptions: KeyError if name is unmapped; AssertionError if in_proj row count is odd.
    # Side effects: logging.
    # KNOWN ISSUE: param_is_tp_sharded is accepted but never used.
    """From our brrr implementation, we get the equivalent tensor in transformers implementation"""

    def _interleave_pattern(N):
        # [FUNCTION DOC]
        # HELPER: builds the interleaving permutation.
        # Behavior: for i from 0 to N/2 - 1 appends i then i + N/2 (e.g. 4 gives 0, 2, 1, 3).
        # Inputs: N (even int). Outputs: list of N ints. Exceptions: AssertionError if N is odd. Side effects: None.
        """
        interleave_pattern(4) -> [0, 2, 1, 3]
        interleave_pattern(8) -> [0, 4, 1, 5, 2, 6, 3, 7]
        """
        assert N % 2 == 0, "N must be even"
        pattern = []
        for i in range(N // 2):
            pattern.append(i)
            pattern.append(i + N // 2)
        return pattern

    hf_name = nanotron_to_hf[name]

    if get_grad is False:

        def _get_tensor(path: str):
            # [FUNCTION DOC]
            # HELPER: reads a tensor by HF name from the reference state dict (closure variant used when get_grad is False).
            # Inputs: path (HF parameter name). Outputs: tensor. Exceptions: KeyError if missing. Side effects: None.
            return ref_module_state_dict[path]

    else:

        def _get_tensor(path: str):
            # [FUNCTION DOC]
            # HELPER: returns the gradient of the HF parameter at path (closure variant used when get_grad is True).
            # Inputs: path. Outputs: gradient tensor or None. Exceptions: AttributeError if the parameter does not exist.
            # Side effects: None.
            param = ref_module.get_parameter(path)
            return param.grad

    param = _get_tensor(hf_name)

    if "in_proj" in hf_name:
        # In Nanotron, we do tensor parallel column so weight need to be split in the column dimension (i.e: xz.view(...))
        # However, the HF weights was trained such that it expected xz.chunk(...) to split the tensor in the row dimension
        # Thus, we need to interleaved the HF weights to make it compatible with Nanotron
        log_rank(
            f"Interleaving {hf_name} to make it compatible with Nanotron", logger=logger, level=logging.INFO, rank=0
        )
        return param[_interleave_pattern(param.shape[0]), :]

    return param


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert HF weights from states-space repo to brrr weights")
    parser.add_argument("--inp_path", type=str, default="state-spaces/mamba-130m-hf")
    parser.add_argument("--out_path", type=str, default="nanotron_weight")
    parser.add_argument("--dp", type=int, default=1)
    parser.add_argument("--pp", type=int, default=1)
    parser.add_argument("--tp", type=int, default=1)
    args = parser.parse_args()

    out_path = Path(args.out_path)

    parallel_config = ParallelismArgs(
        dp=args.dp,
        pp=args.pp,
        tp=args.tp,
        pp_engine=AllForwardAllBackwardPipelineEngine(),
        tp_mode=TensorParallelLinearMode.ALL_REDUCE,
        tp_linear_async_communication=False,
    )
    assert (
        parallel_config.tp_mode == TensorParallelLinearMode.ALL_REDUCE
        and parallel_config.tp_linear_async_communication is False
    )

    parallel_context = ParallelContext(
        data_parallel_size=parallel_config.dp,
        pipeline_parallel_size=parallel_config.pp,
        tensor_parallel_size=parallel_config.tp,
    )

    # Set log log levels
    logging_config = LoggingArgs(
        log_level="info",
        log_level_replica="info",
    )

    # Set log levels
    set_ranks_logging_level(parallel_context=parallel_context, logging_config=logging_config)

    hf_config = HFMambaConfig.from_pretrained(args.inp_path)

    dtype_str = "float32"

    # TODO(fmom): Add support for ssm_cfg
    yaml_content = f"""
    is_mamba_config: true
    d_model: {hf_config.hidden_size}
    dtype: {dtype_str}
    fused_add_norm: true
    is_mamba_config: true
    num_hidden_layers: {hf_config.num_hidden_layers}
    pad_token_id: null
    pad_vocab_size_multiple: 8
    residual_in_fp32: true
    rms_norm: true
    rms_norm_eps: 1.0e-05
    ssm_cfg: null
    vocab_size: {hf_config.vocab_size}
    """

    dtype = getattr(torch, dtype_str)
    device = torch.device("cuda")

    attrs = yaml.safe_load(yaml_content)
    model_config = MambaModelConfig(**attrs)

    model_ref = MambaForCausalLM.from_pretrained(args.inp_path)
    model_ref.to(device, dtype=dtype)
    model_ref.eval()

    nanotron_model = build_model(
        model_builder=lambda: MambaForTraining(
            config=model_config,
            parallel_context=parallel_context,
            parallel_config=parallel_config,
            random_states=None,
        ),
        parallel_context=parallel_context,
        dtype=dtype,
        device=device,
    )

    device_map = {}
    current_pp_rank = dist.get_rank(parallel_context.pp_pg)

    tied_embs_ranks = [nanotron_model.model.token_position_embeddings.rank, nanotron_model.model.lm_head.rank]

    device_map["backbone.embedding"] = (
        nanotron_model.model.token_position_embeddings.rank if current_pp_rank in tied_embs_ranks else "meta"
    )

    for i in range(model_config.num_hidden_layers):
        device_map[f"backbone.layers[{i}]"] = (
            nanotron_model.model.decoder[i].rank if current_pp_rank == nanotron_model.model.decoder[i].rank else "meta"
        )

    device_map["lm_head"] = nanotron_model.model.lm_head.rank if current_pp_rank in tied_embs_ranks else "meta"

    # Get mapping of Nanotron layer to HF layer
    nanotron_to_hf = {}

    # Static mappings
    nanotron_to_hf["token_position_embeddings.pp_block.token_embedding.weight"] = "backbone.embeddings.weight"
    nanotron_to_hf["final_layer_norm.pp_block.weight"] = "backbone.norm_f.weight"
    nanotron_to_hf["lm_head.pp_block.weight"] = "lm_head.weight"

    # Dynamic mappings within a loop
    for i in range(model_config.num_hidden_layers):
        nanotron_to_hf[f"decoder.{i}.pp_block.mixer.A_log"] = f"backbone.layers.{i}.mixer.A_log"
        nanotron_to_hf[f"decoder.{i}.pp_block.mixer.D"] = f"backbone.layers.{i}.mixer.D"
        nanotron_to_hf[f"decoder.{i}.pp_block.mixer.in_proj.weight"] = f"backbone.layers.{i}.mixer.in_proj.weight"
        nanotron_to_hf[f"decoder.{i}.pp_block.mixer.conv1d.weight"] = f"backbone.layers.{i}.mixer.conv1d.weight"
        nanotron_to_hf[f"decoder.{i}.pp_block.mixer.conv1d.bias"] = f"backbone.layers.{i}.mixer.conv1d.bias"
        nanotron_to_hf[f"decoder.{i}.pp_block.mixer.x_proj.weight"] = f"backbone.layers.{i}.mixer.x_proj.weight"
        nanotron_to_hf[f"decoder.{i}.pp_block.mixer.x_proj.bias"] = f"backbone.layers.{i}.mixer.x_proj.bias"
        nanotron_to_hf[f"decoder.{i}.pp_block.mixer.dt_proj.weight"] = f"backbone.layers.{i}.mixer.dt_proj.weight"
        nanotron_to_hf[f"decoder.{i}.pp_block.mixer.dt_proj.bias"] = f"backbone.layers.{i}.mixer.dt_proj.bias"
        nanotron_to_hf[f"decoder.{i}.pp_block.mixer.out_proj.weight"] = f"backbone.layers.{i}.mixer.out_proj.weight"
        nanotron_to_hf[f"decoder.{i}.pp_block.mixer.out_proj.bias"] = f"backbone.layers.{i}.mixer.out_proj.bias"
        nanotron_to_hf[f"decoder.{i}.pp_block.norm.weight"] = f"backbone.layers.{i}.norm.weight"

    # Sync weights
    ref_state_dict = model_ref.state_dict()
    for name, param in tqdm(
        nanotron_model.model.named_parameters(),
        total=len(list(nanotron_model.model.named_parameters())),
        desc="Converting",
    ):
        param_is_tp_sharded = (
            isinstance(param, NanotronParameter)
            and param.is_sharded
            and parallel_context.world_ranks_to_pg[param.get_sharded_info().global_ranks] == parallel_context.tp_pg
        )

        ref_param = get_weight_from_hf(
            name=name,
            ref_module_state_dict=ref_state_dict,
            ref_module=model_ref,
            nanotron_to_hf=nanotron_to_hf,
            param_is_tp_sharded=param_is_tp_sharded,
        )

        if param_is_tp_sharded:
            sharded_info = param.get_sharded_info()
            # copy param data (not just the reference)
            with torch.no_grad():
                for local_global_slices_pair in sharded_info.local_global_slices_pairs:
                    local_slices = local_global_slices_pair.local_slices
                    global_slices = local_global_slices_pair.global_slices
                    param[local_slices].copy_(ref_param[global_slices])
        else:
            assert (
                ref_param.shape == param.shape
            ), f"Parameter shape don't match for {name}\n{ref_param.shape} != {param.shape}"
            # copy param data (not just the reference)
            with torch.no_grad():
                param.copy_(ref_param)
                ref_param = None
                torch.cuda.empty_cache()

    # Marks parameters as NanotronParameters
    mark_tied_parameters(model=nanotron_model, parallel_context=parallel_context)

    sanity_check(root_module=nanotron_model)

    save_weights(model=nanotron_model, parallel_context=parallel_context, root_folder=out_path)
    checkpoint_metadata = {
        "last_train_step": 0,
        "consumed_train_samples": 0,
    }
    save_meta(root_folder=out_path, parallel_context=parallel_context, checkpoint_metadata=checkpoint_metadata)

    if dist.get_rank() == 0:
        with open(out_path / "config.yaml", "w") as f:
            config = MambaConfig(
                general=GeneralArgs(project="test", run="mamba"),
                parallelism=parallel_config,
                model=ModelArgs(
                    init_method=MambaInit(),
                    model_config=model_config,
                ),
                tokenizer=TokenizerArgs(args.inp_path),
            )
            log_rank("Saving config ...", logger=logger, level=logging.INFO, rank=0)
            yaml.dump(config.as_dict(), f)

        with open(out_path / "model_config.json", "w") as f:
            log_rank("Saving model config ...", logger=logger, level=logging.INFO, rank=0)
            json.dump(asdict(model_config), f)
