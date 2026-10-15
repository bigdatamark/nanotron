# [MODULE DOC]
# Purpose: Save and load model weights as one safetensors file per (parameter, shard), with
# TensorMetadata in the file header so that weights can be reloaded under a different TP topology.
# Main contents: save_weights, CheckpointVersionFromShardFileException, version readers
# (read_checkpoint_version_from_shard_file, read_checkpoint_version_from_meta, get_checkpoint_version),
# load_sharded_param_latest, load_weights, get_checkpoint_paths_list.
# Fits in nanotron: called by serialize.main.save and by the trainer/model loading code; load_weights
# returns per-parameter shard metadata consumed by serialize.optimizer.load_optimizer.
# Import-time side effects: creates a module logger.
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import dacite
import torch
from packaging.version import Version
from safetensors.torch import safe_open, save_file
from torch import nn
from tqdm import tqdm

from nanotron import distributed as dist
from nanotron import logging
from nanotron.constants import CHECKPOINT_VERSION
from nanotron.distributed import get_global_rank
from nanotron.logging import log_rank
from nanotron.parallel import ParallelContext
from nanotron.parallel.parameters import NanotronParameter, ShardedInfo, SlicesPair
from nanotron.serialize.metadata import CheckpointMetadata, TensorMetadata, load_meta
from nanotron.serialize.utils import (
    ObjectType,
    extract_tp_pp_rank_from_shard_path,
    get_exp_tp_pp_rank_and_size_from,
    get_path,
    merge_and_shard_tp_tensors,
)

logger = logging.get_logger(__name__)


def save_weights(model: nn.Module, parallel_context: ParallelContext, root_folder: Path):
    # [FUNCTION DOC]
    # Behavior: only dp_cp rank 0 saves. Builds a module-id-to-prefix map, then for each state_dict entry:
    # on expert ranks other than 0 skips non-expert tensors; requires a NanotronParameter; for tied params
    # uses the tied full name and only lets group rank 0 of the tied group save; for sharded params derives
    # expert/tp/pp rank info and header metadata (slices pairs and unsharded shape); builds the file path
    # with get_path under root_folder/model, creates directories and writes the tensor under key "data"
    # (logging and re-raising on error).
    # Inputs: model (nn.Module), parallel_context, root_folder (checkpoint Path).
    # Outputs: None.
    # Exceptions: NotImplementedError for buffers or non-NanotronParameter entries; re-raised save errors.
    # Side effects: directory creation and file writes; progress bar; logging.
    root_folder = root_folder / "model"

    # We save only `dist.get_rank(parallel_context.dp_cp_pg) == 0`
    if dist.get_rank(parallel_context.dp_cp_pg) != 0:
        return

    module_id_to_prefix = {id(module): f"{module_name}." for module_name, module in model.named_modules()}
    # Fix the root_model
    module_id_to_prefix[id(model)] = ""

    # We chunk everything by `tp_world_size` in order to make sure that we gather all the weights into a single device before saving it
    for name, param_or_buffer in tqdm(model.state_dict().items(), desc="Saving weights"):

        # exp_rank=0 saves all weights whereas exp_rank>0 save only MLP weights
        if dist.get_rank(parallel_context.ep_pg) != 0:
            if "experts" not in name:
                continue

        # `state_dict` doesn't return a Param or a buffer, just a tensors which loses some metadata
        try:
            param = model.get_parameter(name)
        except AttributeError:
            # TODO @nouamanetazi: Handle buffers
            param = None

        if isinstance(param, NanotronParameter):
            metadata = {}
            if param.is_tied:
                tied_info = param.get_tied_info()
                base_name = tied_info.get_full_name_from_module_id_to_prefix(module_id_to_prefix=module_id_to_prefix)
                group_ranks = tied_info.global_ranks
                group = parallel_context.world_ranks_to_pg[group_ranks]
                # Only the first rank of the group of the tied weights saves weights
                # TODO @thomasw21: We could rotate in order to balance the load.
                if dist.get_rank(group) != 0:
                    continue
            else:
                base_name = name

            if param.is_sharded:
                sharded_info: ShardedInfo = param.get_sharded_info()
                group = parallel_context.world_ranks_to_pg[sharded_info.global_ranks]
                exp_tp_pp_rank_and_size = get_exp_tp_pp_rank_and_size_from(
                    world_rank=get_global_rank(group=group, group_rank=dist.get_rank(group)),
                    parallel_context=parallel_context,
                )
                is_expert_sharded = sharded_info.is_expert_sharded(parallel_context)
                metadata = TensorMetadata(
                    version=CHECKPOINT_VERSION,
                    local_global_slices_pairs=sharded_info.local_global_slices_pairs,
                    unsharded_shape=sharded_info.unsharded_shape,
                ).to_str_dict()

            else:
                exp_tp_pp_rank_and_size = None
                is_expert_sharded = False

            path = get_path(
                base_name,
                type=ObjectType.MODEL,
                exp_tp_pp_rank_and_size=exp_tp_pp_rank_and_size,
                is_expert_sharded=is_expert_sharded,
                prefix=root_folder,
            )
            path.parent.mkdir(exist_ok=True, parents=True)
            try:
                tensors = {"data": param_or_buffer}
                save_file(tensors=tensors, filename=path, metadata=metadata)
            except Exception as e:
                log_rank(
                    f"Error saving {path} with {metadata}",
                    logger=logger,
                    level=logging.ERROR,
                    rank=0,
                )
                raise e
        else:
            raise NotImplementedError("Parameters are required to be NanotronParameter")


class CheckpointVersionFromShardFileException(Exception):
    # [CLASS DOC]
    # Purpose: Raised when a shard file header lacks valid TensorMetadata (older checkpoints), signalling fallback to the meta file.
    """Raise when loading checkpoint version from shard file fails"""


def read_checkpoint_version_from_shard_file(param_save_path: Path) -> Version:
    # [FUNCTION DOC]
    # Behavior: opens the safetensors file on CPU, parses its header metadata as TensorMetadata and returns its version.
    # Inputs: param_save_path (Path). Outputs: Version.
    # Exceptions: CheckpointVersionFromShardFileException on missing or unexpected metadata keys. Side effects: file read.
    try:
        with safe_open(param_save_path, framework="pt", device=str("cpu")) as fi:
            param_metadata = fi.metadata()
            param_metadata = TensorMetadata.from_str_dict(param_metadata)
            checkpoint_version = param_metadata.version
    except (dacite.exceptions.MissingValueError, dacite.exceptions.UnexpectedDataError):
        raise CheckpointVersionFromShardFileException()
    return checkpoint_version


def read_checkpoint_version_from_meta(parallel_context: ParallelContext, root_folder: Path) -> Version:
    # [FUNCTION DOC]
    # Behavior: loads checkpoint metadata and returns its version.
    # Inputs: parallel_context, root_folder. Outputs: Version.
    # Exceptions: see load_meta. Side effects: file read.
    checkpoint_metadata: CheckpointMetadata = load_meta(parallel_context=parallel_context, root_folder=root_folder)
    checkpoint_version = checkpoint_metadata.version
    return checkpoint_version


def get_checkpoint_version(parallel_context, root_folder, param_save_path: Path) -> Version:
    # [FUNCTION DOC]
    # Behavior: tries the shard file header; on failure logs an error on rank 0 and reads the meta file.
    # Inputs: parallel_context, root_folder, param_save_path. Outputs: Version.
    # Exceptions: propagates meta read errors. Side effects: file reads; logging.
    try:
        checkpoint_version = read_checkpoint_version_from_shard_file(param_save_path=param_save_path)
    except CheckpointVersionFromShardFileException:
        log_rank(
            f"Failed to read checkpoint version from shard file {param_save_path}, reading from meta file.",
            logger=logger,
            level=logging.ERROR,
            rank=0,
        )
        checkpoint_version = read_checkpoint_version_from_meta(
            parallel_context=parallel_context, root_folder=root_folder
        )
    return checkpoint_version


def load_sharded_param_latest(
    param_or_buffer: torch.Tensor,
    sharded_info: ShardedInfo,
    shards_path: List[Path],
    param_shard_metadata: Optional[Dict] = None,
):
    # [FUNCTION DOC]
    # Behavior: for each shard file, reads its tensor and header metadata, collects (tensor, slices pairs),
    # asserts all shards agree on the unsharded shape, and records the metadata in param_shard_metadata keyed
    # by the (pp_rank, tp_rank) strings parsed from the path; then allocates an unsharded tensor and fills
    # param_or_buffer in place with this rank's slice via merge_and_shard_tp_tensors.
    # Inputs: param_or_buffer (destination tensor, current local shard), sharded_info (ShardedInfo of the
    # current parameter), shards_path (list of shard file paths), param_shard_metadata (optional dictionary to fill).
    # Outputs: param_shard_metadata (possibly None).
    # Exceptions: AssertionError on shape disagreement or if no shards were given.
    # Side effects: file reads; mutates param_or_buffer and param_shard_metadata; allocates a full-size tensor.
    # KNOWN ISSUE: the unsharded buffer is allocated with the default dtype (float32) rather than the parameter's dtype, wasting memory.
    checkpoint_unsharded_shape = None
    shards_and_slices_maps: List[Tuple[torch.Tensor, Tuple[SlicesPair, ...]]] = []

    for shard_path in shards_path:
        with safe_open(shard_path, framework="pt", device=str(param_or_buffer.device)) as fi:
            # TODO @thomasw21: Choose only a slice if we switch the TP topology
            param_metadata = fi.metadata()
            param_metadata = TensorMetadata.from_str_dict(param_metadata)
            shards_and_slices_maps.append((fi.get_tensor("data"), param_metadata.local_global_slices_pairs))

            if checkpoint_unsharded_shape is None:
                checkpoint_unsharded_shape = param_metadata.unsharded_shape
            else:
                assert checkpoint_unsharded_shape == param_metadata.unsharded_shape

            if param_shard_metadata is not None:
                # NOTE: store how does model parameter are sharded
                # so that we can shard optimizer checkpoints in this way
                pp_rank, tp_rank = extract_tp_pp_rank_from_shard_path(shard_path)
                param_shard_metadata[(pp_rank, tp_rank)] = param_metadata

    assert checkpoint_unsharded_shape is not None
    # TODO @thomasw21: Interestingly enough we don't actually need to instantiate the entire model at all.
    unsharded_tensor = torch.empty(checkpoint_unsharded_shape, device=param_or_buffer.device)

    merge_and_shard_tp_tensors(
        buffer=param_or_buffer,
        unsharded_buffer=unsharded_tensor,
        shards_and_slices_maps=shards_and_slices_maps,
        shard_metadata=sharded_info,
    )

    return param_shard_metadata


def load_weights(
    model: nn.Module,
    parallel_context: ParallelContext,
    root_folder: Path,
    filtered_state_dict: Optional[Dict[str, Any]] = None,
):
    # [FUNCTION DOC]
    # Behavior: for every entry of filtered_state_dict (default the model's state_dict): requires a
    # NanotronParameter; resolves the base name (tied full name if tied); for sharded params computes the
    # rank info used at save time (group rank 0 of the tied group for tied params, own rank in the shard
    # group otherwise); builds the expected path. If it exists, copies its "data" tensor in place. If its
    # folder does not exist, raises. Otherwise assumes a topology change: requires the param to be sharded,
    # globs all shard files for that name, determines and checks the checkpoint version consistency, and
    # loads via load_sharded_param_latest while recording old shard metadata.
    # Inputs: model, parallel_context, root_folder (checkpoint Path), filtered_state_dict (optional dictionary of name to tensor).
    # Outputs: dictionary from parameter name to a dictionary from (pp_rank, tp_rank) string pair to TensorMetadata (empty for params loaded directly).
    # Exceptions: ValueError for missing folders/shards, non-sharded missing params or unsupported newer
    # version; AssertionError on version mismatch; NotImplementedError for non-NanotronParameter entries.
    # Side effects: file reads; in-place parameter writes; progress bar on world rank 0; logging.
    """Load weights from a checkpoint

    Args:
        model: model to load weights into
        parallel_context: distributed process groups
        root_folder: root folder of the checkpoint
        filtered_state_dict: state dict to load from (overrides model.state_dict()). if None, load from model.state_dict()
    """
    param_root_folder = root_folder / "model"

    module_id_to_prefix = {id(module): f"{module_name}." for module_name, module in model.named_modules()}
    # Fix the root_model
    module_id_to_prefix[id(model)] = ""

    checkpoint_version: Optional[Version] = None

    filtered_state_dict = filtered_state_dict if filtered_state_dict is not None else model.state_dict()
    param_shard_metadata = {}
    for name, param_or_buffer in tqdm(
        filtered_state_dict.items(), disable=dist.get_rank(parallel_context.world_pg) != 0, desc="Loading weights"
    ):
        # NOTE: extract how does the current model parameter are sharded
        # so that we can load optimizer checkpoints in this way
        param_shard_metadata[name] = {}
        # `state_dict` doesn't return a Param or a buffer, just a tensors which loses some metadata
        try:
            param = model.get_parameter(name)
        except AttributeError:
            param = None

        if isinstance(param, NanotronParameter):
            if param.is_tied:
                tied_info = param.get_tied_info()
                base_name = tied_info.get_full_name_from_module_id_to_prefix(module_id_to_prefix=module_id_to_prefix)
            else:
                base_name = name

            if param.is_sharded:
                sharded_info = param.get_sharded_info()

                if param.is_tied:
                    # When params are tied only the first rank of tied param group stores weights (see save_weights)
                    group = parallel_context.world_ranks_to_pg[tied_info.global_ranks]
                    group_rank = 0
                else:
                    group = parallel_context.world_ranks_to_pg[sharded_info.global_ranks]
                    group_rank = dist.get_rank(group)

                exp_tp_pp_rank_and_size = get_exp_tp_pp_rank_and_size_from(
                    world_rank=get_global_rank(group=group, group_rank=group_rank), parallel_context=parallel_context
                )
                # TODO @nouamane: do we consider exp_size=1 expert_sharded?
                is_expert_sharded = sharded_info.is_expert_sharded(parallel_context)
            else:
                exp_tp_pp_rank_and_size = None
                is_expert_sharded = False

            path = get_path(
                base_name,
                type=ObjectType.MODEL,
                exp_tp_pp_rank_and_size=exp_tp_pp_rank_and_size,
                prefix=param_root_folder,
                is_expert_sharded=is_expert_sharded,
            )

            if path.exists():
                with safe_open(path, framework="pt", device=str(param.device)) as fi:
                    # TODO @thomasw21: Choose only a slice if we switch the TP topology
                    param_or_buffer[:] = fi.get_tensor("data")

            elif not path.parent.exists():
                raise ValueError(
                    f"Checkpoint is empty or checkpoint structure is not matching the model architecture."
                    f"Couldn't find folder {path.parent} in checkpoint at {root_folder}"
                )
            else:
                # Let's assume that the topology changed and the param is sharded.
                # We search for all the files from the shards, concatenate the "unsharded" tensor
                # and load the specific shard we're interested in.
                if not param.is_sharded:
                    raise ValueError(
                        f"`{name}` is not a sharded parameter. It's possible you were expecting {path} to exist."
                    )
                # TODO @thomasw21: Make so that we don't need to code this logic somewhere else than in `get_path`
                sharded_info = param.get_sharded_info()
                suffix = base_name.rsplit(".", 1)[-1]
                shards_path = list(path.parent.glob(f"{ObjectType.MODEL.value}_{suffix}*.safetensors"))
                if len(shards_path) <= 0:
                    raise ValueError(
                        f"Could not find any shards {ObjectType.MODEL.value}_{suffix}*.safetensors in {path.parent}."
                        f"If you notice `.safetensors` in the middle of the name of some of the checkpoints files. You need to run `scripts/fix_checkpoint_bad_naming.py`."
                    )

                if checkpoint_version is None:
                    checkpoint_version = get_checkpoint_version(
                        parallel_context, root_folder, param_save_path=shards_path[0]
                    )
                else:
                    current_checkpoint_version = None
                    try:
                        current_checkpoint_version = read_checkpoint_version_from_shard_file(
                            param_save_path=shards_path[0]
                        )
                    except CheckpointVersionFromShardFileException:
                        # The checkpoint version is read from the meta file
                        current_checkpoint_version = checkpoint_version
                    finally:
                        assert (
                            current_checkpoint_version == checkpoint_version
                        ), f"Checkpoint version mismatch at {shards_path[0]}."

                if checkpoint_version <= CHECKPOINT_VERSION:
                    load_sharded_param_latest(
                        param_or_buffer=param_or_buffer,
                        sharded_info=sharded_info,
                        shards_path=shards_path,
                        param_shard_metadata=param_shard_metadata[name],
                    )
                else:
                    raise ValueError(f"Unsupported checkpoint version {checkpoint_version}")

        else:
            raise NotImplementedError(f"Parameters {param} should be a NanotronParameter")

    return param_shard_metadata


def get_checkpoint_paths_list(
    model: nn.Module,
    parallel_context: ParallelContext,
    root_folder: Path,
    only_list_folders: bool = False,
    only_list_current_process: bool = True,
    filtered_state_dict: Optional[Dict[str, Any]] = None,
):
    # [FUNCTION DOC]
    # Behavior: intended to list the files (or folders when only_list_folders) that the current process
    # reads in a checkpoint, mirroring the path logic of load_weights.
    # Inputs: model, parallel_context, root_folder, only_list_folders (default False),
    # only_list_current_process (default True), filtered_state_dict (optional).
    # Outputs: list of Paths.
    # Exceptions: see known issues. Side effects: progress bar.
    # KNOWN ISSUE: iterates over the state dict values (tensors) and uses them as names, so get_parameter always fails and nothing is listed (or AttributeError when only_list_current_process is False since param is None).
    # KNOWN ISSUE: calls get_path without the required is_expert_sharded argument (TypeError), and passes a list to joinpath for folders.
    """Return the list of all the files or folders created/accessed by the current process in a checkpoint

    Args:
        model: model to load weights into
        parallel_context: distributed process groups
        root_folder: root folder of the checkpoint
        filtered_state_dict: state dict to load from (overrides model.state_dict()). if None, load from model.state_dict()
    """
    param_root_folder = root_folder / "model"

    module_id_to_prefix = {id(module): f"{module_name}." for module_name, module in model.named_modules()}
    # Fix the root_model
    module_id_to_prefix[id(model)] = ""

    paths = []

    filtered_state_dict = filtered_state_dict if filtered_state_dict is not None else model.state_dict()
    for name in tqdm(
        filtered_state_dict.values(),
        disable=dist.get_rank(parallel_context.world_pg) != 0,
        desc="Listing checkpoint paths",
    ):
        # `state_dict` doesn't return a Param or a buffer, just a tensors which loses some metadata
        try:
            param = model.get_parameter(name)
        except AttributeError:
            param = None

        if isinstance(param, NanotronParameter) or not only_list_current_process:
            if param.is_tied:
                tied_info = param.get_tied_info()
                base_name = tied_info.get_full_name_from_module_id_to_prefix(module_id_to_prefix=module_id_to_prefix)
            else:
                base_name = name

            if param.is_sharded:
                sharded_info = param.get_sharded_info()

                if param.is_tied:
                    # When params are tied only the first rank of tied param group stores weights (see save_weights)
                    group = parallel_context.world_ranks_to_pg[tied_info.global_ranks]
                    group_rank = 0
                else:
                    group = parallel_context.world_ranks_to_pg[sharded_info.global_ranks]
                    group_rank = dist.get_rank(group)

                exp_tp_pp_rank_and_size = get_exp_tp_pp_rank_and_size_from(
                    world_rank=get_global_rank(group=group, group_rank=group_rank), parallel_context=parallel_context
                )
            else:
                exp_tp_pp_rank_and_size = None

            if only_list_folders:
                paths.append(param_root_folder.joinpath(base_name.split(".")[:-1]))
            else:
                paths.append(
                    get_path(
                        base_name,
                        type=ObjectType.MODEL,
                        exp_tp_pp_rank_and_size=exp_tp_pp_rank_and_size,
                        prefix=param_root_folder,
                    )
                )

    return paths
