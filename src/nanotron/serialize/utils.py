# [MODULE DOC]
# Purpose: Helpers for checkpoint file naming and for re-sharding tensors across TP topologies.
# Main contents: ObjectType enum, get_exp_tp_pp_rank_and_size_from, get_path,
# extract_tp_pp_rank_from_shard_path, merge_and_shard_tp_tensors.
# Fits in nanotron: used by serialize.weights and serialize.optimizer to name per-shard safetensors
# files and to load checkpoints saved with a different TP size.
# Import-time side effects: None.
import re
from enum import Enum
from pathlib import Path
from typing import List, Optional, Tuple

import torch

from nanotron.parallel import ParallelContext
from nanotron.parallel.parameters import SlicesPair
from nanotron.serialize.metadata import TensorMetadata


class ObjectType(Enum):
    # [CLASS DOC]
    # Purpose: Enum of checkpointed object kinds used as file name prefixes: model, optimizer, lr_scheduler.
    MODEL = "model"
    OPTIMIZER = "optimizer"
    LR_SCHEDULER = "lr_scheduler"


def get_exp_tp_pp_rank_and_size_from(
    world_rank: int, parallel_context: ParallelContext
) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    # [FUNCTION DOC]
    # Behavior: looks up a world rank's local ranks and returns (rank, size) pairs for expert, tensor and pipeline parallelism.
    # Inputs: world_rank (int), parallel_context.
    # Outputs: tuple of three (rank, size) pairs: expert, tp, pp.
    # Exceptions: None explicitly raised. Side effects: None.
    # KNOWN ISSUE: the return annotation lists only two pairs while three are returned.
    result = parallel_context.get_local_ranks(world_rank=world_rank)
    return (
        (result["ep"], parallel_context.ep_pg.size()),
        (result["tp"], parallel_context.tp_pg.size()),
        (result["pp"], parallel_context.pp_pg.size()),
    )


def get_path(
    tensor_name: str,
    type: ObjectType,
    exp_tp_pp_rank_and_size: Tuple[Tuple[int, int], Tuple[int, int]],
    is_expert_sharded: bool,
    prefix: Optional[Path] = None,
) -> List[str]:
    # [FUNCTION DOC]
    # Behavior: splits tensor_name on dots; the leading parts become directories and the last part becomes
    # a file name prefixed with the object type. When rank info is given, appends pp and tp rank/size, plus
    # expert rank/size only for expert-sharded tensors with expert size above 1; always ends with
    # .safetensors. Returns path parts, or a Path joined onto prefix if given.
    # Inputs: tensor_name (dotted str), type (ObjectType), exp_tp_pp_rank_and_size (triple of pairs or
    # None/empty), is_expert_sharded (bool), prefix (optional Path).
    # Outputs: list of path components, or a Path when prefix is given.
    # Exceptions: ValueError if the rank info does not unpack. Side effects: None.
    # KNOWN ISSUE: annotated as returning a list of strings but returns a Path when prefix is set.
    suffix = tensor_name.split(".")
    suffix_path, suffix_name = suffix[:-1], suffix[-1]

    if exp_tp_pp_rank_and_size:
        # We always show pp_rank and tp_rank if `exp_tp_pp_rank_and_size` is provided
        (exp_rank, exp_size), (tp_rank, tp_size), (pp_rank, pp_size) = exp_tp_pp_rank_and_size
        if not is_expert_sharded or exp_size == 1:
            suffix_name = (
                f"{type.value}_{suffix_name}_pp-rank-{pp_rank}-of-{pp_size}_tp-rank-{tp_rank}-of-{tp_size}.safetensors"
            )
        else:
            # We only show exp_rank if tensor is exp_sharded and exp_size > 1
            suffix_name = f"{type.value}_{suffix_name}_pp-rank-{pp_rank}-of-{pp_size}_tp-rank-{tp_rank}-of-{tp_size}_exp-rank-{exp_rank}-of-{exp_size}.safetensors"
    else:
        suffix_name = f"{type.value}_{suffix_name}.safetensors"

    suffix_path.append(suffix_name)
    if prefix is None:
        return suffix_path
    else:
        return prefix.joinpath(*suffix_path)


def extract_tp_pp_rank_from_shard_path(shard_path: Path):
    # [FUNCTION DOC]
    # Behavior: regex-searches the pp and tp rank numbers in a shard file path.
    # Inputs: shard_path (Path). Outputs: pair (pp_rank, tp_rank) as strings.
    # Exceptions: AttributeError if the pattern is absent. Side effects: None.
    pattern = r"pp-rank-(\d+)-of-\d+_tp-rank-(\d+)-of-\d+"
    match = re.search(pattern, str(shard_path))
    pp_rank, tp_rank = match.groups()
    return pp_rank, tp_rank


def merge_and_shard_tp_tensors(
    buffer: torch.Tensor,
    unsharded_buffer: torch.Tensor,
    shards_and_slices_maps: List[Tuple[torch.Tensor, Tuple[SlicesPair, ...]]],
    shard_metadata: TensorMetadata,
) -> torch.Tensor:
    # [FUNCTION DOC]
    # Behavior: reconstructs the unsharded tensor by copying each saved shard's local slices into their
    # global slices of unsharded_buffer, then fills buffer (the current rank's shard) from the global
    # slices described by shard_metadata.
    # Inputs: buffer (destination local shard tensor), unsharded_buffer (scratch tensor of full shape),
    # shards_and_slices_maps (list of pairs: saved shard tensor, tuple of SlicesPair),
    # shard_metadata (TensorMetadata of the current shard).
    # Outputs: buffer.
    # Exceptions: None explicitly raised (shape mismatches propagate). Side effects: mutates buffer and unsharded_buffer.
    for shard, slices_pairs in shards_and_slices_maps:
        for slices_pair in slices_pairs:
            local_slices = slices_pair.local_slices
            global_slices = slices_pair.global_slices
            unsharded_buffer[global_slices] = shard[local_slices]

    for slices_pair in shard_metadata.local_global_slices_pairs:
        local_slices = slices_pair.local_slices
        global_slices = slices_pair.global_slices
        buffer[local_slices] = unsharded_buffer[global_slices]

    return buffer
