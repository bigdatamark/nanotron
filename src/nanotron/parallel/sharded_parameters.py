# [MODULE DOC]
# Purpose: create NanotronParameters carrying sharding metadata (which slices of the unsharded tensor each rank
# owns), used for tensor-parallel layers and checkpoint (de)serialization.
# Main contents: SplitConfig, create_sharded_parameter, create_sharded_parameter_from_config,
# mark_all_parameters_in_module_as_sharded.
# Import-time side effects: None.
import dataclasses
from typing import List, Optional, Tuple

import numpy as np
from torch import nn

from nanotron import distributed as dist
from nanotron.parallel.parameters import NanotronParameter, SlicesPair


@dataclasses.dataclass
class SplitConfig:
    # [CLASS DOC]
    # Purpose: dataclass describing how a parameter is split across a process group.
    # Attributes: split_dim (int dimension that is sharded); contiguous_chunks (optional tuple of chunk sizes along
    # the unsharded split_dim, each chunk sharded evenly and independently, e.g. fused QKV; None means one chunk).
    split_dim: int
    # contiguous_chunks is a tuple of chunk sizes along the split_dim
    # sharding happens inside each chunk
    # if None, by default contiguous_chunks = (len(unsharded_param.shape[split_dim]),)
    contiguous_chunks: Optional[Tuple[int, ...]] = None


def create_sharded_parameter(
    parameter: nn.Parameter,
    global_ranks: Tuple[int, ...],
    local_global_slices_pairs: Tuple[SlicesPair, ...],
    unsharded_shape: Tuple[int, ...],
) -> NanotronParameter:
    # [FUNCTION DOC]
    # Behavior: wraps parameter as a NanotronParameter if needed (sharing storage) and marks it as sharded.
    # Inputs: parameter (nn.Parameter), global_ranks (tuple of ints), local_global_slices_pairs (tuple of
    # SlicesPair), unsharded_shape (tuple of ints).
    # Outputs: NanotronParameter.
    # Exceptions: ValueError if already marked sharded. Side effects: mutates the parameter's metadata.
    if not isinstance(parameter, NanotronParameter):
        parameter = NanotronParameter(tensor=parameter)
    parameter.mark_as_sharded(
        global_ranks=global_ranks,
        local_global_slices_pairs=local_global_slices_pairs,
        unsharded_shape=unsharded_shape,
    )
    return parameter


def create_sharded_parameter_from_config(
    parameter: nn.Parameter,
    pg: dist.ProcessGroup,
    split_config: SplitConfig,
) -> NanotronParameter:
    # [FUNCTION DOC]
    # Behavior: with rank r in pg of size n and local length L on split_dim: if no chunks, the local tensor maps to
    # global range r*L to (r+1)*L on split_dim and the unsharded size is n*L. With chunks (global sizes c_i at global
    # offsets G_i, local offsets G_i/n), each chunk must be divisible by n and contributes a pair mapping local range
    # G_i/n to G_(i+1)/n onto global range G_i + r*c_i/n to G_i + (r+1)*c_i/n; unsharded size on split_dim is the
    # total of chunks. Other dims use full slices. Then calls create_sharded_parameter with pg's global ranks.
    # Inputs: parameter (local shard), pg (ProcessGroup), split_config (SplitConfig).
    # Outputs: NanotronParameter.
    # Exceptions: AssertionError if split_dim out of range or a chunk is not divisible by group size; ValueError from
    # zip strict on length mismatch (never in practice) or if already sharded.
    # Side effects: metadata mutation.
    current_rank = dist.get_rank(pg)
    param_num_dims = len(parameter.shape)
    global_ranks = dist.get_global_ranks(pg)
    split_dim = split_config.split_dim
    assert split_dim < param_num_dims
    contiguous_chunks = split_config.contiguous_chunks

    if contiguous_chunks is None:
        # we are assuming that the parameter is contiguous along the split_dim, i.e. 1 whole chunk
        # all parameters are equally shardable across the process group along the split_dim
        shard_length = parameter.shape[split_dim]
        global_slice = slice(current_rank * shard_length, (current_rank + 1) * shard_length)
        # construct a mapping from local slices to global slices, multi-dimensional version
        local_slices = tuple(slice(None) for _ in range(param_num_dims))
        global_slices = tuple(global_slice if dim_id == split_dim else slice(None) for dim_id in range(param_num_dims))
        local_global_slices_pairs = (SlicesPair(local_slices=local_slices, global_slices=global_slices),)
        unsharded_shape = tuple(
            pg.size() * param_dim_size if dim_id == split_dim else param_dim_size
            for dim_id, param_dim_size in enumerate(parameter.shape)
        )
    else:
        # support custom contiguous chunk size for sharding each along the split_dim
        local_global_slices_pairs: List[SlicesPair] = []
        chunks_global_offset = np.cumsum((0,) + contiguous_chunks)
        chunks_local_offset = chunks_global_offset // pg.size()
        for chunk, chunk_global_start, chunk_local_start, chunk_local_end in zip(
            contiguous_chunks,
            chunks_global_offset[:-1],
            chunks_local_offset[:-1],
            chunks_local_offset[1:],
            strict=True,
        ):
            # we assume that we are doing equal split at the chunk level
            assert chunk % pg.size() == 0, f"chunk size {chunk} must be divisible by process group size {pg.size()}"
            shard_length = chunk // pg.size()
            # we have: chunk_local_end = chunk_local_start + shard_length
            local_slice = slice(chunk_local_start, chunk_local_end)
            global_slice = slice(
                current_rank * shard_length + chunk_global_start,
                (current_rank + 1) * shard_length + chunk_global_start,
            )
            local_slices = tuple(
                local_slice if dim_id == split_dim else slice(None) for dim_id in range(param_num_dims)
            )
            global_slices = tuple(
                global_slice if dim_id == split_dim else slice(None) for dim_id in range(param_num_dims)
            )
            local_global_slices_pairs.append(SlicesPair(local_slices=local_slices, global_slices=global_slices))
        local_global_slices_pairs: Tuple[SlicesPair, ...] = tuple(local_global_slices_pairs)
        unsharded_shape = tuple(
            chunks_global_offset[-1] if dim_id == split_dim else param_dim_size
            for dim_id, param_dim_size in enumerate(parameter.shape)
        )

    return create_sharded_parameter(
        parameter=parameter,
        global_ranks=global_ranks,
        local_global_slices_pairs=local_global_slices_pairs,
        unsharded_shape=unsharded_shape,
    )


def mark_all_parameters_in_module_as_sharded(module: nn.Module, pg: dist.ProcessGroup, split_config: SplitConfig):
    # [FUNCTION DOC]
    # Behavior: for every submodule, replaces each direct parameter with its sharded NanotronParameter version
    # (created from split_config over pg) via setattr.
    # Inputs: module (nn.Module), pg (ProcessGroup), split_config (SplitConfig). Outputs: None.
    # Exceptions: as create_sharded_parameter_from_config.
    # Side effects: mutates module parameters (new Parameter objects for non-Nanotron params).
    # KNOWN ISSUE: shared parameters reachable from several submodules get marked multiple times and raise ValueError on the second marking.
    """
    Mark parameters as sharded within a module. We assume that parameters are equally shardable across the process group.

    :param module: nn.Module
    :param pg: dist.ProcessGroup
    :param split_config: SplitConfig
    :return:
    """

    for module_name, submodule in module.named_modules():
        for param_name, param in list(submodule.named_parameters(recurse=False)):
            new_param = create_sharded_parameter_from_config(parameter=param, pg=pg, split_config=split_config)
            setattr(submodule, param_name, new_param)
