# [MODULE DOC]
# Purpose: Tensor-parallel (TP) building-block layers for nanotron models: column-sharded and
# row-sharded linear layers, a tied (replicated) linear layer, and a vocabulary-sharded embedding.
# Main contents: TensorParallelColumnLinear, TensorParallelRowLinear, TiedLinear, TensorParallelEmbedding.
# Fits in nanotron: model definitions (e.g. llama) build their attention/MLP/embedding/LM-head from
# these; the actual communication logic lives in tensor_parallel.functional and the differentiable
# primitives. Parameters are converted to NanotronParameter marked as sharded or tied so that
# checkpointing and gradient sync know how to handle them.
# Import-time side effects: creates a module logger only.
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from typing import Optional, Tuple

import torch
from torch import nn

from nanotron import distributed as dist
from nanotron.distributed import get_global_rank
from nanotron.logging import get_logger
from nanotron.parallel.parameters import NanotronParameter
from nanotron.parallel.sharded_parameters import (
    SplitConfig,
    create_sharded_parameter_from_config,
    mark_all_parameters_in_module_as_sharded,
)
from nanotron.parallel.tensor_parallel.distributed_differentiable_primitives import (
    differentiable_all_gather,
    differentiable_all_reduce_sum,
    differentiable_identity,
    differentiable_reduce_scatter_sum,
)
from nanotron.parallel.tensor_parallel.enum import TensorParallelLinearMode
from nanotron.parallel.tensor_parallel.functional import (
    column_linear,
    row_linear,
)
from nanotron.parallel.tied_parameters import create_tied_parameter

logger = get_logger(__name__)


class TensorParallelColumnLinear(nn.Linear):
    # [CLASS DOC]
    # Purpose: Linear layer whose output dimension is split across the TP process group; each rank holds
    # out_features divided by world size rows of the weight (and bias slice).
    # Key attributes: pg, world_size, in_features (full), out_features (local shard size), mode
    # (TensorParallelLinearMode ALL_REDUCE or REDUCE_SCATTER), async_communication, tp_recompute_allgather.
    # Inheritance: subclass of torch nn.Linear; weight and bias are replaced by sharded NanotronParameters
    # with split dimension 0.
    # Invariant: full out_features is divisible by the TP world size.
    def __init__(
        self,
        in_features,
        out_features,
        pg: dist.ProcessGroup,
        mode: TensorParallelLinearMode,
        bias=True,
        device=None,
        dtype=None,
        async_communication: bool = False,
        contiguous_chunks: Optional[Tuple[int, ...]] = None,
        tp_recompute_allgather: bool = True,
    ):
        # [FUNCTION DOC]
        # Behavior: stores pg and world size, checks divisibility, computes the local out_features, builds the
        # underlying nn.Linear of local size, stores mode/async flags, requires the environment variable
        # CUDA_DEVICE_MAX_CONNECTIONS to equal "1" when world size is greater than 1, validates that
        # contiguous_chunks (if given) sum to the full out_features, then marks all parameters as sharded on
        # dim 0 with that SplitConfig.
        # Inputs: in_features (int), out_features (int, unsharded), pg (TP ProcessGroup), mode
        # (TensorParallelLinearMode), bias (bool, default True), device/dtype (defaults None),
        # async_communication (bool, default False), contiguous_chunks (optional tuple of ints describing
        # logically separate blocks fused along the output dim, e.g. fused QKV), tp_recompute_allgather
        # (bool, default True; recompute the gathered input in backward instead of keeping it).
        # Outputs: None.
        # Exceptions: AssertionError on non-divisible out_features, missing env var setting, or bad chunk sum.
        # Side effects: allocates parameters; reads environment variables; replaces self.weight/self.bias.
        self.pg = pg
        self.world_size = pg.size()

        assert out_features % self.world_size == 0

        self.in_features = in_features
        self.out_features = out_features // self.world_size
        self.tp_recompute_allgather = tp_recompute_allgather

        super().__init__(
            in_features=self.in_features,
            out_features=self.out_features,
            bias=bias,
            device=device,
            dtype=dtype,
        )

        self.mode = mode
        self.async_communication = async_communication

        if self.world_size > 1:
            assert (
                os.environ.get("CUDA_DEVICE_MAX_CONNECTIONS", None) == "1"
            ), "Env variable CUDA_DEVICE_MAX_CONNECTIONS should be set to 1 when using TP>1"

        if contiguous_chunks is not None:
            assert (
                sum(contiguous_chunks) == out_features
            ), f"Sum of contiguous chunks ({sum(contiguous_chunks)}) must equal to out_features ({out_features})"
        split_config = SplitConfig(split_dim=0, contiguous_chunks=contiguous_chunks)

        mark_all_parameters_in_module_as_sharded(
            self,
            pg=self.pg,
            split_config=split_config,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # [FUNCTION DOC]
        # Behavior: delegates to column_linear with this layer's weight, bias, group, mode, async and
        # recompute flags (that function gathers/identity-copies the input, then applies the local matmul).
        # Inputs: x, input activations (sequence-sharded in REDUCE_SCATTER mode, full in ALL_REDUCE mode).
        # Outputs: tensor whose last dim is the local out_features shard.
        # Exceptions: None explicitly raised.
        # Side effects: TP collectives (all-gather in forward/backward depending on mode).
        return column_linear(
            input=x,
            weight=self.weight,
            bias=self.bias,
            group=self.pg,
            tp_mode=self.mode,
            async_communication=self.async_communication,
            tp_recompute_allgather=self.tp_recompute_allgather,
        )

    def extra_repr(self) -> str:
        # [FUNCTION DOC]
        # Behavior: returns the nn.Linear repr string augmented with tp_rank and the unsharded out_features.
        # Inputs: None. Outputs: string. Exceptions: None explicitly raised. Side effects: None.
        return f"tp_rank={dist.get_rank(self.pg)}, {super().extra_repr()}, unsharded_out_features={self.out_features * self.world_size}"


class TensorParallelRowLinear(nn.Linear):
    # [CLASS DOC]
    # Purpose: Linear layer whose input dimension is split across the TP group; each rank holds a column
    # slice of the weight and partial outputs are summed (all-reduce) or reduce-scattered.
    # Key attributes: pg, world_size, in_features (local shard), out_features (full), mode, async_communication.
    # Inheritance: subclass of nn.Linear; weight becomes a sharded NanotronParameter (split dim 1); bias,
    # only created on TP rank 0, becomes a plain (unsharded) NanotronParameter.
    # Invariant: full in_features divisible by TP world size; async communication not allowed in ALL_REDUCE mode.
    def __init__(
        self,
        in_features,
        out_features,
        pg: dist.ProcessGroup,
        mode: TensorParallelLinearMode,
        bias=True,
        device=None,
        dtype=None,
        async_communication: bool = False,
        contiguous_chunks: Optional[Tuple[int, ...]] = None,
    ):
        # [FUNCTION DOC]
        # Behavior: stores pg/world size, asserts divisibility, computes local in_features, enables bias only
        # on TP rank 0, builds the nn.Linear, stores mode/async, rejects async with ALL_REDUCE, validates
        # contiguous_chunks sum against the full in_features, and shards parameters along dim 1.
        # Inputs: in_features (unsharded int), out_features (int), pg, mode, bias (default True),
        # device/dtype (default None), async_communication (default False), contiguous_chunks (optional tuple).
        # Outputs: None.
        # Exceptions: AssertionError on divisibility or chunk sum; ValueError for async with ALL_REDUCE.
        # Side effects: allocates parameters; replaces parameters on self.
        self.pg = pg
        self.world_size = pg.size()

        assert in_features % self.world_size == 0

        self.in_features = in_features // self.world_size
        self.out_features = out_features

        # No need to shard the bias term, only rank 0 would have it
        bias = dist.get_rank(self.pg) == 0 and bias

        super().__init__(
            in_features=self.in_features,
            out_features=self.out_features,
            bias=bias,
            device=device,
            dtype=dtype,
        )
        self.mode = mode
        self.async_communication = async_communication
        if self.mode is TensorParallelLinearMode.ALL_REDUCE and self.async_communication:
            raise ValueError("async_communication is not supported for ALL_REDUCE mode")

        if contiguous_chunks is not None:
            assert (
                sum(contiguous_chunks) == in_features
            ), f"Sum of contiguous chunks ({sum(contiguous_chunks)}) must equal to in_features ({in_features})"

        split_config = SplitConfig(split_dim=1, contiguous_chunks=contiguous_chunks)

        self._mark_all_parameters_in_module_as_sharded(split_config)

    def _mark_all_parameters_in_module_as_sharded(self, split_config: SplitConfig):
        # [FUNCTION DOC]
        # HELPER: converts this layer's parameters to NanotronParameters during construction.
        # Behavior: for each named parameter, wraps "bias" as a plain NanotronParameter and every other
        # parameter as a sharded parameter built from split_config and pg; re-assigns it on self.
        # Inputs: split_config (SplitConfig, split dim 1).
        # Outputs: None. Exceptions: None explicitly raised. Side effects: mutates self's parameters.
        for name, param in list(self.named_parameters()):
            if name == "bias":
                # `bias` only exists in rank 0 because it's not sharded
                new_param = NanotronParameter(tensor=param)
            else:
                new_param = create_sharded_parameter_from_config(
                    parameter=param,
                    pg=self.pg,
                    split_config=split_config,
                )
            setattr(self, name, new_param)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # [FUNCTION DOC]
        # Behavior: delegates to row_linear (local matmul, then all-reduce or reduce-scatter of the partial
        # sums; bias added on the rank that has it).
        # Inputs: x, activations whose last dim is the local in_features shard.
        # Outputs: tensor with full out_features (sequence-sharded in REDUCE_SCATTER mode).
        # Exceptions: None explicitly raised. Side effects: TP collectives.
        return row_linear(
            input=x,
            weight=self.weight,
            bias=self.bias,
            group=self.pg,
            tp_mode=self.mode,
            async_communication=self.async_communication,
        )

    def extra_repr(self) -> str:
        # [FUNCTION DOC]
        # Behavior: nn.Linear repr plus tp_rank and unsharded in_features.
        # Inputs: None. Outputs: string. Exceptions: None explicitly raised. Side effects: None.
        return f"tp_rank={dist.get_rank(self.pg)}, {super().extra_repr()}, unsharded_in_features={self.in_features * self.world_size}"


class TiedLinear(nn.Linear):
    # [CLASS DOC]
    # Purpose: Linear layer fully replicated on every rank of the TP group, with its parameters marked as
    # tied across those ranks so their gradients can be synchronized.
    # Key attributes: pg, world_size, mode.
    # Inheritance: subclass of nn.Linear.
    def __init__(
        self,
        in_features,
        out_features,
        pg: dist.ProcessGroup,
        mode: TensorParallelLinearMode,
        bias=True,
        device=None,
        dtype=None,
    ):
        # [FUNCTION DOC]
        # Behavior: stores pg, world size and mode, builds a full-size nn.Linear, then marks all parameters as tied.
        # Inputs: in_features, out_features (ints), pg, mode, bias (default True), device/dtype (default None).
        # Outputs: None. Exceptions: None explicitly raised. Side effects: allocates parameters.
        self.pg = pg
        self.world_size = pg.size()
        self.mode = mode

        super().__init__(
            in_features=in_features,
            out_features=out_features,
            bias=bias,
            device=device,
            dtype=dtype,
        )

        self._mark_all_parameters_in_module_as_tied()

    def _mark_all_parameters_in_module_as_tied(self):
        # [FUNCTION DOC]
        # HELPER: tie marking used by the constructor.
        # Behavior: for each named parameter, creates a tied NanotronParameter whose global ranks are the
        # sorted global ranks of every member of pg, with reduce op None in ALL_REDUCE mode (gradients are
        # already identical) and SUM otherwise (each rank sees only part of the sequence); root module is self.
        # Inputs: None. Outputs: None. Exceptions: None explicitly raised.
        # Side effects: replaces self's parameters.
        for name, param in list(self.named_parameters()):
            new_param = create_tied_parameter(
                parameter=param,
                name=name,
                global_ranks=tuple(sorted((get_global_rank(self.pg, i) for i in range(self.pg.size())))),
                reduce_op=None if self.mode is TensorParallelLinearMode.ALL_REDUCE else dist.ReduceOp.SUM,
                root_module=self,
            )
            setattr(self, name, new_param)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # [FUNCTION DOC]
        # Behavior: applies the plain linear; in ALL_REDUCE mode passes the result through differentiable_identity
        # (all-reduce of gradient in backward), in REDUCE_SCATTER mode all-gathers the result across pg.
        # Inputs: x, input tensor.
        # Outputs: output tensor (gathered along the first dim in REDUCE_SCATTER mode).
        # Exceptions: ValueError for any other mode.
        # Side effects: TP collectives.
        y = super().forward(x)
        if self.mode is TensorParallelLinearMode.ALL_REDUCE:
            y = differentiable_identity(y, group=self.pg)
        elif self.mode is TensorParallelLinearMode.REDUCE_SCATTER:
            y = differentiable_all_gather(y, group=self.pg)
        else:
            raise ValueError(f"Got unexpected mode: {self.mode}.")

        return y


class TensorParallelEmbedding(nn.Embedding):
    # [CLASS DOC]
    # Purpose: Embedding whose vocabulary is split into contiguous equal blocks across the TP group;
    # each rank looks up only ids in its block and the results are summed across ranks.
    # Key attributes: pg, rank, world_size, original_num_embeddings, min_id (inclusive), max_id (exclusive), mode.
    # Inheritance: subclass of nn.Embedding; weight becomes a sharded NanotronParameter split on dim 0.
    # Invariant: num_embeddings divisible by TP world size.
    def __init__(
        self,
        num_embeddings,
        embedding_dim,
        pg: dist.ProcessGroup,
        mode: TensorParallelLinearMode,
        padding_idx=None,
        max_norm=None,
        norm_type=2.0,
        scale_grad_by_freq=False,
        sparse=False,
        _weight=None,
        device=None,
        dtype=None,
        contiguous_chunks: Optional[Tuple[int, ...]] = None,
    ):
        # [FUNCTION DOC]
        # Behavior: records rank/world size, asserts divisibility, computes block size and this rank's id
        # range, builds an nn.Embedding of block size, stores mode, validates contiguous_chunks sum against
        # num_embeddings, and marks parameters as sharded on dim 0.
        # Inputs: num_embeddings (full vocab size), embedding_dim, pg, mode, plus the standard nn.Embedding
        # options padding_idx, max_norm, norm_type (2.0), scale_grad_by_freq, sparse, _weight, device, dtype,
        # and contiguous_chunks (optional tuple).
        # Outputs: None. Exceptions: AssertionError on divisibility or chunk sum.
        # Side effects: allocates parameters.
        # KNOWN ISSUE: padding_idx is passed through unchanged although it refers to a global id while the local table only covers one block.
        self.pg = pg
        self.rank = dist.get_rank(self.pg)
        self.world_size = pg.size()

        self.original_num_embeddings = num_embeddings

        # TODO @thomasw21: Fix and remove that constraint. Typically there's no reason to have such a constraint.
        assert num_embeddings % self.world_size == 0
        block_size = num_embeddings // self.world_size
        # inputs in `[min_id, max_id[` are handled by `self` to get embeddings
        self.min_id = self.rank * block_size
        self.max_id = (self.rank + 1) * block_size

        super().__init__(
            block_size,
            embedding_dim,
            padding_idx=padding_idx,
            max_norm=max_norm,
            norm_type=norm_type,
            scale_grad_by_freq=scale_grad_by_freq,
            sparse=sparse,
            _weight=_weight,
            device=device,
            dtype=dtype,
        )

        self.mode = mode

        if contiguous_chunks is not None:
            assert (
                sum(contiguous_chunks) == num_embeddings
            ), f"Sum of contiguous chunks ({sum(contiguous_chunks)}) must equal to num_embeddings ({num_embeddings})"

        split_config = SplitConfig(split_dim=0, contiguous_chunks=contiguous_chunks)

        mark_all_parameters_in_module_as_sharded(self, pg=self.pg, split_config=split_config)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        # [FUNCTION DOC]
        # Behavior: when world size is greater than 1, builds a mask of ids outside this rank's range, shifts
        # ids by min_id and replaces masked ids with 0; looks up embeddings; zeroes masked rows; then
        # all-reduces (ALL_REDUCE mode) or reduce-scatters along the first dim (REDUCE_SCATTER mode) so each
        # id ends up with exactly one non-zero contribution.
        # Inputs: input_ids, integer tensor of token ids.
        # Outputs: tensor of shape input_ids shape plus embedding_dim (first dim scattered in REDUCE_SCATTER mode).
        # Exceptions: ValueError for unexpected mode.
        # Side effects: TP collectives.
        if self.pg.size() > 1:
            # `0` if input is in the correct interval, else `1`
            input_mask = torch.logical_or(self.min_id > input_ids, input_ids >= self.max_id)
            # translate for [0, self.max_id - self.min_id[
            masked_input = input_ids.clone() - self.min_id
            # default all out of bounds values to `0`
            masked_input[input_mask] = 0
        else:
            masked_input = input_ids
        out = super().forward(masked_input)

        if self.pg.size() > 1:
            out = out * (~input_mask[..., None])

        if self.mode is TensorParallelLinearMode.ALL_REDUCE:
            out = differentiable_all_reduce_sum(out, group=self.pg)
        elif self.mode is TensorParallelLinearMode.REDUCE_SCATTER:
            out = differentiable_reduce_scatter_sum(
                out, group=self.pg
            )  # this should scatter s / b*s (first dimension)
        else:
            raise ValueError(f"Got unexpected mode: {self.mode}.")

        return out  # [*input_ids.shape, embedding_dim]

    def extra_repr(self) -> str:
        # [FUNCTION DOC]
        # Behavior: nn.Embedding repr plus tp_rank and unsharded vocabulary size.
        # Inputs: None. Outputs: string. Exceptions: None explicitly raised. Side effects: None.
        return f"tp_rank={dist.get_rank(self.pg)}, {super().extra_repr()}, unsharded_num_embeddings={self.original_num_embeddings}"
