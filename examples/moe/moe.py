# [MODULE DOC]
# Purpose: Example Mixture-of-Experts (MoE) building blocks for the LlaMoE example model, built on top of
# megablocks (block-sparse grouped matmuls via stk) and nanotron's tensor-parallel linear layers.
# Main contents: dMoE (router plus dropless expert MLP), LearnedRouter (softmax top-k gate),
# ParallelDroplessMLP (token permutation, optional expert-parallel all-to-all, expert compute),
# ScaleGradient / scale_gradient and ExpertParallel (gradient scaling wrappers), SparseMLP (several experts per
# rank via block-sparse matmuls), MLP (exactly one expert per rank via TP linears), inclusive_cumsum helper.
# Fits in nanotron: used by the examples/moe LlaMoE model as the feed-forward block; expects a ParallelContext
# that exposes ep_pg, tp_pg and tp_and_ep_pg.
# Import-time side effects: imports megablocks.ops and all_to_all inside a try block; on ImportError only a
# warning is emitted, so later uses of ops / all_to_all fail with NameError. Creates a module logger.
# Exports: the classes above, scale_gradient, inclusive_cumsum, logger.
""" LlaMa model with MoEs"""
import warnings
from functools import partial
from typing import Optional, Tuple

import numpy as np
import stk
import torch
import torch.nn.functional as F
from config_llamoe import LlaMoEConfig
from megablocks.layers import weight_parallel as wp
from megablocks.layers.activation_fn import act_fn
from nanotron import distributed as dist
from nanotron import logging
from nanotron.config import ParallelismArgs
from nanotron.parallel.context import ParallelContext
from nanotron.parallel.sharded_parameters import SplitConfig, mark_all_parameters_in_module_as_sharded
from nanotron.parallel.tensor_parallel.enum import TensorParallelLinearMode
from nanotron.parallel.tensor_parallel.nn import (
    TensorParallelColumnLinear,
    TensorParallelRowLinear,
)
from torch import nn

try:
    import megablocks.ops as ops
    from megablocks.layers.all_to_all import all_to_all
except ImportError:
    warnings.warn("Please install megablocks to use MoEs: `pip install megablocks`")


logger = logging.get_logger(__name__)


class dMoE(torch.nn.Module):
    # [CLASS DOC]
    # Purpose: Full dropless MoE feed-forward layer: routes each token to its top-k experts and combines outputs.
    # Key attributes: config (LlaMoEConfig), tp_mode (TensorParallelLinearMode, ALL_REDUCE when no parallel config),
    # gate (LearnedRouter), experts (ParallelDroplessMLP without bias).
    # Construction inputs: config, parallel_context, parallel_config (may be None).
    # Inheritance: torch.nn.Module.
    def __init__(
        self,
        config: LlaMoEConfig,
        parallel_context: "ParallelContext",
        parallel_config: Optional[ParallelismArgs],
    ):
        # [FUNCTION DOC]
        # Behavior: stores config, picks tp_mode from parallel_config (default ALL_REDUCE), warns once on rank 0 if
        # REDUCE_SCATTER is used (experimental), then builds the router and a bias-free ParallelDroplessMLP.
        # Inputs: config (LlaMoEConfig), parallel_context (ParallelContext), parallel_config (ParallelismArgs or None).
        # Outputs: None.
        # Exceptions: None explicitly raised.
        # Side effects: registers submodules; may log a warning.
        super().__init__()
        self.config = config
        self.tp_mode = parallel_config.tp_mode if parallel_config is not None else TensorParallelLinearMode.ALL_REDUCE
        if self.tp_mode == TensorParallelLinearMode.REDUCE_SCATTER:
            logging.warn_once(
                logger=logger,
                msg="TensorParallelLinearMode.REDUCE_SCATTER is still experimental for MoEs. Use at your own risk.",
                rank=0,
            )

        # Token router.
        self.gate = LearnedRouter(config)

        # Expert computation helper.
        self.experts = ParallelDroplessMLP(
            config,
            use_bias=False,
            parallel_context=parallel_context,
            parallel_config=parallel_config,
        )

    def forward(self, x: torch.Tensor):
        # [FUNCTION DOC]
        # Behavior: reads the first two dims of x as batch_size and sequence_length, flattens x to a 2D token matrix
        # of width hidden_size, computes router scores/weights/top experts, runs the experts, and reshapes the result
        # back to the original first two dims.
        # Inputs: x, a 3D tensor whose last dim is hidden_size.
        # Outputs: tensor with the same leading two dims as x and hidden_size last.
        # Exceptions: None explicitly raised.
        # Side effects: expert-parallel communication inside the experts when ep size is above 1.
        # KNOWN ISSUE: the docstring says the input shape is [sequence_length, batch_size, hidden_size] but the code
        # names the dims batch_size, sequence_length; it is shape-agnostic in practice but the naming is inconsistent.
        # Sequence parallelism is not supported (TODO).
        """
        Args:
            x: input tensor of shape [sequence_length, batch_size, hidden_size]
        """
        # Compute the expert scores and assignments.
        # TODO: support sequence parallelism
        batch_size, sequence_length, _ = x.size()
        x = x.view(-1, self.config.hidden_size)
        scores, expert_weights, top_experts = self.gate(x)

        # Compute the experts.
        x = self.experts(x, scores, expert_weights, top_experts)
        return x.reshape(batch_size, sequence_length, -1)


# Adapted from megablocks.layers.router.LearnedRouter
class LearnedRouter(torch.nn.Module):
    # [CLASS DOC]
    # Purpose: Linear gate producing softmax expert probabilities and top-k selections (adapted from megablocks).
    # Key attributes: layer (bias-free Linear from hidden_size to moe_num_experts), config.
    # Construction inputs: config (LlaMoEConfig). Inheritance: torch.nn.Module.
    def __init__(self, config: LlaMoEConfig):
        # [FUNCTION DOC]
        # Behavior: creates the gate Linear (default PyTorch init) and stores config.
        # Inputs: config (LlaMoEConfig). Outputs: None. Exceptions: None explicitly raised. Side effects: registers params.
        super().__init__()
        self.layer = torch.nn.Linear(config.hidden_size, config.moe_num_experts, bias=False)
        # TODO: initialization
        self.config = config

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # [FUNCTION DOC]
        # Behavior: computes router logits, softmax in float32 over experts; if num_experts_per_tok is 1 takes the
        # max, otherwise topk with k equal to num_experts_per_tok.
        # Inputs: x, 2D tensor (tokens, hidden_size).
        # Outputs: tuple of scores (tokens, n_experts, float32), expert_weights (tokens, k) and expert indices
        # (tokens, k, int32).
        # Exceptions: None explicitly raised. Side effects: None.
        router_logits = self.layer(x)  # (batch * sequence_length, n_experts)
        scores = F.softmax(router_logits, dim=-1, dtype=torch.float32)  # TODO: fuse?

        if self.config.num_experts_per_tok == 1:
            expert_weights, expert_indices = scores.max(dim=-1, keepdim=True)
        else:
            expert_weights, expert_indices = torch.topk(scores, self.config.num_experts_per_tok, dim=-1)

        return scores, expert_weights, expert_indices.int()


# Adapted from megablocks.layers.mlp.ParallelDroplessMLP
class ParallelDroplessMLP(torch.nn.Module):
    # [CLASS DOC]
    # Purpose: Dropless expert computation (adapted from megablocks): sorts tokens by expert, pads bins to the
    # block size, builds a block-sparse topology and runs the expert MLP; with expert parallelism it exchanges
    # tokens across the ep group using all-to-all.
    # Key attributes: ep_pg_size / expert_parallel_group, hidden_sharding_degree (ep size divided by the number of
    # expert-holding ranks), experts_per_rank, num_experts, num_experts_per_tok, sort_end_bit (bits for radix sort
    # of expert ids), optional bias, forward_fn (parallel_forward_once if ep size above 1 else forward_once),
    # blocking (128), mlp (MLP when one expert per rank, else SparseMLP), transpose_sort_end_bit.
    # Construction inputs: config, use_bias, parallel_context, parallel_config. Inheritance: torch.nn.Module.
    # Invariant: intermediate_size and padded token counts must be multiples of blocking.
    def __init__(
        self,
        config: LlaMoEConfig,
        use_bias: bool,
        parallel_context: "ParallelContext",
        parallel_config: Optional[ParallelismArgs],
    ):
        # [FUNCTION DOC]
        # Behavior: computes the sharding/partition numbers above, allocates an uninitialized bias if use_bias, selects
        # the forward function and expert MLP implementation, and computes bit widths for sorting.
        # Inputs: config (LlaMoEConfig), use_bias (bool), parallel_context (ParallelContext), parallel_config (or None).
        # Outputs: None. Exceptions: None explicitly raised.
        # Side effects: registers parameters/submodules.
        # KNOWN ISSUE: the bias is created with torch.empty and never initialized (TODO in code).
        super().__init__()
        self.config = config
        self.use_bias = use_bias

        self.ep_pg_size = parallel_context.ep_pg.size()
        self.expert_parallel_group = parallel_context.ep_pg

        self.hidden_sharding_degree = self.ep_pg_size // min(self.ep_pg_size, self.config.moe_num_experts)
        self.experts_per_rank = self.config.moe_num_experts // min(self.ep_pg_size, self.config.moe_num_experts)

        self.num_experts = config.moe_num_experts
        self.num_experts_per_tok = self.config.num_experts_per_tok

        # Calculate the number of bits needed to represent the expert indices
        # so that we can pass it to radix sort.
        self.sort_end_bit = max(int(np.ceil(np.log2(self.num_experts))), 1)

        if use_bias:
            self.bias = torch.nn.Parameter(torch.empty(config.hidden_size))  # TODO: init

        # Select the forward function for the operating mode.
        self.forward_fn = self.parallel_forward_once if self.ep_pg_size > 1 else self.forward_once

        self.blocking = 128

        if self.experts_per_rank == 1:
            self.mlp = MLP(config=config, parallel_config=parallel_config, tp_pg=parallel_context.tp_pg)
        else:
            self.mlp = SparseMLP(config=config, parallel_config=parallel_config, parallel_context=parallel_context)

        max_column_index = (self.config.intermediate_size * self.num_experts) // self.blocking
        self.transpose_sort_end_bit = max(int(np.ceil(np.log2(max_column_index))), 1)

    def indices_and_bins(self, top_expert):
        # [FUNCTION DOC]
        # Behavior: casts expert ids to int, radix-sorts them to get sorted bin ids and permutation indices, histograms
        # tokens per expert, and computes inclusive cumulative bin ends.
        # Inputs: top_expert, flat tensor of expert ids per (token, k) slot.
        # Outputs: tuple (indices, bin_ids, bins, tokens_per_expert).
        # Exceptions: None explicitly raised. Side effects: None.
        # Sort the expert ids to produce the scatter/gather
        # indices for the permutation.
        top_expert = top_expert.int()
        bin_ids, indices = ops.sort(top_expert, self.sort_end_bit)
        tokens_per_expert = ops.histogram(top_expert, self.num_experts)

        # Calculate the bin bounds for the sorted tokens.
        bins = inclusive_cumsum(tokens_per_expert, 0)
        return indices, bin_ids, bins, tokens_per_expert

    def indices_and_padded_bins(self, top_experts):
        # [FUNCTION DOC]
        # Behavior: like indices_and_bins but also rounds per-expert counts up to blocking and returns cumulative padded
        # bin ends. Does not cast the input to int.
        # Inputs: top_experts, flat int tensor of expert ids.
        # Outputs: tuple (indices, bin_ids, bins, padded_bins, tokens_per_expert).
        # Exceptions: None explicitly raised. Side effects: None.
        # Sort the expert ids to produce the scatter/gather
        # indices for the permutation.
        bin_ids, indices = ops.sort(top_experts, self.sort_end_bit)

        # Histogram the expert ids to identify the number of
        # tokens routed to each expert.
        tokens_per_expert = ops.histogram(top_experts, self.num_experts)

        # Round the token counts up to the block size used in
        # the matrix muliplications. Calculate the starting
        # position of each bin.
        padded_tokens_per_expert = ops.round_up(tokens_per_expert, self.blocking)
        padded_bins = inclusive_cumsum(padded_tokens_per_expert, 0)

        # Calculate the bin bounds for the sorted tokens.
        bins = inclusive_cumsum(tokens_per_expert, 0)
        return indices, bin_ids, bins, padded_bins, tokens_per_expert

    def forward_once(self, x, expert_weights, top_experts):  # TODO: sparse
        # [FUNCTION DOC]
        # Behavior (no expert parallelism): computes permutation and padded bins without grad, padded-gathers tokens
        # into expert-contiguous blocks (replicating each token num_experts_per_tok times), builds the sparse topology,
        # runs the expert MLP, then padded-scatters back weighting by expert_weights and summing over the k copies.
        # Inputs: x (tokens, hidden_size), expert_weights (flat), top_experts (flat).
        # Outputs: tuple (output tokens tensor (tokens, hidden_size), tokens_per_expert).
        # Exceptions: None explicitly raised. Side effects: None.
        with torch.no_grad():
            (
                indices,
                bin_ids,
                bins,
                padded_bins,
                tokens_per_expert,
            ) = self.indices_and_padded_bins(top_experts)

        # Route the tokens for MoE computation.
        x = ops.padded_gather(x, indices, bin_ids, bins, padded_bins, self.num_experts_per_tok)

        with torch.no_grad():
            topo = self.topology(x, padded_bins)

        x = self.mlp(x, topo)

        # Un-route the data for the MoE output.
        x = ops.padded_scatter(
            x,
            indices,
            bin_ids,
            expert_weights,
            bins,
            padded_bins,
            self.num_experts_per_tok,
            -1,
        )
        return x, tokens_per_expert

    def parallel_forward_once(self, x, expert_weights, top_experts):
        # [FUNCTION DOC]
        # Behavior (expert parallelism):
        # 1. Without grad, compute local permutation and per-expert counts, repeat counts hidden_sharding_degree times,
        #    and start an async all_to_all_single on the ep group to learn how many tokens each rank will send us.
        # 2. Gather tokens into expert order (k copies per token).
        # 3. Wait, reshape counts to (ep_pg_size, experts_per_rank), derive per-rank send and receive counts on CPU.
        # 4. Repeat x hidden_sharding_degree times and launch async all_to_all of the token data.
        # 5. Without grad, build local expert ids for received tokens (arange modulo experts_per_rank, replicated by
        #    received counts), sort them and compute local bins.
        # 6. Wait for the exchange, run permute_and_compute with k of 1 and no weights.
        # 7. All-to-all the results back, sum over the hidden sharding dimension, and locally scatter back weighting by
        #    expert_weights.
        # Inputs: x (tokens, hidden_size), expert_weights (flat), top_experts (flat).
        # Outputs: tuple (output tokens, flattened tokens_per_expert).
        # Exceptions: None explicitly raised.
        # Side effects: collective communication (all_to_all_single and all_to_all) on the ep group; device-to-host syncs.
        with torch.no_grad():
            indices, bin_ids, bins, tokens_per_expert = self.indices_and_bins(top_experts)
            repeated_tokens_per_expert = ops.repeat(tokens_per_expert, (self.hidden_sharding_degree,))
            parallel_tokens_per_expert = torch.empty_like(repeated_tokens_per_expert)
            tpe_handle = torch.distributed.all_to_all_single(
                parallel_tokens_per_expert,
                repeated_tokens_per_expert,
                group=self.expert_parallel_group,
                async_op=True,
            )

        x = ops.gather(x, indices, bin_ids, bins, self.num_experts_per_tok)

        # Compute the number of tokens that will be received from each
        # device and permute the input data across the devices.
        with torch.no_grad():
            tpe_handle.wait()

            # Reshape to [ep_pg_size, num_experts_per_rank].
            repeated_tokens_per_expert = repeated_tokens_per_expert.view(self.ep_pg_size, self.experts_per_rank)
            parallel_tokens_per_expert = parallel_tokens_per_expert.view(self.ep_pg_size, self.experts_per_rank)

            send_counts = repeated_tokens_per_expert.cpu().sum(dim=-1)
            parallel_tokens_per_expert_cpu = parallel_tokens_per_expert.cpu()
            recv_counts = parallel_tokens_per_expert_cpu.sum(dim=-1)

            # Convert the send/recv counts to lists.
            send_counts = send_counts.tolist()
            recv_counts = recv_counts.tolist()
            tokens_received = sum(recv_counts)

        x = ops.repeat(x, (self.hidden_sharding_degree, 1))

        # Start the cross-device permutation asynchronously so we can
        # overlap communication with computation.
        parallel_x, parallel_x_handle = all_to_all(
            x, recv_counts, send_counts, self.expert_parallel_group, async_op=True
        )

        with torch.no_grad():
            replicate_bins = inclusive_cumsum(parallel_tokens_per_expert.flatten(), 0)

            # Construct the expert indices for the permuted tokens.
            parallel_top_expert = torch.remainder(
                torch.arange(
                    self.num_experts * self.hidden_sharding_degree,
                    dtype=torch.int32,
                    device=indices.device,
                ),
                self.experts_per_rank,
            )
            parallel_top_expert = ops.replicate(
                parallel_top_expert.unsqueeze(dim=0), replicate_bins, tokens_received
            ).flatten()

            parallel_bin_ids, parallel_indices = ops.sort(parallel_top_expert, self.sort_end_bit)

            # Calculate the bins boundaries from the token counts.
            parallel_tokens_per_expert = parallel_tokens_per_expert.sum(dim=0, dtype=torch.int)
            parallel_bins = inclusive_cumsum(parallel_tokens_per_expert, 0)

        # Locally permute the tokens and perform the expert computation.
        # Block to make sure that the cross-device permutation is complete.
        parallel_x_handle.wait()
        parallel_x = self.permute_and_compute(
            parallel_x,
            parallel_tokens_per_expert,
            parallel_indices,
            parallel_bin_ids,
            None,  # expert_weights
            parallel_bins,
            num_experts_per_tok=1,
        )

        # Un-permute the tokens across the devices.
        x, _ = all_to_all(parallel_x, send_counts, recv_counts, self.expert_parallel_group)

        # Reduce along the hidden sharding to get the final outputs.
        shape = (self.hidden_sharding_degree, -1, self.config.hidden_size)
        x = ops.sum(x.view(shape), dim=0)

        # Un-permute locally to setup for the next series of operations.
        x = ops.scatter(
            x,
            indices,
            bin_ids,
            expert_weights,
            bins,
            self.num_experts_per_tok,
        )
        return x, tokens_per_expert.flatten()

    def forward(self, x, scores, expert_weights, top_experts):
        # [FUNCTION DOC]
        # Behavior: flattens expert_weights and top_experts, calls forward_fn, adds bias if configured. scores is unused.
        # Inputs: x (tokens, hidden_size), scores (tokens, n_experts), expert_weights and top_experts (tokens, k).
        # Outputs: tensor (tokens, hidden_size).
        # Exceptions: None explicitly raised. Side effects: communication when ep size is above 1.
        # KNOWN ISSUE: docstring describes x as 3D [sequence_length, batch_size, hidden_size] but callers pass the
        # flattened 2D token matrix.
        """
        Args:
            x: input tensor of shape [sequence_length, batch_size, hidden_size]
            scores: tensor of shape [sequence_length * batch_size, n_experts]
            expert_weights: tensor of shape [sequence_length * batch_size, num_experts_per_tok]
            top_experts: tensor of shape [sequence_length * batch_size, num_experts_per_tok]
        """
        # Compute the experts.
        x, tokens_per_expert = self.forward_fn(x, expert_weights.flatten(), top_experts.flatten())

        if self.use_bias:
            return x + self.bias
        return x

    def permute_and_compute(
        self,
        x,
        tokens_per_expert,
        indices,
        bin_ids,
        expert_weights,
        bins,
        num_experts_per_tok,
    ):
        # [FUNCTION DOC]
        # Behavior: pads per-expert counts to blocking, padded-gathers tokens, builds topology without grad, runs the
        # expert MLP and padded-scatters results back (weighted by expert_weights when given).
        # Inputs: x, tokens_per_expert, indices, bin_ids, expert_weights (or None), bins, num_experts_per_tok.
        # Outputs: tensor of un-permuted outputs.
        # Exceptions: None explicitly raised. Side effects: None.
        # Round the token counts up to the block size used in the matrix
        # multiplication. Calculate the starting position of each bin.
        padded_tokens_per_expert = ops.round_up(tokens_per_expert, self.blocking)
        padded_bins = inclusive_cumsum(padded_tokens_per_expert, 0)

        # Route the tokens for MoE computation.
        x = ops.padded_gather(x, indices, bin_ids, bins, padded_bins, num_experts_per_tok)

        # Perform the expert computation.
        with torch.no_grad():
            topo = self.topology(x, padded_bins)
        x = self.mlp(x, topo)

        # Un-route the data for the MoE output.
        return ops.padded_scatter(x, indices, bin_ids, expert_weights, bins, padded_bins, num_experts_per_tok)

    def sparse_transpose(self, size, row_indices, column_indices, offsets):
        # [FUNCTION DOC]
        # HELPER: builds the transpose metadata of a block-sparse stk matrix.
        # Behavior: sorts column indices to get the gather order, takes row indices in that order as transposed column
        # indices, uses the gather order as block offsets, and computes transposed row offsets as a zero followed by the
        # inclusive cumsum of the per-column nonzero histogram.
        # Inputs: size (rows, cols), row_indices, column_indices, offsets (unused).
        # Outputs: tuple (column_indices_t, offsets_t, block_offsets_t).
        # Exceptions: None explicitly raised. Side effects: None.
        block_columns = size[1] // self.blocking
        _, gather_indices = ops.sort(column_indices.int(), self.transpose_sort_end_bit)
        column_indices_t = row_indices.gather(0, gather_indices.long())
        block_offsets_t = gather_indices.int()

        zero = torch.zeros((1,), dtype=torch.int32, device=row_indices.device)
        nnz_per_column = ops.histogram(column_indices, block_columns)
        nnz_per_column = ops.inclusive_cumsum(nnz_per_column, 0)
        offsets_t = torch.cat([zero, nnz_per_column])
        return column_indices_t, offsets_t, block_offsets_t

    def topology(self, x, padded_bins):
        # [FUNCTION DOC]
        # HELPER: builds the block-sparse stk.Matrix describing which (token block, expert column block) pairs exist.
        # Behavior: asserts padded token count and intermediate_size are multiples of blocking; every row block has
        # intermediate_size divided by blocking nonzero blocks; column indices come from megablocks topology on padded
        # bins; data is a meta-device placeholder; computes row indices and the transpose metadata.
        # Inputs: x (padded tokens, hidden), padded_bins.
        # Outputs: stk.Matrix of shape (padded_tokens, intermediate_size times experts_per_rank).
        # Exceptions: AssertionError when sizes are not multiples of blocking.
        # Side effects: None.
        padded_tokens, _ = x.size()
        assert padded_tokens % self.blocking == 0
        assert self.config.intermediate_size % self.blocking == 0

        # Offsets for the sparse matrix. All rows have the
        # same number of nonzero blocks dictated by the
        # dimensionality of a single expert.
        block_rows = padded_tokens // self.blocking
        blocks_per_row = self.config.intermediate_size // self.blocking
        offsets = torch.arange(0, block_rows * blocks_per_row + 1, blocks_per_row, dtype=torch.int32, device=x.device)

        # Indices for the sparse matrix. The indices for
        # the intermediate matrix are dynamic depending
        # on the mapping of tokens to experts.
        column_indices = ops.topology(padded_bins, self.blocking, block_rows, blocks_per_row)

        # TODO(tgale): This is unused. Remove the need for this in stk.
        # For now, use meta init to save the device memory.
        data = torch.empty(column_indices.numel(), self.blocking, self.blocking, dtype=x.dtype, device="meta")
        shape = (padded_tokens, self.config.intermediate_size * self.experts_per_rank)
        row_indices = stk.ops.row_indices(shape, data, offsets, column_indices)
        column_indices_t, offsets_t, block_offsets_t = self.sparse_transpose(
            shape, row_indices, column_indices, offsets
        )
        return stk.Matrix(
            shape, data, row_indices, column_indices, offsets, column_indices_t, offsets_t, block_offsets_t
        )


class ScaleGradient(torch.autograd.Function):
    # [CLASS DOC]
    # Purpose: autograd function that is identity in forward and multiplies the gradient by a scale in backward.
    # Inheritance: torch.autograd.Function with AMP custom forward/backward decorators.
    @staticmethod
    @torch.cuda.amp.custom_fwd
    def forward(ctx, x, scale):
        # [FUNCTION DOC]
        # Behavior: stores scale on ctx and returns x unchanged.
        # Inputs: ctx, x (tensor), scale (float). Outputs: x. Exceptions: None explicitly raised. Side effects: sets ctx.scale.
        ctx.scale = scale
        return x

    @staticmethod
    @torch.cuda.amp.custom_bwd
    def backward(ctx, grad):
        # [FUNCTION DOC]
        # Behavior: returns grad times ctx.scale for x and None for scale.
        # Inputs: ctx, grad. Outputs: tuple (scaled grad, None). Exceptions: None explicitly raised. Side effects: None.
        return grad * ctx.scale, None


scale_gradient = ScaleGradient.apply


class ExpertParallel(nn.Module):
    # [CLASS DOC]
    # Purpose: wrapper meant to scale expert weight gradients by one over expert_parallel_size, since expert grads
    # are not averaged across the ep group like DP grads.
    # Key attributes: module (wrapped module), expert_parallel_size.
    # Inheritance: nn.Module.
    """
    ExpertParallel serves to scale the gradients of the expert weights because unlike DP the gradients are not averaged across the expert parallel group.
    """

    def __init__(self, module, expert_parallel_size: int):
        # [FUNCTION DOC]
        # Inputs: module (nn.Module to wrap), expert_parallel_size (int). Outputs: None.
        # Exceptions: None explicitly raised. Side effects: registers the submodule.
        super().__init__()
        self.module = module
        self.expert_parallel_size = expert_parallel_size

    def forward(self, *args, **kwargs):
        # [FUNCTION DOC]
        # Behavior: calls scale_gradients then forwards all arguments to the wrapped module.
        # Inputs: arbitrary positional and keyword arguments. Outputs: wrapped module output.
        # Exceptions: None explicitly raised. Side effects: see scale_gradients.
        self.scale_gradients()
        return self.module(*args, **kwargs)

    def scale_gradients(self):
        # [FUNCTION DOC]
        # Behavior: applies scale_gradient to the wrapped module with scale one over expert_parallel_size and discards
        # the result.
        # Inputs: None. Outputs: None. Exceptions: None explicitly raised. Side effects: None effective.
        # KNOWN ISSUE: scale_gradient is an autograd Function applied to an nn.Module rather than to tensors, and its
        # return value is discarded, so no gradient scaling is actually attached to the weights.
        scale_gradient(self.module, 1 / self.expert_parallel_size)


class SparseMLP(nn.Module):
    # [CLASS DOC]
    # Purpose: expert MLP for several experts per rank using block-sparse matmuls (sdd then dsd), optionally
    # weight-parallel over the TP group.
    # Key attributes: ep_pg_size, experts_per_rank, tp_pg, w1 and w2 (ExpertParallel-wrapped bias-free Linears of
    # shape intermediate_size times experts_per_rank divided by TP size, by hidden_size), act (tanh GELU), sdd and
    # dsd ops (megablocks weight-parallel versions when TP size above 1, otherwise stk ops).
    # Construction inputs: config, parallel_config (or None), parallel_context. Inheritance: nn.Module.
    def __init__(
        self,
        config: LlaMoEConfig,
        parallel_config: Optional[ParallelismArgs],
        parallel_context: "ParallelContext",
    ):
        # [FUNCTION DOC]
        # Behavior: computes partitioning, creates w1 and w2, marks all parameters as sharded along dim 0 over the
        # tp_and_ep process group, transposes w1 in place when TP size is 1, and chooses activation and sparse ops.
        # Inputs: config (LlaMoEConfig), parallel_config (ParallelismArgs or None), parallel_context.
        # Outputs: None. Exceptions: None explicitly raised.
        # Side effects: mutates parameter metadata (sharded markers) and w1 data.
        super().__init__()

        self.ep_pg_size = parallel_config.expert_parallel_size if parallel_config is not None else 1
        self.experts_per_rank = config.moe_num_experts // min(self.ep_pg_size, config.moe_num_experts)
        self.tp_pg = parallel_context.tp_pg

        self.w1 = ExpertParallel(
            nn.Linear(
                config.hidden_size, config.intermediate_size * self.experts_per_rank // self.tp_pg.size(), bias=False
            ),
            expert_parallel_size=self.ep_pg_size,
        )
        self.w2 = ExpertParallel(
            nn.Linear(
                config.hidden_size, config.intermediate_size * self.experts_per_rank // self.tp_pg.size(), bias=False
            ),
            expert_parallel_size=self.ep_pg_size,
        )

        mark_all_parameters_in_module_as_sharded(
            self,
            pg=parallel_context.tp_and_ep_pg,
            split_config=SplitConfig(split_dim=0),
        )

        if self.tp_pg.size() == 1:
            self.w1.module.weight.data = self.w1.module.weight.data.T.contiguous()

        # TODO @nouamane: jit
        self.act = partial(F.gelu, approximate="tanh")
        self.sdd = partial(wp.sdd_nt, group=self.tp_pg) if self.tp_pg.size() > 1 else stk.ops.sdd
        self.dsd = partial(wp.dsd_nn, group=self.tp_pg) if self.tp_pg.size() > 1 else stk.ops.dsd

    def forward(self, x, topo):
        # [FUNCTION DOC]
        # Behavior: calls scale_gradients on w1 and w2, computes sparse-dense-dense product of x with w1 into the
        # topology, applies GELU via megablocks act_fn, then dense-sparse-dense product with w2.
        # Inputs: x (padded tokens, hidden_size), topo (stk.Matrix). Outputs: tensor (padded tokens, hidden_size).
        # Exceptions: None explicitly raised. Side effects: TP communication when TP size above 1.
        self.w1.scale_gradients(), self.w2.scale_gradients()
        x = self.sdd(x.contiguous(), self.w1.module.weight, topo)
        activation_fn_out = act_fn(x, self.act)
        return self.dsd(activation_fn_out, self.w2.module.weight)


class MLP(nn.Module):
    # [CLASS DOC]
    # Purpose: dense expert MLP used when each rank holds exactly one expert; TP column linear, GELU, TP row linear.
    # Key attributes: ep_pg_size, experts_per_rank (must be 1), w1, w2 (ExpertParallel-wrapped TP linears), act.
    # Construction inputs: config, parallel_config, tp_pg. Inheritance: nn.Module.
    def __init__(
        self,
        config: LlaMoEConfig,
        parallel_config: Optional[ParallelismArgs],
        tp_pg: dist.ProcessGroup,
    ):
        # [FUNCTION DOC]
        # Behavior: reads tp mode and async communication flag (defaults when no parallel config), computes
        # experts_per_rank, asserts it is 1, and builds w1 (column) and w2 (row; async only with REDUCE_SCATTER).
        # Inputs: config (LlaMoEConfig), parallel_config (ParallelismArgs or None), tp_pg (ProcessGroup).
        # Outputs: None. Exceptions: AssertionError when more than one expert per rank.
        # Side effects: registers submodules.
        # KNOWN ISSUE: expert_parallel_size is read from parallel_config without a None check even though the other
        # fields handle None, so a None parallel_config raises AttributeError.
        super().__init__()

        tp_mode = parallel_config.tp_mode if parallel_config is not None else TensorParallelLinearMode.ALL_REDUCE
        tp_linear_async_communication = (
            parallel_config.tp_linear_async_communication if parallel_config is not None else False
        )

        self.ep_pg_size = parallel_config.expert_parallel_size
        self.experts_per_rank = config.moe_num_experts // min(self.ep_pg_size, config.moe_num_experts)

        assert self.experts_per_rank == 1, "moe.MLP only supports 1 expert per rank, otherwise use moe.SparseMLP"

        self.w1 = ExpertParallel(
            TensorParallelColumnLinear(
                config.hidden_size,
                config.intermediate_size * self.experts_per_rank,
                pg=tp_pg,
                mode=tp_mode,
                bias=False,
                async_communication=tp_linear_async_communication,
            ),
            expert_parallel_size=self.ep_pg_size,
        )

        self.w2 = ExpertParallel(
            TensorParallelRowLinear(
                config.intermediate_size * self.experts_per_rank,
                config.hidden_size,
                pg=tp_pg,
                mode=tp_mode,
                bias=False,
                async_communication=tp_linear_async_communication
                and tp_mode is TensorParallelLinearMode.REDUCE_SCATTER,
            ),
            expert_parallel_size=self.ep_pg_size,
        )
        # TODO @nouamane: jit
        self.act = partial(F.gelu, approximate="tanh")

    def forward(self, hidden_states, topo):  # [seq_length, batch_size, hidden_dim]
        # [FUNCTION DOC]
        # Behavior: w1, GELU, w2. topo is ignored.
        # Inputs: hidden_states (tokens, hidden_size), topo (unused). Outputs: tensor (tokens, hidden_size).
        # Exceptions: None explicitly raised. Side effects: TP collectives inside the linears.
        merged_states = self.w1(hidden_states)
        hidden_states = self.w2(self.act(merged_states))
        return hidden_states


def inclusive_cumsum(x, dim):
    # [FUNCTION DOC]
    # HELPER: wraps megablocks inclusive_cumsum so 0-d results become 1-element tensors.
    # Inputs: x (tensor), dim (int). Outputs: tensor of cumulative sums, at least 1-D.
    # Exceptions: None explicitly raised. Side effects: None.
    scalar = ops.inclusive_cumsum(x, dim)
    return scalar.view(1) if not len(scalar.size()) else scalar
