# [MODULE DOC]
# Mixture-of-experts layers for Qwen2-style MoE models, built on the grouped_gemm library.
# Main contents: Router (float32 top-k softmax gating), GroupedMLP (all local experts' gated MLPs computed with
# grouped GEMMs), Qwen2MoELayer (router, token permute/unpermute, optional shared expert, optional recompute).
# Fit in nanotron: used by the Qwen model when a moe_config is present on Qwen2Config.
# Import-time side effects: creates a module logger and imports grouped_gemm.ops; raises RuntimeError at import time
# with install instructions if grouped_gemm is missing.
# Exports: Router, GroupedMLP, Qwen2MoELayer.
from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import CheckpointFunction

from nanotron import distributed as dist
from nanotron import logging
from nanotron.config import ParallelismArgs
from nanotron.config.models_config import Qwen2Config
from nanotron.models.base import ignore_init_on_device_and_dtype
from nanotron.nn.activations import ACT2FN

logger = logging.get_logger(__name__)


try:
    import grouped_gemm.ops as ops
except ImportError:
    raise RuntimeError(
        "Grouped GEMM is not available. Please run `pip install --no-build-isolation git+https://github.com/fanshiqing/grouped_gemm@main` (takes less than 5 minutes)"
    )


class Router(nn.Module):
    # [CLASS DOC]
    # Expert router: a bias-free float32 linear projection from hidden size to num_experts, followed by softmax and top-k.
    # Key attributes: config, parallel_config, tp_pg, layer_idx (stored, unused in computation), num_experts,
    # num_experts_per_token (top_k), weight (float32 Parameter [num_experts, hidden_size], always on CUDA).
    # Construction inputs: config (Qwen2Config with moe_config), parallel_config, tp_pg, layer_idx.
    # Invariant: weight stays float32.
    def __init__(
        self, config: Qwen2Config, parallel_config: Optional[ParallelismArgs], tp_pg: dist.ProcessGroup, layer_idx: int
    ):
        # [FUNCTION DOC]
        # Stores configuration and creates a randomly initialized float32 CUDA weight of shape [num_experts, hidden_size]
        # inside ignore_init_on_device_and_dtype so the model-wide init device/dtype context does not override it.
        # Inputs: config, parallel_config (optional), tp_pg (process group), layer_idx (int).
        # Outputs: None.
        # Exceptions: AssertionError if the weight is not float32.
        # Side effects: allocates a CUDA tensor; consumes the torch RNG.
        super().__init__()
        self.config = config
        self.parallel_config = parallel_config
        self.tp_pg = tp_pg
        self.layer_idx = layer_idx

        self.num_experts = config.moe_config.num_experts
        self.num_experts_per_token = config.moe_config.top_k

        # float32 routing weights
        # NOTE: qwen keep the routing weights in float32
        # https://github.com/huggingface/transformers/blob/27a25bee4fcb865e8799ba026f1ea4455f2cca98/src/transformers/models/qwen2_moe/modeling_qwen2_moe.py#L608
        with ignore_init_on_device_and_dtype():
            self.weight = nn.Parameter(
                torch.randn(self.num_experts, config.hidden_size, dtype=torch.float32, device="cuda")
            )
        assert self.weight.dtype == torch.float32

    def gating(self, x: torch.Tensor) -> torch.Tensor:
        # [FUNCTION DOC]
        # Computes routing logits by casting x to float32 and applying the weight (no bias, no softmax).
        # Inputs: x, tensor [num_tokens, hidden_size].
        # Outputs: float32 logits [num_tokens, num_experts].
        # Exceptions: None explicitly raised. Side effects: None.
        """Compute logits for all experts (no softmax)."""
        # NOTE: qwen keep the routing logits in float32
        # https://github.com/huggingface/transformers/blob/27a25bee4fcb865e8799ba026f1ea4455f2cca98/src/transformers/models/qwen2_moe/modeling_qwen2_moe.py#L613
        return F.linear(x.to(torch.float32), self.weight, bias=None)

    def routing(self, logits: torch.Tensor):
        # [FUNCTION DOC]
        # Softmax over experts in float32, then top-k selection; indices are cast to int32 as required by ops.permute.
        # The selected weights are not renormalized to sum to one.
        # Inputs: logits [num_tokens, num_experts].
        # Outputs: tuple of routing_weights (float32 [num_tokens, top_k]) and routing_indices (int32 [num_tokens, top_k]).
        # Exceptions: None explicitly raised. Side effects: None.
        """Top-k softmax-normalized routing weights and indices."""
        routing_weights = F.softmax(logits, dim=-1, dtype=torch.float32)
        routing_weights, routing_indices = torch.topk(routing_weights, k=self.num_experts_per_token, dim=-1)
        routing_indices = routing_indices.to(torch.int32)  # NOTE: ops.permute requires indices to be int32
        return routing_weights, routing_indices

    def forward(self, x: torch.Tensor):
        # [FUNCTION DOC]
        # Runs gating then routing.
        # Inputs: x [num_tokens, hidden_size].
        # Outputs: tuple of routing weights and int32 indices, each [num_tokens, top_k].
        # Exceptions: None explicitly raised. Side effects: None.
        logits = self.gating(x)
        return self.routing(logits)


class GroupedMLP(nn.Module):
    # [CLASS DOC]
    # Holds the stacked weights of all experts local to this rank and evaluates them with grouped GEMM.
    # Key attributes: merged_gate_up_proj Parameter [num_local_experts, hidden_size, 2 * moe_intermediate_size]
    # (gate half first, then up half); merged_down_proj Parameter [num_local_experts, moe_intermediate_size,
    # hidden_size]; act, activation module from ACT2FN for config.hidden_act.
    # Construction inputs: config, parallel_config (expert_parallel_size divides num_experts).
    def __init__(self, config: Qwen2Config, parallel_config: Optional[ParallelismArgs]):
        # [FUNCTION DOC]
        # Computes num_local_experts as num_experts integer-divided by expert_parallel_size and allocates randn weights.
        # Inputs: config (Qwen2Config), parallel_config (must not be None despite the Optional hint).
        # Outputs: None.
        # Exceptions: AttributeError if parallel_config is None.
        # Side effects: allocates parameters; consumes the torch RNG.
        super().__init__()

        num_local_experts = config.moe_config.num_experts // parallel_config.expert_parallel_size
        self.merged_gate_up_proj = nn.Parameter(
            torch.randn(num_local_experts, config.hidden_size, 2 * config.moe_config.moe_intermediate_size)
        )
        self.merged_down_proj = nn.Parameter(
            torch.randn(num_local_experts, config.moe_config.moe_intermediate_size, config.hidden_size)
        )
        self.act = ACT2FN[config.hidden_act]

    def forward(
        self,
        hidden_states: torch.Tensor,
        num_tokens_per_expert: torch.Tensor,
    ):
        # [FUNCTION DOC]
        # Gated MLP over expert-sorted tokens: moves num_tokens_per_expert to CPU (required by grouped_gemm), runs a
        # grouped GEMM with merged_gate_up_proj, splits the result into gate and up halves, computes act(gate) times up,
        # then a second grouped GEMM with merged_down_proj.
        # Inputs: hidden_states, tokens already permuted so each expert's tokens are contiguous [num_routed_tokens, hidden]
        # (grouped_gemm expects bfloat16); num_tokens_per_expert, integer tensor [num_local_experts].
        # Outputs: dictionary with key hidden_states holding [num_routed_tokens, hidden_size].
        # Exceptions: errors from grouped_gemm on dtype/shape mismatch.
        # Side effects: device-to-host copy of num_tokens_per_expert (synchronizes the stream).
        """
        assume hidden_states is permuted

        grouped_gemm's notes:
        ops.gemm expect the inputs to have the following criteria:
        + expect a, b are in bfloat16
        + expect num_tokens_per_expert is a on cpu
        """
        # NOTE: ops.gemm requires "batch_sizes" (aka: num_tokens_per_expert here) to be on cpu
        num_tokens_per_expert = num_tokens_per_expert.to("cpu")
        merged_states = ops.gmm(hidden_states, self.merged_gate_up_proj, num_tokens_per_expert, trans_b=False)
        gate_states, up_states = torch.split(merged_states, merged_states.shape[-1] // 2, dim=-1)
        hidden_states = self.act(gate_states) * up_states
        hidden_states = ops.gmm(hidden_states, self.merged_down_proj, num_tokens_per_expert, trans_b=False)

        return {"hidden_states": hidden_states}


class Qwen2MoELayer(nn.Module):
    # [CLASS DOC]
    # Full MoE block for Qwen2: routes each token to top_k experts, runs the experts, combines weighted outputs, and
    # optionally adds a sigmoid-gated shared expert.
    # Key attributes: hidden_size, intermediate_size, num_experts, num_local_experts, num_experts_per_token,
    # expert_parallel_size, router (Router), enable_shared_expert, shared_expert (Qwen2MLP) and shared_expert_gate
    # (Linear hidden to 1) when enabled, experts (GroupedMLP), recompute_layer (bool).
    # Construction inputs: config, parallel_config, tp_pg, layer_idx (default 0).
    # KNOWN ISSUE: no inter-rank token exchange is implemented; with expert_parallel_size above one the global
    # routing indices do not match the local experts.
    """Mixture of experts Layer for Qwen2 models."""

    def __init__(
        self,
        config: Qwen2Config,
        parallel_config: Optional[ParallelismArgs],
        tp_pg: dist.ProcessGroup,
        layer_idx: int = 0,
    ) -> None:
        # [FUNCTION DOC]
        # Reads MoE sizes from config.moe_config, builds the Router, the optional shared expert (lazily importing Qwen2MLP to
        # avoid an import cycle) with its gate, the GroupedMLP, and stores parallel_config.recompute_layer.
        # Inputs: config (Qwen2Config), parallel_config (required in practice), tp_pg (process group), layer_idx (int, 0).
        # Outputs: None.
        # Exceptions: AttributeError if parallel_config is None.
        # Side effects: allocates parameters (router weight on CUDA).
        # KNOWN ISSUE: num_local_experts is computed twice with the same formula (redundant).
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size

        # MoE specific configurations
        self.num_experts = config.moe_config.num_experts  # Total number of experts
        self.num_local_experts = (
            config.moe_config.num_experts // parallel_config.expert_parallel_size
        )  # Experts per device
        self.num_experts_per_token = config.moe_config.top_k  # Number of experts used per token (top-k)
        self.expert_parallel_size = parallel_config.expert_parallel_size
        self.num_local_experts = self.num_experts // self.expert_parallel_size  # Experts per device

        # Get TP mode configuration

        # Router for selecting experts
        self.router = Router(config, parallel_config, tp_pg, layer_idx)

        # Enable shared experts if configured
        self.enable_shared_expert = config.moe_config.enable_shared_expert
        if self.enable_shared_expert:
            from nanotron.models.qwen import Qwen2MLP

            self.shared_expert = Qwen2MLP(
                config=config,
                parallel_config=parallel_config,
                tp_pg=tp_pg,
                intermediate_size=config.moe_config.shared_expert_intermediate_size,
            )
            # TODO: duplicte the shared expert gate
            self.shared_expert_gate = nn.Linear(
                self.hidden_size,
                1,
                bias=False,
            )  # TODO: ensure shared_expert_gate is tied across TP

        # Create the expert MLPs
        self.experts = GroupedMLP(config, parallel_config)
        # Whether to recompute MoE layer during backward pass for memory efficiency
        self.recompute_layer = parallel_config.recompute_layer

    def _dispatch_tokens(
        self,
        hidden_states: torch.Tensor,
        routing_indices: torch.Tensor,
    ):
        # [FUNCTION DOC]
        # HELPER: groups tokens by chosen expert for the grouped GEMM.
        # Counts tokens per expert with bincount over the flattened indices (minimum length num_local_experts), then calls
        # ops.permute to reorder token copies so tokens for expert 0 come first, then expert 1, and so on.
        # Inputs: hidden_states [num_tokens, hidden]; routing_indices int32 [num_tokens, top_k].
        # Outputs: tuple of permuted inputs [num_tokens * top_k, hidden], the inverse permutation mapping, and
        # num_tokens_per_expert.
        # Exceptions: None explicitly raised. Side effects: None (no communication, despite the docstring mentioning it).
        """
        Dispatches tokens to their selected experts.
        In a full implementation, this would handle the actual token routing logic
        including communication between devices.
        """
        # NOTE: start from expert 0 to expert n
        num_tokens_per_expert = torch.bincount(
            routing_indices.flatten(), minlength=self.num_local_experts
        )  # [num_local_experts]
        dispatched_inputs, inverse_permute_mapping = ops.permute(hidden_states, routing_indices)
        return dispatched_inputs, inverse_permute_mapping, num_tokens_per_expert

    def _combine_expert_outputs(self, expert_outputs, inverse_mapping, routing_weights):
        # [FUNCTION DOC]
        # HELPER: restores original token order and sums each token's top_k expert outputs weighted by routing weights,
        # using ops.unpermute.
        # Inputs: expert_outputs [num_tokens * top_k, hidden]; inverse_mapping from _dispatch_tokens; routing_weights
        # [num_tokens, top_k].
        # Outputs: tensor [num_tokens, hidden].
        # Exceptions: None explicitly raised. Side effects: None.
        """
        Combines outputs from different experts back to the original tensor layout.
        """
        hidden_states = ops.unpermute(expert_outputs, inverse_mapping, routing_weights)
        return hidden_states

    def _core_forward(self, hidden_states):
        # [FUNCTION DOC]
        # HELPER: the uncheckpointed MoE computation used by forward.
        # Steps: route, dispatch, run experts, combine; if the shared expert is enabled, add sigmoid of shared_expert_gate
        # times the shared expert output.
        # Inputs: hidden_states [num_tokens, hidden].
        # Outputs: tensor [num_tokens, hidden].
        # Exceptions: None explicitly raised. Side effects: same as GroupedMLP.forward (host sync).
        """Core forward logic for MoE layer."""
        # Get top-k routing weights and indices
        routing_weights, routing_indices = self.router(hidden_states)  # [num_tokens, num_experts_per_token]

        # Dispatch tokens to experts
        dispatched_inputs, inverse_permute_mapping, num_tokens_per_expert = self._dispatch_tokens(
            hidden_states, routing_indices
        )

        expert_outputs = self.experts(dispatched_inputs, num_tokens_per_expert)

        output = self._combine_expert_outputs(
            expert_outputs["hidden_states"], inverse_permute_mapping, routing_weights
        )

        # Add shared expert contribution if enabled
        if self.enable_shared_expert:
            shared_expert_output = self.shared_expert(hidden_states=hidden_states)["hidden_states"]
            shared_gate = torch.sigmoid(self.shared_expert_gate(hidden_states))
            output = output + shared_gate * shared_expert_output

        return output

    def _checkpointed_forward(self, hidden_states):
        # [FUNCTION DOC]
        # HELPER: runs _core_forward under torch's CheckpointFunction with RNG state preservation so activations are
        # recomputed during backward.
        # Inputs: hidden_states tensor. Outputs: tensor as _core_forward.
        # Exceptions: None explicitly raised. Side effects: saves and restores RNG state for recomputation.
        """Apply gradient checkpointing to save memory during training."""
        return CheckpointFunction.apply(self._core_forward, True, hidden_states)

    def forward(self, hidden_states):
        # [FUNCTION DOC]
        # Uses the checkpointed path when recompute_layer is set and the module is training, else the plain path.
        # Inputs: hidden_states [num_tokens, hidden].
        # Outputs: dictionary with key hidden_states holding the output tensor.
        # Exceptions: None explicitly raised. Side effects: see _core_forward.
        """Forward pass for the MoE layer."""
        if self.recompute_layer and self.training:
            hidden_states = self._checkpointed_forward(hidden_states)
        else:
            hidden_states = self._core_forward(hidden_states)

        return {"hidden_states": hidden_states}
