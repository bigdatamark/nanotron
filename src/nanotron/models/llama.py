# coding=utf-8
# [MODULE DOC]
# Purpose: nanotron implementation of the LLaMA decoder-only transformer with tensor parallelism (TP, heads and
# MLP sharded across the TP group), pipeline parallelism (PP, each embedding/decoder layer/norm/head/loss wrapped
# in a PipelineBlock placed on one PP rank by build_model), GQA/MQA, RoPE (interleaved or HF-style), flash-attention
# kernels for training and KV-cache inference, optional activation recomputation and optional z-loss.
# Main contents: rotary embedding modules, GLUActivation, MLP, CoreAttention, pad_to_right, CausalSelfAttention,
# LlamaDecoderLayer, Embedding, LlamaModel (pipeline graph), masked_mean, Loss, LossWithZLoss, LlamaForTraining
# (the NanotronModel used by the trainer) and get_flops (MFU accounting).
# Layout conventions: activations between blocks are sequence-first [seq_length, batch_size, hidden]; masks are
# batch-first [batch_size, seq_length]; logits stay sharded on the vocab dimension across TP ranks.
# Import-time side effects: imports flash_attn (hard dependency, fails without it), creates a module logger, and
# TorchScript-compiles masked_mean.
# Copyright 2018 HuggingFace Inc. team.
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
"""PyTorch LLaMa model."""

from typing import Dict, List, Optional, Union

import torch
from flash_attn import bert_padding
from flash_attn.flash_attn_interface import (
    flash_attn_varlen_func,
)
from torch import nn
from torch.utils.checkpoint import CheckpointFunction

from nanotron import distributed as dist
from nanotron import logging
from nanotron.config import Config, LlamaConfig, ParallelismArgs
from nanotron.config.models_config import RandomInit, SpectralMupInit
from nanotron.generation.generate_store import AttachableStore
from nanotron.logging import log_rank
from nanotron.models import NanotronModel
from nanotron.nn.activations import ACT2FN
from nanotron.nn.layer_norm import TritonRMSNorm
from nanotron.parallel import ParallelContext
from nanotron.parallel.parameters import NanotronParameter
from nanotron.parallel.pipeline_parallel.block import PipelineBlock, TensorPointer
from nanotron.parallel.pipeline_parallel.p2p import P2P
from nanotron.parallel.tensor_parallel.functional import sharded_cross_entropy
from nanotron.parallel.tensor_parallel.nn import (
    TensorParallelColumnLinear,
    TensorParallelEmbedding,
    TensorParallelLinearMode,
    TensorParallelRowLinear,
)
from nanotron.random import RandomStates
from nanotron.scaling.parametrization import SpectralMupParametrizator, StandardParametrizator
from nanotron.utils import checkpoint_method

logger = logging.get_logger(__name__)


class RotaryEmbedding(nn.Module):
    # [CLASS DOC]
    # Purpose: interleaved RoPE (adjacent pairs of channels rotated as complex numbers), lazily building a cache of
    # cos/sin pairs on CUDA and growing it by doubling when longer positions are requested.
    # State: dim (rotary dim, must be even), end (cache length), theta (base), freqs_cis (non-persistent float buffer
    # shaped end by dim/2 by 2, real view of complex unit phasors), _initialized_buffer flag.
    def __init__(self, dim: int, end: int, theta: float = 10000.0):
        # [FUNCTION DOC]
        # Behavior: validate dim is even and store dim, end, theta; buffer is created later.
        # Inputs: dim (int), end (int, max positions), theta (float, default 10000.0). Outputs: None.
        # Exceptions: AssertionError when dim is odd. Side effects: None.
        super().__init__()
        assert dim % 2 == 0
        self.dim = dim
        self.end = end
        self.theta = theta
        # TODO @nouamane: Figure out why we can't set `DTypeInvariantTensor` ...
        # TODO @thomasw21: Complex buffers break DDP, instead we store float and view them as complex
        self.freqs_cis: torch.Tensor
        self._initialized_buffer = False

    def init_rotary_embeddings(self):
        # [FUNCTION DOC]
        # Behavior: return early if already initialized. Register a float32 CUDA buffer of shape end by dim/2 by 2 (cast to
        # float32 if a surrounding dtype context changed it). Compute inverse frequencies theta to the power of minus
        # 2i/dim on CPU (to match HF Transformers numerics), move to CUDA, take the outer product with positions 0..end-1,
        # turn into unit complex numbers via polar form and store their real/imag pairs in the buffer; mark initialized.
        # Inputs: none. Outputs: None. Exceptions: AssertionError if the buffer is not on CUDA or not float32.
        # Side effects: allocates/overwrites the buffer on the CUDA device.
        if self._initialized_buffer is True:
            # Buffer if already initialized
            return
        self.register_buffer(
            "freqs_cis",
            torch.empty(self.end, self.dim // 2, 2, dtype=torch.float, device="cuda"),
            persistent=False,
        )
        assert self.freqs_cis.device.type == "cuda"
        # TODO @nouamane: One we figure out how to do the DTypeInvariantTensor, this can be removed and changed to an assert
        if self.freqs_cis.dtype != torch.float:
            self.freqs_cis = self.freqs_cis.to(torch.float)
        assert self.freqs_cis.dtype == torch.float
        freqs = 1.0 / (
            self.theta ** (torch.arange(0, self.dim, 2, dtype=torch.float, device="cpu")[: (self.dim // 2)] / self.dim)
        ).to(
            "cuda"
        )  # should be computed on CPU, otherwise different results with Transformers.
        t = torch.arange(self.end, device="cuda")
        freqs = torch.outer(t, freqs).float()
        complex_freqs = torch.polar(torch.ones_like(freqs), freqs)
        freqs = torch.view_as_real(complex_freqs)
        self.freqs_cis.copy_(freqs)
        self._initialized_buffer = True

    def forward(
        self,
        x: torch.Tensor,  # [batch_size, seq_length, num_heads, d_qk]
        position_ids: Optional[torch.LongTensor],  # [batch_size, seq_length]
    ):
        # [FUNCTION DOC]
        # Behavior: while the last position id or the sequence length reaches end, double end and mark the cache stale
        # (this check synchronizes with the device); rebuild the cache if stale (printing a message). View the last
        # dimension as pairs, upcast bfloat16 to float32, view as complex, pick phasors for positions 0..seq_length-1
        # (when position_ids is None) or gather them by position_ids, multiply, convert back to real and to the input dtype.
        # Inputs: x [batch_size, seq_length, num_heads, d_qk]; position_ids [batch_size, seq_length] long tensor or None.
        # Outputs: tensor shaped like x, rotated.
        # Exceptions: AssertionError when the head dim is odd; ValueError when the last position id is negative or out of range.
        # Side effects: may grow and reallocate the buffer; prints to stdout; host-device sync.
        # KNOWN ISSUE: only the last position of the last row is validated/used to grow the cache; larger positions elsewhere in the batch are not checked.
        batch_size, seq_length, num_heads, inner_dim = x.shape
        while (
            position_ids is not None and position_ids[-1, -1] >= self.end
        ) or seq_length >= self.end:  # TODO @nouamane: check if this causes cpu-gpu sync
            self.end *= 2
            self._initialized_buffer = False
        if self._initialized_buffer is False:
            print(f"Initializing rotary embeddings with end={self.end}")
            self.init_rotary_embeddings()
        dtype = x.dtype
        assert inner_dim % 2 == 0
        x = x.view(
            batch_size, seq_length, num_heads, inner_dim // 2, 2
        )  # [batch_size, q_length, num_heads, inner_dim]
        if x.dtype == torch.bfloat16:
            x = x.float()
        complex_x = torch.view_as_complex(x)  # [batch_size, q_length, num_heads, inner_dim // 2]
        if position_ids is None:
            freqs_cis = self.freqs_cis[None, :seq_length, None, :]
        else:
            # TODO(kunhao): Should None follow the num_heads dimension?
            if position_ids[-1, -1] < 0 or position_ids[-1, -1] >= self.end:  # Quick test hopefully
                raise ValueError(f"Position ids must be in the range [0, {self.end}), but got {position_ids}")
            freqs_cis = self.freqs_cis[position_ids][:, :, None, :]
        complex_freqs = torch.view_as_complex(freqs_cis)
        x_out = torch.view_as_real(complex_x * complex_freqs).view(batch_size, seq_length, num_heads, inner_dim)
        return x_out.type(dtype)


## Copy from transformers. Non interleaved version of RoPE. Will be refactored later
class LlamaRotaryEmbedding(nn.Module):
    # [CLASS DOC]
    # Purpose: non-interleaved (rotate-half, HF Transformers compatible) RoPE that computes cos/sin on the fly from a
    # stored inverse-frequency buffer.
    # State: dim, end (kept only for the interface), theta (default 500000.0), inv_freq (non-persistent float32 CUDA buffer of length dim/2).
    def __init__(self, dim: int, end: int, theta: float = 500000.0):
        # [FUNCTION DOC]
        # Behavior: store dim, end, theta and immediately build inv_freq.
        # Inputs: dim (int), end (int), theta (float, default 500000.0). Outputs: None. Exceptions: None explicitly raised.
        # Side effects: CUDA allocation.
        super().__init__()
        self.dim = dim
        self.end = end
        self.theta = theta
        self.init_rotary_embeddings()

    def init_rotary_embeddings(self):
        # [FUNCTION DOC]
        # Behavior: compute inverse frequencies on CPU in float32, register an empty CUDA buffer, force it to float32 and copy the values in.
        # Inputs: none. Outputs: None. Exceptions: None explicitly raised. Side effects: registers/overwrites the inv_freq buffer.
        inv_freq = 1.0 / (
            self.theta ** (torch.arange(0, self.dim, 2, dtype=torch.float, device="cpu") / self.dim)
        )  # important to compute on CPU
        self.register_buffer(
            "inv_freq", torch.empty(self.dim // 2, dtype=torch.float, device="cuda"), persistent=False
        )
        self.inv_freq = self.inv_freq.to(
            torch.float
        )  # make it float32 before copy to avoid precision loss during copy_
        self.inv_freq.copy_(inv_freq)

    @torch.no_grad()
    def forward(
        self,
        x: torch.Tensor,  # [batch_size, seq_length, num_heads, d_qk]
        position_ids: Optional[torch.LongTensor],  # [batch_size, seq_length]
    ):
        # [FUNCTION DOC]
        # Behavior (no grad): build angles as inverse frequencies times position ids in float32 with autocast disabled,
        # duplicate them along the last dim, and return their cos and sin cast to x's dtype. x is used only for dtype and device.
        # Inputs: x (any tensor), position_ids [batch_size, seq_length] (required in practice).
        # Outputs: tuple (cos, sin), each [batch_size, seq_length, dim]. Exceptions: AttributeError if position_ids is None. Side effects: None.
        # KNOWN ISSUE: position_ids is typed Optional but None is not supported.
        # x: [bs, num_attention_heads, seq_len, head_size]
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()
        # Force float32 since bfloat16 loses precision on long contexts
        # See https://github.com/huggingface/transformers/pull/29285
        device_type = x.device.type
        device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)

    def rotate_half(self, x):
        # [FUNCTION DOC]
        # Behavior: split the last dim into two halves x1 and x2 and return the concatenation of negated x2 and x1.
        # Inputs: x tensor. Outputs: tensor of same shape. Exceptions: None. Side effects: None.
        """Rotates half the hidden dims of the input."""
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    def apply_rotary_pos_emb(self, q, k, cos, sin, unsqueeze_dim=2):
        # [FUNCTION DOC]
        # Behavior: unsqueeze cos and sin at unsqueeze_dim (heads axis) and return q times cos plus rotate_half(q) times sin, and the same for k.
        # Inputs: q, k [batch_size, seq_length, heads, d]; cos, sin [batch_size, seq_length, d]; unsqueeze_dim (int, default 2).
        # Outputs: tuple (q_embed, k_embed). Exceptions: shape errors from broadcasting. Side effects: None.
        # KNOWN ISSUE: docstring says unsqueeze_dim defaults to 1 but the code default is 2.
        """Applies Rotary Position Embedding to the query and key tensors.

        Args:
            q (`torch.Tensor`): The query tensor.
            k (`torch.Tensor`): The key tensor.
            cos (`torch.Tensor`): The cosine part of the rotary embedding.
            sin (`torch.Tensor`): The sine part of the rotary embedding.
            unsqueeze_dim (`int`, *optional*, defaults to 1):
                The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
                sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
                that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
                k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
                cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
                the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
        Returns:
            `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
        """
        cos = cos.unsqueeze(unsqueeze_dim)
        sin = sin.unsqueeze(unsqueeze_dim)
        q_embed = (q * cos) + (self.rotate_half(q) * sin)
        k_embed = (k * cos) + (self.rotate_half(k) * sin)
        return q_embed, k_embed


class GLUActivation(nn.Module):
    # [CLASS DOC]
    # Purpose: gated linear unit: splits merged gate/up projections and returns act(gate) times up.
    # State: act, the activation function looked up from ACT2FN by name.
    def __init__(self, act_fn_name: str):
        # [FUNCTION DOC]
        # Behavior: look up the activation by name. Inputs: act_fn_name (str). Outputs: None. Exceptions: KeyError for unknown names. Side effects: None.
        super().__init__()
        self.act = ACT2FN[act_fn_name]

    def forward(self, merged_states: torch.Tensor):
        # [FUNCTION DOC]
        # Behavior: split the last dim in two equal halves (gate, up) and return act(gate) times up.
        # Inputs: merged_states [..., 2*k]. Outputs: tensor [..., k]. Exceptions: None explicitly raised. Side effects: None.
        gate_states, up_states = torch.split(merged_states, merged_states.shape[-1] // 2, dim=-1)
        return self.act(gate_states) * up_states


class MLP(nn.Module):
    # [CLASS DOC]
    # Purpose: TP-sharded SwiGLU feed-forward block: column-parallel fused gate+up projection (hidden to 2 times
    # intermediate, split into two contiguous chunks so each TP rank holds matching gate/up slices), GLU activation,
    # row-parallel down projection back to hidden.
    # Construction: config (LlamaConfig), parallel_config (ParallelismArgs or None, defaulting to ALL_REDUCE mode and no async comm), tp_pg.
    def __init__(
        self,
        config: LlamaConfig,
        parallel_config: Optional[ParallelismArgs],
        tp_pg: dist.ProcessGroup,
    ):
        # [FUNCTION DOC]
        # Behavior: derive TP mode and async-communication flag (down projection uses async only in REDUCE_SCATTER mode),
        # then build gate_up_proj, down_proj (both without bias) and the activation.
        # Inputs: config, parallel_config, tp_pg. Outputs: None. Exceptions: None explicitly raised. Side effects: parameter allocation.
        # KNOWN ISSUE: parallel_config is treated as optional but tp_recompute_allgather is read from it unconditionally, so None raises AttributeError.
        super().__init__()

        # TODO @thomasw21: refactor so that we store that default in a single place.
        tp_mode = parallel_config.tp_mode if parallel_config is not None else TensorParallelLinearMode.ALL_REDUCE
        tp_linear_async_communication = (
            parallel_config.tp_linear_async_communication if parallel_config is not None else False
        )

        gate_up_contiguous_chunks = (
            config.intermediate_size,  # shape of gate_linear
            config.intermediate_size,  # shape of up_linear
        )
        self.gate_up_proj = TensorParallelColumnLinear(
            config.hidden_size,
            2 * config.intermediate_size,
            pg=tp_pg,
            mode=tp_mode,
            bias=False,
            async_communication=tp_linear_async_communication,
            contiguous_chunks=gate_up_contiguous_chunks,
            tp_recompute_allgather=parallel_config.tp_recompute_allgather,
        )
        self.down_proj = TensorParallelRowLinear(
            config.intermediate_size,
            config.hidden_size,
            pg=tp_pg,
            mode=tp_mode,
            bias=False,
            async_communication=tp_linear_async_communication and tp_mode is TensorParallelLinearMode.REDUCE_SCATTER,
        )
        self.split_silu_mul = GLUActivation(config.hidden_act)

    def forward(self, hidden_states):  # [seq_length, batch_size, hidden_dim]
        # [FUNCTION DOC]
        # Behavior: gate_up projection, GLU activation, down projection (TP communication happens inside the parallel linears).
        # Inputs: hidden_states [seq_length, batch_size, hidden] (or sequence-sharded in REDUCE_SCATTER mode).
        # Outputs: dict with key hidden_states. Exceptions: None explicitly raised. Side effects: TP collectives.
        merged_states = self.gate_up_proj(hidden_states)
        hidden_states = self.down_proj(self.split_silu_mul(merged_states))
        return {"hidden_states": hidden_states}


class CoreAttention(nn.Module):
    # [CLASS DOC]
    # Purpose: variable-length flash attention over packed (unpadded-by-mask) query/key/value tensors.
    # State: d_qk, d_v (head dims), is_using_mup (switches softmax scale to 1/d), checkpoint_attention (False; flash-attn already recomputes).
    def __init__(self, config: LlamaConfig, parallel_config: Optional[ParallelismArgs], layer_idx: int):
        # [FUNCTION DOC]
        # Behavior: check hidden size divisible by heads and store head dims and mup flag.
        # Inputs: config, parallel_config (unused), layer_idx (unused). Outputs: None. Exceptions: AssertionError on indivisible hidden size. Side effects: None.
        super().__init__()
        # TODO @thomasw21: GPT has a weird `d_kv` config which I'm guessing is essentically a `d_qkv`
        assert (
            config.hidden_size % config.num_attention_heads == 0
        ), f"Hidden size {config.hidden_size} must be divisible by number of attention heads {config.num_attention_heads}."
        self.d_qk = config.hidden_size // config.num_attention_heads
        self.d_v = config.hidden_size // config.num_attention_heads
        self.is_using_mup = config.is_using_mup

        self.checkpoint_attention = False  # Because flash_attn already does checkpointing

    @checkpoint_method(attr_name="checkpoint_attention")
    def forward(
        self,
        query_states: torch.Tensor,  # [batch_size * q_length, n_local_q_heads, inner_dim]
        key_states: torch.Tensor,  # [batch_size * kv_length, n_local_kv_heads, inner_dim]
        value_states: torch.Tensor,  # [batch_size * kv_length, n_local_kv_heads, inner_dim]
        q_sequence_mask: torch.Tensor,  # torch.BoolTensor [batch_size, q_length] (can be broadcasted to that size)
        kv_sequence_mask: torch.Tensor,  # torch.BoolTensor [batch_size, kv_length] (can be broadcasted to that size)
    ):
        # [FUNCTION DOC]
        # Behavior: build cumulative sequence-length vectors (int32, leading zero) from the row sums of the query and key
        # masks; use causal attention unless the query length is 1 (KV-cache decoding); softmax scale is 1/d under muP
        # else the flash-attn default 1/sqrt(d); call flash_attn_varlen_func with no dropout using padded lengths as max lengths.
        # Wrapped by checkpoint_method (recompute disabled by attribute).
        # Inputs: query_states [batch*q_len, local_q_heads, d]; key_states and value_states [batch*kv_len, local_kv_heads, d];
        # q_sequence_mask and kv_sequence_mask boolean [batch, len].
        # Outputs: attention output [batch*q_len, local_q_heads, d_v]. Exceptions: errors from flash-attn. Side effects: None.
        # KNOWN ISSUE: the inputs are not actually unpacked by mask; cumulative lengths assume the valid tokens of each row are contiguous at the start of the flattened rows, so left padding or masked holes give wrong results.
        from flash_attn.flash_attn_interface import flash_attn_varlen_func

        # TODO @thomasw21: Compute once, instead of computing for each layers.
        cu_seqlens_q = torch.zeros((q_sequence_mask.shape[0] + 1), dtype=torch.int32, device=query_states.device)
        cu_seqlens_k = torch.zeros((kv_sequence_mask.shape[0] + 1), dtype=torch.int32, device=query_states.device)
        torch.cumsum(q_sequence_mask.sum(-1, dtype=torch.int32), dim=0, dtype=torch.int32, out=cu_seqlens_q[1:])
        torch.cumsum(kv_sequence_mask.sum(-1, dtype=torch.int32), dim=0, dtype=torch.int32, out=cu_seqlens_k[1:])

        # TODO(kunhao): flash attn's causal means that the query can only attend to the keys before it. This is not
        # what we want if we are using kv cache. This is a hack as we always have q_length == 1 when using kv cache.
        causal = False if q_sequence_mask.shape[1] == 1 else True

        # NOTE: this scale is for µTransfer,
        # in SP, we use sqrt(1/d_h)
        softmax_scale = 1 / query_states.shape[-1] if self.is_using_mup else None
        attn_output = flash_attn_varlen_func(
            q=query_states,
            k=key_states,
            v=value_states,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=q_sequence_mask.shape[1],
            max_seqlen_k=kv_sequence_mask.shape[1],
            dropout_p=0.0,
            softmax_scale=softmax_scale,
            causal=causal,
            return_attn_probs=False,
        )

        return attn_output


def pad_to_right(tensor, mask, new_tensor=None):
    # [FUNCTION DOC]
    # HELPER: converts left-padded KV states into right-padded caches during prefill.
    # Behavior: count valid tokens per row, build a right-aligned mask with that many leading True values, gather the
    # valid entries in row order, and write them at the start of each row of new_tensor (zeros-like of tensor if not given).
    # Inputs: tensor [batch, seqlen, d1, d2]; mask boolean [batch, seqlen]; new_tensor optional [batch, new_seqlen, d1, d2] (filled in place).
    # Outputs: tuple (new_tensor, right_padded_mask [batch, seqlen]). Exceptions: None explicitly raised.
    # Side effects: mutates new_tensor in place when provided.
    """Transform a left-padded tensor into a right-padded tensor. (Useful for prefilling key/value states)
    Args:
        tensor: (batch_size, seqlen, d1, d2)
        mask: (batch_size, seqlen)
        new_tensor: (batch_size, new_tensor_seqlen, d1, d2)
    Returns:
        new_tensor: (batch_size, new_tensor_seqlen, d1, d2)
        right_padded_mask: (batch_size, seqlen)
    """
    # First, we need to find the number of padding for each row
    unpad_seqlens = mask.sum(1)
    # Then, we need to find the maximum length of the tensor
    max_seqlen = mask.shape[1]
    # We can then create the indices to select the padded values
    # The indices are the same for each row
    indices = torch.arange(max_seqlen, device=mask.device)
    # We can then create the mask for the padded values
    right_padded_mask = indices < unpad_seqlens[:, None]
    # We select the useful values
    useful_values = tensor[mask]
    # We create the new tensor (if not provided)
    new_tensor = torch.zeros_like(tensor) if new_tensor is None else new_tensor
    # We fill the new tensor with the useful values
    new_tensor[:, : right_padded_mask.shape[1], :, :][right_padded_mask] = useful_values
    return new_tensor, right_padded_mask


class CausalSelfAttention(nn.Module, AttachableStore):
    # [CLASS DOC]
    # Purpose: TP-sharded multi-head/grouped-query self-attention with RoPE, one fused column-parallel QKV projection
    # and a row-parallel output projection, supporting a training path and a KV-cache inference path (selected by
    # whether an AttachableStore is attached).
    # Key attributes: n_local_q_heads and n_local_kv_heads (per TP rank), n_repeats, is_gqa, d_qk, d_v, d_model,
    # qkv_proj, rotary_embedding (interleaved RotaryEmbedding or LlamaRotaryEmbedding, used at inference),
    # flash_rotary_embedding (flash-attn rotary used in training), o_proj, attention (CoreAttention), prefill_kv_len (KV cache length).
    def __init__(
        self,
        config: LlamaConfig,
        parallel_config: Optional[ParallelismArgs],
        tp_pg: dist.ProcessGroup,
        layer_idx: int,
    ):
        # [FUNCTION DOC]
        # Behavior: assert heads and kv heads divide by TP size (if num_key_value_heads is missing, log a warning on rank 0
        # and set it equal to the number of heads, mutating config), assert heads divisible by kv heads; compute local
        # head counts and dims; build the QKV projection with contiguous chunks (q, k, v) so TP sharding splits within each;
        # build the rotary modules, output projection and core attention; set prefill_kv_len to max_position_embeddings.
        # Inputs: config (LlamaConfig), parallel_config, tp_pg, layer_idx. Outputs: None.
        # Exceptions: AssertionError on divisibility violations; AttributeError if parallel_config is None.
        # Side effects: may mutate config; logging; parameter and CUDA buffer allocation.
        from flash_attn.layers.rotary import RotaryEmbedding as FlashRotaryEmbedding

        super().__init__()
        # Tensor parallel considerations: We split tensors along head dimension
        assert (
            config.num_attention_heads % tp_pg.size() == 0
        ), f"Number of attention heads ({config.num_attention_heads}) must be divisible by TP size ({tp_pg.size()})."
        try:
            assert (
                config.num_key_value_heads % tp_pg.size() == 0
            ), f"Number of key/value heads ({config.num_key_value_heads}) must be divisible by TP size ({tp_pg.size()})."
        except AttributeError:
            log_rank(
                "WARNING: num_key_value_heads not defined, assuming it is equal to num_attention_heads",
                logger=logger,
                level=logging.WARNING,
                rank=0,
            )
            # If num_key_value_heads is not defined, we assume that it is equal to num_attention_heads
            config.num_key_value_heads = config.num_attention_heads
        assert (
            config.num_attention_heads % config.num_key_value_heads == 0
        ), f"Number of attention heads ({config.num_attention_heads}) must be divisible by number of key/value heads ({config.num_key_value_heads})."
        self.n_local_q_heads = config.num_attention_heads // tp_pg.size()
        self.n_local_kv_heads = config.num_key_value_heads // tp_pg.size()
        self.n_repeats = config.num_attention_heads // config.num_key_value_heads
        self.is_gqa = config.num_attention_heads != config.num_key_value_heads  # Whether we are using GQA or not
        self.d_qk = config.hidden_size // config.num_attention_heads
        self.d_v = config.hidden_size // config.num_attention_heads
        self.d_model = config.hidden_size
        self.is_using_mup = config.is_using_mup

        # TODO @thomasw21: refactor so that we store that default in a single place.
        tp_mode = parallel_config.tp_mode if parallel_config is not None else TensorParallelLinearMode.ALL_REDUCE
        tp_linear_async_communication = (
            parallel_config.tp_linear_async_communication if parallel_config is not None else False
        )

        # build the slice config for self.qkv for save/load
        # shard are done within the contiguous chunk
        qkv_contiguous_chunks = (
            config.num_attention_heads * self.d_qk,  # shape of q
            config.num_key_value_heads * self.d_qk,  # shape of k
            config.num_key_value_heads * self.d_qk,  # shape of v
        )
        self.qkv_proj = TensorParallelColumnLinear(
            self.d_model,
            config.num_attention_heads * self.d_qk + 2 * config.num_key_value_heads * self.d_qk,
            pg=tp_pg,
            mode=tp_mode,
            bias=False,
            async_communication=tp_linear_async_communication,
            contiguous_chunks=qkv_contiguous_chunks,
            tp_recompute_allgather=parallel_config.tp_recompute_allgather,
        )
        # TODO(kunhao): We want to have only one version per device and not one version per layer.
        if config.rope_interleaved:
            self.rotary_embedding = RotaryEmbedding(
                dim=self.d_qk,
                end=config.max_position_embeddings,
                theta=config.rope_theta,
            )
        else:
            self.rotary_embedding = LlamaRotaryEmbedding(
                dim=self.d_qk,
                end=config.max_position_embeddings,
                theta=config.rope_theta,
            )
        self.rope_interleaved = config.rope_interleaved

        # NOTE: Only supported for training (TODO(fmom): position_ids not supported yet)
        self.flash_rotary_embedding = FlashRotaryEmbedding(
            dim=self.d_qk, base=config.rope_theta, interleaved=config.rope_interleaved
        )

        self.o_proj = TensorParallelRowLinear(
            config.num_attention_heads * self.d_qk,
            self.d_model,
            pg=tp_pg,
            mode=tp_mode,
            bias=False,
            async_communication=tp_linear_async_communication,
        )

        self.attention = CoreAttention(
            config,
            parallel_config=parallel_config,
            layer_idx=layer_idx,
        )

        self.prefill_kv_len = (
            config.max_position_embeddings
        )  # TODO @nouamane: compute based on free memory, because in rope we can surpass max_position_embeddings

    def forward(
        self,
        hidden_states,  # [seq_length, batch_size, hidden_size]
        sequence_mask,  # [batch_size, seq_length]
    ):
        # [FUNCTION DOC]
        # Behavior: project to QKV [seq, batch, ...]. With GQA split into q, k, v by local head counts and reshape each to
        # [batch, seq, heads, d]; otherwise view as [seq, batch, 3, heads, d] and permute to three batch-first tensors.
        # Dispatch to _forward_inference when a local store is attached, else _forward_training.
        # Inputs: hidden_states [seq_length, batch_size, hidden]; sequence_mask [batch_size, seq_length].
        # Outputs: dict with hidden_states [seq_length, batch_size, hidden] and sequence_mask.
        # Exceptions: those of the sub-paths. Side effects: TP collectives in projections.
        # KNOWN ISSUE: the non-GQA view assumes q, k and v are interleaved per position as three equal blocks, consistent with the fused projection only because their chunk sizes are equal.
        qkv_states = self.qkv_proj(
            hidden_states
        )  # [seq_length, batch_size, n_local_q_heads * d_qk + 2 * n_local_kv_heads * d_qk]
        q_length, batch_size, _ = qkv_states.shape

        # Split QKV states
        if self.is_gqa:
            query_states, key_states, value_states = torch.split(
                qkv_states,
                [
                    self.n_local_q_heads * self.d_qk,
                    self.n_local_kv_heads * self.d_qk,
                    self.n_local_kv_heads * self.d_qk,
                ],
                dim=-1,
            )

            query_states = (
                query_states.transpose(0, 1).contiguous().view(batch_size, q_length, self.n_local_q_heads, self.d_qk)
            )
            key_states = (
                key_states.transpose(0, 1).contiguous().view(batch_size, q_length, self.n_local_kv_heads, self.d_qk)
            )
            value_states = (
                value_states.transpose(0, 1).contiguous().view(batch_size, q_length, self.n_local_kv_heads, self.d_qk)
            )
        else:
            query_states, key_states, value_states = (
                qkv_states.view(q_length, batch_size, 3, self.n_local_q_heads, self.d_qk)
                .permute(2, 1, 0, 3, 4)
                .contiguous()
            )  # [3, batch_size, seq_length, n_local_q_heads, d_qk]

        store = self.get_local_store()
        if store is not None:  # Inference case
            return self._forward_inference(
                query_states, key_states, value_states, sequence_mask, batch_size, q_length, store
            )
        else:  # Training case
            return self._forward_training(query_states, key_states, value_states, sequence_mask, batch_size, q_length)

    def _forward_inference(self, query_states, key_states, value_states, sequence_mask, batch_size, q_length, store):
        # [FUNCTION DOC]
        # HELPER: KV-cache generation path for CausalSelfAttention.forward.
        # Behavior: compute position ids (previous offsets plus mask, or cumulative mask minus one on first call) and the
        # new offsets (last position). Apply RoPE. On first call (prefill, no key in store): assert there is no padding in
        # the middle of a row, allocate zero K/V caches of length prefill_kv_len, unpad q/k/v by mask, run causal
        # varlen flash attention, re-pad the output, and right-pad K/V into the caches. On later calls: fetch caches,
        # grow them with zeros if the rotary cache end grew, assert their length equals rotary end, and call
        # flash_attn_with_kvcache with the new k/v and cache_seqlens set to the position offsets. Store updated caches and
        # offsets, then reshape the output to sequence-first and apply o_proj.
        # Inputs: q/k/v batch-first per-head tensors, sequence_mask, batch_size, q_length, store (dict-like AttachableStore).
        # Outputs: dict with hidden_states and sequence_mask.
        # Exceptions: AssertionError if k/v require grad, on middle padding, or on cache-size mismatch.
        # Side effects: mutates store; allocates caches; TP collectives in o_proj.
        # KNOWN ISSUE: the attention and cache logic is nested inside the non-interleaved branch, so with rope_interleaved true the attention output is never computed (UnboundLocalError) and the store is not updated.
        # KNOWN ISSUE: in the non-interleaved branch RoPE is applied twice to query and key states (once before and once inside the duplicated block).
        # KNOWN ISSUE: the grow-cache check compares rotary end before and after a call that only the interleaved RotaryEmbedding can grow; LlamaRotaryEmbedding never grows, so caches cannot exceed max_position_embeddings.
        from flash_attn.flash_attn_interface import flash_attn_with_kvcache

        assert key_states.requires_grad is False
        assert value_states.requires_grad is False

        if "position_offsets" in store:
            old_position_offsets = store["position_offsets"]
            position_ids = old_position_offsets[:, None] + sequence_mask
        else:
            position_ids = torch.cumsum(sequence_mask, dim=-1, dtype=torch.int32) - 1
        position_offsets = position_ids[:, -1]

        # Compute rotary embeddings
        # Note: keep track of old rotary embedding end to check if we need to enlarge k_cache and v_cache
        old_rotary_embed_end = self.rotary_embedding.end
        if self.rope_interleaved:
            query_states = self.rotary_embedding(query_states, position_ids=position_ids)
            key_states = self.rotary_embedding(key_states, position_ids=position_ids)
        else:
            cos, sin = self.rotary_embedding(value_states, position_ids)
            query_states, key_states = self.rotary_embedding.apply_rotary_pos_emb(query_states, key_states, cos, sin)

            # Compute rotary embeddings
            # Note: keep track of old rotary embedding end to check if we need to enlarge k_cache and v_cache
            old_rotary_embed_end = self.rotary_embedding.end
            # interleaved version.
            if self.rope_interleaved:
                query_states = self.rotary_embedding(query_states, position_ids=position_ids)
                key_states = self.rotary_embedding(key_states, position_ids=position_ids)
            # non interleaved version.
            else:
                cos, sin = self.rotary_embedding(value_states, position_ids)
                query_states, key_states = self.rotary_embedding.apply_rotary_pos_emb(
                    query_states, key_states, cos, sin
                )

            if "key" not in store:
                # First inference iteration (Prefill)
                # TODO @nouamane: support custom masking
                # assert that [ False, False, False, False,  True,  True,  True,  True,  True,  True] is accepted
                # but [ False, False, False, False,  True,  True,  False,  False,  True,  True] is not (can't mask in the middle of sequence)
                assert ~(
                    sequence_mask[:, :-1] & (~sequence_mask[:, 1:])  # True is never followed by False
                ).any(), "Can't mask in the middle of sequence, please make sure that pads are at the left of the sequence if existing"

                # preallocate k_cache, v_cache to self.prefill_kv_len
                k_cache = torch.zeros(
                    (
                        batch_size,
                        self.prefill_kv_len,
                        self.n_local_kv_heads,
                        self.d_qk,
                    ),
                    dtype=query_states.dtype,
                    device=query_states.device,
                )
                v_cache = torch.zeros(
                    (batch_size, self.prefill_kv_len, self.n_local_kv_heads, self.d_v),
                    dtype=query_states.dtype,
                    device=query_states.device,
                )
                # Remove pad tokens from key_states and concatenate samples in key_unpad
                # cu_seqlens_k is the cumulative sequence lengths of key_states
                (query_unpad, indices_q, cu_seqlens_q, max_seqlen_q) = bert_padding.unpad_input(
                    query_states,
                    sequence_mask,
                )
                (key_unpad, indices_k, cu_seqlens_k, max_seqlen_k) = bert_padding.unpad_input(
                    key_states, sequence_mask
                )
                (value_unpad, _, _, _) = bert_padding.unpad_input(value_states, sequence_mask)

                # NOTE: this scale is for µTransfer,
                # in SP, we use sqrt(1/d_h)
                softmax_scale = 1 / query_states.shape[-1] if self.is_using_mup else None
                output_unpad = flash_attn_varlen_func(
                    q=query_unpad,  # (total_q, n_local_q_heads, d_qk)
                    k=key_unpad,  # (total_kv, n_local_kv_heads, d_qk)
                    v=value_unpad,  # (total_kv, n_local_kv_heads, d_v)
                    cu_seqlens_q=cu_seqlens_q,
                    cu_seqlens_k=cu_seqlens_k,
                    max_seqlen_q=max_seqlen_q,
                    max_seqlen_k=max_seqlen_k,
                    dropout_p=0.0,
                    softmax_scale=softmax_scale,
                    causal=True,  # True in prefill phase, False in subsequent phases
                    return_attn_probs=False,
                )  # (total_unpadded, n_local_q_heads, d_v)

                attention_output = bert_padding.pad_input(
                    output_unpad, indices_q, batch_size, q_length
                )  # (batch_size, q_length, n_local_q_heads, d_v)

                pad_to_right(key_states, sequence_mask, new_tensor=k_cache)
                pad_to_right(value_states, sequence_mask, new_tensor=v_cache)

            else:
                # Pull pre-computed key/value states
                # Subsequent inference iterations (q_length=1)
                k_cache = store["key"]
                v_cache = store["value"]

                # NOTE(fmom): According to flash_attn_with_kvcache, "If you pass in k / v, you must make sure that the cache is large enough to hold the new values"
                # Since rotary embedding has changed (to enable larger context), we need to enlarge k_cache and v_cache
                if self.rotary_embedding.end > old_rotary_embed_end:
                    k_cache = torch.cat(
                        [
                            k_cache,
                            torch.zeros(
                                (
                                    batch_size,
                                    self.rotary_embedding.end - old_rotary_embed_end,
                                    self.n_local_kv_heads,
                                    self.d_qk,
                                ),
                                dtype=query_states.dtype,
                                device=query_states.device,
                            ),
                        ],
                        dim=1,
                    )

                    v_cache = torch.cat(
                        [
                            v_cache,
                            torch.zeros(
                                (
                                    batch_size,
                                    self.rotary_embedding.end - old_rotary_embed_end,
                                    self.n_local_kv_heads,
                                    self.d_v,
                                ),
                                dtype=query_states.dtype,
                                device=query_states.device,
                            ),
                        ],
                        dim=1,
                    )

                assert (
                    k_cache.shape[1] == self.rotary_embedding.end
                ), f"Cache size {k_cache.shape[1]} is smaller than rotary embedding end {self.rotary_embedding.end}"
                assert (
                    v_cache.shape[1] == self.rotary_embedding.end
                ), f"Cache size {v_cache.shape[1]} is smaller than rotary embedding end {self.rotary_embedding.end}"

                # [batch_size, seq_length, num_heads, d_qk]
                query_states = query_states.view(
                    batch_size, q_length, self.n_local_q_heads, self.d_qk
                )  # [batch_size, q_length, self.n_heads, d_qk]
                kv_length = key_states.shape[1]
                key_states = key_states.view(
                    batch_size, kv_length, self.n_local_kv_heads, self.d_qk
                )  # [batch_size, kv_length, self.n_heads, d_qk]
                value_states = value_states.view(
                    batch_size, kv_length, self.n_local_kv_heads, self.d_v
                )  # [batch_size, kv_length, self.n_heads, d_v]

                # NOTE: this scale is for µTransfer,
                # in SP, we use sqrt(1/d_h)
                softmax_scale = 1 / query_states.shape[-1] if self.is_using_mup else None
                attention_output = flash_attn_with_kvcache(
                    query_states,
                    k_cache,
                    v_cache,
                    key_states,
                    value_states,
                    rotary_cos=None,
                    rotary_sin=None,
                    # TODO @nouamane: seems like this doesn't help to indicate padding in (for first iteration it's just 0)
                    cache_seqlens=position_offsets.contiguous(),
                    softmax_scale=softmax_scale,
                    causal=True,
                    rotary_interleaved=False,  # the value is not used unless rotary_cos/sin is provided. https://github.com/Dao-AILab/flash-attention
                )

            store.update(
                {
                    "key": k_cache,  # flash-attn has updated with new key_states using cache_seqlens
                    "value": v_cache,
                    "position_offsets": position_offsets,
                }
            )

        attention_output = (
            attention_output.contiguous().view(batch_size, q_length, self.n_local_q_heads * self.d_v).transpose(0, 1)
        )
        output = self.o_proj(attention_output)

        return {"hidden_states": output, "sequence_mask": sequence_mask}

    def _forward_training(self, query_states, key_states, value_states, sequence_mask, batch_size, q_length):
        # [FUNCTION DOC]
        # HELPER: training path for CausalSelfAttention.forward.
        # Behavior: stack k and v as [batch, seq, 2, kv_heads, d], apply flash-attn rotary to q and kv jointly (positions
        # 0..seq-1, ignores padding), split k and v back, flatten batch and sequence for q, k, v, run CoreAttention with
        # the same mask for queries and keys, reshape output to [seq, batch, local_heads*d_v] and apply o_proj.
        # Inputs: q/k/v, sequence_mask, batch_size, q_length. Outputs: dict with hidden_states and sequence_mask.
        # Exceptions: None explicitly raised. Side effects: TP collectives in o_proj.
        # Apply rotary embeddings to query/key states
        # NOTE: The layout is different from models/llama.py which is [batch_size, num_heads, seq_length, d_qk]
        # Here it is, [batch_size, seq_length, num_heads, d_qk]
        # [2, batch_size, seq_length, num_heads, d_qk]
        key_value_states = torch.cat([key_states.unsqueeze(0), value_states.unsqueeze(0)], dim=0)
        # [batch_size, seq_length, 2, num_heads, d_qk]
        key_value_states = key_value_states.permute(1, 2, 0, 3, 4).contiguous()
        query_states, key_value_states = self.flash_rotary_embedding(query_states, kv=key_value_states)
        # [batch_size, seq_length, num_heads, d_qk]
        key_states, value_states = torch.split(key_value_states, 1, dim=2)

        q_sequence_mask = sequence_mask
        kv_sequence_mask = sequence_mask

        kv_length = key_states.shape[1]
        # [batch_size, seq_length, num_heads, d_qk]
        # Shaping for use in `flash-attn` version of flash-attn: `flash_attn_unpadded_func`
        query_states = query_states.view(
            batch_size * q_length, self.n_local_q_heads, self.d_qk
        )  # [batch_size * q_length, self.n_heads, d_qk]

        key_states = key_states.view(
            batch_size * kv_length, self.n_local_kv_heads, self.d_qk
        )  # [batch_size * kv_length, self.n_heads, d_qk]
        value_states = value_states.view(
            batch_size * kv_length, self.n_local_kv_heads, self.d_v
        )  # [batch_size * kv_length, self.n_heads, d_v]

        attention_output = self.attention(
            query_states=query_states,
            key_states=key_states,
            value_states=value_states,
            q_sequence_mask=q_sequence_mask,
            kv_sequence_mask=kv_sequence_mask,
        )

        attention_output = (
            attention_output.contiguous().view(batch_size, q_length, self.n_local_q_heads * self.d_v).transpose(0, 1)
        )
        output = self.o_proj(attention_output)

        return {"hidden_states": output, "sequence_mask": sequence_mask}


class LlamaDecoderLayer(nn.Module):
    # [CLASS DOC]
    # Purpose: one pre-norm transformer block: RMSNorm, attention, residual add, RMSNorm, MLP, residual add; optional
    # full-layer activation checkpointing (recompute_layer from parallel_config).
    def __init__(
        self,
        config: LlamaConfig,
        parallel_config: Optional[ParallelismArgs],
        tp_pg: dist.ProcessGroup,
        layer_idx: int,
    ):
        # [FUNCTION DOC]
        # Behavior: build input_layernorm, attn, post_attention_layernorm, mlp, and read recompute_layer.
        # Inputs: config, parallel_config (required in practice), tp_pg, layer_idx. Outputs: None.
        # Exceptions: AttributeError if parallel_config is None. Side effects: parameter allocation.
        super().__init__()
        self.input_layernorm = TritonRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.attn = CausalSelfAttention(
            config=config,
            parallel_config=parallel_config,
            tp_pg=tp_pg,
            layer_idx=layer_idx,
        )

        self.post_attention_layernorm = TritonRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = MLP(config=config, parallel_config=parallel_config, tp_pg=tp_pg)

        self.recompute_layer = parallel_config.recompute_layer

    def _core_forward(
        self,
        hidden_states: Union[torch.Tensor, TensorPointer],
        sequence_mask: Union[torch.Tensor, TensorPointer],
    ) -> List[Union[torch.Tensor, TensorPointer]]:
        # [FUNCTION DOC]
        # HELPER: the uncheckpointed computation of the layer.
        # Behavior: residual plus attention of normalized input; then residual plus MLP of normalized result.
        # Inputs: hidden_states [seq, batch, hidden]; sequence_mask. Outputs: tuple (hidden_states, sequence_mask).
        # Exceptions: None explicitly raised. Side effects: TP collectives.
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        output = self.attn(hidden_states=hidden_states, sequence_mask=sequence_mask)
        hidden_states = output["hidden_states"]
        hidden_states = hidden_states + residual

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states=hidden_states)["hidden_states"]
        hidden_states = hidden_states + residual

        return hidden_states, output["sequence_mask"]

    def _checkpointed_forward(
        self,
        hidden_states: torch.Tensor,
        sequence_mask: torch.Tensor,
    ) -> List[torch.Tensor]:
        # [FUNCTION DOC]
        # HELPER: activation-checkpointed variant.
        # Behavior: run _core_forward through torch CheckpointFunction with RNG state preservation, so activations are recomputed in backward.
        # Inputs/Outputs: as _core_forward. Exceptions: None explicitly raised. Side effects: recomputation in backward; RNG state save/restore.
        return CheckpointFunction.apply(self._core_forward, True, hidden_states, sequence_mask)

    def forward(
        self,
        hidden_states: Union[torch.Tensor, TensorPointer],
        sequence_mask: Union[torch.Tensor, TensorPointer],
    ) -> Dict[str, Union[torch.Tensor, TensorPointer]]:
        # [FUNCTION DOC]
        # Behavior: use the checkpointed path when recompute_layer is set and the input is a real tensor (not a TensorPointer), else the core path; return a dict.
        # Inputs: hidden_states and sequence_mask (tensors or TensorPointers). Outputs: dict with hidden_states and sequence_mask.
        # Exceptions: None explicitly raised. Side effects: as the chosen path.
        if self.recompute_layer and not isinstance(hidden_states, TensorPointer):
            hidden_states, sequence_mask = self._checkpointed_forward(hidden_states, sequence_mask)
        else:
            hidden_states, sequence_mask = self._core_forward(hidden_states, sequence_mask)

        return {
            "hidden_states": hidden_states,
            "sequence_mask": sequence_mask,
        }


class Embedding(nn.Module, AttachableStore):
    # [CLASS DOC]
    # Purpose: TP-sharded (vocab-parallel) token embedding; during inference also tracks past_length in its AttachableStore.
    # State: token_embedding (TensorParallelEmbedding), pg.
    def __init__(self, tp_pg: dist.ProcessGroup, config: LlamaConfig, parallel_config: Optional[ParallelismArgs]):
        # [FUNCTION DOC]
        # Behavior: build the vocab-parallel embedding with the pad token as padding_idx and the TP mode (default ALL_REDUCE).
        # Inputs: tp_pg, config, parallel_config (optional). Outputs: None. Exceptions: None explicitly raised. Side effects: parameter allocation.
        super().__init__()
        self.token_embedding = TensorParallelEmbedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.hidden_size,
            padding_idx=config.pad_token_id,
            pg=tp_pg,
            mode=parallel_config.tp_mode if parallel_config is not None else TensorParallelLinearMode.ALL_REDUCE,
        )
        self.pg = tp_pg

    def forward(self, input_ids: torch.Tensor, input_mask: torch.Tensor):  # [batch_size, seq_length]
        # [FUNCTION DOC]
        # Behavior: if a store is attached, set its past_length to the previous value (zeros on first call) plus the number of valid tokens per row.
        # Transpose input_ids to [seq, batch] and embed.
        # Inputs: input_ids and input_mask [batch, seq]. Outputs: dict with input_embeds [seq, batch, hidden].
        # Exceptions: None explicitly raised. Side effects: mutates store; TP collective inside the embedding.
        store = self.get_local_store()
        if store is not None:
            if "past_length" in store:
                past_length = store["past_length"]
            else:
                past_length = torch.zeros(1, dtype=torch.long, device=input_ids.device).expand(input_ids.shape[0])

            cumsum_mask = input_mask.cumsum(-1, dtype=torch.long)
            # Store new past_length in store
            store["past_length"] = past_length + cumsum_mask[:, -1]

        # Format input in `[seq_length, batch_size]` to support high TP with low batch_size
        input_ids = input_ids.transpose(0, 1)
        input_embeds = self.token_embedding(input_ids)
        return {"input_embeds": input_embeds}


class LlamaModel(nn.Module):
    # [CLASS DOC]
    # Purpose: builds the pipeline graph as a sequence of PipelineBlocks: token_position_embeddings, decoder (one block
    # per layer), final_layer_norm, lm_head (column-parallel, producing vocab-sharded logits) and cast_to_fp32.
    # Each block only materializes its module on the PP rank assigned by build_model; others hold TensorPointers.
    # State: p2p (P2P communicator over pp_pg on cuda), config, parallel_config, parallel_context, tp_mode.
    """Build pipeline graph"""

    def __init__(
        self,
        config: LlamaConfig,
        parallel_context: ParallelContext,
        parallel_config: Optional[ParallelismArgs],
    ):
        # [FUNCTION DOC]
        # Behavior: create the P2P helper and all PipelineBlocks with their module builders, kwargs and input/output keys;
        # log the RoPE theta (and a warning-style info about interleaved RoPE) on rank 0.
        # Inputs: config (LlamaConfig), parallel_context, parallel_config. Outputs: None.
        # Exceptions: AttributeError if parallel_config is None (tp_recompute_allgather). Side effects: logging; P2P setup.
        super().__init__()

        # Declare all the nodes
        self.p2p = P2P(parallel_context.pp_pg, device=torch.device("cuda"))
        self.config = config
        self.parallel_config = parallel_config
        self.parallel_context = parallel_context
        self.tp_mode = parallel_config.tp_mode if parallel_config is not None else TensorParallelLinearMode.ALL_REDUCE
        tp_linear_async_communication = (
            parallel_config.tp_linear_async_communication if parallel_config is not None else False
        )

        self.token_position_embeddings = PipelineBlock(
            p2p=self.p2p,
            module_builder=Embedding,
            module_kwargs={
                "tp_pg": parallel_context.tp_pg,
                "config": config,
                "parallel_config": parallel_config,
            },
            module_input_keys={"input_ids", "input_mask"},
            module_output_keys={"input_embeds"},
        )
        log_rank(f"Initialize RoPE Theta = {config.rope_theta}", logger=logger, level=logging.INFO, rank=0)
        if config.rope_interleaved:
            log_rank(
                "The RoPE interleaved version differs from the Transformers implementation. It's better to set rope_interleaved=False if you need to convert the weights to Transformers",
                logger=logger,
                level=logging.INFO,
                rank=0,
            )
        self.decoder = nn.ModuleList(
            [
                PipelineBlock(
                    p2p=self.p2p,
                    module_builder=LlamaDecoderLayer,
                    module_kwargs={
                        "config": config,
                        "parallel_config": parallel_config,
                        "tp_pg": parallel_context.tp_pg,
                        "layer_idx": layer_idx,
                    },
                    module_input_keys={"hidden_states", "sequence_mask"},
                    module_output_keys={"hidden_states", "sequence_mask"},
                )
                for layer_idx in range(config.num_hidden_layers)
            ]
        )

        self.final_layer_norm = PipelineBlock(
            p2p=self.p2p,
            module_builder=TritonRMSNorm,
            module_kwargs={"hidden_size": config.hidden_size, "eps": config.rms_norm_eps},
            module_input_keys={"input"},
            module_output_keys={"hidden_states"},
        )  # TODO

        self.lm_head = PipelineBlock(
            p2p=self.p2p,
            # Understand that this means that we return sharded logits that are going to need to be gathered
            module_builder=TensorParallelColumnLinear,
            module_kwargs={
                "in_features": config.hidden_size,
                "out_features": config.vocab_size,
                "pg": parallel_context.tp_pg,
                "bias": False,
                # TODO @thomasw21: refactor so that we store that default in a single place.
                "mode": self.tp_mode,
                "async_communication": tp_linear_async_communication,
                "tp_recompute_allgather": parallel_config.tp_recompute_allgather,
            },
            module_input_keys={"x"},
            module_output_keys={"logits"},
        )

        self.cast_to_fp32 = PipelineBlock(
            p2p=self.p2p,
            module_builder=lambda: lambda x: x.float(),
            module_kwargs={},
            module_input_keys={"x"},
            module_output_keys={"output"},
        )

    def forward(
        self,
        input_ids: Union[torch.Tensor, TensorPointer],  # [batch_size, seq_length]
        input_mask: Union[torch.Tensor, TensorPointer],  # [batch_size, seq_length]
    ):
        # [FUNCTION DOC]
        # Behavior: return only the fp32 sharded logits of forward_with_hidden_states.
        # Inputs: input_ids, input_mask (tensors or TensorPointers). Outputs: logits [seq, batch, vocab/tp] in float32 (or TensorPointer).
        # Exceptions: None explicitly raised. Side effects: PP point-to-point sends/receives, TP collectives.
        return self.forward_with_hidden_states(input_ids=input_ids, input_mask=input_mask)[0]

    def forward_with_hidden_states(
        self,
        input_ids: Union[torch.Tensor, TensorPointer],  # [batch_size, seq_length]
        input_mask: Union[torch.Tensor, TensorPointer],  # [batch_size, seq_length]
    ):
        # [FUNCTION DOC]
        # Behavior: embed, run all decoder blocks passing hidden_states and sequence_mask, final norm, lm_head, cast to fp32.
        # Pipeline blocks transparently send/receive activations between PP ranks.
        # Inputs: input_ids, input_mask. Outputs: tuple (fp32 sharded logits, final hidden states).
        # Exceptions: None explicitly raised. Side effects: PP p2p and TP communication.
        # all tensors are optional as most ranks don't need anything from the dataloader.
        output = self.token_position_embeddings(input_ids=input_ids, input_mask=input_mask)

        hidden_encoder_states = {
            "hidden_states": output["input_embeds"],
            "sequence_mask": input_mask,
        }
        for encoder_block in self.decoder:
            hidden_encoder_states = encoder_block(**hidden_encoder_states)

        hidden_states = self.final_layer_norm(input=hidden_encoder_states["hidden_states"])["hidden_states"]

        sharded_logits = self.lm_head(x=hidden_states)["logits"]

        fp32_sharded_logits = self.cast_to_fp32(x=sharded_logits)["output"]

        return fp32_sharded_logits, hidden_states

    def get_block_compute_costs(self):
        # [FUNCTION DOC]
        # Behavior: return a dict mapping module builder to relative cost: decoder layer cost is 4 times heads times head
        # dim times hidden plus 3 times ffn times hidden; lm_head cost is vocab times hidden. Used by build_model for PP balancing.
        # Inputs: none. Outputs: dict. Exceptions: None. Side effects: None.
        # KNOWN ISSUE: the attention cost ignores GQA (assumes kv heads equal query heads).
        """Computes the compute cost of each block in the model so that we can do a better job of load balancing."""
        model_config = self.config
        d_ff = model_config.intermediate_size
        d_qkv = model_config.hidden_size // model_config.num_attention_heads
        block_compute_costs = {
            # CausalSelfAttention (qkv proj + attn out) + MLP
            LlamaDecoderLayer: 4 * model_config.num_attention_heads * d_qkv * model_config.hidden_size
            + 3 * d_ff * model_config.hidden_size,
            # This is the last lm_head
            TensorParallelColumnLinear: model_config.vocab_size * model_config.hidden_size,
        }
        return block_compute_costs

    def get_flops_per_sec(self, iteration_time_in_sec, sequence_length, global_batch_size):
        # [FUNCTION DOC]
        # Behavior: compute model and hardware flops with get_flops for the global batch and divide by iteration time,
        # world size and 1e12, giving TFLOPs per second per GPU.
        # Inputs: iteration_time_in_sec (float), sequence_length (int), global_batch_size (int).
        # Outputs: tuple (model TFLOPs/s per GPU, hardware TFLOPs/s per GPU). Exceptions: ZeroDivisionError for zero time. Side effects: None.
        """Get flops per second for a given model"""
        world_size = self.parallel_context.world_pg.size()
        try:
            num_key_values_heads = self.config.num_key_value_heads
        except AttributeError:
            num_key_values_heads = self.config.num_attention_heads

        model_flops, hardware_flops = get_flops(
            num_layers=self.config.num_hidden_layers,
            hidden_size=self.config.hidden_size,
            num_heads=self.config.num_attention_heads,
            num_key_value_heads=num_key_values_heads,
            vocab_size=self.config.vocab_size,
            ffn_hidden_size=self.config.intermediate_size,
            seq_len=sequence_length,
            batch_size=global_batch_size,
        )

        model_flops_per_s = model_flops / (iteration_time_in_sec * world_size * 1e12)
        hardware_flops_per_s = hardware_flops / (iteration_time_in_sec * world_size * 1e12)
        return model_flops_per_s, hardware_flops_per_s


@torch.jit.script
def masked_mean(loss, label_mask, dtype):
    # [FUNCTION DOC]
    # HELPER: TorchScript-compiled masked average used by the losses.
    # Behavior: sum of loss times mask accumulated in dtype, divided by the mask sum.
    # Inputs: loss and label_mask tensors of the same shape; dtype. Outputs: 0-dim tensor.
    # Exceptions: None explicitly raised. Side effects: None.
    # KNOWN ISSUE: returns NaN when the mask is all zeros.
    # type: (Tensor, Tensor, torch.dtype) -> Tensor
    return (loss * label_mask).sum(dtype=dtype) / label_mask.sum()


class Loss(nn.Module):
    # [CLASS DOC]
    # Purpose: vocab-parallel cross-entropy loss averaged over masked tokens. State: tp_pg.
    def __init__(self, tp_pg: dist.ProcessGroup):
        # [FUNCTION DOC]
        # Behavior: store tp_pg. Inputs: tp_pg. Outputs: None. Exceptions: None. Side effects: None.
        super().__init__()
        self.tp_pg = tp_pg

    def forward(
        self,
        sharded_logits: torch.Tensor,  # [seq_length, batch_size, logits]
        label_ids: torch.Tensor,  # [batch_size, seq_length]
        label_mask: torch.Tensor,  # [batch_size, seq_length]
    ) -> Dict[str, torch.Tensor]:
        # [FUNCTION DOC]
        # Behavior: compute per-token cross entropy in float32 from TP-sharded logits (sharded_cross_entropy all-reduces
        # over tp_pg) against labels transposed to sequence-first, transpose back, and take masked_mean.
        # Inputs: sharded_logits [seq, batch, vocab/tp]; label_ids and label_mask [batch, seq].
        # Outputs: dict with loss (0-dim float tensor). Exceptions: None explicitly raised. Side effects: TP collectives.
        loss = sharded_cross_entropy(
            sharded_logits,
            label_ids.transpose(0, 1).contiguous(),
            group=self.tp_pg,
            dtype=torch.float,
        ).transpose(0, 1)
        loss = masked_mean(loss, label_mask, dtype=torch.float)
        return {"loss": loss}


class LossWithZLoss(Loss):
    # [CLASS DOC]
    # Purpose: Loss plus the z-loss regularizer (penalty on log-sum-exp of logits) computed inside sharded_cross_entropy.
    # State: z_loss_coef.
    def __init__(self, tp_pg: dist.ProcessGroup, z_loss_coefficient: float):
        # [FUNCTION DOC]
        # Behavior: initialize Loss and store the coefficient. Inputs: tp_pg, z_loss_coefficient (float). Outputs: None. Exceptions: None. Side effects: None.
        super().__init__(tp_pg)
        self.z_loss_coef = z_loss_coefficient

    def forward(
        self,
        sharded_logits: torch.Tensor,  # [seq_length, batch_size, logits]
        label_ids: torch.Tensor,  # [batch_size, seq_length]
        label_mask: torch.Tensor,  # [batch_size, seq_length]
    ) -> Dict[str, torch.Tensor]:
        # [FUNCTION DOC]
        # Behavior: call sharded_cross_entropy with the z-loss coefficient, getting per-token loss and z-loss; average both
        # by mask; the z-loss is detached so it is reported only (any gradient effect comes from inside the loss term).
        # Inputs: as Loss.forward. Outputs: dict with loss and z_loss. Exceptions: None explicitly raised. Side effects: TP collectives.
        loss, z_loss = sharded_cross_entropy(
            sharded_logits,
            label_ids.transpose(0, 1).contiguous(),
            group=self.tp_pg,
            dtype=torch.float,
            z_loss_coef=self.z_loss_coef,
        )
        loss = masked_mean(loss.transpose(0, 1), label_mask, dtype=torch.float)
        z_loss = masked_mean(z_loss.detach().transpose(0, 1), label_mask, dtype=torch.float)
        return {"loss": loss, "z_loss": z_loss}


class LlamaForTraining(NanotronModel):
    # [CLASS DOC]
    # Purpose: the NanotronModel wrapping LlamaModel plus a loss PipelineBlock (Loss or LossWithZLoss depending on
    # config.z_loss_enabled); the object built by the trainer through build_model.
    # State: model, loss, parallel_context, config, parallel_config.
    def __init__(
        self,
        config: LlamaConfig,
        parallel_context: ParallelContext,
        parallel_config: Optional[ParallelismArgs],
        random_states: Optional[RandomStates] = None,
    ):
        # [FUNCTION DOC]
        # Behavior: build LlamaModel and the loss PipelineBlock (sharing the model's P2P) with output keys loss (and z_loss).
        # Inputs: config, parallel_context, parallel_config, random_states (unused). Outputs: None.
        # Exceptions: None explicitly raised. Side effects: allocation.
        super().__init__()
        self.model = LlamaModel(config=config, parallel_context=parallel_context, parallel_config=parallel_config)

        # Choose the appropriate loss class based on config
        loss_kwargs = {
            "tp_pg": parallel_context.tp_pg,
        }
        if config.z_loss_enabled:
            loss_kwargs["z_loss_coefficient"] = config.z_loss_coefficient

        self.loss = PipelineBlock(
            p2p=self.model.p2p,
            module_builder=LossWithZLoss if config.z_loss_enabled else Loss,
            module_kwargs=loss_kwargs,
            module_input_keys={
                "sharded_logits",
                "label_ids",
                "label_mask",
            },
            module_output_keys={"loss", "z_loss"} if config.z_loss_enabled else {"loss"},
        )

        self.parallel_context = parallel_context
        self.config = config
        self.parallel_config = parallel_config

    def forward(
        self,
        input_ids: Union[torch.Tensor, TensorPointer],
        input_mask: Union[torch.Tensor, TensorPointer],
        label_ids: Union[torch.Tensor, TensorPointer],
        label_mask: Union[torch.Tensor, TensorPointer],
    ) -> Dict[str, Union[torch.Tensor, TensorPointer]]:
        # [FUNCTION DOC]
        # Behavior: compute sharded logits via the model, compute the loss block, and return loss (and z_loss when enabled).
        # Inputs: input_ids, input_mask, label_ids, label_mask (tensors or TensorPointers depending on PP rank).
        # Outputs: dict of loss tensors (TensorPointers on non-output ranks). Exceptions: None explicitly raised.
        # Side effects: PP and TP communication.
        sharded_logits = self.model(
            input_ids=input_ids,
            input_mask=input_mask,
        )
        loss = self.loss(
            sharded_logits=sharded_logits,
            label_ids=label_ids,
            label_mask=label_mask,
        )
        if self.config.z_loss_enabled:
            return {"loss": loss["loss"], "z_loss": loss["z_loss"]}
        else:
            return {"loss": loss["loss"]}

    @torch.no_grad()
    def init_model_randomly(self, config: Config):
        # [FUNCTION DOC]
        # Behavior (no grad): pick StandardParametrizator for RandomInit or SpectralMupParametrizator for SpectralMupInit;
        # log the choice; build a module-id-to-prefix map; for each named parameter compute its full name (tied parameters
        # use their shared tied name), skip it if already initialized, otherwise call the parametrizator on the owning
        # submodule; finally assert the set of initialized names equals the expected set.
        # Inputs: config (full Config with model.init_method). Outputs: None.
        # Exceptions: ValueError for unknown init method; AssertionError for non-NanotronParameter or name mismatch.
        # Side effects: overwrites parameter values in place; uses the current RNG state; logging.
        # KNOWN ISSUE: passes config.model to the parametrizator, but StandardParametrizator and SpectralMupParametrizator expect the full Config (they read model and parallelism sections), so this likely fails with AttributeError.
        """Initialize model parameters randomly.
        Note:
            Layernorm weight all 0 or 1 depending on `apply_layernorm_1p`
        """
        init_method = config.model.init_method
        if isinstance(init_method, RandomInit):
            parametrizator_cls = StandardParametrizator
        elif isinstance(init_method, SpectralMupInit):
            parametrizator_cls = SpectralMupParametrizator
        else:
            raise ValueError(f"Unknown init method {init_method}")

        parametrizator = parametrizator_cls(config=config.model)

        log_rank(
            f"Parametrizing model parameters using {parametrizator.__class__.__name__}",
            logger=logger,
            level=logging.INFO,
            rank=0,
        )

        model = self
        initialized_parameters = set()
        # Handle tensor parallelism
        module_id_to_prefix = {id(module): f"{module_name}." for module_name, module in model.named_modules()}
        # Fix the root_model
        module_id_to_prefix[id(model)] = ""

        for param_name, param in model.named_parameters():
            assert isinstance(param, NanotronParameter)

            module_name, param_name = param_name.rsplit(".", 1)

            if param.is_tied:
                tied_info = param.get_tied_info()
                full_param_name = tied_info.get_full_name_from_module_id_to_prefix(
                    module_id_to_prefix=module_id_to_prefix
                )
            else:
                full_param_name = f"{module_name}.{param_name}"

            if full_param_name in initialized_parameters:
                # Already initialized
                continue

            module = model.get_submodule(module_name)
            parametrizator.parametrize(param_name, module)

            assert full_param_name not in initialized_parameters
            initialized_parameters.add(full_param_name)

        assert initialized_parameters == {
            param.get_tied_info().get_full_name_from_module_id_to_prefix(module_id_to_prefix=module_id_to_prefix)
            if param.is_tied
            else name
            for name, param in model.named_parameters()
        }, f"Somehow the initialized set of parameters don't match:\n - Expected: { {name for name, _ in model.named_parameters()} }\n - Got: {initialized_parameters}"

    def get_embeddings_lm_head_tied_names(self):
        # [FUNCTION DOC]
        # Behavior: return the embedding and lm_head weight names when tie_word_embeddings is true, else an empty list.
        # Inputs: none. Outputs: list of strings. Exceptions: None. Side effects: None.
        """Get the names of the tied embeddings and lm_head weights"""
        if self.config.tie_word_embeddings is True:
            return ["model.token_position_embeddings.pp_block.token_embedding.weight", "model.lm_head.pp_block.weight"]
        else:
            return []

    def get_block_compute_costs(self):
        # [FUNCTION DOC]
        # Behavior: delegate to LlamaModel. Inputs: none. Outputs: dict of module class to cost. Exceptions: None. Side effects: None.
        """Computes the compute cost of each block in the model so that we can do a better job of load balancing."""
        return self.model.get_block_compute_costs()

    def get_flops_per_sec(self, iteration_time_in_sec, sequence_length, global_batch_size):
        # [FUNCTION DOC]
        # Behavior: delegate to LlamaModel.get_flops_per_sec. Inputs/Outputs: as there. Exceptions: as there. Side effects: None.
        """Get flops per second for a given model"""
        return self.model.get_flops_per_sec(iteration_time_in_sec, sequence_length, global_batch_size)


def get_flops(
    num_layers,
    hidden_size,
    num_heads,
    num_key_value_heads,
    vocab_size,
    seq_len,
    ffn_hidden_size,
    batch_size=1,
):
    # [FUNCTION DOC]
    # Behavior: count forward flops (2 per multiply-add) for a decoder-only model: QKV projection (q plus 2 times kv
    # heads), QK logits and attention-times-V (both quadratic in sequence length), attention output projection, gated
    # FFN first layer (counted as 4 because of gate and up) and second layer, all times layers, plus the lm_head.
    # Model flops are 3 times the forward flops (forward plus twice for backward); hardware flops are set equal (placeholder).
    # Inputs: num_layers, hidden_size, num_heads, num_key_value_heads (None means equal to num_heads), vocab_size,
    # seq_len, ffn_hidden_size, batch_size (default 1). Outputs: tuple (model_flops, hardware_flops) as numbers.
    # Exceptions: None explicitly raised. Side effects: None.
    # KNOWN ISSUE: hardware_flops ignores activation recomputation, contrary to the referenced paper section.
    """Counts flops in an decoder-only model
    Args:
        num_layers: number of decoder layers
        hidden_size: hidden size of the model
        num_heads: number of heads in the model
        num_key_value_heads: number of key/value heads in the model
        ffn_hidden_size: hidden size of the FFN
        vocab_size: size of the vocabulary
        seq_len: sequence length of the decoder
        batch_size: batch size
    Returns:
        model_flops: flops in the model (should be independent of the hardware and model implementation)
        hardware_flops: flops in the hardware (actual flops performed on the hardware). Check 6.3 in https://arxiv.org/pdf/2205.05198.pdf
    """
    if num_key_value_heads is None:
        num_key_value_heads = num_heads
    hidden_size_per_head = hidden_size // num_heads
    # In the following we mark the reduced dimension with parentheses
    # decoder
    # self attention
    ## qkv projection
    decoder_qkv_proj_flops_fwd = (
        2 * num_layers * batch_size * seq_len * (hidden_size) * num_heads * hidden_size_per_head
        + 2 * num_layers * batch_size * seq_len * (hidden_size) * 2 * num_key_value_heads * hidden_size_per_head
    )
    ## qk logits
    decoder_qk_logits_flops_fwd = 2 * num_layers * batch_size * num_heads * seq_len * (hidden_size_per_head) * seq_len
    ## v logits
    decoder_v_logits_flops_fwd = 2 * num_layers * batch_size * num_heads * seq_len * (seq_len) * hidden_size_per_head
    ## attn out
    decoder_attn_out_flops_fwd = (
        2 * num_layers * batch_size * num_heads * seq_len * (hidden_size_per_head) * hidden_size
    )
    # FF
    ## 1st layer
    decoder_ffn_1_flops_fwd = 4 * num_layers * batch_size * seq_len * (hidden_size) * ffn_hidden_size
    ## 2nd layer
    decoder_ffn_2_flops_fwd = 2 * num_layers * batch_size * seq_len * (ffn_hidden_size) * hidden_size

    decoder_flops_fwd = (
        decoder_qkv_proj_flops_fwd
        + decoder_qk_logits_flops_fwd
        + decoder_v_logits_flops_fwd
        + decoder_attn_out_flops_fwd
        + decoder_ffn_1_flops_fwd
        + decoder_ffn_2_flops_fwd
    )

    # lm head
    lm_head_flops_fwd = 2 * batch_size * seq_len * (hidden_size) * vocab_size

    # the bwd pass requires double the flops in case of matmuls to calculate the gradients with respect to
    # both input and weight tensors
    model_flops = 3 * (decoder_flops_fwd + lm_head_flops_fwd)  # 1 for fwd + 2 for bwd

    hardware_flops = model_flops  # TODO: This is a placeholder for now

    return model_flops, hardware_flops
