# coding=utf-8
# [MODULE DOC]
# Purpose: nanotron implementation of LlaMoE, a Llama-style decoder where each MLP is replaced by a dropless
# mixture-of-experts layer (dMoE from the sibling moe.py, supporting expert parallelism). Supports tensor
# parallelism (head-sharded attention, vocab-sharded embedding/lm_head), pipeline parallelism (PipelineBlocks) and
# KV-cached generation via AttachableStore. Activations are sequence-first (seq, batch, hidden) between blocks.
# Contents: RotaryEmbedding, CoreAttention (flash-attn varlen), pad_to_right, CausalSelfAttention, LlaMoEDecoderLayer,
# Embedding, LlaMoEModel, masked_mean, Loss, LlaMoEForTraining, get_flops.
# Import-time side effects: imports flash_attn and the local config_llamoe module (which itself prints a parameter
# count and creates a checkpoints directory at import) and moe; compiles masked_mean with TorchScript.
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
""" PyTorch LLaMa MoE model."""
import math
from typing import Dict, Optional, Union, List

import torch
from config_llamoe import LlaMoEConfig
from flash_attn import bert_padding
from flash_attn.flash_attn_interface import (
    flash_attn_varlen_func,
    flash_attn_with_kvcache,
)
from flash_attn.layers.rotary import RotaryEmbedding as FlashRotaryEmbedding
from moe import dMoE
from nanotron import distributed as dist
from nanotron import logging
from nanotron.config import ParallelismArgs
from nanotron.generation.generate_store import AttachableStore
from nanotron.logging import log_rank
from nanotron.models import NanotronModel
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
from nanotron.utils import checkpoint_method
from torch import nn
from torch.nn import init

logger = logging.get_logger(__name__)


class RotaryEmbedding(nn.Module):
    # [CLASS DOC]
    # Rotary position embedding using a lazily built CUDA buffer freqs_cis of shape (end, dim / 2, 2) storing cos/sin
    # pairs as real floats (complex buffers break DDP). Used on the inference path; pairs adjacent channels (interleaved).
    # Attributes: dim (even head dim), end (current max positions, doubled on demand), theta (base, 10000),
    # freqs_cis, _initialized_buffer flag.
    def __init__(self, dim: int, end: int, theta: float = 10000.0):
        # [FUNCTION DOC]
        # Behavior: asserts dim even and stores dim, end, theta; buffer is created later.
        # Inputs: dim (int); end (int max positions); theta (float, default 10000.0). Outputs: None.
        # Exceptions: AssertionError if dim is odd. Side effects: None.
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
        # Behavior: no-op if already initialised. Registers a non-persistent float CUDA buffer, computes inverse frequencies
        # theta to the power of minus 2i / dim, the outer product with positions 0..end-1, converts to unit complex numbers and
        # stores their real/imag parts; marks initialised.
        # Inputs: none. Outputs: None. Exceptions: AssertionError if buffer not on CUDA or not float.
        # Side effects: allocates a CUDA buffer on the module.
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
            self.theta
            ** (torch.arange(0, self.dim, 2, dtype=torch.float, device="cuda")[: (self.dim // 2)] / self.dim)
        )
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
        # Behavior: doubles end (and invalidates the buffer) while the last position id or the sequence length reaches it,
        # printing and rebuilding the buffer if needed. Views x as complex pairs (bf16 upcast to float), selects frequencies
        # for positions 0..seq-1 (no position_ids) or by position_ids, multiplies, and returns in the input dtype.
        # Inputs: x (batch, seq, heads, d_qk); position_ids (optional (batch, seq) long tensor).
        # Outputs: rotated tensor with the shape and dtype of x.
        # Exceptions: AssertionError if d_qk odd; ValueError if the last position id is negative or out of range.
        # Side effects: may mutate end and rebuild the buffer; stdout print; possible CPU-GPU sync.
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


class CoreAttention(nn.Module):
    # [CLASS DOC]
    # Thin flash-attention wrapper for training on packed (varlen) sequences. Attributes: d_qk, d_v (hidden / heads),
    # checkpoint_attention (False; flash-attn already recomputes). Construction asserts hidden_size divisible by heads.
    def __init__(self, config: LlaMoEConfig, parallel_config: Optional[ParallelismArgs], layer_idx: int):
        # [FUNCTION DOC]
        # Behavior: asserts divisibility, stores head dims, disables activation checkpointing.
        # Inputs: config (LlaMoEConfig); parallel_config (optional, unused); layer_idx (int, unused). Outputs: None.
        # Exceptions: AssertionError if hidden_size is not divisible by num_attention_heads. Side effects: None.
        super().__init__()
        assert (
            config.hidden_size % config.num_attention_heads == 0
        ), f"Hidden size {config.hidden_size} must be divisible by number of attention heads {config.num_attention_heads}."
        self.d_qk = config.hidden_size // config.num_attention_heads
        self.d_v = config.hidden_size // config.num_attention_heads

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
        # Behavior: builds int32 cumulative sequence lengths from the q and kv masks (prefix zero), sets causal unless the
        # query length is 1, and calls flash_attn_varlen_func with no dropout and default scale. Wrapped by checkpoint_method
        # (inactive since checkpoint_attention is False).
        # Inputs: query_states (batch*q_len, local q heads, d_qk); key_states and value_states (batch*kv_len, local kv heads,
        # dim); q_sequence_mask (batch, q_len bool); kv_sequence_mask (batch, kv_len bool).
        # Outputs: attention output (batch*q_len, local q heads, d_v).
        # Exceptions: None explicitly raised. Side effects: CUDA kernels.
        # KNOWN ISSUE: cu_seqlens are derived from mask sums but q/k are the full unpacked batch*len tensors, so this is only
        # correct when masks are all true (no padding).
        # TODO @thomasw21: Compute once, instead of computing for each layers.
        cu_seqlens_q = torch.zeros((q_sequence_mask.shape[0] + 1), dtype=torch.int32, device=query_states.device)
        cu_seqlens_k = torch.zeros((kv_sequence_mask.shape[0] + 1), dtype=torch.int32, device=query_states.device)
        torch.cumsum(q_sequence_mask.sum(-1, dtype=torch.int32), dim=0, dtype=torch.int32, out=cu_seqlens_q[1:])
        torch.cumsum(kv_sequence_mask.sum(-1, dtype=torch.int32), dim=0, dtype=torch.int32, out=cu_seqlens_k[1:])

        # TODO(kunhao): flash attn's causal means that the query can only attend to the keys before it. This is not
        # what we want if we are using kv cache. This is a hack as we always have q_length == 1 when using kv cache.
        causal = False if q_sequence_mask.shape[1] == 1 else True
        attn_output = flash_attn_varlen_func(
            q=query_states,
            k=key_states,
            v=value_states,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=q_sequence_mask.shape[1],
            max_seqlen_k=kv_sequence_mask.shape[1],
            dropout_p=0.0,
            softmax_scale=None,  # This already defaults to the scale I'm interested in
            causal=causal,
            return_attn_probs=False,
        )

        return attn_output


def pad_to_right(tensor, mask, new_tensor=None):
    # [FUNCTION DOC]
    # HELPER: converts left-padded key/value states into right-padded layout for the KV cache during prefill.
    # Behavior: counts valid tokens per row, builds a right-padded mask (positions below the count), gathers the valid
    # entries of tensor and scatters them into the first positions of new_tensor (zeros like tensor if not given).
    # Inputs: tensor (batch, seq, d1, d2); mask (batch, seq bool); new_tensor (optional (batch, longer seq, d1, d2)).
    # Outputs: tuple (new_tensor, right_padded_mask (batch, seq)).
    # Exceptions: None explicitly raised. Side effects: writes into new_tensor in place when provided.
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
    # Multi-head / grouped-query self-attention with TP sharding over heads. qkv_proj is column parallel (q, k, v
    # contiguous chunks so sharding happens within each), o_proj row parallel. Training uses flash-attn rotary
    # (interleaved) plus CoreAttention; inference (store attached) uses RotaryEmbedding with position ids, varlen
    # flash-attn for prefill with a preallocated KV cache, and flash_attn_with_kvcache for decoding.
    # Attributes: n_local_q_heads, n_local_kv_heads, n_repeats, is_gqa, d_qk, d_v, d_model, qkv_proj, rotary_embedding,
    # flash_rotary_embedding, o_proj, attention, prefill_kv_len (max_position_embeddings).
    def __init__(
        self,
        config: LlaMoEConfig,
        parallel_config: Optional[ParallelismArgs],
        tp_pg: dist.ProcessGroup,
        layer_idx: int,
    ):
        # [FUNCTION DOC]
        # Behavior: asserts q heads and kv heads divisible by TP size (falling back to kv heads = q heads with a warning if
        # the attribute is missing) and q heads divisible by kv heads; computes local head counts and dims; resolves TP mode
        # and async flag; builds qkv_proj, both rotary embeddings, o_proj and CoreAttention.
        # Inputs: config (LlaMoEConfig, may be mutated); parallel_config (optional); tp_pg; layer_idx (int).
        # Outputs: None. Exceptions: AssertionError on divisibility violations.
        # Side effects: may set config.num_key_value_heads; logging; parameter allocation.
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
        )
        # TODO(kunhao): We want to have only one version per device and not one version per layer.
        self.rotary_embedding = RotaryEmbedding(
            dim=self.d_qk,
            end=config.max_position_embeddings,
        )

        # NOTE: Only supported for training (TODO(fmom): position_ids not supported yet)
        self.flash_rotary_embedding = FlashRotaryEmbedding(dim=self.d_qk, interleaved=True)

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
        # Behavior: projects to qkv (seq, batch, ...). GQA: splits into q, k, v and reshapes each to (batch, seq, heads, d);
        # otherwise views as (seq, batch, 3, heads, d) and permutes to three (batch, seq, heads, d) tensors.
        # Inference (store present): asserts no grad; position ids are previous offsets plus mask, or cumulative mask minus 1;
        # applies RotaryEmbedding to q and k. Prefill (no cached key): asserts padding only on the left, allocates zero
        # k/v caches of length prefill_kv_len, unpads q/k/v, runs causal varlen flash-attn, re-pads output, and copies k/v
        # into the caches right-padded. Decode: loads caches, grows them with zeros if the rotary end grew, asserts cache length
        # equals rotary end, and calls flash_attn_with_kvcache with cache_seqlens equal to the position offsets (appends new
        # k/v in place). Stores key, value, position_offsets.
        # Training: applies flash rotary to q and stacked k/v, flattens to (batch*seq, heads, d) and runs CoreAttention.
        # Finally reshapes to (seq, batch, local heads * d_v) and applies o_proj.
        # Inputs: hidden_states (seq, batch, hidden); sequence_mask (batch, seq bool).
        # Outputs: dictionary with hidden_states (seq, batch, hidden) and sequence_mask.
        # Exceptions: AssertionError for grad-enabled inference, middle-of-sequence masking or cache size mismatch.
        # Side effects: mutates the attached store; TP collectives in projections; CUDA kernels.
        # KNOWN ISSUE: decode-time cache_seqlens use the new position offset (already including the current token) rather than
        # the number of cached tokens, which may be off by one; prefill caches are sized prefill_kv_len while decode asserts
        # size equals rotary end, which fails if the rotary end grew during prefill.
        qkv_states = self.qkv_proj(
            hidden_states
        )  # [seq_length, batch_size, n_local_q_heads * d_qk + 2 * n_local_kv_heads * d_qk]
        q_length, batch_size, _ = qkv_states.shape

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
            # Double check that we use store only at inference time
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
            query_states = self.rotary_embedding(query_states, position_ids=position_ids)
            key_states = self.rotary_embedding(key_states, position_ids=position_ids)

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

                output_unpad = flash_attn_varlen_func(
                    q=query_unpad,  # (total_q, n_local_q_heads, d_qk)
                    k=key_unpad,  # (total_kv, n_local_kv_heads, d_qk)
                    v=value_unpad,  # (total_kv, n_local_kv_heads, d_v)
                    cu_seqlens_q=cu_seqlens_q,
                    cu_seqlens_k=cu_seqlens_k,
                    max_seqlen_q=max_seqlen_q,
                    max_seqlen_k=max_seqlen_k,
                    dropout_p=0.0,
                    softmax_scale=None,
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
                    softmax_scale=None,
                    causal=True,
                    rotary_interleaved=False,  # GPT-NeoX style
                )

            store.update(
                {
                    "key": k_cache,  # flash-attn has updated with new key_states using cache_seqlens
                    "value": v_cache,
                    "position_offsets": position_offsets,
                }
            )

        else:  # Training case
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


class LlaMoEDecoderLayer(nn.Module):
    # [CLASS DOC]
    # Pre-norm transformer block: TritonRMSNorm, CausalSelfAttention, residual add, TritonRMSNorm, dMoE sparse MLP,
    # residual add. Attributes: input_layernorm, attn, post_attention_layernorm, block_sparse_moe.
    def __init__(
        self,
        config: LlaMoEConfig,
        parallel_config: Optional[ParallelismArgs],
        tp_pg: dist.ProcessGroup,
        parallel_context: ParallelContext,
        layer_idx: int,
    ):
        # [FUNCTION DOC]
        # Behavior: builds the two norms (eps rms_norm_eps), attention and the dMoE (given parallel_context for expert groups).
        # Inputs: config; parallel_config (optional); tp_pg; parallel_context; layer_idx. Outputs: None.
        # Exceptions: None explicitly raised. Side effects: parameter allocation.
        super().__init__()
        self.input_layernorm = TritonRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.attn = CausalSelfAttention(
            config=config,
            parallel_config=parallel_config,
            tp_pg=tp_pg,
            layer_idx=layer_idx,
        )

        self.post_attention_layernorm = TritonRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.block_sparse_moe = dMoE(
            config,
            parallel_context=parallel_context,
            parallel_config=parallel_config,
        )

    def forward(
        self,
        hidden_states: Union[torch.Tensor, TensorPointer],
        sequence_mask: Union[torch.Tensor, TensorPointer],
    ) -> Dict[str, Union[torch.Tensor, TensorPointer]]:
        # [FUNCTION DOC]
        # Behavior: hidden = hidden + attention(norm(hidden)); hidden = hidden + moe(norm(hidden)); router logits discarded.
        # Inputs: hidden_states (seq, batch, hidden); sequence_mask (batch, seq).
        # Outputs: dictionary with hidden_states and sequence_mask.
        # Exceptions: None explicitly raised. Side effects: TP and expert-parallel collectives.
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        output = self.attn(hidden_states=hidden_states, sequence_mask=sequence_mask)
        hidden_states = output["hidden_states"]
        hidden_states = hidden_states + residual

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.block_sparse_moe(hidden_states)  # TODO @nouamane do we want to return router_logits?
        hidden_states = hidden_states + residual

        return {
            "hidden_states": hidden_states,
            "sequence_mask": output["sequence_mask"],
        }


class Embedding(nn.Module, AttachableStore):
    # [CLASS DOC]
    # Token embedding pipeline stage with a vocab-sharded TensorParallelEmbedding; tracks past_length in a generation
    # store and outputs sequence-first embeddings. Attributes: token_embedding, pg.
    def __init__(self, tp_pg: dist.ProcessGroup, config: LlaMoEConfig, parallel_config: Optional[ParallelismArgs]):
        # [FUNCTION DOC]
        # Behavior: builds the TP embedding (vocab_size by hidden_size, padding_idx pad_token_id, TP mode default ALL_REDUCE).
        # Inputs: tp_pg; config; parallel_config (optional). Outputs: None. Exceptions: None. Side effects: allocation.
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
        # Behavior: when a store is attached adds the per-row count of valid tokens to store past_length (zeros initially);
        # transposes input_ids to (seq, batch) and embeds.
        # Inputs: input_ids (batch, seq); input_mask (batch, seq). Outputs: dictionary with input_embeds (seq, batch, hidden).
        # Exceptions: None. Side effects: store mutation; TP collective.
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


class LlaMoEModel(nn.Module):
    # [CLASS DOC]
    # Pipeline graph of LlaMoE: token_position_embeddings, decoder (LlaMoEDecoderLayer per layer), final_layer_norm
    # (TritonRMSNorm), lm_head (column-parallel, vocab-sharded logits), cast_to_fp32; all PipelineBlocks sharing one P2P on
    # the PP group. Attributes: p2p, config, parallel_config, parallel_context, tp_mode.
    """Build pipeline graph"""

    def __init__(
        self,
        config: LlaMoEConfig,
        parallel_context: ParallelContext,
        parallel_config: Optional[ParallelismArgs],
    ):
        # [FUNCTION DOC]
        # Behavior: creates P2P on CUDA, resolves TP mode/async defaults and declares the pipeline blocks with their keys.
        # Inputs: config; parallel_context; parallel_config (optional). Outputs: None. Exceptions: None. Side effects: None beyond construction.
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

        self.decoder = nn.ModuleList(
            [
                PipelineBlock(
                    p2p=self.p2p,
                    module_builder=LlaMoEDecoderLayer,
                    module_kwargs={
                        "config": config,
                        "parallel_config": parallel_config,
                        "tp_pg": parallel_context.tp_pg,
                        "parallel_context": parallel_context,
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
        # Behavior: returns the first element (fp32 sharded logits) of forward_with_hidden_states.
        # Inputs: input_ids, input_mask ((batch, seq) or TensorPointers). Outputs: logits (seq, batch, local vocab) or TensorPointer.
        # Exceptions: None. Side effects: PP/TP communication.
        return self.forward_with_hidden_states(input_ids=input_ids, input_mask=input_mask)[0]

    def forward_with_hidden_states(
        self,
        input_ids: Union[torch.Tensor, TensorPointer],  # [batch_size, seq_length]
        input_mask: Union[torch.Tensor, TensorPointer],  # [batch_size, seq_length]
    ):
        # [FUNCTION DOC]
        # Behavior: embeds, runs all decoder blocks, final norm, lm_head and fp32 cast.
        # Inputs: input_ids, input_mask. Outputs: tuple (fp32 sharded logits, final hidden states), tensors or TensorPointers.
        # Exceptions: None. Side effects: PP P2P transfers, TP and expert collectives.
        # all tensors are optional as most ranks don't need anything from the dataloader.
        #
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
        # Behavior: relative costs for pipeline balancing: decoder layer 4 * heads * d_qkv * hidden + 3 * d_ff * hidden;
        # lm_head vocab * hidden. Outputs: dictionary class to cost. Exceptions: None. Side effects: None.
        # KNOWN ISSUE: treats the MoE MLP as a single dense MLP (ignores experts per token).
        """Computes the compute cost of each block in the model so that we can do a better job of load balancing."""
        model_config = self.config
        d_ff = model_config.intermediate_size
        d_qkv = model_config.hidden_size // model_config.num_attention_heads
        block_compute_costs = {
            # CausalSelfAttention (qkv proj + attn out) + MLP
            LlaMoEDecoderLayer: 4 * model_config.num_attention_heads * d_qkv * model_config.hidden_size
            + 3 * d_ff * model_config.hidden_size,
            # This is the last lm_head
            TensorParallelColumnLinear: model_config.vocab_size * model_config.hidden_size,
        }
        return block_compute_costs

    def get_flops_per_sec(self, iteration_time_in_sec, sequence_length, global_batch_size):
        # [FUNCTION DOC]
        # Behavior: calls get_flops with config sizes (kv heads default to q heads) and divides model and hardware flops by
        # iteration time, world size and 1e12.
        # Inputs: iteration_time_in_sec; sequence_length; global_batch_size. Outputs: tuple (model TFLOPs/s, hardware TFLOPs/s).
        # Exceptions: None. Side effects: None.
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
    # HELPER: TorchScript-compiled mean of loss over label_mask positions (sum of products in dtype divided by mask sum).
    # Inputs: loss, label_mask (batch, seq), dtype. Outputs: scalar tensor. Exceptions: None. Side effects: None.
    # type: (Tensor, Tensor, torch.dtype) -> Tensor
    return (loss * label_mask).sum(dtype=dtype) / label_mask.sum()


class Loss(nn.Module):
    # [CLASS DOC]
    # Pipeline stage computing masked mean cross entropy from vocab-sharded sequence-first logits. Attribute: tp_pg.
    def __init__(self, tp_pg: dist.ProcessGroup):
        # [FUNCTION DOC]
        # Behavior: stores tp_pg. Inputs: tp_pg. Outputs: None. Exceptions: None. Side effects: None.
        super().__init__()
        self.tp_pg = tp_pg

    def forward(
        self,
        sharded_logits: torch.Tensor,  # [seq_length, batch_size, logits]
        label_ids: torch.Tensor,  # [batch_size, seq_length]
        label_mask: torch.Tensor,  # [batch_size, seq_length]
    ) -> Dict[str, torch.Tensor]:
        # [FUNCTION DOC]
        # Behavior: transposes labels to (seq, batch), computes fp32 sharded_cross_entropy across the TP group, transposes back
        # and takes the masked mean.
        # Inputs: sharded_logits (seq, batch, local vocab); label_ids (batch, seq); label_mask (batch, seq).
        # Outputs: dictionary with loss scalar. Exceptions: None. Side effects: TP all-reduces.
        # Megatron by defaults cast everything in fp32. `--f16-lm-cross-entropy` is an option you can use to keep current precision.
        # https://github.com/NVIDIA/Megatron-LM/blob/f267e6186eae1d6e2055b412b00e2e545a8e896a/megatron/model/gpt_model.py#L38
        loss = sharded_cross_entropy(
            sharded_logits, label_ids.transpose(0, 1).contiguous(), group=self.tp_pg, dtype=torch.float
        ).transpose(0, 1)
        # TODO @thomasw21: It's unclear what kind of normalization we want to do.
        loss = masked_mean(loss, label_mask, dtype=torch.float)
        # I think indexing causes a sync we don't actually want
        # loss = loss[label_mask].sum()
        return {"loss": loss}


class LlaMoEForTraining(NanotronModel):
    # [CLASS DOC]
    # NanotronModel wrapper for training LlaMoE: LlaMoEModel plus a Loss PipelineBlock; provides random init, block
    # costs and flops hooks. Attributes: model, loss, parallel_context, config, parallel_config.
    def __init__(
        self,
        config: LlaMoEConfig,
        parallel_context: ParallelContext,
        parallel_config: Optional[ParallelismArgs],
        random_states: Optional[RandomStates] = None,
    ):
        # [FUNCTION DOC]
        # Behavior: builds LlaMoEModel and the Loss block, stores contexts.
        # Inputs: config; parallel_context; parallel_config (optional); random_states (optional, unused). Outputs: None.
        # Exceptions: None. Side effects: construction only.
        super().__init__()
        self.model = LlaMoEModel(config=config, parallel_context=parallel_context, parallel_config=parallel_config)
        self.loss = PipelineBlock(
            p2p=self.model.p2p,
            module_builder=Loss,
            module_kwargs={"tp_pg": parallel_context.tp_pg},
            module_input_keys={
                "sharded_logits",
                "label_ids",
                "label_mask",
            },
            module_output_keys={"loss"},
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
        # Behavior: model logits then loss. Inputs: input_ids, input_mask, label_ids, label_mask.
        # Outputs: dictionary with loss (tensor on last PP rank, TensorPointer elsewhere). Exceptions: None.
        # Side effects: PP/TP/expert communication.
        sharded_logits = self.model(
            input_ids=input_ids,
            input_mask=input_mask,
        )
        loss = self.loss(
            sharded_logits=sharded_logits,
            label_ids=label_ids,
            label_mask=label_mask,
        )["loss"]
        return {"loss": loss}

    @torch.no_grad()
    def init_model_randomly(self, config):
        # [FUNCTION DOC]
        # Behavior (no grad): std and sigma both come from config.model.init_method.std. For each NanotronParameter, resolves the
        # full (tied-aware) name, skips duplicates, then by owning module: TP column linear weight normal(0, std), bias zero;
        # TP row linear weight normal(0, sigma / sqrt(2 * layers)), bias zero; TritonRMSNorm weight 1, bias 0; plain nn.Linear
        # (e.g. the router) weight Kaiming uniform, bias uniform; TP embedding normal(0, std). Asserts all parameters initialised.
        # Inputs: config (full nanotron Config with a RandomInit). Outputs: None.
        # Exceptions: ValueError for unknown parameter names; Exception for parameters owned by unhandled module types (e.g.
        # dMoE expert weights registered directly on a non-listed module); AssertionError for non-Nanotron params or mismatch.
        # Side effects: in-place parameter writes; RNG consumption.
        # KNOWN ISSUE: for nn.Linear biases fan_in is always None (set only in the weight iteration), so the bias bound is 0;
        # the docstring mentions apply_layernorm_1p which the code does not use.
        """Initialize model parameters randomly.
        Note:
            Layernorm weight all 0 or 1 depending on `apply_layernorm_1p`
        """
        model = self
        initialized_parameters = set()
        # Handle tensor parallelism
        module_id_to_prefix = {id(module): f"{module_name}." for module_name, module in model.named_modules()}
        # Fix the root_model
        module_id_to_prefix[id(model)] = ""

        std = config.model.init_method.std
        sigma = config.model.init_method.std
        num_layers = config.model.model_config.num_hidden_layers

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

            if isinstance(module, TensorParallelColumnLinear):
                if "weight" == param_name:
                    nn.init.normal_(module.weight, mean=0.0, std=std)
                elif "bias" == param_name:
                    module.bias.zero_()
                else:
                    raise ValueError(f"Who the fuck is {param_name}?")
            elif isinstance(module, TensorParallelRowLinear):
                if "weight" == param_name:
                    nn.init.normal_(module.weight, mean=0.0, std=sigma / math.sqrt(2 * num_layers))
                elif "bias" == param_name:
                    param.zero_()
                else:
                    raise ValueError(f"Who the fuck is {param_name}?")
            elif isinstance(module, TritonRMSNorm):
                if "weight" == param_name:
                    # TODO @thomasw21: Sometimes we actually want 0
                    module.weight.fill_(1)
                elif "bias" == param_name:
                    module.bias.zero_()
                else:
                    raise ValueError(f"Who the fuck is {param_name}?")
            elif isinstance(module, nn.Linear):
                fan_in = None

                if "weight" == param_name:
                    fan_in, _ = init._calculate_fan_in_and_fan_out(module.weight)
                    init.kaiming_uniform_(module.weight, a=math.sqrt(5))
                elif "bias" == param_name:
                    bound = 1 / math.sqrt(fan_in) if (fan_in is not None and fan_in > 0) else 0
                    init.uniform_(module.bias, -bound, bound)
                else:
                    raise ValueError(f"Who the fuck is {param_name}?")

            elif isinstance(module, TensorParallelEmbedding):
                nn.init.normal_(module.weight, mean=0.0, std=std)
            else:
                raise Exception(f"Parameter {full_param_name} was not initialized")

            assert full_param_name not in initialized_parameters
            initialized_parameters.add(full_param_name)

        assert initialized_parameters == {
            param.get_tied_info().get_full_name_from_module_id_to_prefix(module_id_to_prefix=module_id_to_prefix)
            if param.is_tied
            else name
            for name, param in model.named_parameters()
        }, f"Somehow the initialized set of parameters don't match:\n - Expected: { {name for name, _ in model.named_parameters()} }\n - Got: {initialized_parameters}"
    
    def get_block_compute_costs(self):
        # [FUNCTION DOC]
        # Behavior: delegates to LlaMoEModel.get_block_compute_costs. Outputs: dictionary. Exceptions: None. Side effects: None.
        """Computes the compute cost of each block in the model so that we can do a better job of load balancing."""
        return self.model.get_block_compute_costs()

    def get_flops_per_sec(self, iteration_time_in_sec, sequence_length, global_batch_size):
        # [FUNCTION DOC]
        # Behavior: delegates to LlaMoEModel.get_flops_per_sec. Outputs: tuple of TFLOPs/s. Exceptions: None. Side effects: None.
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
    # HELPER: analytic flop count for a decoder-only transformer.
    # Behavior: per layer forward flops (2 per MAC) for q and kv projections, QK logits, attention-weighted values, output
    # projection, and a gated FFN (first layer counted twice for gate and up, second once), plus lm_head; model flops are
    # three times forward (forward plus double for backward); hardware flops equal model flops.
    # Inputs: num_layers, hidden_size, num_heads, num_key_value_heads (None means num_heads), vocab_size, seq_len,
    # ffn_hidden_size, batch_size (default 1). Outputs: tuple (model_flops, hardware_flops).
    # Exceptions: None. Side effects: None.
    # KNOWN ISSUE: ignores MoE top-k (counts one dense FFN per token) and the router.
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

    hardware_flops = model_flops  # TODO @nouamanetazi: add hardware flops
    return model_flops, hardware_flops
