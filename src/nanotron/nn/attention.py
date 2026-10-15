# [MODULE DOC]
# Attention backend dispatch for nanotron models.
# Main contents: three attention forward implementations (FlexAttention, FlashAttention-2, PyTorch SDPA), version
# probes, a lazy loader for the Triton ring attention, the ALL_ATTENTION_FUNCTIONS registry mapping implementation
# names to callables (also including the "ring" and "llama3_ring_attention" context-parallel variants), the
# AttentionImplementation literal type built from its keys, and get_attention_mask for packed documents.
# Fit in nanotron: model attention modules look up their configured implementation name in ALL_ATTENTION_FUNCTIONS.
# Import-time side effects: imports ring_attention and llama3_ring_attention; conditionally imports create_block_mask
# and flex_attention (torch 2.5.0 or newer) and flash_attn_func (flash_attn 2.1.0 or newer). When these conditions
# fail, the corresponding names are undefined and calling that backend raises NameError.
# Exports: ALL_ATTENTION_FUNCTIONS, AttentionImplementation, the forward functions, get_attention_mask.
from functools import lru_cache
from typing import Literal, Optional, Tuple

import torch
from packaging import version

from nanotron.nn.ring_attention import ring_flash_attn_varlen_func
from nanotron.nn.llama3_ring_attention import llama3_flash_attn_varlen_qkvpacked_func

# Replace direct import with a function for lazy loading
def get_ring_flash_attn_cuda():
    # [FUNCTION DOC]
    # HELPER: Lazy loader for the Triton-based ring attention, used by the "ring_flash_triton" registry entry.
    # Imports ring_flash_attn_cuda from ring_attention_lucidrain on first call so Triton is only needed when used.
    # Inputs: none.
    # Outputs: the ring_flash_attn_cuda callable.
    # Exceptions: ImportError if the module or its dependencies cannot be imported.
    # Side effects: imports a module (and its dependencies) on first call.
    """Lazily import ring_flash_attn_cuda to avoid early Triton dependency."""
    from nanotron.nn.ring_attention_lucidrain import ring_flash_attn_cuda

    return ring_flash_attn_cuda


@lru_cache()
def is_torch_flex_attn_available():
    # TODO check if some bugs cause push backs on the exact version
    # NOTE: We require torch>=2.5.0 as it is the first release
    # [FUNCTION DOC]
    # Reports whether the installed torch version is 2.5.0 or newer (first release with FlexAttention).
    # Result is cached with lru_cache.
    # Inputs: none. Outputs: bool.
    # Exceptions: None explicitly raised. Side effects: None.
    return version.parse(torch.__version__) >= version.parse("2.5.0")


if is_torch_flex_attn_available():
    from torch.nn.attention.flex_attention import create_block_mask, flex_attention


@lru_cache()
def is_flash_attn_greater_or_equal_2_10():
    # [FUNCTION DOC]
    # Reports whether flash_attn can be imported and its version is at least 2.1.0. Cached with lru_cache.
    # Inputs: none.
    # Outputs: bool; False when flash_attn is not installed.
    # Exceptions: None explicitly raised (ImportError is caught).
    # Side effects: imports flash_attn.
    # KNOWN ISSUE: the name says 2.10 but the comparison is against version 2.1.0.
    try:
        import flash_attn

        return version.parse(flash_attn.__version__) >= version.parse("2.1.0")
    except ImportError:
        return False


if is_flash_attn_greater_or_equal_2_10():
    from flash_attn.flash_attn_interface import flash_attn_func
# adapted from transformers.integrations.flex_attention.flex_attention_forward
def flex_attention_forward(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    sliding_window: Optional[int] = None,
    scaling: Optional[float] = None,
    softcap: Optional[float] = None,
    position_ids: Optional[torch.Tensor] = None,
    document_ids: Optional[torch.Tensor] = None,
    flex_attention_mask: Optional[str] = None,
    **kwargs,
) -> Tuple[torch.Tensor, torch.Tensor]:
    # [FUNCTION DOC]
    # Attention via PyTorch FlexAttention with block-sparse masks.
    # Behavior: imports mask helpers from nanotron.nn.flex_attention; validates the argument combination; score_mod is
    # always None. If flex_attention_mask names a mask type, builds the block mask with get_block_mask_from_type.
    # Otherwise slices attention_mask to the key length, builds a document mask function from document_ids or
    # position_ids, and combines it with causal and optional sliding-window masking via create_attention_mask.
    # Calls flex_attention with grouped-query attention enabled, the block mask, the given scale and return_lse
    # enabled; casts the returned log-sum-exp to value dtype and transposes output to sequence-major layout.
    # Inputs: module (calling module, unused); query [batch, num_heads, seq, head_dim]; key and value
    # [batch, num_kv_heads, seq, head_dim]; attention_mask (optional tensor); sliding_window (optional int);
    # scaling (optional float, softmax scale); softcap (optional float, unused); position_ids (optional tensor);
    # document_ids (optional per-token document index tensor); flex_attention_mask (optional mask type name);
    # extra keyword arguments ignored.
    # Outputs: tuple of attention output [batch, seq, num_heads, head_dim] (contiguous) and the log-sum-exp tensor
    # [batch, num_heads, seq] in value dtype.
    # Exceptions: ValueError from validate_attention_args or get_block_mask_from_type for invalid combinations;
    # NameError if torch is older than 2.5.0.
    # Side effects: None.
    # KNOWN ISSUE: the second return value is the log-sum-exp, not attention weights as the name and docstring suggest.
    # KNOWN ISSUE: the sliced causal_mask is computed but never used; softcap and module are ignored.
    """
    Implementation of attention using PyTorch's FlexAttention.

    Args:
        module: The module calling this function
        query: Query states tensor [batch_size, num_heads, seq_len, head_dim]
        key: Key states tensor [batch_size, num_kv_heads, seq_len, head_dim]
        value: Value states tensor [batch_size, num_kv_heads, seq_len, head_dim]
        attention_mask: Optional attention mask tensor
        sliding_window: Optional sliding window size for efficient attention
        scaling: Optional scaling factor for attention scores
        softcap: Optional softcap for softmax stability
        position_ids: Optional tensor of position IDs used for document masking
        document_ids: Optional tensor explicitly marking document boundaries [seq_len]
                     (e.g., [0,0,0,1,1,2,2,2,2,2,2] for seqs of length 3,2,6)
        flex_attention_mask: Optional string specifying a custom mask type
        
    Returns:
        Tuple of (attention_output, attention_weights)
    """
    from nanotron.nn.flex_attention import (
        create_softcapped_causal_score_mod,
        create_document_mask_func,
        create_attention_mask,
        get_attention_mod_from_type,
        get_block_mask_from_type,
        validate_attention_args,
    )

    # Validate arguments if a flex_attention_mask is specified
    validate_attention_args(
        flex_attention_mask=flex_attention_mask,
        sliding_window=sliding_window,
        position_ids=position_ids,
        document_ids=document_ids,
    )

    # We're setting score_mod to None as requested
    score_mod = None

    # Determine which block mask to use
    if flex_attention_mask is not None:
        # Use the mask type specified by flex_attention_mask
        block_mask = get_block_mask_from_type(
            flex_attention_mask=flex_attention_mask,
            query=query,
            key=key,
            sliding_window=sliding_window,
            position_ids=position_ids,
            document_ids=document_ids,
        )
    else:
        # Use the existing document/sliding window masking logic
        causal_mask = attention_mask
        if causal_mask is not None:
            causal_mask = causal_mask[:, :, :, : key.shape[-2]]
            
        # Create document masking function if needed
        doc_mask_func = create_document_mask_func(query, document_ids, position_ids)

        # Create combined attention mask
        block_mask = create_attention_mask(query, sliding_window, doc_mask_func)

    # Call PyTorch's flex_attention with the appropriate parameters
    attn_output, attention_weights = flex_attention(
        query,
        key,
        value,
        enable_gqa=True,  # Enable grouped query attention
        score_mod=score_mod,
        block_mask=block_mask,  # Efficient mask based on type
        scale=scaling,
        return_lse=True,  # FlexAttention always computes log-sum-exp anyway
    )

    # FlexAttention returns weights in float32, convert to match value dtype
    attention_weights = attention_weights.to(value.dtype)

    # Transpose output to match expected format [batch_size, seq_len, num_heads, head_dim]
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attention_weights


def flash_attention_forward(
    module: torch.nn.Module,
    query: torch.Tensor,  # [b, num_heads, seq_len, head_dim]
    key: torch.Tensor,  # [b, num_kv_heads, seq_len, head_dim]
    value: torch.Tensor,  # [b, num_kv_heads, seq_len, head_dim]
    attention_mask: Optional[torch.Tensor],  # [b, num_heads, seq_len, seq_len]
    max_seqlen: Optional[int],
    dropout: float = 0.0,
    scaling: Optional[float] = None,
    sliding_window: Optional[int] = None,
    **kwargs,
) -> Tuple[torch.Tensor, None]:
    # [FUNCTION DOC]
    # Attention via flash_attn_func (FlashAttention-2, non-varlen).
    # Behavior: reshapes query, key and value to [batch, max_seqlen, heads, head_dim] using module.local_num_heads,
    # module.local_num_kv_heads and module.head_dim; uses causal attention when attention_mask is None (otherwise
    # non-causal, the mask itself is not passed); window size is (sliding_window, sliding_window) when set, else
    # unlimited (-1, -1); calls flash_attn_func without returning probabilities.
    # Inputs: module (provides head counts and head_dim); query, key, value tensors whose elements can be viewed as
    # described above; attention_mask (optional, only its presence matters); max_seqlen (int, sequence length);
    # dropout (float, default 0.0); scaling (optional softmax scale); sliding_window (optional int); extra keyword
    # arguments ignored.
    # Outputs: tuple of contiguous output [batch, max_seqlen, num_heads, head_dim] and None.
    # Exceptions: NameError if flash_attn 2.1.0 or newer was not importable; errors from view on incompatible shapes.
    # Side effects: None.
    # KNOWN ISSUE: the parameter comments claim [b, num_heads, seq_len, head_dim] inputs, but the view to
    # [-1, max_seqlen, heads, head_dim] only reinterprets memory correctly for token-major (sequence-first) layouts.
    # KNOWN ISSUE: a non-None attention_mask silently turns off causal masking without applying the mask.
    query = query.view(-1, max_seqlen, module.local_num_heads, module.head_dim)
    key = key.view(-1, max_seqlen, module.local_num_kv_heads, module.head_dim)
    value = value.view(-1, max_seqlen, module.local_num_kv_heads, module.head_dim)

    if attention_mask is None:
        is_causal = True
    else:
        is_causal = False

    if sliding_window is not None:
        window_size = (sliding_window, sliding_window)
    else:
        window_size = (-1, -1)

    attn_output = flash_attn_func(
        q=query,
        k=key,
        v=value,
        dropout_p=dropout,
        softmax_scale=scaling,
        causal=is_causal,
        window_size=window_size,
        return_attn_probs=False,
    )
    attn_output = attn_output.contiguous()
    return attn_output, None


def sdpa_attention_forward(
    module: torch.nn.Module,
    query: torch.Tensor,  # [b, num_heads, seq_len, head_dim]
    key: torch.Tensor,  # [b, num_kv_heads, seq_len, head_dim]
    value: torch.Tensor,  # [b, num_kv_heads, seq_len, head_dim]
    attention_mask: Optional[torch.Tensor],  # [b, num_heads, seq_len, seq_len]
    max_seqlen: int,
    dropout: float = 0.0,
    scaling: Optional[float] = None,
    **kwargs,
) -> Tuple[torch.Tensor, None]:
    # [FUNCTION DOC]
    # Attention via torch scaled_dot_product_attention.
    # Behavior: causal when attention_mask is None; reshapes inputs to [batch, max_seqlen, heads, head_dim] and transposes
    # to head-major; calls SDPA with the mask, dropout, scale, causal flag and grouped-query mode enabled when query and
    # key head counts differ; transposes the result back to [batch, seq, heads, head_dim].
    # Inputs: module (provides local_num_heads, local_num_kv_heads, head_dim); query, key, value tensors; attention_mask
    # (optional boolean or additive mask); max_seqlen (int); dropout (float, default 0.0); scaling (optional float);
    # extra keyword arguments ignored.
    # Outputs: tuple of contiguous output [batch, seq, num_heads, head_dim] and None.
    # Exceptions: errors from torch when both a mask and causal are set or shapes mismatch.
    # Side effects: None.
    if attention_mask is None:
        is_causal = True
    else:
        is_causal = False
    query = query.view(-1, max_seqlen, module.local_num_heads, module.head_dim).transpose(
        1, 2
    )  # [b, num_heads, seq_length, head_dim]
    key = key.view(-1, max_seqlen, module.local_num_kv_heads, module.head_dim).transpose(1, 2)
    value = value.view(-1, max_seqlen, module.local_num_kv_heads, module.head_dim).transpose(1, 2)
    attn_output = torch.nn.functional.scaled_dot_product_attention(
        query,
        key,
        value,
        attn_mask=attention_mask,
        dropout_p=dropout,
        scale=scaling,
        is_causal=is_causal,
        enable_gqa=query.shape[1] != key.shape[1],
    )
    attn_output = attn_output.transpose(1, 2).contiguous()  # [batch_size, seq_len, num_heads, head_dim]
    return attn_output, None


ALL_ATTENTION_FUNCTIONS = {
    "flash_attention_2": flash_attention_forward,
    "flex_attention": flex_attention_forward,
    "sdpa": sdpa_attention_forward,
    "ring_flash_triton": lambda *args, **kwargs: get_ring_flash_attn_cuda()(*args, **kwargs),
    "ring": ring_flash_attn_varlen_func,
    "llama3_ring_attention": llama3_flash_attn_varlen_qkvpacked_func,
}

AttentionImplementation = Literal[tuple(ALL_ATTENTION_FUNCTIONS.keys())]


# TODO @nouamane: optimize this, and make sure it works with flashattn and flexattn
def get_attention_mask(position_ids, seq_length):
    # [FUNCTION DOC]
    # Builds a block-diagonal causal mask for packed sequences, where a document starts at every position with id 0.
    # Behavior: finds start indices where position_ids equals zero, appends seq_length to form cumulative sequence
    # lengths (int32), then fills a lower-triangular block for each document in a zero matrix.
    # Inputs: position_ids, 1-D tensor of positions; seq_length, int total length.
    # Outputs: tuple of boolean mask [seq_length, seq_length] and cu_seqlens int32 tensor.
    # Exceptions: None explicitly raised.
    # Side effects: None (allocates on position_ids device; the per-block ones are created on CPU and copied).
    # KNOWN ISSUE: start_indices from torch.where is int64 while the appended tensor is int32; concatenation relies on
    # type promotion. A python loop over documents makes this slow (as noted by the TODO).
    attention_mask = torch.zeros(seq_length, seq_length, device=position_ids.device)
    start_indices = torch.where(position_ids == 0)[0]
    cu_seqlens = torch.cat(
        [start_indices, torch.tensor([seq_length], dtype=torch.int32, device=start_indices.device)]
    ).to(torch.int32)
    # make trius for each document
    for i in range(len(cu_seqlens) - 1):
        attention_mask[cu_seqlens[i] : cu_seqlens[i + 1], cu_seqlens[i] : cu_seqlens[i + 1]] = torch.tril(
            torch.ones(cu_seqlens[i + 1] - cu_seqlens[i], cu_seqlens[i + 1] - cu_seqlens[i])
        )
    return attention_mask.to(torch.bool), cu_seqlens  # [seq_length, seq_length]
