# [MODULE DOC]
# Mask and score-modifier builders for PyTorch FlexAttention, used by flex_attention_forward in nanotron.nn.attention.
# Main contents: score-mod factory with softcap, document-mask factories (from document ids or from position ids
# where a zero marks a new document), combined causal / sliding-window / document block-mask builders, a cached
# create_block_mask wrapper, offset/length conversion helpers, a mask-type dispatcher (get_block_mask_from_type for
# "causal", "sliding_window", "document"), a placeholder attention-mod getter and argument validation.
# Mask functions follow the FlexAttention mask_mod signature: batch index, head index, query index, key index,
# returning a boolean (they are traced/vmapped by FlexAttention, so indices are tensors at runtime).
# Import-time side effects: imports torch.nn.attention.flex_attention, so torch 2.5.0 or newer is required.
# Exports: all functions defined here.
"""
Modular masking utilities for FlexAttention implementation.
"""
from typing import Callable, Optional, Tuple

import torch
from torch.nn.attention.flex_attention import create_block_mask
import functools


def create_softcapped_causal_score_mod(
    softcap: Optional[float],
    causal_mask: Optional[torch.Tensor],
) -> Optional[Callable]:
    # [FUNCTION DOC]
    # Builds a FlexAttention score_mod that optionally softcaps scores (softcap times tanh of score divided by softcap) and
    # then adds an additive mask value taken from causal_mask at batch b, head slot 0, row q_idx, column kv_idx.
    # Inputs: softcap (optional float); causal_mask (optional additive mask [batch, 1, q_len, kv_len]).
    # Outputs: the score_mod callable, or None when both inputs are None.
    # Exceptions: None explicitly raised. Side effects: None.
    """Creates a score modifier function that applies softcapping and causal masking.

    Args:
        softcap: Optional softcap value for attention scores
        causal_mask: Optional causal mask tensor [batch, 1, seq_len, seq_len]

    Returns:
        Callable score modifier function or None if no modifications needed
    """
    if softcap is None and causal_mask is None:
        return None

    def score_mod(score: torch.Tensor, b: int, h: int, q_idx: int, kv_idx: int) -> torch.Tensor:
        # [FUNCTION DOC]
        # Score modifier closure: applies softcap if set, then adds the mask entry if a mask was given.
        # Inputs: score (tensor), b, h, q_idx, kv_idx (indices). Outputs: the modified score tensor.
        # Exceptions: None explicitly raised. Side effects: None.
        if softcap is not None:
            score = softcap * torch.tanh(score / softcap)
        if causal_mask is not None:
            score = score + causal_mask[b][0][q_idx][kv_idx]
        return score

    return score_mod


def create_document_mask_func(
    query: torch.Tensor,
    document_ids: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.Tensor] = None,
) -> Optional[Callable]:
    # [FUNCTION DOC]
    # Builds a mask_mod restricting attention to tokens of the same document.
    # Branches: if document_ids is given (moved to query device), compares the ids of flattened positions
    # batch times seq_len plus index, treating out-of-range positions as document -1. Else if position_ids is given
    # (flattened when 2-D), computes document start positions where the position is zero and compares the document
    # index found by searchsorted for both flattened positions. Otherwise returns None.
    # Inputs: query [batch, heads, seq_len, head_dim] (shape and device only); document_ids (optional, flattened ids);
    # position_ids (optional, [batch, seq_len] or [seq_len]).
    # Outputs: a mask function or None.
    # Exceptions: None explicitly raised. Side effects: None.
    # KNOWN ISSUE: the out-of-range check in the document_ids branch is a Python conditional on traced index tensors,
    # which is data-dependent control flow that FlexAttention's vmap tracing generally cannot handle.
    """Creates a document masking function based on either document_ids or position_ids.

    Args:
        query: Query tensor for shape information [batch, num_heads, seq_len, head_dim]
        document_ids: Optional tensor marking document boundaries [seq_len]
        position_ids: Optional tensor of position IDs [batch, seq_len] or [seq_len]

    Returns:
        Document masking function or None if no document masking needed
    """
    if document_ids is not None:
        document_ids = document_ids.to(query.device)

        def document_mask_direct(b: int, h: int, q_idx: int, kv_idx: int) -> bool:
            # [FUNCTION DOC]
            # Mask closure for explicit document ids: true when both flattened positions map to the same document id
            # (positions beyond the id tensor count as -1).
            # Inputs: b, h, q_idx, kv_idx. Outputs: boolean (tensor).
            # Exceptions: None explicitly raised. Side effects: None.
            seq_len = query.size(2)
            q_flat_idx = b * seq_len + q_idx
            kv_flat_idx = b * seq_len + kv_idx

            q_doc = document_ids[q_flat_idx] if q_flat_idx < len(document_ids) else -1
            kv_doc = document_ids[kv_flat_idx] if kv_flat_idx < len(document_ids) else -1

            return q_doc == kv_doc

        return document_mask_direct

    elif position_ids is not None:
        seq_len = query.size(2)

        if position_ids.dim() == 2:
            position_ids = position_ids.view(-1)

        doc_starts = torch.where(position_ids == 0)[0]

        def document_mask_from_positions(b: int, h: int, q_idx: int, kv_idx: int) -> bool:
            # [FUNCTION DOC]
            # Mask closure for position-derived documents: true when searchsorted over the document starts (minus one) yields the
            # same document index for both flattened positions.
            # Inputs: b, h, q_idx, kv_idx. Outputs: boolean (tensor).
            # Exceptions: None explicitly raised. Side effects: None.
            # KNOWN ISSUE: searchsorted with default left side places a position equal to a document start into the previous
            # document, so the first token of each document is grouped with the preceding document.
            q_flat_idx = b * seq_len + q_idx
            kv_flat_idx = b * seq_len + kv_idx

            q_doc_idx = torch.searchsorted(doc_starts, q_flat_idx) - 1
            kv_doc_idx = torch.searchsorted(doc_starts, kv_flat_idx) - 1

            return q_doc_idx == kv_doc_idx

        return document_mask_from_positions

    return None


def create_attention_mask(
    query: torch.Tensor,
    sliding_window: Optional[int] = None,
    doc_mask_func: Optional[Callable] = None,
) -> Optional[torch.Tensor]:
    # [FUNCTION DOC]
    # Builds a BlockMask that is causal and additionally windowed (key at most sliding_window behind query) and/or
    # document-restricted. Batch size is passed to create_block_mask only when a document mask is used (it may depend
    # on batch); heads are broadcast.
    # Inputs: query (shape source); sliding_window (optional int); doc_mask_func (optional mask function).
    # Outputs: a BlockMask, or None when neither windowing nor document masking is requested.
    # Exceptions: None explicitly raised. Side effects: allocates the block mask on the default device of create_block_mask.
    # KNOWN ISSUE: returning None in the plain case makes flex_attention run without any mask, i.e. non-causal, even
    # though every other branch here is causal.
    """Creates an attention mask combining sliding window and document masking.

    Args:
        query: Query tensor for shape information [batch, num_heads, seq_len, head_dim]
        sliding_window: Optional sliding window size
        doc_mask_func: Optional document masking function

    Returns:
        Block mask tensor or None if no masking needed
    """
    if sliding_window is None and doc_mask_func is None:
        return None

    seq_len = query.size(2)

    if sliding_window is not None:
        def sliding_window_causal(b: int, h: int, q_idx: int, kv_idx: int) -> bool:
            # [FUNCTION DOC]
            # Mask closure: causal and within the sliding window, intersected with the document mask when provided.
            # Inputs: b, h, q_idx, kv_idx. Outputs: boolean (tensor).
            # Exceptions: None explicitly raised. Side effects: None.
            causal_mask = q_idx >= kv_idx
            window_mask = q_idx - kv_idx <= sliding_window
            if doc_mask_func is not None:
                doc_mask = doc_mask_func(b, h, q_idx, kv_idx)
                return causal_mask & window_mask & doc_mask
            return causal_mask & window_mask

        mask_func = sliding_window_causal
    else:
        def doc_causal_mask(b: int, h: int, q_idx: int, kv_idx: int) -> bool:
            # [FUNCTION DOC]
            # Mask closure: causal intersected with the document mask.
            # Inputs: b, h, q_idx, kv_idx. Outputs: boolean (tensor).
            # Exceptions: None explicitly raised. Side effects: None.
            causal_mask = q_idx >= kv_idx
            doc_mask = doc_mask_func(b, h, q_idx, kv_idx)
            return causal_mask & doc_mask

        mask_func = doc_causal_mask

    return create_block_mask(
        mask_func,
        B=query.size(0) if doc_mask_func is not None else None,
        H=None,
        Q_LEN=seq_len,
        KV_LEN=seq_len,
    )


@functools.lru_cache(maxsize=32)
def create_block_mask_cached(mask_func, B, H, Q_LEN, KV_LEN, device="cuda"):
    # [FUNCTION DOC]
    # HELPER: memoized wrapper (lru_cache, 32 entries) around create_block_mask for get_block_mask_from_type.
    # Inputs: mask_func, B, H, Q_LEN, KV_LEN as for create_block_mask; device (default "cuda").
    # Outputs: a BlockMask.
    # Exceptions: TypeError if any argument is unhashable.
    # Side effects: caches BlockMask objects (holding device memory) for the process lifetime.
    # KNOWN ISSUE: callers pass freshly created partials/closures each call, which hash by identity, so the cache
    # effectively never hits while still retaining up to 32 masks.
    """Cached version of create_block_mask for better performance."""
    block_mask = create_block_mask(mask_func, B, H, Q_LEN, KV_LEN, device=device)
    return block_mask


def _offsets_to_doc_ids_tensor(offsets):
    # [FUNCTION DOC]
    # HELPER: expands cumulative offsets into a per-token document index tensor for generate_doc_mask_mod.
    # Takes consecutive differences as counts and repeats each document index (int32) by its count.
    # Inputs: offsets, tensor [num_docs + 1].
    # Outputs: int32 tensor of length offsets[-1] minus offsets[0].
    # Exceptions: None explicitly raised. Side effects: None.
    """Converts offsets to document IDs tensor.
    
    Args:
        offsets: Tensor of shape [num_docs + 1] containing cumulative token counts
        
    Returns:
        Tensor of document IDs for each token position
    """
    device = offsets.device
    counts = offsets[1:] - offsets[:-1]
    return torch.repeat_interleave(
        torch.arange(len(counts), device=device, dtype=torch.int32), counts
    )


def lengths_to_offsets(lengths, device):
    # [FUNCTION DOC]
    # HELPER: converts document lengths into cumulative offsets starting at zero.
    # For a Python list, builds an int32 tensor of zero followed by the lengths; for a tensor, prepends an int32 zero.
    # Then takes the cumulative sum.
    # Inputs: lengths (list or tensor of ints); device.
    # Outputs: tensor [num_docs + 1] of offsets (dtype from cumsum promotion).
    # Exceptions: None explicitly raised. Side effects: None.
    """Converts a list of document lengths to offsets.
    
    Args:
        lengths: List or tensor of document lengths
        device: Device for the resulting tensor
        
    Returns:
        Tensor of cumulative offsets
    """
    if not isinstance(lengths, torch.Tensor):
        offsets = [0]
        offsets.extend(lengths)
        offsets = torch.tensor(offsets, device=device, dtype=torch.int32)
    else:
        offsets = torch.cat([torch.zeros(1, device=device, dtype=torch.int32), lengths])
    
    offsets = torch.cumsum(offsets, dim=-1)
    return offsets


def generate_doc_mask_mod(inner_mask_func, offsets):
    # [FUNCTION DOC]
    # Wraps an inner mask function so it only applies within documents, using document-local indices.
    # Precomputes per-token document ids from offsets.
    # Inputs: inner_mask_func (mask function); offsets (tensor [num_docs + 1]).
    # Outputs: a mask function.
    # Exceptions: None explicitly raised. Side effects: None.
    """Generates a document-aware mask function.
    
    Args:
        inner_mask_func: Base mask function to apply within each document
        offsets: Tensor of shape [num_docs + 1] containing cumulative token counts
        
    Returns:
        Document-aware mask function
    """
    document_ids = _offsets_to_doc_ids_tensor(offsets)
    
    def doc_mask_mod(b, h, q_idx, kv_idx):
        # Only attend within the same document
        # [FUNCTION DOC]
        # Mask closure: true when query and key are in the same document and the inner mask, evaluated on indices relative
        # to each document's start offset, is true.
        # Inputs: b, h, q_idx, kv_idx. Outputs: boolean (tensor).
        # Exceptions: None explicitly raised (indexing past the document id tensor errors).
        # Side effects: None.
        same_doc = document_ids[q_idx] == document_ids[kv_idx]
        
        # Convert to document-local indices
        q_local_idx = q_idx - offsets[document_ids[q_idx]]
        kv_local_idx = kv_idx - offsets[document_ids[kv_idx]]
        
        # Apply the inner mask function using document-local indices
        inner_mask = inner_mask_func(b, h, q_local_idx, kv_local_idx)
        
        return same_doc & inner_mask
    
    return doc_mask_mod


def causal_mask_func(b, h, q_idx, kv_idx):
    # [FUNCTION DOC]
    # Causal mask function: query index at or after key index.
    # Inputs: b, h, q_idx, kv_idx. Outputs: boolean (tensor).
    # Exceptions: None explicitly raised. Side effects: None.
    """Simple causal masking function."""
    return q_idx >= kv_idx


def sliding_window_causal_mask_func(window_size, b, h, q_idx, kv_idx):
    # [FUNCTION DOC]
    # Causal mask limited to keys at most window_size positions behind the query (window bound first so it can be
    # bound with functools.partial).
    # Inputs: window_size (int); b, h, q_idx, kv_idx. Outputs: boolean (tensor).
    # Exceptions: None explicitly raised. Side effects: None.
    """Sliding window causal masking function."""
    causal = q_idx >= kv_idx
    window = q_idx - kv_idx <= window_size
    return causal & window


def get_block_mask_from_type(
    flex_attention_mask: Optional[str],
    query: torch.Tensor,
    key: torch.Tensor,
    sliding_window: Optional[int] = None,
    position_ids: Optional[torch.Tensor] = None,
    document_ids: Optional[torch.Tensor] = None,
) -> Optional[torch.Tensor]:
    # [FUNCTION DOC]
    # Builds a BlockMask for a named mask type.
    # None or "causal": returns None. "sliding_window": asserts a positive window and builds a cached sliding window causal
    # mask. "document": asserts ids exist; from document_ids (moved to device, non-empty) computes offsets from the
    # count of each unique id in sorted order; otherwise from position_ids (moved to device, flattened, non-empty, at
    # least one zero) computes document lengths between zeros and their offsets, shifted by the first start when it is
    # not zero; then builds a cached document-aware causal mask. Any other string raises.
    # Inputs: flex_attention_mask (optional string); query (shape and device); key (unused); sliding_window (optional int);
    # position_ids (optional tensor); document_ids (optional tensor).
    # Outputs: a BlockMask or None.
    # Exceptions: AssertionError on invalid sliding window, missing/empty ids or no zeros in position_ids; ValueError on
    # unknown type.
    # Side effects: host-device synchronizations (item calls per document); caches masks.
    # KNOWN ISSUE: "causal" returns None, so flex_attention runs unmasked (non-causal) despite the comment.
    # KNOWN ISSUE: the document_ids path assumes each document's tokens are contiguous and ids are increasing.
    # KNOWN ISSUE: when position_ids does not start with zero, offsets are shifted but the document id tensor does not
    # cover leading tokens, so indices near the end overflow; the final return None is unreachable.
    """Get a specific block mask based on mask type.
    
    Args:
        flex_attention_mask: String identifier for mask type
        query: Query states tensor [batch_size, num_heads, seq_len, head_dim]
        key: Key states tensor [batch_size, num_kv_heads, seq_len, head_dim]
        sliding_window: Optional sliding window size
        position_ids: Optional tensor of position IDs for document masking
        document_ids: Optional explicit document IDs tensor
        
    Returns:
        Block mask tensor or None
    """
    if flex_attention_mask is None:
        return None
        
    seq_len = query.size(2)
    device = query.device
    
    if flex_attention_mask == "causal":
        # Simple causal mask - flex_attention handles this efficiently by default
        return None
        
    elif flex_attention_mask == "sliding_window":
        # Assert that sliding_window is provided and valid
        assert sliding_window is not None and sliding_window > 0, \
            f"For 'sliding_window' mask type, sliding_window must be provided and > 0, got {sliding_window}"
            
        # Use partial to bind the window size parameter
        mask_func = functools.partial(sliding_window_causal_mask_func, sliding_window)
        
        return create_block_mask_cached(
            mask_func,
            B=None,
            H=None,
            Q_LEN=seq_len,
            KV_LEN=seq_len,
            device=device,
        )
        
    elif flex_attention_mask == "document":
        # Assert that at least one of document_ids or position_ids is provided
        assert document_ids is not None or position_ids is not None, \
            "For 'document' mask type, either document_ids or position_ids must be provided"
        
        # Handle document masking
        if document_ids is not None:
            # Ensure document_ids is on the right device
            document_ids = document_ids.to(device)
            
            # Validate document_ids
            assert document_ids.numel() > 0, "document_ids tensor cannot be empty"
            
            # If explicit document_ids are provided, convert to offsets format
            unique_docs = torch.unique(document_ids, sorted=True)
            offsets = []
            current_offset = 0
            
            offsets.append(current_offset)
            for doc_id in unique_docs:
                doc_length = (document_ids == doc_id).sum().item()
                current_offset += doc_length
                offsets.append(current_offset)
                
            offsets = torch.tensor(offsets, device=device, dtype=torch.int32)
            
        elif position_ids is not None:
            # Ensure position_ids is on the right device
            position_ids = position_ids.to(device)
            
            # Validate position_ids
            assert position_ids.numel() > 0, "position_ids tensor cannot be empty"
            
            # If position_ids are provided (where 0s indicate document starts)
            if position_ids.dim() == 2:
                position_ids = position_ids.view(-1)
                
            # Find document boundaries (positions where position_id is 0)
            doc_starts = torch.where(position_ids == 0)[0]
            
            # Validate document boundaries
            assert doc_starts.numel() > 0, "No document boundaries found in position_ids (no zeros found)"
            
            # Add the total sequence length as the final boundary
            doc_ends = torch.cat([doc_starts[1:], torch.tensor([len(position_ids)], device=device)])
            
            # Calculate document lengths
            doc_lengths = doc_ends - doc_starts
            
            # Convert to offsets
            offsets = lengths_to_offsets(doc_lengths, device)
            
            # Adjust offsets to account for actual starting positions
            if doc_starts[0] != 0:
                offsets = offsets + doc_starts[0]
        
        # Generate document-aware causal mask
        doc_mask_mod = generate_doc_mask_mod(causal_mask_func, offsets)
        
        return create_block_mask_cached(
            doc_mask_mod,
            B=None,  # Not needed since we handle batching in the mask function
            H=None,
            Q_LEN=seq_len,
            KV_LEN=seq_len,
            device=device,
        )
    else:
        raise ValueError(f"Unknown flex_attention_mask type: {flex_attention_mask}. " 
                         f"Supported types are: 'causal', 'sliding_window', 'document'")
    
    return None


def get_attention_mod_from_type(
    flex_attention_mask: Optional[str],
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
) -> Optional[Callable]:
    # [FUNCTION DOC]
    # Placeholder that always returns None (no score modifier) regardless of arguments.
    # Inputs: flex_attention_mask, query, key, value (all unused). Outputs: None.
    # Exceptions: None explicitly raised. Side effects: None.
    """Get a specific attention modifier based on mask type.
    
    Args:
        flex_attention_mask: String identifier for mask type
        query: Query states tensor [batch_size, num_heads, seq_len, head_dim]
        key: Key states tensor [batch_size, num_kv_heads, seq_len, head_dim]
        value: Value states tensor [batch_size, num_kv_heads, seq_len, head_dim]
        
    Returns:
        Callable attention modifier function or None
    """
    # We're setting this to None as specified in the requirements
    return None


def validate_attention_args(
    flex_attention_mask: Optional[str],
    sliding_window: Optional[int] = None,
    position_ids: Optional[torch.Tensor] = None, 
    document_ids: Optional[torch.Tensor] = None
) -> None:
    # [FUNCTION DOC]
    # Validates arguments for a named mask type: nothing for None or "causal"; "sliding_window" needs a positive window;
    # "document" needs document_ids or position_ids and neither provided tensor may be empty; other names are rejected.
    # Inputs: flex_attention_mask (optional string); sliding_window, position_ids, document_ids (optional).
    # Outputs: None.
    # Exceptions: ValueError for any invalid combination or unknown type.
    # Side effects: None.
    """Validates arguments for attention mask creation.
    
    Args:
        flex_attention_mask: String identifier for mask type
        sliding_window: Optional sliding window size
        position_ids: Optional position IDs tensor
        document_ids: Optional document IDs tensor
        
    Raises:
        ValueError: If arguments are not valid for the specified mask type
    """
    if flex_attention_mask is None:
        return
        
    if flex_attention_mask == "causal":
        # No additional parameters needed for causal mask
        return
        
    elif flex_attention_mask == "sliding_window":
        if sliding_window is None or sliding_window <= 0:
            raise ValueError(f"For 'sliding_window' mask type, sliding_window must be a positive integer, got {sliding_window}")
            
    elif flex_attention_mask == "document":
        if document_ids is None and position_ids is None:
            raise ValueError("For 'document' mask type, either document_ids or position_ids must be provided")
            
        if position_ids is not None and position_ids.numel() == 0:
            raise ValueError("position_ids tensor cannot be empty")
            
        if document_ids is not None and document_ids.numel() == 0:
            raise ValueError("document_ids tensor cannot be empty")
            
    else:
        raise ValueError(f"Unknown flex_attention_mask type: {flex_attention_mask}. "
                         f"Supported types are: 'causal', 'sliding_window', 'document'")
