# [MODULE DOC]
# Llama3-style context-parallel attention (all-gather KV instead of ring passing), adapted from
# zhuzilin/ring-flash-attention; the original triton_utils.py and utils.py were concatenated into this file.
# Algorithm: the full packed sequence is split evenly across ranks of the context-parallel group. Keys and values
# are all-gathered across ranks in groups of heads_k_stride KV heads (double-buffered so the next head group's
# gather overlaps the current computation); each rank then runs varlen FlashAttention of its local queries against
# only the slice of gathered keys it needs (local_k_slice). Backward repeats the head-group gathers and
# reduce-scatters key/value gradients back to their owners.
# Main contents: llama3_flash_attn_prepare_cu_seqlens (per-rank cu_seqlens and K slice), forward/backward, the
# Llama3FlashAttnVarlenFunc autograd function, three packing variants of the entry function, and copies of the
# LSE utilities and communication helpers from ring_attention (Comm is an alias of AllGatherComm).
# Fit in nanotron: llama3_flash_attn_varlen_qkvpacked_func is registered as "llama3_ring_attention" in
# ALL_ATTENTION_FUNCTIONS.
# Import-time side effects: imports flash_attn and triton; JIT-scripts several functions; later TorchScript
# flatten_varlen_lse and unflatten_varlen_lse definitions replace the earlier Triton ones.
# Exports: all names defined here (no __all__).
"""Ring attention implementation using flash attention adapted from https://github.com/zhuzilin/ring-flash-attention/"""

import torch
import torch.distributed as dist
from flash_attn.flash_attn_interface import (
    _flash_attn_varlen_forward,
    _flash_attn_varlen_backward,
)

def llama3_flash_attn_prepare_cu_seqlens(
    cu_seqlens: torch.Tensor, causal: bool, rank: int, world_size: int
):
    # [FUNCTION DOC]
    # Computes this rank's local query layout and the key range it must attend to.
    # Behavior: asserts the total length divides by world_size; length_per_rank is that quotient. Finds via
    # searchsorted the documents overlapping this rank's span [rank times L, (rank plus one) times L); steps left back
    # one when the span starts mid-document. Query cu_seqlens are the overlapping boundaries shifted to local
    # coordinates and clamped to start at 0 and end at L. Key cu_seqlens use the same boundaries relative to the first
    # overlapping document's start; when causal the last key document ends at this rank's span end, otherwise at the
    # document's true end. Also returns max lengths and the global key slice.
    # Inputs: cu_seqlens (int tensor of global cumulative document lengths); causal (bool); rank, world_size (ints).
    # Outputs: tuple of cu_seqlens_q, cu_seqlens_k, max_seqlen_q (int), max_seqlen_k (int), local_k_slice (slice
    # into the gathered key buffer).
    # Exceptions: AssertionError when total length is not divisible by world_size.
    # Side effects: host-device synchronizations through item calls.
    # KNOWN ISSUE: the docstring lists three return values but five are returned.
    """
    Args:
        cu_seqlens: torch.Tensor, the cu_seqlens of all the sequences across the ring process group.

    Returns:
        cu_seqlens_q: torch.Tensor, the cu_seqlens of the q slice for this rank.
        cu_seqlens_k: torch.Tensor, the cu_seqlens of the k slice that the local q need. Note
            that this may be longer than `total_seq_len // world_size`.
        local_k_slice: slice, the slice of the k that the local q need. Note
            that this may be longer than `total_seq_len // world_size`.
    """
    total_length = cu_seqlens[-1]
    assert total_length % world_size == 0
    length_per_rank = total_length // world_size
    left = torch.searchsorted(cu_seqlens, rank * length_per_rank)
    right = torch.searchsorted(cu_seqlens, (rank + 1) * length_per_rank)
    length_per_rank = length_per_rank.item()

    # after this, cu_seqlens[left:right + 1] contains all the sequence for this rank
    if cu_seqlens[left] != rank * length_per_rank:
        left -= 1
    left = left.item()
    right = right.item()

    # q is always the same. just calculate the cu_seqlens for the local slice
    cu_seqlens_q = cu_seqlens[left : right + 1].clone()
    cu_seqlens_q -= rank * length_per_rank
    cu_seqlens_q[0] = 0
    cu_seqlens_q[-1] = length_per_rank

    cu_seqlens_k = cu_seqlens[left : right + 1].clone()
    if causal:
        # when causal, we hope
        # - the last k seq is of the same length as the last q seq
        slice_right = (rank + 1) * length_per_rank
        cu_seqlens_k[-1] = slice_right
    else:
        # when not causal, we hope
        # - the last k is full seq
        slice_right = cu_seqlens[right].item()

    slice_left = cu_seqlens[left].item()
    cu_seqlens_k -= slice_left

    max_seqlen_q = (cu_seqlens_q[1:] - cu_seqlens_q[:-1]).max().item()
    max_seqlen_k = (cu_seqlens_k[1:] - cu_seqlens_k[:-1]).max().item()
    local_k_slice = slice(slice_left, slice_right)
    return cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, local_k_slice


def llama3_flash_attn_varlen_forward(
    process_group,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_q,
    max_seqlen_k,
    heads_k_stride,
    local_k_slice,
    softmax_scale,
    dropout_p=0,
    causal=True,
    window_size=(-1, -1),
    alibi_slopes=None,
    deterministic=False,
):
    # [FUNCTION DOC]
    # Forward pass. Allocates double KV buffers [2, total_k times world_size, heads_k_stride, head_dim], starts
    # all-gathering the first KV head group, then for each head group: waits, swaps buffers, prefetches the next group,
    # selects the query heads matching this KV group (GQA ratio nheads over nheads_k), slices keys/values with
    # local_k_slice and runs _flash_attn_varlen_forward with default arguments overridden (window size passed in the
    # form the installed flash_attn accepts; probabilities only with dropout). Concatenates outputs along heads and
    # LSE along the head dimension.
    # Inputs: process_group; q [local_tokens, nheads, head_dim]; k, v [local_tokens, nheads_k, head_dim]; cu_seqlens_q,
    # cu_seqlens_k, max_seqlen_q, max_seqlen_k, local_k_slice (from llama3_flash_attn_prepare_cu_seqlens);
    # heads_k_stride (KV heads per gather, must divide nheads_k); softmax_scale; dropout_p (0); causal (True);
    # window_size; alibi_slopes; deterministic (unused).
    # Outputs: tuple of output [local_tokens, nheads, head_dim] and LSE.
    # Exceptions: AssertionError on indivisible heads or unexpected kernel output count.
    # Side effects: async all_gather collectives on process_group; CUDA kernels.
    out_list = []
    lse_list = []

    nheads = q.shape[1]
    total_k, nheads_k, head_dim = k.shape
    assert nheads_k % heads_k_stride == 0, f"nheads_k={nheads_k} must be divisible by heads_k_stride={heads_k_stride}"

    world_size = dist.get_world_size(process_group)
    kv_buffer = torch.empty(
        (2, total_k * world_size, heads_k_stride, head_dim),
        dtype=k.dtype,
        device=k.device,
    )

    kv_buffer_copy = torch.empty_like(kv_buffer)

    k_0 = k[:, :heads_k_stride].contiguous()
    v_0 = v[:, :heads_k_stride].contiguous()
    comm = Comm(process_group)

    comm.all_gather(kv_buffer_copy[0], k_0)
    comm.all_gather(kv_buffer_copy[1], v_0)

    for i in range(0, nheads_k, heads_k_stride):
        comm.wait()
        kv_buffer, kv_buffer_copy = kv_buffer_copy, kv_buffer

        if i < nheads_k - heads_k_stride:
            # all_gather the next kv slice
            kv_slice_left = i + heads_k_stride
            kv_slice_right = kv_slice_left + heads_k_stride
            send_k = k[:, kv_slice_left:kv_slice_right].contiguous()
            send_v = v[:, kv_slice_left:kv_slice_right].contiguous()
            comm.all_gather(kv_buffer_copy[0], send_k)
            comm.all_gather(kv_buffer_copy[1], send_v)

        q_i = q[:, i * nheads // nheads_k : (i + heads_k_stride) * nheads // nheads_k]
        k_i = kv_buffer[0][local_k_slice]
        v_i = kv_buffer[1][local_k_slice]

        params = get_default_args(_flash_attn_varlen_forward).copy()
        params.update(
            {
                "q": q_i,
                "k": k_i,
                "v": v_i,
                "cu_seqlens_q": cu_seqlens_q,
                "cu_seqlens_k": cu_seqlens_k,
                "max_seqlen_q": max_seqlen_q,
                "max_seqlen_k": max_seqlen_k,
                "dropout_p": dropout_p,
                "softmax_scale": softmax_scale,
                "causal": causal,
                "alibi_slopes": alibi_slopes,
                "return_softmax": True and dropout_p > 0,
            }
        )
        if "window_size" in params:
            params.update({"window_size": window_size})
        else:
            params.update(
                {
                    "window_size_left": window_size[0],
                    "window_size_right": window_size[1],
                }
            )
        outputs = _flash_attn_varlen_forward(**params)
        if len(outputs) == 8:
            out, _, _, _, _, lse, _, _ = outputs
        else:
            assert len(outputs) == 4
            out, lse, _, _ = outputs
        out_list.append(out)
        lse_list.append(lse)

    out = torch.cat(out_list, dim=1)
    lse = torch.cat(lse_list, dim=-2)
    return out, lse


def llama3_flash_attn_varlen_backward(
    process_group,
    dout,
    q,
    k,
    v,
    out,
    softmax_lse,
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_q,
    max_seqlen_k,
    heads_k_stride,
    local_k_slice,
    softmax_scale,
    dropout_p=0,
    causal=True,
    window_size=(-1, -1),
    alibi_slopes=None,
    deterministic=False,
):
    # [FUNCTION DOC]
    # Backward pass. For each KV head group: zeroes a full-size dKV buffer, selects query/out/dout/dq/LSE head slices,
    # waits for and swaps the gathered KV buffer, prefetches the next group, runs _flash_attn_varlen_backward writing dq
    # directly into the dq slice and dk/dv into the local_k_slice region of the dKV buffer, then reduce-scatters dK and
    # dV across ranks (summing contributions) into either dk/dv directly (single group) or a contiguous temporary that
    # is copied into the group's head slice.
    # Inputs: process_group; dout; q, k, v; out; softmax_lse; cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k,
    # heads_k_stride, local_k_slice; softmax_scale; dropout_p; causal; window_size; alibi_slopes; deterministic.
    # Outputs: tuple dq, dk, dv matching the shapes of q, k, v.
    # Exceptions: AssertionError when heads_k_stride does not divide nheads_k.
    # Side effects: all_gather and synchronous reduce_scatter collectives; CUDA kernels.
    nheads = q.shape[1]
    total_k, nheads_k, head_dim = k.shape
    assert nheads_k % heads_k_stride == 0

    world_size = dist.get_world_size(process_group)
    kv_buffer = torch.empty(
        (2, total_k * world_size, heads_k_stride, head_dim),
        dtype=k.dtype,
        device=k.device,
    )
    kv_buffer_copy = torch.empty_like(kv_buffer)

    dkv_buffer = torch.empty(
        (2, total_k * world_size, heads_k_stride, head_dim),
        dtype=k.dtype,
        device=k.device,
    )

    if heads_k_stride != nheads_k:
        kv_contiguous_buffer = torch.empty(
            (2, total_k, heads_k_stride, head_dim),
            dtype=k.dtype,
            device=k.device,
        )

    dq = torch.empty_like(q)
    dk = torch.empty_like(k)
    dv = torch.empty_like(v)

    comm = Comm(process_group)

    k_0 = k[:, :heads_k_stride].contiguous()
    v_0 = v[:, :heads_k_stride].contiguous()
    comm.all_gather(kv_buffer_copy[0], k_0)
    comm.all_gather(kv_buffer_copy[1], v_0)

    for i in range(0, nheads_k, heads_k_stride):
        dkv_buffer.zero_()

        q_slice = slice(
            i * nheads // nheads_k, (i + heads_k_stride) * nheads // nheads_k
        )
        q_i = q[:, q_slice]
        dout_i = dout[:, q_slice]
        out_i = out[:, q_slice]
        dq_i = dq[:, q_slice]
        if softmax_lse.dim() == 3:
            lse_i = softmax_lse[:, q_slice].contiguous()
        else:
            lse_i = softmax_lse[q_slice]

        comm.wait()
        kv_buffer, kv_buffer_copy = kv_buffer_copy, kv_buffer

        if i < nheads_k - heads_k_stride:
            # all_gather the next kv slice
            kv_slice_left = i + heads_k_stride
            kv_slice_right = kv_slice_left + heads_k_stride
            send_k = k[:, kv_slice_left:kv_slice_right].contiguous()
            send_v = v[:, kv_slice_left:kv_slice_right].contiguous()
            comm.all_gather(kv_buffer_copy[0], send_k)
            comm.all_gather(kv_buffer_copy[1], send_v)

        k_i = kv_buffer[0][local_k_slice]
        v_i = kv_buffer[1][local_k_slice]
        dk_i = dkv_buffer[0][local_k_slice]
        dv_i = dkv_buffer[1][local_k_slice]

        params = get_default_args(_flash_attn_varlen_backward).copy()
        params.update(
            {
                "dout": dout_i,
                "q": q_i,
                "k": k_i,
                "v": v_i,
                "out": out_i,
                "softmax_lse": lse_i,
                "dq": dq_i,
                "dk": dk_i,
                "dv": dv_i,
                "cu_seqlens_q": cu_seqlens_q,
                "cu_seqlens_k": cu_seqlens_k,
                "max_seqlen_q": max_seqlen_q,
                "max_seqlen_k": max_seqlen_k,
                "dropout_p": dropout_p,
                "softmax_scale": softmax_scale,
                "causal": causal,
                "alibi_slopes": alibi_slopes,
                "deterministic": deterministic,
            }
        )
        if "window_size" in params:
            params.update({"window_size": window_size})
        else:
            params.update(
                {
                    "window_size_left": window_size[0],
                    "window_size_right": window_size[1],
                }
            )
        _flash_attn_varlen_backward(**params)

        if heads_k_stride != nheads_k:
            # reduce_scatter needs contiguous buffer
            dk_i = kv_contiguous_buffer[0]
            dv_i = kv_contiguous_buffer[1]
        else:
            dk_i = dk
            dv_i = dv

        dist.reduce_scatter_tensor(dk_i, dkv_buffer[0], group=process_group)
        dist.reduce_scatter_tensor(dv_i, dkv_buffer[1], group=process_group)

        if heads_k_stride != nheads_k:
            dk[:, i : i + heads_k_stride] = dk_i
            dv[:, i : i + heads_k_stride] = dv_i

    return dq, dk, dv


class Llama3FlashAttnVarlenFunc(torch.autograd.Function):
    # [CLASS DOC]
    # torch.autograd.Function wrapping llama3 forward/backward; saves q, k, v, out, LSE, cu_seqlens_q and cu_seqlens_k
    # plus scalar settings (max lengths, heads_k_stride, local_k_slice, dropout, scale, causal, window, alibi,
    # deterministic, group) on ctx.
    @staticmethod
    def forward(
        ctx,
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        heads_k_stride,
        local_k_slice,
        dropout_p,
        softmax_scale,
        causal,
        window_size,
        alibi_slopes,
        deterministic,
        return_softmax,
        group,
    ):
        # [FUNCTION DOC]
        # Defaults softmax_scale to one over sqrt(head_dim), makes k and v contiguous, runs llama3_flash_attn_varlen_forward
        # (deterministic forced False) and stores state.
        # Inputs: ctx and the seventeen arguments listed in the signature (q, k, v, cu_seqlens_q, cu_seqlens_k,
        # max_seqlen_q, max_seqlen_k, heads_k_stride, local_k_slice, dropout_p, softmax_scale, causal, window_size,
        # alibi_slopes which must be None, deterministic, return_softmax, group).
        # Outputs: output tensor, or tuple of output, LSE and None when return_softmax is true.
        # Exceptions: AssertionError if alibi_slopes is not None.
        # Side effects: collectives; stores tensors on ctx.
        if softmax_scale is None:
            softmax_scale = q.shape[-1] ** (-0.5)

        assert alibi_slopes is None
        k = k.contiguous()
        v = v.contiguous()
        out, softmax_lse = llama3_flash_attn_varlen_forward(
            group,
            q,
            k,
            v,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            heads_k_stride,
            local_k_slice,
            softmax_scale=softmax_scale,
            dropout_p=dropout_p,
            causal=causal,
            window_size=window_size,
            alibi_slopes=alibi_slopes,
            deterministic=False,
        )
        # this should be out_padded
        ctx.save_for_backward(q, k, v, out, softmax_lse, cu_seqlens_q, cu_seqlens_k)
        ctx.max_seqlen_q = max_seqlen_q
        ctx.max_seqlen_k = max_seqlen_k
        ctx.heads_k_stride = heads_k_stride
        ctx.local_k_slice = local_k_slice
        ctx.dropout_p = dropout_p
        ctx.softmax_scale = softmax_scale
        ctx.causal = causal
        ctx.window_size = window_size
        ctx.alibi_slopes = alibi_slopes
        ctx.deterministic = deterministic
        ctx.group = group
        return out if not return_softmax else (out, softmax_lse, None)

    @staticmethod
    def backward(ctx, dout, *args):
        # [FUNCTION DOC]
        # Runs llama3_flash_attn_varlen_backward on saved state and returns dq, dk, dv followed by None values.
        # Inputs: ctx; dout; extra output gradients ignored.
        # Outputs: tuple of 18 entries.
        # Exceptions: None explicitly raised. Side effects: collectives.
        # KNOWN ISSUE: forward takes 17 inputs but backward returns 18 gradients (3 plus 15 None), which autograd rejects
        # with a gradient-count error.
        q, k, v, out, softmax_lse, cu_seqlens_q, cu_seqlens_k = ctx.saved_tensors
        dq, dk, dv = llama3_flash_attn_varlen_backward(
            ctx.group,
            dout,
            q,
            k,
            v,
            out,
            softmax_lse,
            cu_seqlens_q,
            cu_seqlens_k,
            ctx.max_seqlen_q,
            ctx.max_seqlen_k,
            ctx.heads_k_stride,
            ctx.local_k_slice,
            softmax_scale=ctx.softmax_scale,
            dropout_p=ctx.dropout_p,
            causal=ctx.causal,
            window_size=ctx.window_size,
            alibi_slopes=ctx.alibi_slopes,
            deterministic=ctx.deterministic,
        )
        return (dq, dk, dv) + (None,) * 15


def llama3_flash_attn_varlen_qkvpacked_func(
    qkv,
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_q,
    max_seqlen_k,
    heads_k_stride,
    local_k_slice,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),  # -1 means infinite context window
    alibi_slopes=None,
    deterministic=False,
    return_attn_probs=False,
    group=None,
):
    # [FUNCTION DOC]
    # Entry point for packed QKV [tokens, 3, heads, head_dim]: splits along dimension 1 and applies
    # Llama3FlashAttnVarlenFunc.
    # Inputs: qkv; cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, heads_k_stride, local_k_slice;
    # dropout_p (0.0); softmax_scale (None); causal (False); window_size (unlimited); alibi_slopes; deterministic;
    # return_attn_probs; group.
    # Outputs: as Llama3FlashAttnVarlenFunc.forward.
    # Exceptions: see forward. Side effects: collectives.
    # KNOWN ISSUE: it is registered in ALL_ATTENTION_FUNCTIONS, but its signature (qkv first, no module argument,
    # separate q/k lengths) does not match the module, query, key, value calling convention of the other backends.
    return Llama3FlashAttnVarlenFunc.apply(
        qkv[:, 0],
        qkv[:, 1],
        qkv[:, 2],
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        heads_k_stride,
        local_k_slice,
        dropout_p,
        softmax_scale,
        causal,
        window_size,
        alibi_slopes,
        deterministic,
        return_attn_probs,
        group,
    )


def llama3_flash_attn_varlen_kvpacked_func(
    q,
    kv,
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_q,
    max_seqlen_k,
    heads_k_stride,
    local_k_slice,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),  # -1 means infinite context window
    alibi_slopes=None,
    deterministic=False,
    return_attn_probs=False,
    group=None,
):
    # [FUNCTION DOC]
    # Entry point with separate q and packed KV [tokens, 2, kv_heads, head_dim]; splits kv and applies
    # Llama3FlashAttnVarlenFunc. Other inputs, outputs, exceptions and side effects as the qkvpacked variant.
    return Llama3FlashAttnVarlenFunc.apply(
        q,
        kv[:, 0],
        kv[:, 1],
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        heads_k_stride,
        local_k_slice,
        dropout_p,
        softmax_scale,
        causal,
        window_size,
        alibi_slopes,
        deterministic,
        return_attn_probs,
        group,
    )


def llama3_flash_attn_varlen_func(
    q,
    k,
    v,
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_q,
    max_seqlen_k,
    heads_k_stride,
    local_k_slice,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),  # -1 means infinite context window
    alibi_slopes=None,
    deterministic=False,
    return_attn_probs=False,
    group=None,
):
    # [FUNCTION DOC]
    # Entry point with separate q, k and v; applies Llama3FlashAttnVarlenFunc. Other inputs, outputs, exceptions and side
    # effects as the qkvpacked variant.
    return Llama3FlashAttnVarlenFunc.apply(
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        heads_k_stride,
        local_k_slice,
        dropout_p,
        softmax_scale,
        causal,
        window_size,
        alibi_slopes,
        deterministic,
        return_attn_probs,
        group,
    )




## triton_utils.py
import torch
import triton
import triton.language as tl


@triton.jit
def flatten_kernel(
    # pointers to matrices
    OUT,
    LSE,
    CU_SEQLENS,
    # strides
    stride_out_nheads,
    stride_out_seqlen,
    stride_lse_batch,
    stride_lse_nheads,
    stride_lse_seqlen,
    # meta-parameters
    BLOCK_M: tl.constexpr,
):
    # [FUNCTION DOC]
    # Triton kernel copying padded LSE [batch, heads, max_seqlen] into flat [heads, total_tokens]. Program ids are
    # (sequence block, batch, head); each program copies BLOCK_M positions of one document, masked to its length.
    # Inputs: OUT, LSE, CU_SEQLENS pointers; output and LSE strides; BLOCK_M (constexpr).
    # Outputs: None. Exceptions: None explicitly raised. Side effects: writes OUT.
    pid_m = tl.program_id(axis=0)
    pid_batch = tl.program_id(axis=1)
    pid_head = tl.program_id(axis=2)

    start_idx = tl.load(CU_SEQLENS + pid_batch)
    seqlen = tl.load(CU_SEQLENS + pid_batch + 1) - start_idx
    LSE = LSE + pid_batch * stride_lse_batch + pid_head * stride_lse_nheads
    OUT = OUT + pid_head * stride_out_nheads + start_idx * stride_out_seqlen

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)

    LSE = LSE + rm[:, None] * stride_lse_seqlen
    x = tl.load(LSE, mask=rm[:, None] < seqlen, other=0.0)

    OUT = OUT + rm[:, None] * stride_out_seqlen
    tl.store(OUT, x, mask=rm[:, None] < seqlen)


def flatten_varlen_lse(lse, cu_seqlens):
    # [FUNCTION DOC]
    # HELPER: Triton version converting padded LSE to flat LSE (shadowed by the later TorchScript definition).
    # Allocates [heads, total_seqlen] and launches flatten_kernel with a grid of (blocks of 4 positions, batch, heads)
    # on the LSE's CUDA device.
    # Inputs: lse [batch, heads, max_seqlen]; cu_seqlens [batch + 1].
    # Outputs: tensor [heads, total_seqlen].
    # Exceptions: None explicitly raised. Side effects: launches a CUDA kernel; sets the current CUDA device temporarily.
    # KNOWN ISSUE: this definition is dead code because a second flatten_varlen_lse later in the module replaces it.
    """
    Arguments:
        lse: (batch_size, nheads, max_seqlen)
        cu_seqlens: (batch_size + 1,)
    Return:
        flatten_lse: (nheads, total_seqlen)
    """
    total_seqlen = cu_seqlens[-1]
    batch_size, nheads, max_seqlen = lse.shape
    output = torch.empty((nheads, total_seqlen), dtype=lse.dtype, device=lse.device)

    def grid(META):
        # [FUNCTION DOC]
        # HELPER: Triton launch grid: ceil of max_seqlen over BLOCK_M, batch size, number of heads.
        # Inputs: META (kernel meta-parameters). Outputs: tuple of three ints.
        # Exceptions: None explicitly raised. Side effects: None.
        return triton.cdiv(max_seqlen, META["BLOCK_M"]), batch_size, nheads

    BLOCK_M = 4

    with torch.cuda.device(lse.device.index):
        flatten_kernel[grid](
            output,
            lse,
            cu_seqlens,
            # strides
            output.stride(0),
            output.stride(1),
            lse.stride(0),
            lse.stride(1),
            lse.stride(2),
            BLOCK_M,
        )
    return output


@triton.jit
def unflatten_kernel(
    # pointers to matrices
    OUT,
    LSE,
    CU_SEQLENS,
    # strides
    stride_out_batch,
    stride_out_nheads,
    stride_out_seqlen,
    stride_lse_seqlen,
    stride_lse_nheads,
    # meta-parameters
    BLOCK_M: tl.constexpr,
):
    # [FUNCTION DOC]
    # Triton kernel copying flat LSE [tokens, heads] into padded [batch, heads, max_seqlen], one program per
    # (sequence block, batch, head), masked to each document's length (padding left uninitialized).
    # Inputs: OUT, LSE, CU_SEQLENS pointers; strides; BLOCK_M (constexpr).
    # Outputs: None. Exceptions: None explicitly raised. Side effects: writes OUT.
    pid_m = tl.program_id(axis=0)
    pid_batch = tl.program_id(axis=1)
    pid_head = tl.program_id(axis=2)

    start_idx = tl.load(CU_SEQLENS + pid_batch)
    seqlen = tl.load(CU_SEQLENS + pid_batch + 1) - start_idx
    LSE = LSE + pid_head * stride_lse_nheads + start_idx * stride_lse_seqlen
    OUT = OUT + pid_batch * stride_out_batch + pid_head * stride_out_nheads

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)

    LSE = LSE + rm[:, None] * stride_lse_seqlen
    x = tl.load(LSE, mask=rm[:, None] < seqlen, other=0.0)

    OUT = OUT + rm[:, None] * stride_out_seqlen
    tl.store(OUT, x, mask=rm[:, None] < seqlen)


def unflatten_varlen_lse(lse, cu_seqlens, max_seqlen: int):
    # [FUNCTION DOC]
    # HELPER: Triton version converting flat LSE back to padded form (shadowed by the later TorchScript definition).
    # Adds a trailing dimension, allocates [batch, heads, max_seqlen] and launches unflatten_kernel.
    # Inputs: lse [total_seqlen, heads, 1] per docstring; cu_seqlens; max_seqlen (int).
    # Outputs: tensor [batch, heads, max_seqlen].
    # Exceptions: None explicitly raised. Side effects: launches a CUDA kernel.
    # KNOWN ISSUE: dead code (redefined later). It also unsqueezes an input documented as already 3-D, so the heads
    # dimension index would be off for that documented shape.
    """
    Arguments:
        lse: (total_seqlen, nheads, 1)
        cu_seqlens: (batch_size + 1,)
        max_seqlen: int
    Return:
        unflatten_lse: (batch_size, nheads, max_seqlen)
    """
    lse = lse.unsqueeze(dim=-1)
    batch_size = len(cu_seqlens) - 1
    nheads = lse.shape[1]
    output = torch.empty(
        (batch_size, nheads, max_seqlen),
        dtype=lse.dtype,
        device=lse.device,
    )

    def grid(META):
        # [FUNCTION DOC]
        # HELPER: Triton launch grid: ceil of max_seqlen over BLOCK_M, batch size, number of heads.
        # Inputs: META. Outputs: tuple of three ints.
        # Exceptions: None explicitly raised. Side effects: None.
        return triton.cdiv(max_seqlen, META["BLOCK_M"]), batch_size, nheads

    BLOCK_M = 4

    with torch.cuda.device(lse.device.index):
        unflatten_kernel[grid](
            output,
            lse,
            cu_seqlens,
            # strides
            output.stride(0),
            output.stride(1),
            output.stride(2),
            lse.stride(0),
            lse.stride(1),
            BLOCK_M,
        )
    return output


import inspect
from functools import cache
from typing import Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn.functional as F


@cache
def _get_default_args(func):
    # [FUNCTION DOC]
    # HELPER: cached introspection used by get_default_args.
    # Builds a dictionary from every positional parameter name of func to its default value (None for parameters
    # without defaults); sets softcap to 0.0 when present.
    # Inputs: func, a Python function. Outputs: dictionary (shared cached object; callers copy it).
    # Exceptions: None explicitly raised. Side effects: caches per function.
    spec = inspect.getfullargspec(func)
    defaults = spec.defaults if spec.defaults is not None else ()
    padded_defaults = (None,) * (len(spec.args) - len(defaults)) + defaults
    args = dict(zip(spec.args, padded_defaults))
    if "softcap" in args:
        args["softcap"] = 0.0
    return args


def get_default_args(func):
    # [FUNCTION DOC]
    # Returns the default-argument dictionary for a flash_attn function; for torch custom-op wrappers that are not plain
    # functions, introspects their underlying _init_fn.
    # Inputs: func. Outputs: dictionary of parameter names to defaults.
    # Exceptions: AttributeError if func is neither a function nor has _init_fn.
    # Side effects: None.
    if inspect.isfunction(func):
        return _get_default_args(func)
    else:
        # Use the origin _init_fn in CustomOpDef
        return _get_default_args(func._init_fn)


@torch.jit.script
def _update_out_and_lse(
    out: torch.Tensor,
    lse: torch.Tensor,
    block_out: torch.Tensor,
    block_lse: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:

    # [FUNCTION DOC]
    # HELPER: TorchScript LSE merge used by update_out_and_lse.
    # Casts block_out to float32 and reshapes block_lse from [heads, tokens] to [tokens, heads, 1]; then the new output
    # is out minus sigmoid of (block_lse minus lse) times (out minus block_out), and the new lse is lse minus
    # logsigmoid of (lse minus block_lse) (equivalent to log of the sum of exponentials, but stable).
    # Inputs: out float32 [tokens, heads, head_dim]; lse [tokens, heads, 1]; block_out; block_lse [heads, tokens].
    # Outputs: tuple of updated out and lse.
    # Exceptions: None explicitly raised. Side effects: None.
    block_out = block_out.to(torch.float32)
    block_lse = block_lse.transpose(-2, -1).unsqueeze(dim=-1)

    # new_lse = lse + torch.log(1 + torch.exp(block_lse - lse))
    # torch.exp(lse - new_lse) * out + torch.exp(block_lse - new_lse) * block_out
    # For additional context and discussion, please refer to:
    # https://github.com/zhuzilin/ring-flash-attention/pull/34#issuecomment-2076126795
    out = out - F.sigmoid(block_lse - lse) * (out - block_out)
    lse = lse - F.logsigmoid(lse - block_lse)

    return out, lse


def update_out_and_lse(
    out: Optional[torch.Tensor],
    lse: Optional[torch.Tensor],
    block_out: torch.Tensor,
    block_lse: torch.Tensor,
    slice_=None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    # [FUNCTION DOC]
    # Merges a new attention block into the running output. On the first call (out is None) initializes out as float32
    # block_out and lse as reshaped block_lse. With slice_, updates only that slice of out and lse in place; otherwise
    # merges the full tensors.
    # Inputs: out, lse (optional running state); block_out, block_lse (new block); slice_ (optional index).
    # Outputs: tuple of out and lse.
    # Exceptions: RuntimeError when slice_ is given on the first call.
    # Side effects: mutates out and lse in place when slice_ is given.
    if out is None:
        if slice_ is not None:
            raise RuntimeError("first update_out_and_lse should not pass slice_ args")
        out = block_out.to(torch.float32)
        lse = block_lse.transpose(-2, -1).unsqueeze(dim=-1)
    elif slice_ is not None:
        slice_out, slice_lse = out[slice_], lse[slice_]
        slice_out, slice_lse = _update_out_and_lse(slice_out, slice_lse, block_out, block_lse)
        out[slice_], lse[slice_] = slice_out, slice_lse
    else:
        out, lse = _update_out_and_lse(out, lse, block_out, block_lse)
    return out, lse


@torch.jit.script
def flatten_varlen_lse(lse, cu_seqlens):
    # [FUNCTION DOC]
    # HELPER: TorchScript version (the one actually used) converting padded LSE [batch, heads, max_seqlen] to flat
    # [heads, total_tokens] by concatenating each document's first length entries along the sequence dimension.
    # Inputs: lse; cu_seqlens. Outputs: tensor [heads, total_tokens].
    # Exceptions: None explicitly raised. Side effects: None.
    new_lse = []
    for i in range(len(cu_seqlens) - 1):
        start, end = cu_seqlens[i], cu_seqlens[i + 1]
        new_lse.append(lse[i, :, : end - start])
    return torch.cat(new_lse, dim=1)


@torch.jit.script
def unflatten_varlen_lse(lse, cu_seqlens, max_seqlen: int):
    # [FUNCTION DOC]
    # HELPER: TorchScript version (the one actually used) converting flat LSE [tokens, heads, 1] to padded
    # [batch, heads, max_seqlen] float32, contiguous; padding entries are uninitialized.
    # Inputs: lse; cu_seqlens; max_seqlen (int). Outputs: tensor [batch, heads, max_seqlen].
    # Exceptions: None explicitly raised. Side effects: None.
    num_seq = len(cu_seqlens) - 1
    num_head = lse.shape[-2]
    new_lse = torch.empty((num_seq, max_seqlen, num_head, 1), dtype=torch.float32, device=lse.device)
    for i in range(num_seq):
        start, end = cu_seqlens[i], cu_seqlens[i + 1]
        new_lse[i, : end - start] = lse[start:end]
    return new_lse.squeeze(dim=-1).transpose(1, 2).contiguous()


## utils.py
class RingComm:
    # [CLASS DOC]
    # Batched asynchronous point-to-point helper for a ring. Each rank sends to rank plus one and receives from rank
    # minus one (modulo world size), with group ranks translated to global ranks.
    # Key attributes: _process_group, _ops (pending P2POp list), _reqs (outstanding requests or None), rank,
    # world_size, send_rank, recv_rank. Invariant: commit and wait strictly alternate.
    def __init__(self, process_group: dist.ProcessGroup):
        # [FUNCTION DOC]
        # Records rank/world size in the group and computes neighbor ranks (converted to global ranks when a group is given).
        # Inputs: process_group (or None for the default group).
        # Outputs: None. Exceptions: errors from torch.distributed if not initialized. Side effects: None.
        self._process_group = process_group
        self._ops = []
        self.rank = dist.get_rank(self._process_group)
        self.world_size = dist.get_world_size(self._process_group)
        self._reqs = None

        self.send_rank = (self.rank + 1) % self.world_size
        self.recv_rank = (self.rank - 1) % self.world_size

        if process_group is not None:
            self.send_rank = dist.get_global_rank(self._process_group, self.send_rank)
            self.recv_rank = dist.get_global_rank(self._process_group, self.recv_rank)

    def send_recv(self, to_send: torch.Tensor, recv_tensor: Optional[torch.Tensor] = None) -> torch.Tensor:
        # [FUNCTION DOC]
        # Queues an isend of to_send to send_rank and an irecv from recv_rank into recv_tensor (or a new empty tensor like
        # to_send); nothing is issued until commit.
        # Inputs: to_send (tensor); recv_tensor (optional buffer).
        # Outputs: the receive buffer (valid only after wait).
        # Exceptions: None explicitly raised. Side effects: appends to self._ops.
        if recv_tensor is None:
            res = torch.empty_like(to_send)
        else:
            res = recv_tensor

        send_op = dist.P2POp(dist.isend, to_send, self.send_rank, group=self._process_group)
        recv_op = dist.P2POp(dist.irecv, res, self.recv_rank, group=self._process_group)
        self._ops.append(send_op)
        self._ops.append(recv_op)
        return res

    def commit(self):
        # [FUNCTION DOC]
        # Launches all queued operations with batch_isend_irecv.
        # Inputs: none. Outputs: None.
        # Exceptions: RuntimeError if called twice without wait.
        # Side effects: starts network communication; sets self._reqs.
        if self._reqs is not None:
            raise RuntimeError("commit called twice")
        self._reqs = dist.batch_isend_irecv(self._ops)

    def wait(self):
        # [FUNCTION DOC]
        # Waits for all outstanding requests, then clears requests and queued ops.
        # Inputs: none. Outputs: None.
        # Exceptions: RuntimeError if called before commit.
        # Side effects: blocks on communication; resets state.
        if self._reqs is None:
            raise RuntimeError("wait called before commit")
        for req in self._reqs:
            req.wait()
        self._reqs = None
        self._ops = []

    def send_recv_kv(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
        k_buffer: Optional[torch.Tensor] = None,
        v_buffer: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # [FUNCTION DOC]
        # Queues send/recv for both k and v and commits them.
        # Inputs: k, v (tensors); k_buffer, v_buffer (optional receive buffers).
        # Outputs: tuple of receive buffers for next k and next v (valid after wait).
        # Exceptions: RuntimeError if a previous commit was not waited. Side effects: starts communication.
        next_k, next_v = self.send_recv(k, k_buffer), self.send_recv(v, v_buffer)
        self.commit()
        return next_k, next_v


class AllGatherComm:
    # [CLASS DOC]
    # Collects asynchronous all_gather_into_tensor handles so they can be awaited together.
    # Key attributes: group (process group or None), handles (list of pending work handles).
    def __init__(self, group=None) -> None:
        # [FUNCTION DOC]
        # Stores the group and an empty handle list.
        # Inputs: group (optional process group). Outputs: None.
        # Exceptions: None explicitly raised. Side effects: None.
        self.group = group
        self.handles = []

    def all_gather(self, output_tensor: torch.Tensor, input_tensor: torch.Tensor):
        # [FUNCTION DOC]
        # Starts an async all_gather_into_tensor of input_tensor into output_tensor on self.group and records the handle.
        # Inputs: output_tensor (world_size times input size); input_tensor.
        # Outputs: None. Exceptions: None explicitly raised. Side effects: collective communication.
        handle = dist.all_gather_into_tensor(output_tensor, input_tensor, group=self.group, async_op=True)
        self.handles.append(handle)

    def wait(self):
        # [FUNCTION DOC]
        # Waits on all recorded handles and clears the list.
        # Inputs: none. Outputs: None. Exceptions: None explicitly raised. Side effects: blocks on communication.
        for handle in self.handles:
            handle.wait()
        self.handles = []

Comm = AllGatherComm
