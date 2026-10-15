# [MODULE DOC]
# Triton-based ring flash attention adapted from lucidrains/ring-attention-pytorch, with its ring.py,
# distributed.py and triton_flash_attn.py sources concatenated into one file.
# Algorithm: each rank holds a sequence chunk; keys/values (and in backward their gradients) can be passed around
# the ring with blocking point-to-point exchanges while a modified Triton FlashAttention accumulates unnormalized
# output, row maxima and log-sum-exp across passes. Supports causal masking (block-diagonal or striped layout),
# key padding masks (as additive bias), grouped-query attention by repeating KV heads, limited lookback and
# softclamping of attention logits.
# Main contents: RingFlashAttentionCUDAFunction and the ring_flash_attn_cuda entry point (registered lazily as
# "ring_flash_triton" in ALL_ATTENTION_FUNCTIONS), ring index/pass utilities, distributed all-gather helpers, and
# the Triton forward/backward kernels with their Python launchers.
# Import-time side effects: queries the installed version of the "triton-nightly" distribution; when missing it
# prints a message and calls exit(), terminating the process; asserts triton-nightly is 3.0.0 or newer; compiles
# nothing until kernels are launched. Several helper names are redefined by later sections.
# Exports: no __all__; ring_flash_attn_cuda is the intended public entry.
# KNOWN ISSUE: exiting the interpreter at import time when triton-nightly is absent is far more drastic than raising
# ImportError, and regular triton installations are rejected even when new enough.
"""Ring attention implementation adapted from https://github.com/lucidrains/ring-attention-pytorch/blob/main/ring_attention_pytorch/ring_flash_attention_cuda.py"""
from __future__ import annotations

import math
from math import ceil

import torch
from einops import rearrange, reduce, repeat
from torch import Tensor
from torch.amp import autocast
from torch.autograd.function import Function

# helpers


def exists(v):
    # [FUNCTION DOC]
    # HELPER: null check used throughout the module (redefined identically in each concatenated section).
    # Inputs: a value. Outputs: True when it is not None.
    # Exceptions: None explicitly raised. Side effects: None.
    return v is not None


def default(val, d):
    # [FUNCTION DOC]
    # HELPER: returns the first argument unless it is None, in which case returns the fallback (redefined per section).
    # Inputs: value, fallback. Outputs: the chosen value.
    # Exceptions: None explicitly raised. Side effects: None.
    return val if exists(val) else d


def divisible_by(num, den):
    # [FUNCTION DOC]
    # HELPER: divisibility test (redefined per section).
    # Inputs: num, den (ints). Outputs: True when num modulo den is zero.
    # Exceptions: ZeroDivisionError when den is zero. Side effects: None.
    return (num % den) == 0


# ring + (flash) attention forwards and backwards

# flash attention v1 - https://arxiv.org/abs/2205.14135
# flash attention v2 - https://tridao.me/publications/flash2/flash2.pdf
# ring attention - https://arxiv.org/abs/2310.01889


class RingFlashAttentionCUDAFunction(Function):
    # [CLASS DOC]
    # Autograd function implementing ring attention with the Triton flash kernels. Stores on ctx a tuple of settings
    # (causal, softmax_scale, original mask, bucket_size, ring_reduce_col, max_ring_passes, num_lookback_buckets,
    # striped_ring_attn, q_head_groups, softclamp settings, original dtype, ring_pg) and saves q, k, v, o and lse.
    # Inputs layout: q [batch, seq, q_heads, head_dim], k and v [batch, seq, kv_heads, head_dim] on CUDA.
    @staticmethod
    @torch.no_grad()
    def forward(
        ctx,
        q: Tensor,  # [b, num_heads, seq_length, head_dim]
        k: Tensor,
        v: Tensor,
        mask: Tensor | None,
        causal: bool,
        bucket_size: int,
        ring_reduce_col: bool,
        striped_ring_attn: bool,
        max_lookback_seq_len: int | None,
        softclamp_qk_sim: bool,
        softclamp_value: float,
        ring_pg: dist.ProcessGroup | None,
    ):

        # [FUNCTION DOC]
        # Forward pass without gradient tracking.
        # Steps: checks shapes, GQA divisibility and CUDA placement; softmax_scale is one over sqrt(head_dim); float32
        # inputs are cast to half. Ring size comes from ring_pg or the world size. Cross attention (different q and k
        # lengths) disables ring passing and striping. With max_lookback_seq_len (causal only) the number of ring passes is
        # limited to ceil of lookback over chunk length. Causal attention drops the padding mask. bucket_size is clamped to
        # the chunk length. Iterates over ring passes (all_ring_pass when ring_reduce_col, otherwise only the local chunk):
        # repeats KV heads for GQA, converts the boolean mask to a 0 / minus infinity bias, decides block causality (striped:
        # always causal, diagonal masked when this rank is below the source rank; non-striped: causal on the diagonal block
        # and skips blocks from later ranks), then accumulates with flash_attn_forward. When not fused, normalizes the
        # output by exp of m minus lse at the end. Casts output back to the original dtype.
        # Inputs: ctx; q, k, v; mask (optional boolean key padding mask [batch, seq_k]); causal; bucket_size;
        # ring_reduce_col (pass KV around the ring); striped_ring_attn; max_lookback_seq_len (optional); softclamp_qk_sim;
        # softclamp_value; ring_pg (optional process group).
        # Outputs: attention output with q's shape and original dtype.
        # Exceptions: AssertionError on shape/device/lookback violations.
        # Side effects: blocking send/recv plus barriers on the ring group when ring_reduce_col; stores state on ctx.
        # KNOWN ISSUE: the conditional expressions computing causal_mask_diagonal and block_causal are parsed as "get_rank()
        # if ring_pg is None else (comparison)", so without a ring_pg they receive the raw rank integer instead of a boolean.
        # KNOWN ISSUE: with the default ring_reduce_col False only the local chunk is attended (no ring communication).
        # KNOWN ISSUE: flash_attn_forward no longer launches its kernel (see that function), so the output is not computed.
        # KNOWN ISSUE: q_seq_len is taken from dimension 1 of q, which is the sequence dimension only for the documented
        # [b, s, h, d] layout although the parameter comment says [b, num_heads, seq_length, head_dim].
        assert k.shape[-2:] == v.shape[-2:]
        q_heads, kv_heads = q.shape[-2], k.shape[-2]

        assert divisible_by(q_heads, kv_heads)
        q_head_groups = q_heads // kv_heads

        assert all(t.is_cuda for t in (q, k, v)), "inputs must be all on cuda"

        dtype = q.dtype
        softmax_scale = q.shape[-1] ** -0.5

        if q.dtype == torch.float32:
            q = q.half()

        if k.dtype == torch.float32:
            k = k.half()

        if v.dtype == torch.float32:
            v = v.half()

        ring_size = ring_pg.size() if ring_pg else get_world_size()

        cross_attn = q.shape[-3] != k.shape[-3]
        ring_reduce_col &= not cross_attn
        striped_ring_attn &= not cross_attn

        assert (
            k.shape[-1] == v.shape[-1]
        ), "for simplicity when doing ring passing, assume dim_values is equal to dim_queries_keys, majority of transformer do this, not a big issue"

        per_machine_seq_size = k.shape[-3]

        # calculate max ring passes

        max_ring_passes = None
        num_lookback_buckets = float("inf")

        if exists(max_lookback_seq_len):
            assert causal
            assert not (ring_reduce_col and not divisible_by(per_machine_seq_size, bucket_size))

            max_ring_passes = ceil(max_lookback_seq_len / per_machine_seq_size)
            num_lookback_buckets = max_lookback_seq_len // bucket_size

        # ignore key padding mask if autoregressive

        if causal:
            mask = None

        bucket_size = min(per_machine_seq_size, bucket_size)
        per_machine_seq_size // bucket_size

        orig_k, orig_v, orig_mask, q_seq_len, device = k, v, mask, q.shape[1], q.device

        ring_pass_fn = all_ring_pass if ring_reduce_col else null_ring_pass

        kv = torch.stack((k, v))

        # accumulated values

        # o - output
        # m - maximum
        # lse - logsumexp

        o = None
        m = None
        lse = None

        # receive buffers, to be alternated with sent buffer

        receive_kv = None
        receive_mask = None

        # non-causal and causal striped attention can have final normalization of output fused

        can_fuse_final_output_normalization = not causal or (causal and striped_ring_attn)

        for (ring_rank, (is_first, is_last)), ((kv, mask), (receive_kv, receive_mask)) in ring_pass_fn(
            kv,
            mask,
            receive_buffers=(receive_kv, receive_mask),
            max_iters=max_ring_passes,
            ring_size=ring_size,
            ring_pg=ring_pg,
        ):
            k, v = kv

            # account for grouped query attention

            k, v = (repeat(t, "... h d -> ... (g h) d", g=q_head_groups) for t in (k, v))

            # translate key padding mask to bias

            bias = None

            if exists(mask):
                bias = torch.where(mask, 0.0, float("-inf"))

            # for non-striped attention
            # if the kv ring rank is equal to the current rank (block diagonal), then turn on causal
            # for striped attention, it is always causal, but a lt or gt sign needs to be changed to lte or gte within the cuda code, when determining masking out

            block_causal = False
            causal_mask_diagonal = False

            if causal:
                if striped_ring_attn:
                    block_causal = True
                    causal_mask_diagonal = get_rank() if ring_pg is None else dist.get_rank(ring_pg) < ring_rank
                else:
                    block_causal = get_rank() if ring_pg is None else dist.get_rank(ring_pg) == ring_rank

                    if (get_rank() if ring_pg is None else dist.get_rank(ring_pg)) < ring_rank:
                        continue

            o, m, lse = flash_attn_forward(
                q,  # [b, s, nh, d]
                k,
                v,
                causal=block_causal,
                o=o,
                m=m,
                lse=lse,
                bias=bias,
                softmax_scale=softmax_scale,
                causal_mask_diagonal=causal_mask_diagonal,
                return_normalized_output=can_fuse_final_output_normalization and is_last,
                load_accumulated=not is_first,
                softclamp_qk_sim=softclamp_qk_sim,
                softclamp_value=softclamp_value,
            )

        if not can_fuse_final_output_normalization:
            m = m[..., :q_seq_len]

            o_scale = torch.exp(m - lse[..., :q_seq_len])
            o.mul_(rearrange(o_scale, "b h n -> b n h 1"))

        ctx.args = (
            causal,
            softmax_scale,
            orig_mask,
            bucket_size,
            ring_reduce_col,
            max_ring_passes,
            num_lookback_buckets,
            striped_ring_attn,
            q_head_groups,
            softclamp_qk_sim,
            softclamp_value,
            dtype,
            ring_pg,
        )

        ctx.save_for_backward(q, orig_k, orig_v, o, lse)

        # cast back to original dtype

        o = o.type(dtype)
        return o

    @staticmethod
    @torch.no_grad()
    def backward(ctx, do):

        # [FUNCTION DOC]
        # Backward pass without gradient tracking. Restores settings; allocates float32 dq and zero dk, dv; stacks k, v, dk,
        # dv so they travel the ring together. For each pass determines whether to accumulate (striped causal: always;
        # plain causal: only blocks from this rank or earlier; non-causal: always) and the block causality, runs
        # flash_attn_backward under inference mode (reusing delta across passes), sums dk/dv over GQA head groups and adds to
        # the accumulators. When ring_reduce_col is set, passes the gradients along the ring. Finally casts gradients to the
        # original dtype.
        # Inputs: ctx; do (output gradient).
        # Outputs: dq, dk, dv followed by nine None values.
        # Exceptions: AssertionError when k and v dtypes differ, or from flash_attn_backward.
        # Side effects: blocking ring communication and barriers; Triton kernel launches.
        # KNOWN ISSUE: the per-pass ring_dk/ring_dv are accumulated into the local dk/dv, but the stacked kv_and_dkv
        # being passed still carries the stale gradients, so accumulated gradients are not what gets rotated.
        # KNOWN ISSUE: the bias passed to flash_attn_backward is 2-D, while flash_attn_backward asserts a 4-D bias.
        (
            causal,
            softmax_scale,
            mask,
            bucket_size,
            ring_reduce_col,
            max_ring_passes,
            num_lookback_buckets,
            striped_ring_attn,
            q_head_groups,
            softclamp_qk_sim,
            softclamp_value,
            dtype,
            ring_pg,
        ) = ctx.args

        q, k, v, o, lse = ctx.saved_tensors
        ring_size = ring_pg.size() if ring_pg else get_world_size()

        do = do.type(o.dtype)

        device = q.device

        if causal:
            mask = None

        q.shape[-3]

        per_machine_seq_size = k.shape[-3]
        per_machine_seq_size // bucket_size

        ring_pass_fn = all_ring_pass if ring_reduce_col else null_ring_pass

        device = q.device

        dq = torch.zeros(q.shape, device=device, dtype=torch.float32)
        dk = torch.zeros_like(k, device=device)
        dv = torch.zeros_like(v, device=device)

        # k and v will have 16 bits, and dk and dv can also be accumulated safely with the same type, i think

        assert k.dtype == v.dtype

        kv_and_dkv = torch.stack((k, v, dk, dv))

        # receive buffers, to be alternated with sent buffer

        receive_kv_and_dkv = None
        receive_mask = None

        # caching the delta (do * o for backwards pass) across ring reduce

        delta = None

        for (ring_rank, _), ((kv_and_dkv, mask), (receive_kv_and_dkv, receive_mask)) in ring_pass_fn(
            kv_and_dkv,
            mask,
            receive_buffers=(receive_kv_and_dkv, receive_mask),
            max_iters=max_ring_passes,
            ring_size=ring_size,
            ring_pg=ring_pg,
        ):

            k, v, dk, dv = kv_and_dkv

            # account for grouped query attention

            k, v = (repeat(t, "... h d -> ... (g h) d", g=q_head_groups) for t in (k, v))

            # translate key padding mask to bias

            bias = None

            if exists(mask):
                bias = torch.where(mask, 0.0, float("-inf"))
                # bias = rearrange(bias, "b j -> b 1 1 j")

            # determine whether to do causal mask or not
            # depends on whether it is striped attention, as well as current machine rank vs ring rank

            if causal and striped_ring_attn:
                need_accum = True
                block_causal = True
                causal_mask_diagonal = (get_rank() if ring_pg is None else dist.get_rank(ring_pg)) < ring_rank
            elif causal:
                need_accum = (get_rank() if ring_pg is None else dist.get_rank(ring_pg)) >= ring_rank
                block_causal = (get_rank() if ring_pg is None else dist.get_rank(ring_pg)) == ring_rank
                causal_mask_diagonal = False
            else:
                need_accum = True
                block_causal = False
                causal_mask_diagonal = False

            # use flash attention backwards kernel to calculate dq, dk, dv and accumulate

            if need_accum:
                ring_dq = torch.empty(q.shape, device=device, dtype=torch.float32)
                ring_dk = torch.empty_like(k)
                ring_dv = torch.empty_like(v)

                with torch.inference_mode():
                    delta = flash_attn_backward(
                        do,
                        q,
                        k,
                        v,
                        o,
                        lse,
                        ring_dq,
                        ring_dk,
                        ring_dv,
                        delta=delta,
                        bias=bias,
                        causal=block_causal,
                        causal_mask_diagonal=causal_mask_diagonal,
                        softmax_scale=softmax_scale,
                        softclamp_qk_sim=softclamp_qk_sim,
                        softclamp_value=softclamp_value,
                    )

                # account for grouped query attention

                ring_dk = reduce(ring_dk, "... (g h) d -> ... h d", g=q_head_groups, reduction="sum")
                ring_dv = reduce(ring_dv, "... (g h) d -> ... h d", g=q_head_groups, reduction="sum")

                dq.add_(ring_dq)
                dk.add_(ring_dk)
                dv.add_(ring_dv)

            if not ring_reduce_col:
                continue

            dkv = kv_and_dkv[2:]

            max_ring_passes = default(max_ring_passes, ring_size)
            dkv = ring_pass(ring_size - max_ring_passes + 1, dkv, ring_pg=ring_pg)

            dk, dv = dkv

        dq, dk, dv = (t.to(dtype) for t in (dq, dk, dv))

        return dq, dk, dv, None, None, None, None, None, None, None, None, None


ring_flash_attn_cuda_ = RingFlashAttentionCUDAFunction.apply


@autocast("cuda", enabled=False)
def ring_flash_attn_cuda(
    module: torch.nn.Module,
    q: Tensor,
    k: Tensor,
    v: Tensor,
    attention_mask: Tensor | None = None,
    is_causal: bool = False,
    bucket_size: int = 1024,
    ring_reduce_col: bool = False,
    striped_ring_attn: bool = False,
    max_lookback_seq_len: int | None = None,
    softclamp_qk_sim: bool = False,
    softclamp_value: float = 50.0,
    ring_pg: dist.ProcessGroup | None = None,
    **kwargs,
):
    # [FUNCTION DOC]
    # Attention-registry entry point (autocast disabled) applying RingFlashAttentionCUDAFunction.
    # Inputs: module (unused); q, k, v; attention_mask (optional key padding mask); is_causal (False); bucket_size
    # (1024); ring_reduce_col (False); striped_ring_attn (False); max_lookback_seq_len (None); softclamp_qk_sim (False);
    # softclamp_value (50.0); ring_pg (None); other keyword arguments ignored.
    # Outputs: a one-element tuple containing the attention output.
    # Exceptions: see RingFlashAttentionCUDAFunction.forward. Side effects: ring communication.
    # KNOWN ISSUE: returns a 1-tuple rather than the output and weights pair other backends return.
    return (
        ring_flash_attn_cuda_(
            q,
            k,
            v,
            attention_mask,
            is_causal,
            bucket_size,
            ring_reduce_col,
            striped_ring_attn,
            max_lookback_seq_len,
            softclamp_qk_sim,
            softclamp_value,
            ring_pg,
        ),
    )


# ring.py
from collections import namedtuple
from functools import partial

import torch
import torch.distributed as dist
from torch import Tensor
from torch.autograd import Function
from torch.nn import Module

# helper functions


def exists(v):
    # [FUNCTION DOC]
    # HELPER: null check used throughout the module (redefined identically in each concatenated section).
    # Inputs: a value. Outputs: True when it is not None.
    # Exceptions: None explicitly raised. Side effects: None.
    return v is not None


def default(v, d):
    # [FUNCTION DOC]
    # HELPER: returns the first argument unless it is None, in which case returns the fallback (redefined per section).
    # Inputs: value, fallback. Outputs: the chosen value.
    # Exceptions: None explicitly raised. Side effects: None.
    return v if exists(v) else d


def cast_tuple(t, length=1):
    # [FUNCTION DOC]
    # HELPER: wraps a non-tuple value into a tuple repeated length times; tuples pass through.
    # Inputs: t; length (default 1). Outputs: tuple.
    # Exceptions: None explicitly raised. Side effects: None.
    return t if isinstance(t, tuple) else ((t,) * length)


# ring functions


def circular_index_left(pos, ring_size, num=1):
    # [FUNCTION DOC]
    # HELPER: position num steps to the left on a ring of ring_size.
    # Inputs: pos, ring_size, num (default 1). Outputs: int in the range 0 to ring_size minus one.
    # Exceptions: None explicitly raised. Side effects: None.
    return ((pos - num) + ring_size) % ring_size


def circular_index_right(pos, ring_size, num=1):
    # [FUNCTION DOC]
    # HELPER: position num steps to the right on a ring of ring_size.
    # Inputs: pos, ring_size, num (default 1). Outputs: int.
    # Exceptions: None explicitly raised. Side effects: None.
    return (pos + num) % ring_size


# distributed ring


def circular_rank_left(rank=None, ring_size=None, num=1, pg=None):
    # [FUNCTION DOC]
    # HELPER: rank of the left neighbor within this rank's ring (ranks are grouped into consecutive rings of ring_size;
    # the ring offset is preserved).
    # Inputs: rank (default current rank in pg or globally); ring_size (default world size of pg or global); num (1);
    # pg (optional group).
    # Outputs: int rank.
    # Exceptions: None explicitly raised. Side effects: None.
    # KNOWN ISSUE: the result is a rank within pg but is used as a peer for P2P ops that expect global ranks when
    # pg is a subgroup.
    rank = default(rank, get_rank() if pg is None else dist.get_rank(pg))
    ring_size = default(ring_size, get_world_size() if pg is None else dist.get_world_size(pg))
    ring_set_num = rank // ring_size
    offset = ring_set_num * ring_size
    return circular_index_left(rank, ring_size, num) + offset


def circular_rank_right(rank=None, ring_size=None, num=1, pg=None):
    # [FUNCTION DOC]
    # HELPER: rank of the right neighbor within this rank's ring; same conventions as circular_rank_left.
    # Inputs: rank, ring_size, num, pg. Outputs: int rank.
    # Exceptions: None explicitly raised. Side effects: None.
    rank = default(rank, get_rank() if pg is None else dist.get_rank(pg))
    ring_size = default(ring_size, get_world_size() if pg is None else dist.get_world_size(pg))
    ring_set_num = rank // ring_size
    offset = ring_set_num * ring_size
    return circular_index_right(rank, ring_size, num) + offset


# one ring pass


def send_and_receive_(x, receive_buffer, send_to_rank, receive_from_rank, ring_pg=None):
    # [FUNCTION DOC]
    # HELPER: blocking exchange used by ring_pass: isend x to send_to_rank and irecv into receive_buffer from
    # receive_from_rank as one batch, waits for both, then barriers on ring_pg (or the world).
    # Inputs: x; receive_buffer; send_to_rank; receive_from_rank; ring_pg (optional).
    # Outputs: None.
    # Exceptions: errors from torch.distributed.
    # Side effects: P2P communication, barrier, fills receive_buffer.
    send_op = dist.P2POp(dist.isend, x, send_to_rank, ring_pg)
    recv_op = dist.P2POp(dist.irecv, receive_buffer, receive_from_rank, ring_pg)

    reqs = dist.batch_isend_irecv([send_op, recv_op])

    for req in reqs:
        req.wait()

    if ring_pg is not None:
        dist.barrier(ring_pg)
    else:
        dist.barrier()


def ring_pass(
    num_ring_passes: int,
    x: Tensor,
    receive_buffer: Tensor | None = None,
    ring_size: int | None = None,
    ring_pg: dist.ProcessGroup | None = None,
):
    # [FUNCTION DOC]
    # Sends x one step to the right neighbor and receives from the left neighbor.
    # Inputs: num_ring_passes (int); x (tensor, made contiguous); receive_buffer (optional, zeros like x when absent);
    # ring_size (default world size); ring_pg (optional).
    # Outputs: tuple of the received tensor and the sent tensor (so they can be swapped as buffers next time).
    # Exceptions: errors from torch.distributed. Side effects: blocking communication and a barrier.
    # KNOWN ISSUE: num_ring_passes is ignored; exactly one pass is always performed.
    ring_size = default(ring_size, get_world_size())
    x = x.contiguous()

    if not exists(receive_buffer):
        receive_buffer = torch.zeros_like(x)
    else:
        receive_buffer = receive_buffer.contiguous()

    send_and_receive_(
        x,
        receive_buffer,
        circular_rank_right(ring_size=ring_size, pg=ring_pg),
        circular_rank_left(ring_size=ring_size, pg=ring_pg),
        ring_pg=ring_pg,
    )
    return receive_buffer, x


one_ring_pass = partial(ring_pass, 1)

# iterator for all ring passes of all tensors

RingInfo = namedtuple("RingInfo", ["ring_rank", "iter_info"])


def null_ring_pass(*tensors, max_iters=None, receive_buffers=None, ring_size=None, ring_pg=None):
    # [FUNCTION DOC]
    # Generator yielding a single iteration with ring rank 0, flags first and last both true, and the given tensors and
    # receive buffers unchanged (no communication).
    # Inputs: tensors; max_iters, receive_buffers, ring_size, ring_pg (unused except receive_buffers echoed).
    # Outputs: yields one pair of RingInfo and (tensors, receive_buffers).
    # Exceptions: None explicitly raised. Side effects: None.
    # KNOWN ISSUE: the reported ring rank is always 0 instead of the local rank, so causal callers on ranks above 0 treat
    # their own chunk as off-diagonal.
    yield RingInfo(0, (True, True)), (tensors, receive_buffers)


def all_ring_pass(*tensors, max_iters=None, receive_buffers=None, ring_size=None, ring_pg=None):
    # [FUNCTION DOC]
    # Generator over ring passes. Runs between 1 and ring_size iterations (capped by max_iters). Each iteration yields
    # RingInfo with the rank whose chunk is currently held (starting at the local rank, moving left each step) and
    # first/last flags, plus the tensors and receive buffers; between iterations (not after the last) every non-None
    # tensor is passed one step along the ring with ring_pass, swapping send and receive buffers.
    # Inputs: tensors (any may be None); max_iters (default ring_size); receive_buffers (single value or tuple);
    # ring_size (default world size); ring_pg (optional).
    # Outputs: yields pairs of RingInfo and (tensors, receive_buffers).
    # Exceptions: None explicitly raised. Side effects: blocking ring communication.
    ring_size = default(ring_size, get_world_size())
    max_iters = default(max_iters, ring_size)

    receive_buffers = cast_tuple(receive_buffers, len(tensors))

    # make sure iteration is between 1 and world size
    total_iters = max(1, min(ring_size, max_iters))

    curr_ring_pos = get_rank() if ring_pg is None else dist.get_rank(ring_pg)

    for ind in range(total_iters):
        is_first = ind == 0
        is_last = ind == (total_iters - 1)

        yield RingInfo(curr_ring_pos, (is_first, is_last)), (tensors, receive_buffers)

        curr_ring_pos = circular_index_left(curr_ring_pos, ring_size)

        if is_last:
            continue

        new_tensors = []
        new_receive_buffers = []

        for tensor, receive_buffer in zip(tensors, receive_buffers):
            if exists(tensor):
                new_tensor, new_receive_buffer = ring_pass(
                    1, tensor, receive_buffer, ring_size=ring_size, ring_pg=ring_pg
                )
            else:
                new_tensor, new_receive_buffer = None, None

            new_tensors.append(new_tensor)
            new_receive_buffers.append(new_receive_buffer)

        tensors = new_tensors
        receive_buffers = new_receive_buffers


# distributed.py
from functools import lru_cache, partial

import torch
import torch.distributed as dist
import torch.nn.functional as F
from einops import rearrange
from torch.autograd import Function
from torch.nn import Module


def exists(val):
    # [FUNCTION DOC]
    # HELPER: null check used throughout the module (redefined identically in each concatenated section).
    # Inputs: a value. Outputs: True when it is not None.
    # Exceptions: None explicitly raised. Side effects: None.
    return val is not None


def default(val, d):
    # [FUNCTION DOC]
    # HELPER: returns the first argument unless it is None, in which case returns the fallback (redefined per section).
    # Inputs: value, fallback. Outputs: the chosen value.
    # Exceptions: None explicitly raised. Side effects: None.
    return val if exists(val) else d


def divisible_by(num, den):
    # [FUNCTION DOC]
    # HELPER: divisibility test (redefined per section).
    # Inputs: num, den (ints). Outputs: True when num modulo den is zero.
    # Exceptions: ZeroDivisionError when den is zero. Side effects: None.
    return (num % den) == 0


def pad_dim_to(t, length, dim=0):
    # [FUNCTION DOC]
    # HELPER: right-pads tensor t with zeros along dim up to length.
    # Inputs: t; length (int, at least the current size); dim (default 0, negative allowed).
    # Outputs: padded tensor.
    # Exceptions: None explicitly raised. Side effects: None.
    pad_length = length - t.shape[dim]
    zero_pairs = (-dim - 1) if dim < 0 else (t.ndim - dim - 1)
    return F.pad(t, (*((0, 0) * zero_pairs), 0, pad_length))


cache = partial(lru_cache, maxsize=None)

# distributed helpers


@cache()
def get_rank():
    # [FUNCTION DOC]
    # Global rank, or 0 when torch.distributed is not initialized. Cached for the process lifetime.
    # Inputs: none. Outputs: int. Exceptions: None explicitly raised. Side effects: caches the first result.
    return dist.get_rank() if dist.is_initialized() else 0


@cache()
def get_world_size():
    # [FUNCTION DOC]
    # Global world size, or 1 when torch.distributed is not initialized. Cached.
    # Inputs: none. Outputs: int. Exceptions: None explicitly raised. Side effects: caches the first result.
    return dist.get_world_size() if dist.is_initialized() else 1


@cache()
def is_distributed():
    # [FUNCTION DOC]
    # True when torch.distributed is initialized with more than one process. Cached.
    # Inputs: none. Outputs: bool. Exceptions: None explicitly raised. Side effects: caches the first result.
    return dist.is_initialized() and dist.get_world_size() > 1


def all_gather_same_dim(t):
    # [FUNCTION DOC]
    # HELPER: all-gathers a same-shaped tensor from every rank of the default group.
    # Inputs: t (made contiguous). Outputs: list of world_size tensors.
    # Exceptions: errors from torch.distributed. Side effects: all_gather collective.
    t = t.contiguous()
    world_size = dist.get_world_size()
    gathered_tensors = [torch.empty_like(t, device=t.device, dtype=t.dtype) for i in range(world_size)]
    dist.all_gather(gathered_tensors, t)
    return gathered_tensors


def gather_sizes(t, *, dim):
    # [FUNCTION DOC]
    # HELPER: all-gathers the size of dimension dim of t from every rank.
    # Inputs: t; dim (keyword). Outputs: long tensor [world_size].
    # Exceptions: None explicitly raised. Side effects: all_gather collective.
    size = torch.tensor(t.shape[dim], device=t.device, dtype=torch.long)
    sizes = all_gather_same_dim(size)
    return torch.stack(sizes)


def has_only_one_value(t):
    # [FUNCTION DOC]
    # HELPER: True (as a boolean tensor) when all elements equal the first element.
    # Inputs: t (1-D tensor). Outputs: boolean scalar tensor.
    # Exceptions: None explicitly raised. Side effects: None.
    return (t == t[0]).all()


def all_gather_variable_dim(t, dim=0, sizes=None):
    # [FUNCTION DOC]
    # All-gathers tensors whose size along dim may differ per rank. Gathers sizes unless given; if all equal, gathers and
    # concatenates directly; otherwise pads every tensor to the maximum size, gathers, concatenates and removes padding
    # via a validity mask and index_select.
    # Inputs: t; dim (default 0); sizes (optional long tensor of per-rank sizes).
    # Outputs: tuple of concatenated tensor and sizes.
    # Exceptions: None explicitly raised. Side effects: collectives on the default group; host sync via item.
    device, rank, world_size = t.device, dist.get_rank(), dist.get_world_size()

    if not exists(sizes):
        sizes = gather_sizes(t, dim=dim)

    if has_only_one_value(sizes):
        gathered_tensors = all_gather_same_dim(t)
        gathered_tensors = torch.cat(gathered_tensors, dim=dim)
        return gathered_tensors, sizes

    max_size = sizes.amax().item()

    padded_t = pad_dim_to(t, max_size, dim=dim)
    gathered_tensors = all_gather_same_dim(padded_t)

    gathered_tensors = torch.cat(gathered_tensors, dim=dim)
    seq = torch.arange(max_size, device=device)

    mask = seq[None, :] < sizes[:, None]
    mask = rearrange(mask, "i j -> (i j)")
    seq = torch.arange(mask.shape[-1], device=device)
    indices = seq[mask]

    gathered_tensors = gathered_tensors.index_select(dim, indices)

    return gathered_tensors, sizes


class AllGatherFunction(Function):
    # [CLASS DOC]
    # Autograd function for variable-size all-gather; backward returns the slice of the gradient belonging to this rank.
    # ctx stores batch_sizes (list of ints) and dim.
    @staticmethod
    def forward(ctx, x, dim, sizes):
        # [FUNCTION DOC]
        # Casts boolean input to int for communication, gathers with all_gather_variable_dim, records sizes and dim, and
        # casts back to boolean if needed.
        # Inputs: ctx; x; dim; sizes (optional).
        # Outputs: tuple of gathered tensor and sizes tensor.
        # Exceptions: None explicitly raised. Side effects: collectives; host sync for tolist.
        is_bool = x.dtype == torch.bool

        if is_bool:
            x = x.int()

        x, batch_sizes = all_gather_variable_dim(x, dim=dim, sizes=sizes)
        ctx.batch_sizes = batch_sizes.tolist()
        ctx.dim = dim

        if is_bool:
            x = x.bool()

        return x, batch_sizes

    @staticmethod
    def backward(ctx, grads, _):
        # [FUNCTION DOC]
        # Splits the incoming gradient by recorded sizes along dim and returns this rank's piece.
        # Inputs: ctx; grads; second gradient ignored.
        # Outputs: the gradient slice, None, None.
        # Exceptions: None explicitly raised. Side effects: None.
        batch_sizes, rank = ctx.batch_sizes, dist.get_rank()
        grads_by_rank = grads.split(batch_sizes, dim=ctx.dim)
        return grads_by_rank[rank], None, None


class AllGather(Module):
    # [CLASS DOC]
    # torch.nn.Module wrapper over AllGatherFunction with a fixed gather dimension (attribute dim).
    def __init__(self, *, dim=0):
        # [FUNCTION DOC]
        # Stores dim (keyword-only, default 0).
        # Inputs: dim. Outputs: None. Exceptions: None explicitly raised. Side effects: None.
        super().__init__()
        self.dim = dim

    def forward(self, x, sizes=None):
        # [FUNCTION DOC]
        # Applies AllGatherFunction along self.dim.
        # Inputs: x; sizes (optional). Outputs: tuple of gathered tensor and sizes.
        # Exceptions: None explicitly raised. Side effects: collectives.
        return AllGatherFunction.apply(x, self.dim, sizes)


def split_by_rank(x):
    # [FUNCTION DOC]
    # Picks this rank's element from x (a tuple of tensors or a stacked tensor) and builds a sizes tensor: lengths of
    # dimension 0 of each tuple element, or dimension 1 repeated for a stacked tensor.
    # Inputs: x. Outputs: tuple of the local piece and a long sizes tensor on its device.
    # Exceptions: IndexError if fewer entries than ranks. Side effects: None.
    rank = dist.get_rank()
    out = x[rank]

    if isinstance(x, tuple):
        sizes = tuple((t.shape[0] for t in x))
    else:
        sizes = (x.shape[1],) * x.shape[0]

    sizes = torch.tensor(sizes, device=out.device, dtype=torch.long)
    return out, sizes


all_gather = AllGatherFunction.apply

# triton flash attention
# taken from https://github.com/Dao-AILab/flash-attention/blob/main/flash_attn/flash_attn_triton.py
# with fixes for triton 2.3
# forward is modified to return unnormalized accumulation, row maxes, row lse - reduced over passed rings
# both forwards and backwards is modified to allow for masking out the diagonal for striped ring attention

from math import ceil

import torch
from einops import rearrange, repeat
from torch import Tensor


def exists(v):
    # [FUNCTION DOC]
    # HELPER: null check used throughout the module (redefined identically in each concatenated section).
    # Inputs: a value. Outputs: True when it is not None.
    # Exceptions: None explicitly raised. Side effects: None.
    return v is not None


def default(val, d):
    # [FUNCTION DOC]
    # HELPER: returns the first argument unless it is None, in which case returns the fallback (redefined per section).
    # Inputs: value, fallback. Outputs: the chosen value.
    # Exceptions: None explicitly raised. Side effects: None.
    return val if exists(val) else d


def is_contiguous(x: Tensor):
    # [FUNCTION DOC]
    # HELPER: True when the last dimension has stride 1 (what the Triton kernels require).
    # Inputs: x tensor. Outputs: bool. Exceptions: None explicitly raised. Side effects: None.
    return x.stride(-1) == 1


INSTALL_COMMAND = "pip install -U --index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/Triton-Nightly/pypi/simple/ triton-nightly"

# make sure triton 2.1+ is installed

from importlib.metadata import version

import packaging.version as pkg_version

try:
    triton_version = version("triton-nightly")
except:
    print(f"latest triton must be installed. `{INSTALL_COMMAND}` first")
    exit()

assert pkg_version.parse(triton_version) >= pkg_version.parse(
    "3.0.0"
), f"triton must be version 3.0.0 or above. `{INSTALL_COMMAND}` to upgrade"

import triton
import triton.language as tl
from triton.language.extra import libdevice

# kernels


@triton.heuristics(
    {
        "EVEN_M": lambda args: args["seqlen_q"] % args["BLOCK_M"] == 0,
        "EVEN_N": lambda args: args["seqlen_k"] % args["BLOCK_N"] == 0,
        "EVEN_HEADDIM": lambda args: args["headdim"] == args["BLOCK_HEADDIM"],
    }
)
@triton.jit
def _fwd_kernel(
    Q,
    K,
    V,
    Bias,
    Out,
    M,
    Lse,
    softmax_scale,
    stride_qb,
    stride_qh,
    stride_qm,
    stride_kb,
    stride_kh,
    stride_kn,
    stride_vb,
    stride_vh,
    stride_vn,
    stride_bb,
    stride_bh,
    stride_bm,
    stride_ob,
    stride_oh,
    stride_om,
    nheads,
    seqlen_q,
    seqlen_k,
    seqlen_q_rounded,
    headdim,
    CACHE_KEY_SEQLEN_Q,
    CACHE_KEY_SEQLEN_K,
    HAS_BIAS: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    CAUSAL_MASK_DIAGONAL: tl.constexpr,
    LOAD_ACCUMULATED: tl.constexpr,
    RETURN_NORMALIZED_OUTPUT: tl.constexpr,
    SOFTCLAMP_QK_SIM: tl.constexpr,
    SOFTCLAMP_VALUE: tl.constexpr,
    BLOCK_HEADDIM: tl.constexpr,
    EVEN_M: tl.constexpr,
    EVEN_N: tl.constexpr,
    EVEN_HEADDIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    # [FUNCTION DOC]
    # Triton forward FlashAttention kernel, one program per (query block, batch-head). Loads previously accumulated
    # output, row max m and lse when LOAD_ACCUMULATED, otherwise starts from zero and minus infinity. Iterates over key
    # blocks (up to the diagonal when causal), computes scaled scores, optionally softclamps with tanh, masks
    # out-of-range keys and causal positions (strictly-below-diagonal when CAUSAL_MASK_DIAGONAL), adds bias, and
    # updates the unnormalized accumulator, running max and log-sum-exp online. Optionally normalizes the output; stores
    # lse, m (when not normalized) and output with boundary masks.
    # Inputs: tensor pointers Q, K, V, Bias, Out, M, Lse; softmax_scale; strides; sizes; cache keys; compile-time flags
    # and block sizes (EVEN_M, EVEN_N and EVEN_HEADDIM set by heuristics).
    # Outputs: None. Exceptions: None explicitly raised. Side effects: writes Out, M, Lse.
    # KNOWN ISSUE: the running maximum is initialized from and compared against lse_i rather than m_i, mixing the two
    # statistics (as in the upstream adaptation); it is also never launched, since flash_attn_forward has the launch
    # commented out.
    start_m = tl.program_id(0)
    off_hb = tl.program_id(1)
    off_b = off_hb // nheads
    off_h = off_hb % nheads

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_HEADDIM)

    q_ptrs = Q + off_b * stride_qb + off_h * stride_qh + (offs_m[:, None] * stride_qm + offs_d[None, :])
    k_ptrs = K + off_b * stride_kb + off_h * stride_kh + (offs_n[:, None] * stride_kn + offs_d[None, :])
    v_ptrs = V + off_b * stride_vb + off_h * stride_vh + (offs_n[:, None] * stride_vn + offs_d[None, :])

    if HAS_BIAS:
        b_ptrs = Bias + off_b * stride_bb + off_h * stride_bh + offs_n

    # maximum

    m_ptrs = M + off_hb * seqlen_q_rounded + offs_m

    if LOAD_ACCUMULATED:
        m_i = tl.load(m_ptrs)
    else:
        m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")

    # load lse

    lse_ptrs = Lse + off_hb * seqlen_q_rounded + offs_m

    if LOAD_ACCUMULATED:
        lse_i = tl.load(lse_ptrs)
    else:
        lse_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")

    # load accumualted output

    offs_d = tl.arange(0, BLOCK_HEADDIM)

    out_ptrs = Out + off_b * stride_ob + off_h * stride_oh + (offs_m[:, None] * stride_om + offs_d[None, :])

    if LOAD_ACCUMULATED:
        if EVEN_M:
            if EVEN_HEADDIM:
                acc_o = tl.load(out_ptrs)
            else:
                acc_o = tl.load(out_ptrs, mask=offs_d[None, :] < headdim)
        else:
            if EVEN_HEADDIM:
                acc_o = tl.load(out_ptrs, mask=offs_m[:, None] < seqlen_q)
            else:
                acc_o = tl.load(out_ptrs, mask=(offs_m[:, None] < seqlen_q) & (offs_d[None, :] < headdim))

        acc_o = acc_o.to(tl.float32)
    else:
        acc_o = tl.zeros([BLOCK_M, BLOCK_HEADDIM], dtype=tl.float32)

    # load queries, keys, values

    if EVEN_M & EVEN_N:
        if EVEN_HEADDIM:
            q = tl.load(q_ptrs)
        else:
            q = tl.load(q_ptrs, mask=offs_d[None, :] < headdim, other=0.0)
    else:
        if EVEN_HEADDIM:
            q = tl.load(q_ptrs, mask=offs_m[:, None] < seqlen_q, other=0.0)
        else:
            q = tl.load(q_ptrs, mask=(offs_m[:, None] < seqlen_q) & (offs_d[None, :] < headdim), other=0.0)

    end_n = seqlen_k if not IS_CAUSAL else tl.minimum((start_m + 1) * BLOCK_M, seqlen_k)
    for start_n in range(0, end_n, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)

        if EVEN_N & EVEN_M:
            if EVEN_HEADDIM:
                k = tl.load(k_ptrs + start_n * stride_kn)
            else:
                k = tl.load(k_ptrs + start_n * stride_kn, mask=offs_d[None, :] < headdim, other=0.0)
        else:
            if EVEN_HEADDIM:
                k = tl.load(
                    k_ptrs + start_n * stride_kn,
                    mask=(start_n + offs_n)[:, None] < seqlen_k,
                    other=0.0,
                )
            else:
                k = tl.load(
                    k_ptrs + start_n * stride_kn,
                    mask=((start_n + offs_n)[:, None] < seqlen_k) & (offs_d[None, :] < headdim),
                    other=0.0,
                )
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk += tl.dot(q, tl.trans(k))

        if SOFTCLAMP_QK_SIM:
            effective_softclamp_value = SOFTCLAMP_VALUE / softmax_scale
            qk /= effective_softclamp_value
            qk = libdevice.tanh(qk)
            qk *= effective_softclamp_value

        if not EVEN_N:
            qk += tl.where((start_n + offs_n)[None, :] < seqlen_k, 0, float("-inf"))

        if IS_CAUSAL:
            if CAUSAL_MASK_DIAGONAL:
                # needed for stripe attention
                qk += tl.where(offs_m[:, None] > (start_n + offs_n)[None, :], 0, float("-inf"))
            else:
                qk += tl.where(offs_m[:, None] >= (start_n + offs_n)[None, :], 0, float("-inf"))

        if HAS_BIAS:
            if EVEN_N:
                bias = tl.load(b_ptrs + start_n)
            else:
                bias = tl.load(b_ptrs + start_n, mask=(start_n + offs_n) < seqlen_k, other=0.0)
            bias = bias[None, :]

            bias = bias.to(tl.float32)
            qk = qk * softmax_scale + bias
            m_ij = tl.maximum(tl.max(qk, 1), lse_i)
            p = tl.exp(qk - m_ij[:, None])
        else:
            m_ij = tl.maximum(tl.max(qk, 1) * softmax_scale, lse_i)
            p = tl.exp(qk * softmax_scale - m_ij[:, None])

        l_ij = tl.sum(p, 1)

        acc_o_scale = tl.exp(m_i - m_ij)
        acc_o = acc_o * acc_o_scale[:, None]

        if EVEN_N & EVEN_M:
            if EVEN_HEADDIM:
                v = tl.load(v_ptrs + start_n * stride_vn)
            else:
                v = tl.load(v_ptrs + start_n * stride_vn, mask=offs_d[None, :] < headdim, other=0.0)
        else:
            if EVEN_HEADDIM:
                v = tl.load(
                    v_ptrs + start_n * stride_vn,
                    mask=(start_n + offs_n)[:, None] < seqlen_k,
                    other=0.0,
                )
            else:
                v = tl.load(
                    v_ptrs + start_n * stride_vn,
                    mask=((start_n + offs_n)[:, None] < seqlen_k) & (offs_d[None, :] < headdim),
                    other=0.0,
                )

        p = p.to(v.dtype)
        acc_o += tl.dot(p, v)

        # -- update statistics

        m_i = m_ij
        l_i_new = tl.exp(lse_i - m_ij) + l_ij
        lse_i = m_ij + tl.log(l_i_new)

    if RETURN_NORMALIZED_OUTPUT:
        acc_o_scale = tl.exp(m_i - lse_i)
        acc_o = acc_o * acc_o_scale[:, None]

    # offsets for m and lse

    start_m = tl.program_id(0)
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)

    # write back lse and m

    tl.store(lse_ptrs, lse_i)

    if not RETURN_NORMALIZED_OUTPUT:
        tl.store(m_ptrs, m_i)

    # write to output

    if EVEN_M:
        if EVEN_HEADDIM:
            tl.store(out_ptrs, acc_o)
        else:
            tl.store(out_ptrs, acc_o, mask=offs_d[None, :] < headdim)
    else:
        if EVEN_HEADDIM:
            tl.store(out_ptrs, acc_o, mask=offs_m[:, None] < seqlen_q)
        else:
            tl.store(out_ptrs, acc_o, mask=(offs_m[:, None] < seqlen_q) & (offs_d[None, :] < headdim))


def flash_attn_forward(
    q,  # [b, s, nh, d]
    k,
    v,
    bias=None,
    causal=False,
    o=None,
    m=None,
    lse=None,
    softmax_scale=None,
    causal_mask_diagonal=False,
    return_normalized_output=False,
    load_accumulated=True,
    softclamp_qk_sim=False,
    softclamp_value=50.0,
    head_first_dim=False,
    remove_padding=False,
):
    # [FUNCTION DOC]
    # Python launcher for the forward kernel. Makes inputs contiguous, optionally converts head-first layout, validates
    # shapes ([batch, seq, heads, head_dim]), head_dim at most 128, fp16/bf16 CUDA tensors, and an optional bias (2-D
    # key bias expanded to [batch, heads, seq_q, seq_k]). Allocates m and lse [batch, heads, seq_q rounded up to 128]
    # (filled with the most negative float when load_accumulated, else uninitialized) and o (zeros or uninitialized)
    # when not provided; converts back to head-first and trims padding if requested.
    # Inputs: q, k, v; bias (optional); causal; o, m, lse (optional accumulators); softmax_scale (default one over
    # sqrt(d)); causal_mask_diagonal; return_normalized_output; load_accumulated (default True); softclamp_qk_sim;
    # softclamp_value (50.0); head_first_dim (False); remove_padding (False).
    # Outputs: tuple of o, m, lse.
    # Exceptions: AssertionError on invalid shapes, dtypes or devices.
    # Side effects: allocates CUDA tensors.
    # KNOWN ISSUE: the kernel launch is commented out (TODO), so the returned tensors are just the initialized or
    # passed-in buffers and no attention is computed; BLOCK_HEADDIM and bias strides are computed and discarded.
    q, k, v = [x if is_contiguous(x) else x.contiguous() for x in (q, k, v)]

    if head_first_dim:
        q, k, v = tuple(rearrange(t, "b h n d -> b n h d") for t in (q, k, v))

        if exists(o):
            o = rearrange(o, "b h n d -> b n h d")

    batch, seqlen_q, nheads, d = q.shape
    _, seqlen_k, _, _ = k.shape

    assert k.shape == (batch, seqlen_k, nheads, d)
    assert v.shape == (batch, seqlen_k, nheads, d)
    assert d <= 128, "FlashAttention only support head dimensions up to 128"
    assert q.dtype == k.dtype == v.dtype, "All tensors must have the same type"
    assert q.dtype in [torch.float16, torch.bfloat16], "Only support fp16 and bf16"
    assert q.is_cuda and k.is_cuda and v.is_cuda

    softmax_scale = default(softmax_scale, d**-0.5)

    has_bias = exists(bias)

    if has_bias:
        assert bias.dtype in [q.dtype, torch.float]
        assert bias.is_cuda

        if bias.ndim == 2:
            bias = repeat(bias, "b j -> b h i j", h=nheads, i=seqlen_q)

        if not is_contiguous(bias):
            bias = bias.contiguous()

        assert bias.shape[-2:] == (seqlen_q, seqlen_k)
        bias = bias.expand(batch, nheads, seqlen_q, seqlen_k)

    (bias.stride(0), bias.stride(1), bias.stride(2)) if has_bias else (0, 0, 0)

    seqlen_q_rounded = ceil(seqlen_q / 128) * 128

    if not exists(lse):
        max_neg_value = -torch.finfo(torch.float32).max
        init_fn = partial(torch.full, fill_value=max_neg_value) if load_accumulated else torch.empty
        lse = init_fn((batch, nheads, seqlen_q_rounded), device=q.device, dtype=torch.float32)

    if not exists(m):
        max_neg_value = -torch.finfo(torch.float32).max
        init_fn = partial(torch.full, fill_value=max_neg_value) if load_accumulated else torch.empty
        m = init_fn((batch, nheads, seqlen_q_rounded), device=q.device, dtype=torch.float32)

    if not exists(o):
        init_fn = torch.zeros_like if load_accumulated else torch.empty_like
        o = init_fn(q)

    max(triton.next_power_of_2(d), 16)

    def grid(META):
        # [FUNCTION DOC]
        # HELPER: launch grid for the (disabled) forward kernel: ceil of seqlen_q over BLOCK_M by batch times heads.
        # Inputs: META. Outputs: tuple of two ints. Exceptions: None explicitly raised. Side effects: None.
        return triton.cdiv(seqlen_q, META["BLOCK_M"]), batch * nheads

    # TODO @nouamane: what's up with triton?
    # _fwd_kernel[grid](
    #     q,
    #     k,
    #     v,
    #     bias,
    #     o,
    #     m,
    #     lse,
    #     softmax_scale,
    #     q.stride(0),
    #     q.stride(2),
    #     q.stride(1),
    #     k.stride(0),
    #     k.stride(2),
    #     k.stride(1),
    #     v.stride(0),
    #     v.stride(2),
    #     v.stride(1),
    #     *bias_strides,
    #     o.stride(0),
    #     o.stride(2),
    #     o.stride(1),
    #     nheads,
    #     seqlen_q,
    #     seqlen_k,
    #     seqlen_q_rounded,
    #     d,
    #     seqlen_q // 32,
    #     seqlen_k // 32,
    #     has_bias,
    #     causal,
    #     causal_mask_diagonal,
    #     load_accumulated,
    #     return_normalized_output,
    #     softclamp_qk_sim,
    #     softclamp_value,
    #     BLOCK_HEADDIM,
    #     BLOCK_M=BLOCK,
    #     BLOCK_N=BLOCK,
    #     num_warps=num_warps,
    #     num_stages=1,
    # )

    if head_first_dim:
        o = rearrange(o, "b n h d -> b h n d")

    if remove_padding:
        m = m[..., :seqlen_q]
        lse = lse[..., :seqlen_q]

    return o, m, lse


@triton.jit
def _bwd_preprocess_do_o_dot(
    Out,
    DO,
    Delta,
    stride_ob,
    stride_oh,
    stride_om,
    stride_dob,
    stride_doh,
    stride_dom,
    nheads,
    seqlen_q,
    seqlen_q_rounded,
    headdim,
    BLOCK_M: tl.constexpr,
    BLOCK_HEADDIM: tl.constexpr,
):
    # [FUNCTION DOC]
    # Triton kernel computing delta, the per-row dot product of output and output gradient, for each (query block,
    # batch-head) program, storing into Delta at the rounded-length layout.
    # Inputs: Out, DO, Delta pointers; strides; nheads; seqlen_q; seqlen_q_rounded; headdim; BLOCK_M; BLOCK_HEADDIM.
    # Outputs: None. Exceptions: None explicitly raised. Side effects: writes Delta.
    start_m = tl.program_id(0)
    off_hb = tl.program_id(1)
    off_b = off_hb // nheads
    off_h = off_hb % nheads
    # initialize offsets
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_HEADDIM)
    # load
    o = tl.load(
        Out + off_b * stride_ob + off_h * stride_oh + offs_m[:, None] * stride_om + offs_d[None, :],
        mask=(offs_m[:, None] < seqlen_q) & (offs_d[None, :] < headdim),
        other=0.0,
    ).to(tl.float32)
    do = tl.load(
        DO + off_b * stride_dob + off_h * stride_doh + offs_m[:, None] * stride_dom + offs_d[None, :],
        mask=(offs_m[:, None] < seqlen_q) & (offs_d[None, :] < headdim),
        other=0.0,
    ).to(tl.float32)
    delta = tl.sum(o * do, axis=1)
    # write-back
    tl.store(Delta + off_hb * seqlen_q_rounded + offs_m, delta)


@triton.jit
def _bwd_store_dk_dv(
    dk_ptrs,
    dv_ptrs,
    dk,
    dv,
    offs_n,
    offs_d,
    seqlen_k,
    headdim,
    EVEN_M: tl.constexpr,
    EVEN_N: tl.constexpr,
    EVEN_HEADDIM: tl.constexpr,
):
    # [2022-11-01] TD: Same bug. In the case of EVEN_N=True and EVEN_M=False,
    # if we just call tl.store(dv_ptrs), there's a race condition
    # [FUNCTION DOC]
    # HELPER: Triton device function storing dk and dv blocks with the boundary masks implied by the EVEN flags.
    # Inputs: dk_ptrs, dv_ptrs, dk, dv, offs_n, offs_d, seqlen_k, headdim, EVEN_M, EVEN_N, EVEN_HEADDIM.
    # Outputs: None. Exceptions: None explicitly raised. Side effects: writes DK, DV.
    if EVEN_N & EVEN_M:
        if EVEN_HEADDIM:
            tl.store(dv_ptrs, dv)
            tl.store(dk_ptrs, dk)
        else:
            tl.store(dv_ptrs, dv, mask=offs_d[None, :] < headdim)
            tl.store(dk_ptrs, dk, mask=offs_d[None, :] < headdim)
    else:
        if EVEN_HEADDIM:
            tl.store(dv_ptrs, dv, mask=offs_n[:, None] < seqlen_k)
            tl.store(dk_ptrs, dk, mask=offs_n[:, None] < seqlen_k)
        else:
            tl.store(dv_ptrs, dv, mask=(offs_n[:, None] < seqlen_k) & (offs_d[None, :] < headdim))
            tl.store(dk_ptrs, dk, mask=(offs_n[:, None] < seqlen_k) & (offs_d[None, :] < headdim))


@triton.jit
def _bwd_kernel_one_col_block(
    start_n,
    Q,
    K,
    V,
    Bias,
    DO,
    DQ,
    DK,
    DV,
    LSE,
    D,
    softmax_scale,
    stride_qm,
    stride_kn,
    stride_vn,
    stride_bm,
    stride_dom,
    stride_dqm,
    stride_dkn,
    stride_dvn,
    seqlen_q,
    seqlen_k,
    headdim,
    ATOMIC_ADD: tl.constexpr,
    BIAS_TYPE: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    CAUSAL_MASK_DIAGONAL: tl.constexpr,
    SOFTCLAMP_QK_SIM: tl.constexpr,
    SOFTCLAMP_VALUE: tl.constexpr,
    BLOCK_HEADDIM: tl.constexpr,
    EVEN_M: tl.constexpr,
    EVEN_N: tl.constexpr,
    EVEN_HEADDIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    # We need to make sure begin_m is a multiple of BLOCK_M (not BLOCK_N)
    # [FUNCTION DOC]
    # HELPER: Triton device function computing gradients for one key block (column) in _bwd_kernel.
    # Loads the key/value block, loops over query blocks (from the diagonal when causal), recomputes scores with the same
    # softclamp/mask/bias logic as forward, probabilities from the saved lse, accumulates dv and dk, and updates dq
    # either by load-add-store or atomic add (sequence-parallel mode); the softclamp derivative scales ds. Early-exits
    # storing zeros when the block starts past the query length.
    # Inputs: start_n; tensor pointers; strides; sizes; compile-time flags (ATOMIC_ADD, BIAS_TYPE "none", "vector" or
    # "matrix", IS_CAUSAL, CAUSAL_MASK_DIAGONAL, SOFTCLAMP settings, EVEN flags, block sizes).
    # Outputs: None. Exceptions: None explicitly raised. Side effects: writes DQ, DK, DV.
    begin_m = 0 if not IS_CAUSAL else ((start_n * BLOCK_N) // BLOCK_M) * BLOCK_M
    # initialize row/col offsets
    offs_qm = begin_m + tl.arange(0, BLOCK_M)
    offs_n = start_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_m = tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_HEADDIM)
    # initialize pointers to value-like data
    q_ptrs = Q + (offs_qm[:, None] * stride_qm + offs_d[None, :])
    k_ptrs = K + (offs_n[:, None] * stride_kn + offs_d[None, :])
    v_ptrs = V + (offs_n[:, None] * stride_vn + offs_d[None, :])
    do_ptrs = DO + (offs_qm[:, None] * stride_dom + offs_d[None, :])
    dq_ptrs = DQ + (offs_qm[:, None] * stride_dqm + offs_d[None, :])
    if BIAS_TYPE == "vector":
        b_ptrs = Bias + offs_n
    elif BIAS_TYPE == "matrix":
        b_ptrs = Bias + (offs_qm[:, None] * stride_bm + offs_n[None, :])
    # initialize dv and dk
    dv = tl.zeros([BLOCK_N, BLOCK_HEADDIM], dtype=tl.float32)
    dk = tl.zeros([BLOCK_N, BLOCK_HEADDIM], dtype=tl.float32)
    # There seems to be some problem with Triton pipelining that makes results wrong for
    # headdim=64, seqlen=(113, 255), bias_type='matrix'. In this case the for loop
    # may have zero step, and pipelining with the bias matrix could screw it up.
    # So we just exit early.
    if begin_m >= seqlen_q:
        dv_ptrs = DV + (offs_n[:, None] * stride_dvn + offs_d[None, :])
        dk_ptrs = DK + (offs_n[:, None] * stride_dkn + offs_d[None, :])
        _bwd_store_dk_dv(
            dk_ptrs,
            dv_ptrs,
            dk,
            dv,
            offs_n,
            offs_d,
            seqlen_k,
            headdim,
            EVEN_M=EVEN_M,
            EVEN_N=EVEN_N,
            EVEN_HEADDIM=EVEN_HEADDIM,
        )
        return
    # k and v stay in SRAM throughout
    # [2022-10-30] TD: Same bug as the fwd. In the case of EVEN_N=True and EVEN_M=False,
    # if we just call tl.load(k_ptrs), we get the wrong output!
    if EVEN_N & EVEN_M:
        if EVEN_HEADDIM:
            k = tl.load(k_ptrs)
            v = tl.load(v_ptrs)
        else:
            k = tl.load(k_ptrs, mask=offs_d[None, :] < headdim, other=0.0)
            v = tl.load(v_ptrs, mask=offs_d[None, :] < headdim, other=0.0)
    else:
        if EVEN_HEADDIM:
            k = tl.load(k_ptrs, mask=offs_n[:, None] < seqlen_k, other=0.0)
            v = tl.load(v_ptrs, mask=offs_n[:, None] < seqlen_k, other=0.0)
        else:
            k = tl.load(k_ptrs, mask=(offs_n[:, None] < seqlen_k) & (offs_d[None, :] < headdim), other=0.0)
            v = tl.load(v_ptrs, mask=(offs_n[:, None] < seqlen_k) & (offs_d[None, :] < headdim), other=0.0)
    # loop over rows
    num_block_m = tl.cdiv(seqlen_q, BLOCK_M)
    for start_m in range(begin_m, num_block_m * BLOCK_M, BLOCK_M):
        start_m = tl.multiple_of(start_m, BLOCK_M)
        offs_m_curr = start_m + offs_m
        # load q, k, v, do on-chip
        # Same bug as below. Otherwise gives wrong result for headdim=40, seqlen=(128, 117)
        if EVEN_M & EVEN_HEADDIM:
            q = tl.load(q_ptrs)
        else:
            if EVEN_HEADDIM:
                q = tl.load(q_ptrs, mask=offs_m_curr[:, None] < seqlen_q, other=0.0)
            else:
                q = tl.load(
                    q_ptrs,
                    mask=(offs_m_curr[:, None] < seqlen_q) & (offs_d[None, :] < headdim),
                    other=0.0,
                )
        # recompute p = softmax(qk, dim=-1).T
        qk = tl.dot(q, tl.trans(k))

        if SOFTCLAMP_QK_SIM:
            effective_softclamp_value = SOFTCLAMP_VALUE / softmax_scale
            qk /= effective_softclamp_value
            qk = libdevice.tanh(qk)
            dtanh = 1.0 - qk * qk
            qk *= effective_softclamp_value

        # Trying to combine the two masks seem to make the result wrong
        if not EVEN_N:  # Need to mask out otherwise the softmax is wrong
            qk = tl.where(offs_n[None, :] < seqlen_k, qk, float("-inf"))
        if IS_CAUSAL:
            if CAUSAL_MASK_DIAGONAL:
                # needed for stripe attention
                qk = tl.where(offs_m_curr[:, None] > (offs_n[None, :]), qk, float("-inf"))
            else:
                qk = tl.where(offs_m_curr[:, None] >= (offs_n[None, :]), qk, float("-inf"))

        if BIAS_TYPE != "none":
            tl.debug_barrier()  # Race condition otherwise
            if BIAS_TYPE == "vector":
                if EVEN_N:
                    bias = tl.load(b_ptrs).to(tl.float32)
                else:
                    bias = tl.load(b_ptrs, mask=offs_n < seqlen_k, other=0.0).to(tl.float32)
                bias = bias[None, :]
            elif BIAS_TYPE == "matrix":
                if EVEN_M & EVEN_N:
                    bias = tl.load(b_ptrs).to(tl.float32)
                else:
                    bias = tl.load(
                        b_ptrs,
                        mask=(offs_m_curr[:, None] < seqlen_q) & (offs_n[None, :] < seqlen_k),
                        other=0.0,
                    ).to(tl.float32)
            qk = qk * softmax_scale + bias
        # There seems to be a race condition when headdim=48/96, and dq, dk, dv are wrong.
        # Also wrong for headdim=64.
        if not (EVEN_M & EVEN_HEADDIM):
            tl.debug_barrier()
        lse_i = tl.load(LSE + offs_m_curr)
        if BIAS_TYPE == "none":
            p = tl.exp(qk * softmax_scale - lse_i[:, None])
        else:
            p = tl.exp(qk - lse_i[:, None])
        # compute dv
        # [2022-10-30] TD: A Triton bug: if EVEN_M=True and EVEN_HEADDIM=False, if we call
        # do = tl.load(do_ptrs, mask=offs_d[None, :] < headdim, other=0.0), we get wrong outputs
        # in the case of headdim=48/96, seqlen_q & seqlen_k >= 512. If headdim=40 or seqlen < 512,
        # the output is correct.
        if EVEN_M & EVEN_HEADDIM:
            do = tl.load(do_ptrs)
        else:
            # [2022-11-01] TD: Triton bug, there's a race condition if we just use m_mask and not d_mask.
            do = tl.load(
                do_ptrs,
                mask=(offs_m_curr[:, None] < seqlen_q) & (offs_d[None, :] < headdim),
                other=0.0,
            )
        # if EVEN_M:
        #     if EVEN_HEADDIM:
        #         do = tl.load(do_ptrs)
        #     else:
        #         do = tl.load(do_ptrs, mask=offs_d[None, :] < headdim, other=0.0)
        # else:
        #     if EVEN_HEADDIM:
        #         do = tl.load(do_ptrs, mask=offs_m_curr[:, None] < seqlen_q, other=0.0)
        #     else:
        #         do = tl.load(do_ptrs, mask=(offs_m_curr[:, None] < seqlen_q)
        #                                    & (offs_d[None, :] < headdim), other=0.0)
        dv += tl.dot(tl.trans(p.to(do.dtype)), do)
        # compute dp = dot(v, do)
        # There seems to be a race condition when headdim=48/96, and dq, dk are wrong.
        # Also wrong for headdim=128, seqlen=(108, 256), and ATOMIC_ADD=True
        # Also wrong for headdim=64, seqlen=(1023, 1024), and ATOMIC_ADD=False
        if not (EVEN_M & EVEN_HEADDIM):
            tl.debug_barrier()
        dp = tl.dot(do, tl.trans(v))
        # There's a race condition for headdim=48
        if not EVEN_HEADDIM:
            tl.debug_barrier()
        # compute ds = p * (dp - delta[:, None])
        # Putting the subtraction after the dp matmul (instead of before) is slightly faster
        Di = tl.load(D + offs_m_curr)
        # Converting ds to q.dtype here reduces register pressure and makes it much faster
        # for BLOCK_HEADDIM=128
        ds = p * (dp - Di[:, None]) * softmax_scale

        if SOFTCLAMP_QK_SIM:
            ds *= dtanh

        ds = ds.to(q.dtype)

        # compute dk = dot(ds.T, q)
        dk += tl.dot(tl.trans(ds), q)
        # compute dq
        if not (EVEN_M & EVEN_HEADDIM):  # Otherwise there's a race condition when BIAS_TYPE='matrix'
            tl.debug_barrier()
        if not ATOMIC_ADD:
            if EVEN_M & EVEN_HEADDIM:  # Race condition if we just do EVEN_M
                dq = tl.load(dq_ptrs, eviction_policy="evict_last")
                dq += tl.dot(ds, k)
                tl.store(dq_ptrs, dq, eviction_policy="evict_last")
            else:
                if EVEN_HEADDIM:
                    dq = tl.load(
                        dq_ptrs,
                        mask=offs_m_curr[:, None] < seqlen_q,
                        other=0.0,
                        eviction_policy="evict_last",
                    )
                    dq += tl.dot(ds, k)
                    tl.store(
                        dq_ptrs,
                        dq,
                        mask=offs_m_curr[:, None] < seqlen_q,
                        eviction_policy="evict_last",
                    )
                else:
                    dq = tl.load(
                        dq_ptrs,
                        mask=(offs_m_curr[:, None] < seqlen_q) & (offs_d[None, :] < headdim),
                        other=0.0,
                        eviction_policy="evict_last",
                    )
                    dq += tl.dot(ds, k)
                    tl.store(
                        dq_ptrs,
                        dq,
                        mask=(offs_m_curr[:, None] < seqlen_q) & (offs_d[None, :] < headdim),
                        eviction_policy="evict_last",
                    )
        else:  # If we're parallelizing across the seqlen_k dimension
            dq = tl.dot(ds, k)
            if EVEN_M & EVEN_HEADDIM:  # Race condition if we just do EVEN_M
                tl.atomic_add(dq_ptrs, dq, sem="relaxed")
            else:
                if EVEN_HEADDIM:
                    tl.atomic_add(dq_ptrs, dq, mask=offs_m_curr[:, None] < seqlen_q, sem="relaxed")
                else:
                    tl.atomic_add(
                        dq_ptrs,
                        dq,
                        mask=(offs_m_curr[:, None] < seqlen_q) & (offs_d[None, :] < headdim),
                        sem="relaxed",
                    )
        # increment pointers
        dq_ptrs += BLOCK_M * stride_dqm
        q_ptrs += BLOCK_M * stride_qm
        do_ptrs += BLOCK_M * stride_dom
        if BIAS_TYPE == "matrix":
            b_ptrs += BLOCK_M * stride_bm
    # write-back
    dv_ptrs = DV + (offs_n[:, None] * stride_dvn + offs_d[None, :])
    dk_ptrs = DK + (offs_n[:, None] * stride_dkn + offs_d[None, :])
    _bwd_store_dk_dv(
        dk_ptrs,
        dv_ptrs,
        dk,
        dv,
        offs_n,
        offs_d,
        seqlen_k,
        headdim,
        EVEN_M=EVEN_M,
        EVEN_N=EVEN_N,
        EVEN_HEADDIM=EVEN_HEADDIM,
    )


def init_to_zero(name):
    # [FUNCTION DOC]
    # HELPER: builds a Triton autotune pre-hook that zeroes the named kernel argument (used to clear DQ).
    # Inputs: name (argument name). Outputs: a callable taking the kernel argument dictionary.
    # Exceptions: None explicitly raised. Side effects: the returned hook mutates the tensor.
    return lambda nargs: nargs[name].zero_()


@triton.autotune(
    configs=[
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 128, "SEQUENCE_PARALLEL": False},
            num_warps=8,
            num_stages=1,
            pre_hook=init_to_zero("DQ"),
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 128, "SEQUENCE_PARALLEL": True},
            num_warps=8,
            num_stages=1,
            pre_hook=init_to_zero("DQ"),
        ),
        # Other configs seem to give wrong results when seqlen_q % 128 != 0, disabling them for now
        # # Kernel is buggy (give wrong result) if we set BLOCK_m=128, BLOCK_n=64, num_warps=*4*
        # triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "SEQUENCE_PARALLEL": False}, num_warps=8, num_stages=1, pre_hook=init_to_zero('DQ')),
        # triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "SEQUENCE_PARALLEL": True}, num_warps=8, num_stages=1, pre_hook=init_to_zero('DQ')),
        # triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "SEQUENCE_PARALLEL": False}, num_warps=4, num_stages=1, pre_hook=init_to_zero('DQ')),
        # triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "SEQUENCE_PARALLEL": True}, num_warps=4, num_stages=1, pre_hook=init_to_zero('DQ')),
    ],
    key=["CACHE_KEY_SEQLEN_Q", "CACHE_KEY_SEQLEN_K", "BIAS_TYPE", "IS_CAUSAL", "BLOCK_HEADDIM"],
)
@triton.heuristics(
    {
        "EVEN_M": lambda args: args["seqlen_q"] % args["BLOCK_M"] == 0,
        "EVEN_N": lambda args: args["seqlen_k"] % args["BLOCK_N"] == 0,
        "EVEN_HEADDIM": lambda args: args["headdim"] == args["BLOCK_HEADDIM"],
    }
)
@triton.jit
def _bwd_kernel(
    Q,
    K,
    V,
    Bias,
    DO,
    DQ,
    DK,
    DV,
    LSE,
    D,
    softmax_scale,
    stride_qb,
    stride_qh,
    stride_qm,
    stride_kb,
    stride_kh,
    stride_kn,
    stride_vb,
    stride_vh,
    stride_vn,
    stride_bb,
    stride_bh,
    stride_bm,
    stride_dob,
    stride_doh,
    stride_dom,
    stride_dqb,
    stride_dqh,
    stride_dqm,
    stride_dkb,
    stride_dkh,
    stride_dkn,
    stride_dvb,
    stride_dvh,
    stride_dvn,
    nheads,
    seqlen_q,
    seqlen_k,
    seqlen_q_rounded,
    headdim,
    CACHE_KEY_SEQLEN_Q,
    CACHE_KEY_SEQLEN_K,
    BIAS_TYPE: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    CAUSAL_MASK_DIAGONAL: tl.constexpr,
    SOFTCLAMP_QK_SIM: tl.constexpr,
    SOFTCLAMP_VALUE: tl.constexpr,
    BLOCK_HEADDIM: tl.constexpr,
    SEQUENCE_PARALLEL: tl.constexpr,
    EVEN_M: tl.constexpr,
    EVEN_N: tl.constexpr,
    EVEN_HEADDIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    # [FUNCTION DOC]
    # Triton backward kernel (autotuned over sequence-parallel or not, 128 by 128 blocks). Offsets pointers for the
    # batch-head program, then either loops over all key blocks serially or handles the single key block given by
    # program id 0 with atomic dq accumulation.
    # Inputs: tensor pointers, strides, sizes, cache keys and compile-time flags.
    # Outputs: None. Exceptions: None explicitly raised. Side effects: writes DQ, DK, DV.
    off_hb = tl.program_id(1)
    off_b = off_hb // nheads
    off_h = off_hb % nheads
    # offset pointers for batch/head
    Q += off_b * stride_qb + off_h * stride_qh
    K += off_b * stride_kb + off_h * stride_kh
    V += off_b * stride_vb + off_h * stride_vh
    DO += off_b * stride_dob + off_h * stride_doh
    DQ += off_b * stride_dqb + off_h * stride_dqh
    DK += off_b * stride_dkb + off_h * stride_dkh
    DV += off_b * stride_dvb + off_h * stride_dvh
    if BIAS_TYPE != "none":
        Bias += off_b * stride_bb + off_h * stride_bh
    # pointer to row-wise quantities in value-like data
    D += off_hb * seqlen_q_rounded
    LSE += off_hb * seqlen_q_rounded
    if not SEQUENCE_PARALLEL:
        num_block_n = tl.cdiv(seqlen_k, BLOCK_N)
        for start_n in range(0, num_block_n):
            _bwd_kernel_one_col_block(
                start_n,
                Q,
                K,
                V,
                Bias,
                DO,
                DQ,
                DK,
                DV,
                LSE,
                D,
                softmax_scale,
                stride_qm,
                stride_kn,
                stride_vn,
                stride_bm,
                stride_dom,
                stride_dqm,
                stride_dkn,
                stride_dvn,
                seqlen_q,
                seqlen_k,
                headdim,
                ATOMIC_ADD=False,
                BIAS_TYPE=BIAS_TYPE,
                IS_CAUSAL=IS_CAUSAL,
                CAUSAL_MASK_DIAGONAL=CAUSAL_MASK_DIAGONAL,
                SOFTCLAMP_QK_SIM=SOFTCLAMP_QK_SIM,
                SOFTCLAMP_VALUE=SOFTCLAMP_VALUE,
                BLOCK_HEADDIM=BLOCK_HEADDIM,
                EVEN_M=EVEN_M,
                EVEN_N=EVEN_N,
                EVEN_HEADDIM=EVEN_HEADDIM,
                BLOCK_M=BLOCK_M,
                BLOCK_N=BLOCK_N,
            )
    else:
        start_n = tl.program_id(0)
        _bwd_kernel_one_col_block(
            start_n,
            Q,
            K,
            V,
            Bias,
            DO,
            DQ,
            DK,
            DV,
            LSE,
            D,
            softmax_scale,
            stride_qm,
            stride_kn,
            stride_vn,
            stride_bm,
            stride_dom,
            stride_dqm,
            stride_dkn,
            stride_dvn,
            seqlen_q,
            seqlen_k,
            headdim,
            ATOMIC_ADD=True,
            BIAS_TYPE=BIAS_TYPE,
            IS_CAUSAL=IS_CAUSAL,
            CAUSAL_MASK_DIAGONAL=CAUSAL_MASK_DIAGONAL,
            SOFTCLAMP_QK_SIM=SOFTCLAMP_QK_SIM,
            SOFTCLAMP_VALUE=SOFTCLAMP_VALUE,
            BLOCK_HEADDIM=BLOCK_HEADDIM,
            EVEN_M=EVEN_M,
            EVEN_N=EVEN_N,
            EVEN_HEADDIM=EVEN_HEADDIM,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
        )


def flash_attn_backward(
    do,
    q,
    k,
    v,
    o,
    lse,
    dq,
    dk,
    dv,
    delta=None,
    bias=None,
    causal=False,
    causal_mask_diagonal=False,
    softmax_scale=None,
    softclamp_qk_sim=False,
    softclamp_value=50.0,
):
    # Make sure that the last dimension is contiguous
    # [FUNCTION DOC]
    # Python launcher for the backward kernels. Makes do contiguous, checks shapes/strides, defaults softmax_scale to
    # one over sqrt(d); computes delta with _bwd_preprocess_do_o_dot unless provided; classifies the bias as vector
    # ([..., 1, seq_k]) or matrix; launches _bwd_kernel accumulating dq in float32, writes dk and dv directly, and copies
    # the float32 dq into dq.
    # Inputs: do, q, k, v, o ([batch, seq, heads, d]); lse [batch, heads, seq_q rounded to 128]; dq, dk, dv output
    # buffers; delta (optional cached); bias (optional 4-D); causal; causal_mask_diagonal; softmax_scale;
    # softclamp_qk_sim; softclamp_value.
    # Outputs: delta (to reuse in later ring passes).
    # Exceptions: AssertionError on shape/stride/dtype violations; RuntimeError for an unsupported bias shape.
    # Side effects: Triton kernel launches; writes dq, dk, dv.
    if do.stride(-1) != 1:
        do = do.contiguous()
    batch, seqlen_q, nheads, d = q.shape
    _, seqlen_k, _, _ = k.shape
    # assert d in {16, 32, 64, 128}
    assert d <= 128
    seqlen_q_rounded = ceil(seqlen_q / 128) * 128

    assert lse.shape == (batch, nheads, seqlen_q_rounded)
    assert q.stride(-1) == k.stride(-1) == v.stride(-1) == o.stride(-1) == 1
    assert dq.stride(-1) == dk.stride(-1) == dv.stride(-1) == 1
    softmax_scale = softmax_scale or 1.0 / math.sqrt(d)
    # dq_accum = torch.zeros_like(q, dtype=torch.float32)
    dq_accum = torch.empty_like(q, dtype=torch.float32)

    # delta = torch.zeros_like(lse)

    BLOCK_HEADDIM = max(triton.next_power_of_2(d), 16)

    if not exists(delta):
        delta = torch.empty_like(lse)

        def grid(META):
            # [FUNCTION DOC]
            # HELPER: launch grid for _bwd_preprocess_do_o_dot: ceil of seqlen_q over BLOCK_M by batch times heads.
            # Inputs: META. Outputs: tuple. Exceptions: None explicitly raised. Side effects: None.
            return triton.cdiv(seqlen_q, META["BLOCK_M"]), batch * nheads

        _bwd_preprocess_do_o_dot[grid](
            o,
            do,
            delta,
            o.stride(0),
            o.stride(2),
            o.stride(1),
            do.stride(0),
            do.stride(2),
            do.stride(1),
            nheads,
            seqlen_q,
            seqlen_q_rounded,
            d,
            BLOCK_M=128,
            BLOCK_HEADDIM=BLOCK_HEADDIM,
        )

    has_bias = bias is not None
    bias_type = "none"
    if has_bias:
        assert bias.dtype in [q.dtype, torch.float]
        assert bias.is_cuda
        assert bias.dim() == 4
        assert bias.stride(-1) == 1
        if bias.shape[2:] == (1, seqlen_k):
            bias_type = "vector"
        elif bias.shape[2:] == (seqlen_q, seqlen_k):
            bias_type = "matrix"
        else:
            raise RuntimeError("Last 2 dimensions of bias must be (1, seqlen_k)" " or (seqlen_q, seqlen_k)")
        bias = bias.expand(batch, nheads, seqlen_q, seqlen_k)
    bias_strides = (bias.stride(0), bias.stride(1), bias.stride(2)) if has_bias else (0, 0, 0)

    # BLOCK_M = 128
    # BLOCK_N = 64
    # num_warps = 4
    def grid(META):
        # [FUNCTION DOC]
        # HELPER: launch grid for _bwd_kernel: number of key blocks when SEQUENCE_PARALLEL else 1, by batch times heads.
        # Inputs: META. Outputs: tuple. Exceptions: None explicitly raised. Side effects: None.
        return triton.cdiv(seqlen_k, META["BLOCK_N"]) if META["SEQUENCE_PARALLEL"] else 1, batch * nheads

    _bwd_kernel[grid](
        q,
        k,
        v,
        bias,
        do,
        dq_accum,
        dk,
        dv,
        lse,
        delta,
        softmax_scale,
        q.stride(0),
        q.stride(2),
        q.stride(1),
        k.stride(0),
        k.stride(2),
        k.stride(1),
        v.stride(0),
        v.stride(2),
        v.stride(1),
        *bias_strides,
        do.stride(0),
        do.stride(2),
        do.stride(1),
        dq_accum.stride(0),
        dq_accum.stride(2),
        dq_accum.stride(1),
        dk.stride(0),
        dk.stride(2),
        dk.stride(1),
        dv.stride(0),
        dv.stride(2),
        dv.stride(1),
        nheads,
        seqlen_q,
        seqlen_k,
        seqlen_q_rounded,
        d,
        seqlen_q // 32,
        seqlen_k // 32,  # key for triton cache (limit number of compilations)
        # Can't use kwargs here because triton autotune expects key to be args, not kwargs
        # IS_CAUSAL=causal, BLOCK_HEADDIM=d,
        bias_type,
        causal,
        causal_mask_diagonal,
        softclamp_qk_sim,
        softclamp_value,
        BLOCK_HEADDIM,
        # SEQUENCE_PARALLEL=False,
        # BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
        # num_warps=num_warps,
        # num_stages=1,
    )
    dq.copy_(dq_accum)

    return delta
