# [MODULE DOC]
# Context-parallel ring attention for packed variable-length sequences, built on FlashAttention's varlen kernels
# (adapted from zhuzilin/ring-flash-attention; the original triton_utils.py and utils.py were concatenated here).
# Algorithm: each rank of the ring process group holds one contiguous chunk of the sequence. Over world_size steps,
# every rank computes attention of its local queries against the key/value chunk currently held, while
# asynchronously passing that chunk to the next rank (rank plus one) and receiving from the previous rank. Partial
# outputs are merged with a numerically stable log-sum-exp update. With causal masking, rank r only uses chunks
# originating from ranks at or below r (step 0 is the diagonal block, computed causally). Backward re-runs the ring
# and additionally circulates key/value gradients so each gradient arrives back at its owner.
# Main contents: forward/backward ring functions, the RingFlashAttnVarlenFunc autograd function, the
# ring_flash_attn_varlen_func entry point (registered as "ring" in ALL_ATTENTION_FUNCTIONS), Triton and
# TorchScript LSE flatten/unflatten utilities, LSE merge helpers, default-argument introspection, and RingComm /
# AllGatherComm communication helpers.
# Import-time side effects: imports flash_attn and triton; JIT-scripts several functions; the later TorchScript
# definitions of flatten_varlen_lse and unflatten_varlen_lse replace the earlier Triton-based ones.
# Exports: __all__ lists update_out_and_lse, RingComm and get_default_args (assigned mid-file); other names are
# public by convention.
"""Ring attention implementation using flash attention adapted from https://github.com/zhuzilin/ring-flash-attention/"""
import torch
import torch.distributed as dist
from flash_attn.flash_attn_interface import (
    _flash_attn_varlen_backward,
    _flash_attn_varlen_forward,
)


def ring_flash_attn_varlen_forward(
    process_group,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens,
    max_seqlen,
    softmax_scale,
    dropout_p=0,
    causal=True,
    window_size=(-1, -1),
    alibi_slopes=None,
    deterministic=False,
):
    # [FUNCTION DOC]
    # Ring forward pass. For each step: if not the last step, starts sending the current k and v to the next rank and
    # receiving the next chunk; if attention is non-causal or step is at most this rank, calls
    # _flash_attn_varlen_forward with default arguments overridden (same cu_seqlens and max_seqlen for queries and keys,
    # causal only at step 0, probabilities returned only with dropout, window size passed in whichever form the
    # installed flash_attn accepts), extracts the block output and LSE (handles 8-output and 4-output return forms),
    # flattens 3-D LSE from older flash_attn versions, and merges into the running output with update_out_and_lse; then
    # waits for the exchange and swaps in the received chunk.
    # Inputs: process_group (ring group); q, k, v tokens-first varlen tensors [total_tokens, heads, head_dim];
    # cu_seqlens (int32 cumulative lengths of local documents); max_seqlen (int); softmax_scale (float); dropout_p
    # (default 0); causal (default True); window_size (pair, default unlimited); alibi_slopes (default None);
    # deterministic (unused here).
    # Outputs: tuple of output in q dtype [total_tokens, heads, head_dim] and LSE (unflattened to
    # [batch, heads, max_seqlen] for old flash_attn, otherwise [heads, total_tokens]).
    # Exceptions: AssertionError on an unexpected number of kernel outputs.
    # Side effects: point-to-point send/recv on process_group; CUDA kernels; dropout RNG.
    # KNOWN ISSUE: the window_size is applied per block with no offset between chunks, so sliding windows are not
    # correct across chunk boundaries.
    comm = RingComm(process_group)

    out = None
    lse = None
    next_k, next_v = None, None

    old_lse = False
    for step in range(comm.world_size):
        if step + 1 != comm.world_size:
            next_k, next_v = comm.send_recv_kv(k, v)
        if not causal or step <= comm.rank:
            params = get_default_args(_flash_attn_varlen_forward).copy()
            params.update(
                {
                    "q": q,
                    "k": k,
                    "v": v,
                    "cu_seqlens_q": cu_seqlens,
                    "cu_seqlens_k": cu_seqlens,
                    "max_seqlen_q": max_seqlen,
                    "max_seqlen_k": max_seqlen,
                    "dropout_p": dropout_p,
                    "softmax_scale": softmax_scale,
                    "causal": causal and step == 0,
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
                block_out, _, _, _, _, block_lse, _, _ = outputs
            else:
                assert len(outputs) == 4
                block_out, block_lse, _, _ = outputs
            if block_lse.dim() == 3:
                old_lse = True
                block_lse = flatten_varlen_lse(
                    block_lse,
                    cu_seqlens=cu_seqlens,
                )
            out, lse = update_out_and_lse(out, lse, block_out, block_lse)

        if step + 1 != comm.world_size:
            comm.wait()
            k, v = next_k, next_v

    out = out.to(q.dtype)
    if old_lse:
        lse = unflatten_varlen_lse(lse, cu_seqlens, max_seqlen)
    else:
        lse = lse.squeeze(dim=-1).transpose(0, 1)
    return out, lse


def ring_flash_attn_varlen_backward(
    process_group,
    dout,
    q,
    k,
    v,
    out,
    softmax_lse,
    cu_seqlens,
    max_seqlen,
    softmax_scale,
    dropout_p=0,
    causal=True,
    window_size=(-1, -1),
    alibi_slopes=None,
    deterministic=False,
):
    # [FUNCTION DOC]
    # Ring backward pass. Allocates reusable gradient buffers. For each step: prefetches the next k/v chunk; when the
    # block was used in forward, runs _flash_attn_varlen_backward (causal only at step 0) into the buffers; on the first
    # contribution initializes float32 dq, dk, dv, afterwards accumulates dq and adds the block dk/dv to the gradients
    # received from the previous rank (after waiting on the gradient exchange). When a block was skipped (causal, step
    # beyond rank) and step is not zero, simply forwards the received dk/dv. Then swaps k/v and sends the current dk/dv
    # to the next rank. After the loop waits for the final gradient exchange, so each rank holds the gradients of its
    # own key/value chunk.
    # Inputs: process_group; dout (output gradient); q, k, v; out; softmax_lse (from forward); cu_seqlens; max_seqlen;
    # softmax_scale; dropout_p (0); causal (True); window_size; alibi_slopes; deterministic (passed to the kernel).
    # Outputs: tuple dq, dk, dv.
    # Exceptions: None explicitly raised.
    # Side effects: point-to-point communication through two RingComm instances; CUDA kernels.
    # KNOWN ISSUE: dq is always cast to bfloat16 regardless of input dtype, while dk/dv are cast to q dtype.
    kv_comm = RingComm(process_group)
    d_kv_comm = RingComm(process_group)
    dq, dk, dv = None, None, None
    next_dk, next_dv = None, None

    block_dq_buffer = torch.empty(q.shape, dtype=q.dtype, device=q.device)
    block_dk_buffer = torch.empty(k.shape, dtype=k.dtype, device=k.device)
    block_dv_buffer = torch.empty(v.shape, dtype=v.dtype, device=v.device)

    next_dk, next_dv = None, None
    next_k, next_v = None, None
    for step in range(kv_comm.world_size):
        if step + 1 != kv_comm.world_size:
            next_k, next_v = kv_comm.send_recv_kv(k, v)

        if step <= kv_comm.rank or not causal:
            bwd_causal = causal and step == 0
            params = get_default_args(_flash_attn_varlen_backward).copy()
            params.update(
                {
                    "dout": dout,
                    "q": q,
                    "k": k,
                    "v": v,
                    "out": out,
                    "softmax_lse": softmax_lse,
                    "dq": block_dq_buffer,
                    "dk": block_dk_buffer,
                    "dv": block_dv_buffer,
                    "cu_seqlens_q": cu_seqlens,
                    "cu_seqlens_k": cu_seqlens,
                    "max_seqlen_q": max_seqlen,
                    "max_seqlen_k": max_seqlen,
                    "dropout_p": dropout_p,
                    "softmax_scale": softmax_scale,
                    "causal": bwd_causal,
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

            if dq is None:
                dq = block_dq_buffer.to(torch.float32)
                dk = block_dk_buffer.to(torch.float32)
                dv = block_dv_buffer.to(torch.float32)
            else:
                dq += block_dq_buffer
                d_kv_comm.wait()
                dk = block_dk_buffer + next_dk
                dv = block_dv_buffer + next_dv
        elif step != 0:
            d_kv_comm.wait()
            dk, dv = next_dk, next_dv

        if step + 1 != kv_comm.world_size:
            kv_comm.wait()
            k, v = next_k, next_v

        next_dk, next_dv = d_kv_comm.send_recv_kv(dk, dv)

    d_kv_comm.wait()

    return dq.to(torch.bfloat16), next_dk.to(q.dtype), next_dv.to(q.dtype)


class RingFlashAttnVarlenFunc(torch.autograd.Function):
    # [CLASS DOC]
    # torch.autograd.Function wrapping the ring forward/backward. Saves q, k, v, out, LSE and cu_seqlens plus scalar
    # settings (max_seqlen, dropout_p, softmax_scale, causal, window_size, alibi_slopes, deterministic, group) on ctx.
    @staticmethod
    def forward(
        ctx,
        q,
        k,
        v,
        cu_seqlens,
        max_seqlen,
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
        # Defaults softmax_scale to one over sqrt(head_dim), makes k and v contiguous, runs ring_flash_attn_varlen_forward
        # (deterministic forced False) and saves state for backward.
        # Inputs: ctx; q, k, v; cu_seqlens; max_seqlen; dropout_p; softmax_scale (optional); causal; window_size;
        # alibi_slopes (must be None); deterministic; return_softmax (bool); group (process group).
        # Outputs: the output tensor, or a tuple of output, LSE and None when return_softmax is true.
        # Exceptions: AssertionError if alibi_slopes is not None.
        # Side effects: ring communication; stores tensors on ctx.
        if softmax_scale is None:
            softmax_scale = q.shape[-1] ** (-0.5)

        assert alibi_slopes is None
        k = k.contiguous()
        v = v.contiguous()
        out, softmax_lse = ring_flash_attn_varlen_forward(
            group,
            q,
            k,
            v,
            cu_seqlens,
            max_seqlen,
            softmax_scale=softmax_scale,
            dropout_p=dropout_p,
            causal=causal,
            window_size=window_size,
            alibi_slopes=alibi_slopes,
            deterministic=False,
        )
        # this should be out_padded
        ctx.save_for_backward(q, k, v, out, softmax_lse, cu_seqlens)
        ctx.max_seqlen = max_seqlen
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
        # Runs ring_flash_attn_varlen_backward with the saved state.
        # Inputs: ctx; dout (gradient of output); extra gradients ignored.
        # Outputs: dq, dk, dv followed by ten None values for the non-tensor forward inputs.
        # Exceptions: None explicitly raised. Side effects: ring communication.
        q, k, v, out, softmax_lse, cu_seqlens = ctx.saved_tensors
        dq, dk, dv = ring_flash_attn_varlen_backward(
            ctx.group,
            dout,
            q,
            k,
            v,
            out,
            softmax_lse,
            cu_seqlens,
            ctx.max_seqlen,
            softmax_scale=ctx.softmax_scale,
            dropout_p=ctx.dropout_p,
            causal=ctx.causal,
            window_size=ctx.window_size,
            alibi_slopes=ctx.alibi_slopes,
            deterministic=ctx.deterministic,
        )
        return dq, dk, dv, None, None, None, None, None, None, None, None, None, None


def ring_flash_attn_varlen_func(
    module,
    q,
    k,
    v,
    cu_seqlens=None,
    max_seqlen=None,
    dropout=0.0,
    scaling=None,
    causal=False,
    window_size=(-1, -1),  # -1 means infinite context window
    alibi_slopes=None,
    deterministic=False,
    return_attn_probs=False,
    ring_pg=None,
    **kwargs,
):
    # [FUNCTION DOC]
    # Attention-registry entry point for ring attention. Maps registry-style arguments (dropout, scaling, ring_pg) onto
    # RingFlashAttnVarlenFunc.apply.
    # Inputs: module (unused); q, k, v; cu_seqlens and max_seqlen (required); dropout (0.0); scaling (optional softmax
    # scale); causal (default False); window_size (default unlimited); alibi_slopes; deterministic; return_attn_probs;
    # ring_pg (process group for the ring); other keyword arguments ignored.
    # Outputs: a one-element tuple containing the apply result.
    # Exceptions: AssertionError if cu_seqlens or max_seqlen is None.
    # Side effects: ring communication.
    # KNOWN ISSUE: returns a 1-tuple, unlike the other registry backends which return a pair of output and weights.
    assert cu_seqlens is not None
    assert max_seqlen is not None
    return (
        RingFlashAttnVarlenFunc.apply(
            q,
            k,
            v,
            cu_seqlens,
            max_seqlen,
            dropout,
            scaling,
            causal,
            window_size,
            alibi_slopes,
            deterministic,
            return_attn_probs,
            ring_pg,
        ),
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

__all__ = ["update_out_and_lse", "RingComm", "get_default_args"]


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
