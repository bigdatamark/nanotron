# [MODULE DOC]
# Purpose: autograd wrappers around the CUDA selective-scan (Mamba SSM) and causal conv1d kernels, adapted
# from the original mamba_ssm package (Tri Dao, Albert Gu) with tensor-parallel awareness for nanotron's Mamba
# example. Contents: SelectiveScanFn and selective_scan_fn (fused scan), selective_scan_ref (pure PyTorch
# reference), MambaInnerFn and mamba_inner_fn (fused conv1d + x_proj + delta projection + scan + gating,
# TP-aware, without out_proj), mamba_inner_ref (reference including out_proj).
# Notation: b batch, d channels (d_inner, possibly TP-local), l sequence length, n / dstate state size.
# Import-time side effects: imports compiled extensions causal_conv1d_cuda and selective_scan_cuda (ImportError
# if not installed).
# Copyright (c) 2023, Tri Dao, Albert Gu.

import causal_conv1d_cuda
import selective_scan_cuda
import torch
import torch.nn.functional as F
from causal_conv1d import causal_conv1d_fn
from einops import rearrange, repeat
from torch.cuda.amp import custom_bwd, custom_fwd


class SelectiveScanFn(torch.autograd.Function):
    # [CLASS DOC]
    # torch.autograd.Function running the selective scan with the selective_scan_cuda kernel.
    # Context state saved in forward: squeeze_B / squeeze_C flags (B or C were 3-D and got a group axis added),
    # delta_softplus, has_z, and the tensors needed for backward (inputs, scan intermediates x, and out when z given).
    @staticmethod
    def forward(
        ctx,
        u,
        delta,
        A,
        B,
        C,
        D=None,
        z=None,
        delta_bias=None,
        delta_softplus=False,
        return_last_state=False,
    ):
        # [FUNCTION DOC]
        # Behavior: makes u, delta, B, C, z contiguous in the last dim (D always contiguous); adds a group axis of size 1
        # to 3-D B or C; calls selective_scan_cuda.fwd; extracts the last state from the odd positions of the last
        # chunk of x; saves tensors for backward (including out when z is present); returns out, or the z-gated output
        # when z is given, optionally with last_state.
        # Inputs: ctx; u (b, d, l); delta (b, d, l); A (d, n) real or complex; B and C (d, n) or (b, n, l) or (b, g, n, l);
        # D (optional d); z (optional b, d, l gate); delta_bias (optional d, fp32); delta_softplus (bool, default False);
        # return_last_state (bool, default False).
        # Outputs: tensor (b, d, l), or pair of that and last_state (b, d, n).
        # Exceptions: kernel errors propagate. Side effects: CUDA kernel launch; stores state on ctx.
        if u.stride(-1) != 1:
            u = u.contiguous()
        if delta.stride(-1) != 1:
            delta = delta.contiguous()
        if D is not None:
            D = D.contiguous()
        if B.stride(-1) != 1:
            B = B.contiguous()
        if C.stride(-1) != 1:
            C = C.contiguous()
        if z is not None and z.stride(-1) != 1:
            z = z.contiguous()
        if B.dim() == 3:
            B = rearrange(B, "b dstate l -> b 1 dstate l")
            ctx.squeeze_B = True
        if C.dim() == 3:
            C = rearrange(C, "b dstate l -> b 1 dstate l")
            ctx.squeeze_C = True
        out, x, *rest = selective_scan_cuda.fwd(u, delta, A, B, C, D, z, delta_bias, delta_softplus)
        ctx.delta_softplus = delta_softplus
        ctx.has_z = z is not None
        last_state = x[:, :, -1, 1::2]  # (batch, dim, dstate)
        if not ctx.has_z:
            ctx.save_for_backward(u, delta, A, B, C, D, delta_bias, x)
            return out if not return_last_state else (out, last_state)
        else:
            ctx.save_for_backward(u, delta, A, B, C, D, z, delta_bias, x, out)
            out_z = rest[0]
            return out_z if not return_last_state else (out_z, last_state)

    @staticmethod
    def backward(ctx, dout, *args):
        # [FUNCTION DOC]
        # Behavior: restores saved tensors (z and out None when no gate), makes dout contiguous, calls
        # selective_scan_cuda.bwd (dz allocated by the kernel, no recompute of out_z), takes dz from the extra
        # outputs when gated, squeezes the group axis of dB/dC if added in forward, and returns gradients for u, delta,
        # A, B, C, D (if given), z, delta_bias (if given) and None for the two flags.
        # Inputs: ctx; dout (gradient of output, b, d, l); extra ignored gradients (for last_state).
        # Outputs: tuple of 10 gradients. Exceptions: kernel errors propagate. Side effects: CUDA kernel launch.
        if not ctx.has_z:
            u, delta, A, B, C, D, delta_bias, x = ctx.saved_tensors
            z = None
            out = None
        else:
            u, delta, A, B, C, D, z, delta_bias, x, out = ctx.saved_tensors
        if dout.stride(-1) != 1:
            dout = dout.contiguous()
        # The kernel supports passing in a pre-allocated dz (e.g., in case we want to fuse the
        # backward of selective_scan_cuda with the backward of chunk).
        # Here we just pass in None and dz will be allocated in the C++ code.
        du, ddelta, dA, dB, dC, dD, ddelta_bias, *rest = selective_scan_cuda.bwd(
            u,
            delta,
            A,
            B,
            C,
            D,
            z,
            delta_bias,
            dout,
            x,
            out,
            None,
            ctx.delta_softplus,
            False,  # option to recompute out_z, not used here
        )
        dz = rest[0] if ctx.has_z else None
        dB = dB.squeeze(1) if getattr(ctx, "squeeze_B", False) else dB
        dC = dC.squeeze(1) if getattr(ctx, "squeeze_C", False) else dC
        return (
            du,
            ddelta,
            dA,
            dB,
            dC,
            dD if D is not None else None,
            dz,
            ddelta_bias if delta_bias is not None else None,
            None,
            None,
        )


def selective_scan_fn(
    u,
    delta,
    A,
    B,
    C,
    D=None,
    z=None,
    delta_bias=None,
    delta_softplus=False,
    return_last_state=False,
):
    # [FUNCTION DOC]
    # Behavior: functional entry point applying SelectiveScanFn with the same arguments (see SelectiveScanFn.forward).
    # Outputs: out, or out with last_state (b, d, n) when return_last_state; last_state receives no gradient.
    # Exceptions: None explicitly raised. Side effects: CUDA kernel launch.
    """if return_last_state is True, returns (out, last_state)
    last_state has shape (batch, dim, dstate). Note that the gradient of the last state is
    not considered in the backward pass.
    """
    return SelectiveScanFn.apply(u, delta, A, B, C, D, z, delta_bias, delta_softplus, return_last_state)


def selective_scan_ref(
    u,
    delta,
    A,
    B,
    C,
    D=None,
    z=None,
    delta_bias=None,
    delta_softplus=False,
    return_last_state=False,
):
    # [FUNCTION DOC]
    # Behavior: pure PyTorch reference of the selective scan. Casts u and delta to float, adds delta_bias and
    # applies softplus if requested. If A is complex and B/C are input-dependent, views their last axis as
    # interleaved real/imag pairs; otherwise casts to float. Computes deltaA as exp of delta times A per
    # (b, d, l, n) and deltaB_u as delta times B times u (B shared, per-batch (b, n, l), or grouped (b, g, n, l)
    # repeated to d channels). Iterates over time: state x becomes deltaA times x plus deltaB_u; output y is the
    # contraction of x with C (shared, per-batch or grouped); records the final state; complex y is replaced by
    # twice its real part. Stacks to (b, d, l), adds u times D if given, multiplies by silu(z) if given, casts back
    # to the input dtype.
    # Inputs: same as SelectiveScanFn.forward. Outputs: out (b, d, l) in u's dtype, plus last_state (b, d, n)
    # when return_last_state. Exceptions: None explicitly raised. Side effects: None.
    """
    u: r(B D L)
    delta: r(B D L)
    A: c(D N) or r(D N)
    B: c(D N) or r(B N L) or r(B N 2L) or r(B G N L) or (B G N L)
    C: c(D N) or r(B N L) or r(B N 2L) or r(B G N L) or (B G N L)
    D: r(D)
    z: r(B D L)
    delta_bias: r(D), fp32

    out: r(B D L)
    last_state (optional): r(B D dstate) or c(B D dstate)
    """
    dtype_in = u.dtype
    u = u.float()
    delta = delta.float()
    if delta_bias is not None:
        delta = delta + delta_bias[..., None].float()
    if delta_softplus:
        delta = F.softplus(delta)
    batch, dim, dstate = u.shape[0], A.shape[0], A.shape[1]
    is_variable_B = B.dim() >= 3
    is_variable_C = C.dim() >= 3
    if A.is_complex():
        if is_variable_B:
            B = torch.view_as_complex(rearrange(B.float(), "... (L two) -> ... L two", two=2))
        if is_variable_C:
            C = torch.view_as_complex(rearrange(C.float(), "... (L two) -> ... L two", two=2))
    else:
        B = B.float()
        C = C.float()
    x = A.new_zeros((batch, dim, dstate))
    ys = []
    deltaA = torch.exp(torch.einsum("bdl,dn->bdln", delta, A))
    if not is_variable_B:
        deltaB_u = torch.einsum("bdl,dn,bdl->bdln", delta, B, u)
    else:
        if B.dim() == 3:
            deltaB_u = torch.einsum("bdl,bnl,bdl->bdln", delta, B, u)
        else:
            B = repeat(B, "B G N L -> B (G H) N L", H=dim // B.shape[1])
            deltaB_u = torch.einsum("bdl,bdnl,bdl->bdln", delta, B, u)
    if is_variable_C and C.dim() == 4:
        C = repeat(C, "B G N L -> B (G H) N L", H=dim // C.shape[1])
    last_state = None
    for i in range(u.shape[2]):
        x = deltaA[:, :, i] * x + deltaB_u[:, :, i]
        if not is_variable_C:
            y = torch.einsum("bdn,dn->bd", x, C)
        else:
            if C.dim() == 3:
                y = torch.einsum("bdn,bn->bd", x, C[:, :, i])
            else:
                y = torch.einsum("bdn,bdn->bd", x, C[:, :, :, i])
        if i == u.shape[2] - 1:
            last_state = x
        if y.is_complex():
            y = y.real * 2
        ys.append(y)
    y = torch.stack(ys, dim=2)  # (batch dim L)
    out = y if D is None else y + u * rearrange(D, "d -> d 1")
    if z is not None:
        out = out * F.silu(z)
    out = out.to(dtype=dtype_in)
    return out if not return_last_state else (out, last_state)


class MambaInnerFn(torch.autograd.Function):
    # [CLASS DOC]
    # torch.autograd.Function fusing the inner Mamba mixer path (causal depthwise conv1d with silu, x_proj,
    # delta projection, selective scan with z gating) for one TP shard; out_proj is applied by the caller
    # (tensor-parallel row linear). Input xz has x and z interleaved along a size-2 axis per channel (nanotron's
    # column-parallel layout) rather than concatenated halves.
    # Context state: d_inner, tp_pg, delta_softplus, checkpoint_lvl, variable-B/C and bias flags, saved tensors.
    @staticmethod
    @custom_fwd
    def forward(
        ctx,
        d_inner,
        tp_pg,
        xz,
        conv1d_weight,
        conv1d_bias,
        x_proj_weight,
        delta_proj_weight,
        A,
        B=None,
        C=None,
        D=None,
        delta_bias=None,
        B_proj_bias=None,
        C_proj_bias=None,
        delta_softplus=True,
        checkpoint_lvl=1,
    ):
        # [FUNCTION DOC]
        # Behavior: asserts checkpoint_lvl is 0 or 1; under autocast casts x_proj and delta_proj weights to the autocast
        # dtype; makes xz contiguous; reshapes conv weight (d, 1, w) to (d, w); splits xz viewed as (b, channels, 2, l)
        # into x and z, where channels is d_inner divided by 2 when the TP group has more than one rank and d_inner
        # otherwise; runs causal_conv1d_fwd with silu; computes x_dbl = x_proj of the conv output flattened to
        # (b*l, d); delta = delta_proj applied to the first delta_rank columns, laid out (b, d, l); if B/C are not
        # given, slices them from x_dbl (after delta_rank, and the last d_state columns), adds optional biases and
        # reshapes to (b, 1, n, l) (complex variant interleaves pairs); runs selective_scan_cuda.fwd with z gating;
        # with checkpoint_lvl 1 drops conv output and delta to recompute in backward; saves tensors.
        # Inputs: ctx; d_inner (int, full inner width); tp_pg (process group); xz (b, 2*local d, l); conv1d_weight
        # (d, 1, w); conv1d_bias (optional d); x_proj_weight (delta_rank + 2n, d); delta_proj_weight (d, delta_rank);
        # A (d, n); B, C (optional fixed); D (optional d); delta_bias (optional d); B_proj_bias, C_proj_bias (optional);
        # delta_softplus (bool, default True); checkpoint_lvl (0 or 1, default 1).
        # Outputs: gated scan output rearranged to (b, l, d).
        # Exceptions: AssertionError for invalid checkpoint_lvl. Side effects: CUDA kernels; stores ctx state.
        # KNOWN ISSUE: forward divides d_inner by 2 whenever TP size exceeds 1, but backward divides by the TP size; these
        # only agree for TP 1 or 2.
        """
        xz: (batch, dim, seqlen)
        """
        assert checkpoint_lvl in [0, 1]
        batch, L = xz.shape[0], xz.shape[-1]
        delta_rank = delta_proj_weight.shape[1]
        d_state = A.shape[-1] * (1 if not A.is_complex() else 2)
        if torch.is_autocast_enabled():
            x_proj_weight = x_proj_weight.to(dtype=torch.get_autocast_gpu_dtype())
            delta_proj_weight = delta_proj_weight.to(dtype=torch.get_autocast_gpu_dtype())

        if xz.stride(-1) != 1:
            xz = xz.contiguous()
        conv1d_weight = rearrange(conv1d_weight, "d 1 w -> d w")

        if tp_pg.size() > 1:
            x, z = xz.view(batch, d_inner // 2, 2, L).chunk(2, dim=2)
        else:
            x, z = xz.view(batch, d_inner, 2, L).chunk(2, dim=2)

        x = x.squeeze(2)
        z = z.squeeze(2)

        conv1d_bias = conv1d_bias.contiguous() if conv1d_bias is not None else None
        conv1d_out = causal_conv1d_cuda.causal_conv1d_fwd(x, conv1d_weight, conv1d_bias, None, True)
        # We're being very careful here about the layout, to avoid extra transposes.
        # We want delta to have d as the slowest moving dimension
        # and L as the fastest moving dimension, since those are what the ssm_scan kernel expects.
        x_dbl = F.linear(rearrange(conv1d_out, "b d l -> (b l) d"), x_proj_weight)  # (bl d)
        delta = rearrange(delta_proj_weight @ x_dbl[:, :delta_rank].t(), "d (b l) -> b d l", l=L)
        ctx.is_variable_B = B is None
        ctx.is_variable_C = C is None
        ctx.B_proj_bias_is_None = B_proj_bias is None
        ctx.C_proj_bias_is_None = C_proj_bias is None
        if B is None:  # variable B
            B = x_dbl[:, delta_rank : delta_rank + d_state]  # (bl dstate)
            if B_proj_bias is not None:
                B = B + B_proj_bias.to(dtype=B.dtype)
            if not A.is_complex():
                # B = rearrange(B, "(b l) dstate -> b dstate l", l=L).contiguous()
                B = rearrange(B, "(b l) dstate -> b 1 dstate l", l=L).contiguous()
            else:
                B = rearrange(B, "(b l) (dstate two) -> b 1 dstate (l two)", l=L, two=2).contiguous()
        else:
            if B.stride(-1) != 1:
                B = B.contiguous()
        if C is None:  # variable C
            C = x_dbl[:, -d_state:]  # (bl dstate)
            if C_proj_bias is not None:
                C = C + C_proj_bias.to(dtype=C.dtype)
            if not A.is_complex():
                # C = rearrange(C, "(b l) dstate -> b dstate l", l=L).contiguous()
                C = rearrange(C, "(b l) dstate -> b 1 dstate l", l=L).contiguous()
            else:
                C = rearrange(C, "(b l) (dstate two) -> b 1 dstate (l two)", l=L, two=2).contiguous()
        else:
            if C.stride(-1) != 1:
                C = C.contiguous()
        if D is not None:
            D = D.contiguous()
        out, scan_intermediates, out_z = selective_scan_cuda.fwd(
            conv1d_out, delta, A, B, C, D, z, delta_bias, delta_softplus
        )
        ctx.delta_softplus = delta_softplus
        # ctx.out_proj_bias_is_None = out_proj_bias is None
        ctx.checkpoint_lvl = checkpoint_lvl
        if checkpoint_lvl >= 1:  # Will recompute conv1d_out and delta in the backward pass
            conv1d_out, delta = None, None

        ctx.d_inner = d_inner
        ctx.tp_pg = tp_pg

        ctx.save_for_backward(
            xz,
            conv1d_weight,
            conv1d_bias,
            x_dbl,
            x_proj_weight,
            delta_proj_weight,
            conv1d_out,
            delta,
            A,
            B,
            C,
            D,
            delta_bias,
            scan_intermediates,
            out,
        )

        return rearrange(out_z, "b d l -> b l d")

    @staticmethod
    @custom_bwd
    def backward(ctx, dout):
        # [FUNCTION DOC]
        # Behavior: restores saved tensors, re-splits xz into x and z using d_inner divided by TP size (asserting
        # divisibility); with checkpoint_lvl 1 recomputes conv output and delta; allocates dxz and takes dx/dz as views
        # into it so kernels write directly; transposes dout to (b, d, l); calls selective_scan_cuda.bwd (recomputing
        # out_z, writing dz in place); for variable B/C flattens their gradients to (b*l, n), sums them for bias
        # gradients and writes them into dx_dbl; computes delta_proj weight gradient and dx_dbl's delta part; computes
        # x_proj weight gradient and adds x_proj backprop into the conv output gradient (addmm in place); runs
        # causal_conv1d_bwd writing dx into dxz; reshapes conv weight gradient to (d, 1, w).
        # Inputs: ctx; dout (b, l, d). Outputs: tuple of gradients aligned with forward inputs (None for d_inner, tp_pg).
        # Exceptions: AssertionError if d_inner is not divisible by TP size. Side effects: CUDA kernels.
        # KNOWN ISSUE: returns 15 gradients while forward takes 16 inputs (no entry for checkpoint_lvl), which autograd
        # may reject; dB/dC are returned as None for variable B/C, as intended.
        # dout: (batch, seqlen, dim)
        (
            xz,
            conv1d_weight,
            conv1d_bias,
            x_dbl,
            x_proj_weight,
            delta_proj_weight,
            conv1d_out,
            delta,
            A,
            B,
            C,
            D,
            delta_bias,
            scan_intermediates,
            out,
        ) = ctx.saved_tensors
        batch, L = xz.shape[0], xz.shape[-1]
        delta_rank = delta_proj_weight.shape[1]
        d_state = A.shape[-1] * (1 if not A.is_complex() else 2)

        # x, z = xz.chunk(2, dim=1)
        assert ctx.d_inner % ctx.tp_pg.size() == 0
        x, z = xz.view(batch, ctx.d_inner // ctx.tp_pg.size(), 2, L).chunk(2, dim=2)
        x = x.squeeze(2)
        z = z.squeeze(2)

        if ctx.checkpoint_lvl == 1:
            conv1d_out = causal_conv1d_cuda.causal_conv1d_fwd(x, conv1d_weight, conv1d_bias, None, True)
            delta = rearrange(delta_proj_weight @ x_dbl[:, :delta_rank].t(), "d (b l) -> b d l", l=L)
        # The kernel supports passing in a pre-allocated dz (e.g., in case we want to fuse the
        # backward of selective_scan_cuda with the backward of chunk).
        dxz = torch.empty_like(xz)  # (batch, dim, seqlen)

        # dx, dz = dxz.chunk(2, dim=1)
        assert ctx.d_inner % ctx.tp_pg.size() == 0
        dx, dz = dxz.view(batch, ctx.d_inner // ctx.tp_pg.size(), 2, L).chunk(2, dim=2)
        dx = dx.squeeze(2)
        dz = dz.squeeze(2)

        dout = rearrange(dout, "b l e -> b e l")

        if dout.stride(-1) != 1:
            dout = dout.contiguous()

        (dconv1d_out, ddelta, dA, dB, dC, dD, ddelta_bias, dz, out_z,) = selective_scan_cuda.bwd(
            conv1d_out,
            delta,
            A,
            B,
            C,
            D,
            z,
            delta_bias,
            dout,
            scan_intermediates,
            out,
            dz,
            ctx.delta_softplus,
            True,  # option to recompute out_z
        )

        dD = dD if D is not None else None
        dx_dbl = torch.empty_like(x_dbl)
        dB_proj_bias = None
        if ctx.is_variable_B:
            if not A.is_complex():
                dB = rearrange(dB, "b 1 dstate l -> (b l) dstate").contiguous()
            else:
                dB = rearrange(dB, "b 1 dstate (l two) -> (b l) (dstate two)", two=2).contiguous()
            dB_proj_bias = dB.sum(0) if not ctx.B_proj_bias_is_None else None
            dx_dbl[:, delta_rank : delta_rank + d_state] = dB  # (bl d)
            dB = None
        dC_proj_bias = None
        if ctx.is_variable_C:
            if not A.is_complex():
                dC = rearrange(dC, "b 1 dstate l -> (b l) dstate").contiguous()
            else:
                dC = rearrange(dC, "b 1 dstate (l two) -> (b l) (dstate two)", two=2).contiguous()
            dC_proj_bias = dC.sum(0) if not ctx.C_proj_bias_is_None else None
            dx_dbl[:, -d_state:] = dC  # (bl d)
            dC = None
        ddelta = rearrange(ddelta, "b d l -> d (b l)")
        ddelta_proj_weight = torch.einsum("dB,Br->dr", ddelta, x_dbl[:, :delta_rank])
        dx_dbl[:, :delta_rank] = torch.einsum("dB,dr->Br", ddelta, delta_proj_weight)
        dconv1d_out = rearrange(dconv1d_out, "b d l -> d (b l)")
        dx_proj_weight = torch.einsum("Br,Bd->rd", dx_dbl, rearrange(conv1d_out, "b d l -> (b l) d"))
        dconv1d_out = torch.addmm(dconv1d_out, x_proj_weight.t(), dx_dbl.t(), out=dconv1d_out)
        dconv1d_out = rearrange(dconv1d_out, "d (b l) -> b d l", b=x.shape[0], l=x.shape[-1])
        # The kernel supports passing in a pre-allocated dx (e.g., in case we want to fuse the
        # backward of conv1d with the backward of chunk).
        dx, dconv1d_weight, dconv1d_bias = causal_conv1d_cuda.causal_conv1d_bwd(
            x, conv1d_weight, conv1d_bias, dconv1d_out, None, dx, True
        )
        dconv1d_bias = dconv1d_bias if conv1d_bias is not None else None
        dconv1d_weight = rearrange(dconv1d_weight, "d w -> d 1 w")
        return (
            None,  # d_inner
            None,  # tp_pg
            dxz,
            dconv1d_weight,
            dconv1d_bias,
            dx_proj_weight,
            ddelta_proj_weight,
            dA,
            dB,
            dC,
            dD,
            ddelta_bias if delta_bias is not None else None,
            dB_proj_bias,
            dC_proj_bias,
            None,
        )


def mamba_inner_fn(
    d_inner,
    tp_pg,
    xz,
    conv1d_weight,
    conv1d_bias,
    x_proj_weight,
    delta_proj_weight,
    A,
    B=None,
    C=None,
    D=None,
    delta_bias=None,
    B_proj_bias=None,
    C_proj_bias=None,
    delta_softplus=True,
):
    # [FUNCTION DOC]
    # Behavior: functional wrapper calling MambaInnerFn.apply with the given arguments and default checkpoint_lvl 1.
    # Inputs/Outputs: as MambaInnerFn.forward. Exceptions: None explicitly raised. Side effects: CUDA kernels.
    return MambaInnerFn.apply(
        d_inner,
        tp_pg,
        xz,
        conv1d_weight,
        conv1d_bias,
        x_proj_weight,
        delta_proj_weight,
        A,
        B,
        C,
        D,
        delta_bias,
        B_proj_bias,
        C_proj_bias,
        delta_softplus,
    )


def mamba_inner_ref(
    xz,
    conv1d_weight,
    conv1d_bias,
    x_proj_weight,
    delta_proj_weight,
    out_proj_weight,
    out_proj_bias,
    A,
    B=None,
    C=None,
    D=None,
    delta_bias=None,
    B_proj_bias=None,
    C_proj_bias=None,
    delta_softplus=True,
):
    # [FUNCTION DOC]
    # Behavior: unfused reference of the full Mamba inner path including out_proj: chunks xz into x and z halves
    # along channels, applies causal_conv1d_fn with silu, computes x_dbl, delta, and variable B/C (with biases,
    # reshaped to (b, n, l)), runs selective_scan_fn with z gating and softplus, and applies out_proj linear to
    # the (b, l, d) result.
    # Inputs: xz (b, 2d, l); conv1d_weight; conv1d_bias; x_proj_weight; delta_proj_weight; out_proj_weight;
    # out_proj_bias; A; B, C, D, delta_bias, B_proj_bias, C_proj_bias (optional); delta_softplus (default True).
    # Outputs: tensor (b, l, out features). Exceptions: None explicitly raised. Side effects: CUDA kernels.
    # KNOWN ISSUE: delta_softplus argument is ignored (softplus always applied); uses the concatenated-halves xz
    # layout rather than the interleaved layout of MambaInnerFn, and has no TP support.
    L = xz.shape[-1]
    delta_rank = delta_proj_weight.shape[1]
    d_state = A.shape[-1] * (1 if not A.is_complex() else 2)
    x, z = xz.chunk(2, dim=1)
    x = causal_conv1d_fn(x, rearrange(conv1d_weight, "d 1 w -> d w"), conv1d_bias, "silu")
    # We're being very careful here about the layout, to avoid extra transposes.
    # We want delta to have d as the slowest moving dimension
    # and L as the fastest moving dimension, since those are what the ssm_scan kernel expects.
    x_dbl = F.linear(rearrange(x, "b d l -> (b l) d"), x_proj_weight)  # (bl d)
    delta = delta_proj_weight @ x_dbl[:, :delta_rank].t()
    delta = rearrange(delta, "d (b l) -> b d l", l=L)
    if B is None:  # variable B
        B = x_dbl[:, delta_rank : delta_rank + d_state]  # (bl d)
        if B_proj_bias is not None:
            B = B + B_proj_bias.to(dtype=B.dtype)
        if not A.is_complex():
            B = rearrange(B, "(b l) dstate -> b dstate l", l=L).contiguous()
        else:
            B = rearrange(B, "(b l) (dstate two) -> b dstate (l two)", l=L, two=2).contiguous()
    if C is None:  # variable B
        C = x_dbl[:, -d_state:]  # (bl d)
        if C_proj_bias is not None:
            C = C + C_proj_bias.to(dtype=C.dtype)
        if not A.is_complex():
            C = rearrange(C, "(b l) dstate -> b dstate l", l=L).contiguous()
        else:
            C = rearrange(C, "(b l) (dstate two) -> b dstate (l two)", l=L, two=2).contiguous()
    y = selective_scan_fn(x, delta, A, B, C, D, z=z, delta_bias=delta_bias, delta_softplus=True)
    return F.linear(rearrange(y, "b d l -> b l d"), out_proj_weight, out_proj_bias)
