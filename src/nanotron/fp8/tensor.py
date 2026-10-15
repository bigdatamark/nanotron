# [MODULE DOC]
# Purpose: FP8Tensor (a quantized tensor subclass) and the conversion helpers between torch tensors
# and Transformer Engine FP8 storage, plus the scaling-factor update rule (adapted from MS-AMP).
# Contents: FP8Tensor, convert_torch_dtype_to_te_dtype, convert_tensor_to_fp8, convert_tensor_from_fp8,
# update_scaling_factor. Import-time side effects: imports transformer_engine.
import torch
import transformer_engine as te  # noqa
import transformer_engine_extensions as tex

from nanotron.fp8.constants import DTYPE_TO_FP8_MAX, FP8_DTYPES, INITIAL_SCALING_FACTOR
from nanotron.fp8.dtypes import DTypes


class FP8Tensor(torch.Tensor):
    # [CLASS DOC]
    # Purpose: torch.Tensor subclass whose storage holds FP8-quantized values (int8/uint8 bytes) and that
    # carries an fp8_meta attribute (FP8Meta) with amax, scale and format.
    """FP8 Tensor."""

    def __new__(cls, tensor: torch.Tensor, dtype: DTypes) -> torch.Tensor:
        # [FUNCTION DOC]
        # Behavior: validates input; computes amax as the absolute max of the tensor; computes a scale with
        # update_scaling_factor starting from INITIAL_SCALING_FACTOR; builds FP8Meta; casts the tensor to FP8
        # via convert_tensor_to_fp8; wraps the result as an FP8Tensor and attaches fp8_meta.
        # Inputs: tensor - non-FP8 CUDA tensor (float32 expected, since amax must be float32); dtype - DTypes.
        # Outputs: new FP8Tensor.
        # Exceptions: AssertionError if tensor is not a tensor, already has an FP8 storage dtype, is on CPU,
        # dtype is not a DTypes member, or (inside update_scaling_factor) amax is not float32.
        # Side effects: allocates CUDA memory; launches CUDA kernels.
        assert isinstance(tensor, torch.Tensor), "tensor must be a tensor"
        assert tensor.dtype not in FP8_DTYPES, "The tensor already quantized to FP8"
        
        # TODO(xrsrke): there is a circular import issue
        # between tensor.py and meta.py fix this
        from nanotron.fp8.meta import FP8Meta

        # TODO(xrsrke): if the tensor is on cpu, then bypass the quantization
        # because the current kernels only support gpu tensor
        assert tensor.device != torch.device("cpu"), "FP8Tensor only supports CUDA device"
        assert isinstance(dtype, DTypes)

        amax = tensor.abs().max().clone()
        scale = update_scaling_factor(amax, torch.tensor(INITIAL_SCALING_FACTOR, dtype=torch.float32), dtype)
        fp8_meta = FP8Meta(amax, scale, dtype)
        fp8_tensor = convert_tensor_to_fp8(tensor, fp8_meta)

        # TODO(xrsrke): move update inverse scaling to FP8Meta's initialization
        obj = torch.Tensor._make_subclass(cls, fp8_tensor)
        obj.fp8_meta = fp8_meta
        return obj

    def __repr__(self) -> str:
        # [FUNCTION DOC]
        # Behavior: returns a string embedding the tensor itself and its fp8_meta.
        # Outputs: str. Exceptions: None explicitly raised. Side effects: None.
        # KNOWN ISSUE: formatting self inside its own __repr__ likely falls back to str and then this __repr__ again, causing infinite recursion.
        return f"FP8Tensor({self}, fp8_meta={self.fp8_meta})"


def convert_torch_dtype_to_te_dtype(dtype: torch.dtype) -> tex.DType:
    # [FUNCTION DOC]
    # HELPER: maps a dtype to the Transformer Engine DType enum.
    # Behavior: looks up the TE dtype name in a fixed mapping (int32, float32, float16, bfloat16 torch
    # dtypes and the DTypes FP8E4M3, FP8E5M2, KFLOAT16 members) and returns the matching tex.DType attribute.
    # Inputs: dtype - a torch dtype or DTypes member. Outputs: tex.DType.
    # Exceptions: KeyError for unmapped dtypes. Side effects: None.
    # NOTE: transformer engine maintains it own dtype mapping
    # so we need to manually map torch dtypes to TE dtypes
    TORCH_DTYPE_TE_DTYPE_NAME_MAPPING = {
        torch.int32: "kInt32",
        torch.float32: "kFloat32",
        torch.float16: "kFloat16",
        torch.bfloat16: "kBFloat16",
        # torch.fp8e5m2: "kFloat8E5M2",
        # torch.fp8e4m3: "kFloat8E4M3",
        # torch.int8: "kFloat8E5M2",
        # torch.uint8: "kFloat8E4M3",
        DTypes.FP8E4M3: "kFloat8E4M3",
        DTypes.FP8E5M2: "kFloat8E5M2",
        DTypes.KFLOAT16: "kFloat16",
    }
    return getattr(tex.DType, TORCH_DTYPE_TE_DTYPE_NAME_MAPPING[dtype])


# TODO(xrsrke): add type hint for meta after fixing
# circular import between tensor.py and meta.py
def convert_tensor_to_fp8(tensor: torch.Tensor, meta) -> FP8Tensor:
    # [FUNCTION DOC]
    # HELPER: quantizes a tensor using TE.
    # Behavior: resolves the TE dtype from meta.dtype and calls tex.cast_to_fp8 with meta.scale, meta.amax
    # and a fixed inverse scale of 1.0 on the tensor's device.
    # Inputs: tensor - high precision CUDA tensor; meta - FP8Meta.
    # Outputs: raw quantized tensor (uint8/int8 storage) as returned by TE (not wrapped as FP8Tensor).
    # Exceptions: None explicitly raised. Side effects: CUDA kernel; TE may update amax in place.
    # KNOWN ISSUE: annotated as returning FP8Tensor but returns the plain TE output tensor.
    te_dtype = convert_torch_dtype_to_te_dtype(meta.dtype)
    # TODO(xrsrke): after casting to fp8, update the scaling factor
    # TODO(xrsrke): it's weird that TE only take inverse_scale equal to 1
    inverse_scale = torch.tensor(1.0, device=tensor.device, dtype=torch.float32)
    return tex.cast_to_fp8(tensor, meta.scale, meta.amax, inverse_scale, te_dtype)


def convert_tensor_from_fp8(tensor: torch.Tensor, meta, dtype: torch.dtype) -> torch.Tensor:
    # [FUNCTION DOC]
    # HELPER: dequantizes an FP8 tensor using TE.
    # Behavior: resolves TE dtypes for meta.dtype and the requested output dtype, calls tex.cast_from_fp8
    # with meta.inverse_scale.
    # Inputs: tensor - quantized tensor; meta - FP8Meta; dtype - target torch dtype.
    # Outputs: high precision tensor of the requested dtype.
    # Exceptions: AssertionError if tensor is not a tensor or dtype is not a torch dtype; KeyError for
    # unmapped dtypes. Side effects: CUDA kernel.
    assert isinstance(tensor, torch.Tensor)
    assert isinstance(dtype, torch.dtype)
    tensor_dtype = convert_torch_dtype_to_te_dtype(meta.dtype)
    output_dtype = convert_torch_dtype_to_te_dtype(dtype)

    return tex.cast_from_fp8(tensor, meta.inverse_scale, tensor_dtype, output_dtype)


def update_scaling_factor(
    amax: torch.Tensor, scaling_factor: torch.Tensor, dtype: DTypes, margin: float = 0
) -> torch.Tensor:
    # [FUNCTION DOC]
    # Behavior: computes a power-of-two scaling factor so amax maps near the format's max value:
    # ratio is fp8_max over amax; exp is floor(log2(ratio)) minus margin; the new factor is 2 raised to
    # the absolute exp, rounded; if amax is not positive or not finite the previous scaling_factor is
    # kept; if exp is negative the reciprocal is used. Computed in a TorchScript inner function.
    # Inputs: amax - float32 tensor; scaling_factor - previous float32 scaling factor tensor; dtype -
    # DTypes target format; margin - extra exponent bits of headroom (default 0).
    # Outputs: float32 tensor with the new scaling factor.
    # Exceptions: AssertionError if amax or scaling_factor is not float32; KeyError for unknown dtype.
    # Side effects: compiles a TorchScript function on each call.
    """
    Update the scaling factor to quantize a tensor to FP8.
    Credits: https://github.com/Azure/MS-AMP/blob/d562f0f0bcfc9b712fa0726b73428753ff1300ab/msamp/common/tensor/meta.py#L39
    """
    assert amax.dtype == torch.float32
    # TODO(xrsrke): can we use lower precision for scaling_factor?
    assert scaling_factor.dtype == torch.float32

    # NOTE: Since fp8_max is a fixed number based on two FP8 data types,
    # we prefer not to take fp8_max in the input arguments.
    fp8_max = torch.tensor(DTYPE_TO_FP8_MAX[dtype], dtype=torch.float32)

    # NOTE: torch.jit only take a concrete value rather than a DTYPE_TO_FP8_MAX[dtype],
    # so we create an inner function to bypass that
    @torch.jit.script
    def _inner(amax: torch.Tensor, fp8_max: torch.Tensor, scaling_factor: torch.Tensor, margin: float):
        # [FUNCTION DOC]
        # HELPER: TorchScript kernel of update_scaling_factor; same math as described above.
        # Inputs: amax, fp8_max, scaling_factor tensors and margin float. Outputs: new scaling factor tensor.
        # Exceptions/Side effects: None.
        # NOTE: calculate the number of bits to shift the exponent
        ratio = fp8_max / amax
        exp = torch.floor(torch.log2(ratio)) - margin
        new_scaling_factor = torch.round(torch.pow(2, torch.abs(exp)))
        new_scaling_factor = torch.where(amax > 0.0, new_scaling_factor, scaling_factor)
        new_scaling_factor = torch.where(torch.isfinite(amax), new_scaling_factor, scaling_factor)
        new_scaling_factor = torch.where(exp < 0, 1 / new_scaling_factor, new_scaling_factor)
        return new_scaling_factor

    return _inner(amax, fp8_max, scaling_factor, margin)
