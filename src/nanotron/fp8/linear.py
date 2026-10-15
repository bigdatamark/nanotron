# [MODULE DOC]
# Purpose: FP8 linear layer (FP8Linear) and its custom autograd function performing forward and
# backward matmuls with the Transformer Engine FP8 kernel.
# Contents: FP8LinearMeta (typed dict of quantization metadata), FP8Linear, _FP8Matmul.
# Import-time side effects: imports transformer_engine.
from typing import Optional, Tuple, TypedDict, Union

import torch
import torch.nn.functional as F
import transformer_engine as te  # noqa
from torch import nn

from nanotron.fp8.constants import INITIAL_AMAX, INITIAL_SCALING_FACTOR
from nanotron.fp8.dtypes import DTypes
from nanotron.fp8.kernel import fp8_matmul_kernel
from nanotron.fp8.meta import FP8Meta
from nanotron.fp8.parameter import FP8Parameter
from nanotron.fp8.tensor import FP8Tensor, update_scaling_factor


class FP8LinearMeta(TypedDict):
    # [CLASS DOC]
    # Purpose: typed dictionary holding FP8Meta for input_grad, weight_grad and output_grad.
    # Used as the fp8_meta attribute of FP8Linear. No behavior.
    """FP8 metadata for FP8Linear."""

    input_grad: FP8Meta
    weight_grad: FP8Meta
    output_grad: FP8Meta


class FP8Linear(nn.Linear):
    # [CLASS DOC]
    # Purpose: drop-in replacement for nn.Linear whose weight is stored as an FP8Parameter (E4M3) when on
    # a GPU, and whose matmul runs through _FP8Matmul.
    # Key state: weight (FP8Parameter on GPU, ordinary parameter on CPU), bias (high precision),
    # fp8_meta (FP8LinearMeta; only set when the weight is on GPU).
    # Inherits nn.Linear for parameter creation and the CPU fallback path.
    def __init__(self, in_features: int, out_features: int, bias: bool = True, device: Optional[torch.device] = None):
        # [FUNCTION DOC]
        # Behavior: builds a normal nn.Linear; if the weight is not on CPU, wraps it as an E4M3 FP8Parameter,
        # computes initial scaling factors for E4M3 and E5M2 from INITIAL_AMAX and INITIAL_SCALING_FACTOR, and
        # builds fp8_meta with E4M3 meta for input_grad and weight_grad and E5M2 meta for output_grad (amax 1).
        # Inputs: in_features, out_features - ints; bias - whether to create a bias (default True);
        # device - optional torch device (default None, i.e. CPU).
        # Outputs: None. Exceptions: None explicitly raised (FP8Parameter asserts may fire).
        # Side effects: allocates parameters, quantizes the weight on GPU.
        # KNOWN ISSUE: the fp8_meta dictionary is never used by _FP8Matmul's forward or backward.
        super().__init__(in_features, out_features, bias, device)
        # TODO(xrsrke): add device, and 2 fp8 dtypes
        if self.weight.device != torch.device("cpu"):
            self.weight = FP8Parameter(self.weight, dtype=DTypes.FP8E4M3)

            # NOTE: quantization metadata for input gradients, weight gradients, and output gradients
            # TODO(xrsrke): don't fixed this
            fp8e4m3_scale = update_scaling_factor(
                amax=torch.tensor(INITIAL_AMAX, dtype=torch.float32),
                scaling_factor=torch.tensor(INITIAL_SCALING_FACTOR),
                dtype=DTypes.FP8E4M3,
            )
            fp8e5m2_scale = update_scaling_factor(
                amax=torch.tensor(INITIAL_AMAX, dtype=torch.float32),
                scaling_factor=torch.tensor(INITIAL_SCALING_FACTOR, dtype=torch.float32),
                dtype=DTypes.FP8E5M2,
            )
            self.fp8_meta: FP8LinearMeta = {
                # kfloat8_e4m3
                "input_grad": FP8Meta(amax=1, dtype=DTypes.FP8E4M3, scale=fp8e4m3_scale),
                "weight_grad": FP8Meta(amax=1, dtype=DTypes.FP8E4M3, scale=fp8e4m3_scale),
                # kfloat8_e5m2
                "output_grad": FP8Meta(amax=1, dtype=DTypes.FP8E5M2, scale=fp8e5m2_scale),
            }

    def forward(self, input: Union[FP8Tensor, torch.Tensor]) -> torch.Tensor:
        # [FUNCTION DOC]
        # Behavior: if the input or weight is on CPU, falls back to a standard linear. Otherwise creates an
        # empty phony tensor requiring grad (so autograd calls backward even though weight does not require
        # grad), runs _FP8Matmul, then adds the bias in high precision if present.
        # Inputs: input - torch tensor or FP8Tensor of shape (batch, in_features).
        # Outputs: high-precision tensor of the linear output.
        # Exceptions: None explicitly raised. Side effects: launches CUDA kernels.
        # NOTE: only do fp8 kernel if both input and weight are on CUDA device
        if input.device == torch.device("cpu") or self.weight.device == torch.device("cpu"):
            return F.linear(input, self.weight, self.bias)

        # NOTE: just a phony tensor to make pytorch trigger the backward pass
        # because weight and bias's requires_grad are set to False
        # so that we can compute the gradients using the fp8 kernels by ourselves
        phony = torch.empty(0, device=input.device, requires_grad=True)
        output, _ = _FP8Matmul.apply(input, self.weight, self.fp8_meta, phony)

        # TODO(xrsrke): add support for adding bias in fp8
        # TODO(xrsrke): support return an fp8 tensor as output
        # since we will quantize it back to FP8 anyway in the next linear
        output = output if self.bias is None else output + self.bias
        return output


class _FP8Matmul(torch.autograd.Function):
    # [CLASS DOC]
    # HELPER: custom autograd function implementing the FP8 matmul for FP8Linear.
    # Purpose: forward quantizes inputs to E4M3 and runs the FP8 GEMM; backward quantizes grad to E5M2,
    # computes input and weight gradients with FP8 GEMMs and writes the weight gradient directly.
    @staticmethod
    @torch.no_grad()
    def forward(
        ctx, input: FP8Tensor, weight: FP8Tensor, fp8_meta: FP8LinearMeta, phony: torch.Tensor
    ) -> torch.Tensor:
        # [FUNCTION DOC]
        # Behavior: if input is a plain tensor, quantizes it to E4M3 FP8Tensor; saves input and weight for
        # backward and stores fp8_meta on ctx; calls fp8_matmul_kernel with the weight's FP8 data transposed
        # and the input not transposed.
        # Inputs: ctx - autograd context; input - tensor or FP8Tensor; weight - FP8Parameter; fp8_meta -
        # FP8LinearMeta; phony - empty tensor requiring grad.
        # Outputs: tuple of (float32 output tensor, phony).
        # Exceptions: None explicitly raised. Side effects: mutates ctx; CUDA kernels.
        if type(input) == torch.Tensor:
            input = FP8Tensor(input, dtype=DTypes.FP8E4M3)

        ctx.save_for_backward(input, weight)
        ctx.fp8_meta = fp8_meta

        # NOTE: pass FP8Tensor instead of FP8Parameter
        output = fp8_matmul_kernel(
            mat_a=weight.data, transpose_a=True, mat_b=input, transpose_b=False, use_split_accumulator=False
        )

        return output, phony

    @staticmethod
    @torch.no_grad()
    def backward(ctx, grad_output: torch.Tensor, grad_phony: torch.Tensor) -> Tuple[torch.Tensor, None, None, None]:
        # [FUNCTION DOC]
        # Behavior: retrieves saved input and weight; if grad_output is a plain tensor, replaces it with a
        # contiguous tensor of ones of the same shape and quantizes it to E5M2. Computes grad_input from
        # grad_output and weight (both transposed per flags) and grad_weight from input and grad_output, both
        # with split accumulation, assigns grad_weight to weight.grad.
        # Inputs: ctx; grad_output - gradient of the output; grad_phony - gradient for phony (ignored).
        # Outputs: tuple (grad_input, None, None, None).
        # Exceptions: None explicitly raised.
        # Side effects: sets weight.grad directly (bypassing autograd accumulation); CUDA kernels.
        # KNOWN ISSUE: the real upstream gradient is discarded and replaced by a tensor of ones, so gradients are only correct for a sum-of-outputs loss (looks like leftover debugging code).
        # KNOWN ISSUE: weight.grad is overwritten rather than accumulated, so gradient accumulation across micro-batches is broken.
        """
        ∂L/∂X = ∂L/∂Y @ Wᵀ
        ∂L/∂W = Xᵀ @ ∂L/∂Y
        Source: https://web.eecs.umich.edu/~justincj/teaching/eecs442/notes/linear-backprop.html
        """
        # TODO(xrsrke): investigate how does grad_output.contiguous() affect the outputs
        input, weight = ctx.saved_tensors

        if type(grad_output) == torch.Tensor:
            grad_output = torch.ones_like(grad_output)
            grad_output = grad_output.contiguous()
            grad_output = FP8Tensor(grad_output, dtype=DTypes.FP8E5M2)

        grad_input = fp8_matmul_kernel(
            mat_a=grad_output, transpose_a=True, mat_b=weight, transpose_b=True, use_split_accumulator=True
        )
        grad_weight = fp8_matmul_kernel(
            mat_a=input, transpose_a=False, mat_b=grad_output, transpose_b=False, use_split_accumulator=True
        )
        weight.grad = grad_weight

        return grad_input, None, None, None
