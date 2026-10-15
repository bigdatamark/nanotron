# [MODULE DOC]
# Purpose: pytest tests for nanotron.fp8 FP8Parameter construction (requires CUDA and transformer engine).
# Main contents: test_create_fp8_parameter.
import torch
from nanotron.fp8 import DTypes, FP8Parameter, FP8Tensor
from nanotron.fp8.meta import FP8Meta


def test_create_fp8_parameter():
    # [FUNCTION DOC]
    # Behavior: wraps a random 16x16 float32 CUDA tensor into an FP8Parameter with FP8E4M3 and checks that its data is an FP8Tensor, it requires grad, has no grad yet, and both the parameter and its data carry an FP8Meta.
    # Inputs: none.
    # Outputs: None.
    # Exceptions: AssertionError on failure.
    # Side effects: GPU allocation.
    # TODO(xrsrke): test FP8E5M2 format
    # TODO(xrsrke): test take a cpu tensor
    tensor = torch.randn(16, 16, device="cuda", dtype=torch.float32)

    fp8_parameter = FP8Parameter(tensor, DTypes.FP8E4M3)

    assert isinstance(fp8_parameter.data, FP8Tensor)
    assert fp8_parameter.requires_grad is True
    assert fp8_parameter.grad is None
    assert isinstance(fp8_parameter.fp8_meta, FP8Meta)
    assert isinstance(fp8_parameter.data.fp8_meta, FP8Meta)


# TODO(xrsrke): add test for preventing torch autograd do the backward pass
# on a FP8Parameter
