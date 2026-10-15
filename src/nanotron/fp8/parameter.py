# [MODULE DOC]
# Purpose: FP8Parameter, an nn.Parameter subclass whose actual storage is an FP8Tensor, letting
# autograd treat it as a float parameter while kernels use the quantized data.
# Import-time side effects: None beyond imports.
import torch
from torch import nn

from nanotron.fp8.constants import FP8_DTYPES
from nanotron.fp8.dtypes import DTypes
from nanotron.fp8.meta import FP8Meta
from nanotron.fp8.tensor import FP8Tensor


class FP8Parameter(nn.Parameter):
    # [CLASS DOC]
    # Purpose: parameter wrapper around a high-precision tensor that keeps a quantized FP8Tensor in _data.
    # Key state: _data (FP8Tensor, exposed via the data property), fp8_meta (from the FP8Tensor).
    # Invariant: the wrapper itself is created from the original float tensor (so requires_grad is allowed).
    """
    A custom FP8 parameter class that allows gradients
    to flow into FP8 tensors (which are integer tensors).
    """

    def __new__(cls, data: torch.Tensor, dtype: DTypes, requires_grad: bool = True) -> nn.Parameter:
        # [FUNCTION DOC]
        # Behavior: validates input, then under no_grad creates the parameter subclass from the original
        # float data and stores an FP8Tensor quantization of it in _data.
        # Inputs: data - non-FP8 CUDA tensor; dtype - DTypes target format; requires_grad - default True.
        # Outputs: the new FP8Parameter.
        # Exceptions: AssertionError if data is not a tensor, is already an FP8 storage dtype, or is on CPU.
        # Side effects: allocates quantized CUDA memory.
        assert isinstance(data, torch.Tensor), "data must be a tensor"
        assert data.dtype not in FP8_DTYPES, "Currently only support turn a non-fp8 tensor to an fp8 parameter"
        assert data.device != torch.device("cpu"), "FP8Parameter only supports CUDA tensors"
        # TODO(xrsrke): if the tensor is on cpu, then bypass quantization

        with torch.no_grad():
            # TODO(xrsrke): support take an FP8 Tensor as data
            # currently we can't only quantize a tensor to FP8 after the parameter is created
            # because it raise "Only Tensors of floating point and complex dtype can require gradients"
            self = torch.Tensor._make_subclass(cls, data, requires_grad)
            self._data = FP8Tensor(data, dtype=dtype)
        return self

    @property
    def data(self) -> FP8Tensor:
        # [FUNCTION DOC]
        # Behavior: returns the stored FP8Tensor (_data). Outputs: FP8Tensor. Exceptions/Side effects: None.
        return self._data

    @data.setter
    def data(self, data: FP8Tensor):
        # [FUNCTION DOC]
        # Behavior: setter; replaces the stored FP8Tensor. Inputs: data - FP8Tensor. Outputs: None.
        # Exceptions: None. Side effects: mutates self._data.
        self._data = data

    @property
    def fp8_meta(self) -> FP8Meta:
        # [FUNCTION DOC]
        # Behavior: returns the FP8Meta of the stored data. Outputs: FP8Meta. Exceptions/Side effects: None.
        return self.data.fp8_meta

    def __repr__(self) -> str:
        # [FUNCTION DOC]
        # Behavior: returns a string with the data, its fp8_meta and requires_grad.
        # Outputs: str. Exceptions: None explicitly raised. Side effects: None.
        # KNOWN ISSUE: the returned string is missing its closing parenthesis.
        # KNOWN ISSUE: formatting the data calls FP8Tensor.__repr__, which can recurse infinitely (see FP8Tensor.__repr__).
        return f"FP8Parameter({self.data}, fp8_meta={self.fp8_meta}, requires_grad={self.requires_grad}"
