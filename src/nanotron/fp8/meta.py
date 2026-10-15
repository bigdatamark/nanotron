# [MODULE DOC]
# Purpose: FP8Meta dataclass holding per-tensor quantization metadata (amax, scale, dtype).
# Import-time side effects: imports transformer_engine; imports from tensor.py (circular dependency
# handled by tensor.py importing meta lazily).
from dataclasses import dataclass
from typing import Union

import torch
import transformer_engine as te  # noqa
import transformer_engine_extensions as tex

from nanotron.fp8.constants import DTYPE_TO_FP8_MAX
from nanotron.fp8.tensor import convert_torch_dtype_to_te_dtype


@dataclass
class FP8Meta:
    # [CLASS DOC]
    # Purpose: quantization metadata attached to FP8Tensor / FP8Parameter.
    # Attributes: amax - absolute max of the source tensor (converted to a CUDA tensor after init);
    # scale - float32 scaling factor tensor; dtype - a DTypes member (despite the annotation).
    # Derived properties: te_dtype, fp8_max, inverse_scale.
    # KNOWN ISSUE: dtype is annotated as torch.dtype but callers pass DTypes members, which is what the lookups require.
    """Metadata for FP8Tensor."""

    amax: Union[int, float]
    scale: torch.Tensor

    # TODO(xrsrke): change to Literal[torch.int8, torch.uint8]
    dtype: torch.dtype

    @property
    def te_dtype(self) -> tex.DType:
        # [FUNCTION DOC]
        # Behavior: maps dtype to the Transformer Engine dtype via convert_torch_dtype_to_te_dtype.
        # Inputs: None. Outputs: tex.DType. Exceptions: KeyError for unmapped dtypes. Side effects: None.
        return convert_torch_dtype_to_te_dtype(self.dtype)

    def __post_init__(self):
        # [FUNCTION DOC]
        # Behavior: if amax is not already a tensor, converts it to a tensor on the CUDA device.
        # Inputs/Outputs: None. Exceptions: None explicitly raised (CUDA errors if no GPU).
        # Side effects: mutates self.amax; allocates CUDA memory.
        # NOTE: transformer engine only accepts torch tensors
        self.amax = torch.tensor(self.amax, device="cuda") if not isinstance(self.amax, torch.Tensor) else self.amax

    @property
    def fp8_max(self) -> float:
        # [FUNCTION DOC]
        # Behavior: looks up the maximum normal value of dtype in DTYPE_TO_FP8_MAX.
        # Outputs: float. Exceptions: KeyError for unknown dtype. Side effects: None.
        """Return the maximum normal value for the current dtype."""
        return DTYPE_TO_FP8_MAX[self.dtype]

    @property
    def inverse_scale(self) -> torch.Tensor:
        # [FUNCTION DOC]
        # Behavior: returns the reciprocal of scale (computed fresh each call).
        # Outputs: tensor. Exceptions: None. Side effects: None.
        return 1 / self.scale

    def __repr__(self) -> str:
        # [FUNCTION DOC]
        # Behavior: returns a string showing amax, scale, inverse_scale and dtype.
        # Outputs: str. Exceptions/Side effects: None.
        return f"FP8Meta(amax={self.amax}, scale={self.scale}, inverse_scale={self.inverse_scale}, dtype={self.dtype})"
