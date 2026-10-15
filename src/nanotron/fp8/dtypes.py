# [MODULE DOC]
# Purpose: defines the DTypes enum naming the low-precision formats used by nanotron.fp8.
# Import-time side effects: None.
from enum import Enum, auto


class DTypes(Enum):
    # [CLASS DOC]
    # Purpose: enumeration of FP8-related formats: FP8E4M3 (4 exponent bits, 3 mantissa bits; used for
    # weights/inputs), FP8E5M2 (5 exponent, 2 mantissa; used for gradients) and KFLOAT16.
    # Members are auto-numbered. Used as keys in DTYPE_TO_FP8_MAX and the Transformer Engine dtype mapping.
    FP8E4M3 = auto()
    FP8E5M2 = auto()
    KFLOAT16 = auto()
