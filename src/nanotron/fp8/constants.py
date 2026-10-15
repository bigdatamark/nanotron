# [MODULE DOC]
# Purpose: constants for the FP8 subsystem.
# Contents: FP8_GPU_NAMES (lowercase substrings of GPU names that support FP8: h100, rtx 4090);
# INITIAL_AMAX and INITIAL_SCALING_FACTOR (both 1.0); FP8_DTYPES, FP8E4M3_DTYPE, FP8E5M2_DTYPE,
# which stand in for real FP8 torch dtypes using int8 (e4m3) and uint8 (e5m2) storage;
# DTYPE_TO_FP8_MAX mapping each DTypes member to its maximum representable normal value
# (448 for E4M3, 57344 for E5M2, 65504 for float16).
# Import-time side effects: imports torch. No other side effects.
import torch

from nanotron.fp8.dtypes import DTypes

FP8_GPU_NAMES = ["h100", "rtx 4090"]

INITIAL_AMAX = 1.0
INITIAL_SCALING_FACTOR = 1.0

# FP8_DTYPES = [torch.fp8e4m3, torch.fp8e5m2]
# FP8E4M3_DTYPE = torch.fp8e4m3
# FP8E5M2_DTYPE = torch.fp8e5m2

FP8_DTYPES = [torch.int8, torch.uint8]
FP8E4M3_DTYPE = torch.int8
FP8E5M2_DTYPE = torch.uint8

DTYPE_TO_FP8_MAX = {DTypes.FP8E4M3: 448.0, DTypes.FP8E5M2: 57344.0, DTypes.KFLOAT16: 65504.0}
