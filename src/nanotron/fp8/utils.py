# [MODULE DOC]
# Purpose: FP8 hardware availability check. Import-time side effects: imports transformer_engine.
import torch
import transformer_engine as te  # noqa

from nanotron.fp8.constants import FP8_GPU_NAMES


def is_fp8_available() -> bool:
    # [FUNCTION DOC]
    # Behavior: if CUDA is available, gets the current device's name in lowercase and returns True if any
    # entry of FP8_GPU_NAMES is a substring of it; otherwise returns False.
    # Inputs: None. Outputs: bool. Exceptions: None explicitly raised. Side effects: may initialize CUDA.
    """Check if FP8 is available on the current device."""
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(torch.cuda.current_device()).lower()
        return any(gpu_name in device_name for gpu_name in FP8_GPU_NAMES)
    else:
        return False
