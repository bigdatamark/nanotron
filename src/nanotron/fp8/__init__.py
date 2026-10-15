# [PACKAGE DOC]
# Purpose: nanotron.fp8 package - experimental FP8 training primitives built on NVIDIA Transformer Engine.
# Exports: DTypes, FP8Linear, FP8Parameter, FP8Tensor.
# Import-time side effects: imports the submodules (which themselves import transformer_engine and
# fail hard if it is missing); then tries to import transformer_engine and transformer_engine_extensions
# again and emits a warnings.warn message if that import fails.
# KNOWN ISSUE: the guarded import comes after unguarded submodule imports that already require transformer_engine, so the warning path is effectively unreachable (an ImportError is raised first).
import warnings

from nanotron.fp8.dtypes import DTypes  # noqa
from nanotron.fp8.linear import FP8Linear  # noqa
from nanotron.fp8.parameter import FP8Parameter  # noqa
from nanotron.fp8.tensor import FP8Tensor  # noqa

try:
    import transformer_engine as te  # noqa
    import transformer_engine_extensions as tex  # noqa
except ImportError:
    warnings.warn("Please install Transformer engine for FP8 training!")
