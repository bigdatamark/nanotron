# [PACKAGE DOC]
# Purpose: nanotron.models package, holding the model implementations (llama, qwen, starcoder2, ...) and the shared base utilities.
# Exports (re-exported from models.base): DTypeInvariantTensor, NanotronModel, build_model, check_model_has_grad, init_on_device_and_dtype.
# Import-time side effects: imports models.base (and therefore torch, nanotron.distributed and parallel modules); model files are not imported here.
# flake8: noqa
from .base import DTypeInvariantTensor, NanotronModel, build_model, check_model_has_grad, init_on_device_and_dtype
