# [MODULE DOC]
# Purpose: configuration dataclasses for the Mamba example in nanotron. Extends the core nanotron Config
# so YAML configs can describe a Mamba model (MambaModelConfig) and its initialization (MambaInit).
# Contents: MambaInit, ModelArgs (Mamba-specific variant of nanotron's ModelArgs), MambaConfig (top-level
# config), MambaModelConfig (architecture hyperparameters). Re-exports ExistingCheckpointInit via import
# (trainer.py imports it from here). No import-time side effects.
from dataclasses import dataclass
from typing import Optional, Union

import torch
from nanotron.config import Config, ExistingCheckpointInit, NanotronConfigs
from nanotron.config.utils_config import cast_str_to_torch_dtype


@dataclass
class MambaInit:
    # [CLASS DOC]
    # Dataclass describing random initialization for Mamba weights (used by MambaForTraining.init_model_randomly).
    # Attributes: initializer_range (float, default 0.02, std for embedding normal init); rescale_prenorm_residual
    # (bool, default True, divide out_proj weights by sqrt of n_residuals_per_layer times number of layers);
    # n_residuals_per_layer (int, default 1; 2 if blocks had an MLP).
    initializer_range: float = 0.02
    rescale_prenorm_residual: bool = True
    n_residuals_per_layer: int = 1  # Change to 2 if we have MLP


@dataclass
class ModelArgs:
    # [CLASS DOC]
    # Dataclass with model-related arguments for MambaConfig.
    # Attributes: model_config (nanotron model config, e.g. MambaModelConfig); init_method (MambaInit or
    # ExistingCheckpointInit); dtype (torch dtype, default None meaning bfloat16; strings are cast);
    # make_vocab_size_divisible_by (int, default 1); ddp_bucket_cap_mb (int, default 25).
    # Invariant after construction: dtype is a torch.dtype.
    """Arguments related to model architecture"""

    model_config: NanotronConfigs
    init_method: Union[MambaInit, ExistingCheckpointInit]
    dtype: Optional[torch.dtype] = None
    make_vocab_size_divisible_by: int = 1
    ddp_bucket_cap_mb: int = 25

    def __post_init__(self):
        # [FUNCTION DOC]
        # Behavior: sets dtype to bfloat16 when None; if dtype is a string converts it with cast_str_to_torch_dtype.
        # Inputs: self. Outputs: None. Exceptions: whatever cast_str_to_torch_dtype raises for unknown names.
        # Side effects: mutates self.dtype.
        if self.dtype is None:
            self.dtype = torch.bfloat16
        if isinstance(self.dtype, str):
            self.dtype = cast_str_to_torch_dtype(self.dtype)

        # if self.model_config.max_position_embeddings is None:
        #     self.model_config.max_position_embeddings = 0


@dataclass(kw_only=True)  # pylint: disable=unexpected-keyword-arg
class MambaConfig(Config):
    # [CLASS DOC]
    # Top-level Mamba training config: identical to nanotron Config (keyword-only dataclass) but with the
    # model field typed as this module's ModelArgs so YAML loading builds Mamba-specific model arguments.
    """Main configuration class"""

    model: ModelArgs


@dataclass
class MambaModelConfig:
    # [CLASS DOC]
    # Dataclass of Mamba architecture hyperparameters, round-tripped through YAML (types must stay coherent).
    # Attributes: is_mamba_config (bool marker, True, used to detect Mamba models, e.g. in trainer tying logic);
    # d_model (int, 2560); num_hidden_layers (int, 64); vocab_size (int, 50277); ssm_cfg (optional dict of
    # Mamba mixer kwargs such as d_state, d_conv, expand, dt_rank); rms_norm (bool, use RMSNorm instead of
    # LayerNorm); fused_add_norm (bool, use Triton fused add+norm); residual_in_fp32 (bool);
    # pad_vocab_size_multiple (int, 8); dtype (string, "float32", dtype for SSM internals); rms_norm_eps (float, 1e-5);
    # pad_token_id (optional int).
    """Configuration for a Mamba model

    Be careful on having a coherent typing as we use it to reconstruct the model from yaml
    """

    is_mamba_config: bool = True  # We use this help differentiate models in yaml/python conversion
    d_model: int = 2560
    num_hidden_layers: int = 64
    vocab_size: int = 50277
    ssm_cfg: Optional[dict] = None
    rms_norm: bool = True
    fused_add_norm: bool = True
    residual_in_fp32: bool = True
    pad_vocab_size_multiple: int = 8
    # ==== Custom ======
    dtype: str = "float32"
    rms_norm_eps: float = 1e-5
    pad_token_id: Optional[int] = None
