# [MODULE DOC]
# Purpose: Dataclasses describing model architectures and weight initialization for nanotron (serialized to and
# from YAML, so field types must stay coherent).
# Main contents: DEFAULT_ATTENTION_IMPLEMENTATION (flash_attention_2), init configs RandomInit, SpectralMupInit,
# ExistingCheckpointInit, MoEConfig, model configs LlamaConfig, Qwen2Config, Starcoder2Config, and the
# NanotronConfigs union type alias.
# Import-time side effects: imports nanotron.nn.attention (attention registry).
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Optional, Union

from nanotron.config.utils_config import InitScalingMethod
from nanotron.nn.attention import ALL_ATTENTION_FUNCTIONS, AttentionImplementation

# The default attention implementation to use
DEFAULT_ATTENTION_IMPLEMENTATION = "flash_attention_2"


@dataclass
class RandomInit:
    # [CLASS DOC]
    # Purpose: random normal initialization with standard deviation std and a scaling method (default scales by
    # number of layers). Dataclass, no validation.
    std: float
    scaling_method: InitScalingMethod = InitScalingMethod.NUM_LAYERS


@dataclass
class SpectralMupInit:
    # [CLASS DOC]
    # Purpose: selects spectral muP initialization. Attribute use_mup must be True (presence of the section is the
    # switch). Dataclass.
    """This is used to initialize the model with spectral mup. Set it to True to use it."""

    use_mup: bool

    def __post_init__(self):
        # [FUNCTION DOC]
        # Behavior: asserts use_mup is True. Inputs: self. Outputs: None.
        # Exceptions: AssertionError when False. Side effects: None.
        assert self.use_mup, "Remove `use_mup` if you don't want to use it"


@dataclass
class ExistingCheckpointInit:
    # [CLASS DOC]
    # Purpose: initialize model weights from an existing checkpoint path (weights only, no optimizer or scheduler).
    # Attribute: path. Dataclass.
    """This is used to initialize from an already existing model (without optimizer, lr_scheduler...)"""

    path: Path


@dataclass
class MoEConfig:
    # [CLASS DOC]
    # Purpose: Mixture-of-Experts settings: num_experts (8), top_k (2), moe_intermediate_size (1408),
    # shared_expert_intermediate_size (5632), router_aux_loss_coef (0.01), layers (list of layer indices, default
    # a single -1 meaning all layers), enable_shared_expert (False), token_dispatcher_type (alltoall or allgather).
    # Invariants: top_k not above num_experts; dispatcher type valid. Dataclass.
    """Configuration for Mixture of Experts layers"""

    num_experts: int = 8  # Total number of experts
    top_k: int = 2  # Number of experts to route each token to
    moe_intermediate_size: int = 1408  # Intermediate size of the MoE layer
    shared_expert_intermediate_size: int = 5632  # Intermediate size of the shared expert
    router_aux_loss_coef: float = 0.01  # Coefficient for the router auxiliary loss
    layers: List[int] = field(
        default_factory=lambda: [-1]
    )  # Indices of layers that use MoE. -1 means all layers. Default is all layers
    enable_shared_expert: bool = False  # Whether to use a shared expert alongside specialized experts
    token_dispatcher_type: str = "alltoall"  # Communication pattern for MoE ("alltoall" or "allgather")

    def __post_init__(self):
        # [FUNCTION DOC]
        # Behavior: validates top_k and token_dispatcher_type.
        # Inputs: self. Outputs: None. Exceptions: ValueError on invalid values. Side effects: None.
        # Validate the configuration
        if self.top_k > self.num_experts:
            raise ValueError(f"top_k ({self.top_k}) cannot be greater than num_experts ({self.num_experts})")

        if self.token_dispatcher_type not in ["alltoall", "allgather"]:
            raise ValueError(
                f"token_dispatcher_type must be one of ['alltoall', 'allgather'], got {self.token_dispatcher_type}"
            )


@dataclass
class LlamaConfig:
    # [CLASS DOC]
    # Purpose: Llama architecture config (token ids, activation, sizes, heads, optional KV heads for GQA, RMSNorm eps,
    # RoPE theta / scaling / interleaving, tied embeddings, vocab, attention implementation, optional z-loss).
    # is_llama_config marks the type for YAML round-trips. Post-init adds the private flag _is_using_mup (False;
    # ModelArgs may set it). Invariant: num_key_value_heads defaults to num_attention_heads; the attention
    # implementation must be registered. Dataclass.
    """Configuration for a LLAMA model

    Be careful on having a coherent typing as we use it to reconstruct the model from yaml
    """

    bos_token_id: int = 1
    eos_token_id: int = 2
    hidden_act: str = "silu"
    hidden_size: int = 4096
    initializer_range: float = 0.02
    intermediate_size: int = 11008
    is_llama_config: bool = True  # We use this help differentiate models in yaml/python conversion
    max_position_embeddings: int = 2048
    num_attention_heads: int = 32
    num_hidden_layers: int = 32
    attention_bias: bool = False
    num_key_value_heads: Optional[int] = None
    pad_token_id: Optional[int] = None
    pretraining_tp: int = 1
    rms_norm_eps: float = 1e-6
    rope_scaling: Optional[dict] = None
    rope_theta: float = 10000.0
    rope_interleaved: bool = (
        False  # The default value has been True, but for loading Llama3 checkpoints you have to set it to False
    )
    tie_word_embeddings: bool = False
    use_cache: bool = True
    vocab_size: int = 32000
    _attn_implementation: Optional[AttentionImplementation] = DEFAULT_ATTENTION_IMPLEMENTATION
    z_loss_enabled: bool = False  # Z-loss regularization https://www.jmlr.org/papers/volume24/22-1144/22-1144.pdf
    z_loss_coefficient: float = 0.0001  # Default from the paper (10^-4)

    def __post_init__(self):
        # [FUNCTION DOC]
        # Behavior: sets _is_using_mup False, defaults num_key_value_heads to num_attention_heads, and asserts the
        # attention implementation (when not None) exists in ALL_ATTENTION_FUNCTIONS.
        # Inputs: self. Outputs: None. Exceptions: AssertionError on unknown implementation. Side effects: mutates self.
        # NOTE: user don't set self._init_method, ModelArgs will set it
        # then we only pass LlamaConfig around
        self._is_using_mup: bool = False
        # self._init_method: Optional[Union[RandomInit, SpectralMupInit, ExistingCheckpointInit]] = None

        # for backward compatibility
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads

        # Validate that the attention implementation is valid
        if self._attn_implementation is not None:
            assert (
                self._attn_implementation in ALL_ATTENTION_FUNCTIONS
            ), f"Invalid attention implementation: {self._attn_implementation}. Available options are: {ALL_ATTENTION_FUNCTIONS.keys()}"

    @property
    def is_using_mup(self) -> bool:
        # [FUNCTION DOC]
        # Behavior: property returning whether muP initialization is used. Inputs: self. Outputs: bool.
        # Exceptions: None explicitly raised. Side effects: None.
        return self._is_using_mup


@dataclass
class Qwen2Config:
    # [CLASS DOC]
    # Purpose: Qwen2-style architecture config (used by nanotron's main Qwen2 model, also for Llama-like training),
    # with extras over LlamaConfig: RoPE sequence interpolation factor, flex attention mask type, sliding window,
    # no_rope_layer (skip RoPE periodically), fused rotary / RMSNorm / packed QKV / document masking switches,
    # attention-probability logging, ring attention key-head stride, and optional MoEConfig. is_qwen2_config marks
    # the type for YAML. Dataclass.
    """Configuration for a QWEN2 model

    Be careful on having a coherent typing as we use it to reconstruct the model from yaml
    """

    bos_token_id: int = 1
    eos_token_id: int = 2
    hidden_act: str = "silu"
    hidden_size: int = 4096
    initializer_range: float = 0.02
    intermediate_size: int = 11008
    is_qwen2_config: bool = True  # We use this help differentiate models in yaml/python conversion
    max_position_embeddings: int = 2048
    num_attention_heads: int = 32
    num_hidden_layers: int = 32
    num_key_value_heads: Optional[int] = None
    pad_token_id: Optional[int] = None
    pretraining_tp: int = 1
    rms_norm_eps: float = 1e-6
    rope_scaling: Optional[dict] = None
    rope_theta: float = 10000.0
    rope_interleaved: bool = False
    rope_seq_len_interpolation_factor: Optional[float] = None  # if not None, discrete positions will be interpolated by this factor via the trick in https://arxiv.org/abs/2306.15595
    tie_word_embeddings: bool = False
    use_cache: bool = True
    vocab_size: int = 32000
    _attn_implementation: Optional[AttentionImplementation] = DEFAULT_ATTENTION_IMPLEMENTATION
    flex_attention_mask: Optional[str] = None
    attention_bias: bool = False
    sliding_window_size: Optional[int] = None
    z_loss_enabled: bool = False  # Z-loss regularization https://www.jmlr.org/papers/volume24/22-1144/22-1144.pdf
    z_loss_coefficient: float = 0.0001  # Default from the paper (10^-4)
    no_rope_layer: Optional[
        int
    ] = None  # Skip rope every no_rope_layer layers (see https://arxiv.org/abs/2501.18795 https://arxiv.org/abs/2305.19466 and Llama4)
    _fused_rotary_emb: bool = True
    _fused_rms_norm: bool = True
    _use_qkv_packed: bool = True
    _use_doc_masking: bool = False

    log_attn_probs: bool = True # Whether to log the attention probabilities
    ring_attn_heads_k_stride: Optional[int] = None # Stride of the heads in the key tensor for llama3 ring attention

    # MoE configuration
    moe_config: Optional[MoEConfig] = None

    def __post_init__(self):
        # [FUNCTION DOC]
        # Behavior: sets _is_using_mup False; defaults KV heads to attention heads; expands MoE layers [-1] into all
        # layer indices; asserts the attention implementation is registered; with sliding_window_size requires flex,
        # flash_attention_2 or llama3 ring attention; with flex_attention_mask requires flex attention and one of
        # sliding_window, document, sliding_window_document; with no_rope_layer requires num_hidden_layers divisible by
        # it; ring_attn_heads_k_stride must be set exactly when using llama3 ring attention.
        # Inputs: self. Outputs: None.
        # Exceptions: AssertionError on any violated check.
        # Side effects: mutates self and the nested moe_config.
        # KNOWN ISSUE: the no_rope_layer message says it must be a multiple of num_hidden_layers, but the check requires
        # num_hidden_layers to be a multiple of no_rope_layer; the sliding window message omits ring attention, which is
        # also allowed.
        # NOTE: user don't set self._init_method, ModelArgs will set it
        # then we only pass LlamaConfig around
        self._is_using_mup: bool = False
        # self._init_method: Optional[Union[RandomInit, SpectralMupInit, ExistingCheckpointInit]] = None

        # for backward compatibility
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads

        # By default i want all layers to be MoE layers
        if self.moe_config and self.moe_config.layers == [-1]:
            self.moe_config.layers = list(range(self.num_hidden_layers))

        # Validate that the attention implementation is valid
        if self._attn_implementation is not None:
            assert (
                self._attn_implementation in ALL_ATTENTION_FUNCTIONS
            ), f"Invalid attention implementation: {self._attn_implementation}. Available options are: {ALL_ATTENTION_FUNCTIONS.keys()}"

        if self.sliding_window_size is not None:
            assert self._attn_implementation in [
                "flex_attention",
                "flash_attention_2",
                "llama3_ring_attention",
            ], "Sliding window is only supported for Flex Attention and Flash Attention 2"
        if self.flex_attention_mask is not None:
            assert (
                self._attn_implementation == "flex_attention"
            ), "Flex attention mask is only supported for flex attention"
            assert self.flex_attention_mask in [
                "sliding_window",
                "document",
                "sliding_window_document",
            ], "Flex attention mask must be one of ['sliding_window', 'document', 'sliding_window_document']"
        if self.no_rope_layer is not None:
            assert (
                self.num_hidden_layers % self.no_rope_layer == 0
            ), "no_rope_layer must be a multiple of num_hidden_layers"

        if self._attn_implementation == "llama3_ring_attention":
            assert self.ring_attn_heads_k_stride is not None, "ring_attn_heads_k_stride must be specified for llama3 ring attention"
        else:
            assert self.ring_attn_heads_k_stride is None, f"ring_attn_heads_k_stride must be None for non-llama3 ring attention, got attn_implementation={self._attn_implementation}"

    @property
    def is_using_mup(self) -> bool:
        # [FUNCTION DOC]
        # Behavior: property returning the muP flag. Inputs: self. Outputs: bool. Exceptions: None explicitly raised.
        # Side effects: None.
        return self._is_using_mup

    @property
    def is_moe_model(self) -> bool:
        # [FUNCTION DOC]
        # Behavior: property, True when moe_config is set and lists at least one layer. Inputs: self. Outputs: bool.
        # Exceptions: None explicitly raised. Side effects: None.
        """Returns True if the model uses MoE layers"""
        return self.moe_config is not None and len(self.moe_config.layers) > 0


@dataclass
class Starcoder2Config:
    # [CLASS DOC]
    # Purpose: Starcoder2 architecture config (GELU tanh, dropout probabilities, global attention layers, GQA or MQA,
    # layer norm eps, rotary embeddings, sliding window, vocab, attention implementation). is_starcoder2_config marks
    # the type. Several fields are documented as unused. Dataclass.
    """Configuration for a Starcoder2 model

    Be careful on having a coherent typing as we use it to reconstruct the model from yaml
    """

    activation_function: str = "gelu_pytorch_tanh"
    attention_softmax_in_fp32: bool = True  # TODO: not used
    attn_pdrop: float = 0.1
    bos_token_id: int = 49152  # TODO: not used
    embd_pdrop: float = 0.1
    eos_token_id: int = 49152
    global_attn_layers: List[int] = field(default_factory=list)
    grouped_query: bool = False  # GQA
    hidden_size: int = 2048
    initializer_range: float = 0.02  # TODO: not used
    intermediate_size: Optional[int] = None
    is_starcoder2_config: bool = True  # We use this help differentiate models in yaml/python conversion
    layer_norm_epsilon: float = 1e-05
    max_position_embeddings: int = 4096
    multi_query: bool = False  # MQA
    num_attention_heads: int = 16
    num_hidden_layers: int = 24
    num_kv_heads: Optional[int] = None
    resid_pdrop: float = 0.1
    rope_theta: Optional[int] = 10000
    scale_attention_softmax_in_fp32: bool = True
    scale_attn_weights: bool = True
    sliding_window_size: Optional[int] = None
    use_position_embeddings: bool = False  # TODO @nouamane this is not used
    use_rotary_embeddings: bool = True
    vocab_size: int = 49280
    _attn_implementation: Optional[AttentionImplementation] = DEFAULT_ATTENTION_IMPLEMENTATION

    def __post_init__(self):
        # [FUNCTION DOC]
        # Behavior: replaces a None global_attn_layers with an empty list; for grouped_query asserts num_kv_heads is set
        # and multi_query is off; when neither MQA nor GQA is selected, turns multi_query on; asserts the attention
        # implementation is registered.
        # Inputs: self. Outputs: None. Exceptions: AssertionError on invalid combinations. Side effects: mutates self.
        if self.global_attn_layers is None:
            self.global_attn_layers = []

        if self.grouped_query:
            assert self.num_kv_heads is not None, "num_kv_heads must be specified for grouped query"
            assert self.multi_query is False, "Cannot use both multi_query and grouped_query"

        if not self.multi_query and not self.grouped_query:
            self.multi_query = True

        # Validate that the attention implementation is valid
        if self._attn_implementation is not None:
            assert (
                self._attn_implementation in ALL_ATTENTION_FUNCTIONS
            ), f"Invalid attention implementation: {self._attn_implementation}. Available options are: {ALL_ATTENTION_FUNCTIONS.keys()}"

    @property
    def n_embed(self):
        # [FUNCTION DOC]
        # Behavior: property alias of hidden_size. Inputs: self. Outputs: int. Exceptions: None explicitly raised.
        # Side effects: None.
        return self.hidden_size

    @property
    def n_head(self):
        # [FUNCTION DOC]
        # Behavior: property alias of num_attention_heads. Inputs: self. Outputs: int. Exceptions: None explicitly
        # raised. Side effects: None.
        return self.num_attention_heads

    @property
    def n_layer(self):
        # [FUNCTION DOC]
        # Behavior: property alias of num_hidden_layers. Inputs: self. Outputs: int. Exceptions: None explicitly raised.
        # Side effects: None.
        return self.num_hidden_layers

    @property
    def n_positions(self):
        # [FUNCTION DOC]
        # Behavior: property alias of max_position_embeddings. Inputs: self. Outputs: int. Exceptions: None explicitly
        # raised. Side effects: None.
        return self.max_position_embeddings

    @property
    def n_inner(self):
        # [FUNCTION DOC]
        # Behavior: property alias of intermediate_size (may be None). Inputs: self. Outputs: int or None.
        # Exceptions: None explicitly raised. Side effects: None.
        return self.intermediate_size


NanotronConfigs = Union[LlamaConfig, Starcoder2Config, Qwen2Config, Any]
