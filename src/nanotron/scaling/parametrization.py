# [MODULE DOC]
# Purpose: Weight initialization schemes ("parametrizations") and per-parameter learning-rate rules
# for nanotron models: standard parametrization (SP) and spectral muP.
# Main contents: ParametrizationMethod enum, Parametrizator base, StandardParametrizator,
# SpectralMupParametrizator, LearningRateForParametrizator base, LearningRateForSP,
# LearningRateForSpectralMup.
# Fits in nanotron: models call a parametrizator per (parameter name, owning module) during
# init_model_randomly; the optimizer builder uses the learning-rate classes to set per-parameter lr.
# Import-time side effects: None.
import math
from abc import abstractmethod
from enum import Enum, auto
from typing import Dict

from nanotron.config import Config, ModelArgs
from nanotron.config.models_config import InitScalingMethod
from nanotron.nn.layer_norm import LlamaRMSNorm, TritonRMSNorm
from nanotron.nn.moe import GroupedMLP, Router
from nanotron.parallel.tensor_parallel.nn import (
    TensorParallelColumnLinear,
    TensorParallelEmbedding,
    TensorParallelRowLinear,
)
from torch import nn
from torch.nn import init


class ParametrizationMethod(Enum):
    # [CLASS DOC]
    # Purpose: Enum selecting the initialization scheme: STANDARD or SPECTRAL_MUP.
    STANDARD = auto()
    SPECTRAL_MUP = auto()


class Parametrizator:
    # [CLASS DOC]
    # Purpose: Base class dispatching initialization by module type.
    # Key attributes: config; subclasses must define MODULE_TO_PARAMETRIZE, a dictionary from module
    # class to a bound method taking (param_name, module).
    def __init__(self, config: ModelArgs):
        # [FUNCTION DOC]
        # Behavior: stores config. Inputs: config (model or full config, depending on subclass).
        # Outputs: None. Exceptions: None. Side effects: None.
        self.config = config

    def parametrize(self, param_name: str, module: nn.Module):
        # [FUNCTION DOC]
        # Behavior: checks module is an instance of one of the supported classes, then calls the handler
        # registered for the exact type of module.
        # Inputs: param_name (local name such as "weight" or "bias"), module (owning nn.Module).
        # Outputs: whatever the handler returns (None).
        # Exceptions: generic Exception for unsupported modules; KeyError for a subclass of a supported type.
        # Side effects: in-place initialization of the parameter.
        # KNOWN ISSUE: the isinstance check accepts subclasses but the lookup uses the exact type, so subclasses pass the check then fail with KeyError.
        if not isinstance(module, tuple(self.MODULE_TO_PARAMETRIZE.keys())):
            raise Exception(f"Module {type(module)} with parameter {param_name} is not supported for initialization")

        return self.MODULE_TO_PARAMETRIZE[type(module)](param_name, module)


class StandardParametrizator(Parametrizator):
    # [CLASS DOC]
    # Purpose: Standard (Megatron/GPT-style) init: normal(0, std) for linears, embeddings and routers,
    # row-linear (output projection) std divided by a depth scaling factor, ones for norms, zero biases.
    # Key attributes: MODULE_TO_PARAMETRIZE, std, num_layers, tp, scaling_method, hidden_size.
    def __init__(self, config: Config):
        # [FUNCTION DOC]
        # Behavior: registers handlers for TP column/row linear, Triton and Llama RMSNorm, TP embedding,
        # GroupedMLP, Router and plain nn.Linear (as a column linear); reads std, number of layers, TP size,
        # scaling method and hidden size from config.
        # Inputs: config (full Config with model.init_method, model.model_config and parallelism).
        # Outputs: None. Exceptions: AttributeError if config lacks these fields. Side effects: None.
        super().__init__(config)
        self.MODULE_TO_PARAMETRIZE = {
            TensorParallelColumnLinear: self._parametrize_column_linear,
            # TODO: double check if correct initialization for grouped MLP
            TensorParallelRowLinear: self._parametrize_row_linear,
            TritonRMSNorm: self._parametrize_layer_norm,
            LlamaRMSNorm: self._parametrize_layer_norm,
            TensorParallelEmbedding: self._parametrize_embedding,
            # NOTE: MoE's specific initialization
            GroupedMLP: self._parametrize_grouped_mlp,
            Router: self._parametrize_router,
            nn.Linear: self._parametrize_column_linear,
        }

        self.std = config.model.init_method.std
        self.num_layers = config.model.model_config.num_hidden_layers
        self.tp = config.parallelism.tp
        self.scaling_method = config.model.init_method.scaling_method
        self.hidden_size = config.model.model_config.hidden_size

    def _parametrize_column_linear(self, param_name: str, module: nn.Module):
        # [FUNCTION DOC]
        # HELPER: handler for column-parallel and plain linear layers.
        # Behavior: weight drawn from normal(0, std); bias zeroed.
        # Inputs: param_name ("weight" or "bias"), module. Outputs: None.
        # Exceptions: AssertionError for other names. Side effects: in-place init.
        assert param_name in ["weight", "bias"]

        if "weight" == param_name:
            # TODO @nouamane: should we use trunc_normal_
            init.normal_(module.weight, mean=0.0, std=self.std)
        elif "bias" == param_name:
            module.bias.zero_()

    def _parametrize_grouped_mlp(self, param_name: str, module: nn.Module):
        # [FUNCTION DOC]
        # HELPER: handler for MoE GroupedMLP.
        # Behavior: initializes all of the module's parameters (regardless of param_name): merged_gate_up_proj
        # with normal(0, std), merged_down_proj with normal(0, std divided by the scaling factor).
        # Inputs: param_name (ignored), module. Outputs: None.
        # Exceptions: ValueError for any other parameter name. Side effects: in-place init.
        for n, p in module.named_parameters():
            if n == "merged_gate_up_proj":
                # NOTE: the same as parametrization of column linear
                init.normal_(p, mean=0.0, std=self.std)
            elif n == "merged_down_proj":
                # NOTE: the same as parametrization of row linear
                scaling = self._compute_scaling_factor()
                adjusted_std = self.std / scaling
                # TODO @nouamane: should we use trunc_normal_
                init.normal_(p, mean=0.0, std=adjusted_std)
            else:
                raise ValueError(f"Unknown parameter {n}")

    def _parametrize_router(self, param_name: str, module: nn.Module):
        # [FUNCTION DOC]
        # HELPER: handler for MoE Router. Behavior: weight normal(0, std); bias zeroed; other names ignored.
        # Inputs: param_name, module. Outputs: None. Exceptions: None explicitly raised. Side effects: in-place init.
        if "weight" == param_name:
            init.normal_(module.weight, mean=0.0, std=self.std)
        elif "bias" == param_name:
            module.bias.zero_()

    def _compute_scaling_factor(self) -> float:
        # [FUNCTION DOC]
        # HELPER: depth scaling for output projections.
        # Behavior: NONE gives 1.0; NUM_LAYERS gives square root of twice the number of layers; LAYER_INDEX not implemented.
        # Inputs: None. Outputs: float.
        # Exceptions: NotImplementedError for LAYER_INDEX; ValueError for unknown methods. Side effects: None.
        """Compute initialization scaling based on selected method"""
        if self.scaling_method == InitScalingMethod.NONE:
            return 1.0
        elif self.scaling_method == InitScalingMethod.NUM_LAYERS:
            # Scale based on total network depth
            return math.sqrt(2 * self.num_layers)
        elif self.scaling_method == InitScalingMethod.LAYER_INDEX:
            # Scale based on layer position
            raise NotImplementedError("Layer position scaling not yet implemented")
        else:
            raise ValueError(f"Invalid scaling method: {self.scaling_method}")

    def _parametrize_row_linear(self, param_name: str, module: nn.Module):
        # [FUNCTION DOC]
        # HELPER: handler for row-parallel linears. Behavior: weight normal(0, std divided by scaling factor); bias zeroed.
        # Inputs: param_name, module. Outputs: None. Exceptions: AssertionError for other names. Side effects: in-place init.
        assert param_name in ["weight", "bias"]

        if "weight" == param_name:
            scaling = self._compute_scaling_factor()
            adjusted_std = self.std / scaling
            # TODO @nouamane: should we use trunc_normal_
            init.normal_(module.weight, mean=0.0, std=adjusted_std)
        elif "bias" == param_name:
            module.bias.zero_()

    def _parametrize_layer_norm(self, param_name: str, module: nn.Module):
        # [FUNCTION DOC]
        # HELPER: handler for RMSNorm. Behavior: weight filled with 1; bias zeroed.
        # Inputs: param_name, module. Outputs: None. Exceptions: AssertionError for other names. Side effects: in-place init.
        assert param_name in ["weight", "bias"]

        if "weight" == param_name:
            module.weight.fill_(1)
        elif "bias" == param_name:
            module.bias.zero_()

    def _parametrize_embedding(self, param_name: str, module: nn.Module):
        # [FUNCTION DOC]
        # HELPER: handler for TP embedding. Behavior: weight normal(0, std).
        # Inputs: param_name, module. Outputs: None. Exceptions: AssertionError unless name is "weight". Side effects: in-place init.
        assert param_name in ["weight"]

        if "weight" == param_name:
            init.normal_(module.weight, mean=0.0, std=self.std)


class SpectralMupParametrizator(Parametrizator):
    # [CLASS DOC]
    # Purpose: Spectral muP init (Yang et al., arXiv 2310.17813) where linear std depends on global fan-in
    # and fan-out; base std is 1.0.
    # Key attributes: MODULE_TO_PARAMETRIZE (TP column/row linear, TritonRMSNorm, TP embedding), std.
    """
    A Spectral Condition for Feature Learning by Greg Yang, et al.
    https://arxiv.org/abs/2310.17813
    """

    def __init__(self, config: ModelArgs):
        # [FUNCTION DOC]
        # Behavior: registers handlers and sets std to 1.0.
        # Inputs: config. Outputs: None. Exceptions: None. Side effects: None.
        # KNOWN ISSUE: LlamaRMSNorm, GroupedMLP, Router and nn.Linear are not registered, so models using them fail under spectral muP.
        super().__init__(config)
        self.MODULE_TO_PARAMETRIZE = {
            TensorParallelColumnLinear: self._parametrize_mup_weight,
            TensorParallelRowLinear: self._parametrize_mup_weight,
            TritonRMSNorm: self._parametrize_layer_norm,
            TensorParallelEmbedding: self._parametrize_embedding,
        }
        self.std = 1.0

    @staticmethod
    def _compute_spectral_std(std: float, fan_in: int, fan_out: int):
        # [FUNCTION DOC]
        # HELPER: spectral parametrization formula.
        # Behavior: returns std divided by square root of fan_in, times the minimum of 1 and square root of fan_out over fan_in.
        # Inputs: std (float), fan_in, fan_out (ints). Outputs: float. Exceptions: ZeroDivisionError if fan_in is 0. Side effects: None.
        """
        Parametrization 1 (Spectral parametrization)
        Page 8, A Spectral Condition for Feature Learning by Greg Yang, et al.

        σₗ = Θ(1/√nₗ₋₁ min{1, √(nₗ/nₗ₋₁)})
        """
        return (std / math.sqrt(fan_in)) * min(1, math.sqrt(fan_out / fan_in))

    def _parametrize_mup_weight(self, param_name: str, module: nn.Module):
        # [FUNCTION DOC]
        # HELPER: handler for TP linears.
        # Behavior: picks weight or bias, computes local fan-in/out, converts to global dims (fan_out times TP
        # size for column linear, fan_in times TP size for row linear), and draws normal(0, spectral std).
        # Inputs: param_name ("weight" or "bias"), module. Outputs: None.
        # Exceptions: AssertionError on other names; ValueError for unknown module types.
        # Side effects: in-place init.
        # KNOWN ISSUE: applying it to a 1-D bias makes fan computation fail (torch requires at least 2 dims), and biases get random rather than zero init.
        assert param_name in ["weight", "bias"]

        data = module.weight if param_name == "weight" else module.bias
        fan_in, fan_out = init._calculate_fan_in_and_fan_out(data)
        world_size = module.world_size

        if isinstance(module, TensorParallelColumnLinear):
            fan_out = fan_out * world_size
        elif isinstance(module, TensorParallelRowLinear):
            fan_in = fan_in * world_size
        else:
            raise ValueError(f"Unknown module {module}")

        std = SpectralMupParametrizator._compute_spectral_std(std=self.std, fan_in=fan_in, fan_out=fan_out)
        init.normal_(data, mean=0.0, std=std)

    def _parametrize_layer_norm(self, param_name: str, module: nn.Module):
        # [FUNCTION DOC]
        # HELPER: weight filled with 1, bias zeroed.
        # Inputs: param_name, module. Outputs: None. Exceptions: AssertionError for other names. Side effects: in-place init.
        assert param_name in ["weight", "bias"]

        # NOTE: you're free to change the initialization of layer norm
        # as it's not a part of µTransfer
        if "weight" == param_name:
            module.weight.fill_(1)
        elif "bias" == param_name:
            module.bias.zero_()

    def _parametrize_embedding(self, param_name: str, module: nn.Module):
        # [FUNCTION DOC]
        # HELPER: weight normal(0, std of 1.0).
        # Inputs: param_name, module. Outputs: None. Exceptions: AssertionError unless "weight". Side effects: in-place init.
        assert param_name in ["weight"]

        # NOTE: you're free to change the initialization of input embedding/lm head
        if "weight" == param_name:
            init.normal_(module.weight, mean=0.0, std=self.std)


class LearningRateForParametrizator:
    # [CLASS DOC]
    # Purpose: Abstract base computing a per-parameter learning rate.
    # Key attributes: lr (base learning rate), names_to_modules (dictionary from module name to module).
    def __init__(self, lr: float, names_to_modules: Dict[str, nn.Module]):
        # [FUNCTION DOC]
        # Behavior: stores lr and names_to_modules. Inputs: lr (float), names_to_modules (dict).
        # Outputs: None. Exceptions: None. Side effects: None.
        self.lr = lr
        self.names_to_modules = names_to_modules

    @abstractmethod
    def get_lr(self, param_name: str, module: nn.Module) -> float:
        # [FUNCTION DOC]
        # Behavior: abstract; always raises.
        # Inputs: param_name (str), module. Outputs: float in subclasses.
        # Exceptions: NotImplementedError. Side effects: None.
        # KNOWN ISSUE: decorated abstractmethod but the class does not use ABCMeta, so it is instantiable.
        raise NotImplementedError


class LearningRateForSP(LearningRateForParametrizator):
    # [CLASS DOC]
    # Purpose: Standard parametrization: every parameter gets the base lr.
    """All parameters get the same learning rate."""

    def get_lr(self, param_name: str, param: nn.Module) -> float:
        # [FUNCTION DOC]
        # Behavior: returns self.lr. Inputs: param_name, param (ignored). Outputs: float.
        # Exceptions: None. Side effects: None.
        return self.lr


class LearningRateForSpectralMup(LearningRateForParametrizator):
    # [CLASS DOC]
    # Purpose: Spectral muP learning rates: TP linears get lr scaled by global fan_out over fan_in; norms and embeddings get the base lr.
    # Key attributes: MODULE_TO_PARAMETRIZE mapping module type to lr function.
    """
    A Spectral Condition for Feature Learning by Greg Yang, et al.

    NOTE: each parameter gets a custom learning rate based on its fan-in and fan-out.
    """

    def __init__(self, lr: float, names_to_modules: Dict[str, nn.Module]):
        # [FUNCTION DOC]
        # Behavior: calls base init and registers lr handlers.
        # Inputs: lr, names_to_modules. Outputs: None. Exceptions: None. Side effects: None.
        super().__init__(lr, names_to_modules)
        self.MODULE_TO_PARAMETRIZE = {
            TensorParallelColumnLinear: self._get_mup_lr,
            TensorParallelRowLinear: self._get_mup_lr,
            TritonRMSNorm: self._get_global_lr,
            TensorParallelEmbedding: self._get_global_lr,
        }

    def _get_mup_lr(self, param: nn.Parameter, module: nn.Module):
        # [FUNCTION DOC]
        # HELPER: lr for TP linears.
        # Behavior: computes local fan-in/out of param, globalizes by TP size as in initialization, returns lr times fan_out over fan_in.
        # Inputs: param (tensor), module. Outputs: float.
        # Exceptions: ValueError for unknown module types. Side effects: None.
        """
        Parametrization 1 (Spectral parametrization)
        Page 8, A Spectral Condition for Feature Learning by Greg Yang, et al.

        ηₗ = Θ(nₗ/nₗ₋₁)
        """
        fan_in, fan_out = init._calculate_fan_in_and_fan_out(param)
        world_size = module.world_size

        if isinstance(module, TensorParallelColumnLinear):
            fan_out = fan_out * world_size
        elif isinstance(module, TensorParallelRowLinear):
            fan_in = fan_in * world_size
        else:
            raise ValueError(f"Unknown module {module}")

        return self.lr * (fan_out / fan_in)

    def _get_global_lr(self, param: nn.Parameter, module: nn.Module) -> float:
        # [FUNCTION DOC]
        # HELPER: returns the base lr. Inputs: param, module. Outputs: float. Exceptions: None. Side effects: None.
        return self.lr

    def get_lr(self, param_name: str, param: nn.Parameter) -> float:
        # [FUNCTION DOC]
        # Behavior: strips the last dotted component from param_name to get the module name, looks up the
        # module and dispatches on its exact type.
        # Inputs: param_name (full dotted name), param (the parameter tensor).
        # Outputs: float learning rate.
        # Exceptions: KeyError for unknown module names or unsupported types. Side effects: None.
        """Return the learning rate for the given parameter."""
        # NOTE: param_name should be like 'model.token_position_embeddings.pp_block.token_embedding.weight'
        # since names_to_modules map module_name to module
        # so we remove the .weight and .bias from param_name to get the module_name
        module_name = param_name.rsplit(".", 1)[0]
        module = self.names_to_modules[module_name]
        return self.MODULE_TO_PARAMETRIZE[type(module)](param, module)
