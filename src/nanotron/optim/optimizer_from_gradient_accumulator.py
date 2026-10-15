# [MODULE DOC]
# Purpose: optimizer wrapper that lets a gradient accumulator (for example fp32 master weights) own the
# parameters the inner optimizer updates.
# Main contents: OptimizerFromGradientAccumulator.
# Import-time side effects: None.
from functools import cache
from typing import Any, Callable, Dict, Iterable, Optional, Set, Tuple, Union

import torch

from nanotron.optim.base import BaseOptimizer
from nanotron.optim.gradient_accumulator import GradientAccumulator
from nanotron.optim.inherit_from_other_optimizer import InheritFromOtherOptimizer
from nanotron.parallel.parameters import NanotronParameter


class OptimizerFromGradientAccumulator(InheritFromOtherOptimizer):
    # [CLASS DOC]
    # Purpose: builds a gradient accumulator from the named params and builds the inner optimizer over the
    # accumulator's parameters (e.g. fp32 copies); keeps both in sync on step/zero_grad/state_dict.
    # Key state: gradient_accumulator, optimizer, id_to_name (taken from the inner optimizer).
    # Inheritance: InheritFromOtherOptimizer.
    def __init__(
        self,
        gradient_accumulator_builder: Callable[[Iterable[Tuple[str, NanotronParameter]]], GradientAccumulator],
        named_params_or_groups: Iterable[Union[Tuple[str, torch.Tensor], Dict[str, Any]]],
        optimizer_builder: Callable[[Iterable[Dict[str, Any]]], BaseOptimizer],
    ):
        # [FUNCTION DOC]
        # Behavior: normalizes input into named param groups, builds a name-to-param map (rejecting duplicate names),
        # calls gradient_accumulator_builder on it, replaces each param with get_parameter_for_optimizer of its name, builds
        # the inner optimizer with optimizer_builder, and wraps it.
        # Inputs: gradient_accumulator_builder (callable from name/param pairs to GradientAccumulator),
        # named_params_or_groups (pairs or group dicts with "named_params"), optimizer_builder (callable returning a
        # BaseOptimizer exposing id_to_name).
        # Outputs: None. Exceptions: ValueError on duplicate names. Side effects: allocates accumulator buffers.
        named_param_groups = list(named_params_or_groups)
        if len(named_param_groups) == 0 or not isinstance(named_param_groups[0], dict):
            named_param_groups = [{"named_params": named_param_groups}]

        name_to_param = {}
        for named_param_group in named_param_groups:
            for name, param in named_param_group["named_params"]:
                if name in name_to_param:
                    raise ValueError(f"Duplicate key. {name} is already in `name_to_param`")
                else:
                    name_to_param[name] = param

        # Build gradient accumulator
        gradient_accumulator = gradient_accumulator_builder(name_to_param.items())
        self.gradient_accumulator = gradient_accumulator

        # Obtained new params depending on the gradient accumulator
        converted_named_param_group = [
            {
                **{k: v for k, v in named_param_group.items() if k != "named_params"},
                "named_params": [
                    (name, gradient_accumulator.get_parameter_for_optimizer(name))
                    for name, _ in named_param_group["named_params"]
                ],
            }
            for named_param_group in named_param_groups
        ]
        optimizer = optimizer_builder(converted_named_param_group)

        super().__init__(optimizer=optimizer, id_to_name=optimizer.id_to_name)

    def step(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]:
        # [FUNCTION DOC]
        # Behavior: runs inner step, then gradient_accumulator.step to copy updated master weights into model params.
        # Inputs: closure (optional). Outputs: loss from inner step. Exceptions: None explicitly raised.
        # Side effects: updates parameters.
        loss = super().step(closure)
        self.gradient_accumulator.step()
        return loss

    def zero_grad(self):
        # [FUNCTION DOC]
        # Behavior: zeroes inner optimizer grads then the accumulator's. Outputs: accumulator's return (None).
        # Side effects: clears gradients and buffers.
        super().zero_grad()
        return self.gradient_accumulator.zero_grad()

    @cache
    def state_dict_additional_keys(self) -> Set[str]:
        # [FUNCTION DOC]
        # Behavior: parent's keys plus "gradient_accumulator"; memoized. Outputs: set of strings.
        # Exceptions: None. Side effects: per-instance cache.
        # KNOWN ISSUE: functools.cache on a method keeps self alive in a global cache.
        return super().state_dict_additional_keys() | {"gradient_accumulator"}

    def state_dict(self) -> dict:
        # [FUNCTION DOC]
        # Behavior: parent state dict plus "gradient_accumulator" holding the accumulator state.
        # Outputs: dict. Exceptions: AssertionError if the key already exists. Side effects: None.
        state_dict = super().state_dict()

        assert "gradient_accumulator" not in state_dict
        state_dict["gradient_accumulator"] = self.gradient_accumulator.state_dict()
        return state_dict

    def load_state_dict(self, state_dict: dict, map_location: Optional[Union[str, torch.device]] = None) -> None:
        # [FUNCTION DOC]
        # Behavior: pops "gradient_accumulator", loads the rest into the inner optimizer, then loads the accumulator.
        # Inputs: state_dict (dict), map_location (default None). Outputs: None.
        # Exceptions: KeyError if "gradient_accumulator" is missing.
        # Side effects: mutates the passed state_dict (removes key), inner state and accumulator weights.
        gradient_accumulator_state_dict = state_dict.pop("gradient_accumulator")
        super().load_state_dict(state_dict, map_location=map_location)
        self.gradient_accumulator.load_state_dict(gradient_accumulator_state_dict)
