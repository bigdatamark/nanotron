# [MODULE DOC]
# Purpose: generic delegating optimizer wrapper that the other nanotron optimizers build on.
# Main contents: InheritFromOtherOptimizer.
# Import-time side effects: None.
from functools import cache
from typing import Callable, Dict, Optional, Set, Union

import torch

from nanotron.optim.base import BaseOptimizer, Optimizer, custom_load_state_dict


class InheritFromOtherOptimizer(BaseOptimizer):
    # [CLASS DOC]
    # Purpose: wraps an inner optimizer (torch or another BaseOptimizer) and forwards every call to it, allowing
    # wrappers to be stacked (for example ZeRO around a gradient-accumulator optimizer around a NamedOptimizer).
    # Key state: optimizer (the wrapped optimizer), id_to_name (id of parameter to name).
    # Inheritance: concrete BaseOptimizer; subclasses override selected methods.
    def __init__(self, optimizer: Optimizer, id_to_name: Dict[int, str]):
        # [FUNCTION DOC]
        # Behavior: stores id_to_name; if the inner optimizer is a torch optimizer, monkey-patches its load_state_dict
        # so a call with a map_location uses custom_load_state_dict, and without one uses the original; stores the optimizer.
        # Inputs: optimizer (torch optimizer or BaseOptimizer), id_to_name (dict).
        # Outputs: None. Exceptions: None explicitly raised.
        # Side effects: replaces load_state_dict on the passed torch optimizer instance.
        self.id_to_name = id_to_name

        # if self.optimizer is from torch we replace load_state_dict with the one from torch
        if isinstance(optimizer, torch.optim.Optimizer):
            # Replace the load_state_dict method with our custom implementation that enables CPU offload
            original_load_state_dict = optimizer.load_state_dict
            optimizer.load_state_dict = (
                lambda state_dict, map_location=None: custom_load_state_dict(
                    optimizer, state_dict, map_location=map_location
                )
                if map_location is not None
                else original_load_state_dict(state_dict)
            )

        self.optimizer: Optimizer = optimizer

    def __getstate__(self):
        # [FUNCTION DOC]
        # Delegates to the inner optimizer's __getstate__. Outputs: its state. Exceptions/Side effects: None.
        return self.optimizer.__getstate__()

    def __setstate__(self, state):
        # [FUNCTION DOC]
        # Delegates to the inner optimizer's __setstate__. Inputs: state. Outputs: its return. Side effects: mutates inner optimizer.
        return self.optimizer.__setstate__(state)

    def __repr__(self):
        # [FUNCTION DOC]
        # Returns "ClassName(inner repr)". Outputs: str. Exceptions/Side effects: None.
        return f"{self.__class__.__name__}({self.optimizer.__repr__()})"

    def zero_grad(self):
        # [FUNCTION DOC]
        # Delegates to inner zero_grad. Outputs: its return. Side effects: clears gradients.
        return self.optimizer.zero_grad()

    @cache
    def state_dict_additional_keys(self) -> Set[str]:
        # [FUNCTION DOC]
        # Behavior: returns inner optimizer's additional keys if it is a BaseOptimizer, else an empty set; memoized.
        # Outputs: set of strings. Exceptions: None. Side effects: result cached per instance.
        # KNOWN ISSUE: functools.cache on a method keeps a strong reference to self in a global cache, leaking the optimizer for the process lifetime.
        if isinstance(self.optimizer, BaseOptimizer):
            return self.optimizer.state_dict_additional_keys()
        else:
            return set()

    def state_dict(self) -> dict:
        # [FUNCTION DOC]
        # Delegates to inner state_dict. Outputs: dict. Exceptions/Side effects: None.
        return self.optimizer.state_dict()

    def load_state_dict(self, state_dict: dict, map_location: Optional[Union[str, torch.device]] = None) -> None:
        # [FUNCTION DOC]
        # Delegates to inner load_state_dict with map_location. Inputs: state_dict (dict), map_location (default None).
        # Outputs: None. Side effects: mutates inner state.
        return self.optimizer.load_state_dict(state_dict, map_location=map_location)

    @torch.profiler.record_function("InheritFromOtherOptimizer.step")
    def step(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]:
        # [FUNCTION DOC]
        # Delegates to inner step with closure (profiled region). Inputs: closure (optional). Outputs: optional loss.
        # Side effects: parameter updates.
        return self.optimizer.step(closure=closure)

    def get_base_optimizer(self):
        # [FUNCTION DOC]
        # Behavior: walks down the wrapper chain and returns the innermost torch optimizer.
        # Outputs: torch.optim.Optimizer. Exceptions: AttributeError if the chain ends in a non-torch object lacking this method.
        # Side effects: None.
        if isinstance(self.optimizer, torch.optim.Optimizer):
            return self.optimizer
        else:
            return self.optimizer.get_base_optimizer()

    @property
    def param_groups(self):
        # [FUNCTION DOC]
        # Property: inner optimizer's param_groups. Outputs: list of group dicts. Exceptions/Side effects: None.
        return self.optimizer.param_groups

    def inherit_from(self, cls):
        # [FUNCTION DOC]
        # Behavior: True if self is an instance of cls, else recurses into the inner optimizer if it is also a wrapper,
        # else False.
        # Inputs: cls (class). Outputs: bool. Exceptions/Side effects: None.
        if isinstance(self, cls):
            return True
        if isinstance(self.optimizer, InheritFromOtherOptimizer):
            return self.optimizer.inherit_from(cls)
        return False
