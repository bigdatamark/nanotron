# [MODULE DOC]
# Purpose: abstract optimizer interface for nanotron plus a device-aware optimizer state loader.
# Main contents: type aliases Args, Kwargs, StateDict; BaseOptimizer abstract class; Optimizer TypeVar;
# _process_value_according_to_param_policy and custom_load_state_dict, adapted from torch.optim.Optimizer so
# optimizer state can be loaded onto an explicit map_location.
# Import-time side effects: None beyond imports.
from abc import ABC, abstractmethod
from collections import defaultdict
from copy import deepcopy
from itertools import chain
from typing import (
    Any,
    Callable,
    DefaultDict,
    Dict,
    Hashable,
    Iterable,
    List,
    Optional,
    Set,
    Tuple,
    TypeVar,
    Union,
)

import torch
from typing_extensions import TypeAlias

Args: TypeAlias = Tuple[Any, ...]
Kwargs: TypeAlias = Dict[str, Any]
StateDict: TypeAlias = Dict[str, Any]


class BaseOptimizer(ABC):
    # [CLASS DOC]
    # Purpose: abstract interface all nanotron optimizer wrappers implement (mirrors torch.optim.Optimizer).
    # Key attributes: id_to_name (maps id of parameter to its name), param_groups (list of group dicts).
    # Subclasses must implement pickling hooks, repr, zero_grad, state_dict_additional_keys, state_dict,
    # load_state_dict, step.
    id_to_name: Dict[int, str]
    param_groups: List[Dict[str, Any]]

    @abstractmethod
    def __getstate__(self):
        # [FUNCTION DOC]
        # Abstract: return picklable state. Inputs: none. Outputs: subclass-defined. Exceptions/Side effects: None.
        ...

    @abstractmethod
    def __setstate__(self, state):
        # [FUNCTION DOC]
        # Abstract: restore from state. Inputs: state. Outputs: None. Exceptions/Side effects: subclass-defined.
        ...

    @abstractmethod
    def __repr__(self):
        # [FUNCTION DOC]
        # Abstract: string representation. Outputs: str.
        ...

    @abstractmethod
    def zero_grad(self):
        # [FUNCTION DOC]
        # Abstract: reset gradients of managed parameters. Outputs: None. Side effects: clears gradients.
        ...

    @abstractmethod
    def state_dict_additional_keys(self) -> Set[str]:
        # [FUNCTION DOC]
        # Abstract: names of extra per-parameter states stored in state_dict (each keyed by parameter name with tensor values).
        # Outputs: set of strings.
        """Additional states we store in `state_dict`. It has to be a dictionary using parameter name as key, and a tensor as value."""
        ...

    @abstractmethod
    def state_dict(self) -> dict:
        # [FUNCTION DOC]
        # Abstract: serialize optimizer state. Outputs: dict.
        ...

    @abstractmethod
    def load_state_dict(self, state_dict: dict, map_location: Optional[Union[str, torch.device]] = None) -> None:
        # [FUNCTION DOC]
        # Abstract: load state. Inputs: state_dict (dict), map_location (device or string, or None for each
        # parameter's device). Outputs: None. Side effects: mutates optimizer state.
        ...

    @abstractmethod
    def step(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]:
        # [FUNCTION DOC]
        # Abstract: perform an optimization step. Inputs: closure (optional callable returning loss). Outputs: optional
        # loss float. Side effects: updates parameters.
        ...

    def inherit_from(self, cls) -> bool:
        # [FUNCTION DOC]
        # Intended to report whether this optimizer wraps / inherits from optimizer class cls.
        # Inputs: cls (class). Outputs: bool expected; base body returns None.
        # Exceptions: None. Side effects: None.
        # KNOWN ISSUE: not abstract and has an empty body, so subclasses that do not override it return None instead of a bool.
        ...


Optimizer = TypeVar("Optimizer", BaseOptimizer, torch.optim.Optimizer)


# Modified from torch.optim.Optimizer._process_value_according_to_param_policy
@staticmethod
def _process_value_according_to_param_policy(
    param: torch.Tensor,
    value: torch.Tensor,
    param_id: int,
    param_groups: List[Dict[Any, Any]],
    map_location: Optional[Union[str, torch.device]],
    key: Hashable = None,
) -> torch.Tensor:
    # [FUNCTION DOC]
    # HELPER: casts one optimizer state tensor for custom_load_state_dict.
    # Behavior: target device is map_location or the parameter's device. Looks up the param group containing
    # param_id to read its fused and capturable flags. For key "step": converts to float32 on target device if fused or
    # capturable, else returns unchanged. Otherwise casts to param dtype and target device if param is floating
    # point, else only moves it to the target device.
    # Inputs: param (tensor), value (state tensor), param_id (int index), param_groups (list of group dicts),
    # map_location (device/string or None), key (state key, default None).
    # Outputs: tensor.
    # Exceptions: AssertionError if param_groups is None.
    # Side effects: None (may allocate device memory).
    # KNOWN ISSUE: decorated with staticmethod at module level; this works only on Python 3.10+ where staticmethod objects are callable.
    # KNOWN ISSUE: param_groups passed here are the saved groups whose params are integer ids, which is intended, but the "step" non-capturable path never moves the value to map_location.
    # If map_location is specified, use it instead of param.device
    target_device = map_location if map_location is not None else param.device

    fused = False
    capturable = False
    assert param_groups is not None
    for pg in param_groups:
        if param_id in pg["params"]:
            fused = pg["fused"] if "fused" in pg else False
            capturable = pg["capturable"] if "capturable" in pg else False
            break

    if key == "step":
        if capturable or fused:
            return value.to(dtype=torch.float32, device=target_device)
        else:
            return value
    else:
        if param.is_floating_point():
            return value.to(dtype=param.dtype, device=target_device)
        else:
            return value.to(device=target_device)


# Modified from torch.optim.Optimizer.load_state_dict
@torch._disable_dynamo
def custom_load_state_dict(self, state_dict: StateDict, map_location: Union[str, torch.device]) -> None:
    # [FUNCTION DOC]
    # Behavior: a replacement for torch.optim.Optimizer.load_state_dict meant to be bound as a method. Shallow-copies
    # state_dict, runs registered pre-hooks (which may replace it), validates group count and per-group parameter
    # counts against self.param_groups, maps saved parameter ids to live parameters in order, casts each per-parameter
    # state via _cast (tensors through _process_value_according_to_param_policy, recursing into dicts and iterables),
    # keeps unmatched state as-is, rebuilds param groups with saved hyperparameters but live params, calls
    # __setstate__, then runs post-hooks.
    # Inputs: self (a torch optimizer), state_dict (dict with "state" and "param_groups"), map_location (device or
    # string; None means each parameter's device).
    # Outputs: None.
    # Exceptions: ValueError on mismatched group count or group sizes; AssertionError from helper.
    # Side effects: replaces optimizer state and param groups; runs hooks; moves tensors across devices. Dynamo is
    # disabled for this function.
    r"""Loads the optimizer state.

    Args:
        state_dict (dict): optimizer state. Should be an object returned
            from a call to :meth:`state_dict`.
        map_location (str or torch.device, optional): Device where to load the optimizer states.
            If None, states will be loaded to the same device as their corresponding parameters.
            Default: None
    """

    # shallow copy, to be consistent with module API
    state_dict = state_dict.copy()

    for pre_hook in self._optimizer_load_state_dict_pre_hooks.values():
        hook_result = pre_hook(self, state_dict)
        if hook_result is not None:
            state_dict = hook_result

    # Validate the state_dict
    groups = self.param_groups

    # Deepcopy as we write into saved_groups later to update state
    saved_groups = deepcopy(state_dict["param_groups"])

    if len(groups) != len(saved_groups):
        raise ValueError("loaded state dict has a different number of " "parameter groups")
    param_lens = (len(g["params"]) for g in groups)
    saved_lens = (len(g["params"]) for g in saved_groups)
    if any(p_len != s_len for p_len, s_len in zip(param_lens, saved_lens)):
        raise ValueError(
            "loaded state dict contains a parameter group " "that doesn't match the size of optimizer's group"
        )

    # Update the state
    id_map = dict(
        zip(chain.from_iterable(g["params"] for g in saved_groups), chain.from_iterable(g["params"] for g in groups))
    )

    def _cast(param, value, param_id=None, param_groups=None, key=None):
        # [FUNCTION DOC]
        # HELPER: recursive deep-cast of a state value.
        # Behavior: tensors are converted with _process_value_according_to_param_policy; dicts are rebuilt recursing with
        # each entry's key; other iterables are rebuilt with the same type; other values returned unchanged.
        # Inputs: param, value, param_id, param_groups, key (all optional except param, value).
        # Outputs: cast copy of value. Exceptions: None explicitly raised. Side effects: None.
        # KNOWN ISSUE: strings are iterables and would be rebuilt character by character into a generator-constructed str, producing a wrong value for string states.
        r"""Make a deep copy of value, casting all tensors to device of param."""
        if isinstance(value, torch.Tensor):
            return _process_value_according_to_param_policy(param, value, param_id, param_groups, map_location, key)
        elif isinstance(value, dict):
            return {k: _cast(param, v, param_id=param_id, param_groups=param_groups, key=k) for k, v in value.items()}
        elif isinstance(value, Iterable):
            return type(value)(_cast(param, v, param_id=param_id, param_groups=param_groups) for v in value)
        else:
            return value

    # Copy state assigned to params (and cast tensors to appropriate types).
    # State that is not assigned to params is copied as is (needed for
    # backward compatibility).
    state: DefaultDict[torch.Tensor, Dict[Any, Any]] = defaultdict(dict)
    for k, v in state_dict["state"].items():
        if k in id_map:
            param = id_map[k]
            state[param] = _cast(param, v, param_id=k, param_groups=state_dict["param_groups"])
        else:
            state[k] = v

    # Update parameter groups, setting their 'params' value
    def update_group(group: Dict[str, Any], new_group: Dict[str, Any]) -> Dict[str, Any]:
        # [FUNCTION DOC]
        # HELPER: writes the live group's params into the saved group dict and returns it.
        # Inputs: group (live group dict), new_group (saved group dict). Outputs: new_group. Side effects: mutates new_group.
        # Exceptions: None.
        new_group["params"] = group["params"]
        return new_group

    param_groups = [update_group(g, ng) for g, ng in zip(groups, saved_groups)]
    self.__setstate__({"state": state, "param_groups": param_groups})

    for post_hook in self._optimizer_load_state_dict_post_hooks.values():
        post_hook(self)
