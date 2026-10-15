# [MODULE DOC]
# Purpose: optimizer wrapper that tracks parameter names, so optimizer state dicts carry names and can be
# resharded or validated on load.
# Main contents: NamedOptimizer.
# Import-time side effects: creates a module logger.
from typing import Any, Callable, Dict, Iterable, Optional, Tuple, Union

import torch

from nanotron import logging
from nanotron.optim.inherit_from_other_optimizer import InheritFromOtherOptimizer

logger = logging.get_logger(__name__)


class NamedOptimizer(InheritFromOtherOptimizer):
    # [CLASS DOC]
    # Purpose: builds a torch optimizer from named parameters (or named param groups) and records names.
    # Key state: id_to_name (from parent), optimizer (the built torch optimizer).
    # Inheritance: InheritFromOtherOptimizer; overrides state_dict and load_state_dict.
    """Mimics somewhat the torch optimizer API"""

    def __init__(
        self,
        named_params_or_groups: Iterable[Union[Tuple[str, torch.Tensor], Dict[str, Any]]],
        optimizer_builder: Callable[[Iterable[Dict[str, Any]]], torch.optim.Optimizer],
    ):
        # [FUNCTION DOC]
        # Behavior: materializes the input; if it is empty or not a list of dicts, wraps it as one group with key
        # "named_params". For each group asserts "named_params" exists, records id to name (first name wins for shared
        # params), and builds a torch group with the other keys plus "params". Asserts names are unique, asserts every
        # param has numel greater than 0, then calls optimizer_builder on the groups and wraps the result.
        # Inputs: named_params_or_groups (iterable of name/tensor pairs or of group dicts with "named_params"),
        # optimizer_builder (callable from list of torch param group dicts to a torch optimizer).
        # Outputs: None.
        # Exceptions: AssertionError on missing "named_params", duplicate names, or empty params.
        # Side effects: constructs the inner optimizer.
        named_param_groups = list(named_params_or_groups)
        if len(named_param_groups) == 0 or not isinstance(named_param_groups[0], dict):
            named_param_groups = [{"named_params": named_param_groups}]

        id_to_name = {}
        params = []
        for named_param_group in named_param_groups:
            assert "named_params" in named_param_group
            # Don't need to check that param_groups are overlapping since the optimizer will do it for me.
            #  https://github.com/pytorch/pytorch/blob/88b3810c94b45f5982df616e2bc4c471d173f491/torch/optim/optimizer.py#L473
            id_to_name.update(
                {id(param): name for name, param in named_param_group["named_params"] if id(param) not in id_to_name}
            )
            params.append(
                {
                    **{k: v for k, v in named_param_group.items() if k != "named_params"},
                    "params": [param for _, param in named_param_group["named_params"]],
                }
            )

        name_to_id = {v: k for k, v in id_to_name.items()}
        assert len(id_to_name) == len(name_to_id)

        # Sanity check
        for param_group in params:
            _params = param_group["params"]
            for param in _params:
                # https://github.com/pytorch/pytorch/issues/100701
                assert param.numel() > 0
        super().__init__(optimizer=optimizer_builder(params), id_to_name=id_to_name)

    def state_dict(self) -> dict:
        # [FUNCTION DOC]
        # Behavior: gets the inner state dict, asserts it has no "names" key, and adds "names" mapping each state index to
        # the parameter name (matching state objects by identity with the inner optimizer's live state).
        # Outputs: dict with "state", "param_groups", "names".
        # Exceptions: AssertionError if "names" is already present; KeyError if a state has no matching parameter.
        # Side effects: None.
        optim_state_dict = super().state_dict()

        assert "names" not in optim_state_dict

        state_id_to_name = {id(state): self.id_to_name[id(param)] for param, state in self.optimizer.state.items()}
        optim_state_dict["names"] = {
            index: state_id_to_name[id(state)] for index, state in optim_state_dict["state"].items()
        }
        return optim_state_dict

    def load_state_dict(self, state_dict: dict, map_location: Optional[Union[str, torch.device]] = None) -> None:
        # [FUNCTION DOC]
        # Behavior: validates that names in state_dict equal this optimizer's names, that state and names have the same
        # count and are non-empty, and that every state has all non-"step" keys of state index 0; then delegates to parent.
        # Inputs: state_dict (dict with "state" and "names"), map_location (default None).
        # Outputs: None.
        # Exceptions: AssertionError on any mismatch; KeyError if state index 0 is absent.
        # Side effects: replaces inner optimizer state.
        assert set(self.id_to_name.values()) == set(
            state_dict["names"].values()
        ), f"Elements don't match:\n - Elements in `self.id_to_name` that aren't in the other one: {set(self.id_to_name.values()) - set(state_dict['names'].values())}\n - Elements in `state_dict[\"names\"]` that aren't in the other one: {set(state_dict['names'].values()) - set(self.id_to_name.values())}"
        assert len(state_dict["state"]) == len(
            state_dict["names"]
        ), f"Number of params in loaded state dict ({len(state_dict['state'])}) doesn't match number of names ({len(state_dict['names'])})"
        assert len(state_dict["state"]) > 0, "Loading empty state dict"
        OPTIMIZER_STATE_KEYS = sorted(state_dict["state"][0].keys() - {"step"})
        for key in OPTIMIZER_STATE_KEYS:
            for k, state in state_dict["state"].items():
                assert (
                    key in state
                ), f"Key {key} not found in state dict: {state} which corresponds to param_name: {state_dict['names'][k]}"

        return super().load_state_dict(state_dict, map_location=map_location)
