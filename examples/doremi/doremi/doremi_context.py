# [MODULE DOC]
# Purpose: Runtime state shared by DoReMi components: the domain list, current domain weights and their history.
# Main contents: WeightHistory typed dictionary and DoReMiContext dataclass.
# Side effects at import: None.
from dataclasses import dataclass, field
from typing import List, TypedDict

import torch


class WeightHistory(TypedDict):
    # [CLASS DOC]
    # Purpose: Typed dictionary for one history record: step (int training step) and weight (CPU tensor of domain weights).
    step: int
    weight: torch.Tensor


@dataclass
class DoReMiContext:
    # NOTE: this is the current domain weights
    # [CLASS DOC]
    # Purpose: Holds DoReMi configuration and mutable domain-weight state used by the sampler, loss and trainer.
    # Attributes: domain_keys (list of domain names), is_proxy (True when training the proxy model whose weights are updated), step_size (default 1), smoothing_param (default 1e-3), domain_weight_history (list of WeightHistory records), domain_weights (tensor of shape [num_domains], set in __post_init__).
    # Invariants: domain_weights sums to 1 and has one entry per domain.
    domain_keys: List[str]
    is_proxy: bool
    step_size: float = 1
    smoothing_param: float = 1e-3

    domain_weight_history: WeightHistory = field(default_factory=list)

    @property
    def num_domains(self) -> int:
        # [FUNCTION DOC]
        # Behavior: Property returning the number of domains (length of domain_keys).
        # Inputs: self. Outputs: int. Exceptions: None explicitly raised. Side effects: None.
        return len(self.domain_keys)

    def get_domain_name(self, domain_idx: int) -> str:
        # [FUNCTION DOC]
        # Behavior: Returns the domain name at the given index.
        # Inputs: domain_idx: int. Outputs: string.
        # Exceptions: IndexError if out of range. Side effects: None.
        return self.domain_keys[domain_idx]

    def __post_init__(self):
        # NOTE: by default, we do uniform sampling for DoReMi
        # [FUNCTION DOC]
        # Behavior: Initializes domain_weights to the uniform distribution over domains, asserts it sums to 1 and has the right length, and records it in the history at step 0.
        # Inputs: self. Outputs: None.
        # Exceptions: AssertionError (cannot fail for uniform weights); ZeroDivisionError-like NaN if there are no domains.
        # Side effects: sets self.domain_weights and appends to domain_weight_history.
        self.domain_weights = torch.ones(self.num_domains) / self.num_domains
        assert torch.allclose(
            self.domain_weights.sum(dim=-1), torch.tensor(1.0), rtol=0.001
        ), "Domain weights must sum up to 1."
        assert (
            self.domain_weights.shape[0] == self.num_domains
        ), "The length of domain_weights must be equal to the number of domains"
        self.add_weight_with_history(self.domain_weights, 0)

    def add_weight_with_history(self, domain_weights: torch.Tensor, step: int):
        # [FUNCTION DOC]
        # Behavior: Asserts step is non-negative and appends a WeightHistory record with a CPU copy of the weights.
        # Inputs: domain_weights: tensor [num_domains]; step: int.
        # Outputs: None.
        # Exceptions: AssertionError if step is negative.
        # Side effects: appends to self.domain_weight_history; device-to-host copy.
        # KNOWN ISSUE: domain_weight_history is annotated as a single WeightHistory but is actually a list of them.
        assert step >= 0, "Step must be a positive integer"
        self.domain_weight_history.append(WeightHistory(step=step, weight=domain_weights.cpu()))
