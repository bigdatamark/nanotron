# [MODULE DOC]
# Purpose: Small tensor/dataset utilities for the DoReMi example.
# Main contents: masked_mean (TorchScript) and compute_domain_weights_based_on_token_count.
# Side effects at import: compiles masked_mean with torch.jit.script.
from typing import List

import torch
from torch.utils.data import Dataset


@torch.jit.script
def masked_mean(loss: torch.Tensor, label_mask: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    # [FUNCTION DOC]
    # HELPER: masked averaging of per-token losses.
    # Behavior: Multiplies loss by label_mask, sums in the given dtype, and divides by the number of masked-in tokens.
    # Inputs: loss: tensor of per-token losses; label_mask: tensor of same shape (0/1 or bool); dtype: accumulation dtype.
    # Outputs: scalar tensor mean.
    # Exceptions: None explicitly raised (division by zero yields NaN/inf if mask is empty).
    # Side effects: None.
    return (loss * label_mask).sum(dtype=dtype) / label_mask.sum()


def compute_domain_weights_based_on_token_count(datasets: List[Dataset]) -> torch.Tensor:
    # [FUNCTION DOC]
    # Behavior: Computes each dataset's share of the total number of samples (length of each dataset over the sum).
    # Inputs: datasets: list of torch Datasets, one per domain.
    # Outputs: 1-D float tensor of weights summing to 1.
    # Exceptions: ZeroDivisionError if all datasets are empty.
    # Side effects: None.
    # KNOWN ISSUE: the name says token count but it uses sample counts (equal only if all samples have the same length).
    num_samples_per_domain = [len(d) for d in datasets]
    total_samples = sum(num_samples_per_domain)
    weights = torch.tensor([num_sample / total_samples for num_sample in num_samples_per_domain])
    return weights
