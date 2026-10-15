# [MODULE DOC]
# Purpose: small numeric helpers for nanotron's data pipeline (dataset blending).
# Contents: normalize (turn raw weights into proportions) and count_dataset_indexes
# (count how many samples were drawn from each dataset).
# Import-time side effects: None. Exports: normalize, count_dataset_indexes.
from typing import List

import numpy as np


def normalize(weights: List[float]) -> List[np.array]:
    # [FUNCTION DOC]
    # HELPER: converts dataset mixing weights into a probability distribution.
    # Behavior: converts the list to a float64 numpy array, divides each element by the sum.
    # Inputs: weights - list of floats (non-negative, non-zero sum expected).
    # Outputs: 1-D float64 numpy array of the same length whose entries sum to 1.
    # Exceptions: None explicitly raised (a zero sum yields NaN/inf with a numpy warning).
    # Side effects: None.
    # KNOWN ISSUE: return annotation and docstring say a list of numpy arrays, but a single 1-D numpy array is returned.
    """
    Normalize elements of a list

    Args:
        weights (List[float]): The weights

    Returns:
        List[numpy.array]: The normalized weights
    """
    w = np.array(weights, dtype=np.float64)
    w_sum = np.sum(w)
    w = w / w_sum
    return w


def count_dataset_indexes(dataset_idx: np.ndarray, n_datasets: int):
    # [FUNCTION DOC]
    # HELPER: counts samples per dataset in a blended index.
    # Behavior: for each dataset id from 0 to n_datasets minus 1, counts the entries of dataset_idx equal to that id.
    # Inputs: dataset_idx - numpy array of integer dataset ids; n_datasets - number of datasets.
    # Outputs: list of ints of length n_datasets, the count for each dataset id.
    # Exceptions: None explicitly raised. Side effects: None.
    counts = []

    for dataset in range(n_datasets):
        counts.append(np.count_nonzero(dataset_idx == dataset))

    return counts
