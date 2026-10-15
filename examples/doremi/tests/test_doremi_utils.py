# [MODULE DOC]
# Purpose: Unit test for compute_domain_weights_based_on_token_count in the DoReMi utils.
# Import-time side effects: appends the repo root to sys.path.
import torch
from utils import create_dummy_dataset, set_system_path

set_system_path()

from examples.doremi.doremi.utils import compute_domain_weights_based_on_token_count


def test_compute_domain_weights_based_on_token_count():
    # [FUNCTION DOC]
    # Behavior: Builds datasets of sizes 10, 20 and 70 and asserts the weights equal [0.1, 0.2, 0.7] exactly and sum to 1.
    # Inputs: None. Outputs: None. Exceptions: AssertionError on failure. Side effects: None.
    datasets = [
        create_dummy_dataset(10),
        create_dummy_dataset(20),
        create_dummy_dataset(70),
    ]

    domain_weights = compute_domain_weights_based_on_token_count(datasets)

    assert torch.equal(domain_weights, torch.tensor([0.1, 0.2, 0.7]))
    assert torch.allclose(domain_weights.sum(), torch.tensor(1.0))
