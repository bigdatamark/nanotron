# [MODULE DOC]
# Purpose: Unit tests for CombinedDataset from the DoReMi dataloader (length and indexing with flat and nested index lists).
# Main contents: fixtures dataset1 (4000 items) and dataset2 (6000 items) and two tests.
# Import-time side effects: appends the repo root to sys.path.
import pytest
from utils import create_dummy_dataset, set_system_path

set_system_path()
from examples.doremi.doremi.dataloader import CombinedDataset


@pytest.fixture
def dataset1():
    # [FUNCTION DOC]
    # HELPER: pytest fixture.
    # Behavior: Returns a dummy dataset with 4000 items. Inputs: None. Outputs: datasets.Dataset. Exceptions: None. Side effects: None.
    return create_dummy_dataset(4000)


@pytest.fixture
def dataset2():
    # [FUNCTION DOC]
    # HELPER: pytest fixture.
    # Behavior: Returns a dummy dataset with 6000 items. Inputs: None. Outputs: datasets.Dataset. Exceptions: None. Side effects: None.
    return create_dummy_dataset(6000)


def test_combined_dataset_length(dataset1, dataset2):
    # [FUNCTION DOC]
    # Behavior: Asserts the combined length equals the sum of both dataset lengths.
    # Inputs: dataset1, dataset2 fixtures. Outputs: None. Exceptions: AssertionError on failure. Side effects: None.
    combined_dataset = CombinedDataset([dataset1, dataset2])
    assert len(combined_dataset) == len(dataset1) + len(dataset2)


@pytest.mark.parametrize("idx_type", ["idxs", "batch_of_idxs"])
def test_get_item_from_combined_dataset(dataset1, dataset2, idx_type):
    # [FUNCTION DOC]
    # Behavior: Parametrized over flat indices [0, 1] and nested indices (first two and last two global indices). Fetches from CombinedDataset and asserts the result is a dictionary with the same keys as the source, the right total count, and that values match the first dataset's first two items and (nested case) the second dataset's last two items.
    # Inputs: dataset1, dataset2 fixtures; idx_type: "idxs" or "batch_of_idxs".
    # Outputs: None. Exceptions: AssertionError on failure. Side effects: None.
    def count_elements(lst):
        # [FUNCTION DOC]
        # HELPER: counts leaves in a nested list.
        # Behavior: Recursively sums 1 for each non-list element.
        # Inputs: lst: possibly nested list. Outputs: int. Exceptions: None. Side effects: None.
        return sum(count_elements(i) if isinstance(i, list) else 1 for i in lst)

    if idx_type == "batch_of_idxs":
        total_samples = len(dataset1) + len(dataset2)
        idxs = [[0, 1], [total_samples - 2, total_samples - 1]]
    else:
        idxs = [0, 1]

    combined_dataset = CombinedDataset([dataset1, dataset2])
    outputs = combined_dataset[idxs]
    # NOTE: obtain the first key in a dict
    first_key = next(iter(outputs))

    assert isinstance(outputs, dict)
    assert outputs.keys() == dataset1[0].keys()
    assert len(outputs[first_key]) == count_elements(idxs)

    assert outputs[first_key][0] == dataset1[0][first_key]
    assert outputs[first_key][1] == dataset1[1][first_key]
    if idx_type == "batch_of_idxs":
        assert outputs[first_key][2] == dataset2[len(dataset2) - 2][first_key]
        assert outputs[first_key][3] == dataset2[len(dataset2) - 1][first_key]
