# [MODULE DOC]
# Purpose: Unit tests for DoReMiContext (initialization, domain count/name lookup, weight history recording).
# Import-time side effects: appends the repo root to sys.path via set_system_path.
from copy import deepcopy

import torch
from utils import set_system_path

set_system_path()

from examples.doremi.doremi.doremi_context import DoReMiContext


def test_initialization():
    # [FUNCTION DOC]
    # Behavior: Builds a context with two domains and explicit step size and smoothing param; asserts uniform weights [0.5, 0.5] and that all fields are stored.
    # Inputs: None. Outputs: None. Exceptions: AssertionError on failure. Side effects: None.
    domain_keys = ["domain1", "domain2"]
    step_size, smoothing_param = 0.01, 0.001
    is_proxy = False
    doremi_context = DoReMiContext(domain_keys, is_proxy, step_size, smoothing_param=smoothing_param)

    assert torch.equal(doremi_context.domain_weights, torch.tensor([0.5, 0.5]))
    assert doremi_context.domain_keys == domain_keys
    assert doremi_context.is_proxy == is_proxy
    assert doremi_context.step_size == step_size
    assert doremi_context.smoothing_param == smoothing_param


def test_num_domains():
    # [FUNCTION DOC]
    # Behavior: Asserts num_domains equals 2 for two domain keys.
    # Inputs: None. Outputs: None. Exceptions: AssertionError on failure. Side effects: None.
    domain_keys = ["domain1", "domain2"]
    context = DoReMiContext(domain_keys, False)
    assert context.num_domains == 2


def test_get_domain_name():
    # [FUNCTION DOC]
    # Behavior: Asserts get_domain_name maps indices 0 and 1 to the corresponding keys.
    # Inputs: None. Outputs: None. Exceptions: AssertionError on failure. Side effects: None.
    domain_keys = ["domain1", "domain2"]
    context = DoReMiContext(domain_keys, False)
    assert context.get_domain_name(0) == "domain1"
    assert context.get_domain_name(1) == "domain2"


def test_record_domain_weights_history():
    # [FUNCTION DOC]
    # Behavior: Adds two weight vectors at steps 1 and 2, asserts the current domain_weights are unchanged, and that the history holds the initial uniform weights at step 0 followed by the two added vectors with matching steps.
    # Inputs: None. Outputs: None. Exceptions: AssertionError on failure. Side effects: None.
    domain_weights = [torch.tensor([0.1, 0.3, 0.6]), torch.tensor([0.2, 0.3, 0.5])]
    domain_keys = ["domain1", "domain2", "domain3"]

    doremi_context = DoReMiContext(domain_keys, False)
    initial_domain_weights = deepcopy(doremi_context.domain_weights)

    doremi_context.add_weight_with_history(domain_weights[0], step=1)
    doremi_context.add_weight_with_history(domain_weights[1], step=2)
    assert torch.equal(initial_domain_weights, doremi_context.domain_weights)

    expected_weight_history = [initial_domain_weights, *domain_weights]

    for i, history in enumerate(doremi_context.domain_weight_history):
        assert history["step"] == i
        assert torch.equal(history["weight"], expected_weight_history[i])
