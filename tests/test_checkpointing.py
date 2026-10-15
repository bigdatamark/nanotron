# [MODULE DOC]
# Purpose: pytest test for nanotron.utils checkpoint_method: a checkpointed submodule must produce identical outputs and gradients while running its forward twice and storing no activations.
# Main contents: CheckpointedModel, DummyModel, test_activation_checkpointing.
from typing import Union

import torch
from nanotron.parallel.pipeline_parallel.tensor_pointer import TensorPointer
from nanotron.utils import checkpoint_method
from torch import nn


class CheckpointedModel(nn.Module):
    # [CLASS DOC]
    # Purpose: two linear(10,10) layers plus dropout(0.1) whose forward is decorated with checkpoint_method keyed on is_checkpointed.
    # Key attributes: dense1, dense2, dropout; is_checkpointed, boolean; fwd_counter, number of forward executions (including recomputation).
    # Construction inputs: is_checkpointed, default False.
    # Inheritance role: nn.Module.
    def __init__(self, is_checkpointed: bool = False):
        # [FUNCTION DOC]
        # Behavior: creates the layers and sets the flag and counter to 0.
        # Inputs: is_checkpointed, boolean.
        # Outputs: None.
        # Exceptions: None explicitly raised.
        # Side effects: parameter initialization uses the torch RNG.
        super().__init__()
        self.dense1 = nn.Linear(10, 10)
        self.dense2 = nn.Linear(10, 10)
        self.dropout = nn.Dropout(0.1)
        self.is_checkpointed = is_checkpointed
        self.fwd_counter = 0

    @checkpoint_method("is_checkpointed")
    def forward(self, x: Union[torch.Tensor, TensorPointer]):
        # [FUNCTION DOC]
        # Behavior: applies dense1; on the first checkpointed execution asserts the intermediate has no grad and no grad_fn (no-grad mode); applies dense2 and dropout; increments fwd_counter.
        # Inputs: x, tensor of shape batch by 10.
        # Outputs: tensor of the same shape.
        # Exceptions: AssertionError if activations are recorded on the first checkpointed pass.
        # Side effects: increments self.fwd_counter; consumes dropout RNG.
        x = self.dense1(x)
        if self.is_checkpointed and self.fwd_counter == 0:
            assert not x.requires_grad, "x should not require grad when checkpointed, because fwd runs in no_grad mode"
            assert (
                x.grad_fn is None
            ), "x should not store any activation when checkpointed, because fwd runs in no_grad mode"
        x = self.dense2(x)
        x = self.dropout(x)
        self.fwd_counter += 1
        return x


class DummyModel(nn.Module):
    # [CLASS DOC]
    # Purpose: wraps CheckpointedModel between two plain linear layers (dense0, dense3).
    # Key attributes: dense0, checkpointed_model, dense3.
    # Construction inputs: is_checkpointed, forwarded to CheckpointedModel.
    # Inheritance role: nn.Module.
    def __init__(self, is_checkpointed: bool = False):
        # [FUNCTION DOC]
        # Behavior: creates the three submodules.
        # Inputs: is_checkpointed, boolean, default False.
        # Outputs: None.
        # Exceptions: None explicitly raised.
        # Side effects: parameter initialization uses the torch RNG.
        super().__init__()
        self.dense0 = nn.Linear(10, 10)
        self.checkpointed_model = CheckpointedModel(is_checkpointed=is_checkpointed)
        self.dense3 = nn.Linear(10, 10)

    def forward(self, x: Union[torch.Tensor, TensorPointer]):
        # [FUNCTION DOC]
        # Behavior: dense0, then the checkpointed model (asserting its output requires grad), then dense3.
        # Inputs: x, tensor batch by 10.
        # Outputs: tensor batch by 10.
        # Exceptions: AssertionError if the output does not require grad.
        # Side effects: as submodules.
        x = self.dense0(x)
        x = self.checkpointed_model(x)
        assert x.requires_grad  # inside forward, x should require grad even if calculated in no_grad mode
        x = self.dense3(x)
        return x


def test_activation_checkpointing():
    # [FUNCTION DOC]
    # Behavior: builds checkpointed and reference float16 CUDA models with identical weights; runs the reference forward inside a forked CUDA RNG so both see the same dropout mask; checks outputs are equal and the checkpointed forward ran once; after reference backward checks its counter is still 1 and the test model has no grad yet; after checkpointed backward checks its counter is 2; finally compares all parameter gradients.
    # Inputs: none.
    # Outputs: None.
    # Exceptions: AssertionError on failure.
    # Side effects: GPU allocation and RNG use.
    dtype = torch.float16
    device = torch.device("cuda")
    test_model = DummyModel(is_checkpointed=True)
    ref_model = DummyModel(is_checkpointed=False)
    for model in [test_model, ref_model]:
        model.to(device=device, dtype=dtype)

    # copy weights
    test_model.load_state_dict(ref_model.state_dict())
    assert test_model.checkpointed_model.is_checkpointed is True
    assert ref_model.checkpointed_model.is_checkpointed is False

    # generate random input
    x = torch.randn(10, 10, device=device, dtype=dtype)

    # Forward pass
    with torch.random.fork_rng(devices=["cuda"]):
        ref_output = ref_model(x)
    checkpointed_output = test_model(x)
    assert test_model.checkpointed_model.fwd_counter == 1
    torch.testing.assert_close(checkpointed_output, ref_output)

    # Backward pass (check that fwd is called twice, and that we don't store the activations)
    ref_output.sum().backward()
    assert ref_model.checkpointed_model.fwd_counter == 1, "ref_model fwd should not be called twice"

    # make sure grads are not synced between test_model and ref_model
    assert ref_model.dense0.weight.grad is not None
    assert test_model.dense0.weight.grad is None

    assert test_model.checkpointed_model.fwd_counter == 1
    checkpointed_output.sum().backward()
    assert test_model.checkpointed_model.fwd_counter == 2, "test_model fwd should be called twice"

    # compare all models grads
    for ref_param, checkpointed_param in zip(ref_model.parameters(), test_model.parameters()):
        torch.testing.assert_close(ref_param.grad, checkpointed_param.grad)


# TODO @nouamanetazi: test `checkpoint_method` vs `torch.utils.checkpoint.checkpoint`
# TODO @nouamanetazi: test a method with kwargs values
# TODO @nouamanetazi: test `checkpoint_method` in a distributed setting
# TODO @nouamanetazi: test BatchNorm layers with checkpointing
