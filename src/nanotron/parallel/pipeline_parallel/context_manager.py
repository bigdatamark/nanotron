# [MODULE DOC]
# Purpose: context manager to temporarily attach a pipeline engine state to every PipelineBlock of a model.
# Main contents: attach_pipeline_state_to_model.
# Import-time side effects: None.
from contextlib import contextmanager

from nanotron.parallel.pipeline_parallel.block import PipelineBlock
from nanotron.parallel.pipeline_parallel.state import PipelineBatchState
from torch import nn as torch_nn


@contextmanager
def attach_pipeline_state_to_model(model: torch_nn.Module, pipeline_state: PipelineBatchState):
    # [FUNCTION DOC]
    # Behavior: for every PipelineBlock in model asserts it currently has no state, records it, and sets
    # pipeline_state; yields; finally restores each block's previous state (None).
    # Inputs: model (nn.Module), pipeline_state (PipelineBatchState).
    # Outputs: generator yielding once.
    # Exceptions: AssertionError if a block already has a state (blocks set before the failure are not restored).
    # Side effects: mutates blocks' pipeline_state.
    """Attach the pipeline state to all the PipelineBlocks within `model`"""
    old_pipeline_states = []

    # Set new
    for name, module in model.named_modules():
        if not isinstance(module, PipelineBlock):
            continue

        old_pipeline_state = module.pipeline_state
        assert old_pipeline_state is None, "We never replace an old pipeline engine, we just set one when there's none"

        old_pipeline_states.append((old_pipeline_state, module))

        module.set_pipeline_state(pipeline_state)

    try:
        yield
    finally:
        for old_pipeline_state, module in old_pipeline_states:
            module.set_pipeline_state(old_pipeline_state)
