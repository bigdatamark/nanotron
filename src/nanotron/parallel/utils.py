# [MODULE DOC]
# Purpose: Small parallelism utilities: a global reusable GPU buffer and an initial weight sync.
# Main contents: MemoryBuffer (singleton), initial_sync.
# Fits in nanotron: MemoryBuffer is used by TP primitives to avoid reallocating temporaries;
# initial_sync is called after model building so DP replicas and tied copies start identical.
# Import-time side effects: None.
import functools
import operator

import torch
from torch import nn

from nanotron import distributed as dist
from nanotron.parallel import ParallelContext
from nanotron.parallel.tied_parameters import get_tied_id_to_param
from nanotron.utils import Singleton


class MemoryBuffer(metaclass=Singleton):
    # [CLASS DOC]
    # Purpose: Process-wide singleton cache of flat CUDA tensors keyed by (name, dtype), reused for
    # intermediate activations that are not needed for backward.
    # Key attributes: buffer, dictionary mapping (name, dtype) to a 1-D tensor.
    # Inheritance: uses the Singleton metaclass, so all constructions return the same instance.
    """
    Global memory buffer to store intermediate activations that need not to be cached for the backward pass.
    """

    def __init__(self):
        # [FUNCTION DOC]
        # Behavior: initializes an empty buffer dictionary.
        # Inputs: None. Outputs: None. Exceptions: None explicitly raised. Side effects: None.
        self.buffer = {}

    def get(self, name: str, shape: tuple[int], dtype: torch.dtype = torch.bfloat16) -> torch.Tensor:
        # [FUNCTION DOC]
        # Behavior: computes the element count of shape; if no buffer exists for (name, dtype) or it is
        # smaller, allocates a new uninitialized flat tensor on the current CUDA device; returns a view of the
        # first required elements reshaped to shape (memory shared with later calls using the same key).
        # Inputs: name (str), shape (tuple of ints), dtype (default bfloat16).
        # Outputs: tensor of the requested shape (contents uninitialized/stale).
        # Exceptions: None explicitly raised. Side effects: may allocate CUDA memory and mutate self.buffer.
        required_numel = functools.reduce(operator.mul, shape, 1)
        if (name, dtype) not in self.buffer or self.buffer[name, dtype].numel() < required_numel:
            self.buffer[name, dtype] = torch.empty(
                required_numel, dtype=dtype, device=torch.cuda.current_device(), requires_grad=False
            )
        return self.buffer[name, dtype][:required_numel].view(shape)


def initial_sync(model: nn.Module, parallel_context: ParallelContext):
    # [FUNCTION DOC]
    # Behavior: averages every parameter (sorted by name) across the data-parallel group with all-reduce,
    # then averages each tied parameter (sorted by tied key) across its tied process group.
    # Inputs: model (nn.Module), parallel_context (ParallelContext).
    # Outputs: None. Exceptions: None explicitly raised (KeyError if the tied process group is missing).
    # Side effects: in-place modification of parameters; collectives on dp group and tied groups.
    # Synchronize across dp: basic assumption
    sorted_name_params = sorted(model.named_parameters(), key=lambda x: x[0])
    for name, param in sorted_name_params:
        dist.all_reduce(param, op=dist.ReduceOp.AVG, group=parallel_context.dp_pg)

    # Synchronize across tied weights: basic assumption
    for (_, group_ranks), param in sorted(
        get_tied_id_to_param(parameters=model.parameters(), root_module=model).items(), key=lambda x: x[0]
    ):
        group = parallel_context.world_ranks_to_pg[group_ranks]
        dist.all_reduce(param, op=dist.ReduceOp.AVG, group=group)
