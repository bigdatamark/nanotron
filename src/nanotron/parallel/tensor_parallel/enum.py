# [MODULE DOC]
# Purpose: enum of tensor-parallel linear communication modes. Main contents: TensorParallelLinearMode.
# Import-time side effects: None.
from enum import Enum, auto


class TensorParallelLinearMode(Enum):
    # [CLASS DOC]
    # Purpose: ALL_REDUCE (classic Megatron TP: activations replicated, outputs all-reduced) or REDUCE_SCATTER
    # (sequence-parallel: activations sharded along the sequence, all-gather / reduce-scatter). Formats and stringifies as its name.
    ALL_REDUCE = auto()
    REDUCE_SCATTER = auto()

    def __format__(self, format_spec):
        # [FUNCTION DOC]
        # Returns the member name, ignoring format_spec. Outputs: str.
        return self.name

    def __str__(self):
        # [FUNCTION DOC]
        # Returns the member name. Outputs: str.
        return self.name
