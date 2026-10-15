# [MODULE DOC]
# Purpose: Random number generator state handling for nanotron: capturing, restoring, branching into
# named RNG streams and syncing states across a process group.
# Main contents: RandomState dataclass, RandomStates mapping, set_random_seed, set_random_state,
# get_current_random_state, branch_random_state context manager, get_synced_random_state.
# Fits in nanotron: the trainer keeps named states (e.g. one synced across TP ranks for dropout) and
# checkpoints them via serialize.random.
# Import-time side effects: None.
import contextlib
import random
from dataclasses import dataclass
from typing import MutableMapping, Optional, Tuple

import numpy as np
import torch

from nanotron import distributed as dist
from nanotron.distributed import ProcessGroup


@dataclass
class RandomState:
    # [CLASS DOC]
    # Purpose: Snapshot of all RNGs: Python random, NumPy, torch CPU and (optional) torch CUDA states.
    # Key attributes: random (Python state tuple), numpy (NumPy state tuple), torch_cpu (byte tensor),
    # torch_cuda (byte tensor or None when CUDA is unavailable).
    random: Tuple[int, Tuple[int, ...], None]
    numpy: Tuple[str, np.ndarray, int, int, float]
    torch_cpu: torch.Tensor
    torch_cuda: Optional[torch.Tensor]

    def __eq__(self, other):
        # [FUNCTION DOC]
        # Behavior: True only if other is a RandomState and all components match element-wise (NumPy arrays
        # compared with array_equal, tensors with torch.equal, CUDA states both None or equal).
        # Inputs: other (any). Outputs: bool. Exceptions: None explicitly raised. Side effects: None.
        return (
            isinstance(other, RandomState)
            and all(v1 == v2 for v1, v2 in zip(self.random, other.random))
            and all(
                np.array_equal(v1, v2) if isinstance(v1, np.ndarray) else v1 == v2
                for v1, v2 in zip(self.numpy, other.numpy)
            )
            and torch.equal(self.torch_cpu, other.torch_cpu)
            and (
                other.torch_cuda is None if self.torch_cuda is None else torch.equal(self.torch_cuda, other.torch_cuda)
            )
        )


class RandomStates(MutableMapping[str, RandomState]):
    # [CLASS DOC]
    # Purpose: Fixed-key mutable mapping from string names to RandomState; keys cannot be added or
    # deleted after construction, only values replaced.
    # Key attributes: _dict (private copy of the initial dictionary).
    # Inheritance: implements MutableMapping.
    def __init__(self, dict: dict):
        # [FUNCTION DOC]
        # Behavior: type-checks every entry then stores a shallow copy.
        # Inputs: dict, dictionary of str to RandomState.
        # Outputs: None. Exceptions: ValueError on wrong key/value types. Side effects: None.
        for key, value in dict.items():
            self.check_type(key, value)
        # TODO @thomasw21: We make a copy for safety measure.
        self._dict = dict.copy()

    @staticmethod
    def check_type(key, value):
        # [FUNCTION DOC]
        # HELPER: validation for keys and values.
        # Behavior: verifies key is a str and value is a RandomState.
        # Inputs: key, value. Outputs: None. Exceptions: ValueError on mismatch. Side effects: None.
        if not isinstance(key, str):
            raise ValueError(f"Expected key to be of type str. Got {type(key)}")
        if not isinstance(value, RandomState):
            raise ValueError(f"Expected value to be of type `nanotron.dataclass.RandomState`. Got {type(value)}")

    def __getitem__(self, item):
        # [FUNCTION DOC]
        # Behavior: returns the stored state for item. Inputs: item (str). Outputs: RandomState.
        # Exceptions: KeyError if missing. Side effects: None.
        return self._dict[item]

    def __iter__(self):
        # [FUNCTION DOC]
        # Behavior: iterates over keys. Inputs: None. Outputs: key iterator. Exceptions: None. Side effects: None.
        return self._dict.__iter__()

    def __len__(self):
        # [FUNCTION DOC]
        # Behavior: returns number of keys. Inputs: None. Outputs: int. Exceptions: None. Side effects: None.
        return len(self._dict)

    def __delitem__(self, key):
        # [FUNCTION DOC]
        # Behavior: always refuses deletion. Inputs: key. Outputs: None.
        # Exceptions: ValueError always. Side effects: None.
        raise ValueError("Can't delete a random states key")

    def __setitem__(self, key, value):
        # [FUNCTION DOC]
        # Behavior: replaces the value for an existing key after type checking.
        # Inputs: key (str), value (RandomState). Outputs: None.
        # Exceptions: ValueError if key is new or types are wrong. Side effects: mutates the mapping.
        if key not in self._dict:
            raise ValueError("Can't add a new random states after initialisation")
        self.check_type(key, value)
        return self._dict.__setitem__(key, value)

    def __eq__(self, other):
        # [FUNCTION DOC]
        # Behavior: True if other is a RandomStates with an equal underlying dictionary.
        # Inputs: other. Outputs: bool. Exceptions: None explicitly raised. Side effects: None.
        if not isinstance(other, RandomStates):
            return False

        return self._dict == other._dict


def set_random_seed(seed: int):
    # [FUNCTION DOC]
    # Behavior: seeds torch (CPU and, if available, current CUDA device), NumPy and Python random with seed.
    # Inputs: seed (int). Outputs: None. Exceptions: None explicitly raised.
    # Side effects: global RNG state changes.
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def set_random_state(random_state: RandomState):
    # [FUNCTION DOC]
    # Behavior: restores Python, NumPy and torch CPU states, and the CUDA state on device "cuda" if CUDA is
    # available; otherwise asserts the stored CUDA state is None.
    # Inputs: random_state (RandomState). Outputs: None.
    # Exceptions: AssertionError if CUDA is unavailable but a CUDA state is present.
    # Side effects: global RNG state changes.
    random.setstate(random_state.random)
    np.random.set_state(random_state.numpy)
    torch.set_rng_state(random_state.torch_cpu)
    if torch.cuda.is_available():
        torch.cuda.set_rng_state(random_state.torch_cuda, "cuda")
    else:
        assert random_state.torch_cuda is None


def get_current_random_state():
    # [FUNCTION DOC]
    # Behavior: captures current Python, NumPy, torch CPU and (if available) CUDA RNG states.
    # Inputs: None. Outputs: RandomState. Exceptions: None explicitly raised. Side effects: None.
    """Returns a snapshot of current random state"""
    return RandomState(
        random=random.getstate(),
        numpy=np.random.get_state(),
        torch_cpu=torch.random.get_rng_state(),
        torch_cuda=torch.cuda.get_rng_state("cuda") if torch.cuda.is_available() else None,
    )


@contextlib.contextmanager
def branch_random_state(random_states: RandomStates, key: str, enabled: bool):
    # [FUNCTION DOC]
    # Behavior: context manager (generator). If not enabled, yields without touching RNGs. Otherwise saves
    # the current state, installs random_states[key], yields, and on exit (even on error) stores the
    # advanced state back into random_states[key] and restores the saved state.
    # Inputs: random_states (RandomStates), key (str), enabled (bool).
    # Outputs: yields None.
    # Exceptions: KeyError if key missing; propagates exceptions from the body.
    # Side effects: temporarily changes global RNG state; mutates random_states.
    """
    Context manager handling random state:
     - upon entering: Stores current random state and set new random state defined by key.
     - upon exiting: updates key in `random_states` to the new current random state, and set back the old one.
    """
    if not enabled:
        yield
        return

    old_random_state = get_current_random_state()

    # Get the new state associated to the key
    new_random_state = random_states[key]
    set_random_state(new_random_state)

    try:
        yield
    finally:
        # Update state from parallel_context with the newest state
        new_random_state = get_current_random_state()
        random_states[key] = new_random_state

        # Set the old state back
        set_random_state(old_random_state)


def get_synced_random_state(
    random_state: RandomState,
    pg: ProcessGroup,
):
    # [FUNCTION DOC]
    # Behavior: rank 0 of pg broadcasts its random_state (pickled object broadcast via a CUDA device) to
    # all ranks of pg; every rank returns the received state.
    # Inputs: random_state (RandomState, only meaningful on rank 0 of pg), pg (ProcessGroup).
    # Outputs: RandomState identical across pg.
    # Exceptions: AssertionError if the received state is None.
    # Side effects: broadcast collective on pg.
    # We use rank 0 as a reference and broadcast random states from that rank to all the other ranks within a group in order to sync them
    reference_rank = 0
    if dist.get_rank(pg) == reference_rank:
        random_states = [random_state]
    else:
        random_states = [None]

    # TODO @thomasw21: broadcast tensor using `broadcast` in order not to use pickle
    dist.broadcast_object_list(
        random_states, src=dist.get_global_rank(pg, reference_rank), group=pg, device=torch.device("cuda")
    )

    new_random_state = random_states[0]
    assert new_random_state is not None
    return new_random_state
