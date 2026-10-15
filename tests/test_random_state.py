# [MODULE DOC]
# Purpose: tests for nanotron.random: syncing RNG states across a process group and the branch_random_state context manager that temporarily swaps in a stored RNG state.
# Main contents: test_random_state_sync (distributed), _test_random_state_sync (per-rank body), and two single-process fork tests.
# Import-time side effects: None.
import pytest
import torch
from helpers.utils import available_gpus, init_distributed, rerun_if_address_is_in_use
from nanotron import distributed as dist
from nanotron.parallel import ParallelContext
from nanotron.random import (
    RandomStates,
    branch_random_state,
    get_current_random_state,
    get_synced_random_state,
)


@pytest.mark.skipif(available_gpus() < 2, reason="Testing test_random_state_sync requires at least 2 gpus")
@pytest.mark.parametrize("tp,dp,pp", [(2, 1, 1), (1, 2, 1), (1, 1, 2)])
@rerun_if_address_is_in_use()
def test_random_state_sync(tp: int, dp: int, pp: int):
    # [FUNCTION DOC]
    # Summary: Pytest entry (needs at least 2 GPUs), parametrized so exactly one of tp/dp/pp equals 2; launches _test_random_state_sync on 2 processes via init_distributed.
    # Inputs: tp, dp, pp - integer parallel sizes.
    # Outputs: None.
    # Exceptions: Propagates worker failures.
    # Side effects: Spawns distributed processes.
    # TODO @nouamane: Make a test with 4 gpus (2 in one pg, 2 in other pg)
    init_distributed(tp=tp, dp=dp, pp=pp)(_test_random_state_sync)()


def _test_random_state_sync(parallel_context: ParallelContext):
    # [FUNCTION DOC]
    # HELPER: per-rank body of test_random_state_sync.
    # Behavior: Captures the current RNG state, picks the first of tp/dp/pp groups whose size is 2, broadcasts rank 0's state with broadcast_object_list and asserts non-reference ranks have a different state. Then calls get_synced_random_state on the group, broadcasts rank 0's synced state again and asserts on non-reference ranks. Finally destroys the parallel context.
    # Inputs: parallel_context - ParallelContext.
    # Outputs: None.
    # Exceptions: AssertionError on mismatch; StopIteration if no group has size 2.
    # Side effects: Collective broadcasts on the chosen group; destroys process groups.
    # KNOWN ISSUE: After syncing, the test asserts that current_random_state differs from the broadcast synced state instead of asserting that synced_random_state equals it, so it does not actually verify the sync (comment says "Check that they are equal").
    current_random_state = get_current_random_state()
    reference_rank = 0
    pg = next(
        (pg for pg in [parallel_context.tp_pg, parallel_context.dp_pg, parallel_context.pp_pg] if pg.size() == 2)
    )

    # Check that they are not equal across process group
    if dist.get_rank(pg) == reference_rank:
        random_states = [current_random_state]
    else:
        random_states = [None]
    dist.broadcast_object_list(random_states, src=reference_rank, group=pg)
    if dist.get_rank(pg) != reference_rank:
        assert current_random_state != random_states[0]

    # Sync random state
    synced_random_state = get_synced_random_state(current_random_state, pg=pg)

    # Check that they are equal across process group
    random_states = [synced_random_state]
    dist.broadcast_object_list(random_states, src=reference_rank, group=pg)
    if dist.get_rank(pg) != reference_rank:
        assert current_random_state != random_states[0]

    parallel_context.destroy()


def test_random_state_fork_random_operation_in_global_context():
    # [FUNCTION DOC]
    # Summary: Verifies that a random op in the global context advances the global RNG state without altering the state stored in RandomStates, that branch_random_state with enabled set activates the stored state inside the context, and that the global state (post-op) is restored afterwards.
    # Inputs: None.
    # Outputs: None.
    # Exceptions: AssertionError on failure.
    # Side effects: Consumes and restores global torch RNG state.
    key = "my_random_state"
    random_state = get_current_random_state()
    random_states = RandomStates({key: random_state})
    assert random_states[key] == random_state

    # Random operation that updates the random state
    torch.randn(1)

    new_random_state = get_current_random_state()

    # Check that random states changed
    assert new_random_state != random_state
    assert random_states[key] == random_state

    # Check that within the context manager the random state matches the one we stored in `random_states`
    with branch_random_state(random_states=random_states, key=key, enabled=True):
        assert random_states[key] == random_state
        assert get_current_random_state() == random_states[key]

    # Check that random states if back to global one
    assert get_current_random_state() == new_random_state


def test_random_state_fork_random_operation_in_local_context():
    # [FUNCTION DOC]
    # Summary: Verifies that a random op inside branch_random_state advances only the stored local state: on exit the global state equals its original value and the RandomStates entry is updated to the state seen at the end of the context.
    # Inputs: None.
    # Outputs: None.
    # Exceptions: AssertionError on failure.
    # Side effects: Uses torch RNG within the context; mutates the local RandomStates.
    key = "my_random_state"
    random_state = get_current_random_state()
    random_states = RandomStates({key: random_state})

    # Check that within the context manager the random state matches the one we stored in `random_states`
    with branch_random_state(random_states=random_states, key=key, enabled=True):
        old_random_state = get_current_random_state()
        assert old_random_state == random_states[key]

        # Random operation that updates the random state
        torch.randn(1)

        # Check that random states changed
        new_random_state = get_current_random_state()

    # Check that global random_state hasn't changed
    assert get_current_random_state() == random_state

    # Check that local random_state has changed and is equal to `new_random_state`
    assert old_random_state != random_states[key]
    assert new_random_state == random_states[key]
