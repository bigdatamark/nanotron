# [MODULE DOC]
# Purpose: distributed pytest test that ParallelContext creates valid process groups and rank mappings for every 3D layout on up to 4 GPUs.
# Main contents: _test_init_parallel_context and test_init_parallel_context.
import numpy as np
import pytest
import torch.distributed as dist
from helpers.utils import (
    available_gpus,
    get_all_3d_configurations,
    init_distributed,
    rerun_if_address_is_in_use,
)
from nanotron.parallel import ParallelContext
from torch.distributed import ProcessGroup


def _test_init_parallel_context(parallel_context: ParallelContext):
    # [FUNCTION DOC]
    # Behavior: per-rank body. Asserts torch distributed is initialized; the world group is a ProcessGroup and tp/pp/dp groups are too when their size exceeds 1; world_rank_matrix is a numpy array and world_ranks_to_pg a dictionary; locates the current world rank's coordinates in the rank matrix and checks get_global_rank on them returns a numpy int64 equal to the global rank. Destroys the context and checks distributed is no longer initialized.
    # Inputs: parallel_context.
    # Outputs: None.
    # Exceptions: AssertionError on failure.
    # Side effects: destroys process groups.
    assert dist.is_initialized() is True
    assert isinstance(parallel_context.world_pg, ProcessGroup)
    assert isinstance(parallel_context.tp_pg, ProcessGroup) if parallel_context.tensor_parallel_size > 1 else True
    assert isinstance(parallel_context.pp_pg, ProcessGroup) if parallel_context.pipeline_parallel_size > 1 else True
    assert isinstance(parallel_context.dp_pg, ProcessGroup) if parallel_context.data_parallel_size > 1 else True

    world_rank = dist.get_rank(parallel_context.world_pg)

    assert isinstance(parallel_context.world_rank_matrix, np.ndarray)
    assert isinstance(parallel_context.world_ranks_to_pg, dict)

    local_rank = tuple(i.item() for i in np.where(parallel_context.world_rank_matrix == world_rank))
    global_rank = parallel_context.get_global_rank(*local_rank)
    assert isinstance(global_rank, np.int64), f"The type of global_rank is {type(global_rank)}"

    assert global_rank == dist.get_rank()

    parallel_context.destroy()
    assert dist.is_initialized() is False


@pytest.mark.parametrize(
    "tp,dp,pp",
    [
        pytest.param(*all_3d_configs)
        for gpus in range(1, min(available_gpus(), 4) + 1)
        for all_3d_configs in get_all_3d_configurations(gpus)
    ],
)
@rerun_if_address_is_in_use()
def test_init_parallel_context(tp: int, dp: int, pp: int):
    # [FUNCTION DOC]
    # Behavior: spawns tp*dp*pp ranks running _test_init_parallel_context.
    # Inputs: tp, dp, pp, parametrized over all layouts for 1 to min(GPUs, 4).
    # Outputs: None.
    # Exceptions: propagates failures.
    # Side effects: process spawning.
    init_distributed(tp=tp, dp=dp, pp=pp)(_test_init_parallel_context)()
