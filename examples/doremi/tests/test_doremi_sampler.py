# [MODULE DOC]
# Purpose: Distributed unit tests for DistributedSamplerForDoReMi and its use with CombinedDataset in a DataLoader: tp synchronization, no dp overlap, determinism across epochs, per-domain proportions of the global batch, no repeated samples, and per-domain counters while yielding.
# Structure: public parametrized tests set up dummy datasets and a DoReMiContext (uniform weights), then run a private worker via tests.helpers.utils.init_distributed.
# Import-time side effects: appends the repo root to sys.path; requires CUDA at run time.
import pytest
import torch
from nanotron import distributed as dist
from nanotron.parallel import ParallelContext
from nanotron.sanity_checks import assert_tensor_synced_across_pg
from torch.utils.data import DataLoader
from utils import create_dummy_dataset, set_system_path

set_system_path()

from examples.doremi.doremi.dataloader import (
    CombinedDataset,
    DistributedSamplerForDoReMi,
)
from examples.doremi.doremi.doremi_context import DoReMiContext
from tests.helpers.utils import init_distributed


@pytest.fixture
def dataset1():
    # [FUNCTION DOC]
    # HELPER: pytest fixture returning a dummy dataset with 7000 items.
    # Inputs: None. Outputs: datasets.Dataset. Exceptions: None. Side effects: None.
    return create_dummy_dataset(7000)


@pytest.fixture
def dataset2():
    # [FUNCTION DOC]
    # HELPER: pytest fixture returning a dummy dataset with 3000 items.
    # Inputs: None. Outputs: datasets.Dataset. Exceptions: None. Side effects: None.
    return create_dummy_dataset(3000)


@pytest.fixture
def datasets(dataset1, dataset2):
    # [FUNCTION DOC]
    # HELPER: pytest fixture returning the list of dataset1 and dataset2.
    # Inputs: dataset1, dataset2. Outputs: list of datasets. Exceptions: None. Side effects: None.
    return [dataset1, dataset2]


@pytest.mark.parametrize("num_microbatches", [1, 32])
@pytest.mark.parametrize("is_proxy", [True, False])
def test_dist_doremi_sampler_sync_across_tp(num_microbatches, dataset1, is_proxy):
    # [FUNCTION DOC]
    # Behavior: Two domains (both dataset1), micro batch 16; runs the worker with tp=2.
    # Inputs: num_microbatches (1 or 32); dataset1; is_proxy (bool). Outputs: None. Exceptions: AssertionError in workers. Side effects: spawns processes.
    NUM_DOMAINS = 2
    BATCH_SIZE = 16

    datasets = [dataset1 for _ in range(NUM_DOMAINS)]
    domain_keys = [f"domain {i}" for i in range(NUM_DOMAINS)]
    doremi_context = DoReMiContext(domain_keys, is_proxy=is_proxy)

    init_distributed(tp=2, dp=1, pp=1)(_test_dist_doremi_sampler_sync_across_tp)(
        batch_size=BATCH_SIZE,
        num_microbatches=num_microbatches,
        datasets=datasets,
        doremi_context=doremi_context,
    )


def _test_dist_doremi_sampler_sync_across_tp(
    parallel_context: ParallelContext, batch_size: int, num_microbatches: int, datasets, doremi_context: DoReMiContext
):
    # [FUNCTION DOC]
    # HELPER: worker checking that tp ranks get identical indices.
    # Behavior: builds a DistributedSamplerForDoReMi for this rank's dp rank and size, iterates to exhaustion and asserts each micro-batch of indices is synced across the tp group.
    # Inputs: parallel_context; batch_size; num_microbatches; datasets; doremi_context. Outputs: None.
    # Exceptions: AssertionError if unsynced. Side effects: tp collectives.
    dp_size = dist.get_world_size(parallel_context.dp_pg)
    dp_rank = dist.get_rank(parallel_context.dp_pg)

    sampler = DistributedSamplerForDoReMi(
        datasets,
        batch_size=batch_size,
        num_microbatches=num_microbatches,
        num_replicas=dp_size,
        rank=dp_rank,
        doremi_context=doremi_context,
        parallel_context=parallel_context,
    )

    for idxs in sampler:
        idxs = torch.tensor(idxs, device="cuda")
        assert_tensor_synced_across_pg(idxs, parallel_context.tp_pg)


@pytest.mark.parametrize("dp_size", [2, 4])
@pytest.mark.parametrize("num_microbatches", [1, 32])
@pytest.mark.parametrize("is_proxy", [True, False])
def test_dist_doremi_sampler_not_overlapse_across_dp_for_proxy_training(dp_size, num_microbatches, dataset1, is_proxy):
    # [FUNCTION DOC]
    # Behavior: Two domains, global batch 512 divided by micro-batches and dp_size; runs the worker.
    # Inputs: dp_size (2 or 4); num_microbatches; dataset1; is_proxy. Outputs: None. Exceptions: AssertionError in workers. Side effects: spawns processes.
    # KNOWN ISSUE: dp_size is used to compute batch_size but the distributed launch is hard-coded to dp=2.
    NUM_DOMAINS = 2
    GLOBAL_BATCH_SIZE = 512
    batch_size = GLOBAL_BATCH_SIZE // (num_microbatches * dp_size)

    datasets = [dataset1 for _ in range(NUM_DOMAINS)]
    domain_keys = [f"domain {i}" for i in range(NUM_DOMAINS)]
    doremi_context = DoReMiContext(domain_keys, is_proxy=is_proxy)

    init_distributed(tp=1, dp=2, pp=1)(_test_dist_doremi_sampler_not_overlapse_across_dp_for_proxy_training)(
        batch_size=batch_size,
        num_microbatches=num_microbatches,
        datasets=datasets,
        doremi_context=doremi_context,
    )


def _test_dist_doremi_sampler_not_overlapse_across_dp_for_proxy_training(
    parallel_context: ParallelContext,
    batch_size: int,
    num_microbatches: int,
    datasets,
    doremi_context: DoReMiContext,
):
    # [FUNCTION DOC]
    # HELPER: worker checking dp ranks receive disjoint indices.
    # Behavior: builds a DistributedSamplerForDoReMi for this rank's dp rank and size; for each micro-batch all-gathers indices over the world group and asserts the gathered tensors have no common elements.
    # Inputs: parallel_context; batch_size; num_microbatches; datasets; doremi_context. Outputs: None.
    # Exceptions: AssertionError on overlap. Side effects: all_gather collectives.
    # KNOWN ISSUE: unpacks the gathered list into torch.isin, which only works for exactly two ranks.
    dp_size = dist.get_world_size(parallel_context.dp_pg)
    dp_rank = dist.get_rank(parallel_context.dp_pg)

    sampler = DistributedSamplerForDoReMi(
        datasets,
        batch_size=batch_size,
        num_microbatches=num_microbatches,
        num_replicas=dp_size,
        rank=dp_rank,
        doremi_context=doremi_context,
        parallel_context=parallel_context,
    )

    for idxs in sampler:
        idxs = torch.tensor(idxs, device="cuda").view(-1)

        # NOTE: i tried to use assert_fail_except_rank_with, but it mark the test as failed
        # even the test raises an exception as expected
        gathered_idxs = [torch.empty_like(idxs, device="cuda") for _ in range(dp_size)]
        dist.all_gather(gathered_idxs, idxs)

        # NOTE: whether proxy or reference training
        # the idxs should not be overlapse
        assert not torch.any(torch.isin(*gathered_idxs))


@pytest.mark.parametrize("num_microbatches", [1, 32])
@pytest.mark.parametrize("is_proxy", [True, False])
def test_determistic_doremi_sampler(num_microbatches, dataset1, is_proxy):
    # [FUNCTION DOC]
    # Behavior: Two domains, micro batch 100, three epochs, single process.
    # Inputs: num_microbatches; dataset1; is_proxy. Outputs: None. Exceptions: AssertionError in worker. Side effects: spawns a process.
    BATCH_SIZE = 100
    NUM_DOMAINS = 2

    datasets = [dataset1 for _ in range(NUM_DOMAINS)]
    domain_keys = [f"domain {i}" for i in range(NUM_DOMAINS)]
    doremi_context = DoReMiContext(domain_keys, is_proxy=is_proxy)
    n_epochs = 3

    init_distributed(tp=1, dp=1, pp=1)(_test_determistic_doremi_sampler)(
        batch_size=BATCH_SIZE,
        num_microbatches=num_microbatches,
        datasets=datasets,
        doremi_context=doremi_context,
        n_epochs=n_epochs,
    )


def _test_determistic_doremi_sampler(
    parallel_context: ParallelContext,
    batch_size: int,
    num_microbatches: int,
    n_epochs: int,
    datasets,
    doremi_context: DoReMiContext,
):
    # [FUNCTION DOC]
    # HELPER: worker checking sampler determinism across epochs.
    # Behavior: builds a DistributedSamplerForDoReMi for this rank's dp rank and size, collects all micro-batches per epoch calling reset between epochs, and asserts consecutive epochs produced the same sequences.
    # Inputs: parallel_context; batch_size; num_microbatches; n_epochs; datasets; doremi_context. Outputs: None.
    # Exceptions: AssertionError on mismatch. Side effects: None beyond sampler state.
    dp_size = dist.get_world_size(parallel_context.dp_pg)
    dp_rank = dist.get_rank(parallel_context.dp_pg)

    sampler = DistributedSamplerForDoReMi(
        datasets,
        batch_size=batch_size,
        num_microbatches=num_microbatches,
        num_replicas=dp_size,
        rank=dp_rank,
        doremi_context=doremi_context,
        parallel_context=parallel_context,
    )

    idxs_per_epoch = []
    for _ in range(n_epochs):
        all_idxs = []
        for idxs in sampler:
            all_idxs.append(idxs)

        idxs_per_epoch.append(all_idxs)
        sampler.reset()

    # NOTE: check if the sequence of idxs across epochs are all the same
    assert all(
        all(arr1[i] == arr2[i] for i in range(len(arr1))) for arr1, arr2 in zip(idxs_per_epoch, idxs_per_epoch[1:])
    )


@pytest.mark.parametrize("dp_size", [1, 2, 4])
@pytest.mark.parametrize("num_microbatches", [1, 32])
@pytest.mark.parametrize("is_proxy", [True, False])
def test_sampling_from_dist_doremi_sampler_with_global_batch_size(
    dp_size,
    num_microbatches,
    # domain_weights: torch.Tensor,
    dataset1,
    is_proxy,
):
    # [FUNCTION DOC]
    # Behavior: Eight domains, global batch 512 split across micro-batches and dp; runs the worker with dp_size.
    # Inputs: dp_size (1, 2, 4); num_microbatches; dataset1; is_proxy. Outputs: None. Exceptions: AssertionError in workers. Side effects: spawns processes.
    NUM_DOMAINS = 8
    GLOBAL_BATCH_SIZE = 512
    batch_size = GLOBAL_BATCH_SIZE // (num_microbatches * dp_size)

    datasets = [dataset1 for _ in range(NUM_DOMAINS)]
    domain_keys = [f"domain {i}" for i in range(NUM_DOMAINS)]
    doremi_context = DoReMiContext(domain_keys, is_proxy=is_proxy)

    init_distributed(tp=1, dp=dp_size, pp=1)(_test_sampling_from_dist_doremi_sampler_with_global_batch_size)(
        batch_size=batch_size,
        num_microbatches=num_microbatches,
        global_batch_size=GLOBAL_BATCH_SIZE,
        datasets=datasets,
        doremi_context=doremi_context,
    )


def _test_sampling_from_dist_doremi_sampler_with_global_batch_size(
    parallel_context: ParallelContext,
    batch_size: int,
    num_microbatches: int,
    global_batch_size: int,
    datasets,
    doremi_context: DoReMiContext,
):
    # [FUNCTION DOC]
    # HELPER: worker checking per-domain counts of each global batch match the domain weights.
    # Behavior: builds a DistributedSamplerForDoReMi for this rank's dp rank and size; computes expected per-domain counts by rounding global batch times weight; for each micro-batch asserts its length and counts indices per domain by offset range; after the last micro-batch of a global batch, all-reduces counts (sum) and asserts every domain is non-empty and within dp_size of expected, then resets the counts.
    # Inputs: parallel_context; batch_size; num_microbatches; global_batch_size; datasets; doremi_context. Outputs: None.
    # Exceptions: AssertionError on failure. Side effects: all_reduce collectives.
    dp_size = dist.get_world_size(parallel_context.dp_pg)
    dp_rank = dist.get_rank(parallel_context.dp_pg)

    sampler = DistributedSamplerForDoReMi(
        datasets,
        batch_size=batch_size,
        num_microbatches=num_microbatches,
        num_replicas=dp_size,
        rank=dp_rank,
        doremi_context=doremi_context,
        parallel_context=parallel_context,
    )

    domain_weights = doremi_context.domain_weights
    global_batch_size_per_domain = [round(global_batch_size * weight.item()) for weight in domain_weights]

    microbatch_idx = 0
    num_samples_per_domain = [0 for _ in range(len(domain_weights))]
    for idxs in sampler:
        assert batch_size == len(idxs)

        # NOTE: make sure the indices from a batch
        # is proportion to the domain weights
        start_indices = [sum([len(ds) for ds in datasets[:i]]) for i in range(len(datasets))]
        end_indices = [sum([len(ds) for ds in datasets[: i + 1]]) for i in range(len(datasets))]
        for domain_idx in range(len(domain_weights)):
            num_samples = sum(1 for idx in idxs if idx >= start_indices[domain_idx] and idx < end_indices[domain_idx])
            num_samples_per_domain[domain_idx] += num_samples

        if microbatch_idx == num_microbatches - 1:
            # NOTE: if this is the last microbatch => we iterate through all the microbatches
            # now we check if the overall number of samples in each domain is correct across
            # all the microbatches
            num_samples_per_domain = torch.tensor(num_samples_per_domain, dtype=torch.int, device="cuda")

            # NOTE: the domain weights are chosen so that we expect
            # no domains have zero sample in the global batch size
            dist.all_reduce(num_samples_per_domain, op=dist.ReduceOp.SUM)
            assert (num_samples_per_domain == 0).sum().item() == 0

            for expected_bs, bs in zip(global_batch_size_per_domain, num_samples_per_domain):
                assert bs > 0
                # NOTE: take into account rounding errors
                # across all the dp ranks
                assert abs(expected_bs - bs) <= dp_size, f"abs(expected_bs - bs): {abs(expected_bs - bs)}"

            microbatch_idx = 0
            num_samples_per_domain = [0 for _ in range(len(domain_weights))]
        else:
            microbatch_idx += 1


@pytest.mark.parametrize("dp_size", [1, 2, 4])
@pytest.mark.parametrize("num_microbatches", [1, 32])
@pytest.mark.parametrize("is_proxy", [True, False])
def test_dist_doremi_sampler_not_repeating_samples(dp_size, num_microbatches, dataset1, is_proxy):
    # [FUNCTION DOC]
    # Behavior: Two domains, global batch 512; runs the worker with dp_size.
    # Inputs: dp_size (1, 2, 4); num_microbatches; dataset1; is_proxy. Outputs: None. Exceptions: AssertionError in workers. Side effects: spawns processes.
    NUM_DOMAINS = 2
    GLOBAL_BATCH_SIZE = 512
    batch_size = GLOBAL_BATCH_SIZE // (num_microbatches * dp_size)

    datasets = [dataset1 for _ in range(NUM_DOMAINS)]
    domain_keys = [f"domain {i}" for i in range(NUM_DOMAINS)]
    doremi_context = DoReMiContext(domain_keys, is_proxy=is_proxy)

    init_distributed(tp=1, dp=dp_size, pp=1)(_test_dist_doremi_sampler_not_repeating_samples)(
        batch_size=batch_size,
        num_microbatches=num_microbatches,
        datasets=datasets,
        doremi_context=doremi_context,
    )


def _test_dist_doremi_sampler_not_repeating_samples(
    parallel_context: ParallelContext,
    batch_size: int,
    num_microbatches: int,
    datasets,
    doremi_context: DoReMiContext,
):
    # [FUNCTION DOC]
    # HELPER: worker checking no index is ever yielded twice, locally or globally.
    # Behavior: builds a DistributedSamplerForDoReMi for this rank's dp rank and size; for each micro-batch asserts no intersection with previously yielded local and global indices, then all-gathers indices over dp and records them; finally asserts the global record has no duplicates.
    # Inputs: parallel_context; batch_size; num_microbatches; datasets; doremi_context. Outputs: None.
    # Exceptions: AssertionError on repeats. Side effects: all_gather collectives.
    # KNOWN ISSUE: the variable named epoch actually counts micro-batches.
    dp_size = dist.get_world_size(parallel_context.dp_pg)
    dp_rank = dist.get_rank(parallel_context.dp_pg)

    sampler = DistributedSamplerForDoReMi(
        datasets,
        batch_size=batch_size,
        num_microbatches=num_microbatches,
        num_replicas=dp_size,
        rank=dp_rank,
        doremi_context=doremi_context,
        parallel_context=parallel_context,
    )

    local_yieled_idxs = []
    yielded_idxs = []
    epoch = 0
    for idxs in sampler:
        # NOTE: check that the indices are not repeated
        assert not set(idxs).intersection(
            local_yieled_idxs
        ), f"set(idxs): {set(idxs)}, local_yieled_idxs: {local_yieled_idxs}"
        assert not set(idxs).intersection(
            yielded_idxs
        ), f"set(idxs): {set(idxs)}, yielded_idxs: {yielded_idxs} \
        epoch: {epoch}"

        local_yieled_idxs.extend(idxs)

        # NOTE: gather all the indices from all the dp ranks
        idxs = torch.tensor(idxs, dtype=torch.int, device="cuda")
        all_idxs = [torch.zeros_like(idxs) for _ in range(dp_size)]
        dist.all_gather(all_idxs, idxs)
        all_idxs = torch.cat(all_idxs, dim=0).view(-1).cpu().tolist()
        yielded_idxs.extend(all_idxs)
        epoch += 1

    assert len(set(yielded_idxs)) == len(yielded_idxs)


@pytest.mark.parametrize("dp_size", [2, 4, 8])
@pytest.mark.parametrize("num_microbatches", [1, 5])
@pytest.mark.parametrize("is_proxy", [True, False])
def test_yielding(dp_size, num_microbatches, dataset1, is_proxy):
    # [FUNCTION DOC]
    # Behavior: Two domains, micro batch 100, global batch 100*micro-batches*dp; runs the worker.
    # Inputs: dp_size (2, 4, 8); num_microbatches (1 or 5); dataset1; is_proxy. Outputs: None. Exceptions: AssertionError in workers. Side effects: spawns processes.
    NUM_DOMAINS = 2
    BATCH_SIZE = 100
    global_batch_size = BATCH_SIZE * num_microbatches * dp_size

    datasets = [dataset1 for _ in range(NUM_DOMAINS)]
    domain_keys = [f"domain {i}" for i in range(NUM_DOMAINS)]
    doremi_context = DoReMiContext(domain_keys, is_proxy=is_proxy)

    init_distributed(tp=1, dp=dp_size, pp=1)(_test_yielding)(
        batch_size=BATCH_SIZE,
        global_batch_size=global_batch_size,
        num_microbatches=num_microbatches,
        datasets=datasets,
        doremi_context=doremi_context,
    )


def _test_yielding(
    parallel_context: ParallelContext,
    batch_size: int,
    global_batch_size: int,
    num_microbatches: int,
    datasets,
    doremi_context: DoReMiContext,
):
    # [FUNCTION DOC]
    # HELPER: worker checking sampler domain counters while yielding.
    # Behavior: builds a DistributedSamplerForDoReMi for this rank's dp rank and size; for each micro-batch all-gathers indices over dp and asserts the total size; after each full global batch asserts each domain counter equals the number of global batches times global batch size times 0.5.
    # Inputs: parallel_context; batch_size; global_batch_size; num_microbatches; datasets; doremi_context. Outputs: None.
    # Exceptions: AssertionError on failure. Side effects: all_gather collectives.
    dp_size = dist.get_world_size(parallel_context.dp_pg)
    dp_rank = dist.get_rank(parallel_context.dp_pg)

    sampler = DistributedSamplerForDoReMi(
        datasets,
        batch_size=batch_size,
        num_microbatches=num_microbatches,
        num_replicas=dp_size,
        rank=dp_rank,
        doremi_context=doremi_context,
        parallel_context=parallel_context,
    )

    step = 0
    num_yielded_microbatches = 0
    expected_domain_weights = torch.tensor([0.5, 0.5])

    for idxs in sampler:
        idxs = torch.tensor(idxs, dtype=torch.int, device="cuda")
        idxs_dp = [torch.empty_like(idxs) for _ in range(dp_size)]
        dist.all_gather(idxs_dp, idxs)
        idxs_dp = torch.cat(idxs_dp, dim=0)

        assert idxs_dp.numel() == batch_size * dp_size

        # NOTE: if it loops through all the microbatches
        # then we check if the number of samples in each domain
        if (step + 1) % num_microbatches == 0:
            num_yielded_microbatches += 1
            for i, weight in enumerate(expected_domain_weights):
                assert sampler.domain_counters[i] == int(num_yielded_microbatches * global_batch_size * weight)

        step += 1


@pytest.mark.parametrize("dp_size", [2, 4, 8])
@pytest.mark.parametrize("num_microbatches", [1, 5])
@pytest.mark.parametrize("is_proxy", [True, False])
def test_yielding_with_dataloader(dp_size, num_microbatches, dataset1, is_proxy):
    # [FUNCTION DOC]
    # Behavior: Same setup as test_yielding but through a DataLoader.
    # Inputs: dp_size; num_microbatches; dataset1; is_proxy. Outputs: None. Exceptions: AssertionError in workers. Side effects: spawns processes.
    NUM_DOMAINS = 2
    BATCH_SIZE = 100
    global_batch_size = BATCH_SIZE * num_microbatches * dp_size

    datasets = [dataset1 for _ in range(NUM_DOMAINS)]
    domain_keys = [f"domain {i}" for i in range(NUM_DOMAINS)]
    doremi_context = DoReMiContext(domain_keys, is_proxy=is_proxy)

    init_distributed(tp=1, dp=dp_size, pp=1)(_test_yielding_with_dataloader)(
        batch_size=BATCH_SIZE,
        global_batch_size=global_batch_size,
        num_microbatches=num_microbatches,
        datasets=datasets,
        doremi_context=doremi_context,
    )


def _test_yielding_with_dataloader(
    parallel_context: ParallelContext,
    batch_size: int,
    global_batch_size: int,
    num_microbatches: int,
    datasets,
    doremi_context: DoReMiContext,
):
    # [FUNCTION DOC]
    # HELPER: worker checking DataLoader batches with the DoReMi batch sampler.
    # Behavior: builds a DistributedSamplerForDoReMi for this rank's dp rank and size, wraps CombinedDataset in a DataLoader using it as batch_sampler; for each batch asserts the "text" column has micro batch size entries and that their dp sum equals batch size times dp; after each full global batch checks domain counters as in _test_yielding; finally asserts at least one batch was produced.
    # Inputs: parallel_context; batch_size; global_batch_size; num_microbatches; datasets; doremi_context. Outputs: None.
    # Exceptions: AssertionError on failure. Side effects: all_reduce collectives over dp.
    dp_size = dist.get_world_size(parallel_context.dp_pg)
    dp_rank = dist.get_rank(parallel_context.dp_pg)

    sampler = DistributedSamplerForDoReMi(
        datasets,
        batch_size=batch_size,
        num_microbatches=num_microbatches,
        num_replicas=dp_size,
        rank=dp_rank,
        doremi_context=doremi_context,
        parallel_context=parallel_context,
    )
    comebined_dataset = CombinedDataset(datasets)
    dataloader = DataLoader(comebined_dataset, batch_sampler=sampler)

    step = 1
    num_yielded_microbatches = 0
    expected_domain_weights = torch.tensor([0.5, 0.5])

    for idxs in dataloader:
        num_idxs = torch.tensor(len(idxs["text"]), dtype=torch.int, device="cuda")
        assert num_idxs.item() == batch_size

        dist.all_reduce(num_idxs, op=dist.ReduceOp.SUM, group=parallel_context.dp_pg)
        assert num_idxs == batch_size * dp_size

        if step % num_microbatches == 0:
            num_yielded_microbatches += 1
            for i, weight in enumerate(expected_domain_weights):
                assert sampler.domain_counters[i] == int(num_yielded_microbatches * global_batch_size * weight)

        step += 1

    assert step > 1
