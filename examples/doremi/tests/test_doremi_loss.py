# [MODULE DOC]
# Purpose: Distributed unit tests for the DoReMi loss module (sharded per-token cross-entropy, per-domain aggregation with and without dp gathering, DRO proxy loss, cross-entropy with per-domain loss).
# Structure: each public test builds inputs and launches a private worker through tests.helpers.utils.init_distributed with the requested tp/dp, which spawns processes with a ParallelContext.
# Import-time side effects: appends the repo root to sys.path; requires CUDA at run time.
import pytest
import torch
import torch.distributed as dist
import torch.nn.functional as F
from nanotron.parallel import ParallelContext
from nanotron.parallel.tensor_parallel.functional import sharded_cross_entropy
from nanotron.sanity_checks import assert_tensor_synced_across_pg
from utils import set_system_path

set_system_path()

from examples.doremi.doremi.doremi_context import DoReMiContext
from examples.doremi.doremi.loss import (
    CrossEntropyWithPerDomainLoss,
    DomainLossForProxyTraining,
    DoReMiLossForProxyTraining,
    compute_domain_loss_per_replicas,
    compute_per_domain_loss,
)
from tests.helpers.utils import init_distributed


@pytest.fixture
def doremi_context():
    # [FUNCTION DOC]
    # HELPER: pytest fixture.
    # Behavior: Returns a non-proxy DoReMiContext with five domains named "domain 0" to "domain 4".
    # Inputs: None. Outputs: DoReMiContext. Exceptions: None. Side effects: None.
    N_DOMAINS = 5
    domain_keys = [f"domain {i}" for i in range(N_DOMAINS)]
    doremi_context = DoReMiContext(domain_keys, is_proxy=False)
    return doremi_context


def get_partition_logit(logits, parallel_context):
    # [FUNCTION DOC]
    # HELPER: emulates vocab sharding for tensor-parallel tests.
    # Behavior: Splits logits along the last dimension into tp_size equal chunks and returns the chunk of this tp rank.
    # Inputs: logits: tensor [..., vocab]; parallel_context. Outputs: tensor [..., vocab/tp].
    # Exceptions: IndexError if vocab is not divisible so chunk count differs. Side effects: None.
    tp_size = dist.get_world_size(parallel_context.tp_pg)
    tp_rank = dist.get_rank(parallel_context.tp_pg)
    VOCAB_SIZE = logits.shape[-1]
    per_partition = VOCAB_SIZE // tp_size
    chunks = torch.split(logits, per_partition, dim=-1)
    return chunks[tp_rank]


@pytest.mark.parametrize("tp", [1, 2])
def test_computing_per_token_loss(tp: int):
    # [FUNCTION DOC]
    # Behavior: With a fixed seed builds random logits [512,128,4] and targets, computes reference unreduced cross-entropy with torch, and runs the worker under the given tp.
    # Inputs: tp: 1 or 2. Outputs: None. Exceptions: AssertionError in workers. Side effects: spawns distributed processes; sets torch RNG seed.
    BATCH_SIZE = 512
    SEQ_LEN = 128
    VOCAB_SIZE = 4

    torch.manual_seed(69)

    logits = torch.randn(BATCH_SIZE, SEQ_LEN, VOCAB_SIZE)
    targets = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN))

    ref_losses = F.cross_entropy(logits.view(-1, logits.size(2)), targets.view(-1), reduction="none")

    init_distributed(tp=tp, dp=1, pp=1)(_test_computing_per_token_loss)(
        logits=logits, targets=targets, ref_losses=ref_losses
    )


def _test_computing_per_token_loss(parallel_context: ParallelContext, logits, targets, ref_losses):
    # [FUNCTION DOC]
    # HELPER: worker for test_computing_per_token_loss.
    # Behavior: Moves data to CUDA, takes this rank's vocab shard, computes sharded_cross_entropy over the tp group and asserts it matches the reference per-token losses.
    # Inputs: parallel_context; logits; targets; ref_losses. Outputs: None.
    # Exceptions: AssertionError on mismatch. Side effects: CUDA and tp collectives.
    logits = logits.to("cuda")
    targets = targets.to("cuda")
    parallel_logits = get_partition_logit(logits, parallel_context)

    loss = sharded_cross_entropy(parallel_logits, targets, parallel_context.tp_pg)

    assert torch.allclose(loss.cpu().view(-1), ref_losses)


@pytest.mark.parametrize("dp", [1, 2])
def test_domain_loss_for_proxy_training(dp: int):
    # [FUNCTION DOC]
    # Behavior: Launches the worker with dp ranks, global batch 512 split evenly, sequence 128, five domains.
    # Inputs: dp: 1 or 2. Outputs: None. Exceptions: AssertionError in workers. Side effects: spawns processes.
    GLOBAL_BATCH_SIZE = 512
    BATCH_SIZE = GLOBAL_BATCH_SIZE // dp
    SEQ_LEN = 128
    N_DOMAINS = 5
    domain_keys = [f"domain {i}" for i in range(N_DOMAINS)]

    init_distributed(tp=1, dp=dp, pp=1)(_test_domain_loss_for_proxy_training)(
        global_batch_size=GLOBAL_BATCH_SIZE,
        batch_size=BATCH_SIZE,
        seq_len=SEQ_LEN,
        domain_keys=domain_keys,
    )


def _test_domain_loss_for_proxy_training(
    parallel_context: ParallelContext, global_batch_size, batch_size, seq_len, domain_keys
):
    # [FUNCTION DOC]
    # HELPER: worker for test_domain_loss_for_proxy_training.
    # Behavior: Creates random proxy/reference losses and domain ids on CUDA, runs DomainLossForProxyTraining and asserts output keys, positive domain losses and weights, and shapes of num_domains.
    # Inputs: parallel_context; global_batch_size (unused); batch_size; seq_len; domain_keys. Outputs: None.
    # Exceptions: AssertionError on failure. Side effects: CUDA allocation; unseeded RNG use.
    N_DOMAINS = len(domain_keys)
    losses = torch.randn(batch_size, seq_len, device="cuda")
    ref_losses = torch.randn(batch_size, seq_len, device="cuda")
    domain_idxs = torch.randint(0, N_DOMAINS, (batch_size,), device="cuda")

    doremi_context = DoReMiContext(domain_keys, is_proxy=False)
    doremi_context.domain_weights = doremi_context.domain_weights.to("cuda")
    loss_func = DomainLossForProxyTraining(doremi_context, parallel_context)

    outputs = loss_func(losses, ref_losses, domain_idxs)

    assert outputs.keys() == {"dro_loss", "domain_losses", "domain_weights", "samples_per_domain"}

    assert (outputs["domain_losses"] > 0.0).all()
    assert outputs["domain_losses"].shape == (N_DOMAINS,)

    assert (outputs["domain_weights"] > 0.0).all()
    assert outputs["domain_weights"].shape == (N_DOMAINS,)


@pytest.mark.parametrize("dp", [1, 2])
def test_computing_per_domain_loss(dp: int):
    # [FUNCTION DOC]
    # Behavior: Launches the worker with the given dp size, global batch 512 split across ranks.
    # Inputs: dp: 1 or 2. Outputs: None. Exceptions: AssertionError in workers. Side effects: spawns processes.
    GLOBAL_BATCH_SIZE = 512
    BATCH_SIZE = GLOBAL_BATCH_SIZE // dp
    SEQ_LEN = 128
    N_DOMAINS = 5

    domain_keys = [f"domain {i}" for i in range(N_DOMAINS)]

    init_distributed(tp=1, dp=dp, pp=1)(_test_computing_per_domain_loss)(
        batch_size=BATCH_SIZE,
        global_batch_size=GLOBAL_BATCH_SIZE,
        seq_len=SEQ_LEN,
        domain_keys=domain_keys,
    )


def _test_computing_per_domain_loss(
    parallel_context: ParallelContext, batch_size, global_batch_size, seq_len, domain_keys
):
    # [FUNCTION DOC]
    # HELPER: worker for test_computing_per_domain_loss.
    # Behavior: Creates random local losses and domain ids, runs compute_per_domain_loss, and asserts per-domain loss and sample counts have num_domains entries, are synced across dp, and sample counts total the global batch size.
    # Inputs: parallel_context; batch_size; global_batch_size; seq_len; domain_keys. Outputs: None.
    # Exceptions: AssertionError on failure. Side effects: dp all_gathers and sanity-check collectives.
    # KNOWN ISSUE: calls .to on doremi_context.domain_weights without assigning the result, so the weights stay on CPU (no-op).
    N_DOMAINS = len(domain_keys)
    losses = torch.randn(batch_size, seq_len, device="cuda")
    domain_idxs = torch.randint(0, N_DOMAINS, (batch_size,), device="cuda")

    doremi_context = DoReMiContext(domain_keys, is_proxy=False)
    doremi_context.domain_weights.to("cuda")

    losses_dp, per_domain_loss, samples_per_domain = compute_per_domain_loss(
        losses, domain_idxs, doremi_context, parallel_context
    )

    assert per_domain_loss.shape == (N_DOMAINS,)
    assert_tensor_synced_across_pg(
        per_domain_loss, parallel_context.dp_pg, msg=lambda err: f"Per domain loss are not synced across ranks {err}"
    )

    assert samples_per_domain.shape == (N_DOMAINS,)
    assert sum(samples_per_domain) == global_batch_size
    assert_tensor_synced_across_pg(
        samples_per_domain,
        parallel_context.dp_pg,
        msg=lambda err: f"Samples per domain are not synced across ranks {err}",
    )


@pytest.mark.parametrize("dp", [1, 2])
def test_computing_domain_loss_per_replicas(dp: int):
    # [FUNCTION DOC]
    # Behavior: Launches the worker with the given dp size.
    # Inputs: dp: 1 or 2. Outputs: None. Exceptions: AssertionError in workers. Side effects: spawns processes.
    GLOBAL_BATCH_SIZE = 512
    BATCH_SIZE = GLOBAL_BATCH_SIZE // dp
    SEQ_LEN = 128
    N_DOMAINS = 5

    domain_keys = [f"domain {i}" for i in range(N_DOMAINS)]
    init_distributed(tp=1, dp=dp, pp=1)(_test_computing_domain_loss_per_replicas)(
        batch_size=BATCH_SIZE,
        global_batch_size=GLOBAL_BATCH_SIZE,
        seq_len=SEQ_LEN,
        domain_keys=domain_keys,
    )


def _test_computing_domain_loss_per_replicas(
    parallel_context: ParallelContext, batch_size, global_batch_size, seq_len, domain_keys
):
    # [FUNCTION DOC]
    # HELPER: worker for test_computing_domain_loss_per_replicas.
    # Behavior: Creates random local losses and domain ids, runs compute_domain_loss_per_replicas and asserts both outputs have num_domains entries.
    # Inputs: parallel_context; batch_size; global_batch_size (unused); seq_len; domain_keys. Outputs: None.
    # Exceptions: AssertionError on failure. Side effects: CUDA allocation.
    # KNOWN ISSUE: calls .to on doremi_context.domain_weights without assigning the result, so the weights stay on CPU (no-op).
    N_DOMAINS = len(domain_keys)
    losses = torch.randn(batch_size, seq_len, device="cuda")
    domain_idxs = torch.randint(0, N_DOMAINS, (batch_size,), device="cuda")

    doremi_context = DoReMiContext(domain_keys, is_proxy=False)
    doremi_context.domain_weights.to("cuda")

    per_domain_loss, samples_per_domain = compute_domain_loss_per_replicas(losses, domain_idxs, doremi_context)

    assert per_domain_loss.shape == (N_DOMAINS,)
    assert samples_per_domain.shape == (N_DOMAINS,)


@pytest.mark.skip
@pytest.mark.parametrize("tp", [1, 2])
def test_cross_entropy_with_per_domain_loss(tp: int, doremi_context):
    # [FUNCTION DOC]
    # Behavior: (Skipped) Builds seeded random logits, labels, all-true mask and domain ids, a mean-reduced reference cross-entropy, and launches the worker with the given tp.
    # Inputs: tp: 1 or 2; doremi_context fixture. Outputs: None. Exceptions: AssertionError in workers. Side effects: spawns processes.
    BATCH_SIZE = 512
    SEQ_LEN = 128
    VOCAB_SIZE = 4
    N_DOMAINS = doremi_context.num_domains

    torch.manual_seed(69)

    logits = torch.randn(BATCH_SIZE, SEQ_LEN, VOCAB_SIZE)
    label_ids = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN))
    label_mask = torch.ones((BATCH_SIZE, SEQ_LEN), dtype=torch.bool)
    domain_idxs = torch.randint(0, N_DOMAINS, (BATCH_SIZE,))

    ref_losses = F.cross_entropy(logits.view(-1, logits.size(2)), label_ids.view(-1))

    init_distributed(tp=tp, dp=1, pp=1)(_test_cross_entropy_with_per_domain_loss)(
        logits=logits,
        label_ids=label_ids,
        label_mask=label_mask,
        domain_idxs=domain_idxs,
        ref_losses=ref_losses,
        batch_size=BATCH_SIZE,
        doremi_context=doremi_context,
    )


def _test_cross_entropy_with_per_domain_loss(
    parallel_context: ParallelContext,
    logits,
    label_ids,
    label_mask,
    domain_idxs,
    ref_losses,
    batch_size,
    doremi_context,
):
    # [FUNCTION DOC]
    # HELPER: worker for test_cross_entropy_with_per_domain_loss.
    # Behavior: Moves inputs to CUDA, takes the vocab shard, runs CrossEntropyWithPerDomainLoss and checks loss against the reference, shapes of per-domain outputs and that sample counts sum to the batch size.
    # Inputs: parallel_context; logits; label_ids; label_mask; domain_idxs; ref_losses; batch_size; doremi_context. Outputs: None.
    # Exceptions: AssertionError or KeyError. Side effects: CUDA, tp/dp collectives.
    # KNOWN ISSUE: reads outputs["loss"] although the module returns "ce_loss" (likely why the test is skipped).
    logits = logits.to("cuda")
    label_ids = label_ids.to("cuda")
    label_mask = label_mask.to("cuda")
    domain_idxs = domain_idxs.to("cuda")

    parallel_logits = get_partition_logit(logits, parallel_context)

    loss_func = CrossEntropyWithPerDomainLoss(doremi_context, parallel_context)
    outputs = loss_func(parallel_logits, label_ids, label_mask, domain_idxs)

    assert torch.allclose(outputs["loss"].cpu().view(-1), ref_losses)
    assert outputs["domain_losses"].shape == (doremi_context.num_domains,)
    assert outputs["samples_per_domain"].shape == (doremi_context.num_domains,)
    assert sum(outputs["samples_per_domain"]) == batch_size


@pytest.mark.parametrize("tp", [1, 2])
def test_doremi_loss_for_proxy_training(tp: int, doremi_context):
    # [FUNCTION DOC]
    # Behavior: Builds seeded random logits [512,128,4], labels, all-true mask, domain ids, random reference losses and a mean reference cross-entropy, then launches the worker with the given tp.
    # Inputs: tp: 1 or 2; doremi_context fixture. Outputs: None. Exceptions: AssertionError in workers. Side effects: spawns processes; seeds torch RNG.
    BATCH_SIZE = 512
    SEQ_LEN = 128
    VOCAB_SIZE = 4
    N_DOMAINS = doremi_context.num_domains

    torch.manual_seed(69)

    logits = torch.randn(BATCH_SIZE, SEQ_LEN, VOCAB_SIZE)
    label_ids = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN))
    label_mask = torch.ones((BATCH_SIZE, SEQ_LEN), dtype=torch.bool)
    domain_idxs = torch.randint(0, N_DOMAINS, (BATCH_SIZE,))

    ref_losses = torch.randn(BATCH_SIZE, SEQ_LEN)
    ref_ce_loss = F.cross_entropy(logits.view(-1, logits.size(2)), label_ids.view(-1))

    init_distributed(tp=tp, dp=1, pp=1)(_test_doremi_loss_for_proxy_training)(
        logits=logits,
        label_ids=label_ids,
        label_mask=label_mask,
        domain_idxs=domain_idxs,
        ref_losses=ref_losses,
        ref_ce_loss=ref_ce_loss,
        batch_size=BATCH_SIZE,
        n_domains=N_DOMAINS,
        doremi_context=doremi_context,
    )


def _test_doremi_loss_for_proxy_training(
    parallel_context: ParallelContext,
    logits,
    label_ids,
    label_mask,
    domain_idxs,
    ref_losses,
    ref_ce_loss,
    batch_size,
    n_domains,
    doremi_context,
):
    # [FUNCTION DOC]
    # HELPER: worker for test_doremi_loss_for_proxy_training.
    # Behavior: Moves inputs and domain weights to CUDA, shards logits, runs DoReMiLossForProxyTraining, and asserts a positive scalar loss, ce_loss equal to the reference, positive per-domain losses, weights summing to 1 and sample counts summing to the batch size.
    # Inputs: parallel_context; logits; label_ids; label_mask; domain_idxs; ref_losses; ref_ce_loss; batch_size; n_domains; doremi_context. Outputs: None.
    # Exceptions: AssertionError on failure. Side effects: mutates the fixture context's domain_weights device; tp collectives.
    logits = logits.to("cuda")
    label_ids = label_ids.to("cuda")
    label_mask = label_mask.to("cuda")
    domain_idxs = domain_idxs.to("cuda")
    ref_losses = ref_losses.to("cuda")
    doremi_context.domain_weights = doremi_context.domain_weights.to("cuda")

    parallel_logits = get_partition_logit(logits, parallel_context)

    loss_func = DoReMiLossForProxyTraining(doremi_context, parallel_context)
    outputs = loss_func(parallel_logits, label_ids, label_mask, domain_idxs, ref_losses)

    assert outputs["loss"].ndim == 0
    assert outputs["loss"] > 0.0

    assert torch.allclose(outputs["ce_loss"].cpu().view(-1), ref_ce_loss)

    assert outputs["domain_losses"].shape == (doremi_context.num_domains,)
    assert (outputs["domain_losses"] > 0).all()

    assert outputs["domain_weights"].shape == (doremi_context.num_domains,)
    assert torch.allclose(sum(outputs["domain_weights"].cpu()), torch.tensor(1.0))

    samples_per_domain = outputs["samples_per_domain"]
    assert samples_per_domain.shape == (n_domains,)
    assert sum(samples_per_domain) == batch_size
