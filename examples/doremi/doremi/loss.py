# [MODULE DOC]
# Purpose: Loss functions for the DoReMi example: per-domain loss aggregation, the DoReMi excess-loss / group-DRO objective with exponentiated-gradient domain-weight update, and cross-entropy with per-domain logging for reference training.
# Main contents: compute_per_domain_loss (gathers across data parallel), compute_domain_loss_per_replicas (local only), DomainLossForProxyTraining, CrossEntropyWithPerDomainLoss, DoReMiLossForProxyTraining.
# How it fits: the nn.Module losses are wrapped in PipelineBlocks by doremi/llama.py and run on the last pipeline stage.
# Import-time side effects: None (imports TorchScript masked_mean).
from typing import Dict, Tuple

import torch
import torch.distributed as dist
from nanotron.parallel import ParallelContext
from nanotron.parallel.tensor_parallel.functional import sharded_cross_entropy
from torch import nn

from .doremi_context import DoReMiContext
from .utils import masked_mean


def compute_per_domain_loss(
    losses: torch.Tensor, domain_idxs: torch.Tensor, doremi_context: DoReMiContext, parallel_context: ParallelContext
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # [FUNCTION DOC]
    # Behavior: All-gathers per-sample losses and domain ids over the data-parallel group (into contiguous CUDA buffers) and concatenates them along batch; sums each sample's token losses into its domain's bucket (Python loop over the global batch); counts samples per domain with bincount; divides each domain's total by samples times sequence length; replaces NaN (domains with no samples) with 0.
    # Inputs: losses: CUDA tensor [batch, seq] per-token losses; domain_idxs: integer tensor [batch] or [batch,1]; doremi_context (for number of domains); parallel_context (dp group).
    # Outputs: tuple of gathered losses [dp*batch, seq], normalized per-domain losses [num_domains], samples per domain [num_domains] (int64).
    # Exceptions: AssertionError if gathered losses and domain ids differ in length.
    # Side effects: two all_gather collectives on the dp group; CUDA allocations.
    # KNOWN ISSUE: label_mask is not applied, so padded/ignored tokens are counted in domain losses and in the normalization by sequence length.
    dp_size = dist.get_world_size(parallel_context.dp_pg)
    dp_pg = parallel_context.dp_pg

    # NOTE: can't do allgather([tensor_list], [tensor]) if a tensor in tensor_list is not contiguous
    losses_dp = [
        torch.empty_like(losses, device="cuda", memory_format=torch.contiguous_format) for _ in range(dp_size)
    ]
    dist.all_gather(losses_dp, losses.contiguous(), group=dp_pg)
    losses_dp = torch.cat(losses_dp, dim=0)

    domain_ids_dp = [
        torch.empty_like(domain_idxs, device="cuda", memory_format=torch.contiguous_format) for _ in range(dp_size)
    ]
    dist.all_gather(domain_ids_dp, domain_idxs.contiguous(), group=dp_pg)
    domain_ids_dp = torch.cat(domain_ids_dp, dim=0)

    # NOTE: Calculate total loss per domain
    n_domains = doremi_context.num_domains
    domain_losses = torch.zeros(n_domains, device="cuda")
    domain_ids_dp = domain_ids_dp.view(-1)

    assert losses_dp.shape[0] == domain_ids_dp.shape[0]
    GLOBAL_BATCH_SIZE = losses_dp.shape[0]

    for i in range(GLOBAL_BATCH_SIZE):
        # NOTE: sum the excess losses of all tokens in the batch
        # then add it to the domain loss of the corresponding domain
        domain_losses[domain_ids_dp[i]] += losses_dp[i].sum(dim=-1)

    # NOTE: Normalize and smooth domain weights
    samples_per_domain = torch.bincount(domain_ids_dp, minlength=n_domains)
    SEQ_LEN = losses.shape[1]
    normalized_domain_losses = domain_losses / (samples_per_domain * SEQ_LEN)
    # NOTE: if the domain loss is zero, then the normalized domain loss is NaN
    normalized_domain_losses[torch.isnan(normalized_domain_losses)] = 0.0
    return losses_dp, normalized_domain_losses, samples_per_domain


def compute_domain_loss_per_replicas(
    losses: torch.Tensor, domain_idxs: torch.Tensor, doremi_context: DoReMiContext
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # [FUNCTION DOC]
    # Behavior: Same per-domain aggregation as compute_per_domain_loss but only on the local replica's batch (no communication): sums each sample's token losses into its domain, divides by samples-per-domain times sequence length, zeros NaNs.
    # Inputs: losses: CUDA tensor [batch, seq]; domain_idxs: integer tensor [batch] or [batch,1]; doremi_context.
    # Outputs: tuple of normalized per-domain losses [num_domains] and samples per domain [num_domains].
    # Exceptions: AssertionError on batch size mismatch.
    # Side effects: CUDA allocations.
    # KNOWN ISSUE: the return annotation declares three tensors but two are returned.
    domain_idxs = domain_idxs.view(-1)

    # NOTE: Calculate total loss per domain
    n_domains = doremi_context.num_domains
    domain_losses = torch.zeros(n_domains, device="cuda")

    assert losses.shape[0] == domain_idxs.shape[0]
    GLOBAL_BATCH_SIZE = domain_idxs.shape[0]

    for i in range(GLOBAL_BATCH_SIZE):
        # NOTE: sum the excess losses of all tokens in the batch
        # then add it to the domain loss of the corresponding domain
        domain_losses[domain_idxs[i]] += losses[i].sum(dim=-1)

    # NOTE: Normalize domain weights
    SEQ_LEN = losses.shape[1]
    samples_per_domain = torch.bincount(domain_idxs, minlength=n_domains)
    normalized_domain_losses = domain_losses / (samples_per_domain * SEQ_LEN)

    # NOTE: if the domain loss is zero, then the normalized domain loss is NaN
    normalized_domain_losses[torch.isnan(normalized_domain_losses)] = 0.0
    return normalized_domain_losses, samples_per_domain


class DomainLossForProxyTraining:
    # [CLASS DOC]
    # Purpose: Callable computing the DoReMi proxy objective from proxy and reference per-token losses.
    # Attributes: doremi_context (provides current weights, step_size, smoothing_param), parallel_context.
    def __init__(self, doremi_context: DoReMiContext, parallel_context: ParallelContext):
        # [FUNCTION DOC]
        # Behavior: Stores the contexts.
        # Inputs: doremi_context, parallel_context. Outputs: None. Exceptions: None. Side effects: None.
        self.doremi_context = doremi_context
        self.parallel_context = parallel_context

    def __call__(self, losses: torch.Tensor, ref_losses: torch.Tensor, domain_idxs: torch.Tensor):
        # [FUNCTION DOC]
        # Behavior: Asserts shapes match; computes excess losses as proxy minus reference clamped at 0; aggregates them per domain on the local replica; updates weights in log space (log of current weights plus step_size times normalized domain excess loss, normalized with logsumexp), mixes with uniform using smoothing_param; DRO loss is the dot product of these weights with the domain excess losses.
        # Inputs: losses and ref_losses: tensors [batch, seq]; domain_idxs: tensor with batch as first dimension.
        # Outputs: dictionary with dro_loss (scalar), domain_losses [num_domains], domain_weights [num_domains] (the updated train weights), samples_per_domain [num_domains].
        # Exceptions: AssertionError on shape mismatch.
        # Side effects: None (does not write the new weights back to doremi_context).
        # KNOWN ISSUE: weights are computed per dp replica without synchronization (only averaged later for logging), and the gradient flows through the weights as well as the losses since they are not detached.
        assert losses.shape == ref_losses.shape, "losses and ref_losses must have the same shape"
        assert (
            domain_idxs.shape[0] == losses.shape[0]
        ), "the batch size of domain_idxs must match the batch size of losses"

        # NOTE: sometimes you'll see the domain losses equal to zero.
        # this doesn't mean there are bugs, it just means that in that case,
        # the proxy model is performing better than the reference model
        # => clamp(lower loss - higher loss, 0) = clamp(negative, 0) = 0.
        excess_losses = (losses - ref_losses).clamp(min=0)
        normalized_domain_losses, samples_per_domain = compute_domain_loss_per_replicas(
            excess_losses, domain_idxs, self.doremi_context
        )

        domain_weights = self.doremi_context.domain_weights
        step_size = self.doremi_context.step_size
        smoothing_param = self.doremi_context.smoothing_param
        log_new_domain_weights = torch.log(domain_weights) + step_size * normalized_domain_losses
        log_new_domain_weights = log_new_domain_weights - torch.logsumexp(log_new_domain_weights, dim=0)
        train_domain_weights = (1 - smoothing_param) * torch.exp(log_new_domain_weights) + smoothing_param / len(
            log_new_domain_weights
        )
        dro_loss = (train_domain_weights * normalized_domain_losses).sum(dim=-1)

        return {
            "dro_loss": dro_loss,
            "domain_losses": normalized_domain_losses,
            "domain_weights": train_domain_weights,
            "samples_per_domain": samples_per_domain,
        }


class CrossEntropyWithPerDomainLoss(nn.Module):
    # [CLASS DOC]
    # Purpose: Loss module for reference-model training: standard masked cross-entropy plus per-domain loss statistics gathered across dp.
    # Attributes: doremi_context, parallel_context.
    def __init__(self, doremi_context: DoReMiContext, parallel_context: ParallelContext):
        # [FUNCTION DOC]
        # Behavior: Initializes nn.Module and stores contexts.
        # Inputs: doremi_context, parallel_context. Outputs: None. Exceptions: None. Side effects: None.
        super().__init__()
        self.doremi_context = doremi_context
        self.parallel_context = parallel_context

    def forward(
        self,
        sharded_logits: torch.Tensor,  # [seq_length, batch_size, logits]
        label_ids: torch.Tensor,  # [batch_size, seq_length]
        label_mask: torch.Tensor,  # [batch_size, seq_length]
        domain_idxs: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        # [FUNCTION DOC]
        # Behavior: Computes TP-sharded per-token cross-entropy in float32, its masked mean as ce_loss, and per-domain losses/sample counts via compute_per_domain_loss.
        # Inputs: sharded_logits (batch-first as passed by doremi/llama.py, vocab sharded over TP); label_ids [batch, seq]; label_mask [batch, seq]; domain_idxs.
        # Outputs: dictionary with ce_loss, domain_losses, samples_per_domain.
        # Exceptions: None explicitly raised.
        # Side effects: TP collectives in sharded_cross_entropy; dp all_gathers.
        # KNOWN ISSUE: returns the key "ce_loss" but the PipelineBlock in LlamaReferenceForTrainingWithPerDomainLoss expects and reads "loss", which would raise KeyError.
        # KNOWN ISSUE: inline comment says sharded_logits is [seq, batch, vocab] but callers pass batch-first logits.
        per_token_loss = sharded_cross_entropy(
            sharded_logits, label_ids, group=self.parallel_context.tp_pg, dtype=torch.float
        )
        ce_loss = masked_mean(per_token_loss, label_mask, dtype=torch.float)
        _, domain_losses, samples_per_domain = compute_per_domain_loss(
            per_token_loss, domain_idxs, self.doremi_context, self.parallel_context
        )
        return {"ce_loss": ce_loss, "domain_losses": domain_losses, "samples_per_domain": samples_per_domain}


class DoReMiLossForProxyTraining(nn.Module):
    # [CLASS DOC]
    # Purpose: Loss module for DoReMi proxy training wrapping DomainLossForProxyTraining; the optimized loss is the DRO loss.
    # Attributes: parallel_context, doremi_loss (DomainLossForProxyTraining).
    def __init__(self, doremi_context: DoReMiContext, parallel_context: ParallelContext):
        # [FUNCTION DOC]
        # Behavior: Initializes nn.Module, stores parallel_context and builds the DoReMi loss callable.
        # Inputs: doremi_context, parallel_context. Outputs: None. Exceptions: None. Side effects: None.
        super().__init__()
        self.parallel_context = parallel_context
        self.doremi_loss = DomainLossForProxyTraining(doremi_context, parallel_context)

    def forward(
        self,
        sharded_logits: torch.Tensor,  # [seq_length, batch_size, logits]
        label_ids: torch.Tensor,  # [batch_size, seq_length]
        label_mask: torch.Tensor,  # [batch_size, seq_length]
        domain_idxs: torch.Tensor,
        ref_losses: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        # [FUNCTION DOC]
        # Behavior: Computes TP-sharded per-token cross-entropy, the masked mean ce_loss, and the DoReMi outputs from the per-token loss and reference losses.
        # Inputs: sharded_logits (batch-first, TP-sharded vocab); label_ids, label_mask [batch, seq]; domain_idxs; ref_losses [batch, seq].
        # Outputs: dictionary with ce_loss, loss (DRO loss, the optimized one), domain_losses, domain_weights, samples_per_domain.
        # Exceptions: AssertionError from shape checks.
        # Side effects: TP collectives.
        # KNOWN ISSUE: inline comment says sharded_logits is [seq, batch, vocab] but callers pass batch-first logits.
        loss = sharded_cross_entropy(
            sharded_logits,
            label_ids,
            group=self.parallel_context.tp_pg,
            dtype=torch.float,
        )
        ce_loss = masked_mean(loss, label_mask, dtype=torch.float)
        doremi_loss_outputs = self.doremi_loss(loss, ref_losses, domain_idxs)

        return {
            "ce_loss": ce_loss,
            "loss": doremi_loss_outputs["dro_loss"],  # NOTE: this is the one we optimize
            "domain_losses": doremi_loss_outputs["domain_losses"],
            "domain_weights": doremi_loss_outputs["domain_weights"],
            "samples_per_domain": doremi_loss_outputs["samples_per_domain"],
        }
