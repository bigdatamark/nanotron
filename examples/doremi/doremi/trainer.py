# [MODULE DOC]
# Purpose: Trainers for the DoReMi example extending nanotron's DistributedTrainer: DoReMiTrainer (proxy model trained with the DoReMi objective against a frozen reference model) and ReferenceTrainer (reference model trained with fixed domain weights and per-domain loss logging).
# Main contents: print_array_for_human, DoReMiTrainer, ReferenceTrainer.
# Import-time side effects: optional wandb import (None if missing); logger creation.
from typing import Dict, Iterable, List, Optional, Type, Union

import torch
from nanotron import distributed as dist
from nanotron import logging
from nanotron.config import Config, get_config_from_file
from nanotron.logging import log_rank
from nanotron.models import NanotronModel
from nanotron.parallel.pipeline_parallel.tensor_pointer import TensorPointer
from nanotron.sanity_checks import assert_tensor_synced_across_pg
from nanotron.serialize import load_weights
from nanotron.trainer import DistributedTrainer
from torch.nn.parallel import DistributedDataParallel

from .config import DoReMiConfig
from .doremi_context import DoReMiContext
from .llama import (
    LlamaForDoReMiTraining,
    LLaMaForInference,
    LlamaReferenceForTrainingWithPerDomainLoss,
)

try:
    import wandb
except ImportError:
    wandb = None

logger = logging.get_logger(__name__)


def print_array_for_human(arr: List[float], precision: int = 5) -> str:
    # [FUNCTION DOC]
    # HELPER: formats float arrays for log messages (duplicate of examples/doremi/utils.py).
    # Behavior: Formats each value with the given decimals and joins them in square brackets.
    # Inputs: arr: sequence of floats; precision: int, default 5. Outputs: string. Exceptions: None explicitly raised. Side effects: None.
    formatted_elements = [f"{x:.{precision}f}" for x in arr]
    return "[" + ", ".join(formatted_elements) + "]"


class DoReMiTrainer(DistributedTrainer):
    # [CLASS DOC]
    # Purpose: DistributedTrainer for DoReMi proxy training. Builds a DoReMiContext (proxy mode) from the doremi config section, constructs the proxy model (LlamaForDoReMiTraining) and a reference model (LLaMaForInference) loaded from a checkpoint, and logs/saves per-step domain weights.
    # Key state: doremi_context, ref_checkpoint_path, ref_model (plus all DistributedTrainer state).
    def __init__(
        self,
        config_or_config_file: Union[Config, str],
        config_class: Type[Config] = Config,
    ):
        # NOTE: save the initial domain_weights
        # [FUNCTION DOC]
        # Behavior: Loads the config (with config_class), asserts the reference checkpoint path is set, builds a proxy DoReMiContext from domain names, step size and smoothing param, stores the reference path, then calls the parent constructor (which loads the config again and builds the model through _init_model_instance).
        # Inputs: config_or_config_file: Config or YAML path; config_class: Config subclass to parse into (default Config; DoReMiConfig is expected).
        # Outputs: None.
        # Exceptions: AssertionError if no reference path; config parsing errors.
        # Side effects: everything DistributedTrainer construction does (process groups, model creation, logging).
        # KNOWN ISSUE: with the default config_class Config, the doremi section is not parsed and attribute access to config.doremi fails; callers must pass DoReMiConfig.
        config: DoReMiConfig = get_config_from_file(config_or_config_file, config_class=config_class)
        assert (
            config.doremi.ref_model_resume_checkpoint_path is not None
        ), "You must provide a reference model checkpoint path for DoReMi training."

        self.doremi_context = DoReMiContext(
            config.doremi.domain_names,
            is_proxy=True,
            step_size=config.doremi.step_size,
            smoothing_param=config.doremi.smoothing_param,
        )
        self.ref_checkpoint_path = config.doremi.ref_model_resume_checkpoint_path
        super().__init__(config_or_config_file, config_class)

    def _init_model_instance(self) -> Union[NanotronModel, DistributedDataParallel]:
        # [FUNCTION DOC]
        # Behavior: Asserts the reference path, moves domain weights to CUDA, asserts they are identical across the world group, logs a note, builds the proxy model via _init_model, builds the reference inference model the same way, and loads reference weights from the checkpoint into its unwrapped module.
        # Inputs: self. Outputs: the proxy model (possibly DDP-wrapped).
        # Exceptions: AssertionError on missing path or unsynced weights; errors from load_weights.
        # Side effects: GPU allocation, collectives for the sanity check, checkpoint reads, sets self.ref_model.
        # KNOWN ISSUE: both branches of the conditional that unwraps the reference model take .module, so the DistributedDataParallel check is meaningless.
        assert (
            self.ref_checkpoint_path is not None
        ), "You must provide a reference model checkpoint path for DoReMi's proxy training."
        # NOTE: after initializing parallel context, now we can move domain weights to
        # the GPU corresponding to the current rank
        self.doremi_context.domain_weights = self.doremi_context.domain_weights.to("cuda")

        # NOTE: SANITY CHECKS: make sure all ranks have the same domain weights
        assert_tensor_synced_across_pg(
            tensor=self.doremi_context.domain_weights,
            pg=self.parallel_context.world_pg,
            msg=lambda err: f"Domain weights are not synced across ranks {err}",
        )

        log_rank(
            f"""[DoReMi] In DoReMi's proxy training, please note that 'loss' represents DRO loss, and 'ce_loss' represent cross entropy loss.
            [DoReMi] Sampling weights: {self.doremi_context.domain_weights}""",
            logger=logger,
            level=logging.INFO,
        )

        model = self._init_model(
            model_builder=lambda: LlamaForDoReMiTraining(
                config=self.model_config,
                parallel_context=self.parallel_context,
                parallel_config=self.config.parallelism,
                doremi_context=self.doremi_context,
            ),
        )

        log_rank("[DoReMi] Initializing reference model for DoReMi training", logger=logger, level=logging.INFO)

        self.ref_model = self._init_model(
            model_builder=lambda: LLaMaForInference(
                config=self.model_config,
                parallel_config=self.config.parallelism,
                parallel_context=self.parallel_context,
            ),
        )

        normalized_ref_model = (
            self.ref_model.module
            if isinstance(self.ref_model.module, DistributedDataParallel)
            else self.ref_model.module
        )

        log_rank(
            f"Loading weights from {self.ref_checkpoint_path} for reference model",
            logger=logger,
            level=logging.INFO,
            rank=0,
        )
        load_weights(
            model=normalized_ref_model,
            parallel_context=self.parallel_context,
            root_folder=self.ref_checkpoint_path,
        )

        return model

    def train_step_logs(
        self,
        outputs: Iterable[Dict[str, Union[torch.Tensor, TensorPointer]]],
        loss_avg: Optional[torch.Tensor],
    ):
        # [FUNCTION DOC]
        # Behavior: Takes domain weights, domain losses and samples per domain from the first micro-batch output and sums ce_loss over micro-batches; launches async all-reduces over dp (average weights, losses and ce_loss; sum sample counts); runs the parent logging; waits; appends the averaged weights to the context history; logs weights, losses and sample counts on dp rank 0; on the first logger rank, saves the weight history to checkpoints_path/doremi_domain_weights_STEP.pt every checkpoint interval and logs per-domain metrics to wandb when available.
        # Inputs: outputs: list of per-micro-batch output dictionaries; loss_avg: optional averaged loss tensor.
        # Outputs: None.
        # Exceptions: errors from collectives or file I/O.
        # Side effects: in-place all-reduce of output tensors; dp collectives; logging; file writes; wandb calls.
        # KNOWN ISSUE: the updated weights are never written back to doremi_context.domain_weights, so sampling stays uniform (acknowledged by the in-code note); the history records in-flight weights.
        # KNOWN ISSUE: ce_loss is summed rather than averaged over micro-batches, contradicting the ce_loss_avg name.
        domain_weights = outputs[0]["domain_weights"]
        domain_losses = outputs[0]["domain_losses"]
        samples_per_domain = outputs[0]["samples_per_domain"]
        # NOTE: this is cross entropy loss
        ce_loss_avg = torch.stack([output["ce_loss"] for output in outputs]).sum()

        handle_weight = dist.all_reduce(
            domain_weights, group=self.parallel_context.dp_pg, async_op=True, op=dist.ReduceOp.AVG
        )
        handle_loss = dist.all_reduce(
            domain_losses, group=self.parallel_context.dp_pg, async_op=True, op=dist.ReduceOp.AVG
        )
        # NOTE: sum the total samples per domain across dp replicas
        handle_samples_per_domain = dist.all_reduce(
            samples_per_domain, group=self.parallel_context.dp_pg, async_op=True, op=dist.ReduceOp.SUM
        )
        handle_ce_loss = dist.all_reduce(
            ce_loss_avg, group=self.parallel_context.dp_pg, async_op=True, op=dist.ReduceOp.AVG
        )

        super().train_step_logs(outputs, loss_avg)

        handle_weight.wait()
        handle_loss.wait()
        handle_samples_per_domain.wait()
        handle_ce_loss.wait()

        self.doremi_context.add_weight_with_history(domain_weights, self.iteration_step)

        domain_weights = domain_weights.cpu().detach().numpy()
        domain_losses = domain_losses.cpu().detach().numpy()

        # NOTE: the domain weights here aren't the sampling weights
        # but in-flight weights of the current step, we use a fixed uniform weights
        # for sampling
        log_rank(
            f"""[DoReMi] Domain weights: {print_array_for_human(domain_weights)}
            [DoReMi] Domain losses: {print_array_for_human(domain_losses)}
            [DoReMi] Samples per domain: {str(samples_per_domain)}
            """,
            logger=logger,
            level=logging.INFO,
            rank=0,
            group=self.parallel_context.dp_pg,
        )

        if dist.get_rank(self.parallel_context.world_pg) == self.logger_ranks[0]:
            if self.iteration_step % self.config.checkpoints.checkpoint_interval == 0:
                checkpoints_path = self.config.checkpoints.checkpoints_path
                checkpoint_path = checkpoints_path / f"doremi_domain_weights_{self.iteration_step}.pt"
                torch.save(self.doremi_context.domain_weight_history, checkpoint_path)

            if wandb is not None:
                weight_logs = {
                    f"weight_domain_{self.doremi_context.get_domain_name(i)}": weight
                    for i, weight in enumerate(domain_weights)
                }
                loss_logs = {
                    f"loss_domain_{self.doremi_context.get_domain_name(i)}": loss
                    for i, loss in enumerate(domain_losses)
                }
                samples_per_domain_logs = {
                    f"samples_per_domain_{self.doremi_context.get_domain_name(i)}": samples
                    for i, samples in enumerate(samples_per_domain)
                }

                wandb.log(
                    {
                        **weight_logs,
                        **loss_logs,
                        **samples_per_domain_logs,
                        "ce_loss": ce_loss_avg.cpu().detach().numpy(),
                        "iteration_step": self.iteration_step,
                    }
                )


class ReferenceTrainer(DistributedTrainer):
    # [CLASS DOC]
    # Purpose: DistributedTrainer for training the DoReMi reference model with per-domain loss logging; uses a non-proxy DoReMiContext.
    # Key state: doremi_context, valid_dataloader (None).
    def __init__(self, domain_weights: torch.Tensor, domain_keys: List[str], *args, **kwargs):
        # [FUNCTION DOC]
        # Behavior: Builds a DoReMiContext, sets valid_dataloader to None, calls the parent constructor with remaining arguments, moves domain weights to CUDA, asserts they are synced across the world group and logs them.
        # Inputs: domain_weights: tensor of initial weights; domain_keys: list of domain names; positional/keyword arguments forwarded to DistributedTrainer (config file, config class).
        # Outputs: None.
        # Exceptions: AssertionError on unsynced weights; TypeError (see below).
        # Side effects: DistributedTrainer construction; collectives; logging.
        # KNOWN ISSUE: DoReMiContext is called with domain_weights and domain_keys positionally plus is_proxy as a keyword; its signature is (domain_keys, is_proxy, ...), so this raises TypeError for multiple values of is_proxy, and the provided weights would be ignored anyway since the context always starts uniform.
        self.doremi_context = DoReMiContext(domain_weights, domain_keys, is_proxy=False)
        self.valid_dataloader = None
        super().__init__(*args, **kwargs)
        self.doremi_context.domain_weights = self.doremi_context.domain_weights.to("cuda")

        # NOTE: SANITY CHECKS: make sure all ranks have the same domain weights
        assert_tensor_synced_across_pg(
            tensor=self.doremi_context.domain_weights,
            pg=self.parallel_context.world_pg,
            msg=lambda err: f"Domain weights are not synced across ranks {err}",
        )

        log_rank(
            f"[DoReMi] Initial domain weights: {self.doremi_context.domain_weights}", logger=logger, level=logging.INFO
        )

    def _init_model_instance(self) -> Union[NanotronModel, DistributedDataParallel]:
        # [FUNCTION DOC]
        # Behavior: Builds LlamaReferenceForTrainingWithPerDomainLoss through _init_model.
        # Inputs: self. Outputs: model (possibly DDP-wrapped). Exceptions: from model construction. Side effects: GPU allocation.
        model = self._init_model(
            model_builder=lambda: LlamaReferenceForTrainingWithPerDomainLoss(
                config=self.model_config,
                doremi_context=self.doremi_context,
                parallel_context=self.parallel_context,
                parallel_config=self.config.parallelism,
            ),
        )
        return model

    def train_step_logs(
        self,
        outputs: Iterable[Dict[str, Union[torch.Tensor, TensorPointer]]],
        loss_avg: Optional[torch.Tensor],
    ):
        # [FUNCTION DOC]
        # Behavior: Runs parent logging, then logs the first micro-batch's domain losses and samples per domain on rank 0, and on the first logger rank logs them plus loss_avg to wandb if installed.
        # Inputs: outputs: list of output dictionaries; loss_avg: optional tensor.
        # Outputs: None.
        # Exceptions: AttributeError if loss_avg is None on the wandb branch.
        # Side effects: logging; wandb calls; device-to-host copies.
        super().train_step_logs(outputs, loss_avg)

        domain_losses = outputs[0]["domain_losses"].tolist()
        samples_per_domain = outputs[0]["samples_per_domain"].tolist()

        log_rank(
            f"[DoReMi][Train] Domain loss: {print_array_for_human(domain_losses)}",
            logger=logger,
            level=logging.INFO,
            rank=0,
        )

        log_rank(
            f"[DoReMi][Train] Samples per domain: {str(samples_per_domain)}",
            logger=logger,
            level=logging.INFO,
            rank=0,
        )

        if dist.get_rank(self.parallel_context.world_pg) == self.logger_ranks[0] and wandb is not None:
            loss_logs = {
                f"loss_domain_{self.doremi_context.get_domain_name(i)}": loss for i, loss in enumerate(domain_losses)
            }

            samples_per_domain_logs = {
                f"samples_per_domain_{self.doremi_context.get_domain_name(i)}": n_samples
                for i, n_samples in enumerate(samples_per_domain)
            }

            wandb.log(
                {
                    **loss_logs,
                    **samples_per_domain_logs,
                    "loss_avg": loss_avg.item(),
                    "iteration_step": self.iteration_step,
                }
            )
