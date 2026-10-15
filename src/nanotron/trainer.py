# [MODULE DOC]
# Purpose: The main training orchestrator of nanotron.
# Main contents: get_size, DistributedTrainer (config parsing, process groups, model/optimizer/scheduler
# construction and checkpoint resume, the training loop, logging, checkpoint saving with optional S3
# upload and LightEval), and the module-level tying helpers mark_tied_parameters,
# mark_unsharded_params_as_tied_across_tp and mark_unsharded_params_as_tied_across_expert.
# Fits in nanotron: run_train.py builds a DistributedTrainer and calls train with dataloaders; it ties
# together parallel context, models, pipeline engine, optimizers, serialization and sanity checks.
# Import-time side effects: creates loggers and lowers torch.distributed logger verbosity to WARNING;
# defines the mutable CONFIG_TO_MODEL_CLASS registry (Llama, Starcoder2, Qwen2); imports wandb if
# available (else wandb is None).
import datetime
import gc
import json
import os
import shutil
import tempfile
from dataclasses import asdict
from pathlib import Path
from pprint import pformat
from typing import (
    Callable,
    Dict,
    Iterable,
    Iterator,
    List,
    Optional,
    Tuple,
    Type,
    Union,
    cast,
)

import psutil
import torch
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader

from nanotron import distributed as dist
from nanotron import logging
from nanotron.config import (
    Config,
    DatasetStageArgs,
    ExistingCheckpointInit,
    ParallelismArgs,
    RandomInit,
    SpectralMupInit,
    get_config_from_file,
)
from nanotron.constants import MODEL_CONFIG_FILE_NAME
from nanotron.data.dataloader import sanity_check_dataloader
from nanotron.eval import LightEvalRunner
from nanotron.helpers import (
    _vocab_size_with_padding,
    compute_remain_train_steps_of_a_data_stage_from_ckp,
    get_consumed_train_samples_of_a_data_stage_from_ckp,
    get_profiler,
    init_optimizer_and_grad_accumulator,
    init_random_states,
    log_throughput,
    lr_scheduler_builder,
)
from nanotron.logging import (
    LoggerWriter,
    LogItem,
    human_format,
    log_libraries_versions,
    log_memory,
    log_rank,
    set_ranks_logging_level,
)
from nanotron.logging.timers import nanotron_timer
from nanotron.metrics_logging import MetricsLogger
from nanotron.models import NanotronModel, build_model
from nanotron.models.base import check_model_has_grad
from nanotron.models.llama import LlamaForTraining, RotaryEmbedding
from nanotron.models.qwen import Qwen2ForTraining
from nanotron.models.starcoder2 import Starcoder2ForTraining
from nanotron.optim.clip_grads import clip_grad_norm
from nanotron.parallel import ParallelContext
from nanotron.parallel.data_parallel.utils import sync_gradients_across_dp
from nanotron.parallel.parameters import NanotronParameter, sanity_check
from nanotron.parallel.pipeline_parallel.engine import (
    PipelineEngine,
    TensorPointer,
)
from nanotron.parallel.pipeline_parallel.utils import get_pp_rank_of
from nanotron.parallel.tensor_parallel.enum import TensorParallelLinearMode
from nanotron.parallel.tensor_parallel.nn import TensorParallelRowLinear
from nanotron.parallel.tied_parameters import (
    create_pg_for_tied_weights,
    get_tied_id_to_param,
    sync_tied_weights_gradients,
    tie_parameters,
)
from nanotron.random import set_random_seed
from nanotron.s3_checkpoints import S3Mover, check_path_is_local
from nanotron.sanity_checks import (
    after_optim_step_sanity_checks,
    after_tbi_sanity_checks,
    before_optim_step_sanity_checks,
    before_tbi_sanity_checks,
)
from nanotron.scaling.parametrization import ParametrizationMethod
from nanotron.serialize import (
    load_lr_scheduler,
    load_meta,
    load_weights,
    parse_ckpt_path,
    save,
    save_random_states,
)
from nanotron.serialize.metadata import DataStageMetadata, TrainingMetadata
from nanotron.serialize.optimizer import load_optimizer, state_dict_to_device

logger = logging.get_logger(__name__)

# Reduce the logging noise from torch.distributed when creating new process groups
dist_logger = logging.get_logger(dist.dist.__name__)
dist_logger.setLevel(logging.WARNING)

CONFIG_TO_MODEL_CLASS = {
    "LlamaConfig": LlamaForTraining,
    "Starcoder2Config": Starcoder2ForTraining,
    "Qwen2Config": Qwen2ForTraining,
}

try:
    import wandb
except ImportError:
    wandb = None


def get_size(bytes):
    # [FUNCTION DOC]
    # HELPER: human-readable byte formatting.
    # Behavior: divides by 1024 through units "", K, M, B, T, P and returns the first value below 1024
    # formatted with two decimals followed by the unit and "B".
    # Inputs: bytes (number). Outputs: str.
    # Exceptions: None explicitly raised. Side effects: None.
    # KNOWN ISSUE: uses "B" instead of "G" for gigabytes, and returns None for values of 1024 peta or more.
    """Convert bytes to human readable format"""
    for unit in ["", "K", "M", "B", "T", "P"]:
        if bytes < 1024:
            return f"{bytes:.2f}{unit}B"
        bytes /= 1024


class DistributedTrainer:
    # [CLASS DOC]
    # Purpose: End-to-end distributed trainer supporting data, tensor, pipeline, context and expert parallelism.
    # Key attributes: config, model_config, parallel_context, random_states, model (possibly DDP-wrapped),
    # unwrapped_model, optimizer, grad_accumulator, lr_scheduler, metadata (TrainingMetadata),
    # init_checkpoint_path, param_shard_metadata (set when weights are loaded), logger_ranks, loggerwriter,
    # micro_batch_size, n_micro_batches_per_batch, global_batch_size (samples), sequence_length,
    # iteration_step, limit_val_batches, current_dataloader / current_base_dl, iteration_timer, s3_mover,
    # lighteval_runner (world rank 0), metrics_logging, pipeline_engine, num_params, grad_norm_unclipped.
    # Extension points: pre_init, post_init, pre_training, post_train_step, post_training,
    # pre_save_checkpoint, post_save_checkpoint hooks, and _init_model_instance / _mark_tied_parameters.
    def __init__(
        self,
        config_or_config_file: Union[Config, str],
        config_class: Type[Config] = Config,
        model_config_class: Optional[Type] = None,
        model_class: Type[NanotronModel] = None,
    ):
        # [FUNCTION DOC]
        # Behavior: loads the config (from YAML path or object), optionally registers model_class for the
        # config class name, creates the ParallelContext (tp, pp, dp, expert, context sizes), runs pre_init
        # (resolves resume checkpoint), sets per-rank log levels, seeds RNG with seed plus TP rank, builds
        # named random states, builds the model (init_model), chooses STANDARD or SPECTRAL_MUP parametrization,
        # builds optimizer and grad accumulator and loads optimizer state (to CPU) if resuming and enabled,
        # builds the lr scheduler and loads it if enabled, loads TrainingMetadata from the checkpoint (asserting
        # train_steps exceeds the last step) or creates fresh metadata with one DataStageMetadata per config
        # stage, computes logger ranks (output pp rank with other ranks 0), sets batch-size fields and
        # iteration step, logs library versions and configs, optionally logs benchmark throughput, runs
        # post_init and creates the MetricsLogger.
        # Inputs: config_or_config_file (Config or YAML path str), config_class (default Config),
        # model_config_class (optional), model_class (optional NanotronModel subclass).
        # Outputs: None.
        # Exceptions: AssertionError for metadata problems; errors from model/optimizer building and loading.
        # Side effects: initializes distributed process groups; seeds global RNG; allocates the model on GPU;
        # file reads; extensive logging; mutates the CONFIG_TO_MODEL_CLASS registry.
        # KNOWN ISSUE: logging the absolute path of config_or_config_file raises TypeError when a Config object is passed.
        # KNOWN ISSUE: checkpoint metadata is only loaded when load_lr_scheduler is true, otherwise a resumed run restarts its step counters at 0.
        # KNOWN ISSUE: when the resume path is remote, weights are not loaded and param_shard_metadata is never set, so load_optimizer raises AttributeError.
        """
        Nanotron's distributed trainer.

        Args:
            config_or_config_file: Either a `Config` object or a path to a YAML file containing the config.
            config_class: The `Config` class to use.
            model_config_class: The `ModelConfig` class to use (for example `LlamaConfig`). Defaults to `None` which will use the model config class defined in the config.
            model_class: The `NanotronModel` class to use (for example `LlamaForTraining`). Defaults to `None` which will use the model class defined in the config.
        """

        super().__init__()
        # Get config
        if isinstance(config_or_config_file, str):
            # Get the yaml config file
            self.config = get_config_from_file(
                config_path=config_or_config_file,
                config_class=config_class,
                model_config_class=model_config_class,
            )
        else:
            self.config = config_or_config_file

        self.model_config = self.config.model.model_config
        if model_class is not None:
            CONFIG_TO_MODEL_CLASS[self.model_config.__class__.__name__] = model_class

        ########################################
        ## We start with setting up loggers and process groups
        ########################################

        # Initialise all process groups
        self.parallel_context = ParallelContext(
            tensor_parallel_size=self.config.parallelism.tp,
            pipeline_parallel_size=self.config.parallelism.pp,
            data_parallel_size=self.config.parallelism.dp,
            expert_parallel_size=self.config.parallelism.expert_parallel_size,
            context_parallel_size=self.config.parallelism.context_parallel_size,
        )

        self.pre_init()

        # Set log levels
        set_ranks_logging_level(parallel_context=self.parallel_context, logging_config=self.config.logging)

        ########################################
        ## Setting up our model, optimizers, schedulers, etc.
        ########################################

        # Set random states
        # Set different random seed for each TP rank to ensure diversity (especially at weight init)
        tp_rank = dist.get_rank(self.parallel_context.tp_pg)
        set_random_seed(self.config.general.seed + tp_rank)

        # Init model and build on pp ranks
        self.random_states = init_random_states(
            parallel_config=self.config.parallelism, tp_pg=self.parallel_context.tp_pg
        )
        self.model = self.init_model()  # Defines self.model
        self.unwrapped_model: NanotronModel = (
            self.model.module if isinstance(self.model, DistributedDataParallel) else self.model
        )

        # TODO: find a better way to handle this
        parametrization_method = (
            ParametrizationMethod.SPECTRAL_MUP
            if hasattr(self.config.model.init_method, "use_mup") and self.config.model.init_method.use_mup
            else ParametrizationMethod.STANDARD
        )

        # Init optimizer
        self.optimizer, self.grad_accumulator = init_optimizer_and_grad_accumulator(
            parametrization_method=parametrization_method,
            model=self.model,
            optimizer_args=self.config.optimizer,
            parallel_context=self.parallel_context,
        )
        if self.init_checkpoint_path is not None and self.config.checkpoints.load_optimizer:
            load_optimizer(
                optimizer=self.optimizer,
                parallel_context=self.parallel_context,
                root_folder=self.init_checkpoint_path,
                param_shard_metadata=self.param_shard_metadata,
                model=self.unwrapped_model,
                map_location="cpu",
            )

        # Init learning rate scheduler
        self.lr_scheduler = lr_scheduler_builder(
            optimizer=self.optimizer,
            lr_scheduler_args=self.config.optimizer.learning_rate_scheduler,
            total_training_steps=self.config.tokens.train_steps,
        )
        if self.init_checkpoint_path is not None and self.config.checkpoints.load_lr_scheduler:
            load_lr_scheduler(
                lr_scheduler=self.lr_scheduler,
                is_zero=self.config.optimizer.zero_stage,
                parallel_context=self.parallel_context,
                root_folder=self.init_checkpoint_path,
            )

        # Define iteration start state
        if self.init_checkpoint_path is not None and self.config.checkpoints.load_lr_scheduler:
            checkpoint_metadata = load_meta(
                parallel_context=self.parallel_context, root_folder=self.init_checkpoint_path
            )
            assert isinstance(checkpoint_metadata.metas, TrainingMetadata)
            log_rank(str(checkpoint_metadata), logger=logger, level=logging.INFO, rank=0)
            self.metadata: TrainingMetadata = checkpoint_metadata.metas
            # In case of a new datastage, metadata will be updated in `get_dataloader`
            assert (
                self.config.tokens.train_steps > self.metadata.last_train_step
            ), f"Loaded checkpoint has already trained {self.metadata.last_train_step} batches, you need to specify a higher `config.tokens.train_steps`"
        else:
            data_stages = [
                DataStageMetadata(
                    name=stage.name, start_training_step=stage.start_training_step, consumed_train_samples=0, sequence_length=stage.sequence_length
                )
                for stage in self.config.data_stages
            ]
            self.metadata: TrainingMetadata = TrainingMetadata(
                consumed_train_samples=0, consumed_tokens_total=0, last_train_step=0, last_stage_idx=0, data_stages=data_stages
            )

        # Setup tensorboard write and log writers on output rank
        self.logger_ranks = self.parallel_context.get_global_rank(
            ep_rank=0, pp_rank=self.unwrapped_model.output_pp_rank, dp_rank=0, tp_rank=0, cp_rank=0
        ).flatten()
        self.loggerwriter = self.setup_log_writers()

        self.micro_batch_size = self.config.tokens.micro_batch_size
        self.n_micro_batches_per_batch = self.config.tokens.batch_accumulation_per_replica
        self.global_batch_size = (
            self.micro_batch_size * self.n_micro_batches_per_batch * self.parallel_context.dp_pg.size()
        ) # in terms of samples
        self.sequence_length = (
            self.config.tokens.sequence_length
        )  # Global sequence length not divided by context parallel size
        self.iteration_step = self.metadata.last_train_step
        self.limit_val_batches = self.config.tokens.limit_val_batches
        self.current_dataloader: Optional[DataLoader] = None  # used for the current training stage
        self.current_base_dl: Optional[DataLoader] = None  # used for the current training stage
        self.iteration_timer = None  # Will be initialized during training

        log_libraries_versions(logger=logger)
        log_rank("Config:", logger=logger, level=logging.INFO, rank=0, is_separator=True)
        log_rank(
            f"Parsing config: {os.path.abspath(config_or_config_file)}", logger=logger, level=logging.INFO, rank=0
        )  # noqa
        log_rank(pformat(self.config), logger=logger, level=logging.INFO, rank=0)
        log_rank("Model Config:", logger=logger, level=logging.INFO, rank=0, is_separator=True)
        log_rank(pformat(self.model_config), logger=logger, level=logging.INFO, rank=0)
        # Log benchmark info
        if os.environ.get("NANOTRON_BENCHMARK", "0") == "1":
            log_throughput(self.config, self.parallel_context)
        self.post_init()

        # Initialize metrics logger
        self.metrics_logging = MetricsLogger(self.config)

    def pre_init(self):
        # [FUNCTION DOC]
        # Behavior: hook run before setup; sets init_checkpoint_path via parse_ckpt_path (which may download from S3).
        # Inputs: None. Outputs: None. Exceptions: see parse_ckpt_path. Side effects: sets attribute; possible download and collectives.
        self.init_checkpoint_path = parse_ckpt_path(config=self.config, parallel_context=self.parallel_context)

    def post_init(self):
        # [FUNCTION DOC]
        # Behavior: creates an S3Mover if s3_upload is configured (real only on LOCAL_RANK 0) else None; on world
        # rank 0 with a lighteval config creates a LightEvalRunner and, if S3 is enabled, sets its
        # eval_single_checkpoint as the post-upload callback.
        # Inputs: None. Outputs: None.
        # Exceptions: TypeError if LOCAL_RANK is unset while S3 upload is configured.
        # Side effects: sets s3_mover and lighteval_runner.
        # S3 Mover and save initial state
        if self.config.s3_upload is not None:
            # NOTE: Only local rank 0 should upload
            dummy = bool(int(os.environ.get("LOCAL_RANK", None)) != 0)
            self.s3_mover = S3Mover(
                local_path=self.config.checkpoints.checkpoints_path,
                s3_path=self.config.s3_upload.upload_s3_path,
                remove_after_upload=self.config.s3_upload.remove_after_upload,
                s5cmd_numworkers=self.config.s3_upload.s5cmd_numworkers,
                s5cmd_concurrency=self.config.s3_upload.s5cmd_concurrency,
                s5cmd_path=self.config.s3_upload.s5cmd_path,
                dummy=dummy,
            )
        else:
            self.s3_mover = None

        # Initialize LightEval runner on rank 0
        if dist.get_rank(self.parallel_context.world_pg) == 0:
            if self.config.lighteval is not None:
                self.lighteval_runner = LightEvalRunner(config=self.config, parallel_context=self.parallel_context)
                if self.s3_mover is not None:
                    # If we have S3 upload enabled, use the eval_single_checkpoint as post-upload callback
                    self.s3_mover.post_upload_callback = self.lighteval_runner.eval_single_checkpoint

    def pre_training(self, *args, **kwargs):
        # [FUNCTION DOC]
        # Behavior: warns that sanity checks are on (and asserts not in benchmark mode); logs the run summary;
        # if wandb is available, initializes runs: with TP above 1 and detailed metrics, one run per TP rank on
        # dp_cp rank 0; otherwise one run on the first logger rank, with optional system-metrics sampling
        # settings from STATS_SAMPLING_INTERVAL_IN_SEC, and uploads a temporary YAML copy of the config.
        # Inputs: args, kwargs (unused).
        # Outputs: None.
        # Exceptions: AssertionError if sanity checks are enabled under NANOTRON_BENCHMARK.
        # Side effects: logging; wandb network initialization; temp file write; reads environment variables.
        if not self.config.general.ignore_sanity_checks:
            log_rank(
                "Sanity checks are enabled, this will slow down the training. To disable them, set `config.general.ignore_sanity_checks` to `True`",
                logger=logger,
                level=logging.WARNING,
                rank=0,
            )
            assert (
                os.environ.get("NANOTRON_BENCHMARK", "0") != "1"
            ), "Sanity checks are enabled while you're running a benchmark. Make sure to disable them by setting `config.general.ignore_sanity_checks` to `True`"

        metadata: TrainingMetadata = self.metadata

        log_rank("Start training", logger=logger, level=logging.INFO, rank=0, is_separator=True)
        log_rank(
            f"mbs: {self.micro_batch_size} | grad_accum: {self.n_micro_batches_per_batch} | cp: {self.parallel_context.cp_pg.size()} | sequence_length: {self.sequence_length} | global_batch_size: {self.global_batch_size} | train_steps: {self.config.tokens.train_steps} | start_iteration_step: {metadata.last_train_step} | consumed_tokens_total: {metadata.consumed_tokens_total}",  # noqa
            logger=logger,
            level=logging.INFO,
            rank=0,
        )

        current_time = datetime.datetime.now().strftime("%d/%m/%Y_%H:%M:%S")

        # Initialize wandb for each TP group if TP > 1, but only for dp=0 ranks
        if wandb is not None:
            tp_size = self.parallel_context.tp_pg.size()
            dp_cp_rank = dist.get_rank(self.parallel_context.dp_cp_pg)
            tp_rank = dist.get_rank(self.parallel_context.tp_pg)
            world_rank = dist.get_rank(self.parallel_context.world_pg)

            if tp_size > 1 and self.metrics_logging.log_level > 0:
                # Create one wandb logger per TP group for DP=0 ranks
                if dp_cp_rank == 0:
                    # Create a run name that includes the TP group
                    run_name = f"{current_time}_{self.config.general.run}_tp_group_{tp_rank}"

                    wandb.init(
                        project=self.config.general.project,
                        name=run_name,
                        config={"nanotron_config": self.config.as_dict()},
                    )
                    log_rank(
                        f"Initialized wandb run '{run_name}' for TP rank {tp_rank}",
                        logger=logger,
                        level=logging.INFO,
                        rank=world_rank,
                    )
            elif world_rank == self.logger_ranks[0]:
                run_name = f"{current_time}_{self.config.general.run}"
                x_stats_sampling_interval = os.environ.get("STATS_SAMPLING_INTERVAL_IN_SEC", None)

                wandb_settings = {}

                if x_stats_sampling_interval is not None:
                    wandb_settings["x_stats_sampling_interval"] = float(x_stats_sampling_interval)
                    wandb_settings["x_stats_open_metrics_endpoints"] = {
                        "dcgm": "http://localhost:9104/metrics",
                        "node": "http://localhost:9100/metrics",
                        "lustre": "http://localhost:9106/metrics",
                        "gpu": "http://26.0.168.238:9103/metrics",
                        "efa": "http://localhost:9101/metrics",
                    }
                    wandb_settings["x_stats_open_metrics_filters"] = [
                        "DCGM_FI_",
                        "node_",
                        "lustre_",
                        "nvidia_gpu_",
                        "efa_",
                    ]

                wandb.init(
                    project=self.config.general.project,
                    name=run_name,
                    config={"nanotron_config": self.config.as_dict()},
                    settings=wandb.Settings(**wandb_settings),
                )
                # save config file
                temp_config_path = tempfile.mktemp(suffix=".yaml", prefix="config")
                self.config.save_as_yaml(temp_config_path)
                wandb.save(temp_config_path, base_path=os.path.dirname(temp_config_path), policy="now")
                log_rank(
                    f"Initialized wandb run '{run_name}' for TP rank {tp_rank}",
                    logger=logger,
                    level=logging.INFO,
                    rank=world_rank,
                )

    def post_train_step(self):
        # [FUNCTION DOC]
        # Behavior: clears the model's per-step tbi logs and polls the S3 mover.
        # Inputs: None. Outputs: None. Exceptions: None explicitly raised. Side effects: mutates model logs; may spawn subprocesses.
        self.unwrapped_model.clear_all_tbi_logs()
        # Update our background upload/removal of checkpoints
        if self.s3_mover is not None:
            self.s3_mover.update()

    def post_training(self):
        # [FUNCTION DOC]
        # Behavior: waits on all world ranks until pending S3 uploads finish.
        # Inputs: None. Outputs: None. Exceptions: None explicitly raised. Side effects: world collectives, sleeps.
        if self.s3_mover is not None:
            self.s3_mover.distributed_wait_for_completion(group=self.parallel_context.world_pg)

    def _update_dataloader_based_on_training_stages(self, dataloaders: Union[List[DataLoader], DataLoader]):
        # [FUNCTION DOC]
        # HELPER: selects the dataloader for the current data stage, called every iteration.
        # Behavior: without data stages, on first call wraps the given (or first of a tuple) dataloader with
        # sanity_check_dataloader and returns. A generator input (DoReMi) is used as-is. Otherwise finds the
        # stage to resume (latest stage whose start step is not after iteration_step); iterates stages from
        # metadata.last_stage_idx; when a stage starts at the current step, or it is the resume stage on first
        # call, clears the previous stage's dataloader from memory (if a real DataLoader), updates
        # last_stage_idx, logs resume progress, takes (and lazily instantiates) that stage's dataloader and stops.
        # If one was chosen, wraps it with sanity_check_dataloader and stores it.
        # Inputs: dataloaders (dictionary of stage name to dataloader or zero-argument factory, a single dataloader, tuple, or generator).
        # Outputs: None.
        # Exceptions: AssertionError if empty; KeyError for missing stage names.
        # Side effects: mutates current_dataloader, current_base_dl and metadata; frees memory; logging.
        from collections.abc import Generator

        if not hasattr(self.config, "data_stages") or self.config.data_stages is None:
            if self.current_dataloader is None:
                if isinstance(dataloaders, tuple):
                    dataloader = dataloaders[0]
                else:
                    dataloader = dataloaders
                self.current_dataloader = sanity_check_dataloader(
                    dataloader=dataloader, parallel_context=self.parallel_context, config=self.config
                )
                self.current_base_dl = dataloader
            return
        elif isinstance(dataloaders, Generator):
            # TODO(xrsrke): this is a hacky way to handle DoReMi's dataloader
            # remove this in the next PR
            self.current_dataloader = dataloaders
            return

        assert len(dataloaders) > 0, "No dataloaders provided"

        def clear_dataloader_from_memory(dataloader: DataLoader, stage_name: str):
            # [FUNCTION DOC]
            # HELPER: frees a finished stage's dataloader.
            # Behavior: logs, clears current_base_dl, deletes the dataloader's dataset, sampler and batch_sampler attributes and runs garbage collection.
            # Inputs: dataloader (DataLoader), stage_name (str). Outputs: None.
            # Exceptions: AttributeError if attributes are missing. Side effects: mutates dataloader and self; gc.
            import gc

            log_rank(
                f"[Training Stage: {stage_name}] Clearing the previous training stage's dataloader and datasets from memory",
                logger=logger,
                level=logging.INFO,
            )
            self.current_base_dl = None

            # NOTE: Clear dataloader from memory
            del dataloader.dataset
            del dataloader.sampler
            del dataloader.batch_sampler

            gc.collect()

        dataloader = None

        def find_stage_idx_to_resume():
            # [FUNCTION DOC]
            # HELPER: returns the index of the stage active at iteration_step.
            # Behavior: sorts stages by start step descending and returns the original-order index of the first
            # stage whose start step is not after iteration_step, or None.
            # Inputs: None (closure). Outputs: int or None. Exceptions: None. Side effects: None.
            # KNOWN ISSUE: the index conversion assumes config.data_stages is already sorted by start step ascending.
            reversed_data_stages = sorted(self.config.data_stages, key=lambda x: x.start_training_step, reverse=True)
            for idx, stage in enumerate(reversed_data_stages):
                if self.iteration_step >= stage.start_training_step:
                    return len(self.config.data_stages) - idx - 1
            return None

        stage_idx_to_resume = find_stage_idx_to_resume()

        for stage_idx, stage in enumerate(self.config.data_stages):
            if stage_idx < self.metadata.last_stage_idx:
                continue

            stage = cast(DatasetStageArgs, stage)

            is_resume_from_training = self.current_dataloader is None and stage_idx_to_resume == stage_idx
            if (stage.start_training_step == self.iteration_step) or is_resume_from_training:
                if self.current_dataloader is not None:
                    prev_stage_name = self.config.data_stages[stage_idx - 1].name
                    prev_dataloader = dataloaders[prev_stage_name]

                    if isinstance(prev_dataloader, DataLoader):
                        # NOTE: we don't need to clear dummy data generator from memory
                        clear_dataloader_from_memory(prev_dataloader, stage_name=stage.name)

                self.metadata.last_stage_idx = stage_idx

                if is_resume_from_training:
                    remaining_train_steps = compute_remain_train_steps_of_a_data_stage_from_ckp(
                        stage, self.config, self.metadata
                    )
                    (
                        consumed_train_steps,
                        consumed_tokens_per_dataset_folder,
                    ) = get_consumed_train_samples_of_a_data_stage_from_ckp(stage, self.metadata)
                    log_rank(
                        f"Resuming training from stage {stage.name}, it has trained for {consumed_train_steps} samples and has {remaining_train_steps} remaining train steps"
                        f"\nConsumed tokens per dataset folder: {pformat(consumed_tokens_per_dataset_folder)}",
                        logger=logger,
                        level=logging.INFO,
                        rank=0,
                    )

                dataloader = dataloaders[stage.name]
                # NOTE: if a dataloader is lazy initialized, we need to call it to initialize it
                dataloader = dataloader() if callable(dataloader) else dataloader
                break

        if dataloader is not None:
            self.current_dataloader = sanity_check_dataloader(
                dataloader=dataloader, parallel_context=self.parallel_context, config=self.config
            )
            self.current_base_dl = dataloader

    def train(
        self,
        dataloader_or_dls: Dict[
            str, Union[Iterator[Dict[str, Union[torch.Tensor, TensorPointer]]], Tuple[Iterator, ...]]
        ],
        **kwargs,
    ) -> None:
        # [FUNCTION DOC]
        # Behavior: optionally saves an initial checkpoint (fresh runs only); takes the pipeline engine from the
        # config and sets its micro-batch count; builds module_id_to_prefix on the unwrapped model; sets the
        # step range from last_train_step plus 1 to train_steps; runs pre_training; frees memory; under the
        # profiler, for each step: starts a CUDA iteration timer, selects the stage dataloader, runs
        # training_step, updates dataset consumption metrics and per-folder token counts when available,
        # updates metadata counters (samples, tokens, last step, stage samples; asserts stage sequence length
        # matches), logs at the configured interval, and saves a checkpoint at the checkpoint interval. Then
        # barriers, optionally saves a final checkpoint and runs post_training.
        # Inputs: dataloader_or_dls (dictionary of stage name to iterator(s), or a single dataloader), kwargs passed to pre_training.
        # Outputs: None.
        # Exceptions: AssertionError on sequence-length mismatch; propagates training errors.
        # Side effects: the entire training run: GPU compute, collectives, file writes, logging, profiling.
        if self.config.checkpoints.save_initial_state and self.init_checkpoint_path is None:
            self.save_checkpoint()

        self.pipeline_engine: PipelineEngine = self.config.parallelism.pp_engine

        self.pipeline_engine.nb_microbatches = self.n_micro_batches_per_batch

        # TODO @nouamanetazi: refactor this
        # Useful mapping
        self.unwrapped_model = self.model.module if isinstance(self.model, DistributedDataParallel) else self.model
        self.unwrapped_model.module_id_to_prefix = {
            id(module): f"{module_name}." for module_name, module in self.unwrapped_model.named_modules()
        }
        # Fix the root_model
        self.unwrapped_model.module_id_to_prefix[id(self.unwrapped_model)] = ""

        self.initial_iter_step = self.metadata.last_train_step + 1
        self.last_iter_step = self.config.tokens.train_steps
        self.pre_training(**kwargs)

        prof = get_profiler(config=self.config)
        # free memory
        gc.collect()
        torch.cuda.empty_cache()
        with prof:
            for self.iteration_step in range(self.initial_iter_step, self.last_iter_step + 1):
                if isinstance(prof, torch.profiler.profile):
                    logger.info(f"Profiler on for step {self.iteration_step}")
                    prof.step()

                # Use CUDA event-based timing for more accurate GPU-side elapsed time measurement
                self.iteration_timer = nanotron_timer("iteration_time", "cuda", cuda_sync=False, enabled=True)
                self.iteration_timer.start()
                self._update_dataloader_based_on_training_stages(dataloader_or_dls)

                # Training step
                outputs, loss_avg, z_loss_avg, tbi_logs = self.training_step(dataloader=self.current_dataloader)

                # Update consumption tracking for current batch
                if hasattr(self.current_base_dl, "dataset") and hasattr(self.current_base_dl.dataset, "update_consumption_metrics"):
                    # TODO: only works for BlendableDataset
                    self.current_base_dl.dataset.update_consumption_metrics(
                        start_idx=(self.iteration_step - 1) * self.global_batch_size,  # assumes we start from iteration_step=1
                        end_idx=self.iteration_step * self.global_batch_size,
                        sequence_length=self.sequence_length,
                    )

                # Training Logs
                # Track consumed tokens for all dataset folders in current stage
                if hasattr(self.current_base_dl, "dataset"):
                    consumption_stats = self.current_base_dl.dataset.get_consumption_stats()
                    current_stage = self.metadata.data_stages[self.metadata.last_stage_idx]

                    # Update consumed tokens for all folders in the consumption stats
                    for folder_path, stats in consumption_stats.items():
                        current_stage.consumed_tokens_per_dataset_folder[folder_path] = stats["tokens"]

                # Original consumption tracking
                self.metadata.consumed_train_samples += self.global_batch_size # TODO: Legacy: idc abt this
                self.metadata.consumed_tokens_total += self.global_batch_size * self.sequence_length
                self.metadata.last_train_step = self.iteration_step
                self.metadata.current_stage.consumed_train_samples += self.global_batch_size
                assert self.metadata.current_stage.sequence_length == self.sequence_length, "Sequence length mismatch between the current stage and the global sequence length"

                if (self.iteration_step - 1) % self.config.logging.iteration_step_info_interval == 0:
                    self.train_step_logs(outputs=outputs, loss_avg=loss_avg, z_loss_avg=z_loss_avg, tbi_logs=tbi_logs)

                # Checkpoint
                if self.iteration_step % self.config.checkpoints.checkpoint_interval == 0:
                    self.save_checkpoint()

        dist.barrier()  # let's wait for everyone before leaving

        if self.config.checkpoints.save_final_state:
            self.save_checkpoint()

        self.post_training()

    def training_step(
        self, dataloader: Iterator[Dict[str, Union[torch.Tensor, TensorPointer]]]
    ) -> Tuple[Iterable[Dict], Optional[torch.Tensor]]:
        # [FUNCTION DOC]
        # Behavior: before-step sanity checks; runs the pipeline engine train_batch_iter over
        # n_micro_batches_per_batch micro-batches drawn from dataloader (logging memory on the first 5 steps);
        # after-step sanity checks; waits on fp32 grad all-reduce handles when DDP with a grad accumulator;
        # without DDP manually all-reduces gradients across DP (context parallel unsupported); syncs tied
        # gradients; clips gradients if configured (storing grad_norm_unclipped); on ranks with real losses
        # sums micro-batch losses (and z_loss) and starts an async DP-CP average all-reduce of the loss; on the
        # first resumed step moves optimizer states to CUDA; before-optimizer sanity checks; fetches tbi logs;
        # optimizer step and zero_grad; scheduler step; after-optimizer sanity checks; waits on the loss
        # all-reduce; runs post_train_step. Each phase is timed.
        # Inputs: dataloader (iterator of micro-batch dictionaries).
        # Outputs: tuple (outputs list from the engine, loss_avg tensor or None, z_loss_avg tensor or None, tbi_logs).
        # Exceptions: NotImplementedError for context parallel without DDP; AssertionError from sanity checks or missing handles.
        # Side effects: forward/backward, collectives across dp/cp/tp/pp/tied groups, parameter updates, timers.
        # KNOWN ISSUE: return annotation lists two elements but four are returned.
        before_tbi_sanity_checks(
            self.config, self.parallel_context, self.unwrapped_model, self.grad_accumulator, self.lr_scheduler
        )

        if self.iteration_step < self.initial_iter_step + 5:
            log_memory(logger=logger, msg="Before train_batch_iter")

        nanotron_timer("train_batch_iter", "cuda").start()
        with torch.profiler.record_function("train_batch_iter"):
            outputs = self.pipeline_engine.train_batch_iter(
                model=self.model,
                pg=self.parallel_context.pp_pg,
                batch=(next(dataloader) for _ in range(self.n_micro_batches_per_batch)),
                nb_microbatches=self.n_micro_batches_per_batch,
                grad_accumulator=self.grad_accumulator,
            )
        nanotron_timer("train_batch_iter", "cuda").end()

        if self.iteration_step < self.initial_iter_step + 5:
            log_memory(logger=logger, msg="After train_batch_iter")

        after_tbi_sanity_checks(self.config, self.parallel_context, self.unwrapped_model, self.grad_accumulator)

        if isinstance(self.model, DistributedDataParallel) and self.grad_accumulator is not None:
            # Wait for fp32 grads allreduce to finish to make sure grads are synced across DP
            assert (
                self.grad_accumulator.fp32_grads_allreduce_handle is not None
            ), "No fp32_grads_allreduce_handle maybe you're using only a single training process"
            if isinstance(self.grad_accumulator.fp32_grads_allreduce_handle, list):
                for handle in self.grad_accumulator.fp32_grads_allreduce_handle:
                    handle.wait()
            else:
                self.grad_accumulator.fp32_grads_allreduce_handle.wait()

        nanotron_timer("sync_gradients", "cuda").start()
        # Sync tied weights
        if not isinstance(self.model, DistributedDataParallel):
            if self.parallel_context.context_parallel_size > 1:
                raise NotImplementedError("Context parallel size > 1 is not supported yet without DDP")
            # Manually sync across DP if it's not handled by DDP
            sync_gradients_across_dp(
                module=self.model,
                dp_pg=self.parallel_context.dp_pg,
                reduce_op=dist.ReduceOp.AVG,
                # TODO @thomasw21: This is too memory hungry, instead we run all_reduce
                reduce_scatter=False,  # optimizer.inherit_from(ZeroDistributedOptimizer),
                grad_accumulator=self.grad_accumulator,
            )

        # TODO @nouamane: Put this in hooks so we can overlap communication with gradient computation on the last backward pass.
        sync_tied_weights_gradients(
            module=self.unwrapped_model,
            parallel_context=self.parallel_context,
            grad_accumulator=self.grad_accumulator,
        )
        nanotron_timer("sync_gradients", "cuda").end()

        # Clip gradients
        nanotron_timer("clip_gradients", "cuda").start()
        if self.config.optimizer.clip_grad is not None:
            # Unwrap DDP
            named_parameters = [
                (name, param)
                for name, param in self.unwrapped_model.get_named_params_with_correct_tied()
                if param.requires_grad
            ]
            self.grad_norm_unclipped = clip_grad_norm(
                mp_pg=self.parallel_context.mp_pg,
                named_parameters=named_parameters,
                grad_accumulator=self.grad_accumulator,
                max_norm=self.config.optimizer.clip_grad,
            )
        nanotron_timer("clip_gradients", "cuda").end()

        # Compute DP-CP average loss and overlap with optimizer step
        if isinstance(outputs[0]["loss"], torch.Tensor):
            # This is an average on only one data rank.
            loss_avg = torch.stack(
                [output["loss"] for output in outputs]
            ).sum()  # already divided by n_micro_batches_per_batch
            if "z_loss" in outputs[0]:
                z_loss_avg = torch.stack(
                    [output["z_loss"] for output in outputs]
                ).sum()  # already divided by n_micro_batches_per_batch
            else:
                z_loss_avg = None
            # sync loss across DP-CP (we should do the same for z_loss but it's only for logging so let's not sync it rn)
            handle = dist.all_reduce(loss_avg, group=self.parallel_context.dp_cp_pg, async_op=True, op=dist.ReduceOp.AVG)
        else:
            z_loss_avg = None
            loss_avg = None
            handle = None

        # Move optimizer states back to GPU before optimizer step
        if (
            self.init_checkpoint_path is not None
            and self.config.checkpoints.load_optimizer
            and self.iteration_step == self.initial_iter_step
        ):
            state_dict_to_device(self.optimizer.state_dict(), "cuda")

        before_optim_step_sanity_checks(
            self.config, self.parallel_context, self.unwrapped_model, self.grad_accumulator, self.optimizer
        )

        # get_tbi_logs in a non-blocking way since we don't need to wait for it
        tbi_logs = self.unwrapped_model.get_tbi_logs(non_blocking=True)

        # Apply gradient
        nanotron_timer("optimizer_step", "cuda").start()
        self.optimizer.step()
        self.optimizer.zero_grad()
        nanotron_timer("optimizer_step", "cuda").end()

        # Update the learning rate
        self.lr_scheduler.step()

        after_optim_step_sanity_checks(self.config, self.parallel_context, self.unwrapped_model, self.grad_accumulator)

        if handle is not None:
            handle.wait()

        self.post_train_step()

        return outputs, loss_avg, z_loss_avg, tbi_logs

    def validation_step(self, dataloader: Iterator[Dict[str, Union[torch.Tensor, TensorPointer]]]) -> Iterable[Dict]:
        # [FUNCTION DOC]
        # Behavior: runs the pipeline engine's validate_batch_iter over limit_val_batches micro-batches.
        # Inputs: dataloader (iterator). Outputs: iterable of output dictionaries.
        # Exceptions: None explicitly raised. Side effects: forward passes, pipeline communication.
        # KNOWN ISSUE: validate_batch_iter is called without the pg argument passed in training, and pipeline_engine is only set in train.
        outputs = self.pipeline_engine.validate_batch_iter(
            model=self.model,
            batch=(next(dataloader) for _ in range(self.limit_val_batches)),
            nb_microbatches=self.limit_val_batches,
        )
        return outputs

    def train_step_logs(
        self,
        outputs: Iterable[Dict[str, Union[torch.Tensor, TensorPointer]]],
        loss_avg: Optional[torch.Tensor],
        z_loss_avg: Optional[torch.Tensor],
        tbi_logs: Optional[Dict[str, torch.Tensor]],
    ) -> None:
        # [FUNCTION DOC]
        # Behavior: world barrier; stops the iteration timer; computes tokens per second and model/hardware
        # TFLOPs; builds basic log items (tokens, time, throughput, global batch size in tokens, loss, lr,
        # TFLOPs, ETA; plus z_loss and grad_norm when present); logger ranks write them to the console; then
        # adds tbi_logs statistics (overall and per head), CPU/process memory (DEBUG_CPU), timers
        # (ENABLE_TIMERS) and per-dataset consumed tokens (DEBUG_DL, default on); decides which ranks log to
        # wandb and logs either detailed metrics (collected from the MetricsLogger, plus CUDA and disk usage
        # and TP group info) at the detail interval or the basic items; in benchmark mode at step 4 logs
        # throughput and exits the process unless profiling.
        # Inputs: outputs (engine outputs), loss_avg, z_loss_avg (optional tensors), tbi_logs (optional dictionary of name to tensor shaped heads by tokens).
        # Outputs: None.
        # Exceptions: AssertionError for missing loggerwriter/dataloader/wandb run.
        # Side effects: barrier; logging; wandb network calls; reads environment variables; may terminate the process.
        # KNOWN ISSUE: calls item on loss_avg unconditionally, which fails on pipeline ranks that do not compute the loss (loss_avg None).
        # KNOWN ISSUE: the final "Successfully logged ... to WandB" debug message is emitted even when nothing was logged to wandb.
        # TODO @nouamanetazi: Megatron-LM seems to be using a barrier to report their interval time. Check if this is necessary. https://github.com/NouamaneTazi/Megatron-LM/blob/e241a96c3085b18e36c6cee1d68a8155de77b5a6/megatron/training.py#L607
        dist.barrier()
        # End the iteration timer and get elapsed time in milliseconds
        self.iteration_timer.end()
        elapsed_time_per_iteration_ms = self.iteration_timer.elapsed * 1000
        tokens_per_sec = (
            self.global_batch_size * self.sequence_length / (elapsed_time_per_iteration_ms / 1000)
        )  # tokens_per_sec is calculated using sequence_length
        model_tflops, hardware_tflops = self.unwrapped_model.get_flops_per_sec(
            iteration_time_in_sec=elapsed_time_per_iteration_ms / 1000,
            sequence_length=self.sequence_length,
            global_batch_size=self.global_batch_size,
        )

        # Get rank information (used by both console and wandb logging)
        tp_size = self.parallel_context.tp_pg.size()
        dp_cp_rank = dist.get_rank(self.parallel_context.dp_cp_pg)
        tp_rank = dist.get_rank(self.parallel_context.tp_pg)
        world_rank = dist.get_rank(self.parallel_context.world_pg)

        # Prepare basic metrics (needed for both console logging and wandb)
        lr = self.lr_scheduler.get_last_lr()[0]
        remaining_steps = self.config.tokens.train_steps - self.iteration_step
        eta_seconds = int(remaining_steps * (elapsed_time_per_iteration_ms / 1000))
        basic_log_entries = [
            # LogItem("consumed_samples", self.consumed_train_samples, "human_format"),  # , "12d"),
            LogItem("consumed_tokens", self.metadata.consumed_tokens_total, "human_format"),
            LogItem("time_per_iteration_ms", elapsed_time_per_iteration_ms, "human_format"),  # , ".1f"),
            LogItem("tokens_per_sec", tokens_per_sec, "human_format"),  # , "1.6E"),
            LogItem(
                "tokens_per_sec_per_gpu", tokens_per_sec / self.parallel_context.world_pg.size(), "human_format"
            ),  # , "1.6E"),
            LogItem("global_batch_size", self.config.global_batch_size_in_tokens, "human_format"),  # , "5d"),
            LogItem("lm_loss", loss_avg.item(), "human_format"),  # , "1.6E"),
            LogItem("lr", lr, "human_format"),  # , ".3E"),
            LogItem("model_tflops_per_gpu", model_tflops, "human_format"),  # , ".2f"),
            # LogItem("hardware_tflops_per_gpu", hardware_tflops, "human_format"),  # , ".2f"),
            LogItem("eta", str(datetime.timedelta(seconds=eta_seconds))),
        ]

        def get_cpu_logitems():
            # [FUNCTION DOC]
            # HELPER: collects host memory statistics.
            # Behavior: returns log items for virtual memory, swap, and the main process and every child process
            # memory info (rss, shared, vms, text, data, lib, dirty), skipping vanished or inaccessible children.
            # Inputs: None. Outputs: list of LogItem. Exceptions: None explicitly raised (Linux-specific fields). Side effects: reads process info.
            # Add CPU memory usage metrics
            memory = psutil.virtual_memory()
            cpu_memory_log_entries = [
                LogItem("cpu_memory/total", memory.total, "human_format"),
                LogItem("cpu_memory/available_bytes", memory.available, "human_format"),
                LogItem("cpu_memory/used_bytes", memory.used, "human_format"),
                LogItem("cpu_memory/percent", memory.percent, "human_format"),
            ]

            # Add swap memory usage metrics
            swap = psutil.swap_memory()
            swap_memory_log_entries = [
                LogItem("swap_memory/total", swap.total, "human_format"),
                LogItem("swap_memory/free", swap.free, "human_format"),
                LogItem("swap_memory/used", swap.used, "human_format"),
                LogItem("swap_memory/percent", swap.percent, "human_format"),
            ]

            # Add detailed process memory info for main process and workers
            process = psutil.Process()
            worker_processes = []
            # Get all child processes
            try:
                worker_processes = process.children(recursive=True)
            except psutil.NoSuchProcess:
                pass

            # Log main process memory
            mem_info = process.memory_info()
            process_memory_log_entries = [
                LogItem("process_memory/main/rss", mem_info.rss, "human_format"),
                LogItem("process_memory/main/shared", mem_info.shared, "human_format"),
                LogItem("process_memory/main/vms", mem_info.vms, "human_format"),
                LogItem("process_memory/main/text", mem_info.text, "human_format"),
                LogItem("process_memory/main/data", mem_info.data, "human_format"),
                LogItem("process_memory/main/lib", mem_info.lib, "human_format"),
                LogItem("process_memory/main/dirty", mem_info.dirty, "human_format"),
            ]

            # Log worker process memory
            for idx, worker in enumerate(worker_processes):
                try:
                    worker_mem = worker.memory_info()
                    process_memory_log_entries.extend(
                        [
                            LogItem(f"process_memory/worker_{idx}/rss", worker_mem.rss, "human_format"),
                            LogItem(f"process_memory/worker_{idx}/shared", worker_mem.shared, "human_format"),
                            LogItem(f"process_memory/worker_{idx}/vms", worker_mem.vms, "human_format"),
                            LogItem(f"process_memory/worker_{idx}/text", worker_mem.text, "human_format"),
                            LogItem(f"process_memory/worker_{idx}/data", worker_mem.data, "human_format"),
                            LogItem(f"process_memory/worker_{idx}/lib", worker_mem.lib, "human_format"),
                            LogItem(f"process_memory/worker_{idx}/dirty", worker_mem.dirty, "human_format"),
                        ]
                    )
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue

            return cpu_memory_log_entries + swap_memory_log_entries + process_memory_log_entries

        if z_loss_avg is not None:
            basic_log_entries.insert(6, LogItem("z_loss", z_loss_avg.item(), "human_format"))  # , "1.6E"),

        if self.config.optimizer.clip_grad is not None:
            basic_log_entries.insert(
                5, LogItem("grad_norm", self.grad_norm_unclipped.item(), "human_format")
            )  # , ".3f"))

        # Console logging only on logger ranks
        if dist.get_rank(self.parallel_context.world_pg) in self.logger_ranks:
            assert self.loggerwriter is not None, "loggerwriter should be defined on logger ranks"
            self.loggerwriter.add_scalars_from_list(basic_log_entries, self.iteration_step)

        if tbi_logs is not None:
            for name, tensor in tbi_logs.items():
                # attn_probs is [num_local_heads, mbs * seq_len]
                basic_log_entries.append(LogItem(f"tbi_logs/{name}_mean", tensor.mean().item(), "human_format"))
                basic_log_entries.append(LogItem(f"tbi_logs/{name}_std", tensor.std().item(), "human_format"))
                basic_log_entries.append(LogItem(f"tbi_logs/{name}_max", tensor.max().item(), "human_format"))
                basic_log_entries.append(LogItem(f"tbi_logs/{name}_min", tensor.min().item(), "human_format"))
                # per head logs
                for head_idx in range(tensor.shape[0]):
                    basic_log_entries.append(LogItem(f"tbi_logs/{name}/head_{head_idx}_mean", tensor[head_idx].mean().item(), "human_format"))
                    basic_log_entries.append(LogItem(f"tbi_logs/{name}/head_{head_idx}_std", tensor[head_idx].std().item(), "human_format"))
                    basic_log_entries.append(LogItem(f"tbi_logs/{name}/head_{head_idx}_max", tensor[head_idx].max().item(), "human_format"))
                    basic_log_entries.append(LogItem(f"tbi_logs/{name}/head_{head_idx}_min", tensor[head_idx].min().item(), "human_format"))

        if os.environ.get("DEBUG_CPU", "0") == "1":
            basic_log_entries.extend(get_cpu_logitems())

        if os.environ.get("ENABLE_TIMERS", "0") == "1":
            for timer_name, timer in nanotron_timer.items():
                basic_log_entries.append(LogItem(f"timers/{timer_name}", timer.elapsed, ".2f"))

        if os.environ.get("DEBUG_DL", "1") == "1":
            assert self.current_base_dl is not None, "current_base_dl should be defined"

            # Log consumption statistics
            if hasattr(self.current_base_dl, "dataset"):
                for dataset_name, stats in self.current_base_dl.dataset.get_consumption_stats().items():
                    basic_log_entries.extend(
                        [
                            LogItem(f"dataloader/consumed_tokens/{dataset_name}", stats["tokens"], "human_format"),
                        ]
                    )

        # WandB logging - determine if this rank should log to wandb
        should_log_to_wandb = wandb is not None and (
            (tp_size > 1 and dp_cp_rank == 0 and self.metrics_logging.log_level > 0)
            or (tp_size > 1 and world_rank == self.logger_ranks[0] and self.metrics_logging.log_level == 0)
            or (tp_size == 1 and world_rank == self.logger_ranks[0])  # For TP>1, log from each TP group's dp=0 rank
        )
        should_log_detailed_metrics_to_wandb = (
            should_log_to_wandb
            and self.metrics_logging.log_level > 0
            and self.iteration_step % self.metrics_logging.log_detail_interval == 0
        )

        if should_log_detailed_metrics_to_wandb:
            assert not (
                wandb.run is None and tp_size > 1 and dp_cp_rank == 0
            ), f"WandB is not initialized for TP rank {tp_rank}, but logging was requested. Make sure that wandb is initialize before training."
            all_log_entries = list(basic_log_entries)

            # Collect all metrics based on the log level
            detailed_metrics = self.metrics_logging.collect_all_metrics(
                model=self.unwrapped_model,
            )

            # Add all detailed metrics to wandb
            for name, value in detailed_metrics.items():
                if isinstance(value, torch.Tensor):
                    value = value.item()
                all_log_entries.append(LogItem(name, value, "human_format"))

            total, used, free = shutil.disk_usage("/")
            all_log_entries.extend(
                [
                    LogItem(
                        "cuda_memory_allocated", torch.cuda.memory_allocated(), "human_format"
                    ),  #  / 1024**2, ".2f"),
                    LogItem(
                        "cuda_max_memory_reserved", torch.cuda.max_memory_reserved(), "human_format"
                    ),  #  / 1024**2, ".2f"),
                    LogItem("hd_total_memory_tb", total, "human_format"),  #  / (2**40), ".2f"),
                    LogItem("hd_used_memory_tb", used, "human_format"),  #  / (2**40), ".2f"),
                    LogItem("hd_free_memory_tb", free, "human_format"),  #  / (2**40), ".2f"),
                ]
            )
            if tp_size > 1 and self.metrics_logging.log_level > 0:
                tp_group_info = {"tp_rank": tp_rank, "tp_group_size": tp_size}
            else:
                tp_group_info = {}

            wandb.log(
                {
                    **{log_item.tag: log_item.scalar_value for log_item in all_log_entries},
                    **tp_group_info,
                    "iteration_step": self.iteration_step,
                },
                step=self.iteration_step,
            )
            log_rank(
                f"Successfully logged {len(all_log_entries)} metrics to WandB for tp_rank={tp_rank}",
                logger=logger,
                level=logging.DEBUG,
                rank=world_rank,
            )
        elif should_log_to_wandb:
            wandb.log(
                {
                    **{log_item.tag: log_item.scalar_value for log_item in basic_log_entries},
                    "iteration_step": self.iteration_step,
                },
                step=self.iteration_step,
            )
        log_rank(
            f"Successfully logged {len(basic_log_entries)} metrics to WandB for tp_rank={tp_rank}",
            logger=logger,
            level=logging.DEBUG,
            rank=world_rank,
        )

        # Nanotron Benchmark mode: we log the throughput and exit
        if os.environ.get("NANOTRON_BENCHMARK", "0") == "1" and self.iteration_step == 4:
            log_throughput(
                self.config,
                self.parallel_context,
                model_tflops,
                hardware_tflops,
                tokens_per_sec,
                num_params=self.num_params,
            )
            log_rank("Throughput logging complete", logger=logger, level=logging.INFO, rank=0)
            if not self.config.profiler:
                exit(0)

    def init_model(self) -> Union[NanotronModel, DistributedDataParallel]:
        # [FUNCTION DOC]
        # Behavior: pads vocab size to be divisible by TP size times make_vocab_size_divisible_by; warns if
        # max_position_embeddings differs from the training sequence length (fine-tuning vs ignored); builds
        # the model instance and loads or initializes weights.
        # Inputs: None. Outputs: model (NanotronModel or DDP wrapper).
        # Exceptions: see helpers. Side effects: mutates model_config.vocab_size; allocates model; logging.
        """Initialize the model and load weights from checkpoint if needed."""
        # TODO: add max_position_embeddings
        self.model_config.vocab_size = _vocab_size_with_padding(
            self.model_config.vocab_size,
            pg_size=self.parallel_context.tp_pg.size(),
            make_vocab_size_divisible_by=self.config.model.make_vocab_size_divisible_by,
        )

        if (
            getattr(self.model_config, "max_position_embeddings", None) is not None
            and self.model_config.max_position_embeddings != self.config.tokens.sequence_length
        ):
            if isinstance(self.config.model.init_method, ExistingCheckpointInit):
                log_rank(
                    f"Finetuning a model with a sequence length {self.config.tokens.sequence_length} that is different from the checkpoint's max_position_embeddings {self.model_config.max_position_embeddings}.",  # noqa
                    logger=logger,
                    level=logging.WARNING,
                    rank=0,
                )
            else:
                log_rank(
                    f"the model's max_position_embeddings {self.model_config.max_position_embeddings} is ignored because the sequence length we're training on is {self.config.tokens.sequence_length}.",  # noqa
                    logger=logger,
                    level=logging.WARNING,
                    rank=0,
                )
        model = self._init_model_instance()
        model = self._load_model_checkpoint(model)
        return model

    def _init_model_instance(self) -> NanotronModel:
        # [FUNCTION DOC]
        # HELPER: builds the model class registered for the model config.
        # Behavior: asserts the config class name is registered, then calls _init_model with a builder that
        # constructs the class with model config, parallel context, parallelism config and random states.
        # Inputs: None. Outputs: model. Exceptions: AssertionError for unsupported configs. Side effects: see _init_model.
        model_config_cls = self.model_config.__class__.__name__
        assert (
            model_config_cls in CONFIG_TO_MODEL_CLASS
        ), f"Unsupported model config {model_config_cls}. Only {CONFIG_TO_MODEL_CLASS.keys()} are supported"

        model = self._init_model(
            model_builder=lambda: CONFIG_TO_MODEL_CLASS[model_config_cls](
                config=self.model_config,
                parallel_context=self.parallel_context,
                parallel_config=self.config.parallelism,
                random_states=self.random_states,
            ),
        )
        return model

    def _load_model_checkpoint(self, model: NanotronModel) -> NanotronModel:
        # [FUNCTION DOC]
        # HELPER: weight loading or random initialization.
        # Behavior: if resuming from a local checkpoint, loads weights and stores param_shard_metadata (marks
        # reloaded also for remote paths). Otherwise rejects context parallel; for ExistingCheckpointInit loads
        # weights from that path; for random or spectral-muP init calls init_model_randomly then averages all
        # params across DP and tied params across their tied groups.
        # Inputs: model (NanotronModel or DDP). Outputs: the same model.
        # Exceptions: NotImplementedError for context parallel init; ValueError for unsupported init methods.
        # Side effects: file reads; in-place weight changes; collectives; sets param_shard_metadata.
        unwrapped_model = model.module if isinstance(model, DistributedDataParallel) else model

        # Load or initialize model weights
        reloaded_from_checkpoint = False
        if self.init_checkpoint_path is not None:
            # Load from a pre existing checkpoint
            if check_path_is_local(self.init_checkpoint_path):
                # Reload from a training checkpoint
                log_rank(
                    f"Loading weights from {self.init_checkpoint_path}", logger=logger, level=logging.INFO, rank=0
                )
                self.param_shard_metadata = load_weights(
                    model=unwrapped_model,
                    parallel_context=self.parallel_context,
                    root_folder=self.init_checkpoint_path,
                )
            reloaded_from_checkpoint = True
        if not reloaded_from_checkpoint:
            log_rank("No checkpoint path provided.", logger=logger, level=logging.INFO, rank=0)
            if self.parallel_context.context_parallel_size > 1:
                raise NotImplementedError("Init with Context parallel size > 1 not supported yet")
            if isinstance(self.config.model.init_method, ExistingCheckpointInit):
                # Initialize model from an pretrained model checkpoint (without optimizer, lr_scheduler...)
                self.param_shard_metadata = load_weights(
                    model=unwrapped_model,
                    parallel_context=self.parallel_context,
                    root_folder=self.config.model.init_method.path,
                )
            elif isinstance(self.config.model.init_method, (RandomInit, SpectralMupInit)):
                unwrapped_model.init_model_randomly(config=self.config)

                # Synchronize parameters so that the model is consistent
                # sync all params across dp
                for _, param in sorted(model.named_parameters(), key=lambda x: x[0]):
                    dist.all_reduce(param, op=dist.ReduceOp.AVG, group=self.parallel_context.dp_pg)

                # sync tied params across tied groups
                for (_, group_ranks), param in sorted(
                    get_tied_id_to_param(
                        parameters=model.parameters(),
                        root_module=unwrapped_model,
                    ).items(),
                    key=lambda x: x[0],
                ):
                    group = self.parallel_context.world_ranks_to_pg[group_ranks]
                    dist.all_reduce(param, op=dist.ReduceOp.AVG, group=group)
            else:
                raise ValueError(f"Unsupported {self.config.model.init_method}")

        return model

    def _init_model(
        self,
        model_builder: Callable[[], NanotronModel],
        target_pp_ranks: Optional[List[int]] = None,
    ) -> NanotronModel:
        # [FUNCTION DOC]
        # HELPER: constructs, ties, counts and optionally DDP-wraps the model.
        # Behavior: decides on DDP; builds the model on its pipeline ranks with the configured dtype;
        # initializes rotary embeddings; marks tied parameters; counts parameters and bytes, summing across TP
        # and PP (all-reduce), stores num_params, logs counts and memory; wraps in DistributedDataParallel over
        # the dp_cp group (no buffer broadcast, configured bucket size) after checking the model has grads;
        # runs the NanotronParameter sanity check.
        # Inputs: model_builder (zero-argument callable returning a NanotronModel), target_pp_ranks (optional list of ints).
        # Outputs: model (NanotronModel or DDP).
        # Exceptions: errors from build_model, check_model_has_grad or sanity_check.
        # Side effects: GPU allocation; collectives; process group creation for tied weights; logging.
        # KNOWN ISSUE: operator precedence in the DDP decision makes the ZeRO/fp32-accumulation exclusion apply only to the context-parallel clause, so DP above 1 always uses DDP.
        config = self.config
        parallel_context = self.parallel_context

        parallel_config = config.parallelism
        make_ddp = parallel_context.data_parallel_size > 1 or parallel_context.context_parallel_size > 1 and not (
            config.optimizer.accumulate_grad_in_fp32 and config.optimizer.zero_stage > 0
        )

        # Build model and set pp ranks
        model = build_model(
            parallel_context=parallel_context,
            dtype=config.model.dtype,
            target_pp_ranks=target_pp_ranks,
            model_builder=model_builder,
        )

        # Initialize rotary embeddings
        for module in model.modules():
            if not isinstance(module, RotaryEmbedding):
                continue
            module.init_rotary_embeddings()

        # Mark some parameters as tied
        self._mark_tied_parameters(model=model, parallel_context=parallel_context, parallel_config=parallel_config)

        # count number of parameters
        num_params = sum(p.numel() for p in model.parameters())
        size_params = sum(p.numel() * p.element_size() for p in model.parameters())
        total_params = torch.tensor(num_params, device="cuda")
        total_size = torch.tensor(size_params, device="cuda")
        dist.all_reduce(total_params, group=parallel_context.tp_pg, async_op=False, op=dist.ReduceOp.SUM)  # TP
        dist.all_reduce(total_params, group=parallel_context.pp_pg, async_op=False, op=dist.ReduceOp.SUM)  # PP
        dist.all_reduce(total_size, group=parallel_context.tp_pg, async_op=False, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_size, group=parallel_context.pp_pg, async_op=False, op=dist.ReduceOp.SUM)
        total_params = total_params.item()
        self.num_params = {"total": total_params, "local": num_params}

        # TODO @nouamanetazi: better memory logs
        log_rank(
            f"Total number of parameters: {human_format(total_params)} ({total_size.item() / 1024**2:.2f}MiB)",
            logger=logger,
            level=logging.INFO,
            group=parallel_context.world_pg,
            rank=0,
        )
        log_rank(
            f"Local number of parameters: {human_format(num_params)} ({size_params / 1024**2:.2f}MiB)",
            logger=logger,
            level=logging.INFO,
            group=parallel_context.dp_pg,
            rank=0,
        )
        log_rank(
            f"[After model building] Memory usage: {torch.cuda.memory_allocated() / 1024**2:.2f}MiB."
            f" Peak allocated: {torch.cuda.max_memory_allocated() / 1024**2:.2f}MiB"
            f" Peak reserved: {torch.cuda.max_memory_reserved() / 1024**2:.2f}MiB",
            logger=logger,
            level=logging.INFO,
            group=parallel_context.dp_pg,
            rank=0,
        )

        # Model make it DDP
        if make_ddp is True:
            # Check that the model has at least one grad. Necessary for DDP
            check_model_has_grad(model=model, parallel_context=parallel_context)
            # TODO @thomasw21: DDP doesn't support broadcasting complex buffers (and we don't really need that broadcasting anyway)
            model = DistributedDataParallel(
                model,
                process_group=parallel_context.dp_cp_pg,
                broadcast_buffers=False,
                bucket_cap_mb=config.model.ddp_bucket_cap_mb,
            )

        # Sanity check the model, all parameters must be NanotronParameter (either tied or sharded)
        sanity_check(root_module=model)

        return model

    def setup_log_writers(
        self,
    ) -> Optional[LoggerWriter]:
        # [FUNCTION DOC]
        # Behavior: returns a LoggerWriter (global step equal to train_steps) on logger ranks, else None.
        # Inputs: None. Outputs: LoggerWriter or None. Exceptions: None. Side effects: None.
        # KNOWN ISSUE: the docstring documents arguments the method does not take.
        """Setup all log writers on the appropriate ranks

        Args:
            config (Config): The config object
            logger_ranks (Iterable[int]): The ranks that should log
            parallel_context (DistributedProcessGroups): The distributed process groups
        """
        if dist.get_rank(self.parallel_context.world_pg) in self.logger_ranks:
            loggerwriter = LoggerWriter(global_step=self.config.tokens.train_steps)
        else:
            loggerwriter = None

        return loggerwriter

    def pre_save_checkpoint(self) -> Path:
        # [FUNCTION DOC]
        # Behavior: if a lighteval eval_interval_file exists, reads a new interval and applies it when it is a
        # multiple of checkpoint_interval (warning otherwise or on read errors); if S3 is enabled, waits for the
        # previous upload across the world group and logs launched eval job info from the callback output.
        # Inputs: None. Outputs: None.
        # Exceptions: None explicitly raised (read errors are caught).
        # Side effects: file read; mutates config.lighteval; collectives; logging.
        # KNOWN ISSUE: annotated to return a Path but returns None.
        # Check if eval_interval should be updated from file
        eval_interval_file = self.config.lighteval.eval_interval_file if self.config.lighteval is not None else None
        if eval_interval_file is not None and Path(eval_interval_file).exists():
            try:
                with open(eval_interval_file, "r") as f:
                    new_eval_interval = int(f.read().strip())

                # Verify that the new interval is a multiple of checkpoint_interval
                if new_eval_interval == self.config.lighteval.eval_interval:
                    pass
                elif new_eval_interval % self.config.checkpoints.checkpoint_interval == 0:
                    log_rank(
                        f"Updating lighteval.eval_interval from {self.config.lighteval.eval_interval} to {new_eval_interval}",
                        logger=logger,
                        level=logging.INFO,
                        rank=0,
                    )
                    self.config.lighteval.eval_interval = new_eval_interval
                else:
                    log_rank(
                        f"New eval_interval={new_eval_interval} must be a multiple of checkpoint_interval={self.config.checkpoints.checkpoint_interval}. Keeping current value: {self.config.lighteval.eval_interval}",
                        logger=logger,
                        level=logging.WARNING,
                        rank=0,
                    )
            except (ValueError, IOError) as e:
                log_rank(
                    f"Error reading eval_interval from file: {e}. Keeping current value: {self.config.lighteval.eval_interval}",
                    logger=logger,
                    level=logging.WARNING,
                    rank=0,
                )

        if self.s3_mover is not None:
            self.s3_mover.distributed_wait_for_completion(self.parallel_context.world_pg)
            if self.s3_mover.post_upload_callback_outputs is not None:
                slurm_job_id, slurm_log = self.s3_mover.post_upload_callback_outputs
                log_rank(
                    f"launching eval job: job_id={slurm_job_id} log at {slurm_log} slurm_eval",
                    logger=logger,
                    level=logging.WARNING,
                    rank=0,
                )

    def post_save_checkpoint(self):
        # [FUNCTION DOC]
        # Behavior: starts the S3 upload if enabled; on world rank 0 without S3 but with lighteval, runs an
        # evaluation of the just-saved checkpoint when the step matches eval_interval (or no interval is set).
        # Inputs: None. Outputs: None. Exceptions: None explicitly raised.
        # Side effects: subprocess launch; evaluation job launch.
        # KNOWN ISSUE: start_uploading is called without the step sub folder, so the whole checkpoints directory is uploaded each time.
        # Upload to S3
        if self.s3_mover is not None:
            self.s3_mover.start_uploading()

        if dist.get_rank(self.parallel_context.world_pg) == 0:
            if self.config.lighteval is not None and self.s3_mover is None:
                if (
                    self.config.lighteval.eval_interval is None
                    or self.iteration_step % self.config.lighteval.eval_interval == 0
                ):
                    checkpoint_path = Path(self.config.checkpoints.checkpoints_path) / f"{self.config.general.step}"
                    self.lighteval_runner.eval_single_checkpoint(checkpoint_path)

    def save_checkpoint(self) -> Path:
        # [FUNCTION DOC]
        # Behavior: pre-save hook; creates the step folder (world rank 0 on shared filesystems, else each node's
        # LOCAL_RANK 0) and barriers; updates config step and consumed samples; calls save (weights only from
        # dp_cp rank 0, config only from world rank 0, sanity checks unless disabled); saves random states;
        # writes latest.txt with the step; writes the model config JSON; post-save hook.
        # Inputs: None. Outputs: Path of the checkpoint folder.
        # Exceptions: TypeError if LOCAL_RANK is unset on non-shared filesystems; errors from save.
        # Side effects: file writes; world barriers and collectives; S3 upload; mutates config.
        # KNOWN ISSUE: every rank writes latest.txt concurrently, and the path join fails if checkpoints_path is a str.
        self.pre_save_checkpoint()

        checkpoints_path = self.config.checkpoints.checkpoints_path
        checkpoint_path = Path(checkpoints_path) / f"{self.iteration_step}"
        if self.config.checkpoints.checkpoints_path_is_shared_file_system:
            should_mkdir = dist.get_rank(self.parallel_context.world_pg) == 0
        else:
            should_mkdir = bool(int(os.environ.get("LOCAL_RANK", None)) == 0)
        if should_mkdir:
            checkpoint_path.mkdir(parents=True, exist_ok=True)
        dist.barrier(self.parallel_context.world_pg)

        log_rank(f"Saving checkpoint at {checkpoint_path}", logger=logger, level=logging.WARNING, rank=0)

        # Update step/samples numbers before we save the config
        self.config.general.step = self.metadata.last_train_step
        self.config.general.consumed_train_samples = self.metadata.consumed_train_samples # TODO: idc abt this

        save(
            model=self.unwrapped_model,
            optimizer=self.optimizer,
            lr_scheduler=self.lr_scheduler,
            should_save_model=bool(
                dist.get_rank(self.parallel_context.dp_cp_pg) == 0
            ),  # We only save the weights on DP_CP==0
            should_save_optimizer=True,
            should_save_lr_scheduler=True,
            should_save_config=bool(
                dist.get_rank(self.parallel_context.world_pg) == 0
            ),  # We only save the config on world_rank==0
            parallel_context=self.parallel_context,
            root_folder=checkpoint_path,
            training_metadata=self.metadata,
            config=self.config,
            sanity_checks=not self.config.general.ignore_sanity_checks,
        )
        save_random_states(
            random_states=self.random_states, parallel_context=self.parallel_context, root_folder=checkpoint_path
        )
        with open(checkpoints_path / "latest.txt", mode="w") as fo:
            fo.write(f"{self.iteration_step}")

        if hasattr(self.model_config, "to_json_file"):
            self.model_config.to_json_file(checkpoint_path / MODEL_CONFIG_FILE_NAME)
        else:
            with open(checkpoint_path / MODEL_CONFIG_FILE_NAME, mode="w") as fo:
                fo.write(json.dumps(asdict(self.model_config)))

        self.post_save_checkpoint()

        return checkpoint_path

    def _mark_tied_parameters(
        self,
        model: NanotronModel,
        parallel_context: ParallelContext,
        parallel_config: Optional[ParallelismArgs] = None,
    ):
        # [FUNCTION DOC]
        # HELPER: overridable hook delegating to mark_tied_parameters.
        # Inputs: model, parallel_context, parallel_config (optional). Outputs: None. Exceptions: see mark_tied_parameters. Side effects: ties params.
        mark_tied_parameters(model=model, parallel_context=parallel_context, parallel_config=parallel_config)


def mark_tied_parameters(
    model: NanotronModel, parallel_context: ParallelContext, parallel_config: Optional[ParallelismArgs] = None
):
    # [FUNCTION DOC]
    # Behavior: ties embedding and LM-head parameters across the pipeline ranks that own them (same ep, dp,
    # cp, tp coordinates) with SUM gradient reduction; calls the model's tie_custom_params; asserts the model
    # is not DDP; ties unsharded params across TP and across expert groups; creates process groups for all tied rank sets.
    # Inputs: model (NanotronModel), parallel_context, parallel_config (optional ParallelismArgs).
    # Outputs: None.
    # Exceptions: AssertionError if model is DDP.
    # Side effects: replaces parameters with tied NanotronParameters; collective process group creation.
    # Tie embeddings
    embeddings_lm_head_tied_names = model.get_embeddings_lm_head_tied_names()
    if len(embeddings_lm_head_tied_names) > 0:
        shared_embeddings = [
            (
                target,
                (
                    parallel_context.get_global_rank(
                        ep_rank=dist.get_rank(parallel_context.ep_pg),
                        pp_rank=get_pp_rank_of(target, module=model),
                        dp_rank=dist.get_rank(parallel_context.dp_pg),
                        cp_rank=dist.get_rank(parallel_context.cp_pg),
                        tp_rank=dist.get_rank(parallel_context.tp_pg),
                    ),
                ),
            )
            for target in embeddings_lm_head_tied_names
        ]
        tie_parameters(
            root_module=model, ties=shared_embeddings, parallel_context=parallel_context, reduce_op=dist.ReduceOp.SUM
        )

    # Tie custom params
    model.tie_custom_params()

    # Sync all parameters that have the same name and that are not sharded across TP and EXP
    assert not isinstance(model, DistributedDataParallel), "model shouldn't be DDP at this point"
    mark_unsharded_params_as_tied_across_tp(model, parallel_context, parallel_config)
    mark_unsharded_params_as_tied_across_expert(model, parallel_context, parallel_config)

    create_pg_for_tied_weights(root_module=model, parallel_context=parallel_context)


def mark_unsharded_params_as_tied_across_tp(
    model: NanotronModel, parallel_context: ParallelContext, parallel_config: "ParallelismArgs"
):
    # [FUNCTION DOC]
    # Behavior: for every direct parameter of every module, skipping already tied params, TP-sharded params
    # and the TensorParallelRowLinear bias (only on TP rank 0), ties it across all TP group ranks with reduce
    # op None in ALL_REDUCE mode (or when no config is given) and SUM otherwise (sequence-parallel inputs differ).
    # Inputs: model, parallel_context, parallel_config (ParallelismArgs or None).
    # Outputs: None. Exceptions: see tie_parameters. Side effects: replaces parameters on modules.
    for module_name, module in model.named_modules():
        for param_name, param in module.named_parameters(recurse=False):
            name = f"{module_name}.{param_name}"

            if isinstance(param, NanotronParameter):
                # We skip tying if param already tied or sharded along tp
                if param.is_tied:
                    continue

                if param.is_sharded:
                    sharded_info = param.get_sharded_info()
                    if sharded_info.is_tp_sharded(parallel_context=parallel_context):
                        continue

            if isinstance(module, TensorParallelRowLinear) and "bias" == param_name:
                # bias for TensorParallelRowLinear only exists on TP=0 so we don't need to tie it
                continue

            shared_weights = [
                (
                    name,
                    # sync across TP group
                    tuple(sorted(dist.get_process_group_ranks(parallel_context.tp_pg))),
                )
            ]

            if parallel_config is None or parallel_config.tp_mode is TensorParallelLinearMode.ALL_REDUCE:
                # We add `reduce_op=None` in order to signal that the weight are synced by design without needing to reduce
                # when TP=2 we have LN that is duplicated across TP, so by design it's tied
                reduce_op = None
            else:
                reduce_op = dist.ReduceOp.SUM

            tie_parameters(
                root_module=model, ties=shared_weights, parallel_context=parallel_context, reduce_op=reduce_op
            )


def mark_unsharded_params_as_tied_across_expert(
    model: NanotronModel, parallel_context: ParallelContext, parallel_config: "ParallelismArgs"
):
    # [FUNCTION DOC]
    # Behavior: for every direct parameter not already tied and not expert-sharded, ties it across the
    # expert-parallel group ranks with reduce op None (non-expert parts see the full tokens).
    # Inputs: model, parallel_context, parallel_config (unused).
    # Outputs: None. Exceptions: see tie_parameters. Side effects: replaces parameters on modules.
    # KNOWN ISSUE: since the TP pass runs first, every parameter tied across TP is skipped here, so non-expert params are never marked tied across the expert group when TP is used.
    for module_name, module in model.named_modules():
        for param_name, param in module.named_parameters(recurse=False):
            name = f"{module_name}.{param_name}"

            if isinstance(param, NanotronParameter):
                # We skip tying if param already tied or sharded along expert
                if param.is_tied:
                    continue

                if param.is_sharded:
                    sharded_info = param.get_sharded_info()
                    if sharded_info.is_expert_sharded(parallel_context):
                        continue

            shared_weights = [
                (
                    name,
                    # sync across expert group
                    tuple(sorted(dist.get_process_group_ranks(parallel_context.ep_pg))),
                )
            ]

            # Besides MoE block which sees shards tokens, the rest of the model sees the full tokens
            # so we don't need to reduce the gradients across expert group
            reduce_op = None

            tie_parameters(
                root_module=model, ties=shared_weights, parallel_context=parallel_context, reduce_op=reduce_op
            )
