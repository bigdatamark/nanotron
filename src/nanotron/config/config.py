# [MODULE DOC]
# Purpose: Core training configuration schema of nanotron. Defines one dataclass per YAML section (general,
# logging, metrics, datasets, data stages, checkpoints, profiler, model, tokenizer, tokens, LR scheduler,
# optimizers, generation, S3 upload) and the top-level Config that cross-validates them, plus loaders that build
# a Config from a dict or YAML file with dacite (strict mode, string-to-enum/dtype/engine type hooks).
# Main contents: DEFAULT_SEED (42), the dataclasses below, get_config_from_dict, get_config_from_file,
# _calculate_model_params. Re-exported by nanotron.config.
# Import-time side effects: imports torch, datasets, transformers and nanotron parallel modules; creates a logger.
import datetime
import glob
import os
from dataclasses import dataclass, fields
from pathlib import Path
from typing import List, Optional, Type, Union

import dacite
import torch
import yaml
from dacite import from_dict
from datasets.download.streaming_download_manager import xPath
from transformers import AutoTokenizer
from yaml.loader import SafeLoader

from nanotron.config.lighteval_config import LightEvalConfig
from nanotron.config.models_config import (
    ExistingCheckpointInit,
    LlamaConfig,
    MoEConfig,
    NanotronConfigs,
    Qwen2Config,
    RandomInit,
    SpectralMupInit,
    Starcoder2Config,
)
from nanotron.config.parallelism_config import ParallelismArgs
from nanotron.config.utils_config import (
    InitScalingMethod,
    RecomputeGranularity,
    cast_str_to_pipeline_engine,
    cast_str_to_torch_dtype,
    serialize,
)
from nanotron.generation.sampler import SamplerType
from nanotron.logging import get_logger, human_format
from nanotron.parallel.pipeline_parallel.engine import PipelineEngine
from nanotron.parallel.tensor_parallel.nn import TensorParallelLinearMode

logger = get_logger(__name__)

DEFAULT_SEED = 42


@dataclass
class BenchArgs:
    # [CLASS DOC]
    # Purpose: plain record of benchmark settings (model_name, sequence_length, micro_batch_size,
    # batch_accumulation_per_replica, benchmark_csv_path). Dataclass, no validation.
    model_name: str
    sequence_length: int
    micro_batch_size: int
    batch_accumulation_per_replica: int
    benchmark_csv_path: str


@dataclass
class LoggingArgs:
    # [CLASS DOC]
    # Purpose: log verbosity for the main rank (log_level) and replicas (log_level_replica), each one of debug, info,
    # warning, error, critical or passive (default info), and iteration_step_info_interval (default 1). Dataclass.
    """Arguments related to logging"""

    log_level: Optional[str] = None
    log_level_replica: Optional[str] = None
    iteration_step_info_interval: Optional[int] = 1

    def __post_init__(self):
        # [FUNCTION DOC]
        # Behavior: defaults both levels to info and validates them.
        # Inputs: self. Outputs: None. Exceptions: ValueError for an invalid level. Side effects: mutates self.
        if self.log_level is None:
            self.log_level = "info"
        if self.log_level not in [
            "debug",
            "info",
            "warning",
            "error",
            "critical",
            "passive",
        ]:
            raise ValueError(
                f"log_level should be a string selected in ['debug', 'info', 'warning', 'error', 'critical', 'passive'] and not {self.log_level}"
            )
        if self.log_level_replica is None:
            self.log_level_replica = "info"
        if self.log_level_replica not in [
            "debug",
            "info",
            "warning",
            "error",
            "critical",
            "passive",
        ]:
            raise ValueError(
                f"log_level_replica should be a string selected in ['debug', 'info', 'warning', 'error', 'critical', 'passive'] and not {self.log_level_replica}"
            )


@dataclass
class MetricsLoggingArgs:
    # [CLASS DOC]
    # Purpose: detailed metrics logging: log_level 0 (basic) or 1 (full), and log_detail_interval (positive int,
    # default 10). Dataclass.
    """Arguments related to metrics logging and tracking"""

    log_level: int = 0
    log_detail_interval: int = 10

    def __post_init__(self):
        # [FUNCTION DOC]
        # Behavior: validates log_level and log_detail_interval.
        # Inputs: self. Outputs: None.
        # Exceptions: intended ValueError on invalid values.
        # Side effects: None.
        # KNOWN ISSUE: the error messages reference nonexistent attributes level and interval, so an invalid value raises
        # AttributeError instead of the intended ValueError.
        if self.log_level not in [0, 1]:
            raise ValueError(f"metrics_level should be either 0 (basic) or 1 (full) and not {self.level}")
        if self.log_detail_interval <= 0:
            raise ValueError(f"metrics_interval should be a positive integer and not {self.interval}")


@dataclass
class PretrainDatasetsArgs:
    # [CLASS DOC]
    # Purpose: Hugging Face datasets source for pretraining: dataset name(s) or mixture dict, splits (default train),
    # config name, processing processes per rank (1), overwrite cache flag, text column (default text). Dataclass.
    hf_dataset_or_datasets: Union[str, list, dict]
    hf_dataset_splits: Optional[Union[str, list]] = None
    hf_dataset_config_name: Optional[str] = None
    dataset_processing_num_proc_per_process: Optional[int] = 1
    dataset_overwrite_cache: Optional[bool] = False
    text_column_name: Optional[str] = None

    def __post_init__(self):
        # [FUNCTION DOC]
        # Behavior: defaults text_column_name to text and hf_dataset_splits to train.
        # Inputs: self. Outputs: None. Exceptions: None explicitly raised. Side effects: mutates self.
        if self.text_column_name is None:
            self.text_column_name = "text"
        if self.hf_dataset_splits is None:
            self.hf_dataset_splits = "train"


@dataclass
class SFTDatasetsArgs:
    # [CLASS DOC]
    # Purpose: Hugging Face datasets source for supervised fine-tuning: like PretrainDatasetsArgs without a text
    # column, plus sft_dataloader flag (True) and debug_max_samples (cap on samples, None means all). Dataclass.
    # TODO @nouamane: which config do we want for SFT?
    hf_dataset_or_datasets: Union[str, list, dict]
    hf_dataset_splits: Optional[Union[str, list]] = None
    hf_dataset_config_name: Optional[str] = None
    dataset_processing_num_proc_per_process: Optional[int] = 1
    dataset_overwrite_cache: Optional[bool] = False
    sft_dataloader: Optional[bool] = True
    debug_max_samples: Optional[int] = None

    def __post_init__(self):
        # [FUNCTION DOC]
        # Behavior: defaults hf_dataset_splits to train. Inputs: self. Outputs: None.
        # Exceptions: None explicitly raised. Side effects: mutates self.
        if self.hf_dataset_splits is None:
            self.hf_dataset_splits = "train"


@dataclass
class S3UploadArgs:
    # [CLASS DOC]
    # Purpose: checkpoint upload to S3 via s5cmd: upload_s3_path, remove_after_upload, s5cmd worker count and
    # concurrency, s5cmd binary path. Path strings are converted to xPath. Dataclass.
    """Arguments related to uploading checkpoints on s3"""

    upload_s3_path: xPath
    remove_after_upload: bool
    s5cmd_numworkers: Optional[int]
    s5cmd_concurrency: Optional[int]
    s5cmd_path: Optional[xPath]

    def __post_init__(self):
        # [FUNCTION DOC]
        # Behavior: converts string upload_s3_path and s5cmd_path to xPath.
        # Inputs: self. Outputs: None. Exceptions: None explicitly raised. Side effects: mutates self.
        if isinstance(self.upload_s3_path, str):
            self.upload_s3_path = xPath(self.upload_s3_path)
        if isinstance(self.s5cmd_path, str):
            self.s5cmd_path = xPath(self.s5cmd_path)


@dataclass
class NanosetDatasetsArgs:
    # [CLASS DOC]
    # Purpose: pre-tokenized (datatrove / TokenizedBytes) dataset folders: dataset_folder (one or many),
    # dataset_weights (mixture weights), dataset_read_path (optional local copies read instead), tokenizer_name,
    # vocab_size and token_size_in_bytes (auto-filled from metadata files), return_positions, and TokenizedBytes
    # options (skip_in_stream, pad_samples_to_global_batch_size, dataset_max_tokens, shuffle_files,
    # use_old_brrr_dataloader). Dataclass.
    # Invariant: all folders share one tokenizer and token size.
    dataset_folder: Union[str, List[str]]
    dataset_weights: Optional[List[float]] = None
    dataset_read_path: Optional[
        Union[str, List[str]]
    ] = None  # Path to local file/copy to read from. If it exists, we read from this folder instead of from dataset_folder. Useful when we offload some data to remote and only keep the needed files on disk.
    # Tokenizer config, assuming all datasets use the same tokenizer
    tokenizer_name: Optional[str] = None
    vocab_size: Optional[int] = None
    token_size_in_bytes: Optional[int] = None
    return_positions: Optional[
        bool
    ] = True  # read positions stored in disk by datatrove if eos_token_id is None, else computed on the fly

    # Tokenized bytes dataset config
    skip_in_stream: Optional[bool] = False
    pad_samples_to_global_batch_size: Optional[bool] = False
    dataset_max_tokens: Optional[List[int]] = None
    shuffle_files: Optional[bool] = False
    use_old_brrr_dataloader: Optional[bool] = False

    def __post_init__(self):
        # [FUNCTION DOC]
        # Behavior: a single folder string becomes a one-element list with weight 1; checks weights count matches
        # folders; for each folder reads the first line of its first metadata file and, when it has the form
        # tokenizer|token_size, sets tokenizer_name, token_size_in_bytes and vocab_size (by loading the tokenizer) on
        # first encounter or asserts consistency afterwards; checks dataset_read_path count matches folders.
        # Inputs: self. Outputs: None.
        # Exceptions: ValueError on count mismatches; AssertionError on tokenizer or token size mismatch.
        # Side effects: mutates self, reads files, may download a tokenizer from the Hub.
        # KNOWN ISSUE: with a single folder string any user-provided dataset_weights are silently overwritten with 1;
        # a string dataset_read_path is length-checked by characters rather than as one path.
        if isinstance(self.dataset_folder, str):  # Case 1: 1 Dataset folder
            self.dataset_folder = [self.dataset_folder]
            self.dataset_weights = [1]

        # Check if dataset_weights is provided and matches the number of dataset folders
        if self.dataset_weights is not None and len(self.dataset_weights) != len(self.dataset_folder):
            raise ValueError(
                f"Number of dataset weights ({len(self.dataset_weights)}) does not match number of dataset folders ({len(self.dataset_folder)})"
            )

        # Read the first metadata file in the dataset folder to extract tokenizer name and token size.
        for folder in self.dataset_folder:
            # Find all metadata files in the folder
            metadata_files = glob.glob(os.path.join(folder, "*.metadata"))
            if metadata_files:
                # Read the first line of the first metadata file
                with open(metadata_files[0], "r") as f:
                    first_line = f.readline().strip()
                    if "|" in first_line:
                        tokenizer_name, token_size_in_bytes = first_line.split("|")
                        if self.tokenizer_name is None:
                            self.tokenizer_name = tokenizer_name
                            self.token_size_in_bytes = int(token_size_in_bytes)
                            self.vocab_size = len(AutoTokenizer.from_pretrained(tokenizer_name).get_vocab())
                        else:
                            assert (
                                self.tokenizer_name == tokenizer_name
                            ), f"Tokenizer name mismatch while reading datasets metadata file, found both {self.tokenizer_name} and {tokenizer_name}"
                            assert self.token_size_in_bytes == int(
                                token_size_in_bytes
                            ), f"Token size mismatch while reading datasets metadata file, found both {self.token_size_in_bytes} and {token_size_in_bytes}"

        # Check if dataset_read_path is provided and matches the number of dataset folders
        if self.dataset_read_path is not None and len(self.dataset_read_path) != len(self.dataset_folder):
            raise ValueError(
                f"Number of dataset read paths ({len(self.dataset_read_path)}) does not match number of dataset folders ({len(self.dataset_folder)})"
            )


@dataclass
class DataArgs:
    # [CLASS DOC]
    # Purpose: data section of a stage: dataset (pretrain, Nanoset or SFT args; None means dummy random data), seed
    # (default DEFAULT_SEED) and num_loading_workers (1). Dataclass.
    """Arguments related to the data and data files processing"""

    dataset: Optional[
        Union[PretrainDatasetsArgs, NanosetDatasetsArgs, SFTDatasetsArgs]
    ]  # If None we use dummy_infinite_data_generator
    seed: Optional[int]
    num_loading_workers: Optional[int] = 1

    def __post_init__(self):
        # [FUNCTION DOC]
        # Behavior: defaults seed. Inputs: self. Outputs: None. Exceptions: None explicitly raised.
        # Side effects: mutates self.
        if self.seed is None:
            self.seed = DEFAULT_SEED


@dataclass
class DatasetStageArgs:
    # [CLASS DOC]
    # Purpose: one training data stage: name, start_training_step (1-based step where it begins), data (DataArgs),
    # optional sequence_length (filled from tokens config by Config). Dataclass.
    """Arguments for loading dataset in different stages of the training process"""

    name: str
    start_training_step: int
    data: DataArgs
    sequence_length: Optional[int] = None  # if None, we use the sequence length from the config

    def __post_init__(self):
        # [FUNCTION DOC]
        # Behavior: rejects negative start_training_step. Inputs: self. Outputs: None.
        # Exceptions: ValueError when negative. Side effects: None.
        if self.start_training_step < 0:
            raise ValueError(f"training_steps should be a positive integer and not {self.start_training_step}")


@dataclass
class CheckpointsArgs:
    # [CLASS DOC]
    # Purpose: checkpointing: checkpoints_path, checkpoint_interval, save initial / final state flags, optional
    # resume path, whether to load LR scheduler and optimizer states, and whether the checkpoint path is on a shared
    # file system. Path strings become xPath. Dataclass.
    """Arguments related to checkpoints:
    checkpoints_path: where to save the checkpoints
    checkpoint_interval: how often to save the checkpoints
    resume_checkpoint_path: if you want to load from a specific checkpoint path
    """

    checkpoints_path: Path
    checkpoint_interval: int
    save_initial_state: Optional[bool] = False
    save_final_state: Optional[bool] = True
    resume_checkpoint_path: Optional[xPath] = None
    load_lr_scheduler: Optional[bool] = True
    load_optimizer: Optional[bool] = True
    checkpoints_path_is_shared_file_system: Optional[bool] = False

    def __post_init__(self):
        # [FUNCTION DOC]
        # Behavior: converts string paths to xPath. Inputs: self. Outputs: None. Exceptions: None explicitly raised.
        # Side effects: mutates self.
        if isinstance(self.checkpoints_path, str):
            self.checkpoints_path = xPath(self.checkpoints_path)
        if isinstance(self.resume_checkpoint_path, str):
            self.resume_checkpoint_path = xPath(self.resume_checkpoint_path)


@dataclass
class GeneralArgs:
    # [CLASS DOC]
    # Purpose: experiment identity: project, run name (default template with date and job id), seed, step and
    # consumed_train_samples (updated at checkpoint time), benchmark_csv_path, ignore_sanity_checks (default True).
    # Dataclass.
    """General training experiment arguments

    Args:
        project: Name of the project (a project gather several runs in common tensorboard/hub-folders)
        run: Name of the run
        step: Global step (updated when we save the checkpoint)
        consumed_train_samples: Number of samples consumed during training (should be actually just step*batch_size)
        ignore_sanity_checks: Whether to ignore sanity checks
    """

    project: str
    run: Optional[str] = None
    seed: Optional[int] = None
    step: Optional[int] = None
    consumed_train_samples: Optional[int] = None  # TODO: remove this
    benchmark_csv_path: Optional[Path] = None
    ignore_sanity_checks: bool = True

    def __post_init__(self):
        # [FUNCTION DOC]
        # Behavior: defaults seed; defaults run to a template and substitutes the date placeholder with the current
        # timestamp and the jobid placeholder with env SLURM_JOB_ID (or local).
        # Inputs: self. Outputs: None. Exceptions: None explicitly raised.
        # Side effects: mutates self; reads the environment and clock.
        if self.seed is None:
            self.seed = DEFAULT_SEED
        if self.run is None:
            self.run = "%date_%jobid"
        self.run = self.run.replace("%date", datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
        self.run = self.run.replace("%jobid", os.environ.get("SLURM_JOB_ID", "local"))


@dataclass
class ProfilerArgs:
    # [CLASS DOC]
    # Purpose: torch profiler schedule and options: export path, wait/warmup/active/repeat/skip_first step counts,
    # record shapes, memory profiling, stacks, chrome trace export. Dataclass, no validation.
    """Arguments related to profiling"""

    profiler_export_path: Optional[Path]  # e.g. ./tb_logs
    wait: int = 1
    warmup: int = 1
    active: int = 1
    repeat: int = 1
    skip_first: int = 3
    record_shapes: bool = False
    profile_memory: bool = False
    with_stack: bool = True
    export_chrome_trace: bool = False


@dataclass
class ModelArgs:
    # [CLASS DOC]
    # Purpose: model section: model_config (architecture config or raw dict), init_method (random, spectral muP or
    # existing checkpoint), dtype (default bfloat16), make_vocab_size_divisible_by (1), ddp_bucket_cap_mb (25).
    # Dataclass.
    """Arguments related to model architecture"""

    model_config: NanotronConfigs
    init_method: Union[RandomInit, SpectralMupInit, ExistingCheckpointInit]
    dtype: Optional[torch.dtype] = None
    make_vocab_size_divisible_by: int = 1
    ddp_bucket_cap_mb: int = 25

    def __post_init__(self):
        # [FUNCTION DOC]
        # Behavior: defaults dtype to bfloat16 and casts string dtypes; when model_config is a dict, converts a nested
        # moe_config dict to MoEConfig and instantiates Qwen2Config, LlamaConfig or Starcoder2Config according to the
        # is_*_config marker keys (Qwen2Config when none); sets the model config's muP flag when init_method is
        # SpectralMupInit.
        # Inputs: self. Outputs: None.
        # Exceptions: errors from dtype casting or config constructors (TypeError on unknown keys).
        # Side effects: mutates self and the model config.
        if self.dtype is None:
            self.dtype = torch.bfloat16
        if isinstance(self.dtype, str):
            self.dtype = cast_str_to_torch_dtype(self.dtype)

        if isinstance(self.model_config, dict):
            if isinstance(self.model_config.get("moe_config"), dict):
                self.model_config["moe_config"] = MoEConfig(**self.model_config["moe_config"])

            if self.model_config.get("is_qwen2_config"):
                self.model_config = Qwen2Config(**self.model_config)
            elif self.model_config.get("is_llama_config"):
                self.model_config = LlamaConfig(**self.model_config)
            elif self.model_config.get("is_starcoder2_config"):
                self.model_config = Starcoder2Config(**self.model_config)
            else:
                self.model_config = Qwen2Config(**self.model_config)

        self.model_config._is_using_mup = isinstance(self.init_method, SpectralMupInit)

        # if self.model_config.max_position_embeddings is None:
        #     self.model_config.max_position_embeddings = 0


@dataclass
class TokenizerArgs:
    # [CLASS DOC]
    # Purpose: tokenizer name or path, revision and max length. All optional. Dataclass.
    """Arguments related to the tokenizer"""

    tokenizer_name_or_path: Optional[str] = None
    tokenizer_revision: Optional[str] = None
    tokenizer_max_length: Optional[int] = None


@dataclass
class TokensArgs:
    # [CLASS DOC]
    # Purpose: sequence_length, train_steps, micro_batch_size, batch_accumulation_per_replica and validation/test
    # limits (val_check_interval -1, limit batches 0). Dataclass.
    """Arguments related to the tokens, sequence, batch and steps of the training"""

    sequence_length: int
    train_steps: int
    micro_batch_size: int
    batch_accumulation_per_replica: int

    val_check_interval: Optional[int] = -1
    limit_val_batches: Optional[int] = 0
    limit_test_batches: Optional[int] = 0


@dataclass
class LRSchedulerArgs:
    # [CLASS DOC]
    # Purpose: learning-rate schedule: peak learning_rate, warmup steps and style (linear or constant), decay style
    # (linear, cosine or 1-sqrt), decay steps (default computed by Config as train steps minus warmup), decay starting
    # step, min_decay_lr (default equal to learning_rate). Dataclass.
    """Arguments related to the learning rate scheduler

    lr_warmup_steps: number of steps to warmup the learning rate
    lr_warmup_style: linear or constant
    lr_decay_style: linear, cosine or 1-sqrt
    min_decay_lr: minimum learning rate after decay
    lr_decay_steps: optional number of steps to decay the learning rate otherwise will default to train_steps - lr_warmup_steps
    lr_decay_starting_step: optional number of steps to decay the learning rate otherwise will default to lr_warmup_steps
    """

    learning_rate: float
    lr_warmup_steps: int = 0
    lr_warmup_style: str = None
    lr_decay_style: str = None
    lr_decay_steps: Optional[int] = None
    lr_decay_starting_step: Optional[int] = None
    min_decay_lr: float = None

    def __post_init__(self):
        # [FUNCTION DOC]
        # Behavior: validates warmup style, defaults decay style to linear and validates it, defaults min_decay_lr.
        # Inputs: self. Outputs: None.
        # Exceptions: ValueError on invalid styles.
        # Side effects: mutates self.
        # KNOWN ISSUE: lr_warmup_style is validated before its None default is applied, so leaving it unset raises
        # ValueError and the linear default is unreachable.
        if self.lr_warmup_style not in ["linear", "constant"]:
            raise ValueError(
                f"lr_warmup_style should be a string selected in ['linear', 'constant'] and not {self.lr_warmup_style}"
            )
        if self.lr_warmup_style is None:
            self.lr_warmup_style = "linear"
        if self.lr_decay_style is None:
            self.lr_decay_style = "linear"
        if self.lr_decay_style not in ["linear", "cosine", "1-sqrt"]:
            raise ValueError(
                f"lr_decay_style should be a string selected in ['linear', 'cosine', '1-sqrt'] and not {self.lr_decay_style}"
            )
        if self.min_decay_lr is None:
            self.min_decay_lr = self.learning_rate


@dataclass
class SGDOptimizerArgs:
    # [CLASS DOC]
    # Purpose: selects plain SGD; only field is name (sgd). Dataclass.
    name: str = "sgd"


@dataclass
class AdamWOptimizerArgs:
    # [CLASS DOC]
    # Purpose: AdamW hyperparameters: eps, beta1, beta2, whether to use the fused torch implementation; name adamW.
    # Dataclass.
    adam_eps: float
    adam_beta1: float
    adam_beta2: float
    torch_adam_is_fused: bool
    name: str = "adamW"


@dataclass
class OptimizerArgs:
    # [CLASS DOC]
    # Purpose: optimizer section: optimizer_factory (SGD or AdamW args), zero_stage, weight_decay, clip_grad
    # (None disables), accumulate_grad_in_fp32, learning_rate_scheduler, weight_decay_exclude_named_params (regex
    # list, default empty). Dataclass.
    """Arguments related to the optimizer and learning rate"""

    optimizer_factory: Union[SGDOptimizerArgs, AdamWOptimizerArgs]
    zero_stage: int
    weight_decay: float
    clip_grad: Optional[float]
    accumulate_grad_in_fp32: bool
    learning_rate_scheduler: LRSchedulerArgs
    weight_decay_exclude_named_params: Optional[
        List[str]
    ] = None  # List of regex patterns to exclude parameters from weight decay

    def __post_init__(self):
        # [FUNCTION DOC]
        # Behavior: defaults the exclusion list to empty. Inputs: self. Outputs: None.
        # Exceptions: None explicitly raised. Side effects: mutates self.
        if self.weight_decay_exclude_named_params is None:
            self.weight_decay_exclude_named_params: List[str] = []


@dataclass
class GenerationArgs:
    # [CLASS DOC]
    # Purpose: generation settings (sampler converted from string to SamplerType, temperature, top_k, top_p,
    # n_samples, eos, seed default DEFAULT_SEED, use_cache). Dataclass.
    # KNOWN ISSUE: an identical GenerationArgs is also defined in lighteval_config; the package wildcard imports
    # lighteval_config last, so nanotron.config.GenerationArgs resolves to that one, not this one.
    sampler: Optional[Union[str, SamplerType]] = None
    temperature: Optional[float] = None
    top_k: Optional[int] = None
    top_p: Optional[float] = None
    n_samples: Optional[int] = None
    eos: Optional[str] = None
    seed: Optional[int] = None
    use_cache: Optional[bool] = False

    def __post_init__(self):
        # [FUNCTION DOC]
        # Behavior: converts string sampler to the enum and defaults seed.
        # Inputs: self. Outputs: None. Exceptions: KeyError on unknown sampler. Side effects: mutates self.
        if isinstance(self.sampler, str):
            self.sampler = SamplerType[self.sampler.upper()]
        if self.seed is None:
            self.seed = DEFAULT_SEED


@dataclass
class Config:
    # [CLASS DOC]
    # Purpose: top-level nanotron configuration combining all sections: general, parallelism, model (required),
    # tokenizer, checkpoints, logging, metrics_logging, tokens, optimizer, data_stages, profiler, lighteval and
    # s3_upload (optional). Post-init performs cross-section validation and derived defaults.
    # Construction: normally via get_config_from_file / load_from_yaml, or directly. Dataclass.
    """Main configuration class"""

    general: GeneralArgs
    parallelism: ParallelismArgs
    model: ModelArgs
    tokenizer: Optional[TokenizerArgs] = None
    checkpoints: Optional[CheckpointsArgs] = None
    logging: Optional[LoggingArgs] = None
    metrics_logging: Optional[MetricsLoggingArgs] = None
    tokens: Optional[TokensArgs] = None
    optimizer: Optional[OptimizerArgs] = None
    data_stages: Optional[List[DatasetStageArgs]] = None
    profiler: Optional[ProfilerArgs] = None
    lighteval: Optional[LightEvalConfig] = None
    s3_upload: Optional[S3UploadArgs] = None

    @classmethod
    def create_empty(cls):
        # [FUNCTION DOC]
        # Behavior: classmethod building a Config with every field None (bypasses typing; post-init still runs).
        # Inputs: cls. Outputs: Config.
        # Exceptions: AttributeError from post-init because the model section is None.
        # Side effects: None.
        # KNOWN ISSUE: post-init dereferences model and parallelism unconditionally, so this always fails.
        cls_fields = fields(cls)
        return cls(**{f.name: None for f in cls_fields})

    def __post_init__(self):
        # [FUNCTION DOC]
        # Behavior:
        # 1. With s3_upload: re-runs its post-init; with lighteval, defaults eval_interval to checkpoint_interval or
        #    asserts it is a multiple of it.
        # 2. With a profiler export path: asserts train_steps is at least the profiler's total scheduled steps.
        # 3. Defaults the LR decay steps to train steps minus warmup steps.
        # 4. With data_stages: sorts by start step, asserts one starts at step 1, rejects duplicate names or start steps
        #    (ValueError), and for Nanoset stages fills a model vocab_size of -1 from the dataset, asserts vocab sizes
        #    match, fills a missing tokenizer from the dataset and asserts tokenizer names match; asserts increasing order.
        # 5. Asserts head divisibility: attention heads divisible by tp, at least as many as KV heads, KV heads at least tp,
        #    attention heads divisible by KV heads.
        # 6. Fills each stage's sequence_length from tokens.sequence_length.
        # Inputs: self. Outputs: None.
        # Exceptions: AssertionError and ValueError for the checks above.
        # Side effects: mutates self and nested sections; logs warnings.
        # KNOWN ISSUE: the lighteval eval_interval defaulting only runs when s3_upload is set and assumes checkpoints is
        # not None; head checks use num_key_value_heads, which Starcoder2Config does not define (it uses num_kv_heads).
        if self.s3_upload is not None:
            self.s3_upload.__post_init__()
            if self.lighteval is not None:
                if self.lighteval.eval_interval is None:
                    self.lighteval.eval_interval = self.checkpoints.checkpoint_interval
                else:
                    assert (
                        self.lighteval.eval_interval % self.checkpoints.checkpoint_interval == 0
                    ), f"eval_interval={self.lighteval.eval_interval} must be a multiple of checkpoint_interval={self.checkpoints.checkpoint_interval}"

        # Some final sanity checks across separate arguments sections:
        if self.profiler is not None and self.profiler.profiler_export_path is not None:
            total_profiling_steps = self.profiler.skip_first + self.profiler.repeat * (
                self.profiler.wait + self.profiler.warmup + self.profiler.active
            )
            assert (
                self.tokens.train_steps >= total_profiling_steps
            ), f"Profiling steps ({total_profiling_steps}) must be less than or equal to train steps ({self.tokens.train_steps})"

        if self.optimizer is not None and self.optimizer.learning_rate_scheduler.lr_decay_steps is None:
            self.optimizer.learning_rate_scheduler.lr_decay_steps = (
                self.tokens.train_steps - self.optimizer.learning_rate_scheduler.lr_warmup_steps
            )

        if self.data_stages is not None:
            self.data_stages = sorted(self.data_stages, key=lambda stage: stage.start_training_step)
            names = [stage.name for stage in self.data_stages]
            training_steps = [stage.start_training_step for stage in self.data_stages]
            assert any(
                stage.start_training_step == 1 for stage in self.data_stages
            ), "You must have a training stage starting at 1 in the config's data_stages"

            for stage in self.data_stages:
                if names.count(stage.name) > 1:
                    raise ValueError(f"Each stage should have unique names and not {names}")

                if training_steps.count(stage.start_training_step) > 1:
                    raise ValueError(
                        f"Each stage should have unique starting training step, please change the starting training step for stage {stage.name}"
                    )

                if isinstance(stage.data.dataset, NanosetDatasetsArgs):
                    if self.model.model_config.vocab_size == -1:
                        self.model.model_config.vocab_size = stage.data.dataset.vocab_size
                        logger.warning(
                            f"Setting model's vocab_size to {self.model.model_config.vocab_size} from dataset's vocab_size ({stage.data.dataset.vocab_size})"
                        )
                    assert (
                        self.model.model_config.vocab_size == stage.data.dataset.vocab_size
                    ), f"Model's vocab_size ({self.model.model_config.vocab_size}) does not match dataset's ({stage.data.dataset.dataset_folder}) vocab_size ({stage.data.dataset.vocab_size})"
                    if self.tokenizer is None:
                        self.tokenizer = TokenizerArgs(tokenizer_name_or_path=stage.data.dataset.tokenizer_name)
                        logger.warning(
                            f"Setting tokenizer to {self.tokenizer.tokenizer_name_or_path} from dataset's tokenizer ({stage.data.dataset.tokenizer_name})"
                        )
                    assert (
                        self.tokenizer.tokenizer_name_or_path == stage.data.dataset.tokenizer_name
                    ), f"Tokenizer passed in config ({self.tokenizer.tokenizer_name_or_path}) does not match dataset's ({stage.data.dataset.dataset_folder}) tokenizer ({stage.data.dataset.tokenizer_name})"

            # NOTE: must order the stages by start_training_step from lowest to highest
            assert all(
                self.data_stages[i].start_training_step < self.data_stages[i + 1].start_training_step
                for i in range(len(self.data_stages) - 1)
            ), "The stages are not sorted by start_training_step in increasing order"

        # # if lighteval, we need tokenizer to be defined
        # if self.checkpoints.lighteval is not None:
        #     assert self.tokenizer.tokenizer_name_or_path is not None

        # Model verifications
        assert (
            self.model.model_config.num_attention_heads % self.parallelism.tp == 0
        ), f"num_attention_heads ({self.model.model_config.num_attention_heads}) must be divisible by tp ({self.parallelism.tp})"
        assert (
            self.model.model_config.num_attention_heads >= self.model.model_config.num_key_value_heads
        ), f"num_attention_heads ({self.model.model_config.num_attention_heads}) must be >= num_key_value_heads ({self.model.model_config.num_key_value_heads})"
        assert (
            self.model.model_config.num_key_value_heads >= self.parallelism.tp
        ), f"num_key_value_heads ({self.model.model_config.num_key_value_heads}) must be >= tp ({self.parallelism.tp})"  # TODO: remove this once we ensure KV heads get duplicated correctly
        assert (
            self.model.model_config.num_attention_heads % self.model.model_config.num_key_value_heads == 0
        ), f"num_attention_heads ({self.model.model_config.num_attention_heads}) must be divisible by num_key_value_heads ({self.model.model_config.num_key_value_heads})"

        # data_stages
        if self.data_stages is not None:
            for stage in self.data_stages:
                if stage.sequence_length is None:
                    stage.sequence_length = self.tokens.sequence_length

    @property
    def global_batch_size(self):
        # [FUNCTION DOC]
        # Behavior: property: micro batch size times batch accumulation times dp (samples per optimizer step).
        # Inputs: self. Outputs: int. Exceptions: AttributeError if tokens is None. Side effects: None.
        return self.tokens.micro_batch_size * self.tokens.batch_accumulation_per_replica * self.parallelism.dp

    @property
    def global_batch_size_in_tokens(self):
        # [FUNCTION DOC]
        # Behavior: property: global_batch_size times sequence length. Inputs: self. Outputs: int.
        # Exceptions: AttributeError if tokens is None. Side effects: None.
        return self.global_batch_size * self.tokens.sequence_length

    def save_as_yaml(self, file_path: str, sanity_checks: bool = True):
        # [FUNCTION DOC]
        # Behavior: serializes the config to a dict, writes it as YAML, and optionally reloads it with
        # get_config_from_file to verify it round-trips.
        # Inputs: file_path (str or path), sanity_checks (bool, default True). Outputs: None.
        # Exceptions: I/O errors; dacite/validation errors on reload. Side effects: writes the file.
        config_dict = serialize(self)
        file_path = str(file_path)
        with open(file_path, "w") as f:
            yaml.dump(config_dict, f)

        # Sanity test config can be reloaded
        if sanity_checks:
            _ = get_config_from_file(file_path, config_class=self.__class__)

    def get_yaml(self):
        # [FUNCTION DOC]
        # Behavior: returns the serialized config as a YAML string. Inputs: self. Outputs: str.
        # Exceptions: None explicitly raised. Side effects: None.
        config_dict = serialize(self)
        return yaml.dump(config_dict)

    @classmethod
    def load_from_yaml(cls, file_path: str):
        # [FUNCTION DOC]
        # Behavior: classmethod; reads YAML with the safe loader and builds the config via get_config_from_dict.
        # Inputs: file_path (str). Outputs: instance of cls.
        # Exceptions: I/O, YAML and dacite errors. Side effects: opens the file (handle not explicitly closed).
        config_dict = yaml.load(open(file_path), Loader=SafeLoader)
        return get_config_from_dict(config_dict, config_class=cls)

    def as_dict(self) -> dict:
        # [FUNCTION DOC]
        # Behavior: returns the serialized config dictionary. Inputs: self. Outputs: dict.
        # Exceptions: None explicitly raised. Side effects: None.
        return serialize(self)

    def print_config_details(self):
        # [FUNCTION DOC]
        # Behavior: prints model architecture fields with an estimated parameter count, training settings (sequence length,
        # mbs, accumulation, global batch and total tokens), parallelism sizes, ZeRO stage and layer recompute flag.
        # Inputs: self. Outputs: None. Exceptions: AttributeError when tokens or optimizer is None.
        # Side effects: prints to stdout.
        # KNOWN ISSUE: the printed seq_len is max_position_embeddings rather than tokens.sequence_length.
        print("\n=== Model Architecture ===")
        print(f"hidden_size: {self.model.model_config.hidden_size}")
        print(f"num_layers: {self.model.model_config.num_hidden_layers}")
        print(f"intermediate_size: {self.model.model_config.intermediate_size}")
        print(f"num_attention_heads: {self.model.model_config.num_attention_heads}")
        print(f"num_key_value_heads: {self.model.model_config.num_key_value_heads}")
        print(f"tie_word_embeddings: {self.model.model_config.tie_word_embeddings}")
        print(f"vocab_size: {self.model.model_config.vocab_size}")
        print(f"num_params: {_calculate_model_params(self)}")

        print("\n=== Training Configuration ===")
        print(
            f"seq_len: {self.model.model_config.max_position_embeddings} | mbs: {self.tokens.micro_batch_size} | batch_accum: {self.tokens.batch_accumulation_per_replica} | gbs: {human_format(self.global_batch_size_in_tokens)} | train_steps: {self.tokens.train_steps} | total_tokens: {human_format(self.tokens.train_steps * self.global_batch_size_in_tokens)}"
        )

        print("\n=== Parallelism ===")
        print(
            f"tp: {self.parallelism.tp} | pp: {self.parallelism.pp} | dp: {self.parallelism.dp} | cp: {self.parallelism.context_parallel_size} | ep: {self.parallelism.expert_parallel_size}"
        )
        print(f"zero_stage: {self.optimizer.zero_stage} | full_checkpointing: {self.parallelism.recompute_layer}")
        print("=" * 20 + "\n")


def get_config_from_dict(
    config_dict: dict, config_class: Type = Config, skip_unused_config_keys: bool = False, skip_null_keys: bool = False
):
    # [FUNCTION DOC]
    # Behavior: optionally keeps only keys matching fields of config_class (warning), optionally drops None values at
    # the first and second nesting level (warning), then builds config_class with dacite in strict mode, casting Path
    # and converting strings for torch dtype, pipeline engine and upper-cased enum names (TP mode, recompute
    # granularity, init scaling, sampler).
    # Inputs: config_dict (dict), config_class (dataclass type, default Config), skip_unused_config_keys (False),
    # skip_null_keys (False).
    # Outputs: instance of config_class.
    # Exceptions: dacite errors (unexpected keys, wrong types), KeyError for unknown enum names, dataclass validation
    # errors. Side effects: logging warnings.
    """Get a config object from a dictionary

    Args:
        args: dictionary of arguments
        config_class: type of the config object to get as a ConfigTypes (Config, LightevalConfig, LightevalSlurm) or str
        skip_unused_config_keys: whether to skip unused first-nesting-level keys in the config file (for config with additional sections)
        skip_null_keys: whether to skip keys with value None at first and second nesting level
    """
    if skip_unused_config_keys:
        logger.warning("skip_unused_config_keys set")
        config_dict = {
            field.name: config_dict[field.name] for field in fields(config_class) if field.name in config_dict
        }
    if skip_null_keys:
        logger.warning("Skip_null_keys set")
        config_dict = {
            k: {kk: vv for kk, vv in v.items() if vv is not None} if isinstance(v, dict) else v
            for k, v in config_dict.items()
            if v is not None
        }
    return from_dict(
        data_class=config_class,
        data=config_dict,
        config=dacite.Config(
            cast=[Path],
            type_hooks={
                torch.dtype: cast_str_to_torch_dtype,
                PipelineEngine: cast_str_to_pipeline_engine,
                TensorParallelLinearMode: lambda x: TensorParallelLinearMode[x.upper()],
                RecomputeGranularity: lambda x: RecomputeGranularity[x.upper()],
                InitScalingMethod: lambda x: InitScalingMethod[x.upper()],
                SamplerType: lambda x: SamplerType[x.upper()],
            },
            # strict_unions_match=True,
            strict=True,
        ),
    )


def get_config_from_file(
    config_path: str,
    config_class: Type = Config,
    model_config_class: Optional[Type] = None,
    skip_unused_config_keys: bool = False,
    skip_null_keys: bool = False,
) -> Config:
    # [FUNCTION DOC]
    # Behavior: loads YAML from config_path, builds the config with get_config_from_dict, and when model_config_class
    # is given checks the model config is a dict or that class and re-instantiates it with model_config_class.
    # Inputs: config_path (str), config_class (default Config), model_config_class (type or None),
    # skip_unused_config_keys, skip_null_keys (bools).
    # Outputs: config instance.
    # Exceptions: ValueError when the model config type is incompatible; I/O, YAML and dacite errors.
    # Side effects: reads the file.
    # KNOWN ISSUE: when model_config is already an instance of model_config_class, unpacking it as keyword arguments
    # raises TypeError; only dicts work. The docstring calls the parameter config_type and says python files are
    # supported, but only YAML is parsed.
    """Get a config object from a file (python or YAML)

    Args:
        config_path: path to the config file
        config_type: if the file is a python file, type of the config object to get as a
            ConfigTypes (Config, LightevalConfig, LightevalSlurm) or str
            if None, will default to Config
        skip_unused_config_keys: whether to skip unused first-nesting-level keys in the config file (for config with additional sections)
        skip_null_keys: whether to skip keys with value None at first and second nesting level
    """
    # Open the file and load the file
    with open(config_path) as f:
        config_dict = yaml.load(f, Loader=SafeLoader)

    config = get_config_from_dict(
        config_dict,
        config_class=config_class,
        skip_unused_config_keys=skip_unused_config_keys,
        skip_null_keys=skip_null_keys,
    )
    if model_config_class is not None:
        if not isinstance(config.model.model_config, (dict, model_config_class)):
            raise ValueError(
                f"model_config should be a dictionary or a {model_config_class} and not {config.model.model_config}"
            )
        config.model.model_config = model_config_class(**config.model.model_config)
    return config


def _calculate_model_params(config: Config):
    # [FUNCTION DOC]
    # HELPER: human-formatted approximate parameter count for print_config_details.
    # Behavior: vocab times hidden times an embedding factor, plus layers times (3 times hidden times intermediate plus
    # 4 times hidden squared).
    # Inputs: config (Config). Outputs: str.
    # Exceptions: None explicitly raised. Side effects: None.
    # KNOWN ISSUE: the embedding factor is 2 when embeddings are tied and 1 when untied, which is inverted.
    """Calculate and format the number of parameters in the model.
    N = vocab * h * 2 + num_layers * (3 * h * inter + 4 * h * h)
    """
    num_params = human_format(
        config.model.model_config.vocab_size
        * config.model.model_config.hidden_size
        * (2 if config.model.model_config.tie_word_embeddings else 1)
        + config.model.model_config.num_hidden_layers
        * (
            3 * config.model.model_config.hidden_size * config.model.model_config.intermediate_size
            + 4 * config.model.model_config.hidden_size * config.model.model_config.hidden_size
        )
    )
    return num_params
