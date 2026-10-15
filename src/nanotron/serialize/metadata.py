# [MODULE DOC]
# Purpose: Dataclasses and JSON (de)serialization for checkpoint-level and tensor-level metadata.
# Main contents: DataStageMetadata, TrainingMetadata, CheckpointMetadata, TensorMetadata, the
# recursive conversion helpers process_type / to_dict / to_list, and save_meta / load_meta.
# Fits in nanotron: the trainer keeps a TrainingMetadata of consumed samples/tokens per data stage;
# save_meta writes it with topology info into the checkpoint metadata JSON file; TensorMetadata is
# stored in each safetensors file header to describe how a shard maps into the unsharded tensor.
# Import-time side effects: None.
import dataclasses
import json
import re
from pathlib import Path
from typing import Any, Callable, ClassVar, Dict, List, Optional, Tuple, Type, Union

import dacite
import torch
from dacite import from_dict
from packaging.version import Version

from nanotron import distributed as dist
from nanotron.constants import CHECKPOINT_FILE_NAME, CHECKPOINT_VERSION
from nanotron.parallel import ParallelContext
from nanotron.parallel.parameters import SlicesPair
from collections import defaultdict

@dataclasses.dataclass
class DataStageMetadata:
    # [CLASS DOC]
    # Purpose: Progress counters for one data stage (reset at each stage).
    # Key attributes: name, start_training_step, consumed_train_samples (used by the sampler),
    # sequence_length (defaults to 4096 when None), consumed_tokens_per_dataset_folder (dictionary of
    # folder name to token count).
    """
    consumed_train_samples: The number of samples consumed by the model in the this stage (resets at each stage).
    consumed_tokens_per_dataset_folder: The number of tokens consumed by the model in the this stage for each dataset folder. (resets at each stage)
    """

    name: str
    start_training_step: int
    consumed_train_samples: int # We use this for sampler, and it's reset at each stage
    sequence_length: Optional[int] = None # TODO: put back as non-optional
    consumed_tokens_per_dataset_folder: Dict[str, int] = dataclasses.field(default_factory=dict) # this gets reset at each stage

    def __post_init__(self):
        # [FUNCTION DOC]
        # Behavior: fills sequence_length with 4096 when it is None (temporary default).
        # Inputs: None. Outputs: None. Exceptions: None. Side effects: mutates self.
        if self.sequence_length is None:
            self.sequence_length = 4096 # TODO: temp

    def sanity_consumed_train_samples(self):
        # [FUNCTION DOC]
        # Behavior: asserts consumed samples times sequence length equals the sum of per-folder tokens.
        # Inputs: None. Outputs: None. Exceptions: AssertionError on mismatch. Side effects: None.
        assert self.consumed_train_samples*self.sequence_length == sum(self.consumed_tokens_per_dataset_folder.values()), f"Mismatch between the total consumed samples and the sum of consumed samples across dataset folders! consumed_train_samples={self.consumed_train_samples}, sequence_length={self.sequence_length}, consumed_tokens_per_dataset_folder={self.consumed_tokens_per_dataset_folder}"

    @property
    def consumed_tokens_all_datasets(self):
        # [FUNCTION DOC]
        # Behavior: property returning the sum of per-folder token counts.
        # Inputs: None. Outputs: int. Exceptions: None. Side effects: None.
        return sum(self.consumed_tokens_per_dataset_folder.values())

@dataclasses.dataclass
class TrainingMetadata:
    # [CLASS DOC]
    # Purpose: Global training progress across stages.
    # Key attributes: consumed_train_samples (legacy global count), last_train_step, consumed_tokens_total
    # (derived if None), last_stage_idx (optional), data_stages (optional list of DataStageMetadata).
    # Invariant: global sample and token counts equal the sums over stages.
    """
    consumed_train_samples: The number of samples consumed globally, across all stages.
    last_train_step: The last training step across all stages.
    last_stage_idx: The index of the last stage that was trained.
    data_stages: The metadata for each stage.
    """

    consumed_train_samples: int # TODO: Legacy. This assumed same sequence length across all stages. Not used anymore
    last_train_step: int
    consumed_tokens_total: Optional[int] = None # TODO: put back as non-optional

    # TODO(xrsrke): make this not optional, once we entirely remove
    # the old checkpoint version
    last_stage_idx: Optional[int] = None
    data_stages: Optional[List[DataStageMetadata]] = None

    def __post_init__(self):
        # [FUNCTION DOC]
        # Behavior: asserts consumed_train_samples equals the sum over stages; asserts or fills
        # consumed_tokens_total from the per-stage token sums; asserts data_stages is present if
        # last_stage_idx is set.
        # Inputs: None. Outputs: None.
        # Exceptions: AssertionError on mismatches.
        # Side effects: may set consumed_tokens_total.
        # KNOWN ISSUE: the sums iterate data_stages before the None check, so data_stages None (declared allowed) raises TypeError; the last assertion is therefore unreachable in that case.
        # NOTE: this is a sanity check after loading a trained checkpoint
        assert (
            self.consumed_train_samples == sum(stage.consumed_train_samples for stage in self.data_stages)
        ), "Mismatch between the total consumed samples and the sum of consumed samples across stages! Something went wrong in the training."

        if self.consumed_tokens_total is not None:
            assert self.consumed_tokens_total == sum(stage.consumed_tokens_all_datasets for stage in self.data_stages), "Mismatch between the total consumed tokens and the sum of consumed tokens across stages! Something went wrong in the training."
        else:
            self.consumed_tokens_total = sum(stage.consumed_tokens_all_datasets for stage in self.data_stages)

        # TODO(xrsrke): remove this once we entirely remove non-data-stage training
        if self.last_stage_idx is not None:
            assert self.data_stages is not None, "data_stages should not be None if last_stage_idx is not None"

    @property
    def consumed_tokens_per_dataset_folder_total(self):
        # [FUNCTION DOC]
        # Behavior: property summing tokens per dataset folder across all stages.
        # Inputs: None. Outputs: defaultdict of folder name to int. Exceptions: TypeError if data_stages is None. Side effects: None.
        consumed = defaultdict(int)
        for stage in self.data_stages:
            for dataset_folder, tokens in stage.consumed_tokens_per_dataset_folder.items():
                consumed[dataset_folder] += tokens
        return consumed
    
    @property
    def current_stage(self) -> DataStageMetadata:
        # [FUNCTION DOC]
        # Behavior: property returning the stage at last_stage_idx.
        # Inputs: None. Outputs: DataStageMetadata. Exceptions: TypeError/IndexError if unset or out of range. Side effects: None.
        return self.data_stages[self.last_stage_idx]


@dataclasses.dataclass
class CheckpointMetadata:
    # [CLASS DOC]
    # Purpose: Content of the checkpoint metadata file: checkpoint version, tp, dp, cp sizes (cp default 1),
    # the TrainingMetadata, and optional custom_metas dictionary.
    version: Version
    tp: int
    dp: int
    metas: TrainingMetadata
    cp: int = 1
    custom_metas: Optional[Dict[str, Any]] = None


@dataclasses.dataclass
class TensorMetadata:
    # [CLASS DOC]
    # Purpose: Per-tensor metadata stored in safetensors headers as strings.
    # Key attributes: version, local_global_slices_pairs (tuple of SlicesPair mapping local shard slices
    # to global slices), unsharded_shape (tuple of ints); class-level dacite config with type hooks that
    # parse slices pairs and shape strings (also tolerating numpy integer reprs), strict mode.
    # Mandatory for checkpoint version higher than 1.2
    version: Version
    # Anything users want to store
    # Info of to what slice of the unsharded tensor (global_slices) the current sharded tensor corresponds (local_slices)
    local_global_slices_pairs: Tuple[SlicesPair, ...]
    # The shape of the unsharded tensor
    unsharded_shape: Tuple[int, ...]

    _metadata_config: ClassVar[dacite.Config] = dacite.Config(
        cast=[Version],
        type_hooks={
            Tuple[SlicesPair, ...]: SlicesPair.tuple_from_str,
            Tuple[int, ...]: lambda x: torch.Size(
                int(re.search(r"\((\d+)\)", size).group(1)) if "np.int" in size else int(size)
                for size in x.strip("()").split(",")
                if size
            ),
        },
        strict=True,
    )

    def to_str_dict(self) -> Dict[str, str]:
        # [FUNCTION DOC]
        # Behavior: serializes version, slices pairs and unsharded shape to strings.
        # Inputs: None. Outputs: dictionary of str to str. Exceptions: None explicitly raised. Side effects: None.
        return {
            "version": str(self.version),
            "local_global_slices_pairs": SlicesPair.tuple_to_str(self.local_global_slices_pairs),
            "unsharded_shape": str(tuple(self.unsharded_shape)),
        }

    @classmethod
    def from_str_dict(cls, dictionary: Dict[str, str]) -> "TensorMetadata":
        # [FUNCTION DOC]
        # Behavior: class method that parses the string dictionary back via dacite with the class config.
        # Inputs: dictionary (str to str). Outputs: TensorMetadata.
        # Exceptions: dacite errors on missing/extra keys or bad values. Side effects: None.
        tensor_metadata: TensorMetadata = dacite.from_dict(
            data_class=TensorMetadata,
            data=dictionary,
            config=cls._metadata_config,
        )
        return tensor_metadata


def process_type(elt: Any, type_hooks: Dict[Type, Callable[[Any], Any]]):
    # [FUNCTION DOC]
    # HELPER: recursive converter used before JSON dumping.
    # Behavior: dictionaries go through to_dict, objects whose exact class has a hook are converted by the
    # hook, lists/tuples go through to_list, other values are returned unchanged.
    # Inputs: elt (any), type_hooks (dictionary from type to converter). Outputs: converted value.
    # Exceptions: None explicitly raised. Side effects: None.
    if isinstance(elt, dict):
        return to_dict(elt, type_hooks=type_hooks)
    elif elt.__class__ in type_hooks:
        return type_hooks[elt.__class__](elt)
    elif isinstance(elt, (list, tuple)):
        return to_list(elt, type_hooks=type_hooks)
    else:
        return elt


def to_dict(dict_: Dict, type_hooks: Dict[Type, Callable[[Any], Any]]):
    # [FUNCTION DOC]
    # HELPER: applies process_type to every value of a dictionary.
    # Inputs: dict_, type_hooks. Outputs: new dictionary. Exceptions: None. Side effects: None.
    result = {}
    for key, value in dict_.items():
        result[key] = process_type(value, type_hooks=type_hooks)
    return result


def to_list(list_: Union[List, Tuple], type_hooks: Dict[Type, Callable[[Any], Any]]):
    # [FUNCTION DOC]
    # HELPER: applies process_type to every element, preserving the container class (list or tuple).
    # Inputs: list_, type_hooks. Outputs: new list or tuple. Exceptions: None. Side effects: None.
    return list_.__class__((process_type(elt, type_hooks=type_hooks) for elt in list_))


def save_meta(parallel_context: ParallelContext, root_folder: Path, training_metadata: TrainingMetadata):
    # [FUNCTION DOC]
    # Behavior: asserts the metadata type; only world rank 0 continues: creates root_folder, builds
    # CheckpointMetadata with current version and tp/dp/cp sizes, converts Version objects to strings,
    # and writes sorted, indented JSON to the checkpoint metadata file.
    # Inputs: parallel_context, root_folder (Path), training_metadata (TrainingMetadata).
    # Outputs: None. Exceptions: AssertionError on wrong type.
    # Side effects: directory creation and file write on world rank 0.
    assert isinstance(training_metadata, TrainingMetadata)

    if dist.get_rank(parallel_context.world_pg) != 0:
        return

    root_folder.mkdir(exist_ok=True, parents=True)
    checkpoint_metadata = CheckpointMetadata(
        version=CHECKPOINT_VERSION,
        tp=parallel_context.tp_pg.size(),
        dp=parallel_context.dp_pg.size(),
        cp=parallel_context.cp_pg.size(),
        metas=training_metadata,
    )

    # There are some types that require manual casting in order to work correctly.
    processed_metadata = process_type(dataclasses.asdict(checkpoint_metadata), type_hooks={Version: lambda x: str(x)})

    with open(root_folder / CHECKPOINT_FILE_NAME, mode="w") as fo:
        json.dump(processed_metadata, fo, indent=2, sort_keys=True)


def load_meta(parallel_context: ParallelContext, root_folder: Path) -> CheckpointMetadata:
    # [FUNCTION DOC]
    # Behavior: reads the metadata JSON, builds CheckpointMetadata via dacite (casting Version), and asserts
    # its version is not newer than the current checkpoint version.
    # Inputs: parallel_context (unused), root_folder (Path). Outputs: CheckpointMetadata.
    # Exceptions: AssertionError for a newer checkpoint; I/O or dacite errors.
    # Side effects: file read.
    with open(root_folder / CHECKPOINT_FILE_NAME, mode="r") as fi:
        checkpoint_metadata = json.load(fi)
        checkpoint_metadata = from_dict(
            data_class=CheckpointMetadata,
            data=checkpoint_metadata,
            config=dacite.Config(
                cast=[Version],
            ),
        )
        # Assume that we're always backward compatible, we only increment CHECKPOINT_VERSION when there's a breaking change.
        assert (
            checkpoint_metadata.version <= CHECKPOINT_VERSION
        ), f"Checkpoint is of version {checkpoint_metadata.version}, Current `nanotron` checkpoint version is {CHECKPOINT_VERSION}"
    return checkpoint_metadata
