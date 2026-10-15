# [PACKAGE DOC]
# Purpose: public entry point of the nanotron.data package (dataset preparation, collation, samplers and dataloaders
# for training).
# Exports (re-exported from submodules): DataCollatorForCLM and DataCollatorForCLMWithPositionIds (clm_collator);
# dummy_infinite_data_generator, get_train_dataloader, sanity_check_dataloader, set_tensor_pointers (dataloader);
# clm_process, get_datasets (processing); EmptyInfiniteDataset, SkipBatchSampler, get_sampler (samplers);
# prepare_sft_dataset, process_sft (sft_processing).
# Other submodules not re-exported here: nanoset and dataloader_builder (Nanoset datasets), nemo_dataset
# (Megatron/NeMo style indexed datasets), tokenized_bytes, s3_utils.
# Import-time side effects: imports those submodules, which pull in torch, numpy, datasets and nanotron.distributed.
from nanotron.data.clm_collator import DataCollatorForCLM, DataCollatorForCLMWithPositionIds
from nanotron.data.dataloader import (
    dummy_infinite_data_generator,
    get_train_dataloader,
    sanity_check_dataloader,
    set_tensor_pointers,
)
from nanotron.data.processing import clm_process, get_datasets
from nanotron.data.samplers import EmptyInfiniteDataset, SkipBatchSampler, get_sampler
from nanotron.data.sft_processing import prepare_sft_dataset, process_sft

__all__ = [
    "DataCollatorForCLM",
    "DataCollatorForCLMWithPositionIds",
    "dummy_infinite_data_generator",
    "get_train_dataloader",
    "sanity_check_dataloader",
    "set_tensor_pointers",
    "clm_process",
    "get_datasets",
    "EmptyInfiniteDataset",
    "SkipBatchSampler",
    "get_sampler",
    "prepare_sft_dataset",
    "process_sft",
]
