# [MODULE DOC]
# Purpose: Hugging Face datasets based preprocessing for CLM training: loading (optionally mixed) datasets and
# tokenising/grouping text into fixed-length chunks.
# Main contents: clm_process, get_datasets, _get_dataset_mix.
# Fits in nanotron: run_train uses get_datasets then clm_process to produce the datasets.Dataset consumed by
# get_train_dataloader.
# Import-time side effects: imports datasets lazily guarded by try/except; emits a warning if it is not installed
# (the functions then fail with NameError when called).
# KNOWN ISSUE: a comment says SFT processing functions are imported for backward compatibility, but no import follows.
import warnings
from typing import Dict, List, Optional, Union

import numpy as np

try:
    from datasets import (
        Dataset,
        DatasetDict,
        Features,
        Sequence,
        Value,
        concatenate_datasets,
        load_dataset,
    )
except ImportError:
    warnings.warn("Datasets not installed, you'll be unable to use these dataset processing functions.")

# Import SFT processing functions for backward compatibility


def clm_process(
    raw_dataset: "Dataset",
    tokenizer,
    text_column_name: str,
    dataset_processing_num_proc_per_process: int,
    dataset_overwrite_cache: bool,
    sequence_length: int,
):
    # [FUNCTION DOC]
    # Summary: tokenises a text dataset and regroups tokens into chunks of sequence_length + 1 tokens overlapping by one.
    # Behavior: batched map over text_column_name: tokenise with batch_encode_plus (no attention mask or token type ids),
    # convert to numpy arrays, then group_texts concatenates the batch and slices chunks. Original columns are removed;
    # the output feature is a fixed-length int64 sequence "input_ids".
    # Inputs: raw_dataset (datasets.Dataset), tokenizer (Hugging Face tokenizer), text_column_name (string),
    # dataset_processing_num_proc_per_process (int processes for map), dataset_overwrite_cache (bool; when True the map
    # cache is not reused), sequence_length (int).
    # Outputs: datasets.Dataset with an "input_ids" column.
    # Exceptions: None explicitly raised.
    # Side effects: spawns map worker processes, reads/writes the datasets cache, progress output.
    """
    Concatenate all texts from raw_dataset and generate chunks of `sequence_length + 1`,
    where chunks overlap by a single token.

    Args:
        raw_dataset: Dataset containing raw text
        tokenizer: HuggingFace tokenizer
        text_column_name: Name of the column containing text data
        dataset_processing_num_proc_per_process: Number of processes for parallelization
        dataset_overwrite_cache: Whether to overwrite the cache
        sequence_length: Maximum sequence length

    Returns:
        Processed dataset with tokenized sequences
    """
    # Adapted from https://github.com/huggingface/transformers/blob/47e1676255e5dd86b9541f734cd4f4bdcbb50f4a/examples/pytorch/language-modeling/run_clm.py#L391-L439

    def group_texts(examples: Dict[str, List[np.ndarray]]) -> Dict[str, List[np.ndarray]]:
        # [FUNCTION DOC]
        # HELPER: concatenates each column of a tokenised batch and cuts it into overlapping windows.
        # Behavior: if the total length is at least sequence_length + 1, trim it to a multiple of sequence_length plus 1;
        # windows of sequence_length + 1 tokens start every sequence_length tokens; the remainder is dropped.
        # Inputs: examples (dictionary of column name to list of arrays).
        # Outputs: dictionary of column name to list of arrays of length sequence_length + 1.
        # Exceptions: None explicitly raised. Side effects: None.
        # KNOWN ISSUE: the range end excludes the last valid window start (total_length minus sequence_length minus 1), so one
        # full chunk is dropped per batch, and a batch with exactly sequence_length + 1 tokens yields nothing.
        # Concatenate all texts.
        concatenated_examples = {k: np.concatenate(v) for k, v in examples.items()}
        total_length = len(concatenated_examples[next(iter(examples.keys()))])
        # WARNING: We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
        if total_length >= sequence_length + 1:
            total_length = ((total_length - 1) // sequence_length) * sequence_length + 1
        # Split by chunks of sequence_length.
        result = {
            k: [
                t[i : i + sequence_length + 1] for i in range(0, total_length - (sequence_length + 1), sequence_length)
            ]
            for k, t in concatenated_examples.items()
        }
        return result

    def _tokenize_and_group_texts(texts: List[str]) -> Dict[str, List[np.ndarray]]:
        # [FUNCTION DOC]
        # HELPER: tokenises a list of strings and passes the numpy-converted result to group_texts.
        # Inputs: texts (list of strings). Outputs: dictionary as returned by group_texts.
        # Exceptions: None explicitly raised. Side effects: None.
        tokenized_batch = tokenizer.batch_encode_plus(texts, return_attention_mask=False, return_token_type_ids=False)
        tokenized_batch = {k: [np.array(tokenized_texts) for tokenized_texts in v] for k, v in tokenized_batch.items()}
        return group_texts(tokenized_batch)

    train_dataset = raw_dataset.map(
        _tokenize_and_group_texts,
        input_columns=text_column_name,
        remove_columns=raw_dataset.column_names,
        features=Features({"input_ids": Sequence(feature=Value(dtype="int64"), length=sequence_length + 1)}),
        batched=True,
        num_proc=dataset_processing_num_proc_per_process,
        load_from_cache_file=not dataset_overwrite_cache,
        desc=f"Grouping texts in chunks of {sequence_length+1}",
    )
    return train_dataset


def get_datasets(
    hf_dataset_or_datasets: Union[dict, str],
    hf_dataset_config_name: str,
    splits: Optional[Union[List[str], str]] = ["train", "test"],
) -> "DatasetDict":
    # [FUNCTION DOC]
    # Summary: loads datasets from a name or a mixing dictionary into a DatasetDict.
    # Behavior: a single split string is wrapped into a list. A dictionary is delegated to _get_dataset_mix; a string loads
    # each requested split with load_dataset and the given config name.
    # Inputs: hf_dataset_or_datasets (dataset name, or dictionary name to fraction), hf_dataset_config_name (string or
    # None), splits (list of strings or a string, default train and test).
    # Outputs: DatasetDict keyed by split.
    # Exceptions: ValueError for other input types; errors from load_dataset.
    # Side effects: network/disk access through datasets.
    # KNOWN ISSUE: with the dictionary form hf_dataset_config_name is ignored.
    """
    Function to load dataset directly from arguments.

    Args:
        hf_dataset_or_datasets: Dict or string defining datasets to load. When using a dict
                                with probabilities all equal to 1, datasets are concatenated
                                instead of sampled.
        hf_dataset_config_name: Configuration name for the dataset
        splits: Section of the dataset to load, defaults to ["train", "test"]
            Can be a single string or a list of splits to load.

    Returns:
        DatasetDict: DatasetDict object containing the loaded datasets
    """
    if isinstance(splits, str):
        splits = [splits]

    if isinstance(hf_dataset_or_datasets, dict):
        # Structure of the config to read the datasets and their mix
        # datasets_mixer:
        #     - 'dataset1': 0.5
        #     - 'dataset2': 0.3
        #     - 'dataset3': 0.2
        raw_datasets = _get_dataset_mix(hf_dataset_or_datasets, splits=splits)
    elif isinstance(hf_dataset_or_datasets, str):
        # e.g. Dataset = "HuggingFaceH4/testing_alpaca_small"
        # Note this returns things other than just train/test, which may not be intended
        raw_datasets = DatasetDict()
        for split in splits:
            raw_datasets[split] = load_dataset(
                hf_dataset_or_datasets,
                hf_dataset_config_name,
                split=split,
            )
    else:
        raise ValueError(f"hf_dataset_or_datasets must be a dict or string but is {type(hf_dataset_or_datasets)}")

    return raw_datasets


def _get_dataset_mix(dataset_dict: dict, splits: List[str] = None, seed=42) -> "DatasetDict":
    # [FUNCTION DOC]
    # HELPER: loads and mixes several datasets for get_datasets.
    # Behavior: for each dataset name and fraction (negative fractions rejected), load each split; splits whose name
    # contains "train" go to the train list, those containing "test" to the test list, others raise. The train set is the
    # concatenation of the first fraction times length rows of each train dataset, shuffled with seed; the test set is
    # the full concatenation of test datasets, shuffled.
    # Inputs: dataset_dict (dictionary name to fraction), splits (list of strings, default None), seed (default 42).
    # Outputs: DatasetDict with "train" and/or "test".
    # Exceptions: ValueError for negative fraction, unknown split, or nothing loaded.
    # Side effects: network/disk access.
    # KNOWN ISSUE: fractions are zipped with the train dataset list, which misaligns when a dataset has more than one train
    # split; splits default None makes iteration fail with TypeError.
    """
    Helper function to load dataset mix from dict configuration.

    Args:
        dataset_dict: Dictionary containing the dataset names and their training proportions.
                     By default, all test proportions are 1.
        splits: Section of the dataset to load, defaults to ["train", "test"]
        seed: Random seed for shuffling datasets

    Returns:
        DatasetDict containing the mixed datasets
    """
    raw_datasets = DatasetDict()
    raw_train_datasets = []
    raw_test_datasets = []
    fracs = []
    for ds, frac in dataset_dict.items():
        if frac < 0:
            raise ValueError(f"Dataset fraction for dataset {ds} is negative. (= {frac})")

        fracs.append(frac)
        for split in splits:
            if "train" in split:
                raw_train_datasets.append(
                    load_dataset(
                        ds,
                        split=split,
                    )
                )
            elif "test" in split:
                raw_test_datasets.append(
                    load_dataset(
                        ds,
                        split=split,
                    )
                )
            else:
                raise ValueError(f"Split type {split} not recognized as one of test or train.")

    if len(raw_train_datasets) > 0:
        train_subsets = []
        for dataset, frac in zip(raw_train_datasets, fracs):
            train_subset = dataset.select(range(int(frac * len(dataset))))
            train_subsets.append(train_subset)
        raw_datasets["train"] = concatenate_datasets(train_subsets).shuffle(seed=seed)

    # No subsampling for test datasets to enable fair comparison across models
    if len(raw_test_datasets) > 0:
        raw_datasets["test"] = concatenate_datasets(raw_test_datasets).shuffle(seed=seed)

    if len(raw_datasets) == 0:
        raise ValueError(
            f"Dataset {dataset_dict} not recognized with split {splits}. Check the dataset has been correctly formatted."
        )

    return raw_datasets
