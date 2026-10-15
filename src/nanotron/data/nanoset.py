# [MODULE DOC]
# Purpose: Nanoset, a torch Dataset that blends several datatrove tokenised folders by weight into a fixed number of
# training samples, plus the numba helper that builds the blending index.
# Main contents: Nanoset, build_nanoset_index_helper.
# Fits in nanotron: the dataset used for pre-tokenised training data; wrapped by build_nanoset_dataloader.
# Import-time side effects: creates a module logger; numba compiles build_nanoset_index_helper lazily with caching.
import hashlib
import json
import os
import warnings
from typing import Dict, List, Tuple, Union

import numpy as np
import torch
from datatrove.utils.dataset import DatatroveFolderDataset
from numba import jit
from tqdm import tqdm

from nanotron import logging
from nanotron.data.utils import count_dataset_indexes, normalize
from nanotron.logging import log_rank

logger = logging.get_logger(__name__)


class Nanoset(torch.utils.data.Dataset):
    # [CLASS DOC]
    # Purpose: weighted blend of DatatroveFolderDataset instances indexed by a global sample index.
    # Key attributes: dataset_folders, sequence_length, eos_token_id, return_positions, token_size (bytes per token in the
    # .ds files), train_split_num_samples (total samples to expose), random_seed, use_cache, cache_dir
    # ("./.nanoset_cache"), datatrove_datasets (one per folder), dataset_lengths, dataset_weights (normalised array),
    # dataset_index and dataset_sample_index (arrays of length train_split_num_samples giving which dataset and which
    # sample of it each global index maps to).
    # Inherits from torch.utils.data.Dataset.
    # KNOWN ISSUE: the docstring lists a vocab_size argument; the code takes token_size, random_seed, use_cache,
    # eos_token_id and return_positions instead.
    """
    The Nanoset dataset

    Args:
        dataset_folders (List[str]): List of folders with tokenized datasets
        dataset_weights (Union[List[float], None]): List with the weights for weighted datasets. If None, consume all samples from all datasets without weighting. Weights are normalized in __init__
        sequence_length (int): Sequence length of the built samples
        vocab_size (int): Vocab size of the tokenizer used to tokenize the dataset
        train_split_num_samples (int): Number of samples the dataset needs. It's the training steps * global batch size
    """

    def __init__(
        self,
        dataset_folders: List[str],
        sequence_length: int,
        token_size: int,
        train_split_num_samples: int,
        dataset_weights: Union[List[float], None] = None,
        random_seed: int = 1234,
        use_cache: bool = True,
        eos_token_id: int = None,
        return_positions: bool = True,
    ) -> None:
        # [FUNCTION DOC]
        # Summary: opens each folder as a datatrove dataset and builds the blending index.
        # Behavior: a single string for dataset_folders is wrapped into a list with a warning. Store parameters, assert
        # return_positions or eos_token_id is set. Create one DatatroveFolderDataset per folder matching "*.ds" (non
        # recursive, shuffle on, sequence length seq_len, returning positions if requested). Record their lengths; weights
        # default to the lengths normalised (consume everything proportionally) else the given weights normalised; assert one
        # weight per folder. Build the index with build_nanoset_index and log a summary.
        # Inputs: dataset_folders (list of paths), sequence_length (int), token_size (int, 2 or 4), train_split_num_samples
        # (int), dataset_weights (optional list of floats), random_seed (default 1234), use_cache (default True, only used by
        # new_build_nanoset_index), eos_token_id (optional int), return_positions (default True).
        # Outputs: None.
        # Exceptions: AssertionError for the checks above.
        # Side effects: warning, file system reads by datatrove, logging on rank 0.
        # KNOWN ISSUE: the assertion message says eos_token_id is needed when return_positions is True, but the check
        # actually requires eos_token_id only when return_positions is False.
        # Checks

        if isinstance(dataset_folders, str):
            warnings.warn("dataset_folders should be of type List[str] but str was provided. Converting to List[str]")
            dataset_folders = [dataset_folders]

        # Init
        self.dataset_folders = dataset_folders
        self.sequence_length = sequence_length
        self.eos_token_id = eos_token_id
        self.return_positions = return_positions
        assert (
            self.return_positions or self.eos_token_id is not None
        ), "If return_positions is True, eos_token_id must be defined"
        # Number of bytes for the tokens stored in the processed dataset files. 2 for vocab sizes < 65535, 4 otherwise
        self.token_size = token_size
        self.train_split_num_samples = train_split_num_samples
        self.random_seed = random_seed
        self.use_cache = use_cache
        self.cache_dir = "./.nanoset_cache"
        self.datatrove_datasets = []
        for dataset_folder in self.dataset_folders:
            self.datatrove_datasets.append(
                DatatroveFolderDataset(
                    folder_path=dataset_folder,
                    filename_pattern=os.path.join(dataset_folder, "*.ds"),
                    seq_len=sequence_length,
                    recursive=False,
                    token_size=self.token_size,
                    shuffle=True,
                    return_positions=self.return_positions,  # if set to True, the position ids are directly build datatrove
                    eos_token_id=self.eos_token_id,
                )
            )

        # Build Nanoset Index
        ## To build the index we need the length of each dataset
        self.dataset_lengths = [len(datatrove_dataset) for datatrove_dataset in self.datatrove_datasets]
        ## Set dataset weights
        if (
            dataset_weights is None
        ):  # Case of training with > 1 datasets without weighting them: Consume both datasets entirely on each epoch
            self.dataset_weights = normalize(self.dataset_lengths)
        else:
            self.dataset_weights = normalize(dataset_weights)
        assert len(dataset_folders) == len(
            self.dataset_weights
        ), f"Specified {len(self.dataset_weights)} weights but {len(dataset_folders)} datasets were provided."
        ## Build dataset index and dataset sample index
        self.dataset_index, self.dataset_sample_index = self.build_nanoset_index()
        # self.dataset_index, self.dataset_sample_index = self.new_build_nanoset_index() # TODO: Fix this

        self.print_nanoset_info()

    def __len__(self) -> int:
        # [FUNCTION DOC]
        # Summary: number of samples, the length of dataset_index (equals train_split_num_samples).
        # Inputs: None. Outputs: int. Exceptions: None explicitly raised. Side effects: None.
        """
        Returns:
            int: The number of samples of the Nanoset
        """

        return len(self.dataset_index)

    def __getitem__(self, idx: int) -> Dict[str, np.ndarray]:
        # [FUNCTION DOC]
        # Summary: looks up which dataset and local sample a global index maps to, wraps the local index modulo that
        # dataset's length, and returns the datatrove sample (sequence_length + 1 tokens, plus positions if enabled).
        # Inputs: idx, int global sample index.
        # Outputs: dictionary with "input_ids" (and "positions") arrays/tensors as produced by datatrove.
        # Exceptions: IndexError for an out-of-range idx.
        # Side effects: reads from memory-mapped files.
        """
        Returns sequence_length + 1 tokens from the memmap dataset using sequential access
        """
        dataset = self.dataset_index[idx]
        sample_idx = self.dataset_sample_index[idx]

        # Get actual sample index by wrapping around dataset length
        actual_sample = sample_idx % self.dataset_lengths[dataset]

        return self.datatrove_datasets[dataset][actual_sample]

    def new_build_nanoset_index(self) -> Tuple[np.ndarray, np.ndarray]:
        # [FUNCTION DOC]
        # Summary: alternative (currently unused) index builder with on-disk cache allowing sequential reads per dataset.
        # Behavior: hash the index-affecting parameters (folders, lengths, weights, sample count, seed, token size, sequence
        # length) with md5 into a cache file name under cache_dir; if it exists try to load it. Otherwise give each dataset
        # floor(weight times total) samples, distribute the remainder to datasets with the largest fractional parts, repeat
        # dataset ids by those counts, shuffle with the seed, and assign each slot the next sequential sample of its dataset.
        # Try to save the result as an npz cache.
        # Inputs: None (uses self).
        # Outputs: tuple (dataset_index, dataset_sample_index), int64 arrays of length train_split_num_samples.
        # Exceptions: cache load/save failures are caught and logged.
        # Side effects: file reads/writes in cache_dir, logging, tqdm progress output.
        # KNOWN ISSUE: use_cache is ignored; the cache is always consulted and written.
        """
        Build dataset index that enables sequential reading while respecting weights.
        Uses cache if available and parameters match.
        """
        # Create a cache key based on the parameters that affect the index
        cache_params = {
            "dataset_folders": self.dataset_folders,
            "dataset_lengths": self.dataset_lengths,
            "dataset_weights": self.dataset_weights.tolist(),
            "train_split_num_samples": self.train_split_num_samples,
            "random_seed": self.random_seed,
            "token_size": self.token_size,
            "sequence_length": self.sequence_length,
        }

        # Create a deterministic cache key
        cache_key = hashlib.md5(json.dumps(cache_params, sort_keys=True).encode()).hexdigest()
        cache_file = os.path.join(self.cache_dir, f"index_{cache_key}.npz")

        # Try to load from cache
        if os.path.exists(cache_file):
            try:
                logger.info(f"[Nanoset] Loading index from cache: {cache_file}")
                cached_data = np.load(cache_file)
                return cached_data["dataset_index"], cached_data["dataset_sample_index"]
            except Exception as e:
                logger.warning(f"[Nanoset] Failed to load cache, rebuilding index: {e}")

        logger.info(f"[Nanoset] Building sequential Nanoset index for {len(self.dataset_folders)} datasets")

        # Original index building logic
        total_weighted_samples = np.array(self.dataset_weights) * self.train_split_num_samples
        samples_per_dataset = np.floor(total_weighted_samples).astype(np.int64)

        remaining = self.train_split_num_samples - samples_per_dataset.sum()
        if remaining > 0:
            fractional_parts = total_weighted_samples - samples_per_dataset
            indices = np.argsort(fractional_parts)[-remaining:]
            samples_per_dataset[indices] += 1

        dataset_positions = np.zeros(len(self.dataset_folders), dtype=np.int64)
        dataset_index = np.zeros(self.train_split_num_samples, dtype=np.int64)
        dataset_sample_index = np.zeros(self.train_split_num_samples, dtype=np.int64)

        dataset_order = np.repeat(np.arange(len(self.dataset_folders)), samples_per_dataset)
        rng = np.random.RandomState(self.random_seed)
        rng.shuffle(dataset_order)

        for idx, dataset_idx in tqdm(enumerate(dataset_order), desc="Building Nanoset index"):
            dataset_index[idx] = dataset_idx
            dataset_sample_index[idx] = dataset_positions[dataset_idx]
            dataset_positions[
                dataset_idx
            ] += 1  # Read samples sequentially from each datatrove_dataset assuming they're already shuffled

        # Save to cache
        try:
            os.makedirs(self.cache_dir, exist_ok=True)
            np.savez(cache_file, dataset_index=dataset_index, dataset_sample_index=dataset_sample_index)
            logger.info(f"[Nanoset] Saved index to cache: {cache_file}")
        except Exception as e:
            logger.warning(f"[Nanoset] Failed to save cache: {e}")

        return dataset_index, dataset_sample_index

    def build_nanoset_index(self) -> np.ndarray:
        # [FUNCTION DOC]
        # Summary: builds the index actually used: one weighted epoch, shuffled, repeated to cover all training samples.
        # Behavior: samples_per_epoch is the sum of dataset lengths; num_epochs is the integer part of
        # train_split_num_samples over that, plus one. Call build_nanoset_index_helper for one epoch, shuffle both arrays with
        # two fresh RandomState(random_seed) so they get the same permutation, concatenate num_epochs copies and truncate to
        # train_split_num_samples.
        # Inputs: None (uses self).
        # Outputs: tuple of two arrays (dataset ids, per-dataset sample ids).
        # Exceptions: None explicitly raised.
        # Side effects: None.
        # KNOWN ISSUE: the return annotation says a single ndarray but a tuple of two arrays is returned.
        """
        Build dataset index and dataset sample index
        """
        # Compute samples per epoch and number of epochs
        samples_per_epoch = sum(self.dataset_lengths)
        num_epochs = int(self.train_split_num_samples / samples_per_epoch) + 1
        # Build the dataset indexes for 1 epoch
        dataset_index, dataset_sample_index = build_nanoset_index_helper(
            n_samples=samples_per_epoch, weights=self.dataset_weights, dataset_sizes=self.dataset_lengths
        )
        # Shuffle the indexes the same way
        numpy_random_state = np.random.RandomState(self.random_seed)
        numpy_random_state.shuffle(dataset_index)
        numpy_random_state = np.random.RandomState(self.random_seed)
        numpy_random_state.shuffle(dataset_sample_index)
        # Concatenate num_epochs the shuffled indexes
        dataset_index = np.concatenate([dataset_index for _ in range(num_epochs)])
        dataset_sample_index = np.concatenate([dataset_sample_index for _ in range(num_epochs)])
        # Just keep the necessary samples
        dataset_index = dataset_index[: self.train_split_num_samples]
        dataset_sample_index = dataset_sample_index[: self.train_split_num_samples]
        return dataset_index, dataset_sample_index

    def print_nanoset_info(self):
        # [FUNCTION DOC]
        # HELPER: logs dataset statistics on global rank 0.
        # Behavior: log the total samples and tokens (samples times sequence_length), then for each folder the number of
        # samples drawn from it and its share.
        # Inputs: None. Outputs: None. Exceptions: None explicitly raised.
        # Side effects: logging.

        log_rank(f"> Total number of samples: {len(self)}", logger=logger, level=logging.INFO, rank=0)
        log_rank(
            f"> Total number of tokens: {len(self) * self.sequence_length}", logger=logger, level=logging.INFO, rank=0
        )

        # Print samples from each dataset + weight
        dataset_sample_count = count_dataset_indexes(self.dataset_index, len(self.dataset_folders))
        for index, sample_count in enumerate(dataset_sample_count):
            log_rank(
                f">   Total number of samples from the {self.dataset_folders[index]} dataset: {sample_count} ({round(normalize(dataset_sample_count).tolist()[index], 2)})",
                logger=logger,
                level=logging.INFO,
                rank=0,
            )


@jit(nopython=True, cache=True)
def build_nanoset_index_helper(
    n_samples: int, weights: np.ndarray, dataset_sizes: List[int]
) -> Tuple[np.ndarray, np.ndarray]:
    # [FUNCTION DOC]
    # HELPER: numba-jitted greedy blender used by Nanoset.build_nanoset_index.
    # Behavior: for each output position, compute for every dataset the error between its target count (weight times
    # the current position, using at least 1) and the samples already taken; pick the dataset with the largest error,
    # record it and its next local sample id (modulo its size), and increment its counter.
    # Inputs: n_samples (int), weights (normalised float array), dataset_sizes (list of ints).
    # Outputs: tuple (dataset_index as unsigned ints, dataset_sample_index as longs), both of length n_samples.
    # Exceptions: None explicitly raised.
    # Side effects: None.
    """
    Given multiple datasets and a weighting array, build samples indexes
    such that it follows those weights
    """
    # Create empty arrays for dataset indices and dataset sample indices
    dataset_index = np.empty((n_samples,), dtype="uint")
    dataset_sample_index = np.empty((n_samples,), dtype="long")  # Supports dataset with up to 2**64 samples

    # Initialize buffer for number of samples used for each dataset
    current_samples = np.zeros((len(weights),), dtype="long")

    # TODO: Add 0.5% (the 1.005 factor) so in case the bleding dataset does
    # not uniformly distribute the number of samples, we still have
    # samples left to feed to the network

    # Iterate over all samples
    for sample_idx in range(n_samples):
        # Convert sample index to float for comparison against weights
        sample_idx_float = max(sample_idx, 1.0)

        # Find the dataset with the highest error
        errors = weights * sample_idx_float - current_samples
        max_error_index = np.argmax(errors)

        # Assign the dataset index and update the sample index
        dataset_index[sample_idx] = max_error_index
        dataset_sample_index[sample_idx] = current_samples[max_error_index] % dataset_sizes[max_error_index]

        # Update the total samples for the selected dataset
        current_samples[max_error_index] += 1

    return dataset_index, dataset_sample_index
