# [MODULE DOC]
# Purpose: datasets that interleave samples from several datasets according to weights.
# Main contents: BlendedSubsetSplitLog (log record), BlendableDataset (index built by the compiled C++ helper, with
# per-dataset token consumption tracking and resume offsets), MemoryEfficientBlendableDataset (index computed on the
# fly from weight bins).
# Fits in nanotron: used by nanotron.data.nemo_dataset to blend GPTDatasets for multi-corpus training.
# Import-time side effects: creates a module logger.
# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Blendable dataset."""

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, List, Optional

import numpy as np
import torch

from nanotron import logging
from nanotron.logging import log_rank
from nanotron.parallel import ParallelContext
from nanotron.utils import main_rank_first
from pprint import pformat

if TYPE_CHECKING:
    from . import GPTDataset, SubsetSplitLog

logger = logging.get_logger(__name__)


@dataclass
class BlendedSubsetSplitLog:
    # [CLASS DOC]
    # Purpose: dataclass describing a blend: blended_total_num_samples (int), blended_per_subset_samples (list of samples
    # drawn from each dataset), blended_subset (list of each child dataset's SubsetSplitLog).
    blended_total_num_samples: int
    blended_per_subset_samples: List[int]
    blended_subset: List["SubsetSplitLog"]


class BlendableDataset(torch.utils.data.Dataset):
    # [CLASS DOC]
    # Purpose: torch Dataset mapping each global index to (dataset, sample) by a precomputed weighted interleave.
    # Key attributes: datasets, size, dataset_index (int16 per global index), dataset_sample_index (int64, sequential per
    # dataset), dataset_num_samples (per-dataset counts), random_seed, subset_log, consumed_tokens (dictionary from
    # dataset position to tokens consumed in the current stage), sequence_length (set on first metrics update),
    # offsets_in_samples (dictionary from dataset position to a sample offset carried over from earlier stages).
    # Inherits from torch.utils.data.Dataset.
    def __init__(
        self,
        datasets: List["GPTDataset"],
        weights: List[float],
        size: int,
        parallel_context: ParallelContext,
        seed: int,
        consumed_tokens_per_dataset_folder: Optional[Dict[str, int]] = None,
        offsets_in_samples: Optional[Dict[str, int]] = None,
    ):
        # [FUNCTION DOC]
        # Summary: normalises weights, builds the blending index, and initialises consumption tracking.
        # Behavior: assert one weight per dataset, positive total weight and fewer than 32767 datasets; allocate index arrays.
        # Inside main_rank_first over the world group import the C++ helpers module (compiling it via compile_helper if the
        # import fails). Call helpers.build_blending_indices (verbose on global rank 0) to fill dataset_index,
        # dataset_sample_index and dataset_num_samples. Log elapsed time, build the subset log. Initialise consumed_tokens to
        # zero and override entries whose dataset folder_path matches a key of consumed_tokens_per_dataset_folder; likewise
        # set offsets_in_samples from offsets_in_samples.
        # Inputs: datasets (list of datasets with subset_log, and folder_path when tracking is used), weights (list of
        # floats), size (total samples), parallel_context, seed (int, stored only), consumed_tokens_per_dataset_folder
        # (optional dictionary path to tokens), offsets_in_samples (optional dictionary path to sample offset).
        # Outputs: None.
        # Exceptions: AssertionError for invalid weights/count; ImportError if helpers cannot be built.
        # Side effects: may compile C++ code; ordering across ranks via main_rank_first; logging.
        # KNOWN ISSUE: seed is a required parameter but the callers in nanotron.data.nemo_dataset do not pass it, so those
        # calls raise TypeError.
        # KNOWN ISSUE: GPTDataset has no folder_path attribute, so passing either tracking dictionary with GPTDatasets fails.
        self.datasets = datasets
        num_datasets = len(datasets)
        assert num_datasets == len(weights)

        self.size = size

        # Normalize weights.
        weights = np.array(weights, dtype=np.float64)
        sum_weights = np.sum(weights)
        assert sum_weights > 0.0
        weights /= sum_weights

        # Build indices.
        start_time = time.time()
        # from https://github.com/NVIDIA/Megatron-LM/commit/c6e65b2e96e8376ccc84225dd1a9b60dd242fc48
        assert num_datasets < 32767
        self.dataset_index = np.zeros(self.size, dtype=np.int16)
        self.dataset_sample_index = np.zeros(self.size, dtype=np.int64)
        self.dataset_num_samples = np.zeros(num_datasets, dtype=np.int64)
        self.random_seed = seed

        with main_rank_first(parallel_context.world_pg):
            try:
                from . import helpers
            except ImportError:
                try:
                    from .dataset_utils import compile_helper

                    compile_helper()
                    from . import helpers
                except ImportError:
                    raise ImportError(
                        "Could not compile megatron dataset C++ helper functions and therefore cannot import helpers python file."
                    )

        helpers.build_blending_indices(
            self.dataset_index,
            self.dataset_sample_index,  # sequential for each dataset_source
            self.dataset_num_samples,
            weights,
            num_datasets,
            self.size,
            torch.distributed.get_rank() == 0,
        )

        log_rank(
            "> elapsed time for building blendable dataset indices: " "{:.2f} (sec)".format(time.time() - start_time),
            logger=logger,
            level=logging.INFO,
            rank=0,
        )

        self.subset_log = BlendedSubsetSplitLog(
            blended_total_num_samples=self.size,
            blended_per_subset_samples=self.dataset_num_samples.tolist(),
            blended_subset=[d.subset_log for d in self.datasets],
        )

        # Track last 16 items
        # self.history_size = 16
        # self.last_dataset_idx = np.full(self.history_size, -1, dtype=np.int16)  # -1 indicates no data
        # self.last_dataset_sample_idx = np.full(self.history_size, -1, dtype=np.int64)
        # self.last_item_idx = np.full(self.history_size, -1, dtype=np.int64)

        # Initialize consumption tracking
        self.consumed_tokens = {idx: 0 for idx in range(len(datasets))} # current stage's consumed_tokens_per_dataset_folder
        if consumed_tokens_per_dataset_folder is not None:
            # find idx of dataset that matches the folder path
            for idx, dataset in enumerate(datasets):
                for folder_path, consumed_tokens in consumed_tokens_per_dataset_folder.items():
                    if dataset.folder_path == folder_path:
                        self.consumed_tokens[idx] = consumed_tokens
                        log_rank(f"[BlendableDataset] Setting consumed_tokens for dataset {idx} ({dataset.folder_path}) to {consumed_tokens}", logger=logger, level=logging.INFO, rank=0)
        
        self.sequence_length = None  # Will be set when first batch is processed

        # Setup offsets for already consumed tokens from previous stages
        self.offsets_in_samples = {idx: 0 for idx in range(len(datasets))} # last stage's consumed_tokens_per_dataset_folder
        if offsets_in_samples is not None:
            for idx, dataset in enumerate(datasets):
                for folder_path, offset in offsets_in_samples.items():
                    if dataset.folder_path == folder_path:
                        self.offsets_in_samples[idx] = offset
                        log_rank(f"[BlendableDataset] Applying offset {offset} samples to dataset {idx} ({dataset.folder_path})", logger=logger, level=logging.INFO, rank=0)

    def __len__(self):
        # [FUNCTION DOC]
        # Summary: returns size. Inputs: None. Outputs: int. Exceptions: None explicitly raised. Side effects: None.
        return self.size

    def __getitem__(self, idx):
        # [FUNCTION DOC]
        # Summary: looks up the dataset and its sequential sample id for idx, adds that dataset's offset_in_samples, and
        # returns the child dataset's item.
        # Inputs: idx (int). Outputs: whatever the child dataset returns.
        # Exceptions: IndexError for out-of-range indices.
        # Side effects: None.
        dataset_idx = self.dataset_index[idx]
        sample_idx = self.dataset_sample_index[idx]

        return self.datasets[dataset_idx][sample_idx + self.offsets_in_samples[dataset_idx]] # TODO: is it okay to not respect dataset_sample_index? Since it's sequential it's okay for now

    # @property
    # def last_file_idx(self):
    #     return None if self.last_dataset_idx[-1] == -1 else self.datasets[self.last_dataset_idx[-1]].current_file

    # @property
    # def last_file_path(self):
    #     return None if self.last_dataset_idx[-1] == -1 else self.datasets[self.last_dataset_idx[-1]].current_file_path

    # @property
    # def last_dataset_path(self):
    #     return None if self.last_dataset_idx[-1] == -1 else self.datasets[self.last_dataset_idx[-1]].folder_path

    def update_consumption_metrics(self, start_idx: int, end_idx: int, sequence_length: int):
        # [FUNCTION DOC]
        # Summary: adds tokens consumed by a global batch to consumed_tokens.
        # Behavior: remember sequence_length on first call; count how many indices in dataset_index between start_idx
        # (inclusive) and end_idx (exclusive) belong to each dataset; add count times sequence_length to each.
        # Inputs: start_idx, end_idx (global sample indices covering all DP ranks), sequence_length (int).
        # Outputs: None.
        # Exceptions: None explicitly raised.
        # Side effects: mutates consumed_tokens and sequence_length.
        # KNOWN ISSUE: dictionary keys come from numpy int16 values; they hash equal to the int keys so this works, but
        # sequence_length is stored only once and later different values are still used for counting.
        """Update consumed samples/tokens for the current batch.

        Args:
            start_idx: Starting index of current batch for all dp ranks
            end_idx: Ending index of current batch for all dp ranks
            sequence_length: Sequence length for token calculation
        """
        if self.sequence_length is None:
            self.sequence_length = sequence_length

        # Get dataset indices for current batch
        batch_indices = self.dataset_index[start_idx:end_idx]
        unique_indices, counts = np.unique(batch_indices, return_counts=True)

        # Update consumption dictionaries
        for dataset_idx, count in zip(unique_indices, counts):
            self.consumed_tokens[dataset_idx] += int(count * sequence_length)

    def get_consumption_stats(self):
        # [FUNCTION DOC]
        # Summary: returns a dictionary from each dataset's folder_path to a dictionary with its consumed "tokens".
        # Inputs: None. Outputs: dictionary.
        # Exceptions: AssertionError if a folder_path does not contain "s3".
        # Side effects: None.
        """Get current consumption statistics for all datasets.

        Returns:
            dict: Dictionary containing samples and tokens consumed per dataset
        """
        stats = {}
        for dataset_idx, dataset in enumerate(self.datasets):
            assert (
                "s3" in dataset.folder_path
            ), "Only S3 paths are supported for consumption stats"  # TODO: remove this
            stats[dataset.folder_path] = {"tokens": self.consumed_tokens[dataset_idx]}
        return stats


class MemoryEfficientBlendableDataset(torch.utils.data.Dataset):
    # [CLASS DOC]
    # Purpose: blend datasets without storing per-sample indices; global index idx uses bin idx modulo weight_bins, and
    # each bin is statically assigned to a dataset in proportion to weights.
    # Key attributes: datasets, size, weight_bins, weights (normalised), ds_index (dataset per bin), ds_index_size (bins
    # per dataset), ds_bias (rank of the bin within its dataset's bins), ds_size (length of each dataset).
    # Inherits from torch.utils.data.Dataset.
    """
    A BlendableDataset implementation that uses less memory than the original implementation.
    Indices are computed algorithmically instead of storing them in memory.

    To test call: MemoryEfficientBlendableDataset.test_index_blending()
    """

    def __init__(self, datasets, weights, size, weight_bins=100):
        # [FUNCTION DOC]
        # Summary: builds the bin tables.
        # Behavior: cap weight_bins at size; normalise weights (all must be positive); give dataset i the integer part of
        # weight times weight_bins consecutive bins with biases 0, 1, and so on; fill the remaining bins with the last dataset,
        # continuing its biases; assert every dataset got at least one bin; record dataset lengths.
        # Inputs: datasets (list), weights (list of floats), size (int), weight_bins (default 100).
        # Outputs: None.
        # Exceptions: AssertionError for non-positive weights or datasets with no bins.
        # Side effects: None.
        # KNOWN ISSUE: filler bins start their bias at the last existing bias instead of one past it, so the last dataset gets
        # a duplicated bias value.
        self.datasets = datasets
        num_datasets = len(datasets)
        assert num_datasets == len(weights)

        weight_bins = min(weight_bins, size)

        self.size = size
        self.weight_bins = weight_bins

        # Normalize weights.
        weights = np.array(weights, dtype=np.float64)
        assert (weights > 0.0).all()
        sum_weights = np.sum(weights)
        assert sum_weights > 0.0
        self.weights = weights / sum_weights

        # create ds index based on weights
        ds_index = []
        ds_bias = []
        for i, w in enumerate(self.weights):
            n = int(w * weight_bins)
            ds_index.extend([i] * n)
            ds_bias.extend(range(n))
        # make sure arrays have length of weight_bins
        n = weight_bins - len(ds_index)
        ds_index.extend([i] * n)
        ds_bias.extend(range(ds_bias[-1], ds_bias[-1] + n))

        self.ds_index = np.array(ds_index, dtype=np.uint32)
        self.ds_index_size = np.array([(self.ds_index == i).sum() for i in range(num_datasets)], dtype=np.uint32)
        assert (
            self.ds_index_size > 0
        ).all(), f"Some datasets have no samples in the blendable dataset, increase weight_bins or the offending weight. ds_index_size = {self.ds_index_size}"
        self.ds_bias = np.array(ds_bias, dtype=np.uint32)

        self.ds_size = np.array([len(ds) for ds in datasets], dtype=np.uint32)

    def get_ds_sample_idx(self, idx):
        # [FUNCTION DOC]
        # Summary: computes the dataset and local sample for a global idx: dataset is that of bin idx modulo weight_bins;
        # local sample is bin bias plus (idx divided by weight_bins) times the dataset's bin count, modulo the dataset length.
        # Inputs: idx (int). Outputs: tuple (dataset index, sample index).
        # Exceptions: None explicitly raised. Side effects: None.
        """Returns ds index and sample index (within the ds) for the given index in the blendable dataset."""

        bin = idx % self.weight_bins
        ds_idx = self.ds_index[bin]
        sample_idx = (self.ds_bias[bin] + (idx // self.weight_bins) * self.ds_index_size[ds_idx]) % self.ds_size[
            ds_idx
        ]

        return ds_idx, sample_idx

    def __len__(self):
        # [FUNCTION DOC]
        # Summary: returns size. Inputs: None. Outputs: int. Exceptions: None explicitly raised. Side effects: None.
        return self.size

    def __getitem__(self, idx):
        # [FUNCTION DOC]
        # Summary: returns the child item located by get_ds_sample_idx.
        # Inputs: idx. Outputs: child dataset item. Exceptions: None explicitly raised. Side effects: None.
        ds_idx, sample_idx = self.get_ds_sample_idx(idx)

        return self.datasets[ds_idx][sample_idx]

    @classmethod
    def test_index_blending(cls):
        # [FUNCTION DOC]
        # HELPER: manual visual test; builds three toy datasets with weights 0.5, 0.3 and 0.2, blends 50 samples with 10 and
        # 100 bins and plots dataset id and sample id per index with matplotlib in interactive mode.
        # Inputs: cls. Outputs: None.
        # Exceptions: ImportError if matplotlib is missing.
        # Side effects: opens matplotlib figures.
        """Visualize indices of blended dataset"""

        import matplotlib.pyplot as plt

        plt.ion()

        class DS(torch.utils.data.Dataset):
            # [CLASS DOC]
            # HELPER: toy dataset of a given size whose items are characters of a string, used by test_index_blending.
            # Attributes: size (int), data (indexable sequence).
            def __init__(self, size, data):
                # [FUNCTION DOC]
                # HELPER: stores size and data. Inputs: size (int), data (sequence). Outputs: None.
                # Exceptions: None explicitly raised. Side effects: sets attributes.
                self.size = size
                self.data = data

            def __len__(self):
                # [FUNCTION DOC]
                # HELPER: returns size. Inputs: None. Outputs: int. Exceptions: None explicitly raised. Side effects: None.
                return self.size

            def __getitem__(self, idx):
                # [FUNCTION DOC]
                # HELPER: returns the element of data at idx. Inputs: idx. Outputs: element.
                # Exceptions: IndexError if out of range. Side effects: None.
                return self.data[idx]

        for weight_bins in [10, 100]:
            blend_ds = MemoryEfficientBlendableDataset(
                [DS(10, "a"), DS(10, "b"), DS(10, "c")], [0.5, 0.3, 0.2], 50, weight_bins=weight_bins
            )

            ds_sample_idx_list = [blend_ds.get_ds_sample_idx(i) for i in range(50)]
            ds_list = list(zip(*ds_sample_idx_list))[0]
            sample_list = list(zip(*ds_sample_idx_list))[1]

            plt.figure()
            plt.plot(ds_list, label="ds idx")
            plt.plot(sample_list, label="sample")
            plt.legend()
            plt.grid()
            plt.title(f"weight_bins={weight_bins}")
