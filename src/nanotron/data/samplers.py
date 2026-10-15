# [MODULE DOC]
# Purpose: samplers deciding which dataset indices each data-parallel rank reads, including resume support.
# Main contents: SkipBatchSampler, get_sampler (DistributedSampler-based, used by the nanotron dataloaders),
# EmptyInfiniteDataset (placeholder for pipeline ranks without data), and Megatron-style samplers
# (BaseMegatronSampler, MegatronPretrainingSampler, MegatronPretrainingRandomSampler, MegatronPretrainingCyclicSampler).
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

import abc
from dataclasses import dataclass
from typing import Optional, Union

import datasets
import torch
import torch.utils.data
from torch.utils.data import BatchSampler
from torch.utils.data.distributed import DistributedSampler
from transformers.trainer_pt_utils import DistributedSamplerWithLoop

from nanotron import logging
from nanotron.logging import log_rank

logger = logging.get_logger(__name__)


class SkipBatchSampler(BatchSampler):
    # [CLASS DOC]
    # Purpose: wraps another sampler and drops its first items so training resumes where it stopped.
    # Key attributes: batch_sampler (wrapped sampler; in nanotron a per-sample DistributedSampler), skip_batches (number of
    # leading items skipped on this rank, the requested count divided by dp_size).
    # Inherits from torch BatchSampler only for typing; its constructor is not called.
    """
    A `torch.utils.data.BatchSampler` that skips the first `n` batches of another `torch.utils.data.BatchSampler`.

    In case of distributed training, we skip batches on each rank, so a total of `skip_batches * dp_size`
    batches are skipped globally.

    Args:
        batch_sampler: Base batch sampler
        skip_batches: Number of batches to skip
        dp_size: Data parallel world size
    """

    def __init__(self, batch_sampler: BatchSampler, skip_batches: int, dp_size: int):
        # [FUNCTION DOC]
        # Summary: stores the wrapped sampler and the per-rank skip count (skip_batches integer-divided by dp_size).
        # Inputs: batch_sampler, skip_batches (global count, e.g. consumed samples), dp_size (int).
        # Outputs: None. Exceptions: ZeroDivisionError if dp_size is 0. Side effects: sets attributes.
        self.batch_sampler = batch_sampler
        # In case of DDP, we skip batches on each rank, so a total of `skip_batches * dp_size` batches
        self.skip_batches = skip_batches // dp_size

    def __iter__(self):
        # [FUNCTION DOC]
        # Summary: yields items of the wrapped sampler whose position is at least skip_batches.
        # Inputs: None. Outputs: generator of items. Exceptions: None explicitly raised.
        # Side effects: iterates the wrapped sampler.
        for index, samples in enumerate(self.batch_sampler):
            if index >= self.skip_batches:
                yield samples

    @property
    def total_length(self):
        # [FUNCTION DOC]
        # Summary: property, length of the wrapped sampler. Inputs: None. Outputs: int.
        # Exceptions: None explicitly raised. Side effects: None.
        return len(self.batch_sampler)

    def __len__(self):
        # [FUNCTION DOC]
        # Summary: wrapped length minus skip_batches (can be negative if over-skipped). Inputs: None. Outputs: int.
        # Exceptions: None explicitly raised. Side effects: None.
        return len(self.batch_sampler) - self.skip_batches


def get_sampler(
    dl_ranks_size: int,
    dl_rank: int,
    train_dataset: Union["torch.utils.data.Dataset", "datasets.Dataset"],
    consumed_train_samples: int,
    seed: int = 42,
    use_loop_to_round_batch_size: bool = False,
    micro_batch_size: Optional[int] = None,
    drop_last: Optional[bool] = True,
    shuffle: bool = False,
) -> Optional[torch.utils.data.Sampler]:
    # [FUNCTION DOC]
    # Summary: builds the DP-sharded sampler for training.
    # Behavior: if use_loop_to_round_batch_size, assert micro_batch_size and use transformers' DistributedSamplerWithLoop
    # (pads each rank to a multiple of micro_batch_size by looping); otherwise a torch DistributedSampler with optional
    # shuffle. Both shard by dl_rank among dl_ranks_size with seed and drop_last. If consumed_train_samples is positive,
    # wrap in SkipBatchSampler to skip the samples this rank already consumed.
    # Inputs: dl_ranks_size, dl_rank (ints), train_dataset (torch or datasets dataset), consumed_train_samples (int),
    # seed (default 42), use_loop_to_round_batch_size (default False), micro_batch_size (optional), drop_last (default
    # True), shuffle (default False; ignored by the looping sampler, which always shuffles).
    # Outputs: sampler.
    # Exceptions: AssertionError if looping without micro_batch_size.
    # Side effects: None.
    """
    Returns sampler that restricts data loading to a subset of the dataset proper to the DP rank.

    Args:
        dl_ranks_size: Number of data parallel ranks
        dl_rank: Current data parallel rank
        train_dataset: Dataset to sample from
        consumed_train_samples: Number of samples already consumed
        seed: Random seed
        use_loop_to_round_batch_size: Whether to loop at the end back to beginning to ensure
                                     each process has a batch size multiple of micro_batch_size
        micro_batch_size: Batch size for each iteration
        drop_last: Whether to drop the last incomplete batch
        shuffle: Whether to shuffle the dataset

    Returns:
        Sampler for distributed training with appropriate skip behavior
    """
    if use_loop_to_round_batch_size:
        assert micro_batch_size is not None
        # loops at the end back to the beginning of the shuffled samples to make each process have a round multiple of batch_size samples.
        sampler = DistributedSamplerWithLoop(
            train_dataset,
            batch_size=micro_batch_size,
            num_replicas=dl_ranks_size,
            rank=dl_rank,
            seed=seed,
            drop_last=drop_last,
        )
    else:
        sampler = DistributedSampler(
            train_dataset, num_replicas=dl_ranks_size, rank=dl_rank, seed=seed, drop_last=drop_last, shuffle=shuffle
        )

    if consumed_train_samples > 0:
        sampler = SkipBatchSampler(sampler, skip_batches=consumed_train_samples, dp_size=dl_ranks_size)

    return sampler


class EmptyInfiniteDataset:
    # [CLASS DOC]
    # Purpose: dataset of fixed length whose every integer index returns an empty dictionary; given to pipeline ranks that
    # need no data so the DataLoader still iterates the right number of steps.
    # Key attributes: _length (int).
    """
    Hack as removing all columns from a datasets.Dataset makes the number of rows 0.
    This provides a dataset that returns empty dicts but maintains the original length.

    Args:
        length: Number of items in the dataset
    """

    def __init__(self, length: int):
        # [FUNCTION DOC]
        # Summary: stores length. Inputs: length (int). Outputs: None. Exceptions: None explicitly raised.
        # Side effects: sets _length.
        self._length = length

    def __getitem__(self, item) -> dict:
        # [FUNCTION DOC]
        # Summary: returns an empty dictionary for integer indices.
        # Inputs: item. Outputs: empty dictionary.
        # Exceptions: NotImplementedError for non-integer indices (including numpy integers).
        # Side effects: None.
        if isinstance(item, int):
            return {}
        raise NotImplementedError(f"{item} of type {type(item)} is not supported yet")

    def __len__(self) -> int:
        # [FUNCTION DOC]
        # Summary: returns the stored length. Inputs: None. Outputs: int. Exceptions: None explicitly raised. Side effects: None.
        return self._length


@dataclass
class BaseMegatronSampler:
    # [CLASS DOC]
    # Purpose: dataclass base of Megatron-style batch samplers that yield lists of indices per micro-batch for one DP rank.
    # Attributes: total_samples, consumed_samples (global, across DP ranks), micro_batch_size, data_parallel_rank,
    # data_parallel_size, global_batch_size, drop_last (default True), pad_samples_to_global_batch_size (default False),
    # micro_batch_times_data_parallel_size (derived).
    # Inheritance role: subclasses implement __iter__ (and __len__).
    # KNOWN ISSUE: __iter__ is marked abstract but the class does not use ABCMeta, so it is not enforced.
    total_samples: int
    consumed_samples: int
    micro_batch_size: int
    data_parallel_rank: int
    data_parallel_size: int
    global_batch_size: int
    drop_last: bool = True
    pad_samples_to_global_batch_size: Optional[bool] = False

    def __post_init__(self):
        # [FUNCTION DOC]
        # Summary: computes micro_batch_times_data_parallel_size and validates arguments, then logs on rank 0.
        # Exceptions: RuntimeError if total_samples is not positive, consumed_samples is not below total, micro_batch_size or
        # data_parallel_size is not positive, rank is out of range, global_batch_size is not divisible by micro batch times DP
        # size, or padding is requested without global_batch_size.
        # Inputs: self. Outputs: None. Side effects: sets an attribute, logging.
        # KNOWN ISSUE: the None check on global_batch_size comes after a modulo on it, which would already raise TypeError.
        self.micro_batch_times_data_parallel_size = self.micro_batch_size * self.data_parallel_size

        # Sanity checks.
        if self.total_samples <= 0:
            raise RuntimeError("no sample to consume: {}".format(self.total_samples))
        if self.consumed_samples >= self.total_samples:
            raise RuntimeError("no samples left to consume: {}, {}".format(self.consumed_samples, self.total_samples))
        if self.micro_batch_size <= 0:
            raise RuntimeError(f"micro_batch_size size must be greater than 0, but {self.micro_batch_size}")
        if self.data_parallel_size <= 0:
            raise RuntimeError(f"data parallel size must be greater than 0, but {self.data_parallel_size}")
        if self.data_parallel_rank >= self.data_parallel_size:
            raise RuntimeError(
                "data_parallel_rank should be smaller than data size, but {} >= {}".format(
                    self.data_parallel_rank, self.data_parallel_size
                )
            )
        if self.global_batch_size % (self.micro_batch_size * self.data_parallel_size) != 0:
            raise RuntimeError(
                f"`global_batch_size` ({self.global_batch_size}) is not divisible by "
                f"`micro_batch_size ({self.micro_batch_size}) x data_parallel_size "
                f"({self.data_parallel_size})`"
            )
        if self.pad_samples_to_global_batch_size and self.global_batch_size is None:
            raise RuntimeError(
                "`pad_samples_to_global_batch_size` can be `True` only when "
                "`global_batch_size` is set to an integer value"
            )
        log_rank(
            f"Instantiating MegatronPretrainingSampler with total_samples: {self.total_samples} and consumed_samples: {self.consumed_samples}",
            logger=logger,
            level=logging.INFO,
            rank=0,
        )

    @abc.abstractmethod
    def __iter__(self):
        # [FUNCTION DOC]
        # Summary: abstract iterator placeholder; returns None.
        # Inputs: None. Outputs: None. Exceptions: None explicitly raised. Side effects: None.
        ...


@dataclass
class MegatronPretrainingSampler(BaseMegatronSampler):
    # [CLASS DOC]
    # Purpose: sequential sampler: global indices from consumed_samples onward are grouped into global steps of
    # micro_batch_size times DP size, and each rank takes its contiguous micro-batch slice.
    def get_start_end_idx(self):
        # [FUNCTION DOC]
        # HELPER: start and end offsets of this rank's slice inside a step: rank times micro_batch_size, plus micro_batch_size.
        # Inputs: None. Outputs: tuple of two ints. Exceptions: None explicitly raised. Side effects: None.
        start_idx = self.data_parallel_rank * self.micro_batch_size
        end_idx = start_idx + self.micro_batch_size
        return start_idx, end_idx

    def __len__(self):
        # [FUNCTION DOC]
        # Summary: number of remaining steps: remaining samples divided by global_batch_size (rounded down if drop_last else
        # up); if global_batch_size is None, divide by micro batch times DP size instead.
        # Inputs: None. Outputs: int. Exceptions: None explicitly raised. Side effects: None.
        # KNOWN ISSUE: counts global batches while __iter__ yields micro-batches, so the length does not match the number of
        # yielded items when global_batch_size is larger than micro batch times DP size.
        num_available_samples: int = self.total_samples - self.consumed_samples
        if self.global_batch_size is not None:
            if self.drop_last:
                return num_available_samples // self.global_batch_size
            else:
                return (num_available_samples + self.global_batch_size - 1) // self.global_batch_size
        else:
            if self.drop_last:
                return num_available_samples // self.micro_batch_times_data_parallel_size
            else:
                return (num_available_samples - 1) // self.micro_batch_times_data_parallel_size + 1

    def __iter__(self):
        # [FUNCTION DOC]
        # Summary: yields this rank's index list for each full step, advancing consumed_samples by the step size; afterwards
        # handles a partial final step when drop_last is False (optionally padded with -1 indices up to micro_batch_size).
        # Inputs: None. Outputs: generator of lists of ints.
        # Exceptions: IndexError possible in the padding branch (see issue).
        # Side effects: mutates consumed_samples; debug logging.
        # KNOWN ISSUE: the padding branch uses max where min is intended when bounding the index range, reading past the
        # partial batch; for a short last batch the non-padding branch can also yield an empty list.
        batch = []
        batch_idx = 0
        # Last batch will be dropped if drop_last is not set False
        for idx in range(self.consumed_samples, self.total_samples):
            batch.append(idx)
            if len(batch) == self.micro_batch_times_data_parallel_size:
                start_idx, end_idx = self.get_start_end_idx()
                log_rank(
                    f"DP {self.data_parallel_rank} batch {batch_idx} {batch[start_idx:end_idx]} self.consumed_samples {self.consumed_samples}",
                    logger=logger,
                    level=logging.DEBUG,
                )
                # self.last_consumed_sample_all_ranks = batch[-1] # = self.consumed_samples?
                self.consumed_samples += self.micro_batch_times_data_parallel_size
                yield batch[start_idx:end_idx]
                batch = []
                batch_idx += 1

        # Check the last partial batch and see drop_last is set
        if len(batch) > 0 and not self.drop_last:
            if self.pad_samples_to_global_batch_size:
                for i in range(
                    self.data_parallel_rank, self.global_batch_size, self.micro_batch_times_data_parallel_size
                ):
                    indices = [batch[j] for j in range(i, max(len(batch), i + self.micro_batch_size))]
                    num_pad = self.micro_batch_size - len(indices)
                    indices = indices + [-1] * num_pad
                    yield indices
            else:
                start_idx, end_idx = self.get_start_end_idx()
                yield batch[start_idx:end_idx]


@dataclass
class MegatronPretrainingRandomSampler(BaseMegatronSampler):
    # [CLASS DOC]
    # Purpose: random sampler: each rank owns a contiguous bucket of the dataset and reads it in a per-epoch random
    # permutation seeded by the epoch number; supports resuming mid-epoch.
    def __len__(self):
        # [FUNCTION DOC]
        # Summary: same remaining-step formula as MegatronPretrainingSampler.__len__.
        # Inputs: None. Outputs: int. Exceptions: None explicitly raised. Side effects: None.
        num_available_samples: int = self.total_samples - self.consumed_samples
        if self.global_batch_size is not None:
            if self.drop_last:
                return num_available_samples // self.global_batch_size
            else:
                return (num_available_samples + self.global_batch_size - 1) // self.global_batch_size
        else:
            if self.drop_last:
                return num_available_samples // self.micro_batch_times_data_parallel_size
            else:
                return (num_available_samples - 1) // self.micro_batch_times_data_parallel_size + 1

    def __iter__(self):
        # [FUNCTION DOC]
        # Summary: yields random micro-batches from this rank's bucket.
        # Behavior: drop the tail that does not fill a step; epoch is consumed_samples divided by the active total and the
        # in-epoch position must be a whole number of steps. Bucket size is the number of steps times micro_batch_size; this
        # rank's bucket starts at rank times bucket size; skip the in-epoch position divided by DP size. Permute the bucket
        # with a torch generator seeded with the epoch, then yield micro_batch_size chunks, advancing consumed_samples by a
        # step each time; a final partial chunk is yielded only if drop_last is False.
        # Inputs: None. Outputs: generator of lists of ints.
        # Exceptions: AssertionError if consumed_samples is not step-aligned.
        # Side effects: sets last_batch_size and epoch, mutates consumed_samples.
        self.last_batch_size = self.total_samples % self.micro_batch_times_data_parallel_size
        active_total_samples = self.total_samples - self.last_batch_size
        self.epoch = self.consumed_samples // active_total_samples
        current_epoch_samples = self.consumed_samples % active_total_samples
        assert current_epoch_samples % self.micro_batch_times_data_parallel_size == 0

        # data sharding and random sampling
        bucket_size = (self.total_samples // self.micro_batch_times_data_parallel_size) * self.micro_batch_size
        bucket_offset = current_epoch_samples // self.data_parallel_size
        start_idx = self.data_parallel_rank * bucket_size

        g = torch.Generator()
        g.manual_seed(self.epoch)
        random_idx = torch.randperm(bucket_size, generator=g).tolist()
        idx_range = [start_idx + x for x in random_idx[bucket_offset:]]

        batch = []
        # Last batch if not complete will be dropped.
        for idx in idx_range:
            batch.append(idx)
            if len(batch) == self.micro_batch_size:
                self.consumed_samples += self.micro_batch_times_data_parallel_size
                yield batch
                batch = []

        # Check the last partial batch and see drop_last is set
        if len(batch) > 0 and not self.drop_last:
            yield batch


class MegatronPretrainingCyclicSampler(BaseMegatronSampler):
    # [CLASS DOC]
    # Purpose: like the random sampler but without shuffling: each rank reads its contiguous bucket in order and starts
    # over after each epoch.
    # Additional attribute: last_batch_size (samples that do not fill a full step).
    """Cyclic sampler

    This sampler is used for the cyclic pretraining. It will go through the dataset
    once and then start over again without shuffling.

    For data parallelism, the dataset is sharded into `data_parallel_size` chunks at the full dataset level.
    Each rank will then sample from its own shard starting from a different offset in the full dataset.

    Args:
        total_samples (int): total number of samples in the dataset
        consumed_samples (int): number of samples already consumed across all dataparallel ranks
        micro_batch_size (int): number of samples in a micro batch
        data_parallel_rank (int): rank of the data parallel group
        data_parallel_size (int): size of the data parallel group
        drop_last (bool): drop the last batch if it is not complete
        global_batch_size (int): global batch size
        pad_samples_to_global_batch_size (bool): pad the last batch to global batch size
    """

    def __init__(
        self,
        total_samples: int,
        consumed_samples: int,
        micro_batch_size: int,
        data_parallel_rank: int,
        data_parallel_size: int,
        global_batch_size: int,
        drop_last: bool = True,
        pad_samples_to_global_batch_size: Optional[bool] = False,
    ) -> None:
        # [FUNCTION DOC]
        # Summary: forwards all arguments to the dataclass base (running its validation), asserts no padding, and computes
        # last_batch_size.
        # Inputs: total_samples, consumed_samples, micro_batch_size, data_parallel_rank, data_parallel_size,
        # global_batch_size, drop_last (default True), pad_samples_to_global_batch_size (default False).
        # Outputs: None.
        # Exceptions: RuntimeError from validation; AssertionError if padding is requested.
        # Side effects: logging.
        super().__init__(
            total_samples=total_samples,
            consumed_samples=consumed_samples,
            micro_batch_size=micro_batch_size,
            data_parallel_rank=data_parallel_rank,
            data_parallel_size=data_parallel_size,
            drop_last=drop_last,
            global_batch_size=global_batch_size,
            pad_samples_to_global_batch_size=pad_samples_to_global_batch_size,
        )
        assert (
            pad_samples_to_global_batch_size is False
        ), "`MegatronPretrainingCyclicSampler` does not support sample padding"
        self.last_batch_size = self.total_samples % self.micro_batch_times_data_parallel_size

    def __len__(self):
        # [FUNCTION DOC]
        # Summary: same remaining-step formula as MegatronPretrainingSampler.__len__.
        # Inputs: None. Outputs: int. Exceptions: None explicitly raised. Side effects: None.
        num_available_samples: int = self.total_samples - self.consumed_samples
        if self.global_batch_size is not None:
            if self.drop_last:
                return num_available_samples // self.global_batch_size
            else:
                return (num_available_samples + self.global_batch_size - 1) // self.global_batch_size
        else:
            if self.drop_last:
                return num_available_samples // self.micro_batch_times_data_parallel_size
            else:
                return (num_available_samples - 1) // self.micro_batch_times_data_parallel_size + 1

    def __iter__(self):
        # [FUNCTION DOC]
        # Summary: yields consecutive micro-batches of this rank's bucket starting at the in-epoch offset (computed as in the
        # random sampler), advancing consumed_samples per step; partial final chunk only if drop_last is False.
        # Inputs: None. Outputs: generator of lists of ints.
        # Exceptions: AssertionError if consumed_samples is not step-aligned.
        # Side effects: sets epoch, mutates consumed_samples.
        active_total_samples = self.total_samples - self.last_batch_size
        self.epoch = self.consumed_samples // active_total_samples
        current_epoch_samples = self.consumed_samples % active_total_samples
        assert current_epoch_samples % self.micro_batch_times_data_parallel_size == 0

        # data sharding and random sampling
        bucket_size = (self.total_samples // self.micro_batch_times_data_parallel_size) * self.micro_batch_size
        bucket_offset = current_epoch_samples // self.data_parallel_size
        start_idx = self.data_parallel_rank * bucket_size

        batch = []
        # Last batch if not complete will be dropped.
        for idx in range(bucket_offset, bucket_size):
            batch.append(idx + start_idx)
            if len(batch) == self.micro_batch_size:
                self.consumed_samples += self.micro_batch_times_data_parallel_size
                yield batch
                batch = []

        # Check the last partial batch and see drop_last is set
        if len(batch) > 0 and not self.drop_last:
            yield batch
