# [PACKAGE DOC]
# Purpose: package nanotron.data.nemo_dataset, a port of the NeMo/Megatron GPT dataset pipeline: memory-mapped
# indexed token datasets split into train/valid/test, turned into fixed-length samples via cached index mappings,
# optionally blended across several corpora and optionally augmented with Fill-In-the-Middle (FIM).
# Main contents: build_dataset, build_train_valid_test_datasets, _build_train_valid_test_datasets,
# get_indexed_dataset, FIM/EOD special token strings, GPTDataset, SubsetSplitLog, the index builders
# (_build_index_mappings, _num_tokens, _num_epochs, _build_doc_idx, _build_sample_idx, _build_shuffle_idx) and permute.
# Submodules: blendable_dataset (BlendableDataset), dataset_utils (split parsing, weights, C++ helper compile),
# indexed_dataset (MMapIndexedDataset). An optional compiled C++ module "helpers" builds sample indices.
# Index mapping semantics: doc_idx lists document ids repeated over epochs and shuffled; sample_idx has one row per
# sample boundary (index into doc_idx, token offset in that document), sample i spanning rows i to i+1; shuffle_idx
# permutes sample ids. They are saved as .npy files next to the data (or in index_mapping_dir) by the first rank and
# memory-mapped by all ranks.
# Import-time side effects: creates a module logger, defines the token string constants.
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

"""GPT style dataset."""

import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase

from nanotron import distributed as dist
from nanotron import logging
from nanotron.logging import log_rank
from nanotron.parallel import ParallelContext
from nanotron.utils import main_rank_first

from .blendable_dataset import BlendableDataset
from .dataset_utils import (
    get_datasets_weights_and_num_samples,
    get_train_valid_test_split_,
)
from .indexed_dataset import MMapIndexedDataset, make_indexed_dataset

logger = logging.get_logger(__name__)


def build_dataset(
    cfg: Any,
    tokenizer: PreTrainedTokenizerBase,
    data_prefix: List[str],
    num_samples: int,
    seq_length: int,
    seed: Any,
    skip_warmup: bool,
    name: str,
    parallel_context: ParallelContext,
) -> Union["GPTDataset", BlendableDataset]:
    # [FUNCTION DOC]
    # Summary: builds one GPTDataset (single prefix) or a BlendableDataset (several prefixes) covering all documents.
    # Behavior: with exactly one prefix build a GPTDataset over all of its documents for num_samples. Otherwise parse
    # data_prefix as alternating weights and prefixes with get_datasets_weights_and_num_samples, build one GPTDataset per
    # prefix with its own sample count and blend them with the weights into num_samples samples.
    # Inputs: cfg (config object with validation_drop_last, eod_mask_loss, FIM settings, index_mapping_dir, etc.),
    # tokenizer (Hugging Face tokenizer with FIM and EOD tokens), data_prefix (list of path prefixes, optionally with
    # weights), num_samples, seq_length, seed, skip_warmup (skip memmap warm-up read), name ("train", "valid" or "test";
    # for "valid" drop_last follows cfg.validation_drop_last), parallel_context.
    # Outputs: GPTDataset or BlendableDataset.
    # Exceptions: propagates errors from dataset construction.
    # Side effects: file I/O for index mappings; rank 0 logging.
    def _build_dataset(current_data_prefix: str, current_num_samples: int) -> "GPTDataset":
        # [FUNCTION DOC]
        # HELPER: builds a GPTDataset for one prefix over all its documents.
        # Inputs: current_data_prefix (string path prefix), current_num_samples (int).
        # Outputs: GPTDataset.
        # Exceptions: None explicitly raised.
        # Side effects: loads the indexed dataset, logs document counts on rank 0.
        indexed_dataset = get_indexed_dataset(current_data_prefix, skip_warmup)
        total_num_of_documents = indexed_dataset.sizes.shape[0]
        # Print stats about the splits.

        log_rank(" > dataset split:", logger=logger, level=logging.INFO, rank=0)
        log_rank(
            "     Total {} documents is : {} ".format(name, total_num_of_documents),
            logger=logger,
            level=logging.INFO,
            rank=0,
        )
        drop_last = True
        if name == "valid":
            drop_last = cfg.validation_drop_last
        dataset = GPTDataset(
            cfg,
            tokenizer,
            name,
            current_data_prefix,
            np.arange(start=0, stop=total_num_of_documents, step=1, dtype=np.int32),
            indexed_dataset,
            current_num_samples,
            seq_length,
            seed,
            parallel_context,
            drop_last=drop_last,
        )
        return dataset

    if len(data_prefix) == 1:
        return _build_dataset(data_prefix[0], num_samples)

    else:
        output = get_datasets_weights_and_num_samples(data_prefix, num_samples)
        prefixes, weights, datasets_num_samples = output
        datasets = []
        for i in range(len(prefixes)):
            dataset = _build_dataset(prefixes[i], datasets_num_samples[i])
            datasets.append(dataset)
        return BlendableDataset(datasets, weights, num_samples, parallel_context=parallel_context)


def build_train_valid_test_datasets(
    cfg: Any,
    tokenizer: PreTrainedTokenizerBase,
    data_prefix: Union[Dict, List],
    splits_string: str,
    train_valid_test_num_samples: Tuple[int, int, int],
    seq_length: int,
    seed: Any,
    parallel_context: ParallelContext,
    skip_warmup: bool,
) -> Tuple[
    Union["GPTDataset", "BlendableDataset", None],
    Union["GPTDataset", "BlendableDataset", None],
    Union["GPTDataset", "BlendableDataset", None],
]:
    # [FUNCTION DOC]
    # Summary: builds train, validation and test datasets either from explicit per-split prefixes or by splitting.
    # Behavior: if data_prefix is a dictionary it must have train, test and validation keys; splits_string is ignored
    # (warning logged if cfg.splits_string is set) and each split is built with build_dataset. Otherwise: an empty list
    # returns early; a single prefix is split by document ranges via _build_train_valid_test_datasets; several prefixes
    # are parsed into weights and per-dataset sample counts, split individually, and each non-empty split list is blended
    # with BlendableDataset using the summed sample counts.
    # Inputs: cfg, tokenizer, data_prefix (dictionary or list), splits_string (e.g. "969,30,1"),
    # train_valid_test_num_samples (three integers), seq_length, seed, parallel_context, skip_warmup.
    # Outputs: tuple of three datasets, each possibly None.
    # Exceptions: AssertionError for a dictionary missing required keys.
    # Side effects: file I/O, logging.
    # KNOWN ISSUE: the empty-prefix branch returns a nested tuple plus an empty list instead of three values.
    # KNOWN ISSUE: the blended branch passes the same weights to all splits even if some datasets produced an empty split,
    # so weights and datasets can mismatch in length.
    if isinstance(data_prefix, dict):
        assert (
            data_prefix.get("train") is not None
            and data_prefix.get("test") is not None
            and data_prefix.get("validation") is not None
        ), f"Data prefix dictionary should have train, test and validation keys.  data_prefix currently has only {data_prefix.keys()}"
        if cfg.splits_string is not None:
            log_rank(
                cfg.splits_string + " ignored since data prefix is of type dictionary.",
                logger=logger,
                level=logging.WARNING,
                rank=0,
            )
        train_ds = build_dataset(
            cfg,
            tokenizer,
            data_prefix["train"],
            int(train_valid_test_num_samples[0]),
            seq_length,
            seed,
            skip_warmup,
            "train",
            parallel_context,
        )
        validation_ds = build_dataset(
            cfg,
            tokenizer,
            data_prefix["validation"],
            int(train_valid_test_num_samples[1]),
            seq_length,
            seed,
            skip_warmup,
            "valid",
            parallel_context,
        )
        test_ds = build_dataset(
            cfg,
            tokenizer,
            data_prefix["test"],
            int(train_valid_test_num_samples[2]),
            seq_length,
            seed,
            skip_warmup,
            "test",
            parallel_context,
        )
        return train_ds, validation_ds, test_ds

    else:
        # No data
        if len(data_prefix) == 0:
            return (None, None, None), []
        # Single dataset.
        if len(data_prefix) == 1:
            return _build_train_valid_test_datasets(
                cfg,
                tokenizer,
                data_prefix[0],
                splits_string,
                train_valid_test_num_samples,
                seq_length,
                seed,
                parallel_context,
                skip_warmup,
            )

        # Blending dataset.
        # Parse the values.
        output = get_datasets_weights_and_num_samples(data_prefix, train_valid_test_num_samples)
        prefixes, weights, datasets_train_valid_test_num_samples = output

        # Build individual datasets.
        train_datasets: List["GPTDataset"] = []
        valid_datasets: List["GPTDataset"] = []
        test_datasets: List["GPTDataset"] = []
        for i in range(len(prefixes)):
            train_ds, valid_ds, test_ds = _build_train_valid_test_datasets(
                cfg,
                tokenizer,
                prefixes[i],
                splits_string,
                datasets_train_valid_test_num_samples[i],
                seq_length,
                seed,
                parallel_context,
                skip_warmup,
            )
            if train_ds:
                train_datasets.append(train_ds)
            if valid_ds:
                valid_datasets.append(valid_ds)
            if test_ds:
                test_datasets.append(test_ds)

        train_n, valid_n, test_n = map(sum, zip(*datasets_train_valid_test_num_samples))

        # Blend.
        blending_train_dataset = None
        if train_datasets:
            blending_train_dataset = BlendableDataset(train_datasets, weights, train_n, parallel_context)
        blending_valid_dataset = None
        if valid_datasets:
            blending_valid_dataset = BlendableDataset(valid_datasets, weights, valid_n, parallel_context)
        blending_test_dataset = None
        if test_datasets:
            blending_test_dataset = BlendableDataset(test_datasets, weights, test_n, parallel_context)

        return (blending_train_dataset, blending_valid_dataset, blending_test_dataset)


def _build_train_valid_test_datasets(
    cfg: Any,
    tokenizer: PreTrainedTokenizerBase,
    data_prefix: str,
    splits_string: str,
    train_valid_test_num_samples: int,
    seq_length: int,
    seed: Any,
    parallel_context: ParallelContext,
    skip_warmup: bool,
) -> Tuple["GPTDataset", Optional["GPTDataset"], Optional["GPTDataset"]]:
    # [FUNCTION DOC]
    # HELPER: splits a single indexed dataset into train/valid/test GPTDatasets by contiguous document ranges.
    # Behavior: load the indexed dataset, compute split boundaries with get_train_valid_test_split_, log the ranges, and
    # for each split with a non-empty range build a GPTDataset over those document ids with that split's sample count
    # (drop_last from cfg.validation_drop_last for "valid").
    # Inputs: cfg, tokenizer, data_prefix (string), splits_string, train_valid_test_num_samples (sequence of three ints),
    # seq_length, seed, parallel_context, skip_warmup.
    # Outputs: tuple of three GPTDataset or None.
    # Exceptions: None explicitly raised.
    # Side effects: file I/O, logging.
    # KNOWN ISSUE: train_valid_test_num_samples is annotated as int but is indexed as a sequence.
    """Build train, valid, and test datasets."""

    # Indexed dataset.
    indexed_dataset = get_indexed_dataset(data_prefix, skip_warmup)

    total_num_of_documents = indexed_dataset.sizes.shape[0]
    splits = get_train_valid_test_split_(splits_string, total_num_of_documents)

    # Print stats about the splits.
    log_rank(" > dataset split:", logger=logger, level=logging.INFO, rank=0)

    def print_split_stats(name, index):
        # [FUNCTION DOC]
        # HELPER: logs the document range and count of split number index under the given name.
        # Inputs: name (string), index (0, 1 or 2). Outputs: None. Exceptions: None explicitly raised. Side effects: logging.
        log_rank("    {}:".format(name), logger=logger, level=logging.INFO, rank=0)
        log_rank(
            "     document indices in [{}, {}) total of {} "
            "documents".format(splits[index], splits[index + 1], splits[index + 1] - splits[index]),
            logger=logger,
            level=logging.INFO,
            rank=0,
        )

    print_split_stats("train", 0)
    print_split_stats("validation", 1)
    print_split_stats("test", 2)

    def build_dataset(index: int, name: str) -> GPTDataset:
        # [FUNCTION DOC]
        # HELPER: builds the GPTDataset for split number index, or None if the range is empty.
        # Inputs: index (int), name (string). Outputs: GPTDataset or None.
        # Exceptions: None explicitly raised. Side effects: index-mapping file I/O.
        dataset = None
        if splits[index + 1] > splits[index]:
            documents = np.arange(start=splits[index], stop=splits[index + 1], step=1, dtype=np.int32)
            drop_last = True
            if name == "valid":
                drop_last = cfg.validation_drop_last
            dataset = GPTDataset(
                cfg,
                tokenizer,
                name,
                data_prefix,
                documents,
                indexed_dataset,
                train_valid_test_num_samples[index],
                seq_length,
                seed,
                parallel_context,
                drop_last=drop_last,
            )
        return dataset

    train_dataset = build_dataset(0, "train")
    valid_dataset = build_dataset(1, "valid")
    test_dataset = build_dataset(2, "test")

    return (train_dataset, valid_dataset, test_dataset)


def get_indexed_dataset(data_prefix: str, skip_warmup: bool) -> MMapIndexedDataset:
    # [FUNCTION DOC]
    # Summary: opens the memory-mapped indexed dataset at data_prefix via make_indexed_dataset and logs timing and count.
    # Inputs: data_prefix (path prefix of .bin/.idx files), skip_warmup (bool).
    # Outputs: MMapIndexedDataset.
    # Exceptions: propagates file errors.
    # Side effects: file reads, logging on rank 0.
    """Build indexed dataset."""
    log_rank(" > building dataset index ...", logger=logger, level=logging.INFO, rank=0)

    start_time = time.time()
    indexed_dataset = make_indexed_dataset(data_prefix, skip_warmup)
    log_rank(
        " > finished creating indexed dataset in {:4f} " "seconds".format(time.time() - start_time),
        logger=logger,
        level=logging.INFO,
        rank=0,
    )
    log_rank(
        "    number of documents: {}".format(indexed_dataset.sizes.shape[0]), logger=logger, level=logging.INFO, rank=0
    )

    return indexed_dataset


FIM_PREFIX = "<fim_prefix>"
FIM_MIDDLE = "<fim_middle>"
FIM_SUFFIX = "<fim_suffix>"
FIM_PAD = "<fim_pad>"
EOD = "<|endoftext|>"


class GPTDataset(Dataset):
    # [CLASS DOC]
    # Purpose: torch Dataset returning fixed-length token samples of seq_length (+1 extra token unless
    # cfg.no_seqlen_plus_one_input_tokens) from a document subset, with optional FIM augmentation.
    # Key attributes: name, indexed_dataset, drop_last, seq_length, eod_mask_loss, add_extra_token (1 or 0),
    # index_mapping_dir, fim_rate, fim_spm_rate, tokenizer, special token ids (suffix, prefix, middle, pad, eod),
    # fim_split_sample (token id or None), fragment_fim_rate, no_fim_prefix, np_rng (RandomState for FIM),
    # doc_idx, sample_idx, shuffle_idx (index mappings), subset_log (SubsetSplitLog).
    # Inherits from torch.utils.data.Dataset.
    def __init__(
        self,
        cfg: Any,
        tokenizer: PreTrainedTokenizerBase,
        name: str,
        data_prefix: str,
        documents: np.ndarray,
        indexed_dataset: MMapIndexedDataset,
        num_samples: int,
        seq_length: int,
        seed: int,
        parallel_context: ParallelContext,
        drop_last: bool = True,
    ):
        # [FUNCTION DOC]
        # Summary: stores configuration, resolves special token ids, creates the mapping directory and builds/loads the
        # index mappings.
        # Behavior: assert document ids lie within the indexed dataset; read eod/FIM settings from cfg; validate both FIM
        # rates are in [0, 1]; look up FIM_SUFFIX, FIM_PREFIX, FIM_MIDDLE, FIM_PAD and EOD in tokenizer.vocab; resolve
        # fim_split_sample; seed a RandomState. When distributed is initialised, create index_mapping_dir rank by rank
        # (main rank first). Call _build_index_mappings, keep the three arrays and the log, then release the indexed dataset's
        # cached index memory.
        # Inputs: cfg, tokenizer, name, data_prefix, documents (int32 array of document ids), indexed_dataset, num_samples,
        # seq_length, seed, parallel_context, drop_last (default True).
        # Outputs: None.
        # Exceptions: AssertionError for invalid documents; ValueError for rates outside [0, 1]; KeyError if the tokenizer
        # lacks the special tokens.
        # Side effects: directory creation, index file I/O, barriers inside main_rank_first.
        super().__init__()
        self.name = name
        self.indexed_dataset = indexed_dataset
        self.drop_last = drop_last
        self.seq_length = seq_length

        # Checks
        assert np.min(documents) >= 0
        assert np.max(documents) < indexed_dataset.sizes.shape[0]

        self.eod_mask_loss = cfg.eod_mask_loss
        self.no_seqlen_plus_one_input_tokens = cfg.no_seqlen_plus_one_input_tokens
        self.add_extra_token = 1
        if self.no_seqlen_plus_one_input_tokens:
            self.add_extra_token = 0

        # save index mappings to a configurable dir
        self.index_mapping_dir = cfg.index_mapping_dir

        # For FIM
        self.fim_rate = cfg.fim_rate
        self.fim_spm_rate = cfg.fim_spm_rate
        if self.fim_rate > 1 or self.fim_rate < 0:
            raise ValueError("FIM rate must be a probability 0 <= rate <= 1")
        if self.fim_spm_rate > 1 or self.fim_spm_rate < 0:
            raise ValueError("SPM rate must be a probability 0 <= rate <= 1")
        self.tokenizer = tokenizer
        self.suffix_tok_id, self.prefix_tok_id, self.middle_tok_id, self.pad_tok_id, self.eod_tok_id = (
            self.tokenizer.vocab[tok] for tok in [FIM_SUFFIX, FIM_PREFIX, FIM_MIDDLE, FIM_PAD, EOD]
        )
        self.fim_split_sample = (
            self.tokenizer.vocab[cfg.fim_split_sample] if cfg.fim_split_sample is not None else None
        )
        self.fragment_fim_rate = cfg.fragment_fim_rate
        self.no_fim_prefix = cfg.no_fim_prefix
        self.np_rng = np.random.RandomState(seed=seed)  # rng state for FIM

        # create index_mapping_dir on rank 0
        if dist.is_available() and dist.is_initialized():
            with main_rank_first(parallel_context.world_pg):
                if self.index_mapping_dir is not None and not os.path.isdir(self.index_mapping_dir):
                    os.makedirs(self.index_mapping_dir)

        # Build index mappings.
        arrays, subset_log = _build_index_mappings(
            self.name,
            data_prefix,
            documents,
            self.indexed_dataset.sizes,
            num_samples,
            seq_length,
            seed,
            parallel_context,
            index_mapping_dir=self.index_mapping_dir,
            drop_last=drop_last,
            add_extra_token=self.add_extra_token,
        )
        self.doc_idx, self.sample_idx, self.shuffle_idx = arrays
        self.indexed_dataset.deallocate_indexed_dataset_memory()

        self.subset_log = subset_log

    def __len__(self):
        # [FUNCTION DOC]
        # Summary: number of samples, rows of sample_idx minus one. Inputs: None. Outputs: int.
        # Exceptions: None explicitly raised. Side effects: None.
        # -1 is due to data structure used to retrieve the index:
        #    sample i --> [sample_idx[i], sample_idx[i+1])
        return self.sample_idx.shape[0] - 1

    def _get_text(self, idx: int) -> np.ndarray:
        # [FUNCTION DOC]
        # HELPER: assembles the token sample for an index, used by __getitem__.
        # Behavior: map idx through shuffle_idx; read start (document position, offset) and end rows from sample_idx. If both
        # are in the same document, read the span of the right length; otherwise concatenate the tail of the first document,
        # all middle documents, and the head of the last one. If the length is not seq_length plus add_extra_token, warn and
        # right-pad with -1. If fim_rate is 0 return as int64. Otherwise split the sample on EOD tokens, FIM-permute each
        # non-empty segment (via the nested helpers) re-inserting EOD separators, then truncate or pad with the pad token back
        # to the original length.
        # Inputs: idx (int).
        # Outputs: 1-D int64 numpy array.
        # Exceptions: AssertionError if the final FIM length mismatches.
        # Side effects: advances np_rng; reads memmap; may log warnings.
        # Get the shuffled index.
        idx = self.shuffle_idx[idx]
        # Start and end documents and offsets.
        doc_index_f, offset_f = self.sample_idx[idx]
        doc_index_l, offset_l = self.sample_idx[idx + 1]
        # offset_f = self.sample_idx[idx][1]
        # offset_l = self.sample_idx[idx + 1][1]
        # If we are within the same document, just extract the chunk.
        if doc_index_f == doc_index_l:
            sample = self.indexed_dataset.get(
                self.doc_idx[doc_index_f], offset=offset_f, length=offset_l - offset_f + self.add_extra_token
            )
        else:
            # Otherwise, get the rest of the initial document.
            sample_list = [self.indexed_dataset.get(self.doc_idx[doc_index_f], offset=offset_f)]
            # Loop over all in between documents and add the entire document.
            for i in range(doc_index_f + 1, doc_index_l):
                sample_list.append(self.indexed_dataset.get(self.doc_idx[i]))
            # And finally add the relevant portion of last document.
            sample_list.append(
                self.indexed_dataset.get(self.doc_idx[doc_index_l], length=offset_l + self.add_extra_token)
            )
            sample = np.concatenate(sample_list)
        if len(sample) != (self.seq_length + self.add_extra_token):
            log_rank(
                f" > WARNING: Got sample of length: {len(sample)} for sequence length={self.seq_length+self.add_extra_token}, padding the sample to match sequence length",
                logger=logger,
                level=logging.WARNING,
                rank=0,
            )

            sample = np.array(sample, dtype=np.int64)
            sample = np.pad(
                sample, (0, self.seq_length + self.add_extra_token - len(sample)), mode="constant", constant_values=-1
            )

        if self.fim_rate == 0:
            return sample.astype(np.int64)

        # Code from: https://github.com/EleutherAI/gpt-neox/blob/FIM-clean/megatron/data/gpt2_dataset.py#L109
        # TODO(Hailey): can merge the code below this line with code above this line.
        # TODO(Hailey), cont: above already iterates through loop, so just add the permuting in there?
        sample = np.array(sample, dtype=np.int64)
        sample_len = sample.shape[0]
        # # print(sample, sample.shape)
        # # do FIM here, if enabled
        # TODO: Do we handle the following point from FIM paper?
        # To transform data in the character space for context-level FIM, the tokenized documents have to be decoded back into strings before FIM augmentation. Depending on the vocabulary, some care has to be given to ensure decoding does not introduce any spurious characters into training. For example, utf-8 characters are encoded as multiple tokens with a BPE vocabulary; they can result in fragments from chunking and fail to decode. To prevent unforeseen errors midway through training, we encourage checking for these fragments at the beginning or end of a context and removing them.

        segment_breaks = np.argwhere(sample == self.eod_tok_id)  # split sample by document

        def fim_permute_sequence(sequence, rate):
            # [FUNCTION DOC]
            # HELPER: calls permute on a sequence with the dataset's rng, SPM rate, tokenizer and special tokens, without
            # truncation/padding.
            # Inputs: sequence (int array), rate (FIM probability). Outputs: permuted array.
            # Exceptions: None explicitly raised. Side effects: advances np_rng.
            return permute(
                sequence,
                self.np_rng,
                rate,
                self.fim_spm_rate,
                self.tokenizer,
                truncate_or_pad=False,
                suffix_tok_id=self.suffix_tok_id,
                prefix_tok_id=self.prefix_tok_id,
                middle_tok_id=self.middle_tok_id,
                pad_tok_id=self.pad_tok_id,
                no_fim_prefix=self.no_fim_prefix,
            )

        def fim_split_and_permute_sequence(sequence):
            # [FUNCTION DOC]
            # HELPER: applies FIM to a segment, optionally at fragment level.
            # Behavior: without fim_split_sample (or if the split token is absent) permute the whole sequence with fim_rate.
            # Otherwise draw once with fim_rate whether to apply FIM at all; if yes, split on the split token and permute every
            # non-empty fragment with fragment_fim_rate, keeping the split tokens between them.
            # Inputs: sequence (int array). Outputs: array.
            # Exceptions: None explicitly raised. Side effects: advances np_rng.
            """
            If self.fim_split_sample is not None, split the sequence.
            Then apply FIM on the fragments, or the whole sequence if self.fim_split_sample is None.
            """
            if self.fim_split_sample is None:
                return fim_permute_sequence(sequence, self.fim_rate)
            # fim_split_sample is set: split the sample on this token and permute each fragment separately.
            # Typically, if each sample is a repository, then we split again on the file level.
            # Each fragment is a file, and we permute the files.
            fragment_breaks = np.argwhere(sequence == self.fim_split_sample)
            if fragment_breaks.shape == (0, 1):
                # no split token in this sample
                return fim_permute_sequence(sequence, self.fim_rate)
            if not self.np_rng.binomial(1, self.fim_rate):
                # don't do FIM preproc
                return sequence
            # Do FIM on each fragment
            curr_start_position = 0
            new_samples = []
            for loc in np.nditer(fragment_breaks):
                if loc - curr_start_position > 0:
                    permuted = fim_permute_sequence(sequence[curr_start_position:loc], self.fragment_fim_rate)
                    new_samples += [permuted, [self.fim_split_sample]]
                curr_start_position = loc + 1  # Jump over the split token
            # Permute the segment after the last split token
            permuted = fim_permute_sequence(sequence[curr_start_position:], self.fragment_fim_rate)
            new_samples.append(permuted)
            return np.concatenate(new_samples)

        if segment_breaks.shape != (0, 1):  # then there is an EOD token in this example
            curr_start_position = 0
            new_samples = []
            for loc in np.nditer(segment_breaks):
                # Only permute non-empty segments.
                if loc - curr_start_position > 0:
                    # permute {prefix, suffix, middle} or {suffix, prefix, middle}
                    permuted = fim_split_and_permute_sequence(sample[curr_start_position:loc])
                    new_samples += [permuted, [self.eod_tok_id]]

                curr_start_position = loc + 1  # jump over the EOD token
            # Permute the segment after the last EOD
            permuted = fim_split_and_permute_sequence(sample[curr_start_position:])
            new_samples.append(permuted)

            sample = np.concatenate(new_samples)
        else:
            sample = fim_split_and_permute_sequence(sample)

        # Truncate or pad sequence to max-length
        diff = sample.shape[0] - sample_len
        if diff > 0:  # too long
            sample = sample[:sample_len]
        elif diff < 0:  # too short
            sample = np.concatenate([sample, np.full((-1 * diff), self.pad_tok_id)])

        assert sample.shape[0] == sample_len
        # end FIM-specific code
        return sample.astype(np.int64)

    def __getitem__(self, idx: int) -> Dict[str, np.ndarray]:
        # [FUNCTION DOC]
        # Summary: returns a dictionary with "input_ids" set to the sample from _get_text.
        # Inputs: idx (int). Outputs: dictionary of string to int64 array.
        # Exceptions: None explicitly raised. Side effects: as _get_text.
        text = self._get_text(idx)
        return {"input_ids": text}


@dataclass
class SubsetSplitLog:
    # [CLASS DOC]
    # Purpose: dataclass recording how a split's index mappings were built, for logging/inspection.
    # Attributes: name, data_prefix, doc_idx_filename, sample_idx_filename, shuffle_idx_filename, tokens_per_epoch,
    # num_epochs, num_samples (rows of sample_idx), seq_length.
    name: str
    data_prefix: str
    doc_idx_filename: str
    sample_idx_filename: str
    shuffle_idx_filename: str
    tokens_per_epoch: int
    num_epochs: int
    num_samples: int
    seq_length: int


def _build_index_mappings(
    name: str,
    data_prefix: str,
    documents: np.ndarray,
    sizes: np.ndarray,
    num_samples: int,
    seq_length: int,
    seed: Any,
    parallel_context: ParallelContext,
    index_mapping_dir: str = None,
    drop_last: bool = True,
    add_extra_token: int = 1,
) -> Tuple[Tuple[np.ndarray, np.ndarray, np.ndarray], SubsetSplitLog]:
    # [FUNCTION DOC]
    # HELPER: builds (if missing) and loads doc_idx, sample_idx and shuffle_idx for GPTDataset.
    # Behavior: compute tokens per epoch and the epochs needed for num_samples. Derive file names from the prefix (or
    # index_mapping_dir plus its basename) plus name, num_samples, seq_length and seed. Inside main_rank_first over the
    # world group, if any file is missing: decide separate_last_epoch (only when more than one epoch and the last epoch
    # would contain less than 80 percent of a full epoch's samples, with sanity asserts); build and save doc_idx; assert
    # int32 dtypes, import (compiling if needed) the C++ helpers and build/save sample_idx; build/save shuffle_idx where,
    # if the last epoch is separated, the first epochs and last epoch are shuffled separately. Then every rank memory-maps
    # the three files, logs counts and builds a SubsetSplitLog.
    # Inputs: name, data_prefix, documents, sizes (int32 document lengths), num_samples, seq_length, seed,
    # parallel_context, index_mapping_dir (default None), drop_last (default True), add_extra_token (default 1).
    # Outputs: tuple of (doc_idx, sample_idx, shuffle_idx) read-only memmaps and a SubsetSplitLog.
    # Exceptions: AssertionError on sample count or dtype checks; ImportError if the C++ helpers cannot be compiled.
    # Side effects: writes .npy files, may compile C++ code, barrier-style ordering from main_rank_first, logging.
    """Build doc-idx, sample-idx, and shuffle-idx.
    doc-idx: is an array (ordered) of documents to be used in training.
    sample-idx: is the start document index and document offset for each
       training sample.
    shuffle-idx: maps the sample index into a random index into sample-idx.
    """
    # Number of tokens in each epoch and number of required epochs.
    tokens_per_epoch = _num_tokens(documents, sizes)
    num_epochs = _num_epochs(tokens_per_epoch, seq_length, num_samples, add_extra_token)

    # rng state
    np_rng = np.random.RandomState(seed=seed)

    # Filename of the index mappings.
    if index_mapping_dir is not None:
        _filename = os.path.join(index_mapping_dir, os.path.basename(data_prefix))
    else:
        _filename = data_prefix
    _filename += "_{}_indexmap".format(name)
    _filename += "_{}ns".format(num_samples)
    _filename += "_{}sl".format(seq_length)
    _filename += "_{}s".format(seed)
    doc_idx_filename = _filename + "_doc_idx.npy"
    sample_idx_filename = _filename + "_sample_idx.npy"
    shuffle_idx_filename = _filename + "_shuffle_idx.npy"

    # Build the indexed mapping if not exist.
    with main_rank_first(parallel_context.world_pg):
        if (
            (not os.path.isfile(doc_idx_filename))
            or (not os.path.isfile(sample_idx_filename))
            or (not os.path.isfile(shuffle_idx_filename))
        ):
            log_rank(
                " > WARNING: could not find index map files, building " "the indices on rank 0 ...",
                logger=logger,
                level=logging.INFO,
                rank=0,
            )

            # For the last epoch, decide whether include the entire epoch
            # in the global shuffle or not.

            # If we need only one epoch, then separating last epoch  does
            # not mean anything.
            if num_epochs == 1:
                separate_last_epoch = False
                log_rank(
                    " > only one epoch required, setting " "separate_last_epoch to False",
                    logger=logger,
                    level=logging.INFO,
                    rank=0,
                )
            else:
                # Get the number of samples for the last epoch
                num_samples_from_epochs_minus_one = (
                    (num_epochs - 1) * tokens_per_epoch - add_extra_token
                ) // seq_length
                last_epoch_num_samples = num_samples - num_samples_from_epochs_minus_one
                assert last_epoch_num_samples >= 0, "last epoch number of samples should be non-negative."
                num_samples_per_epoch = (tokens_per_epoch - add_extra_token) // seq_length
                # For very small datasets, `last_epoch_num_samples` can be equal to
                # (num_samples_per_epoch + 1).
                # TODO: check that this is not problematic indeed
                #  https://github.com/bigcode-project/Megatron-LM/commit/3a6286ba11181899cccfb11d2e508eca9fd15bea
                assert last_epoch_num_samples <= (
                    num_samples_per_epoch + 1
                ), "last epoch number of samples exceeded max value."
                # If we have less than 80% of the samples for the last epoch,
                # separate out the epoch and treat it differently.
                # Note: the 80% number is just based on common sense and can
                # be adjusted if needed.
                separate_last_epoch = last_epoch_num_samples < int(0.80 * num_samples_per_epoch)
                if separate_last_epoch:
                    string = (
                        " > last epoch number of samples ({}) is smaller "
                        "than 80% of number of samples per epoch ({}), "
                        "setting separate_last_epoch to True"
                    )
                else:
                    string = (
                        " > last epoch number of samples ({}) is larger "
                        "than 80% of number of samples per epoch ({}), "
                        "setting separate_last_epoch to False"
                    )
                log_rank(
                    string.format(last_epoch_num_samples, num_samples_per_epoch),
                    logger=logger,
                    level=logging.INFO,
                    rank=0,
                )

            # doc-idx.
            start_time = time.time()
            doc_idx = _build_doc_idx(documents, num_epochs, np_rng, separate_last_epoch)
            np.save(doc_idx_filename, doc_idx, allow_pickle=True)
            log_rank(
                " > elasped time to build and save doc-idx mapping "
                "(seconds): {:4f}".format(time.time() - start_time),
                logger=logger,
                level=logging.INFO,
                rank=0,
            )

            # sample-idx.
            start_time = time.time()
            # Use C++ implementation for speed.
            # First compile and then import.
            assert doc_idx.dtype == np.int32
            assert sizes.dtype == np.int32

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

            sample_idx = helpers.build_sample_idx(
                sizes, doc_idx, seq_length, num_epochs, tokens_per_epoch, drop_last, add_extra_token
            )
            # sample_idx = _build_sample_idx(sizes, doc_idx, seq_length,
            #                              num_epochs, tokens_per_epoch, drop_last, add_extra_token)
            np.save(sample_idx_filename, sample_idx, allow_pickle=True)
            log_rank(
                " > elasped time to build and save sample-idx mapping "
                "(seconds): {:4f}".format(time.time() - start_time),
                logger=logger,
                level=logging.INFO,
                rank=0,
            )

            # shuffle-idx.
            start_time = time.time()
            # -1 is due to data structure used to retrieve the index:
            #    sample i --> [sample_idx[i], sample_idx[i+1])
            if separate_last_epoch:
                num_samples_ = num_samples_from_epochs_minus_one
            else:
                num_samples_ = sample_idx.shape[0] - 1
            shuffle_idx = _build_shuffle_idx(num_samples_, sample_idx.shape[0] - 1, np_rng)
            np.save(shuffle_idx_filename, shuffle_idx, allow_pickle=True)
            log_rank(
                " > elasped time to build and save shuffle-idx mapping"
                " (seconds): {:4f}".format(time.time() - start_time),
                logger=logger,
                level=logging.INFO,
                rank=0,
            )
    # counts = torch.cuda.LongTensor([1])
    # dist.all_reduce(counts, op=dist.ReduceOp.SUM, group=parallel_context.dp_pg)
    # dist.all_reduce(counts, op=dist.ReduceOp.SUM, group=parallel_context.pp_pg)
    # assert counts[0].item() == (
    #     dist.get_world_size()
    #     // dist.get_world_size(group=parallel_context.tp_pg)
    # )

    # Load mappings.
    start_time = time.time()
    log_rank(" > loading doc-idx mapping from {}".format(doc_idx_filename), logger=logger, level=logging.INFO, rank=0)
    doc_idx: np.ndarray = np.load(doc_idx_filename, allow_pickle=True, mmap_mode="r")
    log_rank(
        " > loading sample-idx mapping from {}".format(sample_idx_filename), logger=logger, level=logging.INFO, rank=0
    )
    sample_idx: np.ndarray = np.load(sample_idx_filename, allow_pickle=True, mmap_mode="r")
    log_rank(
        " > loading shuffle-idx mapping from {}".format(shuffle_idx_filename),
        logger=logger,
        level=logging.INFO,
        rank=0,
    )
    shuffle_idx: np.ndarray = np.load(shuffle_idx_filename, allow_pickle=True, mmap_mode="r")
    log_rank(
        "    loaded indexed file in {:3.3f} seconds".format(time.time() - start_time),
        logger=logger,
        level=logging.INFO,
        rank=0,
    )
    log_rank("    total number of samples: {}".format(sample_idx.shape[0]), logger=logger, level=logging.INFO, rank=0)
    log_rank("    total number of epochs: {}".format(num_epochs), logger=logger, level=logging.INFO, rank=0)

    subset_log = SubsetSplitLog(
        name=name,
        data_prefix=data_prefix,
        doc_idx_filename=doc_idx_filename,
        sample_idx_filename=sample_idx_filename,
        shuffle_idx_filename=shuffle_idx_filename,
        tokens_per_epoch=tokens_per_epoch,
        num_epochs=num_epochs,
        num_samples=sample_idx.shape[0],
        seq_length=seq_length,
    )

    return (doc_idx, sample_idx, shuffle_idx), subset_log


def _num_tokens(documents: np.ndarray, sizes: np.ndarray) -> int:
    # [FUNCTION DOC]
    # HELPER: total tokens of the selected documents, the sum of sizes at the document ids.
    # Inputs: documents (int array), sizes (int array). Outputs: integer. Exceptions: None explicitly raised. Side effects: None.
    """Total number of tokens in the dataset."""
    return np.sum(sizes[documents])


def _num_epochs(tokens_per_epoch: int, seq_length: int, num_samples: int, add_extra_token: int = 1) -> int:
    # [FUNCTION DOC]
    # HELPER: smallest number of epochs whose tokens, minus add_extra_token, integer-divided by seq_length reach num_samples.
    # Inputs: tokens_per_epoch, seq_length, num_samples, add_extra_token (default 1).
    # Outputs: int.
    # Exceptions: None explicitly raised (loops forever if tokens_per_epoch is 0).
    # Side effects: None.
    """Based on number of samples and sequence length, calculate how many
    epochs will be needed."""
    num_epochs = 0
    total_tokens = 0
    while True:
        num_epochs += 1
        total_tokens += tokens_per_epoch
        # -1 is because we need to retrieve seq_length + 1 token each time
        # but the last token will overlap with the first token of the next
        # sample except for the last sample.
        if ((total_tokens - add_extra_token) // seq_length) >= num_samples:
            return num_epochs


def _build_doc_idx(documents: np.ndarray, num_epochs: int, np_rng: Any, separate_last_epoch: bool) -> np.ndarray:
    # [FUNCTION DOC]
    # HELPER: builds the shuffled document order over epochs.
    # Behavior: unless separate_last_epoch (and more than one epoch), tile documents num_epochs times, cast to int32 and
    # shuffle all together. Otherwise build the first num_epochs - 1 epochs and the last epoch separately (each shuffled)
    # and concatenate, so the last partial epoch does not mix with the others.
    # Inputs: documents, num_epochs, np_rng (RandomState), separate_last_epoch (bool).
    # Outputs: 1-D int32 array.
    # Exceptions: None explicitly raised. Side effects: advances np_rng.
    """Build an array with length = number-of-epochs * number-of-dcuments.
    Each index is mapped to a corresponding document."""
    if not separate_last_epoch or num_epochs == 1:
        doc_idx = np.mgrid[0:num_epochs, 0 : len(documents)][1]
        doc_idx[:] = documents
        doc_idx = doc_idx.reshape(-1)
        doc_idx = doc_idx.astype(np.int32)
        np_rng.shuffle(doc_idx)
        return doc_idx

    doc_idx_first = _build_doc_idx(documents, num_epochs - 1, np_rng, False)
    doc_idx_last = _build_doc_idx(documents, 1, np_rng, False)
    return np.concatenate((doc_idx_first, doc_idx_last))


def _build_sample_idx(sizes, doc_idx, seq_length, num_epochs, tokens_per_epoch, drop_last=True, add_extra_token=1):
    # [FUNCTION DOC]
    # HELPER: pure-Python reference of the C++ sample index builder (unused; kept for reference).
    # Behavior: number of samples is (num_epochs times tokens_per_epoch minus add_extra_token) divided by seq_length,
    # rounded down when drop_last else up. Starting at document position 0 offset 0, repeatedly consume seq_length plus
    # add_extra_token tokens across documents in doc_idx order, recording the (doc position, offset) where each sample ends
    # so that consecutive samples overlap by add_extra_token tokens. On the last document without enough tokens the final
    # offset is clamped and the loop ends.
    # Inputs: sizes, doc_idx, seq_length, num_epochs, tokens_per_epoch, drop_last (default True), add_extra_token (default 1).
    # Outputs: int32 array of shape number of samples plus one by 2.
    # Exceptions: AssertionError if documents run out before the last sample.
    # Side effects: None.
    """Sample index mapping is a 2D array with sizes
    [number-of-samples + 1, 2] where [..., 0] contains
    the index into `doc_idx` and [..., 1] is the
    starting offset in that document."""

    # Total number of samples. For -1 see comments in `_num_epochs`.
    if not drop_last:
        num_samples = -(-(num_epochs * tokens_per_epoch - add_extra_token) // seq_length)
    else:
        num_samples = (num_epochs * tokens_per_epoch - add_extra_token) // seq_length
    sample_idx = np.zeros([num_samples + 1, 2], dtype=np.int32)

    # Index into sample_idx.
    sample_index = 0
    # Index into doc_idx.
    doc_idx_index = 0
    # Beginning offset for each document.
    doc_offset = 0
    # Start with first document and no offset.
    sample_idx[sample_index][0] = doc_idx_index
    sample_idx[sample_index][1] = doc_offset
    sample_index += 1
    while sample_index <= num_samples:
        # Start with a fresh sequence.
        remaining_seq_length = seq_length + add_extra_token
        while remaining_seq_length != 0:
            # Get the document length.
            doc_id = doc_idx[doc_idx_index]
            doc_length = sizes[doc_id] - doc_offset
            # And add it to the current sequence.
            remaining_seq_length -= doc_length
            # If we have more than a full sequence, adjust offset and set
            # remaining length to zero so we return from the while loop.
            # Note that -1 here is for the same reason we have -1 in
            # `_num_epochs` calculations.
            if remaining_seq_length <= 0:
                doc_offset += remaining_seq_length + doc_length - add_extra_token
                remaining_seq_length = 0
            else:
                # Otherwise, start from the beginning of the next document.
                if doc_idx_index == (len(doc_idx) - 1):
                    assert (
                        sample_index == num_samples
                    ), f"sample_index={sample_index} and num_samples={num_samples} should be the same"
                    doc_offset = sizes[doc_idx[doc_idx_index]] - add_extra_token
                    break
                doc_idx_index += 1
                doc_offset = 0
        # Record the sequence.
        sample_idx[sample_index][0] = doc_idx_index
        sample_idx[sample_index][1] = doc_offset
        sample_index += 1

    return sample_idx


def _build_shuffle_idx(num_samples: int, total_size: int, np_rng: Any) -> np.ndarray:
    # [FUNCTION DOC]
    # HELPER: returns a permutation of 0 to total_size - 1 where the first num_samples ids and the remaining ids are
    # shuffled separately and concatenated. Uses uint32 unless total_size is near the uint32 limit, then int64.
    # Inputs: num_samples, total_size, np_rng.
    # Outputs: 1-D integer array of length total_size.
    # Exceptions: None explicitly raised. Side effects: logs; advances np_rng.
    """Build the range [0, size) and shuffle."""
    logger.info(
        " > building shuffle index with split [0, {}) and [{}, {}) "
        "...".format(num_samples, num_samples, total_size),
    )

    dtype_ = np.uint32
    if total_size >= (np.iinfo(np.uint32).max - 1):
        dtype_ = np.int64

    shuffle_idx_first = np.arange(start=0, stop=num_samples, step=1, dtype=dtype_)
    np_rng.shuffle(shuffle_idx_first)
    if num_samples == total_size:
        return shuffle_idx_first

    shuffle_idx_last = np.arange(start=num_samples, stop=total_size, step=1, dtype=dtype_)
    np_rng.shuffle(shuffle_idx_last)

    return np.concatenate((shuffle_idx_first, shuffle_idx_last))


# From https://github.com/EleutherAI/gpt-neox/blob/FIM-clean/megatron/data/gpt2_dataset.py#L339
def permute(
    sample: np.ndarray,
    np_rng: np.random.Generator,
    fim_rate: float,
    fim_spm_rate: float,
    tokenizer: PreTrainedTokenizerBase,
    truncate_or_pad: bool = True,
    suffix_tok_id: int = None,
    prefix_tok_id: int = None,
    middle_tok_id: int = None,
    pad_tok_id: int = None,
    no_fim_prefix: str = None,
):
    # [FUNCTION DOC]
    # Summary: Fill-In-the-Middle transform of one token sequence (from GPT-NeoX).
    # Behavior: with probability fim_rate: decode the tokens to text; return unchanged if it starts with no_fim_prefix;
    # pick two sorted random character boundaries to split into prefix, middle, suffix; re-encode each. If truncate_or_pad,
    # adjust the suffix so the total including 3 sentinel tokens equals the original length (pad with pad_tok_id, or
    # truncate; if the suffix is too short to truncate, give up). Then with probability fim_spm_rate produce SPM order
    # (prefix token, suffix token, suffix, middle token, prefix, middle) else PSM order (prefix token, prefix, suffix
    # token, suffix, middle token, middle). Without FIM the sample is returned unchanged.
    # Inputs: sample (1-D int array), np_rng (RandomState), fim_rate, fim_spm_rate (probabilities), tokenizer,
    # truncate_or_pad (default True), suffix_tok_id, prefix_tok_id, middle_tok_id, pad_tok_id (ints), no_fim_prefix
    # (optional string).
    # Outputs: 1-D array of tokens.
    # Exceptions: re-raises ValueError from boundary sampling after printing diagnostics.
    # Side effects: advances np_rng; prints on error.
    # KNOWN ISSUE: when the suffix cannot be truncated it returns a tuple (sample, np_rng) instead of an array.
    """
    Take in a sample (np array w/ size (0,chunklength)) and perform a FIM transformation on it.
    Maintain the same sample length (if transform creates a few extra tokens, drop them).
    """
    if np_rng.binomial(1, fim_rate):  # sample bernoulli dist
        contents = tokenizer.decode(sample)

        # Do not apply FIM if the sample starts with no_fim_prefix
        if no_fim_prefix is not None and contents.startswith(no_fim_prefix):
            return sample

        try:
            # A boundary can be =0 (prefix will be empty)
            # a boundary can be =len(contents) (suffix will be empty)
            # The two boundaries can be equal (middle will be empty)
            boundaries = list(np_rng.randint(low=0, high=len(contents) + 1, size=2))
            boundaries.sort()
        except ValueError as e:
            print(len(contents), contents)
            print(e)
            raise e

        prefix = contents[: boundaries[0]]
        middle = contents[boundaries[0] : boundaries[1]]
        suffix = contents[boundaries[1] :]

        prefix = tokenizer.encode(prefix, return_tensors="np").squeeze(axis=0)
        middle = tokenizer.encode(middle, return_tensors="np").squeeze(axis=0)
        suffix = tokenizer.encode(suffix, return_tensors="np").squeeze(axis=0)

        # here we truncate each given segment to fit the same length as it was before
        # A consequence is that we never reach the end of a file?
        # we should rather truncate at the context-level
        if truncate_or_pad:
            # need to make same length as the input. Take the 3 sentinel tokens into account
            new_length = suffix.shape[0] + prefix.shape[0] + middle.shape[0] + 3
            diff = new_length - sample.shape[0]
            if diff > 0:  # too long
                if (
                    suffix.shape[0] <= diff
                ):  # if there's no space to truncate the suffix: stop and report it. atm i should have stopped this from happening
                    return sample, np_rng
                suffix = suffix[: suffix.shape[0] - diff]
            elif diff < 0:  # too short
                suffix = np.concatenate([suffix, np.full((-1 * diff), pad_tok_id)])

        if np_rng.binomial(1, fim_spm_rate):
            # SPM (variant 2 from FIM paper)
            new_sample = np.concatenate([[prefix_tok_id, suffix_tok_id], suffix, [middle_tok_id], prefix, middle])
        else:
            # PSM
            new_sample = np.concatenate([[prefix_tok_id], prefix, [suffix_tok_id], suffix, [middle_tok_id], middle])

    else:
        # don't do FIM preproc
        new_sample = sample

    return new_sample
