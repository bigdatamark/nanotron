# [MODULE DOC]
# Purpose: Megatron/fairseq style memory-mapped token dataset: a .bin file of concatenated token arrays and a .idx
# file describing them, plus a builder to write such files.
# Index file layout: 9-byte magic "MMIDIDX" plus two zero bytes, little-endian uint64 version 1, uint8 dtype code
# (see dtypes), uint64 number of items, uint64 number of document-index entries, then int32 sizes, int64 byte
# pointers into the .bin file, and int64 document index (item ids where each document starts).
# Main contents: __best_fitting_dtype, make_builder, deallocate_indexed_dataset_memory, make_indexed_dataset, dtypes
# table and code, index_file_path, data_file_path, _warmup_mmap_file, MMapIndexedDataset (with nested Index),
# MMapIndexedDatasetBuilder.
# Fits in nanotron: storage layer used by nanotron.data.nemo_dataset.GPTDataset.
# Import-time side effects: creates a module logger and the dtypes table.
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

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Most of the code here has been copied from:
#   fairseq/fairseq/data/indexed_dataset.py

# with some modifications:

# Removed IndexedRawTextDataset since it relied on Fairseq dictionary
# other slight modifications to remove fairseq dependencies
# Added document index to index file and made it accessible.
#    An empty sentence no longer separates documents.

import os
import shutil
import struct
from functools import lru_cache
from itertools import accumulate
from typing import Tuple

import numpy as np
import torch

from nanotron import logging
from nanotron.logging import log_rank

logger = logging.get_logger(__name__)
# logging.getLogger('botocore').setLevel(logging.WARNING)


def __best_fitting_dtype(vocab_size=None):
    # [FUNCTION DOC]
    # HELPER: chooses the token storage dtype for the builder: uint16 when vocab_size is given and below 65500, else int32.
    # Inputs: vocab_size (optional int). Outputs: numpy dtype class.
    # Exceptions: None explicitly raised. Side effects: None.
    if vocab_size is not None and vocab_size < 65500:
        return np.uint16
    else:
        return np.int32


def make_builder(out_file, vocab_size=None, chunk_size=64, pad_id=0, retrieval_db=False, stride=64):
    # [FUNCTION DOC]
    # Summary: returns an MMapIndexedDatasetBuilder writing to out_file with the best-fitting dtype for vocab_size.
    # Inputs: out_file (path of the .bin file), vocab_size (optional); chunk_size, pad_id, retrieval_db, stride are
    # accepted but unused.
    # Outputs: MMapIndexedDatasetBuilder.
    # Exceptions: OSError if the file cannot be opened.
    # Side effects: opens/creates out_file for writing.
    return MMapIndexedDatasetBuilder(out_file, dtype=__best_fitting_dtype(vocab_size))


def deallocate_indexed_dataset_memory(indexed_dataset):
    # [FUNCTION DOC]
    # Summary: drops the document index of an MMapIndexedDataset, or the sizes and doc_idx of other dataset types, to free
    # memory.
    # Inputs: indexed_dataset. Outputs: None. Exceptions: None explicitly raised.
    # Side effects: mutates the dataset.
    """Deallocate memory of an IndexedDataset."""
    if isinstance(indexed_dataset, MMapIndexedDataset):
        # for MMapIndexedDataset we cannot release any memory of sizes
        indexed_dataset._index._doc_idx = None
    else:
        indexed_dataset.sizes = None
        indexed_dataset.doc_idx = None


def make_indexed_dataset(path: str, skip_warmup: bool = False):
    # [FUNCTION DOC]
    # Summary: opens an MMapIndexedDataset for a path prefix after checking both .idx and .bin exist.
    # Inputs: path (prefix without extension), skip_warmup (default False).
    # Outputs: MMapIndexedDataset.
    # Exceptions: ValueError if the files are missing.
    # Side effects: file reads (memory mapping, optional warm-up).
    # KNOWN ISSUE: the existence check is repeated, so the final raise is unreachable.
    # now handle bin memap
    if not MMapIndexedDataset.exists(path):
        raise ValueError(
            f"Dataset does not exist: {path}"
            f"Path should be a basename that both .idx and .bin can be appended to get full filenames."
        )
    if MMapIndexedDataset.exists(path):
        return MMapIndexedDataset(path, skip_warmup)
    raise ValueError("MMapIndexedDataset doesn't exist")


dtypes = {1: np.uint8, 2: np.int8, 3: np.int16, 4: np.int32, 5: np.int64, 6: np.float64, 7: np.double, 8: np.uint16}


def code(dtype):
    # [FUNCTION DOC]
    # HELPER: reverse lookup in dtypes; returns the integer code for a numpy dtype.
    # Inputs: dtype. Outputs: int.
    # Exceptions: ValueError if the dtype is not in the table.
    # Side effects: None.
    for k in dtypes.keys():
        if dtypes[k] == dtype:
            return k
    raise ValueError(dtype)


def index_file_path(prefix_path: str) -> str:
    # [FUNCTION DOC]
    # HELPER: appends ".idx" to a prefix. Inputs: prefix_path (string). Outputs: string.
    # Exceptions: None explicitly raised. Side effects: None.
    return prefix_path + ".idx"


def data_file_path(prefix_path: str) -> str:
    # [FUNCTION DOC]
    # HELPER: appends ".bin" to a prefix. Inputs: prefix_path (string). Outputs: string.
    # Exceptions: None explicitly raised. Side effects: None.
    return prefix_path + ".bin"


def _warmup_mmap_file(path: str):
    # [FUNCTION DOC]
    # HELPER: reads a file sequentially in 100 MiB chunks to pull it into the OS page cache.
    # Inputs: path (string). Outputs: None.
    # Exceptions: OSError on read errors. Side effects: file I/O.
    with open(path, "rb") as stream:
        while stream.read(100 * 1024 * 1024):
            pass


class MMapIndexedDataset(torch.utils.data.Dataset):
    # [CLASS DOC]
    # Purpose: read-only torch Dataset over a .bin/.idx pair; item i is the token array of the i-th sentence/item.
    # Key attributes: _path (prefix), _index (nested Index), _bin_buffer_mmap (numpy memmap of the .bin file),
    # _bin_buffer (memoryview over it).
    # Inherits from torch.utils.data.Dataset.
    # KNOWN ISSUE: __getstate__ returns only the path and there is no __setstate__ (commented out), so pickling to
    # dataloader workers yields an unusable object.
    class Index(object):
        # [CLASS DOC]
        # Purpose: parsed view of an .idx file with sizes, pointers and document index arrays read via memory map; also
        # provides a writer for producing index files.
        # Key attributes: _HDR_MAGIC, _dtype, _dtype_size, _len (number of items), _bin_buffer_mmap, _bin_buffer, _sizes,
        # _pointers, _doc_idx.
        _HDR_MAGIC = b"MMIDIDX\x00\x00"

        @classmethod
        def writer(cls, path, dtype):
            # [FUNCTION DOC]
            # Summary: returns a context manager object that writes an index file at path for token dtype.
            # Inputs: cls, path (string), dtype (numpy dtype class).
            # Outputs: a _Writer instance.
            # Exceptions: None explicitly raised. Side effects: None until entered.
            class _Writer(object):
                # [CLASS DOC]
                # HELPER: context manager writing the index file layout described in the module block.
                # Attributes: _file (open binary file, set on enter).
                def __enter__(self):
                    # [FUNCTION DOC]
                    # HELPER: opens path for writing and writes the magic, version 1 and the dtype code.
                    # Inputs: None. Outputs: self.
                    # Exceptions: OSError if the file cannot be opened; ValueError for an unknown dtype.
                    # Side effects: creates/truncates the file.
                    self._file = open(path, "wb")

                    self._file.write(cls._HDR_MAGIC)
                    self._file.write(struct.pack("<Q", 1))
                    self._file.write(struct.pack("<B", code(dtype)))

                    return self

                @staticmethod
                def _get_pointers(sizes):
                    # [FUNCTION DOC]
                    # HELPER: computes byte offsets of each item in the .bin file as the running sum of size times dtype item size.
                    # Inputs: sizes (sequence of ints). Outputs: list of ints.
                    # Exceptions: None explicitly raised. Side effects: None.
                    dtype_size = dtype().itemsize
                    address = 0
                    pointers = []

                    for size in sizes:
                        pointers.append(address)
                        address += size * dtype_size

                    return pointers

                def write(self, sizes, doc_idx):
                    # [FUNCTION DOC]
                    # HELPER: writes item count, document-index count, then int32 sizes, int64 pointers and int64 doc_idx.
                    # Inputs: sizes (list of ints), doc_idx (list of ints). Outputs: None.
                    # Exceptions: None explicitly raised. Side effects: file writes.
                    pointers = self._get_pointers(sizes)

                    self._file.write(struct.pack("<Q", len(sizes)))
                    self._file.write(struct.pack("<Q", len(doc_idx)))

                    sizes = np.array(sizes, dtype=np.int32)
                    self._file.write(sizes.tobytes(order="C"))
                    del sizes

                    pointers = np.array(pointers, dtype=np.int64)
                    self._file.write(pointers.tobytes(order="C"))
                    del pointers

                    doc_idx = np.array(doc_idx, dtype=np.int64)
                    self._file.write(doc_idx.tobytes(order="C"))

                def __exit__(self, exc_type, exc_val, exc_tb):
                    # [FUNCTION DOC]
                    # HELPER: closes the file. Inputs: exception info (ignored). Outputs: None (exceptions propagate).
                    # Exceptions: None explicitly raised. Side effects: closes the file.
                    self._file.close()

            return _Writer()

        def __init__(self, path: str, skip_warmup: bool = False):
            # [FUNCTION DOC]
            # Summary: parses the header of an index file and memory-maps its arrays.
            # Behavior: read and assert the magic and version 1, read the dtype code and item count, remember the current offset.
            # Unless skip_warmup, warm up the file. Memory-map the whole file and read sizes (int32, count items), pointers (int64)
            # and the document index right after them.
            # Inputs: path (.idx path), skip_warmup (default False).
            # Outputs: None.
            # Exceptions: AssertionError for wrong magic/version; AttributeError (see issue).
            # Side effects: file reads, memory mapping, rank 0 logging.
            # KNOWN ISSUE: the document-index count written by the writer is never read and self._doc_count is never assigned,
            # so reading the document index raises AttributeError, and sizes/pointers are read 8 bytes too early.
            with open(path, "rb") as stream:
                magic_test = stream.read(9)
                assert self._HDR_MAGIC == magic_test, (
                    "Index file doesn't match expected format. "
                    "Make sure that --dataset-impl is configured properly."
                )
                version = struct.unpack("<Q", stream.read(8))
                assert (1,) == version

                (dtype_code,) = struct.unpack("<B", stream.read(1))
                self._dtype = dtypes[dtype_code]
                self._dtype_size = self._dtype().itemsize

                self._len = struct.unpack("<Q", stream.read(8))[0]
                offset = stream.tell()

            if not skip_warmup:
                log_rank("    warming up index mmap file...", logger=logger, level=logging.INFO, rank=0)
                _warmup_mmap_file(path)

            # self._bin_buffer_mmap = np.memmap(path, mode='r', order='C')
            self._bin_buffer_mmap = np.memmap(path, mode="r", order="C")
            self._bin_buffer = memoryview(self._bin_buffer_mmap)
            log_rank("    reading sizes...", logger=logger, level=logging.INFO, rank=0)
            self._sizes = np.frombuffer(self._bin_buffer, dtype=np.int32, count=self._len, offset=offset)
            log_rank("    reading pointers...", logger=logger, level=logging.INFO, rank=0)
            self._pointers = np.frombuffer(
                self._bin_buffer, dtype=np.int64, count=self._len, offset=offset + self._sizes.nbytes
            )
            log_rank("    reading document index...", logger=logger, level=logging.INFO, rank=0)
            self._doc_idx = np.frombuffer(
                self._bin_buffer,
                dtype=np.int64,
                count=self._doc_count,
                offset=offset + self._sizes.nbytes + self._pointers.nbytes,
            )

        def __del__(self):
            # [FUNCTION DOC]
            # HELPER: closes the underlying mmap when the object is destroyed.
            # Inputs: None. Outputs: None.
            # Exceptions: AttributeError if construction failed before the mmap was created.
            # Side effects: releases the memory map.
            if self._bin_buffer_mmap is not None:
                self._bin_buffer_mmap._mmap.close()
                del self._bin_buffer_mmap

        @property
        def dtype(self):
            # [FUNCTION DOC]
            # Summary: property, numpy dtype of tokens. Inputs: None. Outputs: dtype class. Exceptions: None explicitly raised.
            # Side effects: None.
            return self._dtype

        @property
        def sizes(self):
            # [FUNCTION DOC]
            # Summary: property, int32 array of item lengths in tokens. Inputs: None. Outputs: array.
            # Exceptions: None explicitly raised. Side effects: None.
            return self._sizes

        @property
        def doc_idx(self):
            # [FUNCTION DOC]
            # Summary: property, int64 array of document start item ids (or None after deallocation). Inputs: None.
            # Outputs: array or None. Exceptions: None explicitly raised. Side effects: None.
            return self._doc_idx

        @lru_cache(maxsize=8)
        def __getitem__(self, i: int) -> Tuple[int, int]:
            # [FUNCTION DOC]
            # Summary: returns (byte pointer, size) of item i; cached in an LRU cache of 8 entries.
            # Inputs: i (int). Outputs: tuple of two integers.
            # Exceptions: IndexError if out of range. Side effects: populates the cache (which keeps self alive).
            return self._pointers[i], self._sizes[i]

        def __len__(self):
            # [FUNCTION DOC]
            # Summary: number of items. Inputs: None. Outputs: int. Exceptions: None explicitly raised. Side effects: None.
            return self._len

    def __init__(self, path: str, skip_warmup: bool = False):
        # [FUNCTION DOC]
        # Summary: initialises attributes to None and opens the dataset via _do_init.
        # Inputs: path (prefix), skip_warmup (default False). Outputs: None.
        # Exceptions: propagates from _do_init. Side effects: file reads and memory mapping.
        super().__init__()

        self._path = None
        self._index = None
        self._bin_buffer = None

        self._do_init(path, skip_warmup)

    def __getstate__(self):
        # [FUNCTION DOC]
        # Summary: pickling support returning only the path. Inputs: None. Outputs: string.
        # Exceptions: None explicitly raised. Side effects: None.
        return self._path

    # def __setstate__(self, state):
    #     self._do_init(state)

    def _do_init(self, path: str, skip_warmup: bool):
        # [FUNCTION DOC]
        # HELPER: opens the index, optionally warms up the .bin file, and memory-maps it with a memoryview.
        # Inputs: path (prefix), skip_warmup (bool). Outputs: None.
        # Exceptions: OSError for missing files; errors from Index.
        # Side effects: sets _path, _index, _bin_buffer_mmap, _bin_buffer; file reads; logging.
        self._path = path
        self._index = self.Index(index_file_path(self._path), skip_warmup)

        if not skip_warmup:
            log_rank("    warming up data mmap file...", logger=logger, level=logging.INFO, rank=0)
            _warmup_mmap_file(data_file_path(self._path))
        log_rank("    creating numpy buffer of mmap...", logger=logger, level=logging.INFO, rank=0)
        self._bin_buffer_mmap = np.memmap(data_file_path(self._path), mode="r", order="C")
        log_rank("    creating memory view of numpy buffer...", logger=logger, level=logging.INFO, rank=0)
        self._bin_buffer = memoryview(self._bin_buffer_mmap)

    def __del__(self):
        # [FUNCTION DOC]
        # HELPER: closes the data mmap and deletes the index on destruction.
        # Inputs: None. Outputs: None.
        # Exceptions: AttributeError if _do_init failed before the mmap was created.
        # Side effects: releases memory maps.
        if self._bin_buffer_mmap is not None:
            self._bin_buffer_mmap._mmap.close()
            del self._bin_buffer_mmap
        del self._index

    def __len__(self):
        # [FUNCTION DOC]
        # Summary: number of items in the index. Inputs: None. Outputs: int. Exceptions: None explicitly raised.
        # Side effects: None.
        return len(self._index)

    # @lru_cache(maxsize=8)
    def __getitem__(self, idx):
        # [FUNCTION DOC]
        # Summary: integer index returns that item's token array (zero-copy view); a contiguous slice returns a list of arrays
        # for the items in the slice, read with one buffer access then split by cumulative sizes.
        # Inputs: idx (int or slice).
        # Outputs: numpy array, list of arrays, or None for other index types.
        # Exceptions: ValueError for slices with step other than 1.
        # Side effects: None (reads mapped memory).
        if isinstance(idx, int):
            ptr, size = self._index[idx]
            np_array = np.frombuffer(self._bin_buffer, dtype=self._index.dtype, count=size, offset=ptr)
            return np_array
        elif isinstance(idx, slice):
            start, stop, step = idx.indices(len(self))
            if step != 1:
                raise ValueError("Slices into indexed_dataset must be contiguous")
            ptr = self._index._pointers[start]
            sizes = self._index._sizes[idx]
            offsets = list(accumulate(sizes))
            total_size = sum(sizes)
            np_array = np.frombuffer(self._bin_buffer, dtype=self._index.dtype, count=total_size, offset=ptr)
            sents = np.split(np_array, offsets[:-1])
            return sents

    def get(self, idx, offset=0, length=None):
        # [FUNCTION DOC]
        # Summary: returns a window of item idx starting at token offset with length tokens (default: to the end of item).
        # Inputs: idx (int), offset (default 0), length (optional int).
        # Outputs: numpy array view.
        # Exceptions: None explicitly raised (numpy raises if the window exceeds the buffer).
        # Side effects: None.
        """Retrieves a single item from the dataset with the option to only
        return a portion of the item.

        get(idx) is the same as [idx] but get() does not support slicing.
        """
        ptr, size = self._index[idx]
        if length is None:
            length = size - offset
        ptr += offset * np.dtype(self._index.dtype).itemsize
        np_array = np.frombuffer(self._bin_buffer, dtype=self._index.dtype, count=length, offset=ptr)
        return np_array

    @property
    def sizes(self):
        # [FUNCTION DOC]
        # Summary: property, item sizes from the index. Inputs: None. Outputs: int32 array.
        # Exceptions: None explicitly raised. Side effects: None.
        return self._index.sizes

    @property
    def doc_idx(self):
        # [FUNCTION DOC]
        # Summary: property, document index from the index. Inputs: None. Outputs: array or None.
        # Exceptions: None explicitly raised. Side effects: None.
        return self._index.doc_idx

    def get_doc_idx(self):
        # [FUNCTION DOC]
        # Summary: returns the index's document index array. Inputs: None. Outputs: array or None.
        # Exceptions: None explicitly raised. Side effects: None.
        return self._index._doc_idx

    def set_doc_idx(self, doc_idx_):
        # [FUNCTION DOC]
        # Summary: replaces the index's document index. Inputs: doc_idx_ (array). Outputs: None.
        # Exceptions: None explicitly raised. Side effects: mutates the index.
        self._index._doc_idx = doc_idx_

    @property
    def supports_prefetch(self):
        # [FUNCTION DOC]
        # Summary: property, always False. Inputs: None. Outputs: bool. Exceptions: None explicitly raised. Side effects: None.
        return False

    @staticmethod
    def exists(path):
        # [FUNCTION DOC]
        # Summary: static method; True when both prefix.idx and prefix.bin exist; logs the paths at debug level.
        # Inputs: path (prefix). Outputs: bool. Exceptions: None explicitly raised. Side effects: debug logging.
        logger.debug(
            f"Checking file path: {path}, index_file_path: {index_file_path(path)}, data_file_path: {data_file_path(path)}"
        )
        return os.path.exists(index_file_path(path)) and os.path.exists(data_file_path(path))

    def deallocate_indexed_dataset_memory(self):
        # [FUNCTION DOC]
        # Summary: sets the index's document index to None to release memory.
        # Inputs: None. Outputs: None. Exceptions: None explicitly raised. Side effects: mutates the index.
        """Deallocate memory of an IndexedDataset."""
        self._index._doc_idx = None


class MMapIndexedDatasetBuilder(object):
    # [CLASS DOC]
    # Purpose: writes a .bin data file incrementally and finally its .idx index.
    # Key attributes: _data_file (open binary file), _dtype, _sizes (list of item lengths), _doc_idx (document start item
    # ids, starting with 0).
    def __init__(self, out_file, dtype=np.int64):
        # [FUNCTION DOC]
        # Summary: opens out_file for writing and initialises the bookkeeping lists.
        # Inputs: out_file (path), dtype (default int64). Outputs: None.
        # Exceptions: OSError if the file cannot be opened. Side effects: creates/truncates out_file.
        self._data_file = open(out_file, "wb")
        self._dtype = dtype
        self._sizes = []
        self._doc_idx = [0]

    def add_item(self, tensor):
        # [FUNCTION DOC]
        # Summary: converts a torch tensor to a numpy array of the builder dtype, appends its bytes to the data file and
        # records its size.
        # Inputs: tensor (1-D torch tensor of token ids). Outputs: None.
        # Exceptions: None explicitly raised. Side effects: file write, mutates _sizes.
        np_array = np.array(tensor.numpy(), dtype=self._dtype)
        self._data_file.write(np_array.tobytes(order="C"))
        self._sizes.append(np_array.size)

    def end_document(self):
        # [FUNCTION DOC]
        # Summary: marks the end of a document by appending the current item count to _doc_idx.
        # Inputs: None. Outputs: None. Exceptions: None explicitly raised. Side effects: mutates _doc_idx.
        self._doc_idx.append(len(self._sizes))

    def merge_file_(self, another_file):
        # [FUNCTION DOC]
        # Summary: appends another indexed dataset (prefix another_file) to this builder: its sizes are appended and its .bin
        # bytes copied into the data file.
        # Inputs: another_file (prefix). Outputs: None.
        # Exceptions: AssertionError if dtypes differ; errors from Index.
        # Side effects: file reads/writes.
        # KNOWN ISSUE: the other file's document index is not merged, so document boundaries of merged data are lost.
        # Concatenate index
        index = MMapIndexedDataset.Index(index_file_path(another_file))
        assert index.dtype == self._dtype

        for size in index.sizes:
            self._sizes.append(size)

        # Concatenate data
        with open(data_file_path(another_file), "rb") as f:
            shutil.copyfileobj(f, self._data_file)

    def finalize(self, index_file):
        # [FUNCTION DOC]
        # Summary: closes the data file and writes the index file with the collected sizes and document index.
        # Inputs: index_file (path of the .idx file). Outputs: None.
        # Exceptions: OSError on write failure. Side effects: closes and writes files.
        self._data_file.close()

        with MMapIndexedDataset.Index.writer(index_file, self._dtype) as index:
            index.write(self._sizes, self._doc_idx)
