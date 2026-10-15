# [MODULE DOC]
# Purpose: streaming "tokenized bytes" datasets (raw token files, .ds, on S3 or local disk), their blending into a
# training dataset, and the Megatron-sampler based dataloader used for multi-stage training with resume.
# Main contents: log dataclasses (TBFileDatasetLog, TBFolderDatasetLog, TrainDataLog), TokenizedBytesFileDataset
# (one file streamed sequentially), OldTokenizedBytesFolderDataset (legacy folder of files), TokenizedBytesFolderDataset
# (datatrove based, with S3 file-list caching), build_dataset, get_tb_datasets, get_tb_dataloader.
# Fits in nanotron: alternative to Nanoset for pretraining from S3-hosted shards; get_tb_datasets builds a
# BlendableDataset and get_tb_dataloader wraps it with a MegatronPretrainingSampler and the CLM collators.
# Import-time side effects: creates a module logger; sets tb_logger_available.
# KNOWN ISSUE: the try block guarding tb_logger_available imports nothing, so the flag is always True.
import os
import re
import time
from bisect import bisect
from dataclasses import dataclass
from re import Pattern
from typing import Dict, Optional, Tuple, Union

import numpy as np
import torch
from datatrove.utils.dataset import DatatroveFolderDataset
from torch.utils.data import DataLoader, Dataset

from nanotron import distributed as dist
from nanotron import logging
from nanotron.config import NanosetDatasetsArgs
from nanotron.data import DataCollatorForCLM, DataCollatorForCLMWithPositionIds, EmptyInfiniteDataset
from nanotron.data.dataloader import get_dataloader_worker_init
from nanotron.data.nemo_dataset import BlendableDataset
from nanotron.data.nemo_dataset.dataset_utils import compile_helper
from nanotron.data.s3_utils import BOTO3_AVAILABLE, _get_s3_file_list, _get_s3_object, _stream_file
from nanotron.data.samplers import MegatronPretrainingSampler
from nanotron.logging import human_format, log_rank
from nanotron.parallel import ParallelContext
from nanotron.utils import main_rank_first

try:
    tb_logger_available = True
except ImportError:
    tb_logger_available = False

logger = logging.get_logger(__name__)


@dataclass
class TBFileDatasetLog:
    # [CLASS DOC]
    # Purpose: dataclass logging a single-file dataset: dataset_type (class name), file_path, seq_len, dtype name,
    # skip_in_stream, num_samples, num_tokens, human_format_num_tokens, num_epochs.
    dataset_type: str
    file_path: str
    seq_len: int
    dtype: str
    skip_in_stream: bool
    num_samples: int
    num_tokens: int
    human_format_num_tokens: str
    num_epochs: Optional[int]


@dataclass
class TBFolderDatasetLog:
    # [CLASS DOC]
    # Purpose: dataclass logging a folder dataset: dataset_type, folder_path, filename_pattern, recursive, seq_len, dtype,
    # skip_in_stream, num_samples, num_tokens, human_format_num_tokens, shuffle, seed, num_epochs, files_order (list of
    # file paths in read order).
    dataset_type: str
    folder_path: str
    filename_pattern: str
    recursive: bool
    seq_len: int
    dtype: str
    skip_in_stream: bool
    num_samples: int
    num_tokens: int
    human_format_num_tokens: str
    shuffle: Optional[bool]
    seed: Optional[int]
    num_epochs: Optional[int]
    files_order: Optional[list]


@dataclass
class TrainDataLog:
    # [CLASS DOC]
    # Purpose: dataclass summarising training data: global_batch_size, sequence_length, total_training_tokens,
    # human_total_train_tokens, train/eval/test sample counts and subset logs (eval/test are None in practice).
    global_batch_size: int
    sequence_length: int
    total_training_tokens: int
    human_total_train_tokens: str
    train_num_samples: int
    eval_num_samples: int
    test_num_samples: int
    train_subset: Union[TBFileDatasetLog, TBFolderDatasetLog]
    eval_subset: Union[TBFileDatasetLog, TBFolderDatasetLog]
    test_subset: Union[TBFileDatasetLog, TBFolderDatasetLog]


class TokenizedBytesFileDataset(Dataset):
    # [CLASS DOC]
    # Purpose: Dataset over one binary file of fixed-width tokens, returning consecutive non-overlapping windows of
    # seq_len + 1 tokens; indices wrap around (looping), and sequential access reuses an open stream.
    # Key attributes: file_path, seq_len, dtype, dtype_size, skip_in_stream, skip_tokens, _len (number of windows),
    # _stream (generator of windows or None), _last_item_requested, subset_log.
    # Inherits from torch Dataset.
    def __init__(
        self,
        file_path: str,
        seq_len: int,
        dtype: np.dtype = np.uint16,
        skip_in_stream: bool = True,
        num_samples: Optional[int] = None,
        max_tokens: Optional[int] = None,
        skip_tokens: Optional[int] = None,
    ):
        # [FUNCTION DOC]
        # Summary: measures the file and computes how many windows it holds.
        # Behavior: token count is the file size (S3 content length or local size) divided by dtype size, minus skip_tokens;
        # optionally capped at max_tokens; divided by seq_len + 1 gives _len. Builds a TBFileDatasetLog (num_epochs is
        # num_samples divided by _len when given).
        # Inputs: file_path (S3 URI or local path), seq_len (int), dtype (default uint16), skip_in_stream (default True),
        # num_samples (optional, informative only), max_tokens (optional cap), skip_tokens (optional tokens to skip at start).
        # Outputs: None.
        # Exceptions: ImportError for S3 without boto3; OSError for missing files; ZeroDivisionError if num_samples is given
        # and the file is empty.
        # Side effects: S3 HEAD request or file stat.
        """Streaming dataset for a single TokenizedByte file
        We loop on the dataset if asking for an index larger than the dataset size

        Args:
            file_path (str): path to file on s3 or locally
            seq_len (int): sequence length
            dtype (np.dtype, optional): numpy dtype. Defaults to np.uint16.
            skip_in_stream (bool, optional): skip ahead in stream. Defaults to True.
            num_samples (Optional[int], optional): number of samples. Defaults to None. Only indicative for the number of epoch
        """
        self.file_path = file_path
        self.seq_len = seq_len
        self.dtype = dtype
        self.dtype_size = np.dtype(dtype).itemsize
        self.skip_in_stream = skip_in_stream
        self.skip_tokens = skip_tokens or 0
        # total number of full contexts in this file
        if file_path.startswith("s3://"):
            if not BOTO3_AVAILABLE:
                raise ImportError("boto3 is required: pip install boto3")
            num_tokens = _get_s3_object(file_path).content_length // self.dtype_size - self.skip_tokens
        else:
            num_tokens = os.path.getsize(file_path) // self.dtype_size - self.skip_tokens
        self._len = (min(max_tokens, num_tokens) if max_tokens else num_tokens) // (seq_len + 1)
        self._stream = None
        self._last_item_requested = None

        self.subset_log = TBFileDatasetLog(
            dataset_type=self.__class__.__name__,
            file_path=file_path,
            seq_len=seq_len,
            dtype=np.dtype(dtype).name,
            skip_in_stream=skip_in_stream,
            num_samples=self._len,
            num_tokens=self._len * (seq_len + 1),
            human_format_num_tokens=human_format(self._len * (seq_len + 1)),
            num_epochs=num_samples // self._len if num_samples else 0,
        )

    def _get_new_stream(self, index):
        # [FUNCTION DOC]
        # HELPER: generator opening the file at window index (plus skip_tokens) and yielding each window as an int64 tensor.
        # Inputs: index (int window index). Outputs: yields 1-D int64 torch tensors of seq_len + 1 tokens.
        # Exceptions: AssertionError if a chunk is shorter than a full window (e.g. the file tail).
        # Side effects: network or file I/O.
        # KNOWN ISSUE: skip_tokens is a token count but is added to a window index (offset is in windows), so the skip is
        # wrong by a factor of seq_len + 1.
        """Get a new stream starting from index (in sequence length contexts)

        Note: we pick chunks of seq_len + 1 to account for the label/target of the last tokens
              This means that we drop one token of training per sample.
        """
        chunk_size = self.dtype_size * (self.seq_len + 1)
        index += self.skip_tokens
        for chunk in _stream_file(self.file_path, chunk_size, index):
            assert len(chunk) == self.dtype_size * (self.seq_len + 1), (
                f"Expected {chunk_size} bytes from file but got " f"{len(chunk)}"
            )
            # careful with type conversions here
            yield torch.as_tensor(np.frombuffer(chunk, self.dtype).astype(np.int64), dtype=torch.int64)

    def __getitem__(self, item):
        # [FUNCTION DOC]
        # Summary: returns the window at item modulo length, using the current stream when possible.
        # Behavior: if a stream exists, the request moves forward and skip_in_stream is on, consume windows until just before
        # the target; otherwise, if there is no stream or the request is not the next window, open a new stream at the
        # target. Then read the next window.
        # Inputs: item (int). Outputs: dictionary with "input_ids" tensor.
        # Exceptions: ZeroDivisionError when the dataset is empty; errors from _get_next_from_stream.
        # Side effects: mutates _stream and _last_item_requested; I/O.
        # We loop on the dataset if asking for an index larger than the dataset size
        epoch_item = item % len(self)
        # if item >= len(self):
        #     raise IndexError(f"Index {item} requested for file {self.file_path} but it only has size {len(self)}")
        # skip ahead without creating a new stream
        if self._stream and epoch_item > self._last_item_requested and self.skip_in_stream:
            while self._last_item_requested < epoch_item - 1:
                self._last_item_requested += 1
                self._get_next_from_stream()  # consume stream
        # new stream starting from "epoch_item"
        elif not self._stream or epoch_item != self._last_item_requested + 1:
            self._stream = self._get_new_stream(epoch_item)

        self._last_item_requested = epoch_item

        return {"input_ids": self._get_next_from_stream()}

    def _get_next_from_stream(self):
        # [FUNCTION DOC]
        # HELPER: reads the next window, retrying on any exception by reopening the stream at the last requested item with
        # exponential back-off from 0.01 s, giving up once the delay reaches 2 s.
        # Inputs: None. Outputs: tensor.
        # Exceptions: re-raises the last error after giving up (including StopIteration at end of file).
        # Side effects: sleeps, reopens streams, error logging.
        sleep_time = 0.01
        while True:
            try:
                return next(self._stream)
            except Exception as e:
                if sleep_time >= 2.0:
                    logger.error("Giving up on re-establishing stream.")
                    raise e

                time.sleep(sleep_time)
                self._stream = self._get_new_stream(self._last_item_requested)
                sleep_time *= 2

    def __len__(self):
        # [FUNCTION DOC]
        # Summary: number of windows. Inputs: None. Outputs: int. Exceptions: None explicitly raised. Side effects: None.
        return self._len


class OldTokenizedBytesFolderDataset(Dataset):
    # [CLASS DOC]
    # Purpose: legacy dataset concatenating many TokenizedBytesFileDataset files of a folder; indices wrap around.
    # Key attributes: folder_path, filename_pattern (compiled regex), files (list of file datasets), lens (cumulative
    # window counts starting at 0), current_file (cached file position), subset_log.
    # Inherits from torch Dataset.
    def __init__(
        self,
        folder_path: str,
        seq_len: int,
        filename_pattern: Union[Pattern, str] = None,
        recursive: bool = True,
        dtype: np.dtype = np.uint16,
        skip_in_stream: bool = True,
        num_samples: Optional[int] = None,
        max_tokens: Optional[int] = None,
        skip_tokens: Optional[int] = None,
        shuffle: bool = False,
        seed: int = 42,
    ):
        # [FUNCTION DOC]
        # Summary: lists matching files and builds per-file datasets honouring max_tokens and skip_tokens.
        # Behavior: list files (S3 listing, or os.walk with regex match on the full path), sort, require at least one. For each
        # file build a TokenizedBytesFileDataset with the remaining token budget and remaining skip; files entirely consumed
        # by the skip are dropped; stop once max_tokens is reached. Optionally shuffle file order with a numpy generator from
        # seed. Compute cumulative lengths and the log.
        # Inputs: folder_path, seq_len, filename_pattern (regex string or pattern, default None), recursive (default True),
        # dtype (default uint16), skip_in_stream (default True), num_samples (optional), max_tokens (optional), skip_tokens
        # (optional), shuffle (default False), seed (default 42).
        # Outputs: None.
        # Exceptions: FileNotFoundError if nothing matches.
        # Side effects: S3 listing or directory walk, file stats, logging.
        # KNOWN ISSUE: for local folders a None filename_pattern raises AttributeError.
        # KNOWN ISSUE: each file is constructed with the whole remaining skip even if it exceeds the file size, giving a
        # negative length for skipped files, and the skip for the first kept file is again applied in window units.
        """Dataset for a folder of TokenizedBytes files
        We loop on the dataset if asking for an index larger than the dataset size

        Args:
            folder_path (str): path to folder on S3 or locally
            seq_len (int): sequence length
            filename_pattern (Union[Pattern, str], optional): filename pattern. Defaults to None.
            recursive (bool, optional): search recursively. Defaults to True.
            dtype (np.dtype, optional): numpy dtype. Defaults to np.uint16.
            skip_in_stream (bool, optional): skip ahead in stream. Defaults to True.
            num_samples (Optional[int], optional): number of samples. Defaults to None. Only indicative for the number of epoch
            shuffle (bool, optional): shuffle the files in the folder. Defaults to False.
            seed (int, optional): seed for shuffling. Defaults to 42.
        """
        self.folder_path = folder_path
        if isinstance(filename_pattern, str):
            filename_pattern = re.compile(filename_pattern)
        self.filename_pattern = filename_pattern
        if folder_path.startswith("s3://"):
            matched_file_paths = _get_s3_file_list(folder_path, filename_pattern, recursive)
        else:
            matched_file_paths = [
                os.path.join(root, file)
                for root, _, files in os.walk(folder_path)
                for file in files
                if filename_pattern.match(os.path.join(root, file))
            ]
        matched_file_paths = sorted(matched_file_paths)
        if not matched_file_paths:
            raise FileNotFoundError(f'No files matching "{filename_pattern}" found in {folder_path}')

        self.files = []
        remaining_tokens = max_tokens
        remaining_skip_tokens = skip_tokens or 0
        for path in matched_file_paths:
            file_data = TokenizedBytesFileDataset(
                path,
                seq_len,
                dtype=dtype,
                skip_in_stream=skip_in_stream,
                max_tokens=remaining_tokens,
                skip_tokens=remaining_skip_tokens,
            )
            if remaining_skip_tokens:
                remaining_skip_tokens -= len(file_data) * (seq_len + 1)
                if remaining_skip_tokens <= 0:
                    remaining_skip_tokens = 0
                elif remaining_skip_tokens > 0:
                    continue  # We skip this file entirely
            self.files.append(file_data)
            if remaining_tokens:
                remaining_tokens -= len(file_data) * (seq_len + 1)
                if remaining_tokens <= 0:
                    break

        log_rank(f"Found {len(self.files)} files.", logger=logger, level=logging.INFO, rank=0)
        if shuffle:
            log_rank("Shuffling...", logger=logger, level=logging.INFO, rank=0)
            rand = np.random.default_rng(seed)
            ordering = rand.permutation(range(len(self.files)))
            self.files = [self.files[i] for i in ordering]

        self.lens = np.cumsum([0] + [len(f) for f in self.files]).tolist()

        self.current_file = 0

        self.subset_log = TBFolderDatasetLog(
            dataset_type=self.__class__.__name__,
            folder_path=folder_path,
            filename_pattern=str(filename_pattern),
            recursive=recursive,
            seq_len=seq_len,
            dtype=np.dtype(dtype).name,
            skip_in_stream=skip_in_stream,
            num_samples=self.lens[-1] if self.lens else 0,
            num_tokens=self.lens[-1] * (seq_len + 1),
            human_format_num_tokens=human_format(self.lens[-1] * (seq_len + 1)),
            shuffle=shuffle,
            seed=seed,
            num_epochs=num_samples // self.lens[-1] if num_samples and self.lens else 0,
            files_order=[str(f.file_path) for f in self.files],
        )

    def __getitem__(self, item):
        # [FUNCTION DOC]
        # Summary: maps item modulo length to a file by bisecting cumulative lengths (cached current_file when still inside
        # it) and returns that file's item at the local offset.
        # Inputs: item (int). Outputs: dictionary with "input_ids".
        # Exceptions: ZeroDivisionError if empty. Side effects: updates current_file; I/O.
        epoch_item = item % len(self)
        # if item >= len(self):
        #     raise IndexError(
        #         f"Index {item} requested for dataset {self.folder_path} (pattern: {self.filename_pattern}) "
        #         f"but it only has size {len(self)}"
        #     )
        # check if we are in the same file as before
        if not (self.lens[self.current_file] <= epoch_item < self.lens[self.current_file + 1]):
            # figure out current file
            self.current_file = bisect(self.lens, epoch_item) - 1
        # subtract file starting offset
        return self.files[self.current_file][epoch_item - self.lens[self.current_file]]

    def __len__(self):
        # [FUNCTION DOC]
        # Summary: total windows across files (0 if none). Inputs: None. Outputs: int.
        # Exceptions: None explicitly raised. Side effects: None.
        return self.lens[-1] if self.lens else 0


class TokenizedBytesFolderDataset(DatatroveFolderDataset):
    # [CLASS DOC]
    # Purpose: DatatroveFolderDataset subclass that caches S3 file listings and sizes on disk and adds a subset log.
    # Key attributes: those of DatatroveFolderDataset (files, lens, folder_path, current_file_path, ...), subset_log.
    def __init__(
        self,
        folder_path: str,
        seq_len: int,
        filename_pattern: str = None,
        recursive: bool = True,
        token_size: int = 2,
        max_tokens: int | None = None,
        shuffle: bool = False,
        seed: int = 42,
        return_positions: bool = False,
        eos_token_id: int | None = None,
        skip_in_stream: bool = True,
        num_samples: Optional[int] = None,
        folder_read_path: Optional[str] = None,
        force_update_cache: bool = os.environ.get("FORCE_UPDATE_CACHE_S3", 0) == "1",
    ):
        # [FUNCTION DOC]
        # Summary: resolves the S3 file list (with cache) and initialises the datatrove folder dataset.
        # Behavior: warn if return_positions without eos_token_id. For S3 folders: cache file is the md5 of folder, pattern
        # and recursive flag under ~/.cache/nanotron/s3_cache. Within main_rank_first (world group), global rank 0 loads the
        # pickle cache unless force_update_cache, else lists files via datatrove url_to_fs (find or glob, depth 1 when not
        # recursive), records each file size, and saves the cache. Every other rank then reads the cache. Pass the list and
        # sizes to DatatroveFolderDataset along with sequence, token size, shuffle, seed, positions and read path. Build the
        # TBFolderDatasetLog (dtype derived from token_size).
        # Inputs: folder_path, seq_len, filename_pattern (glob, default None), recursive (default True), token_size (default
        # 2 bytes), max_tokens (optional), shuffle (default False), seed (default 42), return_positions (default False),
        # eos_token_id (optional), skip_in_stream (only logged), num_samples (optional, informative), folder_read_path
        # (optional local mirror path), force_update_cache (default from env FORCE_UPDATE_CACHE_S3 equal to "1", evaluated
        # once at import time).
        # Outputs: None.
        # Exceptions: RuntimeError if a non-zero rank cannot read the cache.
        # Side effects: creates the cache directory, S3 listing, pickle file I/O, barrier on the world group, logging.
        log_rank("Using DatatroveFolderDataset", logger=logger, level=logging.INFO, rank=0)
        if return_positions and not eos_token_id:
            log_rank(
                "Using DatatroveFolderDataset with return_positions=True but no eos_token_id provided. It can be slow...",
                logger=logger,
                level=logging.WARNING,
                rank=0,
            )

        # Handle S3 paths specially
        matched_files = None
        file_sizes = None
        if folder_path.startswith("s3://"):
            cache_dir = os.path.expanduser("~/.cache/nanotron/s3_cache")
            os.makedirs(cache_dir, exist_ok=True)

            # Create a unique cache key based on the folder path and pattern
            import hashlib

            cache_key = hashlib.md5(f"{folder_path}:{filename_pattern}:{recursive}".encode()).hexdigest()
            cache_file = os.path.join(cache_dir, f"{cache_key}.cache")

            with main_rank_first():
                if dist.get_rank() == 0:
                    # Check if we have a valid cache
                    if os.path.exists(cache_file) and not force_update_cache:
                        try:
                            import pickle

                            with open(cache_file, "rb") as f:
                                cached_data = pickle.load(f)
                                matched_files = cached_data["matched_files"]
                                file_sizes = cached_data["file_sizes"]
                            log_rank(
                                "[TokenizedBytesFolderDataset] Using cached S3 file list",
                                logger=logger,
                                level=logging.INFO,
                                rank=0,
                            )
                        except Exception as e:
                            log_rank(
                                f"[TokenizedBytesFolderDataset] Failed to load cache, fetching from S3: {e}",
                                logger=logger,
                                level=logging.WARNING,
                                rank=0,
                            )

                    # If no cache or cache invalid, fetch from S3
                    if matched_files is None:
                        log_rank(
                            "[TokenizedBytesFolderDataset] Fetching file list from S3...",
                            logger=logger,
                            level=logging.INFO,
                            rank=0,
                        )
                        from datatrove.utils.dataset import url_to_fs

                        fs_folder, stripped_folder_path = url_to_fs(folder_path)
                        matched_files = (
                            fs_folder.find(stripped_folder_path, detail=False, maxdepth=1 if not recursive else None)
                            if not filename_pattern
                            else fs_folder.glob(
                                os.path.join(stripped_folder_path, filename_pattern),
                                maxdepth=1 if not recursive else None,
                            )
                        )
                        matched_files = sorted(matched_files)

                        # Get file sizes
                        file_sizes = {}
                        for path in matched_files:
                            file_path = fs_folder.unstrip_protocol(path)
                            fs, file_path = url_to_fs(file_path)
                            file_sizes[file_path] = fs.size(file_path)

                        # Save to cache
                        try:
                            import pickle

                            with open(cache_file, "wb") as f:
                                pickle.dump({"matched_files": matched_files, "file_sizes": file_sizes}, f)
                            log_rank(
                                "[TokenizedBytesFolderDataset] Saved S3 file list to cache",
                                logger=logger,
                                level=logging.INFO,
                                rank=0,
                            )
                        except Exception as e:
                            log_rank(
                                f"[TokenizedBytesFolderDataset] Failed to save cache: {e}",
                                logger=logger,
                                level=logging.WARNING,
                                rank=0,
                            )
            if dist.get_rank() != 0:
                try:
                    import pickle

                    with open(cache_file, "rb") as f:
                        cached_data = pickle.load(f)
                        matched_files = cached_data["matched_files"]
                        file_sizes = cached_data["file_sizes"]
                except Exception as e:
                    raise RuntimeError(f"Failed to read cache file on rank {dist.get_rank()}: {e}")

        super().__init__(
            folder_path=folder_path,
            seq_len=seq_len,
            filename_pattern=filename_pattern,
            recursive=recursive,
            token_size=token_size,
            max_tokens=max_tokens,
            shuffle=shuffle,
            seed=seed,
            return_positions=return_positions,
            eos_token_id=eos_token_id,
            read_path=folder_read_path,
            matched_files=matched_files,
            file_sizes=file_sizes,
        )

        self.subset_log = TBFolderDatasetLog(
            dataset_type=self.__class__.__name__,
            folder_path=folder_path,
            filename_pattern=str(filename_pattern),
            recursive=recursive,
            seq_len=seq_len,
            dtype=np.dtype(np.uint16 if token_size == 2 else np.uint32 if token_size == 4 else np.uint64).name,
            skip_in_stream=skip_in_stream,
            num_samples=self.lens[-1] if self.lens else 0,
            num_tokens=self.lens[-1] * (seq_len + 1),
            human_format_num_tokens=human_format(self.lens[-1] * (seq_len + 1)),
            shuffle=shuffle,
            seed=seed,
            num_epochs=num_samples // self.lens[-1] if num_samples and self.lens else 0,
            files_order=[str(f.file_path) for f in self.files],
        )


def build_dataset(
    dataset_folder: str,
    seq_length: int,
    token_size: int,
    return_positions: bool = False,
    eos_token_id: Optional[int] = None,
    use_old_brrr_dataloader: bool = False,
    skip_in_stream: bool = True,
    num_samples: Optional[int] = None,
    max_tokens: Optional[int] = None,
    skip_tokens: Optional[int] = None,
    shuffle: Optional[bool] = False,
    seed: Optional[int] = 6,
    folder_read_path: Optional[str] = None,
) -> "DatatroveFolderDataset":
    # [FUNCTION DOC]
    # Summary: builds one tokenized-bytes dataset for a folder.
    # Behavior: with use_old_brrr_dataloader, log and return OldTokenizedBytesFolderDataset matching files ending in ".ds"
    # (folder_read_path ignored with a warning; dtype uint16 for 2-byte tokens else uint32). Otherwise return
    # TokenizedBytesFolderDataset with glob "*.ds", non-recursive, forwarding all options.
    # Inputs: dataset_folder, seq_length, token_size, return_positions (default False), eos_token_id, use_old_brrr_dataloader
    # (default False), skip_in_stream (default True), num_samples, max_tokens, skip_tokens (old path only), shuffle
    # (default False), seed (default 6), folder_read_path.
    # Outputs: dataset.
    # Exceptions: propagated from constructors.
    # Side effects: as the constructors.
    # KNOWN ISSUE: the docstring documents a dataset_args parameter that does not exist, and the log message blames
    # skip_in_stream rather than use_old_brrr_dataloader.
    """Build one TokenizedBytes dataset from a file or a folder on S3 or locally

    Args:
        dataset_args (Union[TokenizedBytesDatasetFileArgs, TokenizedBytesDatasetFolderArgs]): dataset config
        seq_length ([type]): sequence length
        skip_in_stream (bool, optional): skip ahead in stream. Defaults to True.
    """
    if use_old_brrr_dataloader:
        log_rank(
            "Using old tokenized bytes folder dataset because skip_in_stream is False",
            logger=logger,
            level=logging.INFO,
            rank=0,
        )
        if folder_read_path:
            log_rank(
                f"Ignoring folder_read_path={folder_read_path} because use_old_brrr_dataloader is True",
                logger=logger,
                level=logging.WARNING,
                rank=0,
            )
        return OldTokenizedBytesFolderDataset(
            dataset_folder,
            seq_length,
            filename_pattern=".*\\.ds$",
            skip_in_stream=skip_in_stream,
            max_tokens=max_tokens,
            num_samples=num_samples,
            skip_tokens=skip_tokens,  # # Optional number of tokens to skip at the beginning (We'll only train on the rest)
            shuffle=shuffle,
            seed=seed,
            dtype=np.uint16 if token_size == 2 else np.uint32,
        )

    return TokenizedBytesFolderDataset(
        folder_path=dataset_folder,
        filename_pattern="*.ds",
        seq_len=seq_length,
        recursive=False,
        token_size=token_size,
        max_tokens=max_tokens, # TODO: remove
        shuffle=shuffle,
        return_positions=return_positions,  # if set to True, the position ids are directly read from datatrove
        eos_token_id=eos_token_id,
        seed=seed,
        skip_in_stream=skip_in_stream,
        num_samples=num_samples,
        folder_read_path=folder_read_path,
    )


def get_tb_datasets(
    config: NanosetDatasetsArgs,
    sequence_length: int,
    global_batch_size: int,
    train_steps: int,
    current_iteration: int,
    parallel_context: ParallelContext,
    eos_token_id: Optional[int] = None,
    shuffle: bool = False,
    seed: int = 6,
    consumed_samples: int = 0,
    consumed_tokens_per_dataset_folder: Optional[Dict[str, int]] = None,
    last_stages_consumed_tokens_per_dataset_folder: Optional[Dict[str, int]] = None,
) -> Tuple[DataLoader, TrainDataLog]:
    # [FUNCTION DOC]
    # Summary: builds the blended training dataset for the current data stage and its log.
    # Behavior: train_num_samples is train_steps times global_batch_size. Convert last-stage consumed tokens per folder
    # to sample offsets by dividing by sequence_length. Build one dataset per configured folder (with optional per-folder
    # max tokens and read path). If dataset_read_path is configured, normalise weights and, per dataset, read the sample
    # at the estimated current position, log which file is read and how many tokens are needed, assert the dataset holds
    # enough samples for all stages, and assert that every file overlapping the needed sample range exists locally. Then
    # global rank 0 compiles the C++ helpers, all ranks barrier, and the datasets are blended with BlendableDataset using
    # config weights (default equal), the seed, consumed tokens and offsets.
    # Inputs: config (NanosetDatasetsArgs-like with dataset_folder, dataset_weights, dataset_max_tokens,
    # token_size_in_bytes, return_positions, use_old_brrr_dataloader, skip_in_stream, dataset_read_path),
    # sequence_length, global_batch_size, train_steps, current_iteration, parallel_context, eos_token_id, shuffle (default
    # False), seed (default 6), consumed_samples (default 0), consumed_tokens_per_dataset_folder (optional),
    # last_stages_consumed_tokens_per_dataset_folder (optional).
    # Outputs: tuple (BlendableDataset, TrainDataLog).
    # Exceptions: AssertionError for missing data; ImportError path; errors from dataset builds.
    # Side effects: S3/file I/O, C++ compilation on rank 0, barrier on the world group, logging.
    # KNOWN ISSUE: last_stages_consumed_tokens_per_dataset_folder defaults to None but is iterated unconditionally.
    # KNOWN ISSUE: the single-dataset branch is disabled by an always-false condition (dead code).
    # KNOWN ISSUE: compile_helper exits the process on failure instead of raising ImportError, so the except is dead.
    # KNOWN ISSUE: the return annotation says a DataLoader but a dataset is returned.
    """Build TokenizedBytes datasets

    Args:
        config (NanosetDatasetsArgs): dataset config
        sequence_length (int): sequence length
        global_batch_size (int): global batch size
        train_steps (int): number of training steps
        parallel_context (ParallelContext): distributed process groups
    """
    log_rank("Building Streamable datasets.", logger=logger, level=logging.INFO, rank=0)
    dataset_max_tokens = config.dataset_max_tokens
    if dataset_max_tokens is None:
        dataset_max_tokens = [None] * len(config.dataset_folder)
    train_num_samples = train_steps * global_batch_size
    last_stages_consumed_samples_per_dataset_folder = {k: v // sequence_length for k, v in last_stages_consumed_tokens_per_dataset_folder.items()}

    datasets = [
        build_dataset(
            dataset_folder,
            sequence_length,
            token_size=config.token_size_in_bytes,
            return_positions=config.return_positions,
            eos_token_id=eos_token_id,
            use_old_brrr_dataloader=config.use_old_brrr_dataloader,
            skip_in_stream=config.skip_in_stream,
            max_tokens=max_tokens,
            num_samples=train_num_samples,
            shuffle=shuffle,
            seed=seed,
            folder_read_path=config.dataset_read_path[i] if config.dataset_read_path else None,
        )
        for i, (dataset_folder, max_tokens) in enumerate(zip(config.dataset_folder, dataset_max_tokens))
    ]

    # in case of dataset_read_path check we have enough files locally for the training
    if config.dataset_read_path:

        weights = config.dataset_weights    
        if not weights:
            weights = [1] * len(datasets)

        # Normalize weights
        weights = np.array(weights, dtype=np.float64)
        sum_weights = np.sum(weights)
        assert sum_weights > 0.0
        weights /= sum_weights

        # check we have enough files locally for the training
        for i, dataset in enumerate(datasets):
            # warmup datasets
            estimate_current_sample = int(consumed_samples * weights[i]) + last_stages_consumed_samples_per_dataset_folder.get(dataset.folder_path, 0)
            _ = dataset[estimate_current_sample]
            # print which file we're currently reading from
            log_rank(f"Dataset {i} ({dataset.folder_path}) is reading from file {dataset.current_file_path}", logger=logger, level=logging.INFO, rank=0)
            # estimate number of tokens needed for this dataset
            needed_num_samples_dataset = int((train_steps - current_iteration) * global_batch_size * weights[i])
            needed_num_tokens_dataset = needed_num_samples_dataset * sequence_length
            needed_size_tokens_dataset = human_format(needed_num_tokens_dataset * config.token_size_in_bytes)
            log_rank(f"Dataset {i} ({dataset.folder_path}) needs {needed_num_tokens_dataset} tokens (size: {needed_size_tokens_dataset}) for current stage", logger=logger, level=logging.INFO, rank=0)

            # NOTE: let's assume that s3 folder keep the same old files when resuming
            # check that sum of lens of files in dataset is greater than needed_num_samples_dataset (use dataset.lens)
            total_num_samples_dataset = int(train_steps * global_batch_size * weights[i])
            log_rank(f"Dataset {i} ({dataset.folder_path}) on s3 has {len(dataset) * sequence_length} tokens (size: {human_format(len(dataset) * sequence_length * config.token_size_in_bytes)}) and needs {total_num_samples_dataset * sequence_length} tokens (size: {human_format(total_num_samples_dataset * sequence_length * config.token_size_in_bytes)}) for all stages", logger=logger, level=logging.INFO, rank=0)
            assert total_num_samples_dataset <= len(dataset), f"Not enough files on s3 for dataset {i} ({dataset.folder_path})"
            # check that local files exist for the needed_num_samples_dataset
            estimate_end_sample = estimate_current_sample + needed_num_samples_dataset
            for file_idx, file in enumerate(dataset.files):
                # intersection [start_sample, end_sample] with [dataset.lens[file_idx], dataset.lens[file_idx+1]]
                a, b, c, d = estimate_current_sample, estimate_end_sample, dataset.lens[file_idx], dataset.lens[file_idx+1]
                if max(a, c) < min(b, d): # ranges overlap
                    assert os.path.exists(file.file_path), f"Dataset {i} ({dataset.folder_path}) will need file {file.file_path} but it does not exist"
                    log_rank(f"Dataset {i} ({dataset.folder_path}) will need file {file.file_path} from sample {max(a, c)} to {min(b, d)} (offset: {last_stages_consumed_samples_per_dataset_folder.get(dataset.folder_path, 0)})", logger=logger, level=logging.INFO, rank=0)
                else:
                    log_rank(f"Dataset {i} ({dataset.folder_path}) will not need file {file.file_path} to train from sample {estimate_current_sample} to {estimate_end_sample} (offset: {last_stages_consumed_samples_per_dataset_folder.get(dataset.folder_path, 0)})", logger=logger, level=logging.INFO, rank=0)
                    

    if len(datasets) == 1 and False:
        outputs_dataset = datasets[0]
    else:
        if dist.get_rank(parallel_context.world_pg) == 0:
            try:
                compile_helper()
            except ImportError:
                raise ImportError(
                    "Could not compile megatron dataset C++ helper functions and therefore cannot import helpers python file."
                )
        dist.barrier(parallel_context.world_pg)
        weights = config.dataset_weights
        if not weights:
            weights = [1] * len(datasets)

        outputs_dataset = BlendableDataset(
            datasets,
            weights,
            train_num_samples,
            parallel_context=parallel_context,
            seed=seed,
            consumed_tokens_per_dataset_folder=consumed_tokens_per_dataset_folder,
            offsets_in_samples=last_stages_consumed_samples_per_dataset_folder,
        )

    log_rank("Streamable datasets ready.", logger=logger, level=logging.INFO, rank=0)
    train_data_log = TrainDataLog(
        train_num_samples=train_num_samples,
        eval_num_samples=None,
        test_num_samples=None,
        global_batch_size=global_batch_size,
        sequence_length=sequence_length,
        total_training_tokens=train_num_samples * sequence_length,
        human_total_train_tokens=human_format(train_num_samples * sequence_length),
        train_subset=outputs_dataset.subset_log,
        eval_subset=None,
        test_subset=None,
    )
    return outputs_dataset, train_data_log


def get_tb_dataloader(
    dataset: Union[DatatroveFolderDataset, BlendableDataset, Dataset],
    sequence_length: int,
    micro_batch_size: int,
    global_batch_size: int,
    cfg: NanosetDatasetsArgs,
    num_workers: int,
    consumed_samples: int,
    num_samples: int,
    parallel_context: ParallelContext,
    input_pp_rank: int,
    output_pp_rank: int,
    dataloader_drop_last: bool = True,
    dataloader_pin_memory: bool = True,
    use_position_ids: bool = False,
    use_doc_masking: bool = False,
) -> DataLoader:
    # [FUNCTION DOC]
    # Summary: wraps a dataset in a DataLoader driven by MegatronPretrainingSampler.
    # Behavior: pipeline ranks owning neither inputs nor labels get an EmptyInfiniteDataset of the same length. Build the
    # sequential Megatron sampler for this DP rank from consumed_samples to num_samples with global_batch_size and the
    # config's padding option. Choose DataCollatorForCLMWithPositionIds (with use_doc_masking) or DataCollatorForCLM.
    # Return a DataLoader with the batch sampler, collator, worker count, pin memory and per-DP-rank worker seeding.
    # Inputs: dataset, sequence_length, micro_batch_size, global_batch_size, cfg (provides
    # pad_samples_to_global_batch_size), num_workers, consumed_samples, num_samples, parallel_context, input_pp_rank,
    # output_pp_rank, dataloader_drop_last (default True), dataloader_pin_memory (default True), use_position_ids
    # (default False), use_doc_masking (default False).
    # Outputs: torch DataLoader.
    # Exceptions: RuntimeError from sampler validation.
    # Side effects: logging.
    # Only some rank require to run the dataloader.
    if dist.get_rank(parallel_context.pp_pg) not in [
        input_pp_rank,
        output_pp_rank,
    ]:
        dataset = EmptyInfiniteDataset(length=len(dataset))

    log_rank(
        f"Building dataloader with consumed samples for current datastage: {consumed_samples}", logger=logger, level=logging.INFO, rank=0
    )
    # Megatron sampler
    # batch_sampler = MegatronPretrainingRandomSampler(
    batch_sampler = MegatronPretrainingSampler(
        total_samples=num_samples,
        consumed_samples=consumed_samples,
        micro_batch_size=micro_batch_size,
        data_parallel_rank=dist.get_rank(parallel_context.dp_pg),
        data_parallel_size=parallel_context.dp_pg.size(),
        drop_last=dataloader_drop_last,
        global_batch_size=global_batch_size,
        pad_samples_to_global_batch_size=cfg.pad_samples_to_global_batch_size,
    )

    # We use the data collator to put the tensors on the right pipeline parallelism rank
    if use_position_ids:
        data_collator = DataCollatorForCLMWithPositionIds(
            sequence_length=sequence_length,
            input_pp_rank=input_pp_rank,
            output_pp_rank=output_pp_rank,
            parallel_context=parallel_context,
            use_doc_masking=use_doc_masking,
        )
    else:
        data_collator = DataCollatorForCLM(
            sequence_length=sequence_length,
            input_pp_rank=input_pp_rank,
            output_pp_rank=output_pp_rank,
            parallel_context=parallel_context,
        )

    return DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        num_workers=num_workers,
        collate_fn=data_collator,
        pin_memory=dataloader_pin_memory,
        worker_init_fn=get_dataloader_worker_init(dp_rank=dist.get_rank(parallel_context.dp_pg)),
    )
