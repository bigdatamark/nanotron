# [MODULE DOC]
# Purpose: test helpers for Nanoset dataset/dataloader tests: create dummy JSON datasets, tokenize them with tools/preprocess_data.py (datatrove), and assert that batches and datasets are consistent across ranks.
# Main contents: dataset path/file creation, preprocessing wrapper, batch and Nanoset synchronization asserts, hashing helpers.
# Import-time side effects: imports the nanotron package to locate the repo root and appends it to sys.path so that tools.preprocess_data can be imported.
import hashlib
import importlib
import json
import os
import sys
from argparse import Namespace
from collections import OrderedDict
from pathlib import Path

package = importlib.import_module("nanotron")
package_path = Path(package.__file__).parent.parent.parent
sys.path.append(str(package_path))

import nanotron.distributed as dist
import torch
from nanotron.data.nanoset import Nanoset
from nanotron.parallel import ParallelContext
from nanotron.parallel.pipeline_parallel.tensor_pointer import TensorPointer
from nanotron.sanity_checks import assert_tensor_synced_across_pg

from tools.preprocess_data import main


def create_dataset_paths(tmp_dir: str, quantity: int):
    # [FUNCTION DOC]
    # HELPER: builds file paths for dataset tests.
    # Behavior: returns two parallel lists of length quantity: JSON file paths named pytest_i.json and tokenized output folder paths named tokenized_documents_i, all inside tmp_dir.
    # Inputs: tmp_dir, directory string or path; quantity, number of datasets.
    # Outputs: tuple of (list of JSON paths, list of tokenized folder paths).
    # Exceptions: None explicitly raised.
    # Side effects: None (nothing is created on disk).
    json_dataset_path = [os.path.join(tmp_dir, f"pytest_{i}.json") for i in range(quantity)]
    datatrove_tokenized_dataset_paths = [os.path.join(tmp_dir, f"tokenized_documents_{i}") for i in range(quantity)]

    return json_dataset_path, datatrove_tokenized_dataset_paths


def create_dummy_json_dataset(path_to_json: str, dummy_text: str, n_samples: int = 50000):
    # [FUNCTION DOC]
    # HELPER: writes a JSON-lines dummy dataset.
    # Behavior: appends n_samples lines to the file, each a JSON object with a "text" field containing the sample index and dummy_text.
    # Inputs: path_to_json, file path; dummy_text, string inserted into every sample; n_samples, default 50000.
    # Outputs: None.
    # Exceptions: OSError on file errors.
    # Side effects: appends to the file (existing content is kept).
    with open(path_to_json, "a") as json_file:
        for sample in range(n_samples):
            sample_dict = {"text": f"[{sample}] Hello! Im sample {sample}! And this is my dummy text: {dummy_text}"}
            json_file.write(json.dumps(sample_dict))
            json_file.write("\n")


def preprocess_dummy_dataset(json_dataset_path: str, datatrove_tokenized_dataset_path: str, tokenizer: str):
    # [FUNCTION DOC]
    # HELPER: tokenizes a JSONL dataset via tools/preprocess_data.py.
    # Behavior: builds an argparse Namespace with reader jsonl, the given input path, column "text", no glob, the given output folder and tokenizer, no extra eos token, one task and no logging dir, then calls the preprocess main.
    # Inputs: json_dataset_path, input file; datatrove_tokenized_dataset_path, output folder; tokenizer, tokenizer name or path.
    # Outputs: None.
    # Exceptions: whatever the preprocessing pipeline raises.
    # Side effects: downloads tokenizer if needed, writes tokenized files to disk.
    # Create args for preprocessing
    args = Namespace(
        readers="jsonl",
        dataset=json_dataset_path,
        column="text",
        glob_pattern=None,
        output_folder=datatrove_tokenized_dataset_path,
        tokenizer_name_or_path=tokenizer,
        eos_token=None,
        n_tasks=1,
        logging_dir=None,
    )
    # tools/preprocess_data.py main
    main(args)


def assert_batch_dataloader(
    batch: dict, parallel_context: ParallelContext, micro_batch_size: int, sequence_length: int
):
    # [FUNCTION DOC]
    # HELPER: validates one dataloader batch in a distributed run.
    # Behavior: for each key in the batch: on pipeline rank 0 keys starting with "input_" must be tensors, on the last pipeline rank keys starting with "label_" must be tensors, everything else must be a TensorPointer. Real tensors must have shape micro_batch_size by sequence_length. A TensorPointer is replaced by a tensor of its group_rank (class 1); boolean masks are cast to int64 (class 2); otherwise class 0. Then asserts the (flattened) tensor is identical across the tensor parallel group and the class value is identical across the data parallel group.
    # Inputs: batch, dictionary with keys input_ids, input_mask, label_ids, label_mask; parallel_context; micro_batch_size; sequence_length.
    # Outputs: None.
    # Exceptions: AssertionError on any failed check.
    # Side effects: collectives on tp and dp groups; moves tensors to CUDA.
    """
    batch (dict): Batch produced from the Dataloader, with keys input_ids, input_mask, label_ids, label_mask

    """
    for element in batch:
        tensor = batch[element]

        # Assert that inputs are only present in input_pp_rank and outputs in output_pp_rank
        input_pp_rank, output_pp_rank = 0, int(parallel_context.pp_pg.size() - 1)
        if dist.get_rank(parallel_context.pp_pg) == input_pp_rank and element.startswith("input_"):
            assert isinstance(tensor, torch.Tensor)
        elif dist.get_rank(parallel_context.pp_pg) == output_pp_rank and element.startswith("label_"):
            assert isinstance(tensor, torch.Tensor)
        else:
            assert isinstance(tensor, TensorPointer)

        data_class = (
            0  # 0 if tensor is from the ids, 1 if TensorPointer and 2 if mask. Used in the data parallel group check
        )

        # Check shape of mask and ids tensors
        if isinstance(tensor, torch.Tensor):
            assert tensor.shape == (micro_batch_size, sequence_length)

        # TensorPointer case: Check that all TensorPointers from the same tp_pg point to the same group_rank. Create torch.tensor with group_rank
        if isinstance(tensor, TensorPointer):
            tensor = torch.tensor(tensor.group_rank)
            data_class = 1

        # Attention Masks case: dtype is torch.bool --> Transform to int64
        if tensor.dtype == torch.bool:
            tensor = tensor.long()
            data_class = 2

        # Assert that we have the SAME element in all the processes belonging to the same tensor parallel group
        assert_tensor_synced_across_pg(
            tensor=tensor.flatten().cuda(),
            pg=parallel_context.tp_pg,
            msg=lambda err: f"{element} is not synchronized across TP {err}",
        )

        # Assert that we have the SAME class of data in all processes belonging to the same data parallel group
        assert_tensor_synced_across_pg(
            tensor=torch.tensor(data_class, device="cuda"),
            pg=parallel_context.dp_pg,
            msg=lambda err: f"{element} is not synchronized across DP {err}",
        )


def compute_hash(identifier: OrderedDict, n_digit: int = 8) -> int:
    # [FUNCTION DOC]
    # HELPER: deterministic hashing of nested identifiers.
    # Behavior: serializes the ordered dictionary to indented JSON, hashes with sha256, interprets the hex digest as an integer and keeps the last n_digit decimal digits.
    # Inputs: identifier, ordered dictionary of JSON-serializable values; n_digit, default 8.
    # Outputs: integer hash.
    # Exceptions: TypeError if values are not JSON-serializable.
    # Side effects: None.
    """
    Creates a sha256 hash from the elements of a OrderedDict
    """
    unique_description = json.dumps(identifier, indent=4)
    # Create n_digit description hash
    unique_description_hash = int(hashlib.sha256(unique_description.encode("utf-8")).hexdigest(), 16) % 10**n_digit
    return unique_description_hash


def assert_nanoset_sync_across_all_ranks(nanoset: Nanoset, parallel_context: ParallelContext):
    # [FUNCTION DOC]
    # HELPER: checks the Nanoset is built identically on all ranks.
    # Behavior: collects the dataset folders, weights, sequence length, number of train samples, seed, length, input_ids of sample 23, dataset index arrays and token size into an ordered dictionary, hashes it with compute_hash and asserts the hash tensor is equal across the world group.
    # Inputs: nanoset, a Nanoset; parallel_context.
    # Outputs: None.
    # Exceptions: AssertionError if ranks differ; IndexError if the dataset has fewer than 24 samples.
    # Side effects: collective on the world group; CUDA allocation.
    """
    Checks that the same Nanoset is created in all processes
    """
    # Extract a sample from the Nanoset
    IDX_SAMPLE = 23

    nanoset_identifiers = OrderedDict()
    nanoset_identifiers["dataset_folders"] = nanoset.dataset_folders
    nanoset_identifiers["dataset_weights"] = nanoset.dataset_weights.tolist()
    nanoset_identifiers["sequence_length"] = nanoset.sequence_length
    nanoset_identifiers["train_split_num_samples"] = nanoset.train_split_num_samples
    nanoset_identifiers["random_seed"] = nanoset.random_seed
    nanoset_identifiers["length"] = len(nanoset)
    nanoset_identifiers["input_ids"] = nanoset[IDX_SAMPLE]["input_ids"].tolist()
    nanoset_identifiers["dataset_index"] = nanoset.dataset_index.tolist()
    nanoset_identifiers["dataset_sample_index"] = nanoset.dataset_sample_index.tolist()
    nanoset_identifiers["token_size"] = nanoset.token_size

    unique_description_hash = compute_hash(nanoset_identifiers)
    assert_tensor_synced_across_pg(
        tensor=torch.tensor(unique_description_hash, device="cuda"),
        pg=parallel_context.world_pg,
        msg=lambda err: f"Nanoset is not synchronized across all processes {err}",
    )


def compute_batch_hash(batch: dict) -> int:
    # [FUNCTION DOC]
    # HELPER: fingerprint of a batch for resume tests.
    # Behavior: for each key, uses the group_rank for a TensorPointer, the int64 list for a boolean tensor, or the plain list for other tensors, then hashes the ordered collection with compute_hash.
    # Inputs: batch, dictionary from the dataloader.
    # Outputs: integer hash.
    # Exceptions: None explicitly raised.
    # Side effects: None (copies tensors to Python lists).
    # KNOWN ISSUE: docstring says it "checks" recovery state, but it only computes a hash; the comparison is done by the caller.
    """
    Checks that the Nanoset/BlendedNanoset is in the same state after recovering from a crash

    batch (dict): Batch produced from the Dataloader, with keys input_ids, input_mask, label_ids, label_mask

    """
    batch_identifiers = OrderedDict()

    for element in batch:
        tensor = batch[element]

        # TensorPointer
        if isinstance(tensor, TensorPointer):
            identifier = tensor.group_rank

        # Attention Masks case: dtype is torch.bool --> Transform to int64
        elif tensor.dtype == torch.bool:
            identifier = tensor.long().tolist()

        # Input IDs tensor
        else:
            identifier = tensor.tolist()

        batch_identifiers[element] = identifier

    unique_description_hash = compute_hash(batch_identifiers)

    return unique_description_hash
