# [MODULE DOC]
# Purpose: builds and wraps training dataloaders for Hugging Face datasets and dummy data in nanotron.
# Main contents: sanity_check_dataloader (moves batches to GPU and checks DP/TP consistency),
# dummy_infinite_data_generator (random synthetic batches), set_tensor_pointers, get_dataloader_worker_init,
# get_train_dataloader (DataLoader over a datasets.Dataset with CLM collators and the DP-aware sampler).
# Fits in nanotron: used by the trainer and run_train scripts to feed the pipeline engine with micro-batches where
# non-owning pipeline ranks hold TensorPointers.
# Import-time side effects: creates a module logger.
from typing import Dict, Iterator, List, Optional, Union

import datasets
import numpy as np
import torch
from torch.utils.data import DataLoader

from nanotron import distributed as dist
from nanotron import logging
from nanotron.config import Config
from nanotron.data.clm_collator import DataCollatorForCLM, DataCollatorForCLMWithPositionIds
from nanotron.data.samplers import EmptyInfiniteDataset, get_sampler
from nanotron.logging.timers import nanotron_timer
from nanotron.parallel import ParallelContext
from nanotron.parallel.pipeline_parallel.tensor_pointer import TensorPointer
from nanotron.random import set_random_seed
from nanotron.sanity_checks import (
    assert_fail_except_rank_with,
    assert_tensor_synced_across_pg,
)

logger = logging.get_logger(__name__)


def sanity_check_dataloader(
    dataloader: Iterator[Dict[str, Union[torch.Tensor, TensorPointer]]],
    parallel_context: ParallelContext,
    config: Config,
) -> Iterator[Dict[str, Union[torch.Tensor, TensorPointer]]]:
    # [FUNCTION DOC]
    # Summary: generator wrapping a dataloader; converts each batch to CUDA tensors and optionally checks synchronisation.
    # Behavior: iterate the dataloader, timing each fetch with the "dataloader_fetch" nanotron timer. Convert numpy arrays
    # to torch tensors, then move every non-TensorPointer value to CUDA (contiguous, non-blocking). Unless
    # config.general.ignore_sanity_checks is set: for each tensor key (sorted, skipping keys containing "mask" or
    # "position_ids") assert the tensor is NOT identical across the DP group (expected to fail on all ranks but rank 0,
    # via assert_fail_except_rank_with); then assert every tensor is identical across the TP group. Yield the batch.
    # Inputs: dataloader (iterable of dictionaries of arrays/tensors/TensorPointers), parallel_context, config (Config).
    # Outputs: yields dictionaries of CUDA tensors or TensorPointers.
    # Exceptions: AssertionError from failed sanity checks; see known issue for end of iteration.
    # Side effects: host-to-device copies; collective communication over DP and TP groups during checks; timer updates.
    # KNOWN ISSUE: re-raising StopIteration inside a generator is converted to RuntimeError (PEP 479) instead of ending
    # iteration cleanly.
    """
    Run sanity checks on batches from the dataloader to ensure data is properly distributed and synchronized.

    Args:
        dataloader: Iterator that yields batches
        parallel_context: Object containing process groups information
        config: Configuration object

    Yields:
        The same batches after performing sanity checks
    """
    # WARNING: This is called in the middle of the training loop, so make sure it's optimized
    dataloader_iter = iter(dataloader)
    while True:
        try:
            # Measure just the time to fetch the next batch
            nanotron_timer("dataloader_fetch").start()
            batch = next(dataloader_iter)
            nanotron_timer("dataloader_fetch").end()

            # Process the batch (this part is not timed as part of the dataloader fetch)
            # maybe numpy to torch
            batch = {k: torch.from_numpy(v) if isinstance(v, np.ndarray) else v for k, v in batch.items()}

            # non_blocking=True seems to be fine? https://discuss.pytorch.org/t/should-we-set-non-blocking-to-true/38234/4
            micro_batch = {
                k: v
                if isinstance(v, TensorPointer)
                else v.to("cuda", memory_format=torch.contiguous_format, non_blocking=True)
                for k, v in batch.items()
            }

            if not config.general.ignore_sanity_checks:
                # SANITY CHECK: Check input are not the same across DP
                for key, value in sorted(micro_batch.items(), key=lambda x: x[0]):
                    if isinstance(value, TensorPointer):
                        continue

                    if "mask" in key or "position_ids" in key:
                        # It's fine if mask is the same across DP
                        continue

                    with assert_fail_except_rank_with(AssertionError, rank_exception=0, pg=parallel_context.dp_pg):
                        assert_tensor_synced_across_pg(
                            tensor=value, pg=parallel_context.dp_pg, msg=lambda err: f"{key} {err}"
                        )

                # SANITY CHECK: Check input are synchronized throughout TP
                for key, value in sorted(micro_batch.items(), key=lambda x: x[0]):
                    if isinstance(value, TensorPointer):
                        continue
                    assert_tensor_synced_across_pg(
                        tensor=value,
                        pg=parallel_context.tp_pg,
                        msg=lambda err: f"{key} are not synchronized throughout TP {err}",
                    )

                # SANITY CHECK: Check that input are synchronized throughout PP
                # TODO @thomasw21: That's really hard to test as input gets sharded across the PP, let's assume it works for now.

                # SANITY CHECK: Check that an input only exists on the PP rank responsible for it
                # TODO @nouamanetazi: add this test

            yield micro_batch

        except StopIteration:
            # End of dataloader
            raise StopIteration


def dummy_infinite_data_generator(
    micro_batch_size: int,
    sequence_length: int,
    input_pp_rank: int,
    output_pp_rank: int,
    vocab_size: int,
    seed: int,
    parallel_context: ParallelContext,
    use_position_ids: bool = False,
    cp_pg: Optional[dist.ProcessGroup] = None,
):
    # [FUNCTION DOC]
    # Summary: returns a zero-argument generator function producing endless random CLM batches on CUDA.
    # Behavior: compute the CP rank/size from cp_pg, assert sequence_length divides by cp_size and compute the local CP
    # slice. The inner generator seeds a CUDA torch.Generator with seed times (1 + DP rank) times (1 + PP rank) so TP
    # ranks agree. Without position ids it yields input_ids and input_mask (owned by input_pp_rank) and label_ids and
    # label_mask (owned by output_pp_rank), each a random or all-True tensor of micro_batch_size by sequence_length cut to
    # the CP slice, or a TensorPointer on non-owning ranks. With use_position_ids it precomputes position_ids where the
    # first row contains documents of lengths 4, 6 and the remainder, other rows a single document, and yields
    # position_ids instead of input_mask.
    # Inputs: micro_batch_size, sequence_length, input_pp_rank, output_pp_rank, vocab_size (token id upper bound),
    # seed, parallel_context, use_position_ids (default False), cp_pg (context-parallel process group, default None).
    # Outputs: the generator function data_generator.
    # Exceptions: AssertionError if sequence_length is not divisible by cp_size; AttributeError if cp_pg is None.
    # Side effects: None until the returned generator is iterated (then CUDA allocations and RNG use).
    # KNOWN ISSUE: cp_pg defaults to None but is used unconditionally, so the default always fails.
    """
    Generate dummy data for testing or benchmark purposes.

    Args:
        micro_batch_size: Size of each micro batch
        sequence_length: Maximum sequence length
        input_pp_rank: Rank responsible for input data
        output_pp_rank: Rank responsible for output/label data
        vocab_size: Size of the vocabulary
        seed: Random seed for reproducibility
        parallel_context: Object containing process groups information
        use_position_ids: Whether to use packed sequences

    Returns:
        Generator function that yields infinite random data batches
    """
    cp_rank = dist.get_rank(cp_pg)
    cp_size = cp_pg.size()
    assert sequence_length % cp_size == 0, f"sequence_length {sequence_length} must be divisible by cp_size {cp_size}"
    local_slice = slice(cp_rank * sequence_length // cp_size, (cp_rank + 1) * sequence_length // cp_size)

    def data_generator() -> Iterator[Dict[str, Union[torch.Tensor, TensorPointer]]]:
        # [FUNCTION DOC]
        # HELPER: the actual infinite batch generator returned by dummy_infinite_data_generator.
        # Behavior, inputs: described in the enclosing function; uses closure variables only.
        # Outputs: yields batch dictionaries forever.
        # Exceptions: None explicitly raised.
        # Side effects: creates a CUDA generator, allocates CUDA tensors each step.
        # Random generator
        generator = torch.Generator(device="cuda")
        # Make sure that TP are synced always
        generator.manual_seed(
            seed * (1 + dist.get_rank(parallel_context.dp_pg)) * (1 + dist.get_rank(parallel_context.pp_pg))
        )

        if use_position_ids:
            document_lengths = [[4, 6, sequence_length - 10]] + [[sequence_length]] * (micro_batch_size - 1)
            position_ids = torch.full(
                (micro_batch_size, sequence_length), fill_value=-1, dtype=torch.long, device="cuda"
            )
            for i in range(micro_batch_size):
                prev_idx = 0
                for doc_idx, doc_len in enumerate(document_lengths[i]):
                    position_ids[i, prev_idx : prev_idx + doc_len] = torch.arange(
                        0, doc_len, dtype=torch.long, device="cuda"
                    )
                    prev_idx += doc_len
            while True:
                yield {
                    "input_ids": torch.randint(
                        0,
                        vocab_size,
                        (micro_batch_size, sequence_length),
                        dtype=torch.long,
                        device="cuda",
                        generator=generator,
                    )[:, local_slice]
                    if dist.get_rank(parallel_context.pp_pg) == input_pp_rank
                    else TensorPointer(group_rank=input_pp_rank),
                    "position_ids": position_ids[:, local_slice]
                    if dist.get_rank(parallel_context.pp_pg) == input_pp_rank
                    else TensorPointer(group_rank=input_pp_rank),
                    "label_ids": torch.randint(
                        0,
                        vocab_size,
                        (micro_batch_size, sequence_length),
                        dtype=torch.long,
                        device="cuda",
                        generator=generator,
                    )[:, local_slice]
                    if dist.get_rank(parallel_context.pp_pg) == output_pp_rank
                    else TensorPointer(group_rank=output_pp_rank),
                    "label_mask": torch.ones(
                        micro_batch_size,
                        sequence_length,
                        dtype=torch.bool,
                        device="cuda",
                    )[:, local_slice]
                    if dist.get_rank(parallel_context.pp_pg) == output_pp_rank
                    else TensorPointer(group_rank=output_pp_rank),
                }
        else:
            while True:
                yield {
                    "input_ids": torch.randint(
                        0,
                        vocab_size,
                        (micro_batch_size, sequence_length),
                        dtype=torch.long,
                        device="cuda",
                        generator=generator,
                    )[:, local_slice]
                    if dist.get_rank(parallel_context.pp_pg) == input_pp_rank
                    else TensorPointer(group_rank=input_pp_rank),
                    "input_mask": torch.ones(
                        micro_batch_size,
                        sequence_length,
                        dtype=torch.bool,
                        device="cuda",
                    )[:, local_slice]
                    if dist.get_rank(parallel_context.pp_pg) == input_pp_rank
                    else TensorPointer(group_rank=input_pp_rank),
                    "label_ids": torch.randint(
                        0,
                        vocab_size,
                        (micro_batch_size, sequence_length),
                        dtype=torch.long,
                        device="cuda",
                        generator=generator,
                    )[:, local_slice]
                    if dist.get_rank(parallel_context.pp_pg) == output_pp_rank
                    else TensorPointer(group_rank=output_pp_rank),
                    "label_mask": torch.ones(
                        micro_batch_size,
                        sequence_length,
                        dtype=torch.bool,
                        device="cuda",
                    )[:, local_slice]
                    if dist.get_rank(parallel_context.pp_pg) == output_pp_rank
                    else TensorPointer(group_rank=output_pp_rank),
                }

    return data_generator


def set_tensor_pointers(
    input_dict: Dict[str, Union[torch.Tensor, TensorPointer]], group: dist.ProcessGroup, group_rank: int
) -> Dict[str, Union[torch.Tensor, TensorPointer]]:
    # [FUNCTION DOC]
    # Summary: keeps values only on group_rank; every other rank of group receives TensorPointers to group_rank.
    # Inputs: input_dict (dictionary of tensors or pointers), group (process group), group_rank (owning rank in group).
    # Outputs: new dictionary with the same keys.
    # Exceptions: None explicitly raised.
    # Side effects: None.
    """
    Make sure only the group_rank rank has the data, others have TensorPointers.

    Args:
        input_dict: Dictionary of tensors or tensor pointers
        group: Process group
        group_rank: Rank in the group that should have the actual data

    Returns:
        Dictionary with tensors or pointers based on current rank
    """
    return {
        k: v if dist.get_rank(group) == group_rank else TensorPointer(group_rank=group_rank)
        for k, v in input_dict.items()
    }


def get_dataloader_worker_init(dp_rank: int):
    # [FUNCTION DOC]
    # Summary: builds a DataLoader worker_init_fn giving each worker a distinct but reproducible seed.
    # Inputs: dp_rank, integer data-parallel rank.
    # Outputs: function taking worker_id.
    # Exceptions: None explicitly raised.
    # Side effects: None (the returned function sets global RNG seeds).
    """
    Creates random states for each worker in order to get different state in each workers.

    Args:
        dp_rank: Data parallel rank

    Returns:
        Worker initialization function to be used by DataLoader
    """

    def dataloader_worker_init(worker_id):
        # [FUNCTION DOC]
        # HELPER: worker init function; seed is 2 to the power (1 + worker_id) times 3 to the power (1 + dp_rank), modulo 2^32,
        # so it is identical across TP/PP ranks and distinct across DP ranks and workers.
        # Inputs: worker_id, integer.
        # Outputs: None.
        # Exceptions: None explicitly raised.
        # Side effects: calls set_random_seed, reseeding Python, numpy and torch RNGs in the worker process.
        # Dataloader is TP/PP synced in random states
        seed = 2 ** (1 + worker_id) * 3 ** (1 + dp_rank) % (2**32)
        set_random_seed(seed)

    return dataloader_worker_init


def get_train_dataloader(
    train_dataset: "datasets.Dataset",
    sequence_length: int,
    parallel_context: ParallelContext,
    input_pp_rank: int,
    output_pp_rank: int,
    micro_batch_size: int,
    consumed_train_samples: int,
    dataloader_num_workers: int,
    seed_worker: int,
    dataloader_drop_last: bool = True,
    dataloader_pin_memory: bool = True,
    use_loop_to_round_batch_size: bool = False,
    sequence_sep_tokens: List[int] = None,
    use_position_ids: bool = True,
) -> DataLoader:
    # [FUNCTION DOC]
    # Summary: creates the training DataLoader for a Hugging Face datasets.Dataset of pre-tokenised samples.
    # Behavior: reject non datasets.Dataset inputs. On pipeline ranks that own inputs or labels, format the dataset as
    # numpy over input_ids (keeping other columns). On other ranks, assert the only column is input_ids, remember the
    # length, drop the column and replace the dataset by EmptyInfiniteDataset of that length, with 0 workers. Choose
    # DataCollatorForCLMWithPositionIds if use_position_ids else DataCollatorForCLM. Build a sampler with get_sampler using
    # the DP rank/size, seed_worker, micro_batch_size, drop_last and consumed_train_samples (to resume). Return a DataLoader
    # with that sampler, collator, worker count, pin memory, worker init function and persistent workers when workers > 0.
    # Inputs: train_dataset, sequence_length, parallel_context, input_pp_rank, output_pp_rank, micro_batch_size,
    # consumed_train_samples, dataloader_num_workers, seed_worker, dataloader_drop_last (default True),
    # dataloader_pin_memory (default True), use_loop_to_round_batch_size (default False), sequence_sep_tokens (list of
    # separator token ids, default None), use_position_ids (default True).
    # Outputs: torch DataLoader.
    # Exceptions: ValueError for wrong dataset type; AssertionError on non-data ranks if columns are unexpected;
    # TypeError (see known issue).
    # Side effects: None beyond constructing objects.
    # KNOWN ISSUE: sequence_sep_tokens is passed to DataCollatorForCLMWithPositionIds, which has no such field, so the
    # default use_position_ids path raises TypeError.
    """
    Get a PyTorch DataLoader for training.

    Args:
        train_dataset: Dataset to use for training
        sequence_length: Maximum sequence length
        parallel_context: Object containing process groups information
        input_pp_rank: Rank responsible for input data
        output_pp_rank: Rank responsible for output/label data
        micro_batch_size: Size of each micro batch
        consumed_train_samples: Number of samples already consumed
        dataloader_num_workers: Number of workers for the DataLoader
        seed_worker: Random seed for workers
        dataloader_drop_last: Whether to drop the last incomplete batch
        dataloader_pin_memory: Whether to use pinned memory for faster data transfer
        use_loop_to_round_batch_size: Whether to loop at the end of dataset to ensure batch size multiple
        sequence_sep_tokens: List of integers representing the sequence separator tokens, used to generate position ids e.g. [tokenizer.bos_token, tokenizer.eos_token, tokenizer.pad_token, tokenizer.unk_token]
        use_position_ids: Whether to use position IDs in the collator

    Returns:
        PyTorch DataLoader for training
    """
    if not isinstance(train_dataset, datasets.Dataset):
        raise ValueError(f"training requires a datasets.Dataset, but got {type(train_dataset)}")

    # Case of ranks requiring data
    if dist.get_rank(parallel_context.pp_pg) in [
        input_pp_rank,
        output_pp_rank,
    ]:
        train_dataset = train_dataset.with_format(type="numpy", columns=["input_ids"], output_all_columns=True)

    # Case of ranks not requiring data. We give them an infinite dummy dataloader
    else:
        #
        assert train_dataset.column_names == ["input_ids"], (
            f"Dataset has to have a single column, with `input_ids` as the column name. "
            f"Current dataset: {train_dataset}"
        )
        dataset_length = len(train_dataset)
        train_dataset = train_dataset.remove_columns(column_names="input_ids")
        assert (
            len(train_dataset) == 0
        ), f"Dataset has to be empty after removing the `input_ids` column. Current dataset: {train_dataset}"
        # HACK as if we remove the last column of a train_dataset, it becomes empty and it's number of rows becomes empty.
        train_dataset = EmptyInfiniteDataset(length=dataset_length)
        # No need to spawn a lot of workers, we can just use main
        dataloader_num_workers = 0

    if use_position_ids:
        # assert sequence_sep_tokens is not None, "sequence_sep_tokens must be provided if use_position_ids is True"
        data_collator = DataCollatorForCLMWithPositionIds(
            sequence_length=sequence_length,
            input_pp_rank=input_pp_rank,
            output_pp_rank=output_pp_rank,
            parallel_context=parallel_context,
            sequence_sep_tokens=sequence_sep_tokens,
        )
    else:
        data_collator = DataCollatorForCLM(
            sequence_length=sequence_length,
            input_pp_rank=input_pp_rank,
            output_pp_rank=output_pp_rank,
            parallel_context=parallel_context,
        )

    # Compute size and rank of dataloader workers
    dp_ranks_size = parallel_context.dp_pg.size()
    dp_rank = parallel_context.dp_pg.rank()

    train_sampler = get_sampler(
        dl_rank=dp_rank,
        dl_ranks_size=dp_ranks_size,
        train_dataset=train_dataset,
        seed=seed_worker,
        use_loop_to_round_batch_size=use_loop_to_round_batch_size,
        micro_batch_size=micro_batch_size,
        drop_last=dataloader_drop_last,
        consumed_train_samples=consumed_train_samples,
    )

    return DataLoader(
        train_dataset,
        batch_size=micro_batch_size,
        sampler=train_sampler,
        collate_fn=data_collator,
        drop_last=dataloader_drop_last,
        num_workers=dataloader_num_workers,
        pin_memory=dataloader_pin_memory,
        worker_init_fn=get_dataloader_worker_init(dp_rank=dp_rank),
        persistent_workers=True if dataloader_num_workers > 0 else False,
    )
