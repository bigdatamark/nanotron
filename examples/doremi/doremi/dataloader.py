# [MODULE DOC]
# Purpose: Data loading for the DoReMi example: a combined multi-domain dataset, a CLM collator that also emits domain ids, a distributed batch sampler that builds each global batch according to the current DoReMi domain weights, and a get_dataloader that (for proxy training) attaches reference-model losses to every batch.
# Main contents: CombinedDataset, DataCollatorForCLM, DistributedSamplerForDoReMi, get_datasets, get_dataloader.
# How it fits: used by train_doremi.py / train_reference.py with DoReMiTrainer or ReferenceTrainer; the sampler reads DoReMiContext.domain_weights, which the proxy trainer updates every step.
# Import-time side effects: warns if the datasets library is missing (then Dataset base class is undefined and the module fails); creates a logger.
import dataclasses
import math
import warnings
from typing import Dict, List, Union

import numpy as np
import torch
from nanotron import distributed as dist
from nanotron import logging
from nanotron.data.dataloader import get_dataloader_worker_init
from nanotron.parallel import ParallelContext
from nanotron.parallel.pipeline_parallel.tensor_pointer import TensorPointer
from nanotron.parallel.pipeline_parallel.utils import get_input_output_pp_ranks
from nanotron.trainer import DistributedTrainer
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from .doremi_context import DoReMiContext

try:
    from datasets import Dataset, concatenate_datasets, load_from_disk
except ImportError:
    warnings.warn("Datasets and/or Transformers not installed, you'll be unable to use the dataloader.")


logger = logging.get_logger(__name__)


class CombinedDataset(Dataset):
    # [CLASS DOC]
    # Purpose: Concatenates per-domain HF datasets into one indexable dataset so global indices (domain offset plus local index) can be used by the sampler.
    # Attributes: comebined_dataset (the concatenated HF dataset).
    # Inheritance: subclasses datasets.Dataset only nominally; the parent constructor is not called.
    def __init__(self, datasets):
        # [FUNCTION DOC]
        # Behavior: Concatenates the given datasets with concatenate_datasets.
        # Inputs: datasets: list of HF datasets with compatible columns.
        # Outputs: None. Exceptions: errors from concatenate_datasets on incompatible schemas. Side effects: sets self.comebined_dataset.
        self.comebined_dataset = concatenate_datasets(datasets)

    def __len__(self):
        # [FUNCTION DOC]
        # Behavior: Returns the total number of rows in the concatenated dataset.
        # Inputs: self. Outputs: int. Exceptions: None explicitly raised. Side effects: None.
        return len(self.comebined_dataset)

    def __getitem__(self, batch):
        # [FUNCTION DOC]
        # Behavior: Wraps a scalar index into a one-element list; asserts the list is non-empty; if it is a list of lists, fetches each sub-list and merges the column arrays with _merge_dicts; otherwise fetches all indices at once.
        # Inputs: batch: int, list of ints, or list of lists of ints.
        # Outputs: dictionary mapping column name to batched values (as given by the HF dataset format, numpy for input_ids).
        # Exceptions: AssertionError on empty list; IndexError on out-of-range indices.
        # Side effects: None.
        if isinstance(batch, list) is False:
            batch = [batch]

        assert len(batch) > 0
        if isinstance(batch[0], list):
            # TODO(xrsrke): do a single index, then split the output
            samples = [self.comebined_dataset[idxs] for idxs in batch]
            return self._merge_dicts(samples)

        return self.comebined_dataset[batch]

    def _merge_dicts(self, data):
        # [FUNCTION DOC]
        # HELPER: merges batches fetched for several index lists.
        # Behavior: For each key of the first dictionary, concatenates along axis 0 the arrays from every dictionary containing that key.
        # Inputs: data: list of dictionaries of arrays. Outputs: merged dictionary of numpy arrays.
        # Exceptions: IndexError if data is empty. Side effects: None.
        merged = {}
        for key in data[0].keys():
            merged[key] = np.concatenate([d[key] for d in data if key in d])
        return merged


@dataclasses.dataclass
class DataCollatorForCLM:
    # [CLASS DOC]
    # Purpose: Causal LM collator for DoReMi; like nanotron's collator but also forwards domain ids. Pipeline ranks other than the input and output ranks get TensorPointers.
    # Attributes: sequence_length, input_pp_rank, output_pp_rank, parallel_context, doremi_context.
    """
    Data collator used for causal language modeling.

    - input_pp_rank: Discards last input id token
    - output_pp_rank: Discards first label id token
    - other pp ranks: Don't have data. Instead, we use `TensorPointer` to point to the rank having the data.
    """

    sequence_length: int
    input_pp_rank: int
    output_pp_rank: int
    parallel_context: ParallelContext
    doremi_context: DoReMiContext

    def __call__(self, examples: List[Dict[str, List[np.ndarray]]]) -> Dict[str, Union[torch.Tensor, TensorPointer]]:
        # Process the case when current rank doesn't require data. We return `TensorPointer` that points to ranks having the data.
        # [FUNCTION DOC]
        # Behavior: Gets this rank's pipeline rank. If it is neither input nor output rank, asserts all examples are empty and returns TensorPointers for input_ids/input_mask (to the input rank) and label_ids/label_mask (to the output rank). Otherwise asserts each example has exactly keys input_ids and domain_ids, stacks input_ids into a (batch, seq+1) array and asserts width seq+1; starts all four fields as TensorPointers, then on the input rank sets input_ids to all but the last token with an all-true mask, and on the output rank sets label_ids to all but the first token with an all-true mask; stacks domain_ids into domain_idxs; converts all numpy arrays to torch tensors.
        # Inputs: examples: list of dictionaries with input_ids (length seq+1) and domain_ids.
        # Outputs: dictionary of torch tensors or TensorPointers with keys input_ids, input_mask, label_ids, label_mask and (on data ranks) domain_idxs.
        # Exceptions: AssertionError on unexpected keys/lengths or non-empty examples on non-data ranks; ValueError checks for wrong lengths (never reached, see below).
        # Side effects: None.
        # KNOWN ISSUE: the length ValueError checks test for torch.Tensor before the numpy-to-torch conversion, so they are dead code; their message also says labels for the input_ids check.
        current_pp_rank = dist.get_rank(self.parallel_context.pp_pg)
        if current_pp_rank not in [
            self.input_pp_rank,
            self.output_pp_rank,
        ]:
            assert all(len(example) == 0 for example in examples)
            return {
                "input_ids": TensorPointer(self.input_pp_rank),
                "input_mask": TensorPointer(self.input_pp_rank),
                "label_ids": TensorPointer(self.output_pp_rank),
                "label_mask": TensorPointer(self.output_pp_rank),
            }

        assert all(list(example.keys()) == ["input_ids", "domain_ids"] for example in examples)

        input_ids = np.vstack([examples[i]["input_ids"] for i in range(len(examples))])  # (b, s)
        batch_size, expanded_input_length = input_ids.shape

        result: Dict[str, Union[np.ndarray, TensorPointer]] = {}

        result["input_ids"] = TensorPointer(group_rank=self.input_pp_rank)
        result["input_mask"] = TensorPointer(group_rank=self.input_pp_rank)
        result["label_ids"] = TensorPointer(group_rank=self.output_pp_rank)
        result["label_mask"] = TensorPointer(group_rank=self.output_pp_rank)

        assert (
            expanded_input_length == self.sequence_length + 1
        ), f"Samples should be of length {self.sequence_length + 1} (seq_len+1), but got {expanded_input_length}"

        # Process inputs: last token is the label
        if current_pp_rank == self.input_pp_rank:
            result["input_ids"] = input_ids[:, :-1]
            result["input_mask"] = np.ones((batch_size, self.sequence_length), dtype=np.bool_)

        # Process labels: shift them to the left
        if current_pp_rank == self.output_pp_rank:
            result["label_ids"] = input_ids[:, 1:]
            result["label_mask"] = np.ones((batch_size, self.sequence_length), dtype=np.bool_)

        result["domain_idxs"] = np.vstack([examples[i]["domain_ids"] for i in range(len(examples))])

        if isinstance(result["input_ids"], torch.Tensor) and result["input_ids"].shape[-1] != self.sequence_length:
            raise ValueError(
                f"`labels` are incorrectly preprocessed. `labels` length is {result['input_ids'].shape[-1]}, but should be"
                f" {self.sequence_length}."
            )
        if isinstance(result["label_ids"], torch.Tensor) and result["label_ids"].shape[-1] != self.sequence_length:
            raise ValueError(
                f"`labels` are incorrectly preprocessed. `labels` length is {result['label_ids'].shape[-1]}, but should be"
                f" {self.sequence_length}."
            )

        # Cast np.array to torch.Tensor
        result = {k: v if isinstance(v, TensorPointer) else torch.from_numpy(v) for k, v in result.items()}
        return result


class DistributedSamplerForDoReMi(DistributedSampler):
    # [CLASS DOC]
    # Purpose: Batch sampler yielding micro-batches of global indices into CombinedDataset, where each global batch is composed of per-domain sample counts proportional to the current DoReMi domain weights, then split across data-parallel ranks and micro-batches.
    # Key state: datasets, batch_size (micro batch size), num_microbatches, doremi_context, parallel_context, global_batch_size (batch_size * num_replicas * num_microbatches), lengths and offsets of each domain in the combined dataset, a CPU torch Generator seeded from seed and dp/pp ranks, domain_indices (per-domain global index lists), domain_counters (next unread position per domain), microbatch_idx, batch (current global batch indices).
    # Inheritance: subclasses torch DistributedSampler for num_replicas/rank bookkeeping only; iteration is fully overridden and the sampler is its own iterator.
    # Invariants: each global batch has every domain with at least one sample and totals global_batch_size; samples are taken in sequential (non-shuffled) order per domain.
    def __init__(
        self,
        datasets: List[Dataset],
        batch_size: int,
        num_microbatches: int,
        num_replicas: int,
        rank: int,
        doremi_context: DoReMiContext,
        parallel_context: ParallelContext,
        shuffle: bool = False,
        seed: int = 42,
        drop_last: bool = False,
    ):
        # [FUNCTION DOC]
        # Behavior: Asserts one dataset per domain weight, initializes the DistributedSampler parent, stores arguments, computes total size, domain lengths and cumulative offsets, global batch size, seeds a CPU generator with seed*(1+dp rank)*(1+pp rank), and calls reset.
        # Inputs: datasets (list of per-domain datasets), batch_size (int micro batch size), num_microbatches (int), num_replicas (int dp size), rank (int dp rank), doremi_context, parallel_context, shuffle (default False, unused beyond parent), seed (default 42), drop_last (default False).
        # Outputs: None. Exceptions: AssertionError on domain count mismatch.
        # Side effects: queries process-group ranks.
        # KNOWN ISSUE: the generator seed depends on the dp rank, so the random rounding in _round_up_domain_batch_sizes can differ between dp ranks, making their views of the global batch inconsistent (possible duplicated/missed samples).
        assert len(datasets) == len(
            doremi_context.domain_weights
        ), "The number of datasets must equal to the number of domain weights"

        super().__init__(datasets, num_replicas=num_replicas, rank=rank, shuffle=shuffle, drop_last=drop_last)

        self.datasets = datasets
        self.batch_size = batch_size
        self.num_microbatches = num_microbatches
        self.doremi_context = doremi_context
        self.parallel_context = parallel_context
        self.total_size = self._calculate_total_size()

        self.lengths = [len(d) for d in self.datasets]
        self.offsets = np.cumsum([0] + self.lengths[:-1])
        self.seed = seed

        # self.global_batch_size = batch_size * dist.get_world_size(parallel_context.dp_pg) * num_microbatches
        self.global_batch_size = batch_size * self.num_replicas * num_microbatches
        # NOTE: Reset the seed of the generator for consistent randomness across epochs
        self.generator = torch.Generator(device="cpu").manual_seed(
            seed * (1 + dist.get_rank(self.parallel_context.dp_pg)) * (1 + dist.get_rank(self.parallel_context.pp_pg))
        )

        self.reset()

    def _calculate_total_size(self):
        # [FUNCTION DOC]
        # HELPER: computes total_size for the parent sampler.
        # Behavior: Rounds the total number of samples across domains up to a multiple of batch_size.
        # Inputs: self. Outputs: int. Exceptions: None explicitly raised. Side effects: None.
        total_samples = sum(len(d) for d in self.datasets)
        return math.ceil(total_samples / self.batch_size) * self.batch_size

    def __iter__(self):
        # [FUNCTION DOC]
        # Behavior: Returns the sampler itself as the iterator (no reset per epoch).
        # Inputs: self. Outputs: self. Exceptions: None. Side effects: None.
        return self

    def _recompute_domain_batch_sizes(self, domain_weights):
        # [FUNCTION DOC]
        # HELPER: converts domain weights to per-domain sample counts for one global batch.
        # Behavior: Rounds global_batch_size times each weight, bumps zeros to 1 so every domain keeps getting gradient signal, and if the sum differs from global_batch_size calls _round_up_domain_batch_sizes; asserts all counts are positive.
        # Inputs: domain_weights: tensor [num_domains]. Outputs: list of ints summing to global_batch_size.
        # Exceptions: AssertionError if a domain ends with zero samples.
        # Side effects: consumes the sampler generator when rounding.
        domain_batch_sizes = [round(self.global_batch_size * weight.item()) for weight in domain_weights]

        # NOTE: in some cases, the weight of a domain is too small
        # resulting in a domain with 0 samples per global batch
        # => zero loss for that domain => we no longer update the weights of that domain
        # so we add a sample to that domain
        domain_batch_sizes = [1 if x < 1 else x for x in domain_batch_sizes]

        if sum(domain_batch_sizes) != self.global_batch_size:
            # NOTE: randomly add a sample to round it up
            domain_batch_sizes = self._round_up_domain_batch_sizes(
                domain_batch_sizes,
                target_total_size=self.global_batch_size,
            )

        assert all(x > 0 for x in domain_batch_sizes), "There is a domain with 0 samples per global batch"
        return domain_batch_sizes

    def __next__(self):
        # [FUNCTION DOC]
        # Behavior: At the start of each global batch (microbatch_idx 0) recomputes domain batch sizes from current weights and fills self.batch by taking the next domain_batch_size indices of each domain (advancing domain_counters). Then selects this dp rank's contiguous slice of batch_size*num_microbatches indices, and from it the slice for the current micro-batch; advances microbatch_idx cyclically.
        # Inputs: self. Outputs: list of global sample indices of length batch_size.
        # Exceptions: StopIteration when a domain, the dp slice or the micro-batch slice runs out of samples; AssertionError from consistency checks.
        # Side effects: mutates domain_counters, batch, domain_batch_sizes, microbatch_idx and generator state.
        if self.microbatch_idx == 0:
            # NOTE: because we randomly add a sample to round up the domain batch sizes
            # so it's better if we recompute the global batch every time we start a new microbatch
            # so that not bias towards a domain (where that domain gets more samples than the others)
            self.domain_batch_sizes = self._recompute_domain_batch_sizes(
                domain_weights=self.doremi_context.domain_weights,
            )

            self.batch = []
            for domain_index, (idxs, domain_batch_size) in enumerate(
                zip(self.domain_indices, self.domain_batch_sizes)
            ):
                start_idx = self.domain_counters[domain_index]
                end_idx = start_idx + domain_batch_size

                if end_idx > len(idxs):
                    raise StopIteration(f"Domain {domain_index}-th ran out of samples")

                assert self.domain_counters[domain_index] + domain_batch_size == end_idx
                self.domain_counters[domain_index] = end_idx
                global_batch_idxs = idxs[start_idx:end_idx]
                self.batch.extend(global_batch_idxs)

        num_samples_per_dp_rank = self.batch_size * self.num_microbatches
        dp_start_idx = self.rank * num_samples_per_dp_rank
        dp_end_idx = dp_start_idx + num_samples_per_dp_rank

        if dp_end_idx > len(self.batch):
            raise StopIteration(f"[DoReMi] Rank {self.rank} ran out of samples, len(batch)={len(self.batch)}")

        dp_batch = self.batch[dp_start_idx:dp_end_idx]

        microbatch_start_idx = self.microbatch_idx * self.batch_size
        microbatch_end_idx = microbatch_start_idx + self.batch_size

        if microbatch_end_idx > len(dp_batch):
            raise StopIteration(
                f"[DoReMi] Rank {self.rank}'s microbatch {self.microbatch_idx}-th ran out of samples, len(dp_batch)={len(dp_batch)}"
            )

        microbatch_idxs = dp_batch[microbatch_start_idx:microbatch_end_idx]

        if self.microbatch_idx == self.num_microbatches - 1:
            self.microbatch_idx = 0
        else:
            self.microbatch_idx += 1

        return microbatch_idxs

    def _recompute_global_batch(self):
        # [FUNCTION DOC]
        # HELPER: alternative global-batch builder.
        # Behavior: Recomputes domain batch sizes and appends the next indices of each domain to self.batch, advancing counters.
        # Inputs: self. Outputs: None.
        # Exceptions: StopIteration when a domain runs out.
        # Side effects: mutates domain_batch_sizes, domain_counters and batch.
        # KNOWN ISSUE: unused anywhere (dead code) and appends to batch without clearing it.
        self.domain_batch_sizes = self._recompute_domain_batch_sizes(
            domain_weights=self.doremi_context.domain_weights,
        )
        for domain_index, (idxs, domain_batch_size) in enumerate(zip(self.domain_indices, self.domain_batch_sizes)):
            start_idx = self.domain_counters[domain_index]
            end_idx = start_idx + domain_batch_size

            if end_idx > len(idxs):
                raise StopIteration(f"Domain {domain_index}-th ran out of samples")

            self.domain_counters[domain_index] = end_idx
            global_batch_idxs = idxs[start_idx:end_idx]
            self.batch.extend(global_batch_idxs)

    def _round_up_domain_batch_sizes(self, domain_batch_sizes: List[int], target_total_size: int) -> List[int]:
        # [FUNCTION DOC]
        # HELPER: fixes rounding so per-domain counts sum to the target.
        # Behavior: Loops until the sum equals target_total_size: picks uniformly at random (with the sampler generator) a domain whose count is greater than 1, adding one if the total is too small or subtracting one if too large.
        # Inputs: domain_batch_sizes: list of ints (mutated); target_total_size: int.
        # Outputs: the same list, adjusted.
        # Exceptions: RuntimeError from randint if no domain has more than one sample.
        # Side effects: mutates the input list; consumes generator state.
        # KNOWN ISSUE: only domains with more than one sample are eligible even when adding, and the greater-than-zero check on decrement is redundant.
        """
        NOTE: Makes sum(domain_batch_sizes) == batch_size
        """
        total_batch_size = sum(domain_batch_sizes)
        while total_batch_size != target_total_size:
            diff = target_total_size - total_batch_size

            # NOTE: Randomly select a domain to increase/decrase a sample
            # to match the target_total_size
            eligible_indices = torch.nonzero(torch.tensor(domain_batch_sizes) > 1).view(-1)
            random_index = torch.randint(
                low=0, high=len(eligible_indices), size=(1,), generator=self.generator, device="cpu"
            ).item()
            selected_domain = eligible_indices[random_index].item()

            if diff > 0:
                domain_batch_sizes[selected_domain] += 1
            elif diff < 0 and domain_batch_sizes[selected_domain] > 0:
                domain_batch_sizes[selected_domain] -= 1

            total_batch_size = sum(domain_batch_sizes)

        return domain_batch_sizes

    def reset(self):
        # [FUNCTION DOC]
        # Behavior: Resets micro-batch index and per-domain counters, and rebuilds each domain's global index list as its local range shifted by the domain offset; records samples per global step and expected total.
        # Inputs: self. Outputs: None. Exceptions: None explicitly raised.
        # Side effects: mutates sampler state.
        # KNOWN ISSUE: total_samples_yielded and out_of_samples are set but never used; self.batch is not cleared here.
        """Reset the state of the sampler for a new epoch."""
        self.microbatch_idx = 0
        self.domain_counters = [0 for _ in self.datasets]
        self.total_samples_yielded = 0
        self.out_of_samples = False

        domain_indices = []
        for i, dataset in enumerate(self.datasets):
            local_indices = torch.arange(0, len(dataset), device="cpu").tolist()

            # NOTE: align the indices across the combined dataset
            global_indices = local_indices + self.offsets[i]
            domain_indices.append(global_indices)

        self.num_samples_per_global_step = self.batch_size * self.num_microbatches * self.num_replicas
        self.domain_indices = domain_indices
        self.expected_total_samples = sum([len(d) for d in domain_indices])


def get_datasets(paths):
    # [FUNCTION DOC]
    # Behavior: Loads each path with datasets.load_from_disk (with a tqdm progress bar) and returns them in order.
    # Inputs: paths: list of directory paths. Outputs: list of HF datasets.
    # Exceptions: errors from load_from_disk on missing paths. Side effects: disk reads; progress bar output.
    datasets = []
    for path in tqdm(paths, desc="Loading dataset from disk"):
        d = load_from_disk(path)
        datasets.append(d)

    return datasets


def get_dataloader(trainer: DistributedTrainer, datasets) -> DataLoader:
    # [FUNCTION DOC]
    # Behavior: Sets each dataset to numpy format for input_ids (keeping other columns), determines input/output pipeline ranks, builds the DoReMi collator and DistributedSamplerForDoReMi (micro batch size, micro-batches per batch, dp size and rank, seed of the first data stage, drop_last), combines the datasets, and builds a DataLoader with that batch sampler and worker init. If the context is proxy, wraps it into a generator that moves each batch to CUDA, runs the reference model on the batch without domain_idxs and adds its per-token losses as ref_losses.
    # Inputs: trainer: DistributedTrainer (DoReMi or reference trainer with doremi_context, and ref_model when proxy); datasets: list of per-domain HF datasets.
    # Outputs: a DataLoader (reference training) or a generator of batch dictionaries (proxy training).
    # Exceptions: errors from sampler/collator assertions.
    # Side effects: CUDA transfers and reference-model forward passes (including its pipeline communication) during iteration.
    # KNOWN ISSUE: the proxy wrapper calls .to on every batch value, which would fail for TensorPointer entries on pipeline ranks without data.
    doremi_context = trainer.doremi_context
    parallel_context = trainer.parallel_context

    datasets = [d.with_format(type="numpy", columns=["input_ids"], output_all_columns=True) for d in datasets]

    # TODO(xrsrke): decouple trainer from dataloader
    # TODO(xrsrke): decouple data collating from data loading
    input_pp_rank, output_pp_rank = get_input_output_pp_ranks(model=trainer.model)
    data_collator = DataCollatorForCLM(
        sequence_length=trainer.sequence_length,
        input_pp_rank=input_pp_rank,
        output_pp_rank=output_pp_rank,
        parallel_context=parallel_context,
        doremi_context=doremi_context,
    )

    sampler = DistributedSamplerForDoReMi(
        datasets,
        batch_size=trainer.micro_batch_size,
        num_microbatches=trainer.n_micro_batches_per_batch,
        num_replicas=parallel_context.dp_pg.size(),
        rank=dist.get_rank(parallel_context.dp_pg),
        seed=trainer.config.data_stages[0].data.seed,
        drop_last=True,
        doremi_context=doremi_context,
        parallel_context=parallel_context,
    )

    comebined_dataset = CombinedDataset(datasets)

    dataloader = DataLoader(
        comebined_dataset,
        batch_sampler=sampler,
        collate_fn=data_collator,
        num_workers=trainer.config.data_stages[0].data.num_loading_workers,
        pin_memory=True,
        worker_init_fn=get_dataloader_worker_init(dp_rank=dist.get_rank(parallel_context.dp_pg)),
    )

    def _data_generator(dataloader):
        # [FUNCTION DOC]
        # HELPER: builds a closure that adds reference losses to batches.
        # Behavior: Returns the inner generator function bound to the given dataloader.
        # Inputs: dataloader: DataLoader. Outputs: generator function. Exceptions: None. Side effects: None.
        def inner():
            # [FUNCTION DOC]
            # HELPER: generator yielding batches with ref_losses.
            # Behavior: For each batch, moves tensors to CUDA, removes domain_idxs for the reference model call, runs trainer.ref_model and stores the returned losses under ref_losses, then yields the batch.
            # Inputs: None (closure over dataloader and trainer). Outputs: yields dictionaries of tensors.
            # Exceptions: errors from the reference model forward. Side effects: GPU memory and compute; reference model communication.
            for batch in dataloader:
                batch = {k: v.to("cuda") for k, v in batch.items()}
                # NOTE: because the inference model don't take `domain_idxs`
                # as input we need to remove it from the batch
                batch_for_inference = {k: v for k, v in batch.items() if k != "domain_idxs"}

                ref_losses = trainer.ref_model(**batch_for_inference)["losses"]
                batch["ref_losses"] = ref_losses
                yield batch

        return inner

    dataloader = _data_generator(dataloader) if doremi_context.is_proxy is True else dataloader

    # NOTE: we need to call the dataloader to generate reference losses
    # if the model is a proxy model
    dataloader = dataloader() if doremi_context.is_proxy is True else dataloader

    return dataloader
