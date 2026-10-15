# [MODULE DOC]
# Purpose: autoregressive text generation for nanotron models under data, tensor and pipeline
# parallelism. Inputs are split into micro-batches (one per DP rank in round robin), up to as many
# micro-batches as pipeline stages are kept in flight so stages work concurrently, the last stage
# (decoder_logit_rank) samples the next token with a TP-aware sampler and sends it back via
# point-to-point to the first stage (decoder_input_rank), which appends it to the sequence. At the end
# results are broadcast across the pipeline group and yielded.
# Main contents: dataclasses GenerationInput, GenerationInputs, GenerationOutput, GenerationStates,
# TokenizerConfig; helpers chunks, micro_batcher, micro_splitter, get_position_ids,
# broadcast_tensors; entry points decode_text (from raw text, with optional benchmarking) and
# decode_tokenized (from pre-tokenized tensors, using the KV-cache store).
# Import-time side effects: creates a module logger; imports the Llama model and pipeline modules.
import dataclasses
import time
from itertools import chain, islice
from typing import TYPE_CHECKING, Generator, Iterable, List, Optional, Tuple, Union

import torch
from tqdm import tqdm

from nanotron import distributed as dist
from nanotron import logging
from nanotron.config import BenchArgs, GenerationArgs
from nanotron.distributed import ProcessGroup, get_global_rank
from nanotron.generation.generate_store import Store, attach_store
from nanotron.generation.sampler import BasicSampler, GreedySampler, SamplerType, TopKSampler, TopPSampler
from nanotron.helpers import log_throughput
from nanotron.models.llama import LlamaModel
from nanotron.parallel import ParallelContext
from nanotron.parallel.pipeline_parallel.block import get_min_max_rank
from nanotron.parallel.pipeline_parallel.context_manager import attach_pipeline_state_to_model
from nanotron.parallel.pipeline_parallel.p2p import P2PTensorMetaData, view_as_contiguous
from nanotron.parallel.pipeline_parallel.state import PipelineEvalBatchState
from nanotron.parallel.pipeline_parallel.tensor_pointer import TensorPointer
from nanotron.utils import get_untyped_storage

if TYPE_CHECKING:
    try:
        from transformers import PreTrainedTokenizer
    except ImportError:
        PreTrainedTokenizer = None


logger = logging.get_logger(__name__)


@dataclasses.dataclass
class GenerationInput:
    # [CLASS DOC]
    # Purpose: one raw prompt; field text (str).
    text: str


@dataclasses.dataclass
class GenerationInputs:
    # [CLASS DOC]
    # Purpose: one micro-batch of model inputs: input_ids (batch, seq) and input_masks (bool, same shape),
    # each a tensor on the input pipeline rank or a TensorPointer elsewhere.
    input_ids: Union[torch.Tensor, TensorPointer]  # [B, S]
    input_masks: Union[torch.Tensor, TensorPointer]


@dataclasses.dataclass
class GenerationOutput:
    # [CLASS DOC]
    # Purpose: one generated sample: input_ids (prompt tokens without padding), generation_ids (prompt plus
    # generated tokens, masked), optional return_logits (unused, default None). TensorPointer on non-input ranks.
    input_ids: Union[torch.Tensor, TensorPointer]
    generation_ids: Union[torch.Tensor, TensorPointer]
    return_logits: Optional[Union[torch.Tensor, TensorPointer]] = None


@dataclasses.dataclass
class GenerationStates:
    # [CLASS DOC]
    # Purpose: per micro-batch decoding state: new_input_ids/new_input_mask (latest tokens fed to the
    # model), store (KV-cache Store), generation_ids/generation_mask (lists of all chunks so far, to be
    # concatenated along the sequence dimension).
    new_input_ids: Union[torch.Tensor, TensorPointer]
    new_input_mask: Union[torch.Tensor, TensorPointer]
    store: Store

    # The rest of the state I need to reconstruct the generated output
    generation_ids: List[Union[torch.Tensor, TensorPointer]]
    generation_mask: List[Union[torch.Tensor, TensorPointer]]


@dataclasses.dataclass
class TokenizerConfig:
    # [CLASS DOC]
    # Purpose: tokenization options: max_input_length (optional int), truncation and padding (optional
    # str or bool, defaulted by micro_batcher).
    max_input_length: Optional[int]
    truncation: Optional[Union[str, bool]] = None
    padding: Optional[Union[str, bool]] = None


def chunks(iterable, chunk_size: int) -> Generator[List, None, None]:
    # [FUNCTION DOC]
    # HELPER: groups an iterable into lists of chunk_size (last may be shorter).
    # Inputs: iterable; chunk_size - int at least 1. Outputs: generator of lists.
    # Exceptions: AssertionError if chunk_size is below 1. Side effects: consumes the iterable lazily.
    """Yield successive n-sized chunks from `iterable`"""
    assert chunk_size >= 1
    iterator = iter(iterable)
    for first in iterator:
        yield list(chain([first], islice(iterator, chunk_size - 1)))


def micro_batcher(
    input_iter: Iterable[GenerationInput],
    tokenizer: "PreTrainedTokenizer",
    max_micro_batch_size: int,
    tokenizer_config: TokenizerConfig,
    parallel_context: ParallelContext,
    input_rank: int,
) -> Generator[GenerationInputs, None, None]:
    # [FUNCTION DOC]
    # HELPER: tokenizes prompts into micro-batches for decode_text.
    # Behavior: defaults padding to "max_length" when max_input_length is set else True, and truncation to
    # True when max_input_length is set else None (mutating tokenizer_config). Chunks inputs into groups
    # of max_micro_batch_size; micro-batch number i is handled only by the DP rank equal to i modulo DP
    # size. On the pipeline rank input_rank, tokenizes the texts to padded tensors, converts the attention
    # mask to bool, moves everything to CUDA and yields GenerationInputs; other pipeline ranks yield
    # TensorPointers to input_rank.
    # Inputs: input_iter - iterable of GenerationInput; tokenizer - HF tokenizer; max_micro_batch_size;
    # tokenizer_config; parallel_context; input_rank - pipeline rank holding inputs.
    # Outputs: generator of GenerationInputs. Exceptions: None explicitly raised.
    # Side effects: mutates tokenizer_config; CUDA allocations.
    """
    Returns:
        input_ids: [max_micro_batch_size, max_input_length]
        input_masks: [max_micro_batch_size, max_input_length]
    """
    if tokenizer_config.padding is None:
        tokenizer_config.padding = "max_length" if tokenizer_config.max_input_length is not None else True
    if tokenizer_config.truncation is None:
        tokenizer_config.truncation = True if tokenizer_config.max_input_length is not None else None

    for micro_batch_id, micro_batch in enumerate(chunks(input_iter, chunk_size=max_micro_batch_size)):
        if len(micro_batch) == 0:
            # Empty micro batches don't matter
            return

        if micro_batch_id % parallel_context.dp_pg.size() != dist.get_rank(parallel_context.dp_pg):
            # Each dp is responsible for its own micro batches
            continue

        if dist.get_rank(parallel_context.pp_pg) == input_rank:
            encodings = tokenizer(
                [elt.text for elt in micro_batch],
                return_tensors="pt",
                return_attention_mask=True,
                padding=tokenizer_config.padding,
                max_length=tokenizer_config.max_input_length,
                truncation=tokenizer_config.truncation,
                # pad_to_multiple_of=8
            )

            encodings["attention_mask"] = encodings.attention_mask.to(dtype=torch.bool, device="cuda")
            encodings.to("cuda")
            yield GenerationInputs(input_ids=encodings.input_ids, input_masks=encodings.attention_mask)
        else:
            yield GenerationInputs(
                input_ids=TensorPointer(group_rank=input_rank), input_masks=TensorPointer(group_rank=input_rank)
            )


def micro_splitter(
    input_ids: torch.Tensor,
    input_mask: torch.Tensor,
    max_micro_batch_size: int,
    parallel_context: ParallelContext,
    input_rank: int,
) -> Generator[GenerationInputs, None, None]:
    # [FUNCTION DOC]
    # HELPER: splits pre-tokenized tensors into micro-batches for decode_tokenized.
    # Behavior: splits input_ids and input_mask along the batch into pieces of max_micro_batch_size; on
    # input_rank yields clones (mask converted to bool on CUDA), elsewhere TensorPointers. All DP ranks
    # process all micro-batches (DP sharding is commented out).
    # Inputs: input_ids, input_mask - (batch, seq) tensors; max_micro_batch_size; parallel_context; input_rank.
    # Outputs: generator of GenerationInputs. Exceptions: None explicitly raised. Side effects: CUDA copies.
    # KNOWN ISSUE: the second move of the mask to CUDA discards its result (no-op), and input_ids are not moved to CUDA.
    """
    Returns:
        input_ids: [max_micro_batch_size, max_input_length]
        input_masks: [max_micro_batch_size, max_input_length]
    """
    for micro_batch_id, (micro_batch_ids, micro_batch_mask) in enumerate(
        zip(torch.split(input_ids, max_micro_batch_size), torch.split(input_mask, max_micro_batch_size))
    ):
        if len(micro_batch_ids) == 0:
            # Empty micro batches don't matter
            return

        # if micro_batch_id % parallel_context.dp_pg.size() != dist.get_rank(parallel_context.dp_pg):
        #     # Each dp is responsible for its own micro batches
        #     continue

        if dist.get_rank(parallel_context.pp_pg) == input_rank:
            micro_batch_mask = micro_batch_mask.to(dtype=torch.bool, device="cuda")
            micro_batch_mask.to("cuda")
            yield GenerationInputs(input_ids=micro_batch_ids.clone(), input_masks=micro_batch_mask.clone())
        else:
            yield GenerationInputs(
                input_ids=TensorPointer(group_rank=input_rank), input_masks=TensorPointer(group_rank=input_rank)
            )


@torch.inference_mode()
def get_position_ids(input_ids, tokenizer):
    # [FUNCTION DOC]
    # HELPER: computes position ids for left-padded batches.
    # Behavior: treats tokens equal to the tokenizer's eos id as padding; for each row containing padding,
    # the content start is one past the last padding token; positions are 0 up to the start and then
    # count up from 0 for the remaining tokens. Runs under inference mode.
    # Inputs: input_ids - (batch, seq) tensor; tokenizer - provides eos_token_id.
    # Outputs: int32 tensor (batch, seq). Exceptions: None explicitly raised. Side effects: None.
    # KNOWN ISSUE: any eos token inside the content (e.g. a previously generated eos) is treated as padding, resetting positions after it.
    # Find where padding ends for each sequence
    batch_size, seq_length = input_ids.shape
    padding_token_id = tokenizer.eos_token_id

    # Create a mask of padding tokens
    padding_mask = input_ids == padding_token_id

    # Find indices where non-padding tokens start
    # For sequences with no padding, this will be 0
    # For sequences with padding, this will be the first non-padding position
    non_padding_indices = torch.zeros(batch_size, dtype=torch.long, device=input_ids.device)

    # Only compute for rows that have padding
    rows_with_padding = padding_mask.any(dim=1).nonzero().squeeze(-1)
    if rows_with_padding.numel() > 0:
        for idx in rows_with_padding:
            # Find the last padding token position
            last_padding_pos = padding_mask[idx].nonzero()[-1].item()
            non_padding_indices[idx] = last_padding_pos + 1

    # Create position_ids tensor initialized with zeros
    position_ids = torch.zeros((batch_size, seq_length), dtype=torch.int32, device=input_ids.device)

    # Set up a range tensor we'll use for all sequences
    range_tensor = torch.arange(seq_length, device=input_ids.device)

    # Vectorized assignment of position IDs
    for i in range(batch_size):
        start_idx = non_padding_indices[i].item()
        # For positions after padding, use range starting from 0
        # For positions before or at padding end, keep as 0
        if start_idx < seq_length:
            position_ids[i, start_idx:] = range_tensor[: seq_length - start_idx]
    return position_ids


@torch.inference_mode()
def decode_text(
    input_iter: Iterable[GenerationInput],
    tokenizer: "PreTrainedTokenizer",
    model: LlamaModel,
    parallel_context: ParallelContext,
    generation_config: GenerationArgs,
    tokenizer_config: Optional[TokenizerConfig],
    max_micro_batch_size: int,
    max_new_tokens: int,
    is_bench: bool = False,
    logits_are_batch_first: bool = True,
) -> Generator[GenerationOutput, None, None]:
    # [FUNCTION DOC]
    # Behavior: finds the first and last pipeline ranks holding the model (get_min_max_rank). Chooses the
    # sampler type from generation_config (string or enum; GREEDY when no config). If n_samples is set,
    # replicates each prompt n_samples times (only allowed for TOP_P and TOP_K). Inside a pipeline eval
    # state, takes groups of up to pipeline-depth micro-batches from micro_batcher; for each of
    # max_new_tokens iterations and each micro-batch: concatenates all tokens so far (no KV cache, full
    # recompute), computes position ids, runs the model, reshapes logits to (batch, seq, vocab shard)
    # (optionally transposing if not batch-first), runs a schedule-dependent number of pending pipeline
    # communications, and on the last stage samples the next token from the final position's logits with a
    # default-parameter sampler over the TP group plus an all-true mask, keeping them locally if first and
    # last stage coincide or sending both to the first stage. The first stage then receives (or reuses)
    # the new tokens and new states are built by appending them. With is_bench, measures time and logs
    # throughput, TFLOPs and bandwidth via log_throughput. Finally flushes communications, and for each
    # micro-batch broadcasts generated ids/masks and inputs from the first stage over the pipeline group,
    # flushes the store, validates shapes and yields one GenerationOutput per sample (real tensors on the
    # first stage, TensorPointers elsewhere).
    # Inputs: input_iter - iterable of GenerationInput (all ranks receive all inputs); tokenizer; model -
    # pipelined LlamaModel; parallel_context; generation_config - GenerationArgs or None; tokenizer_config;
    # max_micro_batch_size; max_new_tokens; is_bench - default False; logits_are_batch_first - default True.
    # Outputs: generator of GenerationOutput.
    # Exceptions: ValueError for n_samples with other samplers; NotImplementedError if use_cache is set or
    # for an unknown sampler; AssertionError on pipeline bookkeeping or shape mismatches.
    # Side effects: P2P and collective communication across PP, TP groups; CUDA work; RNG; logging; tqdm progress output.
    # KNOWN ISSUE: SamplerType is built by value from an upper-cased string, but enum values are auto integers, so string samplers raise ValueError (lookup by name was likely intended).
    # KNOWN ISSUE: accessing use_cache on generation_config fails when generation_config is None, despite the None handling above.
    # KNOWN ISSUE: the code after raising NotImplementedError for use_cache is unreachable.
    # KNOWN ISSUE: samplers are built with default parameters, ignoring top_k, top_p and temperature from generation_config (unlike decode_tokenized).
    # KNOWN ISSUE: on ranks other than the input rank the inputs are TensorPointers, yet get_position_ids and view are called on them, which would fail for pipeline depth above 1.
    # KNOWN ISSUE: the first-iteration time is computed as start minus now (negative and taken at the start of iteration 0), so it is not excluded from the throughput time as the comment intends.
    # KNOWN ISSUE: the sequence-length assertion message prints the mask's batch dimension instead of its sequence dimension.
    """We assume the following:
    - Everyone receives ALL the input text. # TODO @thomasw21: technically only specific ranks need to receive input.
    - Only a specific rank will output the generated text_ids as `torch.Tensor`, the others return a `TensorPointer`. # TODO @thomasw21: Maybe all ranks should return the text.
    - We assume that within a model replica, the inputs are already synchronized.
    """
    decoder_input_rank, decoder_logit_rank = get_min_max_rank(module=model)

    if generation_config:
        if isinstance(generation_config.sampler, str):
            sampler_type = SamplerType(generation_config.sampler.upper())
        else:
            sampler_type = generation_config.sampler
    else:
        sampler_type = SamplerType.GREEDY

    # Compute flag
    is_decoder_input_rank = dist.get_rank(parallel_context.pp_pg) == decoder_input_rank
    is_decoder_logit_rank = dist.get_rank(parallel_context.pp_pg) == decoder_logit_rank
    max_nb_microbatches = decoder_logit_rank - decoder_input_rank + 1

    p2p = model.p2p

    # replicate input for n_samples times when using TOP_P or TOP_K samplers, in order to get diverse results
    if generation_config and generation_config.n_samples:
        if sampler_type != SamplerType.TOP_P and sampler_type != SamplerType.TOP_K:
            raise ValueError("Only support n_samples for TOP_P and TOP_K sampler")
        input_iter = [
            GenerationInput(text=input.text) for input in input_iter for _ in range(generation_config.n_samples)
        ]

    # That's annoying but I need this as soon as there's a change communication "cross"
    pipeline_state = PipelineEvalBatchState()
    with attach_pipeline_state_to_model(model=model, pipeline_state=pipeline_state):
        # We query the first `pipeline_size` batches
        for batches in chunks(
            iterable=micro_batcher(
                input_iter=input_iter,
                tokenizer=tokenizer,
                max_micro_batch_size=max_micro_batch_size,
                tokenizer_config=tokenizer_config,
                input_rank=decoder_input_rank,
                parallel_context=parallel_context,
            ),
            chunk_size=max_nb_microbatches,
        ):
            if len(batches) == 0:
                # It means we're out of element
                return

            # Number of micro batches
            number_states_in_buffer = len(batches)
            # Otherwise the pipelining doesn't work
            assert number_states_in_buffer <= max_nb_microbatches
            is_max_nb_microbatches = number_states_in_buffer == max_nb_microbatches

            # Initialize decoder states
            decoder_states: Iterable[GenerationStates] = (
                GenerationStates(
                    new_input_ids=batch.input_ids,
                    new_input_mask=batch.input_masks,
                    store=Store(),
                    generation_ids=[batch.input_ids],
                    generation_mask=[batch.input_masks],
                )
                for batch in batches
            )

            if is_bench:
                start_time, elapsed_time_first_iteration = time.perf_counter(), 0

            for generation_iter in tqdm(range(max_new_tokens), desc="Generating"):

                if is_bench and generation_iter == 0:
                    torch.cuda.synchronize()
                    elapsed_time_first_iteration = start_time - time.perf_counter()

                all_new_decoder_input_ids_and_mask_same_rank: List[
                    Tuple[Union[torch.LongTensor, TensorPointer], Union[torch.BoolTensor, TensorPointer]]
                ] = []
                new_decoder_states: List[GenerationStates] = []
                for state_id, state in enumerate(decoder_states):
                    new_decoder_states.append(state)
                    # Get the new logits
                    if generation_config.use_cache:
                        raise NotImplementedError("Use-cache is not supported for now")
                        with attach_store(model=model, store=state.store):
                            position_ids = get_position_ids(state.new_input_ids, tokenizer)
                            sharded_logits = model(
                                input_ids=state.new_input_ids,
                                position_ids=position_ids,  # [batch_size, seq_len]
                            )
                    else:
                        if isinstance(state.new_input_ids, torch.Tensor):
                            batch_generated_ids = torch.cat(state.generation_ids, dim=-1)
                            batch_generated_mask = torch.cat(state.generation_mask, dim=-1)
                        else:
                            batch_generated_ids = state.new_input_ids
                            batch_generated_mask = state.new_input_mask
                        position_ids = get_position_ids(batch_generated_ids, tokenizer)
                        sharded_logits = model(
                            input_ids=batch_generated_ids,
                            position_ids=position_ids,  # [batch_size, seq_len]
                        )  # [batch_size*seq_len, vocab_size]

                    sharded_logits = sharded_logits.view(*position_ids.shape, -1)  # [batch_size, seq_len, vocab_size]
                    if isinstance(sharded_logits, torch.Tensor) and not logits_are_batch_first:
                        sharded_logits = sharded_logits.transpose(0, 1)
                    # Communicate
                    nb_send: int = 0
                    if is_decoder_input_rank:
                        if is_max_nb_microbatches:
                            if generation_iter == 0:
                                if state_id == number_states_in_buffer - 1:
                                    # `2` is because we receive decoder_ids AND decoder_mask from last rank
                                    nb_send = len(pipeline_state.microbatches_activations_to_send) - 2
                                else:
                                    # Send everything
                                    nb_send = len(pipeline_state.microbatches_activations_to_send)
                            else:
                                # `2` is because we receive decoder_ids AND decoder_mask from last rank
                                nb_send = len(pipeline_state.microbatches_activations_to_send) - 2
                        else:
                            if number_states_in_buffer - 1 == state_id or generation_iter == 0:
                                # Send everything
                                nb_send = len(pipeline_state.microbatches_activations_to_send)
                            else:
                                # `2` is because we receive decoder_ids AND decoder_mask from last rank
                                nb_send = len(pipeline_state.microbatches_activations_to_send) - 2
                    else:
                        if state_id == number_states_in_buffer - 1:
                            if not is_max_nb_microbatches:
                                nb_send = len(pipeline_state.microbatches_activations_to_send)
                    for _ in range(nb_send):
                        pipeline_state.run_communication()

                    if is_decoder_logit_rank:
                        assert isinstance(sharded_logits, torch.Tensor)

                        # run a logit chooser.
                        if sampler_type == SamplerType.GREEDY:
                            sampler = GreedySampler(pg=parallel_context.tp_pg)
                        elif sampler_type == SamplerType.TOP_K:
                            sampler = TopKSampler(pg=parallel_context.tp_pg)
                        elif sampler_type == SamplerType.TOP_P:
                            sampler = TopPSampler(pg=parallel_context.tp_pg)
                        elif sampler_type == SamplerType.BASIC:
                            sampler = BasicSampler(pg=parallel_context.tp_pg)
                        else:
                            raise NotImplementedError(f"Sampler type {sampler_type} is not implemented")

                        new_decoder_input_ids = sampler(sharded_logits=sharded_logits[:, -1, :])

                        # TODO @thomasw21: Handle this correctly, ie from some point after <eos> this should only generate masked tokens
                        # TODO @thomasw21: Actually I can probably build this thing on the next device directly. Will save some communication
                        new_decoder_input_mask = torch.ones(
                            size=(new_decoder_input_ids.shape[0], 1),
                            dtype=torch.bool,
                            device=new_decoder_input_ids.device,
                        )

                        # TODO @thomasw21: We need to have stop condition.

                        # broadcast new_tokens to everyone
                        if decoder_input_rank == decoder_logit_rank:
                            # It's the same rank so no need to do anything too fancy
                            all_new_decoder_input_ids_and_mask_same_rank.append(
                                (new_decoder_input_ids, new_decoder_input_mask)
                            )
                        else:
                            pipeline_state.register_send_activation(
                                new_decoder_input_ids, to_rank=decoder_input_rank, p2p=p2p
                            )
                            pipeline_state.register_send_activation(
                                new_decoder_input_mask, to_rank=decoder_input_rank, p2p=p2p
                            )
                            if not is_max_nb_microbatches and state_id == number_states_in_buffer - 1:
                                # Send new_decoder_input_ids AND new_decoder_input_ids
                                pipeline_state.run_communication()
                                pipeline_state.run_communication()

                    else:
                        assert isinstance(sharded_logits, TensorPointer)

                all_new_decoder_input_ids_and_mask: Iterable[
                    Tuple[Union[torch.LongTensor, TensorPointer], Union[torch.BoolTensor, TensorPointer]]
                ]
                if is_decoder_input_rank:
                    # We receive the tensor from other ranks unless `decoder_input_rank` == `decoder_logit_rank` in which case `all_new_decoder_input_ids` is already populated.
                    if decoder_input_rank == decoder_logit_rank:
                        # `all_new_decoder_input_ids_and_mask_same_rank` is already populated. Since `decoder_input_rank` and `decoder_logit_rank` are the same, there's no need to communicate as we can just store the new input_ids in a list.
                        assert len(all_new_decoder_input_ids_and_mask_same_rank) == number_states_in_buffer
                        all_new_decoder_input_ids_and_mask = all_new_decoder_input_ids_and_mask_same_rank
                    else:

                        def generator():
                            # [FUNCTION DOC]
                            # HELPER: on the first stage, yields new token ids and masks received from the last stage.
                            # Behavior: for each micro-batch registers two receives from decoder_logit_rank, runs communication
                            # until two activations are buffered, and pops ids then mask.
                            # Inputs/Outputs: yields tuples (ids, mask). Exceptions: None. Side effects: P2P communication.
                            for _ in range(number_states_in_buffer):
                                pipeline_state.register_recv_activation(from_rank=decoder_logit_rank, p2p=p2p)
                                pipeline_state.register_recv_activation(from_rank=decoder_logit_rank, p2p=p2p)
                                while len(pipeline_state.activations_buffer) < 2:
                                    pipeline_state.run_communication()
                                new_decoder_input_ids = pipeline_state.activations_buffer.popleft()
                                new_decoder_input_mask = pipeline_state.activations_buffer.popleft()
                                yield new_decoder_input_ids, new_decoder_input_mask

                        all_new_decoder_input_ids_and_mask = iter(generator())
                else:
                    all_new_decoder_input_ids_and_mask = (
                        (TensorPointer(group_rank=decoder_input_rank), TensorPointer(group_rank=decoder_input_rank))
                        for _ in range(number_states_in_buffer)
                    )

                # Create new decoder states
                decoder_states = (
                    GenerationStates(
                        new_input_ids=new_decoder_input_ids_and_mask[0],
                        new_input_mask=new_decoder_input_ids_and_mask[1],
                        store=state.store,
                        generation_ids=state.generation_ids + [new_decoder_input_ids_and_mask[0]],
                        generation_mask=state.generation_mask + [new_decoder_input_ids_and_mask[1]],
                    )
                    for state, new_decoder_input_ids_and_mask in zip(
                        new_decoder_states, all_new_decoder_input_ids_and_mask
                    )
                )

            if is_bench:
                # Compute throughput (tok/s/gpu). Note that the first generation is done with full seq_len, so we don't count it.
                torch.cuda.synchronize()
                total_time_sec = time.perf_counter() - start_time - elapsed_time_first_iteration
                # We generate 1 token per iteration per batch (batch=microbatch)
                # Number of tokens generated every iteration: gbs/iteration_time
                global_batch_size = len(batches) * parallel_context.dp_pg.size()
                tokens_per_sec = global_batch_size * max_new_tokens / total_time_sec

                model_tflops, hardware_tflops = model.get_flops_per_sec(
                    iteration_time_in_sec=total_time_sec,
                    sequence_length=max_new_tokens,
                    global_batch_size=global_batch_size,
                )

                bench_config = BenchArgs(
                    model_name=model.config._name_or_path,
                    sequence_length=max_new_tokens,
                    micro_batch_size=max_micro_batch_size,
                    batch_accumulation_per_replica=1,
                    benchmark_csv_path="benchmark.csv",
                )

                model_size = sum(
                    [p.numel() * p.data.element_size() for p in chain(model.parameters(), model.buffers())]
                )

                log_throughput(
                    bench_config,
                    parallel_context,
                    model_tflops,
                    hardware_tflops,
                    tokens_per_sec,
                    bandwidth=model_size * tokens_per_sec / 1e9,
                )

            # Flush communication
            for _ in range(
                max(
                    len(pipeline_state.microbatches_activations_to_send),
                    len(pipeline_state.microbatches_activations_to_recv),
                )
            ):
                pipeline_state.run_communication()
            assert len(pipeline_state.microbatches_activations_to_send) == 0
            assert len(pipeline_state.microbatches_activations_to_recv) == 0

            # Yield result
            decoder_states = list(decoder_states)
            for state, batch in zip(decoder_states, batches):
                if is_decoder_input_rank:
                    assert all(isinstance(elt, torch.Tensor) for elt in state.generation_ids)
                    batch_generated_ids = torch.cat(state.generation_ids, dim=-1)
                    batch_generated_mask = torch.cat(state.generation_mask, dim=-1)
                else:
                    assert all(isinstance(elt, TensorPointer) for elt in state.generation_ids)
                    batch_generated_ids = TensorPointer(group_rank=decoder_input_rank)
                    batch_generated_mask = TensorPointer(group_rank=decoder_input_rank)

                # Broadcast all data
                batch_generated_ids, batch_generated_mask = broadcast_tensors(
                    [batch_generated_ids, batch_generated_mask],
                    group_src=decoder_input_rank,
                    group=parallel_context.pp_pg,
                )
                batch.input_ids, batch.input_masks = broadcast_tensors(
                    [batch.input_ids, batch.input_masks], group_src=decoder_input_rank, group=parallel_context.pp_pg
                )

                # Flush the store to release memory
                state.store.flush()
                assert len(state.store) == 0

                if dist.get_rank(parallel_context.pp_pg) == decoder_input_rank:
                    assert (
                        batch_generated_ids.shape[0] == batch.input_ids.shape[0]
                    ), f"Batch size needs to match {batch_generated_ids.shape[0]} != {batch.input_ids.shape[0]}"
                    assert (
                        batch_generated_mask.shape[0] == batch.input_ids.shape[0]
                    ), f"Batch size needs to match {batch_generated_mask.shape[0]} != {batch.input_ids.shape[0]}"
                    assert (
                        batch_generated_ids.shape[1] == batch_generated_mask.shape[1]
                    ), f"Sequence length needs to match {batch_generated_ids.shape[1]} != {batch_generated_mask.shape[0]}"

                for i, (generated_ids, generated_mask) in enumerate(zip(batch_generated_ids, batch_generated_mask)):
                    # TODO @thomasw21: We could actually have all ranks return the output, since it's been already broadcasted
                    if dist.get_rank(parallel_context.pp_pg) == decoder_input_rank:
                        input_ids = batch.input_ids[i]
                        input_mask = batch.input_masks[i]
                        yield GenerationOutput(
                            input_ids=input_ids[input_mask],
                            generation_ids=generated_ids[generated_mask],
                        )
                    else:
                        yield GenerationOutput(
                            input_ids=TensorPointer(group_rank=decoder_input_rank),
                            generation_ids=TensorPointer(group_rank=decoder_input_rank),
                        )


@torch.inference_mode()
def decode_tokenized(
    input_ids: torch.Tensor,
    input_mask: torch.Tensor,
    model: LlamaModel,
    parallel_context: ParallelContext,
    generation_config: GenerationArgs,
    max_micro_batch_size: int,
    max_new_tokens: int,
    tokenizer: "PreTrainedTokenizer",
    returns_logits: Optional[bool] = False,
) -> Generator[GenerationOutput, None, None]:
    # [FUNCTION DOC]
    # Behavior: same pipelined generation loop as decode_text but starting from pre-tokenized tensors
    # split by micro_splitter, without benchmarking or prompt replication, feeding only the newest tokens
    # each step with the micro-batch's Store attached (KV-cache path), always transposing logits from
    # sequence-first to batch-first, and constructing TOP_K and TOP_P samplers with top_k, top_p and
    # temperature from generation_config.
    # Inputs: input_ids, input_mask - (batch, seq) tensors; model; parallel_context; generation_config;
    # max_micro_batch_size; max_new_tokens; tokenizer (for position ids); returns_logits - must be False.
    # Outputs: generator of GenerationOutput.
    # Exceptions: NotImplementedError if returns_logits or for an unknown sampler; AssertionError on
    # bookkeeping/shape mismatches.
    # Side effects: P2P and collective communication; CUDA work; RNG; populates and flushes Stores.
    # KNOWN ISSUE: same string-to-SamplerType conversion problem as decode_text.
    # KNOWN ISSUE: position ids are recomputed from only the new tokens each step, so generated tokens get position 0 rather than continuing the sequence.
    # KNOWN ISSUE: same sequence-length assertion message issue as decode_text.
    """We assume the following:
    - Everyone receives ALL the input text. # TODO @thomasw21: technically only specific ranks need to receive input.
    - Only a specific rank will output the generated text_ids as `torch.Tensor`, the others return a `TensorPointer`. # TODO @thomasw21: Maybe all ranks should return the text.
    - We assume that within a model replica, the inputs are already synchronized.
    """
    if returns_logits:
        raise NotImplementedError("return_logits is not implemented yet")

    if generation_config:
        if isinstance(generation_config.sampler, str):
            sampler_type = SamplerType(generation_config.sampler.upper())
        else:
            sampler_type = generation_config.sampler
    else:
        sampler_type = SamplerType.GREEDY

    decoder_input_rank, decoder_logit_rank = get_min_max_rank(module=model)

    # Compute flag
    is_decoder_input_rank = dist.get_rank(parallel_context.pp_pg) == decoder_input_rank
    is_decoder_logit_rank = dist.get_rank(parallel_context.pp_pg) == decoder_logit_rank
    max_nb_microbatches = decoder_logit_rank - decoder_input_rank + 1

    # TODO @thomasw21: Fix this as we shouldn't get P2P like that
    p2p = model.p2p

    # That's annoying but I need this as soon as there's a change communication "cross"
    pipeline_state = PipelineEvalBatchState()
    with attach_pipeline_state_to_model(model=model, pipeline_state=pipeline_state):
        # We query the first `pipeline_size` batches
        for batches in chunks(
            iterable=micro_splitter(
                input_ids,
                input_mask,
                max_micro_batch_size=max_micro_batch_size,
                parallel_context=parallel_context,
                input_rank=decoder_input_rank,
            ),
            chunk_size=max_nb_microbatches,
        ):
            if len(batches) == 0:
                # It means we're out of element
                return

            # Number of micro batches
            number_states_in_buffer = len(batches)
            # Otherwise the pipelining doesn't work
            assert number_states_in_buffer <= max_nb_microbatches
            is_max_nb_microbatches = number_states_in_buffer == max_nb_microbatches

            # Initialize decoder states
            decoder_states: Iterable[GenerationStates] = (
                GenerationStates(
                    new_input_ids=batch.input_ids,
                    new_input_mask=batch.input_masks,
                    store=Store(),
                    generation_ids=[batch.input_ids],
                    generation_mask=[batch.input_masks],
                )
                for batch in batches
            )

            for generation_iter in range(max_new_tokens):
                all_new_decoder_input_ids_and_mask_same_rank: List[
                    Tuple[Union[torch.LongTensor, TensorPointer], Union[torch.BoolTensor, TensorPointer]]
                ] = []
                new_decoder_states: List[GenerationStates] = []
                for state_id, state in enumerate(decoder_states):
                    new_decoder_states.append(state)
                    # Get the new logits
                    with attach_store(model=model, store=state.store):
                        position_ids = get_position_ids(state.new_input_ids, tokenizer)
                        sharded_logits = model(
                            input_ids=state.new_input_ids,
                            position_ids=position_ids,  # [batch_size, seq_len]
                        )
                        if isinstance(sharded_logits, torch.Tensor):
                            sharded_logits = sharded_logits.transpose(0, 1)

                    # Communicate
                    # TODO @thomasw21: Make a diagram to show how this works
                    nb_send: int = 0
                    if is_decoder_input_rank:
                        if is_max_nb_microbatches:
                            if generation_iter == 0:
                                if state_id == number_states_in_buffer - 1:
                                    # `2` is because we receive decoder_ids AND decoder_mask from last rank
                                    nb_send = len(pipeline_state.microbatches_activations_to_send) - 2
                                else:
                                    # Send everything
                                    nb_send = len(pipeline_state.microbatches_activations_to_send)
                            else:
                                # `2` is because we receive decoder_ids AND decoder_mask from last rank
                                nb_send = len(pipeline_state.microbatches_activations_to_send) - 2
                        else:
                            if number_states_in_buffer - 1 == state_id or generation_iter == 0:
                                # Send everything
                                nb_send = len(pipeline_state.microbatches_activations_to_send)
                            else:
                                # `2` is because we receive decoder_ids AND decoder_mask from last rank
                                nb_send = len(pipeline_state.microbatches_activations_to_send) - 2
                    else:
                        if state_id == number_states_in_buffer - 1:
                            if not is_max_nb_microbatches:
                                nb_send = len(pipeline_state.microbatches_activations_to_send)
                    for _ in range(nb_send):
                        pipeline_state.run_communication()

                    if is_decoder_logit_rank:
                        assert isinstance(sharded_logits, torch.Tensor)

                        # run a logit chooser.
                        if sampler_type == SamplerType.GREEDY:
                            sampler = GreedySampler(pg=parallel_context.tp_pg)
                        elif sampler_type == SamplerType.TOP_K:
                            sampler = TopKSampler(
                                pg=parallel_context.tp_pg,
                                k=generation_config.top_k,
                                temperature=generation_config.temperature,
                            )
                        elif sampler_type == SamplerType.TOP_P:
                            sampler = TopPSampler(
                                pg=parallel_context.tp_pg,
                                p=generation_config.top_p,
                                temperature=generation_config.temperature,
                            )
                        elif sampler_type == SamplerType.BASIC:
                            sampler = BasicSampler(pg=parallel_context.tp_pg)
                        else:
                            raise NotImplementedError(f"Sampler type {sampler_type} is not implemented")

                        new_decoder_input_ids = sampler(sharded_logits=sharded_logits[:, -1, :])

                        # TODO @thomasw21: Handle this correctly, ie from some point after <eos> this should only generate masked tokens
                        # TODO @thomasw21: Actually I can probably build this thing on the next device directly. Will save some communication
                        new_decoder_input_mask = torch.ones(
                            size=(new_decoder_input_ids.shape[0], 1),
                            dtype=torch.bool,
                            device=new_decoder_input_ids.device,
                        )

                        # TODO @thomasw21: We need to have stop condition.

                        # broadcast new_tokens to everyone
                        if decoder_input_rank == decoder_logit_rank:
                            # It's the same rank so no need to do anything too fancy
                            all_new_decoder_input_ids_and_mask_same_rank.append(
                                (new_decoder_input_ids, new_decoder_input_mask)
                            )
                        else:
                            pipeline_state.register_send_activation(
                                new_decoder_input_ids, to_rank=decoder_input_rank, p2p=p2p
                            )
                            pipeline_state.register_send_activation(
                                new_decoder_input_mask, to_rank=decoder_input_rank, p2p=p2p
                            )
                            if not is_max_nb_microbatches and state_id == number_states_in_buffer - 1:
                                # Send new_decoder_input_ids AND new_decoder_input_ids
                                pipeline_state.run_communication()
                                pipeline_state.run_communication()

                    else:
                        assert isinstance(sharded_logits, TensorPointer)

                all_new_decoder_input_ids_and_mask: Iterable[
                    Tuple[Union[torch.LongTensor, TensorPointer], Union[torch.BoolTensor, TensorPointer]]
                ]
                if is_decoder_input_rank:
                    # We receive the tensor from other ranks unless `decoder_input_rank` == `decoder_logit_rank` in which case `all_new_decoder_input_ids` is already populated.
                    if decoder_input_rank == decoder_logit_rank:
                        # `all_new_decoder_input_ids_and_mask_same_rank` is already populated. Since `decoder_input_rank` and `decoder_logit_rank` are the same, there's no need to communicate as we can just store the new input_ids in a list.
                        assert len(all_new_decoder_input_ids_and_mask_same_rank) == number_states_in_buffer
                        all_new_decoder_input_ids_and_mask = all_new_decoder_input_ids_and_mask_same_rank
                    else:

                        def generator():
                            # [FUNCTION DOC]
                            # HELPER: identical to decode_text's receiving generator (receive ids and mask per micro-batch from the last stage).
                            # Inputs/Outputs: yields tuples (ids, mask). Exceptions: None. Side effects: P2P communication.
                            for _ in range(number_states_in_buffer):
                                pipeline_state.register_recv_activation(from_rank=decoder_logit_rank, p2p=p2p)
                                pipeline_state.register_recv_activation(from_rank=decoder_logit_rank, p2p=p2p)
                                while len(pipeline_state.activations_buffer) < 2:
                                    pipeline_state.run_communication()
                                new_decoder_input_ids = pipeline_state.activations_buffer.popleft()
                                new_decoder_input_mask = pipeline_state.activations_buffer.popleft()
                                yield new_decoder_input_ids, new_decoder_input_mask

                        all_new_decoder_input_ids_and_mask = iter(generator())
                else:
                    all_new_decoder_input_ids_and_mask = (
                        (TensorPointer(group_rank=decoder_input_rank), TensorPointer(group_rank=decoder_input_rank))
                        for _ in range(number_states_in_buffer)
                    )

                # Create new decoder states
                decoder_states = (
                    GenerationStates(
                        new_input_ids=new_decoder_input_ids_and_mask[0],
                        new_input_mask=new_decoder_input_ids_and_mask[1],
                        store=state.store,
                        generation_ids=state.generation_ids + [new_decoder_input_ids_and_mask[0]],
                        generation_mask=state.generation_mask + [new_decoder_input_ids_and_mask[1]],
                    )
                    for state, new_decoder_input_ids_and_mask in zip(
                        new_decoder_states, all_new_decoder_input_ids_and_mask
                    )
                )

            # Flush communication
            for _ in range(
                max(
                    len(pipeline_state.microbatches_activations_to_send),
                    len(pipeline_state.microbatches_activations_to_recv),
                )
            ):
                pipeline_state.run_communication()
            assert len(pipeline_state.microbatches_activations_to_send) == 0
            assert len(pipeline_state.microbatches_activations_to_recv) == 0

            # Yield result
            decoder_states = list(decoder_states)
            for state, batch in zip(decoder_states, batches):
                if is_decoder_input_rank:
                    assert all(isinstance(elt, torch.Tensor) for elt in state.generation_ids)
                    batch_generated_ids = torch.cat(state.generation_ids, dim=-1)
                    batch_generated_mask = torch.cat(state.generation_mask, dim=-1)
                else:
                    assert all(isinstance(elt, TensorPointer) for elt in state.generation_ids)
                    batch_generated_ids = TensorPointer(group_rank=decoder_input_rank)
                    batch_generated_mask = TensorPointer(group_rank=decoder_input_rank)

                # Broadcast all data
                batch_generated_ids, batch_generated_mask = broadcast_tensors(
                    [batch_generated_ids, batch_generated_mask],
                    group_src=decoder_input_rank,
                    group=parallel_context.pp_pg,
                )
                batch.input_ids, batch.input_masks = broadcast_tensors(
                    [batch.input_ids, batch.input_masks], group_src=decoder_input_rank, group=parallel_context.pp_pg
                )

                # Flush the store to release memory
                state.store.flush()
                assert len(state.store) == 0

                if dist.get_rank(parallel_context.pp_pg) == decoder_input_rank:
                    assert (
                        batch_generated_ids.shape[0] == batch.input_ids.shape[0]
                    ), f"Batch size needs to match {batch_generated_ids.shape[0]} != {batch.input_ids.shape[0]}"
                    assert (
                        batch_generated_mask.shape[0] == batch.input_ids.shape[0]
                    ), f"Batch size needs to match {batch_generated_mask.shape[0]} != {batch.input_ids.shape[0]}"
                    assert (
                        batch_generated_ids.shape[1] == batch_generated_mask.shape[1]
                    ), f"Sequence length needs to match {batch_generated_ids.shape[1]} != {batch_generated_mask.shape[0]}"

                for i, (generated_ids, generated_mask) in enumerate(zip(batch_generated_ids, batch_generated_mask)):
                    # TODO @thomasw21: We could actually have all ranks return the output, since it's been already broadcasted
                    if dist.get_rank(parallel_context.pp_pg) == decoder_input_rank:
                        input_ids = batch.input_ids[i]
                        input_mask = batch.input_masks[i]
                        yield GenerationOutput(
                            input_ids=input_ids[input_mask],
                            generation_ids=generated_ids[generated_mask],
                        )
                    else:
                        yield GenerationOutput(
                            input_ids=TensorPointer(group_rank=decoder_input_rank),
                            generation_ids=TensorPointer(group_rank=decoder_input_rank),
                        )


# Distributed utilities
def broadcast_tensors(
    tensors: List[Union[torch.Tensor, TensorPointer]], group_src: int, group: Optional[ProcessGroup] = None
) -> List[torch.Tensor]:
    # [FUNCTION DOC]
    # HELPER: broadcasts tensors from one rank of group to all others, which hold TensorPointers.
    # Behavior: for each tensor, the source sends its metadata (dtype, requires_grad, shape, storage size,
    # stride, contiguity, offset) via broadcast_object_list; receivers allocate empty CUDA storage from
    # that metadata while the source takes a contiguous view; the data is broadcast and every rank returns
    # the tensor re-strided to the original shape and stride.
    # Inputs: tensors - list of tensors (source) or TensorPointers (others); group_src - source rank within
    # group; group - ProcessGroup (default None meaning world).
    # Outputs: list of tensors, one per input, on every rank.
    # Exceptions: AssertionError if the source lacks a tensor or another rank lacks a TensorPointer.
    # Side effects: collective communication (object and tensor broadcasts); CUDA allocations.
    result = []
    for tensor in tensors:
        if dist.get_rank(group) == group_src:
            assert isinstance(tensor, torch.Tensor)
            meta = [
                [
                    tensor.dtype,
                    tensor.requires_grad,
                    tensor.shape,
                    get_untyped_storage(tensor).size(),
                    tensor.stride(),
                    tensor.is_contiguous(),
                    tensor.storage_offset(),
                ]
            ]
        else:
            assert isinstance(tensor, TensorPointer)
            meta = [None]
        dist.broadcast_object_list(meta, src=get_global_rank(group_rank=group_src, group=group), group=group)
        dtype, requires_grad, shape, untyped_storage_size, stride, is_contiguous, storage_offset = meta[0]
        meta = P2PTensorMetaData(
            dtype=dtype,
            requires_grad=requires_grad,
            shape=shape,
            untyped_storage_size=untyped_storage_size,
            stride=stride,
            is_contiguous=is_contiguous,
            storage_offset=storage_offset,
        )
        if dist.get_rank(group) != group_src:
            tensor = meta.create_empty_storage(device=torch.device("cuda"))
        else:
            tensor = view_as_contiguous(tensor)
        dist.broadcast(tensor, src=get_global_rank(group_rank=group_src, group=group), group=group)
        # Set shape and stride
        tensor = tensor.as_strided(size=tuple(meta.shape), stride=tuple(meta.stride))
        result.append(tensor)
    return result
