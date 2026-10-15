# [MODULE DOC]
# Purpose: Inference/demo script that loads a nanotron checkpoint and greedily generates text (launch with
# torchrun; CUDA_DEVICE_MAX_CONNECTIONS=1 recommended).
# Main contents: get_args (CLI) and main (build parallel context and model, load weights, decode prompts).
# Import-time side effects: tries to import transformers AutoTokenizer, setting it to None when missing (then a
# tokenized dummy input path is used instead). Creates a module logger.
"""
Nanotron Inference Script

Usage:
```
export CUDA_DEVICE_MAX_CONNECTIONS=1 # important for some distributed operations
torchrun --nproc_per_node=1 run_generate.py --ckpt-path checkpoints/10
```
"""

import argparse
import os
from pathlib import Path

import torch
from nanotron import distributed as dist
from nanotron import logging
from nanotron.config import (
    GenerationArgs,
    LoggingArgs,
    ParallelismArgs,
    get_config_from_file,
)
from nanotron.generation.decode import (
    GenerationInput,
    TokenizerConfig,
    decode_text,
    decode_tokenized,
)
from nanotron.logging import log_rank, set_ranks_logging_level
from nanotron.models import build_model
from nanotron.parallel import ParallelContext
from nanotron.parallel.parameters import sanity_check
from nanotron.parallel.pipeline_parallel.engine import (
    OneForwardOneBackwardPipelineEngine,
)
from nanotron.parallel.pipeline_parallel.tensor_pointer import TensorPointer
from nanotron.parallel.tensor_parallel.enum import TensorParallelLinearMode
from nanotron.random import (
    RandomStates,
    get_current_random_state,
    get_synced_random_state,
    set_random_seed,
)
from nanotron.serialize import load_weights
from nanotron.trainer import CONFIG_TO_MODEL_CLASS, mark_tied_parameters

try:
    from transformers import AutoTokenizer
except ImportError:
    AutoTokenizer = None

# import lovely_tensors as lt

# lt.monkey_patch()

logger = logging.get_logger(__name__)


def get_args():
    # [FUNCTION DOC]
    # Behavior: parses CLI: required ckpt-path (Path), dp (default 1), pp and tp (default 0, meaning take from the
    # checkpoint config), max-new-tokens (128), use-cache flag.
    # Inputs: None (reads sys.argv). Outputs: argparse Namespace. Exceptions: SystemExit on bad args. Side effects: None.
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt-path", type=Path, required=True, help="Checkpoint path")
    parser.add_argument("--dp", type=int, default=1)
    parser.add_argument("--pp", type=int, default=0)
    parser.add_argument("--tp", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=128, help="Maximum number of new tokens to generate")
    parser.add_argument("--use-cache", action="store_true", help="Use KV cache to speed up generation")
    return parser.parse_args()


def main():
    # [FUNCTION DOC]
    # Behavior:
    # 1. Parse args, assert the checkpoint dir exists, load config.yaml from it, take model config and tokenizer path.
    # 2. Build ParallelismArgs (CLI value or config value for dp/pp/tp, 1F1B engine, ALL_REDUCE TP, sync linears) and a
    #    ParallelContext (initializes all process groups); set per-rank log levels to info.
    # 3. Seed RNG with 42; reject model configs not in CONFIG_TO_MODEL_CLASS; build TP-synced random states (all
    #    reduce mode) and build the model in bfloat16 with build_model; mark tied parameters; sanity_check.
    # 4. Load weights from the checkpoint folder and switch to eval.
    # 5. With transformers: load the tokenizer, choose a pad token (eos, else model config pad id, else model eos id,
    #    else add a [PAD] special token), left padding/truncation, run decode_text on two hardcoded prompts with
    #    micro batch size 2 and greedy sampling (benchmark mode when env USE_BENCH is 1), and on ranks holding outputs
    #    log the decoded input and generated continuation from rank 0.
    # 6. Without transformers: decode_tokenized from a single zero token on CUDA for 12 new tokens and log raw ids.
    # 7. Global barrier.
    # Inputs: None (CLI). Outputs: None.
    # Exceptions: AssertionError if the checkpoint path is missing or output types mismatch; ValueError for
    # unsupported model configs.
    # Side effects: initializes torch.distributed process groups, sets RNG state, reads env USE_BENCH, loads files,
    # CUDA allocations, logging, collectives (barrier and those inside generation).
    # KNOWN ISSUE: adding a [PAD] special token grows the tokenizer vocabulary without resizing model embeddings.
    args = get_args()

    assert args.ckpt_path.exists(), f"Checkpoint path {args.ckpt_path} does not exist"

    config = get_config_from_file((args.ckpt_path / "config.yaml").as_posix())
    model_config = config.model.model_config
    tokenizer_path = config.tokenizer.tokenizer_name_or_path

    parallel_config = ParallelismArgs(
        dp=args.dp or config.parallelism.dp,
        pp=args.pp or config.parallelism.pp,
        tp=args.tp or config.parallelism.tp,
        pp_engine=OneForwardOneBackwardPipelineEngine(),
        tp_mode=TensorParallelLinearMode.ALL_REDUCE,
        tp_linear_async_communication=False,
    )

    # Initialise all process groups
    parallel_context = ParallelContext(
        data_parallel_size=parallel_config.dp,
        pipeline_parallel_size=parallel_config.pp,
        tensor_parallel_size=parallel_config.tp,
    )

    # Set log levels
    logging_config = LoggingArgs(
        log_level="info",
        log_level_replica="info",
    )

    # Set log levels
    set_ranks_logging_level(parallel_context=parallel_context, logging_config=logging_config)

    log_rank(f"model_config: {model_config}", logger=logger, level=logging.INFO, rank=0)
    log_rank(f"tokenizer_path: {tokenizer_path}", logger=logger, level=logging.INFO, rank=0)

    dtype = torch.bfloat16

    # Set random states
    set_random_seed(42)

    model_config_cls = model_config.__class__.__name__
    if model_config_cls not in CONFIG_TO_MODEL_CLASS:
        raise ValueError(
            f"Unsupported model config {model_config_cls}. Only {CONFIG_TO_MODEL_CLASS.keys()} are supported"
        )

    # Get synchronized random states
    if parallel_config.tp_mode is TensorParallelLinearMode.ALL_REDUCE:
        random_states = RandomStates(
            {"tp_synced": get_synced_random_state(random_state=get_current_random_state(), pg=parallel_context.tp_pg)}
        )
    else:
        # We don't need to sync across TP when using sequence parallel (REDUCE_SCATTER)
        random_states = RandomStates({})

    model = build_model(
        model_builder=lambda: CONFIG_TO_MODEL_CLASS[model_config_cls](
            config=model_config,
            parallel_context=parallel_context,
            parallel_config=parallel_config,
            random_states=random_states,
        ),
        dtype=dtype,
        parallel_context=parallel_context,
    )

    # Mark some parameters as tied
    # TODO @nouamane: this is only needed for training, can we just mark params as NanotronParameter instead?
    mark_tied_parameters(model=model, parallel_context=parallel_context, parallel_config=parallel_config)

    # Sanity check model
    sanity_check(root_module=model)

    # Load checkpoint
    checkpoint_path = args.ckpt_path
    log_rank(
        f"Loading checkpoint from {checkpoint_path}:",
        logger=logger,
        level=logging.INFO,
        rank=0,
    )
    load_weights(model=model, parallel_context=parallel_context, root_folder=checkpoint_path)

    model.eval()
    if AutoTokenizer is not None:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        # tokenizer.pad_token_id = tokenizer.eos_token_id
        if tokenizer.pad_token_id is None:
            if tokenizer.eos_token_id is not None:
                tokenizer.pad_token_id = tokenizer.eos_token_id
            elif getattr(model.config, "pad_token_id", None) is not None:
                tokenizer.pad_token_id = int(model.config.pad_token_id)
            elif getattr(model.config, "eos_token_id", None) is not None:
                tokenizer.pad_token_id = int(model.config.eos_token_id)
            else:
                tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        tokenizer.padding_side = "left"
        tokenizer.truncation_side = "left"  # TODO @nouamane: do we want this?
        dummy_inputs = [
            # "The future of AI is",
            "Passage: Daniel went back to the garden. Mary travelled to the kitchen. Sandra journeyed to the kitchen. Sandra went to the hallway. John went to the bedroom. Mary went back to the garden. Where is Mary?\nAnswer:",
            "def fib(n)",
            # 'Here is an extract from a webpage: "Have you ever experienced heel pain after a heavy physical activity, or even right after a long period of standing? If you regard this as something usual and normal, then think again. Miscalled as heel pain, plantar fasciitis causes these frequent mild pains experienced in the soles of the feet. It is the inflammation and enlargement the plantar fascia tissue that is located in the heels of the feet, stretching to the base of the toes. This tissue is responsible for absorbing shock in the feet and for supporting the arches. It also plays a vital role in foot movements during walking and standing. Many factors such as excessive walking, standing, and running trigger heel pain and plantar fasciitis. A sudden increase in intensity of activities, increase in weight, and abrupt change of footwear also cause the swelling of the ligament. Non-supportive footwear lacking arch cushions and improper and worn out running or training can also lead to the problem. It is also most evident among those". Write an extensive and detailed course unit suitable for a textbook targeted at college students, related to the given extract, within the context of "Medicine". Do not just list concepts, but develop each one in detail before moving to the next, as we prioritize depth of understanding and comprehensive exploration of the subject matter over breadth. Focus on: - Rigor: Ensure in-depth coverage of the concepts/sections. - Engagement: Write with an academic, professional and engaging tone that captivates interest. - Application: Incorporate specific, practical examples, such as proofs in calculus or critical dates and figures in history. Do not include a title or an introduction, simply write the content without headlines and introductory phrases. Do not use images.',
            # "Advancements in technology will lead to",
            # "Tomorrow's world is shaped by",
        ]

        outputs = decode_text(
            input_iter=(GenerationInput(text=text) for text in dummy_inputs),
            tokenizer=tokenizer,
            model=model.model,
            parallel_context=parallel_context,
            max_new_tokens=args.max_new_tokens,
            max_micro_batch_size=2,
            generation_config=GenerationArgs(sampler="greedy", use_cache=args.use_cache),
            tokenizer_config=TokenizerConfig(max_input_length=None),
            is_bench=os.environ.get("USE_BENCH", "0") == "1",
        )
        for output in outputs:
            input_ids = output.input_ids
            generated_ids = output.generation_ids
            if isinstance(input_ids, TensorPointer):
                assert isinstance(generated_ids, TensorPointer)
                continue
            assert isinstance(generated_ids, torch.Tensor)

            log_rank(
                f"input: {tokenizer.decode(input_ids, clean_up_tokenization_spaces=False)[:1000]}",
                logger=logger,
                level=logging.INFO,
                rank=0,
            )

            log_rank(
                f"generation: {tokenizer.decode(generated_ids[len(input_ids) :], clean_up_tokenization_spaces=False)}",
                logger=logger,
                level=logging.INFO,
                rank=0,
            )

            log_rank(
                "--------------------------------------------------",
                logger=logger,
                level=logging.INFO,
                rank=0,
            )
    else:
        outputs = decode_tokenized(
            input_ids=torch.zeros(1, 1).to(dtype=torch.int64, device="cuda"),
            input_mask=torch.ones(1, 1).to(dtype=torch.bool, device="cuda"),
            model=model.model,
            parallel_context=parallel_context,
            generation_config=GenerationArgs(sampler="greedy", use_cache=True),
            max_micro_batch_size=1,
            max_new_tokens=12,
            returns_logits=False,
        )
        for output in outputs:
            input_ids = output.input_ids
            generated_ids = output.generation_ids
            if isinstance(input_ids, TensorPointer):
                assert isinstance(generated_ids, TensorPointer)
                continue
            assert isinstance(generated_ids, torch.Tensor)
            log_rank(
                f"generation: {generated_ids[len(input_ids) :]}",
                logger=logger,
                level=logging.INFO,
                rank=0,
            )

            log_rank(
                "--------------------------------------------------",
                logger=logger,
                level=logging.INFO,
                rank=0,
            )

    dist.barrier()


if __name__ == "__main__":
    main()
