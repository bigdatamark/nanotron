# [MODULE DOC]
# Purpose: pytest suite verifying the Llama/Qwen2 <-> HuggingFace checkpoint conversion scripts in
# examples/llama (convert_hf_to_nanotron, convert_nanotron_to_hf, convert_weights).
# Contents: model factories (create_nanotron_model, create_huggingface_model), fixtures (fix_seed,
# input_ids), private test bodies (prefixed with underscore) executed under init_distributed, and public
# test functions that launch them with given tp/dp/pp sizes. Logits of both models are compared with
# tolerance ATOL (0.03) on random token batches of shape BATCH_SIZE (6) by SEQUENCE_LENGTH (5).
# Import-time side effects: calls set_system_path (mutates sys.path), seeds RNG via set_random_seed(0),
# sets cudnn deterministic mode, and builds the global CONFIG (Llama or Qwen2 nanotron config, 4 layers,
# hidden 512, vocab 4096) depending on the NT_MODEL_INSTANCE constant (currently Qwen2ForTraining).
# Running as a script executes test_tensor_parallel_conversion only. Requires CUDA.
# KNOWN ISSUE: CONFIG is a Qwen2 config yet HuggingFace side always uses LlamaForCausalLM; conversion
# calls hardcode config_cls NanotronQwen2Config even if NT_MODEL_INSTANCE is switched to Llama.
# ruff: noqa: E402
import dataclasses
import json
from pathlib import Path
from typing import Optional

import pytest
import torch
from transformers import AutoModelForCausalLM, LlamaForCausalLM
from utils import set_system_path

set_system_path()

import nanotron
from nanotron import distributed as dist
from nanotron.config import LlamaConfig as NanotronLlamaConfig
from nanotron.config import NanotronConfigs
from nanotron.config import Qwen2Config as NanotronQwen2Config
from nanotron.models.base import init_on_device_and_dtype
from nanotron.models.llama import LlamaForTraining
from nanotron.models.qwen import Qwen2ForTraining
from nanotron.parallel import ParallelContext
from nanotron.random import set_random_seed
from nanotron.trainer import mark_tied_parameters

from examples.llama.convert_hf_to_nanotron import convert_checkpoint_and_save as convert_hf_to_nt_and_save
from examples.llama.convert_hf_to_nanotron import convert_hf_to_nt, get_nanotron_config
from examples.llama.convert_nanotron_to_hf import convert_checkpoint_and_save as convert_nt_to_hf_and_save
from examples.llama.convert_nanotron_to_hf import convert_nt_to_hf, get_hf_config
from examples.llama.convert_weights import load_nanotron_model, make_parallel_config
from tests.helpers.context import TestContext
from tests.helpers.utils import init_distributed

set_random_seed(0)
torch.backends.cudnn.deterministic = True

BATCH_SIZE = 6
SEQUENCE_LENGTH = 5
ATOL = 0.03

NT_MODEL_INSTANCE = Qwen2ForTraining

if NT_MODEL_INSTANCE == LlamaForTraining:
    CONFIG = NanotronLlamaConfig(
        **{
            "bos_token_id": 1,
            "eos_token_id": 2,
            "hidden_act": "silu",
            "hidden_size": 512,
            "initializer_range": 0.02,
            "intermediate_size": 1024,
            "is_llama_config": True,
            "max_position_embeddings": 128,
            "num_attention_heads": 8,
            "num_hidden_layers": 4,
            "num_key_value_heads": 4,
            "pad_token_id": None,
            "pretraining_tp": 1,
            "rms_norm_eps": 1e-06,
            "rope_scaling": None,
            "tie_word_embeddings": False,
            "use_cache": True,
            "vocab_size": 4096,
            "attention_bias": False,
        }
    )
elif NT_MODEL_INSTANCE == Qwen2ForTraining:
    CONFIG = NanotronQwen2Config(
        **{
            "bos_token_id": 1,
            "eos_token_id": 2,
            "hidden_act": "silu",
            "hidden_size": 512,
            "initializer_range": 0.02,
            "intermediate_size": 1024,
            "max_position_embeddings": 128,
            "num_attention_heads": 8,
            "num_hidden_layers": 4,
            "num_key_value_heads": 8,
            "pad_token_id": None,
            "pretraining_tp": 1,
            "rms_norm_eps": 1e-06,
            "rope_scaling": None,
            "tie_word_embeddings": False,
            "use_cache": True,
            "vocab_size": 4096,
            "_attn_implementation": "sdpa",
            "attention_bias": False,
            "rope_interleaved": False,
        }
    )


def create_nanotron_model(parallel_context: ParallelContext, config: NanotronConfigs = CONFIG) -> NT_MODEL_INSTANCE:
    # [FUNCTION DOC]
    # HELPER: builds the nanotron model under test.
    # Behavior: creates a parallel config matching the context's tp/dp/pp sizes (make_parallel_config),
    # builds NT_MODEL_INSTANCE via nanotron.models.build_model in bfloat16 on CUDA with no random states,
    # then marks tied parameters.
    # Inputs: parallel_context (ParallelContext); config (nanotron model config, default global CONFIG).
    # Outputs: the built nanotron training model (instance of NT_MODEL_INSTANCE).
    # Exceptions: None explicitly raised.
    # Side effects: allocates CUDA memory; uses process groups from parallel_context.
    parallel_config = make_parallel_config(
        tp=parallel_context.tensor_parallel_size,
        dp=parallel_context.data_parallel_size,
        pp=parallel_context.pipeline_parallel_size,
    )
    nanotron_model = nanotron.models.build_model(
        model_builder=lambda: NT_MODEL_INSTANCE(
            config=config,
            parallel_context=parallel_context,
            parallel_config=parallel_config,
            random_states=None,
        ),
        parallel_context=parallel_context,
        dtype=torch.bfloat16,
        device=torch.device("cuda"),
    )
    mark_tied_parameters(model=nanotron_model, parallel_context=parallel_context)
    return nanotron_model


def create_huggingface_model(model_name: Optional[str] = None) -> LlamaForCausalLM:
    # [FUNCTION DOC]
    # HELPER: builds the HuggingFace reference model.
    # Behavior: if model_name is None, instantiates a randomly initialised LlamaForCausalLM from the HF
    # config derived from CONFIG (get_hf_config); otherwise loads AutoModelForCausalLM.from_pretrained(model_name).
    # Both under init_on_device_and_dtype (CUDA, bfloat16).
    # Inputs: model_name (optional string hub id or path, default None).
    # Outputs: a HuggingFace causal LM model.
    # Exceptions: None explicitly raised (HF loading errors propagate).
    # Side effects: CUDA allocation; may download weights from the network.
    # KNOWN ISSUE: return annotation says LlamaForCausalLM but the pretrained branch may return another class.
    if model_name is None:
        with init_on_device_and_dtype(torch.device("cuda"), torch.bfloat16):
            model_hf = LlamaForCausalLM._from_config(get_hf_config(CONFIG))
    else:
        with init_on_device_and_dtype(torch.device("cuda"), torch.bfloat16):
            model_hf = AutoModelForCausalLM.from_pretrained(model_name)
    return model_hf


@pytest.fixture(autouse=True, scope="module")
def fix_seed():
    # [FUNCTION DOC]
    # HELPER: autouse module-scoped pytest fixture.
    # Behavior: seeds torch RNG with 0 once per module, then yields. Inputs: none. Outputs: generator yielding None.
    # Exceptions: None. Side effects: global torch RNG state.
    torch.manual_seed(0)
    yield


@pytest.fixture
def input_ids() -> torch.Tensor:
    # [FUNCTION DOC]
    # HELPER: pytest fixture producing a random token batch.
    # Outputs: int64 CUDA tensor of shape (BATCH_SIZE, SEQUENCE_LENGTH) with values in [0, vocab_size).
    # Inputs: none. Exceptions: None. Side effects: consumes torch RNG; CUDA allocation.
    return torch.randint(0, CONFIG.vocab_size, size=(BATCH_SIZE, SEQUENCE_LENGTH), device="cuda")


def _test_nt_to_hf(parallel_context: ParallelContext, input_ids: torch.Tensor):
    # [FUNCTION DOC]
    # HELPER: body of test_nt_to_hf. Runs inside init_distributed so a ParallelContext is supplied by the test harness.
    # Behavior: builds a nanotron model and a random HF model, copies nanotron weights into HF with
    # convert_nt_to_hf, runs both on input_ids (Qwen2 path passes int32 position ids 0..seq-1 repeated per
    # batch and reshapes output to batch, seq, vocab; Llama path passes an all-ones mask and permutes from
    # seq-first), and asserts logits close within ATOL.
    # Inputs: parallel_context; input_ids (batch, seq tensor). Outputs: None.
    # Exceptions: AssertionError if logits differ. Side effects: CUDA allocation.
    model_nt = create_nanotron_model(parallel_context)
    model_hf = create_huggingface_model()
    convert_nt_to_hf(model_nt, model_hf, CONFIG)
    input_mask = torch.ones_like(input_ids)
    if NT_MODEL_INSTANCE == Qwen2ForTraining:
        position_ids = (
            torch.arange(0, input_ids.shape[1], device=input_ids.device, dtype=torch.int32)
            .unsqueeze(0)
            .repeat(BATCH_SIZE, 1)
        )
        logits_nt = model_nt.model(input_ids, position_ids).view(BATCH_SIZE, SEQUENCE_LENGTH, -1)
    else:
        logits_nt = model_nt.model(input_ids, input_mask).permute(1, 0, 2)
    logits_hf = model_hf(input_ids).logits
    torch.testing.assert_allclose(logits_nt, logits_hf.to(logits_nt.dtype), atol=ATOL, rtol=ATOL)


def test_nt_to_hf(input_ids: torch.Tensor):
    # [FUNCTION DOC]
    # Test: launches _test_nt_to_hf with tp=dp=pp=1. Inputs: input_ids fixture. Outputs: None.
    # Exceptions: propagates assertion failures. Side effects: spawns distributed worker(s).
    init_distributed(tp=1, dp=1, pp=1)(_test_nt_to_hf)(input_ids=input_ids)


def _test_nt_to_hf_with_files(parallel_context: ParallelContext, input_ids: torch.Tensor, test_context: TestContext):
    # [FUNCTION DOC]
    # HELPER: body of test_nt_to_hf_with_files. Runs inside init_distributed so a ParallelContext is supplied by the test harness.
    # Behavior: builds a nanotron model, saves its weights and model_config.json (dataclass of CONFIG as
    # JSON) into a temp dir "nanotron", computes nanotron logits (same Qwen2/Llama branches as
    # _test_nt_to_hf), deletes the model, runs the file-based convert_nt_to_hf_and_save into temp "hf",
    # loads the result as LlamaForCausalLM on CUDA and asserts logits match within ATOL.
    # Inputs: parallel_context; input_ids; test_context (TestContext providing an auto-removed temp dir).
    # Outputs: None. Exceptions: AssertionError on mismatch.
    # Side effects: writes and reads checkpoint files on disk; CUDA allocation.
    # Create and save nanotron model.
    model_nt = create_nanotron_model(parallel_context)
    root = test_context.get_auto_remove_tmp_dir()
    nt_path = root / "nanotron"
    hf_path = root / "hf"
    nanotron.serialize.save_weights(model=model_nt, parallel_context=parallel_context, root_folder=nt_path)
    with open(nt_path / "model_config.json", "w+") as f:
        json.dump(dataclasses.asdict(CONFIG), f)
    input_mask = torch.ones_like(input_ids)
    if NT_MODEL_INSTANCE == Qwen2ForTraining:
        position_ids = (
            torch.arange(0, input_ids.shape[1], device=input_ids.device, dtype=torch.int32)
            .unsqueeze(0)
            .repeat(BATCH_SIZE, 1)
        )
        logits_nt = model_nt.model(input_ids, position_ids).view(BATCH_SIZE, SEQUENCE_LENGTH, -1)
    else:
        logits_nt = model_nt.model(input_ids, input_mask).permute(1, 0, 2)
    del model_nt
    # Perform conversion.
    convert_nt_to_hf_and_save(nt_path, hf_path, config_cls=NanotronQwen2Config)
    # Load huggingface and get logits.
    model_hf = LlamaForCausalLM.from_pretrained(hf_path).cuda()
    logits_hf = model_hf(input_ids).logits
    torch.testing.assert_allclose(logits_nt, logits_hf.to(logits_nt.dtype), atol=ATOL, rtol=ATOL)


def test_nt_to_hf_with_files(input_ids: torch.Tensor):
    # [FUNCTION DOC]
    # Test: launches _test_nt_to_hf_with_files with tp=dp=pp=1 and a new TestContext.
    # Inputs: input_ids fixture. Outputs: None. Exceptions: propagated failures. Side effects: temp files, worker processes.
    init_distributed(tp=1, dp=1, pp=1)(_test_nt_to_hf_with_files)(input_ids=input_ids, test_context=TestContext())


def _test_hf_to_nt(parallel_context: ParallelContext, input_ids: torch.Tensor, model_name: Optional[str]):
    # [FUNCTION DOC]
    # HELPER: body of test_hf_to_nt. Runs inside init_distributed so a ParallelContext is supplied by the test harness.
    # Behavior: builds/loads the HF model (random if model_name is None), builds a nanotron model from the
    # nanotron config derived from the HF config (get_nanotron_config), copies weights with convert_hf_to_nt,
    # computes both logits (same branches as _test_nt_to_hf) and asserts closeness within ATOL.
    # Inputs: parallel_context; input_ids; model_name (optional HF hub id).
    # Outputs: None. Exceptions: AssertionError on mismatch. Side effects: CUDA, possible network download.
    model_hf = create_huggingface_model(model_name)
    model_nt = create_nanotron_model(parallel_context, get_nanotron_config(model_hf.config))
    convert_hf_to_nt(model_hf, model_nt, model_hf.config)
    input_mask = torch.ones_like(input_ids)
    if NT_MODEL_INSTANCE == Qwen2ForTraining:
        position_ids = (
            torch.arange(0, input_ids.shape[1], device=input_ids.device, dtype=torch.int32)
            .unsqueeze(0)
            .repeat(BATCH_SIZE, 1)
        )
        logits_nt = model_nt.model(input_ids, position_ids).view(BATCH_SIZE, SEQUENCE_LENGTH, -1)
    else:
        logits_nt = model_nt.model(input_ids, input_mask).permute(1, 0, 2)
    logits_hf = model_hf(input_ids).logits
    torch.testing.assert_allclose(logits_hf.to(logits_nt.dtype), logits_nt, atol=ATOL, rtol=ATOL)


@pytest.mark.parametrize("model_name", [None, "HuggingFaceTB/SmolLM2-135M"])
def test_hf_to_nt(input_ids: torch.Tensor, model_name: Optional[str]):
    # [FUNCTION DOC]
    # Test: parametrized over model_name None and "HuggingFaceTB/SmolLM2-135M"; launches _test_hf_to_nt with tp=dp=pp=1.
    # Inputs: input_ids fixture; model_name. Outputs: None. Exceptions: propagated failures.
    # Side effects: worker processes; network download for the pretrained case.
    # KNOWN ISSUE: input_ids is drawn from CONFIG.vocab_size (4096), which only coincidentally fits the pretrained model's vocabulary.
    init_distributed(tp=1, dp=1, pp=1)(_test_hf_to_nt)(input_ids=input_ids, model_name=model_name)


def _test_hf_to_nt_with_files(parallel_context: ParallelContext, input_ids: torch.Tensor, test_context: TestContext):
    # [FUNCTION DOC]
    # HELPER: body of test_hf_to_nt_with_files. Runs inside init_distributed so a ParallelContext is supplied by the test harness.
    # Behavior: creates a random HF model, saves it with save_pretrained to temp "hf", records its logits,
    # deletes it, converts the directory with convert_hf_to_nt_and_save into temp "nanotron", loads it via
    # load_nanotron_model (Qwen2 config class), computes nanotron logits and asserts closeness within ATOL.
    # Inputs: parallel_context (unused directly); input_ids; test_context. Outputs: None.
    # Exceptions: AssertionError on mismatch. Side effects: disk writes/reads, CUDA allocation.
    # Create and save hf model.
    model_hf = create_huggingface_model()
    root = test_context.get_auto_remove_tmp_dir()
    nt_path = root / "nanotron"
    hf_path = root / "hf"
    model_hf.save_pretrained(hf_path)
    logits_hf = model_hf(input_ids).logits
    del model_hf
    # Perform conversion.
    convert_hf_to_nt_and_save(hf_path, nt_path)
    # Load nanotron and get logits.
    input_mask = torch.ones_like(input_ids)
    model_nt = load_nanotron_model(checkpoint_path=nt_path, config_cls=NanotronQwen2Config)
    if NT_MODEL_INSTANCE == Qwen2ForTraining:
        position_ids = (
            torch.arange(0, input_ids.shape[1], device=input_ids.device, dtype=torch.int32)
            .unsqueeze(0)
            .repeat(BATCH_SIZE, 1)
        )
        logits_nt = model_nt.model(input_ids, position_ids).view(BATCH_SIZE, SEQUENCE_LENGTH, -1)
    else:
        logits_nt = model_nt.model(input_ids, input_mask).permute(1, 0, 2)
    torch.testing.assert_allclose(logits_nt, logits_hf.to(logits_nt.dtype), atol=ATOL, rtol=ATOL)


def test_hf_to_nt_with_files(input_ids: torch.Tensor):
    # [FUNCTION DOC]
    # Test: launches _test_hf_to_nt_with_files with tp=dp=pp=1 and a new TestContext.
    # Inputs: input_ids fixture. Outputs: None. Exceptions: propagated failures. Side effects: temp files, workers.
    init_distributed(tp=1, dp=1, pp=1)(_test_hf_to_nt_with_files)(input_ids=input_ids, test_context=TestContext())


def _test_composed_conversion(parallel_context: ParallelContext):
    # [FUNCTION DOC]
    # HELPER: body of test_composed_conversion; checks conversions are exact inverses. Runs inside init_distributed so a ParallelContext is supplied by the test harness.
    # Behavior: snapshots (clones) a random HF state dict; converts HF to a new nanotron model and snapshots
    # its state dict; converts that nanotron model into a fresh HF model and asserts same key set and
    # bitwise-equal tensors versus the original HF snapshot; then converts back into a fresh nanotron model
    # and asserts the same versus the nanotron snapshot.
    # Inputs: parallel_context. Outputs: None. Exceptions: AssertionError on any key or value mismatch.
    # Side effects: CUDA allocation.
    # Get HF statedict.
    model_hf = create_huggingface_model()
    hf_sd = {key: val.clone() for key, val in model_hf.state_dict().items()}
    # Convert once to nanotron, save its statedict.
    model_nt = create_nanotron_model(parallel_context)
    convert_hf_to_nt(model_hf, model_nt, CONFIG)
    nt_sd = {key: val.clone() for key, val in model_nt.state_dict().items()}
    # Convert back to HF, compare statedicts.
    del model_hf
    model_hf = create_huggingface_model()
    convert_nt_to_hf(model_nt, model_hf, CONFIG)
    hf_sd_new = model_hf.state_dict()
    assert set(hf_sd_new) == set(hf_sd)
    assert all(torch.all(hf_sd[key] == hf_sd_new[key]) for key in hf_sd_new)
    # Convert to nanotron one more time, compare statedicts.
    del model_nt
    model_nt = create_nanotron_model(parallel_context)
    convert_hf_to_nt(model_hf, model_nt, CONFIG)
    nt_sd_new = model_nt.state_dict()
    assert set(nt_sd_new) == set(nt_sd)
    assert all(torch.all(nt_sd[key] == nt_sd_new[key]) for key in nt_sd_new)


def test_composed_conversion():
    # [FUNCTION DOC]
    # Test: launches _test_composed_conversion with tp=dp=pp=1. Inputs: none. Outputs: None.
    # Exceptions: propagated failures. Side effects: worker processes.
    init_distributed(tp=1, dp=1, pp=1)(_test_composed_conversion)()


def _save_parallel_nanotron(parallel_context: ParallelContext, input_ids: torch.Tensor, nt_path: Path):
    # [FUNCTION DOC]
    # HELPER: first stage of test_tensor_parallel_conversion; runs on every rank of a tp=2 launch.
    # Behavior: builds a tensor-parallel nanotron model, saves weights and model_config.json into nt_path,
    # moves input_ids to the current CUDA device, computes local logits (vocab-sharded), checks the last
    # dimension equals vocab_size divided by tp size, all-gathers the shards over the tp group into a tensor of
    # shape (tp, batch, seq, shard), rearranges to (batch, seq, full vocab), asserts the local slice equals this
    # rank's shard, and on tp rank 0 saves the full logits on CPU to nt_path/logits_nt.pt.
    # Inputs: parallel_context; input_ids (batch, seq); nt_path (Path of checkpoint dir).
    # Outputs: None. Exceptions: AssertionError on shape or shard mismatch.
    # Side effects: disk writes (by all ranks for weights, rank 0 for logits), all_gather_into_tensor over tp_pg, CUDA.
    # Create and save a parallel model.
    model_nt = create_nanotron_model(parallel_context)
    nanotron.serialize.save_weights(model=model_nt, parallel_context=parallel_context, root_folder=nt_path)
    with open(nt_path / "model_config.json", "w+") as f:
        json.dump(dataclasses.asdict(CONFIG), f)

    # Get parallel predictions.
    input_ids = input_ids.cuda()  # Move them to the current device index.
    input_mask = torch.ones_like(input_ids)
    if NT_MODEL_INSTANCE == Qwen2ForTraining:
        position_ids = (
            torch.arange(0, input_ids.shape[1], device=input_ids.device, dtype=torch.int32)
            .unsqueeze(0)
            .repeat(BATCH_SIZE, 1)
        )
        logits_nt = model_nt.model(input_ids, position_ids).view(BATCH_SIZE, SEQUENCE_LENGTH, -1)
    else:
        logits_nt = model_nt.model(input_ids, input_mask).permute(1, 0, 2)

    tp_pg = parallel_context.tp_pg
    tp_rank = dist.get_rank(tp_pg)
    sharded_vocab_size = CONFIG.vocab_size // tp_pg.size()
    logits_nt = logits_nt.view(BATCH_SIZE, SEQUENCE_LENGTH, -1).contiguous()
    assert logits_nt.shape[-1] == sharded_vocab_size
    logits_nt_full = torch.empty(
        tp_pg.size(), BATCH_SIZE, SEQUENCE_LENGTH, sharded_vocab_size, device="cuda", dtype=logits_nt.dtype
    )
    dist.all_gather_into_tensor(logits_nt_full, logits_nt, group=tp_pg)
    logits_nt_full = logits_nt_full.permute(1, 2, 0, 3).reshape(BATCH_SIZE, SEQUENCE_LENGTH, -1)
    torch.testing.assert_close(
        logits_nt, logits_nt_full[:, :, tp_rank * sharded_vocab_size : (tp_rank + 1) * sharded_vocab_size]
    )
    if tp_rank == 0:
        torch.save(logits_nt_full.detach().cpu(), nt_path / "logits_nt.pt")


def _convert_from_parallel(parallel_context: ParallelContext, input_ids: torch.Tensor, nt_path: Path, hf_path: Path):
    # [FUNCTION DOC]
    # HELPER: second stage of test_tensor_parallel_conversion (single process).
    # Behavior: converts the checkpoint at nt_path to HF format at hf_path (Qwen2 config class), loads it as
    # LlamaForCausalLM on CUDA, runs input_ids and saves the logits on CPU to hf_path/logits_hf.pt.
    # Inputs: parallel_context (unused); input_ids; nt_path; hf_path (Paths). Outputs: None.
    # Exceptions: None explicitly raised. Side effects: disk reads/writes, CUDA.
    # Convert parallel nanotron to hf, get and save huggingface predictions.
    convert_nt_to_hf_and_save(nt_path, hf_path, config_cls=NanotronQwen2Config)
    model_hf = LlamaForCausalLM.from_pretrained(hf_path).cuda()
    logits_hf = model_hf(input_ids).logits
    torch.save(logits_hf.detach().cpu(), hf_path / "logits_hf.pt")


def test_tensor_parallel_conversion(input_ids: torch.Tensor):
    # [FUNCTION DOC]
    # Test: end-to-end check that a tp=2 nanotron checkpoint converts to HF with matching logits.
    # Behavior: creates temp paths, runs _save_parallel_nanotron with tp=2, asserts logits_nt.pt exists, runs
    # _convert_from_parallel with tp=1, asserts logits_hf.pt exists, loads both and asserts closeness within ATOL.
    # Inputs: input_ids (batch, seq). Outputs: None. Exceptions: AssertionError on missing files or mismatch.
    # Side effects: temp files, worker processes.
    # KNOWN ISSUE: hf_path is set to the same "nanotron" directory as nt_path (likely meant "hf"), so the HF
    # export is written into the nanotron checkpoint folder.
    # Set up test.
    test_context = TestContext()
    root = test_context.get_auto_remove_tmp_dir()
    nt_path = root / "nanotron"
    hf_path = root / "nanotron"

    # Launch both parts.
    init_distributed(tp=2, dp=1, pp=1)(_save_parallel_nanotron)(input_ids=input_ids, nt_path=nt_path)
    assert (nt_path / "logits_nt.pt").exists()
    init_distributed(tp=1, dp=1, pp=1)(_convert_from_parallel)(input_ids=input_ids, nt_path=nt_path, hf_path=hf_path)
    assert (hf_path / "logits_hf.pt").exists()

    # Load logits and verify they match.
    logits_nt = torch.load(nt_path / "logits_nt.pt")
    logits_hf = torch.load(hf_path / "logits_hf.pt")
    torch.testing.assert_allclose(logits_nt, logits_hf, atol=ATOL, rtol=ATOL)


if __name__ == "__main__":
    # run all tests
    # test_nt_to_hf(input_ids=torch.randint(0, CONFIG.vocab_size, size=(BATCH_SIZE, SEQUENCE_LENGTH), device="cuda"))
    # test_hf_to_nt(input_ids=torch.randint(0, CONFIG.vocab_size, size=(BATCH_SIZE, SEQUENCE_LENGTH), device="cuda"))
    test_tensor_parallel_conversion(
        input_ids=torch.randint(0, CONFIG.vocab_size, size=(BATCH_SIZE, SEQUENCE_LENGTH), device="cuda")
    )

    # Warning: Converting from HF to Nanotron is a better test because we don't initialize weights in standard way. (e.g. Layernorms)
    # Test SmolLM2-135M
    # test_hf_to_nt(
    #     input_ids=torch.randint(0, CONFIG.vocab_size, size=(BATCH_SIZE, SEQUENCE_LENGTH), device="cuda"),
    #     model_name="HuggingFaceTB/SmolLM2-135M",
    # )
