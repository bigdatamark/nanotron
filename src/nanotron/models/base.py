# [MODULE DOC]
# Purpose: shared foundations for nanotron models: the abstract NanotronModel base class, a dtype-locked tensor
# type, the build_model routine that assigns pipeline-parallel (PP) ranks to PipelineBlocks by balancing compute
# cost, and context managers that force parameters/buffers/tensor constructors onto a device and dtype during construction.
# Fits in nanotron: every concrete model (LlamaForTraining, Qwen2ForTraining, Starcoder2ForTraining, ...) subclasses
# NanotronModel and is built through build_model inside init_on_device_and_dtype.
# Import-time side effects: creates a module logger. Exports re-exported by nanotron.models.
import threading
from abc import ABCMeta, abstractmethod
from contextlib import contextmanager
from typing import TYPE_CHECKING, Callable, Dict, Iterator, List, Optional, Tuple

import numpy as np
import torch
from torch import nn

from nanotron import distributed as dist
from nanotron import logging
from nanotron.distributed import ProcessGroup
from nanotron.logging import log_rank
from nanotron.parallel.context import ParallelContext
from nanotron.parallel.pipeline_parallel.block import PipelineBlock
from nanotron.logging import LoggingCollectorMixin
if TYPE_CHECKING:
    from nanotron.config import NanotronConfigs
    from nanotron.parallel.parameters import NanotronParameter

logger = logging.get_logger(__name__)


class NanotronModel(nn.Module, LoggingCollectorMixin, metaclass=ABCMeta):
    # [CLASS DOC]
    # Purpose: abstract base for all nanotron models (nn.Module plus LoggingCollectorMixin, ABCMeta metaclass).
    # Key attributes: parallel_context and config (declared, set by subclasses); input_pp_rank and output_pp_rank (set by
    # build_model); module_id_to_prefix (dict from id of each submodule to its qualified name plus a trailing dot, the
    # root mapping to the empty string), used to compute names of tied parameters.
    # Assumption: PipelineBlocks are registered in forward-pass order. Subclasses must implement init_model_randomly and
    # typically get_block_compute_costs, get_flops_per_sec, etc.
    """Abstract class for Nanotron models
    We make the following assumptions:
    - When building PP blocks, we assume that the modules order are in the same order as the forward pass."""

    def __init__(self, *args, **kwargs) -> None:
        # [FUNCTION DOC]
        # Behavior: run parent initializers, declare typed attributes, build module_id_to_prefix from named_modules (computed at
        # this point, so subclasses usually recompute it after creating submodules), and map self to the empty prefix.
        # Inputs: args/kwargs passed through. Outputs: None. Exceptions: None explicitly raised. Side effects: sets attributes.
        super().__init__(*args, **kwargs)
        self.parallel_context: "ParallelContext"
        self.config: "NanotronConfigs"
        self.module_id_to_prefix: dict[int, str]

        # Attributes defined when building the model
        self.input_pp_rank: int
        self.output_pp_rank: int

        # Useful mapping to get param names
        self.module_id_to_prefix = {id(module): f"{module_name}." for module_name, module in self.named_modules()}
        self.module_id_to_prefix[id(self)] = ""

    def get_named_params_with_correct_tied(self) -> Iterator[Tuple[str, "NanotronParameter"]]:
        # [FUNCTION DOC]
        # Behavior: generator over named_parameters; for tied parameters yield the full name derived from the tie info and
        # module_id_to_prefix (so all ranks agree on one name), otherwise yield the local name.
        # Inputs: none. Outputs: iterator of (name, NanotronParameter) pairs. Exceptions: AttributeError if a parameter is not a NanotronParameter.
        # Side effects: None.
        """Return named parameters with correct tied params names.
        For example in the case of tied kv heads in MQA, we need to make sure tied params names are correct."""

        def params_gen():
            # [FUNCTION DOC]
            # HELPER: inner generator implementing the iteration described by the enclosing method.
            # Inputs: none. Outputs: generator of (name, parameter). Exceptions/Side effects: as the enclosing method.
            for name, param in self.named_parameters():
                if param.is_tied:
                    yield (
                        param.get_tied_info().get_full_name_from_module_id_to_prefix(
                            module_id_to_prefix=self.module_id_to_prefix
                        ),
                        param,
                    )
                else:
                    yield name, param

        yield from params_gen()

    @abstractmethod
    def init_model_randomly(self, config):
        # [FUNCTION DOC]
        # Abstract: subclasses must initialize all parameters randomly according to config.
        # Inputs: config (nanotron Config). Outputs: None. Exceptions: none here. Side effects: none here.
        ...

    def tie_custom_params(self) -> None:
        # [FUNCTION DOC]
        # Behavior: hook to tie model-specific parameters (e.g. MQA kv heads); default does nothing.
        # Inputs: none. Outputs: None. Exceptions: None. Side effects: None.
        """Tie custom parameters. For example for MQA marks kv heads as tied."""
        pass

    def get_embeddings_lm_head_tied_names(self) -> list[str]:
        # [FUNCTION DOC]
        # Behavior: default returns an empty list meaning embeddings and lm_head are not tied; subclasses return the two parameter names.
        # Inputs: none. Outputs: list of strings. Exceptions: None. Side effects: None.
        """Returns the names of the embeddings and lm_head weights that are tied together. Returns empty list if not tied.

        Example for GPT2 model: ["model.token_position_embeddings.pp_block.token_embedding.weight", "model.lm_head.pp_block.weight"]
        """
        return []

    def before_tbi_sanity_checks(self) -> None:
        # [FUNCTION DOC]
        # Hook called before a train-batch iteration; default no-op. Inputs: none. Outputs: None. Exceptions: None. Side effects: None.
        pass

    def after_tbi_sanity_checks(self) -> None:
        # [FUNCTION DOC]
        # Hook called after a train-batch iteration; default no-op. Inputs: none. Outputs: None. Exceptions: None. Side effects: None.
        pass

    def before_optim_step_sanity_checks(self) -> None:
        # [FUNCTION DOC]
        # Hook called before the optimizer step; default no-op. Inputs: none. Outputs: None. Exceptions: None. Side effects: None.
        pass

    def after_optim_step_sanity_checks(self) -> None:
        # [FUNCTION DOC]
        # Hook called after the optimizer step; default no-op. Inputs: none. Outputs: None. Exceptions: None. Side effects: None.
        pass

    def log_modules(self, level: int = logging.DEBUG, group: Optional[ProcessGroup] = None, rank: int = 0):
        # [FUNCTION DOC]
        # Behavior: for every PipelineBlock submodule log its name and assigned PP rank out of the PP group size via log_rank.
        # Inputs: level (logging level, default DEBUG); group (process group, default None); rank (int, default 0) passed to log_rank.
        # Outputs: None. Exceptions: AssertionError if parallel_context is not set. Side effects: logging.
        assert hasattr(self, "parallel_context"), "`NanotronModel` needs to have a `parallel_context` attribute"

        for name, module in self.named_modules():
            if not isinstance(module, PipelineBlock):
                continue
            log_rank(
                f"module_name: {name} | PP: {module.rank}/{self.parallel_context.pp_pg.size()}",
                logger=logger,
                level=level,
                group=group,
                rank=rank,
            )

    @property
    def named_modules_in_pp_rank(self) -> Dict[str, nn.Module]:
        # [FUNCTION DOC]
        # Behavior (property): collect leaf modules (no children) of the model and keep those that are not PipelineBlocks.
        # A PipelineBlock that is a leaf was never built on this rank (it belongs to another PP stage), so it is excluded.
        # Inputs: none. Outputs: dict mapping qualified module name to module. Exceptions: None. Side effects: None.
        """Return the named modules that only belongs to the current pp rank.

        An example output:
        {
            'module_name': module,
            ...
        }

        NOTE: not include module_name.weight or bias, but only module_name
        """

        def get_leaf_modules(module: nn.Module) -> List[Tuple[str, nn.Module]]:
            # [FUNCTION DOC]
            # HELPER: lists leaf modules for the enclosing property.
            # Behavior: return (name, module) pairs from named_modules whose children list is empty.
            # Inputs: module (nn.Module). Outputs: list of (str, nn.Module). Exceptions: None. Side effects: None.
            """
            Return all the leaf modules (modules without any child modules) in a PyTorch module.
            """
            leaf_modules = []
            for n, m in module.named_modules():
                if not list(m.children()):
                    leaf_modules.append((n, m))
            return leaf_modules

        modules = get_leaf_modules(self)
        named_modules_in_current_pp_rank = {}
        for name, module in modules:
            if isinstance(module, PipelineBlock):
                # NOTE: these are the modules that aren't belong to the current pp rank
                continue
            named_modules_in_current_pp_rank[name] = module

        return named_modules_in_current_pp_rank


class DTypeInvariantTensor(torch.Tensor):
    # [CLASS DOC]
    # Purpose: torch.Tensor subclass whose dtype cannot be changed (used for buffers that must stay e.g. float32, such as rotary inverse frequencies).
    # Behavior: detach and every dtype-conversion method raise RuntimeError; to is allowed only without a dtype argument.
    # Data and other attributes may still be modified in place.
    """DTypeInvariantTensor is a subclass of torch.Tensor that disallows modification of its dtype. Note that the data
    and other attributes of the tensor can still be modified."""

    def __new__(cls, *args, **kwargs):
        # [FUNCTION DOC]
        # Behavior: construct via the torch.Tensor constructor with the given args and return it.
        # Inputs: args/kwargs for tensor creation. Outputs: new DTypeInvariantTensor. Exceptions: those of torch. Side effects: allocation.
        tensor = super().__new__(cls, *args, **kwargs)
        return tensor

    def detach(self, *args, **kwargs):
        # [FUNCTION DOC]
        # Always raises RuntimeError (detaching is forbidden). Inputs: ignored. Outputs: none. Side effects: None.
        raise RuntimeError("Cannot detach an DTypeInvariantTensor")

    def to(self, *args, **kwargs):
        # [FUNCTION DOC]
        # Behavior: raise RuntimeError if a dtype keyword or any positional torch.dtype is given; otherwise delegate to torch.Tensor.to.
        # Inputs: same as torch.Tensor.to. Outputs: tensor. Exceptions: RuntimeError on dtype change. Side effects: possible device copy.
        # KNOWN ISSUE: a dtype change via a positional tensor argument (converting to another tensor's dtype) is not detected.
        if "dtype" in kwargs or any(isinstance(arg, torch.dtype) for arg in args):
            raise RuntimeError("Cannot change the type of an DTypeInvariantTensor")
        else:
            return super().to(*args, **kwargs)

    def type(self, *args, **kwargs):
        # [FUNCTION DOC]
        # Always raises RuntimeError. Inputs: ignored. Outputs: none. Side effects: None.
        raise RuntimeError("Cannot change the type of an DTypeInvariantTensor")

    def float(self, *args, **kwargs):
        # [FUNCTION DOC]
        # Always raises RuntimeError. Inputs: ignored. Outputs: none. Side effects: None.
        raise RuntimeError("Cannot convert the type of an DTypeInvariantTensor to float")

    def double(self, *args, **kwargs):
        # [FUNCTION DOC]
        # Always raises RuntimeError. Inputs: ignored. Outputs: none. Side effects: None.
        raise RuntimeError("Cannot convert the type of an DTypeInvariantTensor to double")

    def half(self, *args, **kwargs):
        # [FUNCTION DOC]
        # Always raises RuntimeError. Inputs: ignored. Outputs: none. Side effects: None.
        raise RuntimeError("Cannot convert the type of an DTypeInvariantTensor to half")

    def long(self, *args, **kwargs):
        # [FUNCTION DOC]
        # Always raises RuntimeError. Inputs: ignored. Outputs: none. Side effects: None.
        raise RuntimeError("Cannot convert the type of an DTypeInvariantTensor to long")

    def int(self, *args, **kwargs):
        # [FUNCTION DOC]
        # Always raises RuntimeError. Inputs: ignored. Outputs: none. Side effects: None.
        raise RuntimeError("Cannot convert the type of an DTypeInvariantTensor to int")

    def short(self, *args, **kwargs):
        # [FUNCTION DOC]
        # Always raises RuntimeError. Inputs: ignored. Outputs: none. Side effects: None.
        raise RuntimeError("Cannot convert the type of an DTypeInvariantTensor to short")

    def char(self, *args, **kwargs):
        # [FUNCTION DOC]
        # Always raises RuntimeError. Inputs: ignored. Outputs: none. Side effects: None.
        raise RuntimeError("Cannot convert the type of an DTypeInvariantTensor to char")

    def byte(self, *args, **kwargs):
        # [FUNCTION DOC]
        # Always raises RuntimeError. Inputs: ignored. Outputs: none. Side effects: None.
        raise RuntimeError("Cannot convert the type of an DTypeInvariantTensor to byte")

    def bool(self, *args, **kwargs):
        # [FUNCTION DOC]
        # Always raises RuntimeError. Inputs: ignored. Outputs: none. Side effects: None.
        raise RuntimeError("Cannot convert the type of an DTypeInvariantTensor to bool")

    def bfloat16(self, *args, **kwargs):
        # [FUNCTION DOC]
        # Always raises RuntimeError. Inputs: ignored. Outputs: none. Side effects: None.
        raise RuntimeError("Cannot convert the type of an DTypeInvariantTensor to bfloat16")


def build_model(
    model_builder: Callable[[], NanotronModel],
    parallel_context: ParallelContext,
    dtype: torch.dtype,
    target_pp_ranks: Optional[List[int]] = None,
    device: Optional[torch.device] = torch.device("cuda"),
) -> NanotronModel:
    # [FUNCTION DOC]
    # Behavior:
    # 1. Log "Building model" on world rank 0 and call model_builder to create the (unbuilt) model skeleton.
    # 2. Target PP ranks default to all ranks of the PP group; pp_size is their count.
    # 3. Collect all PipelineBlocks in registration (forward) order.
    # 4. Inside init_on_device_and_dtype(device, dtype): get per-module-builder costs from model.get_block_compute_costs,
    #    form the cumulative sum of block costs (unknown builders cost 0), and thresholds equal to total cost times
    #    (r+1)/pp_size for each stage r. Walk blocks in order: build each block on the current target rank (the block only
    #    instantiates its module on that rank), then advance to the next rank once the cumulative cost exceeds the current threshold.
    # 5. Set input_pp_rank to the first target rank and output_pp_rank to the rank reached at the end.
    # 6. If pp_size above 1, log module-to-rank assignment.
    # Inputs: model_builder (callable returning NanotronModel); parallel_context; dtype (parameter dtype); target_pp_ranks
    # (optional list of PP ranks, default all); device (default cuda, i.e. the process's current device).
    # Outputs: the model with PP ranks assigned and local modules materialized.
    # Exceptions: AssertionError if thresholds are inconsistent or the rank index overflows.
    # Side effects: logging; allocates parameters on device; temporarily monkeypatches torch constructors (see init_on_device_and_dtype).
    """Build the model and set the pp ranks for each pipeline block."""
    # TODO: classes dont take same args
    log_rank(
        "Building model", logger=logger, level=logging.INFO, rank=0, group=parallel_context.world_pg, is_separator=True
    )
    model: NanotronModel = model_builder()

    # If no target pp ranks are specified, we assume that we want to use all pp ranks
    if target_pp_ranks is None:
        pp_size = parallel_context.pp_pg.size()
        target_pp_ranks = list(range(pp_size))
    else:
        pp_size = len(target_pp_ranks)

    # Set rank for each pipeline block
    log_rank("Setting PP block ranks...", logger=logger, level=logging.INFO, rank=0, group=parallel_context.world_pg)
    pipeline_blocks = [module for name, module in model.named_modules() if isinstance(module, PipelineBlock)]
    # "cuda" is already defaulted for each process to it's own cuda device
    with init_on_device_and_dtype(device=device, dtype=dtype):
        # TODO: https://github.com/huggingface/nanotron/issues/65

        # Balance compute across PP blocks
        block_compute_costs = model.get_block_compute_costs()
        block_cumulative_costs = np.cumsum(
            [
                block_compute_costs[module.module_builder] if module.module_builder in block_compute_costs else 0
                for module in pipeline_blocks
            ]
        )

        thresholds = [block_cumulative_costs[-1] * ((rank + 1) / pp_size) for rank in range(pp_size)]
        assert thresholds[-1] >= block_cumulative_costs[-1]
        target_pp_rank_idx = 0
        for block, cumulative_cost in zip(pipeline_blocks, block_cumulative_costs):
            assert target_pp_rank_idx < pp_size
            block.build_and_set_rank(target_pp_ranks[target_pp_rank_idx])

            if cumulative_cost > thresholds[target_pp_rank_idx]:
                target_pp_rank_idx += 1

        model.input_pp_rank = target_pp_ranks[0]
        model.output_pp_rank = target_pp_ranks[target_pp_rank_idx]

    if pp_size > 1:
        model.log_modules(level=logging.INFO, group=parallel_context.world_pg, rank=0)
    return model


@contextmanager
def ignore_init_on_device_and_dtype():
    # [FUNCTION DOC]
    # Behavior (context manager): lazily create a thread-local flag stored as an attribute on this function, set it to True
    # for the duration of the context, then restore the previous value, so parameter/buffer registration inside
    # init_on_device_and_dtype keeps its own dtype and device.
    # Inputs: none. Outputs: yields None. Exceptions: None explicitly raised.
    # Side effects: mutates thread-local state attached to the function object.
    """
    A context manager that temporarily disables dtype enforcement from init_on_device_and_dtype.

    Example:
    ```python
    with init_on_device_and_dtype(device=torch.device("cuda"), dtype=torch.float32):
        with ignore_init_on_device_and_dtype():
            # This parameter will keep its specified dtype (float32)
            self.weight = nn.Parameter(torch.randn(..., dtype=torch.float32))
    ```
    """
    # Create a thread-local storage for the ignore flag
    if not hasattr(ignore_init_on_device_and_dtype, "_ignore_flag"):
        ignore_init_on_device_and_dtype._ignore_flag = threading.local()

    # Set the ignore flag
    old_value = getattr(ignore_init_on_device_and_dtype._ignore_flag, "value", False)
    ignore_init_on_device_and_dtype._ignore_flag.value = True

    try:
        yield
    finally:
        # Restore the previous value
        ignore_init_on_device_and_dtype._ignore_flag.value = old_value


@contextmanager
def init_on_device_and_dtype(
    device: torch.device = torch.device("cpu"),
    dtype: torch.dtype = torch.float,
):
    # [FUNCTION DOC]
    # Behavior (context manager): temporarily replaces nn.Module.register_parameter and register_buffer with versions that
    # register normally and then move the parameter data or buffer to the target device and dtype (unless the ignore flag
    # is set), and replaces torch.empty, zeros, ones and full with wrappers that force the device and dtype keywords.
    # Restores all originals on exit, even on exception.
    # Inputs: device (default cpu); dtype (default float32).
    # Outputs: yields None. Exceptions: None explicitly raised.
    # Side effects: global monkeypatching of torch and nn.Module (not thread-safe), device allocations.
    # KNOWN ISSUE: the patched tensor constructors ignore the ignore_init_on_device_and_dtype flag and always override dtype and device, contrary to the docstring.
    # KNOWN ISSUE: buffers are converted to the target dtype too, which raises for DTypeInvariantTensor buffers unless inside the ignore context.
    """
    A context manager under which models are initialized with all parameters on the specified device.
    Args:
        device (`torch.device` defaults to `cpu`):
            Device to initialize all parameters on.
        dtype (`torch.dtype` defaults to `torch.float`):
            Dtype to initialize all parameters on. If specified, will override any dtype
            set in parameter initialization with a warning, unless within an ignore_init_on_device_and_dtype context.
    """
    old_register_parameter = nn.Module.register_parameter
    old_register_buffer = nn.Module.register_buffer

    def should_ignore_init_on_device_and_dtype():
        # [FUNCTION DOC]
        # HELPER: reads the thread-local ignore flag. Outputs: bool (False if the flag was never created). Exceptions/Side effects: None.
        if not hasattr(ignore_init_on_device_and_dtype, "_ignore_flag"):
            return False
        return getattr(ignore_init_on_device_and_dtype._ignore_flag, "value", False)

    def register_empty_parameter(module, name, param):
        # [FUNCTION DOC]
        # HELPER: patched register_parameter.
        # Behavior: call the original, then (if param not None and not ignored) move param.data to device and dtype.
        # Inputs: module, name, param. Outputs: None. Exceptions: those of the original. Side effects: registers and converts the parameter.
        old_register_parameter(module, name, param)
        if param is not None:
            if should_ignore_init_on_device_and_dtype():
                pass
            else:
                param.data = param.data.to(device, dtype)

    def register_empty_buffer(module, name, buffer, persistent=True):
        # [FUNCTION DOC]
        # HELPER: patched register_buffer.
        # Behavior: call the original with persistent, then (if buffer not None and not ignored) replace the stored buffer with its converted copy.
        # Inputs: module, name, buffer, persistent (default True). Outputs: None. Exceptions: those of the original and of the conversion.
        # Side effects: registers and converts the buffer.
        old_register_buffer(module, name, buffer, persistent=persistent)
        if buffer is not None:
            if should_ignore_init_on_device_and_dtype():
                pass
            else:
                module._buffers[name] = module._buffers[name].to(device, dtype)

    # Patch tensor creation
    tensor_constructors_to_patch = {
        torch_function_name: getattr(torch, torch_function_name)
        for torch_function_name in ["empty", "zeros", "ones", "full"]
    }

    def patch_tensor_constructor(fn):
        # [FUNCTION DOC]
        # HELPER: wraps a torch constructor.
        # Behavior: return a wrapper that overwrites device and dtype keywords then calls fn. Inputs: fn. Outputs: wrapper. Exceptions/Side effects: None.
        def wrapper(*args, **kwargs):
            # [FUNCTION DOC]
            # HELPER: forced-device/dtype constructor call. Inputs: args/kwargs of the constructor. Outputs: new tensor. Exceptions: those of fn. Side effects: allocation.
            kwargs["device"] = device
            kwargs["dtype"] = dtype
            return fn(*args, **kwargs)

        return wrapper

    try:
        nn.Module.register_parameter = register_empty_parameter
        nn.Module.register_buffer = register_empty_buffer
        for torch_function_name in tensor_constructors_to_patch.keys():
            setattr(torch, torch_function_name, patch_tensor_constructor(getattr(torch, torch_function_name)))
        yield
    finally:
        nn.Module.register_parameter = old_register_parameter
        nn.Module.register_buffer = old_register_buffer
        for torch_function_name, old_torch_function in tensor_constructors_to_patch.items():
            setattr(torch, torch_function_name, old_torch_function)


def check_model_has_grad(model: NanotronModel, parallel_context: "ParallelContext"):
    # [FUNCTION DOC]
    # Behavior: return True as soon as any local parameter requires grad; otherwise raise.
    # Inputs: model; parallel_context (used for the PP rank in the message). Outputs: True.
    # Exceptions: ValueError when this PP rank has no trainable parameter (DDP would fail). Side effects: None.
    """Check that there's at least a parameter in current PP rank that has a gradient."""
    for param in model.parameters():
        if param.requires_grad:
            return True
    raise ValueError(
        f"Can't use DDP because model in PP={dist.get_rank(parallel_context.pp_pg)} has no gradient. Consider increasing the number of layers of your model, or put a smaller PP size.\n"
        f"Model: {model}"
    )
