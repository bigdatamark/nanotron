# [MODULE DOC]
# Purpose: ZeRO stage-1 optimizer state sharding across data-parallel ranks and utilities to merge ZeRO-1
# optimizer checkpoints.
# Main contents: ZeroDistributedOptimizer; SlicedFlatTensor and get_sliced_tensor (leaf views on flat slices of
# params whose grad maps to the original param's grad slice); checkpoint helpers find_optim_index_from_param_name,
# extract_parallel_ranks_from_shard_path, merge_dp_shard_in_zero1_optimizer.
# Import-time side effects: creates a module logger; imports functorch and tqdm.
import itertools
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import torch.optim
from functorch.dim import tree_map
from torch import nn
from tqdm import tqdm

from nanotron import distributed as dist
from nanotron import logging
from nanotron.distributed import ProcessGroup
from nanotron.logging import human_format, log_rank, warn_once
from nanotron.optim.base import BaseOptimizer
from nanotron.optim.inherit_from_other_optimizer import InheritFromOtherOptimizer
from nanotron.parallel import ParallelContext
from nanotron.parallel.parameters import NanotronParameter

logger = logging.get_logger(__name__)


class ZeroDistributedOptimizer(InheritFromOtherOptimizer):
    # [CLASS DOC]
    # Purpose: each DP rank owns a contiguous flat slice of every parameter; its inner optimizer only holds those
    # slices (so optimizer states are 1/DP size); after each step updated slices are all-gathered.
    # Key state: zero_named_param_groups (all grad-requiring named params, per group), dp_pg,
    # param_name_to_dp_rank_offsets (name to dict dp_rank to (start, end) flat offsets, empty slices omitted),
    # _orig_param_shapes (name to original shape), optimizer (inner, over sliced tensors), id_to_name (only this
    # rank's params).
    # Inheritance: InheritFromOtherOptimizer.
    """Optimizer that handles partitioning of optimizer's states across DP ranks. See ZeRO Stage 1 in the paper https://arxiv.org/abs/1910.02054v3 for more details."""

    def __init__(
        self,
        named_params_or_groups: Iterable[Union[Tuple[str, NanotronParameter], Dict[str, Any]]],
        optimizer_builder: Callable[[Iterable[Dict[str, Any]]], BaseOptimizer],
        dp_pg: ProcessGroup,
    ):
        # [FUNCTION DOC]
        # Behavior: normalizes input to group dicts (asserting "named_params" in each) keeping only params requiring
        # grad; stores dp_pg; partitions params; for each group builds this rank's named SlicedFlatTensors (skipping params
        # with no slice on this rank, but keeping the group); builds the inner optimizer and wraps it.
        # Inputs: named_params_or_groups (pairs or group dicts), optimizer_builder (callable returning a BaseOptimizer with
        # id_to_name), dp_pg (ProcessGroup).
        # Outputs: None.
        # Exceptions: AssertionError if a group dict lacks "named_params" or a param is non-contiguous.
        # Side effects: logs shard sizes on rank 0.
        named_params_or_groups = list(named_params_or_groups)
        if len(named_params_or_groups) == 0 or isinstance(named_params_or_groups[0], dict):
            # case where named_params_or_groups is Iterable[Dict[str, Any]]
            for d in named_params_or_groups:
                assert (
                    "named_params" in d
                ), f"param_groups must contain a 'named_params' key, got a dict with keys {d.keys()}"

            # keep only named_params_or_groups that require grads
            named_params_or_groups = [
                {
                    "named_params": [
                        (name, param) for name, param in named_param_group["named_params"] if param.requires_grad
                    ],
                    **{k: v for k, v in named_param_group.items() if k != "named_params"},
                }
                for named_param_group in named_params_or_groups
            ]

            self.zero_named_param_groups = named_params_or_groups
        else:
            # case where named_params_or_groups is Iterable[Tuple[str, NanotronParameter]]
            # keep only named_params_or_groups that require grads
            named_params_or_groups = [(name, param) for name, param in named_params_or_groups if param.requires_grad]
            self.zero_named_param_groups = [{"named_params": named_params_or_groups}]

        self.dp_pg = dp_pg  # DP process group

        # partition model's params across DP ranks.
        # `self.param_name_to_dp_rank_offsets` sets mapping between each param inside self.named_params and its rank
        # NOTE: some param_groups may have no params in the current rank. we still keep them in self.optimizer.param_groups
        self.param_name_to_dp_rank_offsets = self._partition_parameters()

        current_dp_rank = dist.get_rank(self.dp_pg)
        param_groups_in_rank = [
            {
                "named_params": [
                    (
                        name,
                        get_sliced_tensor(
                            param=param,
                            start_offset=self.param_name_to_dp_rank_offsets[name][current_dp_rank][0],
                            end_offset=self.param_name_to_dp_rank_offsets[name][current_dp_rank][1],
                        ),
                    )
                    for name, param in param_group["named_params"]
                    if current_dp_rank in self.param_name_to_dp_rank_offsets[name]
                ],
                **{k: v for k, v in param_group.items() if k != "named_params"},
            }
            for param_group in self.zero_named_param_groups
        ]

        # initialize rank's optimizer which is responsible for updating the rank's parameters
        # NOTE: In case of ZeRO, `self.id_to_name` stores only names of parameters that are going to be updated by this DP rank's optimizer.
        # NOTE: In case of ZeRO, `self.optimizer` will only get the parameters that are going to be updated by this DP's optimizer. Which
        # means that `self.optimizer.param_groups` is only a subset of `self.param_groups`.
        optimizer = optimizer_builder(param_groups_in_rank)
        super().__init__(optimizer=optimizer, id_to_name=optimizer.id_to_name)

    @torch.no_grad()
    def step(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]:
        # [FUNCTION DOC]
        # Behavior: under no_grad runs the inner step on local slices, then all-gathers updated params across DP.
        # Inputs: closure (optional). Outputs: loss from inner step. Exceptions: None explicitly raised.
        # Side effects: parameter updates; all-gather collective on dp_pg.
        """Performs a single optimization step (parameter update)."""
        # TODO: @nouamanetazi: handle syncing param groups attrs (e.g. if we update lr)

        loss = super().step(closure=closure)

        # calculate param_size (model) + param_size (grads) + 2*param_size/DP_if_zero1 (optim_states)
        # expected_allocated = sum(
        #     param.numel() * param.element_size() * 2 + param.numel() * param.element_size() * 2 / self.dp_pg.size()
        #     for named_param_group in self.zero_named_param_groups
        #     for _, param in named_param_group["named_params"]
        # )

        # log_rank(
        #     f"[After optim states allocation] Memory usage: {torch.cuda.memory_allocated() / 1024**2:.2f}MB "
        #     f"(Expected 2*param_size + 2*param_size/DP_if_zero1={expected_allocated / 1024**2:.2f}MB). "
        #     f"Peak reserved memory: {torch.cuda.max_memory_reserved() / 1024**2:.2f}MB",
        #     logger=logger,
        #     level=logging.DEBUG,
        #     group=self.dp_pg,
        #     rank=0,
        # )

        # All gather updated params
        self._all_gather_params()
        return loss

    def zero_grad(self):
        # [FUNCTION DOC]
        # Behavior: calls inner zero_grad, then sets grad to None for all model params in zero_named_param_groups (not only
        # this rank's slices).
        # Outputs: None. Exceptions: None. Side effects: clears gradients.
        # KNOWN ISSUE: foreach is hardcoded False, so the foreach branches are dead code (and would zero empty lists).
        """Copied from `torch.optim.optimizer.zero_grad` with the only change of using `self.param_groups` instead of `self.optimizer.param_groups`
        because we want to zero out the gradients of all model params (not just the params in the current rank)"""
        super().zero_grad()

        # TODO @thomasw21: This is a call to torch internal API, we need to fix this
        foreach = False  # self.optimizer.defaults.get("foreach", False)

        # TODO @thomasw21: This is a call to torch internal API, we need to fix this
        # if not hasattr(self.optimizer, "_zero_grad_profile_name"):
        #     self.optimizer._hook_for_profile()

        if foreach:
            per_device_and_dtype_grads = defaultdict(lambda: defaultdict(list))

        # TODO @thomasw21: This is a call to torch internal API, we need to fix this
        # with torch.autograd.profiler.record_function(self.optimizer._zero_grad_profile_name):

        # zero out the gradients of all model params (not just the params in the current rank)
        for named_param_group in self.zero_named_param_groups:
            for _, p in named_param_group["named_params"]:
                if p.grad is not None:
                    p.grad = None
        if foreach:
            for _, per_dtype_grads in per_device_and_dtype_grads.items():
                for grads in per_dtype_grads.values():
                    torch._foreach_zero_(grads)

    def _partition_parameters(self) -> Dict[str, Dict[int, Tuple[int, int]]]:
        # [FUNCTION DOC]
        # HELPER: computes per-DP-rank flat slices of each parameter.
        # Behavior: records original shapes. For each param (asserting contiguous), splits numel into dp size chunks
        # of ceil(numel/dp) where the last remainder chunks are one smaller, computes start/end via cumulative sums, and
        # keeps only non-empty ranges. Logs, on rank 0, each rank's share of total numel.
        # Inputs: none (uses self.zero_named_param_groups and self.dp_pg).
        # Outputs: dict name to dict dp_rank to (start, end) as numpy integers.
        # Exceptions: AssertionError if non-contiguous or offsets do not sum to numel.
        # Side effects: sets self._orig_param_shapes; logging.
        named_params = [
            (name, param)
            for named_param_group in self.zero_named_param_groups
            for name, param in named_param_group["named_params"]
            if param.requires_grad
        ]

        # maps each model's param to the optimizer's dp rank that is responsible for updating it

        # We assume that parameters can be sharded across DP, ie we can "split" a parameter in different DP. This does break some optimizers, like Adafactor and such.
        # `param_name_to_dp_rank_offsets[name]` is a `Dict[int, Tuple[int, int]]` keys are dp_rank, and `Tuple[int, int]` are the offsets of the param belonging to this DP
        param_name_to_dp_rank_offsets = {}

        # NOTE: save the original shapes before flattening the params
        # so that later on, we can reshape the params to their original shapes
        # for topology-agnostic optimizer states loading
        self._orig_param_shapes = {}
        for name, param in named_params:
            self._orig_param_shapes[name] = param.shape

        for name, param in named_params:
            # We assume parameter to be contiguous in order to have an easy way of sharding it.
            assert param.is_contiguous(), f"Parameter {name} is not contiguous"

            numel = param.numel()
            padded_numel_per_dp = (numel - 1) // self.dp_pg.size() + 1
            sizes = np.full(shape=(self.dp_pg.size()), fill_value=padded_numel_per_dp)
            remainder = padded_numel_per_dp * self.dp_pg.size() - numel
            # Last `remainder` indices has one less element
            if remainder > 0:
                # It's weird that `size[-0:]` returns the entire list instead of nothing
                sizes[-remainder:] -= 1
            end_offsets = np.cumsum(sizes)
            assert len(end_offsets) == self.dp_pg.size()
            assert end_offsets[-1] == numel, f"Somehow {end_offsets[-1]} != {numel}"
            # We want start indices,
            start_offsets = np.concatenate([[0], end_offsets[:-1]])

            param_name_to_dp_rank_offsets[name] = {
                dp_rank: (start_offsets[dp_rank], end_offsets[dp_rank])
                for dp_rank in range(self.dp_pg.size())
                if start_offsets[dp_rank] < end_offsets[dp_rank]  # Only if the slice is not empty.
            }

        log_rank("[ZeRO sharding] Size of optimizer params per rank:", logger=logger, level=logging.INFO, rank=0)
        all_numel = sum(
            param_name_to_dp_rank_offsets[name][dp_rank][1] - param_name_to_dp_rank_offsets[name][dp_rank][0]
            for name, param in named_params
            for dp_rank in range(self.dp_pg.size())
            if dp_rank in param_name_to_dp_rank_offsets[name]
        )
        for dp_rank in range(self.dp_pg.size()):
            acc_numel = sum(
                value[dp_rank][1] - value[dp_rank][0]
                for value in param_name_to_dp_rank_offsets.values()
                if dp_rank in value
            )
            log_rank(
                f"[ZeRO sharding] DP Rank {dp_rank} has {human_format(acc_numel)} out of {human_format(all_numel)} ({0 if all_numel == 0 else acc_numel / all_numel * 100:.2f}%) params' optimizer states",
                logger=logger,
                level=logging.INFO,
                rank=0,
            )

        return param_name_to_dp_rank_offsets

    @torch.profiler.record_function("all_gather_params")
    def _all_gather_params(self):
        # [FUNCTION DOC]
        # HELPER: synchronizes updated param slices to all DP ranks.
        # Behavior: returns early if no params or DP size 1. Otherwise one coalesced all-gather: input is this rank's slice
        # of each flattened param (empty if none); outputs are the per-rank slice views of the same flat param, so params
        # are updated in place.
        # Inputs/Outputs: None. Exceptions: None explicitly raised.
        # Side effects: all_gather_coalesced on dp_pg; overwrites param data.
        """All gather updated params"""
        all_named_tensors_to_gather = [
            (name, param.view(-1))
            for named_param_groups in self.zero_named_param_groups
            for name, param in named_param_groups["named_params"]
        ]

        if len(all_named_tensors_to_gather) == 0:
            # No need to broadcast if there's nothing
            return

        if self.dp_pg.size() == 1:
            # They should already be updated
            return

        current_dp_rank = dist.get_rank(self.dp_pg)
        dist.all_gather_coalesced(
            output_tensor_lists=[
                [
                    tensor[slice(*self.param_name_to_dp_rank_offsets[name][dp_rank])]
                    if dp_rank in self.param_name_to_dp_rank_offsets[name]
                    else torch.empty(0, dtype=tensor.dtype, device=tensor.device)
                    for dp_rank in range(self.dp_pg.size())
                ]
                for name, tensor in all_named_tensors_to_gather
            ],
            input_tensor_list=[
                tensor[slice(*self.param_name_to_dp_rank_offsets[name][current_dp_rank])]
                if current_dp_rank in self.param_name_to_dp_rank_offsets[name]
                else torch.empty(0, dtype=tensor.dtype, device=tensor.device)
                for name, tensor in all_named_tensors_to_gather
            ],
            group=self.dp_pg,
        )


# Helpers


class SlicedFlatTensor(torch.Tensor):
    # [CLASS DOC]
    # Purpose: tensor wrapper subclass representing a flat slice of a parameter that is its own leaf for the optimizer,
    # whose grad property reads and writes the corresponding slice of the original param's grad.
    # Key state: sliced_flat_tensor (actual view), orig_data (original param), start_offset, end_offset.
    # Inheritance: torch.Tensor wrapper subclass with __torch_dispatch__ unwrapping to the view; torch_function disabled.
    """Subclass of `torch.Tensor` that unable to define `grad` getter on a slice of a flattened tensor."""

    # Based on torch/testing/_internal/logging_tensor.py
    # https://github.com/pytorch/pytorch/issues/102337#issuecomment-1579673041
    __torch_function__ = torch._C._disabled_torch_function_impl

    @staticmethod
    def get_sliced_flat_tensor(data, start_offset, end_offset):
        # [FUNCTION DOC]
        # HELPER: returns the flat view of data between offsets under no_grad.
        # Inputs: data (tensor), start_offset, end_offset (ints). Outputs: 1D view. Exceptions/Side effects: None.
        with torch.no_grad():
            return data.view(-1)[start_offset:end_offset]

    @staticmethod
    def __new__(cls, data, start_offset, end_offset):
        # [FUNCTION DOC]
        # Behavior: creates a wrapper subclass instance with the size, strides, storage offset, dtype, layout, device and
        # requires_grad of the flat slice.
        # Inputs: cls, data, start_offset, end_offset. Outputs: new SlicedFlatTensor. Exceptions/Side effects: None.
        sliced_tensor = cls.get_sliced_flat_tensor(data=data, start_offset=start_offset, end_offset=end_offset)

        result = torch.Tensor._make_wrapper_subclass(  # type: ignore[attr-defined]
            cls,
            sliced_tensor.size(),
            strides=sliced_tensor.stride(),
            storage_offset=sliced_tensor.storage_offset(),
            # TODO: clone storage aliasing
            dtype=sliced_tensor.dtype,
            layout=sliced_tensor.layout,
            device=sliced_tensor.device,
            requires_grad=sliced_tensor.requires_grad,
        )
        return result

    def __init__(self, data, start_offset, end_offset):
        # [FUNCTION DOC]
        # Behavior: stores the flat view, original data and offsets.
        # Inputs: data, start_offset, end_offset. Outputs: None. Exceptions/Side effects: None.
        super().__init__()
        # TODO @thomasw21: Make is so that you can never update this value
        self.sliced_flat_tensor = self.get_sliced_flat_tensor(
            data=data, start_offset=start_offset, end_offset=end_offset
        )
        self.orig_data = data
        self.start_offset = start_offset
        self.end_offset = end_offset

    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        # [FUNCTION DOC]
        # Behavior: replaces SlicedFlatTensor args with their views, runs func, and returns results unwrapped (never rewrapped).
        # Inputs: func, types, args, kwargs. Outputs: plain tensor results. Exceptions: propagated from func.
        # Side effects: whatever func does (in-place ops modify the original param storage).
        # KNOWN ISSUE: kwargs defaults to None and is passed to tree_map and then unpacked; works only because tree_map returns None and dispatch always provides a dict.
        def unwrap(e):
            # [FUNCTION DOC]
            # HELPER: returns the inner view for SlicedFlatTensor instances, else the input. Inputs: e. Outputs: tensor or e.
            return e.sliced_flat_tensor if isinstance(e, cls) else e

        def never_wrap(e):
            # [FUNCTION DOC]
            # HELPER: identity function. Inputs: e. Outputs: e.
            # Never re-wrap
            return e

        return tree_map(never_wrap, func(*tree_map(unwrap, args), **tree_map(unwrap, kwargs)))

    def _get_grad(self):
        # [FUNCTION DOC]
        # HELPER: grad getter. Returns None if original param has no grad, else the flat slice of the original grad.
        # Outputs: tensor or None. Exceptions/Side effects: None.
        if self.orig_data.grad is None:
            return None
        with torch.no_grad():
            return self.orig_data.grad.view(-1)[self.start_offset : self.end_offset]

    def _set_grad(self, grad):
        # [FUNCTION DOC]
        # HELPER: grad setter. Non-None grad is copied into the original grad slice; None clears the original param's
        # whole grad (with a one-time warning).
        # Inputs: grad (tensor or None). Outputs: None.
        # Exceptions: NotImplementedError if setting a grad while the original grad is not allocated.
        # Side effects: mutates original param grad; logs warning.
        if grad is not None:
            orig_grad = self._get_grad()
            if orig_grad is None:
                raise NotImplementedError(
                    "Trying to set gradient on a sliced tensor when the original tensor hasn't allocated the buffer for the gradient"
                )
            orig_grad.copy_(grad)
            return
        # TODO @thomasw21: This is unfortunately necessary since we might pass `SliceTensor` to the optimizer.
        warn_once(
            logger=logger,
            msg="You're setting a `SlicedTensor` gradient to None. We're going to assume you meant to set the original tensor gradient to None.",
            rank=0,
        )
        self.orig_data.grad = None

    def _del_grad(self):
        # [FUNCTION DOC]
        # HELPER: grad deleter; always raises NotImplementedError. Side effects: None.
        raise NotImplementedError

    # TODO @thomasw21: Figure out why this function doesn't get inherited. https://github.com/pytorch/pytorch/issues/102337#issuecomment-1634363356
    def data_ptr(self):
        # [FUNCTION DOC]
        # Returns the data pointer of the inner view. Outputs: int. Exceptions/Side effects: None.
        return self.sliced_flat_tensor.data_ptr()

    grad = property(_get_grad, _set_grad, _del_grad)


def get_sliced_tensor(param: NanotronParameter, start_offset: int, end_offset: int):
    # [FUNCTION DOC]
    # Behavior: wraps param's flat slice between offsets as a SlicedFlatTensor leaf sharing storage.
    # Inputs: param, start_offset, end_offset. Outputs: SlicedFlatTensor. Exceptions/Side effects: None.
    # This allows us to create a leaf tensor, despite sharing the underlying storage
    result = SlicedFlatTensor(data=param, start_offset=start_offset, end_offset=end_offset)
    return result


def find_optim_index_from_param_name(
    param_name: str,
    # NOTE: (pp_rank, dp_rank, tp_rank) or (pp_rank, tp_rank)
    ckp_sharded_optim_states: Union[Tuple[Tuple[int, int, int], torch.Tensor], Tuple[Tuple[int, int], torch.Tensor]],
    is_zero1: bool,
    pp_rank=0,
) -> int:
    # [FUNCTION DOC]
    # Behavior: strips "module." from the name, reads the "names" map of the shard at (pp_rank, 0, 0) for ZeRO-1 or
    # (pp_rank, 0) otherwise, and returns the first state index whose name matches.
    # Inputs: param_name (str), ckp_sharded_optim_states (dict keyed by rank tuples to loaded optimizer state dicts),
    # is_zero1 (bool), pp_rank (int, default 0).
    # Outputs: int index, or None if not found.
    # Exceptions: KeyError if the required shard key is missing. Side effects: None.
    param_name = param_name.replace("module.", "")
    # NOTE: since all shards have the same optim state names
    # so we take the first shard (except optionally the pp dimension)
    if is_zero1 is True:
        # NOTE: (pp_rank, dp_rank, tp_rank)
        OPTIM_STATE_INDEX_TO_PARAM_NAME = ckp_sharded_optim_states[(pp_rank, 0, 0)]["names"]
    else:
        # NOTE: (pp_rank, tp_rank)
        OPTIM_STATE_INDEX_TO_PARAM_NAME = ckp_sharded_optim_states[(pp_rank, 0)]["names"]

    return next((k for k, v in OPTIM_STATE_INDEX_TO_PARAM_NAME.items() if v == param_name), None)


def extract_parallel_ranks_from_shard_path(
    shard_path: Path, is_zero1: bool
) -> Union[Tuple[int, int, int], Tuple[int, int]]:
    # [FUNCTION DOC]
    # Behavior: regex-parses the file name: for ZeRO-1 pattern optimizer_pp-X-of-N_dp-Y-of-N_tp-Z-of-N.pt returning
    # (pp, dp, tp); otherwise pattern pp-X-of-N_tp-Z-of-N returning (pp, tp).
    # Inputs: shard_path (Path), is_zero1 (bool). Outputs: tuple of ints.
    # Exceptions: AttributeError if the pattern does not match. Side effects: None.
    """Extract parallel ranks from shard path

    For example, if the shard path is:
    + For ZeRO-1: /path/to/optimizer_pp-0-of-1_dp-0-of-2_tp-0-of-1.pt
    then the function will return (0, 0, 0) (pp_rank, dp_rank, tp_rank)

    For ZeRO-0: /path/to/optimizer_pp-0-of-1_tp-0-of-1.pt
    then the function will return (0, 0) (pp_rank, tp_rank)
    """
    if is_zero1 is True:
        # TODO(xrsrke): use the same pattern as weight checkpoints
        # in weight checkpoints, we do pp-rank-.... but here we only do pp-...
        # TODO(xrsrke): don't hardcode this
        pattern = r"optimizer_pp-(\d+)-of-\d+_dp-(\d+)-of-\d+_tp-(\d+)-of-\d+\.pt"
        match = re.search(pattern, str(shard_path))
        pp_rank, dp_rank, tp_rank = match.groups()
        return int(pp_rank), int(dp_rank), int(tp_rank)
    else:
        # NOTE: this is zero0 checkpoint
        pattern = r"pp-(\d+)-of-\d+_tp-(\d+)-of-\d+"
        match = re.search(pattern, str(shard_path))
        pp_rank, tp_rank = match.groups()
        return int(pp_rank), int(tp_rank)


def merge_dp_shard_in_zero1_optimizer(
    model: nn.Module,
    optimizer_config,
    shard_paths: List[Path],
    parallel_context: ParallelContext,
    map_location: Optional[str] = None,
) -> Dict[Tuple[int, int], Dict[str, torch.Tensor]]:  # (pp_rank, tp_rank): param_name -> optim_state
    # [FUNCTION DOC]
    # Behavior: asserts offsets are in the optimizer config; loads every shard with torch.load keyed by (pp, dp, tp).
    # For each (pp, tp) checkpoint pair (progress bar on world rank 0): for each model state_dict key (sorted), finds the
    # state index, and for each state name builds a zeroed float32 CUDA buffer of the unsharded flat size and copies each
    # DP shard into its offsets (offset dict keyed by string dp rank), also copying "step". Copies "names" and
    # "param_groups" from the dp 0 shard. Asserts the merged count and a key sanity check.
    # Inputs: model (nn.Module), optimizer_config (dict with "configs" and "parallelism"), shard_paths (list of Path),
    # parallel_context (ParallelContext), map_location (default None).
    # Outputs: dict (pp_rank, tp_rank) to merged optimizer state dict with flat unsharded state tensors.
    # Exceptions: AssertionError on missing offsets, wrong merged count, or key mismatch; KeyError on missing shards.
    # Side effects: file reads; CUDA allocation; progress bar output.
    # KNOWN ISSUE: find_optim_index_from_param_name is called without pp_rank, so names are always looked up in pp rank 0's shard; wrong when PP size is above 1.
    # KNOWN ISSUE: model.state_dict keys include params that are not in a given pp/tp shard (and buffers), producing None indices or KeyErrors.
    # KNOWN ISSUE: "step" is read from the loop variable of the last DP shard after it has been reassigned; works only because steps are equal.
    # KNOWN ISSUE: the sanity check subtracts original keys from merged keys, so it never detects keys missing from the merged dict (direction inverted relative to its message).
    # KNOWN ISSUE: merged buffers are always float32 and flat, ignoring the original dtype and shape.
    assert (
        optimizer_config["configs"]["param_name_to_dp_rank_offsets"] is not None
    ), "param_name_to_dp_rank_offsets is required"

    checkpoint_pp_size = optimizer_config["parallelism"]["pp_size"]
    checkpoint_tp_size = optimizer_config["parallelism"]["tp_size"]

    ckp_sharded_optim_states = {}
    for shard_path in shard_paths:
        pp_rank, dp_rank, tp_rank = extract_parallel_ranks_from_shard_path(shard_path, is_zero1=True)
        ckp_sharded_optim_states[(pp_rank, dp_rank, tp_rank)] = torch.load(shard_path, map_location=map_location)

    param_name_to_dp_rank_offsets = optimizer_config["configs"]["param_name_to_dp_rank_offsets"]
    optimizer_state_names = ckp_sharded_optim_states[(0, 0, 0)]["state"][0].keys()

    def get_numel_of_unsharded_dp_param(param_name):
        # [FUNCTION DOC]
        # HELPER: returns the max end offset across DP ranks for a param, i.e. its full flat numel.
        # Inputs: param_name. Outputs: int. Exceptions: KeyError if unknown. Side effects: None.
        dp_offsets = param_name_to_dp_rank_offsets[param_name]
        return max(int(value) for values in dp_offsets.values() for value in values)

    def assign_shard_to_buffer(buffer, offset, value):
        # [FUNCTION DOC]
        # HELPER: writes value into buffer between the integer-cast start and end offsets.
        # Inputs: buffer (1D tensor), offset (pair), value (tensor). Outputs: None. Side effects: mutates buffer.
        # Exceptions: shape mismatch errors from torch.
        offset_start, offset_end = map(int, offset)
        buffer[offset_start:offset_end] = value

    param_names = sorted(model.state_dict().keys(), key=lambda x: x)
    ckp_merged_dp_shards_optim_states = {}
    for pp_rank, tp_rank in tqdm(
        list(itertools.product(range(int(checkpoint_pp_size)), range(int(checkpoint_tp_size)))),
        disable=dist.get_rank(parallel_context.world_pg) != 0,
        desc="Merging ZeRO-1's shards across data parallel dimension",
    ):
        # NOTE: filter only the shards that correspond to the current pp_rank and tp_rank
        filtered_ckp_sharded_optim_states = {}
        for (pp, dp, tp), ckp_optim_state in ckp_sharded_optim_states.items():
            if pp == pp_rank and tp == tp_rank:
                filtered_ckp_sharded_optim_states[dp] = ckp_optim_state

        # NOTE: now merge the shards across data parallel dimension
        # for each parameter, we need to merge all shards across data parallel dimension
        merged_dp_shards_optim_states = {}

        merged_dp_shards_optim_states["state"] = {}

        for param_name in param_names:
            unshard_dp_size = get_numel_of_unsharded_dp_param(param_name)
            optim_state_index = find_optim_index_from_param_name(
                param_name=param_name,
                ckp_sharded_optim_states=ckp_sharded_optim_states,
                is_zero1=True,
            )
            merged_dp_shards_optim_states["state"][optim_state_index] = {}
            for state_name in optimizer_state_names:
                unsharded_dp_buffer = torch.zeros(unshard_dp_size, device="cuda")
                # NOTE: now merge all the params across data parallel dimension
                for dp_rank, ckp_optim_state in filtered_ckp_sharded_optim_states.items():
                    # NOTE: extract the optimizer state of the current parameter
                    ckp_optim_state = ckp_optim_state["state"][optim_state_index]
                    ckp_offset = param_name_to_dp_rank_offsets[param_name][str(dp_rank)]
                    assign_shard_to_buffer(unsharded_dp_buffer, ckp_offset, ckp_optim_state[state_name])

                # NOTE: in optimizer states, the "state" use an index to represent the parameter
                # not the parameter name
                merged_dp_shards_optim_states["state"][optim_state_index][state_name] = unsharded_dp_buffer
                # NOTE: each dp shard has the same step
                merged_dp_shards_optim_states["state"][optim_state_index]["step"] = ckp_optim_state["step"]

        ckp_merged_dp_shards_optim_states[(pp_rank, tp_rank)] = merged_dp_shards_optim_states
        # NOTE: each dp shard has the same names, and param_groups since it's the same tp shard
        # the 0 in (pp_rank, 0, tp_rank) is the dp_rank
        ckp_merged_dp_shards_optim_states[(pp_rank, tp_rank)]["names"] = ckp_sharded_optim_states[
            (pp_rank, 0, tp_rank)
        ]["names"]
        ckp_merged_dp_shards_optim_states[(pp_rank, tp_rank)]["param_groups"] = ckp_sharded_optim_states[
            (pp_rank, 0, tp_rank)
        ]["param_groups"]

    assert len(ckp_merged_dp_shards_optim_states) == int(checkpoint_pp_size) * int(
        checkpoint_tp_size
    ), f"Expect {int(checkpoint_pp_size) * int(checkpoint_tp_size)} merged dp shards, got {len(ckp_merged_dp_shards_optim_states)}"

    # NOTE: sanity check, make sure each merged checkpoint
    # has the same dict key as the original checkpoint
    for (pp_rank, tp_rank), ckp_optim_state in ckp_merged_dp_shards_optim_states.items():
        # NOTE: we remove the gradient_accumulator key from sanity check
        # because we don't merge gradient_accumulator states
        missing_keys = set(ckp_optim_state.keys()) - set(ckp_sharded_optim_states[(pp_rank, 0, tp_rank)].keys())
        assert (
            len(missing_keys - {"gradient_accumulator"}) == 0
        ), "Expected the merged dp shards to have the same keys as the original dp shards, but merged dp shard misses: {}".format(
            missing_keys
        )

    return ckp_merged_dp_shards_optim_states
