# [MODULE DOC]
# Purpose: NanotronParameter, a torch Parameter subclass carrying metadata about tying (shared weights across
# modules/ranks) and sharding (TP/expert slices of a larger tensor), plus metadata dataclasses.
# Main contents: SlicesPair, TiedInfo, ShardedInfo, NanotronParameter, sanity_check.
# Import-time side effects: creates a module logger.
import dataclasses
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple

import torch
from torch import nn

from nanotron import distributed as dist
from nanotron import logging

if TYPE_CHECKING:
    from nanotron.models import NanotronModel

logger = logging.get_logger(__name__)


@dataclasses.dataclass
class SlicesPair:
    # [CLASS DOC]
    # Purpose: dataclass pairing local slices of a shard with the global slices of the unsharded tensor they map
    # to, with string (de)serialization used in checkpoint metadata.
    # Attributes: local_slices, global_slices (tuples of slice objects, one per dimension).
    # String format: each slice "start,stop,step" (None as "None"), dims joined by "|", local and global joined by
    # "#", multiple pairs joined by ";".
    local_slices: Tuple[slice, ...]
    global_slices: Tuple[slice, ...]

    @staticmethod
    def slice_to_str(s: slice):
        # [FUNCTION DOC]
        # HELPER: serializes one slice to "start,stop,step". Inputs: s (slice). Outputs: str. Exceptions/Side effects: None.
        # e.g. slice(0, 10, 2) -> "0,10,2"
        # e.g. slice(None, None, None) -> "None,None,None"
        return ",".join(str(x) if x is not None else "None" for x in (s.start, s.stop, s.step))

    @staticmethod
    def str_to_slice(s: str):
        # [FUNCTION DOC]
        # HELPER: parses "start,stop,step" back into a slice. Inputs: s (str). Outputs: slice.
        # Exceptions: ValueError on non-integer fields. Side effects: None.
        return slice(*(int(x) if x != "None" else None for x in s.split(",")))

    def __str__(self):
        # [FUNCTION DOC]
        # Returns "local#global" serialization. Outputs: str. Exceptions/Side effects: None.
        # e.g. local_slices (slice(0, 10, 2), slice(None, None, None)) -> "0,10,2|None,None,None"
        local_slices_str = "|".join(map(self.slice_to_str, self.local_slices))
        # e.g. global_slices (slice(0, 20, 4), slice(None, None, None)) -> "0,20,4|None,None,None"
        global_slices_str = "|".join(map(self.slice_to_str, self.global_slices))
        # e.g. "0,10,2|None,None,None#0,20,4|None,None,None"
        return f"{local_slices_str}#{global_slices_str}"

    @classmethod
    def from_str(cls, string: str):
        # [FUNCTION DOC]
        # Parses a "local#global" string into a SlicesPair. Inputs: string. Outputs: SlicesPair.
        # Exceptions: ValueError on malformed input. Side effects: None.
        local_slices_str, global_slices_str = string.split("#")
        local_slices = tuple(map(cls.str_to_slice, local_slices_str.split("|")))
        global_slices = tuple(map(cls.str_to_slice, global_slices_str.split("|")))
        return cls(local_slices, global_slices)

    @classmethod
    def tuple_to_str(cls, pairs):
        # [FUNCTION DOC]
        # Joins the string forms of several pairs with ";". Inputs: pairs (iterable of SlicesPair). Outputs: str.
        # e.g. 2 SlicesPair, 1st SlicesPair local_slices "0,10,2|None,None,None" and global_slices "0,10,2|None,None,None"
        # 2nd SlicesPair local_slices "0,20,4|None,None,None" and global_slices "0,40,8|None,None,None"
        # -> "0,10,2|None,None,None#0,10,2|None,None,None;0,20,4|None,None,None#0,40,8|None,None,None"
        return ";".join(map(str, pairs))

    @classmethod
    def tuple_from_str(cls, string: str):
        # [FUNCTION DOC]
        # Splits on ";" and parses each pair. Inputs: string. Outputs: tuple of SlicesPair. Exceptions: ValueError.
        return tuple(map(cls.from_str, string.split(";")))


@dataclasses.dataclass
class TiedInfo:
    # [CLASS DOC]
    # Purpose: dataclass describing a tied parameter.
    # Attributes: name (path relative to root_module), root_module (module the name is relative to), global_ranks
    # (tuple of ranks sharing it), reduce_op (ReduceOp used to sum gradients, or None for no reduction).
    name: str
    # name must be defined starting from `root_module` (e.g. root_module.dense0.dense1.weight)
    root_module: nn.Module
    global_ranks: Tuple[int, ...]
    # None signifies that we do not reduce
    reduce_op: Optional[dist.ReduceOp]

    def get_full_name_from_model(self, model: nn.Module) -> str:
        # [FUNCTION DOC]
        # Behavior: builds a map of module id to "prefix." for all modules of model (root mapped to empty), then resolves
        # the full name.
        # Inputs: model (nn.Module). Outputs: str full parameter name.
        # Exceptions: KeyError if root_module is not inside model. Side effects: None.
        module_id_to_prefix = {id(module): f"{module_name}." for module_name, module in model.named_modules()}
        # Fix the root_model
        module_id_to_prefix[id(model)] = ""
        return self.get_full_name_from_module_id_to_prefix(module_id_to_prefix)

    def get_full_name_from_module_id_to_prefix(self, module_id_to_prefix: Dict[int, str]) -> str:
        # [FUNCTION DOC]
        # Returns prefix of root_module concatenated with name. Inputs: module_id_to_prefix (dict). Outputs: str.
        # Exceptions: KeyError if root_module absent. Side effects: None.
        return f"{module_id_to_prefix[id(self.root_module)]}{self.name}"  # this assumes root_module is part of module_id_to_prefix


@dataclasses.dataclass
class ShardedInfo:
    # [CLASS DOC]
    # Purpose: dataclass describing how a parameter is a shard of a larger tensor.
    # Attributes: global_ranks (ranks holding shards), local_global_slices_pairs (tuple of SlicesPair),
    # unsharded_shape (tuple of ints).
    global_ranks: Tuple[int, ...]
    # Info of to what slice of the unsharded tensor (global_slices) the current sharded tensor corresponds (local_slices)
    local_global_slices_pairs: Tuple[SlicesPair, ...]
    # The shape of the unsharded tensor
    unsharded_shape: Tuple[int, ...]

    def is_tp_sharded(self, parallel_context) -> bool:
        # [FUNCTION DOC]
        # True if all TP group ranks are among global_ranks. Inputs: parallel_context. Outputs: bool. Side effects: None.
        return set(dist.get_global_ranks(parallel_context.tp_pg)).issubset(set(self.global_ranks))

    def is_expert_sharded(self, parallel_context) -> bool:
        # [FUNCTION DOC]
        # True if all EP group ranks are among global_ranks. Inputs: parallel_context. Outputs: bool. Side effects: None.
        return set(dist.get_global_ranks(parallel_context.ep_pg)).issubset(set(self.global_ranks))

    def is_dp_sharded(self, parallel_context):
        # [FUNCTION DOC]
        # True if all DP group ranks are among global_ranks. Inputs: parallel_context. Outputs: bool. Side effects: None.
        return set(dist.get_global_ranks(parallel_context.dp_pg)).issubset(set(self.global_ranks))


class NanotronParameter(nn.Parameter):
    # [CLASS DOC]
    # Purpose: base parameter type for every nanotron model weight; stores a metadata dict in attribute
    # "__nanotron_metadata__" with optional "tied" (TiedInfo) and "sharded" (ShardedInfo) entries.
    # Inheritance: nn.Parameter.
    # Invariant: metadata keys are written at most once.
    """Base class for all parameters in Nanotronmodels

    A NanotronParameter can have specific properties:
     - sharded: the parameter is considered to be `sharded` across multiple devices
     - tied: the parameter is considered to be `tied` with other parameters. We sum gradients over those.

    .. note::
        Notes about tied weights:
        - Tied weights means weights that need to be synced only within the same DP rank, regardless if they are part of TP strategy or just shared weights between two layers.
        - Syncing tied weights usually require to sum gradients.
        - Some weights are synced without needing to reduce grads over ranks. They can be in the same device (ex: enc/dec embeds in the same PP stage) or they can be duplicated across TP and duplicate the workload across TP ranks (ex: LN using traditional TP)
        - Even if some weights don't need their grads to be reduced, it's still useful for them to be marked as tied. For example, current serialization format requires to mark them correctly.
    """

    NANOTRON_PARAMETER_METADATA_ATTRIBUTE_NAME = "__nanotron_metadata__"
    NANOTRON_PARAMETER_METADATA_TIED_KEY = "tied"
    NANOTRON_PARAMETER_METADATA_SHARDED_KEY = "sharded"

    def __new__(cls, tensor: torch.Tensor, requires_grad: bool = True):
        # [FUNCTION DOC]
        # Behavior: creates a Parameter over a detached view of tensor's data; if tensor is itself exactly a
        # NanotronParameter its metadata dict is shallow-copied, else an empty dict is attached.
        # Inputs: tensor (tensor), requires_grad (default True). Outputs: NanotronParameter sharing storage.
        # Exceptions: AssertionError if tensor is a subclass of NanotronParameter. Side effects: None.
        param = nn.Parameter.__new__(cls, data=tensor.data.detach(), requires_grad=requires_grad)

        if isinstance(tensor, NanotronParameter):
            # Check that we don't inherit a weird class
            # We copy in order not to make in-place operation
            assert type(tensor) == NanotronParameter
            setattr(
                param,
                cls.NANOTRON_PARAMETER_METADATA_ATTRIBUTE_NAME,
                getattr(tensor, cls.NANOTRON_PARAMETER_METADATA_ATTRIBUTE_NAME).copy(),
            )
        else:
            setattr(param, cls.NANOTRON_PARAMETER_METADATA_ATTRIBUTE_NAME, {})

        return param

    def _set_metadata(self, key: str, value: Any):
        # [FUNCTION DOC]
        # HELPER: writes a metadata entry, refusing to overwrite.
        # Inputs: key (str), value. Outputs: None. Exceptions: ValueError if key exists. Side effects: mutates metadata.
        metadata = getattr(self, self.NANOTRON_PARAMETER_METADATA_ATTRIBUTE_NAME)

        if key in metadata:
            raise ValueError(
                f"We shouldn't override previous metadata. Key to be overridden: {key}, current metadata: {metadata}"
            )
        else:
            metadata[key] = value

    def mark_as_tied(
        self,
        name: str,
        global_ranks: Tuple[int, ...],
        reduce_op: Optional[dist.ReduceOp],
        root_module: "NanotronModel",
    ):
        # [FUNCTION DOC]
        # Stores a TiedInfo under "tied". Inputs: name, global_ranks, reduce_op, root_module. Outputs: None.
        # Exceptions: ValueError if already tied. Side effects: mutates metadata.
        self._set_metadata(
            self.NANOTRON_PARAMETER_METADATA_TIED_KEY,
            TiedInfo(name=name, global_ranks=global_ranks, reduce_op=reduce_op, root_module=root_module),
        )

    def get_tied_info(self) -> TiedInfo:
        # [FUNCTION DOC]
        # Returns the TiedInfo. Outputs: TiedInfo. Exceptions: KeyError if not tied. Side effects: None.
        return getattr(self, self.NANOTRON_PARAMETER_METADATA_ATTRIBUTE_NAME)[
            self.NANOTRON_PARAMETER_METADATA_TIED_KEY
        ]

    @property
    def is_tied(self) -> bool:
        # [FUNCTION DOC]
        # Property: whether "tied" metadata exists. Outputs: bool. Exceptions/Side effects: None.
        return self.NANOTRON_PARAMETER_METADATA_TIED_KEY in getattr(
            self, self.NANOTRON_PARAMETER_METADATA_ATTRIBUTE_NAME
        )

    def mark_as_sharded(
        self,
        global_ranks: Tuple[int, ...],
        local_global_slices_pairs: Tuple[SlicesPair, ...],
        unsharded_shape: Tuple[int, ...],
    ):
        # [FUNCTION DOC]
        # Stores a ShardedInfo under "sharded". Inputs: global_ranks, local_global_slices_pairs, unsharded_shape.
        # Outputs: None. Exceptions: ValueError if already sharded. Side effects: mutates metadata.
        self._set_metadata(
            self.NANOTRON_PARAMETER_METADATA_SHARDED_KEY,
            ShardedInfo(
                global_ranks=global_ranks,
                local_global_slices_pairs=local_global_slices_pairs,
                unsharded_shape=unsharded_shape,
            ),
        )

    def get_sharded_info(self) -> ShardedInfo:
        # [FUNCTION DOC]
        # Returns the ShardedInfo. Outputs: ShardedInfo. Exceptions: KeyError if not sharded. Side effects: None.
        return getattr(self, self.NANOTRON_PARAMETER_METADATA_ATTRIBUTE_NAME)[
            self.NANOTRON_PARAMETER_METADATA_SHARDED_KEY
        ]

    @property
    def is_sharded(self) -> bool:
        # [FUNCTION DOC]
        # Property: whether "sharded" metadata exists. Outputs: bool. Exceptions/Side effects: None.
        return self.NANOTRON_PARAMETER_METADATA_SHARDED_KEY in getattr(
            self, self.NANOTRON_PARAMETER_METADATA_ATTRIBUTE_NAME
        )


def sanity_check(root_module: nn.Module):
    # [FUNCTION DOC]
    # Behavior: iterates over all named parameters of root_module and checks each is a NanotronParameter.
    # Inputs: root_module (nn.Module). Outputs: None.
    # Exceptions: ValueError naming the first offending parameter. Side effects: None.
    """Makes sure that the module is in Nanotronformat

    Format:
     - all parameters are `NanotronParameter`, this allows us to add metadata to a parameter.
    """
    for name, param in root_module.named_parameters():
        if not isinstance(param, NanotronParameter):
            raise ValueError(
                f"Nanotronrequires model to be in Nanotronformat, ie all parameters are required to be a NanotronParameter. {name} isn't."
            )
