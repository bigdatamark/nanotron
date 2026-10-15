# [MODULE DOC]
# Purpose: defines the ParallelismArgs dataclass, the section of the nanotron training config that
# describes how the model is split across devices (data, pipeline, tensor, expert and context parallelism).
# Main contents: ParallelismArgs.
# Fits in nanotron: consumed by the config loader and by ParallelContext / model builders to size process groups
# and pick the pipeline engine and tensor-parallel linear mode.
# Import-time side effects: imports pipeline engine and tensor-parallel modules (no state changes).
# Exports: ParallelismArgs.
from dataclasses import dataclass
from typing import Optional

from nanotron.config.utils_config import (
    cast_str_to_pipeline_engine,
)
from nanotron.parallel.pipeline_parallel.engine import (
    AllForwardAllBackwardPipelineEngine,
    PipelineEngine,
)
from nanotron.parallel.tensor_parallel.nn import TensorParallelLinearMode


@dataclass
class ParallelismArgs:
    # [CLASS DOC]
    # Purpose: plain dataclass holding parallelism degrees and related options; normalises string values into objects.
    # Attributes:
    #   dp, pp, tp: integers, number of data-parallel replicas, pipeline stages and tensor-parallel shards (required).
    #   pp_engine: PipelineEngine instance or the string "afab"/"1f1b"; default None, becomes AllForwardAllBackward.
    #   tp_mode: TensorParallelLinearMode or its name as a string (case-insensitive); default None, becomes ALL_REDUCE.
    #     REDUCE_SCATTER enables sequence parallelism.
    #   tp_linear_async_communication: optional bool, default None, becomes False.
    #   recompute_layer: bool, default False; recompute each transformer layer in backward to save memory.
    #   tp_recompute_allgather: bool, default True; recompute the TP all-gather in backward instead of storing it.
    #   expert_parallel_size: int, default 1 (MoE only).
    #   context_parallel_size: int, default 1; number of ranks a sequence is split across.
    # Invariants: after construction pp_engine, tp_mode and tp_linear_async_communication are never None or strings.
    # KNOWN ISSUE: the docstring does not mention tp_recompute_allgather or context_parallel_size.
    """Arguments related to TP/PP/DP

    Args:
        dp: Number of DP replicas
        pp: Number of PP stages
        tp: Number of TP replicas
        expert_parallel_size: Number of expert parallel replicas (used only for MoEs)
        pp_engine: Pipeline engine to use between "1f1b" and "afab"
        tp_mode: TP mode to use between "all_reduce" and "reduce_scatter": all_reduce is normal, reduce_scatter activate sequence parallelism
        tp_linear_async_communication: Whether to use async communication in TP linear layers
        recompute_layer: Whether to recompute each Transformer layer to save memory.
    """

    dp: int
    pp: int
    tp: int
    pp_engine: Optional[PipelineEngine] = None
    tp_mode: Optional[TensorParallelLinearMode] = None
    tp_linear_async_communication: Optional[bool] = None
    recompute_layer: bool = False
    tp_recompute_allgather: bool = True

    expert_parallel_size: int = 1
    context_parallel_size: int = 1

    def __post_init__(self):
        # [FUNCTION DOC]
        # Summary: fills conservative defaults then converts string fields.
        # Behavior: if pp_engine is None, set an AllForwardAllBackwardPipelineEngine; if tp_mode is None, set ALL_REDUCE;
        # if tp_linear_async_communication is None, set False. Then if pp_engine is a string, convert with
        # cast_str_to_pipeline_engine; if tp_mode is a string, look it up by its upper-cased name in TensorParallelLinearMode.
        # Inputs: self only.
        # Outputs: None.
        # Exceptions: ValueError from cast_str_to_pipeline_engine for unknown engine names; KeyError for an unknown tp_mode name.
        # Side effects: mutates self fields.
        # Conservative defaults
        if self.pp_engine is None:
            self.pp_engine = AllForwardAllBackwardPipelineEngine()
        if self.tp_mode is None:
            self.tp_mode = TensorParallelLinearMode.ALL_REDUCE
        if self.tp_linear_async_communication is None:
            self.tp_linear_async_communication = False

        if isinstance(self.pp_engine, str):
            self.pp_engine = cast_str_to_pipeline_engine(self.pp_engine)
        if isinstance(self.tp_mode, str):
            self.tp_mode = TensorParallelLinearMode[self.tp_mode.upper()]
