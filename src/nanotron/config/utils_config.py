# [MODULE DOC]
# Purpose: small helpers shared by the nanotron config dataclasses: enums for recompute granularity and init
# scaling, recursive dataclass serialisation, and string/object conversions for dtypes and pipeline engines.
# Main contents: RecomputeGranularity, InitScalingMethod, serialize, str_to_dtype and dtype_to_str lookup tables,
# cast_str_to_torch_dtype, cast_str_to_pipeline_engine, cast_pipeline_engine_to_str.
# Fits in nanotron: used when loading YAML configs into dataclasses and when dumping configs back to YAML/JSON.
# Import-time side effects: builds the two module-level dtype dictionaries.
from dataclasses import fields
from enum import Enum, auto
from pathlib import Path

import torch

from nanotron.generation.sampler import SamplerType
from nanotron.parallel.pipeline_parallel.engine import (
    AllForwardAllBackwardPipelineEngine,
    OneForwardOneBackwardPipelineEngine,
    PipelineEngine,
)
from nanotron.parallel.tensor_parallel.nn import TensorParallelLinearMode


class RecomputeGranularity(Enum):
    # [CLASS DOC]
    # Purpose: enum of activation recomputation granularity: SELECTIVE (only some ops) or FULL (whole layer).
    # Attributes: two auto-numbered members. No construction inputs beyond Enum semantics.
    SELECTIVE = auto()
    FULL = auto()


class InitScalingMethod(Enum):
    # [CLASS DOC]
    # Purpose: enum of how weight initialisation std is scaled: NONE, NUM_LAYERS (by total layer count),
    # LAYER_INDEX (by the layer position), MODEL_SCALE (by model size). Auto-numbered members.
    NONE = auto()
    NUM_LAYERS = auto()
    LAYER_INDEX = auto()
    MODEL_SCALE = auto()


def serialize(data) -> dict:
    # [FUNCTION DOC]
    # Summary: recursively converts a (possibly nested) dataclass into a plain dictionary suitable for YAML/JSON.
    # Behavior: None returns None; any non-dataclass value is returned unchanged. For each dataclass field: nested
    # dataclasses are serialised recursively; Path becomes its string; PipelineEngine becomes "afab"/"1f1b";
    # TensorParallelLinearMode, RecomputeGranularity, InitScalingMethod and SamplerType become their member name;
    # torch dtypes become their name via dtype_to_str; lists and tuples become lists with each element serialised;
    # an empty dictionary becomes None (to keep the datasets library happy); anything else is kept as is.
    # Non-empty dictionaries are not recursed into.
    # Inputs: data, a dataclass instance, None, or any value.
    # Outputs: dictionary of field name to serialised value, None, or the original value.
    # Exceptions: KeyError if a torch dtype is not present in dtype_to_str.
    # Side effects: None.
    """Recursively serialize a nested dataclass to a dict - do some type conversions along the way"""
    if data is None:
        return None

    if not hasattr(data, "__dataclass_fields__"):
        return data

    result = {}
    for field in fields(data):
        value = getattr(data, field.name)
        if hasattr(value, "__dataclass_fields__"):
            result[field.name] = serialize(value)
        elif isinstance(value, Path):
            result[field.name] = str(value)
        elif isinstance(value, PipelineEngine):
            result[field.name] = cast_pipeline_engine_to_str(value)
        elif isinstance(value, TensorParallelLinearMode):
            result[field.name] = value.name
        elif isinstance(value, RecomputeGranularity):
            result[field.name] = value.name
        elif isinstance(value, InitScalingMethod):
            result[field.name] = value.name
        elif isinstance(value, SamplerType):
            result[field.name] = value.name
        elif isinstance(value, torch.dtype):
            result[field.name] = dtype_to_str[value]
        elif isinstance(value, (list, tuple)):
            result[field.name] = [serialize(v) for v in value]
        elif isinstance(value, dict) and not value:
            result[field.name] = None  # So we can serialize empty dicts without issue with `datasets` in particular
        else:
            result[field.name] = value

    return result


str_to_dtype = {
    "float32": torch.float32,
    "float64": torch.float64,
    "complex64": torch.complex64,
    "complex128": torch.complex128,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "uint8": torch.uint8,
    "int8": torch.int8,
    "int16": torch.int16,
    "int32": torch.int32,
    "int64": torch.int64,
    "bool": torch.bool,
}

dtype_to_str = {
    torch.float32: "float32",
    torch.float64: "float64",
    torch.complex64: "complex64",
    torch.complex128: "complex128",
    torch.float16: "float16",
    torch.bfloat16: "bfloat16",
    torch.uint8: "uint8",
    torch.int8: "int8",
    torch.int16: "int16",
    torch.int32: "int32",
    torch.int64: "int64",
    torch.bool: "bool",
}


def cast_str_to_torch_dtype(str_dtype: str):
    # [FUNCTION DOC]
    # Summary: maps a dtype name such as "bfloat16" to the torch dtype using str_to_dtype.
    # Inputs: str_dtype, string.
    # Outputs: torch.dtype.
    # Exceptions: ValueError if the name is not in the table.
    # Side effects: None.
    if str_dtype in str_to_dtype:
        return str_to_dtype[str_dtype]
    else:
        raise ValueError(f"dtype should be a string selected in {str_to_dtype.keys()} and not {str_dtype}")


def cast_str_to_pipeline_engine(str_pp_engine: str) -> PipelineEngine:
    # [FUNCTION DOC]
    # Summary: builds a pipeline engine from its short name: "afab" gives AllForwardAllBackwardPipelineEngine,
    # "1f1b" gives OneForwardOneBackwardPipelineEngine.
    # Inputs: str_pp_engine, string.
    # Outputs: a new PipelineEngine instance.
    # Exceptions: ValueError for any other string.
    # Side effects: None.
    if str_pp_engine == "afab":
        return AllForwardAllBackwardPipelineEngine()
    elif str_pp_engine == "1f1b":
        return OneForwardOneBackwardPipelineEngine()
    else:
        raise ValueError(f"pp_engine should be a string selected in ['afab', '1f1b'] and not {str_pp_engine}")


def cast_pipeline_engine_to_str(pp_engine: PipelineEngine) -> str:
    # [FUNCTION DOC]
    # Summary: inverse of cast_str_to_pipeline_engine; returns "afab" or "1f1b" depending on the engine's class.
    # Inputs: pp_engine, a PipelineEngine instance.
    # Outputs: string.
    # Exceptions: ValueError for other engine types.
    # Side effects: None.
    if isinstance(pp_engine, AllForwardAllBackwardPipelineEngine):
        return "afab"
    elif isinstance(pp_engine, OneForwardOneBackwardPipelineEngine):
        return "1f1b"
    else:
        raise ValueError(
            f"pp_engine should be aan instance of AllForwardAllBackwardPipelineEngine or OneForwardOneBackwardPipelineEngine, not {type(pp_engine)}"
        )
