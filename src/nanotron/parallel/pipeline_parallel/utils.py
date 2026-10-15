# [MODULE DOC]
# Purpose: helpers to locate pipeline ranks in a pipeline-parallel model.
# Main contents: get_input_output_pp_ranks, get_pp_rank_of.
# Import-time side effects: imports nanotron.models (heavy import, possible circular import risk).
from nanotron.models import NanotronModel
from nanotron.parallel.pipeline_parallel.block import PipelineBlock
from torch import nn
from torch.nn.parallel import DistributedDataParallel


def get_input_output_pp_ranks(model: NanotronModel | DistributedDataParallel):
    # [FUNCTION DOC]
    # Behavior: unwraps DDP if needed and returns the model's input_pp_rank and output_pp_rank attributes.
    # Inputs: model (NanotronModel or DDP-wrapped). Outputs: tuple (input PP rank, output PP rank).
    # Exceptions: AttributeError if attributes missing. Side effects: None.
    if isinstance(model, DistributedDataParallel):
        input_pp_rank = model.module.input_pp_rank
        output_pp_rank = model.module.output_pp_rank
    else:
        input_pp_rank = model.input_pp_rank
        output_pp_rank = model.output_pp_rank
    return input_pp_rank, output_pp_rank


def get_pp_rank_of(target: str, module: nn.Module):
    # [FUNCTION DOC]
    # Behavior: if module is a PipelineBlock returns its rank. Otherwise walks the dotted target name attribute by
    # attribute, returning the rank of the first PipelineBlock encountered.
    # Inputs: target (dotted name of a submodule or parameter), module (root nn.Module).
    # Outputs: int PP rank.
    # Exceptions: AttributeError if a path element is missing or an intermediate is not a module; ValueError if no
    # PipelineBlock is on the path.
    # Side effects: None.
    # KNOWN ISSUE: a parameter as the final atom (not an nn.Module) raises AttributeError before the ValueError path, so the error message is misleading for params outside blocks.
    """Assuming a model with pipeline blocks, we want to know in which pp rank the module/parameter whose name is `target`"""
    if isinstance(module, PipelineBlock):
        return module.rank

    atoms = target.split(".")
    current_module = module
    for atom in atoms:
        if not hasattr(current_module, atom):
            raise AttributeError(f'{current_module._get_name()} has no attribute `"{atom}"`')

        current_module = getattr(current_module, atom)

        if isinstance(current_module, PipelineBlock):
            return current_module.rank

        if not isinstance(current_module, nn.Module):
            raise AttributeError(f'`"{atom}"` is not an nn.Module')

    raise ValueError(f'`"{target}" is not inside a PipelineBlock and thus does not have a pp_rank')
