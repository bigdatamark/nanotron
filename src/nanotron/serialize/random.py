# [MODULE DOC]
# Purpose: Save and load each process's RandomStates in checkpoints.
# Main contents: save_random_states, load_random_states. Files are written under a "random"
# subfolder named by tp, dp and pp rank and size, using pickle via torch.save.
# Import-time side effects: None.
from pathlib import Path

import torch

from nanotron import distributed as dist
from nanotron.parallel import ParallelContext
from nanotron.random import RandomStates


def save_random_states(
    random_states: RandomStates,
    parallel_context: ParallelContext,
    root_folder: Path,
):
    # [FUNCTION DOC]
    # Behavior: every process builds its filename from its tp/dp/pp ranks and sizes, creates the folder
    # and torch-saves its RandomStates.
    # Inputs: random_states (RandomStates), parallel_context, root_folder (Path).
    # Outputs: None. Exceptions: I/O errors propagate. Side effects: file write.
    # KNOWN ISSUE: the filename omits context-parallel (and expert) rank, so ranks differing only in cp overwrite each other's file.
    """All processes save their own random state"""
    filename = (
        root_folder
        / "random"
        / f"tp-{dist.get_rank(parallel_context.tp_pg)}-of-{parallel_context.tp_pg.size()}_dp-{dist.get_rank(parallel_context.dp_pg)}-of-{parallel_context.dp_pg.size()}_pp-{dist.get_rank(parallel_context.pp_pg)}-of-{parallel_context.pp_pg.size()}.pt"
    )
    filename.parent.mkdir(exist_ok=True, parents=True)

    # TODO @thomasw21: That's annothing but this actually uses pickle, we might need to change that for something else
    torch.save(random_states, filename)


def load_random_states(parallel_context: ParallelContext, root_folder: Path):
    # [FUNCTION DOC]
    # Behavior: rebuilds the same per-process filename (assuming identical topology) and torch-loads it.
    # Inputs: parallel_context, root_folder. Outputs: the loaded RandomStates.
    # Exceptions: FileNotFoundError if topology differs; unpickling errors. Side effects: file read.
    # TODO @thomasw21: This basically assumes that we have exactly the same topology as the one we used when saving.
    filename = (
        root_folder
        / "random"
        / f"tp-{dist.get_rank(parallel_context.tp_pg)}-of-{parallel_context.tp_pg.size()}_dp-{dist.get_rank(parallel_context.dp_pg)}-of-{parallel_context.dp_pg.size()}_pp-{dist.get_rank(parallel_context.pp_pg)}-of-{parallel_context.pp_pg.size()}.pt"
    )

    # TODO @thomasw21: That's annothing but this actually uses pickle, we might need to change that for something else
    state = torch.load(filename)

    return state
