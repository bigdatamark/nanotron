# coding=utf-8
# [MODULE DOC]
# Purpose: nanotron logging utilities (adapted from Optuna/Hugging Face). Provides library-scoped
# loggers that do not propagate to the root logger, rank-aware logging for distributed runs, global
# verbosity/formatter control, a colored rank-annotated formatter, and small formatting helpers.
# Main contents: log_levels (name to level mapping), NewLineStreamHandler, CategoryFilter,
# get_logger, get_verbosity, set_verbosity, set_formatter, log_rank, warn_once, human_format,
# log_memory, LogItem, LoggerWriter, set_logger_verbosity_format, set_ranks_logging_level,
# log_libraries_versions. Module globals: DEFAULT_HANDLER, DEFAULT_LOG_LEVEL (WARNING), LIBRARY_NAME
# ("nanotron"), LEVEL and HANDLER (the currently configured global level and handler, initially None).
# Import-time side effects: configures the "nanotron" root logger (adds DEFAULT_HANDLER, sets the
# level from NANOTRON_LOGGING_LEVEL or WARNING); imports nanotron.parallel.
# Copyright 2020 Optuna, Hugging Face
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Logging utilities. """
import logging
import os
import sys
from dataclasses import dataclass
from functools import lru_cache
from logging import (
    CRITICAL,
    DEBUG,
    ERROR,
    FATAL,
    INFO,
    NOTSET,
    WARNING,
    Formatter,
    Logger,
)
from typing import TYPE_CHECKING, List, Optional, Union

import torch
from torch import distributed as torch_dist

from nanotron import distributed as dist

if TYPE_CHECKING:
    from nanotron.config import LoggingArgs
from nanotron.parallel import ParallelContext

log_levels = {
    "debug": DEBUG,
    "info": INFO,
    "warning": WARNING,
    "error": ERROR,
    "critical": CRITICAL,
    "fatal": FATAL,
    "notset": NOTSET,
}


class NewLineStreamHandler(logging.StreamHandler):
    # [CLASS DOC]
    # Purpose: StreamHandler that emits each line of a multi-line message as its own record so the
    # formatter prefix appears on every line.
    """
    We want to apply formatter before each new line
    https://stackoverflow.com/a/38458877
    """

    def emit(self, record):
        # [FUNCTION DOC]
        # Behavior: splits record.msg on newlines and, for each line, sets record.msg to it and emits via the parent.
        # Inputs: record - LogRecord. Outputs: None.
        # Exceptions: AttributeError if record.msg is not a string. Side effects: writes to the stream; mutates record.msg (left as the last line).
        # KNOWN ISSUE: non-string messages (e.g. logging an exception object or a number) would crash because msg is split directly.
        lines = record.msg.split("\n")
        for line in lines:
            record.msg = line
            super().emit(record)


class CategoryFilter(logging.Filter):
    # [CLASS DOC]
    # Purpose: logging filter ensuring every record has a category attribute formatted for display.
    """Filter to add category field to log records."""

    def filter(self, record):
        # [FUNCTION DOC]
        # Behavior: sets category to empty string if missing; if present and non-empty, prefixes it with a pipe.
        # Always returns True (never drops records).
        # Inputs: record. Outputs: True. Exceptions: None. Side effects: mutates the record.
        # KNOWN ISSUE: applying the filter more than once to the same record adds the pipe prefix repeatedly.
        # Add category attribute if not present
        if not hasattr(record, "category"):
            record.category = ""
        elif record.category:
            # Format the category if it exists
            record.category = f"|{record.category}"
        return True


DEFAULT_HANDLER = NewLineStreamHandler()
DEFAULT_LOG_LEVEL = logging.WARNING
LIBRARY_NAME = __name__.split(".")[0]


def _get_default_logging_level():
    # [FUNCTION DOC]
    # HELPER: determines the default level for nanotron loggers.
    # Behavior: if NANOTRON_LOGGING_LEVEL is set and is a key of log_levels, returns that level; if set but
    # invalid, warns on the root logger; otherwise returns DEFAULT_LOG_LEVEL (WARNING).
    # Inputs: None. Outputs: int level. Exceptions: None. Side effects: reads environment; may log a warning.
    """
    If NANOTRON_LOGGING_LEVEL env var is set to one of the valid choices return that as the new default level. If it is
    not - fall back to ``_default_log_level``
    """
    env_level_str = os.getenv("NANOTRON_LOGGING_LEVEL", None)
    if env_level_str:
        if env_level_str in log_levels:
            return log_levels[env_level_str]
        else:
            logging.getLogger().warning(
                f"Unknown option NANOTRON_LOGGING_LEVEL={env_level_str}, "
                f"has to be one of: { ', '.join(log_levels.keys()) }"
            )
    return DEFAULT_LOG_LEVEL


def get_library_root_logger() -> Logger:
    # [FUNCTION DOC]
    # HELPER: returns the logger named LIBRARY_NAME via get_logger.
    # Inputs: None. Outputs: Logger. Exceptions/Side effects: as get_logger.
    return get_logger(LIBRARY_NAME)


def _configure_library_root_logger() -> None:
    # [FUNCTION DOC]
    # HELPER: called at import; adds DEFAULT_HANDLER to the library root logger and sets its default level.
    # Inputs/Outputs: None. Exceptions: None. Side effects: mutates logging configuration.
    library_root_logger = get_library_root_logger()
    library_root_logger.addHandler(DEFAULT_HANDLER)
    library_root_logger.setLevel(_get_default_logging_level())


def _reset_library_root_logger() -> None:
    # [FUNCTION DOC]
    # HELPER: sets the library root logger level to NOTSET.
    # Inputs/Outputs: None. Exceptions: None. Side effects: mutates logging configuration.
    library_root_logger = get_library_root_logger()
    library_root_logger.setLevel(logging.NOTSET)


def get_logger(name: Optional[str] = None, log_level: Optional[str] = None) -> Logger:
    # [FUNCTION DOC]
    # Behavior: returns the logger for name. If it already existed or name is None (root), it is returned
    # unchanged. For a newly created logger under the nanotron namespace: sets level to log_level (upper
    # cased) if given, else the global LEVEL if set, else the env/default level; if a global HANDLER was
    # set via set_formatter, replaces its handlers with it; disables propagation.
    # Inputs: name - logger name (default None for root); log_level - optional level name.
    # Outputs: Logger. Exceptions: ValueError from logging for invalid level names.
    # Side effects: creates/configures a logger in the global logging registry.
    """
    Return a logger with the specified name.
    """
    logger_already_exists = isinstance(logging.root.manager.loggerDict.get(name, None), Logger)
    logger = logging.getLogger(name)

    if logger_already_exists or name is None:
        # if name is None we return root logger
        return logger

    # If the logger is in a `nanotron` module then we remove the capability to propagate
    if LIBRARY_NAME == name.split(".", 1)[0]:
        if log_level is not None:
            logger.setLevel(log_level.upper())
        elif LEVEL is not None:
            logger.setLevel(LEVEL)
        else:
            logger.setLevel(_get_default_logging_level())
        if HANDLER is not None:
            logger.handlers.clear()
            logger.addHandler(HANDLER)

        logger.propagate = False

    return logger


def get_verbosity() -> int:
    # [FUNCTION DOC]
    # Behavior: returns the effective level of the nanotron root logger.
    # Inputs: None. Outputs: int. Exceptions/Side effects: None.
    """
    Return the current level for the Nanotron root logger as an int.
    Returns:
        :obj:`int`: The logging level.
    .. note::
        Nanotron has following logging levels:
        - 50: ``nanotron.logging.CRITICAL`` or ``nanotron.logging.FATAL``
        - 40: ``nanotron.logging.ERROR``
        - 30: ``nanotron.logging.WARNING`` or ``nanotron.logging.WARN``
        - 20: ``nanotron.logging.INFO``
        - 10: ``nanotron.logging.DEBUG``
    """

    return get_library_root_logger().getEffectiveLevel()


LEVEL = None


def set_verbosity(verbosity: int) -> None:
    # [FUNCTION DOC]
    # Behavior: sets the level of every existing logger named nanotron or starting with "nanotron." and of
    # all their handlers, then stores the level in the global LEVEL for future loggers.
    # Inputs: verbosity - int level. Outputs: None. Exceptions: None. Side effects: mutates loggers, handlers, global LEVEL.
    """
    Set the verbosity level for the all `nanotron` loggers.

    Args:
        verbosity (:obj:`int`):
            Logging level, e.g., one of:
            - ``nanotron.logging.CRITICAL`` or ``nanotron.logging.FATAL``
            - ``nanotron.logging.ERROR``
            - ``nanotron.logging.WARNING`` or ``nanotron.logging.WARN``
            - ``nanotron.logging.INFO``
            - ``nanotron.logging.DEBUG``
    """
    all_nanotron_loggers = {
        name: logger
        for name, logger in logging.root.manager.loggerDict.items()
        if isinstance(logger, Logger) and (name.startswith(f"{LIBRARY_NAME}.") or name == LIBRARY_NAME)
    }
    for name, logger in all_nanotron_loggers.items():
        logger.setLevel(verbosity)

        # We update all handles to be at the current verbosity as well.
        for handle in logger.handlers:
            handle.setLevel(verbosity)

    global LEVEL
    LEVEL = verbosity


HANDLER = None


def set_formatter(formatter: logging.Formatter) -> None:
    # [FUNCTION DOC]
    # Behavior: creates a NewLineStreamHandler on stdout with formatter and the current verbosity (its
    # flush is replaced by stderr's flush), makes it the only handler of every existing nanotron logger,
    # and stores it in global HANDLER for future loggers.
    # Inputs: formatter - logging.Formatter. Outputs: None. Exceptions: None.
    # Side effects: mutates loggers and global HANDLER.
    # KNOWN ISSUE: the handler writes to stdout but its flush is bound to stderr's flush, so stdout is not explicitly flushed.
    """
    Set a new custom formatter as the current handler.
    Note: it's important to first set level and then

    :param formatter:
    :return:
    """
    handler = NewLineStreamHandler(sys.stdout)
    handler.setFormatter(formatter)
    handler.setLevel(get_verbosity())
    handler.flush = sys.stderr.flush

    all_nanotron_loggers = {
        name: logger
        for name, logger in logging.root.manager.loggerDict.items()
        if isinstance(logger, Logger) and (name.startswith(f"{LIBRARY_NAME}.") or name == LIBRARY_NAME)
    }
    for name, logger in all_nanotron_loggers.items():
        # We keep only a single handler
        logger.handlers.clear()
        logger.addHandler(handler)

    global HANDLER
    HANDLER = handler


def log_rank(
    msg: str,
    logger: Logger,
    level: int,
    group: Optional[dist.ProcessGroup] = None,
    rank: Optional[int] = None,
    category: Optional[str] = None,
    is_separator: bool = False,
    main_rank_only: bool = False,
    **kwargs,
):
    # [FUNCTION DOC]
    # Behavior: defaults group to the world group. Adds category and/or separator flags into the
    # logging extra dictionary. If main_rank_only, forces rank to 0. Logs when rank is None (every rank
    # logs) or when the current rank in group equals rank; with is_separator, surrounds the message with
    # lines of 50 equal signs.
    # Inputs: msg - str; logger - Logger; level - int; group - optional ProcessGroup; rank - optional
    # target rank within group; category - optional str; is_separator - default False; main_rank_only -
    # default False; kwargs - forwarded to Logger.log.
    # Outputs: None. Exceptions: RuntimeError (via get_rank) if the process is not in group; errors if
    # torch.distributed is not initialized. Side effects: logging; mutates kwargs extra.
    """Log only if the current process is the rank specified."""
    # Use default group is group is not provided
    if group is None:
        group = torch_dist.distributed_c10d._get_default_group()

    # Add category to the extra kwargs
    if category is not None:
        kwargs["extra"] = kwargs.get("extra", {})
        kwargs["extra"]["category"] = category

    # Add separator to the extra kwargs
    if is_separator:
        kwargs["extra"] = kwargs.get("extra", {})
        kwargs["extra"]["separator"] = True

    if main_rank_only:
        rank = 0
    # rank is None means everyone logs
    if rank is None or dist.get_rank(group) == rank:
        if is_separator:
            logger.log(level, "=" * 50, **kwargs)
        logger.log(level, msg, **kwargs)
        if is_separator:
            logger.log(level, "=" * 50, **kwargs)


@lru_cache(maxsize=None)
def warn_once(
    msg: str, logger: Logger, group: Optional[dist.ProcessGroup] = None, rank: Optional[int] = None, **kwargs
):
    # [FUNCTION DOC]
    # Behavior: logs msg at WARNING via log_rank, memoized with an unbounded lru_cache so the same
    # arguments log only once per process.
    # Inputs: msg, logger, group, rank, kwargs - as log_rank (all must be hashable).
    # Outputs: None. Exceptions: TypeError if any argument is unhashable. Side effects: logging; cache growth.
    log_rank(msg=msg, logger=logger, level=logging.WARNING, group=group, rank=rank, **kwargs)


def human_format(num: float, billions: bool = False, divide_by_1024: bool = False) -> str:
    # [FUNCTION DOC]
    # Behavior: values with magnitude below 1 are formatted with 3 significant digits. Otherwise rounds to
    # 3 significant digits, repeatedly divides by 1000 (or 1024 when divide_by_1024) while magnitude is at
    # least 1000, up to suffix E, and returns the number with trailing zeros/dot stripped plus the suffix
    # from "", K, M, B, T, P, E.
    # Inputs: num - float; billions - unused; divide_by_1024 - default False.
    # Outputs: str. Exceptions/Side effects: None.
    # KNOWN ISSUE: the billions parameter is never used.
    # KNOWN ISSUE: with divide_by_1024 the loop condition still compares against 1000, so values between 1000 and 1023 become fractional (e.g. 0.98K).
    if abs(num) < 1:
        return "{:.3g}".format(num)
    SIZES = ["", "K", "M", "B", "T", "P", "E"]
    num = float("{:.3g}".format(num))
    magnitude = 0
    i = 0
    while abs(num) >= 1000 and i < len(SIZES) - 1:
        magnitude += 1
        num /= 1000.0 if not divide_by_1024 else 1024.0
        i += 1
    return "{}{}".format("{:f}".format(num).rstrip("0").rstrip("."), SIZES[magnitude])


def log_memory(logger: logging.Logger, msg: str = ""):
    # [FUNCTION DOC]
    # Behavior: logs on rank 0 (world) at INFO the current allocated, peak allocated and peak reserved
    # CUDA memory in MiB after msg, then resets the peak memory statistics (on every rank).
    # Inputs: logger; msg - default empty. Outputs: None. Exceptions: None explicitly raised.
    # Side effects: logging; resets CUDA peak memory stats.
    log_rank(
        f"{msg}\n"
        f" Memory usage: {torch.cuda.memory_allocated() / 1024**2:.2f}MiB."
        f" Peak allocated {torch.cuda.max_memory_allocated() / 1024**2:.2f}MiB."
        f" Peak reserved: {torch.cuda.max_memory_reserved() / 1024**2:.2f}MiB",
        logger=logger,
        level=logging.INFO,
        rank=0,
    )
    torch.cuda.reset_peak_memory_stats()


@dataclass
class LogItem:
    # [CLASS DOC]
    # Purpose: dataclass describing one metric to log: tag, scalar_value (number or string) and optional
    # log_format (a format spec or "human_format").
    tag: str
    scalar_value: Union[float, int, str]
    log_format: Optional[str] = None


@dataclass
class LoggerWriter:
    # [CLASS DOC]
    # Purpose: dataclass that formats training metrics into a single console line; global_step is the
    # total number of training steps shown in the line.
    global_step: int

    def add_scalar(self, tag: str, scalar_value: Union[float, int], log_format=None) -> str:
        # [FUNCTION DOC]
        # Behavior: returns "tag: value" where value is human_format output if log_format is "human_format",
        # formatted with log_format as a format spec if given, or plain otherwise.
        # Inputs: tag; scalar_value; log_format - default None. Outputs: str.
        # Exceptions: ValueError for an invalid format spec. Side effects: None.
        if log_format == "human_format":
            log_str = f"{tag}: {human_format(scalar_value)}"
        else:
            log_str = f"{tag}: {scalar_value:{log_format}}" if log_format is not None else f"{tag}: {scalar_value}"
        return log_str

    def add_scalars_from_list(self, log_entries: List[LogItem], iteration_step: int):
        # [FUNCTION DOC]
        # Behavior: builds "iteration: step / global_step" followed by each formatted LogItem, joined with
        # " | ", and logs it at INFO on every rank via log_rank.
        # Inputs: log_entries - list of LogItem; iteration_step - int. Outputs: None.
        # Exceptions: as add_scalar. Side effects: logging.
        log_strs = [f"iteration: {iteration_step} / {self.global_step}"]
        log_strs += [
            self.add_scalar(log_item.tag, log_item.scalar_value, log_item.log_format) for log_item in log_entries
        ]
        log_str = " | ".join(log_strs)
        log_rank(log_str, logger=get_logger(__name__), level=logging.INFO)


def set_logger_verbosity_format(logging_level: str, parallel_context: ParallelContext):
    # [FUNCTION DOC]
    # Behavior: builds a rank prefix listing EP, CP, DP, PP, TP ranks for each parallel dimension larger
    # than one, plus SLURMD_NODENAME if set. Defines SafeFormatter which colors the metadata by level and
    # includes the category and rank prefix. Sets the root logger level and adds a new stdout handler with
    # this formatter, then applies the level and formatter to all nanotron loggers via set_verbosity and
    # set_formatter.
    # Inputs: logging_level - level name key of log_levels; parallel_context - ParallelContext.
    # Outputs: None. Exceptions: KeyError for unknown level names.
    # Side effects: reads environment; mutates global logging configuration.
    # KNOWN ISSUE: each call adds another handler to the root logger, so repeated calls produce duplicate root output.
    # 1. Conditional rank display - only show ranks if their size is > 1
    node_name = os.environ.get("SLURMD_NODENAME")
    ranks = []

    if parallel_context.expert_parallel_size > 1:
        ranks.append(f"EP={dist.get_rank(parallel_context.ep_pg)}")
    if parallel_context.context_parallel_size > 1:
        ranks.append(f"CP={dist.get_rank(parallel_context.cp_pg)}")
    if parallel_context.data_parallel_size > 1:
        ranks.append(f"DP={dist.get_rank(parallel_context.dp_pg)}")
    if parallel_context.pipeline_parallel_size > 1:
        ranks.append(f"PP={dist.get_rank(parallel_context.pp_pg)}")
    if parallel_context.tensor_parallel_size > 1:
        ranks.append(f"TP={dist.get_rank(parallel_context.tp_pg)}")

    if node_name:
        ranks.append(node_name)

    # Join all ranks with separator
    ranks_str = "|".join(ranks)
    ranks_display = f"|{ranks_str}" if ranks_str else ""

    # Use a custom formatter class that handles missing fields
    class SafeFormatter(Formatter):
        # [CLASS DOC]
        # HELPER: formatter used by set_logger_verbosity_format.
        # Purpose: Formatter that tolerates records missing category and applies ANSI colors per level.
        def format(self, record):
            # [FUNCTION DOC]
            # Behavior: ensures record.category exists and is pipe-prefixed; bolds separator messages; picks a
            # color prefix (yellow warning, red error and above, dim green debug, dim otherwise); temporarily swaps
            # the format string to include colored timestamp, level, category and the rank prefix; formats; then
            # restores the original format string and message.
            # Inputs: record. Outputs: formatted str. Exceptions: None. Side effects: mutates record.category; temporarily mutates self._style.
            # Ensure required attributes exist before formatting
            if not hasattr(record, "category"):
                record.category = ""
            elif record.category and not record.category.startswith("|"):
                record.category = f"|{record.category}"

            # Store original message for restoration later
            original_msg = record.msg

            # Apply styling based on record properties
            is_separator = getattr(record, "separator", False)
            if is_separator:
                record.msg = f"\033[1m{record.msg}\033[0m"  # Bold for separators

            # Choose color prefix/suffix based on log level
            if record.levelno == logging.WARNING:
                prefix = "\033[1;33m"  # Bold yellow for warnings
            elif record.levelno >= logging.ERROR:
                prefix = "\033[1;31m"  # Bold red for errors and critical
            elif record.levelno == logging.DEBUG:
                prefix = "\033[2;3;32m"  # Dim and italic green for debug
            else:
                prefix = "\033[2;3m"  # Dim and italic for other levels

            suffix = "\033[0m"

            # Save the original format
            original_fmt = self._style._fmt

            # Use a more consistent format with prefix/suffix applied only to the metadata portion
            self._style._fmt = f"{prefix}%(asctime)s [%(levelname)s%(category)s{ranks_display}]{suffix}: %(message)s"

            # Format the record
            result = super().format(record)

            # Restore the original values
            self._style._fmt = original_fmt
            record.msg = original_msg

            return result

    # Create formatter with the safe handling
    formatter = SafeFormatter(
        fmt=f"\033[2;3m%(asctime)s [%(levelname)s%(category)s{ranks_display}]\033[0m: %(message)s",
        datefmt="%m/%d %H:%M:%S",
    )
    log_level = log_levels[logging_level]

    # main root logger
    root_logger = get_logger()
    root_logger.setLevel(log_level)
    handler = NewLineStreamHandler(sys.stdout)
    handler.setLevel(log_level)
    handler.setFormatter(formatter)
    root_logger.addHandler(handler)

    # Nanotron
    set_verbosity(log_level)
    set_formatter(formatter=formatter)


def set_ranks_logging_level(parallel_context: ParallelContext, logging_config: "LoggingArgs"):
    # [FUNCTION DOC]
    # Behavior: on world rank 0 applies logging_config.log_level, on other ranks log_level_replica, each
    # via set_logger_verbosity_format and only if not None.
    # Inputs: parallel_context; logging_config - LoggingArgs. Outputs: None. Exceptions: as callee.
    # Side effects: mutates logging configuration.
    if dist.get_rank(parallel_context.world_pg) == 0:
        if logging_config.log_level is not None:
            set_logger_verbosity_format(logging_config.log_level, parallel_context=parallel_context)
    else:
        if logging_config.log_level_replica is not None:
            set_logger_verbosity_format(logging_config.log_level_replica, parallel_context=parallel_context)


def log_libraries_versions(logger: logging.Logger):
    # [FUNCTION DOC]
    # Behavior: imports datasets, flash_attn, numpy, torch, transformers, nanotron; on world rank 0 logs a
    # separator header and the versions of each, then the output of torch's collect_env.
    # Inputs: logger. Outputs: None. Exceptions: ImportError if any library is missing.
    # Side effects: logging; collect_env runs subprocesses and prints to stdout.
    # KNOWN ISSUE: torch.utils.collect_env.main prints and returns None, so the logged line shows None; torch.utils.collect_env may also not be imported as an attribute.
    import datasets
    import flash_attn
    import numpy
    import torch
    import transformers

    import nanotron

    if dist.get_rank() == 0:
        log_rank("Libraries versions:", logger=logger, level=logging.INFO, rank=0, is_separator=True)
        log_rank(f"nanotron version: {nanotron.__version__}", logger=logger, level=logging.INFO, rank=0)
        log_rank(f"torch version: {torch.__version__}", logger=logger, level=logging.INFO, rank=0)
        log_rank(f"transformers version: {transformers.__version__}", logger=logger, level=logging.INFO, rank=0)
        log_rank(f"datasets version: {datasets.__version__}", logger=logger, level=logging.INFO, rank=0)
        log_rank(f"flash-attn version: {flash_attn.__version__}", logger=logger, level=logging.INFO, rank=0)
        log_rank(f"numpy version: {numpy.__version__}", logger=logger, level=logging.INFO, rank=0)
        log_rank(
            f"\ntorch.utils.collect_env: {torch.utils.collect_env.main()}", logger=logger, level=logging.INFO, rank=0
        )


_configure_library_root_logger()
