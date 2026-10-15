# [PACKAGE DOC]
# Purpose: nanotron.logging package - public logging API.
# Exports: log level constants, CategoryFilter, LoggerWriter, LogItem, NewLineStreamHandler and the
# helper functions from base.py (get_logger, log_rank, warn_once, set_* verbosity/format helpers,
# human_format, log_memory, log_libraries_versions); TimerRecord, Timers, nanotron_timer from timers;
# LogMixin, LoggingCollectorMixin from logmixin.
# Import-time side effects: importing base.py configures the library root logger (see base.py).
# Export logging functionality from base.py
from nanotron.logging.base import (
    # Constants
    CRITICAL,
    DEBUG,
    ERROR,
    FATAL,
    INFO,
    NOTSET,
    WARNING,
    CategoryFilter,
    LoggerWriter,
    LogItem,
    # Classes
    NewLineStreamHandler,
    # Functions
    get_logger,
    get_verbosity,
    human_format,
    log_libraries_versions,
    log_memory,
    log_rank,
    set_formatter,
    set_logger_verbosity_format,
    set_ranks_logging_level,
    set_verbosity,
    warn_once,
)

# Export timer functionality
from nanotron.logging.timers import TimerRecord, Timers, nanotron_timer
from nanotron.logging.logmixin import LogMixin, LoggingCollectorMixin

__all__ = [
    "CRITICAL",
    "DEBUG",
    "ERROR",
    "FATAL",
    "INFO",
    "NOTSET",
    "WARNING",
    "CategoryFilter",
    "LoggerWriter",
    "LogItem",
    "NewLineStreamHandler",
    "get_logger",
    "get_verbosity",
    "human_format",
    "log_libraries_versions",
    "log_memory",
    "log_rank",
    "set_formatter",
    "set_logger_verbosity_format",
    "set_ranks_logging_level",
    "set_verbosity",
    "warn_once",
    "TimerRecord",
    "Timers",
    "nanotron_timer",
    "LogMixin",
    "LoggingCollectorMixin",
]
