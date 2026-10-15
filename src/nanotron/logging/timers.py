# [MODULE DOC]
# Purpose: timing utilities for nanotron (CPU wall-clock timers and CUDA-event timers) with a global singleton.
# Main contents: TimerType enum, TimerRecord dataclass (one named timer usable as context manager),
# Timers singleton registry, and the module-level instance nanotron_timer.
# Import-time side effects: creates a module logger, reads the ENABLE_TIMERS environment variable
# (both for the Timers class flag and for the default of Timers.__call__'s enabled argument), and
# instantiates the nanotron_timer singleton. Exports: TimerType, TimerRecord, Timers, nanotron_timer.
import os
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Union

import torch

from nanotron import distributed as dist
from nanotron import logging

logger = logging.get_logger(__name__)


class TimerType(Enum):
    # [CLASS DOC]
    # Purpose: enum of timer backends. CPU uses wall-clock time; CUDA uses pairs of CUDA events recorded on the current stream.
    # Values are the strings cpu and cuda so it can be built from those strings.
    CPU = "cpu"  # Regular CPU timer (uses time.time())
    CUDA = "cuda"  # CUDA-aware timer (uses CUDA events)


@dataclass
class TimerRecord:
    # [CLASS DOC]
    # Purpose: state of one named timer accumulating any number of start/end intervals.
    # Key attributes: name (the name dummy disables the timer), timer_type (default CUDA), start_time/end_time
    # (CPU timestamps), running flag, call_count (completed intervals), cuda_sync (synchronize the device around
    # start/end), enabled flag, _cpu_total_time (accumulated CPU seconds), _cuda_events (list of start/end event
    # pairs, one per completed interval), _current_start_event (pending start event while running).
    # Invariants: running is True between start and end; for CUDA timers each completed interval adds exactly one event pair.
    # Supports use as a context manager. CUDA timers fall back to CPU permanently if CUDA is unavailable.
    """
    Records timing information for a single timer.

    By default, uses CUDA events for timing GPU operations, which provides more accurate
    measurements of GPU execution time without forcing CPU-GPU synchronization.

    For CPU-only operations, you can use CPU-based timing by specifying timer_type=TimerType.CPU.
    """

    name: str
    timer_type: TimerType = TimerType.CUDA
    start_time: float = 0.0
    end_time: float = 0.0
    running: bool = False
    call_count: int = 0
    cuda_sync: bool = False  # Option to add CUDA synchronization for more accurate timings
    enabled: bool = True  # Allow individual timer to be enabled/disabled

    # For CPU timers we still track total_time
    _cpu_total_time: float = 0.0

    # CUDA specific fields
    _cuda_events: List[tuple[torch.cuda.Event, torch.cuda.Event]] = field(default_factory=list)
    _current_start_event: Optional[torch.cuda.Event] = None

    def __enter__(self):
        # [FUNCTION DOC]
        # Behavior: call start and return self. Inputs: none. Outputs: this TimerRecord.
        # Exceptions: None explicitly raised. Side effects: same as start.
        """Context manager support: Start the timer when entering a context."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # [FUNCTION DOC]
        # Behavior: call end and return False so exceptions propagate.
        # Inputs: exception type, value, traceback (unused). Outputs: False.
        # Exceptions: None explicitly raised. Side effects: same as end.
        """Context manager support: End the timer when exiting a context."""
        self.end()
        return False  # Don't suppress exceptions

    def start(self) -> "TimerRecord":
        # [FUNCTION DOC]
        # Behavior: no-op if name is dummy or the timer is disabled. Warns if already running (and restarts).
        # For CUDA type: if CUDA available, optionally synchronize the device, then create and record a timing-enabled
        # start event; otherwise warn and switch the timer permanently to CPU and record the wall-clock time.
        # For CPU type: record wall-clock time. Finally mark running.
        # Inputs: none. Outputs: self (for chaining).
        # Exceptions: None explicitly raised.
        # Side effects: logs warnings, records CUDA events on current stream, may synchronize device, mutates fields.
        """Start the timer."""
        if self.name == "dummy" or not self.enabled:  # disabled
            return self

        if self.running:
            logger.warning(f"Timer '{self.name}' already running. Restarting.")

        if self.timer_type == TimerType.CUDA:
            if torch.cuda.is_available():
                # Synchronize before starting timing if requested
                if self.cuda_sync:
                    torch.cuda.synchronize()
                # Create a new start event - we'll create the end event when end() is called
                self._current_start_event = torch.cuda.Event(enable_timing=True)
                self._current_start_event.record()
            else:
                logger.warning("CUDA timer requested but CUDA is not available. Falling back to CPU timer.")
                self.timer_type = TimerType.CPU
                self.start_time = time.time()
        else:
            self.start_time = time.time()

        self.running = True
        return self

    def end(self) -> None:
        # [FUNCTION DOC]
        # Behavior: no-op if dummy or disabled; warn and return if not running. For CUDA with a pending start event:
        # optionally synchronize, record an end event, append the (start, end) pair and clear the pending start.
        # Otherwise for CUDA (no event) warn, switch to CPU, and add wall-clock difference to _cpu_total_time.
        # For CPU: record end time and accumulate the difference. Then increment call_count and clear running.
        # Elapsed time for CUDA is NOT computed here (deferred to avoid sync).
        # Inputs: none. Outputs: None. Exceptions: None explicitly raised.
        # Side effects: logging, CUDA event recording, optional device sync, mutates fields.
        # KNOWN ISSUE: in the CUDA fallback branch start_time may never have been set via the CPU path (it is 0.0 if start used CUDA events), so the accumulated CPU time can be meaningless.
        """End the timer, but don't compute elapsed time yet."""

        if self.name == "dummy" or not self.enabled:  # disabled
            return

        if not self.running:
            logger.warning(f"Timer '{self.name}' was not running. Ignoring end call.")
            return

        if self.timer_type == TimerType.CUDA:
            if torch.cuda.is_available() and self._current_start_event is not None:
                # Synchronize before ending timing if requested
                if self.cuda_sync:
                    torch.cuda.synchronize()
                # Create and record an end event
                end_event = torch.cuda.Event(enable_timing=True)
                end_event.record()

                # Store the start/end event pair for later querying
                self._cuda_events.append((self._current_start_event, end_event))
                self._current_start_event = None
            else:
                logger.warning("CUDA timer end called but CUDA events are not available.")
                self.timer_type = TimerType.CPU
                self.end_time = time.time()
                self._cpu_total_time += self.end_time - self.start_time
        else:
            self.end_time = time.time()
            self._cpu_total_time += self.end_time - self.start_time

        self.call_count += 1
        self.running = False

    def reset(self) -> None:
        # [FUNCTION DOC]
        # Behavior: zero timestamps, running flag, call count and CPU total; clear the event list and pending start event.
        # Inputs: none. Outputs: None. Exceptions: None. Side effects: mutates fields (timer_type and enabled are kept, so a CPU fallback persists).
        """Reset the timer."""
        self.start_time = 0.0
        self.end_time = 0.0
        self.running = False
        self.call_count = 0
        self._cpu_total_time = 0.0
        self._cuda_events = []
        self._current_start_event = None

    @property
    def elapsed(self) -> float:
        # [FUNCTION DOC]
        # Behavior (property): duration in seconds of the last interval, or of the current one if running.
        # Not running: CPU returns end minus start; CUDA returns 0.0 if no events, else synchronizes on the last end event
        # and returns its elapsed time converted from milliseconds to seconds.
        # Running: CUDA with pending start event records a temporary end event (optionally syncing the device first),
        # synchronizes on it and returns elapsed seconds; otherwise returns now minus start_time.
        # Inputs: none. Outputs: float seconds. Exceptions: None explicitly raised.
        # Side effects: CUDA event synchronization (blocks host).
        """Get elapsed time in seconds for the current timer."""
        if not self.running:
            if self.timer_type == TimerType.CPU:
                return self.end_time - self.start_time

            # For CUDA timers, we need to synchronize to get the last elapsed time
            if not self._cuda_events:
                return 0.0

            # Get the last event pair
            start_event, end_event = self._cuda_events[-1]
            end_event.synchronize()  # Make sure the event is complete
            return start_event.elapsed_time(end_event) / 1000.0  # Convert ms to sec

        # Timer is still running
        if self.timer_type == TimerType.CUDA:
            if torch.cuda.is_available() and self._current_start_event is not None:
                # Create a temporary end event to measure elapsed time so far
                if self.cuda_sync:
                    torch.cuda.synchronize()
                tmp_end_event = torch.cuda.Event(enable_timing=True)
                tmp_end_event.record()
                tmp_end_event.synchronize()
                return self._current_start_event.elapsed_time(tmp_end_event) / 1000.0
            else:
                return time.time() - self.start_time
        else:
            return time.time() - self.start_time

    @property
    def total_time(self) -> float:
        # [FUNCTION DOC]
        # Behavior (property): CPU timers return _cpu_total_time; CUDA timers synchronize every stored end event and sum
        # the elapsed times of all pairs, in seconds.
        # Inputs: none. Outputs: float seconds. Exceptions: None explicitly raised. Side effects: host blocks on CUDA events.
        """
        Get total time in seconds across all calls.
        Warning: For CUDA timers, this will synchronize all events!
        """
        if self.timer_type == TimerType.CPU:
            return self._cpu_total_time

        # For CUDA timers, we need to sum up all the event pairs
        total = 0.0
        for start_event, end_event in self._cuda_events:
            end_event.synchronize()  # Make sure the event is complete
            total += start_event.elapsed_time(end_event) / 1000.0  # Convert ms to sec
        return total

    @property
    def average_time(self) -> float:
        # [FUNCTION DOC]
        # Behavior (property): total_time divided by call_count, or 0.0 when no calls.
        # Inputs: none. Outputs: float seconds. Exceptions: None. Side effects: may synchronize CUDA events via total_time.
        """
        Get average time per call in seconds.
        Warning: For CUDA timers, this will synchronize all events!
        """
        if self.call_count == 0:
            return 0.0
        return self.total_time / self.call_count


class Timers:
    # [CLASS DOC]
    # Purpose: process-wide singleton registry of TimerRecord objects keyed by name; also usable as a decorator factory.
    # Key state: class attribute _instance (the singleton), class attribute _enabled (initialized from the ENABLE_TIMERS
    # environment variable being "1"), instance attribute _timers (dict name to TimerRecord).
    # Invariants: only one instance exists; a name maps to a single timer whose settings cannot change.
    # KNOWN ISSUE: the class-level _enabled flag (and enable/disable) only affects items(); it does not stop timers from starting or recording.
    """
    A collection of timers for tracking execution time in Nanotron.

    By default, timers use CUDA events for timing GPU operations, which provides several benefits:
    1. More accurate measurement of GPU execution time
    2. Reduced need for explicit CUDA synchronization
    3. Lower overhead compared to CPU-based timing with synchronization
    4. Better performance monitoring for distributed training

    For CPU-only operations, you can still use CPU-based timing by specifying timer_type=TimerType.CPU.
    """

    _instance = None
    _enabled = os.environ.get("ENABLE_TIMERS", "0") == "1"  # Add global enable/disable flag

    def __new__(cls):
        # [FUNCTION DOC]
        # Behavior: on first call create the instance and give it an empty _timers dict; always return the singleton.
        # Inputs: cls. Outputs: the singleton instance. Exceptions: None. Side effects: sets the class attribute _instance.
        if cls._instance is None:
            cls._instance = super(Timers, cls).__new__(cls)
            cls._instance._timers: Dict[str, TimerRecord] = {}
        return cls._instance

    @classmethod
    def enable(cls) -> None:
        # [FUNCTION DOC]
        # Behavior: classmethod setting the class-wide _enabled flag to True. Inputs: none. Outputs: None. Exceptions: None. Side effects: mutates class state.
        """Enable all timing operations."""
        cls._enabled = True

    @classmethod
    def disable(cls) -> None:
        # [FUNCTION DOC]
        # Behavior: classmethod setting the class-wide _enabled flag to False. Inputs: none. Outputs: None. Exceptions: None. Side effects: mutates class state.
        """Disable all timing operations."""
        cls._enabled = False

    @classmethod
    def is_enabled(cls) -> bool:
        # [FUNCTION DOC]
        # Behavior: classmethod returning the class-wide _enabled flag. Inputs: none. Outputs: bool. Exceptions: None. Side effects: None.
        """Check if timers are enabled."""
        return cls._enabled

    def __call__(
        self,
        name: str,
        timer_type: Union[TimerType, str] = TimerType.CUDA,
        cuda_sync: bool = False,
        enabled: bool = bool(int(os.environ.get("ENABLE_TIMERS", "0"))),
    ) -> TimerRecord:
        # [FUNCTION DOC]
        # Behavior: convert a string timer_type to TimerType. If name is a callable (bare decorator use), wrap it with a
        # timing decorator named after the function and return the wrapped function. Otherwise, if a timer with that
        # name exists, check its type, cuda_sync and enabled match the request and return it; else create, register and
        # return a new TimerRecord.
        # Inputs: name (str, or a function when used as bare decorator); timer_type (TimerType or cpu/cuda string, default CUDA);
        # cuda_sync (bool, default False); enabled (bool, default computed once at import from ENABLE_TIMERS).
        # Outputs: a TimerRecord, or the wrapped function in bare-decorator use.
        # Exceptions: ValueError if an existing timer has different settings or the timer_type string is invalid.
        # Side effects: registers a new timer in the registry.
        # KNOWN ISSUE: the final branch returning a decorator for the decorator-with-name form is unreachable (the preceding check on callable name is always true at that point), so decorating with a name string returns a TimerRecord, which is not a decorator, contrary to the docstring.
        # KNOWN ISSUE: the enabled default is frozen at import time, so changing ENABLE_TIMERS later has no effect.
        """Get or create a timer with the given name.

        Can be used as a decorator, context manager, or directly:
        - @nanotron_timer  # As decorator with default CUDA timing
        - @nanotron_timer("my_function")  # As decorator with custom name
        - @nanotron_timer(timer_type=TimerType.CPU)  # As decorator with CPU timing
        - with nanotron_timer("name"): ...  # As context manager
        - nanotron_timer("name").start(); ...; nanotron_timer("name").end()  # Direct use

        Args:
            name: Name of the timer
            timer_type: Type of timer, either TimerType.CUDA (default) or TimerType.CPU
                        (or 'cuda'/'cpu' strings)
            cuda_sync: Whether to perform torch.cuda.synchronize() for more accurate CUDA timing.
                       Default is False to avoid unnecessary synchronization overhead.
            enabled: Override default enabled setting from environment variable

        Raises:
            ValueError: If a timer with the same name already exists with different settings
        """
        if isinstance(timer_type, str):
            timer_type = TimerType(timer_type)

        if callable(name):
            # Being used as a decorator with specified or default settings
            func = name
            timer_name = func.__name__
            return self._create_timer_decorator(timer_name, timer_type, cuda_sync, enabled)(func)

        if name in self._timers:
            existing_timer = self._timers[name]
            if (
                existing_timer.timer_type != timer_type
                or existing_timer.cuda_sync != cuda_sync
                or existing_timer.enabled != enabled
            ):
                raise ValueError(
                    f"Timer '{name}' already exists with different settings.\n"
                    f"Existing: type={existing_timer.timer_type}, cuda_sync={existing_timer.cuda_sync}, enabled={existing_timer.enabled}\n"
                    f"New: type={timer_type}, cuda_sync={cuda_sync}, enabled={enabled}"
                )
            return existing_timer

        self._timers[name] = TimerRecord(name=name, timer_type=timer_type, cuda_sync=cuda_sync, enabled=enabled)

        # Check if we're being called as a decorator
        if not callable(name):
            timer_record = self._timers[name]
            # Return the timer which can be used directly or as a context manager
            return timer_record

        # If we get here, we're being called as @nanotron_timer("name", timer_type)
        return self._create_timer_decorator(name, timer_type, cuda_sync, enabled)

    def _create_timer_decorator(self, name, timer_type=TimerType.CUDA, cuda_sync=False, enabled=None):
        # [FUNCTION DOC]
        # HELPER: builds the decorator used when Timers is applied to a function.
        # Behavior: returns a decorator that wraps the function so each call runs inside the timer obtained from calling
        # this registry with the given name and settings.
        # Inputs: name (str), timer_type (default CUDA), cuda_sync (default False), enabled (default None).
        # Outputs: decorator function. Exceptions: None at creation. Side effects: None at creation.
        # KNOWN ISSUE: with the default enabled of None the inner call would create a timer whose enabled is None (falsy, so disabled) or raise ValueError on a mismatch with an existing timer.
        """Create a decorator that times the execution of a function."""

        def decorator(func):
            # [FUNCTION DOC]
            # HELPER: the decorator returned by _create_timer_decorator.
            # Behavior: return a functools-wrapped wrapper around func. Inputs: func. Outputs: wrapper function. Exceptions: None. Side effects: None.
            import functools

            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                # [FUNCTION DOC]
                # HELPER: timed wrapper around the decorated function.
                # Behavior: fetch or create the timer, enter it as a context manager, call func with all args and return its result.
                # Inputs: arbitrary args/kwargs. Outputs: func's return value.
                # Exceptions: propagates func's exceptions and the ValueError from mismatched timer settings.
                # Side effects: timer start/end (CUDA events, possible syncs).
                with self(name, timer_type, cuda_sync, enabled):
                    return func(*args, **kwargs)

            return wrapper

        return decorator

    def reset_all(self) -> None:
        # [FUNCTION DOC]
        # Behavior: reset every registered timer. Inputs: none. Outputs: None. Exceptions: None. Side effects: mutates all timers.
        """Reset all timers."""
        for timer in self._timers.values():
            timer.reset()

    def reset(self, name: str) -> None:
        # [FUNCTION DOC]
        # Behavior: reset the named timer if it exists, else do nothing. Inputs: name (str). Outputs: None. Exceptions: None. Side effects: mutates that timer.
        """Reset a specific timer."""
        if name in self._timers:
            self._timers[name].reset()

    def log(self, name: str, logger=None, rank: Optional[int] = 0, group=None) -> None:
        # [FUNCTION DOC]
        # Behavior: return silently if the timer is unknown. Default the logger to this module's logger. Compute this
        # process's rank in group (world if group is None); if rank is not None and differs, return. If the timer has at
        # least one call, log at INFO its total and average time in milliseconds and its call count.
        # Inputs: name (str); logger (optional logger); rank (int or None, default 0; None logs on all ranks); group (process group or None).
        # Outputs: None. Exceptions: None explicitly raised.
        # Side effects: logging; CUDA event synchronization through total_time/average_time.
        """Log a specific timer on the specified rank."""
        if name not in self._timers:
            return

        if logger is None:
            logger = logging.get_logger(__name__)

        world_rank = dist.get_rank() if group is None else dist.get_rank(group)
        if rank is not None and world_rank != rank:
            return

        timer = self._timers[name]
        if timer.call_count > 0:
            # This will trigger synchronization for CUDA timers
            avg_time = timer.average_time * 1000  # Convert to ms
            total_time = timer.total_time * 1000  # Convert to ms
            logger.info(
                f"Timer '{name}' ({timer.timer_type.value}): {total_time:.2f}ms total, "
                f"{avg_time:.2f}ms avg, {timer.call_count} calls"
            )

    def log_all(self, logger=None, rank: Optional[int] = 0, group=None) -> None:
        # [FUNCTION DOC]
        # Behavior: same rank filter as log; then, if any timers exist, log a header, one INFO line per timer with at least
        # one call (sorted by name, totals and averages in ms), and a footer.
        # Inputs: logger, rank (default 0), group, as in log. Outputs: None. Exceptions: None explicitly raised.
        # Side effects: logging; CUDA synchronization for every CUDA timer.
        """Log all timers on the specified rank."""
        if logger is None:
            logger = logging.get_logger(__name__)

        world_rank = dist.get_rank() if group is None else dist.get_rank(group)
        if rank is not None and world_rank != rank:
            return

        # Sort timers by name for consistent output
        sorted_timers = sorted(self._timers.items())

        if sorted_timers:
            logger.info("---- Timing Information ----")
            for name, timer in sorted_timers:
                if timer.call_count > 0:
                    # This will trigger synchronization for CUDA timers
                    avg_time = timer.average_time * 1000  # Convert to ms
                    total_time = timer.total_time * 1000  # Convert to ms
                    logger.info(
                        f"Timer '{name}' ({timer.timer_type.value}): {total_time:.2f}ms total, "
                        f"{avg_time:.2f}ms avg, {timer.call_count} calls"
                    )
            logger.info("----------------------------")

    def items(self):
        # [FUNCTION DOC]
        # Behavior: return an empty list when the class-wide flag is disabled, else the registry's items view.
        # Inputs: none. Outputs: list or dict items view of (name, TimerRecord). Exceptions: None. Side effects: None.
        if not self._enabled:
            return []
        return self._timers.items()


# Create a singleton instance
nanotron_timer = Timers()
