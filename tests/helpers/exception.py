# [MODULE DOC]
# Purpose: test helpers asserting that a code block raises a given exception (optionally on all but one rank) and enforcing a wall-clock timeout via SIGALRM.
# Main contents: assert_fail_with, assert_fail_except_rank_with, timeout_after.
import contextlib
import signal
from typing import Optional

from nanotron import distributed as dist


@contextlib.contextmanager
def assert_fail_with(exception_class, error_msg: Optional[str] = None):
    # [FUNCTION DOC]
    # HELPER: context manager asserting an exception is raised.
    # Behavior: runs the body; if it raises exception_class, succeeds when error_msg is None or exactly equals the message, else fails; any other exception type fails; no exception fails.
    # Inputs: exception_class, the expected exception type; error_msg, optional exact message.
    # Outputs: yields nothing.
    # Exceptions: AssertionError on wrong message, wrong type, or no exception.
    # Side effects: None.
    try:
        yield
    except exception_class as e:
        if error_msg is None:
            return
        if error_msg == str(e):
            return
        else:
            raise AssertionError(f'Expected message to be "{error_msg}", but got "{str(e)}" instead.')
    except Exception as e:
        raise AssertionError(f"Expected {exception_class} to be raised, but got: {type(e)} instead:\n{e}")
    raise AssertionError(f"Expected {exception_class} to be raised, but no exception was raised.")


@contextlib.contextmanager
def assert_fail_except_rank_with(
    exception_class, rank_exception: int, pg: dist.ProcessGroup, error_msg: Optional[str] = None
):
    # [FUNCTION DOC]
    # HELPER: distributed variant of assert_fail_with.
    # Behavior: every rank except rank_exception (rank within pg) must raise exception_class (with optional exact message); rank_exception must not raise it. Any other exception type fails on any rank.
    # Inputs: exception_class; rank_exception, rank in pg that must not raise; pg, process group; error_msg, optional exact message.
    # Outputs: yields nothing.
    # Exceptions: AssertionError on violated expectations.
    # Side effects: None (only queries the local rank).
    try:
        yield
    except exception_class as e:
        if rank_exception == dist.get_rank(pg):
            raise AssertionError(f"Expected rank {rank_exception} to not raise {exception_class}.")
        else:
            if error_msg is None:
                return
            if error_msg == str(e):
                return
            else:
                raise AssertionError(f'Expected message to be "{error_msg}", but got "{str(e)}" instead.')

    except Exception as e:
        raise AssertionError(f"Expected {exception_class} to be raised, but got: {type(e)} instead:\n{e}")
    if dist.get_rank(pg) != rank_exception:
        raise AssertionError(f"Expected {exception_class} to be raised, but no exception was raised.")


@contextlib.contextmanager
def timeout_after(ms=500):
    # [FUNCTION DOC]
    # HELPER: context manager limiting body run time.
    # Behavior: installs a SIGALRM handler, arms the real-time interval timer for ms milliseconds, runs the body, and always disarms the alarm afterwards.
    # Inputs: ms, timeout in milliseconds, default 500.
    # Outputs: yields nothing.
    # Exceptions: TimeoutError raised from the signal handler if the body exceeds the timeout.
    # Side effects: replaces the process SIGALRM handler (not restored); main thread only, Unix only.
    """Timeout context manager."""

    def signal_handler(signum, frame):
        # [FUNCTION DOC]
        # HELPER: SIGALRM handler for timeout_after.
        # Behavior: raises a TimeoutError mentioning ms.
        # Inputs: signum and frame, standard signal handler arguments (unused).
        # Outputs: never returns.
        # Exceptions: TimeoutError always.
        # Side effects: None.
        raise TimeoutError(f"Timed out after {ms} ms.")

    signal.signal(signal.SIGALRM, signal_handler)
    signal.setitimer(signal.ITIMER_REAL, ms / 1000)
    try:
        yield
    finally:
        signal.alarm(0)
