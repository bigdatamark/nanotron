# [MODULE DOC]
# Purpose: core test utilities: GPU counting, environment mocking, deep dictionary comparison, enumeration of 3D parallel layouts, retry decorators for port collisions, and init_distributed which spawns one process per rank and builds a ParallelContext for a test function.
# Main contents: available_gpus, mock_os_environ, is_dict_equal, get_all_3d_configurations, rerun_if_address_is_in_use, rerun_on_exception, global_wrapper, init_distributed.
import contextlib
import os
import re
from inspect import signature
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch.cuda
import torch.multiprocessing as mp
from nanotron.parallel import ParallelContext
from packaging import version


def available_gpus():
    # [FUNCTION DOC]
    # HELPER: counts usable GPUs for test parametrization.
    # Behavior: returns 0 without CUDA; otherwise counts local CUDA devices whose name is not "NVIDIA DGX Display".
    # Inputs: none.
    # Outputs: integer GPU count (local node only).
    # Exceptions: None explicitly raised.
    # Side effects: initializes CUDA device properties queries.
    if not torch.cuda.is_available():
        return 0

    device_properties = [torch.cuda.get_device_properties(i) for i in range(torch.cuda.device_count())]

    # We filter out
    blacklisted_gpu_names = {"NVIDIA DGX Display"}
    device_properties = [property_ for property_ in device_properties if property_.name not in blacklisted_gpu_names]

    # TODO @thomasw21: Can we do this cross node
    return len(device_properties)


# from https://stackoverflow.com/a/34333710/9201239
@contextlib.contextmanager
def mock_os_environ(remove_keys: List[str] = None, update_key_values: Dict[str, Any] = None):
    # [FUNCTION DOC]
    # HELPER: temporary environment variable patching.
    # Behavior: asserts the keys to remove and to update are disjoint; records original values of affected keys that already exist; applies updates and removals; on exit re-applies the recorded original values.
    # Inputs: remove_keys, optional list of names to delete; update_key_values, optional dictionary of names to values.
    # Outputs: yields nothing.
    # Exceptions: AssertionError if the two key sets overlap.
    # Side effects: mutates os.environ in place.
    # KNOWN ISSUE: on exit, keys that were newly added (not previously present) are not removed, so added variables leak after the context.
    """
    Temporarily updates the ``os.environ`` dictionary in-place.
    The ``os.environ`` dictionary is updated in-place so that the modification is sure to work in all situations.
    Args:
      remove_keys: Environment variables to remove.
      update_key_values: Dictionary of environment variables and values to add/update.
    """
    env = os.environ
    update_key_values = update_key_values or {}
    remove_keys = remove_keys or []

    update_keys = set(update_key_values.keys())
    remove_keys = set(remove_keys)
    assert remove_keys.isdisjoint(update_keys)

    stomped = (update_keys | remove_keys) & set(env.keys())
    reverse_change = {
        # Environment variables and values to restore on exit.
        **{k: env[k] for k in update_keys & stomped},
        # Environment variables and values to remove on exit.
        **{k: env[k] for k in remove_keys & stomped},
    }

    try:
        env.update(update_key_values)
        for k in remove_keys:
            env.pop(k, None)
        yield
    finally:
        env.update(reverse_change)


def is_dict_equal(first: Dict, second: Dict, sub_paths: Optional[List[str]] = None) -> Tuple[bool, Optional[str]]:
    # [FUNCTION DOC]
    # HELPER: recursive equality check for state dictionaries.
    # Behavior: compares key sets; for each key recurses into nested dictionaries, compares tensors exactly (atol and rtol 0) via torch testing, and compares other values with inequality; returns at the first mismatch with a message containing the dotted path.
    # Inputs: first and second, dictionaries; sub_paths, optional list of path components used for messages.
    # Outputs: tuple of (boolean match, message string or None).
    # Exceptions: None explicitly raised (tensor assertion errors are caught and turned into the message).
    # Side effects: None.
    """Returns True or False if the dictionaries match, and an additional message when it's False"""
    if sub_paths is None:
        sub_paths = []

    first_keys = set(first.keys())
    second_keys = set(second.keys())
    if first_keys != second_keys:
        return False, f"Keys don't match in {'.'.join(sub_paths)}.\nCur: {first_keys}\nRef: {second_keys}"
    for key in first_keys:
        first_elt = first[key]
        second_elt = second[key]

        if isinstance(first_elt, dict):
            if not isinstance(second_elt, dict):
                return (
                    False,
                    f"Object types don't match in {'.'.join(sub_paths +  [str(key)])}.\nCur: {first_elt}\nRef: {second_elt}",
                )
            match, msg = is_dict_equal(first_elt, second_elt, sub_paths=sub_paths + [str(key)])
            if match is False:
                return False, msg
        elif isinstance(first_elt, torch.Tensor):
            if not isinstance(second_elt, torch.Tensor):
                return (
                    False,
                    f"Object types don't match in {'.'.join(sub_paths +  [str(key)])}.\nCur: {first_elt}\nRef: {second_elt}",
                )
            try:
                torch.testing.assert_close(
                    first_elt,
                    second_elt,
                    atol=0.0,
                    rtol=0.0,
                    msg=lambda msg: f"Tensor at {'.'.join(sub_paths + [str(key)])} don't match.\nCur: {first_elt}\nRef: {second_elt}\n{msg}",
                )
            except AssertionError as error:
                return False, error.args[0]
        else:
            if first_elt != second_elt:
                return (
                    False,
                    f"Objects at key {'.'.join(sub_paths + [str(key)])} don't match.\nCur: {first_elt}\nRef: {second_elt}",
                )

    return True, None


def get_all_3d_configurations(gpus: int) -> List[Tuple[int, int, int]]:
    # [FUNCTION DOC]
    # HELPER: enumerates parallel layouts for a GPU count.
    # Behavior: iterates over tp dividing gpus, dp dividing the remainder, and pp dividing what is left, keeping combinations whose product equals gpus; appends them as (pp, dp, tp).
    # Inputs: gpus, integer.
    # Outputs: list of integer triples.
    # Exceptions: None explicitly raised.
    # Side effects: None.
    # KNOWN ISSUE: tuples are ordered (pp, dp, tp) while callers unpack them as tp, dp, pp; since every permutation is enumerated the overall set is the same, but each tuple's labels are swapped relative to the loop variables.
    """Given a number of gpus, we want all 3d configurations possible such that pp * dp * tp = gpus"""
    result = []
    for tp in range(1, gpus + 1):
        if gpus % tp != 0:
            continue
        gpus_left_after_tp = gpus // tp
        for dp in range(1, gpus_left_after_tp + 1):
            if gpus_left_after_tp % dp != 0:
                continue
            gpus_left_after_dp = gpus_left_after_tp // dp
            for pp in range(1, gpus_left_after_dp + 1):
                if gpus_left_after_dp % pp != 0:
                    continue
                if tp * dp * pp == gpus:
                    result.append((pp, dp, tp))
    return result


def rerun_if_address_is_in_use(max_try: int = 500):
    # [FUNCTION DOC]
    # HELPER: decorator factory retrying tests that hit "Address already in use".
    # Behavior: asserts torch major version is at least 1; picks torch multiprocessing ProcessRaisedException (torch 1.8 or later) or generic Exception as the caught type, and returns rerun_on_exception with the pattern matching "Address already in use".
    # Inputs: max_try, default 500.
    # Outputs: a decorator.
    # Exceptions: AssertionError on very old torch.
    # Side effects: None.
    """
    This function reruns a wrapped function if "address already in use" occurs
    in testing spawned with torch.multiprocessing

    Credits: https://github.com/hpcaitech/ColossalAI/blob/adae123df3badfb15d044bd416f0cf29f250bc86/colossalai/testing/utils.py#L157

    Usage::

        @rerun_if_address_is_in_use()
        def test_something():
            ...

    """
    # check version
    torch_version = version.parse(torch.__version__)
    assert torch_version.major >= 1

    # only torch >= 1.8 has ProcessRaisedException
    if torch_version >= version.parse("1.8.0"):
        exception = torch.multiprocessing.ProcessRaisedException
    else:
        exception = Exception

    func_wrapper = rerun_on_exception(exception_type=exception, pattern=".*Address already in use.*", max_try=max_try)
    return func_wrapper


def rerun_on_exception(exception_type: Exception = Exception, pattern: str = None, max_try: int = 10) -> Callable:
    # [FUNCTION DOC]
    # HELPER: generic retry decorator factory.
    # Behavior: returns a decorator that reruns the wrapped function while it raises exception_type and some line of the message matches pattern (or pattern is None), up to max_try attempts; the wrapper copies the original signature so pytest parametrization works.
    # Inputs: exception_type, default Exception; pattern, optional regex matched at the start of each message line; max_try, default 10, None meaning unlimited.
    # Outputs: a decorator.
    # Exceptions: None at this level.
    # Side effects: None.
    # KNOWN ISSUE: docstring says the default max_try is 5 but the code default is 10.
    """
    A decorator on a function to re-run when an exception occurs.

    Credits: https://github.com/hpcaitech/ColossalAI/blob/adae123df3badfb15d044bd416f0cf29f250bc86/colossalai/testing/utils.py#L71

    Usage::

        # rerun for all kinds of exception
        @rerun_on_exception()
        def test_method():
            print('hey')
            raise RuntimeError('Address already in use')

        # rerun for RuntimeError only
        @rerun_on_exception(exception_type=RuntimeError)
        def test_method():
            print('hey')
            raise RuntimeError('Address already in use')

        # rerun for maximum 10 times if Runtime error occurs
        @rerun_on_exception(exception_type=RuntimeError, max_try=10)
        def test_method():
            print('hey')
            raise RuntimeError('Address already in use')

        # rerun for infinite times if Runtime error occurs
        @rerun_on_exception(exception_type=RuntimeError, max_try=None)
        def test_method():
            print('hey')
            raise RuntimeError('Address already in use')

        # rerun only the exception message is matched with pattern
        # for infinite times if Runtime error occurs
        @rerun_on_exception(exception_type=RuntimeError, pattern="^Address.*$")
        def test_method():
            print('hey')
            raise RuntimeError('Address already in use')

    Args:
        exception_type (Exception, Optional): The type of exception to detect for rerun
        pattern (str, Optional): The pattern to match the exception message.
            If the pattern is not None and matches the exception message,
            the exception will be detected for rerun
        max_try (int, Optional): Maximum reruns for this function. The default value is 5.
            If max_try is None, it will rerun forever if exception keeps occurring
    """

    def _match_lines(lines, pattern):
        # [FUNCTION DOC]
        # HELPER: pattern search for rerun_on_exception.
        # Behavior: returns True if the regex matches the start of any line.
        # Inputs: lines, list of strings; pattern, regex string.
        # Outputs: boolean.
        # Exceptions: re.error for invalid patterns.
        # Side effects: None.
        for line in lines:
            if re.match(pattern, line):
                return True
        return False

    def _wrapper(func):
        # [FUNCTION DOC]
        # HELPER: the decorator returned by rerun_on_exception.
        # Behavior: defines the retry loop function, sets its signature attribute to that of func, and returns it.
        # Inputs: func, the function to wrap.
        # Outputs: the retrying function.
        # Exceptions: None explicitly raised.
        # Side effects: sets a signature attribute on the new function.
        def _run_until_success(*args, **kwargs):
            # [FUNCTION DOC]
            # HELPER: retry loop.
            # Behavior: asserts max_try is None or an int; loops calling func, returning its result on success; on a matching exception with attempts remaining, prints a retry message and continues; otherwise prints a give-up message and re-raises.
            # Inputs: args and kwargs forwarded to func.
            # Outputs: func's return value; None if the loop ends without returning (only when max_try is 0 or less).
            # Exceptions: AssertionError for a bad max_try; the last caught exception when retries are exhausted or the pattern does not match; other exceptions propagate.
            # Side effects: prints to stdout.
            # KNOWN ISSUE: with max_try None, the comparison of try_count against None inside the except branch raises TypeError, so "infinite retries" does not work.
            try_count = 0
            assert max_try is None or isinstance(
                max_try, int
            ), f"Expected max_try to be None or int, but got {type(max_try)}"

            while max_try is None or try_count < max_try:
                try:
                    try_count += 1
                    ret = func(*args, **kwargs)
                    return ret
                except exception_type as e:
                    error_lines = str(e).split("\n")
                    if try_count < max_try and (pattern is None or _match_lines(error_lines, pattern)):

                        print("Exception is caught, retrying...")
                        # when pattern is not specified, we always skip the exception
                        # when pattern is specified, we only skip when pattern is matched
                        continue
                    else:
                        print("Maximum number of attempts is reached or pattern is not matched, no more retrying...")
                        raise e

        # Override signature
        # otherwise pytest.mark.parameterize will raise the following error:
        # function does not use argument xxx
        sig = signature(func)
        _run_until_success.__signature__ = sig

        return _run_until_success

    return _wrapper


def global_wrapper(rank, func, tp, pp, dp, port, kwargs):
    # [FUNCTION DOC]
    # HELPER: per-process entry point used by init_distributed through torch multiprocessing spawn.
    # Behavior: sets distributed environment variables for this rank, constructs a ParallelContext with the given dp/pp/tp sizes (which initializes the process group), then calls func with the context and kwargs.
    # Inputs: rank, spawned process index; func, the test body; tp, pp, dp, sizes; port, master port; kwargs, dictionary forwarded to func.
    # Outputs: None.
    # Exceptions: propagates anything raised by func or process group init.
    # Side effects: writes WORLD_SIZE, RANK, LOCAL_RANK, MASTER_ADDR (localhost) and MASTER_PORT environment variables; initializes torch distributed.
    def setup_dist_env(rank, world_size, port):
        # [FUNCTION DOC]
        # HELPER: environment setup for global_wrapper.
        # Behavior: sets the five distributed environment variables, using rank as both global and local rank (single node).
        # Inputs: rank; world_size; port.
        # Outputs: None.
        # Exceptions: None explicitly raised.
        # Side effects: mutates os.environ.
        os.environ["WORLD_SIZE"] = str(world_size)
        os.environ["RANK"] = str(rank)
        # NOTE: since we do unit tests in a
        # single node => this is fine!
        os.environ["LOCAL_RANK"] = str(rank)
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = str(port)

    world_size = tp * pp * dp
    setup_dist_env(rank, world_size, port)
    parallel_context = ParallelContext(data_parallel_size=dp, pipeline_parallel_size=pp, tensor_parallel_size=tp)
    func(parallel_context, **kwargs)


def init_distributed(tp: int, dp: int, pp: int):
    # [FUNCTION DOC]
    # HELPER: decorator factory that runs a test body on tp*dp*pp spawned processes.
    # Behavior: returns a decorator; the decorated function, called with keyword arguments only, finds a free port and spawns world size processes running global_wrapper.
    # Inputs: tp, dp, pp, integer parallel sizes.
    # Outputs: a decorator.
    # Exceptions: None at this level.
    # Side effects: None at this level.
    def _init_distributed(func):
        # [FUNCTION DOC]
        # HELPER: the decorator produced by init_distributed.
        # Behavior: returns wrapper bound to func.
        # Inputs: func, a function taking a ParallelContext first and keyword arguments.
        # Outputs: wrapper function.
        # Exceptions: None explicitly raised.
        # Side effects: None.
        def wrapper(**kwargs):
            # [FUNCTION DOC]
            # HELPER: launcher.
            # Behavior: computes world size, gets a free port via nanotron.utils find_free_port, and calls torch multiprocessing spawn with global_wrapper, passing func, sizes, port and kwargs; blocks until all processes finish.
            # Inputs: kwargs, keyword arguments forwarded to func (positional arguments are not supported).
            # Outputs: None.
            # Exceptions: ProcessRaisedException (from spawn) if any child fails.
            # Side effects: spawns processes; network ports.
            from nanotron.utils import find_free_port

            world_size = tp * pp * dp
            port = find_free_port()

            # Note that kwargs needs to be passed as part of args in a way that can be unpacked
            args = (func, tp, pp, dp, port, kwargs)
            mp.spawn(global_wrapper, args=args, nprocs=world_size)

        return wrapper

    return _init_distributed
