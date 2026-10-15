# [MODULE DOC]
# Purpose: grab-bag of generic utilities used across nanotron (core library, not tests).
# Main contents: Singleton metaclass; ContextManagers (enter a list of context managers at once);
# main_rank_first and local_ranks_zero_first (barrier-based "rank zero goes first" context managers);
# checkpoint_method (decorator enabling activation checkpointing of a module method based on a boolean attribute);
# get_parameter_and_parent_module (resolve a dotted parameter path); get_untyped_storage and
# tensor_from_untyped_storage (version-portable storage helpers); find_free_port (pick a random bindable TCP port).
# Fit in nanotron: imported by models, parallel helpers, trainer, data loading and tests.
# Import-time side effects: None beyond importing torch, packaging and nanotron.distributed.
# Exports: all names above (no explicit export list).
import functools
import inspect
import os
import random
import socket
from contextlib import ExitStack, contextmanager
from typing import ContextManager, List, Optional

import torch
from packaging import version
from torch import nn
from torch.utils.checkpoint import checkpoint

from nanotron import distributed as dist


class Singleton(type):
    # [CLASS DOC]
    # Purpose: metaclass that makes every class using it a process-wide singleton.
    # Key state: class attribute _instances, a dictionary shared by all classes using this metaclass, mapping each class to its single instance.
    # Construction: used as metaclass of another class; no inputs of its own.
    # Inheritance role: subclass of type; overrides instance creation.
    # Invariant: at most one instance per class per process; later constructor arguments are ignored once the instance exists.
    """
    Singleton metaclass.
    Create objects using this class as the metaclass to enable singleton behaviour.
    For instance:
    ```
    class Logger(metaclass=Singleton):
      ...
    ```
    """

    _instances = {}

    def __call__(cls, *args, **kwargs):
        # [FUNCTION DOC]
        # Behavior: if the class has no cached instance yet, create one via the normal type call with the given arguments and cache it; then return the cached instance.
        # Inputs: cls, the class being instantiated; args and kwargs, constructor arguments (only used on first creation).
        # Outputs: the single instance of cls.
        # Exceptions: whatever the class constructor raises on first creation.
        # Side effects: mutates the shared _instances dictionary on first creation.
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class ContextManagers:
    # [CLASS DOC]
    # Purpose: enter a whole list of context managers as one, using a contextlib ExitStack (port of the transformers helper).
    # Key attributes: context_managers, the list given; stack, the ExitStack that holds the entered managers.
    # Construction inputs: context_managers, a list of context manager objects.
    # Invariant: one instance should be entered at most once at a time since it reuses a single ExitStack.
    """
    Wrapper for `contextlib.ExitStack` which enters a collection of context managers. Adaptation of `ContextManagers`
    in the `transformers` library.
    """

    def __init__(self, context_managers: List[ContextManager]):
        # [FUNCTION DOC]
        # Behavior: store the list and create a fresh ExitStack.
        # Inputs: context_managers, list of context managers.
        # Outputs: None.
        # Exceptions: None explicitly raised.
        # Side effects: sets self.context_managers and self.stack.
        self.context_managers = context_managers
        self.stack = ExitStack()

    def __enter__(self):
        # [FUNCTION DOC]
        # Behavior: enters each context manager in list order through the ExitStack.
        # Inputs: none besides self.
        # Outputs: None (the values produced by the inner managers are discarded).
        # Exceptions: propagates any exception raised while entering an inner manager.
        # Side effects: whatever the inner managers do on entry; registers them on self.stack.
        for context_manager in self.context_managers:
            self.stack.enter_context(context_manager)

    def __exit__(self, *args, **kwargs):
        # [FUNCTION DOC]
        # Behavior: delegates to the ExitStack exit, which exits all entered managers in reverse order.
        # Inputs: args and kwargs, the standard exception type/value/traceback triple.
        # Outputs: None (the ExitStack result is not returned, so exceptions are never suppressed by this wrapper).
        # Exceptions: propagates exceptions from inner managers' exits.
        # Side effects: exits the inner managers.
        # KNOWN ISSUE: the return value of the ExitStack exit is dropped, so an inner manager that wants to suppress an exception cannot do so.
        self.stack.__exit__(*args, **kwargs)

    def __repr__(self) -> str:
        # [FUNCTION DOC]
        # Behavior: builds a string of the class name followed by the list of qualified names of each manager's underlying generator function.
        # Inputs: none besides self.
        # Outputs: string.
        # Exceptions: AttributeError if a manager is not a generator-based context manager (has no gen attribute).
        # Side effects: None.
        return f"{self.__class__.__name__}({[context_manager.gen.__qualname__ for context_manager in self.context_managers]})"


@contextmanager
def main_rank_first(group: Optional[dist.ProcessGroup] = None):
    # [FUNCTION DOC]
    # Behavior: generator context manager. Rank 0 of the given group runs the body first; then all ranks meet at a barrier on the group; then the remaining ranks run the body. Typical use is letting one rank download or build a cache before others read it.
    # Inputs: group, optional process group (None means the default world group).
    # Outputs: yields nothing.
    # Exceptions: None explicitly raised; collective errors propagate.
    # Side effects: a barrier collective on group.
    # KNOWN ISSUE: if the body raises on rank 0, rank 0 never reaches the barrier and the other ranks hang.
    """Context manager that executes the code in the context with the rank zero of the group going first."""
    is_main = dist.get_rank(group) == 0
    if is_main:
        yield

    dist.barrier(group)

    if not is_main:
        yield


@contextmanager
def local_ranks_zero_first(group: Optional[dist.ProcessGroup] = None):
    # [FUNCTION DOC]
    # Behavior: same pattern as main_rank_first, but the ranks that go first are those whose LOCAL_RANK environment variable is 0 (default 0 if unset), i.e. one rank per node; then a barrier on the group; then the other ranks run the body.
    # Inputs: group, optional process group (None means world).
    # Outputs: yields nothing.
    # Exceptions: ValueError if LOCAL_RANK is not an integer string.
    # Side effects: reads the LOCAL_RANK environment variable; barrier collective on group.
    """Context manager that executes the code in the context with all the local rank zero of the group going first.
    Useful to run only once per node first (e.g. to create local files, etc)
    """
    is_main = int(os.environ.get("LOCAL_RANK", 0)) == 0
    if is_main:
        yield

    dist.barrier(group)

    if not is_main:
        yield


def checkpoint_method(attr_name: str):
    # [FUNCTION DOC]
    # Behavior: decorator factory. Returns a decorator that wraps a method so that, when the instance attribute named attr_name is truthy, the call is executed through torch activation checkpointing; otherwise the method runs normally.
    # Inputs: attr_name, string name of a boolean attribute on the instance (the first positional argument).
    # Outputs: the decorator function.
    # Exceptions: None at decoration time (see wrapper).
    # Side effects: None.
    """Decorator to checkpoint a method of a class."""

    def decorator(func):
        # [FUNCTION DOC]
        # HELPER: wraps the target function for checkpoint_method.
        # Behavior: returns wrapper, a functools-wrapped version of func.
        # Inputs: func, the method to wrap.
        # Outputs: the wrapper function.
        # Exceptions: None explicitly raised.
        # Side effects: None.
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # [FUNCTION DOC]
            # HELPER: runtime dispatch for checkpoint_method.
            # Behavior: takes self as the first positional argument and reads attr_name from it. If false, calls func with the original arguments. If true, builds a full positional argument list: starts with the given positional args, then walks the function signature parameters in definition order; any variadic positional or keyword parameter aborts; parameters already covered positionally are skipped; for the rest, uses the keyword value if given, otherwise the declared default. Then asserts the list length equals the number of signature parameters and calls torch.utils.checkpoint.checkpoint with func and the list (self included, so the module is stored in the checkpoint context).
            # Inputs: args and kwargs as passed to the method.
            # Outputs: whatever func returns.
            # Exceptions: NotImplementedError if func has variadic parameters (only on the checkpointed path); AssertionError if a parameter has no value and no default, or the counts mismatch; IndexError if called with no positional args.
            # Side effects: under checkpointing the forward runs without storing activations and is re-run during backward (RNG state is handled by torch checkpoint defaults).
            _self = args[0]
            checkpoint_activated = getattr(_self, attr_name)
            if checkpoint_activated:
                all_args = list(args)
                signature_params = inspect.signature(func).parameters
                # Parameters are ordered in the function definition order: https://docs.python.org/3/library/inspect.html#inspect.Signature.parameters
                for i, (arg_name, arg_value) in enumerate(signature_params.items()):
                    if arg_value.kind in [inspect.Parameter.VAR_KEYWORD, inspect.Parameter.VAR_POSITIONAL]:
                        raise NotImplementedError(
                            "Checkpointing of functions with *args or **kwargs is not supported."
                        )
                    if i < len(args):
                        continue
                    if arg_name not in kwargs:
                        assert (
                            arg_value.default is not inspect.Parameter.empty
                        ), f"Missing argument {arg_name} from {kwargs} for {func.__name__}"
                        all_args.append(arg_value.default)
                    else:
                        all_args.append(kwargs[arg_name])
                assert len(all_args) == len(signature_params), f"Missing arguments for {func.__name__}"
                # TODO @nouamanetazi: we pass `self`(which is module) to checkpoint, so it's stored in `ctx.inputs` whereas some other methods create a custom fwd and pass only tensors without `self`. Need to investigate which is better
                return checkpoint(func, *all_args)
            else:
                return func(*args, **kwargs)

        return wrapper

    return decorator


def get_parameter_and_parent_module(target: str, root_module: nn.Module):
    # [FUNCTION DOC]
    # Behavior: splits target at its last dot into a module path and a parameter name, fetches the submodule from root_module (empty path means root itself), verifies the attribute exists and is a torch Parameter.
    # Inputs: target, dotted parameter name such as "a.b.weight"; root_module, the module to resolve from.
    # Outputs: tuple of (parameter, owning module, parameter attribute name).
    # Exceptions: AttributeError if the submodule lacks the attribute, or the attribute is not an nn.Parameter; get_submodule may raise AttributeError for a bad path.
    # Side effects: None.
    module_path, _, param_name = target.rpartition(".")

    mod: torch.nn.Module = root_module.get_submodule(module_path)

    if not hasattr(mod, param_name):
        raise AttributeError(mod._get_name() + " has no attribute `" + param_name + "`")

    param: torch.nn.Parameter = getattr(mod, param_name)

    if not isinstance(param, torch.nn.Parameter):
        raise AttributeError("`" + param_name + "` is not an " "nn.Parameter")

    return param, mod, param_name


def get_untyped_storage(tensor: torch.Tensor) -> torch.UntypedStorage:
    # [FUNCTION DOC]
    # Behavior: returns the untyped storage of a tensor, using the torch 2.0+ API when available, otherwise the legacy typed storage converted to untyped.
    # Inputs: tensor, any torch tensor.
    # Outputs: torch UntypedStorage backing the tensor.
    # Exceptions: None explicitly raised.
    # Side effects: None.
    if version.parse(torch.__version__) >= version.parse("2.0"):
        return tensor.untyped_storage()
    else:
        return tensor.storage().untyped()


def tensor_from_untyped_storage(untyped_storage: torch.UntypedStorage, dtype: torch.dtype):
    # [FUNCTION DOC]
    # Behavior: creates an empty zero-dimensional tensor of the requested dtype on the storage's device, then rebinds it to the whole given storage (resulting tensor is 1-D, spanning the storage reinterpreted as dtype).
    # Inputs: untyped_storage, torch UntypedStorage; dtype, torch dtype to view it as.
    # Outputs: tensor sharing memory with the storage.
    # Exceptions: errors from torch if the storage size is not compatible with the dtype element size.
    # Side effects: None (memory is shared, not copied).
    # TODO @thomasw21: Figure out what's the best Pytorch way of building a tensor from a storage.
    device = untyped_storage.device
    tensor = torch.empty([], dtype=dtype, device=device)
    tensor.set_(source=untyped_storage)
    return tensor


def find_free_port(min_port: int = 2000, max_port: int = 65000) -> int:
    # [FUNCTION DOC]
    # Behavior: loops forever picking a random integer port in the inclusive range, trying to bind a TCP socket on localhost with address reuse enabled; returns the first port that binds (the socket is closed immediately).
    # Inputs: min_port, default 2000; max_port, default 65000.
    # Outputs: integer port number.
    # Exceptions: None explicitly raised (bind errors are swallowed and retried).
    # Side effects: briefly opens and closes sockets; uses the global Python random module state.
    # KNOWN ISSUE: the port is released before use, so another process can grab it (race); loops forever if no port is free.
    while True:
        port = random.randint(min_port, max_port)
        try:
            with socket.socket() as sock:
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                sock.bind(("localhost", port))
                return port
        except OSError:
            continue
