# [MODULE DOC]
# Purpose: local per-module key/value store used during inference (e.g. to keep the KV cache on the
# GPU between decoding steps without communicating it through the pipeline).
# Contents: Store, AttachableStore (mixin), attach_store (context manager).
# Import-time side effects: None.
import collections
import contextlib

from torch import nn


class Store(collections.defaultdict):
    # [CLASS DOC]
    # Purpose: defaultdict whose missing values are fresh empty dicts; keys are module ids, values are each
    # module's private state dictionary. Constructed with no arguments.
    """
    We use the store to locally store on gpu some states so that we don't have to communicate.
    This is useful at inference if we don't want to recompute kv_cache for example, or that we don't want to communicate it through the pipeline
    """

    def __init__(self):
        # [FUNCTION DOC]
        # Behavior: initializes the defaultdict with dict as the default factory.
        # Inputs/Outputs: None. Exceptions: None. Side effects: None.
        super().__init__(dict)

    def flush(self):
        # [FUNCTION DOC]
        # Behavior: deletes every key, emptying the store.
        # Inputs/Outputs: None. Exceptions: None. Side effects: mutates self.
        for key in list(self.keys()):
            del self[key]


class AttachableStore:
    # [CLASS DOC]
    # Purpose: mixin for modules that want a local store. Adds a private _store attribute while attached;
    # each instance sees its own sub-dictionary keyed by its id.
    def _attach_store(self, store: Store):
        # [FUNCTION DOC]
        # HELPER: used by attach_store to bind a Store.
        # Inputs: store - Store. Outputs: None.
        # Exceptions: AssertionError if a store is already attached. Side effects: sets self._store.
        assert not hasattr(self, "_store"), "You can't assign a store when there's already one attached"
        self._store = store

    def _detach_store(self):
        # [FUNCTION DOC]
        # HELPER: used by attach_store to unbind the Store.
        # Inputs/Outputs: None. Exceptions: AttributeError if no store attached. Side effects: deletes self._store.
        delattr(self, "_store")

    def get_local_store(self):
        # [FUNCTION DOC]
        # Behavior: if a store is attached, returns this instance's sub-dictionary (created on first access);
        # for nn.Module instances it first asserts the module is in eval mode. Returns None when no store.
        # Inputs: None. Outputs: dict or None.
        # Exceptions: AssertionError if the module is in training mode. Side effects: may create an empty entry in the store.
        if hasattr(self, "_store"):
            if isinstance(self, nn.Module):
                assert self.training is False, "Store is used only in evaluation mode"
            return self._store[id(self)]
        else:
            return None


@contextlib.contextmanager
def attach_store(model: nn.Module, store: Store):
    # [FUNCTION DOC]
    # Behavior: context manager; attaches store to every submodule of model that is an AttachableStore,
    # yields, and on exit (even on error) detaches it from all of them.
    # Inputs: model - nn.Module; store - Store. Outputs: yields None.
    # Exceptions: AssertionError if a module already has a store. Side effects: mutates modules temporarily.
    list_module_containing_store = []
    for module in model.modules():
        if not isinstance(module, AttachableStore):
            continue
        module._attach_store(store)
        list_module_containing_store.append(module)

    try:
        yield
    finally:
        for module in list_module_containing_store:
            module._detach_store()
