# [MODULE DOC]
# Purpose: lightweight mixins that let model submodules record named tensors during a forward pass
# ("TBI logs") and let the top-level model gather and clear them afterwards.
# Main contents: LogMixin (per-module log buffer) and LoggingCollectorMixin (model-level collector
# that walks all submodules). NanotronModel inherits LoggingCollectorMixin; attention/MLP modules in
# model files (e.g. qwen) inherit LogMixin and call tbi_logger.
# Import-time side effects: none. Exports: LogMixin, LoggingCollectorMixin.
import torch
from torch import nn
from typing import Dict, List, Any

class LogMixin:
    # [CLASS DOC]
    # Purpose: cooperative mixin giving a module a private list of log entries.
    # State: _tbi_logs, a list of dictionaries mapping string names to tensors (stored by reference).
    # Construction: forwards all args to the next class in the MRO, then creates the empty list.
    # Inheritance role: mix into an nn.Module subclass; must appear in the MRO so its __init__ runs.
    def __init__(self, *args, **kwargs):
        # [FUNCTION DOC]
        # Behavior: call the next initializer in the MRO with all positional and keyword args, then set _tbi_logs to an empty list.
        # Inputs: arbitrary args/kwargs passed through. Outputs: None.
        # Exceptions: whatever the parent initializer raises. Side effects: creates the _tbi_logs attribute on self.
        super().__init__(*args, **kwargs)
        self._tbi_logs: List[Dict[str, torch.Tensor]] = []

    def tbi_logger(self, log_data: Dict[str, torch.Tensor]):
        # [FUNCTION DOC]
        # Behavior: append the given dictionary (not a copy, tensors not cloned) to _tbi_logs.
        # Inputs: log_data, dictionary mapping metric name to tensor.
        # Outputs: None. Exceptions: None explicitly raised.
        # Side effects: mutates self._tbi_logs; keeps tensors alive until cleared.
        """
        Logs a dictionary of named tensors.
        Tensors are stored by reference (i.e., on their original device, e.g., CUDA).
        If you need a snapshot at the exact moment of logging and the tensor might be
        modified in-place later, consider storing {key: tensor.clone() for key, tensor in log_data.items()}.
        """
        # Example check (optional):
        # for tensor in log_data.values():
        #     if not tensor.is_cuda:
        #         # Or raise an error, or move to CUDA, depending on desired behavior
        #         print(f"Warning: Tensor {list(log_data.keys())[list(log_data.values()).index(tensor)]} is not on CUDA.")
        self._tbi_logs.append(log_data)

    def _get_internal_logs(self) -> List[Dict[str, torch.Tensor]]:
        # [FUNCTION DOC]
        # HELPER: accessor used by LoggingCollectorMixin.get_tbi_logs.
        # Behavior: return the internal list itself (not a copy).
        # Inputs: none. Outputs: list of dictionaries of tensors. Exceptions: AttributeError if __init__ never ran. Side effects: None.
        """
        Retrieves the logs stored by this module instance.
        """
        return self._tbi_logs

    def _clear_internal_logs(self):
        # [FUNCTION DOC]
        # HELPER: used by LoggingCollectorMixin.clear_all_tbi_logs.
        # Behavior: rebind _tbi_logs to a new empty list. Inputs: none. Outputs: None.
        # Exceptions: None explicitly raised. Side effects: drops references to logged tensors.
        """
        Clears the logs stored by this module instance.
        Important for managing memory, should be called after logs are processed.
        """
        self._tbi_logs = []


class LoggingCollectorMixin:
    # [CLASS DOC]
    # Purpose: mixin for the top-level nn.Module model that collects and clears TBI logs from every LogMixin submodule.
    # State: none of its own; relies on the host class providing named_modules and modules.
    # Inheritance role: mixed into NanotronModel alongside nn.Module.
    """
    A mixin class for nn.Module-based models to collect logs from submodules
    that use LogMixin.
    The class this is mixed into must be an nn.Module or its subclass.
    """
    # No __init__ is strictly necessary here if the mixin itself doesn't have
    # its own state to initialize. The methods operate on `self` which will be
    # an instance of the class it's mixed into (e.g., Qwen2ForTraining).
    # If an __init__ were added, it should also call super().__init__(*args, **kwargs).

    def get_tbi_logs(self, non_blocking: bool = False) -> Dict[str, List[Dict[str, torch.Tensor]]]:
        # [FUNCTION DOC]
        # Behavior: iterate named_modules of self; for every LogMixin module with a non-empty log list, iterate
        # its entries in order and for each key store the tensor, detached and copied to CPU, under the flat key
        # made of the module's qualified name, a slash, and the metric key.
        # Inputs: non_blocking (bool, default False) passed to the device copy.
        # Outputs: flat dictionary mapping strings like module/metric to CPU tensors.
        # Exceptions: AttributeError if a LogMixin module was never initialized.
        # Side effects: device-to-host copies (may synchronize when non_blocking is False).
        # KNOWN ISSUE: annotated and documented as returning lists of entries per module on their original CUDA device, but it actually returns a flat dict of single CPU tensors.
        # KNOWN ISSUE: when a module logged several entries with the same key, only the last entry survives (earlier ones are overwritten).
        """
        Collects all TBI logs from modules that use LogMixin.
        Returns a dictionary where keys are fully qualified module names and 
        values are lists of log entries (each entry being a dictionary of tensors).
        Tensors remain on their original CUDA devices.
        Assumes `self` is an nn.Module instance with `named_modules()` method.
        """
        all_logs: Dict[str, List[Dict[str, torch.Tensor]]] = {}
        # `self` refers to the instance of the class LoggingCollectorMixin is mixed into.
        # This class is expected to be an nn.Module or subclass.
        for name, module in self.named_modules(): 
            if isinstance(module, LogMixin):
                module_logs = module._get_internal_logs()
                if module_logs:  # Only add if there are logs for this module
                    for entry in module_logs:
                        for k, v in entry.items():
                            all_logs[name + "/" + k] = v.detach().to(device="cpu", non_blocking=non_blocking)
        return all_logs

    def clear_all_tbi_logs(self):
        # [FUNCTION DOC]
        # Behavior: for every submodule (including self) that is a LogMixin, clear its log list, silently ignoring AttributeError.
        # Inputs: none. Outputs: None. Exceptions: None (AttributeError swallowed).
        # Side effects: empties log buffers of all LogMixin submodules.
        """
        Clears TBI logs from all modules that use LogMixin.
        This should be called after processing the logs (e.g., after a forward/backward pass)
        to free up memory.
        Assumes `self` is an nn.Module instance with `modules()` method.
        """
        # `self` refers to the instance of the class LoggingCollectorMixin is mixed into.
        for module in self.modules(): 
            if isinstance(module, LogMixin):
                try:
                    module._clear_internal_logs()
                except AttributeError:
                    # Similar to get_tbi_logs, handle cases where mixin might not be fully initialized.
                    pass
