# [MODULE DOC]
# Purpose: helpers to compute weight/gradient statistics of a nanotron decoder model for experiment logging.
# Main contents: tensor norm/kurtosis/zero-fraction/summary-statistics functions, attribute lookup by dotted
# path, and MetricsLogger, which walks known parameter paths (llama-style decoder layout:
# model.decoder.N.pp_block...) and produces flat dictionaries of scalar tensors keyed by metric name.
# All computations are local to the current rank (no collectives); parameters absent on this rank (other PP
# stage or not built) are simply skipped. Import-time side effects: none (a print only when run as a script).
"""
Utility functions for computing and logging various model metrics.
"""

from typing import Any, Dict, Optional

import torch

########################
# Basic utility functions
########################


def compute_tensor_norm(tensor: torch.Tensor, p: int = 2) -> torch.Tensor:
    # [FUNCTION DOC]
    # HELPER: vector norm used by compute_tensor_stats.
    # Behavior: return the Lp vector norm of all elements (tensor treated as flat).
    # Inputs: tensor; p (int, default 2). Outputs: 0-dim tensor. Exceptions: None explicitly raised. Side effects: None.
    """Compute the Lp norm of a tensor."""
    return torch.linalg.vector_norm(tensor, ord=p)


def get_attribute_by_path(obj: Any, path: str) -> Optional[Any]:
    # [FUNCTION DOC]
    # HELPER: resolves parameter paths for MetricsLogger.
    # Behavior: split path on dots and follow attributes one by one (numeric parts work because ModuleList exposes
    # children as attributes named by index); return None as soon as one is missing. If the result is a tensor,
    # return a detached view of it, else the object itself.
    # Inputs: obj (any object, typically the model); path (dot-separated string).
    # Outputs: the resolved object, a detached tensor, or None. Exceptions: None explicitly raised. Side effects: None.
    """Get an attribute from an object by a dot-separated path."""
    path_parts = path.split(".")
    current = obj

    for i, attr in enumerate(path_parts):
        if hasattr(current, attr):
            current = getattr(current, attr)
        else:
            return None

    if isinstance(current, torch.Tensor) and hasattr(current, "detach"):
        return current.detach()

    return current


########################
# Statistical functions
########################


def compute_kurtosis(tensor: torch.Tensor) -> torch.Tensor:
    # [FUNCTION DOC]
    # Behavior: cast to float32, compute mean and (unbiased) std; if std is zero return 0.0 on the same device;
    # otherwise return the mean of the fourth power of the standardized values minus 3 (excess kurtosis).
    # Inputs: tensor of any shape. Outputs: 0-dim float tensor.
    # Exceptions: None explicitly raised. Side effects: the std equality check forces a device-to-host sync.
    """
    Compute excess kurtosis which measures "tailedness" of distribution.
    High kurtosis indicates outliers, low kurtosis indicates uniform-like distribution.
    """
    device = tensor.device
    tensor = tensor.float()

    mean = tensor.mean()
    std = tensor.std()
    if std == 0:
        return torch.tensor(0.0, device=device)

    normalized = (tensor - mean) / std
    return torch.mean(normalized.pow(4)) - 3  # Excess kurtosis (normal distribution has kurtosis=3)


def compute_zero_fraction(tensor: torch.Tensor, threshold: float = 1e-10) -> torch.Tensor:
    # [FUNCTION DOC]
    # Behavior: fraction of elements whose absolute value is below threshold.
    # Inputs: tensor; threshold (float, default 1e-10). Outputs: 0-dim float tensor. Exceptions: None. Side effects: None.
    """Compute fraction of near-zero elements."""
    return (tensor.abs() < threshold).float().mean()


def compute_tensor_stats(tensor: torch.Tensor) -> Dict[str, torch.Tensor]:
    # [FUNCTION DOC]
    # Behavior: for an empty tensor return zeros for every statistic except zero_frac which is 1.0. Otherwise cast
    # to float32 and return mean, std (square root of unbiased variance), min, max, L1 norm, L2 norm, zero fraction
    # and excess kurtosis.
    # Inputs: tensor of any shape. Outputs: dictionary with keys mean, std, min, max, norm_l1, norm_l2, zero_frac,
    # kurtosis, each a 0-dim tensor on the input's device. Exceptions: None explicitly raised. Side effects: possible sync via kurtosis.
    """Compute comprehensive statistics for a tensor."""
    if tensor.numel() == 0:
        device = tensor.device
        return {
            "mean": torch.tensor(0.0, device=device),
            "std": torch.tensor(0.0, device=device),
            "min": torch.tensor(0.0, device=device),
            "max": torch.tensor(0.0, device=device),
            "norm_l1": torch.tensor(0.0, device=device),
            "norm_l2": torch.tensor(0.0, device=device),
            "zero_frac": torch.tensor(1.0, device=device),
            "kurtosis": torch.tensor(0.0, device=device),
        }

    tensor_f = tensor.float()
    mean = tensor_f.mean()
    std = torch.sqrt(tensor_f.var())

    return {
        "mean": mean,
        "std": std,
        "min": tensor_f.min(),
        "max": tensor_f.max(),
        "norm_l1": compute_tensor_norm(tensor_f, p=1),
        "norm_l2": compute_tensor_norm(tensor_f, p=2),
        "zero_frac": compute_zero_fraction(tensor_f),
        "kurtosis": compute_kurtosis(tensor_f),
    }


class MetricsLogger:
    # [CLASS DOC]
    # Purpose: compute model weight and gradient statistics for logging.
    # Class attributes: MODEL_COMPONENTS (component type to sub-component to path template with a layer-index
    # placeholder for attention qkv/o projections, MLP gate_up/down projections and the two layer norms);
    # EMBEDDING_PATHS (token embedding and lm_head weights).
    # Instance state: config (full nanotron config), log_level and log_detail_interval (from config.metrics_logging,
    # defaults 0 and 1 when that section is None). Construction input: config.
    # KNOWN ISSUE: log_level and log_detail_interval are stored but never used by any method.
    """
    Class for logging experiment metrics with configurable detail levels.

    Supports two modes:
    - Basic (level=0): Only logs essential metrics (loss, accuracy, performance)
    - Full (level=1): Logs all metrics including detailed model statistics
    """

    # Standard component paths for finding model weights
    MODEL_COMPONENTS = {
        "attention": {
            "qkv_proj": "model.decoder.{}.pp_block.attn.qkv_proj.weight",
            "o_proj": "model.decoder.{}.pp_block.attn.o_proj.weight",
        },
        "mlp": {
            "gate_up_proj": "model.decoder.{}.pp_block.mlp.gate_up_proj.weight",
            "down_proj": "model.decoder.{}.pp_block.mlp.down_proj.weight",
        },
        "norm": {
            "input_layernorm": "model.decoder.{}.pp_block.input_layernorm.weight",
            "post_attention_layernorm": "model.decoder.{}.pp_block.post_attention_layernorm.weight",
        },
    }

    EMBEDDING_PATHS = [
        "model.token_position_embeddings.pp_block.token_embedding.weight",
        "model.lm_head.pp_block.weight",
    ]

    def __init__(self, config):
        # [FUNCTION DOC]
        # Behavior: store config; read log_level and log_detail_interval from config.metrics_logging or default to 0 and 1.
        # Inputs: config (nanotron Config with metrics_logging and model.model_config). Outputs: None.
        # Exceptions: AttributeError if config lacks metrics_logging. Side effects: sets attributes.
        """Initialize the logger with configuration."""
        self.config = config
        if self.config.metrics_logging is not None:
            self.log_level = config.metrics_logging.log_level
            self.log_detail_interval = config.metrics_logging.log_detail_interval
        else:
            self.log_level = 0
            self.log_detail_interval = 1

    def _format_paths(self, components: Dict, max_layers: int) -> Dict:
        # [FUNCTION DOC]
        # HELPER: expands path templates for every layer index.
        # Behavior: build a nested dictionary with the same structure as components where each template is replaced by the
        # list of paths for layer indices 0 to max_layers minus 1.
        # Inputs: components (nested dict of templates); max_layers (int). Outputs: nested dict of lists of strings.
        # Exceptions: None. Side effects: None.
        """Pre-format component paths with layer indices for efficiency."""
        formatted = {}
        for comp_type, subcomponents in components.items():
            formatted[comp_type] = {}
            for subcomp_name, pattern in subcomponents.items():
                formatted[comp_type][subcomp_name] = [pattern.format(i) for i in range(max_layers)]
        return formatted

    def collect_embeddings_metrics(self, model: torch.nn.Module) -> Dict[str, torch.Tensor]:
        # [FUNCTION DOC]
        # Behavior: for each path in EMBEDDING_PATHS that resolves on the model, name it by the second-to-last path part
        # (token_embedding or pp_block), add all compute_tensor_stats entries as name/stat, and add name/size as the element count.
        # Inputs: model (nn.Module). Outputs: dict of metric name to 0-dim tensor. Exceptions: None explicitly raised. Side effects: None.
        # KNOWN ISSUE: the lm_head path yields the name pp_block rather than lm_head, which is an unclear metric name.
        """Collect metrics for model embeddings."""
        metrics = {}

        for path in self.EMBEDDING_PATHS:
            embeddings = get_attribute_by_path(model, path)
            if embeddings is not None:
                name = path.split(".")[-2]
                stats = compute_tensor_stats(embeddings)

                for stat_name, value in stats.items():
                    metrics[f"{name}/{stat_name}"] = value

                metrics[f"{name}/size"] = torch.tensor(embeddings.numel(), device=embeddings.device)

        return metrics

    def compute_global_hidden_layer_metrics(self, model: torch.nn.Module) -> Dict[str, torch.Tensor]:
        # [FUNCTION DOC]
        # Behavior: with N = num_hidden_layers (return empty if 0), gather every resolvable decoder weight listed in
        # MODEL_COMPONENTS for all layers (detached, float32). If none, return empty. For each component type with weights,
        # flatten and concatenate them and add its stats under global_TYPE/stat; then concatenate all weights and add their stats under global_global/stat.
        # Inputs: model. Outputs: dict of metric name to 0-dim tensor. Exceptions: None explicitly raised.
        # Side effects: allocates concatenated float32 copies of all weights (memory heavy).
        """
        Compute global metrics across all hidden layers, excluding embeddings, final layernorm, and lm_head.
        Aggregates weights from all decoder layers to provide model-wide statistics.
        """
        metrics = {}
        max_layers = self.config.model.model_config.num_hidden_layers

        if max_layers == 0:
            return metrics

        formatted_paths = self._format_paths(self.MODEL_COMPONENTS, max_layers)

        # Group weights by component type
        component_weights = {comp_type: [] for comp_type in self.MODEL_COMPONENTS}
        all_layer_weights = []

        # Collect all weights from hidden layers
        for layer_idx in range(max_layers):
            for comp_type, subcomponents in self.MODEL_COMPONENTS.items():
                for subcomp_name in subcomponents:
                    path = formatted_paths[comp_type][subcomp_name][layer_idx]
                    param = get_attribute_by_path(model, path)
                    if param is not None:
                        param_tensor = param.detach().float()
                        all_layer_weights.append(param_tensor)
                        component_weights[comp_type].append(param_tensor)

        if not all_layer_weights:
            return metrics

        # Compute statistics for each component type
        for comp_type, weights in component_weights.items():
            if not weights:
                continue

            # Flatten tensors for global statistics calculation
            flat_tensors = [w.reshape(-1) for w in weights]
            all_comp_weights = torch.cat(flat_tensors)
            prefix = f"global_{comp_type}"

            comp_stats = compute_tensor_stats(all_comp_weights)
            for stat_name, value in comp_stats.items():
                metrics[f"{prefix}/{stat_name}"] = value

        # Compute global stats across all hidden layers
        flat_all_tensors = [w.reshape(-1) for w in all_layer_weights]
        all_weights = torch.cat(flat_all_tensors)

        global_stats = compute_tensor_stats(all_weights)
        for stat_name, value in global_stats.items():
            metrics[f"global_global/{stat_name}"] = value

        return metrics

    def collect_parameter_metrics(self, model: torch.nn.Module) -> Dict[str, torch.Tensor]:
        # [FUNCTION DOC]
        # Behavior: for each layer index and each component path that resolves, add stats under layer_I/TYPE/SUB/stat; if
        # the parameter has a gradient, add the gradient's stats under layer_I/TYPE/SUB/grad/stat. Finally add stats of
        # the final layer norm weight under final_layernorm/stat if it resolves.
        # Inputs: model. Outputs: dict of metric name to 0-dim tensor. Exceptions: None explicitly raised. Side effects: None.
        # KNOWN ISSUE: get_attribute_by_path returns a detached tensor, and detach drops the gradient, so the grad branch never triggers and gradient stats are never logged.
        """Collect detailed metrics for model parameters by layer and component."""
        metrics = {}
        max_layers = self.config.model.model_config.num_hidden_layers

        formatted_paths = self._format_paths(self.MODEL_COMPONENTS, max_layers)

        # Collect metrics for each layer
        for layer_idx in range(max_layers):
            layer_name = f"layer_{layer_idx}"

            for comp_type, subcomponents in self.MODEL_COMPONENTS.items():
                for subcomp_name in subcomponents:
                    path = formatted_paths[comp_type][subcomp_name][layer_idx]
                    param = get_attribute_by_path(model, path)

                    if param is not None:
                        # Add parameter metrics
                        stats = compute_tensor_stats(param)
                        for stat_name, value in stats.items():
                            metrics[f"{layer_name}/{comp_type}/{subcomp_name}/{stat_name}"] = value

                        # Add gradient stats if available
                        if hasattr(param, "grad") and param.grad is not None:
                            grad_stats = compute_tensor_stats(param.grad.detach())
                            for stat_name, value in grad_stats.items():
                                metrics[f"{layer_name}/{comp_type}/{subcomp_name}/grad/{stat_name}"] = value

        # Get final layer norm
        final_ln = get_attribute_by_path(model, "model.final_layer_norm.pp_block.weight")
        if final_ln is not None:
            stats = compute_tensor_stats(final_ln)
            for stat_name, value in stats.items():
                metrics[f"final_layernorm/{stat_name}"] = value

        return metrics

    def collect_all_metrics(
        self,
        model: torch.nn.Module,
    ) -> Dict[str, torch.Tensor]:
        # [FUNCTION DOC]
        # Behavior: merge the outputs of compute_global_hidden_layer_metrics, collect_embeddings_metrics and collect_parameter_metrics, in that order.
        # Inputs: model. Outputs: merged dict. Exceptions: None explicitly raised. Side effects: None.
        # KNOWN ISSUE: docstring says it depends on log level and iteration, but it ignores both and always collects everything.
        """Collect all metrics based on the specified log level and iteration."""
        metrics = {}
        metrics.update(self.compute_global_hidden_layer_metrics(model))
        metrics.update(self.collect_embeddings_metrics(model))
        metrics.update(self.collect_parameter_metrics(model))
        return metrics


if __name__ == "__main__":
    # Simple test code
    print("Metrics logging utilities loaded")
