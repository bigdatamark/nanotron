# [MODULE DOC]
# Purpose: Llama model wrappers for the DoReMi example, built on nanotron's LlamaModel with tensor/pipeline parallelism, differing in their loss heads.
# Main contents: BaseLLaMa (shared random init and cost/FLOPs delegation), LLaMaForInference (reference model returning per-token losses), LlamaForDoReMiTraining (proxy model with DoReMi excess-loss objective), LlamaReferenceForTrainingWithPerDomainLoss (reference model training with per-domain loss logging).
# Import-time side effects: creates a logger; requires transformers.
import math
from typing import Dict, Optional, Union

import torch
import torch.nn as nn
from nanotron import logging
from nanotron.config import ParallelismArgs
from nanotron.models import NanotronModel
from nanotron.models.llama import LlamaModel
from nanotron.nn.layer_norm import TritonRMSNorm
from nanotron.parallel import ParallelContext
from nanotron.parallel.parameters import NanotronParameter
from nanotron.parallel.pipeline_parallel.block import PipelineBlock, TensorPointer
from nanotron.parallel.tensor_parallel.functional import sharded_cross_entropy
from nanotron.parallel.tensor_parallel.nn import (
    TensorParallelColumnLinear,
    TensorParallelEmbedding,
    TensorParallelRowLinear,
)
from transformers import LlamaConfig

from .doremi_context import DoReMiContext
from .loss import CrossEntropyWithPerDomainLoss, DoReMiLossForProxyTraining

logger = logging.get_logger(__name__)


class BaseLLaMa(NanotronModel):
    @torch.no_grad()
    # [CLASS DOC]
    # Purpose: Common NanotronModel base for DoReMi Llama variants. Subclasses must set self.model to a LlamaModel.
    # Key behavior: random parameter initialization handling tied and TP-sharded params; delegates block compute costs and FLOPs to the inner model.
    def init_model_randomly(self, config):
        # [FUNCTION DOC]
        # Behavior: Under no_grad, builds a map from module id to name prefix (root as empty string), then for each parameter (asserted to be a NanotronParameter) resolves its full name (tied params use the tied-info full name) and skips ones already initialized. Initialization by owning module type: column-parallel linear weight normal(0, std), bias zero; row-parallel linear weight normal(0, std/sqrt(2*num_layers)), bias zero; TritonRMSNorm weight one, bias zero; embedding weight normal(0, std). Finally asserts the set of initialized names equals the set of all parameter names.
        # Inputs: config: nanotron Config (uses model.init_method.std and model.model_config.num_hidden_layers).
        # Outputs: None.
        # Exceptions: ValueError for unknown parameter names in known modules; generic Exception for parameters in unsupported modules; AssertionError on non-Nanotron params or init set mismatch.
        # Side effects: overwrites parameters in place; consumes the torch RNG of the current device.
        # KNOWN ISSUE: docstring mentions apply_layernorm_1p, which the code does not use (norm weights are always 1).
        """Initialize model parameters randomly.
        Note:
            Layernorm weight all 0 or 1 depending on `apply_layernorm_1p`
        """

        model = self
        initialized_parameters = set()
        # Handle tensor parallelism
        module_id_to_prefix = {id(module): f"{module_name}." for module_name, module in model.named_modules()}
        # Fix the root_model
        module_id_to_prefix[id(model)] = ""

        std = config.model.init_method.std
        sigma = config.model.init_method.std
        num_layers = config.model.model_config.num_hidden_layers

        for param_name, param in model.named_parameters():
            assert isinstance(param, NanotronParameter)

            module_name, param_name = param_name.rsplit(".", 1)

            if param.is_tied:
                tied_info = param.get_tied_info()
                full_param_name = tied_info.get_full_name_from_module_id_to_prefix(
                    module_id_to_prefix=module_id_to_prefix
                )
            else:
                full_param_name = f"{module_name}.{param_name}"

            if full_param_name in initialized_parameters:
                # Already initialized
                continue

            module = model.get_submodule(module_name)

            if isinstance(module, TensorParallelColumnLinear):
                if "weight" == param_name:
                    nn.init.normal_(module.weight, mean=0.0, std=std)
                elif "bias" == param_name:
                    module.bias.zero_()
                else:
                    raise ValueError(f"Who the fuck is {param_name}?")
            elif isinstance(module, TensorParallelRowLinear):
                if "weight" == param_name:
                    nn.init.normal_(module.weight, mean=0.0, std=sigma / math.sqrt(2 * num_layers))
                elif "bias" == param_name:
                    param.zero_()
                else:
                    raise ValueError(f"Who the fuck is {param_name}?")
            elif isinstance(module, TritonRMSNorm):
                if "weight" == param_name:
                    # TODO @thomasw21: Sometimes we actually want 0
                    module.weight.fill_(1)
                elif "bias" == param_name:
                    module.bias.zero_()
                else:
                    raise ValueError(f"Who the fuck is {param_name}?")
            elif isinstance(module, TensorParallelEmbedding):
                nn.init.normal_(module.weight, mean=0.0, std=std)
            else:
                raise Exception(f"Parameter {full_param_name} was not initialized")

            assert full_param_name not in initialized_parameters
            initialized_parameters.add(full_param_name)

        assert initialized_parameters == {
            param.get_tied_info().get_full_name_from_module_id_to_prefix(module_id_to_prefix=module_id_to_prefix)
            if param.is_tied
            else name
            for name, param in model.named_parameters()
        }, f"Somehow the initialized set of parameters don't match:\n - Expected: { {name for name, _ in model.named_parameters()} }\n - Got: {initialized_parameters}"

    def get_block_compute_costs(self):
        # [FUNCTION DOC]
        # Behavior: Delegates to the inner LlamaModel to get per-block compute costs for pipeline load balancing.
        # Inputs: self. Outputs: dictionary mapping block class to cost. Exceptions: None explicitly raised. Side effects: None.
        """Computes the compute cost of each block in the model so that we can do a better job of load balancing."""
        return self.model.get_block_compute_costs()

    def get_flops_per_sec(self, iteration_time_in_sec, sequence_length, global_batch_size):
        # [FUNCTION DOC]
        # Behavior: Delegates FLOPs-per-second computation to the inner LlamaModel.
        # Inputs: iteration_time_in_sec (float), sequence_length (int), global_batch_size (int).
        # Outputs: whatever the inner model returns (model and hardware TFLOPs). Exceptions: None explicitly raised. Side effects: None.
        """Get flops per second for a given model"""
        return self.model.get_flops_per_sec(iteration_time_in_sec, sequence_length, global_batch_size)


class LLaMaForInference(BaseLLaMa):
    # [CLASS DOC]
    # Purpose: Reference Llama used during proxy training only to compute per-token cross-entropy losses (no reduction).
    # Attributes: model (LlamaModel), parallel_context, config, parallel_config.
    def __init__(
        self,
        config: LlamaConfig,
        parallel_config: Optional[ParallelismArgs],
        parallel_context: ParallelContext,
    ):
        # [FUNCTION DOC]
        # Behavior: Builds the inner LlamaModel and stores contexts/configs.
        # Inputs: config (LlamaConfig), parallel_config (optional ParallelismArgs), parallel_context (ParallelContext).
        # Outputs: None. Exceptions: from LlamaModel construction. Side effects: allocates parameters on device.
        super().__init__()
        self.model = LlamaModel(config=config, parallel_context=parallel_context, parallel_config=parallel_config)
        self.parallel_context = parallel_context
        self.config = config
        self.parallel_config = parallel_config

    def forward(
        self,
        input_ids: Union[torch.Tensor, TensorPointer],
        input_mask: Union[torch.Tensor, TensorPointer],
        label_ids: Union[torch.Tensor, TensorPointer],
        label_mask: Union[torch.Tensor, TensorPointer],
    ) -> Dict[str, Union[torch.Tensor, TensorPointer]]:
        # [FUNCTION DOC]
        # Behavior: Runs the model to get TP-sharded logits (sequence-first), computes sharded cross-entropy against labels transposed to sequence-first in float32 across the TP group, and transposes back to batch-first.
        # Inputs: input_ids, input_mask, label_ids, label_mask: tensors of shape [batch, seq] or TensorPointers.
        # Outputs: dictionary with "losses" of shape [batch, seq] (unreduced per-token loss).
        # Exceptions: None explicitly raised.
        # Side effects: TP collectives inside the model and the sharded cross-entropy; pipeline p2p inside LlamaModel.
        # KNOWN ISSUE: label_mask is ignored, and the loss is computed outside a PipelineBlock, so on pipeline ranks that do not hold logits (pp greater than 1) this would fail on a TensorPointer.
        sharded_logits = self.model(
            input_ids=input_ids,
            input_mask=input_mask,
        )
        loss = sharded_cross_entropy(
            sharded_logits,
            label_ids.transpose(0, 1).contiguous(),
            group=self.parallel_context.tp_pg,
            dtype=torch.float,
        ).transpose(0, 1)
        return {"losses": loss}


class LlamaForDoReMiTraining(BaseLLaMa):
    # [CLASS DOC]
    # Purpose: Proxy Llama model trained with the DoReMi objective; its loss block also updates domain weights.
    # Attributes: model (LlamaModel), loss (PipelineBlock wrapping DoReMiLossForProxyTraining with inputs sharded_logits, label_ids, label_mask, domain_idxs, ref_losses and outputs loss, ce_loss, domain_losses, domain_weights, samples_per_domain), parallel_context, config, parallel_config.
    def __init__(
        self,
        config: LlamaConfig,
        parallel_context: ParallelContext,
        doremi_context: DoReMiContext,
        parallel_config: Optional[ParallelismArgs],
    ):
        # [FUNCTION DOC]
        # Behavior: Builds the LlamaModel and the DoReMi loss pipeline block sharing the model's p2p.
        # Inputs: config (LlamaConfig), parallel_context, doremi_context (DoReMiContext), parallel_config (optional ParallelismArgs).
        # Outputs: None. Exceptions: from construction. Side effects: allocates parameters.
        super().__init__()
        self.model = LlamaModel(config=config, parallel_context=parallel_context, parallel_config=parallel_config)
        self.loss = PipelineBlock(
            p2p=self.model.p2p,
            module_builder=DoReMiLossForProxyTraining,
            module_kwargs={
                "parallel_context": parallel_context,
                "doremi_context": doremi_context,
            },
            module_input_keys={
                "sharded_logits",
                "label_ids",
                "label_mask",
                "domain_idxs",
                "ref_losses",
            },
            module_output_keys={
                "loss",
                "ce_loss",
                "domain_losses",
                "domain_weights",
                "samples_per_domain",
            },
        )
        self.parallel_context = parallel_context
        self.config = config
        self.parallel_config = parallel_config

    def forward(
        self,
        input_ids: Union[torch.Tensor, TensorPointer],
        input_mask: Union[torch.Tensor, TensorPointer],
        label_ids: Union[torch.Tensor, TensorPointer],
        label_mask: Union[torch.Tensor, TensorPointer],
        domain_idxs: Optional[Union[torch.Tensor, TensorPointer]],
        ref_losses: Optional[Union[torch.Tensor, TensorPointer]],
    ) -> Dict[str, Union[torch.Tensor, TensorPointer]]:
        # [FUNCTION DOC]
        # Behavior: Computes sharded logits, transposes them to batch-first, and passes them with labels, mask, domain indices and reference losses to the loss block.
        # Inputs: input_ids, input_mask, label_ids, label_mask ([batch, seq] tensors or TensorPointers); domain_idxs ([batch] or [batch,1]); ref_losses ([batch, seq]).
        # Outputs: dictionary of the loss block outputs (tensors or TensorPointers).
        # Exceptions: None explicitly raised.
        # Side effects: pipeline/TP communication; domain weight update inside the loss (see loss module).
        sharded_logits = self.model(
            input_ids=input_ids,
            input_mask=input_mask,
        )
        sharded_logits = sharded_logits.transpose(0, 1).contiguous()
        outputs = self.loss(
            sharded_logits=sharded_logits,
            label_ids=label_ids,
            label_mask=label_mask,
            domain_idxs=domain_idxs,
            ref_losses=ref_losses,
        )
        return outputs


class LlamaReferenceForTrainingWithPerDomainLoss(BaseLLaMa):
    # [CLASS DOC]
    # Purpose: Reference Llama model trained with standard cross-entropy while logging per-domain losses.
    # Attributes: model (LlamaModel), loss (PipelineBlock wrapping CrossEntropyWithPerDomainLoss with outputs loss, domain_losses, samples_per_domain), parallel_context, config, parallel_config.
    def __init__(
        self,
        config: LlamaConfig,
        doremi_context: DoReMiContext,
        parallel_context: ParallelContext,
        parallel_config: Optional[ParallelismArgs],
    ):
        # [FUNCTION DOC]
        # Behavior: Builds the LlamaModel and the per-domain cross-entropy pipeline block.
        # Inputs: config (LlamaConfig), doremi_context, parallel_context, parallel_config (optional).
        # Outputs: None. Exceptions: from construction. Side effects: allocates parameters.
        super().__init__()
        self.model = LlamaModel(config=config, parallel_context=parallel_context, parallel_config=parallel_config)
        self.loss = PipelineBlock(
            p2p=self.model.p2p,
            module_builder=CrossEntropyWithPerDomainLoss,
            module_kwargs={
                "doremi_context": doremi_context,
                "parallel_context": parallel_context,
            },
            module_input_keys={"sharded_logits", "label_ids", "label_mask", "domain_idxs"},
            module_output_keys={"loss", "domain_losses", "samples_per_domain"},
        )
        self.parallel_context = parallel_context
        self.config = config
        self.parallel_config = parallel_config

    def forward(
        self,
        input_ids: Union[torch.Tensor, TensorPointer],
        input_mask: Union[torch.Tensor, TensorPointer],
        label_ids: Union[torch.Tensor, TensorPointer],
        label_mask: Union[torch.Tensor, TensorPointer],
        domain_idxs: Union[torch.Tensor, TensorPointer],
    ) -> Dict[str, Union[torch.Tensor, TensorPointer]]:
        # [FUNCTION DOC]
        # Behavior: Computes sharded logits, transposes to batch-first, applies the loss block, and returns its loss, domain_losses and samples_per_domain entries.
        # Inputs: input_ids, input_mask, label_ids, label_mask, domain_idxs (tensors or TensorPointers).
        # Outputs: dictionary with loss, domain_losses, samples_per_domain.
        # Exceptions: None explicitly raised. Side effects: pipeline/TP communication and the collectives in the loss.
        sharded_logits = self.model(
            input_ids=input_ids,
            input_mask=input_mask,
        )
        sharded_logits = sharded_logits.transpose(0, 1).contiguous()
        outputs = self.loss(
            sharded_logits=sharded_logits,
            label_ids=label_ids,
            label_mask=label_mask,
            domain_idxs=domain_idxs,
        )
        return {
            "loss": outputs["loss"],
            "domain_losses": outputs["domain_losses"],
            "samples_per_domain": outputs["samples_per_domain"],
        }
