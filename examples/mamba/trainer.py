# [MODULE DOC]
# Purpose: Mamba-specific subclass of nanotron's DistributedTrainer. Overrides parameter tying and
# checkpoint loading / random initialization so Mamba models (MambaForTraining) train correctly with
# TP/PP/DP. Contents: MambaTrainer. Import-time side effects: module logger creation only.
from typing import Optional, Type, Union

from config import ExistingCheckpointInit, MambaConfig, MambaInit
from nanotron import distributed as dist
from nanotron import logging
from nanotron.config import ParallelismArgs
from nanotron.logging import log_rank
from nanotron.models import NanotronModel
from nanotron.parallel import ParallelContext
from nanotron.parallel.parameters import NanotronParameter
from nanotron.parallel.pipeline_parallel.utils import get_pp_rank_of
from nanotron.parallel.tensor_parallel.nn import (
    TensorParallelLinearMode,
    TensorParallelRowLinear,
)
from nanotron.parallel.tied_parameters import (
    create_pg_for_tied_weights,
    get_tied_id_to_param,
    tie_parameters,
)
from nanotron.serialize import load_weights, parse_ckpt_path
from nanotron.trainer import DistributedTrainer
from torch.nn.parallel import DistributedDataParallel

logger = logging.get_logger(__name__)


class MambaTrainer(DistributedTrainer):
    # [CLASS DOC]
    # DistributedTrainer specialised for Mamba. Construction requires config_class to be MambaConfig.
    # Overrides _mark_tied_parameters (ties embeddings with lm_head, custom params, and syncs replicated
    # non-sharded params across TP without reduction for Mamba) and _load_model_checkpoint (supports
    # resuming, ExistingCheckpointInit and MambaInit random init followed by DP/tied averaging).
    # State set: init_checkpoint_path, param_shard_metadata (plus everything from DistributedTrainer).
    def __init__(
        self,
        config_or_config_file: Union[MambaConfig, str],
        config_class: Type[MambaConfig] = MambaConfig,
        model_config_class: Optional[Type] = None,
        model_class: Type[NanotronModel] = None,
    ):
        # [FUNCTION DOC]
        # Behavior: asserts config_class is MambaConfig, then delegates to DistributedTrainer.__init__.
        # Inputs: config_or_config_file (MambaConfig or path string); config_class (default MambaConfig);
        # model_config_class (optional class used to parse model_config); model_class (NanotronModel subclass, default None).
        # Outputs: None. Exceptions: AssertionError if config_class differs. Side effects: all those of DistributedTrainer
        # construction (distributed init, model build, logging, etc.).
        assert config_class == MambaConfig
        super().__init__(config_or_config_file, config_class, model_config_class, model_class)

    def _mark_tied_parameters(
        self,
        model: NanotronModel,
        parallel_context: ParallelContext,
        parallel_config: Optional[ParallelismArgs] = None,
    ):
        # [FUNCTION DOC]
        # Behavior:
        # 1. Embedding tying: for each name returned by model.get_embeddings_lm_head_tied_names, compute the global
        #    rank holding it (world_rank_matrix indexed by this rank's ep, the pp rank owning the module, dp, tp) and
        #    tie them with tie_parameters using SUM reduction.
        # 2. Calls model.tie_custom_params.
        # 3. For every direct parameter of every module that is not a sharded or already-tied NanotronParameter and
        #    is not the bias of a TensorParallelRowLinear (bias only on tp rank 0), tie it across all ranks of the
        #    TP group. Reduction is None (synced by design) when parallel_config is None, TP mode is ALL_REDUCE, or the
        #    model config has is_mamba_config; otherwise SUM.
        # 4. Creates process groups for all tied weights (create_pg_for_tied_weights).
        # Inputs: model (NanotronModel, not DDP-wrapped); parallel_context; parallel_config (optional ParallelismArgs).
        # Outputs: None. Exceptions: AssertionError if model is DistributedDataParallel.
        # Side effects: mutates parameter tie metadata on the model; creates new process groups (collective).
        # [FUNCTION DOC]
        # Behavior:
        # 1. Embedding tying: for each name returned by model.get_embeddings_lm_head_tied_names, compute the global
        #    rank holding it (world_rank_matrix indexed by this rank's ep, the pp rank owning the module, dp, tp) and
        #    tie them with tie_parameters using SUM reduction.
        # 2. Calls model.tie_custom_params.
        # 3. For every direct parameter of every module that is not a sharded or already-tied NanotronParameter and
        #    is not the bias of a TensorParallelRowLinear (bias only on tp rank 0), tie it across all ranks of the
        #    TP group. Reduction is None (synced by design) when parallel_config is None, TP mode is ALL_REDUCE, or the
        #    model config has is_mamba_config; otherwise SUM.
        # 4. Creates process groups for all tied weights (create_pg_for_tied_weights).
        # Inputs: model (NanotronModel, not DDP-wrapped); parallel_context; parallel_config (optional ParallelismArgs).
        # Outputs: None. Exceptions: AssertionError if model is DistributedDataParallel.
        # Side effects: mutates parameter tie metadata on the model; creates new process groups (collective).
        # Tie embeddings
        embeddings_lm_head_tied_names = model.get_embeddings_lm_head_tied_names()
        if len(embeddings_lm_head_tied_names) > 0:
            shared_embeddings = [
                (
                    target,
                    (
                        parallel_context.world_rank_matrix[
                            dist.get_rank(parallel_context.ep_pg),
                            get_pp_rank_of(target, module=model),
                            dist.get_rank(parallel_context.dp_pg),
                            dist.get_rank(parallel_context.tp_pg),
                        ],
                    ),
                )
                for target in embeddings_lm_head_tied_names
            ]
            tie_parameters(
                root_module=model,
                ties=shared_embeddings,
                parallel_context=parallel_context,
                reduce_op=dist.ReduceOp.SUM,
            )

        # Tie custom params
        model.tie_custom_params()

        # Sync all parameters that have the same name and that are not sharded
        assert not isinstance(model, DistributedDataParallel), "model shouldn't be DDP at this point"
        for module_name, module in model.named_modules():
            for param_name, param in module.named_parameters(recurse=False):
                name = f"{module_name}.{param_name}"

                if isinstance(param, NanotronParameter) and (param.is_sharded or param.is_tied):
                    continue

                if isinstance(module, TensorParallelRowLinear) and "bias" == param_name:
                    # bias for TensorParallelRowLinear only exists on TP=0 so we don't need to tie it
                    continue

                shared_weights = [
                    (
                        name,
                        # sync across TP group
                        tuple(sorted(dist.get_process_group_ranks(parallel_context.tp_pg))),
                    )
                ]

                if (
                    parallel_config is None
                    or parallel_config.tp_mode is TensorParallelLinearMode.ALL_REDUCE
                    or hasattr(model.config.model.model_config, "is_mamba_config")
                ):
                    # We add `reduce_op=None` in order to signal that the weight are synced by design without needing to reduce
                    # when TP=2 we have LN that is duplicated across TP, so by design it's tied
                    reduce_op = None
                else:
                    reduce_op = dist.ReduceOp.SUM

                tie_parameters(
                    root_module=model, ties=shared_weights, parallel_context=parallel_context, reduce_op=reduce_op
                )

        create_pg_for_tied_weights(root_module=model, parallel_context=parallel_context)

    def _load_model_checkpoint(self, model: NanotronModel) -> NanotronModel:
        # [FUNCTION DOC]
        # Behavior: unwraps DDP; resolves init_checkpoint_path via parse_ckpt_path. If a path exists, loads weights
        # from it (logged on rank 0) and stores param_shard_metadata. Otherwise logs that no checkpoint was given and:
        # with ExistingCheckpointInit loads weights from init_method.path; with MambaInit calls
        # init_model_randomly, then all-reduces (AVG) every parameter over the DP group in name order, and
        # all-reduces (AVG) every tied parameter over its tied group (looked up in world_ranks_to_pg).
        # Inputs: model (NanotronModel, possibly DDP-wrapped). Outputs: the same model object.
        # Exceptions: ValueError for an unsupported init_method.
        # Side effects: sets self.init_checkpoint_path and possibly self.param_shard_metadata; reads files; collectives
        # over DP and tied groups; mutates model weights; logging.
        # KNOWN ISSUE: in the MambaInit branch param_shard_metadata is never set, unlike the other branches.
        unwrapped_model = model.module if isinstance(model, DistributedDataParallel) else model

        # Load or initialize model weights
        self.init_checkpoint_path = parse_ckpt_path(config=self.config)
        reloaded_from_checkpoint = False
        if self.init_checkpoint_path is not None:
            # Reload from a training checkpoint
            log_rank(f"Loading weights from {self.init_checkpoint_path}", logger=logger, level=logging.INFO, rank=0)
            self.param_shard_metadata = load_weights(
                model=unwrapped_model, parallel_context=self.parallel_context, root_folder=self.init_checkpoint_path
            )
            reloaded_from_checkpoint = True
        if not reloaded_from_checkpoint:
            log_rank("No checkpoint path provided.", logger=logger, level=logging.INFO)
            if isinstance(self.config.model.init_method, ExistingCheckpointInit):
                # Initialize model from an pretrained model checkpoint
                self.param_shard_metadata = load_weights(
                    model=unwrapped_model,
                    parallel_context=self.parallel_context,
                    root_folder=self.config.model.init_method.path,
                )
            elif isinstance(self.config.model.init_method, MambaInit):

                unwrapped_model.init_model_randomly(config=self.config)
                # Synchronize parameters so that the model is consistent
                # sync all params across dp
                for name, param in sorted(model.named_parameters(), key=lambda x: x[0]):
                    dist.all_reduce(param, op=dist.ReduceOp.AVG, group=self.parallel_context.dp_pg)

                # sync tied params across tied groups
                for (_, group_ranks), param in sorted(
                    get_tied_id_to_param(
                        parameters=model.parameters(),
                        root_module=unwrapped_model,
                    ).items(),
                    key=lambda x: x[0],
                ):
                    group = self.parallel_context.world_ranks_to_pg[group_ranks]
                    dist.all_reduce(param, op=dist.ReduceOp.AVG, group=group)
            else:
                raise ValueError(f"Unsupported {self.config.model.init_method}")

        return model
