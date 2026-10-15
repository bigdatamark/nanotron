# [MODULE DOC]
# Purpose: builds and holds all torch.distributed process groups used for 5D parallelism (expert, pipeline,
# data, context, tensor).
# Main contents: DistributedBackend literal type, ParallelContext.
# Import-time side effects: None.
import os
from typing import Dict, Literal

import numpy as np
import torch

import nanotron.distributed as dist

DistributedBackend = Literal["gloo", "mpi", "nccl"]


class ParallelContext:
    # [CLASS DOC]
    # Purpose: central object describing parallel topology and owning process groups for the current rank.
    # Key state: sizes (tensor/pipeline/data/context/expert parallel, world, local world); world_pg; local_pg; tp_pg,
    # cp_pg, dp_pg, pp_pg, ep_pg; mp_pg (all ep, pp, tp ranks for a fixed dp and cp); dp_cp_pg (all dp and cp ranks for
    # fixed ep, pp, tp); tp_and_ep_pg (all ep and tp ranks for fixed pp, dp, cp); world_ranks_to_pg (cache of sorted rank
    # tuples to groups); world_rank_matrix (array of global ranks with axes ep, pp, dp, cp, tp, tp fastest);
    # parallel_order (axis names list).
    # Invariant: product of sizes equals WORLD_SIZE; groups are created collectively on all ranks in the same order.
    def __init__(
        self,
        tensor_parallel_size: int,
        pipeline_parallel_size: int,
        data_parallel_size: int,
        context_parallel_size: int = 1,
        expert_parallel_size: int = 1,
        backend: DistributedBackend = "nccl",
    ):
        # [FUNCTION DOC]
        # Behavior: reads WORLD_SIZE and LOCAL_WORLD_SIZE (default 8 when world above 8, else world size), asserts
        # TP*PP*CP*DP equals world size, checks torch.distributed availability, stores sizes, sets the CUDA device from
        # LOCAL_RANK, asserts backend is nccl, initializes torch distributed if needed, creates world_pg over all ranks,
        # and builds all parallel groups.
        # Inputs: tensor_parallel_size, pipeline_parallel_size, data_parallel_size (ints), context_parallel_size (default
        # 1), expert_parallel_size (default 1), backend (default "nccl").
        # Outputs: None.
        # Exceptions: KeyError if WORLD_SIZE unset; AssertionError on size mismatch or non-nccl backend; ValueError if
        # torch.distributed unavailable.
        # Side effects: reads env vars; sets CUDA device; initializes process group; collective group creation and barriers.
        # KNOWN ISSUE: expert_parallel_size is excluded from the world-size assertion yet included in the rank reshape, so any expert size above 1 fails the reshape unless the other sizes are reduced, which then fails the assertion.
        # KNOWN ISSUE: backend is validated only after the device is set and is otherwise ignored (the default distributed backend is used).
        """Initialize parallel context."""
        world_size = int(os.environ["WORLD_SIZE"])
        local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", "8")) if world_size > 8 else world_size

        assert (
            tensor_parallel_size * pipeline_parallel_size * context_parallel_size * data_parallel_size
        ) == world_size, f"TP*CP*DP*PP={tensor_parallel_size}*{pipeline_parallel_size}*{context_parallel_size}*{data_parallel_size}={tensor_parallel_size * pipeline_parallel_size * context_parallel_size * data_parallel_size} != WORLD_SIZE={world_size}"

        if not dist.is_available():
            raise ValueError("torch.distributed is not available as a package, please install it.")

        self.tensor_parallel_size = tensor_parallel_size
        self.pipeline_parallel_size = pipeline_parallel_size
        self.data_parallel_size = data_parallel_size
        self.context_parallel_size = context_parallel_size
        self.expert_parallel_size = expert_parallel_size
        self.world_size = world_size
        self.local_world_size = local_world_size

        self._groups = {}

        self.set_device()

        assert backend == "nccl", "Only nccl backend is supported for now."

        if not dist.is_initialized():
            dist.initialize_torch_distributed()

        ranks = list(range(self.world_size))
        process_group = dist.new_group(
            ranks=ranks,
            backend=dist.get_backend(),
        )
        self.world_pg = process_group

        self._init_parallel_groups()

    def _init_parallel_groups(self):
        # [FUNCTION DOC]
        # HELPER: builds every process group from the rank matrix.
        # Behavior: barrier; reshapes range(world) to (ep, pp, dp, cp, tp); creates local_pg (consecutive blocks of local
        # world size) and asserts LOCAL_RANK equals the rank in it; creates tp, cp, dp, pp, ep groups by moving the target
        # axis last and reshaping; creates mp_pg, dp_cp_pg, tp_and_ep_pg from explicit slices; stores world_rank_matrix and
        # parallel_order.
        # Inputs/Outputs: None.
        # Exceptions: AssertionError on local rank mismatch; TypeError if LOCAL_RANK is unset; ValueError if reshapes fail.
        # Side effects: collective group creation on all ranks.
        # KNOWN ISSUE: comment says model parallel group is tp, pp and expert, which matches code, but mp_pg is what clip_grad_norm uses, and it omits cp ranks; fine only if cp ranks hold distinct gradients reduced elsewhere.
        """Initialize 3D parallelism's all process groups."""
        dist.barrier()
        ranks = np.arange(0, self.world_size).reshape(
            (
                self.expert_parallel_size,
                self.pipeline_parallel_size,
                self.data_parallel_size,
                self.context_parallel_size,
                self.tensor_parallel_size,
            )
        )
        self.world_ranks_to_pg = {}
        self.local_pg = self.create_new_group(ranks.reshape((-1, self.local_world_size)))
        assert int(os.environ.get("LOCAL_RANK")) == dist.get_rank(self.local_pg), "Local rank mismatch"

        # Relevant process groups containing the current rank
        self.tp_pg = self.create_new_group(ranks.transpose((0, 1, 2, 3, 4)).reshape((-1, self.tensor_parallel_size)))
        self.cp_pg = self.create_new_group(ranks.transpose((4, 0, 1, 2, 3)).reshape((-1, self.context_parallel_size)))
        self.dp_pg = self.create_new_group(ranks.transpose((3, 4, 0, 1, 2)).reshape((-1, self.data_parallel_size)))
        self.pp_pg = self.create_new_group(ranks.transpose((2, 3, 4, 0, 1)).reshape((-1, self.pipeline_parallel_size)))
        self.ep_pg = self.create_new_group(
            ranks.transpose((1, 2, 3, 4, 0)).reshape((-1, self.expert_parallel_size))
        )  # TODO: ep should be a subset of dp

        # model parallel group = combination of tp and pp and exp for a given dp rank
        self.mp_pg = self.create_new_group(
            [
                ranks[:, :, dp_rank, cp_rank, :].reshape(-1)
                for cp_rank in range(self.context_parallel_size)
                for dp_rank in range(self.data_parallel_size)
            ]
        )

        self.dp_cp_pg = self.create_new_group(
            [
                ranks[ep_rank, pp_rank, :, :, tp_rank].reshape(-1)
                for tp_rank in range(self.tensor_parallel_size)
                for pp_rank in range(self.pipeline_parallel_size)
                for ep_rank in range(self.expert_parallel_size)
            ]
        )

        self.tp_and_ep_pg = self.create_new_group(
            [
                ranks[:, pp_rank, dp_rank, cp_rank, :].reshape(-1)
                for cp_rank in range(self.context_parallel_size)
                for pp_rank in range(self.pipeline_parallel_size)
                for dp_rank in range(self.data_parallel_size)
            ]
        )

        # self.tp_and_cp_pg = self.create_new_group(
        #     [
        #         ranks[ep_rank, pp_rank, dp_rank, :, :].reshape(-1)
        #         for ep_rank in range(self.expert_parallel_size)
        #         for pp_rank in range(self.pipeline_parallel_size)
        #         for dp_rank in range(self.data_parallel_size)
        #     ]
        # )

        self.world_rank_matrix: np.ndarray = ranks
        self.parallel_order = ["ep", "pp", "dp", "cp", "tp"]

    def create_new_group(self, all_groups_ranks: np.ndarray) -> dist.ProcessGroup:
        # [FUNCTION DOC]
        # Behavior: barrier; for each rank list creates (or reuses from world_ranks_to_pg by sorted tuple) a group, every
        # rank participating in every creation; returns the group containing the current RANK; barrier.
        # Inputs: all_groups_ranks (2D array or list of rank arrays).
        # Outputs: ProcessGroup containing this rank, or None if none does.
        # Exceptions: KeyError if RANK unset.
        # Side effects: new_group collectives; mutates world_ranks_to_pg.
        dist.barrier()
        rank = int(os.environ["RANK"])
        new_group_containing_rank = None
        for group_ranks in all_groups_ranks:
            sorted_ranks = tuple(sorted(group_ranks))

            # add new group to `world_ranks_to_pg`
            if sorted_ranks not in self.world_ranks_to_pg:
                new_group = dist.new_group(ranks=group_ranks)
                self.world_ranks_to_pg[sorted_ranks] = new_group
            else:
                new_group = self.world_ranks_to_pg[sorted_ranks]

            if rank in sorted_ranks:
                new_group_containing_rank = new_group
        dist.barrier()
        return new_group_containing_rank

    def set_device(self):
        # [FUNCTION DOC]
        # Behavior: sets the current CUDA device to LOCAL_RANK (default 0), assuming homogeneous nodes.
        # Inputs/Outputs: None. Exceptions: CUDA errors if device absent. Side effects: CUDA device state.
        local_rank = int(os.getenv("LOCAL_RANK", "0"))

        # NOTE: Set the device id.
        # `torch.cuda.device_count` should return the number of device on a single node.
        # We assume the nodes to be homogeneous (same number of gpus per node)
        device_id = local_rank
        torch.cuda.set_device(torch.cuda.device(device_id))

    def get_local_ranks(self, world_rank: int) -> Dict[str, int]:
        # [FUNCTION DOC]
        # Behavior: finds world_rank in world_rank_matrix and returns its coordinate per axis.
        # Inputs: world_rank (int). Outputs: dict axis name ("ep", "pp", "dp", "cp", "tp") to int.
        # Exceptions: ValueError from item if rank not found. Side effects: None.
        # return tuple(i.item() for i in np.where(self.world_rank_matrix == world_rank))
        local_ranks = np.where(self.world_rank_matrix == world_rank)
        return {ax: local_ranks[i].item() for i, ax in enumerate(self.parallel_order)}

    def destroy(self):
        # [FUNCTION DOC]
        # Behavior: if distributed is initialized, barrier then destroy the default process group.
        # Inputs/Outputs: None. Exceptions: None explicitly raised. Side effects: tears down distributed state.
        if not dist.is_initialized():
            return

        dist.barrier()
        dist.destroy_process_group()

    def get_global_rank(
        self,
        ep_rank: int,
        pp_rank: int,
        dp_rank: int,
        cp_rank: int,
        tp_rank: int,
    ) -> np.int64:
        # [FUNCTION DOC]
        # Behavior: indexes world_rank_matrix with the five coordinates.
        # Inputs: ep_rank, pp_rank, dp_rank, cp_rank, tp_rank (ints; arrays also allowed by numpy indexing).
        # Outputs: numpy int64 global rank. Exceptions: IndexError if out of range. Side effects: None.
        """
        Get the global rank based on the specified ranks in different parallel groups.

        :param ep_rank: int, Rank in the expert parallel group.
        :param pp_rank: int, Rank in the pipeline parallel group.
        :param dp_rank: int, Rank in the data parallel group.
        :param cp_rank: int, Rank in the context parallel group.
        :param tp_rank: int, Rank in the tensor parallel group.

        :return: numpy.int64, The global rank.
        """
        return self.world_rank_matrix[ep_rank, pp_rank, dp_rank, cp_rank, tp_rank]
