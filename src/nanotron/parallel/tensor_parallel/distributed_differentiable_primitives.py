# [MODULE DOC]
# Purpose: differentiable collective primitives for tensor parallelism (Megatron-style f/g operators and
# all-gather/reduce-scatter pairs along dimension 0).
# Main contents: DifferentiableIdentity, DifferentiableAllReduceSum, DifferentiableAllGather,
# DifferentiableReduceScatterSum and their functional wrappers.
# Import-time side effects: None.
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Optional

import torch
from torch import distributed as torch_dist

from nanotron import distributed as dist
from nanotron.distributed import ProcessGroup


class DifferentiableIdentity(torch.autograd.Function):
    # [CLASS DOC]
    # Purpose: identity in forward, all-reduce sum of gradients in backward. Inheritance: torch.autograd.Function.
    """All-reduce gradients in a differentiable fashion"""

    @staticmethod
    def forward(ctx, tensor, group: Optional[ProcessGroup]):
        # [FUNCTION DOC]
        # Stores group on ctx and returns tensor unchanged. Inputs: ctx, tensor, group (ProcessGroup or None).
        # Outputs: tensor. Exceptions/Side effects: None.
        ctx.group = group
        return tensor

    @staticmethod
    def backward(ctx, grad_output):
        # [FUNCTION DOC]
        # Returns the all-reduced (sum over group) gradient and None. Inputs: ctx, grad_output.
        # Outputs: (tensor, None). Side effects: all-reduce collective, in place on grad_output.
        group = ctx.group
        return DifferentiableAllReduceSum.apply(grad_output, group), None


class DifferentiableAllReduceSum(torch.autograd.Function):
    # [CLASS DOC]
    # Purpose: all-reduce sum in forward, identity in backward. Inheritance: torch.autograd.Function.
    """All-reduce in a differentiable fashion"""

    @staticmethod
    def forward(ctx, tensor, group: Optional[ProcessGroup]):
        # [FUNCTION DOC]
        # Behavior: returns tensor unchanged if group size is 1; else all-reduces it in place with SUM and returns it.
        # Inputs: ctx, tensor, group. Outputs: tensor.
        # Exceptions: AttributeError if group is None. Side effects: collective; mutates the input in place.
        # KNOWN ISSUE: group annotated Optional, but None is dereferenced (group.size) before any default-group fallback.
        if group.size() == 1:
            return tensor

        dist.all_reduce(tensor, op=dist.ReduceOp.SUM, group=group)
        return tensor

    @staticmethod
    def backward(ctx, grad_output):
        # [FUNCTION DOC]
        # Returns grad_output unchanged and None. Inputs: ctx, grad_output. Outputs: (tensor, None).
        return grad_output, None


class DifferentiableAllGather(torch.autograd.Function):
    # [CLASS DOC]
    # Purpose: all-gather along dim 0 in forward, reduce-scatter sum in backward. Inheritance: torch.autograd.Function.
    """All gather in a differentiable fashion"""

    @staticmethod
    def forward(ctx, tensor, group: Optional[ProcessGroup]):
        # [FUNCTION DOC]
        # Behavior: stores group; returns tensor if group size 1; else allocates output with dim 0 multiplied by group size,
        # makes input contiguous and all-gathers into it (rank order).
        # Inputs: ctx, tensor (shape [b, ...]), group. Outputs: tensor [b * group size, ...].
        # Exceptions: AttributeError if group is None. Side effects: collective; allocation.
        # KNOWN ISSUE: the None-group fallback to the default group is unreachable because group.size is called first.
        ctx.group = group

        if group.size() == 1:
            return tensor

        # TODO @thomasw21: gather along another dimension
        sharded_batch_size, *rest_size = tensor.shape
        if group is None:
            group = torch_dist.distributed_c10d._get_default_group()
        unsharded_batch_size = sharded_batch_size * group.size()

        unsharded_tensor = torch.empty(
            unsharded_batch_size,
            *rest_size,
            device=tensor.device,
            dtype=tensor.dtype,
            requires_grad=tensor.requires_grad,
        )

        # `tensor` can sometimes not be contiguous
        # https://cs.github.com/pytorch/pytorch/blob/2b267fa7f28e18ca6ea1de4201d2541a40411457/torch/distributed/nn/functional.py#L317
        tensor = tensor.contiguous()

        dist.all_gather_into_tensor(unsharded_tensor, tensor, group=group)
        return unsharded_tensor

    @staticmethod
    def backward(ctx, grad_output):
        # [FUNCTION DOC]
        # Returns reduce-scatter-sum of grad_output over the group, and None. Inputs: ctx, grad_output.
        # Outputs: (tensor, None). Side effects: collective.
        group = ctx.group
        out = DifferentiableReduceScatterSum.apply(grad_output, group)
        return out, None


class DifferentiableReduceScatterSum(torch.autograd.Function):
    # [CLASS DOC]
    # Purpose: reduce-scatter sum along dim 0 in forward, all-gather in backward. Inheritance: torch.autograd.Function.
    """Reduce scatter in a differentiable fashion"""

    @staticmethod
    def forward(ctx, tensor, group: Optional[ProcessGroup]):
        # [FUNCTION DOC]
        # Behavior: stores group; returns tensor if size 1; else asserts dim 0 divisible by group size, makes contiguous,
        # allocates a shard of dim 0 divided by group size, and reduce-scatters with SUM so rank r gets the r-th block.
        # Inputs: ctx, tensor ([B, ...]), group. Outputs: tensor [B / group size, ...].
        # Exceptions: AssertionError on indivisible dim 0; AttributeError if group is None. Side effects: collective.
        # KNOWN ISSUE: the None-group fallback is unreachable (group.size called first).
        ctx.group = group

        if group.size() == 1:
            return tensor

        # TODO @thomasw21: shard along another dimension
        unsharded_batch_size, *rest_size = tensor.shape
        if group is None:
            group = torch_dist.distributed_c10d._get_default_group()
        assert unsharded_batch_size % group.size() == 0

        # TODO @thomasw21: Collectives seem to require tensors to be contiguous
        # https://cs.github.com/pytorch/pytorch/blob/2b267fa7f28e18ca6ea1de4201d2541a40411457/torch/distributed/nn/functional.py#L305
        tensor = tensor.contiguous()

        sharded_tensor = torch.empty(
            unsharded_batch_size // group.size(),
            *rest_size,
            device=tensor.device,
            dtype=tensor.dtype,
            requires_grad=False,
        )
        dist.reduce_scatter_tensor(sharded_tensor, tensor, group=group, op=dist.ReduceOp.SUM)
        return sharded_tensor

    @staticmethod
    def backward(ctx, grad_output):
        # [FUNCTION DOC]
        # Returns all-gather of grad_output over the group and None. Inputs: ctx, grad_output. Outputs: (tensor, None).
        # Side effects: collective.
        group = ctx.group
        return DifferentiableAllGather.apply(grad_output, group), None


# -----------------
# Helper functions.
# -----------------


def differentiable_identity(tensor, group: Optional[ProcessGroup] = None):
    # [FUNCTION DOC]
    # Applies DifferentiableIdentity. Inputs: tensor, group (default None). Outputs: tensor.
    return DifferentiableIdentity.apply(tensor, group)


def differentiable_all_reduce_sum(tensor, group: Optional[ProcessGroup] = None):
    # [FUNCTION DOC]
    # Applies DifferentiableAllReduceSum. Inputs: tensor, group (default None). Outputs: tensor (input mutated in place).
    return DifferentiableAllReduceSum.apply(tensor, group)


def differentiable_all_gather(tensor, group: Optional[ProcessGroup] = None):
    # [FUNCTION DOC]
    # Applies DifferentiableAllGather. Inputs: tensor, group (default None). Outputs: gathered tensor.
    return DifferentiableAllGather.apply(tensor, group)


def differentiable_reduce_scatter_sum(tensor, group: Optional[ProcessGroup] = None):
    # [FUNCTION DOC]
    # Applies DifferentiableReduceScatterSum. Inputs: tensor, group (default None). Outputs: sharded tensor.
    return DifferentiableReduceScatterSum.apply(tensor, group)
