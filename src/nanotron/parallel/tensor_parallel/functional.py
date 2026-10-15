# coding=utf-8
# [MODULE DOC]
# Purpose: tensor-parallel functional ops (Megatron-LM derived): vocab-sharded cross entropy (optionally with
# z-loss), column-parallel and row-parallel linear with ALL_REDUCE or REDUCE_SCATTER (sequence parallel) modes,
# including async variants that overlap communication with matmuls.
# Main contents: _ShardedCrossEntropy, _ShardedCrossEntropyWithZLoss, sharded_cross_entropy,
# _ColumnLinearAsyncCommunication, _ColumnLinearNoAsyncCommunicationReduceScatterMode, column_linear,
# _RowLinearAsyncCommunication, row_linear.
# Conventions: in REDUCE_SCATTER mode activations are sharded along dim 0 across the TP group (rank r owns block
# r); weights of column linear are sharded on output features, of row linear on input features.
# Overlap relies on CUDA_DEVICE_MAX_CONNECTIONS=1 so collectives are scheduled before following matmuls.
# "allgather"/"allgather2" buffers come from the global MemoryBuffer to save memory.
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
import math
from typing import Optional

import torch
from torch.nn import functional as F

import nanotron.distributed as dist
from nanotron.parallel.tensor_parallel.distributed_differentiable_primitives import (
    differentiable_all_reduce_sum,
    differentiable_identity,
    differentiable_reduce_scatter_sum,
)
from nanotron.parallel.tensor_parallel.enum import TensorParallelLinearMode
from nanotron.parallel.utils import MemoryBuffer


class _ShardedCrossEntropy(torch.autograd.Function):
    # [CLASS DOC]
    # Purpose: cross entropy over logits whose vocab (last) dimension is sharded across a TP group; rank r holds vocab
    # range r*V_local to (r+1)*V_local. Inheritance: torch.autograd.Function.
    @staticmethod
    def forward(
        ctx,
        sharded_logits,  # (*, sharded_hidden_size)
        target,  # (*)
        group: dist.ProcessGroup,
    ):
        # [FUNCTION DOC]
        # Behavior: all-reduce MAX of per-row max logits and subtract it; mask targets outside this shard and shift
        # in-shard targets to local indices; gather local target logits (zero for masked) and all-reduce SUM to get the
        # target logit everywhere; exponentiate logits in place, sum and all-reduce SUM to get the partition Z; loss is
        # log Z minus target logit; normalize exponentials into softmax and save softmax, mask and local targets.
        # Inputs: ctx, sharded_logits ([..., V_local]), target (integer tensor [...], global vocab ids), group.
        # Outputs: per-token loss with target's shape.
        # Exceptions: None explicitly raised. Side effects: three all-reduces on group.
        # Maximum value along last dimension across all GPUs.
        logits_max = torch.max(sharded_logits, dim=-1)[0]
        dist.all_reduce(logits_max, op=dist.ReduceOp.MAX, group=group)
        # Subtract the maximum value.
        sharded_logits = sharded_logits - logits_max.unsqueeze(dim=-1)

        # Get the shard's indices
        sharded_hidden_size = sharded_logits.shape[-1]
        rank = dist.get_rank(group)
        start_index = rank * sharded_hidden_size
        end_index = start_index + sharded_hidden_size

        # Create a mask of valid ids (1 means it needs to be masked).
        target_mask = (target < start_index) | (target >= end_index)
        masked_target = target.clone() - start_index
        masked_target[target_mask] = 0

        # Get predicted-logits = logits[target].
        # For Simplicity, we convert logits to a 2-D tensor with size
        # [*, shard-size] and target to a 1-D tensor of size [*].
        logits_2d = sharded_logits.view(-1, sharded_hidden_size)
        masked_target_1d = masked_target.view(-1)
        arange_1d = torch.arange(start=0, end=logits_2d.shape[0], device=logits_2d.device)
        predicted_logits_1d = logits_2d[arange_1d, masked_target_1d]
        if predicted_logits_1d.is_contiguous():
            predicted_logits_1d = predicted_logits_1d.clone()
        else:
            predicted_logits_1d = predicted_logits_1d.contiguous()
        predicted_logits = predicted_logits_1d.view_as(target)
        predicted_logits[target_mask] = 0.0
        # All reduce is needed to get the chunks from other GPUs.
        dist.all_reduce(predicted_logits, op=dist.ReduceOp.SUM, group=group)

        # Sum of exponential of logits along vocab dimension across all GPUs.
        exp_logits = sharded_logits
        torch.exp(sharded_logits, out=exp_logits)
        sum_exp_logits = exp_logits.sum(dim=-1)
        dist.all_reduce(sum_exp_logits, op=dist.ReduceOp.SUM, group=group)

        # Loss = log(sum(exp(logits))) - predicted-logit.
        loss = torch.log(sum_exp_logits) - predicted_logits

        # Normalize and optionally smooth logits
        exp_logits.div_(sum_exp_logits.unsqueeze(dim=-1))

        # Store softmax, target-mask and masked-target for backward pass.
        ctx.save_for_backward(exp_logits, target_mask, masked_target_1d)

        return loss.view_as(target)

    @staticmethod
    def backward(ctx, grad_output):
        # [FUNCTION DOC]
        # Behavior: gradient is softmax minus one-hot of the target (only for in-shard targets), multiplied by grad_output
        # broadcast over the vocab dim; computed in place on the saved softmax.
        # Inputs: ctx, grad_output (shape of target). Outputs: (grad for logits, None, None).
        # Exceptions: None. Side effects: overwrites saved softmax (backward cannot run twice).
        # Retrieve tensors from the forward path.
        softmax, target_mask, masked_target_1d = ctx.saved_tensors

        # All the inputs have softmax as their gradient.
        grad_input = softmax
        # For simplicity, work with the 2D gradient.
        sharded_hidden_size = softmax.size()[-1]
        grad_2d = grad_input.view(-1, sharded_hidden_size)

        # Add the gradient from matching classes.
        arange_1d = torch.arange(start=0, end=grad_2d.size()[0], device=grad_2d.device)
        grad_2d[arange_1d, masked_target_1d] -= 1.0 - target_mask.view(-1).float()

        # Finally elementwise multiplication with the output gradients.
        grad_input.mul_(grad_output.unsqueeze(dim=-1))

        return grad_input, None, None


class _ShardedCrossEntropyWithZLoss(torch.autograd.Function):
    # [CLASS DOC]
    # Purpose: same as _ShardedCrossEntropy plus z-loss regularization z_loss_coef times log(Z) squared.
    # Inheritance: torch.autograd.Function.
    @staticmethod
    def forward(
        ctx,
        sharded_logits,  # (batch_size, length, sharded_hidden_size)
        target,  # (batch_size, length)
        group: dist.ProcessGroup,
        z_loss_coef: float = 0.0,  # Coefficient for z-loss regularization
    ):
        # [FUNCTION DOC]
        # Behavior: as _ShardedCrossEntropy forward; additionally computes log Z, and if z_loss_coef is positive adds
        # z_loss_coef times the square of log Z clamped to between -20 and 20 to the loss (else z_loss is zeros); saves
        # softmax, mask, local targets and log Z, and the coefficient on ctx.
        # Inputs: ctx, sharded_logits, target, group, z_loss_coef (float, default 0.0).
        # Outputs: tuple (total loss including z-loss, z_loss) each shaped like target.
        # Exceptions: None. Side effects: all-reduces on group.
        # Maximum value along last dimension across all GPUs.
        logits_max = torch.max(sharded_logits, dim=-1)[0]
        dist.all_reduce(logits_max, op=dist.ReduceOp.MAX, group=group)
        # Subtract the maximum value.
        sharded_logits = sharded_logits - logits_max.unsqueeze(dim=-1)

        # Get the shard's indices
        sharded_hidden_size = sharded_logits.shape[-1]
        rank = dist.get_rank(group)
        start_index = rank * sharded_hidden_size
        end_index = start_index + sharded_hidden_size

        # Create a mask of valid ids (1 means it needs to be masked).
        target_mask = (target < start_index) | (target >= end_index)
        masked_target = target.clone() - start_index
        masked_target[target_mask] = 0

        # Get predicted-logits = logits[target].
        # For Simplicity, we convert logits to a 2-D tensor with size
        # [*, shard-size] and target to a 1-D tensor of size [*].
        logits_2d = sharded_logits.view(-1, sharded_hidden_size)
        masked_target_1d = masked_target.view(-1)
        arange_1d = torch.arange(start=0, end=logits_2d.shape[0], device=logits_2d.device)
        predicted_logits_1d = logits_2d[arange_1d, masked_target_1d]
        if predicted_logits_1d.is_contiguous():
            predicted_logits_1d = predicted_logits_1d.clone()
        else:
            predicted_logits_1d = predicted_logits_1d.contiguous()
        predicted_logits = predicted_logits_1d.view_as(target)
        predicted_logits[target_mask] = 0.0
        # All reduce is needed to get the chunks from other GPUs.
        dist.all_reduce(predicted_logits, op=dist.ReduceOp.SUM, group=group)

        # Sum of exponential of logits along vocab dimension across all GPUs.
        exp_logits = sharded_logits
        torch.exp(sharded_logits, out=exp_logits)
        sum_exp_logits = exp_logits.sum(dim=-1)
        dist.all_reduce(sum_exp_logits, op=dist.ReduceOp.SUM, group=group)

        # Loss = log(sum(exp(logits))) - predicted-logit.
        loss = torch.log(sum_exp_logits) - predicted_logits

        # Store the log(Z) for backward pass
        log_z = torch.log(sum_exp_logits)

        # Apply z-loss regularization if coefficient is non-zero
        if z_loss_coef > 0.0:
            # Z is the partition function (sum_exp_logits)
            # z_loss = z_loss_coef * log²(Z)
            # Use clamp to prevent extreme values
            z_loss = z_loss_coef * torch.square(log_z.clamp(min=-20.0, max=20.0))
            loss = loss + z_loss
        else:
            z_loss = torch.zeros_like(loss)

        # Normalize and optionally smooth logits
        exp_logits.div_(sum_exp_logits.unsqueeze(dim=-1))

        # Store softmax, target-mask, masked-target and z_loss_coef for backward pass.
        ctx.save_for_backward(exp_logits, target_mask, masked_target_1d, log_z)
        ctx.z_loss_coef = z_loss_coef

        return loss.view_as(target), z_loss.view_as(target)

    @staticmethod
    def backward(ctx, grad_output, grad_z_loss=None):
        # [FUNCTION DOC]
        # Behavior: clones softmax, subtracts in-shard one-hot targets, and if z_loss_coef is positive adds 2 * coef *
        # log Z * softmax; scales by grad_output.
        # Inputs: ctx, grad_output (for the total loss), grad_z_loss (ignored). Outputs: (grad logits, None, None, None).
        # Exceptions: None. Side effects: None.
        # KNOWN ISSUE: gradient of the z-loss uses unclamped log Z although forward clamps it, so gradients disagree with the forward when the clamp is active.
        # KNOWN ISSUE: gradient flowing into the second output (z_loss) is ignored, so using z_loss itself in a loss produces no gradient.
        # Retrieve tensors from the forward path.
        softmax, target_mask, masked_target_1d, log_z = ctx.saved_tensors
        z_loss_coef = ctx.z_loss_coef

        # All the inputs have softmax as their gradient.
        grad_input = softmax.clone()
        # For simplicity, work with the 2D gradient.
        sharded_hidden_size = softmax.size()[-1]
        grad_2d = grad_input.view(-1, sharded_hidden_size)

        # Add the gradient from matching classes.
        arange_1d = torch.arange(start=0, end=grad_2d.size()[0], device=grad_2d.device)
        grad_2d[arange_1d, masked_target_1d] -= 1.0 - target_mask.view(-1).float()

        # If z-loss is applied, we need to add its gradient contribution
        if z_loss_coef > 0.0:
            # The gradient of z_loss with respect to logits is:
            # 2 * z_loss_coef * log(Z) * (1/Z) * exp(logits)
            # But exp(logits)/Z is just the softmax, which we already have

            # Apply the gradient scaling with clamping for stability
            # clamped_log_z = log_z.clamp(min=-20.0, max=20.0) useless i think?
            z_loss_grad_scale = 2.0 * z_loss_coef * log_z

            # Add the z-loss gradient contribution
            # We need to reshape z_loss_grad_scale to match grad_input's dimensions
            z_loss_grad = softmax * z_loss_grad_scale.unsqueeze(-1)
            grad_input = grad_input + z_loss_grad

        # Finally elementwise multiplication with the output gradients.
        grad_input.mul_(grad_output.unsqueeze(dim=-1))

        return grad_input, None, None, None


def sharded_cross_entropy(
    sharded_logits,
    target,
    group: dist.ProcessGroup,
    dtype: torch.dtype = None,
    z_loss_coef: float = 0.0,
):
    # [FUNCTION DOC]
    # Behavior: optionally casts logits to dtype, then applies the z-loss variant when z_loss_coef is positive, else
    # the plain variant.
    # Inputs: sharded_logits, target, group (TP group), dtype (optional), z_loss_coef (float, default 0.0).
    # Outputs: loss tensor, or tuple (loss, z_loss) when z_loss_coef is positive.
    # Exceptions: None explicitly raised. Side effects: collectives.
    # KNOWN ISSUE: return type differs by branch (tensor versus tuple), so callers must know which was used.
    """Helper function for the cross entropy."""
    if dtype is not None:
        # Cast input to specific dtype.
        sharded_logits = sharded_logits.to(dtype=dtype)
    if z_loss_coef > 0.0:
        return _ShardedCrossEntropyWithZLoss.apply(sharded_logits, target, group, z_loss_coef)
    else:
        return _ShardedCrossEntropy.apply(sharded_logits, target, group)


class _ColumnLinearAsyncCommunication(torch.autograd.Function):
    # [CLASS DOC]
    # Purpose: column-parallel linear with communication overlapped with compute (adapted from Megatron-LM).
    # Inheritance: torch.autograd.Function.
    """Adapted from https://github.com/NVIDIA/Megatron-LM/blob/e6d7e09845590d0a36bc7f29eb28db974fb8da4e/megatron/core/tensor_parallel/layers.py#L215"""

    @staticmethod
    def forward(ctx, tensor, weight, bias, group, tp_mode, tp_recompute_allgather):
        # [FUNCTION DOC]
        # Behavior: saves mode, group, recompute flag and input shape. ALL_REDUCE: plain linear, saving input and weight.
        # REDUCE_SCATTER with group size 1: same. Otherwise: makes input contiguous, starts an async all-gather of inputs
        # along dim 0 (into the shared "allgather" buffer when recomputing, else a fresh tensor), meanwhile computes this
        # rank's output block from the local input, waits, saves either the local input (recompute) or gathered input, then
        # computes the output blocks before and after the local block from the gathered input; bias added via addmm.
        # Inputs: ctx, tensor ([b_local, ..., h_in] or full in ALL_REDUCE), weight ([h_out_local, h_in]), bias (or None),
        # group, tp_mode, tp_recompute_allgather (bool).
        # Outputs: tensor [b_local * group size, ..., h_out_local] (REDUCE_SCATTER) or [..., h_out_local].
        # Exceptions: ValueError for unknown mode. Side effects: async all-gather; MemoryBuffer use.
        # KNOWN ISSUE: the group-None fallback is unreachable because group.size is called earlier.
        ctx.use_bias = bias is not None
        ctx.tp_mode = tp_mode
        ctx.group = group
        ctx.tp_recompute_allgather = tp_recompute_allgather
        ctx.tensor_shape = tensor.size()

        if tp_mode is TensorParallelLinearMode.ALL_REDUCE:
            gathered_tensor = tensor
            ctx.save_for_backward(tensor, weight)
            return F.linear(gathered_tensor, weight, bias)
        elif tp_mode is TensorParallelLinearMode.REDUCE_SCATTER:
            group_size = group.size()
            current_rank = dist.get_rank(group)
            if group_size == 1:
                gathered_tensor = tensor
                ctx.save_for_backward(tensor, weight)
                return F.linear(gathered_tensor, weight, bias)
            else:
                # `tensor` can sometimes not be contiguous
                # https://cs.github.com/pytorch/pytorch/blob/2b267fa7f28e18ca6ea1de4201d2541a40411457/torch/distributed/nn/functional.py#L317
                tensor = tensor.contiguous()
                # ctx.save_for_backward(tensor, weight)

                # TODO @thomasw21: gather along another dimension
                sharded_batch_size, *intermediate_size, hidden_size = tensor.shape
                if group is None:
                    group = dist.distributed_c10d._get_default_group()
                gathered_batch_size = sharded_batch_size * group.size()

                if tp_recompute_allgather:
                    gathered_tensor = MemoryBuffer().get(
                        "allgather", (gathered_batch_size, *intermediate_size, hidden_size), dtype=tensor.dtype
                    )
                else:
                    gathered_tensor = torch.empty(
                        gathered_batch_size,
                        *intermediate_size,
                        hidden_size,
                        device=tensor.device,
                        dtype=tensor.dtype,
                        requires_grad=False,
                    )

                handle = dist.all_gather_into_tensor(gathered_tensor, tensor, group=group, async_op=True)

                # Compute a shard of column_linear in the same time of AllGather
                # We could compute the matmul of current holding shard and the current rank's weight
                # We assume that rank 0 holds w0, rank 1 holds w1, etc.
                # weights: w0 w1 w2 w3
                # rank 0:  X  -  -  -
                # rank 1:  -  X  -  -
                # rank 2:  -  -  X  -
                # rank 3:  -  -  -  X
                # We call the corresponding shard of output "same_device_shard"
                output_size = weight.shape[0]
                gathered_output = torch.empty(
                    gathered_batch_size,
                    *intermediate_size,
                    output_size,
                    device=tensor.device,
                    dtype=tensor.dtype,
                    requires_grad=tensor.requires_grad,
                )
                before_shard, same_device_shard, after_shard = torch.split(
                    gathered_output,
                    split_size_or_sections=[
                        sharded_batch_size * current_rank,
                        sharded_batch_size,
                        sharded_batch_size * (group_size - current_rank - 1),
                    ],
                    dim=0,
                )
                first_dims = math.prod([sharded_batch_size, *intermediate_size])
                if bias is None:
                    torch.mm(
                        input=tensor.view(first_dims, hidden_size),
                        mat2=weight.t(),
                        out=same_device_shard.view(first_dims, output_size),
                    )
                else:
                    torch.addmm(
                        input=bias[None, :],
                        mat1=tensor.view(first_dims, hidden_size),
                        mat2=weight.t(),
                        out=same_device_shard.view(first_dims, output_size),
                    )

                # Wait communication
                handle.wait()
                if tp_recompute_allgather:
                    ctx.save_for_backward(tensor, weight)
                else:
                    ctx.save_for_backward(gathered_tensor, weight)

                # Compute all the other shards that are obtained from AllGather
                # weights: w0 w1 w2 w3
                # rank 0:  -  X  X  X
                # rank 1:  X  -  X  X
                # rank 2:  X  X  -  X
                # rank 3:  X  X  X  -
                # As they could be not contiguous (r1 and r2) vertically as they are separated by "same_device_shard"
                # We need to compute them separately, i.e. "before_shard" and "after_shard"
                # For r0, "before_shard" is empty. For r3, "after_shard" is empty.
                if before_shard.numel() > 0:
                    first_dims = math.prod(before_shard.shape[:-1])
                    if bias is None:
                        torch.mm(
                            input=gathered_tensor[: sharded_batch_size * current_rank].view(first_dims, hidden_size),
                            mat2=weight.t(),
                            out=before_shard.view(first_dims, output_size),
                        )
                    else:
                        torch.addmm(
                            input=bias[None, :],
                            mat1=gathered_tensor[: sharded_batch_size * current_rank].view(first_dims, hidden_size),
                            mat2=weight.t(),
                            out=before_shard.view(first_dims, output_size),
                        )
                if after_shard.numel() > 0:
                    first_dims = math.prod(after_shard.shape[:-1])
                    if bias is None:
                        torch.mm(
                            input=gathered_tensor[sharded_batch_size * (current_rank + 1) :].view(
                                first_dims, hidden_size
                            ),
                            mat2=weight.t(),
                            out=after_shard.view(first_dims, output_size),
                        )
                    else:
                        torch.addmm(
                            input=bias[None, :],
                            mat1=gathered_tensor[sharded_batch_size * (current_rank + 1) :].view(
                                first_dims, hidden_size
                            ),
                            mat2=weight.t(),
                            out=after_shard.view(first_dims, output_size),
                        )

                return gathered_output
        else:
            raise ValueError(f"Got unexpected mode: {tp_mode}.")

    @staticmethod
    def backward(ctx, grad_output):
        # [FUNCTION DOC]
        # Behavior: in REDUCE_SCATTER with recompute, async re-all-gathers the input into the shared buffer. Computes input
        # gradient grad_output times weight; launches async reduce-scatter of it (REDUCE_SCATTER, size above 1) or async
        # all-reduce (ALL_REDUCE). Computes bias grad as sum over rows, waits for the gather, computes weight grad
        # grad_output transposed times gathered input, waits for the reduction.
        # Inputs: ctx, grad_output. Outputs: (input grad, weight grad, bias grad or None, None, None, None).
        # Exceptions: ValueError for unknown mode. Side effects: async collectives.
        tensor, weight = ctx.saved_tensors
        group = ctx.group
        use_bias = ctx.use_bias
        tp_mode = ctx.tp_mode

        handle1: Optional[dist.Work] = None
        if tp_mode is TensorParallelLinearMode.REDUCE_SCATTER and ctx.tp_recompute_allgather:
            sharded_batch_size, *rest_size = tensor.shape
            if group is None:
                group = dist.distributed_c10d._get_default_group()

            if group.size() == 1:
                total_tensor = tensor
            else:
                unsharded_batch_size = sharded_batch_size * group.size()

                unsharded_tensor = MemoryBuffer().get(
                    "allgather", (unsharded_batch_size, *rest_size), dtype=tensor.dtype
                )
                handle1 = dist.all_gather_into_tensor(unsharded_tensor, tensor, group=group, async_op=True)
                # Here we rely on CUDA_DEVICE_MAX_CONNECTIONS=1 to ensure that the
                # gather is scheduled before the tensor gradient computation
                total_tensor = unsharded_tensor
        else:
            total_tensor = tensor

        grad_tensor = grad_output.matmul(weight)

        # Doing gather + slicing during the NeMo forward pass can make this tensor
        # not be contiguous. PyTorch only checks if the tensor is contiguous, and only
        # clones it if it's not contiguous:
        # https://github.com/pytorch/pytorch/blob/c47cf9bc7f9e02f649ab4ed53fe4d35732c92ab6/torch/_refs/__init__.py#L2761
        grad_output = grad_output.contiguous()
        # Convert the tensor shapes to 2D for execution compatibility
        grad_output_first_dims, grad_output_last_dim = grad_output.shape[:-1], grad_output.shape[-1]
        total_tensor_first_dims, total_tensor_last_dim = total_tensor.shape[:-1], total_tensor.shape[-1]
        grad_output = grad_output.view(math.prod(grad_output_first_dims), grad_output_last_dim)
        total_tensor = total_tensor.view(math.prod(total_tensor_first_dims), total_tensor_last_dim)

        handle2: Optional[dist.Work] = None
        if tp_mode is TensorParallelLinearMode.REDUCE_SCATTER:
            if group.size() == 1:
                sub_grad_tensor = grad_tensor
            else:
                sub_grad_tensor = torch.empty(
                    ctx.tensor_shape, dtype=grad_tensor.dtype, device=grad_tensor.device, requires_grad=False
                )
                # reduce_scatter
                handle2 = dist.reduce_scatter_tensor(sub_grad_tensor, grad_tensor, group=group, async_op=True)
                # Here we rely on CUDA_DEVICE_MAX_CONNECTIONS=1 to ensure that the
                # reduce scatter is scheduled before the weight gradient computation
        elif tp_mode is TensorParallelLinearMode.ALL_REDUCE:
            # Asynchronous all-reduce
            handle2 = dist.all_reduce(grad_tensor, group=group, async_op=True)
            # Here we rely on CUDA_DEVICE_MAX_CONNECTIONS=1 to ensure that the
            # all-reduce is scheduled before the weight gradient computation
        else:
            raise ValueError()

        grad_bias = grad_output.sum(dim=0) if use_bias else None

        if handle1 is not None:
            handle1.wait()

        # TODO @thomasw21: This sounds like we don't have the optimal physical layout
        grad_weight = grad_output.t().matmul(total_tensor)

        if handle2 is not None:
            handle2.wait()

        if tp_mode is TensorParallelLinearMode.REDUCE_SCATTER:
            return sub_grad_tensor, grad_weight, grad_bias, None, None, None
        elif tp_mode is TensorParallelLinearMode.ALL_REDUCE:
            return grad_tensor, grad_weight, grad_bias, None, None, None
        else:
            raise ValueError(f"Got unexpected mode: {tp_mode}.")


class _ColumnLinearNoAsyncCommunicationReduceScatterMode(torch.autograd.Function):
    # [CLASS DOC]
    # Purpose: synchronous column-parallel linear for REDUCE_SCATTER mode with optional all-gather recomputation in
    # backward. Inheritance: torch.autograd.Function.
    """
    Column linear with memory_buffer for the allgather, context parallel
    enabled (i.e. tp_mode = TensorParallelLinearMode.REDUCE_SCATTER) and
    async communication disabled.
    """

    @staticmethod
    def forward(
        ctx,
        input: torch.Tensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor],
        group: dist.ProcessGroup,
        tp_recompute_allgather: bool,
    ):

        # [FUNCTION DOC]
        # Behavior: all-gathers input along dim 0 (into the shared buffer if recomputing, else a new tensor; no-op for size
        # 1), saves local input (recompute) or gathered input plus weight and bias, returns linear of the gathered input.
        # Inputs: ctx, input ([b_local, ...]), weight, bias (optional), group, tp_recompute_allgather (bool).
        # Outputs: tensor [b_local * group size, ..., h_out_local]. Exceptions: None. Side effects: all-gather.
        # Do allgather.
        sharded_batch_size, *rest_size = input.shape
        unsharded_batch_size = sharded_batch_size * group.size()
        if group.size() == 1:
            total_input = input.contiguous()
        elif tp_recompute_allgather:
            total_input = MemoryBuffer().get("allgather", (unsharded_batch_size, *rest_size), dtype=input.dtype)
            dist.all_gather_into_tensor(total_input, input.contiguous(), group=group)
        else:
            total_input = torch.empty(unsharded_batch_size, *rest_size, dtype=input.dtype, device=input.device)
            dist.all_gather_into_tensor(total_input, input.contiguous(), group=group)

        # Prepare context.
        ctx.group = group
        ctx.tp_recompute_allgather = tp_recompute_allgather
        ctx.input_size = input.shape
        if tp_recompute_allgather:
            ctx.save_for_backward(input, weight, bias)
        else:
            ctx.save_for_backward(total_input, weight, bias)

        # Get linear output.
        out = F.linear(total_input, weight, bias)
        return out

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        # [FUNCTION DOC]
        # Behavior: obtains the gathered input (from saved tensors or by re-all-gathering), flattens to 2D, computes weight
        # grad and input grad; reduce-scatters input grad along dim 0 back to the local shape (size above 1); bias grad is
        # sum over rows.
        # Inputs: ctx, grad_output. Outputs: (input grad, weight grad, bias grad or None, None, None).
        # Exceptions: None. Side effects: collectives.
        # KNOWN ISSUE: total_input is first assigned an integer that is immediately overwritten (dead assignment).
        # Either allgather the inputs again or get them from context.
        group = ctx.group
        tp_recompute_allgather = ctx.tp_recompute_allgather
        input_size = ctx.input_size
        if group.size() == 1 or not tp_recompute_allgather:
            total_input, weight, bias = ctx.saved_tensors
        else:
            input, weight, bias = ctx.saved_tensors
            sharded_batch_size, *rest_size = input.shape
            total_input = sharded_batch_size * group.size()
            unsharded_batch_size = sharded_batch_size * group.size()
            total_input = MemoryBuffer().get("allgather", (unsharded_batch_size, *rest_size), dtype=input.dtype)
            dist.all_gather_into_tensor(total_input, input.contiguous(), group=group)

        # Convert the tensor shapes to 2D for execution compatibility
        grad_output = grad_output.contiguous()
        grad_output_first_dims, grad_output_last_dim = grad_output.shape[:-1], grad_output.shape[-1]
        total_input_first_dims, total_input_last_dim = total_input.shape[:-1], total_input.shape[-1]
        grad_output = grad_output.view(math.prod(grad_output_first_dims), grad_output_last_dim)
        total_input = total_input.view(math.prod(total_input_first_dims), total_input_last_dim)

        # Compute gradients.
        grad_weight = grad_output.T @ total_input
        grad_input = grad_output @ weight
        if group.size() == 1:
            sub_grad_input = grad_input
        else:
            # Seems that `reduce_scatter` need contiguous tensors: https://github.com/pytorch/pytorch/blob/2b267fa7f28e18ca6ea1de4201d2541a40411457/torch/distributed/nn/functional.py#L305
            # We set grad_input to be contiguous in case it isn't already.
            grad_input = grad_input.contiguous()
            sub_grad_input = torch.empty(
                input_size, dtype=total_input.dtype, device=total_input.device, requires_grad=False
            )
            dist.reduce_scatter_tensor(sub_grad_input, grad_input, group=group, op=dist.ReduceOp.SUM)
        grad_bias = torch.sum(grad_output, dim=0) if bias is not None else None

        return sub_grad_input, grad_weight, grad_bias, None, None


def column_linear(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    group: dist.ProcessGroup,
    tp_mode: TensorParallelLinearMode,
    async_communication: bool,
    tp_recompute_allgather: bool = True,
):
    # [FUNCTION DOC]
    # Behavior: dispatch: async_communication uses _ColumnLinearAsyncCommunication; else ALL_REDUCE applies
    # differentiable_identity (all-reduce in backward) then linear; REDUCE_SCATTER uses the no-async function.
    # Inputs: input, weight, bias (optional), group (TP group), tp_mode, async_communication (bool),
    # tp_recompute_allgather (bool, default True).
    # Outputs: output tensor. Exceptions: ValueError on unknown mode. Side effects: collectives.
    if async_communication:
        return _ColumnLinearAsyncCommunication.apply(input, weight, bias, group, tp_mode, tp_recompute_allgather)

    if tp_mode is TensorParallelLinearMode.ALL_REDUCE:
        input = differentiable_identity(input, group=group)
        return F.linear(input, weight, bias)
    if tp_mode is TensorParallelLinearMode.REDUCE_SCATTER:
        return _ColumnLinearNoAsyncCommunicationReduceScatterMode.apply(
            input, weight, bias, group, tp_recompute_allgather
        )
    raise ValueError(f"Got unexpected mode: {tp_mode}.")


class _RowLinearAsyncCommunication(torch.autograd.Function):
    # [CLASS DOC]
    # Purpose: row-parallel linear in REDUCE_SCATTER mode with backward all-gather of grad_output overlapped with the
    # input-gradient matmul. Inheritance: torch.autograd.Function.
    @staticmethod
    def forward(ctx, tensor, weight, bias, group, tp_mode):
        # [FUNCTION DOC]
        # Behavior: asserts REDUCE_SCATTER mode; defaults group; computes linear and (size above 1) reduce-scatters it
        # along dim 0; saves input and weight.
        # Inputs: ctx, tensor ([b, ..., h_in_local]), weight ([h_out, h_in_local]), bias (optional; expected only on one
        # rank since it is summed by the reduction), group, tp_mode.
        # Outputs: tensor [b / group size, ..., h_out].
        # Exceptions: AssertionError for other modes. Side effects: reduce-scatter collective.
        assert (
            tp_mode is TensorParallelLinearMode.REDUCE_SCATTER
        ), f"async communication in RowLinear only supports REDUCE_SCATTER, got {tp_mode}"

        if group is None:
            group = dist.distributed_c10d._get_default_group()

        ctx.use_bias = bias is not None
        ctx.group = group

        out = F.linear(tensor, weight, bias)

        if group.size() > 1:
            out = differentiable_reduce_scatter_sum(out, group=group)

        ctx.save_for_backward(tensor, weight)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        # [FUNCTION DOC]
        # Behavior: starts async all-gather of grad_output into the shared "allgather2" buffer; computes the local block of
        # the input gradient from local grad_output while waiting, then the before/after blocks from the gathered gradient;
        # computes weight grad from the full gathered grad_output and input, bias grad as its row sum.
        # Inputs: ctx, grad_output ([b / n, ..., h_out]). Outputs: (input grad [b, ..., h_in_local], weight grad, bias grad
        # or None, None, None).
        # Exceptions: None explicitly raised. Side effects: async all-gather; MemoryBuffer use.
        tensor, weight = ctx.saved_tensors
        group = ctx.group
        use_bias = ctx.use_bias

        handle: Optional[dist.Work] = None

        sharded_batch_size, *rest_size = grad_output.shape

        if group.size() == 1:
            total_grad_output = grad_output
        else:
            unsharded_batch_size = sharded_batch_size * group.size()

            total_grad_output = MemoryBuffer().get(
                "allgather2", (unsharded_batch_size, *rest_size), dtype=tensor.dtype
            )

            # Doing gather + slicing during the NeMo forward pass can make this tensor
            # not be contiguous. PyTorch only checks if the tensor is contiguous, and only
            # clones it if it's not contiguous:
            # https://github.com/pytorch/pytorch/blob/c47cf9bc7f9e02f649ab4ed53fe4d35732c92ab6/torch/_refs/__init__.py#L2761
            grad_output = grad_output.contiguous()

            handle = dist.all_gather_into_tensor(total_grad_output, grad_output, group=group, async_op=True)

        # total_grad_output: [b, s, h_out]
        # weight: [h_out, h_in/n]
        # total_grad_tensor: [b, s, h_in/n]
        # grad_output: [b/n, s, h_out]
        sharded_batch_size, *rest_size_grad_output = grad_output.shape
        rest_size_grad_tensor = rest_size_grad_output[:-1] + [weight.shape[1]]

        if group.size() == 1:
            total_grad_tensor = grad_output.matmul(weight)
        else:
            unsharded_batch_size = sharded_batch_size * group.size()
            total_grad_tensor = torch.empty(
                unsharded_batch_size,
                *rest_size_grad_tensor,
                device=grad_output.device,
                dtype=grad_output.dtype,
                requires_grad=False,
            )
            before_shard_grad_tensor, same_device_shard_grad_tensor, after_shard_grad_tensor = torch.split(
                total_grad_tensor,
                split_size_or_sections=[
                    sharded_batch_size * dist.get_rank(group),
                    sharded_batch_size,
                    sharded_batch_size * (group.size() - dist.get_rank(group) - 1),
                ],
                dim=0,
            )
            # compute local shard
            torch.mm(
                input=grad_output.view(-1, grad_output.shape[-1]),
                mat2=weight,
                out=same_device_shard_grad_tensor.view(-1, weight.shape[1]),
            )

            if handle is not None:
                handle.wait()

            before_shard_grad_output, _, after_shard_grad_output = torch.split(
                total_grad_output,
                split_size_or_sections=[
                    sharded_batch_size * dist.get_rank(group),
                    sharded_batch_size,
                    sharded_batch_size * (group.size() - dist.get_rank(group) - 1),
                ],
                dim=0,
            )

            # before shard compute
            if before_shard_grad_tensor.numel() > 0:
                torch.mm(
                    input=before_shard_grad_output.view(-1, before_shard_grad_output.shape[-1]),
                    mat2=weight,
                    out=before_shard_grad_tensor.view(-1, weight.shape[1]),
                )
            # after shard compute
            if after_shard_grad_tensor.numel() > 0:
                torch.mm(
                    input=after_shard_grad_output.view(-1, after_shard_grad_output.shape[-1]),
                    mat2=weight,
                    out=after_shard_grad_tensor.view(-1, weight.shape[1]),
                )

        # Convert the tensor shapes to 2D for execution compatibility
        tensor = tensor.contiguous()
        tensor_first_dims, tensor_last_dim = tensor.shape[:-1], tensor.shape[-1]
        tensor = tensor.view(math.prod(tensor_first_dims), tensor_last_dim)

        # Convert the tensor shapes to 2D for execution compatibility
        total_grad_output_first_dims, total_grad_output_last_dim = (
            total_grad_output.shape[:-1],
            total_grad_output.shape[-1],
        )
        total_grad_output = total_grad_output.view(math.prod(total_grad_output_first_dims), total_grad_output_last_dim)

        # TODO @thomasw21: This sounds like we don't have the optimal physical layout
        grad_weight = total_grad_output.t().matmul(tensor)
        grad_bias = total_grad_output.sum(dim=0) if use_bias else None

        return total_grad_tensor, grad_weight, grad_bias, None, None


def row_linear(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    group: dist.ProcessGroup,
    tp_mode: TensorParallelLinearMode,
    async_communication: bool,
):
    # [FUNCTION DOC]
    # Behavior: async_communication uses _RowLinearAsyncCommunication; else computes linear then differentiable
    # all-reduce (ALL_REDUCE) or differentiable reduce-scatter (REDUCE_SCATTER).
    # Inputs: input, weight, bias (optional), group, tp_mode, async_communication (bool).
    # Outputs: tensor. Exceptions: ValueError on unknown mode. Side effects: collectives.
    if async_communication:
        return _RowLinearAsyncCommunication.apply(input, weight, bias, group, tp_mode)

    out = F.linear(input, weight, bias)

    if tp_mode is TensorParallelLinearMode.ALL_REDUCE:
        out = differentiable_all_reduce_sum(out, group=group)
    elif tp_mode is TensorParallelLinearMode.REDUCE_SCATTER:
        out = differentiable_reduce_scatter_sum(out, group=group)
    else:
        raise ValueError(f"Got unexpected mode: {tp_mode}.")

    return out
