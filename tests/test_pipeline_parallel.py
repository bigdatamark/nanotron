# [MODULE DOC]
# Purpose: tests for nanotron pipeline parallelism: PipelineBlock rank assignment (build_and_set_rank), init_on_device_and_dtype, and the AllForwardAllBackward (AFAB) and OneForwardOneBackward (1F1B) pipeline engines, validated against a full reference model held on pipeline rank 0.
# Main contents: pytest entries and per-rank bodies for rank building, engine training on a chain model, on a model passing a non-differentiable tensor between stages, plain forward without an engine, and a diamond-shaped (branching) model on 4 ranks.
# Shared verification pattern: rank 0 builds an unsharded reference model, receives each stage's linear weight/bias via P2P recv_tensors (other ranks send theirs), runs each microbatch through the reference with loss divided by the number of microbatches and backpropagates; pipeline losses are gathered at rank 0 (tensors locally, TensorPointer entries received from the owning rank) and compared with tolerance 1e-6/1e-7; finally every rank asserts its params have grads and stage grads are sent to rank 0 and compared with the reference grads. Microbatch count is pp size plus 5.
# Import-time side effects: Calls available_gpus() during parametrization; instantiates engine objects for parametrization.
from typing import Union

import pytest
import torch
from helpers.dummy import DummyModel, dummy_infinite_data_loader
from helpers.utils import available_gpus, init_distributed, rerun_if_address_is_in_use
from nanotron import distributed as dist
from nanotron.models import init_on_device_and_dtype
from nanotron.parallel import ParallelContext
from nanotron.parallel.pipeline_parallel.block import PipelineBlock
from nanotron.parallel.pipeline_parallel.engine import (
    AllForwardAllBackwardPipelineEngine,
    OneForwardOneBackwardPipelineEngine,
    PipelineEngine,
)
from nanotron.parallel.pipeline_parallel.p2p import P2P
from nanotron.parallel.pipeline_parallel.tensor_pointer import TensorPointer
from torch import nn
from torch.nn import functional as F


@pytest.mark.skipif(available_gpus() < 2, reason="Testing build_and_set_rank requires at least 2 gpus")
@rerun_if_address_is_in_use()
def test_build_and_set_rank():
    # [FUNCTION DOC]
    # Summary: Pytest entry (needs 2 GPUs) running _test_build_and_set_rank with pp=2.
    # Inputs: None. Outputs: None. Exceptions: Propagates worker failures. Side effects: Spawns 2 processes.
    init_distributed(tp=1, dp=1, pp=2)(_test_build_and_set_rank)()


def _test_build_and_set_rank(parallel_context: ParallelContext):
    # [FUNCTION DOC]
    # HELPER: per-rank body of test_build_and_set_rank.
    # Behavior: Builds the helper DummyModel with a P2P over pp_pg, asserts it has one stage per pp rank, assigns stage i linear/activation to rank i and the loss to the last rank on CUDA, then asserts the local stage's blocks report the current rank and that only blocks assigned to this rank have a built pp_block attribute.
    # Inputs: parallel_context - ParallelContext.
    # Outputs: None.
    # Exceptions: AssertionError on failure.
    # Side effects: Allocates modules on CUDA; destroys process groups.
    device = torch.device("cuda")
    p2p = P2P(pg=parallel_context.pp_pg, device=device)
    model = DummyModel(p2p=p2p)

    # Set the ranks
    assert len(model.mlp) == parallel_context.pp_pg.size()
    with init_on_device_and_dtype(device):
        for pp_rank, non_linear in zip(range(parallel_context.pp_pg.size()), model.mlp):
            non_linear.linear.build_and_set_rank(pp_rank=pp_rank)
            non_linear.activation.build_and_set_rank(pp_rank=pp_rank)
        model.loss.build_and_set_rank(pp_rank=parallel_context.pp_pg.size() - 1)

    # Check that the ranks are set correctly
    current_pp_rank = dist.get_rank(parallel_context.pp_pg)
    assert model.mlp[current_pp_rank].linear.rank == current_pp_rank
    assert model.mlp[current_pp_rank].activation.rank == current_pp_rank

    # Check that blocks were built on the correct ranks
    for pp_rank, non_linear in zip(range(parallel_context.pp_pg.size()), model.mlp):
        if pp_rank == current_pp_rank:
            assert hasattr(non_linear.linear, "pp_block")
            assert hasattr(non_linear.activation, "pp_block")
        else:
            assert not hasattr(non_linear.linear, "pp_block")
            assert not hasattr(non_linear.activation, "pp_block")

    parallel_context.destroy()


@pytest.mark.skipif(available_gpus() < 1, reason="Testing test_init_on_device_and_dtype requires at least 1 gpus")
def test_init_on_device_and_dtype():
    # [FUNCTION DOC]
    # Summary: Asserts an nn.Linear created inside init_on_device_and_dtype with CUDA device 0 and bfloat16 has weights of that dtype and device. Requires 1 GPU.
    # Inputs: None. Outputs: None. Exceptions: AssertionError on failure. Side effects: GPU allocation.
    device = torch.device(type="cuda", index=0)
    with init_on_device_and_dtype(device=device, dtype=torch.bfloat16):
        model = nn.Linear(10, 10)

    assert model.weight.dtype == torch.bfloat16, "Model weight wasn't initialised with the correct dtype"
    assert model.weight.device == device, "Model weight wasn't initialised with the correct device"


@pytest.mark.skipif(available_gpus() < 2, reason="Testing AFAB requires at least 2 gpus")
@pytest.mark.parametrize(
    "pipeline_engine", [AllForwardAllBackwardPipelineEngine(), OneForwardOneBackwardPipelineEngine()]
)
@pytest.mark.parametrize("pp", list(range(2, min(4, available_gpus()) + 1)))
@rerun_if_address_is_in_use()
def test_pipeline_engine(pipeline_engine: PipelineEngine, pp: int):
    # [FUNCTION DOC]
    # Summary: Pytest entry (needs 2 GPUs) parametrized over the AFAB and 1F1B engines and pp from 2 to min(4, GPUs); runs _test_pipeline_engine.
    # Inputs: pipeline_engine - PipelineEngine instance; pp - int.
    # Outputs: None. Exceptions: Propagates worker failures. Side effects: Spawns pp processes.
    init_distributed(tp=1, dp=1, pp=pp)(_test_pipeline_engine)(pipeline_engine=pipeline_engine)


def _test_pipeline_engine(parallel_context: ParallelContext, pipeline_engine: PipelineEngine):
    # [FUNCTION DOC]
    # HELPER: per-rank body of test_pipeline_engine.
    # Behavior: Builds the helper DummyModel with stage i on rank i and loss on the last rank, plus a reference DummyModel fully on rank 0; synchronizes weights to the reference, draws pp+5 microbatches from dummy_infinite_data_loader, runs pipeline_engine.train_batch_iter without a grad accumulator, and applies the shared verification pattern (losses and per-stage gradients versus reference).
    # Inputs: parallel_context; pipeline_engine.
    # Outputs: None.
    # Exceptions: AssertionError on mismatch.
    # Side effects: P2P send/recv on pp_pg; backward passes; destroys process groups.
    device = torch.device("cuda")
    p2p = P2P(parallel_context.pp_pg, device=device)
    reference_rank = 0
    has_reference_model = dist.get_rank(parallel_context.pp_pg) == reference_rank
    current_pp_rank = dist.get_rank(parallel_context.pp_pg)

    # spawn model
    model = DummyModel(p2p=p2p)
    if has_reference_model:
        reference_model = DummyModel(p2p=p2p)

    # Set the ranks
    assert len(model.mlp) == parallel_context.pp_pg.size()
    with init_on_device_and_dtype(device):
        for pp_rank, non_linear in zip(range(parallel_context.pp_pg.size()), model.mlp):
            non_linear.linear.build_and_set_rank(pp_rank=pp_rank)
            non_linear.activation.build_and_set_rank(pp_rank=pp_rank)
        model.loss.build_and_set_rank(pp_rank=parallel_context.pp_pg.size() - 1)

        # build reference model
        if has_reference_model:
            for non_linear in reference_model.mlp:
                non_linear.linear.build_and_set_rank(pp_rank=reference_rank)
                non_linear.activation.build_and_set_rank(pp_rank=reference_rank)
            reference_model.loss.build_and_set_rank(pp_rank=reference_rank)

    # synchronize weights
    if has_reference_model:
        with torch.inference_mode():
            for pp_rank in range(parallel_context.pp_pg.size()):
                non_linear = model.mlp[pp_rank]
                reference_non_linear = reference_model.mlp[pp_rank]
                if pp_rank == current_pp_rank:
                    # We already have the weights locally
                    reference_non_linear.linear.pp_block.weight.data.copy_(non_linear.linear.pp_block.weight.data)
                    reference_non_linear.linear.pp_block.bias.data.copy_(non_linear.linear.pp_block.bias.data)
                    continue

                weight, bias = p2p.recv_tensors(num_tensors=2, from_rank=pp_rank)
                reference_non_linear.linear.pp_block.weight.data.copy_(weight.data)
                reference_non_linear.linear.pp_block.bias.data.copy_(bias.data)
    else:
        p2p.send_tensors(
            [model.mlp[current_pp_rank].linear.pp_block.weight, model.mlp[current_pp_rank].linear.pp_block.bias],
            to_rank=reference_rank,
        )

    # Get infinite dummy data iterator
    data_iterator = dummy_infinite_data_loader(pp_pg=parallel_context.pp_pg)  # First rank receives data

    # Have at least as many microbatches as PP size.
    n_micro_batches_per_batch = parallel_context.pp_pg.size() + 5

    batch = [next(data_iterator) for _ in range(n_micro_batches_per_batch)]
    losses = pipeline_engine.train_batch_iter(
        model, pg=parallel_context.pp_pg, batch=batch, nb_microbatches=n_micro_batches_per_batch, grad_accumulator=None
    )

    # Equivalent on the reference model
    if has_reference_model:
        reference_losses = []
        for micro_batch in batch:
            loss = reference_model(**micro_batch)
            loss /= n_micro_batches_per_batch
            loss.backward()
            reference_losses.append(loss.detach())

    # Gather loss in reference_rank
    if has_reference_model:
        _losses = []
    for loss in losses:
        if isinstance(loss["loss"], torch.Tensor):
            if has_reference_model:
                _losses.append(loss["loss"])
            else:
                p2p.send_tensors([loss["loss"]], to_rank=reference_rank)
        else:
            assert isinstance(loss["loss"], TensorPointer)
            if not has_reference_model:
                continue
            _losses.append(p2p.recv_tensors(num_tensors=1, from_rank=loss["loss"].group_rank)[0])
    if has_reference_model:
        losses = _losses

    # Check loss are the same as reference
    if has_reference_model:
        for loss, ref_loss in zip(losses, reference_losses):
            torch.testing.assert_close(loss, ref_loss, atol=1e-6, rtol=1e-7)

    # Check that gradient flows through the entire model
    for param in model.parameters():
        assert param.grad is not None

    # Check that gradient are the same as reference
    if has_reference_model:
        for pp_rank in range(parallel_context.pp_pg.size()):
            non_linear = model.mlp[pp_rank]
            reference_non_linear = reference_model.mlp[pp_rank]
            if pp_rank == current_pp_rank:
                # We already have the weights locally
                torch.testing.assert_close(
                    non_linear.linear.pp_block.weight.grad,
                    reference_non_linear.linear.pp_block.weight.grad,
                    atol=1e-6,
                    rtol=1e-7,
                )
                torch.testing.assert_close(
                    non_linear.linear.pp_block.bias.grad,
                    reference_non_linear.linear.pp_block.bias.grad,
                    atol=1e-6,
                    rtol=1e-7,
                )
                continue

            weight_grad, bias_grad = p2p.recv_tensors(num_tensors=2, from_rank=pp_rank)
            torch.testing.assert_close(
                weight_grad, reference_non_linear.linear.pp_block.weight.grad, atol=1e-6, rtol=1e-7
            )
            torch.testing.assert_close(bias_grad, reference_non_linear.linear.pp_block.bias.grad, atol=1e-6, rtol=1e-7)
    else:
        p2p.send_tensors(
            [
                model.mlp[current_pp_rank].linear.pp_block.weight.grad,
                model.mlp[current_pp_rank].linear.pp_block.bias.grad,
            ],
            to_rank=reference_rank,
        )

    parallel_context.destroy()


@pytest.mark.skipif(
    available_gpus() < 2,
    reason="Testing `test_pipeline_engine_with_tensor_that_does_not_require_grad` requires at least 2 gpus",
)
@pytest.mark.parametrize(
    "pipeline_engine", [AllForwardAllBackwardPipelineEngine(), OneForwardOneBackwardPipelineEngine()]
)
@pytest.mark.parametrize("pp", list(range(2, min(4, available_gpus()) + 1)))
@rerun_if_address_is_in_use()
def test_pipeline_engine_with_tensor_that_does_not_require_grad(pipeline_engine: PipelineEngine, pp: int):
    # [FUNCTION DOC]
    # Summary: Pytest entry (needs 2 GPUs) over both engines and pp sizes; runs _test_pipeline_engine_with_tensor_that_does_not_require_grad.
    # Inputs: pipeline_engine; pp - int. Outputs: None. Exceptions: Propagates worker failures. Side effects: Spawns pp processes.
    init_distributed(pp=pp, dp=1, tp=1)(_test_pipeline_engine_with_tensor_that_does_not_require_grad)(
        pipeline_engine=pipeline_engine
    )


def _test_pipeline_engine_with_tensor_that_does_not_require_grad(
    parallel_context: ParallelContext, pipeline_engine: PipelineEngine
):
    # [FUNCTION DOC]
    # HELPER: per-rank body verifying the engines handle a stage input that does not require grad.
    # Behavior: Defines a model whose pp+1 stages each take (x, y) and return a differentiable output plus y passed through unchanged; stage i lives on rank i, an extra final stage on the last rank, loss on the last rank. Builds a reference on rank 0, syncs weights following the stage-to-rank map, generates microbatches with random differentiable and non-differentiable tensors on rank 0 (TensorPointer elsewhere), trains with the engine and applies the shared verification pattern using the stage-to-rank map.
    # Inputs: parallel_context; pipeline_engine.
    # Outputs: None.
    # Exceptions: AssertionError on mismatch, including requires_grad checks inside the model forward.
    # Side effects: P2P communication; backward passes; destroys process groups.
    def activation(x: torch.Tensor, y: torch.Tensor):
        # [FUNCTION DOC]
        # HELPER: activation stage function.
        # Behavior: Returns a dictionary with output equal to sigmoid of x times y, and y unchanged.
        # Inputs: x, y - tensors of equal shape.
        # Outputs: dictionary with keys output and y.
        # Exceptions: None explicitly raised. Side effects: None.
        return {"output": F.sigmoid(x) * y, "y": y}

    class LinearWithDummyInput(nn.Linear):
        # [CLASS DOC]
        # HELPER: nn.Linear subclass that forwards an extra tensor untouched.
        # Purpose: lets a PipelineBlock carry a non-differentiable tensor y alongside the linear output.
        # Construction: in_features, out_features (bias enabled by default).
        def __init__(self, in_features, out_features):
            # [FUNCTION DOC]
            # Behavior: Initializes the parent Linear with the given sizes.
            # Inputs: in_features, out_features - ints. Outputs: None. Exceptions: None explicitly raised. Side effects: Allocates parameters.
            super().__init__(in_features=in_features, out_features=out_features)

        def forward(self, x: torch.Tensor, y: torch.Tensor):
            # [FUNCTION DOC]
            # Behavior: Applies the linear layer to x and returns a dictionary with output and y passed through.
            # Inputs: x - tensor with last dim in_features; y - any tensor.
            # Outputs: dictionary with keys output and y.
            # Exceptions: None explicitly raised. Side effects: None.
            return {"output": super().forward(x), "y": y}

    class DummyModelPassingNonDifferentiableTensor(nn.Module):
        # [CLASS DOC]
        # HELPER: test model that threads a non-differentiable tensor through pipeline stages.
        # Attributes: p2p - P2P instance; mlp - Sequential of pp size plus 1 ModuleDicts, each with a linear PipelineBlock (LinearWithDummyInput 10 to 10) and an activation PipelineBlock; loss - PipelineBlock summing its input.
        # Construction: p2p.
        def __init__(
            self,
            p2p: P2P,
        ):
            # [FUNCTION DOC]
            # Behavior: Stores p2p and builds the mlp stages and loss block described in the class doc (blocks are not yet assigned to ranks).
            # Inputs: p2p - P2P. Outputs: None. Exceptions: None explicitly raised. Side effects: None beyond module creation.
            super().__init__()
            self.p2p = p2p
            self.mlp = nn.Sequential(
                *(
                    nn.ModuleDict(
                        {
                            "linear": PipelineBlock(
                                p2p=p2p,
                                module_builder=LinearWithDummyInput,
                                module_kwargs={"in_features": 10, "out_features": 10},
                                module_input_keys={"x", "y"},
                                module_output_keys={"output", "y"},
                            ),
                            "activation": PipelineBlock(
                                p2p=p2p,
                                module_builder=lambda: activation,
                                module_kwargs={},
                                module_input_keys={"x", "y"},
                                module_output_keys={"output", "y"},
                            ),
                        }
                    )
                    for _ in range(p2p.pg.size() + 1)
                )
            )

            self.loss = PipelineBlock(
                p2p=p2p,
                module_builder=lambda: lambda x: x.sum(),
                module_kwargs={},
                module_input_keys={"x"},
                module_output_keys={"output"},
            )

        def forward(
            self,
            differentiable_tensor: Union[torch.Tensor, TensorPointer],
            non_differentiable_tensor: Union[torch.Tensor, TensorPointer],
        ):
            # [FUNCTION DOC]
            # Behavior: For each stage, runs linear then activation passing both tensors, asserting when local that the differentiable tensor requires grad and the other does not; then applies the loss block.
            # Inputs: differentiable_tensor, non_differentiable_tensor - tensors or TensorPointer.
            # Outputs: loss tensor or TensorPointer.
            # Exceptions: AssertionError if requires_grad flags are wrong.
            # Side effects: P2P communication via PipelineBlocks.
            for non_linear in self.mlp:
                linear_output = non_linear.linear(x=differentiable_tensor, y=non_differentiable_tensor)
                output = non_linear.activation(x=linear_output["output"], y=linear_output["y"])
                differentiable_tensor, non_differentiable_tensor = output["output"], output["y"]

                if isinstance(differentiable_tensor, torch.Tensor):
                    assert differentiable_tensor.requires_grad is True
                if isinstance(non_differentiable_tensor, torch.Tensor):
                    assert non_differentiable_tensor.requires_grad is False

            differentiable_tensor = self.loss(x=differentiable_tensor)["output"]
            return differentiable_tensor

    device = torch.device("cuda")
    p2p = P2P(parallel_context.pp_pg, device=device)
    reference_rank = 0
    current_pp_rank = dist.get_rank(parallel_context.pp_pg)
    has_reference_model = current_pp_rank == reference_rank

    # spawn model
    model = DummyModelPassingNonDifferentiableTensor(p2p=p2p)
    if has_reference_model:
        reference_model = DummyModelPassingNonDifferentiableTensor(p2p=p2p)

    # Set the ranks
    assert len(model.mlp) == parallel_context.pp_pg.size() + 1
    # An additional mlp is in the end
    mlp_index_pp_rank = [(i, i) for i in range(parallel_context.pp_pg.size())] + [
        (parallel_context.pp_pg.size(), parallel_context.pp_pg.size() - 1)
    ]

    with init_on_device_and_dtype(device):
        for (mlp_index, pp_rank), non_linear in zip(mlp_index_pp_rank, model.mlp):
            non_linear.linear.build_and_set_rank(pp_rank=pp_rank)
            non_linear.activation.build_and_set_rank(pp_rank=pp_rank)
        model.loss.build_and_set_rank(pp_rank=parallel_context.pp_pg.size() - 1)

        # build reference model
        if has_reference_model:
            for non_linear in reference_model.mlp:
                non_linear.linear.build_and_set_rank(pp_rank=reference_rank)
                non_linear.activation.build_and_set_rank(pp_rank=reference_rank)
            reference_model.loss.build_and_set_rank(pp_rank=reference_rank)

    # synchronize weights
    if has_reference_model:
        with torch.inference_mode():
            for (mlp_index, pp_rank) in mlp_index_pp_rank:
                non_linear = model.mlp[mlp_index]
                reference_non_linear = reference_model.mlp[mlp_index]
                if pp_rank == current_pp_rank:
                    # We already have the weights locally
                    reference_non_linear.linear.pp_block.weight.data.copy_(non_linear.linear.pp_block.weight.data)
                    reference_non_linear.linear.pp_block.bias.data.copy_(non_linear.linear.pp_block.bias.data)
                    continue

                weight, bias = p2p.recv_tensors(num_tensors=2, from_rank=pp_rank)
                reference_non_linear.linear.pp_block.weight.data.copy_(weight.data)
                reference_non_linear.linear.pp_block.bias.data.copy_(bias.data)
    else:
        for (mlp_index, pp_rank) in mlp_index_pp_rank:
            if pp_rank == current_pp_rank:
                p2p.send_tensors(
                    [model.mlp[mlp_index].linear.pp_block.weight, model.mlp[mlp_index].linear.pp_block.bias],
                    to_rank=reference_rank,
                )

    # Get infinite dummy data iterator
    def dummy_infinite_data_loader_with_non_differentiable_tensor(
        pp_pg: dist.ProcessGroup, dtype=torch.float, input_pp_rank=0
    ):
        # [FUNCTION DOC]
        # HELPER: infinite microbatch generator.
        # Behavior: Forever yields dictionaries with differentiable_tensor and non_differentiable_tensor, each a random 3x10 CUDA tensor on input_pp_rank or a TensorPointer to it on other ranks.
        # Inputs: pp_pg - process group (unused); dtype - default float; input_pp_rank - default 0.
        # Outputs: generator of dictionaries.
        # Exceptions: None explicitly raised. Side effects: Consumes CUDA RNG.
        micro_batch_size = 3
        # We assume the first linear is always built on the first rank.
        while True:
            yield {
                "differentiable_tensor": torch.randn(micro_batch_size, 10, dtype=dtype, device="cuda")
                if current_pp_rank == input_pp_rank
                else TensorPointer(group_rank=input_pp_rank),
                "non_differentiable_tensor": torch.randn(micro_batch_size, 10, dtype=dtype, device="cuda")
                if current_pp_rank == input_pp_rank
                else TensorPointer(group_rank=input_pp_rank),
            }

    data_iterator = dummy_infinite_data_loader_with_non_differentiable_tensor(
        pp_pg=parallel_context.pp_pg
    )  # First rank receives data

    # Have at least as many microbatches as PP size.
    n_micro_batches_per_batch = parallel_context.pp_pg.size() + 5

    batch = [next(data_iterator) for _ in range(n_micro_batches_per_batch)]
    losses = pipeline_engine.train_batch_iter(
        model, pg=parallel_context.pp_pg, batch=batch, nb_microbatches=n_micro_batches_per_batch, grad_accumulator=None
    )
    # Equivalent on the reference model
    if has_reference_model:
        reference_losses = []
        for micro_batch in batch:
            loss = reference_model(**micro_batch)
            loss /= n_micro_batches_per_batch
            loss.backward()
            reference_losses.append(loss.detach())

    # Gather loss in reference_rank
    if has_reference_model:
        _losses = []
    for loss in losses:
        if isinstance(loss["loss"], torch.Tensor):
            if has_reference_model:
                _losses.append(loss["loss"])
            else:
                p2p.send_tensors([loss["loss"]], to_rank=reference_rank)
        else:
            assert isinstance(loss["loss"], TensorPointer)
            if not has_reference_model:
                continue
            _losses.append(p2p.recv_tensors(num_tensors=1, from_rank=loss["loss"].group_rank)[0])
    if has_reference_model:
        losses = _losses

    # Check loss are the same as reference
    if has_reference_model:
        for loss, ref_loss in zip(losses, reference_losses):
            torch.testing.assert_close(loss, ref_loss, atol=1e-6, rtol=1e-7)

    # Check that gradient flows through the entire model
    for param in model.parameters():
        assert param.grad is not None

    # Check that gradient are the same as reference
    if has_reference_model:
        for (mlp_index, pp_rank) in mlp_index_pp_rank:
            non_linear = model.mlp[mlp_index]
            reference_non_linear = reference_model.mlp[mlp_index]
            if pp_rank == current_pp_rank:
                # We already have the weights locally
                torch.testing.assert_close(
                    non_linear.linear.pp_block.weight.grad,
                    reference_non_linear.linear.pp_block.weight.grad,
                    atol=1e-6,
                    rtol=1e-7,
                )
                torch.testing.assert_close(
                    non_linear.linear.pp_block.bias.grad,
                    reference_non_linear.linear.pp_block.bias.grad,
                    atol=1e-6,
                    rtol=1e-7,
                )
                continue

            weight_grad, bias_grad = p2p.recv_tensors(num_tensors=2, from_rank=pp_rank)
            torch.testing.assert_close(
                weight_grad, reference_non_linear.linear.pp_block.weight.grad, atol=1e-6, rtol=1e-7
            )
            torch.testing.assert_close(bias_grad, reference_non_linear.linear.pp_block.bias.grad, atol=1e-6, rtol=1e-7)
    else:
        for (mlp_index, pp_rank) in mlp_index_pp_rank:
            if pp_rank == current_pp_rank:
                p2p.send_tensors(
                    [model.mlp[mlp_index].linear.pp_block.weight.grad, model.mlp[mlp_index].linear.pp_block.bias.grad],
                    to_rank=reference_rank,
                )

    parallel_context.destroy()


@pytest.mark.parametrize("pp", list(range(2, min(4, available_gpus()) + 1)))
@rerun_if_address_is_in_use()
def test_pipeline_forward_without_engine(pp: int):
    # [FUNCTION DOC]
    # Summary: Pytest entry over pp from 2 to min(4, GPUs); runs _test_pipeline_forward_without_engine.
    # Inputs: pp - int. Outputs: None. Exceptions: Propagates worker failures. Side effects: Spawns pp processes.
    init_distributed(pp=pp, dp=1, tp=1)(_test_pipeline_forward_without_engine)()


def _test_pipeline_forward_without_engine(parallel_context: ParallelContext):
    # [FUNCTION DOC]
    # HELPER: per-rank body testing plain forward through PipelineBlocks without an engine.
    # Behavior: Defines a local DummyModel with one stage per rank (Linear then activation carrying y), builds it with stage i on rank i and a reference on rank 0, syncs weights, generates pp+5 microbatches, runs each through the model in inference mode, runs the reference, gathers losses at rank 0 (receiving TensorPointer-owned losses) and compares them. No backward is done.
    # Inputs: parallel_context.
    # Outputs: None.
    # Exceptions: AssertionError on mismatch.
    # Side effects: P2P communication; destroys process groups.
    def activation(x: torch.Tensor, y: torch.Tensor):
        # [FUNCTION DOC]
        # HELPER: activation stage function; returns output equal to sigmoid of x times y, plus y.
        # Inputs: x, y - tensors. Outputs: dictionary with output and y. Exceptions: None. Side effects: None.
        return {"output": F.sigmoid(x) * y, "y": y}

    class DummyModel(nn.Module):
        # [CLASS DOC]
        # HELPER: local chain model (shadows the imported DummyModel inside this function).
        # Attributes: p2p; mlp - Sequential of pp size ModuleDicts with a Linear(10,10) block and an activation block; loss - summing block.
        # Construction: p2p.
        def __init__(
            self,
            p2p: P2P,
        ):
            # [FUNCTION DOC]
            # Behavior: Stores p2p and builds the stage blocks and loss block.
            # Inputs: p2p - P2P. Outputs: None. Exceptions: None explicitly raised. Side effects: None.
            super().__init__()
            self.p2p = p2p
            self.mlp = nn.Sequential(
                *(
                    nn.ModuleDict(
                        {
                            "linear": PipelineBlock(
                                p2p=p2p,
                                module_builder=nn.Linear,
                                module_kwargs={"in_features": 10, "out_features": 10},
                                module_input_keys={"input"},
                                module_output_keys={"output"},
                            ),
                            "activation": PipelineBlock(
                                p2p=p2p,
                                module_builder=lambda: activation,
                                module_kwargs={},
                                module_input_keys={"x", "y"},
                                module_output_keys={"output", "y"},
                            ),
                        }
                    )
                    for _ in range(p2p.pg.size())
                )
            )

            self.loss = PipelineBlock(
                p2p=p2p,
                module_builder=lambda: lambda x: x.sum(),
                module_kwargs={},
                module_input_keys={"x"},
                module_output_keys={"output"},
            )

        def forward(
            self,
            differentiable_tensor: Union[torch.Tensor, TensorPointer],
            non_differentiable_tensor: Union[torch.Tensor, TensorPointer],
        ):
            # [FUNCTION DOC]
            # Behavior: For each stage applies the linear block to the differentiable tensor, then the activation with y; finally applies the loss block.
            # Inputs: differentiable_tensor, non_differentiable_tensor - tensors or TensorPointer.
            # Outputs: loss tensor or TensorPointer.
            # Exceptions: None explicitly raised. Side effects: P2P communication.
            for non_linear in self.mlp:
                differentiable_tensor = non_linear.linear(input=differentiable_tensor)["output"]
                output = non_linear.activation(x=differentiable_tensor, y=non_differentiable_tensor)
                differentiable_tensor, non_differentiable_tensor = output["output"], output["y"]
            differentiable_tensor = self.loss(x=differentiable_tensor)["output"]
            return differentiable_tensor

    device = torch.device("cuda")
    p2p = P2P(parallel_context.pp_pg, device=device)
    reference_rank = 0
    current_pp_rank = dist.get_rank(parallel_context.pp_pg)
    has_reference_model = current_pp_rank == reference_rank

    # spawn model
    model = DummyModel(p2p=p2p)
    if has_reference_model:
        reference_model = DummyModel(p2p=p2p)

    # Set the ranks
    assert len(model.mlp) == parallel_context.pp_pg.size()
    with init_on_device_and_dtype(device):
        for pp_rank, non_linear in zip(range(parallel_context.pp_pg.size()), model.mlp):
            non_linear.linear.build_and_set_rank(pp_rank=pp_rank)
            non_linear.activation.build_and_set_rank(pp_rank=pp_rank)
        model.loss.build_and_set_rank(pp_rank=parallel_context.pp_pg.size() - 1)

        # build reference model
        if has_reference_model:
            for non_linear in reference_model.mlp:
                non_linear.linear.build_and_set_rank(pp_rank=reference_rank)
                non_linear.activation.build_and_set_rank(pp_rank=reference_rank)
            reference_model.loss.build_and_set_rank(pp_rank=reference_rank)

    # synchronize weights
    if has_reference_model:
        with torch.inference_mode():
            for pp_rank in range(parallel_context.pp_pg.size()):
                non_linear = model.mlp[pp_rank]
                reference_non_linear = reference_model.mlp[pp_rank]
                if pp_rank == current_pp_rank:
                    # We already have the weights locally
                    reference_non_linear.linear.pp_block.weight.data.copy_(non_linear.linear.pp_block.weight.data)
                    reference_non_linear.linear.pp_block.bias.data.copy_(non_linear.linear.pp_block.bias.data)
                    continue

                weight, bias = p2p.recv_tensors(num_tensors=2, from_rank=pp_rank)
                reference_non_linear.linear.pp_block.weight.data.copy_(weight.data)
                reference_non_linear.linear.pp_block.bias.data.copy_(bias.data)
    else:
        p2p.send_tensors(
            [model.mlp[current_pp_rank].linear.pp_block.weight, model.mlp[current_pp_rank].linear.pp_block.bias],
            to_rank=reference_rank,
        )

    # Get infinite dummy data iterator
    def dummy_infinite_data_loader_with_non_differentiable_tensor(
        pp_pg: dist.ProcessGroup, dtype=torch.float, input_pp_rank=0
    ):
        # [FUNCTION DOC]
        # HELPER: infinite microbatch generator identical to the one in the non-differentiable engine test.
        # Inputs: pp_pg (unused); dtype default float; input_pp_rank default 0.
        # Outputs: generator of dictionaries with differentiable_tensor and non_differentiable_tensor.
        # Exceptions: None. Side effects: Consumes CUDA RNG.
        micro_batch_size = 3
        # We assume the first linear is always built on the first rank.
        while True:
            yield {
                "differentiable_tensor": torch.randn(micro_batch_size, 10, dtype=dtype, device="cuda")
                if current_pp_rank == input_pp_rank
                else TensorPointer(group_rank=input_pp_rank),
                "non_differentiable_tensor": torch.randn(micro_batch_size, 10, dtype=dtype, device="cuda")
                if current_pp_rank == input_pp_rank
                else TensorPointer(group_rank=input_pp_rank),
            }

    data_iterator = dummy_infinite_data_loader_with_non_differentiable_tensor(
        pp_pg=parallel_context.pp_pg
    )  # First rank receives data

    # Have at least as many microbatches as PP size.
    n_micro_batches_per_batch = parallel_context.pp_pg.size() + 5

    batch = [next(data_iterator) for _ in range(n_micro_batches_per_batch)]

    # Run the model
    losses = []
    for micro_batch in batch:
        with torch.inference_mode():
            loss = model(**micro_batch)
        losses.append(loss)

    # Equivalent on the reference model
    if has_reference_model:
        reference_losses = []
        for micro_batch in batch:
            loss = reference_model(**micro_batch)
            reference_losses.append(loss.detach())

    # Gather loss in reference_rank
    if has_reference_model:
        _losses = []
    for loss in losses:
        if isinstance(loss, torch.Tensor):
            if has_reference_model:
                _losses.append(loss)
            else:
                p2p.send_tensors([loss], to_rank=reference_rank)
        else:
            assert isinstance(loss, TensorPointer)
            if not has_reference_model:
                continue
            _losses.append(p2p.recv_tensors(num_tensors=1, from_rank=loss.group_rank)[0])
    if has_reference_model:
        losses = _losses

    # Check loss are the same as reference
    if has_reference_model:
        for loss, ref_loss in zip(losses, reference_losses):
            torch.testing.assert_close(loss, ref_loss, atol=1e-6, rtol=1e-7)

    parallel_context.destroy()


@pytest.mark.skipif(available_gpus() < 4, reason="Testing `test_pipeline_engine_diamond` requires at least 4 gpus")
@pytest.mark.parametrize(
    "pipeline_engine", [AllForwardAllBackwardPipelineEngine(), OneForwardOneBackwardPipelineEngine()]
)
@rerun_if_address_is_in_use()
def test_pipeline_engine_diamond(pipeline_engine: PipelineEngine):
    # [FUNCTION DOC]
    # Summary: Pytest entry (needs 4 GPUs) over both engines; runs _test_pipeline_engine_diamond with pp=4.
    # Inputs: pipeline_engine. Outputs: None. Exceptions: Propagates worker failures. Side effects: Spawns 4 processes.
    init_distributed(pp=4, dp=1, tp=1)(_test_pipeline_engine_diamond)(pipeline_engine=pipeline_engine)
    pass


def _test_pipeline_engine_diamond(parallel_context: ParallelContext, pipeline_engine: PipelineEngine):
    # [FUNCTION DOC]
    # HELPER: per-rank body testing engines on a branching (diamond) model.
    # Behavior: Builds DiamondModel with dense_bottom, dense_left, dense_right, dense_top on pp ranks 0 to 3 and loss on rank 3, plus a reference on rank 0; asserts pp size is 4; syncs weights, generates pp+5 microbatches of a random 3x10 x on rank 0, trains with the engine and applies the shared verification pattern per named submodule.
    # Inputs: parallel_context; pipeline_engine.
    # Outputs: None.
    # Exceptions: AssertionError on mismatch or wrong pp size.
    # Side effects: P2P communication; backward passes; destroys process groups.
    class DiamondModel(nn.Module):
        # [CLASS DOC]
        # HELPER: diamond-shaped test model.
        # Attributes: p2p; dense_bottom, dense_left, dense_right - ModuleDicts with Linear(10,10) and ReLU PipelineBlocks; dense_top - Bilinear(10,10 to 10) and ReLU blocks; loss - summing block.
        # Construction: p2p.
        def __init__(self, p2p: P2P):
            # [FUNCTION DOC]
            # Behavior: Stores p2p and builds the four dense ModuleDicts and loss block.
            # Inputs: p2p - P2P. Outputs: None. Exceptions: None explicitly raised. Side effects: None.
            super().__init__()
            self.p2p = p2p
            self.dense_bottom = nn.ModuleDict(
                {
                    "linear": PipelineBlock(
                        p2p=p2p,
                        module_builder=nn.Linear,
                        module_kwargs={"in_features": 10, "out_features": 10},
                        module_input_keys={"input"},
                        module_output_keys={"output"},
                    ),
                    "activation": PipelineBlock(
                        p2p=p2p,
                        module_builder=nn.ReLU,
                        module_kwargs={},
                        module_input_keys={"input"},
                        module_output_keys={"output"},
                    ),
                }
            )
            self.dense_left = nn.ModuleDict(
                {
                    "linear": PipelineBlock(
                        p2p=p2p,
                        module_builder=nn.Linear,
                        module_kwargs={"in_features": 10, "out_features": 10},
                        module_input_keys={"input"},
                        module_output_keys={"output"},
                    ),
                    "activation": PipelineBlock(
                        p2p=p2p,
                        module_builder=nn.ReLU,
                        module_kwargs={},
                        module_input_keys={"input"},
                        module_output_keys={"output"},
                    ),
                }
            )
            self.dense_right = nn.ModuleDict(
                {
                    "linear": PipelineBlock(
                        p2p=p2p,
                        module_builder=nn.Linear,
                        module_kwargs={"in_features": 10, "out_features": 10},
                        module_input_keys={"input"},
                        module_output_keys={"output"},
                    ),
                    "activation": PipelineBlock(
                        p2p=p2p,
                        module_builder=nn.ReLU,
                        module_kwargs={},
                        module_input_keys={"input"},
                        module_output_keys={"output"},
                    ),
                }
            )
            self.dense_top = nn.ModuleDict(
                {
                    "linear": PipelineBlock(
                        p2p=p2p,
                        module_builder=nn.Bilinear,
                        module_kwargs={"in1_features": 10, "in2_features": 10, "out_features": 10},
                        module_input_keys={"input1", "input2"},
                        module_output_keys={"output"},
                    ),
                    "activation": PipelineBlock(
                        p2p=p2p,
                        module_builder=nn.ReLU,
                        module_kwargs={},
                        module_input_keys={"input"},
                        module_output_keys={"output"},
                    ),
                }
            )

            self.loss = PipelineBlock(
                p2p=p2p,
                module_builder=lambda: lambda x: x.sum(),
                module_kwargs={},
                module_input_keys={"x"},
                module_output_keys={"output"},
            )

        def forward(self, x):
            # [FUNCTION DOC]
            # Behavior: Applies bottom (linear then ReLU), feeds the result to both left and right branches, combines them with the bilinear top block followed by ReLU, then applies the loss block.
            # Inputs: x - tensor or TensorPointer.
            # Outputs: loss tensor or TensorPointer.
            # Exceptions: None explicitly raised. Side effects: P2P communication.
            x = self.dense_bottom.activation(input=self.dense_bottom.linear(input=x)["output"])["output"]
            y = self.dense_left.activation(input=self.dense_left.linear(input=x)["output"])["output"]
            z = self.dense_right.activation(input=self.dense_right.linear(input=x)["output"])["output"]
            out = self.dense_top.activation(input=self.dense_top.linear(input1=y, input2=z)["output"])["output"]
            return self.loss(x=out)["output"]

    device = torch.device("cuda")
    p2p = P2P(parallel_context.pp_pg, device=device)
    reference_rank = 0
    current_pp_rank = dist.get_rank(parallel_context.pp_pg)
    has_reference_model = current_pp_rank == reference_rank

    # spawn model
    model = DiamondModel(p2p=p2p)
    if has_reference_model:
        reference_model = DiamondModel(p2p=p2p)

    # Set the ranks
    assert parallel_context.pp_pg.size() == len(
        [model.dense_bottom, model.dense_left, model.dense_right, model.dense_top]
    )
    assert parallel_context.pp_pg.size() == 4
    pp_rank_to_dense_name = ["dense_bottom", "dense_left", "dense_right", "dense_top"]
    with init_on_device_and_dtype(device):
        for pp_rank, module_name in enumerate(pp_rank_to_dense_name):
            non_linear = model.get_submodule(module_name)
            non_linear.linear.build_and_set_rank(pp_rank=pp_rank)
            non_linear.activation.build_and_set_rank(pp_rank=pp_rank)
        model.loss.build_and_set_rank(pp_rank=parallel_context.pp_pg.size() - 1)

        # build reference model
        if has_reference_model:
            for module_name in pp_rank_to_dense_name:
                non_linear = reference_model.get_submodule(module_name)
                non_linear.linear.build_and_set_rank(pp_rank=reference_rank)
                non_linear.activation.build_and_set_rank(pp_rank=reference_rank)
            reference_model.loss.build_and_set_rank(pp_rank=reference_rank)

    # synchronize weights
    if has_reference_model:
        with torch.inference_mode():
            for pp_rank, module_name in enumerate(pp_rank_to_dense_name):
                reference_non_linear = reference_model.get_submodule(module_name).linear.pp_block
                if pp_rank == current_pp_rank:
                    # We already have the weights locally
                    non_linear = model.get_submodule(module_name).linear.pp_block
                    reference_non_linear.weight.data.copy_(non_linear.weight.data)
                    reference_non_linear.bias.data.copy_(non_linear.bias.data)
                    continue

                weight, bias = p2p.recv_tensors(num_tensors=2, from_rank=pp_rank)
                reference_non_linear.weight.data.copy_(weight.data)
                reference_non_linear.bias.data.copy_(bias.data)
    else:
        non_linear = model.get_submodule(pp_rank_to_dense_name[current_pp_rank]).linear.pp_block
        p2p.send_tensors(
            [non_linear.weight, non_linear.bias],
            to_rank=reference_rank,
        )

    # Get infinite dummy data iterator
    def dummy_infinite_data_loader_with_non_differentiable_tensor(
        pp_pg: dist.ProcessGroup, dtype=torch.float, input_pp_rank=0
    ):
        # [FUNCTION DOC]
        # HELPER: infinite microbatch generator yielding a dictionary with x, a random 3x10 CUDA tensor on input_pp_rank or a TensorPointer elsewhere.
        # Inputs: pp_pg (unused); dtype default float; input_pp_rank default 0.
        # Outputs: generator of dictionaries.
        # Exceptions: None. Side effects: Consumes CUDA RNG.
        # KNOWN ISSUE: Name mentions a non-differentiable tensor but it yields only a single differentiable input (copy-paste naming).
        micro_batch_size = 3
        # We assume the first linear is always built on the first rank.
        while True:
            yield {
                "x": torch.randn(micro_batch_size, 10, dtype=dtype, device="cuda")
                if current_pp_rank == input_pp_rank
                else TensorPointer(group_rank=input_pp_rank),
            }

    data_iterator = dummy_infinite_data_loader_with_non_differentiable_tensor(
        pp_pg=parallel_context.pp_pg
    )  # First rank receives data

    # Have at least as many microbatches as PP size.
    n_micro_batches_per_batch = parallel_context.pp_pg.size() + 5

    batch = [next(data_iterator) for _ in range(n_micro_batches_per_batch)]
    losses = pipeline_engine.train_batch_iter(
        model, pg=parallel_context.pp_pg, batch=batch, nb_microbatches=n_micro_batches_per_batch, grad_accumulator=None
    )

    # Equivalent on the reference model
    if has_reference_model:
        reference_losses = []
        for micro_batch in batch:
            loss = reference_model(**micro_batch)
            loss /= n_micro_batches_per_batch
            loss.backward()
            reference_losses.append(loss.detach())

    # Gather loss in reference_rank
    if has_reference_model:
        _losses = []
    for loss in losses:
        if isinstance(loss["loss"], torch.Tensor):
            if has_reference_model:
                _losses.append(loss["loss"])
            else:
                p2p.send_tensors([loss["loss"]], to_rank=reference_rank)
        else:
            assert isinstance(loss["loss"], TensorPointer)
            if not has_reference_model:
                continue
            _losses.append(p2p.recv_tensors(num_tensors=1, from_rank=loss["loss"].group_rank)[0])
    if has_reference_model:
        losses = _losses

    # Check loss are the same as reference
    if has_reference_model:
        for loss, ref_loss in zip(losses, reference_losses):
            torch.testing.assert_close(loss, ref_loss, atol=1e-6, rtol=1e-7)

    # Check that gradient flows through the entire model
    for param in model.parameters():
        assert param.grad is not None

    # Check that gradient are the same as reference
    if has_reference_model:
        for pp_rank, module_name in enumerate(pp_rank_to_dense_name):
            reference_non_linear = reference_model.get_submodule(module_name).linear.pp_block
            if pp_rank == current_pp_rank:
                # We already have the weights locally
                non_linear = model.get_submodule(module_name).linear.pp_block
                torch.testing.assert_close(
                    non_linear.weight.grad,
                    reference_non_linear.weight.grad,
                    atol=1e-6,
                    rtol=1e-7,
                )
                torch.testing.assert_close(
                    non_linear.bias.grad,
                    reference_non_linear.bias.grad,
                    atol=1e-6,
                    rtol=1e-7,
                )
                continue

            weight_grad, bias_grad = p2p.recv_tensors(num_tensors=2, from_rank=pp_rank)
            torch.testing.assert_close(weight_grad, reference_non_linear.weight.grad, atol=1e-6, rtol=1e-7)
            torch.testing.assert_close(bias_grad, reference_non_linear.bias.grad, atol=1e-6, rtol=1e-7)
    else:
        non_linear = model.get_submodule(pp_rank_to_dense_name[current_pp_rank]).linear.pp_block
        p2p.send_tensors(
            [non_linear.weight.grad, non_linear.bias.grad],
            to_rank=reference_rank,
        )

    parallel_context.destroy()
