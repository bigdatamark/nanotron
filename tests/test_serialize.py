# [MODULE DOC]
# Purpose: round-trip tests for nanotron.serialize: saving/loading model weights, optimizer states (NamedOptimizer, ZeroDistributedOptimizer, OptimizerFromGradientAccumulator with FP32 grad accumulator), random states, and TensorMetadata string serialization, across all 3D (tp, dp, pp) layouts fitting in up to 4 GPUs.
# Main contents: pytest entries each launching an underscore-prefixed per-rank body via init_distributed with a TestContext providing an auto-removed temp directory.
# Import-time side effects: Calls available_gpus() and get_all_3d_configurations during parametrization.
# Shared training pattern (used by several bodies below): builds the dummy model, pulls a minibatch from dummy_infinite_data_loader 3 times, runs AllForwardAllBackwardPipelineEngine.train_batch_iter with one microbatch, syncs tied gradients, then optimizer step and zero_grad.
import pytest
import torch
from helpers.context import TestContext
from helpers.dummy import dummy_infinite_data_loader, init_dummy_model
from helpers.utils import (
    available_gpus,
    get_all_3d_configurations,
    init_distributed,
    is_dict_equal,
    rerun_if_address_is_in_use,
)
from nanotron import distributed as dist
from nanotron.constants import CHECKPOINT_VERSION
from nanotron.optim.gradient_accumulator import FP32GradientAccumulator
from nanotron.optim.named_optimizer import NamedOptimizer
from nanotron.optim.optimizer_from_gradient_accumulator import (
    OptimizerFromGradientAccumulator,
)
from nanotron.optim.zero import ZeroDistributedOptimizer
from nanotron.parallel import ParallelContext
from nanotron.parallel.pipeline_parallel.engine import (
    AllForwardAllBackwardPipelineEngine,
)
from nanotron.parallel.sharded_parameters import SplitConfig, create_sharded_parameter_from_config
from nanotron.parallel.tied_parameters import sync_tied_weights_gradients
from nanotron.random import RandomStates, get_current_random_state, get_synced_random_state
from nanotron.serialize import (
    load_optimizer,
    load_random_states,
    load_weights,
    save_optimizer,
    save_random_states,
    save_weights,
)
from nanotron.serialize.metadata import TensorMetadata
from torch.nn.parallel import DistributedDataParallel


def test_save_and_load_with_changed_topolgy():
    # [FUNCTION DOC]
    # Summary: Placeholder for a topology-change save/load test; returns immediately.
    # Inputs: None. Outputs: None. Exceptions: None. Side effects: None.
    # KNOWN ISSUE: Test body is empty (always passes); name also misspells "topology".
    # TODO @thomasw21: We want to be able to support a change of topology mechanism
    return


@pytest.mark.parametrize(
    "tp,dp,pp",
    [
        pytest.param(*all_3d_configs)
        for gpus in range(1, min(available_gpus(), 4) + 1)
        for all_3d_configs in get_all_3d_configurations(gpus)
    ],
)
@rerun_if_address_is_in_use()
def test_save_and_load_model(tp: int, dp: int, pp: int):
    # [FUNCTION DOC]
    # Summary: Pytest entry over all 3D configs; launches _test_save_and_load_model with a fresh TestContext.
    # Inputs: tp, dp, pp - ints. Outputs: None. Exceptions: Propagates worker failures. Side effects: Spawns processes; temp dirs.
    test_context = TestContext()
    # We use DP=2 as we're interested in testing that one
    init_distributed(tp=tp, dp=dp, pp=pp)(_test_save_and_load_model)(test_context=test_context)


def _test_save_and_load_model(parallel_context: ParallelContext, test_context: TestContext):
    # [FUNCTION DOC]
    # HELPER: per-rank body of test_save_and_load_model.
    # Behavior: Saves a dummy model's weights to a temp folder, builds a new dummy model and asserts its state dict differs (unless empty on this rank), loads the weights into it and asserts state dicts are exactly equal. Destroys the context.
    # Inputs: parallel_context; test_context - TestContext.
    # Outputs: None.
    # Exceptions: AssertionError on mismatch.
    # Side effects: Writes/reads checkpoint files; collectives inside save/load; destroys process groups.
    model = init_dummy_model(parallel_context=parallel_context)
    store_folder = test_context.get_auto_remove_tmp_dir()

    # Save
    save_weights(model=model, parallel_context=parallel_context, root_folder=store_folder)

    # Load
    new_model = init_dummy_model(parallel_context=parallel_context)

    # Check that the newly initialised model isn't the same.
    match, msg = is_dict_equal(new_model.state_dict(), model.state_dict())
    if len(model.state_dict()) == 0:
        # Edge case where there's no parameters/buffers stored in the model.
        pass
    else:
        assert not match, "Newly initialised model should not match."

    load_weights(model=new_model, parallel_context=parallel_context, root_folder=store_folder)

    # Assert the weights are exactly the same after loading
    match, msg = is_dict_equal(new_model.state_dict(), model.state_dict())
    assert match, msg

    parallel_context.destroy()


@pytest.mark.parametrize(
    "tp,dp,pp",
    [
        pytest.param(*all_3d_configs)
        for gpus in range(1, min(available_gpus(), 4) + 1)
        for all_3d_configs in get_all_3d_configurations(gpus)
    ],
)
@rerun_if_address_is_in_use()
def test_save_and_load_optimizer(tp: int, dp: int, pp: int):
    # [FUNCTION DOC]
    # Summary: Pytest entry over all 3D configs; skips pp greater than 1; launches _test_save_and_load_optimizer.
    # Inputs: tp, dp, pp - ints. Outputs: None. Exceptions: pytest skip; propagates worker failures. Side effects: Spawns processes.
    test_context = TestContext()
    if pp > 1:
        pytest.skip("Pipeline parallelism not supported for this test yet")
    init_distributed(tp=tp, dp=dp, pp=pp)(_test_save_and_load_optimizer)(test_context=test_context)


def _test_save_and_load_optimizer(parallel_context: ParallelContext, test_context: TestContext):
    # [FUNCTION DOC]
    # HELPER: per-rank body of test_save_and_load_optimizer.
    # Behavior: Trains a dummy model 3 steps with a NamedOptimizer over AdamW (shared pattern), saves the optimizer, barriers, builds a fresh optimizer and asserts it differs (if state non-empty), loads and asserts exact equality. Then loads into another fresh optimizer with map_location cpu and asserts every tensor state value equals the GPU value, is on CPU (except the step key) while the original is on CUDA, and non-tensor values are equal.
    # Inputs: parallel_context; test_context.
    # Outputs: None.
    # Exceptions: AssertionError on mismatch.
    # Side effects: Checkpoint file I/O; world barrier; destroys process groups.
    store_folder = test_context.get_auto_remove_tmp_dir()
    model = init_dummy_model(parallel_context=parallel_context)
    optimizer = NamedOptimizer(
        named_params_or_groups=model.named_parameters(),
        optimizer_builder=lambda params: torch.optim.AdamW(params),
    )

    # Train in order to update the optimizer step a few times
    data_loader = iter(dummy_infinite_data_loader(pp_pg=parallel_context.pp_pg))
    nb_optim_steps = 3
    pipeline_engine = AllForwardAllBackwardPipelineEngine()
    for _ in range(nb_optim_steps):
        minibatch = next(data_loader)
        _ = pipeline_engine.train_batch_iter(
            model=model, pg=parallel_context.pp_pg, batch=[minibatch], nb_microbatches=1, grad_accumulator=None
        )
        # Manually sync tied parameters
        sync_tied_weights_gradients(module=model, parallel_context=parallel_context, grad_accumulator=None)
        # Optimizer steps
        optimizer.step()
        optimizer.zero_grad()

    # Save optimizer
    save_optimizer(optimizer=optimizer, parallel_context=parallel_context, root_folder=store_folder)
    dist.barrier(parallel_context.world_pg)

    # Generate a new optimizer
    new_optimizer = NamedOptimizer(
        named_params_or_groups=model.named_parameters(),
        optimizer_builder=lambda params: torch.optim.AdamW(params),
    )

    # Check that the newly initialised optimizer isn't the same.
    match, msg = is_dict_equal(optimizer.state_dict(), new_optimizer.state_dict())
    if len(optimizer.state_dict()["state"]) == 0:
        # Edge case where there's no state stored in the optimizer.
        pass
    else:
        assert not match, "Newly initialised optimizer should not match."
    load_optimizer(optimizer=new_optimizer, parallel_context=parallel_context, root_folder=store_folder)

    # Assert the optimizer states are exactly the same after loading.
    match, msg = is_dict_equal(optimizer.state_dict(), new_optimizer.state_dict())
    assert match, msg

    # Test loading optimizer states to CPU
    cpu_optimizer = NamedOptimizer(
        named_params_or_groups=model.named_parameters(),
        optimizer_builder=lambda params: torch.optim.AdamW(params),
    )

    # Load optimizer states to CPU
    load_optimizer(
        optimizer=cpu_optimizer, parallel_context=parallel_context, root_folder=store_folder, map_location="cpu"
    )

    # Get state dicts
    gpu_state = optimizer.state_dict()
    cpu_state = cpu_optimizer.state_dict()

    # Check that states match except for device
    for param_id in gpu_state["state"]:
        for key, gpu_value in gpu_state["state"][param_id].items():
            cpu_value = cpu_state["state"][param_id][key]
            if isinstance(gpu_value, torch.Tensor):
                assert torch.equal(gpu_value.cpu(), cpu_value), f"Values don't match for param {param_id}, key {key}"
                if key != "step":  # Skip device checks for 'step' key
                    assert (
                        cpu_value.device.type == "cpu"
                    ), f"CPU optimizer state should be on CPU for param {param_id}, key {key}"
                    assert (
                        gpu_value.device.type == "cuda"
                    ), f"GPU optimizer state should be on CUDA for param {param_id}, key {key}"
            else:
                assert gpu_value == cpu_value, f"Non-tensor values don't match for param {param_id}, key {key}"

    parallel_context.destroy()


@pytest.mark.parametrize(
    "tp,dp,pp",
    [
        pytest.param(*all_3d_configs)
        for gpus in range(1, min(available_gpus(), 4) + 1)
        for all_3d_configs in get_all_3d_configurations(gpus)
    ],
)
@rerun_if_address_is_in_use()
def test_save_zero_optimizer_and_load_optimizer(tp: int, dp: int, pp: int):
    # [FUNCTION DOC]
    # Summary: Pytest entry over all 3D configs; launches _test_save_zero_optimizer_and_load_optimizer.
    # Inputs: tp, dp, pp - ints. Outputs: None. Exceptions: Propagates worker failures. Side effects: Spawns processes.
    test_context = TestContext()
    # We use DP=2 as we're interested in testing that one
    init_distributed(tp=tp, dp=dp, pp=pp)(_test_save_zero_optimizer_and_load_optimizer)(test_context=test_context)


def _test_save_zero_optimizer_and_load_optimizer(parallel_context: ParallelContext, test_context: TestContext):
    # [FUNCTION DOC]
    # HELPER: per-rank body of test_save_zero_optimizer_and_load_optimizer.
    # Behavior: Same as the NamedOptimizer round trip but with a ZeroDistributedOptimizer sharded over dp_pg for both the saved and the reloaded optimizer; asserts the reloaded state dict exactly equals the original.
    # Inputs: parallel_context; test_context.
    # Outputs: None.
    # Exceptions: AssertionError on mismatch.
    # Side effects: Checkpoint I/O; barrier; destroys process groups.
    store_folder = test_context.get_auto_remove_tmp_dir()
    model = init_dummy_model(parallel_context=parallel_context)
    optimizer = ZeroDistributedOptimizer(
        named_params_or_groups=model.named_parameters(),
        optimizer_builder=lambda named_param_groups: NamedOptimizer(
            named_params_or_groups=named_param_groups,
            optimizer_builder=lambda param_groups: torch.optim.AdamW(param_groups),
        ),
        dp_pg=parallel_context.dp_pg,
    )

    # Train in order to update the optimizer step a few times
    data_loader = iter(dummy_infinite_data_loader(pp_pg=parallel_context.pp_pg))
    nb_optim_steps = 3
    pipeline_engine = AllForwardAllBackwardPipelineEngine()
    for _ in range(nb_optim_steps):
        minibatch = next(data_loader)
        _ = pipeline_engine.train_batch_iter(
            model=model, pg=parallel_context.pp_pg, batch=[minibatch], nb_microbatches=1, grad_accumulator=None
        )
        # Manually sync tied parameters
        sync_tied_weights_gradients(module=model, parallel_context=parallel_context, grad_accumulator=None)
        # Optimizer steps
        optimizer.step()
        optimizer.zero_grad()

    # Save optimizer
    save_optimizer(optimizer=optimizer, parallel_context=parallel_context, root_folder=store_folder)
    dist.barrier(parallel_context.world_pg)

    # Generate a new optimizer
    new_optimizer = ZeroDistributedOptimizer(
        named_params_or_groups=model.named_parameters(),
        optimizer_builder=lambda named_param_groups: NamedOptimizer(
            named_params_or_groups=named_param_groups,
            optimizer_builder=lambda param_groups: torch.optim.AdamW(param_groups),
        ),
        dp_pg=parallel_context.dp_pg,
    )

    # Check that the newly initialised optimizer isn't the same.
    match, msg = is_dict_equal(optimizer.state_dict(), new_optimizer.state_dict())
    if len(optimizer.state_dict()["state"]) == 0:
        # Edge case where there's no state stored in the optimizer.
        pass
    else:
        assert not match, "Newly initialised optimizer should not match."

    load_optimizer(optimizer=new_optimizer, parallel_context=parallel_context, root_folder=store_folder)

    # Assert the optimizer states are exactly the same after loading.
    match, msg = is_dict_equal(optimizer.state_dict(), new_optimizer.state_dict())
    assert match, msg

    parallel_context.destroy()


@pytest.mark.skip(reason="Assumption that zero and non zero optimizer have the same serialization format doesn't hold")
@pytest.mark.parametrize(
    "tp,dp,pp",
    [
        pytest.param(*all_3d_configs)
        for gpus in range(1, min(available_gpus(), 4) + 1)
        for all_3d_configs in get_all_3d_configurations(gpus)
    ],
)
@rerun_if_address_is_in_use()
def test_save_zero_optimizer_and_load_data_parallel_optimizer(tp: int, dp: int, pp: int):
    # [FUNCTION DOC]
    # Summary: Skipped pytest entry (zero and non-zero formats differ); would launch _test_save_zero_optimizer_and_load_data_parallel_optimizer.
    # Inputs: tp, dp, pp - ints. Outputs: None. Exceptions: Propagates worker failures. Side effects: Spawns processes.
    test_context = TestContext()
    # We use DP=2 as we're interested in testing that one
    init_distributed(tp=tp, dp=dp, pp=pp)(_test_save_zero_optimizer_and_load_data_parallel_optimizer)(
        test_context=test_context
    )


def _test_save_zero_optimizer_and_load_data_parallel_optimizer(
    parallel_context: ParallelContext, test_context: TestContext
):
    # [FUNCTION DOC]
    # HELPER: per-rank body for saving a ZeRO optimizer and loading into a plain NamedOptimizer.
    # Behavior: Trains with ZeroDistributedOptimizer, saves, builds a NamedOptimizer, asserts it differs, loads the checkpoint into it; no post-load comparison is made.
    # Inputs: parallel_context; test_context.
    # Outputs: None.
    # Exceptions: AssertionError if the fresh optimizer matches; load errors propagate.
    # Side effects: Checkpoint I/O; barrier; destroys process groups.
    # KNOWN ISSUE: No assertion after loading (TODO), so the test cannot detect incorrect loading.
    store_folder = test_context.get_auto_remove_tmp_dir()
    model = init_dummy_model(parallel_context=parallel_context)
    optimizer = ZeroDistributedOptimizer(
        named_params_or_groups=model.named_parameters(),
        optimizer_builder=lambda named_param_groups: NamedOptimizer(
            named_params_or_groups=named_param_groups,
            optimizer_builder=lambda param_groups: torch.optim.AdamW(param_groups),
        ),
        dp_pg=parallel_context.dp_pg,
    )

    # Train in order to update the optimizer step a few times
    data_loader = iter(dummy_infinite_data_loader(pp_pg=parallel_context.pp_pg))
    nb_optim_steps = 3
    pipeline_engine = AllForwardAllBackwardPipelineEngine()
    for _ in range(nb_optim_steps):
        minibatch = next(data_loader)
        _ = pipeline_engine.train_batch_iter(
            model=model, pg=parallel_context.pp_pg, batch=[minibatch], nb_microbatches=1, grad_accumulator=None
        )
        # Manually sync tied parameters
        sync_tied_weights_gradients(module=model, parallel_context=parallel_context, grad_accumulator=None)
        # Optimizer steps
        optimizer.step()
        optimizer.zero_grad()

    # Save optimizer
    save_optimizer(optimizer=optimizer, parallel_context=parallel_context, root_folder=store_folder)
    dist.barrier(parallel_context.world_pg)

    # Generate a new optimizer
    new_optimizer = NamedOptimizer(
        named_params_or_groups=model.named_parameters(),
        optimizer_builder=lambda params: torch.optim.AdamW(params),
    )

    # Check that the newly initialised optimizer isn't the same.
    match, msg = is_dict_equal(optimizer.state_dict(), new_optimizer.state_dict())
    if len(optimizer.state_dict()["state"]) == 0:
        # Edge case where there's no state stored in the optimizer.
        pass
    else:
        assert not match, "Newly initialised optimizer should not match."

    load_optimizer(optimizer=new_optimizer, parallel_context=parallel_context, root_folder=store_folder)

    # TODO @thomasw21: Compare zero optimizer with non zero
    parallel_context.destroy()


@pytest.mark.skip(reason="Assumption that zero and non zero optimizer have the same serialization format doesn't hold")
@pytest.mark.parametrize(
    "tp,dp,pp",
    [
        pytest.param(*all_3d_configs)
        for gpus in range(1, min(available_gpus(), 4) + 1)
        for all_3d_configs in get_all_3d_configurations(gpus)
    ],
)
@rerun_if_address_is_in_use()
def test_save_data_parallel_optimizer_and_load_zero_optimizer(tp: int, dp: int, pp: int):
    # [FUNCTION DOC]
    # Summary: Skipped pytest entry; would launch _test_save_data_parallel_optimizer_and_load_zero_optimizer.
    # Inputs: tp, dp, pp - ints. Outputs: None. Exceptions: Propagates worker failures. Side effects: Spawns processes.
    test_context = TestContext()
    # We use DP=2 as we're interested in testing that one
    init_distributed(tp=tp, dp=dp, pp=pp)(_test_save_data_parallel_optimizer_and_load_zero_optimizer)(
        test_context=test_context
    )


def _test_save_data_parallel_optimizer_and_load_zero_optimizer(
    parallel_context: ParallelContext, test_context: TestContext
):
    # [FUNCTION DOC]
    # HELPER: per-rank body for saving a plain NamedOptimizer and loading into a ZeRO optimizer.
    # Behavior: Trains 3 steps with NamedOptimizer (without tied-gradient sync), saves, builds a ZeroDistributedOptimizer, asserts it differs, and loads; no post-load comparison.
    # Inputs: parallel_context; test_context.
    # Outputs: None.
    # Exceptions: AssertionError if the fresh optimizer matches; load errors propagate.
    # Side effects: Checkpoint I/O; barrier; destroys process groups.
    # KNOWN ISSUE: No assertion after loading (TODO), and unlike the other bodies tied gradients are not synced during training.
    store_folder = test_context.get_auto_remove_tmp_dir()
    model = init_dummy_model(parallel_context=parallel_context)
    optimizer = NamedOptimizer(
        named_params_or_groups=model.named_parameters(),
        optimizer_builder=lambda params: torch.optim.AdamW(params),
    )

    # Train in order to update the optimizer step a few times
    data_loader = iter(dummy_infinite_data_loader(pp_pg=parallel_context.pp_pg))
    nb_optim_steps = 3
    pipeline_engine = AllForwardAllBackwardPipelineEngine()
    for _ in range(nb_optim_steps):
        minibatch = next(data_loader)
        _ = pipeline_engine.train_batch_iter(
            model=model, pg=parallel_context.pp_pg, batch=[minibatch], nb_microbatches=1, grad_accumulator=None
        )
        optimizer.step()
        optimizer.zero_grad()

    # Save optimizer
    save_optimizer(optimizer=optimizer, parallel_context=parallel_context, root_folder=store_folder)
    dist.barrier(parallel_context.world_pg)

    # Generate a new optimizer
    new_optimizer = ZeroDistributedOptimizer(
        named_params_or_groups=model.named_parameters(),
        optimizer_builder=lambda named_param_groups: NamedOptimizer(
            named_params_or_groups=named_param_groups,
            optimizer_builder=lambda param_groups: torch.optim.AdamW(param_groups),
        ),
        dp_pg=parallel_context.dp_pg,
    )

    # Check that the newly initialised optimizer isn't the same.
    match, msg = is_dict_equal(optimizer.state_dict(), new_optimizer.state_dict())
    if len(optimizer.state_dict()["state"]) == 0:
        # Edge case where there's no state stored in the optimizer.
        pass
    else:
        assert not match, "Newly initialised optimizer should not match."

    load_optimizer(optimizer=new_optimizer, parallel_context=parallel_context, root_folder=store_folder)

    # TODO @thomasw21: Compare zero optimizer with non zero
    parallel_context.destroy()


@pytest.mark.parametrize(
    "tp,dp,pp",
    [
        pytest.param(*all_3d_configs)
        for gpus in range(1, min(available_gpus(), 4) + 1)
        for all_3d_configs in get_all_3d_configurations(gpus)
    ],
)
@rerun_if_address_is_in_use()
def test_save_optimizer_with_additional_state_dict_keys(tp: int, dp: int, pp: int):
    # [FUNCTION DOC]
    # Summary: Pytest entry over all 3D configs; launches _test_save_optimizer_with_additional_state_dict_keys.
    # Inputs: tp, dp, pp - ints. Outputs: None. Exceptions: Propagates worker failures. Side effects: Spawns processes.
    test_context = TestContext()
    # We use DP=2 as we're interested in testing that one
    init_distributed(tp=tp, dp=dp, pp=pp)(_test_save_optimizer_with_additional_state_dict_keys)(
        test_context=test_context
    )


def _test_save_optimizer_with_additional_state_dict_keys(parallel_context: ParallelContext, test_context: TestContext):
    # [FUNCTION DOC]
    # HELPER: per-rank body of test_save_optimizer_with_additional_state_dict_keys.
    # Behavior: Builds a float16 dummy model (unwrapping DDP if present) and an OptimizerFromGradientAccumulator with an FP32GradientAccumulator and inner NamedOptimizer/AdamW; asserts it reports additional state dict keys. Trains 3 steps passing the grad accumulator to the engine and tied-grad sync. Saves, builds a fresh equivalent optimizer, asserts it differs when state is non-empty, loads, and asserts the "state" section, the full state dict, and the grad accumulator's per-parameter buffers dictionaries are all exactly equal.
    # Inputs: parallel_context; test_context.
    # Outputs: None.
    # Exceptions: AssertionError on mismatch.
    # Side effects: Checkpoint I/O; barrier; destroys process groups.
    dtype = torch.float16
    store_folder = test_context.get_auto_remove_tmp_dir()
    model = init_dummy_model(parallel_context=parallel_context, dtype=dtype)

    if isinstance(model, DistributedDataParallel):
        # Remove the annoying "module." prefix
        unwrapped_model = model.module
    else:
        unwrapped_model = model

    named_parameters = list(unwrapped_model.named_parameters())

    optimizer = OptimizerFromGradientAccumulator(
        gradient_accumulator_builder=lambda named_params: FP32GradientAccumulator(named_parameters=named_params),
        named_params_or_groups=named_parameters,
        optimizer_builder=lambda named_param_groups: NamedOptimizer(
            named_params_or_groups=named_param_groups,
            optimizer_builder=lambda param_groups: torch.optim.AdamW(param_groups),
        ),
    )
    grad_accumulator = optimizer.gradient_accumulator

    assert len(optimizer.state_dict_additional_keys()) > 0

    # Train in order to update the optimizer step a few times
    data_loader = iter(dummy_infinite_data_loader(pp_pg=parallel_context.pp_pg, dtype=dtype))
    nb_optim_steps = 3
    pipeline_engine = AllForwardAllBackwardPipelineEngine()
    for _ in range(nb_optim_steps):
        minibatch = next(data_loader)
        _ = pipeline_engine.train_batch_iter(
            model=model,
            pg=parallel_context.pp_pg,
            batch=[minibatch],
            nb_microbatches=1,
            grad_accumulator=grad_accumulator,
        )
        # Manually sync tied parameters
        sync_tied_weights_gradients(
            module=unwrapped_model, parallel_context=parallel_context, grad_accumulator=grad_accumulator
        )
        # Optimizer steps
        optimizer.step()
        optimizer.zero_grad()

    # Save optimizer
    save_optimizer(optimizer=optimizer, parallel_context=parallel_context, root_folder=store_folder)
    dist.barrier(parallel_context.world_pg)

    # Generate a new optimizer
    new_optimizer = OptimizerFromGradientAccumulator(
        gradient_accumulator_builder=lambda named_params: FP32GradientAccumulator(named_parameters=named_params),
        named_params_or_groups=named_parameters,
        optimizer_builder=lambda named_param_groups: NamedOptimizer(
            named_params_or_groups=named_param_groups,
            optimizer_builder=lambda param_groups: torch.optim.AdamW(param_groups),
        ),
    )
    new_grad_accumulator = new_optimizer.gradient_accumulator

    # Check that the newly initialised optimizer isn't the same.
    if len(optimizer.state_dict()["state"]) == 0:
        pass
    else:
        match, msg = is_dict_equal(optimizer.state_dict(), new_optimizer.state_dict())
        assert not match, "Newly initialised optimizer should not match."

    load_optimizer(optimizer=new_optimizer, parallel_context=parallel_context, root_folder=store_folder)

    # Assert the optimizer states are exactly the same after loading.
    match, msg = is_dict_equal(optimizer.state_dict()["state"], new_optimizer.state_dict()["state"])
    assert match, msg

    # Assert the optimizer state_dict are exactly the same after loading.
    match, msg = is_dict_equal(optimizer.state_dict(), new_optimizer.state_dict())
    assert match, msg

    # Assert the internal optimizer states are exactly the same after loading.
    keys_to_ignore = []
    match, msg = is_dict_equal(
        {
            name: {key: tensor for key, tensor in elt.items() if key not in keys_to_ignore}
            for name, elt in grad_accumulator.parameters.items()
        },
        {
            name: {key: tensor for key, tensor in elt.items() if key not in keys_to_ignore}
            for name, elt in new_grad_accumulator.parameters.items()
        },
    )
    assert match, msg

    parallel_context.destroy()


# TODO @thomasw21: Test with a optimizer that uses `named_param_groups` instead of `param_groups`


@pytest.mark.skipif(available_gpus() < 2, reason="Testing test_save_and_load_random_states requires at least 2 gpus")
@rerun_if_address_is_in_use()
def test_save_and_load_random_states():
    # [FUNCTION DOC]
    # Summary: Pytest entry (needs at least 2 GPUs) with tp=2; launches _test_save_and_load_random_states.
    # Inputs: None. Outputs: None. Exceptions: Propagates worker failures. Side effects: Spawns 2 processes.
    # KNOWN ISSUE: The comment says DP=2 but the test uses tp=2.
    test_context = TestContext()
    # We use DP=2 as we're interested in testing
    init_distributed(tp=2, dp=1, pp=1)(_test_save_and_load_random_states)(test_context=test_context)


def _test_save_and_load_random_states(parallel_context: ParallelContext, test_context: TestContext):
    # [FUNCTION DOC]
    # HELPER: per-rank body of test_save_and_load_random_states.
    # Behavior: Picks the first group of size 2, builds RandomStates with one state synced over that group and one rank-local state, broadcasts rank 0's RandomStates and asserts other ranks differ. Saves random states, loads them back and asserts each rank recovered its own states exactly.
    # Inputs: parallel_context; test_context.
    # Outputs: None.
    # Exceptions: AssertionError on mismatch; StopIteration if no group of size 2.
    # Side effects: Broadcast collective; checkpoint I/O; destroys process groups.
    pg = next(
        (pg for pg in [parallel_context.tp_pg, parallel_context.dp_pg, parallel_context.pp_pg] if pg.size() == 2)
    )
    random_states = RandomStates(
        {
            "my_synced_random_state": get_synced_random_state(random_state=get_current_random_state(), pg=pg),
            "my_own_random_state": get_current_random_state(),
        }
    )
    store_folder = test_context.get_auto_remove_tmp_dir()

    # Check that random states are unequal between ranks (due to `my_own_random_state`)
    reference_rank = 0
    if dist.get_rank(pg) == reference_rank:
        random_statess = [random_states]
    else:
        random_statess = [None]
    dist.broadcast_object_list(random_statess, src=dist.get_global_rank(group_rank=reference_rank, group=pg), group=pg)
    if dist.get_rank(pg) != reference_rank:
        assert random_states != random_statess[0]

    # save
    save_random_states(random_states=random_states, parallel_context=parallel_context, root_folder=store_folder)

    # load
    new_random_states = load_random_states(parallel_context=parallel_context, root_folder=store_folder)
    # Each rank has restored it's own random state
    assert random_states == new_random_states

    parallel_context.destroy()


@rerun_if_address_is_in_use()
def test_serialize_deserialize_tensormetadata():
    # [FUNCTION DOC]
    # Summary: Pytest entry with tp=2; launches _test_serialize_deserialize_tensormetadata.
    # Inputs: None. Outputs: None. Exceptions: Propagates worker failures. Side effects: Spawns 2 processes.
    test_context = TestContext()
    init_distributed(tp=2, dp=1, pp=1)(_test_serialize_deserialize_tensormetadata)(test_context=test_context)


def _test_serialize_deserialize_tensormetadata(parallel_context: ParallelContext, test_context: TestContext):
    # [FUNCTION DOC]
    # HELPER: per-rank body of test_serialize_deserialize_tensormetadata.
    # Behavior: Creates a 16x64 CPU parameter, shards it over the TP group along dim 0 with contiguous chunks of 8 and 8, builds TensorMetadata from its sharded info and the current CHECKPOINT_VERSION, converts it to a string dictionary (asserting all keys and values are strings), parses it back and asserts equality with the original.
    # Inputs: parallel_context; test_context.
    # Outputs: None.
    # Exceptions: AssertionError on failure.
    # Side effects: RNG use; destroys process groups.
    param = torch.nn.Parameter(torch.randn(16, 64))
    split_config = SplitConfig(
        split_dim=0,
        contiguous_chunks=(8, 8),
    )
    param = create_sharded_parameter_from_config(parameter=param, pg=parallel_context.tp_pg, split_config=split_config)
    sharded_info = param.get_sharded_info()
    metadata = TensorMetadata(
        version=CHECKPOINT_VERSION,
        local_global_slices_pairs=sharded_info.local_global_slices_pairs,
        unsharded_shape=sharded_info.unsharded_shape,
    )
    metadata_str_dict = metadata.to_str_dict()
    # Assert metadata_str_dict is Dict[str, str]
    assert isinstance(metadata_str_dict, dict)
    assert all(isinstance(key, str) for key in metadata_str_dict.keys())
    assert all(isinstance(value, str) for value in metadata_str_dict.values())

    metadata_from_str_dict = TensorMetadata.from_str_dict(metadata_str_dict)
    assert metadata == metadata_from_str_dict

    parallel_context.destroy()
