# [MODULE DOC]
# Purpose: test helper to assert a tensor is identical on every rank of a process group.
# Main contents: assert_tensor_equal_over_group.
import torch
from nanotron import distributed as dist
from nanotron.distributed import ProcessGroup, get_global_rank


def assert_tensor_equal_over_group(tensor: torch.Tensor, group: ProcessGroup, assert_: bool = True) -> bool:
    # [FUNCTION DOC]
    # Behavior: rank 0 of the group is the reference; other ranks allocate an empty tensor of the same shape; the reference is broadcast from rank 0 (translated to its global rank). If assert_ is true, each rank asserts exact equality with the reference. Otherwise each rank computes exact allclose, all ranks all-gather these booleans and the function returns whether all are true.
    # Inputs: tensor, the local tensor (same shape and dtype on all ranks assumed); group, process group; assert_, default True.
    # Outputs: None when assert_ is true; boolean otherwise.
    # Exceptions: AssertionError (from torch testing) on mismatch when assert_ is true.
    # Side effects: broadcast and all_gather_object collectives on group.
    """We assume that tensors are already of correct size."""
    reference_rank = 0
    if dist.get_rank(group) == reference_rank:
        reference_tensor = tensor
    else:
        reference_tensor = torch.empty_like(tensor)
    dist.broadcast(
        reference_tensor,
        src=get_global_rank(group=group, group_rank=reference_rank),
        group=group,
    )
    if assert_:
        torch.testing.assert_close(tensor, reference_tensor, atol=0, rtol=0)
    else:
        result = torch.allclose(tensor, reference_tensor, atol=0.0, rtol=0.0)
        results = [0] * group.size()
        dist.all_gather_object(results, result, group)
        return all(results)
