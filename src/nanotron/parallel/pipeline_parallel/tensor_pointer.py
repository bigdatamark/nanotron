# [MODULE DOC]
# Purpose: defines TensorPointer, the placeholder passed between PipelineBlocks on ranks that do not hold a
# tensor. Main contents: TensorPointer. Import-time side effects: None.
import dataclasses


@dataclasses.dataclass
class TensorPointer:
    # [CLASS DOC]
    # Purpose: dataclass saying which rank of the pipeline process group holds (or will compute) the real tensor.
    # Attributes: group_rank (int, rank within the PP group).
    """Dataclass specifying from which rank we need to query a tensor from in order to access data"""

    # Needed to understand from which rank to get the tensor
    # TODO @thomasw21: Maybe add which group it belongs to as well? Typically this is highly correlated to `p2p.pg`
    group_rank: int
    # TODO @thomasw21: Maybe add a tag (torch.distributed.send/recv allow for tagging)
