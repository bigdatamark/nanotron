# [MODULE DOC]
# Purpose: point-to-point tensor transfer between pipeline-parallel ranks, including shipping tensor metadata
# (shape, stride, dtype, storage size, requires_grad) so the receiver can allocate matching buffers.
# Main contents: metadata encoding tables (ID_TO_DTYPE, DTYPE_TO_ID, requires-grad and contiguity id tables,
# FIRST_METADATA_SIZE of 7 longs, SECOND_METADATA_SIZE initial 1024 longs), P2PTensorMetaData, view_as_contiguous,
# P2P (blocking/async send and recv of tensor lists), BatchTensorSendRecvState (batched isend/irecv in three phases).
# Protocol: first metadata is 7 longs: number of shape dims, number of stride dims, contiguity id, untyped
# storage size in bytes, storage offset, dtype id, requires_grad id. Second metadata is shape followed by stride.
# Then the data (whole storage for non-contiguous tensors; complex sent as real views).
# Import-time side effects: creates a module logger.
import dataclasses
from typing import List, Sequence, Tuple

import torch
from nanotron import distributed as dist
from nanotron import logging
from nanotron.utils import get_untyped_storage, tensor_from_untyped_storage

logger = logging.get_logger(__name__)

FIRST_METADATA_SIZE = 7
SECOND_METADATA_SIZE = 1024
ID_TO_DTYPE = [
    torch.float32,
    torch.float64,
    torch.complex64,
    torch.complex128,
    torch.float16,
    torch.bfloat16,
    torch.uint8,
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
    torch.bool,
]
DTYPE_TO_ID = {dtype: id_ for id_, dtype in enumerate(ID_TO_DTYPE)}

ID_TO_REQUIRES_GRAD = [True, False]
REQUIRES_GRAD_TO_ID = {value: id_ for id_, value in enumerate(ID_TO_REQUIRES_GRAD)}
ID_TO_IS_CONTIGUOUS = [True, False]
IS_CONTIGUOUS_TO_ID = {value: id_ for id_, value in enumerate(ID_TO_IS_CONTIGUOUS)}


@dataclasses.dataclass
class P2PTensorMetaData:
    # [CLASS DOC]
    # Purpose: dataclass describing a tensor to be received; can allocate a receive buffer and reinterpret it.
    # Attributes: shape, stride (sequences of ints), is_contiguous, untyped_storage_size (bytes), storage_offset,
    # dtype, requires_grad.
    shape: Sequence[int]
    stride: Sequence[int]
    is_contiguous: bool
    untyped_storage_size: int
    storage_offset: int
    dtype: torch.dtype
    requires_grad: bool

    def create_empty_storage(self, device: torch.device) -> torch.Tensor:
        # [FUNCTION DOC]
        # Behavior: allocates an int8 buffer of untyped_storage_size bytes on device, views it as dtype, sets requires_grad,
        # and if contiguous applies as_strided with shape, stride and storage offset; complex buffers are viewed as real.
        # Inputs: device. Outputs: receive buffer tensor.
        # Exceptions: None explicitly raised (view errors if size not divisible by element size).
        # Side effects: device allocation.
        buffer = torch.empty(
            size=(self.untyped_storage_size,),
            requires_grad=False,
            dtype=torch.int8,
            device=device,
            memory_format=torch.contiguous_format,
        ).view(dtype=self.dtype)
        buffer.requires_grad = self.requires_grad

        if self.is_contiguous:
            buffer = buffer.as_strided(
                size=tuple(self.shape), stride=tuple(self.stride), storage_offset=self.storage_offset
            )

        # Complex needs to be viewed as real first
        # TODO @thomasw21: Find the issue with send/recv complex tensors
        buffer = torch.view_as_real(buffer) if self.dtype.is_complex else buffer

        return buffer

    def reshape(self, buffer):
        # [FUNCTION DOC]
        # Behavior: views a real buffer back as complex for complex dtypes; if non-contiguous applies as_strided with the
        # recorded shape, stride and offset.
        # Inputs: buffer (tensor). Outputs: tensor in the original layout. Exceptions/Side effects: None.
        """Changes the way we view buffer in order to fit metadata"""
        # TODO @thomasw21: Find the issue with send/recv complex tensors
        buffer = torch.view_as_complex(buffer) if self.dtype.is_complex else buffer

        # Set shape and stride
        if not self.is_contiguous:
            buffer = buffer.as_strided(
                size=tuple(self.shape), stride=tuple(self.stride), storage_offset=self.storage_offset
            )

        return buffer

    @staticmethod
    def to_first_metadata(tensor: torch.Tensor, device: torch.device) -> torch.Tensor:
        # [FUNCTION DOC]
        # Behavior: encodes the 7-element first metadata of tensor as a long tensor on device.
        # Inputs: tensor, device. Outputs: long tensor of length 7.
        # Exceptions: KeyError for unsupported dtypes. Side effects: device allocation.
        # TODO @nouamane: avoid having two metadata comms, and preallocate shape/stride instead
        return torch.tensor(
            [
                len(tensor.shape),
                len(tensor.stride()),
                IS_CONTIGUOUS_TO_ID[tensor.is_contiguous()],
                get_untyped_storage(tensor).size(),
                tensor.storage_offset(),
                DTYPE_TO_ID[tensor.dtype],
                REQUIRES_GRAD_TO_ID[tensor.requires_grad],
            ],
            dtype=torch.long,
            device=device,
        )

    @staticmethod
    def to_second_metadata(tensor: torch.Tensor, device: torch.device) -> torch.Tensor:
        # [FUNCTION DOC]
        # Behavior: encodes shape followed by stride as a long tensor on device.
        # Inputs: tensor, device. Outputs: long tensor. Exceptions/Side effects: None.
        return torch.tensor(tensor.shape + tensor.stride(), dtype=torch.long, device=device)

    @classmethod
    def from_metadata(cls, first_metadata: List[int], second_metadata: List[int]):
        # [FUNCTION DOC]
        # Behavior: decodes first metadata (list of 7 ints) and second metadata (shape then stride, split in half) into
        # a P2PTensorMetaData.
        # Inputs: first_metadata, second_metadata (lists of ints). Outputs: P2PTensorMetaData.
        # Exceptions: ValueError on wrong length; IndexError on unknown ids. Side effects: None.
        # KNOWN ISSUE: splits shape and stride by half the length instead of using the transmitted dim counts; correct only because counts are always equal.
        shape_and_stride = second_metadata
        (
            num_shape,
            num_stride,
            is_contiguous,
            untyped_storage_size,
            storage_offset,
            dtype_id,
            requires_grad_id,
        ) = first_metadata
        return cls(
            shape=shape_and_stride[: len(shape_and_stride) // 2],
            stride=shape_and_stride[len(shape_and_stride) // 2 :],
            is_contiguous=ID_TO_IS_CONTIGUOUS[is_contiguous],
            untyped_storage_size=untyped_storage_size,
            storage_offset=storage_offset,
            dtype=ID_TO_DTYPE[dtype_id],
            requires_grad=ID_TO_REQUIRES_GRAD[requires_grad_id],
        )


def view_as_contiguous(tensor: torch.Tensor):
    # [FUNCTION DOC]
    # Behavior: asserts the tensor covers at least its whole untyped storage, then returns a flat tensor of the
    # tensor's dtype over the entire storage.
    # Inputs: tensor. Outputs: 1D tensor sharing storage.
    # Exceptions: AssertionError if the tensor is a view of only part of its storage. Side effects: None.
    # KNOWN ISSUE: assertion message says storage should be smaller than the tensor, which is the opposite of the intended "tensor must cover its storage" reading; also rejects all sliced views.
    """Given a tensor, we want to view the tensor as a contiguous storage"""
    tensor_numel = tensor.numel()
    tensor_element_size = tensor.element_size()
    untyped_storage = get_untyped_storage(tensor)
    untyped_storage_size = untyped_storage.size()
    untyped_element_size = untyped_storage.element_size()
    assert (
        tensor_numel * tensor_element_size >= untyped_storage_size * untyped_element_size
    ), "Expect storage_size to be smaller than tensor size. It might not be true, when you use slicing for example though. We probably don't want to support it in our P2P system"
    buffer = tensor_from_untyped_storage(untyped_storage=untyped_storage, dtype=tensor.dtype)
    return buffer


class P2P:
    # [CLASS DOC]
    # Purpose: P2P sender/receiver bound to one process group (typically PP group) and device.
    # Key state: pg, device, first_metadata (reusable 7-long buffer), second_metadata (reusable, grown as needed).
    def __init__(self, pg: dist.ProcessGroup, device: torch.device):
        # [FUNCTION DOC]
        # Behavior: stores pg and device and preallocates metadata buffers on device.
        # Inputs: pg (ProcessGroup), device (torch.device). Outputs: None. Exceptions: None. Side effects: device allocation.
        self.pg = pg
        self.device = device
        self.first_metadata = torch.empty(FIRST_METADATA_SIZE, dtype=torch.long, device=self.device)
        self.second_metadata = torch.empty(SECOND_METADATA_SIZE, dtype=torch.long, device=self.device)

    def _send_first_metadata_p2p_op(self, tensor: torch.Tensor, to_rank: int, tag: int = 0) -> dist.P2POp:
        # [FUNCTION DOC]
        # HELPER: builds an isend P2POp of the first metadata of tensor to group rank to_rank.
        # Inputs: tensor, to_rank (group rank), tag (default 0). Outputs: P2POp. Exceptions/Side effects: allocation only.
        first_metadata = P2PTensorMetaData.to_first_metadata(tensor=tensor, device=self.device)
        return dist.P2POp(
            op=dist.isend,
            tensor=first_metadata,
            peer=dist.get_global_rank(group=self.pg, group_rank=to_rank),
            group=self.pg,
            tag=tag,
        )

    def _recv_first_metadata_p2p_op(self, from_rank: int, tag: int = 0) -> Tuple[torch.Tensor, dist.P2POp]:
        # [FUNCTION DOC]
        # HELPER: allocates a 7-long buffer and an irecv P2POp from group rank from_rank.
        # Inputs: from_rank, tag. Outputs: (buffer, P2POp). Exceptions/Side effects: allocation only.
        first_metadata_buffer = torch.empty((FIRST_METADATA_SIZE,), dtype=torch.long, device=self.device)
        return first_metadata_buffer, dist.P2POp(
            op=dist.irecv,
            tensor=first_metadata_buffer,
            peer=dist.get_global_rank(group=self.pg, group_rank=from_rank),
            group=self.pg,
            tag=tag,
        )

    def _send_second_metadata_p2p_op(self, tensor: torch.Tensor, to_rank: int, tag: int = 0) -> dist.P2POp:
        # [FUNCTION DOC]
        # HELPER: builds an isend P2POp of shape+stride of tensor. Inputs: tensor, to_rank, tag. Outputs: P2POp.
        second_metadata = P2PTensorMetaData.to_second_metadata(tensor=tensor, device=self.device)
        return dist.P2POp(
            op=dist.isend,
            tensor=second_metadata,
            peer=dist.get_global_rank(group=self.pg, group_rank=to_rank),
            group=self.pg,
            tag=tag,
        )

    def _recv_second_metadata_p2p_op(
        self, shape_length: int, stride_length: int, from_rank: int, tag: int = 0
    ) -> Tuple[torch.Tensor, dist.P2POp]:
        # [FUNCTION DOC]
        # HELPER: allocates a buffer of shape_length + stride_length longs and an irecv P2POp.
        # Inputs: shape_length, stride_length, from_rank, tag. Outputs: (buffer, P2POp).
        second_metadata_buffer = torch.empty((shape_length + stride_length,), dtype=torch.long, device=self.device)
        return second_metadata_buffer, dist.P2POp(
            op=dist.irecv,
            tensor=second_metadata_buffer,
            peer=dist.get_global_rank(group=self.pg, group_rank=from_rank),
            group=self.pg,
            tag=tag,
        )

    def _send_data_p2p_op(self, tensor: torch.Tensor, to_rank: int, tag: int = 0) -> dist.P2POp:
        # [FUNCTION DOC]
        # HELPER: builds an isend P2POp of tensor. Inputs: tensor, to_rank, tag. Outputs: P2POp.
        return dist.P2POp(
            op=dist.isend,
            tensor=tensor,
            peer=dist.get_global_rank(group=self.pg, group_rank=to_rank),
            group=self.pg,
            tag=tag,
        )

    def _recv_data_p2p_op(
        self, tensor_metadata: P2PTensorMetaData, from_rank: int, tag: int = 0
    ) -> Tuple[torch.Tensor, dist.P2POp]:
        # [FUNCTION DOC]
        # HELPER: allocates a receive buffer from metadata and builds an irecv P2POp.
        # Inputs: tensor_metadata, from_rank, tag. Outputs: (buffer, P2POp). Side effects: device allocation.
        tensor_buffer = tensor_metadata.create_empty_storage(self.device)
        return tensor_buffer, dist.P2POp(
            op=dist.irecv,
            tensor=tensor_buffer,
            peer=dist.get_global_rank(group=self.pg, group_rank=from_rank),
            group=self.pg,
            tag=tag,
        )

    def _send_meta(self, tensor: torch.Tensor, to_rank: int, tag: int):
        # [FUNCTION DOC]
        # HELPER: blocking send of first then second metadata for tensor.
        # Behavior: fills the reusable first metadata buffer and sends it; asserts dim counts; grows the second buffer if
        # needed, fills and sends shape+stride.
        # Inputs: tensor, to_rank (group rank), tag. Outputs: None.
        # Exceptions: AssertionError on inconsistent counts; KeyError for unsupported dtype.
        # Side effects: blocking sends; may reallocate self.second_metadata.
        cpu_tensor = torch.tensor(
            [
                len(tensor.shape),
                len(tensor.stride()),
                IS_CONTIGUOUS_TO_ID[tensor.is_contiguous()],
                get_untyped_storage(tensor).size(),
                tensor.storage_offset(),
                DTYPE_TO_ID[tensor.dtype],
                REQUIRES_GRAD_TO_ID[tensor.requires_grad],
            ],
            dtype=torch.long,
        )
        self.first_metadata.copy_(cpu_tensor)
        dist.send(
            self.first_metadata,
            dst=dist.get_global_rank(group=self.pg, group_rank=to_rank),
            group=self.pg,
            tag=tag,
        )

        second_metadata = tensor.shape + tensor.stride()
        assert len(tensor.shape) == self.first_metadata[0]
        assert len(tensor.stride()) == self.first_metadata[1]

        # increase buffer size
        if len(second_metadata) > len(self.second_metadata):
            self.second_metadata = torch.empty(len(second_metadata), dtype=torch.long, device=self.device)

        self.second_metadata[: len(second_metadata)].copy_(torch.tensor(second_metadata, dtype=torch.long))

        dist.send(
            self.second_metadata[: len(second_metadata)],
            dst=dist.get_global_rank(group=self.pg, group_rank=to_rank),
            group=self.pg,
            tag=tag,
        )

    def _recv_meta(self, from_rank: int, tag: int) -> P2PTensorMetaData:
        # [FUNCTION DOC]
        # HELPER: blocking receive of first and second metadata into reusable buffers.
        # Behavior: receives 7 longs, grows second buffer if needed, receives shape+stride, builds P2PTensorMetaData.
        # Inputs: from_rank, tag. Outputs: P2PTensorMetaData.
        # Exceptions: IndexError on unknown ids. Side effects: blocking receives; may reallocate buffer.
        # KNOWN ISSUE: shape and stride are views into the reusable second_metadata buffer (device tensors), so a later receive overwrites the metadata of earlier received tensors still referencing them; also the scalar fields stay 0-d tensors rather than ints.
        dist.recv(
            self.first_metadata,
            src=dist.get_global_rank(group=self.pg, group_rank=from_rank),
            group=self.pg,
            tag=tag,
        )
        (
            num_shape,
            num_stride,
            is_contiguous,
            untyped_storage_size,
            storage_offset,
            dtype_id,
            requires_grad_id,
        ) = self.first_metadata

        # self.pg.recv([second], from_rank, 0).wait() # more direct API
        second_metadata_num_elements = num_shape + num_stride

        # increase buffer size
        if second_metadata_num_elements > len(self.second_metadata):
            self.second_metadata = torch.empty(second_metadata_num_elements, dtype=torch.long, device=self.device)

        dist.recv(
            self.second_metadata[:second_metadata_num_elements],
            src=dist.get_global_rank(group=self.pg, group_rank=from_rank),
            group=self.pg,
            tag=tag,
        )

        shape = self.second_metadata[:num_shape]
        stride = self.second_metadata[num_shape:second_metadata_num_elements]

        return P2PTensorMetaData(
            dtype=ID_TO_DTYPE[dtype_id],
            requires_grad=ID_TO_REQUIRES_GRAD[requires_grad_id],
            shape=shape,
            stride=stride,
            is_contiguous=ID_TO_IS_CONTIGUOUS[is_contiguous],
            untyped_storage_size=untyped_storage_size,
            storage_offset=storage_offset,
        )

    def isend_tensors(self, tensors: List[torch.Tensor], to_rank: int, tag: int = 0) -> List[dist.Work]:
        # [FUNCTION DOC]
        # Behavior: for each tensor sends metadata (blocking), then async-sends the tensor (or the whole storage if
        # non-contiguous; complex as real view) to to_rank.
        # Inputs: tensors (list), to_rank (group rank), tag (default 0). Outputs: list of Work handles.
        # Exceptions: ValueError if to_rank is the current rank; AssertionError from view_as_contiguous.
        # Side effects: P2P communication; debug logging.
        # KNOWN ISSUE: a contiguous tensor that is a view on a larger storage is sent with only its numel elements, while the receiver allocates and receives the whole storage size, causing size mismatch.
        futures = []
        current_rank = dist.get_rank(self.pg)
        logger.debug(f"Current rank {current_rank} sending to rank {to_rank}. Nb_tensors: {len(tensors)}")
        for tensor in tensors:
            if to_rank != current_rank:
                self._send_meta(tensor, to_rank=to_rank, tag=tag)
                if tensor.is_contiguous():
                    buffer = tensor
                else:
                    # If the tensor is not contiguous we send the entire storage
                    buffer = view_as_contiguous(tensor)

                # TODO @thomasw21: Find the issue with send/recv complex tensors
                buffer = torch.view_as_real(buffer) if buffer.is_complex() else buffer

                futures.append(
                    dist.isend(
                        buffer,
                        dst=dist.get_global_rank(group=self.pg, group_rank=to_rank),
                        group=self.pg,
                        tag=tag,
                    )
                )
            else:
                raise ValueError("Tried sending tensor to itself")
        return futures

    def irecv_tensors(
        self, num_tensors: int, from_rank: int, tag: int = 0
    ) -> Tuple[List[torch.Tensor], List[dist.Work]]:
        # [FUNCTION DOC]
        # Behavior: for each expected tensor receives metadata (blocking), allocates a buffer, posts an async receive, and
        # reshapes the buffer per metadata.
        # Inputs: num_tensors (int), from_rank, tag. Outputs: (list of buffers, list of Work handles).
        # Exceptions: ValueError if from_rank is the current rank. Side effects: P2P communication; allocation.
        futures = []
        buffers = []
        current_rank = dist.get_rank(self.pg)
        logger.debug(f"Current rank {current_rank} receiving from rank {from_rank}. Nb_tensors: {num_tensors}")
        for _ in range(num_tensors):
            if from_rank != current_rank:
                meta = self._recv_meta(from_rank=from_rank, tag=tag)

                buffer = meta.create_empty_storage(device=self.device)

                futures.append(
                    dist.irecv(
                        buffer,
                        src=dist.get_global_rank(group=self.pg, group_rank=from_rank),
                        group=self.pg,
                        tag=tag,
                    )
                )

                buffer = meta.reshape(buffer=buffer)

                # Add to the list
                buffers.append(buffer)
            else:
                raise ValueError("Tried receiving tensor from itself")
        return buffers, futures

    def send_tensors(self, tensors: List[torch.Tensor], to_rank: int, tag: int = 0):
        # [FUNCTION DOC]
        # Blocking variant of isend_tensors (waits on all handles). Inputs: tensors, to_rank, tag. Outputs: None.
        # Exceptions: as isend_tensors. Side effects: P2P communication.
        futures = self.isend_tensors(tensors=tensors, to_rank=to_rank, tag=tag)
        for future in futures:
            future.wait()

    def recv_tensors(self, num_tensors: int, from_rank: int, tag: int = 0) -> List[torch.Tensor]:
        # [FUNCTION DOC]
        # Blocking variant of irecv_tensors. Inputs: num_tensors, from_rank, tag. Outputs: list of received tensors.
        # Exceptions: as irecv_tensors. Side effects: P2P communication.
        buffers, futures = self.irecv_tensors(num_tensors=num_tensors, from_rank=from_rank, tag=tag)
        for future in futures:
            future.wait()
        return buffers


class BatchTensorSendRecvState:
    # [CLASS DOC]
    # Purpose: accumulate sends and receives and execute them together with batch_isend_irecv in flush.
    # Key state: p2p, first_metadata_p2p_ops (send and recv ops), second_metadata_p2p_ops (sends only),
    # data_p2p_ops (sends only), recv_first_metadata_buffers, recv_from_ranks.
    # Invariant: recv buffers and recv ranks have equal length; state is cleared after each flush.
    """
    This class is used to register send/recv batches of tensors, and
    then executes send/recv in `flush()` calls. This is useful for
    amortizing the cost of sending and receiving tensors over multiple
    iterations.
    """

    p2p: P2P
    first_metadata_p2p_ops: List[dist.P2POp]
    second_metadata_p2p_ops: List[dist.P2POp]
    data_p2p_ops: List[dist.P2POp]
    recv_first_metadata_buffers: List[torch.Tensor]
    recv_from_ranks: List[int]

    def __init__(self, p2p: P2P):
        # [FUNCTION DOC]
        # Stores p2p and resets lists. Inputs: p2p. Outputs: None.
        self.p2p = p2p
        self._reset()

    def _reset(self):
        # [FUNCTION DOC]
        # HELPER: clears all queued ops and receive bookkeeping. Outputs: None. Side effects: mutates self.
        self.first_metadata_p2p_ops: List[dist.P2POp] = []
        self.second_metadata_p2p_ops: List[dist.P2POp] = []
        self.data_p2p_ops: List[dist.P2POp] = []
        self.recv_first_metadata_buffers: List[torch.Tensor] = []
        self.recv_from_ranks: List[int] = []

    def __str__(self):
        # [FUNCTION DOC]
        # Returns a summary with counts of queued ops and the receive ranks. Outputs: str.
        return f"BatchTensorSendRecvState(first_metadata_p2p_ops={len(self.first_metadata_p2p_ops)}, second_metadata_p2p_ops={len(self.second_metadata_p2p_ops)}, data_p2p_ops={len(self.data_p2p_ops)}, recv_first_metadata_buffers={len(self.recv_first_metadata_buffers)}, recv_from_ranks={self.recv_from_ranks})"

    def add_send(self, tensor: torch.Tensor, to_rank: int, tag: int = 0):
        # [FUNCTION DOC]
        # Behavior: queues first-metadata, second-metadata and data (whole storage via view_as_contiguous) send ops.
        # Inputs: tensor, to_rank (group rank), tag (default 0). Outputs: None.
        # Exceptions: AssertionError from view_as_contiguous. Side effects: allocates metadata tensors.
        self.first_metadata_p2p_ops.append(
            self.p2p._send_first_metadata_p2p_op(tensor=tensor, to_rank=to_rank, tag=tag)
        )
        self.second_metadata_p2p_ops.append(
            self.p2p._send_second_metadata_p2p_op(tensor=tensor, to_rank=to_rank, tag=tag)
        )
        self.data_p2p_ops.append(
            self.p2p._send_data_p2p_op(tensor=view_as_contiguous(tensor), to_rank=to_rank, tag=tag)
        )

    def add_recv(self, from_rank: int, tag: int = 0) -> int:
        # [FUNCTION DOC]
        # Behavior: queues only the first-metadata receive (later phases need its result) and records the source.
        # Inputs: from_rank, tag. Outputs: index of the receive (position in returned list of flush).
        # Exceptions: None. Side effects: allocation.
        """
        Only add p2p ops for the first operation, as `_recv_second_metadata` and `_recv_data_p2p_op`
        require results from the first metadata to be transfered first.
        Return: index of the recv_buffer in `self.recv_first_metadata_buffers`
        """
        buffer, recv_op = self.p2p._recv_first_metadata_p2p_op(from_rank=from_rank, tag=tag)
        self.first_metadata_p2p_ops.append(recv_op)
        self.recv_first_metadata_buffers.append(buffer)
        self.recv_from_ranks.append(from_rank)
        return len(self.recv_first_metadata_buffers) - 1

    def _send_recv_first_metadata(self) -> List[List[int]]:
        # [FUNCTION DOC]
        # HELPER: runs all first-metadata ops, waits, and copies receive buffers to host lists (early sync).
        # Outputs: list of 7-int lists. Side effects: communication and device sync.
        # Send/Recv first metadata
        reqs = dist.batch_isend_irecv(self.first_metadata_p2p_ops)
        for req in reqs:
            req.wait()
        # We want an early cpu/gpu sync here as we are right after the wait so it's nearly free.
        # Removing the tolist call here delays the sync and will impact performance.
        # We need to instantiate it in a list because it is used twice
        first_metadatas = [tensor.tolist() for tensor in self.recv_first_metadata_buffers]
        return first_metadatas

    def _send_recv_second_metadata(self, first_metadata: List[List[int]]) -> List[List[int]]:
        # [FUNCTION DOC]
        # HELPER: builds second-metadata receives sized from first metadata, runs them with queued second-metadata sends,
        # waits and returns host lists.
        # Inputs: first_metadata (list of lists). Outputs: list of int lists. Side effects: communication.
        # turn a list of tuple into a tuple of list
        recv_second_metadata_buffers, recv_second_metadata_ops = zip(
            *(
                self.p2p._recv_second_metadata_p2p_op(
                    shape_length=num_shape, stride_length=num_stride, from_rank=from_rank
                )
                for (num_shape, num_stride, *_), from_rank in zip(first_metadata, self.recv_from_ranks)
            )
        )
        recv_second_metadata_ops = list(recv_second_metadata_ops)
        # Send/Recv second metadata
        reqs = dist.batch_isend_irecv(self.second_metadata_p2p_ops + recv_second_metadata_ops)
        for req in reqs:
            req.wait()

        # We want an early cpu/gpu sync here as we are right after the wait so it's nearly free.
        # Removing the tolist call here delays the sync and will impact performance.
        second_metadatas = [tensor.tolist() for tensor in recv_second_metadata_buffers]
        return second_metadatas

    def _send_recv_data(self, tensor_metadatas: List[P2PTensorMetaData]) -> List[torch.Tensor]:
        # [FUNCTION DOC]
        # HELPER: allocates data receive buffers from metadata, runs them with queued data sends, waits, and returns
        # buffers as_strided to recorded shape and stride.
        # Inputs: tensor_metadatas (list of P2PTensorMetaData). Outputs: list of tensors. Side effects: communication.
        # KNOWN ISSUE: the final as_strided ignores storage_offset and, for complex dtypes, is applied to the real view with complex shape/stride, giving wrong results.
        # turn a list of tuples into a tuple of list
        recv_data_buffers, recv_data_ops = zip(
            *(
                self.p2p._recv_data_p2p_op(tensor_metadata=tensor_metadata, from_rank=from_rank)
                for tensor_metadata, from_rank in zip(tensor_metadatas, self.recv_from_ranks)
            )
        )
        recv_data_ops = list(recv_data_ops)
        # Send/Recv tensor data
        futures = dist.batch_isend_irecv(self.data_p2p_ops + recv_data_ops)
        for future in futures:
            future.wait()

        # Format tensor by setting the stride
        return [
            recv_data_buffer.as_strided(size=tuple(tensor_metadata.shape), stride=tuple(tensor_metadata.stride))
            for recv_data_buffer, tensor_metadata in zip(recv_data_buffers, tensor_metadatas)
        ]

    def flush(self) -> List[torch.Tensor]:
        # [FUNCTION DOC]
        # Behavior: asserts bookkeeping consistency. Returns empty list if nothing queued. If there are only sends,
        # issues all three phases' ops in one batch, waits, resets, returns empty list. Otherwise runs the three phases
        # (first metadata, second metadata, data), builds metadata objects, resets, and returns received tensors in add_recv order.
        # Inputs: none. Outputs: list of received tensors.
        # Exceptions: AssertionError on inconsistent bookkeeping. Side effects: P2P communication; debug logging; resets state.
        """
        Run all communication in a batch.
        Return `torch.Tensor` in the case of recv.
        """
        assert len(self.recv_first_metadata_buffers) == len(
            self.recv_from_ranks
        ), f"len(self.recv_first_metadata_buffers)={len(self.recv_first_metadata_buffers)}, len(self.recv_from_ranks)={len(self.recv_from_ranks)} but should be equal."

        # If there is no communication, return
        if len(self.first_metadata_p2p_ops) == 0:
            return []

        # If there is no recv
        if len(self.recv_first_metadata_buffers) == 0:
            reqs = dist.batch_isend_irecv(
                self.first_metadata_p2p_ops + self.second_metadata_p2p_ops + self.data_p2p_ops
            )
            for req in reqs:
                req.wait()
            self._reset()
            return []

        # Send/Recv first metadata
        logger.debug(f"First metadata: {[p2pop.op for p2pop in self.first_metadata_p2p_ops]}")
        # TODO(kunhao): We could actually send all at once like the above no recv case. But I need to benchmark the performance.
        first_metadatas = self._send_recv_first_metadata()
        # Send/Recv second metadata
        second_metadatas = self._send_recv_second_metadata(first_metadatas)

        tensor_metadatas = [
            P2PTensorMetaData.from_metadata(first_metadata, second_metadata)
            for first_metadata, second_metadata in zip(first_metadatas, second_metadatas)
        ]

        recv_tensors = self._send_recv_data(tensor_metadatas)
        # Reset state
        self._reset()

        return recv_tensors
