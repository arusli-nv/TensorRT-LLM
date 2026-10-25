# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
import os
import pickle  # nosec B403
import threading
import time
from abc import ABC, abstractmethod
from enum import IntEnum
from functools import lru_cache, wraps
from typing import Any, Callable, List, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist
from numpy.typing import ArrayLike
from torch.distributed.distributed_c10d import (_object_to_tensor,
                                                _tensor_to_object)

try:
    from mpi4py import MPI
except Exception:
    MPI = None  # deferred; functions will error if used when ENABLE_MULTI_DEVICE is True

from tensorrt_llm._torch.distributed.mnnvl_memory import init_helix_cp_comm
# yapf: disable
from tensorrt_llm._utils import (is_mpi_peer_failure, local_mpi_size,
                                 mpi_allgather, mpi_barrier, mpi_comm,
                                 mpi_disabled, mpi_failed_ranks, mpi_isend,
                                 mpi_isend_object, mpi_recv, mpi_recv_object,
                                 mpi_send, mpi_send_object, mpi_world_ranks,
                                 mpi_world_size, prefer_pinned,
                                 torch_pybind11_abi)
# yapf: enable
from tensorrt_llm.bindings.BuildInfo import ENABLE_MULTI_DEVICE
from tensorrt_llm.bindings.internal.process_group import init_pg
from tensorrt_llm.logger import logger
from tensorrt_llm.mapping import Mapping

try:
    import ray
except ModuleNotFoundError:
    from tensorrt_llm.executor.ray import stub as ray


class HostCollectiveInterrupted(Exception):
    """A host collective failed because a peer process died or the
    communicator was revoked.

    The original MPI exception is chained as ``__cause__``.

    Attributes:
        operation: Name of the interrupted ``Distributed`` method.
        failed_ranks: Logical (world) ranks this process knows to have
            failed when the error was raised. May be empty when the MPI
            library cannot report them.
    """

    def __init__(self, operation: str, failed_ranks: frozenset[int]) -> None:
        detail = sorted(failed_ranks) if failed_ranks else "unknown"
        super().__init__(f"Host collective {operation} interrupted by peer "
                         f"failure (failed ranks: {detail})")
        self.operation = operation
        self.failed_ranks = failed_ranks

    def __reduce__(self):
        return type(self), (self.operation, self.failed_ranks)


# How long a host collective error waits for ULFM to report a failed peer
# before it counts as an error of this rank.
_PEER_FAILURE_EVIDENCE_S = 1.0

# Default of the ``departed`` argument of the object gathers: after a peer
# loss the gather raises instead of returning a placeholder for the departed
# rank, so no caller treats a missing rank as an answer by accident.
DEPARTED_UNSPECIFIED = object()


def _errors_return(comm: "MPI.Comm") -> "MPI.Comm":
    """Return *comm* set to return errors: a peer failure in it must raise,
    never abort the job."""
    comm.Set_errhandler(MPI.ERRORS_RETURN)
    return comm


def _failure_evidence(comm: "MPI.Comm", timeout: float) -> frozenset[int]:
    """Failed ranks ULFM reports in *comm*, waiting up to *timeout* seconds
    unless *comm* is revoked first."""
    deadline = time.monotonic() + timeout
    failed = mpi_failed_ranks(comm)
    while not failed and not comm.Is_revoked() and time.monotonic() < deadline:
        time.sleep(0.001)
        failed = mpi_failed_ranks(comm)
    return failed


def _interruptible(func: Callable) -> Callable:
    """With peer-loss recovery enabled, raise :class:`HostCollectiveInterrupted`
    from an ``MPIDist`` method whenever a peer has failed.

    ULFM can complete a collective that a peer dies in without an error on some
    ranks, with that peer's part undefined, so a result counts only once no
    failure is known. An exception counts as an interruption when ULFM reports
    a failed peer shortly after or the communicator was revoked (a garbled
    payload can fail to unpickle with any exception type, also inside mpi4py);
    otherwise it is raised as is. Without recovery the method runs unchanged.
    """

    @wraps(func)
    def wrapper(self: "MPIDist", *args, **kwargs):
        if self._host_world is None:
            return func(self, *args, **kwargs)
        # Read by hang detection: a rank in a host collective waits on its peers.
        self.in_host_collective = True
        try:
            result = func(self, *args, **kwargs)
        except HostCollectiveInterrupted:
            raise
        except Exception as error:  # noqa: BLE001 - classified below; others re-raised as is
            comm = self._world_comm()
            if is_mpi_peer_failure(error):
                raise HostCollectiveInterrupted(
                    func.__name__, mpi_failed_ranks(comm)) from error
            failed = _failure_evidence(comm, _PEER_FAILURE_EVIDENCE_S)
            if not failed and not comm.Is_revoked():
                raise
            raise HostCollectiveInterrupted(func.__name__, failed) from error
        finally:
            self.in_host_collective = False
        failed = mpi_failed_ranks(self._world_comm())
        if failed:
            raise HostCollectiveInterrupted(func.__name__, failed)
        return result

    return wrapper


class ReduceOp(IntEnum):
    SUM = 0
    PRODUCT = 1
    MIN = 2
    MAX = 3
    BAND = 4
    BOR = 5
    BXOR = 6


_reduce_op_to_torch_dict = {
    ReduceOp.SUM: torch.distributed.ReduceOp.SUM,
    ReduceOp.PRODUCT: torch.distributed.ReduceOp.PRODUCT,
    ReduceOp.MIN: torch.distributed.ReduceOp.MIN,
    ReduceOp.MAX: torch.distributed.ReduceOp.MAX,
    ReduceOp.BAND: torch.distributed.ReduceOp.BAND,
    ReduceOp.BOR: torch.distributed.ReduceOp.BOR,
    ReduceOp.BXOR: torch.distributed.ReduceOp.BXOR,
}


def _check_reduce_op(op: ReduceOp) -> None:
    if not isinstance(op, ReduceOp):
        raise TypeError(
            f"expected tensorrt_llm ReduceOp, got {type(op).__name__}: {op!r}. "
            "torch.distributed.ReduceOp and MPI.Op use different numeric values "
            "(e.g. torch MIN==3 collides with tensorrt_llm MAX==3), so a foreign "
            "enum would silently select the wrong reduction.")


def reduce_op_to_torch(op: ReduceOp) -> torch.distributed.ReduceOp:
    _check_reduce_op(op)
    return _reduce_op_to_torch_dict[op]


_reduce_op_to_mpi_dict = {
    ReduceOp.SUM: MPI.SUM,
    ReduceOp.PRODUCT: MPI.PROD,
    ReduceOp.MIN: MPI.MIN,
    ReduceOp.MAX: MPI.MAX,
    ReduceOp.BAND: MPI.BAND,
    ReduceOp.BOR: MPI.BOR,
    ReduceOp.BXOR: MPI.BXOR,
}


def reduce_op_to_mpi(op: ReduceOp) -> MPI.Op:
    _check_reduce_op(op)
    return _reduce_op_to_mpi_dict[op]


def live_rows(dist: "Distributed", rows: np.ndarray) -> np.ndarray:
    """The rows of a ``tp_allgather_int64`` that the ranks in ``dist.live_ranks`` sent, in rank order.

    After a recovered peer loss, gathers fill a departed rank's row; callers that combine only the serving
    ranks' values drop it here. Without a departed rank this returns ``rows``.
    """
    live = dist.live_ranks
    return rows if len(live) == len(rows) else rows[list(live)]


class Distributed(ABC):

    def __init__(self, mapping: Mapping):
        self.mapping = mapping
        # Whether this rank is inside a host collective (set with peer-loss recovery).
        self.in_host_collective = False
        self._tp_ranks = tuple(range(mapping.tp_size))

    @staticmethod
    @lru_cache(maxsize=None)
    def get(mapping: Mapping) -> "Distributed":
        if mpi_disabled():
            return TorchDist(mapping)
        else:
            return MPIDist(mapping)

    @property
    def rank(self):
        return self.mapping.rank

    @property
    def world_size(self):
        return self.mapping.world_size

    @property
    def has_tp(self):
        return self.mapping.has_tp()

    @property
    def has_pp(self):
        return self.mapping.has_pp()

    @property
    def cp_size(self):
        return self.mapping.cp_size

    @property
    def pp_size(self):
        return self.mapping.pp_size

    @property
    def tp_size(self):
        return self.mapping.tp_size

    @property
    def cp_rank(self):
        return self.mapping.cp_rank

    @property
    def tp_rank(self):
        return self.mapping.tp_rank

    @property
    def pp_rank(self):
        return self.mapping.pp_rank

    @property
    def is_last_pp_rank(self):
        return self.mapping.is_last_pp_rank()

    @property
    def is_second_last_pp_rank(self):
        return self.mapping.is_second_last_pp_rank()

    @property
    def is_first_pp_rank(self):
        return self.mapping.is_first_pp_rank()

    @property
    def next_pp_rank(self):
        return self.mapping.next_pp_rank()

    @property
    def prev_pp_rank(self):
        return self.mapping.prev_pp_rank()

    @property
    def has_cp_ulysses(self):
        return self.mapping.has_cp_ulysses()

    @property
    def has_cp_helix(self):
        return self.mapping.has_cp_helix()

    @property
    def cp_config(self):
        return self.mapping.cp_config

    @property
    @abstractmethod
    def local_world_size(self):
        """Number of ranks co-located on this physical node."""

    @property
    def live_ranks(self) -> tuple[int, ...]:
        """Logical TP ranks whose processes take part in host collectives."""
        return self._tp_ranks

    def enable_peer_loss_recovery(self) -> None:
        """Run host collectives on communicators that recovery may revoke."""
        raise NotImplementedError(
            f"{type(self).__name__} does not support peer-loss recovery")

    @abstractmethod
    def barrier(self):
        pass

    @abstractmethod
    def tp_barrier(self):
        pass

    @abstractmethod
    def broadcast(self, obj, root=0, prefer_cpu=False):
        """Broadcast ``obj`` from ``root``.

        ``prefer_cpu`` keeps the collective on the CPU transport where the
        implementation would otherwise stage it through the GPU: callers
        whose peers may PARK in this collective for a long time (e.g. the
        idle-path request-count probe, where non-root ranks wait while rank
        0 blocks on the request queue) must set it -- a GPU-staged NCCL wait
        spins a kernel on the device for the whole park and is subject to
        the NCCL watchdog, which killed idle context engines after 600 s in
        production (Qwen3.5-397B disagg incidents, 2026-08-31). MPI-backed
        implementations may ignore it.
        """

    # Transports that can tell "nothing to send" apart from a payload inside
    # the payload round itself set this (TorchDist's CUDA object path) and
    # override ``broadcast_or_none`` with a single collective round.
    supports_single_round_broadcast = False

    def broadcast_or_none(self, obj, root=0):
        """Broadcast ``obj`` from ``root``; ``None`` on the root means "nothing".

        Every rank returns the root's object, or ``None`` on every rank when
        the root passed ``None``. This generic form spends a flag broadcast
        plus a payload broadcast -- the two rounds callers used to issue by
        hand as request-count probe + payload; single-round transports
        override it. Rank-consistent by construction: the branch is decided
        by the root's data as received by every rank.
        """
        has_payload = self.broadcast(int(obj is not None), root=root)
        if not has_payload:
            return None
        return self.broadcast(obj, root=root)

    @abstractmethod
    def allgather(self, obj, root=0, *, departed=DEPARTED_UNSPECIFIED):
        pass

    @abstractmethod
    def allreduce(self, obj, op: ReduceOp = ReduceOp.SUM):
        pass

    @abstractmethod
    def tp_allreduce(self, obj, op: ReduceOp = ReduceOp.SUM):
        pass

    @abstractmethod
    def tp_broadcast(self, obj, root=0, **kwargs):
        pass

    @abstractmethod
    def cp_broadcast(self, obj, root=0, **kwargs):
        pass

    def tp_cp_broadcast(self, obj, root=0, **kwargs):
        """Broadcast object across both TP and CP groups.

        This is used when both TP and CP parallelism are enabled (e.g., helix parallelism).
        First broadcasts within the TP group, then within the CP group.
        """
        if self.tp_size > 1:
            obj = self.tp_broadcast(obj, root=root, **kwargs)
        if self.cp_size > 1:
            obj = self.cp_broadcast(obj, root=root, **kwargs)
        return obj

    @abstractmethod
    def tp_allgather(self,
                     obj,
                     *,
                     small_payload: bool = False,
                     departed=DEPARTED_UNSPECIFIED):
        """Gather *obj* from every TP rank, in rank order.

        After a recovered peer loss, a departed rank has no entry: the caller
        must pass *departed*, the value that stands in for it (for example
        ``None`` when the caller skips missing entries); without it the gather
        raises. The same holds for :meth:`allgather` and ``tp_gather``.
        """

    @abstractmethod
    def cp_allgather(self, obj, *, small_payload: bool = False):
        pass

    def tp_cp_allgather(self, obj, *, small_payload: bool = False):
        """Allgather across both TP and CP dimensions.

        First gathers within CP group, then across TP groups, returning
        a flattened list with tp_size * cp_size entries.
        """
        # Gather across CP dimension.
        if self.cp_size > 1:
            obj = self.cp_allgather(obj, small_payload=small_payload)
        else:
            obj = [obj]  # Wrap to match cp_allgather output format.

        # Gather across TP dimension.
        if self.tp_size > 1:
            obj = self.tp_allgather(obj, small_payload=small_payload)
        else:
            obj = [obj]  # Wrap to match tp_allgather output format.

        # Flatten: [[cp0, cp1], [cp0, cp1], ...] -> [tp0_cp0, tp0_cp1, tp1_cp0, ...]
        return [entry for tp_group in obj for entry in tp_group]

    # Fixed-size int64 exchanges. MPIDist does each as one buffer collective
    # (no pickle); the defaults below reuse the object paths, sending plain
    # int lists so the pickled payload stays small on every backend.

    def tp_allgather_int64(
            self,
            values: ArrayLike,
            *,
            departed: ArrayLike = DEPARTED_UNSPECIFIED) -> np.ndarray:
        """All-gather a fixed-size int64 vector across the TP group.

        Returns an int64 array of shape ``[tp_size, len(values)]`` whose row
        *i* is rank *i*'s vector. Every rank must pass the same length. Rows
        of ranks outside :attr:`live_ranks` (ranks that left the group after a
        recovered peer loss) are set to *departed* (a scalar or a per-column
        vector), the value that is neutral for how the caller combines rows.
        Without one, a departed rank raises: the caller does not handle it.
        """
        vec = np.asarray(values, dtype=np.int64).reshape(-1)
        gathered = self.tp_allgather(vec.tolist(), small_payload=True)
        return np.asarray(gathered, dtype=np.int64).reshape(len(gathered), -1)

    def cp_allgather_int64(self, values: ArrayLike) -> np.ndarray:
        """All-gather a fixed-size int64 vector across the CP group; rows are
        ordered by CP rank."""
        vec = np.asarray(values, dtype=np.int64).reshape(-1)
        gathered = self.cp_allgather(vec.tolist(), small_payload=True)
        return np.asarray(gathered, dtype=np.int64).reshape(len(gathered), -1)

    def tp_cp_allgather_int64(
            self,
            values: ArrayLike,
            *,
            departed: ArrayLike = DEPARTED_UNSPECIFIED) -> np.ndarray:
        """Fixed-size int64 all-gather across TP x CP; rows are ordered like
        :meth:`tp_cp_allgather` (tp-major, cp-minor). The rows of a TP rank
        that left the group are *departed*, as in :meth:`tp_allgather_int64`."""
        vec = np.asarray(values, dtype=np.int64).reshape(-1)
        n = vec.size
        if self.cp_size > 1:
            vec = self.cp_allgather_int64(vec).reshape(-1)
        if self.tp_size > 1:
            vec = self.tp_allgather_int64(vec, departed=departed).reshape(-1)
        return vec.reshape(-1, n)

    def broadcast_int64(self,
                        values: ArrayLike,
                        root: int = 0,
                        *,
                        prefer_cpu: bool = False) -> np.ndarray:
        """Broadcast a fixed-size int64 vector from *root* to every rank.
        Non-root ranks pass a placeholder vector of the same length.
        ``prefer_cpu`` has the meaning documented on :meth:`broadcast`."""
        vec = np.asarray(values, dtype=np.int64).reshape(-1)
        return np.asarray(self.broadcast(vec.tolist(),
                                         root=root,
                                         prefer_cpu=prefer_cpu),
                          dtype=np.int64)

    def tp_cp_broadcast_int64(self,
                              values: ArrayLike,
                              root: int = 0) -> np.ndarray:
        """Broadcast a fixed-size int64 vector from *root* across TP and CP."""
        vec = np.asarray(values, dtype=np.int64).reshape(-1)
        return np.asarray(self.tp_cp_broadcast(vec.tolist(), root=root),
                          dtype=np.int64)


def safe_broadcast(comm, obj, root=0, chunk_size: int = 4 * 1024 * 1024):
    """
    Safely broadcasts potentially large objects by splitting into fixed-size chunks,
    using raw-byte MPI.Bcast to avoid pickle5's out-of-band buffer allocations.

    Args:
        comm: communicator to broadcast
        obj: Python object to broadcast
        root: Rank of the broadcasting process
        chunk_size: Maximum size of each chunk in bytes (default: 4MB)

    Returns:
        The broadcasted object on all ranks
    """
    if not ENABLE_MULTI_DEVICE:
        return obj
    if ENABLE_MULTI_DEVICE and MPI is None:
        raise RuntimeError(
            "mpi4py is required when ENABLE_MULTI_DEVICE is True")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    rank = comm.Get_rank()

    # ---- Serialization phase (root only) ----
    # Header layout: [ok_flag, total_size, num_chunks] as int64
    header = np.zeros(3, dtype=np.int64)
    if rank == root:
        try:
            serialized = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
            total_size = len(serialized)
            num_chunks = math.ceil(total_size /
                                   chunk_size) if total_size > 0 else 0
            header[:] = (1, total_size, num_chunks)
        except Exception as e:
            # Signal failure to all ranks, then raise
            header[:] = (0, 0, 0)
            comm.Bcast([header, MPI.INT64_T], root=root)
            raise RuntimeError(f"Serialization failed: {str(e)}") from e
    else:
        serialized = None  # not used on non-root before Bcast

    # ---- Metadata broadcast (Bcast the fixed-size header) ----
    comm.Bcast([header, MPI.INT64_T], root=root)
    ok_flag, total_size, num_chunks = int(header[0]), int(header[1]), int(
        header[2])
    if not ok_flag:
        raise RuntimeError("Root rank failed during serialization")

    # ---- Allocate receive buffer (non-root) or build a view (root) ----
    # We broadcast raw bytes chunk by chunk.
    if rank == root:
        src_view = memoryview(serialized)
        dst_buf = None
        dst_view = None
    else:
        # Pre-allocate a contiguous byte buffer to receive the payload
        dst_buf = bytearray(total_size)
        dst_view = memoryview(dst_buf)
        src_view = None  # not used on non-root

    # ---- Chunked raw-byte broadcast with MPI.Bcast ----
    # Each round sends exactly `cur` bytes of the global payload.
    offset = 0
    for i in range(num_chunks):
        cur = min(chunk_size, total_size - offset)
        if cur <= 0:
            break  # safety guard for zero-size payloads

        if rank == root:
            # Root sends a slice of the source view
            part = src_view[offset:offset + cur]
            comm.Bcast([part, MPI.BYTE], root=root)
        else:
            # Non-root receives directly into the destination view
            part = dst_view[offset:offset + cur]
            comm.Bcast([part, MPI.BYTE], root=root)

        offset += cur

    # ---- Reconstruction and deserialization ----
    if rank == root:
        # Root already holds `obj`; rebuilding it from its own serialized bytes
        # would be a needless deep copy.
        return obj
    else:
        # Validate the received byte count and unpickle.
        if len(dst_buf) != total_size:
            raise RuntimeError(
                f"Data size mismatch at rank {rank}: expected {total_size}, got {len(dst_buf)}"
            )
        try:
            return pickle.loads(dst_buf)  # nosec B301
        except Exception as e:
            raise RuntimeError(f"Deserialization failed: {str(e)}") from e


def _serialize_and_exchange_lengths(
    comm: Any,
    obj: Any,
) -> Tuple[int, int, np.ndarray, np.ndarray, np.ndarray]:
    """Serialize *obj* and exchange payload lengths across all ranks.

    Uses buffer-based ``MPI_Allgather`` (uppercase) for the length
    exchange — a single MPI collective with no pickle overhead, which
    is the same work that mpi4py does internally inside
    ``comm.allgather(obj)``.

    Args:
        comm: MPI communicator (``MPI.Comm`` instance).
        obj: Python object to transfer (must be picklable).

    Returns:
        Tuple of ``(rank, size, lengths, displs, sendbuf)`` where:

        - **rank** (*int*) — this process's rank in *comm*.
        - **size** (*int*) — total number of ranks in *comm*.
        - **lengths** (*np.ndarray[int64]*) — per-rank serialized payload
          sizes.  A value of ``-1`` signals a serialization failure.
        - **displs** (*np.ndarray[int64]*) — per-rank byte offsets into a
          concatenated receive buffer.
        - **sendbuf** (*np.ndarray[uint8]*) — this rank's serialized
          payload as a contiguous byte array (empty when serialization
          failed).
    """
    rank = comm.Get_rank()
    size = comm.Get_size()

    local_ser_error = None
    try:
        payload = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
        local_len = np.array([len(payload)], dtype=np.int64)
    except Exception as exc:
        payload = b""
        local_len = np.array([-1], dtype=np.int64)
        local_ser_error = exc

    # Buffer-based Allgather: 1 MPI collective, no pickle overhead.
    lengths = np.empty(size, dtype=np.int64)
    comm.Allgather([local_len, MPI.INT64_T], [lengths, MPI.INT64_T])

    if (lengths < 0).any():
        raise RuntimeError(
            f"Rank {rank}: serialization failed on at least one rank "
            f"(lengths={lengths})") from local_ser_error

    displs = np.zeros(size, dtype=np.int64)
    if size > 1:
        displs[1:] = np.cumsum(lengths[:-1])

    sendbuf = np.frombuffer(payload, dtype=np.uint8)
    return rank, size, lengths, displs, sendbuf


def _chunked_transfer_loop(
    comm: Any,
    rank: int,
    size: int,
    lengths: np.ndarray,
    displs: np.ndarray,
    sendbuf: np.ndarray,
    num_rounds: int,
    chunk_size: int,
    recvbuf: Optional[np.ndarray],
    collective_fn: Callable[[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
                            None],
) -> None:
    """Run the chunked MPI transfer loop used by safe_gather/safe_allgather.

    Each round transfers at most ``chunk_size`` bytes per rank using a
    per-round temporary receive buffer with 0-based int32 displacements,
    then copies the received data into ``recvbuf`` at the correct absolute
    offsets using 64-bit Python-level indexing.

    Args:
        comm: MPI communicator (``MPI.Comm`` instance).
        rank: This rank's index in *comm*.
        size: Total number of ranks in *comm*.
        lengths: Per-rank serialized payload sizes (int64 array of shape
            ``(size,)``).
        displs: Per-rank byte offsets into *recvbuf* (int64 array of shape
            ``(size,)``).
        sendbuf: This rank's serialized payload (uint8 array).
        num_rounds: Number of chunked transfer rounds to execute.
        chunk_size: Per-round max bytes each rank contributes.
        recvbuf: Final contiguous receive buffer (uint8 array), or
            ``None`` for non-root ranks in gather mode (copy-back is
            skipped).
        collective_fn: Callable that performs the per-round MPI
            collective.  Signature: ``collective_fn(send_part,
            round_recvbuf, counts_this_round, round_displs)``.
    """
    for r in range(num_rounds):
        round_offs = r * chunk_size
        counts_this_round = np.minimum(np.maximum(lengths - round_offs, 0),
                                       chunk_size).astype(np.int32)
        sent_so_far = np.minimum(lengths, round_offs)

        round_recvbuf = (np.empty(counts_this_round.sum(), dtype=np.uint8)
                         if recvbuf is not None else None)
        round_displs = np.zeros(size, dtype=np.int32)
        if size > 1:
            round_displs[1:] = np.cumsum(counts_this_round[:-1])

        send_part = sendbuf[sent_so_far[rank]:sent_so_far[rank] +
                            counts_this_round[rank]]

        collective_fn(send_part, round_recvbuf, counts_this_round, round_displs)

        # Copy received chunks into the final buffer at correct
        # absolute offsets (using 64-bit Python-level indexing).
        if recvbuf is not None:
            src_offset = 0
            for i in range(size):
                n = counts_this_round[i]
                if n > 0:
                    dst = displs[i] + sent_so_far[i]
                    recvbuf[dst:dst +
                            n] = (round_recvbuf[src_offset:src_offset + n])
                src_offset += n


def _deserialize_recvbuf(
    recvbuf: np.ndarray,
    lengths: np.ndarray,
    displs: np.ndarray,
    size: int,
) -> List[Any]:
    """Deserialize gathered payloads from a contiguous receive buffer.

    Args:
        recvbuf: Contiguous receive buffer (uint8 array) containing the
            concatenated serialized payloads from all ranks.
        lengths: Per-rank serialized payload sizes (int64 array of shape
            ``(size,)``).
        displs: Per-rank byte offsets into *recvbuf* (int64 array of shape
            ``(size,)``).
        size: Total number of ranks.

    Returns:
        List of deserialized Python objects (``len == size``). Ranks whose
        payload length is zero are represented as ``None``.
    """
    # Zero-length payloads (e.g. from pickling None) are returned as None
    # without calling pickle.loads, which would fail on empty bytes.
    return [
        pickle.loads(recvbuf[displs[i]:displs[i] + lengths[i]])  # nosec B301
        if lengths[i] > 0 else None for i in range(size)
    ]


def safe_gather(
    comm: Any,
    obj: Any,
    root: int = 0,
    chunk_size: int = 4 * 1024 * 1024,
) -> Optional[List[Any]]:
    """Safely gather potentially large objects by splitting into fixed-size
    chunks, using raw-byte MPI.Gatherv with a per-round temp buffer to
    keep counts and displacements within int32.

    The function serializes *obj* once with ``pickle.dumps``, exchanges
    payload lengths via buffer-based ``MPI_Allgather`` (1 MPI collective),
    then transfers the raw bytes with ``MPI_Gatherv`` (1 MPI collective).
    This matches the number of MPI collectives that mpi4py's
    ``comm.gather(obj)`` performs internally, while adding chunking
    safety for payloads whose total exceeds the int32 displacement
    limit (~2 GB).

    Args:
        comm: MPI communicator (``MPI.Comm`` instance) to gather over.
        obj: Python object to gather (must be picklable).
        root: Rank that receives the gathered objects.
        chunk_size: Per-round max bytes each rank contributes (default:
            4 MB).

    Returns:
        On *root*: list of deserialized objects (``len == comm.size``).
        On non-root ranks: ``None``.
    """
    if not ENABLE_MULTI_DEVICE:
        return [obj]
    if MPI is None:
        raise RuntimeError(
            "mpi4py is required when ENABLE_MULTI_DEVICE is True")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")

    # Step 1: serialize once and exchange lengths (1 MPI collective).
    rank, size, lengths, displs, sendbuf = \
        _serialize_and_exchange_lengths(comm, obj)

    total = int(lengths.sum())
    int32_max = np.iinfo(np.int32).max

    # Step 2a: total fits in int32 — single Gatherv (1 MPI collective).
    if total < int32_max:
        counts = lengths.astype(np.int32)
        displs32 = displs.astype(np.int32)
        if rank == root:
            recvbuf = np.empty(total, dtype=np.uint8)
            comm.Gatherv([sendbuf, MPI.BYTE],
                         [recvbuf, counts, displs32, MPI.BYTE],
                         root=root)
            return _deserialize_recvbuf(recvbuf, lengths, displs, size)
        else:
            comm.Gatherv([sendbuf, MPI.BYTE], None, root=root)
            return None

    # Step 2b: total exceeds int32 — chunked Gatherv.
    logger.info(
        "safe_gather: total payload %d bytes exceeds int32 limit, "
        "using chunked Gatherv (size=%d)", total, size)
    max_safe_chunk = int32_max // size
    chunk_size = min(chunk_size, max_safe_chunk)
    max_len = int(lengths.max())
    num_rounds = math.ceil(max_len / chunk_size) if max_len > 0 else 0

    recvbuf = np.empty(total, dtype=np.uint8) if rank == root else None

    def _gatherv(send_part, round_recvbuf, counts, round_displs):
        if rank == root:
            comm.Gatherv([send_part, MPI.BYTE],
                         [round_recvbuf, counts, round_displs, MPI.BYTE],
                         root=root)
        else:
            comm.Gatherv([send_part, MPI.BYTE], None, root=root)

    _chunked_transfer_loop(comm, rank, size, lengths, displs, sendbuf,
                           num_rounds, chunk_size, recvbuf, _gatherv)

    if rank == root:
        return _deserialize_recvbuf(recvbuf, lengths, displs, size)
    return None


def safe_allgather(
    comm: Any,
    obj: Any,
    chunk_size: int = 4 * 1024 * 1024,
) -> List[Any]:
    """Safely allgather potentially large objects by splitting into
    fixed-size chunks, using raw-byte MPI.Allgatherv.

    The function serializes *obj* once with ``pickle.dumps``, exchanges
    payload lengths via buffer-based ``MPI_Allgather`` (1 MPI collective),
    then transfers the raw bytes with ``MPI_Allgatherv`` (1 MPI
    collective).  This matches the number of MPI collectives that
    mpi4py's ``comm.allgather(obj)`` performs internally, while adding
    chunking safety for payloads whose total exceeds the int32
    displacement limit (~2 GB) and avoiding mpi4py's pickle5
    out-of-band buffers that can cause unexpected memory spikes.

    Args:
        comm: MPI communicator (``MPI.Comm`` instance) to allgather over.
        obj: Python object to allgather (must be picklable).
        chunk_size: Per-round max bytes each rank contributes (default:
            4 MB).

    Returns:
        List of deserialized objects from all ranks
        (``len == comm.size``).
    """
    if not ENABLE_MULTI_DEVICE:
        return [obj]
    if MPI is None:
        raise RuntimeError(
            "mpi4py is required when ENABLE_MULTI_DEVICE is True")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")

    # Step 1: serialize once and exchange lengths (1 MPI collective).
    rank, size, lengths, displs, sendbuf = \
        _serialize_and_exchange_lengths(comm, obj)

    total = int(lengths.sum())
    int32_max = np.iinfo(np.int32).max

    # Step 2a: total fits in int32 — single Allgatherv (1 MPI collective).
    if total < int32_max:
        counts = lengths.astype(np.int32)
        displs32 = displs.astype(np.int32)
        recvbuf = np.empty(total, dtype=np.uint8)
        comm.Allgatherv([sendbuf, MPI.BYTE],
                        [recvbuf, counts, displs32, MPI.BYTE])
        return _deserialize_recvbuf(recvbuf, lengths, displs, size)

    # Step 2b: total exceeds int32 — chunked Allgatherv.
    logger.info(
        "safe_allgather: total payload %d bytes exceeds int32 limit, "
        "using chunked Allgatherv (size=%d)", total, size)
    max_safe_chunk = int32_max // size
    chunk_size = min(chunk_size, max_safe_chunk)
    max_len = int(lengths.max())
    num_rounds = math.ceil(max_len / chunk_size) if max_len > 0 else 0

    recvbuf = np.empty(total, dtype=np.uint8)

    def _allgatherv(send_part, round_recvbuf, counts, round_displs):
        comm.Allgatherv([send_part, MPI.BYTE],
                        [round_recvbuf, counts, round_displs, MPI.BYTE])

    _chunked_transfer_loop(comm, rank, size, lengths, displs, sendbuf,
                           num_rounds, chunk_size, recvbuf, _allgatherv)

    return _deserialize_recvbuf(recvbuf, lengths, displs, size)


def _native_peer_failure_watch(*args):
    from tensorrt_llm.bindings.internal.runtime import \
        PeerFailureWatch as Native
    return Native(*args)


class PeerFailureWatch:
    """Background detection of a failed peer on a private communicator.

    A native thread polls a receive that no peer matches, without the GIL,
    every ``peer_failure_poll_s`` of the fault-tolerance config; ULFM fails it
    once a peer fails. On a failure the thread sets :attr:`failed`, then stores
    1 to ``abort_word``, then calls ``on_failure`` once.

    Args:
        comm: Communicator owned by the watch; freed by :meth:`stop`.
        on_failure: Called from the watch thread when a peer fails, or when
            detection itself fails.
        abort_word: Address of a host int32 the MoE all-to-all kernels poll
            as their abort request, or 0.
    """

    def __init__(self,
                 comm: "MPI.Comm",
                 on_failure: Callable[[], None],
                 abort_word: int = 0) -> None:
        self._comm = comm
        self._stopped = False

        def failure(error_class: int) -> None:
            # Release the waiters before the log write, which can stall.
            try:
                on_failure()
            finally:
                logger.warning(
                    f"Peer failure watch fired with MPI error class {error_class}"
                )

        from tensorrt_llm.executor.recovery.config import fault_tolerance_config
        self._native = _native_peer_failure_watch(
            comm.py2f(), abort_word, failure,
            fault_tolerance_config().peer_failure_poll_s)

    @property
    def failed(self) -> bool:
        """Whether a failure was detected; set before the abort word."""
        return self._native.failed

    def stop(self) -> None:
        """Stop watching and release the receive and the communicator.

        Idempotent; ``on_failure`` is not called after it returns. Called from
        ``on_failure`` itself, it only stops the watch; a later call releases.
        """
        if self._stopped:
            return
        if self._native.stop():
            self._stopped = True
            self._comm.Free()


def _coll_component_excluded(name: str) -> bool:
    """Whether Open MPI's environment keeps collective component ``name`` out: disabled, excluded, or not listed."""
    if os.environ.get(f"OMPI_MCA_coll_{name}_enable") in ("0", "false"):
        return True
    selection = os.environ.get("OMPI_MCA_coll", "")
    components = {c.strip() for c in selection.lstrip("^").split(",")}
    if selection.startswith("^"):
        return name in components
    return bool(selection) and name not in components


class MPIDist(Distributed):
    tp_comm: MPI.Comm

    def __init__(self, mapping: Mapping):
        super().__init__(mapping)
        self._cp_comm = None
        self._tp_comm = None
        self._pp_comm = None
        # Private world duplicate that recovery may revoke.
        self._host_world: Optional[MPI.Comm] = None
        # Survivor communicator and its logical ranks, from shrink().
        self._shrunk: Optional[tuple[MPI.Comm, tuple[int, ...]]] = None
        # The installed survivor communicator and its members' logical ranks.
        self._survivor_comm: Optional[MPI.Comm] = None
        self._live_ranks: Optional[tuple[int, ...]] = None

    @property
    def live_ranks(self) -> tuple[int, ...]:
        if self._live_ranks is None:
            return super().live_ranks
        return self._live_ranks

    def enable_peer_loss_recovery(self) -> None:
        """Move world-level host collectives to a private communicator.

        Collective over the world; call before any host collective that
        recovery may need to interrupt. Only pure attention-DP layouts
        (``tp_size == moe_ep_size == world_size``, no PP or CP) are supported;
        there the TP communicator spans the world.

        Raises:
            ValueError: The layout is unsupported.
            RuntimeError: The job was not launched with ULFM fault tolerance,
                the MPI library lacks the ULFM calls, or MPI was not
                initialized with ``MPI_THREAD_MULTIPLE``.
        """
        mapping = self.mapping
        if not (mapping.enable_attention_dp and mapping.pp_size == 1
                and mapping.cp_size == 1 and
                mapping.tp_size == mapping.moe_ep_size == mapping.world_size):
            raise ValueError(
                "Peer-loss recovery requires pure attention DP "
                "(tp_size == moe_ep_size == world_size, pp_size == cp_size == 1)"
            )
        if not hasattr(MPI.Comm, "Shrink"):
            raise RuntimeError("Peer-loss recovery requires mpi4py>=4")
        # Without fault tolerance enabled at launch, a lost process ends the
        # job even though the library exposes the ULFM calls.
        if os.environ.get("OMPI_MCA_mpi_ft_enable") not in ("1", "true"):
            raise RuntimeError(
                "Peer-loss recovery requires Open MPI with ULFM fault "
                "tolerance; launch with `mpirun --with-ft ulfm`")
        if MPI.Query_thread() != MPI.THREAD_MULTIPLE:
            raise RuntimeError(
                "Peer-loss recovery requires MPI_THREAD_MULTIPLE")
        if os.environ.get("OMPI_MCA_pml") != "ob1":
            raise RuntimeError(
                "Peer-loss recovery requires Open MPI's ob1 messaging layer; "
                "launch through trtllm-llmapi-launch or with `--mca pml ob1`")
        if not _coll_component_excluded("hcoll"):
            raise RuntimeError(
                "Peer-loss recovery requires Open MPI's hcoll collectives off; "
                "launch through trtllm-llmapi-launch or with `--mca coll ^hcoll,ucc`"
            )
        try:
            mpi_failed_ranks(mpi_comm())
            mpi_comm().Is_revoked()
        except NotImplementedError as error:
            raise RuntimeError(
                "Peer-loss recovery requires an MPI library with the ULFM "
                "calls (MPIX_Comm_get_failed, MPIX_Comm_revoke, "
                "MPIX_Comm_shrink); this one does not provide them") from error
        if self._host_world is None:
            self._host_world = _errors_return(mpi_comm().Dup())
            # Survivors must shrink the same communicator, so the TP
            # communicator, otherwise built on first use, exists from now on.
            _errors_return(self.tp_comm)

    def revoke(self) -> None:
        """Revoke the communicators host collectives run on so that every
        rank blocked in one of them raises :class:`HostCollectiveInterrupted`.

        Idempotent. Requires :meth:`enable_peer_loss_recovery` and an MPI
        library with ULFM support.
        """
        if self._host_world is None:
            raise RuntimeError("revoke() requires enable_peer_loss_recovery()")
        comms = [self._world_comm()]
        if self._live_ranks is None and self._tp_comm is not None:
            comms.append(self._tp_comm)
        if self._shrunk is not None:
            comms.append(self._shrunk[0])
        for comm in comms:
            if not comm.Is_revoked():
                comm.Revoke()

    def reported_failed_ranks(self) -> frozenset[int]:
        """The ranks this process's ULFM view reports failed in its host communicator.

        Local: no agreement with the other survivors (see :meth:`shrink`).

        Raises:
            RuntimeError: Recovery is not enabled.
        """
        if self._host_world is None:
            raise RuntimeError(
                "reported_failed_ranks() requires enable_peer_loss_recovery()")
        return mpi_failed_ranks(self._world_comm())

    def shrink(self) -> frozenset[int]:
        """Agree with the other survivors on the failed ranks.

        Collective over all surviving ranks; call after :meth:`revoke`. Builds
        the survivor communicator without using it yet and returns every rank
        failed so far, the same set on every survivor. Before
        :meth:`install_survivors` (a further loss during the recovery) it
        shrinks the communicator it built; after it, the installed one.

        Raises:
            RuntimeError: Recovery is not enabled.
        """
        if self._host_world is None:
            raise RuntimeError("shrink() requires enable_peer_loss_recovery()")
        if self._shrunk is not None:
            comm = self._shrunk[0]
        elif self._live_ranks is not None:
            comm = self._survivor_comm
        else:
            comm = self._tp_comm
        survivor_comm = _errors_return(comm.Shrink())
        live_ranks = mpi_world_ranks(survivor_comm.Get_group())
        self._shrunk = (survivor_comm, live_ranks)
        return frozenset(range(self.mapping.world_size)) - frozenset(live_ranks)

    def install_survivors(self) -> None:
        """Switch host collectives to the communicator built by :meth:`shrink`.

        Logical ranks keep their meaning: ``tp_rank``/``tp_size`` are unchanged,
        roots name logical ranks, and gathers return one entry per logical rank,
        with the caller's ``departed`` for failed ranks. The replaced
        communicator is not freed (freeing one with failed members is not
        guaranteed to be local).

        Raises:
            RuntimeError: :meth:`shrink` has not run.
        """
        if self._shrunk is None:
            raise RuntimeError("install_survivors requires a prior shrink()")
        survivor_comm, live_ranks = self._shrunk
        self._survivor_comm = survivor_comm
        self._live_ranks = live_ranks
        self._shrunk = None

    def watch_peer_failures(self,
                            on_failure: Callable[[], None],
                            abort_word: int = 0) -> "PeerFailureWatch":
        """Call *on_failure* once, from a native thread, when a peer fails
        before the returned watch is stopped; first store 1 to *abort_word*
        (an int32 address) if it is not 0.

        Covers time outside host collectives (e.g. GPU waits). Collective over
        the current (survivor) world: posts a receive no peer matches on a
        private duplicate, which ULFM completes with a process-failure error.

        Raises:
            RuntimeError: Recovery is not enabled.
        """
        if self._host_world is None:
            raise RuntimeError(
                "watch_peer_failures() requires enable_peer_loss_recovery()")
        return PeerFailureWatch(_errors_return(self._world_comm().Dup()),
                                on_failure, abort_word)

    def watch_survivor_failures(
            self, on_failure: Callable[[], None]) -> "PeerFailureWatch":
        """Like :meth:`watch_peer_failures`, on the survivor communicator that
        :meth:`shrink` built and :meth:`install_survivors` has not installed:
        a further peer failure during the recovery. Collective over the
        survivors.

        Raises:
            RuntimeError: :meth:`shrink` has not run.
        """
        if self._shrunk is None:
            raise RuntimeError(
                "watch_survivor_failures requires a prior shrink()")
        return PeerFailureWatch(_errors_return(self._shrunk[0].Dup()),
                                on_failure)

    def broadcast_among_survivors(self, obj, root: int = 0):
        """Broadcast *obj* from logical rank *root* over the survivor
        communicator that :meth:`shrink` built, before it is installed.
        Collective over the survivors.

        Raises:
            RuntimeError: :meth:`shrink` has not run.
            MPI.Exception: Another survivor failed, or the communicator was
                revoked (see ``is_mpi_peer_failure``).
        """
        if self._shrunk is None:
            raise RuntimeError(
                "broadcast_among_survivors requires a prior shrink()")
        survivor_comm, live_ranks = self._shrunk
        return survivor_comm.bcast(obj, root=live_ranks.index(root))

    def allgather_among_survivors(self, obj) -> list:
        """Gather *obj* from every survivor over the communicator that
        :meth:`shrink` built, before it is installed: one entry per logical
        rank, None for a failed one. Collective over the survivors.

        Raises:
            RuntimeError: :meth:`shrink` has not run.
            MPI.Exception: Another survivor failed, or the communicator was
                revoked (see ``is_mpi_peer_failure``).
        """
        if self._shrunk is None:
            raise RuntimeError(
                "allgather_among_survivors requires a prior shrink()")
        survivor_comm, live_ranks = self._shrunk
        gathered = [None] * self.mapping.world_size
        for rank, value in zip(live_ranks, survivor_comm.allgather(obj)):
            gathered[rank] = value
        return gathered

    def barrier_among_survivors(self) -> None:
        """Barrier over the survivor communicator that :meth:`shrink` built
        (``MPI.Exception`` if a survivor failed, see ``is_mpi_peer_failure``).

        Raises:
            RuntimeError: :meth:`shrink` has not run.
        """
        if self._shrunk is None:
            raise RuntimeError(
                "barrier_among_survivors requires a prior shrink()")
        self._shrunk[0].Barrier()

    def _world_comm(self) -> MPI.Comm:
        if self._live_ranks is not None:
            return self._survivor_comm
        if self._host_world is not None:
            return self._host_world
        return mpi_comm()

    def _comm_root(self, root: int) -> int:
        """Rank in the current host communicator of logical rank *root*."""
        if self._live_ranks is None:
            return root
        if root not in self._live_ranks:
            raise ValueError(f"Collective root {root} is not a live rank")
        return self._live_ranks.index(root)

    def _to_logical(self, gathered: list, departed, operation: str) -> list:
        """Place the survivors' entries at their logical ranks; departed ranks
        get *departed*, which the caller must have given."""
        if departed is DEPARTED_UNSPECIFIED:
            raise RuntimeError(
                f"{operation} after a peer loss needs departed=: the value that "
                "stands in for a rank that left the group")
        logical = [departed] * self.mapping.tp_size
        for rank, entry in zip(self._live_ranks, gathered):
            logical[rank] = entry
        return logical

    def _validate_world_size(self):
        """Validate world size before creating sub-communicators to prevent segfaults."""

        if ENABLE_MULTI_DEVICE:
            actual_world_size = mpi_world_size()
            max_rank_needed = self.mapping.world_size

            if max_rank_needed > actual_world_size:
                raise RuntimeError(
                    f"Mapping requires world_size={max_rank_needed} "
                    f"(tp_size={self.mapping.tp_size} * pp_size={self.mapping.pp_size} * cp_size={self.mapping.cp_size}), "
                    f"but MPI world size is only {actual_world_size}. ")

    @_interruptible
    def broadcast(self,
                  obj,
                  root=0,
                  chunk_size: int = 4 * 1024 * 1024,
                  prefer_cpu=False):
        # prefer_cpu is a no-op here: MPI object collectives are host-side.
        return safe_broadcast(self._world_comm(),
                              obj,
                              root=self._comm_root(root),
                              chunk_size=chunk_size)

    @_interruptible
    def allgather(self, obj, *, departed=DEPARTED_UNSPECIFIED):
        if self._host_world is None:
            return mpi_allgather(obj)
        gathered = self._world_comm().allgather(obj)
        if self._live_ranks is None:
            return gathered
        return self._to_logical(gathered, departed, "allgather")

    @property
    def local_world_size(self):
        return local_mpi_size()

    @_interruptible
    def barrier(self):
        if self._host_world is None:
            mpi_barrier()
        else:
            self._world_comm().Barrier()

    @_interruptible
    def tp_barrier(self):
        self.tp_comm.Barrier()

    def isend(self, buf: np.ndarray, dest, tag=0):
        # non-blocking send numpy buffer
        return mpi_isend(buf, dest, tag)

    def send(self, buf: np.ndarray, dest, tag=0):
        # blocking send numpy buffer
        mpi_send(buf, dest, tag)

    def recv(self, buf: np.ndarray, src, tag=0):
        # in-place recv numpy buffer
        return mpi_recv(buf, src, tag)

    def send_object(self, obj, dest, tag=0):
        mpi_send_object(obj, dest, tag)

    def isend_object(self, obj, dest, tag=0):
        return mpi_isend_object(obj, dest, tag)

    def recv_object(self, src, tag=0):
        return mpi_recv_object(src, tag)

    @property
    def tp_comm(self):
        if self._live_ranks is not None:
            return self._survivor_comm
        if self._tp_comm is None:
            self._validate_world_size()
            mapping = self.mapping
            new_group = mpi_comm().group.Incl(mapping.tp_group)
            self._tp_comm = mpi_comm().Create_group(new_group)
        return self._tp_comm

    @property
    def pp_comm(self):
        if self._pp_comm is None:
            self._validate_world_size()
            mapping = self.mapping
            new_group = mpi_comm().group.Incl(mapping.pp_group)
            self._pp_comm = mpi_comm().Create_group(new_group)
        return self._pp_comm

    @property
    def cp_comm(self):
        if self._cp_comm is None:
            self._validate_world_size()
            new_group = mpi_comm().group.Incl(self.mapping.cp_group)
            self._cp_comm = mpi_comm().Create_group(new_group)
        return self._cp_comm

    def cp_allgather(self,
                     obj,
                     chunk_size: int = 4 * 1024 * 1024,
                     *,
                     small_payload: bool = False):
        comm = self.cp_comm
        if small_payload:
            # mpi4py's native object allgather is cheaper for tiny payloads;
            # callers must guarantee the payload stays small on every rank.
            return comm.allgather(obj)
        return safe_allgather(comm, obj, chunk_size=chunk_size)

    def cp_broadcast(self,
                     obj,
                     root=0,
                     chunk_size: int = 4 * 1024 * 1024,
                     **kwargs):
        comm = self.cp_comm
        return safe_broadcast(comm, obj, root=root, chunk_size=chunk_size)

    @_interruptible
    def tp_allgather(self,
                     obj,
                     chunk_size: int = 4 * 1024 * 1024,
                     *,
                     small_payload: bool = False,
                     departed=DEPARTED_UNSPECIFIED):
        comm = self.tp_comm
        if small_payload:
            gathered = comm.allgather(obj)
        else:
            gathered = safe_allgather(comm, obj, chunk_size=chunk_size)
        if self._live_ranks is None:
            return gathered
        return self._to_logical(gathered, departed, "tp_allgather")

    @_interruptible
    def tp_gather(self,
                  obj,
                  root=0,
                  chunk_size: int = 4 * 1024 * 1024,
                  *,
                  departed=DEPARTED_UNSPECIFIED):
        comm = self.tp_comm
        gathered = safe_gather(comm,
                               obj,
                               root=self._comm_root(root),
                               chunk_size=chunk_size)
        if self._live_ranks is None or gathered is None:
            return gathered
        return self._to_logical(gathered, departed, "tp_gather")

    @_interruptible
    def tp_broadcast(self,
                     obj,
                     root=0,
                     chunk_size: int = 4 * 1024 * 1024,
                     **kwargs):
        return safe_broadcast(self.tp_comm,
                              obj,
                              root=self._comm_root(root),
                              chunk_size=chunk_size)

    def pp_allgather(self, obj, chunk_size: int = 4 * 1024 * 1024):
        comm = self.pp_comm
        return safe_allgather(comm, obj, chunk_size=chunk_size)

    def pp_gather(self, obj, root=0, chunk_size: int = 4 * 1024 * 1024):
        comm = self.pp_comm
        return safe_gather(comm, obj, root=root, chunk_size=chunk_size)

    def pp_broadcast(self, obj, root=0):
        return self.pp_comm.bcast(obj, root)

    @_interruptible
    def allreduce(self, obj, op: ReduceOp = ReduceOp.SUM):
        reduce_op = reduce_op_to_mpi(op)
        return self._world_comm().allreduce(obj, reduce_op)

    @_interruptible
    def tp_allreduce(self, obj, op: ReduceOp = ReduceOp.SUM):
        reduce_op = reduce_op_to_mpi(op)
        return self.tp_comm.allreduce(obj, reduce_op)

    @staticmethod
    def _allgather_int64_comm(comm, values: ArrayLike) -> np.ndarray:
        sendbuf = np.ascontiguousarray(
            np.asarray(values, dtype=np.int64).reshape(-1))
        size = comm.Get_size()
        recvbuf = np.empty(size * sendbuf.size, dtype=np.int64)
        comm.Allgather([sendbuf, MPI.INT64_T], [recvbuf, MPI.INT64_T])
        return recvbuf.reshape(size, sendbuf.size)

    @staticmethod
    def _broadcast_int64_comm(comm, values: ArrayLike, root: int) -> np.ndarray:
        buf = np.ascontiguousarray(
            np.asarray(values, dtype=np.int64).reshape(-1))
        comm.Bcast([buf, MPI.INT64_T], root=root)
        return buf

    @_interruptible
    def tp_allgather_int64(
            self,
            values: ArrayLike,
            *,
            departed: ArrayLike = DEPARTED_UNSPECIFIED) -> np.ndarray:
        rows = self._allgather_int64_comm(self.tp_comm, values)
        if self._live_ranks is None:
            return rows
        if departed is DEPARTED_UNSPECIFIED:
            raise RuntimeError(
                "tp_allgather_int64 needs departed= for the ranks that "
                "left the group after a peer loss")
        logical = np.empty((self.mapping.tp_size, rows.shape[1]),
                           dtype=np.int64)
        logical[:] = departed
        logical[list(self._live_ranks)] = rows
        return logical

    def cp_allgather_int64(self, values: ArrayLike) -> np.ndarray:
        return self._allgather_int64_comm(self.cp_comm, values)

    @_interruptible
    def broadcast_int64(self,
                        values: ArrayLike,
                        root: int = 0,
                        *,
                        prefer_cpu: bool = False) -> np.ndarray:
        # prefer_cpu is a no-op here: MPI buffer collectives are host-side.
        return self._broadcast_int64_comm(self._world_comm(), values,
                                          self._comm_root(root))

    @_interruptible
    def tp_cp_broadcast_int64(self,
                              values: ArrayLike,
                              root: int = 0) -> np.ndarray:
        buf = np.asarray(values, dtype=np.int64).reshape(-1)
        if self.tp_size > 1:
            buf = self._broadcast_int64_comm(self.tp_comm, buf,
                                             self._comm_root(root))
        if self.cp_size > 1:
            buf = self._broadcast_int64_comm(self.cp_comm, buf, root)
        return buf


class MultiHandleWrapper:
    """
    Wrapper that encapsulates multiple handles and provides a single wait() interface
    to unify the API between MPIDist and TorchDist.
    """

    def __init__(self, handles):
        self.handles = handles if isinstance(handles, list) else [handles]

    def wait(self):
        for handle in self.handles:
            try:
                handle.wait()
            except Exception as e:
                raise RuntimeError(f"Asynchronous operation failed: {e}") from e


class TorchDist(Distributed):

    @property
    def rank(self):
        return torch.distributed.get_rank()

    @property
    def local_world_size(self):
        return dist.get_world_size(group=self.local_comm)

    def __init__(self, mapping: Mapping):
        super().__init__(mapping)
        assert dist.is_initialized(
        ), "torch.distributed should be initialized before TorchDist"

        self.cluster_info = None

        # CUDA-tensor object-collective fast path (see _cuda_obj_pg).
        self._cuda_obj_pgs = {}
        self._cuda_obj_stream = None
        # Per-group staging state of that transport (size tensors, pinned
        # host buffers), guarded by one lock: one object collective at a
        # time per process (see _cuda_obj_pg).
        self._cuda_obj_lock = threading.Lock()
        self._cuda_obj_staging = {}

        from tensorrt_llm._utils import set_torch_comm
        set_torch_comm(self)  # Set as global instance
        mapping.build_mesh()
        # Resolve group names before Dynamo tracing.
        mapping.tp_group_name
        mapping.cp_group_name

        self.setup_local_comm()
        self.default_store = torch.distributed.distributed_c10d._get_default_store(
        )

        init_pg(torch.distributed.group.WORLD, self.local_comm,
                torch_pybind11_abi())

        self._init_cuda_object_groups()

    def _init_cuda_object_groups(self):
        """Create dedicated NCCL groups for the CUDA object collectives.

        These must NOT share process groups with the model: when MPI is
        disabled the model's C++ collectives receive the mapping's groups
        (ops.py boxes tp_group_pg / cp_group_pg into the allreduce /
        allgather / reduce-scatter ops), and a process group owns exactly
        one NCCL communicator, created lazily by its first collective -- so
        any communicator tuning applied for the tiny vote payloads would
        become the model's too.  NCCL additionally caches most env knobs
        process-wide on first read (NCCL_PARAM), so per-communicator tuning
        must go through ProcessGroupNCCL.Options, never scoped env vars.

        Creating the groups eagerly also allocates their communicator GPU
        buffers before KV-cache sizing probes free memory, and fails fast
        on misconfiguration.
        """
        if not torch.cuda.is_available():
            return
        torch.cuda.init()

        member_dims = {"world": tuple(range(self.mapping.world_size))}
        if self.mapping.tp_size > 1:
            member_dims["tp"] = tuple(self.mapping.tp_group)
        if self.mapping.pp_size > 1:
            member_dims["pp"] = tuple(self.mapping.pp_group)
        if self.mapping.cp_size > 1:
            member_dims["cp"] = tuple(self.mapping.cp_group)

        # new_group is collective: every rank must create every subgroup of
        # every dim, in the same order.  Gather the full partitions, and
        # piggyback the kill switch so its value is a WORLD consensus: the
        # failure the switch exists to dodge happens at communicator init /
        # first collective (seen in production), so a disabled run must skip group
        # creation entirely -- and ranks disagreeing on that would deadlock
        # in new_group.  This gather rides the CPU (gloo) path and is safe
        # either way.
        local_enabled = os.environ.get("TLLM_CUDA_OBJECT_COLLECTIVES",
                                       "1") != "0"
        gathered = [None] * self.mapping.world_size
        torch.distributed.all_gather_object(gathered,
                                            (member_dims, local_enabled))
        all_dims = [dims for dims, _ in gathered]
        if not all(enabled for _, enabled in gathered):
            logger.info(
                "TorchDist: CUDA object collectives disabled "
                "(TLLM_CUDA_OBJECT_COLLECTIVES=0); using CPU-backend object "
                "collectives")
            return

        def _small_comm_options():
            # Vote payloads are bytes-to-KB: a small CTA budget keeps these
            # communicators' spin-waiting kernels off the model's SMs.
            try:
                options = dist.ProcessGroupNCCL.Options()
                options.config.max_ctas = 4
                return options
            except (AttributeError, RuntimeError):
                return None

        rank = torch.distributed.get_rank()
        created = {}
        for dim in ("world", "tp", "pp", "cp"):
            for ranks in sorted({d[dim] for d in all_dims if dim in d}):
                pg = created.get(ranks)
                if pg is None:
                    pg = dist.new_group(ranks=list(ranks),
                                        backend="nccl",
                                        pg_options=_small_comm_options())
                    created[ranks] = pg
                if rank in ranks and dim not in self._cuda_obj_pgs:
                    self._cuda_obj_pgs[dim] = pg
        # First collective per communicator: allocates its GPU buffers now
        # (before KV sizing) and validates the path end to end.
        for pg in dict.fromkeys(self._cuda_obj_pgs.values()):
            self._cuda_allgather_object(0, pg)
        logger.info(f"TorchDist: CUDA object-collective groups ready for "
                    f"{sorted(self._cuda_obj_pgs)} (max_ctas=4)")

    def setup_local_comm(self):
        self._get_cluster_info()

        # node IP -> list of ranks
        ip_to_ranks = {}
        for rank, (node_ip, _) in enumerate(self.cluster_info):
            ip_to_ranks.setdefault(node_ip, []).append(int(rank))

        self.local_comm = None
        for ranks in ip_to_ranks.values():
            # All global ranks from the default process group to participate in the call,
            # even if some ranks are not part of the new process group being created
            pg = dist.new_group(ranks=ranks, backend='cuda:nccl,cpu:gloo')
            if int(self.rank) in ranks:
                logger.debug(
                    f"[Rank {self.rank}] Done setting local comm. ip_to_ranks: {ip_to_ranks}"
                )
                self.local_comm = pg

    def _get_cluster_info(self):
        if self.cluster_info is not None:
            return self.cluster_info

        if ray.is_initialized():
            node_ip = ray.util.get_node_ip_address()
        else:
            raise RuntimeError("Ray is not initialized")

        gpu_index = [int(id) for id in ray.get_gpu_ids()]

        assert len(gpu_index) == 1

        # Gather node ip
        node_list = [None] * torch.distributed.get_world_size()

        torch.distributed.all_gather_object(node_list, node_ip)

        # Gather gpu index
        gpu_list = [None] * torch.distributed.get_world_size()
        torch.distributed.all_gather_object(gpu_list, gpu_index[0])

        # Gather rank
        rank_list = [None] * torch.distributed.get_world_size()
        torch.distributed.all_gather_object(rank_list,
                                            torch.distributed.get_rank())

        rank_info_list = [None] * torch.distributed.get_world_size()
        for i in range(len(rank_list)):
            rank_info_list[rank_list[i]] = (node_list[i], gpu_list[i])

        self.cluster_info = rank_info_list

        logger.debug(f"Cluster info: {self.cluster_info}")
        return self.cluster_info

    @staticmethod
    def log_op(func, enable_log=False):

        @wraps(func)
        def wrapper(*args, **kwargs):
            if enable_log:
                logger.debug(
                    f"{func.__name__} enter: {args[1:]}, {kwargs}, rank: {torch.distributed.get_rank()}"
                )
            ret = func(*args, **kwargs)

            if enable_log:
                logger.debug(f"{func.__name__} exit: {ret}")
            return ret

        return wrapper

    # --- CUDA-tensor object collectives ------------------------------------
    #
    # torch's object collectives (all_gather_object and friends) stage the
    # pickled payload through the process group's CPU backend whenever one is
    # registered, even if the group also has NCCL.  Outside MPI that CPU
    # backend is gloo over TCP: a 16-rank object allgather costs milliseconds,
    # and several of these collectives sit on the executor hot loop (the
    # attention-DP can_queue / cuda-graph / disagg-error votes and response
    # gather fire every iteration), where the MPI path costs microseconds.
    # Staging the pickle through CUDA tensors instead lets the payload ride
    # NCCL.  A dedicated side stream keeps the collective from ordering
    # behind model kernels already queued on the current stream (the NCCL
    # input-readiness event is recorded on the calling stream).
    # TLLM_CUDA_OBJECT_COLLECTIVES=0 restores the CPU path.
    #
    # Inside this transport every branch is decided by gathered or config
    # data, never by rank-local state, so all ranks issue the same NCCL
    # call sequence:
    #   * empty-list sentinel: a ``[]`` payload is sent as size 0 (which
    #     pickle.dumps never produces); when the gathered sizes are all 0
    #     every rank skips the payload round and its host sync;
    #   * gather semantics: the destination rank alone unpickles the
    #     gathered payloads, other ranks return None (wire unchanged);
    #   * staging: persistent per-group int64 size tensors and pinned host
    #     buffers instead of pageable H2D copies of the size and of CPU
    #     tensor payloads.

    def _cuda_obj_pg(self, dim: str):
        """Vote group for ``dim`` when the CUDA object path may be used.

        Returns the dedicated NCCL group created by
        ``_init_cuda_object_groups`` or None to fall back to the CPU path.
        All checks are rank-consistent (the groups dict is built by WORLD
        consensus at init -- including the TLLM_CUDA_OBJECT_COLLECTIVES
        kill switch -- and graph-capture state follows lockstep code
        paths), which matters: ranks disagreeing on the transport for the
        same collective would deadlock.

        Contract: one object collective at a time per process.  The side
        stream and the staging buffers are shared by every group and are
        not re-entrant; ``_cuda_obj_guard`` (``_cuda_obj_lock``) wraps
        all-gather, gather, CPU-tensor all-gather, broadcast and scalar
        all-reduce alike (uncontended on the executor thread).  Two
        concurrent collectives on one NCCL group would be undefined on any
        path, so the lock cannot introduce a new deadlock.
        """
        pg = self._cuda_obj_pgs.get(dim)
        if pg is None:
            return None
        if torch.cuda.is_current_stream_capturing():
            return None
        return pg

    def _cuda_obj_stream_ctx(self):
        if self._cuda_obj_stream is None:
            self._cuda_obj_stream = torch.cuda.Stream()
        return torch.cuda.stream(self._cuda_obj_stream)

    def _cuda_obj_guard(self):
        # Taken by every CUDA object collective (all-gather, gather, CPU-
        # tensor all-gather, broadcast, scalar all-reduce): they share the
        # side stream and the staging buffers.
        return self._cuda_obj_lock

    def _cuda_obj_staging_for(self, pg) -> dict:
        """Persistent staging state for ``pg`` (lock held).

        ``size_in`` / ``size_out`` are the int64 CUDA tensors of the size
        exchange, ``pinned_size`` the pinned host mirror of the local size,
        and ``pinned`` one grown-on-demand pinned host buffer per dtype for
        CPU-tensor payloads.  Allocated once per group; steady-state calls
        (fixed RankState / packed-vote widths) never reallocate.
        """
        staging = self._cuda_obj_staging.get(pg)
        if staging is None:
            staging = {
                "size_in":
                torch.empty(1, dtype=torch.int64, device="cuda"),
                "size_out":
                torch.empty(pg.size(), dtype=torch.int64, device="cuda"),
                "pinned_size":
                torch.empty(1, dtype=torch.int64, pin_memory=prefer_pinned()),
                "pinned": {},
                # Broadcast staging (root side). Separate from the allgather
                # buffers above: the root's copies out of these are
                # asynchronous and only fenced by ``bcast_fence``, while the
                # allgather mirrors are rewritten by the host immediately.
                "bcast_size":
                torch.empty(1, dtype=torch.int64, device="cuda"),
                "bcast_pinned_size":
                torch.empty(1, dtype=torch.int64, pin_memory=prefer_pinned()),
                "bcast_pinned":
                None,
                "bcast_fence":
                None,
            }
            self._cuda_obj_staging[pg] = staging
        return staging

    @staticmethod
    def _cuda_obj_pinned(staging: dict, numel: int,
                         dtype: torch.dtype) -> torch.Tensor:
        buf = staging["pinned"].get(dtype)
        if buf is None or buf.numel() < numel:
            buf = torch.empty(max(numel, 1),
                              dtype=dtype,
                              pin_memory=prefer_pinned())
            staging["pinned"][dtype] = buf
        return buf[:numel]

    def _cuda_allgather_object(self,
                               obj,
                               pg,
                               decode: bool = True) -> Optional[list]:
        """All-gather a pickled object through CUDA staging.

        ``decode=False`` issues the very same collectives but returns None
        without unpickling -- ``_cuda_gather_object`` uses it on
        non-destination ranks.
        """
        group_size = pg.size()
        if type(obj) is list and not obj:
            # Empty-list sentinel: size 0 is unambiguous because pickle
            # never yields 0 bytes.  Type-exact on purpose (an empty tuple
            # still pickles); mixed []/() across ranks stays correct.
            local = None
            local_size = 0
        else:
            local = torch.frombuffer(bytearray(
                pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)),
                                     dtype=torch.uint8)
            local_size = local.numel()
        with self._cuda_obj_guard(), self._cuda_obj_stream_ctx():
            staging = self._cuda_obj_staging_for(pg)
            size_in = staging["size_in"]
            size_out = staging["size_out"]
            staging["pinned_size"][0] = local_size
            size_in.copy_(staging["pinned_size"], non_blocking=True)
            dist.all_gather_into_tensor(size_out, size_in, group=pg)
            sizes = size_out.cpu()  # synchronizes the side stream
            max_size = int(sizes.max())
            if max_size == 0:
                # Every rank sent the sentinel.  Decided by the gathered
                # sizes, so the skipped payload round is rank-consistent.
                return [[] for _ in range(group_size)] if decode else None
            send = torch.empty(max_size, dtype=torch.uint8, device="cuda")
            if local_size:
                send[:local_size].copy_(local)
            recv = torch.empty(group_size * max_size,
                               dtype=torch.uint8,
                               device="cuda")
            dist.all_gather_into_tensor(recv, send, group=pg)
            if not decode:
                # Buffers are freed stream-ordered; no host sync needed.
                return None
            flat = recv.cpu()
        return [
            [] if int(sizes[i]) == 0 else pickle.loads(  # nosec B301
                flat[i * max_size:i * max_size +
                     int(sizes[i])].numpy().tobytes())
            for i in range(group_size)
        ]

    def _cuda_gather_object(self, obj, pg, dst: int):
        # Latency-equivalent to allgather at these payload sizes; reusing the
        # allgather path keeps NCCL usage on one well-tested collective.
        # Every rank still receives the bytes (wire protocol unchanged) but
        # only the destination unpickles them.
        is_dst = torch.distributed.get_rank() == dst
        gathered = self._cuda_allgather_object(obj, pg, decode=is_dst)
        return gathered if is_dst else None

    def _cuda_allgather_cpu_tensor(self, tensor: torch.Tensor,
                                   pg) -> List[torch.Tensor]:
        """Stage a CPU-tensor allgather through CUDA so it rides NCCL.

        Same contract as the direct tensor path (equal shapes on every rank,
        CPU tensors returned); one NCCL round instead of a gloo round.
        """
        group_size = pg.size()
        with self._cuda_obj_guard(), self._cuda_obj_stream_ctx():
            # Pinned staging: a pageable .cuda() blocks the host for the
            # whole copy; from pinned memory it is asynchronous on the side
            # stream, and the .cpu() below fences it before the buffer can
            # be reused (the lock is held until then).
            staging = self._cuda_obj_staging_for(pg)
            pinned = self._cuda_obj_pinned(staging, tensor.numel(),
                                           tensor.dtype)
            pinned.copy_(tensor.reshape(-1))
            send = torch.empty(tuple(tensor.shape),
                               dtype=tensor.dtype,
                               device="cuda")
            send.view(-1).copy_(pinned, non_blocking=True)
            recv = torch.empty((group_size, ) + tuple(tensor.shape),
                               dtype=tensor.dtype,
                               device="cuda")
            dist.all_gather_into_tensor(recv, send, group=pg)
            host = recv.cpu()
        return list(host.unbind(0))

    def _cuda_broadcast_object(self,
                               obj,
                               pg,
                               root: int,
                               allow_none: bool = False):
        """Broadcast a pickled object through CUDA staging.

        The root does not wait for the current call: size and payload are
        staged through pinned host buffers with non-blocking copies issued
        BEFORE the two ``dist.broadcast`` calls, and the root never reads a
        collective back. (The previous form read ``size.cpu()`` on the root
        and used pageable H2D copies, each a stream sync: the request-fetch
        root paid the full rank skew on every iteration.) The fence recorded
        after the copies orders the next call's host rewrite of the pinned
        buffers behind them. Those copies are queued on the side stream
        behind the previous call's NCCL completion, so ``fence.synchronize()``
        in call N+1 can block until every peer has joined call N-1: the root
        runs at most one call ahead of the slowest peer (the executor issues
        one such call per iteration, so in practice it never blocks), and it
        is never held on the call it is issuing.

        ``allow_none``: the root passes ``None`` to send the size-0 sentinel;
        every rank then skips the payload round and returns ``None``.
        Unambiguous because pickle never yields 0 bytes; without the flag
        ``None`` pickles like any other object.
        """
        is_root = torch.distributed.get_rank() == root
        with self._cuda_obj_guard(), self._cuda_obj_stream_ctx():
            staging = self._cuda_obj_staging_for(pg)
            size = staging["bcast_size"]
            if is_root:
                if allow_none and obj is None:
                    data = b""
                else:
                    data = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
                nbytes = len(data)
                fence = staging["bcast_fence"]
                if fence is not None:
                    # Previous call's copies out of the pinned buffers (and,
                    # by stream order, the call before that's broadcast
                    # kernels -- see the docstring).
                    fence.synchronize()
                pinned_size = staging["bcast_pinned_size"]
                pinned_size[0] = nbytes
                size.copy_(pinned_size, non_blocking=True)
                buf = None
                if nbytes:
                    pinned = staging["bcast_pinned"]
                    if pinned is None or pinned.numel() < nbytes:
                        pinned = torch.empty(nbytes,
                                             dtype=torch.uint8,
                                             pin_memory=prefer_pinned())
                        staging["bcast_pinned"] = pinned
                    pinned[:nbytes].numpy()[:] = np.frombuffer(data,
                                                               dtype=np.uint8)
                    buf = torch.empty(nbytes, dtype=torch.uint8, device="cuda")
                    buf.copy_(pinned[:nbytes], non_blocking=True)
                fence = torch.cuda.Event()
                fence.record()
                staging["bcast_fence"] = fence
                dist.broadcast(size, src=root, group=pg)
                if buf is not None:
                    dist.broadcast(buf, src=root, group=pg)
                return obj
            dist.broadcast(size, src=root, group=pg)
            nbytes = int(size.cpu())
            if nbytes == 0:
                return None
            buf = torch.empty(nbytes, dtype=torch.uint8, device="cuda")
            dist.broadcast(buf, src=root, group=pg)
            data = buf.cpu()
        return pickle.loads(data.numpy().tobytes())  # nosec B301

    @log_op
    def broadcast(self, obj, root=0, prefer_cpu=False):
        assert not (self.mapping.has_cp_ulysses() and self.mapping.has_tp()
                    ), 'Unsupported mix of Ulysses CP and TP.'

        if mpi_disabled():
            # prefer_cpu: a long-parking collective must not hold a spinning
            # NCCL kernel on the GPU (see the base-class docstring) -- fall
            # through to the gloo object path. Rank-consistency: callers must
            # pass the same value on every rank (the idle flag they derive it
            # from already is), or the transports diverge and deadlock.
            vote_pg = None if prefer_cpu else self._cuda_obj_pg("world")
            if isinstance(obj, torch.Tensor):
                dist.broadcast(obj, src=root)
                return obj
            elif vote_pg is not None:
                return self._cuda_broadcast_object(obj, vote_pg, root)
            else:
                obj_list = [obj]
                dist.broadcast_object_list(obj_list, src=root)
                return obj_list[0]

        if self.mapping.has_cp_ulysses():
            self.broadcast_cp(obj, root)
        elif self.mapping.has_tp():
            self.broadcast_tp(obj, root)

    @property
    def supports_single_round_broadcast(self) -> bool:
        # Rank-consistent: mpi_disabled is process-wide configuration and the
        # group dict is built by WORLD consensus (graph-capture state is
        # lockstep), so every rank sees the same value at the same call.
        return mpi_disabled() and self._cuda_obj_pg("world") is not None

    @log_op
    def broadcast_or_none(self, obj, root=0):
        vote_pg = self._cuda_obj_pg("world") if mpi_disabled() else None
        if vote_pg is not None:
            return self._cuda_broadcast_object(obj,
                                               vote_pg,
                                               root,
                                               allow_none=True)
        return super().broadcast_or_none(obj, root=root)

    @log_op
    def allgather(self, obj, *, departed=DEPARTED_UNSPECIFIED):
        # Peer-loss recovery runs only on MPIDist: no rank departs here.
        vote_pg = self._cuda_obj_pg("world")
        if isinstance(obj, torch.Tensor):
            output_list = [
                torch.empty_like(obj) for _ in range(self.world_size)
            ]
            dist.all_gather(output_list, obj)
            return output_list
        elif vote_pg is not None:
            return self._cuda_allgather_object(obj, vote_pg)
        else:
            obj_list = [None] * self.world_size
            dist.all_gather_object(obj_list, obj)
            return obj_list

    @log_op
    def barrier(self):
        dist.barrier()

    @log_op
    def tp_barrier(self):
        dist.barrier(group=self.mapping.tp_group_pg)

    @log_op
    def isend(self, buf: np.ndarray, dest, tag=0):
        # non-blocking send numpy buffer
        tensor = torch.from_numpy(buf)
        return dist.isend(tensor, dst=dest, tag=tag)

    @log_op
    def send(self, buf: np.ndarray, dest, tag=0):
        raise NotImplementedError(
            "blocking send is not implemented for TorchDist")

    @log_op
    def recv(self, buf: np.ndarray, src, tag=0):
        # in-place recv numpy buffer
        tensor = torch.empty_like(torch.from_numpy(buf))
        dist.recv(tensor, src=src, tag=tag)
        return tensor.numpy()

    @log_op
    def isend_tensor(self, tensor: torch.Tensor, dest, tag=0):
        return dist.isend(tensor, dst=dest, tag=tag)

    @log_op
    def recv_tensor(self, tensor: torch.Tensor, src, tag=0):
        dist.recv(tensor, src=src, tag=tag)
        return tensor

    @log_op
    def recv_object(self, src, tag=0):
        size_tensor = torch.tensor([0], dtype=torch.int32)
        torch.distributed.recv(size_tensor,
                               src=src,
                               tag=tag,
                               group=torch.distributed.group.WORLD)
        bytes_size = size_tensor.item()
        recv_tensor = torch.empty(bytes_size, dtype=torch.uint8)
        torch.distributed.recv(recv_tensor,
                               src=src,
                               tag=tag,
                               group=torch.distributed.group.WORLD)
        return _tensor_to_object(recv_tensor, bytes_size,
                                 torch.distributed.group.WORLD)

    @log_op
    def send_object(self, obj, dest, tag=0):
        self.isend_object(obj, dest, tag).wait()

    @log_op
    def isend_object(self, obj, dest, tag=0):
        input_tensor, local_size = _object_to_tensor(
            obj, torch.device("cpu"), torch.distributed.group.WORLD)

        # Send object size
        works = []
        works.append(
            torch.distributed.isend(torch.tensor([local_size],
                                                 dtype=torch.int32),
                                    dst=dest,
                                    tag=tag))
        works.append(torch.distributed.isend(input_tensor, dst=dest, tag=tag))
        return MultiHandleWrapper(works)

    # NCCL has no bitwise reduce ops; those stay on the CPU path.
    _CUDA_SCALAR_REDUCE_OPS = (ReduceOp.SUM, ReduceOp.PRODUCT, ReduceOp.MIN,
                               ReduceOp.MAX)

    def _cuda_allreduce_scalar(self, obj: int | float, op: ReduceOp, pg):
        # The wire dtype must be rank-uniform even though callers may
        # legitimately disagree on int vs float for the same reduction
        # (e.g. min(config_int, computed_float) in KV-cache sizing), so
        # reduce in float64 regardless of the local python type -- exact
        # for ints up to 2**53, far beyond these scalar votes.
        with self._cuda_obj_guard(), self._cuda_obj_stream_ctx():
            reduced = torch.tensor([float(obj)],
                                   dtype=torch.float64,
                                   device="cuda")
            dist.all_reduce(reduced, op=reduce_op_to_torch(op), group=pg)
            result = reduced.cpu().item()
        return int(result) if isinstance(obj, int) else result

    @log_op
    def allreduce(
        self,
        obj: int | float | torch.Tensor,
        op: ReduceOp = ReduceOp.SUM,
    ):
        is_base_type = isinstance(obj, int) or isinstance(obj, float)
        if is_base_type:
            vote_pg = self._cuda_obj_pg("world")
            if op in self._CUDA_SCALAR_REDUCE_OPS and vote_pg is not None:
                return self._cuda_allreduce_scalar(obj, op, vote_pg)
            obj = torch.tensor(obj)

        dist.all_reduce(obj, op=reduce_op_to_torch(op))

        if is_base_type:
            obj = obj.item()

        return obj

    @log_op
    def tp_allreduce(
        self,
        obj: int | float | torch.Tensor,
        op: ReduceOp = ReduceOp.SUM,
    ):
        is_base_type = isinstance(obj, int) or isinstance(obj, float)
        if is_base_type:
            vote_pg = self._cuda_obj_pg("tp")
            if op in self._CUDA_SCALAR_REDUCE_OPS and vote_pg is not None:
                return self._cuda_allreduce_scalar(obj, op, vote_pg)
            obj = torch.tensor(obj)

        dist.all_reduce(obj,
                        op=reduce_op_to_torch(op),
                        group=self.mapping.tp_group_pg)

        if is_base_type:
            obj = obj.item()

        return obj

    @log_op
    def tp_allgather(self,
                     obj,
                     *,
                     small_payload: bool = False,
                     departed=DEPARTED_UNSPECIFIED):
        vote_pg = self._cuda_obj_pg("tp")
        if isinstance(obj, torch.Tensor):
            if not obj.is_cuda and vote_pg is not None:
                return self._cuda_allgather_cpu_tensor(obj, vote_pg)
            output_list = [
                torch.empty_like(obj)
                for _ in range(self.mapping.tp_group_pg.size())
            ]
            dist.all_gather(output_list, obj, group=self.mapping.tp_group_pg)
            return output_list
        elif vote_pg is not None:
            return self._cuda_allgather_object(obj, vote_pg)
        else:
            output_list = [None] * self.mapping.tp_group_pg.size()
            dist.all_gather_object(output_list,
                                   obj,
                                   group=self.mapping.tp_group_pg)
            return output_list

    def tp_allgather_int64(
            self,
            values: ArrayLike,
            *,
            departed: ArrayLike = DEPARTED_UNSPECIFIED) -> np.ndarray:
        """Fixed-size int64 TP all-gather on the CUDA-staged CPU-tensor path.

        The base implementation ships the vector as a pickled object, which on
        this backend is the two-round object path (size exchange + payload:
        two NCCL kernels and two host syncs per call). The vector is fixed
        width by contract, so it can ride ``_cuda_allgather_cpu_tensor``
        instead -- one kernel and one sync, the same transport as the
        attention-DP RankState exchange. Falls back to the base path when the
        object-collective groups are unavailable (kill switch, graph capture),
        a rank-consistent condition (see ``_cuda_obj_pg``). Peer-loss recovery
        runs only on MPIDist, so no rank departs and ``departed`` is unused.
        """
        vote_pg = self._cuda_obj_pg("tp")
        if vote_pg is None:
            return super().tp_allgather_int64(values, departed=departed)
        vec = torch.from_numpy(
            np.ascontiguousarray(
                np.asarray(values, dtype=np.int64).reshape(-1)))
        rows = self._cuda_allgather_cpu_tensor(vec, vote_pg)
        return torch.stack(rows).numpy().astype(np.int64, copy=False).reshape(
            len(rows), -1)

    @log_op
    def tp_gather(self, obj, dst=0, *, departed=DEPARTED_UNSPECIFIED):
        global_rank = torch.distributed.get_rank()
        if isinstance(obj, torch.Tensor):
            if global_rank == dst:
                output_list = [
                    torch.empty_like(obj)
                    for _ in range(self.mapping.tp_group_pg.size())
                ]
            else:
                output_list = None
            dist.gather(obj,
                        output_list,
                        dst=dst,
                        group=self.mapping.tp_group_pg)
            return output_list
        elif (vote_pg := self._cuda_obj_pg("tp")) is not None:
            return self._cuda_gather_object(obj, vote_pg, dst)
        else:
            output_list = [None] * self.mapping.tp_group_pg.size()
            if global_rank == dst:
                output_list = [None] * self.mapping.tp_group_pg.size()
            else:
                output_list = None
            dist.gather_object(obj,
                               output_list,
                               dst=dst,
                               group=self.mapping.tp_group_pg)
            return output_list

    @log_op
    def tp_broadcast(self, obj, root=0, **kwargs):
        if isinstance(obj, torch.Tensor):
            dist.broadcast(obj, src=root, group=self.mapping.tp_group_pg)
            return obj
        elif (vote_pg := self._cuda_obj_pg("tp")) is not None:
            return self._cuda_broadcast_object(obj, vote_pg, root)
        else:
            ret = [obj]
            torch.distributed.broadcast_object_list(
                ret,
                src=root,
                group=self.mapping.tp_group_pg,
                device=torch.device("cpu"))
            return ret[0]

    @log_op
    def cp_broadcast(self, obj, root=0, **kwargs):
        if isinstance(obj, torch.Tensor):
            dist.broadcast(obj, src=root, group=self.mapping.cp_group_pg)
            return obj
        elif (vote_pg := self._cuda_obj_pg("cp")) is not None:
            return self._cuda_broadcast_object(obj, vote_pg, root)
        else:
            ret = [obj]
            torch.distributed.broadcast_object_list(
                ret,
                src=root,
                group=self.mapping.cp_group_pg,
                device=torch.device("cpu"))
            return ret[0]

    @log_op
    def cp_allgather(self, obj, *, small_payload: bool = False):
        if isinstance(obj, torch.Tensor):
            output_list = [
                torch.empty_like(obj)
                for _ in range(self.mapping.cp_group_pg.size())
            ]
            dist.all_gather(output_list, obj, group=self.mapping.cp_group_pg)
            return output_list
        elif (vote_pg := self._cuda_obj_pg("cp")) is not None:
            return self._cuda_allgather_object(obj, vote_pg)
        else:
            output_list = [None] * self.mapping.cp_group_pg.size()
            dist.all_gather_object(output_list,
                                   obj,
                                   group=self.mapping.cp_group_pg)
            return output_list

    @log_op
    def pp_allgather(self, obj):
        if isinstance(obj, torch.Tensor):
            output_list = [
                torch.empty_like(obj)
                for _ in range(self.mapping.pp_group_pg.size())
            ]
            dist.all_gather(output_list, obj, group=self.mapping.pp_group_pg)
            return output_list
        elif (vote_pg := self._cuda_obj_pg("pp")) is not None:
            return self._cuda_allgather_object(obj, vote_pg)
        else:
            output_list = [None] * self.mapping.pp_group_pg.size()
            dist.all_gather_object(output_list,
                                   obj,
                                   group=self.mapping.pp_group_pg)
            return output_list

    @log_op
    def pp_gather(self, obj, dst=0):
        global_rank = torch.distributed.get_rank()
        if isinstance(obj, torch.Tensor):
            if global_rank == dst:
                output_list = [
                    torch.empty_like(obj)
                    for _ in range(self.mapping.pp_group_pg.size())
                ]
            else:
                output_list = None
            dist.gather(obj,
                        output_list,
                        dst=dst,
                        group=self.mapping.pp_group_pg)
            return output_list
        elif (vote_pg := self._cuda_obj_pg("pp")) is not None:
            return self._cuda_gather_object(obj, vote_pg, dst)
        else:
            output_list = [None] * self.mapping.pp_group_pg.size()
            if global_rank == dst:
                output_list = [None] * self.mapping.pp_group_pg.size()
            else:
                output_list = None
            dist.gather_object(obj,
                               output_list,
                               dst=dst,
                               group=self.mapping.pp_group_pg)
            return output_list

    @log_op
    def pp_broadcast(self, obj, root=0):
        if isinstance(obj, torch.Tensor):
            dist.broadcast(obj, src=root, group=self.mapping.pp_group_pg)
            return obj
        elif (vote_pg := self._cuda_obj_pg("pp")) is not None:
            return self._cuda_broadcast_object(obj, vote_pg, root)
        else:
            ret = [obj]
            torch.distributed.broadcast_object_list(
                ret,
                src=root,
                group=self.mapping.pp_group_pg,
                device=torch.device("cpu"))
            return ret[0]


class PPCommNCCL:

    def __init__(self, global_mapping: Mapping):
        self.mapping = global_mapping
        self.nccl_comm = torch.classes.trtllm.NcclCommunicatorOp(
            self.mapping.world_size,
            self.mapping.rank,
        )
        self.tensor_ready_event = torch.cuda.Event()
        self.send_stream = torch.cuda.Stream()

    def send(self, tensor: torch.Tensor, dest: Optional[int] = None):
        if dest is None:
            dest = self.mapping.next_pp_rank()

        # NCCL send kernel in send_stream cannot be captured,
        # so we send in the current stream instead in CUDA graph cases.
        if torch.cuda.is_current_stream_capturing():
            self.nccl_comm.send(tensor, dest)
            return

        # If the tensor is allocated from non-default memory pool
        # like userbuffers, its underlying memory may be reused
        # before the send operation is completed.
        # We clone the tensor to avoid write-write conflicts.
        tensor = tensor.clone()
        self.send_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(self.send_stream):
            self.nccl_comm.send(tensor, dest)

    def recv(self, tensor: torch.Tensor, src: Optional[int] = None):
        if src is None:
            src = self.mapping.prev_pp_rank()
        self.nccl_comm.recv(tensor, src)


class PPCommTorch:

    def __init__(self, global_mapping: Mapping):
        self.mapping = global_mapping
        self.pg = self.mapping.pp_group_pg
        self.pg_group = self.mapping.pp_group

    def _global_to_local_rank(self, global_rank: int):
        assert global_rank in self.pg_group
        return self.pg_group.index(global_rank)

    def send(self, tensor: torch.Tensor, dest: Optional[int] = None):
        if dest is None:
            dest = self.mapping.next_pp_rank()

        work = self.pg.send([tensor], self._global_to_local_rank(dest), tag=0)
        # Send operation cannot be captured without blocking wait,
        # so we block the current stream in CUDA graph cases.
        if torch.cuda.is_current_stream_capturing():
            work.block_current_stream()

    def recv(self, tensor: torch.Tensor, src: Optional[int] = None):
        if src is None:
            src = self.mapping.prev_pp_rank()

        work = self.pg.recv([tensor], self._global_to_local_rank(src), tag=0)
        work.block_current_stream()


_pp_comm = None


def init_pp_comm(mapping):
    """Initialize PPComm once at startup"""
    global _pp_comm
    if mpi_disabled():
        _pp_comm = PPCommTorch(mapping)
    elif isinstance(_pp_comm, PPCommNCCL) and \
            _pp_comm.mapping.world_size == mapping.world_size:
        # Reuse the existing world NCCL communicator across LLM instances that
        # share the same worker processes (e.g. a reused MpiPoolSession). The
        # underlying comm depends only on (world_size, rank) -- it is a world
        # communicator, independent of the pp/tp/ep layout -- so only the
        # routing mapping needs refreshing. Recreating it would drop the old
        # comm and trigger a collective ncclCommDestroy at an unsynchronized
        # point during the next model build, which can deadlock on reused
        # workers. Single-LLM (production) runs are unaffected: _pp_comm starts
        # as None, so the first call still constructs a fresh PPCommNCCL.
        _pp_comm.mapping = mapping
    else:
        if _pp_comm is not None:
            # Rebinding drops the old comm; its ncclCommDestroy runs at an
            # unsynchronized point and can deadlock on reused worker processes
            # (see the reuse branch above). Surface it instead of hanging
            # silently -- pools sharing workers must keep one world_size.
            logger.warning(
                "init_pp_comm: replacing existing PP comm (world_size "
                f"{_pp_comm.mapping.world_size} -> {mapping.world_size}) on a "
                "live process; this can deadlock on reused MPI workers.")
        _pp_comm = PPCommNCCL(mapping)
    init_helix_cp_comm(mapping)


@TorchDist.log_op
def pp_recv(tensor):
    """Receive tensors from previous pp rank."""
    _pp_comm.recv(tensor)


@TorchDist.log_op
def pp_send(tensor):
    """Send tensors to next pp rank."""
    _pp_comm.send(tensor)


@torch.library.custom_op("trtllm::pp_recv_tensors", mutates_args=("tensors", ))
def pp_recv_tensors(tensors: List[torch.Tensor]) -> None:
    """
    Receive tensors from previous pp rank.
    """
    for tensor in tensors:
        pp_recv(tensor)


@torch.library.custom_op("trtllm::pp_send_tensors", mutates_args=("tensors", ))
def pp_send_tensors(tensors: List[torch.Tensor]) -> None:
    """Send tensors to next pp rank."""
    for tensor in tensors:
        pp_send(tensor)
