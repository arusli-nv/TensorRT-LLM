# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Report evidence of a hung peer to the recovery authority.

Each worker's reporter thread sends the authority a heartbeat (its executor
iteration, and whether the executor waits on something outside its rank) and
the all-to-all waits of its device that last longer than a threshold. The
device waits come from the rank's all-to-all progress row: its current round
(``flag_val``) and the completion flags its peers write into its workspace.
The row is copied to host memory on a side stream, so it is read whether the
step was launched eagerly or replayed from a CUDA graph. Evidence never
changes membership; the authority decides. The same thread carries out the
authority's ``Fence``: of its own rank (its main thread is stuck, this thread
still runs) by killing its process; of a hung node-mate (whose whole process
may be stopped) by killing that process after a grace period, if it still runs
as that rank.
"""

from __future__ import annotations

import os
import signal
import socket
import threading
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path

import torch

from tensorrt_llm._utils import prefer_pinned
from tensorrt_llm.executor.recovery.protocol import DeviceWait, Fence, Heartbeat
from tensorrt_llm.logger import logger

__all__ = [
    "HangEvidenceReporter",
    "ProgressRow",
    "ProgressRowReader",
    "lagging_peers",
]

_FLAG_MASK = 0xFFFFFFFF
# A node-mate fences a hung rank only if that rank's own evidence thread has not done so by then.
_NODE_MATE_FENCE_GRACE_S = 1.0


def _runs_as_rank(pid: int, rank: int) -> bool:
    """Whether process ``pid`` on this host is the MPI worker of ``rank`` (not a reused pid)."""
    try:
        environ = Path(f"/proc/{pid}/environ").read_bytes().split(b"\0")
    except OSError:
        return False
    return f"OMPI_COMM_WORLD_RANK={rank}".encode() in environ


@dataclass(frozen=True)
class ProgressRow:
    """A rank's all-to-all progress: its current round and each peer's last completed rounds."""

    flag_val: int
    dispatch_flags: tuple[int, ...]
    combine_flags: tuple[int, ...]


def lagging_peers(row: ProgressRow, active: Iterable[int], rank: int) -> set[int]:
    """Active peers that have completed neither phase of this rank's current round.

    A round is one dispatch or one combine; a peer that finished it wrote the
    round's value into the matching flag. Between steps every peer has finished
    the last round, so a rank that is not waiting reports no laggard.
    """
    current = row.flag_val & _FLAG_MASK
    return {
        peer
        for peer in active
        if peer != rank
        and row.dispatch_flags[peer] & _FLAG_MASK != current
        and row.combine_flags[peer] & _FLAG_MASK != current
    }


class ProgressRowReader:
    """Copy one rank's progress row from its all-to-all workspace to pinned host memory.

    The copy runs on a side stream behind no other work, so it completes while
    the compute stream spins in an all-to-all wait.

    Args:
        workspace: The ``[ep_size, bytes_per_rank]`` all-to-all workspace.
        rank: This rank's row.
        offsets: Byte offsets of ``flag_val`` and of the dispatch and combine
            completion flags in a row.
    """

    def __init__(
        self,
        workspace: torch.Tensor,
        rank: int,
        flag_val_offset: int,
        dispatch_flags_offset: int,
        combine_flags_offset: int,
    ) -> None:
        ep_size = workspace.size(0)
        row = workspace[rank]
        self._sources = (
            row[flag_val_offset : flag_val_offset + 4].view(torch.int32),
            row[dispatch_flags_offset : dispatch_flags_offset + 4 * ep_size].view(torch.int32),
            row[combine_flags_offset : combine_flags_offset + 4 * ep_size].view(torch.int32),
        )
        self._host = torch.empty(1 + 2 * ep_size, dtype=torch.int32, pin_memory=prefer_pinned())
        self._stream = torch.cuda.Stream(device=workspace.device)
        self._event = torch.cuda.Event()
        self._ep_size = ep_size
        self._pending = False

    def read(self) -> ProgressRow | None:
        """Return the row once the previous copy has landed, starting the next copy; None until then."""
        if self._pending and not self._event.query():
            return None
        row = None
        if self._pending:
            values = [int(v) & _FLAG_MASK for v in self._host.tolist()]
            n = self._ep_size
            row = ProgressRow(values[0], tuple(values[1 : 1 + n]), tuple(values[1 + n :]))
        with torch.cuda.stream(self._stream):
            start = 0
            for source in self._sources:
                self._host[start : start + source.numel()].copy_(source, non_blocking=True)
                start += source.numel()
            self._event.record(self._stream)
        self._pending = True
        return row


class HangEvidenceReporter:
    """Background thread sending heartbeats and long device waits to the authority.

    Args:
        rank: This rank (attention-DP and EP rank).
        send: Sends a message to the authority; called from the reporter thread only.
        read_row: Returns this rank's latest progress row, or None if none is ready.
        active_ranks: Returns the ranks of the committed membership.
        iteration: Returns the executor's iteration counter.
        waiting: Returns whether the executor waits on something outside its rank.
        suspect_s: How long a device wait on a peer lasts before it is reported.
        heartbeat_s: Interval between heartbeats; also the polling interval.
        receive: Returns the authority's next message for this connection, or None; called from the
            reporter thread only.
    """

    def __init__(
        self,
        rank: int,
        send: Callable[[object], None],
        read_row: Callable[[], ProgressRow | None],
        active_ranks: Callable[[], Iterable[int]],
        iteration: Callable[[], int],
        waiting: Callable[[], bool],
        suspect_s: float = 2.0,
        heartbeat_s: float = 1.0,
        receive: Callable[[], object | None] = lambda: None,
    ) -> None:
        self._rank = rank
        self._send = send
        self._receive = receive
        # Node-mate fences due: (monotonic deadline, Fence).
        self._fences: list[tuple[float, Fence]] = []
        self._read_row = read_row
        self._active_ranks = active_ranks
        self._iteration = iteration
        self._waiting = waiting
        self._suspect_s = suspect_s
        self._heartbeat_s = heartbeat_s
        self._host = socket.gethostname()
        self._pid = os.getpid()
        # Peer -> monotonic time since which it has lagged this rank's round.
        self._lagging_since: dict[int, float] = {}
        self._reported: set[int] = set()
        self._last_row: ProgressRow | None = None
        self._row_time = 0.0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name="hang_evidence", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join()

    def _fence(self, now: float) -> None:
        while (message := self._receive()) is not None:
            if not isinstance(message, Fence):
                continue
            if message.rank == self._rank:
                logger.error(
                    f"Rank {self._rank}: fenced as hung by the recovery authority; ending the process"
                )
                os.kill(os.getpid(), signal.SIGKILL)
            self._fences.append((now + _NODE_MATE_FENCE_GRACE_S, message))
        due = [fence for deadline, fence in self._fences if deadline <= now]
        self._fences = [(deadline, fence) for deadline, fence in self._fences if deadline > now]
        for fence in due:
            if _runs_as_rank(fence.pid, fence.rank):
                logger.error(
                    f"Rank {self._rank}: fencing hung node-mate rank {fence.rank} (pid {fence.pid})"
                )
                os.kill(fence.pid, signal.SIGKILL)

    def _run(self) -> None:
        while not self._stop.wait(self._heartbeat_s):
            try:
                self.poll(time.monotonic())
            except Exception as error:  # noqa: BLE001 - evidence is best effort; never kill the worker
                logger.warning(f"Rank {self._rank}: hang evidence reporting failed: {error}")

    def poll(self, now: float) -> None:
        """Carry out fences, then send a heartbeat and any change in the device waits worth reporting."""
        self._fence(now)
        self._send(
            Heartbeat(
                rank=self._rank,
                iteration=int(self._iteration()),
                waiting=bool(self._waiting()),
                host=self._host,
                pid=self._pid,
            )
        )
        row = self._read_row()
        if row is not None:
            self._last_row, self._row_time = row, now
        # A copy that does not land (its own device is stuck) leaves no current evidence about peers, and
        # waits reported earlier must not keep this rank looking like a victim.
        if self._last_row is None or now - self._row_time > self._heartbeat_s:
            lagging = set()
        else:
            lagging = lagging_peers(self._last_row, self._active_ranks(), self._rank)
        for peer in list(self._lagging_since):
            if peer not in lagging:
                del self._lagging_since[peer]
                if peer in self._reported:
                    self._reported.discard(peer)
                    self._send(DeviceWait(rank=self._rank, peer=peer, waited_ms=0))
        for peer in lagging:
            since = self._lagging_since.setdefault(peer, now)
            if now - since >= self._suspect_s and peer not in self._reported:
                self._reported.add(peer)
                self._send(
                    DeviceWait(rank=self._rank, peer=peer, waited_ms=int((now - since) * 1000))
                )
