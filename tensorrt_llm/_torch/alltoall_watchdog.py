# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Host-side watchdog for MoE AlltoAll waits on peers.

In rank-mask mode, the NVLinkOneSided fence-path kernels publish the round flag
value that a wait on a peer waits for in the pinned host abort state once the wait
outlasts about 10 ms. A dead peer in the silent-spin failure mode never completes its round,
so this watchdog polls that abort state from a CPU thread and reports peers that a
wait stays on for longer than a bounded timeout. The polls read host memory
only, so they add no GPU work and also cover replayed CUDA graphs.

Timeouts are detection events, not committed membership changes. Higher-layer
recovery coordination consumes ``on_timeout`` events and commits a new
membership only after the data plane and expert placement are ready for the
same generation.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass

import torch

from tensorrt_llm.logger import logger as tllm_logger

DEFAULT_ALLTOALL_WATCHDOG_TIMEOUT_S = 5.0
DEFAULT_ALLTOALL_WATCHDOG_POLL_INTERVAL_S = 0.1


def reject_rank_mask_cuda_graph_capture(rank_mask_enabled: bool) -> None:
    """Reject graph capture until membership-scoped recapture is available."""
    if rank_mask_enabled and torch.cuda.is_current_stream_capturing():
        raise RuntimeError(
            "rank-mask mode does not support CUDA graphs until generation-scoped "
            "invalidation and recapture are implemented"
        )


@dataclass(frozen=True)
class AlltoAllWatchdogTimeout:
    """Detection details emitted when AlltoAll kernels wait on peers for too long.

    ``missing_ranks`` are the peers whose round flag ``expected_flag`` the
    kernels of this rank wait for. Receiving this event does not publish a new
    active mask.
    """

    phase: str
    expected_flag: int
    missing_ranks: tuple[int, ...]
    elapsed_s: float


class AlltoAllWatchdog:
    """Background host thread that reports peers the AlltoAll kernels wait on for too long.

    ``waited_flags`` returns each peer that a kernel of this rank waits on, with
    the round flag value it waits for. The thread polls it and reports each wait
    that lasts ``timeout_s`` once through ``on_timeout``, without mutating the
    committed EP membership.
    """

    def __init__(
        self,
        *,
        ep_rank: int,
        waited_flags: Callable[[], Mapping[int, int]],
        timeout_s: float = DEFAULT_ALLTOALL_WATCHDOG_TIMEOUT_S,
        poll_interval_s: float = DEFAULT_ALLTOALL_WATCHDOG_POLL_INTERVAL_S,
        on_timeout: Callable[[AlltoAllWatchdogTimeout], None] | None = None,
    ) -> None:
        if timeout_s <= 0:
            raise ValueError(f"timeout_s must be > 0, got {timeout_s}")
        if poll_interval_s <= 0:
            raise ValueError(f"poll_interval_s must be > 0, got {poll_interval_s}")

        self._ep_rank = int(ep_rank)
        self._waited_flags = waited_flags
        self._timeout_s = float(timeout_s)
        self._poll_interval_s = float(poll_interval_s)
        self._on_timeout = on_timeout

        self._cv = threading.Condition()
        self._closed = False
        self._stopping = False
        self._thread: threading.Thread | None = None
        self._last_error: BaseException | None = None

    @property
    def last_error(self) -> BaseException | None:
        """Return the last polling-thread error, if any."""
        with self._cv:
            return self._last_error

    def start(self) -> None:
        """Start the background polling thread. Idempotent."""
        with self._cv:
            if self._closed:
                raise RuntimeError("cannot start a stopped AlltoAllWatchdog")
            if self._thread is not None and self._thread.is_alive():
                return
            self._stopping = False
            self._thread = threading.Thread(
                target=self._run,
                name=f"AlltoAllWatchdog-rank{self._ep_rank}",
                daemon=True,
            )
            self._thread.start()

    def stop(self, timeout_s: float | None = None) -> None:
        """Stop the polling thread and wait for it to exit."""
        with self._cv:
            self._closed = True
            self._stopping = True
            self._cv.notify_all()
            thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=timeout_s)

    def __enter__(self) -> "AlltoAllWatchdog":
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.stop(timeout_s=1.0)

    def _handle_timeout(
        self, expected_flag: int, missing_ranks: tuple[int, ...], elapsed_s: float
    ) -> None:
        # Round N waits for flag value 2N + 1 in dispatch and 2N + 2 in combine, so odd means dispatch.
        phase = "dispatch" if expected_flag & 1 else "combine"
        tllm_logger.warning(
            "AlltoAll watchdog timeout on rank %d during %s: waiting %.3fs for flag %d from ranks %s",
            self._ep_rank,
            phase,
            elapsed_s,
            expected_flag,
            list(missing_ranks),
        )
        if self._on_timeout is not None:
            self._on_timeout(
                AlltoAllWatchdogTimeout(
                    phase=phase,
                    expected_flag=expected_flag,
                    missing_ranks=missing_ranks,
                    elapsed_s=elapsed_s,
                )
            )

    def _stop_after_error(self, exc: Exception) -> None:
        with self._cv:
            self._last_error = exc
            self._closed = True
            self._stopping = True
            self._cv.notify_all()

    def _poll(
        self, waits_since: dict[tuple[int, int], float], reported: set[tuple[int, int]]
    ) -> None:
        """Report the waits that reached the timeout since the previous poll."""
        now = time.monotonic()
        current = {wait: waits_since.get(wait, now) for wait in self._waited_flags().items()}
        waits_since.clear()
        waits_since.update(current)
        reported.intersection_update(current)
        timed_out: dict[int, list[tuple[int, float]]] = {}
        for (peer, flag), since_s in current.items():
            if (peer, flag) not in reported and now - since_s >= self._timeout_s:
                timed_out.setdefault(flag, []).append((peer, since_s))
        for flag, waits in sorted(timed_out.items()):
            self._handle_timeout(
                flag,
                tuple(sorted(peer for peer, _ in waits)),
                now - min(since_s for _, since_s in waits),
            )
            reported.update((peer, flag) for peer, _ in waits)

    def _run(self) -> None:
        # Each wait is a (peer, round flag) pair, timed from the poll that first sees it.
        waits_since: dict[tuple[int, int], float] = {}
        reported: set[tuple[int, int]] = set()
        while True:
            with self._cv:
                if self._stopping:
                    return
            try:
                self._poll(waits_since, reported)
            except Exception as exc:  # noqa: BLE001 - keep watchdog failures visible.
                self._stop_after_error(exc)
                tllm_logger.error("AlltoAll watchdog stopped after polling error: %s", exc)
                return
            with self._cv:
                self._cv.wait_for(lambda: self._stopping, timeout=self._poll_interval_s)
