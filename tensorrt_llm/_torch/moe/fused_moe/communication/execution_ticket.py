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
"""Own one execution's timeout reservation and real abort-publication DMA."""

from __future__ import annotations

import threading
import time
from collections.abc import Iterable

import torch

from tensorrt_llm._torch.alltoall_watchdog import AlltoAllWatchdogTimeout


def wait_cuda_event(event: torch.cuda.Event, timeout_s: float) -> None:
    deadline = time.monotonic() + timeout_s
    while not event.query():
        if time.monotonic() >= deadline:
            raise TimeoutError("CUDA completion is unknown; retain execution resources")
        time.sleep(0.001)


class ExecutionTicket:
    def __init__(
        self, *, abort: torch.Tensor, watches: Iterable[tuple[str, int]], timeout_s: float
    ) -> None:
        self.abort = abort
        self.watches = frozenset(watches)
        if not self.watches:
            raise ValueError("Execution ticket needs actual native phase expectations")
        self.timeout_s = timeout_s
        self._lock = threading.Lock()
        self._closed = False
        self._reserved = False
        self._error = None
        self._callback_done = threading.Event()
        self._callback_done.set()
        self._request_host = torch.ones(1, dtype=torch.int32, device="cpu", pin_memory=True)
        self._control_stream = torch.cuda.Stream(device=abort.device)
        self._published = torch.cuda.Event(blocking=False)
        self.detection = None

    def on_timeout(self, event: AlltoAllWatchdogTimeout) -> bool:
        if (event.phase, event.expected_flag) not in self.watches:
            return False
        return self.request_abort(event)

    def request_abort(self, detection: AlltoAllWatchdogTimeout | None = None) -> bool:
        with self._lock:
            if self._closed or self._reserved:
                return False
            self._reserved = True
            self.detection = detection
            self._callback_done.clear()
        try:
            with torch.cuda.device(self.abort.device), torch.cuda.stream(self._control_stream):
                # Preserve native sticky status; only the requested word is published.
                self.abort[:1].copy_(self._request_host, non_blocking=True)
                self._published.record(self._control_stream)
        except RuntimeError as error:
            with self._lock:
                self._error = error
        finally:
            self._callback_done.set()
        return True

    def close_and_join(self, *, deadline_monotonic: float | None = None) -> bool:
        with self._lock:
            self._closed = True
        remaining = (
            self.timeout_s if deadline_monotonic is None else deadline_monotonic - time.monotonic()
        )
        if remaining <= 0 or not self._callback_done.wait(remaining):
            raise TimeoutError("Abort callback is pending; retain its DMA bank")
        with self._lock:
            error, reserved = self._error, self._reserved
        if error is not None:
            raise RuntimeError("Abort publication completion is unknown") from error
        if reserved:
            remaining = (
                self.timeout_s
                if deadline_monotonic is None
                else deadline_monotonic - time.monotonic()
            )
            if remaining <= 0:
                raise TimeoutError("STOP deadline expired; retain abort-publication DMA")
            wait_cuda_event(self._published, remaining)
        return reserved
