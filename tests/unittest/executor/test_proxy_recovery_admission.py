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
"""Frontend admission control during worker-loss recovery (no GPU)."""

import queue
import time
from unittest.mock import Mock

import pytest

from tensorrt_llm.executor.proxy import GenerationExecutorProxy
from tensorrt_llm.executor.recovery import (
    Abort,
    AdmissionClosed,
    CloseAdmission,
    FrontendAdmission,
    ReopenAdmission,
)
from tensorrt_llm.executor.utils import EngineDeadError, RequestError
from tensorrt_llm.executor.worker_process_monitor import WorkerProcessMonitor

pytestmark = pytest.mark.cpu_only


class _ScriptedClient:
    """Stands in for ``RecoveryClient``: replays queued authority messages."""

    def __init__(self) -> None:
        self.inbox: queue.Queue = queue.Queue()
        self.sent: list = []

    def recv(self, timeout: float):
        try:
            return self.inbox.get(timeout=timeout)
        except queue.Empty:
            return None

    def send(self, message) -> None:
        self.sent.append(message)

    def close(self) -> None:
        pass


def _proxy_with_admission():
    proxy = GenerationExecutorProxy.__new__(GenerationExecutorProxy)
    proxy._engine_dead = False
    proxy._fatal_error = None
    proxy._error_queue = queue.Queue()
    proxy._results = {}
    proxy.doing_shutdown = False
    proxy.workers_started = False
    proxy._multi_frontend_ipc_dir = None
    proxy._owns_mpi_session = False
    proxy._worker_process_monitor = WorkerProcessMonitor()
    proxy.mpi_futures = []
    proxy._submit = Mock(return_value="result")
    client = _ScriptedClient()
    proxy._recovery_admission = FrontendAdmission(
        client,
        is_idle=lambda: not proxy._results,
        on_failure=proxy._handle_worker_death,
        poll_interval=0.01,
    )
    return proxy, client


def _wait_for(predicate, timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while not predicate():
        assert time.monotonic() < deadline, "condition not reached"
        time.sleep(0.01)


def test_close_waits_for_in_flight_then_reopens():
    proxy, client = _proxy_with_admission()
    proxy._results[7] = Mock()
    client.inbox.put(CloseAdmission())
    _wait_for(lambda: proxy._recovery_admission.closed)

    with pytest.raises(RequestError, match="retry"):
        proxy.submit(Mock())
    proxy._submit.assert_not_called()
    assert proxy.check_health()
    time.sleep(0.1)
    assert client.sent == []

    proxy._results.clear()
    _wait_for(lambda: client.sent == [AdmissionClosed()])
    client.inbox.put(ReopenAdmission(1))
    _wait_for(lambda: not proxy._recovery_admission.closed)
    assert proxy.submit(Mock()) == "result"
    assert proxy.check_health()
    proxy._recovery_admission.close()


def test_abort_takes_the_fatal_path():
    proxy, client = _proxy_with_admission()
    client.inbox.put(Abort("rank 0 lost"))
    _wait_for(lambda: proxy._engine_dead)
    with pytest.raises(EngineDeadError):
        proxy.submit(Mock())
    assert not proxy.check_health()


def test_disabled_recovery_submits_directly():
    proxy = GenerationExecutorProxy.__new__(GenerationExecutorProxy)
    proxy._engine_dead = False
    proxy._fatal_error = None
    proxy.workers_started = False
    proxy._multi_frontend_ipc_dir = None
    proxy._submit = Mock(return_value="result")
    assert proxy._recovery_admission is None
    assert proxy.submit(Mock()) == "result"
