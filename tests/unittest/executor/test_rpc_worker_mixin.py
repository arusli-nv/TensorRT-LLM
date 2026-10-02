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

from contextlib import nullcontext
from importlib import import_module
from queue import Queue
from threading import Event, Lock
from types import SimpleNamespace

import pytest

from tensorrt_llm.executor.base_worker import AwaitResponseHelper
from tensorrt_llm.executor.rpc_worker_mixin import RpcWorkerMixin
from tensorrt_llm.executor.utils import ErrorResponse

pytestmark = pytest.mark.cpu_only


class _WorkerBaseStub:
    def await_responses(self, timeout):
        self.await_responses_timeout = timeout
        return getattr(self, "await_responses_result", ["forward", "consume", None])


class _RpcWorkerStub(RpcWorkerMixin, _WorkerBaseStub):
    def __init__(self):
        self.rank = 0
        self._fetch_timeout = 0.1
        self._response_queue = Queue()
        self.enable_postprocess_parallel = False
        self._await_response_helper = AwaitResponseHelper(self)
        self._await_response_helper.responses_handler = self._responses_handler
        self.handler_responses = None
        self.callback_responses = []

    def _responses_handler(self, responses):
        self.handler_responses = responses
        if responses:
            self._response_queue.put(responses)

    def _engine_response_callback(self, response):
        self.callback_responses.append(response)
        if response in ("consume", None):
            return None
        return f"processed-{response}"


def test_fetch_responses_processes_and_filters_engine_responses():
    worker = _RpcWorkerStub()
    worker._await_response_helper.temp_error_responses.put("temporary-error")

    responses = worker.fetch_responses(timeout=0.25)

    assert worker.await_responses_timeout == 0.25
    assert worker.callback_responses == ["forward", "consume", None]
    assert worker.handler_responses == ["processed-forward", "temporary-error"]
    assert responses == ["processed-forward", "temporary-error"]


class _CrashedRpcWorkerStub(RpcWorkerMixin, _WorkerBaseStub):
    def __init__(self):
        self.rank = 0
        self._fetch_timeout = 0.1
        self._response_queue = Queue()
        self._response_error_lock = Lock()
        self.result_queue = self._response_queue
        self.postproc_queues = None
        self.frontend_result_queues = None
        self.enable_postprocess_parallel = False
        self.engine = SimpleNamespace(
            _event_loop_error=RuntimeError("peer rank exited"),
            _event_loop_error_delivered=Event(),
        )
        self._results = {1: object(), 2: object()}
        self.await_responses_result = []
        self._await_response_helper = AwaitResponseHelper(self)
        self.popped = []

    def _pop_result(self, client_id):
        self.popped.append(client_id)
        self._results.pop(client_id, None)


def test_fetch_responses_reports_event_loop_failure_to_pending_rpc_clients():
    worker = _CrashedRpcWorkerStub()

    responses = worker.fetch_responses(timeout=0.01)

    assert [response.client_id for response in responses] == [1, 2]
    assert all(isinstance(response, ErrorResponse) for response in responses)
    assert all("peer rank exited" in response.error_msg for response in responses)
    assert worker.popped == [1, 2]
    assert not worker.engine._event_loop_error_delivered.is_set()
    assert worker.fetch_responses(timeout=0.01) == []
    worker._results[3] = object()
    assert [response.client_id for response in worker.fetch_responses(timeout=0.01)] == [3]


def test_fetch_responses_does_not_replace_completed_rpc_response():
    worker = _CrashedRpcWorkerStub()
    worker._await_response_helper.temp_error_responses.put(
        ErrorResponse(1, "request failed before engine crash", 1)
    )

    responses = worker.fetch_responses(timeout=0.01)

    assert [response.client_id for response in responses] == [1, 2]
    assert responses[0].error_msg == "request failed before engine crash"
    assert "peer rank exited" in responses[1].error_msg
    assert worker.popped == [1, 2]


@pytest.mark.parametrize(
    "module_name,executor_name",
    [
        ("tensorrt_llm.executor.rpc_proxy_mixin", "RpcExecutorMixin"),
        ("tensorrt_llm.executor.ray.executor", "RayExecutor"),
    ],
)
@pytest.mark.parametrize("send_fails", [False, True])
def test_rpc_submit_registers_result_before_remote_call(
    monkeypatch, module_name, executor_name, send_fails
):
    if executor_name == "RayExecutor":
        pytest.importorskip("ray")
    module = import_module(module_name)
    monkeypatch.setattr(module, "GenerationResult", lambda request, **kwargs: object())
    monkeypatch.setattr(module, "nvtx_range_debug", lambda *args, **kwargs: nullcontext())

    request = SimpleNamespace(id=None, disaggregated_params=None)
    request.set_id = lambda client_id: setattr(request, "id", client_id)
    proxy = SimpleNamespace(
        _get_next_client_id=lambda: 7,
        _get_logprob_params=lambda request: None,
        _handle_background_error=lambda: None,
        _results={},
    )

    def remote(*, need_response):
        assert not need_response
        assert 7 in proxy._results
        if send_fails:
            raise RuntimeError("RPC send failed")

    proxy.rpc_client = SimpleNamespace(submit=lambda request: SimpleNamespace(remote=remote))

    if send_fails:
        with pytest.raises(RuntimeError, match="RPC send failed"):
            getattr(module, executor_name).submit(proxy, request)
        assert proxy._results == {}
    else:
        result = getattr(module, executor_name).submit(proxy, request)
        assert proxy._results == {7: result}
