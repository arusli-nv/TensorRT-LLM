# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import asyncio
import pickle
import threading
from collections.abc import AsyncIterator
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import zmq

from tensorrt_llm.executor.executor import GenerationExecutor
from tensorrt_llm.executor.request import GenerationRequest
from tensorrt_llm.executor.rpc import RPCClient, RPCServer
from tensorrt_llm.executor.rpc.rpc_common import get_unique_ipc_addr
from tensorrt_llm.executor.rpc_proxy_mixin import RpcExecutorMixin
from tensorrt_llm.executor.rpc_worker_mixin import RpcWorkerMixin
from tensorrt_llm.executor.utils import EngineDeadError, ErrorResponse, RequestError
from tensorrt_llm.sampling_params import SamplingParams

pytestmark = pytest.mark.cpu_only


class _Executor(RpcExecutorMixin, GenerationExecutor):
    def __init__(self) -> None:
        super().__init__()
        self._results = {}
        self._shutdown_event = threading.Event()
        self.rpc_client = Mock()

    def abort_request(self, request_id: int) -> None:
        raise NotImplementedError

    def shutdown(self) -> None:
        self._shutdown_event.set()


@pytest.fixture
def executor() -> _Executor:
    return _Executor()


def _request() -> GenerationRequest:
    return GenerationRequest([1], SamplingParams(max_tokens=1))


@pytest.mark.parametrize("fatal", [False, True])
def test_llm_entry_preserves_fatal_cause(fatal: bool) -> None:
    """Does a new API request preserve a recorded engine failure without changing clean shutdown?"""
    from tensorrt_llm.llmapi.llm import LLM

    error = RuntimeError("model worker died") if fatal else None
    executor = SimpleNamespace(_fatal_error=error, is_shutdown=lambda: True)
    client = SimpleNamespace(_encode_only=False, _executor=executor)
    expected = EngineDeadError if fatal else RuntimeError
    with pytest.raises(
        expected, match="model worker died" if fatal else "LLM is shutting down"
    ) as captured:
        LLM.generate_async(client, [1], SamplingParams(max_tokens=1))
    if fatal:
        assert captured.value.root_cause is error


def test_submit_registers_before_immediate_request_error(executor: _Executor) -> None:
    def respond(request: GenerationRequest) -> Mock:
        executor.handle_responses([ErrorResponse(request.id, "invalid request", request.id)])
        return Mock()

    executor.rpc_client.submit.side_effect = respond
    result = executor.submit(_request())

    with pytest.raises(RequestError, match="invalid request"):
        result.result(timeout=0.1)
    assert executor._fatal_error is None
    assert not executor._results


def test_fatal_error_wakes_pending_results_and_rejects_new_work(executor: _Executor) -> None:
    results = [executor.submit(_request()) for _ in range(2)]
    error = RuntimeError("peer died")
    executor._set_fatal_error(error)
    executor._set_fatal_error(RuntimeError("later error"))

    for result in results:
        for _ in range(2):
            with pytest.raises(EngineDeadError, match="peer died") as captured:
                result.result(timeout=0.1)
            assert captured.value.root_cause is error
    with pytest.raises(EngineDeadError, match="peer died"):
        executor.submit(_request())
    assert executor.rpc_client.submit.call_count == 2


@pytest.mark.asyncio
async def test_fatal_error_wakes_async_waiter_from_another_thread(executor: _Executor) -> None:
    result = executor.submit(_request())
    waiter = asyncio.create_task(result.aresult())
    await asyncio.sleep(0)
    await asyncio.to_thread(executor._set_fatal_error, RuntimeError("peer died"))

    with pytest.raises(EngineDeadError, match="peer died"):
        await asyncio.wait_for(waiter, timeout=1)
    with pytest.raises(EngineDeadError, match="peer died"):
        await result.aresult()


def test_fatal_error_during_registration_rejects_unsent_request(executor: _Executor) -> None:
    class Results(dict):
        def __setitem__(self, key: int, value: object) -> None:
            executor._set_fatal_error(RuntimeError("registration race"))
            super().__setitem__(key, value)

    executor._results = Results()
    with pytest.raises(EngineDeadError, match="registration race"):
        executor.submit(_request())
    executor.rpc_client.submit.assert_not_called()
    assert not executor._results


def test_fatal_error_during_send_wakes_registered_result(executor: _Executor) -> None:
    executor.rpc_client.submit.return_value.remote.side_effect = lambda **kwargs: (
        executor._set_fatal_error(RuntimeError("send race"))
    )
    result = executor.submit(_request())
    with pytest.raises(EngineDeadError, match="send race"):
        result.result(timeout=0.1)


def test_transport_send_failure_unblocks_older_work(executor: _Executor) -> None:
    """Does a transport send failure wake older requests and reject later submissions?"""
    pending = executor.submit(_request())
    error = zmq.ZMQError(zmq.ENOTSOCK)
    executor.rpc_client.submit.return_value.remote.side_effect = error

    with pytest.raises(EngineDeadError) as captured:
        executor.submit(_request())
    assert captured.value.root_cause is error
    assert list(executor._results) == [pending.request_id]
    with pytest.raises(EngineDeadError):
        pending.result(timeout=0.1)
    with pytest.raises(EngineDeadError):
        executor.submit(_request())
    assert executor.rpc_client.submit.call_count == 2


@pytest.mark.parametrize("error_type", [TypeError, RuntimeError, OSError])
def test_local_send_failure_removes_unsent_result(
    executor: _Executor, error_type: type[Exception]
) -> None:
    """Does a local serialization failure remove its result without killing the engine?"""

    class Unpicklable:
        def __reduce__(self):
            raise error_type("cannot serialize")

    pending = executor.submit(_request())
    executor.rpc_client.submit.return_value.remote.side_effect = lambda **kwargs: pickle.dumps(
        Unpicklable()
    )
    with pytest.raises(error_type, match="cannot serialize"):
        executor.submit(_request())
    assert list(executor._results) == [pending.request_id]
    assert executor._fatal_error is None
    assert pending.queue.empty()
    executor.rpc_client.submit.return_value.remote.side_effect = None
    assert executor.submit(_request()).request_id != pending.request_id


@pytest.mark.asyncio
@pytest.mark.parametrize("error", [RuntimeError("transport failed"), None])
async def test_response_stream_failure_is_sticky(
    executor: _Executor, error: Exception | None
) -> None:
    async def responses() -> AsyncIterator[list]:
        if error is not None:
            raise error
        if False:
            yield []

    executor.rpc_client.fetch_responses_loop_async.return_value.remote_streaming = responses
    await executor._fetch_responses_loop_async()
    with pytest.raises(EngineDeadError, match="transport failed|closed unexpectedly"):
        executor.submit(_request())
    executor.rpc_client.submit.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("cancelled", [False, True])
async def test_response_stream_shutdown_is_not_fatal(executor: _Executor, cancelled: bool) -> None:
    async def responses() -> AsyncIterator[list]:
        executor._shutdown_event.set()
        if cancelled:
            raise asyncio.CancelledError
        if False:
            yield []

    executor.rpc_client.fetch_responses_loop_async.return_value.remote_streaming = responses
    await executor._fetch_responses_loop_async()
    assert executor._fatal_error is None


@pytest.mark.asyncio
async def test_unexpected_response_cancellation_fails_pending_work(executor: _Executor) -> None:
    pending = executor.submit(_request())

    async def responses() -> AsyncIterator[list]:
        raise asyncio.CancelledError
        yield []

    executor.rpc_client.fetch_responses_loop_async.return_value.remote_streaming = responses
    await executor._fetch_responses_loop_async()
    with pytest.raises(EngineDeadError, match="cancelled unexpectedly"):
        pending.result(timeout=0.1)
    with pytest.raises(EngineDeadError, match="cancelled unexpectedly"):
        executor.submit(_request())
    assert executor.rpc_client.submit.call_count == 1


@pytest.mark.asyncio
async def test_worker_loop_failure_crosses_real_rpc_stream(executor: _Executor) -> None:
    class WorkerBase:
        def await_responses(self, timeout: float) -> list:
            return []

    class FailedWorker(RpcWorkerMixin, WorkerBase):
        def __init__(self) -> None:
            self.rank = 0
            self.engine = SimpleNamespace(_event_loop_error=RuntimeError("peer died"))
            self.shutdown_event = threading.Event()

    server = RPCServer(FailedWorker())
    address = get_unique_ipc_addr()
    server.bind(address)
    server.start()
    try:
        with RPCClient(address, hmac_key=server.hmac_key) as client:
            executor.rpc_client = client
            await asyncio.wait_for(executor._fetch_responses_loop_async(), timeout=5)
            with pytest.raises(EngineDeadError, match="peer died"):
                executor.submit(_request())
    finally:
        server.shutdown()


def test_ray_submit_preserves_assigned_client_id(executor: _Executor) -> None:
    pytest.importorskip("ray")
    from tensorrt_llm.executor.ray.executor import RayExecutor

    request = _request()
    request.set_id(42)
    result = RayExecutor.submit(executor, request)
    assert result.request_id == 42
    assert executor._results[42] is result
