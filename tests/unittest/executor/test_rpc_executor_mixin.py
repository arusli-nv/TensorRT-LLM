# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import asyncio
import pickle
import threading
from collections.abc import AsyncIterator
from types import MethodType, SimpleNamespace
from unittest.mock import AsyncMock, Mock

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


def test_submit_registers_before_immediate_request_error(executor: _Executor) -> None:
    """Can an immediate RPC response reach its result before submission returns?"""

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
    """Does the first fatal cause persist for pending results and later submissions?"""
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
    """Does a fatal error from another thread wake an async result waiter?"""
    result = executor.submit(_request())
    waiter = asyncio.create_task(result.aresult())
    await asyncio.sleep(0)
    await asyncio.to_thread(executor._set_fatal_error, RuntimeError("peer died"))

    with pytest.raises(EngineDeadError, match="peer died"):
        await asyncio.wait_for(waiter, timeout=1)
    with pytest.raises(EngineDeadError, match="peer died"):
        await result.aresult()


def test_fatal_error_during_registration_rejects_unsent_request(executor: _Executor) -> None:
    """Does a registration-time fatal error prevent sending and remove the unsent result?"""

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
    """Does a send-time fatal error reach the result already registered for the request?"""
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
    """Does a response-stream error or unexpected closure prevent subsequent submissions?"""

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
    """Can intentional response-stream closure or cancellation create a fatal engine error?"""

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
    """Does unexpected response-task cancellation fail pending and subsequent requests?"""
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
    """Does a stored worker-loop error cross the RPC stream and reach the client?"""

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
    """Does Ray submission preserve the request ID already assigned by the caller?"""
    pytest.importorskip("ray")
    from tensorrt_llm.executor.ray.executor import RayExecutor

    request = _request()
    request.set_id(42)
    result = RayExecutor.submit(executor, request)
    assert result.request_id == 42
    assert executor._results[42] is result


@pytest.fixture
def ray_notifications(executor: _Executor, monkeypatch: pytest.MonkeyPatch):
    ray = pytest.importorskip("ray")
    from ray._private import gcs_pubsub, state

    from tensorrt_llm.executor.ray.executor import RayExecutor

    subscriber = SimpleNamespace(subscribe=AsyncMock(), poll=AsyncMock(), close=AsyncMock())
    snapshot = Mock(return_value={})
    monkeypatch.setattr(gcs_pubsub, "GcsAioActorSubscriber", lambda **kwargs: subscriber)
    monkeypatch.setattr(state, "actors", snapshot)
    monkeypatch.setattr(
        ray,
        "get_runtime_context",
        lambda: SimpleNamespace(gcs_address="test", get_job_id=lambda: "01000000"),
    )
    executor.workers = [
        SimpleNamespace(_ray_actor_id=SimpleNamespace(binary=lambda: bytes(16)), shutdown=Mock())
    ]
    executor._mainloop_started = False
    executor._actor_event_subscriber = None
    executor.has_start_local_cluser = False
    executor.collective_rpc_async = AsyncMock()
    if hasattr(RayExecutor, "_monitor_worker_deaths_async"):
        executor._monitor_worker_deaths_async = MethodType(
            RayExecutor._monitor_worker_deaths_async, executor
        )
    executor._wait_for_cluster_resource_release = MethodType(
        RayExecutor._wait_for_cluster_resource_release, executor
    )

    async def silent_root() -> AsyncIterator[list]:
        await asyncio.Event().wait()
        yield []

    executor.rpc_client.fetch_responses_loop_async.return_value.remote_streaming = silent_root
    return RayExecutor, subscriber, snapshot


@pytest.mark.asyncio
@pytest.mark.parametrize("already_dead", [False, True])
async def test_ray_death_wakes_requests_without_root_rpc_error(
    executor: _Executor, ray_notifications, already_dead: bool
) -> None:
    """Does confirmed actor death fail requests already waiting on a silent root RPC stream?"""
    ray = pytest.importorskip("ray")
    from ray.core.generated.gcs_pb2 import ActorTableData

    RayExecutor, subscriber, actor_snapshot = ray_notifications

    actor_id = bytes(range(16))
    root_id, unrelated_id = bytes(16), bytes([255] * 16)
    executor.workers = [
        SimpleNamespace(_ray_actor_id=SimpleNamespace(binary=lambda: root_id)),
        SimpleNamespace(_ray_actor_id=SimpleNamespace(binary=lambda: actor_id)),
    ]
    subscribed = threading.Event()

    async def subscribe() -> None:
        subscribed.set()

    def snapshot(**kwargs) -> dict:
        assert subscribed.is_set(), "Snapshot before subscription can miss actor death"
        assert kwargs["job_id"] == ray.JobID.from_hex("01000000")
        return {
            unrelated_id.hex(): {"State": "DEAD"},
            root_id.hex(): {"State": "ALIVE"},
            actor_id.hex(): {"State": "DEAD" if already_dead else "ALIVE"},
        }

    subscriber.subscribe.side_effect = subscribe
    subscriber.poll.return_value = [
        (unrelated_id, ActorTableData(state=ActorTableData.DEAD)),
        (actor_id, ActorTableData(state=ActorTableData.DEAD)),
    ]
    actor_snapshot.side_effect = snapshot
    results = [executor.submit(_request()) for _ in range(2)]
    waiters = [asyncio.create_task(result.aresult()) for result in results]
    await asyncio.sleep(0)
    try:
        await RayExecutor.setup_engine_remote_async(executor)
        for waiter in waiters:
            with pytest.raises(EngineDeadError, match="Ray model worker rank 1"):
                await asyncio.wait_for(waiter, timeout=2)
        with pytest.raises(EngineDeadError, match="Ray model worker rank 1"):
            executor.submit(_request())
        assert subscriber.poll.await_count == (0 if already_dead else 1)
    finally:
        executor._shutdown_event.set()
        for waiter in waiters:
            waiter.cancel()
        await asyncio.gather(*waiters, return_exceptions=True)
        if getattr(executor, "main_loop", None) is not None:
            executor.main_loop.call_soon_threadsafe(executor.main_loop_task_obj.cancel)
            await asyncio.to_thread(executor.main_loop_thread.join, 5)
            assert not executor.main_loop_thread.is_alive()
        if hasattr(RayExecutor, "_monitor_worker_deaths_async"):
            subscriber.close.assert_awaited_once()


@pytest.mark.asyncio
async def test_ray_shutdown_during_actor_subscription(
    executor: _Executor, monkeypatch: pytest.MonkeyPatch, ray_notifications
) -> None:
    """Does shutdown retire an actor subscriber while its startup is still pending?"""
    ray = pytest.importorskip("ray")
    RayExecutor, subscriber, snapshot = ray_notifications
    worker = executor.workers[0]
    subscribing = threading.Event()
    subscription_cancelled = threading.Event()
    closed_during_subscription = threading.Event()
    shutdown_completed = threading.Event()

    async def subscribe() -> None:
        subscribing.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            subscription_cancelled.set()
            raise

    async def close() -> None:
        if subscribing.is_set() and not subscription_cancelled.is_set():
            closed_during_subscription.set()

    subscriber.subscribe.side_effect = subscribe
    subscriber.close.side_effect = close
    get_workers, kill_worker = Mock(), Mock()
    monkeypatch.setattr(ray, "get", get_workers)
    monkeypatch.setattr(ray, "kill", kill_worker)

    def shutdown() -> None:
        RayExecutor.shutdown(executor)
        shutdown_completed.set()

    shutdown_thread = threading.Thread(target=shutdown, daemon=True)
    try:
        await RayExecutor.setup_engine_remote_async(executor)
        assert await asyncio.to_thread(subscribing.wait, 2)
        assert executor._actor_event_subscriber is subscriber
        shutdown_thread.start()
        assert await asyncio.to_thread(shutdown_completed.wait, 5), "Ray shutdown hung"
        assert not executor.main_loop_thread.is_alive()
        assert executor.main_loop.is_closed()
        assert subscription_cancelled.is_set()
        assert closed_during_subscription.is_set()
        assert subscriber.close.await_count >= 1
        subscriber.poll.assert_not_awaited()
        snapshot.assert_not_called()
        assert executor._fatal_error is None
        worker.shutdown.remote.assert_called_once_with()
        get_workers.assert_called_once_with([worker.shutdown.remote.return_value], timeout=30.0)
        kill_worker.assert_called_once_with(worker, no_restart=True)
        executor.rpc_client.close.assert_called_once_with()
    finally:
        executor._shutdown_event.set()
        if getattr(executor, "main_loop", None) is not None:
            if executor.main_loop.is_running():
                executor.main_loop.call_soon_threadsafe(executor.main_loop_task_obj.cancel)
            await asyncio.to_thread(executor.main_loop_thread.join, 5)
        if shutdown_thread.ident is not None:
            await asyncio.to_thread(shutdown_thread.join, 5)


@pytest.mark.asyncio
@pytest.mark.parametrize("phase", ["subscribe", "snapshot", "poll"])
async def test_ray_notification_error_fails_pending_requests(
    executor: _Executor, ray_notifications, phase: str
) -> None:
    """Does a reported control-channel error fail waiting requests without inventing rank death?"""
    RayExecutor, subscriber, snapshot = ray_notifications
    unavailable = RuntimeError("actor notification service unavailable")
    if phase == "snapshot":
        snapshot.side_effect = unavailable
    else:
        getattr(subscriber, phase).side_effect = unavailable
    pending = [executor.submit(_request()) for _ in range(2)]
    await RayExecutor._monitor_worker_deaths_async(executor)
    for result in pending:
        with pytest.raises(EngineDeadError, match="actor notification service unavailable"):
            await asyncio.wait_for(result.aresult(), 1)
    with pytest.raises(EngineDeadError, match="actor notification service unavailable"):
        executor.submit(_request())
    assert executor._fatal_error is unavailable
    subscriber.close.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("already_dead", [False, True])
async def test_ray_actor_notification_with_live_control_service(
    executor, monkeypatch, already_dead
):
    """Does the installed Ray API report a real owned actor death before or after subscription?"""
    ray = pytest.importorskip("ray")
    from ray._private.gcs_pubsub import GcsAioActorSubscriber

    from tensorrt_llm.executor.ray.executor import RayExecutor

    @ray.remote(num_cpus=0)
    class CpuActor:
        def ready(self):
            return True

    assert not ray.is_initialized(), "This test requires its own local Ray control service"
    monitor = None
    try:
        await asyncio.to_thread(
            ray.init, address="local", num_cpus=2, num_gpus=0, include_dashboard=False
        )
        root, victim = CpuActor.remote(), CpuActor.remote()
        assert await asyncio.to_thread(
            ray.get, [root.ready.remote(), victim.ready.remote()], timeout=20
        ) == [True, True]
        executor.workers = [root, victim]
        pending = [executor.submit(_request()) for _ in range(2)]
        subscribed = asyncio.Event()
        original = GcsAioActorSubscriber.subscribe

        async def subscribe(self):
            await original(self)
            subscribed.set()

        monkeypatch.setattr(GcsAioActorSubscriber, "subscribe", subscribe)
        if already_dead:
            await asyncio.to_thread(ray.kill, victim)
            with pytest.raises(ray.exceptions.ActorDiedError):
                await asyncio.to_thread(ray.get, victim.ready.remote(), timeout=20)
        monitor = asyncio.create_task(RayExecutor._monitor_worker_deaths_async(executor))
        await asyncio.wait_for(subscribed.wait(), 10)
        if not already_dead:
            await asyncio.to_thread(ray.kill, victim)
        await asyncio.wait_for(monitor, 15)
        for result in pending:
            with pytest.raises(EngineDeadError, match="Ray model worker rank 1"):
                result.result(timeout=1)
        assert await asyncio.to_thread(ray.get, root.ready.remote(), timeout=10)
    finally:
        executor._shutdown_event.set()
        if monitor is not None and not monitor.done():
            monitor.cancel()
            await asyncio.gather(monitor, return_exceptions=True)
        await asyncio.to_thread(ray.shutdown)
