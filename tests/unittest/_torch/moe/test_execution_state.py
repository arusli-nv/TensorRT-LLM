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

import threading
import time
from collections.abc import Iterator
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from tensorrt_llm._torch.moe.fused_moe.communication import execution_state
from tensorrt_llm._torch.moe.fused_moe.communication.execution_state import ExecutionState

pytestmark = pytest.mark.cpu_only


@pytest.fixture
def owner(monkeypatch: pytest.MonkeyPatch) -> Iterator[ExecutionState]:
    state = object.__new__(ExecutionState)
    state._lock = threading.RLock()
    state._phase = "ready"
    state._ticket = None
    state._frame = None
    state._stop_receipt = None
    state._idle_join = None
    state._local_join_owned = False
    state._execution_sequence = 1
    state._installed = SimpleNamespace(generation=0)
    state._construction_snapshot = state._installed
    state.ep_group_health = SimpleNamespace(snapshot=lambda: state._installed)
    state._graphs = {1: (object(), object(), "batch-1", True)}
    state._watchdog = SimpleNamespace(last_error=None)
    state._streams = (object(), object())
    state._check_epoch = Mock()
    state._retained_inventory = Mock(return_value=(state._graphs, state._streams))
    state._finish_ticket = Mock()
    state._release_uncertain_if_ready = Mock()
    state.lifecycle = SimpleNamespace(stop_execution_watchdog=Mock())
    monkeypatch.setattr(execution_state.torch.cuda, "Event", lambda **kwargs: Mock())
    monkeypatch.setattr(execution_state, "wait_cuda_event", Mock())
    yield state
    execution_state._UNCERTAIN_STATES.pop(id(state), None)


def test_stop_joins_before_retiring_watchdog(owner: ExecutionState) -> None:
    observed = []
    execution_state.wait_cuda_event.side_effect = lambda *args: observed.append("join")
    owner.lifecycle.stop_execution_watchdog.side_effect = lambda *args, **kwargs: observed.append(
        "retire"
    )
    receipt = owner.stop(streams=(), deadline_monotonic=time.monotonic() + 5)
    assert observed == ["join", "join", "retire"]
    assert owner.validate_join(receipt) is receipt
    assert receipt.stopped and receipt.watchdog_retired
    assert owner._phase == "stopped"
    assert execution_state._UNCERTAIN_STATES[id(owner)] is owner


def test_incomplete_join_retains_roots_and_publishes_no_stop(owner: ExecutionState) -> None:
    execution_state.wait_cuda_event.side_effect = TimeoutError("incomplete event")
    with pytest.raises(TimeoutError, match="incomplete"):
        owner.stop(streams=(), deadline_monotonic=time.monotonic() + 5)
    assert owner._phase == "blocked"
    assert owner._stop_receipt is None
    assert owner._pending_join[0][0] is owner._graphs
    assert execution_state._UNCERTAIN_STATES[id(owner)] is owner
    owner.lifecycle.stop_execution_watchdog.assert_not_called()


def test_idle_join_cannot_authorize_stopped_execution(owner: ExecutionState) -> None:
    receipt = owner.join_idle(streams=(), deadline_monotonic=time.monotonic() + 5)
    assert not receipt.stopped and not receipt.watchdog_retired
    assert owner._phase == "ready"
    owner.lifecycle.stop_execution_watchdog.assert_not_called()
    owner._execution_sequence += 1
    with pytest.raises(RuntimeError, match="stale"):
        owner.validate_join(receipt)


def test_failed_watchdog_retirement_keeps_joined_roots(owner: ExecutionState) -> None:
    owner.lifecycle.stop_execution_watchdog.side_effect = TimeoutError("pending copy")
    with pytest.raises(TimeoutError, match="pending copy"):
        owner.stop(streams=(), deadline_monotonic=time.monotonic() + 5)
    assert owner._stop_receipt is None
    assert owner._pending_join[0][0] is owner._graphs
    assert execution_state._UNCERTAIN_STATES[id(owner)] is owner


def test_final_stop_join_retains_original_receipt(owner: ExecutionState) -> None:
    first = owner.stop(streams=(), deadline_monotonic=time.monotonic() + 5)
    final = owner.finish_stop(first, deadline_monotonic=time.monotonic() + 5)
    assert final.previous is first
    assert owner.validate_join(final) is final
    assert owner._phase == "stopped"
    owner.lifecycle.stop_execution_watchdog.assert_called_once()


def test_pending_final_join_retains_roots_without_final_receipt(owner: ExecutionState) -> None:
    first = owner.stop(streams=(), deadline_monotonic=time.monotonic() + 5)
    execution_state.wait_cuda_event.side_effect = TimeoutError("pending native bridge")
    with pytest.raises(TimeoutError, match="pending native bridge"):
        owner.finish_stop(first, deadline_monotonic=time.monotonic() + 5)
    assert owner._stop_receipt is first
    assert owner._pending_join[-1] is first
    assert owner._phase == "blocked"
    assert execution_state._UNCERTAIN_STATES[id(owner)] is owner


def test_idle_seal_blocks_gpu_issuance_but_allows_original_stop(owner: ExecutionState) -> None:
    receipt = owner.join_idle(streams=(), deadline_monotonic=time.monotonic() + 5)
    owner.hold_idle(receipt)
    assert owner.validate_join(receipt) is receipt
    assert owner._phase == "idle-held"
    with pytest.raises(RuntimeError, match="Forward blocked"):
        owner.begin_forward()
    with pytest.raises(RuntimeError, match="Replay blocked"):
        owner.before_replay(object(), graph_key="batch-1")
    with pytest.raises(RuntimeError, match="Capture blocked"):
        owner.capture_started(object(), (0, 0), graph_key="batch-1", resource_owner=None)
    with pytest.raises(RuntimeError, match="cannot reset"):
        owner.before_graph_reset()
    stopped = owner.stop(streams=(), deadline_monotonic=time.monotonic() + 5)
    assert stopped.stopped and stopped.watchdog_retired
    assert owner.validate_join(stopped) is stopped
    with pytest.raises(RuntimeError, match="stale"):
        owner.validate_join(receipt)


def test_execution_profile_rejects_captured_lm_head_tensor_parallelism() -> None:
    from tensorrt_llm._torch.pyexecutor.model_engine import PyTorchModelEngine

    engine = object.__new__(PyTorchModelEngine)
    engine.mapping = SimpleNamespace(enable_lm_head_tp_in_adp=True)
    with pytest.raises(RuntimeError, match="LM-head tensor parallelism"):
        engine._qualify_execution_owners(())


def test_local_join_roots_survive_completed_intermediate_join(owner: ExecutionState) -> None:
    owner.begin_local_join()
    ExecutionState._release_uncertain_if_ready(owner)
    assert execution_state._UNCERTAIN_STATES[id(owner)] is owner
    assert owner._phase == "ready"
    owner._phase = "stopped"
    owner.begin_local_join()
    assert execution_state._UNCERTAIN_STATES[id(owner)] is owner


@pytest.mark.parametrize("changed", [None, "executable", "descriptor", "mapping"])
def test_capture_footprint_contains_scalars_and_original_mapping(
    owner: ExecutionState, changed: str | None
) -> None:
    graph = SimpleNamespace(raw_cuda_graph=lambda: 101, raw_cuda_graph_exec=lambda: 102)
    pool, key, token = (1, 2), "batch-1", object()
    allocation = execution_state.CapturedTensorSpan(
        "gather_payload_0", 8192, 32, (1, 16), "torch.bfloat16", "native_torch_empty"
    )
    frontend = SimpleNamespace(captured_allocation_spans=lambda phase: (1, (allocation,)))
    owner._graphs = {id(graph): (graph, pool, key, False)}
    owner._graph_schedules = {}
    owner._capture_footprints = {}
    owner._completed_schedule = (("dispatch", frontend), ("combine", frontend))
    owner._geometry = (2, 0, 16, 32, "torch.bfloat16", 1024, 4096, 2048, 2048)
    owner._clients = (SimpleNamespace(execution_geometry=lambda: owner._geometry),)
    owner._retained_inventory.return_value = ((object(), object(), token),)
    owner._join = Mock()
    owner._healthy_device = Mock()
    owner.abort, owner.descriptor, owner.published, owner.slot = (
        SimpleNamespace(data_ptr=lambda address=address: address) for address in (1, 2, 3, 4)
    )

    owner.capture_finished(graph, pool, graph_key=key)
    (footprint,) = owner.captured_footprints
    assert footprint.mapping_token is token
    assert (footprint.raw_graph, footprint.executable) == (101, 102)
    assert footprint.graph_owner_id == id(graph)
    assert footprint.descriptor_addresses == (1, 2, 3, 4)
    assert footprint.spans == ((allocation,), (allocation,))
    assert footprint.schedule == (("dispatch", id(frontend)), ("combine", id(frontend)))
    assert owner._graphs[id(graph)][3]

    owner._installed = SimpleNamespace(generation=1)
    assert owner.captured_footprints == (footprint,)
    assert footprint.capture_generation == 0 and owner.generation == 1
    if changed == "executable":
        graph.raw_cuda_graph_exec = lambda: 103
    elif changed == "descriptor":
        owner.descriptor = SimpleNamespace(data_ptr=lambda: 999)
    elif changed == "mapping":
        owner._retained_inventory.return_value = ((object(), object(), object()),)
    if changed is not None:
        with pytest.raises(RuntimeError, match="identity changed"):
            _ = owner.captured_footprints

    owner._graphs[id(graph)] = (graph, pool, key, False)
    with pytest.raises(RuntimeError, match="incomplete"):
        _ = owner.captured_footprints


def test_stale_allocation_spans_cannot_publish_capture(owner: ExecutionState) -> None:
    graph = SimpleNamespace(raw_cuda_graph=lambda: 101, raw_cuda_graph_exec=lambda: 102)
    frontend = SimpleNamespace(captured_allocation_spans=lambda phase: (0, ()))
    owner._graphs = {id(graph): (graph, (1, 2), "batch-1", False)}
    owner._graph_schedules = {}
    owner._capture_footprints = {}
    owner._completed_schedule = (("dispatch", frontend),)
    owner._join, owner._healthy_device = Mock(), Mock()
    with pytest.raises(RuntimeError, match="stale"):
        owner.capture_finished(graph, (1, 2), graph_key="batch-1")
    assert not owner._graphs[id(graph)][3]
    assert not owner._capture_footprints


@pytest.mark.parametrize(
    "origin,pointer,rejects",
    [
        ("native_torch_empty", 8192, False),
        ("native_torch_empty", 6200, True),
        ("canonical_workspace_tail", 5120, False),
        ("canonical_workspace_tail", 6200, True),
    ],
)
def test_capture_spans_reject_peer_alias_and_retain_no_tensor(
    monkeypatch: pytest.MonkeyPatch, origin: str, pointer: int, rejects: bool
) -> None:
    import weakref

    from tensorrt_llm._torch.moe.fused_moe.communication.nvlink_one_sided import NVLinkOneSided

    state = SimpleNamespace(execution_sequence=3, validate_frontend_epoch=Mock())
    frontend = object.__new__(NVLinkOneSided)
    frontend._execution_state = state
    frontend._capture_allocation_spans = {}
    frontend.workspace = SimpleNamespace(device="cuda:0")
    frontend.execution_geometry = lambda: (2, 0, 16, 32, "bfloat16", 1024, 4096, 2048, 2048)
    tensor = SimpleNamespace(
        is_contiguous=lambda: True,
        device="cuda:0",
        data_ptr=lambda: pointer,
        numel=lambda: 16,
        element_size=lambda: 2,
        shape=(1, 16),
        dtype="bfloat16",
    )

    # A weak-referenceable test tensor makes unintended ownership observable.
    class Tensor:
        pass

    actual = Tensor()
    actual.__dict__.update(vars(tensor))
    reference = weakref.ref(actual)
    monkeypatch.setattr(execution_state.torch.cuda, "is_current_stream_capturing", lambda: True)
    if rejects:
        with pytest.raises(RuntimeError, match="aliases|canonical local tail"):
            frontend._record_capture_spans("dispatch", (("output", actual, origin),))
        assert not frontend._capture_allocation_spans
    else:
        frontend._record_capture_spans("dispatch", (("output", actual, origin),))
        sequence, spans = frontend.captured_allocation_spans("dispatch")
        assert sequence == 3 and spans[0].pointer == pointer and spans[0].byte_size == 32
    del actual
    assert reference() is None
