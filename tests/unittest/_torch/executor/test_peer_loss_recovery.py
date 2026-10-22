# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Scheduler-side WideEP peer-loss recovery with fake distributed, model, and authority."""

import functools
import signal
import threading
import time
from types import SimpleNamespace
from unittest.mock import ANY, MagicMock, Mock, call

import pytest
import torch

from tensorrt_llm._torch.distributed.communicator import HostCollectiveInterrupted, MPIDist
from tensorrt_llm._torch.moe.fused_moe.ep_group_health import EPGroupHealth
from tensorrt_llm._torch.pyexecutor import llm_request, peer_loss_recovery, py_executor
from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import KVCacheManagerV2
from tensorrt_llm._torch.pyexecutor.llm_request import PyResult
from tensorrt_llm._torch.pyexecutor.peer_loss_recovery import (
    PeerLossRecovery,
    PreparedStep,
    RecoveryHook,
)
from tensorrt_llm._torch.pyexecutor.py_executor import PyExecutor
from tensorrt_llm._torch.pyexecutor.scheduler import FCFSWaitingQueue
from tensorrt_llm._torch.pyexecutor.scheduler.adp_router import DefaultADPRouter
from tensorrt_llm._torch.speculative.interface import SpeculativeDecodingMode as SpecMode
from tensorrt_llm.executor.recovery.client import (
    RecoveryAborted,
    RecoveryInterrupted,
    RecoveryTimeout,
)
from tensorrt_llm.executor.recovery.config import FaultToleranceConfig
from tensorrt_llm.executor.recovery.protocol import (
    Abort,
    Commit,
    Fence,
    Fenced,
    Install,
    Installed,
    Propose,
    Ready,
    Stopped,
)
from tensorrt_llm.runtime.kv_cache_manager_v2 import GPU_LEVEL

pytestmark = pytest.mark.cpu_only

WORLD_SIZE = 4
FAILED_RANK = 2
SURVIVORS = (0, 1, 3)
ADMITTED = frozenset({frozenset({1}), frozenset({FAILED_RANK}), frozenset({3})})


class _FakeClient:
    """Scripted authority replies; records what the worker sent."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.sent = []

    def send(self, message):
        self.sent.append(message)

    def discard_pending(self):
        pass

    def wait(self, message_type, deadline, interrupt=None, skip=None):
        assert deadline > time.monotonic()
        reply = self.replies.pop(0)
        while skip is not None and skip(reply):
            reply = self.replies.pop(0)
        if reply is _FURTHER_LOSS:
            assert interrupt is not None
            raise RecoveryInterrupted("another peer failed")
        if isinstance(reply, Abort):
            raise RecoveryAborted(reply.reason)
        assert isinstance(reply, message_type)
        return reply


# A scripted reply: the survivor watch fires while the worker waits for the authority.
_FURTHER_LOSS = object()


def _happy_replies(generation=1, failed=(FAILED_RANK,), survivors=SURVIVORS):
    return [
        Propose(generation=generation, failed_ranks=failed, survivors=survivors),
        Install(generation=generation),
        Commit(generation=generation),
    ]


def _executor(rank=1, failed=frozenset({FAILED_RANK}), world_size=WORLD_SIZE):
    executor = PyExecutor.__new__(PyExecutor)
    executor.iter_counter = 0
    events = []
    dist = Mock(
        rank=rank,
        tp_rank=rank,
        world_size=world_size,
        live_ranks=tuple(r for r in range(world_size) if r not in failed),
    )
    dist.revoke.side_effect = lambda: events.append("revoke")
    dist.shrink.side_effect = lambda: events.append("shrink") or failed
    dist.reported_failed_ranks.return_value = frozenset()
    dist.install_survivors.side_effect = lambda: events.append("survivors")
    dist.watch_peer_failures.side_effect = lambda callback, abort_word: events.append(
        "watch"
    ) or Mock(failed=False)
    dist.broadcast.side_effect = lambda value, root: value
    dist.allgather.side_effect = lambda value, departed: [
        departed if r in failed else value for r in range(world_size)
    ]
    engine = Mock()
    engine.moe_load_balancer.is_static_routing.return_value = True
    engine.moe_load_balancer.recovery_placements.return_value = ["placement"]
    engine.moe_load_balancer.restore_backing_bytes.return_value = 0
    engine.moe_load_balancer.placement_digest.return_value = "digest"
    engine.moe_load_balancer.install_recovery_placement.side_effect = (
        lambda failed, placements: events.append(("placement", frozenset(failed)))
    )
    engine.moe_execution_state.health = EPGroupHealth(world_size)
    engine.moe_execution_state.aborted.return_value = False
    executor.dist = dist
    executor.model_engine = engine
    executor.executor_request_queue = Mock()
    executor.executor_request_queue.wake.side_effect = lambda: events.append("wake")
    executor.kv_cache_manager = Mock()
    executor.draft_kv_cache_manager = None
    executor.active_requests = []
    executor.waiting_queue = FCFSWaitingQueue()
    executor.request_accumulated = []
    executor._pending_transfer_responses = []
    executor._pending_response_terminations = []
    executor._terminate_request = Mock()
    executor._enqueue_responses = Mock()
    executor.gather_all_responses = False
    executor.previous_batch = None
    executor.hang_detector = MagicMock()
    executor._events = events
    return executor


def _recovery(
    client,
    rank=1,
    failed=frozenset({FAILED_RANK}),
    world_size=WORLD_SIZE,
    admitted=ADMITTED,
):
    executor = _executor(rank, failed, world_size)
    # A static recovery stages its ring copies: by default the plan copies nothing.
    executor.model_engine.moe_load_balancer.restore_copy_plan.return_value = []
    recovery = PeerLossRecovery(executor, client, admitted)
    recovery._join_device_work = Mock()
    recovery._remap_workspaces = Mock()
    recovery._watch = Mock(failed=False)
    executor.discard_interrupted_steps = Mock()
    executor._peer_loss_recovery = recovery
    return recovery


def _interrupted(failed=frozenset({FAILED_RANK})):
    return HostCollectiveInterrupted("tp_allgather", failed)


def test_recovery_runs_protocol_and_installs_in_order():
    client = _FakeClient(_happy_replies(generation=1))
    recovery = _recovery(client)
    request = Mock(is_attention_dp_dummy=False)
    recovery._executor.active_requests.append(request)
    recovery._peer_failed.set()

    recovery._recover(_interrupted())

    assert client.sent == [
        Stopped(rank=1, failed_ranks=(FAILED_RANK,)),
        Ready(rank=1, generation=1, placement_digest="digest"),
        Installed(rank=1, generation=1),
    ]
    failed = frozenset({FAILED_RANK})
    assert recovery._executor._events == [
        "revoke",
        "shrink",
        "survivors",
        ("placement", failed),
        "watch",
    ]
    assert recovery._executor.active_requests == [request]
    assert not recovery._peer_failed.is_set()
    load_balancer = recovery._executor.model_engine.moe_load_balancer
    # A static placement needs no agreed base.
    load_balancer.recovery_placements.assert_called_once_with(failed, None)
    # The placements installed are the ones whose digest went out with Ready, with their ring copy plan.
    load_balancer.placement_digest.assert_called_once_with(
        failed, ["placement"], load_balancer.restore_copy_plan.return_value
    )
    load_balancer.install_recovery_placement.assert_called_once_with([FAILED_RANK], ["placement"])


def test_static_placement_with_replicas_restores_from_full_host_copies():
    """Its replicas stay routed and every node keeps every host copy: the install loads, nothing is staged."""
    recovery = _recovery(_FakeClient(_happy_replies(generation=1)))
    load_balancer = recovery._executor.model_engine.moe_load_balancer
    load_balancer.copy_nodes = None
    recovery._peer_failed.set()

    recovery._recover(_interrupted())

    failed = frozenset({FAILED_RANK})
    # The placement never changes: no update to wait for and no rank 0 base.
    load_balancer.quiesce.assert_not_called()
    load_balancer.recovery_placements.assert_called_once_with(failed, None)
    load_balancer.restore_copy_plan.assert_not_called()
    load_balancer.placement_digest.assert_called_once_with(failed, ["placement"], None)
    load_balancer.install_recovery_placement.assert_called_once_with([FAILED_RANK], ["placement"])


def test_ghost_slots_take_kv_cache_memory_only_when_the_device_is_short(monkeypatch):
    """Ghost slots get their memory before Ready; the KV cache gives up only what free memory lacks."""
    recovery = _recovery(_FakeClient([]))
    executor = recovery._executor
    executor.resize_kv_cache_quota = Mock(return_value=True)
    load_balancer = Mock()
    need = 1 << 30
    load_balancer.restore_backing_bytes.return_value = need
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)

    monkeypatch.setattr(peer_loss_recovery, "_free_device_bytes", lambda: 4 << 30)
    recovery._back_restored_slots(load_balancer, ["placement"])
    executor.resize_kv_cache_quota.assert_not_called()
    load_balancer.back_restored_slots.assert_called_once_with(["placement"])

    free = 512 << 20
    monkeypatch.setattr(peer_loss_recovery, "_free_device_bytes", lambda: free)
    recovery._back_restored_slots(load_balancer, ["placement"])
    lent = need + peer_loss_recovery._GHOST_MARGIN_BYTES - free
    executor.resize_kv_cache_quota.assert_called_once_with(-lent)

    # A plan abandoned for a further loss returns its slots' memory and the KV cache's share.
    load_balancer_of_engine = executor.model_engine.moe_load_balancer
    load_balancer_of_engine.release_uninstalled_backing.return_value = need
    recovery._release_uninstalled_backing()
    executor.resize_kv_cache_quota.assert_called_with(lent)
    assert recovery._kv_lent == 0

    # A KV cache that cannot shrink ends the recovery before any slot changes.
    executor.resize_kv_cache_quota.return_value = False
    load_balancer.back_restored_slots.reset_mock()
    with pytest.raises(RuntimeError, match="could not give up"):
        recovery._back_restored_slots(load_balancer, ["placement"])
    load_balancer.back_restored_slots.assert_not_called()

    # Nor does the recovery go on with a smaller KV cache when it cannot take its share back.
    recovery._kv_lent = lent
    with pytest.raises(RuntimeError, match="could not take back"):
        recovery._release_uninstalled_backing()
    executor.resize_kv_cache_quota.assert_called_with(lent)


def test_remap_releases_on_every_survivor_before_taking_room_and_remapping(monkeypatch):
    """The workspaces are released and the survivors meet before room is made and the fresh pages are mapped
    through the survivors' gather; the KV cache's share is not returned on a later abandon."""
    recovery = _recovery(_FakeClient([]))
    del recovery._remap_workspaces
    executor = recovery._executor
    executor.resize_kv_cache_quota = Mock(return_value=True)
    dist, state = executor.dist, executor.model_engine.moe_execution_state
    order = []
    need, free = 1 << 30, 512 << 20
    state.release_workspaces.side_effect = lambda departed: order.append("release") or need
    dist.barrier_among_survivors.side_effect = lambda: order.append("barrier")
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(
        peer_loss_recovery, "_free_device_bytes", lambda: order.append("room") or free
    )
    state.remap_workspaces.side_effect = lambda gather: order.append(("remap", gather))

    recovery._remap_workspaces(frozenset({FAILED_RANK}), float("inf"))

    state.release_workspaces.assert_called_with(frozenset({FAILED_RANK}))
    assert order[:2] == ["release", "barrier"]
    assert "room" in order[2:-2]
    assert order[-2:] == [("remap", dist.allgather_among_survivors), "barrier"]
    executor.resize_kv_cache_quota.assert_called_once_with(
        -(need + peer_loss_recovery._GHOST_MARGIN_BYTES - free)
    )
    assert recovery._kv_lent == 0
    assert recovery._remap_lent == need + peer_loss_recovery._GHOST_MARGIN_BYTES - free

    # A rank short of room still joins the remap, whose exchange then fails on every survivor.
    executor.resize_kv_cache_quota.return_value = False
    order.clear()
    recovery._remap_workspaces(frozenset({FAILED_RANK}), float("inf"))
    assert order[-2:] == [("remap", dist.allgather_among_survivors), "barrier"]


def test_the_remap_checks_the_recovery_deadline_before_and_after_its_exchanges(monkeypatch):
    """The remap's exchanges block without a deadline: a recovery past its deadline does not start one, and
    one that ran past it stops there."""
    recovery = _recovery(_FakeClient([]))
    del recovery._remap_workspaces
    executor = recovery._executor
    executor.resize_kv_cache_quota = Mock(return_value=True)
    state = executor.model_engine.moe_execution_state
    state.release_workspaces.return_value = 0
    monkeypatch.setattr(peer_loss_recovery, "_free_device_bytes", lambda: 1 << 40)
    now = [100.0]
    monkeypatch.setattr(peer_loss_recovery.time, "monotonic", lambda: now[0])

    with pytest.raises(RecoveryTimeout, match="before remapping"):
        recovery._remap_workspaces(frozenset({FAILED_RANK}), 100.0)
    state.release_workspaces.assert_not_called()

    def slow_remap(gather):
        now[0] += 10

    state.remap_workspaces.side_effect = slow_remap
    with pytest.raises(RecoveryTimeout, match="after the all-to-all workspace remap"):
        recovery._remap_workspaces(frozenset({FAILED_RANK}), 105.0)
    state.remap_workspaces.assert_called_once()


def test_a_commit_returns_the_remap_s_kv_cache_share_as_far_as_memory_is_free(monkeypatch):
    """The failed ranks' old pages are freed only when their processes end: each commit returns what is free."""
    recovery = _recovery(_FakeClient([]))
    executor = recovery._executor
    executor.resize_kv_cache_quota = Mock(return_value=True)
    margin = peer_loss_recovery._GHOST_MARGIN_BYTES
    recovery._remap_lent = lent = 1 << 30

    monkeypatch.setattr(peer_loss_recovery, "_free_device_bytes", lambda: margin)
    recovery._return_remap_room()
    executor.resize_kv_cache_quota.assert_not_called()

    monkeypatch.setattr(peer_loss_recovery, "_free_device_bytes", lambda: margin + (256 << 20))
    recovery._return_remap_room()
    executor.resize_kv_cache_quota.assert_called_once_with(256 << 20)
    assert recovery._remap_lent == lent - (256 << 20)

    # A KV cache that cannot grow keeps the share lent.
    executor.resize_kv_cache_quota.return_value = False
    monkeypatch.setattr(peer_loss_recovery, "_free_device_bytes", lambda: margin + (4 << 30))
    recovery._return_remap_room()
    assert recovery._remap_lent == lent - (256 << 20)

    executor.resize_kv_cache_quota.return_value = True
    recovery._return_remap_room()
    executor.resize_kv_cache_quota.assert_called_with(lent - (256 << 20))
    assert recovery._remap_lent == 0


def test_an_idle_executor_returns_the_remap_s_share_once_memory_is_free(monkeypatch):
    """The failed ranks' old pages are usually freed after the commit; an idle executor returns the share
    then, checking at most once per interval and not at all once nothing is lent."""
    recovery = _recovery(_FakeClient([]))
    executor = recovery._executor
    executor.resize_kv_cache_quota = Mock(return_value=True)
    margin = peer_loss_recovery._GHOST_MARGIN_BYTES
    interval = peer_loss_recovery._REMAP_RETURN_INTERVAL_S
    now = [100.0]
    free = [margin]
    monkeypatch.setattr(peer_loss_recovery.time, "monotonic", lambda: now[0])
    free_device_bytes = Mock(side_effect=lambda: free[0])
    monkeypatch.setattr(peer_loss_recovery, "_free_device_bytes", free_device_bytes)
    recovery._remap_lent = lent = 1 << 30

    recovery.serving_idle()
    executor.resize_kv_cache_quota.assert_not_called()
    assert free_device_bytes.call_count == 1

    # Freed meanwhile, but checked again only after the interval.
    free[0] = margin + (2 << 30)
    now[0] += interval / 2
    recovery.serving_idle()
    assert free_device_bytes.call_count == 1
    now[0] += interval
    recovery.serving_idle()
    executor.resize_kv_cache_quota.assert_called_once_with(lent)
    assert recovery._remap_lent == 0

    # Nothing lent: no check at all.
    now[0] += 2 * interval
    recovery.serving_idle()
    assert free_device_bytes.call_count == 2
    executor.resize_kv_cache_quota.assert_called_once()


@pytest.mark.parametrize("in_flight", [False, True])
def test_the_executor_reports_idle_only_without_a_step_in_flight(monkeypatch, in_flight):
    """The overlap loop has a step in flight while ``previous_batch`` is set; only without one may the KV cache
    resize."""
    executor = PyExecutor.__new__(PyExecutor)
    executor._peer_loss_recovery = Mock(enabled=True)
    executor.previous_batch = Mock() if in_flight else None
    executor.hang_detector = MagicMock()
    executor._profiler = MagicMock()
    executor.device_id = 0
    executor.enable_iter_perf_stats = False
    executor._resource_governor_enabled = False
    executor._is_kv_manager_v2 = False
    executor.is_benchmark_disagg = False
    executor._disagg_coordinator = Mock()
    executor._wait_for_model_engine_input_copy = Mock()
    executor._prepare_and_schedule_batch = Mock(return_value=(None, None))
    executor._flush_pending_transfer_responses = Mock()

    monkeypatch.setattr(torch.cuda, "set_device", lambda device: None)
    monkeypatch.setattr(py_executor, "CUASSERT", lambda result: result)
    monkeypatch.setattr(py_executor, "cudart", Mock())

    PyExecutor._executor_loop_overlap(executor)

    assert executor._peer_loss_recovery.serving_idle.call_count == (0 if in_flight else 1)


def test_every_survivor_resets_its_remapped_rows_before_any_stages_into_them(monkeypatch):
    """The remap resets this rank's rows; a faster peer's ring restore staging writes into them only after
    every survivor passed the barrier that follows that reset."""
    import numpy as np

    recovery = _recovery(_FakeClient(_happy_replies(generation=1)))
    del recovery._remap_workspaces
    monkeypatch.setattr(peer_loss_recovery, "_free_device_bytes", lambda: 4 << 30)
    engine, dist = recovery._executor.model_engine, recovery._executor.dist
    events = []
    engine.moe_execution_state.release_workspaces.side_effect = (
        lambda departed: events.append("release") or 0
    )
    engine.moe_execution_state.remap_workspaces.side_effect = lambda gather: events.append("reset")
    dist.barrier_among_survivors.side_effect = lambda: events.append("barrier")
    load_balancer = engine.moe_load_balancer
    load_balancer.restore_copy_plan.return_value = np.zeros((1, 10), dtype=np.int64)
    load_balancer.stage_restored_slots.side_effect = lambda plan, workspace, barrier: (
        events.append("stage")
    )

    recovery._recover(_interrupted())

    reset, stage = events.index("reset"), events.index("stage")
    assert "barrier" in events[reset + 1 : stage]


def test_kv_cache_quota_cut_suspends_the_active_caches_around_the_resize():
    executor = PyExecutor.__new__(PyExecutor)
    mgr = Mock()
    mgr.impl.get_quota.return_value = 10 << 30
    mgr.impl.resize.return_value = True
    mgr.is_request_active.side_effect = lambda request_id: request_id == 1
    requests = [Mock(py_request_id=1), Mock(py_request_id=2)]
    executor.kv_cache_manager = mgr
    executor.active_requests = requests
    executor.model_engine = SimpleNamespace(cuda_graph_runner=None)
    executor._read_kv_cache_capacity = Mock(return_value={})

    assert executor.resize_kv_cache_quota(-(1 << 30))
    mgr.suspend_request.assert_called_once_with(requests[0])
    mgr.impl.resize.assert_called_once_with(GPU_LEVEL, 9 << 30)
    mgr.resume_request.assert_called_once_with(requests[0])
    # More than the whole quota: nothing to resize.
    assert not executor.resize_kv_cache_quota(-(10 << 30))
    assert mgr.impl.resize.call_count == 1
    # A failed resize still resumes the suspended requests.
    mgr.impl.resize.side_effect = RuntimeError("resize")
    with pytest.raises(RuntimeError, match="resize"):
        executor.resize_kv_cache_quota(-(1 << 30))
    assert mgr.resume_request.call_count == mgr.suspend_request.call_count == 3


def test_a_node_loss_recovers_every_rank_in_one_transaction():
    """Losing node 1 (ranks 4-7) masks all four ranks and runs the requests each one served again."""
    node = frozenset({4, 5, 6, 7})
    client = _FakeClient(_happy_replies(failed=(4, 5, 6, 7), survivors=(0, 1, 2, 3)))
    recovery = _recovery(client, rank=0, failed=node, world_size=8, admitted=ADMITTED | {node})
    recovery._request_owners = {
        request_id: (owner, Mock(id=request_id, request=Mock(client_id=None), child_req_ids=None))
        for request_id, owner in ((11, 4), (12, 7), (13, 2))
    }
    recovery._executor.active_requests.append(
        Mock(is_child=False, py_request_id=13, is_attention_dp_dummy=False)
    )
    sent = _sent_responses(recovery)

    recovery._recover(_interrupted(node))

    assert client.sent[0] == Stopped(rank=0, failed_ranks=(4, 5, 6, 7))
    engine = recovery._executor.model_engine
    engine.moe_load_balancer.recovery_placements.assert_called_once_with(node, None)
    engine.moe_load_balancer.install_recovery_placement.assert_called_once_with(
        [4, 5, 6, 7], ["placement"]
    )
    assert engine.moe_execution_state.health.get_failed_ranks() == node
    assert sent[-1] == []
    assert sorted(item.id for item in recovery._executor.request_accumulated) == [11, 12]
    assert list(recovery._request_owners) == [13]


def test_a_loss_during_the_recovery_plans_again_without_every_lost_rank():
    """Rank 3 fails while rank 1 waits for the plan without rank 2: rank 1 shrinks again, checks that the
    survivors' slots serve every expert without both, and recovers without both; the first plan's
    leftovers are dropped."""
    lost = (FAILED_RANK, 3)
    client = _FakeClient(
        [
            _FURTHER_LOSS,
            Propose(1, (FAILED_RANK,), SURVIVORS),
            Propose(1, lost, (0, 1)),
            Install(1),
            Commit(1),
        ]
    )
    recovery = _recovery(client, failed=frozenset(lost))
    dist = recovery._executor.dist
    shrinks = iter([frozenset({FAILED_RANK}), frozenset(lost)])
    dist.shrink.side_effect = lambda: next(shrinks)
    load_balancer = recovery._executor.model_engine.moe_load_balancer
    load_balancer.covers_loss_of.return_value = True

    recovery._recover(_interrupted())

    assert client.sent == [
        Stopped(rank=1, failed_ranks=(FAILED_RANK,)),
        Stopped(rank=1, failed_ranks=lost),
        Ready(rank=1, generation=1, placement_digest="digest"),
        Installed(rank=1, generation=1),
    ]
    assert dist.watch_survivor_failures.call_count == 2
    load_balancer.covers_loss_of.assert_called_once_with(frozenset(lost))
    load_balancer.install_recovery_placement.assert_called_once_with(list(lost), ["placement"])


def test_a_shrink_again_restarts_the_hang_detection_bound():
    """The executor's hang detector bounds one shrink and the recovery deadline; a second shrink, after up to
    that much time, gets a bound of its own."""
    lost = (FAILED_RANK, 3)
    client = _FakeClient([_FURTHER_LOSS, Propose(1, lost, (0, 1)), Install(1), Commit(1)])
    recovery = _recovery(client, failed=frozenset(lost))
    executor = recovery._executor
    events = executor._events
    shrinks = iter([frozenset({FAILED_RANK}), frozenset(lost)])
    executor.dist.shrink.side_effect = lambda: events.append("shrink") or next(shrinks)
    executor.hang_detector.checkpoint.side_effect = lambda: events.append("checkpoint")
    executor.model_engine.moe_load_balancer.covers_loss_of.return_value = True

    recovery._recover(_interrupted())

    assert [e for e in events if e in ("shrink", "checkpoint")] == [
        "shrink",
        "checkpoint",
        "shrink",
    ]


def test_a_survivor_lost_during_the_recovery_releases_the_survivor_collectives():
    """A survivor blocked in a survivor collective is released by the revocation, not only the event."""
    recovery = _recovery(_FakeClient([]))
    dist = recovery._executor.dist
    further_loss = threading.Event()
    recovery._watch_survivors(further_loss)
    ((on_failure,), _) = dist.watch_survivor_failures.call_args
    dist.revoke.assert_not_called()
    on_failure()
    assert further_loss.is_set()
    dist.revoke.assert_called_once_with()


def test_a_loss_during_the_recovery_the_slots_cannot_serve_fails_closed():
    client = _FakeClient([_FURTHER_LOSS])
    recovery = _recovery(client)
    shrinks = iter([frozenset({FAILED_RANK}), frozenset({FAILED_RANK, 3})])
    recovery._executor.dist.shrink.side_effect = lambda: next(shrinks)
    recovery._executor.model_engine.moe_load_balancer.covers_loss_of.return_value = False
    with pytest.raises(RuntimeError, match="not recoverable"):
        recovery._recover(_interrupted())


@pytest.mark.parametrize("covered", [True, False])
def test_a_loss_after_a_recovery_recovers_only_while_the_slots_cover_every_lost_rank(covered):
    """After the recovery without rank 2 committed, rank 3 is lost: the survivors shrink the installed communicator
    and recover without both at generation 2 if their slots still serve every expert, else fail closed."""
    lost = frozenset({FAILED_RANK, 3})
    client = _FakeClient(
        [
            *_happy_replies(),
            *_happy_replies(generation=2, failed=tuple(sorted(lost)), survivors=(0, 1)),
        ]
    )
    recovery = _recovery(client, failed=lost)
    shrinks = iter([frozenset({FAILED_RANK}), lost])
    recovery._executor.dist.shrink.side_effect = lambda: next(shrinks)
    load_balancer = recovery._executor.model_engine.moe_load_balancer
    load_balancer.covers_loss_of.return_value = covered
    recovery._recover(_interrupted())

    if not covered:
        with pytest.raises(RuntimeError, match="not recoverable"):
            recovery._recover(_interrupted(lost))
        load_balancer.covers_loss_of.assert_called_with(lost)
        return
    recovery._recover(_interrupted(lost))
    assert client.sent[3:] == [
        Stopped(rank=1, failed_ranks=(FAILED_RANK, 3)),
        Ready(rank=1, generation=2, placement_digest="digest"),
        Installed(rank=1, generation=2),
    ]
    load_balancer.recovery_placements.assert_called_with(lost, None)
    load_balancer.install_recovery_placement.assert_called_with([FAILED_RANK, 3], ["placement"])
    assert recovery._executor.model_engine.moe_execution_state.health.get_failed_ranks() == lost
    # The watch is armed again after each recovery.
    assert recovery._executor._events.count("watch") == 2


def _slow_shrink(recovery, seconds):
    """The first shrink returns after ``seconds``: a survivor blocked on a hung peer joins it once it is fenced."""

    def shrink():
        time.sleep(seconds)
        return frozenset({FAILED_RANK})

    recovery._executor.dist.shrink.side_effect = shrink


def test_a_shrink_that_waits_for_a_fence_within_its_timeout_still_recovers():
    """The shrink outlasts the recovery timeout; the recovery's deadline starts once the survivors agree."""
    client = _FakeClient(_happy_replies())
    recovery = _recovery(client)
    recovery._timeout_s = 0.2
    recovery._shrink_timeout_s = 2.0
    _slow_shrink(recovery, 0.4)
    joined = []
    recovery._join_device_work = lambda deadline: joined.append(deadline - time.monotonic())
    recovery._recover(_interrupted())
    assert len(joined) == 1 and joined[0] > 0
    assert isinstance(client.sent[-1], Installed) and not client.replies


def test_a_shrink_beyond_its_timeout_fails_closed():
    client = _FakeClient(_happy_replies())
    recovery = _recovery(client)
    recovery._timeout_s = 0.2
    recovery._shrink_timeout_s = 0.3
    _slow_shrink(recovery, 0.4)
    with pytest.raises(RecoveryTimeout, match="the survivors agreed on the failed ranks"):
        recovery._recover(_interrupted())
    recovery._join_device_work.assert_not_called()
    assert client.sent == []
    assert "survivors" not in recovery._executor._events


def test_rank0_ends_each_departed_rank_the_authority_asks_for_before_the_plan():
    client = _FakeClient([Fence(FAILED_RANK, 0), *_happy_replies()])
    recovery = _recovery(client, rank=0)
    recovery._end_ranks = Mock(return_value={FAILED_RANK: ""})
    recovery._recover(_interrupted())
    # The fence's call comes first; the shrink ends the departed ranks again.
    assert recovery._end_ranks.call_args_list[0] == call([FAILED_RANK])
    assert client.sent[:3] == [
        Stopped(rank=0, failed_ranks=(FAILED_RANK,)),
        Fenced(rank=0, target=FAILED_RANK, error=""),
        Ready(rank=0, generation=1, placement_digest="digest"),
    ]


def test_a_departed_rank_the_runtime_cannot_end_is_reported_and_the_recovery_ends():
    client = _FakeClient([Fence(FAILED_RANK, 0), Abort("departed rank 2 could not be fenced")])
    recovery = _recovery(client, rank=0)
    recovery._end_ranks = Mock(return_value={FAILED_RANK: "PMIx refused to kill rank 2"})
    with pytest.raises(RecoveryAborted, match="could not be fenced"):
        recovery._recover(_interrupted())
    assert client.sent[-1] == Fenced(
        rank=0, target=FAILED_RANK, error="PMIx refused to kill rank 2"
    )
    assert "survivors" not in recovery._executor._events


@pytest.mark.parametrize("outcome", ["ended", "refused", "no answer"])
def test_the_runtime_fence_of_a_departed_rank_reports_why_it_failed(monkeypatch, outcome):
    from tensorrt_llm import _utils
    from tensorrt_llm._torch.distributed import pmix_fence
    from tensorrt_llm.executor.recovery import config

    killed, release = [], threading.Event()

    def kill_rank(rank):
        killed.append(rank)
        if outcome == "refused":
            raise RuntimeError("PMIx refused to kill rank 12")
        if outcome == "no answer":
            release.wait()

    monkeypatch.setattr(_utils, "mpi_comm", lambda: None)
    # The rank in the MPI runtime differs from the rank in the session communicator.
    monkeypatch.setattr(pmix_fence, "pmix_ranks", lambda comm: (10, 11, 12, 13))
    monkeypatch.setattr(pmix_fence, "kill_rank", kill_rank)
    monkeypatch.setattr(config, "RUNTIME_FENCE_CALL_S", 0.1)
    recovery = _recovery(_FakeClient([]), rank=0)
    try:
        (error,) = recovery._end_ranks([FAILED_RANK]).values()
    finally:
        release.set()
        for thread in threading.enumerate():
            if thread.name == "rank_fence":
                thread.join(timeout=5)
    assert killed == [12]
    assert (
        error
        == {
            "ended": "",
            "refused": "PMIx refused to kill rank 12",
            "no answer": "the MPI runtime did not answer within 0.1s",
        }[outcome]
    )


def _runtime(monkeypatch, kill_rank, hang_detection=True):
    """The MPI runtime: session rank r is runtime rank 10 + r; ``kill_rank`` ends a runtime rank."""
    from tensorrt_llm import _utils
    from tensorrt_llm._torch.distributed import pmix_fence

    monkeypatch.setattr(_utils, "mpi_comm", lambda: None)
    monkeypatch.setattr(pmix_fence, "pmix_ranks", lambda comm: (10, 11, 12, 13))
    monkeypatch.setattr(pmix_fence, "kill_rank", kill_rank)
    monkeypatch.setattr(
        peer_loss_recovery,
        "fault_tolerance_config",
        lambda: FaultToleranceConfig(hang_detection=hang_detection),
    )


def test_a_shrink_waiting_on_a_detector_reported_live_rank_completes_once_it_is_killed(monkeypatch):
    """The failure detector declared rank 2 failed while its process is stopped: the agreement needs it dead."""
    dead = threading.Event()
    killed = []

    def kill_rank(rank):
        killed.append(rank)
        dead.set()

    _runtime(monkeypatch, kill_rank)
    recovery = _recovery(_FakeClient(_happy_replies()))
    dist = recovery._executor.dist
    dist.reported_failed_ranks.return_value = frozenset({FAILED_RANK})

    def shrink():
        # The real agreement blocks until the reported rank's process has ended.
        assert dead.wait(5.0), "the shrink waited on a live failed rank"
        return frozenset({FAILED_RANK})

    dist.shrink.side_effect = shrink
    recovery._recover(_interrupted())
    assert killed == [10 + FAILED_RANK]
    assert dist.shrink.call_count == 1


def test_ranks_that_cannot_be_ended_before_the_shrink_are_left_to_the_authority_s_fence(
    monkeypatch,
):
    def kill_rank(rank):
        raise RuntimeError(f"PMIx refused to kill rank {rank}")

    _runtime(monkeypatch, kill_rank)
    client = _FakeClient(_happy_replies())
    recovery = _recovery(client)
    recovery._executor.dist.reported_failed_ranks.return_value = frozenset({FAILED_RANK})
    recovery._recover(_interrupted())
    assert recovery._executor.dist.shrink.call_count == 1
    assert isinstance(client.sent[-1], Installed)


def test_nothing_is_killed_before_a_shrink_when_no_failure_is_reported(monkeypatch):
    # Kills run on fence threads, where a raised assertion would not fail the test: record them instead.
    killed = []
    _runtime(monkeypatch, killed.append)
    recovery = _recovery(_FakeClient(_happy_replies()))
    recovery._recover(_interrupted())
    assert killed == []
    assert recovery._executor.dist.shrink.call_count == 1


def test_without_hang_detection_no_reported_rank_is_killed_before_a_shrink(monkeypatch):
    killed = []
    _runtime(monkeypatch, killed.append, hang_detection=False)
    recovery = _recovery(_FakeClient(_happy_replies()))
    recovery._executor.dist.reported_failed_ranks.return_value = frozenset({FAILED_RANK})
    recovery._recover(_interrupted())
    assert killed == []
    assert recovery._executor.dist.shrink.call_count == 1


def test_a_loss_after_the_recovery_deadline_does_not_shrink_again():
    """A started shrink cannot be interrupted, so none starts once the deadline passed."""
    recovery = _recovery(_FakeClient([_FURTHER_LOSS]))
    recovery._timeout_s = 0.2
    dist = recovery._executor.dist
    # Stopping the survivor watch outlasts the deadline.
    dist.watch_survivor_failures.return_value = Mock(stop=lambda: time.sleep(0.3))
    with pytest.raises(RecoveryTimeout, match="shrinking the survivors again"):
        recovery._recover(_interrupted())
    assert dist.shrink.call_count == 1
    assert dist.revoke.call_count == 1


def test_a_round_after_a_late_shrink_does_not_start_survivor_collectives():
    """A shrink that returns after the deadline ends the recovery before the next round's watch and broadcast."""
    import numpy as np

    recovery = _recovery(_FakeClient([_FURTHER_LOSS]))
    recovery._timeout_s = 0.2
    dist = recovery._executor.dist
    shrinks = iter([frozenset({FAILED_RANK}), frozenset({FAILED_RANK, 3})])

    def shrink():
        lost = next(shrinks)
        if len(lost) > 1:
            time.sleep(0.3)  # the second shrink returns after the deadline
        return lost

    dist.shrink.side_effect = shrink
    load_balancer = recovery._executor.model_engine.moe_load_balancer
    load_balancer.covers_loss_of.return_value = True
    # Online EPLB: every round broadcasts rank 0's placement among the survivors.
    load_balancer.is_static_routing.return_value = False
    load_balancer.copy_nodes = None
    load_balancer.quiesce.return_value = True
    load_balancer.current_placements.return_value = [np.zeros((WORLD_SIZE, 2))]
    dist.broadcast_among_survivors = Mock(return_value=[np.zeros((WORLD_SIZE, 2))])
    with pytest.raises(RecoveryTimeout, match="planning the recovery again"):
        recovery._recover(_interrupted())
    assert dist.watch_survivor_failures.call_count == 1
    assert dist.broadcast_among_survivors.call_count == 1


def test_survivors_resume_from_the_same_iteration():
    """A loss can stop survivors one iteration apart; they resume from the furthest one."""
    recovery = _recovery(_FakeClient(_happy_replies()), rank=1)
    executor = recovery._executor
    executor.iter_counter = 7
    executor.dist.allgather.side_effect = lambda value, departed: [
        (set(), 8, set(), set()),
        value,
        None,
        (set(), 7, set(), set()),
    ]

    recovery._recover(_interrupted())

    assert executor.iter_counter == 8


def test_requests_with_occurrence_penalties_are_rejected():
    recovery = _recovery(_FakeClient([]))
    request = Mock(
        sampling_config=Mock(repetition_penalty=1.0, presence_penalty=0.5, frequency_penalty=None),
        py_multimodal_data=None,
        py_logits_post_processors=None,
    )
    with pytest.raises(ValueError, match="penalties"):
        recovery.validate_request(request)
    request.sampling_config.presence_penalty = 0.0
    recovery.validate_request(request)
    # A logits post-processor may keep state, and a replayed step would run it twice.
    request.py_logits_post_processors = [Mock()]
    with pytest.raises(ValueError, match="post-processors"):
        recovery.validate_request(request)


@pytest.mark.parametrize("overlap", [False, True])
def test_multimodal_requests_are_rejected_only_under_the_overlap_scheduler(overlap):
    recovery = _recovery(_FakeClient([]))
    recovery._executor.disable_overlap_scheduler = not overlap
    request = Mock(
        sampling_config=Mock(repetition_penalty=1.0, presence_penalty=0.0, frequency_penalty=0.0),
        py_multimodal_data={"image": Mock()},
        py_logits_post_processors=None,
    )
    if overlap:
        with pytest.raises(ValueError, match="Multimodal"):
            recovery.validate_request(request)
    else:
        recovery.validate_request(request)


@pytest.mark.parametrize(
    "failed, covered",
    [(frozenset(), True), (frozenset({0, 1}), True), (frozenset({1, 2}), False)],
)
def test_recovery_requires_a_covered_union_of_admitted_domains(failed, covered):
    client = _FakeClient(_happy_replies())
    recovery = _recovery(client, rank=3, failed=failed)
    recovery._executor.model_engine.moe_load_balancer.covers_loss_of.return_value = covered
    with pytest.raises(RuntimeError, match="is not recoverable"):
        recovery._recover(_interrupted())
    assert client.sent == []


@pytest.mark.parametrize(
    "replies, match",
    [
        ([Propose(generation=1, failed_ranks=(3,), survivors=(0, 1, 2))], "authority proposed"),
        ([Propose(generation=1, failed_ranks=(2, 3), survivors=(0, 1))], "authority proposed"),
        ([Propose(generation=1, failed_ranks=(2,), survivors=(0, 3))], "authority proposed"),
        (_happy_replies()[:1] + [Install(generation=2)], "Expected recovery generation 1"),
        (_happy_replies()[:2] + [Commit(generation=2)], "Expected recovery generation 1"),
    ],
    ids=["wrong_failed_rank", "wrong_failed_set", "not_a_survivor", "install_gen", "commit_gen"],
)
def test_recovery_fails_closed_on_protocol_mismatch(replies, match):
    recovery = _recovery(_FakeClient(replies))
    with pytest.raises(RuntimeError, match=match):
        recovery._recover(_interrupted())


def _fail(error):
    def raise_error(*args, **kwargs):
        raise error

    return raise_error


@pytest.mark.parametrize(
    "inject",
    [
        lambda r: setattr(r._client, "replies", [Abort("authority stopped")]),
        lambda r: setattr(r._client, "replies", _happy_replies()[:1] + [Abort("stalled")]),
        lambda r: setattr(r._client, "replies", _happy_replies()[:2] + [Abort("stalled")]),
        lambda r: setattr(r._executor.dist.shrink, "side_effect", _fail(RuntimeError("shrink"))),
        lambda r: setattr(
            r._executor.model_engine.moe_load_balancer.recovery_placements,
            "side_effect",
            _fail(RuntimeError("placement")),
        ),
        lambda r: setattr(
            r._executor.dist.install_survivors, "side_effect", _fail(RuntimeError("survivors"))
        ),
        lambda r: setattr(
            r._executor.model_engine.moe_load_balancer.install_recovery_placement,
            "side_effect",
            _fail(RuntimeError("install")),
        ),
    ],
    ids=[
        "abort_at_propose",
        "abort_at_install",
        "abort_at_commit",
        "shrink",
        "prepare",
        "install_survivors",
        "install_placement",
    ],
)
def test_a_failure_at_any_transition_ends_recovery_without_resuming(inject):
    """Fault injection at every transition: the worker raises and resumes nothing."""
    recovery = _recovery(_FakeClient(_happy_replies()))
    recovery._requeue_unscheduled_requests = Mock()
    recovery._peer_failed.set()
    inject(recovery)
    with pytest.raises(RuntimeError):
        recovery._recover(_interrupted())
    assert "watch" not in recovery._executor._events
    recovery._requeue_unscheduled_requests.assert_not_called()
    assert recovery.peer_failed


def test_run_surfaces_a_failed_recovery_after_stopping_the_watch():
    recovery = _recovery(_FakeClient([Abort("authority stopped")]))
    watch = recovery._watch
    recovery._executor.dist.watch_peer_failures.side_effect = lambda callback, abort_word: watch
    with pytest.raises(RecoveryAborted, match="authority stopped"):
        recovery.run(Mock(side_effect=_interrupted()))
    watch.stop.assert_called()


def test_recovery_discards_the_steps_newest_first_after_joining_their_device_work():
    recovery = _recovery(_FakeClient(_happy_replies()))
    calls = []
    older, newer = (
        Mock(name=name, context_requests=[], **{"all_requests.return_value": []})
        for name in ("older", "newer")
    )
    recovery.step_prepared(older)
    recovery.step_prepared(newer)
    recovery._join_device_work = lambda deadline: calls.append("join")
    recovery._executor.discard_interrupted_steps = lambda steps: calls.append(
        ("discard", [step.batch for step in steps])
    )

    recovery._recover(_interrupted())

    assert calls == ["join", ("discard", [newer, older])]
    assert not recovery._uncommitted_steps


def test_steps_commit_oldest_first_and_release_the_newest():
    recovery = _recovery(_FakeClient([]))
    older, newer = Mock(name="older", context_requests=[]), Mock(name="newer", context_requests=[])
    recovery.step_prepared(older)
    recovery.step_prepared(newer)
    with pytest.raises(RuntimeError, match="order"):
        recovery.commit_step(newer, Mock())
    recovery.step_released()
    recovery.commit_step(older, Mock())
    assert not recovery._uncommitted_steps


def test_rank0_logs_each_recovery_phase_once_in_order(monkeypatch):
    logged = []
    monkeypatch.setattr(peer_loss_recovery.logger, "info", logged.append)
    recovery = _recovery(_FakeClient(_happy_replies(generation=1)), rank=0)
    recovery._recover(_interrupted())
    step = Mock(context_requests=[])
    for _ in range(2):
        recovery.step_prepared(step)
        recovery.commit_step(step, Mock())

    phases = [line.split()[2:4] for line in logged if line.startswith("WideEP recovery")]
    assert phases == [
        ["gen=1", f"phase={name}"]
        for name in (
            "detect",
            "stopped",
            "proposed",
            "ready",
            "install",
            "installed",
            "committed",
            "resumed",
            "first_step",
        )
    ]


@pytest.mark.parametrize("fails", [False, True])
def test_run_logs_a_recovery_after_it_ends(monkeypatch, fails):
    # A log write can stall, and every survivor waits for the slowest at each collective.
    events = []
    monkeypatch.setattr(peer_loss_recovery.logger, "warning", lambda line: events.append(line))
    recovery = _recovery(_FakeClient([]))

    def recover(error):
        recovery._log(peer_loss_recovery.logger.warning, "interrupted")
        events.append("sent Stopped")
        if fails:
            raise RuntimeError("not recoverable")

    recovery._recover = recover
    event_loop = Mock(side_effect=[_interrupted(), None])
    if fails:
        with pytest.raises(RuntimeError, match="not recoverable"):
            recovery.run(event_loop)
    else:
        recovery.run(event_loop)

    assert events == ["sent Stopped", "interrupted"]
    recovery._log(peer_loss_recovery.logger.warning, "serving")
    assert events[-1] == "serving"


def test_run_recovers_and_resumes():
    recovery = _recovery(_FakeClient([]))
    error = _interrupted()
    event_loop = Mock(side_effect=[error, None])
    recovered = []
    recovery._recover = recovered.append

    recovery.run(event_loop)

    assert event_loop.call_count == 2
    assert recovered == [error]
    # The hang detector bounds recovery.
    hang_detector = recovery._executor.hang_detector
    hang_detector.__enter__.assert_called_once()
    hang_detector.__enter__.return_value.checkpoint.assert_called_once()
    hang_detector.__exit__.assert_called_once()


def test_run_reports_hang_evidence_only_when_enabled(monkeypatch):
    from tensorrt_llm._torch.pyexecutor import hang_evidence
    from tensorrt_llm.executor.recovery import client as client_module

    started, stopped, closed = [], [], []
    monkeypatch.setattr(hang_evidence, "ProgressRowReader", lambda *args: Mock(read=lambda: None))
    monkeypatch.setattr(
        hang_evidence.HangEvidenceReporter, "start", lambda self: started.append(self)
    )
    monkeypatch.setattr(
        hang_evidence.HangEvidenceReporter, "stop", lambda self: stopped.append(self) or True
    )
    heartbeats = []
    monkeypatch.setattr(
        hang_evidence,
        "start_native_heartbeat",
        lambda rank, interval_s: heartbeats.append(Mock()) or heartbeats[-1],
    )
    monkeypatch.setattr(
        client_module.RecoveryClient,
        "from_env",
        classmethod(lambda cls, rank: Mock(close=lambda: closed.append(rank))),
    )
    recovery = _recovery(_FakeClient([]))
    recovery._executor.model_engine.moe_execution_state.progress_comm.progress_row_offsets.return_value = (
        0,
        64,
        128,
    )
    recovery._executor.model_engine.cuda_graph_runner._capture_allowed = False

    recovery.run(Mock(return_value=None))
    assert started == []

    monkeypatch.setattr(
        peer_loss_recovery,
        "fault_tolerance_config",
        lambda: FaultToleranceConfig(hang_detection=True),
    )
    recovery.run(Mock(return_value=None))
    assert len(started) == 1 and stopped == started and closed == [1]
    # The native heartbeat runs beside the reporter, from the executor's iteration on.
    (heartbeat,) = heartbeats
    heartbeat.publish.assert_called_once()
    heartbeat.stop.assert_called_once()

    # The reporter's device reads must not run beside a capture on the serving thread.
    recovery._executor.model_engine.cuda_graph_runner._capture_allowed = True
    with pytest.raises(RuntimeError, match="captured only before serving"):
        recovery.run(Mock(return_value=None))


def test_only_an_admitted_serving_rank_outside_recovery_exits_alone():
    unadmitted = _recovery(_FakeClient([]), rank=0)
    unadmitted.run(Mock(return_value=None))
    assert not unadmitted.survives_own_loss

    recovery = _recovery(_FakeClient([]))
    # Not before the serving loop watches its peers.
    assert not recovery.survives_own_loss
    seen = []
    recovery.run(Mock(side_effect=lambda: seen.append(recovery.survives_own_loss)))
    assert seen == [True]

    # After a recovery, only while the survivors' slots also cover this rank's loss.
    for covered in (True, False):
        recovery = _recovery(_FakeClient([]))
        recovery._recover = Mock()
        load_balancer = recovery._executor.model_engine.moe_load_balancer
        load_balancer.covers_loss_of.return_value = covered
        loops = iter([_interrupted(), None])

        def event_loop():
            error = next(loops)
            if error is not None:
                raise error
            seen.append(recovery.survives_own_loss)

        seen = []
        recovery.run(event_loop)
        load_balancer.covers_loss_of.assert_called_once_with(frozenset({FAILED_RANK, 1}))
        assert seen == [covered]


@pytest.mark.parametrize(
    "message, killed",
    [
        ("CUDA error: an illegal memory access was encountered", True),
        ("CUDA out of memory", False),
        ("alltoall timeout", False),
        ("a bug every rank hits", False),
    ],
)
def test_crashed_loop_exits_only_this_rank_when_peers_can_recover(monkeypatch, message, killed):
    from tensorrt_llm._torch.pyexecutor import py_executor

    recovery = _recovery(_FakeClient([]))
    recovery.survives_own_loss = True
    executor = _executor()
    executor._peer_loss_recovery = recovery
    recovery.run = Mock(side_effect=RuntimeError(message))
    executor._is_warmup = True
    executor._event_loop_error_delivered = Mock()
    executor.garbage_collection_gen0_threshold = None
    kills = []
    monkeypatch.setattr(py_executor.os, "kill", lambda pid, sig: kills.append(sig))
    monkeypatch.setattr(py_executor, "hard_kill_on_rank_crash", Mock())
    monkeypatch.setattr(py_executor, "start_rank_crash_kill_watchdog", Mock(return_value=None))
    executor._executor_loop_cleanup = Mock()
    executor.event_loop = Mock()
    with pytest.raises(RuntimeError):
        executor._event_loop_wrapper()
    assert kills == ([signal.SIGKILL] if killed else [])


def test_run_does_not_recover_during_kv_cache_estimation():
    recovery = _recovery(_FakeClient([]))
    recovery._executor.gather_all_responses = True
    with pytest.raises(HostCollectiveInterrupted):
        recovery.run(Mock(side_effect=_interrupted()))
    recovery._executor.dist.watch_peer_failures.assert_not_called()


@pytest.mark.parametrize("peer_failed", [False, True])
def test_commit_step_waits_then_accepts_or_discards(peer_failed):
    recovery = _recovery(_FakeClient([]))
    batch = Mock(name="batch", context_requests=[])
    recovery.step_prepared(batch)
    sample_state = Mock()
    if peer_failed:
        recovery._peer_failed.set()
        with pytest.raises(HostCollectiveInterrupted):
            recovery.commit_step(batch, sample_state)
    else:
        recovery.commit_step(batch, sample_state)
    sample_state.sampler_event.synchronize.assert_called_once()
    assert [step.batch for step in recovery._uncommitted_steps] == ([batch] if peer_failed else [])


def test_commit_step_waits_on_the_sampler_event_or_without_one_the_stream(monkeypatch):
    """A sampled step waits on its sampler event only: under the overlap scheduler the stream holds the next
    step's forward. A step that failed before sampling has no event, so its commit waits for the stream: its
    MoE collectives must finish before the abort record is read and its context blocks are published."""
    stream = Mock()
    monkeypatch.setattr(torch.cuda, "current_stream", lambda *args: stream)
    recovery = _recovery(_FakeClient([]))
    sampled, failed = (
        Mock(name="sampled", context_requests=[]),
        Mock(name="failed", context_requests=[]),
    )
    recovery.step_prepared(sampled)
    recovery.step_prepared(failed)
    sample_state = Mock()

    recovery.commit_step(sampled, sample_state)
    sample_state.sampler_event.synchronize.assert_called_once_with()
    stream.synchronize.assert_not_called()

    aborted = recovery._executor.model_engine.moe_execution_state.aborted
    aborted.side_effect = lambda: stream.synchronize.called is False
    recovery.commit_step(failed, None)
    stream.synchronize.assert_called_once_with()
    assert not recovery._uncommitted_steps


def test_commit_step_discards_a_step_whose_peer_failed_while_it_ran():
    """The flag is read after waiting for the step's device work, not before."""
    recovery = _recovery(_FakeClient([]))
    batch = Mock(name="batch", context_requests=[])
    recovery.step_prepared(batch)
    sample_state = Mock()
    sample_state.sampler_event.synchronize.side_effect = recovery._peer_failed.set
    with pytest.raises(HostCollectiveInterrupted):
        recovery.commit_step(batch, sample_state)


def test_commit_step_discards_a_step_whose_collective_was_released_before_the_failure_is_seen():
    """A released collective is recorded on the device, so the step is discarded before the watch fires."""
    recovery = _recovery(_FakeClient([]))
    batch = Mock(name="batch", context_requests=[])
    recovery.step_prepared(batch)
    state = recovery._executor.model_engine.moe_execution_state
    sample_state = Mock()
    sample_state.sampler_event.synchronize.side_effect = lambda: setattr(
        state.aborted, "return_value", True
    )
    with pytest.raises(HostCollectiveInterrupted):
        recovery.commit_step(batch, sample_state)
    assert not recovery.peer_failed
    assert [step.batch for step in recovery._uncommitted_steps] == [batch]


def test_recovery_aborts_collectives_and_delivers_only_after_commit():
    client = _FakeClient(_happy_replies())
    recovery = _recovery(client)
    executor = recovery._executor
    events = executor._events
    executor.model_engine.moe_execution_state.request_abort.side_effect = lambda: events.append(
        "abort"
    )
    client_wait = client.wait
    client.wait = lambda message_type, deadline, **kwargs: (
        events.extend(
            t.__name__
            for t in (message_type if isinstance(message_type, tuple) else (message_type,))
        )
        or client_wait(message_type, deadline, **kwargs)
    )
    executor._enqueue_responses = lambda responses: events.append("deliver")

    recovery._recover(_interrupted())

    assert events.index("abort") < events.index("shrink")
    assert events.index("Install") < events.index("survivors")
    assert events.index("Commit") < events.index("deliver")
    assert not recovery._gathers


@pytest.mark.parametrize("rank,events", [(0, ["revoke", "wake"]), (1, ["revoke"])])
def test_peer_failure_releases_host_waits(rank, events):
    recovery = _recovery(_FakeClient([]), rank=rank)
    recovery._on_peer_failure()
    assert recovery.peer_failed
    assert recovery._executor._events == events


def test_watch_aborts_the_moe_collectives_natively():
    recovery = _recovery(_FakeClient([]))
    abort = recovery._executor.model_engine.moe_execution_state.abort
    abort.data_ptr.return_value = 4096
    recovery._watch_peer_failures()
    recovery._executor.dist.watch_peer_failures.assert_called_once_with(
        recovery._on_peer_failure, 4096
    )


def test_a_failure_seen_only_by_the_watch_counts():
    recovery = _recovery(_FakeClient([]))
    recovery._watch = Mock(failed=False)
    assert not recovery.peer_failed
    recovery._watch.failed = True
    assert recovery.peer_failed


def test_a_peer_failure_before_serving_ends_the_job(monkeypatch):
    """Nothing recovers a loss during startup, and under ULFM the runtime no longer ends the job."""
    import tensorrt_llm._utils
    from tensorrt_llm._torch.pyexecutor import hang_detector

    monkeypatch.setattr(peer_loss_recovery, "peer_loss_recovery_enabled", lambda: True)
    monkeypatch.setattr(peer_loss_recovery, "_startup_watch", None)
    monkeypatch.setattr(peer_loss_recovery, "atexit", Mock())
    # The executor thread's communicator, not the watch thread's default.
    executor_comm = Mock()
    monkeypatch.setattr(tensorrt_llm._utils, "mpi_comm", lambda: executor_comm)
    kills = []
    monkeypatch.setattr(hang_detector, "propagate_hard_kill", lambda **kwargs: kills.append(kwargs))
    dist = Mock(spec=MPIDist)
    peer_loss_recovery.watch_startup_peer_failures(dist)
    # Once per process: the final executor's creation does not watch again.
    peer_loss_recovery.watch_startup_peer_failures(dist)
    dist.enable_peer_loss_recovery.assert_called_once_with()
    peer_loss_recovery.atexit.register.assert_called_once_with(
        peer_loss_recovery.stop_startup_watch
    )
    ((on_failure,), _) = dist.watch_peer_failures.call_args
    assert dist.watch_peer_failures.call_count == 1 and kills == []
    monkeypatch.setattr(tensorrt_llm._utils, "mpi_comm", lambda: Mock())
    on_failure()
    assert kills == [{"diagnostics": False, "communicator": executor_comm}]


def test_a_failed_executor_creation_stops_the_startup_watch(monkeypatch):
    from tensorrt_llm._torch.pyexecutor import py_executor_creator

    startup = Mock()
    monkeypatch.setattr(peer_loss_recovery, "_startup_watch", startup)
    monkeypatch.setattr(
        py_executor_creator, "_create_py_executor_impl", Mock(side_effect=RuntimeError("load"))
    )
    with pytest.raises(RuntimeError, match="load"):
        py_executor_creator.create_py_executor(Mock())
    startup.stop.assert_called_once_with()
    assert peer_loss_recovery._startup_watch is None


def test_serving_takes_over_from_the_startup_watch(monkeypatch):
    startup = Mock()
    monkeypatch.setattr(peer_loss_recovery, "_startup_watch", startup)
    recovery = _recovery(_FakeClient([]))
    recovery._executor.dist.watch_peer_failures.side_effect = (
        lambda callback, abort_word: startup.stop.assert_not_called() or Mock(failed=False)
    )
    recovery.run(Mock())
    startup.stop.assert_called_once_with()
    assert peer_loss_recovery._startup_watch is None


def test_discard_returns_step_resources_and_padding_request():
    executor = _executor()
    executor.kv_cache_manager = Mock()
    executor.draft_kv_cache_manager = Mock()
    executor.inflight_req_ids = Mock()
    executor._pending_adp_dummy_request = None
    step = PreparedStep(Mock(name="batch"), {7: 64})
    request = Mock(is_attention_dp_dummy=False)
    dummy = Mock(is_attention_dp_dummy=True, py_request_id=99)
    executor.active_requests.extend([request, dummy])

    executor.discard_interrupted_steps([step])

    executor.kv_cache_manager.revert_prepare_resources.assert_called_once_with(step.batch, {7: 64})
    executor.draft_kv_cache_manager.revert_prepare_resources.assert_called_once_with(step.batch, {})
    executor._terminate_request.assert_called_once_with(dummy)
    assert executor.active_requests == [request]
    assert executor.previous_batch is None and request.py_batch_idx is None


def test_discard_undoes_each_step_s_request_changes_newest_first():
    """Under the overlap scheduler two steps can be uncommitted; each one's changes are undone in reverse."""
    executor = _executor()
    executor.inflight_req_ids = Mock()
    executor._pending_adp_dummy_request = None
    request = Mock(is_attention_dp_dummy=False, context_current_position=0, state="CONTEXT_INIT")
    older, newer = PreparedStep(Mock(), {}), PreparedStep(Mock(), {})
    for step, position, state in (
        (older, 64, "CONTEXT_INIT"),
        (newer, 128, "GENERATION_IN_PROGRESS"),
    ):
        executor._step_changes = step.changes
        executor._note_step_change(request)
        request.context_current_position, request.state = position, state

    executor.discard_interrupted_steps([newer, older])

    assert (request.context_current_position, request.state) == (0, "CONTEXT_INIT")
    assert executor._step_changes is None


def test_discard_pauses_a_recompute_victim_after_restoring_its_step():
    """A request of the uncommitted step can be the next step's recompute victim, whose KV cache is freed."""
    executor = _executor()
    executor.inflight_req_ids = Mock()
    executor._pending_adp_dummy_request = None
    request = Mock(is_attention_dp_dummy=False, context_current_position=64, state="CONTEXT_INIT")
    step = PreparedStep(Mock(), {})
    executor._step_changes = step.changes
    executor._note_step_change(request)
    request.context_current_position = 96
    executor._recompute_unpaused = Mock(name="next_batch")
    paused_at = []
    executor._pause_recompute_paused_requests = lambda batch: paused_at.append(
        request.context_current_position
    )

    executor.discard_interrupted_steps([step])

    assert paused_at == [64]
    assert executor._recompute_unpaused is None


def test_discard_restores_the_draft_tokens_sampling_set():
    """Sampling sets a finishing context request's draft tokens before its step is accepted."""
    recovery = _recovery(_FakeClient([]))
    executor = recovery._executor
    executor.inflight_req_ids = Mock()
    executor._pending_adp_dummy_request = None
    drafts = []
    request = Mock(
        is_attention_dp_dummy=False,
        context_current_position=64,
        state="CONTEXT_INIT",
        py_draft_tokens=drafts,
    )
    executor._step_changes = recovery.step_prepared(SimpleNamespace(context_requests=[request]))
    request.py_draft_tokens = [1, 1]
    executor._note_step_change(request)
    request.context_current_position, request.state = 96, "GENERATION_IN_PROGRESS"

    PyExecutor.discard_interrupted_steps(executor, reversed(recovery._uncommitted_steps))

    assert request.py_draft_tokens is drafts
    assert (request.context_current_position, request.state) == (64, "CONTEXT_INIT")


def test_discard_restores_the_exclusion_of_the_last_generation_logits():
    """A step that completes its request next iteration stops excluding the last generation logits; a discard
    restores the exclusion, and the rerun decides again."""
    executor = _executor()
    executor.inflight_req_ids = Mock()
    executor._pending_adp_dummy_request = None
    request = Mock(
        is_attention_dp_dummy=False, context_current_position=64, state="GENERATION_IN_PROGRESS"
    )
    request.py_result.exclude_last_generation_logits = True
    request.set_exclude_last_generation_logits.side_effect = lambda exclude: setattr(
        request.py_result, "exclude_last_generation_logits", exclude
    )
    request.will_complete_next_iteration.return_value = True
    step = PreparedStep(Mock(), {})
    executor._step_changes = step.changes
    executor._update_generation_requests_that_will_complete_next_iteration([request])
    assert request.py_result.exclude_last_generation_logits is False

    executor.discard_interrupted_steps([step])

    assert request.py_result.exclude_last_generation_logits is True
    assert request.state == "GENERATION_IN_PROGRESS"


def test_discard_drops_the_logits_and_additional_outputs_its_steps_appended(monkeypatch):
    """Sampling appends a step's logits and additional outputs before the step is accepted; its rerun appends
    them again."""
    monkeypatch.setattr(llm_request, "prefer_pinned", lambda: False)
    recovery = _recovery(_FakeClient([]))
    executor = recovery._executor
    executor.inflight_req_ids = Mock()
    executor._pending_adp_dummy_request = None
    result = PyResult(
        prompt_len=8,
        max_new_tokens=4,
        use_device_memory=False,
        return_context_logits=True,
        return_generation_logits=True,
        additional_outputs=["hidden"],
    )
    request = Mock(
        is_attention_dp_dummy=False,
        py_result=result,
        py_return_context_logits=True,
        py_return_generation_logits=True,
        py_additional_outputs=["hidden"],
        sampling_config=Mock(
            repetition_penalty=None, presence_penalty=None, frequency_penalty=None
        ),
        py_multimodal_data=None,
        py_logits_post_processors=None,
    )
    recovery.validate_request(request)

    def append(value):
        result.append_context_logits(torch.full((2, 8), value))
        result.append_generation_logits(torch.full((1, 1, 8), value))
        result.transfer_remaining_device_logits()
        result.append_additional_context_outputs("hidden", torch.full((2, 3), value))
        result.append_additional_generation_outputs("hidden", torch.full((1, 1, 3), value))

    append(1.0)
    # Two uncommitted steps under the overlap scheduler.
    for value in (2.0, 3.0):
        batch = SimpleNamespace(context_requests=[], all_requests=lambda: [request])
        recovery.step_prepared(batch)
        recovery.step_outputs_appending(batch)
        append(value)

    PyExecutor.discard_interrupted_steps(executor, reversed(recovery._uncommitted_steps))

    assert torch.equal(result.context_logits, torch.ones(2, 8))
    assert torch.equal(result.generation_logits, torch.ones(1, 1, 8))
    assert torch.equal(result.additional_context_outputs["hidden"], torch.ones(2, 3))
    assert torch.equal(result.additional_generation_outputs["hidden"], torch.ones(1, 1, 3))
    append(2.0)
    assert result.context_logits.shape == (4, 8)
    assert torch.equal(result.generation_logits[0, :, 0], torch.tensor([1.0, 2.0]))


def test_discard_before_prepare_releases_the_tentative_padding_request():
    """A loss in the can-queue exchange precedes prepare_resources."""
    executor = _executor()
    executor.kv_cache_manager = Mock()
    executor.inflight_req_ids = Mock()
    executor.resource_manager = Mock()
    executor.resource_manager.get_resource_manager.return_value = None
    request = Mock(is_attention_dp_dummy=False)
    dummy = Mock(is_attention_dp_dummy=True, py_request_id=99, py_seq_slot=None)
    executor.active_requests.extend([request, dummy])
    executor._pending_adp_dummy_request = dummy

    executor.discard_interrupted_steps([])

    executor.kv_cache_manager.free_resources.assert_called_once_with(dummy)
    executor.kv_cache_manager.revert_prepare_resources.assert_not_called()
    executor._terminate_request.assert_not_called()
    assert executor.active_requests == [request]


def test_publish_defers_terminations_when_the_gather_is_interrupted():
    recovery = _recovery(_FakeClient([]))
    executor = recovery._executor
    executor._enqueue_responses = Mock(side_effect=_interrupted())
    finished = Mock(name="finished")
    with pytest.raises(HostCollectiveInterrupted):
        executor.publish_responses([], [finished])
    executor._terminate_request.assert_not_called()
    assert recovery._undelivered_terminations == [finished]


def test_step_errors_after_a_peer_failure_become_interruptions():
    recovery = _recovery(_FakeClient([]))
    executor = recovery._executor
    executor._raise_if_peer_failed("forward", RuntimeError("garbage"))
    with pytest.raises(HostCollectiveInterrupted):
        executor._raise_if_peer_failed("forward", _interrupted())
    recovery._peer_failed.set()
    with pytest.raises(HostCollectiveInterrupted):
        executor._raise_if_peer_failed("forward", RuntimeError("garbage"))


def _sent_responses(recovery):
    sent = []
    recovery._executor._enqueue_responses = lambda responses: sent.append(list(responses))
    return sent


def test_an_unrecoverable_loss_fails_before_waiting_on_the_device():
    recovery = _recovery(_FakeClient(_happy_replies()), failed=frozenset({0}))
    with pytest.raises(RuntimeError, match=r"ranks \[0\] is not recoverable"):
        recovery._recover(_interrupted(frozenset({0})))
    recovery._join_device_work.assert_not_called()


def test_joining_device_work_reports_a_device_error(monkeypatch):
    recovery = _recovery(_FakeClient([]))

    def fail(device):
        raise RuntimeError("CUDA error: an illegal memory access was encountered")

    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.cuda, "synchronize", fail)
    with pytest.raises(RuntimeError, match="illegal memory access"):
        PeerLossRecovery._join_device_work(recovery, time.monotonic() + 5)


def test_joining_device_work_is_bounded(monkeypatch):
    recovery = _recovery(_FakeClient([]))
    release = threading.Event()
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda device: release.wait())
    try:
        with pytest.raises(RecoveryTimeout, match="device work"):
            PeerLossRecovery._join_device_work(recovery, time.monotonic() + 0.05)
    finally:
        release.set()
        for thread in threading.enumerate():
            if thread.name == "peer_loss_device_join":
                thread.join(timeout=5)


def _supported_executor():
    load_balancer = Mock()
    load_balancer.is_static_routing.return_value = True
    load_balancer.covers_loss_of.return_value = True
    kv_cache_manager = KVCacheManagerV2.__new__(KVCacheManagerV2)
    return SimpleNamespace(
        model_engine=SimpleNamespace(
            moe_load_balancer=load_balancer, moe_execution_state=Mock(), spec_config=None
        ),
        kv_cache_manager=kv_cache_manager,
        draft_kv_cache_manager=None,
        resource_manager=Mock(get_resource_manager=Mock(return_value=None)),
        dist=Mock(
            spec=MPIDist,
            rank=0,
            world_size=4,
            mapping=SimpleNamespace(enable_lm_head_tp_in_adp=False, moe_tp_size=1, gpus_per_node=2),
            allgather=Mock(return_value=["a", "a", "b", "b"]),
        ),
        adp_router=DefaultADPRouter.__new__(DefaultADPRouter),
        drafter=None,
        draft_model_engine=None,
        guided_decoder=None,
        max_beam_width=1,
        kv_cache_transceiver=None,
        kv_connector_manager=None,
        disable_overlap_scheduler=True,
        attention_dp_enable_balance=False,
        enable_kv_pool_rebalance=False,
        enable_kv_cache_events=False,
        hang_detector=SimpleNamespace(timeout=300),
        llm_args=SimpleNamespace(sleep_config=None),
        sampler=Mock(is_generation_model=Mock(return_value=True)),
    )


@pytest.mark.parametrize(
    "unsupported, requirement",
    [
        (lambda ex: setattr(ex, "guided_decoder", Mock()), "no guided decoding"),
        (lambda ex: setattr(ex, "max_beam_width", 2), "no beam search"),
        (lambda ex: setattr(ex, "kv_connector_manager", Mock()), "no KV connector"),
        (lambda ex: setattr(ex, "enable_kv_pool_rebalance", True), "no KV pool rebalance"),
        (lambda ex: setattr(ex, "enable_kv_cache_events", True), "no KV cache events"),
        (
            lambda ex: setattr(ex, "kv_cache_manager", Mock()),
            "KV Cache Manager V2",
        ),
        (lambda ex: setattr(ex, "draft_kv_cache_manager", Mock()), "KV Cache Manager V2"),
        (
            lambda ex: setattr(ex.resource_manager.get_resource_manager, "return_value", Mock()),
            "no PEFT",
        ),
        (
            lambda ex: setattr(
                ex.resource_manager.get_resource_manager,
                "side_effect",
                lambda kind: Mock() if kind.name == "CROSS_KV_CACHE_MANAGER" else None,
            ),
            "no encoder-decoder cross KV cache",
        ),
        (
            lambda ex: setattr(ex.dist.mapping, "enable_lm_head_tp_in_adp", True),
            "no LM head TP",
        ),
        (lambda ex: setattr(ex.dist.mapping, "moe_tp_size", 2), "expert TP 1"),
        (lambda ex: setattr(ex.model_engine, "moe_load_balancer", None), "an EPLB load balancer"),
        (lambda ex: setattr(ex.hang_detector, "timeout", 60), r"hang_detection_timeout \(60 s\)"),
        (lambda ex: setattr(ex, "dist", Mock(mapping=ex.dist.mapping)), "MPI executor"),
        (
            lambda ex: ex.sampler.is_generation_model.configure_mock(return_value=False),
            "a generation sampler",
        ),
    ],
)
def test_create_rejects_configurations_a_discarded_step_cannot_rerun(
    monkeypatch, unsupported, requirement
):
    monkeypatch.setattr(peer_loss_recovery, "peer_loss_recovery_enabled", lambda: True)
    executor = _supported_executor()
    unsupported(executor)
    with pytest.raises(ValueError, match=requirement):
        PeerLossRecovery.create(executor)
    executor.dist.enable_peer_loss_recovery.assert_not_called()


def test_create_requires_the_hang_detector_to_outlast_a_shrink_waiting_for_a_fence(monkeypatch):
    config = FaultToleranceConfig(hang_detection=True)
    monkeypatch.setattr(peer_loss_recovery, "peer_loss_recovery_enabled", lambda: True)
    monkeypatch.setattr(peer_loss_recovery, "fault_tolerance_config", lambda: config)
    executor = _supported_executor()
    # Above the hang's resolution plus a recovery, but not above the shrink plus a recovery's two waits.
    executor.hang_detector.timeout = timeout = config.shrink_timeout_s() + config.recovery_timeout_s
    minimum = config.shrink_timeout_s() + 2 * config.recovery_timeout_s
    with pytest.raises(
        ValueError, match=rf"hang_detection_timeout \({timeout} s\) above {minimum:g} s"
    ):
        PeerLossRecovery.create(executor)


@pytest.mark.parametrize("mode,rank,warns", [("1", 0, True), ("1", 1, False), (None, 0, False)])
def test_create_warns_on_rank0_when_only_detection_is_on(monkeypatch, mode, rank, warns):
    monkeypatch.setattr(peer_loss_recovery, "peer_loss_recovery_enabled", lambda: False)
    if mode is None:
        monkeypatch.delenv("TLLM_FAULT_TOLERANCE_MODE", raising=False)
    else:
        monkeypatch.setenv("TLLM_FAULT_TOLERANCE_MODE", mode)
    warning = Mock()
    monkeypatch.setattr(peer_loss_recovery.logger, "warning", warning)
    executor = _supported_executor()
    executor.dist.rank = rank
    assert type(PeerLossRecovery.create(executor)) is RecoveryHook
    assert warning.called == warns
    if warns:
        assert "TLLM_FAULT_TOLERANCE_AUTHORITY" in warning.call_args.args[0]


def test_create_returns_a_pass_through_hook_when_disabled(monkeypatch):
    monkeypatch.setattr(peer_loss_recovery, "peer_loss_recovery_enabled", lambda: False)
    executor = _supported_executor()
    recovery = PeerLossRecovery.create(executor)
    assert type(recovery) is RecoveryHook
    assert not (recovery.peer_failed or recovery.exits_alone(RuntimeError("illegal memory access")))
    responses = [(1, Mock())]
    assert recovery.gathering_responses(responses) is responses
    event_loop = Mock()
    recovery.run(event_loop)
    event_loop.assert_called_once_with()
    executor.dist.enable_peer_loss_recovery.assert_not_called()


def test_failure_domains_are_every_rank_then_every_host_but_rank_0_s():
    hosts = ["a"] * 4 + ["b"] * 4
    assert peer_loss_recovery.failure_domains(hosts, "rank") == [(r,) for r in range(1, 8)]
    assert peer_loss_recovery.failure_domains(hosts, "node") == [
        *((r,) for r in range(1, 8)),
        (4, 5, 6, 7),
    ]
    # Ranks mapped round-robin over the hosts: a node is the ranks on one host, not a block of ranks.
    assert peer_loss_recovery.failure_domains(["a", "b", "a", "b"], "node")[-1] == (1, 3)
    assert peer_loss_recovery.failure_domains(["a", "b", "c", "d"], "node") == [
        (1,),
        (2,),
        (3,),
    ]


@pytest.mark.parametrize("failure_domain, refused", [("rank", False), ("node", True)])
def test_create_refuses_a_failure_domain_the_slots_cannot_cover(
    monkeypatch, failure_domain, refused
):
    monkeypatch.setattr(peer_loss_recovery, "peer_loss_recovery_enabled", lambda: True)
    monkeypatch.setattr(
        peer_loss_recovery,
        "fault_tolerance_config",
        lambda: FaultToleranceConfig(failure_domain=failure_domain),
    )
    monkeypatch.setattr(
        peer_loss_recovery, "_register", lambda rank, recoverable: (Mock(), frozenset())
    )
    executor = _supported_executor()
    executor.model_engine.moe_load_balancer.covers_loss_of.side_effect = lambda dead: len(dead) == 1
    if refused:
        with pytest.raises(ValueError, match=r"failure_domain=node .* \(2, 3\)"):
            PeerLossRecovery.create(executor)
    else:
        assert isinstance(PeerLossRecovery.create(executor), PeerLossRecovery)


def test_create_accepts_the_dsa_kv_cache_manager(monkeypatch):
    from tensorrt_llm._torch.attention.backends.sparse.dsa import DSACacheManagerV2

    monkeypatch.setattr(peer_loss_recovery, "peer_loss_recovery_enabled", lambda: True)
    monkeypatch.setattr(
        peer_loss_recovery, "_register", lambda rank, recoverable: (Mock(), frozenset())
    )
    executor = _supported_executor()
    executor.kv_cache_manager = DSACacheManagerV2.__new__(DSACacheManagerV2)
    assert isinstance(PeerLossRecovery.create(executor), PeerLossRecovery)


def test_create_accepts_kv_block_reuse(monkeypatch):
    # A step commits its context blocks for reuse only once accepted, so a discarded step published none.
    monkeypatch.setattr(peer_loss_recovery, "peer_loss_recovery_enabled", lambda: True)
    monkeypatch.setattr(
        peer_loss_recovery, "_register", lambda rank, recoverable: (Mock(), frozenset())
    )
    executor = _supported_executor()
    executor.kv_cache_manager.enable_block_reuse = True
    assert isinstance(PeerLossRecovery.create(executor), PeerLossRecovery)


def _spec_config(mode: "SpecMode", **options) -> SimpleNamespace:
    return SimpleNamespace(spec_dec_mode=mode, **options)


@pytest.mark.parametrize(
    "configure, warns",
    [
        (lambda ex: None, False),
        (
            lambda ex: setattr(
                ex.model_engine, "spec_config", _spec_config(SpecMode.MTP_EAGLE_ONE_MODEL)
            ),
            False,
        ),
        (lambda ex: setattr(ex.model_engine, "spec_config", _spec_config(SpecMode.MTP)), True),
        (
            lambda ex: setattr(
                ex.model_engine, "spec_config", _spec_config(SpecMode.EAGLE3_ONE_MODEL)
            ),
            True,
        ),
        (
            lambda ex: setattr(
                ex.model_engine,
                "spec_config",
                _spec_config(SpecMode.MTP_EAGLE_ONE_MODEL, use_dynamic_tree=True),
            ),
            True,
        ),
        (
            lambda ex: setattr(
                ex.model_engine,
                "spec_config",
                _spec_config(
                    SpecMode.MTP_EAGLE_ONE_MODEL, use_relaxed_acceptance_for_thinking=True
                ),
            ),
            True,
        ),
        (
            lambda ex: setattr(
                ex.model_engine,
                "spec_config",
                _spec_config(SpecMode.MTP_EAGLE_ONE_MODEL, sa_config=Mock()),
            ),
            True,
        ),
        (lambda ex: setattr(ex.model_engine, "spec_config", _spec_config(SpecMode.NGRAM)), True),
        (lambda ex: setattr(ex, "drafter", Mock()), True),
    ],
    ids=[
        "none",
        "one_model_mtp",
        "mtp_vanilla",
        "eagle3_one_model",
        "dynamic_tree",
        "relaxed_acceptance",
        "suffix_automaton",
        "ngram",
        "drafter",
    ],
)
def test_create_accepts_any_speculative_decoding_and_warns_when_untested(
    monkeypatch, configure, warns
):
    monkeypatch.setattr(peer_loss_recovery, "peer_loss_recovery_enabled", lambda: True)
    monkeypatch.setattr(
        peer_loss_recovery, "_register", lambda rank, recoverable: (Mock(), frozenset())
    )
    warning = Mock()
    monkeypatch.setattr(peer_loss_recovery.logger, "warning", warning)
    executor = _supported_executor()
    configure(executor)
    assert isinstance(PeerLossRecovery.create(executor), PeerLossRecovery)
    executor.dist.enable_peer_loss_recovery.assert_called_once()
    if warns:
        warning.assert_called_once()
        assert "in-graph draft state" in warning.call_args.args[0]
    else:
        warning.assert_not_called()


def test_create_registers_the_domains_whose_loss_the_placement_covers(monkeypatch):
    monkeypatch.setattr(peer_loss_recovery, "peer_loss_recovery_enabled", lambda: True)
    registered = []
    monkeypatch.setattr(
        peer_loss_recovery,
        "_register",
        lambda rank, recoverable: registered.append((rank, recoverable)) or (Mock(), frozenset()),
    )
    monkeypatch.setattr(
        peer_loss_recovery,
        "fault_tolerance_config",
        lambda: FaultToleranceConfig(failure_domain="node"),
    )
    executor = _supported_executor()
    assert isinstance(PeerLossRecovery.create(executor), PeerLossRecovery)
    assert registered == [(0, ((1,), (2,), (3,), (2, 3)))]
    executor.dist.enable_peer_loss_recovery.assert_called_once()


def test_install_recovery_placement_reroutes_before_masking():
    events = []
    load_balancer = Mock()
    load_balancer.install_recovery_placement.side_effect = lambda dead, placements: events.append(
        ("routed", dead)
    )
    health = EPGroupHealth(WORLD_SIZE)
    state = Mock(health=health)
    state.install_mask.side_effect = lambda: events.append(("mask", health.get_mask_words()))
    install = functools.partial(peer_loss_recovery.install_recovery_placement, placements=[])

    install(state, load_balancer, {FAILED_RANK}, own_rank=0)

    assert health.get_failed_ranks() == frozenset({FAILED_RANK})
    # The kernels skip routes to a masked rank, so routing must exclude it first.
    assert events == [("routed", [FAILED_RANK]), ("mask", (0b1011, 0, 0, 0))]
    state.rearm.assert_called_once()

    # Invalid requests are refused before anything changes.
    events.clear()
    with pytest.raises(RuntimeError, match="already failed"):
        install(state, load_balancer, {FAILED_RANK}, own_rank=0)
    with pytest.raises(RuntimeError, match="this rank"):
        install(state, load_balancer, {0, 3}, own_rank=0)
    assert events == [] and health.get_failed_ranks() == frozenset({FAILED_RANK})

    # A loss after a recovery installs every rank lost so far.
    install(state, load_balancer, {FAILED_RANK, 3}, own_rank=0)
    assert health.get_failed_ranks() == frozenset({FAILED_RANK, 3})
    assert events == [("routed", [FAILED_RANK, 3]), ("mask", (0b0011, 0, 0, 0))]


def test_online_eplb_starts_from_rank_0s_placement_sent_among_the_survivors():
    """A placement can exceed an authority message, so rank 0 broadcasts it over the survivor communicator."""
    import numpy as np

    recovery = _recovery(_FakeClient(_happy_replies(generation=1)))
    load_balancer = recovery._executor.model_engine.moe_load_balancer
    load_balancer.is_static_routing.return_value = False
    load_balancer.copy_nodes = None
    load_balancer.quiesce.return_value = True
    own, rank0 = [np.zeros((WORLD_SIZE, 2))], [np.ones((WORLD_SIZE, 2))]
    load_balancer.current_placements.return_value = own
    broadcast = recovery._executor.dist.broadcast_among_survivors = Mock(return_value=rank0)

    recovery._recover(_interrupted())

    broadcast.assert_called_once_with(None, root=0)
    load_balancer.recovery_placements.assert_called_once_with(frozenset({FAILED_RANK}), rank0)
    # Online EPLB restores from its full host copies: nothing to stage.
    load_balancer.stage_restored_slots.assert_not_called()
    load_balancer.placement_digest.assert_called_once_with(
        frozenset({FAILED_RANK}), ["placement"], None
    )


@pytest.mark.parametrize("online", [True, False], ids=["online", "static"])
def test_online_eplb_only_gathers_statistics_for_a_while_after_a_recovery(online):
    """The recovery clears the load factors; online EPLB's next placements wait for warm-up iterations of them."""
    import numpy as np

    recovery = _recovery(_FakeClient(_happy_replies(generation=1)))
    load_balancer = recovery._executor.model_engine.moe_load_balancer
    load_balancer.is_static_routing.return_value = not online
    load_balancer.copy_nodes = None
    load_balancer.quiesce.return_value = True
    load_balancer.current_placements.return_value = [np.zeros((WORLD_SIZE, 2))]
    recovery._executor.dist.broadcast_among_survivors = Mock(
        return_value=[np.ones((WORLD_SIZE, 2))]
    )

    recovery._recover(_interrupted())

    if online:
        load_balancer.reset_update_schedule.assert_called_once_with()
        load_balancer.set_warm_up_iter_count.assert_called_once_with(
            peer_loss_recovery._EPLB_WARM_UP_ITERS_AFTER_RECOVERY
        )
    else:
        load_balancer.reset_update_schedule.assert_not_called()
        load_balancer.set_warm_up_iter_count.assert_not_called()


@pytest.mark.parametrize("covered", [True, False])
def test_online_eplb_decides_recoverability_from_rank_0s_placement(covered):
    """Under online EPLB a survivor's own placement may not cover a loss that rank 0's does (or the reverse):
    every survivor decides from rank 0's placement, so they all recover or all fail closed."""
    import numpy as np

    lost = frozenset({FAILED_RANK, 3})
    client = _FakeClient(_happy_replies(failed=tuple(sorted(lost)), survivors=(0, 1)))
    recovery = _recovery(client, failed=lost)
    load_balancer = recovery._executor.model_engine.moe_load_balancer
    load_balancer.is_static_routing.return_value = False
    load_balancer.copy_nodes = None
    load_balancer.quiesce.return_value = True
    load_balancer.covers_loss_of.return_value = not covered
    rank0 = [np.ones((WORLD_SIZE, 2))]
    load_balancer.current_placements.return_value = [np.zeros((WORLD_SIZE, 2))]
    recovery._executor.dist.broadcast_among_survivors = Mock(return_value=rank0)
    if not covered:
        load_balancer.recovery_placements.side_effect = ValueError("No surviving replica")

    if covered:
        recovery._recover(_interrupted(lost))
        load_balancer.install_recovery_placement.assert_called_once_with(
            sorted(lost), ["placement"]
        )
    else:
        with pytest.raises(RuntimeError, match="not recoverable"):
            recovery._recover(_interrupted(lost))
    load_balancer.covers_loss_of.assert_not_called()
    load_balancer.recovery_placements.assert_called_once_with(lost, rank0)


def test_a_survivor_lost_while_rank_0_sends_its_placement_interrupts_the_round(monkeypatch):
    from mpi4py import MPI

    recovery = _recovery(_FakeClient([]))
    load_balancer = recovery._executor.model_engine.moe_load_balancer
    recovery._executor.dist.broadcast_among_survivors = Mock(
        side_effect=MPI.Exception(MPI.ERR_OTHER)
    )
    monkeypatch.setattr(peer_loss_recovery, "is_mpi_peer_failure", lambda error: True)
    with pytest.raises(RecoveryInterrupted):
        recovery._rank0_placements(load_balancer)
    monkeypatch.setattr(peer_loss_recovery, "is_mpi_peer_failure", lambda error: False)
    with pytest.raises(MPI.Exception):
        recovery._rank0_placements(load_balancer)


def test_ring_host_copies_stage_the_restore_after_propose_and_digest_the_copy_plan(monkeypatch):
    """The copies run once the failed ranks are fenced (Propose) and before Ready, whose digest covers them."""
    import numpy as np

    client = _FakeClient(_happy_replies(generation=1))
    recovery = _recovery(client)
    engine = recovery._executor.model_engine
    load_balancer = engine.moe_load_balancer
    plan = np.zeros((1, 10), dtype=np.int64)
    load_balancer.restore_copy_plan.return_value = plan
    workspace = engine.moe_execution_state.staging_workspace.return_value
    sent_before_staging = []

    def stage_restored_slots(copy_plan, staging_workspace, barrier):
        sent_before_staging.append(list(client.sent))
        barrier()

    load_balancer.stage_restored_slots.side_effect = stage_restored_slots

    recovery._recover(_interrupted())

    failed = frozenset({FAILED_RANK})
    load_balancer.restore_copy_plan.assert_called_once_with(
        failed, ["placement"], workspace.size.return_value
    )
    workspace.size.assert_called_with(1)
    load_balancer.stage_restored_slots.assert_called_once_with(plan, workspace, ANY)
    # The staging barriers are the survivors'.
    recovery._executor.dist.barrier_among_survivors.assert_called_once_with()
    assert sent_before_staging == [[Stopped(rank=1, failed_ranks=(FAILED_RANK,))]]
    load_balancer.placement_digest.assert_called_once_with(failed, ["placement"], plan)


def test_a_survivor_lost_while_restored_experts_are_copied_interrupts_the_round(monkeypatch):
    from mpi4py import MPI

    recovery = _recovery(_FakeClient([]))
    load_balancer = recovery._executor.model_engine.moe_load_balancer
    load_balancer.stage_restored_slots.side_effect = MPI.Exception(MPI.ERR_OTHER)
    failed = frozenset({FAILED_RANK})
    monkeypatch.setattr(peer_loss_recovery, "is_mpi_peer_failure", lambda error: True)
    with pytest.raises(RecoveryInterrupted):
        recovery._stage_ring_restore(load_balancer, failed, ["placement"])
    # So does a loss a staging barrier observes.
    load_balancer.stage_restored_slots.side_effect = lambda plan, workspace, barrier: barrier()
    recovery._executor.dist.barrier_among_survivors.side_effect = MPI.Exception(MPI.ERR_OTHER)
    with pytest.raises(RecoveryInterrupted):
        recovery._stage_ring_restore(load_balancer, failed, ["placement"])
    load_balancer.stage_restored_slots.side_effect = MPI.Exception(MPI.ERR_OTHER)
    monkeypatch.setattr(peer_loss_recovery, "is_mpi_peer_failure", lambda error: False)
    with pytest.raises(MPI.Exception):
        recovery._stage_ring_restore(load_balancer, failed, ["placement"])


@pytest.mark.parametrize("staged_first_plan", [True, False])
def test_every_survivor_remaps_and_stages_again_without_every_lost_rank_whether_it_staged_or_not(
    staged_first_plan,
):
    """Rank 3 fails during the recovery without rank 2. A survivor that staged the first plan and one
    interrupted before it got the plan both remap the workspaces again and stage the plan without both."""
    import numpy as np

    lost = (FAILED_RANK, 3)
    first = Propose(1, (FAILED_RANK,), SURVIVORS)
    if staged_first_plan:
        replies = [first, _FURTHER_LOSS]
    else:
        replies = [_FURTHER_LOSS, first]
    client = _FakeClient([*replies, Propose(1, lost, (0, 1)), Install(1), Commit(1)])
    recovery = _recovery(client, failed=frozenset(lost))
    shrinks = iter([frozenset({FAILED_RANK}), frozenset(lost)])
    recovery._executor.dist.shrink.side_effect = lambda: next(shrinks)
    load_balancer = recovery._executor.model_engine.moe_load_balancer
    load_balancer.covers_loss_of.return_value = True
    load_balancer.recovery_placements.side_effect = lambda dead, base=None: [sorted(dead)]
    plans = []

    def restore_copy_plan(dead, placements, size):
        plans.append((sorted(dead), placements))
        return np.full((1, 10), len(dead), dtype=np.int64)

    load_balancer.restore_copy_plan.side_effect = restore_copy_plan

    def stage_restored_slots(copy_plan, workspace, barrier):
        # The first plan (dead ranks: 1) stages two chunks.
        for _ in range(3 - int(copy_plan[0, 0])):
            barrier()

    load_balancer.stage_restored_slots.side_effect = stage_restored_slots

    recovery._recover(_interrupted())

    assert plans[-1] == (list(lost), [list(lost)])
    assert len(plans) == (2 if staged_first_plan else 1)
    remapped = [call.args[0] for call in recovery._remap_workspaces.call_args_list]
    assert remapped[-1] == frozenset(lost)
    assert len(remapped) == len(plans)
    stopped = [message for message in client.sent if isinstance(message, Stopped)]
    assert stopped == [
        Stopped(rank=1, failed_ranks=(FAILED_RANK,)),
        Stopped(rank=1, failed_ranks=lost),
    ]
