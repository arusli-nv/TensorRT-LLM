# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Scheduler-side WideEP peer-loss recovery with fake distributed, model, and authority."""

import functools
import pickle
import signal
import threading
import time
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock

import pytest
import torch

from tensorrt_llm._torch.distributed.communicator import HostCollectiveInterrupted, MPIDist
from tensorrt_llm._torch.moe.fused_moe.ep_group_health import EPGroupHealth
from tensorrt_llm._torch.pyexecutor import llm_request, peer_loss_recovery
from tensorrt_llm._torch.pyexecutor.executor_request_queue import (
    CONTROL_REQUEST_ID,
    RequestQueueItem,
)
from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import KVCacheManagerV2
from tensorrt_llm._torch.pyexecutor.llm_request import LlmResponse, PyResult
from tensorrt_llm._torch.pyexecutor.peer_loss_recovery import (
    PeerLossRecovery,
    PreparedStep,
    RecoveryHook,
)
from tensorrt_llm._torch.pyexecutor.py_executor import PyExecutor
from tensorrt_llm._torch.pyexecutor.scheduler import FCFSWaitingQueue
from tensorrt_llm._torch.pyexecutor.scheduler.adp_router import DefaultADPRouter, RankState
from tensorrt_llm._torch.speculative.interface import SpeculativeDecodingMode as SpecMode
from tensorrt_llm.bindings import executor as trtllm
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
from tensorrt_llm.executor.utils import is_peer_loss_retryable_message
from tensorrt_llm.runtime.kv_cache_manager_v2 import GPU_LEVEL
from tensorrt_llm.scheduling_params import SchedulingParams

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
    recovery._end_departed_rank = Mock(return_value="")
    recovery._recover(_interrupted())
    recovery._end_departed_rank.assert_called_once_with(FAILED_RANK)
    assert client.sent[:3] == [
        Stopped(rank=0, failed_ranks=(FAILED_RANK,)),
        Fenced(rank=0, target=FAILED_RANK, error=""),
        Ready(rank=0, generation=1, placement_digest="digest"),
    ]


def test_a_departed_rank_the_runtime_cannot_end_is_reported_and_the_recovery_ends():
    client = _FakeClient([Fence(FAILED_RANK, 0), Abort("departed rank 2 could not be fenced")])
    recovery = _recovery(client, rank=0)
    recovery._end_departed_rank = Mock(return_value="PMIx refused to kill rank 2")
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
        error = recovery._end_departed_rank(FAILED_RANK)
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


def _runtime(monkeypatch, kill_rank):
    """The MPI runtime: session rank r is runtime rank 10 + r; ``kill_rank`` ends a runtime rank."""
    from tensorrt_llm import _utils
    from tensorrt_llm._torch.distributed import pmix_fence

    monkeypatch.setattr(_utils, "mpi_comm", lambda: None)
    monkeypatch.setattr(pmix_fence, "pmix_ranks", lambda comm: (10, 11, 12, 13))
    monkeypatch.setattr(pmix_fence, "kill_rank", kill_rank)


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
    _runtime(monkeypatch, Mock(side_effect=AssertionError("nothing to kill")))
    recovery = _recovery(_FakeClient(_happy_replies()))
    recovery._recover(_interrupted())


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
    load_balancer.quiesce.return_value = True
    load_balancer.current_placements.return_value = [np.zeros((WORLD_SIZE, 2))]
    dist.broadcast_among_survivors = Mock(return_value=[np.zeros((WORLD_SIZE, 2))])
    with pytest.raises(RecoveryTimeout, match="planning the recovery again"):
        recovery._recover(_interrupted())
    assert dist.watch_survivor_failures.call_count == 1
    assert dist.broadcast_among_survivors.call_count == 1


def _queued(request_id):
    return RequestQueueItem(request_id, Mock())


def _broadcast(recovery, items, interrupted=False):
    """Broadcast ``items`` through the recovery hook, as the executor does."""
    if not interrupted:
        recovery.broadcast_requests(lambda _: None, items)
        return
    with pytest.raises(HostCollectiveInterrupted):
        recovery.broadcast_requests(Mock(side_effect=_interrupted()), items)


@pytest.mark.parametrize("rank", [0, 1])
def test_recovery_keeps_unscheduled_requests_on_rank0_only(rank):
    """Rank 0 re-broadcasts received-but-unscheduled requests after recovery."""
    recovery = _recovery(_FakeClient(_happy_replies()), rank=rank)
    executor = recovery._executor
    executor.request_accumulated = [_queued(5)]
    executor.waiting_queue.add_requests([_queued(1), _queued(2)])
    _broadcast(recovery, [_queued(3)], interrupted=True)

    recovery._recover(_interrupted())

    assert len(executor.waiting_queue) == 0
    assert recovery._unbroadcast_requests == []
    expected = [1, 2, 3, 5] if rank == 0 else [5]
    assert [item.id for item in executor.request_accumulated] == expected


def test_recovery_does_not_rebroadcast_a_request_a_survivor_started():
    """Rank 0's broadcast can complete on a survivor while rank 0 sees it interrupted."""
    recovery = _recovery(_FakeClient(_happy_replies()), rank=0)
    executor = recovery._executor
    _broadcast(recovery, [_queued(3), _queued(4)], interrupted=True)
    served = [(set(), 0, set(), set()), ({3}, 0, set(), set()), None, (set(), 0, set(), set())]
    executor.dist.allgather.side_effect = lambda value, departed: served

    recovery._recover(_interrupted())

    assert [item.id for item in executor.request_accumulated] == [4]
    if recovery._request_owners is not None:
        assert recovery._request_owners[3][0] == 1


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


def test_recovery_reports_requests_it_already_answered_as_held():
    """A request a survivor rejected from an interrupted broadcast must not be admitted again."""
    recovery = _recovery(_FakeClient(_happy_replies()), rank=1)
    executor = recovery._executor
    rejected = Mock(spec=LlmResponse, request_id=3, result=None)
    executor._pending_transfer_responses = [(3, rejected)]
    reported = []
    executor.dist.allgather.side_effect = (
        lambda value, departed: reported.append(value) or [value] * WORLD_SIZE
    )

    recovery._recover(_interrupted())

    held, *_ = reported[0]
    assert 3 in held


def test_recovery_rebroadcasts_the_cancellations_of_an_interrupted_broadcast():
    """Rank 0 may finish a broadcast that a survivor never received."""
    recovery = _recovery(_FakeClient(_happy_replies()), rank=0)
    executor = recovery._executor
    cancel = RequestQueueItem(7, None, is_canceled_request=True)
    _broadcast(recovery, [cancel])
    executor.waiting_queue.add_requests([_queued(1)])

    recovery._recover(_interrupted())

    assert [item.id for item in executor.request_accumulated] == [1, 7]


def test_recovery_fails_closed_on_an_interrupted_control_request():
    recovery = _recovery(_FakeClient(_happy_replies()), rank=0)
    _broadcast(recovery, [RequestQueueItem(CONTROL_REQUEST_ID)])
    with pytest.raises(RuntimeError, match="control request"):
        recovery._recover(_interrupted())


def test_gather_numbers_follow_rank0_after_a_recovery():
    """A rank ahead of rank 0 numbers its next gathers like rank 0."""
    recovery = _recovery(_FakeClient(_happy_replies()), rank=1)
    for number in range(3):
        recovery.gathering_responses([(number, Mock())])
    recovery._executor.dist.broadcast.side_effect = (
        lambda value, root: (2, set()) if isinstance(value, tuple) else value
    )
    _sent_responses(recovery)

    recovery._recover(_interrupted())

    assert recovery._gather_count == recovery._last_processed_gather == 2


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
    executor._run_event_loop = Mock(side_effect=RuntimeError(message))
    executor._is_warmup = True
    executor._event_loop_error_delivered = Mock()
    executor.garbage_collection_gen0_threshold = None
    kills = []
    monkeypatch.setattr(py_executor.os, "kill", lambda pid, sig: kills.append(sig))
    monkeypatch.setattr(py_executor, "hard_kill_on_rank_crash", Mock())
    monkeypatch.setattr(py_executor, "start_rank_crash_kill_watchdog", Mock(return_value=None))
    executor._executor_loop_cleanup = Mock()
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


def test_commit_step_discards_a_step_whose_peer_failed_while_it_ran():
    """The flag is read after waiting for the step's device work, not before."""
    recovery = _recovery(_FakeClient([]))
    batch = Mock(name="batch", context_requests=[])
    recovery.step_prepared(batch)
    sample_state = Mock()
    sample_state.sampler_event.synchronize.side_effect = recovery._peer_failed.set
    with pytest.raises(HostCollectiveInterrupted):
        recovery.commit_step(batch, sample_state)


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


def test_routed_requests_are_tracked_until_their_final_response():
    recovery = _recovery(_FakeClient([]), rank=0)
    item = Mock(id=11, request=Mock(client_id="c11"), child_req_ids=[21])
    recovery.requests_routed({FAILED_RANK: [item]})
    assert recovery._request_owners == {11: (FAILED_RANK, item)}
    streaming = Mock(spec=LlmResponse, request_id=11, result=Mock(is_final=False))
    streaming.has_error.return_value = False
    recovery.responses_gathered([(11, streaming)])
    assert 11 in recovery._request_owners
    # The last sequence of an n > 1 request is gathered under its own id.
    final = Mock(spec=LlmResponse, request_id=11, result=Mock(is_final=True))
    final.has_error.return_value = False
    recovery.responses_gathered([(21, final)])
    assert recovery._request_owners == {} and recovery._parent_ids == {}


@pytest.mark.parametrize("sequence_id", [11, 21])
def test_an_error_of_any_sequence_ends_the_request(sequence_id):
    recovery = _recovery(_FakeClient([]), rank=0)
    recovery.requests_routed({1: [Mock(id=11, child_req_ids=[21])]})
    error = Mock(spec=LlmResponse, request_id=sequence_id, result=None)
    error.has_error.return_value = True
    recovery.responses_gathered([(sequence_id, error)])
    assert recovery._request_owners == {} and recovery._parent_ids == {}


def test_recovery_reruns_a_departed_rank_request_that_delivered_nothing():
    """Nothing reached the client: a survivor runs it again, once, after the loss of its rank alone too."""
    recovery = _recovery(_FakeClient(_happy_replies()), rank=0)
    lost, kept = Mock(id=11, child_req_ids=None), Mock(id=12, child_req_ids=None)
    recovery._request_owners = {11: (FAILED_RANK, lost), 12: (3, kept)}
    recovery._executor.active_requests.append(
        Mock(is_child=False, py_request_id=12, is_attention_dp_dummy=False)
    )
    sent = _sent_responses(recovery)

    recovery._recover(_interrupted())

    assert sent[-1] == []
    assert recovery._executor.request_accumulated[:1] == [lost]
    assert recovery._request_owners == {12: (3, kept)}
    # Marked as interrupted until it ends.
    assert recovery._interrupted == {11}
    recovery.requests_routed({1: [lost]})
    recovery.responses_gathered([(11, LlmResponse(request_id=11, result=Mock(is_final=True)))])
    assert recovery._interrupted == set() and recovery._request_owners == {12: (3, kept)}


def test_a_request_pinned_to_a_departed_rank_runs_again_on_a_survivor(monkeypatch):
    """The request was accepted while its rank served: it runs again on a survivor. A new request pinned to the
    departed rank still fails."""
    recovery = _recovery(_FakeClient(_happy_replies()), rank=0)
    warnings = []
    monkeypatch.setattr(peer_loss_recovery.logger, "warning", warnings.append)

    def pinned(request_id):
        params = SchedulingParams(attention_dp_rank=FAILED_RANK, attention_dp_relax=False)
        request = Mock(py_scheduling_params=params, num_input_tokens=4)
        return Mock(id=request_id, request=request, child_req_ids=None)

    lost, new = pinned(11), pinned(12)
    recovery._request_owners = {11: (FAILED_RANK, lost)}
    _sent_responses(recovery)

    recovery._recover(_interrupted())

    assert recovery._executor.request_accumulated[:1] == [lost]
    assert "re-ran 1 requests and failed 0; lent 0 bytes of the KV cache" in warnings[-1]
    assert lost.request.py_scheduling_params == SchedulingParams(
        attention_dp_rank=FAILED_RANK, attention_dp_relax=True
    )
    states = [RankState(rank=rank) for rank in SURVIVORS]
    routable, unservable = DefaultADPRouter.split_unservable_requests([lost, new], states)
    assert routable == [lost] and unservable == [new]
    router = DefaultADPRouter(dist=Mock(tp_rank=0, tp_size=WORLD_SIZE, has_cp_helix=False))
    routed, _ = router.route_requests(states, routable, 8)
    assert set(routed) == set(SURVIVORS)
    assert [req for reqs in routed.values() for req in reqs] == [lost]


@pytest.mark.parametrize("streamed", [True, False])
def test_recovery_fails_a_departed_rank_request_that_streamed_or_already_ran_again(
    monkeypatch, streamed
):
    """Streamed tokens cannot be taken back; a request that already ran again after its rank failed, and lost its
    new rank too, may be what crashes them: it is quarantined."""
    recovery = _recovery(_FakeClient(_happy_replies()), rank=0)
    warnings = []
    monkeypatch.setattr(peer_loss_recovery.logger, "warning", warnings.append)
    item = Mock(id=11, request=Mock(client_id="c11"), child_req_ids=None)
    recovery._request_owners = {11: (FAILED_RANK, item)}
    if streamed:
        recovery._delivered.add(11)
    else:
        recovery._interrupted.add(11)
    sent = _sent_responses(recovery)

    recovery._recover(_interrupted())

    ((request_id, response),) = sent[-1]
    assert request_id == 11 and response.client_id == "c11"
    assert f"rank {FAILED_RANK} failed" in response.error_msg
    assert is_peer_loss_retryable_message(response.error_msg)
    assert ("streamed" in response.error_msg) == streamed
    assert ("already interrupted" in response.error_msg) != streamed
    assert "re-ran 0 requests and failed 1;" in warnings[-1]
    assert recovery._request_owners == {} and recovery._interrupted == set()
    assert recovery._executor.request_accumulated == []


def test_a_request_run_again_fails_when_it_loses_its_rank_again():
    """Rank 2's request runs again on rank 3 after the first recovery; rank 3 is lost next: the request fails, and
    the deployment recovers a second time without both ranks."""
    lost = frozenset({FAILED_RANK, 3})
    client = _FakeClient(
        [
            *_happy_replies(),
            *_happy_replies(generation=2, failed=tuple(sorted(lost)), survivors=(0, 1)),
        ]
    )
    recovery = _recovery(client, rank=0, failed=lost)
    recovery._executor.model_engine.moe_load_balancer.covers_loss_of.return_value = True
    shrinks = iter([frozenset({FAILED_RANK}), lost])
    recovery._executor.dist.shrink.side_effect = lambda: next(shrinks)
    item = Mock(id=11, request=Mock(client_id="c11"), child_req_ids=None)
    recovery._request_owners = {11: (FAILED_RANK, item)}
    sent = _sent_responses(recovery)

    recovery._recover(_interrupted())
    assert recovery._executor.request_accumulated == [item]
    recovery._executor.request_accumulated.clear()
    recovery.requests_routed({3: [item]})

    recovery._recover(_interrupted(lost))

    ((request_id, response),) = sent[-1]
    assert request_id == 11 and "already interrupted" in response.error_msg
    assert recovery._executor.request_accumulated == []
    assert [m.generation for m in client.sent if isinstance(m, Installed)] == [1, 2]


def _interrupting(recovery, failed, interrupted, progressed=frozenset()):
    """Report ``interrupted`` as the requests the loss interrupted on rank 1, which serves requests 11 and 12,
    and ``progressed`` as those of them that committed a step since an earlier loss interrupted them."""
    recovery._executor.dist.allgather.side_effect = lambda value, departed: [
        None
        if r in failed
        else ({11, 12}, 0, interrupted, set(progressed))
        if r == 1
        else (set(), 0, set(), set())
        for r in range(WORLD_SIZE)
    ]


def test_a_request_two_losses_interrupt_fails_whatever_ranks_were_lost():
    """A survivor's request runs on after a loss interrupts its step, but a second interruption fails it: it may
    be what crashes the ranks. A request interrupted once runs on."""
    lost = frozenset({FAILED_RANK, 3})
    client = _FakeClient(
        [
            *_happy_replies(),
            *_happy_replies(generation=2, failed=tuple(sorted(lost)), survivors=(0, 1)),
        ]
    )
    recovery = _recovery(client, rank=0, failed=lost)
    executor = recovery._executor
    executor.model_engine.moe_load_balancer.covers_loss_of.return_value = True
    shrinks = iter([frozenset({FAILED_RANK}), lost])
    executor.dist.shrink.side_effect = lambda: next(shrinks)
    recovery.requests_routed(
        {1: [Mock(id=request_id, child_req_ids=None) for request_id in (11, 12)]}
    )
    failures = []
    executor.fail_requests = lambda request_ids, error_msg: failures.append(set(request_ids))
    _sent_responses(recovery)

    _interrupting(recovery, {FAILED_RANK}, {11})
    recovery._recover(_interrupted())
    assert failures == [set()] and recovery._interrupted == {11}

    _interrupting(recovery, lost, {11, 12})
    recovery._recover(_interrupted(lost))
    assert failures[-1] == {11} and recovery._interrupted == {11, 12}
    assert executor.request_accumulated == []


def test_a_request_that_progressed_after_an_interruption_runs_on_after_another():
    """Under the overlap scheduler most requests sit in a discarded step: one that committed a step since an
    earlier loss interrupted it is not failed by an unrelated second loss."""
    lost = frozenset({FAILED_RANK, 3})
    client = _FakeClient(
        [
            *_happy_replies(),
            *_happy_replies(generation=2, failed=tuple(sorted(lost)), survivors=(0, 1)),
        ]
    )
    recovery = _recovery(client, rank=0, failed=lost)
    executor = recovery._executor
    executor.model_engine.moe_load_balancer.covers_loss_of.return_value = True
    shrinks = iter([frozenset({FAILED_RANK}), lost])
    executor.dist.shrink.side_effect = lambda: next(shrinks)
    recovery.requests_routed(
        {1: [Mock(id=request_id, child_req_ids=None) for request_id in (11, 12)]}
    )
    failures = []
    executor.fail_requests = lambda request_ids, error_msg: failures.append(set(request_ids))
    _sent_responses(recovery)

    _interrupting(recovery, {FAILED_RANK}, {11, 12})
    recovery._recover(_interrupted())
    _interrupting(recovery, lost, {11, 12}, progressed={12})
    recovery._recover(_interrupted(lost))

    assert failures[-1] == {11} and recovery._interrupted == {11, 12}


def test_a_survivor_reports_the_interrupted_requests_whose_step_committed_once():
    recovery = _recovery(_FakeClient(_happy_replies()), rank=1)
    executor = recovery._executor
    requests = {
        request_id: Mock(
            is_child=False, is_dummy=False, is_attention_dp_dummy=False, py_request_id=request_id
        )
        for request_id in (11, 12)
    }
    executor.active_requests = list(requests.values())
    interrupted = Mock(context_requests=[], all_requests=lambda: list(requests.values()))
    recovery.step_prepared(interrupted)
    reported = []
    executor.dist.allgather.side_effect = (
        lambda value, departed: reported.append(value) or [value] * WORLD_SIZE
    )
    recovery._recover(_interrupted())
    assert recovery._unprogressed == {11, 12}

    for _ in range(2):
        batch = Mock(context_requests=[], all_requests=lambda: [requests[12]])
        recovery.step_prepared(batch)
        recovery.commit_step(batch, Mock())

    assert recovery._unprogressed == {11} and recovery._progressed == {12}


def test_survivors_fail_the_sequences_of_the_requests_rank0_quarantines():
    recovery = _recovery(_FakeClient([]), rank=1)
    executor = recovery._executor
    executor.dist.broadcast.side_effect = (
        lambda value, root: (0, {11}) if isinstance(value, tuple) else value
    )
    parent = Mock(is_child=False, py_request_id=11)
    child = Mock(is_child=True, parent_request_id=11, py_request_id=21)
    other = Mock(is_child=False, py_request_id=12)
    executor.active_requests = [parent, child, other]
    executor._handle_errors = Mock()
    _sent_responses(recovery)

    recovery._deliver_responses(frozenset({FAILED_RANK}), ([], []), set())

    (call,) = executor._handle_errors.call_args_list
    assert call.kwargs["requests"] == [parent, child] and not call.kwargs["charge_budget"]
    assert "interrupted this request a second time" in call.args[0]
    assert is_peer_loss_retryable_message(call.args[0])


def test_rank0_tracks_requests_that_delivered_a_response():
    recovery = _recovery(_FakeClient([]), rank=0)
    recovery.requests_routed({FAILED_RANK: [_queued(11), _queued(12)]})
    partial = LlmResponse(request_id=11, result=Mock(is_final=False))
    recovery.responses_gathered([(11, partial), (11, partial)])
    assert recovery._delivered == {11}
    recovery.responses_gathered([(11, LlmResponse(request_id=11, result=Mock(is_final=True)))])
    assert recovery._delivered == set() and set(recovery._request_owners) == {12}


def _response(request_id, is_final=False, error_msg=None):
    result = None if error_msg else SimpleNamespace(is_final=is_final)
    return LlmResponse(request_id=request_id, error_msg=error_msg, result=result)


def test_rank0_keeps_its_books_from_the_notes_of_each_gathered_part():
    """Each rank notes the requests its part finishes or first answers; rank 0 reads only those notes, with the
    same outcome as reading every response."""
    owner = _recovery(_FakeClient([]), rank=1)
    steps = [
        [(11, _response(11)), (12, _response(12)), (13, _response(13)), (23, _response(13))],
        [
            (11, _response(11)),
            (12, _response(12, is_final=True)),
            (23, _response(23, error_msg="boom")),
        ],
    ]
    parts = [owner.gathering_responses(step) for step in steps]
    assert [(part.finished, part.streaming) for part in parts] == [
        ([], [11, 12, 13]),
        ([12, 23], []),
    ]
    fast, scan = _recovery(_FakeClient([]), rank=0), _recovery(_FakeClient([]), rank=0)
    for rank0 in (fast, scan):
        rank0.requests_routed({1: [_queued(11), _queued(12), Mock(id=13, child_req_ids=[23])]})
    for step, part in zip(steps, parts):
        sent = pickle.loads(pickle.dumps(part))
        assert sent == step and (sent.finished, sent.streaming) == (part.finished, part.streaming)
        # Not iterable: rank 0 does not read the responses.
        fast.responses_gathered(object(), [None, sent])
        scan.responses_gathered(step)
    for rank0 in (fast, scan):
        assert (set(rank0._request_owners), rank0._delivered, rank0._parent_ids) == ({11}, {11}, {})


def test_a_resent_part_notes_the_requests_it_answered_again():
    """Rank 0 never read the notes of a gather a loss interrupted, so the resent part carries them again."""
    recovery = _recovery(_FakeClient(_happy_replies()), rank=1)
    recovery.gathering_responses([(11, _response(11))])
    recovery._executor.dist.broadcast.side_effect = (
        lambda value, root: (0, set()) if isinstance(value, tuple) else value
    )
    resent = []
    recovery._executor._enqueue_responses = lambda responses: resent.append(
        recovery.gathering_responses(responses)
    )

    recovery._recover(_interrupted())

    assert resent[0].streaming == [11]


@pytest.mark.parametrize("ran_again, failed_requests", [(False, 1), (True, 2)])
def test_rank0_reports_how_many_requests_the_departed_rank_was_serving(ran_again, failed_requests):
    """The authority's status counts the requests a loss failed: those that had streamed, and those that had
    already run again once."""
    client = _FakeClient(_happy_replies())
    recovery = _recovery(client, rank=0)
    if ran_again:
        recovery._interrupted.add(12)
    recovery._request_owners = {
        11: (FAILED_RANK, Mock(id=11, child_req_ids=None)),
        12: (FAILED_RANK, Mock(id=12, child_req_ids=None)),
        13: (1, Mock(id=13, child_req_ids=None)),
    }
    recovery._delivered.add(11)
    _sent_responses(recovery)

    recovery._recover(_interrupted())

    (installed,) = [message for message in client.sent if isinstance(message, Installed)]
    assert installed.failed_requests == failed_requests


def test_recovery_readmits_requests_that_never_reached_their_rank():
    """A torn broadcast can leave a routed request unknown to its live owner."""
    recovery = _recovery(_FakeClient(_happy_replies()), rank=0)
    torn, delivered = Mock(id=21, child_req_ids=None), Mock(id=22, child_req_ids=None)
    recovery._request_owners = {21: (3, torn), 22: (1, delivered)}
    held = {1: ({22}, 0, set(), set()), 3: (set(), 0, set(), set())}
    recovery._executor.dist.allgather.side_effect = lambda value, departed: [
        held.get(rank, value) for rank in range(WORLD_SIZE)
    ]
    _sent_responses(recovery)

    recovery._recover(_interrupted())

    assert recovery._executor.request_accumulated[:1] == [torn]
    assert recovery._request_owners == {22: (1, delivered)}


def test_a_cancelled_request_is_tracked_until_its_rank_answers_it():
    recovery = _recovery(_FakeClient([]), rank=0)
    recovery.requests_routed({3: [_queued(21)]})
    _broadcast(recovery, [RequestQueueItem(21, is_canceled_request=True)])
    assert 21 in recovery._request_owners and recovery._cancelled == {21}
    recovery.responses_gathered([(21, LlmResponse(request_id=21, result=Mock(is_final=True)))])
    assert recovery._request_owners == {} and recovery._cancelled == set()


def test_a_cancelled_request_of_a_departed_rank_ends_cancelled_not_run_again():
    """Its rank was lost before it answered the cancellation; a caller waiting after abort() needs a final
    response."""
    recovery = _recovery(_FakeClient(_happy_replies()), rank=0)
    request = trtllm.Request(input_token_ids=[1, 2, 3], max_tokens=4)
    recovery.requests_routed({FAILED_RANK: [RequestQueueItem(11, request)]})
    _broadcast(recovery, [RequestQueueItem(11, is_canceled_request=True)])
    sent = _sent_responses(recovery)

    recovery._recover(_interrupted())

    ((request_id, response),) = sent[-1]
    assert request_id == 11 and not response.has_error()
    response.result.deserialize()
    assert response.result.is_final
    assert response.result.finish_reasons == [trtllm.FinishReason.CANCELLED]
    assert recovery._request_owners == {} and recovery._cancelled == set()
    assert all(item.is_canceled_request for item in recovery._executor.request_accumulated)


def test_survivors_join_departed_request_responses():
    """Ranks without the owner table still take part in the response collective."""
    recovery = _recovery(_FakeClient(_happy_replies()), rank=1)
    assert recovery._request_owners is None
    sent = _sent_responses(recovery)
    recovery._recover(_interrupted())
    assert sent == [[]]


@pytest.mark.parametrize("processed", [0, 1, 2])
def test_recovery_resends_gathers_rank0_did_not_process(processed):
    """A rank resends its parts of every gather rank 0 did not process."""
    recovery = _recovery(_FakeClient(_happy_replies()), rank=1)
    first, second = [(5, Mock(name="first"))], [(6, Mock(name="second"))]
    recovery.gathering_responses(first)
    recovery.gathering_responses(second)
    recovery._executor.dist.broadcast.side_effect = (
        lambda value, root: (processed, set()) if isinstance(value, tuple) else value
    )
    finished = Mock(name="finished")
    recovery.defer_terminations([finished])
    order = []
    executor = recovery._executor
    executor._enqueue_responses = lambda responses: order.append(("send", list(responses)))
    executor._terminate_request = lambda request: order.append(("terminate", request))

    recovery._recover(_interrupted())

    expected = [first + second, second, []][processed]
    assert order == [("send", expected), ("terminate", finished)]
    assert recovery._undelivered_terminations == []


def test_recovery_fails_closed_if_an_owed_gather_is_no_longer_held():
    recovery = _recovery(_FakeClient(_happy_replies()), rank=1)
    for number in range(peer_loss_recovery._GATHER_HISTORY + 1):
        recovery.gathering_responses([(number, Mock())])
    recovery._executor.dist.broadcast.side_effect = (
        lambda value, root: (0, set()) if isinstance(value, tuple) else value
    )
    with pytest.raises(RuntimeError, match="no longer held"):
        recovery._recover(_interrupted())


def test_recovery_fails_closed_if_gather_numbers_diverged():
    """Rank 0 cannot have processed a gather this rank never made."""
    recovery = _recovery(_FakeClient(_happy_replies()), rank=1)
    recovery.gathering_responses([(5, Mock())])
    recovery._executor.dist.broadcast.side_effect = (
        lambda value, root: (2, set()) if isinstance(value, tuple) else value
    )
    with pytest.raises(RuntimeError, match="made only 1"):
        recovery._recover(_interrupted())


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
        with pytest.raises(TimeoutError):
            PeerLossRecovery._join_device_work(recovery, time.monotonic() + 0.05)
    finally:
        release.set()
        for thread in threading.enumerate():
            if thread.name == "peer_loss_device_join":
                thread.join(timeout=5)


def test_recovery_delivers_buffered_error_responses_before_readmission():
    """A request that failed on its rank is retired, not re-admitted as never received."""
    recovery = _recovery(_FakeClient(_happy_replies()), rank=1)
    executor = recovery._executor
    failed_request, error = Mock(name="failed"), Mock(name="error")
    executor._pending_transfer_responses = [(9, error)]
    executor._pending_response_terminations = [failed_request]
    order = []
    executor._enqueue_responses = lambda responses: order.append(("send", list(responses)))
    executor._terminate_request = lambda request: order.append(("terminate", request))

    recovery._recover(_interrupted())

    assert order == [("send", [(9, error)]), ("terminate", failed_request)]
    assert executor._pending_transfer_responses == []
    assert executor._pending_response_terminations == []


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
        hang_detector=SimpleNamespace(timeout=300),
        llm_args=SimpleNamespace(sleep_config=None),
    )


@pytest.mark.parametrize(
    "unsupported, requirement",
    [
        (lambda ex: setattr(ex, "guided_decoder", Mock()), "no guided decoding"),
        (lambda ex: setattr(ex, "max_beam_width", 2), "no beam search"),
        (lambda ex: setattr(ex, "kv_connector_manager", Mock()), "no KV connector"),
        (lambda ex: setattr(ex, "enable_kv_pool_rebalance", True), "no KV pool rebalance"),
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


def test_create_accepts_online_eplb(monkeypatch):
    monkeypatch.setattr(peer_loss_recovery, "peer_loss_recovery_enabled", lambda: True)
    monkeypatch.setattr(
        peer_loss_recovery, "_register", lambda rank, recoverable: (Mock(), frozenset())
    )
    executor = _supported_executor()
    executor.model_engine.moe_load_balancer.is_static_routing.return_value = False
    assert isinstance(PeerLossRecovery.create(executor), PeerLossRecovery)
    executor.model_engine.moe_load_balancer.restore_copy_plan.assert_not_called()


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
    load_balancer.stage_restored_slots.side_effect = lambda *args: sent_before_staging.append(
        list(client.sent)
    )

    recovery._recover(_interrupted())

    failed = frozenset({FAILED_RANK})
    load_balancer.restore_copy_plan.assert_called_once_with(
        failed,
        ["placement"],
        workspace.size.return_value,
        engine.moe_execution_state.staging_write_ranges,
        [],
    )
    workspace.size.assert_called_with(1)
    load_balancer.stage_restored_slots.assert_called_once_with(
        plan, workspace, recovery._executor.dist.barrier_among_survivors
    )
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
        recovery._stage_ring_restore(load_balancer, failed, ["placement"], ())
    monkeypatch.setattr(peer_loss_recovery, "is_mpi_peer_failure", lambda error: False)
    with pytest.raises(MPI.Exception):
        recovery._stage_ring_restore(load_balancer, failed, ["placement"], ())


@pytest.mark.parametrize("staged_first_plan", [True, False])
def test_every_survivor_replans_off_the_plans_proposed_earlier_whether_it_staged_them_or_not(
    staged_first_plan,
):
    """Rank 3 fails during the recovery without rank 2. A survivor that staged the first plan and one
    interrupted before it got the plan both keep the next plan off the first plan's copies."""
    import numpy as np

    lost = (FAILED_RANK, 3)
    first = Propose(1, (FAILED_RANK,), SURVIVORS)
    if staged_first_plan:
        replies = [first, _FURTHER_LOSS]
    else:
        replies = [_FURTHER_LOSS, first]
    client = _FakeClient(
        [*replies, Propose(1, lost, (0, 1), ((FAILED_RANK,),)), Install(1), Commit(1)]
    )
    recovery = _recovery(client, failed=frozenset(lost))
    shrinks = iter([frozenset({FAILED_RANK}), frozenset(lost)])
    recovery._executor.dist.shrink.side_effect = lambda: next(shrinks)
    load_balancer = recovery._executor.model_engine.moe_load_balancer
    load_balancer.covers_loss_of.return_value = True
    load_balancer.recovery_placements.side_effect = lambda dead, base=None: [sorted(dead)]
    plans = []

    def restore_copy_plan(dead, placements, size, ranges, staged):
        plans.append((sorted(dead), placements, [plan.tolist() for plan in staged]))
        return np.full((1, 10), len(dead), dtype=np.int64)

    load_balancer.restore_copy_plan.side_effect = restore_copy_plan

    recovery._recover(_interrupted())

    first_plan = [[1] * 10]
    assert plans[-2:] == [
        ([FAILED_RANK], [[FAILED_RANK]], []),
        (list(lost), [list(lost)], [first_plan]),
    ]
    assert len(plans) == (3 if staged_first_plan else 2)


def test_create_checks_the_staging_workspace_fits_with_ring_host_copies(monkeypatch):
    monkeypatch.setattr(peer_loss_recovery, "peer_loss_recovery_enabled", lambda: True)
    monkeypatch.setattr(
        peer_loss_recovery, "_register", lambda rank, recoverable: (Mock(), frozenset())
    )
    executor = _supported_executor()
    load_balancer = executor.model_engine.moe_load_balancer
    load_balancer.restore_copy_plan.side_effect = ValueError("does not fit")
    with pytest.raises(ValueError, match="does not fit"):
        PeerLossRecovery.create(executor)
    load_balancer.restore_copy_plan.side_effect = None
    assert isinstance(PeerLossRecovery.create(executor), PeerLossRecovery)
    state = executor.model_engine.moe_execution_state
    load_balancer.restore_copy_plan.assert_called_with(
        [],
        load_balancer.current_placements(),
        state.staging_workspace().size(1),
        state.staging_write_ranges,
    )
    assert load_balancer.restore_copy_plan.call_count == 2
