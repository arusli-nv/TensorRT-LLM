# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Worker-side hang evidence: lagging peers from the all-to-all progress row, and the reporter."""

import os
import signal
import socket
import threading
import time

import pytest
import torch

from tensorrt_llm._torch.pyexecutor import hang_evidence
from tensorrt_llm._torch.pyexecutor.hang_evidence import (
    HangEvidenceReporter,
    ProgressRow,
    ProgressRowReader,
    lagging_peers,
)
from tensorrt_llm.executor.recovery import protocol
from tensorrt_llm.executor.recovery.config import STALLED_EXIT_CODE
from tensorrt_llm.executor.recovery.protocol import (
    DeviceWait,
    Fence,
    Heartbeat,
    NativeHeartbeat,
    Restart,
)

RANKS = (0, 1, 2, 3)


def _row(flag_val, dispatch, combine):
    return ProgressRow(flag_val, tuple(dispatch), tuple(combine))


def test_no_laggard_between_steps_or_once_every_peer_finished():
    # After a combine (round 8) every peer wrote 8; the dispatch flags hold 7.
    assert lagging_peers(_row(8, [7] * 4, [8] * 4), RANKS, rank=0) == set()
    # Waiting in dispatch round 9: peers 1 and 3 finished it, peer 2 did not.
    assert lagging_peers(_row(9, [9, 9, 7, 9], [8] * 4), RANKS, rank=0) == {2}
    # A peer outside the committed membership is never a laggard.
    assert lagging_peers(_row(9, [9, 9, 7, 9], [8] * 4), (0, 1, 3), rank=0) == set()


def test_flags_compare_as_unsigned_words():
    assert lagging_peers(_row(-1, [0xFFFFFFFF] * 4, [0] * 4), RANKS, rank=1) == set()


class _Reporter:
    def __init__(self, rows, suspect_s=2.0, heartbeat_s=1.0):
        self.sent = []
        self.rows = list(rows)
        self.reporter = HangEvidenceReporter(
            rank=0,
            send=self.sent.append,
            read_row=lambda: self.rows.pop(0) if self.rows else None,
            active_ranks=lambda: RANKS,
            iteration=lambda: 42,
            waiting=lambda: False,
            suspect_s=suspect_s,
            heartbeat_s=heartbeat_s,
        )

    def waits(self):
        return [m for m in self.sent if isinstance(m, DeviceWait)]


def test_reports_a_wait_once_it_lasts_the_suspect_time_and_clears_it():
    stuck = _row(9, [9, 9, 7, 9], [8] * 4)
    done = _row(10, [9] * 4, [10] * 4)
    r = _Reporter([stuck, stuck, stuck, stuck, done])
    for now in (0.0, 1.0, 2.0, 3.0, 4.0):
        r.reporter.poll(now)
    heartbeats = [m for m in r.sent if isinstance(m, Heartbeat)]
    assert len(heartbeats) == 5 and heartbeats[0].iteration == 42 and heartbeats[0].pid > 0
    assert r.waits() == [DeviceWait(0, 2, 2000), DeviceWait(0, 2, 0)]


def test_short_waits_and_stale_rows_report_nothing():
    stuck = _row(9, [9, 9, 7, 9], [8] * 4)
    done = _row(10, [9] * 4, [10] * 4)
    r = _Reporter([stuck, done, stuck])
    for now in (0.0, 1.0, 2.0):
        r.reporter.poll(now)
    assert r.waits() == []
    # The copy stops landing (this rank's own device is stuck): no evidence about peers.
    r = _Reporter([stuck])
    for now in (0.0, 1.0, 2.0, 3.0, 4.0, 5.0):
        r.reporter.poll(now)
    assert r.waits() == []


def test_a_reported_wait_is_cleared_once_rows_stop_landing():
    stuck = _row(9, [9, 9, 7, 9], [8] * 4)
    r = _Reporter([stuck, stuck, stuck])
    for now in (0.0, 1.0, 2.0, 3.0, 4.0):
        r.reporter.poll(now)
    # Reported at 2 s; at 4 s this rank's own copy has not landed for two polls: no current evidence.
    assert r.waits() == [DeviceWait(0, 2, 2000), DeviceWait(0, 2, 0)]


def _fence_reporter(monkeypatch, messages, runtime_kill=None, pmix_ranks=RANKS):
    killed = []
    monkeypatch.setattr(hang_evidence.os, "kill", lambda pid, sig: killed.append((pid, sig)))
    if runtime_kill is not False:
        monkeypatch.setattr(
            hang_evidence,
            "kill_rank",
            runtime_kill or (lambda rank: killed.append(("runtime", rank))),
        )
    inbox = list(messages)
    sent = []
    reporter = HangEvidenceReporter(
        rank=0,
        send=sent.append,
        read_row=lambda: None,
        active_ranks=lambda: RANKS,
        iteration=lambda: 42,
        waiting=lambda: False,
        receive=lambda: inbox.pop(0) if inbox else None,
        pmix_ranks=pmix_ranks,
    )
    reporter.sent = sent
    return reporter, killed, inbox


def _join_runtime_fence(reporter):
    reporter._runtime_fence.join(5)
    assert not reporter._runtime_fence.is_alive()


def test_a_fenced_rank_ends_its_own_process(monkeypatch):
    reporter, killed, _ = _fence_reporter(monkeypatch, [Fence(0, os.getpid())])
    reporter.poll(0.0)
    assert killed == [(os.getpid(), signal.SIGKILL)]


def test_a_restart_ends_the_process_with_the_stalled_exit_code(monkeypatch):
    exits = []
    monkeypatch.setattr(hang_evidence.os, "_exit", exits.append)
    reporter, killed, _ = _fence_reporter(monkeypatch, [Restart("every rank stalled")])
    reporter.poll(0.0)
    assert exits == [STALLED_EXIT_CODE] and killed == []


def test_the_first_heartbeat_is_sent_before_the_rank_serves():
    """A rank stopped right after it starts serving must already be known to the authority."""
    r = _Reporter([], heartbeat_s=60.0)
    senders = []
    r.reporter._send = lambda message: (
        senders.append(threading.current_thread()),
        r.sent.append(message),
    )
    r.reporter.start()
    try:
        assert r.sent == [Heartbeat(0, 42, False, r.reporter._host, os.getpid())]
        # The connection is used from the reporter thread only.
        assert senders == [r.reporter._thread]
    finally:
        assert r.reporter.stop()


def test_stop_gives_up_on_a_blocked_reporter_thread(monkeypatch):
    monkeypatch.setattr(hang_evidence, "_STOP_TIMEOUT_S", 0.05)
    blocked, release = threading.Event(), threading.Event()
    r = _Reporter([], heartbeat_s=0.01)

    def blocked_read_row():
        blocked.set()
        release.wait(30)

    r.reporter._read_row = blocked_read_row
    r.reporter.start()
    assert blocked.wait(5)
    try:
        assert not r.reporter.stop()
    finally:
        release.set()
    r.reporter._thread.join(5)
    assert not r.reporter._thread.is_alive()


def test_the_native_heartbeat_keeps_arriving_while_python_is_starved():
    """``HeartbeatSender`` sends from a native thread while this thread holds the interpreter lock."""
    from tensorrt_llm.bindings.internal.runtime import HeartbeatSender

    key = b"k" * 32
    receiver = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    receiver.bind(("127.0.0.1", 0))
    sender = HeartbeatSender("127.0.0.1", receiver.getsockname()[1], key, 3, os.getpid(), 0.05)
    try:
        sender.publish(17)
        # Busy-wait without releasing the lock: no other Python thread runs meanwhile.
        deadline = time.monotonic() + 1.0
        while time.monotonic() < deadline:
            pass
    finally:
        sender.stop()
    receiver.setblocking(False)
    beats = []
    while True:
        try:
            datagram = receiver.recv(protocol.MAX_MESSAGE_BYTES)
        except BlockingIOError:
            break
        beats.append(protocol.decode_datagram(datagram, key))
    receiver.close()
    assert len(beats) >= 10 and all(isinstance(beat, NativeHeartbeat) for beat in beats)
    assert [beat.beat for beat in beats] == list(range(1, len(beats) + 1))
    assert {(beat.rank, beat.pid) for beat in beats} == {(3, os.getpid())}
    assert beats[-1].iteration == 17


def test_another_rank_has_the_runtime_end_a_hung_rank_after_the_grace(monkeypatch):
    reporter, killed, _ = _fence_reporter(monkeypatch, [Fence(2, 4242)])
    reporter.poll(0.0)
    assert killed == []  # the hung rank's own thread gets the first chance
    reporter.poll(2.0)
    _join_runtime_fence(reporter)
    # Through the runtime, by rank: never a signal to a pid this rank cannot vouch for.
    assert killed == [("runtime", 2)]


def test_the_runtime_fence_needs_a_pmix_runtime(monkeypatch):
    from tensorrt_llm._torch.distributed import pmix_fence

    monkeypatch.delenv("PMIX_NAMESPACE", raising=False)
    with pytest.raises(pmix_fence.PmixUnavailable):
        pmix_fence.kill_rank(2)


def test_pmix_ranks_translate_a_communicator_to_the_job():
    from mpi4py import MPI

    from tensorrt_llm._torch.distributed.pmix_fence import pmix_ranks

    assert pmix_ranks(MPI.COMM_SELF) == (MPI.COMM_WORLD.Get_rank(),)


def test_a_failed_runtime_fence_leaves_the_worker_running(monkeypatch):
    def refuse(rank):
        raise hang_evidence.PmixUnavailable("not launched by PMIx")

    reporter, killed, _ = _fence_reporter(monkeypatch, [Fence(2, 4242)], refuse)
    reporter.poll(0.0)
    reporter.poll(2.0)
    _join_runtime_fence(reporter)
    assert killed == []


class _FakeLibpmix:
    """The libpmix calls ``kill_rank`` makes, recording the job-control target."""

    def __init__(self, answered):
        self.targets = []
        self._answered = answered

    def PMIx_Info_create(self, n):
        return 1

    def PMIx_Info_load(self, info, key, value, type_):
        assert key == b"pmix.jctrl.kill"
        return 0

    def PMIx_Info_free(self, info, n):
        pass

    def PMIx_Job_control(self, targets, ntargets, directives, ndirectives, results, nresults):
        proc = targets._obj
        self.targets.append((proc.nspace, proc.rank))
        self._answered.wait()
        return 0


def test_the_runtime_fence_kills_the_job_rank_off_the_heartbeat_thread(monkeypatch):
    import threading

    from tensorrt_llm._torch.distributed import pmix_fence

    answered = threading.Event()
    lib = _FakeLibpmix(answered)
    monkeypatch.setenv("PMIX_NAMESPACE", "prterun-test@1")
    monkeypatch.setattr(pmix_fence, "_loaded_libpmix", lambda: lib)
    # Session rank 2 is job rank 6.
    reporter, killed, inbox = _fence_reporter(
        monkeypatch, [Fence(2, 4242)], runtime_kill=False, pmix_ranks=(4, 5, 6, 7)
    )
    reporter.poll(0.0)
    reporter.poll(2.0)
    # The runtime has not answered: heartbeats keep flowing and a second fence starts no second call.
    inbox.append(Fence(3, 4343))
    reporter.poll(3.0)
    reporter.poll(5.0)
    assert sum(isinstance(m, Heartbeat) for m in reporter.sent) == 4
    answered.set()
    _join_runtime_fence(reporter)
    assert lib.targets == [(b"prterun-test@1", 6)]
    assert killed == []


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_progress_row_reader_copies_the_row_beside_the_compute_stream():
    ep_size, row_bytes = 4, 256
    workspace = torch.zeros((ep_size, row_bytes), dtype=torch.uint8, device="cuda")
    row = workspace[1]
    row[0:4].view(torch.int32).fill_(9)
    row[64 : 64 + 16].view(torch.int32).copy_(torch.tensor([9, 9, 7, 9], dtype=torch.int32))
    row[128 : 128 + 16].view(torch.int32).fill_(8)
    reader = ProgressRowReader(workspace, 1, 0, 64, 128)
    assert reader.read() is None  # the first call starts the copy
    torch.cuda.synchronize()
    assert reader.read() == _row(9, [9, 9, 7, 9], [8] * 4)
