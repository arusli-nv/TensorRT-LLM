# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Worker-side hang evidence: lagging peers from the all-to-all progress row, and the reporter."""

import pytest
import torch

from tensorrt_llm._torch.pyexecutor.hang_evidence import (
    HangEvidenceReporter,
    ProgressRow,
    ProgressRowReader,
    lagging_peers,
)
from tensorrt_llm.executor.recovery.protocol import DeviceWait, Heartbeat

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
