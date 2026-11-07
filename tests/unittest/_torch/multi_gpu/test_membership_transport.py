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
"""The authority process and the ranks' agents over shared GPU memory: a rank that exits is removed, a replacement
joins, and a store value without the join secret is rejected."""

import multiprocessing as mp
import os
import queue
import time
from datetime import timedelta

import pytest
import torch

WORLD = 4
SECRET = b"test-join-secret"


def _store(port, secret=SECRET):
    from torch.distributed import TCPStore

    from tensorrt_llm._torch.distributed.membership import SignedStore

    return SignedStore(
        TCPStore("127.0.0.1", port, is_master=False, timeout=timedelta(seconds=60)), secret
    )


def _device(rank):
    return rank % torch.cuda.device_count()


def _authority(port, results):
    from tensorrt_llm._torch.distributed.membership import Authority, AuthorityProcess

    torch.cuda.set_device(0)
    authority = Authority(WORLD, covers=lambda mask: True, liveness_timeout_s=0.5)
    results.put(("authority", AuthorityProcess(WORLD, _store(port), 0, authority).run().state.name))


def _rank(port, rank, incarnation, exit_after_s, results):
    """Follow the records and answer them like an executor would; exit at once after ``exit_after_s``."""
    from tensorrt_llm._torch.distributed.membership import (
        SLOT_BYTES,
        MembershipAgent,
        MembershipState,
        Reply,
        SharedRegion,
    )

    device = _device(rank)
    torch.cuda.set_device(device)
    store = _store(port)
    row = SharedRegion(SLOT_BYTES, device)
    if not incarnation:
        store.set(f"row/{rank}", row.handle_bytes)
    inbox = SharedRegion(WORLD * SLOT_BYTES, device, store.get("inbox"))
    agent = None
    membership = {"admitted": False, "excluded": False}

    def on_record(record):
        if record.is_live(rank):
            membership["admitted"] = True
        elif membership["admitted"]:
            membership["excluded"] = True
        if record.state == MembershipState.PROPOSE:
            agent.report(reply=Reply.READY, reply_generation=record.generation)
        elif record.state == MembershipState.INSTALL:
            agent.report(reply=Reply.INSTALLED, reply_generation=record.generation)
        results.put((rank, incarnation, record.generation, record.state.name, record.live_mask))

    agent = MembershipAgent(rank, device, row, inbox, on_record, incarnation=incarnation)
    agent.start()
    if incarnation:
        store.set(f"join/{rank}/{incarnation}", row.handle_bytes)
    start = time.monotonic()
    step = 0
    done = (MembershipState.SHUTDOWN, MembershipState.ABORT)
    while not membership["excluded"] and (agent.record is None or agent.record.state not in done):
        if exit_after_s is not None and time.monotonic() - start > exit_after_s:
            results.close()
            results.join_thread()  # deliver what was reported, then end with no cleanup, like a crash
            os._exit(0)
        step += 1
        agent.report(step=step)
        time.sleep(0.01)
    agent.stop()


def _until(results, predicate, timeout_s=60):
    seen = []
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            item = results.get(timeout=1)
        except queue.Empty:
            continue
        seen.append(item)
        if predicate(item):
            return seen
    raise AssertionError(f"timed out; seen {seen}")


@pytest.fixture
def cluster():
    from tensorrt_llm._torch.distributed.mnnvl_memory import MnnvlMemory

    if not torch.cuda.is_available() or not MnnvlMemory.supports_mnnvl():
        pytest.skip("needs GPUs with fabric memory")
    from torch.distributed import TCPStore

    ctx = mp.get_context("spawn")
    store = TCPStore("127.0.0.1", 0, is_master=True, wait_for_workers=False)
    results = ctx.Queue()
    procs = []

    def start(target, *args):
        proc = ctx.Process(target=target, args=(store.port, *args, results))
        proc.start()
        procs.append(proc)

    yield store, results, start
    from tensorrt_llm._torch.distributed.membership import SignedStore

    SignedStore(store, SECRET).set("shutdown", b"")
    for proc in procs:
        proc.join(timeout=30)
        if proc.is_alive():
            proc.kill()


def test_a_rank_that_exits_is_removed_and_its_replacement_joins(cluster):
    _, results, start = cluster
    start(_authority)
    for r in range(WORLD):
        start(_rank, r, 0, 2.0 if r == 3 else None)
    # Items are (rank, incarnation, generation, state, live mask).
    seen = _until(results, lambda item: item[0] == 0 and item[2:] == (1, "COMMITTED", 0b0111))
    assert (0, 0, 1, "PROPOSE", 0b0111) in seen and (0, 0, 1, "INSTALL", 0b0111) in seen

    start(_rank, 3, 1, None)
    _until(results, lambda item: item[0] == 0 and item[2:] == (2, "COMMITTED", 0b1111))

    # A replacement for rank 2 while its old process still runs: the old process reads its exclusion.
    start(_rank, 2, 1, None)
    seen = _until(results, lambda item: item[0] == 0 and item[2:] == (3, "COMMITTED", 0b1111))
    assert any(item[:2] == (2, 0) and not item[4] >> 2 & 1 for item in seen)


def test_a_value_without_the_join_secret_is_rejected():
    from torch.distributed import TCPStore

    from tensorrt_llm._torch.distributed.membership import SignedStore

    store = TCPStore("127.0.0.1", 0, is_master=True, wait_for_workers=False)
    SignedStore(store, b"another-secret").set("row/0", b"forged")
    with pytest.raises(PermissionError):
        SignedStore(store, SECRET).get("row/0")
    # A signed ("row/1", "2handle") does not verify as ("row/12", "handle").
    SignedStore(store, SECRET).set("row/1", b"2handle")
    raw = store.get("row/1")
    store.set("row/12", raw[:32] + raw[33:])
    with pytest.raises(PermissionError):
        SignedStore(store, SECRET).get("row/12")
