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
"""The collectives of ranks over the rendezvous store and the signing of its values.

Ranks run as threads over an in-memory store.
"""

import os
import pickle
import re
import threading

import pytest
from mpi4py import MPI
from torch.distributed import DistStoreError

from tensorrt_llm._torch.distributed import communicator
from tensorrt_llm._torch.distributed.membership import (
    SHARED_SECRET_ENV,
    SignedStore,
    StoreGroup,
    rendezvous,
)
from tensorrt_llm._torch.distributed.startup_record import StartupRecord
from tensorrt_llm._torch.pyexecutor import hang_detector

pytestmark = pytest.mark.cpu_only

JOIN_TIMEOUT_S = 10.0
_SHORT_SECRET = f"shared secret of at least 32 bytes (64 hex digits) in {SHARED_SECRET_ENV}"


class _Store:
    """In-memory store with the TCPStore calls of a store group, with signed 64-bit counters."""

    def __init__(self):
        self.values = {}
        self._lock = threading.Condition()

    def set(self, key, value):
        with self._lock:
            self.values[key] = bytes(value)
            self._lock.notify_all()

    def wait(self, keys, timeout):
        with self._lock:
            if not self._lock.wait_for(
                lambda: all(key in self.values for key in keys), timeout.total_seconds()
            ):
                raise DistStoreError(f"wait timeout after {timeout}, keys: {keys}")

    def get(self, key):
        return self.values[key]

    def check(self, keys):
        with self._lock:
            return all(key in self.values for key in keys)

    def multi_get(self, keys):
        with self._lock:
            return [self.values[key] for key in keys]

    def add(self, key, amount):
        with self._lock:
            value = int(self.values.get(key, b"0")) + amount
            assert -(1 << 63) <= value < 1 << 63, f"counter {key} overflows"
            self.values[key] = str(value).encode()
            self._lock.notify_all()
            return value

    def delete_key(self, key):
        with self._lock:
            return self.values.pop(key, None) is not None


def _group(store, rank, ranks, **options):
    return StoreGroup(store, "group", rank, ranks, **options)


def _run_ranks(calls):
    """Run each rank's call on its own thread, and return its result or exception by rank."""
    results = {}

    def run(rank, call):
        try:
            results[rank] = call()
        except Exception as error:
            results[rank] = error

    threads = [threading.Thread(target=run, args=item, daemon=True) for item in calls.items()]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(JOIN_TIMEOUT_S)
    assert not any(thread.is_alive() for thread in threads), "a rank did not return"
    return results


def test_split_orders_members_by_key():
    store = _Store()
    groups = {r: _group(store, r, range(4)) for r in range(4)}

    def split(rank):
        subgroup = groups[rank].Split(color=rank % 2, key=-rank)
        return subgroup.membership, subgroup.allgather(rank)

    results = _run_ranks({r: lambda r=r: split(r) for r in groups})
    assert results[0] == results[2] == ((2, 0), [2, 0])
    assert results[1] == results[3] == ((3, 1), [3, 1])


def test_replacement_replays_split_made_in_pause(monkeypatch):
    """Replay a split that rank 0 made while paused, which a replacement makes outside a pause."""
    store = _Store()
    for rank in range(1, 4):
        store.set(f"group/0/{rank}", pickle.dumps((rank % 2, rank)))
    original = StartupRecord()
    monkeypatch.setattr(communicator.Distributed, "startup_record", original)
    original.record()
    original.pause()
    assert _group(store, 0, range(4)).Split(color=0, key=0).membership == (0, 2)
    original.close()
    replacement = StartupRecord()
    replacement.replay(original.published())
    monkeypatch.setattr(communicator.Distributed, "startup_record", replacement)
    group = _group(_Store(), 2, range(4), active=[2], startup=True)
    assert group.Split(color=0, key=2).membership == (0, 2)
    replacement.close()


def test_split_type_groups_ranks_by_node():
    group = _group(_Store(), 3, range(6), gpus_per_node=2)
    assert group.Split_type(MPI.COMM_TYPE_SHARED).membership == (2, 3)
    with pytest.raises(NotImplementedError):
        group.Split_type(MPI.COMM_TYPE_SHARED + 1000)


@pytest.mark.parametrize("op, expected", [("SUM", 6), ("MIN", 1), ("MAX", 3)])
def test_allreduce_over_active_members(op, expected):
    store = _Store()
    values = {0: 3, 1: 1, 3: 2}
    groups = {r: _group(store, r, range(4), active=values) for r in values}
    results = _run_ranks(
        {r: lambda r=r: groups[r].allreduce(values[r], getattr(MPI, op)) for r in groups}
    )
    assert results == {r: expected for r in values}


def test_on_wait_error_fails_exchange():
    waiting = []

    def on_wait(mask):
        waiting.append(mask)
        if mask:
            raise TimeoutError("rank 1 stopped")

    group = _group(_Store(), 0, (0, 1), on_wait=on_wait)
    with pytest.raises(TimeoutError):
        group.allgather(0)
    assert waiting == [0b10, 0]
    assert group.exchange_failed


def test_exchange_failed_tells_latest_exchange():
    """Check that ``exchange_failed`` tells whether the latest exchange raised."""
    failing = [True]

    def on_wait(mask):
        if failing[0]:
            raise TimeoutError("stopped waiting")

    group = _group(_Store(), 0, (0,), on_wait=on_wait)
    with pytest.raises(TimeoutError):
        group.allgather(0)
    assert group.exchange_failed
    failing[0] = False
    assert group.allgather(1) == [1]
    assert not group.exchange_failed


def test_exchange_keeps_only_latest_call_keys():
    """Delete a call's keys once the next call completes, so only the latest call's keys stay."""
    store = _Store()
    groups = {r: _group(store, r, (0, 1)) for r in (0, 1)}
    _run_ranks({r: lambda r=r: [groups[r].allgather(call) for call in range(3)] for r in groups})
    assert sorted(store.values) == ["group/2/0", "group/2/1", "group/2/arrived/0"]


def test_waiting_mask_beyond_first_counter(monkeypatch):
    """Report a missing rank 64, whose arrival bit is in the second counter."""
    monkeypatch.setattr(rendezvous, "REPORT_INTERVAL_S", 0.01)
    store = _Store()
    waiting = {0: [], 63: []}
    rank_63_waits = threading.Event()

    def on_wait(rank):
        def report(mask):
            waiting[rank].append(mask)
            if rank == 63:
                rank_63_waits.set()
            if mask == 1 << 64:
                raise TimeoutError("rank 64 stopped")

        return report

    groups = {r: _group(store, r, (0, 63, 64), on_wait=on_wait(r)) for r in (0, 63)}
    results = _run_ranks(
        {
            0: lambda: rank_63_waits.wait(10) and groups[0].allgather(0),
            63: lambda: groups[63].allgather(63),
        }
    )
    assert all(isinstance(result, TimeoutError) for result in results.values())
    assert waiting[0] == [1 << 64, 0]


def test_missing_communicator_method_named():
    """Name the communicator methods the group has, for one that main's startup code calls."""
    group = _group(_Store(), 0, (0, 1))
    with pytest.raises(AttributeError, match="has no Dup. .* implements only Get_rank"):
        group.Dup()
    assert not hasattr(group, "Iprobe")


def test_hard_kill_ends_only_this_rank(monkeypatch):
    """Check that the hang detector's hard kill through the group exits only this process."""
    exits, kills = [], []
    monkeypatch.setattr(os, "_exit", exits.append)
    monkeypatch.setattr(os, "kill", lambda pid, sig: kills.append(sig))
    monkeypatch.setattr(hang_detector, "ENABLE_MULTI_DEVICE", True)
    monkeypatch.setattr(hang_detector, "mpi_disabled", lambda: False)
    monkeypatch.setattr(hang_detector, "mpi_comm", lambda: _group(_Store(), 1, (0, 1)))
    monkeypatch.setattr(MPI, "Is_initialized", lambda: True)
    monkeypatch.setattr(MPI, "Is_finalized", lambda: False)
    monkeypatch.setattr(MPI, "Query_thread", lambda: MPI.THREAD_MULTIPLE)
    hang_detector.propagate_hard_kill(3, diagnostics=False)
    assert exits == [3]
    assert kills == []


class _Unpickled:
    """A value whose unpickling runs code, as a forged pickle in the store would."""

    calls = []

    @staticmethod
    def record():
        _Unpickled.calls.append("unpickled")

    def __reduce__(self):
        return (_Unpickled.record, ())


@pytest.mark.parametrize("secret", ["ab" * 31, "zz" * 32], ids=["short", "not_hex"])
def test_store_client_requires_ranks_secret(secret):
    """Refuse a store client, as of the node supervisor or the authority, with a weak secret."""
    with pytest.raises(ValueError, match=re.escape(_SHORT_SECRET)):
        SignedStore(_Store(), secret)


@pytest.mark.parametrize("forger_secret", [None, "cd" * 32], ids=["unsigned", "other_secret"])
def test_forged_store_value_is_never_unpickled(monkeypatch, forger_secret):
    """Reject a pickle that a process without the deployment's secret set, before unpickling it."""
    store = _Store()
    forger = store if forger_secret is None else SignedStore(store, forger_secret)
    signed = SignedStore(store, "ab" * 32)
    monkeypatch.setattr(_Unpickled, "calls", [])
    forged = pickle.dumps(_Unpickled())
    # Rank 1's value and arrival at the first call of a group of ranks 0 and 1.
    forger.set("forged/0/1", forged)
    forger.add("forged/0/arrived/0", 1 << 1)
    with pytest.raises(ValueError, match="is not signed with this deployment's"):
        StoreGroup(signed, "forged", 0, [0, 1]).allgather("rank 0")
    assert _Unpickled.calls == []
    # The deployment's own values pass.
    signed.set("own", forged)
    assert signed.get("own") == forged
