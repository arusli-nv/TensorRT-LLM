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
"""Tests for MembershipMnnvlMemory.remap and HostExchangeMnnvlMemory against a fake CUDA driver.

A remap keeps the addresses, maps only the active rows, and puts the requested rows on fresh pages.
"""

import gc
import itertools
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

import tensorrt_llm._torch.distributed.membership_mnnvl_memory as membership_mnnvl
import tensorrt_llm._torch.distributed.mnnvl_memory as mnnvl
from tensorrt_llm._torch.distributed.host_exchange import HostExchangeMnnvlMemory
from tensorrt_llm._torch.distributed.membership import StoreGroup
from tensorrt_llm._torch.distributed.membership_mnnvl_memory import MembershipMnnvlMemory
from tensorrt_llm._torch.distributed.mnnvl_active_group import (
    MnnvlMappingError,
    MnnvlPeerHandleError,
)
from tensorrt_llm.mapping import Mapping

pytestmark = pytest.mark.cpu_only

_FABRIC = "fabric"
_SIZE = 1 << 21
_STRIDE = mnnvl.MnnvlMemory.fabric_page_size
_BASE = 1 << 40


class _FakeDriver:
    """Fake the CUDA driver calls of MnnvlMemory and record the handle mapped at each address."""

    CUresult = SimpleNamespace(CUDA_SUCCESS=0)
    CUmemAllocationHandleType = SimpleNamespace(CU_MEM_HANDLE_TYPE_FABRIC=_FABRIC)
    CUmemAccess_flags = SimpleNamespace(CU_MEM_ACCESS_FLAGS_PROT_READWRITE=3)

    def __init__(self):
        self._ids = itertools.count(100)
        self._bases = itertools.count(1)
        self.mapped = {}  # address -> handle
        self.released = []
        self.reserved = []  # (base, size)
        self.fail_create = False
        self.fail_import = False

    def CUmemAccessDesc(self):
        return SimpleNamespace()

    def CUdeviceptr(self, address):
        return address

    def cuCtxGetDevice(self):
        return 0, 0

    def cuMemAddressReserve(self, size, alignment, addr, flags):
        base = next(self._bases) << 44
        self.reserved.append((base, size))
        return 0, base

    def cuMemAddressFree(self, address, size):
        self.reserved.remove((address, size))
        return 0

    def cuMemCreate(self, size, prop, flags):
        if self.fail_create:
            return 2, None
        return 0, ("local", next(self._ids))

    def cuMemExportToShareableHandle(self, handle, handle_type, flags):
        return 0, SimpleNamespace(data=("export", handle))

    def cuMemImportFromShareableHandle(self, data, handle_type):
        if self.fail_import:
            return 2, None
        return 0, ("import", data)

    def cuMemMap(self, address, size, offset, handle, flags):
        assert address not in self.mapped, "address already mapped"
        self.mapped[address] = handle
        return 0

    def cuMemSetAccess(self, address, size, descs, count):
        return 0

    def cuMemUnmap(self, address, size):
        if self.mapped.pop(address, None) is None:
            return 1
        return 0

    def cuMemRelease(self, handle):
        self.released.append(handle)
        return 0

    def cuMemsetD8(self, address, value, size):
        assert address in self.mapped, "memset of an unmapped row"
        return 0


class _RemapMemory(MembershipMnnvlMemory):
    pass


@pytest.fixture
def driver(monkeypatch):
    driver = _FakeDriver()
    monkeypatch.setattr(mnnvl, "cuda", driver)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *args: None)
    monkeypatch.setattr(mnnvl.MnnvlMemory, "initialized", True)
    monkeypatch.setattr(mnnvl.MnnvlMemory, "dev_id", 0)
    monkeypatch.setattr(
        mnnvl.MnnvlMemory,
        "get_allocation_prop",
        classmethod(lambda cls, dev_id: SimpleNamespace(requestedHandleTypes=_FABRIC, location=0)),
    )
    return driver


@pytest.fixture
def memory(driver):
    """Return rank 1's view of a 4-rank allocation with every row mapped."""
    memory = object.__new__(_RemapMemory)
    memory.ptr, memory.segment_size, memory.rank_stride = _BASE, _SIZE, _STRIDE
    handles = [("old", rank) for rank in range(4)]
    for rank, handle in enumerate(handles):
        driver.mapped[_BASE + rank * _STRIDE] = handle
    _RemapMemory.allocated_map[_BASE] = mnnvl._MnnvlAllocationRecord(
        comm=None,
        comm_size=4,
        comm_rank=1,
        comm_membership=tuple(range(4)),
        aligned_size=_SIZE,
        mem_handles=list(handles),
        start_address=_BASE,
        rank_stride=_STRIDE,
        address_offset=0,
    )
    yield memory
    # The fake allocation owns no address range, so keep it from being closed when collected.
    del memory.ptr
    _RemapMemory.allocated_map.clear()


def _gather_without(absent, size=4):
    return lambda value: [None if rank in absent else value for rank in range(size)]


def _rows(*ranks, base=_BASE):
    return [base + rank * _STRIDE for rank in ranks]


_ALL, _SURVIVORS, _REPLACEMENT = frozenset(range(4)), frozenset({0, 1, 3}), frozenset({2})


def test_remap_maps_active_rows_in_place(memory, driver):
    memory.remap(_gather_without({2}), _ALL)

    assert sorted(driver.released) == sorted(("old", rank) for rank in range(4))
    assert memory.mapped
    assert sorted(driver.mapped) == _rows(0, 1, 3)
    assert driver.mapped[_BASE + _STRIDE][0] == "local"
    assert all(handle[0] != "old" for handle in driver.mapped.values())


@pytest.mark.parametrize("fresh_pages", [False, True])
def test_replacement_maps_own_row_until_admission(driver, monkeypatch, fresh_pages):
    """Check that a replacement maps only its own row until its admission maps every active row.

    At an isolated startup the group has the deployment's layout with only the replacement active. The admission maps
    every active row in place and keeps the replacement's pages unless the active ranks need fresh ones.
    """
    monkeypatch.setattr(mnnvl, "wide_ep_fault_tolerance_enabled", lambda: True)
    monkeypatch.setattr(
        _RemapMemory, "get_allocation_granularity", classmethod(lambda cls, dev_id: _SIZE)
    )
    group = StoreGroup(None, "workspace", 2, range(4), active=[2])
    monkeypatch.setattr(_RemapMemory, "get_comm", classmethod(lambda cls, mapping: group))
    memory = _RemapMemory(Mapping(world_size=4, rank=2, tp_size=4, moe_ep_size=4), _SIZE)
    try:
        rows = [memory.ptr + rank * memory.rank_stride for rank in range(4)]
        assert list(driver.mapped) == [rows[2]]
        record = _RemapMemory.allocated_map[memory.ptr]
        assert record.comm_size == 4
        assert record.comm_rank == 2
        assert record.comm_membership == (0, 1, 2, 3)
        own = driver.mapped[rows[2]]
        memory.remap(_gather_without(set()), _ALL if fresh_pages else frozenset())
        assert (driver.mapped[rows[2]] is own) is not fresh_pages
        assert sorted(driver.mapped) == rows
    finally:
        memory.checkpoint_prepare()
        _RemapMemory.allocated_map.clear()
        _RemapMemory.address_refcnt.clear()
        del memory.ptr


def test_release_skips_unmapped_absent_rows(memory, driver):
    memory.remap(_gather_without({2}), _ALL)
    memory.remap(_gather_without({0, 2}), _ALL)
    assert sorted(driver.mapped) == _rows(1, 3)


def test_remap_logs_one_timed_line(memory, driver, monkeypatch):
    clock = iter([10.0, 10.25])
    monkeypatch.setattr(membership_mnnvl, "time", SimpleNamespace(perf_counter=lambda: next(clock)))
    info = Mock()
    monkeypatch.setattr(mnnvl.logger, "info", info)
    memory.remap(_gather_without({2}), _ALL)
    info.assert_called_once_with(
        f"MNNVL remap of _RemapMemory at {_BASE:#x} among 3 active ranks: 3 rows on fresh pages, imported 2 rows "
        "in 250.0 ms"
    )


def test_failed_remap_is_retryable(memory, driver):
    driver.fail_create = True
    with pytest.raises(RuntimeError, match="handle export failed"):
        memory.remap(_gather_without({2}), _ALL)
    assert driver.mapped == {}
    assert not memory.mapped
    driver.fail_create = False
    memory.remap(_gather_without({2}), _ALL)
    assert memory.mapped


def test_admission_maps_only_joining_row(memory, driver):
    """Check rank 1's admission of rank 2's replacement after rank 2 failed.

    Rank 1 creates and releases nothing. It exports its own handle again, keeps every row it maps with the same
    handle, and imports and maps only rank 2's row.
    """
    memory.remap(_gather_without({2}), _ALL)
    kept = dict(driver.mapped)
    driver.released.clear()
    driver.fail_create = True
    memory.remap(_gather_without(set()), frozenset())
    assert sorted(driver.mapped) == _rows(0, 1, 2, 3)
    assert all(driver.mapped[address] is handle for address, handle in kept.items())
    (own,) = _rows(1)
    assert driver.mapped[_BASE + 2 * _STRIDE] == ("import", ("export", kept[own]))
    assert driver.released == []
    assert memory.mapped


def test_admission_refuses_row_of_departed_rank(memory, driver):
    """Check that an admission refuses to keep the pages of rank 3, which is no longer active.

    Only fresh pages keep its late writes out, so every row is released for the remap.
    """
    with pytest.raises(RuntimeError, match="Rank 3 left while this rank maps its pages"):
        memory.remap(_gather_without({3}), frozenset())
    assert driver.mapped == {}
    assert not memory.mapped
    memory.remap(_gather_without({3}), _ALL)
    assert sorted(driver.mapped) == _rows(0, 1, 2)


def test_failed_admission_releases_kept_pages(memory, driver):
    """Check that a failed admission releases every page rank 1 mapped, including its own.

    A peer may have imported rank 1's pages before the admission failed elsewhere, so the next change maps them on
    fresh pages.
    """
    with pytest.raises(MnnvlMappingError, match="rank 3: RuntimeError"):
        memory.remap(
            _gather_outcomes({3: ("RuntimeError: map failed", [])}, absent=set()), frozenset()
        )
    assert driver.mapped == {}
    assert not memory.mapped
    assert sorted(driver.released) == sorted(("old", rank) for rank in range(4))
    with pytest.raises(RuntimeError, match="row of an unmapped MNNVL allocation needs fresh pages"):
        memory.remap(_gather_without(set()), frozenset())
    memory.remap(_gather_without(set()), _ALL)
    assert memory.mapped


@pytest.mark.parametrize(
    "absent, peers, fresh_pages",
    [
        # Every process that maps the pages is active, so there is nothing to fence.
        (set(), {0: _ALL, 2: _ALL, 3: _ALL}, False),
        # Rank 3 failed, and its process may still write into the pages.
        (set(), {0: _ALL, 2: _ALL}, True),
        # Rank 2's process was replaced before the survivors remapped. Its new process maps only its own row, and
        # its old one may still write into the pages the others map.
        (set(), {0: _ALL, 2: _REPLACEMENT, 3: _ALL}, True),
        # Rank 2's replacement joins the survivors, which alone map their pages since rank 2 failed.
        ({2}, {0: _SURVIVORS, 2: _REPLACEMENT, 3: _SURVIVORS}, False),
        # An earlier admission left the pages unmapped on rank 3.
        ({2}, {0: _SURVIVORS, 2: _REPLACEMENT, 3: frozenset()}, True),
    ],
    ids=["unchanged", "failover", "replaced-holder", "admission", "unmapped-on-a-peer"],
)
def test_fresh_pages_only_for_inactive_mappers(memory, driver, absent, peers, fresh_pages):
    """Check that fresh pages are needed only when a process that maps them may not be active.

    Rank 1 maps the rows of every rank except ``absent``. Each rank of the change posts the ranks it mapped the
    allocation with, and every rank reaches the same answer from all of them.
    """
    if absent:
        memory.remap(_gather_without(absent), _ALL)
    posted = []

    def gather(value):
        posted.append(value)
        return [value if rank == 1 else peers.get(rank) for rank in range(4)]

    assert memory.stale_rows(gather) == (_ALL if fresh_pages else frozenset())
    assert posted == [_ALL - absent]


def test_an_unmapped_allocation_needs_fresh_pages(memory, driver):
    memory.checkpoint_prepare()
    assert memory.stale_rows(lambda value: [value] * 4) == _ALL


def test_admission_replaces_scratch_with_joining_row(rows_driver):
    """Check rank 2's admission on rank 1, which started with ranks 2 and 3 absent and their rows on scratch.

    Rank 1 keeps its own and rank 0's rows and imports rank 2's row in place of the scratch. Rank 3 is still absent,
    so its row goes on scratch again.
    """
    rows = HostExchangeMnnvlMemory(1, 4, _SIZE, _gather_without({2, 3}))
    kept = [rows_driver.mapped[rows.row(rank)] for rank in (0, 1)]
    scratch = rows_driver.mapped[rows.row(2)]

    rows.remap(_gather_without({3}), frozenset())

    assert [rows_driver.mapped[rows.row(rank)] for rank in (0, 1)] == kept
    assert rows_driver.mapped[rows.row(2)][0] == "import"
    assert rows_driver.mapped[rows.row(3)][0] == "local"
    assert rows_driver.mapped[rows.row(3)] not in (scratch, kept[1])
    assert rows_driver.released == [scratch]
    rows.close()


class _PeerFailure(Exception):
    pass


def _gather_outcomes(outcomes, absent, size=4):
    """Return a gather that runs the handle exchange without ``absent`` and then the outcome exchange.

    ``outcomes`` maps the other ranks to their outcomes, or is an exception that the outcome exchange raises.
    """
    calls = []

    def gather(value):
        calls.append(value)
        if len(calls) == 1:
            return [None if rank in absent else value for rank in range(size)]
        if isinstance(outcomes, Exception):
            raise outcomes
        return [value if rank == 1 else outcomes.get(rank) for rank in range(size)]

    return gather


def test_peer_mapping_failure_unmaps_this_rank(memory, driver):
    """Check that a peer's failed mapping makes every active rank raise, not only the peer."""
    memory.checkpoint_prepare()
    driver.released.clear()
    with pytest.raises(RuntimeError, match="rank 3: RuntimeError") as raised:
        memory.remap(_gather_outcomes({3: ("RuntimeError: map failed", [])}, absent={2}), _ALL)
    assert isinstance(raised.value, MnnvlMappingError)
    # The error names the failed rank, which alone reports the failure in a membership change.
    assert raised.value.ranks == (3,)
    assert driver.mapped == {}
    assert not memory.mapped
    # The fresh local handle and the imported ones were released.
    assert len(driver.released) == 3
    memory.remap(_gather_without({2}), _ALL)
    assert sorted(driver.mapped) == _rows(0, 1, 3)


def test_failed_import_reported_to_peers(memory, driver):
    memory.checkpoint_prepare()
    driver.fail_import = True
    reported = []

    def gather(value):
        reported.append(value)
        return [None if rank == 2 else value for rank in range(4)]

    with pytest.raises(MnnvlPeerHandleError, match="rank 1: MnnvlPeerHandleError") as raised:
        memory.remap(gather, _ALL)
    # Rank 1 failed on the first peer it imports, global rank 0. The CUDA result reaches every active rank.
    assert reported[-1][1] == [0]
    assert raised.value.ranks == (0,)
    assert "cuMemImportFromShareableHandle: 2" in reported[-1][0]
    assert driver.mapped == {}
    assert not memory.mapped


def test_only_failed_exporter_is_named(memory, driver):
    """Check that every active rank names only rank 3 when rank 3 could not create its fresh pages.

    Rank 1 learns of the failure in the handle exchange and reports no failure of its own.
    """
    memory.checkpoint_prepare()
    reported = []

    def gather(value):
        reported.append(value)
        # The first exchange carries the handles.
        if len(reported) == 1:
            failed = {**value, "error": "RuntimeError: 2", "handle": None}
            return [None if rank == 2 else failed if rank == 3 else value for rank in range(4)]
        theirs = ("MnnvlMappingError: rank 3: RuntimeError: 2", [])
        return [None if rank == 2 else theirs if rank == 3 else value for rank in range(4)]

    with pytest.raises(MnnvlMappingError, match="rank 3: MnnvlMappingError") as raised:
        memory.remap(gather, _ALL)
    assert raised.value.ranks == (3,)
    assert reported[-1] == ()
    assert driver.mapped == {}
    assert not memory.mapped


def test_any_failure_reaches_outcome_exchange(memory, driver, monkeypatch):
    """Check that a mapping error other than RuntimeError also reaches the peers waiting in the exchange."""
    memory.checkpoint_prepare()

    def failing_map(address, size, offset, handle, flags):
        raise ValueError("bad mapping")

    monkeypatch.setattr(driver, "cuMemMap", failing_map)
    reported = []

    def gather(value):
        reported.append(value)
        return [None if rank == 2 else value for rank in range(4)]

    with pytest.raises(RuntimeError, match="rank 1: ValueError: bad mapping"):
        memory.remap(gather, _ALL)
    assert len(reported) == 2
    assert reported[-1] == ("ValueError: bad mapping", [])
    assert driver.mapped == {}
    assert not memory.mapped


def test_peer_import_failure_names_global_rank(memory, driver):
    """Check global rank 5, communicator rank 1 of ranks 4 to 7, when rank 4 could not import rank 7's handle.

    Rank 5 maps its rows and then raises the same MnnvlPeerHandleError, which names global rank 7.
    """
    _RemapMemory.allocated_map[_BASE].comm_membership = (4, 5, 6, 7)
    memory.checkpoint_prepare()
    calls = []

    def gather(value):
        calls.append(value)
        if len(calls) == 1:
            return [value if rank in (4, 5, 7) else None for rank in range(8)]
        failure = ("MnnvlPeerHandleError: Cannot import", [7])
        return [{4: failure, 5: value}.get(rank) for rank in range(8)]

    with pytest.raises(MnnvlPeerHandleError, match="rank 4: MnnvlPeerHandleError") as raised:
        memory.remap(gather, frozenset(range(4, 8)))
    assert calls[-1] == ()
    assert raised.value.ranks == (7,)
    assert driver.mapped == {}
    assert not memory.mapped


def test_outcome_exchange_failure_leaves_unmapped(memory, driver):
    memory.checkpoint_prepare()
    with pytest.raises(_PeerFailure):
        memory.remap(_gather_outcomes(_PeerFailure(), absent={2}), _ALL)
    assert driver.mapped == {}
    assert not memory.mapped
    memory.remap(_gather_without({2}), _ALL)
    assert memory.mapped


def test_handle_exchange_failure_skips_outcome_exchange(memory, driver):
    """Check that a failed handle exchange reaches the recovery at once, without an outcome exchange."""
    memory.checkpoint_prepare()
    driver.released.clear()
    calls = []

    def gather(value):
        calls.append(value)
        raise _PeerFailure()

    with pytest.raises(_PeerFailure):
        memory.remap(gather, _ALL)
    assert len(calls) == 1
    assert driver.mapped == {}
    assert not memory.mapped
    assert len(driver.released) == 1
    memory.remap(_gather_without({2}), _ALL)
    assert memory.mapped


def test_failed_cleanup_breaks_allocation(memory, driver, monkeypatch):
    """Check that the recovery sees the remap's error even when releasing the fresh pages also fails.

    The cleanup failure is logged, and the allocation is never remapped again.
    """
    memory.checkpoint_prepare()
    errors = []
    monkeypatch.setattr(mnnvl.logger, "error", errors.append)
    monkeypatch.setattr(driver, "cuMemUnmap", lambda address, size: 2)

    with pytest.raises(_PeerFailure):
        memory.remap(_gather_outcomes(_PeerFailure(), absent={2}), _ALL)

    assert len(errors) == 1
    assert "failed MNNVL remap" in errors[0]
    assert _RemapMemory.allocated_map[_BASE].state is mnnvl._MnnvlAllocationState.BROKEN
    with pytest.raises(RuntimeError, match="Cannot remap MNNVL allocation in broken state"):
        memory.remap(_gather_without({2}), _ALL)


@pytest.fixture
def rows_driver(driver, monkeypatch):
    monkeypatch.setattr(HostExchangeMnnvlMemory, "allocation_granularity", _SIZE)
    yield driver
    HostExchangeMnnvlMemory.allocated_map.clear()
    HostExchangeMnnvlMemory.address_refcnt.clear()


def _exchange_once(active, size=4):
    """Return an exchange among the ``active`` ranks, each sending what this rank sends, and the list of its calls."""
    calls = []

    def gather(value):
        calls.append(value)
        return [value if rank in active else None for rank in range(size)]

    return gather, calls


def test_absent_rows_share_one_scratch_allocation(rows_driver):
    """Check rank 1 of 4 with ranks 2 and 3 absent.

    Every row is mapped back to back, and the absent ones share one local scratch allocation. The rows take one
    exchange of handles and one more once this rank's row is zeroed. Closing releases every allocation once and
    frees the range.
    """
    allgather, calls = _exchange_once(active={0, 1})
    rows = HostExchangeMnnvlMemory(1, 4, _SIZE - 100, allgather)

    assert len(calls) == 2
    assert calls[1] == ()
    assert rows.stride == _SIZE
    assert rows.row(3) == rows.ptr + 3 * _SIZE
    assert rows_driver.reserved == [(rows.ptr, 4 * _SIZE)]
    handles = [rows_driver.mapped[rows.row(rank)] for rank in range(4)]
    assert handles[0][0] == "import"
    assert handles[1][0] == "local"
    assert handles[2] is handles[3]
    assert handles[2][0] == "local"
    assert handles[2] != handles[1]

    rows.close()
    assert rows_driver.mapped == {}
    assert rows_driver.reserved == []
    assert sorted(rows_driver.released) == sorted(handles[:3])
    assert HostExchangeMnnvlMemory.allocated_map == {}


def test_membership_change_remaps_rows_in_place(rows_driver, monkeypatch):
    """Check rank 1 of 4 as rank 3 leaves and later joins again.

    Each change maps fresh pages at the same addresses, with rank 3's row on local scratch while it is absent. This
    rank's row is zeroed between the handle exchange and the outcome exchange, so no peer writes into it first.
    """
    events = []
    monkeypatch.setattr(
        rows_driver,
        "cuMemsetD8",
        lambda address, value, size: events.append(("zero", address)) or 0,
    )

    def gather_among(active):
        def gather(value):
            events.append(("outcome" if value == () else "handle", value))
            return [value if rank in active else None for rank in range(4)]

        return gather

    rows = HostExchangeMnnvlMemory(1, 4, _SIZE, gather_among({0, 1, 2, 3}))
    ptr = rows.ptr
    for active in ({0, 1, 2}, {0, 1, 2, 3}):
        before = [rows_driver.mapped[rows.row(rank)] for rank in range(4)]
        events.clear()
        rows.remap(gather_among(active), _ALL)
        after = [rows_driver.mapped[rows.row(rank)] for rank in range(4)]

        assert rows.ptr == ptr
        assert rows_driver.reserved == [(ptr, 4 * _SIZE)]
        assert [event for event, _ in events] == ["handle", "zero", "outcome"]
        assert events[1] == ("zero", rows.row(1))
        assert all(handle in rows_driver.released for handle in before)
        assert not set(after) & set(before)
        assert after[3][0] == ("import" if 3 in active else "local")
    rows.close()
    assert rows_driver.mapped == {}
    assert rows_driver.reserved == []


@pytest.mark.parametrize("failure", ["import", "memset"])
def test_failed_rows_leave_nothing_mapped(rows_driver, monkeypatch, failure):
    """Check that failed rows leave nothing mapped, allocated, or reserved.

    A handle that cannot be imported, because its rank exited after exporting it, raises MnnvlPeerHandleError naming
    that rank. By then this rank's own row, a scratch row, and an imported row are mapped. A later failure closes the
    rows too. Either way the rank still takes part in the outcome exchange, so its peers raise with it instead of
    waiting for it.
    """
    allgather, calls = _exchange_once(active={0, 1, 3})
    if failure == "import":
        imports = []

        def import_handle(data, handle_type):
            imports.append(data)
            return (2, None) if len(imports) == 2 else (0, ("import", data))

        monkeypatch.setattr(rows_driver, "cuMemImportFromShareableHandle", import_handle)
        expected = pytest.raises(
            MnnvlPeerHandleError, match=r"ranks \[3\]\ncuMemImportFromShareableHandle: 2"
        )
    else:
        monkeypatch.setattr(rows_driver, "cuMemsetD8", lambda address, value, size: 2)
        expected = pytest.raises(RuntimeError)

    with expected:
        HostExchangeMnnvlMemory(1, 4, _SIZE, allgather)

    assert rows_driver.mapped == {}
    assert rows_driver.reserved == []
    # This rank created its own row and the scratch row of rank 2.
    created = 2
    imported = 1 if failure == "import" else 2
    released = rows_driver.released
    assert len(released) == created + imported
    assert len({id(handle) for handle in released}) == len(released)
    assert HostExchangeMnnvlMemory.allocated_map == {}
    assert HostExchangeMnnvlMemory.address_refcnt == {}
    assert len(calls) == 2
    assert calls[1] is not None


def test_rows_fail_on_every_active_rank(rows_driver):
    """Check that rank 1 raises when rank 0 could not import rank 3's handle.

    Rank 1 mapped every row, yet it raises the same error and keeps nothing mapped, allocated, or reserved.
    """
    calls = []

    def allgather(payload):
        calls.append(payload)
        if len(calls) == 1:
            return [payload if rank in (0, 1, 3) else None for rank in range(4)]
        failed = ("MnnvlPeerHandleError: rank 3's handle", [3])
        return [failed if rank == 0 else payload if rank in (1, 3) else None for rank in range(4)]

    with pytest.raises(MnnvlPeerHandleError, match="rank 0: MnnvlPeerHandleError") as error:
        HostExchangeMnnvlMemory(1, 4, _SIZE, allgather)

    assert error.value.ranks == (3,)
    assert rows_driver.mapped == {}
    assert rows_driver.reserved == []
    assert HostExchangeMnnvlMemory.allocated_map == {}
    assert HostExchangeMnnvlMemory.address_refcnt == {}


def test_rows_name_only_failed_exporter(rows_driver):
    """Check that every active rank names only rank 0 when rank 0 could not create its row.

    Rank 1 learns of the failure in the handle exchange and reports no failure of its own. Nothing stays mapped,
    allocated, or reserved.
    """
    calls = []

    def allgather(payload):
        calls.append(payload)
        # The first exchange carries the handles.
        if len(calls) == 1:
            failed = {"error": "RuntimeError: 2", "handle": None, "is_fabric": True, "pid": None}
            return [failed if rank == 0 else payload if rank == 1 else None for rank in range(4)]
        theirs = ("MnnvlMappingError: rank 0: RuntimeError: 2", [])
        return [theirs if rank == 0 else payload if rank == 1 else None for rank in range(4)]

    with pytest.raises(MnnvlMappingError, match="rank 0: MnnvlMappingError") as error:
        HostExchangeMnnvlMemory(1, 4, _SIZE, allgather)

    assert error.value.ranks == (0,)
    assert calls[-1] == ()
    assert rows_driver.mapped == {}
    assert rows_driver.reserved == []
    assert HostExchangeMnnvlMemory.allocated_map == {}
    assert HostExchangeMnnvlMemory.address_refcnt == {}


def test_collecting_failed_rows_keeps_retry_open(rows_driver, monkeypatch):
    """Check that collecting failed rows leaves a retry at the same address mapped and open.

    The driver may reserve the failed construction's address range again for the retry while the failed object is
    still alive, here while its error is handled.
    """
    base = 1 << 44

    def reserve(size, alignment, addr, flags):
        rows_driver.reserved.append((base, size))
        return 0, base

    monkeypatch.setattr(rows_driver, "cuMemAddressReserve", reserve)
    allgather, _ = _exchange_once(active={0, 1})
    rows_driver.fail_import = True
    try:
        HostExchangeMnnvlMemory(1, 4, _SIZE, allgather)
    except MnnvlPeerHandleError:
        # The error's traceback keeps the failed object alive during the retry.
        rows_driver.fail_import = False
        rows = HostExchangeMnnvlMemory(1, 4, _SIZE, allgather)
    else:
        pytest.fail("HostExchangeMnnvlMemory did not raise for a handle it could not import")
    assert rows.ptr == base
    gc.collect()

    assert base in HostExchangeMnnvlMemory.allocated_map
    assert all(rows.row(rank) in rows_driver.mapped for rank in range(4))
    rows.close()
    assert rows_driver.mapped == {}
    assert rows_driver.reserved == []


def test_failed_scratch_release_not_repeated(rows_driver, monkeypatch):
    """Check closing rows when releasing the scratch allocation of absent ranks 2 and 3 fails.

    Closing unmaps every row, reports the failure, and does not release the shared scratch allocation again for the
    second absent row.
    """
    allgather, _ = _exchange_once(active={0, 1})
    rows = HostExchangeMnnvlMemory(1, 4, _SIZE, allgather)
    scratch = rows_driver.mapped[rows.row(2)]
    attempts = []

    def release(handle):
        attempts.append(handle)
        return 2 if handle is scratch else 0

    monkeypatch.setattr(rows_driver, "cuMemRelease", release)
    with pytest.raises(RuntimeError):
        rows.close()

    assert rows_driver.mapped == {}
    assert len(attempts) == 3
    assert sum(handle is scratch for handle in attempts) == 1
    assert not hasattr(rows, "ptr")


def test_rows_require_fabric_handles(rows_driver, monkeypatch):
    monkeypatch.setattr(
        mnnvl.MnnvlMemory,
        "get_allocation_prop",
        classmethod(
            lambda cls, dev_id: SimpleNamespace(requestedHandleTypes="posix_fd", location=0)
        ),
    )
    allgather, calls = _exchange_once(active={0, 1})
    with pytest.raises(RuntimeError, match="requires fabric memory handles"):
        HostExchangeMnnvlMemory(1, 4, _SIZE, allgather)
    assert calls == []
    assert rows_driver.reserved == []
