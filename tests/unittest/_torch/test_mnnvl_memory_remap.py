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
"""MnnvlMemory.remap and MnnvlRows against a fake CUDA driver: same addresses, live rows only."""

import gc
import itertools
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

import tensorrt_llm._torch.distributed.mnnvl_memory as mnnvl
from tensorrt_llm.mapping import Mapping

pytestmark = pytest.mark.cpu_only

_FABRIC = "fabric"
_SIZE = 1 << 21
_STRIDE = mnnvl.MnnvlMemory.fabric_page_size
_BASE = 1 << 40


class _FakeDriver:
    """The cu* calls MnnvlMemory makes, recording page ownership per mapped address."""

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


class _RemapMemory(mnnvl.MnnvlMemory):
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
    """Rank 1's view of a 4-rank allocation whose every row is mapped."""
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
    # The fake allocation owns no address range; keep it from being closed when collected.
    del memory.ptr
    _RemapMemory.allocated_map.clear()


def _gather_without(absent, size=4):
    return lambda value: [None if rank in absent else value for rank in range(size)]


def _rows(*ranks, base=_BASE):
    return [base + rank * _STRIDE for rank in ranks]


def test_remap_maps_fresh_pages_at_same_addresses_for_live_rows_only(memory, driver):
    memory.checkpoint_prepare()
    assert driver.mapped == {}
    assert sorted(driver.released) == sorted(("old", rank) for rank in range(4))

    memory.remap(_gather_without({2}))

    assert memory.mapped
    assert sorted(driver.mapped) == _rows(0, 1, 3)
    assert driver.mapped[_BASE + _STRIDE][0] == "local"
    assert all(handle[0] != "old" for handle in driver.mapped.values())


@pytest.mark.parametrize("fresh_pages", [False, True])
def test_a_replacement_maps_only_its_own_row_until_its_admission_maps_every_live_row(
    driver, monkeypatch, fresh_pages
):
    """Isolated startup: the class's group has the deployment's layout with only the replacement live, so only its
    row is mapped; the admission maps every live row in place, keeping the replacement's pages unless the live ranks
    need fresh ones."""
    monkeypatch.setattr(
        _RemapMemory, "get_allocation_granularity", classmethod(lambda cls, dev_id: _SIZE)
    )
    group = mnnvl.LiveGroupComm((0, 1, 2, 3), 2, mnnvl.isolated_gather(2, 4))
    monkeypatch.setattr(_RemapMemory, "get_comm", classmethod(lambda cls, mapping: group))
    memory = _RemapMemory(Mapping(world_size=4, rank=2, tp_size=4, moe_ep_size=4), _SIZE)
    try:
        rows = [memory.ptr + rank * memory.rank_stride for rank in range(4)]
        assert list(driver.mapped) == [rows[2]]
        record = _RemapMemory.allocated_map[memory.ptr]
        assert (record.comm_size, record.comm_rank, record.comm_membership) == (4, 2, (0, 1, 2, 3))
        assert memory.holders == {2}
        own = driver.mapped[rows[2]]
        if fresh_pages:
            memory.checkpoint_prepare()
            memory.remap(_gather_without(set()))
        else:
            memory.admit(_gather_without(set()))
            assert driver.mapped[rows[2]] is own
        assert sorted(driver.mapped) == rows and memory.holders == {0, 1, 2, 3}
    finally:
        memory.checkpoint_prepare()
        _RemapMemory.allocated_map.clear()
        _RemapMemory.address_refcnt.clear()
        del memory.ptr


def test_release_after_remap_skips_unmapped_absent_rows(memory, driver):
    memory.checkpoint_prepare()
    memory.remap(_gather_without({2}))
    memory.checkpoint_prepare()
    assert driver.mapped == {}
    memory.remap(_gather_without({0, 2}))
    assert sorted(driver.mapped) == _rows(1, 3)


def test_a_membership_change_logs_one_line_with_its_time(memory, driver, monkeypatch):
    clock = iter([10.0, 10.25])
    monkeypatch.setattr(mnnvl, "time", SimpleNamespace(perf_counter=lambda: next(clock)))
    info = Mock()
    monkeypatch.setattr(mnnvl.logger, "info", info)
    memory.checkpoint_prepare()
    memory.remap(_gather_without({2}))
    info.assert_called_once_with(
        f"MNNVL remap of _RemapMemory at {_BASE:#x} among 3 live ranks: imported 2 rows in 250.0 ms"
    )


def test_failed_remap_leaves_allocation_unmapped_and_retryable(memory, driver):
    memory.checkpoint_prepare()
    driver.fail_create = True
    with pytest.raises(RuntimeError, match="handle export failed"):
        memory.remap(_gather_without({2}))
    assert driver.mapped == {}
    assert not memory.mapped
    driver.fail_create = False
    memory.remap(_gather_without({2}))
    assert memory.mapped


def test_remap_requires_released_backing(memory):
    with pytest.raises(RuntimeError, match="Cannot remap"):
        memory.remap(_gather_without(set()))


def test_an_admission_keeps_every_mapped_row_and_maps_only_the_joining_rank_s_row(memory, driver):
    """Rank 1 maps the rows of ranks 0, 1 and 3 after rank 2 failed; rank 2's replacement joins. Rank 1 creates and
    releases nothing: it exports its own handle again, keeps every row it maps with the same handle, and imports
    and maps rank 2's row only."""
    memory.checkpoint_prepare()
    memory.remap(_gather_without({2}))
    assert memory.holders == {0, 1, 3}
    kept = dict(driver.mapped)
    driver.released.clear()
    driver.fail_create = True
    memory.admit(_gather_without(set()))
    assert sorted(driver.mapped) == _rows(0, 1, 2, 3)
    assert all(driver.mapped[address] is handle for address, handle in kept.items())
    (own,) = _rows(1)
    assert driver.mapped[_BASE + 2 * _STRIDE] == ("import", ("export", kept[own]))
    assert driver.released == []
    assert memory.mapped and memory.holders == {0, 1, 2, 3}


def test_an_admission_refuses_a_row_whose_rank_left(memory, driver):
    """Rank 3, whose pages rank 1 maps, is not live: only fresh pages keep its late writes out. Every row is
    released, for the remap."""
    with pytest.raises(RuntimeError, match="Rank 3 left while this rank maps its pages"):
        memory.admit(_gather_without({3}))
    assert driver.mapped == {} and not memory.mapped and memory.holders is None
    memory.remap(_gather_without({3}))
    assert sorted(driver.mapped) == _rows(0, 1, 2)


def test_a_failed_admission_releases_the_kept_pages_too(memory, driver):
    """A peer may have imported rank 1's pages before the admission failed elsewhere: rank 1 releases every page
    it mapped, its own included, so that the next change remaps."""
    with pytest.raises(mnnvl.MnnvlMappingError, match="rank 3: RuntimeError"):
        memory.admit(_gather_outcomes({3: ("RuntimeError: map failed", [])}, absent=set()))
    assert driver.mapped == {} and not memory.mapped and memory.holders is None
    assert sorted(driver.released) == sorted(("old", rank) for rank in range(4))
    with pytest.raises(
        RuntimeError, match="Cannot admit ranks to MNNVL allocation in unmapped state"
    ):
        memory.admit(_gather_without(set()))
    memory.remap(_gather_without(set()))
    assert memory.mapped


_ALL, _SURVIVORS, _REPLACEMENT = frozenset(range(4)), frozenset({0, 1, 3}), frozenset({2})


@pytest.mark.parametrize(
    "absent, peers, fresh_pages",
    [
        # Every process that maps the pages is live: nothing to fence.
        (set(), {0: _ALL, 2: _ALL, 3: _ALL}, False),
        # Rank 3 failed: its process may still write into the pages.
        (set(), {0: _ALL, 2: _ALL}, True),
        # Rank 2's process was replaced before the survivors remapped: its new process maps only its own row, and
        # its old one may still write into the pages the others map.
        (set(), {0: _ALL, 2: _REPLACEMENT, 3: _ALL}, True),
        # Rank 2's replacement joins the survivors, which alone map their pages since rank 2 failed.
        ({2}, {0: _SURVIVORS, 2: _REPLACEMENT, 3: _SURVIVORS}, False),
        # An earlier admission left the pages unmapped on rank 3.
        ({2}, {0: _SURVIVORS, 2: _REPLACEMENT, 3: frozenset()}, True),
    ],
    ids=["unchanged", "failover", "replaced-holder", "admission", "unmapped-on-a-peer"],
)
def test_fresh_pages_are_needed_only_when_a_process_that_maps_them_may_not_be_live(
    memory, driver, absent, peers, fresh_pages
):
    """Rank 1 maps the rows of every rank but ``absent`` (since a remap without them); the ranks of the change post
    their holders, and every rank answers alike from all of them."""
    if absent:
        memory.checkpoint_prepare()
        memory.remap(_gather_without(absent))
    posted = []

    def gather(value):
        posted.append(value)
        return [value if rank == 1 else peers.get(rank) for rank in range(4)]

    assert memory.needs_fresh_pages(gather) is fresh_pages
    assert posted == [memory.holders]


def test_an_unmapped_allocation_needs_fresh_pages(memory, driver):
    memory.checkpoint_prepare()
    assert memory.needs_fresh_pages(lambda value: [value] * 4)


def test_an_admission_maps_a_joining_rank_s_row_where_scratch_was(rows_driver):
    """Rank 1 of 4 started with ranks 2 and 3 absent, their rows on scratch, which no other process maps. Rank 2
    joins: rank 1 keeps its own and rank 0's rows, imports rank 2's in place of scratch, and maps rank 3's, still
    absent, on scratch again."""
    rows = mnnvl.MnnvlRows(1, 4, _SIZE, _gather_without({2, 3}))
    assert rows.holders == {0, 1}
    kept = [rows_driver.mapped[rows.row(rank)] for rank in (0, 1)]
    scratch = rows_driver.mapped[rows.row(2)]

    rows.admit(_gather_without({3}))

    assert [rows_driver.mapped[rows.row(rank)] for rank in (0, 1)] == kept
    assert rows_driver.mapped[rows.row(2)][0] == "import"
    assert rows_driver.mapped[rows.row(3)][0] == "local"
    assert rows_driver.mapped[rows.row(3)] not in (scratch, kept[1])
    assert rows_driver.released == [scratch] and rows.holders == {0, 1, 2}
    rows.close()


class _PeerFailure(Exception):
    pass


def _gather_outcomes(outcomes, absent, size=4):
    """The handle exchange without ``absent``, then the remap outcomes: a dict rank -> outcome for the other
    ranks, or an exception the outcome exchange raises."""
    calls = []

    def gather(value):
        calls.append(value)
        if len(calls) == 1:
            return [None if rank in absent else value for rank in range(size)]
        if isinstance(outcomes, Exception):
            raise outcomes
        return [value if rank == 1 else outcomes.get(rank) for rank in range(size)]

    return gather


def test_a_peer_s_failed_mapping_unmaps_this_rank_too(memory, driver):
    """A rank whose import of a failing rank's handle failed makes every live rank raise, not only itself."""
    memory.checkpoint_prepare()
    driver.released.clear()
    with pytest.raises(RuntimeError, match="rank 3: RuntimeError") as raised:
        memory.remap(_gather_outcomes({3: ("RuntimeError: map failed", [])}, absent={2}))
    # It names the rank that failed, which alone reports the failure in a membership change.
    assert isinstance(raised.value, mnnvl.MnnvlMappingError) and raised.value.ranks == (3,)
    assert driver.mapped == {} and not memory.mapped
    # The fresh local handle and the imported ones were released.
    assert len(driver.released) == 3
    memory.remap(_gather_without({2}))
    assert sorted(driver.mapped) == _rows(0, 1, 3)


def test_this_rank_s_failed_import_is_reported_to_the_others(memory, driver):
    memory.checkpoint_prepare()
    driver.fail_import = True
    reported = []

    def gather(value):
        reported.append(value)
        return [None if rank == 2 else value for rank in range(4)]

    with pytest.raises(mnnvl.MnnvlPeerHandleError, match="rank 1: MnnvlPeerHandleError") as raised:
        memory.remap(gather)
    # Rank 1 failed on the first peer it imports, global rank 0; the CUDA result reaches every live rank.
    assert reported[-1][1] == [0] and raised.value.ranks == (0,)
    assert "cuMemImportFromShareableHandle: 2" in reported[-1][0]
    assert driver.mapped == {} and not memory.mapped


def test_only_the_rank_whose_export_failed_is_named(memory, driver):
    """Rank 3 could not create its fresh pages: every live rank raises an error naming rank 3 only, and rank 1, which
    learnt of it in the handle exchange, reports no failure of its own."""
    memory.checkpoint_prepare()
    reported = []

    def gather(value):
        reported.append(value)
        if len(reported) == 1:  # the handles
            failed = {**value, "error": "RuntimeError: 2", "handle": None}
            return [None if rank == 2 else failed if rank == 3 else value for rank in range(4)]
        theirs = ("MnnvlMappingError: rank 3: RuntimeError: 2", [])
        return [None if rank == 2 else theirs if rank == 3 else value for rank in range(4)]

    with pytest.raises(mnnvl.MnnvlMappingError, match="rank 3: MnnvlMappingError") as raised:
        memory.remap(gather)
    assert raised.value.ranks == (3,) and reported[-1] == ()
    assert driver.mapped == {} and not memory.mapped


def test_a_failure_of_any_type_still_reaches_the_outcome_exchange(memory, driver, monkeypatch):
    """A mapping error that is not a RuntimeError is reported to the live peers waiting in the exchange too."""
    memory.checkpoint_prepare()

    def failing_map(address, size, offset, handle, flags):
        raise ValueError("bad mapping")

    monkeypatch.setattr(driver, "cuMemMap", failing_map)
    reported = []

    def gather(value):
        reported.append(value)
        return [None if rank == 2 else value for rank in range(4)]

    with pytest.raises(RuntimeError, match="rank 1: ValueError: bad mapping"):
        memory.remap(gather)
    assert len(reported) == 2 and reported[-1] == ("ValueError: bad mapping", [])
    assert driver.mapped == {} and not memory.mapped


def test_a_peer_s_failed_import_raises_the_lost_global_rank_on_every_live_rank(memory, driver):
    """Global rank 5 (communicator rank 1 of 4..7) maps its rows, but rank 4 could not import rank 7's
    handle: rank 5 raises the same MnnvlPeerHandleError, naming global rank 7."""
    _RemapMemory.allocated_map[_BASE].comm_membership = (4, 5, 6, 7)
    memory.checkpoint_prepare()
    calls = []

    def gather(value):
        calls.append(value)
        if len(calls) == 1:
            return [value if rank in (4, 5, 7) else None for rank in range(8)]
        failure = ("MnnvlPeerHandleError: Cannot import", [7])
        return [{4: failure, 5: value}.get(rank) for rank in range(8)]

    with pytest.raises(mnnvl.MnnvlPeerHandleError, match="rank 4: MnnvlPeerHandleError") as raised:
        memory.remap(gather)
    assert calls[-1] == () and raised.value.ranks == (7,)
    assert driver.mapped == {} and not memory.mapped


def test_a_peer_failure_during_the_outcome_exchange_leaves_the_allocation_unmapped(memory, driver):
    memory.checkpoint_prepare()
    with pytest.raises(_PeerFailure):
        memory.remap(_gather_outcomes(_PeerFailure(), absent={2}))
    assert driver.mapped == {} and not memory.mapped
    memory.remap(_gather_without({2}))
    assert memory.mapped


def test_a_peer_failure_during_the_handle_exchange_propagates_without_the_outcome_exchange(
    memory, driver
):
    """The exchange that failed could not report an outcome either: the failure reaches the recovery at once."""
    memory.checkpoint_prepare()
    driver.released.clear()
    calls = []

    def gather(value):
        calls.append(value)
        raise _PeerFailure()

    with pytest.raises(_PeerFailure):
        memory.remap(gather)
    assert len(calls) == 1
    assert driver.mapped == {} and not memory.mapped
    assert len(driver.released) == 1
    memory.remap(_gather_without({2}))
    assert memory.mapped


def test_a_failed_cleanup_keeps_the_remap_s_error_and_breaks_the_allocation(
    memory, driver, monkeypatch
):
    """A peer failure during the outcome exchange stays the error the recovery sees, even if releasing the fresh
    pages fails too; that failure is logged and the allocation is not remapped again."""
    memory.checkpoint_prepare()
    errors = []
    monkeypatch.setattr(mnnvl.logger, "error", errors.append)
    monkeypatch.setattr(driver, "cuMemUnmap", lambda address, size: 2)

    with pytest.raises(_PeerFailure):
        memory.remap(_gather_outcomes(_PeerFailure(), absent={2}))

    assert len(errors) == 1 and "failed MNNVL remap" in errors[0]
    assert _RemapMemory.allocated_map[_BASE].state is mnnvl._MnnvlAllocationState.BROKEN
    with pytest.raises(RuntimeError, match="Cannot remap MNNVL allocation in broken state"):
        memory.remap(_gather_without({2}))


@pytest.fixture
def rows_driver(driver, monkeypatch):
    monkeypatch.setattr(mnnvl.MnnvlRows, "allocation_granularity", _SIZE)
    yield driver
    mnnvl.MnnvlRows.allocated_map.clear()
    mnnvl.MnnvlRows.address_refcnt.clear()


def _exchange_once(live, size=4):
    """``MnnvlRows``' exchange among the ``live`` ranks, each sending what this rank sends; records each call."""
    calls = []

    def gather(value):
        calls.append(value)
        return [value if rank in live else None for rank in range(size)]

    return gather, calls


def test_mnnvl_rows_sit_back_to_back_with_absent_rows_on_one_scratch_allocation(rows_driver):
    """Rank 1 of 4 with ranks 2 and 3 absent: every row is mapped, the absent ones on one local scratch
    allocation, after one exchange of handles and one once the rank's row is zeroed; closing releases every
    allocation once and frees the range."""
    allgather, calls = _exchange_once(live={0, 1})
    rows = mnnvl.MnnvlRows(1, 4, _SIZE - 100, allgather)

    assert len(calls) == 2 and calls[1] == ()
    assert rows.stride == _SIZE and rows.row(3) == rows.ptr + 3 * _SIZE
    assert rows_driver.reserved == [(rows.ptr, 4 * _SIZE)]
    handles = [rows_driver.mapped[rows.row(rank)] for rank in range(4)]
    assert handles[0][0] == "import" and handles[1][0] == "local"
    assert handles[2] is handles[3] and handles[2][0] == "local" and handles[2] != handles[1]

    rows.close()
    assert rows_driver.mapped == {} and rows_driver.reserved == []
    assert sorted(rows_driver.released) == sorted(handles[:3])
    assert mnnvl.MnnvlRows.allocated_map == {}


def test_a_membership_change_remaps_mnnvl_rows_at_the_same_addresses(rows_driver, monkeypatch):
    """Rank 1 of 4 loses rank 3, which later joins again: each change maps fresh pages at the same addresses, rank
    3's row on local scratch while it is absent, and zeroes this rank's row between the handle exchange and the
    outcome exchange, so that no peer writes into it before it is zeroed."""
    events = []
    monkeypatch.setattr(
        rows_driver,
        "cuMemsetD8",
        lambda address, value, size: events.append(("zero", address)) or 0,
    )

    def gather_among(live):
        def gather(value):
            events.append(("outcome" if value == () else "handle", value))
            return [value if rank in live else None for rank in range(4)]

        return gather

    rows = mnnvl.MnnvlRows(1, 4, _SIZE, gather_among({0, 1, 2, 3}))
    ptr = rows.ptr
    for live in ({0, 1, 2}, {0, 1, 2, 3}):
        before = [rows_driver.mapped[rows.row(rank)] for rank in range(4)]
        events.clear()
        rows.checkpoint_prepare()
        rows.remap(gather_among(live))
        after = [rows_driver.mapped[rows.row(rank)] for rank in range(4)]

        assert rows.ptr == ptr and rows_driver.reserved == [(ptr, 4 * _SIZE)]
        assert [event for event, _ in events] == ["handle", "zero", "outcome"]
        assert events[1] == ("zero", rows.row(1))
        assert all(handle in rows_driver.released for handle in before)
        assert not set(after) & set(before)
        assert after[3][0] == ("import" if 3 in live else "local")
    rows.close()
    assert rows_driver.mapped == {} and rows_driver.reserved == []


@pytest.mark.parametrize("failure", ["import", "memset"])
def test_failed_mnnvl_rows_leave_nothing_mapped_allocated_or_reserved(
    rows_driver, monkeypatch, failure
):
    """A live rank whose handle cannot be imported (it exited after exporting it) surfaces as
    MnnvlPeerHandleError naming it, after this rank's own row, a scratch row and an imported row were mapped;
    a later failure closes the rows too. Either way the rank still takes part in the outcome exchange, so its
    peers raise with it instead of waiting for it."""
    allgather, calls = _exchange_once(live={0, 1, 3})
    if failure == "import":
        imports = []

        def import_handle(data, handle_type):
            imports.append(data)
            return (2, None) if len(imports) == 2 else (0, ("import", data))

        monkeypatch.setattr(rows_driver, "cuMemImportFromShareableHandle", import_handle)
        expected = pytest.raises(
            mnnvl.MnnvlPeerHandleError, match=r"ranks \[3\]\ncuMemImportFromShareableHandle: 2"
        )
    else:
        monkeypatch.setattr(rows_driver, "cuMemsetD8", lambda address, value, size: 2)
        expected = pytest.raises(RuntimeError)

    with expected:
        mnnvl.MnnvlRows(1, 4, _SIZE, allgather)

    assert rows_driver.mapped == {} and rows_driver.reserved == []
    created = 2  # this rank's row and the scratch row of rank 2
    imported = 1 if failure == "import" else 2
    released = rows_driver.released
    assert len(released) == len({id(handle) for handle in released}) == created + imported
    assert mnnvl.MnnvlRows.allocated_map == {} and mnnvl.MnnvlRows.address_refcnt == {}
    assert len(calls) == 2 and calls[1] is not None


def test_mnnvl_rows_fail_on_every_live_rank_when_one_fails(rows_driver):
    """All or nothing: rank 1 mapped every row, but rank 0 could not import rank 3's handle; rank 1 raises the
    same error and keeps nothing mapped, allocated or reserved."""
    calls = []

    def allgather(payload):
        calls.append(payload)
        if len(calls) == 1:
            return [payload if rank in (0, 1, 3) else None for rank in range(4)]
        failed = ("MnnvlPeerHandleError: rank 3's handle", [3])
        return [failed if rank == 0 else payload if rank in (1, 3) else None for rank in range(4)]

    with pytest.raises(mnnvl.MnnvlPeerHandleError, match="rank 0: MnnvlPeerHandleError") as error:
        mnnvl.MnnvlRows(1, 4, _SIZE, allgather)

    assert error.value.ranks == (3,)
    assert rows_driver.mapped == {} and rows_driver.reserved == []
    assert mnnvl.MnnvlRows.allocated_map == {} and mnnvl.MnnvlRows.address_refcnt == {}


def test_mnnvl_rows_name_only_the_rank_whose_export_failed(rows_driver):
    """Rank 0 could not create its row: every live rank raises an error naming rank 0 only, and rank 1, which learnt
    of it in the handle exchange, reports no failure of its own; nothing stays mapped, allocated or reserved."""
    calls = []

    def allgather(payload):
        calls.append(payload)
        if len(calls) == 1:  # the handles
            failed = {"error": "RuntimeError: 2", "handle": None, "is_fabric": True, "pid": None}
            return [failed if rank == 0 else payload if rank == 1 else None for rank in range(4)]
        theirs = ("MnnvlMappingError: rank 0: RuntimeError: 2", [])
        return [theirs if rank == 0 else payload if rank == 1 else None for rank in range(4)]

    with pytest.raises(mnnvl.MnnvlMappingError, match="rank 0: MnnvlMappingError") as error:
        mnnvl.MnnvlRows(1, 4, _SIZE, allgather)

    assert error.value.ranks == (0,) and calls[-1] == ()
    assert rows_driver.mapped == {} and rows_driver.reserved == []
    assert mnnvl.MnnvlRows.allocated_map == {} and mnnvl.MnnvlRows.address_refcnt == {}


def test_collecting_failed_mnnvl_rows_leaves_a_retry_at_the_same_address_open(
    rows_driver, monkeypatch
):
    """The driver may reserve the address range of a failed construction again for its retry while the failed
    object is still alive (here, while its error is handled); collecting the failed object leaves the retry's
    rows mapped and open."""
    base = 1 << 44

    def reserve(size, alignment, addr, flags):
        rows_driver.reserved.append((base, size))
        return 0, base

    monkeypatch.setattr(rows_driver, "cuMemAddressReserve", reserve)
    allgather, _ = _exchange_once(live={0, 1})
    rows_driver.fail_import = True
    try:
        mnnvl.MnnvlRows(1, 4, _SIZE, allgather)
    except mnnvl.MnnvlPeerHandleError:
        # The error's traceback keeps the failed object alive during the retry.
        rows_driver.fail_import = False
        rows = mnnvl.MnnvlRows(1, 4, _SIZE, allgather)
    else:
        pytest.fail("MnnvlRows did not raise for a handle it could not import")
    assert rows.ptr == base
    gc.collect()

    assert base in mnnvl.MnnvlRows.allocated_map
    assert all(rows.row(rank) in rows_driver.mapped for rank in range(4))
    rows.close()
    assert rows_driver.mapped == {} and rows_driver.reserved == []


def test_a_failed_release_of_the_scratch_allocation_is_not_repeated(rows_driver, monkeypatch):
    """Absent ranks 2 and 3 share one scratch allocation: when releasing it fails, closing unmaps every row,
    reports the failure and does not release the scratch allocation again for the second absent row."""
    allgather, _ = _exchange_once(live={0, 1})
    rows = mnnvl.MnnvlRows(1, 4, _SIZE, allgather)
    scratch = rows_driver.mapped[rows.row(2)]
    attempts = []

    def release(handle):
        attempts.append(handle)
        return 2 if handle is scratch else 0

    monkeypatch.setattr(rows_driver, "cuMemRelease", release)
    with pytest.raises(RuntimeError):
        rows.close()

    assert rows_driver.mapped == {}
    assert len(attempts) == 3 and sum(handle is scratch for handle in attempts) == 1
    assert not hasattr(rows, "ptr")


def test_mnnvl_rows_refuse_a_device_without_fabric_handles(rows_driver, monkeypatch):
    monkeypatch.setattr(
        mnnvl.MnnvlMemory,
        "get_allocation_prop",
        classmethod(
            lambda cls, dev_id: SimpleNamespace(requestedHandleTypes="posix_fd", location=0)
        ),
    )
    allgather, calls = _exchange_once(live={0, 1})
    with pytest.raises(RuntimeError, match="requires fabric memory handles"):
        mnnvl.MnnvlRows(1, 4, _SIZE, allgather)
    assert calls == [] and rows_driver.reserved == []
