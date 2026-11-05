# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""MnnvlMemory.release_backing/remap against a fake CUDA driver: same addresses, live rows only."""

import itertools
from types import SimpleNamespace

import pytest
import torch

import tensorrt_llm._torch.distributed.mnnvl_memory as mnnvl

_FABRIC = "fabric"
_SIZE = 1 << 21
_STRIDE = 1 << 29
_BASE = 1 << 40


class _FakeDriver:
    """The cu* calls MnnvlMemory makes, recording page ownership per mapped address."""

    CUresult = SimpleNamespace(CUDA_SUCCESS=0)
    CUmemAllocationHandleType = SimpleNamespace(CU_MEM_HANDLE_TYPE_FABRIC=_FABRIC)
    CUmemAccess_flags = SimpleNamespace(CU_MEM_ACCESS_FLAGS_PROT_READWRITE=3)

    def __init__(self, own_rank: int):
        self._ids = itertools.count(100)
        self.own_rank = own_rank
        self.mapped = {}  # address -> handle
        self.released = []
        self.fail_create = False
        self.fail_import = False

    def CUmemAccessDesc(self):
        return SimpleNamespace()

    def cuCtxGetDevice(self):
        return 0, 0

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


class _RemapMemory(mnnvl.MnnvlMemory):
    pass


def _memory(monkeypatch, own_rank=1, size=4):
    driver = _FakeDriver(own_rank)
    monkeypatch.setattr(mnnvl, "cuda", driver)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *args: None)
    monkeypatch.setattr(mnnvl.MnnvlMemory, "dev_id", 0)
    monkeypatch.setattr(
        mnnvl.MnnvlMemory,
        "get_allocation_prop",
        classmethod(lambda cls, dev_id: SimpleNamespace(requestedHandleTypes=_FABRIC, location=0)),
    )
    memory = object.__new__(_RemapMemory)
    memory.ptr, memory.segment_size, memory.rank_stride = _BASE, _SIZE, _STRIDE
    handles = [("old", rank) for rank in range(size)]
    for rank, handle in enumerate(handles):
        driver.mapped[_BASE + rank * _STRIDE] = handle
    _RemapMemory.allocated_map[_BASE] = mnnvl._MnnvlAllocationRecord(
        comm=None,
        comm_size=size,
        comm_rank=own_rank,
        comm_membership=tuple(range(size)),
        aligned_size=_SIZE,
        mem_handles=list(handles),
        start_address=_BASE,
        rank_stride=_STRIDE,
        address_offset=0,
        pin_count=1,
    )
    return memory, driver


def _gather_without(absent, size=4):
    return lambda value: [None if rank in absent else value for rank in range(size)]


def test_remap_maps_fresh_pages_at_same_addresses_for_live_rows_only(monkeypatch):
    memory, driver = _memory(monkeypatch)
    assert memory.release_backing() == _SIZE
    assert driver.mapped == {}
    assert sorted(driver.released) == sorted(("old", rank) for rank in range(4))

    memory.remap(_gather_without({2}))

    assert memory.mapped
    assert sorted(driver.mapped) == [_BASE + rank * _STRIDE for rank in (0, 1, 3)]
    assert driver.mapped[_BASE + _STRIDE][0] == "local"
    assert all(handle[0] != "old" for handle in driver.mapped.values())


def test_release_after_remap_skips_unmapped_absent_rows(monkeypatch):
    memory, driver = _memory(monkeypatch)
    memory.release_backing()
    memory.remap(_gather_without({2}))
    memory.release_backing()
    assert driver.mapped == {}
    memory.remap(_gather_without({0, 2}))
    assert sorted(driver.mapped) == [_BASE + rank * _STRIDE for rank in (1, 3)]


def test_failed_remap_leaves_allocation_unmapped_and_retryable(monkeypatch):
    memory, driver = _memory(monkeypatch)
    memory.release_backing()
    driver.fail_create = True
    with pytest.raises(RuntimeError, match="handle export failed"):
        memory.remap(_gather_without({2}))
    assert driver.mapped == {}
    assert not memory.mapped
    driver.fail_create = False
    memory.remap(_gather_without({2}))
    assert memory.mapped


def test_remap_requires_released_backing(monkeypatch):
    memory, _ = _memory(monkeypatch)
    with pytest.raises(RuntimeError, match="Cannot remap"):
        memory.remap(_gather_without(set()))


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


def test_a_peer_s_failed_mapping_unmaps_this_rank_too(monkeypatch):
    """A rank whose import of a failing rank's handle failed makes every live rank raise, not only itself."""
    memory, driver = _memory(monkeypatch)
    memory.release_backing()
    driver.released.clear()
    with pytest.raises(RuntimeError, match="rank 3: RuntimeError"):
        memory.remap(_gather_outcomes({3: "RuntimeError: import failed"}, absent={2}))
    assert driver.mapped == {} and not memory.mapped
    # The fresh local handle and the imported ones were released.
    assert len(driver.released) == 3
    memory.remap(_gather_without({2}))
    assert sorted(driver.mapped) == [_BASE + rank * _STRIDE for rank in (0, 1, 3)]


def test_this_rank_s_failed_import_is_reported_to_the_others(monkeypatch):
    memory, driver = _memory(monkeypatch)
    memory.release_backing()
    driver.fail_import = True
    reported = []

    def gather(value):
        reported.append(value)
        return [None if rank == 2 else value for rank in range(4)]

    with pytest.raises(RuntimeError, match="rank 1: RuntimeError"):
        memory.remap(gather)
    assert reported[-1].startswith("RuntimeError")
    assert driver.mapped == {} and not memory.mapped


def test_a_failure_of_any_type_still_reaches_the_outcome_exchange(monkeypatch):
    """A mapping error that is not a RuntimeError is reported to the live peers waiting in the exchange too."""
    memory, driver = _memory(monkeypatch)
    memory.release_backing()

    def failing_map(address, size, offset, handle, flags):
        raise ValueError("bad mapping")

    monkeypatch.setattr(driver, "cuMemMap", failing_map)
    reported = []

    def gather(value):
        reported.append(value)
        return [None if rank == 2 else value for rank in range(4)]

    with pytest.raises(RuntimeError, match="rank 1: ValueError: bad mapping"):
        memory.remap(gather)
    assert len(reported) == 2 and reported[-1] == "ValueError: bad mapping"
    assert driver.mapped == {} and not memory.mapped


def test_a_peer_failure_during_the_outcome_exchange_leaves_the_allocation_unmapped(monkeypatch):
    memory, driver = _memory(monkeypatch)
    memory.release_backing()
    with pytest.raises(_PeerFailure):
        memory.remap(_gather_outcomes(_PeerFailure(), absent={2}))
    assert driver.mapped == {} and not memory.mapped
    memory.remap(_gather_without({2}))
    assert memory.mapped


def test_a_peer_failure_during_the_handle_exchange_propagates_without_the_outcome_exchange(
    monkeypatch,
):
    """The exchange that failed could not report an outcome either: the failure reaches the recovery at once."""
    memory, driver = _memory(monkeypatch)
    memory.release_backing()
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


def test_a_failed_cleanup_keeps_the_remap_s_error_and_breaks_the_allocation(monkeypatch):
    """A peer failure during the outcome exchange stays the error the recovery sees, even if releasing the fresh
    pages fails too; that failure is logged and the allocation is not remapped again."""
    memory, driver = _memory(monkeypatch)
    memory.release_backing()
    errors = []
    monkeypatch.setattr(mnnvl.logger, "error", errors.append)
    monkeypatch.setattr(driver, "cuMemUnmap", lambda address, size: 2)

    with pytest.raises(_PeerFailure):
        memory.remap(_gather_outcomes(_PeerFailure(), absent={2}))

    assert len(errors) == 1 and "failed MNNVL remap" in errors[0]
    assert _RemapMemory.allocated_map[_BASE].state is mnnvl._MnnvlAllocationState.BROKEN
    with pytest.raises(RuntimeError, match="Cannot remap MNNVL allocation in broken state"):
        memory.remap(_gather_without({2}))


def test_an_allocation_remapped_without_absent_ranks_refuses_a_checkpoint(monkeypatch):
    """A checkpoint restore maps every rank's row: after a remap left an absent rank's row unmapped, a
    checkpoint is refused instead of rebuilding that row."""
    memory, driver = _memory(monkeypatch)
    memory.release_backing()
    memory.remap(_gather_without({2}))
    record = _RemapMemory.allocated_map[_BASE]
    record.pin_count = 0
    mapped = dict(driver.mapped)

    try:
        with pytest.raises(RuntimeError, match="remapped without absent ranks"):
            memory.checkpoint_prepare()
    finally:
        # Pinned, the fake allocation is left alone when ``memory`` is collected.
        record.pin_count = 1

    assert driver.mapped == mapped
    assert record.state is mnnvl._MnnvlAllocationState.MAPPED
