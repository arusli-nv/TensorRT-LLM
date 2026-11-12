# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""MNNVL memory that follows the membership of Wide-EP fault tolerance."""

import time
from dataclasses import replace
from typing import Any, Callable, Sequence

import torch

from tensorrt_llm.logger import logger

from .membership import StoreGroup
from .mnnvl_active_group import _mapping_outcome, _raise_reported_failures
from .mnnvl_memory import MnnvlMemory, _MnnvlAllocationState


class MembershipMnnvlMemory(MnnvlMemory):
    """MNNVL memory that can be mapped again among the active ranks at the same addresses."""

    # Map the rows of absent ranks on one local scratch allocation instead of leaving them unmapped.
    absent_rows_on_scratch: bool = False
    _scratch_rows: frozenset[int] = frozenset()

    @property
    def local_device_bytes(self) -> int:
        return type(self).allocated_map[self.ptr].aligned_size

    def stale_rows(self, gather: Callable[[Any], Sequence[Any]]) -> frozenset[int]:
        """Return the rows that need fresh pages because a former member may still write into them.

        The pages stay only if every rank that mapped them is active and mapped them with the same ranks. This call is
        collective over the ranks ``gather`` reaches.
        """
        record = type(self).allocated_map[self.ptr]
        mapped_with = frozenset()
        if record.state is _MnnvlAllocationState.MAPPED:
            mapped_with = frozenset(
                rank
                for index, (rank, handle) in enumerate(
                    zip(record.comm_membership, record.mem_handles)
                )
                if handle is not None and index not in self._scratch_rows
            )
        views = gather(mapped_with)
        stale = any(
            not view or any(views[peer] is None or rank not in views[peer] for peer in view)
            for rank, view in enumerate(views)
            if view is not None
        )
        return frozenset(record.comm_membership) if stale else frozenset()

    def remap(self, gather: Callable[[Any], Sequence[Any]], fresh_rows: frozenset[int]) -> None:
        """Map the allocation among the active ranks at the same addresses, with ``fresh_rows`` on fresh pages.

        Late writes of a former member land in pages that no active rank maps. This call is collective over the ranks
        ``gather`` reaches, and every active rank either maps the allocation or raises the same error type.

        Raises:
            MnnvlPeerHandleError: Some active rank could not import a peer's handle.
            MnnvlMappingError: Mapping failed on some active ranks for another reason.
            RuntimeError: This rank's row is unmapped but not in ``fresh_rows``, or a kept row's rank is not active.
        """
        start_s = time.perf_counter()
        cls = type(self)
        record = cls.allocated_map[self.ptr]
        own_row_fresh = record.comm_membership[record.comm_rank] in fresh_rows
        if record.state is _MnnvlAllocationState.MAPPED:
            drop = [
                rank in fresh_rows or index in self._scratch_rows
                for index, rank in enumerate(record.comm_membership)
            ]
            if any(drop):
                torch.cuda.synchronize()
                try:
                    cls._unmap_and_release_handles(
                        replace(
                            record,
                            mem_handles=[
                                h if d else None for h, d in zip(record.mem_handles, drop)
                            ],
                        )
                    )
                except RuntimeError:
                    record.state = _MnnvlAllocationState.BROKEN
                    raise
                record.mem_handles = [None if d else h for h, d in zip(record.mem_handles, drop)]
        elif record.state is not _MnnvlAllocationState.UNMAPPED:
            raise RuntimeError(f"Cannot remap MNNVL allocation in {record.state.value} state")
        elif not own_row_fresh:
            raise RuntimeError("This rank's row of an unmapped MNNVL allocation needs fresh pages")
        kept = list(record.mem_handles)
        comm = StoreGroup(
            None,
            "MNNVL remap",
            record.comm_membership[record.comm_rank],
            record.comm_membership,
            gather=gather,
        )
        handles, outcome, local_failure = None, (), None

        def release() -> None:
            if handles is not None:
                record.mem_handles = handles
            try:
                cls._unmap_and_release_handles(record)
            except RuntimeError as cleanup_error:
                logger.error(
                    f"Failed to release the pages of a failed MNNVL remap: {cleanup_error}"
                )
                record.state = _MnnvlAllocationState.BROKEN
                return
            record.state = _MnnvlAllocationState.UNMAPPED

        try:
            handles = cls._create_and_map_handles(
                comm,
                record.aligned_size,
                record.start_address,
                record.rank_stride,
                record.address_offset,
                absent_rows_on_scratch=cls.absent_rows_on_scratch,
                mapped_handles=kept,
            )
            if own_row_fresh:
                self._reset_own_row()
        except Exception as failure:
            if comm.exchange_failed:
                release()
                raise
            local_failure = failure
            outcome = _mapping_outcome(failure, record.comm_rank, record.comm_membership)
        try:
            outcomes = gather(outcome)
            _raise_reported_failures("MNNVL remap", outcomes, local_failure)
        except BaseException:
            release()
            raise
        record.mem_handles = handles
        record.state = _MnnvlAllocationState.MAPPED
        self._scratch_rows = frozenset(
            index
            for index, (rank, handle) in enumerate(zip(record.comm_membership, handles))
            if handle is not None and outcomes[rank] is None
        )
        active = sum(reported is not None for reported in outcomes)
        imported = active - sum(handle is not None for handle in kept) - own_row_fresh
        logger.info(
            f"MNNVL remap of {cls.__name__} at {self.ptr:#x} among {active} active ranks: "
            f"{sum(outcomes[rank] is not None for rank in fresh_rows)} rows on fresh pages, imported "
            f"{imported} rows in {(time.perf_counter() - start_s) * 1e3:.1f} ms"
        )

    def _reset_own_row(self) -> None:
        """Initialize this rank's freshly mapped row during :meth:`remap`."""
