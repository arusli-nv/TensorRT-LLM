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
"""Shared rank mask, abort state and workspaces for rank-mask MoE All-to-All collectives.

Every rank-mask MoE layer over one :class:`EPGroupHealth` runs under one :class:`MoeExecutionState`.
The kernels read the rank mask and the abort state at fixed addresses, so captured CUDA graphs
follow every change. Long waits on peers publish the round flag they wait for in the abort state, which
the All-to-All watchdog polls. With Wide-EP fault tolerance, when a peer fails, ``request_abort``
releases the collectives that wait for it. Then, with the model idle, ``remap_workspaces`` maps the
workspaces among the active ranks, ``rearm`` returns them to their first round, and
``install_mask`` writes the new rank mask.
"""

from __future__ import annotations

import weakref
from collections.abc import Callable
from typing import TYPE_CHECKING, Protocol, cast

import torch

from tensorrt_llm._torch.mnnvl_alltoall_workspace import _MnnvlAlltoAllWorkspaceLifecycle
from tensorrt_llm._utils import prefer_pinned
from tensorrt_llm.bindings import internal as _tllm_internal

from ..ep_group_health import EP_MASK_NUM_WORDS, EPGroupHealth

if TYPE_CHECKING:
    from tensorrt_llm._torch.distributed.membership import StoreGroup
    from tensorrt_llm._torch.distributed.membership_mnnvl_memory import MembershipMnnvlMemory

_ABORT_REQUESTED = 0
# The requested and status words.
_ABORT_FLAGS = slice(0, int(_tllm_internal.thop.MOE_A2A_ABORT_WAITED_FLAG_INDEX))
_WAIT_EVIDENCE = slice(int(_tllm_internal.thop.MOE_A2A_ABORT_WAITED_FLAG_INDEX), None)
_MASK_WORD_BITS = 64

# Execution states keyed by id(ep_group_health), kept while a layer holds them.
_STATES: weakref.WeakValueDictionary[int, MoeExecutionState] = weakref.WeakValueDictionary()


def _mask_words(mask: int) -> tuple[int, ...]:
    word_mask = (1 << _MASK_WORD_BITS) - 1
    return tuple((mask >> (i * _MASK_WORD_BITS)) & word_mask for i in range(EP_MASK_NUM_WORDS))


def _require_active(mask_words: tuple[int, ...], ep_rank: int) -> None:
    if not (mask_words[ep_rank >> 6] >> (ep_rank & 63)) & 1:
        raise RuntimeError(f"active_rank_mask must mark the local ep_rank ({ep_rank}) as active")


class MembershipWorkspace(Protocol):
    """Peer-writable memory that a membership change remaps among the active ranks in place."""

    @property
    def device_bytes(self) -> int:
        """Return the device bytes that :meth:`remap` allocates for this rank's row."""

    def rows_needing_fresh_pages(self, comm: StoreGroup) -> frozenset[int]:
        """Return every row if a former member may still write into the workspace, else no row."""

    def remap(self, comm: StoreGroup, fresh_rows: frozenset[int]) -> None:
        """Map the workspace among the active ranks with ``fresh_rows`` on fresh pages."""


class MembershipAlltoAllWorkspaceLifecycle(_MnnvlAlltoAllWorkspaceLifecycle):
    """Lifecycle of an All-to-All workspace in MembershipMnnvlMemory, a :class:`MembershipWorkspace`."""

    @property
    def device_bytes(self) -> int:
        """Return the device bytes that a remap allocates for this rank's row."""
        return cast("MembershipMnnvlMemory", self._memory).local_device_bytes

    def rows_needing_fresh_pages(self, comm: StoreGroup) -> frozenset[int]:
        """Return every row if a former member may still write into the workspace, else no row."""
        return cast("MembershipMnnvlMemory", self._memory).rows_needing_fresh_pages(comm)

    def remap(self, comm: StoreGroup, fresh_rows: frozenset[int]) -> None:
        """Map the workspace among the active ranks in place, with fresh pages for ``fresh_rows``."""
        cast("MembershipMnnvlMemory", self._memory).remap(comm, fresh_rows)


class BindableCommunication(Protocol):
    """MoE communication strategy whose collectives can run under a :class:`MoeExecutionState`."""

    def bind_execution_state(self, state: MoeExecutionState) -> None:
        """Run every later collective with ``state``'s rank mask and abort state."""


class MoeExecutionState:
    """Own the rank mask and abort state that NVLinkOneSided kernels in rank-mask mode read.

    Both are edited only in place, so captured CUDA graphs keep running. The device rank
    mask changes only through :meth:`write_rank_mask`.
    """

    def __init__(self, ep_group_health: EPGroupHealth, ep_rank: int, device: torch.device) -> None:
        self.health = ep_group_health
        self.ep_rank = ep_rank
        self._device = device
        # The kernels read the abort state at its host address, which must be pinned memory.
        self.abort = torch.zeros(
            int(_tllm_internal.thop.MOE_A2A_ABORT_STATE_WORDS),
            dtype=torch.int32,
            pin_memory=prefer_pinned(),
        )
        if not self.abort.is_pinned():
            raise RuntimeError("The MoE All-to-All abort state requires pinned host memory")
        self._abort_flags = self.abort.numpy()[_ABORT_FLAGS]
        snapshot = ep_group_health.snapshot()
        self._generation = snapshot.generation
        self._mask_generation: int | None = snapshot.generation
        self.rank_mask_words = _mask_words(snapshot.mask)
        _require_active(self.rank_mask_words, ep_rank)
        self.active_rank_mask = torch.tensor(self.rank_mask_words, dtype=torch.uint64).to(device)
        self._mask_staging = torch.empty(
            EP_MASK_NUM_WORDS, dtype=torch.uint64, pin_memory=prefer_pinned()
        )
        self._mask_staged: torch.cuda.Event | None = None
        self._workspaces: dict[
            int, tuple[MembershipWorkspace, Callable[[], None], torch.Tensor]
        ] = {}
        self._layer_resets: list[Callable[[], None]] = []

    @staticmethod
    def of(ep_group_health: EPGroupHealth, ep_rank: int, device: torch.device) -> MoeExecutionState:
        """Return the state that every layer over ``ep_group_health`` shares."""
        state = _STATES.get(id(ep_group_health))
        if state is None:
            state = MoeExecutionState(ep_group_health, ep_rank, device)
            _STATES[id(ep_group_health)] = state
        elif state.ep_rank != ep_rank or state._device != device:
            raise ValueError("Layers over one EP group health must share their EP rank and device")
        return state

    def write_rank_mask(self, active_rank_mask: torch.Tensor | None = None) -> None:
        """Write the rank mask that collectives launched later on the current stream read.

        Without ``active_rank_mask``, write the EP group health's mask unless its generation is
        already written. A caller's ``active_rank_mask`` belongs to no generation, so the next write
        without one writes the health's mask again. The write goes through pinned staging, so it
        does not wait for the GPU.
        """
        if active_rank_mask is None:
            if self.health.generation == self._mask_generation:
                return
            snapshot = self.health.snapshot()
            words, generation = _mask_words(snapshot.mask), snapshot.generation
        else:
            words, generation = tuple(int(word) for word in active_rank_mask.tolist()), None
        if words != self.rank_mask_words:
            # The kernels treat this rank's bit as set, so a mask that clears it is refused.
            _require_active(words, self.ep_rank)
            if self._mask_staged is not None:
                self._mask_staged.synchronize()
            self._mask_staging.copy_(torch.tensor(words, dtype=torch.uint64))
            self.active_rank_mask.copy_(self._mask_staging, non_blocking=True)
            self._mask_staged = torch.cuda.Event()
            self._mask_staged.record(torch.cuda.current_stream(self._device))
            self.rank_mask_words = words
        self._mask_generation = generation

    def register_workspace(
        self,
        workspace: MembershipWorkspace,
        reset_workspace: Callable[[], None],
        rows: torch.Tensor,
    ) -> None:
        """Register a bound layer's workspace, once per workspace that layers share.

        ``reset_workspace`` resets this rank's row, and ``rows`` is the workspace's
        ``[ep_size, bytes]`` uint8 view.
        """
        self._workspaces.setdefault(id(workspace), (workspace, reset_workspace, rows))

    def staging_workspace(self) -> torch.Tensor:
        """Return the rows of the largest bound workspace.

        The workspace is idle from :meth:`remap_workspaces` until :meth:`rearm`, so restores may
        stage expert weights through it.
        """
        return max(
            (rows for _, _, rows in self._workspaces.values()), key=lambda rows: rows.shape[1]
        )

    def workspace_bytes(self) -> int:
        """Return the device memory :meth:`remap_workspaces` allocates on this rank."""
        return sum(workspace.device_bytes for workspace, _, _ in self._workspaces.values())

    def workspace_rows_needing_fresh_pages(self, comm: StoreGroup) -> frozenset[int]:
        """Return every row if a former member may still write into any bound workspace."""
        return frozenset().union(
            *(
                workspace.rows_needing_fresh_pages(comm)
                for workspace, _, _ in self._workspaces.values()
            )
        )

    def remap_workspaces(self, comm: StoreGroup, fresh_rows: frozenset[int]) -> None:
        """Map every bound workspace at the same addresses among the active members of ``comm``.

        Call only while no rank is in a collective. Every rank must :meth:`rearm` before the next
        collective.
        """
        for workspace, _, _ in self._workspaces.values():
            workspace.remap(comm, fresh_rows)

    def register_layer(self, reset_state: Callable[[], None]) -> None:
        """Register the reset for a bound layer's dispatch/combine state machine."""
        if self.health.generation != self._generation:
            # A layer bound after a change has a workspace that the change's remaps left out.
            raise RuntimeError("Bind MoE layers only before the EP membership changes")
        self._layer_resets.append(reset_state)

    def rearm(self) -> None:
        """Clear the abort state and return every bound workspace and layer to its first round.

        Call only while the model is idle and before :meth:`install_mask`. The device is
        synchronized before and after, so no queued collective sees a partial reset.
        """
        torch.cuda.synchronize(self._device)
        self.abort.zero_()
        for _, reset, _ in self._workspaces.values():
            reset()
        for reset_state in self._layer_resets:
            reset_state()
        torch.cuda.synchronize(self._device)

    def request_abort(self) -> None:
        """Make running and future collectives return without peers. Safe from any thread."""
        self._abort_flags[_ABORT_REQUESTED] = 1

    def aborted(self) -> bool:
        """Return whether a collective aborted since the last :meth:`rearm`."""
        return bool(self._abort_flags.any())

    def waited_flags(self) -> dict[int, int]:
        """Return each EP rank a kernel has long waited for, with the round flag it waits for."""
        # The abort state stores the uint32 flags in int32 words.
        return {
            peer: round_flag & 0xFFFFFFFF
            for peer, round_flag in enumerate(self.abort[_WAIT_EVIDENCE].tolist())
            if round_flag
        }

    def install_mask(self) -> None:
        """Write the EP group health's rank mask into the device mask that the kernels read.

        Call only while the model is idle and after :meth:`rearm`. The device is synchronized
        before and after, so no queued collective sees a partial mask.
        """
        torch.cuda.synchronize(self._device)
        if self.aborted():
            raise RuntimeError(f"MoE All-to-All abort state is {self.abort[_ABORT_FLAGS].tolist()}")
        self.write_rank_mask()
        torch.cuda.synchronize(self._device)
