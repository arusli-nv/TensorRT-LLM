# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Rank-mask and abort records of the MoE All-to-All collectives of one model."""

from __future__ import annotations

import weakref
from collections.abc import Callable, Sequence
from typing import Any, Protocol

import torch

from tensorrt_llm._torch.alltoall_watchdog import ActiveRankMaskSnapshot
from tensorrt_llm._utils import prefer_pinned
from tensorrt_llm.bindings import internal as _tllm_internal

from ..ep_group_health import EPGroupHealth

# Words of the abort record: ``requested`` and ``status``, then the wait evidence, one word per peer.
_ABORT_REQUESTED = 0
_ABORT_WORDS = slice(0, int(_tllm_internal.thop.MOE_A2A_ABORT_WAIT_EVIDENCE_INDEX))
_WAIT_EVIDENCE = slice(int(_tllm_internal.thop.MOE_A2A_ABORT_WAIT_EVIDENCE_INDEX), None)


def _require_active(mask_words: tuple[int, ...], ep_rank: int) -> None:
    """The kernels of a rank whose own bit is clear would run as a failed rank."""
    if not (mask_words[ep_rank >> 6] >> (ep_rank & 63)) & 1:
        raise RuntimeError(f"The EP rank mask excludes this rank ({ep_rank})")


class MembershipWorkspace(Protocol):
    """Peer-writable memory of a bound layer, which a membership change maps among the live ranks at the same
    addresses. Each method that takes ``gather`` is collective over the ranks it reaches."""

    @property
    def device_bytes(self) -> int:
        """The device memory :meth:`remap` allocates on this rank."""

    def needs_fresh_pages(self, gather: Callable[[Any], Sequence[Any]]) -> bool:
        """Whether the change must map fresh pages (:meth:`remap`) rather than keep this rank's (:meth:`admit`)."""

    def admit(self, gather: Callable[[Any], Sequence[Any]]) -> None:
        """Map the rows of the ranks that join, keeping this rank's pages."""

    def remap(self, gather: Callable[[Any], Sequence[Any]]) -> None:
        """Map fresh pages, shared only among the live ranks."""


class BindableCommunication(Protocol):
    """A MoE communication strategy whose collectives can run under a :class:`MoeExecutionState`, in rank-mask mode
    on the shared ``ep_group_health``: ``bind_execution_state`` registers its layer and workspace, and every later
    collective reads the state's records by address. ``NVLinkOneSided`` is one."""

    ep_group_health: EPGroupHealth | None
    workspace: torch.Tensor

    def bind_execution_state(self, state: MoeExecutionState) -> None:
        """Run every later collective with ``state``'s records."""


# The state of each EPGroupHealth that layers share, keyed by the id of the health, which the state keeps alive.
_shared_states: weakref.WeakValueDictionary[int, MoeExecutionState] = weakref.WeakValueDictionary()


class MoeExecutionState:
    """Own the records that rank-mask NVLinkOneSided kernels read at execution time.

    ``active_rank_mask`` is the device rank mask the kernels read, and ``abort`` the native ``MoeA2AAbortState`` in
    pinned host memory: ``int32`` words ``(requested, status)``, then per peer the round flag a kernel has long waited
    for from it (wait evidence). Both are only edited in place, so captured CUDA graphs keep running.
    :meth:`install_mask` and :meth:`rearm` need an idle model; :meth:`request_abort` works from any thread.

    A shared state (:meth:`for_health`) serves the layers not bound to a state, which publish its mask at each
    dispatch; it has no abort record and binds no layer.
    """

    def __init__(
        self, ep_group_health: EPGroupHealth, device: torch.device, *, abort_record: bool = True
    ) -> None:
        self.health = ep_group_health
        # Kernels read the record at its host address, so a request needs no CUDA work to make progress.
        self.abort: torch.Tensor | None = None
        if abort_record:
            self.abort = torch.zeros(
                int(_tllm_internal.thop.MOE_A2A_ABORT_STATE_WORDS),
                dtype=torch.int32,
                pin_memory=prefer_pinned(),
            )
            if not self.abort.is_pinned():
                raise RuntimeError("The MoE All-to-All abort record requires pinned host memory")
            # A view of the same memory for the per-step check (aborted), without torch dispatch.
            self._abort_words = self.abort.numpy()[_ABORT_WORDS]
        # The installed mask, as the native little-endian uint64 words, its committed generation (None
        # before the first publication and for a caller's mask), and its device record.
        self._mask_words = ep_group_health.get_mask_words()
        self._mask_generation: int | None = None
        self.active_rank_mask = torch.tensor(self._mask_words, dtype=torch.uint64).to(device)
        # Host staging of mask rewrites, so that the copy to the device record does not wait for the GPU, and the
        # event after the last copy from it, which must complete before the staging is rewritten.
        self._mask_staging = torch.empty(
            len(self._mask_words), dtype=torch.uint64, pin_memory=prefer_pinned()
        )
        self._mask_staged: torch.cuda.Event | None = None
        # Bound workspaces with the reset of this rank's part, keyed by workspace.
        self._workspaces: dict[int, tuple[MembershipWorkspace, Callable[[], None]]] = {}
        # Bound layers: (EP rank, reset of the layer's dispatch/combine state machine).
        self._layers: list[tuple[int, Callable[[], None]]] = []

    @classmethod
    def for_health(cls, ep_group_health: EPGroupHealth, device: torch.device) -> MoeExecutionState:
        """Return the state shared by the layers of ``ep_group_health`` that are not bound to a state."""
        state = _shared_states.get(id(ep_group_health))
        if state is None:
            state = cls(ep_group_health, device, abort_record=False)
            _shared_states[id(ep_group_health)] = state
        return state

    @staticmethod
    def model_comms(model: torch.nn.Module) -> list[BindableCommunication]:
        """Return the communicators of the MoE layers of ``model``, which :meth:`for_model` binds; each must be a
        :class:`BindableCommunication` on one shared ``ep_group_health``.

        Raises:
            ValueError: A MoE layer's communication cannot be bound, the model has no MoE layer, or the
                communication of its MoE layers is not in rank-mask mode on one shared EPGroupHealth.
        """
        # Imported here: the MoE package imports this one.
        from ..fused_moe_vanilla import VanillaMoE
        from ..interface import MoE

        comms = {}
        for name, layer in model.named_modules():
            if not isinstance(layer, (MoE, VanillaMoE)):
                continue
            comm = getattr(layer, "comm", None)
            if not hasattr(comm, "bind_execution_state"):
                uses = "no communication strategy" if comm is None else type(comm).__name__
                raise ValueError(
                    "WideEP fault tolerance requires MoE communication that survives a lost rank (a strategy "
                    f"with bind_execution_state, such as NVLinkOneSided), but MoE layer {name} "
                    f"({type(layer).__name__}) uses {uses}; turn it off (enable_wide_ep_fault_tolerance=False) "
                    "or select NVLinkOneSided (TRTLLM_FORCE_COMM_METHOD=NVLINK_ONE_SIDED) with a MoE backend "
                    "that uses it"
                )
            comms[id(comm)] = comm
        if not comms:
            raise ValueError(
                "WideEP fault tolerance requires a model with MoE layers; turn it off "
                "(enable_wide_ep_fault_tolerance=False)"
            )
        healths = {id(comm.ep_group_health) for comm in comms.values()}
        if len(healths) != 1 or next(iter(comms.values())).ep_group_health is None:
            raise ValueError(
                "WideEP fault tolerance requires the communication of every MoE layer in rank-mask mode on "
                "one shared EPGroupHealth: create the communication of every MoE layer with the communication "
                "factory from the model's ModelConfig, which shares one EPGroupHealth through its extra_attrs "
                "(get_wide_ep_ft_options); or turn it off (enable_wide_ep_fault_tolerance=False)"
            )
        return list(comms.values())

    @classmethod
    def for_model(cls, model: torch.nn.Module) -> MoeExecutionState:
        """Create the state and bind the communicator of every MoE layer of ``model`` (:meth:`model_comms`).

        Raises:
            ValueError: See :meth:`model_comms`.
        """
        comms = cls.model_comms(model)
        state = cls(comms[0].ep_group_health, comms[0].workspace.device)
        for comm in comms:
            comm.bind_execution_state(state)
        return state

    def retain(self, workspace: MembershipWorkspace, reset_workspace: Callable[[], None]) -> None:
        """Register a bound layer's workspace; ``reset_workspace`` returns this rank's part of it to
        its initial state. Layers sharing one workspace register it once."""
        self._workspaces.setdefault(id(workspace), (workspace, reset_workspace))

    def workspace_bytes(self) -> int:
        """The device memory :meth:`remap_workspaces` allocates on this rank, on top of the pages it replaces until it
        returns."""
        return sum(workspace.device_bytes for workspace, _ in self._workspaces.values())

    def workspaces_need_fresh_pages(self, gather: Callable[[Any], Sequence[Any]]) -> bool:
        """Whether a change among the ranks ``gather`` reaches must back the bound workspaces with fresh pages
        (:meth:`remap_workspaces`) rather than map only the joining ranks' rows (:meth:`admit_to_workspaces`).
        Collective over those ranks."""
        return any(
            workspace.needs_fresh_pages(gather) for workspace, _ in self._workspaces.values()
        )

    def admit_to_workspaces(self, gather: Callable[[Any], Sequence[Any]]) -> None:
        """Map the joining ranks' rows into every bound workspace, keeping this rank's pages. Only while no rank is in a
        collective; collective over the ranks ``gather`` reaches.

        Raises:
            MnnvlPeerHandleError: As for :meth:`remap_workspaces`.
            RuntimeError: Mapping failed on some rank; the workspace whose admission failed is unmapped.
        """
        for workspace, _ in self._workspaces.values():
            workspace.admit(gather)

    def remap_workspaces(self, gather: Callable[[Any], Sequence[Any]]) -> None:
        """Back every bound workspace with fresh pages at the same addresses, shared only among the ranks ``gather``
        reaches. Only while no rank is in a collective; every rank must :meth:`rearm` before the next one.

        Raises:
            MnnvlPeerHandleError: A rank ``gather`` reached could not import a peer's handle, as when the
                peer exited during the remap; ``ranks`` are global ranks.
            RuntimeError: Mapping failed on some rank; the workspace whose remap failed is unmapped.
        """
        for workspace, _ in self._workspaces.values():
            workspace.remap(gather)

    def track_layer(self, ep_rank: int, reset_state: Callable[[], None]) -> None:
        """Record a bound layer's EP rank and the reset of its dispatch/combine state machine.

        Raises:
            RuntimeError: If the state has no abort record (a shared state of :meth:`for_health`) or the
                installed membership excludes ``ep_rank``.
        """
        if self.abort is None:
            raise RuntimeError("A MoeExecutionState without an abort record binds no layer")
        _require_active(self._mask_words, ep_rank)
        self._layers.append((ep_rank, reset_state))

    def rearm(self) -> None:
        """Clear the abort record, then reset every bound workspace and layer to its first round (an abort requested
        meanwhile is kept). Every live rank rearms while all are quiescent, before any runs another collective."""
        torch.cuda.synchronize(self.active_rank_mask.device)
        self.abort.zero_()
        for _, reset in self._workspaces.values():
            reset()
        for _, reset_state in self._layers:
            reset_state()
        torch.cuda.synchronize(self.active_rank_mask.device)

    def request_abort(self) -> None:
        """Make running and future collectives return without peers (their outputs are invalid); a plain host store
        that never waits for the GPU."""
        self.abort[_ABORT_REQUESTED] = 1

    def aborted(self) -> bool:
        """Whether, since :meth:`rearm`, an abort was requested or a collective returned without some peer's data."""
        return bool(self._abort_words.any())

    def waited_rounds(self) -> dict[int, int]:
        """The wait evidence: each EP rank a kernel has been waiting for long, with the round flag it waits for; any
        thread."""
        return {
            peer: round_flag
            for peer, round_flag in enumerate(self.abort[_WAIT_EVIDENCE].tolist())
            if round_flag
        }

    def publish_rank_mask(self, snapshot: ActiveRankMaskSnapshot, ep_rank: int) -> None:
        """Write the snapshot's mask into the device record before a dispatch, in order on the current stream; an
        unchanged committed generation costs nothing.

        Raises:
            RuntimeError: If the mask excludes ``ep_rank``; the kernels do not check the local rank's bit.
        """
        generation = snapshot.committed_generation
        if generation is not None and generation == self._mask_generation:
            return
        words = tuple(int(word) for word in snapshot.active_rank_mask.tolist())
        if words != self._mask_words:
            _require_active(words, ep_rank)
            self._copy_mask(snapshot.active_rank_mask)
            self._mask_words = words
        self._mask_generation = generation

    def _copy_mask(self, words: torch.Tensor) -> None:
        """Copy the host uint64 ``words`` into the device record on the current stream, through the staging buffer."""
        if self._mask_staged is not None:
            self._mask_staged.synchronize()
        self._mask_staging.copy_(words)
        self.active_rank_mask.copy_(self._mask_staging, non_blocking=True)
        self._mask_staged = torch.cuda.Event()
        self._mask_staged.record(torch.cuda.current_stream(self.active_rank_mask.device))

    def install_mask(self) -> None:
        """Publish the committed EP membership to the kernels. Only call while the model is idle.

        Raises:
            RuntimeError: If a collective of this state has aborted since the
                last :meth:`rearm`, or the membership excludes this rank.
        """
        torch.cuda.synchronize(self.active_rank_mask.device)
        if self.aborted():
            raise RuntimeError(
                f"MoE All-to-All abort record is {self.abort[_ABORT_WORDS].tolist()}"
            )
        words = self.health.get_mask_words()
        for ep_rank, _ in self._layers:
            _require_active(words, ep_rank)
        self._copy_mask(torch.tensor(words, dtype=torch.uint64))
        self._mask_words = words
        torch.cuda.synchronize(self.active_rank_mask.device)
