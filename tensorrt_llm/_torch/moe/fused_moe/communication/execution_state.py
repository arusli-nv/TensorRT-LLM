# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Rank-mask and abort states of the MoE All-to-All collectives of one model."""

from __future__ import annotations

import weakref
from collections.abc import Callable, Sequence
from typing import Any, Protocol

import torch

from tensorrt_llm._torch.alltoall_watchdog import ActiveRankMaskSnapshot
from tensorrt_llm._utils import prefer_pinned
from tensorrt_llm.bindings import internal as _tllm_internal

from ..ep_group_health import EPGroupHealth

_ABORT_REQUESTED = 0
_ABORT_WORDS = slice(0, int(_tllm_internal.thop.MOE_A2A_ABORT_WAIT_EVIDENCE_INDEX))
_WAIT_EVIDENCE = slice(int(_tllm_internal.thop.MOE_A2A_ABORT_WAIT_EVIDENCE_INDEX), None)


def _require_active(mask_words: tuple[int, ...], ep_rank: int) -> None:
    if not (mask_words[ep_rank >> 6] >> (ep_rank & 63)) & 1:
        raise RuntimeError(f"The EP rank mask excludes this rank ({ep_rank})")


class MembershipWorkspace(Protocol):
    """Peer-writable memory that a membership change remaps among the active ranks at the same addresses."""

    @property
    def device_bytes(self) -> int:
        """Return the device memory :meth:`remap` allocates on this rank."""

    def stale_rows(self, gather: Callable[[Any], Sequence[Any]]) -> frozenset[int]:
        """Return the global ranks whose rows need fresh pages."""

    def remap(self, gather: Callable[[Any], Sequence[Any]], fresh_rows: frozenset[int]) -> None:
        """Map the workspace among the active ranks with ``fresh_rows`` on fresh pages."""


class BindableCommunication(Protocol):
    """MoE communication strategy whose collectives can run under a :class:`MoeExecutionState`."""

    ep_group_health: EPGroupHealth | None
    workspace: torch.Tensor
    execution_state: MoeExecutionState | None

    def bind_execution_state(self, state: MoeExecutionState) -> None:
        """Run every later collective with ``state``'s records."""


_shared_states: weakref.WeakValueDictionary[int, MoeExecutionState] = weakref.WeakValueDictionary()


class MoeExecutionState:
    """Own the rank mask and abort states that rank-mask NVLinkOneSided kernels read at execution time.

    Both records are only edited in place, so captured CUDA graphs keep running.
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
                raise RuntimeError("The MoE All-to-All abort state requires pinned host memory")
            self._abort_words = self.abort.numpy()[_ABORT_WORDS]
        self._mask_words = ep_group_health.get_mask_words()
        self._mask_generation: int | None = None
        self.active_rank_mask = torch.tensor(self._mask_words, dtype=torch.uint64).to(device)
        # Pinned staging lets the copy to the device record not wait for the GPU.
        self._mask_staging = torch.empty(
            len(self._mask_words), dtype=torch.uint64, pin_memory=prefer_pinned()
        )
        self._mask_staged: torch.cuda.Event | None = None
        self._workspaces: dict[int, tuple[MembershipWorkspace, Callable[[], None]]] = {}
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
        """Return the communicators of the MoE layers of ``model``.

        Raises:
            ValueError: A MoE layer's communication cannot be bound to one shared EPGroupHealth.
        """
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
                    "Wide-EP fault tolerance requires MoE communication that survives a failed rank (a strategy "
                    f"with bind_execution_state, such as NVLinkOneSided), but MoE layer {name} "
                    f"({type(layer).__name__}) uses {uses}; turn it off (enable_wide_ep_fault_tolerance=False) "
                    "or select NVLinkOneSided (TRTLLM_FORCE_COMM_METHOD=NVLINK_ONE_SIDED) with a MoE backend "
                    "that uses it"
                )
            comms[id(comm)] = comm
        if not comms:
            raise ValueError(
                "Wide-EP fault tolerance requires a model with MoE layers; turn it off "
                "(enable_wide_ep_fault_tolerance=False)"
            )
        healths = {id(comm.ep_group_health) for comm in comms.values()}
        if len(healths) != 1 or next(iter(comms.values())).ep_group_health is None:
            raise ValueError(
                "Wide-EP fault tolerance requires the communication of every MoE layer in rank-mask mode on "
                "one shared EPGroupHealth: create the communication of every MoE layer with the communication "
                "factory from the model's ModelConfig, which shares one EPGroupHealth through its extra_attrs "
                "(get_wide_ep_ft_options); or turn it off (enable_wide_ep_fault_tolerance=False)"
            )
        return list(comms.values())

    @classmethod
    def for_model(cls, model: torch.nn.Module) -> MoeExecutionState:
        """Create the state and bind the communicator of every MoE layer of ``model``."""
        comms = cls.model_comms(model)
        state = cls(comms[0].ep_group_health, comms[0].workspace.device)
        for comm in comms:
            comm.bind_execution_state(state)
        return state

    def retain(self, workspace: MembershipWorkspace, reset_workspace: Callable[[], None]) -> None:
        """Register a bound layer's workspace and the reset of this rank's part of it."""
        self._workspaces.setdefault(id(workspace), (workspace, reset_workspace))

    def workspace_bytes(self) -> int:
        """Return the device memory :meth:`remap_workspaces` allocates on this rank."""
        return sum(workspace.device_bytes for workspace, _ in self._workspaces.values())

    def stale_workspace_rows(self, gather: Callable[[Any], Sequence[Any]]) -> frozenset[int]:
        """Return the global ranks whose rows need fresh pages in any bound workspace."""
        return frozenset().union(
            *(workspace.stale_rows(gather) for workspace, _ in self._workspaces.values())
        )

    def remap_workspaces(
        self, gather: Callable[[Any], Sequence[Any]], fresh_rows: frozenset[int]
    ) -> None:
        """Map every bound workspace at the same addresses among the ranks ``gather`` reaches.

        Call only while no rank is in a collective. Every rank must :meth:`rearm` before the next one.
        """
        for workspace, _ in self._workspaces.values():
            workspace.remap(gather, fresh_rows)

    def track_layer(self, ep_rank: int, reset_state: Callable[[], None]) -> None:
        """Record a bound layer's EP rank and the reset of its dispatch/combine state machine."""
        if self.abort is None:
            raise RuntimeError("A MoeExecutionState without an abort state binds no layer")
        _require_active(self._mask_words, ep_rank)
        self._layers.append((ep_rank, reset_state))

    def rearm(self) -> None:
        """Clear the abort state and reset every bound workspace and layer to its first round."""
        torch.cuda.synchronize(self.active_rank_mask.device)
        self.abort.zero_()
        for _, reset in self._workspaces.values():
            reset()
        for _, reset_state in self._layers:
            reset_state()
        torch.cuda.synchronize(self.active_rank_mask.device)

    def request_abort(self) -> None:
        """Make running and future collectives return without peers. Safe from any thread."""
        self.abort[_ABORT_REQUESTED] = 1

    def aborted(self) -> bool:
        """Return whether a collective aborted since the last :meth:`rearm`."""
        return bool(self._abort_words.any())

    def waited_rounds(self) -> dict[int, int]:
        """Return each EP rank a kernel has long waited for, with the round flag it waits for."""
        return {
            peer: round_flag
            for peer, round_flag in enumerate(self.abort[_WAIT_EVIDENCE].tolist())
            if round_flag
        }

    def publish_rank_mask(self, snapshot: ActiveRankMaskSnapshot, ep_rank: int) -> None:
        """Write the snapshot's mask into the device record on the current stream before a dispatch."""
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
        if self._mask_staged is not None:
            self._mask_staged.synchronize()
        self._mask_staging.copy_(words)
        self.active_rank_mask.copy_(self._mask_staging, non_blocking=True)
        self._mask_staged = torch.cuda.Event()
        self._mask_staged.record(torch.cuda.current_stream(self.active_rank_mask.device))

    def install_mask(self) -> None:
        """Publish the committed EP membership to the kernels. Only call while the model is idle."""
        torch.cuda.synchronize(self.active_rank_mask.device)
        if self.aborted():
            raise RuntimeError(f"MoE All-to-All abort state is {self.abort[_ABORT_WORDS].tolist()}")
        words = self.health.get_mask_words()
        for ep_rank, _ in self._layers:
            _require_active(words, ep_rank)
        self._copy_mask(torch.tensor(words, dtype=torch.uint64))
        self._mask_words = words
        torch.cuda.synchronize(self.active_rank_mask.device)
