# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Rank-mask and abort records of the MoE All-to-All collectives of one model."""

from __future__ import annotations

from collections.abc import Callable

import torch

from tensorrt_llm._torch.distributed.mnnvl_memory import MnnvlMemory
from tensorrt_llm._utils import device_visible_host_zeros

from ..ep_group_health import EPGroupHealth
from .nvlink_one_sided import NVLinkOneSided

# Index of the ``requested`` word in the abort record.
_ABORT_REQUESTED = 0
# EP rank -> byte ranges [start, end) of a workspace row that its collectives may write.
WriteRanges = Callable[[int], list[tuple[int, int]]]


def _mask_tensor(health: EPGroupHealth) -> torch.Tensor:
    """The committed rank mask as the native little-endian uint64 words."""
    return torch.tensor(health.get_mask_words(), dtype=torch.uint64)


class MoeExecutionState:
    """Own the records that bound NVLinkOneSided kernels read at execution time.

    ``active_rank_mask`` is the device copy of the committed rank mask, and
    ``abort`` the native ``MoeA2AAbortState``: an ``int32[2]`` record
    ``(requested, status)`` in pinned host memory. Both are only edited in
    place, so CUDA graphs captured once keep running after :meth:`install_mask`
    changes the membership they observe.

    :meth:`install_mask` and :meth:`rearm` are only valid while the model is
    idle. :meth:`request_abort` may be called from any thread while kernels run.
    """

    def __init__(self, ep_group_health: EPGroupHealth, device: torch.device) -> None:
        self.health = ep_group_health
        # Kernels read the record through its mapped host address, so a request
        # needs no CUDA work to make progress.
        self.abort = device_visible_host_zeros(2, torch.int32)
        self.active_rank_mask = _mask_tensor(ep_group_health).to(device)
        # Retained workspaces with their reset and source write ranges callbacks, keyed by allocation.
        self._workspaces: dict[int, tuple[MnnvlMemory, Callable[[], None], WriteRanges]] = {}
        # Bound layers: (EP rank, reset of the layer's dispatch/combine state machine).
        self._layers: list[tuple[int, Callable[[], None]]] = []
        self.progress_comm: NVLinkOneSided | None = None

    @classmethod
    def for_model(cls, model: torch.nn.Module) -> MoeExecutionState:
        """Create the state and bind every NVLinkOneSided layer of ``model``.

        Bound layers dispatch into fixed receive slots. A backend that cannot
        read them in place (``supports_dispatch_receive_slots``) gets a compact
        copy of the hidden states from the scheduler.

        Raises:
            RuntimeError: The model has no NVLinkOneSided layer, or the layers
                do not share one EPGroupHealth.
            ValueError: A layer cannot dispatch float32 router weights, which
                bound dispatch compacts for the routing sort.
        """
        from ..configurable_moe import ConfigurableMoE  # imports this package

        comms = {}
        for module in model.modules():
            if isinstance(module, ConfigurableMoE) and isinstance(module.comm, NVLinkOneSided):
                scales_dtype = module.backend.input_requirement.routing_scales_dtype
                if module.apply_router_weight_on_input or scales_dtype not in (None, torch.float32):
                    raise ValueError(
                        "WideEP recovery requires float32 router weights in the dispatch; "
                        f"{type(module.backend).__name__} with apply_router_weight_on_input="
                        f"{module.apply_router_weight_on_input} does not dispatch them"
                    )
                comms[id(module.comm)] = module.comm
        if not comms:
            raise RuntimeError("WideEP recovery requires NVLinkOneSided MoE communication")
        healths = {id(comm.ep_group_health) for comm in comms.values()}
        comm = next(iter(comms.values()))
        if len(healths) != 1 or comm.ep_group_health is None:
            raise RuntimeError(
                "WideEP recovery requires one shared EPGroupHealth for all MoE layers"
            )
        state = cls(comm.ep_group_health, comm.workspace.device)
        # Hang detection reads the all-to-all progress of one workspace; every layer runs every round.
        state.progress_comm = comm
        for comm in comms.values():
            comm.bind_execution_state(state)
        return state

    def retain(
        self,
        memory: MnnvlMemory,
        reset_workspace: Callable[[], None],
        write_ranges: WriteRanges,
    ) -> None:
        """Keep a bound layer's workspace mapped until :meth:`close`.

        ``reset_workspace`` returns this rank's part of the workspace to its
        initial state; ``write_ranges(source)`` are the byte ranges of a row that
        EP rank ``source``'s collectives on it may write. Layers sharing one
        workspace register them once.
        """
        if id(memory) not in self._workspaces:
            memory.pin()
            self._workspaces[id(memory)] = (memory, reset_workspace, write_ranges)

    def staging_workspace(self) -> torch.Tensor:
        """The largest workspace, ``[ep_size, bytes]`` uint8; idle from a fenced abort to :meth:`rearm`."""
        return self._staging()[0].as_torch_strided_tensor(torch.uint8)

    def staging_write_ranges(self, source: int) -> list[tuple[int, int]]:
        """Byte ranges ``[start, end)`` of a :meth:`staging_workspace` row that EP rank ``source`` may write."""
        return self._staging()[2](source)

    def _staging(self) -> tuple[MnnvlMemory, Callable[[], None], WriteRanges]:
        return max(self._workspaces.values(), key=lambda entry: entry[0].segment_size)

    def track_layer(self, ep_rank: int, reset_state: Callable[[], None]) -> None:
        """Record a bound layer's EP rank and the reset of its dispatch/combine state machine."""
        self._layers.append((ep_rank, reset_state))

    def rearm(self) -> None:
        """Clear the abort record and reset every bound workspace to its first round.

        After an abort, ranks may have stopped in different collective rounds,
        and an eager layer may have dispatched without combining. Every
        surviving rank must rearm while all of them are quiescent and before
        any of them runs another collective. The abort record is cleared first,
        so an abort requested while the workspaces reset is kept.
        """
        torch.cuda.synchronize(self.active_rank_mask.device)
        self.abort.zero_()
        for _, reset, _ in self._workspaces.values():
            reset()
        for _, reset_state in self._layers:
            reset_state()
        torch.cuda.synchronize(self.active_rank_mask.device)

    def request_abort(self) -> None:
        """Make running and future collectives of this state return without peers.

        Kernels poll the request while they wait, so collectives stuck on a lost
        peer finish and record a cancelled status; their outputs are invalid.
        A plain host store: it never waits for the GPU.
        """
        self.abort[_ABORT_REQUESTED] = 1

    def aborted(self) -> bool:
        """Whether, since :meth:`rearm`, an abort was requested or a collective returned without some peer's data.

        A step whose collectives all completed while this is False read every
        active source's complete data (see ``MoeA2AAbortState``).
        """
        return bool(self.abort.any())

    def install_mask(self) -> None:
        """Publish the committed EP membership to the kernels. Only call while the model is idle.

        Raises:
            RuntimeError: If a collective of this state has aborted since the
                last :meth:`rearm`, or the membership excludes this rank.
        """
        torch.cuda.synchronize(self.active_rank_mask.device)
        if bool(self.abort.any()):
            raise RuntimeError(f"MoE All-to-All abort record is {self.abort.tolist()}")
        words = self.health.get_mask_words()
        for ep_rank, _ in self._layers:
            if not (words[ep_rank >> 6] >> (ep_rank & 63)) & 1:
                raise RuntimeError(f"The committed EP membership excludes this rank ({ep_rank})")
        self.active_rank_mask.copy_(_mask_tensor(self.health))
        torch.cuda.synchronize(self.active_rank_mask.device)

    def close(self) -> None:
        """Release the workspace pins. Only call after the last forward has completed."""
        for memory, _, _ in self._workspaces.values():
            memory.unpin()
        self._workspaces.clear()
