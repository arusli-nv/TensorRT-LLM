# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Execution descriptor of the MoE All-to-All collectives of one model."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import TYPE_CHECKING

import torch

from tensorrt_llm._mnnvl_utils import MnnvlMemory
from tensorrt_llm._utils import device_visible_host_zeros, peer_loss_recovery_enabled

from ..ep_group_health import EP_MASK_NUM_WORDS, EPGroupHealth
from .nvlink_one_sided import NVLinkOneSided

if TYPE_CHECKING:
    from ..moe_load_balancer import MoeLoadBalancer


def _mask_tensor(health: EPGroupHealth) -> torch.Tensor:
    """The committed rank mask as the native little-endian words, viewed as int64."""
    return torch.tensor(health.get_mask_words(), dtype=torch.uint64).view(torch.int64)


class MoeExecutionState:
    """Own the device descriptor that NVLinkOneSided kernels read at execution time.

    ``descriptor`` is the native ``MoeA2AFenceDescriptor``: ``EP_MASK_NUM_WORDS``
    active-rank mask words followed by the address of ``abort``, the
    ``int32[2]`` record ``(requested, status)`` in pinned host memory. The
    dispatch/combine ops validate this size against the native mask width. The
    descriptor is only edited in place, so CUDA graphs captured once keep
    running after :meth:`install_mask` changes the membership they observe.

    :meth:`install_mask` and :meth:`rearm` are only valid while the model is
    idle. :meth:`request_abort` may be called from any thread while kernels run.
    """

    def __init__(self, ep_group_health: EPGroupHealth, device: torch.device) -> None:
        self.health = ep_group_health
        # Kernels read the record through its mapped host address, so a request
        # needs no CUDA work to make progress.
        self.abort = device_visible_host_zeros(2, torch.int32)
        address = torch.tensor([self.abort.data_ptr()], dtype=torch.int64)
        self.descriptor = torch.cat([_mask_tensor(ep_group_health), address]).to(device)
        # Retained workspaces and their reset callbacks, keyed by allocation.
        self._workspaces: dict[int, tuple[MnnvlMemory, Callable[[], None]]] = {}

    @classmethod
    def for_model(cls, model: torch.nn.Module) -> MoeExecutionState | None:
        """Create the state and bind every NVLinkOneSided layer, or return None if recovery is off."""
        if not peer_loss_recovery_enabled():
            return None
        comms = {}
        for module in model.modules():
            comm = getattr(module, "comm", None)
            if isinstance(comm, NVLinkOneSided):
                comms[id(comm)] = comm
        if not comms:
            raise RuntimeError("WideEP recovery requires NVLinkOneSided MoE communication")
        healths = {id(comm.ep_group_health) for comm in comms.values()}
        comm = next(iter(comms.values()))
        if len(healths) != 1 or comm.ep_group_health is None:
            raise RuntimeError(
                "WideEP recovery requires one shared EPGroupHealth for all MoE layers"
            )
        state = cls(comm.ep_group_health, comm.workspace.device)
        for comm in comms.values():
            comm.bind_execution_state(state)
        return state

    def retain(self, memory: MnnvlMemory, reset_workspace: Callable[[], None]) -> None:
        """Keep a bound layer's workspace mapped until :meth:`close`.

        ``reset_workspace`` returns this rank's part of the workspace to its
        initial state; layers sharing one workspace register it once.
        """
        if id(memory) not in self._workspaces:
            memory.pin()
            self._workspaces[id(memory)] = (memory, reset_workspace)

    def rearm(self) -> None:
        """Clear the abort record and reset every bound workspace to its first round.

        After an abort, ranks may have stopped in different collective rounds.
        Every surviving rank must rearm while all of them are quiescent and
        before any of them runs another collective.
        """
        torch.cuda.synchronize(self.descriptor.device)
        for _, reset in self._workspaces.values():
            reset()
        torch.cuda.synchronize(self.descriptor.device)
        self.abort.zero_()

    def request_abort(self) -> None:
        """Make running and future collectives of this state return without peers.

        Kernels poll the request while they wait, so collectives stuck on a lost
        peer finish and record a cancelled status; their outputs are invalid.
        A plain host store: it never waits for the GPU.
        """
        self.abort[0] = 1

    def install_mask(self) -> None:
        """Publish the committed EP membership to the descriptor. Only call while the model is idle.

        Raises:
            RuntimeError: If a collective of this state has aborted since the
                last :meth:`rearm`.
        """
        torch.cuda.synchronize(self.descriptor.device)
        if bool(self.abort.any()):
            raise RuntimeError(f"MoE All-to-All abort record is {self.abort.tolist()}")
        self.descriptor[:EP_MASK_NUM_WORDS].copy_(_mask_tensor(self.health))
        torch.cuda.synchronize(self.descriptor.device)

    def install_resident_placement(
        self, load_balancer: MoeLoadBalancer, failed_ranks: Iterable[int]
    ) -> None:
        """Route MoE traffic around ``failed_ranks`` using resident replicas.

        Only call while the model is idle. Routing must exclude the failed
        ranks before their mask is installed, because the MoE kernels silently
        skip routes to a masked rank.

        Raises:
            RuntimeError: A rank in ``failed_ranks`` had already failed.
        """
        failed = sorted(set(failed_ranks))
        load_balancer.reconfigure_mask_only(failed)
        for rank in failed:
            if not self.health.mark_failed(rank):
                raise RuntimeError(f"EP rank {rank} is already failed")
        self.rearm()
        self.install_mask()

    def close(self) -> None:
        """Release the workspace pins. Only call after the last forward has completed."""
        for memory, _ in self._workspaces.values():
            memory.unpin()
        self._workspaces.clear()
