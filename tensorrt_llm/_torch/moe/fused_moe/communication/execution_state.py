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
"""Execution descriptor of the MoE All-to-All collectives of one model."""

from __future__ import annotations

from collections.abc import Callable

import torch

from tensorrt_llm._mnnvl_utils import MnnvlMemory
from tensorrt_llm._utils import maybe_pin_memory

from ..ep_group_health import EP_MASK_NUM_WORDS, EPGroupHealth
from ..wide_ep_ft import recovery_enabled
from .nvlink_one_sided import NVLinkOneSided


def _mask_words(mask: int) -> tuple[int, ...]:
    """Split a rank mask into the native little-endian words, as int64 tensor values."""
    words = ((mask >> (64 * index)) & ((1 << 64) - 1) for index in range(EP_MASK_NUM_WORDS))
    return tuple(word - (1 << 64) if word >= (1 << 63) else word for word in words)


class MoeExecutionState:
    """Own the device descriptor that NVLinkOneSided kernels read at execution time.

    ``descriptor`` is the native ``MoeA2AFenceDescriptor``: ``EP_MASK_NUM_WORDS``
    active-rank mask words followed by the ``int32[2]`` abort record
    ``(requested, status)``, which ``abort`` views. The dispatch/combine ops
    validate this size against the native mask width. The descriptor is only
    edited in place, so CUDA graphs captured once keep running after
    :meth:`install_mask` changes the membership they observe.

    :meth:`install_mask` and :meth:`rearm` are only valid while the model is
    idle. :meth:`request_abort` may be called from any thread while kernels run.
    """

    def __init__(self, ep_group_health: EPGroupHealth, device: torch.device) -> None:
        snapshot = ep_group_health.snapshot()
        if snapshot.failed_ranks:
            raise ValueError("MoeExecutionState requires the full initial EP membership")
        if ep_group_health.moe_world_size > 64 * EP_MASK_NUM_WORDS:
            raise ValueError("EP size exceeds the native rank-mask width")
        self.health = ep_group_health
        self.descriptor = torch.tensor(
            [*_mask_words(snapshot.mask), 0], dtype=torch.int64, device=device
        )
        self.abort = self.descriptor[EP_MASK_NUM_WORDS:].view(torch.int32)
        self._memories: dict[int, MnnvlMemory] = {}
        self._workspace_resets: dict[int, Callable[[], None]] = {}
        # The abort request travels on its own stream so it does not queue
        # behind the kernels it has to stop.
        self._abort_stream = torch.cuda.Stream(device)
        self._abort_request = maybe_pin_memory(torch.ones(1, dtype=torch.int32))

    @classmethod
    def for_model(cls, model: torch.nn.Module) -> MoeExecutionState | None:
        """Create the state and bind every NVLinkOneSided layer, or return None if recovery is off."""
        if not recovery_enabled():
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
        if id(memory) not in self._memories:
            memory.pin()
            self._memories[id(memory)] = memory
            self._workspace_resets[id(memory)] = reset_workspace

    def rearm(self) -> None:
        """Clear the abort record and reset every bound workspace to its first round.

        After an abort, ranks may have stopped in different collective rounds.
        Every surviving rank must rearm while all of them are quiescent and
        before any of them runs another collective.
        """
        torch.cuda.synchronize(self.abort.device)
        for reset in self._workspace_resets.values():
            reset()
        self.abort.zero_()
        torch.cuda.synchronize(self.abort.device)

    def dispatch_kwargs(self) -> dict[str, torch.Tensor]:
        """Keyword arguments that run ``moe_a2a_dispatch`` under this state."""
        return {"execution_descriptor": self.descriptor}

    def combine_kwargs(self) -> dict[str, torch.Tensor]:
        """Keyword arguments that run ``moe_a2a_combine`` under this state."""
        return {"execution_descriptor": self.descriptor}

    def request_abort(self) -> None:
        """Make running and future collectives of this state return without peers.

        Kernels poll the request while they wait, so collectives stuck on a lost
        peer finish and record a cancelled status; their outputs are invalid.
        Returns once the request is visible in device memory.
        """
        with torch.cuda.device(self.abort.device), torch.cuda.stream(self._abort_stream):
            self.abort[:1].copy_(self._abort_request, non_blocking=True)
        self._abort_stream.synchronize()

    def install_mask(self, words: tuple[int, ...]) -> None:
        """Publish committed mask words to the descriptor. Only call while the model is idle.

        ``words`` must equal the committed :class:`EPGroupHealth` mask, so
        :meth:`EPGroupHealth.commit_failed` precedes this call.

        Raises:
            ValueError: If ``words`` differ from the committed membership.
            RuntimeError: If a collective of this state has aborted since the
                last :meth:`rearm`.
        """
        snapshot = self.health.snapshot()
        committed = sum(word << (64 * index) for index, word in enumerate(words))
        if len(words) != EP_MASK_NUM_WORDS or committed != snapshot.mask:
            raise ValueError(f"Mask words {words} differ from the committed EP membership")
        torch.cuda.synchronize(self.descriptor.device)
        if bool(self.abort.any()):
            raise RuntimeError(f"MoE All-to-All abort record is {self.abort.tolist()}")
        host = torch.tensor(_mask_words(committed), dtype=torch.int64)
        self.descriptor[:EP_MASK_NUM_WORDS].copy_(host)
        torch.cuda.synchronize(self.descriptor.device)

    def close(self) -> None:
        """Release the workspace pins. Only call after the last forward has completed."""
        for memory in self._memories.values():
            memory.unpin()
        self._memories.clear()
        self._workspace_resets.clear()
