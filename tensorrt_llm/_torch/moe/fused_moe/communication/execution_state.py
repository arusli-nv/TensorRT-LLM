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
"""Graph-latched MoE All-to-All execution descriptor shared by one model."""

from __future__ import annotations

import torch

from tensorrt_llm._mnnvl_utils import MnnvlMemory

from ..ep_group_health import EP_MASK_NUM_WORDS, EPGroupHealth
from ..wide_ep_ft import recovery_enabled
from .nvlink_one_sided import NVLinkOneSided


def _mask_words(mask: int) -> tuple[int, ...]:
    """Split a rank mask into the native little-endian words, as int64 tensor values."""
    words = ((mask >> (64 * index)) & ((1 << 64) - 1) for index in range(EP_MASK_NUM_WORDS))
    return tuple(word - (1 << 64) if word >= (1 << 63) else word for word in words)


class MoeExecutionState:
    """Own the device descriptor that NVLinkOneSided kernels read at execution time.

    The native descriptor holds the active-rank mask words and the address of an
    ``int32[2]`` abort record (``MoeA2AFenceDescriptor``). ``published`` holds
    the descriptor address; :meth:`latch` copies it into ``slot`` on the current
    stream, and the dispatch/combine kernels read the descriptor through
    ``slot``. All tensor addresses are fixed for the lifetime of the state, so
    CUDA graphs captured once keep running after :meth:`install_mask` changes
    the membership they observe.

    :meth:`latch` must be enqueued at the start of every forward so that it is
    part of every captured graph. :meth:`install_mask` is only valid while the
    model is idle; it is the only method that synchronizes with the device.
    """

    def __init__(self, ep_group_health: EPGroupHealth, device: torch.device) -> None:
        snapshot = ep_group_health.snapshot()
        if snapshot.failed_ranks:
            raise ValueError("MoeExecutionState requires the full initial EP membership")
        if ep_group_health.moe_world_size > 64 * EP_MASK_NUM_WORDS:
            raise ValueError("EP size exceeds the native rank-mask width")
        self.health = ep_group_health
        self.abort = torch.zeros(2, dtype=torch.int32, device=device)
        self.descriptor = torch.tensor(
            [*_mask_words(snapshot.mask), self.abort.data_ptr()],
            dtype=torch.int64,
            device=device,
        )
        self.published = torch.tensor(
            [self.descriptor.data_ptr()], dtype=torch.int64, device=device
        )
        self.slot = torch.zeros_like(self.published)
        self.installed_generation = snapshot.generation
        self._memories: dict[int, MnnvlMemory] = {}

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

    def retain(self, memory: MnnvlMemory) -> None:
        """Keep a bound layer's workspace mapped until :meth:`close`."""
        if id(memory) not in self._memories:
            memory.pin()
            self._memories[id(memory)] = memory

    def latch(self) -> None:
        """Snapshot the published descriptor for the kernels of this forward."""
        torch.ops.trtllm.moe_a2a_fence_latch_descriptor(self.published, self.slot)

    def dispatch_kwargs(self) -> dict[str, torch.Tensor]:
        return {"execution_descriptor": self.slot}

    def combine_kwargs(self) -> dict[str, torch.Tensor]:
        return {"execution_descriptor": self.slot}

    def install_mask(self, words: tuple[int, ...]) -> None:
        """Publish committed mask words to the descriptor. Only call while the model is idle.

        ``words`` must equal the committed :class:`EPGroupHealth` mask, so
        :meth:`EPGroupHealth.commit_failed` precedes this call.

        Raises:
            ValueError: If ``words`` differ from the committed membership.
            RuntimeError: If a collective of this state has aborted; its
                workspace rounds may differ across ranks and cannot resume.
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
        self.installed_generation = snapshot.generation

    def close(self) -> None:
        """Release the workspace pins. Only call after the last forward has completed."""
        for memory in self._memories.values():
            memory.unpin()
        self._memories.clear()
