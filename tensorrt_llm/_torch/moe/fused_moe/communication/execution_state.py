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
"""The active-rank mask that the MoE all-to-all kernels read under Wide-EP fault tolerance."""

from __future__ import annotations

import torch

from ..ep_group_health import EP_MASK_NUM_WORDS, EPGroupHealth

_MASK_WORD_BITS = 64


def _mask_words(mask: int, ep_rank: int) -> tuple[int, ...]:
    """Split ``mask`` into the uint64 words of the kernels' mask, which must mark ``ep_rank``."""
    word_mask = (1 << _MASK_WORD_BITS) - 1
    words = tuple((mask >> (i * _MASK_WORD_BITS)) & word_mask for i in range(EP_MASK_NUM_WORDS))
    if not (words[ep_rank // _MASK_WORD_BITS] >> (ep_rank % _MASK_WORD_BITS)) & 1:
        raise RuntimeError(f"active_rank_mask must mark the local ep_rank ({ep_rank}) as active")
    return words


class MoeExecutionState:
    """Own the active-rank mask that the kernels of a model's NVLinkOneSided layers read.

    The mask stays at one device address and changes only in place, so CUDA graphs that captured
    the all-to-all kernels follow every mask that :meth:`install_mask` writes.

    Args:
        ep_group_health: The EP membership whose active ranks the mask holds.
        ep_rank: This rank's EP rank, which every mask must mark active.
        device: The device of the all-to-all workspace.
    """

    def __init__(self, ep_group_health: EPGroupHealth, ep_rank: int, device: torch.device) -> None:
        self.health = ep_group_health
        self.ep_rank = ep_rank
        self._device = device
        snapshot = ep_group_health.snapshot()
        self._generation = snapshot.generation
        self.active_rank_mask = torch.tensor(
            _mask_words(snapshot.mask, ep_rank), dtype=torch.uint64, device=device
        )

    def install_mask(self) -> None:
        """Write the active ranks of the EP group health into the device mask.

        Call only while no all-to-all collective is queued or running. The device is synchronized
        first, so no collective reads a partly written mask.
        """
        snapshot = self.health.snapshot()
        if snapshot.generation == self._generation:
            return
        words = _mask_words(snapshot.mask, self.ep_rank)
        torch.cuda.synchronize(self._device)
        self.active_rank_mask.copy_(torch.tensor(words, dtype=torch.uint64))
        self._generation = snapshot.generation
