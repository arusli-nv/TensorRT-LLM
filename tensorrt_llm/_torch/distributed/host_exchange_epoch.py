# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Epochs of :class:`~tensorrt_llm._torch.distributed.ll_exchange.TPHostExchange`, without its GPU dependencies,
so the CPU-only membership authority can compute them."""

from typing import List

# TPHostExchange epochs run from 1 to _EPOCHS, then wrap to 1, keeping the parity alternating (0 is never sent).
_EPOCHS = 0xFFFFFFFE


def next_epoch(epoch: int) -> int:
    """The host-exchange epoch after ``epoch``."""
    return epoch % _EPOCHS + 1


def latest_epoch(epochs: List[int]) -> int:
    """The latest of host-exchange epochs that lie within half the epoch range of each other, across the wrap."""
    return max(epochs, key=lambda epoch: (epoch - epochs[0] + _EPOCHS // 2) % _EPOCHS)
