# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Host-exchange epochs, importable without GPU dependencies."""

from typing import List

# Epochs wrap from _EPOCHS to 1, skipping 0, so the parity keeps alternating.
_EPOCHS = 0xFFFFFFFE


def next_epoch(epoch: int) -> int:
    """Return the host-exchange epoch after ``epoch``."""
    return epoch % _EPOCHS + 1


def latest_epoch(epochs: List[int]) -> int:
    """Return the latest of ``epochs``, which lie within half the epoch range of each other."""
    return max(epochs, key=lambda epoch: (epoch - epochs[0] + _EPOCHS // 2) % _EPOCHS)
