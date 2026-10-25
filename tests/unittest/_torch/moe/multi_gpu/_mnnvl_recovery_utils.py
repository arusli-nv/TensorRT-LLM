# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Helpers of the MNNVL MoE All-to-All tests that run one process per GPU over a TCPStore."""

import pickle
from typing import Any

import torch
from torch.distributed import TCPStore

from tensorrt_llm._torch.distributed.mnnvl_memory import MnnvlMemory
from tensorrt_llm.bindings import internal


class StoreComm:
    """The communicator subset MnnvlMemory uses, over a TCPStore."""

    def __init__(self, store: TCPStore, rank: int, size: int) -> None:
        self.store, self.rank, self.size, self.calls = store, rank, size, 0

    def Get_rank(self) -> int:
        return self.rank

    def Get_size(self) -> int:
        return self.size

    def exchange(self, value: Any, live: list[int]) -> list[Any]:
        """``value`` of every rank in ``live`` by rank, None for the others. Collective over ``live``."""
        self.calls += 1
        self.store.set(f"gather/{self.calls}/{self.rank}", pickle.dumps(value))
        return [
            pickle.loads(self.store.get(f"gather/{self.calls}/{r}")) if r in live else None
            for r in range(self.size)
        ]

    def allgather(self, value: Any) -> list[Any]:
        return self.exchange(value, list(range(self.size)))

    def barrier(self) -> None:
        self.allgather(None)


def workspace_offset(meta: torch.Tensor, name: str) -> int:
    """The offset of region ``name`` (``MOE_A2A_<name>_OFFSET_INDEX``) in a workspace row of layout ``meta``."""
    return int(meta[int(getattr(internal.thop, f"MOE_A2A_{name}_OFFSET_INDEX"))])


def mnnvl_skip_reason(ep_size: int) -> str | None:
    """Why a test of ``ep_size`` processes cannot run here, or None."""
    if torch.cuda.device_count() < ep_size:
        return f"Requires {ep_size} local GPUs"
    MnnvlMemory.initialize()
    if not MnnvlMemory.supports_mnnvl():
        return "Requires an NVLink fabric supported by MNNVL"
    return None
