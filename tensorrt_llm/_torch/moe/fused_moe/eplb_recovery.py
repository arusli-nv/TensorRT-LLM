# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""EPLB placement planning for WideEP peer-loss recovery: restoring lost experts and staging ring host copies."""

from typing import Iterable, List, Optional, Tuple

import numpy as np


def _align_size(size: int) -> int:
    """``HostMoeTensorSharer.align_size``, imported on use: the load balancer imports this module."""
    from .moe_load_balancer import HostMoeTensorSharer

    return HostMoeTensorSharer.align_size(size)


def restore_lost_experts(
    placement: np.ndarray,
    lost: List[int],
    dead_ranks: Iterable[int],
    restored_per_rank: Optional[np.ndarray] = None,
) -> List[int]:
    """Place each ``lost`` expert into a slot of a live rank in ``placement`` (``[ep_size, slots_per_rank]``
    expert ids, -1 = no expert), in place.

    Empty (ghost) slots of live ranks are taken first; otherwise the slot of a
    replica of the expert with the most replicas (so no expert loses
    its last copy). Ties go to the rank that has taken the fewest restored
    experts so far, then the lowest rank and slot: a minimal change that
    spreads the loading across ranks and is the same on every rank.

    Args:
        restored_per_rank: ``[ep_size]`` restored experts per rank so far, updated in place. Pass the same
            array for every layer of a plan to spread the restored experts over the ranks across layers.

    Returns:
        The lost experts that found no slot.
    """
    usable = np.ones(placement.shape, dtype=bool)
    usable[list(dead_ranks)] = False
    counts = np.bincount(placement[placement >= 0], minlength=max(lost) + 1 if lost else 0)
    if restored_per_rank is None:
        restored_per_rank = np.zeros(placement.shape[0], dtype=np.int64)
    unplaced = []
    for expert in lost:
        ranks, slots = np.nonzero(usable)
        held = placement[ranks, slots]
        replicas = np.where(held >= 0, counts[np.maximum(held, 0)], 0)
        free = (held < 0) | (replicas >= 2)
        if not free.any():
            unplaced.append(expert)
            continue
        ranks, slots, held, replicas = (a[free] for a in (ranks, slots, held, replicas))
        best = np.lexsort((slots, ranks, restored_per_rank[ranks], -replicas, held >= 0))[0]
        rank, slot = ranks[best], slots[best]
        if held[best] >= 0:
            counts[held[best]] -= 1
        placement[rank, slot] = expert
        counts[expert] = 1
        restored_per_rank[rank] += 1
    return unplaced


def staging_halves(
    row_bytes: int, taken: List[Tuple[int, int]]
) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
    """The byte ranges ``[start, end)`` of a ``row_bytes`` workspace row outside ``taken``, as the two halves
    ring restores alternate between: split at the middle of the largest range."""
    ranges, start = [], 0
    for begin, end in sorted(taken):
        if begin > start:
            ranges.append((start, begin))
        start = max(start, end)
    if start < row_bytes:
        ranges.append((start, row_bytes))
    begin, end = max(ranges, key=lambda r: r[1] - r[0], default=(0, 0))
    split = (begin + end) // 2
    return (
        [(b, min(e, split)) for b, e in ranges if b < split],
        [(max(b, split), e) for b, e in ranges if e > split],
    )


def staging_room(half: List[Tuple[int, int]]) -> int:
    """The bytes of ``half`` that aligned pieces can use."""
    return sum(max(0, end - _align_size(begin)) for begin, end in half)


def no_staging_room(
    row_bytes: int, halves: Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]
) -> str:
    return (
        f"Ring host copies have {[staging_room(half) for half in halves]} bytes per half of the {row_bytes}-byte "
        "NVLinkOneSided workspace rows to stage restores through, and need some in each; raise "
        "max_num_tokens or TRTLLM_NVLINK_ONE_SIDED_A2A_WORKSPACE_MB"
    )


def next_piece(half: List[Tuple[int, int]], cursor: int, size: int) -> Optional[Tuple[int, int]]:
    """(offset, bytes) of the first aligned piece of at most ``size`` bytes in ``half`` from ``cursor``, or None."""
    for begin, end in half:
        start = _align_size(max(begin, cursor))
        if start < end:
            return start, min(size, end - start)
    return None


def ring_copy_nodes(hosts: List[str]) -> List[Tuple[int, ...]]:
    """Per rank (``hosts``: its host), the ranks of the node keeping its home experts' copies: the node before
    its own in first-rank order, the last node for the first (a single node keeps its own)."""
    nodes = list(dict.fromkeys(hosts))
    copy_nodes = {
        host: tuple(rank for rank, h in enumerate(hosts) if h == previous)
        for host, previous in zip(nodes, nodes[-1:] + nodes[:-1])
    }
    return [copy_nodes[host] for host in hosts]
