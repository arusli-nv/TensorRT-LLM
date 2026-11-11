# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Failures of MNNVL memory mapped among the live members of a group."""

from typing import Any, Optional, Sequence


class MnnvlPeerHandleError(RuntimeError):
    """Raised when the MNNVL handles of peers ``ranks`` cannot be imported, so a caller can exclude them."""

    def __init__(self, ranks: Sequence[int], details: str = "") -> None:
        self.ranks = tuple(sorted(set(ranks)))
        message = f"Cannot import the MNNVL handles of ranks {list(self.ranks)}"
        super().__init__(f"{message}\n{details}" if details else message)


class MnnvlMappingError(RuntimeError):
    """Raised on every live rank when MNNVL memory cannot be mapped on ``ranks`` for reasons of their own."""

    def __init__(self, ranks: Sequence[int], message: str) -> None:
        self.ranks = tuple(sorted(set(ranks)))
        super().__init__(message)


def _mapping_outcome(failure: Exception, rank: int, membership: Sequence[int]) -> tuple:
    if isinstance(failure, MnnvlMappingError) and rank not in failure.ranks:
        return ()
    peers = failure.ranks if isinstance(failure, MnnvlPeerHandleError) else ()
    return f"{type(failure).__name__}: {failure}", [membership[peer] for peer in peers]


def _raise_reported_failures(
    what: str, outcomes: Sequence[Any], local_failure: Optional[Exception]
) -> None:
    """Raise the same error on every live rank for the failures the live ranks reported."""
    failures = [(rank, reported) for rank, reported in enumerate(outcomes) if reported]
    if not failures:
        if local_failure is not None:
            raise local_failure
        return
    message = f"{what} failed on live ranks:\n" + "\n".join(
        f"rank {rank}: {text}" for rank, (text, _) in failures
    )
    lost = [peer for _, (_, peers) in failures for peer in peers]
    if lost:
        raise MnnvlPeerHandleError(lost, message) from local_failure
    raise MnnvlMappingError([rank for rank, _ in failures], message) from local_failure
