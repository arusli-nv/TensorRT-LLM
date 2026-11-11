# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The failures of MNNVL memory mapped among the live members of a group (``membership.StoreGroup``), as every live
member reports them."""

from typing import Any, Optional, Sequence


class MnnvlPeerHandleError(RuntimeError):
    """A peer's exported MNNVL handle could not be imported, as when the peer exited after exporting it.

    ``ranks`` are the peers' ranks in the numbering of the exchange that carried the handles. A caller changing
    membership can exclude them and retry.
    """

    def __init__(self, ranks: Sequence[int], details: str = "") -> None:
        self.ranks = tuple(sorted(set(ranks)))
        message = f"Cannot import the MNNVL handles of ranks {list(self.ranks)}"
        super().__init__(f"{message}\n{details}" if details else message)


class MnnvlMappingError(RuntimeError):
    """MNNVL memory could not be mapped on the live ranks ``ranks``, each for a reason of its own (not a peer's
    handle it could not import); every live rank of the exchange raises it.

    ``ranks`` are numbered as in the exchange that carried the failures. A caller changing membership reports the
    failure on those ranks only.
    """

    def __init__(self, ranks: Sequence[int], message: str) -> None:
        self.ranks = tuple(sorted(set(ranks)))
        super().__init__(message)


def _mapping_outcome(failure: Exception, rank: int, membership: Sequence[int]) -> tuple:
    """What live rank ``rank`` of an exchange, whose mapping raised ``failure``, reports to the others: its failure's
    text and the global ranks (``membership``) of the peers whose handles it could not import; () if ``failure``
    only names other ranks, which report their own failures."""
    if isinstance(failure, MnnvlMappingError) and rank not in failure.ranks:
        return ()
    peers = failure.ranks if isinstance(failure, MnnvlPeerHandleError) else ()
    return f"{type(failure).__name__}: {failure}", [membership[peer] for peer in peers]


def _raise_reported_failures(
    what: str, outcomes: Sequence[Any], local_failure: Optional[Exception]
) -> None:
    """Raise alike on every live rank the failures of ``what`` that the live ranks reported (``outcomes``, by global
    rank): MnnvlPeerHandleError naming the peers whose handles some rank could not import, else MnnvlMappingError
    naming the ranks that failed. Without a reported failure, raises this rank's ``local_failure``, if any."""
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
