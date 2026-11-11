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
"""The startup record of a WideEP fault-tolerant deployment.

A replacement rank starts alone (isolated startup): its host collectives reach only itself. So rank 0 records the
results of the original ranks' startup collectives, and a replacement replays them: each of its startup collectives
returns what rank 0 received at the same call, and fails the replacement's startup
(:class:`~tensorrt_llm._torch.distributed.membership.ReplacementCannotStart`) where the replacement does not start as
the original ranks did.
"""

import pickle  # nosec B403
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, NamedTuple, Optional, Sequence

from .membership import ReplacementCannotStart

# The modules whose frames lie between a startup collective's caller and the record.
_COLLECTIVE_MODULES = frozenset(
    {
        __name__,
        "tensorrt_llm._torch.distributed.communicator",
        "tensorrt_llm._torch.distributed.host_exchange",
    }
)

# The callers, by (file, function), whose payload is the rank's own measurement, which a replacement's differs from:
# they take the original ranks' payloads, its predecessor's included.
RANK_VARYING_CALLERS = frozenset(
    {
        ("autotuner.py", "_merge_cache_data"),
        ("autotuner.py", "post_tune_merge_tactics"),
    }
)


class Caller(NamedTuple):
    """Where a startup collective was called: the ``Distributed`` method and its caller's file, line and function."""

    collective: str
    file: str
    line: int
    function: str

    def __str__(self) -> str:
        return f"{self.collective} in {self.function} ({self.file}:{self.line})"


class RecordedExchange(NamedTuple):
    """A startup collective as rank 0 ran it: where it was called, and every rank's pickled payload that rank 0
    received (its own included; None for a rank whose payload it did not read)."""

    caller: Caller
    payloads: List[Optional[bytes]]


def _caller() -> Caller:
    frame, collective = sys._getframe(1), ""
    while frame.f_globals.get("__name__") in _COLLECTIVE_MODULES:
        collective, frame = frame.f_code.co_name, frame.f_back
    return Caller(
        collective, Path(frame.f_code.co_filename).name, frame.f_lineno, frame.f_code.co_name
    )


class StartupRecord:
    """The results of the startup collectives of the original ranks, as rank 0 got them, for a replacement rank.

    Rank 0 records from :meth:`record`, a replacement replays from :meth:`replay`, until :meth:`close` at the end of
    the startup. Two kinds of results:

    - Values (:meth:`agree`), by name and call: those every rank needs before its host collectives exist, or once per
      process, whenever it first needs them.
    - Exchanges (:meth:`exchange`): every other host collective of ``MnnvlDist``, by call: a replacement makes them in
      the order the original ranks did, from the same callers. Its own payload must equal its predecessor's, except
      for :data:`RANK_VARYING_CALLERS`, and for a reduction by min (max), where it must be at least (at most) the
      result, so the result bounds the replacement too.

    Calls while the record is paused (:meth:`pause`) are neither recorded nor replayed: they reach the ranks the
    host exchange reaches.
    """

    def __init__(self) -> None:
        self.values: Dict[str, List[Any]] = {}
        self.exchanges: List[RecordedExchange] = []
        self._recording = False
        # A replacement rank: whether it replays exchanges (until close), and how many results it took.
        self._replacement = False
        self._replaying = False
        self._taken: Dict[str, int] = {}
        self._replayed_exchanges = 0
        self._pauses = 0

    @property
    def active(self) -> bool:
        """Whether this rank records or replays its host collectives now."""
        return (self._recording or self._replaying) and not self._pauses

    @property
    def replaying(self) -> bool:
        """Whether this rank's host collectives return the original ranks' results now (a replacement rank)."""
        return self._replaying and not self._pauses

    def record(self) -> None:
        """Rank 0: record the startup collectives' results until :meth:`close`."""
        self._recording = True

    def replay(self, published: Dict[str, Any]) -> None:
        """A replacement rank: take its startup collectives' results from ``published``, the record rank 0
        published (:meth:`published`); its exchanges until :meth:`close`."""
        self.values, self.exchanges = published["values"], published["exchanges"]
        self._replacement = self._replaying = True

    def published(self) -> Dict[str, Any]:
        """What a replacement rank replays (:meth:`replay`)."""
        return {"values": self.values, "exchanges": self.exchanges}

    def pause(self) -> None:
        """Neither record nor replay the host collectives until the matching :meth:`resume`."""
        self._pauses += 1

    def resume(self) -> None:
        self._pauses -= 1

    def close(self) -> None:
        """The end of this rank's startup: rank 0 stops recording, a replacement rank replaying exchanges.

        Raises:
            ReplacementCannotStart: A replacement rank made fewer startup collectives than the original ranks.
        """
        if self._replacement:
            counts = {name: len(values) for name, values in self.values.items()}
            taken = {name: self._taken.get(name, 0) for name in counts}
            if taken != counts:
                raise ReplacementCannotStart(
                    f"The startup of this replacement rank takes the startup values {taken} times, the original "
                    f"ranks {counts} times"
                )
            if self._replayed_exchanges != len(self.exchanges):
                raise ReplacementCannotStart(
                    f"The startup of this replacement rank makes {self._replayed_exchanges} host collectives, the "
                    f"original ranks {len(self.exchanges)}: the next one is "
                    f"{self.exchanges[self._replayed_exchanges].caller}"
                )
        self._recording = self._replaying = False

    def agree(self, name: str, gather: Callable[[], Any]) -> Any:
        """The result of ``gather``, a startup collective that needs every rank, as rank 0 got it at the same call of
        ``name``: a replacement rank does not run ``gather``. Without fault tolerance, ``gather()``."""
        if self._replacement:
            recorded = self.values.get(name, [])
            index = self._taken.get(name, 0)
            if index == len(recorded):
                raise ReplacementCannotStart(
                    f"The startup of this replacement rank takes the startup value {name} {index + 1} times, the "
                    f"original ranks {len(recorded)} times"
                )
            self._taken[name] = index + 1
            return recorded[index]
        if not self._recording:
            return gather()
        self.pause()
        try:
            value = gather()
        finally:
            self.resume()
        self.values.setdefault(name, []).append(value)
        return value

    def exchange(
        self,
        rank: int,
        payload: bytes,
        readers: Optional[Sequence[int]],
        run: Callable[[], List[Optional[bytes]]],
        bound: Optional[Callable[[Any, Any], Any]] = None,
    ) -> List[Optional[bytes]]:
        """Every rank's pickled payload of a host collective while :attr:`active`: ``run()`` on rank 0, which records
        it; rank 0's at the same call on a replacement rank.

        Args:
            rank: This rank.
            payload: This rank's pickled payload.
            readers: The ranks whose payloads this rank reads (None: every rank); the others come back as None.
            run: Runs the collective; returns every rank's payload.
            bound: ``min`` or ``max`` for a reduction by it, else None.

        Raises:
            ReplacementCannotStart: A replacement rank's collective is not the original ranks' at that call (another
                collective or caller, or one more than they made), or its payload differs from its predecessor's.
        """
        caller = _caller()
        if self._recording:
            payloads = run()
            recorded = list(payloads)
            recorded[rank] = payload
            self.exchanges.append(RecordedExchange(caller, recorded))
            return payloads
        index = self._replayed_exchanges
        if index == len(self.exchanges):
            raise ReplacementCannotStart(
                f"Host collective {index} of this replacement rank's startup, {caller}, is one more than the "
                "original ranks made"
            )
        recorded = self.exchanges[index]
        if recorded.caller != caller:
            raise ReplacementCannotStart(
                f"Host collective {index} of this replacement rank's startup is {caller}, the original ranks' "
                f"{recorded.caller}"
            )
        predecessor = recorded.payloads[rank]
        if (
            predecessor is not None
            and predecessor != payload
            and (caller.file, caller.function) not in RANK_VARYING_CALLERS
            and not self._bounded(bound, payload, recorded.payloads)
        ):
            raise ReplacementCannotStart(
                f"The payload of this replacement rank to host collective {index} of its startup, {caller}, differs "
                "from the original rank's"
            )
        readers = range(len(recorded.payloads)) if readers is None else readers
        unread = [r for r in readers if recorded.payloads[r] is None]
        if unread:
            raise ReplacementCannotStart(
                f"Host collective {index} of this replacement rank's startup, {caller}, reads the payloads of ranks "
                f"{unread}, which rank 0 did not receive"
            )
        self._replayed_exchanges += 1
        return [p if r in readers else None for r, p in enumerate(recorded.payloads)]

    @staticmethod
    def _bounded(
        bound: Optional[Callable[[Any, Any], Any]], payload: bytes, payloads: List[Optional[bytes]]
    ) -> bool:
        """Whether the reduction of ``payloads`` by ``bound`` (min or max) bounds ``payload`` as it bounds theirs."""
        if bound is None:
            return False
        # Only processes with the join secret write the rendezvous store the record comes from.
        values = [pickle.loads(p) for p in payloads if p is not None]  # nosec B301
        result = values[0]
        for value in values[1:]:
            result = bound(result, value)
        return bound(pickle.loads(payload), result) == result  # nosec B301


# This process's startup record.
startup_record = StartupRecord()
