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
"""Record of the startup collectives that a replacement rank replays, since it starts alone."""

import functools
import pickle  # nosec B403
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, NamedTuple, Optional, Sequence

from .membership import ReplacementCannotStart

_COLLECTIVE_MODULES = frozenset(
    {
        __name__,
        "tensorrt_llm._torch.distributed.communicator",
        "tensorrt_llm._torch.distributed.host_exchange",
        "tensorrt_llm._torch.distributed.membership",
        "tensorrt_llm._utils",
    }
)
# Agreement helpers, whose callers name the startup collective.
_COLLECTIVE_FUNCTIONS = frozenset(
    {("tensorrt_llm._torch.pyexecutor.engine.runners.decoder.warmup", "agree_flag")}
)

# Collectives, by file, function and collective of their caller, whose payload is a per-rank measurement or process
# identity, or the autotuner warmup's skip vote, which a replacement casts without the KV cache estimation's warmup.
# A replacement's payload to them may differ from its predecessor's.
RANK_VARYING_CALLERS = frozenset(
    {
        ("autotuner.py", "_merge_cache_data", "tp_cp_allgather"),
        ("autotuner.py", "post_tune_merge_tactics", "tp_cp_allgather"),
        ("py_executor.py", "_initialize_execution", "tp_allgather"),
        ("warmup.py", "_run_autotuner_warmup", "agree_flag"),
        ("worker.py", "worker_main", "allgather"),
    }
)


class Caller(NamedTuple):
    """The method of a startup collective and the location of its caller."""

    collective: str
    file: str
    line: int
    function: str

    def __str__(self) -> str:
        return f"{self.collective} in {self.function} ({self.file}:{self.line})"


class RecordedExchange(NamedTuple):
    """A startup collective and every rank's pickled payload as rank 0 received it."""

    caller: Caller
    payloads: List[Optional[bytes]]


def _caller() -> Caller:
    frame, collective = sys._getframe(1), ""
    while (
        frame.f_globals.get("__name__") in _COLLECTIVE_MODULES
        or (frame.f_globals.get("__name__"), frame.f_code.co_name) in _COLLECTIVE_FUNCTIONS
    ):
        collective, frame = frame.f_code.co_name, frame.f_back
    return Caller(
        collective, Path(frame.f_code.co_filename).name, frame.f_lineno, frame.f_code.co_name
    )


class StartupRecord:
    """The startup collectives' results as rank 0 got them, which a replacement rank replays.

    A replacement must make the same collectives in the same order from the same callers. Its payload must equal its
    predecessor's, except for :data:`RANK_VARYING_CALLERS` and for min or max reductions. A reduction of a measured
    capacity takes the original ranks' result, which the replacement's own capacity must still cover.
    """

    def __init__(self) -> None:
        self.values: Dict[str, List[Any]] = {}
        self.exchanges: List[RecordedExchange] = []
        self._recording = False
        self._replacement = False
        self._replaying = False
        self._taken: Dict[str, int] = {}
        self._replayed_exchanges = 0
        self._pauses = 0

    @property
    def active(self) -> bool:
        return (self._recording or self._replaying) and not self._pauses

    @property
    def replaying(self) -> bool:
        return self._replaying and not self._pauses

    def record(self) -> None:
        """Record the startup collectives on rank 0 until :meth:`close`."""
        self._recording = True

    def replay(self, published: Dict[str, Any]) -> None:
        """Replay the record that rank 0 published until :meth:`close`."""
        self.values, self.exchanges = published["values"], published["exchanges"]
        self._replacement = self._replaying = True

    def published(self) -> Dict[str, Any]:
        return {"values": self.values, "exchanges": self.exchanges}

    def pause(self) -> None:
        """Neither record nor replay the startup collectives until the matching :meth:`resume`."""
        self._pauses += 1

    def resume(self) -> None:
        self._pauses -= 1

    @contextmanager
    def paused(self) -> Iterator[None]:
        """Pause the record for the scope of the ``with`` statement."""
        self.pause()
        try:
            yield
        finally:
            self.resume()

    def ranks_sharing_free_memory(self, local_ranks: int) -> int:
        """Return how many of the ``local_ranks`` ranks on this node split the memory free on it now.

        The other ranks on a replacement's node hold their memory already, so what is free is the replacement's own.
        """
        return 1 if self._replacement else local_ranks

    def close(self) -> None:
        """Stop recording or replaying at the end of this rank's startup.

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
        """Return the result of ``gather``, or rank 0's recorded result of the same call on a replacement rank."""
        if self._replacement and not self._pauses:
            recorded = self.values.get(name, [])
            index = self._taken.get(name, 0)
            if index == len(recorded):
                raise ReplacementCannotStart(
                    f"The startup of this replacement rank takes the startup value {name} {index + 1} times, the "
                    f"original ranks {len(recorded)} times"
                )
            self._taken[name] = index + 1
            return recorded[index]
        if not self._recording or self._pauses:
            return gather()
        with self.paused():
            value = gather()
        self.values.setdefault(name, []).append(value)
        return value

    def agree_rank_capacity(
        self, name: str, rank: int, capacity: int, gather: Callable[[], Sequence[int]]
    ) -> int:
        """Return the ``capacity`` of rank ``rank``, or its predecessor's on a replacement rank.

        ``gather`` returns every rank's ``capacity``.

        Raises:
            ReplacementCannotStart: The replacement rank's own ``capacity`` is below its predecessor's.
        """
        recorded = self.agree(name, gather)[rank]
        if recorded > capacity:
            raise ReplacementCannotStart(
                f"The {name} of this replacement rank is {capacity}, below its predecessor's {recorded}"
            )
        return recorded

    def adopt_tuned_tactics(self) -> None:
        """Share rank 0's tuned tactics with a replacement, which skips the KV cache estimation warmup."""
        if not self.active:
            return
        from ..autotuner import AutoTuner

        profiling_cache = AutoTuner.get().profiling_cache
        tactics = self.agree(
            "tuned_tactics",
            lambda: profiling_cache._serialize_cache_data(
                profiling_cache._partition_cache_by_strategy()[0]
            ),
        )
        if self.replaying:
            profiling_cache.cache.update(profiling_cache._deserialize_cache_data(tactics))

    def exchange(
        self,
        rank: int,
        payload: bytes,
        readers: Optional[Sequence[int]],
        run: Callable[[], List[Optional[bytes]]],
        bound: Optional[Callable[[Any, Any], Any]] = None,
        decode: Callable[[bytes], Any] = pickle.loads,
    ) -> List[Optional[bytes]]:
        """Return every rank's pickled payload, from ``run()`` on rank 0 or from the record on a replacement rank.

        ``bound`` is ``min`` or ``max`` for a reduction by it, else None. ``decode`` turns a payload into the value
        that the replay compares, since equal values can pickle differently.

        Raises:
            ReplacementCannotStart: The collective or its payload differs from the original ranks' at that call.
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
        # Only processes with the shared secret write the rendezvous store the record comes from.
        if (
            predecessor is not None
            and (caller.file, caller.function, caller.collective) not in RANK_VARYING_CALLERS
            and decode(predecessor) != decode(payload)
        ):
            if bound is None:
                raise ReplacementCannotStart(
                    f"The payload of this replacement rank to host collective {index} of its startup, {caller}, "
                    "differs from the original rank's"
                )
            self._check_bound(index, caller, bound, decode(payload), recorded.payloads, decode)
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
    def _check_bound(
        index: int,
        caller: Caller,
        bound: Callable[[Any, Any], Any],
        own: Any,
        payloads: List[Optional[bytes]],
        decode: Callable[[bytes], Any],
    ) -> None:
        values = [decode(p) for p in payloads if p is not None]
        result = functools.reduce(bound, values)
        if bound(own, result) != result:
            raise ReplacementCannotStart(
                f"The own value of this replacement rank to host collective {index} of its startup, {caller}, is "
                f"{own}, beyond the original ranks' {bound.__name__} {result}"
            )


startup_record = StartupRecord()
