# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fault-tolerance state for the server's health endpoint.

The recovery authority is the only holder of the committed state, so the server asks it rather than
deriving the state itself. Asking changes nothing on the authority. If it does not answer, the state is
``unprotected`` (serving, but a rank failure would not be recovered), never ``healthy``.
"""

import os
import queue
import time
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Optional

import zmq

from tensorrt_llm._utils import FAULT_TOLERANCE_AUTHORITY_ENV, rank_failure_recovery_enabled
from tensorrt_llm.executor.recovery.client import MessageQueue, load_key, receive_message
from tensorrt_llm.executor.recovery.protocol import (
    MAX_MESSAGE_BYTES,
    MalformedMessage,
    Status,
    StatusQuery,
    Unauthenticated,
)
from tensorrt_llm.logger import logger

__all__ = ["SERVING_STATES", "RecoveryStatusProbe", "health_state", "status_body"]

# Authority phase -> state reported by the server. The server asks only after every worker registered, so an
# authority still ``REGISTERING`` restarted and lost the committed state: workers register once per process, so
# a rank failure would end the job. That is ``unprotected``, as when the authority does not answer.
_STATES = {
    "REGISTERING": "unprotected",
    "SERVING": "healthy",
    "STOPPING": "recovering",
    "PREPARING": "recovering",
    "INSTALLING": "recovering",
    "DEGRADED": "degraded",
    "ABORTED": "terminal",
}
# States in which the server answers requests; /health returns 200 only in these. ``unprotected``: the
# authority does not answer or restarted, so a rank failure would end the job, but the server itself serves.
# ``unknown``: the authority runs another version; the server still serves.
SERVING_STATES = frozenset({"healthy", "degraded", "unprotected", "unknown"})
_STAGES = ("stop", "prepare", "install")


def health_state(status: Optional[Status]) -> str:
    """The server state for an authority status.

    ``unprotected`` without one; ``unknown`` (still serving) for a phase this server does not know.
    """
    if status is None:
        return "unprotected"
    return _STATES.get(status.phase, "unknown")


def status_body(status: Optional[Status]) -> dict:
    """JSON body of ``/health``: the state, and the authority's details when it answered.

    The endpoint is unauthenticated, so an abort reason, which names ranks, pids and hosts, stays in the
    authority's log.
    """
    body = {"state": health_state(status)}
    if body["state"] != "unprotected":
        body.update(
            generation=status.generation,
            live_ranks=list(status.live_ranks),
            failed_ranks=list(status.failed_ranks),
            recoveries=status.recoveries,
            last_recovery_stage_ms={
                stage: us / 1000 for stage, us in zip(_STAGES, status.recovery_stage_us)
            },
            failed_requests=status.failed_requests,
        )
    return body


class RecoveryStatusProbe:
    """Asks the recovery authority for its status, at most once per ``max_age_s``.

    Thread-safe: one thread of its own uses the socket, which ZeroMQ requires. A query blocks for at most
    ``timeout_s``. A status older than ``max_age_s`` is never returned: if the authority does not answer in
    time, the result is None.

    Args:
        endpoint: Authority address, e.g. ``tcp://host:5555``.
        key: HMAC key shared with the authority.
        timeout_s: How long one query waits for the answer.
        max_age_s: How long an answer is reused; short against a recovery's ~1 s, while it bounds the queries
            the authority answers however often /health is polled.
    """

    def __init__(
        self, endpoint: str, key: bytes, timeout_s: float = 0.5, max_age_s: float = 0.1
    ) -> None:
        self._timeout_s = timeout_s
        self._max_age_s = max_age_s
        self._thread = ThreadPoolExecutor(max_workers=1, thread_name_prefix="recovery-status-probe")
        self._status: Optional[Status] = None
        self._status_time = float("-inf")
        self._queue = MessageQueue(
            (endpoint, key),
            socket_type=zmq.DEALER,
            is_server=False,
            name="recovery-status-probe",
            # One query waits while the authority is unreachable; later ones fail at once instead of piling up.
            socket_options={zmq.LINGER: 0, zmq.MAXMSGSIZE: MAX_MESSAGE_BYTES, zmq.SNDHWM: 1},
        )

    @classmethod
    def from_env(cls) -> Optional["RecoveryStatusProbe"]:
        """A probe for the authority named by the environment; None when recovery is disabled."""
        if not rank_failure_recovery_enabled():
            return None
        return cls(os.environ[FAULT_TOLERANCE_AUTHORITY_ENV], load_key())

    def status(self) -> Optional[Status]:
        """The authority's status, at most ``max_age_s`` old; None if it did not answer."""
        return self.status_future().result()

    def status_future(self) -> Future:
        """:meth:`status`, answered on the probe's own thread; an event loop awaits it without a thread of its own."""
        return self._thread.submit(self._cached_status)

    def _cached_status(self) -> Optional[Status]:
        # Stamped when the answer arrives: callers queued behind a slow query reuse its answer.
        if time.monotonic() - self._status_time >= self._max_age_s:
            self._status = self._query()
            self._status_time = time.monotonic()
        return self._status

    def _query(self) -> Optional[Status]:
        # Answers to earlier queries that timed out may still be queued: drop them.
        try:
            while True:
                self._receive(0)
        except queue.Empty:
            pass
        try:
            self._queue.put_nowait(StatusQuery())
        except zmq.Again:
            return None
        deadline = time.monotonic() + self._timeout_s
        while (remaining := deadline - time.monotonic()) > 0:
            try:
                message = self._receive(remaining)
            except queue.Empty:
                break
            if type(message) is Status:
                return message
        return None

    def _receive(self, timeout: float) -> Optional[object]:
        """The next message within ``timeout`` s; None for one that fails authentication or is malformed.

        Raises:
            queue.Empty: No message arrived.
        """
        try:
            return receive_message(self._queue, timeout)
        except MalformedMessage as error:
            # E.g. an authority of another version.
            logger.warning(f"Recovery status probe: dropped {error}")
        except Unauthenticated:
            pass
        return None

    def close(self) -> None:
        """Stop the probe's thread and close its connection."""
        self._thread.shutdown()
        self._queue.close()
