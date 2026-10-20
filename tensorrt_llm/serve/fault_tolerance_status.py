# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fault-tolerance state for the server's health endpoint.

The recovery authority is the only holder of the committed state, so the server asks it rather than
deriving the state itself. Asking changes nothing on the authority. If it does not answer, the state is
``unprotected`` (serving, but a loss would not be recovered), never ``healthy``.
"""

import os
import queue
import threading
import time
from typing import Optional

import zmq

from tensorrt_llm._utils import peer_loss_recovery_enabled
from tensorrt_llm.executor.ipc import ZeroMqQueue
from tensorrt_llm.executor.recovery.client import AUTHORITY_ENV, load_key, unauthenticated
from tensorrt_llm.executor.recovery.protocol import MAX_MESSAGE_BYTES, Status, StatusQuery

__all__ = ["SERVING_STATES", "RecoveryStatusProbe", "health_state", "status_body"]

# Authority phase -> state reported by the server.
_STATES = {
    "REGISTERING": "starting",
    "SERVING": "healthy",
    "STOPPING": "recovering",
    "PREPARING": "recovering",
    "INSTALLING": "recovering",
    "RECOVERED": "degraded",
    "ABORTED": "terminal",
}
# States in which the server answers requests; /health returns 200 only in these. ``unprotected``: the
# authority does not answer, so a loss would end the job, but the server itself serves.
SERVING_STATES = frozenset({"healthy", "degraded", "unprotected"})
_STAGES = ("stop", "agree", "install")


def health_state(status: Optional[Status]) -> str:
    """The server state for an authority status.

    ``unprotected`` without one; ``unknown`` (not serving) for a phase this server does not know.
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
    if status is not None:
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

    Thread-safe; a query blocks for at most ``timeout_s``. A status older than ``max_age_s``
    is never returned: if the authority does not answer in time, the result is None.

    Args:
        endpoint: Authority address, e.g. ``tcp://host:5555``.
        key: HMAC key shared with the authority.
        timeout_s: How long one query waits for the answer.
        max_age_s: How long an answer is reused.
    """

    def __init__(
        self, endpoint: str, key: bytes, timeout_s: float = 0.5, max_age_s: float = 1.0
    ) -> None:
        self._timeout_s = timeout_s
        self._max_age_s = max_age_s
        self._lock = threading.Lock()
        self._status: Optional[Status] = None
        self._status_time = float("-inf")
        self._queue = ZeroMqQueue(
            (endpoint, key),
            socket_type=zmq.DEALER,
            is_server=False,
            name="recovery-status-probe",
            # One query waits while the authority is unreachable; later ones fail at once instead of piling up.
            socket_options={zmq.LINGER: 0, zmq.MAXMSGSIZE: MAX_MESSAGE_BYTES, zmq.SNDHWM: 1},
        )

    @classmethod
    def from_env(cls) -> "RecoveryStatusProbe | None":
        """A probe for the authority named by the environment; None when recovery is disabled."""
        if not peer_loss_recovery_enabled():
            return None
        return cls(os.environ[AUTHORITY_ENV], load_key())

    def status(self) -> Optional[Status]:
        """The authority's status, at most ``max_age_s`` old; None if it did not answer."""
        with self._lock:
            now = time.monotonic()
            if now - self._status_time < self._max_age_s:
                return self._status
            self._status = self._query()
            self._status_time = now
            return self._status

    def _query(self) -> Optional[Status]:
        # Answers to earlier queries that timed out may still be queued: drop them, and any message that
        # fails authentication.
        while True:
            try:
                self._queue.drain()
                break
            except RuntimeError as error:
                if not unauthenticated(error):
                    raise
        try:
            self._queue.put_nowait(StatusQuery())
        except zmq.Again:
            return None
        deadline = time.monotonic() + self._timeout_s
        while (remaining := deadline - time.monotonic()) > 0:
            try:
                message = self._queue.get(remaining)
            except queue.Empty:
                break
            except RuntimeError as error:
                if not unauthenticated(error):
                    raise
                continue
            if isinstance(message, Status):
                return message
        return None

    def close(self) -> None:
        self._queue.close()
