# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fault-tolerance state for the server's health and status endpoints.

The recovery authority is the only holder of the committed state, so the server asks it rather than
deriving the state itself. Asking changes nothing on the authority. If it does not answer, the state is
``unprotected`` (serving, but a loss would not be recovered), never ``healthy``.
"""

import os
import secrets
import threading
import time
from typing import Optional

import zmq

from tensorrt_llm._utils import peer_loss_recovery_enabled
from tensorrt_llm.executor.recovery.client import AUTHORITY_ENV, load_key
from tensorrt_llm.executor.recovery.protocol import (
    ProtocolError,
    Status,
    StatusQuery,
    authenticate,
    decode,
    encode,
)

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
    """JSON body of the fault-tolerance status endpoint."""
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
        if status.abort_reason:
            body["abort_reason"] = status.abort_reason
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
        self._key = key
        self._timeout_ms = int(timeout_s * 1000)
        self._max_age_s = max_age_s
        self._lock = threading.Lock()
        self._status: Optional[Status] = None
        self._status_time = float("-inf")
        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.DEALER)
        self._socket.setsockopt(zmq.LINGER, 0)
        # While the authority is unreachable, a query fails at once instead of queueing behind earlier ones
        # (a DEALER queues up to its high-water mark, then blocks the sender: /health would hang).
        self._socket.setsockopt(zmq.IMMEDIATE, 1)
        self._route = f"observer-{secrets.token_hex(8)}".encode()
        self._socket.setsockopt(zmq.ROUTING_ID, self._route)
        self._socket.connect(endpoint)

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
        # Answers to earlier queries that timed out may still be queued: drop them.
        while self._socket.poll(0):
            self._socket.recv_multipart()
        deadline = time.monotonic() + self._timeout_ms / 1000
        # IMMEDIATE refuses to queue before the connection is up (right after connect, or while the authority
        # is down): wait for it within the timeout instead of failing the first query.
        if not self._socket.poll(self._timeout_ms, zmq.POLLOUT):
            return None
        try:
            self._socket.send_multipart(
                encode(StatusQuery(), self._key, self._route), flags=zmq.NOBLOCK
            )
        except zmq.Again:
            return None
        while (remaining := deadline - time.monotonic()) > 0:
            if not self._socket.poll(int(remaining * 1000)):
                break
            frames = self._socket.recv_multipart()
            if not authenticate(frames, self._key, self._route):
                continue
            try:
                message = decode(frames[0])
            except ProtocolError:
                continue
            if isinstance(message, Status):
                return message
        return None

    def close(self) -> None:
        self._socket.close()
        self._context.term()
