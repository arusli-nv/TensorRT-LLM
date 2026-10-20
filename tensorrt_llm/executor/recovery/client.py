# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Worker-side connection to the recovery authority."""

import itertools
import os
import queue
import secrets
import threading
import time
from collections.abc import Callable
from typing import TypeVar

import zmq

from tensorrt_llm._utils import FAULT_TOLERANCE_AUTHORITY_ENV, peer_loss_recovery_enabled

from ..ipc import ZeroMqQueue
from .config import fault_tolerance_config
from .protocol import MAX_MESSAGE_BYTES, Abort, Message, Register, Registered, Sent, is_message

__all__ = [
    "AUTHORITY_ENV",
    "CONNECTION_OPTIONS",
    "AUTHORITY_KEY_ENV",
    "RecoveryAborted",
    "RecoveryClient",
    "RecoveryError",
    "RecoveryInterrupted",
    "RecoveryTimeout",
    "load_key",
    "unauthenticated",
]

AUTHORITY_ENV = FAULT_TOLERANCE_AUTHORITY_ENV
AUTHORITY_KEY_ENV = "TLLM_FAULT_TOLERANCE_AUTHORITY_KEY"

_MIN_KEY_BYTES = 16
# This process's session and the sequence of its messages, shared by all of its connections.
_SESSION = secrets.token_hex(16)
_SEQUENCE = itertools.count(1)
# How often a wait with an interrupt checks it.
_INTERRUPT_POLL_S = 0.005
# ZMTP heartbeats on both ends of a connection: a connection that dies silently (a half-open TCP connection, an
# expired NAT or conntrack entry) is closed within the timeout and the worker reconnects, rather than the next
# recovery's messages vanishing. The pings also keep an idle connection's conntrack entries alive.
CONNECTION_OPTIONS = {
    zmq.HEARTBEAT_IVL: 2000,
    zmq.HEARTBEAT_TIMEOUT: 10000,
    zmq.HEARTBEAT_TTL: 10000,
}

_M = TypeVar("_M")


def load_key() -> bytes:
    """Return the HMAC key held in ``TLLM_FAULT_TOLERANCE_AUTHORITY_KEY`` as hex.

    Raises:
        ValueError: The key is missing, not hex, or shorter than 16 bytes.
    """
    text = os.environ.get(AUTHORITY_KEY_ENV)
    if not text:
        raise ValueError(f"{AUTHORITY_KEY_ENV} is not set")
    try:
        key = bytes.fromhex(text)
    except ValueError as error:
        raise ValueError(f"{AUTHORITY_KEY_ENV} is not a hex string") from error
    if len(key) < _MIN_KEY_BYTES:
        raise ValueError(f"{AUTHORITY_KEY_ENV} must hold at least {_MIN_KEY_BYTES} bytes")
    return key


def unauthenticated(error: Exception) -> bool:
    """Whether ``error``, raised by ``ZeroMqQueue.get()``, refuses a message that does not authenticate.

    The queue raises RuntimeError when the HMAC does not verify, and on a ROUTER a ValueError when
    the message is not one frame after the routing id; any other error is not the peer's.
    """
    if isinstance(error, RuntimeError):
        return str(error) == "HMAC verification failed"
    return isinstance(error, ValueError) and "values to unpack" in str(error)


class RecoveryError(RuntimeError):
    """The recovery exchange failed; the caller must take the terminal path."""


class RecoveryAborted(RecoveryError):
    """The authority aborted recovery."""


class RecoveryTimeout(RecoveryError):
    """The authority did not answer before the deadline."""


class RecoveryInterrupted(Exception):
    """The caller's interrupt was set while it waited for the authority (another peer failed)."""


class RecoveryClient:
    """DEALER connection of one worker rank to the recovery authority.

    The routing id is derived from the rank and kept across reconnects, so the authority
    recognizes the worker after one: the reconnect takes the route over from the connection
    that died. Not thread-safe: use each client from one thread at a time.

    Args:
        endpoint: Authority address, e.g. ``tcp://host:5555``.
        key: HMAC key shared with the authority.
        rank: Logical worker rank.
    """

    def __init__(self, endpoint: str, key: bytes, rank: int) -> None:
        self._rank = rank
        self._endpoint = endpoint
        self._queue = ZeroMqQueue(
            (endpoint, key),
            socket_type=zmq.DEALER,
            is_server=False,
            name=f"recovery-rank-{rank}",
            socket_options={
                zmq.LINGER: 0,
                zmq.MAXMSGSIZE: MAX_MESSAGE_BYTES,
                zmq.ROUTING_ID: f"rank-{rank}-{secrets.token_hex(8)}".encode(),
                **CONNECTION_OPTIONS,
            },
        )

    @classmethod
    def from_env(cls, rank: int) -> "RecoveryClient | None":
        """Connect to the authority named by the environment; None when recovery is disabled."""
        if not peer_loss_recovery_enabled():
            return None
        return cls(os.environ[AUTHORITY_ENV], load_key(), rank)

    def register(
        self, recoverable_domains: tuple[tuple[int, ...], ...], deadline: float
    ) -> frozenset[frozenset[int]]:
        """Register this rank and wait until every rank has registered.

        Args:
            recoverable_domains: Sorted rank sets (a single rank or a whole node) whose
                combined loss this worker's expert placement covers.
            deadline: Absolute ``time.monotonic()`` value.

        Returns:
            The failure domains whose loss the deployment recovers from.

        Raises:
            RecoveryError: Registration was rejected or timed out.
        """
        domains = tuple(map(tuple, recoverable_domains))
        self.send(Register(self._rank, domains, fault_tolerance_config().digest()))
        return frozenset(map(frozenset, self.wait(Registered, deadline).admitted_domains))

    def send(self, message: Message) -> None:
        """Queue ``message`` for the authority; never blocks (the queue has no high-water mark)."""
        self._queue.put(Sent(self._rank, _SESSION, next(_SEQUENCE), message))

    def _recv(self, timeout: float) -> Message | None:
        """Return the next authority message, or None if none arrives within ``timeout`` s.

        Raises:
            RecoveryError: A message failed authentication or is not a protocol message.
        """
        try:
            message = self._queue.get(timeout)
        except queue.Empty:
            return None
        except RuntimeError as error:
            if not unauthenticated(error):
                raise
            raise RecoveryError("Received an unauthenticated message from the authority") from error
        if not is_message(message):
            raise RecoveryError(f"Received a malformed message: {type(message).__name__}")
        return message

    def discard_pending(self) -> None:
        """Drop every authority message already received: none belongs to a recovery that has not started.

        One is left from serving (the refusal of a message) or from an earlier recovery. If the
        authority aborted, it answers this rank's next message with ``Abort`` again.
        """
        while True:
            try:
                if self._recv(0) is None:
                    return
            except RecoveryError:
                continue

    def receive(self) -> Message | None:
        """Return an authority message already received, or None (does not wait).

        Raises:
            RecoveryError: A message failed authentication or is not a protocol message.
        """
        return self._recv(0)

    def wait(
        self,
        message_type: type[_M] | tuple[type, ...],
        deadline: float,
        interrupt: threading.Event | None = None,
        skip: Callable[[Message], bool] | None = None,
    ) -> _M:
        """Block until a ``message_type`` message (or one of a tuple of types) arrives.

        Args:
            message_type: Expected message class, or a tuple of them.
            deadline: Absolute ``time.monotonic()`` value.
            interrupt: Checked every ``_INTERRUPT_POLL_S`` while waiting.
            skip: Messages it accepts are dropped (left over from an earlier round).

        Raises:
            RecoveryAborted: The authority sent ``Abort``.
            RecoveryTimeout: ``deadline`` passed first.
            RecoveryInterrupted: ``interrupt`` was set and no message was waiting.
            RecoveryError: Any other unexpected message arrived.
        """
        expected = (
            message_type.__name__
            if isinstance(message_type, type)
            else " or ".join(cls.__name__ for cls in message_type)
        )
        while (remaining := deadline - time.monotonic()) > 0:
            interrupted = interrupt is not None and interrupt.is_set()
            if interrupt is not None:
                remaining = min(remaining, _INTERRUPT_POLL_S)
            # Messages that arrived before the interrupt still count (e.g. an Install the other ranks act on).
            message = self._recv(0 if interrupted else remaining)
            if message is None and interrupted:
                raise RecoveryInterrupted(f"Rank {self._rank} stopped waiting for {expected}")
            if message is None or (skip is not None and skip(message)):
                continue
            if isinstance(message, Abort):
                raise RecoveryAborted(f"Recovery aborted by authority: {message.reason}")
            if not isinstance(message, message_type):
                raise RecoveryError(f"Expected {expected}, received {message}")
            return message
        raise RecoveryTimeout(
            f"Rank {self._rank} timed out waiting for {expected} from the recovery "
            f"authority at {self._endpoint}"
        )

    def close(self) -> None:
        """Close the connection."""
        self._queue.close()
