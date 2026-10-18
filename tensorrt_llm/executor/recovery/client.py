# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Worker-side connection to the recovery authority."""

import os
import secrets
import threading
import time
from collections.abc import Callable
from typing import TypeVar

import zmq

from tensorrt_llm._utils import FAULT_TOLERANCE_AUTHORITY_ENV, peer_loss_recovery_enabled

from .config import fault_tolerance_config
from .protocol import (
    MAX_MESSAGE_BYTES,
    Abort,
    Message,
    ProtocolError,
    Register,
    Registered,
    authenticate,
    decode,
    encode,
)

__all__ = [
    "AUTHORITY_ENV",
    "AUTHORITY_KEY_ENV",
    "RecoveryAborted",
    "RecoveryClient",
    "RecoveryError",
    "RecoveryInterrupted",
    "RecoveryTimeout",
    "load_key",
]

AUTHORITY_ENV = FAULT_TOLERANCE_AUTHORITY_ENV
AUTHORITY_KEY_ENV = "TLLM_FAULT_TOLERANCE_AUTHORITY_KEY"

_MIN_KEY_BYTES = 16
_SEND_TIMEOUT_MS = 5000
# How often a wait with an interrupt checks it.
_INTERRUPT_POLL_S = 0.005

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

    The socket identity is derived from the rank, so the authority recognizes the
    worker after a reconnect. Not thread-safe: use each client from one thread at a
    time.

    Args:
        endpoint: Authority address, e.g. ``tcp://host:5555``.
        key: HMAC key shared with the authority.
        rank: Logical worker rank.
    """

    def __init__(self, endpoint: str, key: bytes, rank: int) -> None:
        self._key = key
        self._rank = rank
        self._endpoint = endpoint
        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.DEALER)
        self._socket.setsockopt(zmq.LINGER, 0)
        self._socket.setsockopt(zmq.SNDTIMEO, _SEND_TIMEOUT_MS)
        self._socket.setsockopt(zmq.MAXMSGSIZE, MAX_MESSAGE_BYTES)
        # Signed into every message, so a frame replayed on another connection fails authentication.
        self._route = f"rank-{rank}-{secrets.token_hex(8)}".encode()
        self._socket.setsockopt(zmq.ROUTING_ID, self._route)
        self._socket.connect(endpoint)

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
        """Send ``message`` to the authority.

        Raises:
            RecoveryError: The message could not be queued in time.
        """
        try:
            self._socket.send_multipart(encode(message, self._key, self._route))
        except zmq.Again as error:
            raise RecoveryError(f"Timed out sending {message} to the authority") from error

    def _recv(self, timeout: float) -> Message | None:
        """Return the next authority message, or None if none arrives within ``timeout`` s.

        Raises:
            RecoveryError: A frame failed authentication or did not decode.
        """
        if not self._socket.poll(max(1, int(timeout * 1000))):
            return None
        frames = self._socket.recv_multipart()
        if not authenticate(frames, self._key, self._route):
            raise RecoveryError("Received an unauthenticated frame from the authority")
        try:
            return decode(frames[0])
        except ProtocolError as error:
            raise RecoveryError(f"Received a malformed message: {error}") from error

    def receive(self) -> Message | None:
        """Return an authority message already received, or None (waits at most 1 ms).

        Raises:
            RecoveryError: A frame failed authentication or did not decode.
        """
        return self._recv(0)

    def wait(
        self,
        message_type: type[_M],
        deadline: float,
        interrupt: threading.Event | None = None,
        skip: Callable[[Message], bool] | None = None,
    ) -> _M:
        """Block until a ``message_type`` message arrives.

        Args:
            message_type: Expected message class.
            deadline: Absolute ``time.monotonic()`` value.
            interrupt: Checked every ``_INTERRUPT_POLL_S`` while waiting.
            skip: Messages it accepts are dropped (left over from an earlier round).

        Raises:
            RecoveryAborted: The authority sent ``Abort``.
            RecoveryTimeout: ``deadline`` passed first.
            RecoveryInterrupted: ``interrupt`` was set and no message was waiting.
            RecoveryError: Any other unexpected message arrived.
        """
        while (remaining := deadline - time.monotonic()) > 0:
            interrupted = interrupt is not None and interrupt.is_set()
            if interrupt is not None:
                remaining = min(remaining, _INTERRUPT_POLL_S)
            # Messages that arrived before the interrupt still count (e.g. an Install the other ranks act on).
            message = self._recv(0 if interrupted else remaining)
            if message is None and interrupted:
                raise RecoveryInterrupted(
                    f"Rank {self._rank} stopped waiting for {message_type.__name__}"
                )
            if message is None or (skip is not None and skip(message)):
                continue
            if isinstance(message, Abort):
                raise RecoveryAborted(f"Recovery aborted by authority: {message.reason}")
            if not isinstance(message, message_type):
                raise RecoveryError(f"Expected {message_type.__name__}, received {message}")
            return message
        raise RecoveryTimeout(
            f"Rank {self._rank} timed out waiting for {message_type.__name__} from the recovery "
            f"authority at {self._endpoint}"
        )

    def close(self) -> None:
        """Close the connection."""
        self._socket.close()
        self._context.term()
