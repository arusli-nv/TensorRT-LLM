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
"""Worker-side connection to the recovery authority."""

import functools
import os
import time
from typing import TypeVar

import zmq

from .protocol import (
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
    "FAULT_TOLERANCE_MODE_ENV",
    "RecoveryAborted",
    "RecoveryClient",
    "RecoveryError",
    "RecoveryTimeout",
    "load_key",
    "recovery_requested",
]

FAULT_TOLERANCE_MODE_ENV = "TLLM_FAULT_TOLERANCE_MODE"
AUTHORITY_ENV = "TLLM_FAULT_TOLERANCE_AUTHORITY"
AUTHORITY_KEY_ENV = "TLLM_FAULT_TOLERANCE_AUTHORITY_KEY"

_MIN_KEY_BYTES = 16
_SEND_TIMEOUT_MS = 5000

_M = TypeVar("_M")


@functools.cache
def recovery_requested() -> bool:
    """Whether the environment enables recovery from worker loss; read once per process.

    ``TLLM_FAULT_TOLERANCE_MODE=1`` alone enables detection only; adding the
    authority address (``TLLM_FAULT_TOLERANCE_AUTHORITY``) and its HMAC key
    (``TLLM_FAULT_TOLERANCE_AUTHORITY_KEY``) enables recovery.

    Raises:
        ValueError: The authority address is set without fault-tolerance mode.
    """
    if not os.environ.get(AUTHORITY_ENV):
        return False
    if os.environ.get(FAULT_TOLERANCE_MODE_ENV) != "1":
        raise ValueError(f"{AUTHORITY_ENV} requires {FAULT_TOLERANCE_MODE_ENV}=1")
    return True


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
        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.DEALER)
        self._socket.setsockopt(zmq.LINGER, 0)
        self._socket.setsockopt(zmq.SNDTIMEO, _SEND_TIMEOUT_MS)
        self._socket.setsockopt(zmq.ROUTING_ID, f"rank-{rank}".encode())
        self._socket.connect(endpoint)

    @classmethod
    def from_env(cls, rank: int) -> "RecoveryClient | None":
        """Connect to the authority named by the environment; None when recovery is disabled."""
        if not recovery_requested():
            return None
        return cls(os.environ[AUTHORITY_ENV], load_key(), rank)

    def register(self, recoverable_ranks: tuple[int, ...], deadline: float) -> frozenset[int]:
        """Register this rank and wait until every rank has registered.

        Args:
            recoverable_ranks: Ranks whose loss this worker's expert placement covers.
            deadline: Absolute ``time.monotonic()`` value.

        Returns:
            The ranks whose loss the deployment recovers from.

        Raises:
            RecoveryError: Registration was rejected or timed out.
        """
        self.send(Register(self._rank, tuple(recoverable_ranks)))
        return frozenset(self.wait(Registered, deadline).admitted_ranks)

    def send(self, message: Message) -> None:
        """Send ``message`` to the authority.

        Raises:
            RecoveryError: The message could not be queued in time.
        """
        try:
            self._socket.send_multipart(encode(message, self._key))
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
        if not authenticate(frames, self._key):
            raise RecoveryError("Received an unauthenticated frame from the authority")
        try:
            return decode(frames[0])
        except ProtocolError as error:
            raise RecoveryError(f"Received a malformed message: {error}") from error

    def wait(self, message_type: type[_M], deadline: float) -> _M:
        """Block until a ``message_type`` message arrives.

        Args:
            message_type: Expected message class.
            deadline: Absolute ``time.monotonic()`` value.

        Raises:
            RecoveryAborted: The authority sent ``Abort``.
            RecoveryTimeout: ``deadline`` passed first.
            RecoveryError: Any other unexpected message arrived.
        """
        while (remaining := deadline - time.monotonic()) > 0:
            message = self._recv(remaining)
            if message is None:
                continue
            if isinstance(message, Abort):
                raise RecoveryAborted(f"Recovery aborted by authority: {message.reason}")
            if not isinstance(message, message_type):
                raise RecoveryError(f"Expected {message_type.__name__}, received {message}")
            return message
        raise RecoveryTimeout(f"Timed out waiting for {message_type.__name__} from the authority")

    def close(self) -> None:
        """Close the connection."""
        self._socket.close()
        self._context.term()
