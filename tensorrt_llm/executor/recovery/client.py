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
"""Worker- and frontend-side connection to the recovery authority."""

import os
import time
from typing import TypeVar

import zmq

from .protocol import Abort, Message, Register, authenticate, decode, encode

__all__ = [
    "RECOVERY_ENDPOINT_ENV",
    "RECOVERY_KEY_ENV",
    "RecoveryAborted",
    "RecoveryClient",
    "RecoveryError",
]

RECOVERY_ENDPOINT_ENV = "TLLM_WIDEEP_RECOVERY"
RECOVERY_KEY_ENV = "TLLM_WIDEEP_RECOVERY_KEY"

_SEND_TIMEOUT_MS = 5000

_M = TypeVar("_M")


class RecoveryError(RuntimeError):
    """The recovery exchange failed; the caller must take the terminal path."""


class RecoveryAborted(RecoveryError):
    """The authority aborted recovery."""

    def __init__(self, reason: str) -> None:
        super().__init__(f"Recovery aborted by authority: {reason}")
        self.reason = reason


class RecoveryClient:
    """DEALER connection to the recovery authority; registers on construction.

    Not thread-safe: use each client from one thread at a time.

    Args:
        endpoint: Authority address, e.g. ``tcp://host:5555``.
        key: HMAC key shared with the authority.
        role: ``ROLE_WORKER`` or ``ROLE_FRONTEND``.
        rank: Logical worker rank (0 for the frontend).
    """

    def __init__(self, endpoint: str, key: bytes, role: str, rank: int) -> None:
        self._key = key
        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.DEALER)
        self._socket.setsockopt(zmq.LINGER, 0)
        self._socket.setsockopt(zmq.SNDTIMEO, _SEND_TIMEOUT_MS)
        self._socket.connect(endpoint)
        self.send(Register(role, rank, os.getpid()))

    @classmethod
    def from_env(cls, role: str, rank: int) -> "RecoveryClient | None":
        """Connect using ``TLLM_WIDEEP_RECOVERY``/``_KEY``; None when recovery is disabled."""
        endpoint = os.environ.get(RECOVERY_ENDPOINT_ENV)
        if not endpoint:
            return None
        key = os.environ.get(RECOVERY_KEY_ENV)
        if not key:
            raise ValueError(f"{RECOVERY_ENDPOINT_ENV} is set but {RECOVERY_KEY_ENV} is not")
        return cls(endpoint, bytes.fromhex(key), role, rank)

    def send(self, message: Message) -> None:
        try:
            self._socket.send_multipart(encode(message, self._key))
        except zmq.Again as error:
            raise RecoveryError(f"Timed out sending {message} to the authority") from error

    def recv(self, timeout: float) -> Message | None:
        """Return the next authority message, or None if none arrives within ``timeout`` s."""
        if not self._socket.poll(max(0, int(timeout * 1000))):
            return None
        frames = self._socket.recv_multipart()
        if not authenticate(frames, self._key):
            raise RecoveryError("Received an unauthenticated frame from the authority")
        return decode(frames[0])

    def wait(self, message_type: type[_M], deadline: float) -> _M:
        """Block until a ``message_type`` message arrives.

        Args:
            message_type: Expected message class.
            deadline: Absolute ``time.monotonic()`` value.

        Raises:
            RecoveryAborted: The authority sent ``Abort``.
            RecoveryError: Any other unexpected message arrived.
            TimeoutError: ``deadline`` passed first.
        """
        while (remaining := deadline - time.monotonic()) > 0:
            message = self.recv(remaining)
            if message is None:
                continue
            if isinstance(message, Abort):
                raise RecoveryAborted(message.reason)
            if not isinstance(message, message_type):
                raise RecoveryError(f"Expected {message_type.__name__}, received {message}")
            return message
        raise TimeoutError(f"Timed out waiting for {message_type.__name__} from the authority")

    def close(self) -> None:
        self._socket.close()
        self._context.term()
