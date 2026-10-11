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
"""Messages exchanged between the recovery authority, workers, and the frontend.

Messages travel as two ZeroMQ frames: a JSON payload and its HMAC-SHA256 tag.
"""

import dataclasses
import hashlib
import hmac
import json
from dataclasses import dataclass

ROLE_WORKER = "worker"
ROLE_FRONTEND = "frontend"


@dataclass(frozen=True)
class Register:
    """Peer -> authority: announce a worker rank or the frontend."""

    role: str
    rank: int
    pid: int


@dataclass(frozen=True)
class Stopped:
    """Worker -> authority: local work joined after observing ``failed_rank`` depart."""

    rank: int
    failed_rank: int


@dataclass(frozen=True)
class Ready:
    """Worker -> authority: resident placement for ``generation`` is prepared."""

    rank: int
    generation: int
    placement_digest: str


@dataclass(frozen=True)
class Installed:
    """Worker -> authority: survivor membership for ``generation`` is installed."""

    rank: int
    generation: int


@dataclass(frozen=True)
class Propose:
    """Authority -> survivors: prepare to continue without ``failed_rank``."""

    generation: int
    failed_rank: int
    survivors: tuple[int, ...]


@dataclass(frozen=True)
class Install:
    """Authority -> survivors: every survivor is ready; install ``generation``."""

    generation: int


@dataclass(frozen=True)
class Commit:
    """Authority -> survivors: every survivor installed ``generation``; resume serving."""

    generation: int


@dataclass(frozen=True)
class Abort:
    """Authority -> all peers: recovery failed; take the terminal path."""

    reason: str


@dataclass(frozen=True)
class CloseAdmission:
    """Authority -> frontend: stop admitting requests."""


@dataclass(frozen=True)
class AdmissionClosed:
    """Frontend -> authority: admission is closed and no request is in flight."""


@dataclass(frozen=True)
class ReopenAdmission:
    """Authority -> frontend: recovery to ``generation`` committed; admit requests again."""

    generation: int


Message = (
    Register
    | Stopped
    | Ready
    | Installed
    | Propose
    | Install
    | Commit
    | Abort
    | CloseAdmission
    | AdmissionClosed
    | ReopenAdmission
)

_MESSAGE_TYPES: dict[str, type] = {
    cls.__name__: cls
    for cls in (
        Register,
        Stopped,
        Ready,
        Installed,
        Propose,
        Install,
        Commit,
        Abort,
        CloseAdmission,
        AdmissionClosed,
        ReopenAdmission,
    )
}


class ProtocolError(ValueError):
    """An authenticated frame did not decode into a well-formed message."""


def encode(message: Message, key: bytes) -> list[bytes]:
    """Serialize ``message`` into ``[payload, hmac_tag]`` frames."""
    body = dataclasses.asdict(message)
    body["type"] = type(message).__name__
    payload = json.dumps(body, sort_keys=True).encode()
    return [payload, hmac.new(key, payload, hashlib.sha256).digest()]


def authenticate(frames: list[bytes], key: bytes) -> bool:
    """Return whether ``frames`` is a ``[payload, hmac_tag]`` pair signed with ``key``."""
    if len(frames) != 2:
        return False
    expected = hmac.new(key, frames[0], hashlib.sha256).digest()
    return hmac.compare_digest(expected, frames[1])


def decode(payload: bytes) -> Message:
    """Deserialize an authenticated payload; raises ``ProtocolError`` when malformed."""
    try:
        body = json.loads(payload)
        cls = _MESSAGE_TYPES[body.pop("type")]
        fields = {field.name: field.type for field in dataclasses.fields(cls)}
        if body.keys() != fields.keys():
            raise ProtocolError(f"{cls.__name__} fields mismatch: {sorted(body)}")
        if "survivors" in body:
            body["survivors"] = tuple(body["survivors"])
        for name, value in body.items():
            expected = str if fields[name] is str else int
            items = value if name == "survivors" else (value,)
            if not all(type(item) is expected for item in items):
                raise ProtocolError(f"{cls.__name__}.{name} has invalid value {value!r}")
        return cls(**body)
    except (KeyError, TypeError, AttributeError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ProtocolError(f"malformed recovery message: {error!r}") from error
