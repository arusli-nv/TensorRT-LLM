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
"""Messages exchanged between the recovery authority and the workers.

Messages travel as two ZeroMQ frames: a JSON payload and its HMAC-SHA256 tag. The
key, shared by the authority and every worker, authenticates messages, not routing
ids: on a trusted network a peer without the key can at most take over a route,
which fails closed (timeout, then Abort). Messages carry no replay protection.
"""

import dataclasses
import hashlib
import hmac
import json
import typing
from dataclasses import dataclass

__all__ = [
    "Abort",
    "Commit",
    "Install",
    "Installed",
    "Message",
    "Propose",
    "ProtocolError",
    "Ready",
    "Register",
    "Registered",
    "Stopped",
    "authenticate",
    "decode",
    "encode",
]


@dataclass(frozen=True)
class Register:
    """Worker -> authority: announce a rank and the ranks whose loss it can recover from.

    ``recoverable_ranks`` are the ranks whose experts all have a resident replica
    elsewhere in this worker's placement.
    """

    rank: int
    recoverable_ranks: tuple[int, ...]


@dataclass(frozen=True)
class Registered:
    """Authority -> workers: every rank registered; losses of ``admitted_ranks`` recover."""

    admitted_ranks: tuple[int, ...]


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
    """Authority -> workers: recovery failed; take the terminal path."""

    reason: str


Message = Register | Registered | Stopped | Ready | Installed | Propose | Install | Commit | Abort

_MESSAGE_TYPES: dict[str, type] = {cls.__name__: cls for cls in typing.get_args(Message)}


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


def _field_value(field: dataclasses.Field, value):
    """Check a decoded JSON value against the field type; sequences become tuples."""
    if typing.get_origin(field.type) is tuple:
        (item_type, _) = typing.get_args(field.type)
        if isinstance(value, list) and all(type(item) is item_type for item in value):
            return tuple(value)
    elif type(value) is field.type:
        return value
    raise ProtocolError(f"field {field.name} has invalid value {value!r}")


def decode(payload: bytes) -> Message:
    """Deserialize an authenticated payload.

    Raises:
        ProtocolError: The payload is not a well-formed message.
    """
    try:
        body = json.loads(payload)
        cls = _MESSAGE_TYPES[body.pop("type")]
    except (KeyError, TypeError, AttributeError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ProtocolError(f"malformed recovery message: {error!r}") from error
    fields = {field.name: field for field in dataclasses.fields(cls)}
    if body.keys() != fields.keys():
        raise ProtocolError(f"{cls.__name__} fields mismatch: {sorted(body)}")
    return cls(**{name: _field_value(fields[name], value) for name, value in body.items()})
