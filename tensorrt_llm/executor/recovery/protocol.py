# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Messages exchanged between the recovery authority and the workers.

Messages travel as two ZeroMQ frames: a JSON payload and its HMAC-SHA256 tag over the
worker's routing id and the payload, with a key shared by the authority and every
worker. A frame replayed on another connection fails authentication and is dropped;
the authority refuses a second connection with a routing id in use. Traffic is not
encrypted. Payloads are JSON rather than the pickles of ``ipc.ZeroMqQueue``, so neither side
unpickles network input, and the authority addresses each worker through a ROUTER.
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
    "MAX_MESSAGE_BYTES",
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

# Upper bound on a message on either side; real messages are a few hundred bytes.
MAX_MESSAGE_BYTES = 64 * 1024


@dataclass(frozen=True)
class Register:
    """Worker -> authority: announce a rank and the failure domains whose loss it can recover from.

    ``recoverable_domains`` are sorted rank sets (a single rank or a whole node) whose
    combined loss this worker's expert placement absorbs.
    """

    rank: int
    recoverable_domains: tuple[tuple[int, ...], ...]


@dataclass(frozen=True)
class Registered:
    """Authority -> workers: every rank registered; the loss of one admitted domain recovers."""

    admitted_domains: tuple[tuple[int, ...], ...]


@dataclass(frozen=True)
class Stopped:
    """Worker -> authority: local work joined after observing ``failed_ranks`` (sorted) depart."""

    rank: int
    failed_ranks: tuple[int, ...]


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
    """Authority -> survivors: prepare to continue without ``failed_ranks`` (sorted)."""

    generation: int
    failed_ranks: tuple[int, ...]
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


def _tag(key: bytes, route: bytes, payload: bytes) -> bytes:
    # ZeroMQ routing ids are at most 255 bytes, so one length byte keeps route and payload apart.
    return hmac.new(key, bytes([len(route)]) + route + payload, hashlib.sha256).digest()


def encode(message: Message, key: bytes, route: bytes) -> list[bytes]:
    """Serialize ``message`` into ``[payload, hmac_tag]`` frames for the connection ``route``."""
    body = dataclasses.asdict(message)
    body["type"] = type(message).__name__
    payload = json.dumps(body, sort_keys=True).encode()
    return [payload, _tag(key, route, payload)]


def authenticate(frames: list[bytes], key: bytes, route: bytes) -> bool:
    """Return whether ``frames`` is a ``[payload, hmac_tag]`` pair signed with ``key`` for ``route``.

    Binding the tag to the worker's routing id means a frame replayed on another connection fails.
    """
    if len(frames) != 2:
        return False
    return hmac.compare_digest(_tag(key, route, frames[0]), frames[1])


def _field_value(name: str, value_type: type, value):
    """Check a decoded JSON value of field ``name`` against ``value_type``; lists become tuples."""
    if typing.get_origin(value_type) is tuple:
        (item_type, _) = typing.get_args(value_type)
        if isinstance(value, list):
            return tuple(_field_value(name, item_type, item) for item in value)
    elif type(value) is value_type:
        return value
    raise ProtocolError(f"field {name} has invalid value {value!r}")


def decode(payload: bytes) -> Message:
    """Deserialize an authenticated payload.

    Raises:
        ProtocolError: The payload is not a well-formed message.
    """
    try:
        body = json.loads(payload)
        cls = _MESSAGE_TYPES[body.pop("type")]
    except (KeyError, TypeError, AttributeError, ValueError, RecursionError) as error:
        raise ProtocolError(f"malformed recovery message: {error!r}") from error
    fields = {field.name: field for field in dataclasses.fields(cls)}
    if body.keys() != fields.keys():
        raise ProtocolError(f"{cls.__name__} fields mismatch: {sorted(body)}")
    return cls(
        **{name: _field_value(name, fields[name].type, value) for name, value in body.items()}
    )
