# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Messages exchanged between the recovery authority and the workers.

Connections carry them as ``ipc.ZeroMqQueue`` objects: pickles signed with HMAC-SHA256 under a key shared
by the authority and every worker, verified before unpickling. A worker's native heartbeat travels as one UDP
datagram: a JSON payload, then its HMAC-SHA256 tag over a zero byte and the payload, so neither form
authenticates as the other. Traffic is not encrypted.
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
    "DeviceWait",
    "Fence",
    "Fenced",
    "Heartbeat",
    "Install",
    "Installed",
    "MAX_MESSAGE_BYTES",
    "MalformedMessage",
    "Unauthenticated",
    "Message",
    "NativeHeartbeat",
    "PROTOCOL_VERSION",
    "Propose",
    "Ready",
    "Register",
    "Registered",
    "Sent",
    "Status",
    "StatusQuery",
    "Stopped",
    "Terminate",
    "datagram_address",
    "decode_datagram",
    "is_message",
    "is_sent",
    "protocol_class",
]

# Upper bound on a message on either side; real messages are a few hundred bytes.
MAX_MESSAGE_BYTES = 64 * 1024
# Version of the messages below; raise it with any change to them that a peer cannot read.
PROTOCOL_VERSION = 1


@dataclass(frozen=True)
class Register:
    """Worker -> authority: announce a rank and the failure domains whose failure it can recover from.

    ``recoverable_domains`` are sorted rank sets (a single rank or a whole node) whose
    combined failure this worker's expert placement absorbs. ``config_digest`` is the worker's
    ``FaultToleranceConfig.digest()``, and ``protocol_version`` the ``PROTOCOL_VERSION`` it speaks.
    """

    rank: int
    recoverable_domains: tuple[tuple[int, ...], ...]
    config_digest: str
    protocol_version: int = PROTOCOL_VERSION


@dataclass(frozen=True)
class Registered:
    """Authority -> workers: every rank registered; the failure of one admitted domain recovers."""

    admitted_domains: tuple[tuple[int, ...], ...]


@dataclass(frozen=True)
class Stopped:
    """Worker -> authority: local work joined after observing ``failed_ranks`` (sorted) fail."""

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
    """Worker -> authority: the live ranks of ``generation`` are installed.

    ``failed_requests`` is how many requests the recovery fails: the failed ranks' requests that do
    not run again, and the requests a rank failure interrupted a second time without progress. Only rank 0,
    which routes every request, reports them (0 elsewhere).
    """

    rank: int
    generation: int
    failed_requests: int = 0


@dataclass(frozen=True)
class Propose:
    """Authority -> survivors: prepare to continue on ``live_ranks``, without ``failed_ranks`` (sorted)."""

    generation: int
    failed_ranks: tuple[int, ...]
    live_ranks: tuple[int, ...]


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


@dataclass(frozen=True)
class Heartbeat:
    """Worker -> authority, periodically: the process lives and its executor is at ``iteration``.

    ``waiting`` says whether the executor waits on something outside its rank: a host
    collective, or new requests. ``host`` and ``pid`` identify the worker process, so the
    authority can fence it.
    """

    rank: int
    iteration: int
    waiting: bool
    host: str
    pid: int


@dataclass(frozen=True)
class NativeHeartbeat:
    """Worker's native thread -> authority, as a UDP datagram: process ``pid`` of ``rank`` runs.

    Sent without the Python interpreter lock, so it keeps arriving while the worker's Python
    threads are starved. ``beat`` counts the process's datagrams; ``iteration`` is the
    executor iteration of its last finished step.
    """

    rank: int
    pid: int
    beat: int
    iteration: int


@dataclass(frozen=True)
class DeviceWait:
    """Worker -> authority: this rank's device has waited ``waited_ms`` on ``peer``; 0 clears."""

    rank: int
    peer: int
    waited_ms: int


@dataclass(frozen=True)
class Fence:
    """Authority -> a worker's reporter connection: end the worker process ``pid`` of the hung ``rank``.

    Sent to the hung rank itself (its reporter thread kills its own process) and to another
    rank (which has the MPI runtime kill ``rank``'s process after a grace period). During a
    recovery, sent to rank 0's recovery connection for a failed rank not known to be ended:
    rank 0 has the MPI runtime kill it at once and answers ``Fenced``; ``pid`` is 0 if unknown.
    """

    rank: int
    pid: int


@dataclass(frozen=True)
class Fenced:
    """Rank 0 -> authority: the MPI runtime ended failed rank ``target``, or ``error`` says why not."""

    rank: int
    target: int
    error: str


@dataclass(frozen=True)
class Terminate:
    """Authority -> each worker's reporter connection: the deployment hangs and no fence can recover it.

    The worker ends its process with ``STALLED_EXIT_CODE`` so that the deployment is restarted.
    """

    reason: str


@dataclass(frozen=True)
class StatusQuery:
    """Observer -> authority: report the deployment's recovery state; changes nothing."""


@dataclass(frozen=True)
class Status:
    """Authority -> observer: the committed recovery state.

    ``phase`` is the authority's phase name; ``live_ranks`` exclude the
    ``failed_ranks`` (sorted) once they are known. ``recovery_stage_us`` holds the last
    committed recovery's durations in microseconds: stop (first ``Stopped`` to ``Propose``),
    prepare (to ``Install``) and install (to ``Commit``); empty before any recovery.
    ``failed_requests`` counts the requests the recoveries failed (see ``Installed``).
    ``abort_reason`` is empty unless aborted.
    """

    phase: str
    generation: int
    live_ranks: tuple[int, ...]
    failed_ranks: tuple[int, ...]
    recoveries: int
    recovery_stage_us: tuple[int, ...]
    failed_requests: int
    abort_reason: str


Message = (
    Register
    | Registered
    | Stopped
    | Ready
    | Installed
    | Propose
    | Install
    | Commit
    | Abort
    | Heartbeat
    | NativeHeartbeat
    | DeviceWait
    | Fence
    | Fenced
    | Terminate
    | StatusQuery
    | Status
)


@dataclass(frozen=True)
class Sent:
    """Worker -> authority: ``message`` as message ``seq`` of ``rank``'s process ``session``.

    ``session`` is random and fixed for the life of the process, announced with its ``Register``;
    ``seq`` increases strictly over the process's messages. The authority drops a message whose
    session is not its rank's or whose ``seq`` is not above the last it accepted from that rank's
    connection of the same kind (protocol or hang reports), so a replayed message changes nothing,
    whatever connection carries it.
    """

    rank: int
    session: str
    seq: int
    message: Message


_MESSAGE_TYPES = typing.get_args(Message)
_CLASSES = {cls.__name__: cls for cls in (*_MESSAGE_TYPES, Sent)}
_TAG_BYTES = hashlib.sha256().digest_size


class MalformedMessage(Exception):
    """An authenticated message that does not unpickle here or is not a protocol message."""


class Unauthenticated(Exception):
    """A message that does not authenticate with the shared key, or a frame that is not one message."""


def protocol_class(name: str) -> type:
    """The protocol message type, or ``Sent``, named ``name``: the only classes a received message may hold.

    Raises:
        MalformedMessage: ``name`` is no such class.
    """
    cls = _CLASSES.get(name)
    if cls is None:
        raise MalformedMessage(f"a message that holds {name!r}, not a recovery message")
    return cls


def is_message(obj: object) -> bool:
    """Whether ``obj``, received from a connection, is one of the protocol's messages with fields of their types."""
    return type(obj) in _MESSAGE_TYPES and _fields_typed(obj)


def is_sent(obj: object) -> bool:
    """Whether ``obj``, received from a connection, is a ``Sent`` with fields of their types."""
    return type(obj) is Sent and _fields_typed(obj)


def _fields_typed(obj: object) -> bool:
    fields = dataclasses.fields(obj)
    return all(_typed(getattr(obj, field.name, None), field.type) for field in fields)


def _typed(value: object, hint: object) -> bool:
    """Whether ``value`` is of the field type ``hint`` exactly (a ``bool`` is no ``int``), as a datagram's fields."""
    if hint is Message:
        return is_message(value)
    if typing.get_origin(hint) is tuple:
        # Every tuple field is a tuple[T, ...].
        item = typing.get_args(hint)[0]
        return type(value) is tuple and all(_typed(element, item) for element in value)
    return type(value) is hint


def _datagram_tag(key: bytes, payload: bytes) -> bytes:
    return hmac.new(key, b"\x00" + payload, hashlib.sha256).digest()


def decode_datagram(datagram: bytes, key: bytes) -> NativeHeartbeat | None:
    """The native heartbeat in ``datagram``, or None if it does not authenticate or is not one."""
    payload, tag = datagram[:-_TAG_BYTES], datagram[-_TAG_BYTES:]
    if not payload or not hmac.compare_digest(_datagram_tag(key, payload), tag):
        return None
    try:
        body = json.loads(payload)
    except (ValueError, RecursionError):
        return None
    fields = {field.name for field in dataclasses.fields(NativeHeartbeat)}
    if (
        not isinstance(body, dict)
        or body.pop("type", None) != NativeHeartbeat.__name__
        or body.keys() != fields
        or any(type(value) is not int for value in body.values())
    ):
        return None
    return NativeHeartbeat(**body)


def datagram_address(endpoint: str) -> tuple[str, int]:
    """The UDP host and port of the authority's ``tcp://host:port`` endpoint: the same port number."""
    host, _, port = endpoint.removeprefix("tcp://").rpartition(":")
    host = host.strip("[]")
    return ("0.0.0.0" if host == "*" else host), int(port)
