# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Recovery authority: coordinates the survivors of a lost failure domain (a rank or a node).

Run as ``python -m tensorrt_llm.executor.recovery.authority --world-size N`` with
``TLLM_FAULT_TOLERANCE_AUTHORITY_KEY`` set to the hex HMAC key shared with the
workers. Run it outside the workers' launch and termination groups: it must outlive
any one worker.

Experimental: traffic is authenticated but not encrypted, and a host that can reach
the port can still disrupt recovery (for example, by taking over a worker's route; see
:func:`router_queue`), so the authority binds to localhost by default.
A multi-node job binds it to the job's private network interface with
``--endpoint`` and a key generated for that job.
"""

import argparse
import ipaddress
import os
import re
import shlex
import signal
import socket
import subprocess
import threading
import time
from collections.abc import Callable

import zmq

from ...logger import logger
from ..ipc import ZeroMqQueue
from .authority import _REJECTION_LOG_INTERVAL_S, Outbound, Phase, RecoveryAuthority
from .client import CONNECTION_OPTIONS, load_key, unauthenticated
from .config import fault_tolerance_config
from .hang_monitor import HangMonitor, HungRank
from .protocol import (
    MAX_MESSAGE_BYTES,
    Sent,
    StatusQuery,
    datagram_address,
    decode_datagram,
    is_message,
)

__all__ = ["datagram_socket", "main", "router_queue", "serve"]

_CLOSE_LINGER_MS = 1000
# Bounds on what unauthenticated peers can queue: datagrams handled per pass, and messages queued per connection.
# Connections are not bounded and the loop takes one ROUTER message per pass, so N flooding connections still cut
# the workers' share of the loop to about 1/(N+1).
_MAX_DATAGRAMS_PER_PASS = 256
_ROUTER_RCVHWM = 1024
_HOSTNAME = re.compile(r"[A-Za-z0-9](?:[A-Za-z0-9.-]*[A-Za-z0-9])?")


def serve(
    authority: RecoveryAuthority,
    router: ZeroMqQueue,
    key: bytes,
    stop: threading.Event,
    poll_interval: float = 0.1,
    datagrams: socket.socket | None = None,
) -> None:
    """Drive ``authority`` from a ROUTER ``router`` until ``stop`` is set.

    Messages that fail HMAC authentication are dropped and authenticated objects that are
    not protocol messages are refused to their sender (see :meth:`RecoveryAuthority.reject`);
    neither changes the authority's state. Stopping during a recovery aborts it.

    Args:
        authority: State machine to drive.
        router: ROUTER queue from :func:`router_queue`.
        key: HMAC key shared with the workers.
        stop: Event that ends the loop.
        poll_interval: Seconds between deadline checks while no message arrives.
        datagrams: UDP socket from :func:`datagram_socket` receiving the workers' native
            heartbeats; datagrams that do not authenticate are dropped.
    """
    last_rejection = last_unsent = -_REJECTION_LOG_INTERVAL_S
    poller = zmq.Poller()
    poller.register(router.socket, zmq.POLLIN)
    if datagrams is not None:
        poller.register(datagrams.fileno(), zmq.POLLIN)
    while not stop.is_set():
        outbound = authority.expire(time.monotonic())
        ready = dict(poller.poll(int(poll_interval * 1000)))
        if datagrams is not None and datagrams.fileno() in ready:
            outbound += _receive_datagrams(authority, datagrams, key, time.monotonic())
        if router.socket in ready:
            rejection = None
            try:
                message = router.get()
            except (RuntimeError, ValueError) as error:
                if not unauthenticated(error):
                    raise
                rejection = f"dropped a message that failed authentication ({error}); is its key the authority's?"
            else:
                # ZeroMqQueue.get() keeps the sender's routing id for replies.
                route = router._last_identity
                if type(message) is Sent and is_message(message.message):
                    outbound += authority.receive(route, message, time.monotonic())
                elif type(message) is StatusQuery:
                    outbound += authority.handle(route, message, time.monotonic())
                else:
                    rejection = (
                        f"refused a {type(message).__name__} from {route!r}: not a recovery message"
                    )
                    outbound += authority.reject(rejection, route)
            # Rate-limited: a peer that floods bad messages must not flood the log.
            if rejection and time.monotonic() - last_rejection >= _REJECTION_LOG_INTERVAL_S:
                last_rejection = time.monotonic()
                logger.warning(f"Recovery authority: {rejection}")
        last_unsent = _send(router, outbound, last_unsent)
    # A recovery in progress must not leave the survivors waiting out their deadline.
    if authority.phase in (Phase.STOPPING, Phase.PREPARING, Phase.INSTALLING):
        _send(router, authority.abort("the recovery authority stopped"), last_unsent)


def _send(router: ZeroMqQueue, outbound: Outbound, last_log: float) -> float:
    """Send ``outbound`` without blocking; log (rate-limited from ``last_log``) what cannot be sent.

    A message cannot be sent to a route with no connection, or to a peer that does not read it.
    Returns the monotonic time of the last log.
    """
    for route, message in outbound:
        try:
            router.put_nowait(message, routing_id=route)
        except zmq.ZMQError as error:
            if time.monotonic() - last_log >= _REJECTION_LOG_INTERVAL_S:
                last_log = time.monotonic()
                logger.warning(
                    f"Recovery authority: could not send {message} to {route!r}: {error}"
                )
    return last_log


def _receive_datagrams(
    authority: RecoveryAuthority, datagrams: socket.socket, key: bytes, now: float
) -> Outbound:
    """Apply up to ``_MAX_DATAGRAMS_PER_PASS`` native heartbeats waiting on ``datagrams`` at time ``now``."""
    outbound = []
    for _ in range(_MAX_DATAGRAMS_PER_PASS):
        try:
            datagram = datagrams.recv(MAX_MESSAGE_BYTES)
        except BlockingIOError:
            break
        message = decode_datagram(datagram, key)
        if message is not None:
            outbound += authority.handle(b"", message, now)
    return outbound


def datagram_socket(endpoint: str) -> socket.socket | None:
    """Return a non-blocking UDP socket bound to the host and port of a ``tcp://`` ``endpoint``, else None.

    Hang detection needs it to tell a rank whose Python is starved from a stopped one, so
    a port it cannot bind is an error.
    """
    if not endpoint.startswith("tcp://"):
        return None
    host, port = datagram_address(endpoint)
    datagrams = socket.socket(socket.AF_INET6 if ":" in host else socket.AF_INET, socket.SOCK_DGRAM)
    try:
        datagrams.bind((host, port))
    except OSError as error:
        datagrams.close()
        raise RuntimeError(
            f"Recovery authority: cannot receive native heartbeats on UDP {host}:{port}"
        ) from error
    datagrams.setblocking(False)
    return datagrams


def router_queue(endpoint: str, key: bytes) -> ZeroMqQueue:
    """Return a ROUTER queue bound at ``endpoint``.

    A connection with a routing id in use takes the route over (a worker reconnecting after its
    connection died); the session and sequence checks of :meth:`RecoveryAuthority.receive` still
    drop a replay on it. Those checks cover only inbound messages: routing ids travel in cleartext,
    so a peer without the key that connects with a worker's id receives the authority's messages to
    that worker, and the recovery times out. A message to a route with no connection fails rather
    than vanishing.
    """
    router = ZeroMqQueue(
        (endpoint, key),
        socket_type=zmq.ROUTER,
        is_server=True,
        name="recovery-authority",
        # The linger lets a final Abort reach the workers without letting close() hang on one that is gone.
        socket_options={
            zmq.LINGER: _CLOSE_LINGER_MS,
            zmq.MAXMSGSIZE: MAX_MESSAGE_BYTES,
            zmq.RCVHWM: _ROUTER_RCVHWM,
            zmq.ROUTER_HANDOVER: 1,
            zmq.ROUTER_MANDATORY: 1,
            **CONNECTION_OPTIONS,
        },
    )
    if "://*:" in endpoint or "0.0.0.0" in endpoint:
        logger.warning(
            f"Recovery authority listens on every interface ({endpoint}): any host that reaches this one can "
            "connect; bind the job's private interface and use a per-job key"
        )
    return router


def _plain_host(host: str) -> bool:
    try:
        ipaddress.ip_address(host)
    except ValueError:
        return _HOSTNAME.fullmatch(host) is not None
    return True


def command_fence(template: str) -> Callable[[HungRank], None]:
    """Fence a hung worker by running ``template`` with its ``{host}`` and ``{pid}`` filled in.

    The command runs without a shell, detached from the authority's loop; a failure is
    logged, and the rank stays hung until the executor's hang detector ends the job. The
    worker reports its host and pid, so the command runs only for a plain hostname or IP
    address and a positive pid: nothing the command could take as an option.
    """
    arguments = shlex.split(template)

    def fence(hung: HungRank) -> None:
        if not (_plain_host(hung.host) and type(hung.pid) is int and hung.pid > 0):
            logger.error(
                f"Recovery authority: not fencing rank {hung.rank} with the fence command: "
                f"host {hung.host!r} or pid {hung.pid!r} is not plain"
            )
            return
        command = [arg.format(host=hung.host, pid=hung.pid) for arg in arguments]
        try:
            subprocess.Popen(command, stdin=subprocess.DEVNULL, start_new_session=True)
        except OSError as error:
            logger.error(
                f"Recovery authority: fencing rank {hung.rank} with {command} failed: {error}"
            )

    return fence


def main(argv: list[str] | None = None) -> None:
    """Run the recovery authority until SIGINT or SIGTERM."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--endpoint",
        default="tcp://127.0.0.1:5555",
        help="ROUTER bind address; bind a private interface, never a public one",
    )
    parser.add_argument("--world-size", type=int, required=True)
    parser.add_argument(
        "--fence-command",
        default="",
        help="last-resort command that kills a hung worker the workers could not fence (its own "
        "evidence thread, then the MPI runtime), with {host} and {pid} placeholders, run without a "
        "shell, e.g. 'ssh {host} kill -KILL {pid}' or the deployment's node agent",
    )
    args = parser.parse_args(argv)
    if "TLLM_LOG_LEVEL" not in os.environ:
        logger.set_level("info")

    key = load_key()
    # Timeouts and hang detection come from the workers' configuration file (TLLM_FAULT_TOLERANCE_CONFIG).
    config = fault_tolerance_config()
    hang_monitor = None
    if config.hang_detection:
        hang_monitor = HangMonitor(
            tuple(range(args.world_size)), config.hang_confirm_s, config.heartbeat_timeout_s
        )
    fence = command_fence(args.fence_command) if args.fence_command else None
    logger.info(f"Recovery authority: fault-tolerance settings {config.digest()}: {config}")
    authority = RecoveryAuthority(
        args.world_size, config.authority_timeout_s, hang_monitor, fence, config.digest()
    )
    stop = threading.Event()
    for signum in (signal.SIGINT, signal.SIGTERM):
        signal.signal(signum, lambda *_: stop.set())
    datagrams = datagram_socket(args.endpoint) if hang_monitor is not None else None
    router = router_queue(args.endpoint, key)
    try:
        serve(authority, router, key, stop, datagrams=datagrams)
    finally:
        router.close()
        if datagrams is not None:
            datagrams.close()


if __name__ == "__main__":
    main()
