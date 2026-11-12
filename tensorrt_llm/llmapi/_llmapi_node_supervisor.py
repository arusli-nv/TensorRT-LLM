# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Node supervisor of a Wide-EP fault-tolerant deployment, which starts and replaces the ranks of one node."""

import argparse
import contextlib
import ctypes
import functools
import glob
import json
import os
import selectors
import shutil
import signal
import socket
import subprocess  # nosec B404
import sys
import time
from dataclasses import dataclass
from types import FrameType
from typing import Callable, Dict, Hashable, List, Optional, Set, Tuple

# Newest protocol first.
_PMIX_SERVER_URI_VARS = (
    "PMIX_SERVER_URI41",
    "PMIX_SERVER_URI4",
    "PMIX_SERVER_URI3",
    "PMIX_SERVER_URI21",
    "PMIX_SERVER_URI2",
)
# The variables that identify this process to its PMIx server, which prun and pterm must not take as their own.
_PROCESS_IDENTITY_VARS = (
    *_PMIX_SERVER_URI_VARS,
    "PMIX_NAMESPACE",
    "PMIX_RANK",
    "PMIX_ID",
    "PMIX_HOSTNAME",
    "PMIX_VERSION",
    "PMIX_SECURITY_MODE",
    "PMIX_PTL_MODULE",
    "PMIX_GDS_MODULE",
    "PMIX_BFROP_BUFFER_TYPE",
    "PMIX_SERVER_TMPDIR",
    "PMIX_SYSTEM_TMPDIR",
    "OMPI_UNIVERSE_SIZE",
    "OMPI_FIRST_RANKS",
    "OMPI_APP_CTX_NUM_PROCS",
    "OMPI_NUM_APP_CTX",
    "OMPI_FILE_LOCATION",
)
_PROCESS_IDENTITY_PREFIXES = ("OMPI_COMM_WORLD_", "PMIX_DSTORE_")
# The deployment's opt-in to running as root, without which prun and pterm refuse to run as root.
_RUN_AS_ROOT_VARS = (
    "OMPI_ALLOW_RUN_AS_ROOT",
    "OMPI_ALLOW_RUN_AS_ROOT_CONFIRM",
    "PRTE_ALLOW_RUN_AS_ROOT",
    "PRTE_ALLOW_RUN_AS_ROOT_CONFIRM",
)
_PR_SET_PDEATHSIG = 1
_SHM_DIR = "/dev/shm"
_CONNECT_TIMEOUT_S = 2.0
# Twice this stays below the grace period the launcher gives the supervisor (30 s by default on Kubernetes).
_STOP_GRACE_S = 10.0
_MIN_BACKOFF_S = 1.0
_MAX_BACKOFF_S = 60.0
# The replacements of a rank that fail in a row for the same reason before no other one starts.
_MAX_FAILED_REPLACEMENTS = 3
# The relaunched supervisors of a node that end in a row before connecting, after which node 0 relaunches it no more.
_MAX_FAILED_RELAUNCHES = 3
# A replacement that ends after serving this long is a new failure of its rank, not a failed replacement.
_SERVED_S = 600.0


def _exit_status(returncode: int) -> int:
    """Return the exit status as a shell reports it."""
    return 128 - returncode if returncode < 0 else returncode


@dataclass
class _Process:
    child: subprocess.Popen
    replacement: bool
    pidfd: Optional[int]
    incarnation: Optional[int] = None
    listed: bool = False
    # When a COMMITTED record first listed it as an active member.
    admitted_s: Optional[float] = None
    # The reason the membership left it out while it joined.
    left_out_reason: str = ""
    killed: bool = False
    status: Optional[int] = None
    started_s: float = 0.0


class _Backoff:
    """Track per-key retry times with a delay that doubles until :meth:`reset`."""

    def __init__(self):
        self._due_s: Dict[Hashable, float] = {}
        self._delay_s: Dict[Hashable, float] = {}

    def due_s(self, key: Hashable) -> float:
        return self._due_s.get(key, 0.0)

    def delay(self, key: Hashable, now_s: float) -> None:
        delay_s = self._delay_s.get(key, _MIN_BACKOFF_S)
        self._due_s[key] = now_s + delay_s
        self._delay_s[key] = min(2 * delay_s, _MAX_BACKOFF_S)

    def reset(self, key: Hashable) -> None:
        self._delay_s.pop(key, None)


# Selector data of the descriptors other than the rank processes. A rank process has its rank as its data.
_WAKE = "wake"
_PARENT = "parent"
_AUTHORITY = "authority"
_RELAUNCHED = "relaunched"


class NodeSupervisor:
    """Start, supervise and replace the ranks of one node.

    Args:
        command: The command that starts the process of each rank. With ``rank_0_in_session``, it starts every
            rank but 0.
        relaunch_command: The command that starts a process on the host appended to it. Set on node 0 under
            ``mpirun``.
        end_job: The function that ends the whole job as failed. Set on node 0 under ``mpirun``.
    """

    def __init__(
        self,
        node: int,
        nodes: int,
        gpus_per_node: int,
        command: List[str],
        environ: Dict[str, str],
        rank_0_in_session: bool = False,
        relaunch_command: Optional[List[str]] = None,
        end_job: Optional[Callable[[Dict[str, str], float], None]] = None,
    ):
        from tensorrt_llm._torch.distributed import membership
        from tensorrt_llm._wide_ep_fault_tolerance import FAULT_TOLERANCE_RANK_ENV
        from tensorrt_llm.logger import logger
        from tensorrt_llm.mapping import Mapping

        self._membership = membership
        self._logger = logger
        self._rank_env = FAULT_TOLERANCE_RANK_ENV
        self._node = node
        self._nodes = nodes
        self._gpus_per_node = gpus_per_node
        self._world_size = nodes * gpus_per_node
        self._mapping = Mapping(
            world_size=self._world_size, tp_size=self._world_size, gpus_per_node=gpus_per_node
        )
        self._ranks = self._node_ranks(node)
        self._command = command
        self._rank_0_in_session = rank_0_in_session
        self._environ = {
            name: value
            for name, value in environ.items()
            if name in _RUN_AS_ROOT_VARS or not name.startswith(membership.LAUNCHER_ENV_PREFIXES)
        }
        # Each rank started here is a one-process MPI world with no shared-memory peer. Without the sm transport, it
        # creates no /dev/shm segment that an os._exit or a kill would leave behind.
        self._environ["OMPI_MCA_btl"] = environ.get("OMPI_MCA_btl", "^sm")
        self._tool_environ = _prrte_tool_environ(environ)
        self._relaunch_command = relaunch_command
        self._end_job_through_launcher = end_job
        self._parent = os.getppid()
        self._parent_ended = False
        self._selector = selectors.DefaultSelector()
        # Written by the signal handler, so that a signal ends the selector's wait.
        self._wake_read, self._wake_write = socket.socketpair()
        self._wake_write.setblocking(False)
        self._store = None
        self._connection = None
        self._view = None
        self._view_received_s = 0.0
        self._served = False
        self._refused = False
        self._connect_due_s = 0.0
        self._wake_at_s: List[float] = []
        self._processes: Dict[int, _Process] = {}
        self._replacements = _Backoff()
        self._given_up: Set[int] = set()
        # Per rank, the reason the last replacement of it failed, and how many failed in a row for that reason.
        self._failed_replacements: Dict[int, Tuple[str, int]] = {}
        self._relaunched: Dict[int, subprocess.Popen] = {}
        self._relaunches = _Backoff()
        # Per node, the supervisors relaunched since its supervisor last connected.
        self._unconnected_relaunches: Dict[int, int] = {}
        self._host_weights_prefix: Optional[str] = None
        self._first_failure: Optional[int] = None
        self._signum: Optional[int] = None
        prctl = ctypes.CDLL(None, use_errno=True).prctl
        prctl.argtypes = [ctypes.c_int] + [ctypes.c_ulong] * 4
        supervisor = os.getpid()

        def die_with_supervisor() -> None:
            if prctl(_PR_SET_PDEATHSIG, int(signal.SIGKILL), 0, 0, 0) != 0:
                raise OSError(ctypes.get_errno(), "prctl(PR_SET_PDEATHSIG) failed")
            # The supervisor ended before the prctl took effect.
            if os.getppid() != supervisor:
                os._exit(1)

        self._die_with_supervisor = die_with_supervisor

    def on_signal(self, signum: int, frame: Optional[FrameType] = None) -> None:
        """Stop the supervisor and forward the signal to rank 0's process."""
        if self._signum is None:
            self._signum = signum
        with contextlib.suppress(OSError):
            self._wake_write.send(b"\0")

    def run(self) -> int:
        # The bound socket keeps a second supervisor of the deployment off the node.
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as node_lock:
            try:
                node_lock.bind(self._membership.node_supervisor_address())
            except OSError:
                self._logger.error("Another node supervisor of this deployment runs on this node")
                return 1
            try:
                self._watch_parent()
                self._selector.register(self._wake_read, selectors.EVENT_READ, _WAKE)
                self._connect()
                if self._refused:
                    return 1
                if self._view is not None and self._view.record.state in (
                    self._membership.MembershipState.SHUTDOWN,
                    self._membership.MembershipState.ABORTED,
                ):
                    self._logger.error(
                        "The deployment at the rendezvous address ended. Use a new shared secret."
                    )
                    return 1
                if not self._served:
                    for rank in self._ranks:
                        if rank != 0 or not self._rank_0_in_session:
                            self._start(rank, replacement=False)
                while True:
                    status = self._step(time.monotonic())
                    if status is not None:
                        return status
                    self._wait()
            finally:
                for key in list(self._selector.get_map().values()):
                    if isinstance(key.fileobj, int):
                        os.close(key.fileobj)
                    else:
                        key.fileobj.close()
                self._selector.close()
                self._wake_write.close()

    def _node_ranks(self, node: int) -> List[int]:
        return [
            rank for rank in range(self._world_size) if self._mapping.get_node_rank(rank) == node
        ]

    def _watch_parent(self) -> None:
        try:
            pidfd = os.pidfd_open(self._parent)
        except ProcessLookupError:
            self._parent_ended = True
            return
        self._selector.register(pidfd, selectors.EVENT_READ, _PARENT)
        if os.getppid() != self._parent:
            self._parent_ended = True

    def _wait(self) -> None:
        """Wait for the next event or the next due step, and take the events."""
        timeout_s = None
        if self._wake_at_s:
            timeout_s = max(0.0, min(self._wake_at_s) - time.monotonic())
        for key, _ in self._selector.select(timeout_s):
            if key.data == _WAKE:
                self._wake_read.recv(1 << 12)
            elif key.data in (_PARENT, _RELAUNCHED):
                self._parent_ended |= key.data == _PARENT
                self._selector.unregister(key.fileobj)
                os.close(key.fileobj)
            elif key.data == _AUTHORITY:
                self._receive_view()
            else:
                self._reap(key.data, time.monotonic())

    def _step(self, now_s: float) -> Optional[int]:
        """Act on what changed, and return the exit status once the supervisor ends."""
        self._wake_at_s = []
        if self._signum is not None:
            self._logger.warning(f"Stopping the node's ranks on signal {self._signum}")
            return self._stop(128 + self._signum, terminate=False, forward=self._signum)
        if self._parent_ended:
            self._logger.warning(
                "Stopping the node's ranks because the process that started this supervisor ended"
            )
            return self._stop(128 + signal.SIGTERM)
        if self._refused:
            return self._stop(1)
        if self._connection is None:
            if now_s >= self._connect_due_s:
                self._connect()
            self._wake_at_s.append(self._connect_due_s)
        record = None if self._view is None else self._view.record
        state = self._membership.MembershipState
        if record is not None and record.state == state.SHUTDOWN:
            self._logger.info("The deployment shut down")
            return self._stop(0, terminate=False)
        if record is not None and record.state == state.ABORTED:
            self._logger.error(f"Deployment aborted: {record.reason}")
            self._end_job(f"the deployment aborted: {record.reason}")
            return self._stop(1)
        for rank in self._ranks:
            status = self._supervise(rank, record, now_s)
            if status is not None:
                return self._stop(status)
        if self._relaunch_command is not None and self._connection is not None and self._served:
            self._relaunch_failed_nodes(now_s)
        if (
            self._processes
            and all(process.status is not None for process in self._processes.values())
            and self._connection is None
        ):
            if self._served:
                self._logger.error(
                    "Every rank of the node ended, and no membership authority follows the deployment"
                )
            return self._stop(self._first_failure or 0)
        return None

    def _supervise(self, rank: int, record, now_s: float) -> Optional[int]:
        """Act on ``rank``, and return the exit status if the node is to end."""
        process = self._processes.get(rank)
        ended = process is not None and process.status is not None
        if not self._served:
            if ended and process.status != 0:
                self._logger.error(
                    f"Rank {rank} failed before the deployment serves (status {process.status}), so the node fails"
                )
                if rank == 0:
                    self._end_job(
                        f"rank 0 failed before the deployment serves (status {process.status})"
                    )
                return _exit_status(process.status)
            return None
        if rank == 0:
            if ended:
                self._logger.error("Rank 0 ended, so the deployment ends with it")
                self._end_job(f"rank 0 ended (status {process.status})")
                return _exit_status(process.status)
            return None
        if self._connection is None:
            return None
        if process is not None and not ended:
            if record.is_active(rank):
                process.listed = True
                process.incarnation = record.incarnations[rank]
                if (
                    process.admitted_s is None
                    and record.state == self._membership.MembershipState.COMMITTED
                ):
                    process.admitted_s = now_s
            elif process.listed and not process.killed:
                self._logger.warning(
                    f"Killing rank {rank} (incarnation {process.incarnation}), which the membership record "
                    "no longer lists"
                )
                process.killed = True
                process.child.kill()
            elif (
                rank in self._view.stalled_replacements
                and self._view_received_s > process.started_s
                and not process.killed
            ):
                self._logger.warning(
                    f"Killing the replacement of rank {rank}, whose startup made no progress for "
                    f"{self._membership.RENDEZVOUS_TIMEOUT_S:.0f} s"
                )
                process.killed = True
                process.child.kill()
            return None
        if record.is_active(rank) or rank in self._given_up:
            return None
        if now_s < self._replacements.due_s(rank):
            self._wake_at_s.append(self._replacements.due_s(rank))
            return None
        self._replace(rank, now_s)
        return None

    def _end_job(self, reason: str) -> None:
        """End the whole job as failed, since no authority may be left to end the other nodes' ranks."""
        if self._end_job_through_launcher is None:
            return
        self._logger.error(f"Ending the job: {reason}")
        try:
            self._end_job_through_launcher(self._tool_environ, _STOP_GRACE_S)
        except (OSError, subprocess.SubprocessError) as error:
            self._logger.error(f"Cannot end the job: {error!r}")

    def _start(self, rank: int, replacement: bool) -> None:
        from tensorrt_llm._wide_ep_fault_tolerance import RankIdentity

        identity = RankIdentity(rank, self._world_size, self._gpus_per_node, replacement)
        environ = {**self._environ, self._rank_env: identity.env_value()}
        child = subprocess.Popen(  # nosec B603
            self._command,
            env=environ,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
            preexec_fn=self._die_with_supervisor,
        )
        pidfd = os.pidfd_open(child.pid)
        self._selector.register(pidfd, selectors.EVENT_READ, rank)
        self._processes[rank] = _Process(
            child, replacement, pidfd, None if replacement else 0, started_s=time.monotonic()
        )

    def _replace(self, rank: int, now_s: float) -> None:
        self._publish_given_up(rank, False)
        try:
            self._logger.warning(f"Starting a replacement of rank {rank}")
            self._start(rank, replacement=True)
        except (OSError, subprocess.SubprocessError) as error:
            self._logger.error(f"Cannot start a replacement of rank {rank}: {error!r}")
            self._replacements.delay(rank, now_s)
            self._wake_at_s.append(self._replacements.due_s(rank))

    def _give_up(self, rank: int) -> None:
        self._given_up.add(rank)
        self._publish_given_up(rank, True)

    def _publish_given_up(self, rank: int, given_up: bool) -> None:
        """Publish whether replacements of ``rank`` stopped, which a relaunched supervisor resets."""
        if self._store is None:
            return
        try:
            self._store.set(
                f"{self._membership.REPLACEMENT_GIVEN_UP_KEY}/{rank}", b"1" if given_up else b"0"
            )
        except (OSError, RuntimeError) as error:
            self._logger.warning(f"Cannot publish the replacement state of rank {rank}: {error!r}")

    def _reap(self, rank: int, now_s: float) -> None:
        process = self._processes[rank]
        self._selector.unregister(process.pidfd)
        os.close(process.pidfd)
        process.pidfd = None
        process.status = process.child.wait()
        self._logger.warning(
            f"Rank {rank} ({'replacement' if process.replacement else 'first process'}, incarnation "
            f"{process.incarnation}) ended with status {process.status}"
        )
        if process.status != 0 and self._first_failure is None:
            self._first_failure = _exit_status(process.status)
        served = process.admitted_s is not None and now_s - process.admitted_s >= _SERVED_S
        if not process.replacement or served:
            self._replacements.reset(rank)
            self._failed_replacements.pop(rank, None)
            return
        if process.status == self._membership.REPLACEMENT_CANNOT_START:
            self._logger.error(
                f"The replacement of rank {rank} cannot start, nor could another one"
            )
            self._give_up(rank)
            return
        reason = process.left_out_reason or f"each ended before serving for {_SERVED_S:.0f} s"
        last_reason, count = self._failed_replacements.get(rank, (reason, 0))
        count = count + 1 if last_reason == reason else 1
        self._failed_replacements[rank] = (reason, count)
        if count == _MAX_FAILED_REPLACEMENTS:
            self._logger.error(
                f"{count} replacements of rank {rank} failed in a row because {reason}, so no other one starts "
                f"and the deployment serves without rank {rank}"
            )
            self._give_up(rank)
            return
        self._replacements.delay(rank, now_s)

    def _connect(self) -> None:
        """Connect to the authority, and take the view it sends.

        Only reaching the store is bounded by the connect timeout. Once the store answers, the hello and the first
        view take up to the protocol's hello timeout.
        """
        self._connect_due_s = time.monotonic() + _CONNECT_TIMEOUT_S
        membership = self._membership
        address = os.environ[membership.RENDEZVOUS_ENV]
        connection = None
        try:
            store = membership.open_store(
                address, bytes.fromhex(os.environ[membership.SHARED_SECRET_ENV]), _CONNECT_TIMEOUT_S
            )
            connection = membership.open_node_supervisor_connection(
                store, self._node, self._ranks, socket.gethostname()
            )
            connection.sock.settimeout(membership.HELLO_TIMEOUT_S)
            payloads = []
            while not payloads:
                data = connection.sock.recv(1 << 16)
                if not data:
                    self._logger.error(
                        f"The membership authority refused the supervisor of node {self._node}"
                    )
                    self._refused = True
                    connection.sock.close()
                    return
                payloads = connection.lines.decode(data)
            views = [membership.NodeView.model_validate_json(payload) for payload in payloads]
        except PermissionError as error:
            self._logger.error(
                f"The store at {address} belongs to another deployment. Its values are not signed with this "
                f"supervisor's shared secret ({error})"
            )
            self._refused = True
            return
        except (
            OSError,
            RuntimeError,
            ValueError,
        ) as error:
            # Until the deployment serves, the store may not listen yet. After the hello, the authority takes the
            # closed connection as the failure of the node's ranks.
            expected = not self._served and connection is None
            (self._logger.debug if expected else self._logger.warning)(
                f"Cannot connect to the membership authority at {address}: {error!r}"
            )
            if connection is not None:
                connection.sock.close()
            return
        connection.sock.setblocking(False)
        self._store, self._connection = store, connection
        self._selector.register(connection.sock, selectors.EVENT_READ, _AUTHORITY)
        for view in views:
            self._on_view(view)

    def _receive_view(self) -> None:
        try:
            data = self._connection.sock.recv(1 << 16)
        except BlockingIOError:
            return
        except OSError:
            data = b""
        if data:
            try:
                for payload in self._connection.lines.decode(data):
                    self._on_view(self._membership.NodeView.model_validate_json(payload))
                return
            except ValueError as error:
                self._logger.error(
                    f"Received an invalid line from the membership authority: {error}"
                )
        self._logger.warning("The connection to the membership authority closed")
        self._selector.unregister(self._connection.sock)
        self._connection.sock.close()
        self._connection = None

    def _on_view(self, view) -> None:
        if self._view is not None:
            self._note_left_out(self._view.record, view.record)
        self._view = view
        self._view_received_s = time.monotonic()
        for node, _, connected in view.nodes:
            if connected:
                self._unconnected_relaunches.pop(node, None)
        if view.record.state not in (
            self._membership.MembershipState.STARTING,
            self._membership.MembershipState.SHUTDOWN,
            self._membership.MembershipState.ABORTED,
        ):
            self._served = True
        if self._host_weights_prefix is None:
            self._read_host_weights_prefix()

    def _note_left_out(self, previous, record) -> None:
        """Keep the reason of each replacement joining in ``previous`` that ``record`` leaves out with one."""
        for rank in self._ranks:
            process = self._processes.get(rank)
            if (
                process is not None
                and previous.is_joining(rank)
                and not record.is_active(rank)
                and record.reason
            ):
                process.left_out_reason = record.reason

    def _read_host_weights_prefix(self) -> None:
        with contextlib.suppress(OSError, RuntimeError):
            prefix = self._store.get_if_set(self._membership.HOST_WEIGHTS_PREFIX_KEY)
            if prefix is not None:
                self._host_weights_prefix = prefix.decode()

    def _relaunch_failed_nodes(self, now_s: float) -> None:
        """Relaunch the supervisor of every node whose supervisor left."""
        for node, host, connected in self._view.nodes:
            relaunched = self._relaunched.get(node)
            if connected:
                self._relaunches.reset(node)
            if (
                node == self._node
                or connected
                or (relaunched is not None and relaunched.poll() is None)
            ):
                continue
            relaunches = self._unconnected_relaunches.get(node, 0)
            if relaunches == _MAX_FAILED_RELAUNCHES:
                if relaunched is not None:
                    del self._relaunched[node]
                    ranks = self._node_ranks(node)
                    self._logger.error(
                        f"{relaunches} relaunched supervisors of node {node} ({host}) ended before they connected, "
                        f"the last with status {relaunched.returncode}, so node {node} is not relaunched again "
                        f"and the deployment serves without ranks {ranks}"
                    )
                    for rank in ranks:
                        self._publish_given_up(rank, True)
                continue
            if now_s < self._relaunches.due_s(node):
                self._wake_at_s.append(self._relaunches.due_s(node))
                continue
            self._relaunches.delay(node, now_s)
            self._wake_at_s.append(self._relaunches.due_s(node))
            self._logger.warning(f"Node {node} ({host}) failed. Relaunching its supervisor.")
            command = [
                *self._relaunch_command,
                "--host",
                f"{host}:1",
                sys.executable,
                os.path.abspath(__file__),
                "--node-rank",
                str(node),
                "--nodes",
                str(self._nodes),
                "--gpus-per-node",
                str(self._gpus_per_node),
                "--environment-from-stdin",
                "--",
                *self._command,
            ]
            try:
                process = subprocess.Popen(  # nosec B603
                    command, env=self._tool_environ, stdin=subprocess.PIPE
                )
                with contextlib.suppress(BrokenPipeError):
                    process.stdin.write(json.dumps(self._environ).encode() + b"\n")
                    process.stdin.close()
            except OSError as error:
                self._logger.error(f"Cannot relaunch node {node}: {error!r}")
                continue
            self._relaunched[node] = process
            self._unconnected_relaunches[node] = relaunches + 1
            self._selector.register(os.pidfd_open(process.pid), selectors.EVENT_READ, _RELAUNCHED)

    def _stop(self, status: int, terminate: bool = True, forward: Optional[int] = None) -> int:
        """Stop the node's ranks and return ``status``. Processes that are still running die with the supervisor."""
        self._unlink_host_weights()
        rank_0 = self._processes.get(0)
        if forward is not None and rank_0 is not None and rank_0.status is None:
            with contextlib.suppress(ProcessLookupError):
                rank_0.child.send_signal(forward)
        started = [process.child for process in self._processes.values()]
        running = [p for p in [*started, *self._relaunched.values()] if p.poll() is None]
        if not terminate:
            self._wait_for(running)
        for process in running:
            if process.poll() is None:
                process.terminate()
        self._wait_for(running)
        return status

    def _wait_for(self, processes: List[subprocess.Popen]) -> None:
        deadline_s = time.monotonic() + _STOP_GRACE_S
        for process in processes:
            with contextlib.suppress(subprocess.TimeoutExpired):
                process.wait(max(0.0, deadline_s - time.monotonic()))

    def _unlink_host_weights(self) -> None:
        if self._host_weights_prefix is None and self._store is not None:
            self._read_host_weights_prefix()
        if self._host_weights_prefix is None:
            return
        names = [
            name for name in os.listdir(_SHM_DIR) if name.startswith(self._host_weights_prefix)
        ]
        for name in names:
            with contextlib.suppress(FileNotFoundError):
                os.unlink(os.path.join(_SHM_DIR, name))
        self._logger.info(
            f"Unlinked the node's {len(names)} shared-memory segments of host weights"
        )


def _local_gpu_count(environ: Dict[str, str]) -> int:
    visible = environ.get("CUDA_VISIBLE_DEVICES")
    if visible is not None:
        return len([device for device in visible.split(",") if device])
    return len(glob.glob("/dev/nvidia[0-9]*"))


def _prrte_tool_environ(environ: Dict[str, str]) -> Dict[str, str]:
    """Return ``environ`` for prun or pterm, which keeps the launcher's settings but not this process's identity.

    It adds PRRTE's opt-in to running as root when this process runs as root.
    """
    tool_environ = {
        name: value
        for name, value in environ.items()
        if name not in _PROCESS_IDENTITY_VARS and not name.startswith(_PROCESS_IDENTITY_PREFIXES)
    }
    if os.geteuid() != 0:
        return tool_environ
    # mpirun refuses to start as root without this opt-in, so a root process of the job was started with it.
    return {**dict.fromkeys(_RUN_AS_ROOT_VARS, "1"), **tool_environ}


def _end_prrte_job(uri: str, pterm: str, environ: Dict[str, str], timeout_s: float) -> None:
    """End the PRRTE job at ``uri`` as failed by sending SIGTERM to its ``mpirun``, or with ``pterm`` otherwise."""
    # The URI's namespace ends with the PID of mpirun (``prterun-<host>-<PID>@...``).
    pid = uri.split("@", 1)[0].rsplit("-", 1)[-1]
    try:
        with open(f"/proc/{int(pid)}/comm") as comm:
            launcher = comm.read().strip()
    except (OSError, ValueError):
        launcher = None
    if launcher in ("prterun", "mpirun"):
        os.kill(int(pid), signal.SIGTERM)
        return
    from tensorrt_llm.logger import logger

    logger.error(
        f"The PMIx server URI {uri!r} names no mpirun of this host. Ending the job with pterm, after which "
        "mpirun exits with status 0."
    )
    subprocess.run(  # nosec B603
        [pterm, "--dvm-uri", uri],
        env=environ,
        stdin=subprocess.DEVNULL,
        timeout=timeout_s,
        check=True,
    )


def _prrte_job_control(
    environ: Dict[str, str], nodes: int
) -> Tuple[Optional[List[str]], Optional[Callable[[Dict[str, str], float], None]]]:
    """Return the ``prun`` command that starts a process on a host of the job and the function that ends the job.

    Raises:
        ValueError: A job of several nodes started by mpirun lacks the PMIx server URI, prun or pterm.
    """
    if "OMPI_COMM_WORLD_RANK" not in environ:
        return None, None
    uri = next((environ[name] for name in _PMIX_SERVER_URI_VARS if environ.get(name)), None)
    prun = shutil.which("prun", path=environ.get("PATH"))
    pterm = shutil.which("pterm", path=environ.get("PATH"))
    missing = [
        name
        for name, found in (("a PMIx server URI", uri), ("prun", prun), ("pterm", pterm))
        if found is None
    ]
    if missing and nodes > 1:
        raise ValueError(
            "Node 0's supervisor relaunches a failed node and ends the job when rank 0 ends through PRRTE, but it "
            f"lacks {', '.join(missing)}. Start trtllm-llmapi-launch with `mpirun --enable-recovery` of Open MPI 5, "
            "with prun and pterm on PATH"
        )
    if missing:
        return None, None
    relaunch_command = [prun, "--dvm-uri", uri, "-n", "1"]
    relaunch_command += ["--map-by", "ppr:1:node:OVERSUBSCRIBE", "--bind-to", "none"]
    return relaunch_command, functools.partial(_end_prrte_job, uri, pterm)


def start_session_node_supervisor(n_workers: int, env: Dict[str, str]) -> subprocess.Popen:
    """Start the node supervisor of ranks 1 to ``n_workers - 1`` for a session that runs rank 0.

    Raises:
        RuntimeError: trtllm-llmapi-launch is not on PATH.
    """
    launcher = shutil.which("trtllm-llmapi-launch")
    if launcher is None:
        raise RuntimeError(
            "Wide-EP fault tolerance starts the ranks with trtllm-llmapi-launch, which is not on PATH"
        )
    return subprocess.Popen(  # nosec B603
        [
            sys.executable,
            os.path.abspath(__file__),
            "--node-rank",
            "0",
            "--nodes",
            "1",
            "--gpus-per-node",
            str(n_workers),
            "--rank-0-in-session",
            "--",
            launcher,
        ],
        env=env,
    )


def stop_session_node_supervisor(supervisor: subprocess.Popen, grace_s: Optional[float]) -> None:
    """Stop ``supervisor``, after waiting ``grace_s`` seconds for it to exit unless ``grace_s`` is None."""
    if grace_s is not None:
        with contextlib.suppress(subprocess.TimeoutExpired):
            supervisor.wait(grace_s)
            return
    supervisor.terminate()
    if grace_s is not None:
        supervisor.wait()


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="_llmapi_node_supervisor.py",
        description="Supervise the ranks of one node under Wide-EP fault tolerance",
    )
    parser.add_argument("--node-rank", type=int, required=True)
    parser.add_argument("--nodes", type=int, required=True)
    parser.add_argument("--gpus-per-node", type=int, help="default: the GPUs this process sees")
    parser.add_argument(
        "--rank-0-in-session",
        action="store_true",
        help="start every rank but 0, which runs in the MPI session that starts this supervisor (MpiPoolSession)",
    )
    parser.add_argument(
        "--environment-from-stdin",
        action="store_true",
        help="start again with the environment of the JSON line on standard input",
    )
    parser.add_argument("command", nargs=argparse.REMAINDER, help="-- <command of each rank>")
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)
    if args.environment_from_stdin:
        environ = json.loads(sys.stdin.readline())
        restart = [sys.executable, os.path.abspath(__file__), "--node-rank", str(args.node_rank)]
        restart += ["--nodes", str(args.nodes)]
        if args.gpus_per_node is not None:
            restart += ["--gpus-per-node", str(args.gpus_per_node)]
        os.execve(sys.executable, [*restart, *args.command], environ)
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("give the command of each rank after --")
    environ = dict(os.environ)
    # mpirun's --enable-recovery is not visible here (Open MPI 5), so the docs state it instead of a check.
    if (
        "SLURM_PROCID" in environ
        and "OMPI_COMM_WORLD_RANK" not in environ
        and not args.rank_0_in_session
    ):
        parser.error(
            "Wide-EP fault tolerance does not support node supervisors started as Slurm tasks (srun): Slurm's PMIx "
            "plugin cancels the whole job step when one of them exits. Start trtllm-llmapi-launch with `mpirun "
            "--enable-recovery` inside the allocation (submitted with --no-requeue)"
        )
    gpus_per_node = args.gpus_per_node or _local_gpu_count(environ)
    if gpus_per_node <= 0:
        parser.error("this process sees no GPU: give --gpus-per-node")
    # This process uses neither MPI nor a GPU, so it skips the library initialization that calls MPI_Init.
    os.environ.update(
        TRT_LLM_NO_LIB_INIT="1", MPI4PY_RC_INITIALIZE="false", CUDA_VISIBLE_DEVICES=""
    )
    from tensorrt_llm._torch.distributed.membership import RENDEZVOUS_ENV, SHARED_SECRET_ENV

    for name in (SHARED_SECRET_ENV, RENDEZVOUS_ENV):
        if not environ.get(name):
            parser.error(
                f"export the same {name} on every node, as the Wide-EP fault tolerance docs describe"
            )
    relaunch_command, end_job = None, None
    if args.node_rank == 0:
        try:
            relaunch_command, end_job = _prrte_job_control(environ, args.nodes)
        except ValueError as error:
            parser.error(str(error))
    supervisor = NodeSupervisor(
        args.node_rank,
        args.nodes,
        gpus_per_node,
        command,
        environ,
        rank_0_in_session=args.rank_0_in_session,
        relaunch_command=relaunch_command,
        end_job=end_job,
    )
    for signum in (signal.SIGTERM, signal.SIGINT):
        signal.signal(signum, supervisor.on_signal)
    return supervisor.run()


if __name__ == "__main__":
    # The script's directory must not shadow the modules TensorRT-LLM imports.
    del sys.path[0]
    sys.exit(main())
