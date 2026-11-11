# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The node supervisor of a WideEP fault-tolerant deployment: keeps the ranks of one node served.

``trtllm-llmapi-launch`` runs one per node under fault tolerance. It starts the node's ranks as its children (each in
its own session, dying with the supervisor), follows the membership record over its own connection to the authority,
kills a rank the record no longer lists, starts a replacement of a lost rank on the same GPU, relaunches a lost
node's supervisor on node 0 under ``mpirun``, and ends the node's ranks once the deployment ended. It uses no GPU and
no MPI. See ``docs/source/features/wide-ep-fault-tolerance.md``.
"""

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

# The PMIx server URI variables a process mpirun starts holds, newest protocol first: prun reaches the job's PRRTE
# through any of them.
_PMIX_SERVER_URI_VARS = (
    "PMIX_SERVER_URI41",
    "PMIX_SERVER_URI4",
    "PMIX_SERVER_URI3",
    "PMIX_SERVER_URI21",
    "PMIX_SERVER_URI2",
)
_PR_SET_PDEATHSIG = 1
# Where the shared-memory segments of a node are (POSIX shared memory on Linux).
_SHM_DIR = "/dev/shm"
# How long a connection attempt to the authority lasts, and how long until the next one.
_CONNECT_TIMEOUT_S = 2.0
# How long the ranks get to end before SIGTERM, and again after it; twice this stays below the time the launcher gives
# the supervisor to end (Kubernetes' termination grace period, 30 s by default; Slurm's KillWait).
_STOP_GRACE_S = 10.0
# The delay before a replacement or a relaunch is attempted again doubles from the first to the last.
_MIN_BACKOFF_S = 1.0
_MAX_BACKOFF_S = 60.0


def _exit_status(returncode: int) -> int:
    """A process's status as a shell reports it: 128 plus the signal number for a process a signal ended."""
    return 128 - returncode if returncode < 0 else returncode


@dataclass
class _Process:
    """The process of one of the supervisor's ranks."""

    child: subprocess.Popen
    replacement: bool
    pidfd: Optional[int]  # readable once the process ended; None once it was reaped
    # The incarnation the authority assigned it, once a record listed it (an original rank's is 0).
    incarnation: Optional[int] = None
    listed: bool = False  # a record listed it as a live member
    admitted: bool = False  # a COMMITTED record listed it as a live member
    killed: bool = False
    status: Optional[int] = None  # once it ended


class _Backoff:
    """When the next attempt of each of several things is due: the delay doubles with each attempt, until
    :meth:`reset`."""

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


# Selector data of the descriptors other than a rank's process, whose data is its rank.
_WAKE = "wake"
_PARENT = "parent"
_AUTHORITY = "authority"
_RELAUNCHED = "relaunched"  # the process that relaunched a lost node's supervisor


class NodeSupervisor:
    """Supervises the ranks of node ``node`` of ``nodes``, ``gpus_per_node`` each; :meth:`run` returns the exit status.

    ``command`` starts each rank's process (with ``rank_0_in_session``, every rank but 0) in ``environ``, without the
    launcher's MPI variables. On node 0 under ``mpirun``, ``relaunch_command`` starts a process on the host given after
    it, and ``end_job`` ends the whole job as failed.
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
        from tensorrt_llm._utils import FAULT_TOLERANCE_RANK_ENV
        from tensorrt_llm.logger import logger
        from tensorrt_llm.mapping import Mapping

        self._membership = membership
        self._logger = logger
        self._rank_env = FAULT_TOLERANCE_RANK_ENV
        self._node = node
        self._nodes = nodes
        self._gpus_per_node = gpus_per_node
        self._world_size = nodes * gpus_per_node
        mapping = Mapping(
            world_size=self._world_size, tp_size=self._world_size, gpus_per_node=gpus_per_node
        )
        self._ranks = [
            rank for rank in range(self._world_size) if mapping.get_node_rank(rank) == node
        ]
        self._command = command
        self._rank_0_in_session = rank_0_in_session
        self._environ = {
            name: value
            for name, value in environ.items()
            if not name.startswith(membership.LAUNCHER_ENV_PREFIXES)
        }
        # Each rank started here is a one-process MPI world, which has no shared-memory peer: without the sm transport
        # it creates no /dev/shm segment that its exit (os._exit, or a kill) would leave behind.
        self._environ["OMPI_MCA_btl"] = "^sm"
        self._relaunch_command = relaunch_command
        self._end_job_through_launcher = end_job
        self._parent = os.getppid()
        self._parent_ended = False
        self._selector = selectors.DefaultSelector()
        # Written by the signal handler, so that a signal ends the selector's wait.
        self._wake_read, self._wake_write = socket.socketpair()
        self._wake_write.setblocking(False)
        self._store = None
        self._connection = None  # to the authority
        self._view = None  # the last NodeView the authority sent
        self._served = False  # the authority has listed the deployment as serving
        self._refused = False
        self._connect_due_s = 0.0
        self._wake_at_s: List[float] = []  # when the next step is due without an event
        self._processes: Dict[int, _Process] = {}
        self._replacements = _Backoff()  # per rank
        self._given_up: Set[int] = set()  # ranks this supervisor starts no replacement of
        self._relaunched: Dict[int, subprocess.Popen] = {}
        self._relaunches = _Backoff()  # per node
        self._host_weights_prefix: Optional[str] = None
        self._first_failure: Optional[int] = None
        self._signum: Optional[int] = None
        prctl = ctypes.CDLL(None, use_errno=True).prctl
        prctl.argtypes = [ctypes.c_int] + [ctypes.c_ulong] * 4
        supervisor = os.getpid()

        def die_with_supervisor() -> None:
            if prctl(_PR_SET_PDEATHSIG, int(signal.SIGKILL), 0, 0, 0) != 0:
                raise OSError(ctypes.get_errno(), "prctl(PR_SET_PDEATHSIG) failed")
            if os.getppid() != supervisor:  # the supervisor ended before the prctl
                os._exit(1)

        self._die_with_supervisor = die_with_supervisor

    def on_signal(self, signum: int, frame: Optional[FrameType] = None) -> None:
        """Signal handler: stop, forwarding the signal to rank 0's process."""
        if self._signum is None:
            self._signum = signum
        with contextlib.suppress(OSError):  # woken already, or the supervisor ended
            self._wake_write.send(b"\0")

    def run(self) -> int:
        # Bound, never listened on: keeps a second supervisor of the deployment off the node.
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as node_lock:
            try:
                node_lock.bind(self._membership.node_supervisor_address())
            except OSError:
                self._logger.error("another node supervisor of this deployment runs on this node")
                return 1
            try:
                self._watch_parent()
                self._selector.register(self._wake_read, selectors.EVENT_READ, _WAKE)
                self._connect()
                if self._refused:
                    return 1
                if self._view is not None and self._view.record.state in (
                    self._membership.MembershipState.SHUTDOWN,
                    self._membership.MembershipState.ABORT,
                ):
                    self._logger.error(
                        "the deployment at the rendezvous address ended: use a new join secret"
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

    def _watch_parent(self) -> None:
        try:
            pidfd = os.pidfd_open(self._parent)
        except ProcessLookupError:
            self._parent_ended = True
            return
        self._selector.register(pidfd, selectors.EVENT_READ, _PARENT)
        if os.getppid() != self._parent:  # it ended before the pidfd was opened
            self._parent_ended = True

    def _wait(self) -> None:
        """Wait for the next event, or the next step due without one, and take the events."""
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
        """Act on what changed; once the supervisor is to end, stop and return the exit status."""
        self._wake_at_s = []
        if self._signum is not None:
            self._logger.warning(f"signal {self._signum}: stopping the node's ranks")
            return self._stop(128 + self._signum, terminate=False, forward=self._signum)
        if self._parent_ended:
            self._logger.warning(
                "the process that started this supervisor ended: stopping the node's ranks"
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
            self._logger.info("the deployment shut down")
            return self._stop(0, terminate=False)
        if record is not None and record.state == state.ABORT:
            self._logger.error(f"the deployment aborted: {record.reason}")
            return self._stop(1)
        for rank in self._ranks:
            status = self._supervise(rank, record, now_s)
            if status is not None:
                return self._stop(status)
        if self._relaunch_command is not None and self._connection is not None and self._served:
            self._relaunch_lost_nodes(now_s)
        if (
            self._processes
            and all(process.status is not None for process in self._processes.values())
            and self._connection is None
        ):
            if self._served:
                self._logger.error(
                    "every rank of the node ended, and no authority follows the deployment"
                )
            return self._stop(self._first_failure or 0)
        return None

    def _supervise(self, rank: int, record, now_s: float) -> Optional[int]:
        """Act on ``rank``; the exit status if the node is to end."""
        process = self._processes.get(rank)
        ended = process is not None and process.status is not None
        if not self._served:
            if ended and process.status != 0:
                self._logger.error(
                    f"rank {rank} failed before the deployment serves (status {process.status}): the node fails"
                )
                if rank == 0:
                    self._end_job(
                        f"rank 0 failed before the deployment serves (status {process.status})"
                    )
                return _exit_status(process.status)
            return None
        if rank == 0:
            if ended:
                self._logger.error("rank 0 ended: the deployment ends with it")
                self._end_job(f"rank 0 ended (status {process.status})")
                return _exit_status(process.status)
            return None
        if self._connection is None:
            return None
        if process is not None and not ended:
            # Its process is the rank's only one: the authority lists no other process of the rank while it runs.
            if record.is_live(rank):
                process.listed = True
                process.incarnation = record.incarnations[rank]
                process.admitted |= record.state == self._membership.MembershipState.COMMITTED
            elif process.listed and not process.killed:
                self._logger.warning(
                    f"the record no longer lists rank {rank} (incarnation {process.incarnation}): killing it"
                )
                process.killed = True
                process.child.kill()
            return None
        if record.is_live(rank) or rank in self._given_up:
            return None
        if now_s < self._replacements.due_s(rank):
            self._wake_at_s.append(self._replacements.due_s(rank))
            return None
        self._replace(rank, now_s)
        return None

    def _end_job(self, reason: str) -> None:
        """End the whole job as failed through ``end_job``, if any: no authority may be left to end the other nodes'
        ranks."""
        if self._end_job_through_launcher is None:
            return
        self._logger.error(f"{reason}: ending the job")
        try:
            self._end_job_through_launcher(self._environ, _STOP_GRACE_S)
        except (OSError, subprocess.SubprocessError) as error:
            self._logger.error(f"cannot end the job: {error!r}")

    def _start(self, rank: int, replacement: bool) -> None:
        identity = f"{rank},{self._world_size},{self._gpus_per_node},{int(replacement)}"
        environ = {**self._environ, self._rank_env: identity}
        child = subprocess.Popen(  # nosec B603
            self._command,
            env=environ,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
            preexec_fn=self._die_with_supervisor,
        )
        pidfd = os.pidfd_open(child.pid)
        self._selector.register(pidfd, selectors.EVENT_READ, rank)
        self._processes[rank] = _Process(child, replacement, pidfd, None if replacement else 0)

    def _replace(self, rank: int, now_s: float) -> None:
        """Start a replacement of ``rank``, whose process ended or, after this node was lost, ran under the supervisor
        this one replaces; the authority assigns its incarnation."""
        try:
            self._logger.warning(f"starting a replacement of rank {rank}")
            self._start(rank, replacement=True)
        except (OSError, subprocess.SubprocessError) as error:
            self._logger.error(f"cannot start a replacement of rank {rank}: {error!r}")
            self._replacements.delay(rank, now_s)
            self._wake_at_s.append(self._replacements.due_s(rank))

    def _reap(self, rank: int, now_s: float) -> None:
        """Take the status of ``rank``'s process, which ended."""
        process = self._processes[rank]
        self._selector.unregister(process.pidfd)
        os.close(process.pidfd)
        process.pidfd = None
        process.status = process.child.wait()
        self._logger.warning(
            f"rank {rank} ({'replacement' if process.replacement else 'first process'}, incarnation "
            f"{process.incarnation}) ended with status {process.status}"
        )
        if process.status != 0 and self._first_failure is None:
            self._first_failure = _exit_status(process.status)
        if not process.replacement or process.admitted:
            self._replacements.reset(rank)
            return
        if process.status == self._membership.REPLACEMENT_CANNOT_START:
            self._logger.error(
                f"the replacement of rank {rank} cannot start, nor could another one"
            )
            self._given_up.add(rank)
            return
        self._replacements.delay(rank, now_s)

    def _connect(self) -> None:
        """Connect to the authority if it answers within ``_CONNECT_TIMEOUT_S``, and take the view it sends."""
        self._connect_due_s = time.monotonic() + _CONNECT_TIMEOUT_S
        membership = self._membership
        address = os.environ[membership.RENDEZVOUS_ENV]
        deadline_s = time.monotonic() + _CONNECT_TIMEOUT_S

        def left_s() -> float:
            return max(0.001, deadline_s - time.monotonic())

        connection = None
        try:
            store = membership.open_store(
                address, bytes.fromhex(os.environ[membership.JOIN_SECRET_ENV]), left_s()
            )
            connection = membership.open_node_supervisor_connection(
                store, self._node, self._ranks, socket.gethostname(), left_s()
            )
            connection.sock.settimeout(left_s())
            payloads = []
            while not payloads:
                data = connection.sock.recv(1 << 16)
                if not data:
                    self._logger.error(
                        f"the membership authority refused the supervisor of node {self._node}"
                    )
                    self._refused = True
                    connection.sock.close()
                    return
                payloads = connection.lines.decode(data)
            views = [membership.NodeView.model_validate_json(payload) for payload in payloads]
        except PermissionError as error:
            self._logger.error(
                f"the store at {address} is another deployment's: its values are not signed with this "
                f"supervisor's join secret ({error})"
            )
            self._refused = True
            return
        except (
            OSError,
            RuntimeError,
            ValueError,
        ) as error:  # torch's store errors are RuntimeErrors
            # Until the deployment serves, the store may not listen yet.
            (self._logger.warning if self._served else self._logger.debug)(
                f"cannot connect to the membership authority at {address}: {error!r}"
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
                self._logger.error(f"a line from the membership authority is not valid: {error}")
        self._logger.warning("the connection to the membership authority closed")
        self._selector.unregister(self._connection.sock)
        self._connection.sock.close()
        self._connection = None

    def _on_view(self, view) -> None:
        self._view = view
        if view.record.state not in (
            self._membership.MembershipState.STARTING,
            self._membership.MembershipState.SHUTDOWN,
            self._membership.MembershipState.ABORT,
        ):
            self._served = True
        if self._host_weights_prefix is None:
            self._read_host_weights_prefix()

    def _read_host_weights_prefix(self) -> None:
        with contextlib.suppress(OSError, RuntimeError):
            prefix = self._store.get_if_set(self._membership.HOST_WEIGHTS_PREFIX_KEY)
            if prefix is not None:
                self._host_weights_prefix = prefix.decode()

    def _relaunch_lost_nodes(self, now_s: float) -> None:
        """Node 0: relaunch the supervisor of every node whose supervisor left, unless one it relaunched still runs."""
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
            if now_s < self._relaunches.due_s(node):
                self._wake_at_s.append(self._relaunches.due_s(node))
                continue
            self._relaunches.delay(node, now_s)
            self._wake_at_s.append(self._relaunches.due_s(node))
            self._logger.warning(f"node {node} ({host}) was lost: relaunching its supervisor")
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
                process = subprocess.Popen(command, env=self._environ, stdin=subprocess.PIPE)  # nosec B603
                with contextlib.suppress(BrokenPipeError):
                    process.stdin.write(json.dumps(self._environ).encode() + b"\n")
                    process.stdin.close()
            except OSError as error:
                self._logger.error(f"cannot relaunch node {node}: {error!r}")
                continue
            self._relaunched[node] = process
            self._selector.register(os.pidfd_open(process.pid), selectors.EVENT_READ, _RELAUNCHED)

    def _stop(self, status: int, terminate: bool = True, forward: Optional[int] = None) -> int:
        """End: unlink the node's host weights, send ``forward`` (a signal this supervisor received) to rank 0's
        process, SIGTERM to the processes it started at once (``terminate``) or after ``_STOP_GRACE_S``, and return
        ``status`` once they ended or another ``_STOP_GRACE_S`` passed (those left die with this process)."""
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
        """Wait until ``processes`` ended, at most ``_STOP_GRACE_S``."""
        deadline_s = time.monotonic() + _STOP_GRACE_S
        for process in processes:
            with contextlib.suppress(subprocess.TimeoutExpired):
                process.wait(max(0.0, deadline_s - time.monotonic()))

    def _unlink_host_weights(self) -> None:
        """Unlink the node's shared-memory segments of host weights."""
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
            f"unlinked the node's {len(names)} shared-memory segments of host weights"
        )


def _local_gpu_count(environ: Dict[str, str]) -> int:
    visible = environ.get("CUDA_VISIBLE_DEVICES")
    if visible is not None:
        return len([device for device in visible.split(",") if device])
    return len(glob.glob("/dev/nvidia[0-9]*"))


def _end_prrte_job(uri: str, pterm: str, environ: Dict[str, str], timeout_s: float) -> None:
    """End the job of the PRRTE at the PMIx server URI ``uri`` as failed: SIGTERM to its ``mpirun``, whose PID ends
    the URI's namespace (``prterun-<host>-<PID>@...``), which then ends every process of the job and exits with
    status 1. Where that PID is no ``mpirun`` of this host, ``pterm`` ends the job instead (with ``environ``, within
    ``timeout_s``), and ``mpirun`` exits with status 0."""
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
        f"the PMIx server URI {uri!r} names no mpirun of this host: ending the job with pterm, after which "
        "mpirun exits with status 0"
    )
    subprocess.run(  # nosec B603
        [pterm, "--dvm-uri", uri],
        env=environ,
        stdin=subprocess.DEVNULL,
        timeout=timeout_s,
        check=True,
    )


def _prrte_job_control(
    environ: Dict[str, str],
) -> Tuple[Optional[List[str]], Optional[Callable[[Dict[str, str], float], None]]]:
    """Under ``mpirun``, the job's PRRTE command that starts a process on a host of the job (``prun``; the host and
    the command come after it) and the function that ends the whole job as failed (:func:`_end_prrte_job`); None for
    each under another launcher."""
    if "OMPI_COMM_WORLD_RANK" not in environ:
        return None, None
    uri = next((environ[name] for name in _PMIX_SERVER_URI_VARS if environ.get(name)), None)
    prun = shutil.which("prun", path=environ.get("PATH"))
    pterm = shutil.which("pterm", path=environ.get("PATH"))
    if uri is None or prun is None or pterm is None:
        from tensorrt_llm.logger import logger

        logger.error(
            "started by mpirun without a PMIx server URI, prun or pterm: a lost node is not relaunched, and the "
            "end of rank 0 does not end the job"
        )
        return None, None
    relaunch_command = [
        prun,
        "--dvm-uri",
        uri,
        "-n",
        "1",
        "--map-by",
        "ppr:1:node:OVERSUBSCRIBE",
        "--bind-to",
        "none",
    ]
    return relaunch_command, functools.partial(_end_prrte_job, uri, pterm)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="_llmapi_node_supervisor.py",
        description="WideEP fault tolerance: supervise the ranks of one node",
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
            "WideEP fault tolerance does not support node supervisors started as Slurm tasks (srun): Slurm's PMIx "
            "plugin cancels the whole job step when one of them exits. Start trtllm-llmapi-launch with `mpirun "
            "--enable-recovery` inside the allocation (submitted with --no-requeue)"
        )
    gpus_per_node = args.gpus_per_node or _local_gpu_count(environ)
    if gpus_per_node <= 0:
        parser.error("this process sees no GPU: give --gpus-per-node")
    # This process uses neither MPI nor a GPU: importing TensorRT-LLM skips its library initialization, which
    # includes the C++ MPI_Init, and mpi4py does not initialize MPI.
    os.environ.update(
        TRT_LLM_NO_LIB_INIT="1", MPI4PY_RC_INITIALIZE="false", CUDA_VISIBLE_DEVICES=""
    )
    relaunch_command, end_job = _prrte_job_control(environ) if args.node_rank == 0 else (None, None)
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
    # Python puts this script's directory first on the module search path: its modules must not shadow those that
    # TensorRT-LLM imports.
    del sys.path[0]
    sys.exit(main())
