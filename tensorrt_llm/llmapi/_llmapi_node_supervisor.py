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
"""Start and replace the ranks of one node under Wide-EP fault tolerance.

``trtllm-llmapi-launch`` runs one node supervisor per node, as the task that
``mpirun --enable-recovery`` starts there. The supervisor starts every rank on its node as a
separate process, follows the record that the membership authority sends, and starts a
replacement on the same GPU for a rank that the membership drops.
"""

import argparse
import concurrent.futures
import contextlib
import glob
import os
import selectors
import shutil
import signal
import socket
import subprocess  # nosec B404
import sys
import threading
import time
from dataclasses import dataclass
from types import FrameType
from typing import Callable, Dict, List, NoReturn, Optional, Set

_SHM_DIR = "/dev/shm"
_CONNECT_TIMEOUT_S = 2.0
_CONNECT_WARNING_INTERVAL_S = 30.0
# How long the ranks may drain after a stop signal, and then end after SIGTERM. At a job end, mpirun
# kills the supervisor within a second of its signal, so the ranks get no such grace there.
_STOP_GRACE_S = 10.0
# Delay before the next replacement after a failed one. It doubles with each failure in a row.
_BACKOFF_S = 1.0
# Failed replacements in a row, for any reason, after which a rank is given up.
_MAX_FAILED_REPLACEMENTS = 3
# A replacement that served this long before it ended counts as healthy. Its end is a new failure,
# not a failed replacement, so its rank's failure count resets.
_HEALTHY_SERVING_S = 600.0
# Run as ``_SCRIPT --exec-rank <supervisor PID> <command>``, this script execs a rank's command.
_EXEC_RANK = "--exec-rank"
_SCRIPT = os.path.abspath(__file__)


def _exit_status(returncode: int) -> int:
    """Return the exit status as a shell reports it: 128 + N for a process killed by signal N."""
    return 128 - returncode if returncode < 0 else returncode


@dataclass
class _Process:
    """A rank process that this supervisor started."""

    child: subprocess.Popen
    replacement: bool
    pidfd: Optional[int]
    incarnation: Optional[int] = None
    listed: bool = False
    # When a COMMITTED record first listed it as an active member.
    admitted_s: Optional[float] = None
    # The reason the membership left it out while it joined.
    left_out_reason: str = ""
    killed_s: Optional[float] = None
    status: Optional[int] = None
    started_s: float = 0.0


# Selector data for descriptors other than rank pidfds, which carry the rank instead.
_WAKE = "wake"
_PARENT = "parent"
_AUTHORITY = "authority"


class NodeSupervisor:
    """Start, supervise and replace the ranks of one node.

    It unlinks the node's host weights once the deployment ends. On node 0 under ``mpirun``, it ends
    the job as failed.

    Args:
        command: The command that starts each rank process. With ``rank_0_in_session``, it
            starts every rank but 0.
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
        end_job: Optional[Callable[[], None]] = None,
    ):
        from tensorrt_llm._torch.distributed import membership
        from tensorrt_llm.llmapi._llmapi_process_guard import _signal_group
        from tensorrt_llm.logger import logger

        self._membership = membership
        self._logger = logger
        self._signal_group = _signal_group
        self._node = node
        self._gpus_per_node = gpus_per_node
        self._world_size = nodes * gpus_per_node
        self._ranks = list(range(node * gpus_per_node, (node + 1) * gpus_per_node))
        self._command = command
        self._rank_0_in_session = rank_0_in_session
        self._environ = membership.rank_env(environ)
        self._parent = os.getppid()
        self._parent_ended = False
        self._selector = selectors.DefaultSelector()
        # Written by the signal handler and the connector, so that they end the selector's wait.
        self._wake_read, self._wake_write = socket.socketpair()
        self._wake_write.setblocking(False)
        self._connecting: Optional[concurrent.futures.Future] = None
        self._connection = None
        # A connection that waits for the authority's first view until _view_due_s.
        self._connection_awaiting_view = None
        self._view_due_s = 0.0
        self._view = None
        self._view_received_s = 0.0
        # Set once a record is past STARTING. Before that, the failure of any rank stops the node.
        self._deployment_started = False
        # The authority closed the connection before its first view, or its store holds values not
        # signed with this node's shared secret.
        self._refused_by_authority = False
        # The last connect reached the authority, which then sent a view or refused the supervisor.
        # While it is reachable, the node's ranks are replaced rather than stopped with the node.
        self._authority_reached = False
        self._connect_due_s = 0.0
        self._connect_warning_due_s = time.monotonic() + _CONNECT_WARNING_INTERVAL_S
        # Times at which the loop must step again. Each step rebuilds the list.
        self._deadlines_s: List[float] = []
        self._processes: Dict[int, _Process] = {}
        self._given_up: Set[int] = set()
        # Per rank, the failed replacements in a row and when the next one may start.
        self._failed_replacements: Dict[int, int] = {}
        self._replacement_due_s: Dict[int, float] = {}
        self._first_failure: Optional[int] = None
        self._signum: Optional[int] = None
        self._signal_handled = False
        # When the supervisor stops the ranks still running after SHUTDOWN, once the authority left.
        self._shutdown_grace_due_s: Optional[float] = None
        self._signal_grace_due_s = 0.0
        # This process may have threads, so no Python code runs between a rank's fork and exec. A
        # new interpreter arms the rank's parent-death signal instead.
        self._exec_rank = [sys.executable, "-S", _SCRIPT, _EXEC_RANK, str(os.getpid())]
        self._end_job_fn = end_job
        self._host_weights_prefix = f"{membership.host_weights_name()}_"

    def on_signal(self, signum: int, frame: Optional[FrameType] = None) -> None:
        """Unlink the node's host weights, then stop the supervisor and forward the signal to rank 0.

        The unlink runs here, before the loop wakes, since the launcher may kill this process soon
        after the signal (mpirun within a second).
        """
        if self._signum is None:
            self._signum = signum
            self._signal_grace_due_s = time.monotonic() + _STOP_GRACE_S
            self._unlink_host_weights()
        self._wake()

    def _wake(self) -> None:
        with contextlib.suppress(OSError):
            self._wake_write.send(b"\0")

    def run(self) -> int:
        try:
            self._watch_parent()
            self._selector.register(self._wake_read, selectors.EVENT_READ, _WAKE)
            self._connect()
            # The first view tells whether the node starts its ranks.
            while self._signum is None and (
                self._connecting is not None or self._connection_awaiting_view is not None
            ):
                if self._connection_awaiting_view is not None:
                    self._deadlines_s = [self._view_due_s]
                else:
                    self._deadlines_s = [self._connect_due_s]
                self._wait()
                self._take_connection()
                self._check_first_view(time.monotonic())
            if self._signum is not None:
                return self._step(time.monotonic())
            if self._refused_by_authority:
                return 1
            if self._view is not None and self._view.record.state in (
                self._membership.MembershipState.SHUTDOWN,
                self._membership.MembershipState.ABORTED,
            ):
                self._logger.error(
                    "The deployment at the rendezvous address ended. Use a new shared secret."
                )
                return 1
            self._start_first_processes()
            while True:
                status = self._step(time.monotonic())
                if status is not None:
                    return status
                self._wait()
        except Exception as error:
            # No step stops the node any more, so stop its ranks and, on node 0, end the job.
            self._stop(1, end_job_reason=f"the node supervisor failed: {error!r}")
            raise
        finally:
            for key in list(self._selector.get_map().values()):
                if isinstance(key.fileobj, int):
                    os.close(key.fileobj)
                else:
                    key.fileobj.close()
            self._selector.close()
            self._wake_write.close()

    def _start_first_processes(self) -> None:
        """Start every rank's first process, unless the deployment has started."""
        if not self._deployment_started:
            for rank in self._ranks:
                if rank != 0 or not self._rank_0_in_session:
                    self._start(rank, replacement=False)

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
        if self._deadlines_s:
            timeout_s = max(0.0, min(self._deadlines_s) - time.monotonic())
        for key, _ in self._selector.select(timeout_s):
            self._on_event(key)

    def _on_event(self, key: selectors.SelectorKey) -> None:
        if key.data == _WAKE:
            self._wake_read.recv(1 << 12)
        elif key.data == _PARENT:
            self._parent_ended = True
            self._close_pidfd(key)
        elif key.data == _AUTHORITY:
            self._receive_view()
        else:
            self._reap(key.data, time.monotonic())

    def _close_pidfd(self, key: selectors.SelectorKey) -> None:
        self._selector.unregister(key.fileobj)
        os.close(key.fileobj)

    def _step(self, now_s: float) -> Optional[int]:
        """Handle the latest events and timers, and return the exit status once the node stops."""
        self._deadlines_s = []
        if self._signum is not None:
            return self._stop_on_signal(now_s)
        if self._parent_ended:
            self._logger.warning("Stopping the node's ranks because the supervisor's parent ended")
            return self._stop(128 + signal.SIGTERM)
        self._take_connection()
        if self._refused_by_authority:
            return self._stop(1)
        record = None if self._view is None else self._view.record
        if record is not None and record.state == self._membership.MembershipState.SHUTDOWN:
            return self._stop_after_shutdown(record, now_s)
        if (
            self._connection is None
            and self._connection_awaiting_view is None
            and self._connecting is None
        ):
            if now_s >= self._connect_due_s:
                self._connect()
            self._deadlines_s.append(self._connect_due_s)
        self._check_first_view(now_s)
        if record is not None:
            status = self._check_abort(record)
            if status is not None:
                return status
        for rank in self._ranks:
            status = self._supervise(rank, record, now_s)
            if status is not None:
                return status
        if self._connection is None and not self._authority_reached and self._all_ranks_ended():
            if self._deployment_started:
                self._logger.error("Every rank of the node ended with no authority to follow")
            return self._stop(self._first_failure or 0)
        return None

    def _stop_on_signal(self, now_s: float) -> Optional[int]:
        """Stop the node on a signal, and return the exit status once it stopped.

        The launcher signals every node supervisor, and the signal makes rank 0 shut the deployment
        down. The ranks then drain their requests and end on their own. The ones still running a
        grace period after the signal are stopped. A launcher that kills the supervisor sooner, as
        mpirun does at a job end, ends the ranks through their parent-death signal.
        """
        if not self._signal_handled:
            self._signal_handled = True
            rank_0 = self._processes.get(0)
            # Through the pidfd, which unlike Popen.send_signal never reaps the process.
            if rank_0 is not None and rank_0.status is None:
                with contextlib.suppress(ProcessLookupError):
                    signal.pidfd_send_signal(rank_0.pidfd, self._signum)
        record = None if self._view is None else self._view.record
        if record is not None:
            status = self._check_abort(record)
            if status is not None:
                return status
        if not self._active_ranks_running(record):
            if record is not None and record.state == self._membership.MembershipState.SHUTDOWN:
                return self._stop(self._shutdown_status(record))
            return self._stop(128 + self._signum)
        if now_s < self._signal_grace_due_s:
            self._deadlines_s.append(self._signal_grace_due_s)
            return None
        self._logger.warning(f"Stopping the node's ranks on signal {self._signum}")
        return self._stop(128 + self._signum)

    def _stop_after_shutdown(self, record, now_s: float) -> Optional[int]:
        """Stop the node once its active ranks ended after a shutdown, and return the exit status.

        The active ranks end after they leave the membership, and the authority ends once all of
        them left. Ranks still running a grace period after the authority's connection closed are
        stopped. Rank 0 is never stopped here, since its end is the deployment's end, which its
        launcher bounds. Each rank's end wakes the supervisor.
        """
        running = self._active_ranks_running(record)
        if not running:
            self._logger.info("The deployment shut down")
            return self._stop(self._shutdown_status(record))
        if self._connection is not None or 0 in running:
            return None
        if self._shutdown_grace_due_s is None:
            self._shutdown_grace_due_s = now_s + _STOP_GRACE_S
        if now_s < self._shutdown_grace_due_s:
            self._deadlines_s.append(self._shutdown_grace_due_s)
            return None
        self._logger.error(
            f"Stopping ranks {running}, which did not end within {_STOP_GRACE_S:.0f} s of the "
            "authority's end after the shutdown"
        )
        return self._stop(self._shutdown_status(record) or 1)

    def _shutdown_status(self, record) -> int:
        """Return the first nonzero exit status of an active rank, in rank order, or 0."""
        return next(
            (
                _exit_status(process.status)
                for rank, process in sorted(self._processes.items())
                if process.status and record.is_active(rank)
            ),
            0,
        )

    def _check_abort(self, record) -> Optional[int]:
        """Stop the node if ``record`` aborts the deployment, and return the exit status."""
        state = self._membership.MembershipState
        if record.state == state.ABORTED:
            self._logger.error(f"Deployment aborted: {record.reason}")
            return self._stop(1, end_job_reason=f"the deployment aborted: {record.reason}")
        return None

    def _all_ranks_ended(self) -> bool:
        return bool(self._processes) and not self._ranks_running()

    def _ranks_running(self) -> bool:
        return any(process.status is None for process in self._processes.values())

    def _active_ranks_running(self, record) -> List[int]:
        """Return the node's running ranks that ``record`` lists as active, or all without one."""
        return [
            rank
            for rank, process in sorted(self._processes.items())
            if process.status is None and (record is None or record.is_active(rank))
        ]

    def _supervise(self, rank: int, record, now_s: float) -> Optional[int]:
        """Act on ``rank``, and return the exit status once the node stopped."""
        process = self._processes.get(rank)
        if not self._deployment_started:
            return self._check_before_serving(rank, process)
        if rank == 0:
            return self._check_rank_0(process)
        if self._connection is None:
            return None
        if process is not None and process.status is None:
            self._check_running(rank, process, record, now_s)
        else:
            self._replace_when_due(rank, record, now_s)
        return None

    def _check_before_serving(self, rank: int, process: Optional[_Process]) -> Optional[int]:
        """Stop the node if ``rank`` failed before the deployment serves, and return its status."""
        if process is None or process.status in (None, 0):
            return None
        self._logger.error(f"Rank {rank} failed before serving (status {process.status})")
        reason = f"rank 0 failed before serving (status {process.status})" if rank == 0 else None
        return self._stop(_exit_status(process.status), end_job_reason=reason)

    def _check_rank_0(self, process: Optional[_Process]) -> Optional[int]:
        """Stop the node once rank 0 ended, which ends the deployment, and return its status."""
        if process is None or process.status is None:
            return None
        self._logger.error("Rank 0 ended, so the deployment ends with it")
        return self._stop(
            _exit_status(process.status), end_job_reason=f"rank 0 ended (status {process.status})"
        )

    def _check_running(self, rank: int, process: _Process, record, now_s: float) -> None:
        """Track the membership of a running rank, and kill it once the membership drops it."""
        if record.is_active(rank):
            process.listed = True
            process.incarnation = record.incarnations[rank]
            if (
                process.admitted_s is None
                and record.state == self._membership.MembershipState.COMMITTED
            ):
                process.admitted_s = now_s
        elif process.killed_s is not None:
            self._check_killed(rank, process, now_s)
        elif process.listed:
            self._logger.warning(
                f"Killing rank {rank} (incarnation {process.incarnation}), "
                "which the membership no longer lists"
            )
            self._kill(process)
        # A view received before this replacement started reports its predecessor's stall.
        elif rank in self._view.stalled_replacements and self._view_received_s > process.started_s:
            self._logger.warning(
                f"Killing the replacement of rank {rank}, which made no progress for "
                f"{self._membership.RENDEZVOUS_TIMEOUT_S:.0f} s"
            )
            self._kill(process)
        # The authority times a replacement's startup only once it connects, so time the part before
        # that here.
        elif not self._membership.has_rank(self._view.connected_mask, rank):
            due_s = process.started_s + self._membership.RENDEZVOUS_TIMEOUT_S
            if now_s < due_s:
                self._deadlines_s.append(due_s)
                return
            self._logger.warning(
                f"Killing rank {rank}'s process, which is not connected to the membership "
                f"authority {self._membership.RENDEZVOUS_TIMEOUT_S:.0f} s after it started"
            )
            self._kill(process)

    def _check_killed(self, rank: int, process: _Process, now_s: float) -> None:
        """Give up ``rank`` if its killed process did not end within the rendezvous timeout."""
        if rank in self._given_up:
            return
        due_s = process.killed_s + self._membership.RENDEZVOUS_TIMEOUT_S
        if now_s < due_s:
            self._deadlines_s.append(due_s)
            return
        self._logger.error(
            f"Rank {rank}'s process (PID {process.child.pid}) did not end within "
            f"{self._membership.RENDEZVOUS_TIMEOUT_S:.0f} s of SIGKILL, so its GPU memory may not "
            f"be free. The deployment serves without rank {rank}"
        )
        self._give_up(rank)

    def _replace_when_due(self, rank: int, record, now_s: float) -> None:
        """Replace ``rank`` if the membership left it out and its backoff has passed."""
        if record.is_active(rank) or rank in self._given_up:
            return
        due_s = self._replacement_due_s.get(rank, 0.0)
        if now_s < due_s:
            self._deadlines_s.append(due_s)
            return
        self._replace(rank, now_s)

    def _kill(self, process: _Process) -> None:
        """Kill a rank's process group, helper processes included.

        Step again at the rendezvous timeout, so that :meth:`_check_killed` gives the rank up if its
        process has not ended by then.
        """
        process.killed_s = time.monotonic()
        self._deadlines_s.append(process.killed_s + self._membership.RENDEZVOUS_TIMEOUT_S)
        self._signal_group(process.child.pid, signal.SIGKILL)

    def _start(self, rank: int, replacement: bool) -> None:
        from tensorrt_llm._wide_ep_fault_tolerance import FAULT_TOLERANCE_RANK_ENV, RankIdentity

        identity = RankIdentity(rank, self._world_size, self._gpus_per_node, replacement)
        environ = {**self._environ, FAULT_TOLERANCE_RANK_ENV: identity.env_value()}
        child = subprocess.Popen(  # nosec B603
            [*self._exec_rank, *self._command],
            env=environ,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
        )
        pidfd = os.pidfd_open(child.pid)
        self._selector.register(pidfd, selectors.EVENT_READ, rank)
        self._processes[rank] = _Process(
            child, replacement, pidfd, None if replacement else 0, started_s=time.monotonic()
        )

    def _replace(self, rank: int, now_s: float) -> None:
        try:
            self._logger.warning(f"Starting a replacement of rank {rank}")
            self._start(rank, replacement=True)
        except (OSError, subprocess.SubprocessError) as error:
            self._logger.error(f"Cannot start a replacement of rank {rank}: {error!r}")
            self._count_failed_replacement(rank, f"it could not start: {error!r}", now_s)

    def _give_up(self, rank: int) -> None:
        self._given_up.add(rank)
        self._report_given_up()

    def _report_given_up(self) -> None:
        """Report the given-up ranks to the authority, which publishes them in the record.

        Without a connection, the next connection reports them once the authority admitted it.
        """
        connection = self._connection
        if connection is None:
            return
        report = self._membership.NodeReport(given_up=tuple(sorted(self._given_up)))
        try:
            connection.sock.sendall(connection.lines.encode(report.model_dump_json().encode()))
        except OSError as error:
            self._logger.warning(f"Cannot report to the membership authority: {error!r}")
            self._close_connection(connection)

    def _reap(self, rank: int, now_s: float) -> None:
        process = self._processes[rank]
        self._selector.unregister(process.pidfd)
        os.close(process.pidfd)
        process.pidfd = None
        # Until the rank's process is reaped, its PID and so its process group's ID cannot be reused.
        # Kill the group now to end the helpers the rank left, then reap the rank.
        self._signal_group(process.child.pid, signal.SIGKILL)
        process.status = process.child.wait()
        self._logger.warning(
            f"Rank {rank} ({'replacement' if process.replacement else 'first process'}, "
            f"incarnation {process.incarnation}) ended with status {process.status}"
        )
        if process.status != 0 and self._first_failure is None:
            self._first_failure = _exit_status(process.status)
        healthy = (
            process.admitted_s is not None and now_s - process.admitted_s >= _HEALTHY_SERVING_S
        )
        if not process.replacement or healthy:
            self._failed_replacements.pop(rank, None)
            self._replacement_due_s.pop(rank, None)
            return
        if process.status == self._membership.REPLACEMENT_CANNOT_START:
            self._logger.error(
                f"The replacement of rank {rank} cannot start, nor could another one"
            )
            self._give_up(rank)
            return
        reason = (
            process.left_out_reason or f"it ended before serving for {_HEALTHY_SERVING_S:.0f} s"
        )
        self._count_failed_replacement(rank, reason, now_s)

    def _count_failed_replacement(self, rank: int, reason: str, now_s: float) -> None:
        """Count a replacement of ``rank`` that failed because of ``reason``.

        Give the rank up after too many in a row. Otherwise, start the next one after the backoff.
        """
        count = self._failed_replacements.get(rank, 0) + 1
        self._failed_replacements[rank] = count
        if count == _MAX_FAILED_REPLACEMENTS:
            self._logger.error(
                f"{count} replacements of rank {rank} failed in a row, the last because {reason}. "
                f"The deployment serves without rank {rank}"
            )
            self._give_up(rank)
            return
        self._replacement_due_s[rank] = now_s + _BACKOFF_S * 2 ** (count - 1)
        self._deadlines_s.append(self._replacement_due_s[rank])

    def _connect(self) -> None:
        """Start connecting to the authority, whose first view then arrives as an event.

        Reaching the store and sending the hello are bounded by the connect timeout. The first view
        may take up to the protocol's hello timeout. The supervisor keeps taking events meanwhile.
        """
        self._connect_due_s = time.monotonic() + _CONNECT_TIMEOUT_S
        self._connecting = concurrent.futures.Future()
        self._connecting.add_done_callback(lambda _: self._wake())
        # Off the main loop, so that a signal is acted on at once. The process exits without
        # waiting for a daemon thread.
        connector = threading.Thread(
            target=self._open_connection, args=(self._connecting,), daemon=True
        )
        connector.start()

    def _open_connection(self, connecting: concurrent.futures.Future) -> None:
        """Open the connection to the authority as the result of ``connecting``.

        Send no hello once the supervisor stopped, since the authority would count the node as
        connected.
        """
        membership = self._membership
        try:
            store = membership.open_store(os.environ[membership.RENDEZVOUS_ENV], _CONNECT_TIMEOUT_S)
            if self._connecting is not connecting:
                raise ConnectionAbortedError("the node supervisor stopped")
            connection = membership.open_node_supervisor_connection(
                store, self._node, self._ranks, socket.gethostname(), _CONNECT_TIMEOUT_S
            )
        except Exception as error:
            # Any error, which the main loop raises again when it takes the connection.
            connecting.set_exception(error)
        else:
            connecting.set_result(connection)

    def _take_connection(self) -> None:
        """Take the connection to the authority once the connect ended."""
        if self._connecting is None or not self._connecting.done():
            return
        connecting, self._connecting = self._connecting, None
        membership = self._membership
        address = os.environ[membership.RENDEZVOUS_ENV]
        try:
            connection = connecting.result()
        except membership.UnsignedValueError as error:
            # This node's shared secret differs from the authority's, and no later connect can fix
            # that.
            self._logger.error(f"Cannot connect to the membership authority at {address}: {error}")
            self._refused_by_authority = True
            return
        except (
            OSError,
            RuntimeError,
            ValueError,
        ) as error:
            # Until the deployment serves, the store may not listen yet. Warn about those failures
            # once per interval.
            now_s = time.monotonic()
            quiet = not self._deployment_started and now_s < self._connect_warning_due_s
            if not quiet:
                self._connect_warning_due_s = now_s + _CONNECT_WARNING_INTERVAL_S
            (self._logger.debug if quiet else self._logger.warning)(
                f"Cannot connect to the membership authority at {address}: {error!r}"
            )
            self._authority_reached = False
            return
        self._authority_reached = True
        connection.sock.setblocking(False)
        self._connection_awaiting_view = connection
        self._view_due_s = time.monotonic() + membership.HELLO_TIMEOUT_S
        self._selector.register(connection.sock, selectors.EVENT_READ, _AUTHORITY)

    def _check_first_view(self, now_s: float) -> None:
        """Close the pending connection if the authority sent no view within the hello timeout.

        After the hello, the authority treats the closed connection as the node's ranks failing.
        """
        if self._connection_awaiting_view is None:
            return
        if now_s < self._view_due_s:
            self._deadlines_s.append(self._view_due_s)
            return
        self._logger.warning(
            "Cannot connect to the membership authority, which sent no view within "
            f"{self._membership.HELLO_TIMEOUT_S:.0f} s"
        )
        self._authority_reached = False
        self._close_connection(self._connection_awaiting_view)

    def _receive_view(self) -> None:
        connection = self._connection_awaiting_view or self._connection
        try:
            data = connection.sock.recv(1 << 16)
        except BlockingIOError:
            return
        except OSError:
            data = None
        if data:
            try:
                views = [
                    self._membership.NodeView.model_validate_json(payload)
                    for payload in connection.lines.decode(data)
                ]
            except ValueError as error:
                self._logger.error(
                    f"Received an invalid line from the membership authority: {error}"
                )
            else:
                if views and self._connection_awaiting_view is not None:
                    self._connection = self._connection_awaiting_view
                    self._connection_awaiting_view = None
                    # The authority gave up the node's ranks when the last connection closed.
                    self._report_given_up()
                for view in views:
                    self._on_view(view)
                return
        # Once the deployment has started, the authority has admitted this supervisor before. It then
        # refuses a reconnect only while it still holds the previous connection, so connect again.
        if (
            data == b""
            and self._connection_awaiting_view is not None
            and not self._deployment_started
        ):
            self._logger.error(
                f"The membership authority refused the supervisor of node {self._node}"
            )
            self._refused_by_authority = True
        else:
            self._logger.warning("The connection to the membership authority closed")
        self._close_connection(connection)

    def _close_connection(self, connection) -> None:
        self._selector.unregister(connection.sock)
        connection.sock.close()
        self._connection = self._connection_awaiting_view = None

    def _on_view(self, view) -> None:
        if self._view is not None:
            self._note_left_out(self._view.record, view.record)
        self._view = view
        self._view_received_s = time.monotonic()
        if view.record.state not in (
            self._membership.MembershipState.STARTING,
            self._membership.MembershipState.SHUTDOWN,
            self._membership.MembershipState.ABORTED,
        ):
            self._deployment_started = True

    def _note_left_out(self, previous, record) -> None:
        """Note why ``record`` leaves out a replacement that ``previous`` listed as joining."""
        for rank in self._ranks:
            process = self._processes.get(rank)
            if (
                process is not None
                and previous.is_joining(rank)
                and not record.is_active(rank)
                and record.reason
            ):
                process.left_out_reason = record.reason

    def _stop(
        self,
        status: int,
        end_job_reason: Optional[str] = None,
    ) -> int:
        """Stop the node's ranks, and return ``status``.

        With ``end_job_reason``, end the whole job as failed first. Unlink the node's host weights.
        """
        # Before the job ends, since the launcher then kills this process.
        self._unlink_host_weights()
        # A connect in flight then sends no hello.
        self._connecting = None
        if end_job_reason is not None:
            self._end_job(end_job_reason)
        # Every rank not reaped yet, even one whose leader exited, may leave helpers in its group.
        ranks = [p for p in self._processes.values() if p.status is None]
        for process in ranks:
            self._signal_group(process.child.pid, signal.SIGTERM)
        self._wait_for(ranks)
        for process in ranks:
            self._signal_group(process.child.pid, signal.SIGKILL)
        # A rank stopped above may have created a segment after the first pass.
        self._unlink_host_weights()
        return status

    def _wait_for(self, processes: List[_Process]) -> None:
        """Wait up to the stop grace period for ``processes`` to end, without reaping them.

        Until a rank's process is reaped, its process group's ID cannot be reused.
        """
        deadline_s = time.monotonic() + _STOP_GRACE_S
        with selectors.DefaultSelector() as running:
            for process in processes:
                running.register(process.pidfd, selectors.EVENT_READ)
            while running.get_map() and time.monotonic() < deadline_s:
                for key, _ in running.select(max(0.0, deadline_s - time.monotonic())):
                    running.unregister(key.fileobj)

    def _end_job(self, reason: str) -> None:
        """End the whole job as failed on node 0 under ``mpirun``. Elsewhere, do nothing."""
        if self._end_job_fn is None:
            return
        self._logger.error(f"Ending the job: {reason}")
        try:
            self._end_job_fn()
        except OSError as error:
            self._logger.error(f"Cannot end the job: {error!r}")

    def _unlink_host_weights(self) -> None:
        names = [
            name for name in os.listdir(_SHM_DIR) if name.startswith(self._host_weights_prefix)
        ]
        for name in names:
            with contextlib.suppress(FileNotFoundError):
                os.unlink(os.path.join(_SHM_DIR, name))
        if names:
            self._logger.info(
                f"Unlinked the node's {len(names)} shared-memory segments of host weights"
            )


def _local_gpu_count(environ: Dict[str, str]) -> int:
    visible = environ.get("CUDA_VISIBLE_DEVICES")
    if visible is not None:
        return len([device for device in visible.split(",") if device])
    return len(glob.glob("/dev/nvidia[0-9]*"))


def start_session_node_supervisor(n_workers: int, env: Dict[str, str]) -> subprocess.Popen:
    """Start the node supervisor of ranks 1 to ``n_workers - 1`` for a session that runs rank 0.

    Raises:
        RuntimeError: trtllm-llmapi-launch is not on PATH.
    """
    launcher = shutil.which("trtllm-llmapi-launch")
    if launcher is None:
        raise RuntimeError(
            "Wide-EP fault tolerance starts the ranks with trtllm-llmapi-launch, not on PATH"
        )
    return subprocess.Popen(  # nosec B603
        [
            sys.executable,
            _SCRIPT,
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
        # Out of the session's process group, so that a terminal's signal reaches only rank 0, which
        # then shuts the deployment down. The supervisor follows the record that ends it.
        start_new_session=True,
    )


def stop_session_node_supervisor(supervisor: subprocess.Popen, grace_s: Optional[float]) -> None:
    """Stop ``supervisor`` after waiting ``grace_s`` seconds for it to exit, unless that is None."""
    if grace_s is not None:
        with contextlib.suppress(subprocess.TimeoutExpired):
            supervisor.wait(grace_s)
            return
    supervisor.terminate()
    # The supervisor waits one grace period for its ranks after the signal and one more after it sends
    # them SIGTERM. Allow a third for its own exit.
    try:
        supervisor.wait(3 * _STOP_GRACE_S)
    except subprocess.TimeoutExpired:
        supervisor.kill()
        supervisor.wait()


def _exec_rank(supervisor: int, command: List[str]) -> NoReturn:
    """Run a rank's ``command`` in place of this process, to end once ``supervisor`` ends."""
    # The process guard sits next to this script, and this script runs without site packages.
    from _llmapi_process_guard import _arm_parent_death_signal

    _arm_parent_death_signal()
    # The supervisor ended before the parent-death signal was armed.
    if os.getppid() != supervisor:
        os._exit(1)
    os.execvp(command[0], command)  # nosec B606


def main(argv: Optional[List[str]] = None) -> NoReturn:
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
        help="start every rank but 0, which runs in the MpiPoolSession that starts this supervisor",
    )
    parser.add_argument("command", nargs=argparse.REMAINDER, help="-- <command of each rank>")
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("give the command of each rank after --")
    environ = dict(os.environ)
    if (
        "SLURM_PROCID" in environ
        and "OMPI_COMM_WORLD_RANK" not in environ
        and not args.rank_0_in_session
    ):
        parser.error(
            "Wide-EP fault tolerance requires `mpirun --enable-recovery`, not srun, to start "
            "trtllm-llmapi-launch, because Slurm's PMIx plugin cancels the job step when one of its "
            "tasks exits"
        )
    gpus_per_node = args.gpus_per_node or _local_gpu_count(environ)
    if gpus_per_node <= 0:
        parser.error("this process sees no GPU: give --gpus-per-node")
    # This process uses neither MPI nor a GPU, so it skips the library initialization (MPI_Init).
    os.environ.update(
        TRT_LLM_NO_LIB_INIT="1", MPI4PY_RC_INITIALIZE="false", CUDA_VISIBLE_DEVICES=""
    )
    from tensorrt_llm._torch.distributed.membership import (
        NODE_SUPERVISOR_LAUNCH,
        RENDEZVOUS_ENV,
        SHARED_SECRET_ENV,
        parse_shared_secret,
        remove_cpu_affinity_constraints,
        rendezvous_host_port,
    )
    from tensorrt_llm.llmapi._mpirun_job import end_job_function

    local_size = environ.get("OMPI_COMM_WORLD_LOCAL_SIZE", "1")
    if local_size != "1":
        parser.error(
            f"mpirun started {local_size} node supervisors on this node, so start one task per "
            f"node: {NODE_SUPERVISOR_LAUNCH}"
        )
    for name in (SHARED_SECRET_ENV, RENDEZVOUS_ENV):
        if not environ.get(name):
            parser.error(
                f"export the same {name} on every node and start one task per node: "
                f"{NODE_SUPERVISOR_LAUNCH}"
            )
    # Check both values before any rank starts. Every connect with a malformed value fails, and the
    # supervisor would keep retrying.
    try:
        parse_shared_secret(environ[SHARED_SECRET_ENV])
    except ValueError as error:
        parser.error(str(error))
    try:
        rendezvous_host_port(environ[RENDEZVOUS_ENV])
    except ValueError:
        parser.error(f"{RENDEZVOUS_ENV} must be host:port, not {environ[RENDEZVOUS_ENV]!r}")
    end_job = None
    if args.node_rank == 0:
        try:
            end_job = end_job_function(environ, args.nodes)
        except ValueError as error:
            parser.error(str(error))
    remove_cpu_affinity_constraints()
    supervisor = NodeSupervisor(
        args.node_rank,
        args.nodes,
        gpus_per_node,
        command,
        environ,
        rank_0_in_session=args.rank_0_in_session,
        end_job=end_job,
    )
    for signum in (signal.SIGTERM, signal.SIGINT):
        signal.signal(signum, supervisor.on_signal)
    status = supervisor.run()
    # Skip the interpreter shutdown. A connect thread that leaves the store client during the
    # shutdown aborts the process.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(status)


if __name__ == "__main__":
    if sys.argv[1:2] == [_EXEC_RANK]:
        _exec_rank(int(sys.argv[2]), sys.argv[3:])
    # The script's directory must not shadow the modules TensorRT-LLM imports.
    del sys.path[0]
    main()
