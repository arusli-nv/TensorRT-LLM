# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The Wide-EP fault-tolerance node supervisor, with fake ranks and a fake membership authority.

The authority is a real rendezvous store with a listener that speaks the authority's side of a node supervisor's
connection and sends the views a test gives it. The supervisor runs in a thread, or as a process where a test sends
it signals, kills it or restarts it.
"""

import contextlib
import importlib.util
import json
import os
import secrets
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest
from torch.distributed import TCPStore

from tensorrt_llm._torch.distributed.membership import (
    HOST_WEIGHTS_PREFIX_KEY,
    LIVENESS_KEY,
    RENDEZVOUS_ENV,
    SHARED_SECRET_ENV,
    MembershipRecord,
    MembershipState,
    NodeView,
    RendezvousStore,
)

# The tests run supervisors, ranks and launchers as processes: one that hangs fails its test, not the job.
pytestmark = [pytest.mark.cpu_only, pytest.mark.timeout(120)]

_LLMAPI = Path(__file__).parents[3] / "tensorrt_llm" / "llmapi"
_SUPERVISOR = _LLMAPI / "_llmapi_node_supervisor.py"
_spec = importlib.util.spec_from_file_location("_llmapi_node_supervisor", _SUPERVISOR)
supervisor_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(supervisor_module)

NODE, NODES, GPUS = 1, 2, 2  # node 1 runs ranks 2 and 3 of 4
RANKS = (2, 3)
_TIMINGS = dict(_CONNECT_TIMEOUT_S=0.5, _STOP_GRACE_S=2.0, _MIN_BACKOFF_S=0.05)


@pytest.fixture(autouse=True)
def _timings(monkeypatch):
    """The supervisor's timings, short for the tests."""
    for name, value in _TIMINGS.items():
        monkeypatch.setattr(supervisor_module, name, value)


# A rank: records itself in SUPERVISOR_TEST_ROOT as the rank's n-th process (0 for its first, then its replacements
# in turn), then behaves as SUPERVISOR_TEST_MODES says for "<rank>:<n>" ("serves" by default; "records-signals":
# records the first SIGINT or SIGTERM it gets, and exits).
_RANK = r"""
import json, os, signal, sys, time
from pathlib import Path

rank, world_size, _, replacement = os.environ["TLLM_FAULT_TOLERANCE_RANK"].split(",")
root = Path(os.environ["SUPERVISOR_TEST_ROOT"])
started = [int(path.stem.rsplit("-", 1)[1]) for path in root.glob(f"rank-{rank}-*.json")]
incarnation = max(started, default=0) + 1 if replacement == "1" else 0
modes = dict(item.split("=") for item in os.environ.get("SUPERVISOR_TEST_MODES", "").split())
mode = modes.get(f"{rank}:{incarnation}", "serves")
if mode == "records-signals":
    def record(signum, frame):
        (root / f"signals-{rank}-{incarnation}").write_text(signal.Signals(signum).name)
        sys.exit(128 + signum)

    for signum in (signal.SIGINT, signal.SIGTERM):
        signal.signal(signum, record)
(root / f"rank-{rank}-{incarnation}.json").write_text(
    json.dumps({"pid": os.getpid(), "argv": sys.argv, "environ": dict(os.environ)})
)
statuses = {"crashes": 7, "fails": 5, "cannot-start": 78}
if mode in statuses:
    sys.exit(statuses[mode])
time.sleep(600)
"""

# Runs the supervisor as its script does, with the test's timings; its own path stands for the script's, so that a
# restart runs it again.
_SUPERVISOR_SCRIPT = r"""
import importlib.util, sys

spec = importlib.util.spec_from_file_location("node_supervisor", %(supervisor)r)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
module.__file__ = __file__
for name, value in %(timings)r.items():
    setattr(module, name, value)
sys.exit(module.main())
"""


def _record(
    state=MembershipState.COMMITTED, active_mask=0b1111, incarnations=(0, 0, 0, 0), generation=1
):
    return MembershipRecord(
        generation=generation, state=state, active_mask=active_mask, incarnations=incarnations
    )


class FakeAuthority:
    """The membership authority's side of node supervisors' connections.

    The rendezvous address, and once :meth:`start` ran a store there that publishes a listener, which accepts node
    supervisors and sends them the views of :meth:`push`; with ``refuse``, it closes every connection after the hello.
    The listener's thread runs from the construction to :meth:`close`: the fixture's, not a test's.
    """

    def __init__(self, secret: str, refuse: bool = False):
        self._secret = bytes.fromhex(secret)
        self._refuse = refuse
        # Reserved, as ensure_rendezvous does, until the store listens on it.
        self._reserved = socket.socket()
        self._reserved.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._reserved.bind(("127.0.0.1", 0))
        self.address = f"127.0.0.1:{self._reserved.getsockname()[1]}"
        self.store = None
        self.hellos = []
        self.closed = threading.Event()  # a supervisor's connection closed
        self._connections = []
        self._threads = []
        self._lock = threading.Lock()
        self._view = NodeView(record=_record(), nodes=((NODE, "host-1", True),))
        self._listener = socket.create_server(("127.0.0.1", 0))
        self._thread(self._accept)

    def start(self, record=None, nodes=((NODE, "host-1", True),)):
        if record is not None:
            self._view = NodeView(record=record, nodes=nodes)
        port = self._reserved.getsockname()[1]
        self.store = RendezvousStore(
            TCPStore("127.0.0.1", port, is_master=True, wait_for_workers=False), self._secret
        )
        self.store.set(LIVENESS_KEY, f"127.0.0.1:{self._listener.getsockname()[1]}".encode())
        return self

    def _thread(self, target, *args):
        thread = threading.Thread(target=target, args=args, daemon=True)
        thread.start()
        self._threads.append(thread)

    def _accept(self):
        while True:
            try:
                sock, _ = self._listener.accept()
            except OSError:
                return
            nonce = os.urandom(16)
            sock.sendall(nonce.hex().encode() + b"\n")
            hello = b""
            while not hello.endswith(b"\n"):
                data = sock.recv(256)
                if not data:
                    break
                hello += data
            if not hello.endswith(b"\n"):
                sock.close()
                continue
            self.hellos.append(hello.split(b" ", 1)[1].decode().split())
            if self._refuse:
                sock.close()
                continue
            with self._lock:
                self._connections.append(sock)
                sock.sendall(self._view.model_dump_json().encode() + b"\n")
            self._thread(self._watch, sock)

    def _watch(self, sock):
        with contextlib.suppress(OSError):
            while sock.recv(256):
                pass
        self.closed.set()

    def push(self, record, nodes=((NODE, "host-1", True),)):
        with self._lock:
            self._view = NodeView(record=record, nodes=nodes)
            for sock in self._connections:
                with contextlib.suppress(OSError):
                    sock.sendall(self._view.model_dump_json().encode() + b"\n")

    def close(self):
        """Close every socket and join the threads: a shutdown, not a close, wakes a thread blocked on a socket."""
        self._reserved.close()
        with self._lock:
            sockets = [*self._connections, self._listener]
        for sock in sockets:
            with contextlib.suppress(OSError):
                sock.shutdown(socket.SHUT_RDWR)
        for thread in self._threads:
            thread.join(10)
        for sock in sockets:
            sock.close()


@pytest.fixture
def deployment(tmp_path, monkeypatch):
    """The deployment's shared secret and authority (not started), and this test's fake rank.

    Returns the authority and the environment the node's launcher would give the supervisor.
    """
    secret = secrets.token_hex(32)
    monkeypatch.setenv(SHARED_SECRET_ENV, secret)
    authority = FakeAuthority(secret)
    monkeypatch.setenv(RENDEZVOUS_ENV, authority.address)
    (tmp_path / "rank.py").write_text(_RANK)
    environ = {
        **os.environ,
        "SUPERVISOR_TEST_ROOT": str(tmp_path),
        "OMPI_COMM_WORLD_RANK": "1",
        "PMIX_RANK": "1",
    }
    yield authority, environ
    authority.close()


def _read(tmp_path, rank, incarnation):
    path = tmp_path / f"rank-{rank}-{incarnation}.json"
    return json.loads(path.read_text()) if path.exists() else None


def _until(predicate, timeout_s=30.0):
    deadline = time.monotonic() + timeout_s
    while not predicate():
        assert time.monotonic() < deadline, "timed out"
        time.sleep(0.02)


def _alive(pid):
    try:
        with open(f"/proc/{pid}/stat") as stat:
            return stat.read().split(")")[-1].split()[0] != "Z"
    except FileNotFoundError:
        return False


class Running:
    """A node supervisor running in a thread; :attr:`status` once it returned."""

    def __init__(self, tmp_path, environ, node=NODE, **options):
        self.supervisor = supervisor_module.NodeSupervisor(
            node,
            NODES,
            GPUS,
            [sys.executable, str(tmp_path / "rank.py")],
            environ,
            **options,
        )
        self.status = None
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        self.status = self.supervisor.run()

    def wait(self, timeout_s=30.0):
        self._thread.join(timeout_s)
        assert not self._thread.is_alive(), "the supervisor did not end"
        return self.status


def _serving(tmp_path, authority, environ, node=NODE, **options):
    """A supervisor whose ranks started before the authority, as the original ranks do, and that follows it."""
    running = Running(tmp_path, environ, node=node, **options)
    ranks = range(node * GPUS, (node + 1) * GPUS)
    started = [rank for rank in ranks if rank != 0 or not options.get("rank_0_in_session")]
    _until(lambda: all(_read(tmp_path, rank, 0) for rank in started))
    authority.start()
    _until(lambda: authority.hellos and running.supervisor._view is not None)
    return running


def _push_and_wait_for_step(running, authority, record, nodes=((NODE, "host-1", True),)):
    """Push a view the supervisor does not have yet, and wait until it stepped on it and on every earlier event.

    The supervisor steps between two waits for events, so once it took a view pushed after it took this one, it has
    stepped on this one.
    """
    authority.push(record, nodes)
    _until(lambda: running.supervisor._view == NodeView(record=record, nodes=nodes))
    following = record.model_copy(update={"generation": record.generation + 1})
    authority.push(following, nodes)
    _until(lambda: running.supervisor._view.record == following)


def test_the_ranks_start_on_their_gpus_outside_the_launchers_job(tmp_path, deployment):
    authority, environ = deployment
    running = _serving(tmp_path, authority, environ)
    for rank in RANKS:
        started = _read(tmp_path, rank, 0)
        assert started["environ"]["TLLM_FAULT_TOLERANCE_RANK"] == f"{rank},4,2,0"  # on GPU rank % 2
        assert not any(
            key.startswith(("OMPI_", "PMIX_"))
            for key in started["environ"]
            if key != "OMPI_MCA_btl"
        )
        assert started["environ"]["OMPI_MCA_btl"] == "^sm"  # no /dev/shm segment to leave behind
        assert started["environ"][SHARED_SECRET_ENV] == os.environ[SHARED_SECRET_ENV]
        assert os.environ[SHARED_SECRET_ENV] not in " ".join(started["argv"])
    assert authority.hellos == [["node", "1", "2,3", socket.gethostname()]]
    authority.push(_record(MembershipState.SHUTDOWN))
    assert running.wait() == 0


def test_a_crashed_rank_is_replaced_on_its_gpu_once_the_record_lists_it_as_failed(
    tmp_path, deployment
):
    authority, environ = deployment
    running = _serving(tmp_path, authority, environ)
    os.kill(_read(tmp_path, 3, 0)["pid"], signal.SIGKILL)
    _until(lambda: running.supervisor._processes[3].status is not None)
    # The record still lists it: the authority has not decided yet.
    _push_and_wait_for_step(running, authority, _record(generation=2))
    assert not running.supervisor._processes[3].replacement
    authority.push(_record(MembershipState.PROPOSE, active_mask=0b0111))
    _until(lambda: _read(tmp_path, 3, 1))
    replacement = _read(tmp_path, 3, 1)
    assert replacement["environ"]["TLLM_FAULT_TOLERANCE_RANK"] == "3,4,2,1"
    assert _alive(_read(tmp_path, 2, 0)["pid"])
    authority.push(_record(MembershipState.SHUTDOWN, incarnations=(0, 0, 0, 1)))
    assert running.wait() == 0
    assert not _alive(replacement["pid"])


def test_a_rank_the_authority_excludes_while_it_runs_is_killed_and_replaced(tmp_path, deployment):
    """Excluded while hung: the supervisor kills it, then starts its replacement on the same GPU."""
    authority, environ = deployment
    running = _serving(tmp_path, authority, environ)
    hung = _read(tmp_path, 3, 0)["pid"]
    authority.push(_record(MembershipState.PROPOSE, active_mask=0b0111))
    _until(lambda: _read(tmp_path, 3, 1))
    assert not _alive(hung)
    authority.push(_record(MembershipState.ABORTED, active_mask=0b0111))
    assert running.wait() == 1


def test_a_replacement_that_ends_before_its_admission_is_started_again_unless_it_cannot_start(
    tmp_path, deployment, monkeypatch
):
    """Retried after a backoff, until one exits with REPLACEMENT_CANNOT_START."""
    monkeypatch.setattr(supervisor_module, "_MIN_BACKOFF_S", 1.0)
    authority, environ = deployment
    environ["SUPERVISOR_TEST_MODES"] = "3:1=fails 3:2=cannot-start"
    running = _serving(tmp_path, authority, environ)
    os.kill(_read(tmp_path, 3, 0)["pid"], signal.SIGKILL)
    authority.push(_record(MembershipState.COMMITTED, active_mask=0b0111))
    _until(lambda: _read(tmp_path, 3, 2))
    _until(lambda: running.supervisor._given_up == {3})
    _push_and_wait_for_step(
        running, authority, _record(MembershipState.COMMITTED, active_mask=0b0111, generation=2)
    )
    # No replacement runs, as another one would fail alike.
    assert running.supervisor._processes[3].status is not None
    assert running.supervisor._given_up == {3}
    authority.push(_record(MembershipState.SHUTDOWN, active_mask=0b0111))
    assert running.wait() == 0


def test_a_replacement_the_change_that_would_add_it_leaves_out_is_killed_and_started_again(
    tmp_path, deployment
):
    """Listed by a PROPOSE, then left out by the next one: never admitted, yet killed and started again.

    Fails if only an admitted process were killed.
    """
    authority, environ = deployment
    running = _serving(tmp_path, authority, environ)
    os.kill(_read(tmp_path, 3, 0)["pid"], signal.SIGKILL)
    authority.push(_record(MembershipState.PROPOSE, active_mask=0b0111))
    _until(lambda: _read(tmp_path, 3, 1))
    joining = _read(tmp_path, 3, 1)["pid"]
    joined = (0, 0, 0, 1)
    authority.push(_record(MembershipState.PROPOSE, 0b1111, incarnations=joined, generation=2))
    _until(lambda: running.supervisor._processes[3].listed)
    assert running.supervisor._processes[3].incarnation == 1  # the one the authority assigned it
    authority.push(_record(MembershipState.PROPOSE, 0b0111, incarnations=joined, generation=3))
    _until(lambda: not _alive(joining))
    _until(lambda: _read(tmp_path, 3, 2))
    authority.push(_record(MembershipState.SHUTDOWN, active_mask=0b0111, generation=3))
    assert running.wait() == 0


def test_each_rank_runs_in_its_own_session(tmp_path, deployment):
    """A signal for the terminal reaches the supervisor alone (fails without start_new_session)."""
    authority, environ = deployment
    running = _serving(tmp_path, authority, environ)
    for rank in RANKS:
        pid = _read(tmp_path, rank, 0)["pid"]
        assert os.getsid(pid) == pid
    authority.push(_record(MembershipState.SHUTDOWN))
    assert running.wait() == 0


def test_a_rank_that_fails_before_the_deployment_serves_ends_the_node(tmp_path, deployment):
    """No authority yet: as without fault tolerance, the node fails with the rank's status."""
    _, environ = deployment
    environ["SUPERVISOR_TEST_MODES"] = "2:0=crashes"
    running = Running(tmp_path, environ)
    assert running.wait() == 7
    (other,) = [p.child for rank, p in running.supervisor._processes.items() if rank == 3]
    assert other.poll() is not None  # stopped with the node


def _fake_tool(directory, name, root):
    """A fake launcher tool ``name`` in ``directory``: records its arguments in ROOT/<name>-args and exits 0."""
    directory.mkdir(exist_ok=True)
    tool = directory / name
    tool.write_text(f'#!/bin/bash\nprintf "%s\\n" "$@" > "{root / f"{name}-args"}"\n')
    tool.chmod(0o755)
    return tool


@pytest.mark.parametrize("failed, served", [(0, False), (1, False), (0, True)])
def test_rank_0_s_end_ends_the_job(tmp_path, deployment, failed, served):
    """Node 0's supervisor ends the job through its launcher at once when rank 0 ends.

    No authority may be left, or started yet, to end the other nodes' ranks. Another rank's failure before the
    deployment serves ends its node only; the authority aborts the rest. Fails without the change: the supervisor
    took no way to end the job.
    """
    authority, environ = deployment
    ended = []

    def end_job(environ, timeout_s):
        ended.append(timeout_s)

    if served:
        running = _serving(tmp_path, authority, environ, node=0, end_job=end_job)
        os.kill(_read(tmp_path, 0, 0)["pid"], signal.SIGKILL)
        assert running.wait() == 128 + signal.SIGKILL
    else:
        environ["SUPERVISOR_TEST_MODES"] = f"{failed}:0=crashes"
        running = Running(tmp_path, environ, node=0, end_job=end_job)
        assert running.wait() == 7
    assert ended == ([_TIMINGS["_STOP_GRACE_S"]] if failed == 0 else [])


def test_rank_0_is_never_replaced_and_its_end_ends_the_deployment(tmp_path, deployment):
    authority, environ = deployment
    running = _serving(tmp_path, authority, environ, node=0)
    os.kill(_read(tmp_path, 0, 0)["pid"], signal.SIGKILL)
    assert running.wait() == 128 + signal.SIGKILL
    assert _read(tmp_path, 0, 1) is None
    assert not _alive(_read(tmp_path, 1, 0)["pid"])


def test_rank_0_in_the_session_is_started_by_the_session(tmp_path, deployment):
    """As ``MpiPoolSession`` runs rank 0 in its pool, the supervisor starts only the other ranks."""
    authority, environ = deployment
    running = _serving(tmp_path, authority, environ, node=0, rank_0_in_session=True)
    assert _read(tmp_path, 0, 0) is None
    assert list(running.supervisor._processes) == [1]
    authority.push(_record(MembershipState.SHUTDOWN))
    assert running.wait() == 0


def test_a_supervisor_given_another_deployment_s_shared_secret_ends(
    tmp_path, deployment, monkeypatch
):
    """The store's values are not signed with its secret: it ends with status 1 and starts no rank.

    Fails if it tried to connect again and again instead.
    """
    authority, environ = deployment
    authority.start()
    monkeypatch.setenv(SHARED_SECRET_ENV, secrets.token_hex(32))
    supervisor = supervisor_module.NodeSupervisor(
        NODE, NODES, GPUS, [sys.executable, str(tmp_path / "rank.py")], environ
    )
    assert supervisor.run() == 1
    assert supervisor._processes == {}


def test_a_node_has_one_supervisor_of_a_deployment(tmp_path, deployment):
    authority, environ = deployment
    running = _serving(tmp_path, authority, environ)
    second = supervisor_module.NodeSupervisor(
        NODE, NODES, GPUS, [sys.executable, str(tmp_path / "rank.py")], environ
    )
    assert second.run() == 1
    authority.push(_record(MembershipState.SHUTDOWN))
    assert running.wait() == 0


def test_a_supervisor_the_authority_refuses_ends(tmp_path, monkeypatch):
    """Another one supervises its node or one of its ranks."""
    secret = secrets.token_hex(32)
    monkeypatch.setenv(SHARED_SECRET_ENV, secret)
    authority = FakeAuthority(secret, refuse=True).start()
    monkeypatch.setenv(RENDEZVOUS_ENV, authority.address)
    try:
        supervisor = supervisor_module.NodeSupervisor(
            NODE,
            NODES,
            GPUS,
            [sys.executable, "-c", "raise SystemExit(9)"],
            dict(os.environ),
        )
        assert supervisor.run() == 1
        assert supervisor._processes == {}  # no rank started
    finally:
        authority.close()


def test_a_supervisor_started_while_the_deployment_serves_replaces_its_failed_ranks(
    tmp_path, deployment
):
    """The node's launch started again after the node failed: no original rank, a replacement for each rank."""
    authority, environ = deployment
    authority.start(_record(MembershipState.COMMITTED, active_mask=0b0011))
    running = Running(tmp_path, environ)
    _until(lambda: _read(tmp_path, 2, 1) and _read(tmp_path, 3, 1))
    assert _read(tmp_path, 2, 0) is None and _read(tmp_path, 3, 0) is None
    authority.push(_record(MembershipState.SHUTDOWN, active_mask=0b0011))
    assert running.wait() == 0


def test_a_supervisor_that_connects_while_the_deployment_starts_starts_its_ranks(
    tmp_path, deployment
):
    """A supervisor that first sees the record STARTING starts its ranks as originals.

    The authority runs from rank 0's worker entry, so it may answer before every node's supervisor connected: node 0
    relaunches no node whose supervisor has not connected yet.
    """
    authority, environ = deployment
    prun = tmp_path / "prun"
    prun.write_text(_FAKE_PRUN % {"root": str(tmp_path)})
    prun.chmod(0o755)
    authority.start(_record(MembershipState.STARTING), ((0, "host-0", True),))
    running = Running(tmp_path, environ, node=0, relaunch_command=[str(prun), "--dvm-uri", "uri"])
    _until(lambda: _read(tmp_path, 0, 0) and _read(tmp_path, 1, 0))
    _push_and_wait_for_step(
        running,
        authority,
        _record(MembershipState.STARTING),
        ((0, "host-0", True), (1, "host-1", False)),
    )
    assert not running.supervisor._relaunched
    assert not running.supervisor._processes[1].replacement
    authority.push(_record(MembershipState.SHUTDOWN))
    assert running.wait() == 0


@pytest.mark.parametrize(
    "environ, options, rejected",
    [
        ({}, [], False),
        # Started as a Slurm task: Slurm's PMIx plugin cancels the step when one exits.
        ({"SLURM_JOB_ID": "1", "SLURM_PROCID": "1"}, [], True),
        # mpirun inside the allocation; its --enable-recovery is not visible here: nothing to check.
        ({"SLURM_JOB_ID": "1", "SLURM_PROCID": "0", "OMPI_COMM_WORLD_RANK": "1"}, [], False),
        # MpiPoolSession's, inside a Slurm task that is not a node's launch.
        ({"SLURM_JOB_ID": "1", "SLURM_PROCID": "0"}, ["--rank-0-in-session"], False),
    ],
)
def test_a_supervisor_started_as_a_slurm_task_is_rejected(
    monkeypatch, capfd, environ, options, rejected
):
    for name in ("SLURM_JOB_ID", "SLURM_PROCID", "OMPI_COMM_WORLD_RANK"):
        monkeypatch.delenv(name, raising=False)
    for name, value in environ.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")  # past the check, main stops at the GPU count
    with pytest.raises(SystemExit):
        supervisor_module.main(["--node-rank", "1", "--nodes", "2", *options, "--", "true"])
    error = capfd.readouterr().err
    assert ("Slurm tasks (srun)" in error) == rejected
    assert ("sees no GPU" in error) != rejected


def _supervisor_script(tmp_path):
    script = tmp_path / "supervisor.py"
    script.write_text(_SUPERVISOR_SCRIPT % {"supervisor": str(_SUPERVISOR), "timings": _TIMINGS})
    return script


def _supervisor_process(tmp_path, environ, *arguments, **options):
    script = _supervisor_script(tmp_path)
    return subprocess.Popen([sys.executable, str(script), *arguments], env=environ, **options)


_NODE_OPTIONS = ["--node-rank", str(NODE), "--nodes", str(NODES), "--gpus-per-node", str(GPUS)]


def _rank_command(tmp_path):
    return ["--", sys.executable, str(tmp_path / "rank.py")]


@pytest.mark.parametrize("launcher", ["prterun", "sleep"])
def test_under_mpirun_node_0_ends_the_job_through_the_job_s_prrte(tmp_path, deployment, launcher):
    """Rank 0's end sends SIGTERM to the mpirun whose PID the job's PMIx server URI holds, else runs pterm.

    SIGTERM makes mpirun (prterun) exit with a failure status; pterm, used only where that PID is no mpirun of this
    host, makes it exit with status 0. Here a shell that outlives it starts it, as a wrapper script between mpirun
    and the launch does: the shell is not signalled. Fails if the supervisor ran pterm alone (mpirun then exits with
    status 0), or signalled a process other than mpirun.
    """
    _, environ = deployment
    bin_dir = tmp_path / "bin"
    for tool in ("prun", "pterm"):
        _fake_tool(bin_dir, tool, tmp_path)
    (bin_dir / launcher).symlink_to(shutil.which("sleep"))
    mpirun = subprocess.Popen([str(bin_dir / launcher), "60"])
    uri = f"prterun-{socket.gethostname()}-{mpirun.pid}@0.0;tcp4://127.0.0.1:1"
    environ.update(
        PATH=f"{bin_dir}:{environ['PATH']}",
        PMIX_SERVER_URI41=uri,
        SUPERVISOR_TEST_MODES="0:0=crashes",
    )
    node_0 = ["--node-rank", "0", "--nodes", str(NODES), "--gpus-per-node", str(GPUS)]
    shell = subprocess.Popen(
        [
            "bash",
            "-c",
            '"$@"; echo "status $?" > "$0"',
            str(tmp_path / "shell-status"),
            sys.executable,
            str(_supervisor_script(tmp_path)),
            *node_0,
            *_rank_command(tmp_path),
        ],
        env=environ,
    )
    try:
        assert shell.wait(30) == 0
        if launcher == "prterun":
            assert mpirun.wait(10) == -signal.SIGTERM
        else:
            assert mpirun.poll() is None
    finally:
        shell.kill()
        mpirun.kill()
        mpirun.wait()
    assert (tmp_path / "shell-status").read_text() == "status 7\n"
    assert (tmp_path / "pterm-args").exists() == (launcher != "prterun")
    if launcher != "prterun":
        assert (tmp_path / "pterm-args").read_text().split("\n")[:2] == ["--dvm-uri", uri]
    assert not (tmp_path / "prun-args").exists()


def test_the_ranks_die_with_their_supervisor(tmp_path, deployment):
    """Kill -9 of the supervisor is the loss of its node: its ranks end, and its authority connection closes."""
    authority, environ = deployment
    supervisor = _supervisor_process(tmp_path, environ, *_NODE_OPTIONS, *_rank_command(tmp_path))
    try:
        _until(lambda: all(_read(tmp_path, rank, 0) for rank in RANKS))
        authority.start()
        _until(lambda: authority.hellos)
        supervisor.kill()
        supervisor.wait(10)
        _until(lambda: not any(_alive(_read(tmp_path, rank, 0)["pid"]) for rank in RANKS), 10)
        assert authority.closed.wait(10)
    finally:
        supervisor.kill()


@pytest.mark.parametrize("end", ["SIGINT", "SIGTERM", "ABORT", "SHUTDOWN"])
def test_the_node_s_host_weights_are_unlinked_when_the_deployment_ends(tmp_path, deployment, end):
    """However the ranks end, also when the supervisor itself is signalled; another deployment's stay."""
    authority, environ = deployment
    prefix = f"moe_shared_test{os.getpid()}{secrets.token_hex(4)}_"
    segments = [Path("/dev/shm", f"{prefix}l0_lr{rank}_all") for rank in RANKS]
    other = Path("/dev/shm", f"moe_shared_test{os.getpid()}{secrets.token_hex(4)}_l0_lr2_all")
    for segment in [*segments, other]:
        segment.write_bytes(b"host weights")
    supervisor = _supervisor_process(tmp_path, environ, *_NODE_OPTIONS, *_rank_command(tmp_path))
    try:
        _until(lambda: all(_read(tmp_path, rank, 0) for rank in RANKS))
        authority.start()
        authority.store.set(HOST_WEIGHTS_PREFIX_KEY, prefix.encode())
        _until(lambda: authority.hellos)
        authority.push(_record())  # the supervisor reads the prefix with a view
        time.sleep(0.5)
        if end.startswith("SIG"):
            supervisor.send_signal(getattr(signal, end))
        else:
            authority.push(_record(MembershipState[end]))
        status = supervisor.wait(30)
        assert status == {"SIGINT": 130, "SIGTERM": 143, "ABORT": 1, "SHUTDOWN": 0}[end]
        assert not any(segment.exists() for segment in segments)
        assert other.exists()
        assert not any(_alive(_read(tmp_path, rank, 0)["pid"]) for rank in RANKS)
    finally:
        supervisor.kill()
        for segment in [*segments, other]:
            with contextlib.suppress(FileNotFoundError):
                segment.unlink()


def test_the_host_weights_are_unlinked_before_the_ranks_end(tmp_path, deployment, monkeypatch):
    """After SHUTDOWN the segments go first, while the ranks get the grace to end by themselves.

    So a supervisor killed during the grace leaves none behind; fails if they were unlinked once the ranks ended.
    """
    authority, environ = deployment
    prefix = f"moe_shared_test{os.getpid()}{secrets.token_hex(4)}_"
    segments = [Path("/dev/shm", f"{prefix}l0_lr{rank}_all") for rank in RANKS]
    for segment in segments:
        segment.write_bytes(b"host weights")
    try:
        monkeypatch.setattr(supervisor_module, "_STOP_GRACE_S", 5.0)
        running = _serving(tmp_path, authority, environ)
        authority.store.set(HOST_WEIGHTS_PREFIX_KEY, prefix.encode())
        authority.push(_record())  # the supervisor reads the prefix with a view
        _until(lambda: running.supervisor._host_weights_prefix == prefix)
        authority.push(_record(MembershipState.SHUTDOWN))
        _until(lambda: not any(segment.exists() for segment in segments), 3.0)
        assert all(_alive(_read(tmp_path, rank, 0)["pid"]) for rank in RANKS)  # within the grace
        assert running.wait() == 0
    finally:
        for segment in segments:
            with contextlib.suppress(FileNotFoundError):
                segment.unlink()


def test_a_signal_for_the_supervisor_goes_to_rank_0_alone(tmp_path, deployment):
    """Rank 0's process, which runs the deployment's frontend, gets the signal; the others SIGTERM after the grace.

    Fails if the supervisor forwarded the signal to every rank at once.
    """
    authority, environ = deployment
    environ["SUPERVISOR_TEST_MODES"] = "0:0=records-signals 1:0=records-signals"
    node_0 = ["--node-rank", "0", "--nodes", str(NODES), "--gpus-per-node", str(GPUS)]
    supervisor = _supervisor_process(tmp_path, environ, *node_0, *_rank_command(tmp_path))
    try:
        _until(lambda: _read(tmp_path, 0, 0) and _read(tmp_path, 1, 0))
        authority.start()
        _until(lambda: authority.hellos)
        supervisor.send_signal(signal.SIGINT)
        _until(lambda: (tmp_path / "signals-0-0").exists())
        assert (tmp_path / "signals-0-0").read_text() == "SIGINT"
        assert _alive(_read(tmp_path, 1, 0)["pid"])
        assert not (tmp_path / "signals-1-0").exists()
        assert supervisor.wait(30) == 130
        assert (tmp_path / "signals-1-0").read_text() == "SIGTERM"
    finally:
        supervisor.kill()


# A fake prun: records its process ID, arguments and standard input, then runs until ROOT/prun-exit appears.
_FAKE_PRUN = r"""#!/usr/bin/env python3
import os, sys, time

root = %(root)r
calls = len([name for name in os.listdir(root) if name.startswith("prun-") and name.endswith(".args")])
with open(os.path.join(root, f"prun-{calls}.pid"), "w") as pid:
    pid.write(str(os.getpid()))
with open(os.path.join(root, f"prun-{calls}.args"), "w") as args:
    args.write("\n".join(sys.argv[1:]))
with open(os.path.join(root, f"prun-{calls}.stdin"), "w") as stdin:
    stdin.write(sys.stdin.read())
while not os.path.exists(os.path.join(root, "prun-exit")):
    time.sleep(0.05)
"""


def test_node_0_relaunches_a_failed_node_once_through_prun(tmp_path, deployment):
    """Node 0 relaunches a node whose supervisor left, through the job's PRRTE.

    The deployment's environment goes on standard input, never on the command line; once, while that one runs or
    once it connected.
    """
    authority, environ = deployment
    prun = tmp_path / "prun"
    prun.write_text(_FAKE_PRUN % {"root": str(tmp_path)})
    prun.chmod(0o755)
    nodes = ((0, "host-0", True), (1, "host-1", True))
    node_1_failed = ((0, "host-0", True), (1, "host-1", False))
    running = _serving(
        tmp_path, authority, environ, node=0, relaunch_command=[str(prun), "--dvm-uri", "uri"]
    )
    _push_and_wait_for_step(running, authority, _record(), nodes)
    assert not running.supervisor._relaunched
    for _ in range(10):  # the record comes again, also changed
        authority.push(_record(MembershipState.PROPOSE, active_mask=0b0011), node_1_failed)
        time.sleep(0.05)
    _until(lambda: (tmp_path / "prun-0.stdin").exists() and (tmp_path / "prun-0.stdin").read_text())
    args = (tmp_path / "prun-0.args").read_text().split("\n")
    assert args == [
        "--dvm-uri",
        "uri",
        "--host",
        "host-1:1",
        sys.executable,
        os.path.abspath(supervisor_module.__file__),
        "--node-rank",
        "1",
        "--nodes",
        "2",
        "--gpus-per-node",
        "2",
        "--environment-from-stdin",
        "--",
        sys.executable,
        str(tmp_path / "rank.py"),
    ]
    secret = os.environ[SHARED_SECRET_ENV]
    assert secret not in " ".join(args)
    relaunched_environ = json.loads((tmp_path / "prun-0.stdin").read_text())
    assert relaunched_environ[SHARED_SECRET_ENV] == secret
    assert not any(
        key.startswith(("OMPI_", "PMIX_")) for key in relaunched_environ if key != "OMPI_MCA_btl"
    )
    assert relaunched_environ["OMPI_MCA_btl"] == "^sm"
    connected = _record(MembershipState.COMMITTED, active_mask=0b0011)
    _push_and_wait_for_step(running, authority, connected, nodes)
    # Still the first relaunch, which a later one would have replaced.
    assert running.supervisor._relaunched[1].pid == int((tmp_path / "prun-0.pid").read_text())
    authority.push(_record(MembershipState.SHUTDOWN, active_mask=0b0011), nodes)
    assert running.wait() == 0


def test_a_relaunched_supervisor_takes_its_environment_from_standard_input(tmp_path, deployment):
    """Prun forwards no value: the environment, the shared secret included, comes in one JSON line."""
    authority, environ = deployment
    bare = {"PATH": os.environ["PATH"], "PYTHONPATH": os.environ.get("PYTHONPATH", "")}
    supervisor = _supervisor_process(
        tmp_path,
        bare,
        *_NODE_OPTIONS,
        "--environment-from-stdin",
        *_rank_command(tmp_path),
        stdin=subprocess.PIPE,
    )
    try:
        supervisor.stdin.write(json.dumps(environ).encode() + b"\n")
        supervisor.stdin.close()
        _until(lambda: _read(tmp_path, 2, 0))
        assert _read(tmp_path, 2, 0)["environ"][SHARED_SECRET_ENV] == os.environ[SHARED_SECRET_ENV]
        authority.start()
        _until(lambda: authority.hellos)
        authority.push(_record(MembershipState.SHUTDOWN))
        assert supervisor.wait(30) == 0
    finally:
        supervisor.kill()


def _run_launcher(tmp_path, procid, task, **extra):
    """Run trtllm-llmapi-launch as task ``procid`` of 2.

    Its python3 records how a node supervisor or a worker is started, in ``python3-args``, instead of starting it.
    """
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    record = tmp_path / "python3-args"
    fake_python = bin_dir / "python3"
    fake_python.write_text(
        "#!/bin/bash\n"
        'if [ "$1" = "-m" ] || [[ "$1" == *_llmapi_node_supervisor.py ]]; then\n'
        f'    printf "%s\\n" "$@" > "{record}"\n'
        "    exit 0\n"
        "fi\n"
        f'exec "{sys.executable}" "$@"\n'
    )
    fake_python.chmod(0o755)
    environ = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith(("OMPI_", "PMI", "SLURM_", "TLLM_FAULT_TOLERANCE_"))
    }
    environ.update(
        PATH=f"{bin_dir}:{environ['PATH']}",
        HOME=str(tmp_path),
        SLURM_PROCID=str(procid),
        SLURM_NTASKS="2",
        TRTLLM_FLASHINFER_WORKSPACE_PER_PROCESS="0",
        **extra,
    )
    return subprocess.run(
        ["bash", str(_LLMAPI / "trtllm-llmapi-launch"), *task],
        env=environ,
        capture_output=True,
        text=True,
        timeout=60,
    )


@pytest.mark.parametrize("fault_tolerance", [False, True])
def test_under_fault_tolerance_the_launch_of_a_node_runs_its_supervisor(tmp_path, fault_tolerance):
    """The exported shared secret is all the launcher sees of fault tolerance.

    Its launch on node 1 of 2 runs the node supervisor, which runs this launcher for each rank of the node.
    """
    extra = {"TLLM_FAULT_TOLERANCE_SHARED_SECRET": secrets.token_hex(32)} if fault_tolerance else {}
    result = _run_launcher(tmp_path, 1, ["task"], **extra)
    assert result.returncode == 0, result.stderr
    args = (tmp_path / "python3-args").read_text().split()
    if fault_tolerance:
        assert Path(args[0]).resolve() == _SUPERVISOR.resolve()
        assert args[1:6] == ["--node-rank", "1", "--nodes", "2", "--"]
        assert Path(args[6]).resolve() == (_LLMAPI / "trtllm-llmapi-launch").resolve()
        assert args[7:] == ["task"]
    else:
        assert args == ["-m", "tensorrt_llm.llmapi.mgmn_worker_node"]


def test_under_mpirun_in_a_slurm_allocation_a_node_s_launch_takes_mpirun_s_rank(tmp_path):
    """The node's rank and the node count come from mpirun's variables, not the batch step's SLURM_*.

    mpirun leaves the batch step's SLURM_PROCID (0) and SLURM_NTASKS in every process it starts; fails if SLURM_*
    were read first (every node would be node 0).
    """
    result = _run_launcher(
        tmp_path,
        0,
        ["task"],
        TLLM_FAULT_TOLERANCE_SHARED_SECRET=secrets.token_hex(32),
        OMPI_COMM_WORLD_RANK="1",
        OMPI_COMM_WORLD_SIZE="3",
    )
    assert result.returncode == 0, result.stderr
    args = (tmp_path / "python3-args").read_text().split()
    assert Path(args[0]).resolve() == _SUPERVISOR.resolve()
    assert args[1:6] == ["--node-rank", "1", "--nodes", "3", "--"]


def test_a_rank_the_node_supervisor_starts_takes_its_rank_from_its_identity(tmp_path):
    """Not from the launcher's variables.

    The worker takes the launcher's place, so that it dies with the supervisor.
    """
    result = _run_launcher(
        tmp_path,
        0,
        ["task"],
        TLLM_FAULT_TOLERANCE_SHARED_SECRET=secrets.token_hex(32),
        TLLM_FAULT_TOLERANCE_RANK="3,4,2,0",
    )
    assert result.returncode == 0, result.stderr
    assert "mpi_rank: 3" in result.stderr
    assert "with mpi_world_size: 4" in result.stderr
    args = (tmp_path / "python3-args").read_text().split()
    assert args == ["-m", "tensorrt_llm._torch.pyexecutor.rank_failure_recovery"]
    assert "MGMN worker node exit code" not in result.stderr  # exec'd


@contextlib.contextmanager
def _unanswered_port():
    """A local port that refuses connections: bound but not listening."""
    with socket.socket() as held:
        held.bind(("127.0.0.1", 0))
        yield held.getsockname()[1]


@pytest.mark.parametrize("fault_tolerance", [False, True])
def test_rank_0_s_server_ending_before_its_task_fails_only_without_fault_tolerance(
    tmp_path, fault_tolerance
):
    """A clean shutdown under fault tolerance exits 0.

    Rank 0's worker ends its MPI server once the deployment shut down, before the task (the server's frontend) exits.
    """
    with _unanswered_port() as port:
        extra = {}
        if fault_tolerance:
            extra = {
                "TLLM_FAULT_TOLERANCE_SHARED_SECRET": secrets.token_hex(32),
                "TLLM_FAULT_TOLERANCE_RENDEZVOUS": f"127.0.0.1:{port}",
                "TLLM_FAULT_TOLERANCE_RANK": "0,4,2,0",
            }
        result = _run_launcher(tmp_path, 0, ["bash", "-c", "sleep 1"], **extra)
    assert "MPI Comm server exited before the task (status 0)" in result.stderr
    assert result.returncode == (0 if fault_tolerance else 1), result.stderr
