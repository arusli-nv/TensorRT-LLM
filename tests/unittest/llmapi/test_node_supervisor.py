# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the Wide-EP fault-tolerance node supervisor with fake ranks and a fake membership authority.

The fake authority is a real rendezvous store plus a listener that sends the supervisor the views a test gives it.
The supervisor runs in a thread, or in a process when a test signals, kills or restarts it.
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

# The tests run supervisors, ranks and launchers as processes. A hung process fails its test, not the job.
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
    """Shorten the supervisor's timings for the tests."""
    for name, value in _TIMINGS.items():
        monkeypatch.setattr(supervisor_module, name, value)


# A fake rank. It records itself in SUPERVISOR_TEST_ROOT as the rank's n-th process, where 0 is the original and
# each replacement counts up. SUPERVISOR_TEST_MODES maps "<rank>:<n>" to its behavior, "serves" by default. In
# "records-signals" mode it records the first SIGINT or SIGTERM it gets and exits.
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

# Run the supervisor as its script does, with the test's timings. The wrapper's path replaces the script's path so
# that a restart runs the wrapper again.
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
    """Fake membership authority that node supervisors connect to.

    :meth:`start` opens a rendezvous store that publishes the listener's address. The listener accepts node
    supervisors and sends them the views given to :meth:`push`. With ``refuse``, it closes every connection after
    the hello. The listener thread runs until :meth:`close`.
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
        """Close every socket and join the threads.

        Only a shutdown, not a close, wakes a thread blocked on a socket.
        """
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
    """Set up a shared secret, an authority that is not started yet, and the fake rank script.

    Yields the authority and the environment the node's launcher would give the supervisor.
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
    """Run a node supervisor in a thread and keep its exit status in :attr:`status`."""

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
    """Start a supervisor whose ranks start before the authority, as the original ranks do."""
    running = Running(tmp_path, environ, node=node, **options)
    ranks = range(node * GPUS, (node + 1) * GPUS)
    started = [rank for rank in ranks if rank != 0 or not options.get("rank_0_in_session")]
    _until(lambda: all(_read(tmp_path, rank, 0) for rank in started))
    authority.start()
    _until(lambda: authority.hellos and running.supervisor._view is not None)
    return running


def _push_and_wait_for_step(running, authority, record, nodes=((NODE, "host-1", True),)):
    """Push a new view and wait until the supervisor has acted on it and on every earlier event.

    The supervisor acts on each view before it waits for the next event. Once it has taken a later view, it has
    acted on this one.
    """
    authority.push(record, nodes)
    _until(lambda: running.supervisor._view == NodeView(record=record, nodes=nodes))
    following = record.model_copy(update={"generation": record.generation + 1})
    authority.push(following, nodes)
    _until(lambda: running.supervisor._view.record == following)


@pytest.mark.parametrize("btl", [None, "self,tcp"], ids=["default", "user-set"])
def test_ranks_start_outside_launcher_job(tmp_path, deployment, btl):
    authority, environ = deployment
    if btl is not None:
        environ = {**environ, "OMPI_MCA_btl": btl}
    running = _serving(tmp_path, authority, environ)
    for rank in RANKS:
        started = _read(tmp_path, rank, 0)
        assert started["environ"]["TLLM_FAULT_TOLERANCE_RANK"] == f"{rank},4,2,0"  # on GPU rank % 2
        assert not any(
            key.startswith(("OMPI_", "PMIX_"))
            for key in started["environ"]
            if key != "OMPI_MCA_btl"
        )
        # Without the user's setting, no /dev/shm segment to leave behind.
        assert started["environ"]["OMPI_MCA_btl"] == (btl or "^sm")
        assert started["environ"][SHARED_SECRET_ENV] == os.environ[SHARED_SECRET_ENV]
        assert os.environ[SHARED_SECRET_ENV] not in " ".join(started["argv"])
    assert authority.hellos == [["node", "1", "2,3", socket.gethostname()]]
    authority.push(_record(MembershipState.SHUTDOWN))
    assert running.wait() == 0


def test_crashed_rank_replaced_once_failed(tmp_path, deployment):
    authority, environ = deployment
    running = _serving(tmp_path, authority, environ)
    os.kill(_read(tmp_path, 3, 0)["pid"], signal.SIGKILL)
    _until(lambda: running.supervisor._processes[3].status is not None)
    # The authority has not removed the rank from the record yet.
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


def test_excluded_running_rank_is_replaced(tmp_path, deployment):
    """Kill a hung rank that the authority excludes and start its replacement on the same GPU."""
    authority, environ = deployment
    running = _serving(tmp_path, authority, environ)
    hung = _read(tmp_path, 3, 0)["pid"]
    authority.push(_record(MembershipState.PROPOSE, active_mask=0b0111))
    _until(lambda: _read(tmp_path, 3, 1))
    assert not _alive(hung)
    authority.push(_record(MembershipState.ABORTED, active_mask=0b0111))
    assert running.wait() == 1


def test_failed_replacement_restarts_until_cannot_start(tmp_path, deployment, monkeypatch):
    """Restart a replacement after a backoff until one exits with REPLACEMENT_CANNOT_START."""
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
    # No new replacement starts because it would fail the same way.
    assert running.supervisor._processes[3].status is not None
    assert running.supervisor._given_up == {3}
    authority.push(_record(MembershipState.SHUTDOWN, active_mask=0b0111))
    assert running.wait() == 0


def test_dropped_joining_replacement_is_restarted(tmp_path, deployment):
    """Kill and restart a replacement that one PROPOSE lists and the next one leaves out.

    The replacement was never admitted, so this fails if only admitted processes were killed.
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


def test_each_rank_has_own_session(tmp_path, deployment):
    """Check that a terminal signal reaches only the supervisor, which needs start_new_session."""
    authority, environ = deployment
    running = _serving(tmp_path, authority, environ)
    for rank in RANKS:
        pid = _read(tmp_path, rank, 0)["pid"]
        assert os.getsid(pid) == pid
    authority.push(_record(MembershipState.SHUTDOWN))
    assert running.wait() == 0


def test_early_rank_failure_ends_node(tmp_path, deployment):
    """Without an authority, the node fails with the rank's status as it does without fault tolerance."""
    _, environ = deployment
    environ["SUPERVISOR_TEST_MODES"] = "2:0=crashes"
    running = Running(tmp_path, environ)
    assert running.wait() == 7
    (other,) = [p.child for rank, p in running.supervisor._processes.items() if rank == 3]
    assert other.poll() is not None  # stopped with the node


def _fake_tool(directory, name, root):
    """Create a fake launcher tool that records its arguments in ``root/<name>-args`` and exits 0."""
    directory.mkdir(exist_ok=True)
    tool = directory / name
    tool.write_text(f'#!/bin/bash\nprintf "%s\\n" "$@" > "{root / f"{name}-args"}"\n')
    tool.chmod(0o755)
    return tool


@pytest.mark.parametrize("failed, served", [(0, False), (1, False), (0, True)])
def test_rank_0_exit_ends_job(tmp_path, deployment, failed, served):
    """Node 0's supervisor ends the job through its launcher as soon as rank 0 ends.

    No authority may be left to end the other nodes' ranks. A failure of another rank before the deployment serves
    ends only its own node, and the authority aborts the rest.
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


def test_rank_0_is_never_replaced(tmp_path, deployment):
    authority, environ = deployment
    running = _serving(tmp_path, authority, environ, node=0)
    os.kill(_read(tmp_path, 0, 0)["pid"], signal.SIGKILL)
    assert running.wait() == 128 + signal.SIGKILL
    assert _read(tmp_path, 0, 1) is None
    assert not _alive(_read(tmp_path, 1, 0)["pid"])


def test_session_starts_rank_0(tmp_path, deployment):
    """Start only the other ranks because ``MpiPoolSession`` runs rank 0 in its pool."""
    authority, environ = deployment
    running = _serving(tmp_path, authority, environ, node=0, rank_0_in_session=True)
    assert _read(tmp_path, 0, 0) is None
    assert list(running.supervisor._processes) == [1]
    authority.push(_record(MembershipState.SHUTDOWN))
    assert running.wait() == 0


def test_wrong_shared_secret_ends_supervisor(tmp_path, deployment, monkeypatch):
    """End with status 1 and start no rank when the store's values carry another secret's signature.

    This fails if the supervisor kept trying to connect instead.
    """
    authority, environ = deployment
    authority.start()
    monkeypatch.setenv(SHARED_SECRET_ENV, secrets.token_hex(32))
    supervisor = supervisor_module.NodeSupervisor(
        NODE, NODES, GPUS, [sys.executable, str(tmp_path / "rank.py")], environ
    )
    assert supervisor.run() == 1
    assert supervisor._processes == {}


def test_one_supervisor_per_node(tmp_path, deployment):
    authority, environ = deployment
    running = _serving(tmp_path, authority, environ)
    second = supervisor_module.NodeSupervisor(
        NODE, NODES, GPUS, [sys.executable, str(tmp_path / "rank.py")], environ
    )
    assert second.run() == 1
    authority.push(_record(MembershipState.SHUTDOWN))
    assert running.wait() == 0


def test_refused_supervisor_ends(tmp_path, monkeypatch):
    """The authority refuses a supervisor when another one supervises its node or one of its ranks."""
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


def test_late_supervisor_replaces_failed_ranks(tmp_path, deployment):
    """Start a replacement for each rank, and no original, when the node is launched again after it failed."""
    authority, environ = deployment
    authority.start(_record(MembershipState.COMMITTED, active_mask=0b0011))
    running = Running(tmp_path, environ)
    _until(lambda: _read(tmp_path, 2, 1) and _read(tmp_path, 3, 1))
    assert _read(tmp_path, 2, 0) is None
    assert _read(tmp_path, 3, 0) is None
    authority.push(_record(MembershipState.SHUTDOWN, active_mask=0b0011))
    assert running.wait() == 0


def test_supervisor_during_startup_starts_originals(tmp_path, deployment):
    """Start the original ranks when the first record a supervisor sees is STARTING.

    The authority starts from rank 0's worker, so it may answer before every node's supervisor has connected. Node 0
    must not relaunch a node whose supervisor has not connected yet.
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
        # Slurm's PMIx plugin cancels the whole step when one task exits.
        ({"SLURM_JOB_ID": "1", "SLURM_PROCID": "1"}, [], True),
        # mpirun inside the allocation. Its --enable-recovery is not visible here, so nothing is checked.
        ({"SLURM_JOB_ID": "1", "SLURM_PROCID": "0", "OMPI_COMM_WORLD_RANK": "1"}, [], False),
        # MpiPoolSession's, inside a Slurm task that is not a node's launch.
        ({"SLURM_JOB_ID": "1", "SLURM_PROCID": "0"}, ["--rank-0-in-session"], False),
    ],
)
def test_slurm_task_supervisor_is_rejected(monkeypatch, capfd, environ, options, rejected):
    for name in ("SLURM_JOB_ID", "SLURM_PROCID", "OMPI_COMM_WORLD_RANK"):
        monkeypatch.delenv(name, raising=False)
    for name, value in environ.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")  # main() stops at the GPU count after the check
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
def test_node_0_ends_job_through_prrte(tmp_path, deployment, launcher):
    """Send SIGTERM to the mpirun named by the job's PMIx server URI when rank 0 ends, else run pterm.

    SIGTERM makes mpirun (prterun) exit with a failure status. pterm makes it exit with status 0, so it is used only
    when that PID is not an mpirun on this host. A shell starts the supervisor here, as a wrapper script between
    mpirun and the launch would, and the shell must not be signaled.
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


@pytest.mark.parametrize("opt_in", [True, False])
def test_prrte_tools_keep_run_as_root(tmp_path, deployment, opt_in):
    """Check that PRRTE's tools keep the deployment's opt-in to running as root and never gain one.

    Under a root deployment, prun and pterm refuse to run without it.
    """
    _, environ = deployment
    for name in supervisor_module._RUN_AS_ROOT_VARS:
        environ.pop(name, None)
        if opt_in:
            environ[name] = "1"
    bin_dir = tmp_path / "bin"
    _fake_tool(bin_dir, "prun", tmp_path)
    pterm = bin_dir / "pterm"
    pterm.write_text(
        f"#!{sys.executable}\nimport json, os\n"
        f"open({str(tmp_path / 'pterm-environ')!r}, 'w').write(json.dumps(dict(os.environ)))\n"
    )
    pterm.chmod(0o755)
    environ.update(
        PATH=f"{bin_dir}:{environ['PATH']}",
        # The PID in the URI names no mpirun, so the supervisor ends the job with pterm.
        PMIX_SERVER_URI41="prterun-host-0-0@0.0;tcp4://127.0.0.1:1",
        SUPERVISOR_TEST_MODES="0:0=crashes",
    )
    node_0 = ["--node-rank", "0", "--nodes", str(NODES), "--gpus-per-node", str(GPUS)]
    supervisor = _supervisor_process(tmp_path, environ, *node_0, *_rank_command(tmp_path))
    try:
        assert supervisor.wait(30) == 7
    finally:
        supervisor.kill()
    pterm_environ = json.loads((tmp_path / "pterm-environ").read_text())
    for name in supervisor_module._RUN_AS_ROOT_VARS:
        assert (pterm_environ.get(name) == "1") == opt_in


def test_the_ranks_die_with_their_supervisor(tmp_path, deployment):
    """Check that killing the supervisor ends its ranks and closes its authority connection."""
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
def test_host_weights_unlinked_at_end(tmp_path, deployment, end):
    """Unlink the node's host weights however the deployment ends, and keep another deployment's."""
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


def test_host_weights_unlinked_before_ranks_end(tmp_path, deployment, monkeypatch):
    """Unlink the segments on SHUTDOWN before the ranks' grace period ends.

    A supervisor killed during the grace period then leaves no segment behind.
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


def test_signal_forwarded_to_rank_0_only(tmp_path, deployment):
    """Forward the signal to rank 0, which runs the frontend, and SIGTERM the other ranks after the grace period."""
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


# A fake prun. It records its process ID, arguments and standard input, and runs until ROOT/prun-exit exists.
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


def test_node_0_relaunches_failed_node_once(tmp_path, deployment):
    """Relaunch a node whose supervisor left through the job's PRRTE, only once.

    The deployment's environment goes on standard input and never on the command line.
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
    for _ in range(10):  # the failed node is reported repeatedly
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
    kept = ("OMPI_MCA_btl", *supervisor_module._RUN_AS_ROOT_VARS)
    assert not any(
        key.startswith(("OMPI_", "PMIX_")) for key in relaunched_environ if key not in kept
    )
    assert relaunched_environ["OMPI_MCA_btl"] == "^sm"
    connected = _record(MembershipState.COMMITTED, active_mask=0b0011)
    _push_and_wait_for_step(running, authority, connected, nodes)
    # A second relaunch would have replaced the first one.
    assert running.supervisor._relaunched[1].pid == int((tmp_path / "prun-0.pid").read_text())
    authority.push(_record(MembershipState.SHUTDOWN, active_mask=0b0011), nodes)
    assert running.wait() == 0


def test_relaunched_supervisor_reads_environment_stdin(tmp_path, deployment):
    """Read the environment, including the shared secret, as one JSON line because prun forwards none."""
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
        'if [ "$1" = "-m" ] || [[ "$1" == *_llmapi_node_supervisor.py ]] '
        '|| [[ "$1 $2" == "-c "*worker_node_main* ]]; then\n'
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
def test_node_launch_runs_supervisor(tmp_path, fault_tolerance):
    """Run the node supervisor from the launcher when the shared secret is exported.

    The supervisor then runs this launcher again for each rank of the node.
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


def test_node_launch_takes_mpirun_rank(tmp_path):
    """Take the node rank and node count from mpirun's variables, not from the batch step's SLURM_* variables.

    mpirun leaves the batch step's SLURM_PROCID (0) and SLURM_NTASKS in every process it starts. Reading those
    first would make every node node 0.
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


def test_supervised_rank_takes_identity_rank(tmp_path):
    """Take a supervised rank's rank from its identity instead of the launcher's variables.

    The worker replaces the launcher process so that it dies with the supervisor.
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
    args = (tmp_path / "python3-args").read_text().splitlines()
    assert args == [
        "-c",
        "from tensorrt_llm._torch.pyexecutor.rank_failure_recovery import worker_node_main; "
        "worker_node_main()",
    ]
    assert "MGMN worker node exit code" not in result.stderr  # exec'd


@contextlib.contextmanager
def _unanswered_port():
    """Yield a local port that is bound but not listening, so it refuses connections."""
    with socket.socket() as held:
        held.bind(("127.0.0.1", 0))
        yield held.getsockname()[1]


@pytest.mark.parametrize("fault_tolerance", [False, True])
def test_early_mpi_server_exit_status(tmp_path, fault_tolerance):
    """Exit 0 on a clean shutdown under fault tolerance, and 1 without it.

    Rank 0's worker ends its MPI server after the deployment shuts down, before the frontend task exits.
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
