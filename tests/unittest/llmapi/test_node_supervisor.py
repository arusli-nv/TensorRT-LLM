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
    REPLACEMENT_GIVEN_UP_KEY,
    SHARED_SECRET_ENV,
    MembershipRecord,
    MembershipState,
    NodeView,
    RendezvousStore,
    _Lines,
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
    supervisors and sends them the views given to :meth:`push`. The listener thread runs until :meth:`close`.
    """

    def __init__(self, secret: str):
        self._secret = bytes.fromhex(secret)
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
            TCPStore("127.0.0.1", port, is_master=True, wait_for_workers=False),
            self._secret,
            os.urandom(16),
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
            sock.sendall(f"{self.store.deployment_nonce.hex()} {nonce.hex()}\n".encode())
            hello = b""
            while not hello.endswith(b"\n"):
                data = sock.recv(256)
                if not data:
                    break
                hello += data
            if not hello.endswith(b"\n"):
                sock.close()
                continue
            client_nonce = bytes.fromhex(hello.split(b" ")[1].decode())
            nonces = (self.store.deployment_nonce, nonce, client_nonce)
            lines = _Lines(self.store, nonces, authority=True)
            announcement = lines.decode(hello)[0].decode().split()[1:]
            if announcement == ["store"]:
                with sock:
                    sock.sendall(lines.encode(self._view.model_dump_json().encode()))
                continue
            self.hellos.append(announcement)
            with self._lock:
                self._connections.append((sock, lines))
                sock.sendall(lines.encode(self._view.model_dump_json().encode()))
            self._thread(self._watch, sock)

    def _watch(self, sock):
        with contextlib.suppress(OSError):
            while sock.recv(256):
                pass
        self.closed.set()

    def push(self, record, nodes=((NODE, "host-1", True),)):
        with self._lock:
            self._view = NodeView(record=record, nodes=nodes)
            for sock, lines in self._connections:
                with contextlib.suppress(OSError):
                    sock.sendall(lines.encode(self._view.model_dump_json().encode()))

    def close(self):
        """Close every socket and join the threads.

        Only a shutdown, not a close, wakes a thread blocked on a socket.
        """
        self._reserved.close()
        with self._lock:
            sockets = [*(sock for sock, _ in self._connections), self._listener]
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
    _until(lambda: all(_read(tmp_path, rank, 0) for rank in range(node * GPUS, (node + 1) * GPUS)))
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
    environ.pop("OMPI_MCA_btl", None)
    if btl is not None:
        environ["OMPI_MCA_btl"] = btl
    running = _serving(tmp_path, authority, environ)
    kept = ("OMPI_MCA_btl", *supervisor_module._RUN_AS_ROOT_VARS)
    for rank in RANKS:
        started = _read(tmp_path, rank, 0)
        assert started["environ"]["TLLM_FAULT_TOLERANCE_RANK"] == f"{rank},4,2,0"  # on GPU rank % 2
        assert not any(
            key.startswith(("OMPI_", "PMIX_")) for key in started["environ"] if key not in kept
        )
        # Without the user's setting, no /dev/shm segment to leave behind.
        assert started["environ"]["OMPI_MCA_btl"] == (btl or "^sm")
        assert started["environ"][SHARED_SECRET_ENV] == os.environ[SHARED_SECRET_ENV]
        assert os.environ[SHARED_SECRET_ENV] not in " ".join(started["argv"])
        # A terminal signal reaches only the supervisor.
        assert os.getsid(started["pid"]) == started["pid"]
    assert authority.hellos == [["node", "1", "2,3", socket.gethostname()]]
    authority.push(_record(MembershipState.SHUTDOWN))
    assert running.wait() == 0


@pytest.mark.parametrize("failure", ["crash", "hang"])
def test_failed_rank_is_replaced(tmp_path, deployment, failure):
    """Replace a rank on its GPU once the record leaves it out, and kill it first if it still runs."""
    authority, environ = deployment
    running = _serving(tmp_path, authority, environ)
    failed = _read(tmp_path, 3, 0)["pid"]
    if failure == "crash":
        os.kill(failed, signal.SIGKILL)
        _until(lambda: running.supervisor._processes[3].status is not None)
        # The authority has not removed the rank from the record yet.
        _push_and_wait_for_step(running, authority, _record(generation=2))
        assert not running.supervisor._processes[3].replacement
    authority.push(_record(MembershipState.PROPOSE, active_mask=0b0111, generation=4))
    _until(lambda: _read(tmp_path, 3, 1))
    replacement = _read(tmp_path, 3, 1)
    assert replacement["environ"]["TLLM_FAULT_TOLERANCE_RANK"] == "3,4,2,1"
    assert not _alive(failed)
    assert _alive(_read(tmp_path, 2, 0)["pid"])
    authority.push(_record(MembershipState.SHUTDOWN, incarnations=(0, 0, 0, 1), generation=5))
    assert running.wait() == 0
    assert not _alive(replacement["pid"])


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
    assert authority.store.get_if_set(f"{REPLACEMENT_GIVEN_UP_KEY}/3") == b"1"
    authority.push(_record(MembershipState.SHUTDOWN, active_mask=0b0111))
    assert running.wait() == 0


def test_repeated_refusals_stop_replacements(tmp_path, deployment):
    """Check that no replacement starts once the membership left out three in a row for the same reason.

    Each replacement of rank 3 restores another expert placement than the surviving ranks, so each change that adds
    it leaves it out. The node keeps serving without rank 3.
    """
    authority, environ = deployment
    running = _serving(tmp_path, authority, environ)
    os.kill(_read(tmp_path, 3, 0)["pid"], signal.SIGKILL)
    authority.push(_record(MembershipState.COMMITTED, active_mask=0b0111))
    reason = "a replacement restored a different expert placement than the surviving ranks"
    generation = 1
    for incarnation in range(1, supervisor_module._MAX_FAILED_REPLACEMENTS + 1):
        _until(lambda incarnation=incarnation: _read(tmp_path, 3, incarnation))
        assert authority.store.get_if_set(f"{REPLACEMENT_GIVEN_UP_KEY}/3") == b"0"
        joined = (0, 0, 0, incarnation)
        adding = _record(MembershipState.INSTALL, 0b1111, joined, generation + 1)
        authority.push(adding.model_copy(update=dict(joining_mask=0b1000)))
        _until(lambda: running.supervisor._processes[3].listed)
        left_out = _record(MembershipState.PROPOSE, 0b0111, joined, generation + 2)
        authority.push(left_out.model_copy(update=dict(reason=reason)))
        generation += 2
    _until(lambda: running.supervisor._given_up == {3})
    _push_and_wait_for_step(
        running, authority, _record(MembershipState.COMMITTED, 0b0111, joined, generation + 1)
    )
    assert _read(tmp_path, 3, supervisor_module._MAX_FAILED_REPLACEMENTS + 1) is None
    assert authority.store.get_if_set(f"{REPLACEMENT_GIVEN_UP_KEY}/3") == b"1"
    authority.push(_record(MembershipState.SHUTDOWN, 0b0111, joined, generation + 2))
    assert running.wait() == 0


@pytest.mark.parametrize(
    "admitted, served_s, given_up",
    [(False, 600.0, True), (True, 600.0, True), (True, 0.0, False)],
    ids=["before-admission", "after-admission", "after-serving"],
)
def test_ended_replacements_stop_replacements(
    tmp_path, deployment, monkeypatch, admitted, served_s, given_up
):
    """Check that no replacement starts once three in a row ended before serving for ``_SERVED_S``.

    A replacement that ends after serving longer is a new failure of its rank, so its replacement starts.
    """
    monkeypatch.setattr(supervisor_module, "_SERVED_S", served_s)
    authority, environ = deployment
    running = _serving(tmp_path, authority, environ)
    os.kill(_read(tmp_path, 3, 0)["pid"], signal.SIGKILL)
    generation = 1
    authority.push(_record(MembershipState.COMMITTED, active_mask=0b0111, generation=generation))
    for incarnation in range(1, supervisor_module._MAX_FAILED_REPLACEMENTS + 1):
        _until(lambda incarnation=incarnation: _read(tmp_path, 3, incarnation))
        joined = (0, 0, 0, incarnation)
        if admitted:
            serving = _record(MembershipState.COMMITTED, 0b1111, joined, generation + 1)
            _push_and_wait_for_step(running, authority, serving)
            generation += 2
        os.kill(_read(tmp_path, 3, incarnation)["pid"], signal.SIGKILL)
        generation += 1
        authority.push(_record(MembershipState.COMMITTED, 0b0111, joined, generation))
    if given_up:
        _until(lambda: running.supervisor._given_up == {3})
    else:
        _until(lambda: _read(tmp_path, 3, supervisor_module._MAX_FAILED_REPLACEMENTS + 1))
    _push_and_wait_for_step(
        running, authority, _record(MembershipState.COMMITTED, 0b0111, joined, generation + 1)
    )
    assert (_read(tmp_path, 3, supervisor_module._MAX_FAILED_REPLACEMENTS + 1) is None) == given_up
    assert authority.store.get_if_set(f"{REPLACEMENT_GIVEN_UP_KEY}/3") == (
        b"1" if given_up else b"0"
    )
    authority.push(_record(MembershipState.SHUTDOWN, 0b0111, joined, generation + 3))
    assert running.wait() == 0


def test_rank_0_is_never_replaced(tmp_path, deployment):
    authority, environ = deployment
    running = _serving(tmp_path, authority, environ, node=0)
    os.kill(_read(tmp_path, 0, 0)["pid"], signal.SIGKILL)
    assert running.wait() == 128 + signal.SIGKILL
    assert _read(tmp_path, 0, 1) is None
    assert not _alive(_read(tmp_path, 1, 0)["pid"])


def test_wrong_shared_secret_ends_supervisor(tmp_path, deployment, monkeypatch):
    """End with status 1 and start no rank when the store's values carry another secret's signature."""
    authority, environ = deployment
    authority.start()
    monkeypatch.setenv(SHARED_SECRET_ENV, secrets.token_hex(32))
    supervisor = supervisor_module.NodeSupervisor(
        NODE, NODES, GPUS, [sys.executable, str(tmp_path / "rank.py")], environ
    )
    assert supervisor.run() == 1
    assert supervisor._processes == {}


def _fake_tool(directory, name, root):
    """Create a fake launcher tool that records its arguments in ``root/<name>-args`` and exits 0."""
    directory.mkdir(exist_ok=True)
    tool = directory / name
    tool.write_text(f'#!/bin/bash\nprintf "%s\\n" "$@" > "{root / f"{name}-args"}"\n')
    tool.chmod(0o755)
    return tool


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


@pytest.mark.parametrize("name", [SHARED_SECRET_ENV, RENDEZVOUS_ENV])
def test_supervisor_names_unset_variable(tmp_path, deployment, name):
    _, environ = deployment
    del environ[name]
    supervisor = _supervisor_process(
        tmp_path,
        environ,
        *_NODE_OPTIONS,
        *_rank_command(tmp_path),
        stderr=subprocess.PIPE,
        text=True,
    )
    _, stderr = supervisor.communicate(timeout=60)
    assert supervisor.returncode == 2
    assert f"export the same {name} on every node" in stderr


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


@pytest.mark.parametrize(
    "opt_in, root",
    [(True, False), (False, True), (False, False)],
    ids=["opt-in", "root", "non-root"],
)
def test_prrte_tools_keep_run_as_root(tmp_path, deployment, monkeypatch, opt_in, root):
    """Check that prun and pterm opt in to running as root only with the deployment's opt-in or as root.

    Under a root deployment, prun and pterm refuse to run without it.
    """
    authority, environ = deployment
    monkeypatch.setattr(os, "geteuid", lambda: 0 if root else 1000)
    for name in supervisor_module._RUN_AS_ROOT_VARS:
        environ.pop(name, None)
        if opt_in:
            environ[name] = "1"
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    for name in ("prun", "pterm"):
        tool = bin_dir / name
        tool.write_text(
            f"#!{sys.executable}\nimport json, os\n"
            f"open({str(tmp_path / f'{name}-environ')!r}, 'w').write(json.dumps(dict(os.environ)))\n"
        )
        tool.chmod(0o755)
    environ.update(
        PATH=f"{bin_dir}:{environ['PATH']}",
        # The PID in the URI names no mpirun, so the supervisor ends the job with pterm.
        PMIX_SERVER_URI41="prterun-host-0-0@0.0;tcp4://127.0.0.1:1",
    )
    relaunch_command, end_job = supervisor_module._prrte_job_control(environ, NODES)
    nodes = ((0, "host-0", True), (1, "host-1", True))
    node_1_failed = ((0, "host-0", True), (1, "host-1", False))
    running = _serving(
        tmp_path, authority, environ, node=0, relaunch_command=relaunch_command, end_job=end_job
    )
    _push_and_wait_for_step(running, authority, _record(), nodes)
    authority.push(_record(MembershipState.PROPOSE, active_mask=0b0011), node_1_failed)
    _until(lambda: (tmp_path / "prun-environ").exists())
    os.kill(_read(tmp_path, 0, 0)["pid"], signal.SIGKILL)
    assert running.wait() == 128 + signal.SIGKILL
    for name in ("prun", "pterm"):
        tool_environ = json.loads((tmp_path / f"{name}-environ").read_text())
        for variable in supervisor_module._RUN_AS_ROOT_VARS:
            assert (tool_environ.get(variable) == "1") == (opt_in or root)


def test_node_0_requires_prrte_tools(tmp_path, deployment):
    """Reject a multi-node job under mpirun whose node 0 cannot relaunch a node or end the job.

    One node needs neither, since the end of its supervisor ends the job.
    """
    _, environ = deployment
    environ["PATH"] = str(tmp_path)
    for name in supervisor_module._PMIX_SERVER_URI_VARS:
        environ.pop(name, None)
    node_0 = ["--node-rank", "0", "--nodes", str(NODES), "--gpus-per-node", str(GPUS)]
    supervisor = _supervisor_process(
        tmp_path, environ, *node_0, *_rank_command(tmp_path), stderr=subprocess.PIPE, text=True
    )
    try:
        _, stderr = supervisor.communicate(timeout=30)
    finally:
        supervisor.kill()
    assert supervisor.returncode == 2
    assert "lacks a PMIx server URI, prun, pterm" in stderr
    assert _read(tmp_path, 0, 0) is None
    assert supervisor_module._prrte_job_control(environ, 1) == (None, None)


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


@pytest.mark.parametrize("end", ["SIGINT", "SIGTERM", "ABORTED", "SHUTDOWN"])
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
        assert status == {"SIGINT": 130, "SIGTERM": 143, "ABORTED": 1, "SHUTDOWN": 0}[end]
        assert not any(segment.exists() for segment in segments)
        assert other.exists()
        assert not any(_alive(_read(tmp_path, rank, 0)["pid"]) for rank in RANKS)
    finally:
        supervisor.kill()
        for segment in [*segments, other]:
            with contextlib.suppress(FileNotFoundError):
                segment.unlink()


def test_signal_forwarded_to_rank_0_only(tmp_path, deployment):
    """Forward the signal to rank 0, which runs the frontend, and SIGTERM the other ranks after the grace period."""
    authority, environ = deployment
    environ["SUPERVISOR_TEST_MODES"] = "0:0=records-signals 1:0=records-signals"
    for tool in ("prun", "pterm"):
        _fake_tool(tmp_path / "bin", tool, tmp_path)
    environ.update(
        PATH=f"{tmp_path / 'bin'}:{environ['PATH']}",
        PMIX_SERVER_URI41="prterun-host-0-0@0.0;tcp4://127.0.0.1:1",
    )
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

    The deployment's environment, with its opt-in to running as root, goes on standard input and never on the
    command line.
    """
    authority, environ = deployment
    for name in supervisor_module._RUN_AS_ROOT_VARS:
        environ[name] = "1"
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
        *["--node-rank", "1", "--nodes", "2", "--gpus-per-node", "2"],
        "--environment-from-stdin",
        *_rank_command(tmp_path),
    ]
    secret = os.environ[SHARED_SECRET_ENV]
    assert secret not in " ".join(args)
    relaunched_environ = json.loads((tmp_path / "prun-0.stdin").read_text())
    assert relaunched_environ[SHARED_SECRET_ENV] == secret
    kept = ("OMPI_MCA_btl", *supervisor_module._RUN_AS_ROOT_VARS)
    assert not any(
        key.startswith(("OMPI_", "PMIX_")) for key in relaunched_environ if key not in kept
    )
    assert all(relaunched_environ[name] == "1" for name in supervisor_module._RUN_AS_ROOT_VARS)
    assert relaunched_environ["OMPI_MCA_btl"] == "^sm"
    connected = _record(MembershipState.COMMITTED, active_mask=0b0011)
    _push_and_wait_for_step(running, authority, connected, nodes)
    # A second relaunch would have replaced the first one.
    assert running.supervisor._relaunched[1].pid == int((tmp_path / "prun-0.pid").read_text())
    authority.push(_record(MembershipState.SHUTDOWN, active_mask=0b0011), nodes)
    assert running.wait() == 0


def test_relaunched_supervisor_replaces_failed_ranks(tmp_path, deployment):
    """Start a replacement for each rank, and no original, with the environment read from standard input.

    prun forwards no environment, so the shared secret arrives as one JSON line.
    """
    authority, environ = deployment
    authority.start(_record(MembershipState.COMMITTED, active_mask=0b0011))
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
        _until(lambda: _read(tmp_path, 2, 1) and _read(tmp_path, 3, 1))
        assert _read(tmp_path, 2, 0) is None
        assert _read(tmp_path, 3, 0) is None
        assert _read(tmp_path, 2, 1)["environ"][SHARED_SECRET_ENV] == os.environ[SHARED_SECRET_ENV]
        authority.push(_record(MembershipState.SHUTDOWN, active_mask=0b0011))
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

    The node rank and count come from mpirun's variables. mpirun leaves the batch step's SLURM_PROCID (0) and
    SLURM_NTASKS in every process it starts. The supervisor then runs this launcher again for each rank of the node.
    """
    extra = {}
    if fault_tolerance:
        extra = {
            "TLLM_FAULT_TOLERANCE_SHARED_SECRET": secrets.token_hex(32),
            "OMPI_COMM_WORLD_RANK": "1",
            "OMPI_COMM_WORLD_SIZE": "3",
        }
    result = _run_launcher(tmp_path, 0 if fault_tolerance else 1, ["task"], **extra)
    assert result.returncode == 0, result.stderr
    args = (tmp_path / "python3-args").read_text().split()
    if fault_tolerance:
        assert Path(args[0]).resolve() == _SUPERVISOR.resolve()
        assert args[1:6] == ["--node-rank", "1", "--nodes", "3", "--"]
        assert Path(args[6]).resolve() == (_LLMAPI / "trtllm-llmapi-launch").resolve()
        assert args[7:] == ["task"]
    else:
        assert args == ["-m", "tensorrt_llm.llmapi.mgmn_worker_node"]


@pytest.mark.parametrize("fault_tolerance", [False, True])
def test_early_mpi_server_exit_status(tmp_path, fault_tolerance):
    """Exit 0 on a clean shutdown under fault tolerance, and 1 without it.

    Rank 0's worker ends its MPI server after the deployment shuts down, before the frontend task exits.
    """
    with socket.socket() as unanswered:  # bound but not listening, so it refuses connections
        unanswered.bind(("127.0.0.1", 0))
        extra = {}
        if fault_tolerance:
            extra = {
                "TLLM_FAULT_TOLERANCE_SHARED_SECRET": secrets.token_hex(32),
                "TLLM_FAULT_TOLERANCE_RENDEZVOUS": f"127.0.0.1:{unanswered.getsockname()[1]}",
                "TLLM_FAULT_TOLERANCE_RANK": "0,4,2,0",
            }
        result = _run_launcher(tmp_path, 0, ["bash", "-c", "sleep 1"], **extra)
    assert "MPI Comm server exited before the task (status 0)" in result.stderr
    assert result.returncode == (0 if fault_tolerance else 1), result.stderr
