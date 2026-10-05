# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""CPU safety checks: wrong targets, stale triggers and duplicate intent must not signal."""

import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import tarfile
import time
from pathlib import Path

import pytest
from deployment_profile import normalize_ray_log
from process_probe import IDENTITY_KEYS, WorkerProbe, process_identity, record, validate_injection
from regression import assert_engine_error


@pytest.mark.parametrize("port", [1, 65535])
def test_native_selected_head_port_is_admitted(port: int) -> None:
    """Does native port selection admit valid ports without relying on host ephemeral policy?"""
    from process_loss_controller import validate_head_address

    validate_head_address(f"127.0.0.1:{port}", "127.0.0.1:0")


@pytest.mark.parametrize(
    "actual", ["127.0.0.1:0", "127.0.0.1:65536", "127.0.0.1:-1", "127.0.0.1:bad", "127.0.0.2:1234"]
)
def test_dynamic_head_endpoint_rejects_wrong_host_or_invalid_port(actual: str) -> None:
    """Can a different host or an invalid native-selected port authorize head readiness?"""
    from process_loss_controller import validate_head_address

    with pytest.raises(ValueError):
        validate_head_address(actual, "127.0.0.1:0")


def test_tcp_self_connection_is_not_server_readiness() -> None:
    """Can a successful TCP connect occupy a port even though no server was listening?"""
    with socket.socket() as connection:
        connection.settimeout(1)
        connection.bind(("127.0.0.1", 0))
        address = connection.getsockname()
        connection.connect(address)
        assert connection.getsockname() == connection.getpeername()
        connection.sendall(b"self")
        assert connection.recv(4) == b"self"
    with socket.socket() as listener:
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        with pytest.raises(OSError) as rejected:
            listener.bind(("0.0.0.0", address[1]))
        assert rejected.value.errno == 98


@pytest.mark.parametrize("fault", ["run_id", "address", "check_alive", "ray_version"])
def test_head_readiness_rejects_wrong_run_or_endpoint(tmp_path: Path, fault: str) -> None:
    """Can another run, endpoint or Ray version authorize model launch?"""
    from process_loss_controller import validate_head_ready

    address = "127.0.0.1:24628"
    record(tmp_path, "run", {"run_id": "process-loss-123"})
    ready = {
        "run_id": "process-loss-123",
        "address": address,
        "check_alive": True,
        "ray_version": "2.55.1",
    }
    validate_head_ready(ready, tmp_path, "127.0.0.1:0")
    ready[fault] = "wrong"
    with pytest.raises(ValueError):
        validate_head_ready(ready, tmp_path, "127.0.0.1:0")


@pytest.mark.parametrize("published", [False, True])
def test_head_readiness_requires_own_cli_metadata(
    tmp_path: Path, monkeypatch, published: bool
) -> None:
    """Can another endpoint or an exited native CLI trigger readiness without owned metadata?"""
    from types import SimpleNamespace

    import process_loss_controller

    metadata = tmp_path / "ray_current_cluster"
    monkeypatch.setattr(process_loss_controller, "RAY_ADDRESS_FILE", metadata)
    if published:
        metadata.write_text("127.0.0.2:6379")
    child = SimpleNamespace(poll=lambda: 1)
    with pytest.raises(ValueError if published else RuntimeError):
        process_loss_controller.wait_head("127.0.0.1:0", child, time.monotonic() + 1)


def test_service_worker_joins_native_selected_endpoint(tmp_path: Path, monkeypatch) -> None:
    """Does a remote service use the head's actual bound endpoint rather than the port-zero intent?"""
    from argparse import Namespace

    import process_loss_controller

    record(tmp_path, "run", {"run_id": "process-loss-123"})
    record(
        tmp_path,
        "head_ready",
        {
            "address": "127.0.0.1:53123",
            "ray_address": "127.0.0.1:53123",
            "check_alive": True,
            "ray_version": "2.55.1",
        },
    )
    monkeypatch.setattr(
        process_loss_controller.os,
        "environ",
        {**os.environ, "SLURM_PROCID": "1", "SLURM_JOB_ID": "123", "SLURM_STEP_ID": "2"},
    )
    monkeypatch.setattr(
        process_loss_controller.socket, "gethostbyname", lambda hostname: "127.0.0.2"
    )
    commands = []
    monkeypatch.setattr(
        process_loss_controller.os, "execvp", lambda executable, command: commands.append(command)
    )
    (tmp_path / "config.yaml").write_text(
        "tensor_parallel_size: 32\nmoe_expert_parallel_size: 32\nmoe_tensor_parallel_size: 1\ngpus_per_node: 4\n"
    )
    process_loss_controller.bootstrap(Namespace(output_dir=tmp_path, address="127.0.0.1:0"))
    assert "--address=127.0.0.1:53123" in commands[0]
    assert "--address=127.0.0.1:0" not in commands[0]


@pytest.fixture
def target() -> dict:
    return {**process_identity(os.getpid()), "rank": 16, "actor_id": "actor", "run_id": "run"}


@pytest.mark.parametrize("key", IDENTITY_KEYS)
def test_identity_drift_refuses_injection(target: dict, key: str) -> None:
    """Can a changed process or actor identity receive the injection?"""
    changed = {**target, key: "changed"}
    with pytest.raises(ValueError):
        validate_injection(
            target, changed, "idle_kill", {"run_id": "run", "event": "healthy_between_requests"}
        )


@pytest.mark.parametrize(
    "trigger",
    [
        {"run_id": "stale", "event": "healthy_between_requests"},
        {"run_id": "run", "event": "first_nonfinal_output", "token_ids": [], "finished": False},
        {"run_id": "run", "event": "first_nonfinal_output", "token_ids": [1], "finished": True},
    ],
)
def test_stale_or_completed_trigger_is_rejected(target: dict, trigger: dict) -> None:
    """Can stale, empty or finished output authorize a streaming kill?"""
    with pytest.raises(ValueError):
        validate_injection(target, target, "stream_kill", trigger)


def test_rank_zero_requires_explicit_case(target: dict) -> None:
    """Can an ordinary middle-rank case accidentally kill rank zero?"""
    target = {**target, "rank": 0}
    trigger = {"run_id": "run", "event": "healthy_between_requests"}
    with pytest.raises(ValueError):
        validate_injection(target, target, "idle_kill", trigger)
    validate_injection(target, target, "rank0_kill", trigger)


def test_duplicate_intent_cannot_issue_second_signal(
    tmp_path: Path, target: dict, monkeypatch
) -> None:
    """Does an already published action refuse a second signal, even if its target still lives?"""
    record(tmp_path, "run", {"run_id": "run"})
    record(tmp_path, "trigger", {"event": "healthy_between_requests"})
    probe = WorkerProbe()
    monkeypatch.setattr(probe, "wideep_identity", lambda: target)
    signals = []
    monkeypatch.setattr(os, "kill", lambda *args: signals.append(args))
    probe.wideep_signal(target, str(tmp_path), "idle_kill")
    with pytest.raises(FileExistsError):
        probe.wideep_signal(target, str(tmp_path), "idle_kill")
    assert signals == [(os.getpid(), signal.SIGKILL)]


@pytest.mark.parametrize(
    "scenario,action",
    [("idle_kill", signal.SIGKILL), ("idle_term", signal.SIGTERM), ("idle_stop", signal.SIGSTOP)],
)
def test_owned_child_signal_has_matching_durable_intent(
    tmp_path: Path, scenario: str, action: signal.Signals
) -> None:
    """Does the real signal affect only the published child, with STOP observed and cleaned?"""
    script = """
import os
from pathlib import Path
from process_probe import WorkerProbe, process_identity, record
path = Path(os.environ['TEST_ARTIFACTS'])
target = {**process_identity(os.getpid()), 'rank': 16, 'actor_id': 'cpu-owned', 'run_id': path.name}
record(path, 'trigger', {'event': 'healthy_between_requests'})
probe = WorkerProbe()
probe.wideep_identity = lambda: target
probe.wideep_signal(target, str(path), os.environ['TEST_SCENARIO'])
"""
    child = subprocess.Popen(
        [sys.executable, "-c", script],
        env={
            **os.environ,
            "TEST_ARTIFACTS": str(tmp_path),
            "TEST_SCENARIO": scenario,
            "PYTHONPATH": str(Path(__file__).parent),
        },
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    try:
        if action == signal.SIGSTOP:
            deadline = time.monotonic() + 5
            while process_identity(child.pid)["state"] != "T":
                assert child.poll() is None, "Child exited before STOP was observed"
                assert time.monotonic() < deadline, "STOP observation timed out"
                time.sleep(0.01)
        else:
            _, stderr = child.communicate(timeout=10)
            assert child.returncode == -action, stderr
        intent = json.loads((tmp_path / "injection_intent.json").read_text())
        assert intent["target"]["pid"] == child.pid
        assert intent["target"]["uid"] == os.getuid()
        assert intent["signal"] == action.name
    finally:
        if child.poll() is None:
            child.kill()
        child.communicate(timeout=5)


def test_harness_timeout_never_passes_native_error_contract() -> None:
    """Can a harness timeout masquerade as native failure reporting?"""
    with pytest.raises(AssertionError):
        assert_engine_error(
            {"state": "error", "error_type": "TimeoutError", "harness_timeout": True}, 30
        )
    assert_engine_error(
        {
            "state": "error",
            "error_type": "EngineDeadError",
            "error": "Engine has died: RPCStreamingError: model collective failed",
            "harness_timeout": False,
            "seconds_since_injection": 1,
        },
        30,
    )


def test_unrelated_control_channel_failure_does_not_qualify_worker_death() -> None:
    """Can a bounded EngineDeadError from the notification channel pass worker-loss reporting?"""
    with pytest.raises(AssertionError):
        assert_engine_error(
            {
                "state": "error",
                "error_type": "EngineDeadError",
                "error": "Engine has died: RuntimeError: Ray actor notification channel unavailable",
                "harness_timeout": False,
                "seconds_since_injection": 0.1,
            },
            30,
        )


@pytest.fixture
def temporary_run(tmp_path: Path, monkeypatch):
    """Provide a unique numeric allocation and real node-local tree for filesystem checks."""
    import process_probe

    job_id = f"{os.getpid()}{time.monotonic_ns()}"
    monkeypatch.setenv("SLURM_JOB_ID", job_id)
    record(tmp_path, "run", {"job_id": job_id})
    host = process_identity(os.getpid())["hostname"]
    record(
        tmp_path,
        f"cleanup/{host}",
        {
            "hostname": host,
            "clean": True,
            "original_live": [],
            "owned_live": [],
            "owned_gpu_contexts": [],
            "memory_delta_mib": {"gpu": 0},
            "memory_tolerance_mib": 64,
        },
    )
    path = process_probe._temporary_path(job_id)
    try:
        yield tmp_path, path, host
    finally:
        if path.is_symlink():
            path.unlink()
        elif path.exists():
            shutil.rmtree(path)


def test_temporary_cleanup_preserves_logs_and_removes_pinned_tree(temporary_run) -> None:
    """Does cleanup preserve Ray logs while leaving symlink targets outside the owned tree intact?"""
    from process_probe import cleanup_temporary, prepare_temporary

    directory, path, host = temporary_run
    prepare_temporary(directory)
    logs = path / "ray/session_2026/logs"
    logs.mkdir(parents=True)
    (logs / "worker.log").write_text("original worker log\n")
    (path / "ray/session_latest").symlink_to("session_2026", target_is_directory=True)
    external = directory / "external.log"
    external.write_text("retained external file\n")
    (logs / "external.log").symlink_to(external)
    cleanup_temporary(directory)
    result = json.loads((directory / f"temporary_cleanup/{host}.json").read_text())
    assert result["state"] == "absent" and not path.exists()
    assert external.read_text() == "retained external file\n"
    with tarfile.open(directory / result["archive"]) as archive:
        assert (
            archive.extractfile("ray/session_2026/logs/worker.log").read()
            == b"original worker log\n"
        )
        assert archive.getmember("ray/session_2026/logs/external.log").issym()


@pytest.mark.parametrize("replacement", ["directory", "symlink"])
def test_temporary_cleanup_rejects_identity_drift(temporary_run, replacement: str) -> None:
    """Can a replacement directory or symlink be deleted using stale ownership evidence?"""
    from process_probe import cleanup_temporary, prepare_temporary

    directory, path, host = temporary_run
    prepare_temporary(directory)
    original = directory / "original"
    path.rename(original)
    if replacement == "directory":
        path.mkdir()
        (path / "untouched").write_text("replacement")
    else:
        path.symlink_to(original, target_is_directory=True)
    with pytest.raises(ValueError, match="directory|identity"):
        cleanup_temporary(directory)
    assert path.exists() and original.exists()
    assert not (directory / f"temporary_cleanup/{host}.json").exists()


def test_temporary_cleanup_rejects_unpinned_existing_tree(temporary_run) -> None:
    """Can startup adopt or teardown delete a directory that predates its ownership record?"""
    from process_probe import cleanup_temporary, prepare_temporary

    directory, path, _ = temporary_run
    path.mkdir()
    with pytest.raises(FileExistsError):
        prepare_temporary(directory)
    with pytest.raises(ValueError, match="unpinned"):
        cleanup_temporary(directory)
    assert path.exists()


def test_temporary_preparation_publication_failure_removes_only_new_empty_tree(
    temporary_run, monkeypatch
) -> None:
    """Does failed ownership publication leave behind a newly created unpinned directory?"""
    import process_probe

    directory, path, _ = temporary_run

    def fail_record(*args) -> None:
        raise OSError("Ownership publication failed")

    monkeypatch.setattr(process_probe, "record", fail_record)
    with pytest.raises(OSError, match="publication failed"):
        process_probe.prepare_temporary(directory)
    assert not path.exists()


def test_temporary_cleanup_requires_clean_resources(temporary_run) -> None:
    """Can temporary files be removed while an independently observed GPU context remains?"""
    from process_probe import cleanup_temporary, prepare_temporary

    directory, path, host = temporary_run
    prepare_temporary(directory)
    evidence = directory / f"cleanup/{host}.json"
    value = json.loads(evidence.read_text())
    value["owned_gpu_contexts"] = [["gpu", "123", "1"]]
    evidence.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="process and GPU"):
        cleanup_temporary(directory)
    assert path.exists()


@pytest.mark.parametrize("created", [False, True])
def test_temporary_cleanup_handles_partial_startup(temporary_run, created: bool) -> None:
    """Does cleanup verify absence on hosts reached before or after temporary preparation?"""
    from process_probe import cleanup_temporary, prepare_temporary

    directory, path, host = temporary_run
    if created:
        prepare_temporary(directory)
    cleanup_temporary(directory)
    result = json.loads((directory / f"temporary_cleanup/{host}.json").read_text())
    assert result["state"] == ("absent" if created else "never_created")
    assert not path.exists()


def test_forwarded_rank_must_match_actor_pid_and_ip() -> None:
    """Can log arrival order or a conflicting native rank label corrupt attribution?"""
    workers = [{"rank": 16, "pid": 42, "node_ip": "1.2.3.4", "actor_id": "actor"}]
    log = "0: (RayWorkerWrapper pid=42, ip=1.2.3.4) [RANK 16] Selected communication strategy: NVLinkOneSided (forced)"
    assert normalize_ray_log(log, workers, "1.2.3.5").startswith("16: [RANK 16]")
    with pytest.raises(ValueError):
        normalize_ray_log(log.replace("[RANK 16]", "[RANK 0]"), workers, "1.2.3.5")


def test_missing_identity_fields_never_authorize_injection(target: dict) -> None:
    """Can mutually incomplete expected/actual identities authorize a signal?"""
    target.pop("actor_id")
    with pytest.raises(ValueError, match="Incomplete"):
        validate_injection(
            target, target, "idle_kill", {"run_id": "run", "event": "healthy_between_requests"}
        )


def test_cleanup_does_not_claim_container_root_cgroup(tmp_path: Path, monkeypatch) -> None:
    """Does a container's cgroup / accidentally classify host processes as owned?"""
    import process_probe

    baseline = {
        "hostname": process_identity(os.getpid())["hostname"],
        "compute_apps": [],
        "memory_used_mib": {"gpu": 0},
    }
    monkeypatch.setattr(process_probe, "resources", lambda: baseline)
    observed = process_probe.cleanup_snapshot(baseline, [], "123", ["2"], 64)
    assert observed["owned_live"] == []
    assert observed["clean"]


def test_cleanup_tracks_live_original_across_user_namespace(monkeypatch) -> None:
    """Can namespace UID translation hide an original process that still exists?"""
    import process_probe

    original = {**process_identity(os.getpid()), "uid": 0}
    baseline = {"hostname": original["hostname"], "compute_apps": [], "memory_used_mib": {"gpu": 0}}
    monkeypatch.setattr(process_probe, "resources", lambda: baseline)
    observed = process_probe.cleanup_snapshot(baseline, [original], "123", ["2"], 64)
    assert len(observed["original_live"]) == 1
    assert not observed["clean"]


def test_cleanup_rejects_unknown_context_below_memory_allowance(monkeypatch) -> None:
    """Can a small orphaned CUDA context pass just because its PID was not in the worker map?"""
    import process_probe

    baseline = {
        "hostname": process_identity(os.getpid())["hostname"],
        "compute_apps": [],
        "memory_used_mib": {"gpu": 0},
    }
    current = {**baseline, "compute_apps": [["gpu", "99999999", "1"]]}
    monkeypatch.setattr(process_probe, "resources", lambda: current)
    observed = process_probe.cleanup_snapshot(baseline, [], "123", ["2"], 64)
    assert observed["owned_gpu_contexts"] == current["compute_apps"]
    assert not observed["clean"]


@pytest.fixture
def frontend_child(tmp_path: Path, monkeypatch):
    """Provide an owned child and simulated Slurm/container metadata for pidfd safety checks."""
    import process_probe

    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    original_identity = process_identity
    expected = original_identity(child.pid)
    monkeypatch.setenv("SLURM_JOB_ID", "123")
    for name, value in (
        ("run", {"job_id": "123", "scenario": "frontend_kill"}),
        (
            "step-driver",
            {"job_id": "123", "step_id": "3", "identity": {**expected, "uid": 0, "cgroup": "/"}},
        ),
        ("frontend", {"identity": {**expected, "uid": 0, "cgroup": "/"}}),
        ("trigger", {"event": "healthy_http_completed"}),
    ):
        record(tmp_path, name, value)

    def host_identity(pid: int) -> dict:
        return {**original_identity(pid), "cgroup": "/job_123/step_3/"}

    monkeypatch.setattr(process_probe, "process_identity", host_identity)
    try:
        yield child, tmp_path
    finally:
        if child.poll() is None:
            child.kill()
        child.wait(timeout=5)


def test_frontend_pidfd_kills_only_published_child(frontend_child) -> None:
    """Does a pinned process die after a matching durable intent despite container UID translation?"""
    from process_probe import kill_frontend

    child, directory = frontend_child
    kill_frontend(directory, 2)
    assert child.wait(timeout=2) == -signal.SIGKILL
    intent = json.loads((directory / "injection_intent.json").read_text())
    death = json.loads((directory / "death.json").read_text())
    assert intent["target"]["pid"] == child.pid and intent["target"]["uid"] == os.getuid()
    assert death["target"] == intent["target"]


def test_frontend_refuses_another_allocation(frontend_child, monkeypatch) -> None:
    """Can matching same-user metadata authorize a kill from a different allocation?"""
    from process_probe import kill_frontend

    child, directory = frontend_child
    monkeypatch.setenv("SLURM_JOB_ID", "456")
    with pytest.raises(ValueError, match="owned step"):
        kill_frontend(directory, 2)
    assert child.poll() is None and not (directory / "injection_intent.json").exists()


def test_frontend_pidfd_error_is_not_death_proof(frontend_child, monkeypatch) -> None:
    """Does a pidfd error without readable exit status incorrectly become recorded death?"""
    import select
    from types import SimpleNamespace

    import process_probe

    _, directory = frontend_child
    descriptors = []
    monkeypatch.setattr(
        process_probe.select,
        "poll",
        lambda: SimpleNamespace(
            register=lambda fd, events: descriptors.append(fd),
            poll=lambda timeout: [(descriptors[0], select.POLLERR)],
        ),
    )
    with pytest.raises(TimeoutError, match="did not exit"):
        process_probe.kill_frontend(directory, 2)
    assert (directory / "injection_intent.json").exists() and not (
        directory / "death.json"
    ).exists()


@pytest.mark.parametrize("phase", ["ready", "archived", "prepare_tmp", "cleanup"])
def test_interrupted_cleanup_preserves_original_evidence_and_rechecks_storage(
    temporary_run, monkeypatch, phase
):
    """Can independent rescue remove pinned storage while retaining immutable failure evidence?"""
    import process_loss_controller
    import process_probe

    directory, path, host = temporary_run
    run = json.loads((directory / "run.json").read_text())
    before = (directory / "run.json").read_bytes()
    config = {
        "tensor_parallel_size": 4,
        "moe_expert_parallel_size": 4,
        "moe_tensor_parallel_size": 1,
        "gpus_per_node": 4,
    }
    (directory / "config.yaml").write_text(json.dumps(config))
    (directory / "build_manifest.json").write_text(
        json.dumps({"driver_report": "NVIDIA GB200, 580.126.20"})
    )
    for name in ("process_probe.py", "deployment_profile.py"):
        shutil.copy2(Path(__file__).with_name(name), directory / name)
    baseline = {
        "hostname": host,
        "compute_apps": [],
        "memory_used_mib": {f"gpu-{n}": 0 for n in range(4)},
        "gpu_memory_csv": "\n".join(
            f"NVIDIA GB200, gpu-{n}, 580.126.20, 100, 0, {n}" for n in range(4)
        ),
    }
    record(directory, f"baseline/{host}", baseline)
    pending = {}
    original_record = process_probe.record
    if phase != "prepare_tmp":
        process_probe.prepare_temporary(directory)
        logs = path / "ray/session_test/logs"
        logs.mkdir(parents=True)
        (logs / "worker.log").write_text("preserved\n")
    if phase == "cleanup":

        def delay_receipt(destination, name, value):
            if name == f"temporary_cleanup/{host}":
                pending["receipt"] = (destination, name, value)
            else:
                original_record(destination, name, value)

        monkeypatch.setattr(process_probe, "record", delay_receipt)
    if phase in {"archived", "cleanup"}:
        process_probe.cleanup_temporary(directory)
    if phase == "cleanup":
        assert not path.exists() and not (directory / f"temporary_cleanup/{host}.json").exists()
    monkeypatch.setattr(process_probe, "resources", lambda: baseline)
    waiting = phase in {"prepare_tmp", "cleanup"}
    monkeypatch.setattr(
        process_loss_controller,
        "discover_owned_step",
        lambda job, name, timeout: "7"
        if waiting and name == f"wideep-probe-{phase}-{job}"
        else None,
    )

    def wait_step(job, step, timeout):
        nonlocal waiting
        assert job == run["job_id"] and step == "7" and timeout > 0
        if phase == "prepare_tmp":
            process_probe.prepare_temporary(directory)
        else:
            monkeypatch.setattr(process_probe, "record", original_record)
            original_record(*pending["receipt"])
        waiting = False
        return {"state": "COMPLETED", "exit_code": "0:0"}

    monkeypatch.setattr(process_loss_controller, "wait_step_terminal", wait_step, raising=False)

    def host_probe(destination, action, timeout):
        assert action == "rescue" and not waiting, "Rescue raced an unfinished host probe"
        record(
            destination,
            f"rescue_cleanup/{host}",
            process_probe.cleanup_snapshot(baseline, [], run["job_id"], [], 64),
        )
        process_probe.cleanup_temporary(destination, rescue=True)

    monkeypatch.setattr(process_loss_controller, "host_probe", host_probe)
    process_loss_controller.rescue_cleanup(directory, 10)
    assert (directory / "run.json").read_bytes() == before
    assert not path.exists()
    proof = json.loads((directory / "interruption_cleanup.json").read_text())
    assert proof["state"] == "verified"
    process_probe.validate_cleanup(directory, allow_partial=True, rescue=True)
    resource_path = directory / f"rescue_cleanup/{host}.json"
    contradictory = json.loads(resource_path.read_text())
    contradictory["memory_used_mib"]["gpu-0"] = 128
    resource_path.write_text(json.dumps(contradictory))
    with pytest.raises(ValueError, match="GPU memory"):
        process_probe.validate_cleanup(directory, allow_partial=True, rescue=True)


@pytest.mark.parametrize("launcher", ["probe", "pre_exec_controller"])
def test_rescue_waits_local_launcher_before_slurm_registration(tmp_path, monkeypatch, launcher):
    """Can an orphan still create a probe step after rescue has certified storage absence?"""
    import process_loss_controller

    directory = tmp_path / "process-loss-123"
    process = tmp_path / "456"
    process.mkdir()
    argv = (
        [
            "srun",
            "--job-name=wideep-probe-prepare_tmp-123",
            str(directory / "process_probe.py"),
            "prepare_tmp",
            "--directory",
            str(directory),
        ]
        if launcher == "probe"
        else [
            "python3",
            str(directory / "process_loss_controller.py"),
            "--output-dir",
            str(directory),
        ]
    )
    (process / "cmdline").write_bytes("\0".join(argv).encode())
    original_iterdir = Path.iterdir
    monkeypatch.setattr(
        Path,
        "iterdir",
        lambda path: iter([process]) if path == Path("/proc") else original_iterdir(path),
    )
    pending = True
    monkeypatch.setattr(
        process_loss_controller,
        "process_identity",
        lambda pid: {"state": "R" if pending else "Z", "cgroup": "0::/job_123/step_batch/"},
    )

    def complete_launch(interval):
        nonlocal pending
        assert interval > 0
        pending = False

    monkeypatch.setattr(process_loss_controller.time, "sleep", complete_launch)

    def discover(job, name, timeout):
        assert not pending, "Rescue inspected Slurm before the orphan launcher finished"
        return None

    monkeypatch.setattr(process_loss_controller, "discover_owned_step", discover)
    process_loss_controller.wait_host_probes(directory, "123", time.monotonic() + 10)
    assert not pending


@pytest.mark.parametrize("fresh_rescue", [False, True])
def test_rescue_does_not_certify_evidence_after_deadline(temporary_run, monkeypatch, fresh_rescue):
    """Can slow evidence validation produce a passing bounded-cleanup receipt?"""
    import process_loss_controller

    directory, _, _ = temporary_run
    now = 0.0
    monkeypatch.setattr(process_loss_controller.time, "monotonic", lambda: now)
    monkeypatch.setattr(process_loss_controller, "discover_owned_step", lambda *args: None)
    monkeypatch.setattr(process_loss_controller, "wait_host_probes", lambda *args: None)
    monkeypatch.setattr(process_loss_controller, "host_probe", lambda *args: None)

    def validate(destination, *, allow_partial, rescue=False):
        nonlocal now
        if fresh_rescue and not rescue:
            raise ValueError("Original cleanup is incomplete")
        now = 11.0

    monkeypatch.setattr(process_loss_controller, "validate_cleanup", validate)
    with pytest.raises(TimeoutError, match="exceeded the cleanup deadline"):
        process_loss_controller.rescue_cleanup(directory, 10)
    assert not (directory / "interruption_cleanup.json").exists()


def test_interrupted_cleanup_rejects_foreign_run_before_mutating_files(temporary_run, monkeypatch):
    """Can rescue rewrite evidence or touch storage from another allocation?"""
    from process_loss_controller import rescue_cleanup

    directory, _, _ = temporary_run
    monkeypatch.setenv("SLURM_JOB_ID", "456")
    with pytest.raises(ValueError, match="original run"):
        rescue_cleanup(directory, 10)
    assert not (directory / "interrupted_cleanup").exists()
