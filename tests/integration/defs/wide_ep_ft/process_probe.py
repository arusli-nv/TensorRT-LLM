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
"""Test-only Ray actor injection and independent host resource observations."""

import argparse
import faulthandler
import json
import os
import re
import select
import shutil
import signal
import socket
import stat
import subprocess
import tarfile
import tempfile
import time
from pathlib import Path

IDENTITY_KEYS = ("rank", "hostname", "pid", "start_ticks", "boot_id", "uid", "actor_id")
WORKER_SIGNALS = {
    "idle_kill": signal.SIGKILL,
    "stream_kill": signal.SIGKILL,
    "rank0_kill": signal.SIGKILL,
    "idle_term": signal.SIGTERM,
    "idle_stop": signal.SIGSTOP,
}


def process_identity(pid: int) -> dict:
    """Pin a Linux process to its host boot, UID and start time, without MPI variables."""
    process = Path("/proc") / str(pid)
    fields = (process / "stat").read_text().rsplit(")", 1)[1].split()
    return {
        "hostname": socket.gethostname(),
        "pid": pid,
        "start_ticks": int(fields[19]),
        "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
        "uid": process.stat().st_uid,
        "state": fields[0],
        "cgroup": (process / "cgroup").read_text(),
    }


def write_record(path: Path, value: object) -> None:
    """Publish one complete record exclusively, without overwriting an earlier attempt."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("x") as stream:
        json.dump(value, stream, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.link(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink()


def record(directory: Path, name: str, value: dict) -> None:
    """Publish immutable evidence; monotonic timestamps are comparable only on this host."""
    write_record(
        directory / f"{name}.json",
        {
            "run_id": json.loads((directory / "run.json").read_text())["run_id"]
            if (directory / "run.json").exists()
            else directory.name,
            "clock_source": "CLOCK_MONOTONIC on producer host; UTC is informational",
            "producer_host": socket.gethostname(),
            "monotonic_s": time.monotonic(),
            "utc_epoch_s": time.time(),
            **value,
        },
    )


def _temporary_path(job_id: str) -> Path:
    if not job_id.isdecimal() or os.environ.get("SLURM_JOB_ID") != job_id:
        raise ValueError("Temporary directory requires the owned numeric allocation")
    return Path("/tmp") / f"wideep-ray-{job_id}"


def _temporary_identity(path: Path) -> dict:
    metadata = path.lstat()
    if not stat.S_ISDIR(metadata.st_mode) or metadata.st_uid != os.getuid():
        raise ValueError("Temporary directory must be a real owned directory")
    return {
        "path": str(path),
        "device": metadata.st_dev,
        "inode": metadata.st_ino,
        "uid": metadata.st_uid,
    }


def prepare_temporary(directory: Path) -> None:
    """Create and pin only this allocation's new node-local temporary directory."""
    run = json.loads((directory / "run.json").read_text())
    if run["run_id"] != directory.name:
        raise ValueError("Temporary directory evidence belongs to another run")
    path = _temporary_path(run["job_id"])
    path.mkdir(mode=0o700)
    identity = _temporary_identity(path)
    try:
        record(
            directory,
            f"temporary/{socket.gethostname()}",
            {"job_id": run["job_id"], **identity},
        )
    except (OSError, ValueError):
        try:
            if _temporary_identity(path) == identity:
                path.rmdir()
        except (OSError, ValueError):
            # A changed or populated tree requires independent cleanup verification.
            pass
        raise


def resources_clean(snapshot: dict) -> bool:
    return not any(
        snapshot[key] for key in ("original_live", "owned_live", "owned_gpu_contexts")
    ) and all(
        delta <= snapshot["memory_tolerance_mib"] for delta in snapshot["memory_delta_mib"].values()
    )


def cleanup_temporary(directory: Path, *, rescue: bool = False) -> None:
    """Preserve Ray logs and remove the pinned tree only after independent resource cleanup."""
    from deployment_profile import file_sha256

    host = socket.gethostname()
    run = json.loads((directory / "run.json").read_text())
    prefix = "rescue_" if rescue else ""
    resources_record = json.loads((directory / f"{prefix}cleanup" / f"{host}.json").read_text())
    if (
        run["run_id"] != directory.name
        or resources_record["run_id"] != directory.name
        or resources_record["hostname"] != host
        or not resources_record["clean"]
        or not resources_clean(resources_record)
    ):
        raise ValueError("Temporary cleanup requires verified process and GPU cleanup")
    path = _temporary_path(run["job_id"])
    ownership = directory / "temporary" / f"{host}.json"
    if not ownership.exists():
        if os.path.lexists(path):
            raise ValueError("Refusing to remove an unpinned temporary directory")
        record(
            directory,
            f"{prefix}temporary_cleanup/{host}",
            {"job_id": run["job_id"], "path": str(path), "state": "never_created"},
        )
        return
    expected = json.loads(ownership.read_text())
    if not os.path.lexists(path):
        previous = json.loads((directory / "temporary_cleanup" / f"{host}.json").read_text())
        if (
            previous["state"] != "absent"
            or previous["job_id"] != run["job_id"]
            or any(previous[key] != expected[key] for key in ("path", "device", "inode", "uid"))
            or file_sha256(directory / previous["archive"]) != previous["archive_sha256"]
        ):
            raise ValueError("Missing owned storage lacks verified preserved logs")
        record(
            directory,
            f"{prefix}temporary_cleanup/{host}",
            {
                **{
                    key: previous[key]
                    for key in (
                        "job_id",
                        "path",
                        "device",
                        "inode",
                        "uid",
                        "archive",
                        "archive_sha256",
                    )
                },
                "state": "absent",
                "reused_archive": True,
            },
        )
        return
    actual = _temporary_identity(path)
    if (
        expected["run_id"] != directory.name
        or expected["producer_host"] != host
        or expected["job_id"] != run["job_id"]
        or any(actual[key] != expected[key] for key in actual)
    ):
        raise ValueError("Pinned temporary directory identity changed")
    archive = directory / "ray_logs" / f"{host}{'-rescue' if rescue else ''}.tar.gz"
    archive.parent.mkdir(exist_ok=True)
    sessions = path / "ray"
    if sessions.is_symlink():
        raise ValueError("Ray session root must not be a symlink")
    with tarfile.open(archive, "x:gz", dereference=False) as stream:
        for session in sorted(sessions.glob("session_*")):
            logs = session / "logs"
            if session.is_symlink():
                continue
            if logs.is_symlink():
                raise ValueError("Ray log directory must not be a symlink")
            if logs.is_dir():
                stream.add(logs, arcname=str(logs.relative_to(path)))
    if _temporary_identity(path) != actual:
        raise ValueError("Pinned temporary directory changed during log preservation")
    shutil.rmtree(path)
    if os.path.lexists(path):
        raise RuntimeError("Owned temporary directory remains after cleanup")
    record(
        directory,
        f"{prefix}temporary_cleanup/{host}",
        {
            "job_id": run["job_id"],
            **actual,
            "state": "absent",
            "archive": str(archive.relative_to(directory)),
            "archive_sha256": file_sha256(archive),
        },
    )


def validate_cleanup(directory: Path, *, allow_partial: bool = False, rescue: bool = False) -> None:
    """Reconcile independent process/GPU/storage evidence, including interrupted startup."""
    import yaml
    from deployment_profile import deployment_shape, file_sha256, validate_hardware

    def read(name: str) -> dict:
        value = json.loads((directory / f"{name}.json").read_text())
        if value["run_id"] != directory.name:
            raise ValueError(f"Stale {name} evidence")
        return value

    run = read("run")
    config = yaml.safe_load((directory / "config.yaml").read_text())
    manifest = json.loads((directory / "build_manifest.json").read_text())
    _, node_count, _ = deployment_shape(config)
    prefix = "rescue_" if rescue else ""
    resources = [
        json.loads(path.read_text()) for path in (directory / f"{prefix}cleanup").glob("*.json")
    ]
    baseline = [json.loads(path.read_text()) for path in (directory / "baseline").glob("*.json")]
    if any(row["run_id"] != directory.name for row in baseline):
        raise ValueError("Stale cleanup baseline")
    hosts = {row["hostname"] for row in baseline}
    validate_hardware(baseline, hosts, config, manifest)
    validate_hardware(resources, hosts, config, manifest)
    for row in resources:
        before = next(report for report in baseline if report["hostname"] == row["hostname"])
        readings = []
        for report in (before, row):
            memory = {
                fields[1]: int(fields[4])
                for fields in (line.split(", ") for line in report["gpu_memory_csv"].splitlines())
            }
            if memory != report["memory_used_mib"]:
                raise ValueError("GPU memory summary differs from NVML readings")
            readings.append(memory)
        previous, current = readings
        if set(current) != set(previous) or row["memory_delta_mib"] != {
            uuid: used - previous[uuid] for uuid, used in current.items()
        }:
            raise ValueError("GPU memory delta differs from original readings")
        original_contexts = {(app[0], app[1]) for app in before["compute_apps"]}
        if any((app[0], app[1]) not in original_contexts for app in row["compute_apps"]):
            raise ValueError("A new GPU context remains after cleanup")
    if any(row["run_id"] != directory.name or not resources_clean(row) for row in resources):
        raise ValueError("Raw resource checks contradict cleanup")
    if (
        {row["hostname"] for row in resources} != hosts
        or len(resources) != node_count
        or not all(row["clean"] for row in resources)
    ):
        raise ValueError("Resource cleanup is unverified")
    temporary = [
        json.loads(path.read_text())
        for path in (directory / f"{prefix}temporary_cleanup").glob("*.json")
    ]
    if len(temporary) != node_count or {row["producer_host"] for row in temporary} != hosts:
        raise ValueError("Owned temporary storage cleanup is unverified")
    for row in temporary:
        host = row["producer_host"]
        if row["run_id"] != directory.name or row["job_id"] != run["job_id"]:
            raise ValueError("Storage cleanup belongs to another run")
        if row["state"] == "never_created" and allow_partial:
            if (directory / "temporary" / f"{host}.json").exists() or row[
                "path"
            ] != f"/tmp/wideep-ray-{run['job_id']}":
                raise ValueError("Absent storage conflicts with ownership evidence")
            continue
        original = read(f"temporary/{host}")
        if (
            row["run_id"] != directory.name
            or row["state"] != "absent"
            or row["job_id"] != run["job_id"]
            or row["path"] != f"/tmp/wideep-ray-{run['job_id']}"
            or original["producer_host"] != host
            or original["job_id"] != run["job_id"]
            or any(row[key] != original[key] for key in ("path", "device", "inode", "uid"))
            or row["archive"]
            not in (
                {f"ray_logs/{host}.tar.gz", f"ray_logs/{host}-rescue.tar.gz"}
                if rescue
                else {f"ray_logs/{host}.tar.gz"}
            )
            or file_sha256(directory / row["archive"]) != row["archive_sha256"]
        ):
            raise ValueError("Temporary cleanup differs from owned storage or retained Ray logs")


def validate_injection(expected: dict, actual: dict, scenario: str, trigger: dict) -> None:
    """Reject identity drift and an unproven named trigger before issuing a signal."""
    if any(
        key not in expected or key not in actual or expected[key] is None or actual[key] is None
        for key in IDENTITY_KEYS
    ):
        raise ValueError("Incomplete process identity")
    if scenario not in WORKER_SIGNALS or (actual["rank"] == 0) != (scenario == "rank0_kill"):
        raise ValueError("Unsupported scenario or target rank")
    if any(expected.get(key) != actual.get(key) for key in IDENTITY_KEYS):
        raise ValueError("Target identity changed")
    if trigger.get("run_id") != expected.get("run_id"):
        raise ValueError("Trigger belongs to another run")
    if scenario == "stream_kill":
        if (
            trigger.get("event") != "first_nonfinal_output"
            or not trigger.get("token_ids")
            or trigger.get("finished") is not False
        ):
            raise ValueError("Streaming injection requires copied nonfinal output")
    elif trigger.get("event") != "healthy_between_requests":
        raise ValueError("Between-request injection requires healthy completion")


def kill_frontend(directory: Path, timeout: float) -> None:
    """Kill only the published frontend process, using a pinned host pidfd."""
    run = json.loads((directory / "run.json").read_text())
    step = json.loads((directory / "step-driver.json").read_text())
    frontend = json.loads((directory / "frontend.json").read_text())
    trigger = json.loads((directory / "trigger.json").read_text())
    if (
        any(row["run_id"] != directory.name for row in (run, step, frontend, trigger))
        or run["scenario"] != "frontend_kill"
        or step["job_id"] != run["job_id"]
        or step["job_id"] != os.environ.get("SLURM_JOB_ID")
        or not step["job_id"].isdecimal()
        or not step["step_id"].isdecimal()
        or trigger["event"] != "healthy_http_completed"
    ):
        raise ValueError("Frontend signal requires this run's owned step and HTTP trigger")
    expected = frontend["identity"]
    pins = ("hostname", "pid", "start_ticks", "boot_id")
    if any(expected[key] != step["identity"][key] for key in pins):
        raise ValueError("Frontend does not identify the originally published driver")
    if (
        expected["hostname"] != socket.gethostname()
        or expected["pid"] <= 1
        or expected["pid"] == os.getpid()
    ):
        raise ValueError("Frontend must be another process on the controller host")
    descriptor = os.pidfd_open(expected["pid"])
    try:
        actual = process_identity(expected["pid"])
        if (
            any(actual[key] != expected[key] for key in pins)
            or actual["uid"] != os.getuid()
            or not re.search(
                rf"/job_{step['job_id']}/step_{step['step_id']}(?:/|$)", actual["cgroup"]
            )
        ):
            raise ValueError("Frontend host identity or owned cgroup changed")
        # Host UID/cgroup are resolved explicitly; container namespace values may differ.
        record(directory, "injection", {"target": actual, "signal": "SIGKILL"})
        record(
            directory,
            "injection_intent",
            {
                "target": actual,
                "signal": "SIGKILL",
                "scenario": "frontend_kill",
                "trigger": trigger,
            },
        )
        started = time.monotonic()
        signal.pidfd_send_signal(descriptor, signal.SIGKILL)
        poller = select.poll()
        poller.register(descriptor, select.POLLIN)
        events = poller.poll(int(timeout * 1000))
        if not any(fd == descriptor and mask & select.POLLIN for fd, mask in events):
            raise TimeoutError("Pinned frontend process did not exit")
        record(
            directory,
            "death",
            {
                "target": actual,
                "proof": "pidfd readable after SIGKILL",
                "seconds_since_signal": time.monotonic() - started,
            },
        )
    finally:
        os.close(descriptor)


class WorkerProbe:
    """Native Ray worker extension. Replies are diagnostics, never recovery authority."""

    def wideep_identity(self) -> dict:
        import ray
        import torch

        runner = self.engine.model_engine.cuda_graph_runner
        return {
            **process_identity(os.getpid()),
            "rank": self.rank,
            "actor_id": str(ray.get_runtime_context().get_actor_id()),
            "node_ip": ray.util.get_node_ip_address(),
            "assigned_gpus": ray.get_gpu_ids(),
            "current_device": torch.cuda.current_device(),
            "graphs_enabled": runner.enabled,
            "graph_keys": [str(key) for key in runner.graphs],
            "startup_metrics": self.get_startup_metrics(),
            "resolved": self.llm_args.model_dump(mode="json", exclude={"ray_placement_config"}),
        }

    def wideep_cuda_probe(self, directory: str) -> dict:
        import torch

        identity = process_identity(os.getpid())
        with tempfile.TemporaryFile(mode="w+") as trace:
            faulthandler.dump_traceback(file=trace, all_threads=True)
            trace.seek(0)
            stacks = trace.read()
        error = self.engine._event_loop_error
        record(
            Path(directory),
            f"diagnostics/rank-{self.rank:02d}",
            {
                **identity,
                "rank": self.rank,
                "event_loop_error": None if error is None else str(error),
                "thread_stacks": stacks,
                "scope": "Host thread snapshot before the independent CUDA operation",
            },
        )
        return {
            **identity,
            "rank": self.rank,
            "cuda_value": torch.ones(1, device=f"cuda:{self.device_id}").sum().item(),
        }

    def wideep_signal(self, expected: dict, directory: str, scenario: str) -> None:
        path = Path(directory)
        trigger = json.loads((path / "trigger.json").read_text())
        actual = self.wideep_identity()
        validate_injection(expected, actual, scenario, trigger)
        action = WORKER_SIGNALS[scenario]
        record(
            path,
            "injection_intent",
            {
                "target": {key: actual[key] for key in IDENTITY_KEYS},
                "scenario": scenario,
                "signal": action.name,
                "trigger": trigger,
            },
        )
        # Exclusive intent publication above also prevents a second action for this run.
        os.kill(os.getpid(), action)


def resources() -> dict:
    """Read NVML-backed GPU accounting without opening a CUDA context."""

    def query(arguments: list[str]) -> str:
        return subprocess.run(
            ["nvidia-smi", *arguments], capture_output=True, text=True, check=True, timeout=10
        ).stdout

    gpu_csv = query(
        [
            "--query-gpu=name,uuid,driver_version,memory.total,memory.used,index",
            "--format=csv,noheader,nounits",
        ]
    )
    apps_csv = query(
        ["--query-compute-apps=gpu_uuid,pid,used_gpu_memory", "--format=csv,noheader,nounits"]
    )
    return {
        "hostname": socket.gethostname(),
        "gpu_memory_csv": gpu_csv,
        "compute_apps": [line.split(", ") for line in apps_csv.splitlines()],
        "memory_used_mib": {
            row.split(", ")[1]: int(row.split(", ")[4]) for row in gpu_csv.splitlines()
        },
    }


def cleanup_snapshot(
    baseline: dict, identities: list[dict], job_id: str, step_ids: list[str], tolerance_mib: int
) -> dict:
    """Require original processes, owned step processes and GPU allocations to be gone."""
    current = resources()
    original_live = []
    for expected in identities:
        if expected["hostname"] != current["hostname"]:
            continue
        try:
            actual = process_identity(expected["pid"])
        except (FileNotFoundError, ProcessLookupError):
            continue
        # Container user/cgroup namespaces can report UID 0 and cgroup /.
        # PID/start time/boot identify the process in the shared PID namespace.
        if all(actual[key] == expected[key] for key in ("start_ticks", "boot_id")):
            original_live.append(actual)
    owned_live = []
    for process in Path("/proc").iterdir():
        if not process.name.isdecimal():
            continue
        try:
            cgroup = (process / "cgroup").read_text()
            if any(re.search(rf"/job_{job_id}/step_{step}(?:/|$)", cgroup) for step in step_ids):
                owned_live.append(process_identity(int(process.name)))
        except (FileNotFoundError, ProcessLookupError):
            continue
    baseline_contexts = {(row[0], row[1]) for row in baseline["compute_apps"]}
    gpu_live = [row for row in current["compute_apps"] if (row[0], row[1]) not in baseline_contexts]
    if set(current["memory_used_mib"]) != set(baseline["memory_used_mib"]):
        raise ValueError("GPU identity changed during cleanup")
    deltas = {
        uuid: used - baseline["memory_used_mib"][uuid]
        for uuid, used in current["memory_used_mib"].items()
    }
    snapshot = {
        **current,
        "original_live": original_live,
        "owned_live": owned_live,
        "owned_gpu_contexts": gpu_live,
        "memory_delta_mib": deltas,
        "memory_tolerance_mib": tolerance_mib,
    }
    return {**snapshot, "clean": resources_clean(snapshot)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action", choices=("baseline", "prepare_tmp", "cleanup", "rescue", "stopped")
    )
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=180)
    parser.add_argument("--memory-tolerance-mib", type=int, default=64)
    args = parser.parse_args()
    host = socket.gethostname()
    if args.action == "baseline":
        record(args.directory, f"baseline/{host}", resources())
        return
    if args.action == "prepare_tmp":
        prepare_temporary(args.directory)
        return
    if args.action == "stopped":
        intent = json.loads((args.directory / "injection_intent.json").read_text())
        expected = intent["target"]
        if intent["run_id"] != args.directory.name or intent["signal"] != "SIGSTOP":
            raise ValueError("Stall confirmation requires this run's SIGSTOP intent")
        if expected["hostname"] != host:
            return
        deadline = time.monotonic() + args.timeout
        while True:
            actual = process_identity(expected["pid"])
            if any(actual[key] != expected[key] for key in ("hostname", "start_ticks", "boot_id")):
                raise ValueError("Stalled process identity changed")
            if actual["state"] == "T":
                record(args.directory, "stopped", {"target": expected, "observed": actual})
                return
            if time.monotonic() >= deadline:
                raise TimeoutError("Target never entered OS stopped state")
            time.sleep(0.1)
    prefix = "rescue_" if args.action == "rescue" else ""
    baseline = json.loads((args.directory / f"baseline/{host}.json").read_text())
    workers_file = args.directory / "workers.json"
    workers = json.loads(workers_file.read_text())["workers"] if workers_file.exists() else []
    step_records = [json.loads(path.read_text()) for path in args.directory.glob("step-*.json")]
    job_id = os.environ["SLURM_JOB_ID"]
    if not job_id.isdecimal() or any(
        row["job_id"] != job_id or not row["step_id"].isdecimal() for row in step_records
    ):
        raise ValueError("Owned step records do not match the allocation")
    step_ids = sorted({row["step_id"] for row in step_records})
    deadline = time.monotonic() + args.timeout
    while True:
        snapshot = cleanup_snapshot(baseline, workers, job_id, step_ids, args.memory_tolerance_mib)
        if snapshot["clean"] or time.monotonic() >= deadline:
            record(args.directory, f"{prefix}cleanup/{host}", snapshot)
            if not snapshot["clean"]:
                raise RuntimeError("Owned processes or GPU memory remain after cleanup")
            cleanup_temporary(args.directory, rescue=args.action == "rescue")
            return
        time.sleep(0.2)


if __name__ == "__main__":
    main()
