# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Test-only worker faults and independent CPU resource observations."""

import json
import os
import re
import signal
import socket
import subprocess
import tempfile
import threading
import time
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tensorrt_llm._torch.moe.fused_moe.communication.nvlink_one_sided import NVLinkOneSided
    from tensorrt_llm.executor.base_worker import BaseWorker

IDENTITY_KEYS = ("rank", "hostname", "pid", "start_ticks", "boot_id", "uid", "pid_namespace")
RUN_ID_ENV = "WIDEEP_FT_RUN_ID"
IDENTITY_DIR_ENV = "WIDEEP_FT_IDENTITY_DIR"
SCENARIOS = ("worker_sigkill_idle", "worker_sigkill_streaming", "fence_round_mismatch")


def process_identity(pid: int) -> dict:
    """Read process identity in the caller's PID and UID namespaces."""
    process = Path("/proc") / str(pid)
    fields = (process / "stat").read_text().rsplit(")", 1)[1].split()
    return {
        "hostname": socket.gethostname(),
        "pid": pid,
        "start_ticks": int(fields[19]),
        "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
        "uid": process.stat().st_uid,
        "state": fields[0],
        "pid_namespace": (process / "ns/pid").stat().st_ino,
    }


def worker_processes(workers: Sequence[dict]) -> list[dict]:
    """Read known worker identities without inspecting unrelated process environments."""
    processes = []
    for expected in workers:
        if expected["hostname"] != socket.gethostname():
            continue
        try:
            actual = process_identity(expected["pid"])
        except (FileNotFoundError, ProcessLookupError):
            continue
        if actual["uid"] != os.getuid():
            raise PermissionError(f"Worker PID {expected['pid']} is not owned by this observer")
        if any(actual[key] != expected[key] for key in ("start_ticks", "boot_id", "pid_namespace")):
            continue
        processes.append(actual)
    return processes


def record(directory: Path, name: str, value: dict) -> None:
    """Publish complete evidence once; producer monotonic clocks are host-local."""
    path = directory / f"{name}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        json.dump(
            {
                "producer_host": socket.gethostname(),
                "monotonic_s": time.monotonic(),
                "utc_epoch_s": time.time(),
                "clock_source": "producer CLOCK_MONOTONIC",
                **value,
            },
            stream,
            indent=2,
        )
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.link(temporary, path)
        descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        temporary.unlink()


def validate_injection(expected: dict, actual: dict, scenario: str, trigger: dict) -> None:
    """Reject stale identity, root targets, and triggers from another run or boundary."""
    if scenario not in SCENARIOS or type(actual.get("rank")) is not int or actual["rank"] <= 0:
        raise ValueError("Unsupported scenario or target rank")
    if any(
        expected.get(key) is None or expected[key] != actual.get(key) for key in IDENTITY_KEYS
    ) or (expected.get("gpu_uuid") != actual.get("gpu_uuid")):
        raise ValueError("Target identity changed or incomplete")
    if trigger.get("run_id") != expected.get("run_id") or not expected.get("run_id"):
        raise ValueError("Trigger belongs to another run")
    if scenario == "worker_sigkill_streaming":
        if (
            trigger.get("event") != "first_nonfinal_output"
            or not trigger.get("token_ids")
            or trigger.get("finished") is not False
        ):
            raise ValueError("Streaming trigger requires nonfinal output")
    elif trigger.get("event") != "between_requests":
        raise ValueError("Idle trigger requires completed healthy requests")


def _nvlink_communication(worker: "BaseWorker") -> "NVLinkOneSided":
    from tensorrt_llm._torch.moe.fused_moe.communication.nvlink_one_sided import NVLinkOneSided

    for module in worker.engine.model_engine.model.modules():
        comm = getattr(module, "comm", None)
        if isinstance(comm, NVLinkOneSided):
            return comm
    raise ValueError("This profile requires NVLinkOneSided communication")


def worker_identity(worker: "BaseWorker") -> dict:
    """Record native worker placement, captured graphs, and the faulted communicator."""
    import torch

    runner = worker.engine.model_engine.cuda_graph_runner
    comm = _nvlink_communication(worker)
    communication = {
        "NVLinkOneSided": {
            "ep_size": comm.ep_size,
            "ep_rank": comm.ep_rank,
            "cft_capable": comm.can_use_cft_counted_writes,
        }
    }
    return {
        **process_identity(os.getpid()),
        "rank": worker.rank,
        "run_id": os.environ[RUN_ID_ENV],
        "gpu_uuid": "GPU-"
        + str(torch.cuda.get_device_properties(comm.workspace.device).uuid).removeprefix("GPU-"),
        "graphs_enabled": runner.enabled,
        "graph_keys": [str(key) for key in runner.graphs],
        "communication": communication,
        "startup_metrics": worker.get_startup_metrics(),
        "resolved_config": worker.llm_args.model_dump(mode="json") if worker.rank == 0 else None,
    }


def inject_fault(worker: "BaseWorker", expected: dict, directory: Path, scenario: str) -> None:
    """Revalidate this worker and durably record intent before issuing one action."""
    trigger = json.loads((directory / "trigger.json").read_text())
    actual = worker_identity(worker)
    validate_injection(expected, actual, scenario, trigger)
    record(
        directory, "injection_intent", {"scenario": scenario, "target": actual, "trigger": trigger}
    )
    if scenario.startswith("worker_sigkill_"):
        os.kill(os.getpid(), signal.SIGKILL)
        raise AssertionError("SIGKILL returned")

    import torch

    comm = _nvlink_communication(worker)
    if comm.can_use_cft_counted_writes:
        raise ValueError("Fence fault requires a non-CFT NVLinkOneSided workspace")
    with torch.cuda.stream(worker.engine.execution_stream):
        offset = int(comm.moe_a2a_metainfo[comm.FLAG_VAL_OFFSET_INDEX].item())
        flag = comm.workspace[comm.ep_rank, offset : offset + 4].view(torch.int32)
        before = flag.item()
        flag.add_(2)
        worker.engine.execution_stream.synchronize()
        after = flag.item()
    if after != before + 2:
        raise ValueError("Fence counter mutation was not verified")
    record(
        directory,
        "injection_result",
        {
            "scenario": scenario,
            "before": before,
            "after": after,
            "offset": offset,
            "workspace_address": comm.workspace.data_ptr(),
        },
    )


def install_worker_hook() -> None:
    """Instrument native MGMN worker startup and the single selected fault target."""
    from tensorrt_llm.executor.base_worker import BaseWorker

    original = BaseWorker.setup_engine

    def setup_engine(worker: BaseWorker) -> None:
        original(worker)
        directory = Path(os.environ[IDENTITY_DIR_ENV])
        identity = worker_identity(worker)
        record(directory / "workers", str(worker.rank), identity)
        if worker.rank != int(os.environ["WIDEEP_FT_TARGET_RANK"]):
            return

        def watch_fault() -> None:
            command = directory / "fault_request.json"
            while not command.exists():
                time.sleep(0.05)
            try:
                payload = json.loads(command.read_text())
                inject_fault(worker, payload["target"], directory, payload["scenario"])
            except (ValueError, KeyError, OSError, RuntimeError) as error:
                record(directory, "injection_failed", {"error": repr(error)})

        threading.Thread(target=watch_fault, daemon=True, name="fault_injection").start()

    BaseWorker.setup_engine = setup_engine


def _slurm_job_id(process: Path) -> str | None:
    jobs = set(
        re.findall(r"/slurmstepd\.scope/job_(\d+)(?:/|$)", (process / "cgroup").read_text().strip())
    )
    return next(iter(jobs)) if len(jobs) == 1 else None


def owned_processes(
    run_id: str, known: Sequence[dict] = (), *, allow_unreadable: bool = False
) -> list[dict]:
    """Find this run's processes in the observer's PID namespace, including partial startup."""
    marker = f"{RUN_ID_ENV}={run_id}".encode()
    pinned = {row["pid"]: row for row in known if row["hostname"] == socket.gethostname()}
    found = []
    for process in Path("/proc").iterdir():
        if not process.name.isdigit() or process.name == str(os.getpid()):
            continue
        try:
            if process.stat().st_uid != os.getuid():
                continue
            if expected := pinned.get(int(process.name)):
                actual = process_identity(int(process.name))
                if all(
                    actual[key] == expected[key]
                    for key in ("start_ticks", "boot_id", "uid", "pid_namespace")
                ):
                    found.append(actual)
                    continue
            try:
                environment = (process / "environ").read_bytes()
            except PermissionError as error:
                if allow_unreadable:
                    continue
                # Nondumpable Slurm step daemons are accounted for through step lifecycle checks.
                comm = (process / "comm").read_text().strip()
                if comm == "slurmstepd" and re.fullmatch(
                    rb"slurmstepd: \[\d+\.(?:batch|extern|\d+)\]\x00*",
                    (process / "cmdline").read_bytes(),
                ):
                    continue
                if (
                    comm in ("systemd", "(sd-pam)")
                    and (process / "cgroup").read_text().strip()
                    == f"0::/user.slice/user-{os.getuid()}.slice/user@{os.getuid()}.service/init.scope"
                    and (
                        (
                            comm == "systemd"
                            and (process / "cmdline").read_bytes()
                            in (
                                b"/usr/lib/systemd/systemd\0--user\0",
                                b"/lib/systemd/systemd\0--user\0",
                            )
                        )
                        or (
                            comm == "(sd-pam)"
                            and re.fullmatch(
                                rb"\(sd-pam\)\x00*", (process / "cmdline").read_bytes()
                            )
                        )
                    )
                ):
                    continue
                if job_id := os.environ.get("SLURM_JOB_ID"):
                    observer_job = _slurm_job_id(Path("/proc") / str(os.getpid()))
                    process_job = _slurm_job_id(process)
                    if observer_job == job_id and process_job is not None and process_job != job_id:
                        continue
                raise PermissionError(
                    f"Unreadable process {process.name}: comm={comm!r}, "
                    f"cgroup={(process / 'cgroup').read_text()!r}, "
                    f"command={(process / 'cmdline').read_bytes() if comm in ('systemd', '(sd-pam)') else None!r}"
                ) from error
            if marker in environment.split(b"\0"):
                actual = process_identity(int(process.name))
                environment = (process / "environ").read_bytes()
                verified = process_identity(int(process.name))
                if (
                    actual["uid"] == os.getuid()
                    and marker in environment.split(b"\0")
                    and all(
                        actual[key] == verified[key]
                        for key in ("start_ticks", "boot_id", "uid", "pid_namespace")
                    )
                ):
                    found.append(actual)
        except (FileNotFoundError, ProcessLookupError):
            continue
    return found


def snapshot(run_id: str, terminate: bool = False, known: Sequence[dict] = ()) -> dict:
    """Never opens CUDA; forced cleanup signals only freshly verified, owned run processes."""
    if terminate:
        for expected in owned_processes(run_id, known):
            descriptor = None
            try:
                descriptor = os.pidfd_open(expected["pid"])
                actual = process_identity(expected["pid"])
                if all(
                    actual[key] == expected[key]
                    for key in ("start_ticks", "boot_id", "uid", "pid_namespace")
                ):
                    signal.pidfd_send_signal(descriptor, signal.SIGKILL)
            except (FileNotFoundError, ProcessLookupError):
                continue
            finally:
                if descriptor is not None:
                    os.close(descriptor)

    def query(fields: str, kind: str) -> list[list[str]]:
        output = subprocess.run(
            ["nvidia-smi", f"--query-{kind}={fields}", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        ).stdout
        return [[field.strip() for field in line.split(",")] for line in output.splitlines()]

    return {
        "run_id": run_id,
        "hostname": socket.gethostname(),
        "gpus": query("uuid,memory.used,driver_version,name", "gpu"),
        "compute_apps": query("gpu_uuid,pid,used_gpu_memory", "compute-apps"),
        "live_processes": owned_processes(run_id, known),
    }


def resources_released(baseline: list[dict], current: list[dict], tolerance_mib: int = 64) -> bool:
    """Require complete host/GPU observations, no owned processes, and bounded memory delta."""
    if (
        not baseline
        or len(baseline) != len(current)
        or any(row.get("observation_scope", "full") != "full" for row in (*baseline, *current))
    ):
        return False
    before = {row["hostname"]: row for row in baseline}
    if len(before) != len(baseline) or {row["hostname"] for row in current} != set(before):
        return False
    for row in current:
        original = {gpu[0]: int(gpu[1]) for gpu in before[row["hostname"]]["gpus"]}
        after = {gpu[0]: int(gpu[1]) for gpu in row["gpus"]}
        if (
            not original
            or row["compute_apps"]
            or row["live_processes"]
            or set(original) != set(after)
            or any(after[uuid] - used > tolerance_mib for uuid, used in original.items())
        ):
            return False
    return True
