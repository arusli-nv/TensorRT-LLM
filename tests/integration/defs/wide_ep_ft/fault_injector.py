# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Test-only worker faults and independent resource observations."""

import json
import os
import shutil
import signal
import socket
import subprocess
import tempfile
import time
from pathlib import Path

IDENTITY_KEYS = ("rank", "hostname", "pid", "start_ticks", "boot_id", "uid", "pid_namespace")
RUN_ID_ENV = "WIDEEP_FT_RUN_ID"
IDENTITY_DIR_ENV = "WIDEEP_FT_IDENTITY_DIR"
SCENARIOS = (
    "worker_sigkill_idle",
    "worker_sigkill_streaming",
    "process_group_destroy",
    "fence_round_mismatch",
)


def process_identity(pid: int) -> dict:
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


def record_worker_identity() -> None:
    """Record immutable process identity before Ray constructs the GPU worker."""
    identity = process_identity(os.getpid())
    record(
        Path(os.environ[IDENTITY_DIR_ENV]) / "startup_identities" / identity["hostname"],
        f"{identity['pid']}-{identity['start_ticks']}",
        {**identity, "run_id": os.environ[RUN_ID_ENV], "job_id": os.environ["RAY_JOB_ID"]},
    )


def validate_injection(expected: dict, actual: dict, scenario: str, trigger: dict) -> None:
    if scenario not in SCENARIOS or type(actual.get("rank")) is not int or actual["rank"] <= 0:
        raise ValueError("Unsupported scenario or target rank")
    if any(expected.get(key) is None or expected[key] != actual.get(key) for key in IDENTITY_KEYS):
        raise ValueError("Target identity changed or incomplete")
    if expected.get("actor_id") is not None and expected["actor_id"] != actual.get("actor_id"):
        raise ValueError("Target actor identity changed")
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


class WorkerExtension:
    def fault_identity(self) -> dict:
        import ray
        import torch

        runner = self.engine.model_engine.cuda_graph_runner
        communication = {}
        for module in self.engine.model_engine.model.modules():
            comm = getattr(module, "comm", None)
            if comm is not None and hasattr(comm, "ep_size"):
                communication[type(comm).__name__] = {
                    "ep_size": comm.ep_size,
                    "ep_rank": comm.ep_rank,
                    "cft_capable": getattr(comm, "can_use_cft_counted_writes", None),
                }
        return {
            **process_identity(os.getpid()),
            "rank": self.rank,
            "actor_id": str(ray.get_runtime_context().get_actor_id()),
            "gpu_uuid": "GPU-"
            + str(torch.cuda.get_device_properties(self.device_id).uuid).removeprefix("GPU-"),
            "graphs_enabled": runner.enabled,
            "graph_keys": [str(key) for key in runner.graphs],
            "startup_metrics": self.get_startup_metrics(),
            "communication": communication,
            "resolved_config": self.llm_args.model_dump(
                mode="json", exclude={"ray_placement_config"}
            )
            if self.rank == 0
            else None,
        }

    def inject_fault(self, expected: dict, directory: str, scenario: str) -> dict:
        path = Path(directory)
        trigger = json.loads((path / "trigger.json").read_text())
        actual = self.fault_identity()
        validate_injection(expected, actual, scenario, trigger)
        record(
            path, "injection_intent", {"scenario": scenario, "target": actual, "trigger": trigger}
        )
        if scenario.startswith("worker_sigkill_"):
            os.kill(os.getpid(), signal.SIGKILL)
        if scenario == "process_group_destroy":
            import torch

            group = torch.distributed.group.WORLD
            backend = group._get_backend(torch.device("cpu"))
            result = {
                "group": "WORLD",
                "backend": type(backend).__name__,
                "group_size": group.size(),
                "tp_group_is_world": group is self.engine.dist.mapping.tp_group_pg,
            }
            torch.distributed.destroy_process_group(group)
        else:
            import torch

            from tensorrt_llm._torch.moe.fused_moe.communication.nvlink_one_sided import (
                NVLinkOneSided,
            )

            comm = next(
                (
                    module.comm
                    for module in self.engine.model_engine.model.modules()
                    if isinstance(getattr(module, "comm", None), NVLinkOneSided)
                ),
                None,
            )
            if comm is None or comm.can_use_cft_counted_writes:
                raise ValueError("Fence fault requires a non-CFT NVLinkOneSided workspace")
            offset = int(comm.moe_a2a_metainfo[comm.FLAG_VAL_OFFSET_INDEX].item())
            flag = comm.workspace[comm.ep_rank, offset : offset + 4].view(torch.int32)
            with torch.cuda.stream(self.engine.execution_stream):
                before = flag.item()
                flag.add_(2)
                self.engine.execution_stream.synchronize()
                result = {
                    "before": before,
                    "after": flag.item(),
                    "offset": offset,
                    "workspace_address": comm.workspace.data_ptr(),
                }
        record(path, "injection_result", {"scenario": scenario, **result})
        return result

    def fault_cuda_probe(self) -> dict:
        import torch

        return {
            "rank": self.rank,
            "value": torch.ones(1, device=f"cuda:{self.device_id}").sum().item(),
        }


class NodeProbe:
    """CPU-only accounting; never opens a CUDA context or kills unowned processes."""

    def pin_actor_processes(self, actors: dict, directory: str) -> list[dict]:
        import ray

        node_id = ray.get_runtime_context().get_node_id()
        path = Path(directory)
        job = json.loads((path / "job.json").read_text())
        identities = []
        for actor_id, actor in actors.items():
            if actor["Address"]["NodeID"] != node_id or actor["Pid"] <= 0:
                continue
            try:
                identity = process_identity(actor["Pid"])
                evidence = json.loads(
                    (
                        path
                        / "startup_identities"
                        / identity["hostname"]
                        / f"{identity['pid']}-{identity['start_ticks']}.json"
                    ).read_text()
                )
                if (
                    evidence["run_id"] != job["run_id"]
                    or evidence["job_id"] != job["job_id"]
                    or any(
                        evidence[key] != identity[key]
                        for key in ("hostname", "pid", "start_ticks", "boot_id", "uid")
                    )
                ):
                    continue
            except (FileNotFoundError, ProcessLookupError):
                continue
            if identity["uid"] != os.getuid():
                raise ValueError("Owned Ray actor process has an unexpected UID")
            identities.append({**identity, "actor_id": actor_id, "run_id": job["run_id"]})
        return identities

    def read_run_id(self, directory: str) -> str:
        return json.loads((Path(directory) / "run.json").read_text())["run_id"]

    def terminate_owned_processes(self, identities: list[dict]) -> list[int]:
        terminated = []
        for expected in identities:
            if expected["hostname"] != socket.gethostname() or expected["pid"] == os.getpid():
                continue
            descriptor = None
            try:
                descriptor = os.pidfd_open(expected["pid"])
                actual = process_identity(expected["pid"])
                if actual["uid"] != os.getuid() or any(
                    actual[key] != expected[key] for key in ("start_ticks", "boot_id", "uid")
                ):
                    continue
                signal.pidfd_send_signal(descriptor, signal.SIGKILL)
                terminated.append(expected["pid"])
            except (FileNotFoundError, ProcessLookupError):
                continue
            finally:
                if descriptor is not None:
                    os.close(descriptor)
        return terminated

    def collect_logs(self, directory: str, job_id: str) -> list[str]:
        from ray._private.worker import global_worker

        logs = Path(global_worker.node.get_logs_dir_path())
        destination = Path(directory) / "ray_logs" / socket.gethostname()
        destination.mkdir(parents=True, exist_ok=True)
        copied = []
        for path in logs.glob(f"worker-*-{job_id}-*.*"):
            if path.suffix in (".out", ".err") and path.is_file():
                shutil.copy2(path, destination / path.name)
                copied.append(path.name)
        return copied

    def snapshot(self, identities: list[dict]) -> dict:
        def query(fields: str, kind: str) -> list[list[str]]:
            output = subprocess.run(
                ["nvidia-smi", f"--query-{kind}={fields}", "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True,
                check=True,
                timeout=10,
            ).stdout
            return [[field.strip() for field in line.split(",")] for line in output.splitlines()]

        gpus = query("uuid,memory.used,driver_version,name", "gpu")
        apps = query("gpu_uuid,pid,used_gpu_memory", "compute-apps")
        live = []
        for expected in identities:
            if expected["hostname"] != socket.gethostname():
                continue
            try:
                actual = process_identity(expected["pid"])
            except (FileNotFoundError, ProcessLookupError):
                continue
            if all(actual[key] == expected[key] for key in ("start_ticks", "boot_id", "uid")):
                live.append(actual)
        return {
            "hostname": socket.gethostname(),
            "gpus": gpus,
            "compute_apps": apps,
            "live_processes": live,
        }


def resources_released(baseline: list[dict], current: list[dict], tolerance_mib: int = 64) -> bool:
    if not baseline or len(baseline) != len(current):
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
