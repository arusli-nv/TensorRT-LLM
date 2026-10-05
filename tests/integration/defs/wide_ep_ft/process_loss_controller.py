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
"""Own one Ray/Slurm deployment, qualify it, then verify bounded cleanup independently."""

import argparse
import http.client
import json
import math
import os
import re
import shutil
import signal
import site
import socket
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path

from deployment_profile import (
    PROMPTS,
    deployment_shape,
    file_sha256,
    normalize_ray_log,
    validate_execution_log,
    validate_hardware,
    validate_profile,
    validate_ray_version,
)
from process_probe import WORKER_SIGNALS, kill_frontend, process_identity, record, validate_cleanup
from slurm_lifecycle import (
    cancel_owned_step,
    cleanup_model_step,
    discover_owned_step,
    wait_step_terminal,
)

OVERLAY_MODULES = (
    "executor/rpc_worker_mixin.py",
    "executor/rpc_proxy_mixin.py",
    "executor/ray/executor.py",
    "executor/rpc/rpc_client.py",
    "llmapi/llm.py",
)
RAY_ADDRESS_FILE = Path("/tmp/ray/ray_current_cluster")


def install_overlay(directory: Path, mode: str) -> None:
    """Install and hash only the explicitly frozen native Python fixes before any imports."""
    manifest = directory / "python_overlay.json"
    if not manifest.exists():
        return
    package = next(
        Path(path) / "tensorrt_llm"
        for path in site.getsitepackages()
        if (Path(path) / "tensorrt_llm").is_dir()
    )
    records = {}
    for relative, digest in json.loads(manifest.read_text())["files"].items():
        if relative not in OVERLAY_MODULES:
            raise ValueError("Unqualified runtime overlay module")
        source, target = directory / "overlay" / relative, package / relative
        if file_sha256(source) != digest:
            raise ValueError("Frozen Python overlay hash changed")
        records[relative] = {"original_sha256": file_sha256(target), "installed_sha256": digest}
        shutil.copy2(source, target)
        if file_sha256(target) != digest:
            raise ValueError("Installed Python overlay does not match manifest")
    record(
        directory,
        f"runtime-{socket.gethostname()}-{mode}",
        {"files": records, "scope": "Base compiled image plus frozen Python fixes"},
    )


def bootstrap(args: argparse.Namespace) -> None:
    import yaml

    _, _, gpus = deployment_shape(yaml.safe_load((args.output_dir / "config.yaml").read_text()))
    rank = int(os.environ["SLURM_PROCID"])
    record(
        args.output_dir,
        f"step-service-{rank}",
        {
            "job_id": os.environ["SLURM_JOB_ID"],
            "step_id": os.environ["SLURM_STEP_ID"],
            "identity": process_identity(os.getpid()),
        },
    )
    install_overlay(args.output_dir, "service")
    for key in list(os.environ):
        if key.startswith(("SLURM_", "PMI", "PMIX", "MPI", "OMPI")):
            del os.environ[key]
    os.environ["RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES"] = "1"
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, range(gpus)))
    command = [
        "/raydeps/venv/bin/ray",
        "start",
        "--disable-usage-stats",
        f"--num-cpus={max(32, gpus)}",
        f"--num-gpus={gpus}",
        "--block",
        "--temp-dir=/tmp/ray",
        f"--node-ip-address={socket.gethostbyname(socket.gethostname())}",
    ]
    if rank == 0:
        command += ["--head", "--port=0", "--include-dashboard=false"]
        head = subprocess.Popen(command)
        try:
            proof = wait_head(args.address, head, time.monotonic() + 120)
            record(
                args.output_dir,
                "head_ready",
                {**proof, "head_cli_identity": process_identity(head.pid)},
            )
            if head.wait() != 0:
                raise RuntimeError("Owned Ray head launcher failed")
        finally:
            if head.poll() is None:
                head.terminate()
                try:
                    head.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    head.kill()
                    head.wait(timeout=5)
        return
    else:
        ready = wait_record(args.output_dir / "head_ready.json", None, time.monotonic() + 120)
        validate_head_ready(ready, args.output_dir, args.address)
        command += [f"--address={ready['address']}"]
    os.execvp(command[0], command)


def wait_head(address: str, child: subprocess.Popen, deadline: float) -> dict:
    """Require this Ray CLI's published address and a bounded GCS protocol response."""
    while not RAY_ADDRESS_FILE.exists() or not RAY_ADDRESS_FILE.read_text().strip():
        if child.poll() is not None:
            raise RuntimeError("Owned Ray head exited before publishing its address")
        if time.monotonic() >= deadline:
            raise TimeoutError("Owned Ray head did not publish its address")
        time.sleep(0.1)
    actual_address = RAY_ADDRESS_FILE.read_text().strip()
    validate_head_address(actual_address, address)
    import grpc
    import ray
    from ray._private.gcs_utils import create_gcs_channel
    from ray.core.generated.gcs_service_pb2 import CheckAliveRequest
    from ray.core.generated.gcs_service_pb2_grpc import NodeInfoGcsServiceStub

    with create_gcs_channel(actual_address) as channel:
        stub = NodeInfoGcsServiceStub(channel)
        while time.monotonic() < deadline:
            if child.poll() is not None:
                raise RuntimeError("Owned Ray head exited before GCS readiness")
            try:
                reply = stub.CheckAlive(
                    CheckAliveRequest(), timeout=min(1, max(0.001, deadline - time.monotonic()))
                )
            except grpc.RpcError as error:
                if error.code() not in {
                    grpc.StatusCode.UNAVAILABLE,
                    grpc.StatusCode.DEADLINE_EXCEEDED,
                }:
                    raise
                time.sleep(0.1)
                continue
            if reply.status.code != 0 or reply.ray_version != ray.__version__:
                raise ValueError("GCS readiness response failed or differs from installed Ray")
            return {
                "address": actual_address,
                "check_alive": True,
                "ray_version": reply.ray_version,
            }
    raise TimeoutError("Owned Ray head did not answer its bounded GCS health RPC")


def validate_head_address(actual: str, requested: str) -> None:
    """Require the owned head's host and a nonzero native-selected GCS port."""
    host, separator, port = actual.rpartition(":")
    if (
        requested != f"{host}:0"
        or not separator
        or not port.isdecimal()
        or not 1 <= int(port) <= 65535
    ):
        raise ValueError("Owned Ray head published an invalid dynamic address")


def validate_head_ready(ready: dict, directory: Path, address: str) -> None:
    validate_head_address(ready["address"], address)
    validate_ray_version(ready["ray_version"])
    if (
        ready["run_id"] != json.loads((directory / "run.json").read_text())["run_id"]
        or ready["check_alive"] is not True
    ):
        raise ValueError("Head readiness does not identify this run's qualified Ray endpoint")


def wait_record(path: Path, child: subprocess.Popen | None, deadline: float) -> dict:
    while not path.exists():
        if child is not None and child.poll() is not None:
            raise RuntimeError(f"Launcher exited before {path.name}; inspect launcher logs")
        if time.monotonic() >= deadline:
            raise TimeoutError(f"Deadline waiting for {path.name}")
        time.sleep(0.1)
    return json.loads(path.read_text())


def host_probe(directory: Path, action: str, timeout: float) -> None:
    import yaml

    _, nodes, gpus = deployment_shape(yaml.safe_load((directory / "config.yaml").read_text()))
    subprocess.run(
        [
            "srun",
            f"--job-name=wideep-probe-{action}-{os.environ['SLURM_JOB_ID']}",
            "--overlap",
            "--exact",
            "--mpi=none",
            "--gpu-bind=none",
            f"--gpus-per-node={gpus}",
            "--cpus-per-task=1",
            f"--nodes={nodes}",
            f"--ntasks={nodes}",
            "--ntasks-per-node=1",
            "python3",
            str(directory / "process_probe.py"),
            action,
            "--directory",
            str(directory),
            "--timeout",
            str(max(0.1, timeout - 10)),
        ],
        check=True,
        timeout=timeout,
    )


def observe_frontend_loss(
    directory: Path, driver: subprocess.Popen, deadline: float, bound: float
) -> None:
    """Does an independent HTTP client see transport failure after verified frontend loss?"""
    frontend = wait_record(directory / "frontend.json", driver, deadline)
    if frontend["run_id"] != directory.name:
        raise ValueError("Frontend endpoint belongs to another run")
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    while True:
        try:
            with opener.open(frontend["url"] + "/health", timeout=1) as response:
                if response.status != 200:
                    raise ValueError("Unexpected HTTP health status")
                break
        except (urllib.error.URLError, TimeoutError):
            if time.monotonic() >= deadline or driver.poll() is not None:
                raise TimeoutError("Native HTTP frontend did not become ready") from None
            time.sleep(0.1)
    request = urllib.request.Request(
        frontend["url"] + "/v1/completions",
        data=json.dumps(
            {
                "model": "/model",
                "prompt": PROMPTS[0],
                "max_tokens": 32,
                "temperature": 0,
                "seed": 2026,
            }
        ).encode(),
        headers={"Content-Type": "application/json"},
    )
    with opener.open(request, timeout=bound) as response:
        healthy = json.load(response)
    if not healthy.get("choices") or "Paris" not in healthy["choices"][0]["text"]:
        raise ValueError("HTTP frontend did not complete healthy inference")
    record(
        directory,
        "http_healthy",
        {"response": healthy, "client": "Allocation controller, separate from frontend process"},
    )
    record(directory, "trigger", {"event": "healthy_http_completed", "response_id": healthy["id"]})
    started = time.monotonic()
    kill_frontend(directory, min(5, bound))
    try:
        with opener.open(
            request, timeout=max(0.001, started + bound - time.monotonic())
        ) as response:
            response.read()
    except (urllib.error.URLError, OSError, http.client.HTTPException) as error:
        cause = error.reason if isinstance(error, urllib.error.URLError) else error
        record(
            directory,
            "client_result",
            {
                "state": "error",
                "error_type": type(error).__name__,
                "error": str(error),
                "transport_cause": type(cause).__name__,
                "harness_timeout": isinstance(cause, TimeoutError),
                "seconds_since_injection": time.monotonic() - started,
            },
        )
    else:
        record(
            directory,
            "client_result",
            {"state": "completed", "seconds_since_injection": time.monotonic() - started},
        )


def write_rank_logs(directory: Path, normalized: str, ranks: int) -> None:
    """Write each rank's attributed log in one pass over the normalized stream."""
    lines_by_rank = {rank: [] for rank in range(ranks)}
    for line in normalized.splitlines():
        rank, _ = line.split(": ", 1)
        lines_by_rank[int(rank)].append(line)
    (directory / "ranks").mkdir(exist_ok=True)
    for rank, lines in lines_by_rank.items():
        (directory / "ranks" / f"rank-{rank:02d}.log").write_text("\n".join(lines) + "\n")


def run(args: argparse.Namespace) -> None:
    import yaml

    directory = args.output_dir
    directory.mkdir(parents=True, exist_ok=False)
    job_id = os.environ["SLURM_JOB_ID"]
    repository = Path(__file__).resolve().parents[4]
    source_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository,
        capture_output=True,
        text=True,
        check=True,
        timeout=10,
    ).stdout.strip()
    record(
        directory,
        "run",
        {
            "scenario": args.scenario,
            "job_id": job_id,
            "source_commit": source_commit,
            "startup_timeout_s": args.startup_timeout,
            "client_timeout_s": args.client_timeout,
            "cleanup_timeout_s": args.cleanup_timeout,
            "qualification": "One attempt; no retries",
            "cache_policy": "Fresh processes; filesystem and compiler cache state uncontrolled; autotuning disabled",
        },
    )
    for name in (
        "process_probe.py",
        "process_loss_controller.py",
        "client_driver.py",
        "deployment_profile.py",
        "slurm_lifecycle.py",
        "regression.py",
        "test_process_loss_integration.py",
        "launch_process_loss.slurm",
    ):
        shutil.copy2(Path(__file__).with_name(name), directory / name)
    shutil.copy2(args.config, directory / "config.yaml")
    shutil.copy2(args.build_manifest, directory / "build_manifest.json")
    shutil.copy2(args.model / "config.json", directory / "model_config.json")
    config = yaml.safe_load(args.config.read_text())
    validate_profile(config, json.loads((args.model / "config.json").read_text()))
    ranks, node_count, gpus = deployment_shape(config)
    manifest = json.loads(args.build_manifest.read_text())
    image_hash = file_sha256(args.image)
    record(directory, "image", {"sha256": image_hash, "path": str(args.image)})
    if image_hash != manifest["runtime_image_sha256"]:
        raise ValueError("Runtime image differs from its bound manifest")
    if args.python_overlay:
        changed = subprocess.run(
            [
                "git",
                "diff",
                "--name-only",
                manifest["source_commit"],
                source_commit,
                "--",
                "tensorrt_llm",
                "cpp",
            ],
            cwd=repository,
            capture_output=True,
            text=True,
            check=True,
            timeout=20,
        ).stdout.splitlines()
        if set(changed) - {"tensorrt_llm/" + relative for relative in OVERLAY_MODULES}:
            raise ValueError(
                "Current native source differs beyond the qualified Python overlay; rebuild the image"
            )
        files = {}
        for relative in OVERLAY_MODULES:
            target = directory / "overlay" / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(repository / "tensorrt_llm" / relative, target)
            files[relative] = file_sha256(target)
        record(
            directory,
            "python_overlay",
            {
                "files": files,
                "source_commit": source_commit,
                "base_source_commit": manifest["source_commit"],
            },
        )
    record(
        directory,
        "test_sources",
        {
            "sha256": {
                path.name: file_sha256(path)
                for path in directory.iterdir()
                if path.suffix in {".py", ".slurm"}
            }
        },
    )
    nodes = subprocess.run(
        ["scontrol", "show", "hostnames", os.environ["SLURM_JOB_NODELIST"]],
        capture_output=True,
        text=True,
        check=True,
        timeout=10,
    ).stdout.splitlines()
    if len(nodes) != node_count:
        raise ValueError("Allocation differs from the requested deployment shape")
    host_probe(directory, "baseline", 60)
    reports = [json.loads(path.read_text()) for path in (directory / "baseline").glob("*.json")]
    validate_hardware(reports, set(nodes), config, manifest)
    address = f"{socket.gethostbyname(nodes[0])}:0"
    temporary = f"/tmp/wideep-ray-{job_id}"
    common = [
        "srun",
        "--overlap",
        "--label",
        "--mpi=none",
        "--kill-on-bad-exit=1",
        "--gpu-bind=none",
        f"--container-image={args.image}",
        f"--container-mounts={temporary}:/tmp,{args.ray_dependencies}:/raydeps,{directory}:/artifacts,{args.model}:/model:ro,/dev/nvidia-caps-imex-channels:/dev/nvidia-caps-imex-channels,/dev/gdrdrv:/dev/gdrdrv",
        "--no-container-mount-home",
        "--container-workdir=/tmp",
        "--container-env=PYTHONPATH",
    ]
    if args.python_overlay:
        common += ["--container-writable"]
    services_name, driver_name = f"wideep-services-{job_id}", f"wideep-client-{job_id}"
    services_command = common + [
        f"--job-name={services_name}",
        f"--nodes={node_count}",
        f"--ntasks={node_count}",
        "--ntasks-per-node=1",
        "/raydeps/venv/bin/python3",
        "/artifacts/process_loss_controller.py",
        "--mode=service",
        "--address",
        address,
        "--output-dir=/artifacts",
    ]
    driver_command = common + [
        f"--job-name={driver_name}",
        "--nodes=1",
        "--ntasks=1",
        f"--nodelist={nodes[0]}",
        "/raydeps/venv/bin/python3",
        "/artifacts/process_loss_controller.py",
        "--mode=driver",
        "--address",
        address,
        "--output-dir=/artifacts",
        "--scenario",
        args.scenario,
        "--client-timeout",
        str(args.client_timeout),
    ]
    launch = {
        "services": services_command,
        "driver": driver_command,
        "address": address,
        "requested_address": address,
        "runtime_environment": {
            key: os.environ.get(key)
            for key in (
                "TRTLLM_FORCE_COMM_METHOD",
                "TRTLLM_MOE_A2A_FORCE_CFT",
                "TLLM_FAULT_TOLERANCE_MODE",
                "TLLM_RANK_CRASH_HARD_KILL_GRACE",
            )
        },
    }
    children = []
    summary = {}
    environment = {
        **os.environ,
        "PYTHONPATH": "/artifacts",
        "RAY_DEDUP_LOGS": "0",
        "TLLM_DISABLE_MPI": "1",
    }
    with (
        (directory / "services.log").open("w") as services_log,
        (directory / "driver.log").open("w") as driver_log,
    ):

        def interrupted(signum, frame) -> None:
            raise InterruptedError("Controller cleanup requested")

        previous = signal.signal(signal.SIGUSR1, interrupted)
        try:
            host_probe(directory, "prepare_tmp", 60)
            started = time.monotonic()
            services = subprocess.Popen(
                services_command, env=environment, stdout=services_log, stderr=subprocess.STDOUT
            )
            children.append((services, services_name, "step-service-0.json"))
            ready = wait_record(
                directory / "head_ready.json", services, started + args.startup_timeout
            )
            validate_head_ready(ready, directory, address)
            address = ready["address"]
            driver_command[driver_command.index("--address") + 1] = address
            launch["address"] = address
            record(directory, "launch", launch)
            driver = subprocess.Popen(
                driver_command, env=environment, stdout=driver_log, stderr=subprocess.STDOUT
            )
            children.append((driver, driver_name, "step-driver.json"))
            healthy = wait_record(
                directory / "healthy.json", driver, started + args.startup_timeout
            )
            summary["startup_to_readiness_s"] = time.monotonic() - started
            summary["readiness_clock"] = (
                "Controller CLOCK_MONOTONIC from service launch to receipt of healthy.json"
            )
            workers = json.loads((directory / "workers.json").read_text())["workers"]
            # Ray forwards startup logs asynchronously; wait for complete native evidence, not a fixed sleep.
            deadline = time.monotonic() + 30
            while True:
                try:
                    normalized = normalize_ray_log(
                        (directory / "driver.log").read_text(errors="replace"),
                        workers,
                        address.split(":")[0],
                    )
                    qualified = validate_execution_log(
                        normalized, config, json.loads((args.model / "config.json").read_text())
                    )
                except ValueError:
                    if time.monotonic() >= deadline:
                        raise
                    time.sleep(0.1)
                else:
                    (directory / "model.log").write_text(normalized)
                    write_rank_logs(directory, normalized, ranks)
                    record(directory, "qualification", qualified)
                    break
            if healthy["run_id"] != directory.name:
                raise ValueError("Stale healthy record")
            initialized = json.loads((directory / "initialized.json").read_text())
            record(
                directory,
                "cold_start",
                {
                    "startup_to_readiness_s": summary["startup_to_readiness_s"],
                    "readiness_clock": summary["readiness_clock"],
                    "api_construction_s": initialized["construction_seconds"],
                    "construction_clock": "Client CLOCK_MONOTONIC around LLM construction",
                    "per_rank_native_metrics": {
                        str(worker["rank"]): worker["startup_metrics"] for worker in workers
                    },
                    "phase_semantics": "Native process-local elapsed intervals; nested and overlapping, do not sum",
                    "communicator_init": "Not isolated by current native startup metrics",
                    "autotuning_enabled": config["enable_autotuner"],
                    "cache_policy": (
                        "Fresh services and model processes; filesystem, page and compiler caches uncontrolled"
                    ),
                    "queue_time": "Excluded; allocation_timing.json records scheduler timestamps separately",
                },
            )
            record(directory, "permit", {"scenario": args.scenario})
            if args.scenario == "frontend_kill":
                observe_frontend_loss(
                    directory, driver, started + args.startup_timeout, args.client_timeout
                )
            if args.scenario == "idle_stop":
                wait_record(directory / "injection_intent.json", driver, time.monotonic() + 20)
                host_probe(directory, "stopped", 20)
            summary["driver_returncode"] = driver.wait(
                timeout=args.client_timeout + args.cleanup_timeout + 30
            )
            if args.scenario == "frontend_kill":
                if driver.returncode != 137 or (directory / "shutdown.json").exists():
                    raise RuntimeError("Frontend driver did not terminate by the verified SIGKILL")
            elif driver.returncode != 0 or not (directory / "shutdown.json").exists():
                raise RuntimeError("Driver failed or native shutdown did not complete")
        finally:
            signal.signal(signal.SIGUSR1, signal.SIG_IGN)
            cleanup_started = time.monotonic()
            deadline = cleanup_started + args.cleanup_timeout
            errors = {}
            try:
                if not (directory / "launch.json").exists():
                    record(directory, "launch", launch)
            except (OSError, ValueError) as error:
                errors["launch"] = str(error)
            for child, name, filename in reversed(children):
                try:
                    identity_file = directory / filename
                    identity = (
                        json.loads(identity_file.read_text()) if identity_file.exists() else None
                    )
                    if identity is not None and identity["job_id"] != job_id:
                        raise ValueError("Cleanup identity belongs to another allocation")
                    summary[name] = cleanup_model_step(
                        child,
                        job_id,
                        step_id=identity["step_id"] if identity else None,
                        step_name=name,
                        timeout_s=max(0.1, deadline - time.monotonic() - 15),
                    )
                except (
                    OSError,
                    ValueError,
                    LookupError,
                    RuntimeError,
                    subprocess.SubprocessError,
                ) as error:
                    errors[name] = str(error)
            try:
                host_probe(directory, "cleanup", max(0.1, deadline - time.monotonic()))
                cleanup = [
                    json.loads(path.read_text()) for path in (directory / "cleanup").glob("*.json")
                ]
                if (
                    len(cleanup) != node_count
                    or {row["hostname"] for row in cleanup} != set(nodes)
                    or not all(row["clean"] for row in cleanup)
                ):
                    raise RuntimeError("Incomplete resource cleanup verification")
            except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
                errors["resources"] = str(error)
            try:
                if (directory / "workers.json").exists():
                    workers = json.loads((directory / "workers.json").read_text())["workers"]
                    normalized = normalize_ray_log(
                        (directory / "driver.log").read_text(errors="replace"),
                        workers,
                        address.split(":")[0],
                    )
                    write_rank_logs(directory, normalized, ranks)
            except (OSError, ValueError) as error:
                errors["rank_logs"] = str(error)
            summary["cleanup_seconds"] = time.monotonic() - cleanup_started
            summary["cleanup_errors"] = errors
            record(directory, "summary", summary)
            signal.signal(signal.SIGUSR1, previous)
            if errors:
                raise RuntimeError(f"Unverified cleanup: {errors}")


def wait_host_probes(directory: Path, job_id: str, deadline: float) -> None:
    """Let orphaned probe writers finish before rescue inspects or mutates their storage."""
    actions = ("baseline", "prepare_tmp", "stopped", "cleanup")
    names = {f"--job-name=wideep-probe-{action}-{job_id}" for action in actions}
    while True:
        live = False
        for process in Path("/proc").iterdir():
            if not process.name.isdecimal():
                continue
            try:
                if process.stat().st_uid != os.getuid():
                    continue
                argv = (process / "cmdline").read_bytes().decode().split("\0")
                probe = (
                    any(name in argv for name in names)
                    and str(directory / "process_probe.py") in argv
                    and "--directory" in argv
                    and argv[argv.index("--directory") + 1] == str(directory)
                )
                controller = (
                    any(arg.endswith("/process_loss_controller.py") for arg in argv)
                    and "--output-dir" in argv
                    and argv[argv.index("--output-dir") + 1] == str(directory)
                    and not any(arg.startswith("--mode=") for arg in argv)
                    and "--mode" not in argv
                )
                if not (probe or controller):
                    continue
                identity = process_identity(int(process.name))
                if identity["state"] != "Z" and re.search(
                    rf"/job_{job_id}/step_batch(?:/|$)", identity["cgroup"], re.MULTILINE
                ):
                    live = True
            except (FileNotFoundError, ProcessLookupError):
                continue
        if not live:
            break
        if time.monotonic() >= deadline:
            raise TimeoutError("Owned host-probe launcher did not terminate")
        time.sleep(min(0.1, max(0, deadline - time.monotonic())))
    for action in actions:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("Owned host-probe reconciliation deadline exceeded")
        step = discover_owned_step(job_id, f"wideep-probe-{action}-{job_id}", min(10, remaining))
        if step is not None:
            wait_step_terminal(job_id, step, max(0.001, deadline - time.monotonic()))


def rescue_cleanup(directory: Path, timeout_s: float) -> None:
    """Recheck an interrupted controller's owned steps using separate rescue records."""
    run = json.loads((directory / "run.json").read_text())
    if run["run_id"] != directory.name or run["job_id"] != os.environ.get("SLURM_JOB_ID"):
        raise ValueError("Rescue requires this allocation's original run")
    deadline = time.monotonic() + timeout_s
    steps = set()
    for path in directory.glob("step-*.json"):
        row = json.loads(path.read_text())
        if row["run_id"] != directory.name or row["job_id"] != run["job_id"]:
            raise ValueError("Rescue step belongs to another run")
        steps.add(row["step_id"])
    for role in ("services", "client"):
        step = discover_owned_step(
            run["job_id"],
            f"wideep-{role}-{run['job_id']}",
            min(10, max(0.1, deadline - time.monotonic())),
        )
        if step is not None:
            steps.add(step)
    for step in sorted(steps):
        cancel_owned_step(run["job_id"], step, max(0.1, deadline - time.monotonic() - 15))
    wait_host_probes(directory, run["job_id"], deadline)
    try:
        validate_cleanup(directory, allow_partial=True)
    except (OSError, ValueError, KeyError, StopIteration):
        pass
    else:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("Rescue evidence exceeded the cleanup deadline")
        record(
            directory,
            "interruption_cleanup",
            {"state": "verified", "evidence": str(directory), "seconds": timeout_s - remaining},
        )
        return
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise TimeoutError("No time remains for independent rescue")
    host_probe(directory, "rescue", remaining)
    validate_cleanup(directory, allow_partial=True, rescue=True)
    if time.monotonic() >= deadline:
        raise TimeoutError("Rescue evidence exceeded the cleanup deadline")
    record(
        directory,
        "interruption_cleanup",
        {
            "state": "verified",
            "evidence": str(directory),
            "rescue": True,
            "seconds": timeout_s - (deadline - time.monotonic()),
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", choices=("controller", "service", "driver", "cleanup"), default="controller"
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--address")
    parser.add_argument(
        "--scenario",
        choices=("healthy", "frontend_kill", *WORKER_SIGNALS),
        default="healthy",
    )
    parser.add_argument("--image", type=Path)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--build-manifest", type=Path)
    parser.add_argument("--ray-dependencies", type=Path)
    parser.add_argument("--python-overlay", action="store_true")
    parser.add_argument("--startup-timeout", type=float, default=720)
    parser.add_argument("--client-timeout", type=float, default=30)
    parser.add_argument("--cleanup-timeout", type=float, default=180)
    args = parser.parse_args()
    if any(
        not math.isfinite(value) or value <= 0
        for value in (args.startup_timeout, args.client_timeout, args.cleanup_timeout)
    ):
        parser.error("Deadlines must be finite and positive")
    if args.mode == "controller":
        for path in (
            args.output_dir,
            args.image,
            args.model,
            args.config,
            args.build_manifest,
            args.ray_dependencies,
        ):
            if (
                path is None
                or not path.is_absolute()
                or any(character in str(path) for character in (",", ":"))
            ):
                parser.error("Controller inputs require absolute paths without mount separators")
    if args.mode == "cleanup":
        rescue_cleanup(args.output_dir, args.cleanup_timeout)
    elif args.mode == "service":
        bootstrap(args)
    elif args.mode == "driver":
        install_overlay(args.output_dir, "driver")
        from client_driver import main as client_main

        client_main(args)
    else:
        run(args)


if __name__ == "__main__":
    main()
