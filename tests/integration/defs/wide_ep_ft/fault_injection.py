# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Observe one MPI WideEP fault, verify teardown, and restart on the same GPUs."""

import argparse
import hashlib
import json
import math
import os
import re
import shlex
import signal
import socket
import subprocess
import time
import traceback
import uuid
from contextlib import suppress
from pathlib import Path
from types import FrameType

from fault_injector import (
    IDENTITY_DIR_ENV,
    RUN_ID_ENV,
    SCENARIOS,
    record,
    resources_released,
    snapshot,
)

PROMPTS = ("The capital of France is", "The capital of Italy is")


def load_config(path: Path, model: Path) -> dict:
    """Read native LLM arguments and fill static placement for supported MoE layouts."""
    import yaml

    config = yaml.safe_load(path.read_text())
    placement = config.get("moe_config", {}).get("load_balancer")
    if placement is not None and "initial_global_assignments" not in placement:
        layout = json.loads((model / "config.json").read_text())
        if layout.get("model_type") == "deepseek_v3" and layout.get("moe_layer_freq", 1) == 1:
            experts, first = layout["n_routed_experts"], layout["first_k_dense_replace"]
        elif (
            layout.get("model_type") == "qwen3_moe"
            and layout.get("decoder_sparse_step", 1) == 1
            and not layout.get("mlp_only_layers")
        ):
            experts, first = layout["num_experts"], 0
        else:
            raise ValueError("Provide explicit initial_global_assignments for this model layout")
        slots = placement["num_slots"]
        if experts <= 0 or slots < experts or slots % config["moe_expert_parallel_size"]:
            raise ValueError("Static placement requires expert coverage and equal rank slots")
        placement.update(
            layer_updates_per_iter=0,
            initial_global_assignments={
                layer: [slot % experts for slot in range(slots)]
                for layer in range(first, layout["num_hidden_layers"])
            },
        )
    return config


def validate_restart(initial: list[dict], restarted: list[dict]) -> None:
    """Require fresh process identities with the original rank-to-GPU mapping."""
    before, after = ({row["rank"]: row for row in rows} for rows in (initial, restarted))
    if (
        not before
        or len(before) != len(initial)
        or len(after) != len(restarted)
        or set(before) != set(after)
    ):
        raise ValueError("Incomplete restart identities")
    for rank, row in before.items():
        new = after[rank]
        if (
            row["gpu_uuid"] != new["gpu_uuid"]
            or row["hostname"] != new["hostname"]
            or (row["pid"], row["start_ticks"], row["boot_id"])
            == (new["pid"], new["start_ticks"], new["boot_id"])
        ):
            raise ValueError("Restart must use new workers on the same GPUs")


def validate_workers(workers: list[dict], ranks: int, graphs_requested: bool) -> None:
    """Check complete membership, requested captures, and qualified communication geometry."""
    if (
        sorted(row["rank"] for row in workers) != list(range(ranks))
        or len({row["gpu_uuid"] for row in workers}) != ranks
    ):
        raise ValueError("Incomplete model worker identities")
    for row in workers:
        if graphs_requested and (not row["graphs_enabled"] or not row["graph_keys"]):
            raise ValueError("Requested CUDA graphs were not captured")
        communication = row["communication"].get("NVLinkOneSided", {})
        if (
            communication.get("cft_capable") is not False
            or communication.get("ep_size") != ranks
            or communication.get("ep_rank") != row["rank"]
        ):
            raise ValueError(
                "Worker communication differs from the requested non-CFT WideEP geometry"
            )


def _read(directory: Path, name: str) -> dict:
    return json.loads((directory / f"{name}.json").read_text())


def _wait_file(directory: Path, name: str, timeout_s: float) -> dict:
    deadline = time.monotonic() + timeout_s
    while not (directory / f"{name}.json").exists():
        if (directory / "injection_failed.json").exists():
            raise RuntimeError(_read(directory, "injection_failed"))
        if time.monotonic() >= deadline:
            raise TimeoutError(f"Missing {name} receipt: {directory}")
        time.sleep(0.05)
    return _read(directory, name)


def _wait_workers(directory: Path, ranks: int, timeout_s: float) -> list[dict]:
    deadline = time.monotonic() + timeout_s
    workers = [
        _wait_file(directory / "workers", str(rank), _remaining(deadline)) for rank in range(ranks)
    ]
    _remaining(deadline)
    return workers


def client(args: argparse.Namespace) -> None:
    # TensorRT-LLM initializes native library paths before importing torch.
    import yaml

    import tensorrt_llm
    from tensorrt_llm import LLM, SamplingParams

    directory = args.output_dir
    config = yaml.safe_load(args.config.read_text())
    started = time.monotonic()
    with LLM(model=str(args.model), **config) as llm:
        loaded = time.monotonic()
        workers = _wait_workers(
            directory, config["moe_expert_parallel_size"], args.client_timeout_s
        )
        validate_workers(
            workers, config["moe_expert_parallel_size"], bool(config.get("cuda_graph_config"))
        )
        results = [
            llm.generate_async(
                prompt, SamplingParams(temperature=0, seed=2026, max_tokens=32)
            ).result(timeout=args.client_timeout_s)
            for prompt in PROMPTS
        ]
        texts = [result.outputs[0].text for result in results]
        if not all(word in text for word, text in zip(("Paris", "Rome"), texts)):
            raise ValueError(f"Healthy inference failed: {texts}")
        record(
            directory,
            "healthy",
            {
                "constructor_s": loaded - started,
                "constructor_to_readiness_s": time.monotonic() - started,
                "results": [
                    {"text": result.outputs[0].text, "token_ids": list(result.outputs[0].token_ids)}
                    for result in results
                ],
                "tensorrt_llm": tensorrt_llm.__version__,
            },
        )
        _wait_file(directory, "proceed", args.cleanup_timeout_s)
        if args.scenario != "healthy":
            target = next(row for row in workers if row["rank"] == args.target_rank)
            result = None
            trigger = {"run_id": args.run_id, "event": "between_requests"}
            if args.scenario == "worker_sigkill_streaming":
                result = llm.generate_async(
                    PROMPTS[0],
                    SamplingParams(temperature=0, seed=2026, max_tokens=256, ignore_eos=True),
                    streaming=True,
                )
                deadline = time.monotonic() + args.client_timeout_s
                while not result.outputs[0].token_ids:
                    result._result_step(timeout=max(0.001, deadline - time.monotonic()))
                    if result.finished or time.monotonic() >= deadline:
                        raise TimeoutError("Stream ended or timed out before a nonfinal token")
                trigger.update(
                    event="first_nonfinal_output",
                    token_ids=list(result.outputs[0].token_ids),
                    finished=result.finished,
                )
            record(directory, "trigger", trigger)
            record(directory, "fault_request", {"target": target, "scenario": args.scenario})
            injected = time.monotonic()
            _wait_file(
                directory,
                "injection_result"
                if args.scenario == "fence_round_mismatch"
                else "injection_intent",
                15,
            )
            try:
                if result is None:
                    result = llm.generate_async(
                        PROMPTS[0],
                        SamplingParams(temperature=0, seed=2026, max_tokens=32),
                        streaming=True,
                    )
                result.result(timeout=args.client_timeout_s)
            except Exception as error:
                record(
                    directory,
                    "client_error",
                    {
                        "type": type(error).__name__,
                        "message": str(error),
                        "elapsed_s": time.monotonic() - injected,
                        "traceback": traceback.format_exc(),
                    },
                )
                raise
            record(directory, "client_result", {"outcome": "completed"})
            raise AssertionError("Inference completed after injection")
    record(directory, "shutdown", {"state": "complete"})


def _remaining(deadline: float) -> float:
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise TimeoutError("Deadline exceeded")
    return remaining


def _steps(name: str, timeout_s: float = 10) -> list[str]:
    job_id = os.environ["SLURM_JOB_ID"]
    result = subprocess.run(
        ["scontrol", "--oneliner", "show", "step", job_id],
        capture_output=True,
        text=True,
        check=True,
        timeout=timeout_s,
    )
    steps = []
    for line in result.stdout.splitlines():
        step = re.search(r"(?:^|\s)StepId=(\S+)", line)
        step_name = re.search(r"(?:^|\s)Name=(\S+)", line)
        if (
            step
            and step_name
            and step_name[1] == name
            and re.fullmatch(rf"{re.escape(job_id)}\.\d+", step[1])
        ):
            steps.append(step[1])
    return steps


def _stop(process: subprocess.Popen, name: str, deadline: float) -> None:
    if process.poll() is None:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    try:
        for step in _steps(name, min(10, _remaining(deadline))):
            subprocess.run(["scancel", step], check=True, timeout=min(10, _remaining(deadline)))
    finally:
        try:
            process.wait(timeout=max(0, min(10, deadline - time.monotonic())))
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait(timeout=max(0, min(10, deadline - time.monotonic())))
    while pending := _steps(name, min(10, _remaining(deadline))):
        for step in pending:
            subprocess.run(["scancel", step], check=True, timeout=min(10, _remaining(deadline)))
        time.sleep(min(0.1, _remaining(deadline)))


def _probe(
    args: argparse.Namespace,
    directory: Path,
    run_id: str,
    terminate: bool = False,
    deadline: float | None = None,
) -> list[dict]:
    directory.mkdir()
    if deadline is None:
        deadline = time.monotonic() + args.cleanup_timeout_s
    _remaining(deadline)
    name = f"weft-{uuid.uuid4().hex}"
    environment = os.environ.copy()
    for key in (RUN_ID_ENV, IDENTITY_DIR_ENV):
        environment.pop(key, None)
    command = [
        *args.probe_launcher,
        f"--job-name={name}",
        f"--jobid={os.environ['SLURM_JOB_ID']}",
        args.python,
        str(Path(__file__).resolve()),
        "--probe",
        "--run-id",
        run_id,
        "--output-dir",
        str(directory),
    ]
    if terminate:
        command.append("--terminate")
    with (directory / "launcher.log").open("x") as stream:
        process = subprocess.Popen(
            command,
            env=environment,
            stdout=stream,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            if process.wait(timeout=_remaining(deadline)):
                raise RuntimeError(f"CPU resource probe failed: {directory}")
        finally:
            _stop(process, name, deadline)
    _remaining(deadline)
    return [json.loads(path.read_text()) for path in directory.glob("*.json")]


def _visible_workers(workers: list[dict], observed: list[dict]) -> None:
    # Host probes check ownership; container UID numbers may be remapped.
    keys = ("hostname", "pid", "start_ticks", "boot_id", "pid_namespace")
    host_processes = {
        tuple(process[key] for key in keys) for row in observed for process in row["live_processes"]
    }
    if any(tuple(worker[key] for key in keys) not in host_processes for worker in workers):
        raise ValueError(
            "Independent probe cannot observe every live model worker; share the host PID namespace"
        )


def validate_fault_evidence(directory: Path, scenario: str, workers: list[dict]) -> None:
    """Require matching injection receipts and the measured native MPI failure outcome."""
    if scenario == "healthy":
        if not (directory / "shutdown.json").exists():
            raise AssertionError("Clean shutdown receipt missing")
        return
    intent = _read(directory, "injection_intent")
    from fault_injector import validate_injection

    target = intent["target"]
    expected = next(row for row in workers if row["rank"] == target["rank"])
    validate_injection(expected, target, scenario, _read(directory, "trigger"))
    if intent["scenario"] != scenario or (directory / "client_result.json").exists():
        raise AssertionError("Injection mismatch or completed post-fault inference")
    logs = "\n".join(path.read_text(errors="replace") for path in directory.glob("*.log"))
    if scenario.startswith("worker_sigkill_"):
        if not re.search(rf"task {target['rank']}:.*(?:Killed|137)", logs):
            raise AssertionError("No native launcher confirmation of target death")
        if (directory / "client_error.json").exists():
            error = _read(directory, "client_error")
            if error["type"] != "RequestError" or "MPI_ERR_" not in error["message"]:
                raise AssertionError("Unexpected process-loss client error")
    else:
        mutation = _read(directory, "injection_result")
        if mutation["after"] != mutation["before"] + 2:
            raise AssertionError("Fence mutation missing")
        error = _read(directory, "client_error")
        if error["type"] != "RequestError" or not re.search(
            r"(?:dispatch|combine).*timed out waiting for completion flag", logs
        ):
            raise AssertionError("Expected native fence timeout and RequestError missing")


def run_phase(
    args: argparse.Namespace, baseline: list[dict], phase: str, scenario: str
) -> list[dict]:
    directory = args.output_dir / phase
    directory.mkdir()
    run_id = uuid.uuid4().hex
    name = f"weft-{run_id}"
    environment = {
        **os.environ,
        RUN_ID_ENV: run_id,
        IDENTITY_DIR_ENV: str(directory),
        "WIDEEP_FT_TARGET_RANK": str(args.target_rank),
        "TRTLLM_EPLB_SHM_NAME": f"weft_{run_id}",
        "PYTHONPATH": os.pathsep.join(
            filter(None, [str(Path(__file__).resolve().parent), os.environ.get("PYTHONPATH")])
        ),
    }
    command = [
        *args.launcher,
        f"--job-name={name}",
        f"--jobid={os.environ['SLURM_JOB_ID']}",
        f"--output={directory}/rank-%t.log",
        "trtllm-llmapi-launch",
        args.python,
        str(Path(__file__).resolve()),
        "--client",
        "--model",
        str(args.model),
        "--config",
        str(args.output_dir / "config.yaml"),
        "--scenario",
        scenario,
        "--output-dir",
        str(directory),
        "--run-id",
        run_id,
        "--target-rank",
        str(args.target_rank),
        "--client-timeout-s",
        str(args.client_timeout_s),
        "--cleanup-timeout-s",
        str(args.cleanup_timeout_s),
    ]
    record(directory, "launch", {"run_id": run_id, "step_name": name, "command": command})
    started = time.monotonic()
    received = {}
    probe_index = 0

    def observe(terminate: bool = False, deadline: float | None = None) -> list[dict]:
        nonlocal probe_index
        probe_index += 1
        return _probe(args, directory / f"probe-{probe_index}", run_id, terminate, deadline)

    with (directory / "launcher.log").open("x") as log:
        process = subprocess.Popen(
            command, env=environment, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
        )

        def wait_for_cleanup(deadline: float, forced: bool = False) -> None:
            while True:
                observed = observe(deadline=deadline)
                pending = _steps(name, min(10, _remaining(deadline)))
                _remaining(deadline)
                if (
                    process.poll() is not None
                    and not pending
                    and resources_released(baseline, observed)
                ):
                    record(
                        directory,
                        "forced_cleanup" if forced else "cleanup",
                        {
                            "step_name": name,
                            "owned_steps": pending,
                            "snapshots": observed,
                            "forced": forced,
                        },
                    )
                    return
                time.sleep(min(0.1, _remaining(deadline)))

        try:
            deadline = started + args.startup_timeout_s
            workers = []
            while process.poll() is None:
                for event in (
                    "healthy",
                    "trigger",
                    "injection_intent",
                    "injection_result",
                    "client_error",
                    "shutdown",
                ):
                    if event in received or not (directory / f"{event}.json").exists():
                        continue
                    received[event] = time.monotonic() - started
                    if event == "healthy":
                        workers = _wait_workers(directory, args.ranks, _remaining(deadline))
                        validate_workers(workers, args.ranks, args.graphs_requested)
                        if {row["gpu_uuid"] for row in workers} != {
                            gpu[0] for row in baseline for gpu in row["gpus"]
                        }:
                            raise ValueError(
                                "Model GPUs differ from independently observed allocation"
                            )
                        _visible_workers(workers, observe(deadline=deadline))
                        live_steps = _steps(name, min(10, _remaining(deadline)))
                        if len(live_steps) != 1:
                            raise AssertionError("Cannot identify the live owned model step")
                        _remaining(deadline)
                        record(
                            directory,
                            "proceed",
                            {"live_identity_probe": "complete", "owned_steps": live_steps},
                        )
                        deadline = (
                            time.monotonic() + args.client_timeout_s + args.shutdown_timeout_s
                        )
                    elif event == "client_error":
                        _remaining(deadline)
                        deadline = time.monotonic() + args.shutdown_timeout_s
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"{phase} exceeded its startup/client/teardown deadline")
                time.sleep(0.05)
            _remaining(deadline)
            received["step_exit"] = time.monotonic() - started
            for event in (
                "trigger",
                "injection_intent",
                "injection_result",
                "client_error",
                "shutdown",
            ):
                if event not in received and (directory / f"{event}.json").exists():
                    received[event] = time.monotonic() - started
            if "healthy" not in received:
                raise AssertionError("MPI step ended before healthy readiness")
            expected_exit = 0 if scenario == "healthy" else 137
            if process.returncode != expected_exit:
                raise AssertionError(
                    f"MPI step exited {process.returncode}, expected {expected_exit}"
                )
            wait_for_cleanup(time.monotonic() + args.cleanup_timeout_s)
            validate_fault_evidence(directory, scenario, workers)
        except (Exception, KeyboardInterrupt) as error:
            # Evidence storage may be the failure cause; cleanup must still run.
            with suppress(OSError):
                record(directory, "intervention", {"error": repr(error), "forced_cleanup": True})
            cleanup_deadline = time.monotonic() + args.cleanup_timeout_s
            try:
                _stop(process, name, min(cleanup_deadline, time.monotonic() + 20))
            except Exception as control_error:
                with suppress(OSError):
                    record(directory, "cleanup_control_error", {"error": repr(control_error)})
            try:
                observe(terminate=True, deadline=cleanup_deadline)
                wait_for_cleanup(cleanup_deadline, forced=True)
            except Exception as cleanup_error:
                with suppress(OSError):
                    record(directory, "cleanup_failed", {"error": repr(cleanup_error)})
                with suppress(OSError, subprocess.SubprocessError):
                    _stop(process, name, cleanup_deadline)
            raise
        finally:
            record(
                directory,
                "timing",
                {
                    "seconds_to_received_event": received,
                    "total_s": time.monotonic() - started,
                    "exit_code": process.poll(),
                    "clock_source": "single parent CLOCK_MONOTONIC; complete file receipt",
                },
            )
    return workers


def run(args: argparse.Namespace) -> None:
    import yaml

    args.output_dir.mkdir(parents=True)
    config = load_config(args.config, args.model)
    args.ranks = config["moe_expert_parallel_size"]
    args.graphs_requested = bool(config.get("cuda_graph_config"))
    if args.target_rank is None:
        args.target_rank = args.ranks // 2
    if not 0 < args.target_rank < args.ranks:
        raise ValueError("Target must be a nonzero model rank")
    (args.output_dir / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    record(
        args.output_dir,
        "run",
        {
            "run_id": args.run_id,
            "scenario": args.scenario,
            "model": str(args.model),
            "launcher": args.launcher,
            "probe_launcher": args.probe_launcher,
            "allocation_job_id": os.environ["SLURM_JOB_ID"],
            "source_sha256": {
                path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                for path in Path(__file__).parent.glob("*.py")
            },
            "allocation_queue_seconds": None,
            "autotuning_enabled": config.get("enable_autotuner"),
            "cache_state": "uncontrolled filesystem/compiler caches",
            "time_bounds_s": {
                phase: getattr(args, f"{phase}_timeout_s")
                for phase in ("startup", "client", "shutdown", "cleanup")
            },
        },
    )
    try:
        baseline = _probe(args, args.output_dir / "baseline", uuid.uuid4().hex)
        if (
            not baseline
            or any(row["compute_apps"] for row in baseline)
            or len({row["hostname"] for row in baseline}) != len(baseline)
            or sum(len(row["gpus"]) for row in baseline) != args.ranks
        ):
            raise ValueError("Allocation must be dedicated and contain exactly the requested GPUs")
        initial = run_phase(args, baseline, "initial", args.scenario)
        restarted = run_phase(args, baseline, "restart", "healthy")
        validate_restart(initial, restarted)
    except (Exception, KeyboardInterrupt) as error:
        record(
            args.output_dir,
            "summary",
            {"state": "FAIL", "error": repr(error), "traceback": traceback.format_exc()},
        )
        raise
    record(
        args.output_dir,
        "summary",
        {
            "state": "PASS",
            "same_gpu_restart": True,
            "initial": _read(args.output_dir / "initial", "timing"),
            "restart": _read(args.output_dir / "restart", "timing"),
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--launcher", help="Direct srun prefix for native MPI model workers")
    parser.add_argument(
        "--probe-launcher", help="Direct srun --mpi=none prefix, one CPU probe per GPU host"
    )
    parser.add_argument(
        "--python", default="python3", help="Python executable inside launched tasks"
    )
    parser.add_argument("--model", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--scenario", choices=("healthy", *SCENARIOS), default="healthy")
    parser.add_argument("--target-rank", type=int)
    parser.add_argument("--run-id", default=uuid.uuid4().hex)
    parser.add_argument("--startup-timeout-s", type=float, default=720)
    parser.add_argument("--client-timeout-s", type=float)
    parser.add_argument("--shutdown-timeout-s", type=float, default=180)
    parser.add_argument("--cleanup-timeout-s", type=float, default=60)
    parser.add_argument("--client", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--probe", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--terminate", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    args.output_dir = args.output_dir.resolve()
    if args.probe:
        # Retain host-observed identities even when dead tasks lose their environment.
        known_path = args.output_dir.parent / "probe-1" / f"{socket.gethostname()}.json"
        known = json.loads(known_path.read_text()) if known_path.exists() else None
        if known is not None and known["run_id"] != args.run_id:
            raise ValueError("Host identity receipt belongs to another run")
        record(
            args.output_dir,
            socket.gethostname(),
            snapshot(args.run_id, args.terminate, known["live_processes"] if known else ()),
        )
        return
    args.client_timeout_s = (
        args.client_timeout_s
        if args.client_timeout_s is not None
        else (420 if args.scenario == "fence_round_mismatch" else 30)
    )
    if any(
        not math.isfinite(getattr(args, f"{phase}_timeout_s"))
        or getattr(args, f"{phase}_timeout_s") <= 0
        for phase in ("startup", "client", "shutdown", "cleanup")
    ):
        parser.error("Timeouts must be finite and positive")
    if not args.model or not args.config:
        parser.error("--model and --config are required")
    args.model, args.config = args.model.resolve(), args.config.resolve()
    if args.client:
        client(args)
        return
    if "SLURM_JOB_ID" not in os.environ or not args.launcher or not args.probe_launcher:
        parser.error("Provide --launcher and --probe-launcher inside an existing Slurm allocation")
    for option in ("launcher", "probe_launcher"):
        command = shlex.split(getattr(args, option))
        if (
            not command
            or Path(command[0]).name != "srun"
            or any(token.startswith(("--job-name", "-J", "--jobid")) for token in command)
        ):
            parser.error("Launch prefixes must use direct srun; the test owns step names")
        setattr(args, option, command)

    def interrupt(signum: int, _frame: FrameType | None) -> None:
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        raise KeyboardInterrupt(f"Interrupted by signal {signum}")

    signal.signal(signal.SIGTERM, interrupt)
    run(args)


if __name__ == "__main__":
    main()
