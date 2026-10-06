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
"""Observe one WideEP fault, verify cleanup, and explicitly restart on the same GPUs."""

import argparse
import hashlib
import json
import os
import re
import signal
import subprocess
import sys
import time
import traceback
import uuid
from pathlib import Path
from queue import Empty
from types import FrameType, ModuleType
from typing import TYPE_CHECKING

from fault_injector import SCENARIOS, NodeProbe, record, resources_released

if TYPE_CHECKING:
    from ray.actor import ActorHandle

    from tensorrt_llm import LLM

PROMPTS = ("The capital of France is", "The capital of Italy is")


def load_config(path: Path, model: Path) -> dict:
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
            or row["actor_id"] == new["actor_id"]
            or (row["pid"], row["start_ticks"], row["boot_id"])
            == (new["pid"], new["start_ticks"], new["boot_id"])
        ):
            raise ValueError("Restart must use new workers on the same GPUs")


def child(args: argparse.Namespace) -> None:
    # TensorRT-LLM initializes native library paths before torch is imported.
    import tensorrt_llm

    # isort: split
    import ray
    import torch
    import yaml
    from ray.util.placement_group import placement_group, remove_placement_group

    from tensorrt_llm import LLM, SamplingParams
    from tensorrt_llm.llmapi.llm_args import RayPlacementConfig

    directory = args.output_dir
    config = yaml.safe_load((directory.parent / "config.yaml").read_text())
    ranks = config["moe_expert_parallel_size"]
    ray.init(
        address=args.address,
        runtime_env={"working_dir": str(Path(__file__).parent)},
        log_to_driver=True,
    )
    record(directory, "job", {"job_id": str(ray.get_runtime_context().get_job_id())})
    nodes = sorted(
        [node for node in ray.nodes() if node["Alive"] and node["Resources"].get("GPU")],
        key=lambda node: (
            node["NodeManagerAddress"] != ray.util.get_node_ip_address(),
            node["NodeManagerAddress"],
        ),
    )
    if not nodes or nodes[0]["NodeManagerAddress"] != ray.util.get_node_ip_address():
        raise ValueError("Driver must run on a reserved GPU node for rank-0 local IPC")
    bundles = [
        {"GPU": 1, "CPU": 1, f"node:{node['NodeManagerAddress']}": 0.001}
        for node in nodes
        for _ in range(int(node["Resources"]["GPU"]))
    ]
    if len(bundles) != ranks:
        raise ValueError("Dedicated Ray cluster GPU count differs from deployment")
    placement = placement_group(bundles, strategy="PACK")
    ray.get(placement.ready(), timeout=args.startup_timeout_s)
    try:
        with LLM(
            model=str(args.model),
            orchestrator_type="ray",
            ray_worker_extension_cls="fault_injector.WorkerExtension",
            ray_placement_config=RayPlacementConfig(
                placement_groups=[placement], placement_bundle_indices=[list(range(ranks))]
            ),
            **config,
        ) as llm:
            workers = ray.get(
                [
                    worker.call_worker_method.remote("fault_identity")
                    for worker in llm._executor.workers
                ],
                timeout=30,
            )
            if (
                sorted(row["rank"] for row in workers) != list(range(ranks))
                or len({row["actor_id"] for row in workers}) != ranks
                or len({row["gpu_uuid"] for row in workers}) != ranks
            ):
                raise ValueError("Incomplete model worker identities")
            for row in workers:
                row["run_id"] = args.run_id
                if config.get("cuda_graph_config") and (
                    not row["graphs_enabled"] or not row["graph_keys"]
                ):
                    raise ValueError("Requested CUDA graphs were not captured")
                if "NVLinkOneSided" not in row["communication"]:
                    raise ValueError("Expected WideEP NVLinkOneSided communication is not active")
            record(
                directory,
                "workers",
                {
                    "workers": workers,
                    "startup_metrics": llm.startup_metrics,
                    "runtime": {
                        "tensorrt_llm": tensorrt_llm.__version__,
                        "ray": ray.__version__,
                        "torch": torch.__version__,
                        "cuda": torch.version.cuda,
                    },
                },
            )
            outputs = [
                llm.generate_async(
                    prompt, SamplingParams(temperature=0, seed=2026, max_tokens=32)
                ).result(timeout=args.client_timeout_s)
                for prompt in PROMPTS
            ]
            texts = [out.outputs[0].text for out in outputs]
            if not all(word in text for word, text in zip(("Paris", "Rome"), texts)):
                raise ValueError(f"Healthy inference failed: {texts}")
            record(directory, "healthy", {"results": texts})
            if args.scenario != "healthy":
                observe_fault(llm, workers, args)
        record(directory, "shutdown", {"state": "complete"})
    finally:
        remove_placement_group(placement)
        ray.shutdown()


def observe_fault(llm: "LLM", workers: list[dict], args: argparse.Namespace) -> None:
    import ray

    from tensorrt_llm import SamplingParams
    from tensorrt_llm.executor import EngineDeadError

    directory = args.output_dir
    streaming = args.scenario == "worker_sigkill_streaming"
    result = None
    last_output = None
    if streaming:
        result = llm.generate_async(
            PROMPTS[0],
            SamplingParams(temperature=0, seed=2026, max_tokens=256, ignore_eos=True),
            streaming=True,
        )
        deadline = time.monotonic() + args.client_timeout_s
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("No nonfinal output reached the streaming trigger")
            result._result_step(timeout=remaining)
            last_output = {
                "token_ids": list(result.outputs[0].token_ids),
                "text": result.outputs[0].text,
                "finished": bool(result.finished),
            }
            if last_output["finished"]:
                raise RuntimeError("Stream finished before injection")
            if last_output["token_ids"]:
                break
        trigger = {"event": "first_nonfinal_output", **last_output}
    else:
        trigger = {"event": "between_requests"}
    if llm._executor._fatal_error is not None or llm._executor.is_shutdown():
        raise RuntimeError("Engine failed before injection")
    record(directory, "trigger", {"run_id": args.run_id, **trigger})
    target_rank = len(workers) // 2
    target = next(row for row in workers if row["rank"] == target_rank)
    started = time.monotonic()
    action = llm._executor.workers[target_rank].call_worker_method.remote(
        "inject_fault", target, str(directory), args.scenario
    )
    try:
        ray.get(action, timeout=min(15, args.client_timeout_s))
        if args.scenario.startswith("worker_sigkill_"):
            raise AssertionError("SIGKILL actor unexpectedly replied")
    except ray.exceptions.ActorDiedError as error:
        if not args.scenario.startswith("worker_sigkill_") or target["actor_id"] not in str(error):
            raise
        record(directory, "actor_death", {"actor_id": target["actor_id"], "error": str(error)})
    try:
        if result is None:
            result = llm.generate_async(
                PROMPTS[0], SamplingParams(temperature=0, seed=2026, max_tokens=32), streaming=True
            )
        deadline = started + args.client_timeout_s
        while not result.finished:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("Client did not report a terminal failure")
            result._result_step(timeout=remaining)
            last_output = {
                "token_ids": list(result.outputs[0].token_ids),
                "text": result.outputs[0].text,
                "finished": bool(result.finished),
            }
        raise AssertionError("Faulted request completed successfully")
    except EngineDeadError as error:
        if time.monotonic() - started > args.client_timeout_s:
            raise TimeoutError("Terminal error arrived after the client deadline") from error
        record(
            directory,
            "client_error",
            {
                "type": type(error).__name__,
                "error": str(error),
                "seconds_from_injection_rpc": time.monotonic() - started,
                "last_output": last_output,
            },
        )
    except Empty as error:
        raise TimeoutError("Client did not report a terminal failure") from error
    probes = [
        worker.call_worker_method.remote("fault_cuda_probe")
        for rank, worker in enumerate(llm._executor.workers)
        if rank != target_rank or not args.scenario.startswith("worker_sigkill_")
    ]
    ready, pending = ray.wait(probes, num_returns=len(probes), timeout=5)
    observations = []
    for reply in ready:
        try:
            observations.append(ray.get(reply))
        except ray.exceptions.RayError as error:
            observations.append({"error": str(error)})
    record(
        directory,
        "cuda_probes",
        {
            "replies": observations,
            "unknown": len(pending),
            "scope": "successful operation only; no recovery claim",
        },
    )


def validate_fault_evidence(directory: Path, scenario: str) -> None:
    if scenario == "healthy":
        return
    intent = json.loads((directory / "injection_intent.json").read_text())
    if intent["scenario"] != scenario:
        raise ValueError("Fault evidence belongs to a different scenario")
    if scenario.startswith("worker_sigkill_"):
        death = json.loads((directory / "actor_death.json").read_text())
        if death["actor_id"] != intent["target"]["actor_id"]:
            raise ValueError("Confirmed death does not match the target")
        return
    result = json.loads((directory / "injection_result.json").read_text())
    log = "\n".join(
        path.read_text()
        for path in [directory / "driver.log", *sorted((directory / "ray_logs").glob("*/*"))]
    )
    if scenario == "fence_round_mismatch":
        if result["after"] != result["before"] + 2 or not re.search(
            r"(?:dispatch|combine):.*timed out waiting for completion flag", log
        ):
            raise AssertionError("No native fence timeout evidence for the verified round mismatch")
    elif "gloo" not in result["backend"].lower() or not re.search(
        r"(?:Group.*not registered|Invalid process group|Connection (?:closed|reset) by peer|gloo.*(?:Error|error))",
        log,
    ):
        raise AssertionError("No host collective error evidence for the aborted Gloo group")


def owned_actors(ray: ModuleType, job_id: str) -> dict:
    from ray._private import state

    actors = state.actors(job_id=ray.JobID.from_hex(job_id))
    return {
        actor_id: row
        for actor_id, row in actors.items()
        if row["JobID"] == job_id and row["State"] != "DEAD"
    }


def cleanup(
    ray: ModuleType,
    probes: list["ActorHandle"],
    baseline: list[dict],
    directory: Path,
    identities: list[dict],
    timeout_s: float,
    forced: bool,
) -> None:
    job_id = (
        json.loads((directory / "job.json").read_text())["job_id"]
        if (directory / "job.json").exists()
        else None
    )
    deadline = time.monotonic() + timeout_s
    forced_actor_ids = set()
    terminated_pids = set()
    native_cleanup_failed = False
    try:
        while True:
            if forced and job_id is not None:
                from ray._private.worker import global_worker

                actors = owned_actors(ray, job_id)
                pinned = ray.get(
                    [probe.pin_actor_processes.remote(actors) for probe in probes], timeout=15
                )
                identities.extend(
                    identity for node in pinned for identity in node if identity not in identities
                )
                for actor_id in actors:
                    global_worker.core_worker.kill_actor(ray.ActorID.from_hex(actor_id), True)
                    forced_actor_ids.add(actor_id)
                terminated = ray.get(
                    [probe.terminate_owned_processes.remote(identities) for probe in probes],
                    timeout=15,
                )
                terminated_pids.update(pid for node in terminated for pid in node)
            current = ray.get([probe.snapshot.remote(identities) for probe in probes], timeout=15)
            if (
                (job_id is None or not owned_actors(ray, job_id))
                and (
                    job_id is None
                    or any(
                        row["JobID"] == job_id and row["IsDead"]
                        for row in ray._private.state.jobs()
                    )
                )
                and resources_released(baseline, current)
                and ray.available_resources().get("GPU", 0)
                >= sum(len(row["gpus"]) for row in baseline)
            ):
                record(
                    directory,
                    "cleanup",
                    {
                        "snapshots": current,
                        "forced": forced,
                        "forced_actor_ids": sorted(forced_actor_ids),
                        "terminated_pids": sorted(terminated_pids),
                    },
                )
                break
            if time.monotonic() >= deadline:
                record(
                    directory,
                    "cleanup_backstop_failed" if native_cleanup_failed else "cleanup_failed",
                    {
                        "snapshots": current,
                        "forced": forced,
                        "forced_actor_ids": sorted(forced_actor_ids),
                        "terminated_pids": sorted(terminated_pids),
                    },
                )
                if not forced:
                    native_cleanup_failed = True
                    forced = True
                    deadline = time.monotonic() + timeout_s
                    continue
                raise TimeoutError("Model processes or GPU resources remain after teardown")
            time.sleep(0.1)
        if native_cleanup_failed:
            raise TimeoutError("Native teardown leaked resources; forced backstop was required")
    finally:
        if job_id is not None:
            original_failure = sys.exc_info()[0] is not None
            try:
                logs = ray.get(
                    [probe.collect_logs.remote(str(directory), job_id) for probe in probes],
                    timeout=30,
                )
                record(directory, "logs", {"files_by_node": logs})
            except (OSError, TimeoutError, ray.exceptions.RayError) as error:
                record(directory, "logs_failed", {"error": str(error)})
                if not original_failure:
                    raise


def run_phase(
    args: argparse.Namespace,
    ray: ModuleType,
    probes: list["ActorHandle"],
    baseline: list[dict],
    name: str,
    scenario: str,
) -> dict:
    directory = args.output_dir / name
    directory.mkdir()
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--child",
        "--address",
        args.address,
        "--model",
        str(args.model),
        "--config",
        str(args.config),
        "--output-dir",
        str(directory),
        "--scenario",
        scenario,
        "--run-id",
        args.run_id,
        "--client-timeout-s",
        str(args.client_timeout_s),
        "--startup-timeout-s",
        str(args.startup_timeout_s),
        "--shutdown-timeout-s",
        str(args.shutdown_timeout_s),
        "--cleanup-timeout-s",
        str(args.cleanup_timeout_s),
    ]
    started = time.monotonic()
    received = {}
    identities = []
    forced = True
    with (directory / "driver.log").open("x") as log:
        process = subprocess.Popen(
            command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
        )
        try:
            deadline = started + args.startup_timeout_s
            while process.poll() is None:
                for event in ("healthy", "trigger", "client_error", "shutdown"):
                    if event not in received and (directory / f"{event}.json").exists():
                        received[event] = time.monotonic() - started
                        if event == "healthy":
                            deadline = time.monotonic() + (
                                args.shutdown_timeout_s
                                if scenario == "healthy"
                                else args.client_timeout_s
                            )
                        elif event == "trigger":
                            deadline = time.monotonic() + args.client_timeout_s
                        elif event == "client_error":
                            deadline = time.monotonic() + args.shutdown_timeout_s
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"{name} exceeded its startup or terminal deadline")
                time.sleep(0.05)
            if process.returncode:
                raise AssertionError(
                    f"{name} exited {process.returncode}; see {directory / 'driver.log'}"
                )
            if not (directory / "shutdown.json").exists():
                raise AssertionError("Native shutdown completion is unverified")
            for event in ("healthy", "trigger", "client_error", "shutdown"):
                if event not in received and (directory / f"{event}.json").exists():
                    received[event] = time.monotonic() - started
            forced = False
            if scenario != "healthy" and not (directory / "client_error.json").exists():
                raise AssertionError("Expected EngineDeadError is missing")
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=15)
            if (directory / "workers.json").exists():
                identities = json.loads((directory / "workers.json").read_text())["workers"]
            try:
                cleanup(
                    ray, probes, baseline, directory, identities, args.cleanup_timeout_s, forced
                )
            finally:
                record(
                    directory,
                    "timing",
                    {
                        "seconds_to_received_event": received,
                        "total_seconds": time.monotonic() - started,
                        "clock_source": "single parent CLOCK_MONOTONIC; file publication receipt",
                    },
                )
    validate_fault_evidence(directory, scenario)
    return {"workers": identities, "received": received, "parent_started_s": started}


def run(args: argparse.Namespace) -> None:
    import ray
    import yaml
    from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

    args.output_dir.mkdir(parents=True, exist_ok=False)
    config = load_config(args.config, args.model)
    (args.output_dir / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    record(
        args.output_dir,
        "run",
        {
            "run_id": args.run_id,
            "scenario": args.scenario,
            "model": str(args.model),
            "source_sha256": {
                path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                for path in Path(__file__).parent.glob("*.py")
            },
            "config_sha256": hashlib.sha256(
                (args.output_dir / "config.yaml").read_bytes()
            ).hexdigest(),
            "allocation_queue_seconds": None,
            "autotuning_enabled": config.get("enable_autotuner"),
            "cache_state": "uncontrolled filesystem/compiler caches",
        },
    )
    ray.init(address=args.address, runtime_env={"working_dir": str(Path(__file__).parent)})
    probes = []
    try:
        probe_class = ray.remote(num_cpus=0)(NodeProbe)
        probes = [
            probe_class.options(
                scheduling_strategy=NodeAffinitySchedulingStrategy(node["NodeID"], soft=False)
            ).remote()
            for node in ray.nodes()
            if node["Alive"] and node["Resources"].get("GPU")
        ]
        visible = ray.get(
            [probe.read_run_id.remote(str(args.output_dir)) for probe in probes], timeout=15
        )
        if not visible or any(run_id != args.run_id for run_id in visible):
            raise ValueError("Evidence directory must be shared across all GPU nodes")
        baseline = ray.get([probe.snapshot.remote([]) for probe in probes], timeout=30)
        if not baseline or any(row["compute_apps"] for row in baseline):
            raise ValueError("Dedicated Ray cluster must have no existing GPU compute contexts")
        record(args.output_dir, "baseline", {"snapshots": baseline})
        expected_gpus = {gpu[0] for row in baseline for gpu in row["gpus"]}
        if len(expected_gpus) != config["moe_expert_parallel_size"]:
            raise ValueError("Dedicated cluster does not match the requested EP size")
        initial = run_phase(args, ray, probes, baseline, "initial", args.scenario)
        if {row["gpu_uuid"] for row in initial["workers"]} != expected_gpus:
            raise ValueError("Model worker GPU identities differ from the independent baseline")
        restarted = run_phase(args, ray, probes, baseline, "restart", "healthy")
        validate_restart(initial["workers"], restarted["workers"])
        record(
            args.output_dir,
            "summary",
            {
                "state": "PASS",
                "same_gpu_restart": True,
                "fault_observed_to_restart_readiness_s": (
                    restarted["parent_started_s"]
                    + restarted["received"]["healthy"]
                    - initial["parent_started_s"]
                    - initial["received"].get("client_error", initial["received"]["healthy"])
                ),
                "initial": initial["received"],
                "restart": restarted["received"],
            },
        )
    except (Exception, KeyboardInterrupt) as error:
        record(
            args.output_dir,
            "summary",
            {"state": "FAIL", "error": repr(error), "traceback": traceback.format_exc()},
        )
        raise
    finally:
        for probe in probes:
            ray.kill(probe, no_restart=True)
        ray.shutdown()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--address", required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--scenario", choices=("healthy", *SCENARIOS), required=True)
    parser.add_argument("--run-id", default=uuid.uuid4().hex)
    parser.add_argument("--startup-timeout-s", type=float, default=720)
    parser.add_argument("--client-timeout-s", type=float, default=30)
    parser.add_argument("--shutdown-timeout-s", type=float, default=180)
    parser.add_argument("--cleanup-timeout-s", type=float, default=60)
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    for name in ("model", "config", "output_dir"):
        setattr(args, name, getattr(args, name).resolve())
    if not args.child:

        def interrupt(signum: int, _frame: FrameType | None) -> None:
            raise KeyboardInterrupt(f"Interrupted by signal {signum}")

        signal.signal(signal.SIGTERM, interrupt)
    if any(
        getattr(args, name) <= 0
        for name in (
            "startup_timeout_s",
            "client_timeout_s",
            "shutdown_timeout_s",
            "cleanup_timeout_s",
        )
    ):
        parser.error("Timeouts must be positive")
    (child if args.child else run)(args)


if __name__ == "__main__":
    main()
