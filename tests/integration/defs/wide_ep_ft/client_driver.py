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
"""Ray LLM API client: healthy readiness, one named process signal and bounded result."""

import argparse
import asyncio
import faulthandler
import json
import os
import platform
import socket
import tempfile
import time
from pathlib import Path
from queue import Empty

from deployment_profile import (
    PROMPTS,
    deployment_shape,
    validate_healthy,
    validate_ray_version,
    validate_resolved_profile,
    validate_runtime,
)
from process_probe import WORKER_SIGNALS, process_identity, record

import tensorrt_llm


def run(args: argparse.Namespace) -> None:
    # Load TensorRT-LLM before torch so its native library paths are initialized.
    import ray
    import torch
    import yaml
    from ray.util.placement_group import placement_group, remove_placement_group

    from tensorrt_llm import LLM, SamplingParams
    from tensorrt_llm.llmapi.llm_args import RayPlacementConfig

    directory = args.output_dir
    run_id = json.loads((directory / "run.json").read_text())["run_id"]
    config = yaml.safe_load((directory / "config.yaml").read_text())
    model = json.loads(Path("/model/config.json").read_text())
    ranks, node_count, gpus = deployment_shape(config)
    runtime = {
        "tensorrt_llm_version": tensorrt_llm.__version__,
        "torch_version": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "nccl_runtime_version": list(torch.cuda.nccl.version()),
        "architecture": platform.machine(),
        "ray_version": ray.__version__,
    }
    validate_ray_version(ray.__version__)
    validate_runtime(runtime, json.loads((directory / "build_manifest.json").read_text()))
    record(directory, "runtime", runtime)
    ray.init(address=args.address, namespace="trtllm", log_to_driver=True)
    try:
        deadline = time.monotonic() + 120
        while True:
            nodes = [node for node in ray.nodes() if node["Alive"]]
            if (
                len(nodes) == node_count
                and sum(node["Resources"].get("GPU", 0) for node in nodes) == ranks
            ):
                break
            if time.monotonic() >= deadline:
                raise TimeoutError("Ray resources differ from requested deployment shape")
            time.sleep(0.1)
        head_ip = args.address.split(":")[0]
        head = [node for node in nodes if node["NodeManagerAddress"] == head_ip]
        if len(head) != 1:
            raise ValueError("Ambiguous Ray head node")
        ordered = head + sorted(
            [node for node in nodes if node not in head],
            key=lambda node: node["NodeManagerAddress"],
        )
        bundles = [
            {"GPU": 1, "CPU": 1, f"node:{node['NodeManagerAddress']}": 0.001}
            for node in ordered
            for _ in range(gpus)
        ]
        placement = placement_group(bundles, strategy="PACK")
        ray.get(placement.ready(), timeout=60)
        record(directory, "placement", {"nodes": nodes, "bundles": bundles})
        initialization = time.monotonic()
        with LLM(
            model="/model",
            orchestrator_type="ray",
            ray_worker_extension_cls="process_probe.WorkerProbe",
            ray_placement_config=RayPlacementConfig(
                placement_groups=[placement], placement_bundle_indices=[list(range(ranks))]
            ),
            **config,
        ) as llm:
            record(
                directory,
                "initialized",
                {
                    "construction_seconds": time.monotonic() - initialization,
                    "rank0_startup_metrics": llm.startup_metrics,
                },
            )
            workers = ray.get(
                [
                    worker.call_worker_method.remote("wideep_identity")
                    for worker in llm._executor.workers
                ],
                timeout=30,
            )
            if (
                sorted(worker["rank"] for worker in workers) != list(range(ranks))
                or len({worker["actor_id"] for worker in workers}) != ranks
            ):
                raise ValueError("Incomplete or ambiguous model actor identities")
            for worker in workers:
                worker["run_id"] = run_id
                if not worker["graphs_enabled"] or len(worker["graph_keys"]) != len(
                    config["cuda_graph_config"]["batch_sizes"]
                ):
                    raise ValueError("Missing decode graph variants")
                validate_resolved_profile(
                    {"run_id": run_id, "configuration": worker["resolved"]}, run_id, config, model
                )
            record(directory, "workers", {"workers": workers})
            outputs = [
                llm.generate_async(
                    prompt, SamplingParams(temperature=0, seed=2026, max_tokens=32)
                ).result(timeout=120)
                for prompt in PROMPTS
            ]
            healthy = {
                "run_id": run_id,
                "state": "paused_between_requests",
                "results": [
                    {
                        "prompt": out.prompt,
                        "text": out.outputs[0].text,
                        "token_ids": list(out.outputs[0].token_ids),
                    }
                    for out in outputs
                ],
            }
            validate_healthy(healthy, run_id)
            record(directory, "healthy", healthy)
            deadline = time.monotonic() + 60
            while not (directory / "permit.json").exists():
                if time.monotonic() >= deadline:
                    raise TimeoutError(
                        "External deployment qualification did not release admission"
                    )
                time.sleep(0.1)
            permit = json.loads((directory / "permit.json").read_text())
            if permit["run_id"] != run_id or permit["scenario"] != args.scenario:
                raise ValueError("Stale or conflicting admission permit")
            if args.scenario == "frontend_kill":
                from tensorrt_llm.serve.openai_server import OpenAIServer

                server = OpenAIServer(
                    generator=llm,
                    model="/model",
                    tool_parser=None,
                    server_role=None,
                    metadata_server_cfg=None,
                )
                with socket.socket() as listener:
                    listener.bind(("0.0.0.0", 0))
                    listener.listen()
                    port = listener.getsockname()[1]
                    record(
                        directory,
                        "frontend",
                        {
                            "identity": process_identity(os.getpid()),
                            "url": f"http://{head_ip}:{port}",
                            "implementation": "Native OpenAIServer over the same Ray LLM",
                        },
                    )
                    asyncio.run(server("0.0.0.0", port, sockets=[listener]))
            elif args.scenario != "healthy":
                observe_failure(llm, args, workers, run_id)
        record(directory, "shutdown", {"state": "complete"})
        remove_placement_group(placement)
    finally:
        ray.shutdown()


def observe_failure(
    llm: tensorrt_llm.LLM, args: argparse.Namespace, workers: list[dict], run_id: str
) -> None:
    import ray

    from tensorrt_llm import SamplingParams

    directory = args.output_dir
    streaming = args.scenario == "stream_kill"
    result = None
    last_output = None
    sequence = 0

    def next_output(deadline: float) -> dict:
        nonlocal sequence
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("Client observation deadline exceeded")
        try:
            result._result_step(timeout=remaining)
        except Empty:
            raise TimeoutError("Client observation deadline exceeded") from None
        snapshot = {
            "native_request_id": result.request_id,
            "sequence": sequence,
            "token_ids": list(result.outputs[0].token_ids),
            "text": str(result.outputs[0].text),
            "finished": bool(result.finished),
        }
        record(directory, f"stream/{sequence:06d}", snapshot)
        sequence += 1
        return snapshot

    trigger = {"run_id": run_id, "event": "healthy_between_requests"}
    if streaming:
        result = llm.generate_async(
            PROMPTS[0],
            SamplingParams(temperature=0, seed=2026, max_tokens=256, ignore_eos=True),
            streaming=True,
        )
        deadline = time.monotonic() + args.client_timeout
        while last_output is None or not last_output["token_ids"]:
            last_output = next_output(deadline)
            if last_output["finished"]:
                raise RuntimeError("Stream completed before its nonfinal trigger")
        trigger = {**last_output, "run_id": run_id, "event": "first_nonfinal_output"}
    fatal = llm._executor._fatal_error
    shutting_down = llm._executor.is_shutdown()
    record(
        directory,
        "pre_injection",
        {"fatal_error": str(fatal) if fatal is not None else None, "is_shutdown": shutting_down},
    )
    if fatal is not None or shutting_down:
        raise RuntimeError("Engine failed before fault injection")
    record(directory, "trigger", trigger)
    rank = 0 if args.scenario == "rank0_kill" else len(workers) // 2
    victim = next(worker for worker in workers if worker["rank"] == rank)
    record(
        directory,
        "injection",
        {
            "scenario": args.scenario,
            "target": victim,
            "signal": WORKER_SIGNALS[args.scenario].name,
            "selection": "fixed characterization rank; not recovery admission",
        },
    )
    started = time.monotonic()
    action = llm._executor.workers[rank].call_worker_method.remote(
        "wideep_signal", victim, str(directory), args.scenario
    )
    if args.scenario == "idle_stop":
        deadline = started + 25
        while not (directory / "stopped.json").exists():
            if time.monotonic() >= deadline:
                raise TimeoutError("Independent OS stop confirmation did not arrive")
            time.sleep(0.1)
        stopped = json.loads((directory / "stopped.json").read_text())
        if stopped["run_id"] != run_id or stopped["target"]["actor_id"] != victim["actor_id"]:
            raise ValueError("Stall confirmation belongs to another actor or run")
    else:
        try:
            ray.get(action, timeout=15)
        except ray.exceptions.ActorDiedError as error:
            if victim["actor_id"] not in str(error):
                raise ValueError("Ray reported death of a different actor") from error
            record(
                directory,
                "death",
                {
                    "actor_id": victim["actor_id"],
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "seconds_since_injection": time.monotonic() - started,
                },
            )
        else:
            raise RuntimeError("Injection did not produce the expected actor death")
    # Issue diagnostic probes independently; never delay or authorize client execution on replies.
    probes = [
        (
            worker,
            llm._executor.workers[worker["rank"]].call_worker_method.remote(
                "wideep_cuda_probe", str(directory)
            ),
        )
        for worker in workers
        if worker["rank"] != rank
    ]
    deadline = started + args.client_timeout
    try:
        if streaming:
            while not result.finished:
                last_output = next_output(deadline)
        else:
            result = llm.generate_async(
                PROMPTS[0], SamplingParams(temperature=0, seed=2026, max_tokens=32)
            )
            result.result(timeout=max(0.001, deadline - time.monotonic()))
    except (RuntimeError, TimeoutError) as error:
        record(
            directory,
            "client_result",
            {
                "state": "error",
                "error_type": type(error).__name__,
                "error": str(error),
                "harness_timeout": isinstance(error, TimeoutError),
                "last_partial_output": last_output,
                "seconds_since_injection": time.monotonic() - started,
            },
        )
    else:
        record(
            directory,
            "client_result",
            {"state": "completed", "seconds_since_injection": time.monotonic() - started},
        )
    observations = []
    with tempfile.TemporaryFile(mode="w+") as trace:
        faulthandler.dump_traceback(file=trace, all_threads=True)
        trace.seek(0)
        stacks = trace.read()
    record(
        directory,
        "diagnostics/client",
        {
            "fatal_error": str(llm._executor._fatal_error),
            "pending_request_ids": list(llm._executor._results),
            "observed_request_id": None if result is None else result.request_id,
            "thread_stacks": stacks,
            "scope": "Client state immediately after bounded observation, before shutdown",
        },
    )
    for worker, ref in probes:
        try:
            reply = ray.get(ref, timeout=0)
        except ray.exceptions.RayError as error:
            observations.append(
                {"rank": worker["rank"], "error_type": type(error).__name__, "error": str(error)}
            )
        else:
            if reply["pid"] != worker["pid"] or reply["start_ticks"] != worker["start_ticks"]:
                raise ValueError("Diagnostic probe identity changed")
            observations.append(reply)
    record(
        directory,
        "cuda_observations",
        {
            "scope": "Diagnostic operations only; not agreement or recovery",
            "observations": observations,
        },
    )


def main(args: argparse.Namespace) -> None:
    record(
        args.output_dir,
        "step-driver",
        {
            "job_id": os.environ["SLURM_JOB_ID"],
            "step_id": os.environ["SLURM_STEP_ID"],
            "identity": process_identity(os.getpid()),
        },
    )
    for key in list(os.environ):
        if key.startswith(("SLURM_", "PMI", "PMIX", "MPI", "OMPI")):
            del os.environ[key]
    os.environ["TLLM_DISABLE_MPI"] = "1"
    run(args)
