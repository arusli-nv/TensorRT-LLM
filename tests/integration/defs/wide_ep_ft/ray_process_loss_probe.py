# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Probe one model-actor process loss in an externally launched Ray cluster."""

from __future__ import annotations

import argparse
import json
import math
import os
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from survivor_probe import make_probe_proposal, summarize_probe_replies


def _write_json(path: Path, record: dict) -> None:
    with path.open("x") as output:
        json.dump(record, output, indent=2)
        output.write("\n")
        output.flush()
        os.fsync(output.fileno())


def _request_result(llm, prompt: str, timeout_s: float) -> dict:
    from tensorrt_llm import SamplingParams

    result = {}

    def request() -> None:
        try:
            response = llm.generate([prompt], SamplingParams(max_tokens=16))
            result.update({"status": "response", "text": response[0].outputs[0].text})
        except Exception as error:
            result.update(
                {"status": "error", "error_type": type(error).__name__, "error": str(error)}
            )

    started_at_utc = datetime.now(timezone.utc).isoformat()
    thread = threading.Thread(target=request, daemon=True)
    thread.start()
    thread.join(timeout_s)
    if thread.is_alive():
        result = {"status": "timeout"}
    return {
        "started_at_utc": started_at_utc,
        **result,
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def _load_llm_kwargs(path: Path) -> dict:
    kwargs = json.loads(path.read_text())
    if not isinstance(kwargs, dict):
        raise ValueError("LLM kwargs must be a JSON object")
    forbidden = {"model", "orchestrator_type", "ray_worker_extension_cls"} & kwargs.keys()
    if forbidden:
        raise ValueError(f"LLM kwargs cannot override {sorted(forbidden)}")
    if "kv_cache_config" in kwargs and isinstance(kwargs["kv_cache_config"], dict):
        from tensorrt_llm.llmapi import KvCacheConfig

        kwargs["kv_cache_config"] = KvCacheConfig(**kwargs["kv_cache_config"])
    return kwargs


def _validate_duration(name: str, value: float, allow_zero: bool = False) -> None:
    if not math.isfinite(value) or (value < 0 if allow_zero else value <= 0):
        raise ValueError(f"{name} must be finite and {'nonnegative' if allow_zero else 'positive'}")


def _submit_survivor_probes(workers, proposal: dict) -> tuple[dict, dict]:
    """Collect RPC submission errors without losing the remaining observations."""
    references = {}
    errors = {}
    for rank in proposal["survivors"]:
        try:
            references[rank] = workers[rank].call_worker_method.remote(
                "prepare_survivor_probe", proposal
            )
        except Exception as error:
            errors[rank] = {
                "status": "submission_error",
                "error_type": type(error).__name__,
                "error": str(error),
            }
    return references, errors


def run_case(
    model: str,
    llm_kwargs_path: Path,
    ray_address: str,
    expected_ranks: int,
    target_rank: int,
    output_dir: Path,
    readiness_timeout_s: float,
    request_timeout_s: float,
    rpc_timeout_s: float,
    probe_delay_s: float,
    require_all_survivors: bool = False,
) -> dict:
    """Observe one worker loss; keep diagnostic replies separate from recovery."""
    for name, value in (
        ("readiness_timeout_s", readiness_timeout_s),
        ("request_timeout_s", request_timeout_s),
        ("rpc_timeout_s", rpc_timeout_s),
    ):
        _validate_duration(name, value)
    _validate_duration("probe_delay_s", probe_delay_s, allow_zero=True)
    if expected_ranks < 2 or not 0 < target_rank < expected_ranks:
        raise ValueError("target must be one nonzero model world rank")
    hard_kill_grace = os.environ.get("TLLM_RANK_CRASH_HARD_KILL_GRACE")
    if probe_delay_s >= 10 and hard_kill_grace != "-1":
        raise ValueError("late diagnostic probes require TLLM_RANK_CRASH_HARD_KILL_GRACE=-1")

    import ray
    from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

    from tensorrt_llm import LLM

    llm_kwargs = _load_llm_kwargs(llm_kwargs_path)
    output_dir.mkdir(parents=True, exist_ok=False)
    _write_json(
        output_dir / "configuration.json",
        {
            "model": model,
            "llm_kwargs": json.loads(llm_kwargs_path.read_text()),
            "ray_address": ray_address,
            "expected_ranks": expected_ranks,
            "target_rank": target_rank,
            "hard_kill_grace": hard_kill_grace,
            "readiness_timeout_s": readiness_timeout_s,
            "request_timeout_s": request_timeout_s,
            "rpc_timeout_s": rpc_timeout_s,
            "probe_delay_s": probe_delay_s,
            "require_all_survivors": require_all_survivors,
        },
    )
    summary = {
        "status": "not_injected",
        "stage": "ray_readiness",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    connected = False
    try:
        ray.init(address=ray_address, include_dashboard=False)
        connected = True
        deadline = time.monotonic() + readiness_timeout_s
        while ray.cluster_resources().get("GPU", 0) < expected_ranks:
            if time.monotonic() >= deadline:
                raise TimeoutError("Ray cluster did not expose the required GPUs")
            time.sleep(1)

        summary["stage"] = "model_startup"
        llm = LLM(
            model=model,
            orchestrator_type="ray",
            ray_worker_extension_cls="survivor_probe_worker_extension.SurvivorProbeWorkerExtension",
            **llm_kwargs,
        )
        workers = llm._executor.workers
        if len(workers) != expected_ranks:
            raise ValueError(f"model has {len(workers)} workers, expected {expected_ranks}")

        summary["stage"] = "healthy_request"
        healthy = _request_result(llm, "The capital of France is", request_timeout_s)
        _write_json(output_dir / "healthy_request.json", healthy)
        if healthy["status"] != "response" or not healthy["text"]:
            raise RuntimeError("healthy request did not produce a nonempty completion")

        summary["stage"] = "worker_identity"
        identities = ray.get(
            [worker.call_worker_method.remote("probe_identity") for worker in workers],
            timeout=rpc_timeout_s,
        )
        if [identity["world_rank"] for identity in identities] != list(range(expected_ranks)):
            raise ValueError("Ray worker order does not match model world ranks")
        _write_json(output_dir / "rank_map.json", {"ranks": identities})

        @ray.remote(num_cpus=0.1)
        def signal_rank(rank_map_path: str, rank: int, injection_dir: str) -> None:
            from pathlib import Path

            from fault_injector import kill_rank

            kill_rank(Path(rank_map_path), rank, Path(injection_dir))

        summary["stage"] = "injection"
        summary["status"] = "injection_uncertain"
        ray.get(
            signal_rank.options(
                scheduling_strategy=NodeAffinitySchedulingStrategy(
                    node_id=identities[target_rank]["ray_node_id"], soft=False
                )
            ).remote(str(output_dir / "rank_map.json"), target_rank, str(output_dir / "injection")),
            timeout=rpc_timeout_s,
        )
        summary["status"] = "signal_sent"

        time.sleep(probe_delay_s)
        summary["stage"] = "survivor_replies"
        proposal = make_probe_proposal(expected_ranks, target_rank, uuid.uuid4().hex)
        _write_json(output_dir / "diagnostic_proposal.json", proposal)
        references, errors = _submit_survivor_probes(workers, proposal)
        pending = []
        if references:
            try:
                _, pending = ray.wait(
                    list(references.values()), num_returns=len(references), timeout=rpc_timeout_s
                )
            except Exception as error:
                pending = list(references.values())
                for rank in references:
                    errors[rank] = {
                        "status": "wait_error",
                        "error_type": type(error).__name__,
                        "error": str(error),
                    }
        replies = {}
        for rank, reference in references.items():
            if reference in pending:
                errors.setdefault(rank, {"status": "timeout"})
                continue
            try:
                replies[rank] = ray.get(reference)
            except Exception as error:
                errors[rank] = {
                    "status": "error",
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
        rank_to_pid = {identity["world_rank"]: identity["pid"] for identity in identities}
        probe = {
            **summarize_probe_replies(proposal, replies, rank_to_pid),
            "replies": replies,
            "errors": errors,
        }
        _write_json(output_dir / "survivor_probe.json", probe)

        summary["stage"] = "post_failure_request"
        post_failure = _request_result(llm, "The capital of Italy is", request_timeout_s)
        _write_json(output_dir / "post_failure_request.json", post_failure)
        summary["status"] = (
            "observation_complete"
            if not require_all_survivors or probe["all_survivors_replied"]
            else "survivor_probe_incomplete"
        )
        summary["stage"] = "done"
    except (OSError, ValueError, RuntimeError, TimeoutError, ray.exceptions.RayError) as error:
        summary["error_type"] = type(error).__name__
        summary["error"] = str(error)
    finally:
        summary["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        _write_json(output_dir / "run_summary.json", summary)
        if connected:
            ray.shutdown()
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--llm-kwargs", type=Path, required=True)
    parser.add_argument("--ray-address", required=True)
    parser.add_argument("--expected-ranks", type=int, required=True)
    parser.add_argument("--target-rank", type=int, default=1)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--readiness-timeout-s", type=float, default=120)
    parser.add_argument("--request-timeout-s", type=float, default=60)
    parser.add_argument("--rpc-timeout-s", type=float, default=30)
    parser.add_argument("--probe-delay-s", type=float, default=0)
    parser.add_argument("--require-all-survivors", action="store_true")
    args = parser.parse_args()
    summary = run_case(
        args.model,
        args.llm_kwargs,
        args.ray_address,
        args.expected_ranks,
        args.target_rank,
        args.output_dir,
        args.readiness_timeout_s,
        args.request_timeout_s,
        args.rpc_timeout_s,
        args.probe_delay_s,
        args.require_all_survivors,
    )
    print(json.dumps(summary, sort_keys=True), flush=True)
    return 0 if summary["status"] == "observation_complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
