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
"""Separate infrastructure validity from the narrowly scoped native client expectation."""

import json
import math
from pathlib import Path

import yaml
from deployment_profile import (
    deployment_shape,
    file_sha256,
    validate_execution_log,
    validate_healthy,
    validate_ray_version,
    validate_runtime,
)
from process_loss_controller import validate_head_ready
from process_probe import IDENTITY_KEYS, WORKER_SIGNALS, validate_cleanup, validate_injection


def check_infrastructure(directory: Path, scenario: str) -> dict:
    """Reject incomplete evidence before any known client issue can become XFAIL."""

    def read(name: str) -> dict:
        value = json.loads((directory / f"{name}.json").read_text())
        if value.get("run_id") != directory.name:
            raise ValueError(f"Stale {name} evidence")
        return value

    run, summary = read("run"), read("summary")
    frontend_loss = scenario == "frontend_kill"
    if (
        run["scenario"] != scenario
        or summary["driver_returncode"] != (137 if frontend_loss else 0)
        or summary["cleanup_errors"]
    ):
        raise ValueError("Scenario or teardown failed")
    if (
        summary["startup_to_readiness_s"] > run["startup_timeout_s"]
        or summary["cleanup_seconds"] > run["cleanup_timeout_s"]
    ):
        raise ValueError("Readiness or cleanup exceeded its deadline")
    for role in ("client", "services"):
        step = summary[f"wideep-{role}-{run['job_id']}"]
        published = read("step-driver" if role == "client" else "step-service-0")
        if (
            step["accounting"]["step_id"] != published["step_id"]
            or published["job_id"] != run["job_id"]
        ):
            raise ValueError("Accounting does not identify the originally owned step")
        if (
            step["accounting"]["job_id"] != run["job_id"]
            or not step["accounting"]["step_id"].isdecimal()
        ):
            raise ValueError("Teardown refers to another job or an invalid step")
        if role == "client" and frontend_loss:
            if (
                step["accounting"]["state"] != "CANCELLED"
                or step["accounting"]["exit_code"] != "0:9"
                or step["cancellation_required"]
            ):
                raise ValueError("Frontend step lacks natural SIGKILL termination")
            continue
        if step["accounting"]["state"] not in {"COMPLETED", "CANCELLED"}:
            raise ValueError("Owned step did not terminate cleanly")
        if role == "client" and step["accounting"]["exit_code"] != "0:0":
            raise ValueError("Native client process exit failed")
        if role == "client" and (
            step["cancellation_required"] or step["accounting"]["state"] != "COMPLETED"
        ):
            raise ValueError("Native client teardown required harness cancellation")
    if not frontend_loss and read("shutdown")["state"] != "complete":
        raise ValueError("Native LLM teardown is incomplete")
    head, launch = read("head_ready"), read("launch")
    validate_head_ready(head, directory, launch["requested_address"])
    service = read("step-service-0")["identity"]
    child = head["head_cli_identity"]
    if (
        head["address"] != launch["address"]
        or launch["driver"][launch["driver"].index("--address") + 1] != head["address"]
        or head["producer_host"] != service["hostname"]
        or any(child[key] != service[key] for key in ("hostname", "boot_id", "uid"))
        or child["pid"] == service["pid"]
        or child["start_ticks"] < service["start_ticks"]
    ):
        raise ValueError("GCS readiness does not identify the originally owned head launcher")
    validate_healthy(read("healthy"), directory.name)
    manifest = json.loads((directory / "build_manifest.json").read_text())
    config = yaml.safe_load((directory / "config.yaml").read_text())
    ranks, node_count, _ = deployment_shape(config)
    runtime = read("runtime")
    validate_runtime(runtime, manifest)
    validate_ray_version(runtime.get("ray_version"))
    if read("image")["sha256"] != manifest["runtime_image_sha256"]:
        raise ValueError("Runtime image provenance differs from its manifest")
    for name, digest in read("test_sources")["sha256"].items():
        if Path(name).name != name or file_sha256(directory / name) != digest:
            raise ValueError("Frozen test source changed")
    overlay = directory / "python_overlay.json"
    if overlay.exists():
        files = read("python_overlay")["files"]
        installs = [json.loads(path.read_text()) for path in directory.glob("runtime-*.json")]
        if len(installs) != node_count + 1 or any(
            row["run_id"] != directory.name for row in installs
        ):
            raise ValueError("Incomplete overlay installation evidence")
        for relative, digest in files.items():
            if file_sha256(directory / "overlay" / relative) != digest or any(
                row["files"][relative]["installed_sha256"] != digest for row in installs
            ):
                raise ValueError("Installed runtime overlay differs from frozen source")
    raw_qualification = validate_execution_log(
        (directory / "model.log").read_text(),
        config,
        json.loads((directory / "model_config.json").read_text()),
    )
    qualified = read("qualification")
    if any(
        qualified[key] != json.loads(json.dumps(value)) for key, value in raw_qualification.items()
    ):
        raise ValueError("Qualification summary differs from native model evidence")
    workers = read("workers")["workers"]
    if len(workers) != ranks or sorted(worker["rank"] for worker in workers) != list(range(ranks)):
        raise ValueError("Original worker map is incomplete")
    validate_cleanup(directory)
    if scenario == "healthy":
        benchmark, initialized = read("cold_start"), read("initialized")
        if (
            benchmark["startup_to_readiness_s"] != summary["startup_to_readiness_s"]
            or benchmark["api_construction_s"] != initialized["construction_seconds"]
            or benchmark["per_rank_native_metrics"]
            != {str(worker["rank"]): worker["startup_metrics"] for worker in workers}
        ):
            raise ValueError("Cold-start report differs from its original timing evidence")
        for elapsed in (benchmark["startup_to_readiness_s"], benchmark["api_construction_s"]):
            if not isinstance(elapsed, (int, float)) or not math.isfinite(elapsed) or elapsed <= 0:
                raise ValueError("Invalid cold-start timing interval")
        return summary
    injection, intent, trigger = (
        read("injection"),
        read("injection_intent"),
        read("trigger"),
    )
    target = injection["target"]
    if frontend_loss:
        frontend, driver, death, http_healthy = (
            read("frontend"),
            read("step-driver"),
            read("death"),
            read("http_healthy"),
        )
        pins = ("hostname", "pid", "start_ticks", "boot_id")
        if (
            any(
                target[key] != frontend["identity"][key] or target[key] != driver["identity"][key]
                for key in pins
            )
            or target != intent["target"]
            or target != death["target"]
            or death["proof"] != "pidfd readable after SIGKILL"
            or trigger["event"] != "healthy_http_completed"
            or trigger["response_id"] != http_healthy["response"]["id"]
            or not http_healthy["response"]["choices"]
            or "Paris" not in http_healthy["response"]["choices"][0]["text"]
        ):
            raise ValueError(
                "Frontend loss lacks matching original identity, healthy trigger or pinned death"
            )
    else:
        before = read("pre_injection")
        if before["fatal_error"] is not None or before["is_shutdown"]:
            raise ValueError("Engine was already terminal before the tested failure")
        if (
            before["producer_host"] != injection["producer_host"]
            or before["monotonic_s"] > injection["monotonic_s"]
        ):
            raise ValueError(
                "Pre-injection state does not precede the action on its producer clock"
            )
        if target["rank"] != (0 if scenario == "rank0_kill" else ranks // 2):
            raise ValueError("Scenario does not target its declared worker rank")
        original = next(worker for worker in workers if worker["rank"] == target["rank"])
        if any(target[key] != original[key] for key in IDENTITY_KEYS):
            raise ValueError("Injection target differs from original worker map")
        validate_injection(
            target, {**intent["target"], "run_id": directory.name}, scenario, trigger
        )
    if (
        intent["scenario"] != scenario
        or intent["signal"] != ("SIGKILL" if frontend_loss else WORKER_SIGNALS[scenario].name)
        or injection["signal"] != intent["signal"]
        or intent["trigger"] != trigger
    ):
        raise ValueError("Failure lacks matching single-action intent")
    if scenario == "idle_stop":
        stopped = read("stopped")
        if (
            any(stopped["target"][key] != target[key] for key in IDENTITY_KEYS)
            or any(
                stopped["observed"][key] != target[key]
                for key in ("hostname", "pid", "start_ticks", "boot_id")
            )
            or stopped["observed"]["state"] != "T"
        ):
            raise ValueError("Independent OS evidence does not confirm the pinned stopped process")
    elif not frontend_loss:
        death = read("death")
        if death["actor_id"] != target["actor_id"] or death["error_type"] != "ActorDiedError":
            raise ValueError("Ray death evidence does not identify the selected actor")
    client = read("client_result")
    if client.get("error_type") == "EngineDeadError":
        assert_engine_error(client, run["client_timeout_s"])
        if "Ray model worker rank" in client["error"] and (
            f"Ray model worker rank {target['rank']} (actor {target['actor_id']}) died"
            not in client["error"]
        ):
            raise ValueError("Terminal actor-death cause identifies a different worker")
    elapsed = client.get("seconds_since_injection")
    if (
        not isinstance(elapsed, (int, float))
        or not math.isfinite(elapsed)
        or elapsed < 0
        or elapsed > run["client_timeout_s"] + 1
    ):
        raise ValueError(
            "Client observation exceeded its deadline plus one-second recording allowance"
        )
    if scenario == "stream_kill":
        snapshots = [
            json.loads(path.read_text()) for path in sorted((directory / "stream").glob("*.json"))
        ]
        if not snapshots or any(
            row["run_id"] != directory.name
            or row["sequence"] != index
            or row["native_request_id"] != trigger["native_request_id"]
            for index, row in enumerate(snapshots)
        ):
            raise ValueError("Stream evidence is incomplete or belongs to another request")
        for previous, following in zip(snapshots, snapshots[1:]):
            if (
                following["token_ids"][: len(previous["token_ids"])] != previous["token_ids"]
                or following["producer_host"] != previous["producer_host"]
                or following["monotonic_s"] < previous["monotonic_s"]
            ):
                raise ValueError("Stream prefix or producer clock changed")
        if any(row["finished"] for row in snapshots):
            raise ValueError("Failed stream contains a final output")
        first = next(row for row in snapshots if row["token_ids"] and not row["finished"])
        if any(
            first[key] != trigger[key]
            for key in ("sequence", "native_request_id", "token_ids", "text", "finished")
        ):
            raise ValueError("Trigger differs from first persisted nonfinal output")
        partial = client.get("last_partial_output")
        if partial is None or any(
            partial[key] != snapshots[-1][key]
            for key in ("sequence", "native_request_id", "token_ids", "text", "finished")
        ):
            raise ValueError("Terminal partial does not match the last persisted snapshot")
    return client


def assert_engine_error(client: dict, bound: float) -> None:
    """Did native inference report EngineDeadError within the injection-to-result bound?"""
    assert client["state"] == "error", client
    assert client["error_type"] == "EngineDeadError" and not client["harness_timeout"], client
    assert client["error"].startswith(
        (
            "Engine has died: RPCStreamingError: ",
            "Engine has died: RuntimeError: Ray model worker rank ",
        )
    ), client
    assert client["seconds_since_injection"] <= bound, client
