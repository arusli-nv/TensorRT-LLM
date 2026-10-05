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
"""Opt-in Ray WideEP regressions. Qualification repeats are explicit independent attempts."""

import json
import os
import subprocess
import time
from datetime import datetime
from pathlib import Path

import pytest
import yaml
from deployment_profile import deployment_shape
from process_probe import record, validate_cleanup
from regression import assert_engine_error, check_infrastructure
from slurm_lifecycle import ACTIVE_STATES, TERMINAL_STATES

pytestmark = pytest.mark.skipif(
    os.environ.get("WIDEEP_FT_RUN") != "1",
    reason="Requires a qualified WideEP allocation, image, Ray dependencies and model weights",
)
RUN_COUNT = int(os.environ.get("WIDEEP_QUALIFICATION_RUNS", "1"))
if RUN_COUNT not in {1, 5}:
    raise ValueError("Use one routine attempt or five qualification attempts; no implicit retries")


@pytest.mark.parametrize(
    "observation", [("healthy", attempt) for attempt in range(1, RUN_COUNT + 1)], indirect=True
)
def test_cold_start(observation: tuple[dict, float]) -> None:
    """Does a fresh WideEP deployment become ready and shut down within the recorded bounds?"""
    summary, _ = observation
    assert summary["startup_to_readiness_s"] > 0


@pytest.fixture
def observation(request) -> tuple[dict, float]:
    """Run one owned allocation and validate raw infrastructure before client assertions."""
    scenario, attempt = request.param
    required = (
        "WIDEEP_CONTAINER_IMAGE",
        "WIDEEP_MODEL_PATH",
        "WIDEEP_CONFIG_PATH",
        "WIDEEP_BUILD_MANIFEST",
        "WIDEEP_RAY_DEPENDENCIES",
        "WIDEEP_RUN_ROOT",
        "WIDEEP_SLURM_ACCOUNT",
    )
    missing = [name for name in required if not os.environ.get(name)]
    if missing:
        pytest.fail(f"Missing deployment inputs: {missing}")
    output = Path(os.environ["WIDEEP_RUN_ROOT"])
    output.mkdir(parents=True, exist_ok=True)
    environment = {**os.environ, "WIDEEP_SCENARIO": scenario}
    command = [
        "sbatch",
        "--parsable",
        f"--account={environment['WIDEEP_SLURM_ACCOUNT']}",
        f"--partition={environment.get('WIDEEP_SLURM_PARTITION', 'batch')}",
        f"--qos={environment.get('WIDEEP_SLURM_QOS', 'short')}",
        f"--output={output}/regression-%j.log",
        str(Path(__file__).with_name("launch_process_loss.slurm")),
    ]
    _, nodes, gpus = deployment_shape(
        yaml.safe_load(Path(environment["WIDEEP_CONFIG_PATH"]).read_text())
    )
    segment = environment.get("WIDEEP_SLURM_SEGMENT")
    if segment:
        if not segment.isdecimal() or int(segment) <= 0:
            raise ValueError("WIDEEP_SLURM_SEGMENT must be a positive integer")
        command.insert(1, f"--segment={segment}")
    command[1:1] = [f"--nodes={nodes}", f"--gpus-per-node={gpus}"]
    submitted = time.monotonic()
    job_id = (
        subprocess.run(
            command, env=environment, capture_output=True, text=True, check=True, timeout=30
        )
        .stdout.strip()
        .split(";")[0]
    )
    if not job_id.isdecimal():
        raise ValueError("Unexpected sbatch job identity")
    directory = output / f"process-loss-{job_id}"
    request.node.user_properties.append(("wideep_evidence", str(directory)))
    print(f"scenario={scenario} attempt={attempt} job={job_id} evidence={directory}", flush=True)
    terminal = False
    try:
        # The allocation has its own wall limit; this outer deadline also bounds queue waiting.
        deadline = submitted + float(environment.get("WIDEEP_ALLOCATION_TIMEOUT", "7200"))
        while time.monotonic() < deadline:
            result = subprocess.run(
                [
                    "sacct",
                    "-j",
                    job_id,
                    "-n",
                    "-P",
                    "--format=JobID,State,ExitCode,Submit,Start,End",
                ],
                capture_output=True,
                text=True,
                check=True,
                timeout=20,
            )
            row = next(
                (
                    line.split("|")
                    for line in result.stdout.splitlines()
                    if line.split("|")[0] == job_id
                ),
                None,
            )
            if row is not None and row[1].split()[0] in TERMINAL_STATES:
                terminal = True
                directory.mkdir(parents=True, exist_ok=True)
                (directory / "allocation_accounting.txt").write_text(result.stdout)
                if row[4] not in {"Unknown", "None", ""}:
                    record(
                        directory,
                        "allocation_timing",
                        {
                            "submit": row[3],
                            "start": row[4],
                            "end": row[5],
                            "queue_seconds": (
                                datetime.fromisoformat(row[4]) - datetime.fromisoformat(row[3])
                            ).total_seconds(),
                            "clock_source": "Slurm accounting timestamps on scheduler clock; second resolution",
                        },
                    )
                assert row[1] == "COMPLETED" and row[2] == "0:0", (
                    f"Allocation failed: {row}; evidence: {directory}"
                )
                return check_infrastructure(directory, scenario), float(
                    environment.get("WIDEEP_CLIENT_TIMEOUT", "30")
                )
            if row is not None and row[1].split()[0] not in ACTIVE_STATES:
                raise ValueError(f"Unrecognized allocation state: {row}")
            time.sleep(2)
        raise TimeoutError(f"Allocation deadline; evidence: {directory}")
    finally:
        if not terminal:
            # Keep the allocation alive long enough for its independent batch wrapper to clean up.
            try:
                subprocess.run(
                    ["scancel", "--signal=USR1", "--batch", job_id],
                    capture_output=True,
                    check=True,
                    timeout=20,
                )
                deadline = time.monotonic() + 240
                while time.monotonic() < deadline:
                    result = subprocess.run(
                        ["squeue", "-h", "-j", job_id, "-o", "%i"],
                        capture_output=True,
                        text=True,
                        check=True,
                        timeout=10,
                    )
                    if not result.stdout.strip():
                        break
                    time.sleep(0.2)
                else:
                    raise TimeoutError(f"Cooperative cleanup deadline: {directory}")
                evidence = json.loads((directory / "interruption_cleanup.json").read_text())
                if evidence["evidence"] != str(directory) or evidence["state"] != "verified":
                    raise RuntimeError(f"Interrupted cleanup is unverified: {directory}")
                validate_cleanup(
                    directory, allow_partial=True, rescue=evidence.get("rescue", False)
                )
            except (OSError, ValueError, subprocess.SubprocessError, RuntimeError):
                # Also cancel pending allocations or a wrapper that could not publish proof.
                subprocess.run(["scancel", job_id], capture_output=True, check=False, timeout=20)
                deadline = time.monotonic() + 60
                while time.monotonic() < deadline:
                    result = subprocess.run(
                        ["squeue", "-h", "-j", job_id, "-o", "%i"],
                        capture_output=True,
                        text=True,
                        check=True,
                        timeout=10,
                    )
                    if not result.stdout.strip():
                        break
                    time.sleep(0.2)
                else:
                    raise RuntimeError(
                        f"Allocation still active; cleanup unverified: {directory}"
                    ) from None
                raise


@pytest.mark.parametrize(
    "observation",
    [
        (scenario, attempt)
        for scenario in ("idle_kill", "stream_kill", "idle_term")
        for attempt in range(
            1, (10 if scenario == "stream_kill" and RUN_COUNT == 5 else RUN_COUNT) + 1
        )
    ],
    ids=lambda case: f"{case[0]}-{case[1]}",
    indirect=True,
)
def test_middle_worker_loss(observation: tuple[dict, float]) -> None:
    """Does one middle worker death become a bounded native error with complete teardown?"""
    client, bound = observation
    assert_engine_error(client, bound)


@pytest.mark.parametrize(
    "observation",
    [("idle_stop", attempt) for attempt in range(1, RUN_COUNT + 1)],
    ids=[f"idle_stop-{attempt}" for attempt in range(1, RUN_COUNT + 1)],
    indirect=True,
)
def test_worker_stall_reaches_observation_deadline(observation: tuple[dict, float]) -> None:
    """Does a verified stopped worker leave the client waiting until its bounded observation ends?"""
    client, bound = observation
    assert client["state"] == "error"
    assert client["error_type"] == "TimeoutError" and client["harness_timeout"]
    assert bound <= client["seconds_since_injection"] <= bound + 1


@pytest.mark.parametrize(
    "observation",
    [("rank0_kill", attempt) for attempt in range(1, RUN_COUNT + 1)],
    ids=[f"rank0_kill-{attempt}" for attempt in range(1, RUN_COUNT + 1)],
    indirect=True,
)
def test_rank0_loss_reports_native_error(observation: tuple[dict, float]) -> None:
    """Does losing the RPC-root rank reach the client as a native error rather than a timeout?"""
    client, bound = observation
    assert_engine_error(client, bound)


@pytest.mark.parametrize(
    "observation",
    [("frontend_kill", attempt) for attempt in range(1, RUN_COUNT + 1)],
    ids=[f"frontend_kill-{attempt}" for attempt in range(1, RUN_COUNT + 1)],
    indirect=True,
)
def test_frontend_loss_reaches_independent_client(observation: tuple[dict, float]) -> None:
    """Does verified frontend loss make the independent HTTP client fail promptly without leaked resources?"""
    client, bound = observation
    assert client["state"] == "error" and not client["harness_timeout"], client
    assert client["transport_cause"] in {
        "ConnectionRefusedError",
        "ConnectionResetError",
        "RemoteDisconnected",
    }, client
    assert client["seconds_since_injection"] <= bound, client
