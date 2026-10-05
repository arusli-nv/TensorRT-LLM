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
"""Bounded accounting and cleanup for one explicitly owned Slurm model step."""

import math
import os
import re
import subprocess
import time

TERMINAL_STATES = {
    "BOOT_FAIL",
    "CANCELLED",
    "COMPLETED",
    "DEADLINE",
    "FAILED",
    "NODE_FAIL",
    "OUT_OF_MEMORY",
    "PREEMPTED",
    "REVOKED",
    "TIMEOUT",
}
ACTIVE_STATES = {
    "CONFIGURING",
    "COMPLETING",
    "PENDING",
    "RUNNING",
    "RESIZING",
    "REQUEUED",
    "REQUEUE_FED",
    "REQUEUE_HOLD",
    "SIGNALING",
    "SPECIAL_EXIT",
    "STAGE_OUT",
    "STOPPED",
    "SUSPENDED",
}


def _step_identity(job_id: str, step_id: str) -> str:
    if not job_id.isdecimal() or not step_id.isdecimal():
        raise ValueError("An exact numeric job and model step are required")
    return f"{job_id}.{step_id}"


def _deadline(timeout_s: float) -> float:
    if not math.isfinite(timeout_s) or timeout_s <= 0:
        raise ValueError("A finite positive lifecycle deadline is required")
    return time.monotonic() + timeout_s


def _remaining(deadline: float) -> float:
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise TimeoutError("Slurm model-step lifecycle deadline exceeded")
    return remaining


def parse_step_accounting(stdout: str, job_id: str, step_id: str) -> dict | None:
    """Select the exact step, excluding allocation, batch, and unrelated step rows."""
    expected = _step_identity(job_id, step_id)
    matches = []
    for line in stdout.splitlines():
        fields = line.strip().split("|")
        if fields[0] != expected:
            continue
        if len(fields) != 3:
            raise ValueError("Malformed model-step accounting row")
        state = fields[1].split()[0] if fields[1].split() else ""
        if state not in TERMINAL_STATES | ACTIVE_STATES or not re.fullmatch(r"\d+:\d+", fields[2]):
            raise ValueError("Invalid model-step state or exit code")
        matches.append(
            {"job_id": job_id, "step_id": step_id, "state": state, "exit_code": fields[2]}
        )
    if len(matches) > 1:
        raise ValueError("Ambiguous model-step accounting rows")
    return matches[0] if matches else None


def _read_step_accounting(job_id: str, step_id: str, deadline: float) -> dict | None:
    expected = _step_identity(job_id, step_id)
    result = subprocess.run(
        [
            "sacct",
            "-j",
            expected,
            "--noheader",
            "--parsable2",
            "--format=JobID%64,State%64,ExitCode",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=min(10, _remaining(deadline)),
    )
    return parse_step_accounting(result.stdout, job_id, step_id)


def wait_step_terminal(job_id: str, step_id: str, timeout_s: float) -> dict:
    _step_identity(job_id, step_id)
    deadline = _deadline(timeout_s)
    while True:
        record = _read_step_accounting(job_id, step_id, deadline)
        if record is not None and record["state"] in TERMINAL_STATES:
            return record
        time.sleep(min(0.2, _remaining(deadline)))


def discover_owned_step(job_id: str, step_name: str, timeout_s: float = 10) -> str | None:
    if not job_id.isdecimal() or not step_name:
        raise ValueError("An owned job and unique model-step name are required")
    deadline = _deadline(timeout_s)
    result = subprocess.run(
        ["squeue", "--steps", "-j", job_id, "--format=%i|%j", "--noheader"],
        check=True,
        capture_output=True,
        text=True,
        timeout=_remaining(deadline),
    )
    matches = []
    for line in result.stdout.splitlines():
        fields = line.strip().split("|")
        if len(fields) != 2 or fields[1] != step_name:
            continue
        identity = fields[0].split(".")
        if len(identity) != 2 or identity[0] != job_id:
            raise ValueError("Named model step belongs to another allocation")
        _step_identity(*identity)
        matches.append(identity[1])
    if len(matches) > 1:
        raise ValueError("Ambiguous named model-step discovery")
    return matches[0] if matches else None


def cleanup_model_step(
    process: subprocess.Popen,
    job_id: str,
    step_id: str | None = None,
    timeout_s: float = 180,
    *,
    step_name: str | None = None,
) -> dict:
    """Cancel only the owned step and independently verify remote and local termination."""
    if os.environ.get("SLURM_JOB_ID") != job_id:
        raise ValueError("Cleanup must run inside the owned allocation")
    deadline = _deadline(timeout_s)
    try:
        if step_id is None:
            if step_name is None:
                raise ValueError("Cleanup requires a step identity or unique step name")
            step_id = discover_owned_step(job_id, step_name, min(10, _remaining(deadline)))
            if step_id is None:
                raise LookupError(
                    "Owned model step could not be discovered; remote cleanup is unverified"
                )
        expected = _step_identity(job_id, step_id)
        accounting = _read_step_accounting(job_id, step_id, deadline)
        if process.poll() is not None and (
            accounting is None or accounting["state"] not in TERMINAL_STATES
        ):
            # Accounting can lag behind srun exit; reserve time for cancellation if it stays active.
            try:
                accounting = wait_step_terminal(job_id, step_id, min(10, _remaining(deadline) / 2))
            except TimeoutError:
                pass
        cancellation_required = accounting is None or accounting["state"] not in TERMINAL_STATES
        if cancellation_required:
            cancelled = subprocess.run(
                ["scancel", expected],
                check=False,
                capture_output=True,
                text=True,
                timeout=min(10, _remaining(deadline)),
            )
            accounting = wait_step_terminal(job_id, step_id, max(0.001, _remaining(deadline) - 5))
            if cancelled.returncode != 0:
                raise RuntimeError(f"Model-step cancellation failed: {cancelled.stderr.strip()}")
        return {
            "accounting": accounting,
            "cancellation_required": cancellation_required,
            "launcher_returncode": process.wait(timeout=_remaining(deadline)),
        }
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=min(2, max(0.001, deadline - time.monotonic())))
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=max(0.001, deadline - time.monotonic()))


def cancel_owned_step(job_id: str, step_id: str, timeout_s: float) -> dict:
    """Terminate an exactly identified remote step when its local launcher is unavailable."""
    if os.environ.get("SLURM_JOB_ID") != job_id:
        raise ValueError("Cleanup must run inside the owned allocation")
    deadline = _deadline(timeout_s)
    row = _read_step_accounting(job_id, step_id, deadline)
    if row is None or row["state"] not in TERMINAL_STATES:
        subprocess.run(
            ["scancel", _step_identity(job_id, step_id)],
            check=True,
            capture_output=True,
            text=True,
            timeout=min(10, _remaining(deadline)),
        )
        row = wait_step_terminal(job_id, step_id, _remaining(deadline))
    return row
