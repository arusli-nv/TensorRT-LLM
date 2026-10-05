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
"""CPU tests for exact-step accounting and bounded remote cleanup."""

import subprocess

import pytest
import slurm_lifecycle
from slurm_lifecycle import (
    cleanup_model_step,
    discover_owned_step,
    parse_step_accounting,
    wait_step_terminal,
)


class FakeLauncher:
    def __init__(self, returncode: int | None = None) -> None:
        self.returncode = returncode
        self.terminated = False
        self.killed = False

    def poll(self) -> int | None:
        return self.returncode

    def wait(self, timeout: float) -> int:
        assert timeout > 0
        if self.returncode is None:
            self.returncode = -15 if self.terminated else 143
        return self.returncode

    def terminate(self) -> None:
        self.terminated = True

    def kill(self) -> None:
        self.killed = True
        self.returncode = -9


def test_step_accounting_does_not_use_successful_batch_row() -> None:
    """Can successful allocation or batch accounting conceal a cancelled model step?"""
    output = "123|COMPLETED|0:0\n123.batch|COMPLETED|0:0\n123.0|CANCELLED by 0|0:15\n"
    assert parse_step_accounting(output, "123", "0") == {
        "job_id": "123",
        "step_id": "0",
        "state": "CANCELLED",
        "exit_code": "0:15",
    }
    assert parse_step_accounting(output, "123", "1") is None


@pytest.mark.parametrize(
    "output",
    [
        "123.0|COMPLETED|bad",
        "123.0|GUESSED|0:0",
        "123.0|COMPLETED",
        "123.0|COMPLETED|0:0\n123.0|FAILED|1:0",
    ],
)
def test_invalid_or_ambiguous_accounting_is_rejected(output: str) -> None:
    """Can malformed or conflicting step accounting establish termination?"""
    with pytest.raises(ValueError):
        parse_step_accounting(output, "123", "0")


def test_waits_for_remote_step_after_delayed_accounting(monkeypatch) -> None:
    """Does cleanup wait through missing and active accounting for exact-step termination?"""
    outputs = iter(["", "123.0|RUNNING|0:0", "123.0|FAILED|1:0"])
    calls = []

    def run(command: list[str], **kwargs) -> subprocess.CompletedProcess:
        calls.append((command, kwargs["timeout"]))
        return subprocess.CompletedProcess(command, 0, next(outputs), "")

    monkeypatch.setattr(slurm_lifecycle.subprocess, "run", run)
    monkeypatch.setattr(slurm_lifecycle.time, "sleep", lambda _: None)
    record = wait_step_terminal("123", "0", 3)
    assert record["state"] == "FAILED" and len(calls) == 3
    assert all(command[2] == "123.0" and 0 < timeout <= 3 for command, timeout in calls)


def install_commands(
    monkeypatch,
    queue: str = "123.0|wideep-model-current",
    cancel_code: int = 0,
    state: str = "CANCELLED",
    initial_state: str = "RUNNING",
) -> list[list[str]]:
    commands = []
    accounting_calls = 0

    def run(command: list[str], **kwargs) -> subprocess.CompletedProcess:
        nonlocal accounting_calls
        assert kwargs["timeout"] > 0
        commands.append(command)
        if command[0] == "squeue":
            return subprocess.CompletedProcess(command, 0, queue, "")
        if command[0] == "scancel":
            return subprocess.CompletedProcess(
                command, cancel_code, "", "cancel denied" if cancel_code else ""
            )
        assert command[0] == "sacct"
        accounting_state = initial_state if accounting_calls == 0 else state
        accounting_calls += 1
        return subprocess.CompletedProcess(command, 0, f"123.0|{accounting_state}|0:15", "")

    monkeypatch.setenv("SLURM_JOB_ID", "123")
    monkeypatch.setattr(slurm_lifecycle.subprocess, "run", run)
    return commands


def test_startup_cleanup_discovers_only_uniquely_named_owned_step(monkeypatch) -> None:
    """Does startup cleanup cancel only the uniquely named step in its own allocation?"""
    commands = install_commands(monkeypatch, queue="123.1|other-model\n123.0|wideep-model-current")
    result = cleanup_model_step(FakeLauncher(), "123", step_name="wideep-model-current")
    assert result["accounting"]["state"] == "CANCELLED"
    assert [command for command in commands if command[0] == "scancel"] == [["scancel", "123.0"]]


def test_missing_startup_identity_is_unverified_and_never_cancels_whole_job(monkeypatch) -> None:
    """Does missing step identity fail cleanup without cancelling the allocation?"""
    commands = install_commands(monkeypatch, queue="123.1|other-model")
    process = FakeLauncher()
    with pytest.raises(LookupError, match="unverified"):
        cleanup_model_step(process, "123", step_name="wideep-model-current")
    assert process.terminated and process.poll() is not None
    assert all(command[0] != "scancel" for command in commands)


def test_unresponsive_local_launcher_is_killed_and_reaped(monkeypatch) -> None:
    """Does cleanup kill and reap a launcher that ignores termination?"""
    install_commands(monkeypatch, queue="")

    class UnresponsiveLauncher(FakeLauncher):
        def wait(self, timeout: float) -> int:
            if not self.killed:
                raise subprocess.TimeoutExpired("srun", timeout)
            return super().wait(timeout)

    process = UnresponsiveLauncher()
    with pytest.raises(LookupError):
        cleanup_model_step(process, "123", step_name="wideep-model-current")
    assert process.terminated and process.killed and process.poll() == -9


@pytest.mark.parametrize(
    "queue",
    ["123.0|wideep-model-current\n123.1|wideep-model-current", "999.0|wideep-model-current"],
)
def test_ambiguous_or_foreign_step_discovery_cannot_cancel(monkeypatch, queue: str) -> None:
    """Can ambiguous names or a foreign allocation authorize step cancellation?"""
    commands = install_commands(monkeypatch, queue=queue)
    with pytest.raises(ValueError):
        discover_owned_step("123", "wideep-model-current")
    assert all(command[0] != "scancel" for command in commands)


def test_failed_cancellation_is_reported_even_when_accounting_is_terminal(monkeypatch) -> None:
    """Can terminal accounting conceal a failed cancellation command?"""
    install_commands(monkeypatch, cancel_code=1)
    process = FakeLauncher()
    with pytest.raises(RuntimeError, match="cancellation failed"):
        cleanup_model_step(process, "123", "0")
    assert process.terminated and process.poll() is not None


@pytest.mark.parametrize("state", ["COMPLETED", "CANCELLED", "FAILED"])
def test_already_terminal_remote_step_is_reaped_without_second_cancellation(
    monkeypatch, state: str
) -> None:
    """Does cleanup reap an already terminal step without cancelling it again?"""
    commands = install_commands(monkeypatch, initial_state=state)
    process = FakeLauncher()
    result = cleanup_model_step(process, "123", "0")
    assert result["accounting"]["state"] == state
    assert result["cancellation_required"] is False
    assert process.poll() is not None
    assert all(command[0] != "scancel" for command in commands)


def test_local_launcher_exit_does_not_prove_remote_step_termination(monkeypatch) -> None:
    """Can a successful local launcher exit substitute for remote-step termination?"""
    commands = install_commands(monkeypatch, state="RUNNING")
    clock = iter(index * 0.1 for index in range(100))
    monkeypatch.setattr(slurm_lifecycle.time, "monotonic", lambda: next(clock))
    monkeypatch.setattr(slurm_lifecycle.time, "sleep", lambda _: None)
    with pytest.raises(TimeoutError):
        cleanup_model_step(FakeLauncher(0), "123", "0", timeout_s=1)
    assert [command for command in commands if command[0] == "scancel"] == [["scancel", "123.0"]]


@pytest.mark.parametrize("initial_row", ["", "123.0|RUNNING|0:0"])
def test_completed_launcher_waits_for_terminal_accounting_without_cancellation(
    monkeypatch, initial_row: str
) -> None:
    """Can delayed accounting cause cleanup to cancel a client that already exited successfully?"""
    outputs = iter([initial_row, "123.0|RUNNING|0:0", "123.0|COMPLETED|0:0"])
    commands = []

    def run(command: list[str], **kwargs) -> subprocess.CompletedProcess:
        assert kwargs["timeout"] > 0
        commands.append(command)
        return subprocess.CompletedProcess(
            command, 0, next(outputs) if command[0] == "sacct" else "", ""
        )

    monkeypatch.setenv("SLURM_JOB_ID", "123")
    monkeypatch.setattr(slurm_lifecycle.subprocess, "run", run)
    monkeypatch.setattr(slurm_lifecycle.time, "sleep", lambda _: None)
    result = cleanup_model_step(FakeLauncher(0), "123", "0", timeout_s=3)
    assert result["accounting"]["state"] == "COMPLETED"
    assert result["launcher_returncode"] == 0
    assert result["cancellation_required"] is False
    assert all(command[0] != "scancel" for command in commands)


def test_cleanup_refuses_another_allocation(monkeypatch) -> None:
    """Can cleanup issue commands for an allocation other than its own?"""
    commands = install_commands(monkeypatch)
    with pytest.raises(ValueError, match="owned allocation"):
        cleanup_model_step(FakeLauncher(), "999", "0")
    assert commands == []


@pytest.mark.parametrize("deadline", [float("inf"), float("nan"), 0, -1])
def test_invalid_accounting_deadline_is_rejected(deadline: float) -> None:
    """Can an infinite, undefined or nonpositive deadline enter accounting polling?"""
    with pytest.raises(ValueError):
        wait_step_terminal("123", "0", deadline)


@pytest.mark.parametrize("state", ["RUNNING", "COMPLETED"])
def test_rescue_cancels_only_owned_active_step(monkeypatch, state):
    """Can independent rescue cancel a completed step or omit terminal verification?"""
    from slurm_lifecycle import cancel_owned_step

    commands = install_commands(monkeypatch, initial_state=state)
    result = cancel_owned_step("123", "0", 10)
    assert result["state"] in {"COMPLETED", "CANCELLED"}
    assert [command for command in commands if command[0] == "scancel"] == (
        [["scancel", "123.0"]] if state == "RUNNING" else []
    )


def test_rescue_cannot_cancel_another_allocation(monkeypatch):
    """Can independent rescue use another allocation's recorded numeric step?"""
    from slurm_lifecycle import cancel_owned_step

    commands = install_commands(monkeypatch)
    with pytest.raises(ValueError, match="owned allocation"):
        cancel_owned_step("456", "0", 10)
    assert commands == []
