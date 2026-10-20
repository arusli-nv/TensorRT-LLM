# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
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
"""Tests for how trtllm-llmapi-launch exits with its MGMN worker's status."""

import os
import subprocess
from pathlib import Path

import pytest

pytestmark = pytest.mark.cpu_only

_LAUNCHER = Path(__file__).resolve().parents[3] / "tensorrt_llm" / "llmapi" / "trtllm-llmapi-launch"
_FUNCTIONS = ("remove_flashinfer_temporary_workspace", "exit_with_worker_status")


def _launcher_functions() -> str:
    lines = _LAUNCHER.read_text().splitlines()
    source = []
    for name in _FUNCTIONS:
        start = lines.index(f"function {name} {{")
        end = lines.index("}", start)
        source.extend(lines[start : end + 1])
    return "\n".join(source)


def _exit_with_worker_status(
    status: int, ft_mode: str | None, tmp_path: Path, authority: str | None = "tcp://host:5555"
) -> int:
    workspace = tmp_path / "flashinfer"
    workspace.mkdir()
    script = "\n".join(
        [
            'log_stderr() { echo "$@" >&2; }',
            f"flashinfer_temporary_workspace={workspace}",
            _launcher_functions(),
            f"exit_with_worker_status {status}",
        ]
    )
    env = {
        k: v
        for k, v in os.environ.items()
        if k not in ("TLLM_FAULT_TOLERANCE_MODE", "TLLM_FAULT_TOLERANCE_AUTHORITY")
    }
    if ft_mode is not None:
        env["TLLM_FAULT_TOLERANCE_MODE"] = ft_mode
    if authority is not None:
        env["TLLM_FAULT_TOLERANCE_AUTHORITY"] = authority
    result = subprocess.run(["bash", "-c", script], env=env, timeout=30)
    return result.returncode


@pytest.mark.parametrize(
    "status, ft_mode, expected_returncode",
    [
        (137, None, 137),  # FT off: plain exit status
        (143, "0", 143),
        (1, "0", 1),
        (0, "1", 0),
        (75, "1", 75),  # FT on, every rank exits for a deployment restart: ends the job
        # FT on, any other abnormal end of the worker (a lost rank): the launcher dies by SIGKILL.
        (137, "1", -9),
        (143, "1", -9),
        (139, "1", -9),  # SIGSEGV
        (134, "1", -9),  # SIGABRT
        (135, "1", -9),  # SIGBUS
        (1, "1", -9),  # e.g. a CUDA error
        (255, "1", -9),
    ],
)
def test_exit_with_worker_status(status, ft_mode, expected_returncode, tmp_path):
    assert _exit_with_worker_status(status, ft_mode, tmp_path) == expected_returncode


@pytest.mark.parametrize("status", [1, 137, 139])
def test_detection_only_mode_ends_the_job_with_the_worker_status(status, tmp_path):
    """Without an authority nothing recovers a lost rank, so its exit status ends the job as before."""
    assert _exit_with_worker_status(status, "1", tmp_path, authority=None) == status


def test_signal_exit_removes_temporary_workspace(tmp_path):
    assert _exit_with_worker_status(137, "1", tmp_path) == -9
    assert not (tmp_path / "flashinfer").exists()
