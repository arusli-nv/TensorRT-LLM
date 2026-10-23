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
"""Tests for how trtllm-llmapi-launch runs and ends its MGMN worker."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.cpu_only

_LAUNCHER = Path(__file__).resolve().parents[3] / "tensorrt_llm" / "llmapi" / "trtllm-llmapi-launch"
# Reports how the launcher ran the worker and the MPI transport it inherits, then exits with $FAKE_WORKER_STATUS;
# every other python3 call of the launcher runs the real interpreter.
_FAKE_PYTHON = """#!/bin/bash
if [ "$2" = tensorrt_llm.llmapi.mgmn_worker_node ]; then
    if [ -e /proc/$$/fd/200 ]; then lock=open; else lock=closed; fi
    echo "$$ $PPID $lock ${OMPI_MCA_pml:-unset} ${OMPI_MCA_btl:-unset} ${OMPI_MCA_coll:-unset} ${*:3}"
    exit "${FAKE_WORKER_STATUS:-0}"
fi
exec %s "$@"
"""
_CLEARED_ENV = (
    "SLURM_PROCID",
    "SLURM_NTASKS",
    "PMI_RANK",
    "PMI_ID",
    "PMI_SIZE",
    "TRTLLM_CACHE_DIR",
    "FLASHINFER_WORKSPACE_BASE",
    "TRTLLM_FLASHINFER_WORKSPACE_PER_PROCESS",
    "TLLM_FAULT_TOLERANCE_MODE",
    "TLLM_FAULT_TOLERANCE_AUTHORITY",
    "OMPI_MCA_pml",
    "OMPI_MCA_btl",
    "OMPI_MCA_coll",
)


def _launch_worker(
    tmp_path: Path, recovery: bool, status: int = 0, fd_200_open: bool = False, **extra_env: str
):
    """Run the launcher as MPI rank 1 of 2; returns its pid, exit status and the worker's report."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    fake_python = bin_dir / "python3"
    fake_python.write_text(_FAKE_PYTHON % sys.executable)
    fake_python.chmod(0o755)
    env = {k: v for k, v in os.environ.items() if k not in _CLEARED_ENV}
    env.update(
        PATH=f"{bin_dir}:{env['PATH']}",
        HOME=str(tmp_path / "home"),
        TMPDIR=str(tmp_path),
        OMPI_COMM_WORLD_RANK="1",
        OMPI_COMM_WORLD_SIZE="2",
        FAKE_WORKER_STATUS=str(status),
        **extra_env,
    )
    if recovery:
        env.update(TLLM_FAULT_TOLERANCE_MODE="1", TLLM_FAULT_TOLERANCE_AUTHORITY="tcp://host:5555")
    # A descriptor 200 that the caller holds leaves the launcher a temporary workspace.
    command = ["bash", "-c", 'exec 200>/dev/null; exec "$0"', str(_LAUNCHER)]
    if not fd_200_open:
        command = [str(_LAUNCHER)]
    process = subprocess.Popen(command, env=env, stdout=subprocess.PIPE, text=True)
    stdout, _ = process.communicate(timeout=60)
    pid, ppid, lock, pml, btl, coll, *args = stdout.split()
    return process.pid, process.returncode, int(pid), int(ppid), lock, args, (pml, btl, coll)


@pytest.mark.parametrize("status", [0, 3])
def test_recovery_execs_the_worker_holding_the_workspace_lock(status, tmp_path):
    launcher, returncode, pid, _, lock, args, transport = _launch_worker(tmp_path, True, status)
    # The worker is the launched process, so the MPI runtime sees its own exit status.
    assert pid == launcher and returncode == status
    assert lock == "open"
    assert args == ["--flashinfer-lock-fd", "200"]
    assert (tmp_path / "home" / ".cache" / "tensorrt_llm" / "flashinfer" / "rank-1").is_dir()
    # Recovery runs on ob1 over TCP, without the hcoll and ucc collectives.
    assert transport == ("ob1", "self,tcp", "^hcoll,ucc")


def test_recovery_keeps_the_users_transport_choice(tmp_path):
    *_, transport = _launch_worker(tmp_path, True, OMPI_MCA_btl="self,sm,tcp", OMPI_MCA_coll="^hcoll")
    assert transport == ("ob1", "self,sm,tcp", "^hcoll")


@pytest.mark.parametrize("status", [0, 3, 137])
def test_without_recovery_the_worker_is_a_child_without_the_lock(status, tmp_path):
    launcher, returncode, pid, ppid, lock, args, transport = _launch_worker(tmp_path, False, status)
    assert pid != launcher and ppid == launcher and returncode == status
    assert lock == "closed" and args == [] and transport == ("unset", "unset", "unset")


def test_recovery_passes_the_temporary_workspace_on(tmp_path):
    launcher, returncode, pid, _, lock, args, _ = _launch_worker(tmp_path, True, fd_200_open=True)
    assert pid == launcher and returncode == 0
    flag, workspace = args
    assert flag == "--flashinfer-temporary-workspace"
    assert Path(workspace).parent == tmp_path and Path(workspace).is_dir()


def test_without_recovery_the_launcher_removes_the_temporary_workspace(tmp_path):
    _, returncode, _, _, _, args, _ = _launch_worker(tmp_path, False, status=3, fd_200_open=True)
    assert returncode == 3 and args == []
    assert not list(tmp_path.glob("trtllm-flashinfer-rank-1.*"))
