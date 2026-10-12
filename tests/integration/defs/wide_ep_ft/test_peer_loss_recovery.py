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
"""WideEP peer-loss recovery end to end: EP4 survives the SIGKILL of one worker."""

import importlib.metadata
import json
import os
import secrets
import shutil
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest

from ..conftest import llm_models_root

VICTIM = 2
WORLD_SIZE = 4
DRIVER = Path(__file__).with_name("peer_loss_driver.py")
KILL_MARKER = "PEER_LOSS_DRIVER_KILL"
# Model loading, CUDA graph capture and both request rounds.
SERVE_TIMEOUT_S = 1800
# Shutdown of a world that lost a rank ends in MPI_Abort after a 60 s worker grace period.
TEARDOWN_TIMEOUT_S = 180
RETRY_ERROR = f"Attention DP rank {VICTIM} failed while serving this request; retry it"


def _ulfm_unavailable_reason() -> str | None:
    if shutil.which("mpirun") is None or shutil.which("ompi_info") is None:
        return "requires Open MPI"
    if int(importlib.metadata.version("mpi4py").split(".")[0]) < 4:
        return "requires mpi4py>=4"
    info = subprocess.run(["ompi_info"], capture_output=True, text=True, timeout=60)
    if "ftmpi" not in info.stdout:
        return "requires Open MPI with the ULFM (ftmpi) extension"
    return None


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _wait_for_teardown(mpirun: subprocess.Popen, result_path: Path) -> None:
    """Wait until mpirun exits; any exit status counts, since a degraded world ends in Abort."""
    deadline = time.monotonic() + SERVE_TIMEOUT_S
    while mpirun.poll() is None and not result_path.exists() and time.monotonic() < deadline:
        time.sleep(5)
    try:
        mpirun.wait(timeout=TEARDOWN_TIMEOUT_S if result_path.exists() else 0)
    except subprocess.TimeoutExpired:
        os.killpg(mpirun.pid, signal.SIGKILL)
        mpirun.wait()
        pytest.fail(
            "mpirun did not exit within the bound"
            + (" after the driver finished" if result_path.exists() else "")
        )


def _assert_answered(rows: list[dict]) -> None:
    for row in rows:
        assert "error" not in row, row
        assert row["expected"] in row["text"], row


@pytest.mark.skip_less_device(WORLD_SIZE)
@pytest.mark.skip_device_not_contain(["GB200"])
@pytest.mark.parametrize("mode", ["idle", "inflight"])
def test_peer_loss_recovery(mode: str, tmp_path: Path):
    reason = _ulfm_unavailable_reason()
    if reason is not None:
        pytest.skip(reason)
    model = Path(llm_models_root()) / "DeepSeek-V3-Lite" / "bf16"
    if not model.exists():
        pytest.skip(f"{model} does not exist")

    port = _free_port()
    # Ranks come from mpirun, not from an enclosing Slurm or PMIx launch.
    launcher_prefixes = ("SLURM_", "PMIX_", "PMI_", "OMPI_MCA_ess", "PRTE_MCA_ras")
    env = dict(
        {k: v for k, v in os.environ.items() if not k.startswith(launcher_prefixes)},
        TLLM_FAULT_TOLERANCE_MODE="1",
        TLLM_FAULT_TOLERANCE_AUTHORITY=f"tcp://127.0.0.1:{port}",
        TLLM_FAULT_TOLERANCE_AUTHORITY_KEY=secrets.token_hex(32),
        # The fault-tolerant MoE all-to-all runs the NVLinkOneSided fence path.
        TRTLLM_FORCE_COMM_METHOD="NVLINK_ONE_SIDED",
        TLLM_LOG_LEVEL="info",
        PYTHONUNBUFFERED="1",
    )
    # Recovery aborts kernels through a side stream, which one hardware queue would serialize.
    env.pop("CUDA_DEVICE_MAX_CONNECTIONS", None)
    authority_log = tmp_path / "authority.log"
    mpirun_log = tmp_path / "mpirun.log"
    result_path = tmp_path / "result.json"

    with open(authority_log, "w") as authority_out, open(mpirun_log, "w") as mpirun_out:
        authority = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "tensorrt_llm.executor.recovery.authority",
                "--endpoint",
                f"tcp://127.0.0.1:{port}",
                "--world-size",
                str(WORLD_SIZE),
            ],
            env=dict(env, CUDA_VISIBLE_DEVICES=""),
            stdout=authority_out,
            stderr=subprocess.STDOUT,
        )
        try:
            # ULFM needs the ob1 PML and collectives without hcoll/ucc.
            mpirun = subprocess.Popen(
                [
                    "mpirun",
                    "--with-ft",
                    "ulfm",
                    "--allow-run-as-root",
                    "--oversubscribe",
                    "--mca",
                    "pml",
                    "ob1",
                    "--mca",
                    "btl",
                    "self,tcp",
                    "--mca",
                    "coll",
                    "^hcoll,ucc",
                    "-np",
                    str(WORLD_SIZE),
                    "trtllm-llmapi-launch",
                    sys.executable,
                    str(DRIVER),
                    "--model",
                    str(model),
                    "--mode",
                    mode,
                    "--victim",
                    str(VICTIM),
                    "--output",
                    str(result_path),
                ],
                env=env,
                stdout=mpirun_out,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            _wait_for_teardown(mpirun, result_path)
        finally:
            authority.send_signal(signal.SIGTERM)
            authority.wait(timeout=30)

    output = mpirun_log.read_text(errors="replace")
    assert result_path.exists(), f"The driver wrote no result:\n{output[-20000:]}"
    result = json.loads(result_path.read_text())
    assert "after_loss" in result, f"The driver did not finish serving:\n{output[-20000:]}"

    _assert_answered(result["healthy"])
    if mode == "inflight":
        during = {row["rank"]: row for row in result["during_loss"]}
        assert RETRY_ERROR in during.pop(VICTIM).get("error", ""), result["during_loss"]
        _assert_answered(list(during.values()))
    _assert_answered(result["after_loss"])

    before_kill, kill_marker, after_kill = output.partition(KILL_MARKER)
    assert kill_marker, "The driver did not kill the victim"
    for rank in range(WORLD_SIZE):
        if rank != VICTIM:
            assert f"Rank {rank}: resumed serving without rank {VICTIM}" in after_kill, rank
    assert "CUDA graph capture" in before_kill, "Decode CUDA graphs were not captured"
    assert "CUDA graph capture" not in after_kill, "Recovery captured CUDA graphs"
    assert f"committed generation 1 without rank {VICTIM}" in authority_log.read_text()
