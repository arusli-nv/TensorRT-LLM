# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""WideEP peer-loss recovery end to end: EP4 survives the loss of workers, and the loss of its authority."""

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
import urllib.error
import urllib.request
from pathlib import Path

import pytest
import yaml

from ..conftest import llm_models_root
from .peer_loss_driver import PROMPTS, SHORT_MAX_TOKENS, llm_kwargs

VICTIM = 2
WORLD_SIZE = 4
DRIVER = Path(__file__).with_name("peer_loss_driver.py")
WORKER_HOOKS = Path(__file__).with_name("worker_hooks")
KILL_MARKER = "PEER_LOSS_DRIVER_KILL"
SELF_KILL_MARKER = "WIDEEP_FT_TEST_SELF_KILL"
# Model loading, CUDA graph capture and both request rounds.
SERVE_TIMEOUT_S = 1800
# Shutdown of a world that lost a rank ends in MPI_Abort after a 60 s worker grace period.
TEARDOWN_TIMEOUT_S = 180
# The server's view of the authority is at most 1 s old and a query waits 0.5 s.
HEALTH_SETTLE_S = 30
# ULFM needs the ob1 PML and collectives without hcoll/ucc.
MPIRUN = [
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
]


def _ulfm_unavailable_reason() -> str | None:
    if shutil.which("mpirun") is None or shutil.which("ompi_info") is None:
        return "requires Open MPI"
    if int(importlib.metadata.version("mpi4py").split(".")[0]) < 4:
        return "requires mpi4py>=4"
    info = subprocess.run(["ompi_info"], capture_output=True, text=True, timeout=60)
    if "ftmpi" not in info.stdout:
        return "requires Open MPI with the ULFM (ftmpi) extension"
    return None


def _model() -> Path:
    reason = _ulfm_unavailable_reason()
    if reason is not None:
        pytest.skip(reason)
    model = Path(llm_models_root()) / "DeepSeek-V3-Lite" / "nvfp4_moe_only"
    if not model.exists():
        pytest.skip(f"{model} does not exist")
    return model


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _fault_tolerance_env(tmp_path: Path, ft_config: dict | None, **extra: str) -> dict:
    """The environment of the authority and every rank; ``ft_config`` becomes the FT config file."""
    # Ranks come from mpirun, not from an enclosing Slurm or PMIx launch.
    launcher_prefixes = ("SLURM_", "PMIX_", "PMI_", "OMPI_MCA_ess", "PRTE_MCA_ras")
    env = dict(
        {k: v for k, v in os.environ.items() if not k.startswith(launcher_prefixes)},
        TLLM_FAULT_TOLERANCE_MODE="1",
        TLLM_FAULT_TOLERANCE_AUTHORITY=f"tcp://127.0.0.1:{_free_port()}",
        TLLM_FAULT_TOLERANCE_AUTHORITY_KEY=secrets.token_hex(32),
        # The fault-tolerant MoE all-to-all runs the NVLinkOneSided fence path.
        TRTLLM_FORCE_COMM_METHOD="NVLINK_ONE_SIDED",
        TLLM_LOG_LEVEL="info",
        PYTHONUNBUFFERED="1",
        **extra,
    )
    # Recovery aborts kernels through a side stream, which one hardware queue would serialize.
    env.pop("CUDA_DEVICE_MAX_CONNECTIONS", None)
    if ft_config is not None:
        config_path = tmp_path / "fault_tolerance.yaml"
        config_path.write_text(yaml.safe_dump(ft_config))
        env["TLLM_FAULT_TOLERANCE_CONFIG"] = str(config_path)
    return env


def _start_authority(env: dict, log) -> subprocess.Popen:
    return subprocess.Popen(
        [
            sys.executable,
            "-m",
            "tensorrt_llm.executor.recovery.authority",
            "--endpoint",
            env["TLLM_FAULT_TOLERANCE_AUTHORITY"],
            "--world-size",
            str(WORLD_SIZE),
        ],
        env=dict(env, CUDA_VISIBLE_DEVICES=""),
        stdout=log,
        stderr=subprocess.STDOUT,
    )


def _stop_authority(authority: subprocess.Popen) -> None:
    authority.send_signal(signal.SIGTERM)
    try:
        authority.wait(timeout=30)
    except subprocess.TimeoutExpired:
        authority.kill()


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


def _run_driver(
    tmp_path: Path,
    mode: str,
    victims: list[int],
    *driver_args: str,
    overlap: bool = False,
    ft_config: dict | None = None,
    **extra_env: str,
) -> tuple[dict, str, str]:
    """Serve with the driver under ULFM mpirun; return its result, the mpirun log and the authority log."""
    model = _model()
    env = _fault_tolerance_env(tmp_path, ft_config, **extra_env)
    authority_log = tmp_path / "authority.log"
    mpirun_log = tmp_path / "mpirun.log"
    result_path = tmp_path / "result.json"

    with open(authority_log, "w") as authority_out, open(mpirun_log, "w") as mpirun_out:
        authority = _start_authority(env, authority_out)
        try:
            mpirun = subprocess.Popen(
                [
                    *MPIRUN,
                    sys.executable,
                    str(DRIVER),
                    "--model",
                    str(model),
                    "--mode",
                    mode,
                    "--victims",
                    *map(str, victims),
                    "--output",
                    str(result_path),
                    *(["--overlap"] if overlap else []),
                    *driver_args,
                ],
                env=env,
                stdout=mpirun_out,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            _wait_for_teardown(mpirun, result_path)
        finally:
            _stop_authority(authority)

    output = mpirun_log.read_text(errors="replace")
    assert result_path.exists(), f"The driver wrote no result:\n{output[-20000:]}"
    result = json.loads(result_path.read_text())
    assert "after_loss" in result, f"The driver did not finish serving:\n{output[-20000:]}"
    return result, output, authority_log.read_text()


def _assert_answered(rows: list[dict]) -> None:
    for row in rows:
        assert "error" not in row, row
        assert row["expected"] in row["text"], row


def _assert_recovered(
    result: dict, output: str, authority_text: str, mode: str, lost: list[int]
) -> str:
    """Every request answered, the survivors resumed without ``lost`` and nothing was rebuilt.

    Returns the mpirun log from the first fault on.
    """
    _assert_answered(result["healthy"])
    if mode == "inflight":
        # The lost ranks' requests streamed nothing, so the survivors ran them again from their prompts.
        _assert_answered(result["during_loss"])
        assert f"Running {len(lost)} requests of the lost ranks again" in output, output[-20000:]
    _assert_answered(result["after_loss"])

    before_kill, kill_marker, after_kill = output.partition(KILL_MARKER)
    assert kill_marker, "The driver did not fault a victim"
    for rank in range(WORLD_SIZE):
        if rank not in lost:
            assert f"Rank {rank}: resumed serving without ranks {lost}" in after_kill, rank
    assert "CUDA graph capture" in before_kill, "Decode CUDA graphs were not captured"
    assert "CUDA graph capture" not in after_kill, "Recovery captured CUDA graphs"
    assert f"committed generation 1 without ranks {lost}" in authority_text, authority_text[-20000:]
    return after_kill


@pytest.mark.skip_less_device(WORLD_SIZE)
@pytest.mark.skip_device_not_contain(["GB200"])
@pytest.mark.parametrize("overlap", [False, True], ids=["sync", "overlap"])
@pytest.mark.parametrize("mode", ["idle", "inflight"])
def test_peer_loss_recovery(mode: str, overlap: bool, tmp_path: Path):
    result, output, authority_text = _run_driver(tmp_path, mode, [VICTIM], overlap=overlap)
    _assert_recovered(result, output, authority_text, mode, [VICTIM])


@pytest.mark.skip_less_device(WORLD_SIZE)
@pytest.mark.skip_device_not_contain(["GB200"])
def test_two_ranks_lost_together(tmp_path: Path):
    """Ranks 2 and 3 die in one decode step and one recovery serves on without both.

    ``failure_domain: node`` needs ranks on more than one host; on one host each rank is its own domain, and
    the placement (every expert also on the rank two further) covers the union of two.
    """
    result, output, authority_text = _run_driver(
        tmp_path,
        "inflight",
        [2, 3],
        "--replica-offset",
        "2",
        ft_config=dict(failure_domain="rank"),
    )
    _assert_recovered(result, output, authority_text, "inflight", [2, 3])


@pytest.mark.skip_less_device(WORLD_SIZE)
@pytest.mark.skip_device_not_contain(["GB200"])
def test_rank_lost_during_recovery(tmp_path: Path):
    """Rank 2 dies in decode and rank 3 right after reporting it: the authority plans the recovery again."""
    result, output, authority_text = _run_driver(
        tmp_path,
        "inflight",
        [2],
        "--also-lost",
        "3",
        "--replica-offset",
        "2",
        ft_config=dict(failure_domain="rank"),
        PYTHONPATH=os.pathsep.join(filter(None, [str(WORKER_HOOKS), os.environ.get("PYTHONPATH")])),
        WIDEEP_FT_TEST_KILL_AFTER_STOPPED="3",
    )
    _assert_recovered(result, output, authority_text, "inflight", [2, 3])
    assert f"{SELF_KILL_MARKER} rank=3" in output, "Rank 3 did not end itself after Stopped"
    assert "ranks [3] failed during the recovery; planning again without ranks [2, 3]" in (
        authority_text
    ), authority_text[-20000:]


@pytest.mark.skip_less_device(WORLD_SIZE)
@pytest.mark.skip_device_not_contain(["GB200"])
def test_restore_into_borrowed_slots(tmp_path: Path):
    """The extra slots hold no memory until rank 2 dies in decode; the restore maps memory behind them."""
    result, output, authority_text = _run_driver(
        tmp_path, "inflight", [VICTIM], ft_config=dict(extra_slots="borrowed")
    )
    after_kill = _assert_recovered(result, output, authority_text, "inflight", [VICTIM])
    assert "behind borrowed expert slots" in after_kill, output[-20000:]


@pytest.mark.skip_less_device(WORLD_SIZE)
@pytest.mark.skip_device_not_contain(["GB200"])
def test_hung_rank_fenced(tmp_path: Path):
    """Rank 2 is SIGSTOPped in decode; the authority confirms the hang, the workers fence it, survivors resume."""
    result, output, authority_text = _run_driver(
        tmp_path,
        "inflight",
        [VICTIM],
        "--fault",
        "stop",
        ft_config=dict(hang_detection=True, hang_confirm_s=10.0),
    )
    _assert_recovered(result, output, authority_text, "inflight", [VICTIM])
    assert any(
        f"rank {VICTIM} (pid" in line and line.rstrip().endswith("; fencing it")
        for line in authority_text.splitlines()
    ), authority_text[-20000:]
    # The workers fenced it (through the MPI runtime): no fence command is configured.
    assert "survived its fence" not in authority_text, authority_text[-20000:]


def _get(url: str) -> tuple[int, dict]:
    try:
        with urllib.request.urlopen(url, timeout=10) as response:
            return response.status, json.loads(response.read() or b"{}")
    except urllib.error.HTTPError as error:
        return error.code, json.loads(error.read() or b"{}")


def _complete(base_url: str, model: Path, prompt: str) -> str:
    request = urllib.request.Request(
        f"{base_url}/v1/completions",
        data=json.dumps(
            dict(model=str(model), prompt=prompt, max_tokens=SHORT_MAX_TOKENS, temperature=0)
        ).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=300) as response:
        return json.loads(response.read())["choices"][0]["text"]


def _assert_completions(base_url: str, model: Path) -> None:
    for prompt, expected in PROMPTS.values():
        text = _complete(base_url, model, prompt)
        assert expected in text, (prompt, text)


@pytest.mark.skip_less_device(WORLD_SIZE)
@pytest.mark.skip_device_not_contain(["GB200"])
def test_authority_lost_while_serving(tmp_path: Path):
    """trtllm-serve keeps serving after its authority dies; /health answers 200 ``unprotected``."""
    model = _model()
    env = _fault_tolerance_env(tmp_path, None)
    config_path = tmp_path / "serve.yaml"
    config_path.write_text(yaml.safe_dump(llm_kwargs()))
    port = _free_port()
    base_url = f"http://127.0.0.1:{port}"
    mpirun_log = tmp_path / "mpirun.log"

    with (
        open(tmp_path / "authority.log", "w") as authority_out,
        open(mpirun_log, "w") as mpirun_out,
    ):
        authority = _start_authority(env, authority_out)
        mpirun = None
        try:
            mpirun = subprocess.Popen(
                [
                    *MPIRUN,
                    "trtllm-serve",
                    str(model),
                    "--host",
                    "127.0.0.1",
                    "--port",
                    str(port),
                    "--config",
                    str(config_path),
                ],
                env=env,
                stdout=mpirun_out,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            deadline = time.monotonic() + SERVE_TIMEOUT_S
            while True:
                assert mpirun.poll() is None, mpirun_log.read_text(errors="replace")[-20000:]
                assert time.monotonic() < deadline, "trtllm-serve did not become healthy"
                try:
                    if _get(f"{base_url}/health") == (200, {"state": "healthy"}):
                        break
                except (urllib.error.URLError, ConnectionError, TimeoutError):
                    pass
                time.sleep(5)
            _assert_completions(base_url, model)

            authority.kill()
            authority.wait()
            deadline = time.monotonic() + HEALTH_SETTLE_S
            while (health := _get(f"{base_url}/health"))[1].get("state") != "unprotected":
                assert time.monotonic() < deadline, health
                time.sleep(1)
            assert health == (200, {"state": "unprotected"}), health
            assert _get(f"{base_url}/fault_tolerance") == (200, {"state": "unprotected"})
            _assert_completions(base_url, model)
        finally:
            if mpirun is not None and mpirun.poll() is None:
                os.killpg(mpirun.pid, signal.SIGTERM)
                try:
                    mpirun.wait(timeout=TEARDOWN_TIMEOUT_S)
                except subprocess.TimeoutExpired:
                    os.killpg(mpirun.pid, signal.SIGKILL)
                    mpirun.wait()
            if authority.poll() is None:
                _stop_authority(authority)
