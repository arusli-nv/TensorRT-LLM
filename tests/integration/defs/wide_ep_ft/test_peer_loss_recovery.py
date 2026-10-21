# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""WideEP peer-loss recovery end to end: EP4 survives the loss of workers, and the loss of its authority."""

import importlib.metadata
import json
import os
import re
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
from .peer_loss_driver import PROMPTS, RUN_TOKEN_ENV, SHORT_MAX_TOKENS, llm_kwargs

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
# The authority imports TensorRT-LLM before it listens; another process may take its port first.
AUTHORITY_START_TIMEOUT_S = 120
AUTHORITY_START_ATTEMPTS = 3
# Set to 1 where the image must provide ULFM: its absence then fails these tests instead of skipping them.
REQUIRE_ULFM_ENV = "TLLM_REQUIRE_ULFM"
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
        if os.environ.get(REQUIRE_ULFM_ENV) == "1":
            pytest.fail(f"{reason} ({REQUIRE_ULFM_ENV}=1)")
        pytest.skip(reason)
    model = Path(llm_models_root()) / "DeepSeek-V3-Lite" / "nvfp4_moe_only"
    if not model.exists():
        pytest.skip(f"{model} does not exist")
    return model


def _free_port() -> int:
    """A localhost port free for TCP and UDP (the authority takes both) when this returns."""
    while True:
        with socket.socket() as tcp, socket.socket(type=socket.SOCK_DGRAM) as udp:
            tcp.bind(("127.0.0.1", 0))
            port = tcp.getsockname()[1]
            try:
                udp.bind(("127.0.0.1", port))
            except OSError:
                continue
            return port


def _listens(pid: int, port: int) -> bool:
    """Whether process ``pid`` holds a TCP socket listening on localhost ``port``."""
    sockets = set()
    for line in Path("/proc/net/tcp").read_text().splitlines()[1:]:
        fields = line.split()
        # Columns: index, local address:port (hex), remote, state (0A is LISTEN), ..., inode.
        if fields[3] == "0A" and int(fields[1].split(":")[1], 16) == port:
            sockets.add(f"socket:[{fields[9]}]")
    try:
        fds = list(Path(f"/proc/{pid}/fd").iterdir())
    except OSError:
        return False
    for fd in fds:
        try:
            if os.readlink(fd) in sockets:
                return True
        except OSError:
            continue
    return False


def _fault_tolerance_env(tmp_path: Path, ft_config: dict | None, **extra: str) -> dict:
    """The environment of the authority and every rank; ``ft_config`` becomes the FT config file.

    ``_start_authority`` adds the authority's endpoint.
    """
    # Ranks come from mpirun, not from an enclosing Slurm or PMIx launch.
    launcher_prefixes = ("SLURM_", "PMIX_", "PMI_", "OMPI_MCA_ess", "PRTE_MCA_ras")
    env = dict(
        {k: v for k, v in os.environ.items() if not k.startswith(launcher_prefixes)},
        TLLM_FAULT_TOLERANCE_MODE="1",
        TLLM_FAULT_TOLERANCE_AUTHORITY_KEY=secrets.token_hex(32),
        # The driver faults only worker processes with this run's token.
        **{RUN_TOKEN_ENV: secrets.token_hex(8)},
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
    """Start the authority on a free port and set its endpoint in ``env`` once it listens there."""
    for _ in range(AUTHORITY_START_ATTEMPTS):
        port = _free_port()
        endpoint = f"tcp://127.0.0.1:{port}"
        authority = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "tensorrt_llm.executor.recovery.authority",
                "--endpoint",
                endpoint,
                "--world-size",
                str(WORLD_SIZE),
            ],
            env=dict(env, CUDA_VISIBLE_DEVICES=""),
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        deadline = time.monotonic() + AUTHORITY_START_TIMEOUT_S
        while authority.poll() is None and time.monotonic() < deadline:
            if _listens(authority.pid, port):
                env["TLLM_FAULT_TOLERANCE_AUTHORITY"] = endpoint
                return authority
            time.sleep(0.5)
        # Another process bound the port first, or the authority did not start.
        _stop_authority(authority)
    pytest.fail(f"The recovery authority did not listen after {AUTHORITY_START_ATTEMPTS} attempts")


def _stop_authority(authority: subprocess.Popen) -> None:
    authority.send_signal(signal.SIGTERM)
    try:
        authority.wait(timeout=30)
    except subprocess.TimeoutExpired:
        authority.kill()


def _wait_for_teardown(mpirun: subprocess.Popen, result_path: Path) -> int:
    """Wait until mpirun exits and return its exit status."""
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
    return mpirun.returncode


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
            returncode = _wait_for_teardown(mpirun, result_path)
        finally:
            _stop_authority(authority)

    output = mpirun_log.read_text(errors="replace")
    assert result_path.exists(), f"The driver wrote no result:\n{output[-20000:]}"
    result = json.loads(result_path.read_text())
    assert "after_loss" in result, f"The driver did not finish serving:\n{output[-20000:]}"
    # The survivors of a recovered world finalize MPI on shutdown and exit 0.
    assert returncode == 0, f"mpirun exited with {returncode}:\n{output[-20000:]}"
    return result, output, authority_log.read_text()


def _assert_answered(rows: list[dict]) -> None:
    for row in rows:
        assert "error" not in row, row
        assert row["expected"] in row["text"], row


def _assert_coherent(rows: list[dict]) -> None:
    """Each answer reads as text: mostly printable and not one token repeated over and over."""
    for row in rows:
        text, token_ids = row["text"], row["token_ids"]
        assert sum(c.isprintable() or c.isspace() for c in text) >= 0.9 * len(text), row
        assert len(set(token_ids)) >= len(token_ids) // 2, row


def _rerun_counts(output: str) -> tuple[int, int]:
    """The requests rank 0 ran again and failed, summed over its recoveries."""
    counts = [
        tuple(map(int, match.groups()))
        for line in output.splitlines()
        if "Rank 0: resumed serving without ranks" in line
        and (match := re.search(r"re-ran (\d+) requests and failed (\d+);", line))
    ]
    assert counts, "Rank 0 logged no resumed recovery"
    return sum(c[0] for c in counts), sum(c[1] for c in counts)


def _assert_recovered(
    result: dict,
    output: str,
    authority_text: str,
    mode: str,
    lost: list[int],
    per_rank: int = 1,
    online_eplb: bool = False,
) -> str:
    """Every request answered, the survivors resumed without ``lost`` and nothing was rebuilt.

    ``per_rank`` is the driver's in-flight requests per attention-DP rank; ``online_eplb`` whether
    online EPLB moves experts between rounds. Returns the mpirun log from the first fault on.
    """
    _assert_answered(result["healthy"])
    healthy_ids = {row["rank"]: row["token_ids"] for row in result["healthy"]}
    before_kill, kill_marker, after_kill = output.partition(KILL_MARKER)
    assert kill_marker, "The driver did not fault a victim"
    if mode == "inflight":
        # The lost ranks' requests streamed nothing, so the survivors ran them again from their prompts.
        _assert_answered(result["during_loss"])
        if per_rank == 1 and not online_eplb:
            # A survivor's request decoded its first tokens before the loss, alone on its rank as in
            # the healthy round and on the same expert placement: the same greedy tokens. Online
            # EPLB may move experts between the rounds, which changes the reduction order.
            for row in result["during_loss"]:
                if row["rank"] not in lost:
                    assert row["token_ids"][:SHORT_MAX_TOKENS] == healthy_ids[row["rank"]], row
        assert _rerun_counts(after_kill) == (len(lost) * per_rank, 0), after_kill[-20000:]
    _assert_answered(result["after_loss"])
    # Exact tokens are not asserted after the loss: the lost experts are served from other GPUs,
    # which changes the reduction order and so may change greedy tokens. Each answer still holds its
    # expected word and reads as text.
    _assert_coherent(result["after_loss"])
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
def test_peer_loss_recovery_under_load(tmp_path: Path):
    """Rank 2 dies while 32 requests decode, a full batch of 8 on every attention-DP rank."""
    per_rank = 8
    result, output, authority_text = _run_driver(
        tmp_path, "inflight", [VICTIM], "--per-rank", str(per_rank)
    )
    _assert_recovered(result, output, authority_text, "inflight", [VICTIM], per_rank)


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
@pytest.mark.parametrize("eplb", ["static", "online"])
def test_restore_into_ghost_slots(eplb: str, tmp_path: Path):
    """The appended slots hold no memory until rank 2 dies in decode; the restore maps memory behind them.

    Three ranks cannot hold the 72 experts in 18 slots each (a static placement of one slot per expert) or in 20
    (online EPLB over 80 slots): each rank gets 6 or 4 ghost slots after its EPLB slots, which the restore fills
    once any replicas are used up. Online EPLB's updates then serve on.
    """
    if eplb == "online":
        args, appended = ["--online-slots", "80"], "4 ghost slots per rank to EPLB's 20"
    else:
        args, appended = ["--replica-offset", "0"], "6 ghost slots per rank to EPLB's 18"
    result, output, authority_text = _run_driver(
        tmp_path, "inflight", [VICTIM], *args, ft_config=dict(extra_slots="ghost")
    )
    before_kill = output.partition(KILL_MARKER)[0]
    assert f"WideEP peer-loss recovery appends {appended}" in before_kill, output[-20000:]
    after_kill = _assert_recovered(
        result, output, authority_text, "inflight", [VICTIM], online_eplb=eplb == "online"
    )
    assert "behind ghost expert slots" in after_kill, output[-20000:]


@pytest.mark.skip_less_device(WORLD_SIZE)
@pytest.mark.skip_device_not_contain(["GB200"])
def test_restore_into_appended_empty_slots(tmp_path: Path):
    """Online EPLB's empty extra slots are held from startup and filled by a restore.

    They are the default where GDRCopy cannot map ghost slot memory. Each rank gets 4 after its 20 EPLB slots,
    holding memory from startup, and the restore after rank 2 dies in decode loads them without mapping any.
    """
    result, output, authority_text = _run_driver(
        tmp_path, "inflight", [VICTIM], "--online-slots", "80", ft_config=dict(extra_slots="empty")
    )
    before_kill = output.partition(KILL_MARKER)[0]
    assert "WideEP peer-loss recovery appends 4 empty slots per rank to EPLB's 20" in before_kill, (
        output[-20000:]
    )
    after_kill = _assert_recovered(
        result, output, authority_text, "inflight", [VICTIM], online_eplb=True
    )
    assert re.search(rf"Restored [1-9]\d* expert slots without ranks \[{VICTIM}\]", after_kill), (
        after_kill[-20000:]
    )
    assert "behind ghost expert slots" not in after_kill, after_kill[-20000:]


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


def _assert_completions(base_url: str, model: Path) -> list[str]:
    texts = [_complete(base_url, model, prompt) for prompt, _ in PROMPTS.values()]
    for (prompt, expected), text in zip(PROMPTS.values(), texts):
        assert expected in text, (prompt, text)
    return texts


@pytest.mark.skip_less_device(WORLD_SIZE)
@pytest.mark.skip_device_not_contain(["GB200"])
def test_authority_lost_while_serving(tmp_path: Path):
    """trtllm-serve keeps serving after its authority dies; /health answers 200 ``unprotected``."""
    model = _model()
    env = _fault_tolerance_env(tmp_path, None)
    config_path = tmp_path / "serve.yaml"
    config_path.write_text(yaml.safe_dump(llm_kwargs()))
    # The server binds a port of the kernel's choice and reports it here.
    address_path = tmp_path / "serve_address"
    base_url = None
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
                    "0",
                    "--report_addr",
                    str(address_path),
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
                if base_url is None and address_path.exists():
                    base_url = f"http://{address_path.read_text().strip()}"
                if base_url is not None:
                    try:
                        health = _get(f"{base_url}/health")
                        if health[0] == 200 and health[1].get("state") == "healthy":
                            break
                    except (urllib.error.URLError, ConnectionError, TimeoutError):
                        pass
                time.sleep(5)
            # The authority's details ride on /health while it answers.
            assert health[1]["recoveries"] == 0 and health[1]["failed_ranks"] == [], health
            healthy_texts = _assert_completions(base_url, model)

            authority.kill()
            authority.wait()
            deadline = time.monotonic() + HEALTH_SETTLE_S
            while (health := _get(f"{base_url}/health"))[1].get("state") != "unprotected":
                assert time.monotonic() < deadline, health
                time.sleep(1)
            # No details without an answer: nothing stale is reported.
            assert health == (200, {"state": "unprotected"}), health
            # No rank was lost: the same greedy completions as before.
            assert _assert_completions(base_url, model) == healthy_texts
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
