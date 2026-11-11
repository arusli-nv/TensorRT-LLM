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
"""WideEP fault tolerance end to end: DeepSeek keeps serving under load when ranks fail; a replacement rejoins.

By default each test starts DeepSeek-V3-Lite with attention DP and EP over 4 GPUs (``trtllm-llmapi-launch`` under
``mpirun --enable-recovery``; :func:`test_rank_rejoins_when_the_server_spawns_its_ranks` starts plain ``trtllm-serve``,
which spawns the ranks itself). With ``TLLM_FT_E2E_SERVER`` set, a test attaches to a deployment started elsewhere
instead (for example DeepSeek-R1 over 8 nodes from a Slurm script), one test per deployment, and runs on the node of
the ranks it fails:

- ``TLLM_FT_E2E_SERVER``: the server's base URL.
- ``TLLM_FT_E2E_SERVER_LOGS``: a glob pattern of the files the deployment's ranks log to.
- ``TLLM_FT_E2E_VICTIM``: the rank to fail, which runs on this node.
- ``TLLM_FT_E2E_SEQUENCE``: the failures :func:`test_rank_failure_sequence` runs, comma-separated (default
  ``crash,hang``; ``crash,node`` on several nodes).
- ``TLLM_FT_E2E_WORLD_SIZE``: the number of ranks of a deployment without fault tolerance (which has no status
  endpoint to read it from).
- ``TLLM_FT_E2E_STALL_DIR``: a directory every node of the deployment watches. On a stall the test writes a stall
  request there (``stall-<n>.request``, JSON: the stall number, when the tokens stopped and when it was requested, and
  whether gdb stacks are wanted) instead of dumping stacks itself; each node's watcher, run by whatever launched the
  deployment, dumps the stacks of that node's rank workers.
- The deployment's join secret (``TLLM_FAULT_TOLERANCE_JOIN_SECRET``) identifies its processes.

In both modes the measurements are written as JSON to ``TLLM_FT_E2E_REPORT`` (default: the test's tmp_path).
``TLLM_FT_E2E_CONFIG`` (JSON) is merged into the configuration of a deployment a test starts. When the streamed tokens
stop for STALL_DUMP_S, the stacks of the node's rank workers are dumped with ``py-spy`` (where installed) next to the
report, or requested from every node through ``TLLM_FT_E2E_STALL_DIR``.

A test runs its failures in sequence in one deployment, under one closed-loop load that runs throughout:

- Load clients send requests of a fixed length, streamed and not, greedy and sampled (with a seed); a client sends a
  request again once when it fails with the error code ``rank_failure``.
- Eval clients answer GSM8K questions (main's lm-eval task, through the same server); each phase is scored.
- Every failure is a cycle of phases: ``failover`` (the requests in flight at the failure), ``degraded`` (N-K ranks
  serve while the replacement starts on its own), ``admission`` (the requests in flight when the replacement's
  admission begins) and ``rejoined`` (back at N ranks); the run starts with a ``healthy`` phase.
- Membership events are timed from the membership store, read with the server's own reader every STORE_POLL_S.
- Per phase: TTFT and TPOT (streamed requests), output and goodput tokens per second, GSM8K accuracy. Per cycle:
  pauses, times to the smaller membership and back to N, the status, the ranks' incarnations, the survivors' process
  ids, the replacements' GPUs and every rank's CUDA graph captures. Per run: every rank's CUDA graph captures after
  warmup and at the end. Also a per-second token timeline with the events, the membership transactions, GPU memory
  and /dev/shm segments.

Greedy probe requests, sent one at a time before and after the load, are a smoke signal (reported, not checked).
"""

import contextlib
import copy
import glob
import http.client
import inspect
import json
import math
import os
import random
import re
import shutil
import signal
import socket
import statistics
import subprocess
import threading
import time
import traceback
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union

import psutil
import pytest
import yaml

from tensorrt_llm._torch.distributed.membership import (
    JOIN_SECRET_ENV,
    RENDEZVOUS_ENV,
    Authority,
    open_liveness_connection,
    open_store,
)
from tensorrt_llm._torch.pyexecutor.rank_failure_recovery import FaultToleranceHealth
from tensorrt_llm._utils import FAULT_TOLERANCE_RANK_ENV
from tensorrt_llm.executor.utils import RANK_FAILURE_CODE
from tensorrt_llm.llmapi.mpi_session import split_mpi_env

from .conftest import llm_models_root

WORLD_SIZE = 4
VICTIM = 2
# Model loading and CUDA graph capture.
STARTUP_TIMEOUT_S = 1800
# The deployment's status endpoint: generation, live and lost ranks, the last membership transaction.
STATUS_PATH = "/fault_tolerance_status"
# The endpoint refreshes from the membership store once a second; membership events are timed, and acted on during a
# membership change, from the store itself, read this often.
STORE_POLL_S = 0.01
EXTERNAL = bool(os.environ.get("TLLM_FT_E2E_SERVER"))
STALL_DIR = os.environ.get("TLLM_FT_E2E_STALL_DIR")
# Both arms on the fence path of the NVLink one-sided all-to-all (fault tolerance needs it).
FENCE_PATH_ENV = {"TRTLLM_NVLINK_ONE_SIDED_A2A_FORCE_CFT": "0"}

# Closed-loop load: per serving rank, this many clients (at most MAX_CLIENTS), each sending its next request when the
# previous one ends; by client index, half stream and, of each half, half sample (SAMPLING, with a per-request seed).
# Requests last several seconds, so many are in flight at a failure.
CLIENTS_PER_RANK = 16
MAX_CLIENTS = 256
LOAD_INPUT_LEN = (128, 512)
LOAD_MAX_TOKENS = 512
SAMPLING = {"temperature": 0.8, "top_p": 0.95}
REQUEST_TIMEOUT_S = 900
# Greedy probe requests, sent one at a time without other load, compared with the healthy reference.
NUM_PROBES = 8
PROBE_MAX_TOKENS = 64

# GSM8K through the same server: main's lm-eval task (5-shot, its stop strings, greedy, 256 tokens at most) on the
# first EVAL_QUESTIONS questions of its seed-0 shuffle. Every phase asks the same questions: the eval clients start
# again from the first question when a window phase starts, and the window lasts until every question was answered in
# it (at most EVAL_COVER_EXTRA_S past WINDOW_S). Per serving rank this many eval clients (every rank serves some); at
# DEP4 a question takes about 3 s under the load, so the 16 clients answer the subset in about 50 s.
EVAL_CLIENTS_PER_RANK = 4
MAX_EVAL_CLIENTS = 64
EVAL_QUESTIONS = 256
EVAL_COVER_EXTRA_S = 120
# Accuracy pass bar, paired: a phase scores each question once (its first answer in the phase) with the task's
# strict-match filter. The healthy phase answers every question; every other phase answers at least one of them. On
# the questions both the phase and the healthy phase answered, the discordant pairs (right when healthy and wrong in
# the phase, or the reverse) must not show a loss: a one-sided exact sign test (McNemar) at ACCURACY_ALPHA divided by
# the number of phases past the healthy one (Bonferroni).
ACCURACY_FILTER = "strict-match"
ACCURACY_ALPHA = 0.05

# Load phases: warmup before the healthy window, settle time after a membership change, and each throughput window
# (measured in SUBWINDOWS parts, whose spread is the healthy run's noise).
WARMUP_S = 20
SETTLE_S = 10
# A stall: no streamed token for this long while the load runs. Each stall dumps the workers' stacks once, for at
# most MAX_STALL_DUMPS stalls of a run.
STALL_DUMP_S = 5.0
MAX_STALL_DUMPS = 4
WINDOW_S = 60
SUBWINDOWS = 3

# Targets: failover pause after a crash, after a hang (beyond the authority's progress timeout), and the survivors'
# pause while a replacement is admitted; throughput after the rejoin within the healthy noise, at least this tolerance.
CRASH_PAUSE_TARGET_S = 5.0
HANG_PAUSE_MARGIN_S = 5.0
T_HUNG_S = inspect.signature(Authority).parameters["progress_timeout_s"].default
ADMISSION_PAUSE_TARGET_S = 3.0
THROUGHPUT_TOLERANCE = 0.05
# Time to detect a crash and commit the smaller membership; a hang is detected T_HUNG_S later.
RECOVERY_TIMEOUT_S = 120
# A replacement loads the model and captures its CUDA graphs alone before it is admitted.
REJOIN_TIMEOUT_S = 1200
# An excluded process exits once it runs again and reads its exclusion.
EXCLUDED_EXIT_TIMEOUT_S = 60
# How long an invalid join is watched for a membership change it must not cause.
INVALID_JOIN_WATCH_S = 20
# Repeated reuse of a rank: GPU memory a second cycle may add on a GPU over the first, at the same point of the cycle,
# before it counts as a leak.
GPU_MEMORY_LEAK_TOLERANCE_MIB = 256
# The authority logs why it refuses a liveness connection: this prefix, then the reason.
REFUSAL_LOG = "Membership: closing a liveness connection: "
# Rank 0 logs, per membership change, how many requests that no live rank holds the rank failure ended.
RETRYABLE_LOG = re.compile(
    r"Rank 0 resumed serving with membership generation (\d+),.* (\d+) requests no live rank holds ended by "
    r"the rank failure"
)

# Each rank logs these per graph shape in the warmup's capture pass, the only place its CUDA graph runner may
# capture (CUDAGraphRunner.allow_capture). A rank's count grows only when it captures CUDA graphs: a replacement
# adds its own at startup; a survivor that adds one recaptured its graphs.
CAPTURE_LINE = re.compile(r"\[RANK (\d+)\].*(CUDA graph capture|Capturing \d+ graphs)")
# The variables a launcher sets to a process's rank, in the order trtllm-llmapi-launch reads them; without fault
# tolerance, the ranks a server spawns have PMIX_RANK.
RANK_ENV = ("SLURM_PROCID", "OMPI_COMM_WORLD_RANK", "PMI_RANK", "PMIX_RANK")
GPU_DEVICE = re.compile(r"/dev/nvidia\d+")
HEALTH_STATES = {"healthy", "recovering", "degraded"}


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("", 0))
        return sock.getsockname()[1]


def _error_body(error: urllib.error.HTTPError) -> bytes:
    """The body of an HTTP error response; empty if it breaks off."""
    try:
        return error.read()
    except (OSError, http.client.HTTPException):
        return b""


def _error_code(body: bytes) -> Optional[Union[int, str]]:
    """The ``code`` of an OpenAI error response body; None if the body is not one."""
    try:
        return json.loads(body).get("code")
    except (ValueError, AttributeError):
        return None


def _get_json(url: str, timeout_s: float = 5.0) -> Tuple[Optional[int], Optional[dict]]:
    """(HTTP status, JSON body); (None, None) if the server does not answer or its response breaks off."""
    try:
        with urllib.request.urlopen(url, timeout=timeout_s) as response:
            status, body = response.status, response.read()
    except urllib.error.HTTPError as error:
        status, body = error.code, _error_body(error)
    except (OSError, http.client.HTTPException):
        return None, None
    try:
        return status, json.loads(body)
    except ValueError:
        return status, None


def _post(base: str, path: str, body: dict):
    request = urllib.request.Request(
        f"{base}{path}", json.dumps(body).encode(), {"Content-Type": "application/json"}
    )
    return urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT_S)


def _status(base: str) -> Optional[dict]:
    """The deployment's status; None if the endpoint does not answer."""
    code, body = _get_json(f"{base}{STATUS_PATH}")
    return body if code == 200 else None


@dataclass
class Deployment:
    base: str
    model: str
    logs: str  # glob pattern of the files the ranks log to
    secret: Optional[str]  # None without fault tolerance
    world_size: int
    victim: int

    @property
    def fault_tolerance(self) -> bool:
        return self.secret is not None


def _serving_config(model: Path, fault_tolerance: bool, serving_options: bool = False) -> dict:
    """The deployment's configuration.

    With ``serving_options``, also the options large deployments serve with: attention-DP balancing, CUDA graphs up
    to batch 512, an fp8 KV cache with a host tier, and online EPLB with a redundant slot per rank.
    """
    num_experts = json.loads((model / "config.json").read_text())["n_routed_experts"]
    config = {
        "enable_wide_ep_fault_tolerance": fault_tolerance,
        "enable_attention_dp": True,
        "disable_overlap_scheduler": False,
        "cuda_graph_config": {"enable_padding": True, "max_batch_size": 32},
        "kv_cache_config": {"free_gpu_memory_fraction": 0.5, "use_kv_cache_manager_v2": True},
        "speculative_config": {"decoding_type": "MTP", "num_nextn_predict_layers": 1},
        # One slot per expert (static EPLB): fault tolerance adds the spare slots the survivors of a failure need to
        # hold every expert, and restores a lost rank's experts into them from host copies.
        "moe_config": {"load_balancer": {"num_slots": num_experts}},
    }
    if serving_options:
        config.update(
            {
                "attention_dp_config": {
                    "enable_balance": True,
                    "batching_wait_iters": 0,
                    "timeout_iters": 60,
                },
                "cuda_graph_config": {"enable_padding": True, "max_batch_size": 512},
                "kv_cache_config": {
                    **config["kv_cache_config"],
                    "dtype": "fp8",
                    "host_cache_size": 16 << 30,
                },
                "moe_config": {
                    "load_balancer": {
                        "num_slots": num_experts + WORLD_SIZE,
                        "layer_updates_per_iter": 1,
                    }
                },
            }
        )
    return _merged(config, json.loads(os.environ.get("TLLM_FT_E2E_CONFIG", "{}")))


def _merged(config: dict, overrides: dict) -> dict:
    """``config`` with ``overrides`` merged in, nested mappings key by key."""
    merged = dict(config)
    for name, value in overrides.items():
        both_dicts = isinstance(value, dict) and isinstance(merged.get(name), dict)
        merged[name] = _merged(merged[name], value) if both_dicts else value
    return merged


def _wait_until_healthy(base: str, alive: Callable[[], bool], fault_tolerance: bool) -> None:
    deadline = time.monotonic() + STARTUP_TIMEOUT_S
    while True:
        code, _ = _get_json(f"{base}/health")
        # /health carries no state; with fault tolerance the status endpoint does.
        if code == 200 and (not fault_tolerance or (_status(base) or {}).get("state") == "healthy"):
            return
        assert alive(), "the server exited during startup"
        assert time.monotonic() < deadline, "the server did not become healthy"
        time.sleep(5)


def _chosen_secret(since: float) -> str:
    """The join secret a server that spawns its ranks chose, from the environment of its node supervisor."""
    for process in psutil.process_iter(["cmdline", "create_time"]):
        with contextlib.suppress(psutil.AccessDenied, psutil.NoSuchProcess, psutil.ZombieProcess):
            cmdline = process.info["cmdline"] or []
            if process.info["create_time"] >= since and any(
                arg.endswith("_llmapi_node_supervisor.py") for arg in cmdline
            ):
                return process.environ()[JOIN_SECRET_ENV]
    raise LookupError("no node supervisor of the deployment runs on this node")


@contextlib.contextmanager
def _deploy(
    tmp_path: Path,
    fault_tolerance: bool,
    serving_options: bool = False,
    server_spawns_ranks: bool = False,
):
    """A deployment (:func:`_serving_config`) for the test, or the one at ``TLLM_FT_E2E_SERVER``.

    ``server_spawns_ranks``: plain ``trtllm-serve``, which spawns its ranks itself and chooses the join secret and the
    rendezvous address (none is exported); otherwise ``trtllm-llmapi-launch`` under ``mpirun``. A deployment this
    starts leaves no /dev/shm segment once it ended (whoever starts the external one checks that).
    """
    if EXTERNAL:
        base = os.environ["TLLM_FT_E2E_SERVER"]
        _wait_until_healthy(base, lambda: True, fault_tolerance)
        if fault_tolerance:
            status = _status(base)
            assert status is not None, f"{base}{STATUS_PATH} does not answer"
            world_size = len(status["live_ranks"])
        else:
            world_size = int(os.environ["TLLM_FT_E2E_WORLD_SIZE"])
        yield Deployment(
            base,
            _get_json(f"{base}/v1/models")[1]["data"][0]["id"],
            os.environ["TLLM_FT_E2E_SERVER_LOGS"],
            os.environ[JOIN_SECRET_ENV] if fault_tolerance else None,
            world_size,
            int(os.environ.get("TLLM_FT_E2E_VICTIM", VICTIM)),
        )
        return

    model = Path(llm_models_root()) / "DeepSeek-V3-Lite" / "bf16"
    if not model.exists():
        pytest.skip(f"{model} does not exist")
    config = tmp_path / "config.yaml"
    config.write_text(yaml.safe_dump(_serving_config(model, fault_tolerance, serving_options)))
    shm_before = set(os.listdir("/dev/shm"))
    port = _free_port()
    base = f"http://localhost:{port}"
    log = tmp_path / "server.log"
    serve = [
        "trtllm-serve",
        str(model),
        f"--tp_size={WORLD_SIZE}",
        f"--ep_size={WORLD_SIZE}",
        f"--port={port}",
        f"--config={config}",
    ]
    # Info level on every rank (only the leader takes the server's level): each rank logs its CUDA graph captures.
    env = {**os.environ, **FENCE_PATH_ENV, "TLLM_LOG_LEVEL": "info"}
    env.pop(JOIN_SECRET_ENV, None)
    env.pop(RENDEZVOUS_ENV, None)
    secret = None
    if server_spawns_ranks:
        # One process of no MPI or Slurm job, as a user starts it.
        env = {**split_mpi_env()[0], **FENCE_PATH_ENV, "TLLM_LOG_LEVEL": "info"}
        if "PMIX_HOSTNAME" in os.environ:
            env["PMIX_HOSTNAME"] = os.environ["PMIX_HOSTNAME"]
        command = serve
    else:
        # Under trtllm-llmapi-launch the server does not start the workers: they take the join secret and the
        # rendezvous address (rank 0's host, where the membership store listens) from the launch.
        if fault_tolerance:
            secret = os.urandom(32).hex()
            env[JOIN_SECRET_ENV] = secret
            env[RENDEZVOUS_ENV] = f"{socket.gethostname()}:{_free_port()}"
        # --enable-recovery keeps the other ranks running when one exits. With fault tolerance the launcher runs once
        # on this node: its node supervisor starts the node's ranks.
        command = [
            "mpirun",
            "--enable-recovery",
            "--allow-run-as-root",
            "-np",
            str(1 if fault_tolerance else WORLD_SIZE),
            "trtllm-llmapi-launch",
            *serve,
        ]
    launched_s = time.time()
    with open(log, "w") as log_file:
        launcher = subprocess.Popen(
            command, env=env, stdout=log_file, stderr=subprocess.STDOUT, start_new_session=True
        )
    try:
        _wait_until_healthy(base, lambda: launcher.poll() is None, fault_tolerance)
        if server_spawns_ranks and fault_tolerance:
            secret = _chosen_secret(launched_s)
        yield Deployment(base, str(model), str(log), secret, WORLD_SIZE, VICTIM)
    finally:
        if secret is not None:
            for process in _rank_workers(secret).values():
                with contextlib.suppress(psutil.NoSuchProcess):
                    process.resume()  # a stopped process does not take the interrupt
        os.killpg(launcher.pid, signal.SIGINT)
        try:
            launcher.wait(120)
        except subprocess.TimeoutExpired:
            os.killpg(launcher.pid, signal.SIGKILL)
            launcher.wait()
    if secret is not None:  # replacements run outside the launcher's process group
        psutil.wait_procs(_node_processes(secret), timeout=60)
    left = sorted(set(os.listdir("/dev/shm")) - shm_before)
    assert not left, f"/dev/shm segments the deployment left after it ended: {left}"


@pytest.fixture
def deployment(tmp_path: Path):
    with _deploy(tmp_path, fault_tolerance=True) as deployment:
        yield deployment


@pytest.fixture
def serving_options_deployment(tmp_path: Path):
    """The deployment with the options large deployments serve with (:func:`_serving_config`)."""
    with _deploy(tmp_path, fault_tolerance=True, serving_options=True) as deployment:
        yield deployment


@pytest.fixture
def spawned_deployment(tmp_path: Path):
    """The deployment as plain ``trtllm-serve`` starts it: the server spawns the ranks and chooses the secrets."""
    with _deploy(tmp_path, fault_tolerance=True, server_spawns_ranks=True) as deployment:
        yield deployment


@pytest.fixture
def baseline_deployment(tmp_path: Path):
    """The same deployment without fault tolerance (main), on the same fence path."""
    with _deploy(tmp_path, fault_tolerance=False) as deployment:
        yield deployment


def _holds_gpu(pid: int) -> bool:
    try:
        return any(
            GPU_DEVICE.fullmatch(os.readlink(f"/proc/{pid}/fd/{fd}"))
            for fd in os.listdir(f"/proc/{pid}/fd")
        )
    except OSError:
        return False


def _process_rank(env: Dict[str, str]) -> Optional[int]:
    """The rank a process runs.

    A rank the node supervisor starts runs outside the launcher's job: its rank comes from its identity.
    """
    if FAULT_TOLERANCE_RANK_ENV in env:
        return int(env[FAULT_TOLERANCE_RANK_ENV].split(",")[0])
    rank = next((env[name] for name in RANK_ENV if name in env), None)
    return None if rank is None else int(rank)


def _rank_workers(secret: str) -> Dict[int, psutil.Process]:
    """The worker process of each rank on this node of the deployment with join secret ``secret``.

    That is the outermost process of the rank that holds a GPU: a launcher or node supervisor holds
    none, and a worker's helper processes have it as parent.
    """
    candidates = {}
    for process in psutil.process_iter():
        if process.pid == os.getpid():
            continue  # under a launcher, this test carries a rank and the join secret too
        try:
            env = process.environ()
        except (psutil.AccessDenied, psutil.NoSuchProcess, psutil.ZombieProcess):
            continue
        rank = _process_rank(env)
        if env.get(JOIN_SECRET_ENV) == secret and rank is not None and _holds_gpu(process.pid):
            candidates[process.pid] = (rank, process)
    workers = {}
    for pid, (rank, process) in candidates.items():
        try:
            parent = process.ppid()
        except psutil.NoSuchProcess:
            continue
        if candidates.get(parent, (None,))[0] == rank:
            continue
        if rank in workers:
            raise LookupError(
                f"rank {rank} has two worker processes: {workers[rank].pid} and {pid}"
            )
        workers[rank] = process
    return workers


def _node_processes(secret: str) -> List[psutil.Process]:
    """Every process on this node of the deployment with join secret ``secret``: what the loss of the node ends.

    That is every process carrying the secret in its environment (rank workers, node supervisors, launcher
    processes), except this test and the processes it runs under.
    """
    test = psutil.Process()
    own = {test.pid, *(p.pid for p in test.parents())}
    found = []
    for process in psutil.process_iter():
        if process.pid in own:
            continue
        with contextlib.suppress(psutil.AccessDenied, psutil.NoSuchProcess, psutil.ZombieProcess):
            if process.environ().get(JOIN_SECRET_ENV) == secret:
                found.append(process)
    return found


def _rendezvous(secret: str) -> str:
    """The address of the membership store, from the environment of a process of the deployment on this node."""
    for process in psutil.process_iter():
        with contextlib.suppress(psutil.AccessDenied, psutil.NoSuchProcess, psutil.ZombieProcess):
            env = process.environ()
            if env.get(JOIN_SECRET_ENV) == secret and RENDEZVOUS_ENV in env:
                return env[RENDEZVOUS_ENV]
    raise LookupError("no process of the deployment on this node has the rendezvous address")


def _capture_lines(logs: str) -> Dict[int, int]:
    """Per rank, the CUDA graph capture lines in the deployment's logs (a glob pattern)."""
    counts: Dict[int, int] = {}
    for log in sorted(glob.glob(logs)):
        with open(log, errors="replace") as lines:
            for line in lines:
                match = CAPTURE_LINE.search(line)
                if match:
                    rank = int(match.group(1))
                    counts[rank] = counts.get(rank, 0) + 1
    return counts


def _log_count(logs: str, text: str) -> int:
    """How many lines of the deployment's logs (a glob pattern) contain ``text``."""
    count = 0
    for log in sorted(glob.glob(logs)):
        with open(log, errors="replace") as lines:
            count += sum(text in line for line in lines)
    return count


def _retryable_failures_logged(logs: str) -> int:
    """How many requests rank 0 failed as retryable, over every membership change (its log lines)."""
    per_generation = {}
    for log in sorted(glob.glob(logs)):
        with open(log, errors="replace") as lines:
            for line in lines:
                match = RETRYABLE_LOG.search(line)
                if match:
                    per_generation[int(match.group(1))] = int(match.group(2))
    return sum(per_generation.values())


def _nvidia_smi(kind: str, fields: str) -> List[List[str]]:
    out = subprocess.run(
        ["nvidia-smi", f"--query-{kind}={fields}", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    return [[v.strip() for v in line.split(",")] for line in out.splitlines() if line.strip()]


def _rank_gpus(workers: Dict[int, psutil.Process]) -> Dict[int, List[int]]:
    """Per rank, the indices of the GPUs its worker process runs on (nvidia-smi's compute processes)."""
    index = {uuid: int(i) for i, uuid in _nvidia_smi("gpu", "index,uuid")}
    gpus: Dict[int, List[int]] = {}
    for pid, uuid in _nvidia_smi("compute-apps", "pid,gpu_uuid"):
        gpus.setdefault(int(pid), []).append(index[uuid])
    return {r: sorted(gpus.get(p.pid, [])) for r, p in sorted(workers.items())}


def _shm_orphans() -> Dict[str, int]:
    """The /dev/shm segments no process on this node maps or holds open, and their sizes."""
    used = set()
    for pid in filter(str.isdigit, os.listdir("/proc")):
        with contextlib.suppress(OSError):
            with open(f"/proc/{pid}/maps") as maps:
                for line in maps:
                    at = line.find("/dev/shm/")
                    if at >= 0:
                        used.add(line[at:].rstrip("\n").removesuffix(" (deleted)"))
            for fd in os.listdir(f"/proc/{pid}/fd"):
                with contextlib.suppress(OSError):
                    used.add(os.readlink(f"/proc/{pid}/fd/{fd}"))
    return {
        entry.name: entry.stat().st_size
        for entry in os.scandir("/dev/shm")
        if entry.is_file() and entry.path not in used
    }


def _resources(secret: Optional[str]) -> dict:
    """GPU memory per GPU and per rank worker on this node (nvidia-smi), and the /dev/shm segments."""
    by_pid = {int(pid): int(mib) for pid, mib in _nvidia_smi("compute-apps", "pid,used_memory")}
    workers = _rank_workers(secret) if secret is not None else {}
    return {
        "time": time.monotonic(),
        "gpu_memory_used_mib": {
            int(i): int(mib) for i, mib in _nvidia_smi("gpu", "index,memory.used")
        },
        "rank_memory_mib": {r: by_pid.get(p.pid) for r, p in sorted(workers.items())},
        # Per rank worker: (pid, open file descriptors), to see descriptors a surviving process leaks per cycle.
        "rank_open_fds": {
            r: (p.pid, len(os.listdir(f"/proc/{p.pid}/fd"))) for r, p in sorted(workers.items())
        },
        "shm_segments": sorted(os.listdir("/dev/shm")),
        "shm_bytes": sum(e.stat().st_size for e in os.scandir("/dev/shm") if e.is_file()),
        "shm_orphans": _shm_orphans(),
    }


def _prompt(seed: int, length: int) -> List[int]:
    rng = random.Random(seed)
    return [rng.randrange(1000, 30000) for _ in range(length)]


def _probes(deployment: Deployment) -> List[List[int]]:
    """The greedy outputs of the probe requests, sent one at a time.

    Without logprobs: the deployment decodes with MTP, which does not return them.
    """
    outputs = []
    for i in range(NUM_PROBES):
        body = {
            "model": deployment.model,
            "prompt": _prompt(-1 - i, 64),
            "max_tokens": PROBE_MAX_TOKENS,
            "ignore_eos": True,
            "temperature": 0.0,
            "detokenize": False,
        }
        with _post(deployment.base, "/v1/completions", body) as response:
            outputs.append(json.load(response)["choices"][0]["token_ids"])
    return outputs


class Gsm8k:
    """GSM8K questions and their scoring.

    Main's lm-eval task (``tensorrt_llm.evaluate.GSM8K``) on the offline dataset, as main's accuracy tests run it.
    """

    def __init__(self, num_questions: int):
        from tensorrt_llm.evaluate import GSM8K

        path = Path(llm_models_root()) / "datasets" / "openai" / "gsm8k"
        assert path.exists(), f"GSM8K is not available offline at {path}"
        self._task = GSM8K(dataset_path=str(path), random_seed=0).task_dict["gsm8k"]
        self._task.build_all_requests(limit=num_questions, rank=0, world_size=1)
        self.instances = self._task.instances

    def request(self, question: int) -> dict:
        """The completion request for a question: its few-shot context and the task's generation arguments."""
        context, gen_kwargs = self.instances[question].args
        return {
            "prompt": context,
            "max_tokens": gen_kwargs.get("max_gen_toks", 256),
            "temperature": gen_kwargs.get("temperature", 0.0),
            "stop": gen_kwargs.get("until", []),
        }

    def score(self, question: int, text: str) -> Dict[str, float]:
        """The exact match of one answer per task filter (strict and flexible answer extraction).

        Scored as ``_RunningScoreTracker.update`` in tensorrt_llm/evaluate/lm_eval.py does; main's GSM8K score is the
        mean over the filters.
        """
        probe = copy.copy(self.instances[question])
        probe.resps = [text]
        probe.filtered_resps = {}
        scores = {}
        for ensemble in self._task._filters:
            ensemble.apply([probe])
            metrics = self._task.process_results(probe.doc, [probe.filtered_resps[ensemble.name]])
            scores[ensemble.name] = float(metrics["exact_match"])
        return scores


@dataclass
class Request:
    stream: bool
    start: float
    question: Optional[int] = None  # an eval request's GSM8K question
    retry: bool = False  # sent again after a retryable failure
    end: float = 0.0
    # Streamed: (arrival time, token count) per chunk with tokens; otherwise one entry when the response arrives.
    arrivals: List[Tuple[float, int]] = field(default_factory=list)
    token_ids: List[int] = field(default_factory=list)
    text: str = ""
    completion_tokens: Optional[int] = None
    error: Optional[str] = None
    error_code: Optional[Union[int, str]] = None  # the OpenAI error's ``code``

    def spans(self, t: float) -> bool:
        return self.start < t < self.end

    @property
    def retryable(self) -> bool:
        return self.error_code == RANK_FAILURE_CODE


def _thread(target: Callable[..., None], args: tuple, failures: List[str]) -> threading.Thread:
    """A daemon thread that runs ``target``; an exception that ends it is appended to ``failures``."""

    def run() -> None:
        try:
            target(*args)
        except Exception:
            failures.append(f"{threading.current_thread().name} ended: {traceback.format_exc()}")
            raise

    return threading.Thread(target=run, daemon=True)


def _join(threads: List[threading.Thread], timeout_s: float, failures: List[str]) -> List[str]:
    """Join ``threads`` within ``timeout_s`` in all; returns ``failures`` and a failure per thread still running."""
    deadline = time.monotonic() + timeout_s
    for thread in threads:
        thread.join(max(0.0, deadline - time.monotonic()))
    return failures + [
        f"{thread.name} still runs {timeout_s} s after it was told to stop"
        for thread in threads
        if thread.is_alive()
    ]


class Clients:
    """Closed-loop load and eval clients (see the module docstring)."""

    def __init__(self, deployment: Deployment, gsm8k: Gsm8k):
        self._deployment = deployment
        self._gsm8k = gsm8k
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._seed = 0
        self._question = 0
        self.requests: List[Request] = []
        self._failures: List[str] = []
        world = deployment.world_size
        self._threads = [
            _thread(self._load_client, (i,), self._failures)
            for i in range(min(CLIENTS_PER_RANK * world, MAX_CLIENTS))
        ] + [
            _thread(self._eval_client, (), self._failures)
            for _ in range(min(EVAL_CLIENTS_PER_RANK * world, MAX_EVAL_CLIENTS))
        ]

    def start(self) -> None:
        for thread in self._threads:
            thread.start()

    def stop(self) -> List[str]:
        """Stop sending; returns once every request in flight ended, with the clients that failed.

        A client failed if an exception ended it, or if it still runs REQUEST_TIMEOUT_S later.
        """
        self._stop.set()
        return _join(self._threads, REQUEST_TIMEOUT_S, self._failures)

    def restart_questions(self) -> None:
        """The eval clients ask the subset again from its first question."""
        with self._lock:
            self._question = 0

    def answered(self, since: float) -> set:
        """The questions answered by requests that started at ``since`` or later."""
        with self._lock:
            requests = list(self.requests)
        return {
            r.question
            for r in requests
            if r.question is not None and r.error is None and r.start >= since
        }

    def _next(self, attribute: str) -> int:
        with self._lock:
            value = getattr(self, attribute)
            setattr(self, attribute, value + 1)
            return value

    def _load_client(self, index: int) -> None:
        stream, sampled = index % 2 == 0, index // 2 % 2 == 1
        while not self._stop.is_set():
            seed = self._next("_seed")
            body = {
                "prompt": _prompt(seed, random.Random(seed).randint(*LOAD_INPUT_LEN)),
                "max_tokens": LOAD_MAX_TOKENS,
                "ignore_eos": True,
                "detokenize": False,
                **({**SAMPLING, "seed": seed} if sampled else {"temperature": 0.0}),
            }
            self._send_with_retry(body, stream)

    def _eval_client(self) -> None:
        while not self._stop.is_set():
            question = self._next("_question") % len(self._gsm8k.instances)
            self._send_with_retry(self._gsm8k.request(question), False, question)

    def _send_with_retry(self, body: dict, stream: bool, question: Optional[int] = None) -> None:
        """Send a request; once more if it fails as retryable."""
        if self._send(body, stream, question).retryable:
            self._send(body, stream, question, retry=True)

    def _send(
        self, body: dict, stream: bool, question: Optional[int], retry: bool = False
    ) -> Request:
        request = Request(stream=stream, start=time.monotonic(), question=question, retry=retry)
        body = {"model": self._deployment.model, **body, "stream": stream}
        if stream:
            body["stream_options"] = {"include_usage": True}
        try:
            with _post(self._deployment.base, "/v1/completions", body) as response:
                if stream:
                    self._read_stream(response, request)
                else:
                    result = json.load(response)
                    choice = result["choices"][0]
                    request.token_ids = choice.get("token_ids") or []
                    request.text = choice.get("text") or ""
                    request.completion_tokens = result["usage"]["completion_tokens"]
                    request.arrivals.append((time.monotonic(), request.completion_tokens))
        except urllib.error.HTTPError as error:
            body = _error_body(error)
            request.error = f"HTTP {error.code}: {body.decode(errors='replace')}"
            request.error_code = _error_code(body)
        except (OSError, http.client.HTTPException, ValueError, LookupError, TypeError) as error:
            # No answer, a response that broke off (IncompleteRead, BadStatusLine), or one that is not a completion.
            request.error = repr(error)
        finally:
            # Also when anything else ends the client: the run fails on that (Clients.stop).
            request.end = time.monotonic()
            with self._lock:
                self.requests.append(request)
        return request

    @staticmethod
    def _read_stream(response, request: Request) -> None:
        for line in response:
            line = line.strip()
            if not line.startswith(b"data:"):
                continue
            data = line[len(b"data:") :].strip()
            if data == b"[DONE]":
                return
            chunk = json.loads(data)
            if "error" in chunk:
                request.error = chunk["error"]["message"]
                request.error_code = chunk["error"].get("code")
                continue
            now = time.monotonic()
            for choice in chunk.get("choices") or []:
                ids = choice.get("token_ids") or []
                if ids:
                    request.arrivals.append((now, len(ids)))
                    request.token_ids.extend(ids)
            if chunk.get("usage"):
                request.completion_tokens = chunk["usage"]["completion_tokens"]

    def load_requests(self) -> List[Request]:
        return [r for r in self.requests if r.question is None]

    def stream_arrivals(self) -> List[float]:
        return sorted(t for r in self.load_requests() if r.stream for t, _ in r.arrivals)

    def throughput(self, start: float, end: float) -> List[float]:
        """Load output tokens per second in each of SUBWINDOWS equal parts of [start, end)."""
        part = (end - start) / SUBWINDOWS
        tokens = [0] * SUBWINDOWS
        for request in self.load_requests():
            for t, n in request.arrivals:
                if start <= t < end:
                    tokens[min(int((t - start) / part), SUBWINDOWS - 1)] += n
        return [n / part for n in tokens]


def _largest_gap(times: List[float], start: float, end: float) -> Tuple[float, float]:
    """(length, start time) of the longest interval in [start, end] without an arrival.

    An interval that runs past ``end`` lasts until the next arrival.
    """
    inside = [start] + [t for t in times if start < t < end]
    later = [t for t in times if t >= end]
    inside.append(later[0] if later else end)
    return max((b - a, a) for a, b in zip(inside, inside[1:]))


def _percentiles(values: List[float]) -> Optional[dict]:
    if not values:
        return None
    ordered = sorted(values)

    def at(q: float) -> float:
        return round(ordered[min(len(ordered) - 1, int(q * len(ordered)))], 4)

    return {"p50": at(0.5), "p99": at(0.99), "n": len(ordered)}


class Monitor:
    """Watches the deployment: its ``/health`` four times a second and, with fault tolerance, its membership.

    ``health``: (time, HTTP status of ``/health``, state from ``/fault_tolerance_status``) per sample.
    ``membership``: (time, status) per change of the status (``FaultToleranceHealth.status``, with the record's own
    ``membership_state``), read from the membership store every STORE_POLL_S with the server's own reader; the time is
    when the change was first read.
    """

    def __init__(self, deployment: Deployment):
        self._base = deployment.base
        self._fault_tolerance = deployment.fault_tolerance
        self._stop = threading.Event()
        self._changed = threading.Condition()
        self._failures: List[str] = []
        self.health: List[Tuple[float, Optional[int], Optional[str]]] = []
        self.membership: List[Tuple[float, dict]] = []
        self._threads = [_thread(self._sample_health, (), self._failures)]
        if deployment.fault_tolerance:
            reader = FaultToleranceHealth(
                _rendezvous(deployment.secret), bytes.fromhex(deployment.secret)
            )
            self._threads.append(_thread(self._sample_membership, (reader,), self._failures))

    def start(self) -> None:
        for thread in self._threads:
            thread.start()

    def stop(self) -> List[str]:
        """Stop watching; returns the samplers that failed (an exception ended them, or they still run)."""
        self._stop.set()
        return _join(self._threads, 60.0, self._failures)

    def _sample_health(self) -> None:
        while not self._stop.is_set():
            code, _ = _get_json(f"{self._base}/health")
            status = _status(self._base) if self._fault_tolerance else None
            self.health.append(
                (time.monotonic(), code, None if status is None else status["state"])
            )
            self._stop.wait(0.25)

    def _sample_membership(self, reader: FaultToleranceHealth) -> None:
        while not self._stop.is_set():
            status = reader.status()
            # The record's own state, which the public status leaves out: the checks time kills within a change.
            record = reader._record
            status["membership_state"] = None if record is None else record.state.name
            t = time.monotonic()
            if not self.membership or status != self.membership[-1][1]:
                with self._changed:
                    self.membership.append((t, status))
                    self._changed.notify_all()
            self._stop.wait(STORE_POLL_S)

    def wait_for(
        self, predicate: Callable[[dict], bool], since: float, timeout_s: float
    ) -> Tuple[float, dict]:
        """The first membership status read after ``since`` that satisfies ``predicate``, and when it was read.

        Returns as soon as the sampler reads it.
        """
        deadline = time.monotonic() + timeout_s
        checked = 0
        with self._changed:
            while True:
                for t, status in self.membership[checked:]:
                    if t > since and predicate(status):
                        return t, status
                checked = len(self.membership)
                remaining = deadline - time.monotonic()
                if remaining <= 0 or self._failures:
                    last = self.membership[-1] if self.membership else None
                    raise AssertionError(
                        f"not reached within {timeout_s} s; last membership read: {last}; "
                        f"failed samplers: {self._failures}"
                    )
                self._changed.wait(min(remaining, 1.0))

    def wait_for_health(self, code: int, state: str, since: float, timeout_s: float) -> float:
        """When ``/health`` first answered ``code``, with the status endpoint's ``state``, after ``since``."""
        checked = 0
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline and not self._failures:
            samples = self.health[checked:]
            checked += len(samples)
            for t, sample_code, sample_state in samples:
                if t > since and (sample_code, sample_state) == (code, state):
                    return t
            time.sleep(0.1)
        last = self.health[-1] if self.health else None
        raise AssertionError(
            f"/health did not answer {code} in state {state} within {timeout_s} s; last sample: {last}; "
            f"failed samplers: {self._failures}"
        )

    def transactions(self) -> Dict[int, Tuple[float, dict]]:
        """Per generation, the time the transaction that ended there was first seen and its summary."""
        seen = {}
        for t, status in self.membership:
            last = status["last_transaction"]
            if last is not None and last["generation"] not in seen:
                seen[last["generation"]] = (t, last)
        return seen


class StallDumps:
    """Dumps the stacks of the node's rank workers once per stall.

    A stall: no streamed token for STALL_DUMP_S while the load runs. Each rank worker's stack, native frames included
    (``py-spy dump --native``), goes to ``stall-<n>-rank<r>.txt`` in ``directory``.

    With ``TLLM_FT_E2E_STALL_DIR``, a stall writes the request ``stall-<n>.request`` there instead, and every node's
    watcher dumps its own rank workers (the node of a lost rank may have none left, while the survivors stall).

    ``dumps``: per stall, when (monotonic) the tokens stopped and were dumped or requested, and the files or the error
    per rank, or the request file.
    ``gdb_from``: from then on (monotonic), a stall also dumps every thread's native stack with gdb (where installed),
    in ``stall-<n>-rank<r>.gdb.txt``. gdb stops a rank for longer than py-spy, long enough for the hang rule to
    remove it, so a run sets it only where a stall is already a failure.
    """

    def __init__(self, deployment: Deployment, clients: "Clients", directory: Path):
        self._secret = deployment.secret
        self._clients = clients
        self._directory = directory
        self._requests = Path(STALL_DIR) if STALL_DIR else None
        self._stop = threading.Event()
        self._failures: List[str] = []
        self._thread = _thread(self._watch, (), self._failures)
        self.dumps: List[dict] = []
        self.gdb_from: Optional[float] = None

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> List[str]:
        self._stop.set()
        return _join([self._thread], 120.0, self._failures) if self._thread.ident else []

    def _watch(self) -> None:
        py_spy = shutil.which("py-spy")
        stalled = False
        while (
            (py_spy or self._requests)
            and len(self.dumps) < MAX_STALL_DUMPS
            and not self._stop.wait(0.5)
        ):
            arrivals = self._clients.stream_arrivals()
            if not arrivals:
                continue  # the first token is still to come
            last = arrivals[-1]
            was_stalled, stalled = stalled, time.monotonic() - last > STALL_DUMP_S
            if stalled and not was_stalled:
                self.dumps.append(
                    self._request(last) if self._requests else self._dump(py_spy, last)
                )

    def _gdb_wanted(self, tokens_stopped: float) -> bool:
        return self.gdb_from is not None and tokens_stopped >= self.gdb_from

    def _request(self, tokens_stopped: float) -> dict:
        """Ask every node's watcher for its rank workers' stacks: ``stall-<n>.request``, written atomically."""
        now, now_unix = time.monotonic(), time.time()
        path = self._requests / f"stall-{len(self.dumps) + 1}.request"
        request = {
            "stall": len(self.dumps) + 1,
            "tokens_stopped_unix": round(now_unix - (now - tokens_stopped), 3),
            "requested_unix": round(now_unix, 3),
            "gdb": self._gdb_wanted(tokens_stopped),
        }
        partial = path.with_name(f".{path.name}.partial")
        partial.write_text(json.dumps(request))
        partial.replace(path)
        return {"tokens_stopped": tokens_stopped, "dumped": now, "request": str(path)}

    def _dump(self, py_spy: str, tokens_stopped: float) -> dict:
        dump = {"tokens_stopped": tokens_stopped, "dumped": time.monotonic(), "ranks": {}}
        try:
            workers = _rank_workers(self._secret)
        except LookupError as error:
            dump["error"] = str(error)
            return dump
        gdb = shutil.which("gdb") if self._gdb_wanted(tokens_stopped) else None
        for rank, process in sorted(workers.items()):
            name = f"stall-{len(self.dumps) + 1}-rank{rank}"
            commands = {"txt": [py_spy, "dump", "--native", "--pid", str(process.pid)]}
            if gdb:
                commands["gdb.txt"] = [
                    gdb,
                    "-p",
                    str(process.pid),
                    "-batch",
                    "-ex",
                    "thread apply all bt",
                ]
            for suffix, command in commands.items():
                path = self._directory / f"{name}.{suffix}"
                try:
                    result = subprocess.run(command, capture_output=True, text=True, timeout=120)
                    path.write_text(result.stdout + result.stderr)
                    dump["ranks"].setdefault(rank, []).append(path.name)
                except subprocess.TimeoutExpired:
                    dump["ranks"].setdefault(rank, []).append(f"{command[0]} timed out")
        return dump


def _all_live(world: int) -> Callable[[dict], bool]:
    return lambda s: s["membership_state"] == "COMMITTED" and s["live_ranks"] == list(range(world))


def _committed_after(generation: int) -> Callable[[dict], bool]:
    return lambda s: s["membership_state"] == "COMMITTED" and s["generation"] > generation


def _admitting(
    ranks: List[int], after_generation: int, states: Tuple[str, ...] = ("PROPOSE", "INSTALL")
) -> Callable[[dict], bool]:
    """A membership change after ``after_generation`` that adds ``ranks`` is in progress, in one of ``states``."""
    return lambda s: (
        s["membership_state"] in states
        and s["generation"] > after_generation
        and set(ranks) <= set(s["live_ranks"])
    )


def _wait_for_replacement(
    secret: str, rank: int, replaced_pid: int, timeout_s: float
) -> psutil.Process:
    """The worker process of ``rank`` once it is no longer ``replaced_pid``."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        process = _rank_workers(secret).get(rank)
        if process is not None and process.pid != replaced_pid:
            return process
        time.sleep(1)
    raise AssertionError(f"no replacement of rank {rank} started within {timeout_s} s")


def _victims(deployment: Deployment, failure: str, workers: Dict[int, psutil.Process]):
    """The ranks to fail on this node: one rank, or every rank for ``node``."""
    if failure == "node":
        assert 0 not in workers, (
            "a node without rank 0 must be failed: run the test on another node"
        )
        return dict(workers)
    assert deployment.victim in workers, (
        f"rank {deployment.victim} does not run on this node (ranks here: {sorted(workers)})"
    )
    return {deployment.victim: workers[deployment.victim]}


def _inject(victims: Dict[int, psutil.Process], failure: str, secret: str) -> List[str]:
    """Fail the victims: ``crash`` kills them, ``hang`` stops them.

    ``node`` kills every process of the deployment on this node (:func:`_node_processes`), the victims' among them, and
    returns those processes ("<pid> <name>").
    """
    if failure != "node":
        for process in victims.values():
            if failure == "hang":
                process.suspend()
            else:
                process.kill()
        return []
    killed = []
    # A round stops every process before it kills any, so that none sees another one end and acts on it (a supervisor
    # starting a replacement); a further round finds a process started meanwhile.
    for _ in range(3):
        processes = _node_processes(secret)
        if not processes:
            break
        for process in processes:
            with contextlib.suppress(psutil.NoSuchProcess):
                killed.append(f"{process.pid} {process.name()}")
                process.suspend()
        for process in processes:
            with contextlib.suppress(psutil.NoSuchProcess):
                process.kill()
        psutil.wait_procs(processes, timeout=10)
    return killed


@dataclass
class Phase:
    name: str
    cycle: int
    start: float
    end: float
    # A phase that spans an event: its requests are those in flight at that time.
    event: Optional[float] = None

    def covers(self, request: Request) -> bool:
        if self.event is not None:
            return request.spans(self.event)
        return self.start <= request.start and request.end <= self.end


class Run:
    """One deployment's run: the load and the monitor, the failures in sequence, the report and the checks.

    Entering starts the load and runs the healthy phase. Leaving stops the load and writes the report, also when a
    step failed, and then checks the requests and the accuracy.
    """

    def __init__(self, deployment: Deployment, tmp_path: Path, name: str):
        self.deployment = deployment
        self.name = name
        self.report_path = Path(os.environ.get("TLLM_FT_E2E_REPORT", tmp_path / "report.json"))
        self.gsm8k = Gsm8k(EVAL_QUESTIONS)
        self.monitor = Monitor(deployment)
        self.clients = Clients(deployment, self.gsm8k)
        self.stall_dumps = StallDumps(deployment, self.clients, self.report_path.parent)
        self.phases: List[Phase] = []
        self.cycles: List[dict] = []
        self.events: List[Tuple[str, float]] = []
        # Times a rank failed or an invalid join was made: only requests in flight then may fail.
        self.failure_times: List[float] = []
        self.resources: List[dict] = []
        self.joins: List[dict] = []
        self.failures: List[str] = []
        self.probes: Dict[str, object] = {}
        self.t0 = time.monotonic()
        self.healthy: Optional[Phase] = None
        # After warmup: the status (with fault tolerance) and each rank's CUDA graph capture lines.
        self.status_after_warmup: Optional[dict] = None
        self.captures_after_warmup: Dict[int, int] = {}

    def check(self, ok: bool, message: str) -> None:
        if not ok:
            self.failures.append(message)

    def event(self, name: str, t: Optional[float] = None) -> float:
        t = time.monotonic() if t is None else t
        self.events.append((name, t))
        return t

    def __enter__(self) -> "Run":
        d = self.deployment
        if d.fault_tolerance:
            status = _status(d.base)
            assert status["live_ranks"] == list(range(d.world_size)) and not status["lost_ranks"], (
                status
            )
            self.status_after_warmup = status
        self.captures_after_warmup = _capture_lines(d.logs)
        self.resources.append({"label": "start", **_resources(d.secret)})
        self.probes = {"reference": _probes(d), "healthy_rerun": _probes(d)}
        self.t0 = time.monotonic()
        self.monitor.start()
        self.clients.start()
        if d.secret is not None:
            self.stall_dumps.start()
        try:
            time.sleep(WARMUP_S)
            self.healthy = self.window("healthy", 0)
            self.resources.append(
                {"label": "at the end of the healthy phase", **_resources(d.secret)}
            )
        except BaseException:
            self.stall_dumps.stop()
            self.clients.stop()
            self.monitor.stop()
            raise
        return self

    def __exit__(self, *exc) -> None:
        for failure in self.stall_dumps.stop() + self.clients.stop() + self.monitor.stop():
            self.check(False, failure)
        failed = exc[0] is not None
        if not failed:
            self.probes["after"] = _probes(self.deployment)
        report = self._report()
        if not failed:
            self._check_requests()
            self._check_accuracy(report["phases"])
            self._check_captures(report["cuda_graph_capture_lines"])
            if self.deployment.fault_tolerance:
                states = {str(s[2]) for s in self.monitor.health}
                self.check(states <= HEALTH_STATES, f"states during the run: {sorted(states)}")
                retryable = sum(1 for r in self.clients.requests if r.retryable)
                logged = _retryable_failures_logged(self.deployment.logs)
                self.check(
                    retryable <= logged,
                    f"{retryable} requests failed as retryable, but rank 0 failed {logged} as retryable",
                )
        # Whole-node restart is not in place yet: a node cycle's rejoin checks are pending (an expected failure), until
        # they all pass.
        pending = [p for c in self.cycles for p in c.get("pending_node_restart", [])]
        self.check(
            not pending or not all(p["ok"] for p in pending),
            "every pending rejoin check of the node cycles passed: whole-node restart works, so make them real checks",
        )
        report["check_failures"] = self.failures
        self.report_path.write_text(json.dumps(report, indent=2))
        print(json.dumps(report, indent=2))
        if not failed:
            assert not self.failures, "\n".join(self.failures) + f"\nreport: {self.report_path}"
            if pending:
                pytest.xfail(
                    "whole-node restart is not in place yet; pending node-cycle checks: "
                    + "; ".join(f"{'ok' if p['ok'] else 'NOT ok'}: {p['check']}" for p in pending)
                )

    def window(self, name: str, cycle: int) -> Phase:
        """Serve WINDOW_S, and on until the phase answered every eval question (at most EVAL_COVER_EXTRA_S more)."""
        self.clients.restart_questions()
        start = time.monotonic()
        time.sleep(WINDOW_S)
        deadline = time.monotonic() + EVAL_COVER_EXTRA_S
        while len(self.clients.answered(start)) < EVAL_QUESTIONS and time.monotonic() < deadline:
            time.sleep(1)
        phase = Phase(name, cycle, start, time.monotonic())
        self.phases.append(phase)
        return phase

    def cycle(
        self,
        failure: str,
        at_admission: Optional[Callable[[], Dict[int, psutil.Process]]] = None,
        at_admission_state: Optional[str] = None,
    ) -> dict:
        """Fail ranks (``crash``, ``hang`` or ``node``) and wait until replacements are back at N ranks.

        Then serve a window there. ``at_admission`` returns processes to kill once the replacement's admission has
        begun, or, with ``at_admission_state`` (``PROPOSE`` or ``INSTALL``), once the store shows it in that state.
        """
        d, index = self.deployment, len(self.cycles) + 1
        world, all_ranks = d.world_size, list(range(d.world_size))
        status = _status(d.base)
        assert status["live_ranks"] == all_ranks and status["lost_ranks"] == [], status
        workers_before = _rank_workers(d.secret)  # scanning the processes takes seconds
        gpus_before = _rank_gpus(workers_before)
        captures_before = _capture_lines(d.logs)
        victims = _victims(d, failure, workers_before)
        lost = sorted(victims)
        record = {"cycle": index, "failure": failure, "lost_ranks": lost}
        self.cycles.append(record)

        t_fail = self.event(f"cycle {index}: {failure} of ranks {lost}")
        killed = _inject(victims, failure, d.secret)
        self.failure_times.append(t_fail)
        if failure == "node":
            record["node_processes_killed"] = killed
            left = [p.pid for p in _node_processes(d.secret)]
            assert {p.pid for p in victims.values()} <= {
                int(k.split()[0]) for k in killed
            } and not left, (
                f"the node's processes were not all killed: killed {killed}, left {left}"
            )
        detection_s = T_HUNG_S if failure == "hang" else 0.0
        t_degraded, failover_status = self.monitor.wait_for(
            _committed_after(status["generation"]), t_fail, detection_s + RECOVERY_TIMEOUT_S
        )
        self.event(f"cycle {index}: smaller membership committed", t_degraded)
        self.monitor.wait_for_health(200, "degraded", t_degraded, RECOVERY_TIMEOUT_S)
        self.phases.append(Phase("failover", index, t_fail, t_degraded, event=t_fail))
        time.sleep(SETTLE_S)
        t_settled = time.monotonic()
        self.clients.restart_questions()
        if failure == "hang":
            # The excluded process runs again under load: it must not disturb the survivors, and it exits.
            (process,) = victims.values()
            try:
                process.resume()
                process.wait(EXCLUDED_EXIT_TIMEOUT_S)
                record["excluded_process"] = "exited"
            except psutil.NoSuchProcess:
                record["excluded_process"] = "exited"
            except psutil.TimeoutExpired:
                record["excluded_process"] = (
                    f"still running {EXCLUDED_EXIT_TIMEOUT_S} s after it was resumed"
                )
            self.check(
                record["excluded_process"] == "exited",
                f"cycle {index}: the excluded process is {record['excluded_process']}",
            )

        killed_at_admission: List[int] = []
        if at_admission is not None:
            processes = at_admission()
            killed_at_admission = sorted(processes)
            # Killed as soon as the admission is read from the store: at its first PROPOSE, or in
            # ``at_admission_state``.
            states = ("PROPOSE", "INSTALL") if at_admission_state is None else (at_admission_state,)
            t_admitting, admitting = self.monitor.wait_for(
                _admitting(lost, failover_status["generation"], states),
                t_degraded,
                REJOIN_TIMEOUT_S,
            )
            for process in processes.values():
                process.kill()
            t_kill = self.event(
                f"cycle {index}: killed ranks {killed_at_admission} during the admission"
            )
            self.event(f"cycle {index}: admission of ranks {lost} began", t_admitting)
            self.failure_times.append(t_kill)
            # The kill must land before the admission commits: the admission's generation never commits, and the
            # next committed record leaves the killed ranks out.
            t_after_kill, after_kill = self.monitor.wait_for(
                _committed_after(admitting["generation"]), t_kill, RECOVERY_TIMEOUT_S
            )
            self.event(f"cycle {index}: membership after the kill committed", t_after_kill)
            self.phases.append(
                Phase("admission_failure", index, t_kill, t_after_kill, event=t_kill)
            )
            record["admission_failure"] = {
                "killed_ranks": killed_at_admission,
                "status_at_kill": admitting,
                "status_after": after_kill,
            }
            self.check(
                admitting["generation"] not in self.monitor.transactions()
                and not set(killed_at_admission) & set(after_kill["live_ranks"]),
                f"cycle {index}: the kill did not land during the admission: {after_kill}",
            )

        # Whole-node restart is not in place yet: a node cycle's rejoin checks are pending (Run.__exit__), and it
        # serves one window at N-K instead of waiting for a rejoin that nothing starts.
        rejoin_check = self._pending_check(record) if failure == "node" else self.check
        rejoined = rejoined_status = t_rejoined = None
        t_admission = None
        try:
            t_rejoined, rejoined_status = self.monitor.wait_for(
                _all_live(world), t_degraded, WINDOW_S if failure == "node" else REJOIN_TIMEOUT_S
            )
            # Serving at N ranks must not stall from here (checked below): a stall gets native stacks.
            self.stall_dumps.gdb_from = t_rejoined
            self.monitor.wait_for_health(200, "healthy", t_rejoined, RECOVERY_TIMEOUT_S)
        except AssertionError as error:
            if failure != "node":
                raise
            rejoin_check(False, f"cycle {index}: back to N ranks: {error}")
        if rejoined_status is not None:
            admission = rejoined_status["last_transaction"]
            # The commit was read within STORE_POLL_S; the authority measures the transaction itself.
            t_admission = (
                t_rejoined - (admission["ended_unix_ms"] - admission["began_unix_ms"]) / 1e3
            )
            self.event(f"cycle {index}: admission began", t_admission)
            self.event(f"cycle {index}: back to N ranks", t_rejoined)
        degraded_end = (time.monotonic() if t_admission is None else t_admission) - 1.0
        self.phases.append(Phase("degraded", index, t_settled, degraded_end))
        if rejoined_status is not None:
            self.phases.append(
                Phase("admission", index, t_admission, t_rejoined, event=t_admission)
            )
            time.sleep(SETTLE_S)
            rejoined = self.window("rejoined", index)
            # Serving resumed at N ranks: the streamed tokens do not stall from the commit to the window's end.
            stall_s = _largest_gap(self.clients.stream_arrivals(), t_rejoined, rejoined.end)[0]
            self.check(
                stall_s <= STALL_DUMP_S,
                f"cycle {index}: serving stalled after the admission: no streamed token for {stall_s:.1f} s",
            )

        workers_after = _rank_workers(d.secret)
        gpus_after = _rank_gpus(workers_after)
        captures_after = _capture_lines(d.logs)
        self.resources.append({"label": f"after cycle {index}", **_resources(d.secret)})
        survivors = [r for r in all_ranks if r not in lost]
        local_survivors = [
            r for r in workers_before if r not in lost and r not in killed_at_admission
        ]

        # Pauses: the longest time no stream received a token, at the failure and while the replacement was admitted.
        arrivals = self.clients.stream_arrivals()
        pause_end = (degraded_end + 1.0 if at_admission is None else t_admitting) - 1.0
        pause_s, pause_start = _largest_gap(arrivals, t_fail - 1.0, pause_end)
        if at_admission is not None:
            kill_pause_s = _largest_gap(arrivals, t_admitting - 1.0, t_after_kill + SETTLE_S)[0]
            record["admission_failure"]["pause_s"] = round(kill_pause_s, 3)
            self.check(
                kill_pause_s < CRASH_PAUSE_TARGET_S,
                f"cycle {index}: pause {kill_pause_s:.2f} s at the kill during the admission, target < "
                f"{CRASH_PAUSE_TARGET_S} s",
            )
        pause_target_s = (
            T_HUNG_S + HANG_PAUSE_MARGIN_S if failure == "hang" else CRASH_PAUSE_TARGET_S
        )
        spanning = [r for r in self.clients.load_requests() if r.spans(t_fail)]
        # Requests sent again after a retryable failure at this cycle's failures.
        retries = [
            r for r in self.clients.requests if r.retry and t_fail <= r.start <= degraded_end
        ]
        healthy_tput = self.clients.throughput(self.healthy.start, self.healthy.end)
        record.update(
            {
                "pause_s": round(pause_s, 3),
                "pause_starts_after_failure_s": round(pause_start - t_fail, 3),
                "pause_target_s": pause_target_s,
                "time_to_degraded_s": round(t_degraded - t_fail, 3),
                "in_flight_at_failure": len(spanning),
                "in_flight_non_streamed_completed": sum(
                    1 for r in spanning if not r.stream and r.error is None
                ),
                "in_flight_streamed_completed": sum(
                    1 for r in spanning if r.stream and r.error is None
                ),
                "retries": {
                    "sent": len(retries),
                    "succeeded": sum(1 for r in retries if r.error is None),
                },
                "throughput_tok_s": {
                    "healthy": healthy_tput,
                    "degraded": self.clients.throughput(t_settled, degraded_end),
                },
                "status": {"before": status, "failover": failover_status},
                "worker_pids": {
                    "before": {r: p.pid for r, p in sorted(workers_before.items())},
                    "after": {r: p.pid for r, p in sorted(workers_after.items())},
                },
                "rank_gpus": {"before": gpus_before, "after": gpus_after},
                "cuda_graph_capture_lines": {"before": captures_before, "after": captures_after},
            }
        )

        self.check(
            record["in_flight_non_streamed_completed"] > 0,
            f"cycle {index}: no non-streamed request in flight at the failure completed",
        )
        self.check(
            record["in_flight_streamed_completed"] > 0,
            f"cycle {index}: no streamed request in flight at the failure completed",
        )
        self.check(
            record["retries"]["succeeded"] > 0,
            f"cycle {index}: no request sent again after a retryable failure succeeded ({record['retries']})",
        )
        self.check(
            pause_s <= pause_target_s if failure == "hang" else pause_s < pause_target_s,
            f"cycle {index}: pause {pause_s:.2f} s, target {pause_target_s} s",
        )
        # Status: generation, live and lost ranks.
        self.check(
            failover_status["generation"] > status["generation"]
            and failover_status["lost_ranks"] == lost
            and failover_status["live_ranks"] == survivors
            and failover_status["authority_alive"],
            f"cycle {index}: status after the failover: {failover_status}",
        )
        # No survivor restarted, and none captured a CUDA graph after startup.
        for rank in local_survivors:
            self.check(
                rank in workers_after and workers_after[rank].pid == workers_before[rank].pid,
                f"cycle {index}: survivor rank {rank} was restarted",
            )
        for rank in survivors:
            if rank in killed_at_admission:
                continue
            self.check(
                captures_before.get(rank, 0) > 0, f"no CUDA graph capture lines of rank {rank}"
            )
            self.check(
                captures_after.get(rank, 0) == captures_before.get(rank, 0),
                f"cycle {index}: rank {rank} captured CUDA graphs after the failure",
            )
        if rejoined_status is None:
            return record

        # The rejoin.
        admission_pause_s = _largest_gap(arrivals, t_admission - 1.0, t_rejoined + SETTLE_S)[0]
        rejoined_tput = self.clients.throughput(rejoined.start, rejoined.end)
        record.update(
            {
                "replacement_startup_s": round(t_admission - t_degraded, 3),
                "admission_pause_s": round(admission_pause_s, 3),
                "admission_pause_target_s": ADMISSION_PAUSE_TARGET_S,
                "time_to_n_s": round(t_rejoined - t_fail, 3),
            }
        )
        record["throughput_tok_s"]["rejoined"] = rejoined_tput
        record["status"]["rejoined"] = rejoined_status
        rejoin_check(
            rejoined_status["lost_ranks"] == []
            and rejoined_status["generation"] > failover_status["generation"],
            f"cycle {index}: status after the rejoin: {rejoined_status}",
        )
        # A survivor keeps its incarnation; a replacement has the one after the process it replaces (a replacement
        # killed during its admission is replaced once more).
        expected = [
            n + (r in lost) + (r in killed_at_admission)
            for r, n in enumerate(status["incarnations"])
        ]
        rejoin_check(
            rejoined_status["incarnations"] == expected,
            f"cycle {index}: incarnations {status['incarnations']} -> {rejoined_status['incarnations']}, "
            f"expected {expected}",
        )
        # A replacement runs on the GPU of the process it replaces.
        for rank in sorted((set(lost) | set(killed_at_admission)) & set(workers_before)):
            rejoin_check(
                bool(gpus_before[rank]) and gpus_after.get(rank) == gpus_before[rank],
                f"cycle {index}: rank {rank}'s replacement runs on GPUs {gpus_after.get(rank)}, the process it "
                f"replaces ran on {gpus_before[rank]}",
            )
        # A replacement captures its own CUDA graphs at startup.
        for rank in sorted(set(lost) | set(killed_at_admission)):
            rejoin_check(
                captures_after.get(rank, 0) > captures_before.get(rank, 0),
                f"cycle {index}: no CUDA graph capture lines of rank {rank}'s replacement",
            )
        rejoin_check(
            admission_pause_s < ADMISSION_PAUSE_TARGET_S,
            f"cycle {index}: admission pause {admission_pause_s:.2f} s, target < {ADMISSION_PAUSE_TARGET_S} s",
        )
        healthy_mean = statistics.mean(healthy_tput)
        noise = (max(healthy_tput) - min(healthy_tput)) / healthy_mean
        rejoined_mean = statistics.mean(rejoined_tput)
        rejoin_check(
            rejoined_mean >= healthy_mean * (1 - max(noise, THROUGHPUT_TOLERANCE)),
            f"cycle {index}: throughput after the rejoin {rejoined_mean:.0f} tok/s, healthy {healthy_mean:.0f} "
            f"tok/s (noise {noise:.1%})",
        )
        return record

    @staticmethod
    def _pending_check(record: dict) -> Callable[[bool, str], None]:
        """A check recorded under ``pending_node_restart``.

        The run is an expected failure while one of them fails, and fails once they all pass (Run.__exit__).
        """
        pending = record.setdefault("pending_node_restart", [])
        return lambda ok, message: pending.append({"check": message, "ok": bool(ok)})

    @contextlib.contextmanager
    def connect(self, rank: int, replacement: bool):
        """Connect to the authority for ``rank`` and nothing more; yields the time.

        As a ``replacement`` or as the rank's first process. That is how a replacement asks to join once it has
        started (its signed liveness connection). The connection stays open until the context ends.
        """
        d = self.deployment
        store = open_store(_rendezvous(d.secret), bytes.fromhex(d.secret), timeout_s=60.0)
        t = self.event(f"liveness connection: rank {rank}, replacement {replacement}")
        with open_liveness_connection(store, rank, replacement):
            yield t

    @contextlib.contextmanager
    def refused(self, rank: int, replacement: bool, reason: str):
        """:meth:`connect`, which the authority must refuse with ``reason``: it logs the refusal."""
        line = REFUSAL_LOG + reason
        logged = _log_count(self.deployment.logs, line)
        with self.connect(rank, replacement) as t:
            yield t
        self.check(
            _log_count(self.deployment.logs, line) > logged,
            f"the authority logged no refusal of a hello for rank {rank} (replacement {replacement}): {line!r}",
        )

    def _phase_report(
        self, phase: Phase, scores: Dict[int, Dict[str, float]]
    ) -> Tuple[dict, Dict[int, float]]:
        """The phase's measurements, and per eval question its first answer's score with ACCURACY_FILTER."""
        load = [r for r in self.clients.load_requests() if phase.covers(r)]
        streamed = [r for r in load if r.stream and r.error is None and r.arrivals]
        span = max(phase.end - phase.start, 1e-3)
        tokens = good = 0
        for r in self.clients.requests:
            n = sum(c for t, c in r.arrivals if phase.start <= t < phase.end)
            tokens += n
            good += n if r.error is None else 0
        first: Dict[int, Dict[str, float]] = {}
        for i in sorted(scores, key=lambda i: self.clients.requests[i].start):
            r = self.clients.requests[i]
            if phase.covers(r):
                first.setdefault(r.question, scores[i])
        accuracy = {
            name: round(statistics.mean(s[name] for s in first.values()), 4)
            for name in (next(iter(first.values())) if first else {})
        }
        return {
            "phase": phase.name,
            "cycle": phase.cycle,
            "start_s": round(phase.start - self.t0, 3),
            "duration_s": round(phase.end - phase.start, 3),
            "load_requests": len(load),
            "load_requests_failed": sum(1 for r in load if r.error is not None),
            "ttft_s": _percentiles([r.arrivals[0][0] - r.start for r in streamed]),
            "tpot_s": _percentiles(
                [
                    (r.arrivals[-1][0] - r.arrivals[0][0]) / (len(r.token_ids) - 1)
                    for r in streamed
                    if len(r.token_ids) > 1
                ]
            ),
            # Every client's output tokens that arrived in the phase; goodput counts those of requests that succeeded.
            "output_tok_s": round(tokens / span, 1),
            "goodput_tok_s": round(good / span, 1),
            "gsm8k": {"questions": len(first), "accuracy": accuracy},
        }, {q: s[ACCURACY_FILTER] for q, s in first.items()}

    def _phase_reports(self, scores: Dict[int, Dict[str, float]]) -> List[dict]:
        """Per phase its measurements; past the healthy phase, the paired comparison with it (ACCURACY_FILTER)."""
        reports, answers = (
            zip(*(self._phase_report(p, scores) for p in self.phases)) if self.phases else ((), ())
        )
        healthy = answers[0] if answers else {}
        alpha = ACCURACY_ALPHA / max(1, len(reports) - 1)
        for report, answer in zip(reports[1:], answers[1:]):
            paired = set(answer) & set(healthy)
            lost = sum(1 for q in paired if healthy[q] == 1.0 and answer[q] == 0.0)
            gained = sum(1 for q in paired if healthy[q] == 0.0 and answer[q] == 1.0)
            report["gsm8k"].update(
                {
                    "paired": len(paired),
                    "right_when_healthy_wrong_here": lost,
                    "wrong_when_healthy_right_here": gained,
                    "p_loss": _sign_test_p(lost, gained),
                    "alpha": alpha,
                }
            )
        return list(reports)

    def _report(self) -> dict:
        scores = {
            i: self.gsm8k.score(r.question, r.text)
            for i, r in enumerate(self.clients.requests)
            if r.question is not None and r.error is None
        }
        end = max([self.t0] + [s[0] for s in self.monitor.health])
        timeline = [0] * (int(end - self.t0) + 1)
        for r in self.clients.load_requests():
            for t, n in r.arrivals:
                if r.stream and self.t0 <= t <= end:
                    timeline[int(t - self.t0)] += n
        reference = self.probes.get("reference") or []
        first_difference = {
            name: [
                next((i for i, (a, b) in enumerate(zip(ref, other)) if a != b), None)
                for ref, other in zip(reference, outputs)
            ]
            for name, outputs in self.probes.items()
            if name in ("healthy_rerun", "after")
        }
        requests = self.clients.requests
        report = {
            "test": self.name,
            "world_size": self.deployment.world_size,
            "fault_tolerance": self.deployment.fault_tolerance,
            "load_clients": min(CLIENTS_PER_RANK * self.deployment.world_size, MAX_CLIENTS),
            "eval_clients": min(
                EVAL_CLIENTS_PER_RANK * self.deployment.world_size, MAX_EVAL_CLIENTS
            ),
            "phases": self._phase_reports(scores),
            "cycles": self.cycles,
            "cuda_graph_capture_lines": {
                "after_warmup": self.captures_after_warmup,
                "end": _capture_lines(self.deployment.logs),
            },
            "joins": self.joins,
            "transactions": [
                {"seen_s": round(t - self.t0, 3), **summary}
                for _, (t, summary) in sorted(self.monitor.transactions().items())
            ],
            # Every change of the membership read from the store.
            "membership": [
                {
                    "t_s": round(t - self.t0, 3),
                    **{
                        k: s[k]
                        for k in ("generation", "membership_state", "live_ranks", "authority_alive")
                    },
                }
                for t, s in self.monitor.membership
            ],
            "events": [{"name": n, "t_s": round(t - self.t0, 3)} for n, t in self.events],
            # Streamed chunks with tokens per second from the start of the load.
            "stream_chunks_per_s": timeline,
            "requests": {
                "total": len(requests),
                "eval": sum(1 for r in requests if r.question is not None),
                "streamed": sum(1 for r in requests if r.stream),
                "failed": sum(1 for r in requests if r.error is not None),
                "retried": sum(1 for r in requests if r.retry),
                "retries_failed": sum(1 for r in requests if r.retry and r.error is not None),
            },
            "resources": [
                {**r, "time": round(r["time"] - self.t0, 3), "shm_segments": len(r["shm_segments"])}
                for r in self.resources
            ],
            # Smoke signal: greedy probes sent one at a time before and after the load.
            "probes": {
                "reproducible_when_healthy": sum(
                    a == b for a, b in zip(reference, self.probes.get("healthy_rerun") or [])
                ),
                "matching_after": sum(
                    a == b for a, b in zip(reference, self.probes.get("after") or [])
                ),
                "first_difference": first_difference,
            },
            "stall_dumps": [
                {
                    **dump,
                    "tokens_stopped": round(dump["tokens_stopped"] - self.t0, 3),
                    "dumped": round(dump["dumped"] - self.t0, 3),
                }
                for dump in self.stall_dumps.dumps
            ],
        }
        return report

    def _check_requests(self) -> None:
        """Serving continued: every request completed, a load request with its full length.

        Except a streamed request in flight at a failure that had streamed tokens: it fails as retryable. A request
        sent again after a retryable failure must complete. No streamed token is duplicated.
        """
        for r in self.clients.requests:
            kind = ("retried " if r.retry else "") + ("streamed" if r.stream else "non-streamed")
            kind += " eval" if r.question is not None else ""
            if r.error is None:
                if r.question is None:
                    self.check(
                        len(r.token_ids) == r.completion_tokens == LOAD_MAX_TOKENS,
                        f"a {kind} request returned {len(r.token_ids)} tokens (usage {r.completion_tokens}), "
                        f"not {LOAD_MAX_TOKENS}",
                    )
                continue
            if r.retry:
                self.check(
                    False,
                    f"a {kind} request failed (started {r.start - self.t0:.1f} s into the run): {r.error[:500]}",
                )
                continue
            spanned = [t for t in self.failure_times if r.spans(t)]
            self.check(
                bool(r.stream and r.retryable and spanned and r.token_ids),
                f"a {kind} request failed (started {r.start - self.t0:.1f} s, ended {r.end - self.t0:.1f} s "
                f"into the run, failures at {[round(t - self.t0, 1) for t in self.failure_times]}, "
                f"{len(r.token_ids)} tokens): {r.error[:500]}",
            )
            self.check(
                len(r.token_ids) < LOAD_MAX_TOKENS, "a failed stream returned too many tokens"
            )

    def _check_accuracy(self, phases: List[dict]) -> None:
        """The healthy phase answered every GSM8K question of the subset.

        Every other phase answered some of them, and none lost accuracy against the healthy phase (paired).
        """
        for i, phase in enumerate(phases):
            g, label = phase["gsm8k"], f"cycle {phase['cycle']} {phase['phase']}"
            if i == 0:
                self.check(
                    g["questions"] == EVAL_QUESTIONS,
                    f"{label}: {g['questions']} of the {EVAL_QUESTIONS} GSM8K questions answered",
                )
                continue
            self.check(
                g["paired"] > 0,
                f"{label}: no GSM8K question answered here and in the healthy phase",
            )
            self.check(
                g["p_loss"] >= g["alpha"],
                f"{label}: GSM8K lost accuracy against the healthy phase: of {g['paired']} questions both "
                f"answered, {g['right_when_healthy_wrong_here']} went right -> wrong and "
                f"{g['wrong_when_healthy_right_here']} wrong -> right (p {g['p_loss']:.2g} < {g['alpha']:.2g})",
            )

    def _check_captures(self, captures: Dict[str, Dict[int, int]]) -> None:
        """Every rank captured CUDA graphs at startup, and no rank still served by its first process captured since.

        A rank is still served by its first process while it is live with the incarnation it had after warmup.
        """
        d = self.deployment
        first_processes = list(range(d.world_size))
        if d.fault_tolerance:
            start, end = self.status_after_warmup["incarnations"], _status(d.base)
            first_processes = [r for r in end["live_ranks"] if end["incarnations"][r] == start[r]]
        before, after = captures["after_warmup"], captures["end"]
        for rank in range(d.world_size):
            self.check(before.get(rank, 0) > 0, f"no CUDA graph capture lines of rank {rank}")
        for rank in first_processes:
            self.check(
                after.get(rank, 0) == before.get(rank, 0),
                f"rank {rank} captured CUDA graphs after warmup: {before.get(rank, 0)} -> "
                f"{after.get(rank, 0)} capture lines",
            )


def _sign_test_p(lost: int, gained: int) -> float:
    """One-sided exact sign test (McNemar on discordant pairs) for a loss.

    The chance of at least ``lost`` losses among ``lost + gained`` discordant pairs when a loss and a gain are equally
    likely (1 without discordant pairs).
    """
    n = lost + gained
    return sum(math.comb(n, k) for k in range(lost, n + 1)) / 2**n


def _join_record(
    kind: str, replacement: bool, before: dict, after: dict, workers, after_workers
) -> dict:
    return {
        "join": kind,
        "replacement": replacement,
        "status_before": before,
        "status_after": after,
        "worker_pids": {
            "before": {r: p.pid for r, p in sorted(workers.items())},
            "after": {r: p.pid for r, p in sorted(after_workers.items())},
        },
    }


def _sequence() -> List[str]:
    return os.environ.get("TLLM_FT_E2E_SEQUENCE", "crash,hang").split(",")


@pytest.mark.skip_less_device(WORLD_SIZE)
@pytest.mark.timeout(STARTUP_TIMEOUT_S + 2 * REJOIN_TIMEOUT_S + 1200)
def test_rank_failure_sequence(deployment: Deployment, tmp_path: Path):
    """The failures of ``TLLM_FT_E2E_SEQUENCE`` (default: a crash, then a hang) in one deployment under load.

    After each, a replacement rejoins. The survivors keep serving on the same CUDA graphs, with the healthy accuracy.
    ``none`` serves one more healthy window instead, scored against the first like every later phase: how far two
    healthy windows differ is the noise the paired comparison sees.
    """
    with Run(deployment, tmp_path, "sequence") as run:
        for index, failure in enumerate(_sequence(), start=1):
            if failure == "none":
                run.window("healthy_again", index)
            else:
                run.cycle(failure)


@pytest.mark.skip_less_device(WORLD_SIZE)
@pytest.mark.timeout(STARTUP_TIMEOUT_S + 2 * REJOIN_TIMEOUT_S + 1200)
def test_rank_failure_sequence_with_serving_options(
    serving_options_deployment: Deployment, tmp_path: Path
):
    """:func:`test_rank_failure_sequence` with the options large deployments serve with.

    Attention-DP balancing, CUDA graphs up to batch 512, an fp8 KV cache with a host tier, online EPLB with a redundant
    slot per rank (:func:`_serving_config`).
    """
    with Run(serving_options_deployment, tmp_path, "sequence_with_serving_options") as run:
        for failure in _sequence():
            run.cycle(failure)


@pytest.mark.skip_less_device(WORLD_SIZE)
@pytest.mark.timeout(STARTUP_TIMEOUT_S + REJOIN_TIMEOUT_S + 1200)
@pytest.mark.parametrize(
    "failure",
    [
        "crash",
        "hang",
        pytest.param(
            "node",
            marks=pytest.mark.skipif(
                not EXTERNAL,
                reason="whole-node loss needs a multi-node deployment (TLLM_FT_E2E_SERVER)",
            ),
        ),
    ],
)
def test_rank_failure_under_load(deployment: Deployment, tmp_path: Path, failure: str):
    """One failure under load and the rejoin (a shorter run of :func:`test_rank_failure_sequence`).

    ``crash``: a rank is killed; ``hang``: a rank is stopped; ``node``: every process of the deployment on a node is
    killed (rank workers, node supervisors, launcher processes).
    """
    with Run(deployment, tmp_path, failure) as run:
        run.cycle(failure)


@pytest.mark.skip_less_device(WORLD_SIZE)
@pytest.mark.timeout(STARTUP_TIMEOUT_S + 2 * REJOIN_TIMEOUT_S + 1200)
def test_repeated_rank_reuse(deployment: Deployment, tmp_path: Path):
    """The same rank crashes twice, each time replaced on its GPU.

    After each cycle's rejoined window, no GPU uses more memory (beyond GPU_MEMORY_LEAK_TOLERANCE_MIB) than at the end
    of the healthy phase or after the cycle before, and no /dev/shm segment is orphaned (no process maps it or holds
    it open) that was not orphaned then; no process that survived both cycles holds more file descriptors after the
    second than after the first.
    """
    with Run(deployment, tmp_path, "repeated_reuse") as run:
        run.cycle("crash")
        run.cycle("crash")
        snapshots = {r["label"]: r for r in run.resources}
        for later, earlier in (
            ("after cycle 1", "at the end of the healthy phase"),
            ("after cycle 2", "at the end of the healthy phase"),
            ("after cycle 2", "after cycle 1"),
        ):
            new, old = snapshots[later], snapshots[earlier]
            for gpu, used in new["gpu_memory_used_mib"].items():
                before = old["gpu_memory_used_mib"].get(gpu)
                run.check(
                    before is not None and used <= before + GPU_MEMORY_LEAK_TOLERANCE_MIB,
                    f"GPU {gpu}: {used} MiB used {later}, {before} MiB {earlier}",
                )
            orphaned = sorted(set(new["shm_orphans"]) - set(old["shm_orphans"]))
            run.check(
                not orphaned, f"/dev/shm segments orphaned {later}, not {earlier}: {orphaned}"
            )
        # Once a cycle has run, a second one opens no more descriptors in a process that survived both.
        new, old = (
            snapshots["after cycle 2"]["rank_open_fds"],
            snapshots["after cycle 1"]["rank_open_fds"],
        )
        for rank, (pid, fds) in new.items():
            if rank in old and old[rank][0] == pid:
                run.check(
                    fds <= old[rank][1],
                    f"rank {rank} (pid {pid}): {fds} open descriptors after cycle 2, {old[rank][1]} after cycle 1",
                )


def _refused_join(run: Run, kind: str, replacement: bool, reason: str) -> None:
    """A join for the victim's rank, which the authority refuses with ``reason``.

    As a ``replacement`` or as the rank's first process. Watched INVALID_JOIN_WATCH_S with the connection open: the
    membership does not change and no process restarts.
    """
    d = run.deployment
    workers, before = _rank_workers(d.secret), _status(d.base)
    with run.refused(d.victim, replacement, reason) as t_join:
        time.sleep(INVALID_JOIN_WATCH_S)
    run.phases.append(Phase(f"{kind}_join", len(run.cycles), t_join, time.monotonic()))
    after, after_workers = _status(d.base), _rank_workers(d.secret)
    run.joins.append(_join_record(kind, replacement, before, after, workers, after_workers))
    run.check(
        after["generation"] == before["generation"]
        and after["live_ranks"] == list(range(d.world_size)),
        f"a {kind} join for rank {d.victim} changed the membership: {before} -> {after}",
    )
    run.check(
        {r: p.pid for r, p in after_workers.items()} == {r: p.pid for r, p in workers.items()},
        f"a process restarted after a {kind} join",
    )


@pytest.mark.skip_less_device(WORLD_SIZE)
@pytest.mark.timeout(STARTUP_TIMEOUT_S + 3 * REJOIN_TIMEOUT_S + 1200)
def test_invalid_joins(deployment: Deployment, tmp_path: Path):
    """Joins the authority refuses, and a replacement that fails during its admission, under load in one deployment.

    - A second connection of a live first process: the process's crash is still failed over at once.
    - A replaced first process, and a replacement of a live rank (as a second supervisor would send).
    - A replacement killed during its admission: the survivors go on without it, and the next replacement rejoins.

    The authority logs each refusal; the deployment keeps serving, and no survivor restarts.
    """
    d = deployment
    with Run(d, tmp_path, "invalid_joins") as run:
        with run.refused(d.victim, False, f"rank {d.victim}'s first process is connected already"):
            run.cycle("crash")
        _refused_join(run, "stale", False, f"rank {d.victim}'s first process was replaced")
        _refused_join(
            run, "duplicate", True, f"rank {d.victim} is live: no replacement joins for it"
        )

        victim_pid = _rank_workers(d.secret)[d.victim].pid
        run.cycle(
            "crash",
            at_admission=lambda: {
                d.victim: _wait_for_replacement(d.secret, d.victim, victim_pid, REJOIN_TIMEOUT_S)
            },
        )


@pytest.mark.skip_less_device(WORLD_SIZE)
@pytest.mark.timeout(STARTUP_TIMEOUT_S + 1200)
def test_silent_joiner_does_not_hold_survivors(deployment: Deployment, tmp_path: Path):
    """A joiner for a lost rank that connects and never replies pauses the survivors less than an admission does."""
    d = deployment
    with Run(d, tmp_path, "silent_joiner") as run:
        victim = _rank_workers(d.secret)[d.victim]
        t_fail = run.event(f"crash of rank {d.victim}")
        victim.kill()
        run.failure_times.append(t_fail)
        t_degraded = run.monitor.wait_for(
            lambda s: s["membership_state"] == "COMMITTED" and d.victim in s["lost_ranks"],
            t_fail,
            RECOVERY_TIMEOUT_S,
        )[0]
        time.sleep(SETTLE_S)
        # It never reports that it is ready to join, so the authority never stops the survivors for it.
        with run.connect(d.victim, True) as t_join:
            time.sleep(T_HUNG_S + SETTLE_S)
        run.phases.append(Phase("failover", 0, t_fail, t_degraded, event=t_fail))
        run.phases.append(Phase("silent_joiner", 0, t_join, time.monotonic()))
        pause_s = _largest_gap(
            run.clients.stream_arrivals(), t_join - 1.0, t_join + T_HUNG_S + SETTLE_S
        )[0]
        run.joins.append({"join": "silent", "pause_s": round(pause_s, 3)})
        run.check(
            pause_s < ADMISSION_PAUSE_TARGET_S,
            f"a joiner that never replied paused the survivors {pause_s:.2f} s, target < "
            f"{ADMISSION_PAUSE_TARGET_S} s",
        )


@pytest.mark.skip_less_device(WORLD_SIZE)
@pytest.mark.timeout(STARTUP_TIMEOUT_S + 2 * REJOIN_TIMEOUT_S + 1200)
def test_failure_during_admission(deployment: Deployment, tmp_path: Path):
    """A second rank crashes while the first one's replacement is admitted.

    The survivors go on without it, the first replacement is admitted, the second rank's replacement rejoins, and
    the deployment is back at N.
    """
    d = deployment
    with Run(d, tmp_path, "failure_during_admission") as run:
        workers = _rank_workers(d.secret)
        second = min((r for r in workers if r not in (0, d.victim)), default=None)
        assert second is not None, (
            f"no second rank to fail on this node (ranks here: {sorted(workers)})"
        )
        run.cycle("crash", at_admission=lambda: {second: workers[second]})


@pytest.mark.skip_less_device(WORLD_SIZE)
@pytest.mark.timeout(STARTUP_TIMEOUT_S + 2 * REJOIN_TIMEOUT_S + 1200)
def test_failure_during_install(deployment: Deployment, tmp_path: Path):
    """A second rank crashes while the live ranks install the first one's admission, not before it.

    INSTALL is the remap of the all-to-all memory and the expert placement. The admission does not commit, the
    survivors go on without the second rank, and both replacements rejoin.
    """
    d = deployment
    with Run(d, tmp_path, "failure_during_install") as run:
        workers = _rank_workers(d.secret)
        second = min((r for r in workers if r not in (0, d.victim)), default=None)
        assert second is not None, (
            f"no second rank to fail on this node (ranks here: {sorted(workers)})"
        )
        run.cycle(
            "crash", at_admission=lambda: {second: workers[second]}, at_admission_state="INSTALL"
        )


@pytest.mark.skip_less_device(WORLD_SIZE)
@pytest.mark.skipif(EXTERNAL, reason="starts its own deployment")
@pytest.mark.timeout(STARTUP_TIMEOUT_S + REJOIN_TIMEOUT_S + 1200)
def test_rank_rejoins_when_the_server_spawns_its_ranks(
    spawned_deployment: Deployment, tmp_path: Path
):
    """Plain trtllm-serve, without trtllm-llmapi-launch and with no variable exported.

    The server starts with fault tolerance (it chooses the join secret and the rendezvous address before it spawns
    the ranks), a killed rank fails over, and its node supervisor restarts it as a replacement that rejoins.
    """
    with Run(spawned_deployment, tmp_path, "server_spawns_ranks") as run:
        run.cycle("crash")


@pytest.mark.skip_less_device(WORLD_SIZE)
@pytest.mark.timeout(STARTUP_TIMEOUT_S + 1200)
@pytest.mark.parametrize(
    "launch",
    [
        "deployment",
        pytest.param(
            "spawned_deployment",
            marks=pytest.mark.skipif(EXTERNAL, reason="starts its own deployment"),
        ),
    ],
)
def test_every_rank_serves_as_its_own_process(launch: str, request, tmp_path: Path):
    """Every rank is its own process, given its rank, under trtllm-llmapi-launch and plain trtllm-serve alike.

    Rank 0's worker runs in its server's session, every other rank outside the MPI job, with the arguments rank 0
    published in the rendezvous store; the deployment serves.
    """
    deployment = request.getfixturevalue(launch)
    workers = _rank_workers(deployment.secret)
    assert sorted(workers) == list(range(deployment.world_size))
    for rank, process in workers.items():
        environ = process.environ()
        given_rank, world_size, _, replacement = environ[FAULT_TOLERANCE_RANK_ENV].split(",")
        assert (given_rank, world_size, replacement) == (str(rank), str(deployment.world_size), "0")
        if rank != 0:
            assert not any(name in environ for name in RANK_ENV)
    with Run(deployment, tmp_path, f"independent_ranks_{launch}"):
        pass


@pytest.mark.skip_less_device(WORLD_SIZE)
@pytest.mark.timeout(STARTUP_TIMEOUT_S + 1200)
def test_healthy_baseline(baseline_deployment: Deployment, tmp_path: Path):
    """The same load and measurements without fault tolerance (main, fence path).

    For comparison with the healthy phase of the other tests.
    """
    with Run(baseline_deployment, tmp_path, "baseline"):
        pass
