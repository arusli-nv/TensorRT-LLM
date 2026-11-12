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
"""End-to-end tests of Wide-EP fault tolerance with DeepSeek serving under load while ranks fail and rejoin.

Each test starts DeepSeek-V3-Lite with attention DP and EP over 4 GPUs under ``mpirun --enable-recovery``, or
attaches to an external deployment, on the node of the ranks it fails, with these variables:

- ``TLLM_FT_E2E_SERVER`` is the server's base URL.
- ``TLLM_FT_E2E_SERVER_LOGS`` is a glob pattern of the files the deployment's ranks log to.
- ``TLLM_FT_E2E_VICTIM`` is the rank to fail, which must run on this node.
- ``TLLM_FT_E2E_SEQUENCE`` is the comma-separated list of failures that :func:`test_rank_failure_sequence` runs.
  The default is ``crash,hang``. Use ``crash,node`` on several nodes.
- ``TLLM_FAULT_TOLERANCE_SHARED_SECRET`` identifies the deployment's processes.
- ``TLLM_FT_E2E_REPORT`` receives the phases and cycles as JSON, by default in the test's tmp_path.
- ``TLLM_FT_E2E_CONFIG`` (JSON) is merged into the configuration of a deployment that a test starts.

Load and GSM8K clients run throughout. After a ``healthy`` phase, each failure adds the phases ``failover`` (the
requests in flight at the failure), ``degraded``, ``admission`` (in flight when the replacement is admitted) and
``rejoined``.
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
    RENDEZVOUS_ENV,
    SHARED_SECRET_ENV,
    Authority,
    open_liveness_connection,
    open_store,
)
from tensorrt_llm._torch.pyexecutor.rank_failure_recovery import FaultToleranceHealth
from tensorrt_llm._wide_ep_fault_tolerance import FAULT_TOLERANCE_RANK_ENV
from tensorrt_llm.executor.utils import RANK_FAILURE_CODE
from tensorrt_llm.llmapi.mpi_session import split_mpi_env

from .conftest import llm_models_root

WORLD_SIZE = 4
VICTIM = 2
STARTUP_TIMEOUT_S = 1800
STATUS_PATH = "/fault_tolerance_status"
# The endpoint refreshes once a second, so the test times membership events by reading the store itself.
STORE_POLL_S = 0.01
EXTERNAL = bool(os.environ.get("TLLM_FT_E2E_SERVER"))
# Fault tolerance runs the NVLink one-sided all-to-all on its fence path.
FENCE_PATH_ENV = {"TRTLLM_NVLINK_ONE_SIDED_A2A_FORCE_CFT": "0"}

# Closed-loop load. Half of the clients stream, and half of each group samples with a per-request seed.
CLIENTS_PER_RANK = 16
MAX_CLIENTS = 256
LOAD_INPUT_LEN = (128, 512)
LOAD_MAX_TOKENS = 512
SAMPLING = {"temperature": 0.8, "top_p": 0.95}
REQUEST_TIMEOUT_S = 900

# GSM8K with main's lm-eval task on the first EVAL_QUESTIONS questions. A window phase lasts until every question
# is answered in it, at most EVAL_COVER_EXTRA_S past WINDOW_S.
EVAL_CLIENTS_PER_RANK = 4
MAX_EVAL_CLIENTS = 64
EVAL_QUESTIONS = 256
EVAL_COVER_EXTRA_S = 120
# A phase must not lose paired accuracy against the healthy phase: a one-sided exact sign test (McNemar) at
# ACCURACY_ALPHA, Bonferroni-corrected by the number of later phases.
ACCURACY_FILTER = "strict-match"
ACCURACY_ALPHA = 0.05

WARMUP_S = 20
SETTLE_S = 10
WINDOW_S = 60
# A stall is a gap of this long without a streamed token while the load runs.
STALL_S = 5.0

# Pause targets for the failover after a crash or a hang, and while a replacement is admitted.
CRASH_PAUSE_TARGET_S = 5.0
HANG_PAUSE_MARGIN_S = 5.0
HANG_TIMEOUT_S = inspect.signature(Authority).parameters["hang_timeout_s"].default
ADMISSION_PAUSE_TARGET_S = 3.0
RECOVERY_TIMEOUT_S = 120
REJOIN_TIMEOUT_S = 1200
EXCLUDED_EXIT_TIMEOUT_S = 60
INVALID_JOIN_WATCH_S = 20
REFUSAL_LOG = "Closing a liveness connection: "
RETRYABLE_LOG = re.compile(
    r"Rank 0 resumed serving with membership generation (\d+),.* (\d+) requests no active rank holds ended by "
    r"the rank failure"
)

# A rank logs this per graph shape when it captures, which only the warmup may do (CUDAGraphRunner.allow_capture).
CAPTURE_LINE = re.compile(r"\[RANK (\d+)\].*(CUDA graph capture|Capturing \d+ graphs)")
RANK_ENV = ("SLURM_PROCID", "OMPI_COMM_WORLD_RANK", "PMI_RANK", "PMIX_RANK")
GPU_DEVICE = re.compile(r"/dev/nvidia\d+")
HEALTH_STATES = {"healthy", "recovering", "degraded"}


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("", 0))
        return sock.getsockname()[1]


def _error_body(error: urllib.error.HTTPError) -> bytes:
    """Return the body of an HTTP error response, or empty bytes if it breaks off."""
    try:
        return error.read()
    except (OSError, http.client.HTTPException):
        return b""


def _get_json(url: str, timeout_s: float = 5.0) -> Tuple[Optional[int], Optional[dict]]:
    """Return the HTTP status and JSON body, or (None, None) if the server does not answer completely."""
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
    """Return the deployment's status, or None if the endpoint does not answer."""
    code, body = _get_json(f"{base}{STATUS_PATH}")
    return body if code == 200 else None


@dataclass
class Deployment:
    base: str
    model: str
    logs: str  # glob pattern of the files the ranks log to
    secret: str
    world_size: int
    victim: int


def _serving_config(model: Path) -> dict:
    num_experts = json.loads((model / "config.json").read_text())["n_routed_experts"]
    config = {
        "enable_wide_ep_fault_tolerance": True,
        "enable_attention_dp": True,
        "disable_overlap_scheduler": False,
        "cuda_graph_config": {"enable_padding": True, "max_batch_size": 32},
        "kv_cache_config": {"free_gpu_memory_fraction": 0.5, "use_kv_cache_manager_v2": True},
        "speculative_config": {"decoding_type": "MTP", "num_nextn_predict_layers": 1},
        # Static EPLB with one slot per expert. Fault tolerance adds the spare slots that the survivors need to hold
        # every expert, and restores a failed rank's experts into them from host weights.
        "moe_config": {"load_balancer": {"num_slots": num_experts}},
    }
    return _merged(config, json.loads(os.environ.get("TLLM_FT_E2E_CONFIG", "{}")))


def _merged(config: dict, overrides: dict) -> dict:
    """Return ``config`` with ``overrides`` merged in, recursing into nested mappings."""
    merged = dict(config)
    for name, value in overrides.items():
        both_dicts = isinstance(value, dict) and isinstance(merged.get(name), dict)
        merged[name] = _merged(merged[name], value) if both_dicts else value
    return merged


def _wait_until_healthy(base: str, alive: Callable[[], bool]) -> None:
    deadline = time.monotonic() + STARTUP_TIMEOUT_S
    while True:
        code, _ = _get_json(f"{base}/health")
        # The state comes from the status endpoint because /health does not report it.
        if code == 200 and (_status(base) or {}).get("state") == "healthy":
            return
        assert alive(), "the server exited during startup"
        assert time.monotonic() < deadline, "the server did not become healthy"
        time.sleep(5)


def _chosen_secret(since: float) -> str:
    """Return the shared secret that a server spawning its ranks chose, read from its node supervisor."""
    for process in psutil.process_iter(["cmdline", "create_time"]):
        with contextlib.suppress(psutil.AccessDenied, psutil.NoSuchProcess, psutil.ZombieProcess):
            cmdline = process.info["cmdline"] or []
            if process.info["create_time"] >= since and any(
                arg.endswith("_llmapi_node_supervisor.py") for arg in cmdline
            ):
                return process.environ()[SHARED_SECRET_ENV]
    raise LookupError("no node supervisor of the deployment runs on this node")


@contextlib.contextmanager
def _deploy(tmp_path: Path, server_spawns_ranks: bool = False):
    """Start a deployment, or attach to the one at ``TLLM_FT_E2E_SERVER``.

    With ``server_spawns_ranks``, plain ``trtllm-serve`` spawns its ranks. A started deployment must leave no /dev/shm
    segment behind.
    """
    if EXTERNAL:
        base = os.environ["TLLM_FT_E2E_SERVER"]
        _wait_until_healthy(base, lambda: True)
        status = _status(base)
        assert status is not None, f"{base}{STATUS_PATH} does not answer"
        yield Deployment(
            base,
            _get_json(f"{base}/v1/models")[1]["data"][0]["id"],
            os.environ["TLLM_FT_E2E_SERVER_LOGS"],
            os.environ[SHARED_SECRET_ENV],
            len(status["active_ranks"]),
            int(os.environ.get("TLLM_FT_E2E_VICTIM", VICTIM)),
        )
        return

    model = Path(llm_models_root()) / "DeepSeek-V3-Lite" / "bf16"
    if not model.exists():
        pytest.skip(f"{model} does not exist")
    config = tmp_path / "config.yaml"
    config.write_text(yaml.safe_dump(_serving_config(model)))
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
    # Set info level on every rank so that each rank logs its CUDA graph captures. Only the leader takes the
    # server's level.
    env = {**os.environ, **FENCE_PATH_ENV, "TLLM_LOG_LEVEL": "info"}
    secret = None
    if server_spawns_ranks:
        # A single process outside any MPI or Slurm job, as a user starts it.
        env = {**split_mpi_env()[0], **FENCE_PATH_ENV, "TLLM_LOG_LEVEL": "info"}
        if "PMIX_HOSTNAME" in os.environ:
            env["PMIX_HOSTNAME"] = os.environ["PMIX_HOSTNAME"]
        command = serve
    else:
        # Under trtllm-llmapi-launch the server does not start the workers. They take the shared secret and the
        # rendezvous address on rank 0's host from the launch. The launcher runs once on this node, and its node
        # supervisor starts the node's ranks. --enable-recovery keeps the launcher running when a rank exits.
        secret = os.urandom(32).hex()
        env[SHARED_SECRET_ENV] = secret
        env[RENDEZVOUS_ENV] = f"{socket.gethostname()}:{_free_port()}"
        command = ["mpirun", "--enable-recovery", "--allow-run-as-root", "-np", "1"]
        command += ["trtllm-llmapi-launch", *serve]
    launched_s = time.time()
    with open(log, "w") as log_file:
        launcher = subprocess.Popen(
            command, env=env, stdout=log_file, stderr=subprocess.STDOUT, start_new_session=True
        )
    try:
        _wait_until_healthy(base, lambda: launcher.poll() is None)
        if server_spawns_ranks:
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
        psutil.wait_procs(list(_node_processes(secret)), timeout=60)
    left = sorted(set(os.listdir("/dev/shm")) - shm_before)
    assert not left, f"/dev/shm segments the deployment left after it ended: {left}"


@pytest.fixture
def deployment(tmp_path: Path):
    with _deploy(tmp_path) as deployment:
        yield deployment


@pytest.fixture
def spawned_deployment(tmp_path: Path):
    """Start the deployment with plain ``trtllm-serve``, which spawns the ranks and chooses the secrets."""
    with _deploy(tmp_path, server_spawns_ranks=True) as deployment:
        yield deployment


def _holds_gpu(pid: int) -> bool:
    try:
        return any(
            GPU_DEVICE.fullmatch(os.readlink(f"/proc/{pid}/fd/{fd}"))
            for fd in os.listdir(f"/proc/{pid}/fd")
        )
    except OSError:
        return False


def _node_processes(secret: str) -> Dict[psutil.Process, Dict[str, str]]:
    """Return the environment of each process of the deployment on this node, other than this test's own."""
    test = psutil.Process()
    own = {test.pid, *(p.pid for p in test.parents())}
    found = {}
    for process in psutil.process_iter():
        if process.pid in own:
            continue
        with contextlib.suppress(psutil.AccessDenied, psutil.NoSuchProcess, psutil.ZombieProcess):
            env = process.environ()
            if env.get(SHARED_SECRET_ENV) == secret:
                found[process] = env
    return found


def _process_rank(env: Dict[str, str]) -> Optional[int]:
    """Return the rank that a process runs, from its identity if the node supervisor started it."""
    if FAULT_TOLERANCE_RANK_ENV in env:
        return int(env[FAULT_TOLERANCE_RANK_ENV].split(",")[0])
    rank = next((env[name] for name in RANK_ENV if name in env), None)
    return None if rank is None else int(rank)


def _rank_workers(secret: str) -> Dict[int, psutil.Process]:
    """Return each rank's outermost process on this node that holds a GPU."""
    candidates = {}
    for process, env in _node_processes(secret).items():
        rank = _process_rank(env)
        if rank is not None and _holds_gpu(process.pid):
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


def _rendezvous(secret: str) -> str:
    """Return the rendezvous store address from a process of the deployment on this node."""
    for env in _node_processes(secret).values():
        if RENDEZVOUS_ENV in env:
            return env[RENDEZVOUS_ENV]
    raise LookupError("no process of the deployment on this node has the rendezvous address")


def _log_lines(logs: str):
    """Yield every line of the deployment's logs, given as a glob pattern."""
    for log in sorted(glob.glob(logs)):
        with open(log, errors="replace") as lines:
            yield from lines


def _capture_lines(logs: str) -> Dict[int, int]:
    """Count the CUDA graph capture lines of each rank in the deployment's logs."""
    counts: Dict[int, int] = {}
    for match in filter(None, map(CAPTURE_LINE.search, _log_lines(logs))):
        rank = int(match.group(1))
        counts[rank] = counts.get(rank, 0) + 1
    return counts


def _retryable_failures_logged(logs: str) -> int:
    """Count the requests that rank 0 logged as failed with a retryable error over all membership changes."""
    per_generation = {}
    for match in filter(None, map(RETRYABLE_LOG.search, _log_lines(logs))):
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
    """Return the indices of the GPUs that each rank's worker runs on, from nvidia-smi's compute processes."""
    index = {uuid: int(i) for i, uuid in _nvidia_smi("gpu", "index,uuid")}
    gpus: Dict[int, List[int]] = {}
    for pid, uuid in _nvidia_smi("compute-apps", "pid,gpu_uuid"):
        gpus.setdefault(int(pid), []).append(index[uuid])
    return {r: sorted(gpus.get(p.pid, [])) for r, p in sorted(workers.items())}


def _prompt(seed: int, length: int) -> List[int]:
    rng = random.Random(seed)
    return [rng.randrange(1000, 30000) for _ in range(length)]


class Gsm8k:
    """GSM8K questions and their scoring with the lm-eval task ``tensorrt_llm.evaluate.GSM8K``."""

    def __init__(self, num_questions: int):
        from tensorrt_llm.evaluate import GSM8K

        path = Path(llm_models_root()) / "datasets" / "openai" / "gsm8k"
        assert path.exists(), f"GSM8K is not available offline at {path}"
        self._task = GSM8K(dataset_path=str(path), random_seed=0).task_dict["gsm8k"]
        self._task.build_all_requests(limit=num_questions, rank=0, world_size=1)
        self.instances = self._task.instances

    def request(self, question: int) -> dict:
        """Return the completion request for a question with its few-shot context and generation arguments."""
        context, gen_kwargs = self.instances[question].args
        return {
            "prompt": context,
            "max_tokens": gen_kwargs.get("max_gen_toks", 256),
            "temperature": gen_kwargs.get("temperature", 0.0),
            "stop": gen_kwargs.get("until", []),
        }

    def score(self, question: int, text: str) -> float:
        """Return the exact match of one answer with ACCURACY_FILTER, as ``_RunningScoreTracker.update`` does."""
        probe = copy.copy(self.instances[question])
        probe.resps = [text]
        probe.filtered_resps = {}
        (ensemble,) = [e for e in self._task._filters if e.name == ACCURACY_FILTER]
        ensemble.apply([probe])
        metrics = self._task.process_results(probe.doc, [probe.filtered_resps[ensemble.name]])
        return float(metrics["exact_match"])


@dataclass
class Request:
    stream: bool
    start: float
    question: Optional[int] = None  # an eval request's GSM8K question
    retry: bool = False  # sent again after a retryable failure
    end: float = 0.0
    # (arrival time, token count) per streamed chunk with tokens, or one entry when a non-streamed response arrives.
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
    """Return a daemon thread that runs ``target`` and appends any exception that ends it to ``failures``."""

    def run() -> None:
        try:
            target(*args)
        except Exception:  # re-raised, recorded so that the run fails on it
            failures.append(f"{threading.current_thread().name} ended: {traceback.format_exc()}")
            raise

    return threading.Thread(target=run, daemon=True)


def _join(threads: List[threading.Thread], timeout_s: float, failures: List[str]) -> List[str]:
    """Join ``threads`` within ``timeout_s`` in total and return ``failures`` plus one per thread still running."""
    deadline = time.monotonic() + timeout_s
    for thread in threads:
        thread.join(max(0.0, deadline - time.monotonic()))
    return failures + [
        f"{thread.name} still runs {timeout_s} s after it was told to stop"
        for thread in threads
        if thread.is_alive()
    ]


class Clients:
    """Closed-loop load and eval clients, as the module docstring describes."""

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
        """Stop sending, wait for the requests in flight, and return the clients that raised or still run."""
        self._stop.set()
        return _join(self._threads, REQUEST_TIMEOUT_S, self._failures)

    def restart_questions(self) -> None:
        """Restart the eval clients from the first question of the subset."""
        with self._lock:
            self._question = 0

    def answered(self, since: float) -> set:
        """Return the questions answered by requests that started at ``since`` or later."""
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
        """Send a request, and send it once more if it fails with a retryable error."""
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
            with contextlib.suppress(ValueError, AttributeError):
                request.error_code = json.loads(body).get("code")
        except (OSError, http.client.HTTPException, ValueError, LookupError, TypeError) as error:
            # No answer, a response that broke off (IncompleteRead, BadStatusLine), or one that is not a completion.
            request.error = repr(error)
        finally:
            # Record the request even when another exception ends the client. Clients.stop fails the run then.
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

    def throughput(self, start: float, end: float) -> float:
        """Return the output tokens per second that all clients received in [start, end)."""
        tokens = sum(n for r in self.requests for t, n in r.arrivals if start <= t < end)
        return round(tokens / max(end - start, 1e-3), 1)


def _largest_gap(times: List[float], start: float, end: float) -> Tuple[float, float]:
    """Return the length and start of the longest interval from [start, end] to the next arrival."""
    inside = [start] + [t for t in times if start < t < end]
    later = [t for t in times if t >= end]
    inside.append(later[0] if later else end)
    return max((b - a, a) for a, b in zip(inside, inside[1:]))


class Monitor:
    """Sample ``/health`` with the status state, and record each membership change read from the store."""

    def __init__(self, deployment: Deployment):
        self._base = deployment.base
        self._stop = threading.Event()
        self._changed = threading.Condition()
        self._failures: List[str] = []
        self.health: List[Tuple[float, Optional[int], Optional[str]]] = []
        self.membership: List[Tuple[float, dict]] = []
        reader = FaultToleranceHealth(
            _rendezvous(deployment.secret), bytes.fromhex(deployment.secret)
        )
        self._threads = [
            _thread(self._sample_health, (), self._failures),
            _thread(self._sample_membership, (reader,), self._failures),
        ]

    def start(self) -> None:
        for thread in self._threads:
            thread.start()

    def stop(self) -> List[str]:
        """Stop watching and return the samplers that raised or still run."""
        self._stop.set()
        return _join(self._threads, 60.0, self._failures)

    def _sample_health(self) -> None:
        while not self._stop.is_set():
            code, _ = _get_json(f"{self._base}/health")
            status = _status(self._base)
            self.health.append(
                (time.monotonic(), code, None if status is None else status["state"])
            )
            self._stop.wait(0.25)

    def _sample_membership(self, reader: FaultToleranceHealth) -> None:
        while not self._stop.is_set():
            status = reader.status()
            # The public status leaves out the record's own state, which the checks need to time kills in a change.
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
        """Return the time and the first membership status after ``since`` that satisfies ``predicate``."""
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
        """Return when ``/health`` first answered ``code`` with the status ``state`` after ``since``."""
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

    def committed_generations(self) -> set:
        """Return the generations of the membership transactions that were seen to end."""
        return {
            status["last_transaction"]["generation"]
            for _, status in self.membership
            if status["last_transaction"] is not None
        }


def _all_active(world: int) -> Callable[[dict], bool]:
    return lambda s: s["membership_state"] == "COMMITTED" and s["active_ranks"] == list(
        range(world)
    )


def _committed_after(generation: int) -> Callable[[dict], bool]:
    return lambda s: s["membership_state"] == "COMMITTED" and s["generation"] > generation


def _admitting(
    ranks: List[int], after_generation: int, states: Tuple[str, ...] = ("PROPOSE", "INSTALL")
) -> Callable[[dict], bool]:
    """Match a membership change after ``after_generation`` that adds ``ranks`` and is in one of ``states``."""
    return lambda s: (
        s["membership_state"] in states
        and s["generation"] > after_generation
        and set(ranks) <= set(s["active_ranks"])
    )


def _wait_for_replacement(
    secret: str, rank: int, replaced_pid: int, timeout_s: float
) -> psutil.Process:
    """Wait for and return the worker process of ``rank`` that replaces ``replaced_pid``."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        process = _rank_workers(secret).get(rank)
        if process is not None and process.pid != replaced_pid:
            return process
        time.sleep(1)
    raise AssertionError(f"no replacement of rank {rank} started within {timeout_s} s")


def _victims(deployment: Deployment, failure: str, workers: Dict[int, psutil.Process]):
    """Return the ranks to fail on this node, which is every rank for a ``node`` failure."""
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
    """Kill (``crash``) or stop (``hang``) the victims, or kill every process of the deployment on this node.

    Returns the processes a ``node`` failure killed as "<pid> <name>" strings.
    """
    if failure != "node":
        for process in victims.values():
            if failure == "hang":
                process.suspend()
            else:
                process.kill()
        return []
    killed = []
    # Each round stops every process before it kills any, so that no supervisor sees a rank end and starts a
    # replacement. A later round catches a process started in the meantime.
    for _ in range(3):
        processes = list(_node_processes(secret))
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
    # For a phase that spans an event, the phase's requests are those in flight at that time.
    event: Optional[float] = None

    def covers(self, request: Request) -> bool:
        if self.event is not None:
            return request.spans(self.event)
        return self.start <= request.start and request.end <= self.end


class Run:
    """Run the load, the monitor and a sequence of failures on one deployment, then report and check."""

    def __init__(self, deployment: Deployment, tmp_path: Path, name: str):
        self.deployment = deployment
        self.name = name
        self.report_path = Path(os.environ.get("TLLM_FT_E2E_REPORT", tmp_path / "report.json"))
        self.gsm8k = Gsm8k(EVAL_QUESTIONS)
        self.monitor = Monitor(deployment)
        self.clients = Clients(deployment, self.gsm8k)
        self.phases: List[Phase] = []
        self.cycles: List[dict] = []
        # Times of rank failures and invalid joins. Only requests in flight at these times may fail.
        self.failure_times: List[float] = []
        self.failures: List[str] = []
        self.t0 = time.monotonic()
        self.healthy: Optional[Phase] = None
        # The status and each rank's CUDA graph capture lines after warmup.
        self.status_after_warmup: Optional[dict] = None
        self.captures_after_warmup: Dict[int, int] = {}

    def check(self, ok: bool, message: str) -> None:
        if not ok:
            self.failures.append(message)

    def __enter__(self) -> "Run":
        d = self.deployment
        status = _status(d.base)
        assert status["active_ranks"] == list(range(d.world_size)), status
        assert not status["failed_ranks"], status
        self.status_after_warmup = status
        self.captures_after_warmup = _capture_lines(d.logs)
        self.t0 = time.monotonic()
        self.monitor.start()
        self.clients.start()
        try:
            time.sleep(WARMUP_S)
            self.healthy = self.window("healthy", 0)
        except BaseException:
            self.clients.stop()
            self.monitor.stop()
            raise
        return self

    def __exit__(self, *exc) -> None:
        for failure in self.clients.stop() + self.monitor.stop():
            self.check(False, failure)
        failed = exc[0] is not None
        phases = self._phase_reports()
        if not failed:
            self._check_requests()
            self._check_accuracy(phases)
            self._check_captures()
            states = {str(s[2]) for s in self.monitor.health}
            self.check(states <= HEALTH_STATES, f"states during the run: {sorted(states)}")
            retryable = sum(1 for r in self.clients.requests if r.retryable)
            logged = _retryable_failures_logged(self.deployment.logs)
            self.check(
                retryable <= logged,
                f"{retryable} requests failed as retryable, but rank 0 failed {logged} as retryable",
            )
        report = {
            "test": self.name,
            "world_size": self.deployment.world_size,
            "phases": phases,
            "cycles": self.cycles,
            "check_failures": self.failures,
        }
        self.report_path.write_text(json.dumps(report, indent=2))
        print(json.dumps(report, indent=2))
        if not failed:
            assert not self.failures, "\n".join(self.failures) + f"\nreport: {self.report_path}"

    def window(self, name: str, cycle: int) -> Phase:
        """Serve for WINDOW_S and then until every eval question is answered, at most EVAL_COVER_EXTRA_S more."""
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
        """Fail ranks with ``crash``, ``hang`` or ``node``, wait for N ranks again, and serve a window.

        ``at_admission`` returns processes to kill once the store shows the admission in ``at_admission_state``.
        """
        d, index = self.deployment, len(self.cycles) + 1
        world, all_ranks = d.world_size, list(range(d.world_size))
        status = _status(d.base)
        assert status["active_ranks"] == all_ranks, status
        assert status["failed_ranks"] == [], status
        workers_before = _rank_workers(d.secret)  # scanning the processes takes seconds
        gpus_before = _rank_gpus(workers_before)
        captures_before = _capture_lines(d.logs)
        victims = _victims(d, failure, workers_before)
        failed_ranks = sorted(victims)
        record = {"cycle": index, "failure": failure, "failed_ranks": failed_ranks}
        self.cycles.append(record)

        t_fail = time.monotonic()
        killed = _inject(victims, failure, d.secret)
        self.failure_times.append(t_fail)
        if failure == "node":
            left = [p.pid for p in _node_processes(d.secret)]
            message = f"the node's processes were not all killed: killed {killed}, left {left}"
            assert {p.pid for p in victims.values()} <= {int(k.split()[0]) for k in killed}, message
            assert not left, message
        detection_s = HANG_TIMEOUT_S if failure == "hang" else 0.0
        t_degraded, failover_status = self.monitor.wait_for(
            _committed_after(status["generation"]), t_fail, detection_s + RECOVERY_TIMEOUT_S
        )
        record["time_to_degraded_s"] = round(t_degraded - t_fail, 3)
        self.monitor.wait_for_health(200, "degraded", t_degraded, RECOVERY_TIMEOUT_S)
        self.phases.append(Phase("failover", index, t_fail, t_degraded, event=t_fail))
        time.sleep(SETTLE_S)
        t_settled = time.monotonic()
        self.clients.restart_questions()
        if failure == "hang":
            # The excluded process runs again under load. It must exit without disturbing the survivors.
            (process,) = victims.values()
            with contextlib.suppress(psutil.NoSuchProcess):
                process.resume()
                try:
                    process.wait(EXCLUDED_EXIT_TIMEOUT_S)
                except psutil.TimeoutExpired:
                    self.check(
                        False,
                        f"cycle {index}: the excluded process still runs {EXCLUDED_EXIT_TIMEOUT_S} s after it "
                        "was resumed",
                    )

        killed_at_admission: List[int] = []
        if at_admission is not None:
            processes = at_admission()
            killed_at_admission = sorted(processes)
            # Kill them as soon as the store shows the admission, at its first PROPOSE or in ``at_admission_state``.
            states = ("PROPOSE", "INSTALL") if at_admission_state is None else (at_admission_state,)
            t_admitting, admitting = self.monitor.wait_for(
                _admitting(failed_ranks, failover_status["generation"], states),
                t_degraded,
                REJOIN_TIMEOUT_S,
            )
            for process in processes.values():
                process.kill()
            t_kill = time.monotonic()
            self.failure_times.append(t_kill)
            # The kill must land before the admission commits. The admission's generation then never commits, and
            # the next committed record leaves the killed ranks out.
            t_after_kill, after_kill = self.monitor.wait_for(
                _committed_after(admitting["generation"]), t_kill, RECOVERY_TIMEOUT_S
            )
            self.phases.append(
                Phase("admission_failure", index, t_kill, t_after_kill, event=t_kill)
            )
            record["admission_failure"] = {
                "killed_ranks": killed_at_admission,
                "status_at_kill": admitting,
            }
            self.check(
                admitting["generation"] not in self.monitor.committed_generations()
                and not set(killed_at_admission) & set(after_kill["active_ranks"]),
                f"cycle {index}: the kill did not land during the admission: {after_kill}",
            )

        t_rejoined, rejoined_status = self.monitor.wait_for(
            _all_active(world), t_degraded, REJOIN_TIMEOUT_S
        )
        self.monitor.wait_for_health(200, "healthy", t_rejoined, RECOVERY_TIMEOUT_S)
        admission = rejoined_status["last_transaction"]
        # The commit is read within STORE_POLL_S, and the authority measures the transaction's duration.
        t_admission = t_rejoined - (admission["ended_unix_ms"] - admission["began_unix_ms"]) / 1e3
        record["time_to_n_s"] = round(t_rejoined - t_fail, 3)
        degraded_end = t_admission - 1.0
        self.phases.append(Phase("degraded", index, t_settled, degraded_end))
        self.phases.append(Phase("admission", index, t_admission, t_rejoined, event=t_admission))
        time.sleep(SETTLE_S)
        rejoined = self.window("rejoined", index)
        # Streamed tokens must not stall between the commit at N ranks and the window's end.
        stall_s = _largest_gap(self.clients.stream_arrivals(), t_rejoined, rejoined.end)[0]
        self.check(
            stall_s <= STALL_S,
            f"cycle {index}: serving stalled after the admission: no streamed token for {stall_s:.1f} s",
        )

        workers_after = _rank_workers(d.secret)
        gpus_after = _rank_gpus(workers_after)
        captures_after = _capture_lines(d.logs)
        record["cuda_graph_capture_lines"] = {"before": captures_before, "after": captures_after}
        survivors = [r for r in all_ranks if r not in failed_ranks]
        local_survivors = [
            r for r in workers_before if r not in failed_ranks and r not in killed_at_admission
        ]

        # A pause is the longest time without a streamed token, at the failure and during the admission.
        arrivals = self.clients.stream_arrivals()
        pause_end = (degraded_end + 1.0 if at_admission is None else t_admitting) - 1.0
        pause_s = _largest_gap(arrivals, t_fail - 1.0, pause_end)[0]
        record["pause_s"] = round(pause_s, 3)
        if at_admission is not None:
            kill_pause_s = _largest_gap(arrivals, t_admitting - 1.0, t_after_kill + SETTLE_S)[0]
            self.check(
                kill_pause_s < CRASH_PAUSE_TARGET_S,
                f"cycle {index}: pause {kill_pause_s:.2f} s at the kill during the admission, target < "
                f"{CRASH_PAUSE_TARGET_S} s",
            )
        pause_target_s = (
            HANG_TIMEOUT_S + HANG_PAUSE_MARGIN_S if failure == "hang" else CRASH_PAUSE_TARGET_S
        )
        self.check(
            pause_s <= pause_target_s if failure == "hang" else pause_s < pause_target_s,
            f"cycle {index}: pause {pause_s:.2f} s, target {pause_target_s} s",
        )
        spanning = [r for r in self.clients.load_requests() if r.spans(t_fail)]
        for stream in (False, True):
            self.check(
                any(r.stream == stream and r.error is None for r in spanning),
                f"cycle {index}: no {'streamed' if stream else 'non-streamed'} request in flight at the failure "
                "completed",
            )
        # Requests sent again after a retryable failure at this cycle's failures.
        retries = [
            r for r in self.clients.requests if r.retry and t_fail <= r.start <= degraded_end
        ]
        self.check(
            any(r.error is None for r in retries),
            f"cycle {index}: none of the {len(retries)} requests sent again after a retryable failure succeeded",
        )
        self.check(
            failover_status["generation"] > status["generation"]
            and failover_status["failed_ranks"] == failed_ranks
            and failover_status["active_ranks"] == survivors
            and failover_status["authority_alive"],
            f"cycle {index}: status after the failover: {failover_status}",
        )
        # No survivor restarted. Run._check_captures checks that none captured a CUDA graph after startup.
        for rank in local_survivors:
            self.check(
                rank in workers_after and workers_after[rank].pid == workers_before[rank].pid,
                f"cycle {index}: survivor rank {rank} was restarted",
            )
        # Reported only. The standard serving benchmark measures performance.
        record["throughput_tok_s"] = {
            "healthy": self.clients.throughput(self.healthy.start, self.healthy.end),
            "degraded": self.clients.throughput(t_settled, degraded_end),
        }
        # The rejoin.
        admission_pause_s = _largest_gap(arrivals, t_admission - 1.0, t_rejoined + SETTLE_S)[0]
        record["admission_pause_s"] = round(admission_pause_s, 3)
        record["throughput_tok_s"]["rejoined"] = self.clients.throughput(
            rejoined.start, rejoined.end
        )
        self.check(
            rejoined_status["failed_ranks"] == []
            and rejoined_status["generation"] > failover_status["generation"],
            f"cycle {index}: status after the rejoin: {rejoined_status}",
        )
        # A survivor keeps its incarnation. A replacement takes the next incarnation, and a replacement killed during
        # its admission is replaced once more.
        expected = [
            n + (r in failed_ranks) + (r in killed_at_admission)
            for r, n in enumerate(status["incarnations"])
        ]
        self.check(
            rejoined_status["incarnations"] == expected,
            f"cycle {index}: incarnations {status['incarnations']} -> {rejoined_status['incarnations']}, "
            f"expected {expected}",
        )
        # A replacement runs on the GPU of the process it replaces.
        for rank in sorted((set(failed_ranks) | set(killed_at_admission)) & set(workers_before)):
            self.check(
                bool(gpus_before[rank]) and gpus_after.get(rank) == gpus_before[rank],
                f"cycle {index}: rank {rank}'s replacement runs on GPUs {gpus_after.get(rank)}, the process it "
                f"replaces ran on {gpus_before[rank]}",
            )
        # A replacement captures its own CUDA graphs at startup.
        for rank in sorted(set(failed_ranks) | set(killed_at_admission)):
            self.check(
                captures_after.get(rank, 0) > captures_before.get(rank, 0),
                f"cycle {index}: no CUDA graph capture lines of rank {rank}'s replacement",
            )
        self.check(
            admission_pause_s < ADMISSION_PAUSE_TARGET_S,
            f"cycle {index}: admission pause {admission_pause_s:.2f} s, target < {ADMISSION_PAUSE_TARGET_S} s",
        )
        return record

    @contextlib.contextmanager
    def refused(self, rank: int, replacement: bool, reason: str):
        """Join as ``rank``, as its first process or a ``replacement``, and yield the time.

        The connection stays open until the context ends. The authority must log a refusal with ``reason``.
        """
        d, line = self.deployment, REFUSAL_LOG + reason

        def logged() -> int:
            return sum(line in log_line for log_line in _log_lines(d.logs))

        before = logged()
        store = open_store(_rendezvous(d.secret), bytes.fromhex(d.secret), timeout_s=60.0)
        t = time.monotonic()
        with open_liveness_connection(store, rank, replacement):
            yield t
        self.check(
            logged() > before,
            f"the authority logged no refusal of a hello for rank {rank} (replacement {replacement}): {line!r}",
        )

    def _phase_reports(self) -> List[dict]:
        """Return each phase's output tokens per second and its GSM8K answers paired with the healthy phase's."""
        requests = self.clients.requests
        scores = {
            i: self.gsm8k.score(r.question, r.text)
            for i, r in enumerate(requests)
            if r.question is not None and r.error is None
        }
        reports, healthy = [], None
        alpha = ACCURACY_ALPHA / max(1, len(self.phases) - 1)
        for phase in self.phases:
            # Score each question by its first answer in the phase.
            answers: Dict[int, float] = {}
            for i in sorted(scores, key=lambda i: requests[i].start):
                if phase.covers(requests[i]):
                    answers.setdefault(requests[i].question, scores[i])
            gsm8k = {"questions": len(answers)}
            if healthy is None:
                healthy = answers
            else:
                paired = set(answers) & set(healthy)
                lost = sum(1 for q in paired if healthy[q] == 1.0 and answers[q] == 0.0)
                gained = sum(1 for q in paired if healthy[q] == 0.0 and answers[q] == 1.0)
                gsm8k.update(
                    {
                        "paired": len(paired),
                        "right_when_healthy_wrong_here": lost,
                        "wrong_when_healthy_right_here": gained,
                        "p_loss": _sign_test_p(lost, gained),
                        "alpha": alpha,
                    }
                )
            if answers:
                gsm8k["accuracy"] = round(statistics.mean(answers.values()), 4)
            reports.append(
                {
                    "phase": phase.name,
                    "cycle": phase.cycle,
                    "output_tok_s": self.clients.throughput(phase.start, phase.end),
                    "gsm8k": gsm8k,
                }
            )
        return reports

    def _check_requests(self) -> None:
        """Check that every request completed in full, but streams in flight at a failure that end as retryable."""
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
        """Check that the healthy phase answered every question and that no other phase lost paired accuracy."""
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

    def _check_captures(self) -> None:
        """Check that every rank captured CUDA graphs at startup and that no original process captured any since."""
        d = self.deployment
        start, end = self.status_after_warmup["incarnations"], _status(d.base)
        first_processes = [r for r in end["active_ranks"] if end["incarnations"][r] == start[r]]
        before, after = self.captures_after_warmup, _capture_lines(d.logs)
        for rank in range(d.world_size):
            self.check(before.get(rank, 0) > 0, f"no CUDA graph capture lines of rank {rank}")
        for rank in first_processes:
            self.check(
                after.get(rank, 0) == before.get(rank, 0),
                f"rank {rank} captured CUDA graphs after warmup: {before.get(rank, 0)} -> "
                f"{after.get(rank, 0)} capture lines",
            )


def _sign_test_p(lost: int, gained: int) -> float:
    """Return the p-value of a one-sided exact sign test (McNemar on discordant pairs) for a loss."""
    n = lost + gained
    return sum(math.comb(n, k) for k in range(lost, n + 1)) / 2**n


@pytest.mark.skip_less_device(WORLD_SIZE)
@pytest.mark.timeout(STARTUP_TIMEOUT_S + 2 * REJOIN_TIMEOUT_S + 1200)
def test_rank_failure_sequence(deployment: Deployment, tmp_path: Path):
    """Run the failures of ``TLLM_FT_E2E_SEQUENCE`` under load, each followed by a rejoin.

    ``none`` serves one more healthy window instead, which measures the noise of the paired comparison.
    """
    with Run(deployment, tmp_path, "sequence") as run:
        sequence = os.environ.get("TLLM_FT_E2E_SEQUENCE", "crash,hang").split(",")
        for index, failure in enumerate(sequence, start=1):
            if failure == "none":
                run.window("healthy_again", index)
            else:
                run.cycle(failure)


def _refused_join(run: Run, kind: str, replacement: bool, reason: str) -> None:
    """Send a join for the victim's rank that the authority refuses, and check that nothing changes."""
    d = run.deployment
    workers, before = _rank_workers(d.secret), _status(d.base)
    with run.refused(d.victim, replacement, reason) as t_join:
        time.sleep(INVALID_JOIN_WATCH_S)
    run.phases.append(Phase(f"{kind}_join", len(run.cycles), t_join, time.monotonic()))
    after, after_workers = _status(d.base), _rank_workers(d.secret)
    run.check(
        after["generation"] == before["generation"]
        and after["active_ranks"] == list(range(d.world_size)),
        f"a {kind} join for rank {d.victim} changed the membership: {before} -> {after}",
    )
    run.check(
        {r: p.pid for r, p in after_workers.items()} == {r: p.pid for r, p in workers.items()},
        f"a process restarted after a {kind} join",
    )


@pytest.mark.skip_less_device(WORLD_SIZE)
@pytest.mark.timeout(STARTUP_TIMEOUT_S + 3 * REJOIN_TIMEOUT_S + 1200)
def test_invalid_joins(deployment: Deployment, tmp_path: Path):
    """Send joins that the authority refuses, and kill a replacement during its admission, under load.

    The joins are a second connection of an active rank, a replaced first process, and a replacement of an active rank.
    """
    d = deployment
    with Run(d, tmp_path, "invalid_joins") as run:
        with run.refused(d.victim, False, f"rank {d.victim}'s first process is connected already"):
            run.cycle("crash")
        _refused_join(run, "stale", False, f"rank {d.victim}'s first process was replaced")
        _refused_join(
            run, "duplicate", True, f"rank {d.victim} is active: no replacement joins for it"
        )

        victim_pid = _rank_workers(d.secret)[d.victim].pid
        run.cycle(
            "crash",
            at_admission=lambda: {
                d.victim: _wait_for_replacement(d.secret, d.victim, victim_pid, REJOIN_TIMEOUT_S)
            },
        )


def _second_rank(d: Deployment, workers: Dict[int, psutil.Process]) -> int:
    """Return a rank on this node, other than rank 0 and the victim, to fail during the victim's admission."""
    second = min((r for r in workers if r not in (0, d.victim)), default=None)
    assert second is not None, (
        f"no second rank to fail on this node (ranks here: {sorted(workers)})"
    )
    return second


@pytest.mark.skip_less_device(WORLD_SIZE)
@pytest.mark.timeout(STARTUP_TIMEOUT_S + 2 * REJOIN_TIMEOUT_S + 1200)
def test_failure_during_admission(deployment: Deployment, tmp_path: Path):
    """Crash a second rank while the first one's replacement is admitted."""
    d = deployment
    with Run(d, tmp_path, "failure_during_admission") as run:
        workers = _rank_workers(d.secret)
        second = _second_rank(d, workers)
        run.cycle("crash", at_admission=lambda: {second: workers[second]})


@pytest.mark.skip_less_device(WORLD_SIZE)
@pytest.mark.timeout(STARTUP_TIMEOUT_S + 2 * REJOIN_TIMEOUT_S + 1200)
def test_failure_during_install(deployment: Deployment, tmp_path: Path):
    """Crash a second rank while the active ranks install the first one's admission."""
    d = deployment
    with Run(d, tmp_path, "failure_during_install") as run:
        workers = _rank_workers(d.secret)
        second = _second_rank(d, workers)
        run.cycle(
            "crash", at_admission=lambda: {second: workers[second]}, at_admission_state="INSTALL"
        )


@pytest.mark.skip_less_device(WORLD_SIZE)
@pytest.mark.skipif(EXTERNAL, reason="starts its own deployment")
@pytest.mark.timeout(STARTUP_TIMEOUT_S + REJOIN_TIMEOUT_S + 1200)
def test_rejoin_when_server_spawns_ranks(spawned_deployment: Deployment, tmp_path: Path):
    """Fail over and rejoin a killed rank under plain trtllm-serve, which chooses the shared secret itself."""
    with Run(spawned_deployment, tmp_path, "server_spawns_ranks") as run:
        run.cycle("crash")
