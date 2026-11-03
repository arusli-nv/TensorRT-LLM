"""Benchmark driver for one trtllm-serve case (runs on node 0, next to ranks 0-3).

Every load runs TensorRT-LLM's benchmark_serving client in its own process: random tokens, exactly 1k in
and 1k out, --ignore-eos, a fixed concurrency (load per GPU x serving GPUs). Measured cases are not
profiled; profile cases (rank.sh) capture a fixed window of their own server.

Modes:
- sweep: every load of --loads, then the first again (drift check). With --after-loss: then kill rank 2
  while idle, wait for all survivors to resume, and sweep again on 31 GPUs.
- failover: run one load and kill rank 2 during it (--kill idle|random|decode|prefill); for FT on, wait
  for every survivor to resume and time a new request's first token; for FT off, the job aborts.
- restart: time the server start and a new request's first token (the restart after an FT-off abort).
- stress (review/PRODUCT_STRESS_PLAN.md): mixed lengths (--isl/--osl with --range-ratio), /health sampled every
  --health-period-s; a closed-loop 32/GPU row (calib-32), then Poisson arrivals at --stress-rate (or --load-fraction of
  calib-32's request rate) for --stress-minutes through streamed completions and chat completions (stress_client.py
  records each stream's events), the --stress-kill failures on the way, and a closed-loop row again at the end.
- accuracy: GSM8K through the server (accuracy_client.py); with --kill-during-eval the victim kills itself in a
  decode step while the evaluation's requests are in flight; with --kill-before-eval it kills itself in a decode step
  of a short warm load, and the evaluation runs once every survivor resumed and the load finished.
Writes RUN/result.json after every step and RUN/frontend.done at the end, which ends the case.
"""

import argparse
import ast
import json
import os
import random
import signal
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

import steady_state

MODEL = os.environ.get("WIDEEP_MODEL", "/lustre/fs1/portfolios/coreai/projects/coreai_comparch_trtllm/common/DeepSeek-R1-0528-FP4-V2")
# A directory with only the tokenizer files, for models whose config.json the container's transformers rejects
# (GLM-5.2: layer_types "deepseek_sparse_attention").
TOKENIZER = os.environ.get("WIDEEP_TOKENIZER", MODEL)
WORLD = int(os.environ.get("WIDEEP_WORLD", "32"))
VICTIM = int(os.environ.get("WIDEEP_VICTIM", "2"))  # a node-0 rank (1-3): the driver finds and kills it locally
LOST = int(os.environ.get("WIDEEP_LOST", "1"))  # ranks lost at once (4: a node whose ranks kill themselves)


class HarnessInvalid(RuntimeError):
    """The case did not exercise what it was meant to (for example the kill missed the load)."""
# Most concurrent streams one benchmark client process runs (256/GPU on 32 GPUs ran clean in one process);
# larger loads are split over several.
CLIENT_MAX_CONCURRENCY = int(os.environ.get("WIDEEP_CLIENT_MAX_CONCURRENCY", "8192"))
# How long a full-load decode may take to ramp before a profile window gives up (512/GPU prefills for minutes).
DECODE_LOAD_TIMEOUT_S = float(os.environ.get("WIDEEP_DECODE_LOAD_TIMEOUT_S", "900"))
# A client error that is a reset connection with no token received: the server's accept queue dropped it (16k
# streams at 512/GPU), not a request the server failed.
RESET = "Connection reset by peer"


def merge_parts(path: Path, parts: list) -> None:
    """Write at ``path`` the benchmark result of client processes that ran one load together.

    Per-request lists are concatenated and every metric is recomputed from them as benchmark_serving computes
    it; the parts ran concurrently, so throughput is over the longest part's duration.
    """
    import numpy as np

    results = [json.loads(part.read_text()) for part in parts if part.exists()]
    if not results:
        return
    merged = dict(results[0])
    for key in ("input_lens", "output_lens", "ttfts", "itls", "e2els", "generated_texts", "errors"):
        merged[key] = [value for result in results for value in result.get(key, [])]
    for key in ("num_prompts", "completed", "total_input_tokens", "total_output_tokens", "max_concurrency"):
        merged[key] = sum(result.get(key) or 0 for result in results)
    duration = max(result["duration"] for result in results)
    merged["duration"] = duration
    merged["request_throughput"] = merged["completed"] / duration
    merged["output_throughput"] = merged["total_output_tokens"] / duration
    merged["total_token_throughput"] = (merged["total_input_tokens"] + merged["total_output_tokens"]) / duration
    ok = [i for i, error in enumerate(merged["errors"]) if not error]
    series = {
        "ttft": [merged["ttfts"][i] for i in ok],
        "tpot": [(merged["e2els"][i] - merged["ttfts"][i]) / (merged["output_lens"][i] - 1)
                 for i in ok if merged["output_lens"][i] > 1],
        "itl": [itl for i in ok for itl in merged["itls"][i]],
        "e2el": [merged["e2els"][i] for i in ok],
    }
    for name, values in series.items():
        ms = np.array(values or [0.0]) * 1000
        merged.update({f"mean_{name}_ms": float(ms.mean()), f"median_{name}_ms": float(np.median(ms)),
                       f"std_{name}_ms": float(ms.std()), f"p99_{name}_ms": float(np.percentile(ms, 99))})
    path.write_text(json.dumps(merged))


def healthy(port: int) -> bool:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=5) as response:
            return response.status == 200
    except OSError:
        return False


def fault_tolerance_status(port: int) -> dict:
    """The server's FT status body and /health status code. Builds from f10bee1a0f on carry the body on /health;
    older ones on /fault_tolerance (absent on builds without either)."""
    status = {}
    for path in ("health", "fault_tolerance"):
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/{path}", timeout=10) as response:
                code, raw = response.status, response.read()
        except urllib.error.HTTPError as error:
            code, raw = error.code, error.read()
        except (OSError, ValueError) as error:
            status.setdefault(path, repr(error))
            continue
        try:
            body = json.loads(raw or b"null")
        except ValueError:
            body = None
        if path == "health":
            status["health"] = code
            if isinstance(body, dict) and "state" in body:
                status["fault_tolerance"] = body
        elif code != 404:
            status["fault_tolerance"] = body
        if isinstance(status.get("fault_tolerance"), dict) and len(status["fault_tolerance"]) > 1:
            break
    return status


def first_token_s(port: int) -> float:
    """Seconds until the first streamed token of one new short request."""
    body = json.dumps({"model": MODEL, "prompt": "The capital of France is", "max_tokens": 8,
                       "stream": True}).encode()
    request = urllib.request.Request(f"http://127.0.0.1:{port}/v1/completions", data=body,
                                     headers={"Content-Type": "application/json"})
    start = time.time()
    with urllib.request.urlopen(request, timeout=120) as response:
        for line in response:
            if line.startswith(b"data:") and b'"text"' in line:
                return time.time() - start
    raise RuntimeError("no token streamed")


def find_worker(rank: int) -> int:
    """Pid of the local worker process of MPI ``rank`` (ranks 0-3 run on node 0)."""
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            cmdline = (entry / "cmdline").read_bytes()
            environ = (entry / "environ").read_bytes().split(b"\0")
        except OSError:
            continue
        if b"mgmn_worker_node" in cmdline and f"OMPI_COMM_WORLD_RANK={rank}".encode() in environ:
            return int(entry.name)
    raise LookupError(f"no local worker process for rank {rank}")


class Driver:
    def __init__(self, args):
        self.args = args
        self.run = args.run
        (self.run / "bench").mkdir(parents=True, exist_ok=True)
        self.result = dict(mode=args.mode, ft=args.ft, rows=[])

    def save(self) -> None:
        (self.run / "result.json").write_text(json.dumps(self.result, indent=1) + "\n")

    def client(self, label: str, load: int, gpus: int, iterations: int, osl: int = 1024, open_loop: tuple = None) -> list:
        """Start the benchmark clients of a load: one per CLIENT_MAX_CONCURRENCY streams, sharing it evenly.

        At 16k concurrent streams one client process lost requests to connection resets in every mode.
        open_loop=(rate, prompts, backend, seed): one client sending `prompts` requests at `rate` per second with no
        concurrency limit (stress mode).
        """
        concurrency = load * gpus
        parts = 1 if open_loop else -(-concurrency // CLIENT_MAX_CONCURRENCY)
        clients = []
        for part in range(parts):
            name = label if parts == 1 else f"{label}.part{part}"
            # WIDEEP_TIMED_CLIENT=1: the same client, also writing each request's tokens per wall second (timed_client.py).
            timed = os.environ.get("WIDEEP_TIMED_CLIENT") == "1"
            runner = ([str(Path(__file__).with_name("timed_client.py")), str(self.run / "bench" / f"{name}.tokens.jsonl")]
                      if timed else ["-m", "tensorrt_llm.serve.scripts.benchmark_serving"])
            # Stress mode: every stream's events recorded (stress_client.py).
            if self.args.mode == "stress":
                runner = [str(Path(__file__).with_name("stress_client.py")), str(self.run / "bench" / f"{name}.sse.jsonl")]
            rate, prompts, backend, seed = open_loop or (self.args.request_rate / parts, concurrency * iterations // parts,
                                                         "openai", part)
            cmd = [sys.executable, *runner,
                   "--model", MODEL, "--tokenizer", TOKENIZER, "--backend", backend, "--port", str(self.args.port),
                   *(["--endpoint", "/v1/chat/completions"] if backend == "openai-chat" else []),
                   "--dataset-name", "random", "--tokenize-on-client",
                   *(["--dataset-path", self.args.text_prompts] if self.args.text_prompts else ["--random-ids"]),
                   "--random-input-len", str(self.args.isl), "--random-output-len", str(osl),
                   "--random-range-ratio", str(self.args.range_ratio),
                   "--num-prompts", str(prompts),
                   *([] if open_loop else ["--max-concurrency", str(concurrency // parts)]), "--seed", str(seed),
                   "--request-rate", str(rate), "--burstiness", str(self.args.burstiness),
                   "--ignore-eos", "--percentile-metrics", "ttft,tpot,itl,e2el", "--save-result",
                   *(["--no-test-input"] if open_loop else ["--save-detailed"]),
                   "--result-dir", str(self.run / "bench"), "--result-filename", f"{name}.json"]
            log = open(self.run / "bench" / f"{name}.log", "w")
            clients.append(subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT))
        return clients

    def finish(self, label: str, clients: list, start: float, **row) -> dict:
        # The clients have no timeout of their own: stop them if the server is gone.
        while any(client.poll() is None for client in clients):
            if (self.run / "done").exists():
                for client in clients:
                    client.kill()
            time.sleep(1)
        rc = max(client.wait() for client in clients)
        row.update(label=label, start_wall_s=start, elapsed_s=time.time() - start, rc=rc)
        path = self.run / "bench" / f"{label}.json"
        if len(clients) > 1:
            merge_parts(path, [path.with_name(f"{label}.part{part}.json") for part in range(len(clients))])
        if path.exists():
            metrics = json.loads(path.read_text())
            row.update({k: metrics.get(k) for k in (
                "completed", "duration", "output_throughput", "total_token_throughput", "request_throughput",
                "median_ttft_ms", "p99_ttft_ms", "median_tpot_ms", "p99_tpot_ms", "median_itl_ms",
                "p99_itl_ms", "median_e2el_ms", "p99_e2el_ms")})
            # Without --save-detailed (stress mode's open-loop rows) only stress_client.py's records tell.
            row["failed_requests"] = sum(1 for e in metrics["errors"] if e) if "errors" in metrics else None
            row["reset_requests"] = sum(1 for e, n in zip(metrics.get("errors", []), metrics.get("output_lens", []))
                                        if e and RESET in e and n == 0)
            # A request the server fails mid-stream ends the stream normally, so the client reports no error; with
            # --ignore-eos every served request returns exactly osl tokens, so a shorter one was cut off (with mixed
            # lengths only stress_client.py's records tell).
            row["short_requests"] = None if self.args.range_ratio else sum(
                1 for n in metrics.get("output_lens", []) if n < row["osl"])
        self.result["rows"].append(row)
        self.save()
        print(json.dumps(row), flush=True)
        if rc != 0:
            raise RuntimeError(f"benchmark client failed at {label}; see bench/{label}.log")
        return row

    def load(self, label: str, load: int, gpus: int, iterations: int, osl: int = 1024):
        start = time.time()
        client = self.client(label, load, gpus, iterations, osl)
        if self.args.profile_first_load_steps and not self.result.get("profile_first_load_done"):
            # EV5/EV6: the slow steps cluster in the first seconds of the first load, so cover them from its
            # first full-load decode step (on the warm row).
            self.result["profile_first_load_done"] = True
            threading.Thread(target=self.profile_first_load, args=(label, load), daemon=True).start()
        if self.args.profile_on_slow_step and not label.startswith("warm"):
            threading.Thread(target=self.profile_slow_episode, args=(label, load), daemon=True).start()
        if self.args.profile_after_s and not label.startswith(("warm", "burnin")):
            # Profile cases: a window of steady decode on the nsys ranks (cudaProfilerApi capture), opened only
            # once rank 0 decodes the whole load (the client first runs a single test prompt).
            time.sleep(self.args.profile_after_s)
            self.wait_decode_load(load)
            # WIDEEP_PROFILE_STEPS: a longer window also catches the context steps that interleave with decode.
            body = json.dumps({"activities": ["CUDA_PROFILER"], "num_steps": int(os.environ.get("WIDEEP_PROFILE_STEPS", "50"))}).encode()
            request = urllib.request.Request(f"http://127.0.0.1:{self.args.port}/start_profile", data=body,
                                             headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(request, timeout=120) as response:
                self.result.setdefault("profile_windows", []).append(
                    dict(label=label, wall_s=time.time(), status=response.status))
        return self.finish(label, client, start, load_per_gpu=load, gpus=gpus,
                           concurrency=load * gpus, iterations=iterations, osl=osl)

    def profile_first_load(self, label: str, load: int) -> None:
        self.wait_decode_load(load, steps=1)
        body = json.dumps({"activities": ["CUDA_PROFILER"], "num_steps": self.args.profile_first_load_steps}).encode()
        request = urllib.request.Request(f"http://127.0.0.1:{self.args.port}/start_profile", data=body,
                                         headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=120) as response:
            self.result.setdefault("profile_windows", []).append(
                dict(label=label, wall_s=time.time(), status=response.status, first_load=True))

    def profile_slow_episode(self, label: str, load: int, baseline_steps: int = 100, window_steps: int = 600) -> None:
        """EV6: open one nsys window when a decode step at full load takes profile_on_slow_step times the median
        of the first baseline_steps such steps (rank-0.log), so the window covers a slow-step episode."""
        log = self.run / "rank-0.log"
        offset = log.stat().st_size if log.exists() else 0
        times = []
        while not self.result.get("profile_windows_slow_done"):
            time.sleep(0.5)
            with open(log, "rb") as lines:
                lines.seek(offset)
                chunk = lines.read()
            offset += chunk.rfind(b"\n") + 1
            for line in chunk[: chunk.rfind(b"\n") + 1].decode(errors="replace").splitlines():
                match = steady_state.LINE.search(line)
                if not match:
                    continue
                states = ast.literal_eval(match.group(4))
                if states.get("num_ctx_requests", 0) or states.get("num_generation_tokens", 0) < 0.9 * load:
                    continue
                step_ms = float(match.group(2))
                if len(times) < baseline_steps:
                    times.append(step_ms)
                    continue
                median = sorted(times)[len(times) // 2]
                if step_ms < self.args.profile_on_slow_step * median:
                    continue
                body = json.dumps({"activities": ["CUDA_PROFILER"], "num_steps": window_steps}).encode()
                request = urllib.request.Request(f"http://127.0.0.1:{self.args.port}/start_profile", data=body,
                                                 headers={"Content-Type": "application/json"})
                with urllib.request.urlopen(request, timeout=120) as response:
                    self.result.setdefault("profile_windows", []).append(
                        dict(label=label, wall_s=time.time(), status=response.status, trigger_step_ms=step_ms,
                             median_step_ms=median, iteration=int(match.group(1))))
                self.result["profile_windows_slow_done"] = True
                return

    def wait_decode_load(self, load: int, steps: int = 5, timeout_s: float = DECODE_LOAD_TIMEOUT_S,
                         consecutive: bool = True) -> None:
        """Wait until rank 0 logs `steps` decode-only iterations with at least 90% of `load` tokens, consecutive or in
        total since the call."""
        log, offset, streak = self.run / "rank-0.log", 0, 0
        if log.exists():
            offset = log.stat().st_size
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            with open(log, "rb") as lines:
                lines.seek(offset)
                chunk = lines.read()
            offset += chunk.rfind(b"\n") + 1
            for line in chunk[: chunk.rfind(b"\n") + 1].decode(errors="replace").splitlines():
                match = steady_state.LINE.search(line)
                if match:
                    states = ast.literal_eval(match.group(4))
                    full = states.get("num_ctx_requests", 0) == 0 and states.get("num_generation_tokens", 0) >= 0.9 * load
                    streak = streak + 1 if full else (0 if consecutive else streak)
                    if streak >= steps:
                        return
            time.sleep(0.5)
        raise HarnessInvalid(f"rank 0 never decoded {load} tokens per step within {timeout_s:.0f} s; no profile")

    def wait_ready(self) -> None:
        launched = time.time()
        while not healthy(self.args.port):
            timeline = self.run / "timeline"
            if (self.run / "done").exists() or "RANK_EXIT" in (timeline.read_text() if timeline.exists() else ""):
                raise RuntimeError("a rank exited before the server became healthy")
            if time.time() - launched > self.args.ready_timeout_s:
                raise TimeoutError("server not healthy")
            time.sleep(1)
        self.result["healthy_wall_s"] = time.time()
        self.result["ready_s"] = time.time() - launched
        (self.run / "server_ready").touch()
        self.save()

    def warmup(self, gpus: int, load: int) -> None:
        """The CUDA graphs and prefill shapes of the coming loads, short outputs; not measured."""
        self.load(f"warmup-{gpus}", load, gpus, 1, osl=64)

    def warm_loads(self) -> None:
        """--warm-loads: one unmeasured wave per listed per-GPU concurrency, all requests at once and short outputs, so
        each rank prefills that many contexts in one step (the context-token buckets the steady load skips)."""
        log = self.run / "rank-0.log"
        for load in self.args.warm_loads:
            offset = log.stat().st_size if log.exists() else 0
            self.load(f"warm-{load}", load, WORLD, 1, osl=8)
            with open(log, "rb") as lines:
                lines.seek(offset)
                chunk = lines.read().decode(errors="replace")
            # Rank 0's local tokens (context + generation) of each step with a context request: the buckets it reached.
            steps = [ast.literal_eval(m.group(4)) for m in steady_state.LINE.finditer(chunk)]
            self.result.setdefault("warm_loads", []).append(dict(load_per_gpu=load, r0_ctx_step_tokens=sorted({
                s["num_ctx_tokens"] + s.get("num_generation_tokens", 0) for s in steps if s.get("num_ctx_requests")})))
            self.save()

    def require_load_running(self, client: list) -> None:
        """A kill under load must land while the load runs; otherwise the case tests nothing."""
        if any(one.poll() is not None for one in client):
            raise HarnessInvalid("the load ended before the kill; shorten --kill-window-s or lengthen the load")

    def wait_self_kill(self, client: list, timeout_s: float = 60) -> None:
        """The victim kills, stalls or faults itself at its next decode replay / prefill forward; it must do so
        under load."""
        log = self.run / f"rank-{VICTIM}.log"
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            text = log.read_bytes() if log.exists() else b""
            if any(marker in text for marker in (b"WIDEEP_SELF_KILL", b"WIDEEP_SELF_STALL", b"WIDEEP_SELF_FAULT")):
                return
            self.require_load_running(client)
            time.sleep(0.2)
        raise HarnessInvalid(f"rank {VICTIM} did not kill itself within {timeout_s:.0f} s of the trigger")

    def kill_victim(self) -> None:
        os.kill(find_worker(VICTIM), signal.SIGKILL)
        self.result["kill_wall_s"] = time.time()
        self.save()

    def wait_resumed(self) -> None:
        """Wait until the authority committed this case's next recovery and rank 0 resumed serving after it.

        Every survivor resumes after the same Commit and the collectives that follow it, so rank 0's "resumed serving"
        line marks them all. Reading only rank 0's log and the authority's keeps the 0.5 s poll off the other ranks'
        logs (on Lustre, a reader contends with a writer's locks; node-local logs are not visible here).
        """
        self.recoveries = getattr(self, "recoveries", 0) + 1
        generation = self.recoveries
        resumed = self._count_lines(self.run / "rank-0.log", b"resumed serving without rank")
        committed = self._count_lines(self.run / "authority.log", f"committed generation {generation} ".encode())
        deadline = time.time() + 300
        while next(resumed) < generation or next(committed) < 1:
            if time.time() > deadline or (self.run / "done").exists():
                raise RuntimeError(f"recovery {generation} did not complete")
            time.sleep(0.5)
        self.result["all_resumed_seen_wall_s" if generation == 1 else f"resumed{generation}_seen_wall_s"] = time.time()
        self.save()

    @staticmethod
    def _count_lines(path: Path, mark: bytes):
        """A generator of how many complete lines of ``path`` contain ``mark`` so far (reads only what was appended)."""
        offset, count = 0, 0
        while True:
            if path.exists():
                with open(path, "rb") as log:
                    log.seek(offset)
                    chunk = log.read()
                end = chunk.rfind(b"\n") + 1
                count += chunk[:end].count(mark)
                offset += end
            yield count

    def sweep(self, phase: str, gpus: int) -> None:
        loads = self.args.loads
        if self.args.warm_each_load:
            for load in loads:
                self.load(f"warm-{phase}-{load}", load, gpus, 1, osl=64)
                self.load(f"{phase}-{load}", load, gpus, self.iterations(load), osl=self.args.osl)
            return
        for i, load in enumerate([*loads, loads[0]]):
            label = f"{phase}-{load}" + ("-repeat" if i == len(loads) else "")
            self.load(label, load, gpus, self.iterations(load), osl=self.args.osl)

    def iterations(self, load: int) -> int:
        return min(self.args.iterations, self.args.load_iterations.get(load, self.args.iterations))

    def run_sweep(self) -> None:
        self.wait_ready()
        if self.args.profile_from_ready_steps:
            # The first steps after ready, near idle (the next load's client starts with a single test prompt).
            body = json.dumps({"activities": ["CUDA_PROFILER"], "num_steps": self.args.profile_from_ready_steps}).encode()
            request = urllib.request.Request(f"http://127.0.0.1:{self.args.port}/start_profile", data=body,
                                             headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(request, timeout=120) as response:
                self.result.setdefault("profile_windows", []).append(
                    dict(label="from-ready", wall_s=time.time(), status=response.status))
        if not self.args.warm_each_load:
            self.warmup(WORLD, max(self.args.loads))
        if self.args.burnin:
            # Whatever runs after startup lands here, not on the first measured load.
            load, waves = self.args.burnin
            self.load(f"burnin-{load}", load, WORLD, waves, osl=self.args.osl)
        self.sweep("healthy", WORLD)
        if self.args.after_loss:
            self.kill_victim()
            self.wait_resumed()
            self.warmup(WORLD - LOST, max(self.args.loads))
            self.sweep("after-loss", WORLD - LOST)

    def kill_during_eval(self, evaluation: subprocess.Popen) -> None:
        """Trigger the victim's decode self-kill --kill-window-s after the server answered the evaluation's first
        request (rank 0's access log), so the kill lands while the evaluation's requests are in flight."""
        log = self.run / "rank-0.log"
        offset = log.stat().st_size if log.exists() else 0
        while True:
            self.require_load_running([evaluation])
            with open(log, "rb") as lines:
                lines.seek(offset)
                chunk = lines.read()
            if b'"POST /v1/completions' in chunk:
                break
            offset += chunk.rfind(b"\n") + 1
            time.sleep(0.2)
        self.result["first_answer_wall_s"] = time.time()
        time.sleep(random.uniform(*self.args.kill_window_s))
        self.require_load_running([evaluation])
        Path(os.environ["TLLM_TEST_KILL_TRIGGER"]).touch()
        self.result["trigger_wall_s"] = time.time()
        self.save()
        self.wait_self_kill([evaluation])
        self.result["kill_wall_s"] = time.time()
        self.save()
        if self.args.ft:
            self.wait_resumed()

    def kill_before_eval(self) -> None:
        """The victim kills itself in a decode step of a warm load (--load per GPU, --iterations waves, 1k out)
        --kill-window-s after the load starts; returns once every survivor resumed and the load finished."""
        start = time.time()
        client = self.client("warm-kill", self.args.load, WORLD, self.args.iterations)
        time.sleep(random.uniform(*self.args.kill_window_s))
        # The client's single-prompt test run comes first and holds one request on rank 0 only (it can outlast the
        # window): trigger once rank 0 decodes the full load, so the victim holds requests in a decode step.
        self.wait_decode_load(self.args.load)
        self.require_load_running(client)
        Path(os.environ["TLLM_TEST_KILL_TRIGGER"]).touch()
        self.result["trigger_wall_s"] = time.time()
        self.save()
        self.wait_self_kill(client)
        self.result["kill_wall_s"] = time.time()
        self.save()
        self.wait_resumed()
        self.finish("warm-kill", client, start, load_per_gpu=self.args.load, gpus=WORLD,
                    concurrency=self.args.load * WORLD, iterations=self.args.iterations, osl=1024)

    def run_accuracy(self) -> None:
        """GSM8K through the server; with --after-loss, after losing the victim while idle; with --kill-before-eval,
        after losing it in a decode step under a warm load; with --kill-during-eval, while the evaluation runs."""
        self.wait_ready()
        if self.args.after_loss:
            self.kill_victim()
            self.wait_resumed()
        elif self.args.kill_before_eval:
            self.kill_before_eval()
        start = time.time()
        output = self.run / "bench" / "gsm8k.json"
        try:
            import lm_eval  # noqa: F401
        except ImportError:
            with open(self.run / "bench" / "lm_eval-install.log", "w") as log:
                subprocess.call([sys.executable, "-m", "pip", "install", "lm_eval[api]==0.4.10"], stdout=log,
                                stderr=subprocess.STDOUT)
        self.result["accuracy_start_wall_s"] = start
        # The evaluation's requests are the access-log lines rank 0 writes from here on.
        rank0_log = self.run / "rank-0.log"
        self.result["accuracy_log_offset"] = rank0_log.stat().st_size if rank0_log.exists() else 0
        with open(self.run / "bench" / "gsm8k.log", "w") as log:
            evaluation = subprocess.Popen([sys.executable, str(Path(__file__).with_name("accuracy_client.py")),
                                           str(self.args.port), MODEL, str(output), "--tokenizer", TOKENIZER,
                                           "--max-retries", str(self.args.accuracy_max_retries)],
                                          stdout=log, stderr=subprocess.STDOUT)
            try:
                if self.args.kill_during_eval:
                    self.kill_during_eval(evaluation)
                rc = evaluation.wait()
            finally:
                if evaluation.poll() is None:
                    evaluation.kill()
        if self.args.ft:
            self.result["fault_tolerance_status"] = fault_tolerance_status(self.args.port)
        self.result["accuracy"] = dict(rc=rc, elapsed_s=time.time() - start,
                                       **(json.loads(output.read_text()) if output.exists() else {}))
        self.save()
        if rc != 0:
            raise RuntimeError("accuracy client failed; see bench/gsm8k.log")

    def run_failover(self) -> None:
        self.wait_ready()
        if self.args.first_token_at_ready:
            # A cold restart's last milestone: the first request this (re)launched server answers.
            self.result["first_token_s"] = first_token_s(self.args.port)
            self.result["first_token_wall_s"] = time.time()
            self.save()
        self.warmup(WORLD, self.args.load)
        self.warm_loads()
        self.result["healthy_first_token_s"] = first_token_s(self.args.port)
        kind = self.args.kill
        if kind == "idle":
            self.kill_victim()
        else:
            # Load, then kill at a random moment of its steady state (decode/prefill: rank 2 kills itself
            # at its next decode replay / prefill forward once the trigger file exists).
            start = time.time()
            client = self.client("load", self.args.load, WORLD, self.args.iterations)
            if self.args.kill_after_steps:
                # A deterministic moment: the same number of full-load decode-only steps since the load started,
                # for every repeat, mode and arm (a prefill kill then lands in the next context step).
                self.wait_decode_load(self.args.load, steps=self.args.kill_after_steps, consecutive=False)
                self.result["kill_after_steps"] = self.args.kill_after_steps
            else:
                time.sleep(random.uniform(*self.args.kill_window_s))
            self.require_load_running(client)
            if self.args.kill_in_decode_steady and kind in ("decode", "stop") and not self.args.kill_after_steps:
                # Headline kills land in decode-only steps: wait for rank 0 to log consecutive decode-only steps
                # at the load, so the first step after recovery is a decode step, not a prefill burst.
                self.wait_decode_load(self.args.load)
                self.result["kill_in_decode_steady"] = True
            if kind == "random":
                self.kill_victim()
            elif kind == "control":
                # Negative control (hang tests): the trigger starts a legitimate slowdown (stall hooks); the load
                # must finish with no rank fenced or lost.
                Path(os.environ["TLLM_TEST_KILL_TRIGGER"]).touch()
                self.result["trigger_wall_s"] = time.time()
                self.finish("load", client, start, load_per_gpu=self.args.load, gpus=WORLD,
                            concurrency=self.args.load * WORLD, iterations=self.args.iterations, osl=1024)
                status = fault_tolerance_status(self.args.port)
                self.result["fault_tolerance_status"] = status
                body = status.get("fault_tolerance")
                if isinstance(body, dict) and body.get("recoveries"):
                    raise RuntimeError(f"negative control: a recovery ran ({body})")
                self.save()
                return
            elif kind == "freeze":
                # Negative control for the MPI in-band detector: the victim is frozen (SIGSTOP) for --freeze-s
                # seconds, then resumed (SIGCONT) while the load keeps running; the deployment must not lose it
                # unless the freeze outlasts the configured detection (the result records what happened).
                victim = find_worker(VICTIM)
                os.kill(victim, signal.SIGSTOP)
                self.result["freeze_wall_s"] = time.time()
                time.sleep(self.args.freeze_s)
                try:
                    os.kill(victim, signal.SIGCONT)
                    self.result["resume_wall_s"] = time.time()
                except ProcessLookupError:
                    self.result["victim_gone_during_freeze"] = True
                self.save()
                self.finish("load", client, start, load_per_gpu=self.args.load, gpus=WORLD,
                            concurrency=self.args.load * WORLD, iterations=self.args.iterations, osl=1024)
                self.result["fault_tolerance_status"] = fault_tolerance_status(self.args.port)
                self.save()
                return
            elif kind == "stop":
                os.kill(find_worker(VICTIM), signal.SIGSTOP)
                self.result["kill_wall_s"] = time.time()
                self.save()
            else:
                # The victim (on this node) polls a local file at every decode replay / prefill forward.
                Path(os.environ["TLLM_TEST_KILL_TRIGGER"]).touch()
                self.result["trigger_wall_s"] = time.time()
                self.save()
                self.wait_self_kill(client)
        if self.args.ft:
            self.wait_resumed()
            self.result["first_token_after_kill_s"] = first_token_s(self.args.port)
            status = fault_tolerance_status(self.args.port)
            self.result["fault_tolerance_status"] = status
            # A victim triggered at its next decode replay can be idle (attention DP runs dummy batches), so the
            # case then covers recovery but not the failing of in-flight requests: rank 0 counted none.
            body = status.get("fault_tolerance")
            if kind != "idle" and isinstance(body, dict) and body.get("failed_requests") == 0:
                self.result["victim_held_no_requests"] = True
                print("WARNING: the victim held no requests when it was lost", flush=True)
            self.save()
            if kind != "idle":
                self.finish("load", client, start, load_per_gpu=self.args.load, gpus=WORLD,
                            concurrency=self.args.load * WORLD, iterations=self.args.iterations, osl=1024)
        else:
            # Without FT the job aborts: wait until mpirun has exited (launch/task.sh writes done).
            while not (self.run / "done").exists():
                time.sleep(0.2)
            self.result["job_ended_seen_wall_s"] = time.time()
            if kind != "idle":
                self.count_aborted(client, start)

    def count_aborted(self, client: list, start: float) -> None:
        """FT off, after the abort: with --wait-clients-s, let the load's clients see the server gone and count the
        requests the abort failed: in flight (an error other than a refused connection, or a stream cut short) and
        unsent (refused: the server was already gone). Without it, stop the clients."""
        if self.args.wait_clients_s <= 0:
            for one in client:
                one.kill()
            return
        deadline = time.time() + self.args.wait_clients_s
        while time.time() < deadline and any(one.poll() is None for one in client):
            time.sleep(1)
        try:
            self.finish("load", client, start, load_per_gpu=self.args.load, gpus=WORLD,
                        concurrency=self.args.load * WORLD, iterations=self.args.iterations, osl=1024)
        except RuntimeError:  # the clients' rc after an abort (or after being stopped) is not a result
            pass
        path = self.run / "bench" / "load.json"
        if not path.exists():
            return
        metrics = json.loads(path.read_text())
        refused = inflight = 0
        for error, n in zip(metrics.get("errors", []), metrics.get("output_lens", [])):
            if error and ("ClientConnectorError" in error or "Connection refused" in error):
                refused += 1
            elif error or n < 1024:
                inflight += 1
        self.result.update(failed_inflight=inflight, failed_unsent=refused, completed_before_abort=metrics.get("completed"))
        self.save()

    def run_restart(self) -> None:
        self.wait_ready()
        self.result["first_token_s"] = first_token_s(self.args.port)
        self.result["first_token_wall_s"] = time.time()

    def run_stress(self) -> None:
        """Stress mode (module docstring). Each --stress-kill is KIND:SECONDS:TRIGGER:RANKS (KIND at = SECONDS after the
        open-loop load started, after = SECONDS after the previous recovery resumed; RANKS r or r+r+...): the driver
        creates TRIGGER, whose ranks kill themselves at their next decode step (testhooks TLLM_TEST_KILL_TRIGGER_FOR)."""
        args = self.args
        self.wait_ready()
        sys.path.insert(0, str(Path(__file__).with_name("draft_cases")))
        from ft_watch import Sampler  # health.jsonl / fault_tolerance.jsonl until frontend.done
        sampler = Sampler(self.run, args.port, args.health_period_s)
        sampler.start()
        self.warmup(WORLD, 32)
        calib = self.load("calib-32", 32, WORLD, args.calib_waves, osl=args.osl)
        rate = args.stress_rate or args.load_fraction * calib["request_throughput"]
        prompts = round(rate * args.stress_minutes * 60)
        chat = round(prompts * args.chat_fraction)
        mark = self.run / "bench" / "load_start"
        os.environ["STRESS_LOAD_START"] = str(mark)
        self.result.update(stress_rate=rate, stress_prompts=prompts, stress_chat_prompts=chat)
        self.save()
        start = time.time()
        main = self.client("stress", 32, WORLD, 0, args.osl, open_loop=(rate * (prompts - chat) / prompts,
                                                                        prompts - chat, "openai", 100))
        os.environ.pop("STRESS_LOAD_START")  # the main client's first request marks the load's start
        side = self.client("stress-chat", 32, WORLD, 0, args.osl,
                           open_loop=(rate * chat / prompts, chat, "openai-chat", 200)) if chat else []
        deadline = time.time() + 900
        while not mark.exists():  # the client's prompts are generated before its first request
            self.require_load_running(main)
            if time.time() > deadline:
                raise HarnessInvalid("the open-loop client sent no request within 900 s")
            time.sleep(1)
        load_start = previous = float(mark.read_text())
        self.result["load_start_wall_s"] = load_start
        self.result["kills"] = kills = []
        for spec in args.stress_kill:
            kind, seconds, trigger, ranks = spec.split(":")
            kill = dict(kind=kind, seconds=float(seconds), ranks=[int(r) for r in ranks.split("+")])
            kills.append(kill)
            target = (load_start if kind == "at" else previous) + float(seconds)
            while time.time() < target:
                self.require_load_running(main)
                time.sleep(1)
            Path(trigger).touch()
            kill["trigger_wall_s"] = time.time()
            self.save()
            kill["kill_wall_s"] = self.wait_ranks_killed(kill["ranks"], main)
            self.save()
            try:
                self.wait_resumed()
            except RuntimeError as error:  # e.g. a loss the remaining slots cannot cover: the job ends
                kill["recovery_error"] = repr(error)
                self.save()
                break
            previous = kill["resumed_seen_wall_s"] = time.time()
            kill["fault_tolerance_status"] = fault_tolerance_status(args.port)
            self.save()
        self.finish("stress", main, start, load_per_gpu=32, gpus=WORLD, rate=rate, iterations=0, osl=args.osl)
        if side:
            try:
                self.finish("stress-chat", side, start, load_per_gpu=32, gpus=WORLD, rate=rate * chat / prompts,
                            iterations=0, osl=args.osl)
            except RuntimeError as error:  # the chat client's failure is a finding, not the end of the case
                self.result["chat_client_error"] = repr(error)
        if args.calib_end_waves and not (self.run / "done").exists():
            self.load("calib-end-32", 32, WORLD, args.calib_end_waves, osl=args.osl)
        if args.ft:
            self.result["fault_tolerance_status"] = fault_tolerance_status(args.port)
        self.save()
        print("SAMPLER_ROWS health", len(sampler.health), (self.run / "health.jsonl").stat().st_size, "status",
              len(sampler.status), (self.run / "fault_tolerance.jsonl").stat().st_size, flush=True)

    def wait_ranks_killed(self, ranks: list, client: list, timeout_s: float = 120) -> float:
        """The wall time (from the victims' own WIDEEP_SELF_KILL lines) at which all of ``ranks`` killed themselves."""
        deadline, seen = time.time() + timeout_s, {}
        while len(seen) < len(ranks):
            for rank in ranks:
                log = self.run / f"rank-{rank}.log"
                text = log.read_bytes() if rank not in seen and log.exists() else b""
                if b"WIDEEP_SELF_KILL" in text:
                    line = text[text.index(b"WIDEEP_SELF_KILL"):].split(b"\n", 1)[0]
                    stamp = line.split(b" t=")[-1].split()[0] if b" t=" in line else b""
                    seen[rank] = float(stamp) if stamp.replace(b".", b"").isdigit() else time.time()
            if time.time() > deadline:
                raise HarnessInvalid(f"ranks {sorted(set(ranks) - set(seen))} did not kill themselves within {timeout_s:.0f} s")
            self.require_load_running(client)
            time.sleep(0.5)
        return max(seen.values())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--request-rate", type=float, default=float("inf"),
                        help="new requests per second over all clients; a finite rate opens connections gradually "
                             "(16k at once overflow the server's listen backlog)")
    parser.add_argument("--burstiness", type=float, default=1.0,
                        help="arrival regularity at a finite --request-rate: 1 is Poisson, larger is more regular")
    parser.add_argument("--ft", action="store_true", help="the server runs with fault tolerance")
    parser.add_argument("--text-prompts", default=None,
                        help="a ShareGPT-format JSON: prompts are real text (each conversation's first turn, repeated "
                             "or cut to exactly 1k tokens) instead of random token ids, so expert routing and draft "
                             "acceptance see real traffic")
    parser.add_argument("--mode", choices=("sweep", "failover", "restart", "accuracy", "stress"), default="sweep")
    parser.add_argument("--loads", type=lambda s: [int(n) for n in s.split(",")],
                        default=[8, 32, 64, 128, 256, 512], help="concurrent requests per GPU")
    parser.add_argument("--iterations", type=int, default=5, help="requests per load = iterations x concurrency")
    parser.add_argument("--load-iterations", type=lambda s: {int(k): int(v) for k, v in (p.split("=") for p in s.split(",") if p)},
                        default=os.environ.get("WIDEEP_LOAD_ITERATIONS", "512=2"),
                        help="sweep: most --iterations at a load, LOAD=N,... (a 512/GPU wave runs ~2.5 min)")
    parser.add_argument("--isl", type=int, default=1024, help="input tokens per request")
    parser.add_argument("--osl", type=int, default=1024, help="output tokens per request")
    parser.add_argument("--range-ratio", type=float, default=0.0,
                        help="input and output lengths uniform in [len x (1 - r), len x (1 + r)] (benchmark_serving)")
    parser.add_argument("--stress-rate", type=float, default=0.0,
                        help="stress: open-loop requests per second (0: --load-fraction x calib-32's request rate)")
    parser.add_argument("--load-fraction", type=float, default=0.8, help="stress: see --stress-rate")
    parser.add_argument("--stress-minutes", type=float, default=60.0, help="stress: open-loop arrivals last this long")
    parser.add_argument("--chat-fraction", type=float, default=0.1,
                        help="stress: share of the open-loop requests sent to /v1/chat/completions")
    parser.add_argument("--calib-waves", type=int, default=6,
                        help="stress: requests of the closed-loop rows = waves x 32/GPU x 32 GPUs")
    parser.add_argument("--calib-end-waves", type=int, default=6,
                        help="stress: the closed-loop row after the open-loop load (calib-end-32; 0: none)")
    parser.add_argument("--health-period-s", type=float, default=0.1, help="stress: /health sampling period")
    parser.add_argument("--stress-kill", action="append", default=[], metavar="KIND:SECONDS:TRIGGER:RANKS",
                        help="stress: a failure (run_stress); repeat for more")
    parser.add_argument("--burnin", type=lambda s: tuple(int(x) for x in s.split(":")),
                        default=tuple(int(x) for x in os.environ["WIDEEP_BURNIN"].split(":"))
                        if os.environ.get("WIDEEP_BURNIN") else None,
                        help="sweep: LOAD:WAVES, one unmeasured row (label burnin-LOAD) before the measured loads")
    parser.add_argument("--warm-each-load", action="store_true",
                        help="sweep: before each measured load, one unmeasured wave at its concurrency (short "
                             "outputs), and no repeat of the first load at the end")
    parser.add_argument("--after-loss", action="store_true",
                        help="sweep: then lose rank 2 and sweep on 31 GPUs; accuracy: evaluate after losing rank 2")
    parser.add_argument("--kill", choices=("idle", "random", "decode", "prefill", "stop", "control", "freeze"), default="random",
                        help="stop: SIGSTOP the victim under load (a hang the authority must confirm and fence)")
    parser.add_argument("--load", type=int, default=256, help="failover: concurrent requests per GPU")
    parser.add_argument("--warm-loads", type=lambda s: [int(n) for n in s.split(",") if n], default=[],
                        help="failover: after the warm-up, one wave (osl 8) per listed per-GPU concurrency, e.g. 3,5")
    parser.add_argument("--kill-after-steps", type=int, default=0, metavar="STEPS",
                        help="failover: trigger the failure once rank 0 has logged STEPS full-load decode-only steps "
                             "since the load started (instead of a random moment of --kill-window-s)")
    parser.add_argument("--freeze-s", type=float, default=10.0, help="--kill freeze: seconds the victim stays stopped")
    parser.add_argument("--kill-window-s", type=lambda s: tuple(float(x) for x in s.split(",")),
                        default=(15.0, 45.0),
                        help="failover: kill at a uniformly random time in this window after the load starts")
    parser.add_argument("--kill-in-decode-steady", action="store_true",
                        default=os.environ.get("WIDEEP_KILL_IN_DECODE_STEADY") == "1",
                        help="failover decode/stop kills: trigger only once rank 0 logs consecutive decode-only "
                             "steps at the load")
    parser.add_argument("--ready-timeout-s", type=float, default=2400.0)
    parser.add_argument("--first-token-at-ready", action="store_true",
                        help="failover: time a new request's first token as soon as the server is healthy, before the "
                             "warm-up (a restarted server's first token served)")
    parser.add_argument("--wait-clients-s", type=float, default=0.0,
                        help="failover, FT off: wait up to this long after the abort for the load's clients to end, "
                             "then count the failed requests (0: stop the clients at once)")
    parser.add_argument("--kill-during-eval", action="store_true",
                        help="accuracy: the victim kills itself in a decode step (TLLM_TEST_KILL_TRIGGER) --kill-window-s "
                             "after the evaluation's first answered request")
    parser.add_argument("--kill-before-eval", action="store_true",
                        help="accuracy: the victim kills itself in a decode step (TLLM_TEST_KILL_TRIGGER) of a warm load "
                             "(--load, --iterations) --kill-window-s after it starts; the evaluation runs after the "
                             "recovery, on the degraded deployment")
    parser.add_argument("--accuracy-max-retries", type=int, default=1,
                        help="accuracy: attempts per request in lm_eval")
    parser.add_argument("--profile-after-s", type=float, default=0.0,
                        help="profile cases: open a 50-step nsys window this long after each load starts")
    parser.add_argument("--profile-on-slow-step", type=float, default=0.0, metavar="FACTOR",
                        help="EV6: open one 600-step nsys window at the first full-load decode step this many times "
                             "slower than the median of the load's first 100 such steps")
    parser.add_argument("--profile-first-load-steps", type=int, default=0, metavar="STEPS",
                        help="open one nsys window of STEPS steps at the first full-load decode step of the first load")
    parser.add_argument("--profile-from-ready-steps", type=int, default=0, metavar="STEPS",
                        help="sweep: open one nsys window of STEPS steps as soon as the server is ready")
    args = parser.parse_args()
    driver = Driver(args)
    try:
        {"sweep": driver.run_sweep, "failover": driver.run_failover, "restart": driver.run_restart,
         "accuracy": driver.run_accuracy, "stress": driver.run_stress}[args.mode]()
        driver.result["state"] = "DONE"
    except HarnessInvalid as error:  # the case did not test what it was meant to: not an FT result
        driver.result["state"] = "INVALID"
        driver.result["error"] = repr(error)
    except Exception as error:  # the case must end and free the nodes whatever failed
        driver.result["state"] = "FAIL"
        driver.result["error"] = repr(error)
    finally:
        driver.save()
        (args.run / "frontend.done").touch()


if __name__ == "__main__":
    main()
