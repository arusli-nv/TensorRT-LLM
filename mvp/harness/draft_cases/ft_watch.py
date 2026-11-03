"""Side driver for the TWOKILL-R1 and DRH2-R1 draft cases (node 0, next to bench.py / tokens/frontend.py).

twokill RUN: samples /health every --health-period-s and /fault_tolerance every second from server_ready until
  RUN/frontend.done; once --first-count rank logs report generation 1 ("resumed serving ... at membership
  generation 1"), waits --margin-s and creates --second-trigger (a later rank's decode trigger, see
  TLLM_TEST_KILL_TRIGGER_FOR), then waits for --final-count rank logs at generation 2. Writes RUN/verdict.json
  and prints one TWOKILL_VERDICT line. Without --second-trigger the second loss comes from the case itself
  (TLLM_TEST_KILL_RERUN_OWNER: the new owner of a re-run request kills itself).
drh2 RUN: no sampling; after RUN/frontend.done, reads the tokens frontend's result.json, the rank logs, the timeline,
  the victim GPU's memory samples and the per-node kernel-log snapshots, writes RUN/verdict.json and prints one
  DRH2_VERDICT line.
Benchmark-only (never part of TRT-LLM).
"""

import argparse
import json
import re
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

RESUMED = re.compile(rb"resumed serving without ranks \[([0-9, ]*)\] at membership generation (\d+)")
PHASE = re.compile(rb"WIDEEP_PHASE (\w+) ([0-9.]+)")
COUNTERS = re.compile(rb"WIDEEP_(CAPTURE|COMPILE|AUTOTUNE) ")
# Survivor-side device errors (Xid itself is only in the kernel log).
CUDA_ERRORS = re.compile(
    rb"illegal memory access|unspecified launch failure|misaligned address|illegal instruction|"
    rb"device-side assert|CUDA error|cudaError|uncorrectable ECC|Xid", re.IGNORECASE)


def get(port: int, path: str, timeout: float = 2.0):
    """(HTTP status, JSON body or None, ms) of one GET; status is the error repr if the request failed."""
    start = time.time()
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/{path}", timeout=timeout) as response:
            status, raw = response.status, response.read()
    except urllib.error.HTTPError as error:
        status, raw = error.code, error.read()
    except OSError as error:
        return repr(error)[:80], None, (time.time() - start) * 1e3
    try:
        body = json.loads(raw or b"null")
    except ValueError:
        body = None
    return status, body, (time.time() - start) * 1e3


def rank_logs(run: Path) -> dict:
    return {int(p.stem.split("-")[1]): p for p in run.glob("rank-*.log") if p.stem.split("-")[1].isdigit()}


class Resumes:
    """Rank -> {generation: failed ranks it resumed without}, reading only what each rank log gained since the last
    call (the logs grow with every iteration)."""

    def __init__(self, run: Path):
        self.run, self.offsets, self.found = run, {}, {}

    def __call__(self, generation: int) -> dict:
        for rank, path in rank_logs(self.run).items():
            with open(path, "rb") as log:
                log.seek(self.offsets.get(rank, 0))
                chunk = log.read()
            complete = chunk.rfind(b"\n") + 1
            self.offsets[rank] = self.offsets.get(rank, 0) + complete
            for match in RESUMED.finditer(chunk[:complete]):
                self.found.setdefault(rank, {})[int(match.group(2))] = sorted(
                    int(r) for r in match.group(1).split(b",") if r.strip())
        return {rank: gens[generation] for rank, gens in self.found.items() if generation in gens}


def resumed(run: Path, generation: int) -> dict:
    """One full read of every rank log (verdicts after the case)."""
    return Resumes(run)(generation)


def counters_after_resume(run: Path, live: list) -> dict:
    """Per survivor: captures, compilations and autotuner runs logged after its first resume."""
    counts = {}
    for rank in live:
        path = run / f"rank-{rank}.log"
        text = path.read_bytes() if path.exists() else b""
        start = RESUMED.search(text)
        counts[rank] = len(COUNTERS.findall(text, start.end())) if start else None
    return counts


def load_running(run: Path) -> bool:
    """bench.py appends its "load" row only when the load finished."""
    try:
        result = json.loads((run / "result.json").read_text())
    except (OSError, ValueError):
        return True
    return "state" not in result and not any(row.get("label") == "load" for row in result.get("rows", []))


class Sampler:
    def __init__(self, run: Path, port: int, period_s: float):
        self.run, self.port, self.period_s = run, port, period_s
        self.health, self.status = [], []
        self.lock = threading.Lock()

    def done(self) -> bool:
        return (self.run / "frontend.done").exists() or (self.run / "done").exists()

    def health_loop(self) -> None:
        with open(self.run / "health.jsonl", "w", buffering=1) as out:
            while not self.done():
                code, body, ms = get(self.port, "health")
                row = dict(t=round(time.time(), 3), code=code, state=(body or {}).get("state"), ms=round(ms, 1))
                with self.lock:
                    self.health.append(row)
                out.write(json.dumps(row) + "\n")
                time.sleep(max(0.0, self.period_s - ms / 1e3))

    def status_loop(self) -> None:
        with open(self.run / "fault_tolerance.jsonl", "w", buffering=1) as out:
            while not self.done():
                # The recovery status is the /health body (state, generation, live/failed ranks, failed_requests).
                code, body, _ = get(self.port, "health", timeout=10)
                if isinstance(body, dict):
                    body["t"] = round(time.time(), 3)
                    with self.lock:
                        self.status.append(body)
                    out.write(json.dumps(body) + "\n")
                time.sleep(1.0)

    def start(self) -> None:
        for loop in (self.health_loop, self.status_loop):
            threading.Thread(target=self.logged, args=(loop,), daemon=True).start()

    @staticmethod
    def logged(loop) -> None:
        """Run a sampling loop; a loop that dies says so in the driver's log."""
        try:
            loop()
        except Exception as error:
            print(f"SAMPLER_DIED {loop.__name__} {error!r}", flush=True)
            raise

    def states(self) -> list:
        """Run-length sequence of the /health states seen: [[state or code, samples, first t, last t], ...]."""
        runs = []
        with self.lock:
            rows = list(self.health)
        for row in rows:
            key = row["state"] or str(row["code"])
            if runs and runs[-1][0] == key:
                runs[-1][1] += 1
                runs[-1][3] = row["t"]
            else:
                runs.append([key, 1, row["t"], row["t"]])
        return runs


def wait_for(predicate, timeout_s: float, done) -> bool:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if predicate():
            return True
        if done():
            return predicate()
        time.sleep(0.5)
    return False


def twokill(args) -> dict:
    run = args.run
    verdict = dict(case=run.name, checks={})
    while not (run / "server_ready").exists():
        if (run / "frontend.done").exists():
            verdict.update(verdict="FAIL", reason="server never ready")
            return verdict
        time.sleep(1)
    sampler = Sampler(run, args.port, args.health_period_s)
    sampler.start()
    resumes = Resumes(run)
    if not wait_for(lambda: len(resumes(1)) >= args.first_count, 900, sampler.done):
        verdict.update(verdict="FAIL", reason=f"generation 1 reached by {len(resumes(1))} ranks")
        return finish_twokill(args, verdict, sampler)
    verdict["generation1_seen_wall_s"] = time.time()
    if args.second_trigger:
        time.sleep(args.margin_s)
        if not load_running(run):
            verdict.update(verdict="INVALID", reason="the load ended before the second kill")
            return finish_twokill(args, verdict, sampler)
        Path(args.second_trigger).touch()
        verdict["second_trigger_wall_s"] = time.time()
    if not wait_for(lambda: len(resumes(2)) >= args.final_count, 600, sampler.done):
        verdict.update(verdict="FAIL", reason=f"generation 2 reached by {len(resumes(2))} ranks")
        return finish_twokill(args, verdict, sampler)
    verdict["generation2_seen_wall_s"] = time.time()
    time.sleep(3)  # a few /fault_tolerance samples at generation 2 while the load still runs
    return finish_twokill(args, verdict, sampler)


def finish_twokill(args, verdict: dict, sampler: Sampler) -> dict:
    run = args.run
    while not sampler.done():
        time.sleep(1)
    time.sleep(2)
    verdict["health_states"] = sampler.states()
    verdict["health_samples"] = len(sampler.health)
    verdict["health_max_ms"] = max((row["ms"] for row in sampler.health), default=None)
    final = next((body for body in reversed(sampler.status) if "generation" in body), {})
    verdict["fault_tolerance_last"] = final
    claim = Path(f"{args.rerun_owner}.claim") if args.rerun_owner else None
    if claim is not None and claim.exists():
        verdict["rerun_owner_claim"] = claim.read_text().split()
    try:
        bench = json.loads((run / "result.json").read_text())
    except (OSError, ValueError):
        bench = {}
    verdict["bench_state"] = bench.get("state")
    verdict["load_row"] = next(({k: row.get(k) for k in ("completed", "failed_requests", "short_requests")}
                                for row in bench.get("rows", []) if row.get("label") == "load"), None)
    if "verdict" in verdict:  # already decided (FAIL / INVALID before generation 2)
        return verdict
    expected = set(args.expect_failed)
    if claim is not None:
        owner = verdict.get("rerun_owner_claim")
        if owner:
            expected.add(int(owner[0]))
    survivors = sorted(set(range(args.world)) - expected)
    gen2 = resumed(run, 2)
    rank0 = (run / "rank-0.log").read_bytes() if (run / "rank-0.log").exists() else b""
    quarantine = [int(m.group(1)) for m in re.finditer(rb"WIDEEP_QUARANTINE .*failed=\[[0-9, ]*\] count=(\d+)", rank0)]
    errors = b"".join(p.read_bytes() for p in (run / "bench").glob("load*.json")) if (run / "bench").exists() else b""
    after = counters_after_resume(run, survivors)
    states = [entry[0] for entry in verdict["health_states"]]
    checks = verdict["checks"]
    checks["generation2_on_every_survivor"] = sorted(gen2) == survivors and all(
        sorted(f) == sorted(expected) for f in gen2.values())
    checks["survivors"] = len(survivors) == args.world - len(expected) == args.final_count
    checks["status_generation2"] = final.get("generation") == 2 and final.get("recoveries") == 2
    checks["status_failed_ranks"] = sorted(final.get("failed_ranks", [])) == sorted(expected)
    checks["status_live_ranks"] = sorted(final.get("live_ranks", [])) == survivors
    checks["health_saw_degraded"] = "degraded" in states
    checks["health_never_terminal"] = "terminal" not in states
    checks["load_done"] = verdict["bench_state"] == "DONE"
    checks["no_capture_compile_autotune_after_resume"] = all(n == 0 for n in after.values())
    verdict["after_resume_counters_nonzero"] = {r: n for r, n in after.items() if n}
    if args.rerun_owner:
        checks["rerun_owner_killed"] = bool(verdict.get("rerun_owner_claim"))
        checks["quarantined_at_second_recovery"] = len(quarantine) >= 2 and quarantine[1] >= 1
        # Reported only: a stream the server ends with an error may reach benchmark_serving as a short request.
        verdict["client_saw_quarantine_error"] = b"already run again" in errors
    # Not a pass criterion: "recovering" lasts ~1 s and may fall between samples; reported, not required.
    verdict["health_saw_recovering"] = "recovering" in states
    verdict["verdict"] = "PASS" if all(checks.values()) else "FAIL"
    return verdict


def wall(text: bytes, pattern: bytes):
    match = re.search(pattern + rb".*? t=([0-9.]+)", text)
    return float(match.group(1)) if match else None


def drh2(args) -> dict:
    run = args.run
    while not ((run / "frontend.done").exists() or (run / "done").exists()):
        time.sleep(1)
    time.sleep(5)  # rank logs and the other nodes' snapshots
    verdict = dict(case=run.name, checks={})
    try:
        result = json.loads((run / "result.json").read_text())
    except (OSError, ValueError):
        result = {}
    victim = args.victim
    logs = rank_logs(run)
    victim_log = logs[victim].read_bytes() if victim in logs else b""
    timeline = (run / "timeline").read_text() if (run / "timeline").exists() else ""
    exit_match = re.search(rf"RANK_EXIT {victim} rc=\S+ ([0-9.]+)", timeline)
    t_exit = float(exit_match.group(1)) if exit_match else None
    t_arm = wall(victim_log, rb"WIDEEP_SELF_STALL rank \d+") or wall(victim_log, rb"WIDEEP_SELF_KILL rank \d+")
    resumed_at, notice, abort_set, errors = [], [], [], {}
    for rank, path in logs.items():
        if rank == victim:
            continue
        text = path.read_bytes()
        resumed_at += [float(m.group(2)) for m in PHASE.finditer(text) if m.group(1) == b"resumed"]
        if (t := wall(text, rb"WIDEEP_ABORT_DELAY rank \d+")) is not None:
            notice.append(t)
        if (t := wall(text, rb"WIDEEP_ABORT_SET rank \d+")) is not None:
            abort_set.append(t)
        # From the loss on (the first recovery marker): startup lines are not this case's evidence.
        first = PHASE.search(text)
        hits = CUDA_ERRORS.findall(text, first.start() if first else 0)
        if hits:
            errors[rank] = sorted({h.decode(errors="replace") for h in hits})
    # Victim GPU memory, sampled every 200 ms on its node: when it was freed.
    t_free = None
    samples = run / "victim-gpumem.csv"
    if samples.exists() and t_arm is not None:
        for line in samples.read_text().splitlines():
            fields = [f.strip() for f in line.split(",")]
            try:
                t, used = float(fields[0]), float(fields[1].split()[0])
            except (IndexError, ValueError):
                continue
            if t > t_arm and used < 1024:
                t_free = t
                break
    xid_new = {}
    for after in run.glob("kernel-*-after.txt"):
        before = after.with_name(after.name.replace("-after", "-before"))
        old = set(before.read_text().splitlines()) if before.exists() else set()
        new = [line for line in after.read_text().splitlines() if line not in old and "xid" in line.lower()]
        if new:
            xid_new[after.name] = new[:10]
    readable = [p.name for p in run.glob("kernel-*-before.txt") if "not permitted" not in p.read_text().lower()
                and "denied" not in p.read_text().lower()]
    verdict.update(
        mode=args.mode, frontend_state=result.get("state"), exact_tokens_after_loss=result.get("exact_tokens_after_loss"),
        victim_finished_before_kill=result.get("victim_finished_before_kill"),
        victim_request_failed=result.get("victim_request_failed"), t_arm=t_arm, t_victim_exit=t_exit,
        t_victim_gpu_freed=t_free, t_first_notice=min(notice, default=None), t_abort_set_max=max(abort_set, default=None),
        t_resumed_max=max(resumed_at, default=None), survivors_resumed=len(resumed(run, 1)),
        survivor_cuda_errors=errors, kernel_log_readable_nodes=sorted(readable), new_xid_lines=xid_new)
    checks = verdict["checks"]
    checks["recovered"] = verdict["survivors_resumed"] == args.world - 1
    checks["exact_tokens_after_loss"] = result.get("exact_tokens_after_loss") is True
    checks["kill_mid_generation"] = result.get("victim_finished_before_kill") is False
    checks["no_survivor_cuda_error"] = not errors
    checks["no_new_xid"] = not xid_new
    # Preconditions: without them the case did not test DR-H2 (INVALID, not FAIL).
    pre = verdict["preconditions"] = {}
    if args.mode == "late-write":
        landed = t_arm + args.delay_s if t_arm is not None else None
        verdict["t_late_write_est"] = landed
        # Reported only: whether the late write fell inside the compared decode (else only the fault checks bite).
        if landed is not None and result.get("kill_wall_s") and result.get("recovered_requests_done_s"):
            verdict["late_write_during_compared_decode"] = (
                landed < result["kill_wall_s"] + result["recovered_requests_done_s"])
        pre["victim_alive_at_late_write"] = landed is not None and (t_exit is None or t_exit > landed + 1)
        pre["late_write_after_survivors_resumed"] = (landed is not None and verdict["t_resumed_max"] is not None
                                                     and landed > verdict["t_resumed_max"])
    elif args.mode == "delay-abort":
        pre["abort_after_victim_memory_freed"] = (t_free is not None and verdict["t_abort_set_max"] is not None
                                                  and min(abort_set) > t_free + 3)
    pre["kernel_log_or_rank_logs"] = True  # rank logs always; the kernel log only where readable (recorded)
    verdict["verdict"] = ("INVALID" if not all(pre.values()) else "PASS" if all(checks.values()) else "FAIL")
    return verdict


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("kind", choices=("twokill", "drh2"))
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--world", type=int, default=32)
    parser.add_argument("--health-period-s", type=float, default=0.05)
    parser.add_argument("--first-count", type=int, default=31, help="rank logs that must reach generation 1")
    parser.add_argument("--final-count", type=int, default=30, help="rank logs that must reach generation 2")
    parser.add_argument("--margin-s", type=float, default=10.0, help="wait after generation 1 before the 2nd kill")
    parser.add_argument("--second-trigger", help="file the second victim polls (TLLM_TEST_KILL_TRIGGER_FOR)")
    parser.add_argument("--expect-failed", type=lambda s: [int(r) for r in s.split(",") if r], default=[])
    parser.add_argument("--rerun-owner", help="TLLM_TEST_KILL_RERUN_OWNER prefix (quarantine case)")
    parser.add_argument("--victim", type=int, default=5)
    parser.add_argument("--mode", choices=("control", "delay-abort", "late-write"), default="control")
    parser.add_argument("--delay-s", type=float, default=0.0, help="late-write: TLLM_TEST_LATE_WRITE_DELAY_S")
    args = parser.parse_args()
    args.run.mkdir(parents=True, exist_ok=True)
    verdict = twokill(args) if args.kind == "twokill" else drh2(args)
    (args.run / "verdict.json").write_text(json.dumps(verdict, indent=1) + "\n")
    label = "TWOKILL_VERDICT" if args.kind == "twokill" else "DRH2_VERDICT"
    summary = {k: v for k, v in verdict.items() if k in ("checks", "preconditions", "reason", "health_states",
                                                          "fault_tolerance_last", "rerun_owner_claim")}
    print(label, args.run.name, verdict.get("verdict"), json.dumps(summary)[:1500], flush=True)


if __name__ == "__main__":
    main()
