"""EP32 worker-loss acceptance: serve, SIGKILL one worker, check what the survivors serve.

Runs as the trtllm-llmapi-launch task on MPI rank 0 and writes result.json into --output.

- idle (default): kill while no request runs, then serve the requests again. They must match
  --reference, a healthy run whose EPLB routing already excludes the victim (greedy tokens
  depend on which rank serves each expert).
- --inflight-after S: serve once, then resubmit the same requests plus one owned by the victim
  and kill S seconds later, mid-decode. Run with routing that already excludes the victim, so the
  loss changes no routing: every survivor-owned request must reproduce the first pass exactly and
  the victim-owned request must fail.
"""
import argparse
import json
import os
import signal
import time
from pathlib import Path

import yaml

from tensorrt_llm import LLM, SamplingParams
from tensorrt_llm.scheduling_params import SchedulingParams

# (prompt, expected substring, attention-DP rank that serves it)
WORLD = int(os.environ.get("WIDEEP_WORLD", "32"))
REQUESTS = [("The capital of France is", "Paris", 0),
            ("The capital of Italy is", "Rome", WORLD // 2 + 1),
            ("The capital of Japan is", "Tokyo", WORLD - 1)]


def find_worker(rank: int) -> int:
    """Pid of the local mgmn worker process for MPI ``rank``."""
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


def submit(llm: LLM, requests, max_tokens: int) -> list:
    params = SamplingParams(temperature=0, max_tokens=max_tokens, ignore_eos=True)
    return [
        llm.generate_async(prompt, params,
                           scheduling_params=SchedulingParams(attention_dp_rank=adp_rank,
                                                              attention_dp_relax=False))
        for prompt, _, adp_rank in requests
    ]


def collect(requests, pending: list, deadline: float) -> list[dict]:
    rows = []
    for (prompt, expected, adp_rank), request in zip(requests, pending):
        row = dict(prompt=prompt, adp_rank=adp_rank)
        try:
            output = request.result(timeout=max(1.0, deadline - time.monotonic())).outputs[0]
        except Exception as error:  # noqa: BLE001 - the victim's request is expected to fail
            row["error"] = repr(error)
        else:
            row.update(token_ids=list(output.token_ids), text=output.text,
                       plausible=expected in output.text)
        rows.append(row)
    return rows


def bench(llm: LLM, per_rank: int, max_tokens: int, ranks=range(WORLD)) -> dict:
    """Full decode batches on the given attention-DP ranks; returns throughput and latency."""
    requests = [(f"Request {i} on rank {rank}: the capital of France is", "", rank)
                for rank in ranks for i in range(per_rank)]
    start = time.monotonic()
    pending = submit(llm, requests, max_tokens)
    latencies = []
    for request in pending:
        request.result(timeout=600)
        latencies.append(time.monotonic() - start)
    elapsed = time.monotonic() - start
    latencies.sort()
    return dict(requests=len(requests), output_tokens=len(requests) * max_tokens,
                elapsed_s=elapsed, tokens_per_s=len(requests) * max_tokens / elapsed,
                latency_p50_s=latencies[len(latencies) // 2], latency_max_s=latencies[-1])


def tokens(rows: list[dict]) -> list:
    return [row.get("token_ids") for row in rows]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--victim", type=int, default=2)
    parser.add_argument("--no-kill", action="store_true", help="healthy run only")
    parser.add_argument("--reference", type=Path, help="result.json of the same-routing healthy run")
    parser.add_argument("--inflight-after", type=float,
                        help="kill this many seconds into a decode instead of while idle")
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--bench-tokens", type=int, default=480)
    parser.add_argument("--load-kill", type=float, metavar="S",
                        help="kill S seconds into a full load of long, chunk-prefilled prompts")
    parser.add_argument("--degraded-bench", type=int, metavar="PER_RANK",
                        help="after an idle-mode recovery, benchmark the survivors")
    parser.add_argument("--bench", type=int, metavar="PER_RANK",
                        help="measure decode throughput with PER_RANK requests on every "
                        "attention-DP rank; no kill")
    parser.add_argument("--trigger", type=Path,
                        help="create this file instead of sending SIGKILL; the victim kills itself "
                        "right after launching a graph replay (TLLM_TEST_KILL_IN_REPLAY)")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    result = dict(state="FAIL", victim=args.victim)

    def save() -> None:
        (args.output / "result.json").write_text(json.dumps(result, indent=2) + "\n")

    def kill_victim() -> float:
        if args.trigger is None:
            pid = find_worker(args.victim)
            result["victim_pid"] = pid
            os.kill(pid, signal.SIGKILL)
            result["kill_wall_s"] = time.time()
            return time.monotonic()
        # The victim may run on another node; it logs a line right before killing itself.
        args.trigger.touch()
        log = args.output / f"rank-{args.victim}.log"
        deadline = time.monotonic() + 60
        while b"WIDEEP_SELF_KILL" not in log.read_bytes():
            if time.monotonic() > deadline:
                raise TimeoutError("victim did not kill itself after the trigger")
            time.sleep(0.001)
        result["kill_wall_s"] = time.time()
        return time.monotonic()

    llm = None
    try:
        start = time.monotonic()
        llm = LLM(model=args.model, **yaml.safe_load(args.config.read_text()))
        result["startup_s"] = time.monotonic() - start
        result["healthy"] = collect(REQUESTS, submit(llm, REQUESTS, args.max_tokens),
                                    time.monotonic() + 300)
        result["healthy_repeat"] = collect(REQUESTS, submit(llm, REQUESTS, args.max_tokens),
                                           time.monotonic() + 300)
        result["deterministic"] = tokens(result["healthy"]) == tokens(result["healthy_repeat"])
        save()
        if args.bench:
            result["bench"] = [bench(llm, args.bench, args.max_tokens) for _ in range(3)]
            result["state"] = "BENCH"
            return
        if args.no_kill:
            result["state"] = "PASS_HEALTHY" if result["deterministic"] else "FAIL_NONDETERMINISTIC"
            return

        if args.load_kill is not None:
            # About 400 prompt tokens, so prefill spans several chunks of max_num_tokens.
            filler = " ".join(f"item{i}" for i in range(130))
            requests = [(f"Request {i} on rank {rank}. {filler}. The capital of France is", "", rank)
                        for rank in range(WORLD) for i in range(args.bench or 8)]
            pending = submit(llm, requests, 64)
            time.sleep(args.load_kill)
            killed = kill_victim()
            rows = collect(requests, pending, time.monotonic() + 600)
            result["recovered_requests_done_s"] = time.monotonic() - killed
            survivors = [row for row in rows if row["adp_rank"] != args.victim]
            victims = [row for row in rows if row["adp_rank"] == args.victim]
            result["survivor_requests"] = len(survivors)
            result["survivor_failures"] = sum("error" in row or len(row["token_ids"]) != 64
                                              for row in survivors)
            result["victim_requests_failed"] = sum("error" in row for row in victims)
            result["victim_requests"] = len(victims)
            # A victim request may have finished before the kill; any other outcome must be the
            # retry error, and at least one must show the kill landed mid-load.
            victims_ok = all("error" in row or len(row["token_ids"]) == 64 for row in victims)
            ok = (result["survivor_failures"] == 0 and victims_ok
                  and result["victim_requests_failed"] > 0)
            result["state"] = "PASS" if ok else "FAIL"
            return

        if args.inflight_after is not None:
            victim_request = ("The capital of Spain is", "Madrid", args.victim)
            requests = [*REQUESTS, victim_request]
            pending = submit(llm, requests, args.max_tokens)
            time.sleep(args.inflight_after)
            killed = kill_victim()
            rows = collect(requests, pending, time.monotonic() + 300)
            result["recovered"], result["victim_request"] = rows[:-1], rows[-1]
            result["recovered_requests_done_s"] = time.monotonic() - killed
            same = tokens(result["recovered"]) == tokens(result["healthy"])
            victim_failed = "error" in result["victim_request"]
            result["victim_request_failed"] = victim_failed
            result["exact_tokens_after_loss"] = same
            result["state"] = "PASS" if same and victim_failed else "FAIL"
            return

        # Idle: every request above has completed. The post-loss requests are submitted right
        # after the kill, so they race the survivors' detection of the loss.
        time.sleep(2)
        killed = kill_victim()
        prompt, _, adp_rank = REQUESTS[0]
        probe = llm.generate_async(
            prompt, SamplingParams(temperature=0, max_tokens=4, ignore_eos=True), streaming=True,
            scheduling_params=SchedulingParams(attention_dp_rank=adp_rank, attention_dp_relax=False))
        next(iter(probe))
        result["first_token_after_kill_s"] = time.monotonic() - killed
        result["recovered"] = collect(REQUESTS, submit(llm, REQUESTS, args.max_tokens),
                                      time.monotonic() + 300)
        result["recovered_requests_done_s"] = time.monotonic() - killed
        expected = json.loads(args.reference.read_text())["healthy"] if args.reference else result["healthy"]
        same = tokens(expected) == tokens(result["recovered"])
        result["exact_tokens_after_loss"] = same
        result["state"] = "PASS" if same else "FAIL_TOKENS_DIFFER"
        if args.degraded_bench:
            survivors = [rank for rank in range(WORLD) if rank != args.victim]
            result["degraded_bench"] = [bench(llm, args.degraded_bench, args.bench_tokens, survivors)
                                        for _ in range(3)]
    except BaseException as error:
        result["error"] = repr(error)
        raise
    finally:
        save()
        summary = {k: v for k, v in result.items() if k not in ("healthy", "healthy_repeat", "recovered")}
        print("WIDEEP_RESULT", json.dumps(summary), flush=True)
        if llm is not None:
            llm.shutdown()
        (args.output / "frontend.done").touch()


if __name__ == "__main__":
    main()
