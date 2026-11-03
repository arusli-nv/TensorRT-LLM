"""Steady-state decode from the server's per-iteration log (print_iter_log on rank 0), per benchmark row.

A steady iteration is a pure-decode step (no context requests) of the row's load that runs at least 90% of
the row's per-GPU concurrency on rank 0. Per row: their count, the median step time (host_step_time: one loop of
the overlap scheduler = one decode step), the median decode tokens per step, and from those the per-GPU output
throughput (tokens / step time) and the per-user rate (1 / step time; one token per request per step). Also the
mean and p99 step time, the mean throughput (tokens / summed step time) of the steady steps, and of every step
in the row's window (window_tok_s_gpu, context steps included).
Rank 0 stands for every rank: attention-DP ranks step in lockstep and balancing evens their batches.
Usage: python3 steady_state.py CASE_DIR   (prints one line per healthy row)
"""

import ast
import datetime
import json
import pathlib
import re
import statistics
import sys

LINE = re.compile(r"iter = (\d+),.*?host_step_time = ([\d.]+)ms.*?timestamp = ([\d-]+ [\d:]+), states = (\{[^}]*\})(?:, num_active = (\d+))?")
MIN_STEADY = 20


def iterations(log: pathlib.Path) -> list:
    """(wall seconds, step ms, decode tokens, context requests, active requests) per logged iteration.

    Decode tokens count MTP draft tokens too, so full load is judged on active requests (num_active).
    """
    out = []
    with open(log, errors="replace") as lines:
        for line in lines:
            if "iter = " not in line:
                continue
            match = LINE.search(line)
            if not match:
                continue
            states = ast.literal_eval(match.group(4))
            wall = datetime.datetime.strptime(match.group(3), "%Y-%m-%d %H:%M:%S").timestamp()
            out.append((wall, float(match.group(2)), states.get("num_generation_tokens", 0),
                        states.get("num_ctx_requests", 0), int(match.group(5) or 0)))
    return out


def steady(its: list, row: dict) -> dict:
    start, end = row["start_wall_s"], row["start_wall_s"] + row["elapsed_s"]
    floor = 0.9 * row["load_per_gpu"]
    window = [(ms, tokens) for wall, ms, tokens, ctx, active in its if start <= wall <= end]
    picked = [(ms, tokens) for wall, ms, tokens, ctx, active in its
              if start <= wall <= end and ctx == 0 and active >= floor]
    if len(picked) < MIN_STEADY:
        return {"steady_steps": len(picked)}
    step_ms = statistics.median(ms for ms, _ in picked)
    tokens = statistics.median(t for _, t in picked)
    times = sorted(ms for ms, _ in picked)
    # Means (tokens / summed step time) count slow steps that the median hides; cost claims use them.
    return {"steady_steps": len(picked), "steady_step_ms": step_ms, "steady_tokens_per_step": tokens,
            "steady_tok_s_gpu": tokens / step_ms * 1e3, "steady_user_tok_s": 1e3 / step_ms,
            "steady_mean_step_ms": statistics.mean(times),
            "steady_p99_step_ms": times[min(len(times) - 1, int(0.99 * len(times)))],
            "steady_mean_tok_s_gpu": sum(t for _, t in picked) / sum(times) * 1e3,
            "window_tok_s_gpu": sum(t for _, t in window) / sum(ms for ms, _ in window) * 1e3}


def annotate(case: pathlib.Path, rows: list) -> list:
    """The rows with their steady-state fields added (rank-0.log of the case)."""
    log = case / "rank-0.log"
    if not log.exists():
        return rows
    its = iterations(log)
    return [{**row, **steady(its, row)} for row in rows]


def main() -> None:
    case = pathlib.Path(sys.argv[1])
    rows = json.loads((case / "result.json").read_text()).get("rows", [])
    for row in annotate(case, [r for r in rows if r.get("label", "").startswith("healthy")]):
        print(json.dumps({k: row.get(k) for k in ("label", "load_per_gpu", "output_throughput", "median_tpot_ms",
                                                  "steady_steps", "steady_step_ms", "steady_tok_s_gpu",
                                                  "steady_user_tok_s")}))


if __name__ == "__main__":
    main()
