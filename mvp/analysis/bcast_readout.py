"""Request-broadcast and context-step readout of testhooks TLLM_TEST_BCAST_LOG, per arm case and logging rank.

For each case dir (result.json + rank-<r>.log) in each RUN_DIR, within the measured loads' windows (result.json rows
healthy-*): WIDEEP_BCAST lines (one per broadcast of new requests: requests, wall ms, pickled bytes) and WIDEEP_CTX
lines (one per step that schedules context: tokens, wall ms of _prepare_and_schedule_batch, which holds the fetch and
the broadcast). Prints per case and rank: broadcasts n, ms p50/p90/mean, bytes p50, MB/s (sum bytes / sum ms), and
context steps n, tokens p50, schedule ms p50/p90/mean, and the broadcast share of the schedule time (sum bcast ms /
sum ctx sched ms).
Usage: python3 -I bcast_readout.py RUN_DIR... [--ranks 0,31]
"""
import json
import pathlib
import re
import statistics
import sys

BCAST = re.compile(r"^WIDEEP_BCAST rank (\d+) t=([0-9.]+) n=(\d+) ms=([0-9.]+) bytes=(\d+|None)")
CTX = re.compile(r"^WIDEEP_CTX rank (\d+) t=([0-9.]+) call=\d+ reqs=(\d+) tokens=(\d+) sched_ms=([0-9.]+)")


def pct(values, q):
    if not values:
        return float("nan")
    values = sorted(values)
    return values[min(len(values) - 1, int(q * len(values)))]


def windows(case):
    try:
        rows = json.loads((case / "result.json").read_text()).get("rows", [])
    except (OSError, ValueError):
        return []
    return [(r["start_wall_s"], r["start_wall_s"] + r["elapsed_s"]) for r in rows
            if str(r.get("label", "")).startswith("healthy-") and r.get("start_wall_s") and r.get("elapsed_s")]


def case_rank(case, rank, spans):
    log = case / f"rank-{rank}.log"
    bcast, ctx = [], []
    if log.exists():
        for line in log.read_text(errors="replace").splitlines():
            if (m := BCAST.match(line)) and any(a <= float(m.group(2)) <= b for a, b in spans):
                bcast.append((float(m.group(4)), None if m.group(5) == "None" else int(m.group(5))))
            elif (m := CTX.match(line)) and any(a <= float(m.group(2)) <= b for a, b in spans):
                ctx.append((int(m.group(4)), float(m.group(5))))
    ms = [b[0] for b in bcast]
    sizes = [b[1] for b in bcast if b[1] is not None]
    sched = [c[1] for c in ctx]
    mb_s = sum(sizes) / 1e6 / (sum(ms) / 1e3) if sizes and sum(ms) else float("nan")
    share = sum(ms) / sum(sched) if sched and sum(sched) else float("nan")
    return (f"| {case.parent.name} | {case.name} | {rank} | {len(ms)} | {pct(ms, .5):.2f} / {pct(ms, .9):.2f} / "
            f"{statistics.fmean(ms) if ms else float('nan'):.2f} | {pct(sizes, .5):.0f} | {mb_s:.0f} | {len(ctx)} | "
            f"{pct([c[0] for c in ctx], .5):.0f} | {pct(sched, .5):.1f} / {pct(sched, .9):.1f} / "
            f"{statistics.fmean(sched) if sched else float('nan'):.1f} | {share:.2f} |")


def main():
    args = sys.argv[1:]
    ranks = ["0", "31"]
    if "--ranks" in args:
        i = args.index("--ranks")
        ranks = args[i + 1].split(",")
        del args[i:i + 2]
    print("| run | arm | rank | bcasts | bcast ms p50/p90/mean | bytes p50 | MB/s | ctx steps | ctx tokens p50 "
          "| sched ms p50/p90/mean | bcast/sched |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for run in map(pathlib.Path, args):
        for case in sorted(d for d in run.iterdir() if (d / "result.json").exists()):
            spans = windows(case)
            for rank in ranks:
                print(case_rank(case, rank, spans))


if __name__ == "__main__":
    main()
