"""FT on vs off rank-0 decode step at equal KV-cache fill: the steady device-step median per cached_kv_tokens bin.

Within a load window the KV cache fills as requests grow, and the step grows with it, so two arms whose windows
cover different fill ranges differ in step time without any cost (red-team B: 32/GPU drifts 13.26 -> 13.99 ms over
the window). Per bin (edges at quantiles of both arms' fills): steps per arm, each arm's median, the delta in %;
then the bins' delta weighted by their step counts.
Usage: python3 kv_bins.py OFF_CASE ON_CASE [--load 32] [--bins 5] [--floor 0.9]
"""
import argparse
import datetime
import json
import pathlib
import re
import statistics

LINE = re.compile(r"prev_device_step_time = ([\d.]+)ms, timestamp = ([\d-]+ [\d:]+), states = "
                  r"\{'num_ctx_requests': (\d+), [^}]*'cached_kv_tokens': (\d+)\}, num_active = (\d+)")
parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument("off", type=pathlib.Path)
parser.add_argument("on", type=pathlib.Path)
parser.add_argument("--load", type=int, default=32)
parser.add_argument("--bins", type=int, default=5)
parser.add_argument("--floor", type=float, default=0.9)
args = parser.parse_args()


def steps(case):
    """(cached_kv_tokens, step ms) of the decode-only full-load steps in the case's healthy-LOAD window.

    A line's prev_device_step_time belongs to the step before it, so it pairs with that step's fill and load.
    """
    row = next(r for r in json.loads((case / "result.json").read_text())["rows"] if r["label"] == f"healthy-{args.load}")
    start, end = row["start_wall_s"], row["start_wall_s"] + row["elapsed_s"]
    out, prev = [], None
    for line in open(case / "rank-0.log", errors="replace"):
        m = LINE.search(line)
        if not m:
            continue
        ms, stamp, ctx, kv, active = float(m[1]), m[2], int(m[3]), int(m[4]), int(m[5])
        wall = datetime.datetime.strptime(stamp, "%Y-%m-%d %H:%M:%S").timestamp()
        if prev and start <= prev[0] <= end and prev[1] == 0 and prev[3] >= args.floor * args.load:
            out.append((prev[2], ms))
        prev = (wall, ctx, kv, active)
    return out


off, on = steps(args.off), steps(args.on)
fills = sorted(kv for kv, _ in off + on)
edges = [fills[int(i * (len(fills) - 1) / args.bins)] for i in range(args.bins + 1)]
total_w = total_d = 0.0
for lo, hi in zip(edges, edges[1:]):
    a = [ms for kv, ms in off if lo <= kv <= hi]
    b = [ms for kv, ms in on if lo <= kv <= hi]
    if len(a) < 10 or len(b) < 10:
        print(f"kv {lo:7d}-{hi:7d}: off {len(a)} on {len(b)} steps (too few)")
        continue
    ma, mb = statistics.median(a), statistics.median(b)
    d = 100 * (mb - ma) / ma
    w = min(len(a), len(b))
    total_w += w
    total_d += w * d
    print(f"kv {lo:7d}-{hi:7d}: off {len(a):5d} steps {ma:7.3f} ms  on {len(b):5d} steps {mb:7.3f} ms  delta {d:+6.2f}%")
print(f"KV_BINNED_DELTA load={args.load} {total_d / total_w:+.2f}% (weighted by min steps per bin)" if total_w else "no bins")
