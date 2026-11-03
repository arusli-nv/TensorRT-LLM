"""Prefill (context) steps, FT on vs off: per group, mode and load, rank 0's context iterations inside the healthy-LOAD
window: count, total host time, median step, and host ms per 1k context tokens (normalizes the mix of chunk sizes, which
makes the median swing +-10% between jobs). The decode-only step metric excludes these steps; with the overlap scheduler
they still lengthen streams (TPOT) and the row. Report the per-token change paired over groups, with its spread.
Usage: python3 ctx_steps.py RUN_DIR [RUN_DIR ...]   (each run dir one node group of std-<mode>[-ft]-<round> cases)
"""
import datetime
import json
import os
import pathlib
import re
import statistics
import sys
import time

os.environ["TZ"] = "America/Los_Angeles"
time.tzset()
LINE = re.compile(r"host_step_time = ([\d.]+)ms.*?timestamp = ([\d-]+ [\d:]+), states = "
                  r"\{'num_ctx_requests': (\d+), 'num_ctx_tokens': (\d+)")


def context_steps(case: pathlib.Path) -> list:
    """(wall s, host ms, context tokens) of rank 0's iterations that ran context requests."""
    out = []
    for line in open(case / "rank-0.log", errors="replace"):
        match = LINE.search(line)
        if match and int(match[3]) > 0:
            wall = datetime.datetime.strptime(match[2], "%Y-%m-%d %H:%M:%S").timestamp()
            out.append((wall, float(match[1]), int(match[4])))
    return out


print("| group | mode | load | ctx steps off/on | ctx s off/on | median ms off/on | median % | ms/1k ctx tok off/on | per-token % |")
print("|---|---|---|---|---|---|---|---|---|")
for run in map(pathlib.Path, sys.argv[1:]):
    cases = {p.name: p for p in run.iterdir() if (p / "result.json").exists() and (p / "rank-0.log").exists()}
    for mode in ("noeplb", "offline", "online"):
        for off_name in sorted(n for n in cases if n.rsplit("-", 1)[0] == f"std-{mode}"):
            on_name = f"std-{mode}-ft-{off_name.rsplit('-', 1)[1]}"
            if on_name not in cases:
                continue
            arms = [({r["label"]: r for r in json.loads((cases[n] / "result.json").read_text())["rows"]},
                     context_steps(cases[n])) for n in (off_name, on_name)]
            for load in (8, 32, 128, 512):
                stats = []
                for rows, steps in arms:
                    row = rows.get(f"healthy-{load}")
                    if not row:
                        break
                    start, end = row["start_wall_s"], row["start_wall_s"] + row["elapsed_s"]
                    window = [(ms, tokens) for wall, ms, tokens in steps if start <= wall <= end]
                    if not window:
                        break
                    total, tokens = sum(ms for ms, _ in window), sum(t for _, t in window)
                    stats.append((len(window), total / 1e3, statistics.median(ms for ms, _ in window), total / tokens * 1e3))
                if len(stats) < 2:
                    continue
                (n0, t0, m0, p0), (n1, t1, m1, p1) = stats
                print(f"| {run.name} | {mode} | {load} | {n0}/{n1} | {t0:.1f}/{t1:.1f} | {m0:.0f}/{m1:.0f} | "
                      f"{100 * (m1 - m0) / m0:+.1f} | {p0:.1f}/{p1:.1f} | {100 * (p1 - p0) / p0:+.1f} |")
