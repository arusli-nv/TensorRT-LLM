#!/usr/bin/env python3
"""Rank 0's stalled time after a case's post-ready page-cache drop (dropcache-0.log start), and its step-time series.

Usage: python3 -I tools/drop_stalls.py [--csv OUT] CASE_DIR...
Per case: drop time (PT), stalled seconds per 10 s bucket from the drop (sum of host steps > 50 ms), and the excess over
the ~2 s/10 s context-step baseline summed over buckets 2-9 (10 s buckets numbered from 0; the first request's ~15 s
step falls in bucket 1).
--csv writes (case, seconds since drop, host step ms) for every step in the first 120 s: the figure's data.
"""
import argparse
import csv
import datetime as dt
import re
from pathlib import Path

STEP = re.compile(r"\[(\d\d/\d\d/\d{4}-\d\d:\d\d:\d\d)\].*iter = (\d+),.*host_step_time = ([\d.]+)ms")
DROP = re.compile(r"start=([\d.]+)")
SLOW_MS, BASELINE_S, BUCKETS = 50.0, 2.0, 12


def series(case: Path, drop: float):
    for line in open(case / "rank-0.log", errors="replace"):
        match = STEP.search(line)
        if match:
            t = dt.datetime.strptime(match.group(1), "%m/%d/%Y-%H:%M:%S").timestamp()
            if 0 <= t - drop < 10 * BUCKETS:
                yield t - drop, float(match.group(3))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv")
    parser.add_argument("cases", nargs="+")
    args = parser.parse_args()
    rows = []
    print("case | drop (PT) | stalled s per 10 s from the drop | excess s, buckets 2-9")
    for case in map(Path, args.cases):
        log = case / "dropcache-0.log"
        match = DROP.search(log.read_text()) if log.exists() else None
        if not match:
            print(f"{case} | no drop")
            continue
        drop = float(match.group(1))
        buckets = [0.0] * BUCKETS
        for since, ms in series(case, drop):
            rows.append((str(case), round(since, 1), ms))
            if ms > SLOW_MS:
                buckets[int(since // 10)] += ms / 1000
        excess = sum(max(0.0, b - BASELINE_S) for b in buckets[2:10])
        when = dt.datetime.fromtimestamp(drop).strftime("%H:%M:%S")
        print(f"{case} | {when} | {' '.join(f'{b:.1f}' for b in buckets)} | {excess:.1f}")
    if args.csv:
        with open(args.csv, "w", newline="") as out:
            writer = csv.writer(out)
            writer.writerow(["case", "s_since_drop", "host_step_ms"])
            writer.writerows(rows)


if __name__ == "__main__":
    main()
