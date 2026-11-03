#!/usr/bin/env python3
"""Cold-start step readout: rank 0's first N iterations after the server is ready (from iter 1, whatever load runs).

Usage: python3 -I tools/cold_steps.py [--first 150] CASE_DIR...   (CASE_DIR holds rank-0.log)
Prints per case: FT extra-slot mode (rank 0's "appends N <mode> slots" line; none = FT off), /dev/shm GB at ready
(hostmem-0.log), steps read, steps > 1 s, their sum, p50/p90 host step ms, and the slowest five (iter, ms).
Iteration 1 (the first request's step, ~15 s in every arm) is reported apart and excluded from the rest.
"""
import argparse
import re
from pathlib import Path

ITER = re.compile(r"iter = (\d+), .*?host_step_time = ([\d.]+)ms")
SLOTS = re.compile(r"recovery appends (\d+) (\w+) slots")


def slots_and_shm(case: Path) -> tuple[str, str]:
    mode = "none"
    with open(case / "rank-0.log", errors="replace") as lines:
        for line in lines:
            match = SLOTS.search(line)
            if match:
                mode = f"{match.group(2)}x{match.group(1)}"
                break
    shm = "?"
    hostmem = case / "hostmem-0.log"
    if hostmem.exists():
        for line in hostmem.read_text(errors="replace").splitlines():
            if line.endswith("/dev/shm"):
                shm = f"{int(line.split()[2]) / 1e9:.0f}"
    return mode, shm


def first_steps(log: Path, first: int) -> list[tuple[int, float]]:
    """The serving executor's iterations restart at 1 after the warm-up executor's: keep the last run from iter 1."""
    steps = []
    with open(log, errors="replace") as lines:
        for line in lines:
            match = ITER.search(line)
            if not match:
                continue
            it = int(match.group(1))
            if it == 1:
                steps = []
            if it <= first:
                steps.append((it, float(match.group(2))))
    return steps


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--first", type=int, default=150)
    parser.add_argument("cases", nargs="+")
    args = parser.parse_args()
    print("case | slots | shm GB | iter1 ms | steps | >1 s | >1 s sum s | p50 ms | p90 ms | slowest (iter, ms)")
    for case in args.cases:
        log = Path(case) / "rank-0.log"
        if not log.exists():
            print(f"{case} | no rank-0.log")
            continue
        steps = first_steps(log, args.first)
        if not steps:
            print(f"{case} | no iteration lines")
            continue
        first = steps[0][1] if steps[0][0] == 1 else None
        rest = [s for s in steps if s[0] != 1]
        times = sorted(ms for _, ms in rest)
        big = [ms for ms in times if ms > 1000]
        p50 = times[len(times) // 2] if times else float("nan")
        p90 = times[int(0.9 * (len(times) - 1))] if times else float("nan")
        slowest = sorted(rest, key=lambda s: -s[1])[:5]
        mode, shm = slots_and_shm(Path(case))
        print(f"{case} | {mode} | {shm} | {first or float('nan'):.0f} | {len(rest)} | {len(big)} | {sum(big) / 1000:.1f} | {p50:.1f} | "
              f"{p90:.1f} | {', '.join(f'{i}:{ms:.0f}' for i, ms in slowest)}")


if __name__ == "__main__":
    main()
