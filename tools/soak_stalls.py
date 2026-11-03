#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""SOAK readout (review/drafts/soak_cmd.txt header): stall events and loaded hours per arm, and the decision numbers.

Usage: soak_stalls.py [--verdict] CASE_DIR...   (e.g. mvp/runs/STD*-<job>-fillseg*/std-online*-1)
--verdict: also the 5 VERDICT lines of the soak_cmd.txt rule (per job: interim; the rule is decided on the pooled chain).
A case with a new Xid/NVRM kernel line (kernel-<node>-after.txt vs -before.txt) is excluded from the counts and listed.

Window: each 32/GPU row of result.json (burnin-32, healthy-32). It starts at the first rank-0 iteration with
num_scheduled_requests > 0 at or after start_wall_s. A finished row (rc 0) ends at its last rank-0 iteration inside
[start_wall_s, start_wall_s + elapsed_s]; an unfinished one (rc != 0) ends at start_wall_s + elapsed_s, capped by
MPIRUN_EXIT. If the case ended inside a load whose row was never written, that load's window runs from the end of the
last written row to MPIRUN_EXIT. Exposure (time at risk): first to last rank-0 iteration of each window. Event: a gap of more than 60 s between consecutive rank-0 iterations in a window, or
(unfinished windows) from the last iteration to the window end. Rank-0 iteration timestamps are node-local
America/Los_Angeles wall time with 1 s resolution.
"""

import datetime
import json
import math
import re
import sys
from pathlib import Path

GAP_S = 60.0
try:
    from zoneinfo import ZoneInfo

    TZ = ZoneInfo("America/Los_Angeles")
except Exception:  # no tz database: the node-local zone the logs were written in
    TZ = None
ITER = re.compile(rb"iter = (\d+), .*?num_scheduled_requests = (\d+), .*?timestamp = (\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)")


def iterations(log: Path) -> list:
    out = []
    with open(log, "rb") as lines:
        for line in lines:
            if b"[RANK 0] iter = " not in line:
                continue
            m = ITER.search(line)
            if m:
                t = datetime.datetime.strptime(m.group(3).decode(), "%Y-%m-%d %H:%M:%S").replace(tzinfo=TZ).timestamp()  # TZ None: local time
                out.append((t, int(m.group(2))))
    return out


def mpirun_exit(case: Path):
    for line in (case / "timeline").read_text(errors="replace").splitlines() if (case / "timeline").exists() else []:
        if line.startswith("MPIRUN_EXIT "):
            return float(line.split()[1])
    return None


def windows(case: Path) -> list:
    result = json.loads((case / "result.json").read_text())
    rows = [r for r in result.get("rows", []) if r.get("load_per_gpu") == 32 and r.get("osl", 1024) == 1024]
    end_case = mpirun_exit(case)
    out = []
    for r in rows:
        start, end = r["start_wall_s"], r["start_wall_s"] + r["elapsed_s"]
        finished = r.get("rc") == 0
        if not finished and end_case is not None:
            end = min(end, end_case)
        out.append((r["label"], start, end, finished))
    # A load the case ended inside, before its row was written (e.g. the case timeout killed the client).
    labels = [r["label"] for r in result.get("rows", [])]
    if end_case is not None and "healthy-32" not in labels and labels:
        last_end = max(r["start_wall_s"] + r["elapsed_s"] for r in result["rows"])
        if end_case - last_end > GAP_S:
            out.append(("healthy-32-unwritten", last_end, end_case, False))
    return out


def poisson_upper(k: int, conf: float = 0.95) -> float:
    """One-sided upper bound on the Poisson mean given k events."""
    lo, hi = 0.0, 10.0 * (k + 3)
    for _ in range(200):
        mu = (lo + hi) / 2
        cdf = sum(math.exp(-mu) * mu**i / math.factorial(i) for i in range(k + 1))
        lo, hi = (mu, hi) if cdf > 1 - conf else (lo, mu)
    return hi


def new_kernel_lines(case: Path) -> int:
    """Xid / NVRM lines in any node's after-log that its before-log lacks (-1: a node's dmesg was unreadable)."""
    n = 0
    for after in case.glob("kernel-*-after.txt"):
        before = after.with_name(after.name.replace("-after", "-before"))
        a = after.read_text(errors="replace").splitlines()
        b = set(before.read_text(errors="replace").splitlines()) if before.exists() else set()
        if a and a[0].startswith("dmesg not permitted"):
            return -1
        n += sum(1 for line in a[1:] if line not in b)
    return n


def main() -> None:
    args = sys.argv[1:]
    verdict = "--verdict" in args
    arms, excluded = {}, []
    for arg in [a for a in args if a != "--verdict"]:
        case = Path(arg)
        if not (case / "rank-0.log").exists() or not (case / "result.json").exists():
            print(f"SKIP {case}: no rank-0.log or result.json")
            continue
        arm = re.sub(r"-\d+$", "", case.name)
        its = iterations(case / "rank-0.log")
        hang_lines = sum(1 for _ in re.finditer(rb"Hang detected after", (case / "rank-0.log").read_bytes()))
        timeline = (case / "timeline").read_text(errors="replace") if (case / "timeline").exists() else ""
        stall_stop = "STALL_STOP" in timeline
        case_s, events = 0.0, []
        for label, start, end, finished in windows(case):
            ts = [t for t, sched in its if start <= t <= end]
            first = next((t for t, sched in its if t >= start and t <= end and sched > 0), None)
            if first is None:
                if not finished and end - start > GAP_S:
                    events.append((label, start, end - start, "no iteration in window"))
                continue
            ts = [t for t in ts if t >= first]
            case_s += ts[-1] - first  # time at risk: up to the last iteration (a hang's own length is not exposure)
            for a, b in zip(ts, ts[1:]):
                if b - a > GAP_S:
                    events.append((label, a, b - a, "recovered"))
            if not finished and end - ts[-1] > GAP_S:
                events.append((label, ts[-1], end - ts[-1], "never recovered (hang)"))
        kern = new_kernel_lines(case)
        print(f"CASE {case} arm={arm} loaded_s={case_s:.0f} events={len(events)} hang_detected_lines={hang_lines} "
              f"stall_stop={int(stall_stop)} new_kernel_lines={kern}")
        for label, t0, gap, kind in events:
            when = datetime.datetime.fromtimestamp(t0, TZ).strftime("%m-%d %H:%M:%S")
            print(f"  EVENT {label} last_iter={when} gap_s={gap:.0f} {kind}")
        if kern != 0:
            excluded.append(f"{case.parent.name}/{case.name}({'dmesg unreadable' if kern < 0 else f'{kern} new Xid/NVRM'})")
            continue
        d = arms.setdefault(arm, dict(cases=0, s=0.0, events=0, hangs=0))
        d["cases"] += 1
        d["s"] += case_s
        d["events"] += len(events)
        d["hangs"] += sum(1 for e in events if e[3] != "recovered")
    for arm, d in sorted(arms.items()):
        h = d["s"] / 3600
        up = poisson_upper(d["events"]) / h if h else float("nan")
        print(f"ARM {arm} cases={d['cases']} loaded_h={h:.2f} events={d['events']} rate_per_h={d['events'] / h if h else 0:.3f} "
              f"upper95_per_h={up:.3f}")
    ft = next((a for a in arms if a.endswith("-ft")), None)
    off = next((a for a in arms if a != ft), None)
    tail = None
    if ft and off:
        da, db = arms[ft], arms[off]
        n = da["events"] + db["events"]
        if n:
            p = da["s"] / (da["s"] + db["s"])
            tail = sum(math.comb(n, i) * p**i * (1 - p) ** (n - i) for i in range(da["events"], n + 1))
            print(f"SPLIT n={n} {ft}={da['events']} {off}={db['events']} P({ft}>={da['events']} | equal rates, exposure "
                  f"share {p:.2f})={tail:.3f}")
    if not verdict:
        return
    z = dict(cases=0, s=0.0, events=0, hangs=0)
    da, db = arms.get(ft or "std-online-ft", z), arms.get(off or "off", z)
    print(f"VERDICT 1 HOURS {ft or 'std-online-ft'}={da['s'] / 3600:.2f} h ({da['cases']} cases) | "
          f"{off or 'off'}={db['s'] / 3600:.2f} h ({db['cases']} cases)")
    print(f"VERDICT 2 EVENTS {ft or 'std-online-ft'}={da['events']} (hangs {da['hangs']}) | {off or 'off'}={db['events']} "
          f"(hangs {db['hangs']})")
    k, m = da["events"], db["events"]
    if k and m:
        reading = "events in both arms -> the transport (ob1/TCP + ULFM together), not FT's code"
    elif k:
        reading = (f"FT arm only, k={k}: " + ("FT's code" if k >= 5 else "consistent with FT's code, not shown")
                   + (f" (P under equal rates {tail:.3f})" if tail is not None else ""))
    elif m:
        reading = "off arm only -> the transport (ob1/TCP + ULFM together); FT's code is not needed for it"
    else:
        ub = lambda d: f"{poisson_upper(0) / (d['s'] / 3600):.2f}/h" if d["s"] else "n/a"
        reading = f"no events -> rare; one-sided 95% upper bound FT {ub(da)}, off {ub(db)}"
    print(f"VERDICT 3 INTERIM (this input only): {reading}")
    print(f"VERDICT 4 EXCLUDED: {', '.join(excluded) if excluded else 'none'}")
    print("VERDICT 5 RULE: decided only on the pooled chain (soak_stalls.py --verdict over every SOAK job's fillseg "
          "dirs); a per-job reading is interim and is never the result")


if __name__ == "__main__":
    main()
