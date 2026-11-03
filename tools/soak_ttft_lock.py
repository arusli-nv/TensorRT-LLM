#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""SOAK TTFT locking test (launch/soak_cmd.txt header, "TTFT locking test (written before data, 10-26)").

Usage: soak_ttft_lock.py JOB_ID...          full readout over mvp/runs/STD{,-G1}-<job>-fillseg*/std-online*-1
       soak_ttft_lock.py --check CASE_DIR    one case: its per-sub-window lines only (no arm comparison)

Sub-windows: the healthy-32 row's rank-0 iterations (it.load: [start_wall_s, start_wall_s + elapsed_s], so the
burnin-32 and warm rows before it are outside) are cut into waves with wave.waves. Starting at the row's first wave,
a sub-window takes the consecutive whole waves whose first fetch is < 300 s after the sub-window's first wave; the next
sub-window starts at the next wave. A trailing sub-window with no wave starting >= 300 s later is incomplete and
dropped. Each sub-window is classified exactly as state.py classifies a finals window: median of
[w[-1] - w[0] for w in its waves][2:] <= 15 iterations -> locked, else dispersed.

Request placement: bench/healthy-32.json has per-request ttfts and e2els (list order = submission order) but no
start times. benchmark_serving submits in order through a FIFO asyncio.Semaphore(1024), so requests 0..1023 start at
t = 0 and request j >= 1024 starts when the earliest outstanding request completes (replayed with a heap of
start + e2el). The replay is stretched linearly so its last completion equals the reported "duration" (absorbs the
client's per-request overhead), and t = 0 is anchored at the first rank-0 fetch of the first wave with > 1 fetched
request (the single test prompt is 1 request). A request belongs to the sub-window its replayed start falls in. The
same method is used for both arms.
"""

import argparse
import datetime
import hashlib
import heapq
import importlib.util
import json
import mmap
import re
import statistics as st
import sys
from pathlib import Path

import numpy as np

R = Path("/lustre/fs1/portfolios/coreai/projects/coreai_comparch_trtllm/users/arusli/src/TensorRT-LLM/.wideep-ft-runs/claude")
SCRIPTS = R / "review/diag/why_ttft_restart_scripts"
SHA_PREFIX = "88930f297bfa"
SUB_S = 300.0
LOCK_IT = 15
CONC = 1024
SMOKE_JOBS = {"8302926"}
ARMS = {"std-online-ft": "FT", "std-online-ulfm": "OFF"}


def scripts_sha() -> str:
    return hashlib.sha256(b"".join(p.read_bytes() for p in sorted(SCRIPTS.glob("*.py")))).hexdigest()


def import_scripts():
    """it.py and wave.py read sys.argv at import (path insert, it.py's report loop): give them the scripts dir."""
    argv, sys.argv = sys.argv, ["", str(SCRIPTS)]
    sys.path.insert(0, str(SCRIPTS))
    try:
        import it
        import wave
    finally:
        sys.argv = argv
    return it, wave


def import_soak_stalls():
    spec = importlib.util.spec_from_file_location("soak_stalls", R / "tools/soak_stalls.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def iter_times(it, case: Path, row: dict) -> list:
    """Wall time of each iteration it.load(case) returns (same regex, same filter, same order)."""
    s, e = row["start_wall_s"], row["start_wall_s"] + row["elapsed_s"]
    out = []
    for line in open(case / "rank-0.log", errors="replace"):
        m = it.P.search(line)
        if not m:
            continue
        w = datetime.datetime.strptime(m[7], "%Y-%m-%d %H:%M:%S").timestamp()
        if s <= w <= e:
            out.append((int(m[1]), w))
    return out


def bench_arrays(path: Path):
    with open(path, "rb") as f, mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as m:
        def arr(key):
            a = m.find(b'"%s": [' % key)
            if a < 0 or m.find(b'"%s": [' % key, a + 1) >= 0:
                raise ValueError(f"{path}: key {key!r} missing or repeated")
            a += len(key) + 4
            return np.array(json.loads(m[a:m.find(b"]", a) + 1]), dtype=float)

        d = m.find(b'"duration": ')
        return arr(b"ttfts"), arr(b"e2els"), float(m[d + 12:m.find(b",", d)])


def replay_starts(e2el: np.ndarray, duration: float):
    start = np.zeros(len(e2el))
    heap = [(e2el[i], i) for i in range(min(CONC, len(e2el)))]
    heapq.heapify(heap)
    for j in range(CONC, len(e2el)):
        t, _ = heapq.heappop(heap)
        start[j] = t
        heapq.heappush(heap, (t + e2el[j], j))
    end = float((start + e2el).max())
    return start * (duration / end), end


def row_subwindows(it, wave, case: Path) -> dict:
    """Per-sub-window classification and median TTFT of the case's healthy-32 row."""
    row = {r["label"]: r for r in json.load(open(case / "result.json"))["rows"]}["healthy-32"]
    x = it.load(case)
    tx = iter_times(it, case, row)
    if len(tx) != len(x) or any(i != xi["it"] for (i, _), xi in zip(tx, x)):
        raise ValueError("timestamp parse does not align with it.load")
    t = [w for _, w in tx]
    W = wave.waves(x)
    t0 = [t[w[0]] for w in W]
    subs, k = [], 0
    while k < len(W):
        j = k
        while j < len(W) and t0[j] - t0[k] < SUB_S:
            j += 1
        if j == len(W):
            break  # no wave starts >= 300 s after this sub-window's first wave: incomplete
        subs.append((k, j))
        k = j
    ttft, e2el, duration = bench_arrays(case / "bench/healthy-32.json")
    start, sim_end = replay_starts(e2el, duration)
    anchor = next(t0[i] for i, w in enumerate(W) if sum(x[k]["fet"] for k in w) > 1)
    arrive = anchor + start
    out = []
    for n, (k, j) in enumerate(subs):
        sp = [w[-1] - w[0] for w in W[k:j]][2:]
        a, b = t0[k], t0[j]
        sel = ttft[(arrive >= a) & (arrive < b)]
        med = st.median(sp) if sp else None
        out.append(dict(i=n, a=a - row["start_wall_s"], b=b - row["start_wall_s"], waves=j - k, medspan=med,
                        locked=None if med is None else med <= LOCK_IT, nreq=len(sel),
                        ttft_ms=float(np.median(sel)) * 1000 if len(sel) else None))
    return dict(subs=out, nwaves=len(W), anchor_rel=anchor - row["start_wall_s"], sim_end=sim_end, duration=duration,
                last_it_rel=t[-1] - row["start_wall_s"], nreq=len(ttft))


def fmt_sub(case: Path, s: dict) -> str:
    state = "n/a" if s["locked"] is None else ("locked" if s["locked"] else "dispersed")
    ttft = "n/a" if s["ttft_ms"] is None else f"{s['ttft_ms']:.0f}"
    med = "n/a" if s["medspan"] is None else f"{s['medspan']:.1f}"
    return (f"SUB {case.parent.name}/{case.name} sub={s['i']} t={s['a']:.0f}-{s['b']:.0f}s waves={s['waves']} "
            f"medspan={med} state={state} nreq={s['nreq']} median_ttft_ms={ttft}")


def case_status(case: Path, kernel_lines) -> str:
    """'' if the case's healthy-32 row is usable, else why not."""
    if not (case / "result.json").exists() or not (case / "rank-0.log").exists():
        return "incomplete: no result.json or rank-0.log"
    rows = {r["label"]: r for r in json.load(open(case / "result.json")).get("rows", [])}
    if "healthy-32" not in rows:
        return "incomplete: no healthy-32 row"
    if rows["healthy-32"].get("rc") != 0:
        return f"incomplete: healthy-32 rc={rows['healthy-32'].get('rc')}"
    if not (case / "bench/healthy-32.json").exists():
        return "incomplete: no bench/healthy-32.json"
    k = kernel_lines(case)
    if k != 0:
        return "excluded: " + ("dmesg unreadable" if k < 0 else f"{k} new Xid/NVRM")
    return ""


def fisher_two_sided(table) -> float:
    try:
        from scipy.stats import fisher_exact

        return float(fisher_exact(table, alternative="two-sided")[1])
    except ImportError:
        from math import comb

        (a, b), (c, d) = table
        r1, c1, n = a + b, a + c, a + b + c + d
        p = lambda x: comb(c1, x) * comb(n - c1, r1 - x) / comb(n, r1)
        p0 = p(a)
        return min(1.0, sum(p(x) for x in range(max(0, r1 + c1 - n), min(r1, c1) + 1) if p(x) <= p0 * (1 + 1e-7)))


def within_state(rows: dict, locked: bool):
    """median over FT sub-windows in the state of their median TTFT minus the same for OFF."""
    vals = {arm: [s["ttft_ms"] for r in rs for s in r if s["locked"] is locked and s["ttft_ms"] is not None]
            for arm, rs in rows.items()}
    if not vals["FT"] or not vals["OFF"]:
        return None
    return st.median(vals["FT"]) - st.median(vals["OFF"])


def bootstrap_ci(rows: dict, locked: bool, n_boot: int = 10000, seed: int = 0):
    rng = np.random.default_rng(seed)
    est, skipped = [], 0
    for _ in range(n_boot):
        res = {arm: [rs[i] for i in rng.integers(0, len(rs), len(rs))] for arm, rs in rows.items()}
        d = within_state(res, locked)
        if d is None:
            skipped += 1
        else:
            est.append(d)
    if not est:
        return None, None, skipped
    return float(np.percentile(est, 2.5)), float(np.percentile(est, 97.5)), skipped


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", metavar="CASE_DIR")
    ap.add_argument("jobs", nargs="*")
    args = ap.parse_args()
    sha = scripts_sha()
    print(f"SCRIPTS_SHA256 {sha} expect {SHA_PREFIX}... {'OK' if sha.startswith(SHA_PREFIX) else 'MISMATCH'}")
    if not sha.startswith(SHA_PREFIX):
        sys.exit(1)
    it, wave = import_scripts()
    kernel_lines = import_soak_stalls().new_kernel_lines
    if args.check:
        case = Path(args.check)
        why = case_status(case, kernel_lines)
        if why:
            sys.exit(f"CHECK {case}: {why}")
        r = row_subwindows(it, wave, case)
        print(f"CHECK {case.parent.name}/{case.name} waves={r['nwaves']} requests={r['nreq']} "
              f"anchor=+{r['anchor_rel']:.0f}s replay_end={r['sim_end']:.1f}s duration={r['duration']:.1f}s "
              f"anchor+duration=+{r['anchor_rel'] + r['duration']:.0f}s last_rank0_iter=+{r['last_it_rel']:.0f}s")
        for s in r["subs"]:
            print(fmt_sub(case, s))
        return
    if not args.jobs:
        ap.error("give job ids or --check CASE_DIR")
    rows = {"FT": [], "OFF": []}
    for job in args.jobs:
        if job in SMOKE_JOBS:
            print(f"SKIP job {job}: smoke job, not part of the soak")
            continue
        cases = sorted(R.glob(f"mvp/runs/STD-{job}-fillseg*/std-online*-1")) + \
            sorted(R.glob(f"mvp/runs/STD-G1-{job}-fillseg*/std-online*-1"))
        if not cases:
            print(f"SKIP job {job}: no fillseg cases")
        for case in cases:
            arm = ARMS.get(re.sub(r"-\d+$", "", case.name))
            if arm is None:
                print(f"SKIP {case.parent.name}/{case.name}: arm not in this design")
                continue
            why = case_status(case, kernel_lines)
            if why:
                print(f"SKIP {case.parent.name}/{case.name}: {why}")
                continue
            r = row_subwindows(it, wave, case)
            subs = [s for s in r["subs"] if s["locked"] is not None and s["ttft_ms"] is not None]
            if not subs:
                print(f"SKIP {case.parent.name}/{case.name}: no complete 300 s sub-window")
                continue
            for s in r["subs"]:
                print(fmt_sub(case, s) + f" arm={arm}")
            rows[arm].append(subs)
    first = {arm: [rs[0] for rs in rows[arm]] for arm in rows}
    tab = [[sum(s["locked"] for s in first[a]), sum(not s["locked"] for s in first[a])] for a in ("FT", "OFF")]
    p = fisher_two_sided(tab) if all(rows.values()) else float("nan")
    print(f"PRIMARY first sub-window per row: FT locked {tab[0][0]}/{sum(tab[0])} | OFF locked {tab[1][0]}/{sum(tab[1])} "
          f"| two-sided Fisher exact p={p:.4f}")
    for a in ("FT", "OFF"):
        allsub = [s for r in rows[a] for s in r]
        print(f"SECONDARY {a}: locked {sum(s['locked'] for s in allsub)}/{len(allsub)} sub-windows over {len(rows[a])} rows")
    cis_zero = True
    for locked, name in ((True, "locked"), (False, "dispersed")):
        d = within_state(rows, locked) if all(rows.values()) else None
        lo, hi, skipped = bootstrap_ci(rows, locked) if d is not None else (None, None, 0)
        if d is None or lo is None:
            cis_zero = False
            print(f"WITHIN {name}: FT - OFF median TTFT n/a (a state has no sub-window in one arm)")
            continue
        cis_zero &= lo <= 0 <= hi
        print(f"WITHIN {name}: FT - OFF median TTFT {d:+.0f} ms, bootstrap 95% CI by row [{lo:+.0f}, {hi:+.0f}] "
              f"(resamples without the state in an arm: {skipped})")
    if p >= 0.05 and cis_zero:
        print("benchmark bistability, no FT TTFT cost at this load")
    elif p < 0.05:
        print("p < 0.05: FT biases runs toward locking")


if __name__ == "__main__":
    main()
