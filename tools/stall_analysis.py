#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""STALL-ATTR readout: which rank's step causes rank 0's far-from-prefill stalls (offline EPLB, 32/GPU).

Input: case dirs whose rank-<r>.log files all carry print_iter_log lines (TLLM_PROFILE_LOG_RANKS=all), plus the case's
result.json (the healthy-<load> row's wall window). Read-only; prints a report and the pre-registered verdict.

Definitions (fixed before data; stall_cmd.txt header):
  window      rank 0's iteration lines whose wall timestamp is in [start_wall_s, start_wall_s + elapsed_s] of the
              healthy-<load> row (as review/diag/offline32_analysis.md and harness steady_state.py:45-50).
  full step   rank 0: num_ctx_requests == 0 and num_active >= ceil(0.9 * load) (29 at 32/GPU).
  stall       a full step with host_step_time > 1.5 x the median host_step_time of the window's full steps.
  far stall   a stall more than 5 iterations from every rank-0 context step (|N - c| > 5 for every c with
              num_ctx_requests > 0 on rank 0, over the whole server segment).
  rank r's context step at iteration j: rank r's line j has num_ctx_requests > 0 and num_scheduled_requests > 0
              (the second guards against iter_states left over from an earlier forward).
  covers      rank r covers far stall N if it has a context step at any j in [N - 2, N + 1] (ALIGN window).
Clock: the iteration index. Every step all 32 ranks join the attention-DP allgather and the MoE all-to-all, so line N
on every rank is the same collective step. The +-1 slack covers the overlap scheduler (a rank's long forward in loop j
is waited on by the other ranks in loop j + 1). Wall timestamps (1 s resolution, node clocks) are a cross-check only.

Usage: stall_analysis.py [--load 32] --ft CASE_DIR... --off CASE_DIR...
"""

import argparse
import ast
import bisect
import datetime
import json
import math
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path

LINE = re.compile(
    r"iter = (\d+), global_rank = (\d+),.*?num_scheduled_requests = (\d+),.*?"
    r"currank_total_requests = (\d+)/(\d+), host_step_time = ([\d.]+)ms, prev_device_step_time = ([\d.]+|N/A)(?:ms)?, "
    r"timestamp = ([\d-]+ [\d:]+), states = (\{[^}]*\}), num_active = (\d+)")
ALIGN = (-2, 1)        # rank r's context step at N-2..N+1 covers rank 0's stall at N
NEAR_CTX = 5           # far = more than 5 iterations from any rank-0 context step
STALL_X = 1.5
GPUS_PER_NODE = 4      # ranks 4n..4n+3 on node n (CUDA_VISIBLE_DEVICES=0,1,2,3, mpirun maps by slot)
ATTRIBUTED, MARGIN, MIN_FAR = 0.80, 0.50, 10


def parse(log: Path) -> list:
    """Server segments (iteration counter restarts start a new one) of a rank's iteration lines."""
    segs = [[]]
    with open(log, errors="replace") as lines:
        for line in lines:
            if "iter = " not in line:
                continue
            m = LINE.search(line)
            if not m:
                continue
            st = ast.literal_eval(m.group(9))
            x = dict(i=int(m.group(1)), sched=int(m.group(3)), cur=int(m.group(4)), tot=int(m.group(5)),
                     ms=float(m.group(6)), dev=None if m.group(7) == "N/A" else float(m.group(7)),
                     wall=datetime.datetime.strptime(m.group(8), "%Y-%m-%d %H:%M:%S").timestamp(),
                     ctx=st.get("num_ctx_requests", 0), ctx_tok=st.get("num_ctx_tokens", 0), act=int(m.group(10)))
            if segs[-1] and x["i"] <= segs[-1][-1]["i"]:
                segs.append([])
            segs[-1].append(x)
    return [s for s in segs if s]


def segment(segs: list, start: float, end: float) -> dict:
    for s in segs:
        if any(start <= x["wall"] <= end for x in s):
            return {x["i"]: x for x in s}
    return {}


def is_ctx(x) -> bool:
    return x is not None and x["ctx"] > 0 and x["sched"] > 0


def case_report(case: Path, load: int) -> dict:
    row = next(r for r in json.loads((case / "result.json").read_text())["rows"] if r.get("label") == f"healthy-{load}")
    start, end = row["start_wall_s"], row["start_wall_s"] + row["elapsed_s"]
    ranks = sorted(int(p.stem.split("-")[1]) for p in case.glob("rank-*.log"))
    by = {r: segment(parse(case / f"rank-{r}.log"), start, end) for r in ranks}
    r0 = by[0]
    ctx0 = sorted(i for i, x in r0.items() if x["ctx"] > 0)
    win = [x for x in r0.values() if start <= x["wall"] <= end]
    floor = math.ceil(0.9 * load)
    full = [x for x in win if x["ctx"] == 0 and x["act"] >= floor]
    med = statistics.median(x["ms"] for x in full)

    def near0(i):
        k = bisect.bisect_left(ctx0, i - NEAR_CTX)
        return k < len(ctx0) and ctx0[k] <= i + NEAR_CTX

    far_full = [x for x in full if not near0(x["i"])]
    stalls = [x for x in full if x["ms"] > STALL_X * med]
    far = [x for x in far_full if x["ms"] > STALL_X * med]
    calm = [x for x in far_full if x["ms"] <= STALL_X * med]
    others = [r for r in ranks if r != 0]

    # Validity: every rank logs rank 0's window iterations, steps in lockstep (lag 0), and wall stamps agree.
    wins = [x["i"] for x in win]
    valid = {}
    for r in others:
        have = sum(1 for i in wins if i in by[r])
        corr = {}
        for lag in (-1, 0, 1):
            pairs = [(math.log(r0[i]["ms"]), math.log(by[r][i + lag]["ms"])) for i in wins if i + lag in by[r]]
            corr[lag] = statistics.correlation(*zip(*pairs)) if len(pairs) > 100 else float("nan")
        dwall = max((abs(by[r][i]["wall"] - r0[i]["wall"]) for i in wins if i in by[r]), default=float("nan"))
        valid[r] = dict(coverage=have / max(1, len(wins)), corr=corr, max_dwall_s=dwall,
                        ok=have >= 0.999 * len(wins) and corr[0] >= max(corr[-1], corr[1]) and dwall <= 2.0)

    def covers(r, n):
        return any(is_ctx(by[r].get(n + d)) for d in range(ALIGN[0], ALIGN[1] + 1))

    cov = {r: sum(covers(r, x["i"]) for x in far) / len(far) if far else float("nan") for r in others}
    base = {r: sum(covers(r, x["i"]) for x in calm) / len(calm) if calm else float("nan") for r in others}
    cov_ex = {r: sum(x["ms"] - med for x in far if covers(r, x["i"])) / max(1e-9, sum(x["ms"] - med for x in far))
              for r in others}
    union = sum(any(covers(r, x["i"]) for r in others) for x in far) / len(far) if far else float("nan")
    union_base = sum(any(covers(r, x["i"]) for r in others) for x in calm) / len(calm) if calm else float("nan")

    # Descriptive: out-of-phase prefill events (a non-rank-0 context step at a rank-0 far full step) and what follows.
    events = [i for i in (x["i"] for x in far_full) if any(is_ctx(by[r].get(i)) for r in others)]
    stall_after = sum(any(r0.get(i + d) and r0[i + d]["ms"] > STALL_X * med for d in (0, 1, 2)) for i in events)
    offs = Counter()
    for x in far:
        for r in others:
            for d in range(-6, 4):
                if is_ctx(by[r].get(x["i"] + d)):
                    offs[d] += 1
    per_node = defaultdict(float)
    for r in others:
        per_node[r // GPUS_PER_NODE] = max(per_node[r // GPUS_PER_NODE], cov[r] if far else 0.0)
    uncovered = [x for x in far if not any(covers(r, x["i"]) for r in others)]
    slowest = Counter()  # uncovered stalls: the rank with the longest prev_device_step_time at N-1..N+1 (descriptive)
    for x in uncovered:
        best = max(others, key=lambda r: max((by[r].get(x["i"] + d) or {}).get("dev") or 0 for d in (-1, 0, 1)))
        slowest[best] += 1
    return dict(case=str(case), window_s=row["elapsed_s"], median_ms=med, full=len(full), stalls=len(stalls),
                far=len(far), far_excess_s=sum(x["ms"] - med for x in far) / 1e3, calm=len(calm),
                valid=valid, cov=cov, base=base, cov_ex=cov_ex, union=union, union_base=union_base,
                events=len(events), events_with_stall=stall_after, offsets=dict(sorted(offs.items())),
                per_node=dict(per_node), uncovered=len(uncovered), uncovered_slowest_dev=dict(slowest.most_common(5)))


def verdict(reps: list) -> str:
    """Pre-registered rule (stall_cmd.txt): per case with >= MIN_FAR far stalls, then over the arm's cases pooled."""
    reps = [r for r in reps if r["far"] >= MIN_FAR]
    if not reps:
        return "INCONCLUSIVE (no case with >= 10 far stalls: the mode did not occur)"
    if any(not v["ok"] for r in reps for v in r["valid"].values()):
        return "INVALID (a rank failed the coverage / lockstep / wall-clock check; see VALID lines)"
    far = sum(r["far"] for r in reps)
    pooled = {k: sum(r["cov"][k] * r["far"] for r in reps) / far for k in reps[0]["cov"]}
    pooled_base = {k: sum(r["base"][k] * r["calm"] for r in reps) / max(1, sum(r["calm"] for r in reps)) for k in pooled}
    top = max(pooled, key=pooled.get)
    single = [k for k in pooled if pooled[k] >= ATTRIBUTED and pooled[k] - pooled_base[k] >= MARGIN]
    union = sum(r["union"] * r["far"] for r in reps) / far
    union_base = sum(r["union_base"] * r["calm"] for r in reps) / max(1, sum(r["calm"] for r in reps))
    head = f"pooled over {len(reps)} case(s), {far} far stalls: top rank {top} covers {pooled[top]:.0%} " \
           f"(baseline {pooled_base[top]:.0%}); any non-0 rank covers {union:.0%} (baseline {union_base:.0%})"
    if single:
        per_case = all(r["cov"][single[0]] >= ATTRIBUTED for r in reps)
        return f"ATTRIBUTED to rank {single[0]} (node {single[0] // GPUS_PER_NODE})" + \
               ("" if per_case else " [pooled only: not every case reaches 80%]") + f"; {head}"
    if union >= ATTRIBUTED and union - union_base >= MARGIN:
        return f"SPREAD-PREFILL (other ranks' prefill, no single rank); {head}"
    if union >= 0.5:
        return f"PARTLY-PREFILL (50-80% covered by other ranks' prefill); {head}"
    return f"NOT-PREFILL (look elsewhere: uncovered stalls' slowest-device ranks below); {head}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--load", type=int, default=32)
    ap.add_argument("--ft", nargs="*", default=[])
    ap.add_argument("--off", nargs="*", default=[])
    args = ap.parse_args()
    for arm, cases in (("FT", args.ft), ("OFF", args.off)):
        reps = [case_report(Path(c), args.load) for c in cases]
        for r in reps:
            bad = {k: v for k, v in r["valid"].items() if not v["ok"]}
            top = sorted(r["cov"], key=r["cov"].get, reverse=True)[:5]
            print(f"{arm} {r['case']}: window {r['window_s']:.0f} s, median {r['median_ms']:.3f} ms, full {r['full']}, "
                  f"stalls {r['stalls']}, far {r['far']} ({r['far_excess_s']:.2f} s excess)")
            print(f"  VALID {len(r['valid']) - len(bad)}/{len(r['valid'])} ranks ok"
                  + (f"; FAILED {json.dumps(bad, default=str)[:400]}" if bad else ""))
            if r["far"]:
                print("  top ranks (cover / baseline / excess-weighted): " + ", ".join(
                    f"r{k} {r['cov'][k]:.0%}/{r['base'][k]:.0%}/{r['cov_ex'][k]:.0%}" for k in top))
                print(f"  any non-0 rank: {r['union']:.0%} (baseline {r['union_base']:.0%}); "
                      f"offsets ctx-step minus stall: {r['offsets']}; uncovered {r['uncovered']} "
                      f"(slowest device step: {r['uncovered_slowest_dev']})")
            print(f"  out-of-phase prefill events {r['events']}, followed by a stall within 2 steps {r['events_with_stall']}")
        print(f"VERDICT {arm}: {verdict(reps)}")


if __name__ == "__main__":
    main()
