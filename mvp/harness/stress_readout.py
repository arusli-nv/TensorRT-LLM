#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Readout of the product stress tests (review/PRODUCT_STRESS_PLAN.md; launch files review/drafts/stress_*_cmd.txt).

Usage:
  stress_readout.py case CASE_DIR [--test s2|s3|s1]   facts and checks of one stress case (bench.py --mode stress)
  stress_readout.py pair OFF_DIR ON_DIR                S1: one FT off / FT on pair on the same nodes
Benchmark-only (never part of TRT-LLM). Every rule below was written before any stress data (2026-10-25).

Client records (stress_client.py, bench/<row>.sse.jsonl), one per request:
  ok         HTTP 200, no error event, exactly one [DONE] and nothing after it, completion_tokens == requested.
  retryable  HTTP 200, exactly one error event {type: server_error, code: peer_loss_retryable, message with
             "[peer_loss_retryable]"}, then exactly one [DONE] and nothing after it (the documented format), or HTTP 503
             with Retry-After (a non-streamed answer).
  bad        anything else: another error format, an exception (reset, timeout), a stream without [DONE], or a silent
             short stream (no error, fewer tokens than requested).
A failed request belongs to kill k if its error names a rank lost at k ("Attention DP rank R failed ...") or is a
second-loss quarantine ("interrupted this request a second time", kills after the first), and it ended between k's
kill and 60 s after k's resume.
Windows (open-loop row "stress"): served output tokens per second from each stream's chunk arrivals (wall second) x its
tokens per chunk. Kill k: before = [max(load start + 300 s, previous resume + 120 s), kill); after = [resume + 60 s,
min(resume + 1260 s, next trigger, last arrival)).
Stall: two consecutive rank-0 iterations more than 60 s apart, or no streamed chunk for more than 60 s while a request
was open, inside [load start, last arrival].
Memory (WIDEEP_MEM_SAMPLER_S samples, mem-<node>.log): growth = median of the last 5 samples - median of the first 5
in [load start + 600 s, last arrival], per GPU (MiB), per rank process RssAnon (MiB) and per node MemAvailable (MiB).
"""

import argparse
import datetime
import json
import re
import statistics
import sys
from pathlib import Path

try:
    from zoneinfo import ZoneInfo

    TZ = ZoneInfo("America/Los_Angeles")  # rank logs' iteration timestamps are node-local Pacific time
except Exception:  # no tz database: the zone the readout runs in
    TZ = None
RETRYABLE = "peer_loss_retryable"
OWNER = re.compile(r"Attention DP rank (\d+) failed")
QUARANTINE = "interrupted this request a second time"
RESUMED = re.compile(r"resumed serving without ranks \[([0-9, ]*)\] at membership generation (\d+)")
RERAN = re.compile(r"re-ran (\d+) requests and failed (\d+)")
ITER = re.compile(r"\[RANK 0\] iter = \d+, .*?timestamp = (\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)")
GAP_S = 60.0
# Proposed memory-growth thresholds (the plan says only "no memory growth"): confirm with the user before submit.
MEM_GPU_MIB, MEM_RSS_MIB, MEM_HOST_MIB = 256, 512, 2048


def load_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return {}


def records(case: Path, row: str) -> list:
    path = case / "bench" / f"{row}.sse.jsonl"
    out = []
    for line in path.read_text().splitlines() if path.exists() else []:
        try:
            out.append(json.loads(line))
        except ValueError:
            out.append({"exc": "unparsable record"})
    return out


def classify(rec: dict) -> str:
    errors, status = rec.get("errors") or [], rec.get("status")
    if status == 503 and rec.get("retry_after"):
        return "retryable"
    if status != 200 or rec.get("exc") or rec.get("done") != 1 or rec.get("after_done"):
        return "bad"
    if not errors:
        return "ok" if rec.get("chunks") and rec.get("tok") == rec.get("exp") else "bad"
    e = errors[0]
    well_formed = (len(errors) == 1 and isinstance(e, dict) and e.get("type") == "server_error"
                   and e.get("code") == RETRYABLE and f"[{RETRYABLE}]" in str(e.get("message")))
    return "retryable" if well_formed else "bad"


def message(rec: dict) -> str:
    return " ".join(str(e.get("message", e)) if isinstance(e, dict) else str(e) for e in rec.get("errors") or []) \
        or str(rec.get("body") or rec.get("exc") or "")


def rank0_iterations(case: Path) -> list:
    out, log = [], case / "rank-0.log"
    for line in log.read_text(errors="replace").splitlines() if log.exists() else []:
        if (m := ITER.search(line)):
            t = datetime.datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S").replace(tzinfo=TZ).timestamp()
            out.append(t)
    return out


def chunk_seconds(recs: list) -> dict:
    """Wall second -> served output tokens (each stream's chunks x its tokens per chunk)."""
    per_chunk = [r["tok"] / r["chunks"] for r in recs if r.get("tok") and r.get("chunks")]
    default = statistics.median(per_chunk) if per_chunk else 1.0
    out = {}
    for r in recs:
        factor = r["tok"] / r["chunks"] if r.get("tok") and r.get("chunks") else default
        for second, n in (r.get("ts") or {}).items():
            out[int(second)] = out.get(int(second), 0.0) + n * factor
    return out


def rate(seconds: dict, start: float, end: float):
    if end - start < 60:
        return None
    return sum(v for s, v in seconds.items() if start <= s < end) / (end - start)


def pct(values: list, q: float):
    if not values:
        return None
    values = sorted(values)
    return round(values[min(len(values) - 1, int(q * len(values)))] * 1000, 1)


def latencies(recs: list, start: float, end: float) -> dict:
    ok = [r for r in recs if classify(r) == "ok" and start <= r["start"] < end]
    ttft = [r["ttft"] for r in ok if r.get("ttft") is not None]
    tpot = [(r["end"] - r["start"] - r["ttft"]) / (r["tok"] - 1) for r in ok if r.get("ttft") is not None and r["tok"] > 1]
    return dict(n=len(ok), ttft_p50_ms=pct(ttft, 0.5), ttft_p99_ms=pct(ttft, 0.99), tpot_p50_ms=pct(tpot, 0.5),
                tpot_p99_ms=pct(tpot, 0.99))


def stalls(case: Path, recs: list, start: float, end: float) -> list:
    out = []
    iters = [t for t in rank0_iterations(case) if start <= t <= end]
    out += [("rank0", a, b - a) for a, b in zip(iters, iters[1:]) if b - a > GAP_S]
    seconds = sorted(s for s, v in chunk_seconds(recs).items() if v and start <= s <= end)
    open_at = lambda t: any(r["start"] <= t <= r.get("end", t) for r in recs)
    out += [("client", a, b - a) for a, b in zip(seconds, seconds[1:]) if b - a > GAP_S and open_at(a + 1)]
    return out


def memory(case: Path, start: float, end: float) -> dict:
    series = {}
    for log in sorted(case.glob("mem-*.log")):
        node = log.stem.split("-")[1]
        for line in log.read_text(errors="replace").splitlines():
            f = line.split()
            if len(f) < 3 or not f[0].isdigit() or not start <= int(f[0]) <= end:
                continue
            t, kind = int(f[0]), f[1]
            kv = dict(x.split("=", 1) for x in f[2:] if "=" in x)
            if kind == "gpu" and len(f) >= 4:
                series.setdefault(("gpu", f"n{node}g{f[2]}"), []).append((t, float(f[3])))
            elif kind == "host" and "MemAvailable" in kv:
                series.setdefault(("host", f"n{node}"), []).append((t, -float(kv["MemAvailable"]) / 1024))
            elif kind == "proc" and kv.get("rank", "-") != "-" and "RssAnon" in kv:
                series.setdefault(("rss", f"n{node}r{kv['rank']}p{f[2]}"), []).append((t, float(kv["RssAnon"]) / 1024))
    growth = {"gpu": {}, "rss": {}, "host": {}}
    for (kind, key), points in series.items():
        if len(points) >= 10:
            values = [v for _, v in sorted(points)]
            growth[kind][key] = round(statistics.median(values[-5:]) - statistics.median(values[:5]), 1)
    # ok needs a growth value of every kind (>= 10 samples of a series); too few samples is not a pass.
    top = {k: max(v.items(), key=lambda kv: kv[1]) if v else None for k, v in growth.items()}
    return dict(samples=sum(len(p) for p in series.values()), gpu_max=top["gpu"], rss_max=top["rss"],
                host_used_max=top["host"],
                ok=all(growth.values()) and all(v <= MEM_GPU_MIB for v in growth["gpu"].values())
                and all(v <= MEM_RSS_MIB for v in growth["rss"].values())
                and all(v <= MEM_HOST_MIB for v in growth["host"].values()))


def health_runs(case: Path) -> list:
    """[[state or code, http code, first t, last t, samples, max ms], ...] from health.jsonl."""
    runs, path = [], case / "health.jsonl"
    for line in path.read_text().splitlines() if path.exists() else []:
        try:
            h = json.loads(line)
        except ValueError:
            continue
        key = h.get("state") or str(h.get("code"))
        if runs and runs[-1][0] == key:
            runs[-1][3], runs[-1][4], runs[-1][5] = h["t"], runs[-1][4] + 1, max(runs[-1][5], h.get("ms") or 0)
        else:
            runs.append([key, h.get("code"), h["t"], h["t"], 1, h.get("ms") or 0])
    return runs


def case_readout(case: Path, test: str) -> bool:
    name = case.name
    result = load_json(case / "result.json")
    rows = {r.get("label"): r for r in result.get("rows", [])}
    kills = result.get("kills") or []
    load_start = result.get("load_start_wall_s")
    recs = records(case, "stress") + records(case, "stress-chat")
    calib = records(case, "calib-32") + records(case, "calib-end-32")
    last_arrival = max((r["start"] for r in recs if "start" in r), default=None)
    log0 = (case / "rank-0.log").read_text(errors="replace") if (case / "rank-0.log").exists() else ""
    resumes, reran = RESUMED.findall(log0), [(int(a), int(b)) for a, b in RERAN.findall(log0)]
    timeline = (case / "timeline").read_text(errors="replace") if (case / "timeline").exists() else ""
    mpirun_exit = next((float(l.split()[1]) for l in timeline.splitlines() if l.startswith("MPIRUN_EXIT ")), None)
    classes = {}
    for r in recs:
        classes[classify(r)] = classes.get(classify(r), 0) + 1
    calib_bad = sum(classify(r) != "ok" for r in calib)
    print("STRESS_CASE", name, f"state={result.get('state')}", f"error={result.get('error')}",
          f"rate={round(result['stress_rate'], 2) if result.get('stress_rate') else None}",
          f"requests={len(recs)}", *(f"{k}={v}" for k, v in sorted(classes.items())),
          f"calib_bad={calib_bad}", f"chat_client_error={result.get('chat_client_error')}",
          *(f"{label}_tok_s={round(rows[label]['output_throughput'])}" for label in ("calib-32", "calib-end-32")
            if rows.get(label, {}).get("output_throughput")),
          f"load_start={load_start}", f"last_arrival={last_arrival}", f"mpirun_exit={mpirun_exit}")
    checks = {}
    if not recs or load_start is None or last_arrival is None:
        print("STRESS_VERDICT", name, test, "INVALID no open-loop records")
        return False
    seconds = chunk_seconds(recs)
    lost, assigned = [], set()
    for k, kill in enumerate(kills, 1):
        lost += kill["ranks"]
        t_kill, t_res = kill.get("kill_wall_s"), kill.get("resumed_seen_wall_s")
        prev_res = kills[k - 2].get("resumed_seen_wall_s") if k > 1 else None
        nxt = kills[k].get("trigger_wall_s") if k < len(kills) else None
        mine = []
        for i, r in enumerate(recs):
            if i in assigned or classify(r) == "ok" or t_kill is None:
                continue
            owner = OWNER.search(message(r))
            ends_in = t_kill - 1 <= r.get("end", 0) <= (t_res or mpirun_exit or t_kill + 300) + 60
            if ends_in and ((owner and int(owner.group(1)) in kill["ranks"]) or (k > 1 and QUARANTINE in message(r))):
                assigned.add(i)
                mine.append(r)
        server_failed = reran[k - 1][1] if k <= len(reran) else None
        before = rate(seconds, max(load_start + 300, (prev_res or 0) + 120), t_kill) if t_kill else None
        after_end = min(x for x in (t_res + 1260 if t_res else None, nxt, last_arrival) if x is not None)
        after = rate(seconds, t_res + 60, after_end) if t_res else None
        floor = 0.75 if len(kill["ranks"]) == 4 else 0.85
        ratio = round(after / before, 3) if before and after else None
        # /health around the loss: recovering (503) then degraded (200), nothing else.
        seen = [run for run in health_runs(case) if t_kill and t_kill - 1 <= run[3] and run[2] <= (t_res or t_kill) + 60]
        states = [run[0] for run in seen]
        rec_runs = [run for run in seen if run[0] == "recovering"]
        health_ok = ("recovering" in states and "degraded" in states[states.index("recovering"):]
                     and set(states) <= {"healthy", "degraded", "recovering"}
                     and all(run[1] == 503 for run in rec_runs)
                     and all(run[1] == 200 for run in seen if run[0] in ("healthy", "degraded")))
        resumed_ok = len(resumes) >= k and sorted(int(x) for x in resumes[k - 1][0].split(",") if x.strip()) == sorted(lost)
        print("STRESS_KILL", name, f"k={k}", f"ranks={'+'.join(map(str, kill['ranks']))}", f"kind={kill['kind']}",
              f"at_s={round(t_kill - load_start, 1) if t_kill else None}",
              f"resume_s={round(t_res - t_kill, 2) if t_res and t_kill else None}",
              f"recovery_error={kill.get('recovery_error')}", f"resumed_line={resumed_ok}",
              f"client_failed={len(mine)}", f"server_failed={server_failed}",
              f"reran={reran[k - 1][0] if k <= len(reran) else None}",
              f"tok_s_before={round(before) if before else None}", f"tok_s_after={round(after) if after else None}",
              f"ratio={ratio}", f"floor={floor}", f"after_window_s={round(after_end - t_res - 60) if t_res else None}",
              f"health={'>'.join(states)}", f"recovering_s={round(rec_runs[-1][3] - rec_runs[0][2], 2) if rec_runs else None}",
              f"health_max_ms={max((run[5] for run in seen), default=None)}")
        checks[f"k{k}_recovery_commits"] = resumed_ok and not kill.get("recovery_error")
        checks[f"k{k}_failed_match_server"] = server_failed is not None and len(mine) <= server_failed
        checks[f"k{k}_tput"] = ratio is not None and ratio >= floor
        checks[f"k{k}_health"] = health_ok
    unassigned = [r for i, r in enumerate(recs) if i not in assigned and classify(r) != "ok"]
    malformed = [r for r in recs if classify(r) == "bad"]
    for r in unassigned[:10]:
        print("STRESS_UNASSIGNED", name, classify(r), f"start={r.get('start')}", f"end={r.get('end')}",
              f"status={r.get('status')}", f"tok={r.get('tok')}/{r.get('exp')}", f"done={r.get('done')}",
              f"msg={message(r)[:160]!r}")
    end = last_arrival if mpirun_exit is None else min(last_arrival, mpirun_exit)
    st = stalls(case, recs, load_start, end)
    for kind, t, gap in st:
        print("STRESS_STALL", name, kind, f"at_s={round(t - load_start)}", f"gap_s={round(gap)}")
    last_res = max((k.get("resumed_seen_wall_s") or 0 for k in kills), default=0)
    lat = latencies(recs, load_start + 300, kills[0]["kill_wall_s"] if kills and kills[0].get("kill_wall_s") else end)
    print("STRESS_LATENCY", name, "healthy", *(f"{k}={v}" for k, v in lat.items()))
    if kills and last_res:
        print("STRESS_LATENCY", name, "after_last_resume", *(f"{k}={v}" for k, v in latencies(recs, last_res + 60, end).items()))
    mem = memory(case, load_start + 600, end)
    print("STRESS_MEMORY", name, *(f"{k}={v}" for k, v in mem.items()), f"thresholds_mib=gpu:{MEM_GPU_MIB},rss:{MEM_RSS_MIB},host:{MEM_HOST_MIB}")
    checks["no_other_request_failed"] = not unassigned
    checks["errors_well_formed"] = not malformed and calib_bad == 0
    checks["no_stall"] = not st
    if test == "s1":
        checks = dict(no_stall=not st, no_client_failure=not unassigned and not malformed and calib_bad == 0,
                      no_memory_growth=mem["ok"], completed=result.get("state") == "DONE")
    elif test == "s3":
        # Both recover, or the job ends cleanly (mpirun exits on its own within 180 s of a loss the slots cannot
        # cover; every node cleaned up); no hang.
        failed = [k for k in kills if k.get("recovery_error")]
        clean_end = bool(failed) and mpirun_exit is not None and failed[0].get("kill_wall_s") is not None \
            and mpirun_exit - failed[0]["kill_wall_s"] <= 180 \
            and len(set(re.findall(r"NODE_CLEAN (\d+)", timeline))) >= len(list(case.glob("agent-*.log")))
        checks = dict(both_recover_or_clean_end=(len(kills) == 2 and all(checks.get(f"k{k}_recovery_commits") for k in (1, 2)))
                      or clean_end, no_hang=not st, no_other_request_failed=not unassigned,
                      errors_well_formed=not malformed)
    elif test == "s2":
        checks["one_kill"] = len(kills) == 1
        # The open-loop ratio stays ~1 while the survivors keep up with the offered rate; the closed-loop rows compare
        # capacity at 32/GPU before the kill and after it (same 1024 streams), against the same floor.
        cal, cal_end = (rows.get(x, {}).get("output_throughput") for x in ("calib-32", "calib-end-32"))
        floor = 0.75 if kills and len(kills[0]["ranks"]) == 4 else 0.85
        capacity = round(cal_end / cal, 3) if cal and cal_end else None
        print("STRESS_CAPACITY", name, f"calib_tok_s={cal}", f"calib_end_tok_s={cal_end}", f"ratio={capacity}", f"floor={floor}")
        checks["k1_capacity"] = capacity is not None and capacity >= floor
    bad = [k for k, v in checks.items() if not v]
    print("STRESS_VERDICT", name, test, "PASS" if not bad else "FAIL", *(f"{k}={v}" for k, v in checks.items()))
    return not bad


def pair_readout(off: Path, on: Path) -> None:
    """S1: FT on vs off on the same nodes: closed-loop throughput, open-loop served throughput and latency."""
    out = {}
    for arm, case in (("off", off), ("on", on)):
        result = load_json(case / "result.json")
        rows = {r.get("label"): r for r in result.get("rows", [])}
        recs = records(case, "stress") + records(case, "stress-chat")
        start = result.get("load_start_wall_s") or 0
        end = max((r["start"] for r in recs if "start" in r), default=start)
        out[arm] = dict(calib=(rows.get("calib-32") or {}).get("output_throughput"),
                        calib_end=(rows.get("calib-end-32") or {}).get("output_throughput"),
                        served=rate(chunk_seconds(recs), start + 300, end), **latencies(recs, start + 300, end))
    diff = lambda k: (round(100 * (out["on"][k] / out["off"][k] - 1), 2)
                      if out["on"].get(k) and out["off"].get(k) else None)
    print("STRESS_PAIR", off.name, on.name, *(f"{arm}_{k}={round(v, 1) if isinstance(v, float) else v}"
                                            for arm in ("off", "on") for k, v in out[arm].items()),
          *(f"on_vs_off_pct_{k}={diff(k)}" for k in ("calib", "calib_end", "served", "ttft_p50_ms", "ttft_p99_ms",
                                                    "tpot_p50_ms", "tpot_p99_ms")))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    one = sub.add_parser("case")
    one.add_argument("case", type=Path)
    one.add_argument("--test", choices=("s1", "s2", "s3"), default=None)
    two = sub.add_parser("pair")
    two.add_argument("off", type=Path)
    two.add_argument("on", type=Path)
    args = parser.parse_args()
    if args.cmd == "case":
        result = load_json(args.case / "result.json")
        test = args.test or ("s1" if not result.get("kills") else "s3" if len(result["kills"]) > 1 else "s2")
        sys.exit(0 if case_readout(args.case, test) else 1)
    pair_readout(args.off, args.on)


if __name__ == "__main__":
    main()
