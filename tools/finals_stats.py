#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Healthy-cost finals statistics, exactly as review/FINALS_PLAN.md PRE-REG 1-8.

Input: group run dirs (mvp/runs/STD[-G<k>]-<job>). A group is the 8 nodes that ran a run dir, found from the
bundle log whose task ran `run_job.sh STD[-G<k>]` (several run dirs may share one group). Arms are case dirs
std-<mode>[-ft|-main|-lb|-ulfm]-<r>, mode in noeplb/offline/online. Only <mode> (FT off) vs <mode>-ft (FT on) form
the pre-registered pairs; -main enters only the descriptive parity tables; -lb / -ulfm are ignored here.
1. Unit = one group. A pair = the off and ft arms of one mode in that group (rounds of one group are averaged into
   its single pair; finals run one round). Order = which arm's healthy window started first.
5. Validity is decided per arm before any cost is computed, from logs only: PIN_CHECK OK and cache_miss equal
   between the pair's arms (tactic_check.txt); no ListenOverflows increase on any node of the group during the case
   (CASE <arm> net before/after in the group's bundle logs); result.json state DONE and a healthy-<load> row with
   rc 0 for every --loads load; >= 20 steady steps (dist32.case_stats); rank logs present and no hang-detector /
   peer-loss / crash line in them, and no new Xid/NVRM kernel line where kernel-*.txt was recorded; every row
   all planned waves completed (concurrency x iterations requests; user decision 10-23).
6. An invalid arm drops its whole pair, at every load (printed with reasons). Reruns enter as extra group dirs.
2. Per pair and load: % = (on - off) / off of the trim mean decode step, the full mean step (mean of all steady
   steps), stall ms/step (dist32: steps > 1.5x median, the one stall threshold), client output tok/s/GPU over the
   bench run (result.json) and rank-0 generated tok/s/GPU over the row window (ctx_stall_cost.case_metrics). Step
   metrics: positive = FT slower; tok/s: negative = FT slower.
2-4. Cell = mode x headline load (--loads): n, mean, 95% t-CI (df n-1), one-sided 95% upper bound. "FT cost < 1%" on
   a step metric only if every headline cell (all modes x --loads) has an upper bound < 1% (IUT); the per-step claim
   is reported under rule A (trim AND full mean, pre-registered) and rule B (trim only, added 10-23 after the
   rehearsal reading), side by side (user decision 10-23). Client tok/s: estimate +/- CI only. Per-mode summary averages a pair's
   headline loads first (pairs with every headline load only); modes are never pooled. --appendix-loads: cells only.
7. Order (off first vs ft first) and node-0 vs node-8 group means are reported as checks only.
8. Parity (our FT off vs main) and FT on vs main: same group, descriptive, own tables; not part of the claim.
Usage: python3 -I tools/finals_stats.py [--window-s S] [--loads 32,128] [--appendix-loads 8,512] [--csv OUT] RUN_DIR...
"""
import argparse
import csv
import importlib.util
import json
import math
import pathlib
import re
import statistics
import subprocess
import sys
import time

C = pathlib.Path(__file__).resolve().parents[1]


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


dist32 = _load("dist32", C / "mvp" / "analysis" / "dist32.py")
sys.path.insert(0, str(C / "mvp" / "analysis"))
import ctx_stall_cost as csc  # noqa: E402  (imports harness finals_readout: t_ppf, tactic_status)

fr = csc.fr
MODES = ("noeplb", "offline", "online")
ARM = re.compile(r"std-(noeplb|offline|online)(-ft|-main|-lb|-ulfm)?-(\d+)")
FAULT = re.compile(r"Hang detected|HangDetector:|a peer failed|hung rank|Executor loop crashed|"
                   r"peer failed before serving")
STAMP = re.compile(r"\[(\d\d/\d\d/\d{4}-\d\d:\d\d:\d\d)\]")
METRICS = (("trim", "trim mean step"), ("full", "full mean step"), ("stall", "stall ms/step"),
           ("client", "client tok/s/GPU"), ("window", "rank-0 window tok/s/GPU"))
STEP = ("trim", "full")
LIMIT = 1.0
GROUP_NODES = 8  # one DEP32 group
# (base variant, compared variant, title): % = (compared - base) / base. Only PAIRS enters the claim.
PAIRS = ("", "-ft", "FT off -> FT on")
DESCRIPTIVE = (("-main", "", "main -> our FT off (parity)"), ("-main", "-ft", "main -> our FT on"),
               ("", "-ulfm", "our FT off -> FT off over ob1/ULFM (transport)"),
               ("", "-lb", "our FT off -> FT off with the static LB (floor)"))


def bundle_logs(run):
    """The bundle logs of the group whose task ran RUN (its `run_job.sh <prefix>` line) and its first node."""
    m = re.fullmatch(r"(STD(?:-G\d+)?)-(\d+)", run.name)
    if not m:
        return [], None
    task = re.compile(rf"run_job\.sh {m[1]}(?![-\w])")
    for log in sorted((C / "logs" / "bundle").glob(f"{m[2]}-*.log")):
        if task.search(log.read_text(errors="replace")):
            return sorted(log.parent.glob(log.name + "*")), int(log.name[len(m[2]) + 1:-len(".log")])
    return [], None


def overflow_increase(logs, case):
    """Per node: ListenOverflows after - before for this case; None if no node recorded both."""
    rise, seen = 0, False
    for log in logs:
        vals = {}
        for line in log.read_text(errors="replace").splitlines():
            m = re.match(rf"CASE {re.escape(case)} net (before|after): .*ListenOverflows=(\d+)", line)
            if m:
                vals[m[1]] = int(m[2])
        if len(vals) == 2:
            seen = True
            rise = max(rise, vals["after"] - vals["before"])
    return rise if seen else None


def kernel_new_lines(case):
    """Xid/NVRM lines after the case that were not there before (kernel-<node>-{before,after}.txt), None if any node's
    log is unrecorded or unreadable (a check that could not run is not a pass)."""
    new, nodes = 0, 0
    for after in case.glob("kernel-*-after.txt"):
        before = after.with_name(after.name.replace("-after", "-before"))
        if not before.exists():
            return None
        lines = [p.read_text(errors="replace").splitlines() for p in (before, after)]
        if not all(ls and ls[0].startswith("dmesg ok") for ls in lines):
            return None
        nodes += 1
        old = set(lines[0][1:])
        new += sum(line not in old for line in lines[1][1:])
    return new if nodes == GROUP_NODES else None


def fault_lines(logs, end):
    """Hang-detector / peer-loss / crash lines in the rank logs: (those stamped before the last row's end, or
    unstamped; the count after it, i.e. at server teardown)."""
    out = subprocess.run(["grep", "-h", "-E", FAULT.pattern, *map(str, logs)], stdin=subprocess.DEVNULL,
                         capture_output=True, text=True).stdout.splitlines()
    during, after = [], 0
    for line in out:
        m = STAMP.match(line)
        # Log stamps are whole PT seconds (finals_readout sets TZ): a line in the row's last second is teardown.
        if m and time.mktime(time.strptime(m[1], "%m/%d/%Y-%H:%M:%S")) + 1 > end:
            after += 1
        else:
            during.append(line)
    return during, after


def arm_record(run, case, tactics, logs, window_s, loads):
    """Metrics per load and the reasons (blind to cost) the arm is invalid; any reason invalidates the whole arm."""
    why = []
    pin, _ = tactics.get(case.name, ("MISSING", None))
    if pin != "OK":
        why.append(f"PIN_CHECK {pin}")
    miss = re.search(rf"TACTIC_CHECK {re.escape(case.name)} .*?cache_miss=(\d+)",
                     (run / "tactic_check.txt").read_text() if (run / "tactic_check.txt").exists() else "")
    rise = overflow_increase(logs, case.name)
    if rise is None:
        why.append("ListenOverflows not recorded")
    elif rise > 0:
        why.append(f"ListenOverflows +{rise}")
    result = case / "result.json"
    doc = json.loads(result.read_text()) if result.exists() else {}
    if doc.get("state") != "DONE":
        why.append(f"result state {doc.get('state', 'missing')}")
    ranks = sorted(case.glob("rank-*.log"))
    if not ranks:
        why.append("no rank logs")
    end = max((r["start_wall_s"] + r["elapsed_s"] for r in doc.get("rows", [])), default=math.inf)
    faults, teardown = fault_lines(ranks, end) if ranks else ([], 0)
    if faults:
        why.append(f"{len(faults)} fault lines before the last row ended, first: {faults[0][:160]}")
    xid = kernel_new_lines(case)
    if xid:
        why.append(f"{xid} new Xid/NVRM kernel lines")
    rows = {r["load_per_gpu"]: r for r in doc.get("rows", []) if r.get("label", "").startswith("healthy-")}
    stats = dist32.case_stats(case) if rows and (case / "rank-0.log").exists() else {}
    why += [f"healthy-{load} row missing" for load in loads if load not in rows]
    metrics = {}
    for load, row in rows.items():
        s = stats.get(row["label"])
        if row.get("rc") != 0:
            why.append(f"healthy-{load} rc {row.get('rc')}")
        if not s:
            why.append(f"healthy-{load} < 20 steady steps")
        planned = row["concurrency"] * row["iterations"]
        if row["completed"] < planned:
            why.append(f"healthy-{load} completed {row['completed']} of {planned} planned requests (waves incomplete)")
        if window_s and row["elapsed_s"] < window_s:
            why.append(f"healthy-{load} window {row['elapsed_s']:.0f} s < {window_s:.0f} s")
        if s and row.get("rc") == 0:
            w = csc.case_metrics(case, row)
            metrics[load] = {"trim": s["trim_mean"], "full": s["mean"], "stall": s["stall_ms_per_step"],
                             "client": row["output_throughput"] / row["gpus"], "window": w["tok_s"] / row["gpus"],
                             "steps": s["n"], "start": row["start_wall_s"]}
    return {"why": why, "cache_miss": int(miss[1]) if miss else None, "loads": metrics, "xid": xid,
            "teardown": teardown}


def collect(runs, window_s, loads):
    """groups: group -> {(mode, round, variant): (arm path, record)}; a group is (job, first node), so run dirs of
    one group merge. teardown: [(run, arm, fault lines after the last row)]."""
    groups, node, teardown, xid_unchecked = {}, {}, [], []
    for run in runs:
        tactics, (logs, first) = fr.tactic_status(run), bundle_logs(run)
        g = f"{run.name.rsplit('-', 1)[1]}@n{first}" if first is not None else run.name
        node[g] = "-" if first is None else f"n{first}"
        for case in sorted(p for p in run.iterdir() if p.is_dir() and ARM.fullmatch(p.name)):
            mode, variant, rnd = ARM.fullmatch(case.name).groups()
            key = (mode, rnd, variant or "")
            if key in groups.setdefault(g, {}):
                sys.exit(f"{case.name} twice in group {g} ({run.name})")
            rec = arm_record(run, case, tactics, logs, window_s, loads)
            groups[g][key] = (f"{run.name}/{case.name}", rec)
            if rec["xid"] is None:
                xid_unchecked.append(f"{run.name}/{case.name}")
            if rec["teardown"]:
                teardown.append((run.name, case.name, rec["teardown"]))
    return groups, node, teardown, xid_unchecked


def pair_up(groups, node, base, other, need=("", "-ft"), shared_loads=False):
    """pairs: (group, node, mode, load) -> per-round (base, other) metrics; dropped: [(group, mode, reasons)].
    Rounds are paired where any arm of NEED ran; a missing arm of the pair is then reported. The off/on pair runs
    one load sweep per arm, so its load sets must match; descriptive pairs (shared_loads) compare the loads both
    arms ran, as a main arm runs the headline loads only next to an off arm that sweeps the appendix loads too."""
    pairs, dropped = {}, []
    for g, arms in groups.items():
        for mode, rnd in sorted({(m, r) for m, r, v in arms if v in need}):
            (na, a), (nb, b) = (arms.get((mode, rnd, v), (f"std-{mode}{v}-{rnd}", None)) for v in (base, other))
            if a is None or b is None:
                if shared_loads and a is not None:
                    continue  # a descriptive arm this group does not run by design
                dropped.append((g, mode, f"{na if a is None else nb}: arm missing (its pair arm ran)"))
                continue
            why = [f"{na}: {w}" for w in a["why"]] + [f"{nb}: {w}" for w in b["why"]]
            if a["cache_miss"] != b["cache_miss"]:
                why.append(f"cache_miss {a['cache_miss']} vs {b['cache_miss']}")
            common = set(a["loads"]) & set(b["loads"])
            if (not common) if shared_loads else set(a["loads"]) != set(b["loads"]):
                why.append(f"loads {sorted(a['loads'])} vs {sorted(b['loads'])}")
            if why:
                dropped.append((g, mode, "; ".join(why)))
                continue
            for load in sorted(common):
                pairs.setdefault((g, node[g], mode, load), []).append((a["loads"][load], b["loads"][load]))
    return pairs, dropped


def pct(on, off):
    return 100 * (on - off) / off if off else math.nan


def pair_row(group, node, mode, load, rounds):
    """Rounds of one group averaged into the group's pair (PRE-REG 1)."""
    row = {"group": group, "set": node, "mode": mode, "load": load, "rounds": len(rounds),
           "first": "off" if rounds[0][0]["start"] < rounds[0][1]["start"] else "ft"}
    for key, _ in METRICS:
        row[key] = statistics.mean(pct(b[key], a[key]) for a, b in rounds)
        row[f"{key}_off"] = statistics.mean(a[key] for a, _ in rounds)
        row[f"{key}_on"] = statistics.mean(b[key] for _, b in rounds)
    row["stall_ms"] = row["stall_on"] - row["stall_off"]
    return row


def summarize(values):
    n = len(values)
    out = {"n": n, "mean": statistics.mean(values) if n else math.nan, "lo": math.nan, "hi": math.nan,
           "lb": math.nan, "ub": math.nan}
    if n >= 2:
        se = statistics.stdev(values) / math.sqrt(n)
        half, one = fr.t_ppf(0.975, n - 1) * se, fr.t_ppf(0.95, n - 1) * se
        out.update(lo=out["mean"] - half, hi=out["mean"] + half, lb=out["mean"] - one, ub=out["mean"] + one)
    return out


def f(x, spec="+.2f"):
    return "-" if x is None or (isinstance(x, float) and math.isnan(x)) else format(x, spec)


def cell_line(label, s):
    return f"{label} | {s['n']} | {f(s['mean'])} | [{f(s['lo'])}, {f(s['hi'])}] | {f(s['lb'])} | {f(s['ub'])}"


def print_cells(rows, loads, appendix):
    """Cell table (headline and appendix loads); returns {(mode, load): rows} for the headline loads."""
    cells = {}
    for r in rows:
        cells.setdefault((r["mode"], r["load"]), []).append(r)
    print("| mode | load | metric | n | mean % | 95% CI | one-sided 95% LB | one-sided 95% UB |")
    print("|---|---|---|---|---|---|---|---|")
    for (mode, load), rs in sorted(cells.items(), key=lambda kv: (MODES.index(kv[0][0]), kv[0][1])):
        tag = "" if load in loads else " (appendix)" if load in appendix else " (unplanned)"
        for key, label in METRICS:
            print(f"| {mode} | {load}{tag} | {cell_line(label, summarize([r[key] for r in rs]))} |")
    return {k: v for k, v in cells.items() if k[1] in loads}


def print_per_mode(rows, loads):
    print("| mode | metric | n | mean % | 95% CI | one-sided 95% LB | one-sided 95% UB |")
    print("|---|---|---|---|---|---|---|")
    for mode in MODES:
        by_group = {}
        for r in rows:
            if r["mode"] == mode and r["load"] in loads:
                by_group.setdefault(r["group"], []).append(r)
        full = [rs for rs in by_group.values() if len(rs) == len(loads)]
        for key, label in METRICS if full else ():
            print(f"| {mode} | {cell_line(label, summarize([statistics.mean(r[key] for r in rs) for rs in full]))} |")


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("runs", nargs="+", type=pathlib.Path)
    p.add_argument("--window-s", type=float, default=0.0, help="optional extra floor on the window length; the validity rule is all planned waves completed (user decision 10-23)")
    p.add_argument("--loads", default="32,128", help="headline loads per GPU (the IUT cells)")
    p.add_argument("--appendix-loads", default="8,512", help="extended loads: cells only, no claim, no per-mode")
    p.add_argument("--csv", type=pathlib.Path, help="one row per pair and load, for the plots")
    args = p.parse_args()
    loads = [int(x) for x in args.loads.split(",")]
    appendix = [int(x) for x in args.appendix_loads.split(",") if x]
    groups, node, teardown, xid_unchecked = collect([r.resolve() for r in args.runs], args.window_s, loads)
    pairs, dropped = pair_up(groups, node, *PAIRS[:2])
    rows = [pair_row(g, n, m, load, r) for (g, n, m, load), r in sorted(pairs.items())]

    print("## Validity (PRE-REG 5/6, decided before any cost)\n")
    n_arms = sum(len(v) for v in groups.values())
    print(f"Window check: all planned waves completed{f' and >= {args.window_s:.0f} s' if args.window_s else ''}; Xid/NVRM checked on all {GROUP_NODES} nodes in "
          f"{n_arms - len(xid_unchecked)} of {n_arms} arms.\n")
    for arm in xid_unchecked:
        print(f"- note {arm}: Xid/NVRM unchecked (kernel log missing or unreadable on a node)")
    for g, m, why in dropped:
        print(f"- INVALID {g} {m}: {why}")
    if not dropped:
        print("- all pairs valid")
    for g, arm, n in teardown:
        print(f"- note {g} {arm}: {n} fault lines after the last row (server teardown; not a healthy-window fault)")

    print("\n## Pairs (% = (on - off) / off; step metrics + = FT slower, tok/s - = FT slower)\n")
    print("| group | first node | mode | load | first | rounds | trim % | full % | stall % | stall ms diff | "
          "client % | window tok/s % | trim off/on ms |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        print(f"| {r['group']} | {r['set']} | {r['mode']} | {r['load']} | {r['first']} | {r['rounds']} | "
              f"{f(r['trim'])} | {f(r['full'])} | {f(r['stall'])} | {f(r['stall_ms'], '+.3f')} | {f(r['client'])} | "
              f"{f(r['window'])} | {r['trim_off']:.3f}/{r['trim_on']:.3f} |")

    print("\n## Cells (mean % over pairs, 95% t-CI, one-sided 95% bounds)\n")
    cells = print_cells(rows, loads, appendix)
    print(f"\n## Per mode (headline loads {loads} averaged within each pair first; modes not pooled)\n")
    print_per_mode(rows, loads)

    print(f"\n## Claim (PRE-REG 3, IUT over every mode x {loads} cell)\n")
    missing = [(m, load) for m in MODES for load in loads if len(cells.get((m, load), [])) < 2]
    verdict = {key: not missing and all(summarize([r[key] for r in rs])["ub"] < LIMIT for rs in cells.values())
               for key in STEP}
    if missing:
        print(f"- cells missing or n < 2 (no bound, so no claim): {missing}")
    print(f"- trim mean: every upper bound < {LIMIT}%: {'yes' if verdict['trim'] else 'no'}; full mean: "
          f"{'yes' if verdict['full'] else 'no'}")
    print(f"- rule A (pre-registered: trim AND full mean): per-step 'FT cost < 1%' "
          f"{'CLAIMED' if verdict['trim'] and verdict['full'] else 'NOT claimed'}")
    print(f"- rule B (trim mean only; added 10-23 after the rehearsal's n=1 reading): per-step 'FT cost < 1%' "
          f"{'CLAIMED' if verdict['trim'] else 'NOT claimed'}")
    print("- client tok/s: estimate +/- CI only (no < 1% claim at this n)")

    print("\n## Checks only (PRE-REG 7: not used to adjust)\n")
    print("| mode | load | metric | off first: n, mean % | ft first: n, mean % | n0: n, mean % | n8: n, mean % |")
    print("|---|---|---|---|---|---|---|")
    for (mode, load), rs in sorted(cells.items(), key=lambda kv: (MODES.index(kv[0][0]), kv[0][1])):
        for key in STEP:
            parts = []
            for field, value in (("first", "off"), ("first", "ft"), ("set", "n0"), ("set", "n8")):
                v = [r[key] for r in rs if r[field] == value]
                parts.append(f"{len(v)}, {f(statistics.mean(v)) if v else '-'}")
            print(f"| {mode} | {load} | {key} | " + " | ".join(parts) + " |")

    for base, other, title in DESCRIPTIVE:
        dpairs, ddropped = pair_up(groups, node, base, other, need=(base,), shared_loads=True)
        if not dpairs and not ddropped:
            continue
        print(f"\n## Descriptive (PRE-REG 8, not in the claim): {title}; % = (second - first) / first\n")
        for g, m, why in ddropped:
            print(f"- INVALID {g} {m}: {why}")
        drows = [pair_row(g, n, m, load, r) for (g, n, m, load), r in sorted(dpairs.items())]
        print("\n| group | mode | load | rounds | trim % | full % | client % | trim ms first/second |")
        print("|---|---|---|---|---|---|---|---|")
        for r in drows:
            print(f"| {r['group']} | {r['mode']} | {r['load']} | {r['rounds']} | {f(r['trim'])} | {f(r['full'])} | "
                  f"{f(r['client'])} | {r['trim_off']:.3f}/{r['trim_on']:.3f} |")
        print()
        print_cells(drows, loads, appendix)

    if args.csv and rows:
        with open(args.csv, "w", newline="") as out:
            writer = csv.DictWriter(out, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nCSV: {args.csv}")


if __name__ == "__main__":
    main()
