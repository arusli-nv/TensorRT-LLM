"""FT healthy-cost finals readout (FINALS PROTOCOL + AMENDMENTS 1-3 in review/HANDOFF.md).

Input: finals run dirs (RUNS/STD-<job>, RUNS/STD-G1-<job>). One run dir = one node set; cases are <arm>-<round>.
Pairs: within a node set, std-X-<r> vs std-X-ft-<r> (same round r, run adjacently in ABBA order) for X in noeplb,
offline, online. Rounds in one dir are repeats on the same nodes, not independent node sets. Static-LB control:
std-noeplb vs std-noeplb-lb vs std-noeplb-ft.

Primary: rank-0 host loop-step median (steady_state.steady: decode-only steps of the healthy-<load> window with
num_active >= 0.9 x load). Cost % per pair = (on - off) / off (positive = FT slower). Per cell (mode x load): per
node set n, mean, sd, 95% CI (paired t); pooled estimate = mean over all pairs, 95% CI t with df = n_pairs - 1.
Pass on the 9 headline cells (8/32/128 x 3 modes): CI upper bound < +1%, or explained in the readout.
NODESET_DIFF flags (not a test) a cell whose node-set means differ by more than the larger of the two CI half-widths.
Mean step cost % and stall-share difference (points) are reported beside the median. 512/GPU is reported, not
tested, with load attainment per arm.
Exclusions: PIN_CHECK FAIL (tactic_check.txt), startup failure (no server_ready / result.json), missing healthy row
or steady steps; every excluded case and dropped pair is printed with its reason, and n per cell before / after.
A row with > 5% failed or short requests keeps its step metrics, not its client metrics (listed in section 3).

Student t: CDF from the regularized incomplete beta (continued fraction, Numerical Recipes 6.4) and quantiles by
bisection on it, stdlib math only (no scipy); checked against t(.95,3) = 2.353, t(.975,3) = 3.182.
Log timestamps are PT wall time: TZ is forced to America/Los_Angeles before parsing.

Usage: python3 finals_readout.py [--json OUT.json] [--keep-pin-fail] RUN_DIR [RUN_DIR ...]
  --keep-pin-fail keeps PIN_CHECK FAIL cases in the statistics (debug only; output is stamped NOT PROTOCOL).
"""
import argparse
import datetime
import itertools
import json
import math
import os
import pathlib
import re
import statistics
import sys
import time

os.environ["TZ"] = "America/Los_Angeles"
time.tzset()
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
sys.path.append("/lustre/fs1/portfolios/coreai/projects/coreai_comparch_trtllm/users/arusli/src/TensorRT-LLM/.wideep-ft-runs/claude/mvp/harness")
import steady_state as ss  # noqa: E402

MODES = ("noeplb", "offline", "online")
HEADLINE = (8, 32, 128)
LIMIT = 1.0  # % cost bound
MAX_BAD_FRACTION = 0.05
SLOW_MODE = 1.05  # 8/GPU step median above this x the mode's FT-off median = slow-mode launch
KV_LINE = re.compile(r"host_step_time = ([\d.]+)ms, prev_device_step_time = (?:([\d.]+)ms|N/A), timestamp = "
                     r"([\d-]+ [\d:]+), states = \{'num_ctx_requests': (\d+), [^}]*'cached_kv_tokens': (\d+)\}, "
                     r"num_active = (\d+)")
# A stall: a steady decode step longer than this multiple of the median (PRE-REG 2, as trim_readout).
STALL_X = 1.5
NAN = float("nan")


# ---------------------------------------------------------------- statistics (stdlib)
def _betacf(a, b, x):
    tiny = 1e-300
    c, d = 1.0, 1.0 - (a + b) * x / (a + 1.0)
    d = 1.0 / (d if abs(d) > tiny else tiny)
    h = d
    for m in range(1, 300):
        for num in (m * (b - m) * x / ((a + 2 * m - 1) * (a + 2 * m)),
                    -(a + m) * (a + b + m) * x / ((a + 2 * m) * (a + 2 * m + 1))):
            d = 1.0 + num * d
            d = 1.0 / (d if abs(d) > tiny else tiny)
            c = 1.0 + num / c
            c = c if abs(c) > tiny else tiny
            h *= d * c
        if abs(d * c - 1.0) < 1e-14:
            break
    return h


def _ibeta(a, b, x):
    if x <= 0.0 or x >= 1.0:
        return max(0.0, min(1.0, x))
    front = math.exp(math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log(1 - x))
    if x < (a + 1) / (a + b + 2):
        return front * _betacf(a, b, x) / a
    return 1.0 - front * _betacf(b, a, 1 - x) / b


def t_cdf(t, df):
    if math.isnan(t) or df < 1:
        return NAN
    if math.isinf(t):
        return 1.0 if t > 0 else 0.0
    tail = 0.5 * _ibeta(df / 2.0, 0.5, df / (df + t * t))
    return 1.0 - tail if t > 0 else tail


def t_ppf(q, df):
    if df < 1:
        return NAN
    lo, hi = -1e7, 1e7
    for _ in range(200):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if t_cdf(mid, df) < q else (lo, mid)
    return (lo + hi) / 2


def summarize(costs):
    """n, mean, sd, 95% CI (t, df = n - 1) and its half-width; pass = CI upper bound < LIMIT."""
    n = len(costs)
    out = {"n": n, "mean": statistics.mean(costs) if n else NAN, "sd": NAN, "half": NAN, "ci95": [NAN, NAN]}
    if n >= 2:
        sd = statistics.stdev(costs)
        half = t_ppf(0.975, n - 1) * sd / math.sqrt(n)
        out.update(sd=sd, half=half, ci95=[out["mean"] - half, out["mean"] + half])
    out["pass"] = (out["ci95"][1] < LIMIT) if n >= 2 else None
    return out


def nodeset_diff(per):
    """Node-set means differ by more than the larger CI half-width of any two node sets: True/False, None if any
    node set has n < 2."""
    if len(per) < 2:
        return None
    if any(math.isnan(v["half"]) for v in per.values()):
        return None
    return any(abs(x["mean"] - y["mean"]) > max(x["half"], y["half"])
               for x, y in itertools.combinations(per.values(), 2))


def hodges_lehmann(d):
    walsh = [(a + b) / 2 for a, b in itertools.combinations_with_replacement(d, 2)]
    return statistics.median(walsh) if walsh else NAN


def wilcoxon_greater(d):
    """Exact one-sided signed-rank p (H1: d > 0), zeros dropped, average ranks; with its floor 1/2^n."""
    d = [x for x in d if x != 0]
    n = len(d)
    if not n:
        return NAN, NAN
    order = sorted(range(n), key=lambda i: abs(d[i]))
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and abs(d[order[j + 1]]) == abs(d[order[i]]):
            j += 1
        for k in range(i, j + 1):
            ranks[order[k]] = (i + j) / 2 + 1
        i = j + 1
    w = sum(r for r, x in zip(ranks, d) if x > 0)
    hits = sum(sum(r for r, s in zip(ranks, signs) if s) >= w - 1e-9 for signs in itertools.product((0, 1), repeat=n))
    return hits / 2 ** n, 1 / 2 ** n


# ---------------------------------------------------------------- per case
def pct(on, off):
    return 100 * (on - off) / off if off and not (math.isnan(on) or math.isnan(off)) else NAN


def row_metrics(its, row, case):
    load = row["load_per_gpu"]
    start, end, floor = row["start_wall_s"], row["start_wall_s"] + row["elapsed_s"], 0.9 * load
    decode = [(ms, active) for wall, ms, tok, ctx, active in its if start <= wall <= end and ctx == 0]
    picked = [ms for ms, active in decode if active >= floor]
    st = ss.steady(its, row)
    m = {"load": load, "steps": st["steady_steps"], "decode_steps": len(decode),
         "attain": len(picked) / len(decode) if decode else NAN,
         "attain_time": sum(picked) / sum(ms for ms, _ in decode) if decode else NAN,
         "median_active": statistics.median(a for _, a in decode) if decode else NAN}
    if "steady_step_ms" in st:
        med = st["steady_step_ms"]
        m.update(step_ms=med, mean_step_ms=st["steady_mean_step_ms"], p99_step_ms=st["steady_p99_step_ms"],
                 tok_per_step=st["steady_tokens_per_step"], tok_s_gpu=st["steady_tok_s_gpu"],
                 stall_share=100 * sum(ms for ms in picked if ms > STALL_X * med) / sum(picked),
                 stall_excess=100 * sum(ms - med for ms in picked if ms > STALL_X * med) / sum(picked))
    failed, short = row.get("failed_requests") or 0, row.get("short_requests") or 0
    requests = (row.get("completed") or 0) + failed
    m["failed"], m["short"] = failed, short
    m["client_ok"] = row.get("rc") == 0 and requests > 0 and max(failed, short) <= MAX_BAD_FRACTION * requests
    if m["client_ok"]:
        bench = case / "bench" / f"{row['label']}.json"
        mean_tpot = json.loads(bench.read_text()).get("mean_tpot_ms", NAN) if bench.exists() else NAN
        m.update(tpot_median=row.get("median_tpot_ms", NAN), tpot_mean=mean_tpot,
                 out_tok_s_gpu=row["output_throughput"] / row.get("gpus", 32))
    return m


def kv_steps(case, row, host):
    """(cached_kv_tokens, ms) of the row's decode-only full-load steps; host = the line's own host_step_time,
    device = the next line's prev_device_step_time (it belongs to this line's step), as kv_bins.py."""
    start, end, floor = row["start_wall_s"], row["start_wall_s"] + row["elapsed_s"], 0.9 * row["load_per_gpu"]
    out, prev = [], None
    for line in open(case / "rank-0.log", errors="replace"):
        m = KV_LINE.search(line)
        if not m:
            continue
        wall = datetime.datetime.strptime(m[3], "%Y-%m-%d %H:%M:%S").timestamp()
        cur = (wall, int(m[4]), int(m[5]), int(m[6]), float(m[1]))
        ok = lambda s: s and start <= s[0] <= end and s[1] == 0 and s[3] >= floor  # noqa: E731
        if host and ok(cur):
            out.append((cur[2], cur[4]))
        if not host and ok(prev) and m[2]:
            out.append((prev[2], float(m[2])))
        prev = cur
    return out


def kv_binned(off, on, bins=5):
    fills = sorted(kv for kv, _ in off + on)
    if not fills:
        return NAN
    edges = [fills[int(i * (len(fills) - 1) / bins)] for i in range(bins + 1)]
    tw = td = 0.0
    for lo, hi in zip(edges, edges[1:]):
        a = [ms for kv, ms in off if lo <= kv <= hi]
        b = [ms for kv, ms in on if lo <= kv <= hi]
        if len(a) >= 10 and len(b) >= 10:
            w = min(len(a), len(b))
            tw += w
            td += w * pct(statistics.median(b), statistics.median(a))
    return td / tw if tw else NAN


def tactic_status(run):
    out = {}
    path = run / "tactic_check.txt"
    if path.exists():
        for line in path.read_text().splitlines():
            m = re.search(r"TACTIC_CHECK (\S+) .*?autotune=(\d+).*?PIN_CHECK (\w+)", line)
            if m:
                out[m[1]] = (m[3], int(m[2]))
    return out


def load_group(run, loads, keep_pin_fail):
    tactics = tactic_status(run)
    groups = {}
    for case in sorted(p for p in run.iterdir() if p.is_dir() and re.fullmatch(r"std-.+-\d+", p.name)):
        arm, _, rnd = case.name.rpartition("-")
        pin, autotune = tactics.get(case.name, ("MISSING", None))
        result = case / "result.json"
        rows = json.loads(result.read_text()).get("rows", []) if result.exists() else []
        startup = (case / "server_ready").exists() and result.exists()
        healthy = {}
        for r in rows:
            if r.get("label") == f"healthy-{r.get('load_per_gpu')}" and r["load_per_gpu"] in loads:
                healthy[r["load_per_gpu"]] = r  # last such row
        reason = "PIN_CHECK FAIL" if pin == "FAIL" else "startup failure" if not startup else ""
        info = {"run": run.name, "case": case.name, "arm": arm, "pin": pin, "autotune": autotune,
                "startup_ok": startup, "rows": sorted(healthy), "excluded": bool(reason), "reason": reason,
                "used": not reason or (keep_pin_fail and reason == "PIN_CHECK FAIL"), "loads": {}, "case_path": case}
        if healthy and (case / "rank-0.log").exists():
            its = ss.iterations(case / "rank-0.log")
            info["loads"] = {L: row_metrics(its, r, case) for L, r in healthy.items()}
            info["_rows"] = healthy
        groups.setdefault(f"{run.name}/r{rnd}", {})[arm] = info
    return groups


def job_of(group):
    m = re.search(r"(\d+)(?:/r\d+)?$", group)
    return m[1] if m else group


# ---------------------------------------------------------------- readout
def fmt(x, spec="+.2f"):
    return "-" if x is None or (isinstance(x, float) and math.isnan(x)) else format(x, spec)


def nodeset(unit):
    return unit.rpartition("/r")[0]


def pairs(groups, off_arm, on_arm, load, drop=frozenset(), dropped=None):
    """(unit, off metrics, on metrics) per node set and round; the units skipped, with why, go to dropped."""
    out = []
    for g, arms in sorted(groups.items()):
        a, b = arms.get(off_arm), arms.get(on_arm)
        if not (a or b):
            continue
        why = []
        for arm, c in ((off_arm, a), (on_arm, b)):
            if not c:
                why.append(f"no {arm} case")
            elif not c["used"]:
                why.append(f"{c['case']} {c['reason']}")
            elif load not in c["loads"]:
                why.append(f"{c['case']} no healthy-{load} row")
            elif "step_ms" not in c["loads"][load]:
                why.append(f"{c['case']} no steady steps")
        if g in drop:
            why.append("slow-mode launch")
        if why:
            if dropped is not None:
                dropped.append((g, "; ".join(why)))
            continue
        out.append((g, a["loads"][load], b["loads"][load]))
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("runs", nargs="+", type=pathlib.Path)
    p.add_argument("--json", type=pathlib.Path)
    p.add_argument("--loads", default="8,32,128,512")
    p.add_argument("--keep-pin-fail", action="store_true")
    a = p.parse_args()
    loads = [int(x) for x in a.loads.split(",")]
    groups = {}
    for run in a.runs:
        groups.update(load_group(run.resolve(), loads, a.keep_pin_fail))
    out = {"runs": [str(r) for r in a.runs], "keep_pin_fail": a.keep_pin_fail, "cases": [], "cells": {},
           "client": {}, "stall": {}, "kv32": {}, "sensitivity": {}, "lb": []}
    print("# FT healthy-cost finals readout")
    print(f"runs: {', '.join(r.name for r in a.runs)}; node sets: {len({nodeset(g) for g in groups})}; node set x round units: {len(groups)}; cost % = (FT on - off) / off, "
          f"positive = FT slower")
    if a.keep_pin_fail:
        print("\n**NOT PROTOCOL: --keep-pin-fail keeps PIN_CHECK FAIL cases in the statistics.**")

    # (1) inclusion
    print("\n## 1. Inclusion\n\n| node set / round | case | PIN_CHECK | autotune | startup ok | healthy rows | excluded |")
    print("|---|---|---|---|---|---|---|")
    for g, arms in sorted(groups.items()):
        for arm, c in sorted(arms.items()):
            print(f"| {g} | {c['case']} | {c['pin']} | {c['autotune']} | {'yes' if c['startup_ok'] else 'NO'} | "
                  f"{','.join(map(str, c['rows'])) or 'none'} | {c['reason'] or 'no'}"
                  f"{' (kept: --keep-pin-fail)' if c['excluded'] and c['used'] else ''} |")
            out["cases"].append({k: v for k, v in c.items() if not k.startswith("_") and k != "case_path"}
                                | {"group": g, "case_path": str(c["case_path"])})

    excl = [f"- {c['case_path'].parent.name}/{c['case']}: {c['reason']}"
            f"{' (kept: --keep-pin-fail)' if c['used'] else ''}"
            for arms in groups.values() for c in arms.values() if c["excluded"]]
    print("\nExcluded cases:\n" + ("\n".join(excl) or "- none"))

    # slow-mode launches: 8/GPU step median > SLOW_MODE x median of the mode's FT-off cases
    slow = {}
    for mode in MODES:
        offs = [arms[f"std-{mode}"]["loads"][8]["step_ms"] for arms in groups.values()
                if arms.get(f"std-{mode}", {}).get("used") and "step_ms" in arms[f"std-{mode}"]["loads"].get(8, {})]
        ref = statistics.median(offs) if offs else NAN
        slow[mode] = sorted({f"{g}:{arm}" for g, arms in groups.items() for arm in (f"std-{mode}", f"std-{mode}-ft")
                             if arms.get(arm, {}).get("used")
                             and arms[arm]["loads"].get(8, {}).get("step_ms", 0) > SLOW_MODE * ref})

    # (2) primary cells
    print("\n## 2. Primary: rank-0 host loop-step median, FT on vs off (paired by node set and round)\n")
    print("Estimate = mean over pairs, 95% CI paired t (df = n pairs - 1); UB<1% = CI upper bound < +1% (headline "
          "cells; a 'no' passes only if explained). Mean step = cost % of the steady mean step; stall = FT on - off "
          "stall share in points (mean over pairs). "
          "Flag '-': fewer than 2 node sets, or a node set with < 2 pairs (no CI half-width).\n")
    print("| mode | load | n before / after | node sets | per-pair cost % | estimate % | 95% CI | UB<1% | per node set "
          "mean [CI] (n) | between node sets | flag | mean step % [CI] | stall pts | delta us | off / on step ms | "
          "off / on tok/s/GPU |")
    print("|---" * 16 + "|")
    cells = {}
    for mode in MODES:
        for load in loads:
            dropped = []
            pr = pairs(groups, f"std-{mode}", f"std-{mode}-ft", load, dropped=dropped)
            costs = [pct(b["step_ms"], a["step_ms"]) for _, a, b in pr]
            s = summarize(costs)
            by_set = {}
            for (g, _, _), cst in zip(pr, costs):
                by_set.setdefault(nodeset(g), []).append(cst)
            per = {k: summarize(v) for k, v in by_set.items()}
            means = [v["mean"] for v in per.values()]
            s.update(groups=[g for g, _, _ in pr], costs=costs, headline=load in HEADLINE, dropped=dropped,
                     n_before=len(pr) + len(dropped), nodesets=per,
                     between=(max(means) - min(means)) if len(means) >= 2 else NAN, nodeset_diff=nodeset_diff(per),
                     mean_step=summarize([pct(b["mean_step_ms"], a["mean_step_ms"]) for _, a, b in pr]),
                     stall_pts=statistics.mean(b["stall_share"] - a["stall_share"] for _, a, b in pr) if pr else NAN,
                     delta_us=statistics.mean(1e3 * (b["step_ms"] - a["step_ms"]) for _, a, b in pr) if pr else NAN,
                     off_step_ms=statistics.mean(a["step_ms"] for _, a, _ in pr) if pr else NAN,
                     on_step_ms=statistics.mean(b["step_ms"] for _, _, b in pr) if pr else NAN,
                     off_tok_s_gpu=statistics.mean(a["tok_s_gpu"] for _, a, _ in pr) if pr else NAN,
                     on_tok_s_gpu=statistics.mean(b["tok_s_gpu"] for _, _, b in pr) if pr else NAN,
                     attain={arm: [(g, x["loads"].get(load, {}).get("attain", NAN),
                                    x["loads"].get(load, {}).get("attain_time", NAN),
                                    x["loads"].get(load, {}).get("median_active", NAN))
                                   for g, arms in sorted(groups.items()) for arm2, x in arms.items()
                                   if arm2 == arm and x["used"]] for arm in (f"std-{mode}", f"std-{mode}-ft")})
            cells[f"{mode}/{load}"] = s
    for k, c in cells.items():
        mode, load = k.split("/")
        verdict = ("n<2" if c["pass"] is None else "yes" if c["pass"] else "no") if c["headline"] else "not tested"
        flag = {None: "-", True: "NODESET_DIFF", False: "no"}[c["nodeset_diff"]] if len(c["nodesets"]) >= 2 else "-"
        sets = "; ".join(f"{ns} {fmt(v['mean'])}"
                         + (f" [{fmt(v['ci95'][0])}, {fmt(v['ci95'][1])}]" if v["n"] >= 2 else "") + f" ({v['n']})"
                         for ns, v in c["nodesets"].items()) or "-"
        ms = c["mean_step"]
        print(f"| {mode} | {load} | {c['n_before']} / {c['n']} | {len(c['nodesets'])} | "
              f"{' '.join(fmt(x) for x in c['costs']) or '-'} | {fmt(c['mean'])} | "
              f"[{fmt(c['ci95'][0])}, {fmt(c['ci95'][1])}] | {verdict} | {sets} | {fmt(c['between'], '.2f')} | {flag} | "
              f"{fmt(ms['mean'])} [{fmt(ms['ci95'][0])}, {fmt(ms['ci95'][1])}] | {fmt(c['stall_pts'])} | "
              f"{fmt(c['delta_us'], '+.0f')} | {fmt(c['off_step_ms'], '.3f')} / {fmt(c['on_step_ms'], '.3f')} | "
              f"{fmt(c['off_tok_s_gpu'], '.0f')} / {fmt(c['on_tok_s_gpu'], '.0f')} |")
    print("\nDropped pairs (cell: unit: reason):")
    drops = [f"- {k}: {g}: {why}" for k, c in cells.items() for g, why in c["dropped"]]
    print("\n".join(drops) or "- none")
    head = [c for c in cells.values() if c["headline"]]
    every = all(c["pass"] for c in head) if all(c["pass"] is not None for c in head) else None
    fails = [k for k, c in cells.items() if c["headline"] and c["pass"] is False]
    print(f"\nAll 9 headline cells CI upper bound < +1%: "
          f"{'n<2 in some cell' if every is None else 'yes' if every else 'no'}"
          + (f" (needs an explanation: {', '.join(fails)})" if fails else "") + ".")
    print("\nLoad attainment over the row's decode steps: share of steps with num_active >= 0.9 x load (and share of "
          "decode time), median num_active; per arm:\n")
    for k, c in cells.items():
        if not c["headline"]:
            for arm, vals in c["attain"].items():
                print(f"- {k} {arm}: " + ("; ".join(f"{g} {fmt(100 * s, '.1f')}% of steps ({fmt(100 * t, '.1f')}% of time)"
                                                      f", median active {fmt(m, '.0f')}" for g, s, t, m in vals)
                                            or "no used case"))
    print("\nCLAIMS-ready lines:")
    for k, c in cells.items():
        print(f"- {k}/GPU: FT cost {fmt(c['mean'])}% [95% CI {fmt(c['ci95'][0])}, {fmt(c['ci95'][1])}%], n={c['n']} "
              f"pairs over {len(c['nodesets'])} node sets ({', '.join(c['groups'])}), rank-0 host loop-step median, "
              f"decode-only num_active >= 0.9 load"
              + ("" if c["headline"] else ", beyond recipe, not tested"))
    out["cells"] = cells
    out["every_cell_ub_below_limit"] = every
    out["slow_mode"] = slow

    # (3) client descriptive
    print("\n## 3. Client (descriptive; rows with > 5% failed/short requests dropped): median over pairs' cases [min-max]\n")
    bad = [f"- {g}:{arm} {L}/GPU: failed {m['failed']}, short {m['short']}" for g, arms in sorted(groups.items())
           for arm, x in sorted(arms.items()) if x["used"] for L, m in sorted(x["loads"].items()) if not m["client_ok"]]
    print("Client metrics dropped (row kept for step metrics):\n" + ("\n".join(bad) or "- none") + "\n")
    print("| mode | load | arm | n | TPOT median ms | TPOT mean ms | out tok/s/GPU |\n|---|---|---|---|---|---|---|")
    for mode in MODES:
        for load in loads:
            for arm in (f"std-{mode}", f"std-{mode}-ft"):
                ms = [x["loads"][load] for arms in groups.values() for a2, x in arms.items()
                      if a2 == arm and x["used"] and x["loads"].get(load, {}).get("client_ok")]
                if not ms:
                    continue
                cell = {}
                for f in ("tpot_median", "tpot_mean", "out_tok_s_gpu"):
                    v = [m[f] for m in ms if not math.isnan(m.get(f, NAN))]
                    cell[f] = [statistics.median(v), min(v), max(v)] if v else [NAN] * 3
                out["client"][f"{mode}/{load}/{arm}"] = cell | {"n": len(ms)}
                print(f"| {mode} | {load} | {arm} | {len(ms)} | " + " | ".join(
                    f"{fmt(v[0], '.2f')} [{fmt(v[1], '.2f')}-{fmt(v[2], '.2f')}]" for v in cell.values()) + " |")

    # (4) stall share
    print(f"\n## 4. Stall share (% of steady time in steps > {STALL_X:g}x the row median), FT on - off in points\n")
    print("Rule: Hodges-Lehmann <= +0.5 points and no single pair > +2 points. Wilcoxon p descriptive (floor 1/2^n).\n")
    print("| mode | load | n | off % per pair | on % per pair | HL | max pair | rule | Wilcoxon p (floor) |")
    print("|---|---|---|---|---|---|---|---|---|")
    for mode in MODES:
        for load in loads:
            pr = pairs(groups, f"std-{mode}", f"std-{mode}-ft", load)
            if not pr:
                continue
            d = [b["stall_share"] - a["stall_share"] for _, a, b in pr]
            hl, mx = hodges_lehmann(d), max(d)
            wp, floor = wilcoxon_greater(d)
            ok = hl <= 0.5 and mx <= 2.0
            out["stall"][f"{mode}/{load}"] = {"off": [a["stall_share"] for _, a, _ in pr],
                                              "on": [b["stall_share"] for _, _, b in pr], "diff": d, "hl": hl,
                                              "max_pair": mx, "pass": ok, "wilcoxon_p": wp, "wilcoxon_floor": floor,
                                              "excess_off": [a["stall_excess"] for _, a, _ in pr],
                                              "excess_on": [b["stall_excess"] for _, _, b in pr]}
            print(f"| {mode} | {load} | {len(d)} | {' '.join(fmt(a['stall_share'], '.1f') for _, a, _ in pr)} | "
                  f"{' '.join(fmt(b['stall_share'], '.1f') for _, _, b in pr)} | {fmt(hl)} | {fmt(mx)} | "
                  f"{'PASS' if ok else 'fail'} | {fmt(wp, '.3f')} ({fmt(floor, '.3f')}) |")

    # (5) KV-fill binned at 32/GPU
    print("\n## 5. 32/GPU cost within KV-fill bins (5 quantile bins of both arms' cached_kv_tokens, weighted by min "
          "steps per bin)\n")
    print("| mode | metric | per-pair binned cost % | n | mean % | 95% CI |")
    print("|---|---|---|---|---|---|")
    for mode in MODES:
        pr32 = [(g, groups[g][f"std-{mode}"], groups[g][f"std-{mode}-ft"]) for g, _, _ in
                pairs(groups, f"std-{mode}", f"std-{mode}-ft", 32)]
        for host in (True, False):
            per = [(g, kv_binned(kv_steps(x["case_path"], x["_rows"][32], host),
                                 kv_steps(y["case_path"], y["_rows"][32], host))) for g, x, y in pr32]
            vals = [v for _, v in per if not math.isnan(v)]
            if not per:
                continue
            s = summarize(vals) | {"per_group": per}
            out["kv32"][f"{mode}/{'host' if host else 'device'}"] = s
            print(f"| {mode} | {'host loop-step' if host else 'device step'} | "
                  f"{' '.join(f'{g}:{fmt(v)}' for g, v in per)} | {s['n']} | {fmt(s['mean'])} | "
                  f"[{fmt(s['ci95'][0])}, {fmt(s['ci95'][1])}] |")

    # (6) sensitivity
    print("\n## 6. Sensitivity\n")
    print("Job-averaged: the pairs of one job (both node sets) averaged first, t over jobs. Slow-mode launch: 8/GPU step median > "
          f"{SLOW_MODE:.2f} x the median of its mode's FT-off cases; its pair is dropped for that mode.\n")
    for mode in MODES:
        print(f"- slow-mode launches {mode}: {', '.join(slow[mode]) or 'none'}")
    print("\n| mode | load | variant | n | mean % | 95% CI | UB<1% | agrees with primary |")
    print("|---|---|---|---|---|---|---|---|")
    for mode in MODES:
        dropped = {s.split(':')[0] for s in slow[mode]}
        for load in loads:
            prim = cells[f"{mode}/{load}"]
            by_job = {}
            for g, cst in zip(prim["groups"], prim["costs"]):
                by_job.setdefault(job_of(g), []).append(cst)
            variants = {"job-averaged": summarize([statistics.mean(v) for v in by_job.values()]),
                        "without slow-mode": summarize([pct(b["step_ms"], a["step_ms"]) for _, a, b in
                                                        pairs(groups, f"std-{mode}", f"std-{mode}-ft", load,
                                                              dropped)])}
            for name, s in variants.items():
                out["sensitivity"][f"{mode}/{load}/{name}"] = s
                if load not in HEADLINE:
                    verdict = "not tested"
                else:
                    verdict = "n<2" if s["pass"] is None else "yes" if s["pass"] else "no"
                agree = ("-" if s["pass"] is None or prim["pass"] is None or load not in HEADLINE
                         else "yes" if s["pass"] == prim["pass"] else "NO -> not established")
                print(f"| {mode} | {load} | {name} | {s['n']} | {fmt(s['mean'])} | [{fmt(s['ci95'][0])}, {fmt(s['ci95'][1])}] | {verdict} | "
                      f"{agree} |")

    # (7) static-LB control
    print("\n## 7. Static-LB control (std-noeplb vs std-noeplb-lb vs std-noeplb-ft), host loop-step median\n")
    print("| group | load | noeplb ms | lb ms | ft ms | lb vs noeplb % | ft vs lb % | ft vs noeplb % |")
    print("|---|---|---|---|---|---|---|---|")
    for g, arms in sorted(groups.items()):
        if "std-noeplb-lb" not in arms:
            continue
        for load in loads:
            ms = {k: (arms[f"std-noeplb{k}"]["loads"].get(load, {}).get("step_ms", NAN)
                      if arms.get(f"std-noeplb{k}", {}).get("used") else NAN) for k in ("", "-lb", "-ft")}
            row = {"group": g, "load": load, "noeplb": ms[""], "lb": ms["-lb"], "ft": ms["-ft"],
                   "lb_vs_noeplb": pct(ms["-lb"], ms[""]), "ft_vs_lb": pct(ms["-ft"], ms["-lb"]),
                   "ft_vs_noeplb": pct(ms["-ft"], ms[""])}
            out["lb"].append(row)
            print(f"| {g} | {load} | {fmt(ms[''], '.3f')} | {fmt(ms['-lb'], '.3f')} | {fmt(ms['-ft'], '.3f')} | "
                  f"{fmt(row['lb_vs_noeplb'])} | {fmt(row['ft_vs_lb'])} | {fmt(row['ft_vs_noeplb'])} |")
    if not out["lb"]:
        print("| (no group with std-noeplb-lb) |||||||| ")
    unpaired = [f"{g}:{arm}" for g, arms in sorted(groups.items()) for arm in arms
                if arm.endswith("-ft") and arm[:-3] not in arms and not arm.startswith("std-noeplb-lb")]
    if unpaired:
        print(f"\nUnpaired FT cases (reported, not in any cell): {', '.join(unpaired)}")
    out["unpaired"] = unpaired

    if a.json:
        def clean(o):
            if isinstance(o, float) and (math.isnan(o) or math.isinf(o)):
                return None
            if isinstance(o, dict):
                return {str(k): clean(v) for k, v in o.items()}
            if isinstance(o, (list, tuple)):
                return [clean(v) for v in o]
            return o
        out["groups"] = {g: {arm: {str(L): m for L, m in c["loads"].items()} for arm, c in arms.items()}
                         for g, arms in groups.items()}
        a.json.write_text(json.dumps(clean(out), indent=1))
        print(f"\njson: {a.json}")


if __name__ == "__main__":
    main()
