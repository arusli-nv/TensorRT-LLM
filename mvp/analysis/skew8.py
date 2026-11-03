"""Skew / admission / client-refill analysis of one STD run (off, ft, ulfm arms), loads 8 and 32.

Usage: python3 skew8.py RUN_DIR > out.md
Reads rank-<r>.log hook lines (WIDEEP_SKEW, WIDEEP_COLL, WIDEEP_FETCH, WIDEEP_ADMIT), rank-0 print_iter_log
lines and bench/healthy-<load>.tokens.jsonl, restricted to each healthy row's [start_wall_s, +elapsed_s] window.
"""
import os
import ast
import json
import pathlib
import re
import statistics as st
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "harness"))
import steady_state as ss  # noqa: E402

# SKEW8_ARMS="name=case,..." overrides the arms, e.g. "off=std-noeplb-1,ft=std-noeplb-ft-1,sm=std-noeplb-ft-sm-1".
ARMS = tuple(tuple(a.split("=", 1)) for a in os.environ["SKEW8_ARMS"].split(",")) if os.environ.get("SKEW8_ARMS") else (
    ("off", "std-noeplb-1"), ("ft", "std-noeplb-ft-1"), ("ulfm", "std-noeplb-ulfm-1"))
LOADS = (8, 32)
SLOW_MS = 150.0
TIMEOUT_MIN_HOLD = 50  # a release after a hold of >= this many balance calls is a gate-timeout release
HOOK = re.compile(r"(WIDEEP_\w+) rank (\d+) t=([\d.]+) (.*)")
KV = re.compile(r"(\w+)=([\w.]+?)(?=INFO:|\s|$)")  # server access-log text can follow a hook line unseparated


def med(v):
    return st.median(v) if v else float("nan")


def hooks(case):
    out = {}
    for log in sorted(case.glob("rank-*.log")):
        with open(log, errors="replace") as f:
            for line in f:
                if "WIDEEP_" not in line:
                    continue
                m = HOOK.search(line)
                if not m:
                    continue
                kv = dict(KV.findall(m.group(4)))
                out.setdefault(m.group(1), []).append((int(m.group(2)), float(m.group(3)), kv))
    return out


def rank0_iters(case):
    rows = []
    with open(case / "rank-0.log", errors="replace") as f:
        for line in f:
            if "iter = " not in line:
                continue
            m = ss.LINE.search(line)
            if not m:
                continue
            states = ast.literal_eval(m.group(4))
            wall = __import__("datetime").datetime.strptime(m.group(3), "%Y-%m-%d %H:%M:%S").timestamp()
            rows.append((int(m.group(1)), wall, float(m.group(2)), states.get("num_ctx_requests", 0)))
    return rows


def steps(its, row):
    s, e = row["start_wall_s"], row["start_wall_s"] + row["elapsed_s"]
    win = [x for x in its if s <= x[1] <= e]
    dec = [ms for _, _, ms, c in win if c == 0]
    ctx = [ms for _, _, ms, c in win if c > 0]
    slow = [k for k, (_, _, ms, _) in enumerate(win) if ms >= SLOW_MS]
    ctx_idx = [k for k, x in enumerate(win) if x[3] > 0]
    ctxset = set(ctx_idx)
    remote = [k for k in slow if not ({k, k - 1, k - 2} & ctxset)]
    bursts = [k for j, k in enumerate(ctx_idx) if j == 0 or k - ctx_idx[j - 1] > 5]
    full = list(zip(bursts[1:-1], bursts[2:]))  # waves between rank-0 bursts, startup and last partial dropped
    per_wave = [sum(1 for k in remote if a <= k < b) for a, b in full]
    mins = row["elapsed_s"] / 60
    band = sum(1 for _, _, ms, c in win if c == 0 and 180.0 <= ms <= 210.0)  # ctx_stall_cost.py band
    return {"band_pm": band / (row["elapsed_s"] / 60), "n": len(win), "mean_all": st.mean(ms for _, _, ms, _ in win),
            "mean_dec": st.mean(ms for _, _, ms, c in win if c == 0 and ms < SLOW_MS), "dec_med": med(dec), "ctx_med": med(ctx), "ctx_n": len(ctx), "slow_pm": len(slow) / mins,
            "remote_pm": len(remote) / mins, "own_pm": (len(slow) - len(remote)) / mins,
            "waves": len(full), "rel_wave": st.mean(per_wave) if per_wave else float("nan"),
            "waves_ge1": sum(1 for x in per_wave if x)}


def in_win(recs, row):
    s, e = row["start_wall_s"], row["start_wall_s"] + row["elapsed_s"]
    return [r for r in recs if s <= r[1] <= e]


def coll(h, row):
    by = {}
    for rank, _, kv in in_win(h.get("WIDEEP_COLL", []), row):
        by.setdefault(kv["method"], {}).setdefault(rank, []).append((float(kv["p50_us"]), float(kv["p99_us"])))
    out = {}
    for meth, ranks in by.items():
        p50 = {r: med([a for a, _ in v]) for r, v in ranks.items()}
        p99 = {r: med([b for _, b in v]) for r, v in ranks.items()}
        lo, hi = min(p50, key=p50.get), max(p50, key=p50.get)
        out[meth] = {"ranks": len(ranks), "lines": sum(len(v) for v in ranks.values()), "p50_med": med(list(p50.values())),
                     "p50_min": p50[lo], "p50_max": p50[hi], "p50_spread": p50[hi] - p50[lo], "lo": lo, "hi": hi,
                     "p99_med": med(list(p99.values())), "p99_max": max(p99.values())}
    return out


def skew(h, row):
    by = {}
    for rank, t, kv in in_win(h.get("WIDEEP_SKEW", []), row):
        by.setdefault(rank, []).append((t, float(kv["p50_us"]), float(kv["p99_us"]), kv.get("after", "")))
    if not by:  # a run without the TLLM_TEST_LAUNCH_SKEW hook
        return None
    n = min(len(v) for v in by.values())
    spread50, spread99, med50, med99, tspan, hi_rank = [], [], [], [], [], {}
    for k in range(n):
        col = {r: v[k] for r, v in by.items()}
        p50 = [c[1] for c in col.values()]
        p99 = [c[2] for c in col.values()]
        spread50.append(max(p50) - min(p50))
        spread99.append(max(p99) - min(p99))
        med50.append(med(p50))
        med99.append(med(p99))
        tspan.append(max(c[0] for c in col.values()) - min(c[0] for c in col.values()))
        top = max(col, key=lambda r: col[r][1])
        hi_rank[top] = hi_rank.get(top, 0) + 1
    afters = {c[3] for v in by.values() for c in v}
    return {"ranks": len(by), "windows": n, "spread50": med(spread50), "spread50_max": max(spread50),
            "spread99": med(spread99), "spread99_max": max(spread99), "med50": med(med50), "med99": med(med99),
            "tspan": med(tspan), "hi_rank": sorted(hi_rank.items(), key=lambda kv: -kv[1])[:3], "after": afters}


def admission(h, row, its, anchors):
    """Per client wave (anchors: main start group (first, last) wall s): fetches, gate timeouts, remote slow steps.

    Main refill = fetches within [-5, +10] calls of the wave's median request call; off-phase = the rest (early/late).
    Call numbers are global (ranks fetch in lockstep), so wave bounds map to calls via any rank's log.
    """
    fetch = [(int(kv["call"]), r, int(kv["new"]), t) for r, t, kv in in_win(h.get("WIDEEP_FETCH", []), row)]
    timeline = sorted((t, c) for c, _, _, t in fetch)

    def call_at(wall):
        return next((c for t, c in timeline if t >= wall), timeline[-1][1] + 1)

    # Timeout releases: per rank, a release of context held since the previous ADMIT line >= TIMEOUT_MIN_HOLD calls.
    by, rel, holds = {}, [], []
    for r, t, kv in h.get("WIDEEP_ADMIT", []):
        by.setdefault(r, []).append((int(kv["call"]), int(kv["ctx"]), int(kv["held"]), t))
    for r, lines in by.items():
        lines.sort()
        for prev, cur in zip(lines, lines[1:]):
            if prev[2] > 0 and cur[1] > 0 and row["start_wall_s"] <= cur[3] <= row["start_wall_s"] + row["elapsed_s"]:
                holds.append(cur[0] - prev[0])
                if cur[0] - prev[0] >= TIMEOUT_MIN_HOLD:
                    rel.append((cur[0], r, cur[1]))
    remote = remote_slow_calls(its, row)
    kind = {}  # release call (ranks other than 0) -> {"T"imeout, "I"mmediate}
    for r, lines in by.items():
        if r == 0:
            continue
        for prev, cur in zip([None] + lines, lines):
            if cur[1] > 0 and row["start_wall_s"] <= cur[3] <= row["start_wall_s"] + row["elapsed_s"]:
                held = prev is not None and prev[2] > 0 and cur[0] - prev[0] >= TIMEOUT_MIN_HOLD
                kind.setdefault(cur[0], set()).add("T" if held else "I")
    cause = {"timeout": 0, "immediate": 0, "none": 0}
    for c in remote:
        k = kind.get(c - 1, set()) | kind.get(c, set())
        cause["timeout" if "T" in k else "immediate" if k else "none"] += 1
    per = []
    for (g0, g1), (n0, _) in zip(anchors, anchors[1:]):
        lo, hi = call_at(g0 - 0.3), call_at(n0 - 0.3)
        wf = [x for x in fetch if lo <= x[0] < hi]
        reqs = sorted(c for c, _, n, _ in wf for _ in range(n))
        mid = reqs[len(reqs) // 2]
        off = [x for x in wf if not mid - 5 <= x[0] <= mid + 10]
        late = [x for x in off if x[0] > mid]
        wr = [x for x in rel if lo <= x[0] < hi]
        per.append({"calls": len({x[0] for x in wf}), "main_calls": len({x[0] for x in wf} - {x[0] for x in off}),
                    "new": sum(x[2] for x in wf), "off_calls": len({x[0] for x in off}), "off_new": sum(x[2] for x in off),
                    "off_ranks": len(off), "late_new": sum(x[2] for x in late),
                    "late_calls": len({x[0] for x in late}), "rel_ranks": len(wr), "rel_calls": len({x[0] for x in wr}),
                    "rel_req": sum(x[2] for x in wr), "remote": sum(1 for c in remote if lo <= c < hi), "len": hi - lo})
    agg = lambda k: st.mean(p[k] for p in per) if per else float("nan")  # noqa: E731
    hb = {"<10": sum(1 for x in holds if x < 10), "10-49": sum(1 for x in holds if 10 <= x < 50),
          "50-69": sum(1 for x in holds if 50 <= x < 70), ">=70": sum(1 for x in holds if x >= 70)}
    return {"waves": len(per), **{k: agg(k) for k in ("calls", "main_calls", "new", "off_calls", "off_new", "off_ranks", "late_calls", "late_new",
                                                         "rel_ranks", "rel_calls", "rel_req", "remote", "len")},
            "waves_rel": sum(1 for p in per if p["rel_ranks"]), "holds": hb, "cause": cause}


def remote_slow_calls(its, row):
    """Fetch-call numbers (rank-0 iter + 3) of slow rank-0 steps with no rank-0 context in that or the 2 prior steps."""
    s, e = row["start_wall_s"], row["start_wall_s"] + row["elapsed_s"]
    ctx = {i for i, _, _, c in its if c > 0}
    return [i + 3 for i, wall, ms, _ in its
            if s <= wall <= e and ms >= SLOW_MS and not ({i, i - 1, i - 2} & ctx)]


def client(case, row, load):
    """Client waves anchored on the main group of starts (gaps <= 0.5 s, >= half the concurrency)."""
    path = case / "bench" / f"healthy-{load}.tokens.jsonl"
    if not path.exists():
        return None
    s, e = row["start_wall_s"], row["start_wall_s"] + row["elapsed_s"]
    starts = sorted(x for x in (json.loads(l)["start"] for l in open(path) if l.strip()) if s <= x <= e)
    groups = []
    for x in starts:
        if groups and x - groups[-1][-1] <= 0.5:
            groups[-1].append(x)
        else:
            groups.append([x])
    anchors = [g for g in groups if len(g) >= 16 * load]
    per = []
    for g, nxt in zip(anchors[:-1], anchors[1:]):
        off = [x - g[0] for x in starts if g[-1] < x < nxt[0]]
        per.append({"main": len(g), "span": (g[-1] - g[0]) * 1e3, "off": len(off), "off_delay": med(off)})
    ends = [max(float(k) for k in r["tokens"]) + 1 for r in (json.loads(l) for l in open(path) if l.strip())
            if r["tokens"] and s <= r["start"] <= e]
    period = st.mean(b[0] - a[0] for a, b in zip(anchors, anchors[1:]))
    return {"period": period, "tail": max(ends) - anchors[-1][0], "last_start": starts[-1] - anchors[-1][0],
            "anchors": [(g[0], g[-1]) for g in anchors], "waves": len(per), "main": med([p["main"] for p in per]), "span": med([p["span"] for p in per]),
            "span_max": max(p["span"] for p in per), "off": st.mean(p["off"] for p in per),
            "off_waves": sum(1 for p in per if p["off"]),
            "off_delay": med([p["off_delay"] for p in per if p["off"]])}


def main(run):
    res = {}
    for arm, cname in ARMS:
        case = run / cname
        rows = {r["label"]: r for r in json.load(open(case / "result.json"))["rows"]}
        h = hooks(case)
        its = rank0_iters(case)
        for load in LOADS:
            row = rows[f"healthy-{load}"]
            cl = client(case, row, load)
            res[arm, load] = {"steps": steps(its, row), "coll": coll(h, row), "skew": skew(h, row),
                              "adm": admission(h, row, its, cl["anchors"]), "client": cl, "row": row}
    return res


def f(x, d=1):
    return f"{x:.{d}f}"


def report(res):
    p = print
    p("## 1. Rank-0 steps (print_iter_log in the healthy window)\n")
    p("| load | arm | decode median ms | ctx median ms (n) | slow >=150 ms /min | own /min | remote /min | full waves | "
      "remote slow per wave | waves with >=1 | 180-210 ms decode /min | client out tok/s |")
    p("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for load in LOADS:
        for arm, _ in ARMS:
            s, row = res[arm, load]["steps"], res[arm, load]["row"]
            p(f"| {load} | {arm} | {f(s['dec_med'], 2)} | {f(s['ctx_med'])} ({s['ctx_n']}) | {f(s['slow_pm'], 2)} | "
              f"{f(s['own_pm'], 2)} | {f(s['remote_pm'], 2)} | {s['waves']} | {f(s['rel_wave'], 2)} | "
              f"{s['waves_ge1']}/{s['waves']} | {f(s['band_pm'], 2)} | {row['output_throughput']:.0f} |")
    p("\n### 1b. Wave cycle vs benchmark duration (client waves of section 5)\n")
    p("| load | arm | iters in window | mean step ms (all) | mean decode step ms (<150 ms, no ctx) | wave period s | "
      "client duration s | last start after last main wave s | last token after last main wave s (1 s res) |")
    p("|---|---|---|---|---|---|---|---|---|")
    for load in LOADS:
        for arm, _ in ARMS:
            s_, c, row = res[arm, load]["steps"], res[arm, load]["client"], res[arm, load]["row"]
            p(f"| {load} | {arm} | {s_['n']} | {f(s_['mean_all'], 2)} | {f(s_['mean_dec'], 3)} | {f(c['period'], 3)} | "
              f"{f(row['duration'])} | {f(c['last_start'], 2)} | {f(c['tail'], 0)} |")
    p("\n## 2. Host collectives (WIDEEP_COLL, every rank; per rank: median over 1000-call lines in the window)\n")
    p("| load | method | arm | ranks | lines | p50 us median over ranks | p50 min..max (rank) | p50 spread max-min | "
      "p99 us median over ranks | p99 max rank |")
    p("|---|---|---|---|---|---|---|---|---|---|")
    for load in LOADS:
        meths = sorted({m for arm, _ in ARMS for m in res[arm, load]["coll"]})
        for m in meths:
            for arm, _ in ARMS:
                c = res[arm, load]["coll"].get(m)
                if not c:
                    continue
                p(f"| {load} | {m} | {arm} | {c['ranks']} | {c['lines']} | {f(c['p50_med'])} | {f(c['p50_min'])} (r{c['lo']})"
                  f"..{f(c['p50_max'])} (r{c['hi']}) | {f(c['p50_spread'])} | {f(c['p99_med'])} | {f(c['p99_max'])} |")
    p("\n## 3. Launch skew (WIDEEP_SKEW: host gap last collective -> graph replay; per 500-replay window k, across ranks)\n")
    p("| load | arm | ranks | windows | gap p50 median us | p50 spread (max-min) median us | p50 spread max | "
      "gap p99 median us | p99 spread median us | p99 spread max | line t span median s | ranks most often slowest p50 |")
    p("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for load in LOADS:
        for arm, _ in ARMS:
            k = res[arm, load]["skew"]
            if k is None:
                p(f"| {load} | {arm} | no WIDEEP_SKEW lines |" + " |" * 9)
                continue
            p(f"| {load} | {arm} | {k['ranks']} | {k['windows']} | {f(k['med50'])} | {f(k['spread50'])} | {f(k['spread50_max'])} | "
              f"{f(k['med99'])} | {f(k['spread99'])} | {f(k['spread99_max'])} | {f(k['tspan'], 2)} | "
              f"{', '.join(f'r{r}:{n}' for r, n in k['hi_rank'])} |")
    p("\n## 4. Admission (WIDEEP_FETCH / WIDEEP_ADMIT, all ranks; per full wave, mean)\n")
    p("Wave = the client waves of section 5 (main start group to the next, from 0.3 s before each), mapped to "
      "fetch calls by the hooks' wall times. Main refill = fetches within [-5, +10] calls of the wave's median "
      "request call; off-phase = the rest, late = off-phase after it. Timeout release = an ADMIT release of context held >= 50 balance calls "
      "(every such hold was 50-59 calls: the 60-iteration gate timeout). Remote slow = rank-0 steps >= 150 ms with no "
      "rank-0 context in that or the 2 prior steps, mapped to fetch calls (call = iter + 3).\n")
    p("| load | arm | waves | calls per wave | requests | fetch calls delivering | main-phase calls | "
      "off-phase calls | off-phase requests | off-phase rank-fetches | late calls | late requests | timeout releases (rank-events) | "
      "distinct release calls | released requests | waves with >=1 release | rank-0 remote slow steps | "
      "holds <10 / 10-49 / 50-69 / >=70 calls |")
    p("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for load in LOADS:
        for arm, _ in ARMS:
            a = res[arm, load]["adm"]
            hb = a["holds"]
            p(f"| {load} | {arm} | {a['waves']} | {f(a['len'], 0)} | {f(a['new'])} | {f(a['calls'], 2)} | "
              f"{f(a['main_calls'], 2)} | {f(a['off_calls'], 2)} | {f(a['off_new'], 2)} | {f(a['off_ranks'], 2)} | "
              f"{f(a['late_calls'], 2)} | {f(a['late_new'], 2)} | "
              f"{f(a['rel_ranks'], 2)} | {f(a['rel_calls'], 2)} | {f(a['rel_req'], 2)} | {a['waves_rel']}/{a['waves']} | "
              f"{f(a['remote'], 2)} | {hb['<10']} / {hb['10-49']} / {hb['50-69']} / {hb['>=70']} |")
    p("\n### 4b. What precedes each rank-0 remote slow step (release on another rank at call c-1 or c)\n")
    p("| load | arm | remote slow steps | after a timeout release | after an immediate release only | none |")
    p("|---|---|---|---|---|---|")
    for load in LOADS:
        for arm, _ in ARMS:
            k = res[arm, load]["adm"]["cause"]
            p(f"| {load} | {arm} | {sum(k.values())} | {k['timeout']} | {k['immediate']} | {k['none']} |")
    p("\n## 5. Client request starts per wave (bench/healthy-<load>.tokens.jsonl; closed loop, start = a completion)\n")
    p("Main group = starts with gaps <= 0.5 s holding >= half the concurrency; wave = main group to the next. "
      "Off-phase = starts between the main group's end and the next main group.\n")
    p("| load | arm | waves | main group size (median) | main span median ms | main span max ms | "
      "off-phase starts per wave (mean) | waves with off-phase | off-phase delay after wave start median s |")
    p("|---|---|---|---|---|---|---|---|---|")
    for load in LOADS:
        for arm, _ in ARMS:
            c = res[arm, load]["client"]
            if c:
                p(f"| {load} | {arm} | {c['waves']} | {f(c['main'], 0)} | {f(c['span'])} | {f(c['span_max'])} | "
                  f"{f(c['off'], 2)} | {c['off_waves']}/{c['waves']} | {f(c['off_delay'], 2)} |")


if __name__ == "__main__":
    report(main(pathlib.Path(sys.argv[1])))
