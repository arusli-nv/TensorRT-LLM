#!/usr/bin/env python3
"""Classify rank-0 steps of std cost cases (decode / ctx-only / mixed, refill vs backlog) and compare FT off -> on.

The CLAIMS 170 analysis as one script. Rank 0's iteration lines inside one bench window are classified:
  decode = no context request; ctxonly = context with no generation tokens; mixed = both;
  refill = a context step that fetched new requests (num_fetched > 0); backlog = a context step that fetched none.
A context step after > 20 steps without one starts a wave; pos is the step's index within its wave.
Matched-step comparison: the first 3 full waves (a wave reaching --full-pos: 15 at 128/GPU, 5 at 32/GPU) are aligned by (wave, pos) across cases,
and FT on - off is reported for refill and backlog steps, plus an OLS fit excess = a + b*fetched + c*ctx_ktok.

Usage: ctx_classify.py --load healthy-128 --off CASE_DIR... --on CASE_DIR...
  CASE_DIR is a case directory with rank-0.log and bench.log (e.g. mvp/runs/STD-<job>/std-online-1).
"""

import argparse
import datetime
import json
import pathlib
import re
import statistics as st

LINE = re.compile(
    r"^\[(\d+/\d+/\d+-\d+:\d+:\d+)\].*?iter = (\d+), .*?num_scheduled_requests = (\d+), .*?"
    r"host_step_time = ([\d.]+)ms, prev_device_step_time = ([\d.N/A]+)(?:ms)?, .*?"
    r"'num_ctx_requests': (\d+), 'num_ctx_tokens': (\d+), 'num_generation_tokens': (\d+).*?num_fetched = (\d+)"
)
WAVE_GAP = 20


def utc(t):
    return (t - datetime.datetime(1970, 1, 1)).total_seconds()


def load_case(case, load):
    rows = []
    for line in open(case / "rank-0.log", errors="replace"):
        if "iter = " not in line:
            continue
        m = LINE.search(line)
        if not m:
            continue
        g = m.groups()
        rows.append(
            dict(t=datetime.datetime.strptime(g[0], "%m/%d/%Y-%H:%M:%S"), it=int(g[1]), sched=int(g[2]),
                 host=float(g[3]), pdev=None if g[4] == "N/A" else float(g[4]), nctx=int(g[5]),
                 ctxtok=int(g[6]), gentok=int(g[7]), fetched=int(g[8])))
    # prev_device_step_time on iteration i+1 is the device time of iteration i.
    for a, b in zip(rows, rows[1:] + [None]):
        a["dev"] = b["pdev"] if b is not None and b["it"] == a["it"] + 1 else None
    bench = [json.loads(x) for x in open(case / "bench.log") if x.strip().startswith("{")]
    b = [x for x in bench if x["label"] == load][0]
    # Log stamps are local time; bench windows are UTC seconds: the offset is a whole number of half hours.
    off = round((utc(rows[0]["t"]) - b["start_wall_s"]) / 1800) * 1800
    win = [r for r in rows if b["start_wall_s"] <= utc(r["t"]) - off <= b["start_wall_s"] + b["elapsed_s"]]
    win = [r for r in win if r["sched"] > 0]
    wave, last_ctx, pos = -1, -(10**9), 0
    for i, r in enumerate(win):
        r["cls"] = "decode" if r["nctx"] == 0 else ("ctxonly" if r["gentok"] == 0 else "mixed")
        if r["nctx"] > 0:
            if i - last_ctx > WAVE_GAP:
                wave, pos = wave + 1, 0
            r["wave"], r["pos"] = wave, pos
            pos, last_ctx = pos + 1, i
        r["refill"] = r["nctx"] > 0 and r["fetched"] > 0
    return win, b


def q90(v):
    s = sorted(v)
    return s[min(len(s) - 1, int(round(0.9 * (len(s) - 1))))]


CLASSES = [
    ("decode", lambda r: r["cls"] == "decode"),
    ("ctxonly", lambda r: r["cls"] == "ctxonly"),
    ("mixed", lambda r: r["cls"] == "mixed"),
    ("refill (fetched>0)", lambda r: r["refill"]),
    ("backlog (fetched=0)", lambda r: r["nctx"] > 0 and not r["refill"]),
    ("wave first ctx", lambda r: r["nctx"] > 0 and r["pos"] == 0),
]


def matched(win, full_pos):
    full = sorted({r["wave"] for r in win if r["nctx"] > 0 and r["pos"] >= full_pos})[:3]
    return {(full.index(r["wave"]), r["pos"]): r for r in win
            if r["nctx"] > 0 and r.get("wave") in full and r["dev"]}


def ols(x, y):
    k = len(x[0])
    a = [[sum(v[i] * v[j] for v in x) for j in range(k)] for i in range(k)]
    b = [sum(v[i] * w for v, w in zip(x, y)) for i in range(k)]
    for i in range(k):
        for j in range(i + 1, k):
            f = a[j][i] / a[i][i]
            a[j] = [p - f * c for p, c in zip(a[j], a[i])]
            b[j] -= f * b[i]
    c = [0.0] * k
    for i in reversed(range(k)):
        c[i] = (b[i] - sum(a[i][j] * c[j] for j in range(i + 1, k))) / a[i][i]
    return c


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--load", default="healthy-128")
    p.add_argument("--full-pos", type=int, default=15, help="position a full wave reaches (15 at 128/GPU, 5 at 32/GPU)")
    p.add_argument("--off", nargs="+", required=True)
    p.add_argument("--on", nargs="+", required=True)
    args = p.parse_args()
    cases = {}
    for arm, dirs in (("off", args.off), ("on", args.on)):
        for d in dirs:
            path = pathlib.Path(d)
            cases[f"{arm}:{path.name}"] = load_case(path, args.load)

    for metric in ("host", "dev"):
        print(f"\n### {metric} step ms, median / p90 / mean (n), load {args.load}\n")
        print("| class | " + " | ".join(cases) + " |")
        print("|---|" + "---|" * len(cases))
        for name, sel in CLASSES:
            cells = []
            for win, _ in cases.values():
                v = [r[metric] for r in win if sel(r) and r[metric] is not None]
                cells.append(f"{st.median(v):.1f} / {q90(v):.1f} / {st.mean(v):.1f} ({len(v)})" if v else "-")
            print(f"| {name} | " + " | ".join(cells) + " |")

    print("\n### Totals per case\n")
    print("| case | waves | ctx dev s | ctx dev s, steps < 2 s | decode dev s | client out tok/s | median TTFT ms |")
    print("|---|---|---|---|---|---|---|")
    for name, (win, b) in cases.items():
        waves = max((r.get("wave", -1) for r in win), default=-1) + 1
        ctx_s = sum(r["dev"] or 0 for r in win if r["nctx"] > 0) / 1e3
        ctx_short_s = sum(r["dev"] for r in win if r["nctx"] > 0 and r["dev"] and r["dev"] < 2000) / 1e3
        dec_s = sum(r["dev"] or 0 for r in win if r["nctx"] == 0) / 1e3
        print(f"| {name} | {waves} | {ctx_s:.1f} | {ctx_short_s:.1f} | {dec_s:.1f} | {b.get('output_throughput', float('nan')):.0f} "
              f"| {b.get('median_ttft_ms', float('nan')):.0f} |")

    m = {name: matched(win, args.full_pos) for name, (win, _) in cases.items()}
    off = [n for n in m if n.startswith("off:")]
    on = [n for n in m if n.startswith("on:")]
    keys = sorted(set.intersection(*[set(v) for v in m.values()]))
    print(f"\n### Matched context steps, first 3 full waves ({len(keys)} keys), FT on - off device ms\n")
    if not keys:
        print("no common (wave, pos) keys")
        return
    x, y, refill, backlog = [], [], [], []
    for k in keys:
        excess = st.mean(m[n][k]["dev"] for n in on) - st.mean(m[n][k]["dev"] for n in off)
        fetched = st.mean(m[n][k]["fetched"] for n in on + off)
        x.append([1, fetched, m[on[0]][k]["ctxtok"] / 1e3])
        y.append(excess)
        (refill if fetched > 0 else backlog).append(excess)
    fmt = lambda v: f"{st.mean(v):+.1f} (n={len(v)})" if v else "-"
    print(f"- mean of arms: refill {fmt(refill)}, backlog {fmt(backlog)}")
    if len(keys) >= 3:
        print("- fit: excess = %.1f %+.2f*fetched %+.1f*ctx_ktok" % tuple(ols(x, y)))
    for a, b in zip(on, off):
        kk = set(m[a]) & set(m[b])
        ref = [m[a][k]["dev"] - m[b][k]["dev"] for k in kk if m[a][k]["fetched"] + m[b][k]["fetched"] > 0]
        bl = [m[a][k]["dev"] - m[b][k]["dev"] for k in kk if m[a][k]["fetched"] + m[b][k]["fetched"] == 0]
        print(f"- {a} - {b}: refill {fmt(ref)}, backlog {fmt(bl)}")


if __name__ == "__main__":
    main()
