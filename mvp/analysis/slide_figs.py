# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Slide figures R1-R6 (mvp/SLIDES.md) regenerated from run directories.

Every plotted number is read from the run files; anything not measured is
drawn hatched/grey and stamped "MOCK" on the figure.

    python3 slide_figs.py --kill-runs RUN... --restart-runs RUN... --out DIR
"""
import argparse
import datetime as dt
import glob
import json
import os
import re
import statistics

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

plt.rcParams.update({"font.size": 18, "axes.titlesize": 22, "axes.labelsize": 19,
                     "legend.fontsize": 15, "axes.spines.top": False,
                     "axes.spines.right": False})
FIGSIZE, DPI, GPUS = (13.33, 7.5), 150, 32
FAILURES_PER_DAY = 46  # Dynamo trace (SLIDES.md slide 2), an input, not a run measurement
C_RANK, C_NODE, C_OFF, C_MOCK = "#1f77b4", "#ff7f0e", "#d62728", "#9e9e9e"
PHASES = ["detect", "stopped", "proposed", "ready", "install", "installed",
          "committed", "resumed", "first_step"]
ITER_RE = re.compile(r"^\[(\d\d/\d\d/\d{4}-\d\d:\d\d:\d\d)\].*?iter = \d+,.*?"
                     r"host_step_time = ([\d.]+)ms.*?'num_generation_tokens': (\d+)")
PHASE_RE = re.compile(r"WideEP recovery gen=1 phase=(\w+) t=([\d.]+)")


def kv(line):
    return dict(tok.split("=", 1) for tok in line.split() if "=" in tok)


def num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def mock(ax, text="MOCK"):
    ax.text(0.5, 0.5, text, transform=ax.transAxes, ha="center", va="center",
            fontsize=40, color="red", alpha=0.35, fontweight="bold", rotation=15)


def save(fig, out, name):
    path = os.path.join(out, name)
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print("wrote", path)


# ---------------------------------------------------------------- loading
def read_lines(path, prefix):
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return [kv(l) | {"_case": l.split()[1]} for l in f if l.startswith(prefix + " ")]


def scan_rank0(case_dir):
    """Stream rank-0.log: (naive_stamp_s, step_s, gen_tokens) per iteration, phase times."""
    steps, phases = [], {}
    with open(os.path.join(case_dir, "rank-0.log"), errors="replace") as f:
        for line in f:
            if "iter = " in line:
                m = ITER_RE.match(line)
                if m:
                    stamp = dt.datetime.strptime(m[1], "%m/%d/%Y-%H:%M:%S")
                    stamp = stamp.replace(tzinfo=dt.timezone.utc).timestamp()
                    steps.append((stamp, float(m[2]) / 1e3, int(m[3])))
            elif "WideEP recovery gen=1" in line:
                m = PHASE_RE.search(line)
                if m:
                    phases.setdefault(m[1], float(m[2]))
    return steps, phases


def step_walls(steps, ref_wall):
    """Per-step wall end times: cumulative step times clamped into each 1 s stamp.

    Log stamps are local time without a zone, so the whole-quarter-hour offset
    is taken from ref_wall (an epoch known to fall within the log's span).
    """
    if not steps:
        return []
    off = round((ref_wall - steps[-1][0]) / 900) * 900
    out, t = [], -1e18
    for stamp, step_s, tok in steps:
        s = stamp + off
        t = min(max(s, t + step_s), s + 0.999)
        out.append((t, step_s, tok))
    return out


def rate_series(walls, t0, live_after, t_detect, lo, hi, binw=0.5):
    """Rank-0 generation tokens/s x live ranks, binned, relative to t0."""
    nb = int((hi - lo) / binw)
    toks = [0] * nb
    for t, _, tok in walls:
        i = int((t - t0 - lo) / binw)
        if 0 <= i < nb:
            toks[i] += tok * (GPUS if t < t_detect else live_after)
    return [lo + (i + 0.5) * binw for i in range(nb)], [x / binw / 1e3 for x in toks]


def load_kill_runs(runs):
    verdicts, cases = [], []
    for run in runs:
        for v in read_lines(os.path.join(run, "kill_final.txt"), "KILL_FINAL_VERDICT"):
            case = os.path.join(run, v["_case"])
            v["live"] = GPUS - len(v.get("failed_ranks", "").split(","))
            v["kind"] = "rank" if v["_case"].startswith("rank") else "node"
            verdicts.append(v)
            if os.path.exists(os.path.join(case, "result.json")):
                res = json.load(open(os.path.join(case, "result.json")))
                cases.append(dict(v, dir=case, res=res))
    return verdicts, cases


def load_restart_runs(runs):
    out = []
    for run in runs:
        for c in read_lines(os.path.join(run, "cold_restart.txt"), "COLD_RESTART"):
            c["dir"] = os.path.join(run, c["_case"] + "-" + c.get("run", "1"))
            c["run_name"] = os.path.basename(run.rstrip("/"))
            out.append(c)
    return out


# ---------------------------------------------------------------- figures
def fig_r1(cases, restarts, out):
    fig, (a, b) = plt.subplots(1, 2, figsize=FIGSIZE, gridspec_kw={"width_ratios": [1, 1.4]})
    for kind, color in (("rank", C_RANK), ("node", C_NODE)):
        sel = [c for c in cases if c["kind"] == kind][:1]
        for c in sel:
            trig = c["res"]["trigger_wall_s"]
            steps, ph = scan_rank0(c["dir"])
            walls = step_walls(steps, trig)
            x, y = rate_series(walls, trig, c["live"], ph.get("detect", trig), -5, 10, 0.5)
            lab = f"FT on, {kind} kill ({c['live']}/{GPUS} left), {os.path.basename(c['dir'])}"
            a.plot(x, y, color=color, lw=2.5, label=lab)
            if "first_step" in ph:
                a.axvline(ph["first_step"] - trig, color=color, ls=":", lw=1.5)
    for i, r in enumerate(restarts[:2]):
        res_path = os.path.join(r["dir"], "result.json")
        if not os.path.exists(res_path):
            continue
        trig = json.load(open(res_path))["trigger_wall_s"]
        steps, _ = scan_rank0(r["dir"])
        walls = step_walls(steps, trig)
        x, y = rate_series(walls, trig, GPUS, trig + 1e9, -20, 0, 1.0)
        down = num(r.get("down_s"))
        lab = f"FT off, rank kill + restart ({r['run_name']})" if i == 0 else None
        b.plot(x + [0, 0, down], y + [y[-1] if y else 0, 0, 0], color=C_OFF, lw=2.5, label=lab)
        pre = statistics.median(y[-5:]) if y else 0
        b.plot([down, down, 380], [0, pre, pre], color=C_MOCK, ls="--", lw=2,
               label="after restart: assumed pre-kill rate (MOCK)" if i == 0 else None)
        for key, ls in (("ready_s", ":"), ("first_token_s", "-.")):
            if num(r.get(key)) is not None and i == 0:
                b.axvline(num(r[key]), color=C_OFF, ls=ls, lw=1.2)
                b.text(num(r[key]), 0.6, f" {key}={r[key]} s", rotation=90, fontsize=12,
                       color=C_OFF, transform=b.get_xaxis_transform())
        b.annotate(f"down {down:.0f} s", (down / 2, pre * 0.5), ha="center", fontsize=20,
                   color=C_OFF)
    a.set(title="FT on: dip, then reduced level", xlabel="seconds from kill",
          ylabel="tokens/s (k), rank-0 proxy x live ranks", xlim=(-5, 10))
    b.set(title="FT off: restart gap", xlabel="seconds from kill", xlim=(-20, 380))
    a.set_ylim(bottom=0)
    b.set_ylim(bottom=0)
    a.legend(loc="lower right", fontsize=12)
    b.legend(loc="lower right", fontsize=12)
    fig.suptitle("Throughput around a failure (rank-0 generation tokens/s x live ranks: a proxy)",
                 fontsize=20)
    fig.tight_layout()
    save(fig, out, "R1_throughput_timeline.png")


def fig_r2(cases, out):
    fig, ax = plt.subplots(figsize=FIGSIZE)
    rows = sorted(cases, key=lambda c: (c["kind"] != "rank", c["dir"]))
    colors = plt.cm.tab10.colors
    for y, c in enumerate(rows):
        _, ph = scan_rank0(c["dir"])
        t = [(p, ph[p] - ph["detect"]) for p in PHASES if p in ph] if "detect" in ph else []
        for (p0, t0), (p1, t1) in zip(t, t[1:]):
            ax.barh(y, t1 - t0, left=t0, color=colors[PHASES.index(p1) % 10],
                    label=f"{p0}->{p1}")
        if t:
            ax.text(t[-1][1] + 0.02, y, f"{t[-1][1]:.2f} s", va="center", fontsize=15)
    ax.set_yticks(range(len(rows)),
                  [f"{c['kind']} {os.path.basename(os.path.dirname(c['dir']))[-14:]}/"
                   f"{os.path.basename(c['dir'])}" for c in rows], fontsize=12)
    h, l = ax.get_legend_handles_labels()
    uniq = dict(zip(l, h))
    ax.legend(uniq.values(), uniq.keys(), loc="lower right", fontsize=12, ncol=2)
    ax.set(xlabel="seconds from detect (rank-0 phase stamps)",
           title="Recovery phases: rank kill vs node kill")
    ax.invert_yaxis()
    save(fig, out, "R2_recovery_gantt.png")


def fig_r3(verdicts, out):
    fig, ax = plt.subplots(figsize=FIGSIZE)
    groups = []
    for kind, color in (("rank", C_RANK), ("node", C_NODE)):
        vs = [v for v in verdicts if v["kind"] == kind and num(v.get("r0_tok_s_before"))]
        if not vs:
            continue
        live = vs[0]["live"]
        kept = [num(v["r0_tok_s_after"]) * v["live"] / (num(v["r0_tok_s_before"]) * GPUS) * 100
                for v in vs]
        groups.append((f"lose 1 {kind}\n({live}/{GPUS} GPUs)", live / GPUS * 100,
                       statistics.median(kept), kept, color, False))
    groups.append((f"lose 2 ranks\n(30/{GPUS} GPUs)", 30 / GPUS * 100, 30 / GPUS * 100, [],
                   C_MOCK, True))
    for i, (lab, share, med, pts, color, is_mock) in enumerate(groups):
        ax.bar(i, med, color=color, width=0.6, hatch="//" if is_mock else None,
               alpha=0.5 if is_mock else 0.9)
        ax.hlines(share, i - 0.4, i + 0.4, colors="k", linestyles="--", lw=2)
        ax.scatter([i] * len(pts), pts, color="k", zorder=3, s=40)
        ax.text(i, med + 1, "MOCK" if is_mock else f"{med:.1f}% (n={len(pts)})",
                ha="center", fontsize=18, color="red" if is_mock else "k")
    ax.set_xticks(range(len(groups)), [g[0] for g in groups])
    ax.set(ylim=(70, 102), ylabel="throughput kept, % of before",
           title="Throughput kept vs share of GPUs left (dashed)")
    ax.text(0.01, 0.02, "kept = r0_tok_s_after x live / (r0_tok_s_before x 32); dots = repeats",
            transform=ax.transAxes, fontsize=13)
    save(fig, out, "R3_throughput_kept.png")


def fig_r4(verdicts, cases, restarts, out):
    fig, ax = plt.subplots(figsize=FIGSIZE)
    inflight = next((r["concurrency"] for c in cases for r in c["res"].get("rows", [])
                     if r.get("label") == "load"), None)
    cats = {"continued from exact tokens": "#2ca02c", "finished before the abort": "#98df8a",
            "retryable error (held on lost ranks)": "#ffbf00", "failed": C_OFF}
    bars = []
    for kind in ("rank", "node"):
        vs = [v for v in verdicts if v["kind"] == kind]
        if vs and inflight:
            cut = statistics.median(num(v["client_cut"]) for v in vs)
            hard = statistics.median(num(v["client_failed"]) for v in vs)
            bars.append((f"FT on\n{kind} kill (n={len(vs)})", [inflight - cut - hard, 0, cut, hard]))
    for r in restarts if inflight else []:
        failed = num(r["failed_inflight"])
        bars.append((f"FT off restart\n{r['run_name'][:14]}", [0, inflight - failed, 0, failed]))
    seen = set()
    for i, (_, counts) in enumerate(bars):
        bottom = 0
        for (cat, color), val in zip(cats.items(), counts):
            ax.bar(i, val, bottom=bottom, color=color, width=0.6,
                   label=cat if val and cat not in seen else None)
            if val:
                seen.add(cat)
            if val:
                ax.text(i, bottom + val / 2, f"{int(val)}", ha="center", va="center", fontsize=16)
            bottom += val
    ax.set_xticks(range(len(bars)), [b[0] for b in bars], fontsize=14)
    ax.set(ylabel=f"requests in flight at the kill ({inflight})", ylim=(0, (inflight or 1) * 1.05),
           title="Fate of the in-flight requests at the kill")
    ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0), fontsize=13)
    save(fig, out, "R4_request_fate.png")


def fig_r5(cases, out):
    fig, ax = plt.subplots(figsize=FIGSIZE)
    files = [f for c in cases for f in glob.glob(os.path.join(c["dir"], "bench", "*.tokens.jsonl"))]
    if not files:
        ax.set(xlabel="seconds from kill", ylabel="per-request latency (ms)",
               title="Client latency over time around a kill")
        mock(ax, "MOCK: needs timed client\n(no bench/*.tokens.jsonl)")
        save(fig, out, "R5_client_latency.png")
        return
    for f, color in zip(files[:2], (C_RANK, C_NODE)):
        case = next(c for c in cases if f.startswith(c["dir"]))
        trig = case["res"]["trigger_wall_s"]
        xs, ys = [], []
        with open(f) as fh:
            for line in fh:
                rec = json.loads(line)
                ts = rec.get("token_wall_s") or rec.get("token_times") or []
                for t0, t1 in zip(ts, ts[1:]):
                    xs.append(t1 - trig)
                    ys.append((t1 - t0) * 1e3)
        ax.scatter(xs, ys, s=2, color=color, alpha=0.3, label=os.path.basename(case["dir"]))
    ax.set(xlabel="seconds from kill", ylabel="inter-token latency (ms)", yscale="log",
           title="Client inter-token latency around a kill")
    ax.legend()
    save(fig, out, "R5_client_latency.png")


def fig_r6(verdicts, restarts, out):
    fig, ax = plt.subplots(figsize=FIGSIZE)
    down = statistics.median(num(r["down_s"]) for r in restarts) if restarts else None
    bars = [("FT off\n(restart)", [(down * GPUS / 3600 * FAILURES_PER_DAY, C_OFF,
                                    f"down {down:.0f} s x {GPUS} GPUs")])] if down else []
    for kind, color in (("rank", C_RANK), ("node", C_NODE)):
        vs = [v for v in verdicts if v["kind"] == kind]
        if not vs or not down:
            continue
        resume = statistics.median(num(v["resume_s"]) for v in vs)
        kept = statistics.median(num(v["r0_tok_s_after"]) * v["live"] /
                                 (num(v["r0_tok_s_before"]) * GPUS) for v in vs)
        bars.append((f"FT on\n{kind} loss", [
            (resume * GPUS / 3600 * FAILURES_PER_DAY, color, f"stall {resume:.2f} s x {GPUS}"),
            ((1 - kept) * GPUS * down / 3600 * FAILURES_PER_DAY, C_MOCK,
             f"{(1 - kept) * 100:.0f}% capacity lost for {down:.0f} s (assumed)")]))
    for i, (lab, parts) in enumerate(bars):
        bottom = 0
        for val, color, txt in parts:
            ax.bar(i, val, bottom=bottom, color=color, width=0.6)
            ax.text(i + 0.33, bottom + val / 2, f"{txt}: {val:.2f}", va="center", fontsize=13)
            bottom += val
        ax.text(i, bottom * 1.02, f"{bottom:.1f} GPU-h/day", ha="center", fontsize=18)
    ax.set_xticks(range(len(bars)), [b[0] for b in bars])
    ax.set(ylabel="GPU-hours lost per day", title=f"Cost of failures at {FAILURES_PER_DAY}/day")
    ax.text(0.01, 0.97, f"Assumptions: {FAILURES_PER_DAY} failures/day (Dynamo trace); FT off "
            f"idles all {GPUS} GPUs for down_s;\nFT on stalls {GPUS} GPUs for resume_s, then "
            "runs at the measured kept share until a restart-length repair (grey, assumed)",
            transform=ax.transAxes, va="top", fontsize=12)
    save(fig, out, "R6_cost_per_day.png")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--kill-runs", nargs="+", default=[])
    p.add_argument("--restart-runs", nargs="+", default=[])
    p.add_argument("--out", required=True)
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    verdicts, cases = load_kill_runs(a.kill_runs)
    restarts = load_restart_runs(a.restart_runs)
    print(f"{len(verdicts)} verdicts, {len(cases)} kill cases, {len(restarts)} restarts")
    fig_r1(cases, restarts, a.out)
    fig_r2(cases, a.out)
    fig_r3(verdicts, a.out)
    fig_r4(verdicts, cases, restarts, a.out)
    fig_r5(cases, a.out)
    fig_r6(verdicts, restarts, a.out)


if __name__ == "__main__":
    main()
