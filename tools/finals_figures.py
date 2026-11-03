#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Figures 1-9 of review/FINALS_PLAN.md section 2, one PNG + SVG (+ the plotted values as CSV) per figure.

Healthy-cost numbers come only from tools/finals_stats.py (its validity, pairs and summarize(), imported, or its --csv);
recovery numbers only from the harness verdict lines (KILL_FINAL_VERDICT, COLD_RESTART, ACCURACY_KILL_CASE/_VERDICT).
The Fig 5 timeline is the rank-0 iteration log itself (generated tokens per 1 s bin x live ranks, the same deployment
estimate as the verdict's est_tok_s fields). A figure without its inputs is skipped with the reason printed.

Run dirs are classified by content: kill_final.txt (kill matrix), cold_restart.txt (FT off restart),
accuracy_kill.txt (GSM8K), std-<mode>[-variant]-<r> case dirs (healthy sets). Kill / restart modes come from the
case's config-0.yaml (layer_updates_per_iter > 0: online; initial_global_assignments or > 256 slots: offline;
else noeplb).

Usage: python3 -I tools/finals_figures.py --out DIR [--window-s S] [--loads 32,128] [--appendix-loads 8,512]
           [--stats-csv FINALS_STATS.csv] [--label "TEST DATA - not finals"] [--fig4-note TEXT]
           [--footer "build v5-final <hash> | jobs 8xxxx,8yyyy"] RUN_DIR...
"""
import argparse
import csv
import importlib.util
import json
import math
import os
import pathlib
import re
import sys
import tempfile
import time
from collections import Counter, defaultdict

TOOLS = pathlib.Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), f"mplconfig-{os.getuid()}"))
try:
    import matplotlib
except ImportError:
    sys.path.insert(0, str(TOOLS / "pylib"))
    import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

MODES = ("noeplb", "offline", "online")
MODE_LABEL = {"noeplb": "no EPLB", "offline": "offline EPLB", "online": "online EPLB", "?": "mode unknown"}
# Categorical slots 1-3 of the dataviz reference palette (validated all-pairs for three series), then muted ink.
MODE_COLOR = {"noeplb": "#2a78d6", "offline": "#eb6834", "online": "#1baf7a", "?": "#898781"}
INK, INK2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#fcfcfb"
RAMP = ("#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281", "#0d366b")  # ordinal blue 250..700
KIND_LABEL = {"rank5": "rank kill", "rank5pf": "rank kill in prefill", "rank5upd": "rank kill in EPLB update",
              "node1": "node kill (4 ranks)", "twokill": "two sequential losses", "stop": "frozen rank (SIGSTOP)",
              "rank5acc": "rank kill + GSM8K", "rank0": "rank 0 kill (must end)", "deadline": "deadline (must end)"}
KIND_MARKER = {"rank5": "o", "rank5pf": "s", "rank5upd": "D", "node1": "^", "twokill": "v", "stop": "P",
               "rank5acc": "X"}
GPUS = 32
FOOTER = ""  # --footer: printed under every figure

plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 10.5, "axes.edgecolor": "#c3c2b7", "axes.labelcolor": INK2,
    "axes.titlesize": 11, "axes.titlecolor": INK, "xtick.color": INK2, "ytick.color": INK2, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True, "axes.spines.top": False,
    "axes.spines.right": False, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "legend.frameon": False, "lines.linewidth": 2, "lines.solid_capstyle": "round"})


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def kv(line):
    """key=value fields of one verdict line."""
    return dict(m.groups() for m in re.finditer(r"(\w+)=(\S+)", line))


def num(x):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(v) else v


def config_mode(case):
    cfg = case / "config-0.yaml"
    if not cfg.exists():
        return "?"
    text = cfg.read_text(errors="replace")
    if "load_balancer:" not in text:
        return "noeplb"
    upd = re.search(r"layer_updates_per_iter:\s*(\d+)", text)
    slots = re.search(r"num_slots:\s*(\d+)", text)
    if upd and int(upd[1]) > 0:
        return "online"
    if "initial_global_assignments" in text or (slots and int(slots[1]) > 256):
        return "offline"
    return "noeplb"


def job_of(run):
    m = re.search(r"-(\d{6,})(?:-fill\d+)?$", run.name)
    return m[1] if m else run.name


# ---------------------------------------------------------------- inputs

def invalid_cases(run):
    """Case names a run dir's INVALID_CASES file excludes (one per line; "*" = every case), e.g. after a node fault."""
    path = run / "INVALID_CASES"
    return {line.strip() for line in path.read_text().splitlines() if line.strip()} if path.exists() else set()



class Inputs:
    def __init__(self, args):
        self.args, self.healthy, self.kills, self.colds, self.acc_cases, self.acc_verdicts = args, [], [], [], {}, {}
        for run in (pathlib.Path(r).resolve() for r in args.runs):
            if not run.is_dir():
                print(f"note: {run} is not a directory, ignored")
                continue
            if (run / "kill_final.txt").exists():
                self._kills(run)
            if (run / "cold_restart.txt").exists():
                self._colds(run)
            if (run / "accuracy_kill.txt").exists():
                self._accuracy(run)
            if any(p.is_dir() and re.fullmatch(r"std-(noeplb|offline|online)(-\w+)?-\d+", p.name)
                   for p in run.iterdir()):
                self.healthy.append(run)
        self.fs = load_module("finals_stats", TOOLS / "finals_stats.py")
        self.loads = [int(x) for x in args.loads.split(",")]
        self.appendix = [int(x) for x in args.appendix_loads.split(",") if x]
        self.groups = None
        if self.healthy:
            self.groups, self.node, _, _ = self.fs.collect(self.healthy, args.window_s, self.loads)

    def _kills(self, run):
        for line in (run / "kill_final.txt").read_text(errors="replace").splitlines():
            if not line.startswith("KILL_FINAL_VERDICT "):
                continue
            parts = line.split()
            name, verdict = parts[1], parts[2]
            if name in invalid_cases(run) or "*" in invalid_cases(run):
                continue
            f = kv(line)
            case = run / name
            self.kills.append({"run": run, "case": case, "name": name, "kind": name.rsplit("-", 1)[0],
                               "pass": verdict == "PASS", "mode": config_mode(case), "f": f})

    def _colds(self, run):
        for line in (run / "cold_restart.txt").read_text(errors="replace").splitlines():
            if not line.startswith("COLD_RESTART "):
                continue
            f = kv(line)
            again = run / f.get("relaunch", "")
            warm = again.is_dir() and not any(again.glob("dropcache-[0-9]*.log"))
            self.colds.append({"run": run, "kind": line.split()[1], "f": f, "warm": warm,
                               "mode": config_mode(again)})

    def _accuracy(self, run):
        job = job_of(run)
        for line in (run / "accuracy_kill.txt").read_text(errors="replace").splitlines():
            if line.startswith("ACCURACY_KILL_CASE "):
                self.acc_cases[(job, line.split()[1])] = {"run": run, "f": kv(line)}
            elif line.startswith("ACCURACY_KILL_VERDICT "):
                self.acc_verdicts[(job, line.split()[1])] = {"run": run, "pass": line.split()[2] == "PASS"}

    # -- healthy, through finals_stats only

    def pair_rows(self):
        """finals_stats pair rows (FT off -> FT on): from --stats-csv, else from its own collect / pair_up."""
        if self.args.stats_csv:
            with open(self.args.stats_csv) as fh:
                rows = list(csv.DictReader(fh))
            for r in rows:
                for k, v in r.items():
                    if k not in ("group", "set", "mode", "first"):
                        r[k] = float(v)
                r["load"] = int(r["load"])
            return rows
        if not self.groups:
            return []
        pairs, _ = self.fs.pair_up(self.groups, self.node, *self.fs.PAIRS[:2])
        return [self.fs.pair_row(g, n, m, load, r) for (g, n, m, load), r in sorted(pairs.items())]

    def arm_pairs(self, base, other, need):
        """finals_stats-valid pairs as (mode, load, group, base arm dir, other arm dir, pair_row)."""
        if not self.groups:
            return []
        pairs, _ = self.fs.pair_up(self.groups, self.node, base, other, need=need,
                                  shared_loads=(base, other) != tuple(self.fs.PAIRS[:2]))
        owner = {}
        for g, arms in self.groups.items():
            for (mode, rnd, variant), (path, rec) in arms.items():
                for load, metrics in rec["loads"].items():
                    owner[id(metrics)] = path
        out = []
        for (g, n, mode, load), rounds in sorted(pairs.items()):
            row = self.fs.pair_row(g, n, mode, load, rounds)
            for a, b in rounds:
                out.append((mode, load, g, owner[id(a)], owner[id(b)], row))
        return out


# ---------------------------------------------------------------- helpers


def save(fig, out, name, label, rows, header):
    if label:
        fig.text(0.995, 1.0, label, ha="right", va="bottom", fontsize=12, fontweight="bold", color="#d03b3b")
    if FOOTER:  # below everything already drawn (tick labels can sit under the figure's bottom edge)
        bottom = fig.get_tightbbox(fig.canvas.get_renderer()).y0 / fig.get_figheight()
        fig.text(0.01, bottom - 0.01, FOOTER, ha="left", va="top", fontsize=7.5, color=MUTED)
    for ext in ("png", "svg"):
        fig.savefig(out / f"{name}.{ext}", dpi=160, bbox_inches="tight")
    plt.close(fig)
    with open(out / f"{name}.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)
    print(f"rendered {name}: {out / name}.png (+ .svg, .csv)")


def jitter(i, n, width=0.18):
    return 0 if n <= 1 else -width + 2 * width * i / (n - 1)


def mode_legend(ax, modes, extra=(), loc="best"):
    handles = [Line2D([], [], marker="o", ls="", color=MODE_COLOR[m], markersize=7, label=MODE_LABEL[m])
               for m in MODES + ("?",) if m in modes]
    ax.legend(handles=handles + list(extra), loc=loc, fontsize=8)


# ---------------------------------------------------------------- figures


def fig1(inp, out, label):
    rows = [r for r in inp.pair_rows() if r["load"] in inp.loads]
    if not rows:
        return "no finals_stats pair rows at the headline loads (no valid FT off/on pair)"
    fs = inp.fs
    cats = [(m, load) for m in MODES for load in inp.loads]
    metrics = (("trim", "trim-mean decode step", 1.0, "+ = FT slower"),
               ("full", "full-mean step (mean of all steady decode steps)", 1.0, "+ = FT slower"),
               ("client", "client output tok/s/GPU", -1.0, "- = FT slower"))
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2), sharey=False)
    table = []
    for ax, (key, title, line, sign) in zip(axes, metrics):
        for x, (mode, load) in enumerate(cats):
            vals = [r[key] for r in rows if r["mode"] == mode and r["load"] == load]
            s = fs.summarize(vals)
            for i, v in enumerate(vals):
                ax.plot(x + jitter(i, len(vals)), v, "o", color=MODE_COLOR[mode], alpha=0.55, markersize=6,
                        markeredgecolor=SURFACE, markeredgewidth=1)
            if vals:
                if not math.isnan(s["lo"]):
                    ax.errorbar(x + 0.32, s["mean"], yerr=[[s["mean"] - s["lo"]], [s["hi"] - s["mean"]]], fmt="none",
                                ecolor=INK, elinewidth=1.5, capsize=4)
                ax.plot(x + 0.32, s["mean"], "_", color=INK, markersize=14, markeredgewidth=2)
            ax.text(x, 0.01, f"n={s['n']}", transform=ax.get_xaxis_transform(), ha="center", fontsize=8, color=INK2)
            table.append([key, mode, load, s["n"], s["mean"], s["lo"], s["hi"], s["ub"],
                          ";".join(f"{v:.4f}" for v in vals)])
        ax.axhline(line, color="#d03b3b", lw=1.2, ls="--")
        ax.axhline(0, color="#c3c2b7", lw=1)
        ax.text(len(cats) - 0.5, line, f" {line:+.0f}% limit", color=INK2, fontsize=8, va="bottom", ha="right")
        ax.set_xticks(range(len(cats)), [f"{MODE_LABEL[m].replace(' ', chr(10), 1)}\n{load}/GPU" for m, load in cats],
                      fontsize=9)
        ax.set_xlim(-0.6, len(cats) - 0.4)
        ax.set_title(title)
        ax.set_ylabel(f"FT on vs off, % ({sign})")
        ax.grid(axis="x", visible=False)
    handles = [Line2D([], [], marker="o", ls="", color=MUTED, label="one pair (one 8-node group)"),
               Line2D([], [], marker="_", ls="", color=INK, markersize=12, markeredgewidth=2,
                      label="mean, 95% t-CI (finals_stats)")]
    fig.legend(handles=handles, loc="lower left", ncol=2, fontsize=8, bbox_to_anchor=(0.01, -0.02))
    fig.suptitle("Fig 1. FT healthy cost per mode x load (pairs valid per PRE-REG 5; n = pairs per cell)", x=0.01,
                 ha="left", fontsize=11, color=INK)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    save(fig, out, "fig1_ft_cost", label, table,
         ["metric", "mode", "load", "n", "mean_pct", "ci95_lo", "ci95_hi", "one_sided_ub", "pairs_pct"])


def result_row(arm_path, root_runs, load):
    run, case = arm_path.split("/")
    for r in root_runs:
        if r.name == run:
            doc = json.loads((r / case / "result.json").read_text())
            return next(x for x in doc["rows"] if x.get("label") == f"healthy-{load}")
    raise KeyError(arm_path)


def fig_absolute(inp, out, label, loads, name, title):
    if not inp.groups:
        return "no healthy run dirs (std-* case dirs)"
    pairs = [p for p in inp.arm_pairs("", "-ft", ("", "-ft")) if p[1] in loads]
    if not pairs:
        return f"no finals_stats-valid FT off/on pair at loads {loads}"
    metrics = (("output_throughput", "client output tok/s per GPU", lambda row: row["output_throughput"] / row["gpus"]),
               ("median_tpot_ms", "median TPOT (ms)", lambda row: row["median_tpot_ms"]),
               ("median_ttft_ms", "median TTFT (ms)", lambda row: row["median_ttft_ms"]))
    fig, axes = plt.subplots(3, 3, figsize=(13, 9.5), sharey="row", squeeze=False)
    table = []
    for c, mode in enumerate(MODES):
        mp = [p for p in pairs if p[0] == mode]
        for r, (key, ylabel, get) in enumerate(metrics):
            ax = axes[r][c]
            for x, load in enumerate(loads):
                lp = [p for p in mp if p[1] == load]
                for i, (_, _, g, off, on, _) in enumerate(lp):
                    v = [get(result_row(arm, inp.healthy, load)) for arm in (off, on)]
                    xs = [x - 0.15 + jitter(i, len(lp), 0.05), x + 0.15 + jitter(i, len(lp), 0.05)]
                    ax.plot(xs, v, "-", color=MODE_COLOR[mode], lw=1, alpha=0.6)
                    ax.plot(xs[0], v[0], "o", mfc=SURFACE, mec=MODE_COLOR[mode], mew=1.5, ms=7)
                    ax.plot(xs[1], v[1], "o", color=MODE_COLOR[mode], mec=SURFACE, mew=1, ms=7)
                    table.append([mode, load, key, g, off, on, v[0], v[1]])
                if r == 0:
                    ax.text(x, 1.0, f"n={len({p[2] for p in lp})} pairs ({len(lp)} rounds)", transform=ax.get_xaxis_transform(), ha="center", va="bottom",
                            fontsize=8, color=INK2)
            ax.set_xticks(range(len(loads)), [f"{x}/GPU" for x in loads])
            ax.set_xlim(-0.6, len(loads) - 0.4)
            ax.grid(axis="x", visible=False)
            if c == 0:
                ax.set_ylabel(ylabel)
            if r == 0:
                ax.set_title(f"{MODE_LABEL[mode]}\n", pad=2)
    handles = [Line2D([], [], marker="o", ls="", mfc=SURFACE, mec=MUTED, mew=1.5, ms=7, label="FT off"),
               Line2D([], [], marker="o", ls="", color=MUTED, ms=7, label="FT on (line joins one pair)")]
    fig.legend(handles=handles, loc="upper right", ncol=2, fontsize=9, bbox_to_anchor=(0.99, 0.995))
    fig.suptitle(title, x=0.01, ha="left", fontsize=11, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    save(fig, out, name, label, table, ["mode", "load", "metric", "group", "off_arm", "on_arm", "off", "on"])


def fig_relative(inp, out, label, name, title, series, modes, note=""):
    """Descriptive per-pair % vs a base arm: series = [(base, other, need, legend label, color)]."""
    if not inp.groups:
        return "no healthy run dirs (std-* case dirs)"
    data = {}
    for base, other, need, lab, color in series:
        for mode, load, g, a, b, row in inp.arm_pairs(base, other, need):
            if mode in modes and load in inp.loads:
                data.setdefault(lab, {})[(mode, load, g)] = row
    if not any(data.values()):
        return f"no finals_stats-valid pair for {[s[3] for s in series]} in {modes}"
    cats = [(m, load) for m in modes for load in inp.loads]
    fig, axes = plt.subplots(1, 2, figsize=(5 + 2.2 * len(cats), 4.4))
    table = []
    for ax, (key, ttl, sign) in zip(axes, (("trim", "trim-mean decode step", "+ = slower"),
                                           ("client", "client output tok/s/GPU", "- = slower"))):
        for x, (mode, load) in enumerate(cats):
            for k, (_, _, _, lab, color) in enumerate(series):
                vals = [(g, r[key]) for (m, ld, g), r in sorted(data.get(lab, {}).items()) if (m, ld) == (mode, load)]
                xo = x + (k - (len(series) - 1) / 2) * 0.3
                for i, (g, v) in enumerate(vals):
                    ax.plot(xo + jitter(i, len(vals), 0.06), v, "o", color=color, ms=7, mec=SURFACE, mew=1)
                    table.append([key, lab, mode, load, g, v])
                if vals:
                    mean = inp.fs.summarize([v for _, v in vals])["mean"]
                    ax.plot(xo, mean, "_", color=INK, ms=14, mew=2)
                ax.text(xo, 0.01, f"n={len(vals)}", transform=ax.get_xaxis_transform(), ha="center", fontsize=7,
                        color=INK2)
        ax.axhline(0, color="#c3c2b7", lw=1)
        ax.set_xticks(range(len(cats)), [f"{MODE_LABEL[m].replace(' ', chr(10), 1)}\n{load}/GPU" for m, load in cats],
                      fontsize=9)
        ax.set_xlim(-0.6, len(cats) - 0.4)
        ax.grid(axis="x", visible=False)
        ax.set_title(ttl)
        ax.set_ylabel(f"% vs {'main' if series[0][0] == '-main' else 'FT off'} ({sign})")
    handles = [Line2D([], [], marker="o", ls="", color=c, ms=7, label=lab) for _, _, _, lab, c in series]
    handles.append(Line2D([], [], marker="_", ls="", color=INK, ms=12, mew=2, label="mean (descriptive, no CI)"))
    axes[0].legend(handles=handles, loc="best", fontsize=8)
    fig.suptitle(title + (f"\n{note}" if note else ""), x=0.01, ha="left", fontsize=11, color=INK)
    fig.tight_layout()
    save(fig, out, name, label, table, ["metric", "series", "mode", "load", "group", "pct"])


def kill_times(case, ranks):
    """Victims' own WIDEEP_SELF_KILL times, clustered into losses (> 2 s apart)."""
    ts = sorted(float(m[1]) for r in ranks if (case / f"rank-{r}.log").exists()
                for m in re.finditer(r"WIDEEP_SELF_KILL rank \d+ t=([0-9.]+)",
                                     (case / f"rank-{r}.log").read_text(errors="replace")))
    losses = []
    for t in ts:
        if not losses or t - losses[-1] > 2:
            losses.append(t)
    return losses


def kill_load(case):
    """Per-GPU load of the case's kill wave (result.json), '?' without it."""
    try:
        doc = json.loads((case / "result.json").read_text())
    except (OSError, ValueError):
        return "?"
    return next((r.get("load_per_gpu") for r in doc.get("rows", []) if r.get("label") in ("load", "warm-kill")), "?")


ITER = re.compile(r"^\[(\d\d/\d\d/\d{4}-\d\d:\d\d:\d\d)\].*?iter = \d+,.*?num_scheduled_requests = (\d+),.*?"
                  r"currank_total_requests = (\d+)/\d+, host_step_time = ([0-9.]+)ms.*?'num_ctx_requests': (\d+).*?"
                  r"'num_generation_tokens': (\d+)", re.M)
PHASE = re.compile(r"^\[(\d\d/\d\d/\d{4}-\d\d:\d\d:\d\d)\].*WideEP recovery gen=(\d+) phase=(\w+) t=([0-9.]+)", re.M)


def rank0_iters(case):
    """Rank-0 iteration rows (epoch second, scheduled, cumulative requests, host step ms, ctx requests, generated
    tokens) and recovery phases. Log stamps are whole local seconds: the epoch offset is read off the phase lines."""
    log = (case / "rank-0.log").read_text(errors="replace")
    phases, offset = {}, None
    for m in PHASE.finditer(log):
        phases.setdefault((int(m[2]), m[3]), float(m[4]))
        if offset is None:
            offset = round((float(m[4]) - time.mktime(time.strptime(m[1], "%m/%d/%Y-%H:%M:%S"))) / 900) * 900
    if offset is None:
        return [], phases
    rows = [(int(time.mktime(time.strptime(m[1], "%m/%d/%Y-%H:%M:%S")) + offset), int(m[2]), int(m[3]), float(m[4]),
             int(m[5]), int(m[6])) for m in ITER.finditer(log)]
    return rows, phases


def wave_starts(rows, load):
    """Benchmark wave starts: rank 0's cumulative request count rises by >= load/2 within 5 s (a new wave arrives)."""
    starts = []
    for i in range(1, len(rows)):
        t, base = rows[i][0], rows[i - 1][2]
        if rows[i][2] <= base or (starts and t - starts[-1] < 10):
            continue
        if max(r[2] for r in rows[i:i + 5000] if r[0] <= t + 5) >= base + load / 2:
            starts.append(t)
    return starts


def fig5(inp, out, label):
    kinds = ("rank5", "node1", "twokill")
    want = inp.args.timeline_load
    cases = []
    for k in inp.kills:
        if k["kind"] not in kinds or not k["pass"] or not (k["case"] / "rank-0.log").exists():
            continue
        if kill_load(k["case"]) != want:
            print(f"note fig5: {k['run'].name}/{k['name']} dropped (load {kill_load(k['case'])}/GPU, panels are "
                  f"{want}/GPU)")
            continue
        cases.append(k)
    if not cases:
        return f"no passing rank5 / node1 / twokill KILL_FINAL_VERDICT at {want}/GPU with a rank-0 log"
    cases.sort(key=lambda k: (MODES.index(k["mode"]) if k["mode"] in MODES else 9, k["run"].name))
    ncol = max(sum(k["kind"] == kind for k in cases) for kind in kinds)
    fig = plt.figure(figsize=(4.6 * ncol, 4.4 * len(kinds)))
    grid = fig.add_gridspec(2 * len(kinds), ncol, height_ratios=[3, 1.2] * len(kinds), hspace=0.45, wspace=0.25)
    table, lo, hi = [], -10, 30
    for r, kind in enumerate(kinds):
        for c, k in enumerate(x for x in cases if x["kind"] == kind):
            ax, axs = fig.add_subplot(grid[2 * r, c]), fig.add_subplot(grid[2 * r + 1, c])
            name = f"{k['run'].name}/{k['name']}"
            rows, phases = rank0_iters(k["case"])
            losses = kill_times(k["case"], [int(x) for x in k["f"].get("failed_ranks", "").split(",") if x])
            doc = json.loads((k["case"] / "result.json").read_text())
            row = next(x for x in doc["rows"] if x.get("label") in ("load", "warm-kill"))
            starts = [s for s in wave_starts(rows, want) if s >= int(row["start_wall_s"])] if rows and losses else []
            t0 = losses[0] if losses else None
            kw = [s for s in starts if s <= t0] if t0 else []
            ax.set_title(f"{KIND_LABEL[kind]}, {MODE_LABEL[k['mode']]}, {want}/GPU\n{name}", fontsize=8.5)
            if len(kw) < 2:
                why = "no phase lines / kill line" if not rows or not losses else "no earlier wave in the load"
                ax.text(0.5, 0.5, f"no reference: {why}", transform=ax.transAxes, ha="center", color=INK2)
                ax.set_xticks([]), ax.set_yticks([]), axs.axis("off")
                print(f"note fig5: {name}: {why}")
                continue
            wk, wref = kw[-1], kw[-2]
            d = wk - wref
            tok, ctx, full = Counter(), defaultdict(int), defaultdict(int)
            for t, sched, _, _, nctx, gen in rows:
                tok[t] += gen
                ctx[t], full[t] = max(ctx[t], nctx), max(full[t], sched)
            # Reference: the previous wave repeated at its own period d, aligned on each wave's first full decode
            # second after its prefill (no context request in that second, >= load/2 scheduled); before the kill it
            # overlays the actual trace.
            def decode_start(w):
                pre = [t for t in range(w, w + 120) if ctx[t] > 0]
                return next((t for t in range(pre[0], w + 120) if t in tok and ctx[t] == 0 and full[t] >= want / 2),
                            w) if pre else w
            dk, dref = decode_start(wk), decode_start(wref)
            secs = range(math.floor(t0) + lo, math.floor(t0) + hi)
            x = [s - t0 + 0.5 for s in secs]
            act, ref = [tok[s] / 1e3 for s in secs], [tok[dref + (s - dk) % d] / 1e3 for s in secs]
            ax.step(x, ref, where="mid", color=MUTED, lw=1.4, ls=(0, (3, 1.5)))
            ax.step(x, act, where="mid", color=MODE_COLOR[k["mode"]], lw=1.6)
            ax.fill_between(x, act, ref, where=[s >= math.floor(t0) and a < b for s, a, b in zip(secs, act, ref)],
                            step="mid", color="#eb6834", alpha=0.35, lw=0)
            # Service lost, from the wave starts alone: the same wave of work took this much longer.
            nxt = [s for s in starts if s > wk]
            period = nxt[0] - wk if nxt else None
            after = nxt[1] - nxt[0] if len(nxt) > 1 else None
            for gen, tk in enumerate(losses, start=1):
                for a in (ax, axs):
                    a.axvline(tk - t0, color="#d03b3b", lw=1)
                    if (gen, "detect") in phases and (gen, "first_step") in phases:
                        a.axvspan(phases[(gen, "detect")] - t0, phases[(gen, "first_step")] - t0, color=MUTED,
                                  alpha=0.3, lw=0)
            txt = (f"wave length (s): before {d}, failure wave {period if period else '?'}"
                   + (f" ({period - d:+d})" if period else "") + f", after {after if after else '-'}")
            ax.text(0.02, 0.97, txt,
                    transform=ax.transAxes, va="top", fontsize=7.5, color=INK,
                    bbox=dict(facecolor=SURFACE, edgecolor=GRID, alpha=0.9))
            table.append([kind, k["mode"], want, name, wref, wk, round(t0, 3), d, period, after])
            # Step time per iteration; within a whole-second stamp, iterations are placed by cumulative step time.
            pts, cur, acc = [], None, 0.0
            for t, _, _, ms, _, _ in rows:
                acc = acc if t == cur else 0.0
                cur = t
                if lo <= t - t0 <= hi:
                    pts.append((t + min(acc / 1e3, 0.999) - t0, ms))
                acc += ms
            axs.plot([p[0] for p in pts], [p[1] for p in pts], ".", color=MODE_COLOR[k["mode"]], ms=2.5)
            axs.set_yscale("log")
            axs.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
            for a in (ax, axs):
                a.set_xlim(lo, hi)
            ax.set_ylim(bottom=0)
            if c == 0:
                ax.set_ylabel("rank-0 generated\ntokens per s (x1000)")
                axs.set_ylabel("host step\n(ms, log)")
            if r == len(kinds) - 1:
                axs.set_xlabel("seconds from the victim's kill")
    handles = [Line2D([], [], color=MUTED, lw=1.4, ls=(0, (3, 1.5)), label="no-failure reference: the previous "
                      "wave of the same run, repeated at its period, aligned on the first full decode second"),
               Line2D([], [], color=INK2, lw=1.6, label="actual (line color = mode)"),
               Patch(color="#eb6834", alpha=0.35, label="after the kill, below the reference"),
               Line2D([], [], color="#d03b3b", lw=1, label="victim killed"),
               Patch(color=MUTED, alpha=0.3, label="recovery: detect -> first step (rank-0 phase lines)")]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.01, 0.975), ncol=2, fontsize=8)
    fig.suptitle(f"Fig 5. Service through a loss, {want}/GPU: rank-0 generated tokens per 1 s (incl. MTP draft "
                 "tokens; rank 0's own log, not scaled) and host step time", x=0.01, ha="left", fontsize=11, color=INK)
    save(fig, out, "fig5_recovery_timeline", label, table,
         ["kind", "mode", "load", "case", "ref_wave_start", "kill_wave_start", "kill_t",
          "wave_before_s", "failure_wave_s", "wave_after_s"])


PHASE_SEGMENTS = (("detect", "kill -> detected"), ("stopped", "stop"), ("proposed", "agree"),
                  ("ready", "restore experts"), ("committed", "remap + commit"), ("first_step", "resume -> first step"))


def phase_bar(ax, y, phases, start=0.0):
    """Stacked kill -> first step segments from a verdict phases field (seconds from its origin)."""
    prev = start
    for i, (p, _) in enumerate(PHASE_SEGMENTS):
        if p in phases:
            ax.barh(y, phases[p] - prev, left=prev, height=0.6, color=RAMP[i], edgecolor=SURFACE, linewidth=1)
            prev = phases[p]
    return prev


def fig6(inp, out, label):
    order = ("rank5", "rank5pf", "rank5upd", "node1", "twokill")
    runs = [k for k in inp.kills if k["kind"] in order + ("stop",) and k["f"].get("phases1") not in (None, "None")]
    if not runs:
        return "no KILL_FINAL_VERDICT with phases1 (recoverable rows)"
    fast = sorted([k for k in runs if k["kind"] in order],
                  key=lambda k: (order.index(k["kind"]), MODES.index(k["mode"]) if k["mode"] in MODES else 9))
    stops = [k for k in runs if k["kind"] == "stop"]
    rows = []
    for k in fast:
        rows.append((k, "phases1", f"{KIND_LABEL[k['kind']]}" + (" (1st loss)" if k["kind"] == "twokill" else "")))
        if k["kind"] == "twokill" and k["f"].get("phases2") not in (None, "None"):
            rows.append((k, "phases2", "two sequential losses (2nd loss, from detect)"))
    fig, axes = plt.subplots(1, 2, figsize=(14, 0.42 * max(len(rows), len(stops), 4) + 2),
                             gridspec_kw={"width_ratios": (3, 2)})
    table = []
    ax = axes[0]
    for y, (k, field, lab) in enumerate(rows):
        ph = {p: float(v) for p, v in (x.split("=") for x in k["f"][field].split(","))}
        end = phase_bar(ax, y, ph)
        res = num(k["f"].get("resume2_s" if field == "phases2" else "resume_s"))
        if res is not None:
            ax.plot(res, y, "o", color=MODE_COLOR[k["mode"]], ms=8, mec=INK, mew=0.8)
        ax.text(max(end, res or 0) + 0.08, y, f"{k['run'].name}/{k['name']}  {'PASS' if k['pass'] else 'FAIL'}",
                va="center", fontsize=7, color=INK2)
        table.append([k["kind"], k["mode"], f"{k['run'].name}/{k['name']}", field, k["f"][field], res, k["pass"]])
    ax.set_yticks(range(len(rows)), [f"{lab}, {MODE_LABEL[k['mode']]}" for k, _, lab in rows], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("seconds from the victim's kill (2nd loss: from its detection)")
    ax.set_title(f"Losses detected by the peers (n={len(fast)} runs; each run shown, no CI at n=2)")
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0, max([5.0] + [num(k["f"].get("resume_s")) or 0 for k in fast]) * 1.45)
    ax = axes[1]
    if stops:
        for y, k in enumerate(stops):
            ph = {p: float(v) for p, v in (x.split("=") for x in k["f"]["phases1"].split(","))}
            phase_bar(ax, y, ph)
            confirm, resume = num(k["f"].get("h23_confirm_s")), num(k["f"].get("h23_resume_s"))
            if confirm is not None:
                ax.plot(confirm, y, "|", color=INK, ms=16, mew=2)
            if confirm is not None and resume is not None:
                ax.plot(confirm + resume, y, "o", color=MODE_COLOR[k["mode"]], ms=8, mec=INK, mew=0.8)
                ax.text(confirm + resume + 0.6, y, f"confirm {confirm:.1f} + resume {resume:.1f} s", va="center",
                        fontsize=7, color=INK2)
            table.append(["stop", k["mode"], f"{k['run'].name}/{k['name']}", "phases1", k["f"]["phases1"],
                          f"confirm={confirm};resume={resume}", k["pass"]])
        ax.set_yticks(range(len(stops)), [f"{MODE_LABEL[k['mode']]}: {k['run'].name}/{k['name']}" for k in stops],
                      fontsize=7)
        ax.invert_yaxis()
        ax.set_xlim(0, 60)
        ax.set_xlabel("seconds from SIGSTOP")
        ax.grid(axis="y", visible=False)
        ax.set_title(f"Frozen rank: hang confirm + resume (n={len(stops)})")
    else:
        ax.axis("off")
        ax.text(0.5, 0.5, "no frozen-rank (stop) verdict", ha="center", transform=ax.transAxes, color=INK2)
    handles = [Patch(color=RAMP[i], label=lab) for i, (_, lab) in enumerate(PHASE_SEGMENTS)]
    handles += [Line2D([], [], marker="o", ls="", color=MODE_COLOR[m], mec=INK, mew=0.8, ms=7,
                       label=f"resume_s (client sees serving), {MODE_LABEL[m]}")
                for m in MODES + ("?",) if any(k["mode"] == m for k in runs)]
    handles.append(Line2D([], [], marker="|", ls="", color=INK, ms=12, mew=2, label="hang confirmed (stop)"))
    fig.legend(handles=handles, loc="lower center", ncol=5, fontsize=8, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Fig 6. Time to resume per failure type and mode (rank-0 phase lines from KILL_FINAL_VERDICT)",
                 x=0.01, ha="left", fontsize=11, color=INK)
    fig.tight_layout(rect=(0, 0.08, 1, 0.97))
    save(fig, out, "fig6_resume_time", label, table, ["kind", "mode", "case", "field", "phases", "resume_s", "pass"])


def fig7(inp, out, label):
    runs = [k for k in inp.kills if num(k["f"].get("h23_tput_ratio")) is not None and "/" in k["f"].get("h23_live", "")]
    if not runs:
        return "no KILL_FINAL_VERDICT with h23_tput_ratio and h23_live"
    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    table = []
    xs = [eval_frac(k["f"]["h23_live"]) for k in runs]
    lo = min(xs) - 0.03
    grid = [lo, 1.0]
    ax.plot(grid, grid, color=INK2, lw=1.2, label="ideal: live / total")
    ax.plot(grid, [0.8 * g for g in grid], color=INK2, lw=1, ls="--", label="H23 floor: 0.8 x live / total")
    counts = Counter()
    for k, x in zip(runs, xs):
        counts[k["kind"]] += 1
        jit = (hash(k["name"] + k["run"].name) % 7 - 3) * 0.0015
        ax.plot(x + jit, float(k["f"]["h23_tput_ratio"]), KIND_MARKER.get(k["kind"], "o"), color=MODE_COLOR[k["mode"]],
                ms=8, mec=SURFACE, mew=1)
        table.append([k["kind"], k["mode"], f"{k['run'].name}/{k['name']}", k["f"]["h23_live"],
                      k["f"]["h23_tput_ratio"], k["f"].get("est_tok_s_before"), k["f"].get("est_tok_s_after")])
    kinds = [Line2D([], [], marker=KIND_MARKER.get(kd, "o"), ls="", color=MUTED, ms=7,
                    label=f"{KIND_LABEL.get(kd, kd)} (n={counts[kd]})") for kd in counts]
    first = ax.legend(loc="lower right", fontsize=8)
    ax.add_artist(first)
    mode_legend(ax, {k["mode"] for k in runs}, kinds, loc="upper left")
    ax.set_xlabel("live ranks / total ranks after the loss")
    ax.set_ylabel("decode tok/s after / before the loss\n(rank-0 median x live ranks, est_tok_s fields)")
    ax.set_title(f"Fig 7. Post-loss throughput vs live share (n={len(runs)} runs)", loc="left")
    ax.set_xlim(lo, 1.005)
    save(fig, out, "fig7_post_loss_tput", label, table,
         ["kind", "mode", "case", "live", "tput_ratio", "est_tok_s_before", "est_tok_s_after"])


def eval_frac(s):
    a, b = s.split("/")
    return int(a) / int(b)


def fig8(inp, out, label):
    runs = [k for k in inp.kills if num(k["f"].get("completed")) is not None and k["kind"] in KIND_MARKER]
    if not runs:
        return "no KILL_FINAL_VERDICT with request counts (completed / client_short / client_cut)"
    runs.sort(key=lambda k: (list(KIND_MARKER).index(k["kind"]), k["mode"], k["run"].name))
    series = (("full", "completed in full", RAMP[2]), ("rerun", "re-run by the server (in full)", RAMP[4]),
              ("cut", "cut by the loss: error to the client, retryable", "#eb6834"),
              ("reset", "reset before the first token", "#6250d6"),
              ("failed", "client failed", "#d03b3b"))
    fig, ax = plt.subplots(figsize=(10, 0.42 * len(runs) + 2.2))
    table = []
    for y, k in enumerate(runs):
        f = k["f"]
        short = num(f.get("client_short")) or 0
        vals = {"full": num(f["completed"]) - short, "rerun": num(f.get("reruns")), "cut": num(f.get("client_cut")),
                "reset": num(f.get("client_reset")), "failed": num(f.get("client_failed"))}
        for i, (key, _, color) in enumerate(series):
            v = vals[key]
            if v:  # zero counts are listed in the CSV; a log axis has no 0
                ax.plot(v, y + (i - 2) * 0.12, "o", color=color, ms=7, mec=SURFACE, mew=1)
        held = num(f.get("lost_held"))
        if held:
            ax.plot(held, y, "|", color=INK, ms=14, mew=1.5)
        table.append([k["kind"], k["mode"], f"{k['run'].name}/{k['name']}", f["completed"], *(vals[s[0]] for s in series),
                      f.get("lost_held"), f.get("recovery_failed")])
    ax.set_xscale("log")
    ax.set_yticks(range(len(runs)), [f"{KIND_LABEL[k['kind']]}, {MODE_LABEL[k['mode']]}, {kill_load(k['case'])}/GPU  "
                                     f"({k['run'].name}/{k['name']})"
                                     for k in runs], fontsize=7)
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("requests (log scale; zero counts not drawn, see CSV)")
    handles = [Line2D([], [], marker="o", ls="", color=c, ms=7, label=lab) for _, lab, c in series]
    handles.append(Line2D([], [], marker="|", ls="", color=INK, ms=12, mew=1.5, label="in flight on the lost ranks"))
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.45, -0.08 - 1.2 / (len(runs) + 5)), ncol=3,
              fontsize=8)
    ax.set_title(f"Fig 8. Request fate per failure (KILL_FINAL_VERDICT; n={len(runs)} runs, each shown)", loc="left")
    save(fig, out, "fig8_request_fate", label, table,
         ["kind", "mode", "case", "completed", *(s[0] for s in series), "lost_held", "recovery_failed"])


def fig8_gsm8k(inp, out, label):
    want = inp.args.acc_config
    cases = []
    for (job, name), c in sorted(inp.acc_cases.items()):
        f = c["f"]
        if f.get("state") != "DONE" or num(f.get("strict")) is None:
            print(f"note fig8_gsm8k: {job}/{name} skipped (state {f.get('state')})")
            continue
        if f.get("config") != want:
            print(f"note fig8_gsm8k: {job}/{name} skipped (config {f.get('config')} != {want})")
            continue
        arm = "FT off" if f.get("ft") == "False" else "FT on, after a rank kill" if f.get("kill") == "1" else \
            "FT on, healthy"
        v = inp.acc_verdicts.get((job, name))
        cases.append((arm, job, name, float(f["strict"]), num(f.get("strict_se")),
                      "ref" if v is None and arm == "FT off" else ("PASS" if v and v["pass"] else "FAIL")))
    if not cases:
        return f"no DONE ACCURACY_KILL_CASE with config {want}"
    arms = ("FT off", "FT on, healthy", "FT on, after a rank kill")
    color = dict(zip(arms, ("#898781", "#2a78d6", "#eb6834")))
    fig, ax = plt.subplots(figsize=(7, 4.2))
    table = []
    for x, arm in enumerate(arms):
        sel = [c for c in cases if c[0] == arm]
        for i, (_, job, name, acc, se, verdict) in enumerate(sel):
            xo = x + jitter(i, len(sel), 0.15)
            if se:
                ax.errorbar(xo, acc, yerr=se, fmt="none", ecolor=INK2, elinewidth=1, capsize=3)
            ax.plot(xo, acc, "o", color=color[arm], ms=8, mec=SURFACE, mew=1)
            table.append([arm, job, name, acc, se, verdict])
        ax.text(x, 0.01, f"n={len(sel)}", transform=ax.get_xaxis_transform(), ha="center", fontsize=8, color=INK2)
    ax.set_xticks(range(len(arms)), arms)
    ax.grid(axis="x", visible=False)
    ax.set_ylabel("GSM8K strict-match accuracy (1319 questions)")
    ax.set_title(f"Fig 8b. GSM8K appendix, {want} (bars: +/-1 s.e. reported by lm_eval)", loc="left")
    save(fig, out, "fig8b_gsm8k", label, table, ["arm", "job", "case", "strict", "strict_se", "verdict"])


COLD_STAGES = (("exit_s", "failure -> all ranks exited", None), ("weights_s", "relaunch -> weights loaded", "exit_s"),
               ("warmup_s", "autotune / warm-up", "weights_s"), ("graphs_s", "CUDA graph capture", "warmup_s"),
               ("first_token_s", "ready -> first token", "graphs_s"))


def fig9(inp, out, label):
    warm = [c for c in inp.colds if c["warm"] and num(c["f"].get("down_s")) is not None]
    for c in inp.colds:
        if not c["warm"]:
            print(f"note fig9: {c['run'].name} {c['kind']} run={c['f'].get('run')} not drawn (cold restart: the "
                  f"relaunch dropped the page cache; finals compare warm restarts)")
    if not warm:
        return "no warm-restart COLD_RESTART line (relaunch kept the page cache) with down_s"
    kinds = ("rank5", "node1", "stop")
    warm.sort(key=lambda c: (kinds.index(c["kind"]) if c["kind"] in kinds else 9, c["mode"]))
    fig, axes = plt.subplots(1, 2, figsize=(15, 0.5 * len(warm) + 3.2), gridspec_kw={"width_ratios": (3, 2)})
    ax = axes[0]
    table = []
    for y, c in enumerate(warm):
        f = c["f"]
        prev = 0.0
        for i, (key, _, _) in enumerate(COLD_STAGES):
            v = num(f.get(key))
            if v is None:
                continue
            end = v if key == "exit_s" else num(f["exit_s"]) + v
            ax.barh(y, end - prev, left=prev, height=0.6, color=RAMP[i], edgecolor=SURFACE, linewidth=1)
            prev = end
        ft = [num(k["f"].get("resume_s")) for k in inp.kills
              if k["kind"] == c["kind"] and k["mode"] == c["mode"] and num(k["f"].get("resume_s")) is not None]
        ft_txt = f"FT resume {', '.join(f'{v:.1f}' for v in ft)} s" if ft else "no FT run of this type + mode"
        ax.text(prev + 8, y, f"{num(f['down_s']):.0f} s  |  {ft_txt}", va="center", fontsize=7.5, color=INK2)
        table.append([c["kind"], c["mode"], c["run"].name, f.get("run"), f.get("down_s"), f.get("exit_s"),
                      f.get("first_token_s"), ";".join(str(v) for v in ft)])
    ax.set_yticks(range(len(warm)), [f"{KIND_LABEL[c['kind']]}, {MODE_LABEL[c['mode']]} ({c['run'].name} "
                                     f"run {c['f'].get('run')})" for c in warm], fontsize=7)
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0, max(num(c["f"]["down_s"]) for c in warm) * 1.5)
    ax.set_xlabel("seconds from the failure (scheduler wait and harness gap_s excluded)")
    ax.set_title(f"Warm restart (FT off) downtime vs FT resume (n={len(warm)} restarts)")
    ax.legend(handles=[Patch(color=RAMP[i], label=lab) for i, (_, lab, _) in enumerate(COLD_STAGES)],
              loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=5, fontsize=7)
    # Break-even: FT's healthy cost (finals_stats cell at the kill load, client tok/s) vs downtime per failure.
    ax = axes[1]
    load = inp.args.breakeven_load
    rows = [r for r in inp.pair_rows() if r["load"] == load]
    reasons, pts = [], []
    for c in warm:
        vals = [r["client"] for r in rows if r["mode"] == c["mode"]]
        if not vals:
            reasons.append(f"{MODE_LABEL[c['mode']]}: no healthy pair at {load}/GPU")
            continue
        s = inp.fs.summarize(vals)
        cost = -s["mean"] / 100  # client tok/s %: negative = FT slower
        if cost <= 0:
            reasons.append(f"{MODE_LABEL[c['mode']]}: FT cost <= 0 at {load}/GPU (n={s['n']}): no break-even")
            continue
        pts.append((c, cost, s))
    if pts:
        rate_max = max(86400 * cost / num(c["f"]["down_s"]) for c, cost, _ in pts) * 3
        xr = [0, rate_max]
        for c, cost, s in pts:
            down = num(c["f"]["down_s"])
            ax.plot(xr, [x * down * GPUS / 3600 for x in xr], color=RAMP[3], lw=1)
            ft_h = cost * GPUS * 24
            ax.axhline(ft_h, color=MODE_COLOR[c["mode"]], lw=1.5, ls="--")
            be = 86400 * cost / down
            ax.plot(be, ft_h, "o", color=MODE_COLOR[c["mode"]], ms=8, mec=INK, mew=0.8)
            ax.annotate(f"{be:.2f}/day", (be, ft_h), textcoords="offset points", xytext=(6, -12), fontsize=7.5,
                        color=INK2)
            table.append(["breakeven", c["mode"], c["run"].name, c["f"].get("run"), down, f"cost={cost:.5f}",
                          f"n={s['n']}", f"{be:.4f}/day"])
        ax.set_xlim(0, rate_max)
        ax.set_ylim(bottom=0)
        ax.set_xlabel("failures per day (one 32-GPU deployment)\n"
                      "FT resume time and post-loss capacity (fig 7) not counted")
        ax.set_ylabel("GPU-hours lost per day")
        ax.set_title(f"Break-even: FT cost (client tok/s, {load}/GPU, finals_stats mean) vs restarts", fontsize=9)
        ax.legend(handles=[Line2D([], [], color=RAMP[3], lw=1, label="warm restart: rate x down_s x 32 GPUs"),
                           Line2D([], [], color=MUTED, lw=1.5, ls="--", label="FT on: cost x 32 GPUs x 24 h"),
                           Line2D([], [], marker="o", ls="", color=MUTED, mec=INK, label="break-even")],
                  loc="upper left", fontsize=7)
        if reasons:
            ax.text(0.99, 0.02, "\n".join(reasons), transform=ax.transAxes, ha="right", fontsize=7, color=INK2)
    else:
        ax.axis("off")
        ax.text(0.5, 0.5, "break-even not drawn:\n" + "\n".join(sorted(set(reasons)) or ["no healthy data"]),
                ha="center", va="center", transform=ax.transAxes, color=INK2, fontsize=8)
        print("note fig9 break-even panel skipped: " + "; ".join(sorted(set(reasons)) or ["no healthy data"]))
    fig.suptitle("Fig 9. Downtime per failure: warm restart vs FT recovery (COLD_RESTART and KILL_FINAL_VERDICT lines)",
                 x=0.01, ha="left", fontsize=11, color=INK)
    fig.tight_layout()
    save(fig, out, "fig9_downtime", label, table,
         ["kind", "mode", "run", "n", "down_s", "exit_s", "first_token_s", "ft_resume_s"])


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("runs", nargs="+")
    p.add_argument("--out", type=pathlib.Path, required=True)
    p.add_argument("--window-s", type=float, help="finals_stats --window-s (optional; validity is all planned waves completed)")
    p.add_argument("--loads", default="32,128")
    p.add_argument("--appendix-loads", default="8,512")
    p.add_argument("--stats-csv", type=pathlib.Path, help="finals_stats --csv output: Fig 1 / Fig 9 cost from it")
    p.add_argument("--label", default="", help='stamped on every figure, e.g. "TEST DATA - not finals"')
    p.add_argument("--fig4-note", default="", help="Fig 4 subtitle (the gate-fix effect, stated from the readout)")
    p.add_argument("--footer", default="", help='small footer on every figure, e.g. "build v5-final <hash> | jobs ..."')
    p.add_argument("--acc-config", default="dep32-final.yaml", help="GSM8K appendix deployment")
    p.add_argument("--timeline-load", type=int, default=128, help="Fig 5: the kill rows' planned per-GPU load")
    p.add_argument("--breakeven-load", type=int, default=128, help="healthy cell load for the Fig 9 break-even")
    args = p.parse_args()
    global FOOTER
    FOOTER = args.footer
    args.out.mkdir(parents=True, exist_ok=True)
    inp = Inputs(args)
    print(f"inputs: {len(inp.healthy)} healthy run dirs, {len(inp.kills)} kill verdicts, {len(inp.colds)} restart "
          f"lines, {len(inp.acc_cases)} GSM8K cases")
    lab = args.label
    figs = [
        ("fig1", lambda: fig1(inp, args.out, lab)),
        ("fig2", lambda: fig_absolute(inp, args.out, lab, inp.loads, "fig2_modes_abs",
                                      "Fig 2. FT off vs on per mode: throughput, TPOT, TTFT (finals_stats-valid pairs)")),
        ("fig2x", lambda: fig_absolute(inp, args.out, lab, inp.appendix, "fig2x_modes_abs_appendix",
                                       "Fig 2 appendix. Extended loads, same method (labelled appendix)")),
        ("fig3", lambda: fig_relative(inp, args.out, lab, "fig3_noeplb_decomposition",
                                      "Fig 3. No-EPLB decomposition: FT off -> static-LB floor -> FT on (descriptive)",
                                      [("", "-lb", ("-lb",), "static-LB floor vs FT off", "#eb6834"),
                                       ("", "-ft", ("", "-ft"), "FT on vs FT off", "#2a78d6")], ("noeplb",))),
        ("fig4", lambda: fig_relative(inp, args.out, lab, "fig4_main_parity",
                                      "Fig 4. Main parity: our FT off and FT on vs main (same groups, descriptive)",
                                      [("-main", "", ("-main",), "our FT off vs main", "#2a78d6"),
                                       ("-main", "-ft", ("-main",), "our FT on vs main", "#1baf7a")],
                                      ("noeplb", "online"), args.fig4_note)),
        ("fig5", lambda: fig5(inp, args.out, lab)),
        ("fig6", lambda: fig6(inp, args.out, lab)),
        ("fig7", lambda: fig7(inp, args.out, lab)),
        ("fig8", lambda: fig8(inp, args.out, lab)),
        ("fig8b", lambda: fig8_gsm8k(inp, args.out, lab)),
        ("fig9", lambda: fig9(inp, args.out, lab)),
    ]
    skipped = []
    for name, make in figs:
        why = make()
        if why:
            skipped.append(name)
            print(f"SKIPPED {name}: {why}")
    print(f"done: {len(figs) - len(skipped)} rendered, {len(skipped)} skipped {skipped} -> {args.out}")


if __name__ == "__main__":
    main()
