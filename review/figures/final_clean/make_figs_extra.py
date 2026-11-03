"""Extra evaluation figures (Fig 9, A3-A5) in the TensorRT-LLM tech blog 30 chart style (reference/*.png).

usage: /usr/bin/python3 make_figs_extra.py   (writes png + svg + csv next to this script)
       /usr/bin/python3 make_figs_extra.py --slide   (writes <fig>_slide.png/.svg: near-square canvases, panels
       stacked, for the right half of a slide; the doc figures and CSVs are left untouched)

Fig 9  kill timeline: product stress runs S2 (STRESS-8299473/s2-rank14) and S3 (STRESS-8314070/s3-rank29-rank14);
       served output tok/s per 10 s bin from each stream's per-second chunk counts x its tokens per chunk (the same
       rule as mvp/harness/stress_readout.py chunk_seconds), kill times from result.json.
Fig A3 a2a ablation: review/diag/ablate-8293196/verdict.txt (CLAIMS 210).
Fig A4 restart vs FT: the pooled restarts of fig0_today_restart_slide.csv (CLAIMS 224, 228) and the relaunches' rank-0
       total_model_loading_seconds (CLAIMS 211); FT recovery = fig7_recovery.csv rank-kill means (CLAIMS 193/202/203/213).
Fig A5 remap phases: "Remap phases ms" lines of KILL-FINAL-{OFF,OFF-B,ON}-8308514 (CLAIMS 214).
"""
import csv
import json
import math
import os
import re
import statistics
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

ROOT = "/lustre/fs1/portfolios/coreai/projects/coreai_comparch_trtllm/users/arusli/src/TensorRT-LLM/.wideep-ft-runs/claude"
RUNS, REV = f"{ROOT}/mvp/runs", f"{ROOT}/review"
OUT = os.path.dirname(os.path.abspath(__file__))
for f in ("Lato-Regular.ttf", "Lato-Bold.ttf"):
    font_manager.fontManager.addfont(f"{ROOT}/mvp/analysis/fonts/{f}")
# Blog 30 palette: warm grey page, dark ink for titles and values, blue-grey for labels.
BG, AXIS, GRID = "#FFFFFF", "#BFC0C0", "#D5D5D5"
INK, LABEL = "#2D3142", "#4F5D75"
GREEN, PALE_GREEN = "#458F19", "#EBF1E0"  # FT on; darker than lime #76B900 for contrast on BG
SLATE = ["#2F3E4E", "#6B7A89", "#A9B4BE", "#DCE1E6"]  # components (outline / fill shades), never green
ACCENT, PALE_ACCENT = "#D9822B", "#FBEBDB"  # the key component
MINUS = "−"
# Fixed canvas width shared with make_figs.py: 2056 px at 200 dpi, title block at a 60 px left margin. Sizes are given
# in canvas px (SIZE_SPEC.md) and converted with PX; the canvas height grows per figure when text needs it.
DPI, W_PX, H_PX, LEFT_PX = 200, 2056, 1064, 60
PX = 72 / DPI  # pt per canvas px
REF = "#4A5568"  # reference / threshold lines
SLIDE = "--slide" in sys.argv[1:]


def half_up(v, digits=1):
    """Round half up (binary floats otherwise round .x5 either way), as make_figs.py does."""
    f = 10 ** digits
    return math.floor(v * f + 0.5 + 1e-9) / f
SLIDE_FIGS = {"fig9_kill_timeline", "figA3_a2a_ablation", "figA4_restart_vs_ft", "figA5_remap_phases"}
plt.rcParams.update({"font.family": "Lato", "font.size": 28 * PX, "axes.titlesize": 34 * PX, "axes.titleweight": "bold",
                     "axes.titlecolor": INK, "axes.labelsize": 30 * PX, "xtick.labelsize": 28 * PX,
                     "ytick.labelsize": 28 * PX, "legend.fontsize": 28 * PX, "figure.facecolor": BG,
                     "axes.facecolor": BG, "savefig.facecolor": BG,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": False,
                     "axes.edgecolor": AXIS, "axes.linewidth": 1.0, "axes.labelcolor": LABEL,
                     "xtick.color": LABEL, "ytick.color": LABEL, "xtick.major.size": 0, "ytick.major.size": 0,
                     "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRID, "grid.linewidth": 1.5 * PX,
                     "grid.linestyle": (0, (1, 3)), "axes.axisbelow": True, "legend.frameon": False,
                     "axes.unicode_minus": True, "savefig.dpi": DPI, "svg.fonttype": "none"})
MODES = [("noeplb", "No EPLB"), ("offline", "Offline EPLB"), ("online", "Online EPLB")]


def size(h_px=H_PX, slide_h_px=None):
    """Doc canvas, or in slide mode the same width (same text scale) at slide_h_px tall."""
    return (W_PX / DPI, (slide_h_px if SLIDE and slide_h_px else h_px) / DPI)


def box(face, edge, label, lw=3 * PX):
    """Outlined square legend glyph."""
    return Patch(facecolor=face, edgecolor=edge, lw=lw, label=label)


def finish(fig, title, sub, handles, top=None, h_pad=None):
    """Bold title, one grey subtitle line and one legend row at the fixed 60 px left margin; the axes fill the rest
    of the canvas."""
    fx, fy = 1 / W_PX, 1 / (fig.get_figheight() * DPI)
    y_sub = 1 - 100 * fy
    y_leg = y_sub - 58 * fy
    head = top or y_leg - 64 * fy
    fig.tight_layout(rect=(30 * fx, 24 * fy, 1 - 30 * fx, head), w_pad=2.5, h_pad=h_pad)
    x0 = LEFT_PX * fx
    fig.text(x0, 1 - 40 * fy, title, ha="left", va="top", fontsize=40 * PX, fontweight="bold", color=INK)
    fig.text(x0, y_sub, sub, ha="left", va="top", fontsize=30 * PX, color=LABEL)
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(x0 - 0.004, y_leg), ncol=len(handles),
               handlelength=1.1, handleheight=1.1, columnspacing=2.0, labelcolor=LABEL, borderaxespad=0)


def save(fig, name, rows, header, prov):
    if SLIDE:  # figure files only; the doc run writes the CSVs
        if name in SLIDE_FIGS:
            fig.savefig(f"{OUT}/{name}_slide.png")
            fig.savefig(f"{OUT}/{name}_slide.svg")
            print(f"{name}_slide: {fig.get_figwidth() * DPI:.0f} x {fig.get_figheight() * DPI:.0f} px")
        plt.close(fig)
        return
    fig.savefig(f"{OUT}/{name}.png")
    fig.savefig(f"{OUT}/{name}.svg")
    plt.close(fig)
    with open(f"{OUT}/{name}.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)
    print(f"{name}: {prov}")


def value(ax, x, y, text, **kw):
    kw = {"ha": "center", "va": "bottom", "fontsize": 28 * PX, "fontweight": "bold", "color": INK, **kw}
    ax.text(x, y, text, **kw)


# ---------------------------------------------------------------- Fig 9: throughput around a kill (stress runs)
BIN, PRE, POST = 10, 120, 300
H9 = 1064


def stress_case(case):
    """Per-second served output tokens (chunk_seconds rule of stress_readout.py) and the requests' (start, exp)."""
    recs = []
    for row in ("stress", "stress-chat"):
        with open(f"{case}/bench/{row}.sse.jsonl") as fh:
            for line in fh:
                try:
                    recs.append(json.loads(line))
                except ValueError:
                    pass
    per_chunk = [r["tok"] / r["chunks"] for r in recs if r.get("tok") and r.get("chunks")]
    default = statistics.median(per_chunk)
    sec = {}
    for r in recs:
        k = r["tok"] / r["chunks"] if r.get("tok") and r.get("chunks") else default
        for s, n in (r.get("ts") or {}).items():
            sec[int(s)] = sec.get(int(s), 0.0) + n * k
    starts = [(r["start"], r["exp"]) for r in recs if r.get("start") and r.get("exp")]
    return sec, starts, json.load(open(f"{case}/result.json"))["kills"]


# Every loss under realistic traffic (CLAIMS 216, 218, 219, 236). The doc figure draws the first three (DRAWN); the
# CSV and the slide chart carry all eight.
S2 = f"{RUNS}/STRESS-8299473/s2-rank14"
S3 = f"{RUNS}/STRESS-8314070/s3-rank29-rank14"
CASES9 = [(S2, "S2"), (f"{RUNS}/STRESS-G1-8299473/s2-rank13", "S2"), (S3, "S3"),
          (f"{RUNS}/STRESS-G1-8314070/s3-rank12-rank31", "S3"), (f"{RUNS}/STRESS-8318006/s2-rank25", "S2"),
          (f"{RUNS}/STRESS-G1-8318006/s2-node2", "S2")]
DRAWN = {S2, S3}
data, extra = [], []
for case, label in CASES9:
    sec, starts, kills = stress_case(case)
    for k, kill in enumerate(kills, 1):
        t0 = math.floor(kill["kill_wall_s"])
        edges = list(range(-PRE, POST, BIN))
        served = [sum(sec.get(t0 + e + i, 0.0) for i in range(BIN)) / BIN for e in edges]
        # Offered load: output tokens requested by the requests that arrived in the plotted window, per second.
        offered = sum(x for s, x in starts if t0 - PRE <= s < t0 + POST) / (PRE + POST)
        (data if case in DRAWN else extra).append((label, k, kill, edges, served, offered, os.path.relpath(case, RUNS)))
# Healthy bin noise: +-2 SD of the pre-kill bins' deviation from the offered load, pooled over the three panels.
pre = [v / off - 1 for _, _, _, edges, served, off, _ in data for e, v in zip(edges, served) if e < 0]
NOISE = 2 * statistics.stdev(pre)
NPRE = PRE // BIN  # pre-kill bins per panel
assert len(pre) == NPRE * len(data) == 36, len(pre)
BAND = "#CDD3DA"  # >= 12% darker than BG
fig, axes = plt.subplots(*((len(data), 1) if SLIDE else (1, len(data))), figsize=size(H9, 2570), sharey=True)
rows = []
DRAWN9 = {(label, k, case) for label, k, _, _, _, _, case in data}
for c, (ax, (label, k, kill, edges, served, offered, case)) in enumerate(zip(axes, data)):
    xs = [(e + BIN / 2) / 60 for e in edges]
    lo_b, hi_b = offered * (1 - NOISE), offered * (1 + NOISE)
    x0 = -PRE / 60 - 0.3
    ax.fill_between([x0, 0], lo_b, hi_b, color=BAND, lw=0, zorder=0)  # band only where it was measured
    ax.axhline(offered, color=REF, lw=3 * PX, ls=(0, (4, 3)), zorder=1)
    ax.text(POST / 60 + 0.2, 36500, f"offered load \u2248 {offered / 1000:.1f}k", ha="right", va="bottom",
            fontsize=26 * PX, color=REF, zorder=5)  # below the trace, which never drops under ~43k
    ax.plot(xs, served, color=GREEN, lw=5 * PX, zorder=3)
    ax.axvline(0, color=INK, lw=3 * PX, zorder=2)
    ax.text(0.1, 64600, "kill", ha="left", va="top", fontsize=26 * PX, color=INK)
    for dx in (-0.006, 0.006):  # axis-break glyph: the value axis starts at 35k, not 0
        ax.plot([dx - 0.006, dx + 0.006], [-0.012, 0.012], transform=ax.transAxes, color=AXIS, lw=1.2,
                clip_on=False, zorder=6)
    rank = "+".join(map(str, kill["ranks"]))
    ax.set_xlim(-PRE / 60 - 0.3, POST / 60 + 0.3)
    ax.set_xticks(range(-2, 6), [f"{MINUS}{-v}" if v < 0 else f"+{v}" if v else "0" for v in range(-2, 6)])
    run = {"S2": "Run A", "S3": "Run B"}[label]
    loss = "" if label == "S2" else " (1st loss)" if k == 1 else " (2nd loss)"
    ax.set_title(f"{run}: rank {rank} killed{loss}", loc="left", pad=10)
    if c == (len(data) - 1 if SLIDE else 1):
        ax.set_xlabel("minutes from the kill")
for label, k, kill, edges, served, offered, case in data + extra:  # CSV: every loss, drawn or not
    rank = "+".join(map(str, kill["ranks"]))
    resume = kill["resumed_seen_wall_s"] - kill["kill_wall_s"]
    for e, v in zip(edges, served):
        rows.append([label, k, rank, f"{kill['kill_wall_s']:.3f}", f"{resume:.2f}", e, e + BIN, f"{v:.1f}",
                     f"{offered:.1f}", f"{offered * (1 - NOISE):.1f}", f"{offered * (1 + NOISE):.1f}", case,
                     "node" if len(kill["ranks"]) > 1 else "rank", "yes" if (label, k, case) in DRAWN9 else "no"])
axes[0].set_ylim(35000, 65000)
axes[0].set_yticks(range(35000, 65001, 5000))
axes[0].yaxis.set_major_formatter(lambda v, _: f"{v / 1000:.0f}k" if v else "0")
axes[1 if SLIDE else 0].set_ylabel("client output tok/s")
finish(fig, "Throughput around a kill",
       "Lose 1 rank; injected by kill; DEP32 on 32 GB200; online EPLB; open-loop load below capacity; 10 s bins",
       [Line2D([], [], color=GREEN, lw=5 * PX, label="served, FT on"),
        Line2D([], [], color=REF, lw=3 * PX, ls=(0, (4, 3)), label="offered load"),
        box(BAND, BAND, "pre-kill ±2 SD")], h_pad=3 if SLIDE else None)
save(fig, "fig9_kill_timeline", rows,
     ["test", "kill", "ranks", "kill_wall_s", "resume_s", "bin_start_s_from_kill", "bin_end_s_from_kill",
      "served_out_tok_s", "offered_out_tok_s", "noise_band_lo", "noise_band_hi", "case", "loss", "in_doc_figure"],
     "all 8 losses of STRESS 8299473, 8314070, 8318006 (doc figure: STRESS-8299473/s2-rank14 and "
     "STRESS-8314070/s3-rank29-rank14): bench/stress{,-chat}.sse.jsonl per-second chunk "
     "counts x tokens/chunk, 10 s bins aligned to floor(kill_wall_s); offered = requested output tokens of requests "
     "arriving in [-120 s, +300 s) / 420 s; noise band = offered x (1 +- 2 SD of the pooled pre-kill bins' "
     f"served/offered - 1), 2 SD = {NOISE * 100:.1f}%")

# ---------------------------------------------------------------- Fig A3: FT a2a cost per round trip, item by item
ABL = f"{REV}/diag/ablate-8293196/verdict.txt"
HA3 = 1064
ctrl, cut, reps = {}, {}, {}
for line in open(ABL):
    if (m := re.match(r"ABLATE_CONTROL ep=32 T=64 stats=(\d) D0=([+-][\d.]+)us", line)):
        ctrl[int(m[1])] = float(m[2])
    if (m := re.match(r"ABLATE_ITEM (\w+) stats=(\d) n=3 D=([+-][\d.]+)us D0=[+-][\d.]+us cut=([+-][\d.]+)us "
                      r"\(([+-][\d.]+) ([+-][\d.]+) ([+-][\d.]+)\)", line)):
        cut[(int(m[2]), m[1])] = float(m[4])
        reps[(int(m[2]), m[1])] = [float(m[i]) for i in (5, 6, 7)]  # per-launch cuts
assert ctrl == {1: 3.00, 0: 0.88}
# FT-off round trip (the baseline the cost is added to): mean of every launch's "off=" in summary.txt.
offs = {0: [], 1: []}
for line in open(f"{REV}/diag/ablate-8293196/summary.txt"):
    if (m := re.search(r"stats=(\d) load=64 off=([\d.]+)us", line)):
        offs[int(m[1])].append(float(m[2]))
BASE = {k: float(np.mean(v)) for k, v in offs.items()}
assert len(offs[0]) == len(offs[1]) == 27, {k: len(v) for k, v in offs.items()}
ITEMS = [("S", "mask check in stats loop"), ("F", "release-flag read"), ("L", "lane mask checks"),
         ("A", "abort-flag poll"), ("SO", "abort OR in combine barrier"), ("ALL", "all checks together")]
CASES = [(1, "with EPLB stats", SLATE[1], INK), (0, "without EPLB stats", "white", INK)]
fig, ax = plt.subplots(figsize=size(HA3, 1800))
rows = []
h = 0.34
for j, (key, label) in enumerate(ITEMS):
    for c, (st, name, face, edge) in enumerate(CASES):
        v = cut[(st, key)]
        lo, hi = min(reps[(st, key)]), max(reps[(st, key)])
        y = j + (c - 0.5) * (h + 0.04)
        ax.barh(y, v, h, color=face, edgecolor=edge, lw=3 * PX, zorder=2)
        # The min-max over the 3 launches is at most 0.04 us, narrower than the bar end: CSV and caption only.
        assert hi - lo <= 0.045, (key, st, lo, hi)
        value(ax, v + 0.04, y, f"{v:.2f}", ha="left", va="center", zorder=5,
              bbox=dict(boxstyle="square,pad=0.12", fc=BG, ec="none"))
        rows.append([key, name, f"{v:.2f}", "3 launches", f"{lo + 0.0:.2f}".replace("-0.00", "0.00"), f"{hi:.2f}", f"{ctrl[st]:.2f}",
                     f"{BASE[st]:.2f}"])
for st, name, face, edge in CASES:  # both reference lines styled alike
    ax.axvline(ctrl[st], color=REF, lw=3 * PX, ls=(0, (4, 3)), zorder=1)
    ax.text(ctrl[st] + (-0.04 if st else 0.04), -1.1,
            f"measured FT total, {'with' if st else 'without'} stats: {ctrl[st]:.2f} µs",
            ha="right" if st else "left", va="top", fontsize=26 * PX, color=INK)
ax.set_yticks(range(len(ITEMS)), [i[1] for i in ITEMS])
ax.tick_params(axis="y", pad=8)
ax.set_ylim(len(ITEMS) - 0.5, -1.15)
ax.grid(axis="y", visible=False)
ax.grid(axis="x", visible=True)
ax.set_xlim(0, 3.3)
ax.set_xticks(np.arange(0, 3.01, 0.5))
ax.xaxis.set_major_formatter(lambda v, _: f"{v:.1f}" if v else "0")
ax.set_xlabel("µs per round trip")
finish(fig, "FT check cost per all-to-all round trip",
       f"DEP32 on 32 GB200; 64 tokens/rank; median of n = 3 launches; FT-off round trip {BASE[0]:.1f} µs (no stats) / "
       f"{BASE[1]:.1f} µs (stats)",
       [box(face, edge, name) for _, name, face, edge in CASES])
save(fig, "figA3_a2a_ablation", rows, ["check_removed", "case", "us_saved", "n", "min_us", "max_us",
                                        "measured_ft_total_us", "ft_off_round_trip_us"],
     "ABLATE 8293196 verdict.txt (CLAIMS 210): cut = D0 - D per variant, median of n=3 launches, min-max over the 3; "
     "single-item cuts overlap and are not additive; dashed lines = control D0; FT-off round trip = mean of the "
     f"27 off= values per stats setting in summary.txt ({BASE[0]:.3f} / {BASE[1]:.3f} us)")

# ---------------------------------------------------------------- Fig A4: FT-off restart vs FT recovery
# The pooled FT-off restarts of the slide-1 chart (make_fig_today.py RUNS; CLAIMS 224, 228), read from its CSV.
POOLED = [(r["mode"], *re.fullmatch(r"(\S+) run (\d+)", r["source"]).groups())
          for r in csv.DictReader(open(f"{OUT}/fig0_today_restart_slide.csv")) if not r["source"].startswith("MEAN")]
runs = {m: [] for m, *_ in POOLED}
rows = []
for m, d, run in POOLED:
    for line in open(f"{RUNS}/{d}/cold_restart.txt"):
        if not (line.startswith("COLD_RESTART ") and f" run={run} " in line):
            continue
        kv = {k: v for k, v in (t.split("=", 1) for t in line.split() if "=" in t)}
        f = {k: float(kv[k]) for k in ("exit_s", "weights_s", "warmup_s", "graphs_s", "first_token_s", "down_s")}
        rl = kv["relaunch"]
        load = float(re.search(r"total_model_loading_seconds=([\d.]+)",
                               open(f"{RUNS}/{d}/{rl}/rank-0.log", errors="replace").read())[1])
        seg = {"load": load, "autotune": f["warmup_s"] - f["weights_s"], "graphs": f["graphs_s"] - f["warmup_s"],
               "other": f["exit_s"] + (f["weights_s"] - load) + (f["first_token_s"] - f["graphs_s"])}
        assert abs(sum(seg.values()) - f["down_s"]) < 0.15, (d, seg, f)
        runs[m].append(seg)
        rows.append(["restart", m, f"{d} run {kv['run']} (relaunch {rl})", *(f"{seg[k]:.1f}" for k in seg),
                     f"{f['down_s']:.1f}"])
FTREC = {}
for r in csv.DictReader(open(f"{OUT}/fig7_recovery.csv")):
    if r["panel"] == "a" and r["failure"] == "rank" and r["source"].startswith("MEAN") and r["mode"] in runs:
        FTREC[r["mode"]] = (float(r["value_s"].split()[0]), r["row"])
# Restart (FT off) components: dark outline, fills from slate to white.
HA4 = 1064
# "autotune" = the harness's weights -> warmup interval: mostly attention-kernel JIT (~51 s), the autotuner ~10 s.
RSEG = [("load", "model load", SLATE[1]), ("autotune", "compile + tune kernels", SLATE[2]),
        ("graphs", "CUDA graph capture", SLATE[3]), ("other", "other", "white")]
fig, ax = plt.subplots(figsize=size(HA4, 2056))
yt, yl = [], []
for i, (m, name) in enumerate(MODES):
    y = i * 1.0
    left_ = 0.0
    mean = {k: float(np.mean([s[k] for s in runs[m]])) for k, *_ in RSEG}
    for k, _, c in RSEG:  # white gap lines between segments, one dark outline around the bar
        ax.barh(y - 0.19, mean[k], 0.34, left=left_, color=c, edgecolor="white", lw=4 * PX)
        left_ += mean[k]
    ax.barh(y - 0.19, left_, 0.34, color="none", edgecolor=INK, lw=3 * PX, zorder=3)
    totals = [sum(r.values()) for r in runs[m]]
    if len(totals) > 1:  # each run's total as a marker on a min-max whisker
        ax.errorbar(left_, y - 0.19, xerr=[[left_ - min(totals)], [max(totals) - left_]], fmt="none", ecolor=INK,
                    elinewidth=3.5 * PX, capsize=12 * PX, capthick=3.5 * PX, zorder=4)
        ax.plot(totals, [y - 0.19] * len(totals), ls="none", marker="D", ms=24 * PX, mfc="white", mec=INK,
                mew=3 * PX, zorder=5)
    value(ax, max(totals + [left_]) + 16, y - 0.19, f"{half_up(left_ / 60):.1f} min (n = {len(totals)})", ha="left",
          va="center")
    rows.append(["restart", m, f"MEAN n={len(runs[m])}", *(f"{mean[k]:.1f}" for k, *_ in RSEG), f"{left_:.1f}"])
    ft, n = FTREC[m]
    # A ~1 s bar is < 1 px wide: a dot just right of the axis (x = 0.02 min), the label carries the value.
    ax.plot(1.2, y + 0.19, marker="o", ms=32 * PX, color=GREEN, mec=INK, mew=2 * PX, zorder=4, clip_on=False)
    ax.annotate(f"{half_up(ft, 2):.2f} s", (1.2, y + 0.19), xytext=(26 * PX, 0), textcoords="offset points", ha="left",
                va="center", fontsize=28 * PX, fontweight="bold", color=INK)
    rows.append(["FT on recovery", m, f"MEAN {n} (fig7_recovery.csv)", "", "", "", "", f"{ft:.3f}"])
    yt += [y - 0.19, y + 0.19]
    yl += [f"{name}, restart (FT off)", f"{name}, FT on recovery"]
ax.set_yticks(yt, yl)
ax.tick_params(axis="y", pad=12)
ax.set_ylim(2.5, -0.5)
ax.grid(axis="y", visible=False)
ax.grid(axis="x", visible=True)
ax.set_xlim(0, 720)
ax.set_xticks(range(0, 721, 120), [f"{v // 60} min" if v else "0" for v in range(0, 721, 120)])
ax.set_xlabel("time to serve again")
finish(fig, "Restart vs FT recovery",
       f"Lose 1 rank; injected by kill; DEP32 on 32 GB200; restart n = {min(map(len, runs.values()))}–"
       f"{max(map(len, runs.values()))} per mode, FT on n = 3–6 per mode",
       [box(c, INK, l) for _, l, c in RSEG] +
       [Line2D([], [], ls="none", marker="D", ms=24 * PX, mfc="white", mec=INK, mew=3 * PX, label="one restart run")] +
       [Line2D([], [], color=GREEN, marker="o", ms=32 * PX, mec=INK, mew=2 * PX, lw=0, label="FT on, recovery")])
save(fig, "figA4_restart_vs_ft", rows, ["what", "mode", "source", "model_load_s", "autotune_s", "graph_capture_s",
                                         "other_s", "total_s"],
     "the restarts of fig0_today_restart_slide.csv (CLAIMS 224, 228): cold_restart.txt + relaunch rank-0 "
     "total_model_loading_seconds (CLAIMS 211); bars = mean of runs; FT = fig7_recovery.csv rank means")

# ---------------------------------------------------------------- Fig A5: remap sub-phases
PHL = re.compile(r"RANK (\d+)\] Remap phases ms: release=([\d.]+) .*?barrier1=([\d.]+) make_room=([\d.]+)\(.*?\) "
                 r"create=([\d.]+) export=([\d.]+) gather=([\d.]+) import_map=([\d.]+) outcome=([\d.]+) "
                 r"remap_rest=([\d.]+) barrier2=([\d.]+) total=([\d.]+)")
RCASES = [("OFF", "rank5-1", "No EPLB, kill 1"), ("OFF", "rank5-2", "No EPLB, kill 2"),
          ("OFF-B", "rank5-1", "Offline EPLB, kill 1"), ("ON", "rank5-1", "Online EPLB, kill 1"),
          ("ON", "rank5-2", "Online EPLB, kill 2")]
HA5, MIN_WHISKER_MS = 1064, 15
PARTS = ["release", "barrier", "make_room", "cxg", "rest"]
PSEG = [("other", "other steps", "white", INK), ("import_map", "import + map", PALE_ACCENT, ACCENT),
        ("outcome", "wait for slowest", SLATE[2], INK)]
fig, ax = plt.subplots(figsize=size(HA5, 2056))
rows = []
for i, (mode, row, label) in enumerate(RCASES):
    d = f"{RUNS}/KILL-FINAL-{mode}-8308514/{row}"
    ranks = []
    for fn in sorted(os.listdir(d)):
        if fn.startswith("rank-") and fn.endswith(".log"):
            for m in PHL.finditer(open(f"{d}/{fn}", errors="replace").read()):
                v = [float(x) for x in m.groups()[1:]]
                rel, b1, mr, cr, ex, ga, im, oc, rr, b2, tot = v
                ranks.append({"rank": int(m[1]), "release": rel, "barrier": b1 + b2, "make_room": mr,
                              "cxg": cr + ex + ga, "import_map": im, "outcome": oc,
                              "rest": tot - (rel + b1 + mr + cr + ex + ga + im + oc + b2), "total": tot})
    assert len(ranks) == 31, (d, len(ranks))
    ranks.sort(key=lambda r: r["total"])
    med = ranks[len(ranks) // 2]  # survivor with the median total (16th of 31)
    med["other"] = sum(med[k] for k in PARTS)
    # Printed parts: largest-remainder rounding so they add up to the printed total.
    target = int(f"{med['total']:.0f}")
    floors = {k: math.floor(med[k]) for k, *_ in PSEG}
    order = sorted(PSEG, key=lambda t: med[t[0]] - floors[t[0]], reverse=True)
    shown = dict(floors)
    for t in order[:target - sum(floors.values())]:
        shown[t[0]] += 1
    assert sum(shown.values()) == target, (d, shown, target)
    bottom = 0.0
    for k, _, face, edge in PSEG:
        # A segment under ~8 ms would draw as a double outline: fill only, so the stack still sums to the total.
        ax.bar(i, med[k], 0.5, bottom=bottom, color=edge if med[k] < 8 else face,
               edgecolor="none" if med[k] < 8 else edge, lw=3 * PX)
        if med[k] > 50:
            ax.text(i, bottom + med[k] / 2, f"{shown[k]:,}", ha="center", va="center", fontsize=28 * PX,
                    fontweight="bold", color=INK)
        elif k == "outcome":  # small wait: a leader line from the stack cap to the label
            ax.annotate(f"+{shown[k]} ms wait", (i + 0.25, bottom + med[k] / 2), xytext=(12, -22),
                        textcoords="offset points", ha="left", va="center", fontsize=26 * PX, color=LABEL,
                        arrowprops=dict(arrowstyle="-", color=LABEL, lw=2 * PX, shrinkA=6, shrinkB=0))
        bottom += med[k]
    assert abs(bottom - med["total"]) < 0.05, (d, bottom, med["total"])
    tmin, tmax = ranks[0]["total"], ranks[-1]["total"]  # spread of the total across the 31 survivors
    if tmax - tmin >= MIN_WHISKER_MS:  # a narrower min-max does not show at 900 px: CSV and caption only
        ax.errorbar(i, med["total"], yerr=[[med["total"] - tmin], [tmax - med["total"]]], fmt="none", ecolor=INK,
                    elinewidth=3.5 * PX, capsize=12 * PX, capthick=3.5 * PX, zorder=4)
    value(ax, i, (max(tmax, bottom) if tmax - tmin >= MIN_WHISKER_MS else bottom) + 12, f"{target:,} ms")
    rows.append([f"KILL-FINAL-{mode}-8308514/{row}", med["rank"], *(f"{med[k]:.1f}" for k in PARTS),
                 *(f"{med[k]:.1f}" for k, *_ in PSEG), *(str(shown[k]) for k, *_ in PSEG),
                 f"{med['total']:.1f}", f"{ranks[0]['total']:.1f}", f"{ranks[-1]['total']:.1f}",
                 f"{max(r['import_map'] for r in ranks):.1f}"])
ax.set_xticks(range(len(RCASES)), [c[2] for c in RCASES])
ax.set_xlim(-0.5, len(RCASES) - 0.5 + 0.35)  # room for the last wait label inside the axes
ax.grid(axis="x", visible=False)
ax.set_ylim(0, 1080)
ax.yaxis.set_major_formatter(lambda v, _: f"{v:,.0f}")
ax.set_ylabel("remap time (ms)")
finish(fig, "Remap phases",
       "Lose 1 rank; injected by kill; DEP32 on 32 GB200; n = 1 kill per bar; bar: survivor with the median total",
       [box(face, edge, l) for _, l, face, edge in PSEG] +
       [Line2D([], [], color=INK, lw=3.5 * PX, marker="_", ms=24 * PX, mew=3.5 * PX,
               label="min–max over 31 survivors")])
save(fig, "figA5_remap_phases", rows, ["case", "rank", *(k + "_ms" for k in PARTS),
                                        *(k + "_ms" for k, *_ in PSEG),
                                        *(k + "_printed_ms" for k, *_ in PSEG), "total_ms",
                                        "min_total_ms", "max_total_ms", "max_import_map_ms"],
     "REMAP-TIMER 8308514 rank logs 'Remap phases ms' (CLAIMS 214): survivor with the median total of 31; "
     "barrier = barrier1 + barrier2, rest = remap_rest + rounding; other = release + barrier + make_room + "
     "create/export/gather + rest")
