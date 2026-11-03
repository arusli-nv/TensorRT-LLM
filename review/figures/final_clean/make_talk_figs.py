"""Slide versions of the evaluation charts, drawn from the CSVs the doc scripts write (one style for every slide).

Wide frames, larger text and heavier marks than the doc figures; titles and labels only, takeaways stay on the slide.
usage: /usr/bin/python3 make_talk_figs.py   (writes <name>_talk.png/.svg next to this script)
Data: fig0_today_restart_slide.csv (CLAIMS 224), fig5_healthy.csv, fig6_ft_change.csv, fig7_recovery.csv,
fig7b_phases.csv, fig8_after_loss.csv, fig9_kill_timeline.csv, fig10_realistic_latency.csv, figA1_step_cost.csv,
figA3_a2a_ablation.csv, figA5_remap_phases.csv (each written by make_figs.py / make_figs_extra.py / make_fig_today.py).
"""
import csv
import glob
import math
import os
import re
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.container import BarContainer  # noqa: E402
from matplotlib.patches import FancyBboxPatch, Patch  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402

ROOT = "/lustre/fs1/portfolios/coreai/projects/coreai_comparch_trtllm/users/arusli/src/TensorRT-LLM/.wideep-ft-runs/claude"
OUT = os.path.dirname(os.path.abspath(__file__))
for f in ("Lato-Regular.ttf", "Lato-Bold.ttf"):
    font_manager.fontManager.addfont(f"{ROOT}/mvp/analysis/fonts/{f}")
INK, LABEL, GRID, AXIS = "#2D3142", "#4F5D75", "#D5D5D5", "#BFC0C0"
MODES = [("noeplb", "No EPLB", "#1B4F12"), ("offline", "Offline EPLB", "#458F19"), ("online", "Online EPLB", "#76B900")]
MNAME = {m: n for m, n, _ in MODES}
MCOL = {m: c for m, _, c in MODES}
OFF_C, ON_C, ON_EDGE = "#8C99A6", "#76B900", "#458F19"  # FT off = slate, FT on = green, as in every figure
SLATE = ["#2F3E4E", "#6B7A89", "#A9B4BE", "#DCE1E6"]
ACCENT = ON_EDGE  # the key part, in FT's green (one palette with the design figures)
W_IN, DPI = 11.0, 200
FS_TITLE, FS_SUB, FS_PANEL, FS_AXIS, FS_TICK, FS_VAL, FS_NOTE = 22, 14.5, 17, 15, 14, 15, 13
plt.rcParams.update({"font.family": "Lato", "font.size": FS_TICK, "axes.titlesize": FS_PANEL, "axes.titleweight": "bold",
                     "axes.titlecolor": INK, "axes.titlelocation": "left", "axes.titlepad": 10,
                     "axes.labelsize": FS_AXIS, "axes.labelcolor": LABEL, "xtick.labelsize": FS_TICK,
                     "ytick.labelsize": FS_TICK, "xtick.color": LABEL, "ytick.color": LABEL, "legend.fontsize": FS_TICK,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": False,
                     "axes.edgecolor": AXIS, "axes.linewidth": 1.0, "xtick.major.size": 0, "ytick.major.size": 0,
                     "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRID, "grid.linestyle": ":",
                     "grid.linewidth": 0.8, "axes.axisbelow": True, "legend.frameon": False,
                     "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
                     "savefig.dpi": DPI, "svg.fonttype": "none", "axes.unicode_minus": True})
DOT = dict(marker="o", ms=7, mfc="white", mec=INK, mew=1.3, ls="none", zorder=5)


def rows(name):
    return list(csv.DictReader(open(f"{OUT}/{name}.csv")))


def figure(h_in, nrows=1, ncols=1, **kw):
    fig, axes = plt.subplots(nrows, ncols, figsize=(W_IN, h_in - 0.9), **kw)
    return fig, axes


def header(fig, title, sub=None, handles=None, ncol=None, left=0.08, right=0.98, bottom=0.12, wspace=None,
           hspace=None, gap=0.25):
    """An optional legend row at a fixed left margin; the axes fill the rest. The slide carries the title, so title and
    sub are not drawn (kept as the figure's description in the source)."""
    h = fig.get_figheight()
    y = 0.12
    if handles:
        fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.6 / W_IN - 0.006, 1 - y / h),
                   ncol=ncol or len(handles), handlelength=1.2, handleheight=1.0, columnspacing=1.6,
                   labelcolor=LABEL, borderaxespad=0)
        y += 0.4
    fig.subplots_adjust(top=1 - (y + gap) / h, left=left, right=right, bottom=bottom, wspace=wspace, hspace=hspace)


def round_bars(fig, r_px=7):
    """Redraw every bar with slightly rounded corners (r_px at the 2200 px export), after the layout is final."""
    fig.canvas.draw()
    for ax in fig.axes:
        to_data = ax.transData.inverted()
        (x0, y0), (x1, y1) = to_data.transform([(0, 0), (1, 1)])
        dx, dy = abs(x1 - x0) * fig.dpi / DPI, abs(y1 - y0) * fig.dpi / DPI  # data units per export pixel
        for c in ax.containers:
            if not isinstance(c, BarContainer):
                continue
            for rect in c.patches:
                x, y, w, h = rect.get_x(), rect.get_y(), rect.get_width(), rect.get_height()
                if not w or not h:
                    continue
                r = min(r_px * dx, abs(w) / 2, abs(h) / 2 * dx / dy)
                ax.add_patch(FancyBboxPatch((min(x, x + w), min(y, y + h)), abs(w), abs(h),
                                            boxstyle=f"round,pad=0,rounding_size={r}", mutation_aspect=dy / dx,
                                            fc=rect.get_facecolor(), ec=rect.get_edgecolor(),
                                            lw=rect.get_linewidth(), zorder=rect.get_zorder()))
                rect.set_visible(False)


def save(fig, name):
    round_bars(fig)
    fig.savefig(f"{OUT}/{name}_talk.png")
    fig.savefig(f"{OUT}/{name}_talk.svg")
    plt.close(fig)
    print(f"{name}_talk")


def value(ax, x, y, text, **kw):
    ax.text(x, y, text, **{"ha": "center", "va": "bottom", "fontsize": FS_VAL, "fontweight": "bold", "color": INK, **kw})


def half_up(v, digits=1):
    """Round half up, as the doc's figures do (CLAIMS 220's 4-decimal medians, e.g. 95.05 -> 95.1)."""
    f = 10 ** digits
    return math.floor(v * f + 0.5 + 1e-9) / f


def pct(v, digits=1):
    return f"{v:+.{digits}f}%".replace("-", "−")


# ---------------------------------------------------------------- restart after losing one rank (slide 1)
R0 = rows("fig0_today_restart_slide")
PH = [("teardown_s", "tear down", SLATE[2]), ("load_s", "start + load weights", SLATE[0]),
      ("warmup_s", "compile + tune kernels", SLATE[1]), ("graphs_s", "capture CUDA graphs", SLATE[3])]
restart = {m: [r for r in R0 if r["mode"] == m and not r["source"].startswith("MEAN")] for m, _, _ in MODES}
rmean = {m: next(r for r in R0 if r["mode"] == m and r["source"].startswith("MEAN")) for m, _, _ in MODES}
fig, ax = figure(5.6)
for i, (m, name, _) in enumerate(MODES):
    left = 0.0
    for k, _, c in PH:
        v = float(rmean[m][k])
        ax.barh(i, v, 0.6, left=left, color=c, edgecolor="white", lw=1.5)
        left += v
    ax.barh(i, left, 0.6, color="none", edgecolor=INK, lw=1.4, zorder=3)
    tot = [float(r["total_s"]) for r in restart[m]]
    ax.plot(tot, [i] * len(tot), marker="D", ms=8, mfc="white", mec=INK, mew=1.3, ls="none", zorder=5)
ax.set_yticks(range(3), [f"{n} (n = {len(restart[m])})" for m, n, _ in MODES])
ax.set_ylim(2.55, -0.6)
ax.grid(axis="y", visible=False)
ax.grid(axis="x", visible=True)
ax.set_xlim(0, 720)
ax.set_xticks(range(0, 721, 120), [f"{v // 60} min" if v else "0" for v in range(0, 721, 120)])
ax.set_xlabel("time from the kill to the new server's first token")
header(fig, "Restart after losing one rank", "FT off; warm restart on the same nodes; bar = mean",
       [Patch(fc=c, ec=INK, lw=1.2, label=l) for _, l, c in PH]
       + [Line2D([], [], marker="D", ms=8, mfc="white", mec=INK, mew=1.3, ls="none", label="one restart")],
       ncol=3, left=0.2, right=0.96, bottom=0.14)
save(fig, "restart")

# ---------------------------------------------------------------- restart vs FT recovery, log scale (slide 11)
R7 = rows("fig7_recovery")
ft_runs = defaultdict(list)
for r in R7:
    if r["panel"] == "a" and r["failure"] in ("rank", "node") and r["mode"] in MCOL and r["source"].startswith("KILL"):
        ft_runs[(r["failure"], r["mode"])].append(float(r["value_s"]))
fig, ax = figure(4.6)
for i, (m, name, col) in enumerate(MODES):
    ft = ft_runs[("rank", m)]
    ftm = sum(ft) / len(ft)
    rs = float(rmean[m]["total_s"])
    ax.plot([ftm, rs], [i, i], color=GRID, lw=2, zorder=1, solid_capstyle="round")
    ax.plot(ftm, i, "o", ms=17, color=ON_C, mec=ON_EDGE, mew=1.5, zorder=4)
    ax.plot(rs, i, "o", ms=17, color=OFF_C, mec=INK, mew=1.5, zorder=4)
    ax.text(ftm / 1.25, i, f"{half_up(ftm, 2):.2f} s", ha="right", va="center", fontsize=FS_VAL, fontweight="bold", color=INK)
    ax.text(rs * 1.25, i, f"{rs / 60:.1f} min", ha="left", va="center", fontsize=FS_VAL, fontweight="bold", color=INK)
    ax.text(math.sqrt(ftm * rs), i - 0.16, f"{rs / ftm:.0f}× faster", ha="center", va="bottom", fontsize=FS_VAL,
            fontweight="bold", color=INK)
ax.set_xscale("log")
ax.set_xlim(0.3, 3000)
ax.set_xticks([1, 10, 60, 600], ["1 s", "10 s", "1 min", "10 min"])
ax.minorticks_off()
ax.set_yticks(range(3), [n for _, n, _ in MODES])
ax.set_ylim(2.6, -0.7)
ax.grid(axis="y", visible=False)
ax.grid(axis="x", visible=True)
ax.set_xlabel("time to serve again after losing one rank (log scale)")
header(fig, "FT recovery vs restart",
       "FT on: kill to first decode step; restart (FT off): kill to first token; mean of runs",
       [Line2D([], [], marker="o", ms=13, color=ON_C, mec=ON_EDGE, ls="none", label="FT on, recovery"),
        Line2D([], [], marker="o", ms=13, color=OFF_C, mec=INK, ls="none", label="restart without FT")],
       left=0.14, bottom=0.16)
save(fig, "restart_vs_ft")

# ---------------------------------------------------------------- time to serve again (slide 11)
frozen = defaultdict(list)  # mode -> [(confirm s, total s)] per run
for r in R7:
    if r["panel"] == "a" and r["failure"] == "frozen" and r["mode"] in MCOL and r["source"].startswith("KILL"):
        frozen[r["mode"]].append((float(r["fence_s"]), float(r["value_s"])))
fig = plt.figure(figsize=(W_IN, 5.4 - 0.9))
gs = fig.add_gridspec(1, 3, width_ratios=[3, 3, 2.4])
a0 = fig.add_subplot(gs[0])
axes = [a0, fig.add_subplot(gs[1], sharey=a0)]
for ax, (failure, title) in zip(axes, [("rank", "Lose 1 rank"), ("node", "Lose 1 node (4 GPUs)")]):
    for i, (m, name, col) in enumerate(MODES):
        v = ft_runs[(failure, m)]
        mean = sum(v) / len(v)
        ax.bar(i, mean, 0.62, color=ON_EDGE, zorder=2)
        ax.plot([i + d for d in [(-0.16 + 0.32 * k / max(1, len(v) - 1)) for k in range(len(v))]], v, **DOT)
        ax.text(i, 0.08, f"{half_up(mean, 2):.2f} s", ha="center", va="bottom", fontsize=FS_VAL, fontweight="bold",
                color="white", zorder=6)
    ax.set_xticks(range(3), [f"{n.split()[0]}\n(n = {len(ft_runs[(failure, m)])})" for m, n, _ in MODES])
    ax.set_title(title)
    ax.set_ylim(0, 3.2)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g} s"))
axes[1].tick_params(labelleft=False)
axes[0].set_ylabel("kill to first decode step")
fig.text(0.5, 0.02, "EPLB mode", ha="center", va="bottom", fontsize=FS_AXIS, color=LABEL)
af = fig.add_subplot(gs[2])
fm = [(m, n) for m, n, _ in MODES if frozen[m]]
for i, (m, name) in enumerate(fm):
    conf = sum(c for c, _ in frozen[m]) / len(frozen[m])
    tot = sum(t for _, t in frozen[m]) / len(frozen[m])
    af.bar(i, conf, 0.62, color=SLATE[2], zorder=2)
    af.bar(i, tot - conf, 0.62, bottom=conf, color=ON_EDGE, zorder=2)
    nf = len(frozen[m])
    af.plot([i + (j - (nf - 1) / 2) * 0.09 for j in range(nf)], [t for _, t in frozen[m]], **DOT)
    af.text(i, max([tot] + [t for _, t in frozen[m]]) + 1.6, f"{half_up(tot):.1f} s", ha="center", va="bottom", fontsize=FS_VAL, fontweight="bold", color=INK)
af.set_xticks(range(len(fm)), [f"{n.split()[0]}\n(n = {len(frozen[m])})" for m, n in fm])
af.set_xlim(-0.6, len(fm) - 0.4)
af.set_ylim(0, 46)
af.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g} s"))
af.set_title("Frozen rank")
header(fig, "Time to serve again with FT", "bar = mean; dot = one run",
       [Patch(fc=ON_EDGE, label="recovery (mean of runs)"), Patch(fc=SLATE[2], label="hang confirmation"),
        Line2D([], [], **DOT, label="one run")],
       left=0.08, bottom=0.2, wspace=0.22, gap=0.55)
save(fig, "recovery")

# ---------------------------------------------------------------- where recovery time goes (slide 12)
R7b = rows("fig7b_phases")
ph = defaultdict(dict)
for r in R7b:
    ph[(r["failure"], r["mode"])][r["phase"]] = float(r["mean_s"])
GROUPS = [(("detect", "agree"), "detect + agree", SLATE[2], INK), (("map",), "map spare slots", ON_C, INK),
          (("remap",), "remap workspace", ON_EDGE, "white"), (("stage",), "stage copies (no EPLB)", "#1B4F12", "white"),
          (("install", "first"), "install + first step", SLATE[3], INK)]
fig, ax = figure(5.6)
ticks, labels, y = [], [], 0.0
for failure, head in (("rank", "Lose 1 rank"), ("node", "Lose 1 node (4 GPUs)")):
    ax.text(0, y - 0.75, head, transform=ax.get_yaxis_transform(), ha="left", va="center", fontsize=FS_AXIS,
            fontweight="bold", color=INK)
    for m, name, _ in MODES:
        left = 0.0
        for keys, _, c, tc in GROUPS:
            v = sum(ph[(failure, m)][k] for k in keys)
            ax.barh(y, v, 0.66, left=left, color=c, edgecolor="white", lw=1.5, zorder=2)
            if v >= 0.2 and keys != ("install", "first"):  # install + first step is < 0.25 s everywhere; the slide says so
                ax.text(left + v / 2, y, f"{half_up(v, 2):.2f}", ha="center", va="center", fontsize=FS_TICK, fontweight="bold",
                        color=tc, zorder=5)
            left += v
        ax.text(left + 0.04, y, f"{half_up(left, 2):.2f} s", ha="left", va="center", fontsize=FS_VAL, fontweight="bold",
                color=INK)
        ticks.append(y)
        labels.append(name)
        y += 1
    y += 1.0
ax.set_yticks(ticks, labels)
ax.set_ylim(y - 1.4, -1.3)
ax.set_xlim(0, 3.0)
ax.set_xticks([0, 0.5, 1, 1.5, 2, 2.5, 3], ["0", "0.5 s", "1 s", "1.5 s", "2 s", "2.5 s", "3 s"])
ax.grid(axis="y", visible=False)
ax.grid(axis="x", visible=True)
ax.set_xlabel("time from the kill to the first decode step (mean per phase)")
header(fig, "Where recovery time goes", None, [Patch(fc=c, label=l) for _, l, c, _ in GROUPS], ncol=3,
       left=0.15, right=0.97, bottom=0.12, gap=0.1)
save(fig, "phases")

# ---------------------------------------------------------------- recovery by situation (slide 18)
R7c = rows("fig7c_other_situations")
SIT = [("rank", "Lose 1 rank, in decode"), ("prefill", "Kill during prefill"), ("eplb_update", "Kill in an EPLB update"),
       ("second_of_two", "Second of two losses *"), ("node", "Lose 1 node (4 GPUs)")]
fig, ax = figure(5.0)
for i, (key, name) in enumerate(SIT):
    vals = [float(r["value_s"]) for r in R7c if r["situation"] == key]
    ax.plot([min(vals), max(vals)], [i, i], color=GRID, lw=10, solid_capstyle="round", zorder=1)
    for k, (m, _, col) in enumerate(MODES):
        mv = [float(r["value_s"]) for r in R7c if r["situation"] == key and r["mode"] == m]
        ax.plot(mv, [i + (k - 1) * 0.1] * len(mv), marker="o", ms=9, mfc=col, mec="white", mew=1.2, ls="none",
                zorder=3)
    ax.text(max(vals) + 0.08, i, f"{half_up(min(vals)):.1f}–{half_up(max(vals)):.1f} s  (n = {len(vals)})", ha="left",
            va="center", fontsize=FS_VAL, fontweight="bold", color=INK)
frozen = [float(r["value_s"]) for r in R7c if r["situation"] == "frozen"]
i = len(SIT) + 0.2  # off scale: an arrow to the axis end, the value beside it
ax.annotate("", (2.95, i), xytext=(0.02, i), arrowprops=dict(arrowstyle="-|>", color=SLATE[2], lw=2.4,
                                                              mutation_scale=18), zorder=1)
ax.text(3.03, i, f"{half_up(min(frozen)):.0f}–{half_up(max(frozen)):.0f} s  (n = {len(frozen)})", ha="left",
        va="center", fontsize=FS_VAL, fontweight="bold", color=INK)
ax.set_yticks(list(range(len(SIT))) + [i], [n for _, n in SIT] + ["Frozen rank (off scale)"])
ax.set_ylim(len(SIT) + 0.7, -0.6)
ax.set_xlim(0, 3.6)
ax.set_xticks([0, 0.5, 1, 1.5, 2, 2.5, 3], ["0", "0.5 s", "1 s", "1.5 s", "2 s", "2.5 s", "3 s"])
ax.grid(axis="y", visible=False)
ax.grid(axis="x", visible=True)
ax.set_xlabel("kill to first decode step (* from the second loss's detect)")
header(fig, "Recovery time by situation", None,
       [Line2D([], [], marker="o", ms=9, mfc=c, mec="white", ls="none", label=n) for _, n, c in MODES],
       left=0.24, right=0.97, bottom=0.14, gap=0.1)
save(fig, "other_situations")

# ---------------------------------------------------------------- throughput after a loss (slide 14)
R8 = rows("fig8_after_loss")
fig, axes = figure(5.3, 1, 2, sharey=True)
for ax, (failure, title) in zip(axes, [("rank", "Lose 1 rank (31 of 32 GPUs left)"),
                                       ("node", "Lose 1 node (28 of 32 GPUs left)")]):
    share = float(next(r for r in R8 if r["failure"] == failure)["share_of_gpus_left"]) * 100
    ns = []
    for i, (m, name, _) in enumerate(MODES):
        n = len([r for r in R8 if r["failure"] == failure and r["mode"] == m and r["source"].startswith("KILL")])
        med = next(float(r["ratio"]) * 100 for r in R8 if r["failure"] == failure and r["mode"] == m
                   and r["source"] == "MEDIAN")
        ax.bar(i, med, 0.62, color=ON_EDGE, zorder=2)
        ax.plot([i - 0.38, i + 0.38], [share, share], color=INK, lw=2, ls=(0, (3, 2)), zorder=3)
        value(ax, i, share + 2.5, f"{half_up(med):.1f}%")
        ns.append(n)
    ax.set_xticks(range(3), [f"{nm}\n(n = {n})" for (_, nm, _), n in zip(MODES, ns)])
    ax.set_title(title)
    ax.set_ylim(0, 112)
    ax.set_yticks(range(0, 101, 25))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0f}%"))
axes[0].set_ylabel("throughput after the loss\n(% of before)")
header(fig, "Throughput after a loss", None,
       [Patch(fc=ON_EDGE, label="median of runs"),
        Line2D([], [], color=INK, lw=2, ls=(0, (3, 2)), label="proportional to GPUs left (96.9% / 87.5%)")],
       left=0.11, bottom=0.17, wspace=0.1, gap=0.55)
save(fig, "after_loss")

# ---------------------------------------------------------------- throughput around a kill (slide 15)
R9 = rows("fig9_kill_timeline")
series = defaultdict(list)
for r in R9:
    if r["loss"] == "rank":  # every rank kill under realistic traffic (CLAIMS 236); the one node kill is in the notes
        series[(r["case"], r["kill"])].append(r)
fig, ax = figure(5.0)
offers, traces = [], []
for k in sorted(series):
    s9 = sorted(series[k], key=lambda r: float(r["bin_start_s_from_kill"]))
    offers.append(float(s9[0]["offered_out_tok_s"]) / 1000)
    x = [(float(r["bin_start_s_from_kill"]) + float(r["bin_end_s_from_kill"])) / 2 / 60 for r in s9]
    traces.append([float(r["served_out_tok_s"]) / 1000 for r in s9])
    ax.plot(x, traces[-1], color="#C9CFD6", lw=1.0, zorder=2)
assert len({len(t) for t in traces}) == 1
mean9 = [sum(v) / len(v) for v in zip(*traces)]
ax.plot(x, mean9, color=ON_EDGE, lw=3.2, zorder=4)
j = min((i for i, t in enumerate(x) if 0 < t < 0.5), key=lambda i: mean9[i])
dip = (mean9[j] / (sum(offers) / len(offers)) - 1) * 100
ax.annotate(f"{pct(dip, 0)} in the first 10 s", (x[j], mean9[j]), xytext=(x[j] + 0.35, 40.5), fontsize=FS_NOTE,
            fontweight="bold", color=INK, ha="left", va="center",
            arrowprops=dict(arrowstyle="-", color=INK, lw=1.0, shrinkA=2, shrinkB=4))
ax.axhline(sum(offers) / len(offers), color="#4A5568", lw=1.4, ls=(0, (5, 3)), zorder=1)
ax.axvline(0, color=INK, lw=1.4, zorder=2)
ax.text(0.06, 63.5, "kill", ha="left", va="top", fontsize=FS_NOTE, color=INK)
ax.set_xlim(-2, 5)
ax.set_xticks(range(-2, 6), [f"{v:+d}".replace("-", "\u2212") if v else "0" for v in range(-2, 6)])
ax.set_ylim(38, 64)
ax.set_yticks([40, 45, 50, 55, 60])
ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0f}k"))
ax.set_xlabel("minutes from the kill (10 s bins)")
ax.set_ylabel("output tok/s")
header(fig, "Throughput around a kill", None,
       [Line2D([], [], color=ON_EDGE, lw=3.2, label=f"mean of {len(traces)} rank kills"),
        Line2D([], [], color="#C9CFD6", lw=1.0, label="each loss"),
        Line2D([], [], color="#4A5568", lw=1.4, ls=(0, (5, 3)), label="offered load (mean)")], ncol=3,
       left=0.1, bottom=0.14)
save(fig, "kill_timeline")

# ---------------------------------------------------------------- healthy throughput (slide 16)
R5 = rows("fig5_healthy")
fig, axes = figure(5.6, 1, 2, sharey=True)
for ax, load in zip(axes, ("32", "128")):
    for i, (m, name, _) in enumerate(MODES):
        pairs = [r for r in R5 if r["load"] == load and r["mode"] == m and r["metric"] == "output_throughput"
                 and "@" in r["pair"]]
        mean = next(r for r in R5 if r["load"] == load and r["mode"] == m and r["metric"] == "output_throughput"
                    and r["pair"] == "MEAN")
        off, on = float(mean["ft_off"]), float(mean["ft_on"])
        for dx, v, c, e, key in ((-0.2, off, OFF_C, INK, "ft_off"), (0.2, on, ON_C, ON_EDGE, "ft_on")):
            ax.bar(i + dx, v, 0.38, color=c, zorder=2)
        ch = next(float(r["ft_off"]) for r in R5 if r["load"] == load and r["mode"] == m
                  and r["metric"] == "output_throughput" and r["pair"].startswith("CHANGE"))
        value(ax, i, max(off, on) + 120, pct(ch, 2))
    n = len([r for r in R5 if r["load"] == load and r["mode"] == "noeplb" and r["metric"] == "output_throughput"
             and "@" in r["pair"]])
    ax.set_title(f"{load} requests/GPU (n = {n} pairs)")
    ax.set_xticks(range(3), [nm for _, nm, _ in MODES])
    ax.set_ylim(0, 7000)
    ax.set_yticks(range(0, 7001, 1000))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))
axes[0].set_ylabel("output tok/s per GPU")
header(fig, "Healthy throughput, FT off vs on", "same-node pairs; bar = mean; label = mean per-pair change with FT on",
       [Patch(fc=OFF_C, label="FT off"), Patch(fc=ON_C, label="FT on")], left=0.1, bottom=0.11, wspace=0.08, gap=0.55)
save(fig, "healthy")

# ---------------------------------------------------------------- per-step cost (slide 17)
RA1 = rows("figA1_step_cost")
fig, axes = figure(5.2, 1, 2, sharey=True)
for ax, load in zip(axes, ("32", "128")):
    for i, (m, name, col) in enumerate(MODES):
        r = next(r for r in RA1 if r["load"] == load and r["mode"] == m)
        mean, lo, hi, us = (float(r[k]) for k in ("trim_mean_pct", "ci_lo", "ci_hi", "extra_us_per_step"))
        ax.plot([i, i], [lo, hi], color=INK, lw=2, zorder=3, solid_capstyle="butt")
        for yv in (lo, hi):
            ax.plot([i - 0.08, i + 0.08], [yv, yv], color=INK, lw=2, zorder=3)
        ax.plot(i, mean, "o", ms=15, color=ON_C, mec=INK, mew=1.2, zorder=4)
        ax.text(i + 0.14, mean + 0.02, pct(mean, 2), ha="left", va="bottom", fontsize=FS_NOTE, fontweight="bold",
                color=INK)
        ax.text(i + 0.14, mean - 0.02, f"{us:.0f} µs", ha="left", va="top", fontsize=FS_NOTE, color=LABEL)
    ax.axhline(0, color=AXIS, lw=1.2)
    ax.set_title(f"{load} requests/GPU (n = {r['n']} pairs)")
    ax.set_xticks(range(3), [nm for _, nm, _ in MODES])
    ax.set_xlim(-0.5, 2.7)
    ax.set_ylim(-0.1, 1.1)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: pct(v, 1) if v else "0%"))
axes[0].set_ylabel("step time vs FT off")
header(fig, "Decode step cost of FT", "trimmed mean of per-pair change; whisker = 95% CI; label = mean, µs per step",
       [Line2D([], [], marker="o", ms=12, color=ON_C, mec=INK, ls="none", label="mean of pairs"),
        Line2D([], [], color=INK, lw=2, label="95% CI")], left=0.11, bottom=0.11, wspace=0.08, gap=0.55)
save(fig, "step_cost")

# ---------------------------------------------------------------- realistic traffic latency (slide 18)
R10 = rows("fig10_realistic_latency")
m10 = {r["metric"]: r for r in R10 if r["pair"].startswith("MEAN")}
p10 = defaultdict(list)
for r in R10:
    if r["pair"].startswith("pair"):
        p10[r["metric"]].append(r)
fig, axes = figure(5.2, 1, 2)
for ax, (kind, title, unit) in zip(axes, [("ttft", "Time to first token", "ms"), ("tpot", "Time per output token", "ms")]):
    top = max(float(r[c]) for r in p10[f"{kind}_p99_ms"] for c in ("ft_off_ms", "ft_on_ms"))
    for i, q in enumerate(("p50", "p99")):
        key = f"{kind}_{q}_ms"
        off, on = float(m10[key]["ft_off_ms"]), float(m10[key]["ft_on_ms"])
        runs = p10[key]
        for dx, v, c, col in ((-0.2, off, OFF_C, "ft_off_ms"), (0.2, on, ON_C, "ft_on_ms")):
            ax.bar(i + dx, v, 0.38, color=c, zorder=2)
            ax.plot([i + dx + (-0.12 + 0.24 * k / max(1, len(runs) - 1)) for k in range(len(runs))],
                    [float(r[col]) for r in runs], **DOT)
        ch = sum(float(r["change_pct"]) for r in runs) / len(runs)  # mean of per-pair changes, as in the slide text
        top_run = max(float(r[c]) for r in runs for c in ("ft_off_ms", "ft_on_ms"))
        value(ax, i, max(off, on, top_run) + 0.06 * top, "~0%" if abs(ch) < 0.05 else pct(ch))
    ax.set_xticks(range(2), ["median", "p99"])
    ax.set_title(title)
    ax.set_ylim(0, top * 1.22)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}" if v >= 100 or v == 0 else f"{v:g}"))
    ax.set_ylabel(unit)
header(fig, "Latency under realistic traffic, FT off vs on",
       f"51 requests/s (~80% load); bar = mean of {len(p10['ttft_p50_ms'])} runs; label = mean change per pair",
       [Patch(fc=OFF_C, label="FT off"), Patch(fc=ON_C, label="FT on"), Line2D([], [], **DOT, label="one run")],
       left=0.1, bottom=0.1, wspace=0.28, gap=0.55)
save(fig, "realistic_latency")

# ---------------------------------------------------------------- remap phases (backup)
RA5 = rows("figA5_remap_phases")
CASE = {"KILL-FINAL-OFF-8308514/rank5-1": "No EPLB\nkill 1", "KILL-FINAL-OFF-8308514/rank5-2": "No EPLB\nkill 2",
        "KILL-FINAL-OFF-B-8308514/rank5-1": "Offline EPLB\nkill 1", "KILL-FINAL-ON-8308514/rank5-1": "Online EPLB\nkill 1",
        "KILL-FINAL-ON-8308514/rank5-2": "Online EPLB\nkill 2"}
A5G = [("other_printed_ms", "other steps", SLATE[3]), ("import_map_printed_ms", "import + map peers' memory", ACCENT),
       ("outcome_printed_ms", "wait for the slowest", SLATE[1])]
fig, ax = figure(5.4)
for i, r in enumerate(RA5):
    bottom = 0.0
    for k, _, c in A5G:
        v = float(r[k])
        ax.bar(i, v, 0.6, bottom=bottom, color=c, edgecolor="white", lw=1.5, zorder=2)
        if v >= 60:  # a thinner segment cannot hold the label; its value is in the CSV
            ax.text(i, bottom + v / 2, f"{v:.0f}", ha="center", va="center", fontsize=FS_NOTE, fontweight="bold",
                    color="white" if c != SLATE[3] else INK)
        bottom += v
    value(ax, i, bottom + 15, f"{float(r['total_ms']):,.0f} ms")
ax.set_xticks(range(len(RA5)), [CASE[r["case"]] for r in RA5])
ax.set_ylim(0, 1150)
ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f} ms"))
header(fig, "Where the remap's time goes", "lose 1 rank; the survivor with the median total of 31",
       [Patch(fc=c, label=l) for _, l, c in A5G], left=0.1, bottom=0.1, gap=0.3)
save(fig, "remap_phases")

# ---------------------------------------------------------------- all-to-all check cost (backup)
RA3 = rows("figA3_a2a_ablation")
ITEMS = [("S", "mask check in stats loop"), ("F", "release-flag read"), ("L", "lane mask checks"),
         ("A", "abort-flag poll"), ("SO", "abort OR in combine"), ("ALL", "all checks together")]
a3 = {(r["check_removed"], r["case"]): float(r["us_saved"]) for r in RA3}
tot = {r["case"]: float(r["measured_ft_total_us"]) for r in RA3}
fig, ax = figure(5.6)
for j, (key, label) in enumerate(ITEMS):
    for c, (case, col) in enumerate((("with EPLB stats", SLATE[1]), ("without EPLB stats", SLATE[2]))):
        y = j + (c - 0.5) * 0.36
        v = a3[(key, case)]
        ax.barh(y, v, 0.34, color=col, zorder=2)
        ax.text(v + 0.04, y, f"{half_up(v, 2):.2f}", ha="left", va="center", fontsize=FS_NOTE, fontweight="bold", color=INK)
ax.axhline(len(ITEMS) - 1.5, color=AXIS, lw=1.2)
ax.set_yticks(range(len(ITEMS)), [l for _, l in ITEMS])
ax.set_ylim(len(ITEMS) - 0.5, -0.6)
ax.grid(axis="y", visible=False)
ax.grid(axis="x", visible=True)
ax.set_xlim(0, 3.3)
ax.set_xlabel("µs saved per all-to-all round trip by removing the check")
header(fig, "FT check cost per all-to-all round trip",
       f"32 GPUs; 64 tokens/rank; FT total {tot['with EPLB stats']:.2f} µs with EPLB stats, "
       f"{tot['without EPLB stats']:.2f} µs without",
       [Patch(fc=SLATE[1], label="with EPLB stats"), Patch(fc=SLATE[2], label="without EPLB stats")],
       left=0.26, bottom=0.12)
save(fig, "a2a_ablation")

# ---------------------------------------------------------------- FT change per metric, 32/GPU (backup)
R6 = rows("fig6_ft_change")
METRICS = [("output_throughput", "output tok/s"), ("median_tpot_ms", "median TPOT"), ("p99_tpot_ms", "p99 TPOT"),
           ("median_ttft_ms", "median TTFT")]
fig, ax = figure(5.8)
y, yt = 0, []
for metric, mlabel in METRICS:
    ax.text(-0.02, y, mlabel, transform=ax.get_yaxis_transform(), ha="right", va="center", fontsize=FS_TICK,
            fontweight="bold", color=INK)
    y += 0.9
    for m, name, col in MODES:
        r = next((r for r in R6 if r["metric"] == metric and r["mode"] == m and r["load"] == "32"), None)
        if r is None:
            continue
        mean, lo, hi = (float(r[k]) for k in ("mean_pct", "ci_lo", "ci_hi"))
        ax.plot([lo, hi], [y, y], color=INK, lw=1.8, zorder=3)
        ax.plot(mean, y, "o", ms=11, color=ON_C, mec=INK, mew=1.1, zorder=4)
        yt.append((y, name))
        y += 1
    y += 0.3
ax.axvline(0, color=INK, lw=1.2)
ax.set_yticks([v for v, _ in yt], [t for _, t in yt], fontsize=FS_NOTE)
ax.set_ylim(y - 0.6, -0.5)
ax.grid(axis="y", visible=False)
ax.grid(axis="x", visible=True)
ax.set_xlim(-12, 32)
ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: pct(v, 0) if v else "0%"))
ax.set_xlabel("change with FT on")
header(fig, "Change with FT on, 32 requests/GPU", None,
       [Line2D([], [], marker="o", ms=10, color=ON_C, mec=INK, ls="none", label="mean of 8 pairs"),
        Line2D([], [], color=INK, lw=1.8, label="95% CI")], left=0.2, right=0.95, bottom=0.11, gap=0.1)
save(fig, "ft_change")

# ---------------------------------------------------------------- late-write designs compared (slide 16b)
# Left: healthy cost per decode step, no EPLB, 32 requests/GPU, one job (STD-8164194) on two node sets, FT off in the
# same run (CLAIMS 171, tools/trim_readout.py): fixed mailboxes = build v5-land, remap = build v5-zero (the final
# design's first build). Values: ms over FT off per node set, and FT off's step per set.
DESIGN_COST = {"Fixed\nmailboxes": [0.221, 0.224], "Remap": [0.112, 0.156]}
OFF_STEP = [13.446, 13.505]
# Right: online EPLB rank kill -> first decode step, mean per step. Fixed mailboxes = the Oct 20 build's kills
# (KILL-FINAL-ON-A/B-8086324, CLAIMS 121); remap = v5-final (fig7b_phases.csv, CLAIMS 241).
oct20 = []
for f in sorted(glob.glob(f"{ROOT}/mvp/runs/KILL-FINAL-ON-?-8086324/kill_final.txt")):
    for line in open(f):
        parts = line.split()
        if line.startswith("KILL_FINAL_VERDICT") and parts[1].startswith("rank5-"):
            p = {k: float(v) for k, v in (t.split("=") for t in line.split("phases1=")[1].split()[0].split(","))}
            oct20.append({"detect": p["detect"], "agree": p["stopped"] - p["detect"], "map": p["proposed"] - p["stopped"],
                          "remap": p["ready"] - p["proposed"], "stage": 0.0, "install": p["committed"] - p["ready"],
                          "first": p["first_step"] - p["committed"]})
STEPS = list(oct20[0])
final = {k: ph[("rank", "online")][k] for k in STEPS}
oct20_mean = {k: sum(r[k] for r in oct20) / len(oct20) for k in STEPS}
fig, (a1, a2) = figure(4.4, 1, 2, sharey=True, gridspec_kw={"width_ratios": [1, 1.6]})
ROWS = [("Quarantine", OFF_C, "Fixed\nmailboxes", oct20_mean), ("Remap", ON_EDGE, "Remap", final)]
for y, (label, col, key, d) in enumerate(ROWS):
    m = sum(DESIGN_COST[key]) / len(DESIGN_COST[key])
    a1.barh(y, m, 0.56, color=SLATE[1], zorder=2)  # one colour: the rows name the designs; greens are the steps
    a1.text(m + 0.01, y, f"{m:.2f} ms", ha="left", va="center", fontsize=FS_VAL, fontweight="bold", color=INK)
    left = 0.0
    for keys, _, c, tc in GROUPS:
        seg = sum(d[k] for k in keys)
        if seg >= 0.005:
            a2.barh(y, seg, 0.56, left=left, color=c, edgecolor="white", lw=1.5, zorder=2)
            left += seg
    a2.text(left + 0.03, y, f"{half_up(left, 2):.2f} s", ha="left", va="center", fontsize=FS_VAL, fontweight="bold",
            color=INK)
a1.set_yticks(range(len(ROWS)), [r[0] for r in ROWS])
a1.set_ylim(len(ROWS) - 0.4, -0.6)
a1.set_xlim(0, 0.32)
a1.set_xticks([0, 0.1, 0.2, 0.3], ["0", "0.1 ms", "0.2 ms", "0.3 ms"])
a1.set_title("Healthy cost per decode step")
a2.set_xlim(0, 1.3)
a2.set_xticks([0, 0.25, 0.5, 0.75, 1, 1.25], ["0", "", "0.5 s", "", "1 s", ""])
a2.set_title("Rank kill to serving again")
for a in (a1, a2):
    a.grid(axis="y", visible=False)
    a.grid(axis="x", visible=True)
    a.tick_params(axis="y", length=0)
a2.legend(handles=[Patch(fc=c, label=l) for keys, l, c, _ in GROUPS if keys != ("stage",)], ncol=2,
          loc="upper left", bbox_to_anchor=(-0.02, -0.2), handlelength=1.0, handleheight=0.9, columnspacing=1.2,
          labelcolor=LABEL, borderaxespad=0, fontsize=FS_NOTE)
header(fig, "Late writes: quarantine vs remap", None, left=0.13, right=0.95, bottom=0.3, wspace=0.3, gap=0.55)
save(fig, "late_write_designs")
