"""Slide figure for the motivation slide: FT off's restart after losing one rank, per EPLB mode, by phase.

usage: /usr/bin/python3 make_fig_today.py   (writes fig0_today_restart_slide.png/.svg/.csv next to this script)
Data (CLAIMS 224): every logged FT-off warm restart on the standard recipe (std-* configs: MTP, overlap scheduler,
25 CUDA graph sizes): rank 5 SIGKILLed in a decode step at 128/GPU, the job aborts and relaunches in the same
allocation with page and autotuner caches kept; COLD_RESTART lines of cold_restart.txt. Time = kill -> the new
server's first token (down_s); the harness gap between the old job's exit and the relaunch and any scheduler wait
are excluded, so it is a lower bound.
"""
import csv
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

ROOT = "/lustre/fs1/portfolios/coreai/projects/coreai_comparch_trtllm/users/arusli/src/TensorRT-LLM/.wideep-ft-runs/claude"
OUT = os.path.dirname(os.path.abspath(__file__))
for f in ("Lato-Regular.ttf", "Lato-Bold.ttf"):
    font_manager.fontManager.addfont(f"{ROOT}/mvp/analysis/fonts/{f}")
# make_figs_extra.py's palette and SIZE_SPEC.md sizes; restart phases in the slate shades of Figure A4.
BG, AXIS, GRID = "#FFFFFF", "#BFC0C0", "#D5D5D5"
INK, LABEL = "#2D3142", "#4F5D75"
SLATE = ["#2F3E4E", "#6B7A89", "#A9B4BE", "#DCE1E6"]
DPI, W, H = 200, 2056, 1500
PX = 72 / DPI  # pt per canvas px
plt.rcParams.update({"font.family": "Lato", "font.size": 28 * PX, "axes.labelsize": 30 * PX,
                     "xtick.labelsize": 28 * PX, "ytick.labelsize": 28 * PX, "legend.fontsize": 28 * PX,
                     "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": False,
                     "axes.edgecolor": AXIS, "axes.labelcolor": LABEL, "xtick.color": LABEL, "ytick.color": LABEL,
                     "xtick.major.size": 0, "ytick.major.size": 0, "grid.color": GRID,
                     "grid.linewidth": 1.5 * PX, "grid.linestyle": (0, (1, 3)), "axes.axisbelow": True,
                     "legend.frameon": False, "savefig.dpi": DPI, "svg.fonttype": "none"})
# Restart runs per mode: (run directory, run number). 8131417 (online on dep32-final.yaml) and the restarts on older
# configs (CLAIMS 44, 117) are not the standard recipe.
# RESTART2 8353534 (CLAIMS 228) adds 3 / 3 / 4 runs on the same recipe.
RUNS = {"noeplb": [("COLD-RESTART-A-8193512", 1), ("COLD-RESTART-A-8206870", 1), ("COLD-RESTART-A-8134113", 1)]
        + [("COLD-RESTART-A-8353534", r) for r in (1, 2, 3)],
        "offline": [("COLD-RESTART-B-8194123", 1), ("COLD-RESTART-B-8206870", 1), ("COLD-RESTART-B-8206870", 2),
                    ("COLD-RESTART-B-8134113", 1)] + [("COLD-RESTART-B-8353534", r) for r in (1, 2, 3)],
        "online": [("COLD-RESTART-8194123", 1), ("COLD-RESTART-8206870", 1), ("COLD-RESTART-8206870", 2)]
        + [("COLD-RESTART-8353534", r) for r in (1, 2, 3, 4)]}
MODES = [("noeplb", "No EPLB"), ("offline", "Offline EPLB"), ("online", "Online EPLB")]
# In time order. The last CUDA graph is captured ~4-6 s before the first token (final warm-up, /health, the first
# request); that tail is drawn with the capture.
PHASES = [("teardown", "tear down", "white"), ("load", "start + load weights", SLATE[1]),
          ("warmup", "compile + tune kernels", SLATE[2]), ("graphs", "capture CUDA graphs", SLATE[3])]


def restart(run_dir, run):
    """Phases of one restart from its COLD_RESTART line: exit_s (kill -> the old job gone) is the first exit_s;
    the other milestones count from the relaunch's start."""
    for line in open(f"{ROOT}/mvp/runs/{run_dir}/cold_restart.txt"):
        if line.startswith("COLD_RESTART ") and re.search(rf"\brun={run}\b", line):
            first = {}
            for k, v in re.findall(r"(\w+_s)=([\d.]+)", line):
                first.setdefault(k, float(v))
            seg = {"teardown": first["exit_s"], "load": first["weights_s"],
                   "warmup": first["warmup_s"] - first["weights_s"],
                   "graphs": first["first_token_s"] - first["warmup_s"]}
            assert abs(sum(seg.values()) - first["down_s"]) < 0.15, (run_dir, run, seg, first)
            return seg | {"total": first["down_s"]}
    raise LookupError(f"{run_dir} run {run}")


data = {m: [restart(*r) for r in runs] for m, runs in RUNS.items()}
fig, ax = plt.subplots(figsize=(W / DPI, H / DPI))
fig.subplots_adjust(left=420 / W, right=1 - 66 / W, top=1 - 300 / H, bottom=150 / H)
rows = []
for i, (m, name) in enumerate(MODES):
    mean = {k: sum(r[k] for r in data[m]) / len(data[m]) for k, *_ in PHASES}
    left = 0.0
    for k, _, c in PHASES:  # white gaps between segments, one dark outline around the bar
        ax.barh(i, mean[k], 0.5, left=left, color=c, edgecolor="white", lw=4 * PX)
        left += mean[k]
    ax.barh(i, left, 0.5, color="none", edgecolor=INK, lw=3 * PX, zorder=3)
    totals = [r["total"] for r in data[m]]
    ax.plot(totals, [i] * len(totals), ls="none", marker="D", ms=24 * PX, mfc="white", mec=INK, mew=3 * PX, zorder=5)
    ax.text(max(totals) + 16, i, f"{left / 60:.1f} min", ha="left", va="center", fontsize=28 * PX,
            fontweight="bold", color=INK)
    for (d, run), r in zip(RUNS[m], data[m]):
        rows.append([m, f"{d} run {run}", *(f"{r[k]:.1f}" for k, *_ in PHASES), f"{r['total']:.1f}"])
    rows.append([m, f"MEAN n={len(totals)}", *(f"{mean[k]:.1f}" for k, *_ in PHASES), f"{left:.1f}"])
ax.set_yticks(range(len(MODES)), [f"{name} (n = {len(data[m])})" for m, name in MODES])
ax.tick_params(axis="y", pad=12)
ax.set_ylim(len(MODES) - 0.5, -0.5)
ax.grid(axis="x", visible=True)
ax.set_xlim(0, 720)
ax.set_xticks(range(0, 721, 120), [f"{v // 60} min" if v else "0" for v in range(0, 721, 120)])
ax.set_xlabel("time from the kill to the new server's first token")
fig.text(60 / W, 1 - 40 / H, "Restart after losing one rank", ha="left", va="top", fontsize=40 * PX,
         fontweight="bold", color=INK)
fig.text(60 / W, 1 - 100 / H, "DeepSeek-R1; DEP32 on 32 GB200; FT off; warm restart on the same nodes; bar = mean",
         ha="left", va="top", fontsize=30 * PX, color=LABEL)
fig.legend(handles=[Patch(fc=c, ec=INK, lw=3 * PX, label=l) for _, l, c in PHASES] + [
    Line2D([], [], ls="none", marker="D", ms=24 * PX, mfc="white", mec=INK, mew=3 * PX, label="one restart")],
           loc="upper left", bbox_to_anchor=(56 / W, 1 - 158 / H), ncol=len(PHASES) + 1, handlelength=1.1,
           handleheight=1.1, columnspacing=2.0, labelcolor=LABEL, borderaxespad=0)
name = "fig0_today_restart_slide"
fig.savefig(f"{OUT}/{name}.png")
fig.savefig(f"{OUT}/{name}.svg")
with open(f"{OUT}/{name}.csv", "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["mode", "source", *(f"{k}_s" for k, *_ in PHASES), "total_s"])
    w.writerows(rows)
print(name, {m: [round(r["total"]) for r in data[m]] for m in data})
