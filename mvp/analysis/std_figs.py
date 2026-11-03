"""Healthy-cost figures from std_collect.py output: throughput-vs-speed per EPLB mode and a results table.

usage: std_figs.py DATA.json OUTDIR FONTDIR STAMP [mode=off_arm:on_arm ...]
STAMP: text for the red corner stamp, or "" for final data.
"""
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np

DATA, OUT, FONTS, STAMP = sys.argv[1:5]
pairs = [a.split("=") for a in sys.argv[5:]] or [["No EPLB", "noeplb:noeplb-ft"], ["Offline EPLB", "offline:offline-ft"],
                                                  ["Online EPLB", "online:online-ft"]]
SLOTS = {"No EPLB": "8 slots/GPU; FT: 8 + 2 ghost", "Offline EPLB": "9 slots/GPU; FT: 9 + 1 ghost",
         "Online EPLB": "9 slots/GPU; FT: 9 + 1 ghost"}
for f in ("Lato-Regular.ttf", "Lato-Bold.ttf"):
    fm.fontManager.addfont(f"{FONTS}/{f}")
plt.rcParams.update({"font.family": "Lato", "font.size": 11.5, "axes.titlesize": 12.5, "axes.titleweight": "bold",
    "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#9A9A9A", "axes.grid": True,
    "grid.color": "#EFEFEF", "axes.axisbelow": True, "legend.frameon": False, "savefig.dpi": 220,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.08})
COLORS = {"No EPLB": "#1F77B4", "Offline EPLB": "#FF7F0E", "Online EPLB": "#2CA02C"}
GRAY, MUTED, RED = "#7F7F7F", "#707070", "#C8372D"
d = json.load(open(DATA))
GPUS = 32


def stat(arm, key):
    loads = sorted((int(L) for L in d.get(arm, {})), key=int)
    return loads, [float(np.median([r[key] for r in d[arm][str(L)]])) for L in loads]


def save(fig, name):
    if STAMP:
        fig.text(0.995, 0.995, STAMP, color=RED, ha="right", va="top", fontsize=11, fontweight="bold")
    fig.savefig(f"{OUT}/{name}.png"); fig.savefig(f"{OUT}/{name}.svg"); plt.close(fig)


# Throughput per GPU vs per user, FT off vs FT on, one panel per mode.
fig, axes = plt.subplots(1, len(pairs), figsize=(3.85 * len(pairs), 3.9), sharey=True, squeeze=False)
ymax = xmax = 0
for ax, (mode, arms) in zip(axes[0], pairs):
    off, on = arms.split(":")
    for arm, color, label in ((off, GRAY, "FT off"), (on, COLORS.get(mode, "#1F77B4"), "FT on")):
        if arm not in d:
            continue
        loads, x = stat(arm, "user_throughput")
        _, y = stat(arm, "output_throughput")
        y = [v / GPUS for v in y]
        xs = [[r["user_throughput"] for r in d[arm][str(L)]] for L in loads]
        ys = [[r["output_throughput"] / GPUS for r in d[arm][str(L)]] for L in loads]
        xerr = [[xi - min(v) for xi, v in zip(x, xs)], [max(v) - xi for xi, v in zip(x, xs)]]
        yerr = [[yi - min(v) for yi, v in zip(y, ys)], [max(v) - yi for yi, v in zip(y, ys)]]
        ax.plot(x, y, color=color, marker="o", ms=5, lw=1.8, label=label)
        ax.errorbar(x, y, xerr=xerr, yerr=yerr, fmt="none", ecolor=color, elinewidth=1, capsize=3, alpha=0.8)
        if label == "FT off":
            for L, xi, yi in zip(loads, x, y):
                ax.annotate(str(L), (xi, yi), xytext=(6, 4), textcoords="offset points", fontsize=8.5, color=MUTED)
        ymax, xmax = max(ymax, max(y)), max(xmax, max(x))
    ax.set_title(mode, loc="left", pad=17)
    ax.text(0, 1.012, SLOTS.get(mode, ""), transform=ax.transAxes, fontsize=9.5, color="#333333", va="bottom")
    ax.set_xlabel("Output tok/s per user")
for ax in axes[0]:
    ax.set_xlim(0, xmax * 1.12); ax.set_ylim(0, ymax * 1.12)
axes[0][0].set_ylabel("Output tok/s per GPU")
from matplotlib.lines import Line2D
fig.legend([Line2D([], [], color=GRAY, marker="o"), Line2D([], [], color="#333333", marker="o", mfc="#333333")],
           ["FT off", "FT on"], loc="upper center", ncol=2, bbox_to_anchor=(0.5, -0.05), fontsize=10)
fig.tight_layout(w_pad=1.5)
save(fig, "fig_throughput")

# Results table as an image: per mode and load, tok/s/GPU off/on, throughput and TPOT change, n.
rows = []
for mode, arms in pairs:
    off, on = arms.split(":")
    if off not in d or on not in d:
        continue
    for L in sorted(set(d[off]) & set(d[on]), key=int):
        o, n = d[off][L], d[on][L]
        med = lambda rs, k: float(np.median([r[k] for r in rs]))
        rng = lambda rs, k, s=1.0: f"{min(r[k] for r in rs) * s:,.0f}-{max(r[k] for r in rs) * s:,.0f}"
        to, tn = med(o, "output_throughput") / GPUS, med(n, "output_throughput") / GPUS
        po, pn = med(o, "median_tpot_ms"), med(n, "median_tpot_ms")
        rows.append([mode, L, f"{to:,.0f}", f"{tn:,.0f}", f"{(tn / to - 1) * 100:+.1f}%", f"{po:.2f}", f"{pn:.2f}",
                     f"{(pn / po - 1) * 100:+.1f}%", f"{len(o)}/{len(n)}",
                     f"{rng(o, 'output_throughput', 1 / GPUS)} / {rng(n, 'output_throughput', 1 / GPUS)}"])
cols = ["Mode", "Req/GPU", "tok/s/GPU off", "tok/s/GPU on", "Throughput", "TPOT off (ms)", "TPOT on (ms)", "TPOT",
        "n off/on", "range tok/s/GPU off / on"]
fig, ax = plt.subplots(figsize=(12.5, 0.36 * len(rows) + 0.7))
ax.axis("off")
t = ax.table(cellText=rows, colLabels=cols, loc="center", cellLoc="center")
t.auto_set_font_size(False); t.set_fontsize(9.5); t.scale(1, 1.35)
for (r, c), cell in t.get_celld().items():
    cell.set_edgecolor("#DDDDDD")
    if r == 0:
        cell.set_text_props(fontweight="bold"); cell.set_facecolor("#F4F4F4")
save(fig, "table_healthy")
print("ok", len(rows), "rows")
