"""FT on vs off change per mode and load as heatmaps, with raw medians in each cell.

usage: delta_fig.py CLIENT.json OUTDIR FONTDIR
"""
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

DATA, OUT, FONTS = sys.argv[1:4]
for f in ("Lato-Regular.ttf", "Lato-Bold.ttf"):
    fm.fontManager.addfont(f"{FONTS}/{f}")
plt.rcParams.update({"font.family": "Lato", "font.size": 11.5, "axes.titlesize": 12.5, "axes.titleweight": "bold",
                     "savefig.dpi": 220, "savefig.bbox": "tight", "savefig.pad_inches": 0.08})
d = json.load(open(DATA))
MODES = [("noeplb", "No EPLB"), ("offline", "Offline EPLB"), ("online", "Online EPLB")]
LOADS = ["8", "32", "128", "512"]
METRICS = [("output_throughput", "Output tok/s per GPU", 1 / 32, True, "{:,.0f}"),
           ("median_tpot_ms", "Median TPOT (ms)", 1, False, "{:.2f}"),
           ("median_ttft_ms", "Median TTFT (s)", 1e-3, False, "{:.2f}")]
cmap = LinearSegmentedColormap.from_list("worse", ["#2B6CB0", "#F7F7F7", "#C0392B"])
fig, axes = plt.subplots(1, 3, figsize=(15.5, 3.9))
for ax, (key, title, sc, higher_better, fmt) in zip(axes, METRICS):
    worse = np.zeros((len(MODES), len(LOADS)))
    for r, (m, _) in enumerate(MODES):
        for c, L in enumerate(LOADS):
            o = np.median([x[key] for x in d[m][L]]) * sc
            n = np.median([x[key] for x in d[m + "-ft"][L]]) * sc
            pct = (n / o - 1) * 100
            worse[r, c] = -pct if higher_better else pct
            ax.text(c, r - 0.13, f"{pct:+.1f}%", ha="center", va="center", fontsize=12, fontweight="bold")
            ax.text(c, r + 0.2, f"{fmt.format(o)} → {fmt.format(n)}", ha="center", va="center", fontsize=9,
                    color="#333333")
    ax.imshow(np.clip(worse, -8, 8), cmap=cmap, vmin=-8, vmax=8, aspect="auto")
    ax.set_xticks(range(len(LOADS)), LOADS); ax.set_yticks(range(len(MODES)), [m[1] for m in MODES])
    ax.set_xlabel("Concurrent requests per GPU"); ax.set_title(title, loc="left")
    ax.tick_params(which="both", length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xticks(np.arange(-0.5, len(LOADS)), minor=True); ax.set_yticks(np.arange(-0.5, len(MODES)), minor=True)
    ax.grid(which="minor", color="white", lw=2.5)
    if ax is not axes[0]:
        ax.set_yticklabels([])
fig.suptitle("FT's change vs FT off (red = worse with FT; cells: FT off → FT on)", x=0.01, ha="left", fontsize=14, fontweight="bold")
fig.tight_layout(w_pad=1.0)
fig.savefig(f"{OUT}/fig_healthy_delta.png"); fig.savefig(f"{OUT}/fig_healthy_delta.svg")
print("ok")
