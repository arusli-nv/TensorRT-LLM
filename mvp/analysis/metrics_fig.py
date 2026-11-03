"""Client metrics FT off vs FT on per mode and load (TRT-LLM blog style).

usage: metrics_fig.py CLIENT.json OUTDIR FONTDIR
Rows: output tok/s per GPU, median TPOT, median TTFT, p99 TPOT. Bars: median over runs; whiskers: min-max;
label: FT on vs FT off change of the medians.
"""
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np

DATA, OUT, FONTS = sys.argv[1:4]
for f in ("Lato-Regular.ttf", "Lato-Bold.ttf"):
    fm.fontManager.addfont(f"{FONTS}/{f}")
plt.rcParams.update({"font.family": "Lato", "font.size": 11.5, "axes.titlesize": 11.5, "axes.titleweight": "bold",
    "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#9A9A9A", "axes.grid": True,
    "grid.color": "#EFEFEF", "axes.axisbelow": True, "legend.frameon": False, "savefig.dpi": 220,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.08})
d = json.load(open(DATA))
GPUS = 32
MODES = [("noeplb", "No EPLB (8 + 2 ghost slots/GPU)", "#1F77B4"), ("offline", "Offline EPLB (9 + 1)", "#FF7F0E"),
         ("online", "Online EPLB (9 + 1)", "#2CA02C")]
METRICS = [("output_throughput", "Output tok/s per GPU", 1 / GPUS, True),
           ("median_tpot_ms", "Median TPOT (ms)", 1, False),
           ("median_ttft_ms", "Median TTFT (s)", 1e-3, False)]
loads = ["8", "32", "128", "512"]
fig, axes = plt.subplots(len(METRICS), len(MODES), figsize=(13, 9.6), squeeze=False)
w = 0.38
for c, (m, title, color) in enumerate(MODES):
    for r, (key, ylabel, scale, higher_better) in enumerate(METRICS):
        ax = axes[r][c]
        for i, L in enumerate(loads):
            meds = []; top = 0
            for k, (arm, col) in enumerate(((m, "#9A9A9A"), (m + "-ft", color))):
                vals = [x[key] * scale for x in d.get(arm, {}).get(L, [])]
                if not vals:
                    meds.append(None); continue
                v = float(np.median(vals)); meds.append(v); top = max(top, max(vals))
                x = i + (k - 0.5) * w
                ax.bar(x, v, w * 0.92, color=col, label=("FT off" if k == 0 else "FT on") if i == 0 else None)
                ax.errorbar(x, v, yerr=[[v - min(vals)], [max(vals) - v]], fmt="none", ecolor="#3A3A3A",
                            elinewidth=0.9, capsize=2.5)
            if None not in meds:
                pct = (meds[1] / meds[0] - 1) * 100
                worse = pct < 0 if higher_better else pct > 0
                ax.text(i, top * 1.04, f"{pct:+.1f}%", ha="center", va="bottom", fontsize=10,
                        color="#B03A2E" if worse and abs(pct) >= 1 else "#333333")
        ax.set_xticks(range(len(loads)), loads)
        ax.set_ylim(0, ax.get_ylim()[1] * 1.12)
        ax.grid(axis="x", visible=False)
        if r == 0:
            ax.set_title(title, loc="left")
        if c == 0:
            ax.set_ylabel(ylabel)
        if r == len(METRICS) - 1:
            ax.set_xlabel("Concurrent requests per GPU")
axes[0][0].legend(loc="upper left", fontsize=9.5)
fig.suptitle("Healthy serving, FT off vs FT on (DeepSeek-R1, 32 GB200)", x=0.01, ha="left", fontsize=14, fontweight="bold")
fig.tight_layout(h_pad=1.2, w_pad=1.2)
fig.savefig(f"{OUT}/fig_healthy_metrics.png"); fig.savefig(f"{OUT}/fig_healthy_metrics.svg")
print("ok")
