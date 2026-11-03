"""FT's per-step cost per mode and load from finals_readout.py --json (TRT-LLM blog style).

usage: cost_fig.py READOUT.json OUTDIR FONTDIR
Bars: mean paired cost over node groups; whiskers: one-sided 95% upper bound; dashed line: +1%.
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
plt.rcParams.update({"font.family": "Lato", "font.size": 11.5, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": "#9A9A9A", "axes.grid": True, "grid.color": "#EFEFEF", "axes.axisbelow": True,
                     "legend.frameon": False, "savefig.dpi": 220, "savefig.bbox": "tight"})
d = json.load(open(DATA))["cells"]
modes = [("noeplb", "No EPLB (8 + 2 ghost slots/GPU)", "#1F77B4"), ("offline", "Offline EPLB (9 + 1)", "#FF7F0E"),
         ("online", "Online EPLB (9 + 1)", "#2CA02C")]
loads = ["8", "32", "128"]
fig, ax = plt.subplots(figsize=(9.6, 3.8))
w = 0.26
for k, (m, name, c) in enumerate(modes):
    for i, L in enumerate(loads):
        cell = d[f"{m}/{L}"]
        x = i + (k - 1) * (w + 0.03)
        ax.bar(x, cell["mean"], w, color=c, label=name if i == 0 else None)
        ub = cell.get("ub95")
        if ub is not None:
            ax.errorbar(x, cell["mean"], yerr=[[0], [ub - cell["mean"]]], fmt="none", ecolor="#3A3A3A",
                        elinewidth=1.1, capsize=4)
        top = max(cell["mean"], ub if ub is not None else cell["mean"])
        y = max(top, 0) + 0.08
        ax.text(x, y, f"{cell['mean']:+.2f}%", ha="center", va="bottom", fontsize=9)
ax.axhline(0, color="#1A1A1A", lw=1)
ax.set_xticks(range(len(loads)), loads)
ax.set_xlabel("Concurrent requests per GPU")
ax.set_ylabel("FT cost per decode step (%)\nmean; whisker: 95% upper bound")
ax.grid(axis="x", visible=False)
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=3)
fig.savefig(f"{OUT}/fig_step_cost.png"); fig.savefig(f"{OUT}/fig_step_cost.svg")
print("ok")
