"""Recovery figures from kill_collect.py output and KILL-FINAL verdicts, TRT-LLM blog style.

usage: rec_figs.py KILLS.json AFTER_LOSS.txt OUTDIR FONTDIR
fig_recovery: (a) time to serve again per failure type: FT on per EPLB mode (median, min-max), FT off restart;
              (b) the same bars split into recovery phases (median of each phase).
fig_after_loss: decode throughput after the loss as % of before, per mode and failure, vs the share of GPUs left.
FT-off restart times: CLAIMS #117 (relaunch in the same allocation, online EPLB, 128/GPU, earlier build).
Frozen rank: CLAIMS #109 (SIGSTOP, hang detection on, earlier build, n=1).
"""
import sys
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np

DATA, AFTER, OUT, FONTS = sys.argv[1:5]
for f in ("Lato-Regular.ttf", "Lato-Bold.ttf"):
    fm.fontManager.addfont(f"{FONTS}/{f}")
plt.rcParams.update({"font.family": "Lato", "font.size": 11, "axes.titlesize": 12.5, "axes.titleweight": "bold",
    "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#9A9A9A", "axes.grid": True,
    "grid.color": "#EFEFEF", "axes.axisbelow": True, "legend.frameon": False, "savefig.dpi": 220,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.08})
MODES = [("noeplb", "No EPLB", "#1F77B4"), ("offline", "Offline EPLB", "#FF7F0E"), ("online", "Online EPLB", "#2CA02C")]
KINDS = [("rank5", "Lose 1 rank\n(process killed)"), ("node1", "Lose 1 node\n(4 processes killed)"),
         ("frozen", "Frozen rank\n(SIGSTOP)")]
COLD = {"rank5": 477, "node1": 539, "frozen": 855}
FROZEN = 4.9  # earlier build, hang detection on, n=1
GRAY, INK = "#9A9A9A", "#1A1A1A"
PHASES = ["detect", "stop", "plan", "install", "first step"]
BLUES = ["#08306B", "#2171B5", "#6BAED6", "#9ECAE1", "#DEEBF7"]
d = json.load(open(DATA))
XMAX = 6.0


def axes_layout(ax, title, xlabel):
    ax.set_xlim(0, XMAX); ax.set_ylim(len(KINDS) - 0.5, -0.5)
    ax.set_yticks(range(len(KINDS)), [k[1] for k in KINDS]); ax.grid(axis="y", visible=False)
    ax.set_xlabel(xlabel); ax.set_title(title, loc="left")


fig, (ax, bx) = plt.subplots(1, 2, figsize=(13.5, 4.8), gridspec_kw={"width_ratios": [1.45, 1], "wspace": 0.25})
w = 0.8 / (len(MODES) + 1)
slots = MODES + [("cold", "FT off (restart)", GRAY)]
for i, (kind, _) in enumerate(KINDS):
    for k, (m, name, c) in enumerate(slots):
        x = i + (k - len(MODES) / 2) * w
        if m == "cold":
            v, lo, hi, lab = COLD[kind], None, None, f"{COLD[kind] / 60:.0f} min"
            ax.bar(x, v, w * 0.88, color=GRAY, label=name if i == 0 else None)
        elif kind == "frozen":
            if k != len(MODES) - 1:
                continue
            v, lo, hi, lab = FROZEN, None, None, f"{FROZEN:.1f} s"
            x = i - w / 2
            ax.bar(x, v, w * 0.88, color="white", edgecolor="#555555", hatch="///", lw=1,
                   label="FT on, earlier build (n=1)")
        else:
            vals = [r["total"] for r in d.get(m, {}).get(kind, [])]
            if not vals:
                continue
            v, lo, hi, lab = float(np.median(vals)), min(vals), max(vals), f"{float(np.median(vals)):.2f} s"
            ax.bar(x, v, w * 0.88, color=c, label=f"FT on, {name}" if i == 0 else None)
            ax.errorbar(x, v, yerr=[[v - lo], [hi - v]], fmt="none", ecolor="#3A3A3A", elinewidth=1, capsize=2.5)
        ax.text(x, (hi or v) * 1.15, lab, ha="center", va="bottom", fontsize=9, rotation=0)
ax.set_yscale("log"); ax.set_ylim(0.1, 3000)
ax.set_yticks([0.1, 1, 10, 100, 1000], ["0.1 s", "1 s", "10 s", "100 s", "1000 s"])
ax.set_xticks(range(len(KINDS)), [k[1] for k in KINDS]); ax.grid(axis="x", visible=False)
ax.set_ylabel("Failure until serving again (log scale)")
ax.set_title("(a) Time from failure until serving again", loc="left")
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3, fontsize=9.5, handlelength=1.1)

w2 = 0.8 / len(MODES)
for i, (kind, _) in enumerate(KINDS[:2]):
    for k, (m, name, c) in enumerate(MODES):
        rs = d.get(m, {}).get(kind, [])
        if not rs:
            continue
        x = i + (k - (len(MODES) - 1) / 2) * w2
        bottom = 0
        for p, pc in zip(PHASES, BLUES):
            s_ = float(np.median([r[p] for r in rs]))
            bx.bar(x, s_, w2 * 0.85, bottom=bottom, color=pc, edgecolor="white", lw=0.6,
                   label=p if (i == 0 and k == 0) else None)
            bottom += s_
        bx.text(x, bottom + 0.03, name.replace(" EPLB", "\nEPLB"), ha="center", va="bottom", fontsize=8.5, color=c)
bx.set_ylim(0, 1.85); bx.set_ylabel("Seconds from the failure")
bx.set_xticks(range(2), [k[1] for k in KINDS[:2]]); bx.grid(axis="x", visible=False)
bx.set_title("(b) Where FT's recovery time goes", loc="left")
bx.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=5, fontsize=9.5, handlelength=1.1)
fig.savefig(f"{OUT}/fig7_recovery_bars.png"); fig.savefig(f"{OUT}/fig7_recovery_bars.svg"); plt.close(fig)

rows = [l.split() for l in open(AFTER) if l.strip()]
fig, ax = plt.subplots(figsize=(9.8, 3.8))
kinds = [("rank5", "Lose 1 rank", 31 / 32), ("node1", "Lose 1 node", 28 / 32)]
w = 0.8 / len(MODES)
for i, (kind, label, ideal) in enumerate(kinds):
    for k, (m, name, c) in enumerate(MODES):
        vals = [float(a) / float(b) * 100 for mm, case, b, a in rows if mm == m and case.startswith(kind)]
        if not vals:
            continue
        x = i + (k - (len(MODES) - 1) / 2) * w
        v = float(np.median(vals))
        ax.bar(x, v, w * 0.85, color=c, label=f"FT on, {name}" if i == 0 else None)
        ax.errorbar(x, v, yerr=[[v - min(vals)], [max(vals) - v]], fmt="none", ecolor="#3A3A3A", elinewidth=1,
                    capsize=3)
        ax.text(x, 71.5, f"{v:.1f}%", ha="center", va="bottom", fontsize=10, color="white", fontweight="bold")
    ax.plot([i - 0.45, i + 0.45], [ideal * 100] * 2, color=INK, lw=1.2, ls=(0, (4, 3)),
            label="Share of GPUs left" if i == 0 else None)
ax.set_xticks(range(len(kinds)), [k[1] for k in kinds]); ax.grid(axis="x", visible=False)
ax.set_ylim(70, 102); ax.set_ylabel("Throughput after the loss\n(% of before)")
ax.set_title("Throughput after losing a rank or a node", loc="left")
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=4, fontsize=9.5, handlelength=1.4)
fig.savefig(f"{OUT}/fig_after_loss.png"); fig.savefig(f"{OUT}/fig_after_loss.svg")
print("ok")
