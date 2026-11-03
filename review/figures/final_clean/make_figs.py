"""Evaluation figures (Fig 5-8, A1, A2) from the final v5-final data only.

usage: /usr/bin/python3 make_figs.py   (writes png + svg + csv next to this script)
       /usr/bin/python3 make_figs.py --slide   (writes <fig>_slide.png/.svg: near-square canvases, panels stacked,
       for the right half of a slide; the doc figures and CSVs are left untouched)

Every plotted number is read from the raw/readout files below and written to a CSV per figure.
Healthy cost: pairs CSVs and readouts of CLAIMS 206 (32/GPU pooled, n=8) and CLAIMS 232 (128/GPU pooled, n=8);
absolute TPOT/TTFT per arm from each arm's bench.log, matched to its pair by output tok/s.
Recovery: KILL_FINAL_VERDICT lines (CLAIMS 193, 202, 203, 213); FT-off restart: COLD_RESTART lines (CLAIMS 198);
throughput after a loss: client-side ratios, review/diag/client_tput_after_kill.txt (CLAIMS 209).
The CSVs carry every sample, n and CI; the charts show only means/medians, as the TensorRT-LLM tech blogs do.
"""
import csv
import glob
import json
import os
import re
import sys
import warnings
from decimal import ROUND_HALF_UP, Decimal

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib import patheffects  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from matplotlib.transforms import ScaledTranslation  # noqa: E402

ROOT = "/lustre/fs1/portfolios/coreai/projects/coreai_comparch_trtllm/users/arusli/src/TensorRT-LLM/.wideep-ft-runs/claude"
RUNS, REV = f"{ROOT}/mvp/runs", f"{ROOT}/review"
OUT = os.path.dirname(os.path.abspath(__file__))
# Look of the TensorRT-LLM tech blog 30 charts: light grey page, unboxed plot area with only a bottom axis line,
# dotted grid, outlined bars (FT off: white with a dark outline; FT on / EPLB modes: pale fill, green outline),
# bold dark value labels, left-aligned bold title and a frameless legend row of outlined squares.
INK, LAB, AXIS, GRIDC, BG = "#2D3142", "#4F5D75", "#BFC0C0", "#D9D9D9", "#FFFFFF"
MUTED, REF, LIGHT, DELTA = LAB, "#8A8F99", "#BBBBBB", LAB
OFF = "#A6A6A6"
GREEN, ON_FILL = "#76B900", "#EBF1E0"
ON_EDGE = "#458F19"  # FT on / Online EPLB bar outline: lime is too faint as a thin line on BG
PX = 72 / 200  # pt per canvas px at 200 dpi; sizes below are canvas px (SIZE_SPEC.md)
FS_TITLE, FS_PANEL, FS_SUB, FS_AXIS, FS_TICK, FS_VAL, FS_NOTE = (v * PX for v in (40, 34, 30, 30, 28, 28, 26))
LW_DATA, LW_WHISK, LW_REF, LW_BAR, LW_LEAD, LW_GRID = (v * PX for v in (5, 3.5, 3, 3, 2, 1.5))
MS_MAIN, MS_RUN, CAP_W = 32 * PX, 14 * PX, 24 * PX  # marker diameters and whisker cap width (pt)
S_MAIN, S_RUN = MS_MAIN ** 2, MS_RUN ** 2  # scatter areas (pt^2)
RUN_DOT = dict(s=S_RUN, edgecolors="white", linewidths=2 * PX)  # per-run dots: 14 px with a 2 px white edge
for _w in ("Regular", "Bold"):  # Lato, as in the design figures
    font_manager.fontManager.addfont(f"{ROOT}/mvp/analysis/fonts/Lato-{_w}.ttf")
plt.rcParams.update({"font.family": "Lato", "axes.unicode_minus": True, "font.size": FS_TICK, "axes.titlesize": FS_PANEL,
                     "axes.labelsize": FS_AXIS, "xtick.labelsize": FS_TICK, "ytick.labelsize": FS_TICK,
                     "legend.fontsize": FS_TICK,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": False,
                     "axes.edgecolor": AXIS, "axes.linewidth": 1.0, "axes.labelcolor": LAB, "xtick.color": LAB,
                     "ytick.color": LAB, "text.color": INK, "axes.titlecolor": INK, "figure.facecolor": BG,
                     "axes.facecolor": BG, "savefig.facecolor": BG, "xtick.major.size": 0, "ytick.major.size": 0,
                     "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRIDC, "grid.linestyle": ":",
                     "grid.linewidth": LW_GRID, "axes.axisbelow": True, "legend.frameon": False,
                     "savefig.dpi": 200, "savefig.bbox": None, "svg.fonttype": "none"})
MODES = [("noeplb", "No EPLB", "#1B4F12"), ("offline", "Offline EPLB", "#458F19"), ("online", "Online EPLB", "#76B900")]
MFILL = {"noeplb": "#E3E8E1", "offline": "#E7EFE1", "online": ON_FILL}
MMARK = {"noeplb": "o", "offline": "s", "online": "^"}
OFF_BAR = dict(facecolor="white", edgecolor=INK, lw=LW_BAR)
ON_BAR = dict(facecolor=ON_FILL, edgecolor=ON_EDGE, lw=LW_BAR)
MNAME = {m: n for m, n, _ in MODES}
MCOL = {m: c for m, _, c in MODES}
RIM = 0.75  # dark rim on lime markers so they read in grayscale print


def rim(m, lw=0.0):
    """Marker style that rims Online EPLB's lime marker in INK: an edge-less marker gets an INK edge, a marker with a
    lime edge of width lw keeps it and gets an INK stroke RIM wide on both sides of it."""
    if m != "online":
        return {}
    if lw == 0:
        return {"edgecolor": INK, "lw": RIM}
    return {"path_effects": [patheffects.Stroke(linewidth=lw + 2 * RIM, foreground=INK), patheffects.Normal()]}
GPUS = 32
DPI = 200
SIZE = (2056 / DPI, 1064 / DPI)  # every chart 2056 px wide; default height 1064 px
SHORT_SIZE = (2056 / DPI, 838 / DPI)
LEFT_PX = 60  # title block x, the same in every chart (make_figs_extra.py uses 60 too)
REFC = "#4A5568"  # reference lines and their labels
SLIDE = "--slide" in sys.argv[1:]
SLIDE_FIGS = {"fig5_healthy", "fig7_recovery", "fig7b_phases", "fig8_after_loss", "fig10_realistic_latency",
              "figA1_step_cost"}


def canvas(doc, slide_h_px):
    """Doc figure size, or in slide mode the same 2056 px width (same text scale) at slide_h_px tall."""
    return (2056 / DPI, slide_h_px / DPI) if SLIDE else doc


def finish(fig, title, handles=None, top=None, bottom=0.0, w_pad=2.5, sub=None, h_pad=None):
    """Header geometry in px of the fixed canvas: title (40 px) top 40 px, one-line subtitle (30 px) top 100 px,
    legend row (28 px) 58 px below the subtitle, axes from 64 px below the legend row; all at the LEFT_PX margin."""
    fx, fy = 1 / (fig.get_figwidth() * DPI), 1 / (fig.get_figheight() * DPI)
    assert not sub or "\n" not in sub, "subtitles are one line"
    y_sub = 1 - 100 * fy
    y_leg = y_sub - 58 * fy
    head = top or (y_leg - 64 * fy if handles else y_sub - 70 * fy)
    fig.tight_layout(rect=(30 * fx, 24 * fy + bottom, 1 - 30 * fx, head), w_pad=w_pad, h_pad=h_pad)
    x0 = LEFT_PX * fx
    fig.text(x0, 1 - 40 * fy, title, ha="left", va="top", fontsize=FS_TITLE, fontweight="bold", color=INK)
    if sub:
        fig.text(x0, y_sub, sub, ha="left", va="top", fontsize=FS_SUB, color=LAB)
    if handles:
        fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(x0 - 0.004, y_leg), ncol=len(handles),
                   handlelength=1.1, handleheight=1.1, columnspacing=2.0, handletextpad=0.6, labelcolor=LAB, borderaxespad=0)


def box(face, edge, label, lw=LW_BAR):
    return Patch(facecolor=face, edgecolor=edge, lw=lw, label=label)


CHECKS = []  # every derived label, with the displayed inputs it must reproduce from


def check(fig, label, shown, recomputed):
    CHECKS.append((fig, label, shown, recomputed, shown == recomputed))


def pct1(a, b):
    """Signed one-decimal % change from two displayed values, rounded half up."""
    return signed(half_up((b / a - 1) * 100))


def value_label(ax, x, y, text, dy=4, **kw):
    ax.annotate(text, (x, y), xytext=(0, dy), textcoords="offset points", ha="center", va="bottom", fontsize=FS_VAL,
                fontweight="bold", color=INK, **kw)


def panel_title(ax, text):
    ax.set_title(text, loc="left", fontsize=FS_PANEL, fontweight="bold", color=INK, pad=12)


def signed(v, nd=1, unit="%"):
    """Signed value with a true minus sign; zero has no sign."""
    t = f"{v:+.{nd}f}"
    if float(t) == 0:
        t = f"{0:.{nd}f}"
    return t.replace("-", "\u2212") + unit


def half_up(v, nd=1):
    """Round half away from zero at nd decimals (binary floats otherwise round .x5 either way)."""
    q = Decimal(1).scaleb(-nd)
    return float(Decimal(repr(round(v, 6))).quantize(q, rounding=ROUND_HALF_UP))


def shared_xlabel(axes, text, gap=24):
    """One x title centred under all panels, at the normal label distance below the first panel's ticks."""
    a, b = axes[0].get_position(), axes[-1].get_position()
    axes[0].set_xlabel(text)
    h_pt = a.height * axes[0].figure.get_figheight() * 72
    axes[0].xaxis.set_label_coords(((a.x0 + b.x1) / 2 - a.x0) / a.width, -gap / h_pt)
    axes[0].xaxis.label.set_va("top")


def delta_label(ax, x, y, text):
    ax.text(x, y, text, ha="center", va="bottom", fontsize=9.5, color=DELTA,
            bbox=dict(facecolor=BG, edgecolor="none", pad=1.0))


def stub(ax, y, lo, hi, xmin=None, xmax=None, color=INK, vertical=False, zorder=2):
    """Dark CI / range whisker with CAP_W-wide caps; a clipped end gets "//" instead of a cap."""
    lo_c = max(lo, xmin) if xmin is not None else lo
    hi_c = min(hi, xmax) if xmax is not None else hi
    xy = (lambda a, b: ([y, y], [a, b])) if vertical else (lambda a, b: ([a, b], [y, y]))
    ax.plot(*xy(lo_c, hi_c), color=color, lw=LW_WHISK, zorder=zorder, solid_capstyle="butt")
    for v, c in ((lo, lo_c), (hi, hi_c)):
        if v == c:
            ax.plot(*xy(v, v), marker="_" if vertical else "|", ms=CAP_W, mew=LW_WHISK, color=color, lw=0,
                    zorder=zorder)
        else:
            ax.text(c, y, "//", ha="center", va="center", fontsize=FS_TICK, fontweight="bold", color=color, zorder=4,
                    bbox=dict(facecolor=BG, edgecolor="none", pad=0.2))


def ci_glyph(ax, label, color=INK, hollow=False):
    """Legend handle drawn as a dot with a capped whisker (an off-canvas errorbar)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return ax.errorbar([np.nan], [np.nan], yerr=[[1], [1]], fmt="o", color=color, ecolor=INK, ms=9,
                           mfc=BG if hollow else color, mew=1.5, elinewidth=LW_WHISK, capsize=CAP_W / 2,
                           capthick=LW_WHISK, label=label)


def mode_marks(prefix="", ms=10, rim_all=False):
    return [Line2D([], [], color=c, marker=MMARK[m], ms=ms, lw=0, label=f"{prefix}{n}",
                   **({"mec": INK, "mew": RIM} if rim_all or m == "online" else {})) for m, n, c in MODES]


def mode_handles(prefix=""):
    return [box(MFILL[m], c, f"{prefix}{n}") for m, n, c in MODES]


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


def tci(vals):
    """Mean and two-sided 95% t-CI (same method as finals_stats.py)."""
    t975 = {3: 3.182, 5: 2.571, 7: 2.365}
    v = np.asarray(vals, float)
    h = t975[len(v) - 1] * v.std(ddof=1) / np.sqrt(len(v))
    return v.mean(), v.mean() - h, v.mean() + h


def nogrid_x(ax):
    ax.grid(axis="x", visible=False)
    ax.tick_params(axis="x", length=0)


# ---------------------------------------------------------------- healthy data
POOLED32 = f"{REV}/diag/repl_cost_pooled_pairs.csv"   # CLAIMS 206, n=8 per mode at 32
ORIG = f"{REV}/v5f_pairs_full.csv"                    # CLAIMS 194, n=4 per mode at 32 and 128
COST_JOBS32 = "8269103, 8270779, 8271451, 8285727, 8287475, 8289412, 8289662"
COST_JOBS128 = "8269103, 8270779, 8271451, 8355466, 8355979, 8357606"
POOLED128 = f"{REV}/diag/healthy128_pooled_pairs.csv"  # CLAIMS 232, n=8 per mode at 128

bench = {}
for blog in glob.glob(f"{RUNS}/STD*-8[23][0-9]*/std-*-1/bench.log"):
    for line in open(blog):
        if line.startswith("{"):
            r = json.loads(line)
            if r.get("label") in ("healthy-32", "healthy-128") and "output_throughput" in r:
                bench[(blog, r["label"])] = r


def arm(load, client):
    hits = [(b, r) for (b, lab), r in bench.items()
            if lab == f"healthy-{load}" and abs(r["output_throughput"] / GPUS - client) < 1e-6]
    assert len(hits) == 1, (load, client, len(hits))
    return hits[0]


def pairs(path, load):
    out = []
    for p in csv.DictReader(open(path)):
        if p["load"] != str(load):
            continue
        boff, off = arm(load, float(p["client_off"]))
        bon, on = arm(load, float(p["client_on"]))
        out.append({"mode": p["mode"], "group": p["group"], "off_dir": os.path.relpath(os.path.dirname(boff), RUNS),
                    "on_dir": os.path.relpath(os.path.dirname(bon), RUNS), "off": off, "on": on,
                    "trim_off": float(p["trim_off"]), "trim_on": float(p["trim_on"]), "trim": float(p["trim"])})
    return out


H = {32: pairs(POOLED32, 32), 128: pairs(POOLED128, 128)}
H32_ORIG = pairs(ORIG, 32)


def readout(path, section=None):
    """Rows '| mode | load | metric | n | mean | [lo, hi] |' (tpot_ttft files)."""
    res, on = {}, section is None
    for line in open(path):
        if line.startswith("=="):
            on = section is None or section in line
        m = re.match(r"\| (\w+) \| (\d+) \| (\w+) \| (\d+) \| ([+-][\d.]+) \| \[([+-][\d.]+), ([+-][\d.]+)\] \|", line)
        if m and on:
            res[(m[1], int(m[2]), m[3])] = (int(m[4]), float(m[5]), float(m[6]), float(m[7]))
    return res


def trim_readout(path):
    res = {}
    for line in open(path):
        m = re.match(r"\| (\w+) \| (\d+)[^|]* \| trim mean step \| (\d+) \| ([+-][\d.]+) \| \[([+-][\d.]+), "
                     r"([+-][\d.]+)\] \|", line)
        if m and (m[1], int(m[2])) not in res:
            res[(m[1], int(m[2]))] = (int(m[3]), float(m[4]), float(m[5]), float(m[6]))
    return res


CI = {32: readout(f"{REV}/diag/repl_cost_tpot_ttft.txt", "pooled"), 128: readout(f"{REV}/diag/healthy128_tpot_ttft.txt", "pooled")}
TRIM = {32: trim_readout(f"{REV}/diag/repl_cost_pooled.txt"), 128: trim_readout(f"{REV}/diag/healthy128_pooled_readout.txt")}
METRICS = [("output_throughput", "Output tok/s per GPU", 1 / GPUS, True, "{:,.0f}"),
           ("median_tpot_ms", "Median TPOT (ms)", 1, False, "{:.2f}"),
           ("p99_tpot_ms", "p99 TPOT (ms)", 1, False, "{:.2f}"),
           ("median_ttft_ms", "Median TTFT (s)", 1e-3, False, "{:.2f}")]

# Cross-check: per-pair % recomputed from bench.log reproduces every readout mean.
for L in (32, 128):
    for m, _, _ in MODES:
        ps = [p for p in H[L] if p["mode"] == m]
        for key, *_ in METRICS:
            mean = np.mean([(p["on"][key] / p["off"][key] - 1) * 100 for p in ps])
            n, rmean, _, _ = CI[L][(m, L, key)]
            assert n == len(ps) and abs(mean - rmean) < 0.006, (L, m, key, mean, rmean)
        n, rmean, _, _ = TRIM[L][(m, L)]
        assert n == len(ps) and abs(np.mean([p["trim"] for p in ps]) - rmean) < 0.006


# ---------------------------------------------------------------- Fig 5: healthy serving throughput
fig, axes = plt.subplots(*((2, 1) if SLIDE else (1, 2)), figsize=canvas(SIZE, 2056), sharey=True)
rows = []
w = 0.29
for c, L in enumerate((32, 128)):
    ax = axes[c]
    chg = []
    for r, (key, ylabel, sc, _, fmt) in enumerate(METRICS[:2]):
        for i, (m, name, col) in enumerate(MODES):
            ps = [p for p in H[L] if p["mode"] == m]
            vo = [p["off"][key] * sc for p in ps]
            vn = [p["on"][key] * sc for p in ps]
            n, pct, lo, hi = CI[L][(m, L, key)]
            if r == 0:
                xo, xn = i - w / 2 - 0.02, i + w / 2 + 0.02
                ax.bar(xo, np.mean(vo), w, **OFF_BAR)
                ax.bar(xn, np.mean(vn), w, **ON_BAR)
                jit = np.linspace(-w * 0.3, w * 0.3, len(vo))
                for a, b, dx in zip(vo, vn, jit):
                    ax.plot([xo + dx, xn + dx], [a, b], color=REF, lw=LW_LEAD, alpha=0.8, zorder=3)
                for xb, vals in ((xo, vo), (xn, vn)):
                    ax.scatter(xb + jit, vals, color=INK, zorder=4, **RUN_DOT)
                so, sn = round(np.mean(vo)), round(np.mean(vn))
                value_label(ax, xo, max(vo + [np.mean(vo)]), f"{so:,}")
                value_label(ax, xn, max(vn + [np.mean(vn)]), f"{sn:,}")
                chg.append(pct1(so, sn))
                check("fig5", f"{L}/GPU {m}", chg[-1], signed(half_up((sn / so - 1) * 100)))
            for p in ps:
                rows.append([L, m, key, p["group"], p["off_dir"], p["on_dir"], f"{p['off'][key] * sc:.4f}",
                             f"{p['on'][key] * sc:.4f}"])
            rows.append([L, m, key, "MEAN", "", "", f"{np.mean(vo):.4f}", f"{np.mean(vn):.4f}"])
            rows.append([L, m, key, f"CHANGE % n={n} [95% CI]", "", "", f"{pct:+.2f}", f"[{lo:+.2f}, {hi:+.2f}]"])
    ax.set_xticks(range(3), [f"{n}\nchange {d}" for (_, n, _), d in zip(MODES, chg)])
    nogrid_x(ax)
    panel_title(ax, f"{L} requests/GPU (n = {len([p for p in H[L] if p['mode'] == 'noeplb'])} pairs)")
    ax.tick_params(labelleft=True)
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:,.0f}")
for ax in axes if SLIDE else axes[:1]:
    ax.set_ylabel("client output tok/s per GPU")
axes[0].set_ylim(0, 7000)
finish(fig, "Healthy throughput",
       [box("white", INK, "FT off", 1.0), box(ON_FILL, ON_EDGE, "FT on"),
        Line2D([], [], color=INK, marker="o", ms=MS_RUN, mec="white", mew=2 * PX, lw=LW_LEAD, mfc=INK,
               label="each run (line joins a pair)")],
       sub="DeepSeek-R1 NVFP4; DEP32 on 32 GB200; 1k/1k random, fixed concurrency; bar = mean of "
           "same-node pairs", h_pad=4 if SLIDE else None)
save(fig, "fig5_healthy", rows, ["load", "mode", "metric", "pair", "off_arm", "on_arm", "ft_off", "ft_on"],
     f"STD jobs 32/GPU {COST_JOBS32} (n=8 pairs/mode), 128/GPU {COST_JOBS128} (n=8 pairs/mode); "
     "means from bench.log, % and CI from repl_cost_tpot_ttft.txt / healthy128_tpot_ttft.txt (pooled)")

# ---------------------------------------------------------------- Fig 6: FT change per metric (32/GPU)
fig, ax = plt.subplots(figsize=SIZE)
rows = []
LOADS = (32, 128)
SHORT = {"output_throughput": "Client output tok/s", "median_tpot_ms": "Median TPOT", "p99_tpot_ms": "p99 TPOT",
         "median_ttft_ms": "Median TTFT"}
XMIN, XMAX = -15, 20
yt = []
HW = []
for j, (key, title, sc, higher_better, fmt) in enumerate(METRICS):
    head = j * 4.5 - 1.1
    if j % 2 == 0:
        ax.axhspan(head - 0.6, j * 4.5 + 2.6, color="#ECECEC", zorder=0, lw=0)
    better = "higher" if higher_better else "lower"
    ax.text(XMIN + 0.3, head, f"{SHORT[key]} ({better} is better)", ha="left", va="center", fontsize=FS_PANEL,
            fontweight="bold", color=INK)
    for r, (m, name, col) in enumerate(MODES):
        for c, L in enumerate(LOADS):
            n, pct, lo, hi = CI[L][(m, L, key)]
            ps = [p for p in H[L] if p["mode"] == m]
            o = np.mean([p["off"][key] * sc for p in ps])
            nn = np.mean([p["on"][key] * sc for p in ps])
            noise = lo <= 0 <= hi
            if L == 32:
                y = j * 4.5 + r
                yt.append((y, name))
                stub(ax, y, lo, hi, XMIN, XMAX)
                ax.scatter(pct, y, s=S_MAIN * (1.2 if MMARK[m] == "^" else 1), marker=MMARK[m], zorder=5,
                           facecolor=BG if noise else col, edgecolor=col, lw=1.8, **rim(m, 1.8))
                HW.append((key, hi - pct))
                if hi > XMAX:
                    ax.text(XMAX - 1.6, y - 0.3, f"CI to {signed(hi)}", ha="right", va="bottom", fontsize=FS_NOTE,
                            color=INK)
            rows.append([key, m, L, n, f"{pct:+.2f}", f"{lo:+.2f}", f"{hi:+.2f}", "yes" if noise else "no",
                         f"{o:.4f}", f"{nn:.4f}"])
ax.axvline(0, color=LAB, lw=LW_REF, zorder=1)
ax.set_yticks([y for y, _ in yt], [n for _, n in yt])
ax.set_ylim(len(METRICS) * 4.5 - 1.9, -1.7)
ax.grid(axis="y", visible=False)
ax.grid(axis="x", visible=True)
ax.tick_params(length=0)
ax.set_xlim(XMIN, XMAX)
ax.set_xticks(range(-15, 21, 5))
ax.xaxis.set_major_formatter(lambda v, _: signed(v, 1))
ax.set_xlabel("change vs FT off (%)")
hw_tok = [h for k, h in HW if k == "output_throughput"]
hw_all = [h for _, h in HW]
finish(fig, "Change with FT on",
       mode_marks() +
       [ci_glyph(ax, "mean and 95% CI", color=INK, hollow=True),
        Line2D([], [], color=LAB, marker="o", ms=10, lw=0, mfc=BG, mew=1.8,
               label="open marker = 95% CI includes 0")],
       sub="32 requests/GPU; n = 8 same-node pairs per mode; mean of per-pair % change, 95% t-CI")
save(fig, "fig6_ft_change", rows, ["metric", "mode", "load", "n", "mean_pct", "ci_lo", "ci_hi", "ci_spans_0",
                                    "ft_off_mean", "ft_on_mean"],
     f"STD jobs 32/GPU {COST_JOBS32} (n=8), 128/GPU {COST_JOBS128} (n=8); % and CI from "
     "repl_cost_tpot_ttft.txt and healthy128_tpot_ttft.txt (pooled); absolute means from bench.log")

# ---------------------------------------------------------------- recovery data
KILL_FILES = {"noeplb": ["KILL-FINAL-OFF-8269102", "KILL-FINAL-OFF-8285728"],
              "online": ["KILL-FINAL-ON-8269102", "KILL-FINAL-ON-8285728"],
              "offline": ["KILL-FINAL-OFF-B-8283311", "KILL-FINAL-OFF-B-8302191", "KILL-FINAL-OFF-B-8363511"]}
STOP_FILES = {"noeplb": ["KILL-FINAL-OFF-A-8269102", "KILL-FINAL-OFF-A-8302191", "KILL-FINAL-OFF-8363511"],
              "online": ["KILL-FINAL-ON-A-8269102", "KILL-FINAL-ON-A-8302191", "KILL-FINAL-ON-B-8363511"],
              "offline": ["KILL-FINAL-OFF-B-8353506", "KILL-FINAL-OFF-A-8363511"]}  # RECOV4 labels are directory names only
UPD_FILES = ["KILL-FINAL-ON-A-8271210", "KILL-FINAL-ON-A-8353506", "KILL-FINAL-ON-A-8361282"]


def verdicts(d):
    out = []
    for line in open(f"{RUNS}/{d}/kill_final.txt"):
        if line.startswith("KILL_FINAL_VERDICT"):
            f = line.split()
            kv = dict(t.split("=", 1) for t in f[3:] if "=" in t)
            ph = dict(t.split("=") for t in kv["phases1"].split(","))
            out.append({"src": d, "row": f[1], "verdict": f[2], "ratio": float(kv["h23_tput_ratio"]),
                        "fence": kv.get("h23_confirm_s"), "ph": {k: float(v) for k, v in ph.items()}})
    return out


KILLS = {m: {"rank": [], "node": []} for m, _, _ in MODES}
for m, ds in KILL_FILES.items():
    for d in ds:
        for v in verdicts(d):
            kind = "rank" if v["row"].startswith("rank5-") else "node" if v["row"].startswith("node1-") else None
            if kind:
                KILLS[m][kind].append(v)
KILLS["online"]["rank"] += [v for d in UPD_FILES for v in verdicts(d) if v["row"].startswith("rank5upd")]
FROZEN = {m: [v for d in ds for v in verdicts(d) if v["row"].startswith("stop-")] for m, ds in STOP_FILES.items()}
assert all(v["verdict"] == "PASS" for m in KILLS for k in KILLS[m] for v in KILLS[m][k])

# Every v5-final recovery row by situation (slide 18's chart): kill -> first decode step; the second of two losses from
# its own detect (phases2); a frozen rank from the freeze. Prefill / EPLB-update / two-loss rows: CLAIMS 193, 213, 227, 235.
OTHER_FILES = {"noeplb": ["KILL-FINAL-OFF-A-8269102", "KILL-FINAL-OFF-A-8302191", "KILL-FINAL-OFF-8353506",
                          "KILL-FINAL-OFF-8361282"],
               "offline": ["KILL-FINAL-OFF-B-8353506", "KILL-FINAL-OFF-B-8361282"],
               "online": ["KILL-FINAL-ON-A-8269102", "KILL-FINAL-ON-A-8302191", "KILL-FINAL-ON-8353506"] + UPD_FILES}
if not SLIDE:
    other = [[kind, m, v["src"], v["row"], f"{v['ph']['first_step']:.2f}"]
             for kind in ("rank", "node") for m in KILLS for v in KILLS[m][kind] if not v["row"].startswith("rank5upd")]
    other += [["frozen", m, v["src"], v["row"], f"{v['ph']['first_step']:.2f}"] for m in FROZEN for v in FROZEN[m]]
    for m, ds in OTHER_FILES.items():
        for d in ds:
            for line in open(f"{RUNS}/{d}/kill_final.txt"):
                if not line.startswith("KILL_FINAL_VERDICT"):
                    continue
                f = line.split()
                kv = dict(t.split("=", 1) for t in f[3:] if "=" in t)
                assert f[2] == "PASS", (d, f[1])
                first = {k: float(v) for k, v in (t.split("=") for t in kv["phases1"].split(","))}["first_step"]
                if f[1].startswith("rank5pf"):
                    other.append(["prefill", m, d, f[1], f"{first:.2f}"])
                elif f[1].startswith("rank5upd"):
                    other.append(["eplb_update", m, d, f[1], f"{first:.2f}"])
                elif f[1].startswith("twokill"):
                    second = {k: float(v) for k, v in (t.split("=") for t in kv["phases2"].split(","))}["first_step"]
                    other += [["first_of_two", m, d, f[1], f"{first:.2f}"], ["second_of_two", m, d, f[1], f"{second:.2f}"]]
    with open(f"{OUT}/fig7c_other_situations.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["situation", "mode", "source", "row", "value_s"])
        w.writerows(other)
    print(f"fig7c_other_situations: {len(other)} v5-final rows from kill_final.txt (CLAIMS 193, 202, 203, 213, 227)")

# The pooled FT-off warm restarts (CLAIMS 224, 228), as make_fig_today.py lists them (it runs first in regen_figs.sh).
RESTART = [(r["mode"], r["source"], float(r["total_s"])) for r in csv.DictReader(open(f"{OUT}/fig0_today_restart_slide.csv"))
           if not r["source"].startswith("MEAN")]



def dur(s):
    return f"{s / 60:.0f} min" if s >= 60 else f"{half_up(s, 2):.2f} s" if s < 10 else f"{half_up(s):.1f} s"


# ---------------------------------------------------------------- Fig 7: recovery time
GROUPS = [("rank", "Lose 1 rank"), ("node", "Lose 1 node"), ("frozen", "Frozen rank")]
FENCE_C = ("#C9D0D8", LAB)  # slate phase colour, as fig7b's "Detect + plan"
N_FROZEN = sum(1 for m, _, _ in MODES if FROZEN.get(m))
fig, (ax, fx_) = plt.subplots(2, 1, figsize=canvas((2056 / DPI, (1100 + 115 * (N_FROZEN - 2)) / DPI), 2056),
                              gridspec_kw={"height_ratios": [8.4, 1.15 * N_FROZEN]})
rows = []
BH = 0.62


def end_label(a, x, y, text):
    a.annotate(text, (x, y), xytext=(12, 0), textcoords="offset points", ha="left", va="center", fontsize=FS_VAL,
               fontweight="bold", color=INK, zorder=5)


yt, yf = [], []
for kind, glabel in GROUPS:
    a = fx_ if kind == "frozen" else ax
    base = 0 if kind in ("rank", "frozen") else 4.4
    if kind != "frozen":
        a.text(0, base - 1.05, glabel, transform=a.get_yaxis_transform(), ha="left", va="center", fontsize=FS_PANEL,
               fontweight="bold", color=INK)
    k = 0
    for m, name, col in MODES:
        rs = FROZEN.get(m, []) if kind == "frozen" else KILLS[m][kind]
        if not rs:
            continue
        y = base + k
        k += 1
        (yf if kind == "frozen" else yt).append((y, f"{name} (n = {len(rs)})"))
        vals = [v["ph"]["first_step"] for v in rs]
        mean = float(np.mean(vals))
        if kind == "frozen":  # split the total at the measured fence (h23_confirm_s, from the freeze)
            fences = [float(v["fence"]) for v in rs]
            fmean = float(np.mean(fences))
            fence = f"{fmean:.1f}"
            ex = f"{np.mean([v['ph']['first_step'] - fv for v, fv in zip(rs, fences)]):.1f}"
            check("fig7", f"frozen {m} recovery after fence", ex, f"{mean - fmean:.1f}")
            check("fig7", f"frozen {m} fence + recovery = total", dur(mean), dur(float(fence) + float(ex)))
            a.barh(y, fmean, BH, facecolor=FENCE_C[0], edgecolor=col, lw=LW_BAR, zorder=2)
            a.barh(y, mean - fmean, BH, left=fmean, facecolor=ON_FILL, edgecolor=col, lw=LW_BAR, zorder=2)
            a.text(fmean / 2, y, f"fence (hang confirmation) {fence} s", ha="center", va="center", fontsize=FS_NOTE,
                   color=INK, zorder=4)
            a.text(fmean + (mean - fmean) / 2, y, f"+{ex} s", ha="center", va="center", fontsize=FS_NOTE, color=INK,
                   zorder=4)
        else:
            a.barh(y, mean, BH, facecolor=ON_FILL, edgecolor=col, lw=LW_BAR, zorder=2)
        check("fig7", f"{kind} {m} mean", dur(mean), dur(sum(vals) / len(vals)))
        stub(a, y, min(vals), max(vals), zorder=4)
        end_label(a, max(vals + [mean]), y, dur(mean))
        for v in rs:
            rows.append(["a", kind, m, v["src"], v["row"], f"{v['ph']['first_step']:.2f}"] +
                        (["", v["fence"]] if kind == "frozen" else []))
        rows.append(["a", kind, m, "MEAN (min-max)", f"n={len(vals)}", f"{mean:.3f} ({min(vals):.2f}-{max(vals):.2f})"] +
                    (["", f"{np.mean(fences):.3f}"] if kind == "frozen" else []))
    if kind == "rank":  # FT-off restart stays in the CSV only; the chart is FT on (restart vs FT is Figure A4)
        vals = [s for _, _, s in RESTART]
        mean = float(np.mean(vals))
        for m, d, s in RESTART:
            rows.append(["a", "rank", "FT off restart " + m, d, "rank5", f"{s:.1f}"])
        rows.append(["a", "rank", "FT off restart", "MEAN (min-max)", f"n={len(vals)}",
                     f"{mean:.1f} ({min(vals):.1f}-{max(vals):.1f})"])
for a, ticks, lim, xmax, step in ((ax, yt, (6.75, -1.6), 3.0, 0.5), (fx_, yf, (N_FROZEN - 0.45, -0.55), 40, 5)):
    a.set_yticks([y for y, _ in ticks], [n for _, n in ticks])
    a.set_ylim(*lim)
    a.set_xlim(0, xmax)
    xs = np.arange(0, xmax + 1e-9, step)
    a.set_xticks(xs, [f"{v:g} s" for v in xs])
    a.grid(axis="y", visible=False)
    a.grid(axis="x", visible=True)
    a.tick_params(length=0)
fx_.set_title("Frozen rank", loc="left", fontsize=FS_PANEL, fontweight="bold", color=INK, pad=14)
finish(fig, "Time to serve again",
       [box(ON_FILL, c, n) for _, n, c in MODES] + [box(*FENCE_C, "fence (hang confirmation)"), box(ON_FILL, LAB, "recovery")],
       sub="FT on; from the failure to the first decode step after it; bar = mean, whisker = min–max of runs")
fig.subplots_adjust(hspace=0.3 if SLIDE else 0.42)

PH = [("detect", "Detect"), ("agree", "Agree"), ("map", "Map spare slots"), ("remap", "Remap workspace"),
      ("stage", "Stage copies"), ("install", "Load experts, commit"), ("first", "First step")]
MERGED = [("Detect + agree", ("detect", "agree"), ("#C9D0D8", LAB)),
          ("Map spare slots", ("map",), ("#E3F1CC", ON_EDGE)),
          ("Remap workspace", ("remap",), ("#FBEBDB", "#D9822B")),
          ("Stage copies (no EPLB)", ("stage",), ("#F3D9C2", "#A85F17")),
          ("Install + first step", ("install", "first"), ("white", INK))]
# Slowest survivor's staging per case (review/diag/recovery_substeps.txt, tools/recovery_substeps.py): the part of
# proposed -> ready after the workspace remap; 0 with EPLB.
SUBSTEPS = {}
for line in open(f"{REV}/diag/recovery_substeps.txt"):
    f_ = line.split()
    SUBSTEPS[f_[1]] = {k: float(v) / 1000 for k, v in (t.split("=") for t in f_[2:])}


def phases(v):
    p = v["ph"]
    ready = p["ready"] - p["proposed"]
    stage = min(SUBSTEPS[f"{v['src']}/{v['row']}"]["staged_ms"], ready)
    return {"detect": p["detect"], "agree": p["stopped"] - p["detect"], "map": p["proposed"] - p["stopped"],
            "remap": ready - stage, "stage": stage, "install": p["committed"] - p["ready"],
            "first": p["first_step"] - p["committed"]}


fig_b, bx = plt.subplots(figsize=canvas(SIZE, 2056))
INSIDE_MIN = 0.2  # s; a thinner segment cannot hold a 28 px label and gets a callout
rows_b = []
w2 = 0.18
ticks = []
for i, (kind, glabel) in enumerate(GROUPS[:2]):
    for k, (m, name, col) in enumerate(MODES):
        rs = KILLS[m][kind]
        x = i + (k - 1) * (w2 + 0.12)
        ticks.append((x, f"{name}\nn = {len(rs)}"))
        avg = {}
        for pk, plabel in PH:
            s = float(np.median([phases(v)[pk] for v in rs]))
            avg[pk] = float(np.mean([phases(v)[pk] for v in rs]))
            rows.append(["b", kind, m, "median", pk, f"{s:.3f}", f"n={len(rs)}"])
            rows_b.append([kind, m, pk, f"{s:.3f}", f"{avg[pk]:.4f}", len(rs)])
        total = float(np.mean([v["ph"]["first_step"] for v in rs]))
        assert abs(sum(avg.values()) - total) < 1e-6, (kind, m, sum(avg.values()), total)
        bottom = 0.0
        for _, keys, pc in MERGED:
            seg = sum(avg[k] for k in keys)
            if seg < 0.005:  # stage copies with EPLB: no segment, no label
                continue
            bx.bar(x, seg, w2, bottom=bottom, facecolor=pc[0], edgecolor=pc[1], lw=LW_BAR)
            if seg >= INSIDE_MIN:
                bx.text(x, bottom + seg / 2, f"{half_up(seg, 2):.2f}", ha="center", va="center", fontsize=FS_VAL,
                        fontweight="bold", color=INK)
            else:
                bx.annotate(f"{half_up(seg, 2):.2f}", (x + w2 / 2, bottom + seg / 2), xytext=(16, -2),
                            textcoords="offset points", ha="left", va="center", fontsize=FS_NOTE, color=LAB,
                            arrowprops=dict(arrowstyle="-", color=LAB, lw=LW_LEAD, shrinkA=0, shrinkB=0))
            bottom += seg
        tots = [v["ph"]["first_step"] for v in rs]
        stub(bx, x + w2 / 4, min(tots), max(tots), vertical=True, zorder=4)  # clear of the centred segment labels
        value_label(bx, x, max(tots), dur(total), dy=8)
    bx.text(i, 1.02, glabel, transform=bx.get_xaxis_transform(), ha="center", va="bottom", fontsize=FS_PANEL,
            fontweight="bold", color=INK)
bx.set_xticks([t[0] for t in ticks], [t[1] for t in ticks])
nogrid_x(bx)
bx.set_ylim(0, 3.2)
bx.set_ylabel("seconds")
finish(fig_b, "Recovery phases", [box(c[0], c[1], l) for l, _, c in MERGED],
       sub="FT on; mean per phase over the kills in each bar; total = time to serve again; "
           "whisker = min\u2013max of runs", bottom=0.03)

save(fig, "fig7_recovery", rows, ["panel", "failure", "mode", "source", "row", "value_s", "n", "fence_s"],
     "kill->rank-0 first_step from KILL_FINAL_VERDICT phases1: no EPLB 8269102+8285728 (rank n=4, node n=4), "
     "online 8269102+8285728+8271210/8353506/8361282 rank5upd (rank n=8, node n=4), offline 8283311+8302191+8363511 (n=5 each); frozen: stop "
     "rows of 8269102+8302191+8353506+8363511 (n=3 per mode; fence_s = h23_confirm_s, freeze->fence; label "
     "splits the mean total at mean fence_s, +mean(first_step - fence_s)); restart: pooled warm restarts of "
     "fig0_today_restart_slide.csv (CLAIMS 224, 228)")
save(fig_b, "fig7b_phases", rows_b, ["failure", "mode", "phase", "median_s", "mean_s", "n"],
     "median and mean per phase from KILL_FINAL_VERDICT phases1 (same runs as fig7_recovery panel a): detect = kill -> "
     "detect, agree = detect -> stopped, map = stopped -> proposed, remap + stage = proposed -> ready split by the slowest "
     "survivor's staging (review/diag/recovery_substeps.txt), install = ready -> committed, first = committed -> "
     "first_step; bars: means")

# ---------------------------------------------------------------- Fig 8: throughput after a loss (client side)
CLIENT_TXT = f"{REV}/diag/client_tput_after_kill.txt"  # CLAIMS 209, 239
CMODE = {"ON": "online", "OFF": "noeplb", "OFF-B": "offline"}
CLIENT = {m: {"rank": [], "node": []} for m, _, _ in MODES}
for line in open(CLIENT_TXT):
    mm = re.match(r"KILL-FINAL-(ON|OFF|OFF-B)-(\d+)/(rank5|node1)-(\d)\s+\S+\s+(\d+)/32\s+(\d+)\s+(\d+)\s+(\S+)\s+([\d.]+)\s+([\d.]+)",
                  line)
    if mm:
        kind = "rank" if mm[3] == "rank5" else "node"
        CLIENT[CMODE[mm[1]]][kind].append({"src": f"KILL-FINAL-{mm[1]}-{mm[2]}", "row": f"{mm[3]}-{mm[4]}",
                                           "before": int(mm[6]), "after": int(mm[7]), "win": mm[8],
                                           "ratio": float(mm[9]), "ideal": float(mm[10])})
fig, axes = plt.subplots(*((2, 1) if SLIDE else (1, 2)), figsize=canvas(SHORT_SIZE, 2056), sharey=True)
rows = []
NROW = {}  # runs per mode; the shared row label carries n, so it must be the same for both failures
KINDS = [("rank", "Lose 1 rank (31 of 32 GPUs left)", 31 / 32), ("node", "Lose 1 node (28 of 32 GPUs left)", 28 / 32)]
for ax, (kind, label, ideal) in zip(axes, KINDS):
    allv = []
    for k, (m, name, col) in enumerate(MODES):
        rs = CLIENT[m][kind]
        vals = [v["ratio"] * 100 for v in rs]
        allv += vals
        med = float(np.median(vals))
        ax.barh(k, med, 0.56, facecolor=MFILL[m], edgecolor=col, lw=LW_BAR, zorder=2)
        ax.scatter(vals, k + np.linspace(-0.18, 0.18, len(vals)), color=INK, zorder=3, **RUN_DOT)
        NROW.setdefault(m, set()).add(len(vals))
        shown, ref = half_up(med), half_up(ideal * 100)
        gap = signed(half_up(shown - ref), unit=" pts")
        check("fig8", f"{kind} {m}", gap, signed(round(shown - ref, 1), unit=" pts"))
        ax.annotate(f"{half_up(med):.1f}%", (min(vals + [med]), k), xytext=(-8, 0), textcoords="offset points", ha="right",
                    va="center", fontsize=FS_VAL, fontweight="bold", color=INK, zorder=4)
        ax.annotate(gap, (max(vals + [ideal * 100]), k),
                    xytext=(8, 0), textcoords="offset points", ha="left", va="center", fontsize=FS_NOTE, color=LAB,
                    zorder=4, annotation_clip=False)
        for v in rs:
            assert abs(v["ideal"] - ideal) < 0.001
            rows.append([kind, m, v["src"], v["row"], v["before"], v["after"], v["win"], f"{v['ratio']:.3f}",
                         f"{ideal:.4f}"])
        rows.append([kind, m, "MEDIAN", f"n={len(vals)}", "", "", "", f"{med / 100:.4f}", f"{ideal:.4f}"])
    allmed = float(np.median(allv))
    rows.append([kind, "all modes", "MEDIAN", f"n={len(allv)}", "", "", "", f"{allmed / 100:.4f}", f"{ideal:.4f}"])
    ax.plot([ideal * 100] * 2, [0, 1.04], transform=ax.get_xaxis_transform(), color=REFC, lw=LW_REF, ls="--",
            clip_on=False, zorder=3)
    ax.text(ideal * 100 + 1, 1.05, f"proportional to GPUs left: {half_up(ideal * 100):.1f}%",
            transform=ax.get_xaxis_transform(),
            ha="right", va="bottom", fontsize=FS_NOTE, color=INK)
    ax.set_xlim(0, 100)
    ax.set_xticks(range(0, 101, 20), [f"{v}%" for v in range(0, 101, 20)])
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", visible=True)
    ax.tick_params(length=0)
    ax.set_title(label, loc="left", color=INK, fontsize=FS_PANEL, fontweight="bold", pad=34)
assert all(len(v) == 1 for v in NROW.values()), NROW
axes[0].set_yticks(range(3), [f"{n} (n = {NROW[m].pop()})" for m, n, _ in MODES])
axes[0].set_ylim(2.5, -0.5)
finish(fig, "Throughput after a loss",
       [Line2D([], [], color=INK, marker="o", ms=MS_RUN, mec="white", mew=2 * PX, lw=0, label="each run")],
       w_pad=5, bottom=0 if SLIDE else 0.07, h_pad=4 if SLIDE else None,
       sub="client output tok/s on a decode plateau, after vs before the loss; bar = median per mode")
if SLIDE:
    axes[-1].set_xlabel("throughput after / before (%)")
else:
    shared_xlabel(axes, "throughput after / before (%)", gap=24)
save(fig, "fig8_after_loss", rows, ["failure", "mode", "source", "row", "client_tok_s_before", "client_tok_s_after",
                                     "after_window", "ratio", "share_of_gpus_left"],
     "client after/before ratio from review/diag/client_tput_after_kill.txt (CLAIMS 209, 239): rank5/node1 rows of "
     f"{', '.join(sorted({r[2] for r in rows if r[2] != 'MEDIAN'}))}; n per failure: "
     + ", ".join(f"{k} {sum(len(CLIENT[m][k]) for m, _, _ in MODES)}" for k, _, _ in KINDS))

# ---------------------------------------------------------------- Fig A1: per-step cost
fig, ax = plt.subplots(figsize=canvas(SIZE, 1800))
rows = []
w = 0.17
GC = (0, 0.75)
xt = []
for i, L in enumerate(LOADS):
    for k, (m, name, col) in enumerate(MODES):
        n, mean, lo, hi = TRIM[L][(m, L)]
        ps = [p for p in H[L] if p["mode"] == m]
        us = np.mean([(p["trim_on"] - p["trim_off"]) * 1000 for p in ps])
        x = GC[i] + (k - 1) * w
        xt.append((x, name))
        stub(ax, x, lo, hi, vertical=True)
        crosses = lo <= 0 <= hi
        tv = [p["trim"] for p in ps]
        ax.scatter(x + np.linspace(-0.02, 0.02, len(tv)) + 0.07, tv, color=REF, zorder=2, **RUN_DOT)
        ax.scatter(x, mean, s=S_MAIN * (1.2 if MMARK[m] == "^" else 1), marker=MMARK[m], zorder=3,
                   color="white" if crosses else col, edgecolor=col, lw=3 if crosses else 2,
                   **rim(m, 3 if crosses else 2))
        rows.append([L, m, n, f"{mean:+.2f}", f"{lo:+.2f}", f"{hi:+.2f}", f"{us:.1f}",
                     f"{np.mean([p['trim_off'] for p in ps]):.3f}", f"{np.mean([p['trim_on'] for p in ps]):.3f}"])
    ax.text(GC[i], 1.02, f"{L} requests/GPU (n = {n} pairs)", transform=ax.get_xaxis_transform(), ha="center",
            va="bottom", fontsize=FS_PANEL, fontweight="bold", color=INK)
    for m, _, _ in MODES:  # the original n=4 cells (CLAIMS 194), for comparison with the pooled n=8
        ps = [p for p in pairs(ORIG, L) if p["mode"] == m]
        if True:
            rows.append([f"{L} (original n=4, CLAIMS 194)", m, len(ps), f"{np.mean([p['trim'] for p in ps]):+.2f}",
                         "", "", f"{np.mean([(p['trim_on'] - p['trim_off']) * 1000 for p in ps]):.1f}", "", ""])
ax.axhline(1, color=REFC, lw=LW_REF, ls="--")
ax.text(1.08, 1.02, "1% threshold", ha="right", va="bottom", fontsize=FS_NOTE, color=INK)
ax.set_xticks([x for x, _ in xt], [t for _, t in xt])
nogrid_x(ax)
ax.set_xlim(-0.35, 1.1); ax.set_ylim(-0.25, 1.2)
ax.spines["bottom"].set_visible(False)
ax.axhline(0, color=AXIS, lw=1.0)
ax.set_yticks([-0.2, 0, 0.2, 0.4, 0.6, 0.8, 1.0], [signed(v, 1) for v in (-0.2, 0, 0.2, 0.4, 0.6, 0.8, 1.0)])
ax.set_ylabel("change vs FT off (%)")
finish(fig, "Decode step time (trimmed mean), FT on vs off",
       mode_marks(ms=8) + [ci_glyph(ax, "mean, 95% CI", color=INK, hollow=True),
                           Line2D([], [], color=LAB, marker="o", ms=10, lw=0, mfc="white", mew=1.8,
                                  label="open: CI includes 0"),
                           Line2D([], [], color=REF, marker="o", ms=MS_RUN, mec="white", mew=2 * PX, lw=0,
                                  label="each pair")],
       sub="trimmed mean = mean of rank-0 steady decode steps \u2264 1.5\u00d7 the median; mean of "
           "per-pair % change, 95% t-CI")
save(fig, "figA1_step_cost", rows, ["load", "mode", "n", "trim_mean_pct", "ci_lo", "ci_hi", "extra_us_per_step",
                                     "trim_off_ms", "trim_on_ms"],
     f"trim % and CI from repl_cost_pooled.txt (32, n=8) and healthy128_pooled_readout.txt (128, n=8); us from pair "
     "trim off/on ms columns")

# ---------------------------------------------------------------- Fig A2: throughput per GPU vs per user
fig, axes = plt.subplots(1, 3, figsize=SIZE, sharey=True, sharex=True)
rows = []
for ax, (m, name, col) in zip(axes, MODES):
    for armk, fc, lab in (("off", OFF, "FT off"), ("on", GREEN, "FT on")):
        xs, ys = [], []
        for L in LOADS:
            ps = [p for p in H[L] if p["mode"] == m]
            px = [1000 / p[armk]["median_tpot_ms"] for p in ps]
            py = [p[armk]["output_throughput"] / GPUS for p in ps]
            xs.append(np.mean(px)); ys.append(np.mean(py))
            for p, a, b in zip(ps, px, py):
                rows.append([m, L, armk, p["group"], f"{a:.3f}", f"{b:.2f}"])
            rows.append([m, L, armk, f"MEAN n={len(ps)}", f"{np.mean(px):.3f}", f"{np.mean(py):.2f}"])
        off_pts = ScaledTranslation(-3 / 72, 0, fig.dpi_scale_trans)
        on_pts = ScaledTranslation(3 / 72, 0, fig.dpi_scale_trans)
        if armk == "off":
            ax.plot(xs, ys, lw=0, marker="o", ms=10, mfc="white", mec=INK, mew=1.1, zorder=2,
                    transform=ax.transData + off_pts)
        else:
            ax.plot(xs, ys, lw=0, marker="s", ms=7, color=GREEN, zorder=3, transform=ax.transData + on_pts)
            for L, a, b in zip(LOADS, xs, ys):
                ax.annotate(f"{L} requests/GPU", (a, b), xytext=(12, 0), textcoords="offset points", ha="left",
                            va="center", fontsize=9.5, color=LAB)
    panel_title(ax, name)
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:,.0f}")
axes[0].set_ylabel("client output tok/s per GPU")
axes[0].set_xlim(40, 165); axes[0].set_xticks([50, 100, 150]); axes[0].set_ylim(2500, 6800)
finish(fig, "Throughput vs per-user speed",
       [Line2D([], [], lw=0, marker="o", ms=10, mfc="white", mec=INK, mew=1.1, label="FT off"),
        Line2D([], [], lw=0, marker="s", ms=7, color=GREEN, label="FT on")],
       sub="mean of n = 8 same-node pairs per load (32 and 128 requests/GPU)", bottom=0.05)
shared_xlabel(axes, "client output tok/s per user")
save(fig, "figA2_throughput", rows, ["mode", "load", "arm", "pair", "tok_s_per_user", "tok_s_per_gpu"],
     f"bench.log healthy-32 (n=8 pairs/mode, {COST_JOBS32}) and healthy-128 (n=8, {COST_JOBS128}); "
     "per-user tok/s = 1000 / median TPOT")

# ---------------------------------------------------------------- Fig 10: realistic traffic (S1, ~80% load)
S1_JOBS = ["8310100", "8359063", "8363138"]  # STRESS-S1 jobs, two same-node off/on pairs each at 51 req/s (CLAIMS 217, 234, 240)
S1_KEYS = [("ttft_p50_ms", "TTFT", "p50"), ("ttft_p99_ms", "TTFT", "p99"), ("tpot_p50_ms", "TPOT", "p50"),
           ("tpot_p99_ms", "TPOT", "p99")]
S1 = []
for job in S1_JOBS:
    for line in open(f"{REV}/diag/stress_s1_{job}.txt"):
        if line.startswith("STRESS_PAIR "):
            kv = dict(t.split("=", 1) for t in line.split()[3:])
            S1.append({k: (float(kv[f"off_{k}"]), float(kv[f"on_{k}"])) for k, _, _ in S1_KEYS})
assert len(S1) == 2 * len(S1_JOBS)
N1 = len(S1)
fig, axes = plt.subplots(*((2, 1) if SLIDE else (1, 2)), figsize=canvas(SIZE, 2056))
rows = []
w = 0.29
PANEL = {"TTFT": "Time to first token (TTFT)", "TPOT": "Time per output token (TPOT)"}
SPREAD = []
for ax, metric, ymax in zip(axes, ("TTFT", "TPOT"), (2500, 22)):
    keys = [(k, q) for k, mt, q in S1_KEYS if mt == metric]
    chg = []
    for i, (k, q) in enumerate(keys):
        off = float(np.mean([p[k][0] for p in S1]))
        on = float(np.mean([p[k][1] for p in S1]))
        xo, xn = i - w / 2 - 0.02, i + w / 2 + 0.02
        nd = 0 if metric == "TTFT" else 2
        ax.bar(xo, off, w, **OFF_BAR)
        ax.bar(xn, on, w, **ON_BAR)
        so, sn = half_up(off, nd), half_up(on, nd)
        value_label(ax, xo, max([off] + [p[k][0] for p in S1]), f"{so:,.{nd}f}", dy=8)
        value_label(ax, xn, max([on] + [p[k][1] for p in S1]), f"{sn:,.{nd}f}", dy=8)
        pp = [(p[k][1] / p[k][0] - 1) * 100 for p in S1]
        c = (sn / so - 1) * 100
        lab = "~0%" if min(pp) < 0 < max(pp) else pct1(so, sn)
        dms = {round(p[k][1] - p[k][0], 6) for p in S1}
        if N1 >= 3 and len(dms) == 1:  # every pair moved by the same step of the readout's 0.1 ms resolution
            lab = f"{signed(dms.pop(), unit=' ms')} in all {N1} pairs"
        elif N1 >= 3:  # 95% t-CI of the per-pair % change
            mu, lo, hi = tci(pp)
            lab = f"{signed(half_up(mu))} [{signed(half_up(lo))}, {signed(half_up(hi))}]"
        chg.append(lab)
        check("fig10", k, pct1(so, sn), signed(half_up((sn / so - 1) * 100)))
        SPREAD.append((k, round(c, 2), [round(x, 2) for x in pp], chg[-1]))
        for xb, j in ((xo, 0), (xn, 1)):  # runs spread across the bar's centreline, in job order
            ax.scatter(xb + np.linspace(-0.15, 0.15, N1) * w, [p[k][j] for p in S1], color=INK, zorder=4, **RUN_DOT)
        for j, p in enumerate(S1, 1):
            rows.append([k, f"pair{j}", f"{p[k][0]:.1f}", f"{p[k][1]:.1f}", f"{(p[k][1] / p[k][0] - 1) * 100:+.2f}"])
        rows.append([k, f"MEAN n={N1}", f"{off:.2f}", f"{on:.2f}", f"{(on / off - 1) * 100:+.2f}"])
    ax.set_xticks(range(len(keys)), [f"{q}\nchange {d} (n = {N1}{', no CI' if N1 < 3 else ''})"
                                     for (_, q), d in zip(keys, chg)])
    nogrid_x(ax)
    ax.set_ylim(0, ymax)
    ax.set_yticks(range(0, ymax + 1, 500 if metric == "TTFT" else 5))
    panel_title(ax, PANEL[metric])
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:,.0f}")
    ax.set_ylabel("ms")
finish(fig, "Latency under realistic traffic",
       [box("white", INK, "FT off", 1.0), box(ON_FILL, ON_EDGE, "FT on"),
        Line2D([], [], color=INK, marker="o", ms=MS_RUN, mec="white", mew=2 * PX, lw=0, label="each run")],
       sub="DeepSeek-R1 NVFP4; DEP32 on 32 GB200; 51 requests/s (\u224879\u201381% of the calibrated "
           f"rate); bar = mean of {N1} runs", h_pad=4 if SLIDE else None)
if SLIDE:
    fig.align_ylabels(axes)
save(fig, "fig10_realistic_latency", rows, ["metric", "pair", "ft_off_ms", "ft_on_ms", "change_pct"],
     f"STRESS_PAIR lines of review/diag/stress_s1_<job>.txt (jobs {', '.join(S1_JOBS)}, 51 req/s, {N1} same-node pairs); "
     "bars: mean of pairs")

for fig_, lab, shown, rec, ok in CHECKS:
    print(f"CHECK {fig_} {lab}: shown {shown} recomputed {rec} {'OK' if ok else 'MISMATCH'}")
for k, c, pp, lab in SPREAD:
    print(f"SPREAD fig10 {k}: change of displayed means {c:+.2f}% per-pair {pp} -> label {lab}")
print(f"HW fig6 tok/s {min(hw_tok):.2f}-{max(hw_tok):.2f} all max {max(hw_all):.2f}")
assert all(ok for *_, ok in CHECKS)
