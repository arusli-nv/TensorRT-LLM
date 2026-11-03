"""Evaluation figures in the design figures' style: soft fills, coloured outlines, rounded bars, compact."""
import json, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np

DATA, OUT, FONTS = sys.argv[1:4]
for f in ("Lato-Regular.ttf", "Lato-Bold.ttf"):
    fm.fontManager.addfont(f"{FONTS}/{f}")
plt.rcParams.update({"font.family": "Lato", "font.size": 11.5, "axes.titlesize": 12.5, "axes.titleweight": "bold",
    "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": False, "axes.edgecolor": "#9A9A9A",
    "axes.grid": True, "grid.color": "#EFEFEF", "axes.axisbelow": True, "ytick.left": False,
    "xtick.color": "#555555", "ytick.color": "#555555", "legend.frameon": False, "savefig.dpi": 220,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.08})
GREEN, GREEN_FILL = "#2CA02C", None
GRAY, GRAY_FILL, INK, MUTED = "#7F7F7F", None, "#1A1A1A", "#707070"
BLUE, BLUE_FILL = "#1F77B4", None
ORANGE = "#FF7F0E"
RED = "#C8372D"
d = json.load(open(DATA))
DEPS = [("Online EPLB", "LADDER-8099447", "noft", "ft", GREEN, GREEN_FILL),
        ("Offline EPLB (mock)", None, None, None, ORANGE, None),
        ("No EPLB", "GHOST-COST-8099447", "lay-eplb0", "ft-ghost320", BLUE, BLUE_FILL)]
# Offline EPLB has no runs yet: placeholder (mean, min, max) per load for [step, throughput], clearly labelled mock.
MOCK = {0: [(1.2, 1.0, 1.4), (0.8, 0.6, 1.0), (0.8, 0.6, 1.0), (1.0, 0.7, 1.3)],
        1: [(2.0, 0.5, 3.5), (0.8, 0.2, 1.4), (0.4, 0.0, 0.8), (0.5, -1.5, 2.5)]}
LOADS = [8, 32, 128, 512]


def stamp(fig, text):
    fig.text(0.995, 0.995, text, color=RED, ha="right", va="top", fontsize=11, fontweight="bold")


def save(fig, name):
    fig.savefig(f"{OUT}/{name}.png"); fig.savefig(f"{OUT}/{name}.svg"); plt.close(fig)


def vbar(ax, x, y0, y1, w, fill, edge, r=0.06, alpha=1.0, hatch=None):
    lo, hi = min(y0, y1), max(y0, y1)
    ax.add_patch(FancyBboxPatch((x - w / 2, lo), w, max(hi - lo, 1e-6), boxstyle=f"round,pad=0,rounding_size={r}",
                                facecolor=fill, edgecolor=edge, lw=1.6, alpha=alpha, hatch=hatch,
                                mutation_aspect=1))


def hbar(ax, y, x0, x1, h, fill, edge, lw=1.6):
    ax.add_patch(FancyBboxPatch((x0, y - h / 2), x1 - x0, h, boxstyle="round,pad=0,rounding_size=0.06",
                                facecolor=fill, edgecolor=edge, lw=lw, mutation_aspect=1))


# Figure 5: FT cost in %, two panels (median step | throughput). Bar = mean, T whiskers = min..max of the paired runs.
def cost_rows(idx, sign):
    rows = []
    for k, (name, pack, off, on, edge, fill) in enumerate(DEPS):
        for i, L in enumerate(LOADS):
            x = i + (k - 1) * 0.28
            if pack is None:
                rows.append((x, *MOCK[idx][i], edge, fill)); continue
            o = [r[idx] for r in d[pack][off][str(L)]]; n = [r[idx] for r in d[pack][on][str(L)]]
            pairs = [sign * (b - a) / a * 100 for a, b in zip(o, n)]
            m = sign * (np.mean(n) - np.mean(o)) / np.mean(o) * 100
            rows.append((x, m, min(pairs), max(pairs), edge, fill))
    return rows


def data_per_point(ax):
    """Data units per typographic point along x and y, for the axes' current limits and size."""
    bb = ax.get_window_extent(); k = 72 / ax.figure.dpi
    (x0, x1), (y0, y1) = ax.get_xlim(), ax.get_ylim()
    return (x1 - x0) / bb.width / k, (y1 - y0) / bb.height / k


def rbox(ax, x, y, w, h, color, r_pt=2.0, **kw):
    """Rectangle (x, y, w, h) in data units with corners rounded by r_pt points."""
    dx, dy = data_per_point(ax)
    r = min(r_pt * dx, abs(w) / 2, abs(h) / 2 * dx / dy)
    return FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}", facecolor=color,
                          edgecolor="none", mutation_aspect=dy / dx, **kw)


def rounded_vbar(ax, x, y, w, color, r_pt=2.0):
    """Bar from 0 to y, rounded only at the far end: a rounded box clipped at the zero line."""
    ext = 2 * r_pt * data_per_point(ax)[1]
    lo, hi = (-ext, y) if y >= 0 else (y, ext)
    p = ax.add_patch(rbox(ax, x - w / 2, lo, w, hi - lo, color, r_pt, zorder=2))
    p.set_clip_path(plt.Rectangle((x - w, min(0, y)), 2 * w, abs(y), transform=ax.transData))


def rounded_stack(ax, y, segs, h, colors, r_pt=2.0):
    """Horizontal stacked bar from 0 with rounded outer corners and thin white gaps between segments."""
    outline = rbox(ax, 0, y - h / 2, sum(segs), h, "none", r_pt, transform=ax.transData)
    left = 0
    for s_, c in zip(segs, colors):
        bar = ax.barh(y, s_, h, left=left, color=c, edgecolor="white", lw=0.8)[0]
        bar.set_clip_path(outline)
        left += s_
    return left


fig, ax = plt.subplots(figsize=(9.6, 3.6))
axes = [ax]
for ax, (idx, sign, title) in zip(axes, [(0, 1, "Decode step time, FT on vs FT off")]):
    rows = cost_rows(idx, sign)
    lo = min(min(r[1], r[2]) for r in rows); hi = max(max(r[1], r[3]) for r in rows)
    pad = 0.14 * (hi - min(lo, 0))
    ylim = (lo - pad if lo < 0 else 0, hi + pad)
    ax.set_xlim(-0.6, 3.6); ax.set_ylim(*ylim)
    bb = ax.get_window_extent()
    aspect = (ylim[1] - ylim[0]) / bb.height / ((4.2) / bb.width)
    ax.axhline(0, color=INK, lw=1, zorder=3)
    placed = []
    for x, m, wlo, whi, edge, fill in rows:
        rounded_vbar(ax, x, m, 0.25, edge)
        ax.errorbar(x, m, yerr=[[m - wlo], [whi - m]], fmt="none", ecolor="#3A3A3A", elinewidth=1.1,
                    capsize=4, capthick=1.1, zorder=4)
        up = m >= 0
        step = 0.06 * (ylim[1] - ylim[0])
        y = (whi + 0.012 * (ylim[1] - ylim[0])) if up else (wlo - 0.012 * (ylim[1] - ylim[0]))
        placed.append((x, y))
        ax.text(x, y, f"{m:+.1f}%", ha="center", va="bottom" if up else "top", fontsize=9.5, color=INK, zorder=5)
    ax.set_xticks(range(4), [str(L) for L in LOADS])
    ax.set_xlabel("Requests per GPU")
    ax.set_title(title, loc="left")
    ax.grid(axis="x", visible=False)
axes[0].set_ylabel("Slower with FT (%)")
fig.tight_layout()
fig.subplots_adjust(bottom=0.3)
handles = [plt.Rectangle((0, 0), 1, 1, color=e) for _, _, _, _, e, _ in DEPS]
fig.legend(handles, [n for n, *_ in DEPS], loc="lower center", ncol=3, bbox_to_anchor=(0.5, 0.0))
stamp(fig, "PRELIMINARY DATA; OFFLINE EPLB IS MOCK")
save(fig, "fig5_cost")

# Figure 6: recovery time. (a) time to serve again, log x; (b) FT on breakdown; (c) cold restart breakdown.
rows = ["1 rank\n(process killed)", "1 node\n(all ranks killed)", "Frozen rank\n(SIGSTOP)"]
series = [("FT on, online EPLB", [1.54, 1.49, 5.24], GREEN), ("FT on, offline EPLB (mock)", [1.3, 1.8, 6.0], ORANGE),
          ("FT on, no EPLB", [1.00, 2.08, 6.63], BLUE), ("FT off, cold restart", [477, 539, 855], GRAY)]
fig = plt.figure(figsize=(9.6, 7.4))
gs = fig.add_gridspec(2, 2, height_ratios=[1.15, 1], hspace=0.5, wspace=0.38)
ax = fig.add_subplot(gs[0, :])
# Plotted in log10(seconds) on a linear axis so the rounded corners stay round.
x0 = np.log10(0.3)
ax.set_xlim(x0, np.log10(6000)); ax.set_ylim(-0.55, 2.55)
y = np.arange(3)[::-1] * 1.0
h = 0.19
for k, (lab, vals, color) in enumerate(series):
    yy = y + (1.5 - k) * (h + 0.03)
    for j, (yi, v) in enumerate(zip(yy, vals)):
        ax.add_patch(rbox(ax, x0, yi - h / 2, np.log10(v) - x0, h, color, 1.5))
        txt = (f"{v:.2f} s" if v < 100 else f"{v:.0f} s") + (f"   {lab}" if j == 0 else "")
        ax.text(np.log10(v) + 0.05, yi, txt, va="center", fontsize=9.5, color=color if j == 0 else INK)
ax.set_xticks(range(4), [f"$10^{{{e}}}$" for e in range(4)])
ax.set_xticks([np.log10(m * 10 ** e) for e in range(-1, 4) for m in range(2, 10)
               if x0 <= np.log10(m * 10 ** e) <= np.log10(6000)], minor=True)
ax.set_yticks(y, rows); ax.grid(axis="y", visible=False)
ax.set_xlabel("Time from the failure to serving again (s, log scale)")
ax.set_title("(a) Time to serve again", loc="left")

greens = ["#00441B", "#238B45", "#41AB5D", "#74C476", "#C7E9C0"]
ax = fig.add_subplot(gs[1, 0])
ax.set_xlim(0, 6.6); ax.set_ylim(-0.6, 2.9)
br = {"1 rank": [0.08, 0.66, 0.01, 0.04, 0.75], "1 node": [0.03, 0.62, 0.02, 0.05, 0.77],
      "Frozen rank": [4.25, 0.20, 0.02, 0.05, 0.72]}
yb = np.arange(3)[::-1]
for yi, seg in zip(yb, br.values()):
    total = rounded_stack(ax, yi, seg, 0.5, greens)
    ax.text(total + 0.1, yi, f"{total:.1f} s", va="center", fontsize=9.5)
ax.legend([plt.Rectangle((0, 0), 1, 1, color=c) for c in greens], ["detect", "agree", "plan", "install", "first step"],
          loc="upper center", bbox_to_anchor=(0.5, -0.3), ncol=3, fontsize=9.5, handlelength=1.1, columnspacing=1.0)
ax.set_yticks(yb, list(br)); ax.grid(axis="y", visible=False)
ax.set_xlabel("Seconds from the failure")
ax.set_title("(b) FT on (online EPLB)", loc="left")

ax = fig.add_subplot(gs[1, 1])
ax.set_xlim(0, 17); ax.set_ylim(-0.6, 2.9)
cold = {"1 rank": [2, 8, 254, 60, 113, 40], "1 node": [3, 9, 330, 58, 105, 34], "Frozen rank": [299, 9, 380, 55, 102, 10]}
grays = ["#252525", "#525252", "#737373", "#969696", "#BDBDBD", "#D9D9D9"]
for yi, seg in zip(yb, cold.values()):
    total = rounded_stack(ax, yi, [x / 60 for x in seg], 0.5, grays)
    ax.text(total + 0.2, yi, f"{total:.0f} min", va="center", fontsize=9.5)
ax.legend([plt.Rectangle((0, 0), 1, 1, color=c) for c in grays],
          ["detect", "teardown", "load weights", "autotune, warmup", "capture graphs", "first token"],
          loc="upper center", bbox_to_anchor=(0.5, -0.3), ncol=3, fontsize=9.5, handlelength=1.1, columnspacing=1.0)
ax.set_yticks(yb, list(cold)); ax.grid(axis="y", visible=False)
ax.set_xlabel("Minutes from the failure")
ax.set_title("(c) FT off (cold restart)", loc="left")
stamp(fig, "PRELIMINARY DATA; OFFLINE EPLB IS MOCK")
save(fig, "fig6_recovery")

# Figure 4: throughput vs per-user speed (TRT-LLM blog style), FT off vs FT on, one panel per EPLB mode.
P = json.load(open(sys.argv[4]))
def pts(pack, arm):
    xs, ys = [], []
    for L in LOADS:
        r = P[pack][arm][str(L)]
        xs.append(np.mean([q[1] for q in r])); ys.append(np.mean([q[0] for q in r]))
    return np.array(xs), np.array(ys)
noeplb = {"off": pts("GHOST-COST-8099447", "lay-eplb0"), "on": pts("GHOST-COST-8099447", "ft-ghost320")}
online = {"off": pts("LADDER-8099447", "noft"), "on": pts("LADDER-8099447", "ft")}
# Offline EPLB has no runs yet: mock curve between the two measured modes.
mx, my = (noeplb["off"][0] + online["off"][0]) / 2, (noeplb["off"][1] + online["off"][1]) / 2
offline = {"off": (mx, my), "on": (mx * 0.995, my * 0.993)}
fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.9), sharey=True)
for ax, (title, dd, color) in zip(axes, [("No EPLB", noeplb, BLUE), ("Offline EPLB (mock)", offline, ORANGE),
                                        ("Online EPLB", online, GREEN)]):
    ax.plot(*dd["off"], color=GRAY, marker="o", ms=5, lw=1.8, label="FT off")
    ax.plot(*dd["on"], color=color, marker="o", ms=5, lw=1.8, label="FT on")
    for L, x, y in zip(LOADS, *dd["off"]):
        ax.annotate(str(L), (x, y), xytext=(6, 4), textcoords="offset points", fontsize=8.5, color=MUTED)
    ax.set_title(title, loc="left"); ax.set_xlabel("Output tok/s per user")
    ax.set_xlim(0, 135); ax.set_ylim(0, 8000)
    ax.grid(axis="x", visible=True)
    ax.legend(loc="upper right", fontsize=9.5)
axes[0].set_ylabel("Output tok/s per GPU")
fig.text(0.5, -0.04, "Point labels: concurrent requests per GPU.", ha="center", fontsize=9.5, color=MUTED)
fig.tight_layout(w_pad=1.5)
stamp(fig, "PRELIMINARY DATA; OFFLINE EPLB IS MOCK")
save(fig, "fig4_throughput")
print("ok")
