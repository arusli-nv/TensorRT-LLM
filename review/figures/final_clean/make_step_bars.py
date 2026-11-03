"""Thin chevron bars that show where a slide sits: the 5 recovery steps (design slides) and the 3 result sections.

usage: /usr/bin/python3 make_step_bars.py   (writes FINAL/figures/talk/bars/<name>.png (transparent) and .svg)
Current item = solid green, earlier items = light green, later items = light gray.
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.patches import Polygon  # noqa: E402

ROOT = "/lustre/fs1/portfolios/coreai/projects/coreai_comparch_trtllm/users/arusli/src/TensorRT-LLM/.wideep-ft-runs/claude"
OUT = f"{ROOT}/FINAL/figures/talk/bars"
for f in ("Lato-Regular.ttf", "Lato-Bold.ttf"):
    font_manager.fontManager.addfont(f"{ROOT}/mvp/analysis/fonts/{f}")
plt.rcParams.update({"font.family": "Lato", "svg.fonttype": "none"})
INK, LABEL = "#2D3142", "#4F5D75"
STYLE = {"current": ("#76B900", INK, "bold"), "done": ("#E3F1CC", "#2F5F10", "normal"),
         "next": ("#ECEFF2", LABEL, "normal")}
STEPS = ["Detect", "Agree", "Plan", "Install", "Resume"]
SECTIONS = ["Recovery", "Healthy cost", "Other situations"]
H, GAP, DPI, FS = 0.40, 0.05, 200, 13  # inches, inches, dpi, pt


def bar(name, items, current, width):
    """One chevron per item; ``current`` is the set of items to highlight."""
    first = min(items.index(c) for c in current)
    n, d = len(items), H * 0.32
    w = (width - GAP * (n - 1)) / n
    fig = plt.figure(figsize=(width, H))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, width)
    ax.set_ylim(0, H)
    ax.axis("off")
    for i, item in enumerate(items):
        state = "current" if item in current else "done" if i < first else "next"
        fill, ink, weight = STYLE[state]
        x = i * (w + GAP)
        left = [(x, 0), (x, H)] if i == 0 else [(x, 0), (x + d, H / 2), (x, H)]
        right = [(x + w, H), (x + w, 0)] if i == n - 1 else [(x + w - d, H), (x + w, H / 2), (x + w - d, 0)]
        ax.add_patch(Polygon(left + right, closed=True, fc=fill, ec="none"))
        cx = x + (w + (d if i else 0) - (d if i < n - 1 else 0)) / 2
        ax.text(cx, H / 2, item, ha="center", va="center", fontsize=FS, color=ink, fontweight=weight)
    for ext in ("png", "svg"):
        fig.savefig(f"{OUT}/{name}.{ext}", dpi=DPI, transparent=True)
    plt.close(fig)
    print(name)


os.makedirs(OUT, exist_ok=True)
for name, cur in [("step_detect", {"Detect"}), ("step_agree", {"Agree"}), ("step_plan_install", {"Plan", "Install"}),
                  ("step_plan", {"Plan"}), ("step_resume", {"Resume"})]:
    bar(name, STEPS, cur, 7.5)
for name, cur in [("section_recovery", "Recovery"), ("section_healthy_cost", "Healthy cost"),
                  ("section_other", "Other situations")]:
    bar(name, SECTIONS, {cur}, 5.0)
