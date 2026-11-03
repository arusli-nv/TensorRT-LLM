# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Render the finals deck (16:9 PNG per slide + one PDF): figures 1-11 of review/FINALS_PLAN.md v4, then the appendix.

Usage: PYTHONPATH=$TMPDIR/figlib python3 slides_deck.py --out <dir>
Every data slide carries a label: REAL (final runs), LEAD/INDICATIVE (earlier runs), or MOCK.
"""
import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "..", "figures")
GREEN, DARK, GREY, RED, BLUE = "#76B900", "#1A1A1A", "#6B6B6B", "#C0392B", "#4A6FA5"
plt.rcParams.update({"font.family": "Liberation Sans", "font.size": 16})
LABEL_COLORS = {"REAL": GREEN, "LEAD": "#E69F00", "INDICATIVE": "#E69F00", "MOCK": RED}


def new_slide(title, takeaway=None, label=None, source=None):
    fig = plt.figure(figsize=(13.33, 7.5), dpi=150)
    fig.patches.append(plt.Rectangle((0, 0.915), 1, 0.085, transform=fig.transFigure, color=DARK, zorder=-1))
    fig.patches.append(plt.Rectangle((0, 0.91), 1, 0.006, transform=fig.transFigure, color=GREEN))
    fig.text(0.03, 0.955, title, color="white", fontsize=28, weight="bold", va="center")
    if takeaway:
        fig.text(0.03, 0.865, takeaway, color=DARK, fontsize=19, va="center", style="italic")
    if label:
        fig.text(0.97, 0.955, label, color=LABEL_COLORS.get(label.split()[0], GREY), fontsize=15, weight="bold",
                 ha="right", va="center")
    if source:
        fig.text(0.03, 0.02, source, color=GREY, fontsize=11)
    return fig


def bullets(fig, items, x=0.04, y=0.80, width=0.9, size=19, gap=0.075):
    for item in items:
        sub = item.startswith("  ")
        fig.text(x + (0.03 if sub else 0), y, ("– " if sub else "• ") + item.strip(), fontsize=size - (2 if sub else 0),
                 color=GREY if sub else DARK, va="top", wrap=True)
        y -= gap * (0.8 if sub else 1)
    return y


def image(fig, name, rect):
    path = os.path.join(FIG, name)
    ax = fig.add_axes(rect)
    ax.axis("off")
    if os.path.exists(path):
        ax.imshow(mpimg.imread(path))
    else:
        ax.text(0.5, 0.5, f"[missing figure: {name}]", ha="center", va="center", color=RED, fontsize=16)



MOCK_SRC = "MOCK data: layout only, no measured values. Final slides cite each number's mvp/CLAIMS.md row."
MODES = ["no EPLB", "offline EPLB", "online EPLB"]
FT_OFF, FT_ON = GREY, GREEN


def mock_slide(title, takeaway, source=MOCK_SRC):
    return new_slide(title, takeaway, "MOCK", source)


def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def slide_headline():
    fig = mock_slide("1. Healthy cost of FT, all modes", "FT on − off with 95% CI at 32 and 128 req/GPU; "
                     "the bar to beat is 1%.")
    cost = {"decode step (primary)": [[0.2, 0.1], [0.3, 0.2], [0.4, 0.3]],
            "client tok/s (end-to-end)": [[0.4, 0.3], [0.5, 0.4], [0.6, 0.5]]}
    for k, (metric, vals) in enumerate(cost.items()):
        ax = fig.add_axes([0.07 + 0.47 * k, 0.14, 0.40, 0.62])
        for i, mode in enumerate(MODES):
            for j, load in enumerate((32, 128)):
                ax.bar(i * 3 + j, vals[i][j], yerr=0.3, color=(BLUE, GREEN)[j], capsize=5,
                       label=f"{load}/GPU" if i == 0 else None)
        ax.axhline(1, color=RED, ls="--")
        ax.text(8.2, 1.03, "1%", color=RED, fontsize=14)
        ax.set_xticks([0.5, 3.5, 6.5], MODES, fontsize=14)
        ax.set_ylim(-0.5, 1.6)
        ax.set_ylabel("FT cost, % (n=8 pairs, 95% CI)")
        ax.set_title(metric, fontsize=16)
        ax.legend(frameon=False, fontsize=13)
        style(ax)
    return fig


def slide_modes(loads=(32, 128), title="2. Throughput, TPOT and TTFT by mode",
                takeaway="FT off vs on on the same nodes; same axes for every mode."):
    fig = mock_slide(title, takeaway)
    for r, metric in enumerate(("tok/s per GPU", "TPOT, ms", "TTFT, s")):
        for c, mode in enumerate(MODES):
            ax = fig.add_axes([0.08 + 0.31 * c, 0.58 - 0.24 * r, 0.26, 0.17])
            x = range(len(loads))
            base = [1 + i for i in x]
            ax.plot(x, base, "o-", color=FT_OFF, label="FT off")
            ax.plot(x, [b * 1.01 for b in base], "s--", color=FT_ON, label="FT on")
            ax.set_xticks(list(x), [str(v) for v in loads] if r == 2 else [], fontsize=11)
            ax.set_yticks([])
            if c == 0:
                ax.set_ylabel(metric, fontsize=12)
            if r == 0:
                ax.set_title(mode, fontsize=14)
            if r == 0 and c == 2:
                ax.legend(frameon=False, fontsize=11)
            style(ax)
    fig.text(0.5, 0.07, "requests per GPU", ha="center", fontsize=13)
    return fig


def slide_decomposition():
    fig = mock_slide("3. No EPLB: where the cost comes from", "FT off → static-LB floor (main's indirection) → FT "
                     "on: FT adds the last step only.")
    ax = fig.add_axes([0.12, 0.15, 0.76, 0.6])
    steps = ["FT off", "static-LB floor", "FT on"]
    for j, load in enumerate((32, 128)):
        ax.bar([i * 3 + j for i in range(3)], [0, 0.3, 0.4], color=(BLUE, GREEN)[j], yerr=0.2, capsize=5,
               label=f"{load}/GPU")
    ax.set_xticks([0.5, 3.5, 6.5], steps)
    ax.set_ylabel("decode step vs FT off, %")
    ax.legend(frameon=False)
    style(ax)
    return fig


def slide_parity():
    fig = mock_slide("4. Our FT off vs main", "Same nodes, same table; the difference includes the ADP gate fix "
                     "(stated per cell).")
    ax = fig.add_axes([0.12, 0.15, 0.76, 0.6])
    cells = [f"{m}\n{load}/GPU" for m in ("no EPLB", "online EPLB") for load in (32, 128)]
    ax.bar(range(4), [0.1, 0.2, 0.1, 0.2], yerr=0.3, color=BLUE, capsize=6)
    ax.axhline(0, color=DARK, lw=1)
    ax.set_xticks(range(4), cells)
    ax.set_ylabel("ours − main, % (95% CI)")
    style(ax)
    return fig


def slide_timeline():
    fig = mock_slide("5. Serving through failures", "Served tok/s over time: rank kill, node kill, two sequential "
                     "losses; detect, fence, remap, resume marked.")
    for k, name in enumerate(("rank kill", "node kill", "two losses")):
        ax = fig.add_axes([0.07 + 0.31 * k, 0.16, 0.27, 0.58])
        t = [i / 2 for i in range(121)]
        dips = (30,) if k < 2 else (25, 40)
        level = 1.0
        y = []
        for s in t:
            if any(d <= s < d + 1.5 for d in dips):
                y.append(0)
                continue
            level = 1 - (0.13 if k == 1 else 0.03) * sum(s >= d + 1.5 for d in dips)
            y.append(level)
        ax.plot(t, y, color=GREEN)
        for d in dips:
            for off in (0, 0.5, 1.0, 1.5):  # detect, fence, remap, resume
                ax.axvline(d + off, color=GREY, lw=0.6, ls=":")
        ax.set_title(name, fontsize=15)
        ax.set_ylim(0, 1.15)
        ax.set_xlabel("time, s")
        ax.set_yticks([0, 1], ["0", "before"] if k == 0 else ["", ""])
        style(ax)
    return fig


ROWS = ["rank", "node", "two losses", "frozen rank", "in prefill", "EPLB update"]


def slide_resume():
    fig = mock_slide("6. Resume time per failure type", "Per mode; a frozen rank adds the hang confirmation before "
                     "the resume.")
    ax = fig.add_axes([0.1, 0.15, 0.8, 0.6])
    for j, mode in enumerate(("no EPLB", "online EPLB")):
        xs = [i * 3 + j for i in range(len(ROWS))]
        ax.bar(xs, [1] * len(ROWS), color=(BLUE, GREEN)[j], label=f"{mode}: resume")
        ax.bar(xs[3], 3, bottom=1, color=(BLUE, GREEN)[j], alpha=0.35, hatch="//",
               label="frozen: hang confirm" if j == 0 else None)
    ax.set_xticks([i * 3 + 0.5 for i in range(len(ROWS))], ROWS)
    ax.set_yticks([])
    ax.set_ylabel("seconds (scale set by final data)")
    ax.legend(frameon=False, fontsize=13)
    style(ax)
    return fig


def slide_post_loss():
    fig = mock_slide("7. Throughput after a loss vs ideal", "Post-loss tok/s against live/total GPUs per failure "
                     "type.")
    ax = fig.add_axes([0.1, 0.15, 0.8, 0.6])
    ideal = [31 / 32, 28 / 32, 30 / 32, 31 / 32, 31 / 32, 31 / 32]
    ax.bar(range(len(ROWS)), [v * 0.97 for v in ideal], color=GREEN, label="measured (mock)")
    ax.scatter(range(len(ROWS)), ideal, marker="_", s=900, color=DARK, label="ideal live/total")
    ax.set_xticks(range(len(ROWS)), ROWS)
    ax.set_ylim(0.8, 1.0)
    ax.set_ylabel("fraction of pre-loss tok/s")
    ax.legend(frameon=False, fontsize=13)
    style(ax)
    return fig


def slide_fate():
    fig = mock_slide("8. What happens to in-flight requests", "Completed / retryable error / failed per failure "
                     "type (GSM8K after a kill: appendix).")
    ax = fig.add_axes([0.1, 0.15, 0.8, 0.6])
    done, retry = [0.9] * len(ROWS), [0.1] * len(ROWS)
    ax.bar(range(len(ROWS)), done, color=GREEN, label="completed")
    ax.bar(range(len(ROWS)), retry, bottom=done, color="#E69F00", label="retryable error")
    ax.bar(range(len(ROWS)), [0] * len(ROWS), bottom=[1] * len(ROWS), color=RED, label="failed")
    ax.set_xticks(range(len(ROWS)), ROWS)
    ax.set_ylabel("share of in-flight requests")
    ax.legend(frameon=False, fontsize=13, ncol=3, loc="lower center")
    style(ax)
    return fig


def slide_downtime():
    fig = mock_slide("9. Downtime and GPU-hours", "FT recovery (seconds) vs warm restart (minutes) per mode, and the "
                     "failure rate where FT pays off.")
    ax = fig.add_axes([0.07, 0.15, 0.4, 0.6])
    for i, mode in enumerate(MODES):
        ax.barh(i * 2, 300, color=FT_OFF, label="FT off: warm restart" if i == 0 else None)
        ax.barh(i * 2 + 0.8, 1, color=FT_ON, label="FT on: recovery" if i == 0 else None)
    ax.set_yticks([0.4, 2.4, 4.4], MODES)
    ax.set_xscale("log")
    ax.set_xlabel("seconds until serving again (log)")
    ax.legend(frameon=False, fontsize=12)
    style(ax)
    ax = fig.add_axes([0.57, 0.15, 0.38, 0.6])
    rate = [0.1 * i for i in range(51)]
    ax.plot(rate, [r * 2 for r in rate], color=FT_OFF, label="GPU-h lost to restarts")
    ax.plot(rate, [3] * len(rate), color=FT_ON, label="GPU-h FT costs while healthy")
    ax.set_xlabel("failures per day")
    ax.set_ylabel("GPU-hours per day")
    ax.legend(frameon=False, fontsize=12)
    style(ax)
    return fig


def slide_design():
    fig = new_slide("10. Design", "Stop → agree → remap → restore → resume; late writes land in remapped slots.")
    image(fig, "fig1_architecture.png", [0.01, 0.46, 0.48, 0.38])
    image(fig, "fig4_recovery_sequence.png", [0.51, 0.46, 0.48, 0.38])
    image(fig, "fig5_placement.png", [0.01, 0.05, 0.48, 0.38])
    image(fig, "fig4_late_writes.png", [0.51, 0.05, 0.48, 0.38])
    return fig


def slide_limits():
    fig = new_slide("11. Limits", "What the MVP does not cover.")
    bullets(fig, ["A frozen rank is detected by the hang confirmation (30 s), not at once",
                  "A second loss inside the remap window ends the job",
                  "KV-cache events and the CFT all-to-all are unsupported",
                  "Untested failure types: power loss, NVLink fault",
                  "Disaggregated serving is out of scope"])
    return fig


def slide_gsm8k():
    fig = mock_slide("Appendix: GSM8K after a kill", "Strict accuracy, 1319 questions: FT off, FT on healthy, FT on "
                     "after a rank kill (n=2 each).")
    ax = fig.add_axes([0.2, 0.15, 0.6, 0.6])
    ax.bar(range(3), [0.9] * 3, yerr=0.01, color=[FT_OFF, FT_ON, BLUE], capsize=8)
    ax.set_xticks(range(3), ["FT off", "FT on", "FT on, after kill"])
    ax.set_ylim(0.8, 1.0)
    ax.set_ylabel("strict accuracy")
    style(ax)
    return fig


def build(out):
    os.makedirs(out, exist_ok=True)
    slides = [new_slide("WideEP Fault Tolerance in TensorRT-LLM")]
    slides[0].text(0.05, 0.62, "Keep serving when a GPU or node fails", fontsize=26, color=DARK)
    slides[0].text(0.05, 0.50, "Finals · DeepSeek-R1 NVFP4 · 32 GB200 GPUs", fontsize=18, color=GREY)
    fig = new_slide("Setup and method", "One method for every bar in the main figures.")
    bullets(fig, ["R1 NVFP4, DEP32 on 8 GB200 nodes, attention DP, overlap, MTP, CUDA graphs, 1k in / 1k out",
                  "3 modes: no EPLB, offline EPLB, online EPLB; FT off = main's path, FT on = same config + FT",
                  "Replication unit = 8-node group; FT off / on alternate; 95% CI over pairs",
                  "Primary: decode step cost and stall ms/step; end-to-end: client tok/s over 600 s windows",
                  "Failures: process kills and a frozen rank at a fixed decode step; both modes; n=4 per row"])
    slides += [fig, slide_headline(), slide_modes(), slide_decomposition(), slide_parity(), slide_timeline(),
               slide_resume(), slide_post_loss(), slide_fate(), slide_downtime(), slide_design(), slide_limits(),
               slide_modes((8, 512), "Appendix: extended loads (8 and 512)",
                           "Same method, 300 s windows, 4 pairs per mode."), slide_gsm8k()]
    with PdfPages(os.path.join(out, "deck.pdf")) as pdf:
        for i, fig in enumerate(slides, 1):
            fig.savefig(os.path.join(out, f"slide{i:02d}.png"))
            pdf.savefig(fig)
            plt.close(fig)
    print(f"wrote {len(slides)} slides to {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=os.path.join(FIG, "deck"))
    build(parser.parse_args().out)
