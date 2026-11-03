#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Design figures 10a-10d and the limits table (fig 11) for mvp/DESIGN_DOC_v5final_draft.md, as SVG + PNG.

Every box, arrow and label is backed by a code reference listed in the <name>.refs.md sidecar next to each figure.
Usage: python3 -I make_design_figures.py [OUT_DIR]   (default: this file's directory)
"""
import os
import pathlib
import sys
import tempfile

HERE = pathlib.Path(__file__).resolve().parent
TOOLS = HERE.parents[2] / "tools"
os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), f"mplconfig-{os.getuid()}"))
try:
    import matplotlib
except ImportError:
    sys.path.insert(0, str(TOOLS / "pylib"))
    import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle  # noqa: E402

# Dataviz reference palette (references/palette.md): chrome + categorical slots in fixed order + status critical.
SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
BLUE, ORANGE, AQUA, YELLOW, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#6250d6"
CRITICAL = "#d03b3b"
WASH = {BLUE: "#eaf2fc", ORANGE: "#fdefe9", AQUA: "#e7f7f1", YELLOW: "#fdf5e3", VIOLET: "#efedfb",
        CRITICAL: "#fbeaea", MUTED: "#f1f0ed"}

plt.rcParams.update({"font.family": "sans-serif", "font.size": 11, "figure.facecolor": SURFACE,
                     "savefig.facecolor": SURFACE, "text.color": INK})


def canvas(w=16, h=9):
    fig = plt.figure(figsize=(w, h))
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    ax.axis("off")
    return fig, ax


def box(ax, x, y, w, h, title=None, body=None, color=BLUE, fill=True, lw=1.4, title_size=11.5, body_size=9.8,
        dashed=False, align="left"):
    """A rounded box: a colored outline (identity), a light wash, ink text (text never wears the series color)."""
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.08", lw=lw,
                                edgecolor=color, facecolor=WASH.get(color, SURFACE) if fill else SURFACE,
                                linestyle=(0, (4, 2)) if dashed else "solid"))
    tx = x + 0.14 if align == "left" else x + w / 2
    ha = "left" if align == "left" else "center"
    if title:
        ax.text(tx, y + h - 0.12, title, ha=ha, va="top", fontsize=title_size, fontweight="bold", color=INK)
    if body:
        ax.text(tx, y + h - (0.46 if title else 0.12), body, ha=ha, va="top", fontsize=body_size, color=INK2,
                linespacing=1.35)


def arrow(ax, p, q, label=None, color=INK2, lw=1.4, style="-|>", rad=0.0, label_xy=None, size=9.5, ha="center",
          dashed=False):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle=style, mutation_scale=13, lw=lw, color=color,
                                 connectionstyle=f"arc3,rad={rad}", linestyle=(0, (4, 2)) if dashed else "solid",
                                 shrinkA=2, shrinkB=2))
    if label:
        lx, ly = label_xy if label_xy else ((p[0] + q[0]) / 2, (p[1] + q[1]) / 2)
        ax.text(lx, ly, label, ha=ha, va="center", fontsize=size, color=INK2,
                bbox=dict(facecolor=SURFACE, edgecolor="none", pad=1.2))


def title(ax, text, sub=None, w=16, h=9):
    ax.text(0.3, h - 0.25, text, ha="left", va="top", fontsize=15, fontweight="bold", color=INK)
    if sub:
        ax.text(0.3, h - 0.72, sub, ha="left", va="top", fontsize=10.5, color=INK2)


def save(fig, out, name):
    for ext in ("svg", "png"):
        fig.savefig(out / f"{name}.{ext}", dpi=150)
    plt.close(fig)
    print(f"rendered {out / name}.svg/.png")


# ------------------------------------------------------------------ 10a architecture


def fig10a(out):
    fig, ax = canvas()
    title(ax, "WideEP FT: components on every rank and the recovery authority",
          "One survivor shown in detail; every survivor runs the same. Code refs per box: fig10a_architecture.refs.md")

    # Authority
    box(ax, 4.2, 7.0, 7.6, 1.05, "Recovery authority (1 process, no GPU; ZMQ + HMAC to every rank)",
        "admits unions of registered failure domains, never rank 0  |  has rank 0 fence each failed rank before Propose\n"
        "Install only when every survivor's placement digest matches  |  Commit when all Installed  |  Abort at deadline",
        color=VIOLET, body_size=9.4)

    # Survivor rank
    box(ax, 0.3, 0.25, 10.6, 6.35, "Survivor rank r (attention DP + EP rank)", None, color=BLUE, fill=False, lw=1.8)
    ax.text(10.75, 6.42, "HOST (CPU)", fontsize=9, color=MUTED, fontweight="bold", ha="right", va="top")
    ax.text(1.45, 3.98, "DEVICE (GPU)", fontsize=9, color=MUTED, fontweight="bold", ha="left", va="top")
    ax.plot([0.45, 10.75], [4.05, 4.05], color=GRID, lw=1)

    box(ax, 0.5, 4.4, 2.9, 1.4, "PeerFailureWatch",
        "native thread, no GIL; polls a receive\nno peer matches every\npeer_failure_poll_s (0.02 s)",
        color=BLUE, body_size=9.2)
    box(ax, 3.65, 4.4, 3.3, 1.4, "PeerLossRecovery",
        "wraps the executor event loop; runs\n_recover on HostCollectiveInterrupted:\nshrink, plan, remap, restore, install",
        color=BLUE, body_size=9.2)
    box(ax, 7.2, 4.4, 3.5, 1.4, "Host collectives (ULFM Open MPI)",
        "pml ob1, btl self,tcp (TCP also in a node)\nShrink -> survivor communicator,\ninstalled at Install",
        color=BLUE, body_size=9.2)

    box(ax, 0.5, 1.75, 2.2, 1.65, "Abort record",
        "int32[2] (requested,\nstatus) in pinned host\nmemory; kernels poll it", color=ORANGE, body_size=9.2)
    box(ax, 2.9, 1.75, 2.2, 1.65, "Device rank mask",
        "64-bit words; kernels\nskip inactive ranks\n(ENABLE_RANK_MASK\nkernel template)", color=ORANGE, body_size=9.2)
    box(ax, 5.3, 1.75, 2.4, 1.65, "EPLB placement tables",
        "incl. expert_to_slot\n(static: fixed-address\nlookup, not route())", color=ORANGE, body_size=9.2)
    box(ax, 7.9, 1.75, 2.8, 1.65, "Expert slots per layer",
        "home slots + appended\nghost slots: one VMM range,\nalways mapped (fig 10c)", color=ORANGE, body_size=9.2)

    box(ax, 0.5, 0.45, 10.2, 0.8, None, None, color=AQUA)
    ax.text(0.65, 0.85, "MoE all-to-all kernels: NVLinkOneSided, fence path; dispatch/combine write peers' workspace "
            "rows over NVLink (fig 10d)", fontsize=10, va="center", color=INK)

    # Watch: abort word, then revoke
    arrow(ax, (1.2, 4.4), (1.2, 3.4), "store 1", label_xy=(1.62, 3.72), size=9)
    arrow(ax, (2.6, 5.8), (8.4, 5.8), "revoke", rad=-0.13, label_xy=(5.5, 6.33), size=9)
    # Recovery writes the tables
    for x in (1.9, 4.0, 6.5, 9.3):
        arrow(ax, (5.3, 4.4), (x, 3.4), None, color=BLUE)
    ax.text(10.7, 4.22, "edited in place at a recovery; nothing recaptured", fontsize=9, ha="right",
            va="center", color=INK2)
    for x in (1.6, 4.0, 6.5, 9.3):
        arrow(ax, (x, 1.75), (x, 1.25), None, color=MUTED, lw=1.1)

    # Authority <-> survivor
    arrow(ax, (5.0, 6.62), (5.0, 7.0), None, color=VIOLET)
    arrow(ax, (6.2, 7.0), (6.2, 6.62), None, color=VIOLET)
    ax.text(4.9, 6.81, "Register / Stopped / Ready / Installed", fontsize=9, ha="right", va="center", color=INK2)
    ax.text(6.3, 6.81, "Propose / Install / Commit / Abort  (Fence: to rank 0)", fontsize=9, ha="left", va="center",
            color=INK2)

    # Failed rank
    box(ax, 11.3, 4.3, 4.4, 2.3, "Failed rank f",
        "process ended (killed, crashed, OOM)\nor, with hang detection, a frozen rank\nfenced by its peers\n"
        "rank 0 ends it through PMIx before\nany Propose", color=CRITICAL, body_size=9.4)
    arrow(ax, (11.3, 5.2), (10.9, 5.2), "ULFM\nreports", label_xy=(11.1, 5.75), size=8.5, color=CRITICAL)
    ax.text(13.5, 3.95, "its in-flight NVLink writes may still\nland in survivors' workspace pages\n-> same-address "
            "remap (fig 10d)", ha="center", va="top", fontsize=9.4, color=INK2)

    # Host copies
    box(ax, 11.3, 0.45, 4.4, 2.55, "Host copies in /dev/shm (per node)",
        "with replicas (online EPLB, or > 1 slot per\nexpert): every node keeps every expert\n"
        "without replicas: ring, each node keeps the\nnext node's home experts\n"
        "a restore never reads the lost GPU", color=YELLOW, body_size=9.2)
    arrow(ax, (11.3, 2.3), (10.7, 2.5), "restore", label_xy=(11.0, 2.75), size=9, color=INK2)

    save(fig, out, "fig10a_architecture")


# ------------------------------------------------------------------ 10b recovery sequence

STEPS = [
    # (phase line, step text, where, cost class, authority side)
    ("detect", "Abort: watch stores the abort word, revokes host comms; kernels\nwaiting on the lost peer return cancelled",
     "COM:850-865; PLR:636-662, 950-959", "every", None),
    (None, "Agree: ULFM Shrink, the same failed set on every survivor;\ncheck it is an admitted domain or a covered union",
     "PLR:962-967, 1198-1261; COM:1009-1033", "every", None),
    (None, "Stop: join device work, discard uncommitted steps (KV and\nrequests reverted); online EPLB: wait for in-flight updates",
     "PLR:969-987", "every", None),
    ("stopped", "Send Stopped(failed set)", "PLR:1008-1013", "every", "all Stopped -> rank 0 fences each failed rank"),
    (None, "Ghost backing: (online) rank 0's placement broadcast; plan;\nmap ghost memory from free memory or the KV cache",
     "PLR:1016-1017, 1323-1376", "overlap", "-> Propose(generation, failed, survivors)"),
    ("proposed", "Check the proposal matches this rank's view", "PLR:1018-1029", "every", None),
    (None, "Remap the all-to-all workspaces at the same addresses (fig 10d)", "PLR:1030, 1378-1424", "every", None),
    (None, "Ring staging: copy restored experts from ring host copies\nthrough the fresh staging pages",
     "PLR:1031-1035, 1459-1482", "ring", None),
    ("ready", "Send Ready(placement digest)", "PLR:1036-1039", "every", "all digests match -> Install"),
    ("install", "Install: survivor communicator; load restored slots, switch\nrouting; mark failed; rearm; install mask",
     "PLR:1052-1056, 1748-1783", "every", None),
    (None, "Online EPLB: restart the update rotation (20 warm-up iterations)", "PLR:98, 1065-1068", "online", None),
    ("installed", "Outcome exchange (allgather): served ids, iteration counter,\ninterrupted / progressed; rank 0 quarantines; "
     "send Installed", "PLR:1069-1093", "every", "all Installed -> Commit"),
    ("committed", "Return KV lent to the remap (as far as the memory is free)", "PLR:1097, 1439-1457", "every", None),
    ("resumed", "Resume: requeue, deliver owed responses, re-run or fail the lost\nranks' requests, re-admit; restart the watch",
     "PLR:1099-1108", "every", None),
    ("first_step", "First committed step after the recovery (rank 0)", "PLR:739-741", "every", None),
]
COST = {"every": (BLUE, "on the critical path of every recovery"),
        "overlap": (AQUA, "every recovery, overlaps the authority's gather of Stopped"),
        "online": (ORANGE, "online EPLB only"),
        "ring": (YELLOW, "placements without replicas (ring host copies) only")}


def fig10b(out):
    w, h = 16, 10.8
    fig, ax = canvas(w, h)
    title(ax, "Recovery sequence on every survivor (not to scale)",
          "Left: the rank-0 log `phase=` line written when the step is reached (PLR:1125-1130; fig 6 times these). "
          "Color: when the step costs time. Refs: fig10b_recovery_sequence.refs.md", w, h)
    top, row = h - 1.4, 0.56
    xl, xs, ws, xa = 0.3, 1.75, 9.0, 11.55
    for x, lab in ((xl, "phase line"), (xs, "step (every survivor)"), (xs + ws - 0.1, "code"),
                   (xa, "recovery authority")):
        ax.text(x, top + 0.05, lab, fontsize=9.5, color=MUTED, fontweight="bold", va="bottom",
                ha="right" if lab == "code" else "left")
    ys = []
    for i, (phase, text, where, cost, auth) in enumerate(STEPS):
        y = top - (i + 1) * row
        ys.append(y)
        color = COST[cost][0]
        ax.add_patch(Rectangle((xs, y + 0.04), 0.11, row - 0.08, color=color, lw=0))
        ax.add_patch(Rectangle((xs + 0.13, y + 0.04), ws - 0.13, row - 0.08, facecolor=WASH[color], lw=0))
        ax.text(xs + 0.25, y + row / 2, text, va="center", fontsize=9.6, color=INK, linespacing=1.2)
        ax.text(xs + ws - 0.1, y + row / 2, where, va="center", ha="right", fontsize=8.6, color=INK2)
        if phase:
            ax.text(xl, y + row / 2, phase, va="center", fontsize=9.5, color=INK, family="monospace")
    mid = [y + row / 2 for y in ys]
    end = xs + ws
    # Messages: survivor -> authority on the row that sends, authority -> survivor on the row that receives.
    for send, recv, out_lab, in_lab, note in (
            (3, 5, "Stopped", "Propose", "once all Stopped: rank 0 ends (fences) each\nfailed rank through PMIx, then Propose"),
            (8, 9, "Ready", "Install", "once every placement digest matches: Install"),
            (11, 12, "Installed", "Commit", "once all Installed: Commit")):
        arrow(ax, (end + 0.05, mid[send]), (xa - 0.05, mid[send]), out_lab, color=VIOLET, lw=1.2,
              label_xy=(end + 0.75, mid[send] + 0.15), size=8.8)
        arrow(ax, (xa - 0.05, mid[recv]), (end + 0.05, mid[recv]), in_lab, color=VIOLET, lw=1.2,
              label_xy=(end + 0.75, mid[recv] + 0.15), size=8.8)
        ax.text(xa, (mid[send] + mid[recv]) / 2, note, va="center", fontsize=9.2, color=INK, linespacing=1.25)
    ax.plot([xa - 0.05, xa - 0.05], [ys[-1], top], color=VIOLET, lw=1, alpha=0.5)

    def bracket(x, y0, y1, label, color):
        ax.plot([x, x - 0.08, x - 0.08, x], [y1, y1, y0, y0], color=color, lw=1.4)
        ax.text(x - 0.15, (y0 + y1) / 2, label, rotation=90, ha="right", va="center", fontsize=8.8, color=INK2)

    bracket(1.62, ys[8] + 0.04, ys[3] + row - 0.04, "re-plan window", CRITICAL)
    ax.text(xa, (mid[6] + mid[7]) / 2 + 0.05, "a further loss from Stopped until Install:\nRecoveryInterrupted -> shrink again, release\n"
            "the abandoned plan's ghost backing and KV,\nplan again (PLR:1043-1046, 1263-1302;\nAUT:534-555)",
            va="center", fontsize=8.8, color=INK2, linespacing=1.25)
    ax.text(xa, mid[13], "after Install a further loss ends the job", va="center", fontsize=8.8, color=INK2)
    ax.text(xs, ys[-1] - 0.12, "Deadlines: D1 = agreement + recovery_timeout_s bounds the stop (PLR:966). D2 = first "
            "Stopped + recovery_timeout_s bounds the rest (PLR:1012-1013); the authority\naborts at first Stopped + "
            "authority_timeout_s, which is shorter (AUT:473, 282-285). Healthy-path costs (rank mask, abort polling, "
            "step commit) are not in this figure.", fontsize=8.8, color=INK2, va="top", linespacing=1.3)
    lx = xs
    for key in ("every", "overlap", "online", "ring"):
        color, lab = COST[key]
        ax.add_patch(Rectangle((lx, 0.18), 0.25, 0.18, color=color, lw=0))
        ax.text(lx + 0.33, 0.27, lab, va="center", fontsize=9, color=INK)
        lx += 0.6 + 0.071 * len(lab)
    save(fig, out, "fig10b_recovery_sequence")


# ------------------------------------------------------------------ 10c ghost slots


def slot_row(ax, x0, y, slot_w, home, ghost, label_ghost_state, ghost_color):
    for i in range(home + ghost):
        is_ghost = i >= home
        color = ghost_color if is_ghost else BLUE
        ax.add_patch(Rectangle((x0 + i * slot_w + 0.02, y), slot_w - 0.04, 0.55, facecolor=WASH[color],
                               edgecolor=color, lw=1.3, linestyle=(0, (4, 2)) if is_ghost and ghost_color == MUTED
                               else "solid"))
        ax.text(x0 + (i + 0.5) * slot_w, y + 0.275, f"g{i - home}" if is_ghost else f"s{i}", ha="center",
                va="center", fontsize=9.5, color=INK)


def chunk_row(ax, x0, y, slot_w, home, ghost, n, backed):
    """``n`` VMM chunks under the slots: chunks wholly inside the ghost slots form the ghost piece."""
    start_ghost, end = x0 + home * slot_w, x0 + (home + ghost) * slot_w
    chunk_w = (end - x0) / n
    first_in = None
    for k in range(n):
        cx = x0 + k * chunk_w
        inside = cx >= start_ghost - 1e-6
        if inside and first_in is None:
            first_in = cx
        color = (ORANGE if backed else MUTED) if inside else BLUE
        ax.add_patch(Rectangle((cx + 0.01, y), chunk_w - 0.02, 0.32, facecolor=WASH[color], edgecolor=color, lw=1))
    return first_in, end, chunk_w


def fig10c(out):
    fig, ax = canvas()
    title(ax, "Ghost slots: spare expert slots that hold no device memory until a restore",
          "One layer's slot tensor on one rank (sizes not to scale). Example DeepSeek-R1 DEP32, 4 GPUs/node: "
          "8 home + 2 ghost slots without EPLB, 9 + 1 with 288 EPLB slots. Refs: fig10c_ghost_slots.refs.md")
    home, ghost, slot_w, x0, n = 8, 2, 1.05, 0.9, 16
    for y, head, backed, gcolor in ((5.3, "While healthy", False, MUTED),
                                    (1.75, "At a restore (before Ready)", True, ORANGE)):
        ax.text(0.3, y + 2.35, head, fontsize=13, fontweight="bold", color=INK)
        ax.text(x0 - 0.1, y + 1.33, "slots", ha="right", va="center", fontsize=9.5, color=MUTED)
        slot_row(ax, x0, y + 1.05, slot_w, home, ghost, None, gcolor)
        ax.text(x0 - 0.1, y + 0.56, "VMM\nchunks", ha="right", va="center", fontsize=9.5, color=MUTED)
        first_in, ge, cw = chunk_row(ax, x0, y + 0.4, slot_w, home, ghost, n, backed)
        ax.annotate("", (x0, y + 0.2), (ge, y + 0.2), arrowprops=dict(arrowstyle="<->", color=INK2, lw=1))
        ax.text(5.2, y + 0.05, "one reserved virtual address range, always fully mapped "
                "(the CUDA graphs keep its address)", ha="center", va="top", fontsize=9.5, color=INK2)
        # the chunk a ghost slot shares with a home slot keeps its own memory
        ax.annotate("chunk shared with s7:\nown device memory", (first_in - cw / 2, y + 0.4),
                    (first_in - cw / 2 + 1.3, y - 0.45), fontsize=9, color=INK2, ha="left",
                    arrowprops=dict(arrowstyle="-", color=INK2, lw=0.9))
        ax.text(x0 + home * slot_w, y + 1.82,
                "routing table: -1 at the ghost slots (nothing routes there)" if not backed else
                "the plan loads restored experts here; Install switches routing to them",
                fontsize=9.6, color=INK2, ha="left" if not backed else "center", va="bottom")
        if not backed:
            box(ax, 12.3, y + 0.15, 3.4, 1.65, "Per-process scratch",
                "one allocation each of 1, 2, 4 and 8\nchunks, shared by every range: a\nstray access reads "
                "scratch, never faults", color=MUTED, body_size=9.2)
            arrow(ax, (ge - 0.2, y + 0.56), (12.3, y + 0.9), "ghost piece\nmaps", label_xy=(11.85, y + 0.22), size=9)
        else:
            box(ax, 12.3, y + 0.15, 3.4, 1.65, "Device memory",
                "free memory first (256 MiB margin),\nthe shortfall from the KV cache\nquota; if the KV cache cannot "
                "\nshrink, this rank fails before Ready", color=ORANGE, body_size=9.2)
            arrow(ax, (ge - 0.2, y + 0.56), (12.3, y + 0.9), "back()\nmaps", label_xy=(11.85, y + 0.22), size=9)
    lx, ly = 0.9, 0.55
    for color, lab, dx, dy in ((BLUE, "chunk with its own device memory, mapped at allocation", 0, 0),
                               (MUTED, "ghost piece on scratch", 0, -0.38),
                               (ORANGE, "ghost piece backed, pending for the plan: Install keeps it, an abandoned "
                                "plan returns it to scratch", 3.6, -0.38)):
        ax.add_patch(Rectangle((lx + dx, ly + dy), 0.35, 0.22, facecolor=WASH[color], edgecolor=color))
        ax.text(lx + dx + 0.45, ly + dy + 0.11, lab, va="center", fontsize=9.3, color=INK)
    save(fig, out, "fig10c_ghost_slots")


# ------------------------------------------------------------------ 10d same-address remap


def ws_panel(ax, x, y, head, sub, rows_state, page_state, failed_write, note):
    """One survivor's view of a workspace: the VA rows (one per EP rank) and the pages behind its own row."""
    ax.text(x, y + 4.6, head, fontsize=12.5, fontweight="bold", color=INK)
    ax.text(x, y + 4.2, sub, fontsize=9.4, color=INK2, va="top", linespacing=1.3)
    labels = ["row r0", "row r1 (this rank's own)", "row f (failed)", "row r3"]
    for i, lab in enumerate(labels):
        state = rows_state[i]
        ry = y + 2.7 - i * 0.5
        color = {"mapped": BLUE, "unmapped": MUTED, "fresh": AQUA}[state]
        ax.add_patch(Rectangle((x, ry), 2.1, 0.4, facecolor=WASH[color], edgecolor=color, lw=1.3,
                               linestyle=(0, (4, 2)) if state == "unmapped" else "solid"))
        ax.text(x + 1.05, ry + 0.2, lab, ha="center", va="center", fontsize=9.2, color=INK)
    ax.text(x, y + 3.25, "same virtual addresses", fontsize=8.8, color=MUTED)
    # pages
    old_c = {"live": BLUE, "orphan": CRITICAL}[page_state[0]]
    ax.add_patch(Rectangle((x + 2.6, y + 1.7), 1.25, 0.9, facecolor=WASH[old_c], edgecolor=old_c, lw=1.3))
    ax.text(x + 3.22, y + 2.15, "old pages", ha="center", va="center", fontsize=9.2, color=INK)
    if page_state[1]:
        ax.add_patch(Rectangle((x + 2.6, y + 0.55), 1.25, 0.9, facecolor=WASH[AQUA], edgecolor=AQUA, lw=1.3))
        ax.text(x + 3.22, y + 1.0, "fresh pages", ha="center", va="center", fontsize=9.2, color=INK)
    # own row (r1) -> the pages behind it
    ax.plot([x + 2.1, x + 2.6], [y + 2.4, y + 2.15], color=INK2, lw=1) if page_state[0] == "live" else None
    if page_state[1]:
        ax.plot([x + 2.1, x + 2.6], [y + 2.4, y + 1.0], color=INK2, lw=1)
    if failed_write:
        arrow(ax, (x + 4.5, y + 2.15), (x + 3.87, y + 2.15), None, color=CRITICAL)
        ax.text(x + 4.2, y + 1.85, "f's late\nNVLink write", ha="center", va="top", fontsize=8.6, color=INK2)
    ax.text(x, y + 0.35, note, fontsize=9.2, color=INK2, va="top", linespacing=1.3)


def fig10d(out):
    fig, ax = canvas()
    title(ax, "Late writes from a failed rank are contained by the same-address remap",
          "A failed rank's kernels may still write into survivors' workspace pages it mapped; nothing bounds when. "
          "FT never reuses those pages. Refs: fig10d_late_write_remap.refs.md")
    y = 2.9
    ws_panel(ax, 0.4, y, "1. Before the loss", "every rank maps every rank's row; the\npages behind r1's own row are imported\nby every rank, incl. f", ["mapped"] * 4, ("live", False), False,
             "dispatch/combine of every rank\nwrite into these rows over NVLink")
    ws_panel(ax, 5.6, y, "2. release_workspaces", "mask clears f first; then each survivor\nunmaps every row and releases "
             "its handles,\nkeeping the addresses", ["unmapped"] * 4, ("orphan", False), True,
             "pages stay allocated while any importer\n(incl. f's process) maps them: a late write\n"
             "lands in pages no survivor maps again")
    ws_panel(ax, 10.8, y, "3. remap", "barrier; room from the KV cache if need be;\nfresh pages mapped at the same "
             "addresses,\nexchanged only among the live ranks", ["fresh", "fresh", "unmapped", "fresh"],
             ("orphan", True), True,
             "all live ranks map or all raise; rows reset,\nbarrier before any peer writes (ring staging);\n"
             "graphs replay on the fresh pages; the\ninstalled mask keeps collectives off row f")
    arrow(ax, (4.75, y + 2.4), (5.45, y + 2.4), None, color=INK2, lw=1.6)
    arrow(ax, (9.95, y + 2.4), (10.65, y + 2.4), None, color=INK2, lw=1.6)
    ax.text(0.4, 1.45, "Afterwards: the old pages are freed only when f's process ends, so the remap may borrow KV "
            "cache; it is returned at the commit if the memory is free by then,\nelse at an idle iteration (at most "
            "once a second) or a later recovery's commit. A survivor that cannot release or map its pages ends the "
            "job; a peer\nfailure during the remap exchanges re-plans.", fontsize=9.6, color=INK2, va="top",
            linespacing=1.35)
    lx = 0.4
    for color, lab, dashed in ((BLUE, "mapped", False), (MUTED, "unmapped (address kept)", True),
                               (AQUA, "mapped to fresh pages", False), (CRITICAL, "old pages: no survivor maps them", False)):
        ax.add_patch(Rectangle((lx, 0.2), 0.35, 0.22, facecolor=WASH[color], edgecolor=color,
                               linestyle=(0, (4, 2)) if dashed else "solid"))
        ax.text(lx + 0.45, 0.31, lab, va="center", fontsize=9.3, color=INK)
        lx += 0.9 + 0.075 * len(lab)
    save(fig, out, "fig10d_late_write_remap")


# ------------------------------------------------------------------ 11 limits table

LIMITS = [
    ("Detection", "A frozen rank is fenced only with hang detection, after about 35 s; two ranks hung at once are "
     "not confirmed. Without it, a stopped rank holds the shrink until the MoE device timeout or the 300 s executor "
     "hang detector ends the job.",
     "by design (off by default)"),
    ("Second loss", "After Install a further loss ends the job; so does a Commit lost on the way to a survivor "
     "(survivors wait to D2). Before Install, a further loss re-plans.", "by design"),
    ("Not recoverable", "Rank 0 or its node; a node lost together with its ring copy node; a loss the remaining "
     "slots cannot cover (DeepSeek-R1 DEP32: a second node).", "by design"),
    ("Out of FT's reach", "Host death, GPU device loss and NVLink fabric faults end the job as without FT (depends "
     "on PRRTE); silent wrong results are not detected.", "by design"),
    ("No rejoin", "The deployment runs on N - k ranks until it is restarted; a lost rank does not rejoin.",
     "next phase"),
    ("Memory", "A survivor that cannot map its ghost slots or remap its workspaces ends the job. KV lent to restored "
     "experts stays until restart; KV lent to the remap can stay lent until a later commit.", "known"),
    ("Spec. decoding", "With the overlap scheduler, a loss right after a commit can skip that step's KV update for "
     "rejected draft tokens (no wrong output). Other than one-model linear MTP: accepted with a warning.", "known"),
    ("Shutdown", "Under ULFM, MPI_Finalize can hang; each process gives it 30 s, then exits with its status.",
     "known"),
    ("Platform", "NUMA-memory GPUs with online EPLB fall back to extra_slots: empty (memory held permanently).",
     "known"),
    ("Reload", "Host shared-memory mappings leak on each FT-on weight reload.", "follow-up"),
    ("Scope (refused)", "Pure attention DP + EP, expert TP 1; NVLinkOneSided fence path (CFT refused); KV Cache "
     "Manager V2 without transceiver (no disaggregated serving), KV cache events or connector; ULFM Open MPI "
     "(pml ob1: host collectives over TCP).", "startup refuses"),
]


def fig11(out):
    w, h = 16, 9
    fig, ax = canvas(w, h)
    title(ax, "Known limits and requirements of WideEP FT (Sections 11 and 12)",
          "Refs: fig11_limits.refs.md", w, h)
    import textwrap
    x0, x1, x2, x3 = 0.3, 2.55, 13.2, 15.7
    y = h - 1.25
    for x, lab in ((x0, "area"), (x1, "limit"), (x2, "status")):
        ax.text(x + 0.05, y, lab, fontsize=10, fontweight="bold", color=MUTED, va="bottom")
    y -= 0.08
    ax.plot([x0, x3], [y, y], color=AXIS, lw=1)
    for area, text, status in LIMITS:
        lines = textwrap.wrap(text, 140)
        rh = 0.24 * len(lines) + 0.22
        ax.text(x0 + 0.05, y - 0.13, area, fontsize=10.3, fontweight="bold", color=INK, va="top")
        ax.text(x1 + 0.05, y - 0.13, "\n".join(lines), fontsize=10, color=INK, va="top", linespacing=1.25)
        ax.text(x2 + 0.05, y - 0.13, status, fontsize=9.8, color=INK2, va="top")
        y -= rh
        ax.plot([x0, x3], [y, y], color=GRID, lw=0.8)
    ax.text(x0, y - 0.2, "Validated: [FINALS: test matrix, runs and n]. Untested failure types (finals plan): power "
            "loss, NVLink fault.", fontsize=9.6, color=INK2, va="top")
    save(fig, out, "fig11_limits")


def main():
    out = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else HERE
    out.mkdir(parents=True, exist_ok=True)
    for make in (fig10a, fig10b, fig10c, fig10d, fig11):
        make(out)


if __name__ == "__main__":
    main()
