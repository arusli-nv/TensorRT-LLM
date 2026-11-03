"""Design-doc figures for the final design (v5-final 27442bd91d), built on make_arch's SVG helpers.

A copy of mvp/analysis/fig_design.py (Oct 20), same style, content updated to
review/DESIGN_DOC_v5final_tonight.md. Captions live in the document.
Usage: FONTCONFIG_FILE=<conf with Lato> PYTHONPATH=<cairosvg> python3 fig_design_final.py <out dir>
"""

import re

import numpy as np

from make_arch import BLUE, BLUE_FILL, GRAY, GRAY_FILL, GREEN, GREEN_FILL, INK, RED, RED_FILL, Svg

SUB = "#4A4A4A"
SLATE, SLATE_SUB, CANVAS = "#2D3142", "#4F5D75", "#FFFFFF"
KIND = {"plain": ("white", "#6B7A89"), "new": (GREEN_FILL, GREEN), "old": (GRAY_FILL, GRAY), "blue": (BLUE_FILL, BLUE),
        "red": (RED_FILL, RED), "slate": ("white", SLATE)}


DARK_GREEN = "#1B4F12"
# Figures that mark FT's parts without colour too: a 1.5x outline (grayscale, colour-blind readers).
FT_CUE = {"on": False}
# Dashed message arrows and zoom lines: 3 px on the 2056 px export, mid-grey.
MID, THIN = "#6B6B6B", 1.5
MID_MARKER = (f'<defs><marker id="mid" markerWidth="10" markerHeight="10" refX="8" refY="5" orient="auto-start-reverse" '
              f'markerUnits="userSpaceOnUse"><path d="M0,0 L10,5 L0,10 z" fill="{MID}"/></marker></defs>')


def dashed_line(s, x1, y1, x2, y2, arrow=True):
    end = ' marker-end="url(#mid)"' if arrow else ""
    s.out.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{MID}" stroke-width="{THIN}" '
                 f'stroke-dasharray="6 5"{end}/>')


def ft_cue(kind):
    return FT_CUE["on"] and kind == "new"


def box(s, x, y, w, h, title, sub=None, kind="new", size=21, dash=None):
    fill, stroke = KIND[kind]
    s.rect(x, y, w, h, fill, stroke, 3 if ft_cue(kind) else 2, 8, dash=dash)
    cy = y + h / 2 + (-4 if sub else 7)
    s.text(x + w / 2, cy, title, size, INK, weight="bold")
    if sub:
        s.text(x + w / 2, cy + 22, sub, 16, SUB)


def box3(s, x, y, w, h, title, lines, kind="new", size=18):
    """A box with a title and two lines of detail."""
    fill, stroke = KIND[kind]
    s.rect(x, y, w, h, fill, stroke, 2, 8)
    s.text(x + w / 2, y + 27, title, size, INK, weight="bold")
    for i, t in enumerate(lines):
        s.text(x + w / 2, y + 49 + i * 19, t, 16, SUB)


def label(s, x, y, t, anchor="start", size=17, color=INK):
    s.text(x, y, t, size, color, anchor=anchor)


def slot(s, x, y, w, h, kind, t=None, size=17):
    if kind == "ghost":
        s.out.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="6" fill="url(#hatch)" stroke="{GRAY}" '
                     f'stroke-width="1.8" stroke-dasharray="5 4"/>')
    else:
        fill, stroke = {"expert": (BLUE_FILL, BLUE), "restored": (GREEN_FILL, GREEN)}[kind]
        s.rect(x, y, w, h, fill, stroke, 3.3 if kind == "restored" else 1.8, 6)
    if t:
        s.text(x + w / 2, y + h / 2 + 6, t, size, INK, weight="bold" if kind == "restored" else "normal")


def cyl(s, x, y, w, h, title, sub=None, kind="old", size=19):
    """A memory or data object: a stack of two sheets with square corners."""
    fill, stroke = KIND[kind]
    d = 5 if w > 40 else 3
    k = 1.5 if ft_cue(kind) else 1
    s.rect(x + d, y - d, w, h, fill, stroke, 1.6 * k, 0)
    s.rect(x, y, w, h, fill, stroke, 2 * k, 0)
    if not title:
        return
    cy = y + h / 2 + (-4 if sub else 7)
    s.text(x + w / 2, cy, title, size, INK, weight="bold")
    if sub:
        s.text(x + w / 2, cy + 22, sub, 16, SUB)


def ft_swatch(s, x, y):
    """Legend swatch for FT's parts: the fill and the 1.5x outline."""
    s.rect(x, y, 22, 16, GREEN_FILL, GREEN, 2.7, 4)


def legend(s, y, ghost=True):
    ft_swatch(s, 40, y)
    label(s, 70, y + 14, "added or changed by FT")
    s.rect(285, y, 22, 16, GRAY_FILL, GRAY, 1.8, 4)
    label(s, 315, y + 14, "existing")
    s.rect(410, y - 2, 26, 20, "white", INK, 1.6, 7)
    label(s, 444, y + 14, "process or kernel")
    cyl(s, 600, y - 2, 26, 20, "", None, "old")
    label(s, 636, y + 14, "memory")
    if ghost:
        slot(s, 730, y - 4, 20, 24, "ghost")
        label(s, 758, y + 14, "unbacked spare slot")


def architecture():
    FT_CUE["on"] = True
    s = Svg(1000, 685)
    for y, h, name in ((20, 132, "CPU"), (162, 473, "GPU")):
        s.rect(20, y, 960, h, "#FAFAFA", "#E2E2E2", 1.5, 12)
        s.text(32, y + 28, name, 18, INK, anchor="start", weight="bold")
    box(s, 44, 56, 196, 64, "Failure detection", "watch + hang reporter", "new", 18)
    box(s, 322, 56, 186, 64, "Recovery authority", "separate process", "new", 17)
    box(s, 590, 56, 200, 64, "Recovery hook", "discard, replay, restore", "new", 18)
    cyl(s, 814, 61, 142, 56, "Host copies", "of expert weights", "old", 18)
    s.arrow(240, 88, 318, 88)
    label(s, 279, 80, "heartbeats", "middle", 14)
    s.arrow(508, 80, 586, 80)
    label(s, 547, 72, "directs", "middle", 14)
    s.arrow(586, 100, 512, 100)
    label(s, 547, 116, "reports", "middle", 14)
    # The watcher tells the executor's recovery hook that a peer is gone.
    s.path("M 210 120 L 210 140 L 640 140 L 640 124", "green")
    label(s, 415, 136, "notifies", "middle", 14)
    # What the kernels read: FT's descriptor and EPLB's routing table.
    cyl(s, 60, 234, 330, 56, "Execution state", "live-rank mask, abort flag", "new")
    cyl(s, 440, 234, 340, 56, "EPLB routing table", "expert to slot")
    s.arrow(160, 120, 160, 226, "green")
    label(s, 171, 210, "sets abort flag", size=16)
    s.path("M 610 120 L 610 188 L 330 188 L 330 226", "green")
    label(s, 470, 180, "writes mask", "middle", 16)
    s.arrow(750, 120, 750, 226, "green")
    label(s, 640, 210, "writes routing", size=16)
    # The captured MoE layer reads both on every launch.
    s.rect(60, 330, 720, 285, "white", "#BDBDBD", 1.6, 12, dash="7 5")
    s.text(74, 605, "CUDA graph", 17, INK, anchor="start", weight="bold")
    box(s, 80, 410, 130, 52, "Dispatch", None, "new", 18)
    s.rect(240, 350, 380, 160, GRAY_FILL, GRAY, 2, 8)
    s.text(430, 376, "Experts", 19, INK, weight="bold")
    for i in range(10):
        slot(s, 252 + i * 36, 392, 30, 96, "ghost" if i >= 8 else "expert")
    box(s, 650, 410, 115, 52, "Combine", None, "new", 18)
    s.arrow(210, 436, 236, 436)
    s.arrow(620, 436, 646, 436)
    s.arrow(145, 290, 145, 326)
    s.arrow(520, 290, 520, 346)
    # The all-to-all buffers keep their addresses; a recovery maps fresh memory behind them.
    cyl(s, 80, 532, 440, 52, "All-to-all workspace", "remapped on a loss", "new", 18)
    s.arrow(145, 462, 145, 524, both=True)
    # Restoring a lost expert into a ghost slot: weights from the host copies, memory from the KV cache.
    s.path("M 882 117 L 882 312 L 591 312 L 591 388", "green")
    label(s, 892, 252, "copies", size=16)
    label(s, 892, 272, "weights", size=16)
    cyl(s, 810, 420, 145, 50, "KV cache")
    s.path("M 882 470 L 882 522 L 591 522 L 591 492", "green")
    label(s, 892, 494, "lends", size=16)
    label(s, 892, 514, "memory", size=16)
    label(s, 892, 534, "if needed", size=16)
    legend(s, 655)
    FT_CUE["on"] = False
    return s.render()


def band(s, y, h, name, x=20, w=960, color=INK):
    s.rect(x, y, w, h, "#FAFAFA", "#E2E2E2", 1.5, 12)
    s.text(x + 14, y + 28, name, 18, color, anchor="start", weight="bold")


def ghost_slots():
    s = Svg(1000, 470)
    band(s, 16, 244, "GPU 0 (restores 1 of rank 2's 8 lost experts)")
    band(s, 274, 180, "")
    s.text(500, 302, "Host copies of expert weights (Grace memory)", 18, INK, weight="bold")
    x0, sw, gap = 170, 74, 6
    s.text(40, 82, "Expert slots", 17, INK, anchor="start", weight="bold")
    s.text(40, 104, "(virtual)", 16, INK, anchor="start")
    s.text(40, 190, "GPU memory", 17, INK, anchor="start", weight="bold")
    s.text(40, 212, "(physical)", 16, INK, anchor="start")
    xs = [x0 + i * (sw + gap) for i in range(10)]
    for i, x in enumerate(xs):
        if i < 8:
            slot(s, x, 62, sw, 50, "expert", f"e{i}", 18)
            s.rect(x, 140, sw, 44, BLUE_FILL, BLUE, 1.6, 4)
            s.arrow(x + sw / 2, 114, x + sw / 2, 136, "gray")
        elif i == 8:
            slot(s, x, 62, sw, 50, "restored", None)
            s.text(x + sw / 2, 84, "e16", 17, INK, weight="bold")
            s.text(x + sw / 2, 104, "from rank 2", 13, INK)
        else:
            slot(s, x, 62, sw, 50, "ghost")
            s.text(x + sw / 2, 132, "maps shared", 14, INK)
            s.text(x + sw / 2, 150, "scratch", 14, INK)
    kx = xs[8]
    # Physical memory: the KV cache, then free GPU memory. A restore maps free memory first and borrows from the KV
    # cache only if free memory is short (none was borrowed in any logged run).
    ky, kh = 194, 56
    fx = kx - 6
    s.rect(x0, ky, fx - x0, kh, GRAY_FILL, GRAY, 1.6, 4)
    s.text(x0 + 300, ky + 34, "KV cache", 18, INK, weight="bold")
    s.rect(fx, ky, xs[9] + sw - fx, kh, "white", GRAY, 1.6, 4)
    s.text(xs[9] + sw / 2, ky + 25, "free", 14, INK)
    s.text(xs[9] + sw / 2, ky + 43, "memory", 14, INK)
    s.rect(kx + 5, ky + 5, sw - 10, kh - 10, GREEN_FILL, GREEN, 3.6, 3)
    s.arrow(kx + sw / 2, 114, kx + sw / 2, ky + 1, "green")
    label(s, kx + sw / 2 + 8, 174, "CUDA VMM", "start", 14)

    # Without online EPLB, each node's Grace memory keeps the NEXT node's experts (a ring); node 7 keeps node 0's.
    ny, nh, nw = 316, 118, 266
    nodes = ((40, 0, "e32\u2013e63", 1), (326, 1, "e64\u2013e95", 2), (672, 7, "e0\u2013e31", 0))
    for nx, k, held, owner in nodes:
        s.rect(nx, ny, nw, nh, "white", GRAY, 1.8, 10)
        s.text(nx + 12, ny + 24, f"Node {k}", 16, INK, anchor="start", weight="bold")
        for j in range(4):
            r = 4 * k + j
            dead = r == 2
            x = nx + 8 + j * 64
            s.rect(x, ny + 34, 58, 28, RED_FILL if dead else GRAY_FILL, RED if dead else GRAY, 1.8, 4,
                   dash="4 2" if dead else None)
            s.text(x + 29, ny + 53, f"GPU {r}", 13, INK, weight="bold")
        cyl(s, nx + 8, ny + 78, 244, 28, f"Grace memory: Node {owner}'s {held}", None, "new" if k == 7 else "old", 14)
    s.text(632, ny + nh / 2 + 8, "...", 26, INK, weight="bold")
    # GPU 0 of node 0 and the top panel share a dark outline: the panel is that GPU.
    s.rect(48, ny + 34, 58, 28, "white", SLATE, 2.4, 4)
    s.text(77, ny + 53, "GPU 0", 13, INK, weight="bold")
    s.rect(20, 16, 960, 244, "none", SLATE, 2.4, 12)
    for xa, xb in ((48, 20), (106, 380)):
        dashed_line(s, xa, ny + 34, xb, 260, arrow=False)
    # Rank 2's e16 is restored from node 7's copy, over NVLink, into rank 0's ghost slot.
    cx = 672 + nw / 2
    s.path(f"M {cx} {ny} L {cx} 266 L {kx + sw / 2} 266 L {kx + sw / 2} {ky + kh - 1}", "green")
    label(s, cx + 10, 304, "e16 over NVLink", "start", 15)
    return s.render()


def late_writes(per_sender_rows=False):
    """The MoE all-to-all of one layer, and a zoom on the part FT changes: the token rows in the receive workspace.

    per_sender_rows draws the Oct 20 design (Figure G1) instead of the remap."""
    comm = "old" if per_sender_rows else "new"
    FT_CUE["on"] = True
    s = Svg(1000, 458)
    band(s, 16, 150, "One MoE layer on one GPU (captured in the CUDA graph)")
    y, h = 70, 60
    stages = ((30, 116, "Router", "top 8 experts", "old"), (166, 110, "Dispatch", "NVLink", comm),
              (296, 200, None, None, None), (516, 162, "moe_sort", "group by local expert", "old"),
              (698, 144, "Expert GEMMs", "CuteDSL, NVFP4", "new"), (862, 104, "Combine", "NVLink", comm))
    for x, w, t, sub, kind in stages:
        if t:
            box(s, x, y, w, h, t, sub, kind, 18)
    cyl(s, 296, y + 4, 200, h - 8, "Receive workspace" if per_sender_rows else "All-to-all workspace",
        "hidden states, routing", "new", 18)
    for (x1, w1, *_), (x2, *_) in zip(stages, stages[1:]):
        s.arrow(x1 + w1 + (8 if x1 == 296 else 0), y + h / 2, x2 - 4, y + h / 2)
    # Zoom on the token rows.
    for x_top, x_bot in ((296, 20), (501, 980)):
        dashed_line(s, x_top, y + h - 4, x_bot, 186, arrow=False)
    band(s, 186, 224, "")
    # The FT half is tinted green, like the parts FT changes in the pipeline above.
    s.out.append('<path d="M 400 187 L 968 187 Q 979 187 979 198 L 979 398 Q 979 409 968 409 L 400 409 Z" '
                 'fill="#F1F8E6" stroke="none"/>')
    s.out.append('<line x1="400" y1="187" x2="400" y2="409" stroke="#D5D5D5" stroke-width="1.5"/>')
    bh = 32

    def block(x, w, yy, kind, t=None):
        if kind == "dead":
            s.out.append(f'<rect x="{x}" y="{yy}" width="{w - 3}" height="{bh}" rx="4" '
                         f'fill="#EAEAEA" stroke="{GRAY}" stroke-width="1.5" stroke-dasharray="5 4"/>')
        else:
            fill, stroke = {"live": (BLUE_FILL, BLUE), "hit": (RED_FILL, RED)}[kind]
            s.rect(x, yy, w - 3, bh, fill, stroke, 2.4 if kind == "hit" else 1.5, 4)
        if t is not None:
            s.text(x + w / 2, yy + 22, str(t), 15, INK if kind != "dead" else "#8A8A8A", weight="bold")

    def late(x, y_from, y_to):
        s.out.append(f'<line x1="{x}" y1="{y_from}" x2="{x}" y2="{y_to - 4}" stroke="{RED}" stroke-width="2.4" '
                     f'stroke-dasharray="6 4" marker-end="url(#red)"/>')
        s.text(x + 8, (y_from + y_to) / 2 + 5, "late write", 14, RED, anchor="start", weight="bold")

    yN, yM, ymask = 236, 304, 356
    for x, title, color in ((230, "FT off", INK),
                            (690, "FT on (per-sender rows)" if per_sender_rows else "FT on (remapping)", DARK_GREEN)):
        s.text(x, 214, title, 17, color, weight="bold")
    for yy, t in ((yN, "step N"), (yM, "step N+1")):
        s.text(40, yy + 21, t, 15, INK, anchor="start", weight="bold")
    # Both panels share one geometry: sender s's rows start at s x (the step's max tokens per rank) x (row width), so
    # each sender's rows grow from step N to N+1 and rank 1's rows now sit where rank 2 wrote in step N.
    x0, w4, n1, n2 = 130, 200, 25, 50
    for yy in (yN, yM):
        s.rect(x0, yy, w4, bh, "white", "#DADADA", 1.3, 4)
    for r in range(4):
        block(x0 + r * n1, n1, yN, "live", r)
        block(x0 + r * n2, n2, yM, "dead" if r == 2 else "live", None if r == 1 else r)
    block(x0 + 2 * n1, n1, yM, "hit")
    s.text(x0 + n2 + 1.5 * n1, yM + 22, "1", 15, INK, weight="bold")
    late(x0 + 2.5 * n1, yN + bh + 2, yM)
    if per_sender_rows:
        # Sender s always writes its own fixed rows (one max-size share per sender), which the mask skips.
        x1, cap = 475, 120
        for yy in (yN, yM):
            for r in range(4):
                s.rect(x1 + r * cap, yy, cap - 5, bh, "white", "#BDBDBD", 1.3, 4)
        for r in range(4):
            block(x1 + r * cap, n1, yN, "live", r)
            block(x1 + r * cap, n2 * 0.8, yM, "dead" if r == 2 else "live", None if r == 2 else r)
        block(x1 + 2 * cap, n1, yM, "hit")
        late(x1 + 2 * cap + n1 / 2, yN + bh + 2, yM)
        mask_x = [x1 + r * cap + (cap - 5) / 2 for r in range(4)]
    # Same-address remap: step N's pages stay with the dead rank's mapping, so its late write drops straight into
    # them (left); step N+1 runs on fresh pages at the same addresses (right, green outline).
    else:
        x1, m1, m2, wb = 475, n1, n2, w4
        nx = x1 + wb + 40
        s.rect(x1, yN, wb, bh, "white", "#DADADA", 1.3, 4)
        for r in range(4):
            block(x1 + r * m1, m1, yN, "live", r)
        s.out.append(f'<rect x="{x1}" y="{yM}" width="{wb}" height="{bh}" rx="4" '
                     f'fill="#EAEAEA" stroke="{GRAY}" stroke-width="1.5" stroke-dasharray="5 4"/>')
        block(x1 + 2 * m1, m1, yM, "hit")
        late(x1 + 2.5 * m1, yN + bh + 2, yM)
        s.text(x1 + wb / 2, yM + bh + 20, "old pages", 14, INK)
        s.text(nx + wb / 2, yM - 14, "new pages, same addresses", 14, DARK_GREEN, weight="bold")
        s.rect(nx - 5, yM - 5, wb + 7, bh + 10, "white", GREEN, 1.8, 6)  # a frame clear of the blocks
        for r in range(4):
            block(nx + r * m2, m2, yM, "dead" if r == 2 else "live", r)
        mask_x = [nx + r * m2 + (m2 - 3) / 2 for r in range(4)]
    for r, cx in enumerate(mask_x):
        live = r != 2
        s.rect(cx - 14, ymask, 28, 26, GREEN_FILL if live else RED_FILL, GREEN if live else RED, 1.5, 4)
        s.text(cx, ymask + 19, "1" if live else "0", 14, INK, weight="bold")
    s.text(min(mask_x) - 26, ymask + 19, "mask", 15, INK, anchor="end", weight="bold")
    s.text(min(mask_x) - 14, ymask + 46, "(1 = sender alive)", 14, INK, anchor="start")
    ft_swatch(s, 40, 428)
    for x, kind, t in ((40, "new", "added or changed by FT"), (270, "old", "existing"), (390, "red", "late write lands")):
        fill, stroke = KIND[kind]
        if kind != "new":
            s.rect(x, 428, 22, 16, fill, stroke, 2.4 if kind == "red" else 1.8, 4)
        label(s, x + 30, 442, t, size=15)
    FT_CUE["on"] = False
    return s.render()


def sequence():
    """Five phases, left to right; steps top to bottom. Colour shows who acts; band headers give time since the kill."""
    s = Svg(1000, 350)
    s.out.append(MID_MARKER)
    kinds = {"S": "new", "A": "slate", "R": "red"}
    cols = ((20, "Detect"), (214, "Agree"), (408, "Plan"), (602, "Install"), (796, "Resume"))
    for x, t in cols:
        band(s, 16, 288, t, x=x, w=176)
    bw, bh, y0, dy = 150, 52, 62, 86

    def bx(c):
        return cols[c][0] + 13

    def by(i):
        return y0 + i * dy

    def step(c, i, who, title, sub=None):
        box(s, bx(c), by(i), bw, bh, title, sub, kinds[who], 17, dash="7 4" if who == "R" else None)

    def link(c, i, message=False, t=None, up=False):
        x = bx(c) + bw / 2
        y1, y2 = by(i) + bh, by(i + 1) - 4
        if up:
            y1, y2 = by(i + 1), by(i) + bh + 4
        if message:
            dashed_line(s, x, y1, x, y2)
        else:
            s.arrow(x, y1, x, y2, "ink")
        if t:
            label(s, x + 10, (by(i) + bh + by(i + 1)) / 2 + 6, t, "start", 15)

    def across(c, i_from, i_to, t=None, message=True, dy_from=0, dx=0, at=None):
        """An arrow to the next column; t is written horizontally at `at` = (x, y, anchor)."""
        x1, y1 = bx(c) + bw, by(i_from) + bh / 2 + dy_from
        x2, y2 = bx(c + 1), by(i_to) + bh / 2
        xm = (x1 + x2) / 2 + dx
        style = f'stroke="{MID}" stroke-width="{THIN}" stroke-dasharray="6 5"' if message else \
            f'stroke="{INK}" stroke-width="2.2"'
        s.out.append(f'<path d="M {x1} {y1} L {xm} {y1} L {xm} {y2} L {x2 - 4} {y2}" fill="none" {style} '
                     f'marker-end="url(#{"mid" if message else "ink"})"/>')
        if t:
            tx, ty, anchor = at(xm, y1, y2)
            label(s, tx, ty, t, anchor, 15)

    step(0, 0, "R", "Rank 2 dies", "process ends")
    step(0, 1, "S", "Failure watch", "polls ULFM")
    step(0, 2, "S", "Abort flag set", "kernels return")
    link(0, 0)
    link(0, 1)
    step(1, 0, "S", "Shrink", "agree on failed")
    step(1, 1, "S", "Stop", "discard steps")
    link(1, 0)
    step(2, 0, "S", "Placement", "map unbacked slots")
    step(2, 1, "S", "Remap, stage", "send plan hash")
    step(2, 2, "A", "Fence, propose", "generation 1")
    link(2, 0)
    link(2, 1, True, None, up=True)
    label(s, bx(2) + bw / 2 + 10, (by(1) + bh + by(2)) / 2 + 6, "Propose", "start", 15)
    step(3, 0, "A", "Check", "plan hashes match")
    step(3, 1, "S", "Install", "load, route, mask")
    step(3, 2, "A", "Commit", "generation 1")
    link(3, 0, True, "Install")
    link(3, 1, True, "all Installed")
    step(4, 0, "S", "Replay", "discarded steps")
    step(4, 1, "S", "Serve", "same CUDA graphs")
    link(4, 0)
    across(0, 2, 0, None, message=False)
    across(1, 1, 0, None, message=False, dy_from=-10, dx=-6)
    # Labels sit in the free space beside each arrow: below "Stop", between "Placement" and "Remap", below "Serve".
    across(1, 1, 2, "all Stopped", dy_from=10, dx=6, at=lambda xm, y1, y2: (xm - 8, y2 + 5, "end"))
    across(2, 1, 0, "all Ready", at=lambda xm, y1, y2: (xm - 8, (by(0) + bh + by(1)) / 2 + 6, "end"))
    across(3, 2, 0, "Commit", at=lambda xm, y1, y2: (xm + 8, y1 + 5, "start"))
    for x, kind, t in ((40, "new", "survivors"), (170, "slate", "recovery authority"), (360, "red", "failed rank")):
        fill, stroke = KIND[kind]
        s.rect(x, 320, 22, 14, fill, stroke, 1.8, 3, dash="4 2" if kind == "red" else None)
        label(s, x + 30, 332, t, size=15)
    s.arrow(720, 327, 760, 327, "ink")
    label(s, 768, 332, "local", size=15)
    dashed_line(s, 830, 327, 870, 327)
    label(s, 878, 332, "ZeroMQ", size=15)
    return s.render()


def restore_lost_experts(placement, lost, dead_ranks, restored_per_rank=None):
    """The placement rule of moe_load_balancer.restore_lost_experts (v5-final), copied to draw its result."""
    usable = np.ones(placement.shape, dtype=bool)
    usable[list(dead_ranks)] = False
    counts = np.bincount(placement[placement >= 0], minlength=max(lost) + 1 if lost else 0)
    if restored_per_rank is None:
        restored_per_rank = np.zeros(placement.shape[0], dtype=np.int64)
    for expert in lost:
        ranks, slots = np.nonzero(usable)
        held = placement[ranks, slots]
        replicas = np.where(held >= 0, counts[np.maximum(held, 0)], 0)
        free = (held < 0) | (replicas >= 2)
        ranks, slots, held, replicas = (a[free] for a in (ranks, slots, held, replicas))
        best = np.lexsort((slots, ranks, restored_per_rank[ranks], -replicas, held >= 0))[0]
        rank, slot = ranks[best], slots[best]
        if held[best] >= 0:
            counts[held[best]] -= 1
        placement[rank, slot] = expert
        counts[expert] = 1
        restored_per_rank[rank] += 1


def placement_cases():
    """(title, sub, dead ranks, slots per rank, placement before, placement after) for DeepSeek-R1 on 32 GPUs."""
    cases = []
    # No EPLB: 8 experts + 2 ghost slots per rank (slots 8 and 9 empty).
    static = np.full((32, 10), -1)
    static[:, :8] = np.arange(256).reshape(32, 8)
    for title, dead in (("Rank 2 lost", [2]), ("Node 1 lost", [4, 5, 6, 7])):
        after = static.copy()
        restore_lost_experts(after, sorted(static[dead, :8].ravel().tolist()), dead)
        cases.append((title, ("No EPLB",), dead, static, after))
    # Online EPLB, illustrative: 9 slots + 1 ghost per rank, 288 slots for 256 experts (32 extra copies). Node 1 holds
    # 36 single-copy experts. On the other ranks, 32 experts have two copies: 28 in slots 7 and 8 (the copies 14
    # survivors apart), 4 in slot 6 of ranks 24-31.
    eplb = np.full((32, 10), -1)
    eplb[4:8, :9] = np.arange(36).reshape(4, 9)
    survivors = [r for r in range(32) if not 4 <= r < 8]
    ids = iter(range(36, 256))
    for k, r in enumerate(survivors):
        eplb[r, :6] = [next(ids) for _ in range(6)]
        if k < 20:
            eplb[r, 6] = next(ids)
    for k in range(28):
        eplb[survivors[k], 7] = eplb[survivors[(k + 14) % 28], 8] = next(ids)
    for k in range(4):
        eplb[survivors[20 + k], 6] = eplb[survivors[24 + k], 6] = next(ids)
    assert sorted(set(eplb[eplb >= 0].tolist())) == list(range(256)) and (eplb >= 0).sum() == 288
    after = eplb.copy()
    restore_lost_experts(after, list(range(36)), [4, 5, 6, 7])
    cases.append(("Node 1 lost", ("Online EPLB",), [4, 5, 6, 7], eplb, after))
    return cases


def placement(slide=False):
    """Where lost experts go: empty (ghost) slots first, then a replica's slot; fewest restored, then lowest rank.

    The slide variant drops the left title column: each panel's title sits on a line above its grid.
    """
    gx = 20 if slide else 120
    s = Svg(736, 720) if slide else Svg(860, 700)
    hatch = '<defs><pattern id="hatch" width="6" height="6" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">' \
            '<line x1="0" y1="0" x2="0" y2="6" stroke="#C8C8C8" stroke-width="2"/></pattern></defs>'
    s.out.append(hatch)
    s.out.append(f'<defs><pattern id="lhatch" width="4" height="4" patternUnits="userSpaceOnUse" '
                 f'patternTransform="rotate(45)"><line x1="0" y1="0" x2="0" y2="4" stroke="{RED}" stroke-width="0.8"/>'
                 '</pattern></defs>')
    rep_fill, rep_stroke = "#C9D1DA", BLUE

    def cell(x, y, kind, w=16, h=14):
        if kind == "ghost":
            s.out.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="2" fill="url(#hatch)" stroke="{GRAY}" '
                         f'stroke-width="1.2" stroke-dasharray="3 2"/>')
        else:
            fill, stroke, sw = {"expert": (BLUE_FILL, BLUE, 1.0), "replica": (rep_fill, rep_stroke, 1.4),
                                "restored": ("#BBDC80", GREEN, 1.6), "lost": (RED_FILL, RED, 1.2)}[kind]
            s.rect(x, y, w, h, fill, stroke, sw, 2)
            if kind == "replica":  # a dot, so replicas and restored experts differ in grayscale too
                s.out.append(f'<circle cx="{x + w / 2}" cy="{y + h / 2}" r="2.9" fill="{BLUE}"/>')
            if kind == "lost":
                s.out.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="2" fill="url(#lhatch)" stroke="none"/>')

    for p, (title, sub, dead, before, after) in enumerate(placement_cases()):
        gy = 62 + p * 214 if slide else p * 212 + 46
        if slide:
            s.text(gx, gy - 10, title, 16, INK, anchor="start", weight="bold")
            s.text(gx + 105, gy - 10, " ".join(sub), 13, SUB, anchor="start")
        else:
            s.text(14, gy + 90, title, 16, INK, anchor="start", weight="bold")
            for k, t in enumerate(sub):
                s.text(14, gy + 110 + k * 17, t, 13, SUB, anchor="start")
        live = after.copy()
        live[list(dead)] = -1
        counts = np.bincount(live[live >= 0], minlength=256)
        for r in range(32):
            x = gx + r * 20 + (r // 4) * 8
            if p == 0 and r % 4 == 0:
                s.text(x + 38, 18, f"Node {r // 4}", 13, INK)
            for row in range(10):
                slot_id = 9 - row
                y = gy + row * 17
                if r in dead:
                    cell(x, y, "lost")
                elif after[r, slot_id] != before[r, slot_id]:
                    cell(x, y, "restored")
                elif before[r, slot_id] < 0:
                    cell(x, y, "ghost")
                else:
                    cell(x, y, "replica" if counts[after[r, slot_id]] >= 2 else "expert")
            if p == 2:
                s.text(x + 8, gy + 182, str(r), 11, INK)
    ly = 696 if slide else 675
    items = (("expert", "expert slot", 120), ("ghost", "unbacked spare slot", 228), ("replica", "expert with 2 copies", 405),
             ("restored", "restored expert", 572), ("lost", "lost rank", 712))
    for kind, t, x in items:
        x += gx - 120
        cell(x, ly, kind, 18, 10)
        s.text(x + 24, ly + 11, t, 14, INK, anchor="start")
    return s.render()


def failure_detection():
    """The two detection paths (crash, hang), where they meet, and what the survivors do next."""
    s = Svg(1000, 612)
    s.out.append(MID_MARKER)

    def auth(x, y, w, h, title, sub=None, lines=None):
        """A step of the recovery authority: FT's fill, a slate outline."""
        if lines:
            box3(s, x, y, w, h, title, lines, "new")
        else:
            box(s, x, y, w, h, title, sub, "new", 18)
        s.rect(x, y, w, h, "none", SLATE, 2.6, 8)

    def message(x1, y1, x2, y2, above=None, below=None):
        dashed_line(s, x1, y1, x2 - 4, y2)
        if above:
            label(s, (x1 + x2) / 2, y1 - 9, above, "middle", 15)
        if below:
            label(s, (x1 + x2) / 2, y1 + 22, below, "middle", 15)

    # Hang path: the authority confirms a hung rank from its peers' reports, then has its process killed.
    band(s, 16, 140, "Hang detection (opt-in)")
    box(s, 40, 70, 140, 60, "Rank 2 stuck", "alive", "red", 18, dash="7 4")
    message(180, 100, 290, 100, "peers report", "every 1 s")
    auth(290, 56, 210, 84, "Suspects rank 2", lines=("every live peer waits on it", "wait > 2 s, or silent 5 s"))
    s.arrow(500, 100, 586, 100)
    label(s, 543, 91, "holds 30 s", "middle", 15)
    auth(590, 70, 150, 60, "Confirms hung")
    message(740, 100, 810, 100, "Fence")
    box(s, 810, 70, 150, 60, "Fence", "kill its process", "new", 18)
    s.path("M 885 130 L 885 178 L 110 178 L 110 236", "gray")
    label(s, 500, 172, "then as a crash", "middle", 15)

    # Crash path, on every survivor.
    band(s, 192, 140, "Crash detection", w=720)
    box(s, 40, 240, 140, 60, "Rank 2 ends", "killed or exits", "red", 18, dash="7 4")
    s.arrow(180, 270, 286, 270)
    label(s, 233, 261, "process gone", "middle", 15)
    box(s, 290, 240, 150, 60, "ULFM", "marks rank failed", "new", 18)
    s.arrow(440, 270, 536, 270)
    label(s, 488, 261, "receive fails", "middle", 15)
    box(s, 540, 240, 170, 60, "Failure watch", "polls every 20 ms", "new", 18)

    # The kernels and what they read.
    band(s, 346, 210, "Kernels and execution state", w=720)
    box3(s, 40, 422, 210, 100, "Dispatch, combine", ("return early on abort", "skip masked ranks"), "new")
    s.rect(452, 368, 268, 176, "white", "#BDBDBD", 1.6, 12, dash="7 5")
    s.text(466, 393, "Execution state", 17, INK, anchor="start", weight="bold")
    cyl(s, 470, 412, 230, 52, "Abort flag", "pinned host memory", "new", 18)
    cyl(s, 470, 480, 230, 52, "Live-rank mask", "GPU memory", "new", 18)
    s.arrow(625, 300, 625, 404, "green")
    label(s, 615, 360, "sets abort flag", "end", 15)
    s.arrow(466, 438, 254, 438)
    label(s, 358, 429, "polled in waits", "middle", 15)
    s.arrow(466, 506, 254, 506)
    label(s, 358, 497, "read each launch", "middle", 15)

    # What every survivor does next, with the authority.
    band(s, 192, 364, "Survivors next", x=754, w=226)
    bx, bw = 772, 190
    box(s, bx, 245, bw, 50, "Recovery hook", "notified", "new", 18)
    s.arrow(710, 270, bx - 4, 270, "green")
    label(s, 741, 261, "notifies", "middle", 15)
    box(s, bx, 325, bw, 50, "Shrink", "agree on failed ranks", "new", 18)
    auth(bx, 405, bw, 50, "Proposes plan", "new generation")
    box(s, bx, 485, bw, 50, "Hook writes mask", "clears failed ranks", "new", 18)
    cx = bx + bw / 2
    s.arrow(cx, 295, cx, 321)
    message(cx, 375, cx, 405)
    label(s, cx + 10, 395, "Stopped", "start", 15)
    message(cx, 455, cx, 485)
    label(s, cx + 10, 475, "Propose", "start", 15)
    s.arrow(bx, 506, 704, 506, "green")
    label(s, 738, 497, "writes", "middle", 15)

    # Legend.
    ly = 580
    ft_swatch(s, 40, ly)
    label(s, 70, ly + 14, "added or changed by FT", size=15)
    s.rect(370, ly, 22, 16, RED_FILL, RED, 1.8, 4, dash="4 2")
    label(s, 400, ly + 14, "lost rank", size=15)
    s.rect(492, ly, 22, 16, GREEN_FILL, SLATE, 2.6, 4)
    label(s, 522, ly + 14, "recovery authority (own process)", size=15)
    dashed_line(s, 770, ly + 8, 810, ly + 8)
    label(s, 818, ly + 14, "ZeroMQ message", size=15)
    return s.render()


# Slide variants: the same figures, rearranged to near-square for the right half of a slide.


def auth_box(s, x, y, w, h, title, sub=None, lines=None):
    """A step of the recovery authority: FT's fill, a slate outline."""
    if lines:
        box3(s, x, y, w, h, title, lines, "new")
    else:
        box(s, x, y, w, h, title, sub, "new", 18)
    s.rect(x, y, w, h, "none", SLATE, 2.6, 8)


def message(s, x1, y1, x2, y2, above=None, below=None):
    """A ZeroMQ message; labels sit above and below its middle."""
    dashed_line(s, x1, y1, x2 + (4 if x2 < x1 else -4), y2)
    if above:
        label(s, (x1 + x2) / 2, y1 - 9, above, "middle", 15)
    if below:
        label(s, (x1 + x2) / 2, y1 + 22, below, "middle", 15)


def failure_detection_slide():
    """figD stacked: hang path, crash path, survivors' steps, then the kernels and descriptor."""
    s = Svg(680, 850)
    s.out.append(MID_MARKER)
    # Hang path in two rows: suspect, hold, confirm, then fence.
    band(s, 16, 206, "Hang detection (opt-in)", w=640)
    box(s, 40, 64, 140, 56, "Rank 2 stuck", "alive", "red", 18, dash="7 4")
    message(s, 180, 92, 290, 92, "peers report", "every 1 s")
    auth_box(s, 290, 52, 210, 80, "Suspects rank 2", lines=("every live peer waits on it", "wait > 2 s, or silent 5 s"))
    s.path("M 500 92 L 565 92 L 565 148", "gray")
    label(s, 575, 131, "holds 30 s", "start", 15)
    auth_box(s, 490, 152, 150, 56, "Confirms hung")
    message(s, 490, 180, 430, 180, "Fence")
    box(s, 280, 152, 150, 56, "Fence", "kill its process", "new", 18)
    c = 254
    s.path(f"M 355 208 L 355 240 L 150 240 L 150 {c + 40}", "gray")
    label(s, 252, 234, "then as a crash", "middle", 15)

    # Crash path, on every survivor.
    band(s, c, 112, "Crash detection", w=640)
    yc = c + 72
    box(s, 40, c + 44, 130, 56, "Rank 2 ends", "killed or exits", "red", 18, dash="7 4")
    s.arrow(170, yc, 246, yc)
    label(s, 208, yc - 9, "process", "middle", 15)
    label(s, 208, yc + 22, "gone", "middle", 15)
    box(s, 250, c + 44, 145, 56, "ULFM", "marks rank failed", "new", 18)
    s.arrow(395, yc, 461, yc)
    label(s, 428, yc - 9, "receive", "middle", 15)
    label(s, 428, yc + 22, "fails", "middle", 15)
    box(s, 465, c + 44, 160, 56, "Failure watch", "polls every 20 ms", "new", 18)

    # What every survivor does next, with the authority: a U from the hook to the mask.
    v = c + 134
    band(s, v, 178, "Survivors next", w=610)
    lx, rx, bw = 60, 410, 200
    r1, r2 = v + 42, v + 116
    s.arrow(530, c + 100, 530, r1 - 4, "green")
    label(s, 540, v + 28, "notifies", "start", 15)
    box(s, rx, r1, bw, 50, "Recovery hook", "notified", "new", 18)
    s.arrow(rx, r1 + 25, lx + bw + 4, r1 + 25)
    box(s, lx, r1, bw, 50, "Shrink", "agree on failed ranks", "new", 18)
    message(s, lx + bw / 2, r1 + 50, lx + bw / 2, r2)
    label(s, lx + bw / 2 + 10, r1 + 66, "Stopped", "start", 15)
    auth_box(s, lx, r2, bw, 50, "Proposes plan", "new generation")
    message(s, lx + bw, r2 + 25, rx, r2 + 25, "Propose")
    box(s, rx, r2, bw, 50, "Hook writes mask", "clears failed ranks", "new", 18)

    # The kernels and what they read; the descriptor's two fields stack on the right.
    k = v + 200
    band(s, k, 194, "Kernels and execution state", w=640)
    ym, ya = k + 48, k + 114
    box3(s, 40, k + 58, 200, 92, "Dispatch, combine", ("return early on abort", "skip masked ranks"), "new")
    s.rect(380, k + 12, 236, 170, "white", "#BDBDBD", 1.6, 12, dash="7 5")
    s.text(394, k + 37, "Execution state", 17, INK, anchor="start", weight="bold")
    cyl(s, 396, ym, 200, 52, "Live-rank mask", "GPU memory", "new", 18)
    cyl(s, 396, ya, 200, 52, "Abort flag", "pinned host memory", "new", 18)
    s.arrow(392, ym + 26, 244, ym + 26)
    label(s, 316, ym + 17, "read each launch", "middle", 15)
    s.arrow(392, ya + 26, 244, ya + 26)
    label(s, 316, ya + 17, "polled in waits", "middle", 15)
    s.arrow(560, r2 + 50, 560, ym - 8, "green")
    label(s, 550, k - 6, "writes", "end", 15)
    s.path(f"M 625 {yc} L 645 {yc} L 645 {ya + 26} L 604 {ya + 26}", "green")
    # The watch's line runs down the right edge, clear of the other labels; its label runs along it.
    s.out.append(f'<text x="651" y="{v + 89}" font-size="15" fill="{INK}" text-anchor="middle" '
                 f'font-weight="normal" transform="rotate(90 651 {v + 89})">sets abort flag</text>')

    # Legend, two rows.
    ly = k + 214
    ft_swatch(s, 40, ly)
    label(s, 70, ly + 14, "added or changed by FT", size=15)
    s.rect(390, ly, 22, 16, RED_FILL, RED, 1.8, 4, dash="4 2")
    label(s, 420, ly + 14, "lost rank", size=15)
    ly += 28
    s.rect(40, ly, 22, 16, GREEN_FILL, SLATE, 2.6, 4)
    label(s, 70, ly + 14, "recovery authority (own process)", size=15)
    dashed_line(s, 390, ly + 8, 430, ly + 8)
    label(s, 438, ly + 14, "ZeroMQ message", size=15)
    return s.render()


def late_writes_slide(per_sender_rows=False):
    """Fig 4 stacked: the pipeline as a loop around the workspace, then FT off above FT on.

    per_sender_rows draws the Oct 20 design instead (Figure G1): only the workspace and expert GEMMs change, and
    each sender owns fixed rows, so a late write stays in the dead sender's own rows.
    """
    FT_CUE["on"] = True
    comm = "old" if per_sender_rows else "new"
    s = Svg(680, 690)
    band(s, 16, 202, "One MoE layer on one GPU (captured in the CUDA graph)", w=640)
    h = 60
    y1, y2 = 64, 142
    box(s, 40, y1, 116, h, "Router", "top 8 experts", "old", 18)
    box(s, 268, y1, 110, h, "Dispatch", "NVLink", comm, 18)
    # The workspace spans both rows; the second row runs back to the left.
    wx, ww, wy, wh = 525, 120, y1 + 4, y2 + h - y1 - 8
    cyl(s, wx, wy, ww, wh, "", None, "new")
    for i, t in enumerate(("Receive" if per_sender_rows else "All-to-all", "workspace")):
        s.text(wx + ww / 2, wy + wh / 2 - 26 + i * 22, t, 18, INK, weight="bold")
    for i, t in enumerate(("hidden states,", "routing")):
        s.text(wx + ww / 2, wy + wh / 2 + 20 + i * 20, t, 16, SUB)
    box(s, 337, y2, 160, h, "moe_sort", "group by local expert", "old", 18)
    box(s, 161, y2, 150, h, "Expert GEMMs", "CuteDSL, NVFP4", "new", 18)
    box(s, 40, y2, 95, h, "Combine", "NVLink", comm, 18)
    s.arrow(156, y1 + h / 2, 264, y1 + h / 2)
    s.arrow(378, y1 + h / 2, wx - 4, y1 + h / 2)
    s.arrow(wx, y2 + h / 2, 501, y2 + h / 2)
    s.arrow(337, y2 + h / 2, 315, y2 + h / 2)
    s.arrow(161, y2 + h / 2, 139, y2 + h / 2)
    # Zoom on the token rows.
    top = 240
    for x_top, x_bot in ((wx, 20), (wx + ww + 5, 660)):
        dashed_line(s, x_top, wy + wh, x_bot, top, arrow=False)
    yon = 412
    band(s, top, 664 - top - 20, "", w=640)
    # The FT panel is tinted green, like the parts FT changes in the pipeline above.
    s.out.append(f'<path d="M 21 {yon} L 659 {yon} L 659 {632} Q 659 643 648 643 L 32 643 Q 21 643 21 632 Z" '
                 'fill="#F1F8E6" stroke="none"/>')
    s.out.append(f'<line x1="21" y1="{yon}" x2="659" y2="{yon}" stroke="#D5D5D5" stroke-width="1.5"/>')
    bh = 32

    def block(x, w, yy, kind, t=None):
        if kind == "dead":
            s.out.append(f'<rect x="{x}" y="{yy}" width="{w - 3}" height="{bh}" rx="4" '
                         f'fill="#EAEAEA" stroke="{GRAY}" stroke-width="1.5" stroke-dasharray="5 4"/>')
        else:
            fill, stroke = {"live": (BLUE_FILL, BLUE), "hit": (RED_FILL, RED)}[kind]
            s.rect(x, yy, w - 3, bh, fill, stroke, 2.4 if kind == "hit" else 1.5, 4)
        if t is not None:
            s.text(x + w / 2, yy + 22, str(t), 15, INK if kind != "dead" else "#8A8A8A", weight="bold")

    def steps(yN, yM):
        for yy, t in ((yN, "step N"), (yM, "step N+1")):
            s.text(40, yy + 21, t, 15, INK, anchor="start", weight="bold")

    x0, w4, n1, n2 = 120, 240, 30, 60
    # FT off: sender s's rows start at s x (the step's max tokens per rank) x (row width).
    yN, yM = top + 52, top + 120
    s.text(340, top + 30, "FT off", 17, INK, weight="bold")
    steps(yN, yM)
    for yy in (yN, yM):
        s.rect(x0, yy, w4, bh, "white", "#DADADA", 1.3, 4)
    for r in range(4):
        block(x0 + r * n1, n1, yN, "live", r)
        block(x0 + r * n2, n2, yM, "dead" if r == 2 else "live", None if r == 1 else r)
    block(x0 + 2 * n1, n1, yM, "hit")
    s.text(x0 + n2 + 1.5 * n1, yM + 22, "1", 15, INK, weight="bold")
    xl = x0 + 2.5 * n1
    s.out.append(f'<line x1="{xl}" y1="{yN + bh + 2}" x2="{xl}" y2="{yM - 4}" stroke="{RED}" stroke-width="2.4" '
                 f'stroke-dasharray="6 4" marker-end="url(#red)"/>')
    s.text(xl + 8, (yN + bh + 2 + yM) / 2 + 5, "late write", 14, RED, anchor="start", weight="bold")

    yN, yM = yon + 52, yon + 120
    ymask = yM + 52
    s.text(340, yon + 30, "FT on (per-sender rows)" if per_sender_rows else "FT on (remapping)", 17, DARK_GREEN,
           weight="bold")
    steps(yN, yM)
    if per_sender_rows:
        # FT on: sender s always writes its own rows (one max-size share per sender), which the mask skips.
        cap = 128
        for yy in (yN, yM):
            for r in range(4):
                s.rect(x0 + r * cap, yy, cap - 5, bh, "white", "#BDBDBD", 1.3, 4)
        for r in range(4):
            block(x0 + r * cap, n1, yN, "live", r)
            block(x0 + r * cap, n2 * 0.8, yM, "dead" if r == 2 else "live", None if r == 2 else r)
        block(x0 + 2 * cap, n1, yM, "hit")
        xl = x0 + 2 * cap + n1 / 2
        s.out.append(f'<line x1="{xl}" y1="{yN + bh + 2}" x2="{xl}" y2="{yM - 4}" stroke="{RED}" stroke-width="2.4" '
                     f'stroke-dasharray="6 4" marker-end="url(#red)"/>')
        s.text(xl + 8, (yN + bh + 2 + yM) / 2 + 5, "late write", 14, RED, anchor="start", weight="bold")
        mask_x = [x0 + r * cap + (cap - 5) / 2 for r in range(4)]
        return late_writes_slide_tail(s, ymask, mask_x)
    # FT on: the same addresses, backed by fresh memory at step N+1; the late write lands in the old pages.
    x1, m1, m2, wb = x0, n1, n2, w4
    s.rect(x1, yN, wb, bh, "white", "#DADADA", 1.3, 4)
    s.rect(x1, yM, wb, bh, "white", GREEN, 1.8, 4)
    for r in range(4):
        block(x1 + r * m1, m1, yN, "live", r)
        block(x1 + r * m2, m2, yM, "dead" if r == 2 else "live", r)
    ox, ow = x1 + wb + 40, wb
    s.out.append(f'<rect x="{ox}" y="{yM}" width="{ow}" height="{bh}" rx="4" '
                 f'fill="#EAEAEA" stroke="{GRAY}" stroke-width="1.5" stroke-dasharray="5 4"/>')
    hx = ox + 2 * m1
    block(hx, m1, yM, "hit")
    s.text(ox + ow / 2, yM + bh + 20, "old pages", 14, INK)
    xa, xb, ym = x1 + 2.5 * m1, hx + m1 / 2 - 1.5, yN + bh + 19
    s.out.append(f'<path d="M {xa} {yN + bh + 2} L {xa} {ym} L {xb} {ym} L {xb} {yM - 4}" fill="none" stroke="{RED}" '
                 f'stroke-width="2.4" stroke-dasharray="6 4" marker-end="url(#red)"/>')
    s.text(x1 + wb + 6, ym - 5, "late write", 14, RED, anchor="start", weight="bold")
    return late_writes_slide_tail(s, ymask, [x1 + r * m2 + (m2 - 3) / 2 for r in range(4)])


def late_writes_slide_tail(s, ymask, mask_x):
    """The FT-on panel's mask row under the given sender centres, the legend, and the render."""
    s.text(40, ymask + 21, "mask", 15, INK, anchor="start", weight="bold")
    for r, cx in enumerate(mask_x):
        live = r != 2
        s.rect(cx - 14, ymask, 28, 26, GREEN_FILL if live else RED_FILL, GREEN if live else RED, 1.5, 4)
        s.text(cx, ymask + 19, "1" if live else "0", 14, INK, weight="bold")
    s.text(40, ymask + 46, "(live-rank mask: 1 = sender alive)", 14, INK, anchor="start")
    ly = 662
    ft_swatch(s, 40, ly)
    for x, kind, t in ((40, "new", "added or changed by FT"), (270, "old", "existing"), (390, "red", "late write lands")):
        fill, stroke = KIND[kind]
        if kind != "new":
            s.rect(x, ly, 22, 16, fill, stroke, 2.4 if kind == "red" else 1.8, 4)
        label(s, x + 30, ly + 14, t, size=15)
    FT_CUE["on"] = False
    return s.render()


def sequence_slide():
    """The recovery sequence transposed: five phases top to bottom; steps left to right."""
    s = Svg(680, 716)
    s.out.append(MID_MARKER)
    kinds = {"S": "new", "A": "slate", "R": "red"}
    phases = ("Detect", "Agree", "Plan", "Install", "Resume")
    bw, bh, gap = 146, 52, 88

    def bandy(p):
        return 16 + p * 136

    for p, t in enumerate(phases):
        band(s, bandy(p), 106, t, w=640)

    def bx(c):
        return 40 + c * (bw + gap)

    def by(p):
        return bandy(p) + 40

    def step(p, c, who, title, sub=None):
        box(s, bx(c), by(p), bw, bh, title, sub, kinds[who], 17, dash="7 4" if who == "R" else None)

    def link(p, c, message=False, t=None, back=False):
        """Within a phase: c to c + 1, or c + 1 back to c."""
        y = by(p) + bh / 2
        x1, x2 = bx(c) + bw, bx(c + 1) - 4
        if back:
            x1, x2 = bx(c + 1), bx(c) + bw + 4
        if message:
            dashed_line(s, x1, y, x2, y)
        else:
            s.arrow(x1, y, x2, y, "ink")
        if t:
            label(s, (bx(c) + bw + bx(c + 1)) / 2, y - 9, t, "middle", 15)

    def down(p, c_from, c_to, message=True, dx_from=0, dx_to=0, t=None, t_anchor="start"):
        """To the next phase: down from a step, along the gap between bands, down into a step."""
        x1, y1 = bx(c_from) + bw / 2 + dx_from, by(p) + bh
        x2, y2 = bx(c_to) + bw / 2 + dx_to, by(p + 1)
        ym = bandy(p) + 121
        style = f'stroke="{MID}" stroke-width="{THIN}" stroke-dasharray="6 5"' if message else \
            f'stroke="{INK}" stroke-width="2.2"'
        s.out.append(f'<path d="M {x1} {y1} L {x1} {ym} L {x2} {ym} L {x2} {y2 - 4}" fill="none" {style} '
                     f'marker-end="url(#{"mid" if message else "ink"})"/>')
        if t:
            label(s, x2 + (10 if t_anchor == "start" else -10), bandy(p + 1) + 28, t, t_anchor, 15)

    step(0, 0, "R", "Rank 2 dies", "process ends")
    step(0, 1, "S", "Failure watch", "polls ULFM")
    step(0, 2, "S", "Abort flag set", "kernels return")
    link(0, 0)
    link(0, 1)
    step(1, 0, "S", "Shrink", "agree on failed")
    step(1, 1, "S", "Stop", "discard steps")
    link(1, 0)
    step(2, 0, "S", "Placement", "map unbacked slots")
    step(2, 1, "S", "Remap, stage", "send plan hash")
    step(2, 2, "A", "Fence, propose", "generation 1")
    link(2, 0)
    link(2, 1, True, "Propose", back=True)
    step(3, 0, "A", "Check", "plan hashes match")
    step(3, 1, "S", "Install", "load, route, mask")
    step(3, 2, "A", "Commit", "generation 1")
    link(3, 0, True, "Install")
    link(3, 1, True, "all Installed")
    step(4, 0, "S", "Replay", "discarded steps")
    step(4, 1, "S", "Serve", "same CUDA graphs")
    link(4, 0)
    # Phase to phase; the first step of a phase sits under its title, so arrows land right of the title.
    down(0, 2, 0, message=False, dx_to=30)
    down(1, 1, 0, message=False, dx_from=-12, dx_to=30)
    down(1, 1, 2, dx_from=12, t="all Stopped", t_anchor="end")
    down(2, 1, 0, dx_to=30, t="all Ready")
    down(3, 2, 0, dx_to=30, t="Commit")
    ly = 688
    for x, kind, t in ((40, "new", "survivors"), (150, "slate", "recovery authority"), (322, "red", "failed rank")):
        fill, stroke = KIND[kind]
        s.rect(x, ly, 22, 14, fill, stroke, 1.8, 3, dash="4 2" if kind == "red" else None)
        label(s, x + 30, ly + 12, t, size=15)
    s.arrow(450, ly + 7, 485, ly + 7, "ink")
    label(s, 493, ly + 12, "local", size=15)
    dashed_line(s, 545, ly + 7, 580, ly + 7)
    label(s, 588, ly + 12, "ZeroMQ", size=15)
    return s.render()


def ghost_slots_slide():
    """Fig 2 stacked: GPU 0's ten slots in two rows of five, then the host copies: nodes 0, 1 and 7 and the title in a 2 x 2 grid."""
    s = Svg(650, 740)
    gb = 410
    band(s, 16, gb - 16, "GPU 0 (restores 1 of rank 2's 8 lost experts)", w=600)
    x0, sw, gap = 170, 80, 8
    s.text(40, 82, "Expert slots", 17, INK, anchor="start", weight="bold")
    s.text(40, 104, "(virtual)", 16, INK, anchor="start")
    s.text(40, 160, "GPU memory", 17, INK, anchor="start", weight="bold")
    s.text(40, 182, "(physical)", 16, INK, anchor="start")
    rows = (62, 206)
    for i in range(10):
        x, y = x0 + (i % 5) * (sw + gap), rows[i // 5]
        if i < 8:
            slot(s, x, y, sw, 50, "expert", f"e{i}", 18)
            s.rect(x, y + 78, sw, 44, BLUE_FILL, BLUE, 1.6, 4)
            s.arrow(x + sw / 2, y + 52, x + sw / 2, y + 74, "gray")
        elif i == 8:
            kx = x
            slot(s, x, y, sw, 50, "restored", None)
            s.text(x + sw / 2, y + 22, "e16", 17, INK, weight="bold")
            s.text(x + sw / 2, y + 42, "from rank 2", 13, INK)
        else:
            slot(s, x, y, sw, 50, "ghost")
            s.text(x + sw / 2, y + 70, "maps shared", 14, INK)
            s.text(x + sw / 2, y + 88, "scratch", 14, INK)
    # The KV cache: most of GPU memory, drawn open-ended. It lends one block, mapped behind the restored slot.
    ky, kh = 338, 56
    xe = x0 + 5 * (sw + gap) - gap
    s.rect(x0, ky, xe - x0, kh, GRAY_FILL, GRAY, 1.6, 4)
    s.text(x0 + 110, ky + 34, "KV cache", 18, INK, weight="bold")
    s.rect(kx + 5, ky + 5, sw - 10, kh - 10, GREEN_FILL, GREEN, 3.6, 3)
    s.arrow(kx + sw / 2, rows[1] + 52, kx + sw / 2, ky + 1, "green")
    label(s, kx + sw / 2 + 8, 318, "CUDA VMM", "start", 14)

    # Without online EPLB, each node's Grace memory keeps the NEXT node's experts (a ring); node 7 keeps node 0's.
    hb = gb + 36
    band(s, hb, 278, "", w=600)
    nh, nw = 118, 266
    ny1, ny2 = hb + 14, hb + 146
    nodes = ((40, ny1, 0, "e32\u2013e63", 1), (336, ny1, 1, "e64\u2013e95", 2), (336, ny2, 7, "e0\u2013e31", 0))
    for nx, ny, k, held, owner in nodes:
        s.rect(nx, ny, nw, nh, "white", GRAY, 1.8, 10)
        s.text(nx + 12, ny + 24, f"Node {k}", 16, INK, anchor="start", weight="bold")
        for j in range(4):
            r = 4 * k + j
            dead = r == 2
            x = nx + 8 + j * 64
            s.rect(x, ny + 34, 58, 28, RED_FILL if dead else GRAY_FILL, RED if dead else GRAY, 1.8, 4,
                   dash="4 2" if dead else None)
            s.text(x + 29, ny + 53, f"GPU {r}", 13, INK, weight="bold")
        cyl(s, nx + 8, ny + 78, 244, 28, f"Grace memory: Node {owner}'s {held}", None, "new" if k == 7 else "old", 14)
    s.text(321, ny2 + nh / 2 + 8, "...", 26, INK, weight="bold")
    # The band's title sits in the free cell of the grid.
    for i, t in enumerate(("Host copies of", "expert weights", "(Grace memory)")):
        s.text(48, ny2 + 38 + i * 24, t, 18, INK, anchor="start", weight="bold")
    # GPU 0 of node 0 and the top panel share a dark outline: the panel is that GPU.
    s.rect(48, ny1 + 34, 58, 28, "white", SLATE, 2.4, 4)
    s.text(77, ny1 + 53, "GPU 0", 13, INK, weight="bold")
    s.rect(20, 16, 600, gb - 16, "none", SLATE, 2.4, 12)
    for xa, xb in ((48, 20), (106, 200)):
        dashed_line(s, xa, ny1 + 34, xb, gb, arrow=False)
    # Rank 2's e16 is restored from node 7's copy, over NVLink, into rank 0's ghost slot.
    ym, xg = gb + 18, 632
    s.path(f"M {336 + nw} {ny2 + nh / 2} L {xg} {ny2 + nh / 2} L {xg} {ym} L {kx + sw / 2} {ym} "
           f"L {kx + sw / 2} {ky + kh - 1}", "green")
    label(s, 622, ym + 14, "e16 over NVLink", "end", 15)
    return s.render()


EXPORT_WIDTH = 2056
# Slide exports: twice the doc figures' pixels per SVG unit.
SLIDE_SCALE = 2 * EXPORT_WIDTH / 1000


def restyle(svg):
    """The data charts' palette: grey canvas, slate text instead of black."""
    return (svg.replace('<rect width="100%" height="100%" fill="white"/>',
                        f'<rect width="100%" height="100%" fill="{CANVAS}"/>', 1)
            .replace(INK, SLATE).replace(SUB, SLATE_SUB))


if __name__ == "__main__":
    import sys

    import cairosvg

    out = sys.argv[1]
    for name, draw in (("fig1_architecture_final", architecture), ("fig2_ghost_slots_final", ghost_slots),
                       ("fig3_placement_final", placement), ("fig4_late_writes_final", late_writes),
                       ("fig_recovery_sequence_final", sequence),
                       ("figD_failure_detection_final", failure_detection),
                       ("figH_oct20_late_writes", lambda: late_writes(per_sender_rows=True))):
        svg = restyle(draw())
        open(f"{out}/{name}.svg", "w").write(svg)
        width = float(re.search(r'<svg [^>]*width="([0-9.]+)"', svg).group(1))
        cairosvg.svg2png(bytestring=svg.encode(), write_to=f"{out}/{name}.png", scale=EXPORT_WIDTH / width)
    for name, draw in (("figD_failure_detection_slide", failure_detection_slide),
                       ("fig4_late_writes_slide", late_writes_slide),
                       ("figH_oct20_late_writes_slide", lambda: late_writes_slide(per_sender_rows=True)),
                       ("fig_recovery_sequence_slide", sequence_slide),
                       ("fig2_ghost_slots_slide", ghost_slots_slide),
                       ("fig3_placement_slide", lambda: placement(slide=True))):
        svg = restyle(draw())
        open(f"{out}/{name}.svg", "w").write(svg)
        cairosvg.svg2png(bytestring=svg.encode(), write_to=f"{out}/{name}.png", scale=SLIDE_SCALE)
    print("ok")
