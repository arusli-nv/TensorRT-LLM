"""Design-doc figures, built on make_arch's SVG helpers. Captions live in the document."""

import make_arch as m
from make_arch import BLUE, BLUE_FILL, GRAY, GRAY_FILL, GREEN, GREEN_FILL, INK, RED, RED_FILL, Svg

SUB = "#4A4A4A"
KIND = {"plain": ("white", "#6E6E6E"), "new": (GREEN_FILL, GREEN), "old": (GRAY_FILL, GRAY), "blue": (BLUE_FILL, BLUE), "red": (RED_FILL, RED)}


def box(s, x, y, w, h, title, sub=None, kind="new", size=21):
    fill, stroke = KIND[kind]
    s.rect(x, y, w, h, fill, stroke, 2, 8)
    cy = y + h / 2 + (-4 if sub else 7)
    s.text(x + w / 2, cy, title, size, INK, weight="bold")
    if sub:
        s.text(x + w / 2, cy + 22, sub, 16, SUB)


def label(s, x, y, t, anchor="start", size=17, color=INK):
    s.text(x, y, t, size, color, anchor=anchor)


def slot(s, x, y, w, h, kind, t=None, size=17):
    if kind == "ghost":
        s.out.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="6" fill="url(#hatch)" stroke="{GRAY}" '
                     f'stroke-width="1.8" stroke-dasharray="5 4"/>')
    else:
        fill, stroke = {"expert": (BLUE_FILL, BLUE), "restored": (GREEN_FILL, GREEN)}[kind]
        s.rect(x, y, w, h, fill, stroke, 2.2 if kind == "restored" else 1.8, 6)
    if t:
        s.text(x + w / 2, y + h / 2 + 6, t, size, INK, weight="bold" if kind == "restored" else "normal")


def cyl(s, x, y, w, h, title, sub=None, kind="old", size=19):
    """A memory or data object: a stack of two sheets with square corners."""
    fill, stroke = KIND[kind]
    d = 5 if w > 40 else 3
    s.rect(x + d, y - d, w, h, fill, stroke, 1.6, 0)
    s.rect(x, y, w, h, fill, stroke, 2, 0)
    if not title:
        return
    cy = y + h / 2 + (-4 if sub else 7)
    s.text(x + w / 2, cy, title, size, INK, weight="bold")
    if sub:
        s.text(x + w / 2, cy + 22, sub, 16, SUB)


def legend(s, y, ghost=True):
    s.rect(40, y, 22, 16, GREEN_FILL, GREEN, 1.8, 4)
    label(s, 70, y + 14, "added by FT")
    s.rect(190, y, 22, 16, GRAY_FILL, GRAY, 1.8, 4)
    label(s, 220, y + 14, "existing")
    s.rect(330, y - 2, 26, 20, "white", INK, 1.6, 7)
    label(s, 364, y + 14, "process or kernel")
    cyl(s, 520, y - 2, 26, 20, "", None, "old")
    label(s, 556, y + 14, "memory")
    if ghost:
        slot(s, 650, y - 4, 20, 24, "ghost")
        label(s, 678, y + 14, "ghost slot")


def architecture():
    s = Svg(1000, 610)
    for y, h, name in ((20, 132, "CPU"), (162, 398, "GPU")):
        s.rect(20, y, 960, h, "#FAFAFA", "#E2E2E2", 1.5, 12)
        s.text(32, y + 28, name, 18, INK, anchor="start", weight="bold")
    box(s, 44, 56, 196, 64, "Failure detection", "ULFM, heartbeats", "new", 18)
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
    cyl(s, 60, 234, 330, 56, "Descriptor", "live-rank mask, abort flag", "new")
    cyl(s, 440, 234, 340, 56, "EPLB routing table", "expert to slot")
    s.arrow(160, 120, 160, 226, "green")
    label(s, 171, 210, "sets abort flag", size=16)
    s.path("M 610 120 L 610 188 L 330 188 L 330 226", "green")
    label(s, 470, 180, "writes mask", "middle", 16)
    s.arrow(750, 120, 750, 226, "green")
    label(s, 640, 210, "writes routing", size=16)
    # The captured MoE layer reads both on every launch.
    s.rect(60, 330, 720, 210, "white", "#BDBDBD", 1.6, 12, dash="7 5")
    s.text(74, 530, "CUDA graph", 17, INK, anchor="start", weight="bold")
    box(s, 80, 410, 130, 52, "Dispatch", None, "old", 18)
    s.rect(240, 350, 380, 160, GRAY_FILL, GRAY, 2, 8)
    s.text(430, 376, "Experts", 19, INK, weight="bold")
    for i in range(10):
        slot(s, 252 + i * 36, 392, 30, 96, "ghost" if i >= 8 else "expert")
    box(s, 650, 410, 115, 52, "Combine", None, "old", 18)
    s.arrow(210, 436, 236, 436)
    s.arrow(620, 436, 646, 436)
    s.arrow(145, 290, 145, 326)
    s.arrow(520, 290, 520, 346)
    # Restoring a lost expert into a ghost slot: weights from the host copies, memory from the KV cache.
    s.path("M 882 117 L 882 312 L 591 312 L 591 388", "green")
    label(s, 892, 252, "copies", size=16)
    label(s, 892, 272, "weights", size=16)
    cyl(s, 810, 420, 145, 50, "KV cache")
    s.path("M 882 470 L 882 522 L 555 522 L 555 492", "green")
    label(s, 892, 494, "lends", size=16)
    label(s, 892, 514, "memory", size=16)
    legend(s, 580)
    return s.render()


def band(s, y, h, name, x=20, w=960, color=INK):
    s.rect(x, y, w, h, "#FAFAFA", "#E2E2E2", 1.5, 12)
    s.text(x + 14, y + 28, name, 18, color, anchor="start", weight="bold")


def ghost_slots():
    s = Svg(1000, 470)
    band(s, 16, 244, "GPU 0 (restores 1 of rank 2's 8 lost experts)")
    band(s, 274, 180, "")
    s.text(500, 302, "Expert backups in host memory", 18, INK, weight="bold")
    x0, sw, gap = 196, 70, 7
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
            s.text(x + sw / 2, 103, "lost", 13, INK)
        else:
            slot(s, x, 62, sw, 50, "ghost")
            s.text(x + sw / 2, 132, "unmapped", 14, INK)
    kx = xs[8]
    # The KV cache: most of GPU memory, drawn open-ended. It lends one block, mapped behind the restored slot.
    ky, kh = 194, 56
    s.rect(x0, ky, xs[9] + sw - x0, kh, GRAY_FILL, GRAY, 1.6, 4)
    s.text(x0 + 300, ky + 34, "KV cache", 18, INK, weight="bold")
    s.rect(kx + 5, ky + 5, sw - 10, kh - 10, GREEN_FILL, GREEN, 2.4, 3)
    s.arrow(kx + sw / 2, 114, kx + sw / 2, ky + 1, "green")
    label(s, kx + sw / 2 + 8, 174, "CUDA VMM", "start", 14)

    # Without online EPLB, each node's Grace memory keeps the NEXT node's experts (a ring); node 7 keeps node 0's.
    ny, nh, nw = 316, 118, 266
    nodes = ((40, 0, "e32-e63", 1), (326, 1, "e64-e95", 2), (672, 7, "e0-e31", 0))
    for nx, k, held, owner in nodes:
        s.rect(nx, ny, nw, nh, "white", GRAY, 1.8, 10)
        s.text(nx + 12, ny + 24, f"Node {k}", 16, INK, anchor="start", weight="bold")
        for j in range(4):
            r = 4 * k + j
            dead = r == 2
            x = nx + 8 + j * 64
            s.rect(x, ny + 34, 58, 28, RED_FILL if dead else GRAY_FILL, RED if dead else GRAY, 1.8, 4)
            s.text(x + 29, ny + 53, f"GPU {r}", 13, INK, weight="bold")
        cyl(s, nx + 8, ny + 78, 244, 28, f"Grace memory: Node {owner}'s {held}", None, "new" if k == 7 else "old", 14)
    s.text(632, ny + nh / 2 + 8, "...", 26, INK, weight="bold")
    # GPU 0 of node 0 and the top panel share a dark outline: the panel is that GPU.
    s.rect(48, ny + 34, 58, 28, BLUE_FILL, BLUE, 2.4, 4)
    s.text(77, ny + 53, "GPU 0", 13, INK, weight="bold")
    s.rect(20, 16, 960, 244, "none", BLUE, 2.4, 12)
    for xa, xb in ((48, 20), (106, 380)):
        s.out.append(f'<line x1="{xa}" y1="{ny + 34}" x2="{xb}" y2="260" stroke="{BLUE}" stroke-width="1.5" '
                     f'stroke-dasharray="4 4"/>')
    # Rank 2's e16 is restored from node 7's copy, over NVLink, into rank 0's ghost slot.
    cx = 672 + nw / 2
    s.path(f"M {cx} {ny} L {cx} 266 L {kx + sw / 2} 266 L {kx + sw / 2} {ky + kh - 1}", "green")
    label(s, cx + 10, 304, "e16 over NVLink", "start", 15)
    return s.render()


def late_writes():
    """The MoE all-to-all of one layer, and a zoom on the part FT changes: the token rows in the receive workspace."""
    s = Svg(1000, 426)
    band(s, 16, 150, "One MoE layer on one GPU (captured in the CUDA graph)")
    y, h = 70, 60
    stages = ((30, 116, "Router", "top 8 experts", "old"), (166, 110, "Dispatch", "NVLink", "old"),
              (296, 200, None, None, None), (516, 162, "moe_sort", "group by local expert", "old"),
              (698, 144, "Expert GEMMs", "CuteDSL, NVFP4", "new"), (862, 104, "Combine", "NVLink", "old"))
    for x, w, t, sub, kind in stages:
        if t:
            box(s, x, y, w, h, t, sub, kind, 18)
    cyl(s, 296, y + 4, 200, h - 8, "Receive workspace", "hidden states, routing", "new", 18)
    for (x1, w1, *_), (x2, *_) in zip(stages, stages[1:]):
        s.arrow(x1 + w1 + (8 if x1 == 296 else 0), y + h / 2, x2 - 4, y + h / 2)
    # Zoom on the token rows.
    for x_top, x_bot in ((296, 20), (501, 980)):
        s.out.append(f'<line x1="{x_top}" y1="{y + h - 4}" x2="{x_bot}" y2="186" stroke="#C8C8C8" '
                     f'stroke-width="1.5" stroke-dasharray="4 4"/>')
    band(s, 186, 224, "")
    # The FT half is tinted green, like the parts FT changes in the pipeline above.
    s.out.append('<path d="M 512 187 L 968 187 Q 979 187 979 198 L 979 398 Q 979 409 968 409 L 512 409 Z" '
                 'fill="#F1F8E6" stroke="none"/>')
    s.out.append('<line x1="512" y1="187" x2="512" y2="409" stroke="#D5D5D5" stroke-width="1.5"/>')
    bh = 32

    def block(x, w, yy, kind, t=None):
        if kind == "dead":
            s.out.append(f'<rect x="{x}" y="{yy}" width="{w - 3}" height="{bh}" rx="4" fill="url(#hatch)" '
                         f'stroke="#C9C9C9" stroke-width="1.5"/>')
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
    for x, title, color in ((300, "Without FT: packed", INK), (755, "With FT: fixed mailboxes", GREEN)):
        s.text(x, 214, title, 17, color, weight="bold")
    s.text(525, ymask + 21, "mask", 15, INK, anchor="start", weight="bold")
    for yy, t in ((yN, "step N"), (yM, "step N+1")):
        s.text(40, yy + 21, t, 15, INK, anchor="start", weight="bold")
    # Packed: offsets depend on every sender's count, so step N+1's rank 1 rows sit where rank 2 wrote in step N.
    x0, w4, n1, n2 = 130, 360, 38, 76
    for yy in (yN, yM):
        s.rect(x0, yy, w4, bh, "white", "#DADADA", 1.3, 4)
    for r in range(4):
        block(x0 + r * n1, n1, yN, "live", r)
        block(x0 + r * n2, n2, yM, "dead" if r == 2 else "live", None if r == 1 else r)
    block(x0 + 2 * n1, n1, yM, "hit")
    s.text(x0 + n2 + 1.5 * n1, yM + 22, "1", 15, INK, weight="bold")
    late(x0 + 2.5 * n1, yN + bh + 2, yM)
    # Fixed mailboxes: rank 2's rows only ever land in mailbox 2, which the mask skips.
    x1, cap = 560, 98
    for yy in (yN, yM):
        for r in range(4):
            s.rect(x1 + r * cap, yy, cap - 5, bh, "white", "#BDBDBD", 1.3, 4)
    for r in range(4):
        block(x1 + r * cap, n1, yN, "live", r)
        block(x1 + r * cap, n2 * 0.8, yM, "dead" if r == 2 else "live", None if r == 2 else r)
    block(x1 + 2 * cap, n1, yM, "hit")
    late(x1 + 2 * cap + n1 / 2, yN + bh + 2, yM)
    for r in range(4):
        live = r != 2
        cx = x1 + r * cap + (cap - 5) / 2
        s.rect(cx - 14, ymask, 28, 26, GREEN_FILL if live else RED_FILL, GREEN if live else RED, 1.5, 4)
        s.text(cx, ymask + 19, "1" if live else "0", 14, INK, weight="bold")
    return s.render()


def sequence():
    """Three phases, left to right; steps top to bottom. Colour shows who acts."""
    s = Svg(1000, 312)
    kinds = {"S": "new", "A": "blue", "R": "red"}
    cols = ((20, "Stop"), (350, "Agree"), (680, "Apply"))
    for x, t in cols:
        band(s, 16, 258, t, x=x, w=300)

    def step(c, i, who, title, sub=None):
        box(s, cols[c][0] + 30, 60 + i * 72, 240, 50, title, sub, kinds[who], 18)

    def link(c, i, message=False, t=None):
        x = cols[c][0] + 150
        y1, y2 = 60 + i * 72 + 50, 60 + (i + 1) * 72 - 4
        s.arrow(x, y1, x, y2, "gray" if message else "ink", dash="6 5" if message else None)
        if t:
            label(s, x + 10, (y1 + y2) / 2 + 6, t, "start", 15)

    def across(c, i_from, i_to, t):
        x1, y1 = cols[c][0] + 270, 60 + i_from * 72 + 25
        x2, y2 = cols[c + 1][0] + 30, 60 + i_to * 72 + 25
        xm = (x1 + x2) / 2
        s.out.append(f'<path d="M {x1} {y1} L {xm} {y1} L {xm} {y2} L {x2 - 4} {y2}" fill="none" stroke="{GRAY}" '
                     f'stroke-width="2.2" stroke-dasharray="6 5" marker-end="url(#gray)"/>')
        s.out.append(f'<text transform="translate({xm - 9},{(y1 + y2) / 2}) rotate(-90)" font-size="15" '
                     f'fill="{INK}" text-anchor="middle">{t}</text>')

    step(0, 0, "R", "Rank 2 dies", "seen via ULFM")
    step(0, 1, "S", "Detect, stop", "abort flag, discard steps")
    step(0, 2, "S", "Shrink", "agree on failed ranks")
    link(0, 0)
    link(0, 1)
    step(1, 0, "A", "Fence, propose", "kill rank 2, generation 1")
    step(1, 1, "S", "Plan", "new placement, map memory")
    step(1, 2, "A", "Check", "placement hashes match")
    link(1, 0, True)
    link(1, 1, True, "all Ready")
    step(2, 0, "S", "Apply", "copy weights, route, mask")
    step(2, 1, "A", "Commit", "generation 1")
    step(2, 2, "S", "Resume", "same CUDA graphs")
    link(2, 0, True, "all Installed")
    link(2, 1, True)
    across(0, 2, 0, "all Stopped")
    across(1, 2, 0, "Install")
    for x, kind, t in ((40, "new", "survivors"), (170, "blue", "recovery authority")):
        fill, stroke = KIND[kind]
        s.rect(x, 300, 22, 14, fill, stroke, 1.8, 3)
        label(s, x + 30, 312, t, size=15)
    s.arrow(720, 307, 760, 307, "ink")
    label(s, 768, 312, "local", size=15)
    s.arrow(830, 307, 870, 307, "gray", dash="6 5")
    label(s, 878, 312, "ZeroMQ", size=15)
    return s.render()


def placement():
    """Where rank 2's and node 0's experts go (restore_lost_experts on the static 256 + 64 layout)."""
    s = Svg(1000, 312)
    rank_loss = {0: 1, 1: 1, 3: 1, 4: 1, 5: 1, 6: 1, 7: 1, 8: 1}
    node_loss = {r: 1 for r in range(4, 32)}
    node_loss.update({4: 2, 5: 2, 6: 2, 7: 2})
    x0, w, gap = 150, 20, 3
    for y, title, dead, extra in ((28, "Rank 2 lost", {2}, rank_loss), (160, "Node 0 lost", {0, 1, 2, 3}, node_loss)):
        s.text(30, y + 62, title, 17, INK, anchor="start", weight="bold")
        for r in range(32):
            x = x0 + r * (w + gap) + (r // 4) * 10
            if r % 4 == 0 and y < 100:
                s.text(x + 2 * (w + gap) - gap / 2, y + 14, f"node {r // 4}", 13, INK)
            yy = y + 26
            if r in dead:
                s.rect(x, yy + 44, w, 22, RED_FILL, RED, 1.6, 3)
            else:
                s.rect(x, yy + 44, w, 22, GRAY_FILL, GRAY, 1.4, 3)
                for k in range(extra.get(r, 0)):
                    s.rect(x, yy + 18 - k * 24, w, 22, GREEN_FILL, GREEN, 1.8, 3)
            s.text(x + w / 2, yy + 84, str(r), 11, INK)
    s.rect(150, 288, 18, 14, GRAY_FILL, GRAY, 1.4, 3)
    label(s, 174, 300, "GPU (rank)", size=14)
    s.rect(290, 288, 18, 14, RED_FILL, RED, 1.6, 3)
    label(s, 314, 300, "lost", size=14)
    s.rect(370, 288, 18, 14, GREEN_FILL, GREEN, 1.8, 3)
    label(s, 394, 300, "one restored expert", size=14)
    return s.render()


if __name__ == "__main__":
    import sys

    import cairosvg

    out = sys.argv[1]
    for name, draw in (("fig1_architecture", architecture), ("fig2_ghost_slots", ghost_slots),
                       ("fig3_late_writes", late_writes), ("fig5_placement", placement), ("fig4_recovery_sequence", sequence)):
        svg = draw()
        open(f"{out}/{name}.svg", "w").write(svg)
        cairosvg.svg2png(bytestring=svg.encode(), write_to=f"{out}/{name}.png", scale=2)
    print("ok")
