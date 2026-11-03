"""Slide versions of two design figures, in the design figures' style with fewer words and larger text.

detect_talk: how a lost rank is detected (code at v5-final: communicator.py PeerFailureWatch polls ULFM every
peer_failure_poll_s = 20 ms and sets the abort word; hang_monitor.py confirms a rank every live peer waits on for
hang_confirm_s = 30 s; the authority then fences it).
phases_rise_talk: the RISE phases (v7 design doc).
usage: FONTCONFIG_FILE=<conf with Lato> PYTHONPATH=<cairosvg> python3 make_talk_design.py <out dir>
"""
import sys

import cairosvg

from fig_design_final import (DARK_GREEN, MID_MARKER, SLATE, auth_box, band, box, box3, dashed_line, ft_swatch, label,
                              message, restyle)
from make_arch import GRAY, GRAY_FILL, RED, RED_FILL, Svg

OUT_PX = 2200


def detect():
    """Crash path along the bottom; the opt-in hang path feeds into it from the top left; legend top right."""
    s = Svg(1000, 252)
    s.out.append(MID_MARKER)
    box(s, 30, 30, 180, 60, "Rank 2 hangs", "opt-in detection", "red", 20, dash="7 4")
    message(s, 210, 60, 250, 60)
    auth_box(s, 250, 28, 200, 64, "Authority", "all peers stuck 30 s")
    s.arrow(350, 92, 350, 146)
    label(s, 360, 125, "kill", "start", 16)
    box(s, 250, 150, 200, 60, "Rank 2 dies", "killed or crashed", "red", 20, dash="7 4")
    s.arrow(450, 180, 486, 180)
    box(s, 490, 150, 140, 60, "ULFM", "marks it failed", "new", 20)
    s.arrow(630, 180, 666, 180)
    box(s, 670, 150, 150, 60, "Failure watch", "every 20 ms", "new", 20)
    s.path("M 820 170 L 834 170 L 834 148 L 846 148", "green")
    s.path("M 820 190 L 834 190 L 834 212 L 846 212", "green")
    box(s, 850, 124, 140, 48, "Abort flag", "kernels return", "new", 18)
    box(s, 850, 188, 140, 48, "Recovery", "survivors recover", "new", 18)
    lx, ly = 560, 34
    ft_swatch(s, lx, ly)
    label(s, lx + 30, ly + 14, "added by FT", size=16)
    s.rect(lx + 220, ly, 22, 16, RED_FILL, RED, 1.8, 4, dash="4 2")
    label(s, lx + 250, ly + 14, "lost rank", size=16)
    s.rect(lx, ly + 34, 22, 16, "#EAF4D6", SLATE, 2.6, 4)
    label(s, lx + 30, ly + 48, "own process", size=16)
    dashed_line(s, lx + 220, ly + 42, lx + 248, ly + 42)
    label(s, lx + 256, ly + 48, "ZeroMQ message", size=16)
    return s.render()


def phases_rise():
    s = Svg(900, 114)
    rows = [("Phase 1 (this talk)", ("Survive a loss", "keep serving on N−k"), "new"),
            ("Phase 2", ("Restore back to N", "lost ranks rejoin"), "plain"),
            ("Phase 3", ("Planned scaling", "grow or shrink, N ↔ M"), "plain")]
    w, h, gap, y = 250, 90, 45, 12
    for i, (title, lines, kind) in enumerate(rows):
        x = 20 + i * (w + gap)
        box3(s, x, y, w, h, title, lines, kind, 22)
        if i:
            s.arrow(x - gap + 6, y + h / 2, x - 6, y + h / 2)
    return s.render()


if __name__ == "__main__":
    out = sys.argv[1]
    for name, draw in (("detect_talk", detect), ("phases_rise_talk", phases_rise)):
        svg = restyle(draw())
        open(f"{out}/{name}.svg", "w").write(svg)
        width = float(svg.split('width="', 1)[1].split('"', 1)[0])
        cairosvg.svg2png(bytestring=svg.encode(), write_to=f"{out}/{name}.png", scale=OUT_PX / width)
    print("ok")
