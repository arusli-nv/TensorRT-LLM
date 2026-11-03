"""Slide figure for the scoping slide: the three phases of WideEP fault tolerance in the RISE plan (v7 design doc:
"Phase 1 — Survive and continue at N−K"; "Phase 2 restores physical capacity to N; Phase 3 generalizes planned
scaling"), in the design figures' style.

usage: FONTCONFIG_FILE=<conf with Lato> PYTHONPATH=<cairosvg> python3 make_fig_phases.py <out dir>
"""
import sys

import cairosvg

from fig_design_final import DARK_GREEN, SLIDE_SCALE, box3, restyle
from make_arch import Svg


def phases():
    s = Svg(680, 380)
    rows = [("Phase 1", ("Survive a loss", "keep serving on N−k ranks"), "new"),
            ("Phase 2", ("Restore back to N", "lost ranks rejoin"), "plain"),
            ("Phase 3", ("Planned scaling", "grow or shrink: N ↔ M"), "plain")]
    x, w, h, gap = 40, 460, 86, 40
    for i, (title, lines, kind) in enumerate(rows):
        y = 20 + i * (h + gap)
        box3(s, x, y, w, h, title, lines, kind, 20)
        if i:
            s.arrow(x + w / 2, y - gap + 4, x + w / 2, y - 6)
    s.text(x + w + 22, 20 + h / 2 + 6, "this talk", 18, DARK_GREEN, anchor="start", weight="bold")
    return s.render()


if __name__ == "__main__":
    svg = restyle(phases())
    name = "fig_rise_phases_slide"
    open(f"{sys.argv[1]}/{name}.svg", "w").write(svg)
    cairosvg.svg2png(bytestring=svg.encode(), write_to=f"{sys.argv[1]}/{name}.png", scale=SLIDE_SCALE)
    print("ok")
