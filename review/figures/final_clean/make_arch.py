"""Design diagrams as SVG + PNG, in the style of the NVIDIA technical blog figures.

Green tint: added by FT. Gray tint: existing TensorRT-LLM. Red: the lost rank. Captions live in the
document, not in the image. Renders with Lato (fonts/), registered through FONTCONFIG_FILE.
Usage: FONTCONFIG_FILE=<conf with fonts/> PYTHONPATH=<cairosvg> python3 make_arch.py
"""

from html import escape

FONT = 'font-family="Lato, Helvetica, Arial, sans-serif"'
GREEN, GREEN_FILL = "#76B900", "#EAF4D6"
GRAY, GRAY_FILL, INK, MUTED = "#8C99A6", "#EEF1F4", "#1A1A1A", "#707070"  # slate: existing, as in the charts
RED, RED_FILL = "#C8372D", "#FBE6E4"
BLUE, BLUE_FILL = "#6B7A89", "#E3E8EE"  # expert slots: existing, slate


class Svg:
    def __init__(self, w, h):
        self.out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" {FONT}>',
                    '<rect width="100%" height="100%" fill="white"/><defs>']
        for name, color in (("ink", INK), ("green", GREEN), ("gray", GRAY), ("red", RED)):
            self.out.append(f'<marker id="{name}" markerWidth="10" markerHeight="10" refX="8" refY="5" '
                            f'orient="auto-start-reverse" markerUnits="userSpaceOnUse">'
                            f'<path d="M0,0 L10,5 L0,10 z" fill="{color}"/></marker>')
        self.out.append('<pattern id="hatch" width="8" height="8" patternUnits="userSpaceOnUse" '
                        'patternTransform="rotate(45)"><rect width="8" height="8" fill="#F4F4F4"/>'
                        '<line x1="0" y1="0" x2="0" y2="8" stroke="#D2D2D2" stroke-width="3"/></pattern></defs>')

    def text(self, x, y, s, size=17, color=INK, anchor="middle", weight="normal"):
        self.out.append(f'<text x="{x}" y="{y}" font-size="{size}" fill="{color}" text-anchor="{anchor}" '
                        f'font-weight="{weight}">{escape(s)}</text>')

    def rect(self, x, y, w, h, fill, stroke, width=2.0, rx=10, dash=None):
        d = f' stroke-dasharray="{dash}"' if dash else ""
        self.out.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" '
                        f'stroke="{stroke}" stroke-width="{width}"{d}/>')

    def box(self, x, y, w, h, title, sub=None, kind="new"):
        fill, stroke, color = {"new": (GREEN_FILL, GREEN, INK), "old": (GRAY_FILL, GRAY, INK),
                               "blue": (BLUE_FILL, BLUE, INK)}[kind]
        self.rect(x, y, w, h, fill, stroke)
        cy = y + h / 2 + (-4 if sub else 7)
        self.text(x + w / 2, cy, title, 19 if len(title) < 12 else 17, color, weight="bold")
        if sub:
            self.text(x + w / 2, cy + 23, sub, 15, MUTED)

    def arrow(self, x1, y1, x2, y2, color="gray", width=2.2, dash=None, both=False):
        c = {"ink": INK, "green": GREEN, "gray": GRAY, "red": RED}[color]
        d = f' stroke-dasharray="{dash}"' if dash else ""
        start = f' marker-start="url(#{color})"' if both else ""
        self.out.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{c}" stroke-width="{width}"{d} '
                        f'marker-end="url(#{color})"{start}/>')

    def path(self, d, color="gray", width=2.2):
        c = {"gray": GRAY, "green": GREEN, "red": RED}[color]
        self.out.append(f'<path d="{d}" fill="none" stroke="{c}" stroke-width="{width}" '
                        f'stroke-linejoin="round" marker-end="url(#{color})"/>')

    def legend(self, x, y):
        self.rect(x, y, 22, 16, GREEN_FILL, GREEN, 1.8, 4)
        self.text(x + 30, y + 13, "added by FT", 15, MUTED, anchor="start")
        self.rect(x + 140, y, 22, 16, GRAY_FILL, GRAY, 1.8, 4)
        self.text(x + 170, y + 13, "existing", 15, MUTED, anchor="start")

    def render(self):
        return "\n".join(self.out + ["</svg>"])


def arch():
    """One surviving GPU: who detects, who decides, and what the captured graphs read."""
    s = Svg(1010, 500)
    s.box(40, 20, 250, 70, "Recovery authority", "separate process", "blue")
    s.arrow(165, 92, 165, 150, dash="6 5", both=True)
    s.text(173, 110, "ZeroMQ", 13, MUTED, anchor="start")
    s.rect(20, 120, 740, 340, "none", "#C9C9C9", 1.6, 16, dash="7 6")
    s.text(744, 146, "each surviving GPU", 15, MUTED, anchor="end")
    s.box(40, 156, 250, 70, "Recovery driver", "in the serving loop")
    s.box(330, 156, 210, 70, "Failure watcher", "C++ thread, MPI ULFM")
    s.arrow(328, 191, 294, 191)
    s.text(311, 182, "loss", 13, MUTED)
    s.box(40, 262, 250, 62, "EPLB", "reroute experts", "old")
    s.arrow(120, 228, 120, 258)
    s.text(128, 248, "new routing", 13, MUTED, anchor="start")
    s.box(330, 262, 210, 62, "Descriptor", "rank mask, abort address")
    s.path("M250,228 L250,244 L400,244 L400,258")
    s.text(258, 240, "new mask", 13, MUTED, anchor="start")
    s.box(580, 262, 160, 62, "Abort flag", "host memory")
    s.path("M520,228 L520,244 L660,244 L660,258")
    s.text(528, 240, "set on a loss", 13, MUTED, anchor="start")
    s.arrow(542, 293, 576, 293, dash="4 4")
    s.rect(40, 360, 700, 84, "#FAFAFA", "#D8D8D8", 1.4, 12)
    for i, (name, kind) in enumerate((("Dispatch", "old"), ("Routing sort", "new"), ("Experts", "old"),
                                      ("Combine", "old"))):
        s.box(56 + i * 172, 374, 150, 50, name, None, kind)
        if i:
            s.arrow(56 + i * 172 - 21, 399, 56 + i * 172 - 3, 399)
    s.text(732, 438, "CUDA graph", 14, MUTED, anchor="end")
    s.arrow(435, 326, 435, 358)
    s.text(427, 346, "mask read every launch", 13, MUTED, anchor="end")
    s.arrow(660, 358, 660, 328)
    s.text(652, 346, "polled on long waits", 13, MUTED, anchor="end")
    s.arrow(131, 326, 131, 370)
    s.text(139, 352, "routes", 13, MUTED, anchor="start")
    gx, gy = 812, 150
    s.text(gx + 82, gy - 14, "32 GPUs", 16, MUTED)
    for r in range(8):
        for c in range(4):
            lost = (r, c) == (0, 2)
            s.rect(gx + c * 42, gy + r * 36, 32, 26, RED_FILL if lost else GRAY_FILL,
                   RED if lost else "#C4C4C4", 1.8 if lost else 1.2, 6)
    s.text(gx + 82, gy + 8 * 36 + 22, "one lost", 16, RED, weight="bold")
    s.arrow(742, 399, 806, 399, both=True)
    s.text(774, 390, "NVLink", 13, MUTED)
    s.legend(40, 474)
    return s.render()


def fixed_regions():
    """The same receive buffer and the same tokens, without and with FT, and one late write from dead rank 2.

    The buffer holds 4 senders x 512 rows (the capacity per rank) either way. Without FT, sender r's rows start
    at r x (the step's max tokens per rank), so they move when that maximum changes; with FT, at r x 512. With
    FT the MoE layer reads the live senders' rows from the slots in place; nothing is copied.
    Bands are labeled with the sending rank's number."""
    s = Svg(1240, 680)
    x0, width, bh = 190, 1020, 50
    cap, n1, n2 = 512, 128, 256
    row = width / (4 * cap)

    def band(x, rows, y, label, kind):
        fill, stroke = {"live": (BLUE_FILL, BLUE), "dead": ("url(#hatch)", "#C9C9C9"),
                        "free": ("#FAFAFA", "#DADADA"), "hit": (RED_FILL, RED)}[kind]
        w = rows * row
        s.rect(x, y, w - 4, bh, fill, stroke, 3 if kind == "hit" else 1.6, 7)
        if label is not None:
            s.text(x + w / 2, y + 34, str(label), 24, MUTED if kind == "dead" else INK, weight="bold")

    def late_write(x, y_from, y_to, label):
        s.out.append(f'<line x1="{x}" y1="{y_from}" x2="{x}" y2="{y_to - 5}" stroke="{RED}" stroke-width="3" '
                     f'stroke-dasharray="8 5" marker-end="url(#red)"/>')
        s.text(x + 14, (y_from + y_to) / 2 + 7, label, 19, RED, anchor="start")

    def step(y, name, sub=None):
        s.text(x0 - 18, y + (25 if sub else 33), name, 19, INK, anchor="end", weight="bold")
        if sub:
            s.text(x0 - 18, y + 46, sub, 15, MUTED, anchor="end")

    # Without FT: rows packed at the front, n1 per sender in step N, n2 in step N+1.
    s.text(30, 42, "Without FT", 26, anchor="start", weight="bold")
    s.text(220, 42, "receive buffer (written by peers over NVLink)", 17, MUTED, anchor="start")
    y1, y2 = 66, 176
    step(y1, "step N")
    band(x0, 4 * cap, y1, None, "free")
    for r in range(4):
        band(x0 + r * n1 * row, n1, y1, r, "live")
    step(y2, "step N+1")
    band(x0, 4 * cap, y2, None, "free")
    for r in range(4):
        band(x0 + r * n2 * row, n2, y2, None if r == 1 else r, "dead" if r == 2 else "live")
    band(x0 + n2 * row, n1, y2, None, "hit")
    s.text(x0 + (n2 + n1 + n1 / 2) * row, y2 + 34, "1", 24, weight="bold")
    late_write(x0 + (2 * n1 + n1 / 2) * row, y1 + bh + 4, y2, "late write (corrupting)")

    # With FT: the same buffer as one fixed slot of `cap` rows per sender.
    y3, y4, y5 = 352, 462, 600
    s.out.append(f'<line x1="30" y1="{y3 - 74}" x2="1210" y2="{y3 - 74}" stroke="#E6E6E6" stroke-width="2"/>')
    s.text(30, y3 - 26, "With FT", 26, GREEN, anchor="start", weight="bold")
    s.text(220, y3 - 26, "receive buffer: one fixed slot per sender", 17, MUTED, anchor="start")
    for y, used, name in ((y3, n1, "step N"), (y4, n2, "step N+1")):
        step(y, name)
        for r in range(4):
            band(x0 + r * cap * row, cap, y, None, "free")
            if r == 2 and used == n2:
                band(x0 + r * cap * row, used, y, None, "dead")
                s.text(x0 + (r * cap + n1 + (n2 - n1) / 2) * row, y + 34, "2", 24, MUTED, weight="bold")
            else:
                band(x0 + r * cap * row, used, y, r, "live")
    band(x0 + 2 * cap * row, n1, y4, None, "hit")
    late_write(x0 + (2 * cap + n1 / 2) * row, y3 + bh + 4, y4, "late write: lands in rank 2's own slot")
    # The experts read the live senders' rows in place; only expert ids and weights are copied, compactly.
    s.rect(x0, y5, 4 * cap * row - 4, bh + 6, GRAY_FILL, GRAY, 2, 10)
    s.text(x0 + 2 * cap * row, y5 + 25, "Experts read each live slot's sent rows in place; slot 2 is skipped", 19,
           weight="bold")
    s.text(x0 + 2 * cap * row, y5 + 48, "the routing sort reads the expert ids and weights from the same slots",
           15, MUTED)
    for r in (0, 1, 3):
        s.arrow(x0 + (r * cap + n2 / 2) * row, y4 + bh + 4, x0 + (r * cap + n2 / 2) * row, y5 - 4, "gray", 2.2)
    s.text(x0 - 18, (y4 + bh + y5) / 2 + 7, "no copy", 17, MUTED, anchor="end")
    return s.render()


def descriptor():
    """What the CUDA graphs capture, who writes the descriptor and abort flag, and when kernels read them."""
    s = Svg(1060, 420)
    # Writers, above.
    s.box(322, 20, 330, 56, "Recovery driver", "between steps, model idle")
    s.box(724, 20, 300, 56, "Failure watcher (C++)", "on a loss, any time")
    s.arrow(487, 78, 487, 116)
    s.text(495, 102, "writes the new rank mask", 13, MUTED, anchor="start")
    s.arrow(874, 78, 874, 156)
    s.text(882, 122, "plain CPU store", 13, MUTED, anchor="start")
    # The captured kernel argument.
    s.rect(30, 150, 220, 70, GRAY_FILL, GRAY, 2, 12)
    s.text(140, 180, "Kernel argument", 18, weight="bold")
    s.text(140, 204, "fixed at graph capture", 15, MUTED)
    s.arrow(252, 185, 318, 185)
    s.text(285, 172, "pointer", 13, MUTED)
    # The descriptor in GPU memory.
    s.rect(322, 120, 330, 160, GREEN_FILL, GREEN, 2, 14)
    s.text(487, 148, "Descriptor", 18, weight="bold")
    s.text(487, 168, "GPU memory", 14, MUTED)
    s.rect(342, 180, 290, 38, "white", GREEN, 1.6, 8)
    s.text(487, 205, "rank mask: one bit per rank", 16)
    s.rect(342, 228, 290, 38, "white", GREEN, 1.6, 8)
    s.text(487, 253, "abort flag address", 16)
    s.arrow(634, 247, 720, 200, dash="5 4")
    # The abort flag in host memory.
    s.rect(724, 160, 300, 76, GREEN_FILL, GREEN, 2, 12)
    s.text(874, 192, "Abort flag", 18, weight="bold")
    s.text(874, 216, "pinned host memory", 14, MUTED)
    # Readers, below.
    s.box(322, 340, 702, 56, "All-to-all kernels in the CUDA graph", "dispatch, combine", "old")
    s.arrow(487, 336, 487, 284)
    s.text(495, 316, "read the mask on every launch", 13, MUTED, anchor="start")
    s.arrow(874, 336, 874, 240)
    s.text(882, 300, "read only after a peer", 13, MUTED, anchor="start")
    s.text(882, 318, "wait passes ~50 µs", 13, MUTED, anchor="start")
    return s.render()


if __name__ == "__main__":
    import cairosvg

    for name, draw in (("arch", arch), ("descriptor", descriptor), ("fixed_regions", fixed_regions)):
        svg = draw()
        open(f"{name}.svg", "w").write(svg)
        cairosvg.svg2png(bytestring=svg.encode(), write_to=f"{name}.png", scale=2)
