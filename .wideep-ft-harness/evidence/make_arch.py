"""Write integration.svg (architecture and recovery phases) and late_writes.svg (fixed source regions)."""

from html import escape

FONT = 'font-family="Helvetica, Arial, sans-serif"'
NEW, REUSED, AUTH, GPU, DEAD = "#76b900", "#757575", "#1a1a1a", "#1a1a1a", "#d32f2f"
SRC = ["#f5f5f5", "#dedede"] * 3


class Svg:
    def __init__(self, w, h):
        self.w, self.h = w, h
        self.out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" {FONT}>',
                    '<rect width="100%" height="100%" fill="white"/>', "<defs>"]
        for name, color in (("blue", NEW), ("purple", AUTH), ("orange", GPU), ("red", DEAD),
                            ("gray", "#555"), ("green", REUSED)):
            self.out.append(f'<marker id="{name}" markerWidth="11" markerHeight="11" refX="9" refY="5.5" '
                            f'orient="auto-start-reverse" markerUnits="userSpaceOnUse">'
                            f'<path d="M0,0 L11,5.5 L0,11 z" fill="{color}"/></marker>')
        self.out.append('<pattern id="hatch" width="8" height="8" patternUnits="userSpaceOnUse" '
                        'patternTransform="rotate(45)"><rect width="8" height="8" fill="#eeeeee"/>'
                        '<line x1="0" y1="0" x2="0" y2="8" stroke="#bdbdbd" stroke-width="4"/></pattern>')
        self.out.append("</defs>")

    def text(self, x, y, s, size=14, color="#222", anchor="middle", weight="normal", italic=False):
        style = ' font-style="italic"' if italic else ""
        self.out.append(f'<text x="{x}" y="{y}" font-size="{size}" fill="{color}" text-anchor="{anchor}" '
                        f'font-weight="{weight}"{style}>{escape(s)}</text>')

    def box(self, x, y, w, h, fill="white", stroke="#999", rx=8, width=1.5, dash=None):
        d = f' stroke-dasharray="{dash}"' if dash else ""
        self.out.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" '
                        f'stroke="{stroke}" stroke-width="{width}"{d}/>')

    def chip(self, x, y, w, h, label, color, sub=None, filled=False):
        self.box(x, y, w, h, color if filled else "white", color, 6, 2)
        fg = "white" if filled else ("#4a7d00" if color == NEW else color)
        if sub:
            self.text(x + w / 2, y + h / 2 - 2, label, 14, fg, weight="bold")
            self.text(x + w / 2, y + h / 2 + 15, sub, 11.5, fg)
        else:
            self.text(x + w / 2, y + h / 2 + 5, label, 14, fg, weight="bold")

    def arrow(self, x1, y1, x2, y2, color="gray", width=2, dash=None, both=False):
        c = {"blue": NEW, "purple": AUTH, "orange": GPU, "red": DEAD, "gray": "#555", "green": REUSED}[color]
        d = f' stroke-dasharray="{dash}"' if dash else ""
        start = f' marker-start="url(#{color})"' if both else ""
        self.out.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{c}" stroke-width="{width}"'
                        f'{d} marker-end="url(#{color})"{start}/>')

    def line(self, x1, y1, x2, y2, color, width=2, dash=None):
        d = f' stroke-dasharray="{dash}"' if dash else ""
        self.out.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="{width}"{d}/>')

    def cross(self, x, y, r=10, color=DEAD, width=4):
        self.line(x - r, y - r, x + r, y + r, color, width)
        self.line(x - r, y + r, x + r, y - r, color, width)

    def render(self):
        return "\n".join(self.out + ["</svg>"])


def integration():
    s = Svg(1200, 700)
    s.text(30, 36, "WideEP FT architecture", 22, anchor="start", weight="bold")
    s.chip(990, 18, 92, 26, "reused", REUSED, filled=True)
    s.chip(1092, 18, 78, 26, "new", NEW, filled=True)

    # Recovery authority: outside the worker job, above the executor it talks to.
    s.box(60, 62, 360, 58, "white", AUTH, 10, 2)
    s.text(240, 88, "Recovery authority", 17, AUTH, weight="bold")
    s.text(240, 108, "separate CPU process, outside mpirun", 12.5, AUTH)

    # One surviving rank, in detail.
    s.box(30, 140, 760, 350, "white", "#bdbdbd", 12, 1.5)
    s.text(780, 162, "each surviving rank", 13, "#777", anchor="end", weight="bold")
    s.arrow(170, 122, 170, 196, "purple", 2.5, "7 5", both=True)
    s.text(182, 165, "ZeroMQ + HMAC", 12.5, AUTH, anchor="start", weight="bold")
    s.text(40, 232, "CPU", 12, "#888", anchor="start", weight="bold")
    s.chip(80, 200, 180, 56, "PeerLossRecovery", NEW, "executor hook")
    s.chip(280, 200, 150, 56, "EPLB", REUSED, "mask-only API")
    s.chip(450, 200, 150, 56, "EPGroupHealth", REUSED, "live-rank mask")
    s.chip(620, 200, 150, 56, "ULFM MPI", NEW, "watcher, Shrink")

    s.line(40, 285, 780, 285, "#bdbdbd", 1.5, "4 4")
    s.chip(80, 296, 180, 42, "abort flag", NEW, "pinned host memory")
    s.chip(330, 296, 270, 42, "execution descriptor", NEW, "mask | abort-flag address")
    s.arrow(170, 258, 170, 293, "blue", 2)
    s.arrow(355, 258, 355, 293, "green", 2)
    s.arrow(525, 258, 525, 293, "green", 2)
    s.line(262, 317, 328, 317, NEW, 2, "4 3")
    s.text(40, 372, "GPU", 12, "#888", anchor="start", weight="bold")

    # Captured CUDA graph: one MoE layer reading the descriptor.
    s.box(80, 356, 690, 120, "#f7f7f7", "#9e9e9e", 10, 1.5)
    stages = [("dispatch", 100), ("gather", 260), ("experts", 420), ("combine", 580)]
    s.line(465, 340, 465, 368, NEW, 2)
    s.line(170, 368, 650, 368, NEW, 2)
    for name, x in stages:
        s.chip(x, 384, 140, 50, name, NEW if name == "gather" else "#1a1a1a", None)
        if name in ("dispatch", "combine"):
            s.arrow(x + 70, 368, x + 70, 381, "blue", 2)
    for (_, x1), (_, x2) in zip(stages, stages[1:]):
        s.arrow(x1 + 142, 409, x2 - 3, 409, "orange", 2)
    s.text(425, 464, "CUDA graph, captured once at startup", 13, "#555", weight="bold")

    # The other ranks: EP32 grid with the lost rank.
    gx, gy, cell = 870, 190, 34
    s.text(1006, 176, "32 ranks", 15, "#444", weight="bold")
    for i in range(32):
        x, y = gx + (i % 8) * cell, gy + (i // 8) * cell
        dead = i == 2
        s.box(x, y, cell - 6, cell - 6, "#ffebee" if dead else "white", DEAD if dead else "#9e9e9e", 4, 1.5)
        if dead:
            s.cross(x + 14, y + 14, 8)
        else:
            s.text(x + 14, y + 19, str(i), 11, "#455a64")
    s.text(1006, gy + 4 * cell + 20, "rank 2 lost", 13, DEAD, weight="bold")
    s.arrow(774, 228, 864, 228, "blue", 2.5, both=True)
    s.text(818, 218, "MPI", 12, NEW, weight="bold")
    s.arrow(774, 420, 905, 340, "orange", 2.5, both=True)
    s.text(872, 410, "NVLink", 12, GPU, anchor="start", weight="bold")
    s.text(872, 425, "all-to-all", 12, GPU, anchor="start", weight="bold")

    # Recovery phases.
    py, ph = 545, 105
    phases = [
        ("1  Cancel the attempt", "#1a1a1a", ["abort flag: kernels exit", "Revoke: collectives exit", "step discarded"],
         "Stopped"),
        ("2  Install (replay paused)", "#1a1a1a", ["Shrink: agree on lost rank", "EPLB: route to replicas",
                                                    "write mask in place"], "Propose · Ready · Install · Installed"),
        ("3  Permission to resume", "#1a1a1a", ["only after every Installed", "same CUDA graphs"], "Commit"),
    ]
    w = 365
    for i, (title, color, rows, msgs) in enumerate(phases):
        x = 30 + i * (w + 18)
        tip = 22
        pts = f"{x},{py} {x + w},{py} {x + w + tip},{py + ph / 2} {x + w},{py + ph} {x},{py + ph}"
        if i:
            pts += f" {x + tip},{py + ph / 2}"
        s.out.append(f'<polygon points="{pts}" fill="#f7f7f7" stroke="#9e9e9e" '
                     f'stroke-width="2"/>')
        s.text(x + 38, py + 28, title, 16, color, anchor="start", weight="bold")
        for j, r in enumerate(rows):
            s.text(x + 38, py + 52 + j * 19, r, 13, "#333", anchor="start")
        s.text(x + w / 2 + 10, py + ph + 24, msgs, 13, "#555", weight="bold", italic=True)
    s.text(30, 528, "Recovery", 15, "#444", anchor="start", weight="bold")
    return s.render()


def late_writes():
    s = Svg(1200, 440)
    s.text(30, 36, "Late writes from a lost rank", 22, anchor="start", weight="bold")
    s.text(30, 60, "One receiver's dispatch workspace; each cell is a token row, numbered by the source rank "
           "that writes it", 13.5, "#666", anchor="start")
    cw, x0 = 44, 190

    def bar(y, owners, hatch=None, label=None):
        for i, k in enumerate(owners):
            fill = "url(#hatch)" if k == hatch else SRC[k]
            s.box(x0 + i * cw, y, cw - 4, 38, fill, "white", 3, 1)
            s.text(x0 + i * cw + cw / 2 - 2, y + 25, str(k), 15, "#333", weight="bold")
        if label:
            s.text(x0 - 14, y + 25, label, 14, "#555", anchor="end")

    def lost_rank(x, y):
        s.box(x, y, 78, 42, "#ffebee", DEAD, 6, 2)
        s.text(x + 39, y + 27, "rank 2", 14, DEAD, weight="bold")
        s.cross(x + 74, y + 4, 8)

    # Main: the layout depends on the step's runtime_max_tokens_per_rank.
    s.text(30, 112, "Main", 18, anchor="start", weight="bold")
    s.text(30, 132, "rows move", 13, "#666", anchor="start")
    s.text(30, 148, "every step", 13, "#666", anchor="start")
    bar(92, [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2], label="step n")
    bar(160, [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5], label="step n+1")
    target = x0 + 8 * cw
    s.box(target - 3, 157, cw + 2, 44, "none", DEAD, 4, 3)
    s.box(target - 3, 89, cw + 2, 44, "none", DEAD, 4, 2, "5 4")
    lost_rank(760, 120)
    s.arrow(760, 150, target + cw - 6, 166, "red", 2.5)
    s.text(800, 186, "a write issued in step n", 13, DEAD, anchor="start")
    s.text(800, 203, "lands in rank 4's row in step n+1", 13, DEAD, anchor="start", weight="bold")

    s.line(30, 238, 1170, 238, "#e0e0e0", 1.5)

    # FT: one fixed region per source.
    s.text(30, 290, "FT", 18, anchor="start", weight="bold")
    s.text(30, 310, "fixed region", 13, "#666", anchor="start")
    s.text(30, 326, "per source", 13, "#666", anchor="start")
    bar(276, [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5], hatch=2, label="every step")
    region = x0 + 4 * cw
    s.box(region - 3, 273, 2 * cw + 2, 44, "none", DEAD, 4, 3)
    lost_rank(region + 6, 360)
    s.arrow(region + 45, 358, region + 45, 322, "red", 2.5)
    s.text(region + 100, 380, "stays in rank 2's own region, never read again", 13, "#4a7d00", anchor="start",
           weight="bold")
    s.text(region + 100, 398, "completion flags, receive counters and EPLB stats are per-source slots too",
           13, "#555", anchor="start")
    s.text(region + 100, 416, "buffers stay mapped until teardown", 13, "#555", anchor="start")

    # The gather kernel turns the fixed regions into the dense expert input.
    s.arrow(x0 + 12 * cw + 6, 295, 790, 295, "gray", 2)
    s.chip(795, 268, 150, 54, "gather (new)", NEW, "skips rank 2")
    s.arrow(947, 295, 995, 295, "gray", 2)
    s.chip(1000, 268, 160, 54, "expert GEMMs", "#1a1a1a", "dense input")
    return s.render()


if __name__ == "__main__":
    open("integration.svg", "w").write(integration())
    open("late_writes.svg", "w").write(late_writes())
