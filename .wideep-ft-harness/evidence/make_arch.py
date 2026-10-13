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
    s = Svg(1200, 640)
    s.text(30, 36, "Late writes from a lost rank", 22, anchor="start", weight="bold")
    s.text(30, 60, "One receiver's all-to-all workspace; each cell is a token row, numbered by the source rank "
           "that writes it", 13.5, "#666", anchor="start")

    def gpu_icon(x, y, label, dead=False):
        s.box(x, y, 64, 44, "#ffebee" if dead else "white", DEAD if dead else "#1a1a1a", 6, 2)
        s.text(x + 32, y + 28, label, 13, DEAD if dead else "#1a1a1a", weight="bold")
        if dead:
            s.cross(x + 58, y + 6, 9)

    def bar(x, y, owners, cw, hatch=None, label=None):
        for i, k in enumerate(owners):
            fill = "url(#hatch)" if k == hatch else SRC[k]
            s.box(x + i * cw, y, cw - 3, 36, fill, "white", 3, 1)
            s.text(x + i * cw + cw / 2 - 1, y + 24, str(k), 14, "#333", weight="bold")
        if label:
            s.text(x - 12, y + 24, label, 13.5, "#444", anchor="end", weight="bold")

    # Panel A: main, rows move between steps.
    s.box(20, 80, 560, 250, "#fff", "#e0e0e0", 10)
    s.text(40, 110, "Main: rows move every step", 17, anchor="start", weight="bold")
    cw = 36
    ax = 130
    bar(ax, 140, [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2], cw, label="step n")
    bar(ax, 250, [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5], cw, label="step n+1")
    gpu_icon(500, 190, "rank 2", dead=True)
    tx = ax + 8 * cw + cw / 2
    s.arrow(500, 214, tx + 6, 247, "red", 3)
    s.box(ax + 8 * cw - 3, 247, cw + 3, 42, "none", DEAD, 4, 3)
    s.text(ax + 8.5 * cw, 312, "lands in rank 4's row", 13.5, DEAD, weight="bold")

    # Panel B: FT, one fixed region per source.
    s.box(620, 80, 560, 250, "#fff", "#e0e0e0", 10)
    s.text(640, 110, "FT: a fixed region per source", 17, anchor="start", weight="bold")
    bx = 720
    bar(bx, 195, [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5], cw, hatch=2, label="every step")
    gpu_icon(1100, 120, "rank 2", dead=True)
    s.arrow(1100, 150, bx + 5 * cw - 2, 192, "red", 3)
    s.text(bx + 5 * cw, 260, "stays in its own region,", 13.5, "#4a7d00", weight="bold")
    s.text(bx + 5 * cw, 278, "which is never read or reused", 13.5, "#4a7d00", weight="bold")

    # Every per-source structure is indexed by the source.
    s.text(30, 380, "Everything rank k writes into a peer is indexed by k", 17, anchor="start", weight="bold")
    gx, gy, cwid, rh = 230, 400, 56, 40
    rows = ["payloads", "completion flags", "receive counters", "EPLB statistics"]
    for c in range(6):
        s.text(gx + c * cwid + cwid / 2, gy - 4 + 0, "", 1)
        s.text(gx + c * cwid + cwid / 2, gy + len(rows) * rh + 22, f"src {c}", 12.5, "#666")
    for r, name in enumerate(rows):
        y = gy + r * rh
        s.text(gx - 14, y + 26, name, 14, "#444", anchor="end")
        for c in range(6):
            fill = "url(#hatch)" if c == 2 else SRC[c]
            s.box(gx + c * cwid, y + 3, cwid - 4, rh - 6, fill, "white", 4, 1)
    s.box(gx + 2 * cwid - 4, gy - 2, cwid + 4, len(rows) * rh + 4, "none", DEAD, 6, 2.5)

    # Gather kernel: skips masked sources, produces the dense expert input.
    s.arrow(gx + 6 * cwid + 10, gy + 2 * rh, 640, gy + 2 * rh, "orange", 3)
    s.chip(645, gy + 2 * rh - 30, 150, 60, "gather (new)", NEW, "skips lost rank")
    s.arrow(797, gy + 2 * rh, 860, gy + 2 * rh, "orange", 3)
    s.chip(865, gy + 2 * rh - 30, 170, 60, "expert GEMMs", GPU, "dense input")
    s.text(840, gy + 2 * rh + 70, "Workspaces stay mapped until teardown,", 13, "#444")
    s.text(840, gy + 2 * rh + 88, "so late writes never hit freed memory.", 13, "#444")
    return s.render()


if __name__ == "__main__":
    open("integration.svg", "w").write(integration())
    open("late_writes.svg", "w").write(late_writes())
