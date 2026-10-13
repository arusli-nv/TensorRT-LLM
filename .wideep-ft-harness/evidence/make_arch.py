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
    s = Svg(1200, 760)
    s.text(30, 36, "WideEP FT architecture", 22, anchor="start", weight="bold")
    s.chip(800, 18, 100, 26, "existing", "#1a1a1a")
    s.chip(910, 18, 80, 26, "lost", DEAD, filled=True)
    s.chip(1000, 18, 90, 26, "reused", REUSED, filled=True)
    s.chip(1100, 18, 70, 26, "new", NEW, filled=True)

    # Recovery authority: outside the worker job, above the executor it talks to.
    s.box(60, 62, 360, 58, "white", NEW, 10, 2)
    s.text(240, 88, "Recovery authority", 17, "#4a7d00", weight="bold")
    s.text(240, 108, "separate CPU process, outside mpirun", 12.5, "#4a7d00")

    # One surviving rank, in detail.
    s.box(30, 140, 760, 410, "white", "#bdbdbd", 12, 1.5)
    s.text(780, 162, "each surviving rank", 13, "#777", anchor="end", weight="bold")
    s.arrow(170, 122, 170, 196, "purple", 2.5, "7 5", both=True)
    s.text(182, 165, "ZeroMQ + HMAC", 12.5, AUTH, anchor="start", weight="bold")

    # CPU side, including the abort flag in pinned host memory.
    s.text(40, 232, "CPU", 12, "#888", anchor="start", weight="bold")
    s.chip(80, 200, 180, 56, "PeerLossRecovery", NEW, "executor hook")
    s.chip(280, 200, 150, 56, "EPLB", REUSED, "mask-only API")
    s.chip(450, 200, 150, 56, "EPGroupHealth", REUSED, "live-rank mask")
    s.chip(620, 200, 150, 56, "ULFM MPI", NEW, "watcher, Revoke, Shrink")
    s.chip(80, 280, 180, 42, "abort flag", NEW, "pinned host memory")
    s.arrow(170, 258, 170, 277, "blue", 2)

    # GPU side: the descriptor and one MoE layer of the captured CUDA graph.
    s.line(40, 340, 780, 340, "#bdbdbd", 1.5, "4 4")
    s.text(40, 372, "GPU", 12, "#888", anchor="start", weight="bold")
    s.chip(390, 352, 270, 42, "execution descriptor", NEW, "mask | abort-flag address")
    s.arrow(525, 258, 525, 349, "gray", 2)
    s.text(533, 300, "mask", 12, "#555", anchor="start")
    s.line(262, 301, 300, 301, NEW, 2, "4 3")
    s.line(300, 301, 300, 373, NEW, 2, "4 3")
    s.line(300, 373, 388, 373, NEW, 2, "4 3")
    s.text(344, 366, "address", 11.5, "#4a7d00")
    s.box(80, 410, 690, 120, "#f7f7f7", "#9e9e9e", 10, 1.5)
    stages = [("dispatch", 100), ("gather", 260), ("experts", 420), ("combine", 580)]
    s.line(525, 396, 525, 422, NEW, 2)
    s.line(170, 422, 650, 422, NEW, 2)
    for name, x in stages:
        s.chip(x, 438, 140, 50, name, NEW if name == "gather" else "#1a1a1a", None)
        if name != "experts":
            s.arrow(x + 70, 422, x + 70, 435, "blue", 2)
    for (_, x1), (_, x2) in zip(stages, stages[1:]):
        s.arrow(x1 + 142, 463, x2 - 3, 463, "gray", 2)
    s.text(425, 518, "CUDA graph, captured once at startup", 13, "#555", weight="bold")

    # The other ranks: EP32 grid with the lost rank, reached through a bracket.
    cols, cell = 4, 34
    gx, gy = 900, 200
    rows = 32 // cols
    s.text(gx + cols * cell / 2 - 3, gy - 14, "32 ranks", 15, "#444", weight="bold")
    for i in range(32):
        x, y = gx + (i % cols) * cell, gy + (i // cols) * cell
        dead = i == 2
        s.box(x, y, cell - 6, cell - 6, "#ffebee" if dead else "white", DEAD if dead else "#9e9e9e", 4, 1.5)
        if dead:
            s.cross(x + 14, y + 14, 8)
        else:
            s.text(x + 14, y + 19, str(i), 11, "#455a64")
    s.text(gx + cols * cell / 2 - 3, gy + rows * cell + 18, "rank 2 lost", 13, DEAD, weight="bold")
    bracket = gx - 20
    s.line(bracket, gy - 2, bracket, gy + rows * cell - 4, "#555", 2)
    for y, top, bottom in ((228, "MPI", None), (463, "NVLink", "all-to-all")):
        s.arrow(794, y, bracket - 4, y, "gray", 2.5, both=True)
        s.text((794 + bracket) / 2, y - 10, top, 12, "#555", weight="bold")
        if bottom:
            s.text((794 + bracket) / 2, y + 20, bottom, 12, "#555", weight="bold")

    # Recovery phases.
    py, ph = 600, 105
    phases = [
        ("1  Cancel the attempt", ["abort flag: kernels exit", "Revoke: collectives exit", "step discarded"],
         "Stopped"),
        ("2  Install (replay paused)", ["Shrink: agree on lost rank", "EPLB: route to replicas",
                                        "write mask in place"], "Propose · Ready · Install · Installed"),
        ("3  Permission to resume", ["after every survivor's Installed", "same CUDA graphs"], "Commit"),
    ]
    w = 365
    for i, (title, rows, msgs) in enumerate(phases):
        x = 30 + i * (w + 18)
        tip = 22
        pts = f"{x},{py} {x + w},{py} {x + w + tip},{py + ph / 2} {x + w},{py + ph} {x},{py + ph}"
        if i:
            pts += f" {x + tip},{py + ph / 2}"
        s.out.append(f'<polygon points="{pts}" fill="#f7f7f7" stroke="#9e9e9e" stroke-width="2"/>')
        s.text(x + 38, py + 28, title, 16, "#1a1a1a", anchor="start", weight="bold")
        for j, r in enumerate(rows):
            s.text(x + 38, py + 52 + j * 19, r, 13, "#333", anchor="start")
        s.text(x + w / 2 + 10, py + ph + 24, msgs, 13, "#555", weight="bold", italic=True)
    s.text(30, 583, "Recovery", 15, "#444", anchor="start", weight="bold")
    s.text(1170, 583, "any failure in any phase: Abort to every worker, job ends as without FT", 13,
           "#d32f2f", anchor="end", weight="bold")
    return s.render()


def late_writes():
    s = Svg(1200, 610)
    s.text(30, 36, "Late writes from a lost rank", 22, anchor="start", weight="bold")
    s.text(30, 60, "One receiver's dispatch workspace; each cell is a token row, numbered by the source rank "
           "that writes it", 13.5, "#666", anchor="start")
    cw, x0 = 44, 190

    def bar(y, owners, hatch=None, label=None):
        for i, k in enumerate(owners):
            fill = "url(#hatch)" if k == hatch else SRC[k]
            s.box(x0 + i * cw, y, cw - 4, 38, fill, "white", 3, 1)
            s.text(x0 + i * cw + cw / 2 - 2, y + 25, str(k), 15, "#333", weight="bold")
        s.text(x0 + len(owners) * cw + 4, y + 25, "…", 16, "#777", anchor="start")
        if label:
            s.text(x0 - 14, y + 25, label, 14, "#555", anchor="end")

    def lost_rank(x, y):
        s.box(x, y, 78, 42, "#ffebee", DEAD, 6, 2)
        s.text(x + 39, y + 27, "rank 2", 14, DEAD, weight="bold")
        s.cross(x + 74, y + 4, 8)

    # Main: rows per source follow the step's runtime_max_tokens_per_rank.
    s.text(30, 124, "Main", 18, anchor="start", weight="bold")
    s.text(30, 144, "rows move", 13, "#666", anchor="start")
    s.text(30, 160, "every step", 13, "#666", anchor="start")
    bar(104, [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3], label="step n")
    bar(176, [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5], label="step n+1")
    target = x0 + 6 * cw
    s.box(target - 3, 101, cw + 2, 44, "none", DEAD, 4, 2, "5 4")
    s.box(target - 3, 173, cw + 2, 44, "none", DEAD, 4, 3)
    s.arrow(target + cw / 2 - 2, 147, target + cw / 2 - 2, 170, "red", 2.5)
    lost_rank(820, 100)
    tx = target + cw / 2 - 2
    s.out.append(f'<polyline points="820,112 800,112 800,86 {tx},86 {tx},98" fill="none" stroke="{DEAD}" '
                 f'stroke-width="2.5" marker-end="url(#red)"/>')
    s.text(830, 196, "aimed at rank 2's row in step n,", 13, DEAD, anchor="start")
    s.text(830, 213, "lands in rank 3's row in step n+1", 13, DEAD, anchor="start", weight="bold")

    s.line(30, 248, 1170, 248, "#e0e0e0", 1.5)

    # FT: one fixed region per source, sized for the maximum.
    s.text(30, 300, "FT", 18, anchor="start", weight="bold")
    s.text(30, 320, "fixed region", 13, "#666", anchor="start")
    s.text(30, 336, "per source,", 13, "#666", anchor="start")
    s.text(30, 352, "sized by the max", 13, "#666", anchor="start")
    bar(286, [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3], hatch=2, label="every step")
    region = x0 + 6 * cw
    s.box(region - 3, 283, 3 * cw + 2, 44, "none", DEAD, 4, 3)
    lost_rank(region + 27, 372)
    s.arrow(region + 66, 370, region + 66, 332, "red", 2.5)
    s.text(region + 130, 388, "stays in rank 2's own region, never read again", 13, "#4a7d00", anchor="start",
           weight="bold")
    s.text(region + 130, 406, "buffers stay mapped until teardown", 13, "#555", anchor="start")

    # The whole FT workspace: separate arrays, each indexed by source rank.
    s.line(30, 448, 1170, 448, "#e0e0e0", 1.5)
    s.text(30, 488, "Whole", 18, anchor="start", weight="bold")
    s.text(30, 508, "workspace", 18, anchor="start", weight="bold")
    s.text(30, 528, "(FT)", 13, "#666", anchor="start")
    arrays = [("receive counters (x2)", 24), ("dispatch flags", 24), ("combine flags", 24),
              ("EPLB stats", 24), ("payload regions", 48)]
    x, y = 190, 500
    for name, w in arrays:
        s.text(x + 3 * w - 2, y - 12, name, 12.5, "#444", weight="bold")
        for k in range(6):
            fill = "url(#hatch)" if k == 2 else SRC[k]
            s.box(x + k * w, y, w - 3, 34, fill, "white", 3, 1)
            s.text(x + k * w + w / 2 - 1.5, y + 22, str(k), 11.5, "#333")
        s.box(x + 2 * w - 3, y - 3, w + 3, 40, "none", DEAD, 3, 2.5)
        x += 6 * w + 30
    s.text(190, y + 62, "Separate arrays, each with one slot per source rank: rank 2's late writes only reach "
           "slot 2 of each.", 13, "#555", anchor="start")
    s.text(190, y + 82, "Only the payload layout is new; the counters, flags and EPLB stats already had one fixed "
           "slot per source on main.", 13, "#555", anchor="start")

    # The gather kernel turns the fixed regions into the dense expert input.
    s.arrow(x0 + 12 * cw + 26, 305, 790, 305, "gray", 2)
    s.chip(795, 278, 150, 54, "gather (new)", NEW, "skips rank 2")
    s.arrow(947, 305, 995, 305, "gray", 2)
    s.chip(1000, 278, 160, 54, "expert GEMMs", "#1a1a1a", "dense input")
    return s.render()


if __name__ == "__main__":
    open("integration.svg", "w").write(integration())
    open("late_writes.svg", "w").write(late_writes())
