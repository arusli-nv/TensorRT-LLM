"""Write integration.svg (recovery phases across components) and late_writes.svg (fixed source regions)."""

from html import escape

FONT = 'font-family="Helvetica, Arial, sans-serif"'
NEW, REUSED, AUTH, GPU = "#1565c0", "#6a8f3a", "#6a1b9a", "#ef6c00"
PHASE_BG = ["#fdecea", "#e8f0fe", "#e6f4ea"]


def text(out, x, y, s, size=13, color="#222", anchor="start", weight="normal"):
    out.append(f'<text x="{x}" y="{y}" font-size="{size}" fill="{color}" text-anchor="{anchor}" '
               f'font-weight="{weight}">{escape(s)}</text>')


def lines(out, x, y, rows, size=12.5, color="#222", step=17):
    for i, s in enumerate(rows):
        weight = "bold" if s.startswith("*") else "normal"
        text(out, x, y + i * step, s.lstrip("*"), size, color, weight=weight)


def integration():
    W, H = 1240, 740
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" {FONT}>',
           '<rect width="100%" height="100%" fill="white"/>']
    text(out, 20, 32, "WideEP FT: recovery from one rank loss", 20, weight="bold")
    text(out, 20, 54, "Each survivor runs the same steps; the recovery authority is the only decision-maker. "
         "[R] reused from main   [N] new", 13, "#555")

    left, top, lane_h, label_w = 20, 110, 108, 250
    col_w = (W - left - label_w - 20) / 3
    phases = ["1. Cancel the attempt", "2. Install routing and mask", "3. Permission to resume"]
    notes = ["each survivor on its own", "graph replay paused; authority-driven", "only after every Installed"]
    for i, (p, n) in enumerate(zip(phases, notes)):
        x = left + label_w + i * col_w
        out.append(f'<rect x="{x + 4:.0f}" y="{top - 40}" width="{col_w - 8:.0f}" height="{5 * lane_h + 44}" '
                   f'rx="8" fill="{PHASE_BG[i]}"/>')
        text(out, x + col_w / 2, top - 20, p, 15, anchor="middle", weight="bold")
        text(out, x + col_w / 2, top - 4, n, 12, "#555", anchor="middle")

    lanes = [
        ("Recovery authority", "separate CPU process [N]", AUTH),
        ("Executor", "PeerLossRecovery [N]", NEW),
        ("Host collectives", "ULFM survivor communicator [N]", NEW),
        ("Routing and membership", "EPLB mask-only API, EPGroupHealth [R]", REUSED),
        ("GPU all-to-all", "graph-visible descriptor [N]", GPU),
    ]
    cells = [
        [["Receives Stopped from", "every survivor"],
         ["Propose(generation, survivors)", "Checks all Ready digests match", "Install"],
         ["*Commit only after every", "*survivor sent Installed", "(otherwise Abort: fail closed)"]],
        [["Watcher detects the death", "Commit gate discards the step", "KV Cache Manager V2 rollback", "→ Stopped"],
         ["Placement digest → Ready", "On Install: apply, then", "→ Installed"],
         ["Resume; redo the step", "Dead rank's requests fail", "with a retry error"]],
        [["Revoke wakes ranks blocked", "in host collectives", "Shrink: survivors agree on", "the dead rank"],
         ["Switch to the survivor", "communicator (rank ids kept)"],
         ["Per-step allgathers run on", "survivors; dead rank's entry", "is empty"]],
        [["Unchanged"],
         ["reconfigure_mask_only(dead):", "its experts → their replicas [R]", "EPGroupHealth marks it failed [R]"],
         ["Tokens for the dead rank's", "experts go to replicas"]],
        [["Abort flag (pinned host memory)", "releases spinning kernels;", "the step's output is invalid"],
         ["rearm(): reset round counters", "install_mask(): new mask", "written in place"],
         ["*Same CUDA graphs replay", "on N-1 ranks; kernels skip", "the masked rank"]],
    ]
    for r, (name, sub, color) in enumerate(lanes):
        y = top + r * lane_h
        out.append(f'<rect x="{left}" y="{y + 6}" width="{label_w - 12}" height="{lane_h - 12}" rx="6" '
                   f'fill="{color}"/>')
        text(out, left + 12, y + 40, name, 15, "white", weight="bold")
        text(out, left + 12, y + 60, sub, 11.5, "white")
        for c in range(3):
            x = left + label_w + c * col_w
            out.append(f'<rect x="{x + 14:.0f}" y="{y + 8}" width="{col_w - 28:.0f}" height="{lane_h - 16}" '
                       f'rx="6" fill="white" stroke="#ccc"/>')
            rows = cells[r][c]
            y0 = y + lane_h / 2 - (len(rows) - 1) * 8.5 + 4
            lines(out, x + 26, y0, rows)

    # Message arrows between the authority and the executor lane.
    ay, ey = top + lane_h - 8, top + lane_h + 8
    for c, (label, up) in enumerate([("Stopped", True), ("Ready ↑  Install ↓  Installed ↑", None),
                                     ("Commit", False)]):
        x = left + label_w + c * col_w + col_w - 40
        if up is None:
            continue
        y1, y2 = (ey, ay) if up else (ay, ey)
        out.append(f'<line x1="{x:.0f}" y1="{y1}" x2="{x:.0f}" y2="{y2}" stroke="{AUTH}" stroke-width="2.5" '
                   f'marker-end="url(#a)"/>')
    out.insert(2, f'<defs><marker id="a" markerWidth="8" markerHeight="8" refX="4" refY="4" orient="auto">'
                  f'<path d="M0,0 L8,4 L0,8 z" fill="{AUTH}"/></marker></defs>')
    text(out, 20, H - 22, "Messages are HMAC-signed ZeroMQ: Stopped → Propose → Ready → Install → Installed → Commit. "
         "Any violation, mismatch or timeout sends Abort to every worker.", 12.5, "#555")
    text(out, 20, H - 4, "Later, #15785's evidence transport can replace the watcher as the detector "
         "(Executor lane, phase 1); the authority stays the only decision-maker.", 12.5, "#555")
    out.append("</svg>")
    return "\n".join(out)


def late_writes():
    W, H = 1240, 700
    colors = ["#90caf9", "#a5d6a7", "#ffcc80", "#ce93d8", "#80cbc4", "#f48fb1"]
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" {FONT}>',
           '<rect width="100%" height="100%" fill="white"/>',
           '<defs><pattern id="h" width="8" height="8" patternUnits="userSpaceOnUse" '
           'patternTransform="rotate(45)"><rect width="8" height="8" fill="#e0e0e0"/>'
           '<line x1="0" y1="0" x2="0" y2="8" stroke="#9e9e9e" stroke-width="3"/></pattern>'
           '<marker id="r" markerWidth="8" markerHeight="8" refX="4" refY="4" orient="auto">'
           '<path d="M0,0 L8,4 L0,8 z" fill="#c62828"/></marker></defs>']
    text(out, 20, 32, "Late writes from a dead rank: fixed per-source regions", 20, weight="bold")
    text(out, 20, 54, "One receiver's dispatch workspace (symmetric memory, written by peers over NVLink). "
         "Each cell is one token row; color = source rank that owns it.", 13, "#555")

    x0, cw, ch = 220, 58, 34
    layouts = [
        ("Main, step n", [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2], "runtime_max_tokens_per_rank = 4"),
        ("Main, step n+1", [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5], "runtime_max_tokens_per_rank = 2"),
        ("FT, every step", [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2], "max_num_tokens_per_rank = 4 (fixed)"),
    ]
    ys = [100, 175, 290]
    for (label, owners, note), y in zip(layouts, ys):
        text(out, x0 - 16, y + 22, label, 15, anchor="end", weight="bold")
        for i, k in enumerate(owners):
            fill = "url(#h)" if (label.startswith("FT") and k == 2) else colors[k]
            out.append(f'<rect x="{x0 + i * cw}" y="{y}" width="{cw - 3}" height="{ch}" rx="3" fill="{fill}"/>')
            text(out, x0 + i * cw + cw / 2 - 1, y + 22, str(k), 14, anchor="middle", weight="bold")
        text(out, x0 + 12 * cw + 14, y + 22, note, 13, "#444")
    for i in range(12):
        text(out, x0 + i * cw + cw / 2 - 1, 92, f"row {i}", 10.5, "#888", anchor="middle")

    # Main: rank 2's late write aimed at row 8 in step n lands in rank 4's row in step n+1.
    bx = x0 + 8 * cw + cw / 2
    out.append(f'<line x1="{bx}" y1="{ys[0] + ch + 2}" x2="{bx}" y2="{ys[1] - 4}" stroke="#c62828" '
               f'stroke-width="3" marker-end="url(#r)"/>')
    out.append(f'<rect x="{x0 + 8 * cw - 2}" y="{ys[1] - 2}" width="{cw + 1}" height="{ch + 4}" fill="none" '
               f'stroke="#c62828" stroke-width="3" rx="4"/>')
    text(out, x0, ys[1] + 58, "Dead rank 2's in-flight write, aimed at row 8 in step n, lands in rank 4's row "
         "in step n+1: it corrupts a survivor's data.", 13, "#c62828", weight="bold")
    text(out, x0, ys[2] + ch + 24, "Rank 2's region (hatched) is never reassigned or read after it dies: "
         "its late writes stay there.", 13, "#2e7d32", weight="bold")

    # What source k writes into a peer, and why it stays harmless.
    ty = 400
    text(out, 20, ty, "Everything source k writes into a peer's workspace", 16, weight="bold")
    rows = [
        ("Written by source k", "Location in the peer", "After rank k is lost"),
        ("Dispatch payloads", "fixed region k: [source][payload][max_num_tokens_per_rank]",
         "gather skips masked sources; region never reassigned"),
        ("Dispatch / combine completion flags", "word k of each flag array", "masked sources are never waited on"),
        ("Receive counters (two parities)", "word k of each parity", "not read for masked sources"),
        ("EPLB gathered statistics", "row k", "ignored for masked sources"),
        ("Combine outputs", "none: combine pulls from peers", "—"),
    ]
    widths = [300, 450, 450]
    for r, row in enumerate(rows):
        y = ty + 18 + r * 32
        x = 20
        fill = "#eceff1" if r == 0 else ("#fafafa" if r % 2 else "white")
        out.append(f'<rect x="20" y="{y}" width="{sum(widths)}" height="32" fill="{fill}" stroke="#ddd"/>')
        for c, s in enumerate(row):
            text(out, x + 10, y + 21, s, 13, weight="bold" if r == 0 else "normal")
            x += widths[c]
    text(out, 20, ty + 18 + 6 * 32 + 30, "Backing memory stays valid: bound workspaces stay mapped until model teardown, "
         "so late writes never hit freed or reused memory.", 13, "#333")
    text(out, 20, ty + 18 + 6 * 32 + 52, "Gather kernel (new): packs valid rows of the fixed regions into the dense "
         "input the expert GEMMs expect; masked or aborted rows become expert id -1 with zero data.", 13, "#333")
    out.append("</svg>")
    return "\n".join(out)


if __name__ == "__main__":
    open("integration.svg", "w").write(integration())
    open("late_writes.svg", "w").write(late_writes())
