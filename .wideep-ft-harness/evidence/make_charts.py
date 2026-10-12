"""Write results.svg: recovery time (log scale) and throughput, from the EP32 measurements."""
import math

RECOVERY = [  # label, low s, high s, color
    ("With FT: serving new requests", 0.16, 0.34, "#2e7d32"),
    ("With FT: in-flight requests finished", 0.8, 3.2, "#66bb6a"),
    ("Without FT: restart (load + warmup only)", 81, 173, "#c62828"),
]
THROUGHPUT = [  # label, tok/s, color
    ("FT off, 32 GPUs", 14689, "#9e9e9e"),
    ("FT on, 32 GPUs", 14488, "#1565c0"),
    ("FT on, 31 GPUs after losing one", 13960, "#42a5f5"),
]
W, LEFT, BAR, GAP = 900, 300, 26, 12
out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="420" font-family="sans-serif" font-size="13">',
       '<rect width="100%" height="100%" fill="white"/>',
       '<text x="10" y="24" font-size="16" font-weight="bold">Time from killing one GPU worker process until… (DeepSeek-R1 on 32 GPUs; log scale)</text>']
lo, hi, y0 = math.log10(0.1), math.log10(300), 44
x = lambda s: LEFT + (math.log10(s) - lo) / (hi - lo) * (W - LEFT - 40)
for t in (0.1, 1, 10, 100):
    out.append(f'<line x1="{x(t):.0f}" y1="{y0}" x2="{x(t):.0f}" y2="{y0 + len(RECOVERY) * (BAR + GAP)}" stroke="#ddd"/>')
    out.append(f'<text x="{x(t):.0f}" y="{y0 + len(RECOVERY) * (BAR + GAP) + 14}" text-anchor="middle" fill="#666">{t:g} s</text>')
for i, (label, a, b, color) in enumerate(RECOVERY):
    y = y0 + i * (BAR + GAP)
    out.append(f'<text x="{LEFT - 8}" y="{y + 18}" text-anchor="end">{label}</text>')
    out.append(f'<rect x="{x(0.1):.0f}" y="{y}" width="{x(b) - x(0.1):.0f}" height="{BAR}" fill="{color}" opacity="0.35"/>')
    out.append(f'<rect x="{x(0.1):.0f}" y="{y}" width="{x(a) - x(0.1):.0f}" height="{BAR}" fill="{color}"/>')
    out.append(f'<text x="{x(b) + 6:.0f}" y="{y + 18}">{a:g} to {b:g} s</text>')
y1 = 250
out.append(f'<text x="10" y="{y1 - 16}" font-size="16" font-weight="bold">Generation throughput (output tokens per second)</text>')
xs = lambda v: LEFT + v / 16000 * (W - LEFT - 120)
for i, (label, v, color) in enumerate(THROUGHPUT):
    y = y1 + i * (BAR + GAP)
    out.append(f'<text x="{LEFT - 8}" y="{y + 18}" text-anchor="end">{label}</text>')
    out.append(f'<rect x="{LEFT}" y="{y}" width="{xs(v) - LEFT:.0f}" height="{BAR}" fill="{color}"/>')
    out.append(f'<text x="{xs(v) + 6:.0f}" y="{y + 18}">{v:,}</text>')
out.append(f'<text x="10" y="{y1 + 3 * (BAR + GAP) + 16}" fill="#666">FT = fault tolerance. Without FT the job must also be torn down and rescheduled before the restart, and all in-flight requests are lost.</text>')
out.append("</svg>")
open("results.svg", "w").write("\n".join(out))
