"""Write results.svg: time to serve again after a rank loss, and throughput (DeepSeek-R1, 32 GPUs)."""
import math

W, H = 860, 360
LEFT, RIGHT = 250, 150
GREEN, RED, GRAY, BLUE = "#2e7d32", "#c62828", "#9e9e9e", "#1565c0"
out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" font-family="Helvetica, Arial, sans-serif" font-size="14">',
       '<rect width="100%" height="100%" fill="white"/>']


def title(y, text, note):
    out.append(f'<text x="20" y="{y}" font-size="16" font-weight="bold">{text}</text>')
    out.append(f'<text x="20" y="{y + 18}" fill="#666" font-size="12">{note}</text>')


def bar(y, label, x0, x1, color, value):
    out.append(f'<text x="{LEFT - 10}" y="{y + 17}" text-anchor="end">{label}</text>')
    out.append(f'<rect x="{x0:.0f}" y="{y}" width="{max(x1 - x0, 2):.0f}" height="24" rx="3" fill="{color}"/>')
    out.append(f'<text x="{x1 + 8:.0f}" y="{y + 17}" font-weight="bold">{value}</text>')


# Time from the kill until new requests are served again (log scale).
title(28, "Time from a rank loss until new requests are served", "log scale; without FT: model load and warmup only, before teardown and rescheduling")
lo, hi = math.log10(0.1), math.log10(300)
xt = lambda s: LEFT + (math.log10(s) - lo) / (hi - lo) * (W - LEFT - RIGHT)
for t in (0.1, 1, 10, 100):
    out.append(f'<line x1="{xt(t):.0f}" y1="58" x2="{xt(t):.0f}" y2="128" stroke="#e0e0e0"/>')
    out.append(f'<text x="{xt(t):.0f}" y="144" text-anchor="middle" fill="#888" font-size="12">{t:g} s</text>')
bar(64, "With fault tolerance", xt(0.1), xt(0.34), GREEN, "0.16 – 0.34 s")
bar(98, "Without (restart)", xt(0.1), xt(173), RED, "81 – 173 s")

# Throughput at production shape: 64 requests per GPU, 9k-token sequences, fp8 KV cache.
title(186, "Generation throughput (output tokens/s), 64 requests per GPU", "medians of three runs; the last bar uses the overlap scheduler and MTP, which FT does not support yet")
xs = lambda v: LEFT + v / 200000 * (W - LEFT - RIGHT)
bar(216, "FT off, 32 GPUs", LEFT, xs(82661), GRAY, "82.7k")
bar(250, "FT on, 32 GPUs", LEFT, xs(79636), BLUE, "79.6k  (−3.7%)")
bar(284, "FT on, 31 GPUs after a loss", LEFT, xs(75750), "#64b5f6", "75.8k  (98% per GPU)")
bar(318, "Production config, FT off", LEFT, xs(157183), "#616161", "157.2k")
out.append("</svg>")
open("results.svg", "w").write("\n".join(out))
