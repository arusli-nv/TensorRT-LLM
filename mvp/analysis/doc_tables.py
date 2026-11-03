"""Markdown tables for DESIGN_DOC.md from finals_readout.py --json output.

usage: doc_tables.py READOUT.json
Prints the healthy-cost table (per-step cost, the protocol's primary metric) and the static-LB control table.
"""
import json
import sys

d = json.load(open(sys.argv[1]))
MODES = [("noeplb", "No EPLB"), ("offline", "Offline EPLB"), ("online", "Online EPLB")]


def f(x, fmt):
    return "—" if x is None else format(x, fmt)


print("| Mode | Req/GPU | FT cost per step (mean, %) | Upper 95% bound | n groups | Δ per step (µs) | "
      "Decode tok/s/GPU off → on | Median TPOT off → on (ms) |")
print("|---|---|---|---|---|---|---|---|")
for m, name in MODES:
    for L in ("8", "32", "128", "512"):
        c = d["cells"].get(f"{m}/{L}")
        if not c or not c.get("n"):
            print(f"| {name} | {L} | not measured | | 0 | | | |")
            continue
        off = d["client"].get(f"{m}/{L}/std-{m}", {})
        on = d["client"].get(f"{m}/{L}/std-{m}-ft", {})
        tp = lambda r: (sorted(r["tpot_median"])[len(r["tpot_median"]) // 2] if r.get("tpot_median") else None)
        note = " (beyond recipe; not in headline)" if L == "512" else ""
        print(f"| {name} | {L}{note} | {f(c['mean'], '+.2f')} | {f(c.get('ub95'), '+.2f')} | {c['n']} | "
              f"{f(c.get('delta_us'), '+.0f')} | {f(c.get('off_tok_s_gpu'), ',.0f')} → {f(c.get('on_tok_s_gpu'), ',.0f')} | "
              f"{f(tp(off), '.2f')} → {f(tp(on), '.2f')} |")
lb = d.get("lb") or []
if lb:
    print("\n| Group | Req/GPU | No EPLB (ms) | Static LB, FT off (ms) | FT on (ms) | LB path vs off | FT vs LB |")
    print("|---|---|---|---|---|---|---|")
    for r in lb:
        print(f"| {r.get('group')} | {r.get('load')} | {f(r.get('noeplb_ms'), '.3f')} | {f(r.get('lb_ms'), '.3f')} | "
              f"{f(r.get('ft_ms'), '.3f')} | {f(r.get('lb_vs_noeplb_pct'), '+.2f')}% | {f(r.get('ft_vs_lb_pct'), '+.2f')}% |")
print(f"\nEvery-cell test (all headline cells below +1%): {d.get('iut_every_cell')}")
