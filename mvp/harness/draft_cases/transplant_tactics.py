"""Give a ghost-layout autotuner cache the eplb0 layout's MoE tactics (GHOST-TAC).

The MoE runners key their cache entries by (num_slots, top_k, num_local_experts, offset, ...), so eplb0 (256, 8, 8)
and the static 320-slot ghost layout (320, 8, 10) tune separately. The tuning profile spreads tokens over every local
slot, ghost slots included, so the two layouts can pick different tactics for the same real load. This rewrites
each ghost-layout MoE entry to the tactic eplb0 picked for the nearest matching shape: same runner and runner
config, and the closest token count, where a permuted-row bucket M of a tile-T runner with L local experts holds
M - (T - 1) * L tokens. Non-MoE entries and entries without a match keep their own tactic.
Usage: transplant_tactics.py PINNED.json OUT.json [SRC_SLOTS,SRC_LOCAL DST_SLOTS,DST_LOCAL] (default 256,8 320,10)
"""

import json
import re
import sys

HEAD = re.compile(r"'\((\d+), (\d+), (\d+), (\d+), ([^']*)\)'")
NUM = re.compile(r"-?\d+")


def parse(key, slots, local):
    """(group, numbers) of a MoE key of the given layout, or None."""
    head = HEAD.search(key)
    if not head or (int(head[1]), int(head[3])) != (slots, local):
        return None
    rest = head[5]
    tile = int(rest.split(",")[0]) if rest.split(",")[0].strip().isdigit() else 0
    shapes = key[head.end():]
    # Local-expert-count dims are layout, not load: drop them before comparing.
    shapes = re.sub(rf"\({local}, ", "(L, ", shapes)
    shapes = re.sub(rf"\({local},\)", "(L,)", shapes)
    numbers = [int(n) for n in NUM.findall(shapes)]
    skeleton = NUM.sub("#", shapes)
    runner = key[:head.start()]
    return (runner, head[2], rest, skeleton), numbers, tile


def tokens(m, tile, local):
    return m - (tile - 1) * local if tile and m > 0 else m


def main(src, dst, src_layout="256,8", dst_layout="320,10"):
    s_slots, s_local = map(int, src_layout.split(","))
    d_slots, d_local = map(int, dst_layout.split(","))
    cache = json.load(open(src))
    changed = matched = unmatched = 0
    for rank in (k for k in cache if k.startswith("rank_")):
        entries = cache[rank]
        sources = {}
        for key, entry in entries.items():
            parsed = parse(key, s_slots, s_local)
            if parsed:
                sources.setdefault(parsed[0], []).append((parsed[1], parsed[2], entry["tactic"]))
        for key, entry in entries.items():
            parsed = parse(key, d_slots, d_local)
            if not parsed:
                continue
            group, numbers, tile = parsed
            best = None
            for s_numbers, s_tile, tactic in sources.get(group, []):
                if len(s_numbers) != len(numbers):
                    continue
                gap = sum(abs(tokens(a, tile, d_local) - tokens(b, s_tile, s_local))
                          for a, b in zip(numbers, s_numbers) if a != b)
                if best is None or gap < best[0]:
                    best = (gap, tactic)
            if best is None:
                unmatched += 1
                continue
            matched += 1
            changed += entry["tactic"] != best[1]
            entry["tactic"] = best[1]
    print(f"TRANSPLANT {src} {src_layout} -> {dst_layout}: {matched} matched, {changed} changed, "
          f"{unmatched} without a source shape (kept)")
    json.dump(cache, open(dst, "w"))


if __name__ == "__main__":
    main(*sys.argv[1:5])
