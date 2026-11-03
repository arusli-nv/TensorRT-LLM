"""Give every rank the same MoE tactic per shape in an autotuner cache file (C-COMBINE3-PIN).

The MoE runners key their cache entries by the rank's local expert offset, so each rank tunes its own tactics and
ranks can pick different ones for the same shape. This rewrites each rank's MoE entry to the tactic most ranks
picked for that shape (ties: the lowest rank's pick). Usage: pin_tactics.py IN.json OUT.json
"""

import collections
import json
import re
import sys

MOE = ("grouped_gemm", "run_moe")
# The runner config tuple starts (num_slots, top_k, num_local_experts, local_expert_offset, ...).
OFFSET = re.compile(r"'\((\d+), (\d+), (\d+), \d+, ")


def shape_key(key):
    return OFFSET.sub(r"'(\1, \2, \3, X, ", key, count=1)


def main(src, dst):
    cache = json.load(open(src))
    ranks = sorted((k for k in cache if k.startswith("rank_")), key=lambda k: int(k[5:]))
    votes = collections.defaultdict(list)
    for rank in ranks:
        for key, entry in cache[rank].items():
            if any(m in key for m in MOE):
                votes[shape_key(key)].append(json.dumps(entry["tactic"]))
    pinned = {k: collections.Counter(v).most_common(1)[0][0] for k, v in votes.items()}
    changed = 0
    for rank in ranks:
        for key, entry in cache[rank].items():
            if any(m in key for m in MOE):
                tactic = json.loads(pinned[shape_key(key)])
                changed += entry["tactic"] != tactic
                entry["tactic"] = tactic
    split = sum(len(set(v)) > 1 for v in votes.values())
    print(f"PIN_TACTICS {src}: {len(votes)} MoE shapes, {split} with rank-disagreeing tactics, "
          f"{changed} rank entries changed")
    json.dump(cache, open(dst, "w"))


if __name__ == "__main__":
    main(*sys.argv[1:3])
