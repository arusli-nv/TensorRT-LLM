"""Write an EPLB placement whose spare slots cover the loss of each protected rank.

Rank r holds its primary experts plus SPARES spare slots. The spares replicate every expert of the
protected ranks (never on the expert's own rank); leftover spares replicate other experts.
Usage: [WIDEEP_WORLD=32] [SPARES=1] make_placement.py BASE_CONFIG OUT_CONFIG RANKS EXPERTS
       FIRST_MOE_LAYER NUM_LAYERS
  e.g. DeepSeek-R1: ... 1,2,30,31 256 3 61     Kimi-K2: ... 1,2 384 1 61
       EP16: WIDEEP_WORLD=16 SPARES=2 ... 2 256 3 61
"""
import os
import sys

import yaml

EP = int(os.environ.get("WIDEEP_WORLD", "32"))
SPARES = int(os.environ.get("SPARES", "1"))
base, out = sys.argv[1], sys.argv[2]
protected = [int(r) for r in sys.argv[3].split(",")]
experts, first_layer, num_layers = (int(a) for a in sys.argv[4:7])
per_rank = experts // EP
slots = EP * SPARES
assert per_rank * EP == experts and len(protected) * per_rank <= slots
covered = [e for r in protected for e in range(r * per_rank, (r + 1) * per_rank)]
filler = [e for e in range(experts) if e // per_rank not in protected]
spares = covered + filler[: slots - len(covered)]


def spares_of(rank: int, shift: int) -> list[int]:
    return [spares[(rank * SPARES + j + shift) % slots] for j in range(SPARES)]


shift = next(
    s
    for s in [slots // 2, *range(1, slots)]
    if all(
        all(e // per_rank != r for e in spares_of(r, s)) and len(set(spares_of(r, s))) == SPARES
        for r in range(EP)
    )
)
placement = []
for rank in range(EP):
    placement += list(range(rank * per_rank, (rank + 1) * per_rank)) + spares_of(rank, shift)
for dead in protected:
    alive = {e for i, e in enumerate(placement) if i // (per_rank + SPARES) != dead}
    assert alive == set(range(experts)), dead
config = yaml.safe_load(open(base))
config["moe_config"]["load_balancer"]["initial_global_assignments"] = {
    layer: placement for layer in range(first_layer, num_layers)
}
config["moe_config"]["load_balancer"]["num_slots"] = len(placement)
config["tensor_parallel_size"] = config["moe_expert_parallel_size"] = EP
yaml.safe_dump(config, open(out, "w"), default_flow_style=None, width=1000)
