"""Write an EPLB placement with one spare slot per rank that covers the loss of each protected rank.

Rank r holds its primary experts plus one spare slot. The spares replicate every expert of the
protected ranks (never on the expert's own rank); leftover spares replicate other experts.
Usage: make_placement.py BASE_CONFIG OUT_CONFIG RANKS EXPERTS FIRST_MOE_LAYER NUM_LAYERS [SHIFT]
(SHIFT rotates the spares; by default the first valid one from 16 is used.)
  e.g. DeepSeek-R1: ... 1,2,30,31 256 3 61     Kimi-K2: ... 1,2 384 1 61
"""
import sys

import yaml

EP = 32
base, out = sys.argv[1], sys.argv[2]
protected = [int(r) for r in sys.argv[3].split(",")]
experts, first_layer, num_layers = (int(a) for a in sys.argv[4:7])
per_rank = experts // EP
assert per_rank * EP == experts and len(protected) * per_rank <= EP
covered = [e for r in protected for e in range(r * per_rank, (r + 1) * per_rank)]
filler = [e for e in range(experts) if e // per_rank not in protected]
spares = covered + filler[: EP - len(covered)]
shifts = [int(sys.argv[7])] if len(sys.argv) > 7 else [EP // 2, *range(1, EP)]
shift = next(s for s in shifts if all(spares[(r + s) % EP] // per_rank != r for r in range(EP)))
placement = []
for rank in range(EP):
    placement += list(range(rank * per_rank, (rank + 1) * per_rank)) + [spares[(rank + shift) % EP]]
for dead in protected:
    alive = {e for i, e in enumerate(placement) if i // (per_rank + 1) != dead}
    assert alive == set(range(experts)), dead
config = yaml.safe_load(open(base))
config["moe_config"]["load_balancer"]["initial_global_assignments"] = {
    layer: placement for layer in range(first_layer, num_layers)
}
config["moe_config"]["load_balancer"]["num_slots"] = len(placement)
yaml.safe_dump(config, open(out, "w"), default_flow_style=None, width=1000)
