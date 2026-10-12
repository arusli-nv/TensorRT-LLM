"""Write a 288-slot EPLB placement in which every expert of the protected ranks has a replica elsewhere.

Rank r holds its 8 primary experts 8r..8r+7 plus one spare slot. The 32 spare slots replicate the
32 experts of 4 protected ranks, never on the expert's own rank, so any single protected rank can
be lost with full coverage. Usage: make_placement.py BASE_CONFIG OUT_CONFIG RANK,RANK,RANK,RANK [SPARE_SHIFT]
"""
import sys

import yaml

EXPERTS, EP, SLOTS_PER_RANK, MOE_LAYERS = 256, 32, 9, range(3, 61)
base, out, protected = sys.argv[1], sys.argv[2], [int(r) for r in sys.argv[3].split(",")]
shift = int(sys.argv[4]) if len(sys.argv) > 4 else EP // 2
per_rank = EXPERTS // EP
assert len(protected) * per_rank == EP, "spare slots must exactly cover the protected ranks"
spares = [e for r in protected for e in range(r * per_rank, (r + 1) * per_rank)]
placement = []
for rank in range(EP):
    spare = spares[(rank + shift) % EP]
    assert spare // per_rank != rank
    placement += list(range(rank * per_rank, (rank + 1) * per_rank)) + [spare]
assert len(placement) == EP * SLOTS_PER_RANK and set(placement) == set(range(EXPERTS))
for dead in protected:
    alive = {e for i, e in enumerate(placement) if i // SLOTS_PER_RANK != dead}
    assert alive == set(range(EXPERTS)), dead
config = yaml.safe_load(open(base))
config["moe_config"]["load_balancer"]["initial_global_assignments"] = {
    layer: placement for layer in MOE_LAYERS}
yaml.safe_dump(config, open(out, "w"), default_flow_style=None, width=1000)
