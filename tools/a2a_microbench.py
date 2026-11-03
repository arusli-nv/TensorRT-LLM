"""NVLinkOneSided dispatch + combine time, FT off (unbound) vs FT on (bound to a MoeExecutionState), per tokens/rank.

DeepSeek-R1 decode shapes: NVFP4 hidden 7168 (3584 B + 448 B scales per token), top-k 8, 256 experts; FT on adds the
w/o EPLB deployment's ghost slots (10 per rank instead of 8). Fence path (CFT needs driver 615+), low-precision combine.
One mode per launch, so the two never share a workspace:
  mpirun -n EP python3 -u a2a_microbench.py {off,ft} OUT.json EXPERTS [TOKENS ...]
EXPERTS = 8 * EP gives DEP32's per-rank layout (8 local slots, 10 with FT); 256 gives R1's expert count.
Rank 0 writes {tokens: {dispatch_us, combine_us, round_trip_us, graph_round_trip_us}}: eager medians of 200 timed
iterations after 50 warm-up, and the median of 20 replays of a CUDA graph of 20 back-to-back round trips (sub-us
resolution, layers back to back as in serving). Compare the FT-off/on delta at the EP size of interest.
"""
import json
import os
import statistics
import sys

import torch
from mpi4py import MPI

import tensorrt_llm  # noqa: F401
from tensorrt_llm._torch.distributed.mnnvl_memory import MnnvlMemory
from tensorrt_llm._torch.moe.fused_moe.communication.execution_state import MoeExecutionState
from tensorrt_llm._torch.moe.fused_moe.communication.nvlink_one_sided import NVLinkOneSided
from tensorrt_llm._torch.moe.fused_moe.ep_group_health import EPGroupHealth
from tensorrt_llm.mapping import Mapping

HIDDEN, TOP_K, WARMUP, ITERS, GRAPH_TRIPS, REPLAYS = 7168, 8, 50, 200, 20, 20

mode, out_path = sys.argv[1], sys.argv[2]
EXPERTS = int(sys.argv[3])
token_counts = [int(t) for t in sys.argv[4:]] or [16, 32, 64, 128, 256]
world = MPI.COMM_WORLD
rank, ep = world.Get_rank(), world.Get_size()
torch.cuda.set_device(rank % torch.cuda.device_count())
MnnvlMemory.initialize()

slots_per_rank = EXPERTS // ep + (2 if mode == "ft" else 0)
num_slots = slots_per_rank * ep
health = EPGroupHealth(ep) if mode == "ft" else None
comm = NVLinkOneSided(
    mapping=Mapping(rank=rank, world_size=ep, tp_size=ep, moe_ep_size=ep),
    num_slots=num_slots,
    top_k=TOP_K,
    # Serving sizes the workspace for max_num_tokens (8192 in the std configs): bound receive slots are that many rows
    # per source. TLLM_TEST_A2A_CAPACITY sets it; the default is the largest swept token count.
    max_num_tokens_per_rank=int(os.environ.get("TLLM_TEST_A2A_CAPACITY", max(token_counts))),
    hidden_size=HIDDEN,
    dtype=torch.bfloat16,
    payload_in_workspace=True,
    # Serving passes num_experts only with a load balancer, which the ghost slots imply.
    num_experts=EXPERTS if mode == "ft" else None,
    use_low_precision_combine=True,
    ep_group_health=health,
)
if mode == "ft":
    comm.bind_execution_state(MoeExecutionState(health, torch.device("cuda")))

# FT on dispatches expert ids through the load balancer's slot table, as serving does.
local = EXPERTS // ep
expert_to_slot = ((torch.arange(EXPERTS) // local) * slots_per_rank + torch.arange(EXPERTS) % local).int().cuda()
results = {}
generator = torch.Generator(device="cuda").manual_seed(rank)
# Routing: uniform by default; TLLM_TEST_A2A_EXPERT_LOAD replays a served per-expert load (a JSON "expert_load" list, e.g.
# tools/a2a_expert_load.json from a DeepSeek-R1 run's expert statistics), so receive counts differ by rank as in serving.
LOAD_FILE = os.environ.get("TLLM_TEST_A2A_EXPERT_LOAD")
popularity = None
if LOAD_FILE:
    popularity = torch.tensor(json.load(open(LOAD_FILE))["expert_load"], dtype=torch.float32, device="cuda")
    assert popularity.numel() == EXPERTS, "the expert load must have one entry per expert"
for tokens in token_counts:
    # Experts sit on the first EXPERTS // ep slots of each rank; the ghost slots get no tokens.
    if popularity is not None:
        experts = torch.multinomial(popularity.repeat(tokens, 1), TOP_K, replacement=False, generator=generator)
    else:
        experts = torch.rand(tokens, EXPERTS, device="cuda", generator=generator).topk(TOP_K).indices
    experts = experts.int()
    route = {"expert_to_slot": expert_to_slot} if mode == "ft" else {}
    routed = experts if mode == "ft" else expert_to_slot[experts.long()]
    weights = torch.rand(tokens, TOP_K, device="cuda", generator=generator)
    hidden = torch.randint(0, 255, (tokens, HIDDEN // 2), dtype=torch.uint8, device="cuda")
    scales = torch.randint(0, 255, (tokens, HIDDEN // 16), dtype=torch.uint8, device="cuda")
    counts = [tokens] * ep
    times = {"dispatch_us": [], "combine_us": [], "round_trip_us": []}
    for it in range(WARMUP + ITERS):
        start, mid, end = (torch.cuda.Event(enable_timing=True) for _ in range(3))
        world.Barrier()
        # Queue the launches behind ~1 ms of GPU work, so host time is not timed (serving replays CUDA graphs).
        torch.cuda._sleep(2_000_000)
        start.record()
        comm.dispatch(hidden, scales, routed, weights, counts, **route)
        mid.record()
        # The MoE writes its output into the workspace in serving; combine then reads it in place.
        expert_out = comm.get_combine_payload_tensor_in_workspace(tokens, HIDDEN, torch.bfloat16)
        if it == 0:
            expert_out.copy_(torch.randn_like(expert_out))
        comm.combine(expert_out, all_rank_max_num_tokens=tokens)
        end.record()
        end.synchronize()
        if it >= WARMUP:
            times["dispatch_us"].append(start.elapsed_time(mid) * 1e3)
            times["combine_us"].append(mid.elapsed_time(end) * 1e3)
            times["round_trip_us"].append(start.elapsed_time(end) * 1e3)
    medians = {k: statistics.median(v) for k, v in times.items()}

    def round_trip():
        comm.dispatch(hidden, scales, routed, weights, counts, **route)
        comm.combine(comm.get_combine_payload_tensor_in_workspace(tokens, HIDDEN, torch.bfloat16),
                     all_rank_max_num_tokens=tokens)

    try:
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for _ in range(GRAPH_TRIPS):
                round_trip()
        replays = []
        for _ in range(REPLAYS):
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            world.Barrier()
            torch.cuda._sleep(2_000_000)
            start.record()
            graph.replay()
            end.record()
            end.synchronize()
            replays.append(start.elapsed_time(end) * 1e3 / GRAPH_TRIPS)
        medians["graph_round_trip_us"] = statistics.median(replays)
    except RuntimeError as error:
        print(f"rank {rank} tokens/rank={tokens}: no CUDA graph timing ({error})", flush=True)
        medians["graph_round_trip_us"] = float("nan")
    # The slowest rank sets the step: report the max over ranks of each rank's median.
    gathered = world.gather(medians, root=0)
    if rank == 0:
        results[tokens] = {k: max(g[k] for g in gathered) for k in medians}
        print(f"{mode} tokens/rank={tokens}: " + ", ".join(f"{k} {v:.1f}" for k, v in results[tokens].items()),
              flush=True)

world.Barrier()
comm.destroy()
if rank == 0:
    json.dump({"mode": mode, "ep": ep, "experts": EXPERTS, "num_slots": num_slots,
               "capacity": comm.max_num_tokens_per_rank, "expert_load": LOAD_FILE, "results": results}, open(out_path, "w"), indent=1)
