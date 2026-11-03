# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A2A-EP: per-round-trip time of the fence-path MoE A2A, FT's rank-mask instantiation vs FT off, at any EP size.

scratch-a2abench/a2a_bench.py's --graph mode (A2A-MICRO, CLAIMS 191) as one MPI rank per GPU, so EP spans nodes
(mpirun through launch/task.sh). Same raw ops, shapes and variants: DeepSeek-R1 NVFP4 payloads (hidden 7168: FP4
3584 B + FP8 block scales 448 B per token), top-8 over 256 experts, int32 slots and fp32 scales, low-precision combine
of a bf16 [ep, T, 7168] input, workspace from NVLinkOneSided's own layout on a real MNNVL allocation.
  off  = enable_rank_mask False, no abort record
  ft   = enable_rank_mask True, the all-live mask and the abort record, as a bound MoeExecutionState passes them
  off2 = a second, separately captured graph of off: the off/off control
Per load T, one CUDA graph of --calls back-to-back round trips (dispatch + combine) per variant. Blocks replay one graph
each, in cycles of the three variants whose order rotates by one each cycle. Per block, every rank records its replay
time / calls; the block's value is the minimum over ranks (the last rank to arrive waits least).

  mpirun ... python3 a2aep_bench.py --out RUN.json [--loads 32,64,256] [--cycles 20] [--calls 200] [--eplb]
  python3 a2aep_bench.py --summarize DIR...        (every RUN.json below DIR: per-launch and per-cell table)

--ablate V (A2A FT ablation build, a2a-ablate; default off = the variants above): ft is captured with
TRTLLM_A2A_FT_ABLATE=V and a fourth graph ft0 with TRTLLM_A2A_FT_ABLATE=0 (the unchanged FT kernel) joins the rotation.
Before capture, each variant also runs one eager round trip on check inputs (combine input constant per rank, EPLB
stats arange + 1000 * rank) and must return the same combine output as off, bit for bit, and with --eplb the expected
gathered stats: A2A_EP_CHECK lines, and the launch fails if any rank differs.
"""

import argparse
import json
import os
import statistics
import sys
from pathlib import Path

VARIANTS = ("off", "ft", "off2")
# TRTLLM_A2A_FT_ABLATE values and their FT_ABL bits (moeAlltoAllKernels.cu kFtAbl*, a2a-ablate d47a3bcf88).
ABLATE_BITS = {"0": 0, "S": 1, "S1": 2, "F": 4, "F8": 64, "L": 8, "A": 16, "SO": 32, "ALL": 61}
TOP_K, EXPERTS, HIDDEN = 8, 256, 7168
FP4_BYTES, SF_BYTES = HIDDEN // 2, HIDDEN // 16


def bench(args) -> None:
    import torch
    from mpi4py import MPI

    import tensorrt_llm  # noqa: F401
    from tensorrt_llm._torch.distributed.mnnvl_memory import MnnvlMemory
    from tensorrt_llm._torch.moe.fused_moe.communication.execution_state import MoeExecutionState
    from tensorrt_llm._torch.moe.fused_moe.communication.nvlink_one_sided import NVLinkOneSided
    from tensorrt_llm._torch.moe.fused_moe.ep_group_health import EPGroupHealth
    from tensorrt_llm.bindings import internal
    from tensorrt_llm.mapping import Mapping

    world = MPI.COMM_WORLD
    rank, ep = world.Get_rank(), world.Get_size()
    gpus = torch.cuda.device_count()
    torch.cuda.set_device(rank % gpus)
    loads = [int(x) for x in args.loads.split(",")]
    mapping = Mapping(ep, rank, gpus, tp_size=ep, moe_ep_size=ep)
    MnnvlMemory.initialize()
    meta = NVLinkOneSided._make_workspace_layout(
        ep, TOP_K, max(loads), HIDDEN, torch.bfloat16, EXPERTS if args.eplb else None, 0, False, True)
    memory = MnnvlMemory(mapping, int(meta[int(internal.thop.MOE_A2A_WORKSPACE_SIZE_INDEX)]))
    workspace = memory.as_torch_strided_tensor(torch.uint8)
    torch.ops.trtllm.moe_a2a_initialize(workspace, meta, rank, ep, barrier=False)
    state = MoeExecutionState(EPGroupHealth(ep), workspace.device)
    world.Barrier()

    gen = torch.Generator(device="cuda").manual_seed(1234 + rank)
    variants = {
        "off": dict(enable=False, mask=None, abort=None),
        "ft": dict(enable=True, mask=state.active_rank_mask, abort=state.abort),
        "off2": dict(enable=False, mask=None, abort=None),
    }
    names = VARIANTS
    if args.ablate is not None:
        variants["ft"]["ablate"] = args.ablate
        variants["ft0"] = dict(enable=True, mask=state.active_rank_mask, abort=state.abort, ablate="0")
        names = VARIANTS + ("ft0",)
    results = {}
    graph_envs = {}
    for tokens in loads:
        slots = torch.stack([torch.randperm(EXPERTS, device="cuda", generator=gen)[:TOP_K] for _ in range(tokens)])
        slots = slots.to(torch.int32)
        payloads = [
            torch.randint(0, 255, (tokens, FP4_BYTES), dtype=torch.uint8, device="cuda", generator=gen),
            torch.randint(0, 255, (tokens, SF_BYTES), dtype=torch.uint8, device="cuda", generator=gen),
            slots,
            torch.rand(tokens, TOP_K, device="cuda", generator=gen),
        ]
        expert_out = torch.randn(ep, tokens, HIDDEN, dtype=torch.bfloat16, device="cuda", generator=gen)
        stats = torch.zeros(EXPERTS, dtype=torch.int32, device="cuda") if args.eplb else None

        def round_trip(v, combine_in=expert_out, local_stats=stats):
            if "ablate" in v:
                os.environ["TRTLLM_A2A_FT_ABLATE"] = v["ablate"]  # read by every dispatch/combine launch
            else:
                os.environ.pop("TRTLLM_A2A_FT_ABLATE", None)
            _, offset, gathered = torch.ops.trtllm.moe_a2a_dispatch(
                slots, payloads, workspace, meta, tokens, rank, ep, TOP_K, EXPERTS, local_stats, False, None, None,
                v["enable"], v["mask"], abort_state=v["abort"])
            out = torch.ops.trtllm.moe_a2a_combine(
                combine_in, tokens, workspace, meta, tokens, rank, ep, TOP_K, offset, False, True, False,
                v["enable"], v["mask"], abort_state=v["abort"])
            return gathered, out

        checks = {}
        if args.ablate is not None:
            # Data-independent kernels: these inputs only make the result deterministic (no dependence on the
            # receive-slot order) so every variant must reproduce off's result exactly.
            check_in = torch.full((ep, tokens, HIDDEN), (rank + 1) / 64, dtype=torch.bfloat16, device="cuda")
            check_stats = (torch.arange(EXPERTS, dtype=torch.int32, device="cuda") + 1000 * rank
                           if args.eplb else None)
            expected_stats = (torch.arange(EXPERTS, dtype=torch.int32, device="cuda")[None, :]
                              + 1000 * torch.arange(ep, dtype=torch.int32, device="cuda")[:, None])
            for name, v in variants.items():
                world.Barrier()
                gathered, out = round_trip(v, check_in, check_stats)
                torch.cuda.synchronize()
                checks[name] = (gathered.clone(), out.clone())
                world.Barrier()
            bad = []
            for name in names:
                same_out = torch.equal(checks[name][1], checks["off"][1])
                same_stats = not args.eplb or torch.equal(checks[name][0], expected_stats)
                bad += [] if same_out and same_stats else [f"{name}(out={same_out},stats={same_stats})"]
            failed = world.allgather(bad)
            if rank == 0:
                print(f"A2A_EP_CHECK ep={ep} stats={int(args.eplb)} load={tokens} ablate={args.ablate} "
                      + ("OK" if not any(failed) else f"FAIL {[(r, b) for r, b in enumerate(failed) if b]}"), flush=True)
            if any(failed):
                sys.exit(1)

        graphs = {}
        graph_env = {}
        for name, v in variants.items():
            world.Barrier()
            round_trip(v)  # eager warm-up before capture
            torch.cuda.synchronize()
            world.Barrier()
            graphs[name] = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graphs[name]):
                for _ in range(args.calls):
                    round_trip(v)
            # The value every launch in this capture read (round_trip sets it before each op call).
            env = os.environ.get("TRTLLM_A2A_FT_ABLATE")
            graph_env[name] = {"env": env, "bits": ABLATE_BITS.get(env) if env is not None else None}
            if rank == 0 and args.ablate is not None:
                print(f"A2A_EP_VARIANT launch={Path(args.out).stem} load={tokens} graph={name} "
                      f"variant={v.get('ablate', 'none')} env={env if env is not None else 'unset'} "
                      f"bits={graph_env[name]['bits']}", flush=True)
            world.Barrier()
        os.environ.pop("TRTLLM_A2A_FT_ABLATE", None)
        times = {name: [] for name in names}
        # Cycle 0 warms every graph up and is not recorded.
        for cycle in range(args.cycles + 1):
            order = names[cycle % len(names):] + names[:cycle % len(names)]
            for name in order:
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                world.Barrier()
                torch.cuda.synchronize()
                start.record()
                graphs[name].replay()
                end.record()
                torch.cuda.synchronize()
                if cycle:
                    times[name].append(start.elapsed_time(end) * 1e3 / args.calls)
        gathered = world.gather(times, root=0)
        if rank == 0:
            results[str(tokens)] = {name: [g[name] for g in gathered] for name in names}
            graph_envs[str(tokens)] = graph_env
            print(launch_lines(results[str(tokens)], tokens, ep, args.eplb, args.ablate), flush=True)
        del graphs
        torch.cuda.synchronize()
        world.Barrier()
    if rank == 0:
        run = {"ep": ep, "eplb_stats": args.eplb, "calls": args.calls, "cycles": args.cycles,
               "capacity": max(loads), "results": results}
        if args.ablate is not None:
            run["ablate"] = args.ablate
            run["graph_env"] = graph_envs
        Path(args.out).write_text(json.dumps(run))
    world.Barrier()


def block_values(per_rank: list[list[float]]) -> list[float]:
    """Per block, the minimum over ranks of the per-round-trip time."""
    return [min(block) for block in zip(*per_rank)]


def cell(load: dict) -> dict:
    """M_v = median over blocks of the min-over-ranks per-round-trip time; delta = M_ft - M_off, noise = M_off2 - M_off.
    With an ablation run's ft0 (unchanged FT kernel): delta0 = M_ft0 - M_off, cut = M_ft0 - M_ft."""
    m = {name: statistics.median(block_values(times)) for name, times in load.items()}
    c = {**m, "delta": m["ft"] - m["off"], "noise": m["off2"] - m["off"]}
    if "ft0" in m:
        c.update(delta0=m["ft0"] - m["off"], cut=m["ft0"] - m["ft"])
    return c


def launch_lines(load: dict, tokens: int, ep: int, eplb: bool, ablate=None) -> str:
    c = cell(load)
    line = (f"A2A_EP ep={ep} stats={int(eplb)} load={tokens} off={c['off']:.2f}us ft={c['ft']:.2f}us "
            f"off2={c['off2']:.2f}us delta={c['delta']:+.2f}us noise={c['noise']:+.2f}us")
    if ablate is not None:
        line += f" ablate={ablate} ft0={c['ft0']:.2f}us delta0={c['delta0']:+.2f}us cut={c['cut']:+.2f}us"
    return line


def summarize(dirs: list[str]) -> None:
    """Per launch, then per (EP, stats, load) cell: the median over launches of delta and of noise."""
    cells = {}
    for path in sorted(p for d in dirs for p in Path(d).rglob("*.json")):
        run = json.loads(path.read_text())
        if "results" not in run:
            continue
        ablate = run.get("ablate")
        for tokens, load in run["results"].items():
            print(f"{path.parent.name}/{path.name} "
                  + launch_lines(load, int(tokens), run["ep"], run["eplb_stats"], ablate))
            key = (run["ep"], run["eplb_stats"], int(tokens)) + ((ablate,) if ablate is not None else ())
            cells.setdefault(key, []).append(cell(load))
    print("ep | stats | load | launches | off us | ft us | delta us (median, launches) | noise us (median, max |.|) "
          "| delta x 59 us/step" + (" | ablate" if any(len(k) == 4 for k in cells) else ""))
    for (ep, eplb, tokens, *ablate), runs in sorted(cells.items()):
        delta = statistics.median(r["delta"] for r in runs)
        noise = statistics.median(r["noise"] for r in runs)
        each = " ".join("%+.2f" % r["delta"] for r in runs)
        print(f"{ep} | {int(eplb)} | {tokens} | {len(runs)} | {statistics.median(r['off'] for r in runs):.2f} | "
              f"{statistics.median(r['ft'] for r in runs):.2f} | {delta:+.2f} ({each}) | {noise:+.2f} "
              f"({max(abs(r['noise']) for r in runs):.2f}) | {59 * delta:+.0f}" + "".join(f" | {a}" for a in ablate))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out")
    parser.add_argument("--loads", default="32,64,256")
    parser.add_argument("--cycles", type=int, default=20, help="recorded cycles; each replays every variant once")
    parser.add_argument("--calls", type=int, default=200, help="round trips per graph")
    parser.add_argument("--eplb", action="store_true", help="pass EPLB local stats (the online instantiation)")
    parser.add_argument("--ablate", help="TRTLLM_A2A_FT_ABLATE value for ft; adds the ft0 control graph (a2a-ablate)")
    parser.add_argument("--summarize", nargs="+", metavar="DIR")
    args = parser.parse_args()
    if args.summarize:
        summarize(args.summarize)
    elif args.out:
        bench(args)
    else:
        sys.exit("--out or --summarize is required")


if __name__ == "__main__":
    main()
