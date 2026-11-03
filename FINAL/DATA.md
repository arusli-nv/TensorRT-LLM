# WideEP Fault Tolerance: data package

Andrew Rusli | Oct 26, 2026 | Code: v5-final 27442bd91d | Companion to `review/DESIGN_DOC_final.md`

> **Updated 10-27 ~15:05 PT:** the recovery and restart blocks (`kill_runs`, `restart_runs`, `fig7_agg`, `fig7b_agg`,
> `figA4_agg`) include RECOV2 8353506, RECOV3 8361282, RECOV4 8363511 and RESTART2 8353534 (CLAIMS 227, 228, 235, 238);
> the after-loss blocks (`after_loss_runs`, `fig8_agg`) include RECOV4 (n=13 per loss, CLAIMS 239); `fig7b_agg` splits
> recovery into 7 steps (`kill_runs` `stage_s`, CLAIMS 241); `fig9_bins` holds all 8 losses (CLAIMS 236);
> `recreate.py` checks pass. The healthy-cost
> blocks include the 128/GPU replication (n=8, CLAIMS 232) and the realistic-traffic blocks S1B and S1C (6 pairs, CLAIMS 234, 240).
> Not in blocks: client throughput after prefill kills and offline stops (CLAIMS 229), exact-token checks (CLAIMS 230,
> 231, 233).

This file holds the data behind every data figure and headline table of the design doc, how each number was produced,
and a script that redraws the figures from the CSV blocks below. It contains only v5-final data, or FT-off runs on code
identical to v5-final. Data from older builds (pre-fix FT-on arms, fix-1, v3, Oct 20, the doc's Appendix I) is left out.

Paths are relative to the runs root `.wideep-ft-runs/claude/` (R): run directories are `R/mvp/runs/<RUN>-<JOB>/<case>/`,
readouts are in `R/review/diag/`, the harness is in `R/mvp/harness/`. "CLAIMS N" is row N of `R/mvp/CLAIMS.md`. A job ID
such as 8269102 is a Slurm job.

Contents:

1. Project context
2. Setup and configs
3. Metric definitions and statistics
4. Data per figure: Fig 5, 6, 7, 7b, 8, 9, 10, A1, A2, A3, A4, A5, and the remap table
5. Other headline tables: accuracy, memory, request fate, stress S1/S2/S3, per-step cost, pooled bounds
6. Recreating the figures
7. Known limits of the data

Every data block is a fenced `csv` block whose first line is `# block: <name>`; the script in section 6 reads them by
name.

## 1. Project context

**What WideEP FT is.** Without fault tolerance, losing one rank of a wide expert-parallel (WideEP) deployment ends
the whole job, and a restart takes 5-10 minutes. WideEP FT keeps the survivors serving instead. After a loss the
survivors keep the CUDA graphs captured at startup and change only the small tables those graphs read:

- the live-rank mask;
- an abort flag;
- EPLB's expert-to-slot tables;
- the contents of expert slots.

Three mechanisms make this possible:

- **Unbacked spare slots** (`extra_slots: ghost`). Spare expert slots at addresses the graphs already know. On a loss they get GPU memory, and the lost
  experts are copied in from host copies in `/dev/shm`.
- **Same-address remap.** Survivors move the all-to-all buffers to fresh memory at the same virtual addresses, so a dead
  rank's late NVLink writes land in memory no survivor reads.
- **Recovery authority.** A process outside MPI that admits a loss, fences (ends) the failed ranks, and makes every
  survivor apply the same plan, or none.

The deployment then serves on N-k ranks. A lost rank does not rejoin until a restart.

**What was measured, and why.**

1. **Healthy cost.** FT on vs FT off, same nodes, back to back. FT must cost little while nothing fails (goal: < 1%).
2. **Recovery time.** From the kill to the first step served after the recovery, for a rank kill, a node kill and a
   frozen rank. Compared with a restart without FT.
3. **Throughput after a loss.** Compared with the share of GPUs left.
4. **Realistic traffic.** Open-loop Poisson load, healthy and with random kills (stress tests S1, S2, S3).
5. **Correctness.** GSM8K accuracy of the recovered deployment, and the fate of every request.

**Code.** v5-final = commit `27442bd91d`. It ran as the wheel `build-freeze-58caa0fde6`
(`tensorrt_llm-1.4.0rc2-cp312-cp312-linux_aarch64.whl`, commit `58caa0fde6`) plus the Python overlay
`overlays/fix-dropmap-9f1fd4cab4`. The overlay is one file,
`tensorrt_llm/_torch/moe/fused_moe/moe_load_balancer.py`, with the same sha256 as at `27442bd91d`. The only product
difference between `58caa0fde6` and `27442bd91d` is that file, and its changes act only when FT is on
(`data_quality_audit.md` section 1). So an FT-off run on the bare `58caa0fde6` wheel runs the same code as v5-final with
FT off. 15 of the 20 pooled restarts (COLD-RESTART 8206870, RESTART2 8353534) are such runs; 5 come from the older v3 /
7e8dfb37df builds (CLAIMS 224, 228). Every other run here carries the overlay
(`MVP_BUILD` / `MVP_OVERLAY` in each case's `env-0.txt`). Two diagnostic figures use instrumented or ablation builds;
they are labelled where they appear (Fig A3, Fig A5).

**Hardware.** 32 NVIDIA GB200 GPUs: 8 nodes x 4 GPUs in one NVL72 rack (Slurm `segment 8`). Container
`nvcr.io/nvidia/tensorrt-llm/devel:1.4.0rc0.dev202610090112` (from the job logs, e.g.
`R/logs/slurm/slurm-8270779.out`).

**Model and parallelism.** DeepSeek-R1-0528 NVFP4 (`DeepSeek-R1-0528-FP4-V2`), DEP32: attention data parallel on each
of the 32 ranks, plus expert parallel 32 for the MoE layers. Overlap scheduler on; MTP with one draft token.

## 2. Setup and configs

### 2.1 Serving config

All arms share one config body. The six standard files (`R/mvp/harness/configs/std-*.yaml`) differ only in
`moe_config.load_balancer`. Below is the config of a representative healthy run,
`STD-G2-8270779/std-noeplb-ft-1/config-0.yaml` (no EPLB, FT on), exactly as written:

```yaml
tensor_parallel_size: 32
moe_expert_parallel_size: 32
enable_attention_dp: true
gpus_per_node: 4
max_batch_size: 512
max_num_tokens: 8192
max_seq_len: 2304
attn_backend: TRTLLM
attention_dp_config:
  enable_balance: true
  batching_wait_iters: 0
  timeout_iters: 60
cuda_graph_config:
  enable_padding: true
  max_batch_size: 512
kv_cache_config:
  dtype: fp8
  enable_block_reuse: false
  free_gpu_memory_fraction: 0.8
  host_cache_size: 17179869184      # 16 GiB host KV-cache offload tier
speculative_config:
  decoding_type: MTP
  max_draft_len: 1
stream_interval: 10
num_postprocess_workers: 4
print_iter_log: true
moe_config:
  backend: CUTEDSL
  use_low_precision_moe_combine: true
  load_balancer:                     # this block differs per arm, see the table below
    num_slots: 256
    layer_updates_per_iter: 0
```

`moe_config.load_balancer` per arm (from each arm's `config-0.yaml`):

| Mode | FT off | FT on |
|---|---|---|
| No EPLB | no `load_balancer` block | `num_slots: 256`, `layer_updates_per_iter: 0` (EPLB's static mode, needed for slot routing) |
| Offline EPLB | `num_slots: 288`, `layer_updates_per_iter: 0`, `initial_global_assignments` (59 layers x 288 slots) | identical to FT off |
| Online EPLB | `num_slots: 288`, `layer_updates_per_iter: 1` | identical to FT off |

The offline placement comes from `R/mvp/runs/STD-8109649/std-offline-placement.yaml` (harness `STD_PLACEMENT`). It
starts like this: `initial_global_assignments: {3: [88, 117, 146, 8, 1, 184, ...], ...}`. FT appends unbacked spare
slots at startup: every FT-on kill run logs 320 slots in total.

Other differences between arms:

- **FT settings.** These come from environment variables plus an FT config file (`ft-config-0.yaml`):
  - FT on, no EPLB: `extra_slots: ghost`, `failure_domain: node`, `hang_detection: true`, `hang_confirm_s: 30`;
  - FT on, offline or online EPLB: the same without `extra_slots` (the code picks the slot kind);
  - FT off: `failure_domain: node`, `hang_detection: true`, `hang_confirm_s: 30` are written, but without
    `TLLM_FAULT_TOLERANCE_MODE=1` they do nothing.
- **Kill runs (KILL-FINAL).** These add two CUDA graph batch sizes, 136 and 148 (`WIDEEP_EXTRA_GRAPH_BATCH_SIZES`), just
  above the survivors' per-rank batch after a loss. The configs are `std-noeplb-ft.yaml`, `std-online.yaml` and
  `std-offline-ft-filled.yaml`. The offline kill runs' `config-0.yaml` has the same `initial_global_assignments` as the
  healthy offline runs (checked: identical).
- **Stress runs.** These use `std-online-stress.yaml`: `std-online.yaml` with `max_seq_len: 6400` (the run's
  `config-0.yaml` differs from the healthy online config only in that line).

### 2.2 FT environment, launcher and authority

FT on (from `env-0.txt` of an FT-on case; the key is redacted there):

```
TLLM_FAULT_TOLERANCE_MODE=1
TLLM_FAULT_TOLERANCE_AUTHORITY=tcp://<node0 host>:<port>
TLLM_FAULT_TOLERANCE_AUTHORITY_KEY=<64 hex chars, os.urandom(32).hex()>
TLLM_FAULT_TOLERANCE_CONFIG=<run>/ft-config-<node>.yaml
TLLM_FAULT_TOLERANCE_EXTRA_SLOTS=ghost            # no-EPLB arms only
TLLM_FAULT_TOLERANCE_HANG_DETECTION=1
TLLM_FAULT_TOLERANCE_HANG_CONFIRM_S=30
```

All arms set the following:

- `TLLM_SPEC_DECODE_FORCE_NUM_ACCEPTED_TOKENS=1` (forced MTP acceptance; not set in the accuracy runs);
- `TRTLLM_ENABLE_PDL=1`;
- `TRTLLM_SERVER_DISABLE_GC=1` and `TRTLLM_WORKER_DISABLE_GC=1`;
- `TLLM_FMHA_LIBS=-cute_dsl_mla`.

The kill and stress runs also set `TRTLLM_NVLINK_ONE_SIDED_A2A_FORCE_CFT=0`, so the all-to-all uses its fence path.

**Authority.** On node 0, before the server starts (`R/mvp/harness/serve.sh`):
`python3 -m tensorrt_llm.executor.recovery.authority --endpoint tcp://*:<port> --world-size 32`.

**Launcher.** `R/mvp/harness/launch/task.sh`. The command below is followed by
`trtllm-llmapi-launch trtllm-serve <model> --config config-0.yaml`:

```
FT on:  mpirun --with-ft ulfm --mca pml ob1 --mca btl self,tcp --mca btl_tcp_if_include eth0 --mca coll ^hcoll,ucc \
          --host <8 nodes x 4> -np 32 --map-by ppr:4:node --bind-to none ...
FT off: mpirun --host <8 nodes x 4> -np 32 --map-by ppr:4:node --bind-to none ...   (Open MPI defaults, UCX)
```

The healthy runs log the MCA string in `timeline` (`MPIRUN_START ... mca=[...]`): FT arms show the line above, FT-off
arms show `mca=[]`. The kill and stress runs pass `--mca btl_tcp_if_include eth0 --mca coll ^hcoll,ucc` in both arms;
with FT on, the launcher selects ob1/TCP. The S1 FT-off arms ran plain `mpirun` (no ULFM) with those two options.
Open MPI's own failure detector is off.

### 2.3 Benchmark (healthy cost)

The driver is `R/mvp/harness/bench.py`. Each load runs TensorRT-LLM's `benchmark_serving`:

```
python3 -m tensorrt_llm.serve.scripts.benchmark_serving --model <R1> --tokenizer <R1> --backend openai --port 8000 \
  --dataset-name random --tokenize-on-client --random-ids --random-input-len 1024 --random-output-len 1024 \
  --random-range-ratio 0.0 --num-prompts <concurrency x waves / clients> --max-concurrency <concurrency / clients> \
  --seed <client index> --request-rate inf --burstiness 1.0 --ignore-eos --percentile-metrics ttft,tpot,itl,e2el \
  --save-result --save-detailed
```

- **Lengths.** Random token IDs, exactly 1024 in and 1024 out (`--ignore-eos`). All requests are released at once
  (`--request-rate inf`).
- **Load.** N/GPU means N x 32 concurrent requests. At 128/GPU the 4096 streams are split over 2 client processes of
  2048 each.
- **Waves.** `WIDEEP_LOAD_ITERATIONS=32=28,128=12`:
  - `healthy-32`: 28 waves, 28,672 requests, ~300 s;
  - `healthy-128`: 12 waves, 49,152 requests, ~300-340 s.
- **Warm-up.** First a burn-in (`WIDEEP_BURNIN=32:6`: 6 unmeasured waves at 32/GPU). Then, before each measured load,
  one unmeasured wave at its concurrency with 64 output tokens (`--warm-each-load`).
- **MTP.** One draft token, acceptance forced to 1.
- **Stream interval.** 10.
- **Pinned tactics.** Each arm's autotuner cache is seeded with the same `pinned.json` per mode (`FT_COST_E2E_SEED`).
  A pair is valid only if `tactic_check.txt` shows `PIN_CHECK OK` for both arms and equal `cache_miss`.
- **Validity, decided before any cost was computed** (`R/tools/finals_stats.py` PRE-REG 5/6). An arm is invalid if any
  of these holds; an invalid arm drops its whole pair:
  - result state is not DONE, or rc is not 0;
  - not every planned wave completed;
  - fewer than 20 steady steps;
  - a hang, peer-loss or crash line in the rank logs;
  - a new Xid/NVRM kernel line;
  - a ListenOverflows increase.
- **Jobs.**
  - 32/GPU: 8269103, 8270779 and 8271451 (original, CLAIMS 194), plus 8285727, 8287475, 8289412 and 8289662
    (replication, CLAIMS 206).
  - 128/GPU: 8269103, 8270779 and 8271451 (original), plus 8355466, 8355979 and 8357606 (replication, CLAIMS 232).
  - Each 16-node job runs two 8-node groups (`@n0`, `@n8`).
  - One replication online pair (STD-8285727 group n0) is excluded: its FT arm hung (CLAIMS 204). STD-8289662
    replaced it like for like.

### 2.4 Recovery runs (KILL-FINAL)

- **Load.** 128 requests/GPU (4096 concurrent), 1k/1k, forced MTP acceptance, 4 waves (16,384 requests).
- **Trigger.** The kill fires once rank 0 has logged 1260 full-load decode-only steps (`--kill-after-steps 1260`), which
  is mid-decode of the third wave. Before the load, one short wave each at 3 and 5 requests/GPU.
- **How a rank is killed.** The victim rank SIGKILLs itself (`os.kill(os.getpid(), SIGKILL)` in
  `R/mvp/harness/testhooks.py`):
  - it acts once the driver creates a trigger file;
  - at its next decode CUDA-graph replay that serves a real request, after that step's host collectives;
  - so the survivors are waiting for it inside the step's MoE all-to-all;
  - it logs `WIDEEP_SELF_KILL rank r t=<epoch>` just before the kill.
- **Rows.**
  - rank: one rank. No EPLB victims rotate 5, 22; online 13, 29; offline 5, 22.
  - node: the 4 ranks of one node, each SIGKILLed with the host alive (nodes 1 and 5, or 3 and 7).
  - frozen: rank 2 is SIGSTOPped by `bench.py` at 64/GPU, 8 waves; FT's hang detection must confirm and fence it.
  - prefill: the victim kills itself before a forward that holds context requests.
  - eplb_update: online only; the victim kills itself right after starting an iteration that updates EPLB weights.
  - twoloss: a second rank is killed after the first recovery committed.
- **Failure domain.** Node.
- **Jobs.**
  - No EPLB and online: V5F-REC 8269102 (CLAIMS 193).
  - Offline: V5F-REC-OFF 8283311 (CLAIMS 202).
  - Online EPLB-update row: 8271210.
  - Replication: REPL-REC 8285728 (CLAIMS 203).
  - THIN-CELLS 8302191 (CLAIMS 213).

### 2.5 Restart baseline (FT off)

- **Jobs.** COLD-RESTART 8206870 and RESTART2 8353534 (58caa0fde6 wheel), 8193512 and 8194123 (7e8dfb37df), 8134113 (v3)
  (CLAIMS 198, 211, 224, 228).
- **Code.** FT off, plain mpirun. 15 of 20 runs on the bare `58caa0fde6` wheel (code identical to v5-final with FT off);
  5 on the older builds above.
- **Kill.** Rank 5 is SIGKILLed 200 decode steps into a 128/GPU load. The job aborts.
- **Relaunch.** In the same allocation, page cache kept, autotuner cache kept.

### 2.6 Stress load (S1, S2, S3)

- **Driver.** `bench.py --mode stress` (`R/mvp/harness/stress_client.py` records every streamed event).
- **Lengths.** `--isl 2112 --osl 1056 --range-ratio 0.9394`: input uniform in 128-4096 tokens, output uniform in
  64-2048 tokens.
- **Arrivals.** Open loop, Poisson (`--burstiness 1`). 90% of requests go to `/v1/completions` and 10% to
  `/v1/chat/completions`, all streamed. Online EPLB, forced MTP acceptance.
- **Rate.**
  - S2 and S3: 0.8 x the case's own closed-loop 32/GPU request rate (`calib-32`), i.e. 50.3-51.6 req/s, for ~62 min.
  - S1: a fixed 51.0 req/s for ~35 min per arm (107,100 requests per arm).
- **Closed-loop rows.** A 32/GPU closed-loop row runs before (`calib-32`) and after (`calib-end-32`) the open-loop
  load.
- **Kills** (FT arms only). Each kill's ranks SIGKILL themselves at their next decode step that serves a request, or at
  their next context step. `at:S` fires S seconds after the open-loop load starts; `after:S` fires S seconds after the
  previous recovery resumed. Victims were drawn at random from ranks 1-31.
- **Jobs.** S1 8310100 (CLAIMS 217), S2 8299473 and 8318006 (CLAIMS 216, 219), S3 8314070 (CLAIMS 218).

### 2.7 Accuracy

- **Job.** ACCURACY-KILL-ON-8269102.
- **Benchmark.** GSM8K, 1319 questions, strict-match accuracy, through `trtllm-serve`. MTP acceptance not forced.
- **FT-on arm.** Rank 5 kills itself in a decode step of a short warm load. The evaluation runs on the 31 survivors
  after the recovery.
- **Reference.** FT off, no kill, same nodes.

## 3. Metric definitions and statistics

### 3.1 Healthy cost

- **Client output tok/s per GPU.** `benchmark_serving`'s `output_throughput` for the `healthy-<load>` row (completed
  output tokens / benchmark duration), divided by 32.
- **TPOT, TTFT.** `benchmark_serving`'s median and p99 over the row's requests.
- **Per-user tok/s** (Fig A2) = 1000 / median TPOT (ms).
- **Decode step time.** Rank 0's `host_step_time` from the `print_iter_log` line of each iteration (one loop of the
  overlap scheduler = one decode step). Steady steps (condensed from `R/mvp/analysis/dist32.py` `stats()`) are the steps inside the
  row's wall-clock window with no context request and at least 90% of the per-GPU load active:

  ```
  win = [x for x in its if start <= x.wall <= start + elapsed]              # the healthy-<load> row's window
  steady = [ms for wall, ms, tok, ctx, active in win if ctx == 0 and active >= 0.9 * load_per_gpu]
  med = statistics.median(steady)
  "trim_mean": statistics.mean(v for v in steady if v <= 1.5 * med)          # trimmed mean step
  "mean":      statistics.mean(steady)                                       # full mean step
  ```

  The **trimmed mean** drops only the steady steps longer than 1.5x the median (stall steps). The **full mean** keeps
  every steady step.
- **Per pair:** `% = 100 * (FT on - FT off) / FT off` (`finals_stats.py` `pct`). A positive step or latency % means FT
  is slower; a negative tok/s % means FT is slower. **Extra µs per step** = mean over pairs of (trim on - trim off).
- **Cell** (mode x load) = mean of the per-pair % over the n pairs, with a two-sided 95% Student t interval and
  one-sided 95% bounds (`finals_stats.py` `summarize`):

  ```
  se = statistics.stdev(values) / math.sqrt(n)
  half, one = t_ppf(0.975, n - 1) * se, t_ppf(0.95, n - 1) * se
  ci = (mean - half, mean + half); lb, ub = mean - one, mean + one
  ```

  t(0.975) = 3.182 (n=4), 2.365 (n=8); t(0.95) = 2.353 (n=4), 1.895 (n=8).
- **Pooled bound.** "FT's throughput loss is at most X%" means X = -LB of the client tok/s cell, so the one-sided 95%
  lower bound of (on - off) / off is -X%. The bound was checked two ways in `pooled_onesided_check.txt`.
- **Pairing and order.**
  - A pair = the FT-off and FT-on arms of one mode, run back to back on the same 8 nodes (`<job>@n0` or `<job>@n8`).
  - Each mode at each load has 4 pairs on 4 node sets: 2 run FT off first and 2 run FT on first (column `first`).
  - The 32/GPU replication adds 4 more pairs per mode, also 2/2, giving pooled n=8.
  - The off-first/on-first and n0/n8 splits are reported as checks only.
- **Means vs medians.** Healthy-cost cells, Fig 7 and Fig 7b's bars use means (Fig 7b's block also gives medians).
  Fig 8 ratios and the remap table use medians.

**Pre-registered rules** (`R/review/FINALS_PLAN.md` v6, `finals_stats.py` docstring):

- **"FT cost < 1% per step."** It may be claimed only if every headline cell's one-sided 95% upper bound is below 1%.
  - Rule A (pre-registered) requires this for both the trimmed and the full mean.
  - Rule B (trimmed only) was added on 10-23, after the rehearsal.
  - Original n=4 data: both rules fail (full mean UB up to +3.76%; trimmed UB +1.18% no EPLB, +1.09% online, both at
    32/GPU).
  - Pooled n=8 at 32/GPU: the trimmed UBs are below 1% and the full-mean UB is not (+2.18% offline). The doc therefore
    does not claim "< 1%".
- **Client tok/s.** Estimate and CI only; no "< 1%" claim.
- **Replication** (CLAIMS 206). The rule was set before any replication data: if the pooled offline-32 client CI
  includes 0, report "the original -2.0% did not replicate". It does include 0.

### 3.2 Recovery time (kill -> first step)

- **Phases** come from server timestamps. Rank 0 logs `WideEP recovery gen=1 phase=<p> t=<epoch s>` for each phase:
  detect, stopped, proposed, ready, install, installed, committed, resumed, first_step.
- **The kill time** t0 is the victim's `WIDEEP_SELF_KILL ... t=` (for a node, the earliest of the 4 victims). For the
  frozen rank, t0 is `bench.py`'s SIGSTOP wall time (`result.json` `kill_wall_s`).
- **Cumulative times.** Each phase is reported as `phase - t0` (`job.sh` `kf_verdict`, the `phases1=` field of each
  `KILL_FINAL_VERDICT` line in `kill_final.txt`).
- **Fig 7 time** = `first_step - t0`.
- **Fig 7b steps** (per run, then the mean and the median over runs; CLAIMS 241):
  - Detect = detect;
  - Agree = stopped - detect (revoke, shrink, roll back);
  - Map spare slots = proposed - stopped (rank 0 maps its spare-slot memory, then fences the failed ranks);
  - Stage copies = the slowest survivor's ring-copy staging (`stage_s`, no EPLB only; capped at ready - proposed);
  - Remap workspace = ready - proposed - stage;
  - Install = committed - ready;
  - First step = first_step - committed.
- **`stage_s`** is the largest `Staged N expert weight pieces in X ms` over the case's rank logs in the first recovery
  (`tools/recovery_substeps.py` -> `review/diag/recovery_substeps.txt`); 0 with EPLB.
- **Two-loss rows.** The second loss is timed from its own detect (`phases2=`).
- **Script-observed resume** (`script_resume_s`): kill -> the test script seeing rank 0's "resumed" line, polled every
  0.5 s.
- **Frozen rank, `confirm_s`.** The authority log's hang-confirmation stamp minus the SIGSTOP time.
- **Restart (FT off).** `down_s = exit_s + first_token_s`, where `exit_s` = kill -> every node's GPUs free, and
  `first_token_s` = relaunch `mpirun` start -> first request answered. The harness gap between the two (`gap_s`) and any
  scheduler wait are excluded.
- **Pass rule** (H23, set before data, `draft_cases/val_verdicts.py`). A row passes when all of these hold:
  - the server serves again;
  - the expected ranks are lost;
  - resume <= 10 s;
  - rank-0 estimated tok/s after / before >= 0.8 x live/32 (floor 0.775 rank, 0.70 node, 0.75 two losses);
  - >= 20 requests ran in full after the kill.

  Every row here passed.

### 3.3 Throughput after a loss (client after / before)

- **Script.** `R/review/diag/client_tput_after_kill.py`, which reads the case's `bench/load.json`.
- **Request timeline.** The client is closed loop at 4096 concurrency (frozen-rank rows: 2048), so the load runs in
  synchronized waves. Request start times are not saved, so they are replayed: a request starts when the earliest of the
  in-flight requests ends (FIFO). Its stream chunks sit at start + TTFT + cumsum(ITL), and each chunk carries
  output_len / n_chunks tokens.
- **Kill and resume in client time.** The median start and the median end of the one long inter-chunk gap of the
  requests in flight over the kill.
- **Window per wave.** The decode plateau: [99th percentile of the wave's first-token times + 0.5 s, 5th percentile of
  its completions - 0.5 s].
- **before** = mean client tok/s over every full plateau that ends before the kill (waves 1-2).
- **after** = the first full plateau that starts >= 2 s after the client-side resume (in practice 10-25 s after it;
  12-14.5 s long).
- **ratio** = after / before.
- **ideal** = live/32.
- **Other loss types** (`after_loss_other`): the frozen-rank rows also give the mean over all 5 post waves. Two-loss
  rows with no full wave after the second resume use a partial window.

### 3.4 Remap time

Each survivor logs `Remapped 7535067136 bytes of all-to-all workspace without ranks [...] in X ms, 0 bytes of them from
the KV cache`. The remap table reports the min / median / max of X over the survivors of one kill. "proposed -> ready"
= `ready - proposed` from the same run's phases.

The **remap phases** (Fig A5) come from an instrumented overlay (logging only):
`overlays/remap-timer-943a1e7841` = v5-final + timers, job REMAP-TIMER 8308514, CLAIMS 214. Each survivor logs
`Remap phases ms: release= barrier1= make_room= create= export= gather= import_map= outcome= remap_rest= barrier2=
total=`. The plotted survivor is the one with the median total (16th of 31), with:

- barrier = barrier1 + barrier2;
- create+export+gather = create + export + gather;
- rest = total - (sum of the other plotted parts).

### 3.5 Stress (S1, S2, S3)

Definitions from `R/mvp/harness/stress_readout.py`; rules set before data, `R/review/PRODUCT_STRESS_PLAN.md`:

- **Served output tok/s.** Per wall second, from each stream's chunk arrivals x that stream's tokens per chunk.
- **Kill k, before window** = [max(load start + 300 s, previous resume + 120 s), kill).
- **Kill k, after window** = [resume + 60 s, min(resume + 1260 s, next trigger, last arrival)).
- **Pass floors.** ratio = after / before must be >= 0.85, or >= 0.75 for a node kill. Capacity = calib-end-32 /
  calib-32 tok/s must clear the same floor.
- **Request classes.**
  - ok: HTTP 200 with exactly the requested tokens.
  - retryable: one well-formed `peer_loss_retryable` error event, then `[DONE]`.
  - Anything else fails the case.
- **Other checks.**
  - The failed requests must equal the server's count for the lost ranks.
  - No stall: no 60 s gap in rank-0 iterations or in streamed chunks.
  - `/health` must show `recovering` then `degraded`.
  - Memory (S1): growth over the window <= 256 MiB per GPU, <= 512 MiB RssAnon per rank, <= 2048 MiB host
    MemAvailable per node.
- **TTFT/TPOT.** "healthy" = requests before the first kill; "after" = requests after the last resume.
- **Fig 9.** Served tok/s summed per wall second, then averaged over 10 s bins aligned to floor(kill time). Offered load
  = requested output tokens of the requests that arrived in [-120 s, +300 s) / 420 s.

## 4. Data per figure

### Shared healthy-cost data (Figs 5, 6, A1, A2; per-step and bounds tables)

One row per FT-off / FT-on pair. Arms are case directories under `R/mvp/runs/`.

- 32/GPU: 24 pairs (8 per mode; `batch` = original, CLAIMS 194, or replication, CLAIMS 206).
- 128/GPU: 24 pairs (8 per mode; `batch` = original, CLAIMS 194, or replication, CLAIMS 232).

Columns:

- `trim_*`, `full_*`: rank-0 trimmed and full mean decode step (ms);
- `*_tok_s_gpu`: client output tok/s per GPU;
- TPOT in ms, TTFT in ms.

Sources:

- trim and full: `review/diag/repl_cost_pooled_pairs.csv` (32) and `review/v5f_pairs_full.csv` (128);
- client and latency: each arm's `bench.log` `healthy-<load>` row (matched to its pair by output tok/s, as
  `make_figs.py` does).

```csv
# block: healthy_pairs
load,mode,pair,batch,first,off_case,on_case,trim_off_ms,trim_on_ms,full_off_ms,full_on_ms,off_tok_s_gpu,on_tok_s_gpu,off_med_tpot_ms,on_med_tpot_ms,off_p99_tpot_ms,on_p99_tpot_ms,off_med_ttft_ms,on_med_ttft_ms
32,noeplb,8270779@n0,original,ft,STD-G2-8270779/std-noeplb-1,STD-G2-8270779/std-noeplb-ft-1,13.8678,14.0149,14.3612,14.5068,3066.0182,3044.2326,8.4229,8.4927,10.1994,10.2396,2053.52,2021.40
32,noeplb,8270779@n8,original,off,STD-G3-8270779/std-noeplb-1,STD-G3-8270779/std-noeplb-ft-1,13.7487,13.8832,14.6220,14.3380,3024.5197,3088.4966,8.8509,8.3450,10.3397,9.8734,1735.97,2119.42
32,noeplb,8271451@n0,original,off,STD-G2-8271451/std-noeplb-1,STD-G2-8271451/std-noeplb-ft-1,13.8612,13.9592,14.3306,14.4390,3070.6190,3071.5257,8.5552,8.3477,10.0310,10.0587,1880.06,2093.70
32,noeplb,8271451@n8,original,ft,STD-G3-8271451/std-noeplb-1,STD-G3-8271451/std-noeplb-ft-1,13.7919,13.9450,14.1844,14.3083,3039.7435,3057.4042,8.7107,8.5076,10.4985,10.0661,1831.24,1872.62
32,noeplb,8285727@n8,replication,ft,STD-G1-8285727/std-noeplb-1,STD-G1-8285727/std-noeplb-ft-1,13.8285,13.9241,14.3258,14.3685,3063.6916,3095.7719,8.4879,8.1300,10.0695,9.7065,1965.38,2190.59
32,noeplb,8287475@n0,replication,off,STD-G2-8287475/std-noeplb-1,STD-G2-8287475/std-noeplb-ft-1,13.7881,13.8891,14.1144,14.3189,3115.7545,3097.5583,8.1559,8.1895,9.6545,9.7126,2142.97,2146.71
32,noeplb,8289412@n0,replication,off,STD-G2-8289412/std-noeplb-1,STD-G2-8289412/std-noeplb-ft-1,13.8782,13.9761,14.2500,14.4194,3059.4337,3007.3916,8.5229,8.8401,10.1845,10.3722,2001.59,1773.04
32,noeplb,8289412@n8,replication,ft,STD-G3-8289412/std-noeplb-1,STD-G3-8289412/std-noeplb-ft-1,13.8848,14.0176,14.6052,14.3907,3051.6150,3088.2281,8.6883,8.3182,10.0965,10.0334,1786.98,2041.08
32,offline,8269103@n0,original,ft,STD-G2-8269103/std-offline-1,STD-G2-8269103/std-offline-ft-1,14.1434,14.2037,14.2789,14.3898,3432.6312,3381.1969,7.9385,8.1807,8.7117,8.9760,1367.52,1262.09
32,offline,8269103@n8,original,off,STD-G3-8269103/std-offline-1,STD-G3-8269103/std-offline-ft-1,14.1073,14.1968,14.2858,14.4297,3402.9466,3289.6393,8.2511,8.5845,9.2012,9.5434,1192.32,1222.83
32,offline,8270779@n8,original,ft,STD-G1-8270779/std-offline-1,STD-G1-8270779/std-offline-ft-1,14.0649,14.1272,14.2471,14.3667,3458.5064,3419.2368,7.9477,8.0057,8.7248,9.1050,1329.42,1364.52
32,offline,8271451@n8,original,off,STD-G1-8271451/std-offline-1,STD-G1-8271451/std-offline-ft-1,14.0716,14.1897,14.2235,14.8365,3379.9472,3307.4417,8.2328,8.3160,9.2789,9.5659,1275.01,1332.46
32,offline,8285727@n0,replication,ft,STD-G2-8285727/std-offline-1,STD-G2-8285727/std-offline-ft-1,14.1556,14.1970,14.2803,14.6152,3450.4479,3319.1901,8.0179,8.3842,8.7499,9.4816,1316.37,1276.25
32,offline,8287475@n0,replication,off,STD-8287475/std-offline-1,STD-8287475/std-offline-ft-1,14.1156,14.1554,14.2590,14.3226,3380.4853,3452.1636,8.2458,7.8693,9.1534,8.6676,1225.64,1390.69
32,offline,8287475@n8,replication,ft,STD-G3-8287475/std-offline-1,STD-G3-8287475/std-offline-ft-1,14.1726,14.2309,14.5079,14.4002,3349.1089,3443.8174,8.2605,7.9097,9.3412,8.6732,1320.66,1357.18
32,offline,8289412@n8,replication,off,STD-G1-8289412/std-offline-1,STD-G1-8289412/std-offline-ft-1,14.1556,14.2480,14.4834,14.5012,3374.5587,3424.2243,8.3307,8.0162,9.1682,8.9500,1176.78,1338.58
32,online,8269103@n0,original,off,STD-8269103/std-online-1,STD-8269103/std-online-ft-1,14.5326,14.7120,14.8241,15.1237,3056.5432,3002.7208,9.1733,9.0331,10.2370,10.2855,1285.60,1613.27
32,online,8269103@n8,original,ft,STD-G1-8269103/std-online-1,STD-G1-8269103/std-online-ft-1,14.7138,14.7256,14.9879,15.0714,2997.5239,2996.7754,8.8427,8.7706,10.4794,10.2484,1752.28,1922.63
32,online,8270779@n0,original,off,STD-8270779/std-online-1,STD-8270779/std-online-ft-1,14.6806,14.7374,15.1112,15.1506,2995.3539,3003.1531,8.6301,8.9203,10.1074,10.5164,2086.61,1715.23
32,online,8271451@n0,original,ft,STD-8271451/std-online-1,STD-8271451/std-online-ft-1,14.6607,14.6905,15.0836,15.0356,2988.4729,3012.1277,8.8375,8.6459,10.4584,10.0860,1829.86,2001.38
32,online,8285727@n8,replication,ft,STD-G3-8285727/std-online-1,STD-G3-8285727/std-online-ft-1,14.6660,14.6253,15.5966,15.0022,2947.7618,2998.1247,9.5519,8.5849,10.9272,10.0588,1377.58,2123.74
32,online,8287475@n8,replication,off,STD-G1-8287475/std-online-1,STD-G1-8287475/std-online-ft-1,14.5548,14.6727,15.2263,15.1366,2947.4166,2960.4723,8.9458,8.7355,13.3063,10.1844,1775.60,2037.66
32,online,8289412@n0,replication,ft,STD-8289412/std-online-1,STD-8289412/std-online-ft-1,14.6581,14.6694,15.0548,14.9604,2999.9279,2983.8639,8.5449,8.5779,10.0573,10.1325,2108.16,2110.55
32,online,8289662@n0,replication,off,STD-8289662/std-online-1,STD-8289662/std-online-ft-1,14.6081,14.6507,15.0042,14.9981,3011.6607,2997.9597,8.5356,8.6931,10.0043,10.0761,2027.09,2059.33
128,noeplb,8270779@n0,original,ft,STD-G2-8270779/std-noeplb-1,STD-G2-8270779/std-noeplb-ft-1,26.6529,26.9118,27.5847,27.2396,4871.9173,4911.1807,19.5878,19.9325,24.4770,24.3543,6719.67,6227.41
128,noeplb,8270779@n8,original,off,STD-G3-8270779/std-noeplb-1,STD-G3-8270779/std-noeplb-ft-1,26.5376,26.6200,26.8978,27.1880,4969.9375,4878.0856,20.1029,19.5876,24.2056,24.3031,5757.49,6831.55
128,noeplb,8271451@n0,original,off,STD-G2-8271451/std-noeplb-1,STD-G2-8271451/std-noeplb-ft-1,26.7977,26.9168,27.1923,27.6924,4867.3101,4803.8524,19.5110,19.8441,24.3560,24.8860,6910.45,6786.20
128,noeplb,8271451@n8,original,ft,STD-G3-8271451/std-noeplb-1,STD-G3-8271451/std-noeplb-ft-1,26.6458,26.7531,27.0634,27.6189,4960.3447,4849.2953,20.1831,19.5549,24.2258,24.4475,5775.93,6742.41
128,offline,8269103@n0,original,ft,STD-G2-8269103/std-offline-1,STD-G2-8269103/std-offline-ft-1,26.7024,26.7038,27.2167,27.3062,6134.4475,6107.4519,17.0083,17.0741,19.2903,19.3967,3774.71,3862.36
128,offline,8269103@n8,original,off,STD-G3-8269103/std-offline-1,STD-G3-8269103/std-offline-ft-1,26.3676,26.4937,27.0132,26.9822,6177.6977,6238.6964,17.0117,17.0051,19.3102,19.1178,3679.30,3571.12
128,offline,8270779@n8,original,ft,STD-G1-8270779/std-offline-1,STD-G1-8270779/std-offline-ft-1,26.1316,26.2418,26.9040,26.7194,6131.9823,6236.6484,17.0702,16.8038,19.4079,19.0279,3811.92,3702.54
128,offline,8271451@n8,original,off,STD-G1-8271451/std-offline-1,STD-G1-8271451/std-offline-ft-1,26.2585,26.3766,26.4418,27.1992,6362.6409,6130.3730,16.9635,16.9888,18.9560,19.4147,3295.73,3801.09
128,online,8269103@n0,original,off,STD-8269103/std-online-1,STD-8269103/std-online-ft-1,24.6349,24.7556,24.8794,24.9719,5455.3360,5377.4948,17.5367,17.7346,21.6448,22.0143,5888.02,6066.42
128,online,8269103@n8,original,ft,STD-G1-8269103/std-online-1,STD-G1-8269103/std-online-ft-1,24.7885,24.8958,25.0232,25.1927,5428.8128,5456.3879,17.8803,17.8536,22.0518,21.9375,5795.01,5666.54
128,online,8270779@n0,original,off,STD-8270779/std-online-1,STD-8270779/std-online-ft-1,24.7232,24.7534,24.9246,24.9867,5496.8337,5487.1710,17.5507,17.4256,21.5100,21.4850,5792.94,6006.06
128,online,8271451@n0,original,ft,STD-8271451/std-online-1,STD-8271451/std-online-ft-1,24.7154,24.8435,24.9458,25.1402,5430.6598,5478.9133,17.5813,17.4604,21.6804,21.8076,6148.47,5833.22
128,offline,8355466@n0,replication,ft,STD-G2-8355466/std-offline-1,STD-G2-8355466/std-offline-ft-1,26.3438,26.4848,26.6612,26.6214,6213.5373,6297.4020,16.9720,17.0090,19.1266,18.9568,3637.32,3392.16
128,online,8355466@n0,replication,off,STD-8355466/std-online-1,STD-8355466/std-online-ft-1,24.6813,24.8026,24.9088,25.0766,5487.8877,5420.1482,17.8579,17.5796,21.4413,22.0512,5588.88,6085.71
128,noeplb,8355466@n8,replication,ft,STD-G1-8355466/std-noeplb-1,STD-G1-8355466/std-noeplb-ft-1,26.7876,26.9702,27.6264,27.8734,4875.0205,4815.6125,19.5944,19.8732,24.4311,24.6916,6829.33,6798.61
128,online,8355466@n8,replication,ft,STD-G3-8355466/std-online-1,STD-G3-8355466/std-online-ft-1,24.4684,24.6014,24.7758,24.9072,5425.0553,5433.6879,17.5889,17.6018,21.6814,21.9235,6162.69,5978.28
128,noeplb,8355979@n0,replication,off,STD-G2-8355979/std-noeplb-1,STD-G2-8355979/std-noeplb-ft-1,27.2858,27.3941,27.9830,27.7133,4778.6195,4847.5752,19.9335,20.1235,24.6800,24.5626,7008.36,6418.57
128,offline,8355979@n0,replication,off,STD-8355979/std-offline-1,STD-8355979/std-offline-ft-1,26.7992,26.8865,27.3377,27.5307,6012.9090,6011.9210,17.3332,17.3061,19.7361,19.8151,3964.54,3990.51
128,offline,8355979@n8,replication,ft,STD-G3-8355979/std-offline-1,STD-G3-8355979/std-offline-ft-1,26.5600,26.6422,27.1928,27.2312,6140.6211,6032.9604,17.2520,17.2915,19.2835,19.7774,3504.13,3882.31
128,online,8355979@n8,replication,off,STD-G1-8355979/std-online-1,STD-G1-8355979/std-online-ft-1,24.8464,24.9084,25.2340,25.1904,5560.5987,5443.7300,17.4301,17.6678,21.7118,21.7866,5579.63,5879.79
128,noeplb,8357606@n0,replication,off,STD-G2-8357606/std-noeplb-1,STD-G2-8357606/std-noeplb-ft-1,26.8684,27.0332,28.0092,27.9573,4760.9785,4847.2867,19.9738,20.6925,24.9774,24.6753,6975.34,5826.37
128,online,8357606@n0,replication,ft,STD-8357606/std-online-1,STD-8357606/std-online-ft-1,24.5312,24.6237,25.0030,24.8906,5437.7577,5455.6020,17.7446,17.8764,21.7358,21.7757,5900.10,5672.95
128,noeplb,8357606@n8,replication,ft,STD-G3-8357606/std-noeplb-1,STD-G3-8357606/std-noeplb-ft-1,26.6792,26.8560,27.4572,27.3251,4862.0134,4856.5253,19.5600,20.1113,24.5361,24.7817,6783.30,6366.40
128,offline,8357606@n8,replication,off,STD-G1-8357606/std-offline-1,STD-G1-8357606/std-offline-ft-1,26.2592,26.4148,27.1132,26.7380,6049.2607,6199.6431,17.2608,17.0030,19.6257,19.1417,3888.47,3651.54
```

### Figure 5. Healthy throughput, FT off vs FT on

- **What it shows.** Mean output tok/s per GPU, FT off vs FT on, per mode, at 32 and 128 requests/GPU. The label is the
  mean % change.
- **Source.** Jobs and CLAIMS as in 2.3 (CLAIMS 194, 206).
- **Raw.** `healthy_pairs` (columns `off_tok_s_gpu`, `on_tok_s_gpu`).
- **Aggregation.**
  - Bar = mean over the pairs of the cell (n=8 at both loads).
  - Label = mean of the per-pair % (`output_throughput` rows of the block below).
- **Doc table.** "3061 -> 3069" etc. are these means. "Median TPOT off -> on" = mean over pairs of each arm's median
  TPOT (`median_tpot_ms` rows, `ft_off_mean` / `ft_on_mean`).

```csv
# block: healthy_latency_agg
metric,mode,load,n,mean_pct,ci_lo,ci_hi,ft_off_mean,ft_on_mean,unit_of_means
output_throughput,noeplb,32,8,+0.25,-0.78,+1.28,3061.4244,3068.8261,tok/s/GPU
output_throughput,noeplb,128,8,-0.33,-1.63,+0.96,4868.2677,4851.1767,tok/s/GPU
output_throughput,offline,32,8,-0.69,-2.80,+1.42,3403.5790,3379.6138,tok/s/GPU
output_throughput,offline,128,8,+0.08,-1.60,+1.77,6152.8871,6156.8870,tok/s/GPU
output_throughput,online,32,8,+0.05,-0.81,+0.91,2993.0826,2994.3997,tok/s/GPU
output_throughput,online,128,8,-0.38,-1.28,+0.51,5465.3677,5444.1419,tok/s/GPU
median_tpot_ms,noeplb,32,8,-1.75,-4.40,+0.90,8.5493,8.3963,ms
median_tpot_ms,noeplb,128,8,+0.82,-1.19,+2.84,19.8058,19.9650,ms
median_tpot_ms,offline,32,8,+0.10,-3.08,+3.28,8.1531,8.1583,ms
median_tpot_ms,offline,128,8,-0.28,-0.94,+0.37,17.1090,17.0602,ms
median_tpot_ms,online,32,8,-1.42,-4.80,+1.96,8.8827,8.7452,ms
median_tpot_ms,online,128,8,+0.02,-0.82,+0.87,17.6463,17.6500,ms
p99_tpot_ms,noeplb,32,8,-1.22,-3.29,+0.85,10.1342,10.0078,ms
p99_tpot_ms,noeplb,128,8,+0.42,-0.49,+1.34,24.4861,24.5878,ms
p99_tpot_ms,offline,32,8,+0.97,-3.49,+5.42,9.0412,9.1203,ms
p99_tpot_ms,offline,128,8,-0.05,-1.61,+1.52,19.3420,19.3310,ms
p99_tpot_ms,online,32,8,-3.90,-11.16,+3.36,10.6972,10.1985,ms
p99_tpot_ms,online,128,8,+0.77,-0.14,+1.68,21.6821,21.8477,ms
median_ttft_ms,noeplb,32,8,+6.07,-2.83,+14.97,1.9247,2.0323,s
median_ttft_ms,noeplb,128,8,-0.65,-10.95,+9.64,6.5950,6.4997,s
median_ttft_ms,offline,32,8,+3.62,-2.51,+9.74,1.2755,1.3181,s
median_ttft_ms,offline,128,8,+1.31,-5.35,+7.96,3.6945,3.7317,s
median_ttft_ms,online,32,8,+12.18,-5.47,+29.82,1.7803,1.9480,s
median_ttft_ms,online,128,8,+0.85,-3.39,+5.09,5.8570,5.8986,s
```

`ft_*_mean` are arm means in the unit of `unit_of_means`. TTFT means are in seconds, while its % is computed from ms.

### Figure 6. Change with FT on, per metric, 32/GPU

- **What it shows.** Mean % change with FT on and its 95% t-CI, for output tok/s, median TPOT, p99 TPOT and median TTFT,
  at 32/GPU (pooled n=8).
- **Source.** As Fig 5. The plotted % and CI come from `repl_cost_tpot_ttft.txt` (pooled, 32/GPU) and
  `v5f_tpot_ttft.txt` (128/GPU).
- **Raw.** `healthy_pairs`.
- **Aggregate.** The `load=32` rows of `healthy_latency_agg` above. The 128 rows are the doc's Appendix F values.
- **Aggregation.** Per pair `100 * (on - off) / off`. Mean and 95% t-CI over the 8 pairs (section 3.1).
- **Pooled TTFT** (doc text, not plotted; CLAIMS 208). Over all 24 pairs at 32/GPU, median TTFT is +7.29%
  [+1.32, +13.26] (df 23), and 19 of 24 pairs are positive. Single pairs at 32/GPU range from -3.80% to +2.83% in
  output tok/s.

### Figure 7. Time from failure until serving again

- **What it shows.** Mean kill -> rank 0's first step after the recovery, per mode, for a rank kill, a node kill and a
  frozen rank (log scale). It also shows a restart without FT after a rank kill.
- **Source.** KILL-FINAL runs listed in 2.4 plus RECOV2 8353506, RECOV3 8361282 and RECOV4 8363511 (CLAIMS 193, 202,
  203, 213, 227, 235, 238) and
  the pooled restarts
  of 2.5 (CLAIMS 224, 228).
- **Raw: kill runs.** One row per kill case, all PASS. Fig 7 uses `failure` in {rank, node, frozen}. The online rank
  cell includes the EPLB-update rows (`rank5upd-*`, jobs 8271210, 8353506 and 8361282), so n=8.
  - `*_s` columns: seconds since the kill (3.2).
  - `second_first_step_s`: the second loss's first step, from its own detect.
  - `script_resume_s`: script-observed resume (two losses: first+second). For the frozen rank it is measured from the
    stop.
  - `confirm_s`: the hang confirmation, frozen rank only.
  - Request columns: see 5.3.
  - `restored_mb_*`: the largest and the median restored-expert memory per survivor; `restored_ratio` = max / median.
  - `stage_s`: the slowest survivor's ring-copy staging in the first recovery (3.2, Fig 7b steps).

```csv
# block: kill_runs
failure,mode,source,row,lost_ranks,detect_s,stopped_s,proposed_s,ready_s,committed_s,first_step_s,second_first_step_s,script_resume_s,confirm_s,client_cut,client_failed,reruns,restored_mb_max,restored_mb_med,restored_ratio,stage_s
rank,noeplb,KILL-FINAL-OFF-8269102,rank5-1,5,0.01,0.06,0.22,1.47,1.48,1.55,,1.7,,128,0,0,444,360,1.23,0.273
node,noeplb,KILL-FINAL-OFF-8269102,node1-1,4+5+6+7,0.03,0.08,0.76,2.45,2.47,2.54,,2.71,,512,0,0,1752,1668,1.05,1.051
rank,noeplb,KILL-FINAL-OFF-8269102,rank5-2,22,0.02,0.05,0.21,1.41,1.42,1.48,,1.69,,128,0,0,444,360,1.23,0.232
node,noeplb,KILL-FINAL-OFF-8269102,node1-2,20+21+22+23,0.03,0.08,0.72,2.43,2.45,2.52,,2.7,,512,0,0,1752,1668,1.05,0.926
rank,noeplb,KILL-FINAL-OFF-8285728,rank5-1,5,0.02,0.07,0.25,1.45,1.47,1.53,,1.71,,128,0,0,444,360,1.23,0.204
node,noeplb,KILL-FINAL-OFF-8285728,node1-1,4+5+6+7,0.03,0.08,0.73,2.34,2.35,2.42,,2.71,,512,0,0,1752,1668,1.05,0.922
rank,noeplb,KILL-FINAL-OFF-8285728,rank5-2,22,0.01,0.06,0.24,1.38,1.39,1.45,,1.52,,128,0,0,444,360,1.23,0.241
node,noeplb,KILL-FINAL-OFF-8285728,node1-2,20+21+22+23,0.02,0.07,0.72,2.70,2.71,2.81,,3.22,,512,0,0,1752,1668,1.05,0.980
twoloss,noeplb,KILL-FINAL-OFF-A-8269102,twokill-1,5+9,0.01,0.06,0.23,1.56,1.57,1.73,2.04,1.71+2.07,,253,0,7,912,744,1.23,0.272
frozen,noeplb,KILL-FINAL-OFF-A-8269102,stop-1,2,35.58,35.63,35.92,37.15,37.16,37.21,,37.6,34.1,64,0,0,444,360,1.23,0.225
prefill,noeplb,KILL-FINAL-OFF-A-8269102,rank5pf-1,5,0.01,0.06,0.27,1.43,1.44,2.44,,1.71,,8,0,2,444,360,1.23,0.258
twoloss,noeplb,KILL-FINAL-OFF-A-8302191,twokill-1,5+9,0.01,0.07,0.23,1.51,1.53,1.65,1.69,1.7+2.12,,258,0,2,912,744,1.23,0.261
frozen,noeplb,KILL-FINAL-OFF-A-8302191,stop-1,2,35.88,35.93,36.09,37.36,37.37,37.42,,37.6,34.11,64,0,0,444,360,1.23,0.230
rank,offline,KILL-FINAL-OFF-B-8283311,rank5-1,5,0.03,0.11,0.26,1.26,1.31,1.37,,1.71,,128,0,0,420,360,1.17,0.000
node,offline,KILL-FINAL-OFF-B-8283311,node1-1,4+5+6+7,0.03,0.08,0.69,1.49,1.65,1.72,,1.71,,512,0,0,1476,1476,1.00,0.000
rank,offline,KILL-FINAL-OFF-B-8302191,rank5-1,5,0.01,0.06,0.22,1.12,1.16,1.23,,1.19,,128,0,0,420,360,1.17,0.000
node,offline,KILL-FINAL-OFF-B-8302191,node1-1,4+5+6+7,0.02,0.07,0.73,1.33,1.48,1.55,,1.69,,512,0,0,1476,1476,1.00,0.000
rank,offline,KILL-FINAL-OFF-B-8302191,rank5-2,22,0.01,0.06,0.22,1.14,1.18,1.24,,1.51,,128,0,0,420,336,1.25,0.000
node,offline,KILL-FINAL-OFF-B-8302191,node1-2,20+21+22+23,0.02,0.07,0.73,1.35,1.54,1.61,,1.71,,512,0,0,1476,1476,1.00,0.000
rank,online,KILL-FINAL-ON-8269102,rank5-1,13,0.02,0.08,0.28,0.82,0.84,0.91,,1.21,,128,0,0,396,312,1.27,0.000
node,online,KILL-FINAL-ON-8269102,node1-1,12+13+14+15,0.03,0.08,0.81,1.29,1.35,1.42,,1.72,,512,0,0,1476,1476,1.00,0.000
rank,online,KILL-FINAL-ON-8269102,rank5-2,29,0.01,0.04,0.24,0.66,0.69,0.76,,1.2,,128,0,0,396,336,1.18,0.000
node,online,KILL-FINAL-ON-8269102,node1-2,28+29+30+31,0.01,0.18,1.03,1.71,1.76,1.83,,1.9,,512,0,0,1476,1476,1.00,0.000
rank,online,KILL-FINAL-ON-8285728,rank5-1,13,0.03,0.08,0.28,0.79,0.81,0.89,,1.21,,128,0,0,396,312,1.27,0.000
node,online,KILL-FINAL-ON-8285728,node1-1,12+13+14+15,0.03,0.09,0.88,1.29,1.35,1.42,,1.71,,512,0,0,1476,1476,1.00,0.000
rank,online,KILL-FINAL-ON-8285728,rank5-2,29,0.02,0.07,0.26,0.72,0.77,0.83,,1.2,,128,0,0,396,312,1.27,0.000
node,online,KILL-FINAL-ON-8285728,node1-2,28+29+30+31,0.03,0.09,0.79,1.40,1.45,1.53,,1.7,,512,0,0,1476,1476,1.00,0.000
twoloss,online,KILL-FINAL-ON-A-8269102,twokill-1,13+17,0.01,0.06,0.26,0.72,0.75,0.82,1.77,1.21+1.56,,146,0,7,780,696,1.12,0.000
frozen,online,KILL-FINAL-ON-A-8269102,stop-1,2,32.15,32.20,32.40,32.99,33.01,33.07,,33.09,29.99,64,0,0,396,336,1.18,0.000
prefill,online,KILL-FINAL-ON-A-8269102,rank5pf-1,13,0.01,0.06,0.42,0.90,0.93,1.88,,1.06,,120,0,4,396,336,1.18,0.000
twoloss,online,KILL-FINAL-ON-A-8302191,twokill-1,13+17,0.01,0.04,0.23,0.95,0.97,1.14,0.85,1.2+1.51,,258,0,2,780,696,1.12,0.000
frozen,online,KILL-FINAL-ON-A-8302191,stop-1,2,32.09,32.15,32.34,32.78,32.81,32.86,,33.1,30.53,64,0,0,396,336,1.18,0.000
rank,online,KILL-FINAL-ON-A-8271210,rank5upd-1,13,0.02,0.05,0.28,0.75,0.78,0.84,,1.19,,128,0,0,396,336,1.18,0.000
prefill,online,KILL-FINAL-ON-8353506,rank5pf-1,13,0.02,0.07,0.27,0.72,0.74,1.73,,1.2,,0,0,1,396,336,1.18,0.000
prefill,online,KILL-FINAL-ON-8353506,rank5pf-2,29,0.01,0.03,0.22,0.76,0.78,1.65,,1.19,,121,0,1,396,336,1.18,0.000
rank,online,KILL-FINAL-ON-A-8353506,rank5upd-1,13,0.02,0.07,0.26,0.87,0.90,0.96,,1.01,,128,0,0,396,336,1.18,0.000
prefill,noeplb,KILL-FINAL-OFF-8353506,rank5pf-1,5,0.01,0.06,0.25,1.47,1.49,2.72,,1.68,,0,0,1,444,360,1.23,0.239
frozen,offline,KILL-FINAL-OFF-B-8353506,stop-1,2,32.44,32.49,32.65,33.59,33.63,34.14,,34.1,30.1,64,0,0,420,336,1.25,0.000
prefill,offline,KILL-FINAL-OFF-B-8353506,rank5pf-1,5,0.02,0.07,0.23,1.25,1.29,1.89,,1.53,,110,0,1,420,360,1.17,0.000
frozen,offline,KILL-FINAL-OFF-B-8353506,stop-2,2,31.67,31.69,31.89,32.80,32.86,32.91,,33.11,29.49,64,0,0,420,336,1.25,0.000
prefill,offline,KILL-FINAL-OFF-B-8353506,rank5pf-2,22,0.01,0.06,0.22,1.21,1.25,2.14,,1.68,,0,0,2,420,336,1.25,0.000
twoloss,offline,KILL-FINAL-OFF-B-8361282,twokill-1,5+9,0.01,0.06,0.23,1.21,1.25,1.32,1.55,1.7+1.97,,253,0,7,804,744,1.08,0.000
twoloss,offline,KILL-FINAL-OFF-B-8361282,twokill-2,22+26,0.01,0.05,0.20,1.24,1.30,1.36,2.18,1.7+2.24,,155,0,1,804,696,1.16,0.000
rank,online,KILL-FINAL-ON-A-8361282,rank5upd-1,13,0.02,0.13,0.33,0.81,0.84,1.00,,1.16,,128,0,0,396,336,1.18,0.000
rank,online,KILL-FINAL-ON-A-8361282,rank5upd-2,29,0.01,0.06,0.27,0.76,0.78,0.85,,1.2,,128,0,0,396,336,1.18,0.000
prefill,noeplb,KILL-FINAL-OFF-8361282,rank5pf-1,5,0.02,0.07,0.24,1.50,1.51,2.52,,1.59,,2,0,1,444,360,1.23,0.258
rank,offline,KILL-FINAL-OFF-B-8363511,rank5-1,5,0.01,0.07,0.23,1.16,1.20,1.33,,1.21,,128,0,0,420,360,1.17,0.000
node,offline,KILL-FINAL-OFF-B-8363511,node1-1,4+5+6+7,0.02,0.07,0.64,1.30,1.44,1.51,,1.72,,512,0,0,1476,1476,1.00,0.000
rank,offline,KILL-FINAL-OFF-B-8363511,rank5-2,22,0.02,0.06,0.24,1.13,1.17,1.24,,1.21,,128,0,0,420,336,1.25,0.000
node,offline,KILL-FINAL-OFF-B-8363511,node1-2,20+21+22+23,0.03,0.11,0.72,1.37,1.49,1.72,,1.7,,512,0,0,1476,1476,1.00,0.000
frozen,offline,KILL-FINAL-OFF-A-8363511,stop-1,2,32.61,32.66,32.94,33.89,33.93,33.98,,34.1,29.99,64,0,0,420,336,1.25,0.000
frozen,noeplb,KILL-FINAL-OFF-8363511,stop-1,2,32.57,32.62,32.80,34.05,34.06,34.11,,34.08,30.32,64,0,0,444,360,1.23,0.245
frozen,online,KILL-FINAL-ON-B-8363511,stop-1,2,34.91,34.96,35.19,35.71,35.74,35.79,,36.08,33.4,64,0,0,420,336,1.25,0.000
```

- **Raw: restart runs** (FT off). `down_s` is the plotted value. `rank0_model_load_s` is rank 0's
  `total_model_loading_seconds` from the relaunch's `rank-0.log` (CLAIMS 211).

```csv
# block: restart_runs
mode,source,run,relaunch_case,detect_s,exit_s,weights_s,warmup_s,graphs_s,ready_s,first_token_s,down_s,rank0_model_load_s,harness_gap_s_excluded,failed_inflight
noeplb,COLD-RESTART-A-8206870,1,restart-after-rank5-1,0.0,6.0,82.7,146.7,305.6,310.1,312.1,318.1,61.2,12.6,4096
online,COLD-RESTART-8206870,1,rank5-2,2.4,10.9,327.7,393.7,567.0,571.8,573.5,584.4,307.0,29.1,4096
online,COLD-RESTART-8206870,2,restart-after-rank5-2,0.0,11.0,361.5,426.5,607.4,611.8,614.1,625.1,341.5,13.2,4096
offline,COLD-RESTART-B-8206870,1,rank5-2,0.0,6.0,299.6,365.6,529.0,533.6,535.3,541.3,276.8,13.9,4096
offline,COLD-RESTART-B-8206870,2,restart-after-rank5-2,0.0,9.8,80.2,146.2,317.9,321.9,323.7,333.5,57.7,7.5,4096
noeplb,COLD-RESTART-A-8193512,1,restart-after-rank5-1,0.0,10.3,68.7,130.7,284.1,288.7,290.6,300.9,48.4,7.9,4096
noeplb,COLD-RESTART-A-8134113,1,restart-after-rank5-1,0.0,9.9,76.6,166.6,305.5,309.3,311.4,321.3,55.4,24.8,4096
noeplb,COLD-RESTART-A-8353534,1,rank5-2,0.0,10.1,185.7,248.7,399.1,403.9,405.9,416.0,164.5,25.6,4096
noeplb,COLD-RESTART-A-8353534,2,rank5-3,0.0,9.9,79.3,201.3,354.7,358.6,360.4,370.3,58.6,10.8,4096
noeplb,COLD-RESTART-A-8353534,3,restart-after-rank5-3,0.0,10.5,78.7,141.7,293.3,297.4,299.3,309.8,59.3,16.3,4096
offline,COLD-RESTART-B-8194123,1,restart-after-rank5-1,0.0,9.8,77.8,142.8,303.8,308.4,310.1,319.9,57.2,7.2,4096
offline,COLD-RESTART-B-8134113,1,restart-after-rank5-1,1.5,9.9,92.8,172.8,335.4,339.2,341.4,351.3,70.6,24.9,3111
offline,COLD-RESTART-B-8353534,1,rank5-2,0.0,10.1,89.4,152.4,318.0,322.0,323.8,333.9,68.6,10.1,4096
offline,COLD-RESTART-B-8353534,2,rank5-3,0.0,9.7,78.6,141.6,308.0,313.0,314.8,324.5,57.0,10.4,4096
offline,COLD-RESTART-B-8353534,3,restart-after-rank5-3,0.0,9.6,85.5,215.5,379.8,384.1,386.0,395.6,64.2,8.3,4096
online,COLD-RESTART-8194123,1,restart-after-rank5-1,2.2,9.5,344.6,408.6,582.2,586.3,588.0,597.5,322.5,13.6,4096
online,COLD-RESTART-8353534,1,rank5-2,2.1,10.6,366.1,491.1,676.5,680.6,682.4,693.0,344.4,28.1,4096
online,COLD-RESTART-8353534,2,rank5-3,2.1,10.4,325.4,390.4,563.7,568.0,569.8,580.2,304.2,13.6,4096
online,COLD-RESTART-8353534,3,rank5-4,2.0,10.2,397.4,463.4,646.6,651.3,653.1,663.3,377.1,12.1,4096
online,COLD-RESTART-8353534,4,restart-after-rank5-4,2.0,10.6,385.0,451.0,622.1,626.4,629.5,640.1,363.4,12.9,4096
```

- **Aggregate as plotted.** Mean, min and max over runs (`fig7_recovery.csv`). The restart row is the mean of all 20
  restarts (all modes).

```csv
# block: fig7_agg
failure,mode,n,mean_s,min_s,max_s
rank,noeplb,4,1.503,1.45,1.55
rank,offline,5,1.282,1.23,1.37
rank,online,8,0.880,0.76,1.00
rank,FT off restart,20,451.0,300.9,693.0
node,noeplb,4,2.573,2.42,2.81
node,offline,5,1.622,1.51,1.72
node,online,4,1.550,1.42,1.83
frozen,noeplb,3,36.247,34.11,37.42
frozen,offline,3,33.677,32.91,34.14
frozen,online,3,33.907,32.86,35.79
```

- **Doc ranges.**
  - All rank kills span 0.76-1.55 s ("0.8-1.6 s") and all node kills 1.42-2.81 s ("1.4-2.8 s").
  - Restarts span 300.9-693.0 s.
  - Frozen rank: confirmed (fenced) at 34.10 / 34.11 / 30.32 s with no EPLB, 30.1 / 29.49 / 29.99 s offline and
    29.99 / 30.53 / 33.4 s online (`confirm_s`).

### Figure 7b. Where the recovery time goes

- **What it shows.** The mean duration of each recovery step, per mode, for rank and node kills: detect + agree, map
  spare slots, remap workspace, stage copies (no EPLB), install + first step.
- **Source.** The same runs as Fig 7 (rank and node rows); staging from `review/diag/recovery_substeps.txt` (CLAIMS 241).
- **Raw.** `kill_runs` (cumulative phase columns and `stage_s`).
- **Aggregation.** Per run, the step durations of 3.2. Then the mean (plotted; the stacked total equals Fig 7's mean)
  and the median per step over the runs of the cell.

```csv
# block: fig7b_agg
failure,mode,phase,median_s,mean_s,n
rank,noeplb,detect,0.015,0.0150,4
rank,noeplb,agree,0.050,0.0450,4
rank,noeplb,map,0.170,0.1700,4
rank,noeplb,remap,0.972,0.9599,4
rank,noeplb,stage,0.237,0.2376,4
rank,noeplb,install,0.010,0.0125,4
rank,noeplb,first,0.060,0.0625,4
rank,offline,detect,0.010,0.0160,5
rank,offline,agree,0.050,0.0560,5
rank,offline,map,0.160,0.1620,5
rank,offline,remap,0.920,0.9280,5
rank,offline,stage,0.000,0.0000,5
rank,offline,install,0.040,0.0420,5
rank,offline,first,0.070,0.0780,5
rank,online,detect,0.020,0.0188,8
rank,online,agree,0.050,0.0537,8
rank,online,map,0.200,0.2025,8
rank,online,remap,0.485,0.4975,8
rank,online,stage,0.000,0.0000,8
rank,online,install,0.030,0.0287,8
rank,online,first,0.070,0.0787,8
node,noeplb,detect,0.030,0.0275,4
node,noeplb,agree,0.050,0.0500,4
node,noeplb,map,0.650,0.6550,4
node,noeplb,remap,0.736,0.7778,4
node,noeplb,stage,0.953,0.9697,4
node,noeplb,install,0.015,0.0150,4
node,noeplb,first,0.070,0.0775,4
node,offline,detect,0.020,0.0240,5
node,offline,agree,0.050,0.0560,5
node,offline,map,0.610,0.6220,5
node,offline,remap,0.650,0.6660,5
node,offline,stage,0.000,0.0000,5
node,offline,install,0.150,0.1520,5
node,offline,first,0.070,0.1020,5
node,online,detect,0.030,0.0250,4
node,online,agree,0.060,0.0850,4
node,online,map,0.760,0.7675,4
node,online,remap,0.545,0.5450,4
node,online,stage,0.000,0.0000,4
node,online,install,0.055,0.0550,4
node,online,first,0.070,0.0725,4
```

### Figure 8. Throughput after a loss

- **What it shows.** Client throughput after / before each loss, as the median per mode. The dashed line is the share of
  GPUs left (31/32 = 0.9688, 28/32 = 0.875).
- **Source.**
  - Readout: `review/diag/client_tput_after_kill.txt`.
  - Runs: the rank5/node1 rows of KILL-FINAL-OFF/ON-8269102, -8285728, OFF-B-8283311, OFF-B-8302191 and OFF-B-8363511.
  - CLAIMS 209 (method), CLAIMS 220 (the earlier n=11 medians) and CLAIMS 239 (n=13 with RECOV4).
- **Raw.** One row per kill (`after_window` = which wave's plateau was used).

```csv
# block: after_loss_runs
failure,mode,source,row,client_tok_s_before,client_tok_s_after,after_window,ratio,share_of_gpus_left
rank,noeplb,KILL-FINAL-OFF-8269102,rank5-1,278049,251079,wave4,0.903,0.9688
rank,noeplb,KILL-FINAL-OFF-8269102,rank5-2,279503,268682,wave4,0.961,0.9688
rank,noeplb,KILL-FINAL-OFF-8285728,rank5-1,279111,263691,wave4,0.945,0.9688
rank,noeplb,KILL-FINAL-OFF-8285728,rank5-2,278732,266600,wave4,0.956,0.9688
rank,offline,KILL-FINAL-OFF-B-8283311,rank5-1,283125,273794,wave4,0.967,0.9688
rank,offline,KILL-FINAL-OFF-B-8302191,rank5-1,279316,258318,wave4,0.925,0.9688
rank,offline,KILL-FINAL-OFF-B-8302191,rank5-2,281370,268434,wave4,0.954,0.9688
rank,offline,KILL-FINAL-OFF-B-8363511,rank5-1,284393,256082,wave4,0.900,0.9688
rank,offline,KILL-FINAL-OFF-B-8363511,rank5-2,280605,257139,wave4,0.916,0.9688
rank,online,KILL-FINAL-ON-8269102,rank5-1,298683,280403,wave4,0.939,0.9688
rank,online,KILL-FINAL-ON-8269102,rank5-2,298339,272792,wave4,0.914,0.9688
rank,online,KILL-FINAL-ON-8285728,rank5-1,296108,282916,wave4,0.955,0.9688
rank,online,KILL-FINAL-ON-8285728,rank5-2,300827,287546,wave4,0.956,0.9688
node,noeplb,KILL-FINAL-OFF-8269102,node1-1,278490,244149,wave4,0.877,0.8750
node,noeplb,KILL-FINAL-OFF-8269102,node1-2,280625,241355,wave4,0.860,0.8750
node,noeplb,KILL-FINAL-OFF-8285728,node1-1,277699,244424,wave4,0.880,0.8750
node,noeplb,KILL-FINAL-OFF-8285728,node1-2,278101,238056,wave4,0.856,0.8750
node,offline,KILL-FINAL-OFF-B-8283311,node1-1,285617,245072,wave4,0.858,0.8750
node,offline,KILL-FINAL-OFF-B-8302191,node1-1,281879,248376,wave4,0.881,0.8750
node,offline,KILL-FINAL-OFF-B-8302191,node1-2,279657,241127,wave4,0.862,0.8750
node,offline,KILL-FINAL-OFF-B-8363511,node1-1,283673,247560,wave4,0.873,0.8750
node,offline,KILL-FINAL-OFF-B-8363511,node1-2,282566,242584,wave4,0.859,0.8750
node,online,KILL-FINAL-ON-8269102,node1-1,298303,255134,wave4,0.855,0.8750
node,online,KILL-FINAL-ON-8269102,node1-2,291360,249521,wave4,0.856,0.8750
node,online,KILL-FINAL-ON-8285728,node1-1,293933,262039,wave4,0.891,0.8750
node,online,KILL-FINAL-ON-8285728,node1-2,298594,253410,wave4,0.849,0.8750
```

- **Aggregation.** Median of `ratio` over the runs of each mode; "all modes" = median over all 13.

```csv
# block: fig8_agg
failure,mode,n,median_ratio,share_of_gpus_left
rank,noeplb,4,0.9505,0.9688
rank,offline,5,0.9250,0.9688
rank,online,4,0.9470,0.9688
rank,all modes,13,0.9450,0.9688
node,noeplb,4,0.8685,0.8750
node,offline,5,0.8620,0.8750
node,online,4,0.8555,0.8750
node,all modes,13,0.8600,0.8750
```

- **Doc text.** Other loss types (not plotted; same method; DIAGNOSTIC rows, CLAIMS 209 and 213):

```csv
# block: after_loss_other
failure,mode,case,lost,live,client_tok_s_before,client_tok_s_after,after_window,ratio_first_window,ideal,ratio_all_post_waves,n_post_waves
frozen,noeplb,KILL-FINAL-OFF-A-8269102/stop-1,2,31/32,211209,172904,wave4,0.819,0.969,0.880,5
frozen,noeplb,KILL-FINAL-OFF-A-8302191/stop-1,2,31/32,212266,176231,wave4,0.830,0.969,0.866,5
frozen,online,KILL-FINAL-ON-A-8269102/stop-1,2,31/32,210798,184399,wave4,0.875,0.969,0.874,5
frozen,online,KILL-FINAL-ON-A-8302191/stop-1,2,31/32,209011,182577,wave4,0.874,0.969,0.892,5
prefill,noeplb,KILL-FINAL-OFF-A-8269102/rank5pf-1,5,31/32,281815,250846,wave3,0.890,0.969,0.910,2
prefill,online,KILL-FINAL-ON-A-8269102/rank5pf-1,13,31/32,299789,217732,wave3,0.726,0.969,0.818,2
twoloss,noeplb,KILL-FINAL-OFF-A-8269102/twokill-1,5+9,30/32,278773,253966,partial,0.911,0.938,-,0
twoloss,noeplb,KILL-FINAL-OFF-A-8302191/twokill-1,5+9,30/32,281970,254458,partial,0.902,0.938,-,0
twoloss,online,KILL-FINAL-ON-A-8269102/twokill-1,13+17,30/32,297728,271345,wave4,0.911,0.938,0.911,1
twoloss,online,KILL-FINAL-ON-A-8302191/twokill-1,13+17,30/32,297850,268916,partial,0.903,0.938,-,0
```

### Figure 9. Throughput around a kill under realistic traffic

- **What it shows.** Served output tok/s (FT on) in 10 s bins, from 2 min before to 5 min after a kill. The dashed
  line is the offered load. The doc figure draws three losses (`in_doc_figure` = yes); the slide chart (backup B15)
  averages the 7 rank kills (CLAIMS 236).
- **Source.** All 8 losses under realistic traffic (`case` column):
  - S2 `STRESS-8299473/s2-rank14` and `STRESS-G1-8299473/s2-rank13` (CLAIMS 216); `STRESS-8318006/s2-rank25` and the
    node kill `STRESS-G1-8318006/s2-node2` (CLAIMS 219);
  - S3 `STRESS-8314070/s3-rank29-rank14` and `STRESS-G1-8314070/s3-rank12-rank31` (two losses each; CLAIMS 218).
  - Per-token timestamps from `bench/stress{,-chat}.sse.jsonl`; kill times from `result.json`.
- **Raw and as plotted.** The bins below are the plotted series (`noise_band_*` = offered x (1 +- 2 SD of the pooled
  pre-kill bins of the drawn losses)). The per-request client records they are binned from (~190k streams per case) are
  too large to include. They are in the run directories.
- **Aggregation.** 3.5 (Fig 9).

```csv
# block: fig9_bins
test,kill,ranks,kill_wall_s,resume_s,bin_start_s_from_kill,bin_end_s_from_kill,served_out_tok_s,offered_out_tok_s,noise_band_lo,noise_band_hi,case,loss,in_doc_figure
S2,1,14,1792993943.502,1.03,-120,-110,50145.4,52166.3,46987.5,57345.1,STRESS-8299473/s2-rank14,rank,yes
S2,1,14,1792993943.502,1.03,-110,-100,55304.0,52166.3,46987.5,57345.1,STRESS-8299473/s2-rank14,rank,yes
S2,1,14,1792993943.502,1.03,-100,-90,53833.1,52166.3,46987.5,57345.1,STRESS-8299473/s2-rank14,rank,yes
S2,1,14,1792993943.502,1.03,-90,-80,50087.5,52166.3,46987.5,57345.1,STRESS-8299473/s2-rank14,rank,yes
S2,1,14,1792993943.502,1.03,-80,-70,51919.8,52166.3,46987.5,57345.1,STRESS-8299473/s2-rank14,rank,yes
S2,1,14,1792993943.502,1.03,-70,-60,55053.4,52166.3,46987.5,57345.1,STRESS-8299473/s2-rank14,rank,yes
S2,1,14,1792993943.502,1.03,-60,-50,53285.2,52166.3,46987.5,57345.1,STRESS-8299473/s2-rank14,rank,yes
S2,1,14,1792993943.502,1.03,-50,-40,54389.1,52166.3,46987.5,57345.1,STRESS-8299473/s2-rank14,rank,yes
S2,1,14,1792993943.502,1.03,-40,-30,53799.7,52166.3,46987.5,57345.1,STRESS-8299473/s2-rank14,rank,yes
S2,1,14,1792993943.502,1.03,-30,-20,50710.4,52166.3,46987.5,57345.1,STRESS-8299473/s2-rank14,rank,yes
S2,1,14,1792993943.502,1.03,-20,-10,51484.1,52166.3,46987.5,57345.1,STRESS-8299473/s2-rank14,rank,yes
S2,1,14,1792993943.502,1.03,-10,0,50119.4,52166.3,46987.5,57345.1,STRESS-8299473/s2-rank14,rank,yes
S2,1,14,1792993943.502,1.03,0,10,44445.3,52166.3,46987.5,57345.1,STRESS-8299473/s2-rank14,rank,yes
S2,1,14,1792993943.502,1.03,10,20,46090.7,52166.3,46987.5,57345.1,STRESS-8299473/s2-rank14,rank,yes
S2,1,14,1792993943.502,1.03,20,30,53750.1,52166.3,46987.5,57345.1,STRESS-8299473/s2-rank14,rank,yes
S2,1,14,1792993943.502,1.03,30,40,53330.6,52166.3,46987.5,57345.1,STRESS-8299473/s2-rank14,rank,yes
S2,1,14,1792993943.502,1.03,40,50,44865.2,52166.3,46987.5,57345.1,STRESS-8299473/s2-rank14,rank,yes
S2,1,14,1792993943.502,1.03,50,60,50043.3,52166.3,46987.5,57345.1,STRESS-8299473/s2-rank14,rank,yes
S2,1,14,1792993943.502,1.03,60,70,51710.3,52166.3,46987.5,57345.1,STRESS-8299473/s2-rank14,rank,yes
S2,1,14,1792993943.502,1.03,70,80,48959.1,52166.3,46987.5,57345.1,STRESS-8299473/s2-rank14,rank,yes
S2,1,14,1792993943.502,1.03,80,90,52654.4,52166.3,46987.5,57345.1,STRESS-8299473/s2-rank14,rank,yes
S2,1,14,1792993943.502,1.03,90,100,51879.7,52166.3,46987.5,57345.1,STRESS-8299473/s2-rank14,rank,yes
S2,1,14,1792993943.502,1.03,100,110,55149.4,52166.3,46987.5,57345.1,STRESS-8299473/s2-rank14,rank,yes
S2,1,14,1792993943.502,1.03,110,120,51363.0,52166.3,46987.5,57345.1,STRESS-8299473/s2-rank14,rank,yes
S2,1,14,1792993943.502,1.03,120,130,55827.1,52166.3,46987.5,57345.1,STRESS-8299473/s2-rank14,rank,yes
S2,1,14,1792993943.502,1.03,130,140,50736.8,52166.3,46987.5,57345.1,STRESS-8299473/s2-rank14,rank,yes
S2,1,14,1792993943.502,1.03,140,150,51719.2,52166.3,46987.5,57345.1,STRESS-8299473/s2-rank14,rank,yes
S2,1,14,1792993943.502,1.03,150,160,54475.7,52166.3,46987.5,57345.1,STRESS-8299473/s2-rank14,rank,yes
S2,1,14,1792993943.502,1.03,160,170,56297.2,52166.3,46987.5,57345.1,STRESS-8299473/s2-rank14,rank,yes
S2,1,14,1792993943.502,1.03,170,180,52257.4,52166.3,46987.5,57345.1,STRESS-8299473/s2-rank14,rank,yes
S2,1,14,1792993943.502,1.03,180,190,49624.5,52166.3,46987.5,57345.1,STRESS-8299473/s2-rank14,rank,yes
S2,1,14,1792993943.502,1.03,190,200,52362.2,52166.3,46987.5,57345.1,STRESS-8299473/s2-rank14,rank,yes
S2,1,14,1792993943.502,1.03,200,210,55805.3,52166.3,46987.5,57345.1,STRESS-8299473/s2-rank14,rank,yes
S2,1,14,1792993943.502,1.03,210,220,59054.9,52166.3,46987.5,57345.1,STRESS-8299473/s2-rank14,rank,yes
S2,1,14,1792993943.502,1.03,220,230,50691.3,52166.3,46987.5,57345.1,STRESS-8299473/s2-rank14,rank,yes
S2,1,14,1792993943.502,1.03,230,240,49604.0,52166.3,46987.5,57345.1,STRESS-8299473/s2-rank14,rank,yes
S2,1,14,1792993943.502,1.03,240,250,51509.6,52166.3,46987.5,57345.1,STRESS-8299473/s2-rank14,rank,yes
S2,1,14,1792993943.502,1.03,250,260,51757.7,52166.3,46987.5,57345.1,STRESS-8299473/s2-rank14,rank,yes
S2,1,14,1792993943.502,1.03,260,270,50934.5,52166.3,46987.5,57345.1,STRESS-8299473/s2-rank14,rank,yes
S2,1,14,1792993943.502,1.03,270,280,53292.0,52166.3,46987.5,57345.1,STRESS-8299473/s2-rank14,rank,yes
S2,1,14,1792993943.502,1.03,280,290,54740.5,52166.3,46987.5,57345.1,STRESS-8299473/s2-rank14,rank,yes
S2,1,14,1792993943.502,1.03,290,300,53953.2,52166.3,46987.5,57345.1,STRESS-8299473/s2-rank14,rank,yes
S3,1,29,1793021770.436,1.02,-120,-110,49483.2,51360.8,46262.0,56459.7,STRESS-8314070/s3-rank29-rank14,rank,yes
S3,1,29,1793021770.436,1.02,-110,-100,48877.0,51360.8,46262.0,56459.7,STRESS-8314070/s3-rank29-rank14,rank,yes
S3,1,29,1793021770.436,1.02,-100,-90,55021.6,51360.8,46262.0,56459.7,STRESS-8314070/s3-rank29-rank14,rank,yes
S3,1,29,1793021770.436,1.02,-90,-80,51180.5,51360.8,46262.0,56459.7,STRESS-8314070/s3-rank29-rank14,rank,yes
S3,1,29,1793021770.436,1.02,-80,-70,53397.9,51360.8,46262.0,56459.7,STRESS-8314070/s3-rank29-rank14,rank,yes
S3,1,29,1793021770.436,1.02,-70,-60,51985.4,51360.8,46262.0,56459.7,STRESS-8314070/s3-rank29-rank14,rank,yes
S3,1,29,1793021770.436,1.02,-60,-50,51069.9,51360.8,46262.0,56459.7,STRESS-8314070/s3-rank29-rank14,rank,yes
S3,1,29,1793021770.436,1.02,-50,-40,47933.5,51360.8,46262.0,56459.7,STRESS-8314070/s3-rank29-rank14,rank,yes
S3,1,29,1793021770.436,1.02,-40,-30,47152.2,51360.8,46262.0,56459.7,STRESS-8314070/s3-rank29-rank14,rank,yes
S3,1,29,1793021770.436,1.02,-30,-20,47132.6,51360.8,46262.0,56459.7,STRESS-8314070/s3-rank29-rank14,rank,yes
S3,1,29,1793021770.436,1.02,-20,-10,49223.5,51360.8,46262.0,56459.7,STRESS-8314070/s3-rank29-rank14,rank,yes
S3,1,29,1793021770.436,1.02,-10,0,54530.5,51360.8,46262.0,56459.7,STRESS-8314070/s3-rank29-rank14,rank,yes
S3,1,29,1793021770.436,1.02,0,10,43248.5,51360.8,46262.0,56459.7,STRESS-8314070/s3-rank29-rank14,rank,yes
S3,1,29,1793021770.436,1.02,10,20,49179.3,51360.8,46262.0,56459.7,STRESS-8314070/s3-rank29-rank14,rank,yes
S3,1,29,1793021770.436,1.02,20,30,56524.3,51360.8,46262.0,56459.7,STRESS-8314070/s3-rank29-rank14,rank,yes
S3,1,29,1793021770.436,1.02,30,40,50980.3,51360.8,46262.0,56459.7,STRESS-8314070/s3-rank29-rank14,rank,yes
S3,1,29,1793021770.436,1.02,40,50,51936.0,51360.8,46262.0,56459.7,STRESS-8314070/s3-rank29-rank14,rank,yes
S3,1,29,1793021770.436,1.02,50,60,51806.5,51360.8,46262.0,56459.7,STRESS-8314070/s3-rank29-rank14,rank,yes
S3,1,29,1793021770.436,1.02,60,70,48330.5,51360.8,46262.0,56459.7,STRESS-8314070/s3-rank29-rank14,rank,yes
S3,1,29,1793021770.436,1.02,70,80,54996.8,51360.8,46262.0,56459.7,STRESS-8314070/s3-rank29-rank14,rank,yes
S3,1,29,1793021770.436,1.02,80,90,58404.2,51360.8,46262.0,56459.7,STRESS-8314070/s3-rank29-rank14,rank,yes
S3,1,29,1793021770.436,1.02,90,100,49464.9,51360.8,46262.0,56459.7,STRESS-8314070/s3-rank29-rank14,rank,yes
S3,1,29,1793021770.436,1.02,100,110,51614.4,51360.8,46262.0,56459.7,STRESS-8314070/s3-rank29-rank14,rank,yes
S3,1,29,1793021770.436,1.02,110,120,52712.1,51360.8,46262.0,56459.7,STRESS-8314070/s3-rank29-rank14,rank,yes
S3,1,29,1793021770.436,1.02,120,130,55958.7,51360.8,46262.0,56459.7,STRESS-8314070/s3-rank29-rank14,rank,yes
S3,1,29,1793021770.436,1.02,130,140,52725.9,51360.8,46262.0,56459.7,STRESS-8314070/s3-rank29-rank14,rank,yes
S3,1,29,1793021770.436,1.02,140,150,48105.8,51360.8,46262.0,56459.7,STRESS-8314070/s3-rank29-rank14,rank,yes
S3,1,29,1793021770.436,1.02,150,160,50845.1,51360.8,46262.0,56459.7,STRESS-8314070/s3-rank29-rank14,rank,yes
S3,1,29,1793021770.436,1.02,160,170,52079.7,51360.8,46262.0,56459.7,STRESS-8314070/s3-rank29-rank14,rank,yes
S3,1,29,1793021770.436,1.02,170,180,55332.1,51360.8,46262.0,56459.7,STRESS-8314070/s3-rank29-rank14,rank,yes
S3,1,29,1793021770.436,1.02,180,190,53019.7,51360.8,46262.0,56459.7,STRESS-8314070/s3-rank29-rank14,rank,yes
S3,1,29,1793021770.436,1.02,190,200,51225.4,51360.8,46262.0,56459.7,STRESS-8314070/s3-rank29-rank14,rank,yes
S3,1,29,1793021770.436,1.02,200,210,53735.4,51360.8,46262.0,56459.7,STRESS-8314070/s3-rank29-rank14,rank,yes
S3,1,29,1793021770.436,1.02,210,220,52360.0,51360.8,46262.0,56459.7,STRESS-8314070/s3-rank29-rank14,rank,yes
S3,1,29,1793021770.436,1.02,220,230,50231.5,51360.8,46262.0,56459.7,STRESS-8314070/s3-rank29-rank14,rank,yes
S3,1,29,1793021770.436,1.02,230,240,49668.2,51360.8,46262.0,56459.7,STRESS-8314070/s3-rank29-rank14,rank,yes
S3,1,29,1793021770.436,1.02,240,250,50535.3,51360.8,46262.0,56459.7,STRESS-8314070/s3-rank29-rank14,rank,yes
S3,1,29,1793021770.436,1.02,250,260,52556.1,51360.8,46262.0,56459.7,STRESS-8314070/s3-rank29-rank14,rank,yes
S3,1,29,1793021770.436,1.02,260,270,49487.6,51360.8,46262.0,56459.7,STRESS-8314070/s3-rank29-rank14,rank,yes
S3,1,29,1793021770.436,1.02,270,280,50716.1,51360.8,46262.0,56459.7,STRESS-8314070/s3-rank29-rank14,rank,yes
S3,1,29,1793021770.436,1.02,280,290,45826.2,51360.8,46262.0,56459.7,STRESS-8314070/s3-rank29-rank14,rank,yes
S3,1,29,1793021770.436,1.02,290,300,46914.9,51360.8,46262.0,56459.7,STRESS-8314070/s3-rank29-rank14,rank,yes
S3,2,14,1793022671.651,0.99,-120,-110,50963.2,51466.0,46356.7,56575.3,STRESS-8314070/s3-rank29-rank14,rank,yes
S3,2,14,1793022671.651,0.99,-110,-100,48561.3,51466.0,46356.7,56575.3,STRESS-8314070/s3-rank29-rank14,rank,yes
S3,2,14,1793022671.651,0.99,-100,-90,46014.6,51466.0,46356.7,56575.3,STRESS-8314070/s3-rank29-rank14,rank,yes
S3,2,14,1793022671.651,0.99,-90,-80,54539.4,51466.0,46356.7,56575.3,STRESS-8314070/s3-rank29-rank14,rank,yes
S3,2,14,1793022671.651,0.99,-80,-70,52892.7,51466.0,46356.7,56575.3,STRESS-8314070/s3-rank29-rank14,rank,yes
S3,2,14,1793022671.651,0.99,-70,-60,53462.2,51466.0,46356.7,56575.3,STRESS-8314070/s3-rank29-rank14,rank,yes
S3,2,14,1793022671.651,0.99,-60,-50,46855.9,51466.0,46356.7,56575.3,STRESS-8314070/s3-rank29-rank14,rank,yes
S3,2,14,1793022671.651,0.99,-50,-40,50300.8,51466.0,46356.7,56575.3,STRESS-8314070/s3-rank29-rank14,rank,yes
S3,2,14,1793022671.651,0.99,-40,-30,48705.2,51466.0,46356.7,56575.3,STRESS-8314070/s3-rank29-rank14,rank,yes
S3,2,14,1793022671.651,0.99,-30,-20,50148.9,51466.0,46356.7,56575.3,STRESS-8314070/s3-rank29-rank14,rank,yes
S3,2,14,1793022671.651,0.99,-20,-10,52219.6,51466.0,46356.7,56575.3,STRESS-8314070/s3-rank29-rank14,rank,yes
S3,2,14,1793022671.651,0.99,-10,0,55055.0,51466.0,46356.7,56575.3,STRESS-8314070/s3-rank29-rank14,rank,yes
S3,2,14,1793022671.651,0.99,0,10,48965.9,51466.0,46356.7,56575.3,STRESS-8314070/s3-rank29-rank14,rank,yes
S3,2,14,1793022671.651,0.99,10,20,50733.9,51466.0,46356.7,56575.3,STRESS-8314070/s3-rank29-rank14,rank,yes
S3,2,14,1793022671.651,0.99,20,30,49867.9,51466.0,46356.7,56575.3,STRESS-8314070/s3-rank29-rank14,rank,yes
S3,2,14,1793022671.651,0.99,30,40,45684.4,51466.0,46356.7,56575.3,STRESS-8314070/s3-rank29-rank14,rank,yes
S3,2,14,1793022671.651,0.99,40,50,52670.5,51466.0,46356.7,56575.3,STRESS-8314070/s3-rank29-rank14,rank,yes
S3,2,14,1793022671.651,0.99,50,60,54503.6,51466.0,46356.7,56575.3,STRESS-8314070/s3-rank29-rank14,rank,yes
S3,2,14,1793022671.651,0.99,60,70,48430.5,51466.0,46356.7,56575.3,STRESS-8314070/s3-rank29-rank14,rank,yes
S3,2,14,1793022671.651,0.99,70,80,54024.6,51466.0,46356.7,56575.3,STRESS-8314070/s3-rank29-rank14,rank,yes
S3,2,14,1793022671.651,0.99,80,90,53160.3,51466.0,46356.7,56575.3,STRESS-8314070/s3-rank29-rank14,rank,yes
S3,2,14,1793022671.651,0.99,90,100,51241.5,51466.0,46356.7,56575.3,STRESS-8314070/s3-rank29-rank14,rank,yes
S3,2,14,1793022671.651,0.99,100,110,55420.4,51466.0,46356.7,56575.3,STRESS-8314070/s3-rank29-rank14,rank,yes
S3,2,14,1793022671.651,0.99,110,120,52898.7,51466.0,46356.7,56575.3,STRESS-8314070/s3-rank29-rank14,rank,yes
S3,2,14,1793022671.651,0.99,120,130,50458.0,51466.0,46356.7,56575.3,STRESS-8314070/s3-rank29-rank14,rank,yes
S3,2,14,1793022671.651,0.99,130,140,51145.2,51466.0,46356.7,56575.3,STRESS-8314070/s3-rank29-rank14,rank,yes
S3,2,14,1793022671.651,0.99,140,150,49673.2,51466.0,46356.7,56575.3,STRESS-8314070/s3-rank29-rank14,rank,yes
S3,2,14,1793022671.651,0.99,150,160,49095.3,51466.0,46356.7,56575.3,STRESS-8314070/s3-rank29-rank14,rank,yes
S3,2,14,1793022671.651,0.99,160,170,52996.5,51466.0,46356.7,56575.3,STRESS-8314070/s3-rank29-rank14,rank,yes
S3,2,14,1793022671.651,0.99,170,180,45770.7,51466.0,46356.7,56575.3,STRESS-8314070/s3-rank29-rank14,rank,yes
S3,2,14,1793022671.651,0.99,180,190,53793.1,51466.0,46356.7,56575.3,STRESS-8314070/s3-rank29-rank14,rank,yes
S3,2,14,1793022671.651,0.99,190,200,52392.3,51466.0,46356.7,56575.3,STRESS-8314070/s3-rank29-rank14,rank,yes
S3,2,14,1793022671.651,0.99,200,210,54035.3,51466.0,46356.7,56575.3,STRESS-8314070/s3-rank29-rank14,rank,yes
S3,2,14,1793022671.651,0.99,210,220,53209.2,51466.0,46356.7,56575.3,STRESS-8314070/s3-rank29-rank14,rank,yes
S3,2,14,1793022671.651,0.99,220,230,50512.1,51466.0,46356.7,56575.3,STRESS-8314070/s3-rank29-rank14,rank,yes
S3,2,14,1793022671.651,0.99,230,240,51253.1,51466.0,46356.7,56575.3,STRESS-8314070/s3-rank29-rank14,rank,yes
S3,2,14,1793022671.651,0.99,240,250,52703.5,51466.0,46356.7,56575.3,STRESS-8314070/s3-rank29-rank14,rank,yes
S3,2,14,1793022671.651,0.99,250,260,47561.5,51466.0,46356.7,56575.3,STRESS-8314070/s3-rank29-rank14,rank,yes
S3,2,14,1793022671.651,0.99,260,270,54498.7,51466.0,46356.7,56575.3,STRESS-8314070/s3-rank29-rank14,rank,yes
S3,2,14,1793022671.651,0.99,270,280,51633.9,51466.0,46356.7,56575.3,STRESS-8314070/s3-rank29-rank14,rank,yes
S3,2,14,1793022671.651,0.99,280,290,50672.5,51466.0,46356.7,56575.3,STRESS-8314070/s3-rank29-rank14,rank,yes
S3,2,14,1793022671.651,0.99,290,300,55480.8,51466.0,46356.7,56575.3,STRESS-8314070/s3-rank29-rank14,rank,yes
S2,1,13,1792994959.578,1.03,-120,-110,55385.5,53281.7,47992.2,58571.2,STRESS-G1-8299473/s2-rank13,rank,no
S2,1,13,1792994959.578,1.03,-110,-100,52769.3,53281.7,47992.2,58571.2,STRESS-G1-8299473/s2-rank13,rank,no
S2,1,13,1792994959.578,1.03,-100,-90,55161.6,53281.7,47992.2,58571.2,STRESS-G1-8299473/s2-rank13,rank,no
S2,1,13,1792994959.578,1.03,-90,-80,55505.7,53281.7,47992.2,58571.2,STRESS-G1-8299473/s2-rank13,rank,no
S2,1,13,1792994959.578,1.03,-80,-70,55088.1,53281.7,47992.2,58571.2,STRESS-G1-8299473/s2-rank13,rank,no
S2,1,13,1792994959.578,1.03,-70,-60,54204.8,53281.7,47992.2,58571.2,STRESS-G1-8299473/s2-rank13,rank,no
S2,1,13,1792994959.578,1.03,-60,-50,47842.6,53281.7,47992.2,58571.2,STRESS-G1-8299473/s2-rank13,rank,no
S2,1,13,1792994959.578,1.03,-50,-40,49575.3,53281.7,47992.2,58571.2,STRESS-G1-8299473/s2-rank13,rank,no
S2,1,13,1792994959.578,1.03,-40,-30,55087.6,53281.7,47992.2,58571.2,STRESS-G1-8299473/s2-rank13,rank,no
S2,1,13,1792994959.578,1.03,-30,-20,51795.9,53281.7,47992.2,58571.2,STRESS-G1-8299473/s2-rank13,rank,no
S2,1,13,1792994959.578,1.03,-20,-10,55708.0,53281.7,47992.2,58571.2,STRESS-G1-8299473/s2-rank13,rank,no
S2,1,13,1792994959.578,1.03,-10,0,53624.0,53281.7,47992.2,58571.2,STRESS-G1-8299473/s2-rank13,rank,no
S2,1,13,1792994959.578,1.03,0,10,45066.5,53281.7,47992.2,58571.2,STRESS-G1-8299473/s2-rank13,rank,no
S2,1,13,1792994959.578,1.03,10,20,44930.7,53281.7,47992.2,58571.2,STRESS-G1-8299473/s2-rank13,rank,no
S2,1,13,1792994959.578,1.03,20,30,56610.7,53281.7,47992.2,58571.2,STRESS-G1-8299473/s2-rank13,rank,no
S2,1,13,1792994959.578,1.03,30,40,53121.8,53281.7,47992.2,58571.2,STRESS-G1-8299473/s2-rank13,rank,no
S2,1,13,1792994959.578,1.03,40,50,49709.8,53281.7,47992.2,58571.2,STRESS-G1-8299473/s2-rank13,rank,no
S2,1,13,1792994959.578,1.03,50,60,53385.6,53281.7,47992.2,58571.2,STRESS-G1-8299473/s2-rank13,rank,no
S2,1,13,1792994959.578,1.03,60,70,52473.6,53281.7,47992.2,58571.2,STRESS-G1-8299473/s2-rank13,rank,no
S2,1,13,1792994959.578,1.03,70,80,56581.4,53281.7,47992.2,58571.2,STRESS-G1-8299473/s2-rank13,rank,no
S2,1,13,1792994959.578,1.03,80,90,54505.5,53281.7,47992.2,58571.2,STRESS-G1-8299473/s2-rank13,rank,no
S2,1,13,1792994959.578,1.03,90,100,55145.4,53281.7,47992.2,58571.2,STRESS-G1-8299473/s2-rank13,rank,no
S2,1,13,1792994959.578,1.03,100,110,58421.7,53281.7,47992.2,58571.2,STRESS-G1-8299473/s2-rank13,rank,no
S2,1,13,1792994959.578,1.03,110,120,54311.8,53281.7,47992.2,58571.2,STRESS-G1-8299473/s2-rank13,rank,no
S2,1,13,1792994959.578,1.03,120,130,54446.0,53281.7,47992.2,58571.2,STRESS-G1-8299473/s2-rank13,rank,no
S2,1,13,1792994959.578,1.03,130,140,49305.0,53281.7,47992.2,58571.2,STRESS-G1-8299473/s2-rank13,rank,no
S2,1,13,1792994959.578,1.03,140,150,55358.6,53281.7,47992.2,58571.2,STRESS-G1-8299473/s2-rank13,rank,no
S2,1,13,1792994959.578,1.03,150,160,53312.3,53281.7,47992.2,58571.2,STRESS-G1-8299473/s2-rank13,rank,no
S2,1,13,1792994959.578,1.03,160,170,52706.5,53281.7,47992.2,58571.2,STRESS-G1-8299473/s2-rank13,rank,no
S2,1,13,1792994959.578,1.03,170,180,53114.9,53281.7,47992.2,58571.2,STRESS-G1-8299473/s2-rank13,rank,no
S2,1,13,1792994959.578,1.03,180,190,50626.6,53281.7,47992.2,58571.2,STRESS-G1-8299473/s2-rank13,rank,no
S2,1,13,1792994959.578,1.03,190,200,59104.8,53281.7,47992.2,58571.2,STRESS-G1-8299473/s2-rank13,rank,no
S2,1,13,1792994959.578,1.03,200,210,52215.5,53281.7,47992.2,58571.2,STRESS-G1-8299473/s2-rank13,rank,no
S2,1,13,1792994959.578,1.03,210,220,54869.1,53281.7,47992.2,58571.2,STRESS-G1-8299473/s2-rank13,rank,no
S2,1,13,1792994959.578,1.03,220,230,48748.7,53281.7,47992.2,58571.2,STRESS-G1-8299473/s2-rank13,rank,no
S2,1,13,1792994959.578,1.03,230,240,54642.9,53281.7,47992.2,58571.2,STRESS-G1-8299473/s2-rank13,rank,no
S2,1,13,1792994959.578,1.03,240,250,54828.5,53281.7,47992.2,58571.2,STRESS-G1-8299473/s2-rank13,rank,no
S2,1,13,1792994959.578,1.03,250,260,53845.1,53281.7,47992.2,58571.2,STRESS-G1-8299473/s2-rank13,rank,no
S2,1,13,1792994959.578,1.03,260,270,53026.3,53281.7,47992.2,58571.2,STRESS-G1-8299473/s2-rank13,rank,no
S2,1,13,1792994959.578,1.03,270,280,52200.3,53281.7,47992.2,58571.2,STRESS-G1-8299473/s2-rank13,rank,no
S2,1,13,1792994959.578,1.03,280,290,51954.6,53281.7,47992.2,58571.2,STRESS-G1-8299473/s2-rank13,rank,no
S2,1,13,1792994959.578,1.03,290,300,54184.1,53281.7,47992.2,58571.2,STRESS-G1-8299473/s2-rank13,rank,no
S3,1,12,1793022350.069,0.87,-120,-110,49540.7,51833.9,46688.1,56979.6,STRESS-G1-8314070/s3-rank12-rank31,rank,no
S3,1,12,1793022350.069,0.87,-110,-100,54518.1,51833.9,46688.1,56979.6,STRESS-G1-8314070/s3-rank12-rank31,rank,no
S3,1,12,1793022350.069,0.87,-100,-90,53587.1,51833.9,46688.1,56979.6,STRESS-G1-8314070/s3-rank12-rank31,rank,no
S3,1,12,1793022350.069,0.87,-90,-80,53417.6,51833.9,46688.1,56979.6,STRESS-G1-8314070/s3-rank12-rank31,rank,no
S3,1,12,1793022350.069,0.87,-80,-70,57659.5,51833.9,46688.1,56979.6,STRESS-G1-8314070/s3-rank12-rank31,rank,no
S3,1,12,1793022350.069,0.87,-70,-60,53855.5,51833.9,46688.1,56979.6,STRESS-G1-8314070/s3-rank12-rank31,rank,no
S3,1,12,1793022350.069,0.87,-60,-50,49494.6,51833.9,46688.1,56979.6,STRESS-G1-8314070/s3-rank12-rank31,rank,no
S3,1,12,1793022350.069,0.87,-50,-40,47889.7,51833.9,46688.1,56979.6,STRESS-G1-8314070/s3-rank12-rank31,rank,no
S3,1,12,1793022350.069,0.87,-40,-30,53881.3,51833.9,46688.1,56979.6,STRESS-G1-8314070/s3-rank12-rank31,rank,no
S3,1,12,1793022350.069,0.87,-30,-20,51915.7,51833.9,46688.1,56979.6,STRESS-G1-8314070/s3-rank12-rank31,rank,no
S3,1,12,1793022350.069,0.87,-20,-10,53502.9,51833.9,46688.1,56979.6,STRESS-G1-8314070/s3-rank12-rank31,rank,no
S3,1,12,1793022350.069,0.87,-10,0,54819.2,51833.9,46688.1,56979.6,STRESS-G1-8314070/s3-rank12-rank31,rank,no
S3,1,12,1793022350.069,0.87,0,10,50464.3,51833.9,46688.1,56979.6,STRESS-G1-8314070/s3-rank12-rank31,rank,no
S3,1,12,1793022350.069,0.87,10,20,49944.3,51833.9,46688.1,56979.6,STRESS-G1-8314070/s3-rank12-rank31,rank,no
S3,1,12,1793022350.069,0.87,20,30,48167.7,51833.9,46688.1,56979.6,STRESS-G1-8314070/s3-rank12-rank31,rank,no
S3,1,12,1793022350.069,0.87,30,40,50598.4,51833.9,46688.1,56979.6,STRESS-G1-8314070/s3-rank12-rank31,rank,no
S3,1,12,1793022350.069,0.87,40,50,48986.4,51833.9,46688.1,56979.6,STRESS-G1-8314070/s3-rank12-rank31,rank,no
S3,1,12,1793022350.069,0.87,50,60,57237.5,51833.9,46688.1,56979.6,STRESS-G1-8314070/s3-rank12-rank31,rank,no
S3,1,12,1793022350.069,0.87,60,70,52544.3,51833.9,46688.1,56979.6,STRESS-G1-8314070/s3-rank12-rank31,rank,no
S3,1,12,1793022350.069,0.87,70,80,53025.7,51833.9,46688.1,56979.6,STRESS-G1-8314070/s3-rank12-rank31,rank,no
S3,1,12,1793022350.069,0.87,80,90,52445.6,51833.9,46688.1,56979.6,STRESS-G1-8314070/s3-rank12-rank31,rank,no
S3,1,12,1793022350.069,0.87,90,100,51182.6,51833.9,46688.1,56979.6,STRESS-G1-8314070/s3-rank12-rank31,rank,no
S3,1,12,1793022350.069,0.87,100,110,47649.6,51833.9,46688.1,56979.6,STRESS-G1-8314070/s3-rank12-rank31,rank,no
S3,1,12,1793022350.069,0.87,110,120,51396.3,51833.9,46688.1,56979.6,STRESS-G1-8314070/s3-rank12-rank31,rank,no
S3,1,12,1793022350.069,0.87,120,130,50784.5,51833.9,46688.1,56979.6,STRESS-G1-8314070/s3-rank12-rank31,rank,no
S3,1,12,1793022350.069,0.87,130,140,49100.4,51833.9,46688.1,56979.6,STRESS-G1-8314070/s3-rank12-rank31,rank,no
S3,1,12,1793022350.069,0.87,140,150,53772.9,51833.9,46688.1,56979.6,STRESS-G1-8314070/s3-rank12-rank31,rank,no
S3,1,12,1793022350.069,0.87,150,160,48518.9,51833.9,46688.1,56979.6,STRESS-G1-8314070/s3-rank12-rank31,rank,no
S3,1,12,1793022350.069,0.87,160,170,53018.5,51833.9,46688.1,56979.6,STRESS-G1-8314070/s3-rank12-rank31,rank,no
S3,1,12,1793022350.069,0.87,170,180,51973.5,51833.9,46688.1,56979.6,STRESS-G1-8314070/s3-rank12-rank31,rank,no
S3,1,12,1793022350.069,0.87,180,190,50139.0,51833.9,46688.1,56979.6,STRESS-G1-8314070/s3-rank12-rank31,rank,no
S3,1,12,1793022350.069,0.87,190,200,56417.1,51833.9,46688.1,56979.6,STRESS-G1-8314070/s3-rank12-rank31,rank,no
S3,1,12,1793022350.069,0.87,200,210,49532.9,51833.9,46688.1,56979.6,STRESS-G1-8314070/s3-rank12-rank31,rank,no
S3,1,12,1793022350.069,0.87,210,220,51160.5,51833.9,46688.1,56979.6,STRESS-G1-8314070/s3-rank12-rank31,rank,no
S3,1,12,1793022350.069,0.87,220,230,48866.2,51833.9,46688.1,56979.6,STRESS-G1-8314070/s3-rank12-rank31,rank,no
S3,1,12,1793022350.069,0.87,230,240,51732.0,51833.9,46688.1,56979.6,STRESS-G1-8314070/s3-rank12-rank31,rank,no
S3,1,12,1793022350.069,0.87,240,250,46794.5,51833.9,46688.1,56979.6,STRESS-G1-8314070/s3-rank12-rank31,rank,no
S3,1,12,1793022350.069,0.87,250,260,51191.7,51833.9,46688.1,56979.6,STRESS-G1-8314070/s3-rank12-rank31,rank,no
S3,1,12,1793022350.069,0.87,260,270,54460.2,51833.9,46688.1,56979.6,STRESS-G1-8314070/s3-rank12-rank31,rank,no
S3,1,12,1793022350.069,0.87,270,280,51675.8,51833.9,46688.1,56979.6,STRESS-G1-8314070/s3-rank12-rank31,rank,no
S3,1,12,1793022350.069,0.87,280,290,50525.3,51833.9,46688.1,56979.6,STRESS-G1-8314070/s3-rank12-rank31,rank,no
S3,1,12,1793022350.069,0.87,290,300,51890.8,51833.9,46688.1,56979.6,STRESS-G1-8314070/s3-rank12-rank31,rank,no
S3,2,31,1793023251.076,1.03,-120,-110,54077.7,52897.8,47646.4,58149.2,STRESS-G1-8314070/s3-rank12-rank31,rank,no
S3,2,31,1793023251.076,1.03,-110,-100,57357.9,52897.8,47646.4,58149.2,STRESS-G1-8314070/s3-rank12-rank31,rank,no
S3,2,31,1793023251.076,1.03,-100,-90,52816.8,52897.8,47646.4,58149.2,STRESS-G1-8314070/s3-rank12-rank31,rank,no
S3,2,31,1793023251.076,1.03,-90,-80,53019.0,52897.8,47646.4,58149.2,STRESS-G1-8314070/s3-rank12-rank31,rank,no
S3,2,31,1793023251.076,1.03,-80,-70,48471.9,52897.8,47646.4,58149.2,STRESS-G1-8314070/s3-rank12-rank31,rank,no
S3,2,31,1793023251.076,1.03,-70,-60,51339.3,52897.8,47646.4,58149.2,STRESS-G1-8314070/s3-rank12-rank31,rank,no
S3,2,31,1793023251.076,1.03,-60,-50,55486.0,52897.8,47646.4,58149.2,STRESS-G1-8314070/s3-rank12-rank31,rank,no
S3,2,31,1793023251.076,1.03,-50,-40,50449.9,52897.8,47646.4,58149.2,STRESS-G1-8314070/s3-rank12-rank31,rank,no
S3,2,31,1793023251.076,1.03,-40,-30,55633.7,52897.8,47646.4,58149.2,STRESS-G1-8314070/s3-rank12-rank31,rank,no
S3,2,31,1793023251.076,1.03,-30,-20,54624.8,52897.8,47646.4,58149.2,STRESS-G1-8314070/s3-rank12-rank31,rank,no
S3,2,31,1793023251.076,1.03,-20,-10,56574.2,52897.8,47646.4,58149.2,STRESS-G1-8314070/s3-rank12-rank31,rank,no
S3,2,31,1793023251.076,1.03,-10,0,50551.4,52897.8,47646.4,58149.2,STRESS-G1-8314070/s3-rank12-rank31,rank,no
S3,2,31,1793023251.076,1.03,0,10,49368.4,52897.8,47646.4,58149.2,STRESS-G1-8314070/s3-rank12-rank31,rank,no
S3,2,31,1793023251.076,1.03,10,20,54980.2,52897.8,47646.4,58149.2,STRESS-G1-8314070/s3-rank12-rank31,rank,no
S3,2,31,1793023251.076,1.03,20,30,57349.2,52897.8,47646.4,58149.2,STRESS-G1-8314070/s3-rank12-rank31,rank,no
S3,2,31,1793023251.076,1.03,30,40,50214.8,52897.8,47646.4,58149.2,STRESS-G1-8314070/s3-rank12-rank31,rank,no
S3,2,31,1793023251.076,1.03,40,50,56539.6,52897.8,47646.4,58149.2,STRESS-G1-8314070/s3-rank12-rank31,rank,no
S3,2,31,1793023251.076,1.03,50,60,50135.7,52897.8,47646.4,58149.2,STRESS-G1-8314070/s3-rank12-rank31,rank,no
S3,2,31,1793023251.076,1.03,60,70,51748.4,52897.8,47646.4,58149.2,STRESS-G1-8314070/s3-rank12-rank31,rank,no
S3,2,31,1793023251.076,1.03,70,80,53606.2,52897.8,47646.4,58149.2,STRESS-G1-8314070/s3-rank12-rank31,rank,no
S3,2,31,1793023251.076,1.03,80,90,50927.1,52897.8,47646.4,58149.2,STRESS-G1-8314070/s3-rank12-rank31,rank,no
S3,2,31,1793023251.076,1.03,90,100,53791.8,52897.8,47646.4,58149.2,STRESS-G1-8314070/s3-rank12-rank31,rank,no
S3,2,31,1793023251.076,1.03,100,110,53365.1,52897.8,47646.4,58149.2,STRESS-G1-8314070/s3-rank12-rank31,rank,no
S3,2,31,1793023251.076,1.03,110,120,49614.8,52897.8,47646.4,58149.2,STRESS-G1-8314070/s3-rank12-rank31,rank,no
S3,2,31,1793023251.076,1.03,120,130,49944.4,52897.8,47646.4,58149.2,STRESS-G1-8314070/s3-rank12-rank31,rank,no
S3,2,31,1793023251.076,1.03,130,140,49124.0,52897.8,47646.4,58149.2,STRESS-G1-8314070/s3-rank12-rank31,rank,no
S3,2,31,1793023251.076,1.03,140,150,50670.5,52897.8,47646.4,58149.2,STRESS-G1-8314070/s3-rank12-rank31,rank,no
S3,2,31,1793023251.076,1.03,150,160,54793.4,52897.8,47646.4,58149.2,STRESS-G1-8314070/s3-rank12-rank31,rank,no
S3,2,31,1793023251.076,1.03,160,170,50539.1,52897.8,47646.4,58149.2,STRESS-G1-8314070/s3-rank12-rank31,rank,no
S3,2,31,1793023251.076,1.03,170,180,54391.8,52897.8,47646.4,58149.2,STRESS-G1-8314070/s3-rank12-rank31,rank,no
S3,2,31,1793023251.076,1.03,180,190,52884.8,52897.8,47646.4,58149.2,STRESS-G1-8314070/s3-rank12-rank31,rank,no
S3,2,31,1793023251.076,1.03,190,200,53269.6,52897.8,47646.4,58149.2,STRESS-G1-8314070/s3-rank12-rank31,rank,no
S3,2,31,1793023251.076,1.03,200,210,55555.6,52897.8,47646.4,58149.2,STRESS-G1-8314070/s3-rank12-rank31,rank,no
S3,2,31,1793023251.076,1.03,210,220,52623.1,52897.8,47646.4,58149.2,STRESS-G1-8314070/s3-rank12-rank31,rank,no
S3,2,31,1793023251.076,1.03,220,230,52804.8,52897.8,47646.4,58149.2,STRESS-G1-8314070/s3-rank12-rank31,rank,no
S3,2,31,1793023251.076,1.03,230,240,57541.0,52897.8,47646.4,58149.2,STRESS-G1-8314070/s3-rank12-rank31,rank,no
S3,2,31,1793023251.076,1.03,240,250,54390.8,52897.8,47646.4,58149.2,STRESS-G1-8314070/s3-rank12-rank31,rank,no
S3,2,31,1793023251.076,1.03,250,260,50803.4,52897.8,47646.4,58149.2,STRESS-G1-8314070/s3-rank12-rank31,rank,no
S3,2,31,1793023251.076,1.03,260,270,49808.7,52897.8,47646.4,58149.2,STRESS-G1-8314070/s3-rank12-rank31,rank,no
S3,2,31,1793023251.076,1.03,270,280,51534.8,52897.8,47646.4,58149.2,STRESS-G1-8314070/s3-rank12-rank31,rank,no
S3,2,31,1793023251.076,1.03,280,290,56501.4,52897.8,47646.4,58149.2,STRESS-G1-8314070/s3-rank12-rank31,rank,no
S3,2,31,1793023251.076,1.03,290,300,51563.1,52897.8,47646.4,58149.2,STRESS-G1-8314070/s3-rank12-rank31,rank,no
S2,1,25,1793029815.681,1.05,-120,-110,55872.7,52783.9,47543.8,58024.0,STRESS-8318006/s2-rank25,rank,no
S2,1,25,1793029815.681,1.05,-110,-100,51793.7,52783.9,47543.8,58024.0,STRESS-8318006/s2-rank25,rank,no
S2,1,25,1793029815.681,1.05,-100,-90,54402.6,52783.9,47543.8,58024.0,STRESS-8318006/s2-rank25,rank,no
S2,1,25,1793029815.681,1.05,-90,-80,53780.6,52783.9,47543.8,58024.0,STRESS-8318006/s2-rank25,rank,no
S2,1,25,1793029815.681,1.05,-80,-70,52731.7,52783.9,47543.8,58024.0,STRESS-8318006/s2-rank25,rank,no
S2,1,25,1793029815.681,1.05,-70,-60,50216.1,52783.9,47543.8,58024.0,STRESS-8318006/s2-rank25,rank,no
S2,1,25,1793029815.681,1.05,-60,-50,53932.2,52783.9,47543.8,58024.0,STRESS-8318006/s2-rank25,rank,no
S2,1,25,1793029815.681,1.05,-50,-40,55481.1,52783.9,47543.8,58024.0,STRESS-8318006/s2-rank25,rank,no
S2,1,25,1793029815.681,1.05,-40,-30,57420.5,52783.9,47543.8,58024.0,STRESS-8318006/s2-rank25,rank,no
S2,1,25,1793029815.681,1.05,-30,-20,50286.0,52783.9,47543.8,58024.0,STRESS-8318006/s2-rank25,rank,no
S2,1,25,1793029815.681,1.05,-20,-10,54255.0,52783.9,47543.8,58024.0,STRESS-8318006/s2-rank25,rank,no
S2,1,25,1793029815.681,1.05,-10,0,50332.9,52783.9,47543.8,58024.0,STRESS-8318006/s2-rank25,rank,no
S2,1,25,1793029815.681,1.05,0,10,46474.2,52783.9,47543.8,58024.0,STRESS-8318006/s2-rank25,rank,no
S2,1,25,1793029815.681,1.05,10,20,52636.4,52783.9,47543.8,58024.0,STRESS-8318006/s2-rank25,rank,no
S2,1,25,1793029815.681,1.05,20,30,51390.9,52783.9,47543.8,58024.0,STRESS-8318006/s2-rank25,rank,no
S2,1,25,1793029815.681,1.05,30,40,54403.6,52783.9,47543.8,58024.0,STRESS-8318006/s2-rank25,rank,no
S2,1,25,1793029815.681,1.05,40,50,51946.2,52783.9,47543.8,58024.0,STRESS-8318006/s2-rank25,rank,no
S2,1,25,1793029815.681,1.05,50,60,49670.4,52783.9,47543.8,58024.0,STRESS-8318006/s2-rank25,rank,no
S2,1,25,1793029815.681,1.05,60,70,48059.9,52783.9,47543.8,58024.0,STRESS-8318006/s2-rank25,rank,no
S2,1,25,1793029815.681,1.05,70,80,54179.9,52783.9,47543.8,58024.0,STRESS-8318006/s2-rank25,rank,no
S2,1,25,1793029815.681,1.05,80,90,48511.2,52783.9,47543.8,58024.0,STRESS-8318006/s2-rank25,rank,no
S2,1,25,1793029815.681,1.05,90,100,50905.0,52783.9,47543.8,58024.0,STRESS-8318006/s2-rank25,rank,no
S2,1,25,1793029815.681,1.05,100,110,51339.0,52783.9,47543.8,58024.0,STRESS-8318006/s2-rank25,rank,no
S2,1,25,1793029815.681,1.05,110,120,50589.2,52783.9,47543.8,58024.0,STRESS-8318006/s2-rank25,rank,no
S2,1,25,1793029815.681,1.05,120,130,53795.3,52783.9,47543.8,58024.0,STRESS-8318006/s2-rank25,rank,no
S2,1,25,1793029815.681,1.05,130,140,52377.9,52783.9,47543.8,58024.0,STRESS-8318006/s2-rank25,rank,no
S2,1,25,1793029815.681,1.05,140,150,57792.7,52783.9,47543.8,58024.0,STRESS-8318006/s2-rank25,rank,no
S2,1,25,1793029815.681,1.05,150,160,53319.1,52783.9,47543.8,58024.0,STRESS-8318006/s2-rank25,rank,no
S2,1,25,1793029815.681,1.05,160,170,55636.0,52783.9,47543.8,58024.0,STRESS-8318006/s2-rank25,rank,no
S2,1,25,1793029815.681,1.05,170,180,53086.0,52783.9,47543.8,58024.0,STRESS-8318006/s2-rank25,rank,no
S2,1,25,1793029815.681,1.05,180,190,55466.9,52783.9,47543.8,58024.0,STRESS-8318006/s2-rank25,rank,no
S2,1,25,1793029815.681,1.05,190,200,49679.4,52783.9,47543.8,58024.0,STRESS-8318006/s2-rank25,rank,no
S2,1,25,1793029815.681,1.05,200,210,49808.3,52783.9,47543.8,58024.0,STRESS-8318006/s2-rank25,rank,no
S2,1,25,1793029815.681,1.05,210,220,54640.0,52783.9,47543.8,58024.0,STRESS-8318006/s2-rank25,rank,no
S2,1,25,1793029815.681,1.05,220,230,50610.9,52783.9,47543.8,58024.0,STRESS-8318006/s2-rank25,rank,no
S2,1,25,1793029815.681,1.05,230,240,54223.3,52783.9,47543.8,58024.0,STRESS-8318006/s2-rank25,rank,no
S2,1,25,1793029815.681,1.05,240,250,56577.4,52783.9,47543.8,58024.0,STRESS-8318006/s2-rank25,rank,no
S2,1,25,1793029815.681,1.05,250,260,59165.1,52783.9,47543.8,58024.0,STRESS-8318006/s2-rank25,rank,no
S2,1,25,1793029815.681,1.05,260,270,49777.0,52783.9,47543.8,58024.0,STRESS-8318006/s2-rank25,rank,no
S2,1,25,1793029815.681,1.05,270,280,53601.9,52783.9,47543.8,58024.0,STRESS-8318006/s2-rank25,rank,no
S2,1,25,1793029815.681,1.05,280,290,50691.3,52783.9,47543.8,58024.0,STRESS-8318006/s2-rank25,rank,no
S2,1,25,1793029815.681,1.05,290,300,53407.2,52783.9,47543.8,58024.0,STRESS-8318006/s2-rank25,rank,no
S2,1,8+9+10+11,1793028451.545,1.96,-120,-110,49399.3,52431.4,47226.2,57636.5,STRESS-G1-8318006/s2-node2,node,no
S2,1,8+9+10+11,1793028451.545,1.96,-110,-100,53081.0,52431.4,47226.2,57636.5,STRESS-G1-8318006/s2-node2,node,no
S2,1,8+9+10+11,1793028451.545,1.96,-100,-90,53356.1,52431.4,47226.2,57636.5,STRESS-G1-8318006/s2-node2,node,no
S2,1,8+9+10+11,1793028451.545,1.96,-90,-80,54169.7,52431.4,47226.2,57636.5,STRESS-G1-8318006/s2-node2,node,no
S2,1,8+9+10+11,1793028451.545,1.96,-80,-70,53616.9,52431.4,47226.2,57636.5,STRESS-G1-8318006/s2-node2,node,no
S2,1,8+9+10+11,1793028451.545,1.96,-70,-60,51881.4,52431.4,47226.2,57636.5,STRESS-G1-8318006/s2-node2,node,no
S2,1,8+9+10+11,1793028451.545,1.96,-60,-50,51706.8,52431.4,47226.2,57636.5,STRESS-G1-8318006/s2-node2,node,no
S2,1,8+9+10+11,1793028451.545,1.96,-50,-40,48793.4,52431.4,47226.2,57636.5,STRESS-G1-8318006/s2-node2,node,no
S2,1,8+9+10+11,1793028451.545,1.96,-40,-30,51565.1,52431.4,47226.2,57636.5,STRESS-G1-8318006/s2-node2,node,no
S2,1,8+9+10+11,1793028451.545,1.96,-30,-20,48036.9,52431.4,47226.2,57636.5,STRESS-G1-8318006/s2-node2,node,no
S2,1,8+9+10+11,1793028451.545,1.96,-20,-10,59072.1,52431.4,47226.2,57636.5,STRESS-G1-8318006/s2-node2,node,no
S2,1,8+9+10+11,1793028451.545,1.96,-10,0,52089.4,52431.4,47226.2,57636.5,STRESS-G1-8318006/s2-node2,node,no
S2,1,8+9+10+11,1793028451.545,1.96,0,10,36473.6,52431.4,47226.2,57636.5,STRESS-G1-8318006/s2-node2,node,no
S2,1,8+9+10+11,1793028451.545,1.96,10,20,55621.0,52431.4,47226.2,57636.5,STRESS-G1-8318006/s2-node2,node,no
S2,1,8+9+10+11,1793028451.545,1.96,20,30,52658.3,52431.4,47226.2,57636.5,STRESS-G1-8318006/s2-node2,node,no
S2,1,8+9+10+11,1793028451.545,1.96,30,40,53927.1,52431.4,47226.2,57636.5,STRESS-G1-8318006/s2-node2,node,no
S2,1,8+9+10+11,1793028451.545,1.96,40,50,53114.2,52431.4,47226.2,57636.5,STRESS-G1-8318006/s2-node2,node,no
S2,1,8+9+10+11,1793028451.545,1.96,50,60,52994.9,52431.4,47226.2,57636.5,STRESS-G1-8318006/s2-node2,node,no
S2,1,8+9+10+11,1793028451.545,1.96,60,70,56135.7,52431.4,47226.2,57636.5,STRESS-G1-8318006/s2-node2,node,no
S2,1,8+9+10+11,1793028451.545,1.96,70,80,55165.1,52431.4,47226.2,57636.5,STRESS-G1-8318006/s2-node2,node,no
S2,1,8+9+10+11,1793028451.545,1.96,80,90,51672.2,52431.4,47226.2,57636.5,STRESS-G1-8318006/s2-node2,node,no
S2,1,8+9+10+11,1793028451.545,1.96,90,100,49851.4,52431.4,47226.2,57636.5,STRESS-G1-8318006/s2-node2,node,no
S2,1,8+9+10+11,1793028451.545,1.96,100,110,56934.2,52431.4,47226.2,57636.5,STRESS-G1-8318006/s2-node2,node,no
S2,1,8+9+10+11,1793028451.545,1.96,110,120,58554.0,52431.4,47226.2,57636.5,STRESS-G1-8318006/s2-node2,node,no
S2,1,8+9+10+11,1793028451.545,1.96,120,130,54009.5,52431.4,47226.2,57636.5,STRESS-G1-8318006/s2-node2,node,no
S2,1,8+9+10+11,1793028451.545,1.96,130,140,48987.1,52431.4,47226.2,57636.5,STRESS-G1-8318006/s2-node2,node,no
S2,1,8+9+10+11,1793028451.545,1.96,140,150,52127.8,52431.4,47226.2,57636.5,STRESS-G1-8318006/s2-node2,node,no
S2,1,8+9+10+11,1793028451.545,1.96,150,160,52963.7,52431.4,47226.2,57636.5,STRESS-G1-8318006/s2-node2,node,no
S2,1,8+9+10+11,1793028451.545,1.96,160,170,53292.2,52431.4,47226.2,57636.5,STRESS-G1-8318006/s2-node2,node,no
S2,1,8+9+10+11,1793028451.545,1.96,170,180,56550.1,52431.4,47226.2,57636.5,STRESS-G1-8318006/s2-node2,node,no
S2,1,8+9+10+11,1793028451.545,1.96,180,190,52847.8,52431.4,47226.2,57636.5,STRESS-G1-8318006/s2-node2,node,no
S2,1,8+9+10+11,1793028451.545,1.96,190,200,48404.5,52431.4,47226.2,57636.5,STRESS-G1-8318006/s2-node2,node,no
S2,1,8+9+10+11,1793028451.545,1.96,200,210,51806.9,52431.4,47226.2,57636.5,STRESS-G1-8318006/s2-node2,node,no
S2,1,8+9+10+11,1793028451.545,1.96,210,220,47099.3,52431.4,47226.2,57636.5,STRESS-G1-8318006/s2-node2,node,no
S2,1,8+9+10+11,1793028451.545,1.96,220,230,52313.9,52431.4,47226.2,57636.5,STRESS-G1-8318006/s2-node2,node,no
S2,1,8+9+10+11,1793028451.545,1.96,230,240,52191.5,52431.4,47226.2,57636.5,STRESS-G1-8318006/s2-node2,node,no
S2,1,8+9+10+11,1793028451.545,1.96,240,250,56158.6,52431.4,47226.2,57636.5,STRESS-G1-8318006/s2-node2,node,no
S2,1,8+9+10+11,1793028451.545,1.96,250,260,49236.7,52431.4,47226.2,57636.5,STRESS-G1-8318006/s2-node2,node,no
S2,1,8+9+10+11,1793028451.545,1.96,260,270,49896.6,52431.4,47226.2,57636.5,STRESS-G1-8318006/s2-node2,node,no
S2,1,8+9+10+11,1793028451.545,1.96,270,280,49199.5,52431.4,47226.2,57636.5,STRESS-G1-8318006/s2-node2,node,no
S2,1,8+9+10+11,1793028451.545,1.96,280,290,50163.4,52431.4,47226.2,57636.5,STRESS-G1-8318006/s2-node2,node,no
S2,1,8+9+10+11,1793028451.545,1.96,290,300,52840.1,52431.4,47226.2,57636.5,STRESS-G1-8318006/s2-node2,node,no
```

- **Doc text.**
  - The first post-kill bins are 44,445, 43,249 and 48,966 tok/s.
  - The bins from +20 s on are back near the offered 51-52k tok/s.
  - Bin-to-bin noise is about +-10%.

### Figure 10. Latency under realistic traffic (S1)

- **What it shows.** TTFT and TPOT p50/p99 for FT off vs FT on at 51.0 req/s. The bars are the mean of the 6
  same-node pairs.
- **Source.** S1 8310100, S1B 8359063 and S1C 8363138 (`review/diag/stress_s1_<job>.txt`, `STRESS_PAIR` lines; CLAIMS 217, 234, 240).
  - pair1 = group G0, FT off first.
  - pair2 = group G8, FT on first.
  - pair3 / pair4 = S1B's G0 / G8, and pair5 / pair6 = S1C's G0 / G8, the same orders.
- **Raw.** Per pair:

```csv
# block: fig10_pairs
metric,pair,ft_off_ms,ft_on_ms,change_pct
ttft_p50_ms,pair1,816.9,816.6,-0.04
ttft_p50_ms,pair2,827.1,831.9,+0.58
ttft_p50_ms,pair3,819.6,826.9,+0.89
ttft_p50_ms,pair4,815.3,829.8,+1.78
ttft_p50_ms,pair5,827.4,825.3,-0.25
ttft_p50_ms,pair6,816.9,835.1,+2.23
ttft_p99_ms,pair1,2018.0,2052.5,+1.71
ttft_p99_ms,pair2,2039.5,2049.9,+0.51
ttft_p99_ms,pair3,2000.1,2046.7,+2.33
ttft_p99_ms,pair4,1998.4,2016.4,+0.90
ttft_p99_ms,pair5,2108.0,2016.4,-4.35
ttft_p99_ms,pair6,2004.9,2337.8,+16.60
tpot_p50_ms,pair1,11.7,11.8,+0.85
tpot_p50_ms,pair2,11.8,11.9,+0.85
tpot_p50_ms,pair3,11.7,11.8,+0.85
tpot_p50_ms,pair4,11.8,11.9,+0.85
tpot_p50_ms,pair5,11.8,11.9,+0.85
tpot_p50_ms,pair6,11.8,11.8,+0.00
tpot_p99_ms,pair1,13.4,13.5,+0.75
tpot_p99_ms,pair2,13.7,13.6,-0.73
tpot_p99_ms,pair3,13.5,13.4,-0.74
tpot_p99_ms,pair4,13.4,13.6,+1.49
tpot_p99_ms,pair5,19.8,13.5,-31.82
tpot_p99_ms,pair6,13.4,14.5,+8.21
```

- **Aggregation.** Mean of the 6 pairs; change % = mean on / mean off - 1.

```csv
# block: fig10_agg
metric,pair,ft_off_ms,ft_on_ms,change_pct
ttft_p50_ms,MEAN n=6,820.53,827.60,+0.86
ttft_p99_ms,MEAN n=6,2028.15,2086.62,+2.88
tpot_p50_ms,MEAN n=6,11.77,11.85,+0.71
tpot_p99_ms,MEAN n=6,14.53,13.68,-5.85
```

### Figure A1. Decode step cost

- **What it shows.** FT's change in rank 0's trimmed mean decode step, with its 95% CI.
  - 32/GPU: pooled n=8.
  - 128/GPU: pooled n=8.
  - Extra µs per step.
- **Source.** `repl_cost_pooled.txt` (32/GPU) and `v5f_readout_full.txt` (128/GPU): CLAIMS 206 and 194.
- **Raw.** `healthy_pairs` (`trim_off_ms`, `trim_on_ms`).
- **Aggregation.**
  - Per pair, the trimmed %.
  - Mean and t-CI over pairs.
  - µs = mean of (trim_on - trim_off) x 1000.
  - The "original n=4" rows are the first 4 pairs at 32/GPU (CLAIMS 194).

```csv
# block: figA1_agg
load,mode,n,trim_mean_pct,ci_lo,ci_hi,extra_us_per_step,trim_off_ms,trim_on_ms
32,noeplb,8,+0.87,+0.72,+1.02,120.0,13.831,13.951
32,offline,8,+0.50,+0.34,+0.66,70.3,14.123,14.194
32,online,8,+0.35,-0.04,+0.75,51.1,14.634,14.685
128,noeplb,8,+0.56,+0.38,+0.74,150.0,26.782,26.932
128,offline,8,+0.39,+0.24,+0.54,102.7,26.428,26.531
128,online,8,+0.40,+0.28,+0.53,99.4,24.674,24.773
```

### Figure A2. Throughput per GPU vs per-user speed

- **What it shows.** Output tok/s per GPU against per-user tok/s, FT off and on, per mode, at 32 and 128/GPU.
- **Raw.** `healthy_pairs`. Per user = 1000 / median TPOT.
- **Aggregation.** Means over pairs.

```csv
# block: figA2_agg
mode,load,arm,pair,tok_s_per_user,tok_s_per_gpu
noeplb,32,off,MEAN n=8,117.031,3061.42
noeplb,128,off,MEAN n=8,50.498,4868.27
noeplb,32,on,MEAN n=8,119.171,3068.83
noeplb,128,on,MEAN n=8,50.102,4851.18
offline,32,off,MEAN n=8,122.693,3403.58
offline,128,off,MEAN n=8,58.453,6152.89
offline,32,on,MEAN n=8,122.676,3379.61
offline,128,on,MEAN n=8,58.621,6156.89
online,32,off,MEAN n=8,112.724,2993.08
online,128,off,MEAN n=8,56.673,5465.37
online,32,on,MEAN n=8,114.382,2994.40
online,128,on,MEAN n=8,56.662,5444.14
```

### Figure A3. All-to-all cost per FT check (diagnostic build)

- **What it shows.** µs saved per dispatch + combine round trip when one FT check is removed, at EP32 with 64
  tokens/rank, with and without EPLB statistics.
- **Source.** ABLATE 8293196 (`review/diag/ablate-8293196/verdict.txt`, CLAIMS 210).
- **Build.** `build-ablate-d47a3bcf88`: experiment switches only; with all switches off its SASS is identical to
  v5-final, and every variant's output is bit-identical to FT off.
- **Method.** Kernel-only benchmark (`R/tools/a2aep_bench.py`):
  - one CUDA graph of 200 round trips, 20 cycles, variants rotated;
  - per-rank block minimum, median over blocks;
  - D = ft - off, cut = D0 - D;
  - 3 launches per item.
- **Raw.** Per-launch cuts:

```csv
# block: figA3_launches
item_removed,case,D_us,D0_us,cut_mean_us,cut_launch1_us,cut_launch2_us,cut_launch3_us
CONTROL_D0,with_stats,,+3.00,,,,
0,with_stats,+2.99,+3.00,+0.00,+0.03,+0.00,+0.00
S,with_stats,+0.86,+3.00,+2.13,+2.13,+2.16,+2.13
S1,with_stats,+1.37,+2.99,+1.62,+1.62,+1.62,+1.64
F,with_stats,+2.52,+3.00,+0.48,+0.48,+0.48,+0.51
F8,with_stats,+2.82,+3.01,+0.18,+0.18,+0.18,+0.19
L,with_stats,+2.56,+3.01,+0.45,+0.45,+0.46,+0.43
A,with_stats,+2.91,+3.00,+0.10,+0.11,+0.07,+0.10
SO,with_stats,+2.99,+3.01,+0.03,+0.03,+0.04,+0.02
ALL,with_stats,+0.09,+2.97,+2.86,+2.86,+2.86,+2.88
CONTROL_D0,without_stats,,+0.88,,,,
0,without_stats,+0.87,+0.87,+0.00,-0.01,+0.00,+0.01
S,without_stats,+0.87,+0.89,+0.02,+0.02,+0.03,-0.00
S1,without_stats,+0.87,+0.89,+0.01,-0.01,+0.01,+0.01
F,without_stats,+0.43,+0.89,+0.45,+0.45,+0.44,+0.45
F8,without_stats,+0.73,+0.87,+0.14,+0.15,+0.13,+0.14
L,without_stats,+0.39,+0.88,+0.49,+0.49,+0.49,+0.50
A,without_stats,+0.82,+0.87,+0.05,+0.06,+0.05,+0.05
SO,without_stats,+0.86,+0.89,+0.03,+0.03,+0.05,+0.02
ALL,without_stats,+0.02,+0.88,+0.85,+0.85,+0.84,+0.85
```

- **Aggregation.**
  - Bar = the mean cut over 3 launches; A+SO = cut(A) + cut(SO).
  - The dotted lines are the control FT cost D0: +3.00 µs with statistics, +0.88 µs without.
  - Single-item cuts overlap: they sum to 3.20 / 1.04 µs against ALL 2.86 / 0.85 µs.

```csv
# block: figA3_agg
check_removed,case,us_saved,parts_us,n,measured_ft_total_us
S,With EPLB stats,2.13,2.13,3 launches each,3.00
S,Without EPLB stats,0.02,0.02,3 launches each,0.88
F,With EPLB stats,0.48,0.48,3 launches each,3.00
F,Without EPLB stats,0.45,0.45,3 launches each,0.88
L,With EPLB stats,0.45,0.45,3 launches each,3.00
L,Without EPLB stats,0.49,0.49,3 launches each,0.88
A+SO,With EPLB stats,0.13,0.10+0.03,3 launches each,3.00
A+SO,Without EPLB stats,0.08,0.05+0.03,3 launches each,0.88
ALL,With EPLB stats,2.86,2.86,3 launches each,3.00
ALL,Without EPLB stats,0.85,0.85,3 launches each,0.88
```

### Figure A4. Restart without FT vs FT recovery

- **What it shows.** Where a restart after a rank kill spends its time, as the mean per mode, next to FT's rank-kill
  recovery (Fig 7 means).
- **Raw.** `restart_runs` (Fig 7). Segments per run:
  - model load = `rank0_model_load_s`;
  - autotune = `warmup_s - weights_s`;
  - graph capture = `graphs_s - warmup_s`;
  - other = `exit_s + (weights_s - model load) + (first_token_s - graphs_s)`.

  These sum to `down_s`.
- **Aggregation.** Means over runs: n=6 no EPLB, n=7 offline, n=7 online.

```csv
# block: figA4_agg
what,mode,source,model_load_s,autotune_s,graph_capture_s,other_s,total_s
restart,noeplb,MEAN n=6,74.6,77.3,151.1,36.4,339.4
FT on recovery,noeplb,MEAN n=4 (fig7_recovery.csv),,,,,1.503
restart,offline,MEAN n=7,93.1,76.1,165.0,37.1,371.4
FT on recovery,offline,MEAN n=3 (fig7_recovery.csv),,,,,1.280
restart,online,MEAN n=7,337.2,73.9,177.3,37.9,626.2
FT on recovery,online,MEAN n=8 (fig7_recovery.csv),,,,,0.880
```

### Figure A5. Remap phases (instrumented overlay, diagnostic)

- **What it shows.** The all-to-all remap time of the median survivor, split by phase, for 5 rank kills.
- **Source.** REMAP-TIMER 8308514 (CLAIMS 214; `review/diag/remap_timer_8308514.md`). All 5 kills PASS.
- **Raw.** Every survivor's `Remap phases ms` line (ms):

```csv
# block: figA5_survivors
case,rank,release,barrier1,make_room,create,export,gather,import_map,outcome,remap_rest,barrier2,total
KILL-FINAL-OFF-8308514/rank5-1,0,15.8,59.0,20.6,3.4,0.0,16.4,346.3,477.1,2.6,0.8,942.0
KILL-FINAL-OFF-8308514/rank5-1,1,13.6,57.2,23.9,1.5,0.0,14.8,346.4,476.9,2.7,0.9,938.1
KILL-FINAL-OFF-8308514/rank5-1,2,25.0,41.2,16.9,3.6,0.0,19.7,346.6,476.8,2.7,0.8,933.3
KILL-FINAL-OFF-8308514/rank5-1,3,10.4,56.5,16.9,3.8,0.0,19.7,346.5,476.9,2.7,0.8,934.2
KILL-FINAL-OFF-8308514/rank5-1,4,8.1,53.7,17.9,3.7,0.0,18.7,822.4,1.0,3.1,0.4,929.0
KILL-FINAL-OFF-8308514/rank5-1,6,13.2,55.5,13.9,3.8,0.0,22.6,822.2,1.1,2.6,1.0,935.9
KILL-FINAL-OFF-8308514/rank5-1,7,12.4,57.8,15.1,2.5,0.0,22.7,822.3,1.1,2.6,0.9,937.4
KILL-FINAL-OFF-8308514/rank5-1,8,23.3,27.6,23.5,3.7,0.0,13.3,347.8,475.6,2.8,0.6,918.2
KILL-FINAL-OFF-8308514/rank5-1,9,11.5,41.9,18.7,5.2,0.0,16.6,347.9,475.5,2.9,0.5,920.5
KILL-FINAL-OFF-8308514/rank5-1,10,14.3,43.1,18.6,5.2,0.0,16.6,347.7,475.6,2.7,0.8,924.7
KILL-FINAL-OFF-8308514/rank5-1,11,11.7,40.4,27.3,1.8,0.0,11.4,347.7,475.6,2.6,0.8,919.3
KILL-FINAL-OFF-8308514/rank5-1,12,12.6,34.2,27.8,4.8,0.0,8.0,347.9,475.4,2.6,0.8,914.2
KILL-FINAL-OFF-8308514/rank5-1,13,11.2,36.9,25.4,6.9,0.0,8.1,347.9,475.5,2.6,0.7,915.3
KILL-FINAL-OFF-8308514/rank5-1,14,22.9,22.9,30.4,2.1,0.0,7.8,347.8,475.6,2.8,0.7,913.1
KILL-FINAL-OFF-8308514/rank5-1,15,11.3,34.8,32.5,1.3,0.0,6.4,347.9,475.5,2.7,0.7,913.3
KILL-FINAL-OFF-8308514/rank5-1,16,23.0,17.8,28.3,7.2,0.0,5.2,347.8,475.6,2.8,0.5,908.2
KILL-FINAL-OFF-8308514/rank5-1,17,11.8,32.0,28.0,7.1,0.0,5.3,347.8,475.5,2.9,0.5,911.0
KILL-FINAL-OFF-8308514/rank5-1,18,12.8,33.1,28.0,7.2,0.0,5.3,347.7,475.6,2.6,0.9,913.2
KILL-FINAL-OFF-8308514/rank5-1,19,11.1,31.1,35.5,1.4,0.0,3.6,347.6,475.7,2.6,0.8,909.4
KILL-FINAL-OFF-8308514/rank5-1,20,12.5,42.6,33.6,4.0,0.0,3.0,347.6,475.7,2.6,0.8,922.4
KILL-FINAL-OFF-8308514/rank5-1,21,12.7,44.7,30.4,7.0,0.0,3.0,347.9,475.4,2.6,0.8,924.6
KILL-FINAL-OFF-8308514/rank5-1,22,23.6,28.7,27.3,6.0,0.0,7.0,347.8,475.5,3.0,0.6,919.5
KILL-FINAL-OFF-8308514/rank5-1,23,15.1,45.3,30.3,7.0,0.0,3.0,348.0,475.3,2.9,0.6,927.6
KILL-FINAL-OFF-8308514/rank5-1,24,12.5,56.3,28.3,6.9,0.0,5.3,346.3,477.1,2.7,0.7,936.1
KILL-FINAL-OFF-8308514/rank5-1,25,12.5,54.4,36.6,1.8,0.0,2.1,346.2,477.1,2.8,0.6,934.1
KILL-FINAL-OFF-8308514/rank5-1,26,12.1,55.1,35.2,1.4,0.0,3.8,346.5,476.9,3.1,0.3,934.4
KILL-FINAL-OFF-8308514/rank5-1,27,24.7,39.8,25.0,10.0,0.0,5.4,346.4,476.9,3.1,0.3,931.7
KILL-FINAL-OFF-8308514/rank5-1,28,27.8,0.2,36.7,2.9,0.0,1.0,343.5,479.9,2.6,0.7,895.3
KILL-FINAL-OFF-8308514/rank5-1,29,19.2,17.8,30.9,5.4,0.0,4.2,345.7,477.6,2.6,0.7,904.2
KILL-FINAL-OFF-8308514/rank5-1,30,21.2,16.8,30.9,5.6,0.0,4.0,345.8,477.5,2.8,0.6,905.3
KILL-FINAL-OFF-8308514/rank5-1,31,22.9,18.3,30.9,5.6,0.0,4.0,345.5,477.7,2.8,0.6,908.4
KILL-FINAL-OFF-8308514/rank5-2,0,10.7,92.5,38.2,6.2,0.0,13.1,820.7,13.4,2.6,0.7,998.1
KILL-FINAL-OFF-8308514/rank5-2,1,12.2,90.9,34.8,9.4,0.0,13.3,820.8,13.3,2.7,0.7,998.1
KILL-FINAL-OFF-8308514/rank5-2,2,25.0,78.0,26.5,8.1,0.0,22.9,820.8,13.2,2.9,0.5,998.0
KILL-FINAL-OFF-8308514/rank5-2,3,11.3,91.7,37.7,6.7,0.0,13.1,820.9,13.2,2.9,0.5,998.0
KILL-FINAL-OFF-8308514/rank5-2,4,24.5,71.2,36.2,6.2,0.0,15.2,820.7,13.4,2.8,0.6,990.7
KILL-FINAL-OFF-8308514/rank5-2,5,11.4,86.0,39.7,3.2,0.0,14.6,820.7,13.4,2.8,0.6,992.4
KILL-FINAL-OFF-8308514/rank5-2,6,12.2,85.3,28.3,5.2,0.0,24.0,820.9,13.2,2.7,0.8,992.5
KILL-FINAL-OFF-8308514/rank5-2,7,13.4,88.3,33.4,6.3,0.0,17.7,820.8,13.3,2.6,0.8,996.6
KILL-FINAL-OFF-8308514/rank5-2,8,13.4,85.7,30.8,10.2,0.0,16.5,820.7,13.4,2.7,0.6,994.0
KILL-FINAL-OFF-8308514/rank5-2,9,13.8,83.7,34.0,9.4,0.0,13.9,820.8,13.3,2.7,0.7,992.4
KILL-FINAL-OFF-8308514/rank5-2,10,11.3,85.6,34.0,7.1,0.0,16.3,820.6,13.5,2.6,0.8,991.8
KILL-FINAL-OFF-8308514/rank5-2,11,23.0,70.7,41.1,3.0,0.0,13.4,820.8,13.3,2.5,0.8,988.7
KILL-FINAL-OFF-8308514/rank5-2,12,16.4,79.2,36.8,8.4,0.0,12.3,819.5,14.6,2.6,0.7,990.4
KILL-FINAL-OFF-8308514/rank5-2,13,16.9,80.9,32.7,6.9,0.0,17.8,819.6,14.5,2.7,0.7,992.7
KILL-FINAL-OFF-8308514/rank5-2,14,15.5,79.9,40.1,5.1,0.0,12.2,819.6,14.4,2.8,0.6,990.3
KILL-FINAL-OFF-8308514/rank5-2,15,23.9,64.0,25.7,11.1,0.0,20.8,819.7,14.4,2.7,0.5,982.8
KILL-FINAL-OFF-8308514/rank5-2,16,52.0,33.4,47.4,6.4,0.0,3.9,821.2,12.8,2.7,0.5,980.4
KILL-FINAL-OFF-8308514/rank5-2,17,70.3,15.0,53.8,2.8,0.0,0.9,821.3,12.7,2.8,0.6,980.2
KILL-FINAL-OFF-8308514/rank5-2,18,51.7,33.5,44.2,9.4,0.0,4.0,821.1,13.0,2.6,0.7,980.2
KILL-FINAL-OFF-8308514/rank5-2,19,84.4,0.3,50.3,3.5,0.0,3.8,821.1,12.9,2.6,0.7,979.7
KILL-FINAL-OFF-8308514/rank5-2,20,9.6,93.3,22.7,7.2,0.0,27.7,832.9,1.2,2.7,0.6,997.8
KILL-FINAL-OFF-8308514/rank5-2,21,11.0,91.8,22.6,7.3,0.0,27.7,833.0,1.1,2.7,0.6,997.8
KILL-FINAL-OFF-8308514/rank5-2,23,10.4,90.9,30.0,3.5,0.0,24.0,833.1,1.0,3.0,0.4,996.3
KILL-FINAL-OFF-8308514/rank5-2,24,13.0,74.5,30.2,10.3,0.0,17.1,816.5,17.6,2.9,0.4,982.4
KILL-FINAL-OFF-8308514/rank5-2,25,14.3,73.1,40.0,3.7,0.0,13.9,816.6,17.5,2.9,0.4,982.4
KILL-FINAL-OFF-8308514/rank5-2,26,12.7,73.1,36.4,7.1,0.0,14.0,816.7,17.4,2.8,0.6,980.8
KILL-FINAL-OFF-8308514/rank5-2,27,22.9,60.7,30.0,6.5,0.0,21.1,816.7,17.3,2.7,0.6,978.6
KILL-FINAL-OFF-8308514/rank5-2,28,12.3,90.5,25.4,14.6,0.0,17.6,820.2,13.8,3.0,0.3,997.7
KILL-FINAL-OFF-8308514/rank5-2,29,10.9,91.9,40.8,2.9,0.0,13.9,821.7,12.4,2.9,0.4,997.7
KILL-FINAL-OFF-8308514/rank5-2,30,11.0,91.8,34.1,6.6,0.0,16.8,821.3,12.8,2.7,0.6,997.6
KILL-FINAL-OFF-8308514/rank5-2,31,25.0,77.8,33.9,6.8,0.0,16.8,821.3,12.7,2.7,0.6,997.7
KILL-FINAL-OFF-B-8308514/rank5-1,0,24.0,101.8,17.9,12.7,0.0,22.1,812.7,13.9,2.6,0.7,1008.4
KILL-FINAL-OFF-B-8308514/rank5-1,1,11.1,114.6,17.9,12.6,0.0,22.2,812.7,13.9,2.6,0.8,1008.3
KILL-FINAL-OFF-B-8308514/rank5-1,2,11.4,114.2,18.0,12.6,0.0,22.0,812.9,13.7,2.7,0.8,1008.3
KILL-FINAL-OFF-B-8308514/rank5-1,3,12.7,112.9,17.9,12.6,0.0,22.1,812.8,13.8,2.7,0.7,1008.2
KILL-FINAL-OFF-B-8308514/rank5-1,4,8.7,103.0,19.8,3.8,0.0,29.1,812.6,13.9,3.0,0.3,994.4
KILL-FINAL-OFF-B-8308514/rank5-1,6,11.2,104.1,15.5,4.1,0.0,33.1,812.8,13.8,2.6,0.8,998.0
KILL-FINAL-OFF-B-8308514/rank5-1,7,10.2,105.5,13.1,6.2,0.0,33.3,812.6,14.1,2.6,0.7,998.4
KILL-FINAL-OFF-B-8308514/rank5-1,8,24.5,101.0,18.2,7.1,0.0,27.3,813.0,13.6,2.8,0.5,1008.1
KILL-FINAL-OFF-B-8308514/rank5-1,9,11.3,114.2,28.1,3.1,0.0,21.3,813.0,13.7,2.8,0.5,1008.1
KILL-FINAL-OFF-B-8308514/rank5-1,10,10.5,115.1,25.2,3.0,0.0,24.4,812.9,13.7,2.7,0.7,1008.2
KILL-FINAL-OFF-B-8308514/rank5-1,11,11.6,114.0,18.2,7.0,0.0,27.4,812.7,13.8,2.6,0.7,1008.1
KILL-FINAL-OFF-B-8308514/rank5-1,12,11.9,113.6,19.0,5.7,0.0,28.0,814.4,12.3,2.6,0.8,1008.2
KILL-FINAL-OFF-B-8308514/rank5-1,13,10.6,114.8,19.0,5.7,0.0,28.1,814.5,12.1,2.6,0.7,1008.1
KILL-FINAL-OFF-B-8308514/rank5-1,14,23.8,101.6,24.6,6.2,0.0,21.7,814.3,12.3,2.9,0.5,1008.0
KILL-FINAL-OFF-B-8308514/rank5-1,15,10.8,114.6,24.8,6.0,0.0,21.8,814.5,12.1,2.9,0.5,1008.0
KILL-FINAL-OFF-B-8308514/rank5-1,16,23.8,101.5,22.3,6.8,0.0,23.7,814.2,12.4,2.8,0.5,1007.9
KILL-FINAL-OFF-B-8308514/rank5-1,17,11.2,114.1,28.5,3.6,0.0,20.7,814.3,12.3,2.7,0.6,1008.0
KILL-FINAL-OFF-B-8308514/rank5-1,18,10.1,115.2,25.1,6.9,0.0,20.7,814.2,12.4,2.7,0.7,1008.1
KILL-FINAL-OFF-B-8308514/rank5-1,19,11.3,114.1,19.0,6.7,0.0,27.1,814.1,12.5,2.6,0.7,1008.0
KILL-FINAL-OFF-B-8308514/rank5-1,20,12.2,16.9,48.7,3.0,0.0,1.0,825.6,1.0,2.6,0.7,911.6
KILL-FINAL-OFF-B-8308514/rank5-1,21,27.0,0.3,42.2,6.5,0.0,4.1,825.6,1.0,2.7,0.6,909.9
KILL-FINAL-OFF-B-8308514/rank5-1,22,12.9,14.4,41.5,7.0,0.0,4.2,825.9,0.7,2.7,0.7,910.0
KILL-FINAL-OFF-B-8308514/rank5-1,23,15.6,16.6,34.1,8.0,0.0,10.7,825.7,0.8,2.7,0.7,914.9
KILL-FINAL-OFF-B-8308514/rank5-1,24,12.0,113.2,21.6,7.1,0.0,24.0,811.4,15.3,2.7,0.6,1007.9
KILL-FINAL-OFF-B-8308514/rank5-1,25,10.2,115.1,18.7,9.8,0.0,24.2,811.3,15.3,2.7,0.5,1007.9
KILL-FINAL-OFF-B-8308514/rank5-1,26,11.0,114.3,28.7,2.9,0.0,21.0,811.6,15.1,2.7,0.7,1007.9
KILL-FINAL-OFF-B-8308514/rank5-1,27,23.7,101.5,21.6,6.9,0.0,24.1,811.5,15.2,2.7,0.6,1007.8
KILL-FINAL-OFF-B-8308514/rank5-1,28,10.7,114.6,29.5,3.4,0.0,19.8,811.1,15.6,2.7,0.5,1007.9
KILL-FINAL-OFF-B-8308514/rank5-1,29,11.7,113.5,22.5,7.2,0.0,23.0,809.1,17.5,2.8,0.4,1007.8
KILL-FINAL-OFF-B-8308514/rank5-1,30,23.2,102.0,18.6,10.8,0.0,23.2,810.6,16.1,3.0,0.3,1007.8
KILL-FINAL-OFF-B-8308514/rank5-1,31,11.3,113.9,22.5,10.0,0.0,20.3,811.0,15.6,3.0,0.3,1007.8
KILL-FINAL-ON-8308514/rank5-1,0,11.5,19.5,23.4,11.7,0.0,7.5,316.0,3.9,2.7,0.7,397.0
KILL-FINAL-ON-8308514/rank5-1,1,25.4,5.4,37.0,2.7,0.0,2.9,316.0,3.9,2.9,0.7,396.9
KILL-FINAL-ON-8308514/rank5-1,2,10.9,20.0,23.5,9.5,0.0,9.6,315.9,4.1,2.6,0.9,396.9
KILL-FINAL-ON-8308514/rank5-1,3,12.6,18.2,33.0,3.7,0.0,5.9,315.8,4.2,2.6,0.8,396.9
KILL-FINAL-ON-8308514/rank5-1,4,11.2,19.5,31.8,6.6,0.0,4.3,316.3,3.6,2.6,0.8,396.8
KILL-FINAL-ON-8308514/rank5-1,5,12.5,18.3,28.8,9.4,0.0,4.4,316.2,3.7,2.7,0.8,396.8
KILL-FINAL-ON-8308514/rank5-1,6,11.2,19.6,38.3,2.7,0.0,1.4,316.3,3.7,2.9,0.6,396.7
KILL-FINAL-ON-8308514/rank5-1,7,25.2,5.5,31.1,7.2,0.0,4.3,316.4,3.5,2.8,0.6,396.7
KILL-FINAL-ON-8308514/rank5-1,8,25.1,5.5,29.9,11.2,0.0,1.5,315.9,4.0,2.9,0.6,396.6
KILL-FINAL-ON-8308514/rank5-1,9,11.1,19.6,29.9,11.1,0.0,1.6,316.2,3.8,2.9,0.6,396.6
KILL-FINAL-ON-8308514/rank5-1,10,11.9,18.8,29.7,11.4,0.0,1.5,315.9,4.0,2.6,0.8,396.7
KILL-FINAL-ON-8308514/rank5-1,11,10.3,20.4,29.7,11.2,0.0,1.7,316.0,3.9,2.6,0.8,396.7
KILL-FINAL-ON-8308514/rank5-1,12,10.5,20.1,26.5,2.9,0.0,13.2,313.4,6.5,3.1,0.2,396.5
KILL-FINAL-ON-8308514/rank5-1,14,9.5,21.2,19.7,6.5,0.0,16.2,313.4,6.5,2.6,0.8,396.6
KILL-FINAL-ON-8308514/rank5-1,15,11.2,19.5,19.8,6.6,0.0,16.2,313.3,6.6,2.6,0.8,396.6
KILL-FINAL-ON-8308514/rank5-1,16,10.7,19.8,34.9,5.7,0.0,2.2,315.9,4.1,2.8,0.4,396.4
KILL-FINAL-ON-8308514/rank5-1,17,30.2,0.3,27.8,7.0,0.0,7.9,316.0,4.0,2.8,0.5,396.5
KILL-FINAL-ON-8308514/rank5-1,18,10.4,20.1,34.8,5.7,0.0,2.2,315.8,4.1,2.6,0.7,396.5
KILL-FINAL-ON-8308514/rank5-1,19,11.4,19.1,28.0,6.8,0.0,7.9,315.8,4.1,2.5,0.8,396.5
KILL-FINAL-ON-8308514/rank5-1,20,11.4,19.0,28.0,7.2,0.0,7.6,316.1,3.8,2.9,0.4,396.4
KILL-FINAL-ON-8308514/rank5-1,21,24.7,5.6,35.3,6.3,0.0,1.1,316.0,3.9,3.0,0.4,396.4
KILL-FINAL-ON-8308514/rank5-1,22,9.9,20.5,35.3,6.2,0.0,1.1,316.3,3.6,2.6,0.8,396.5
KILL-FINAL-ON-8308514/rank5-1,23,11.0,19.4,28.0,7.1,0.0,7.5,316.2,3.7,2.5,0.8,396.4
KILL-FINAL-ON-8308514/rank5-1,24,10.1,20.3,27.7,6.9,0.0,8.1,316.0,4.0,2.5,0.8,396.4
KILL-FINAL-ON-8308514/rank5-1,25,12.2,18.2,35.0,6.0,0.0,1.7,316.1,3.9,2.6,0.7,396.4
KILL-FINAL-ON-8308514/rank5-1,26,10.8,19.5,27.9,6.8,0.0,7.9,316.0,3.9,2.9,0.5,396.3
KILL-FINAL-ON-8308514/rank5-1,27,24.1,6.2,34.9,6.0,0.0,1.7,315.9,4.1,2.8,0.6,396.3
KILL-FINAL-ON-8308514/rank5-1,28,9.3,21.0,26.7,11.2,0.0,4.9,318.9,1.0,2.7,0.7,396.2
KILL-FINAL-ON-8308514/rank5-1,29,11.0,19.3,34.9,3.7,0.0,4.1,318.6,1.3,2.7,0.6,396.2
KILL-FINAL-ON-8308514/rank5-1,30,24.2,6.1,38.5,2.7,0.0,1.5,316.2,3.7,2.9,0.5,396.2
KILL-FINAL-ON-8308514/rank5-1,31,10.3,20.0,30.6,7.3,0.0,4.8,316.0,3.9,2.9,0.5,396.2
KILL-FINAL-ON-8308514/rank5-2,0,24.8,11.1,22.4,11.6,0.0,1.2,332.0,1.0,2.8,0.4,407.3
KILL-FINAL-ON-8308514/rank5-2,1,11.6,24.2,24.8,9.3,0.0,1.0,332.0,1.0,2.9,0.4,407.2
KILL-FINAL-ON-8308514/rank5-2,2,10.7,25.1,24.7,9.4,0.0,1.0,331.9,1.1,2.7,0.6,407.3
KILL-FINAL-ON-8308514/rank5-2,3,12.0,23.8,24.7,9.3,0.0,1.1,332.1,0.9,2.6,0.6,407.3
KILL-FINAL-ON-8308514/rank5-2,4,10.7,17.7,20.0,6.3,0.0,8.9,331.8,1.2,2.6,0.5,399.8
KILL-FINAL-ON-8308514/rank5-2,5,13.1,19.6,26.2,6.5,0.0,2.5,332.0,1.0,2.7,0.6,404.2
KILL-FINAL-ON-8308514/rank5-2,6,27.4,0.3,28.7,4.0,0.0,2.4,331.9,1.1,2.9,0.3,399.1
KILL-FINAL-ON-8308514/rank5-2,7,13.0,20.4,19.8,6.5,0.0,8.9,332.0,1.0,3.0,0.3,404.8
KILL-FINAL-ON-8308514/rank5-2,8,11.5,24.2,22.8,8.8,0.0,3.6,331.9,1.1,2.7,0.4,407.1
KILL-FINAL-ON-8308514/rank5-2,9,24.3,11.4,22.8,8.5,0.0,3.7,331.8,1.2,2.8,0.4,407.1
KILL-FINAL-ON-8308514/rank5-2,10,11.9,23.9,19.7,7.4,0.0,8.0,331.7,1.3,2.6,0.6,407.2
KILL-FINAL-ON-8308514/rank5-2,11,10.5,25.2,27.2,4.3,0.0,3.6,331.8,1.2,2.6,0.6,407.1
KILL-FINAL-ON-8308514/rank5-2,12,24.6,11.0,19.7,10.9,0.0,4.6,331.8,1.2,2.8,0.4,407.0
KILL-FINAL-ON-8308514/rank5-2,13,11.6,24.0,22.0,8.8,0.0,4.4,331.9,1.1,2.9,0.4,407.0
KILL-FINAL-ON-8308514/rank5-2,14,11.9,23.8,22.0,8.8,0.0,4.4,332.0,1.1,2.5,0.6,407.0
KILL-FINAL-ON-8308514/rank5-2,15,10.8,24.6,20.0,10.9,0.0,4.5,332.0,1.0,2.5,0.5,406.9
KILL-FINAL-ON-8308514/rank5-2,16,11.0,24.4,19.6,8.4,0.0,7.2,331.8,1.1,2.8,0.3,406.9
KILL-FINAL-ON-8308514/rank5-2,17,24.3,11.1,25.1,3.3,0.0,6.9,332.1,0.9,2.8,0.4,406.9
KILL-FINAL-ON-8308514/rank5-2,18,9.9,25.6,19.4,8.7,0.0,7.2,332.0,1.0,2.6,0.5,406.9
KILL-FINAL-ON-8308514/rank5-2,19,11.4,24.1,28.4,2.9,0.0,4.0,331.9,1.1,2.6,0.5,406.9
KILL-FINAL-ON-8308514/rank5-2,20,24.2,11.2,19.2,4.0,0.0,12.1,332.1,0.8,2.8,0.4,406.8
KILL-FINAL-ON-8308514/rank5-2,21,11.2,24.2,23.2,8.2,0.0,3.7,332.0,0.9,2.8,0.4,406.8
KILL-FINAL-ON-8308514/rank5-2,22,10.7,24.7,23.3,8.2,0.0,3.8,331.9,1.1,2.6,0.5,406.8
KILL-FINAL-ON-8308514/rank5-2,23,12.0,23.3,23.3,8.2,0.0,3.8,332.0,0.9,2.6,0.5,406.8
KILL-FINAL-ON-8308514/rank5-2,24,11.6,23.8,19.7,9.1,0.0,6.4,331.8,1.2,2.5,0.6,406.8
KILL-FINAL-ON-8308514/rank5-2,25,9.9,25.5,25.2,3.6,0.0,6.4,331.9,1.1,2.6,0.6,406.8
KILL-FINAL-ON-8308514/rank5-2,26,24.2,11.2,22.0,6.7,0.0,6.5,331.7,1.2,2.8,0.4,406.8
KILL-FINAL-ON-8308514/rank5-2,27,10.4,25.0,28.9,3.1,0.0,3.2,331.9,1.0,2.8,0.4,406.8
KILL-FINAL-ON-8308514/rank5-2,28,9.6,25.7,21.4,2.8,0.0,11.0,325.2,7.7,2.7,0.4,406.7
KILL-FINAL-ON-8308514/rank5-2,30,9.5,25.7,17.1,4.0,0.0,14.1,323.7,9.2,2.9,0.3,406.7
KILL-FINAL-ON-8308514/rank5-2,31,10.5,24.8,14.6,6.4,0.0,14.3,324.8,8.2,2.8,0.3,406.7
```

- **Aggregation** (3.4).
  - Row = the survivor with the median total.
  - `min_total_ms` and `max_total_ms` are over the 31 survivors.
  - `max_import_map_ms` is the largest import_map.

```csv
# block: figA5_agg
case,rank,release_ms,barrier_ms,make_room_ms,cxg_ms,import_map_ms,outcome_ms,rest_ms,total_ms,min_total_ms,max_total_ms,max_import_map_ms
KILL-FINAL-OFF-8308514/rank5-1,9,11.5,42.4,18.7,21.8,347.9,475.5,2.7,920.5,895.3,942.0,822.4
KILL-FINAL-OFF-8308514/rank5-2,9,13.8,84.4,34.0,23.3,820.8,13.3,2.8,992.4,978.6,998.1,833.1
KILL-FINAL-OFF-B-8308514/rank5-1,28,10.7,115.1,29.5,23.2,811.1,15.6,2.7,1007.9,909.9,1008.4,825.9
KILL-FINAL-ON-8308514/rank5-1,19,11.4,19.9,28.0,14.7,315.8,4.1,2.6,396.5,396.2,397.0,318.9
KILL-FINAL-ON-8308514/rank5-2,15,10.8,25.1,20.0,15.4,332.0,1.0,2.6,406.9,399.1,407.3,332.1
```

### Remap table (doc, Late Writes: "Cost at recovery")

- **What it shows.** Per-survivor remap time of the 7.5 GB all-to-all workspace for single v5-final kills.
- **Source.** V5F-REC 8269102 rank logs (CLAIMS 193).
  - The two-loss rows are the first and second remap of `twokill-1`.
  - "Proposed -> ready" comes from `kill_runs`: 0.54 / 0.48 / 0.48 / 1.25 s for the rank, node, prefill and no-EPLB
    rows. The frozen and two-loss rows show "-" in the doc.
- **Raw.** Every survivor's remap ms (`rank:ms`):

```csv
# block: remap_survivors
case,run,n_survivors,remap_ms_by_survivor_rank
rank_online,KILL-FINAL-ON-8269102/rank5-1,31,0:537.0 1:537.1 2:536.9 3:537.0 4:536.8 5:536.8 6:536.8 7:536.7 8:536.7 9:536.7 10:536.7 11:536.7 12:536.7 14:536.6 15:536.6 16:536.5 17:536.6 18:536.7 19:536.6 20:536.7 21:536.5 22:536.5 23:536.5 24:536.5 25:536.5 26:536.5 27:536.5 28:455.0 29:456.5 30:461.2 31:467.2
node_online,KILL-FINAL-ON-8269102/node1-1,28,0:484.5 1:484.4 2:484.4 3:484.4 4:484.3 5:484.3 6:484.4 7:484.3 8:484.2 9:484.2 10:484.2 11:484.2 16:476.5 17:484.1 18:484.0 19:484.0 20:408.1 21:408.4 22:414.0 23:405.0 24:484.0 25:484.0 26:484.0 27:484.0 28:483.8 29:483.9 30:483.9 31:483.8
prefill_online,KILL-FINAL-ON-A-8269102/rank5pf-1,31,0:485.2 1:485.1 2:485.2 3:485.2 4:485.0 5:485.0 6:485.0 7:484.9 8:484.9 9:484.9 10:484.8 11:484.9 12:484.8 14:484.8 15:484.8 16:484.6 17:484.8 18:484.8 19:484.8 20:484.7 21:484.6 22:484.6 23:484.6 24:484.6 25:484.6 26:484.6 27:484.6 28:484.5 29:484.5 30:484.5 31:484.5
frozen_online,KILL-FINAL-ON-A-8269102/stop-1,31,0:591.4 1:591.4 3:582.2 4:591.3 5:591.2 6:591.2 7:591.2 8:584.0 9:586.0 10:585.9 11:588.7 12:591.1 13:591.1 14:591.1 15:591.1 16:588.2 17:586.5 18:591.0 19:586.6 20:460.9 21:453.0 22:451.9 23:448.2 24:590.8 25:590.8 26:590.9 27:590.9 28:590.7 29:590.7 30:590.8 31:590.7
twoloss_first_online,KILL-FINAL-ON-A-8269102/twokill-1,31,0:466.4 1:466.4 2:464.7 3:466.3 4:466.3 5:466.3 6:466.3 7:466.1 8:451.6 9:459.8 10:460.2 11:457.0 12:466.2 14:466.2 15:466.2 16:466.0 17:466.1 18:466.1 19:466.1 20:465.8 21:466.0 22:466.0 23:465.6 24:466.0 25:466.0 26:466.0 27:466.0 28:466.0 29:466.0 30:466.0 31:466.0
twoloss_second_online,KILL-FINAL-ON-A-8269102/twokill-1,30,0:478.8 1:478.7 2:478.7 3:478.6 4:478.5 5:478.5 6:478.5 7:478.4 8:478.4 9:478.5 10:478.4 11:478.4 12:478.4 14:478.4 15:478.3 16:478.1 18:478.3 19:478.3 20:478.3 21:478.3 22:478.3 23:478.3 24:478.2 25:478.2 26:478.2 27:478.3 28:478.1 29:478.2 30:478.2 31:478.2
rank_noeplb,KILL-FINAL-OFF-8269102/rank5-1,31,0:973.1 1:973.1 2:973.0 3:973.0 4:943.1 6:948.4 7:949.8 8:969.1 9:968.9 10:972.9 11:971.2 12:911.7 13:911.7 14:913.5 15:917.6 16:951.0 17:957.8 18:959.6 19:952.1 20:961.1 21:961.3 22:961.1 23:969.2 24:965.3 25:959.3 26:962.1 27:964.2 28:972.5 29:971.7 30:972.5 31:972.5
```

- **Aggregate.** Min / median / max over the survivors, as in the doc. All rows took 0 bytes from the KV cache.

```csv
# block: remap_agg
case,n_survivors,remap_min_ms,remap_median_ms,remap_max_ms,proposed_to_ready_s
rank_online,31,455.0,536.6,537.1,0.54
node_online,28,405.0,484.0,484.5,0.48
prefill_online,31,484.5,484.8,485.2,0.48
frozen_online,31,448.2,590.8,591.4,
twoloss_first_online,31,451.6,466.0,466.4,
twoloss_second_online,30,478.1,478.3,478.8,
rank_noeplb,31,911.7,962.1,973.1,1.25
```

## 5. Other headline tables

### 5.1 Accuracy (GSM8K, v5-final)

- **Source.** `R/mvp/runs/ACCURACY-KILL-ON-8269102/accuracy_kill.txt` (CLAIMS 193).
- **Raw.** One evaluation per arm, 1319 questions each.

```csv
# block: accuracy
arm,ft,kill,strict_match,strict_se,flexible_extract,flex_se,n,non200,failed_ranks,resume_s
acc-kill-final-1,on,rank 5 before eval,0.9477,0.0061,0.9507,0.0060,1319,0,5,1.01
acc-ref-final,off,none,0.9469,0.0062,0.9492,0.0060,1319,0,,
```

- **Aggregate.**
  - Delta = +0.0008.
  - 2 SE = 2 x sqrt(0.0061^2 + 0.0062^2) = 0.0174; the delta is within 2 SE.
  - Questions answered differently (strict match): right-to-wrong 21, wrong-to-right 22, out of 57 differing answers.
  - The doc's pre-fix and fix-1 accuracy rows are older builds and are not included here.

### 5.2 Memory

- **Restored-expert GPU memory per survivor.** The `restored_mb_max` column of `kill_runs`:
  - no EPLB: rank 444 MB, node 1752 MB, two losses 912 MB;
  - online: 396 / 1476 / 780 MB;
  - offline: 420 / 1476 MB.
- **Busiest vs median survivor** (`restored_ratio`): rank kills 1.17-1.27, node kills 1.00-1.05.
- **Host memory.** `/dev/shm` use and the minimum MemFree over the 8 nodes, from each arm's `hostmem-<node>.log`
  (`df` line 2 and `free` line 4, in GB = 1e9 bytes):

```csv
# block: host_memory
arm,mode,ft,dev_shm_used_gb_per_node,min_memfree_gb_over_8_nodes
STD-G2-8269103/std-offline-1,offline,off,0.0,745.1
STD-G2-8269103/std-offline-ft-1,offline,on,390.4,357.6
STD-G3-8269103/std-offline-1,offline,off,0.0,759.2
STD-G3-8269103/std-offline-ft-1,offline,on,390.4,356.2
STD-G1-8270779/std-offline-1,offline,off,0.0,752.5
STD-G1-8270779/std-offline-ft-1,offline,on,390.4,356.7
STD-G1-8271451/std-offline-1,offline,off,0.0,753.3
STD-G1-8271451/std-offline-ft-1,offline,on,390.4,374.4
STD-G2-8270779/std-noeplb-1,noeplb,off,0.0,
STD-G2-8270779/std-noeplb-ft-1,noeplb,on,48.8,
STD-8269103/std-online-1,online,off,390.4,
STD-8269103/std-online-ft-1,online,on,390.4,
```

- **What the host-memory rows show.**
  - Offline FT holds 390.4 GB of host copies per node, while offline FT off holds none.
  - No EPLB FT holds ring copies (48.8 GB per node).
  - Online EPLB holds 390.4 GB in both arms (main's own host copies).
  - Min MemFree for the offline 32/GPU arms: FT 356.2-374.4 GB vs off 745.1-759.2 GB (`offline32_analysis.md`
    section 3).

### 5.3 Request fate

- **Source.** The request columns of `kill_runs`:
  - `client_cut` = streams cut off by the loss (errored or short, less resets);
  - `client_failed` = other client failures;
  - `reruns` = requests of the lost rank that ran again.
- **Rank kills.** 128 requests cut (3.1% of 4096).
- **Node kills.** 512 cut (12.5%).
- **Every row.** `client_failed` = 0.
- **Reruns.** 0 in every rank and node row. Up to 7 in the other rows (two losses 7 / 2 / 7 / 2, prefill 2 / 4).
- **Restart instead.** All 4096 in-flight requests fail (`failed_inflight`).
- **Stress runs.** The failed requests equal the server's count for the lost ranks in all 8 losses (`stress_kills`
  below).

### 5.4 Stress tests

**S1** (healthy, FT on vs off, jobs 8310100, S1B 8359063 and S1C 8363138, CLAIMS 217, 234, 240): per arm.

- Measured requests = the requests in the healthy window.
- Two FAILs, both FT-off arms in group G8, by the memory rule: job 1 node 0's MemAvailable fell 6.9 GB, which was not in
  any serving process (RssAnon growth <= 4.7 MiB); S1B node 4's used memory grew 3.2 GB (> 2048 MiB). FT-on arms: 6/6 PASS;
  S1C: 4/4 PASS.

```csv
# block: stress_s1_arms
group,order_in_group,arm,req_per_s,requests,ok,measured_requests,calib32_tok_s,ttft_p50_ms,ttft_p99_ms,tpot_p50_ms,tpot_p99_ms,verdict,failed_checks
G0,first,s1-off,51.0,107100,107100,92218,67875,816.9,2018.0,11.7,13.4,PASS,-
G0,second,s1-on,51.0,107100,107100,92266,66802,816.6,2052.5,11.8,13.5,PASS,-
G8,second,s1-off,51.0,107100,107100,92253,66369,827.1,2039.5,11.8,13.7,FAIL,no_memory_growth=False
G8,first,s1-on,51.0,107100,107100,92254,65921,831.9,2049.9,11.9,13.6,PASS,-
S1B-G0,first,s1-off,51.0,107100,107100,92268,66124,819.6,2000.1,11.7,13.5,PASS,-
S1B-G0,second,s1-on,51.0,107100,107100,92274,63580,826.9,2046.7,11.8,13.4,PASS,-
S1B-G8,second,s1-off,51.0,107100,107100,92281,65738,815.3,1998.4,11.8,13.4,FAIL,no_memory_growth=False
S1B-G8,first,s1-on,51.0,107100,107100,92265,66817,829.8,2016.4,11.9,13.6,PASS,-
S1C-G0,first,s1-off,51.0,107100,107100,92269,66165,827.4,2108.0,11.8,19.8,PASS,-
S1C-G0,second,s1-on,51.0,107100,107100,92262,67045,825.3,2016.4,11.9,13.5,PASS,-
S1C-G8,second,s1-off,51.0,107100,107100,92278,62973,816.9,2004.9,11.8,13.4,PASS,-
S1C-G8,first,s1-on,51.0,107100,107100,92242,66848,835.1,2337.8,11.8,14.5,PASS,-
```

```csv
# block: stress_s1_served
pair,off_served_tok_s,on_served_tok_s,off_calib32_tok_s,on_calib32_tok_s,served_change_pct
pair1,52365.2,52257.6,67874.9,66802.3,-0.21
pair2,52216.2,52188.4,66369.1,65921.0,-0.05
pair3,52165.9,52163.7,66124.4,63580.1,-0.00
pair4,52144.4,52163.5,65737.6,66816.6,+0.04
```

**S2 and S3** (random kills under traffic, FT on). One row per loss:

```csv
# block: stress_kills
test,job,case,kill,ranks,timing,at_s,resume_s,client_failed,server_failed,reran,served_tok_s_before,served_tok_s_after,ratio,floor,health_sequence,health_recovering_s
S2,8299473,s2-rank14,1,14,at,953.7,1.03,21,21,0,51917,51890,0.999,0.85,healthy>recovering>degraded,0.91
S2,8299473,s2-rank13,1,13,at,1895.0,1.03,19,19,1,52558,53073,1.01,0.85,healthy>degraded,None
S2,8318006,s2-rank25,1,25,at,2182.7,1.05,18,18,1,52728,52883,1.003,0.85,healthy>degraded,None
S2,8318006,s2-node2,1,8+9+10+11,at,730.7,1.96,73,73,2,52037,52325,1.006,0.75,healthy>recovering>degraded,0.9
S3,8314070,s3-rank29-rank14,1,29,at,731.0,1.02,22,22,0,51061,51292,1.005,0.85,healthy>recovering>degraded,0.9
S3,8314070,s3-rank29-rank14,2,14,after,1632.2,0.99,23,23,1,51168,51752,1.011,0.85,degraded>recovering>degraded,0.9
S3,8314070,s3-rank12-rank31,1,12,at,1221.2,0.87,19,19,4,52786,52642,0.997,0.85,healthy>recovering>degraded,0.9
S3,8314070,s3-rank12-rank31,2,31,after,2122.2,1.03,20,20,4,52758,52927,1.003,0.85,degraded,None
```

Per case:

```csv
# block: stress_cases
test,job,case,req_per_s,requests,ok,retryable,healthy_ttft_p50_ms,healthy_tpot_p50_ms,after_ttft_p50_ms,after_tpot_p50_ms,capacity_calib_end_over_calib,verdict,failed_checks
S2,8299473,s2-rank14,50.73,188706,188685,21,826.2,11.8,1020.8,13.2,0.93,PASS,-
S2,8299473,s2-rank13,51.33,190958,190939,19,842.1,11.8,1049.2,13.1,0.927,FAIL,k1_health=False
S2,8318006,s2-rank25,51.57,191838,191820,18,833.3,11.8,1035.7,13.2,0.928,FAIL,k1_health=False
S2,8318006,s2-node2,51.28,190769,190696,73,825.1,12.0,1058.6,13.3,0.919,PASS,-
S3,8314070,s3-rank29-rank14,50.32,187183,187138,45,826.9,11.9,1033.0,13.3,,PASS,-
S3,8314070,s3-rank12-rank31,51.61,191971,191932,39,842.5,11.8,1044.2,13.2,,PASS,-
```

- **S2.** 4 recoveries (3 rank kills, 1 node kill), all recovered.
  - Failed requests = the server's count in all 4.
  - Two cases fail only the `/health` check: `recovering` was not observed, because of the 1 s answer cache.
- **S3.** Two servers, each with two losses. Both PASS.
- **Recovering window.** The authority's recovering windows were 0.862 / 0.741 / 0.667 / 0.670 s (CLAIMS 218; from
  `authority.log`, not in the blocks above).

### 5.5 Per-step cost and pooled bounds

- **Source.** `finals_stats.py` cells (3.1): `review/diag/repl_cost_pooled.txt` (pooled n=8 at 32/GPU),
  `review/v5f_readout_full.txt` (original n=4, 32 and 128/GPU), `review/diag/repl_cost_readout.txt` (replication n=4
  at 32/GPU) and `review/diag/healthy128_{repl,pooled}_readout.txt` (replication n=4 and pooled n=8 at 128/GPU).
- **Raw.** `healthy_pairs`.

```csv
# block: healthy_cells_agg
load,mode,batch,metric,n,mean_pct,ci95_lo,ci95_hi,one_sided95_lb,one_sided95_ub
32,noeplb,original_n4,client_tok_s_gpu,4,+0.50,-1.40,+2.41,-0.91,+1.91
32,noeplb,original_n4,full_step,4,+0.18,-2.08,+2.43,-1.49,+1.84
32,noeplb,original_n4,trim_step,4,+0.96,+0.68,+1.25,+0.75,+1.18
32,offline,original_n4,client_tok_s_gpu,4,-2.03,-3.56,-0.49,-3.16,-0.89
32,offline,original_n4,full_step,4,+1.73,-1.00,+4.47,-0.29,+3.76
32,offline,original_n4,trim_step,4,+0.59,+0.28,+0.89,+0.36,+0.81
32,online,original_n4,client_tok_s_gpu,4,-0.18,-1.94,+1.57,-1.48,+1.12
32,online,original_n4,full_step,4,+0.63,-0.95,+2.22,-0.54,+1.80
32,online,original_n4,trim_step,4,+0.48,-0.35,+1.30,-0.14,+1.09
32,noeplb,pooled_n8,client_tok_s_gpu,8,+0.25,-0.78,+1.28,-0.58,+1.07
32,noeplb,pooled_n8,full_step,8,+0.27,-0.79,+1.33,-0.58,+1.12
32,noeplb,pooled_n8,trim_step,8,+0.87,+0.72,+1.02,+0.75,+0.99
32,offline,pooled_n8,client_tok_s_gpu,8,-0.69,-2.80,+1.42,-2.38,+1.00
32,offline,pooled_n8,full_step,8,+1.14,-0.16,+2.43,+0.10,+2.18
32,offline,pooled_n8,trim_step,8,+0.50,+0.34,+0.66,+0.37,+0.63
32,online,pooled_n8,client_tok_s_gpu,8,+0.05,-0.81,+0.91,-0.63,+0.74
32,online,pooled_n8,full_step,8,-0.32,-1.70,+1.06,-1.42,+0.79
32,online,pooled_n8,trim_step,8,+0.35,-0.04,+0.75,+0.03,+0.67
32,noeplb,replication_n4,client_tok_s_gpu,4,-0.01,-2.22,+2.20,-1.64,+1.62
32,noeplb,replication_n4,full_step,4,+0.37,-1.73,+2.47,-1.19,+1.92
32,noeplb,replication_n4,trim_step,4,+0.77,+0.57,+0.97,+0.62,+0.92
32,offline,replication_n4,client_tok_s_gpu,4,+0.65,-4.16,+5.46,-2.90,+4.21
32,offline,replication_n4,full_step,4,+0.54,-1.53,+2.61,-0.99,+2.07
32,offline,replication_n4,trim_step,4,+0.41,+0.14,+0.68,+0.21,+0.61
32,online,replication_n4,client_tok_s_gpu,4,+0.29,-1.37,+1.95,-0.94,+1.52
32,online,replication_n4,full_step,4,-1.27,-4.00,+1.46,-3.29,+0.75
32,online,replication_n4,trim_step,4,+0.23,-0.50,+0.95,-0.31,+0.76
128,noeplb,original_n4,client_tok_s_gpu,4,-1.15,-3.31,+1.01,-2.74,+0.45
128,noeplb,original_n4,full_step,4,+0.93,-1.48,+3.34,-0.85,+2.71
128,noeplb,original_n4,trim_step,4,+0.53,+0.06,+1.01,+0.18,+0.88
128,offline,original_n4,client_tok_s_gpu,4,-0.35,-4.13,+3.43,-3.14,+2.45
128,offline,original_n4,full_step,4,+0.60,-1.90,+3.09,-1.25,+2.44
128,offline,original_n4,trim_step,4,+0.34,-0.02,+0.69,+0.08,+0.60
128,online,original_n4,client_tok_s_gpu,4,-0.05,-1.67,+1.57,-1.25,+1.15
128,online,original_n4,full_step,4,+0.52,+0.12,+0.92,+0.23,+0.81
128,online,original_n4,trim_step,4,+0.39,+0.10,+0.68,+0.18,+0.61
128,noeplb,replication_n4,trim_step,4,+0.59,+0.38,+0.80,+0.43,+0.74
128,noeplb,replication_n4,full_step,4,-0.18,-1.44,+1.07,-1.11,+0.74
128,noeplb,replication_n4,client_tok_s_gpu,4,+0.48,-1.76,+2.72,-1.17,+2.14
128,offline,replication_n4,trim_step,4,+0.44,+0.21,+0.67,+0.27,+0.61
128,offline,replication_n4,full_step,4,-0.17,-1.58,+1.23,-1.21,+0.87
128,offline,replication_n4,client_tok_s_gpu,4,+0.52,-2.39,+3.42,-1.63,+2.67
128,online,replication_n4,trim_step,4,+0.42,+0.21,+0.62,+0.26,+0.57
128,online,replication_n4,full_step,4,+0.15,-0.72,+1.01,-0.49,+0.78
128,online,replication_n4,client_tok_s_gpu,4,-0.71,-2.56,+1.14,-2.08,+0.65
128,noeplb,pooled_n8,trim_step,8,+0.56,+0.38,+0.74,+0.42,+0.70
128,noeplb,pooled_n8,full_step,8,+0.37,-0.68,+1.43,-0.47,+1.22
128,noeplb,pooled_n8,client_tok_s_gpu,8,-0.33,-1.63,+0.96,-1.37,+0.70
128,offline,pooled_n8,trim_step,8,+0.39,+0.24,+0.54,+0.27,+0.51
128,offline,pooled_n8,full_step,8,+0.21,-0.83,+1.26,-0.62,+1.05
128,offline,pooled_n8,client_tok_s_gpu,8,+0.08,-1.60,+1.77,-1.27,+1.43
128,online,pooled_n8,trim_step,8,+0.40,+0.28,+0.53,+0.30,+0.50
128,online,pooled_n8,full_step,8,+0.33,-0.03,+0.70,+0.04,+0.63
128,online,pooled_n8,client_tok_s_gpu,8,-0.38,-1.28,+0.51,-1.10,+0.34
```

Reading the block:

- **Per-step table** (doc "Per-step cost"): the `trim_step` rows, `pooled_n8` at both loads.
  The µs per step is in `figA1_agg`.
- **Pooled one-sided bounds** (doc Overview and "Healthy cost").
  - "FT's throughput loss is at most 0.58% / 2.38% / 0.63%" = -LB of `client_tok_s_gpu`, pooled_n8, no EPLB /
    offline / online.
  - At 128/GPU, pooled n=8: 1.37% / 1.27% / 1.10% (original n=4: 2.74% / 3.14% / 1.25%).
  - Trimmed-step UB at 32/GPU pooled: +0.99 / +0.63 / +0.67%. Full-mean UB: +1.12 / +2.18 / +0.79%.
  - Trimmed-step UB at 128/GPU pooled: +0.70 / +0.51 / +0.50%. Full-mean UB: +1.22 / +1.05 / +0.63%.
- **Replication table** (doc "Replication at 32/GPU"): the `client_tok_s_gpu` rows for original_n4, replication_n4 and
  pooled_n8.
- **Appendix F** (original n=4): the `original_n4` rows.

## 6. Recreating the figures

The original generators are `R/review/figures/final_clean/make_figs.py` (Figs 5, 6, 7, 7b, 8, 10, A1, A2) and
`R/review/figures/final_clean/make_figs_extra.py` (Figs 9, A3, A4, A5). They read the raw run directories and write the
`fig*.csv` files that most blocks above copy.

The script below needs only this file and matplotlib. It recomputes the plotted values from the raw blocks where the
figure is a mean or a median of runs. It compares each one with the aggregate block and prints any mismatch beyond
rounding. CI-based values (Fig 6, Fig A1) are plotted from the aggregate blocks, which use exact t quantiles. It writes
12 PNGs in a plain style. Save it as `recreate.py` and run `python3 recreate.py DATA.md out/`.

```python
"""Redraw the WideEP FT data figures from the CSV blocks of DATA.md (plain matplotlib).
usage: python3 recreate.py DATA.md OUT_DIR"""
import csv, io, math, os, re, statistics as st, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

src, out = open(sys.argv[1]).read(), sys.argv[2]
os.makedirs(out, exist_ok=True)
D = {m[1]: list(csv.DictReader(io.StringIO(m[2])))
     for m in re.finditer(r"`{3}csv\n# block: (\w+)\n(.*?)`{3}", src, re.S)}
MODES = ["noeplb", "offline", "online"]
C = {"noeplb": "#1B4F12", "offline": "#458F19", "online": "#76B900", "off": "#A6A6A6", "on": "#76B900"}
F = float
bad = []


def check(what, got, want, tol):  # a value recomputed from raw rows vs the plotted aggregate
    if abs(got - F(want)) > tol:
        bad.append(f"{what}: raw gives {got:.4f}, aggregate block says {want}")


def save(fig, name):
    fig.tight_layout()
    fig.savefig(f"{out}/{name}.png", dpi=120)
    plt.close(fig)


def rows(block, **eq):
    return [r for r in D[block] if all(r[k] == v for k, v in eq.items())]


# Fig 5: healthy output tok/s per GPU, mean over pairs
fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
for ax, load in zip(axes, ("32", "128")):
    for i, m in enumerate(MODES):
        ps = rows("healthy_pairs", load=load, mode=m)
        agg = rows("healthy_latency_agg", metric="output_throughput", mode=m, load=load)[0]
        for j, arm in enumerate(("off", "on")):
            v = st.mean(F(p[f"{arm}_tok_s_gpu"]) for p in ps)
            check(f"fig5 {load} {m} {arm}", v, agg[f"ft_{arm}_mean"], 0.01)
            ax.bar(i + (j - 0.5) * 0.35, v, 0.33, color=C[arm], label=f"FT {arm}" if i == 0 else None)
        ax.text(i, v + 100, f"{F(agg['mean_pct']):+.1f}%", ha="center")
    ax.set_xticks(range(3), MODES)
    ax.set_title(f"{load} requests/GPU (n={len(ps)} pairs)")
axes[0].set_ylabel("Output tok/s per GPU")
axes[0].legend()
save(fig, "fig5_healthy")

# Fig 6: change with FT on at 32/GPU, mean and 95% t-CI over pairs
METRICS = [("output_throughput", "tok_s_gpu"), ("median_tpot_ms", "med_tpot_ms"), ("p99_tpot_ms", "p99_tpot_ms"),
           ("median_ttft_ms", "med_ttft_ms")]
fig, ax = plt.subplots(figsize=(9, 4.5))
for j, (metric, col) in enumerate(METRICS):
    for k, m in enumerate(MODES):
        a = rows("healthy_latency_agg", metric=metric, mode=m, load="32")[0]
        ps = rows("healthy_pairs", load="32", mode=m)
        check(f"fig6 {metric} {m}", st.mean((F(p[f"on_{col}"]) / F(p[f"off_{col}"]) - 1) * 100 for p in ps),
              a["mean_pct"], 0.006)
        y, mean = j + (k - 1) * 0.25, F(a["mean_pct"])
        ax.barh(y, mean, 0.22, color=C[m], label=m if j == 0 else None)
        ax.errorbar(mean, y, xerr=[[mean - F(a["ci_lo"])], [F(a["ci_hi"]) - mean]], fmt="none", ecolor="grey")
ax.axvline(0, color="k", lw=0.8)
ax.set_yticks(range(4), [m for m, _ in METRICS])
ax.invert_yaxis()
ax.set_xlabel("FT on vs FT off, % (mean, 95% CI), 32 requests/GPU")
ax.legend()
save(fig, "fig6_ft_change")

# Fig 7 and 7b: kill -> rank 0's first step (server timestamps); phases
kills = D["kill_runs"]
fig, ax = plt.subplots(figsize=(9, 4.5))
for i, kind in enumerate(("rank", "node", "frozen")):
    bars = []
    for m in MODES:
        v = [F(r["first_step_s"]) for r in kills if r["failure"] == kind and r["mode"] == m]
        if v:
            check(f"fig7 {kind} {m}", st.mean(v), rows("fig7_agg", failure=kind, mode=m)[0]["mean_s"], 0.001)
            bars.append((st.mean(v), C[m], f"FT on, {m}"))
    if kind == "rank":
        v = [F(r["down_s"]) for r in D["restart_runs"]]
        check("fig7 restart", st.mean(v), rows("fig7_agg", mode="FT off restart")[0]["mean_s"], 0.05)
        bars.append((st.mean(v), C["off"], "FT off, restart"))
    for b, (v, col, lab) in enumerate(bars):
        x = i + (b - (len(bars) - 1) / 2) * 0.18
        ax.bar(x, v, 0.16, color=col, label=lab if i == 0 or "restart" in lab else None)
        ax.text(x, v * 1.1, f"{v:.2f} s" if v < 60 else f"{v / 60:.1f} min", ha="center", fontsize=8)
ax.set_yscale("log")
ax.set_xticks(range(3), ["Lose 1 rank", "Lose 1 node", "Frozen rank"])
ax.set_ylabel("Time to serve again (s, log)")
ax.legend(fontsize=8)
save(fig, "fig7_recovery")

def steps(r):  # per-kill step durations (s); stage = slowest survivor's ring-copy staging, 0 with EPLB
    ready = F(r["ready_s"]) - F(r["proposed_s"])
    stage = min(F(r["stage_s"]), ready)
    return {"detect": F(r["detect_s"]), "agree": F(r["stopped_s"]) - F(r["detect_s"]),
            "map": F(r["proposed_s"]) - F(r["stopped_s"]), "remap": ready - stage, "stage": stage,
            "install": F(r["committed_s"]) - F(r["ready_s"]), "first": F(r["first_step_s"]) - F(r["committed_s"])}


PH = ["detect", "agree", "map", "remap", "stage", "install", "first"]
fig, ax = plt.subplots(figsize=(9, 4.5))
ticks = []
for i, kind in enumerate(("rank", "node")):
    for k, m in enumerate(MODES):
        rs = [steps(r) for r in kills if r["failure"] == kind and r["mode"] == m]
        x, bottom = i + (k - 1) * 0.26, 0.0
        for name, col in zip(PH, ["#D9D9D9", "#A6A6A6", "#C9E3A0", "#76B900", "#1B4F12", "#FBEBDB", "#F3D9C2"]):
            agg = rows("fig7b_agg", failure=kind, mode=m, phase=name)[0]
            check(f"fig7b {kind} {m} {name} median", st.median(r[name] for r in rs), agg["median_s"], 0.006)
            v = st.mean(r[name] for r in rs)
            check(f"fig7b {kind} {m} {name} mean", v, agg["mean_s"], 0.002)
            ax.bar(x, v, 0.22, bottom=bottom, color=col, label=name if (i, k) == (0, 0) else None)
            bottom += v
        ticks.append((x, f"{kind}\n{m}"))
ax.set_xticks([t[0] for t in ticks], [t[1] for t in ticks], fontsize=8)
ax.set_ylabel("Seconds since the kill (mean per step)")
ax.legend()
save(fig, "fig7b_phases")

# Fig 8: client throughput after / before, median per mode
fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
for ax, kind, ideal in zip(axes, ("rank", "node"), (31 / 32, 28 / 32)):
    for k, m in enumerate(MODES):
        v = st.median(F(r["ratio"]) for r in rows("after_loss_runs", failure=kind, mode=m))
        check(f"fig8 {kind} {m}", v, rows("fig8_agg", failure=kind, mode=m)[0]["median_ratio"], 0.0006)
        ax.barh(k, v * 100, 0.6, color=C[m])
        ax.text(v * 100 - 1, k, f"{v * 100:.1f}%", ha="right", va="center", color="white")
    ax.axvline(ideal * 100, ls="--", color="grey")
    ax.set_xlim(0, 100)
    ax.set_title(f"Lose 1 {kind} (dashed: share of GPUs left)")
    ax.set_xlabel("Client throughput after / before (%)")
axes[0].set_yticks(range(3), MODES)
axes[0].invert_yaxis()
save(fig, "fig8_after_loss")

# Fig 9: served output tok/s in 10 s bins around a kill (the doc figure's three losses; the block holds all eight)
panels = sorted({(r["test"], r["kill"], r["ranks"]) for r in D["fig9_bins"] if r["in_doc_figure"] == "yes"})
fig, axes = plt.subplots(1, len(panels), figsize=(12, 4), sharey=True)
for ax, (test, kill, rank) in zip(axes, panels):
    rs = rows("fig9_bins", test=test, kill=kill, ranks=rank, in_doc_figure="yes")
    xs = [(F(r["bin_start_s_from_kill"]) + 5) / 60 for r in rs]
    ax.plot(xs, [F(r["served_out_tok_s"]) for r in rs], color=C["offline"], label="served, FT on")
    ax.axhline(F(rs[0]["offered_out_tok_s"]), ls="--", color="grey", label="offered")
    ax.axvline(0, ls=":", color="grey")
    ax.set_title(f"{test} kill {kill}: rank {rank}")
    ax.set_xlabel("Minutes from the kill")
    ax.set_ylim(0, 70000)
axes[0].set_ylabel("Output tok/s (10 s bins)")
axes[0].legend()
save(fig, "fig9_kill_timeline")

# Fig 10: realistic traffic latency (S1), mean of the same-node pairs
fig, axes = plt.subplots(1, 2, figsize=(9, 4))
for ax, kind in zip(axes, ("ttft", "tpot")):
    for i, q in enumerate(("p50", "p99")):
        rs = rows("fig10_pairs", metric=f"{kind}_{q}_ms")
        for j, arm in enumerate(("off", "on")):
            v = st.mean(F(r[f"ft_{arm}_ms"]) for r in rs)
            check(f"fig10 {kind} {q} {arm}", v, rows("fig10_agg", metric=f"{kind}_{q}_ms")[0][f"ft_{arm}_ms"], 0.006)
            ax.bar(i + (j - 0.5) * 0.35, v, 0.33, color=C[arm], label=f"FT {arm}" if i == 0 else None)
    ax.set_xticks(range(2), ["p50", "p99"])
    ax.set_title(kind.upper() + " (ms)")
axes[0].legend()
save(fig, "fig10_realistic_latency")

# Fig A1: per decode step cost (trimmed mean, 95% CI): pooled n=8 at 32 and 128/GPU
fig, ax = plt.subplots(figsize=(8, 4))
for i, (load, batch) in enumerate((("32", "pooled_n8"), ("128", "pooled_n8"))):
    for k, m in enumerate(MODES):
        a = rows("healthy_cells_agg", load=load, mode=m, batch=batch, metric="trim_step")[0]
        ps = rows("healthy_pairs", load=load, mode=m)
        us = st.mean((F(p["trim_on_ms"]) - F(p["trim_off_ms"])) * 1000 for p in ps)
        check(f"figA1 us {load} {m}", us, rows("figA1_agg", load=load, mode=m)[0]["extra_us_per_step"], 0.15)
        mean, x = F(a["mean_pct"]), i + (k - 1) * 0.22
        ax.bar(x, mean, 0.2, color=C[m], label=m if i == 0 else None)
        ax.errorbar(x, mean, yerr=[[mean - F(a["ci95_lo"])], [F(a["ci95_hi"]) - mean]], fmt="none", ecolor="grey")
        ax.text(x, F(a["ci95_hi"]) + 0.03, f"{us:.0f} us", ha="center", fontsize=8)
ax.axhline(1, ls="--", color="grey")
ax.set_xticks(range(2), ["32 requests/GPU", "128 requests/GPU"])
ax.set_ylabel("Trimmed step time, FT on vs off (%)")
ax.legend()
save(fig, "figA1_step_cost")

# Fig A2: output tok/s per GPU vs per-user tok/s (1000 / median TPOT), means over pairs
fig, axes = plt.subplots(1, 3, figsize=(11, 4), sharey=True, sharex=True)
for ax, m in zip(axes, MODES):
    for arm in ("off", "on"):
        xs, ys = [], []
        for load in ("32", "128"):
            ps = rows("healthy_pairs", load=load, mode=m)
            xs.append(st.mean(1000 / F(p[f"{arm}_med_tpot_ms"]) for p in ps))
            ys.append(st.mean(F(p[f"{arm}_tok_s_gpu"]) for p in ps))
            check(f"figA2 {m} {load} {arm}", xs[-1], rows("figA2_agg", mode=m, load=load, arm=arm)[0]["tok_s_per_user"], 0.002)
        ax.plot(xs, ys, marker="o", color=C[arm], label=f"FT {arm}")
    ax.set_title(m)
    ax.set_xlabel("tok/s per user")
axes[0].set_ylabel("Output tok/s per GPU")
axes[0].legend()
save(fig, "figA2_throughput")

# Fig A3: us saved per all-to-all round trip by removing each FT check (EP32, 64 tokens/rank)
ITEMS = [("S", ["S"]), ("F", ["F"]), ("L", ["L"]), ("A+SO", ["A", "SO"]), ("ALL", ["ALL"])]
fig, ax = plt.subplots(figsize=(8, 4))
for j, (name, parts) in enumerate(ITEMS):
    for c, (case, col) in enumerate((("with_stats", C["offline"]), ("without_stats", C["off"]))):
        v = sum(F(rows("figA3_launches", item_removed=p, case=case)[0]["cut_mean_us"]) for p in parts)
        want = rows("figA3_agg", check_removed=name, case=case.replace("_", " ").capitalize().replace("stats", "EPLB stats"))
        check(f"figA3 {name} {case}", v, want[0]["us_saved"], 0.006)
        ax.barh(j + (c - 0.5) * 0.36, v, 0.34, color=col, label=case if j == 0 else None)
for case in ("with_stats", "without_stats"):
    ax.axvline(F(rows("figA3_launches", item_removed="CONTROL_D0", case=case)[0]["D0_us"]), ls=":", color="grey")
ax.set_yticks(range(len(ITEMS)), [n for n, _ in ITEMS])
ax.invert_yaxis()
ax.set_xlabel("us saved per round trip (dotted: FT total)")
ax.legend()
save(fig, "figA3_a2a_ablation")

# Fig A4: restart without FT (mean of runs, by part) vs FT recovery after a rank kill (Fig 7 means)
fig, ax = plt.subplots(figsize=(9, 4))
for i, m in enumerate(MODES):
    segs = []
    for r in rows("restart_runs", mode=m):
        load = F(r["rank0_model_load_s"])
        segs.append([load, F(r["warmup_s"]) - F(r["weights_s"]), F(r["graphs_s"]) - F(r["warmup_s"]),
                     F(r["exit_s"]) + F(r["weights_s"]) - load + F(r["first_token_s"]) - F(r["graphs_s"])])
    mean, left = [st.mean(s[k] for s in segs) for k in range(4)], 0.0
    check(f"figA4 {m}", sum(mean), rows("figA4_agg", what="restart", mode=m)[0]["total_s"], 0.15)
    for k, (v, lab) in enumerate(zip(mean, ["model load", "autotune", "graph capture", "other"])):
        ax.barh(i - 0.19, v, 0.34, left=left, color=["#6E6E6E", "#A6A6A6", "#C8C8C8", "#E6E6E6"][k], label=lab if i == 0 else None)
        left += v
    ft = st.mean(F(r["first_step_s"]) for r in kills if r["failure"] == "rank" and r["mode"] == m)
    ax.barh(i + 0.19, ft, 0.34, color=C[m])
    ax.text(left + 5, i - 0.19, f"{left / 60:.1f} min", va="center")
    ax.text(ft + 5, i + 0.19, f"FT {ft:.1f} s", va="center")
ax.set_yticks(range(3), MODES)
ax.invert_yaxis()
ax.set_xlabel("Seconds until serving again (rank kill)")
ax.legend(fontsize=8)
save(fig, "figA4_restart_vs_ft")

# Fig A5: remap phases of the median survivor (16th of 31 by total), REMAP-TIMER 8308514
fig, ax = plt.subplots(figsize=(9, 4.5))
cases = list(dict.fromkeys(r["case"] for r in D["figA5_survivors"]))
for i, case in enumerate(cases):
    rs = sorted(rows("figA5_survivors", case=case), key=lambda r: F(r["total"]))
    r = {k: F(v) for k, v in rs[len(rs) // 2].items() if k != "case"}
    segs = {"release": r["release"], "barrier": r["barrier1"] + r["barrier2"], "make_room": r["make_room"],
            "create+export+gather": r["create"] + r["export"] + r["gather"], "import_map": r["import_map"],
            "outcome (wait)": r["outcome"]}
    segs["rest"] = r["total"] - sum(segs.values())
    check(f"figA5 {case}", r["total"], rows("figA5_agg", case=case)[0]["total_ms"], 0.06)
    bottom = 0.0
    for k, (lab, v) in enumerate(segs.items()):
        ax.bar(i, v, 0.55, bottom=bottom, color=plt.cm.Greens(0.15 + 0.12 * k), label=lab if i == 0 else None)
        bottom += v
ax.set_xticks(range(len(cases)), [c.replace("KILL-FINAL-", "").replace("-8308514/", "\n") for c in cases], fontsize=8)
ax.set_ylabel("Remap time (ms), median survivor")
ax.legend(fontsize=8)
save(fig, "figA5_remap_phases")

print(f"wrote 12 figures to {out}")
print("all recomputed values match the aggregate blocks" if not bad else "\n".join(bad))
```

Test (Oct 26, 2026): run with `/usr/bin/python3` against this file, the script wrote all 12 figures and reported no
mismatch: `wrote 12 figures to <out>` / `all recomputed values match the aggregate blocks`.

## 7. Known limits of the data

**n per cell.**

| Data | n |
|---|---|
| Healthy cost, 32/GPU | 8 pairs per mode (4 original + 4 replication) |
| Healthy cost, 128/GPU | 4 pairs per mode |
| Rank / node kills | no EPLB 4 / 4; offline 3 / 3; online 5 / 4 (rank includes 1 EPLB-update kill) |
| Frozen rank | 2 per mode (no EPLB, online); offline not run |
| Prefill kill, EPLB-update kill | 1 per mode |
| Two losses | 2 per mode (no EPLB, online) |
| Restart (FT off) | 1 no EPLB, 2 offline, 2 online |
| Client after / before | 11 rank + 11 node kills, one post-kill wave (12-14.5 s) each; healthy waves vary 0-5% |
| S1 | 2 same-node pairs, no CI |
| S2 / S3 | 4 single losses; 2 servers x 2 losses |
| Remap table | one kill per row; n = survivors of that kill |
| Accuracy | one evaluation per arm |

**Claims vs diagnostics.**

- **Claims (CLAIMS rows marked PRODUCT / FINAL / VALIDATION / REPLICATION).** Figs 5-10, Fig A1, Fig A2, Fig A4, the
  remap table, accuracy, request fate and the stress tables.
- **Diagnostic.**
  - Fig A3: ablation build, kernel-only benchmark.
  - Fig A5: instrumented overlay.
  - The pooled TTFT reading (CLAIMS 208).
  - The `after_loss_other` rows (CLAIMS 209).
- **Healthy cost.** At these sample sizes, FT's client throughput cost at 32/GPU cannot be separated from run-to-run
  noise. Every latency and client CI includes 0. The per-step cost is the measured FT cost. "< 1% per step" is not
  claimed (3.1).

**Open items** (no data here):

- Why throughput after a loss sits 1-2 points below the share of GPUs left.
- Why a frozen rank, or a prefill kill with online EPLB, dips more right after recovery.
- Why the first import from the dead GPU's node stalls ~0.5 s in the no-EPLB and offline remaps.
- The cause of the closed-loop TTFT shift (read as a benchmark effect; soak pending).
- The one healthy FT-on hang (1 of 36 FT-on runs, 0 of 31 FT-off; CLAIMS 204).
- The late-write unit-test rerun on 27442bd91d (the doc lists it as queued).

**Not included.**

- FT off vs main (doc "Baseline", CLAIMS 180): main's wheel arms; no raw data here.
- All older-build data the doc labels as historical: Appendix I, the pre-fix and fix-1 FT-on rows, Oct 20
  comparisons, the v3 diagnostics, and the LB-floor arm.

**A note on the offline kill placement.**

- Doc lines 666-667 and CLAIMS 202 say the offline kill runs' placement (`std-offline-ft-filled.yaml`) "differs
  slightly" from the healthy runs'.
- The run directories show otherwise. `initial_global_assignments` in `config-0.yaml` of KILL-FINAL-OFF-B-8283311 and
  -8302191 is identical to that of the healthy offline arms (both come from the same `STD_PLACEMENT` file). Only the
  CUDA graph batch sizes differ.
