## Results

DeepSeek-R1 NVFP4 on 32 GB200 GPUs (8 nodes x 4). Overlap scheduler, MTP, 1k/1k tokens.
FT on vs off on the same build and nodes. Source: review/DESIGN_DOC_v5final_filled_provenance.md.

### Recovery

Time from the kill until every surviving rank serves again (median, min-max, n).

| Failure | No EPLB | Online EPLB |
|---|---|---|
| One rank killed, 32/GPU | 1.7 s (1.2-2.2, n=2) | 1.1 s (1.0-1.2, n=2) |
| One rank killed, 128/GPU | 1.7 s (1.2-2.2, n=3) | 1.1 s (1.0-1.2, n=2) |
| One rank killed in prefill | 1.6 s (1.1-2.1, n=3) | 1.2 s (n=2) |
| One rank killed in an EPLB update | - | 1.2 s (n=2) |
| One node killed (4 ranks) | 3.2 s (n=2) | 1.7 s (n=2) |
| Two losses, one after the other | 1.2 s each (n=2) | 1.2 s each (n=2) |
| Frozen rank (SIGSTOP), 64/GPU | 34.4 s total (34.1-34.6, n=2) | 36.6 s total (36.1-37.1, n=2) |
| Restart instead (FT off) | 318 s (n=1) | 605 s (585-625, n=2) |

- Phases of a rank or node kill (median): detect 0.02 s, plan 0.28 s, ready 1.3 s, commit 1.35 s, first step 1.43 s.
- Rank 0 or the deadline: the job ends cleanly, as designed.

### After the loss

Decode throughput after / before the loss (rank-0 tok/s x live ranks).

| Failure | No EPLB | Online EPLB |
|---|---|---|
| One rank (31/32 live), 128/GPU | 0.95x | 0.96x |
| One node (28/32 live), 128/GPU | 0.88x | 0.88x |

### Accuracy

GSM8K strict match, 1319 questions, a rank killed during the eval.

| Run | FT on, kill | FT off, no kill | Delta |
|---|---|---|---|
| 1 | 0.9469 | 0.9416 | +0.005 (2 SE 0.018) |
| 2 | 0.9431 | 0.9409 | +0.002 (2 SE 0.018) |

### Memory

- Restored expert slots per survivor: 0.4 GB after a rank loss, 1.6 GB after a node loss (busiest survivor).
- Healthy: ghost slots map a few MB of scratch. [FINALS: exact bytes per GPU]

### Baseline

- Our FT off matches main: trim step within -0.26% to +0.37% at 32 and 128/GPU (n=4).

### Healthy cost

Measured with TensorRT-LLM's `benchmark_serving` against `trtllm-serve`: random 1k/1k tokens, `--ignore-eos`,
fixed concurrency (load per GPU x 32). FT on vs off on the same 8 nodes, back to back, order alternated; 4 pairs per
cell. Mean % (95% CI). Build: fix 1 (d71f169e4b) [to be replaced by the v5-final rerun].

| Mode | Load | Output tok/s/GPU | Median TPOT | p99 TPOT |
|---|---|---|---|---|
| No EPLB | 32 | +0.82% (-2.15, +3.78) | -1.97% | -0.76% |
| No EPLB | 128 | -0.54% (-1.50, +0.42) | +1.34% | +0.64% |
| Offline EPLB | 32 | -0.13% (-0.81, +0.55) | -0.18% | +0.47% |
| Offline EPLB | 128 | +0.38% (-1.55, +2.31) | +0.01% | -0.86% |
| Online EPLB | 32 | -3.26% (-9.09, +2.56) | -0.59% | +16.7% |
| Online EPLB | 128 | -0.58% (-2.75, +1.59) | +1.32% | +0.91% |

- Median TPOT moves by at most ~2% in every cell. TTFT shows no significant change (CIs +-6% to +-23%).
- Online 32/GPU: one of four pairs had a ~80 s stall burst (see Known costs); it shows in throughput and p99, not the median.

Per-step breakdown (rank 0's decode step from the server log; diagnostic):

| Mode | Load | Full-mean step | Trim-mean step |
|---|---|---|---|
| No EPLB | 32 | +0.21% (-2.02, +2.45) | +0.95% (+0.83, +1.07) |
| No EPLB | 128 | +1.19% (-0.60, +2.98) | +0.44% (-0.03, +0.92) |
| Offline EPLB | 32 | +1.43% (+0.58, +2.29) | +0.56% (+0.26, +0.86) |
| Offline EPLB | 128 | +0.17% (-0.89, +1.24) | +0.38% (+0.07, +0.69) |
| Online EPLB | 32 | +4.13% (-5.16, +13.42) | +1.09% (+0.28, +1.90) |
| Online EPLB | 128 | +0.88% (-0.53, +2.30) | +0.41% (-0.09, +0.91) |

- Trim mean is the per-step cost without stall steps; full mean includes them.
- 'FT costs < 1% per step' is not claimed: online and no-EPLB at 32/GPU reach above 1%.
- Online 32/GPU throughput: the burst pair -8.6%; the other three -2.1% to -0.4%.
- Recovery and accuracy below are from the frozen build (n=2 per row); on this build every restore row passed again
  with the same times (rank and node kills n=2, other rows n=1) and GSM8K with a kill matched FT off.

### Known costs

- Startup on a used node: when the weights leave the page cache, the kernel compacts memory for ~20-90 s.
  It moves host-copy pages, and every process that maps them is interrupted.
  - FT ranks no longer prefault their peers' copies: stall 21-46 s -> 6-11 s (n=3 each, offline).
  - Each rank also unmaps its own copy after writing it: 16.8 / 15.0 s -> 2.7 / 5.7 s (FT off 2.2 / 3.2 s).
  - Fresh nodes show no burst. The burst falls before the measured windows.
- Steady: FT adds +0.3% to +1.0% per decode step (trim mean).
  - Not from hang detection, the peer-failure poll, the spare-slot memory or the extra slots (320 vs 288: ~0).
  - The all-to-all kernels add ~0.6-0.8 us per dispatch+combine under graph replay (4 ranks, one node);
    at 59 per step that is ~0.04 ms (~0.3%). The rest at 32 GPUs is not yet isolated.
- Offline at 32/GPU: ~2% lower throughput. Offline FT keeps every expert in host memory (~390 GB per node) for
  restores; FT off keeps none. Keeping ring copies (~49 GB, as without EPLB) is the follow-up.
- Mid-window stall bursts (30-80 s, one node's threads waiting on kernel compaction) hit 3 of 12 online FT windows
  before the host-copy fixes; with them, 0 of 4 FT and 0 of 4 FT-off windows.
  - Online EPLB itself reads the host copies every step, so main shares the exposure (one FT-off burst seen in a
    diagnostic run). Driver-allocated host memory cut stalls ~10x in a 1-node test; it needs allocation before
    weight loading and is the next step.
- All arms ran the fence all-to-all (driver 580 is below the 615 that CFT needs). Not measured on CFT.
