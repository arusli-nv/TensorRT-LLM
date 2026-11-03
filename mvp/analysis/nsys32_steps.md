# nsys 32 req/GPU: FT on vs off, decode-step idle vs busy

Data: STD-8141738 {std-noeplb-1 (off), std-noeplb-ft-1 (ft)}, nsys-rank{0,5}.1.nsys-rep; n=1 run per arm. Script: `nsys32_steps.py` (beside this file). Rank-0 SQLite from /tmp/claude-161238/ns32, rank-5 re-exported (nsys export) to scratch.

## Answer (read this first)

**This trace does not reproduce the +0.35 ms FT slowdown, and it shows no host-bound signature in either arm.**
- In the profiled window FT is *faster*: median step period (rank 0) off 15.92 ms vs ft 15.22 ms (-0.70 ms); rank 5 15.92 vs 15.22. (These periods are ~1.4 ms above the 13.79/14.14 ms in the benchmark, so nsys itself perturbs steps, and the window is only 50 steps.)
- GPU idle per step is the same in both arms: median 0.68 (off) vs 0.66 ms (ft) on rank 0; 0.64 vs 0.59 on rank 5. Almost all of it (0.66 ms) is the fixed gap between one graph's last kernel and the next graph's first kernel. Idle inside the graph is 0.08/0.06 ms.
- The host is never late: cudaGraphLaunch is issued a median 5.3 ms (off) / 4.5 ms (ft) BEFORE the previous graph finishes (lateness negative in every percentile incl. p90: -2.9 / -1.7 ms), and each cudaGraphLaunch call blocks for 6.0 / 5.3 ms because the launch queue is full. So the host is ahead; the step is GPU/peer-paced.
- The period difference is in GPU busy time (off 15.17 vs ft 14.57 ms), and sweep attribution says it is MoE dispatch: moeA2ADispatchKernel 0.83 ms/step (off) vs 0.29 (ft), plus moeA2ASanitizeExpertIdsKernel 0.09 vs 0 (the FT graph has 2070 kernels per step vs 2129: the 59 sanitize kernels are gone). Dispatch time includes waiting for the slowest peer rank, so this is cross-rank skew, not a cost FT adds. Everything else is within +/-0.08 ms per kernel.
- Therefore the benchmark's +0.35 ms is not explained by this data: it is neither extra GPU idle nor extra busy here. FT-added host work that does exist is hidden: cuMemRetainAllocationHandle goes from 22 to 205 calls/step on rank 0 (16 -> 105 on rank 5), +0.28 ms/step of host CPU time on rank 0, but the host is ~5 ms ahead so it does not reach the GPU. cudaEventSynchronize is 5 -> 6 per step (+0.004 ms), cudaMemcpyAsync/cudaEventQuery/cudaStreamIsCapturing counts are unchanged. No new per-step sync call of significance.
- The only host-bound steps are the two off-arm stall steps (rank 0, steps 0 and 11 of 49): period 36.9 and 39.8 ms, busy 15.9 and 15.7 ms, so 20.9 and 24.1 ms idle, with the cudaGraphLaunch issued 20.2 and 23.5 ms after the previous graph ended. These are one-off host stalls (cause not visible in the CUDA API trace: no single long sync call in those steps), in the FT-off arm, and they explain why the off mean (15.75 ms excl. stalls vs 16.67 ms) is hurt. FT has none in this window. The host-bound hypothesis can only appear if the host loses its ~5 ms lead.

## Method

- Segmentation: each decode step is exactly one cudaGraphLaunch (50 launches per trace, one graphId 206, so no prefill/ctx steps are present; 49 complete steps). With `--cuda-graph-trace node` the graph kernels have correlationId 0 and graphId set, on 3 graph streams, so they are bound to steps by order: on each stream the k-th block of (kernels on the stream / 50) kernels is step k (counts divide exactly). Step start = earliest graph kernel start. A per-step marker kernel was not needed.
- period = start(N+1) - start(N). busy = union of all kernel + memcpy + memset intervals on all streams (graph and eager) clipped to the period; idle = period - busy. Union is PDL-robust; per-kernel durations are not used for any claim (the "sum of graph kernel durations" row is informational and is ~5 ms above busy because of overlap/PDL spinning).
- Launch lateness = cudaGraphLaunch(N+1) API start - last graph kernel end of step N (>0 would mean the GPU was already idle when the host issued the launch).
- Kernel attribution: sweep over graph kernels sorted by end time, each charged end - max(start, running max end), so overlapped time is charged once and the sums equal graph busy time.
- Stall steps: period > 2x median. Off arm: 2 of 49 steps on both ranks (mean period 16.67 vs 15.75 ms without them). FT arm: 0.

## Per-step tables (ms; n = 49 steps each; "excluding stalls" block drops steps >2x median)


#### r0_off: 49 steps (50 graph launches), 2 stall steps (>2x median 15.92 ms) excluded in the second block

| metric (ms) | median | mean | p10 | p90 |
|---|---|---|---|---|
| **all steps** (n=49) | | | | |
| period | 15.919 | 16.669 | 15.060 | 16.405 |
| GPU busy (union) | 15.173 | 15.060 | 14.380 | 15.635 |
| GPU idle | 0.683 | 1.610 | 0.656 | 0.872 |
| graph window (first-last graph kernel) | 15.189 | 15.080 | 14.401 | 15.654 |
| idle inside graph window | 0.079 | 0.079 | 0.077 | 0.082 |
| graph end -> next graph start | 0.660 | 1.590 | 0.636 | 0.852 |
| launch lateness (cudaGraphLaunch start - prev graph end) | -5.238 | -4.483 | -8.375 | -2.275 |
| next first kernel start - its cudaGraphLaunch start | 5.906 | 6.072 | 3.148 | 9.031 |
| cudaGraphLaunch API duration | 5.904 | 6.070 | 3.147 | 9.030 |
| sum of graph kernel durations (PDL-inflated, info only) | 20.541 | 20.256 | 19.010 | 21.206 |
| **excluding stalls** (n=47) | | | | |
| period | 15.913 | 15.748 | 15.045 | 16.276 |
| GPU busy (union) | 15.167 | 15.028 | 14.363 | 15.526 |
| GPU idle | 0.683 | 0.719 | 0.656 | 0.862 |
| graph window (first-last graph kernel) | 15.186 | 15.048 | 14.383 | 15.547 |
| idle inside graph window | 0.079 | 0.079 | 0.077 | 0.082 |
| graph end -> next graph start | 0.660 | 0.700 | 0.635 | 0.845 |
| launch lateness (cudaGraphLaunch start - prev graph end) | -5.254 | -5.603 | -8.383 | -2.869 |
| next first kernel start - its cudaGraphLaunch start | 6.041 | 6.303 | 3.609 | 9.036 |
| cudaGraphLaunch API duration | 6.040 | 6.301 | 3.608 | 9.034 |
| sum of graph kernel durations (PDL-inflated, info only) | 20.520 | 20.204 | 18.995 | 21.153 |

Host API calls per step (r0_off, non-stall steps): mean count / mean total ms

| API | calls/step | ms/step |
|---|---|---|
| cuKernelGetName | 16.00 | 0.004 |
| cuMemRetainAllocationHandle | 22.00 | 0.095 |
| cudaEventCreateWithFlags | 4.00 | 0.010 |
| cudaEventDestroy | 4.00 | 0.005 |
| cudaEventQuery | 26.45 | 0.060 |
| cudaEventRecordWithFlags | 20.00 | 0.051 |
| cudaEventSynchronize | 5.00 | 0.025 |
| cudaGraphLaunch | 1.00 | 5.939 |
| cudaLaunchKernel | 16.00 | 0.143 |
| cudaMemcpyAsync | 22.00 | 0.204 |
| cudaStreamIsCapturing | 38.00 | 0.025 |
| cudaStreamWaitEvent | 2.00 | 0.006 |

graph kernels per step: 2129

#### r0_ft: 49 steps (50 graph launches), 0 stall steps (>2x median 15.22 ms) excluded in the second block

| metric (ms) | median | mean | p10 | p90 |
|---|---|---|---|---|
| **all steps** (n=49) | | | | |
| period | 15.222 | 15.344 | 15.135 | 15.672 |
| GPU busy (union) | 14.571 | 14.650 | 14.491 | 14.856 |
| GPU idle | 0.664 | 0.694 | 0.628 | 0.826 |
| graph window (first-last graph kernel) | 14.576 | 14.653 | 14.496 | 14.859 |
| idle inside graph window | 0.064 | 0.064 | 0.062 | 0.066 |
| graph end -> next graph start | 0.658 | 0.691 | 0.621 | 0.824 |
| launch lateness (cudaGraphLaunch start - prev graph end) | -4.546 | -4.157 | -6.421 | -1.661 |
| next first kernel start - its cudaGraphLaunch start | 5.256 | 4.848 | 2.359 | 7.057 |
| cudaGraphLaunch API duration | 5.256 | 4.846 | 2.357 | 7.054 |
| sum of graph kernel durations (PDL-inflated, info only) | 19.215 | 19.296 | 19.107 | 19.495 |
| **excluding stalls** (n=49) | | | | |
| period | 15.222 | 15.344 | 15.135 | 15.672 |
| GPU busy (union) | 14.571 | 14.650 | 14.491 | 14.856 |
| GPU idle | 0.664 | 0.694 | 0.628 | 0.826 |
| graph window (first-last graph kernel) | 14.576 | 14.653 | 14.496 | 14.859 |
| idle inside graph window | 0.064 | 0.064 | 0.062 | 0.066 |
| graph end -> next graph start | 0.658 | 0.691 | 0.621 | 0.824 |
| launch lateness (cudaGraphLaunch start - prev graph end) | -4.546 | -4.157 | -6.421 | -1.661 |
| next first kernel start - its cudaGraphLaunch start | 5.256 | 4.848 | 2.359 | 7.057 |
| cudaGraphLaunch API duration | 5.256 | 4.846 | 2.357 | 7.054 |
| sum of graph kernel durations (PDL-inflated, info only) | 19.215 | 19.296 | 19.107 | 19.495 |

Host API calls per step (r0_ft, non-stall steps): mean count / mean total ms

| API | calls/step | ms/step |
|---|---|---|
| cuKernelGetName | 16.00 | 0.004 |
| cuMemRetainAllocationHandle | 205.00 | 0.373 |
| cudaEventCreateWithFlags | 4.00 | 0.009 |
| cudaEventDestroy | 4.00 | 0.005 |
| cudaEventQuery | 26.73 | 0.063 |
| cudaEventRecordWithFlags | 20.00 | 0.051 |
| cudaEventSynchronize | 6.00 | 0.029 |
| cudaGraphLaunch | 1.00 | 4.716 |
| cudaLaunchKernel | 16.00 | 0.149 |
| cudaMemcpyAsync | 22.00 | 0.203 |
| cudaStreamIsCapturing | 38.00 | 0.024 |
| cudaStreamWaitEvent | 2.00 | 0.007 |

graph kernels per step: 2070

#### r5_off: 49 steps (50 graph launches), 2 stall steps (>2x median 15.93 ms) excluded in the second block

| metric (ms) | median | mean | p10 | p90 |
|---|---|---|---|---|
| **all steps** (n=49) | | | | |
| period | 15.930 | 16.665 | 15.045 | 16.403 |
| GPU busy (union) | 15.282 | 15.113 | 14.420 | 15.609 |
| GPU idle | 0.639 | 1.553 | 0.616 | 0.784 |
| graph window (first-last graph kernel) | 15.320 | 15.150 | 14.453 | 15.656 |
| idle inside graph window | 0.095 | 0.097 | 0.092 | 0.102 |
| graph end -> next graph start | 0.604 | 1.515 | 0.583 | 0.744 |
| launch lateness (cudaGraphLaunch start - prev graph end) | -5.322 | -4.564 | -8.454 | -2.412 |
| next first kernel start - its cudaGraphLaunch start | 5.925 | 6.079 | 3.039 | 9.043 |
| cudaGraphLaunch API duration | 5.925 | 6.080 | 3.041 | 9.043 |
| sum of graph kernel durations (PDL-inflated, info only) | 20.786 | 20.431 | 19.124 | 21.346 |
| **excluding stalls** (n=47) | | | | |
| period | 15.924 | 15.746 | 15.024 | 16.291 |
| GPU busy (union) | 15.281 | 15.089 | 14.403 | 15.559 |
| GPU idle | 0.639 | 0.657 | 0.616 | 0.731 |
| graph window (first-last graph kernel) | 15.315 | 15.126 | 14.437 | 15.597 |
| idle inside graph window | 0.095 | 0.096 | 0.092 | 0.103 |
| graph end -> next graph start | 0.604 | 0.620 | 0.583 | 0.691 |
| launch lateness (cudaGraphLaunch start - prev graph end) | -5.331 | -5.688 | -8.462 | -2.917 |
| next first kernel start - its cudaGraphLaunch start | 5.939 | 6.308 | 3.620 | 9.050 |
| cudaGraphLaunch API duration | 5.940 | 6.309 | 3.620 | 9.050 |
| sum of graph kernel durations (PDL-inflated, info only) | 20.722 | 20.393 | 19.108 | 21.290 |

Host API calls per step (r5_off, non-stall steps): mean count / mean total ms

| API | calls/step | ms/step |
|---|---|---|
| cuKernelGetName | 16.00 | 0.004 |
| cuMemRetainAllocationHandle | 16.00 | 0.062 |
| cudaEventCreateWithFlags | 4.00 | 0.009 |
| cudaEventDestroy | 4.00 | 0.005 |
| cudaEventQuery | 26.45 | 0.056 |
| cudaEventRecordWithFlags | 20.00 | 0.042 |
| cudaEventSynchronize | 5.00 | 0.021 |
| cudaGraphLaunch | 1.00 | 5.946 |
| cudaLaunchKernel | 16.00 | 0.111 |
| cudaMemcpyAsync | 22.00 | 0.161 |
| cudaStreamIsCapturing | 38.00 | 0.022 |
| cudaStreamWaitEvent | 2.00 | 0.006 |

graph kernels per step: 2129

#### r5_ft: 49 steps (50 graph launches), 0 stall steps (>2x median 15.22 ms) excluded in the second block

| metric (ms) | median | mean | p10 | p90 |
|---|---|---|---|---|
| **all steps** (n=49) | | | | |
| period | 15.218 | 15.345 | 15.141 | 15.703 |
| GPU busy (union) | 14.632 | 14.719 | 14.563 | 14.957 |
| GPU idle | 0.585 | 0.627 | 0.571 | 0.748 |
| graph window (first-last graph kernel) | 14.634 | 14.722 | 14.569 | 14.963 |
| idle inside graph window | 0.062 | 0.063 | 0.060 | 0.065 |
| graph end -> next graph start | 0.582 | 0.623 | 0.567 | 0.755 |
| launch lateness (cudaGraphLaunch start - prev graph end) | -4.564 | -4.254 | -6.458 | -1.730 |
| next first kernel start - its cudaGraphLaunch start | 5.271 | 4.877 | 2.310 | 7.043 |
| cudaGraphLaunch API duration | 5.275 | 4.879 | 2.310 | 7.043 |
| sum of graph kernel durations (PDL-inflated, info only) | 19.344 | 19.431 | 19.228 | 19.722 |
| **excluding stalls** (n=49) | | | | |
| period | 15.218 | 15.345 | 15.141 | 15.703 |
| GPU busy (union) | 14.632 | 14.719 | 14.563 | 14.957 |
| GPU idle | 0.585 | 0.627 | 0.571 | 0.748 |
| graph window (first-last graph kernel) | 14.634 | 14.722 | 14.569 | 14.963 |
| idle inside graph window | 0.062 | 0.063 | 0.060 | 0.065 |
| graph end -> next graph start | 0.582 | 0.623 | 0.567 | 0.755 |
| launch lateness (cudaGraphLaunch start - prev graph end) | -4.564 | -4.254 | -6.458 | -1.730 |
| next first kernel start - its cudaGraphLaunch start | 5.271 | 4.877 | 2.310 | 7.043 |
| cudaGraphLaunch API duration | 5.275 | 4.879 | 2.310 | 7.043 |
| sum of graph kernel durations (PDL-inflated, info only) | 19.344 | 19.431 | 19.228 | 19.722 |

Host API calls per step (r5_ft, non-stall steps): mean count / mean total ms

| API | calls/step | ms/step |
|---|---|---|
| cuKernelGetName | 16.00 | 0.005 |
| cuMemRetainAllocationHandle | 105.00 | 0.200 |
| cudaEventCreateWithFlags | 4.00 | 0.009 |
| cudaEventDestroy | 4.00 | 0.005 |
| cudaEventQuery | 26.73 | 0.059 |
| cudaEventRecordWithFlags | 20.00 | 0.043 |
| cudaEventSynchronize | 6.00 | 0.026 |
| cudaGraphLaunch | 1.00 | 4.748 |
| cudaLaunchKernel | 16.00 | 0.121 |
| cudaMemcpyAsync | 22.00 | 0.178 |
| cudaStreamIsCapturing | 38.00 | 0.022 |
| cudaStreamWaitEvent | 2.00 | 0.006 |

graph kernels per step: 2070

## Sweep-attributed graph kernel time, rank 0 (ms per step, 50 launches)

| kernel (sweep-attributed ms/step) | off | ft | ft-off |
|---|---|---|---|
| moeA2ADispatchKernel | 0.832 | 0.286 | -0.546 |
| moeA2ASanitizeExpertIdsKernel | 0.092 | 0.000 | -0.092 |
| routingIndicesClusterKernel | 0.282 | 0.366 | +0.084 |
| nvjet_sm100_ootst_128x128_256x_15... | 0.272 | 0.330 | +0.058 |
| quantize_with_block_size | 1.075 | 1.111 | +0.037 |
| silu_and_mul_kernel | 0.172 | 0.147 | -0.024 |
| moeA2ACombineKernel | 2.024 | 2.039 | +0.015 |
| moeOutputMemsetKernel | 0.337 | 0.325 | -0.012 |
| nvjet_sm100_ootst_128x128_256x_9... | 1.068 | 1.078 | +0.010 |
| nvjet_sm100_tst_64x64_64x16_2x_8... | 0.310 | 0.321 | +0.010 |
| kernel_cutlass_kernel_tensorrt_13... | 0.586 | 0.595 | +0.010 |
| moeA2APrepareDispatchKernel | 0.076 | 0.069 | -0.007 |
| device_kernel | 0.235 | 0.241 | +0.006 |
| expandInputRowsFromSourceSlotsKernel | 0.000 | 0.006 | +0.006 |
| TOTAL graph attributed | 14.116 | 13.669 | -0.447 |

Per-name attribution is a decomposition of busy time, not a per-kernel cost; dispatch includes peer waiting.
