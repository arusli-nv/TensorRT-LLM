# Finals: deviations from the standard recipe (for the doc and the deck appendix)

Baseline recipe: perf-sanity aggregated GB200 R1 FP4 v2 (`deepseek_r1_fp4_v2_2_nodes_grace_blackwell.yaml`,
`r1_fp4_v2_dep8_mtp1_1k1k`), run at DEP32 (8 nodes x 4 GB200), DeepSeek-R1 NVFP4, MTP 1, attention DP, overlap
scheduler, CUDA graphs, fixed 1k/1k. Source of the settings: `mvp/harness/configs/std-*.yaml` (the six files share one
body and differ only in `moe_config.load_balancer`) and `mvp/harness/serve.sh`.

Every item applies to every arm (FT off, FT on, main, floor, ULFM) unless marked ASYMMETRIC.

| Setting | Finals | Recipe | Why |
|---|---|---|---|
| max_seq_len | 2304 | unset (163840) | ISL+OSL+draft = 2049 fits; startup memory scales with it, and DEP32 does not start without it (CLAIMS 82) |
| stream_interval | 10 | 1 | at 512/GPU one frontend streams 16k requests; 10 is a curated value (examples/configs/database 10-20, wide_ep slurm 20). Makes ITL chunk gaps, so ITL is not reported |
| num_postprocess_workers | 4 | 0 | the DEP32 gen configs' value; with 0 the frontend, not the engine, limits throughput |
| print_iter_log | true | false | the step-time readout reads it (same logging cost in every arm) |
| kv_cache_config.host_cache_size | 16 GiB per rank | auto | fixed so FT's host copies cannot shrink the auto-sized host tier in one arm |
| TLLM_FMHA_LIBS | -cute_dsl_mla | default | main's default MLA decode kernel (cute_dsl_mla) faults out of bounds at large batch here (serve.sh; FREEZE_CHECKLIST H33, its 8/32 cost still to quantify) |
| TRTLLM_KVCACHE_POOL_USE_FABRIC_MEMORY | 0 | default | fabric KV pool off |
| prompts | random token ids, 1k | random (benchmark_serving random) | as the recipe; real-text routing is out of scope |
| autotuner tactics | one table per mode, tuned once on the frozen build, pinned (FT_COST_E2E_SEED; PIN_CHECK every case) | tuned per launch | removes tactic-choice noise between arms and groups |
| burn-in | WIDEEP_BURNIN 32:6 + warm-each-load | none | identical in every arm; runs past the first online-EPLB placement update |
| launcher / transport | FT arms: ULFM Open MPI, ob1 over TCP for host collectives (hang detection on) | mpirun (UCX) | ASYMMETRIC: part of FT's cost. The FT-off `*-ulfm` arm (n=2) separates launcher/transport from FT code |
| no-EPLB FT arm | static load balancer with ghost slots (FT needs the LB to restore lost experts) | no load balancer | ASYMMETRIC by design: the static-LB floor arm (FT off, same placement, main's routing kernel) separates main's LB indirection from FT |
| kill rows | KF_GRAPH_SIZES 136,148 (extra CUDA graph sizes for the post-loss batch) | n/a | recovery rows only; healthy rows use the default graph sizes |
| diagnostics | WIDEEP_NO_SAMPLERS=1, WIDEEP_DMON=1 (clock/throttle log only) | none | same in every arm |

Metrics (pre-registered in FINALS_PLAN.md): decode trim-mean step and full-mean step (rank 0's
`prev_device_step_time`), stall ms/step, whole-run client output tok/s/GPU (benchmark_serving `output_throughput`,
ramp and drain included), steady iteration-log tok/s/GPU, TPOT, TTFT.

Rank-0 step time as the per-step cost: only rank 0 prints iteration stats. In DEP32 every decode step runs a
dispatch and a combine all-to-all in every MoE layer, and a rank cannot finish a layer before every peer's tokens
arrive, so rank 0's step spans the slowest rank's MoE work. What rank 0 alone sees is its local tail (LM head,
sampling, MTP draft), the same code in every arm. Earlier nsys on ranks 0 and 5 (CLAIMS 66, 126/127) showed the same
per-step pattern on both ranks.
