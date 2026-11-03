# DESIGN_DOC.md claim audit 1 (behavioural claims vs code of record)

Doc: `.wideep-ft-runs/claude/mvp/DESIGN_DOC.md` (510 lines, Oct 21 version). Code: `/tmp/claude-161238/wt-v5` at
`fa884734ad` (branch v5). Paths below are relative to `tensorrt_llm/` unless they start with `cpp/` or `docker/`.
Abbreviations: PLR = `_torch/pyexecutor/peer_loss_recovery.py`, AUTH = `executor/recovery/authority.py`,
CFG = `executor/recovery/config.py`, LB = `_torch/moe/fused_moe/moe_load_balancer.py`,
NVL = `_torch/moe/fused_moe/communication/nvlink_one_sided.py`, A2A = `cpp/tensorrt_llm/kernels/moe/communication/moeAlltoAllKernels.cu`.
Pure measurements are skipped. Statements about the evaluation setup (job ids, run protocol) are skipped too.

| ID | Doc line | Claim | Verdict | Code evidence | Note |
|---|---|---|---|---|---|
| C01 | 8-9, 81 | Survivors continue on the same CUDA graphs captured at startup; ranks, slots, addresses and kernels stay fixed | MATCH | PLR:589-590 (`CUDA graphs are captured only before serving`); `_recover` PLR:891-1091 never recaptures; install only rewrites tables (PLR:1651-1686) | |
| C02 | 24 | A lost rank does not rejoin | MATCH | PLR:129-150 (`_register` is `functools.cache`, once per process); no rejoin path | |
| C03 | 44-45 | A new request strictly pinned to a lost ADP rank gets a non-retryable error | MATCH | `_torch/pyexecutor/py_executor.py:6925-6946` (message has no retryable tag); `scheduler/adp_router.py:248` | |
| C04 | 44-45 | Requests on failed ranks rerun or end with a retryable error | PARTIAL | PLR:1557-1590 | Cancelled ones end cancelled; one interrupted before reruns never (fails) |
| C05 | 49, 52 | Requires an MoE model on NVLinkOneSided all-to-all | MATCH | `communication/execution_state.py:91` (`WideEP recovery requires NVLinkOneSided`); PLR:438-439 | |
| C06 | 51 | Attention DP and EP, expert TP = 1 | MATCH | PLR:394 (`expert TP 1`); `_torch/distributed/communicator.py:948-953` (tp==moe_ep==world, ADP on) | |
| C07 | 53 | Offline and online EPLB supported | MATCH | PLR:437 (`an EPLB load balancer` required); PLR:930-934 online quiesce | No-EPLB mode also runs the static load balancer |
| C08 | 54 | Overlap scheduler supported | MATCH | PLR:306-308 (two uncommitted steps), PLR:681-716 | |
| C09 | 55 | MTP supported; other spec-dec methods untested | MATCH | PLR:99-110, 442-452 (warning, not refusal) | |
| C10 | 56 | KV Cache Manager V2 required | MATCH | PLR:401-404 | DSACacheManagerV2 also accepted |
| C11 | 57-58 | ULFM Open MPI with `--with-ft ulfm`; mpi4py 4.1.2 | MATCH | communicator.py:955-965 (mpi4py>=4, ULFM, THREAD_MULTIPLE); `docker/common/install_mpi4py.sh:13` | Code checks mpi4py>=4, not 4.1.2 exactly |
| C12 | 57, 339-341 | ob1/TCP transport; ULFM recovers only over TCP | NOT FOUND | none in code; only the launch docs (`docs/source/features/wide-ep-fault-tolerance.md:56-62`) | Launch requirement, not enforced |
| C13 | 60-61 | FT refuses to start with guided decoding, beam search, disagg KV transfer, KV connectors, PEFT, PP or CP | MATCH | PLR:396-410 (`no guided decoding`, `no beam search`, `no KV cache transceiver`, `no KV connector`, `no PEFT`); communicator.py:948-953 (pp/cp==1) | Also refuses SWA scratch reuse, sleep/wakeup, KV pool rebalance, LM-head TP |
| C14 | 65-66 | One rank or one node (host up) is covered | PARTIAL | PLR:113-127 `failure_domains`; CFG:56-61 (`failure_domain` default `node`) | Node coverage only under the default `failure_domain=node` |
| C15 | 67-68 | Further losses after a recovery, or during one before install, while survivors' slots hold every expert | MATCH | AUTH:436-439 (RECOVERED, failed > failed_ranks), AUTH:458-459 replan; PLR:1213-1235, 567-573 | |
| C16 | 69 | A frozen process (SIGSTOP) recovers in ~5 s; only a rank that hangs *while alive* needs hang detection | MISMATCH | PLR:1173-1192: "Without hang detection, recovery does not use the MPI runtime's job control, and a stopped rank holds the agreement until it ends" | SIGSTOP needs hang detection too; doc l.275 and the l.17 table say so, l.69 contradicts them |
| C17 | 69-70 | Hang detection is off by default, ~35-40 s | MATCH | CFG:65-68 (`hang_detection=False`); CFG:107-118 (max(2,5)+30+5 = 40 s) | |
| C18 | 71, 201 | Rank 0, its node, a host death, NVLink fabric or >1 node end the job | PARTIAL | Rank 0: AUTH:422-424, PLR:113-127. >1 node: AUTH:476-479 and PLR:1226-1235 admit any union of admitted domains the slots cover | ">1 node" holds only because DeepSeek-R1 at 8+2/9+1 slots cannot cover it; the code does not forbid it |
| C19 | 76, 85-86, 105-107 | Ghost slots: reserved addresses at the end of the stack, no memory until a restore; the table never points at them while healthy | MATCH | LB:476-489, 633-658 (`placement[:, -ghost:] = -1`); LB:1286-1297 | |
| C20 | 86-87, 112 | Restore memory comes from free GPU memory first, then the KV cache | MATCH | PLR:1314-1343 (`empty_cache`, then `resize_kv_cache_quota(-short)`) | |
| C21 | 87, 111-112 | Lost experts are copied from host copies, never from the dead GPU | MATCH | LB:1640-1666 (`restore_placement` from host weights); LB:1591-1620 (ring: host to survivor workspace) | |
| C22 | 88-90 | A failed GPU is fenced before the survivors change membership | PARTIAL | AUTH:451-457, 461-482 (fence before Propose); PLR:1173-1192 | Without hang detection the ULFM shrink (the host communicator's membership) runs before the fence; only the mask and placement install comes after it |
| C23 | 91 | The authority is one process outside MPI that decides membership and placement | PARTIAL | AUTH:1-14, 857-905 | It decides membership and checks that the placement digests match (AUTH:589-591); survivors compute the placement (PLR:1295-1312) |
| C24 | 92 | Every survivor applies the same change, or none does | PARTIAL | AUTH:582-596 (Install only after all digests match); PLR:1651-1686 docstring: an install error "leaves the state mixed and ends the job" | Holds only in the sense that a partial install ends the job (fails closed) |
| C25 | 94-96 | Tables are rewritten while no graph runs; the next launch picks them up | MATCH | PLR:921 `_join_device_work`; LB:1650-1653 (`in_iter` check); PLR:1679-1685 | |
| C26 | 120-122 | With replicas, every node keeps every expert in host memory (/dev/shm) | MATCH | LB:1159-1164, 1266 (`host_copies=... or FT`), LB:1176-1182 | |
| C27 | 121-122 | With replicas, CPU threads write the copy through GDRCopy, the EPLB path | PARTIAL | LB:1286-1290: static routing (offline EPLB) or memory not host-accessible: `set_use_gpu_memcpy(True)` "Only GPU copies load ghost slots" | GDRCopy only for online EPLB with host-accessible ghost memory; offline EPLB uses GPU copies |
| C28 | 122 | FT off with offline EPLB keeps no host copies | MATCH | LB:1266 (`host_copies=updates_enabled or peer_loss_recovery_enabled()`) | |
| C29 | 123-124 | No replicas: each node keeps the next node's experts | MATCH | LB:1094-1102 `ring_copy_nodes`; LB:1176-1182, 1248-1251 | |
| C30 | 126 | Spare slots 8+2 (no EPLB) / 9+1 (EPLB) for DeepSeek-R1 on 32 GPUs | MATCH | LB:1245-1253: `ceil(E/(N-largest_loss)) - slots` = ceil(256/28)-8 = 2, -9 = 1 | Assumes 4 GPUs per node and the default node domain |
| C31 | 126-127 | At startup FT checks that the survivors of any one-node loss can hold every expert | PARTIAL | PLR:456-464 (`covers_loss_of` per domain) | Per `failure_domain`; with `rank` only one-rank losses are checked |
| C32 | 128-130 | The GEMM tuning table is keyed by total slot count | NOT FOUND | not traced within budget | |
| C33 | 130-131, 342-343 | FT's extra slots put the routing sort in a larger (512) tier | NOT FOUND | tiers up to 512 exist (`cpp/.../moe/trtllmGen/routing/RoutingCustomPolicy.cuh:417`); which one 320 maps to was not traced | |
| C34 | 132-133 | FT turns on the load-balancer path that FT off without EPLB never runs | MATCH | PLR:437 (EPLB load balancer required) | |
| C35 | 133 | With online EPLB, ghost memory holds GDRCopy host mappings | MATCH | LB:1353-1365 (`ghost_slot_memory_host_mappings`) | |
| C36 | 133-134, 306-307 | If GPU memory can't be made CPU-writable, online EPLB falls back to memory-backed empty slots, kept *only* as that fallback | PARTIAL | LB:1314-1318 (default); CFG:47-54 | `extra_slots: empty` can also be chosen in config for any mode, so it is not only a fallback |
| C37 | 138-141 | Same plan on every survivor; lost expert to a ghost slot on the survivor with the fewest restored so far, lowest rank first; then slots of experts with a second copy | MATCH | LB:1001-1045 `restore_lost_experts` (lexsort: empty first, then most replicas, fewest restored, rank, slot); LB:1432-1493 | Among replicas it takes the expert with the *most* copies |
| C38 | 149-151 | Upstream: a sender's region stride follows the step's largest sender | MATCH | A2A:790-793 (`rows_per_source = ... : max_tokens_per_rank`) | |
| C39 | 153-155 | Counters and flags per sender are fixed; payload rows are not, so a late payload can land anywhere in the payload region | MATCH | NVL:784-809 `source_write_ranges` | The comment at A2A:790 ("keeps a lost rank's late writes in its own rows") overstates this; source_write_ranges agrees with the doc |
| C40 | 158-159 | Expert GEMMs read hidden states in place from the mailbox; only the routing row is compacted | PARTIAL | `fused_moe_cute_dsl.py:2190-2193` (NVFP4 single tensor, not SM107), `fused_moe_cutlass.py:958-962` (unquantized, no LoRA only); `moe_scheduler.py:1093-1101` compacts otherwise | Holds for DeepSeek-R1 NVFP4 on CuteDSL only |
| C41 | 160-162 | Rank 0 ends every failed rank through the MPI runtime before Propose, with or without hang detection; recovery aborts if one cannot be ended | MATCH | AUTH:451-457, 461-482; AUTH:895-902 (`fence_departed=True` in `main`); PLR:1107-1171 | The class default is `fence_departed=False` (tests); the deployed `main` sets it |
| C42 | 161-162 | Survivors resume only after agreement and install | MATCH | PLR:1033-1041 (wait Commit before resume) | |
| C43 | 172-173 | Detection: a native watcher plus Open MPI's detector; an abort flag stops kernels waiting on the dead rank | MATCH | PLR:614-641 (`watch_peer_failures(..., abort.data_ptr())`) | |
| C44 | 174-175 | Survivors stop, discard uncommitted steps, agree on who failed; the authority fences and proposes a new generation | MATCH | PLR:914-928; AUTH:451-504 | In code the agreement (shrink) precedes the discard |
| C45 | 176-177 | Plan: after Propose, each survivor computes the placement, maps ghost memory, replies with the plan hash | PARTIAL | PLR:973-978: placement + mapping run *before* Propose ("mapped while the authority gathers the survivors' Stopped"); ring staging copies at PLR:990-1000 | Order differs from the doc: mapping overlaps the fence |
| C46 | 178-179 | Install: once hashes match, survivors copy in lost experts, update routing table and mask; authority commits | PARTIAL | AUTH:589-596; PLR:1013-1033; LB:1660-1665 (`slots_loaded=self.copy_nodes is not None`) | Without EPLB (ring copies) the experts are copied in Plan, before Ready (PLR:990-1000, LB:1591-1620), not in Install |
| C47 | 180 | Discarded steps replay on the new placement, same graphs | MATCH | PLR:921-929 (`discard_interrupted_steps`), PLR:1052-1072 | |
| C48 | 184, 326 | A loss before install restarts the round with every lost rank; a loss during install fails closed | MATCH | PLR:1006-1011 (`_shrink_again`, replan); AUTH:458-459 vs. AUTH:463-464 (Stopped in INSTALLING violates and aborts) | |
| C49 | 185-186, 289-290 | Surviving requests continue from exactly their tokens; lost rank's requests rerun if they returned no output, else retryable error | PARTIAL | PLR:1503-1517 (survivors' requests interrupted twice without progress are failed); PLR:1566-1576 (lost-rank request interrupted before fails even with no output; cancelled ends cancelled) | Second-interruption quarantine is not mentioned |
| C50 | 187-189 | /health 200 for healthy/degraded/unprotected; 503 for recovering, terminal *or unknown* | MISMATCH | `serve/fault_tolerance_status.py:37-40` (`SERVING_STATES` includes `unknown`); `serve/openai_server.py:1710` | `unknown` returns 200 |
| C51 | 188 | `recovering` lasts at most `authority_timeout_s` | MATCH | AUTH:443, 274-275; AUTH:902 (`config.authority_timeout_s`) | |
| C52 | 198-199 | Launch adds one recovery-authority process | MATCH | AUTH:1-14, 857-905 | |
| C53 | 200, 354-358 | MPI_Finalize under ULFM is bounded at 30 s; shutdown 43-45 s vs ~17 s | MISMATCH | `llmapi/mpi_session.py:138-149`: 3 s with `OMPI_MCA_mpi_ft_detector` on (commit a02f262be7), 30 s otherwise | Doc describes the earlier build; the FT launch uses the detector, so the bound is 3 s |
| C54 | 202 | Disagg, guided decoding and per-request penalties are not supported with FT | MATCH | PLR:396-399, 476-497 | Multimodal under overlap and logits post-processors are rejected too (not in doc) |
| C55 | 208-210 | FT enabled by `TLLM_FAULT_TOLERANCE_MODE` + YAML; extra_slots ghost/empty, failure_domain node, recovery_timeout_s 30, authority_timeout_s 20, optional hang detection | PARTIAL | CFG:43-98; `_utils.py:515-540` | Recovery also needs `TLLM_FAULT_TOLERANCE_AUTHORITY` (+ key); MODE alone only turns the mask on (PLR:371-377). `extra_slots` default is None (resolved per mode) |
| C56 | 211 | The authority runs as `executor/recovery/authority_server.py` | MISMATCH | no such file; `executor/recovery/authority.py:857-909` (`python -m tensorrt_llm.executor.recovery.authority`) | Wrong file name |
| C57 | 262-263, 463-466 | FT dispatch adds one block per source rank that copies its routing into compact lists; FT skips a sanitize kernel | MATCH | A2A:600-605, 1522-1530; NVL:1177-1179, 1272-1273 (no sanitize when bound) | |
| C58 | 324-326 | Overlap-safe replay lets surviving requests continue from their tokens | MATCH | PLR:681-716 (commit only if abort record clear), PLR:921 | |
| C59 | 336-338 | CFT must be off or FT fails at startup | MATCH | NVL:749-750 (`Bound execution requires the fence path; CFT is enabled`) | |
| C60 | 344-345 | In-place mailbox reads exist for CuteDSL and Cutlass; DeepGEMM and TRTLLM-Gen compact | PARTIAL | see C40 | CuteDSL only NVFP4 single-tensor (not SM107); Cutlass only unquantized without LoRA |
| C61 | 346 | A second loss that is a GPU error ends the job | PARTIAL | PLR:513-522 (`exits_alone` while `survives_own_loss`), PLR:547-553 | True during a recovery; after one, a rank with a CUDA-fatal error exits alone and survivors recover if their slots cover it |
| C62 | 346-347 | Loss after commit and before resume, or during install, ends the job | MATCH | PLR:891-899 docstring; PLR:1013-1091 (no interrupt handling after Install) | |
| C63 | 347 | Losing a node together with the node holding its host copies ends the job | MATCH | LB:1474-1484 (`uncopied` experts make recovery_placements raise) | |
| C64 | 347-348 | Rank 0 is never recoverable by design | MATCH | AUTH:422-424 (`recoverable = range(1, N)`); PLR:113-127 | |
| C65 | 348 | A lost commit message stalls ~30 s, then ends the job | MATCH | PLR:965-966, 1040 (`wait(Commit, deadline)`, deadline = Stopped + `recovery_timeout_s`) | ≤30 s from Stopped, not from the lost Commit |
| C66 | 348-349 | Losing the authority leaves the deployment serving but unprotected | MATCH | `serve/fault_tolerance_status.py:26-51` (no answer gives `unprotected`, 200) | |
| C67 | 350-351 | MTP: a narrow-window loss can leak rejected-draft KV pages once per recovery | NOT FOUND | no code marker; PY executor discard at `py_executor.py:8683-8700` reverts draft KV | Known gap per doc; not checkable statically |
| C68 | 352-353 | FT-off changes: mpi4py 4.1.2; failed OpenAI streams end with an error event | MATCH | `docker/common/install_mpi4py.sh:13`; `serve/openai_server.py:264-293` (unconditional) | |
| C69 | 359-360 | Rank 0 logs a traceback from mpi4py's futures manager thread on every recovery; should be quiet | MISMATCH | `llmapi/mpi_session.py:114-135, 889-890` (`_quiet_departed_peer_manager_failures` logs one line under FT) | Already fixed in code; doc is stale |
| C70 | 377-379 | Survivors log just before sending each recovery message (next step: send first, log after) | MISMATCH | PLR:301-303, 547-553, 1093-1098 (logs deferred to after recovery) | Already done in code; doc is stale |
| C71 | 412-413 | The final build finishes prefaulting host copies before reporting ready | MATCH | LB:1330-1335 (`populate_thread.join()` in `finalize_model`) | Doc l.483-484 cites first-minute stalls from compacting host copies; that is not a code property |
| C72 | 499-500 | Ghost slots take 0 GiB while healthy | MATCH | LB:1286-1297, CFG:47-54 | Only with `extra_slots=ghost` |

## Counts

MATCH 46, PARTIAL 16, MISMATCH 6, NOT FOUND 4 (72 claims).
