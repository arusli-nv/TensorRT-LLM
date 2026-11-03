# Doc audit 3 — DESIGN_DOC.md (DD) and docs/source/features/wide-ep-fault-tolerance.md (FD) vs code (wt-v5, v5 @ fa884734ad)

Scope: behaviour claims only; measured numbers (throughput, recovery seconds) are not code-checkable and are only listed where the two docs disagree.
Paths are relative to /tmp/claude-161238/wt-v5. "plr" = tensorrt_llm/_torch/pyexecutor/peer_loss_recovery.py, "auth" = tensorrt_llm/executor/recovery/authority.py, "mlb" = tensorrt_llm/_torch/moe/fused_moe/moe_load_balancer.py.

Totals: MATCH 74, PARTIAL 16, MISMATCH 6, NOT FOUND 0 (96 rows; a few rows merge duplicate claims).

## Safety: fencing, late writes, agreement, fail-closed

| id | doc | line | claim | verdict | code evidence | note |
|---|---|---|---|---|---|---|
| H21 | DD | 160-162 | Rank 0 ends every failed rank via the MPI runtime before the plan, with or without hang detection; recovery aborts if one cannot be ended | MATCH | auth:504-533 (`_on_fenced` aborts on error), auth:893 `fence_departed=True` | HEAD commit fa884734ad |
| H19 | FD | 94-101 | Only with hang_detection does rank 0 kill failed ranks before the plan; "Without hang_detection, recovery uses no MPI runtime job control" | MISMATCH | auth:893 `fence_departed=True` | Authority fence is now unconditional; only the survivors' pre-shrink kill is gated (plr:1185-1192). Disagrees with DD 160-162. plr:1179-1181 `_shrink` docstring is stale the same way |
| H20 | FD | 109-110 | Exit list: "with hang_detection ... and rank 0 each failed rank before a recovery plan" | MISMATCH | auth:504, 893 | Rank 0's fence no longer needs hang_detection |
| A12 | FD | 99 | Aborts if the runtime refuses or does not answer within 2 s | MATCH | plr:1137-1170 (`RUNTIME_FENCE_CALL_S`), config.py:31, auth:517-521 | |
| M18 | DD | 153-155 | Per-sender counters/flags fixed; payload rows not fixed (`source_write_ranges`) | MATCH | communication/nvlink_one_sided.py:784-810 | |
| M22/H33 | DD | 172-173 | Native watcher + abort flag stops kernels waiting on the dead rank | MATCH | communicator.py:1058-1063; plr:614-620, 919 `request_abort`; moeAlltoAllKernels.cu:271-311 | |
| A1 | DD | 184 | Loss before install restarts the round with every lost rank; loss during install fails closed | MATCH | plr:893-896, 1017-1023 (`RecoveryInterrupted` -> `_shrink_again`); auth:492 `_replan`, auth:496-497 violation in INSTALLING | |
| A16 | FD | 135-139 | Several losses together / one during recovery (before install) / after recovery recovered if slots cover | MATCH | auth:470-498; plr:1222-1233 `_check_recoverable` | |
| A17 | FD | 147-148 | Loss after install, before commit, fails closed | MATCH | auth:496-497 (Stopped outside STOPPING/PREPARING -> `_Violation` -> abort) | Consistent with DD 184 |
| A6 | DD/FD | DD 346; FD 227-228 | Loss after commit, before resume, ends the job | MATCH | plr:1056-1065 (allgather `departed=None`), plr:544-557 (exception escapes `run`) | |
| A5 | DD/FD | DD 348; FD 228-229 | Lost Commit: survivors wait up to recovery timeout, then exit | MATCH | plr:1048 `client.wait(Commit, deadline)`; deadline = recovery_timeout_s (plr:976) | |
| A3 | DD | 177-179 | Plan hashes must match before Install; authority commits after all Installed | MATCH | auth:590-605 (digest mismatch -> violation), auth:607-631 | |
| A2 | DD | 175-177 | Order: agree -> authority fences + proposes -> Plan: survivors map ghost memory | PARTIAL | plr:990-994 (`_restore_placements` maps memory before `_wait_for_proposal`) | Mapping happens while the authority gathers Stopped/fences, before Propose |
| A15 | DD | 174-175 | Survivors stop, discard uncommitted steps, agree on who failed | MATCH | plr:918-940 (abort, revoke, shrink, join device work, discard) | Agreement precedes discard |
| A14 | FD | 262-264 | A running step finishes first; bounded by one step | MATCH | plr:930 `_join_device_work` | |
| A4 | DD | 180 | Discarded steps replay on the new placement | MATCH | py_executor.py:8678-8703 (steps undone, requests scheduled again unchanged) | |
| A19 | FD | 26-27 | Online EPLB quiesces, survivors adopt rank 0's placement and restart rotation | MATCH | plr:946-948 `quiesce`, ~992 `_rank0_placements`, 1043 `reset_update_schedule` | |
| A7 | DD | 346 | "a second loss that is a GPU error ... end[s] the job" | PARTIAL | plr:513-520 `exits_alone`, plr:549/557 `survives_own_loss` | After a completed recovery, a GPU-error loss is recovered if slots cover; it ends the job only during a recovery. FD 124-125 lists CUDA-context errors as covered |
| A8 | DD/FD | DD 347; FD 31-32, 222-223 | Losing a node with its copy node is unrecoverable | MATCH | mlb:1094-1102 `ring_copy_nodes`; plr:1222-1233 | |
| M17 | DD/FD | DD 71, 201; FD 22, 126, 143 | Rank 0 and its node never recoverable | MATCH | plr:113-126 | |
| R19 | FD | 116-118 | Loss before serving starts ends job (`MPI_Abort`) | PARTIAL | plr:180-184 `propagate_hard_kill`; hang_detector.py:67-82 | Prefers MPI_Abort but can fall back to another kill path |

## Hang detection, second failures, exits

| id | doc | line | claim | verdict | code evidence | note |
|---|---|---|---|---|---|---|
| H1 | FD | 89 | Confirmed hung rank kills itself from its evidence thread | MATCH | hang_evidence.py:250-256 SIGKILL | |
| H2 | FD | 89-91 | 1 s later another rank has PMIx kill it | MATCH | hang_evidence.py:257-270; config.py:29; auth:358-363 | Sent to up to 2 live helpers |
| H3 | FD | 91-92 | `--fence-command` 5 s later, `{host}`/`{pid}`, no shell | MATCH | auth:818-855; config.py:34 | |
| H4 | FD | 103-112 | SIGKILL only in the listed places; stall/rank-0 hang exits 75 | MATCH | py_executor.py:1508; hang_evidence.py:248, 256; config.py:37 | Pre-existing HangDetector self-kill (hang_detector.py:121) not listed |
| H5 | FD | 105-106 | Unusable error (sticky CUDA) -> rank kills itself | MATCH | py_executor.py:1497-1508 | |
| H6 | FD | 185-188 | Fenced only if every fresh peer waits on it for hang_confirm_s and it finished no step | MATCH | hang_monitor.py check / `_waited_on_by_every_peer` | |
| H7 | FD | 188 | Rank whose own device waits on another is not blamed | MATCH | hang_monitor.py:207-208 | Only while its heartbeat is fresh |
| H8 | FD | 188-192 | All-rank stall with no culprit -> terminal, exit 75 | MATCH | hang_monitor.py all_stalled; auth:309-310, 367-371 | |
| H9 | FD | 192-193 | Idle matches neither rule; nothing decided with < half heartbeats | MATCH | hang_monitor.py `_has_view` | |
| H10 | FD | 182-185 | Native (no GIL, iteration) + Python heartbeat (host wait, device waits) | MATCH | heartbeatSender.cpp:169; hang_evidence.py:272-281, 320-328 | |
| H11 | FD | 199-200 | Without native heartbeat, Python heartbeat alone | MATCH | hang_evidence.py:141-160 | |
| H16 | FD | 212-213 | Two ranks hung at once are not confirmed | PARTIAL | hang_monitor.py:203-213 | Two *stopped* ranks (stale heartbeats) each count as waited on by every fresh peer and can be confirmed; claim holds for alive hung ranks |
| H22 | DD | 68-70, 172-173 | Frozen process detected in ~5 s; implies no hang detection needed | PARTIAL | plr:1185-1192 | Without hang_detection the shrink waits on the stopped process (fence comes after shrink); DD 17 and 275 say hang detection is needed; DD 69 reads otherwise |
| H23 | FD | 202-204 | In-band detector declares a frozen process failed; "the recovery then fences it and continues" | PARTIAL | plr:1185-1192 | Only with hang_detection; FD 100-101 itself says a stopped process holds the agreement |
| H24 | DD vs FD | DD 17, 284; FD 178 | SIGSTOP to serving ~5 s (DD) vs 10.6/15.9/36.7 s (FD) | PARTIAL | n/a (measurements) | Docs disagree; different builds/rules |
| H17/H18 | FD | 151-153, 253-254 | Executor hang detector fixed 300 s; startup needs shrink wait + 2x recovery_timeout_s below it (= 3x + hang resolution) | MATCH | hang_detector.py:444; plr:386, 414-419; config.py:116-124 | Both FD phrasings are equivalent |
| H12 | FD | 48-50 | Authority exits if it cannot bind UDP (hang detection) | MATCH | auth:756-775, 900 | |
| H13 | FD | 47-48 | Without `--endpoint` binds localhost | MATCH | auth:861 | |
| H14 | FD | 45-47 | Authenticated not encrypted; routing-id takeover | MATCH | auth:777-790 | |
| H15 | FD | 74 | Key is 64 hex characters | PARTIAL | client.py:54-68 | Any hex key of >=16 bytes accepted; 64 hex is the example |
| H30 | FD | 215-217 | Nothing restarts the authority; register only at startup | MATCH | client.py:137-155; serve/fault_tolerance_status.py:27-29 | |
| H31 | DD/FD | DD 348-349; FD 156-157 | Authority gone -> keep serving, `unprotected` | MATCH | fault_tolerance_status.py:47-50 | |
| H34 | FD | 156 | Authority stop during recovery aborts it | MATCH | auth:717-719 | |

## Clients, /health, startup

| id | doc | line | claim | verdict | code evidence | note |
|---|---|---|---|---|---|---|
| R1 | DD/FD | DD 185-186, 289-290; FD 6-7, 265-267 | Lost rank's requests rerun if nothing delivered, else retryable error | MATCH | plr:1560-1593; 847-865 | |
| R2 | FD | 7-9, 266-269 | Request interrupted twice without progress is quarantined | MATCH | plr:1506-1524, 845 | DD silent |
| R3 | FD | 269-270 | Lost rank's steps count only once reported with responses | MATCH | plr:841-845 | |
| R4 | FD | 272 | Non-streaming: 503, Retry-After: 1, `peer_loss_retryable` | MATCH | openai_server.py:1207-1211; executor/utils.py:178 | |
| R5 | FD/DD | FD 273-275; DD 353 | Streaming endpoints end with SSE error event + [DONE] | MATCH | openai_server.py:264-294, 2419, 2929, 3381 | |
| R6 | FD | 276 | Unanswered cancellation ends cancelled | MATCH | plr:1566-1568 | |
| R7 | DD/FD | DD 45; FD 163-165 | New request pinned to lost ADP rank: "Attention DP rank N is not serving", HTTP 400, non-retryable | PARTIAL | py_executor.py:6927-6943; openai_server.py:1198-1204 | Message matches; 400 only via default error path, not traced end to end |
| R8 | FD | 162-163 | Re-run may land on any survivor even with relax=False | MATCH | plr:1572 | |
| R9 | FD | 290-295 | healthy/degraded/unprotected/unknown -> 200; recovering/terminal -> 503 | MATCH | serve/fault_tolerance_status.py:28-40 | |
| R10 | DD | 187-188 | `/health` 503 with `recovering`, `terminal` **or `unknown`** | MISMATCH | fault_tolerance_status.py:40 `SERVING_STATES` includes `unknown` | `unknown` is 200; DD disagrees with FD and code |
| R11 | FD | 295-300 | Body fields; `unprotected` only `state` | MATCH | fault_tolerance_status.py:41, 54-71 | |
| R12 | FD | 300-301 | Engine failure -> 503 | MATCH | openai_server.py ~1734 | |
| R13 | FD | 72 | MODE=1 alone: rank 0 warns, `/health` unprotected | MATCH | plr:369-375; openai_server.py:1716-1718 | |
| R14 | FD | 75 | Authority refuses differing config digest; logs digest | MATCH | auth:428-431, 886 | |
| R15 | FD | 281-283 | Resume log line text and fields | MATCH | plr:1084-1091 | |
| R16 | DD/FD | DD 60-61, 202; FD 248-254 | Startup refusals list | MATCH | plr:380-431; communicator.py:946-953 | DD list is a consistent subset |
| R18 | DD/FD | DD 202; FD 256-258 | Per-request rejections (penalties, logits post-processors, multimodal+overlap) | MATCH | plr:476-500 | |
| R20/A11 | DD/FD | DD 57; FD 35 | mpi4py >=4 (`Shrink`) and ULFM launch checked; container 4.1.2 | MATCH | communicator.py:955-962; docker/common/install_mpi4py.sh:13 | Checks launch env, not `ompi_info` |
| R21 | DD/FD | DD 55; FD 240-244 | All spec-dec methods accepted, drafts discarded, untested warn | MATCH | plr:357-362, 432-444 | |
| R22 | FD | 14-15 | Context blocks reusable only once a step is accepted | PARTIAL | py_executor.py:6077 | Only a comment found; interrupted-step behaviour not traced |
| M15 | DD/FD | DD 126-127; FD 114-117, 126-133 | Startup checks one failure domain; s*(N-K) >= experts; 8+2 / 9+1 = 320 | MATCH | plr:446-454; mlb:1246-1249 | |
| M16 | FD | 114-115 | Rank 0 logs recoverable ranks/nodes | MATCH | plr:466-470 | |
| A9/A10 | DD/FD | DD 208-210; FD 79-88 | Settings and defaults (node, 30, 20, hang off, 2/30/1/5 s, heartbeat <=2 s, authority < recovery) | MATCH | executor/recovery/config.py:45-102 | |
| H32 | DD | 211 | Authority is `executor/recovery/authority_server.py` | MISMATCH | no such file; auth:857 `main` (`python -m tensorrt_llm.executor.recovery.authority`) | FD 43 is correct |

## Ghost slots, host copies, placement, kernels

| id | doc | line | claim | verdict | code evidence | note |
|---|---|---|---|---|---|---|
| M1 | DD/FD | DD 105-108; FD 80 | Ghost slots: no memory, no routing while healthy | MATCH | mlb:652-658, 692-699 | |
| M2 | DD/FD | DD 86-87, 111-112; FD 80 | Restore memory: free GPU memory first, then KV cache | MATCH | plr:1317-1332 | |
| M3 | FD | 80 | KV stays smaller; recovery fails if it can't shrink | MATCH | plr:1328-1330, 1040-1041 | |
| M4 | FD | 79-80 | `extra_slots` default ghost; empty with online EPLB w/o host access (incl. NUMA) | MATCH | mlb:101-123, 1314-1318 | |
| M5 | FD | 80 | Explicit ghost on online w/o host access: no warning, GPU copies | MATCH | mlb:1286-1289 | |
| M6 | DD | 121-122 | With replicas (online **or offline**) CPU threads write via GDRCopy | PARTIAL | mlb:1287-1289 `if self.is_static_routing() or not ...: self.set_use_gpu_memcpy(True)` | Offline (static) EPLB with ghost slots uses GPU memcpy; conflicts with FD 80 |
| M7 | DD | 122 | FT off with offline EPLB keeps no host copies | MATCH | mlb:1263 | |
| M8 | DD/FD | DD 123-124; FD 27-30 | Ring copies: next node, last keeps first's, single node own | MATCH | mlb:1094-1102 | |
| M9 | FD | 30-31 | Restore staged "in chunks of half the workspace" | PARTIAL | mlb:1048-1062 `staging_halves` | Halves of the row bytes the all-to-alls leave free (split at the middle of the largest free range), not half the workspace |
| M10 | FD | 34 | Startup fails with sizes if copies don't fit | MATCH | mlb:40-66 | |
| M11 | DD | 412-413 | Host-copy prefault finishes before ready | MATCH | mlb:73-93, 414, 1336 | |
| M12 | DD | 138-139 | Fewest restored so far, lowest rank first | MATCH | mlb:1037-1038 | |
| M13 | DD/FD | DD 139-141; FD 25-26 | Out of ghosts: take slots of experts with a second copy | MATCH | mlb:1032 | |
| M19 | DD | 158-159 | GEMMs read hidden states in place, "FT adds no extra copy" | PARTIAL | fused_moe_cute_dsl.py:2190-2193; fused_moe_cutlass.py:958-962; moe_scheduler.py:1097-1107 | Only for CuteDSL NVFP4 / Cutlass unquantized no-LoRA; others compact (DD 344 and FD 16-18 say so) |
| M20 | DD | 344 | In-place for CuteDSL and Cutlass; DeepGEMM/TRTLLM-Gen compact | PARTIAL | as M19; impl_blocks.py:89-96 | Omits the Cutlass unquantized/no-LoRA limit FD states |
| M21 | DD | 159 | ~64-byte routing row compacted | MATCH | nvlink_one_sided.py:1198-1237; moeAlltoAllKernels.cu:531 | |
| M23 | DD | 336-337 | CFT must be off or FT fails at startup | MATCH | nvlink_one_sided.py:750 | Error does not name the env var |
| M24 | DD | 466-467 | FT skips a sanitize kernel | MATCH | nvlink_one_sided.py:1272 | |
| M25 | DD | 464 | Dispatch adds one block per source rank | MATCH | moeAlltoAllKernels.cu:602-605 | |
| M26 | DD | 82, 179 | Graphs read a live-rank mask rewritten in place | MATCH | nvlink_one_sided.py:671, 1189-1199 | |
| M27 | DD/FD | DD 133-134; FD 80 | No CPU-writable GPU memory -> online EPLB uses backed empty slots | MATCH | mlb:1315-1318 | |
| M14 | FD | 26-27 | (see A19) | MATCH | plr:992, 1043 | |

## Shutdown and logging

| id | doc | line | claim | verdict | code evidence | note |
|---|---|---|---|---|---|---|
| H25 | DD | 200, 354-358 | `MPI_Finalize` bounded at 30 s; shutdown 43-45 s vs 17 s; grace > 45 s | MISMATCH | llmapi/mpi_session.py `_DETECTOR_FINALIZE_GRACE_S = 3.0` when `OMPI_MCA_mpi_ft_detector` is true (commit a02f262be7) | Under the documented launch (detector on) the bound is 3 s; 43-45 s predates v5 |
| H26 | FD | 158-159, 218-219 | Waits at most 30 s for `MPI_Finalize`; stop up to ~30 s longer | PARTIAL | same | True only without the detector |
| H27 | DD | 359-360 | Rank 0 logs a traceback from the mpi4py futures manager thread; should be quiet | MISMATCH | mpi_session.py:115-136 (one line, commit 957ac92c5f) | Fixed; FD 284-285 matches code |
| H28 | FD | 284-285 | `MPI_ERR_PROC_FAILED` line, then "a rank departed; only shutdown is accepted from now on" | MATCH | mpi_session.py:130, 1184 | |
| H29 | DD | 377-379 | Next step: send recovery messages first, log after | PARTIAL | plr:544-557, 1093-1098 (`_log` deferred during recovery, commit ee76f03e74) | Already done in v5; stale as a to-do |
| A21 | DD | 350; FD 232-233 | MTP/overlap: loss after a committed step can skip the rejected-draft KV update | MATCH (docs agree) | no code marker; known gap | Docs consistent with each other; not code-verifiable |

## Doc-vs-doc disagreements
1. Fencing without hang detection: DD 160-162 (always; matches code) vs FD 97-101, 109-110 (only with hang_detection; stale).
2. `/health` `unknown`: DD 188 says 503; FD 294 and code say 200.
3. Ghost-slot host writes with offline EPLB: DD 121-122 (GDRCopy) vs FD 80 / code (GPU copy for static routing).
4. Cutlass in-place reads: DD 344 unqualified vs FD 16-18 (unquantized, no LoRA).
5. Futures-manager traceback: DD 359 (traceback) vs FD 284 / code (one line).
6. Frozen-rank recovery: DD 17/284 (~5 s) vs FD 178 (10.6-36.7 s); DD 69 vs DD 275 on whether hang detection is needed.
7. Authority entry point: DD 211 `authority_server.py` vs FD 43 / code `authority.py`.
