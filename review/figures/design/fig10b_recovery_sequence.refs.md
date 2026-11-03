# fig10b_recovery_sequence: refs

Refs are path:line at the draft's code of record 7e8dfb37df (verified by reading the lines). The frozen build 1117f7e0e7 differs only in
peer_loss_recovery.py (+1 line after 1088, about +19 after 1558) and mnnvl_memory.py (remap +5 lines); no change to what is drawn.
Abbreviations as in DESIGN_DOC_v5final_draft.md (PLR, LB, ES, N1S, KRN, GSM, MLB, MNV, COM, AUT, CFG, CLI, PYE).

| Row / element | Claim drawn | Ref | Status |
|---|---|---|---|
| phase column | rank-0 log line "WideEP recovery gen=G phase=NAME t=..." per step | PLR:1125-1130; calls at 951 (detect), 1009 (stopped), 1029 (proposed), 1039 (ready), 1042 (install), 1093 (installed), 1096 (committed), 1105 (resumed), 741 (first_step) | verified |
| Abort | watch stores abort word, revokes; _recover requests abort, stops watch, revokes; kernels return cancelled | COM:850-865; PLR:636-662, 957-959; KRN:265-315 | verified |
| Agree | ULFM Shrink, same failed set; admitted-domain / coverage check | PLR:962-967, 1198-1218, 1239-1261; COM:1009-1033 | verified |
| Stop | join device work, discard uncommitted steps; online: quiesce EPLB updates | PLR:969-987 | verified |
| Stopped | send Stopped | PLR:1008-1013 | verified |
| Ghost backing (overlap) | rank 0 placement broadcast (online), plan, map memory from free/KV, while authority gathers Stopped | PLR:1014-1017, 1304-1376 | verified ("overlaps" = code order: done after Stopped, before waiting for Propose) |
| Authority note | once all Stopped, rank 0 fences failed ranks, then Propose | AUT:489-499, 501-527 | verified |
| Proposed | check proposal | PLR:1018-1029 | verified |
| Remap | same-address workspace remap | PLR:1030, 1378-1424 | verified |
| Ring staging (conditional) | only when copy_nodes (no-replica placements) | PLR:1031-1035, 1459-1482; LB:1213-1216 | verified |
| Ready / Install | Ready(digest); Install once all digests match | PLR:1036-1042; AUT:557-572 | verified |
| Install | survivor comm; load slots + routing (or mask only); mark failed; rearm; install mask | PLR:1052-1056, 1748-1783; LB:1580-1606; ES:143-192 | verified |
| Online EPLB (conditional) | reset update schedule, 20 warm-up iterations | PLR:98, 1065-1068 | verified |
| Outcome exchange + Installed | allgather of served ids, iter counter, interrupted, progressed; quarantine; Installed | PLR:1069-1093 (frozen build adds _record_started_owners at 1089) | verified |
| Commit | authority Commit when all Installed | AUT:574-598; PLR:1094-1096 | verified |
| Return KV | return remap-lent KV as far as free | PLR:1097, 1439-1457 | verified |
| Resume | requeue, deliver, re-run/fail, re-admit, restart watch | PLR:1099-1108 | verified |
| first_step | first committed step logged on rank 0 | PLR:739-741, 1106 | verified |
| re-plan window bracket | further loss Stopped..Install -> RecoveryInterrupted, shrink again, release ghost backing + KV, plan again | PLR:988-1050 (loop), 1043-1046, 1263-1302; AUT:534-555 | verified (window drawn Stopped..Ready; it extends through the wait for Install, PLR:1040) |
| after Install | further loss ends the job | draft Section 8 / 12; code: no RecoveryInterrupted handler after the loop (PLR:1050-1108) | verified by structure |
| Deadlines note | D1, D2, authority deadline shorter | PLR:963-966, 1012-1013; AUT:473, 282-285; CFG (authority_timeout_s < recovery_timeout_s validator) | verified |
| Color classes | "critical path" = runs sequentially on every recovery; "overlap", "online only", "ring only" as above | code order above | interpretation, not a measurement; per-step times come from fig 6 |
