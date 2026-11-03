# DESIGN_DOC (and feature doc) vs code: reconciled audit, 2026-10-21 19:10 PT

Code of record: v5 (`/tmp/claude-161238/wt-v5`, fa884734ad). Three independent audits, full tables:
`doc_audit_1.md` (front to back, 72 claims: 46 match, 16 partial, 6 mismatch, 4 not found),
`doc_audit_2.md` (back to front, 64 claims: 42 / 17 / 3 / 2), `doc_audit_3.md` (safety first + the feature doc
`docs/source/features/wide-ep-fault-tolerance.md`, 96 claims: 74 / 16 / 6 / 0). Below: findings reconciled across the
three; "n/3" = how many audits reported it. No audit contradicted another on any finding (they differ only in what each
covered), so no fourth read was needed; single-audit findings are marked and were spot-checked where noted.

## MISMATCH (doc wrong; fix the doc unless the code is wrong)
| # | Claim (doc line) | Code | n/3 | Action |
|---|---|---|---|---|
| M1 | `/health` `unknown` returns 503 (DD 187-188) | `serve/fault_tolerance_status.py:37-40`: `unknown` is in SERVING_STATES (200) | 3/3 | doc: 200 (matches the feature doc) |
| M2 | authority is `executor/recovery/authority_server.py` (DD 211) | file absent; `executor/recovery/authority.py`, run with `-m` | 3/3 | doc path |
| M3 | MPI_Finalize bounded at 30 s; FT shutdown 43-45 s (DD 200, 354-358; feature doc H26) | 3 s with the detector on (`mpi_session.py:138-149`, a02f262be7) | 2/3 + 1 partial | doc: state the open fix (DECISIONS 18:00: bound is a band-aid, diagnosis pending); no number until the fix lands |
| M4 | a frozen (SIGSTOP) process recovers in ~5 s without hang detection (DD 69, 172) | without hang detection a stopped rank holds the agreement until it ends (`peer_loss_recovery.py:1173-1192`); DD 17 and 275 already say so | 1/3 + 2 partial | doc: frozen process covered only with hang detection on (finals config) |
| M5 | rank 0 logs an mpi4py futures-thread traceback (DD 359-360) | one line now (`mpi_session.py:114-135`, 957ac92c5f) | 2/3 | doc: drop |
| M6 | Next step "send first, log after" (DD 377-379) | done (recovery logs deferred, ee76f03e74) | 1/3 + 1 partial | doc: drop from next steps |
| M7 | offline-EPLB ghost slots are written by CPU threads through GDRCopy (DD 120-122) | static routing uses GPU memcpy (`moe_load_balancer.py:1286-1290`); GDRCopy only for online EPLB's host-accessible ghosts | 1/3 + 2 partial | doc: split by mode |
| M8 | feature doc: rank 0 fences failed ranks only with hang_detection (FD 94-101, 109-110); `_shrink` docstring says the same | fence-always (`authority.py:893`, fa884734ad) | 1/3 (feature doc only audited by #3) | v5 commit: feature doc + docstring |
| M9 | kernel comment: fixed slots keep a source's late writes "in its own rows" (`moeAlltoAllKernels.cu:790`) | payload offsets are packed per call; R1's MTP vs NVFP4 layers alias (C32) | 1/3 + DECISIONS 19:05 | fixed by design C (C32) |

## PARTIAL (true only on one path / under a default / for one failure type)
| # | Claim | Qualification | n/3 |
|---|---|---|---|
| P1 | experts read the mailbox in place (DD 155-156, 344) | only CuteDSL NVFP4 single-tensor (not SM107) and Cutlass unquantized without LoRA; other paths get a compact copy | 3/3 |
| P2 | a second loss that is a GPU error ends the job (DD 346) | only during a recovery; after one, such a rank exits alone and is recovered if the slots cover it (`peer_loss_recovery.py:557`) | 3/3 |
| P3 | survivors continue from exact tokens; lost requests rerun if no output (DD 179, 185-186, 289-290) | a request interrupted twice with no committed step is failed, on survivors too; a lost rank's request an earlier loss interrupted fails; cancelled stays cancelled (`peer_loss_recovery.py:1508-1524`) | 2/3 |
| P4 | memory mapped / experts copied "after" agreement (DD 178, flow) | ghost memory is mapped before Propose; without EPLB experts are copied in Plan (after Propose, before Ready; copy plan in the hash) | 3/3 |
| P5 | the fence precedes membership change (DD 88) | the ULFM shrink runs first; the fence precedes the routing/mask install | 2/3 |
| P6 | "every survivor applies the same change, or none does" (DD 91) | holds because a mixed install ends the job (at the authority deadline) | 2/3 |
| P7 | FT is enabled by MODE + YAML (DD 208) | recovery also needs TLLM_FAULT_TOLERANCE_AUTHORITY and its key; MODE alone gives detection/mask only | 2/3 |
| P8 | FT runs ob1/TCP with mpi4py 4.1.2 (DD 57-58, 198-199, 339, 352) | not selected or checked in code (launch docs only); code checks mpi4py>=4 and `--with-ft ulfm` | 2/3 |
| P9 | more than one node lost ends the job (DD 68) | not enforced: any union of admitted domains the slots cover is accepted; holds for R1's slot counts (single audit, to verify) | 1/3 |
| P10 | memory-backed empty slots are only a fallback | `extra_slots: empty` is a config choice | 1/3 |
| P11 | the auth key must be 64 hex chars | any hex key >= 16 bytes accepted | 1/3 |
| P12 | SIGSTOP recovery time: ~5 s (DD) vs 10.6-36.7 s (feature doc) | both from earlier builds; re-measure on v5 (C7) | 1/3 |
| P13 | refused features list | also refused: logits post-processors; multimodal with the overlap scheduler | 1/3 |

## NOT FOUND (no code to check; keep as stated assumptions or drop)
- the ob1/TCP transport (launch configuration only), the GEMM tuning key using the total slot count, the routing tier
  320 slots map to, the MTP KV-page leak marker.

## Follow-ups
- Doc edits for M1-M7, P1-P13 (design doc), M8 (feature doc + docstring, in the v5 commit series).
- P9: verify and either enforce in code or narrow the claim.
EOF
echo ok