# fig10a_architecture: refs

Refs are path:line at the draft's code of record 7e8dfb37df (verified by reading the lines). The frozen build 1117f7e0e7 differs only in
peer_loss_recovery.py (+1 line after 1088, about +19 after 1558) and mnnvl_memory.py (remap +5 lines); no change to what is drawn.
Abbreviations as in DESIGN_DOC_v5final_draft.md (PLR, LB, ES, N1S, KRN, GSM, MLB, MNV, COM, AUT, CFG, CLI, PYE).

| Element | Claim drawn | Ref | Status |
|---|---|---|---|
| Authority box | separate process; ZMQ + HMAC to workers | AUT:30-33 (import zmq, ZeroMqQueue), AUT:636-643, 744-754; CLI:44-71 (HMAC key, hex, >= 16 bytes) | verified |
| Authority box | "no GPU" | authority.py imports no torch / CUDA module (checked by grep of its imports only) | UNVERIFIED beyond imports |
| Authority box | admits unions of registered domains, never rank 0 | AUT:440-458 (recoverable = range(1, world)), AUT:529-532 | verified |
| Authority box | has rank 0 fence each failed rank before Propose | AUT:489-499 (Fence to routes[0]), AUT:501-521, AUT:860 (fence_departed=True) | verified |
| Authority box | Install only when every placement digest matches | AUT:557-572 | verified |
| Authority box | Commit when all Installed | AUT:574-598 | verified |
| Authority box | Abort at deadline | AUT:282-285, 473 | verified |
| Arrows authority <-> rank | Register / Stopped / Ready / Installed up; Propose / Install / Commit / Abort down; Fence to rank 0 | PLR:1008, 1037-1038, 1092; AUT:437-438 (Registered), 495-498, 527, 572, 598 | verified |
| PeerFailureWatch | native thread, no GIL, polls a receive no peer matches every peer_failure_poll_s | COM:850-885; CFG:67-72 (0.02 s per draft) | verified (0.02 default from draft, not re-read) |
| Arrow "store 1" | watch stores 1 to the abort word (after setting failed) | COM:853-856; PLR:636-643 | verified |
| Arrow "revoke" | revoke host communicators on failure | PLR:645-658 | verified |
| PeerLossRecovery | wraps the event loop, runs _recover on HostCollectiveInterrupted | PLR:544-585 | verified |
| Host collectives | pml ob1, btl self,tcp (launcher default; startup refuses other pml) | trtllm-llmapi-launch:201-205; COM:960-963 | verified |
| Host collectives | Shrink -> survivor communicator, installed at Install | COM:1009-1033; PLR:1052 (install_survivors) | verified |
| Abort record | int32[2] (requested, status) in pinned, device-visible host memory | ES:33-35, 47-49 | verified |
| Device rank mask | 64-bit words; kernels skip inactive ranks; ENABLE_RANK_MASK template | ES:25-27, 50; KRN:220-248, 342, 530, 545 | verified |
| EPLB tables | incl. expert_to_slot; static one-slot-per-expert dispatch looks up a table instead of route() | moe_scheduler.py:266-278; KRN:342 (USE_EXPERT_TO_SLOT) | verified |
| Expert slots | home + appended ghost slots, one VMM range always mapped | LB:1208-1223; GSM.h:38-41 | verified |
| Blue fan "edited in place, nothing recaptured" | recovery writes abort record, mask, EPLB tables, slots; graphs not recaptured | ES:35-39 (edited in place), ES:143-192; PLR:1748-1783; draft Invariant 3 (PLR:607-609) | verified |
| Kernels band | NVLinkOneSided on the fence path; dispatch/combine write peers' rows | N1S:740-745 (fence path required); MNV:938-1012 | verified |
| Failed rank box | ended process or, with hang detection, a frozen rank fenced by peers; rank 0 ends it through PMIx before Propose | PLR:1198-1218; AUT:489-521; CFG:27-34 | verified |
| Failed rank note | late NVLink writes may land in survivors' pages | MNV:938-945; PLR:1378-1386 | verified |
| Arrow "ULFM reports" | ULFM fails the watch's receive on a peer loss | COM:853-856 | verified |
| Host copies box | with replicas every node keeps every expert; without, ring of next node's home experts; restore never reads lost GPU | LB:44-54 (message), LB:1055-1063, 1236-1242; draft Section 6 | verified (ring direction: ring_copy_nodes docstring says "the node before its own"; equivalently each node keeps the next node's home experts) |
| Arrow "restore" | host copies -> slots | LB:1580-1606; PLR:1459-1482 | verified |
