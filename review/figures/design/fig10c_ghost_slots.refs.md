# fig10c_ghost_slots: refs

Refs are path:line at the draft's code of record 7e8dfb37df (verified by reading the lines). The frozen build 1117f7e0e7 differs only in
peer_loss_recovery.py (+1 line after 1088, about +19 after 1558) and mnnvl_memory.py (remap +5 lines); no change to what is drawn.
Abbreviations as in DESIGN_DOC_v5final_draft.md (PLR, LB, ES, N1S, KRN, GSM, MLB, MNV, COM, AUT, CFG, CLI, PYE).

| Element | Claim drawn | Ref | Status |
|---|---|---|---|
| Example counts | 8 + 2 (no EPLB), 9 + 1 (288 slots) for DeepSeek-R1 DEP32, 4 GPUs/node | LB:1208-1213: ghost = ceil(E / (N - K)) - s; E=256, N=32, K=4 -> ceil(256/28)=10; s=8 -> 2, s=9 -> 1 | verified (arithmetic) |
| One VA range always fully mapped | CUDA graphs capture its address | GSM.h:38-41 | verified |
| Ghost piece = chunks wholly inside empty slots | cut at slot boundaries | GSM.h:39-40 | verified |
| Shared boundary chunk has own memory | other chunks mapped to own memory at allocate() | GSM.h:43-46; GSM.cpp:437-546 (per draft) | verified (header); chunk count in drawing is illustrative |
| Healthy: scratch 1/2/4/8 chunks shared by every range; stray access reads scratch | GSM.h:40-41; GSM.cpp:40 (kMaxScratchChunks = 8), 120-135, 200-216 | verified |
| Routing -1 at ghost slots | LB:640-655 | verified |
| back() maps device memory in place of scratch, pending for the owner | GSM.cpp:626-672; GSM.h:42-49 | verified |
| Install keeps, abandoned plan returns to scratch | GSM.cpp:674-711 (releasePending, keepPending); PLR:1289-1302; MLB:1353 (per draft) | verified (MLB:1353 not re-read) |
| Memory source: free memory first, 256 MiB margin, shortfall from KV quota; fail before Ready | PLR:94, 1323-1376 | verified |
| "before Ready" heading | backing happens in _restore_placements before Ready | PLR:1017, 1036-1038 | verified |
