WideEP Fault Tolerance MVP Design Doc
Andrew Rusli US | Created:  Oct 19, 2026 | Update: Oct 20, 2026 | Status: Prototyping
Overview & Motivation
Today, losing one rank of a wide expert-parallel (WideEP) deployment ends the whole job, and serving resumes only after the job restarts (minutes). This design allows the surviving ranks to regroup (~1s) and continue serving on the same CUDA graphs captured at startup. The next big phase after this is to support full recovery from N-k back to N ranks while the N-k survivors serve requests. 
See RISE: WideEP Fault Tolerance In TRTLLM for more details on the broader project plan.
This is motivated because failures are more common at large scale and restarts are costly. In an 18-week trace of NVIDIA's Dynamo serving cluster running ~150 model replicas (mostly 8 GPUs or fewer per deployment), the cluster saw restarts occurring 46 times a day with median recovery of 3.5 min. 77% of these restarts were caused by engine process deaths or hangs, leaving the GPU intact (Dynamo Paper). WideEP deployments span 32 to 64+ GPUs, making them even more susceptible to hardware and process failures.
See TRTLLM WARP: Warm Artifact Reuse Pipeline for more details on speeding up (re)startup.

Goals & Scope
Main goals:
Performance: FT mode while healthy (no failures) should cost as little throughput loss as possible.
Throughput after failover should be reasonable as well, but since we aim to support recovery back to N later, this need not be hyper optimized.
Recovery time: Rank/node loss should recover in seconds by using the same CUDA graphs. 
Correctness: Requests on survivors continue; requests on the failed ranks rerun or end with a retryable error.

This design currently supports:
MoE models (Tested with DeepSeek-R1, other models are the next priority)
Aggregated serving (Disagg is also next priority)
Attention DP and EP, with expert TP=1
NVLinkOneSided MoE all-to-all
No EPLB, offline EPLB, and online EPLB
Overlap scheduler
MTP (Other spec dec methods are currently untested)
KV Cache Manager V2
Open MPI with ULFM (Open MPI ≥ 5.0, launched with mpirun --with-ft ulfm; mpi4py ≥ 4)

Failures covered:
One process-level rank failure e.g. SIGKILL
One process-level node failure (machine/host stays up)
Further process-level rank failures after a recovery (or during one before new placement applied) while the remaining expert slots on survivors can hold every expert
A frozen rank e.g. SIGSTOP (detected in ~5s)
Anything else (rank 0, rank 0's node, host death, NVLink fabric, >1 node) fails closed.

Design Overview
FT uses the same CUDA graphs, even when a rank is lost. This means the number of ranks and expert slots, buffer addresses, and kernels stay fixed. A failure changes small tables the graphs read on every launch, like the live-rank mask and EPLB's expert-to-slot table (Figure 1). Three ideas make this work:

Ghost slots: spare expert slots (reserved virtual addresses the CUDA graphs already know) with no memory behind them while healthy. On a failure, memory is mapped behind them, from free GPU memory first and taken from the KV cache if needed. Lost experts are copied in from host copies.
Per-sender mailboxes: each GPU writes into its own fixed region of every peer's all-to-all buffer, which is read in place. A dead GPU's late writes stay in its own region, and that region is never read again. 
Recovery authority: one process outside MPI that decides who is alive and the new expert placement. Every survivor applies the same change, or none does.

In Figure 1, a failure detector on each rank reports to the recovery authority. The authority tells every rank's recovery hook what to change. The hook rewrites the tables and fills ghost slots from host copies while no graph is running, and the next graph launch picks up the change.

Figure 1.

Ghost Slots
On each GPU, a layer's experts are stored stacked in one array, and each position in that stack is an expert slot. EPLB's table tells the kernels which slot holds which expert. FT adds a few spare positions at the end of the stack whose addresses are reserved but have no memory behind them: these are ghost slots. Since the table never points at them while healthy, they cost no memory and no extra GEMM work.

When a rank dies, each survivor that receives one of its experts maps memory behind a ghost slot and copies the expert in from a host copy (Figure 2). The memory comes from free GPU memory first, and from the KV cache only if needed. The copy never reads from the dead GPU.

Figure 2.
Where the host copies live:
With online EPLB: every node already keeps every expert in host memory, so the copy stays on the node. CPU threads write it through GDRCopy, the same path EPLB uses for its updates.
Without online EPLB: each node keeps the next node's experts, so every expert has a copy on a different node. 

Each GPU gets a few spare slots (e.g. 8 + 2, or 9 + 1 with online EPLB, for DeepSeek-R1 on 32 GPUs). At startup, FT checks that the survivors of any one-node loss can still hold every expert. Kernel tuning uses only the loaded slots, so FT picks the same kernels as FT off. If GPU memory can't be made CPU-writable, online EPLB falls back to spare slots that are backed by memory and left empty.

Expert Placement
Every survivor computes the same plan from the same inputs. Each lost expert goes to a ghost slot on the survivor that has received the fewest so far, lowest rank first, which spreads the extra work evenly (Figure 3). With online EPLB, if ghost slots run out, the remaining experts take slots of experts that still have a second copy, and the next EPLB update rebalances as usual.

Figure 3.

[WIP] Late writes from a dead rank
A dead rank may still have NVLink writes in flight. In a packed receive buffer, each sender's position depends on what the others sent that step, so a late write can land on a live sender's tokens in the next step. FT instead gives each sender a fixed mailbox in every receiver's buffer (Figure 4):

A late write from a dead sender can only land in its own mailbox, which is never read again.
Reusing that mailbox, when scaling back to N ranks, will need a guarantee that the dead GPU's writes have stopped. If we can get this guarantee we can also reconsider this mailbox design.
The expert GEMMs read each token's hidden state straight from its mailbox, so FT adds no extra copy of the hidden states; only each token's small routing row (~64 bytes) is compacted.
The recovery authority kills failed ranks before changing membership, so the new membership starts only once no failed process can still write.

Figure 4.
Note: I am working on a token-major layout which is much less changes to the all-to-all kernels as it will already be compacted like on the left of Figure 4, but just a different layout. This may or may not improve performance of the fault tolerance mode. 

Recovery Flow
Detect: a native watcher plus Open MPI's failure detector notice the loss: a killed rank in ~0.1 s, a frozen one in ~5 s. An abort flag stops kernels waiting on the dead rank.
Agree: survivors stop, discard uncommitted steps and agree on who failed. The authority fences the failed ranks and proposes a new generation.
Plan: each survivor computes the new placement, maps memory for its ghost slots, and replies with a hash of the plan.
Install: once all hashes match, survivors copy in the lost experts, update the routing table and live-rank mask, and the authority commits.
Resume: discarded steps replay on the new placement, and serving continues on the same CUDA graphs.

Other notes:
A loss before install restarts the round with every lost rank; a loss during install fails closed. 
Surviving requests continue from exactly the tokens they had. A lost rank's requests rerun if they had returned no output yet; otherwise they end with a retryable error. 
/health reports healthy, degraded or recovering so orchestrators don't restart a job that is only recovering (recovering, 503) and can see reduced capacity (degraded).
