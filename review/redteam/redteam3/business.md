# Red team lane: business / product (EM + PM view)

Inputs: mvp/DESIGN_DOC.md (Oct 20), v2:docs/source/features/wide-ep-fault-tolerance.md, redteam2/PR_DESCRIPTIONS_V2.md,
mvp/CLAIMS.md rows 117-136. Read-only review. Ranked by how likely an exec or a customer-facing PM is to hit it.

## Verdict
The engineering story is strong. The product story is not written yet. The material explains *how* FT works and
lists its limits in detail, but never says who should turn it on, what it buys in user terms, or what it costs in
user terms. The best value numbers are in CLAIMS and appear in no reader-facing document. The four documents
also give four different answers on cost and recovery time.

## Ranked gaps

1. **No value statement in user terms (the biggest gap).** The doc says "cold restart 8-9 min" but leaves out the
   two numbers a customer cares about most. Both are in CLAIMS #117/#119/#121:
   - A restart fails **every** in-flight request: 4,096 of 4,096 at 128/GPU. FT fails only the lost ranks'
     streamed requests: ~125 for a rank loss, ~500 for a node loss. No survivor request fails.
   - For a hung rank, the restart path takes **~14 min** (855 s: main's 300 s hang detector plus the relaunch).
     FT takes ~6 s.
   Suggested opening: "On DeepSeek-R1 at DEP32 (CLAIMS #117, #119, #121), losing one GPU's process today takes the
   deployment down for ~8 min (~14 min for a hang) and fails all ~4,096 in-flight requests. With FT the deployment
   serves again in about 1-3 s, and only the ~125 requests already streaming on the lost rank end with a retryable
   error."

2. **The numbers disagree across documents. An exec will see this first.**
   - Recovery: the design doc says 0.4-0.8 s (rank) and 1.0-1.6 s (node), n=1. The final kill matrix (#119) says
     1.25-1.79 s and 2.0-2.6 s for no EPLB; online EPLB (#121) is 0.9-1.3 s. The guide says "well under a second"
     measured from detection. The ask says ~1-3 s.
   - Healthy cost: the design doc says "within ±2%, noise". The guide says "lowers healthy throughput, most at low
     load", with no number. #123 measured a decisive 1.3-1.5% (no EPLB). #122 measured 3.7-9% (online EPLB, older
     build). #135/#136 measure 0.3-1.6% on the final build. Every PR says `<PERF_TBD>` and "nothing run yet".
   Fix: give one headline per mode, with its CLAIMS row and the same scope in every document. Show the range, for
   example "1-2% step time without EPLB; ~0.3-1.4% online EPLB (final build, n=2)". Say that it is above the 1%
   budget where that is true. Never print "±2% noise" next to a CI that excludes 0.

3. **Throughput after a failover is missing, and it is a large drop.** CLAIMS #120: after one rank is lost, the
   no-EPLB deployment serves at 0.63-0.74 of its pre-loss throughput, against ~0.97 expected from losing 1/32 of
   the GPUs. The cause is CUDA-graph padding: survivors go from 128 to 129-133 requests and round up to the next
   captured size, 192. The design doc only says it "need not be hyper optimized". A PM will ask: "after a failure,
   do I lose 3% of capacity or 30%?" Suggested wording: "After a rank loss, capacity drops to ~65-75% (#120) until
   restart. The cause is graph padding, not FT. Capture a batch size a few % above the healthy per-rank maximum;
   the fix is measured in <run>."

4. **The real value proposition is not stated: FT turns an unplanned outage into a planned one.** A lost rank never
   rejoins. The deployment stays `degraded` until someone restarts it, and that restart costs the same 8-9 min.
   Say so: "FT does not avoid the restart. It lets you choose when to take it: drain the replica at the load
   balancer, then restart, with no failed requests." Recovery back to N ranks is the next phase that removes this.
   It belongs in the overview with a target date.

5. **No break-even or "who should turn this on".** A 1% healthy cost is ~14 min of deployment capacity per day.
   One restart is ~8 min down plus 4,096 failed requests. So on raw throughput alone, FT pays off at roughly 1-2
   losses per deployment per day. The real gain is availability and failed requests, not throughput. Add a short
   "When to enable" section. For example: large DEP (32+ GPUs) aggregated deployments with SLOs on availability or
   error rate; not deployments behind a replica fleet that already absorbs a restart cheaply. Also add the expected
   failure rate per 32-GPU deployment. The Dynamo trace in [1] is mostly ≤8-GPU replicas, so scale it explicitly.

6. **The scope cuts the target customer, and the doc does not say so.** Most production WideEP deployments of R1 at
   this scale run disaggregated (Dynamo). FT refuses disaggregated serving and guided decoding (JSON mode, tool
   calls). It also **rejects repetition, presence and frequency penalties per request**, which OpenAI-API clients
   send routinely. That last limit is only in the guide's "Not supported" section. A customer will hit it as HTTP
   errors after turning FT on. Put it in the doc's scope list and the guide's first screen: "With FT on, requests
   that set frequency/presence/repetition penalties are rejected."

7. **The adoption path is unclear.** To use FT, a customer must:
   - build or obtain Open MPI with ULFM;
   - run `mpirun` over TCP only;
   - run and secure an extra authority process with a per-job HMAC key;
   - size `/dev/shm` at 368 GB per node (with replicas);
   - turn off CFT on new drivers;
   - set env vars instead of `LlmArgs`;
   - give Kubernetes more shutdown grace than its 30 s default (43-45 s measured).
   Open questions a customer will ask:
   - Does the release container ship ULFM? CI has no ULFM image, so probably not.
   - Is there a Dynamo, NIM or k8s operator path, or only bare `mpirun`/Slurm?
   - Is FT supported or experimental?
   Add an "Adoption checklist" table: requirement, ships today (yes/no), owner, ETA.

8. **The coverage share is not quantified.** These are not recoverable: rank 0, rank 0's node, host death,
   GPU/NVLink faults, more than one node, and silent errors. One GPU process lost at random is uncovered 1/32 of
   the time; a node's processes, 1/8 of the time. Combine this with the Dynamo split (77% are process deaths or
   hangs with the GPU intact) to state "FT covers ~X% of the failures we expect". Without it, "survives a GPU
   loss" overclaims. A real GPU or host death is the case a customer will picture.

9. **The related-work table shows "TBD" for this design's recovery time and cost.** Next to SGLang (6-11 s, up to
   4.4%), the TBD invites the competitive question. Fill it from the final rows (1-3 s; ~1-2%) and add a
   failed-requests column, where this design is clearly better.

10. **Maturity and CI gap: present but scattered.** The PRs say "not run yet". CI cannot run ULFM. Many paths are
    n=1 or unit-only (ghost slots, host loss, hang detection, online EPLB, MTP). #118 has 5 stale-fake unit
    failures. Add one maturity line to the doc's Status: "Experimental. Pre-merge CI cannot exercise recovery until
    a ULFM image exists (infra owner, ETA). The 32-GPU evidence comes from a separate harness."

11. **There is no roadmap section.** "Next" items are spread across the doc: N-k to N, disaggregated, other models,
    `LlmArgs`, a 384-slot routing tier, mailbox reuse, the MTP KV leak, the shutdown hang, a ULFM CI image, and
    other models (the project roadmap names GLM-5.3). Add a 5-line roadmap ordered by customer impact: CI image,
    `LlmArgs`, disaggregated, N-k to N, models. Give each an owner and a quarter.

12. **Hidden resource costs.** The doc states 368 GB of host `/dev/shm` per node, 13 GiB of BAR1 on rank 0 (online
    EPLB), the extra authority host, and the KV cache that restored experts take after a loss. It does not frame
    them as a deployment cost. Add a "Resource cost" row next to throughput: host RAM per node, GPU memory (0 GiB
    ghost / 1.3 GiB empty), and KV capacity lost per recovery.

## Questions the material does not answer
- What is the expected number of WideEP failures per day at 32/64 GPUs, and which share does FT cover?
- What is the end-to-end goodput over a day with N failures, FT on vs off (one chart)?
- Does it work under Dynamo / Kubernetes / NIM, and what is the owner and date for that path?
- When does FT become supported, and with which models (GLM, Kimi, Qwen MoE)?
- What does a client see, and what does it have to change? (Retry on `peer_loss_retryable`; drop penalties.)
- What happens at 64+ GPUs or with 8 GPUs per node? None of it has been measured.
