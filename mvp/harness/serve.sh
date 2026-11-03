#!/bin/bash
# One Slurm task per node: serve DeepSeek-R1 on all 32 GPUs with trtllm-serve under ULFM mpirun, and on
# node 0 run the benchmark driver against it. The case ends when the driver writes RUN_DIR/frontend.done.
# Usage: serve.sh DIST_DIR RUN_DIR ft|noft|noft-ulfm [bench.py args...]   (bench.py gets --ft for ft)
set -euo pipefail
dist=$1; run=$2; mode=$3; shift 3
H=$(dirname "$(realpath "$0")")
C=/lustre/fs1/portfolios/coreai/projects/coreai_comparch_trtllm/users/arusli/src/TensorRT-LLM/.wideep-ft-runs/claude; L=$H/launch
mkdir -p "$run"
# WIDEEP_DRY_RUN=1 (tools/dry_run.sh, CPU host): print what the case would launch and write a stub result, so a launch
# file's preflights, arms and knobs run end to end without GPUs.
if [[ -n ${WIDEEP_DRY_RUN:-} ]]; then
    knobs=$(compgen -e | grep -E '^(TLLM_TEST_|TLLM_FAULT_TOLERANCE_|WIDEEP_|MVP_OVERLAY$|BENCH_CONFIG$|TRTLLM_ENABLE_PDL$|TRTLLM_A2A_FT_ABLATE$)' \
        | grep -v '^WIDEEP_DRY_RUN$' | sort | while read -r v; do printf '%s=%s ' "$v" "${!v}"; done)
    echo "DRY_RUN_CASE node=${SLURM_NODEID:-0} run=${run##*/} mode=$mode dist=$dist args=[$*] $knobs" | tee -a "$run/timeline"
    [[ -f $H/configs/${BENCH_CONFIG:-dep32.yaml} ]] || { echo "DRY_RUN_FAIL missing config ${BENCH_CONFIG:-dep32.yaml}"; exit 1; }
    [[ -z ${MVP_OVERLAY:-} || -d $MVP_OVERLAY/tensorrt_llm ]] || { echo "DRY_RUN_FAIL missing overlay $MVP_OVERLAY"; exit 1; }
    compgen -G "$dist/*.whl" > /dev/null || { echo "DRY_RUN_FAIL no wheel in $dist"; exit 1; }
    [[ -z ${TOKENS_CONFIG:-} || -f $H/tokens/$TOKENS_CONFIG ]] || { echo "DRY_RUN_FAIL missing tokens config $TOKENS_CONFIG"; exit 1; }
    # "deterministic": a tokens reference's gate (EXACT-FINAL) lets the cases after it run too.
    echo '{"state": "DONE", "ready_s": 0, "dry_run": true, "deterministic": true, "rows": []}' > "$run/result.json"
    # An empty tactic cache, so a tuning pass's merge and PIN_CHECK run too.
    [[ -n ${TLLM_AUTOTUNER_CACHE_PATH:-} && ! -s $TLLM_AUTOTUNER_CACHE_PATH ]] && echo '{"metadata": {}, "shared": {}}' > "$TLLM_AUTOTUNER_CACHE_PATH"
    touch "$run/done" "$run/frontend.done" "$run/server_ready"
    for n in $(seq 0 $(( ${SLURM_NNODES:-1} - 1 ))); do echo "NODE_CLEAN $n $(date +%s.%N)" >> "$run/timeline"; done
    exit 0
fi
# WIDEEP_STOCK=1: a stock launch, as examples/wide_ep documents it: plain trtllm-serve under mpirun with the recipe's
# config and environment, memory bound with numactl -m 0,1 (wide_ep README). None of the harness's additions: no
# test hooks, recovery markers, rank wrapper or per-rank logs, samplers, iteration log, config overrides, info log
# level or pinned tactics. The node preparation before the launch (stale /dev/shm, cold page cache, checks) and
# the benchmark client (bench.py: main's benchmark_serving) are the same as in every case.
if [[ -n ${WIDEEP_STOCK:-} ]]; then
    for var in $(compgen -e | grep -E '^(TLLM_TEST_|TLLM_PROFILE_LOG_RANKS$|TLLM_EPLB_UPDATE_LOG$|TLLM_AUTOTUNER_CACHE_PATH$|TLLM_LOG_LEVEL$|WIDEEP_(HOST_CACHE_SIZE|EXTRA_GRAPH_BATCH_SIZES|PIN_BUSY|THREAD_SNAPSHOT|DMON|MARKERS)$)'); do
        unset "$var"
    done
    echo "STOCK $SLURM_NODEID" >> "$run/timeline"
fi
# Every process a case starts on a node (agent, PRTE daemon, ranks, and rank 0's frontend subshell, which
# trtllm-llmapi-launch runs outside MPI's process tree) inherits WIDEEP_CASE_TAG=<job>:<run>.
tag="${SLURM_JOB_ID:-0}:$run"
harness_tag="${SLURM_JOB_ID:-0}:harness:$run"  # the authority and the driver: outlive mpirun, ended below
tagged_pids() {  # PREFIX: this node's processes whose WIDEEP_CASE_TAG starts with PREFIX
    local p
    for p in /proc/[0-9]*; do
        grep -qz "^WIDEEP_CASE_TAG=$1" "$p/environ" 2>/dev/null && echo "${p#/proc/}"
    done
    return 0
}
sweep() {  # PREFIX LABEL: SIGKILL the tagged processes (again, until none is left: a dying parent can start more)
    local pids
    for _ in $(seq 10); do
        pids=$(tagged_pids "$1" | tr '\n' ' ')
        [[ -z "${pids// /}" ]] && return 0
        echo "$2 $SLURM_NODEID $(date +%s.%N) $(ps -o pid=,args= -p "$(echo $pids | tr ' ' ',')" 2>/dev/null | cut -c1-160 | tr '\n' ';')" >> "$run/timeline"
        kill -9 $pids 2>/dev/null || true
        sleep 1
    done
}
# Start only on a clean node: an earlier case of this job may have left processes behind; then wait for the GPUs
# (a killed process can hold GPU memory for a while; up to ~6 min after a case that profiled rank 0,
# STD-PROFILE2-8028706).
sweep "${SLURM_JOB_ID:-0}:" NODE_SWEEP_BEFORE
start=$(date +%s.%N)
for _ in $(seq 1200); do
    [[ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null)" ]] && break
    sleep 0.5
done
busy=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | wc -l)
echo "NODE_READY $SLURM_NODEID waited=$(awk "BEGIN{printf \"%.1f\", $(date +%s.%N) - $start}") busy=$busy" >> "$run/timeline"
if (( busy > 0 )); then
    pids=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | paste -sd,)
    echo "NODE_BUSY $SLURM_NODEID $(ps -o pid=,args= -p "$pids" 2>/dev/null | cut -c1-160 | tr '\n' ';')" | tee -a "$run/timeline"
    touch "$run/done" "$run/frontend.done"; exit 1
fi
bash "$C/install_wheel.sh" "$dist" "$run/install-$SLURM_NODEID.log"
# EPLB host copies left in /dev/shm by an earlier case: main names them moe_shared_l<L>_lr<r>_all and unlinks them
# only on a clean shutdown (a SIGTERM-ended main case leaves ~390 GB that only main's next start reuses); this build
# names them per job. Before the nodes' arrival barrier no rank of this case runs, so every such
# segment is stale: free it, then require /dev/shm nearly empty.
python3 - "$run/shm-$SLURM_NODEID.log" <<'PY' || { echo "SHM_NOT_CLEAN node $SLURM_NODEID"; touch "$run/done" "$run/frontend.done"; exit 1; }
import os, sys, time
log = open(sys.argv[1], "w")
freed = 0
for name in sorted(os.listdir("/dev/shm")):
    if name.startswith("moe_shared"):
        path = os.path.join("/dev/shm", name)
        freed += os.stat(path).st_size
        os.unlink(path)
        print(f"STALE_SHM {name}", file=log)
# A previous case's ranks still exiting keep unlinked segments mapped until they end.
for _ in range(180):
    st = os.statvfs("/dev/shm")
    used, total = (st.f_blocks - st.f_bfree) * st.f_frsize, st.f_blocks * st.f_frsize
    if used <= 0.05 * total:
        break
    time.sleep(1)
print(f"SHM_FREED {freed / 2**30:.1f} GiB; /dev/shm used {used / 2**30:.1f} of {total / 2**30:.1f} GiB", file=log)
sys.exit(0 if used <= 0.05 * total else 1)
PY
if [[ -n ${WIDEEP_POLLUTION_CHECKS:-} ]]; then  # opt-in until a GPU run shows no false positive
# Other state an earlier case can leave on this node, checked here for the same reason: each fails the case loudly.
# Each check logs its finding; it ends the case only with WIDEEP_POLLUTION_CHECKS=hard (after a run shows no false positive).
stale() {
    echo "$1 $SLURM_NODEID ${2:-}" | tee -a "$run/timeline"
    [[ ${WIDEEP_POLLUTION_CHECKS:-} == hard ]] || return 0
    touch "$run/done" "$run/frontend.done"; exit 1
}
# GPU memory a context of an exited process still holds (no compute process is listed any more).
for _ in $(seq 60); do
    gpu_used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | sort -n | tail -1)
    (( ${gpu_used:-0} < 1024 )) && break
    sleep 1
done
(( ${gpu_used:-0} < 1024 )) || stale GPU_MEM_NOT_FREE "max_used_mib=$gpu_used"
# GPU modes no case sets, for the record; a non-default compute mode or an MPS daemon changes every rank's context.
nvidia-smi --query-gpu=index,persistence_mode,compute_mode,mig.mode.current,clocks.applications.graphics,clocks.applications.memory,power.limit \
    --format=csv,noheader > "$run/gpustate-$SLURM_NODEID.csv" 2>&1
modes=$(nvidia-smi --query-gpu=compute_mode --format=csv,noheader 2>&1 | sort -u | paste -sd,)
[[ $modes == Default ]] || stale GPU_COMPUTE_MODE "$modes"
pgrep -x 'nvidia-cuda-mps-(control|server)' > /dev/null && stale MPS_RUNNING
# Harness and TRT-LLM processes the tag sweep cannot see (untagged), e.g. an earlier case's authority or driver.
left=$(pgrep -af 'tensorrt_llm\.executor\.recovery\.authority|/(bench|drop_page_cache|node_agent)\.py( |$)|(^|/)(prterun|prted|mpirun|trtllm-serve|trtllm-llmapi-launch)( |$)' || true)
# Logged only until a GPU run shows no false positive (this case's agent or PRTE daemon must not match yet).
[[ -z $left ]] || echo "NODE_STALE_PROCS_WARN $SLURM_NODEID $(cut -c1-160 <<< "$left" | tr '\n' ';')" >> "$run/timeline"
# A server still answering on the benchmark port (node 0), which the driver would measure instead.
[[ $SLURM_NODEID != 0 ]] || ! python3 -c "import socket,sys; socket.create_connection(('127.0.0.1', int(sys.argv[1])), 1).close()" \
    "${BENCH_PORT:-8000}" 2>/dev/null || stale PORT_IN_USE "${BENCH_PORT:-8000}"
# Trigger files (testhooks.py) that exist before the case: the victim would act at its first step.
for var in $(compgen -e | grep -E '^TLLM_TEST_(KILL_TRIGGER(_FOR)?|[A-Z_]+_IN_(DECODE|PREFILL))$'); do
    for path in $(tr ',' '\n' <<< "${!var}" | sed 's/^[0-9]*=//'); do
        [[ $path != /* || ! -e $path ]] || stale STALE_TRIGGER "$var=$path"
    done
done
# The weights cold in every case: a case that failed before server_ready never dropped them (WIDEEP_KEEP_PAGE_CACHE=1
# keeps the previous case's cache on purpose).
[[ -n "${WIDEEP_KEEP_PAGE_CACHE:-}" ]] \
    || python3 "$H/drop_page_cache.py" "${WIDEEP_MODEL:-${LLM_MODELS_ROOT:-/lustre/fs1/portfolios/coreai/projects/coreai_comparch_trtllm/common}/DeepSeek-R1-0528-FP4-V2}" \
        > "$run/dropcache-pre-$SLURM_NODEID.log" 2>&1
fi
# Every node enters the case together: a node still tearing down the previous case must not leave the others
# timing out on its launch agent.
mkdir -p "$run/arrived"; touch "$run/arrived/$SLURM_NODEID"
for _ in $(seq 900); do (( $(ls "$run/arrived" | wc -l) >= SLURM_NNODES )) && break; sleep 1; done
(( $(ls "$run/arrived" | wc -l) >= SLURM_NNODES )) || { echo "NODES_MISSING"; touch "$run/done" "$run/frontend.done"; exit 1; }
# One HTTP connection per in-flight request: up to 16k on node 0.
ulimit -n "$(ulimit -Hn)" 2>/dev/null || true; echo "NOFILE $SLURM_NODEID $(ulimit -n)" >> "$run/timeline"
# Benchmark-only additions to the installed package (never to TRT-LLM): recovery timing markers and the
# counters / timed-kill hooks of testhooks.py. Both refuse to apply if the code they hook has moved.
pkg=$(cd / && python3 -c "import importlib.util as u, os; print(os.path.dirname(u.find_spec('tensorrt_llm').origin))")
# Python-only changes under test on top of the wheel (MVP_OVERLAY: a directory holding tensorrt_llm/...). The wheel
# is installed once per container, so every case first restores the files an earlier case overlaid.
while read -r orig; do cp "$orig" "${orig%.wideep-orig}"; done < <(find "$pkg" -name '*.wideep-orig'; find "$(dirname "$(command -v trtllm-llmapi-launch)")" -maxdepth 1 -name '*.wideep-orig')
if [[ -n ${WIDEEP_POLLUTION_CHECKS:-} ]]; then
# The installed package is now the wheel's: every Python file matches the wheel's RECORD hash (an earlier case's
# overlay or markers left behind fail the case). Modules an overlay added are listed only: the wheel never imports them.
python3 - "$run/pkgcheck-$SLURM_NODEID.log" <<'PY' || echo "PKG_NOT_WHEEL_WARN $SLURM_NODEID $(head -c 300 "$run/pkgcheck-$SLURM_NODEID.log" | tr '\n' ';')" >> "$run/timeline"
import base64, hashlib, sys
from importlib.metadata import distribution
dist = distribution("tensorrt_llm")
root = dist.locate_file("tensorrt_llm")
recorded, bad = set(), []
for f in dist.files:
    if not (str(f).startswith("tensorrt_llm/") and str(f).endswith(".py") and f.hash):
        continue
    path = f.locate()
    recorded.add(str(path))
    try:
        digest = hashlib.new(f.hash.mode, path.read_bytes()).digest()
    except FileNotFoundError:
        bad.append(f"MISSING {f}")
        continue
    if base64.urlsafe_b64encode(digest).rstrip(b"=").decode() != f.hash.value:
        bad.append(f"MODIFIED {f}")
extra = sorted(str(p) for p in root.rglob("*.py") if str(p) not in recorded)
with open(sys.argv[1], "w") as log:
    print("\n".join(bad + [f"EXTRA {p}" for p in extra] + [f"PKG_CHECK files={len(recorded)} bad={len(bad)} extra={len(extra)}"]), file=log)
sys.exit(1 if bad else 0)
PY
fi
if [[ -n "${MVP_OVERLAY:-}" ]]; then
    while read -r rel; do
        # A module the wheel lacks is added as is; only the wheel's own files are restored later.
        [[ -e "$pkg/$rel.wideep-orig" || ! -e "$pkg/$rel" ]] || cp "$pkg/$rel" "$pkg/$rel.wideep-orig"
        cp "$MVP_OVERLAY/tensorrt_llm/$rel" "$pkg/$rel"
    done < <(cd "$MVP_OVERLAY/tensorrt_llm" && find . -name '*.py' | sed 's|^\./||')
    # A launcher under test (trtllm-llmapi-launch is a script outside the package).
    if [[ -f "$MVP_OVERLAY/tensorrt_llm/llmapi/trtllm-llmapi-launch" ]]; then
        launcher=$(command -v trtllm-llmapi-launch)
        [[ -e "$launcher.wideep-orig" ]] || cp "$launcher" "$launcher.wideep-orig"
        cp "$MVP_OVERLAY/tensorrt_llm/llmapi/trtllm-llmapi-launch" "$launcher"
    fi
    echo "OVERLAY $SLURM_NODEID $MVP_OVERLAY" >> "$run/timeline"
fi
# Fault-tolerance settings (TLLM_FAULT_TOLERANCE_CONFIG, read by workers and authority) from the cases' variables;
# wheels before the config object read the variables themselves.
{
    [[ -n "${TLLM_FAULT_TOLERANCE_EXTRA_SLOTS:-}" && $TLLM_FAULT_TOLERANCE_EXTRA_SLOTS != replicas ]] \
        && echo "extra_slots: $TLLM_FAULT_TOLERANCE_EXTRA_SLOTS"
    # Explicit, so cases keep their meaning whatever the code default is (single ranks unless a case asks).
    echo "failure_domain: ${WIDEEP_FAILURE_DOMAIN:-rank}"
    [[ -n "${TLLM_FAULT_TOLERANCE_RECOVERY_TIMEOUT_S:-}" ]] && echo "recovery_timeout_s: $TLLM_FAULT_TOLERANCE_RECOVERY_TIMEOUT_S"
    [[ ${TLLM_FAULT_TOLERANCE_HANG_DETECTION:-} == 1 ]] && echo "hang_detection: true"
    [[ -n "${TLLM_FAULT_TOLERANCE_HANG_CONFIRM_S:-}" ]] && echo "hang_confirm_s: $TLLM_FAULT_TOLERANCE_HANG_CONFIRM_S"
    [[ -n "${TLLM_FAULT_TOLERANCE_HANG_SUSPECT_S:-}" ]] && echo "hang_suspect_s: $TLLM_FAULT_TOLERANCE_HANG_SUSPECT_S"
    [[ -n "${TLLM_FAULT_TOLERANCE_HEARTBEAT_S:-}" ]] && echo "heartbeat_s: $TLLM_FAULT_TOLERANCE_HEARTBEAT_S"
    [[ -n "${TLLM_FAULT_TOLERANCE_PEER_FAILURE_POLL_S:-}" ]] && echo "peer_failure_poll_s: $TLLM_FAULT_TOLERANCE_PEER_FAILURE_POLL_S"
    [[ -n "${WIDEEP_HOST_COPY_NODES:-}" ]] && echo "host_copy_nodes: $WIDEEP_HOST_COPY_NODES"
    true
} > "$run/ft-config-$SLURM_NODEID.yaml"
export TLLM_FAULT_TOLERANCE_CONFIG=$run/ft-config-$SLURM_NODEID.yaml
# Final runs (FINAL_BUILD set) measure the built wheel unchanged: it logs its own recovery phases, so no markers
# unless WIDEEP_MARKERS=1.
if [[ -z ${WIDEEP_STOCK:-} && -e "$pkg/_torch/pyexecutor/peer_loss_recovery.py" && ( -z "${FINAL_BUILD:-}" || "${WIDEEP_MARKERS:-0}" == 1 ) ]]; then  # absent on main
    # Restored by the next case like an overlaid file, so a case without markers runs the wheel's file.
    plr=$pkg/_torch/pyexecutor/peer_loss_recovery.py
    [[ -e "$plr.wideep-orig" ]] || cp "$plr" "$plr.wideep-orig"
    python3 "$H/add_markers.py" "$pkg/_torch/pyexecutor/peer_loss_recovery.py" || { echo "MARKERS_FAILED"; exit 1; }
fi
site=$(python3 -c "import site; print(site.getsitepackages()[0])")
if [[ -n ${WIDEEP_STOCK:-} ]]; then
    [[ -e "$site/wideep_testhooks.pth" ]] && : > "$site/wideep_testhooks.pth"  # an earlier case's hooks, emptied
else
    cp "$H/testhooks.py" "$site/wideep_testhooks.py"; echo "import wideep_testhooks" > "$site/wideep_testhooks.pth"
fi

export LLM_MODELS_ROOT=${LLM_MODELS_ROOT:-/lustre/fs1/portfolios/coreai/projects/coreai_comparch_trtllm/common}
# Standard TRT-LLM perf environment (Python GC off in server and workers, PDL on), plus the TRT-LLM MLA
# decode kernel: the CuteDSL one faults out of bounds at large batches with or without FT.
export TRTLLM_SERVER_DISABLE_GC=${TRTLLM_SERVER_DISABLE_GC:-1} TRTLLM_WORKER_DISABLE_GC=${TRTLLM_WORKER_DISABLE_GC:-1} TRTLLM_ENABLE_PDL=${TRTLLM_ENABLE_PDL:-1} TLLM_FMHA_LIBS=-cute_dsl_mla
# Info logging as in perf-sanity: the per-iteration lines of print_iter_log (engine step times) and the
# [startup] stage timings of the restart breakdown.
export TRTLLM_KVCACHE_POOL_USE_FABRIC_MEMORY=0
if [[ -z ${WIDEEP_STOCK:-} ]]; then
    export PYTHONUNBUFFERED=1 TLLM_LOG_LEVEL=info
    # Listen backlog above the client's concurrent fresh connections (testhooks.py), the same in every arm.
    export TLLM_TEST_SERVE_BACKLOG=${TLLM_TEST_SERVE_BACKLOG:-16384}
fi
# Node-local Triton cache: ranks of 8 nodes compiling into the shared home cache once read a file another node was
# replacing (Stale file handle, a rank lost at startup in GLM-MTP 8043252).
export TRITON_CACHE_DIR=/tmp/triton-cache-${SLURM_JOB_ID:-0}
for var in $(compgen -e | grep -E '^(PMIX_|PMI_|OMPI_MCA_ess|PRTE_MCA_ras)'); do unset "$var"; done
port=${BENCH_PORT:-8000}
if [[ "$SLURM_NODEID" == 0 && "$mode" == ft ]]; then
    (umask 077; python3 -c "import os;print(os.urandom(32).hex())" > "$run/recovery.key")
    # Below the ephemeral range (32768-60999): ULFM's TCP transport opens many outgoing connections.
    authority_port=$((20000 + ${SLURM_JOB_ID:-0} % 300 * 30 + ${CASE_INDEX:-0}))
    # The first port from there that binds (a fixed port once collided with a socket already on the node).
    authority_port=$(python3 -c "
import socket, sys
port = int(sys.argv[1])
while True:
    with socket.socket() as s:
        try:
            s.bind(('', port))
            break
        except OSError:
            port += 1
print(port)" "$authority_port")
    export TLLM_FAULT_TOLERANCE_MODE=1 TLLM_FAULT_TOLERANCE_AUTHORITY_KEY=$(cat "$run/recovery.key")
    export TLLM_FAULT_TOLERANCE_AUTHORITY=tcp://$(hostname):$authority_port
    # Hang detection cases: WIDEEP_FENCE_COMMAND (a {host}/{pid} template) lets the authority kill a hung rank.
    fence=(); [[ -n "${WIDEEP_FENCE_COMMAND:-}" ]] && fence=(--fence-command "$WIDEEP_FENCE_COMMAND")
    # Tagged apart from the case (the case's own sweep would stop it early); the next case's sweep catches a leftover.
    CUDA_VISIBLE_DEVICES= WIDEEP_CASE_TAG=$harness_tag python3 -m tensorrt_llm.executor.recovery.authority \
        --endpoint "tcp://*:$authority_port" --world-size "${WIDEEP_WORLD:-32}" "${fence[@]}" \
        > "$run/authority.log" 2>&1 &
    authority=$!
    listening() { python3 -c "import socket,sys; socket.create_connection(('127.0.0.1', int(sys.argv[1])), 1).close()" "$authority_port" 2>/dev/null; }
    # Wait until it listens: importing tensorrt_llm takes longer than a fixed sleep, so a failed bind went unseen.
    for _ in $(seq 120); do
        listening && break
        kill -0 $authority 2>/dev/null || break
        sleep 1
    done
    listening || { echo "AUTHORITY_FAILED"; touch "$run/done" "$run/frontend.done"; exit 1; }
fi
if [[ "$SLURM_NODEID" == 0 && -z "${TOKENS_ARGS:-}" ]]; then
    WIDEEP_CASE_TAG=$harness_tag python3 "$H/bench.py" --run "$run" --port "$port" $([[ $mode == ft ]] && echo --ft) "$@" > "$run/bench.log" 2>&1 &
    bench=$!
fi
model=${WIDEEP_MODEL:-$LLM_MODELS_ROOT/DeepSeek-R1-0528-FP4-V2}
if [[ -z ${WIDEEP_STOCK:-} ]]; then
# Benchmark hygiene: once the server is ready, drop the weights from this node's page cache. Otherwise the kernel
# drops them on its own ~10 min after the read, and every rank's host loop stalls for a minute in the middle of a
# measurement (analysis/AGING3.md).
# WIDEEP_KEEP_PAGE_CACHE=1 keeps them (a warm restart after an FT-off abort).
[[ -z "${WIDEEP_KEEP_PAGE_CACHE:-}" ]] && (
    for _ in $(seq 3600); do [[ -e "$run/server_ready" || -e "$run/done" ]] && break; sleep 1; done
    [[ -e "$run/server_ready" ]] && python3 "$H/drop_page_cache.py" "$model" > "$run/dropcache-$SLURM_NODEID.log" 2>&1
) &
# WIDEEP_NO_SAMPLERS=1 (final cost packs): neither of the two samplers below runs.
# GPU memory every 5 s while the case runs, per node: whether a dead peer's mappings are freed after a kill.
[[ -z ${WIDEEP_NO_SAMPLERS:-} ]] && (
    while [[ ! -e "$run/done" ]]; do
        nvidia-smi --query-gpu=timestamp,index,memory.used,memory.free --format=csv,noheader 2>/dev/null
        sleep 5
    done > "$run/gpumem-$SLURM_NODEID.csv"
) &
# Host pressure every second while the case runs, per node (EV6: host-side slow steps): vmstat plus the kernel's
# CPU / memory / IO pressure (PSI some avg10).
[[ -z ${WIDEEP_NO_SAMPLERS:-} ]] && (
    while [[ ! -e "$run/done" ]]; do
        psi=$(for r in cpu memory io; do awk -v r=$r '/^some/ {split($2, a, "="); printf " %s=%s", r, a[2]}' /proc/pressure/$r 2>/dev/null; done)
        dstate=$(ps -eLo stat=,wchan:32=,comm= 2>/dev/null | awk '$1 ~ /^D/ {print $2 "/" $3}' | sort | uniq -c | sort -rn | head -4 | tr -s ' \n' ' ;')
        mig=$(awk '/^(pgmigrate_success|compact_migrate_scanned|compact_daemon_migrate_scanned|compact_stall|kcompactd_wake|thp_migration_success|thp_collapse_alloc|numa_pages_migrated) / {printf " %s=%s", $1, $2}' /proc/vmstat)
        echo "$(date +%s) $(vmstat -w 1 2 | tail -1)$psi D:${dstate:-none}$mig"
    done > "$run/hostpressure-$SLURM_NODEID.log" 2>&1
) &
# WIDEEP_NUMA_SAMPLER=1: per-NUMA meminfo and all of /proc/vmstat every second while the case runs, per node (host
# memory placement and reclaim around host-side slow steps), one line each per sample.
[[ -n ${WIDEEP_NUMA_SAMPLER:-} ]] && (
    while [[ ! -e "$run/done" ]]; do
        t=$(date +%s.%N)
        for f in /sys/devices/system/node/node*/meminfo; do
            echo "$t numa $(awk 'NR == 1 {printf "node=%s ", $2} {sub(":", "", $3); printf "%s=%s ", $3, $4}' "$f")"
        done
        echo "$t vmstat $(tr ' ' '=' < /proc/vmstat | tr '\n' ' ')"
        sleep 1
    done > "$run/numastat-$SLURM_NODEID.log" 2>&1
) &
# WIDEEP_MEM_SAMPLER_S=N (stress tests, memory growth): every N seconds while the case runs, per node: each GPU's
# memory used (MiB), the node's MemAvailable / Shmem / AnonPages and /dev/shm use (KiB), and each of the case's
# processes' VmRSS / RssAnon / RssShmem (KiB) with its MPI rank.
[[ -n ${WIDEEP_MEM_SAMPLER_S:-} ]] && (
    while [[ ! -e "$run/done" ]]; do
        t=$(date +%s)
        nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits 2>/dev/null | awk -v t="$t" -F', *' '{print t, "gpu", $1, $2}'
        echo "$t host $(awk '/^(MemAvailable|Shmem|AnonPages|Cached|SReclaimable|SUnreclaim|KernelStack|PageTables):/ {sub(":", "", $1); printf "%s=%s ", $1, $2}' /proc/meminfo)shm_kb=$(df -k /dev/shm | awk 'NR == 2 {print $3}')"
        for p in $(tagged_pids "$tag"); do
            rank=$(tr '\0' '\n' 2>/dev/null < "/proc/$p/environ" | sed -n 's/^OMPI_COMM_WORLD_RANK=//p') || rank=
            echo "$t proc $p rank=${rank:--} comm=$(cat "/proc/$p/comm" 2>/dev/null) $(awk '/^(VmRSS|RssAnon|RssShmem):/ {sub(":", "", $1); printf "%s=%s ", $1, $2}' "/proc/$p/status" 2>/dev/null)"
        done
        sleep "$WIDEEP_MEM_SAMPLER_S"
    done > "$run/mem-$SLURM_NODEID.log" 2>&1
) &
# WIDEEP_DSTATE_SAMPLER=1: every WIDEEP_DSTATE_SAMPLER_S seconds (default 1) while the case runs, per node: every thread in D state (pid, tid, stat, wait
# channel, command), and every 5 s the Lustre client stats if this container can read them (blocked-thread diagnosis).
[[ -n ${WIDEEP_DSTATE_SAMPLER:-} ]] && (
    i=0
    while [[ ! -e "$run/done" ]]; do
        t=$(date +%s.%N)
        ps -eLo pid=,tid=,stat=,wchan:40=,comm= 2>/dev/null | awk -v t="$t" '$3 ~ /^D/ {print t, $0}'
        if (( i % 5 == 0 )); then
            for f in /proc/fs/lustre/llite/*/stats /sys/kernel/debug/lustre/llite/*/stats; do
                [[ -r $f ]] && echo "$t lustre $f $(awk '{printf "%s=%s ", $1, $2}' "$f")"
            done
            command -v lctl > /dev/null && echo "$t lctl $(lctl get_param -n llite.*.stats 2>/dev/null | awk '{printf "%s=%s ", $1, $2}')"
        fi
        i=$((i + 1))
        sleep "${WIDEEP_DSTATE_SAMPLER_S:-1}"
    done > "$run/dstate-$SLURM_NODEID.log" 2>&1
) &
# Host memory once serving (EPLB host copies live in /dev/shm), per node.
(
    for _ in $(seq 3600); do [[ -e "$run/server_ready" || -e "$run/done" ]] && break; sleep 1; done
    sleep 30
    { df -B1 /dev/shm; free -b; } > "$run/hostmem-$SLURM_NODEID.log" 2>&1
) &
# WIDEEP_PIN_BUSY=1 (PIN-AB): 45 s into serving, each busy thread of this node's ranks gets its own core and every
# other thread of the case the node's remaining cores (pin_busy.py); the thread snapshot at 60 s shows the result.
[[ -n "${WIDEEP_PIN_BUSY:-}" ]] && (
    for _ in $(seq 3600); do [[ -e "$run/server_ready" || -e "$run/done" ]] && break; sleep 1; done
    sleep 45
    [[ -e "$run/done" ]] || python3 "$H/pin_busy.py" $(tagged_pids "$tag") > "$run/pin-$SLURM_NODEID.log" 2>&1
    sleep 20  # the threads' CPUs and affinities after pinning, as the thread snapshot below records them
    for p in $(tagged_pids "$tag"); do
        ps -L -o pid=,tid=,psr=,pcpu=,comm= -p "$p" 2>/dev/null | while read -r pid tid psr pcpu comm; do
            echo "$pid $tid cpu=$psr pcpu=$pcpu $comm affinity=$(taskset -pc "$tid" 2>/dev/null | sed 's/.*: //')"
        done
    done > "$run/threads-pinned-$SLURM_NODEID.log" 2>&1
) &
# WIDEEP_THREAD_SNAPSHOT=1: every thread of the case's processes with its CPU, utilization and affinity, 60 s into
# serving (where the LB copy threads, ULFM's detector/progress threads and Python run).
[[ -n "${WIDEEP_THREAD_SNAPSHOT:-}" ]] && (
    for _ in $(seq 3600); do [[ -e "$run/server_ready" || -e "$run/done" ]] && break; sleep 1; done
    sleep 60
    for p in $(tagged_pids "$tag"); do
        ps -L -o pid=,tid=,psr=,pcpu=,comm= -p "$p" 2>/dev/null | while read -r pid tid psr pcpu comm; do
            echo "$pid $tid cpu=$psr pcpu=$pcpu $comm affinity=$(taskset -pc "$tid" 2>/dev/null | sed 's/.*: //')"
        done
    done > "$run/threads-$SLURM_NODEID.log" 2>&1
) &
fi
if [[ -n "${TOKENS_ARGS:-}" ]]; then
    # Exact-token check: the LLM-API client in tokens/ on a small reproducible config (writes frontend.done).
    # shellcheck disable=SC2206
    program=(trtllm-llmapi-launch python3 "$H/tokens/frontend.py" --model "$model"
             --config "$H/tokens/${TOKENS_CONFIG:-config-small.yaml}" --output "$run" $TOKENS_ARGS)
else
    config=$H/configs/${BENCH_CONFIG:-dep32.yaml}
    # Case overrides of the serving config: kv_cache_config.host_cache_size (WIDEEP_HOST_CACHE_SIZE, bytes per rank)
    # extra CUDA graph batch sizes (WIDEEP_EXTRA_GRAPH_BATCH_SIZES, comma-separated) and
    # attention_dp_config.batching_wait_iters (WIDEEP_ADP_WAIT_ITERS).
    if [[ -n "${WIDEEP_HOST_CACHE_SIZE:-}${WIDEEP_EXTRA_GRAPH_BATCH_SIZES:-}${WIDEEP_ADP_WAIT_ITERS:-}" ]]; then
        python3 - "$config" "$run/config-$SLURM_NODEID.yaml" "${WIDEEP_HOST_CACHE_SIZE:-}" "${WIDEEP_EXTRA_GRAPH_BATCH_SIZES:-}" \
            "${WIDEEP_ADP_WAIT_ITERS:-}" <<'PY' \
            || { echo "CONFIG_FAILED"; touch "$run/done" "$run/frontend.done"; exit 1; }
import sys
import yaml
src, dst, host_cache, sizes, wait_iters = sys.argv[1:]
config = yaml.safe_load(open(src))
if wait_iters:
    config.setdefault("attention_dp_config", {})["batching_wait_iters"] = int(wait_iters)
if host_cache:
    config.setdefault("kv_cache_config", {})["host_cache_size"] = int(host_cache)
if sizes:
    graphs = config.setdefault("cuda_graph_config", {})
    base = graphs.get("batch_sizes")
    if not base:  # the sizes CudaGraphConfig generates from max_batch_size (llm_args.py), which must stay their max
        top = graphs.get("max_batch_size") or 128
        if graphs.get("enable_padding"):
            base = [1, 2, 4] + [i * 8 for i in range(1, 17)]
            while base[-1] + 64 <= top:
                base.append(base[-1] + 64)
        else:
            base = list(range(1, 32)) + [32, 64, 128] + [2**i for i in range(8, (top - 1).bit_length())]
        base = [n for n in base if n <= top] + [top]
    graphs["batch_sizes"] = sorted(set(base) | {int(n) for n in sizes.split(",")})
    graphs["max_batch_size"] = max(graphs["batch_sizes"])
yaml.safe_dump(config, open(dst, "w"), sort_keys=False)
PY
        config=$run/config-$SLURM_NODEID.yaml
    fi
    # The recipe's print_iter_log (false) for a stock launch or WIDEEP_NO_ITER_LOG=1: the harness's readout turns it on.
    if [[ -n ${WIDEEP_STOCK:-}${WIDEEP_NO_ITER_LOG:-} ]]; then
        grep -v '^print_iter_log:' "$config" > "$run/config-$SLURM_NODEID.yaml"; config=$run/config-$SLURM_NODEID.yaml
    fi
    program=(trtllm-llmapi-launch trtllm-serve "$model" --config "$config" --host 0.0.0.0 --port "$port")
fi
# The case's effective launch, to diff arms: server config, wheel, mode, argv and the environment that shapes the
# server (secrets redacted).
[[ -n ${config:-} ]] && cp "$config" "$run/effective-config-$SLURM_NODEID.yaml"
{ echo "mode=$mode dist=$dist"; echo "program=${program[*]}"
  env | { grep -E '^(TLLM_|TRTLLM_|WIDEEP_|NCCL_|UCX_|OMPI_|PRTE_|CUDA_|PYTORCH_|TORCH_|MVP_|BENCH_CONFIG=|LD_PRELOAD=)' || true; } \
      | sed -E 's/^([A-Z0-9_]*(KEY|TOKEN|SECRET|PASSWORD)[A-Z0-9_]*)=.*/\1=<redacted>/' | sort; } > "$run/env-$SLURM_NODEID.txt"
rc=0
# FT on runs under ULFM mpirun; FT off under plain mpirun, as deployed without FT (a rank's death aborts
# the job); noft-ulfm is FT off under ULFM mpirun, to check that the launcher alone costs nothing.
# The driver ends a case by writing frontend.done; launch/task.sh then stops mpirun.
# Diagnosis only: GPU clocks, power and throttle state every 2 s, and CPU/memory/IO activity every 1 s.
if [[ -n "${WIDEEP_DMON:-}" ]]; then
    nvidia-smi dmon -s pucvt -o T -d 2 > "$run/dmon-$SLURM_NODEID.log" 2>&1 & dmon=$!
    vmstat -t -w 1 > "$run/vmstat-$SLURM_NODEID.log" 2>&1 & vmstat=$!
fi
WIDEEP_NO_ULFM=$([[ $mode == ft || $mode == noft-ulfm ]] || echo 1) WIDEEP_STALL_S=${WIDEEP_STALL_S:-5400} \
    WIDEEP_CASE_TAG=$tag bash "$L/task.sh" "$run" 4 \
    $([[ -n ${WIDEEP_STOCK:-} ]] || echo bash "$H/rank.sh" "$run"; [[ -n ${WIDEEP_STOCK:-} ]] && command -v numactl > /dev/null && echo numactl -m 0,1) "${program[@]}" || rc=$?
# mpirun has exited (node 0) or node 0 wrote done (other nodes): whatever of the case still runs here is left over.
# Clean-shutdown cases record it before the sweep kills it: after a graceful stop, a case process still alive
# WIDEEP_LEFTOVER_GRACE_S after mpirun exited is one the deployment would leave behind (CASE_LEFTOVER fails the case).
if [[ -n "${WIDEEP_GRACEFUL_STOP:-}" ]]; then
    left=""
    for _ in $(seq "${WIDEEP_LEFTOVER_GRACE_S:-15}"); do
        left=$(tagged_pids "$tag" | paste -sd,)
        [[ -z $left ]] && break
        sleep 1
    done
    if [[ -n $left ]]; then
        echo "CASE_LEFTOVER $SLURM_NODEID $(date +%s.%N) $(ps -o pid=,args= -p "$left" 2>/dev/null | cut -c1-160 | tr '\n' ';')" >> "$run/timeline"
    else
        echo "CASE_NO_LEFTOVER $SLURM_NODEID $(date +%s.%N)" >> "$run/timeline"
    fi
fi
sweep "$tag" NODE_SWEEP
# TLLM_TEST_LOCAL_LOGS: this node's rank logs replace their links in the run dir.
if [[ -n "${TLLM_TEST_LOCAL_LOGS:-}" ]]; then
    for log in /tmp/wideep-ranklogs/${run//\//_}/rank-*.log; do
        [[ -e $log ]] && cp --remove-destination "$log" "$run/${log##*/}"
    done
fi
# Teardown is complete on this node once no GPU has a compute process left.
for _ in $(seq 1200); do
    [[ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null)" ]] && break
    sleep 0.1
done
# A process the case left on a GPU (e.g. a profiled rank's re-parented child) would block this node's next case;
# the node is this job's alone, so end it.
leftover=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | paste -sd,)
if [[ -n $leftover ]]; then
    echo "NODE_LEFTOVER $SLURM_NODEID pids=$leftover $(ps -o pid=,args= -p "$leftover" 2>/dev/null | cut -c1-200 | tr '\n' ';')" >> "$run/timeline"
    kill -9 ${leftover//,/ } 2>/dev/null || true
fi
echo "NODE_CLEAN $SLURM_NODEID $(date +%s.%N)" >> "$run/timeline"
[[ -n "${dmon:-}" ]] && kill "$dmon" "$vmstat" 2>/dev/null
[[ -n "${authority:-}" ]] && kill "$authority" 2>/dev/null
# The driver stops on its own once the server is gone; bound the wait anyway.
[[ -n "${bench:-}" ]] && { timeout 120 tail --pid="$bench" -f /dev/null || kill -9 "$bench"; } 2>/dev/null
exit $rc
