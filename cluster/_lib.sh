#!/bin/bash
# ============================================================================
# _lib.sh — Shared library for the T2G cluster scripts.
#
# SINGLE SOURCE OF TRUTH for the .chain_state paths and the core chain
# primitives used by run_all.sh, chain_tick.sh and aliases.sh.
#
# NOTE: this file is also sourced by INTERACTIVE shells (via aliases.sh), so
# it MUST NOT: print anything, cd, `set -e`, or create files at source time.
# Each script sets its own shell options AFTER sourcing.
#
# NOTE: src/utils/chain_monitor.py keeps its own constants (same values).
# Keep the two in sync whenever a path changes.
#
# NOTE: the login node has NO python (only shell commands: ssh, sbatch,
# squeue, sacct, cat, ...). Everything defined here is pure shell.
#
# OFFLINE INVARIANT (cluster hardening):
#   - login node  → may have network but NEVER runs project python;
#   - compute job → has python/GPU but NO internet (no DNS).
# Therefore NOTHING that runs on a compute node may pip install, download,
# or fall back to downloading. All compute scripts MUST call
# export_offline_env BEFORE any run_py/python/apptainer invocation and MUST
# call require_cluster_artifacts instead of regenerating/downloading data.
# All acquisition (dependencies, HF model, dataset, vocab/bigram artifacts)
# happens in a separate network-enabled environment (login workflow / local
# machine) and is synced to the shared NFS project dir before submission.
# ============================================================================

PROJ_DIR="$HOME/neuro_symbolic_t2g"

# ── .chain_state paths (single source of truth) ──────────────────────────────
STATE_DIR="$PROJ_DIR/.chain_state"
CHAIN_FILE="$STATE_DIR/job_chain"        # queue: one "type:cfg:tag[:extra]" per line
ERRORS_FILE="$STATE_DIR/chain_errors"    # JSONL failure log (read by the monitor)
LAST_JOB_FILE="$STATE_DIR/last_job"      # "id:type:cfg:tag:retries" of last submission
STOPPED_FILE="$STATE_DIR/chain_stopped"  # present ⇒ pipeline paused by chain-stop
TICK_LOCK="$STATE_DIR/tick.lock"         # flock file — max 1 tick at a time
TICK_STAMP="$STATE_DIR/tick_stamp"       # epoch int, throttle for the bashrc hook
CHAIN_LOG="$PROJ_DIR/logs/chain.log"

# Max auto-resume for TIMEOUT/OOM/CUDA on the SAME train job.
MAX_RETRIES=2

# Common SLURM account/partition/qos defaults. train.sh/eval.sh keep their own
# #SBATCH headers; these defaults are only for the helper sbatch invocations.
SLURM_ACCOUNT_DEFAULT="${SLURM_ACCOUNT:-thesis-course}"
SLURM_QOS_DEFAULT="${SLURM_QOS:-gpu-xlarge}"

# ── Offline environment (compute nodes have NO internet) ────────────────────
# Shared NFS makes login-prepared snapshots visible to compute jobs.
T2G_HF_HOME_DEFAULT="$HOME/.cache/huggingface"
# Default MiniLM id — keep in sync with
# src/retrieval/example_retriever.py::_DEFAULT_MINILM_MODEL.
T2G_MINILM_DEFAULT="sentence-transformers/all-MiniLM-L6-v2"

# export_offline_env — MUST be called by every compute-node script BEFORE the
# first run_py/python/apptainer invocation; hub clients read these at import.
# Sets:
#   HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
#   WANDB_MODE=offline WANDB_DISABLE_WEAVE=true WANDB_SILENT=true
#   PYTHONUNBUFFERED=1
#   HF_HOME → $PROJ_DIR/.hf_cache unless the caller already set it.
export_offline_env() {
    export HF_HUB_OFFLINE=1
    export TRANSFORMERS_OFFLINE=1
    export HF_DATASETS_OFFLINE=1
    export WANDB_MODE=offline
    export WANDB_DISABLE_WEAVE=true
    export WANDB_SILENT=true
    export PYTHONUNBUFFERED=1
    if [ -z "${HF_HOME:-}" ]; then
        export HF_HOME="$T2G_HF_HOME_DEFAULT"
    fi
    mkdir -p "$HF_HOME" 2>/dev/null || true
    export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
}

# Globals used by the chain primitives (initialised here for `set -u` safety).
LAST_JOB_ID=""
LAST_JOB_TYPE=""
LAST_JOB_CFG=""
LAST_JOB_TAG=""
LAST_JOB_RETRIES=0
_SACCT_STATE=""
_SACCT_EXIT_CODE=""

# ── Logging ──────────────────────────────────────────────────────────────────
log_line() {
    echo "[chain] $(date '+%F %T') $*" >> "$CHAIN_LOG" 2>/dev/null || true
}

# log_submit / log_job_id MUST keep these exact formats: the monitor
# (src/utils/chain_monitor.py::_parse_chain_log) greps for
# "[chain] Sottometto: <type> <tag>" and "[chain] Job ID: <id>".
log_submit() {
    echo "[chain] Sottometto: $*" >> "$CHAIN_LOG" 2>/dev/null || true
}
log_job_id() {
    echo "[chain] Job ID: $*" >> "$CHAIN_LOG" 2>/dev/null || true
}

# Append one JSONL entry to chain_errors (read by the monitor).
# Usage: log_job_error <job_id> <type> <cfg> <tag> <state> <exit> <err_type> [retry] [resolved]
log_job_error() {
    local job_id="${1:-}" job_type="${2:-}" config="${3:-}" tag="${4:-}"
    local state="${5:-}" exit_code="${6:-}" error_type="${7:-UNKNOWN}"
    local retry_num="${8:-0}" resolved="${9:-false}"
    local logfile="" snippet="" ts
    [ -n "$job_id" ] && logfile="$PROJ_DIR/logs/slurm-${job_type}-${job_id}.log"
    if [ -f "$logfile" ]; then
        snippet=$(tail -200 "$logfile" 2>/dev/null \
            | grep -iE 'error|cuda|oom|traceback|exception|illegal|killed|out of memory|sigkill|acceleratorerror|device-side assert' \
            | tail -5 | tr '\n' ' ' | cut -c1-800) || true
    fi
    # JSON-escape the snippet (best effort — configs/tags are alnum + -_./).
    snippet=$(printf '%s' "$snippet" | sed 's/\\/\\\\/g; s/"/\\"/g')
    ts=$(date '+%Y-%m-%d %H:%M:%S')
    printf '{"tag":"%s","job_type":"%s","slurm_id":"%s","config":"%s","error_type":"%s","slurm_state":"%s","exit_code":"%s","timestamp":"%s","error_snippet":"%s","retry_num":%s,"resolved":%s}\n' \
        "$tag" "$job_type" "$job_id" "$config" "$error_type" "$state" "$exit_code" \
        "$ts" "$snippet" "$retry_num" "$resolved" >> "$ERRORS_FILE" 2>/dev/null || true
}

# ── Chain state helpers ──────────────────────────────────────────────────────
_ensure_state_dir() {
    mkdir -p "$STATE_DIR"
}

# Number of entries still queued (0 when none).
chain_remaining() {
    if [ -s "$CHAIN_FILE" ]; then
        wc -l < "$CHAIN_FILE"
    else
        echo 0
    fi
}

# rebuild_chain <entry> — reinsert one entry at the head of job_chain.
rebuild_chain() {
    local entry="$1" tmp
    tmp=$(mktemp)
    echo "$entry" > "$tmp"
    [ -s "$CHAIN_FILE" ] && cat "$CHAIN_FILE" >> "$tmp"
    mv "$tmp" "$CHAIN_FILE"
}

# Pop the head of job_chain, printing it (removes the line).
pop_chain_head() {
    [ -s "$CHAIN_FILE" ] || return 1
    head -1 "$CHAIN_FILE"
    tail -n +2 "$CHAIN_FILE" > "$CHAIN_FILE.tmp" 2>/dev/null && mv "$CHAIN_FILE.tmp" "$CHAIN_FILE"
    if [ ! -s "$CHAIN_FILE" ]; then
        rm -f "$CHAIN_FILE"
    fi
    return 0
}

# Read .chain_state/last_job into the LAST_JOB_* globals (empty-safe).
chain_read_last_job() {
    LAST_JOB_ID=""; LAST_JOB_TYPE=""; LAST_JOB_CFG=""; LAST_JOB_TAG=""; LAST_JOB_RETRIES=0
    [ -f "$LAST_JOB_FILE" ] || return 0
    local line
    line=$(cat "$LAST_JOB_FILE" 2>/dev/null) || return 0
    [ -n "$line" ] || return 0
    LAST_JOB_ID=$(echo "$line" | cut -d: -f1)
    LAST_JOB_TYPE=$(echo "$line" | cut -d: -f2)
    LAST_JOB_CFG=$(echo "$line" | cut -d: -f3)
    LAST_JOB_TAG=$(echo "$line" | cut -d: -f4)
    LAST_JOB_RETRIES=$(echo "$line" | cut -d: -f5)
    case "$LAST_JOB_RETRIES" in
        ''|*[!0-9]*) LAST_JOB_RETRIES=0 ;;
    esac
}

# ── SLURM helpers ────────────────────────────────────────────────────────────
# First active job id for this user (QoS allows exactly 1), empty if none.
active_job_id() {
    squeue --me -h -o '%A|%T' 2>/dev/null | awk -F'|' 'NF && $1!="" {print $1; exit}'
}
active_job_name() {
    squeue --me -h -o '%A|%j' 2>/dev/null | awk -F'|' 'NF && $1!="" {print $2; exit}'
}

# Query sacct for the state of a finished job, retrying while it is not yet
# in a terminal/known state. Sets globals _SACCT_STATE and _SACCT_EXIT_CODE.
# Usage: query_sacct_with_retry <job_id> [max_attempts] [wait_secs]
query_sacct_with_retry() {
    local job_id="$1" max_attempts="${2:-6}" wait_secs="${3:-5}" attempt
    _SACCT_STATE=""; _SACCT_EXIT_CODE=""
    for attempt in $(seq 1 "$max_attempts"); do
        _SACCT_EXIT_CODE=$(sacct -j "$job_id" --format=ExitCode --noheader --parsable2 2>/dev/null | head -1 | cut -d: -f1) || true
        _SACCT_STATE=$(sacct -j "$job_id" --format=State --noheader --parsable2 2>/dev/null | head -1 | tr -d '[:space:]') || true
        case "$_SACCT_STATE" in
            COMPLETED|FAILED|CANCELLED|CANCELLED+|TIMEOUT|OUT_OF_MEMORY|NODE_FAIL|RUNNING|PENDING)
                return 0 ;;
        esac
        [ "$attempt" -lt "$max_attempts" ] && sleep "$wait_secs"
    done
    if [ -z "$_SACCT_STATE" ]; then
        _SACCT_STATE="UNKNOWN"
        return 1
    fi
    return 0
}

# 0 iff the last queried job (_SACCT_STATE/_SACCT_EXIT_CODE) succeeded.
job_succeeded() {
    [ "$_SACCT_STATE" = "COMPLETED" ] || return 1
    if [ -z "$_SACCT_EXIT_CODE" ] || [ "$_SACCT_EXIT_CODE" = "0" ]; then
        return 0
    fi
    return 1
}

# After query_sacct_with_retry: 0 = treat the last job as STILL active.
# RUNNING/PENDING are obvious; UNKNOWN with a recent last_job record means
# sacct lag right after a submission (never fail a job we cannot see yet).
last_job_still_active() {
    case "$_SACCT_STATE" in
        RUNNING|PENDING)
            return 0 ;;
        UNKNOWN)
            if [ -f "$LAST_JOB_FILE" ] && [ -n "$(find "$LAST_JOB_FILE" -mmin -30 2>/dev/null)" ]; then
                return 0
            fi
            return 1
            ;;
        *)
            return 1 ;;
    esac
}

# OOM / CUDA detection from the SLURM log of the finished training job.
is_oom_failure() {
    local job_id="$1" exit_code="$2" state="$3" logfile
    [ "$state" = "OUT_OF_MEMORY" ] && return 0
    [ "$exit_code" = "137" ] && return 0
    logfile="$PROJ_DIR/logs/slurm-train-${job_id}.log"
    if [ -f "$logfile" ]; then
        if tail -200 "$logfile" 2>/dev/null | grep -qiE 'out.of.memory|OutOfMemoryError|CUDA out of memory|oom-kill|OOM|torch.cuda.OutOfMemoryError|std::bad_alloc|excessive GPU RAM|GPU RAM usage'; then
            return 0
        fi
    fi
    return 1
}
is_cuda_transient_failure() {
    local job_id="$1" logfile
    logfile="$PROJ_DIR/logs/slurm-train-${job_id}.log"
    if [ -f "$logfile" ]; then
        if tail -200 "$logfile" 2>/dev/null | grep -qiE 'cudaErrorIllegalAddress|illegal memory access|cudaErrorLaunchFailure|device-side assert|AcceleratorError.*CUDA error'; then
            return 0
        fi
    fi
    return 1
}

# ── Core chain logic (shared by chain_tick.sh) ─────────────────────────────────
# Classify the last finished job (globals from chain_read_last_job +
# _SACCT_STATE/_SACCT_EXIT_CODE) and apply the pipeline policy:
#   - train + TIMEOUT/OOM/CUDA-transient + retries<MAX_RETRIES
#       → reinsert "train:<cfg>:<tag>:--resume" at the head (auto-resume);
#   - anything else
#       → log to chain_errors, drop the matching queued eval when a train
#         failed (it would otherwise eval an untrained model), and continue
#         with the next entry (continue-on-failure, ablation-friendly).
# Returns 0 → caller may submit the next entry; 1 → chain is finished.
chain_handle_failure() {
    local error_type="UNKNOWN"
    case "$_SACCT_STATE" in
        TIMEOUT)        error_type="TIMEOUT" ;;
        OUT_OF_MEMORY)  error_type="OOM" ;;
        CANCELLED|CANCELLED+) error_type="CANCELLED" ;;
    esac
    is_oom_failure "$LAST_JOB_ID" "$_SACCT_EXIT_CODE" "$_SACCT_STATE" && error_type="OOM"
    is_cuda_transient_failure "$LAST_JOB_ID" "$_SACCT_EXIT_CODE" "$_SACCT_STATE" && error_type="CUDA_ERROR"

    if [ "$error_type" = "TIMEOUT" ] || [ "$error_type" = "OOM" ] || [ "$error_type" = "CUDA_ERROR" ]; then
        if [ "$LAST_JOB_TYPE" = "train" ] && [ "$LAST_JOB_RETRIES" -lt "$MAX_RETRIES" ]; then
            local new_retries=$((LAST_JOB_RETRIES + 1)) failed_id="$LAST_JOB_ID"
            rebuild_chain "train:${LAST_JOB_CFG}:${LAST_JOB_TAG}:--resume"
            echo ":train:${LAST_JOB_CFG}:${LAST_JOB_TAG}:${new_retries}" > "$LAST_JOB_FILE"
            log_job_error "$failed_id" "$LAST_JOB_TYPE" "$LAST_JOB_CFG" "$LAST_JOB_TAG" \
                "$_SACCT_STATE" "$_SACCT_EXIT_CODE" "$error_type" "$new_retries" "true"
            log_line "⏰ train $LAST_JOB_TAG (job $failed_id) $error_type → auto-retry ${new_retries}/${MAX_RETRIES} (--resume)"
            LAST_JOB_ID=""; LAST_JOB_RETRIES=$new_retries
            return 0
        fi
        log_line "❌ train $LAST_JOB_TAG ${error_type} — max retry ($MAX_RETRIES) raggiunto"
    fi

    log_job_error "$LAST_JOB_ID" "$LAST_JOB_TYPE" "$LAST_JOB_CFG" "$LAST_JOB_TAG" \
        "$_SACCT_STATE" "$_SACCT_EXIT_CODE" "$error_type" "0" "false"
    log_line "⚠️  $LAST_JOB_TYPE $LAST_JOB_TAG (job $LAST_JOB_ID) FALLITO state=$_SACCT_STATE exit=$_SACCT_EXIT_CODE — registrato in chain_errors"

    # A failed train must not leave its eval queued (silent zero-shot eval).
    if [ "$LAST_JOB_TYPE" = "train" ] && [ -s "$CHAIN_FILE" ]; then
        local head_entry head_type head_tag
        head_entry=$(head -1 "$CHAIN_FILE")
        head_type=$(echo "$head_entry" | cut -d: -f1)
        head_tag=$(echo "$head_entry" | cut -d: -f3)
        if [ "$head_type" = "eval" ] && [ "$head_tag" = "$LAST_JOB_TAG" ]; then
            pop_chain_head >/dev/null
            log_line "⏭  rimosso eval di $LAST_JOB_TAG dalla catena (train fallito)"
        fi
    fi

    if [ ! -s "$CHAIN_FILE" ]; then
        return 1
    fi
    return 0
}

# Pop the head of job_chain, validate it, submit it via sbatch and record it
# in .chain_state/last_job. Sets the LAST_JOB_* globals.
# Returns 0 on submission; 1 on any failure (the entry is reinserted at the
# head so the next tick can retry it — never lost).
chain_submit_next() {
    [ -s "$CHAIN_FILE" ] || { log_line "❌ job_chain vuoto — niente da sottomettere"; return 1; }

    local next type cfg tag extra out job_id
    next=$(head -1 "$CHAIN_FILE")
    type=$(echo "$next" | cut -d: -f1)
    cfg=$(echo "$next" | cut -d: -f2)
    tag=$(echo "$next" | cut -d: -f3)
    extra=$(echo "$next" | cut -d: -f4-)
    pop_chain_head >/dev/null

    if [ -z "$cfg" ]; then
        log_line "❌ Entry corrotta (config vuoto): '$next' — rimossa"
        log_job_error "" "$type" "" "$tag" "CHAIN_CORRUPT" "1" "CHAIN_CORRUPT" "0" "false"
        return 1
    fi

    case "$type" in
        train)
            out=$(CONFIG="$cfg" EXTRA_ARGS="$extra" sbatch --job-name="train-${tag}" --parsable cluster/train.sh 2>&1) || true
            ;;
        eval)
            out=$(CONFIG="$cfg" sbatch --job-name="eval-${tag}" --parsable cluster/eval.sh 2>&1) || true
            ;;
        *)
            log_line "❌ Tipo sconosciuto '$type' in entry: '$next' — rimossa"
            return 1
            ;;
    esac

    job_id=$(echo "$out" | grep -oE '^[0-9]+$' | head -1) || true
    if [ -z "$job_id" ]; then
        log_line "⚠️  sbatch non ha restituito un job id per $type $tag — entry reinserita"
        log_line "   $(echo "$out" | tr '\n' ' ' | cut -c1-300)"
        rebuild_chain "$next"
        return 1
    fi

    # Carry the retry counter only across an auto --resume retry of the same
    # train job (a fresh job of a different tag restarts from 0).
    local new_retries=0
    if [ "$type" = "train" ] && [ "$LAST_JOB_TYPE" = "train" ] && [ "$LAST_JOB_TAG" = "$tag" ] && [ "$extra" = "--resume" ]; then
        new_retries=$LAST_JOB_RETRIES
    fi

    echo "${job_id}:${type}:${cfg}:${tag}:${new_retries}" > "$LAST_JOB_FILE"
    LAST_JOB_ID=$job_id
    LAST_JOB_TYPE=$type
    LAST_JOB_CFG=$cfg
    LAST_JOB_TAG=$tag
    LAST_JOB_RETRIES=$new_retries

    log_submit "$type $tag ($cfg) extra='$extra'"
    log_job_id "$job_id"
    return 0
}

# ── Monitor cache (best-effort, shell-only) ──────────────────────────────────
# The login node has no python, so the cache is maintained with plain files:
# pipeline_keys (plain list) + regenerated monitor_cache JSON. Without python
# the jobs-dict is reset to {} — the monitor rebuilds it from sacct/logs on
# its next run, so this is only a cosmetic degradation.
_monitor_cache_minimal() {
    local k list="" cache="$STATE_DIR/monitor_cache" keys="$STATE_DIR/pipeline_keys"
    [ -f "$keys" ] || return 0
    while IFS= read -r k; do
        [ -n "$k" ] || continue
        list="$list\"$k\","
    done < "$keys"
    list="${list%,}"
    printf '{"jobs": {}, "pipeline_jobs": [%s]}\n' "$list" > "$cache"
}

# Register a "type-tag" key in the monitor pipeline view.
monitor_cache_add() {
    local key="$1" keys="$STATE_DIR/pipeline_keys" cache="$STATE_DIR/monitor_cache"
    _ensure_state_dir
    if [ ! -f "$keys" ] || ! grep -qF "$key" "$keys"; then
        echo "$key" >> "$keys"
        if command -v python3 >/dev/null 2>&1; then
            python3 - "$cache" "$keys" <<'PYEOF' 2>/dev/null || _monitor_cache_minimal
import json, pathlib, sys
cache_path = pathlib.Path(sys.argv[1])
keys_path = pathlib.Path(sys.argv[2])
jobs = {}
if cache_path.exists():
    try:
        cache = json.loads(cache_path.read_text())
        jobs = cache.get("jobs", {})
    except Exception:
        pass
pipeline_jobs = [l for l in keys_path.read_text().splitlines() if l.strip()]
cache_path.write_text(json.dumps({"jobs": jobs, "pipeline_jobs": pipeline_jobs}, indent=2))
PYEOF
        else
            _monitor_cache_minimal
        fi
    fi
}

monitor_cache_clear() {
    rm -f "$STATE_DIR/monitor_cache" "$STATE_DIR/pipeline_keys"
}

# ── Compute-node python runner (Apptainer) ───────────────────────────────────
# Run python inside the Apptainer SIF when available. Only call this from
# COMPUTE-node scripts (train.sh/eval.sh/setup.sh/preflight.sh jobs) — never
# from login-node scripts. Set RUN_PY_FORCE_BARE=1 to skip Apptainer (used
# inside the container by setup.sh/preflight.sh).
# The offline env (see export_offline_env) is passed to the container BOTH via
# inherited host env and via explicit --env args, so the guarantee holds
# regardless of apptainer's env handling.
run_py() {
    if [ "${RUN_PY_FORCE_BARE:-0}" = "1" ]; then
        python3 "$@"
        return
    fi
    if command -v apptainer >/dev/null 2>&1 && [ -f /shared/sifs/latest.sif ]; then
        apptainer run --nv \
            --env "HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}" \
            --env "TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}" \
            --env "HF_DATASETS_OFFLINE=${HF_DATASETS_OFFLINE:-1}" \
            --env "WANDB_MODE=${WANDB_MODE:-offline}" \
            --env "WANDB_DISABLE_WEAVE=${WANDB_DISABLE_WEAVE:-true}" \
            --env "WANDB_SILENT=${WANDB_SILENT:-true}" \
            --env "PYTHONUNBUFFERED=${PYTHONUNBUFFERED:-1}" \
            --env "HF_HOME=${HF_HOME:-$T2G_HF_HOME_DEFAULT}" \
            --env "HF_HUB_CACHE=${HF_HUB_CACHE:-${HF_HOME:-$T2G_HF_HOME_DEFAULT}/hub}" \
            /shared/sifs/latest.sif \
            python3 "$@"
    else
        python3 "$@"
    fi
}

# ── Offline artifact verification (fail-fast, NO download fallback) ──────────
# Fail fast when artifacts needed by offline compute jobs are unavailable.

# Standard fail message for a missing offline artifact. Always returns 1 so
# callers can `_artifact_fail ... || return 1` under any shell mode.
_artifact_fail() {
    local kind="$1" path="$2" detail="${3:-}"
    {
        echo ""
        echo "═══════════════════════════════════════════════════════════"
        echo "  ❌ ARTIFACT OFFLINE MANCANTE: $kind"
        echo "  Percorso atteso: $path"
        [ -n "$detail" ] && echo "  $detail"
        echo ""
        echo "  I compute node NON hanno internet: da qui NESSUN download"
        echo "  è possibile (niente pip install, niente HF download, niente"
        echo "  fallback di rigenerazione)."
        echo ""
        echo "  Prepara gli artifact in un ambiente con rete (workflow di"
        echo "  setup separato / macchina locale) e sincronizzali sul"
        echo "  cluster PRIMA di sottomettere train/eval:"
        echo "   1. dataset HF (ASLG-PC12) + modello nella cache HF condivisa"
        echo "      ($T2G_HF_HOME_DEFAULT di default);"
        echo "   2. data/gloss_vocab.txt + relativo sidecar .meta.json."
        echo "      La matrice bigram è opzionale per train/eval standard."
        echo "═══════════════════════════════════════════════════════════"
    } >&2
    return 1
}

# Directory nonempty check (pure shell, no ls -A quirks on bash 4).
_dir_nonempty() {
    [ -d "$1" ] || return 1
    local _entry
    for _entry in "$1"/* "$1"/.[!.]*; do
        [ -e "$_entry" ] && return 0
    done
    return 1
}


# Robust HF-cache-only snapshot check (no python). True iff the hub cache
# holds at least one snapshot of <model_id> containing a nonempty file.
# Layout: ${HF_HUB_CACHE:-$HF_HOME/hub}/models--<org>--<name>/snapshots/<rev>/
_hf_model_cached_shell() {
    local model_id="$1" hub_dir id_dir snap f
    hub_dir="${HF_HUB_CACHE:-${HF_HOME:-$T2G_HF_HOME_DEFAULT}/hub}"
    id_dir="$hub_dir/models--$(printf '%s' "$model_id" | sed 's#/#--#g')"
    [ -d "$id_dir" ] || return 1
    for snap in "$id_dir"/snapshots/*; do
        [ -d "$snap" ] || continue
        for f in "$snap"/* "$snap"/.[!.]*; do
            [ -s "$f" ] && return 0
        done
    done
    return 1
}

# Same, but for ANY model: at least one cached snapshot with a nonempty file.
_hf_cache_has_any_snapshot() {
    local hub_dir snap f
    hub_dir="${HF_HUB_CACHE:-${HF_HOME:-$T2G_HF_HOME_DEFAULT}/hub}"
    [ -d "$hub_dir" ] || return 1
    for snap in "$hub_dir"/models--*/snapshots/*; do
        [ -d "$snap" ] || continue
        for f in "$snap"/* "$snap"/.[!.]*; do
            [ -s "$f" ] && return 0
        done
    done
    return 1
}

# ONE short offline probe: reads the resolved config (extends-aware) and
# resolves the model/MiniLM sources exactly like
# src.models.model_loader.resolve_model_source (snapshot_download with
# local_files_only → NEVER downloads, never hits the network). Emits
# shell-parsable KEY=VALUE lines. Config-read failure is non-fatal here:
# the caller falls back to shell-only cache checks.
_T2G_ARTIFACT_PROBE='
import sys
from pathlib import Path

model = ret_enabled = ret_backend = ret_model = dataset_name = ""
vocab_path = vocab_meta = bigram_path = dataset_cache = ""
try:
    from src.utils.config import resolve_config

    cfg = resolve_config(sys.argv[1])
    model = (cfg.get("model") or {}).get("name") or ""
    ret = cfg.get("retrieval") or {}
    ret_enabled = "1" if ret.get("enabled", False) else "0"
    ret_backend = str(ret.get("backend") or "tfidf")
    ret_model = str(ret.get("model_name") or "")
    ds = cfg.get("dataset") or {}
    vocab_path = str(ds.get("vocab_path") or "data/gloss_vocab.txt")
    from src.utils.cache_meta import cache_meta_path
    vocab_meta = str(cache_meta_path(vocab_path))
    bigram_path = str(ds.get("bigram_matrix_path") or "data/bigram_transition.npy")
    dataset_cache = str(ds.get("dataset_cache") or "data/aslg_pc12")
    dataset_name = str(ds.get("dataset_name") or "achrafothman/aslg_pc12")
except Exception:
    print("CONFIG_READ=FAILED")
    sys.exit(0)

print("CONFIG_READ=OK")
print("MODEL=" + model)
print("RETRIEVAL_ENABLED=" + ret_enabled)
print("RETRIEVAL_BACKEND=" + ret_backend)
print("RETRIEVAL_MODEL=" + ret_model)
print("VOCAB_PATH=" + vocab_path)
print("VOCAB_META=" + vocab_meta)
print("BIGRAM_PATH=" + bigram_path)
print("DATASET_CACHE=" + dataset_cache)
print("DATASET_NAME=" + dataset_name)

try:
    from huggingface_hub import snapshot_download
except Exception:
    snapshot_download = None


def resolve_source(mid):
    if not mid or Path(mid).exists():
        return "LOCAL"
    if snapshot_download is None:
        return "UNKNOWN"
    try:
        # local_files_only=True → pure cache scan, zero network.
        return snapshot_download(mid, local_files_only=True)
    except Exception:
        return "MISSING"


if model:
    print("MODEL_SOURCE=" + resolve_source(model))
if ret_enabled == "1" and ret_backend == "minilm":
    print("MINILM_SOURCE=" + resolve_source(ret_model or "sentence-transformers/all-MiniLM-L6-v2"))
'

# See function docstring above.
require_cluster_artifacts() {
    local config="${1:-}"
    local probe_out="" line key value
    local model="" ret_enabled="" ret_backend="" ret_model=""
    local vocab_path="data/gloss_vocab.txt" bigram_path="data/bigram_transition.npy"
    local dataset_cache="data/aslg_pc12" model_source="" minilm_source=""
    local dataset_name="achrafothman/aslg_pc12" vocab_meta="data/gloss_vocab.meta.json"
    local config_ok=0

    if [ -n "$config" ]; then
        # run_py is a shell function, not an executable: invoking it through
        # `timeout run_py ...` always fails with "No such file or directory".
        # The probe is intrinsically bounded/offline: it only imports local
        # config and calls snapshot_download(local_files_only=True), so it
        # cannot block on network retries.
        probe_out=$(run_py -c "$_T2G_ARTIFACT_PROBE" "$config" 2>/dev/null) || probe_out=""
        while IFS='=' read -r key value; do
            [ -n "$key" ] || continue
            case "$key" in
                CONFIG_READ)       [ "$value" = "OK" ] && config_ok=1 ;;
                MODEL)             model=$value ;;
                RETRIEVAL_ENABLED) ret_enabled=$value ;;
                RETRIEVAL_BACKEND) ret_backend=$value ;;
                RETRIEVAL_MODEL)   ret_model=$value ;;
                VOCAB_PATH)        [ -n "$value" ] && vocab_path=$value ;;
                VOCAB_META)        [ -n "$value" ] && vocab_meta=$value ;;
                BIGRAM_PATH)       [ -n "$value" ] && bigram_path=$value ;;
                DATASET_CACHE)     [ -n "$value" ] && dataset_cache=$value ;;
                DATASET_NAME)      [ -n "$value" ] && dataset_name=$value ;;
                MODEL_SOURCE)      model_source=$value ;;
                MINILM_SOURCE)     minilm_source=$value ;;
            esac
        done <<< "$probe_out"
        if [ "$config_ok" != "1" ]; then
            echo "⚠️  Probe config offline non riuscita (${config}) — uso i controlli cache HF di solo shell." >&2
        fi
    fi

    if [ "$dataset_name" != "achrafothman/aslg_pc12" ]; then
        _artifact_fail "Dataset non supportato" "$dataset_name" \
            "Il loader supporta solo achrafothman/aslg_pc12; correggere dataset.dataset_name." \
            || return 1
    fi

    # ── 1. HF dataset cache (ASLG-PC12): directory nonempty ─────────────
    if ! _dir_nonempty "$dataset_cache"; then
        _artifact_fail "Cache dataset HF ASLG-PC12" "$dataset_cache" \
            "I loader fanno load_dataset(name, cache_dir='${dataset_cache}') in offline: serve la cache preparata." \
            || return 1
    fi

    # ── 2. Vocabolario gloss ────────────────────────────────────────────
    if [ ! -s "$vocab_path" ]; then
        _artifact_fail "Vocabolario gloss" "$vocab_path" \
            "Deve essere pre-costruito nell'ambiente con rete (prepare_only) e caricato sul cluster." \
            || return 1
    fi
    if [ ! -s "$vocab_meta" ]; then
        _artifact_fail "Sidecar vocabolario gloss" "$vocab_meta" \
            "Il loader usa questo sidecar per verificare seed e dimensione del train set." \
            || return 1
    fi

    # Bigram is a diagnostic only; eval disables it explicitly when absent.
    if [ ! -s "$bigram_path" ]; then
        echo "   ℹ️  Bigram assente: diagnostica bigram omessa; train/eval core non la richiedono."
    else
        echo "   ✅ Bigram disponibile per la diagnostica: $bigram_path"
    fi

    # ── 4. Snapshot del modello config ──────────────────────────────────
    if [ -n "$model" ]; then
        case "$model_source" in
            LOCAL)
                echo "   ✅ Modello (path locale): $model"
                ;;
            MISSING)
                _artifact_fail "Snapshot HF del modello" "$model" \
                    "snapshot_download(local_files_only=True) non ha trovato la snapshot nella cache HF." \
                    || return 1
                ;;
            UNKNOWN)
                if ! _hf_model_cached_shell "$model"; then
                    _artifact_fail "Snapshot HF del modello" "$model" \
                        "huggingface_hub non disponibile nel probe e nessuna snapshot in cache HF." \
                        || return 1
                fi
                echo "   ✅ Modello in cache HF (verifica shell): $model"
                ;;
            "")
                if ! _hf_model_cached_shell "$model"; then
                    _artifact_fail "Snapshot HF del modello" "$model" \
                        "Nessuna snapshot trovata nella cache HF (verifica shell)." \
                        || return 1
                fi
                echo "   ✅ Modello in cache HF (verifica shell): $model"
                ;;
            *)
                echo "   ✅ Modello in cache HF: $model_source"
                ;;
        esac
    else
        # Nessun model id dal config (probe fallito o config assente):
        # check robusto di solo shell sulla cache HF.
        if ! _hf_cache_has_any_snapshot; then
            _artifact_fail "Cache HF modelli" "${HF_HUB_CACHE:-$T2G_HF_HOME_DEFAULT/hub}" \
                "Nessuna snapshot modello (models--*/snapshots) trovata nella cache HF condivisa." \
                || return 1
        fi
        echo "   ✅ Cache HF contiene almeno una snapshot modello (verifica shell)"
    fi

    # ── 5. Backend retrieval minilm: NESSUN download di SentenceTransformer ──
    if [ "$ret_enabled" = "1" ] && [ "$ret_backend" = "minilm" ]; then
        local minilm_id="$ret_model"
        [ -n "$minilm_id" ] || minilm_id="$T2G_MINILM_DEFAULT"
        case "$minilm_source" in
            LOCAL)
                echo "   ✅ MiniLM (path locale): $minilm_id"
                ;;
            MISSING|UNKNOWN|"")
                if ! _hf_model_cached_shell "$minilm_id"; then
                    _artifact_fail "Modello SentenceTransformer (backend retrieval minilm)" "$minilm_id" \
                        "backend=minilm richiede la snapshot già in cache HF (o un model_name locale): NON viene scaricata dai compute node." \
                        || return 1
                fi
                echo "   ✅ MiniLM in cache HF (verifica shell): $minilm_id"
                ;;
            *)
                echo "   ✅ MiniLM in cache HF: $minilm_source"
                ;;
        esac
        # L'indice retriever (data/retriever_index) NON è richiesto: si
        # ricostruisce offline dal dataset cachato (meta mismatch → rebuild).
        echo "   ℹ️  Indice retriever: ricostruzione offline consentita (non richiesto su disco)."
    fi

    return 0
}
