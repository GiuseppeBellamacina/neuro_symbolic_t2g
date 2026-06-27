#!/bin/bash
# ============================================================================
# Watcher — Esegue i job dalla catena .job_chain uno alla volta.
#
# Gira sul login node (NON dentro un job SLURM). Controlla ogni 60s
# se la coda è vuota e sottomette il prossimo job.
#
# PID guard: esci se esiste già un watcher attivo
PROJ_DIR="$HOME/neuro_symbolic_t2g"
CHAIN_PID_FILE="$PROJ_DIR/.chain_pid"
if [ -f "$CHAIN_PID_FILE" ]; then
    OLD_PID=$(cat "$CHAIN_PID_FILE")
    if [ -n "$OLD_PID" ] && kill -0 "$OLD_PID" 2>/dev/null; then
        echo "[chain] ❌ Watcher già attivo con PID $OLD_PID — esco."
        exit 1
    fi
fi

#
# Se un job SLURM fallisce (exit code != 0, CANCELLED), il watcher si
# blocca, scrive .chain_failed con le info del job, e NON sottomette
# altri job. Usare run_all.sh --resume per riprendere.
#
# Se un job di TRAINING va in TIMEOUT (tempo limite superato), il watcher
# lo rilancia automaticamente con --resume (max 2 tentativi). Se anche
# il retry va in timeout, la pipeline si ferma normalmente.
#
# Uso:
#   nohup bash src/cluster/chain_next.sh >> logs/chain_watcher.log 2>&1 &
#
# Per interrompere: kill $(cat .chain_pid), oppure cancella .job_chain
# ============================================================================

PROJ_DIR="$HOME/neuro_symbolic_t2g"
CHAIN_FILE="$PROJ_DIR/.job_chain"
FAILED_FILE="$PROJ_DIR/.chain_failed"
ERRORS_FILE="$PROJ_DIR/.chain_errors"
POLL_INTERVAL=60  # secondi tra un check e l'altro
MAX_TIMEOUT_RETRIES=2  # max auto-resume per TIMEOUT sullo stesso job
MAX_OOM_RETRIES=2      # max auto-resume per OOM sullo stesso job
MAX_CUDA_RETRIES=2     # max auto-resume per CUDA transient errors
GPU_UTIL_DECREMENT="0.05"  # quanto ridurre gpu_memory_utilization ad ogni OOM

cd "$PROJ_DIR"

# ── Helpers ───────────────────────────────────────────────────────────────────

# Rileva OOM dal log SLURM di un job di training
is_oom_failure() {
    local job_id="$1" exit_code="$2" state="$3" tag="$4"
    [ "$state" = "OUT_OF_MEMORY" ] && return 0
    [ "$exit_code" = "137" ] && return 0
    local logfile="$PROJ_DIR/logs/slurm-train-${job_id}.log"
    if [ -f "$logfile" ]; then
        if tail -200 "$logfile" | grep -qiE 'out.of.memory|OutOfMemoryError|CUDA out of memory|oom-kill|OOM|torch.cuda.OutOfMemoryError|std::bad_alloc|excessive GPU RAM|GPU RAM usage'; then
            return 0
        fi
    fi
    return 1
}

# Rileva CUDA errors transitori
is_cuda_transient_failure() {
    local job_id="$1" exit_code="$2" state="$3" tag="$4"
    local logfile="$PROJ_DIR/logs/slurm-train-${job_id}.log"
    if [ -f "$logfile" ]; then
        if tail -200 "$logfile" | grep -qiE 'cudaErrorIllegalAddress|illegal memory access|cudaErrorLaunchFailure|device-side assert|AcceleratorError.*CUDA error'; then
            return 0
        fi
    fi
    return 1
}

# Logga un errore nel file persistente .chain_errors (formato JSONL)
log_job_error() {
    local job_id="$1" job_type="$2" config="$3" tag="$4"
    local state="$5" exit_code="$6" error_type="$7"
    local retry_num="${8:-0}" resolved="${9:-false}"
    local logfile="$PROJ_DIR/logs/slurm-${job_type}-${job_id}.log"

    python3 -c "
import json, datetime, os, re
logfile = '${logfile}'
snippet = ''
if os.path.isfile(logfile):
    with open(logfile, errors='replace') as f:
        lines = f.readlines()
    tail = lines[-200:]
    keywords = ['error', 'cuda', 'oom', 'traceback', 'exception',
                'illegal', 'killed', 'out of memory', 'sigkill',
                'acceleratorerror', 'device-side assert']
    err_lines = [l.strip() for l in tail
                 if any(w in l.lower() for w in keywords)]
    snippet = ' | '.join(err_lines[-5:])[:800]

entry = {
    'tag': '${tag}',
    'job_type': '${job_type}',
    'slurm_id': '${job_id}',
    'config': '${config}',
    'error_type': '${error_type}',
    'slurm_state': '${state}',
    'exit_code': '${exit_code}',
    'timestamp': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    'error_snippet': snippet,
    'retry_num': int('${retry_num}' or '0'),
    'resolved': '${resolved}' == 'true'
}
with open('${ERRORS_FILE}', 'a') as f:
    f.write(json.dumps(entry, ensure_ascii=False) + '\n')
" 2>/dev/null
    echo "[chain] 📝 Errore registrato in .chain_errors (${error_type}, ${tag}, job ${job_id})"
}

# Query sacct con retry
query_sacct_with_retry() {
    local job_id="$1"
    local max_attempts=6
    local wait_secs=5
    _SACCT_STATE=""
    _SACCT_EXIT_CODE=""

    for attempt in $(seq 1 $max_attempts); do
        _SACCT_EXIT_CODE=$(sacct -j "$job_id" --format=ExitCode --noheader --parsable2 2>/dev/null | head -1 | cut -d: -f1)
        _SACCT_STATE=$(sacct -j "$job_id" --format=State --noheader --parsable2 2>/dev/null | head -1)
        _SACCT_STATE=$(echo "$_SACCT_STATE" | tr -d '[:space:]')
        _SACCT_EXIT_CODE=$(echo "$_SACCT_EXIT_CODE" | tr -d '[:space:]')

        case "$_SACCT_STATE" in
            COMPLETED|FAILED|CANCELLED|CANCELLED+|TIMEOUT|OUT_OF_MEMORY|NODE_FAIL)
                return 0
                ;;
        esac

        if [ "$attempt" -lt "$max_attempts" ]; then
            echo "[chain] ⏳ sacct per job $job_id: stato='$_SACCT_STATE' — attendo ${wait_secs}s (tentativo $attempt/$max_attempts)"
            sleep "$wait_secs"
        fi
    done

    if [ -z "$_SACCT_STATE" ]; then
        echo "[chain] ⚠️  sacct non ha restituito stato per job $job_id dopo $max_attempts tentativi"
        _SACCT_STATE="UNKNOWN"
        return 1
    fi
    return 0
}

# Riduce gpu_memory_utilization nel YAML
reduce_gpu_memory_util() {
    local cfg="$1"
    local cfg_path="$PROJ_DIR/$cfg"
    [ -f "$cfg_path" ] || return 1

    local current
    current=$(python3 -c "
import yaml, sys
cfg = yaml.safe_load(open('${cfg_path}'))
v = cfg.get('model', {}).get('gpu_memory_utilization', 0.9)
print(f'{v:.4f}')
" 2>/dev/null)
    [ -z "$current" ] && return 1

    local new_val
    new_val=$(python3 -c "
v = $current - $GPU_UTIL_DECREMENT
if v < 0.10:
    print('TOO_LOW')
else:
    print(f'{v:.2f}')
" 2>/dev/null)

    if [ "$new_val" = "TOO_LOW" ] || [ -z "$new_val" ]; then
        echo "[chain] ⚠️  gpu_memory_utilization=$current già troppo basso, non riduco"
        return 1
    fi

    sed -i "s/gpu_memory_utilization: *[0-9.]\+/gpu_memory_utilization: $new_val/" "$cfg_path"
    echo "[chain] 🔧 gpu_memory_utilization: $current → $new_val in $cfg"
    return 0
}

echo $$ > "$PROJ_DIR/.chain_pid"

echo "[chain] Watcher avviato (PID $$) — $(date)"
echo "[chain] File catena: $CHAIN_FILE"

LAST_JOB_ID=""
LAST_JOB_DESC=""
TIMEOUT_RETRIES=0
LAST_RETRY_TAG=""
OOM_RETRIES=0
LAST_OOM_TAG=""
CUDA_RETRIES=0
LAST_CUDA_TAG=""
SBATCH_RETRIES=0
MAX_SBATCH_RETRIES=5
SBATCH_RETRY_WAIT=60

while true; do
    if [ ! -f "$CHAIN_FILE" ] || [ ! -s "$CHAIN_FILE" ]; then
        if [ -n "$LAST_JOB_ID" ]; then
            query_sacct_with_retry "$LAST_JOB_ID"
            EXIT_CODE="$_SACCT_EXIT_CODE"
            STATE="$_SACCT_STATE"

            FINAL_FAILED=0
            if [ "$STATE" = "TIMEOUT" ] || [ "$STATE" = "CANCELLED" ] || [ "$STATE" = "CANCELLED+" ] || [ "$STATE" = "UNKNOWN" ]; then
                FINAL_FAILED=1
            elif [ -n "$EXIT_CODE" ] && [ "$EXIT_CODE" != "0" ]; then
                FINAL_FAILED=1
            fi

            if [ "$FINAL_FAILED" -eq 1 ]; then
                FINAL_TYPE=$(echo "$LAST_JOB_DESC" | cut -d: -f1)
                FINAL_CFG=$(echo "$LAST_JOB_DESC" | cut -d: -f2)
                FINAL_TAG=$(echo "$LAST_JOB_DESC" | cut -d: -f3)

                if [ "$STATE" = "TIMEOUT" ] && [ "$FINAL_TYPE" = "train" ]; then
                    if [ "$FINAL_TAG" = "$LAST_RETRY_TAG" ]; then
                        TIMEOUT_RETRIES=$((TIMEOUT_RETRIES + 1))
                    else
                        TIMEOUT_RETRIES=1
                        LAST_RETRY_TAG="$FINAL_TAG"
                    fi
                    if [ "$TIMEOUT_RETRIES" -le "$MAX_TIMEOUT_RETRIES" ]; then
                        echo "[chain] ⏰ Ultimo job $LAST_JOB_ID ($FINAL_TAG) TIMEOUT — auto-resume ($TIMEOUT_RETRIES/$MAX_TIMEOUT_RETRIES) — $(date)"
                        log_job_error "$LAST_JOB_ID" "$FINAL_TYPE" "$FINAL_CFG" "$FINAL_TAG" "$STATE" "$EXIT_CODE" "TIMEOUT" "$TIMEOUT_RETRIES" "true"
                        echo "train:${FINAL_CFG}:${FINAL_TAG}:--resume" > "$CHAIN_FILE"
                        LAST_JOB_ID=""
                        sleep 5
                        continue
                    fi
                elif is_oom_failure "$LAST_JOB_ID" "$EXIT_CODE" "$STATE" "$FINAL_TAG" && [ "$FINAL_TYPE" = "train" ]; then
                    if [ "$FINAL_TAG" = "$LAST_OOM_TAG" ]; then
                        OOM_RETRIES=$((OOM_RETRIES + 1))
                    else
                        OOM_RETRIES=1
                        LAST_OOM_TAG="$FINAL_TAG"
                    fi
                    if [ "$OOM_RETRIES" -le "$MAX_OOM_RETRIES" ]; then
                        echo "[chain] 💥 Ultimo job $LAST_JOB_ID ($FINAL_TAG) OOM — retry ($OOM_RETRIES/$MAX_OOM_RETRIES) — $(date)"
                        if reduce_gpu_memory_util "$FINAL_CFG"; then
                            log_job_error "$LAST_JOB_ID" "$FINAL_TYPE" "$FINAL_CFG" "$FINAL_TAG" "$STATE" "$EXIT_CODE" "OOM" "$OOM_RETRIES" "true"
                            echo "train:${FINAL_CFG}:${FINAL_TAG}:--resume" > "$CHAIN_FILE"
                            LAST_JOB_ID=""
                            sleep 5
                            continue
                        else
                            echo "[chain] ❌ gpu_memory_utilization non riducibile — stop"
                            log_job_error "$LAST_JOB_ID" "$FINAL_TYPE" "$FINAL_CFG" "$FINAL_TAG" "$STATE" "$EXIT_CODE" "OOM" "$OOM_RETRIES" "false"
                        fi
                    fi
                elif is_cuda_transient_failure "$LAST_JOB_ID" "$EXIT_CODE" "$STATE" "$FINAL_TAG" && [ "$FINAL_TYPE" = "train" ]; then
                    if [ "$FINAL_TAG" = "$LAST_CUDA_TAG" ]; then
                        CUDA_RETRIES=$((CUDA_RETRIES + 1))
                    else
                        CUDA_RETRIES=1
                        LAST_CUDA_TAG="$FINAL_TAG"
                    fi
                    if [ "$CUDA_RETRIES" -le "$MAX_CUDA_RETRIES" ]; then
                        echo "[chain] ⚡ Ultimo job $LAST_JOB_ID ($FINAL_TAG) CUDA transient error — auto-resume ($CUDA_RETRIES/$MAX_CUDA_RETRIES) — $(date)"
                        log_job_error "$LAST_JOB_ID" "$FINAL_TYPE" "$FINAL_CFG" "$FINAL_TAG" "$STATE" "$EXIT_CODE" "CUDA_ERROR" "$CUDA_RETRIES" "true"
                        echo "train:${FINAL_CFG}:${FINAL_TAG}:--resume" > "$CHAIN_FILE"
                        LAST_JOB_ID=""
                        sleep 5
                        continue
                    fi
                fi

                _ERR_TYPE="UNKNOWN"
                [ "$STATE" = "TIMEOUT" ] && _ERR_TYPE="TIMEOUT"
                [ "$STATE" = "CANCELLED" ] || [ "$STATE" = "CANCELLED+" ] && _ERR_TYPE="CANCELLED"
                is_oom_failure "$LAST_JOB_ID" "$EXIT_CODE" "$STATE" "$FINAL_TAG" 2>/dev/null && _ERR_TYPE="OOM"
                is_cuda_transient_failure "$LAST_JOB_ID" "$EXIT_CODE" "$STATE" "$FINAL_TAG" 2>/dev/null && _ERR_TYPE="CUDA_ERROR"
                log_job_error "$LAST_JOB_ID" "$FINAL_TYPE" "$FINAL_CFG" "$FINAL_TAG" "$STATE" "$EXIT_CODE" "$_ERR_TYPE" "0" "false"

                echo "[chain] ❌ Ultimo job $LAST_JOB_ID ($LAST_JOB_DESC) FALLITO (state=$STATE exit=$EXIT_CODE) — $(date)"
                echo "${LAST_JOB_DESC}" > "$FAILED_FILE"
                echo "[chain] Pipeline interrotta. Usa: bash src/cluster/run_all.sh --resume"
                rm -f "$PROJ_DIR/.chain_pid"
                exit 1
            fi
        fi
        echo "[chain] ✅ Pipeline completata! — $(date)"
        rm -f "$CHAIN_FILE" "$PROJ_DIR/.chain_pid" "$FAILED_FILE"
        exit 0
    fi

    ACTIVE=$(squeue --me --noheader 2>/dev/null | wc -l)
    if [ "$ACTIVE" -gt 0 ]; then
        sleep "$POLL_INTERVAL"
        continue
    fi

    # ── Coda vuota: controlla se l'ultimo job è fallito ───────────────────
    if [ -n "$LAST_JOB_ID" ]; then
        query_sacct_with_retry "$LAST_JOB_ID"
        EXIT_CODE="$_SACCT_EXIT_CODE"
        STATE="$_SACCT_STATE"

        JOB_FAILED=0
        IS_TIMEOUT=0
        if [ "$STATE" = "TIMEOUT" ]; then
            JOB_FAILED=1
            IS_TIMEOUT=1
        elif [ "$STATE" = "CANCELLED" ] || [ "$STATE" = "CANCELLED+" ] || [ "$STATE" = "UNKNOWN" ]; then
            JOB_FAILED=1
        elif [ -n "$EXIT_CODE" ] && [ "$EXIT_CODE" != "0" ]; then
            JOB_FAILED=1
        fi

        if [ "$JOB_FAILED" -eq 1 ]; then
            FAILED_TYPE=$(echo "$LAST_JOB_DESC" | cut -d: -f1)
            FAILED_CFG=$(echo "$LAST_JOB_DESC" | cut -d: -f2)
            FAILED_TAG=$(echo "$LAST_JOB_DESC" | cut -d: -f3)

            if [ "$IS_TIMEOUT" -eq 1 ] && [ "$FAILED_TYPE" = "train" ]; then
                if [ "$FAILED_TAG" = "$LAST_RETRY_TAG" ]; then
                    TIMEOUT_RETRIES=$((TIMEOUT_RETRIES + 1))
                else
                    TIMEOUT_RETRIES=1
                    LAST_RETRY_TAG="$FAILED_TAG"
                fi
                if [ "$TIMEOUT_RETRIES" -le "$MAX_TIMEOUT_RETRIES" ]; then
                    echo "[chain] ⏰ Job $LAST_JOB_ID ($FAILED_TAG) TIMEOUT — auto-resume ($TIMEOUT_RETRIES/$MAX_TIMEOUT_RETRIES) — $(date)"
                    log_job_error "$LAST_JOB_ID" "$FAILED_TYPE" "$FAILED_CFG" "$FAILED_TAG" "$STATE" "$EXIT_CODE" "TIMEOUT" "$TIMEOUT_RETRIES" "true"
                    RESUME_CHAIN=$(mktemp)
                    echo "train:${FAILED_CFG}:${FAILED_TAG}:--resume" > "$RESUME_CHAIN"
                    [ -f "$CHAIN_FILE" ] && [ -s "$CHAIN_FILE" ] && cat "$CHAIN_FILE" >> "$RESUME_CHAIN"
                    mv "$RESUME_CHAIN" "$CHAIN_FILE"
                    LAST_JOB_ID=""
                    sleep 5
                    continue
                else
                    echo "[chain] ❌ Job $LAST_JOB_ID ($FAILED_TAG) TIMEOUT — max retry ($MAX_TIMEOUT_RETRIES) raggiunto — $(date)"
                fi
            elif is_oom_failure "$LAST_JOB_ID" "$EXIT_CODE" "$STATE" "$FAILED_TAG" && [ "$FAILED_TYPE" = "train" ]; then
                if [ "$FAILED_TAG" = "$LAST_OOM_TAG" ]; then
                    OOM_RETRIES=$((OOM_RETRIES + 1))
                else
                    OOM_RETRIES=1
                    LAST_OOM_TAG="$FAILED_TAG"
                fi
                if [ "$OOM_RETRIES" -le "$MAX_OOM_RETRIES" ]; then
                    echo "[chain] 💥 Job $LAST_JOB_ID ($FAILED_TAG) OOM — retry ($OOM_RETRIES/$MAX_OOM_RETRIES) — $(date)"
                    if reduce_gpu_memory_util "$FAILED_CFG"; then
                        log_job_error "$LAST_JOB_ID" "$FAILED_TYPE" "$FAILED_CFG" "$FAILED_TAG" "$STATE" "$EXIT_CODE" "OOM" "$OOM_RETRIES" "true"
                        RESUME_CHAIN=$(mktemp)
                        echo "train:${FAILED_CFG}:${FAILED_TAG}:--resume" > "$RESUME_CHAIN"
                        [ -f "$CHAIN_FILE" ] && [ -s "$CHAIN_FILE" ] && cat "$CHAIN_FILE" >> "$RESUME_CHAIN"
                        mv "$RESUME_CHAIN" "$CHAIN_FILE"
                        LAST_JOB_ID=""
                        sleep 5
                        continue
                    else
                        echo "[chain] ❌ gpu_memory_utilization non riducibile — stop"
                        log_job_error "$LAST_JOB_ID" "$FAILED_TYPE" "$FAILED_CFG" "$FAILED_TAG" "$STATE" "$EXIT_CODE" "OOM" "$OOM_RETRIES" "false"
                    fi
                else
                    echo "[chain] ❌ Job $LAST_JOB_ID ($FAILED_TAG) OOM — max retry ($MAX_OOM_RETRIES) raggiunto — $(date)"
                fi
            elif is_cuda_transient_failure "$LAST_JOB_ID" "$EXIT_CODE" "$STATE" "$FAILED_TAG" && [ "$FAILED_TYPE" = "train" ]; then
                if [ "$FAILED_TAG" = "$LAST_CUDA_TAG" ]; then
                    CUDA_RETRIES=$((CUDA_RETRIES + 1))
                else
                    CUDA_RETRIES=1
                    LAST_CUDA_TAG="$FAILED_TAG"
                fi
                if [ "$CUDA_RETRIES" -le "$MAX_CUDA_RETRIES" ]; then
                    echo "[chain] ⚡ Job $LAST_JOB_ID ($FAILED_TAG) CUDA transient error — auto-resume ($CUDA_RETRIES/$MAX_CUDA_RETRIES) — $(date)"
                    log_job_error "$LAST_JOB_ID" "$FAILED_TYPE" "$FAILED_CFG" "$FAILED_TAG" "$STATE" "$EXIT_CODE" "CUDA_ERROR" "$CUDA_RETRIES" "true"
                    RESUME_CHAIN=$(mktemp)
                    echo "train:${FAILED_CFG}:${FAILED_TAG}:--resume" > "$RESUME_CHAIN"
                    [ -f "$CHAIN_FILE" ] && [ -s "$CHAIN_FILE" ] && cat "$CHAIN_FILE" >> "$RESUME_CHAIN"
                    mv "$RESUME_CHAIN" "$CHAIN_FILE"
                    LAST_JOB_ID=""
                    sleep 5
                    continue
                else
                    echo "[chain] ❌ Job $LAST_JOB_ID ($FAILED_TAG) CUDA error — max retry ($MAX_CUDA_RETRIES) raggiunto — $(date)"
                fi
            else
                echo "[chain] ❌ Job $LAST_JOB_ID ($LAST_JOB_DESC) FALLITO — state=$STATE exit=$EXIT_CODE — $(date)"
            fi

            _ERR_TYPE="UNKNOWN"
            [ "$STATE" = "TIMEOUT" ] && _ERR_TYPE="TIMEOUT"
            if [ "$STATE" = "CANCELLED" ] || [ "$STATE" = "CANCELLED+" ]; then _ERR_TYPE="CANCELLED"; fi
            is_oom_failure "$LAST_JOB_ID" "$EXIT_CODE" "$STATE" "$FAILED_TAG" 2>/dev/null && _ERR_TYPE="OOM"
            is_cuda_transient_failure "$LAST_JOB_ID" "$EXIT_CODE" "$STATE" "$FAILED_TAG" 2>/dev/null && _ERR_TYPE="CUDA_ERROR"
            log_job_error "$LAST_JOB_ID" "$FAILED_TYPE" "$FAILED_CFG" "$FAILED_TAG" "$STATE" "$EXIT_CODE" "$_ERR_TYPE" "0" "false"

            if [ "$FAILED_TYPE" = "train" ] && [ -f "$CHAIN_FILE" ] && [ -s "$CHAIN_FILE" ]; then
                NEXT_IN_CHAIN=$(head -1 "$CHAIN_FILE")
                NEXT_TYPE=$(echo "$NEXT_IN_CHAIN" | cut -d: -f1)
                NEXT_TAG=$(echo "$NEXT_IN_CHAIN" | cut -d: -f3)
                if [ "$NEXT_TYPE" = "eval" ] && [ "$NEXT_TAG" = "$FAILED_TAG" ]; then
                    echo "[chain] ⏭  Rimosso eval di $FAILED_TAG dalla catena (train fallito)"
                    tail -n +2 "$CHAIN_FILE" > "$CHAIN_FILE.tmp" && mv "$CHAIN_FILE.tmp" "$CHAIN_FILE"
                    [ ! -s "$CHAIN_FILE" ] && rm -f "$CHAIN_FILE"
                fi
            fi

            echo "${LAST_JOB_DESC}" > "$FAILED_FILE"
            REMAINING=$([ -f "$CHAIN_FILE" ] && wc -l < "$CHAIN_FILE" || echo 0)
            echo "[chain] Pipeline interrotta. Rimanenti: $REMAINING job"
            echo "[chain] Per riprendere: bash src/cluster/run_all.sh --resume"
            rm -f "$PROJ_DIR/.chain_pid"
            exit 1
        fi
        echo "[chain] ✓ Job $LAST_JOB_ID ($LAST_JOB_DESC) completato — state=$STATE — $(date)"

        TIMEOUT_RETRIES=0
        LAST_RETRY_TAG=""
        OOM_RETRIES=0
        LAST_OOM_TAG=""
        CUDA_RETRIES=0
        LAST_CUDA_TAG=""
    fi

    # ── Sottometti il prossimo job ────────────────────────────────────────
    NEXT=$(head -1 "$CHAIN_FILE")
    tail -n +2 "$CHAIN_FILE" > "$CHAIN_FILE.tmp" && mv "$CHAIN_FILE.tmp" "$CHAIN_FILE"
    if [ ! -s "$CHAIN_FILE" ]; then
        rm -f "$CHAIN_FILE"
    fi

    TYPE=$(echo "$NEXT" | cut -d: -f1)
    CFG=$(echo "$NEXT" | cut -d: -f2)
    TAG=$(echo "$NEXT" | cut -d: -f3)
    EXTRA=$(echo "$NEXT" | cut -d: -f4-)

    if [ -z "$CFG" ]; then
        echo "[chain] ❌ Config vuoto per $TYPE $TAG — catena corrotta"
        echo "${NEXT}" > "$FAILED_FILE"
        echo "[chain] Pipeline interrotta. Per riprendere: bash src/cluster/run_all.sh --resume"
        rm -f "$PROJ_DIR/.chain_pid"
        exit 1
    fi

    REMAINING=$([ -f "$CHAIN_FILE" ] && wc -l < "$CHAIN_FILE" || echo 0)
    echo "[chain] Sottometto: $TYPE $TAG ($CFG) extra='$EXTRA' — $REMAINING rimanenti — $(date)"

    LAST_JOB_DESC="$NEXT"

    case "$TYPE" in
        train)
            LAST_JOB_ID=$(CONFIG="$CFG" EXTRA_ARGS="$EXTRA" sbatch --job-name="train-${TAG}" --parsable src/cluster/train.sh 2>&1 | grep -oP '^\d+$')
            ;;
        eval)
            SKIP_N=0
            if echo "$EXTRA" | grep -qP '^--skip-stages=\d+$'; then
                SKIP_N=$(echo "$EXTRA" | grep -oP '\d+')
            fi
            LAST_JOB_ID=$(CONFIG="$CFG" SKIP_STAGES="$SKIP_N" sbatch --job-name="eval-${TAG}" --parsable src/cluster/eval.sh 2>&1 | grep -oP '^\d+$')
            ;;
        *)
            echo "[chain] ❌ Tipo sconosciuto: $TYPE — skip"
            LAST_JOB_ID=""
            continue
            ;;
    esac

    if [ -z "$LAST_JOB_ID" ]; then
        SBATCH_RETRIES=$((SBATCH_RETRIES + 1))
        echo "[chain] ⚠️  sbatch non ha restituito un job ID per $TYPE $TAG — retry $SBATCH_RETRIES/$MAX_SBATCH_RETRIES — $(date)"
        if [ "$SBATCH_RETRIES" -gt "$MAX_SBATCH_RETRIES" ]; then
            echo "[chain] ❌ sbatch fallito $MAX_SBATCH_RETRIES volte consecutive — pipeline interrotta"
            echo "${NEXT}" > "$FAILED_FILE"
            echo "[chain] Per riprendere: bash src/cluster/run_all.sh --resume"
            rm -f "$PROJ_DIR/.chain_pid"
            exit 1
        fi
        REINSERTION=$(mktemp)
        echo "$NEXT" > "$REINSERTION"
        [ -f "$CHAIN_FILE" ] && [ -s "$CHAIN_FILE" ] && cat "$CHAIN_FILE" >> "$REINSERTION"
        mv "$REINSERTION" "$CHAIN_FILE"
        LAST_JOB_ID=""
        echo "[chain] ⏳ Entry reinserita in catena, riprovo tra ${SBATCH_RETRY_WAIT}s"
        sleep "$SBATCH_RETRY_WAIT"
        continue
    fi

    SBATCH_RETRIES=0
    echo "[chain] Job ID: $LAST_JOB_ID"
    sleep 10
done
