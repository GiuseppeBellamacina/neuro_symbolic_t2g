#!/bin/bash
# ============================================================================
# chain_next.sh — Watcher FALLBACK per la pipeline T2G (non più primario).
#
# La modalità primaria è il tick one-shot (chain_tick.sh) schedulato via
# `at` (chain_tick.sh --schedule) o dal bashrc-hook (chain-hook-install).
# Questo watcher resta come fallback quando `at` non è disponibile.
#
# VINCOLO REAPER: il login node uccide i processi long-lived (ipotesi
# principale: systemd KillUserProcesses al logout — setsid/nohup/trap '' non
# sopravvivono a SIGKILL). Quindi il watcher è BEST-EFFORT: se muore, la coda
# resta su disco (.chain_state/job_chain) e chain-resume la riprende.
#
# PID guard: esce se esiste già un watcher attivo.
# Poll adattivo: 60s per i primi 10 minuti dopo una sottomissione, poi 300s
# (meno CPU → minor probabilità di reaping per uso risorse).
# nice -n 19: priorità minima, stesso motivo.
# Retry: TIMEOUT/OOM/CUDA su train auto-ripresi (max 2), continue-on-failure
# per errori non retryable (vedi _lib.sh::chain_handle_failure).
#
# Uso:
#   setsid nohup nice -n 19 bash cluster/chain_next.sh >> logs/chain_watcher.log 2>&1 &
# ============================================================================

set -uo pipefail  # NO set -e: loop con error handling esplicito

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=cluster/_lib.sh
source "$SCRIPT_DIR/_lib.sh"

# ── nice: priorità minima (minor profilo risorse → meno reaping) ──
if [ -z "${_CHAIN_NEXT_NICED:-}" ]; then
    export _CHAIN_NEXT_NICED=1
    exec nice -n 19 bash "$SCRIPT_DIR/chain_next.sh" "$@"
fi

cd "$PROJ_DIR"
mkdir -p "$STATE_DIR" logs

# ── PID guard ──
if ! pid_guard; then
    exit 1
fi
echo $$ > "$CHAIN_PID_FILE"

# Ignore SIGHUP/SIGTERM (best effort — il reaper può comunque SIGKILL).
trap '' SIGHUP SIGTERM

echo "[chain] Watcher FALLBACK avviato (PID $$) — $(date)"
echo "[chain] Coda: $CHAIN_FILE"
echo "[chain] Modalità: watcher (at non disponibile). Suggerimento: chain-hook-install"

POLL_FAST=60
POLL_SLOW=300
POLL_FAST_WINDOW=600   # 10 minuti dopo una sottomissione
POLL_INTERVAL=$POLL_FAST
MAX_SBATCH_RETRIES=5
SBATCH_RETRIES=0
SBATCH_RETRY_WAIT=60
LAST_SUBMIT_TS=0

while true; do
    touch "$HEARTBEAT_FILE"

    # Pausa?
    if [ -f "$STOPPED_FILE" ]; then
        echo "[chain] chain_stopped presente — watcher esce — $(date)"
        rm -f "$CHAIN_PID_FILE"
        exit 0
    fi

    chain_read_last_job

    # ── Coda vuota → verifica finale sull'ultimo job ──
    if [ ! -s "$CHAIN_FILE" ]; then
        if [ -n "$LAST_JOB_ID" ]; then
            query_sacct_with_retry "$LAST_JOB_ID" 6 5 || true
            if last_job_still_active; then
                echo "[chain] ⏳ ultimo job $LAST_JOB_ID ancora $_SACCT_STATE — back to sleep — $(date)"
                sleep "$POLL_INTERVAL"
                continue
            fi
            if ! job_succeeded; then
                if chain_handle_failure; then
                    # retry reinserito in coda → sottometti subito
                    if chain_submit_next; then
                        LAST_SUBMIT_TS=$(date +%s)
                        sleep 10
                        continue
                    fi
                    sleep "$SBATCH_RETRY_WAIT"
                    continue
                fi
                echo "[chain] ⚠️  Pipeline completata CON ERRORI — vedi .chain_state/chain_errors — $(date)"
                rm -f "$CHAIN_FILE" "$CHAIN_PID_FILE" "$FAILED_FILE"
                exit 0
            fi
        fi
        echo "[chain] ✅ Pipeline completata! — $(date)"
        rm -f "$CHAIN_FILE" "$CHAIN_PID_FILE" "$FAILED_FILE"
        exit 0
    fi

    # ── Job attivo? → dormi ──
    if [ -n "$(active_job_id)" ]; then
        sleep "$POLL_INTERVAL"
        continue
    fi

    # ── Ultimo job fallito? → retry o log-and-skip (poi si sottomette) ──
    if [ -n "$LAST_JOB_ID" ]; then
        query_sacct_with_retry "$LAST_JOB_ID" 6 5 || true
        if last_job_still_active; then
            echo "[chain] ⏳ job $LAST_JOB_ID ancora $_SACCT_STATE (squeue non lo vede) — $(date)"
            sleep "$POLL_INTERVAL"
            continue
        fi
        if ! job_succeeded; then
            if ! chain_handle_failure; then
                echo "[chain] ⚠️  coda svuotata dopo gestione errore — completata con errori — $(date)"
                rm -f "$CHAIN_FILE" "$CHAIN_PID_FILE" "$FAILED_FILE"
                exit 0
            fi
            # fall-through: retry reinserito o eval saltato → sottometti
        fi
    fi

    # ── Sottometti il prossimo ──
    if ! chain_submit_next; then
        SBATCH_RETRIES=$((SBATCH_RETRIES + 1))
        echo "[chain] ⚠️  sbatch fallito — retry $SBATCH_RETRIES/$MAX_SBATCH_RETRIES — $(date)"
        if [ "$SBATCH_RETRIES" -gt "$MAX_SBATCH_RETRIES" ]; then
            next_entry=$(head -1 "$CHAIN_FILE" 2>/dev/null || true)
            [ -n "$next_entry" ] && echo "$next_entry" > "$FAILED_FILE"
            echo "[chain] ❌ sbatch fallito $MAX_SBATCH_RETRIES volte consecutive — pipeline interrotta — $(date)"
            echo "[chain] Per riprendere: chain-resume"
            rm -f "$CHAIN_PID_FILE"
            exit 1
        fi
        sleep "$SBATCH_RETRY_WAIT"
        continue
    fi
    SBATCH_RETRIES=0
    LAST_SUBMIT_TS=$(date +%s)

    # Poll adattivo: veloce per 10 min dopo la sottomissione, poi lento
    if [ $(( $(date +%s) - LAST_SUBMIT_TS )) -lt "$POLL_FAST_WINDOW" ]; then
        POLL_INTERVAL=$POLL_FAST
    else
        POLL_INTERVAL=$POLL_SLOW
    fi
    echo "[chain] ✓ sottoposto job $LAST_JOB_ID — prossimo poll tra ${POLL_INTERVAL}s — $(date)"
    sleep 10
done
