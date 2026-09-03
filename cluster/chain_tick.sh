#!/bin/bash
# ============================================================================
# chain_tick.sh — One-shot, idempotent tick that advances the T2G pipeline.
#
# CORE della catena: tick BREVI e statelèssi, invocabili da:
#   - l'hook bashrc PROMPT_COMMAND      (chain-hook-install — PRIMARIO)
#   - il server esterno / cron         (remote/cluster_helper.sh tick)
#   - un comando manuale               (chain-start / run-all)
#
# Ogni tick:
#   1. flock (non-blocking): max 1 tick alla volta (hook+manuale+server)
#   2. .chain_stopped presente           → exit 0 (pausa)
#   3. active SLURM job (squeue)        → exit 0 (il tick è innocuo)
#   4. job_chain empty                  → exit 0 (catena completa)
#   5. last submitted job finished?     → classify via sacct: retry
#      (train TIMEOUT/OOM/CUDA, max 2, tracked in .chain_state/last_job)
#      or log-and-skip (continue-on-failure, eval-after-failed-train dropped)
#   6. submit the next job (sbatch), record .chain_state/last_job
#
# Exit codes: 0 = ok (o tick rimandato: pausa/job attivo/coda vuota/Slurm
#              non raggiungibile); 2 = usage error; 4 = errore interno
#              soft (loggato in chain.log, retry al prossimo tick — vedi
#              ERR trap: un tick NON allarma mai il driver, è idempotente).
#
# Uso:
#   bash cluster/chain_tick.sh                 # one-shot check
#   bash cluster/chain_tick.sh --quiet         # no stdout (hook/server)
# ============================================================================

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=cluster/_lib.sh
source "$SCRIPT_DIR/_lib.sh"
cd "$PROJ_DIR"

QUIET=0
for arg in "$@"; do
    case "$arg" in
        --quiet)    QUIET=1 ;;
        -h|--help)
            echo "Uso: bash cluster/chain_tick.sh [--quiet]"
            echo ""
            echo "  --quiet   nessun output su stdout (per hook bashrc / server)"
            echo ""
            echo "Exit codes: 0 ok · 2 usage"
            exit 0 ;;
        *)
            echo "❌ argomento sconosciuto: $arg" >&2
            exit 2 ;;
    esac
done

mkdir -p "$STATE_DIR" logs

# ── ERR trap: errore interno → soft-fail ─────────────────────────────────────
# Un blip Slurm/DNS NON deve mai allarmare il driver (ssh rc=3 → 502 →
# notifica rossa in TUI): il tick è idempotente e il prossimo (≈5 min)
# riprova. L'errore resta visibile in chain.log con riga e comando.
trap 'log_line "tick errore interno (riga $LINENO: $BASH_COMMAND) — rimandato al prossimo tick"; exit 4' ERR

# ── flock: un solo tick alla volta (hook + manuale + server possono sovrapporsi) ─
# Se `flock` (util-linux) manca sul login node, fallback a un lock mkdir con
# guardia anti-stale (un tick crashato non blocca per sempre la catena).
if command -v flock >/dev/null 2>&1; then
    exec 9>"$TICK_LOCK"
    if ! flock -n 9; then
        [ "$QUIET" -eq 0 ] && echo "tick già in esecuzione — skip"
        exit 0
    fi
else
    LOCK_DIR="$TICK_LOCK.dir"
    if ! mkdir "$LOCK_DIR" 2>/dev/null; then
        # lock esistente: se ha >10 min è stale (tick morto) → rimuovi e riprova
        if [ -d "$LOCK_DIR" ] && [ -n "$(find "$LOCK_DIR" -mmin +10 2>/dev/null)" ]; then
            rm -rf "$LOCK_DIR"
            mkdir "$LOCK_DIR" 2>/dev/null || { [ "$QUIET" -eq 0 ] && echo "tick già in esecuzione — skip"; exit 0; }
        else
            [ "$QUIET" -eq 0 ] && echo "tick già in esecuzione — skip"
            exit 0
        fi
    fi
    trap 'rmdir "$LOCK_DIR" 2>/dev/null || true' EXIT
fi

# 1) Pipeline in pausa
if [ -f "$STOPPED_FILE" ]; then
    [ "$QUIET" -eq 0 ] && echo "chain fermata (chain_stopped) — tick esce"
    exit 0
fi

# 2) Job attivo → via (il tick è innocuo: NON tocca la coda).
#    La QoS consente un solo job alla volta, quindi qui non si sottomette mai.
#    ROBUSTEZZA: se squeue NON risponde (blip Slurm/DNS del login node) il
#    tick è RIMANDATO (exit 0): trattare "squeue giù" come "nessun job
#    attivo" rischierebbe doppio submit (job pendente fuori QoS) o pop
#    della coda fuori ordine. Vedi evento 2026-09-03 13:41:44.
if ! JOB_ACTIVE=$(active_job_id); then
    log_line "tick: squeue non raggiungibile — tick rimandato"
    [ "$QUIET" -eq 0 ] && echo "squeue non raggiungibile — tick rimandato"
    exit 0
fi
if [ -n "$JOB_ACTIVE" ]; then
    [ "$QUIET" -eq 0 ] && echo "job attivo — tick innocuo"
    exit 0
fi

# 3) Catena completa
if [ ! -s "$CHAIN_FILE" ]; then
    rm -f "$FAILED_FILE"   # legacy marker non più necessario
    [ "$QUIET" -eq 0 ] && echo "catena completa — nessun job in coda"
    exit 0
fi

# 4) L'ultimo job è terminato? → retry o log-and-skip (poi si sottomette)
chain_read_last_job
if [ -n "$LAST_JOB_ID" ]; then
    query_sacct_with_retry "$LAST_JOB_ID" 4 3 || true
    if last_job_still_active; then
        [ "$QUIET" -eq 0 ] && echo "ultimo job ancora $_SACCT_STATE — tick innocuo"
        exit 0
    fi
    if ! job_succeeded; then
        if ! chain_handle_failure; then
            [ "$QUIET" -eq 0 ] && echo "catena terminata dopo gestione errore"
            exit 0
        fi
        # fall-through: retry reinserito o eval saltato → sottometti
    fi
fi

# 5) Sottometti il prossimo
if ! chain_submit_next; then
    [ "$QUIET" -eq 0 ] && echo "⚠️  sottomissione fallita — riproverà al prossimo tick"
    exit 0
fi

[ "$QUIET" -eq 0 ] && echo "tick ok — job $LAST_JOB_ID sottoposto, $(chain_remaining) in coda"
