#!/bin/bash
# ============================================================================
# Lancia training + evaluation per il modello T2G in catena.
#
# La QoS permette un solo job alla volta, quindi un watcher in background
# controlla ogni 60s se la coda è vuota e sottomette il prossimo job.
#
# Uso:
#   bash cluster/run_all.sh                       # train + eval
#   bash cluster/run_all.sh --eval-only            # solo evaluation
#   bash cluster/run_all.sh --train-only           # solo training
#   bash cluster/run_all.sh --resume               # riprendi pipeline fallita
#   bash cluster/run_all.sh --append               # aggiungi job a pipeline attiva
#   bash cluster/run_all.sh --remove               # rimuovi job dalla pipeline
#
# Monitorare:
#   tail -f logs/chain_watcher.log           # log del watcher
#   t2g-monitor                              # monitor live
#   myjobs                                   # job attivo su SLURM
#
# Interrompere:
#   kill $(cat .chain_pid)                   # uccidi il watcher
#   killalljobs                              # cancella anche il job SLURM attivo
# ============================================================================

set -e

# ── Parsing argomenti ─────────────────────────────────────────────────────────
GLOBAL_TRAIN=1
GLOBAL_EVAL=1
RESUME=0
APPEND=0
REMOVE=0
for arg in "$@"; do
    case "$arg" in
        --eval-only)   GLOBAL_TRAIN=0 ;;
        --train-only)  GLOBAL_EVAL=0 ;;
        --append)      APPEND=1 ;;
        --remove)      REMOVE=1 ;;
        --resume)      RESUME=1 ;;
        --help|-h)
            echo "Uso: bash cluster/run_all.sh [opzioni]"
            echo ""
            echo "Opzioni:"
            echo "  --eval-only      Solo evaluation (skip training)"
            echo "  --train-only     Solo training (skip eval)"
            echo "  --resume         Riprendi pipeline da dove si era fermata"
            echo "  --append         Aggiungi job alla pipeline attiva"
            echo "  --remove         Rimuovi job dalla pipeline attiva"
            exit 0
            ;;
    esac
done

# ── Modello T2G ───────────────────────────────────────────────────────────────
MODELS=("qwen05:experiments/configs/t2g/grpo_qwen05.yaml")

PROJ_DIR="$HOME/neuro_symbolic_t2g"
CHAIN_FILE="$PROJ_DIR/.job_chain"
FAILED_FILE="$PROJ_DIR/.chain_failed"

# ── Resume mode ───────────────────────────────────────────────────────────────
if [ "$RESUME" -eq 1 ]; then
    if [ ! -f "$FAILED_FILE" ]; then
        echo "❌ Nessun .chain_failed trovato. Non c'è nulla da riprendere."
        echo "   Usa: bash cluster/run_all.sh (senza --resume) per una nuova pipeline."
        exit 1
    fi

    FAILED_JOB=$(cat "$FAILED_FILE")
    FAILED_TYPE=$(echo "$FAILED_JOB" | cut -d: -f1)
    FAILED_CFG=$(echo "$FAILED_JOB" | cut -d: -f2)
    FAILED_TAG=$(echo "$FAILED_JOB" | cut -d: -f3)

    echo "============================================"
    echo "  RESUME Pipeline"
    echo "  Date:      $(date)"
    echo "  Failed job: $FAILED_TYPE $FAILED_TAG"
    echo "  Config:     $FAILED_CFG"
    echo "============================================"
    echo ""

    RESUME_CHAIN=$(mktemp)
    if [ "$FAILED_TYPE" = "train" ]; then
        echo "train:${FAILED_CFG}:${FAILED_TAG}:--resume" > "$RESUME_CHAIN"
        echo "eval:${FAILED_CFG}:${FAILED_TAG}" >> "$RESUME_CHAIN"
        echo "→ Training $FAILED_TAG verrà ripreso dall'ultimo checkpoint"
    else
        echo "eval:${FAILED_CFG}:${FAILED_TAG}" > "$RESUME_CHAIN"
        echo "→ Eval $FAILED_TAG verrà rieseguito da capo"
    fi

    if [ -f "$CHAIN_FILE" ] && [ -s "$CHAIN_FILE" ]; then
        cat "$CHAIN_FILE" >> "$RESUME_CHAIN"
    fi
    mv "$RESUME_CHAIN" "$CHAIN_FILE"
    rm -f "$FAILED_FILE"

    TOTAL=$(wc -l < "$CHAIN_FILE")
    echo ""
    echo "Catena ($TOTAL job):"
    cat -n "$CHAIN_FILE"
    echo ""

    if [ -f .chain_pid ]; then
        OLD_PID=$(cat .chain_pid)
        kill "$OLD_PID" 2>/dev/null && echo "Watcher precedente (PID $OLD_PID) terminato."
        rm -f .chain_pid
    fi

    nohup bash cluster/chain_next.sh >> logs/chain_watcher.log 2>&1 &
    WATCHER_PID=$!
    echo "$WATCHER_PID" > .chain_pid

    echo "============================================"
    echo "  Pipeline ripresa!"
    echo "  Watcher PID: $WATCHER_PID"
    echo "  Log: logs/chain_watcher.log"
    echo "============================================"
    exit 0
fi

# ── Funzione helper: controlla se il watcher è attivo ─────────────────────────
_watcher_is_alive() {
    if [ -f "$PROJ_DIR/.chain_pid" ]; then
        local pid=$(cat "$PROJ_DIR/.chain_pid")
        if ps -p "$pid" > /dev/null 2>&1; then
            return 0
        fi
    fi
    return 1
}

# ── Auto-detect: se watcher attivo e non --append/--remove esplicito ──────────
if [ "$APPEND" -eq 0 ] && [ "$REMOVE" -eq 0 ] && _watcher_is_alive; then
    echo "⚠️  Pipeline già attiva (watcher PID $(cat "$PROJ_DIR/.chain_pid"))."
    echo "   I nuovi job verranno AGGIUNTI alla coda esistente."
    echo "   (Usa Ctrl-C per annullare, oppure uccidi il watcher prima: t2g-watcher-kill)"
    echo ""
    APPEND=1
fi

# ── Remove mode ───────────────────────────────────────────────────────────────
if [ "$REMOVE" -eq 1 ]; then
    if [ ! -f "$CHAIN_FILE" ] || [ ! -s "$CHAIN_FILE" ]; then
        echo "❌ Nessuna catena attiva (.job_chain vuoto o non trovato)."
        exit 1
    fi

    echo "Catena attuale:"
    cat -n "$CHAIN_FILE"
    echo ""

    # Rimuovi tutto (catena T2G ha solo un modello)
    rm -f "$CHAIN_FILE"
    echo "✅ Catena svuotata."

    # Update monitor cache
    python3 -c "
import json, pathlib
cache_path = pathlib.Path('$PROJ_DIR/.monitor_cache')
if cache_path.exists():
    cache_path.unlink()
" 2>/dev/null

    exit 0
fi

# ── Costruisci la catena ──────────────────────────────────────────────────────
if [ "$APPEND" -eq 0 ]; then
    > "$CHAIN_FILE"  # svuota/crea il file
fi

EXISTING_ENTRIES=""
if [ "$APPEND" -eq 1 ] && [ -f "$CHAIN_FILE" ]; then
    EXISTING_ENTRIES=$(cat "$CHAIN_FILE")
fi

NEW_JOBS=0
NEW_KEYS=()
SKIPPED=0
for entry in "${MODELS[@]}"; do
    TAG=$(echo "$entry" | cut -d: -f1)
    CFG=$(echo "$entry" | cut -d: -f2)

    if [ "$GLOBAL_TRAIN" -eq 1 ]; then
        E="train:${CFG}:${TAG}"
        if [ "$APPEND" -eq 1 ] && echo "$EXISTING_ENTRIES" | grep -qF "$E"; then
            SKIPPED=$((SKIPPED + 1))
        else
            echo "$E" >> "$CHAIN_FILE"
            NEW_JOBS=$((NEW_JOBS + 1))
            NEW_KEYS+=("train-${TAG}")
        fi
    fi
    if [ "$GLOBAL_EVAL" -eq 1 ]; then
        E="eval:${CFG}:${TAG}"
        if [ "$APPEND" -eq 1 ] && echo "$EXISTING_ENTRIES" | grep -qF "$E"; then
            SKIPPED=$((SKIPPED + 1))
        else
            echo "$E" >> "$CHAIN_FILE"
            NEW_JOBS=$((NEW_JOBS + 1))
            NEW_KEYS+=("eval-${TAG}")
        fi
    fi
done

# Update monitor cache
if [ ${#NEW_KEYS[@]} -gt 0 ]; then
    KEYS_JSON=$(printf '"%s",' "${NEW_KEYS[@]}")
    KEYS_JSON="[${KEYS_JSON%,}]"
    CLEAR_OLD=0
    [ "$APPEND" -eq 0 ] && CLEAR_OLD=1
    python3 -c "
import json, pathlib
cache_path = pathlib.Path('$PROJ_DIR/.monitor_cache')
cache = json.loads(cache_path.read_text()) if cache_path.exists() else {'jobs': {}, 'pipeline_jobs': []}
cache.setdefault('pipeline_jobs', [])
if $CLEAR_OLD:
    cache['pipeline_jobs'] = []
    cache['jobs'] = {}
new_keys = $KEYS_JSON
for k in new_keys:
    if k not in cache['pipeline_jobs']:
        cache['pipeline_jobs'].append(k)
cache_path.write_text(json.dumps(cache, indent=2))
" 2>/dev/null
fi

TOTAL=$(wc -l < "$CHAIN_FILE")

if [ "$APPEND" -eq 1 ]; then
    if [ "$NEW_JOBS" -eq 0 ]; then
        echo "⚠️  Nessun nuovo job da aggiungere (tutti già in coda). Skippati: $SKIPPED"
        exit 0
    fi
    SKIP_MSG=""
    [ "$SKIPPED" -gt 0 ] && SKIP_MSG="  Skippati: $SKIPPED (già in coda)"
    echo "============================================"
    echo "  Jobs aggiunti alla pipeline attiva"
    echo "  Date:  $(date)"
    echo "  Nuovi: $NEW_JOBS job"
    [ -n "$SKIP_MSG" ] && echo "$SKIP_MSG"
    echo "  Totale in coda: $TOTAL"
    echo "============================================"
    echo ""
    echo "Catena completa:"
    cat -n "$CHAIN_FILE"
    echo ""
    echo "✅ Il watcher (PID $(cat "$PROJ_DIR/.chain_pid")) li eseguirà automaticamente."
    exit 0
fi

echo "============================================"
echo "  T2G GRPO Pipeline (self-chaining)"
echo "  Date:  $(date)"
echo "  Total jobs: $TOTAL"
echo "============================================"
echo ""
echo "Catena:"
cat -n "$CHAIN_FILE"
echo ""

# ── Avvia il watcher in background (nohup) ────────────────────────────────────
mkdir -p logs

if [ -f .chain_pid ]; then
    OLD_PID=$(cat .chain_pid)
    kill "$OLD_PID" 2>/dev/null && echo "Watcher precedente (PID $OLD_PID) terminato."
    rm -f .chain_pid
fi

nohup bash cluster/chain_next.sh >> logs/chain_watcher.log 2>&1 &
WATCHER_PID=$!
echo "$WATCHER_PID" > .chain_pid

echo ""
echo "============================================"
echo "  Pipeline avviata!"
echo "  Watcher PID: $WATCHER_PID"
echo "  Log: logs/chain_watcher.log"
echo "  Catena: .job_chain"
echo ""
echo "  Per monitorare:"
echo "    tail -f logs/chain_watcher.log"
echo "    t2g-monitor"
echo "    myjobs"
echo ""
echo "  Per interrompere:"
echo "    kill \$(cat .chain_pid)"
echo "    killalljobs"
echo "============================================"
