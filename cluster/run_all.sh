#!/bin/bash
# ============================================================================
# Lancia training + evaluation per il modello T2G in catena.
#
# La QoS permette un solo job alla volta, quindi un watcher in background
# controlla ogni 60s se la coda è vuota e sottomette il prossimo job.
#
# Uso:
#   bash cluster/run_all.sh                       # train + eval (default: grpo_optimal)
#   bash cluster/run_all.sh grpo_qwen05             # train + eval con config specifico
#   bash cluster/run_all.sh grpo_qwen05 --train-only # solo training con config specifico
#   bash cluster/run_all.sh --ablation             # ablation study completo
#   bash cluster/run_all.sh --eval-only            # solo evaluation
#   bash cluster/run_all.sh --train-only           # solo training
#   bash cluster/run_all.sh --resume               # riprendi pipeline fallita
#   bash cluster/run_all.sh --append               # aggiungi job a pipeline attiva
#   bash cluster/run_all.sh --remove               # rimuovi job dalla pipeline
#
# Config specifici (passa il nome senza .yaml):
#   bash cluster/run_all.sh grpo_qwen05             # config base
#   bash cluster/run_all.sh grpo_optimal            # config ottimale (default)
#   bash cluster/run_all.sh sft                     # SFT baseline
#   bash cluster/run_all.sh grpo_no_grammar         # ablation senza grammar
#   (cerca in experiments/configs/t2g/ e experiments/configs/t2g/ablation/)
#
# Ablation Study (--ablation):
#   1. Base Model Zero-shot (senza grammar)              [eval only]
#   2. Base Model + GRAMMAR-LLM (con grammar, no training) [eval only]
#   3. GRPO senza grammar (train + eval)
#   4. GRPO senza SFT pre-training (train + eval)
#   5. GRPO + GRAMMAR-LLM (train + eval — metodo base)
#   6. SFT (train + eval — baseline supervisionata)
#   7. GRPO + GrammarLLM PDA (train + eval — LL(1) constrained)
#   8. GRPO + PDA + Token-Boundary Lookahead (train + eval — grammarllm v0.5.0)
#   9. GRPO + Soft Viterbi (train + eval — DVL differentiable)
#  10. GRPO + Verifier-Scaled (train + eval — RECIPE-inspired)
#  11. GRPO + Experimental All Modules (train + eval — tutti i 9 moduli)
#  12. GRPO + Optimal (train + eval — config ottimale v2.1)
#
# Monitorare:
#   tail -f logs/chain_watcher.log           # log del watcher
#   monitor                              # monitor live
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
ABLATION=0
RESUME=0
APPEND=0
REMOVE=0
CONFIG_NAME=""
for arg in "$@"; do
    case "$arg" in
        --ablation)    ABLATION=1 ;;
        --eval-only)   GLOBAL_TRAIN=0 ;;
        --train-only)  GLOBAL_EVAL=0 ;;
        --append)      APPEND=1 ;;
        --remove)      REMOVE=1 ;;
        --resume)      RESUME=1 ;;
        --help|-h)
            echo "Uso: bash cluster/run_all.sh [opzioni] [config_name]"
            echo ""
            echo "Opzioni:"
            echo "  (nessun argomento)  Default: grpo_optimal (train + eval)"
            echo "  config_name         Nome del config senza .yaml (es. grpo_qwen05)"
            echo "  --ablation          Ablation study completo (12 varianti)"
            echo "  --eval-only         Solo evaluation (skip training)"
            echo "  --train-only        Solo training (skip eval)"
            echo "  --resume            Riprendi pipeline da dove si era fermata"
            echo "  --append            Aggiungi job alla pipeline attiva"
            echo "  --remove            Rimuovi job dalla pipeline attiva"
            echo ""
            echo "Config disponibili (passa il nome senza .yaml):"
            echo "  grpo_optimal           GRPO + tutti i moduli v2.1 (default, post-OOM-fix)"
            echo "  grpo_qwen05            GRPO + grammar (config base)"
            echo "  sft                    SFT supervised baseline"
            echo "  grpo_experimental_all  GRPO + tutti i 9 moduli reward (experimental)"
            echo "  grpo_no_grammar        Ablation: GRPO senza grammar"
            echo "  grpo_no_sft            Ablation: GRPO senza SFT pre-training"
            echo "  grpo_pda               Ablation: GRPO + PDA (LL(1) baseline)"
            echo "  grpo_pda_lookahead     Ablation: GRPO + PDA + token-boundary lookahead (v0.5.0)"
            echo "  grpo_soft_viterbi      Ablation: GRPO + Soft Viterbi"
            echo "  grpo_verifier_scaled   Ablation: GRPO + Verifier-Scaled"
            echo "  zero_shot              Ablation: zero-shot senza grammar"
            echo "  zero_shot_grammar      Ablation: zero-shot con grammar"
            echo ""
            echo "Esempi:"
            echo "  bash cluster/run_all.sh grpo_qwen05              # train + eval con config base"
            echo "  bash cluster/run_all.sh grpo_qwen05 --train-only  # solo training"
            echo "  bash cluster/run_all.sh --ablation               # tutti i 12 config"
            exit 0
            ;;
        -*)  # ignora flag non riconosciuti
            ;;
        *)
            # Primo argomento non-flag = nome del config
            if [ -z "$CONFIG_NAME" ]; then
                CONFIG_NAME="$arg"
            fi
            ;;
    esac
done

# ── Modelli T2G ───────────────────────────────────────────────────────────────
if [ "$ABLATION" -eq 1 ]; then
    # Ablation study: 9 varianti in ordine (dai più semplici ai più complessi)
    # Formato: TAG:CONFIG[:MODE]
    # MODE: te=train+eval (default), e=eval-only, t=train-only
    MODELS=(
        "zero-shot:experiments/configs/t2g/ablation/zero_shot.yaml:e"
        "zero-shot-gram:experiments/configs/t2g/ablation/zero_shot_grammar.yaml:e"
        "grpo-no-grammar:experiments/configs/t2g/ablation/grpo_no_grammar.yaml:te"
        "grpo-no-sft:experiments/configs/t2g/ablation/grpo_no_sft.yaml:te"
        "grpo-grammar:experiments/configs/t2g/grpo_qwen05.yaml:te"
        "sft:experiments/configs/t2g/sft.yaml:te"
        "grpo-pda:experiments/configs/t2g/ablation/grpo_pda.yaml:te"
        "grpo-pda-lookahead:experiments/configs/t2g/ablation/grpo_pda_lookahead.yaml:te"
        "grpo-soft-viterbi:experiments/configs/t2g/ablation/grpo_soft_viterbi.yaml:te"
        "grpo-verifier:experiments/configs/t2g/ablation/grpo_verifier_scaled.yaml:te"
        "grpo-experimental-all:experiments/configs/t2g/grpo_experimental_all.yaml:te"
        "grpo-optimal:experiments/configs/t2g/grpo_optimal.yaml:te"
    )
elif [ -n "$CONFIG_NAME" ]; then
    # Config specifico passato come argomento (es. "grpo_qwen05")
    # Cerca in experiments/configs/t2g/ e experiments/configs/t2g/ablation/
    CONFIG_PATH=""
    for dir in "experiments/configs/t2g" "experiments/configs/t2g/ablation"; do
        for ext in ".yaml" ""; do
            candidate="${dir}/${CONFIG_NAME}${ext}"
            if [ -f "$candidate" ]; then
                CONFIG_PATH="$candidate"
                break 2
            fi
        done
    done
    if [ -z "$CONFIG_PATH" ]; then
        echo "❌ Config non trovato: $CONFIG_NAME"
        echo "   Cercato in: experiments/configs/t2g/ e experiments/configs/t2g/ablation/"
        echo "   Usa: bash cluster/run_all.sh --help per la lista dei config"
        exit 1
    fi
    # Deriva il tag dal nome del config (senza percorso ed estensione)
    TAG=$(basename "$CONFIG_PATH" .yaml)
    MODELS=("${TAG}:${CONFIG_PATH}")
else
    # Default: config ottimale
    MODELS=("qwen05:experiments/configs/t2g/grpo_optimal.yaml")
fi

PROJ_DIR="$HOME/neuro_symbolic_t2g"
STATE_DIR="$PROJ_DIR/.chain_state"
mkdir -p "$STATE_DIR"
CHAIN_FILE="$STATE_DIR/job_chain"
FAILED_FILE="$STATE_DIR/chain_failed"

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

    if [ -f "$STATE_DIR/chain_pid" ]; then
        OLD_PID=$(cat "$STATE_DIR/chain_pid")
        kill "$OLD_PID" 2>/dev/null && echo "Watcher precedente (PID $OLD_PID) terminato."
        rm -f "$STATE_DIR/chain_pid"
    fi

    nohup bash cluster/chain_next.sh >> logs/chain_watcher.log 2>&1 &

    # Attendi che il watcher scriva il suo PID, poi leggilo
    sleep 2
    WATCHER_PID=""
    if [ -f "$STATE_DIR/chain_pid" ]; then
        WATCHER_PID=$(cat "$STATE_DIR/chain_pid")
    fi

    echo "============================================"
    echo "  Pipeline ripresa!"
    echo "  Watcher PID: ${WATCHER_PID:-sconosciuto}"
    echo "  Log: logs/chain_watcher.log"
    echo "============================================"
    exit 0
fi

# ── Funzione helper: controlla se il watcher è attivo ─────────────────────────
_watcher_is_alive() {
    if [ -f "$STATE_DIR/chain_pid" ]; then
        local pid=$(cat "$STATE_DIR/chain_pid")
        if ps -p "$pid" > /dev/null 2>&1; then
            return 0
        fi
    fi
    return 1
}

# ── Auto-detect: se watcher attivo e non --append/--remove esplicito ──────────
if [ "$APPEND" -eq 0 ] && [ "$REMOVE" -eq 0 ] && _watcher_is_alive; then
    echo "⚠️  Pipeline già attiva (watcher PID $(cat "$STATE_DIR/chain_pid"))."
    echo "   I nuovi job verranno AGGIUNTI alla coda esistente."
    echo "   (Usa Ctrl-C per annullare, oppure uccidi il watcher prima: watcher-kill)"
    echo ""
    APPEND=1
fi

# ── Remove mode ───────────────────────────────────────────────────────────────
if [ "$REMOVE" -eq 1 ]; then
    if [ ! -f "$CHAIN_FILE" ] || [ ! -s "$CHAIN_FILE" ]; then
        echo "❌ Nessuna catena attiva (job_chain vuoto o non trovato)."
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
cache_path = pathlib.Path('$STATE_DIR/monitor_cache')
if cache_path.exists():
    cache_path.unlink()
" 2>/dev/null

    exit 0
fi

# ── Costruisci la catena ──────────────────────────────────────────────────────
# Pulisci .chain_state/ se partenza fresca (non --resume, --append, --remove)
if [ "$RESUME" -eq 0 ] && [ "$APPEND" -eq 0 ] && [ "$REMOVE" -eq 0 ]; then
    rm -rf "$STATE_DIR"
    mkdir -p "$STATE_DIR"
fi

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
    MODE=$(echo "$entry" | cut -d: -f3)

    # Default: train+eval if no MODE specified
    DO_TRAIN=$GLOBAL_TRAIN
    DO_EVAL=$GLOBAL_EVAL
    case "$MODE" in
        e)   DO_TRAIN=0; DO_EVAL=1 ;;
        t)   DO_TRAIN=1; DO_EVAL=0 ;;
        te|"") DO_TRAIN=$GLOBAL_TRAIN; DO_EVAL=$GLOBAL_EVAL ;;
    esac

    if [ "$DO_TRAIN" -eq 1 ]; then
        E="train:${CFG}:${TAG}"
        if [ "$APPEND" -eq 1 ] && echo "$EXISTING_ENTRIES" | grep -qF "$E"; then
            SKIPPED=$((SKIPPED + 1))
        else
            echo "$E" >> "$CHAIN_FILE"
            NEW_JOBS=$((NEW_JOBS + 1))
            NEW_KEYS+=("train-${TAG}")
        fi
    fi
    if [ "$DO_EVAL" -eq 1 ]; then
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
cache_path = pathlib.Path('$STATE_DIR/monitor_cache')
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
    echo "✅ Il watcher (PID $(cat "$STATE_DIR/chain_pid")) li eseguirà automaticamente."
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

if [ -f "$STATE_DIR/chain_pid" ]; then
    OLD_PID=$(cat "$STATE_DIR/chain_pid")
    kill "$OLD_PID" 2>/dev/null && echo "Watcher precedente (PID $OLD_PID) terminato."
    rm -f "$STATE_DIR/chain_pid"
fi

nohup bash cluster/chain_next.sh >> logs/chain_watcher.log 2>&1 &

# Attendi che il watcher scriva il suo PID, poi leggilo
sleep 2
WATCHER_PID=""
if [ -f "$STATE_DIR/chain_pid" ]; then
    WATCHER_PID=$(cat "$STATE_DIR/chain_pid")
fi

echo ""
echo "============================================"
echo "  Pipeline avviata!"
echo "  Watcher PID: ${WATCHER_PID:-sconosciuto}"
echo "  Log: logs/chain_watcher.log"
echo "  Catena: .chain_state/job_chain"
echo ""
echo "  Per monitorare:"
echo "    tail -f logs/chain_watcher.log"
echo "    monitor"
echo "    myjobs"
echo ""
echo "  Per interrompere:"
echo "    kill \$(cat .chain_state/chain_pid)"
echo "    killalljobs"
echo "============================================"
