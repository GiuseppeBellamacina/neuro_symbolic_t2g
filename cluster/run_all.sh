#!/bin/bash
# ============================================================================
# Lancia training + evaluation per il modello T2G in catena (self-chaining).
#
# La QoS permette un solo job alla volta (1 attivo, 0 pending), quindi la
# catena viene avanzata da chain_tick.sh (one-shot idempotente), guidata da:
#   bashrc-hook (PRIMARIO) → chain_tick.sh --quiet via PROMPT_COMMAND
#                            (chain-hook-install) — su gcluster `at` NON c'è
#   server esterno         → POST /tick → cluster_helper.sh tick
#   manuale                → chain-start / run-all (tick immediato)
#
# MAI rm -rf automatico dello stato con job pendenti: se una catena risulta
# interrotta (job_chain non vuota, nessun job attivo)
# run_all RIFIUTA e chiede chain-resume (o --force per ricominciare).
#
# Uso:
#   bash cluster/run_all.sh                          # default campaign
#   bash cluster/run_all.sh sft-grpo-zero             # config specifico
#   bash cluster/run_all.sh --ablation               # alias della campagna default
#   bash cluster/run_all.sh --eval-only              # solo evaluation
#   bash cluster/run_all.sh --train-only             # solo training
#   bash cluster/run_all.sh --resume                 # riparte dalla coda esistente
#   bash cluster/run_all.sh --append                 # aggiungi job alla coda attiva
#   bash cluster/run_all.sh --remove                 # svuota la coda
#   bash cluster/run_all.sh --force                  # azzera lo stato (catena interrotta)
#
# Campagna default: 2 baseline eval-only + 5 train/eval = 12 entry.
# Ogni entry eval addestrata esegue due leg (zero-shot, poi retrieval).
# Le ablation sono disponibili solo come selezioni manuali.

# Interrompere:
#   chain-stop                               # ferma (preserva stato + tick at)
#   killalljobs                              # cancella anche il job SLURM attivo
# ============================================================================

set -euo pipefail

# BEGIN T2G_LIB_RESOLVER
_lib_source=${BASH_SOURCE[0]}
_lib_dir=""
_lib_candidate=$(cd "$(dirname "$_lib_source")" 2>/dev/null && pwd) || _lib_candidate=""
if [ -n "$_lib_candidate" ] && [ -f "$_lib_candidate/_lib.sh" ]; then
    _lib_dir=$_lib_candidate
elif [ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -f "${SLURM_SUBMIT_DIR}/cluster/_lib.sh" ]; then
    _lib_dir=$(cd "${SLURM_SUBMIT_DIR}/cluster" && pwd)
elif [ -n "${HOME:-}" ] && [ -f "${HOME}/neuro_symbolic_t2g/cluster/_lib.sh" ]; then
    _lib_dir=$(cd "${HOME}/neuro_symbolic_t2g/cluster" && pwd)
else
    printf 'ERROR: cannot locate cluster/_lib.sh (BASH_SOURCE=%s, SLURM_SUBMIT_DIR=%s)\n' \
        "$_lib_source" "${SLURM_SUBMIT_DIR:-<unset>}" >&2
    exit 1
fi
SCRIPT_DIR=$_lib_dir
# shellcheck source=cluster/_lib.sh
source "$SCRIPT_DIR/_lib.sh"
# END T2G_LIB_RESOLVER
cd "$PROJ_DIR"

# ── Parsing argomenti ─────────────────────────────────────────────────────────
GLOBAL_TRAIN=1
GLOBAL_EVAL=1
ABLATION=0
RESUME=0
APPEND=0
REMOVE=0
FORCE=0
CONFIG_NAME=""
for arg in "$@"; do
    case "$arg" in
        --ablation)    ABLATION=1 ;;
        --eval-only)   GLOBAL_TRAIN=0 ;;
        --train-only)  GLOBAL_EVAL=0 ;;
        --append)      APPEND=1 ;;
        --remove)      REMOVE=1 ;;
        --resume)      RESUME=1 ;;
        --force)       FORCE=1 ;;
        --help|-h)
            echo "Uso: bash cluster/run_all.sh [opzioni] [config_name]"
            echo ""
            echo "Opzioni:"
            echo "  (nessun argomento)  Campagna default (7 celle, 12 entry)"
            echo "  config_name         ID semantico del config"
            echo "  --ablation          Alias della campagna default"
            echo "  --eval-only         Solo evaluation (skip training)"
            echo "  --train-only        Solo training (skip eval)"
            echo "  --resume            Riprendi dalla coda esistente"
            echo "  --append            Aggiungi job alla coda attiva"
            echo "  --remove            Svuota la coda"
            echo "  --force             Azzera lo stato anche se ci sono job pendenti"
            echo ""
            echo "Config disponibili (passa il nome senza .yaml):"
            echo "  baseline-zero baseline-few sft grpo-zero grpo-few"
            echo "  sft-grpo-zero sft-grpo-few sft-grpo-zero-pda sft-grpo-zero-hot"
            echo "  grpo-few-reward-edit grpo-few-reward-token-f1 grpo-few-reward-chrfpp"
            echo "  grpo-few-reward-rouge-l grpo-few-reward-sbleu2"
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

# ── Config registry ───────────────────────────────────────────────────────────
config_path() {
    case "$1" in
        baseline-zero) echo "experiments/configs/qwen25-05b/baseline/zero-shot.yaml" ;;
        baseline-few) echo "experiments/configs/qwen25-05b/baseline/few-shot.yaml" ;;
        sft) echo "experiments/configs/qwen25-05b/sft/zero-shot.yaml" ;;
        grpo-zero) echo "experiments/configs/qwen25-05b/grpo/zero-shot.yaml" ;;
        grpo-few) echo "experiments/configs/qwen25-05b/grpo/few-shot.yaml" ;;
        sft-grpo-zero) echo "experiments/configs/qwen25-05b/sft-grpo/zero-shot.yaml" ;;
        sft-grpo-few) echo "experiments/configs/qwen25-05b/sft-grpo/few-shot.yaml" ;;
        sft-grpo-zero-pda) echo "experiments/configs/qwen25-05b/ablations/sft-grpo-zero-pda.yaml" ;;
        sft-grpo-zero-hot) echo "experiments/configs/qwen25-05b/ablations/sft-grpo-zero-hot.yaml" ;;
        grpo-few-reward-edit) echo "experiments/configs/qwen25-05b/ablations/rewards/edit.yaml" ;;
        grpo-few-reward-token-f1) echo "experiments/configs/qwen25-05b/ablations/rewards/token-f1.yaml" ;;
        grpo-few-reward-chrfpp) echo "experiments/configs/qwen25-05b/ablations/rewards/chrfpp.yaml" ;;
        grpo-few-reward-rouge-l) echo "experiments/configs/qwen25-05b/ablations/rewards/rouge-l.yaml" ;;
        grpo-few-reward-sbleu2) echo "experiments/configs/qwen25-05b/ablations/rewards/sbleu2.yaml" ;;
        *) return 1 ;;
    esac
}

if [ "$ABLATION" -eq 1 ] || [ -z "$CONFIG_NAME" ]; then
    MODELS=(
        "baseline-zero:experiments/configs/qwen25-05b/baseline/zero-shot.yaml:e"
        "baseline-few:experiments/configs/qwen25-05b/baseline/few-shot.yaml:e"
        "sft:experiments/configs/qwen25-05b/sft/zero-shot.yaml:te"
        "grpo-zero:experiments/configs/qwen25-05b/grpo/zero-shot.yaml:te"
        "grpo-few:experiments/configs/qwen25-05b/grpo/few-shot.yaml:te"
        "sft-grpo-zero:experiments/configs/qwen25-05b/sft-grpo/zero-shot.yaml:te"
        "sft-grpo-few:experiments/configs/qwen25-05b/sft-grpo/few-shot.yaml:te"
    )
else
    if ! CONFIG_PATH=$(config_path "$CONFIG_NAME"); then
        echo "❌ Config non trovato: $CONFIG_NAME"
        echo "   Usa: bash cluster/run_all.sh --help per la lista dei config"
        exit 1
    fi
    MODELS=("${CONFIG_NAME}:${CONFIG_PATH}")
    if [[ "$CONFIG_NAME" == grpo-few-reward-* ]] && [[ "${EXTRA_ARGS:-}" != *"--reward-qualification-report"* ]]; then
        echo "❌ $CONFIG_NAME richiede EXTRA_ARGS='--reward-qualification-report PATH'" >&2
        exit 2
    fi
fi

mkdir -p "$STATE_DIR" logs

# ── Funzioni di lancio ────────────────────────────────────────────────────────
# Kick della pipeline: UN tick immediato. Il tick è one-shot e idempotente
# (flock interno): se la coda è vuota o il job è già attivo, non fa nulla.
# La resilienza a lungo termine è l'hook bashrc (chain-hook-install) e il
# server esterno (POST /tick): nessun daemon da tenere in vita.
_launch_pipeline() {
    mkdir -p logs

    local rc=0
    bash cluster/chain_tick.sh --quiet || rc=$?
    [ "$rc" -ne 0 ] && echo "⚠️  tick rc=$rc - riproverà al prossimo tick (hook/server)."
    echo ""
    echo "▶ Tick eseguito. La catena avanza a ogni tick:"
    echo "   chain-show                      # stato pipeline"
    echo "   tail -f logs/chain.log          # log della catena"
    echo "   chain-hook-install              # resilienza: hook bashrc (consigliato)"
}
_cmd_resume() {
    echo "============================================"
    echo "  RESUME Pipeline"
    echo "  Date:  $(date)"
    echo "============================================"

    if [ -s "$CHAIN_FILE" ]; then
        echo "Coda esistente ($(wc -l < "$CHAIN_FILE") job):"
        cat -n "$CHAIN_FILE"
    else
        echo "❌ Nessuna coda da riprendere (job_chain vuoto)."
        echo "   Usa: bash cluster/run_all.sh (senza --resume) per una nuova pipeline."
        exit 1
    fi


    _launch_pipeline
    echo ""
    echo "============================================"
    echo "  Pipeline ripresa!"
    echo "============================================"
}

# ── Resume mode ───────────────────────────────────────────────────────────────
if [ "$RESUME" -eq 1 ]; then
    _cmd_resume
    exit 0
fi

# ── Auto-append: se la catena è già attiva, i nuovi job vengono AGGIUNTI ──────
if [ "$APPEND" -eq 0 ] && [ "$REMOVE" -eq 0 ]; then
    if [ -n "$(active_job_id)" ]; then
        echo "⚠️  Pipeline già attiva (job SLURM in corso)."
        echo "   I nuovi job verranno AGGIUNTI alla coda esistente."
        echo ""
        APPEND=1
    fi
fi

# ── Remove mode ───────────────────────────────────────────────────────────────
if [ "$REMOVE" -eq 1 ]; then
    if [ ! -s "$CHAIN_FILE" ]; then
        echo "❌ Nessuna catena attiva (job_chain vuoto)."
        exit 1
    fi
    echo "Catena attuale:"
    cat -n "$CHAIN_FILE"
    echo ""
    rm -f "$CHAIN_FILE"
    monitor_cache_clear
    echo "✅ Catena svuotata."
    exit 0
fi

# ── Fresh start: MAI rm -rf automatico con job pendenti ──────────────────────
# Anti-rm-rf: se la coda esiste non vuota e nessun job attivo
# e nessun at-tick → la catena è interrotta (es. daemon ucciso dal reaper).
# Azzerare qui significherebbe CANCELLARE i job rimanenti.
if [ "$RESUME" -eq 0 ] && [ "$APPEND" -eq 0 ] && [ "$REMOVE" -eq 0 ]; then
    if [ -s "$CHAIN_FILE" ]; then
        REMAINING=$(wc -l < "$CHAIN_FILE")
        if [ "$FORCE" -eq 1 ]; then
            echo "⚠️  Catena interrotta con $REMAINING job rimanenti — --force: reset."
            rm -rf "$STATE_DIR"
            mkdir -p "$STATE_DIR"
        else
            echo "❌ Catena interrotta con $REMAINING job rimanenti."
            echo "   → chain-resume      riparte dalla coda esistente"
            echo "   → run-all --force   azzera lo stato e ricomincia da zero"
            exit 1
        fi
    else
        # Nessun job pendente → reset sicuro
        rm -rf "$STATE_DIR"
        mkdir -p "$STATE_DIR"
    fi
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
        if [[ "$TAG" == grpo-few-reward-* ]]; then
            E="${E}:${EXTRA_ARGS}"
        fi
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

# Update monitor cache (shell-only — niente python sul login node)
for key in ${NEW_KEYS[@]+"${NEW_KEYS[@]}"}; do
    monitor_cache_add "$key"
done

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
    # Se nulla sta già avanzando la coda, avviala ora
    if [ -z "$(active_job_id)" ]; then
        echo "✅ Nessun driver attivo — avvio la pipeline."
        _launch_pipeline
    else
        echo "✅ La catena avanza a ogni tick (hook bashrc / server / manuale)."
    fi
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

# ── Avvia la pipeline (tick immediato; avanza poi via hook/server) ─
_launch_pipeline

echo ""
echo "============================================"
echo "  Pipeline avviata!"
echo "  Log:  logs/chain.log"
echo "  Coda: .chain_state/job_chain"
echo ""
echo "  Per monitorare:"
echo "    tail -f logs/chain.log"
echo "    monitor"
echo "    myjobs"
echo ""
echo "  Per interrompere:"
echo "    chain-stop"
echo "    killalljobs"
echo "============================================"
