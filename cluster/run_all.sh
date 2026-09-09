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
#   bash cluster/run_all.sh                          # train+eval (default: sft-grpo/few-shot)
#   bash cluster/run_all.sh sft-grpo/few-shot     # train+eval con config specifico
#   bash cluster/run_all.sh --ablation               # ablation study completo
#   bash cluster/run_all.sh --eval-only              # solo evaluation
#   bash cluster/run_all.sh --train-only             # solo training
#   bash cluster/run_all.sh --resume                 # riparte dalla coda esistente
#   bash cluster/run_all.sh --append                 # aggiungi job alla coda attiva
#   bash cluster/run_all.sh --remove                 # svuota la coda
#   bash cluster/run_all.sh --force                  # azzera lo stato (catena interrotta)
#
# Config specifici (passa il path relativo a qwen25-05b, senza .yaml):
#   bash cluster/run_all.sh sft-grpo/few-shot       # config base
#   bash cluster/run_all.sh sft-grpo/few-shot       # config di riferimento (default)
#   bash cluster/run_all.sh sft/zero-shot           # SFT supervised da solo
#   bash cluster/run_all.sh ablations/decoding/no-grammar  # ablation senza grammar
#   (cerca sotto experiments/configs/qwen25-05b/ in modo ricorsivo)
#
# Campagna (--ablation): matrice completa 3 baseline eval-only + 12 celle train+eval.
# Ordine massimizza il riuso: le baseline zero-shot (Trie) cachano il --compare
# per le celle successive; sft/zero-shot addestra l'adapter SFT riusato dalle
# celle sft-grpo (fingerprint identica); le ablazioni riusano SFT + baseline.

# Interrompere:
#   chain-stop                               # ferma (preserva stato + tick at)
#   killalljobs                              # cancella anche il job SLURM attivo
# ============================================================================

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=cluster/_lib.sh
source "$SCRIPT_DIR/_lib.sh"
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
            echo "  (nessun argomento)  Default: sft-grpo/few-shot (train + eval)"
            echo "  config_name         Path del config relativo a qwen25-05b, senza .yaml (es. grpo/few-shot)"
            echo "  --ablation          Campagna completa (15 celle: 3 baseline eval-only + 12 train+eval)"
            echo "  --eval-only         Solo evaluation (skip training)"
            echo "  --train-only        Solo training (skip eval)"
            echo "  --resume            Riprendi dalla coda esistente"
            echo "  --append            Aggiungi job alla coda attiva"
            echo "  --remove            Svuota la coda"
            echo "  --force             Azzera lo stato anche se ci sono job pendenti"
            echo ""
            echo "Config disponibili (path relativo a experiments/configs/qwen25-05b, senza .yaml):"
            echo "  baseline/zero-shot            Base model + Trie (eval-only)"
            echo "  baseline/zero-shot-no-grammar Base model senza vincolo (eval-only, lower bound)"
            echo "  baseline/few-shot             Base model + few-shot retrieval (eval-only)"
            echo "  sft/zero-shot                 SFT supervised da solo"
            echo "  grpo/zero-shot                GRPO dal base, zero-shot"
            echo "  grpo/few-shot                 GRPO dal base, few-shot"
            echo "  sft-grpo/zero-shot            Pipeline SFT→GRPO zero-shot"
            echo "  sft-grpo/few-shot             Pipeline SFT→GRPO few-shot (default)"
            echo "  ablations/decoding/no-grammar         GRPO senza vincolo simbolico"
            echo "  ablations/decoding/hot-rollout        Rollout sampler a T=1.3"
            echo "  ablations/rewards/edit-validity       Reward edit-validity singola"
            echo "  ablations/rewards/historical-stack    Stack storico su init zero-shot"
            echo "  ablations/loss/dr-grpo                Obiettivo Dr-GRPO"
            echo "  ablations/objectives/sft-allowed-mass SFT + massa ammessa"
            echo "  ablations/objectives/sft-structured   SFT + loss strutturata"
            echo ""
            echo "Esempi:"
            echo "  bash cluster/run_all.sh sft-grpo/few-shot               # train + eval pipeline principale"
            echo "  bash cluster/run_all.sh sft-grpo/few-shot --train-only  # solo training"
            echo "  bash cluster/run_all.sh --ablation             # tutte le 15 celle"
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

# ── Dual eval ─────────────────────────────────────────────────────────────────
# NON è più una variabile d'ambiente (il vecchio DUAL_EVAL=1): il dual
# prompting è un knob del config (evaluation.dual_prompting, default true su
# base.yaml, disattivo sulle celle baseline/*). Ogni job eval lo onora da
# solo, niente propagazione di env var attraverso i tick della catena.

# ── Modelli T2G ───────────────────────────────────────────────────────────────
if [ "$ABLATION" -eq 1 ]; then
    # Campagna completa: 3 baseline eval-only + 12 celle train+eval.
    # Ordine ALLINEATO ad app.py:ABLATION_MODELS (il TUI batch usa la stessa
    # lista). Le baseline zero-shot (Trie) cachano il --compare per le celle
    # successive; sft/zero-shot addestra l'adapter SFT riusato dalle celle
    # sft-grpo (fingerprint identica).
    # Formato: TAG:CONFIG[:MODE]
    # MODE: te=train+eval (default), e=eval-only, t=train-only
    MODELS=(
        "baseline-zero-shot:experiments/configs/qwen25-05b/baseline/zero-shot.yaml:e"
        "baseline-zero-shot-no-grammar:experiments/configs/qwen25-05b/baseline/zero-shot-no-grammar.yaml:e"
        "baseline-few-shot:experiments/configs/qwen25-05b/baseline/few-shot.yaml:e"
        "sft-zero-shot:experiments/configs/qwen25-05b/sft/zero-shot.yaml:te"
        "grpo-zero-shot:experiments/configs/qwen25-05b/grpo/zero-shot.yaml:te"
        "grpo-few-shot:experiments/configs/qwen25-05b/grpo/few-shot.yaml:te"
        "sft-grpo-zero-shot:experiments/configs/qwen25-05b/sft-grpo/zero-shot.yaml:te"
        "sft-grpo-few-shot:experiments/configs/qwen25-05b/sft-grpo/few-shot.yaml:te"
        "ablations-decoding-no-grammar:experiments/configs/qwen25-05b/ablations/decoding/no-grammar.yaml:te"
        "ablations-decoding-hot-rollout:experiments/configs/qwen25-05b/ablations/decoding/hot-rollout.yaml:te"
        "ablations-rewards-edit-validity:experiments/configs/qwen25-05b/ablations/rewards/edit-validity.yaml:te"
        "ablations-rewards-historical-stack:experiments/configs/qwen25-05b/ablations/rewards/historical-stack.yaml:te"
        "ablations-loss-dr-grpo:experiments/configs/qwen25-05b/ablations/loss/dr-grpo.yaml:te"
        "ablations-objectives-sft-allowed-mass:experiments/configs/qwen25-05b/ablations/objectives/sft-allowed-mass.yaml:te"
        "ablations-objectives-sft-structured:experiments/configs/qwen25-05b/ablations/objectives/sft-structured.yaml:te"
    )
elif [ -n "$CONFIG_NAME" ]; then
    # Config specifico passato come argomento (es. "grpo/few-shot").
    # Cerca sotto experiments/configs/qwen25-05b/ in modo ricorsivo.
    CONFIG_PATH=""
    for ext in ".yaml" ""; do
        candidate="experiments/configs/qwen25-05b/${CONFIG_NAME}${ext}"
        if [ -f "$candidate" ]; then
            CONFIG_PATH="$candidate"
            break
        fi
    done
    if [ -z "$CONFIG_PATH" ]; then
        # Fallback ricorsivo: tollera anche il solo basename (es. "no-grammar"
        # senza il path relativo completo). `find` è disponibile sul login node.
        CONFIG_PATH=$(find experiments/configs/qwen25-05b -type f \
            \( -name "${CONFIG_NAME}.yaml" -o -name "${CONFIG_NAME}" \) 2>/dev/null | head -1)
    fi
    if [ -z "$CONFIG_PATH" ]; then
        echo "❌ Config non trovato: $CONFIG_NAME"
        echo "   Cercato in: experiments/configs/qwen25-05b/ (ricorsivo)"
        echo "   Usa: bash cluster/run_all.sh --help per la lista dei config"
        exit 1
    fi
    # Deriva il tag dal path relativo a qwen25-05b (slash → trattini).
    TAG=$(echo "${CONFIG_PATH#experiments/configs/qwen25-05b/}" | sed 's/\.yaml$//' | tr '/_' '--')
    MODELS=("${TAG}:${CONFIG_PATH}")
else
    # Default: pipeline principale SFT+GRPO few-shot.
    MODELS=("sft-grpo-few-shot:experiments/configs/qwen25-05b/sft-grpo/few-shot.yaml")
fi

mkdir -p "$STATE_DIR" logs

# ── Funzioni di lancio ────────────────────────────────────────────────────────
# Kick della pipeline: UN tick immediato. Il tick è one-shot e idempotente
# (flock interno): se la coda è vuota o il job è già attivo, non fa nulla.
# La resilienza a lungo termine è l'hook bashrc (chain-hook-install) e il
# server esterno (POST /tick): nessun daemon da tenere in vita.
_launch_pipeline() {
    mkdir -p logs

    bash cluster/chain_tick.sh --quiet
    local rc=$?
    [ "$rc" -ne 0 ] && echo "⚠️  tick rc=$rc - riproverà al prossimo tick (hook/server)."
    echo ""
    echo "▶ Tick eseguito. La catena avanza a ogni tick:"
    echo "   chain-show                      # stato pipeline"
    echo "   tail -f logs/chain.log          # log della catena"
    echo "   chain-hook-install              # resilienza: hook bashrc (consigliato)"
}
# Riprendi dalla coda ESISTENTE: basta che job_chain sia non vuota
# (il caso reale: daemon ucciso dal reaper).
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
