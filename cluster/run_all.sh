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
#   bash cluster/run_all.sh                          # train+eval (default: sft-grpo)
#   bash cluster/run_all.sh sft-grpo              # train+eval con config specifico
#   bash cluster/run_all.sh --ablation               # ablation study completo
#   bash cluster/run_all.sh --eval-only              # solo evaluation
#   bash cluster/run_all.sh --train-only             # solo training
#   bash cluster/run_all.sh --resume                 # riparte dalla coda esistente
#   bash cluster/run_all.sh --append                 # aggiungi job alla coda attiva
#   bash cluster/run_all.sh --remove                 # svuota la coda
#   bash cluster/run_all.sh --force                  # azzera lo stato (catena interrotta)
#
# Config specifici (passa il nome senza .yaml):
#   bash cluster/run_all.sh sft-grpo              # config base
#   bash cluster/run_all.sh sft-grpo             # config ottimale (default)
#   bash cluster/run_all.sh sft                      # SFT baseline
#   bash cluster/run_all.sh grpo_no_grammar          # ablation senza grammar
#   (cerca in experiments/configs/t2g/ e experiments/configs/t2g/)
#
# Campagna (--ablation): decomposizione + ablation moduli + zero-shot
#   1. GRPO-only (base, senza SFT)               [train + eval]
#   2. SFT+GRPO pipeline principale              [train + eval]
#   3-6. SFT+GRPO + singolo modulo sperimentale  [train + eval]
#   7. SFT+GRPO + tutti i moduli                 [train + eval]
#   8. SFT+GRPO senza constrained decoding       [train + eval]
#   9. Zero-shot base (senza grammar)            [eval only]
#  10. Zero-shot base + grammar                  [eval only]

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
            echo "  (nessun argomento)  Default: sft-grpo (train + eval)"
            echo "  config_name         Nome del config senza .yaml (es. sft-grpo)"
            echo "  --ablation          Campagna completa (7 celle train+eval + 2 zero-shot)"
            echo "  --eval-only         Solo evaluation (skip training)"
            echo "  --train-only        Solo training (skip eval)"
            echo "  --resume            Riprendi dalla coda esistente (non richiede chain_failed)"
            echo "  --append            Aggiungi job alla coda attiva"
            echo "  --remove            Svuota la coda"
            echo "  --force             Azzera lo stato anche se ci sono job pendenti"
            echo ""
            echo "Config disponibili (passa il nome senza .yaml):"
            echo "  sft-grpo               Pipeline principale SFT+GRPO (default)"
            echo "  sft-only               SFT supervised da solo (cella decomposizione)"
            echo "  grpo-only              GRPO dal base, senza SFT (cella decomposizione)"
            echo "  sft-grpo-structure     SFT+GRPO + structural_dense (ablation moduli)"
            echo "  sft-grpo-viterbi       SFT+GRPO + viterbi_distance (ablation moduli)"
            echo "  sft-grpo-soft-viterbi  SFT+GRPO + soft_viterbi (ablation moduli)"
            echo "  sft-grpo-all-rewards   SFT+GRPO + tutti e 3 i moduli sperimentali"
            echo "  sft-grpo-no-grammar     SFT+GRPO senza constrained decoding"
            echo "  zero-shot               Base model senza grammar (solo eval)"
            echo "  zero-shot-grammar       Base model con grammar (solo eval)"
            echo ""
            echo "Esempi:"
            echo "  bash cluster/run_all.sh sft-grpo               # train + eval pipeline principale"
            echo "  bash cluster/run_all.sh sft-grpo --train-only  # solo training"
            echo "  bash cluster/run_all.sh --ablation             # tutte le 7 celle"
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
    # Campagna di decomposizione + ablation moduli: 7 celle train+eval.
    # Ordine ALLINEATO ad app.py:ABLATION_MODELS (il TUI batch usa la stessa
    # lista). sft-only NON è in coda: la sua cella si valuta con l'adapter
    # già addestrato della pipeline (CHECKPOINT esplicito, vedi sft-only.yaml).
    # Formato: TAG:CONFIG[:MODE]
    # MODE: te=train+eval (default), e=eval-only, t=train-only
    MODELS=(
        "grpo-only:experiments/configs/t2g/grpo-only.yaml:te"
        "sft-grpo:experiments/configs/t2g/sft-grpo.yaml:te"
        "sft-grpo-soft-viterbi:experiments/configs/t2g/sft-grpo-soft-viterbi.yaml:te"
        "sft-grpo-all-rewards:experiments/configs/t2g/sft-grpo-all-rewards.yaml:te"
        "sft-grpo-structure:experiments/configs/t2g/sft-grpo-structure.yaml:te"
        "sft-grpo-viterbi:experiments/configs/t2g/sft-grpo-viterbi.yaml:te"
        "sft-grpo-no-grammar:experiments/configs/t2g/sft-grpo-no-grammar.yaml:te"
        "zero-shot:experiments/configs/t2g/zero-shot.yaml:e"
        "zero-shot-grammar:experiments/configs/t2g/zero-shot-grammar.yaml:e"
    )
elif [ -n "$CONFIG_NAME" ]; then
    # Config specifico passato come argomento (es. "sft-grpo")
    # Cerca in experiments/configs/t2g/
    CONFIG_PATH=""
    for ext in ".yaml" ""; do
        candidate="experiments/configs/t2g/${CONFIG_NAME}${ext}"
        if [ -f "$candidate" ]; then
            CONFIG_PATH="$candidate"
            break
        fi
    done
    if [ -z "$CONFIG_PATH" ]; then
        echo "❌ Config non trovato: $CONFIG_NAME"
        echo "   Cercato in: experiments/configs/t2g/"
        echo "   Usa: bash cluster/run_all.sh --help per la lista dei config"
        exit 1
    fi
    # Deriva il tag dal nome del config (senza percorso ed estensione)
    TAG=$(basename "$CONFIG_PATH" .yaml | tr '_' '-')
    MODELS=("${TAG}:${CONFIG_PATH}")
else
    # Default: pipeline principale SFT+GRPO.
    MODELS=("sft-grpo:experiments/configs/t2g/sft-grpo.yaml")
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
# Riprendi dalla coda ESISTENTE: non richiede più .chain_failed, basta che
# job_chain sia non vuota (il caso reale: daemon ucciso dal reaper). Legacy:
# ricostruisce da .chain_failed se la coda è vuota.
_cmd_resume() {
    echo "============================================"
    echo "  RESUME Pipeline"
    echo "  Date:  $(date)"
    echo "============================================"

    if [ -s "$CHAIN_FILE" ]; then
        echo "Coda esistente ($(wc -l < "$CHAIN_FILE") job):"
        cat -n "$CHAIN_FILE"
    elif [ -f "$FAILED_FILE" ]; then
        local fjob ftype fcfg ftag fext
        fjob=$(cat "$FAILED_FILE")
        ftype=$(echo "$fjob" | cut -d: -f1)
        fcfg=$(echo "$fjob" | cut -d: -f2)
        ftag=$(echo "$fjob" | cut -d: -f3)
        fext=$(echo "$fjob" | cut -d: -f4-)
        if [ "$ftype" != "train" ] && [ "$ftype" != "eval" ]; then
            echo "❌ chain_failed malformato: $fjob"
            exit 1
        fi
        if [ "$ftype" = "train" ]; then
            [ -n "$fext" ] || fext="--resume"
            printf 'train:%s:%s:%s\neval:%s:%s\n' "$fcfg" "$ftag" "$fext" "$fcfg" "$ftag" > "$CHAIN_FILE"
            echo "→ Ricostruita da .chain_failed: train $ftag ($fext) + eval"
        else
            printf 'eval:%s:%s\n' "$fcfg" "$ftag" > "$CHAIN_FILE"
            echo "→ Ricostruita da .chain_failed: eval $ftag"
        fi
        rm -f "$FAILED_FILE"
    else
        echo "❌ Nessuna coda da riprendere (job_chain vuoto, nessun chain_failed)."
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
