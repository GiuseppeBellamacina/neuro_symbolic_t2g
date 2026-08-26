#!/bin/bash
# ============================================================================
# Lancia training + evaluation per il modello T2G in catena (self-chaining).
#
# La QoS permette un solo job alla volta (1 attivo, 0 pending), quindi la
# catena viene avanzata da chain_tick.sh (one-shot idempotente), guidata da:
#   bashrc-hook (PRIMARIO) → chain_tick.sh --quiet via PROMPT_COMMAND
#                            (chain-hook-install) — su gcluster `at` NON c'è
#   watcher (fallback auto)→ chain_next.sh setsid (può essere ucciso dal reaper)
#   at (opportunistico)    → se un giorno `at` comparisse sul login node,
#                            chain_tick.sh --schedule lo userebbe senza danni
#
# MAI rm -rf automatico dello stato con job pendenti: se una catena risulta
# interrotta (job_chain non vuota, nessun job attivo, nessun tick/watcher)
# run_all RIFIUTA e chiede chain-resume (o --force per ricominciare).
#
# Uso:
#   bash cluster/run_all.sh                          # train+eval (default: grpo_optimal)
#   bash cluster/run_all.sh grpo_qwen05              # train+eval con config specifico
#   bash cluster/run_all.sh --ablation               # ablation study completo
#   bash cluster/run_all.sh --eval-only              # solo evaluation
#   bash cluster/run_all.sh --train-only             # solo training
#   bash cluster/run_all.sh --resume                 # riparte dalla coda esistente
#   bash cluster/run_all.sh --append                 # aggiungi job alla coda attiva
#   bash cluster/run_all.sh --remove                 # svuota la coda
#   bash cluster/run_all.sh --force                  # azzera lo stato (catena interrotta)
#
# Config specifici (passa il nome senza .yaml):
#   bash cluster/run_all.sh grpo_qwen05              # config base
#   bash cluster/run_all.sh grpo_optimal             # config ottimale (default)
#   bash cluster/run_all.sh sft                      # SFT baseline
#   bash cluster/run_all.sh grpo_no_grammar          # ablation senza grammar
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
#   tail -f logs/chain_watcher.log           # log del tick/watcher
#   monitor                                  # monitor live
#   myjobs                                   # job attivo su SLURM
#
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
            echo "  (nessun argomento)  Default: grpo_optimal (train + eval)"
            echo "  config_name         Nome del config senza .yaml (es. grpo_qwen05)"
            echo "  --ablation          Ablation study completo (12 varianti)"
            echo "  --eval-only         Solo evaluation (skip training)"
            echo "  --train-only        Solo training (skip eval)"
            echo "  --resume            Riprendi dalla coda esistente (non richiede chain_failed)"
            echo "  --append            Aggiungi job alla coda attiva"
            echo "  --remove            Svuota la coda"
            echo "  --force             Azzera lo stato anche se ci sono job pendenti"
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
    # Ablation study: 12 varianti in ordine (dai più semplici ai più complessi)
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
    TAG=$(basename "$CONFIG_PATH" .yaml | tr '_' '-')
    MODELS=("${TAG}:${CONFIG_PATH}")
else
    # Default: config ottimale. Il TAG deve essere COERENTE col config
    # (il default storico "qwen05" non matchano output_dir/clean_model).
    MODELS=("grpo-optimal:experiments/configs/t2g/grpo_optimal.yaml")
fi

mkdir -p "$STATE_DIR" logs

# ── Funzioni di lancio ────────────────────────────────────────────────────────
# Kick della pipeline. PRIMARIO su gcluster: `at` NON è disponibile → il
# watcher viene avviato subito come fallback automatico e l'HOME hook
# (chain-hook-install) è la resilienza raccomandata. `at` resta solo come
# rilevamento opportunistico: se un giorno comparisse sul login node, il tick
# --schedule lo userebbe senza alcun danno (dedup ≤1 pending).
_launch_pipeline() {
    mkdir -p logs

    if command -v at >/dev/null 2>&1; then
        bash cluster/chain_tick.sh --quiet --schedule=3
        local rc=$?
        if [ "$rc" -ne 3 ]; then
            [ "$rc" -ne 0 ] && echo "⚠️  tick rc=$rc (sottomissione fallita) — retry automatico al prossimo tick."
            echo ""
            echo "✅ Pipeline attiva in at-mode (tick ogni 3 min via 'at')."
            echo "   atq                   → tick schedulati (max 1)"
            echo "   chain-stop            → ferma (cancella anche i tick at)"
            echo "   tail -f logs/chain_watcher.log"
            return 0
        fi
        echo "⚠️  at non raggiungibile (rc=3) — fallback watcher."
    fi

    local pid=""
    setsid nohup bash cluster/chain_next.sh >> logs/chain_watcher.log 2>&1 &
    disown
    sleep 2
    [ -f "$CHAIN_PID_FILE" ] && pid=$(cat "$CHAIN_PID_FILE")
    echo ""
    echo "Pipeline avviata con watcher fallback (PID ${pid:-?})."
    echo ""
    echo "⚠️  at NON è disponibile su gcluster — il watcher può essere ucciso dal"
    echo "    reaper del login node. Per la resilienza installa l'HOME hook:"
    echo ""
    echo "        chain-hook-install && source ~/.bashrc"
    echo ""
    echo "    (e se la catena si ferma comunque: chain-resume)"
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

    # Pulisci pid stale del vecchio watcher
    if [ -f "$CHAIN_PID_FILE" ] && ! watcher_alive; then
        rm -f "$CHAIN_PID_FILE"
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
    if watcher_alive || [ -n "$(active_job_id)" ] || _at_tick_pending; then
        echo "⚠️  Pipeline già attiva (watcher / job SLURM / at-tick)."
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
# Anti-rm-rf: se la coda esiste non vuota, nessun watcher, nessun job attivo
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
    if ! watcher_alive && [ -z "$(active_job_id)" ] && ! _at_tick_pending; then
        echo "✅ Nessun driver attivo — avvio la pipeline."
        _launch_pipeline
    else
        echo "✅ Il driver attivo (watcher/at-tick) eseguirà i nuovi job automaticamente."
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

# ── Avvia la pipeline (hook/watcher; at solo se opportunisticamente presente) ─
_launch_pipeline

echo ""
echo "============================================"
echo "  Pipeline avviata!"
echo "  Log:  logs/chain_watcher.log"
echo "  Coda: .chain_state/job_chain"
echo ""
echo "  Per monitorare:"
echo "    tail -f logs/chain_watcher.log"
echo "    monitor"
echo "    myjobs"
echo ""
echo "  Per interrompere:"
echo "    chain-stop"
echo "    killalljobs"
echo "============================================"
