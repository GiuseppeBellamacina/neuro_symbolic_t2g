#!/bin/bash
# ============================================================================
# SLURM batch script — T2G Evaluation sul cluster
#
# Uso:
#   CONFIG=experiments/configs/qwen25-05b/sft-grpo/zero-shot.yaml sbatch cluster/eval.sh
#
# I checkpoint addestrati sono valutati in sequenza zero-shot e retrieval
# (few-shot user-facing), con lo stesso checkpoint e gli stessi parametri.
# BEST_OF_N=1 abilita la selezione best-of-N: è un ORACOLO DIAGNOSTICO
#   (limite superiore: quanto può essere buono il modello scegliendo il
#   migliore di N campioni), NON la metrica primaria — quella resta Pass@1.
#
# OFFLINE: i compute node NON hanno internet. Tutto il necessario
# (dipendenze nell'immagine, cache HF di dataset+modello, vocab/bigram)
# viene preparato in un ambiente con rete separato e sincronizzato sul
# cluster PRIMA della sottomissione. Nessun pip install, nessun download,
# nessun fallback: artifact mancanti → fail-fast immediato.
# ============================================================================

# ┌────────────────────────────────────────────────────────┐
# │  CONFIGURA QUI — modifica account/partition/qos/email  │
# └────────────────────────────────────────────────────────┘
#SBATCH --job-name=eval-t2g
#SBATCH --account=thesis-course
#SBATCH --partition=thesis-course
#SBATCH --qos=gpu-xlarge
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1 --gres=shard:22528
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=bellamacina50@gmail.com
#SBATCH --output=logs/slurm-eval-%j.log

# ── Variabili progetto ────────────────────────────────────────────────────────
CHECKPOINT="${CHECKPOINT:-}"


if [ -z "$CONFIG" ]; then
    echo "❌ CONFIG non impostato. Uso:"
    echo "  CONFIG=experiments/configs/qwen25-05b/sft-grpo/zero-shot.yaml sbatch cluster/eval.sh"
    exit 1
fi

# ── Setup ambiente ───────────────────────────────────────────────────────────
set -e

# SLURM copia lo script nella sua spool dir (/var/lib/slurm/slurmd/<job>/)
# e lo esegue da lì: BASH_SOURCE NON punta al repo e `source _lib.sh`
# fallirebbe ("File o directory non esistente"). La directory di
# sottomissione (SLURM_SUBMIT_DIR — la cwd al momento dello sbatch, che è
# sempre la repo root, sia da shell che da chain_tick.sh) è il posto giusto.
_lib_dir=$(dirname "${BASH_SOURCE[0]}")
if [ ! -f "${_lib_dir}/_lib.sh" ]; then
    _lib_dir="${SLURM_SUBMIT_DIR:-$HOME/neuro_symbolic_t2g}/cluster"
fi
SCRIPT_DIR=$(cd "${_lib_dir}" && pwd)
# shellcheck source=cluster/_lib.sh
source "$SCRIPT_DIR/_lib.sh"
cd "$PROJ_DIR"

echo "============================================"
echo "  T2G Evaluation — Cluster"
echo "  Job ID:    ${SLURM_JOB_ID}"
echo "  Node:      $(hostname)"
echo "  Date:      $(date)"
echo "  Config:    ${CONFIG}"
echo "  Checkpoint: ${CHECKPOINT:-auto}"
echo "============================================"

mkdir -p logs

# ── Offline env PRIMA di ogni python/apptainer (inclusa la risoluzione del
#    checkpoint qui sotto) ─────────────────────────────────────────────────────
# HF_HUB_OFFLINE/TRANSFORMERS_OFFLINE/HF_DATASETS_OFFLINE/WANDB_*: i compute
# node non hanno DNS — con questi export transformers/datasets trattano ogni
# modello come locale, saltano i check di rete (nessun retry ~30s, nessun
# ConnectError: vedi slurm-eval-7077) e W&B resta offline (parità con
# train.sh). huggingface_hub legge i flag all'import: devono precedere la
# PRIMA invocazione python (qui: resolve_output_dir).
export_offline_env

# ── Auto-detect trained checkpoint from canonical experiment identity ─────────
# Il python gira DENTRO Apptainer sul compute node (stessa pattern di train.sh)
# ed eredita l'env offline esportato sopra.
#
# Politica anti-silent-zero-shot:
#   - config kind=train|ablation ma NESSUN checkpoint trovato
#     → FAIL LOUD (exit 1). Un eval in zero-shot su modello non addestrato
#     produrrebbe numeri senza senso senza alcun errore.
#   - config kind=baseline → eval senza checkpoint
#     legittimo e voluto.
resolve_output_dir() {
    local out=""
    out=$(run_py -c "
from src.utils.config import resolve_config
from src.utils.paths import cell_from_config, cell_base_dir
try:
    cfg = resolve_config('${CONFIG}')
    if cfg.get('experiment', {}).get('kind') in {'train', 'ablation'}:
        print(cell_base_dir('experiments', 'checkpoints', cell_from_config(cfg)))
    else:
        print('')
except Exception:
    print('')
")
    echo "$out"
}

resolve_experiment_identity() {
    run_py -c "
from src.utils.config import resolve_config
cfg = resolve_config('${CONFIG}')
exp = cfg.get('experiment', {})
print(f\"{exp.get('kind', '')}|{exp.get('train_prompt_mode', '')}\")
"
}

# Trova il checkpoint più recente sotto la base canonica: run_*/final o
# run_*/checkpoint-*.
find_newest_checkpoint() {
    local out_dir="$1" latest_run best="" c
    latest_run=$(ls -1d "${out_dir}"/run_* 2>/dev/null | tail -1) || true
    if [ -n "$latest_run" ] && [ -d "$latest_run" ]; then
        if [ -d "$latest_run/final" ]; then
            best="$latest_run/final"
        else
            for c in "$latest_run"/checkpoint-*; do
                [ -d "$c" ] && best="$c"
            done
        fi
    fi
    if [ -n "$best" ]; then
        echo "$best"
        return 0
    fi
    return 1
}

if [ -z "$CHECKPOINT" ]; then
    OUTPUT_DIR=$(resolve_output_dir)
    if [ -n "$OUTPUT_DIR" ]; then
        if DETECTED=$(find_newest_checkpoint "$OUTPUT_DIR"); then
            CHECKPOINT="$DETECTED"
            echo "Auto-detected trained checkpoint: $CHECKPOINT"
        else
            echo ""
            echo "═══════════════════════════════════════════════════════════"
            echo "  ❌ CHECKPOINT NON TROVATO — eval RIFIUTATO (exit 1)"
            echo "  Config:        ${CONFIG}"
            echo "  checkpoint base: ${OUTPUT_DIR}"
            echo ""
            echo "  Non esiste nessun run_*/final|checkpoint-* canonico:"
            echo "  il modello NON è stato"
            echo "  addestrato. Un eval in zero-shot su modello non addestrato"
            echo "  produrrebbe numeri senza senso SENZA errori — rifiutiamo."
            echo ""
            echo "  Soluzioni:"
            echo "   1. addestra prima (run-all / chain / sbatch cluster/train.sh)"
            echo "   2. oppure forza un checkpoint esplicito:"
            echo "        CONFIG=${CONFIG} CHECKPOINT=<path> sbatch cluster/eval.sh"
            echo "═══════════════════════════════════════════════════════════"
            echo ""
            exit 1
        fi
    else
        echo "Identità esperimento non risolta - zero-shot inteso"
    fi
fi

EXPERIMENT_IDENTITY=$(resolve_experiment_identity)
EXPERIMENT_KIND=${EXPERIMENT_IDENTITY%%|*}
TRAIN_PROMPT_MODE=${EXPERIMENT_IDENTITY#*|}

# ── Verifica artifact offline (fail-fast, NESSUN download) ───────────────────
# Sostituisce la vecchia prepare_data: sui compute node il download/la
# rigenerazione NON è possibile. Manca qualcosa → il job fallisce in pochi
# secondi con le istruzioni per preparare/caricare gli artifact da un
# ambiente con rete.
require_cluster_artifacts "$CONFIG"

# ── Modalità eval ─────────────────────────────────────────────────────────────
# Baseline: una sola leg, esplicita e coerente con l'identità del config.
# Train/ablation: due leg sul medesimo checkpoint, prima zero-shot e poi
# retrieval. `set -e` rende il job fallito se fallisce una qualsiasi leg.
COMMON_EVAL_ARGS="--config ${CONFIG} --plot"

# Override opzionale del numero di campioni (default: quello del config,
# oggi 2000 da base.yaml). Esempio eval rapido: MAX_SAMPLES=500 CONFIG=...
if [ -n "${MAX_SAMPLES:-}" ]; then
    COMMON_EVAL_ARGS="${COMMON_EVAL_ARGS} --max-samples ${MAX_SAMPLES}"
    echo "MAX_SAMPLES override: ${MAX_SAMPLES}"
fi

if [ -n "$CHECKPOINT" ] && [ "$EXPERIMENT_KIND" != "baseline" ]; then
    COMMON_EVAL_ARGS="${COMMON_EVAL_ARGS} --checkpoint ${CHECKPOINT}"
elif [ "$EXPERIMENT_KIND" = "baseline" ]; then
    echo "Baseline eval-only: eventuale CHECKPOINT ignorato"
else
    echo "Zero-shot mode: nessun checkpoint (base model pulito)"
fi

# Best-of-N selection (opzionale — passa BEST_OF_N=1 per attivare)
# ORACOLO DIAGNOSTICO: misura il limite superiore della qualità selezionando
# il miglior completamento tra i N campionati. Non è la metrica primaria
# (quella resta Pass@1). Richiede evaluation.num_samples>1 nel config.
if [ "${BEST_OF_N:-0}" = "1" ]; then
    COMMON_EVAL_ARGS="${COMMON_EVAL_ARGS} --best-of-n"
    echo "Best-of-N selection enabled"
fi


run_evaluation() {
    local prompt_mode="$1" mode_args
    mode_args="${COMMON_EVAL_ARGS} --prompt-mode ${prompt_mode}"
    if [ "$EXPERIMENT_KIND" = "baseline" ]; then
        mode_args="${mode_args} --eval-baseline-only"
    else
        mode_args="${mode_args} --compare"
    fi

    echo ""
    echo "Avvio evaluation (${prompt_mode})..."
    echo "  Args: ${mode_args}"
    echo ""

    run_py -m src.training.eval_t2g ${mode_args}
}

if [ "$EXPERIMENT_KIND" = "baseline" ]; then
    if [ "$TRAIN_PROMPT_MODE" = "few-shot" ]; then
        run_evaluation retrieval
    else
        run_evaluation zero-shot
    fi
else
    run_evaluation zero-shot
    run_evaluation retrieval
fi

echo ""
echo "============================================"
echo "  Evaluation completata!"
echo "  $(date)"
echo "============================================"
