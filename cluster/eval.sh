#!/bin/bash
# ============================================================================
# SLURM batch script — T2G Evaluation sul cluster
#
# Uso:
#   CONFIG=experiments/configs/t2g/grpo_qwen05.yaml sbatch cluster/eval.sh
#   CONFIG=experiments/configs/t2g/grpo_qwen05.yaml CHECKPOINT="path/to/ckpt" sbatch cluster/eval.sh
#   CONFIG=experiments/configs/t2g/grpo_qwen05.yaml CHECKPOINT="path/to/ckpt" BEST_OF_N=1 sbatch cluster/eval.sh
#
# --compare è sempre attivo: valuta baseline (zero-shot) + GRPO e genera
#   grafici di confronto + comparison.json + wandb con tag dedicati.
# BEST_OF_N=1 abilita la selezione best-of-N: è un ORACOLO DIAGNOSTICO
#   (limite superiore: quanto può essere buono il modello scegliendo il
#   migliore di N campioni), NON la metrica primaria — quella resta Pass@1.
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
    echo "  CONFIG=experiments/configs/t2g/grpo_qwen05.yaml sbatch cluster/eval.sh"
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

# ── Auto-detect trained checkpoint (config-heritage-aware) ────────────────────
# Risolve output_dir con src.utils.config.resolve_config (che gestisce
# `extends: base.yaml`) — NIENTE yaml.safe_load del solo file figlio.
# Il python gira DENTRO Apptainer sul compute node (stessa pattern di train.sh).
#
# Politica anti-silent-zero-shot:
#   - config di training (ha training.output_dir) ma NESSUN checkpoint trovato
#     → FAIL LOUD (exit 1). Un eval in zero-shot su modello non addestrato
#     produrrebbe numeri senza senso senza alcun errore.
#   - config eval-only (zero_shot*, SENZA training.output_dir) → zero-shot
#     legittimo e voluto.
resolve_output_dir() {
    local out=""
    if command -v apptainer >/dev/null 2>&1 && [ -f /shared/sifs/latest.sif ]; then
        out=$(apptainer exec /shared/sifs/latest.sif python -c "
from src.utils.config import resolve_config
try:
    print(resolve_config('${CONFIG}')['training']['output_dir'])
except Exception:
    print('')
" 2>/dev/null) || true
    else
        out=$(python3 -c "
from src.utils.config import resolve_config
try:
    print(resolve_config('${CONFIG}')['training']['output_dir'])
except Exception:
    print('')
" 2>/dev/null) || true
    fi
    echo "$out"
}

# Trova il checkpoint più recente sotto output_dir: run_*/final o
# run_*/checkpoint-* ; poi output_dir/final o output_dir/checkpoint-*.
# Stesso ordine del legacy, ma su un output_dir RISOLTO (extends-aware).
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
    if [ -z "$best" ]; then
        if [ -d "$out_dir/final" ]; then
            best="$out_dir/final"
        else
            for c in "$out_dir"/checkpoint-*; do
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
            echo "  output_dir:    ${OUTPUT_DIR}   (risolto via resolve_config)"
            echo ""
            echo "  La config dichiara training.output_dir ma non esiste"
            echo "  nessun run_*/final|checkpoint-*: il modello NON è stato"
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
        echo "ℹ️  Config senza training.output_dir (eval-only, es. zero_shot*) — zero-shot inteso"
    fi
fi

# Prepara dataset/vocab/bigram se mancanti (funzione shared da _lib.sh,
# idempotente — era triplicata tra setup.sh/train.sh/eval.sh)
prepare_data
EVAL_ARGS="--config ${CONFIG} --plot --compare"
if [ -n "$CHECKPOINT" ]; then
    EVAL_ARGS="${EVAL_ARGS} --checkpoint ${CHECKPOINT}"
else
    echo "Zero-shot mode: nessun checkpoint (base model pulito)"
fi

# Best-of-N selection (opzionale — passa BEST_OF_N=1 per attivare)
# ORACOLO DIAGNOSTICO: misura il limite superiore della qualità selezionando
# il miglior completamento tra i N campionati. Non è la metrica primaria
# (quella resta Pass@1). Richiede evaluation.num_samples>1 nel config.
if [ "${BEST_OF_N}" = "1" ]; then
    EVAL_ARGS="${EVAL_ARGS} --best-of-n"
    echo "Best-of-N selection enabled"
fi


echo ""
echo "Avvio evaluation..."
echo "  Args: ${EVAL_ARGS}"
echo ""

# ── Esecuzione ────────────────────────────────────────────────────────────────
if command -v apptainer &>/dev/null && [ -f /shared/sifs/latest.sif ]; then
    apptainer run --nv \
        --env WANDB_MODE=offline \
        --env PYTORCH_ALLOC_CONF=garbage_collection_threshold:0.8 \
        /shared/sifs/latest.sif \
        python -m src.training.eval_t2g ${EVAL_ARGS}
else
    python -m src.training.eval_t2g ${EVAL_ARGS}
fi

echo ""
echo "============================================"
echo "  Evaluation completata!"
echo "  $(date)"
echo "============================================"
