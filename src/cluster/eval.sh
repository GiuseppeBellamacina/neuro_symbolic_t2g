#!/bin/bash
# ============================================================================
# SLURM batch script — T2G Evaluation sul cluster
#
# Uso:
#   CONFIG=config/grpo_t2g_qwen05.yaml sbatch src/cluster/eval.sh
#   CONFIG=config/grpo_t2g_qwen05.yaml CHECKPOINT="path/to/ckpt" sbatch src/cluster/eval.sh
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
SKIP_STAGES="${SKIP_STAGES:-0}"

if [ -z "$CONFIG" ]; then
    echo "❌ CONFIG non impostato. Uso:"
    echo "  CONFIG=config/grpo_t2g_qwen05.yaml sbatch src/cluster/eval.sh"
    exit 1
fi

# ── Setup ambiente ───────────────────────────────────────────────────────────
set -e

echo "============================================"
echo "  T2G Evaluation — Cluster"
echo "  Job ID:    ${SLURM_JOB_ID}"
echo "  Node:      $(hostname)"
echo "  Date:      $(date)"
echo "  Config:    ${CONFIG}"
echo "  Checkpoint: ${CHECKPOINT:-auto}"
echo "============================================"

mkdir -p logs

cd "$HOME/neuro_symbolic_t2g"

# Prepara dataset eval se mancante
if [ ! -d "data/aslg_pc12_test" ]; then
    echo "Dataset test non trovato, download in corso..."
    python3 -c "
from src.data.aslg_dataset import ASLGDataPipeline
pipeline = ASLGDataPipeline()
pipeline.download()
test_ds = pipeline.build_hf_dataset('test')
test_ds.save_to_disk('data/aslg_pc12_test')
print('Dataset salvato.')
"
fi

EVAL_ARGS="--config ${CONFIG}"
if [ -n "$CHECKPOINT" ]; then
    EVAL_ARGS="${EVAL_ARGS} --checkpoint ${CHECKPOINT}"
fi
if [ "$SKIP_STAGES" != "0" ] && [ -n "$SKIP_STAGES" ]; then
    EVAL_ARGS="${EVAL_ARGS} --skip-stages ${SKIP_STAGES}"
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
