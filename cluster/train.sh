#!/bin/bash
# ============================================================================
# SLURM batch script — T2G GRPO Training sul cluster
#
# Uso:
#   CONFIG=experiments/configs/t2g/grpo_qwen05.yaml sbatch cluster/train.sh
#   CONFIG=experiments/configs/t2g/grpo_qwen05.yaml EXTRA_ARGS="--resume" sbatch cluster/train.sh
#
# Per il primo avvio eseguire prima:  bash cluster/setup.sh
# ============================================================================

# ┌────────────────────────────────────────────────────────┐
# │  CONFIGURA QUI — modifica account/partition/qos/email  │
# └────────────────────────────────────────────────────────┘
#SBATCH --job-name=train-t2g
#SBATCH --account=thesis-course
#SBATCH --partition=thesis-course
#SBATCH --qos=gpu-xlarge
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1 --gres=shard:22528
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=bellamacina50@gmail.com
#SBATCH --output=logs/slurm-train-%j.log

# ── Variabili progetto ────────────────────────────────────────────────────────
EXTRA_ARGS="${EXTRA_ARGS:-}"

if [ -z "$CONFIG" ]; then
    echo "❌ CONFIG non impostato. Uso:"
    echo "  CONFIG=experiments/configs/t2g/grpo_qwen05.yaml sbatch cluster/train.sh"
    exit 1
fi

# ── Setup ambiente ───────────────────────────────────────────────────────────
set -e

echo "============================================"
echo "  T2G GRPO Training — Cluster"
echo "  Job ID:    ${SLURM_JOB_ID}"
echo "  Node:      $(hostname)"
echo "  Date:      $(date)"
echo "  Config:    ${CONFIG}"
echo "  Extra:     ${EXTRA_ARGS}"
echo "============================================"

mkdir -p logs

export WANDB_MODE=offline

cd "$HOME/neuro_symbolic_t2g"

# Prepara dataset se mancante
if [ ! -d "data/aslg_pc12_train" ]; then
    echo "Dataset ASLG-PC12 non trovato, download in corso..."
    python3 -c "
from src.data.aslg_dataset import download_aslg_dataset, build_t2g_dataset
dataset = download_aslg_dataset()
train_ds = build_t2g_dataset(dataset, split='train')
train_ds.save_to_disk('data/aslg_pc12_train')
print('Dataset salvato.')
"
fi

if [ ! -f "data/bigram_transition.npy" ]; then
    echo "Matrici di transizione non trovate, calcolo in corso..."
    python3 -c "
from src.data.aslg_dataset import download_aslg_dataset, load_vocabulary
from src.data.transition_matrix import compute_bigram_transitions, save_transition_matrix
dataset = download_aslg_dataset()
vocab = load_vocabulary('data/gloss_vocab.txt')
bigram = compute_bigram_transitions(dataset, vocab, split='train', smoothing=1.0)
save_transition_matrix(bigram, 'data/bigram_transition.npy')
print('Matrici salvate.')
"
fi

echo ""
echo "Avvio training..."
echo ""

# ── Esecuzione ────────────────────────────────────────────────────────────────
# Se Apptainer è disponibile, usalo
if command -v apptainer &>/dev/null && [ -f /shared/sifs/latest.sif ]; then
    apptainer run --nv \
        --env WANDB_MODE=offline \
        --env PYTORCH_ALLOC_CONF=garbage_collection_threshold:0.8 \
        /shared/sifs/latest.sif \
        python -m src.training --config "${CONFIG}" ${EXTRA_ARGS}
else
    python -m src.training --config "${CONFIG}" ${EXTRA_ARGS}
fi

echo ""
echo "============================================"
echo "  Training completato!"
echo "  $(date)"
echo "============================================"
