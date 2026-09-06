#!/bin/bash
# ============================================================================
# SLURM batch script — T2G Training sul cluster (GRPO o SFT)
#
# Rileva automaticamente il tipo di training dal YAML (training.trainer: sft|grpo).
#
# Uso:
#   CONFIG=experiments/configs/qwen25-05b/sft-grpo/zero-shot.yaml sbatch cluster/train.sh
#   CONFIG=experiments/configs/qwen25-05b/sft/zero-shot.yaml sbatch cluster/train.sh
#   CONFIG=experiments/configs/qwen25-05b/sft-grpo/zero-shot.yaml EXTRA_ARGS="--resume" sbatch cluster/train.sh
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
if [[ "$CONFIG" == *"/ablations/rewards/"* ]] && [[ "$EXTRA_ARGS" != *"--reward-qualification-report"* ]]; then
    echo "❌ Reward ablations require EXTRA_ARGS='--reward-qualification-report PATH'" >&2
    exit 2
fi

if [ -z "$CONFIG" ]; then
    echo "❌ CONFIG non impostato. Uso:"
    echo "  CONFIG=experiments/configs/qwen25-05b/sft-grpo/zero-shot.yaml sbatch cluster/train.sh"
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
echo "  T2G Training — Cluster"
echo "  Job ID:    ${SLURM_JOB_ID}"
echo "  Node:      $(hostname)"
echo "  Date:      $(date)"
echo "  Config:    ${CONFIG}"
echo "  Extra:     ${EXTRA_ARGS}"
echo "============================================"

mkdir -p logs

# ── Offline env PRIMA di ogni python/apptainer ───────────────────────────────
# HF_HUB_OFFLINE/TRANSFORMERS_OFFLINE/HF_DATASETS_OFFLINE/WANDB_*: i compute
# node non hanno DNS — con questi export transformers/datasets trattano ogni
# modello come locale, saltano i check di rete (nessun retry ~30s, nessun
# ConnectError: vedi slurm-eval-7077) e W&B resta offline. huggingface_hub
# legge i flag all'import: devono precedere la PRIMA invocazione python.
export_offline_env

# ── Verifica artifact offline (fail-fast, NESSUN download) ───────────────────
# Sostituisce la vecchia prepare_data: sui compute node il download/la
# rigenerazione NON è possibile. Manca qualcosa → il job fallisce in pochi
# secondi con le istruzioni per preparare/caricare gli artifact da un
# ambiente con rete.
require_cluster_artifacts "$CONFIG"

echo ""
echo "Avvio training..."
echo ""

# ── Esecuzione ────────────────────────────────────────────────────────────────
# Se Apptainer è disponibile, usalo (env offline passato esplicitamente)
if command -v apptainer &>/dev/null && [ -f /shared/sifs/latest.sif ]; then
    apptainer run --nv \
        --env "HF_HUB_OFFLINE=${HF_HUB_OFFLINE}" \
        --env "TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE}" \
        --env "HF_DATASETS_OFFLINE=${HF_DATASETS_OFFLINE}" \
        --env "WANDB_MODE=${WANDB_MODE}" \
        --env "WANDB_DISABLE_WEAVE=${WANDB_DISABLE_WEAVE}" \
        --env "WANDB_SILENT=${WANDB_SILENT}" \
        --env "PYTHONUNBUFFERED=${PYTHONUNBUFFERED}" \
        --env "HF_HOME=${HF_HOME}" \
        --env "HF_HUB_CACHE=${HF_HUB_CACHE}" \
        --env "PYTORCH_ALLOC_CONF=garbage_collection_threshold:0.8" \
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
