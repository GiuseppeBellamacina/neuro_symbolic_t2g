#!/bin/bash
# ============================================================================
# SLURM batch script — T2G Training sul cluster (GRPO o SFT)
#
# Rileva automaticamente il tipo di training dal YAML (training.trainer: sft|grpo).
#
# Uso:
#   CONFIG=experiments/configs/qwen25-05b/sft-grpo/few-shot.yaml sbatch cluster/train.sh
#   CONFIG=experiments/configs/qwen25-05b/sft/zero-shot.yaml sbatch cluster/train.sh
#   CONFIG=experiments/configs/qwen25-05b/sft-grpo/few-shot.yaml EXTRA_ARGS="--resume" sbatch cluster/train.sh
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
# Walltime esplicito (prima valeva il default della partizione: un job poteva
# essere ucciso a metà). QoS gpu-xlarge = 12h MAX (CLUSTER.md §"Vincoli del
# cluster"): oltre, sbatch RIFIUTA il job alla sottomissione. Calcolo:
#   GRPO: 5000 passi × ~4,3 s/step ≈ 6h, più setup (model load, prepare_data)
#   e salvataggi → ~6,5h con margine. 11:45:00 lascia ~15 min sotto il cap
#   QoS per il salvataggio finale.
#   NB celle sft-grpo SENZA adapter SFT riusabile (fingerprint): la Phase 0
#   SFT (3 epoche ≈ 13.700 passi) si somma e può sfiorare il cap — in quel
#   caso addestrare prima sft/zero-shot (adapter riusato) o contare su
#   --resume dopo un eventuale TIMEOUT (save_steps: 500 nel config).
#SBATCH --time=11:45:00
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
    echo "  CONFIG=experiments/configs/qwen25-05b/sft-grpo/few-shot.yaml sbatch cluster/train.sh"
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

# Ambiente offline centralizzato (_lib.sh): va esportato PRIMA di qualunque
# python/apptainer, perche i client HF leggono queste variabili all import.
export_offline_env

# Prepara dataset/vocab/bigram se mancanti (funzione shared da _lib.sh,
# idempotente — era triplicata tra setup.sh/train.sh/eval.sh).
# set -e qui: se la preparazione fallisce, il job fallisce LOUD (niente
# training silenzioso su dati mancanti).
prepare_data

# ── Offline-first: i compute node NON hanno DNS ──────────────────────────────
# Tutto il necessario è pre-cacheato da setup.sh/prepare_data. Senza questi
# export ogni richiesta hub costa ~30s di retry (HEAD dataset/modello, lookup
# peft al save) o può CRASHARE (transformers 5.3 tokenizer _patch_mistral_regex
# → model_info → ConnectError, vedi slurm-eval-7077). Con HF_HUB_OFFLINE=1
# transformers tratta ogni modello come locale e salta i check di rete.
# DOPO prepare_data: il fallback download al primo avvio conserva la rete.
# (export offline gia effettuato sopra da export_offline_env)

echo ""
echo "Avvio training..."
echo ""

# ── Esecuzione ────────────────────────────────────────────────────────────────
# Se Apptainer è disponibile, usalo
if command -v apptainer &>/dev/null && [ -f /shared/sifs/latest.sif ]; then
    apptainer run --nv \
        --env WANDB_MODE=offline \
        --env PYTHONUNBUFFERED=1 \
        --env HF_HUB_OFFLINE=1 \
        --env TRANSFORMERS_OFFLINE=1 \
        --env HF_DATASETS_OFFLINE=1 \
        --env PYTORCH_ALLOC_CONF=garbage_collection_threshold:0.8 \
        /shared/sifs/latest.sif \
        python -m src.training --config "${CONFIG}" ${EXTRA_ARGS}
else
    export PYTHONUNBUFFERED=1
    python -m src.training --config "${CONFIG}" ${EXTRA_ARGS}
fi

echo ""
echo "============================================"
echo "  Training completato!"
echo "  $(date)"
echo "============================================"
