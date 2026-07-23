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
# BEST_OF_N=1 attiva la selezione best-of-N (richiede num_samples>1 nel config).
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

# ── Auto-detect trained checkpoint from config YAML ──────────────────────────
# Se CHECKPOINT non è stato passato, cerca automaticamente il modello addestrato
# nel training.output_dir/final (dove il trainer salva il modello finale).
# Per config eval-only (senza training.output_dir) rimane in zero-shot.
if [ -z "$CHECKPOINT" ]; then
    DETECTED=$(python3 -c "
import yaml, os, sys, glob
try:
    cfg = yaml.safe_load(open('${CONFIG}'))
    out_dir = cfg.get('training', {}).get('output_dir', '')
    if out_dir:
        run_folders = sorted(glob.glob(os.path.join(out_dir, 'run_*')))
        if run_folders:
            latest_run = run_folders[-1]
            final_path = os.path.join(latest_run, 'final')
            if os.path.isdir(final_path):
                print(final_path)
                sys.exit(0)
            ckpts = sorted(glob.glob(os.path.join(latest_run, 'checkpoint-*')))
            if ckpts:
                print(ckpts[-1])
                sys.exit(0)
        final_path = os.path.join(out_dir, 'final')
        if os.path.isdir(final_path):
            print(final_path)
            sys.exit(0)
        ckpts = sorted(glob.glob(os.path.join(out_dir, 'checkpoint-*')))
        if ckpts:
            print(ckpts[-1])
            sys.exit(0)
except Exception:
    pass
sys.exit(1)
" 2>/dev/null) && CHECKPOINT="$DETECTED"
    if [ -n "$CHECKPOINT" ]; then
        echo "Auto-detected trained checkpoint: $CHECKPOINT"
    fi
fi

# Prepara dataset eval se mancante — usa Apptainer se disponibile
# (il login node / compute node non ha le dipendenze Python installate)
if [ ! -d "data/aslg_pc12_test" ]; then
    echo "Dataset test non trovato, download in corso..."
    if command -v apptainer &>/dev/null && [ -f /shared/sifs/latest.sif ]; then
        apptainer run --nv /shared/sifs/latest.sif python3 -c "
from src.datasets.aslg_dataset import download_aslg_dataset, build_t2g_dataset
dataset = download_aslg_dataset()
test_ds = build_t2g_dataset(dataset, split='test')
test_ds.save_to_disk('data/aslg_pc12_test')
print('Dataset salvato.')
"
    else
        python3 -c "
from src.datasets.aslg_dataset import download_aslg_dataset, build_t2g_dataset
dataset = download_aslg_dataset()
test_ds = build_t2g_dataset(dataset, split='test')
test_ds.save_to_disk('data/aslg_pc12_test')
print('Dataset salvato.')
"
    fi
fi
EVAL_ARGS="--config ${CONFIG} --plot --compare"
if [ -n "$CHECKPOINT" ]; then
    EVAL_ARGS="${EVAL_ARGS} --checkpoint ${CHECKPOINT}"
else
    echo "Zero-shot mode: nessun checkpoint (base model pulito)"
fi

# Best-of-N selection (opzionale — passa BEST_OF_N=1 per attivare)
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
