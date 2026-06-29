#!/bin/bash
# ============================================================================
# Setup one-tantum per il cluster.
#
# Uso (dal login node):
#   cd ~/neuro_symbolic_t2g
#   bash cluster/setup.sh
#
# Lo script rilancia se stesso dentro srun + Apptainer automaticamente.
# ============================================================================

# ── 0. Auto-rilancio dentro srun + Apptainer se siamo sul login node ─────────
if [ -z "$APPTAINER_CONTAINER" ]; then
    echo "🚀 Login node rilevato → rilancio inside srun + Apptainer..."
    ACCOUNT="${SLURM_ACCOUNT:-thesis-course}"
    exec srun --account "$ACCOUNT" --partition "$ACCOUNT" --qos gpu-xlarge \
         --gres=gpu:1 --gres=shard:22000 --mem=48G --cpus-per-task=8 \
         apptainer run --nv /shared/sifs/latest.sif \
         bash "$0" "$@"
fi

set -e

echo "=== Setup Neuro-Symbolic T2G (Cluster) ==="
echo ""

# ── 1. Verifica ambiente ──────────────────────────────────────────────────────
if command -v python3 &>/dev/null; then
    PY=python3
elif command -v python &>/dev/null; then
    PY=python
else
    echo "❌ Python non trovato nel container!"
    exit 1
fi
echo "   Python: $($PY --version 2>&1)"
if command -v nvidia-smi &>/dev/null; then
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | head -1
fi

# ── 2. Installa dipendenze dal pyproject.toml ─────────────────────────────────
echo ""
echo "📦 Installazione dipendenze..."
$PY -m pip install --user -e . --retries 10 --timeout 60

# ── 3. Scarica e processa il dataset ASLG-PC12 ────────────────────────────────
echo ""
echo "📊 Download e processing dataset ASLG-PC12..."

$PY -c "
from src.datasets.aslg_dataset import (
    download_aslg_dataset,
    extract_gloss_vocabulary,
    save_vocabulary,
    build_t2g_dataset,
)

dataset = download_aslg_dataset()
vocab = extract_gloss_vocabulary(dataset, split='train')
save_vocabulary(vocab, 'data/gloss_vocab.txt')
print(f'  Gloss unici: {len(vocab)}')

# Build dataset splits
train_ds = build_t2g_dataset(dataset, split='train')
test_ds = build_t2g_dataset(dataset, split='test')
print(f'  Train samples: {len(train_ds)}')
print(f'  Test samples: {len(test_ds)}')
train_ds.save_to_disk('data/aslg_pc12_train')
test_ds.save_to_disk('data/aslg_pc12_test')
print('Dataset salvato in data/')
" || echo "⚠️  Dataset processing fallito — verrà fatto al primo training"

# ── 4. Calcola matrici di transizione ─────────────────────────────────────────
echo ""
echo "📊 Calcolo matrici di transizione bigram..."

$PY -c "
from src.datasets.aslg_dataset import download_aslg_dataset, load_vocabulary
from src.datasets.transition_matrix import compute_bigram_transitions, save_transition_matrix

vocab = load_vocabulary('data/gloss_vocab.txt')
dataset = download_aslg_dataset()

bigram = compute_bigram_transitions(dataset, vocab, split='train', smoothing=1.0)
save_transition_matrix(bigram, 'data/bigram_transition.npy')
print(f'  Bigram matrix: {bigram.shape}')
print(f'  Salvato in data/bigram_transition.npy')
" || echo "⚠️  Matrici di transizione non calcolate"

# ── 5. Verifica installazione ─────────────────────────────────────────────────
echo ""
echo "🔍 Verifica installazione..."
$PY -c "
import torch, transformers, trl, peft, datasets
print(f'  PyTorch:       {torch.__version__}')
print(f'  CUDA:          {torch.cuda.is_available()}')
print(f'  Transformers:  {transformers.__version__}')
print(f'  TRL:           {trl.__version__}')
print(f'  PEFT:          {peft.__version__}')
print(f'  Datasets:      {datasets.__version__}')
"

echo ""
echo "=== ✅ Setup completato! ==="
echo ""
echo "💡 Per aggiungere ~/.local/bin al PATH in modo persistente:"
echo "   source cluster/aliases.sh && install-aliases"
echo ""
echo "Prossimi passi:"
echo "  1. Modifica cluster/train.sh con la tua queue, email e QoS"
echo "  2. Lancia: sbatch cluster/train.sh"
echo "  3. Oppure lancia pipeline completa: bash cluster/run_all.sh"
