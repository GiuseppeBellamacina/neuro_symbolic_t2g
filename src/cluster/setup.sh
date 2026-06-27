#!/bin/bash
# ============================================================================
# Setup one-tantum per il cluster.
#
# Uso (dal login node):
#   cd ~/neuro_symbolic_t2g
#   bash src/cluster/setup.sh
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

# ── 1. Verifica GPU ──────────────────────────────────────────────────────────
echo "🔍 Rilevamento GPU..."

# Trova il comando python disponibile
if command -v python3 &>/dev/null; then
    PY=python3
elif command -v python &>/dev/null; then
    PY=python
else
    echo "❌ Python non trovato nel container!"
    exit 1
fi
echo "   Python: $($PY --version 2>&1)"

# Prima rileva la GPU via nvidia-smi (non richiede torch)
GPU_NAME=""
GPU_VRAM_MB=0
if command -v nvidia-smi &>/dev/null; then
    GPU_INFO_SMI=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | head -1)
    if [ -n "$GPU_INFO_SMI" ]; then
        GPU_NAME=$(echo "$GPU_INFO_SMI" | cut -d',' -f1 | xargs)
        GPU_VRAM_MB=$(echo "$GPU_INFO_SMI" | cut -d',' -f2 | grep -oP '\d+')
        GPU_VRAM_GB=$((GPU_VRAM_MB / 1024))
        echo "   nvidia-smi: $GPU_NAME (~${GPU_VRAM_GB} GB)"
    fi
fi

# Stima Compute Capability dal nome GPU (fallback se torch non disponibile)
estimate_cc() {
    case "$1" in
        *L40S*)    echo "8.9" ;;
        *L40*)     echo "8.9" ;;
        *A100*)    echo "8.0" ;;
        *A40*)     echo "8.6" ;;
        *A30*)     echo "8.0" ;;
        *A10*)     echo "8.6" ;;
        *V100*)    echo "7.0" ;;
        *T4*)      echo "7.5" ;;
        *P100*)    echo "6.0" ;;
        *P40*)     echo "6.1" ;;
        *K80*)     echo "3.7" ;;
        *K40*)     echo "3.5" ;;
        *RTX*40*)  echo "8.9" ;;
        *RTX*30*)  echo "8.6" ;;
        *RTX*20*)  echo "7.5" ;;
        *GTX*10*)  echo "6.1" ;;
        *H100*)    echo "9.0" ;;
        *H200*)    echo "9.0" ;;
        *)         echo "0" ;;
    esac
}

CC_MAJOR=0
CC_MAJOR="${CC_MAJOR:-0}"  # safety default
# Prova con torch (preciso) se disponibile
GPU_INFO=$($PY -c "
import torch
print(f'  PyTorch: {torch.__version__}')
print(f'  CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    name = torch.cuda.get_device_name(0)
    cc = torch.cuda.get_device_capability()
    vram = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f'  GPU: {name} (CC {cc[0]}.{cc[1]}, {vram:.1f} GB)')
    print(f'CC_MAJOR={cc[0]}')
else:
    print('  GPU: NESSUNA GPU rilevata')
    print('CC_MAJOR=0')
" 2>/dev/null) && {
    echo "$GPU_INFO" | grep -v CC_MAJOR
    CC_MAJOR=$(echo "$GPU_INFO" | grep CC_MAJOR | cut -d= -f2)
} || {
    # Fallback: stima CC dal nome GPU rilevato via nvidia-smi
    if [ -n "$GPU_NAME" ]; then
        CC_MAJOR=$(estimate_cc "$GPU_NAME")
        echo "   ⚠️  torch non disponibile — CC stimato da nvidia-smi: $CC_MAJOR (GPU: $GPU_NAME)"
    else
        echo "   ⚠️  Nessuna GPU rilevata (né torch né nvidia-smi). Assumo CC=0 (CPU-only)."
        CC_MAJOR=0
    fi
}

# ── 2. Installa dipendenze ────────────────────────────────────────────────────
echo ""
echo "📦 Installazione dipendenze..."

# Installa torch dall'index CUDA 12.1 (cu121) PRIMA del resto.
# Questo evita che vllm forzi torch cu130, incompatibile col driver CUDA 12.0 del cluster.
# Se torch cu121 NON funziona col tuo driver, contatta l'admin per aggiornare il driver NVIDIA.
echo "   [1/2] torch da index cu121..."
pip install --user torch --index-url https://download.pytorch.org/whl/cu121

echo "   [2/2] Progetto + dipendenze GPU (Unsloth + vLLM) da index cu121..."
# Use --extra-index-url (not --index-url) so pip can fetch
# build dependencies (setuptools, wheel) from the default PyPI
# while still resolving torch-related packages from the cu121 index.
# The [gpu] extra installs unsloth and vllm (optional deps).
pip install --user -e ".[gpu]" --extra-index-url https://download.pytorch.org/whl/cu121

# ── 3. Scarica e processa il dataset ASLG-PC12 ────────────────────────────────
echo ""
echo "📊 Download e processing dataset ASLG-PC12..."

$PY -c "
from src.data.aslg_dataset import (
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
from src.data.aslg_dataset import download_aslg_dataset, load_vocabulary
from src.data.transition_matrix import compute_bigram_transitions, save_transition_matrix

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
try:
    import unsloth
    print(f'  Unsloth:       {unsloth.__version__}')
except (ImportError, NotImplementedError, RuntimeError):
    print(f'  Unsloth:       NON disponibile (GPU/CUDA non compatibile)')
try:
    import vllm
    print(f'  vLLM:          {vllm.__version__}')
except (ImportError, NotImplementedError, RuntimeError):
    print(f'  vLLM:          NON disponibile')
"

echo ""
echo "=== ✅ Setup completato! ==="
echo ""
echo "💡 Per aggiungere ~/.local/bin al PATH in modo persistente:"
echo "   source src/cluster/aliases.sh && t2g-install-aliases"
echo ""
echo "Prossimi passi:"
echo "  1. Modifica src/cluster/train.sh con la tua queue, email e QoS"
echo "  2. Lancia: sbatch src/cluster/train.sh"
echo "  3. Oppure lancia pipeline completa: bash src/cluster/run_all.sh"
