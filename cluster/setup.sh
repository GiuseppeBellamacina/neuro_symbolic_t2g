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

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=cluster/_lib.sh
source "$SCRIPT_DIR/_lib.sh"
cd "$PROJ_DIR"

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
# TUTTE le dipendenze opzionali TRANNE l'extra "dev" (isort/black/ruff/pytest =
# formattazione e test, volutamente esclusi dal cluster):
#   core      → incluso (scikit-learn incluso: backend retrieval tfidf, default)
#   retrieval → incluso (sentence-transformers: backend minilm opzionale)
#   dev       → ESCLUSO di proposito
# sentence-transformers scaricherà il modello MiniLM in modo LAZY al primo uso
# del backend minilm (retrieval.backend: "minilm"); il default resta tfidf →
# zero costi se non attivi il backend.
echo ""
echo "📦 Installazione dipendenze (core + retrieval, niente dev)..."
$PY -m pip cache purge 2>/dev/null || true
echo "   Cache pip ripulita"
$PY -m pip install --user -e ".[retrieval]" --retries 10 --timeout 60

# ── 3+4. Dataset, vocabolario e matrici di transizione ────────────────────────
# Funzione shared da _lib.sh (era triplicata in setup.sh/train.sh/eval.sh).
# setup.sh gira già DENTRO il container → forza python bare (RUN_PY_FORCE_BARE).
echo ""
echo "📊 Download e processing dataset ASLG-PC12 + matrici bigram..."
RUN_PY_FORCE_BARE=1 prepare_data || echo "⚠️  Dataset processing fallito — verrà fatto al primo training"

# ── 5. Pre-download modello per Unsloth (offline cache) ────────────────────────
echo ""
echo "📥 Download modello Qwen2.5-0.5B-Instruct per la cache offline..."
$PY -c "
try:
    from unsloth import FastLanguageModel
    print('  Download Qwen2.5-0.5B-Instruct BNB 4-bit...')
    FastLanguageModel.from_pretrained('Qwen/Qwen2.5-0.5B-Instruct', load_in_4bit=True)
    print('  ✅ Modello scaricato e salvato nella cache locale.')
except Exception as e:
    print(f'  ⚠️ Errore nel download del modello: {e}')
"

# ── 6. Verifica installazione ─────────────────────────────────────────────────
echo ""
echo "🔍 Verifica installazione..."
$PY -c "
import torch, transformers, trl, peft, datasets, sklearn
print(f'  PyTorch:       {torch.__version__}')
print(f'  CUDA:          {torch.cuda.is_available()}')
print(f'  Transformers:  {transformers.__version__}')
print(f'  TRL:           {trl.__version__}')
print(f'  PEFT:          {peft.__version__}')
print(f'  Datasets:      {datasets.__version__}')
print(f'  scikit-learn:  {sklearn.__version__}')
try:
    import sentence_transformers
    print(f'  sentence-transformers: {sentence_transformers.__version__}')
except Exception as e:
    print('  ⚠️  sentence-transformers NON importabile (extra retrieval mancante?)')
    print(f'      {e}')
    print('      Il backend minilm non sarà disponibile; il default tfidf funziona.')
try:
    import unsloth
    print(f'  Unsloth:       {unsloth.__version__}')
except ImportError:
    print('  Unsloth:       Non installato')
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
