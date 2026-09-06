#!/bin/bash
# ============================================================================
# Setup one-tantum per il cluster — SOLO VERIFICA (verify-only).
#
# Uso (dal login node):
#   cd ~/neuro_symbolic_t2g
#   bash cluster/setup.sh
#
# Lo script rilancia se stesso dentro srun + Apptainer automaticamente e poi
# VERIFICA soltanto: nessun `pip install`, nessun download HF, nessun
# load_dataset con download, nessuna eccezione di download "ingoiata".
#
# ⚠️  I compute node NON hanno internet: l'acquisizione di dipendenze,
#     modelli e dataset deve avvenire in un setup separato CON RETE
#     (workflow di login / macchina locale) e va sincronizzata sul cluster
#     prima di sottomettere train/eval. Questo script NON crea alcun
#     workflow python sul login node.
#
# Opzionale:
#   CONFIG=experiments/configs/qwen25-05b/sft-grpo/zero-shot.yaml bash cluster/setup.sh
#   → verifica anche lo snapshot del modello specifico della config.
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

# ── Offline env PRIMA di ogni python (verify-only: tutto locale) ─────────────
export_offline_env

echo "=== Setup Neuro-Symbolic T2G (Cluster) — VERIFY-ONLY ==="
echo ""
echo "ℹ️  I compute node NON hanno internet: questo script NON installa"
echo "   nulla e NON scarica nulla. Verifica soltanto gli artifact già"
echo "   presenti (dipendenze nell'immagine, cache HF, vocab/bigram)."
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

# ── 2. Verifica artifact offline (fail-fast, NESSUN download) ─────────────────
# Al posto di preparare dati o scaricare il modello, qui si verifica
# SOLO che dataset/vocab/bigram/modello siano già presenti nelle path condivise.
# CONFIG (opzionale) restringe la verifica dello snapshot HF al modello della
# config; senza CONFIG si verifica che la cache HF contenga almeno una snapshot.
echo ""
echo "📦 Verifica artifact offline (dataset, vocab, bigram, modello HF)..."
# setup.sh gira già DENTRO il container → forza python bare (RUN_PY_FORCE_BARE).
RUN_PY_FORCE_BARE=1 require_cluster_artifacts "${CONFIG:-}"

# ── 3. Verifica import/versioni locali (nessun download) ──────────────────────
echo ""
echo "🔍 Verifica installazione (import locali, zero rete)..."
$PY -c "
import os
assert os.environ.get('HF_HUB_OFFLINE') == '1', 'HF_HUB_OFFLINE non è 1: env offline non attiva'
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
echo "=== ✅ Verifica completata! ==="
echo ""
echo "💡 Per aggiungere ~/.local/bin al PATH in modo persistente:"
echo "   source cluster/aliases.sh && install-aliases"
echo ""
echo "⚠️  ACQUISIZIONE DIPENDENZE/MODELLI/DATASET — fuori dai compute node:"
echo "   I compute node non hanno internet: NESSUN pip install, NESSUN"
echo "   download HF, NESSUN load_dataset con download è possibile da qui."
echo "   Tutto va acquisito in un AMBIENTE SEPARATO CON RETE (workflow di"
echo "   login con immagine apportata / macchina locale) e sincronizzato"
echo "   sul cluster PRIMA di sottomettere train/eval:"
echo ""
echo "     1. dipendenze Python  → già bakeate nell'immagine Apptainer"
echo "        (aggiornare/ricostruire il SIF fuori dai compute node);"
echo "     2. cache HF           → dataset ASLG-PC12 + snapshot del modello"
echo "        in $T2G_HF_HOME_DEFAULT (condivisa su NFS);"
echo "     3. artifact dati      → data/gloss_vocab.txt,"
echo "        data/bigram_transition.npy (+ sidecar *.meta.json) in $PROJ_DIR/data."
echo ""
echo "Prossimi passi:"
echo "  1. Modifica cluster/train.sh con la tua queue, email e QoS"
echo "  2. Verifica rapida: bash cluster/preflight.sh"
echo "  3. Lancia SFT: CONFIG=experiments/configs/qwen25-05b/sft/zero-shot.yaml sbatch cluster/train.sh"
echo "  4. Oppure lancia pipeline completa: bash cluster/run_all.sh"
