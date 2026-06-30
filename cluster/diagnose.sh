#!/bin/bash
# ============================================================================
# Diagnostica ambiente pip sul cluster — individua conflitti e cache spurie.
#
# Uso (dal login node):
#   cd ~/neuro_symbolic_t2g
#   bash cluster/diagnose.sh
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

echo "============================================"
echo "  Diagnostica ambiente pip — Neuro-Symbolic T2G"
echo "  $(date)"
echo "============================================"
echo ""

# Trova python
if command -v python3 &>/dev/null; then
    PY=python3
elif command -v python &>/dev/null; then
    PY=python
else
    echo "❌ Python non trovato"
    exit 1
fi
echo "Python: $($PY --version 2>&1)"
echo "Pip:    $($PY -m pip --version 2>&1)"
echo ""

# ── 1. Dove sono installati i pacchetti? ──────────────────────────────────
echo "── 1. Percorsi di installazione pacchetti ──"
$PY -c "import site; print('site-packages:'); [print(f'  {p}') for p in site.getsitepackages()]; print('user site:', site.getusersitepackages())"
echo ""

# ── 2. Quale trl è installato e da dove viene? ────────────────────────────
echo "── 2. Pacchetto TRL ──"
$PY -m pip show trl 2>&1 || echo "❌ trl NON installato"
echo ""

# ── 3. mergekit è installato? ─────────────────────────────────────────────
echo "── 3. Pacchetto mergekit ──"
$PY -m pip show mergekit 2>&1 || echo "❌ mergekit NON installato"
echo ""

# ── 4. Il file mergekit_utils.py: come importa mergekit? ──────────────────
echo "── 4. Contenuto mergekit_utils.py (prime 40 righe) ──"
TRL_PATH=$($PY -c "import trl; print(trl.__path__[0])" 2>/dev/null)
if [ -n "$TRL_PATH" ] && [ -f "$TRL_PATH/mergekit_utils.py" ]; then
    echo "  File: $TRL_PATH/mergekit_utils.py"
    echo ""
    head -40 "$TRL_PATH/mergekit_utils.py"
    echo ""
    echo "  ---"
    echo ""
    echo "  Riga 22:"
    sed -n '20,25p' "$TRL_PATH/mergekit_utils.py"
else
    echo "  ⚠️  mergekit_utils.py non trovato in $TRL_PATH"
fi
echo ""

# ── 5. Pacchetti installati in ~/.local ───────────────────────────────────
echo "── 5. Pacchetti in ~/.local ──"
USER_SITE=$($PY -c "import site; print(site.getusersitepackages())")
if [ -d "$USER_SITE" ]; then
    COUNT=$(ls "$USER_SITE" 2>/dev/null | wc -l)
    echo "  $COUNT entries in $USER_SITE"
    echo ""
    echo "  Pacchetti principali:"
    $PY -m pip list --user --format=columns 2>/dev/null | head -30
else
    echo "  ⚠️  $USER_SITE non esiste"
fi
echo ""

# ── 6. Cache pip ──────────────────────────────────────────────────────────
echo "── 6. Cache pip ──"
PIP_CACHE=$($PY -m pip cache dir 2>/dev/null)
echo "  Cache dir: $PIP_CACHE"
if [ -d "$PIP_CACHE" ]; then
    echo "  Dimensione cache: $(du -sh "$PIP_CACHE" 2>/dev/null | cut -f1)"
fi
echo ""

# ── 7. Test import ────────────────────────────────────────────────────────
echo "── 7. Test import catena trl → GRPOTrainer ──"
$PY -c "
import sys
print('  Importing trl...', end=' ')
try:
    import trl
    print(f'OK (v{trl.__version__}, {trl.__path__[0]})')
except Exception as e:
    print(f'FAIL: {e}')
    sys.exit(1)

print('  Importing GRPOTrainer...', end=' ')
try:
    from trl import GRPOTrainer
    print('OK')
except Exception as e:
    print(f'FAIL: {e}')
    sys.exit(1)

print('  Importing mergekit...', end=' ')
try:
    import mergekit
    print(f'OK (v{mergekit.__version__})')
except Exception as e:
    print(f'FAIL: {e}')
    sys.exit(1)

print()
print('✅ Tutti gli import OK')
"
echo ""

echo "============================================"
echo "  Diagnostica completata"
echo "============================================"
