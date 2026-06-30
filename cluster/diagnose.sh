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

# ── 4b. File trl che referenziano llm_blender ────────────────────────────
echo "── 4b. File trl che importano llm_blender ──"
if [ -n "$TRL_PATH" ] && [ -d "$TRL_PATH" ]; then
    echo "  (con contesto: -B3 -A3 per vedere se l'import è condizionale)"
    echo ""
    MATCHES=$(grep -rn -B 3 -A 3 "llm_blender" "$TRL_PATH" --include="*.py" 2>/dev/null)
    if [ -n "$MATCHES" ]; then
        echo "$MATCHES"
    else
        echo "  (nessun riferimento a llm_blender trovato)"
    fi
else
    echo "  ⚠️  trl path non disponibile"
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
# ── 6b. Wheels di trl nella cache pip ────────────────────────────────────
echo ""
echo "  Wheels di trl nella cache:"
find "$PIP_CACHE" -name "trl*.whl" -type f 2>/dev/null | while read whl; do
    echo "    $(basename "$whl")  ($(stat -c%s "$whl" 2>/dev/null || echo '?') bytes)"
    echo "    SHA256: $(sha256sum "$whl" 2>/dev/null | cut -d' ' -f1 || echo 'N/A')"
done
echo ""

# ── 7. Test import ────────────────────────────────────────────────────────
echo "── 7. Test import catena trl → GRPOTrainer ──"
$PY -c "
import sys, traceback
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
    print(f'FAIL')
    tb = traceback.format_exc()
    # Mostra solo le ultime 15 righe del traceback (la parte rilevante)
    lines = tb.strip().split('\n')
    for line in lines[-20:]:
        print(f'    {line}')
    sys.exit(1)

print('  Importing mergekit...', end=' ')
try:
    import mergekit
    print(f'OK (v{mergekit.__version__})')
except Exception as e:
    print(f'FAIL: {e}')
    sys.exit(1)

print('  Importing llm_blender...', end=' ')
try:
    import llm_blender
    print(f'OK')
except Exception as e:
    print(f'FAIL: {e}')

print()
print('✅ Tutti gli import OK')
"
echo ""

# ── 8. Caccia ai moduli esterni importati da trl (potenziali dipendenze nascoste) ──
echo "── 8. Moduli esterni referenziati nei file .py di trl ──"
TRL_PATH=$($PY -c "import trl; print(trl.__path__[0])" 2>/dev/null)
if [ -n "$TRL_PATH" ] && [ -d "$TRL_PATH" ]; then
    echo "  Cercando import di moduli non-stdlib in $TRL_PATH..."
    echo ""
    # Cerca tutti gli "import X" e "from X import" che NON siano:
    # - import interni di trl (trl.X, .X, ..X)
    # - moduli stdlib Python
    # - torch, transformers, accelerate, datasets (core deps)
    $PY -c "
import re, os, sys

TRL_PATH = '$TRL_PATH'
stdlib = set(sys.stdlib_module_names)
# Moduli core che sappiamo essere già dipendenze
core_deps = {'torch', 'transformers', 'accelerate', 'datasets', 'peft', 
             'numpy', 'huggingface_hub', 'safetensors', 'tqdm', 'yaml',
             'PIL', 'wandb', 'rich', 'regex', 'scipy', 'pandas'}

# Pattern per trovare import esterni
imports_found = set()
for root, dirs, files in os.walk(TRL_PATH):
    for f in files:
        if f.endswith('.py'):
            fpath = os.path.join(root, f)
            rel = os.path.relpath(fpath, TRL_PATH)
            try:
                with open(fpath, 'r') as fh:
                    content = fh.read()
            except:
                continue
            # Cerca 'from MODULE import' e 'import MODULE'
            for match in re.finditer(r'(?:from\s+)(\w+)(?:\.|\s+import)|(?:^import\s+)([\w.]+)', content, re.MULTILINE):
                mod = match.group(1) or match.group(2)
                if mod and mod not in stdlib and mod != 'trl' and not mod.startswith('_'):
                    if mod not in core_deps:
                        imports_found.add((mod, rel))

# Ordina e mostra
if imports_found:
    print('  Moduli esterni trovati (potrebbero essere opzionali mancanti):')
    for mod, fname in sorted(imports_found):
        print(f'    {mod:25s} ← {fname}')
else:
    print('  Nessun modulo esterno aggiuntivo trovato.')
"
else
    echo "  ⚠️  trl path non trovato"
fi
echo ""

echo "============================================"
echo "  Diagnostica completata"
echo "============================================"
