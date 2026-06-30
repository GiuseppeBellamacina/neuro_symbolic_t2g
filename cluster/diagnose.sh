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

# ── 1. Percorsi ──────────────────────────────────────────────────────────
echo "── 1. Percorsi di installazione pacchetti ──"
$PY -c "import site; print('site-packages:'); [print(f'  {p}') for p in site.getsitepackages()]; print('user site:', site.getusersitepackages())"
echo ""

# ── 2. TRL + mergekit info ───────────────────────────────────────────────
echo "── 2. Pacchetto TRL ──"
$PY -m pip show trl 2>&1 || echo "❌ trl NON installato"
echo ""
echo "── 3. Pacchetto mergekit ──"
$PY -m pip show mergekit 2>&1 || echo "❌ mergekit NON installato"
echo ""

# ── 4. Contenuto mergekit_utils.py ───────────────────────────────────────
TRL_PATH=$($PY -c "import trl; print(trl.__path__[0])" 2>/dev/null)
echo "── 4. mergekit_utils.py (riga 20-25) ──"
if [ -n "$TRL_PATH" ] && [ -f "$TRL_PATH/mergekit_utils.py" ]; then
    sed -n '20,25p' "$TRL_PATH/mergekit_utils.py"
else
    echo "  ⚠️  mergekit_utils.py non trovato"
fi
echo ""
echo "── 5. judges.py (riga 25-32) ──"
if [ -n "$TRL_PATH" ] && [ -f "$TRL_PATH/trainer/judges.py" ]; then
    sed -n '25,32p' "$TRL_PATH/trainer/judges.py"
else
    echo "  ⚠️  judges.py non trovato"
fi
echo ""

# ── 6. La verità: cosa restituisce _is_package_available? ────────────────
echo "── 6. Cosa restituisce _is_package_available (dopo import trl) ──"
$PY -c "
import importlib, importlib.util, importlib.metadata, sys, os

# ── find_spec ──
print('  find_spec:')
for pkg in ['mergekit', 'llm_blender']:
    spec = importlib.util.find_spec(pkg)
    print(f'    {pkg:15s} → {\"FOUND: \" + spec.origin if spec else \"NOT FOUND\"} ')

# ── importlib.metadata distributions ── (dove _is_package_available cerca davvero)
print()
print('  importlib.metadata distributions (mergekit / llm.blender / llm_blender):')
for dist in importlib.metadata.distributions():
    name = dist.metadata['Name'].lower()
    if 'mergekit' in name or 'llm' in name or 'blender' in name:
        try:
            loc = str(dist.locate_file(''))
        except Exception:
            loc = str(getattr(dist, '_path', '?'))
        print(f'    {dist.metadata[\"Name\"]:20s} v{dist.version:10s}  at {loc}')

# ── Importa trl e guarda le variabili interne ──
print()
import trl
print(f'  trl v{trl.__version__}')
from trl.import_utils import _mergekit_available, _llm_blender_available
print(f'  _mergekit_available      = {_mergekit_available}')
print(f'  _llm_blender_available   = {_llm_blender_available}')

# ── Da dove viene _is_package_available? ──
try:
    import transformers.utils.import_utils as tiu
    import inspect
    src = inspect.getsource(tiu._is_package_available)
    lines = src.split('\n')[:15]
    print()
    print('  _is_package_available (prime 15 righe da transformers.utils.import_utils):')
    for line in lines:
        print(f'    {line}')
except Exception:
    print()
    print('  _is_package_available: (source not available)')
"
echo ""

# ── 7. Test import GRPOTrainer ───────────────────────────────────────────
echo "── 7. Test import catena trl → GRPOTrainer ──"
$PY -c "
import sys, traceback
print('  Importing trl...', end=' ')
import trl
print(f'OK (v{trl.__version__})')

print('  Importing GRPOTrainer...', end=' ')
try:
    from trl import GRPOTrainer
    print('OK')
except Exception:
    print('FAIL')
    tb = traceback.format_exc()
    lines = tb.strip().split('\n')
    for line in lines[-15:]:
        print(f'    {line}')
    sys.exit(1)

print()
print('✅ Tutti gli import OK')
"
echo ""

# ── 8. External imports (early warning per future missing deps) ──────────
echo "── 8. Moduli esterni referenziati nei file .py di trl ──"
if [ -n "$TRL_PATH" ] && [ -d "$TRL_PATH" ]; then
    echo "  Cercando import di moduli non-stdlib in trl..."
    echo ""
    $PY -c "
import re, os, sys
TRL_PATH = '$TRL_PATH'
stdlib = set(sys.stdlib_module_names)
core_deps = {'torch', 'transformers', 'accelerate', 'datasets', 'peft', 
             'numpy', 'huggingface_hub', 'safetensors', 'tqdm', 'yaml',
             'PIL', 'wandb', 'rich', 'regex', 'scipy', 'pandas', 'packaging'}
imports_found = set()
for root, dirs, files in os.walk(TRL_PATH):
    for f in files:
        if f.endswith('.py'):
            fpath = os.path.join(root, f)
            rel = os.path.relpath(fpath, TRL_PATH)
            try:
                with open(fpath, 'r') as fh:
                    content = fh.read()
            except: continue
            for match in re.finditer(r'(?:from\s+)(\w+)(?:\.|\s+import)|(?:^import\s+)([\w.]+)', content, re.MULTILINE):
                mod = match.group(1) or match.group(2)
                if mod and mod not in stdlib and mod != 'trl' and not mod.startswith('_'):
                    if mod not in core_deps:
                        imports_found.add((mod, rel))
if imports_found:
    for mod, fname in sorted(imports_found):
        print(f'    {mod:25s} ← {fname}')
else:
    print('  Nessun modulo esterno aggiuntivo trovato.')
"
else
    echo "  ⚠️  trl path non trovato"
fi
echo ""
echo "── 9. Cache pip ──"
PIP_CACHE=$($PY -m pip cache dir 2>/dev/null)
echo "  Cache dir: $PIP_CACHE  ($(du -sh "$PIP_CACHE" 2>/dev/null | cut -f1))"
echo ""

echo "============================================"
echo "  Diagnostica completata"
echo "============================================"
