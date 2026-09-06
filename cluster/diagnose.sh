#!/bin/bash
# ============================================================================
# Diagnostica ambiente pip sul cluster — individua conflitti e cache spurie.
#
# Uso (dal login node):
#   cd ~/neuro_symbolic_t2g
#   bash cluster/diagnose.sh
# ============================================================================

#SBATCH --job-name=diagnose-t2g
#SBATCH --account=thesis-course
#SBATCH --partition=thesis-course
#SBATCH --qos=gpu-xlarge
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1 --gres=shard:22528
#SBATCH --output=logs/slurm-diagnose-%j.log

# ── 0. Il login alloca compute; non creare srun annidati nei job esistenti. ──
if [ -z "${SLURM_JOB_ID:-}" ]; then
    echo "🚀 Login node rilevato → rilancio sul compute con srun..."
    ACCOUNT="${SLURM_ACCOUNT:-thesis-course}"
    exec srun --account "$ACCOUNT" --partition "$ACCOUNT" --qos gpu-xlarge \
         --gres=gpu:1 --gres=shard:22528 --mem=48G --cpus-per-task=8 \
         bash "$0" "$@"
fi

set -euo pipefail
_lib_dir=$(dirname "${BASH_SOURCE[0]}")
if [ ! -f "${_lib_dir}/_lib.sh" ]; then
    _lib_dir="${SLURM_SUBMIT_DIR:-$HOME/neuro_symbolic_t2g}/cluster"
fi
SCRIPT_DIR=$(cd "${_lib_dir}" && pwd)
# shellcheck source=cluster/_lib.sh
source "$SCRIPT_DIR/_lib.sh"
cd "$PROJ_DIR"
export_offline_env

echo "============================================"
echo "  Diagnostica ambiente pip — Neuro-Symbolic T2G"
echo "  $(date)"
echo "============================================"
echo ""

echo "Python: $(run_py --version 2>&1)"
echo "Pip:    $(run_py -m pip --version 2>&1)"
echo ""

# ── 1. Percorsi ──────────────────────────────────────────────────────────
echo "── 1. Percorsi di installazione pacchetti ──"
run_py -c "import site; print('site-packages:'); [print(f'  {p}') for p in site.getsitepackages()]; print('user site:', site.getusersitepackages())"
echo ""

# ── 2. TRL + mergekit info ───────────────────────────────────────────────
echo "── 2. Pacchetto TRL ──"
run_py -m pip show trl 2>&1 || echo "❌ trl NON installato"
echo ""
echo "── 3. Pacchetto mergekit ──"
run_py -m pip show mergekit 2>&1 || echo "❌ mergekit NON installato"
echo ""

# ── 4. Contenuto mergekit_utils.py ───────────────────────────────────────
TRL_PATH=$(run_py -c "import trl; print(trl.__path__[0])" 2>/dev/null)
echo "── 4. mergekit_utils.py (riga 20-25) ──"
run_py -c "
from pathlib import Path
p = Path('$TRL_PATH') / 'mergekit_utils.py'
print('\\n'.join(p.read_text().splitlines()[19:25]) if p.is_file() else '  ⚠️  mergekit_utils.py non trovato')
"
echo ""
echo "── 5. judges.py (riga 25-32) ──"
run_py -c "
from pathlib import Path
p = Path('$TRL_PATH') / 'trainer' / 'judges.py'
print('\\n'.join(p.read_text().splitlines()[24:32]) if p.is_file() else '  ⚠️  judges.py non trovato')
"
echo ""

# ── 6. La verità: cosa restituisce _is_package_available? ────────────────
echo "── 6. Cosa restituisce _is_package_available (dopo import trl) ──"
run_py -c "
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
run_py -c "
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
echo "  Cercando import di moduli non-stdlib in trl..."
echo ""
run_py -c "
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
echo ""
echo "── 9. Cache pip ──"
run_py -m pip cache info
echo ""

echo "============================================"
echo "  Diagnostica completata"
echo "============================================"
