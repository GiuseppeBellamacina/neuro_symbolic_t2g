#!/bin/bash
# ============================================================================
# SLURM/preflight — Validazione pre-volo OFFLINE per i compute node T2G.
#
# COSA FA (solo verifiche, MAI training):
#   1. env offline completo (HF_*/W&B/PYTHONUNBUFFERED) prima di ogni python;
#   2. artifact offline presenti (dataset cache, vocab+sidecar, snapshot HF
#      del modello della config) — fail-fast senza alcun download;
#   3. import delle dipendenze e versioni (locale, zero rete);
#   4. load_dataset OFFLINE dalla cache HF condivisa (nessun download);
#   5. sorgente modello risolta offline (snapshot in cache / path locale);
#   6. W&B in modalità offline senza rete;
#   7. opzionale: gate PDA full-vocabulary quando PDA=1
#      (T2G_PDA_FULL_VOCAB=1 pytest tests/test_pda_grammar.py -k full_vocabulary).
#
# Uso (stesse convenzioni di setup.sh/train.sh):
#   sbatch cluster/preflight.sh                          # dal login node
#   CONFIG=experiments/configs/qwen25-05b/sft-grpo/zero-shot.yaml sbatch cluster/preflight.sh
#   srun ... cluster/preflight.sh                        # srun diretto
#   PDA=1 CONFIG=experiments/configs/qwen25-05b/ablations/sft-grpo-zero-pda.yaml sbatch cluster/preflight.sh
#
# NOTA: i compute node NON hanno internet: questo script non installa e non
# scarica nulla; un fallimento indica che l'acquisizione (dipendenze/modelli/
# dataset) va rifatta nell'ambiente separato con rete e sincronizzata qui.
# ============================================================================

# ┌────────────────────────────────────────────────────────┐
# │  CONFIGURA QUI — modifica account/partition/qos/email  │
# └────────────────────────────────────────────────────────┘
#SBATCH --job-name=preflight-t2g
#SBATCH --account=thesis-course
#SBATCH --partition=thesis-course
#SBATCH --qos=gpu-xlarge
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1 --gres=shard:22528
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=bellamacina50@gmail.com
#SBATCH --output=logs/slurm-preflight-%j.log

CONFIG="${CONFIG:-}"
REWARD_QUALIFICATION_REPORT="${REWARD_QUALIFICATION_REPORT:-}"

# ── 0. Login → compute. Il container è gestito solo da run_py dopo gli export.
if [ -z "${SLURM_JOB_ID:-}" ]; then
    echo "🚀 Login node rilevato → rilancio sul compute con srun..."
    ACCOUNT="${SLURM_ACCOUNT:-thesis-course}"
    exec srun --account "$ACCOUNT" --partition "$ACCOUNT" --qos gpu-xlarge \
         --gres=gpu:1 --gres=shard:22528 --mem=48G --cpus-per-task=8 \
         bash "$0" "$@"
fi

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=cluster/_lib.sh
source "$SCRIPT_DIR/_lib.sh"
cd "$PROJ_DIR"

# ── 1. Env offline PRIMA di ogni python/apptainer ────────────────────────────
export_offline_env

echo "============================================"
echo "  T2G Preflight — validazione offline"
echo "  Job ID:    ${SLURM_JOB_ID:-none}"
echo "  Node:      $(hostname)"
echo "  Date:      $(date)"
echo "  Config:    ${CONFIG:-<nessuna — check cache generici>}"
echo "  PDA gate:  $([ "${PDA:-0}" = "1" ] && echo yes || echo no)"
echo "============================================"
echo ""

# ── Check 1: variabili offline effettivamente esportate ──────────────────────
check_env_var() {
    local var="$1" want="$2" got
    got=$(eval "printf '%s' \"\${${var}:-}\"")
    if [ "$got" != "$want" ]; then
        echo "❌ ${var}='${got}' atteso '${want}' — env offline incompleto"
        exit 1
    fi
    echo "   ✅ ${var}=${got}"
}
echo "── 1. Env offline ──"
check_env_var HF_HUB_OFFLINE "1"
check_env_var TRANSFORMERS_OFFLINE "1"
check_env_var HF_DATASETS_OFFLINE "1"
check_env_var WANDB_MODE "offline"
check_env_var WANDB_DISABLE_WEAVE "true"
check_env_var WANDB_SILENT "true"
check_env_var PYTHONUNBUFFERED "1"
echo "   ✅ HF_HOME=${HF_HOME}"
echo ""

# ── Check 2: artifact offline (dataset/vocab/modello; bigram opzionale) ──────
echo "── 2. Artifact offline (require_cluster_artifacts) ──"
require_cluster_artifacts "$CONFIG"
if [[ "$CONFIG" == *"/ablations/rewards/"* ]]; then
    [ -n "$REWARD_QUALIFICATION_REPORT" ] || {
        echo "❌ REWARD_QUALIFICATION_REPORT required for reward ablation preflight" >&2
        exit 2
    }
    run_py -c "
import sys
from src.training.grpo_t2g_train import validate_reward_qualification
from src.utils.config import resolve_config
validate_reward_qualification(resolve_config(sys.argv[1]), sys.argv[2])
print('  ✅ scientific reward qualification report valid')
" "$CONFIG" "$REWARD_QUALIFICATION_REPORT"
fi
echo ""

# ── Check 3: import dipendenze + versioni (locale, zero rete) ────────────────
echo "── 3. Import dipendenze ──"
run_py -c "
import os
assert os.environ.get('HF_HUB_OFFLINE') == '1', 'HF_HUB_OFFLINE non è 1'
import torch, transformers, trl, peft, datasets, sklearn, sacrebleu
assert sacrebleu.__version__ == '2.6.0', f'sacrebleu must be 2.6.0, got {sacrebleu.__version__}'
print(f'  PyTorch:       {torch.__version__} (CUDA: {torch.cuda.is_available()})')
print(f'  Transformers:  {transformers.__version__}')
print(f'  TRL:           {trl.__version__}')
print(f'  PEFT:          {peft.__version__}')
print(f'  Datasets:      {datasets.__version__}')
print(f'  scikit-learn:  {sklearn.__version__}')
print(f'  SacreBLEU:     {sacrebleu.__version__}')
try:
    import sentence_transformers
    print(f'  sentence-transformers: {sentence_transformers.__version__}')
except Exception:
    print('  sentence-transformers: NON installabile (solo backend tfidf)')
"
echo ""

# ── Check 4: dataset load OFFLINE dalla cache HF condivisa ───────────────────
# Prova di caricamento RAW dalla cache (niente dedup/split, niente rete):
# con HF_DATASETS_OFFLINE=1 qualsiasi tentativo di rete fallirebbe — se
# questo check passa, la cache è realmente autosufficiente. Con CONFIG usa
# dataset_cache/dataset_name della config risolta; altrimenti i default.
echo "── 4. Load dataset offline dalla cache HF ──"
run_py -c "
import sys
from datasets import load_dataset

cache, name = 'data/aslg_pc12', 'achrafothman/aslg_pc12'
config = sys.argv[1] if len(sys.argv) > 1 and sys.argv[1] else ''
if config:
    from src.utils.config import resolve_config
    dscfg = resolve_config(config).get('dataset') or {}
    cache = str(dscfg.get('dataset_cache') or cache)
    name = str(dscfg.get('dataset_name') or name)
ds = load_dataset(name, cache_dir=cache)
counts = {k: len(v) for k, v in ds.items()}
print(f'  ✅ dataset offline OK: {name} from {cache} → {counts}')
" "$CONFIG"
echo ""

# ── Check 5: sorgente modello risolta OFFLINE (resolve_model_source) ─────────
# Stessa primitiva usata dai loader (snapshot_download con local_files_only →
# mai rete). Con CONFIG verifica il modello della config; senza, salta (il
# check generico della cache HF è già coperto da require_cluster_artifacts).
if [ -n "$CONFIG" ]; then
    echo "── 5. Sorgente modello (resolve_model_source, offline) ──"
    run_py -c "
import sys
from src.utils.config import resolve_config
from src.models.model_loader import resolve_model_source

mid = (resolve_config(sys.argv[1]).get('model') or {}).get('name') or ''
assert mid, 'Config senza model.name'
print(f'  ✅ {mid} → {resolve_model_source(mid)}')
" "$CONFIG"
    echo ""
fi

# ── Check 6: W&B in modalità offline senza rete ──────────────────────────────
echo "── 6. W&B offline ──"
run_py -c "
import os
import wandb

assert os.environ.get('WANDB_MODE') == 'offline', 'WANDB_MODE non è offline'
assert os.environ.get('WANDB_DISABLE_WEAVE') == 'true', 'WANDB_DISABLE_WEAVE non è true'
print(f'  ✅ wandb {wandb.__version__} importato con WANDB_MODE=offline (nessuna rete)')
"
echo ""

# ── Check 7 (opzionale): gate PDA full-vocabulary quando PDA=1 ───────────────
# Focused gate: T2G_PDA_FULL_VOCAB=1 pytest -k full_vocabulary (mai training).
# Richiede pytest nell'immagine (extra dev): se manca → fail loud, perché il
# gate è stato esplicitamente richiesto con PDA=1.
if [ "${PDA:-0}" = "1" ]; then
    echo "── 7. Gate PDA full-vocabulary (T2G_PDA_FULL_VOCAB=1) ──"
    if ! run_py -c "import pytest" >/dev/null 2>&1; then
        echo "❌ pytest non disponibile nel container (extra 'dev' escluso dal SIF)." >&2
        echo "   Il gate PDA=1 è stato richiesto esplicitamente: ricostruisci" >&2
        echo "   il SIF nell'ambiente con rete includendo pytest, oppure esegui" >&2
        echo "   il gate fuori dai compute node." >&2
        exit 1
    fi
    T2G_PDA_FULL_VOCAB=1 run_py -m pytest tests/test_pda_grammar.py -k full_vocabulary -q -s
    echo ""
fi

echo "============================================"
echo "  ✅ Preflight OK — artifact, env e W&B pronti (offline)."
echo "  Nessun training eseguito."
echo "  $(date)"
echo "============================================"
