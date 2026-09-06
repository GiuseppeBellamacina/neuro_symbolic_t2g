#!/bin/bash
# Online bootstrap. All Python and pip execution stays in Apptainer.
# Optional overrides: T2G_SIF, SLURM_ACCOUNT, T2G_SLURM_PARTITION,
# T2G_SLURM_QOS, T2G_SETUP_GPU_GRES,
# T2G_SETUP_SHARD_GRES, T2G_SETUP_MEM, and T2G_SETUP_CPUS.

set -euo pipefail

SIF="${T2G_SIF:-/shared/sifs/latest.sif}"
if [ "${T2G_SETUP_CONTAINER:-0}" != "1" ] && [ -z "${APPTAINER_CONTAINER:-}" ]; then
    [ -f "$SIF" ] || { echo "ERROR: container not found: $SIF" >&2; exit 1; }
    command -v srun >/dev/null 2>&1 || {
        echo "ERROR: srun is unavailable; setup must launch Apptainer on a compute node." >&2
        exit 1
    }
    HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
    HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
    ACCOUNT="${SLURM_ACCOUNT:-thesis-course}"
    PARTITION="${T2G_SLURM_PARTITION:-${SLURM_PARTITION:-$ACCOUNT}}"
    QOS="${T2G_SLURM_QOS:-${SLURM_QOS:-gpu-xlarge}}"
    GPU_GRES="${T2G_SETUP_GPU_GRES:-gpu:1}"
    SHARD_GRES="${T2G_SETUP_SHARD_GRES:-shard:22000}"
    MEM="${T2G_SETUP_MEM:-48G}"
    CPUS="${T2G_SETUP_CPUS:-8}"
    echo "==> Requesting the setup compute allocation with srun"
    exec srun --account "$ACCOUNT" --partition "$PARTITION" --qos "$QOS" \
        --gres="$GPU_GRES" --gres="$SHARD_GRES" --mem="$MEM" --cpus-per-task="$CPUS" \
        apptainer run --nv \
        --cleanenv \
        --home "$HOME:$HOME" \
        --bind "$HOME:$HOME" \
        --env "HF_HOME=$HF_HOME" \
        --env "HF_HUB_CACHE=$HF_HUB_CACHE" \
        --env "T2G_SETUP_CONTAINER=1" \
        "$SIF" bash "$0" "$@"
fi

[ "${T2G_SETUP_CONTAINER:-0}" = "1" ] || [ -n "${APPTAINER_CONTAINER:-}" ] || {
    echo "ERROR: setup relaunch did not enter Apptainer; refusing bare host execution." >&2
    exit 1
}

# BEGIN T2G_LIB_RESOLVER
_lib_source=${BASH_SOURCE[0]}
_lib_dir=""
_lib_candidate=$(cd "$(dirname "$_lib_source")" 2>/dev/null && pwd) || _lib_candidate=""
if [ -n "$_lib_candidate" ] && [ -f "$_lib_candidate/_lib.sh" ]; then
    _lib_dir=$_lib_candidate
elif [ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -f "${SLURM_SUBMIT_DIR}/cluster/_lib.sh" ]; then
    _lib_dir=$(cd "${SLURM_SUBMIT_DIR}/cluster" && pwd)
elif [ -n "${HOME:-}" ] && [ -f "${HOME}/neuro_symbolic_t2g/cluster/_lib.sh" ]; then
    _lib_dir=$(cd "${HOME}/neuro_symbolic_t2g/cluster" && pwd)
else
    printf 'ERROR: cannot locate cluster/_lib.sh (BASH_SOURCE=%s, SLURM_SUBMIT_DIR=%s)\n' \
        "$_lib_source" "${SLURM_SUBMIT_DIR:-<unset>}" >&2
    exit 1
fi
SCRIPT_DIR=$_lib_dir
# shellcheck source=cluster/_lib.sh
source "$SCRIPT_DIR/_lib.sh"
# END T2G_LIB_RESOLVER
cd "$PROJ_DIR"

export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export PYTHONUNBUFFERED=1
mkdir -p "$HF_HOME" "$HF_HUB_CACHE"

PYTHON_BIN=$(command -v python3 || command -v python || true)
[ -n "$PYTHON_BIN" ] || { echo "ERROR: Python missing inside $SIF" >&2; exit 1; }

CONSTRAINTS=$(mktemp "${TMPDIR:-/tmp}/t2g-constraints.XXXXXX")
STACK_INFO=$(mktemp "${TMPDIR:-/tmp}/t2g-stack.XXXXXX")
trap 'rm -f "$CONSTRAINTS" "$STACK_INFO"' EXIT

echo "==> Capturing the critical container stack"
"$PYTHON_BIN" - "$CONSTRAINTS" "$STACK_INFO" <<'PY'
import importlib.metadata as metadata
import json
import re
import sys

constraints, stack_info = sys.argv[1:]
critical = {
    "torch", "torchao", "triton", "xformers", "bitsandbytes", "transformers",
    "accelerate", "trl", "peft", "unsloth", "unsloth-zoo",
    "sentence-transformers", "sacrebleu",
}

def normalize_name(name):
    return re.sub(r"[-_.]+", "-", name).lower()

names = {
    normalize_name(dist.metadata["Name"])
    for dist in metadata.distributions()
    if dist.metadata.get("Name")
}
protected = sorted(
    name for name in names if name in critical or name.startswith("nvidia-")
)
before = {name: metadata.version(name) for name in protected}
with open(constraints, "w", encoding="utf-8") as handle:
    for name, version in before.items():
        handle.write(f"{name}=={version}\n")

import torch

with open(stack_info, "w", encoding="utf-8") as handle:
    json.dump(
        {
            "packages": before,
            "torch_identity": {"torch": torch.__version__, "cuda": torch.version.cuda},
        },
        handle,
        sort_keys=True,
    )
print(f"Protected {len(before)} installed distributions")
print(f"Expected torch={torch.__version__}, CUDA={torch.version.cuda}")
PY

echo "==> Installing project core + retrieval into the shared user site"
"$PYTHON_BIN" -m pip install --user --constraint "$CONSTRAINTS" ".[retrieval]"

echo "==> Verifying the complete critical stack"
"$PYTHON_BIN" - "$STACK_INFO" <<'PY'
import importlib.metadata as metadata
import json
import re
import sys

from packaging.version import Version
import torch

def normalize_name(name):
    return re.sub(r"[-_.]+", "-", name).lower()

def validate_versions(before, after, torch_before, torch_after):
    changed = {
        name: (version, after.get(name))
        for name, version in before.items()
        if after.get(name) != version
    }
    if changed:
        raise RuntimeError(f"pip changed protected packages: {changed}")
    if torch_after != torch_before:
        raise RuntimeError(
            f"torch/CUDA identity changed: expected {torch_before}, got {torch_after}"
        )
    exact = {
        "transformers": "5.3.0",
        "trl": "0.24.0",
        "peft": "0.19.1",
        "unsloth": "2026.7.1",
        "unsloth-zoo": "2026.7.1",
        "sacrebleu": "2.6.0",
        "sentence-transformers": "5.2.3",
    }
    wrong = {
        name: (wanted, after.get(name))
        for name, wanted in exact.items()
        if after.get(name) != wanted
    }
    torchao = after.get("torchao")
    if torchao is None or not (Version("0.16") <= Version(torchao) < Version("0.18")):
        wrong["torchao"] = (">=0.16,<0.18", torchao)
    if wrong:
        raise RuntimeError(f"tested package versions not active: {wrong}")

expected = json.load(open(sys.argv[1], encoding="utf-8"))
installed = {
    normalize_name(dist.metadata["Name"]): dist.version
    for dist in metadata.distributions()
    if dist.metadata.get("Name")
}
torch_identity = {"torch": torch.__version__, "cuda": torch.version.cuda}
validate_versions(
    expected["packages"], installed, expected["torch_identity"], torch_identity
)
print(f"torch={torch.__version__}, CUDA={torch.version.cuda} (preserved)")
print("tested critical package versions are active")
PY

MODEL_ID="${MODEL_ID:-Qwen/Qwen2.5-0.5B-Instruct}"
BUILD_BIGRAM="${BUILD_BIGRAM:-0}"
echo "==> Downloading model/tokenizer and preparing ASLG-PC12 artifacts"
"$PYTHON_BIN" -m src.utils.setup_artifacts \
    --model-id "$MODEL_ID" \
    --dataset-cache "data/aslg_pc12" \
    --vocab-path "data/gloss_vocab.txt" \
    --bigram-path "data/bigram_transition.npy" \
    --build-bigram "$BUILD_BIGRAM"

echo "==> Lightweight online verification"
"$PYTHON_BIN" - "$MODEL_ID" <<'PY'
import sys

from datasets import load_dataset
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer

model_id = sys.argv[1]
snapshot = snapshot_download(model_id, local_files_only=True)
AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
dataset = load_dataset("achrafothman/aslg_pc12", cache_dir="data/aslg_pc12")
assert "train" in dataset and len(dataset["train"]) > 0
print("model/tokenizer and dataset cache are readable")
PY

echo ""
echo "Online setup complete inside Apptainer."
echo "Next, submit the offline compute verification:"
echo "  CONFIG=${CONFIG:-experiments/configs/qwen25-05b/sft-grpo/zero-shot.yaml} sbatch cluster/preflight.sh"
