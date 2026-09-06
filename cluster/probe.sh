#!/bin/bash
# Offline frozen-generation probes. This launcher never trains or downloads.
#
# Login node:
#   INPUT=experiments/results/.../generations_*.json \
#     bash cluster/probe.sh {rollouts|rewards|markov} [config.yaml]
# Batch submission:
#   INPUT=experiments/results/.../generations_*.json \
#     sbatch cluster/probe.sh {rollouts|rewards|markov} [config.yaml]
#
#SBATCH --job-name=probe-t2g
#SBATCH --account=thesis-course
#SBATCH --partition=thesis-course
#SBATCH --qos=gpu-xlarge
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1 --gres=shard:4096
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=bellamacina50@gmail.com
#SBATCH --output=logs/slurm-probe-%j.log

COMMAND="${1:-rollouts}"
case "$COMMAND" in
    rollouts|rewards|markov) ;;
    *) echo "usage: INPUT=path/to/generations.json $0 {rollouts|rewards|markov} [config.yaml]" >&2; exit 2 ;;
esac
CONFIG="${2:-experiments/configs/qwen25-05b/probes/${COMMAND}.yaml}"

# The login node has no project Python. Relaunch first; do not source project
# helpers, inspect artifacts, or invoke Python before entering the allocation.
if [ -z "${SLURM_JOB_ID:-}" ] && [ -z "${APPTAINER_CONTAINER:-}" ]; then
    echo "Login node detected: relaunching probe with srun + Apptainer..."
    ACCOUNT="${SLURM_ACCOUNT:-thesis-course}"
    exec srun --account "$ACCOUNT" --partition "$ACCOUNT" --qos gpu-xlarge \
        --gres=gpu:1 --gres=shard:4096 --mem=16G --cpus-per-task=4 \
        apptainer run --nv /shared/sifs/latest.sif bash "$0" "$@"
fi

set -euo pipefail

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

# srun's explicit Apptainer relaunch is already inside the image. An sbatch
# invocation remains on the compute host and lets run_py enter the image once.
if [ -n "${APPTAINER_CONTAINER:-}" ]; then
    export RUN_PY_FORCE_BARE=1
fi

# Hub clients consume these flags at import time, so this precedes run_py.
export_offline_env

[ -f "$CONFIG" ] || { echo "missing probe config: $CONFIG" >&2; exit 1; }
[ -n "${INPUT:-}" ] || {
    echo "INPUT is required and must name an existing eval generations_*.json file" >&2
    echo "usage: INPUT=experiments/results/.../generations_name.json $0 $COMMAND [$CONFIG]" >&2
    exit 2
}
[ -f "$INPUT" ] || { echo "missing generations input: $INPUT" >&2; exit 1; }

# These are the canonical probe artifacts. src.analysis performs the final
# config-aware validation; for Markov it also validates matrix shape/content.
VOCAB="data/gloss_vocab.txt"
BIGRAM="data/bigram_transition.npy"
[ -s "$VOCAB" ] || { echo "missing probe vocabulary: $VOCAB" >&2; exit 1; }
if [ "$COMMAND" = "markov" ]; then
    [ -s "$BIGRAM" ] || { echo "missing Markov bigram matrix: $BIGRAM" >&2; exit 1; }
fi

ARGS=("$COMMAND" --config "$CONFIG" --input "$INPUT")
if [ -n "${OUTPUT:-}" ]; then
    ARGS+=(--output "$OUTPUT")
fi
if [ "${FORCE:-0}" = "1" ]; then
    ARGS+=(--force)
fi
run_py -m src.analysis "${ARGS[@]}"
