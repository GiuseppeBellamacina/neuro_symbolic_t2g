#!/bin/bash
# Offline GPU research benchmark. Extracts frozen Qwen features, then trains
# four small emission heads; it never updates or downloads the backbone.
# Usage: sbatch cluster/structured_probe.sh [config.yaml]
#SBATCH --job-name=structured-t2g
#SBATCH --account=thesis-course
#SBATCH --partition=thesis-course
#SBATCH --qos=gpu-xlarge
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1 --gres=shard:22528
#SBATCH --output=logs/slurm-structured-%j.log

CONFIG="${1:-experiments/configs/qwen25-05b/probes/structured.yaml}"

# Login nodes have no project Python: allocate compute first. run_py enters the
# container only after the offline environment has been exported.
if [ -z "${SLURM_JOB_ID:-}" ]; then
    ACCOUNT="${SLURM_ACCOUNT:-thesis-course}"
    exec srun --account "$ACCOUNT" --partition "$ACCOUNT" --qos gpu-xlarge \
        --gres=gpu:1 --gres=shard:22528 --mem=48G --cpus-per-task=4 \
        bash "$0" "$@"
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

# Offline flags precede every Python invocation. Setup must have cached model/data.
export_offline_env
[ -f "$CONFIG" ] || { echo "missing structured config: $CONFIG" >&2; exit 1; }
require_cluster_artifacts "$CONFIG"
FEATURES="${FEATURES:-experiments/analysis/qwen25-05b/structured/features.npz}"
if [ ! -s "$FEATURES" ]; then
    run_py -m src.analysis.structured_benchmark extract --config "$CONFIG" --features "$FEATURES"
fi
run_py -m src.analysis.structured_benchmark run --config "$CONFIG" --features "$FEATURES"
