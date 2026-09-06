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

# Login nodes have no project Python: allocation/container relaunch comes first.
if [ -z "${SLURM_JOB_ID:-}" ] && [ -z "${APPTAINER_CONTAINER:-}" ]; then
    ACCOUNT="${SLURM_ACCOUNT:-thesis-course}"
    exec srun --account "$ACCOUNT" --partition "$ACCOUNT" --qos gpu-xlarge \
        --gres=gpu:1 --gres=shard:22528 --mem=48G --cpus-per-task=4 \
        apptainer run --nv /shared/sifs/latest.sif bash "$0" "$@"
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
[ -n "${APPTAINER_CONTAINER:-}" ] && export RUN_PY_FORCE_BARE=1

# Offline flags precede every Python invocation. Setup must have cached model/data.
export_offline_env
[ -f "$CONFIG" ] || { echo "missing structured config: $CONFIG" >&2; exit 1; }
FEATURES="${FEATURES:-experiments/analysis/qwen25-05b/structured/features.npz}"
if [ ! -s "$FEATURES" ]; then
    run_py -m src.analysis.structured_benchmark extract --config "$CONFIG" --features "$FEATURES"
fi
run_py -m src.analysis.structured_benchmark run --config "$CONFIG" --features "$FEATURES"
