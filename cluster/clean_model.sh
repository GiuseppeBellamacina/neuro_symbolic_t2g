#!/bin/bash
# Remove one experiment cell from the canonical
# model/method/prompt[/ablations/variant]/run artifact hierarchy.

set -euo pipefail
cd "$HOME/neuro_symbolic_t2g"

MODEL=""
FORCE=0
for arg in "$@"; do
    case "$arg" in
        --all) FORCE=1 ;;
        --help|-h)
            echo "Usage: bash cluster/clean_model.sh <ID> [--all]"
            echo "IDs: baseline-zero baseline-few sft grpo-zero grpo-few"
            echo "     sft-grpo-zero sft-grpo-few sft-grpo-zero-pda sft-grpo-zero-hot"
            exit 0
            ;;
        *)
            if [ -n "$MODEL" ]; then
                echo "Too many arguments: $arg" >&2
                exit 1
            fi
            MODEL="$arg"
            ;;
    esac
done

cell_path() {
    case "$MODEL" in
        baseline-zero) echo "qwen25-05b/baseline/zero-shot" ;;
        baseline-few) echo "qwen25-05b/baseline/few-shot" ;;
        sft) echo "qwen25-05b/sft/zero-shot" ;;
        grpo-zero) echo "qwen25-05b/grpo/zero-shot" ;;
        grpo-few) echo "qwen25-05b/grpo/few-shot" ;;
        sft-grpo-zero) echo "qwen25-05b/sft-grpo/zero-shot" ;;
        sft-grpo-few) echo "qwen25-05b/sft-grpo/few-shot" ;;
        sft-grpo-zero-pda) echo "qwen25-05b/sft-grpo/zero-shot/ablations/pda" ;;
        sft-grpo-zero-hot) echo "qwen25-05b/sft-grpo/zero-shot/ablations/hot" ;;
        *) return 1 ;;
    esac
}

slurm_logs_for_model() {
    local start
    start=$(date -d '14 days ago' +%Y-%m-%d 2>/dev/null || date +%Y-%m-%d)
    sacct --me --noheader --format=JobID,JobName --parsable2 \
        --starttime="$start" 2>/dev/null \
        | awk -F'|' -v m="$MODEL" '
            $2 == "train-" m || $2 == "eval-" m {
                if ($1 ~ /^[0-9]+$/) {
                    if ($2 ~ /^train-/) print "logs/slurm-train-" $1 ".log"
                    else print "logs/slurm-eval-" $1 ".log"
                }
            }' | sort -u
}

emit_targets() {
    local rel kind target
    rel=$(cell_path) || return 1
    for kind in checkpoints logs results figures; do
        target="experiments/$kind/$rel"
        [ -e "$target" ] && echo "$target"
    done
    slurm_logs_for_model
}

if [ -z "$MODEL" ]; then
    echo "=== Canonical checkpoint runs ==="
    find experiments/checkpoints -type d -name 'run_*' -print 2>/dev/null \
        | sort | while read -r d; do
            echo "  ${d#experiments/checkpoints/} ($(du -sh "$d" 2>/dev/null | cut -f1))"
        done
    echo ""
    echo "To remove a cell: bash cluster/clean_model.sh <ID> --all"
    exit 0
fi

if ! cell_path >/dev/null; then
    echo "Unknown canonical ID: $MODEL" >&2
    exit 1
fi

TARGETS=$(emit_targets || true)
if [ "$FORCE" -eq 0 ]; then
    echo "=== DRY RUN for '$MODEL' — add --all to remove ==="
    if [ -z "$TARGETS" ]; then
        echo "  (nothing found)"
    else
        while IFS= read -r target; do
            [ -n "$target" ] || continue
            size=$(du -sh "$target" 2>/dev/null | cut -f1 || echo "?")
            echo "  $target ($size)"
        done <<< "$TARGETS"
    fi
    exit 0
fi

CLEANED=0
while IFS= read -r target; do
    [ -n "$target" ] || continue
    echo "Removing $target"
    rm -rf "$target"
    CLEANED=1
done <<< "$TARGETS"

if [ "$CLEANED" -eq 1 ]; then
    echo "Cleanup complete for '$MODEL'."
else
    echo "Nothing found for '$MODEL'."
fi
