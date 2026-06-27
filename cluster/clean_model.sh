#!/bin/bash
# ============================================================================
# Pulizia selettiva — rimuove checkpoints e logs di un modello specifico.
#
# Uso:
#   bash cluster/clean_model.sh qwen05    # dry-run
#   bash cluster/clean_model.sh qwen05 --all  # cancella tutto
# ============================================================================

set -e
cd "$HOME/neuro_symbolic_t2g"

VALID_MODELS=("qwen05")
MODEL=""
FORCE=0

for arg in "$@"; do
    case "$arg" in
        --all) FORCE=1 ;;
        --help|-h)
            echo "Uso: bash cluster/clean_model.sh <MODEL_TAG> [--all]"
            echo ""
            echo "Modelli disponibili:"
            for m in "${VALID_MODELS[@]}"; do echo "  $m"; done
            exit 0
            ;;
        *)
            if [ -z "$MODEL" ]; then
                MODEL="$arg"
            else
                echo "❌ Troppi argomenti: $arg"
                exit 1
            fi
            ;;
    esac
done

if [ -z "$MODEL" ]; then
    echo "=== DRY RUN — mostra cosa esiste ==="
    echo ""
    for m in "${VALID_MODELS[@]}"; do
        DIRS_FOUND=()
        [ -d "experiments/checkpoints/grpo/t2g/$m" ] && DIRS_FOUND+=("experiments/checkpoints/grpo/t2g/$m")
        [ -d "logs" ] && ls logs/slurm-*-${m}*.log 2>/dev/null | while read f; do echo "  $f ($(du -sh "$f" | cut -f1))"; done

        if [ ${#DIRS_FOUND[@]} -gt 0 ]; then
            echo "  $m:"
            for d in "${DIRS_FOUND[@]}"; do
                SIZE=$(du -sh "$d" 2>/dev/null | cut -f1)
                echo "    $d ($SIZE)"
            done
        else
            echo "  $m: (niente)"
        fi
    done
    echo ""
    echo "Per cancellare: bash cluster/clean_model.sh <MODEL_TAG> --all"
    exit 0
fi

FOUND=0
for m in "${VALID_MODELS[@]}"; do
    [ "$m" = "$MODEL" ] && FOUND=1
done
if [ $FOUND -eq 0 ]; then
    echo "⚠️  '$MODEL' non è un modello noto. Procedo comunque..."
fi

if [ "$FORCE" -eq 0 ]; then
    echo "=== DRY RUN per $MODEL — aggiungi --all per cancellare ==="
    echo ""
    [ -d "experiments/checkpoints/grpo/t2g/$MODEL" ] && echo "  experiments/checkpoints/grpo/t2g/$MODEL ($(du -sh experiments/checkpoints/grpo/t2g/$MODEL 2>/dev/null | cut -f1))"
    echo "  logs/slurm-*-${MODEL}*:"
    ls logs/slurm-*-${MODEL}*.log 2>/dev/null | while read f; do echo "    $f"; done
    exit 0
fi

echo "Pulizia modello: $MODEL"
CLEANED=0

# Checkpoints
DIR="experiments/checkpoints/grpo/t2g/$MODEL"
if [ -d "$DIR" ]; then
    echo "[CHECKPOINTS] $DIR"
    rm -rf "$DIR"
    CLEANED=1
fi

# Logs
for f in logs/slurm-*-${MODEL}*.log; do
    [ -f "$f" ] && echo "[LOG] $f" && rm -f "$f" && CLEANED=1
done

echo ""
if [ $CLEANED -eq 1 ]; then
    echo "✅ Pulizia completata per '$MODEL'."
else
    echo "ℹ️  Nessuna cartella da pulire per '$MODEL'."
fi
