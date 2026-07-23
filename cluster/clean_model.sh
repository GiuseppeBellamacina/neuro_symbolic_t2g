#!/bin/bash
# ============================================================================
# Pulizia selettiva — rimuove checkpoints, logs, results, figures di un
# modello specifico (per tag).
#
# Cerca in:
#   experiments/checkpoints/qwen25-05b-*/  (struttura flat, config v2+)
#   experiments/checkpoints/grpo/t2g/*/    (struttura vecchia, retrocompat)
#   experiments/logs/qwen25-05b-*/         (log training/eval)
#   experiments/results/*/                (eval JSON)
#   experiments/figures/                   (plot, ablation_summary)
#   logs/slurm-*-${TAG}*.log              (log SLURM)
#
# Uso:
#   bash cluster/clean_model.sh grpo-optimal           # dry-run
#   bash cluster/c_clean_model.sh grpo-optimal --all   # cancella tutto
#   bash cluster/clean_model.sh                        # lista tutti i tag
# ============================================================================

set -e
cd "$HOME/neuro_symbolic_t2g"

MODEL=""
FORCE=0

for arg in "$@"; do
    case "$arg" in
        --all) FORCE=1 ;;
        --help|-h)
            echo "Uso: bash cluster/clean_model.sh <TAG> [--all]"
            echo ""
            echo "TAG = il tag del config (es. grpo-optimal, grpo-pda, sft, ...)"
            echo "Senza argomenti: lista tutti i tag trovati"
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

# ── Nessun modello specificato: lista tutti i tag trovati ─────────────────
if [ -z "$MODEL" ]; then
    echo "=== Modelli trovati (dry-run) ==="
    echo ""

    # Checkpoints (struttura flat)
    for d in experiments/checkpoints/*/; do
        [ -d "$d" ] || continue
        name=$(basename "$d")
        SIZE=$(du -sh "$d" 2>/dev/null | cut -f1)
        echo "  $name ($SIZE)"
    done

    # Checkpoints (struttura vecchia grpo/t2g/)
    for d in experiments/checkpoints/grpo/t2g/*/ 2>/dev/null; do
        [ -d "$d" ] || continue
        name=$(basename "$d")
        SIZE=$(du -sh "$d" 2>/dev/null | cut -f1)
        echo "  grpo/t2g/$name ($SIZE)"
    done

    # Results
    if [ -d "experiments/results" ]; then
        for d in experiments/results/*/; do
            [ -d "$d" ] || continue
            name=$(basename "$d")
            SIZE=$(du -sh "$d" 2>/dev/null | cut -f1)
            echo "  results/$name ($SIZE)"
        done
    fi

    echo ""
    echo "Per cancellare: bash cluster/clean_model.sh <TAG> --all"
    exit 0
fi

# ── Dry-run per il modello specificato ────────────────────────────────────
if [ "$FORCE" = "0" ]; then
    echo "=== DRY RUN per '$MODEL' — aggiungi --all per cancellare ==="
    echo ""
    FOUND=0

    # Checkpoints (flat)
    for d in experiments/checkpoints/*${MODEL}*/; do
        [ -d "$d" ] || continue
        SIZE=$(du -sh "$d" 2>/dev/null | cut -f1)
        echo "  [CHECKPOINTS] $d ($SIZE)"
        FOUND=1
    done

    # Checkpoints (vecchia struttura)
    DIR="experiments/checkpoints/grpo/t2g/$MODEL"
    if [ -d "$DIR" ]; then
        SIZE=$(du -sh "$DIR" 2>/dev/null | cut -f1)
        echo "  [CHECKPOINTS] $DIR ($SIZE)"
        FOUND=1
    fi

    # Logs
    for d in experiments/logs/*${MODEL}*/; do
        [ -d "$d" ] || continue
        SIZE=$(du -sh "$d" 2>/dev/null | cut -f1)
        echo "  [LOGS] $d ($SIZE)"
        FOUND=1
    done

    # Results
    for d in experiments/results/*${MODEL}*/; do
        [ -d "$d" ] || continue
        SIZE=$(du -sh "$d" 2>/dev/null | cut -f1)
        echo "  [RESULTS] $d ($SIZE)"
        FOUND=1
    done

    # SLURM logs
    ls logs/slurm-*-${MODEL}*.log 2>/dev/null | while read f; do
        echo "  [SLURM] $f"
        FOUND=1
    done

    if [ $FOUND -eq 0 ]; then
        echo "  (niente trovato per '$MODEL')"
    fi
    echo ""
    echo "Per cancellare: bash cluster/clean_model.sh $MODEL --all"
    exit 0
fi

# ── Cancella ───────────────────────────────────────────────────────────────
echo "Pulizia modello: $MODEL"
CLEANED=0

# Checkpoints (flat — es. experiments/checkpoints/qwen25-05b-optimal/)
for d in experiments/checkpoints/*${MODEL}*/; do
    [ -d "$d" ] || continue
    echo "  [CHECKPOINTS] $d"
    rm -rf "$d"
    CLEANED=1
done

# Checkpoints (vecchia struttura grpo/t2g/)
DIR="experiments/checkpoints/grpo/t2g/$MODEL"
if [ -d "$DIR" ]; then
    echo "  [CHECKPOINTS] $DIR"
    rm -rf "$DIR"
    CLEANED=1
fi

# Logs (experiments/logs/*${MODEL}*/)
for d in experiments/logs/*${MODEL}*/; do
    [ -d "$d" ] || continue
    echo "  [LOGS] $d"
    rm -rf "$d"
    CLEANED=1
done

# Results (experiments/results/*${MODEL}*/)
for d in experiments/results/*${MODEL}*/; do
    [ -d "$d" ] || continue
    echo "  [RESULTS] $d"
    rm -rf "$d"
    CLEANED=1
done

# SLURM logs
for f in logs/slurm-*-${MODEL}*.log; do
    [ -f "$f" ] || continue
    echo "  [SLURM] $f"
    rm -f "$f"
    CLEANED=1
done

echo ""
if [ $CLEANED -eq 1 ]; then
    echo "✅ Pulizia completata per '$MODEL'."
else
    echo "ℹ️  Nessuna cartella da pulire per '$MODEL'."
fi
