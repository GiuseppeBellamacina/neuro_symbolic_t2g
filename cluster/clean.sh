#!/bin/bash
# ============================================================================
# Pulizia workspace sul cluster — rimuove tutti gli artifact generati.
#
# Uso:
#   bash cluster/clean.sh          # dry-run (mostra cosa cancellerebbe)
#   bash cluster/clean.sh --force  # cancella davvero
# ============================================================================

set -e
cd "$HOME/neuro_symbolic_t2g"

FORCE=0
if [ "$1" = "--force" ]; then
    FORCE=1
fi

if [ "$FORCE" = "0" ]; then
    echo "=== DRY RUN — aggiungi --force per cancellare davvero ==="
    echo ""
    CMD="echo [DRY] rm -rf"
else
    CMD="rm -rf"
fi

echo "Pulizia workspace: $PWD"
echo ""

# ── 1. data/ (dataset scaricato, verrà riscaricato) ──────────────────────
echo "[1/10] data/ (dataset ASLG-PC12)"
if [ -d "data" ]; then
    $CMD data/*
fi

# ── 2. Checkpoints ────────────────────────────────────────────────────────
echo "[2/10] experiments/checkpoints/"
if [ -d "experiments/checkpoints" ]; then
    $CMD experiments/checkpoints/*
fi

# ── 3. Logs SLURM + experiments/logs/ ──────────────────────────────────────
echo "[3/10] logs/ (SLURM) + experiments/logs/ (training+eval)"
if [ -d "logs" ]; then
    $CMD logs/*
fi
if [ -d "experiments/logs" ]; then
    $CMD experiments/logs/*
fi

# ── 4. Results (eval JSON: eval_*.json, comparison.json, generations) ───
echo "[4/10] experiments/results/ (eval JSON + comparison)"
if [ -d "experiments/results" ]; then
    $CMD experiments/results/*
fi

# ── 5. Figures (plot, chart, ablation summary) ───────────────────────────
echo "[5/10] experiments/figures/ (plot + ablation_summary)"
if [ -d "experiments/figures" ]; then
    $CMD experiments/figures/*
fi

# ── 6. Cache Python __pycache__ ───────────────────────────────────────────
echo "[6/10] __pycache__/ (Python bytecode)"
find . -type d -name "__pycache__" -print -exec $CMD {} + 2>/dev/null || true

# ── 7. Artifact LoRA del GRPOTrainer ──────────────────────────────────────
echo "[7/10] grpo_trainer_lora_model_*/"
for d in grpo_trainer_lora_model_*; do
    [ -d "$d" ] && $CMD "$d"
done

# ── 8. Unsloth compiled cache ─────────────────────────────────────────────
echo "[8/10] unsloth_compiled_cache/"
if [ -d "unsloth_compiled_cache" ]; then
    $CMD unsloth_compiled_cache
fi

# ── 9. grammarllm temp (parsing tables, debug logs) ──────────────────────
echo "[9/10] grammarllm/temp/ (parsing tables + debug logs)"
if [ -d "grammarllm/temp" ]; then
    $CMD grammarllm/temp
fi

# ── 10. Stato pipeline + wandb local + egg-info ──────────────────────────
echo "[10/10] .chain_state/ + wandb/ + *.egg-info/"
[ -d ".chain_state" ] && $CMD .chain_state
[ -d "wandb" ] && $CMD wandb
for d in *.egg-info; do
    [ -d "$d" ] && $CMD "$d"
done

echo ""
if [ "$FORCE" = "0" ]; then
    echo "=== Nessun file cancellato (dry-run). Usa: bash cluster/clean.sh --force ==="
else
    echo "✅ Pulizia completata (10 step)."
fi
