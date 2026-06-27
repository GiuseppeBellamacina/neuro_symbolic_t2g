#!/bin/bash
# ============================================================================
# Pulizia workspace sul cluster
#
# Uso:
#   bash cluster/clean.sh          # dry-run (mostra cosa cancellerebbe)
#   bash cluster/clean.sh --force  # cancella davvero
# ============================================================================

set -e
cd "$HOME/neuro_symbolic_t2g"
STATE_DIR="$HOME/neuro_symbolic_t2g/.chain_state"

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

# ── Svuota data/ (dataset scaricato, verrà riscaricato) ──────────────────
echo "[1/7] data/ (dataset ASLG-PC12)"
if [ -d "data" ]; then
    $CMD data/*
fi

# ── Checkpoints ───────────────────────────────────────────────────
echo "[2/7] experiments/checkpoints/"
if [ -d "experiments/checkpoints" ]; then
    $CMD experiments/checkpoints/*
fi

# ── Logs ───────────────────────────────────────────────────────────
echo "[3/7] logs/ (SLURM output) + experiments/logs/"
if [ -d "logs" ]; then
    $CMD logs/*
fi
if [ -d "experiments/logs" ]; then
    $CMD experiments/logs/*
fi

# ── Cache Python ─────────────────────────────────────────────────────────
echo "[4/7] __pycache__/"
find . -type d -name "__pycache__" -print -exec $CMD {} + 2>/dev/null || true

# ── Artifact LoRA del GRPOTrainer ────────────────────────────────────────
echo "[5/7] grpo_trainer_lora_model_*/"
for d in grpo_trainer_lora_model_*; do
    [ -d "$d" ] && $CMD "$d"
done

# ── Unsloth compiled cache ───────────────────────────────────────────────
echo "[6/7] unsloth_compiled_cache/"
if [ -d "unsloth_compiled_cache" ]; then
    $CMD unsloth_compiled_cache
fi

# ── Stato pipeline ─────────────────────────────────────────────────────
echo "[7/7] .chain_state/ (stato pipeline)"
[ -d ".chain_state" ] && $CMD .chain_state

echo ""
if [ "$FORCE" = "0" ]; then
    echo "=== Nessun file cancellato (dry-run). Usa: bash cluster/clean.sh --force ==="
else
    echo "Pulizia completata."
fi
