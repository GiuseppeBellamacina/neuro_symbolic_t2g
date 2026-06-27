#!/bin/bash
# ============================================================================
# Pulizia workspace sul cluster
#
# Uso:
#   bash src/cluster/clean.sh          # dry-run (mostra cosa cancellerebbe)
#   bash src/cluster/clean.sh --force  # cancella davvero
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

# ── Svuota data/ (dataset scaricato, verrà riscaricato) ──────────────────
echo "[1/7] data/ (dataset ASLG-PC12)"
if [ -d "data" ]; then
    $CMD data/*
fi

# ── Checkpoints ──────────────────────────────────────────────────────────
echo "[2/7] checkpoints/"
if [ -d "checkpoints" ]; then
    $CMD checkpoints/*
fi

# ── Logs ─────────────────────────────────────────────────────────────────
echo "[3/7] logs/ (SLURM output)"
if [ -d "logs" ]; then
    $CMD logs/*
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

# ── File watcher / pipeline ──────────────────────────────────────────────
echo "[7/7] .job_chain, .chain_pid, .chain_failed, .chain_stopped, .monitor_cache"
[ -f ".job_chain" ] && $CMD .job_chain
[ -f ".chain_pid" ] && $CMD .chain_pid
[ -f ".chain_failed" ] && $CMD .chain_failed
[ -f ".chain_stopped" ] && $CMD .chain_stopped
[ -f ".monitor_cache" ] && $CMD .monitor_cache

echo ""
if [ "$FORCE" = "0" ]; then
    echo "=== Nessun file cancellato (dry-run). Usa: bash src/cluster/clean.sh --force ==="
else
    echo "Pulizia completata."
fi
