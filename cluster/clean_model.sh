#!/bin/bash
# ============================================================================
# Pulizia selettiva — rimuove checkpoints, logs, results, figures e log SLURM
# di un modello specifico. Accetta sia il TAG di pipeline (es. sft-grpo)
# sia il nome reale della cartella (es. qwen25-05b-sft-grpo).
#
# Cerca in:
#   experiments/checkpoints/*<MODEL>*        (struttura flat)
#   experiments/logs/*<MODEL>*
#   experiments/results/*<MODEL>*
#   experiments/figures/*<MODEL>*
#   logs/slurm-{train,eval}-<JOBID>.log     (mappati via sacct JobName)
#
# Mapping tag→cartella reale, shell-only (il login node NON ha python): se il
# tag corrisponde a un config experiments/configs/t2g/*.yaml, il basename di
# training.output_dir viene estratto con grep e usato come candidato aggiuntivo
# (es. clean-model sft-grpo trova experiments/checkpoints/qwen25-05b-sft-grpo).
#
# Uso:
#   bash cluster/clean_model.sh                    # lista tutti i tag
#   bash cluster/clean_model.sh sft-grpo       # dry-run
#   bash cluster/clean_model.sh sft-grpo --all # cancella davvero
# ============================================================================

set -euo pipefail
cd "$HOME/neuro_symbolic_t2g"

MODEL=""
FORCE=0
for arg in "$@"; do
    case "$arg" in
        --all) FORCE=1 ;;
        --help|-h)
            echo "Uso: bash cluster/clean_model.sh <TAG> [--all]"
            echo ""
            echo "TAG = tag del config (es. sft-grpo, grpo-only, sft-only, ...)"
            echo "     oppure nome reale della cartella (es. qwen25-05b-sft-grpo)"
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

# Candidati: il tag stesso + i basename di training.output_dir dei config il
# cui nome matchano il tag (shell-only, niente python sul login node).
model_candidates() {
    local cfg tag dir
    echo "$MODEL"
    for cfg in experiments/configs/t2g/*.yaml experiments/configs/t2g/*.yaml; do
        [ -f "$cfg" ] || continue
        tag=$(basename "$cfg" .yaml | tr '_' '-')
        if [ "$tag" = "$MODEL" ]; then
            dir=$(sed -n 's/.*output_dir:[[:space:]]*"\([^"]*\)".*/\1/p' "$cfg" | head -1) || true
            if [ -n "$dir" ]; then
                echo "$(basename "$dir")"
            fi
        fi
    done
}

# Log SLURM reali per un modello: i file sono logs/slurm-{train,eval}-<JOBID>.log
# e il JOBID si mappa dal JobName SLURM (train-<TAG>/eval-<TAG> via sacct).
slurm_logs_for_model() {
    local model="$1"
    local start
    start=$(date -d '14 days ago' +%Y-%m-%d 2>/dev/null || date +%Y-%m-%d)
    sacct --me --noheader --format=JobID,JobName --parsable2 \
        --starttime="$start" 2>/dev/null \
        | awk -F'|' -v m="$model" '
            $2 == "train-" m || $2 == "eval-" m {
                if ($1 ~ /^[0-9]+$/) {
                    if ($2 ~ /^train-/) print "logs/slurm-train-" $1 ".log"
                    else print "logs/slurm-eval-" $1 ".log"
                }
            }' | sort -u
}

# Emette tutti i path (dir/file) da pulire, uno per riga.
emit_targets() {
    local cand d
    for cand in $(model_candidates); do
        [ -n "$cand" ] || continue
        for d in experiments/checkpoints/*"${cand}"*/ experiments/checkpoints/grpo/t2g/*"${cand}"*/; do
            [ -d "$d" ] && echo "$d"
        done
        for d in experiments/logs/*"${cand}"*/; do
            [ -d "$d" ] && echo "$d"
        done
        for d in experiments/results/*"${cand}"*/; do
            [ -d "$d" ] && echo "$d"
        done
        for d in experiments/figures/*"${cand}"*/; do
            [ -d "$d" ] && echo "$d"
        done
        slurm_logs_for_model "$cand"
    done | sort -u
}

# ── Nessun modello specificato: lista tutti i tag trovati ─────────────────
if [ -z "$MODEL" ]; then
    echo "=== Modelli trovati (dry-run) ==="
    echo ""
    for d in experiments/checkpoints/*/; do
        [ -d "$d" ] || continue
        echo "  $(basename "$d") ($(du -sh "$d" 2>/dev/null | cut -f1))"
    done
    for d in experiments/checkpoints/grpo/t2g/*/; do
        [ -d "$d" ] || continue
        echo "  grpo/t2g/$(basename "$d") ($(du -sh "$d" 2>/dev/null | cut -f1))"
    done
    if [ -d "experiments/results" ]; then
        for d in experiments/results/*/; do
            [ -d "$d" ] || continue
            echo "  results/$(basename "$d") ($(du -sh "$d" 2>/dev/null | cut -f1))"
        done
    fi
    echo ""
    echo "Per cancellare: bash cluster/clean_model.sh <TAG> --all"
    exit 0
fi

TARGETS="$(emit_targets || true)"

# ── Dry-run per il modello specificato ────────────────────────────────────
if [ "$FORCE" = "0" ]; then
    echo "=== DRY RUN per '$MODEL' — aggiungi --all per cancellare ==="
    echo ""
    if [ -z "$TARGETS" ]; then
        echo "  (niente trovato per '$MODEL')"
    else
        while IFS= read -r t; do
            [ -z "$t" ] && continue
            size=$(du -sh "$t" 2>/dev/null | cut -f1 || echo "?")
            kind="FILE"
            [ -d "$t" ] && kind="DIR "
            echo "  [$kind] $t ($size)"
        done <<< "$TARGETS"
        if [ "$(model_candidates | wc -l)" -gt 1 ]; then
            echo ""
            echo "  (candidati mappati dai config: $(model_candidates | tr '\n' ' '))"
        fi
    fi
    echo ""
    echo "Per cancellare: bash cluster/clean_model.sh $MODEL --all"
    exit 0
fi

# ── Cancella ───────────────────────────────────────────────────────────────
echo "Pulizia modello: $MODEL"
CLEANED=0
if [ -n "$TARGETS" ]; then
    while IFS= read -r t; do
        [ -z "$t" ] && continue
        kind="FILE"
        [ -d "$t" ] && kind="DIR "
        echo "  [$kind] $t"
        rm -rf "$t"
        CLEANED=1
    done <<< "$TARGETS"
fi

echo ""
if [ "$CLEANED" -eq 1 ]; then
    echo "✅ Pulizia completata per '$MODEL'."
else
    echo "ℹ️  Nessuna cartella da pulire per '$MODEL'."
fi
