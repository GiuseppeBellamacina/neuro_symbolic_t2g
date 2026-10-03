#!/bin/bash
# ============================================================================
# Pulizia selettiva — rimuove checkpoints, logs, results, figures e log SLURM
# di UNA cella. Accetta:
#   - il TAG di pipeline (es. aslg-pc12-grpo-few-shot, aslg-pc12-ablations-decoding-no-grammar,
#     phoenix-2014t-grpo-few-shot): lo stesso dei job SLURM train-<TAG>;
#   - il path della cella (es. phoenix-2014t/qwen25-05b/grpo/few-shot), cioe'
#     il path del config sotto experiments/configs/ senza .yaml.
#
# Layout: experiments/{checkpoints,logs,results,figures}/<dataset>/<modello>/
# <cella>/run_*/ (vedi src/utils/run_paths.py). Il TAG viene risolto in cella
# leggendo i config (shell-only, il login node NON ha python): un config il
# cui tag (_lib.sh::t2g_tag_from_config) e' uguale al TAG da' la cella, e
# training.output_dir (se dichiarato) da' la directory dei checkpoint.
#
# Match ESATTO, mai per sottostringa: col layout <dataset>/<modello>/ un glob
# experiments/checkpoints/*<TAG>*/ a un livello avrebbe matchato l'intera
# radice di un dataset (es. "aslg" → experiments/checkpoints/aslg-pc12/).
#
# Uso:
#   bash cluster/clean_model.sh                    # lista le celle presenti
#   bash cluster/clean_model.sh aslg-pc12-grpo-few-shot      # dry-run
#   bash cluster/clean_model.sh aslg-pc12-grpo-few-shot --all # cancella davvero
# ============================================================================

set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=cluster/_lib.sh
source "$SCRIPT_DIR/_lib.sh"
cd "$PROJ_DIR"

MODEL=""
FORCE=0
for arg in "$@"; do
    case "$arg" in
        --all) FORCE=1 ;;
        --help|-h)
            echo "Uso: bash cluster/clean_model.sh <TAG|CELLA> [--all]"
            echo ""
            echo "TAG   = tag del job (es. aslg-pc12-grpo-few-shot, phoenix-2014t-grpo-few-shot)"
            echo "CELLA = path sotto experiments/configs/ senza .yaml"
            echo "        (es. aslg-pc12/qwen25-05b/ablations/decoding/no-grammar)"
            echo "Senza argomenti: lista le celle con dei run"
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

# Celle (path sotto experiments/<kind>/) corrispondenti a MODEL, una per riga:
# la cella del config (path sotto configs/) e, se diversa, quella derivata da
# training.output_dir.
cell_candidates() {
    local cfg rel dir
    if [[ "$MODEL" == */* ]]; then
        echo "${MODEL%/}"
        return 0
    fi
    while IFS= read -r cfg; do
        [ -f "$cfg" ] || continue
        [ "$(t2g_tag_from_config "$cfg")" = "$MODEL" ] || continue
        rel="${cfg#experiments/configs/}"
        echo "${rel%.yaml}"
        dir=$(sed -n 's/.*output_dir:[[:space:]]*"\([^"]*\)".*/\1/p' "$cfg" | head -1) || true
        if [ -n "$dir" ] && [[ "$dir" == experiments/checkpoints/* ]]; then
            echo "${dir#experiments/checkpoints/}"
        fi
    done < <(find experiments/configs -type f -name '*.yaml' ! -name 'base.yaml' 2>/dev/null | sort)
}

# Log SLURM reali per un tag: i file sono logs/slurm-{train,eval}-<JOBID>.log
# e il JOBID si mappa dal JobName SLURM (train-<TAG>/eval-<TAG> via sacct).
slurm_logs_for_tag() {
    local tag="$1"
    local start
    start=$(date -d '14 days ago' +%Y-%m-%d 2>/dev/null || date +%Y-%m-%d)
    sacct --me --noheader --format=JobID,JobName --parsable2 \
        --starttime="$start" 2>/dev/null \
        | awk -F'|' -v m="$tag" '
            $2 == "train-" m || $2 == "eval-" m {
                if ($1 ~ /^[0-9]+$/) {
                    if ($2 ~ /^train-/) print "logs/slurm-train-" $1 ".log"
                    else print "logs/slurm-eval-" $1 ".log"
                }
            }' | sort -u
}

# Emette tutti i path (dir/file) da pulire, uno per riga.
emit_targets() {
    local cell kind
    while IFS= read -r cell; do
        [ -n "$cell" ] || continue
        for kind in checkpoints logs results figures; do
            [ -d "experiments/${kind}/${cell}" ] && echo "experiments/${kind}/${cell}/"
        done
    done < <(cell_candidates | sort -u)
    if [[ "$MODEL" != */* ]]; then
        slurm_logs_for_tag "$MODEL"
    fi
}

# ── Nessuna cella specificata: lista le celle con dei run ─────────────────
if [ -z "$MODEL" ]; then
    echo "=== Celle trovate (dry-run) ==="
    echo ""
    # Una cella e' la directory che contiene i run_* (profondita' variabile).
    while IFS= read -r d; do
        [ -d "$d" ] || continue
        echo "  ${d#experiments/checkpoints/} ($(du -sh "$d" 2>/dev/null | cut -f1))"
    done < <(find experiments/checkpoints -type d -name 'run_*' -prune -print 2>/dev/null \
        | sed 's|/run_[^/]*$||' | sort -u)
    echo ""
    echo "Per cancellare: bash cluster/clean_model.sh <TAG|CELLA> --all"
    exit 0
fi

TARGETS="$(emit_targets || true)"

# ── Dry-run per la cella specificata ──────────────────────────────────────
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
        echo ""
        echo "  (celle risolte: $(cell_candidates | sort -u | tr '\n' ' '))"
    fi
    echo ""
    echo "Per cancellare: bash cluster/clean_model.sh $MODEL --all"
    exit 0
fi

# ── Cancella ───────────────────────────────────────────────────────────────
echo "Pulizia cella: $MODEL"
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
