#!/bin/bash
# ============================================================================
# migrate_dataset_layout.sh — porta un clone del cluster dal layout
#   experiments/<kind>/qwen25-05b/...            (modello in testa)
# al layout
#   experiments/<kind>/aslg-pc12/qwen25-05b/...  (dataset/modello/...)
# introdotto insieme a PHOENIX-2014T (vedi src/utils/run_paths.py).
#
# Cosa fa (DRY-RUN di default, --apply per eseguire):
#   1. experiments/{checkpoints,logs,results,figures}/qwen25-05b
#        → experiments/<kind>/aslg-pc12/qwen25-05b  (mv, stesso filesystem);
#      Nessun symlink di compatibilita' di proposito: `sync_cluster.ps1
#      download-checkpoints` (scp -r) lo seguirebbe e scaricherebbe due volte
#      ogni checkpoint. L'unico path assoluto dentro i checkpoint e'
#      trainer_state.json::best_model_checkpoint: un --resume di un SFT
#      interrotto PRIMA della migrazione emette solo un warning di
#      transformers ("Could not locate the best model") e tiene l'ultimo
#      checkpoint. Migrare a catena ferma lo evita del tutto;
#   2. sposta il vecchio albero experiments/configs/qwen25-05b (rimasto sul
#      cluster perche' la sync via scp non cancella) in
#      experiments/.legacy-configs-qwen25-05b: lanciarne un config scriverebbe
#      di nuovo nel layout vecchio;
#   3. riscrive i path dei config nei file di .chain_state (job_chain,
#      last_job, chain_stopped, chain_errors), con backup *.pre-dataset-layout.
#
# I tag dei job ASLG-PC12 NON cambiano (grpo-few-shot resta grpo-few-shot),
# quindi retry, monitor e preset continuano a funzionare.
#
# Sicurezza: rifiuta di procedere con un job SLURM attivo/pending (spostare la
# directory di un job in esecuzione lo romperebbe) e, per ogni kind, se la
# destinazione esiste gia' (nessun merge automatico).
#
# Uso (sul cluster, dalla root del progetto, DOPO la sync del nuovo codice):
#   bash cluster/migrate_dataset_layout.sh            # mostra cosa farebbe
#   bash cluster/migrate_dataset_layout.sh --apply    # esegue
# ============================================================================

set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=cluster/_lib.sh
source "$SCRIPT_DIR/_lib.sh"
cd "$PROJ_DIR"

APPLY=0
for arg in "$@"; do
    case "$arg" in
        --apply) APPLY=1 ;;
        --help|-h)
            sed -n '2,/^# =====/p' "$0" | sed '$d'
            exit 0
            ;;
        *)
            echo "❌ Argomento sconosciuto: $arg" >&2
            exit 1
            ;;
    esac
done

OLD_MODEL="qwen25-05b"
NEW_PREFIX="aslg-pc12"

run() {
    if [ "$APPLY" -eq 1 ]; then
        echo "  + $*"
        "$@"
    else
        echo "  (dry-run) $*"
    fi
}

# Il servizio su Render manda un tick ogni 5 minuti: senza la catena ferma un
# tick puo' sottomettere un job fra il controllo su squeue e gli spostamenti.
if [ "$APPLY" -eq 1 ] && [ ! -f "$STOPPED_FILE" ]; then
    echo "❌ La catena non e' ferma: esegui chain-stop prima di migrare" >&2
    echo "   (altrimenti un tick puo' lanciare un job a meta' migrazione)." >&2
    exit 1
fi

if [ "$APPLY" -eq 1 ] && command -v squeue >/dev/null 2>&1; then
    if [ -n "$(squeue --me -h 2>/dev/null)" ]; then
        echo "❌ C'e' un job SLURM attivo o in coda: attendi la fine (o chain-stop" >&2
        echo "   + killalljobs) prima di migrare." >&2
        squeue --me >&2 || true
        exit 1
    fi
fi

echo "=== 1. Output degli esperimenti ==="
for kind in checkpoints logs results figures; do
    src="experiments/${kind}/${OLD_MODEL}"
    dst="experiments/${kind}/${NEW_PREFIX}/${OLD_MODEL}"
    if [ ! -d "$src" ]; then
        echo "  ${src}: assente, niente da fare"
        continue
    fi
    if [ -e "$dst" ]; then
        echo "  ⚠️  ${dst} esiste gia': unire a mano con ${src} (nessun merge automatico)"
        continue
    fi
    run mkdir -p "experiments/${kind}/${NEW_PREFIX}"
    run mv "$src" "$dst"
done

echo ""
echo "=== 2. Vecchio albero dei config ==="
old_cfg="experiments/configs/${OLD_MODEL}"
legacy_cfg="experiments/.legacy-configs-${OLD_MODEL}"
if [ -d "$old_cfg" ] && [ ! -L "$old_cfg" ]; then
    if [ ! -d "experiments/configs/${NEW_PREFIX}/${OLD_MODEL}" ]; then
        echo "  ⚠️  experiments/configs/${NEW_PREFIX}/${OLD_MODEL} non c'e': sincronizza prima"
        echo "     il nuovo codice (sync_cluster / pre-push), poi rilancia."
    elif [ -e "$legacy_cfg" ]; then
        echo "  ⚠️  ${legacy_cfg} esiste gia': rimuovere a mano ${old_cfg}"
    else
        run mv "$old_cfg" "$legacy_cfg"
    fi
else
    echo "  ${old_cfg}: assente, niente da fare"
fi

echo ""
echo "=== 3. Path dei config in .chain_state ==="
for f in "$CHAIN_FILE" "$LAST_JOB_FILE" "$STOPPED_FILE" "$ERRORS_FILE"; do
    [ -f "$f" ] || continue
    if grep -q "experiments/configs/${OLD_MODEL}/" "$f"; then
        n=$(grep -c "experiments/configs/${OLD_MODEL}/" "$f" || true)
        echo "  ${f#"$PROJ_DIR"/}: ${n} riga/e da riscrivere"
        run cp -p "$f" "${f}.pre-dataset-layout"
        run sed -i "s#experiments/configs/${OLD_MODEL}/#experiments/configs/${NEW_PREFIX}/${OLD_MODEL}/#g" "$f"
    else
        echo "  ${f#"$PROJ_DIR"/}: gia' nel layout nuovo"
    fi
done

echo ""
if [ "$APPLY" -eq 1 ]; then
    echo "✅ Migrazione completata."
else
    echo "Dry-run: nessuna modifica. Rilancia con --apply per eseguire."
fi
