#!/bin/bash
# ============================================================================
# SLURM batch script — T2G Evaluation sul cluster
#
# Uso:
#   CONFIG=experiments/configs/qwen25-05b/sft-grpo/few-shot.yaml sbatch cluster/eval.sh
#   CONFIG=experiments/configs/qwen25-05b/sft-grpo/few-shot.yaml CHECKPOINT="path/to/ckpt" sbatch cluster/eval.sh
#   CONFIG=experiments/configs/qwen25-05b/sft-grpo/few-shot.yaml CHECKPOINT="path/to/ckpt" BEST_OF_N=1 sbatch cluster/eval.sh
#   CONFIG=... PROMPTING=zero-shot sbatch cluster/eval.sh        # override modalità di prompting della sola eval
#   CONFIG=... DUAL_EVAL=1 sbatch cluster/eval.sh                # seconda passata con la modalità complementare
#
# --compare è sempre attivo sulle celle di training: valuta baseline (base
#   model SENZA checkpoint, STESSA config ⇒ stessa modalità di prompting e
#   decodifica della cella) + checkpoint, e genera grafici di confronto +
#   comparison.json + wandb con tag dedicati.
# BEST_OF_N=1 abilita la selezione best-of-N: è un ORACOLO DIAGNOSTICO
#   (limite superiore: quanto può essere buono il modello scegliendo il
#   migliore di N campioni), NON la metrica primaria — quella resta Pass@1.
#
# TERMINOLOGIA (due concetti distinti, non usarli come sinonimi):
#   - "no-checkpoint / base model": mancano i pesi addestrati (eval del base
#     model). NON dice nulla sul prompting.
#   - "zero-shot / few-shot": modalità di PROMPTING (con o senza esempi
#     few-shot nel prompt, da retrieval.enabled o da PROMPTING/--prompting).
# ============================================================================

# ┌────────────────────────────────────────────────────────┐
# │  CONFIGURA QUI — modifica account/partition/qos/email  │
# └────────────────────────────────────────────────────────┘
#SBATCH --job-name=eval-t2g
#SBATCH --account=thesis-course
#SBATCH --partition=thesis-course
#SBATCH --qos=gpu-xlarge
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1 --gres=shard:22528
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=bellamacina50@gmail.com
#SBATCH --output=logs/slurm-eval-%j.log

# ── Variabili progetto ────────────────────────────────────────────────────────
CHECKPOINT="${CHECKPOINT:-}"


if [ -z "$CONFIG" ]; then
    echo "❌ CONFIG non impostato. Uso:"
    echo "  CONFIG=experiments/configs/qwen25-05b/sft-grpo/few-shot.yaml sbatch cluster/eval.sh"
    exit 1
fi

# ── Setup ambiente ───────────────────────────────────────────────────────────
set -e

# SLURM copia lo script nella sua spool dir (/var/lib/slurm/slurmd/<job>/)
# e lo esegue da lì: BASH_SOURCE NON punta al repo e `source _lib.sh`
# fallirebbe ("File o directory non esistente"). La directory di
# sottomissione (SLURM_SUBMIT_DIR — la cwd al momento dello sbatch, che è
# sempre la repo root, sia da shell che da chain_tick.sh) è il posto giusto.
_lib_dir=$(dirname "${BASH_SOURCE[0]}")
if [ ! -f "${_lib_dir}/_lib.sh" ]; then
    _lib_dir="${SLURM_SUBMIT_DIR:-$HOME/neuro_symbolic_t2g}/cluster"
fi
SCRIPT_DIR=$(cd "${_lib_dir}" && pwd)
# shellcheck source=cluster/_lib.sh
source "$SCRIPT_DIR/_lib.sh"

# Ambiente offline PRIMA di qualunque python/apptainer: la prima
# invocazione e resolve_output_dir(), molto piu in alto di prepare_data,
# e i client HF leggono queste variabili all'import. Esportarle dopo non
# avrebbe alcun effetto.
export_offline_env
cd "$PROJ_DIR"

echo "============================================"
echo "  T2G Evaluation — Cluster"
echo "  Job ID:    ${SLURM_JOB_ID}"
echo "  Node:      $(hostname)"
echo "  Date:      $(date)"
echo "  Config:    ${CONFIG}"
echo "  Checkpoint: ${CHECKPOINT:-auto}"
echo "============================================"

mkdir -p logs

# ── Auto-detect trained checkpoint (config-heritage-aware) ────────────────────
# Risolve output_dir con src.utils.config.resolve_config (che gestisce
# `extends: base.yaml`) — NIENTE yaml.safe_load del solo file figlio.
# Il python gira DENTRO Apptainer sul compute node (stessa pattern di train.sh).
#
# Politica anti-silent-no-checkpoint (qui "zero-shot" NON c'entra: il
# pericolo è valutare il BASE MODEL senza pesi addestrati):
#   - config di training (ha training.output_dir) ma NESSUN checkpoint trovato
#     → FAIL LOUD (exit 1). Un eval del base model (senza checkpoint) su una
#     cella addestrata produrrebbe numeri senza senso senza alcun errore.
#   - config SENZA training.output_dir → eval senza checkpoint (base model)
#     legittimo e voluto.
resolve_output_dir() {
    local out=""
    if command -v apptainer >/dev/null 2>&1 && [ -f /shared/sifs/latest.sif ]; then
        out=$(apptainer exec /shared/sifs/latest.sif python -c "
from src.utils.config import resolve_config
try:
    print(resolve_config('${CONFIG}')['training']['output_dir'])
except Exception:
    print('')
" 2>/dev/null) || true
    else
        out=$(python3 -c "
from src.utils.config import resolve_config
try:
    print(resolve_config('${CONFIG}')['training']['output_dir'])
except Exception:
    print('')
" 2>/dev/null) || true
    fi
    echo "$out"
}

# Trova il checkpoint più recente sotto output_dir: run_*/final o
# run_*/checkpoint-* ; poi output_dir/final o output_dir/checkpoint-*.
# Stesso ordine del legacy, ma su un output_dir RISOLTO (extends-aware).
find_newest_checkpoint() {
    local out_dir="$1" latest_run best="" c
    latest_run=$(ls -1d "${out_dir}"/run_* 2>/dev/null | tail -1) || true
    if [ -n "$latest_run" ] && [ -d "$latest_run" ]; then
        if [ -d "$latest_run/final" ]; then
            best="$latest_run/final"
        else
            for c in "$latest_run"/checkpoint-*; do
                [ -d "$c" ] && best="$c"
            done
        fi
    fi
    if [ -z "$best" ]; then
        if [ -d "$out_dir/final" ]; then
            best="$out_dir/final"
        else
            for c in "$out_dir"/checkpoint-*; do
                [ -d "$c" ] && best="$c"
            done
        fi
    fi
    if [ -n "$best" ]; then
        echo "$best"
        return 0
    fi
    return 1
}

if [ -z "$CHECKPOINT" ]; then
    OUTPUT_DIR=$(resolve_output_dir)
    if [ -n "$OUTPUT_DIR" ]; then
        if DETECTED=$(find_newest_checkpoint "$OUTPUT_DIR"); then
            CHECKPOINT="$DETECTED"
            echo "Auto-detected trained checkpoint: $CHECKPOINT"
        else
            echo ""
            echo "═══════════════════════════════════════════════════════════"
            echo "  ❌ CHECKPOINT NON TROVATO — eval RIFIUTATO (exit 1)"
            echo "  Config:        ${CONFIG}"
            echo "  output_dir:    ${OUTPUT_DIR}   (risolto via resolve_config)"
            echo ""
            echo "  La config dichiara training.output_dir ma non esiste"
            echo "  nessun run_*/final|checkpoint-*: il modello NON è stato"
            echo "  addestrato. Un eval del base model (senza checkpoint) su"
            echo "  una cella non addestrata produrrebbe numeri senza senso"
            echo "  SENZA errori — rifiutiamo."
            echo ""
            echo "  Soluzioni:"
            echo "   1. addestra prima (run-all / chain / sbatch cluster/train.sh)"
            echo "   2. oppure forza un checkpoint esplicito:"
            echo "        CONFIG=${CONFIG} CHECKPOINT=<path> sbatch cluster/eval.sh"
            echo "═══════════════════════════════════════════════════════════"
            echo ""
            exit 1
        fi
    else
        echo "Config without training.output_dir - no-checkpoint eval intended (base model)"
    fi
fi

# Prepara dataset/vocab/bigram se mancanti (funzione shared da _lib.sh,
# idempotente — era triplicata tra setup.sh/train.sh/eval.sh)
prepare_data

# ── Offline-first: i compute node NON hanno DNS ──────────────────────────────
# Tutto il necessario è pre-cacheato da setup.sh/prepare_data. Senza questi
# export ogni richiesta hub costa ~30s di retry (HEAD dataset/modello, lookup
# peft al save) o CRASHA il job: transformers 5.3 in tokenizer init chiama
# model_info() per id non-locali (_patch_mistral_regex) → ConnectError
# (vedi slurm-eval-7077). Con HF_HUB_OFFLINE=1 transformers forza is_local=True
# e salta il check di rete; datasets usa la cache locale direttamente.
# Dopo prepare_data (non prima!) così il fallback download al primo avvio
# conserva la rete.
# (ambiente offline gia esportato subito dopo il source di _lib.sh)

# ── Modalità eval ─────────────────────────────────────────────────────────────
# (terminologia: "no-checkpoint" = mancano i pesi addestrati; "zero-shot /
#  few-shot" = modalità di PROMPTING — due concetti distinti, vedi header)
# - Config di TRAINING (ha output_dir + checkpoint auto/explicito): --compare
#   (baseline senza checkpoint, cachata/riusata, con la STESSA config della
#   cella ⇒ stessa modalità di prompting e decodifica + checkpoint).
# - Config EVAL-ONLY (senza training.output_dir) e nessun checkpoint
#   esplicito: --eval-baseline-only — singolo eval del base model. Con
#   --compare il base model verrebbe valutato DUE volte (Step A baseline +
#   Step B "checkpoint" = entrambi senza pesi addestrati).
# NB: OUTPUT_DIR è risolta nel blocco auto-detect sopra (solo quando
# CHECKPOINT è vuoto — con un checkpoint esplicito il confronto ha senso).
if [ -z "${CHECKPOINT}" ] && [ -z "${OUTPUT_DIR:-}" ]; then
    EVAL_ONLY_MODE=1
fi

if [ "${EVAL_ONLY_MODE:-0}" = "1" ]; then
    EVAL_ARGS="--config ${CONFIG} --plot --eval-baseline-only"
    echo "Eval-only config (no checkpoint, base model): --eval-baseline-only (niente --compare)"
else
    EVAL_ARGS="--config ${CONFIG} --plot --compare"
    # NB: con --compare la baseline riusa la STESSA config della cella (e lo
    # stesso eventuale --prompting): su una cella few-shot anche la baseline
    # di confronto è few-shot. L'eval stampa la modalità effettiva e la sua
    # provenienza (Prompting: ...).
fi

# Override opzionale del numero di campioni (default: quello del config,
# oggi 5000 da base.yaml). Esempio eval rapido: MAX_SAMPLES=500 CONFIG=...
# NOTA: un override rende i numeri NON confrontabili con le altre celle.
if [ -n "${MAX_SAMPLES:-}" ]; then
    EVAL_ARGS="${EVAL_ARGS} --max-samples ${MAX_SAMPLES}"
    echo "MAX_SAMPLES override: ${MAX_SAMPLES}"
fi

# Override opzionale della modalità di prompting (default: quella del
# config, comportamento invariato). Valori: zero-shot | few-shot.
# ATTENZIONE: l'override vale per QUESTA eval sola (non tocca la config di
# training) e con few-shot l'eval abortisce se max_prompt_length < 512.
if [ -n "${PROMPTING:-}" ]; then
    case "${PROMPTING}" in
        zero-shot|few-shot)
            EVAL_ARGS="${EVAL_ARGS} --prompting ${PROMPTING}"
            echo "PROMPTING override: ${PROMPTING}"
            ;;
        *)
            echo "❌ PROMPTING non valido: '${PROMPTING}' (valori amessi: zero-shot, few-shot)"
            exit 1
            ;;
    esac
fi

if [ -n "$CHECKPOINT" ]; then
    EVAL_ARGS="${EVAL_ARGS} --checkpoint ${CHECKPOINT}"
else
    echo "No-checkpoint mode: nessun peso addestrato, si valuta il base model (modalità di prompting: quella della config o PROMPTING)"
fi

# Best-of-N selection (opzionale — passa BEST_OF_N=1 per attivare)
# ORACOLO DIAGNOSTICO: misura il limite superiore della qualità selezionando
# il miglior completamento tra i N campionati. Non è la metrica primaria
# (quella resta Pass@1). Richiede evaluation.num_samples>1 nel config.
if [ "${BEST_OF_N}" = "1" ]; then
    EVAL_ARGS="${EVAL_ARGS} --best-of-n"
    echo "Best-of-N selection enabled"
fi


# Modalità di prompting dichiarata dalla config (per il dual pass):
# "few-shot" se retrieval.enabled, altrimenti "zero-shot". Stessa pattern di
# resolve_output_dir (python dentro Apptainer sul compute node). Stamp "err"
# quando la config non è risolvibile: il chiamante decude se procedere.
resolve_config_prompting() {
    local enabled=""
    if command -v apptainer >/dev/null 2>&1 && [ -f /shared/sifs/latest.sif ]; then
        enabled=$(apptainer exec /shared/sifs/latest.sif python -c "
from src.utils.config import resolve_config
try:
    print('yes' if resolve_config('${CONFIG}').get('retrieval', {}).get('enabled') else 'no')
except Exception:
    print('err')
" 2>/dev/null) || true
    else
        enabled=$(python3 -c "
from src.utils.config import resolve_config
try:
    print('yes' if resolve_config('${CONFIG}').get('retrieval', {}).get('enabled') else 'no')
except Exception:
    print('err')
" 2>/dev/null) || true
    fi
    echo "$enabled"
}

# Esegue l'eval python nell'ambiente del job (Apptainer se presente).
# Gli argomenti passano NON quotati di proposito: stesso word-splitting
# dell'invocazione originale (${EVAL_ARGS}).
run_eval_python() {
    if command -v apptainer &>/dev/null && [ -f /shared/sifs/latest.sif ]; then
        apptainer run --nv \
            --env WANDB_MODE=offline \
            --env PYTHONUNBUFFERED=1 \
            --env HF_HUB_OFFLINE=1 \
            --env TRANSFORMERS_OFFLINE=1 \
            --env HF_DATASETS_OFFLINE=1 \
            --env PYTORCH_ALLOC_CONF=garbage_collection_threshold:0.8 \
            /shared/sifs/latest.sif \
            python -m src.training.eval_t2g "$@"
    else
        python -m src.training.eval_t2g "$@"
    fi
}

echo ""
echo "Avvio evaluation..."
echo "  Args: ${EVAL_ARGS}"
echo ""

# ── Esecuzione (passata primaria) ─────────────────────────────────────────────
run_eval_python ${EVAL_ARGS}

# ── Dual eval (OPT-IN esplicito: DUAL_EVAL=1) ─────────────────────────────────
# Valuta la cella ANCHE con la modalità di prompting complementare: una cella
# "train few-shot / eval few-shot" accanto a "train few-shot / eval zero-shot"
# misura se il modello ha interiorizzato la mappatura o se dipende dal prompt
# come stampella. DEFAULT: disattivato — nessuna seconda passata, tempi e
# matrice di celle invariati.
# - Con PROMPTING impostato, il complemento è l'altra modalità.
# - Senza PROMPTING, il complemento si risolve dalla config (retrieval.enabled).
#   Se la config non è risolvibile, il dual pass viene SALTATO con messaggio
#   loud invece di tirare a indovinare.
# - Una dual pass few-shot su una config con max_prompt_length < 512 FALLISCE
#   LOUD (validazione runtime in eval_t2g.py): è voluto, non un bug.
# - I file di output della seconda passata hanno suffisso __<mode> (gestito
#   da eval_t2g), quindi le due modalità non si sovrascrivono.
# NB: se EVAL_ARGS contiene già --prompting (PROMPTING impostato), l'ultimo
# --prompting sulla riga vince in argparse — qui si appende il complemento.
if [ "${DUAL_EVAL:-0}" = "1" ]; then
    if [ -n "${PROMPTING:-}" ]; then
        if [ "${PROMPTING}" = "zero-shot" ]; then
            DUAL_PROMPTING="few-shot"
        else
            DUAL_PROMPTING="zero-shot"
        fi
    else
        CFG_PROMPTING=$(resolve_config_prompting)
        case "${CFG_PROMPTING}" in
            yes)  DUAL_PROMPTING="zero-shot" ;;  # config few-shot → complemento zero-shot
            no)   DUAL_PROMPTING="few-shot" ;;   # config zero-shot → complemento few-shot
            *)
                echo "⚠️  DUAL_EVAL=1 ma config non risolvibile: seconda passata SALTATA (imposta PROMPTING=... per forzarla)."
                DUAL_PROMPTING=""
                ;;
        esac
    fi
    if [ -n "${DUAL_PROMPTING}" ]; then
        DUAL_ARGS="${EVAL_ARGS} --prompting ${DUAL_PROMPTING}"
        echo ""
        echo "DUAL_EVAL attivo: seconda passata con --prompting ${DUAL_PROMPTING} (complemento della primaria)"
        echo "  Args: ${DUAL_ARGS}"
        echo ""
        run_eval_python ${DUAL_ARGS}
    fi
fi

echo ""
echo "============================================"
echo "  Evaluation completata!"
echo "  $(date)"
echo "============================================"
