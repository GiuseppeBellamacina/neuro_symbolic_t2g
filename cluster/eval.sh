#!/bin/bash
# ============================================================================
# SLURM batch script — T2G Evaluation sul cluster
#
# Uso:
#   CONFIG=experiments/configs/qwen25-05b/sft-grpo/few-shot.yaml sbatch cluster/eval.sh
#   CONFIG=experiments/configs/qwen25-05b/sft-grpo/few-shot.yaml CHECKPOINT="path/to/ckpt" sbatch cluster/eval.sh
#
# TUTTI i knob comportamentali dell'eval vivono nella sezione `evaluation:` del
# config YAML — qui NON ci sono variabili d'ambiente comportamentali (chi
# cercava MAX_SAMPLES=500, PROMPTING=zero-shot, DUAL_EVAL=1 o BEST_OF_N=1 ora
# imposta evaluation.max_samples / evaluation.prompting /
# evaluation.dual_prompting / evaluation.best_of_n nel config — vedi base.yaml).
# Di questo script restano solo gli IDENTIFICATORI:
#   CONFIG      quale config eseguire (obbligatorio)
#   CHECKPOINT  quale checkpoint valutare (opzionale: auto-rilevato sotto
#               training.output_dir)
#
# Comportamento (dedotto da eval_t2g.py, nessun flag da passare):
#   - cella di training (training.output_dir + checkpoint): confronto con la
#     baseline del base model (STESSA config ⇒ stessa modalità di prompting e
#     decodifica della cella; baseline cachata fra run) + grafici +
#     comparison.json;
#   - cella eval-only (baseline/*, senza training.output_dir e senza
#     checkpoint): solo eval del base model;
#   - evaluation.dual_prompting: true (default) → seconda passata con la
#     modalità di prompting complementare, file con suffisso __<mode>
#     (l'eval raddoppia: ~25 min per passata a 5000 prompt).
#
# TERMINOLOGIA (due concetti distinti, non usarli come sinonimi):
#   - "no-checkpoint / base model": mancano i pesi addestrati (eval del base
#     model). NON dice nulla sul prompting.
#   - "zero-shot / few-shot": modalità di PROMPTING (con o senza esempi
#     few-shot nel prompt, da retrieval.enabled o da evaluation.prompting).
# ============================================================================

# ┌────────────────────────────────────────────────────────┐
# │  CONFIGURA QUI — modifica account/partition/qos/email  │
# └────────────────────────────────────────────────────────┘
#SBATCH --job-name=eval-t2g
#SBATCH --account=thesis-course
#SBATCH --partition=thesis-course
#SBATCH --qos=gpu-xlarge
# --time esplicito: senza questa direttiva vale il default della partizione,
# e la valutazione doppia (evaluation.dual_prompting) l'ha resa lunga.
#
# Misurato su un run reale con vincolo di decodifica attivo (2000 prompt x 5
# generazioni): eval_baseline 05:19 -> eval_final 06:21 = 62 min, cioe' ~1,86
# s per prompt. Le celle senza grammatica costano circa un quinto.
#
#   3000 prompt              -> ~1,6 h per passata
#   dual (2 passate)         -> ~3,1 h
#   dual + 2 baseline        -> ~6,2 h  (solo al primo giro: la baseline e'
#                               cacheata e riusata dalle celle successive)
#
# 8h copre il caso peggiore con margine e resta sotto il cap di 12h della QoS
# gpu-xlarge (CLUSTER.md). Chi alza evaluation.max_samples oltre 3000 deve
# ricalcolare: il costo e' lineare nel numero di prompt.
#SBATCH --time=08:00:00
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
#
# GUARDIA CHECKPOINT PARZIALE: 'final' viene scritto SOLO a training
# completato (grpo_t2g_train.py fa save_model su output_dir/final). Se final
# manca e si ricade su un checkpoint-<step>, il modello è PARZIALE (training
# interrotto, TIMEOUT o ancora in corso). Il percorso automatico della catena
# è già protetto (_lib.sh rimuove l'eval dalla coda se il train fallisce),
# ma un eval manuale o ri-accodato no: si avverte LOUD qui e il JSON dei
# risultati porta checkpoint_incomplete: true + checkpoint_step (stamp di
# eval_t2g.py) — un eval su modello parziale NON è mai indistinguibile da uno
# su modello completo. NON si rifiuta l'eval: valutare un checkpoint
# intermedio è diagnostica legittima (progresso mid-training, sanity del
# resume) e il workflow TIMEOUT→resume di CLUSTER.md lo produce di proposito.
#
# ORDINAMENTO NUMERICO, non lessicografico: il glob della shell ordina
# `checkpoint-1000` PRIMA di `checkpoint-500` (confronto carattere per
# carattere: '1' < '5'), quindi prendere l'ultimo elemento del glob
# selezionerebbe checkpoint-500, cioe' il modello MENO addestrato. Con
# save_steps: 500 e max_steps: 5000 i checkpoint arrivano a 4 cifre, quindi
# il difetto e' sistematico, non un caso limite. `sort -t- -k2 -n` ordina
# sul numero dopo il trattino.
_newest_checkpoint_dir() {
    local parent="$1" c best=""
    for c in "$parent"/checkpoint-*; do
        [ -d "$c" ] && printf '%s\n' "$c"
    done | sort -t- -k2 -n | tail -1
}

find_newest_checkpoint() {
    local out_dir="$1" latest_run best="" used_partial=0
    latest_run=$(ls -1d "${out_dir}"/run_* 2>/dev/null | tail -1) || true
    if [ -n "$latest_run" ] && [ -d "$latest_run" ]; then
        if [ -d "$latest_run/final" ]; then
            best="$latest_run/final"
        else
            best=$(_newest_checkpoint_dir "$latest_run")
            [ -n "$best" ] && used_partial=1
        fi
    fi
    if [ -z "$best" ]; then
        if [ -d "$out_dir/final" ]; then
            best="$out_dir/final"
        else
            best=$(_newest_checkpoint_dir "$out_dir")
            [ -n "$best" ] && used_partial=1
        fi
    fi
    if [ -n "$best" ]; then
        if [ "$used_partial" = "1" ]; then
            # stderr DI PROPOSITO: la funzione è usata in command substitution,
            # ogni echo su stdout finirebbe nel valore catturato.
            echo "⚠️  CHECKPOINT PARZIALE: 'final' assente, uso l'ultimo checkpoint intermedio: $best" >&2
            echo "    Il training potrebbe essere interrotto o ancora in corso: l'eval misura un modello incompleto." >&2
            echo "    Il JSON dei risultati sarà marcato checkpoint_incomplete: true (con checkpoint_step)." >&2
        fi
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

# ── Costruzione comando eval ──────────────────────────────────────────────────
# NIENTE flag comportamentali: eval_t2g.py deduce compare/eval-baseline-only
# dal config (presenza di training.output_dir e del checkpoint) e legge ogni
# altro knob (plot, dual_prompting, prompting, max_samples, num_samples,
# best_of_n, …) dalla sezione evaluation:. Qui passiamo SOLO gli
# identificatori (config + checkpoint).
# - Cella di TRAINING (ha output_dir + checkpoint auto/esplicito): l'eval
#   confronta con la baseline del base model — STESSA config, quindi stessa
#   modalità di prompting e decodifica della cella; la baseline è cachata
#   fra run (fingerprint del contesto prompt).
# - Cella EVAL-ONLY (baseline/*, nessun output_dir): solo eval del base model.
# NB: OUTPUT_DIR è risolta nel blocco auto-detect sopra (solo quando
# CHECKPOINT è vuoto — con un checkpoint esplicito il confronto ha senso).
EVAL_ARGS="--config ${CONFIG}"

if [ -n "$CHECKPOINT" ]; then
    EVAL_ARGS="${EVAL_ARGS} --checkpoint ${CHECKPOINT}"
else
    echo "No-checkpoint mode: si valuta il base model (celle eval-only come baseline/*)."
    echo "  Modalità di prompting: quella della config (retrieval.enabled), salvo"
    echo "  override in evaluation.prompting."
fi

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

# ── Esecuzione ────────────────────────────────────────────────────────────────
# La passata dual (se evaluation.dual_prompting: true nel config) è gestita
# DENTRO eval_t2g.py: seconda passata con la modalità di prompting
# complementare nello stesso processo, file con suffisso __<mode> e cache
# baseline separata. Niente logica dual qui: era il vecchio DUAL_EVAL=1.
run_eval_python ${EVAL_ARGS}

echo ""
echo "============================================"
echo "  Evaluation completata!"
echo "  $(date)"
echo "============================================"
