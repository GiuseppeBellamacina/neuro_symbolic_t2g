#!/bin/bash
# ============================================================================
# Alias utili per il cluster DMI — progetto neuro_symbolic_t2g
#
# Uso:
#   source cluster/aliases.sh
#
# Per caricarli automaticamente, aggiungi al tuo ~/.bashrc:
#   source ~/neuro_symbolic_t2g/cluster/aliases.sh
# ============================================================================

# shellcheck source=cluster/_lib.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_lib.sh"

# ── Job management ───────────────────────────────────────────────────────────

# Controlla i miei job attivi
alias myjobs='squeue --me --format="%.10i %.20j %.8T %.10M %.6D %.20R %o"'

# Info dettagliata su un job (uso: jobinfo <JOB_ID>)
jobinfo() {
    if [ -z "$1" ]; then
        echo "Uso: jobinfo <JOB_ID>"
        return 1
    fi
    scontrol show job "$1"
}

# Cancella un job (uso: killjob <JOB_ID>)
alias killjob='scancel'

# Cancella tutti i miei job SLURM
alias killalljobs='scancel --me'

# ── Log monitoring ───────────────────────────────────────────────────────────

# Segui il log di un job di training (uso: trainlog <JOB_ID>)
trainlog() {
    if [ -z "$1" ]; then
        echo "Uso: trainlog <JOB_ID>"
        return 1
    fi
    local logfile="$PROJ_DIR/logs/slurm-train-${1}.log"
    if [ ! -f "$logfile" ]; then
        echo "Log non trovato: $logfile"
        return 1
    fi
    tail -f "$logfile"
}

# Segui il log di un job di eval (uso: evallog <JOB_ID>)
evallog() {
    if [ -z "$1" ]; then
        echo "Uso: evallog <JOB_ID>"
        return 1
    fi
    local logfile="$PROJ_DIR/logs/slurm-eval-${1}.log"
    if [ ! -f "$logfile" ]; then
        echo "Log non trovato: $logfile"
        return 1
    fi
    tail -f "$logfile"
}

# Mostra l'ultimo log — uso: lastlog [N_RIGHE]
lastlog() {
    local logfile
    logfile=$(ls -t "$PROJ_DIR"/logs/slurm*.log 2>/dev/null | head -1)
    if [ -z "$logfile" ]; then
        echo "Nessun log trovato in $PROJ_DIR/logs/"
        return 1
    fi
    echo "==> $logfile <=="
    if [ -n "$1" ]; then
        tail -n "$1" "$logfile"
    else
        tail -f "$logfile"
    fi
}

# ── Filesystem ───────────────────────────────────────────────────────────────

# Tree ricorsivo di una cartella (uso: tree <DIR> [DEPTH])
tree() {
    local dir="${1:-.}"
    local depth="${2:-3}"
    find "$dir" -maxdepth "$depth" | sed -e "s|[^/]*/|  |g" -e "s|  |├─|"
}

# ── GPU & risorse ────────────────────────────────────────────────────────────

# Stato GPU
gpu() {
    local jobid
    jobid=$(squeue --me --noheader --format="%i" 2>/dev/null | head -1)
    if [ -z "$jobid" ]; then
        echo "❌ Nessun job SLURM attivo."
        return 1
    fi
    srun --jobid="$jobid" --overlap nvidia-smi
}

# Uso disco del progetto
alias quota='quota -s'

# ── Quick commands ───────────────────────────────────────────────────────────

# Vai alla directory del progetto
alias proj='cd "$PROJ_DIR"'

# Mostra i checkpoint disponibili (layout FLAT: experiments/checkpoints/*/)
ckpts() {
    local base="$PROJ_DIR/experiments/checkpoints"
    if [ ! -d "$base" ]; then
        echo "Nessun checkpoint trovato."
        return 0
    fi
    echo "──── Checkpoints (flat layout) ────"
    local model run found c2
    for model in "$base"/*/; do
        [ -d "$model" ] || continue
        echo "  $(basename "$model"):"
        found=0
        for run in "$model"run_*; do
            [ -d "$run" ] || continue
            found=1
            echo "    $(basename "$run"):"
            ls -d "$run"/final "$run"/checkpoint-* 2>/dev/null | while read -r c2; do
                [ -n "$c2" ] && echo "      $(basename "$c2")"
            done
        done
        if [ "$found" -eq 0 ]; then
            ls -d "$model"final "$model"checkpoint-* 2>/dev/null | while read -r c2; do
                [ -n "$c2" ] && echo "      $(basename "$c2")"
            done
        fi
    done
}

# Lancia training (uso: train [--config PATH] [extra args...])
train() {
    local config="experiments/configs/t2g/sft-grpo.yaml"
    local extra_args=""
    while [ $# -gt 0 ]; do
        case "$1" in
            --config) config="$2"; shift 2 ;;
            *) extra_args="$extra_args $1"; shift ;;
        esac
    done
    cd "$PROJ_DIR" && CONFIG="$config" EXTRA_ARGS="$extra_args" sbatch cluster/train.sh
}

# Lancia eval (uso: run-eval [--config PATH] [--checkpoint PATH])
run-eval() {
    local config="experiments/configs/t2g/sft-grpo.yaml"
    local checkpoint=""
    while [ $# -gt 0 ]; do
        case "$1" in
            --config) config="$2"; shift 2 ;;
            --checkpoint) checkpoint="$2"; shift 2 ;;
            *) echo "❌ Argomento sconosciuto: $1"; return 1 ;;
        esac
    done
    cd "$PROJ_DIR" && CONFIG="$config" CHECKPOINT="$checkpoint" sbatch cluster/eval.sh
}

# Lancia train + eval (uso: run-all)
run-all() {
    cd "$PROJ_DIR" && bash cluster/run_all.sh "$@"
}

# Controlla lo stato della pipeline (job attivo / coda)
chain-status() {
    if [ -f "$STATE_DIR/chain_failed" ]; then
        local failed
        failed=$(cat "$STATE_DIR/chain_failed")
        echo "❌ Pipeline FALLITA - job: $failed"
        echo "   Per riprendere: chain-resume"
        return 1
    fi
    if [ -f "$STATE_DIR/chain_stopped" ]; then
        local info st_type st_tag
        info=$(cat "$STATE_DIR/chain_stopped")
        st_type=$(echo "$info" | cut -d: -f1)
        st_tag=$(echo "$info" | cut -d: -f3)
        echo "⚠️  Pipeline FERMATA su: $st_type $st_tag"
        echo "   Per riprendere: chain-start"
        return 0
    fi
    if [ -s "$STATE_DIR/job_chain" ]; then
        echo "⏳ Job in coda: $(wc -l < "$STATE_DIR/job_chain")"
    fi
    if [ -n "$(active_job_id)" ]; then
        echo "▶️  Job SLURM attivo: $(active_job_id) ($(active_job_name))"
    else
        echo "Nessuna pipeline attiva."
    fi
}

# Pulizia workspace (uso: clean [--force] [--all-cache])
clean() {
    cd "$PROJ_DIR" && bash cluster/clean.sh "$@"
}

# Pulizia selettiva di un modello (uso: clean-model <TAG> [--all])
clean-model() {
    cd "$PROJ_DIR" && bash cluster/clean_model.sh "$@"
}

# Aggiungi job alla pipeline attiva (uso: chain-add)
chain-add() {
    cd "$PROJ_DIR" && bash cluster/run_all.sh --append "$@"
}

# Rimuovi job dalla pipeline attiva (uso: chain-remove --models=1)
chain-remove() {
    cd "$PROJ_DIR" && bash cluster/run_all.sh --remove "$@"
}

# Ferma la pipeline senza perdere lo stato (uso: chain-stop [--force])
# Cancella il job SLURM attivo, salva lo stato per chain-start
# (.chain_stopped) con il config ATTIVO letto da .chain_state — MAI hardcoded.
chain-stop() {
    local force=0
    for arg in "$@"; do
        case "$arg" in
            --force) force=1 ;;
            --help|-h)
                echo "Uso: chain-stop [--force]"
                echo "  (default)  Ferma pipeline: cancella il job SLURM attivo."
                echo "             Salva lo stato per chain-start."
                echo "  --force    Cancella TUTTI i file di stato."
                return 0
                ;;
        esac
    done

    cd "$PROJ_DIR" || return 1
    mkdir -p "$STATE_DIR"

    # 2) Cancella il job SLURM attivo
    local active_id active_name
    active_id=$(active_job_id)
    active_name=$(active_job_name)
    if [ -n "$active_id" ]; then
        scancel "$active_id" 2>/dev/null
        echo "✅ Job SLURM $active_id ($active_name) cancellato"
    else
        echo "⚠️  Nessun job SLURM attivo"
    fi

    if [ "$force" -eq 1 ]; then
        rm -rf "$STATE_DIR"
        mkdir -p "$STATE_DIR"
        echo "🗑️  Stato pipeline cancellato (.chain_state/)"
        echo "Pipeline terminata definitivamente. Per ricominciare: run-all"
        return 0
    fi

    # Config ATTIVO dallo stato: ultimo job loggato o testa della coda.
    # Il vecchio hardcode di sft-grpo faceva ripartire chain-start col
    # config SBAGLIATO — ora il config è sempre quello reale della catena.
    local st_type="" st_cfg="" st_tag="" entry="" last=""
    if [ -s "$STATE_DIR/last_job" ]; then
        last=$(cat "$STATE_DIR/last_job")
        st_type=$(echo "$last" | cut -d: -f2)
        st_cfg=$(echo "$last" | cut -d: -f3)
        st_tag=$(echo "$last" | cut -d: -f4)
    elif [ -s "$STATE_DIR/job_chain" ]; then
        entry=$(head -1 "$STATE_DIR/job_chain")
        st_type=$(echo "$entry" | cut -d: -f1)
        st_cfg=$(echo "$entry" | cut -d: -f2)
        st_tag=$(echo "$entry" | cut -d: -f3)
    fi
    [ -n "$st_type" ] || st_type="none"
    echo "${st_type}:${st_cfg}:${st_tag}:0:${active_id}" > "$STATE_DIR/chain_stopped"
    rm -f "$STATE_DIR/chain_failed"
    echo "Pipeline fermata (config letto dallo stato: ${st_type}/${st_tag})."
    echo "Per riprendere: chain-start"
}

# Report di ripresa ("chain resumed: ...")
_chain_resume_report() {
    local last="" type="" tag="" remaining=0
    [ -s "$STATE_DIR/last_job" ] && last=$(cat "$STATE_DIR/last_job")
    type=$(echo "$last" | cut -d: -f2)
    tag=$(echo "$last" | cut -d: -f4)
    [ -s "$STATE_DIR/job_chain" ] && remaining=$(wc -l < "$STATE_DIR/job_chain")
    if [ -n "$type" ]; then
        echo "chain resumed: ${type}-${tag} in esecuzione, next in queue: ${remaining}"
    else
        echo "chain resumed: nessun job ancora sottoposto, ${remaining} in coda"
    fi
}

# Logica condivisa di ripresa (chain-start e chain-resume)
_chain_resume_impl() {
    cd "$PROJ_DIR" || return 1
    mkdir -p "$STATE_DIR" logs

    # 1) Ricostruisci la coda dal marcatore .chain_stopped (da chain-stop)
    if [ -f "$STATE_DIR/chain_stopped" ]; then
        local info st_type st_cfg st_tag head=""
        info=$(cat "$STATE_DIR/chain_stopped")
        st_type=$(echo "$info" | cut -d: -f1)
        st_cfg=$(echo "$info" | cut -d: -f2)
        st_tag=$(echo "$info" | cut -d: -f3)
        [ -s "$STATE_DIR/job_chain" ] && head=$(head -1 "$STATE_DIR/job_chain")

        case "$st_type" in
            train)
                rebuild_chain "train:${st_cfg}:${st_tag}:--resume"
                # Evita eval duplicato se già in testa alla coda originale
                if [ "$(echo "$head" | cut -d: -f1)" != "eval" ] || [ "$(echo "$head" | cut -d: -f3)" != "$st_tag" ]; then
                    rebuild_chain "eval:${st_cfg}:${st_tag}"
                fi
                echo "→ Training $st_tag verrà ripreso dall'ultimo checkpoint"
                ;;
            eval)
                rebuild_chain "eval:${st_cfg}:${st_tag}"
                echo "→ Eval $st_tag verrà rieseguito"
                ;;
            none|"")
                echo "ℹ️  La pipeline era già in pausa — riavvio."
                ;;
            *)
                echo "⚠️  chain_stopped malformato: $info"
                ;;
        esac
        rm -f "$STATE_DIR/chain_stopped"
    fi

    # 2) Sanity: c'è qualcosa da eseguire?
    if [ ! -s "$STATE_DIR/job_chain" ]; then
        echo "⚠️  Nessun job in coda (job_chain vuoto). Nulla da riprendere."
        return 1
    fi
    echo ""
    echo "Catena ($(wc -l < "$STATE_DIR/job_chain") job):"
    cat -n "$STATE_DIR/job_chain"
    echo ""

    # 3) Kick: un tick immediato — la catena avanza poi con l'hook bashrc
    #    (chain-hook-install) e/o il server esterno (POST /tick).
    bash cluster/chain_tick.sh --quiet
    local rc=$?
    [ "$rc" -ne 0 ] && echo "⚠️  tick rc=$rc - riproverà al prossimo tick."
    _chain_resume_report
}

# Riprendi la pipeline dopo chain-stop (uso: chain-start)
chain-start() {
    _chain_resume_impl
}

# Riprendi una catena interrotta (uso: chain-resume)
# Es. daemon ucciso dal reaper: job_chain non vuota, nessun job attivo.
# Non richiede .chain_failed: la coda stessa è lo stato.
chain-resume() {
    _chain_resume_impl
}

# Mostra la catena di job attuale (uso: chain-show)
chain-show() {
    chain-status
    echo ""
    if [ -f "$STATE_DIR/chain_stopped" ]; then
        local info st_type st_tag
        info=$(cat "$STATE_DIR/chain_stopped")
        st_type=$(echo "$info" | cut -d: -f1)
        st_tag=$(echo "$info" | cut -d: -f3)
        [ "$st_type" != "none" ] && echo "⏸️  Pipeline fermata su: $st_type $st_tag"
        echo "   Per riprendere: chain-start"
        echo ""
    fi
    if [ ! -s "$STATE_DIR/job_chain" ]; then
        echo "Nessun job in coda."
        return 0
    fi
    echo "Job in coda ($(wc -l < "$STATE_DIR/job_chain")):"
    cat -n "$STATE_DIR/job_chain"
}

# Installa l'hook bashrc (PROMPT_COMMAND throttled) che avanza la catena
# con chain_tick.sh --quiet quando job_chain non è vuota. Silenzioso, veloce,
# sicuro se il progetto non esiste. Va ri-eseguito dopo un wipe della home.
chain-hook-install() {
    local bashrc="$HOME/.bashrc"
    if grep -qF "# >>> t2g-chain-hook >>>" "$bashrc" 2>/dev/null; then
        echo "⚠️  Hook già presente in ~/.bashrc"
        return 1
    fi
    cat >> "$bashrc" <<'HOOKEOF'

# >>> t2g-chain-hook >>>
# Silent PROMPT_COMMAND hook: while a chain is pending, run the one-shot tick
# at most every 300s (throttled via .chain_state/tick_stamp). Safe if the
# project dir is missing. Re-run `chain-hook-install` after a home wipe.
_t2g_chain_hook() {
    local f="$HOME/neuro_symbolic_t2g/.chain_state/job_chain"
    local s="$HOME/neuro_symbolic_t2g/.chain_state/tick_stamp"
    [ -f "$f" ] && [ -s "$f" ] || return 0
    local now last=0
    now=$(date +%s)
    [ -f "$s" ] && last=$(cat "$s" 2>/dev/null || echo 0)
    case "$last" in
        ''|*[!0-9]*) last=0 ;;
    esac
    if [ $((now - last)) -gt 300 ]; then
        echo "$now" > "$s"
        bash "$HOME/neuro_symbolic_t2g/cluster/chain_tick.sh" --quiet >/dev/null 2>&1 || true
    fi
}
case ";${PROMPT_COMMAND:-};" in
    *";_t2g_chain_hook;"*) ;;
    *) PROMPT_COMMAND="_t2g_chain_hook;${PROMPT_COMMAND:-}" ;;
esac
# <<< t2g-chain-hook <<<
HOOKEOF
    echo "✅ Hook installato in ~/.bashrc (PRIMARIO — resilienza della catena; attivo dal prossimo login, per ora: source ~/.bashrc)."
    echo "   Nota: ri-eseguire chain-hook-install dopo un wipe della home."
}

chain-hook-uninstall() {
    local bashrc="$HOME/.bashrc"
    if ! grep -qF "# >>> t2g-chain-hook >>>" "$bashrc" 2>/dev/null; then
        echo "⚠️  Hook non presente in ~/.bashrc"
        return 1
    fi
    sed -i '/^# >>> t2g-chain-hook >>>/,/^# <<< t2g-chain-hook <<</d' "$bashrc"
    echo "✅ Hook rimosso da ~/.bashrc"
}

# Monitor live della pipeline (uso: monitor [--poll N])
monitor() {
    cd "$PROJ_DIR" && python3 -u -m src.utils.chain_monitor "$@"
}

# Alias t2g-* (allineati alla documentazione CLUSTER.md)
alias t2g-train='train'
alias t2g-eval='run-eval'
alias t2g-run-all='run-all'
alias t2g-monitor='monitor'
alias t2g-chain-show='chain-show'
alias t2g-chain-stop='chain-stop'
alias t2g-chain-start='chain-start'
alias t2g-chain-resume='chain-resume'
alias t2g-clean='clean'
alias t2g-gpu='gpu'
alias t2g-trainlog='trainlog'
alias t2g-help='diego'

# Genera tabella + grafico cross-config dopo l'ablation (uso: ablation-summary)
ablation-summary() {
    cd "$PROJ_DIR" && python3 -u -m src.utils.ablation_summary "$@"
}

# ── Pip / Environment ────────────────────────────────────────────────────────

# Pulisci tutti i pacchetti --user
pip-clean() {
    echo "🗑️  Rimozione pacchetti pip --user..."
    rm -rf ~/.local/lib/python3.*/site-packages/*
    rm -rf ~/.local/bin/*
    echo "✅ ~/.local ripulito"
}

# (Re)installa dipendenze: tutto da pyproject.toml (core + extra "retrieval";
# l'extra "dev" — formattazione/test — è escluso di proposito dal cluster).
pip-setup() {
    echo "📦 Installazione dipendenze (core + retrieval, niente dev)..."
    cd "$PROJ_DIR" && bash cluster/setup.sh
}

# Pulisci e reinstalla da zero
pip-reset() {
    pip-clean
    pip-setup
}

# ── Meta ─────────────────────────────────────────────────────────────────────

_DIEGO_ALIASES="myjobs jobinfo killjob killalljobs trainlog evallog lastlog tree gpu quota proj ckpts train run-eval run-all chain-status clean clean-model chain-add chain-remove chain-stop chain-start chain-resume chain-show chain-hook-install chain-hook-uninstall monitor ablation-summary pip-clean pip-setup pip-reset unload-aliases install-aliases uninstall-aliases t2g-train t2g-eval t2g-run-all t2g-monitor t2g-chain-show t2g-chain-stop t2g-chain-start t2g-chain-resume t2g-clean t2g-gpu t2g-trainlog t2g-help"

# Mostra i comandi disponibili
diego() {
    echo "Comandi disponibili:"
    echo ""
    echo "── Job management ──"
    echo "   myjobs            — lista job attivi"
    echo "   jobinfo <ID>      — dettagli job"
    echo "   killjob <ID>      — cancella job"
    echo "   killalljobs       — cancella tutti i miei job SLURM"
    echo ""
    echo "── Log monitoring ──"
    echo "   trainlog <ID> — segui log training"
    echo "   evallog <ID>  — segui log eval"
    echo "   lastlog [N]   — segui l'ultimo log (N=ultime N righe)"
    echo ""
    echo "── Training & eval ──"
    echo "   train [--config PATH] [extra args...]"
    echo "                     — lancia training (default: experiments/configs/t2g/sft-grpo.yaml)"
    echo "   run-eval [--config PATH] [--checkpoint PATH]"
    echo "                     — lancia evaluation"
    echo "   run-all [config_name] [--ablation|--train-only|--eval-only|--resume|--append|--force]"
    echo "                     — lancia pipeline train+eval (tick + avanza via hook/server)"
    echo ""
    echo "   Config disponibili (passa il nome senza .yaml):"
    echo "     sft-grpo               pipeline principale SFT+GRPO (default)"
    echo "     sft-only               SFT supervised da solo (decomposizione)"
    echo "     grpo-only              GRPO senza SFT (decomposizione)"
    echo "     sft-grpo-structure     SFT+GRPO + structural_dense (ablation)"
    echo "     sft-grpo-viterbi       SFT+GRPO + viterbi_distance (ablation)"
    echo "     sft-grpo-soft-viterbi  SFT+GRPO + soft_viterbi (ablation)"
    echo "     sft-grpo-all-rewards   SFT+GRPO + tutti i moduli sperimentali"
    echo "     sft-grpo-no-grammar    SFT+GRPO senza constrained decoding"
    echo "     zero-shot              Base model senza grammar (solo eval)"
    echo "     zero-shot-grammar      Base model con grammar (solo eval)"
    echo ""
    echo "── Pipeline (tick-based) ──"
    echo "   chain-show   — mostra stato pipeline + job in coda"
    echo "   chain-add    — aggiungi job alla pipeline attiva"
    echo "   chain-remove — rimuovi job dalla coda"
    echo "   chain-stop   — ferma pipeline (preserva stato, config dallo stato)"
    echo "   chain-start  — riprendi pipeline dopo chain-stop"
    echo "   chain-resume — riprendi catena interrotta (es. daemon ucciso)"
    echo "   chain-hook-install/uninstall"
    echo "                    — hook bashrc che avanza la catena al login"
    echo ""
    echo "── Monitor ──"
    echo "   monitor [--poll N] [--tab] [--samples [N]] [--metrics] [--all [N]]"
    echo "                    — monitor live della pipeline"
    echo "   ablation-summary  — genera tabella + grafico cross-config dopo l'ablation"
    echo ""
    echo "── Utilità ──"
    echo "   proj         — cd al progetto"
    echo "   ckpts        — mostra checkpoint (layout flat)"
    echo "   gpu          — stato GPU"
    echo "   quota        — uso disco progetto"
    echo "   clean [--force] [--all-cache]"
    echo "                    — pulizia workspace (preserva retriever_index)"
    echo "   clean-model <TAG> [--all]"
    echo "                    — pulisci checkpoints/logs/results/figures/slurm di un modello"
    echo ""
    echo "── Pip / Environment ──"
    echo "   pip-clean    — rimuovi pacchetti pip --user"
    echo "   pip-setup    — (re)installa dipendenze (core + retrieval da pyproject.toml)"
    echo "   pip-reset    — pip-clean + pip-setup"
    echo ""
    echo "── Alias t2g-* ──"
    echo "   t2g-train / t2g-eval / t2g-run-all / t2g-monitor"
    echo "   t2g-chain-show / t2g-chain-stop / t2g-chain-start / t2g-chain-resume"
    echo ""
    echo "── Meta ──"
    echo "   diego          — mostra questo messaggio"
    echo "   unload-aliases — rimuovi alias (sessione corrente)"
    echo "   install-aliases  — aggiungi alias al .bashrc (permanente)"
    echo "   uninstall-aliases — rimuovi alias dal .bashrc"
}

# Rimuovi tutti gli alias e funzioni custom (solo sessione corrente)
unload-aliases() {
    for cmd in $_DIEGO_ALIASES; do
        unalias "$cmd" 2>/dev/null
        unset -f "$cmd" 2>/dev/null
    done
    unset _DIEGO_ALIASES PROJ_DIR
    echo "✅ Alias rimossi (sessione corrente)."
}

_ALIASES_SOURCE_LINE="source ~/neuro_symbolic_t2g/cluster/aliases.sh"

# Aggiungi alias al .bashrc
install-aliases() {
    if grep -qF "$_ALIASES_SOURCE_LINE" ~/.bashrc 2>/dev/null; then
        echo "⚠️  Alias già presenti in ~/.bashrc"
    else
        echo "$_ALIASES_SOURCE_LINE" >> ~/.bashrc
        echo "✅ Alias aggiunti a ~/.bashrc (attivi dal prossimo login)"
    fi
    # Aggiungi ~/.local/bin al PATH (persistente, per i binari pip --user)
    if ! grep -qF 'export PATH="$HOME/.local/bin:$PATH"' ~/.bashrc 2>/dev/null; then
        echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
        echo "✅ ~/.local/bin aggiunto al PATH in ~/.bashrc"
    fi
    # Installa anche l'hook della chain (idempotente: skip se già presente;
    # una volta deployato il servizio esterno si può rimuovere con
    # chain-hook-uninstall — vedi CLUSTER.md § Chain)
    chain-hook-install || true
}

# Rimuovi alias dal .bashrc
uninstall-aliases() {
    if grep -qF "$_ALIASES_SOURCE_LINE" ~/.bashrc 2>/dev/null; then
        sed -i "\|$_ALIASES_SOURCE_LINE|d" ~/.bashrc
        echo "✅ Alias rimossi da ~/.bashrc"
    else
        echo "⚠️  Alias non presenti in ~/.bashrc"
    fi
    # Rimuove anche l'hook della chain se installato
    chain-hook-uninstall || true
    unload-aliases
}

echo "✅ Alias caricati. Digita 'diego' per la lista comandi."
