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

PROJ_DIR="$HOME/neuro_symbolic_t2g"
STATE_DIR="$PROJ_DIR/.chain_state"

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

# Cancella tutti i miei job
alias killalljobs='watcher-kill && scancel --me'

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

# Mostra i checkpoint disponibili
ckpts() {
    local base="$PROJ_DIR/experiments/checkpoints"
    if [ ! -d "$base" ]; then
        echo "Nessun checkpoint trovato."
        return 0
    fi
    echo "──── Checkpoints ────"
    for d in "$base"/grpo/t2g/*/; do
        [ -d "$d" ] || continue
        echo "  $(basename "$d"):"
        for run_dir in "$d"*/; do
            [ -d "$run_dir" ] || continue
            echo "    $(basename "$run_dir"):"
            ls -d "$run_dir"checkpoint-* 2>/dev/null | while read -r c2; do echo "      $(basename "$c2")"; done
        done
    done
}

# Lancia training (uso: train [--config PATH] [extra args...])
train() {
    local config="experiments/configs/t2g/grpo_qwen05.yaml"
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
    local config="experiments/configs/t2g/grpo_qwen05.yaml"
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

# Controlla se il watcher è attivo
watcher-status() {
    if [ -f "$STATE_DIR/chain_failed" ]; then
        local failed=$(cat "$STATE_DIR/chain_failed")
        echo "❌ Pipeline FALLITA — job: $failed"
        echo "   Per riprendere: run-all --resume"
        return 1
    fi
    if [ -f "$STATE_DIR/chain_pid" ]; then
        local pid=$(cat "$STATE_DIR/chain_pid")
        if ps -p "$pid" > /dev/null 2>&1; then
            echo "✅ Watcher attivo (PID $pid)"
        else
            echo "❌ Watcher morto (PID $pid non trovato)"
            rm -f "$STATE_DIR/chain_pid"
            return 1
        fi
    else
        echo "❌ Nessun watcher attivo"
        return 1
    fi
}

# Uccidi il watcher
watcher-kill() {
    if [ -f "$STATE_DIR/chain_pid" ]; then
        local pid=$(cat "$STATE_DIR/chain_pid")
        if ! ps -p "$pid" > /dev/null 2>&1; then
            echo "⚠️  Watcher (PID $pid) già morto"
            rm -f "$STATE_DIR/chain_pid"
            return 0
        fi
        read -p "Uccidere il watcher (PID $pid)? [y/N] " confirm
        case "$confirm" in
            [yY]|[yY][eE][sS])
                if kill "$pid" 2>/dev/null; then
                    echo "✅ Watcher (PID $pid) terminato"
                else
                    echo "⚠️  Watcher (PID $pid) già morto"
                fi
                rm -f "$STATE_DIR/chain_pid"
                ;;
            *)
                echo "Annullato."
                ;;
        esac
    else
        echo "Nessun watcher attivo"
    fi
}

# Pulizia workspace (uso: clean [--force])
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
chain-stop() {
    local force=0
    for arg in "$@"; do
        case "$arg" in
            --force) force=1 ;;
            --help|-h)
                echo "Uso: chain-stop [--force]"
                echo "  (default)   Ferma pipeline. Cancella job SLURM attivo + watcher."
                echo "              Salva lo stato per poter fare chain-start."
                echo "  --force     Cancella TUTTI i file di stato."
                return 0
                ;;
        esac
    done

    cd "$PROJ_DIR"

    local active_job=""
    active_job=$(squeue --me --noheader --format="%i %j" 2>/dev/null | head -1 | awk '{print $1}')
    local active_name=$(squeue --me --noheader --format="%i %j" 2>/dev/null | head -1 | awk '{print $2}')

    if [ -n "$active_job" ]; then
        scancel "$active_job" 2>/dev/null
        echo "✅ Job SLURM $active_job ($active_name) cancellato"
    else
        echo "⚠️  Nessun job SLURM attivo"
    fi

    if [ -f "$STATE_DIR/chain_pid" ]; then
        local pid=$(cat "$STATE_DIR/chain_pid")
        kill "$pid" 2>/dev/null && echo "✅ Watcher (PID $pid) terminato"
        rm -f "$STATE_DIR/chain_pid"
    fi

    if [ "$force" -eq 1 ]; then
        rm -rf "$STATE_DIR"
        mkdir -p "$STATE_DIR"
        echo "🗑️  Stato pipeline cancellato (.chain_state/)"
        echo "Pipeline terminata definitivamente. Per ricominciare: run-all"
    else
        local stopped_type=$(echo "$active_name" | cut -d- -f1)
        local stopped_tag=$(echo "$active_name" | cut -d- -f2-)
        echo "${stopped_type}:experiments/configs/t2g/grpo_qwen05.yaml:${stopped_tag}:0:${active_job}" > "$STATE_DIR/chain_stopped"
        rm -f "$STATE_DIR/chain_failed"
        echo "Pipeline fermata. Per riprendere: chain-start"
    fi
}

# Riprendi la pipeline dopo chain-stop (uso: chain-start)
chain-start() {
    cd "$PROJ_DIR"

    if [ ! -f "$STATE_DIR/chain_stopped" ]; then
        echo "❌ Nessun chain_stopped trovato."
        echo "   chain-start funziona solo dopo chain-stop (senza --force)."
        return 1
    fi

    local stopped_info=$(cat "$STATE_DIR/chain_stopped")
    local stopped_type=$(echo "$stopped_info" | cut -d: -f1)
    local stopped_cfg=$(echo "$stopped_info" | cut -d: -f2)
    local stopped_tag=$(echo "$stopped_info" | cut -d: -f3)
    local stopped_stage=$(echo "$stopped_info" | cut -d: -f4)

    if [ "$stopped_type" = "none" ]; then
        echo "ℹ️  La pipeline era già in pausa. Riavvio il watcher."
    elif [ "$stopped_type" = "train" ]; then
        local RESUME_CHAIN=$(mktemp)
        echo "train:${stopped_cfg}:${stopped_tag}:--resume" > "$RESUME_CHAIN"
        local next_in_chain=""
        [ -f "$STATE_DIR/job_chain" ] && [ -s "$STATE_DIR/job_chain" ] && next_in_chain=$(head -1 "$STATE_DIR/job_chain")
        local next_type=$(echo "$next_in_chain" | cut -d: -f1)
        local next_tag=$(echo "$next_in_chain" | cut -d: -f3)
        if [ "$next_type" != "eval" ] || [ "$next_tag" != "$stopped_tag" ]; then
            echo "eval:${stopped_cfg}:${stopped_tag}" >> "$RESUME_CHAIN"
        fi
        [ -f "$STATE_DIR/job_chain" ] && [ -s "$STATE_DIR/job_chain" ] && cat "$STATE_DIR/job_chain" >> "$RESUME_CHAIN"
        mv "$RESUME_CHAIN" "$STATE_DIR/job_chain"
        echo "→ Training $stopped_tag verrà ripreso dall'ultimo checkpoint"
    elif [ "$stopped_type" = "eval" ]; then
        local RESUME_CHAIN=$(mktemp)
        if [ "$stopped_stage" -gt 0 ]; then
            echo "eval:${stopped_cfg}:${stopped_tag}:--skip-stages=${stopped_stage}" > "$RESUME_CHAIN"
        else
            echo "eval:${stopped_cfg}:${stopped_tag}" > "$RESUME_CHAIN"
        fi
        [ -f "$STATE_DIR/job_chain" ] && [ -s "$STATE_DIR/job_chain" ] && cat "$STATE_DIR/job_chain" >> "$RESUME_CHAIN"
        mv "$RESUME_CHAIN" "$STATE_DIR/job_chain"
        echo "→ Eval $stopped_tag verrà rieseguito"
    fi

    rm -f "$STATE_DIR/chain_stopped" "$STATE_DIR/chain_failed"

    if [ ! -f "$STATE_DIR/job_chain" ] || [ ! -s "$STATE_DIR/job_chain" ]; then
        echo "⚠️  Nessun job rimanente nella catena."
        return 0
    fi

    local TOTAL=$(wc -l < "$STATE_DIR/job_chain")
    echo "Catena ($TOTAL job):"
    cat -n "$STATE_DIR/job_chain"

    if [ -f "$STATE_DIR/chain_pid" ]; then
        local OLD_PID=$(cat "$STATE_DIR/chain_pid")
        kill "$OLD_PID" 2>/dev/null
        rm -f "$STATE_DIR/chain_pid"
    fi

    mkdir -p logs
    nohup bash cluster/chain_next.sh >> logs/chain_watcher.log 2>&1 &

    # Attendi che il watcher scriva il suo PID
    sleep 2
    local WATCHER_PID=""
    if [ -f "$STATE_DIR/chain_pid" ]; then
        WATCHER_PID=$(cat "$STATE_DIR/chain_pid")
    fi
    echo "Pipeline ripresa! Watcher PID: ${WATCHER_PID:-sconosciuto}"
}

# Mostra la catena di job attuale (uso: chain-show)
chain-show() {
    watcher-status
    echo ""
    if [ -f "$STATE_DIR/chain_stopped" ]; then
        local info=$(cat "$STATE_DIR/chain_stopped")
        local st_type=$(echo "$info" | cut -d: -f1)
        local st_tag=$(echo "$info" | cut -d: -f3)
        [ "$st_type" != "none" ] && echo "⏸️  Pipeline fermata su: $st_type $st_tag"
        echo "   Per riprendere: chain-start"
        echo ""
    fi
    if [ ! -f "$STATE_DIR/job_chain" ] || [ ! -s "$STATE_DIR/job_chain" ]; then
        echo "Nessun job in coda."
        return 0
    fi
    local total=$(wc -l < "$STATE_DIR/job_chain")
    echo "Job in coda ($total):"
    cat -n "$STATE_DIR/job_chain"
}

# Monitor live della pipeline (uso: monitor [--poll N])
monitor() {
    cd "$PROJ_DIR" && python3 -u -m src.utils.chain_monitor "$@"
}

# ── Pip / Environment ────────────────────────────────────────────────────────

# Pulisci tutti i pacchetti --user
pip-clean() {
    echo "🗑️  Rimozione pacchetti pip --user..."
    rm -rf ~/.local/lib/python3.*/site-packages/*
    rm -rf ~/.local/bin/*
    echo "✅ ~/.local ripulito"
}

# (Re)installa dipendenze
pip-setup() {
    echo "📦 Installazione dipendenze..."
    cd "$PROJ_DIR" && bash cluster/setup.sh
}

# Pulisci e reinstalla da zero
pip-reset() {
    pip-clean
    pip-setup
}

# ── Meta ─────────────────────────────────────────────────────────────────────

_DIEGO_ALIASES="myjobs jobinfo killjob killalljobs trainlog evallog lastlog tree gpu quota proj ckpts train run-eval run-all watcher-status watcher-kill clean clean-model chain-add chain-remove chain-stop chain-start chain-show monitor pip-clean pip-setup pip-reset unload-aliases install-aliases uninstall-aliases"

# Mostra i comandi disponibili
diego() {
    echo "Comandi disponibili:"
    echo ""
    echo "── Job management ──"
    echo "   myjobs            — lista job attivi"
    echo "   jobinfo <ID>      — dettagli job"
    echo "   killjob <ID>      — cancella job"
    echo "   killalljobs       — cancella tutti i miei job + watcher"
    echo ""
    echo "── Log monitoring ──"
    echo "   trainlog <ID> — segui log training"
    echo "   evallog <ID>  — segui log eval"
    echo "   lastlog [N]   — segui l'ultimo log (N=ultime N righe)"
    echo ""
    echo "── Training & eval ──"
    echo "   train [--config PATH] [extra args...]"
    echo "                     — lancia training (default: experiments/configs/t2g/grpo_qwen05.yaml)"
    echo "   run-eval [--config PATH] [--checkpoint PATH]"
    echo "                     — lancia evaluation"
    echo "   run-all [--resume]"
    echo "                     — lancia pipeline train+eval"
    echo ""
    echo "── Pipeline ──"
    echo "   chain-show   — mostra stato pipeline + job in coda"
    echo "   chain-add    — aggiungi job alla pipeline attiva"
    echo "   chain-remove --models=1"
    echo "                    — rimuovi job dalla coda"
    echo "   chain-stop   — ferma pipeline (preserva stato)"
    echo "   chain-start  — riprendi pipeline dopo chain-stop"
    echo "   watcher-status — controlla se il watcher è attivo"
    echo "   watcher-kill — uccidi il watcher"
    echo ""
    echo "── Monitor ──"
    echo "   monitor [--poll N] [--tab] [--samples [N]] [--metrics] [--all [N]]"
    echo "                    — monitor live della pipeline"
    echo ""
    echo "── Utilità ──"
    echo "   proj         — cd al progetto"
    echo "   ckpts        — mostra checkpoint"
    echo "   gpu          — stato GPU"
    echo "   quota            — uso disco progetto"
    echo "   clean        — pulizia workspace"
    echo "   clean-model <TAG> [--all]"
    echo "                    — pulisci checkpoints/logs di un modello"
    echo ""
    echo "── Pip / Environment ──"
    echo "   pip-clean    — rimuovi pacchetti pip --user"
    echo "   pip-setup    — (re)installa dipendenze"
    echo "   pip-reset    — pip-clean + pip-setup"
    echo ""
    echo "── Meta ──"
    echo "   diego         — mostra questo messaggio"
    echo "   unload-aliases   — rimuovi alias (sessione corrente)"
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
}

# Rimuovi alias dal .bashrc
uninstall-aliases() {
    if grep -qF "$_ALIASES_SOURCE_LINE" ~/.bashrc 2>/dev/null; then
        sed -i "\|$_ALIASES_SOURCE_LINE|d" ~/.bashrc
        echo "✅ Alias rimossi da ~/.bashrc"
    else
        echo "⚠️  Alias non presenti in ~/.bashrc"
    fi
    unload-aliases
}

echo "✅ Alias caricati. Digita 'diego' per la lista comandi."
