#!/bin/bash
# ============================================================================
# Alias utili per il cluster DMI — progetto neuro_symbolic_t2g
#
# Uso:
#   source src/cluster/aliases.sh
#
# Per caricarli automaticamente, aggiungi al tuo ~/.bashrc:
#   source ~/neuro_symbolic_t2g/src/cluster/aliases.sh
# ============================================================================

PROJ_DIR="$HOME/neuro_symbolic_t2g"

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
alias killalljobs='t2g-watcher-kill && scancel --me'

# ── Log monitoring ───────────────────────────────────────────────────────────

# Segui il log di un job di training (uso: t2g-trainlog <JOB_ID>)
t2g-trainlog() {
    if [ -z "$1" ]; then
        echo "Uso: t2g-trainlog <JOB_ID>"
        return 1
    fi
    local logfile="$PROJ_DIR/logs/slurm-train-${1}.log"
    if [ ! -f "$logfile" ]; then
        echo "Log non trovato: $logfile"
        return 1
    fi
    tail -f "$logfile"
}

# Segui il log di un job di eval (uso: t2g-evallog <JOB_ID>)
t2g-evallog() {
    if [ -z "$1" ]; then
        echo "Uso: t2g-evallog <JOB_ID>"
        return 1
    fi
    local logfile="$PROJ_DIR/logs/slurm-eval-${1}.log"
    if [ ! -f "$logfile" ]; then
        echo "Log non trovato: $logfile"
        return 1
    fi
    tail -f "$logfile"
}

# Mostra l'ultimo log — uso: t2g-lastlog [N_RIGHE]
t2g-lastlog() {
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
t2g-gpu() {
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
alias t2g-proj='cd "$PROJ_DIR"'

# Mostra i checkpoint disponibili
t2g-ckpts() {
    local base="$PROJ_DIR/checkpoints"
    if [ ! -d "$base" ]; then
        echo "Nessun checkpoint trovato."
        return 0
    fi
    echo "──── Checkpoints ────"
    for d in "$base"/*/; do
        [ -d "$d" ] || continue
        echo "  $(basename "$d"):"
        ls -d "$d"checkpoint-* 2>/dev/null | while read -r c2; do echo "    $(basename "$c2")"; done
    done
}

# Lancia training (uso: t2g-train [--config PATH] [extra args...])
t2g-train() {
    local config="config/grpo_t2g_qwen05.yaml"
    local extra_args=""
    while [ $# -gt 0 ]; do
        case "$1" in
            --config) config="$2"; shift 2 ;;
            *) extra_args="$extra_args $1"; shift ;;
        esac
    done
    cd "$PROJ_DIR" && CONFIG="$config" EXTRA_ARGS="$extra_args" sbatch src/cluster/train.sh
}

# Lancia eval (uso: t2g-eval [--config PATH] [--checkpoint PATH])
t2g-eval() {
    local config="config/grpo_t2g_qwen05.yaml"
    local checkpoint=""
    while [ $# -gt 0 ]; do
        case "$1" in
            --config) config="$2"; shift 2 ;;
            --checkpoint) checkpoint="$2"; shift 2 ;;
            *) echo "❌ Argomento sconosciuto: $1"; return 1 ;;
        esac
    done
    cd "$PROJ_DIR" && CONFIG="$config" CHECKPOINT="$checkpoint" sbatch src/cluster/eval.sh
}

# Lancia train + eval (uso: t2g-run-all)
t2g-run-all() {
    cd "$PROJ_DIR" && bash src/cluster/run_all.sh "$@"
}

# Controlla se il watcher è attivo
t2g-watcher-status() {
    if [ -f "$PROJ_DIR/.chain_failed" ]; then
        local failed=$(cat "$PROJ_DIR/.chain_failed")
        echo "❌ Pipeline FALLITA — job: $failed"
        echo "   Per riprendere: t2g-run-all --resume"
        return 1
    fi
    if [ -f "$PROJ_DIR/.chain_pid" ]; then
        local pid=$(cat "$PROJ_DIR/.chain_pid")
        if ps -p "$pid" > /dev/null 2>&1; then
            echo "✅ Watcher attivo (PID $pid)"
        else
            echo "❌ Watcher morto (PID $pid non trovato)"
            rm -f "$PROJ_DIR/.chain_pid"
            return 1
        fi
    else
        echo "❌ Nessun watcher attivo"
        return 1
    fi
}

# Uccidi il watcher
t2g-watcher-kill() {
    if [ -f "$PROJ_DIR/.chain_pid" ]; then
        local pid=$(cat "$PROJ_DIR/.chain_pid")
        if ! ps -p "$pid" > /dev/null 2>&1; then
            echo "⚠️  Watcher (PID $pid) già morto"
            rm -f "$PROJ_DIR/.chain_pid"
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
                rm -f "$PROJ_DIR/.chain_pid"
                ;;
            *)
                echo "Annullato."
                ;;
        esac
    else
        echo "Nessun watcher attivo"
    fi
}

# Pulizia workspace (uso: t2g-clean [--force])
t2g-clean() {
    cd "$PROJ_DIR" && bash src/cluster/clean.sh "$@"
}

# Aggiungi job alla pipeline attiva (uso: t2g-chain-add)
t2g-chain-add() {
    cd "$PROJ_DIR" && bash src/cluster/run_all.sh --append "$@"
}

# Rimuovi job dalla pipeline attiva (uso: t2g-chain-remove --models=1)
t2g-chain-remove() {
    cd "$PROJ_DIR" && bash src/cluster/run_all.sh --remove "$@"
}

# Ferma la pipeline senza perdere lo stato (uso: t2g-chain-stop [--force])
t2g-chain-stop() {
    local force=0
    for arg in "$@"; do
        case "$arg" in
            --force) force=1 ;;
            --help|-h)
                echo "Uso: t2g-chain-stop [--force]"
                echo "  (default)   Ferma pipeline. Cancella job SLURM attivo + watcher."
                echo "              Salva lo stato per poter fare t2g-chain-start."
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

    if [ -f .chain_pid ]; then
        local pid=$(cat .chain_pid)
        kill "$pid" 2>/dev/null && echo "✅ Watcher (PID $pid) terminato"
        rm -f .chain_pid
    fi

    if [ "$force" -eq 1 ]; then
        rm -f .job_chain .chain_pid .chain_failed .chain_stopped .monitor_cache
        echo "🗑️  File di stato cancellati"
        echo "Pipeline terminata definitivamente. Per ricominciare: t2g-run-all"
    else
        local stopped_type=$(echo "$active_name" | cut -d- -f1)
        local stopped_tag=$(echo "$active_name" | cut -d- -f2-)
        echo "${stopped_type}:config/grpo_t2g_qwen05.yaml:${stopped_tag}:0:${active_job}" > .chain_stopped
        rm -f .chain_failed
        echo "Pipeline fermata. Per riprendere: t2g-chain-start"
    fi
}

# Riprendi la pipeline dopo chain-stop (uso: t2g-chain-start)
t2g-chain-start() {
    cd "$PROJ_DIR"

    if [ ! -f .chain_stopped ]; then
        echo "❌ Nessun .chain_stopped trovato."
        echo "   t2g-chain-start funziona solo dopo t2g-chain-stop (senza --force)."
        return 1
    fi

    local stopped_info=$(cat .chain_stopped)
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
        [ -f .job_chain ] && [ -s .job_chain ] && next_in_chain=$(head -1 .job_chain)
        local next_type=$(echo "$next_in_chain" | cut -d: -f1)
        local next_tag=$(echo "$next_in_chain" | cut -d: -f3)
        if [ "$next_type" != "eval" ] || [ "$next_tag" != "$stopped_tag" ]; then
            echo "eval:${stopped_cfg}:${stopped_tag}" >> "$RESUME_CHAIN"
        fi
        [ -f .job_chain ] && [ -s .job_chain ] && cat .job_chain >> "$RESUME_CHAIN"
        mv "$RESUME_CHAIN" .job_chain
        echo "→ Training $stopped_tag verrà ripreso dall'ultimo checkpoint"
    elif [ "$stopped_type" = "eval" ]; then
        local RESUME_CHAIN=$(mktemp)
        if [ "$stopped_stage" -gt 0 ]; then
            echo "eval:${stopped_cfg}:${stopped_tag}:--skip-stages=${stopped_stage}" > "$RESUME_CHAIN"
        else
            echo "eval:${stopped_cfg}:${stopped_tag}" > "$RESUME_CHAIN"
        fi
        [ -f .job_chain ] && [ -s .job_chain ] && cat .job_chain >> "$RESUME_CHAIN"
        mv "$RESUME_CHAIN" .job_chain
        echo "→ Eval $stopped_tag verrà rieseguito"
    fi

    rm -f .chain_stopped .chain_failed

    if [ ! -f .job_chain ] || [ ! -s .job_chain ]; then
        echo "⚠️  Nessun job rimanente nella catena."
        return 0
    fi

    local TOTAL=$(wc -l < .job_chain)
    echo "Catena ($TOTAL job):"
    cat -n .job_chain

    if [ -f .chain_pid ]; then
        local OLD_PID=$(cat .chain_pid)
        kill "$OLD_PID" 2>/dev/null
        rm -f .chain_pid
    fi

    mkdir -p logs
    nohup bash src/cluster/chain_next.sh >> logs/chain_watcher.log 2>&1 &
    local WATCHER_PID=$!
    echo "$WATCHER_PID" > .chain_pid
    echo "Pipeline ripresa! Watcher PID: $WATCHER_PID"
}

# Mostra la catena di job attuale (uso: t2g-chain-show)
t2g-chain-show() {
    t2g-watcher-status
    echo ""
    if [ -f "$PROJ_DIR/.chain_stopped" ]; then
        local info=$(cat "$PROJ_DIR/.chain_stopped")
        local st_type=$(echo "$info" | cut -d: -f1)
        local st_tag=$(echo "$info" | cut -d: -f3)
        [ "$st_type" != "none" ] && echo "⏸️  Pipeline fermata su: $st_type $st_tag"
        echo "   Per riprendere: t2g-chain-start"
        echo ""
    fi
    if [ ! -f "$PROJ_DIR/.job_chain" ] || [ ! -s "$PROJ_DIR/.job_chain" ]; then
        echo "Nessun job in coda."
        return 0
    fi
    local total=$(wc -l < "$PROJ_DIR/.job_chain")
    echo "Job in coda ($total):"
    cat -n "$PROJ_DIR/.job_chain"
}

# Monitor live della pipeline (uso: t2g-monitor [--poll N])
t2g-monitor() {
    cd "$PROJ_DIR" && python3 -u -m src.utils.chain_monitor "$@"
}

# ── Pip / Environment ────────────────────────────────────────────────────────

# Pulisci tutti i pacchetti --user
t2g-pip-clean() {
    echo "🗑️  Rimozione pacchetti pip --user..."
    rm -rf ~/.local/lib/python3.*/site-packages/*
    rm -rf ~/.local/bin/*
    echo "✅ ~/.local ripulito"
}

# (Re)installa dipendenze
t2g-pip-setup() {
    echo "📦 Installazione dipendenze..."
    cd "$PROJ_DIR" && bash src/cluster/setup.sh
}

# Pulisci e reinstalla da zero
t2g-pip-reset() {
    t2g-pip-clean
    t2g-pip-setup
}

# ── Meta ─────────────────────────────────────────────────────────────────────

_T2G_ALIASES="myjobs jobinfo killjob killalljobs t2g-trainlog t2g-evallog t2g-lastlog tree t2g-gpu quota t2g-proj t2g-ckpts t2g-train t2g-eval t2g-run-all t2g-watcher-status t2g-watcher-kill t2g-clean t2g-chain-add t2g-chain-remove t2g-chain-stop t2g-chain-start t2g-chain-show t2g-monitor t2g-pip-clean t2g-pip-setup t2g-pip-reset t2g-unload-aliases t2g-install-aliases t2g-uninstall-aliases"

# Mostra i comandi disponibili
t2g-help() {
    echo "Comandi T2G disponibili:"
    echo ""
    echo "── Job management ──"
    echo "   myjobs            — lista job attivi"
    echo "   jobinfo <ID>      — dettagli job"
    echo "   killjob <ID>      — cancella job"
    echo "   killalljobs       — cancella tutti i miei job + watcher"
    echo ""
    echo "── Log monitoring ──"
    echo "   t2g-trainlog <ID> — segui log training"
    echo "   t2g-evallog <ID>  — segui log eval"
    echo "   t2g-lastlog [N]   — segui l'ultimo log (N=ultime N righe)"
    echo ""
    echo "── Training & eval ──"
    echo "   t2g-train [--config PATH] [extra args...]"
    echo "                     — lancia training (default: config/grpo_t2g_qwen05.yaml)"
    echo "   t2g-eval [--config PATH] [--checkpoint PATH]"
    echo "                     — lancia evaluation"
    echo "   t2g-run-all [--resume]"
    echo "                     — lancia pipeline train+eval"
    echo ""
    echo "── Pipeline ──"
    echo "   t2g-chain-show   — mostra stato pipeline + job in coda"
    echo "   t2g-chain-add    — aggiungi job alla pipeline attiva"
    echo "   t2g-chain-remove --models=1"
    echo "                    — rimuovi job dalla coda"
    echo "   t2g-chain-stop   — ferma pipeline (preserva stato)"
    echo "   t2g-chain-start  — riprendi pipeline dopo chain-stop"
    echo "   t2g-watcher-status — controlla se il watcher è attivo"
    echo "   t2g-watcher-kill — uccidi il watcher"
    echo ""
    echo "── Monitor ──"
    echo "   t2g-monitor [--poll N] [--tab] [--samples [N]] [--metrics] [--all [N]]"
    echo "                    — monitor live della pipeline"
    echo ""
    echo "── Utilità ──"
    echo "   t2g-proj         — cd al progetto"
    echo "   t2g-ckpts        — mostra checkpoint"
    echo "   t2g-gpu          — stato GPU"
    echo "   quota            — uso disco progetto"
    echo "   t2g-clean        — pulizia workspace"
    echo ""
    echo "── Pip / Environment ──"
    echo "   t2g-pip-clean    — rimuovi pacchetti pip --user"
    echo "   t2g-pip-setup    — (re)installa dipendenze"
    echo "   t2g-pip-reset    — pip-clean + pip-setup"
    echo ""
    echo "── Meta ──"
    echo "   t2g-help         — mostra questo messaggio"
    echo "   t2g-unload-aliases   — rimuovi alias (sessione corrente)"
    echo "   t2g-install-aliases  — aggiungi alias al .bashrc (permanente)"
    echo "   t2g-uninstall-aliases — rimuovi alias dal .bashrc"
}

# Rimuovi tutti gli alias e funzioni custom (solo sessione corrente)
t2g-unload-aliases() {
    for cmd in $_T2G_ALIASES; do
        unalias "$cmd" 2>/dev/null
        unset -f "$cmd" 2>/dev/null
    done
    unset _T2G_ALIASES PROJ_DIR
    echo "✅ Alias T2G rimossi (sessione corrente)."
}

_ALIASES_SOURCE_LINE="source ~/neuro_symbolic_t2g/src/cluster/aliases.sh"

# Aggiungi alias al .bashrc
t2g-install-aliases() {
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
t2g-uninstall-aliases() {
    if grep -qF "$_ALIASES_SOURCE_LINE" ~/.bashrc 2>/dev/null; then
        sed -i "\|$_ALIASES_SOURCE_LINE|d" ~/.bashrc
        echo "✅ Alias rimossi da ~/.bashrc"
    else
        echo "⚠️  Alias non presenti in ~/.bashrc"
    fi
    t2g-unload-aliases
}

echo "✅ Alias T2G caricati. Digita 't2g-help' per la lista comandi."
