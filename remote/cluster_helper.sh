#!/bin/bash
# ============================================================================
# cluster_helper.sh — Snapshot di stato + mutazioni per la catena T2G.
#
# Deployato sul login node di gcluster (NON c'e' python lì: solo shell) in
#   ~/neuro_symbolic_t2g/cluster/cluster_helper.sh
# e invocato dal driver esterno (remote/app.py su Render) con UNA SOLA
# connessione ssh per tick. Sostituisce i ~8 comandi separati (squeue, cat,
# wc, test, ...) che il servizio dovrebbe altrimenti eseguire uno a uno.
#
# PROTOCOLLO DI OUTPUT (machine-readable, niente python → key=value per riga):
#   STATUS_OK=1            → snapshot valido prodotto
#   ACTIVE_JOB=<id>|<name>|<state>   (vuoto se nessun job attivo)
#   QUEUE=<e1>|<e2>|...    coda: entry separate da \x1f (unit separator,
#                          carattere che non compare mai nelle entry)
#                          su UNA sola riga; ogni entry è
#                          "type:cfg:tag[:extra]" (formato job_chain)
#   QUEUE_COUNT=<n>        numero di entry in coda
#   LAST_JOB=<id>:<type>:<cfg>:<tag>:<retries>   (vuoto se nessuno)
#   STOPPED=0|1            chain_stopped presente → pausa
#   ERRORS_COUNT=<n>       righe totali di chain_errors (JSONL)
#   ERRORS_TAIL=[...]      ultime 5 righe RAW di chain_errors come array JSON
#                          (su una riga, escaping manuale con sed)
#
# SUBCOMANDI:
#   status                     (default) stampa lo snapshot completo
#   monitor [nlines]           snapshot + LOG_TAIL_B64 (base64 delle ultime
#                              nlines righe del log del job ATTIVO, default 200;
#                              vuoto se nessun job attivo o log assente)
#   enqueue <entry>            appende una entry "type:cfg:tag[:extra]" alla coda
#   rewrite_queue <content>    rimpiazza la coda (entry separate da \x1f;
#                              stringa vuota = svuota la coda)
#   pause                      crea .chain_state/chain_stopped (stop soft)
#   resume                     rimuove .chain_state/chain_stopped
#   tick                       esegue chain_tick.sh --quiet poi lo snapshot
#   scancel                    cancella il job SLURM attivo (exit 1 se nessuno)
#
# Dopo ogni mutazione (enqueue/rewrite_queue/pause/resume/tick) il helper
# stampa COMUNQUE lo snapshot fresco: così il driver fa 1 sola connessione
# e riceve stato + esito insieme.
#
# Exit codes: 0 ok · 2 usage · 3 chain_tick.sh fallito/mancante.
# ============================================================================

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=cluster/_lib.sh
source "$SCRIPT_DIR/_lib.sh"

# ── Snapshot ─────────────────────────────────────────────────────────────────
dump_status() {
    local active="" last="" queue="" qcount=0 stopped=0
    local errors_count=0 errors_tail="[]"
    local aid aname astate sep="" e out="" first=1

    # Job SLURM attivo (la QoS consente max 1): id|name|state
    aid=$(active_job_id)
    aname=$(active_job_name)
    astate=$(squeue --me -h -o '%T' 2>/dev/null | head -1 | tr -d '[:space:]')
    [ -n "$aid" ] && active="${aid}|${aname}|${astate}"

    # Coda: separatore \x1f (mai usato nelle entry) → una sola riga.
    if [ -s "$CHAIN_FILE" ]; then
        qcount=$(wc -l < "$CHAIN_FILE")
        sep=$(printf '\x1f')
        queue=$(paste -sd "$sep" "$CHAIN_FILE")
    fi

    [ -f "$LAST_JOB_FILE" ] && last=$(cat "$LAST_JOB_FILE")
    [ -f "$STOPPED_FILE" ] && stopped=1

    # Errori (JSONL): totale righe + ultime 5 righe raw come array JSON.
    # L'escaping manuale (backslash e doppi apici) basta per il JSON in uscita:
    # le righe sono già JSON valido prodotto da _lib.sh::log_job_error.
    if [ -f "$ERRORS_FILE" ]; then
        errors_count=$(wc -l < "$ERRORS_FILE")
        out=""
        while IFS= read -r e; do
            [ -n "$e" ] || continue
            # JSON-escape: backslash, doppi apici e TAB (un tab raw in una
            # stringa JSON è invalido e farebbe scartare l'intera ERRORS_TAIL).
            e=$(printf '%s' "$e" | sed -e 's/\\/\\\\/g' -e 's/"/\\"/g' -e 's/\t/\\t/g')
            if [ "$first" -eq 1 ]; then out="\"$e\""; first=0; else out="$out,\"$e\""; fi
        done < <(tail -n 5 "$ERRORS_FILE")
        errors_tail="[$out]"
    fi

    printf 'STATUS_OK=1\n'
    printf 'ACTIVE_JOB=%s\n' "$active"
    printf 'QUEUE=%s\n' "$queue"
    printf 'QUEUE_COUNT=%s\n' "$qcount"
    printf 'LAST_JOB=%s\n' "$last"
    printf 'STOPPED=%s\n' "$stopped"
    printf 'ERRORS_COUNT=%s\n' "$errors_count"
    printf 'ERRORS_TAIL=%s\n' "$errors_tail"
}

# ── Monitor: snapshot + log tail del job attivo (base64, single line) ───────
# Il log del job attivo: il nome SLURM è train-<tag>/eval-<tag> (da
# chain_submit_next in _lib.sh), lo script usa #SBATCH --output=logs/slurm-<x>-%j.log
# con x=train/eval → logs/slurm-{train,eval}-<JOBID>.log (stessa convenzione
# di chain_monitor.py::_find_log_file). Base64 evita qualunque problema di
# escaping multi-riga nel protocollo KEY=VALUE.
# LIVE_STATUS: contenuto grezzo di logs/live_status.json (una riga JSON già
# pronta) se il file esiste ed è FRESCO (modificato negli ultimi 10 minuti —
# oltre, il job che lo scriveva è morto e lo status è stale).
dump_monitor() {
    local nlines="${1:-200}"
    dump_status
    local aid aname prefix logpath b64=""
    aid=$(active_job_id)
    if [ -n "$aid" ]; then
        aname=$(active_job_name)
        case "$aname" in
            eval-*) prefix="eval" ;;
            *)      prefix="train" ;;
        esac
        logpath="$PROJ_DIR/logs/slurm-${prefix}-${aid}.log"
        if [ -f "$logpath" ]; then
            b64=$(tail -n "$nlines" "$logpath" 2>/dev/null | base64 -w 0)
        fi
    fi
    printf 'LOG_PATH=%s\n' "${logpath:-}"
    printf 'LOG_TAIL_B64=%s\n' "$b64"
    local live="$PROJ_DIR/logs/live_status.json"
    if [ -f "$live" ] && [ -z "$(find "$live" -mmin +10 2>/dev/null)" ]; then
        # Single-line JSON: echo strips the trailing newline, safe on the
        # KEY=VALUE protocol (no newlines inside).
        printf 'LIVE_STATUS=%s\n' "$(cat "$live")"
    fi
}

# ── Mutazioni ────────────────────────────────────────────────────────────────
enqueue() {
    local entry="$1"
    mkdir -p "$STATE_DIR"
    printf '%s\n' "$entry" >> "$CHAIN_FILE"
    log_line "enqueue (driver esterno): $entry"
    echo "OK_ENQUEUE=1"
}

rewrite_queue() {
    local content="$1" sep active_id
    sep=$(printf '\x1f')
    mkdir -p "$STATE_DIR"
    active_id=$(active_job_id)
    if [ -z "$active_id" ]; then
        rm -f "$LAST_JOB_FILE"
        log_line "rewrite_queue (driver esterno): last_job stale rimosso (nessun job SLURM attivo)"
    else
        log_line "rewrite_queue (driver esterno): last_job preservato (job SLURM attivo: $active_id)"
    fi
    if [ -z "$content" ]; then
        rm -f "$CHAIN_FILE"
    else
        # Le entry arrivate separate da \x1f → una per riga.
        # La newline FINALE è obbligatoria: senza, `wc -l`/chain_remaining
        # sottostimano la coda e un successivo enqueue fonderebbe l'ultima
        # entry con la nuova su una sola riga (corruzione della coda).
        printf '%s\n' "$content" | tr "$sep" '\n' > "$CHAIN_FILE"
    fi
    log_line "rewrite_queue (driver esterno): $(chain_remaining) entry"
    echo "OK_REWRITE=1"
}

pause() {
    mkdir -p "$STATE_DIR"
    touch "$STOPPED_FILE"
    log_line "pause (driver esterno): chain_stopped creato"
    echo "OK_PAUSE=1"
}

resume() {
    rm -f "$STOPPED_FILE"
    log_line "resume (driver esterno): chain_stopped rimosso"
    echo "OK_RESUME=1"
}

tick() {
    if [ ! -f "$SCRIPT_DIR/chain_tick.sh" ]; then
        echo "ERR_TICK=chain_tick.sh mancante" >&2
        exit 3
    fi
    local rc=0
    bash "$SCRIPT_DIR/chain_tick.sh" --quiet || rc=$?
    if [ "$rc" -eq 4 ]; then
        # Errore interno soft del tick (chain_tick ERR trap): già loggato in
        # chain.log, il prossimo tick (~5 min) riprova. Nessun allarme: un
        # blip Slurm/DNS NON deve produrre 502/notify rosse nel driver.
        echo "OK_TICK=1"
        return 0
    fi
    if [ "$rc" -ne 0 ]; then
        echo "ERR_TICK=chain_tick rc=$rc" >&2
        exit 3
    fi
    echo "OK_TICK=1"
}

# Kill del job attivo (per la TUI). Exit 1 + messaggio se nessun job attivo:
# in quel caso non viene stampato STATUS_OK e il driver risolve in 409.
scancel_active() {
    local aid
    aid=$(active_job_id)
    if [ -z "$aid" ]; then
        echo "ERR_NO_ACTIVE_JOB=1" >&2
        exit 1
    fi
    if ! scancel "$aid" 2>/dev/null; then
        echo "ERR_SCANCEL_FAILED=$aid" >&2
        exit 1
    fi
    log_line "scancel (driver esterno): job $aid cancellato"
    echo "OK_SCANCEL=$aid"
}

usage() {
    echo "Uso: bash cluster/cluster_helper.sh [status|monitor [n]|enqueue <entry>|rewrite_queue <content>|pause|resume|tick|scancel]" >&2
    exit 2
}

# ── Dispatch ─────────────────────────────────────────────────────────────────
CMD="${1:-status}"
case "$CMD" in
    status)          dump_status ;;
    monitor)         dump_monitor "${2:-200}" ;;
    enqueue)         [ $# -ge 2 ] || usage; enqueue "$2" ;;
    rewrite_queue)   [ $# -ge 2 ] || usage; rewrite_queue "$2" ;;
    pause)           pause ;;
    resume)          resume ;;
    tick)            tick ;;
    scancel)         scancel_active ;;
    -h|--help|help)  usage ;;
    *)               usage ;;
esac

# Dopo una mutazione il driver riceve subito lo snapshot fresco (1 sola ssh).
# scancel NON ristampa lo snapshot: il job resta RUNNING qualche secondo prima
# di passare a CANCELLED — il driver farà lo status al prossimo tick.
case "$CMD" in
    enqueue|rewrite_queue|pause|resume|tick) dump_status ;;
esac
