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
#   enqueue_batch <content>    appende PIU' entry (separate da \x1f) poi snapshot:
#                              N job con 1 sola connessione invece di N
#   start_batch <content>      enqueue_batch + tick + snapshot MONITOR completo
#                              (log tail incluso): POST /jobs/start e /jobs/batch
#                              fanno enqueue+tick+monitor con 1 SOLA ssh
#                              (prima: N+2 ssh seriali, fino a ~90s)
#   rewrite_queue <content>    rimpiazza la coda (entry separate da \x1f;
#                              stringa vuota = svuota la coda)
#   pause                      crea .chain_state/chain_stopped (stop soft)
#   resume                     rimuove .chain_state/chain_stopped
#   tick                       esegue chain_tick.sh --quiet poi lo snapshot
#   timeseries <tag>           serie delle righe KV `step=N loss=...` dell'INTERO
#                              log del job (attivo, altrimenti ultimo) col tag
#                              dato: TS_* keys, log in base64 (per i grafici)
#   results <config>           gli eval_*.json (preferito eval_final.json) di
#                              ogni run_* della dir risultati del config, in
#                              base64 (RUN_ID_n/RUN_B64_n); token VUOTO = elenco
#                              delle dir disponibili (RESULTS_DIRS)
#   scancel                    cancella il job SLURM attivo (exit 1 se nessuno)
#                              e stampa poi lo snapshot monitor (1 sola ssh)
#
# Dopo ogni mutazione (enqueue/rewrite_queue/enqueue_batch/pause/resume/tick)
# il helper stampa COMUNQUE lo snapshot fresco: così il driver fa 1 sola
# connessione e riceve stato + esito insieme. start_batch e scancel stampano
# da soli lo snapshot monitor completo (LOG_TAIL_B64 incluso).
#
# Exit codes: 0 ok · 2 usage · 3 chain_tick.sh fallito/mancante (il tick
# dentro start_batch NON interrompe l'output: lo snapshot viene stampato
# comunque e l'exit 3 arriva alla fine, così il driver distingue "enqueue
# riuscito + tick fallito" e mostra lo stato reale della coda).
# ============================================================================

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=cluster/_lib.sh
source "$SCRIPT_DIR/_lib.sh"

# ── Snapshot ─────────────────────────────────────────────────────────────────
dump_status() {
    local active="" last="" queue="" qcount=0 stopped=0
    local errors_count=0 errors_tail="[]"
    # aid/aname/astate SEMPRE inizializzate: con set -u un reference a una
    # local solo dichiarata è errore fatale (succedeva su blip di squeue,
    # quando slurm resta vuoto e aid non viene mai assegnata).
    local aid="" aname="" astate="" slurm sep="" e out="" first=1

    # Job SLURM attivo (la QoS consente max 1): id|name|state
    # Una sola query squeue invece di tre: evita snapshot incoerenti (il job
    # puo' cambiare stato tra le chiamate) e riduce il carico sullo scheduler.
    # Un blip di squeue lascia semplicemente ACTIVE_JOB vuoto senza abbattere
    # l'helper sotto set -e.
    slurm=$(squeue --me -h -o '%A|%j|%T' 2>/dev/null | head -1) || true
    if [ -n "$slurm" ]; then
        aid=$(printf '%s' "$slurm" | cut -d'|' -f1)
        aname=$(printf '%s' "$slurm" | cut -d'|' -f2)
        astate=$(printf '%s' "$slurm" | cut -d'|' -f3 | tr -d '[:space:]')
    fi
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
    local aid="" aname="" prefix="" logpath="" b64=""
    # `|| true`: un blip di squeue NON deve abbattere l'helper (set -e) —
    # si limita a non stampare il log tail.
    aid=$(active_job_id) || true
    if [ -n "$aid" ]; then
        aname=$(active_job_name) || true
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
    local content="$1" sep
    sep=$(printf '\x1f')
    mkdir -p "$STATE_DIR"
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

# Appende PIU' entry (separate da \x1f, come rewrite_queue) SENZA rimpiazzare
# la coda esistente: un batch di N job costa 1 connessione invece di N.
enqueue_many() {
    local content="$1" sep entry n=0
    sep=$(printf '\x1f')
    mkdir -p "$STATE_DIR"
    # La newline finale da printf è obbligatoria: senza, `read` perderà
    # l'ultima entry (classico pitfall del while read senza newline finale).
    while IFS= read -r entry; do
        [ -n "$entry" ] || continue
        printf '%s\n' "$entry" >> "$CHAIN_FILE"
        n=$((n + 1))
    done < <(printf '%s\n' "$content" | tr "$sep" '\n')
    log_line "enqueue_batch (driver esterno): $n entry"
    echo "OK_ENQUEUE_N=$n"
}

enqueue_batch() {
    enqueue_many "$1"
}

# /jobs/start e /jobs/batch in UNA sola connessione: appende le entry, esegue
# il tick e stampa lo snapshot monitor completo (dump_monitor include già
# dump_status). Se il tick fallisce lo snapshot viene stampato COMUNQUE e
# l'exit 3 arriva alla fine: il driver distingue "enqueue riuscito + tick
# fallito" e mostra lo stato reale della coda invece di un 502 opaco.
start_batch() {
    local content="$1" rc=0
    enqueue_many "$content"
    _tick_run || rc=$?
    dump_monitor
    return "$rc"
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

# Core del tick SENZA exit: ritorna il rc così start_batch può continuare
# (stampare lo snapshot) anche quando il tick fallisce, e il driver riceve
# comunque lo stato della coda aggiornato con l'enqueue.
_tick_run() {
    if [ ! -f "$SCRIPT_DIR/chain_tick.sh" ]; then
        echo "ERR_TICK=chain_tick.sh mancante" >&2
        return 3
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
        return 3
    fi
    echo "OK_TICK=1"
    return 0
}

# Il tick stampa lo snapshot DA SOLO, prima di uscire: `exit "$rc"` salta il
# dispatch finale in coda al file, quindi delegargli il dump_status lasciava
# l'output del solo `OK_TICK=1` senza STATUS_OK e il driver lo interpretava
# come violazione di protocollo (tick riuscito segnalato come errore).
tick() {
    local rc=0
    _tick_run || rc=$?
    dump_status
    exit "$rc"
}

# Kill del job attivo (per la TUI). Exit 1 + messaggio se nessun job attivo:
# in quel caso non viene stampato STATUS_OK e il driver risolve in 409.
# Dopo lo scancel stampa lo snapshot monitor: il driver fa 1 sola ssh invece
# di due (kill + monitor separati). Il job può risultare ancora RUNNING per
# qualche secondo prima di passare a CANCELLED — è normale e dichiarato.
scancel_active() {
    local aid
    aid=$(active_job_id) || true
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
    dump_monitor
}

# ── Timeseries: righe KV `step=N ...` dell'intero log del job col tag dato ───
# Il driver le trasforma in serie per i grafici (sparkline della TUI). Il
# grep gira sul login node: trasportare SOLO le righe KV (grep) invece del
# log intero mantiene il payload piccolo; base64 evita problemi di escaping
# multi-riga nel protocollo KEY=VALUE. Il job viene risolto prima nello
# squeue (job attivo) e poi in last_job (ultimo sottomesso): i log SLURM sono
# nominati per JOBID, quindi il mapping tag→jobid esiste solo lì.
timeseries() {
    local tag="$1" aid="" aname="" jtype="" logpath="" total="" b64=""
    local lj_id lj_type lj_cfg lj_tag rest
    # `|| true`: blip di squeue → job "non attivo", non crash dell'helper.
    aid=$(active_job_id) || true
    if [ -n "$aid" ]; then
        aname=$(active_job_name) || true
        case "$aname" in
            "train-$tag") jtype="train" ;;
            "eval-$tag")  jtype="eval" ;;
        esac
    fi
    if [ -z "$jtype" ] && [ -f "$LAST_JOB_FILE" ]; then
        # last_job = "id:type:cfg:tag:retries[:extra]": la cfg non contiene
        # ':', quindi il tag è sempre il 4° campo.
        IFS=':' read -r lj_id lj_type lj_cfg lj_tag rest < "$LAST_JOB_FILE"
        if [ "$lj_tag" = "$tag" ]; then
            jtype="$lj_type"
            aid="$lj_id"
        fi
    fi
    if [ -n "$jtype" ]; then
        logpath="$PROJ_DIR/logs/slurm-${jtype}-${aid}.log"
        if [ -f "$logpath" ]; then
            # Stesso pattern di _KV_STEP in chain_monitor (righe KV con step+loss):
            # una sola fonte di verità per cosa è una "riga metrica". Il `|| true`
            # protegge da set -o pipefail quando grep non trova nulla (log eval).
            b64=$(grep -E '^[[:space:]]+step=[0-9]+[[:space:]]+loss=' "$logpath" 2>/dev/null | base64 -w 0) || true
            # total_steps: ultimo marker di stage ([stage N] steps=M, scritto
            # dal curriculum) oppure max_steps= (fallback SFT/base).
            total=$(grep -oE '\[stage [0-9]+\] steps=[0-9]+' "$logpath" 2>/dev/null | tail -1 | grep -oE '[0-9]+$') || true
            if [ -z "$total" ]; then
                total=$(grep -oE 'max_steps=[0-9]+' "$logpath" 2>/dev/null | tail -1 | grep -oE '[0-9]+$') || true
            fi
        fi
    fi
    printf 'TS_MATCH=%s\n' "$([ -n "$jtype" ] && echo 1 || echo 0)"
    printf 'TS_JOB_ID=%s\n' "$aid"
    printf 'TS_JOB_TYPE=%s\n' "$jtype"
    printf 'TS_LOG_PATH=%s\n' "$logpath"
    printf 'TS_TOTAL_STEPS=%s\n' "$total"
    printf 'TS_LOG_B64=%s\n' "$b64"
}

# ── Results: eval_*.json della dir risultati del config, in base64 ───────────
# I JSON di valutazione sono immutabili una volta scritti: il driver li mette
# in cache in modo aggressivo e non li richiede se freschi. Il parsing JSON
# avviene nel driver (nessun python sul login node): qui solo cat+base64.
RESULTS_N=0
_emit_run() {
    local rdir="$1" chosen f
    if [ -f "$rdir/eval_final.json" ]; then
        # eval_final è LA scelta canonica (stessa convenzione di
        # src/utils/ablation_summary.py): mai scegliere per mtime se c'è.
        chosen="$rdir/eval_final.json"
    else
        f=$(ls -t "$rdir"eval_*.json 2>/dev/null | grep -v '/eval_baseline\.json$' | head -1) || true
        if [ -z "$f" ]; then
            # Solo eval_baseline presente (run eval-only): meglio di niente.
            f=$(ls -t "$rdir"eval_*.json 2>/dev/null | head -1) || true
        fi
        chosen="$f"
    fi
    if [ -n "$chosen" ] && [ -f "$chosen" ]; then
        RESULTS_N=$((RESULTS_N + 1))
        printf 'RUN_ID_%s=%s\n' "$RESULTS_N" "$(basename "$rdir")"
        printf 'RUN_B64_%s=%s\n' "$RESULTS_N" "$(base64 -w 0 < "$chosen")"
    fi
}

# Elenca, relative a experiments/results, le dir di CELLA che contengono
# risultati. I risultati non vivono più in una sola dir piatta per cella
# (qwen25-05b-sft-grpo/): ora sono annidati come il config
# (qwen25-05b/grpo/zero-shot/run_*/), quindi un glob a un livello vedrebbe
# solo "qwen25-05b". Si parte dai file eval_*.json e si risale alla cella,
# togliendo l'eventuale componente run_<timestamp> finale.
_results_cells() {
    [ -d "$PROJ_DIR/experiments/results" ] || return 0
    (
        cd "$PROJ_DIR/experiments/results" 2>/dev/null || exit 0
        find . -maxdepth 7 -name 'eval_*.json' -type f 2>/dev/null |
            sed -e 's|^\./||' -e 's|/[^/]*$||' \
                -e 's|/run_[^/]*$||' -e 's|/zero_shot_[^/]*$||' |
            sort -u
    )
}

results() {
    local token="$1" dir=""
    # Token vuoto = discovery: elenco delle dir con risultati (il client
    # mostra la lista invece di duplicare la mappa config→dir).
    if [ -z "$token" ]; then
        local d list=""
        while IFS= read -r d; do
            [ -n "$d" ] || continue
            if [ -z "$list" ]; then
                list="$d"
            else
                list="$list$(printf '\x1f')$d"
            fi
        done <<EOF
$(_results_cells)
EOF
        printf 'RESULTS_DIRS=%s\n' "$list"
        return 0
    fi
    # Risoluzione tollerante: path relativo esatto → substring su una cella
    # nota → senza il suffisso prompting (-zero-shot/-few-shot: il wandb
    # run_name della cella non sempre lo contiene). Il primo match in ordine
    # alfabetico vince.
    if [ -d "$PROJ_DIR/experiments/results/$token" ]; then
        dir="$PROJ_DIR/experiments/results/$token"
    else
        local c
        for c in $(_results_cells); do
            case "$c" in
                *"$token"*) dir="$PROJ_DIR/experiments/results/$c"; break ;;
            esac
        done
    fi
    if [ -z "$dir" ]; then
        local t c
        for t in "${token%-zero-shot}" "${token%-few-shot}"; do
            if [ "$t" != "$token" ]; then
                for c in $(_results_cells); do
                    case "$c" in
                        *"$t"*) dir="$PROJ_DIR/experiments/results/$c"; break ;;
                    esac
                done
                [ -n "$dir" ] && break
            fi
        done
    fi
    if [ -z "$dir" ]; then
        printf 'RESULTS_DIR=\nRESULTS_COUNT=0\n'
        return 0
    fi
    printf 'RESULTS_DIR=%s\n' "$dir"
    RESULTS_N=0
    local r has=0
    for r in "$dir"/run_*/; do
        if [ -d "$r" ]; then
            has=1
            _emit_run "$r"
        fi
    done
    # Nessun run_* subdir: i risultati vivono direttamente nella dir config.
    if [ "$has" -eq 0 ]; then
        _emit_run "$dir/"
    fi
    printf 'RESULTS_COUNT=%s\n' "$RESULTS_N"
}

usage() {
    echo "Uso: bash cluster/cluster_helper.sh [status|monitor [n]|enqueue <entry>|enqueue_batch <content>|start_batch <content>|rewrite_queue <content>|pause|resume|tick|timeseries <tag>|results <config>|scancel]" >&2
    exit 2
}

# ── Dispatch ─────────────────────────────────────────────────────────────────
CMD="${1:-status}"
case "$CMD" in
    status)          dump_status ;;
    monitor)         dump_monitor "${2:-200}" ;;
    enqueue)         [ $# -ge 2 ] || usage; enqueue "$2" ;;
    enqueue_batch)   [ $# -ge 2 ] || usage; enqueue_batch "$2" ;;
    start_batch)     [ $# -ge 2 ] || usage; start_batch "$2" ;;
    rewrite_queue)   [ $# -ge 2 ] || usage; rewrite_queue "$2" ;;
    pause)           pause ;;
    resume)          resume ;;
    tick)            tick ;;
    timeseries)      [ $# -ge 2 ] || usage; timeseries "$2" ;;
    results)         [ $# -ge 2 ] || usage; results "$2" ;;
    scancel)         scancel_active ;;
    -h|--help|help)  usage ;;
    *)               usage ;;
esac

# Dopo una mutazione il driver riceve subito lo snapshot fresco (1 sola ssh).
# scancel NON ristampa più solo OK_SCANCEL: stampa già lo snapshot monitor
# dentro scancel_active (vedi sopra). start_batch/timeseries/results stampano
# da soli il loro output completo: nessun dump aggiuntivo.
# tick NON è in questa lista: esce con `exit "$rc"` per propagare il rc del
# chain_tick, quindi non arriverebbe mai qui e stampa il proprio snapshot.
case "$CMD" in
    enqueue|rewrite_queue|enqueue_batch|pause|resume) dump_status ;;
esac
