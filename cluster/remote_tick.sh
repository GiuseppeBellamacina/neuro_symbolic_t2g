#!/bin/bash
# ============================================================================
# remote_tick.sh — Fallback ESTERNO per la pipeline T2G (ultima ratio).
#
# Quando at / bashrc-hook / watcher falliscono tutti (login node senza atd,
# hook non installato, reaper che uccide i daemon), la catena può essere
# avanzata da un cron/scheduler ESTERNO che invoca il tick sul login node
# via ssh. Il tick è one-shot e idempotente → chiamarlo ogni pochi minuti
# è sicuro (flock + check di stato).
#
# Uso (su una macchina esterna con ssh verso il cluster):
#   */5 * * * *  bash ~/neuro_symbolic_t2g/cluster/remote_tick.sh >> ~/t2g_tick.log 2>&1
#
# Oppure da GitHub Actions / render cron job (ogni 5 min):
#   - name: Advance T2G chain
#     run: ssh -o BatchMode=yes "${GC_USER:-user}@gcluster.dmi.unict.it" \
#            'cd ~/neuro_symbolic_t2g && bash cluster/chain_tick.sh'
#
# Requisiti: chiave ssh senza passphrase (ssh-agent / keypair) e BatchMode.
# ============================================================================

set -euo pipefail

GC_USER="${GC_USER:-user}"
GC_HOST="${GC_HOST:-gcluster.dmi.unict.it}"

if ! ssh -o BatchMode=yes -o ConnectTimeout=15 "${GC_USER}@${GC_HOST}" \
    'cd ~/neuro_symbolic_t2g && bash cluster/chain_tick.sh' 2>&1; then
    echo "$(date '+%F %T') remote_tick: ssh fallito (${GC_USER}@${GC_HOST})" >> "${HOME}/t2g_tick_errors.log" 2>/dev/null || true
    exit 1
fi
exit 0
