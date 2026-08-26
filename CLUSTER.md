# Guida al Cluster — Neuro-Symbolic T2G

Guida passo-passo per eseguire la pipeline neuro-simbolica T2G
(Constrained Decoding + GRPO) sul cluster GPU del DMI UniCT.

---

## Indice

1. [Panoramica](#1-panoramica)
2. [Accesso e Upload](#2-accesso-e-upload)
3. [Setup Iniziale](#3-setup-iniziale)
4. [Configurazione](#4-configurazione)
5. [Lanciare il Training](#5-lanciare-il-training)
6. [Chain / Pipeline orchestration](#6-chain--pipeline-orchestration)
7. [Pipeline Completa](#7-pipeline-completa)
8. [Monitorare](#8-monitorare)
9. [Checkpoint e Resume](#9-checkpoint-e-resume)
10. [Scaricare i Risultati](#10-scaricare-i-risultati)
11. [Troubleshooting](#11-troubleshooting)

---

## 1. Panoramica

### Cosa fa il progetto

Traduzione English → ASL Glosses (T2G) con:

- **Modello**: Qwen2.5-0.5B-Instruct (~1 GB)
- **Constrained Decoding**: LogitsProcessor che forza l'output a sole glosse ASL
- **GRPO Training**: RLHF con 9 reward deterministiche (translation quality, gold-structure, structural dense, gloss-order, verifier-scaled, soft-viterbi, viterbi, format, repetition)
- **LoRA/QLoRA**: Training iper-efficiente via Unsloth (o PEFT standard)

### Vincoli del cluster (verificati)

| Vincolo | Valore | Conseguenza |
| ------- | ------ | ----------- |
| Coda | **max 1 job attivo, 0 pending** (QOSMaxSubmitJobPerUserLimit=1) | la pipeline è **sequenziale**; niente array job |
| `sbatch` | funziona solo dal **login node** | il tick/watcher girano sul login node |
| `at` | **NON disponibile su gcluster** (verificato) | primaria = **hook bashrc** (`chain-hook-install`); watcher = fallback automatico; `at` usato solo se un giorno comparisse |
| cron | NON disponibile sul login node | serve il fallback esterno (remote_tick) |
| python/pip | NON presenti sul login node | i comandi login-node sono **solo shell** |
| Login-node reaper | uccide i processi long-lived (ipotesi: systemd KillUserProcesses al logout) | niente daemon fidato: la catena è guidata da **tick one-shot** |
| QoS time limits | gpu-small 4h · gpu-medium 6h · gpu-large/xlarge 12h | i training lunghi devono usare `--resume` dopo il TIMEOUT |

### GPU supportate

| GPU  | CC  | Unsloth | 4-bit | Note                         |
| ---- | --- | ------- | ----- | ---------------------------- |
| L40S | 8.9 | ✅      | ✅    | Ideale, tutto attivo         |
| V100 | 7.0 | ✅      | ✅    | Ottimo, no bf16              |
| K80  | 3.7 | ❌      | ❌    | Solo fp16, no quantizzazione |

---

## 2. Accesso e Upload

### 2.1. Connettiti al cluster

```bash
ssh <codice-fiscale>@gcluster.dmi.unict.it
```

### 2.2. Carica il progetto

**Da Windows PowerShell:**

```powershell
.\neuro_symbolic_t2g\sync_cluster.ps1 -Action upload
```

**Da Linux/macOS (rsync):**

```bash
rsync -avz --exclude '__pycache__' --exclude 'data/' --exclude 'logs/' \
    neuro_symbolic_t2g/ <utente>@gcluster.dmi.unict.it:~/neuro_symbolic_t2g/
```

> **Nota**: `data/` e `logs/` sono esclusi — il dataset viene scaricato sul cluster.

---

## 3. Setup Iniziale

### 3.1. Apri una sessione interattiva

```bash
srun --account <queue> --partition <queue> --qos gpu-xlarge \
     --gres=gpu:1 --gres=shard:5000 --mem=8G --pty bash
```

### 3.2. Esegui lo script di setup

```bash
cd ~/neuro_symbolic_t2g
bash cluster/setup.sh
```

Lo script (rilancia se stesso dentro srun + Apptainer):

- Installa le dipendenze da `pyproject.toml`
  (`pip install --user -e ".[retrieval]"`): core **include scikit-learn**
  (backend retrieval tfidf, default in `grpo_optimal.yaml`) + extra
  `retrieval` (sentence-transformers); l'extra `dev` (formattazione/test) è
  escluso di proposito
- Scarica il dataset ASLG-PC12 (~50 MB), estrae il vocabolario gloss (15K
  token), calcola le matrici bigram e salva le split su disco
- Pre-downloada Qwen2.5-0.5B-Instruct per la cache offline
- Verifica gli import (incluso `sklearn` e `sentence_transformers`, con
  warning se manca — non fa fallire il setup)

> **`sentence-transformers` viene installato** via l'extra `retrieval`
> (`pip install --user -e ".[retrieval]"`). Il modello MiniLM viene scaricato
> in modo **lazy** al primo uso del backend `retrieval.backend: "minilm"`;
> il default resta `tfidf` (deterministico, scikit-learn) → **zero costi** se
> non attivi minilm. L'extra `dev` (isort/black/ruff/pytest) è escluso di
> proposito dal cluster.

### 3.3. Carica gli alias e chiudi

```bash
source ~/neuro_symbolic_t2g/cluster/aliases.sh
# (persistente: install-aliases)
exit
```

---

## 4. Configurazione

### 4.1. Modifica lo script SLURM

Apri `cluster/train.sh` e imposta i tuoi parametri:

```bash
#SBATCH --account=<queue>          # ← la tua queue
#SBATCH --partition=<queue>        # ← idem
#SBATCH --qos=gpu-xlarge           # ← il tuo QoS (xlarge = 12h/22 GB VRAM)
#SBATCH --mail-user=tua@email.com  # ← la tua email (opzionale)
#SBATCH --gres=gpu:1 --gres=shard:22528  # ← VRAM in MB
```

### 4.2. Adatta il config YAML alla GPU

I config T2G ereditano da `experiments/configs/t2g/base.yaml` via la chiave
`extends:` (risolta da `src/utils/config.py::resolve_config`). Per GPU diverse
da L40S:

**Per V100 (no bf16):**

```yaml
model:
  dtype: "float16"
  quantization: "4bit"
  use_unsloth: true
training:
  bf16: false
```

**Per K80 (no Unsloth, no 4bit):**

```yaml
model:
  dtype: "float16"
  quantization: null
  use_unsloth: false
  fast_inference: false
training:
  bf16: false
  optim: "adamw_torch"
grpo:
  num_generations: 2
  max_completion_length: 128
```

---

## 5. Lanciare il Training

### 5.1. Training singolo

```bash
cd ~/neuro_symbolic_t2g
mkdir -p logs
CONFIG=experiments/configs/t2g/grpo_qwen05.yaml sbatch cluster/train.sh
```

Il checkpoint viene salvato in `experiments/checkpoints/<model>/run_<timestamp>/`
(vedi [Checkpoint e Resume](#9-checkpoint-e-resume)).

### 5.2. Evaluation su checkpoint

```bash
CONFIG=experiments/configs/t2g/grpo_qwen05.yaml CHECKPOINT=experiments/checkpoints/qwen25-05b/run_20260403_120000/final sbatch cluster/eval.sh
```

Senza `CHECKPOINT`, `eval.sh` **auto-detecta** il checkpoint con
`resolve_config` (risolve `extends:`, a differenza del vecchio
`yaml.safe_load` sul solo file figlio). Se la config dichiara
`training.output_dir` ma il checkpoint non esiste, l'eval **FALLISCE LOUD**
(exit 1) invece di fare zero-shot silenzioso su un modello non addestrato.
Solo i config eval-only (es. `zero_shot.yaml`, senza `output_dir`) girano
legittimamente in zero-shot.

### 5.3. Riprendere da un checkpoint

```bash
CONFIG=experiments/configs/t2g/grpo_qwen05.yaml EXTRA_ARGS="--resume" sbatch cluster/train.sh
```

---

## 6. Chain / Pipeline orchestration

> **Questa è la parte centrale.** La catena è un file `~/.chain_state/job_chain`
> (una entry `type:config:tag[:extra]` per riga, es. `train:experiments/configs/t2g/grpo_optimal.yaml:grpo-optimal`).
> Un **tick one-shot idempotente** (`cluster/chain_tick.sh`) la fa avanzare di
> un passo per invocazione e NON esiste più un daemon long-lived da tenere vivo.

### 6.1. Architettura del tick

```
                    +---------------------------------------------+
                    |          chain_tick.sh (one-shot)            |
                    | 1. flock (max 1 tick alla volta)             |
                    | 2. .chain_stopped?          → exit 0         |
                    | 3. job attivo (squeue)?     → heartbeat,exit|
                    | 4. job_chain vuota?         → exit 0         |
                    | 5. ultimo job (sacct):                       |
                    |    TIMEOUT/OOM/CUDA + train + retry<2        |
                    |       → reinserisce "train:..:--resume"      |
                    |    altro FAILED → log in chain_errors,       |
                    |       salta l'eval del train fallito         |
                    | 6. sbatch della prossima entry               |
                    | 7. --schedule[=N] → rischedula via at (dedup)|
                    +-------------+-----+------+-----+-------------+
                    |             |           |     |
        bashrc-hook (PRIMARIO)  watcher     manuale  remoto (cron)
        al prompt, throttled    (fallback   chain-   remote_tick.sh
        5 min                   automatico) resume
        (at: opportunistico,    setsid
        se un giorno presente)
```

Le modalità di guida (primaria → ultima ratio):

| # | Modalità | Prerequisito | Comando |
| - | -------- | ------------ | ------- |
| 1 | **bashrc-hook** (PRIMARIO / raccomandato) | `~/.bashrc` col blocco `t2g-chain-hook` | `chain-hook-install` (poi `source ~/.bashrc`); `PROMPT_COMMAND` throttled 300s |
| 2 | **watcher fallback** (automatico) | nessuno | avviato da `run-all` / `chain-resume` quando `at` manca (il caso gcluster); best-effort — il reaper può ucciderlo → riprendi con `chain-resume` |
| 3 | **at-mode** (opportunistico) | `at` presente sul login node (oggi ASSENTE su gcluster) | `chain_tick.sh --schedule`; dedup = max 1 job at pendente; se un giorno `at` comparisse, rilevato automaticamente senza danni |
| 4 | **esterno** (primario se `remote/` è deployato) | Render + cronjob.org (o cron/ssh semplice) | `remote/app.py` → `cluster_helper.sh` / `cluster/remote_tick.sh` |

Il tick è **innocuo e idempotente**: se c'è un job attivo non tocca la coda,
se la coda è vuota non fa nulla, se due istanze partono insieme il `flock` ne
scarta una. Può quindi essere chiamato da più sorgenti senza rischi.

### 6.2. Stato su disco (`.chain_state/`)

| File | Contenuto |
| ---- | --------- |
| `job_chain` | coda: una entry per riga |
| `last_job` | `id:type:config:tag:retries` dell'ultima sottomissione (i retry TIMEOUT/OOM/CUDA sono contati qui, max 2) |
| `chain_errors` | log JSONL degli errori (letto da `monitor`) |
| `chain_stopped` | presente = pausa fatta con `chain-stop` (config salvato dallo stato) |
| `chain_failed` | legacy, solo per fallimenti non più ripetibili |
| `heartbeat` / `tick_stamp` | tick attivi / throttle del hook |
| `chain_pid` | solo in modalità watcher fallback |
| `tick.lock` | flock del tick |

### 6.3. Diagnostica (3 comandi sul login node)

Su gcluster `at` NON è disponibile: la resilienza si ottiene con l'HOME hook
bashrc (primario) e il watcher resta il fallback automatico del lancio.

```bash
# 1. Linger è attivo? (se no, i processi muoiono al logout)
loginctl show-user $USER --property=Linger

# 2. Attiva il linger (sopravvivenza ai logout — di solito serve sudo/root,
#    altrimenti chiedi all'amministratore del cluster)
loginctl enable-linger $USER

# 3. `at` è presente? (su gcluster: atteso ASSENTE → usare l'hook bashrc)
command -v at          # vuoto = ok, non serve; installa comunque l'hook
```

### 6.4. Tabella "sintomo → soluzione"

| Sintomo | Causa probabile | Soluzione |
| ------- | --------------- | --------- |
| Pipeline ferma, `job_chain` non vuota, nessun job attivo | tick/watcher morto (reaper del login node) | `chain-resume` (riparte dalla coda esistente, niente `--force`) |
| `run-all` rifiuta con "catena interrotta con N job rimanenti" | stato pendente rilevato (anti-rm-rf) | `chain-resume`, oppure `run-all --force` solo se vuoi davvero ricominciare |
| La catena si ferma quando ti scolleghi | hook bashrc non installato (gcluster non ha `at`) | `chain-hook-install && source ~/.bashrc` (PRIMARIO) |
| `atq` vuoto | normale su gcluster: `at` NON è disponibile | ignorare: il watcher parte da solo; per la resilienza usare l'hook bashrc |
| Eval che prima girava "zero-shot senza errori" | checkpoint mancante | ora è FAIL LOUD: addestra prima, oppure passa `CHECKPOINT=` esplicito |
| `chain-stop`/`chain-start` col config sbagliato | hardcode grpo_qwen05 | risolto: il config è letto da `.chain_state` (`last_job`/testa coda) |
| Training `ModuleNotFoundError: sklearn` | ambiente pip --user non allineato al pyproject | `pip-reset` (clean + reinstall da `pyproject.toml`) |
| `clean-model grpo-optimal` non trovava nulla | glob/nome sbagliato | ora mappa tag→output_dir dai config e i log SLURM via sacct |

### 6.5. Retry automatico (TIMEOUT / OOM / CUDA)

Se un job di training termina con `TIMEOUT` (time limit QoS), `OUT_OF_MEMORY`
o errore CUDA transitorio, il tick **reinserisce in testa** la stessa entry
con `--resume` e la ripresenta (max **2 tentativi**, contati in `last_job`).
Se anche il retry fallisce, l'errore finisce in `chain_errors`, l'eventuale
eval associato viene saltato (per non valutare un modello non addestrato) e la
catena continua con l'entry successiva (continue-on-failure, pensato per
l'ablation study).

### 6.6. Driver esterno (Render + cronjob.org)

> Quando `remote/` è deployato, il driver diventa la modalità **PRIMARIA** di
> avanzamento; hook bashrc e watcher restano installati come fallback innocui
> (il tick è idempotente, più driver non si pestano i piedi).

**Architettura (in 5 righe):** cronjob.org POSTa `/tick` ogni 5 min (header
`X-Auth-Token`) → il servizio FastAPI su Render apre **una** ssh verso il
login node → esegue `cluster_helper.sh tick` (che chiama
`chain_tick.sh --quiet`, one-shot e idempotente) → riceve uno snapshot
`KEY=VALUE` machine-readable (coda separata da `\x1f`) → lo sincronizza in un
DB SQLite locale. La **fonte di verità resta `.chain_state/` sul cluster**:
il filesystem di Render è effimero, quindi il DB è solo cache/diario (il
puntatore `errors_offset` evita di rileggere gli errori già visti).

**Quando usarlo:** primario se deployato (piloti la coda via API:
aggiungi/rimuovi job, `POST /pause`/`/resume`, `GET /status`). Hook bashrc
(`chain-hook-install`) e watcher (`chain_next.sh`) diventano **fallback** e
possono convivere senza danni.

**Deploy:** tutto in `remote/` — README passo-passo (Web Service su Render,
build `pip install -r remote/requirements.txt`, start
`cd remote && uvicorn app:app --host 0.0.0.0 --port $PORT`), env vars,
setup chiave dedicata e cronjob.org: **`remote/README.md`**.

**Sicurezza:** chiave SSH **dedicata** senza passphrase generata solo per il
driver (`ssh-keygen -t ed25519 -f ~/.ssh/t2g_driver -N ""` + la pubblica in
`~/.ssh/authorized_keys`): se trapela da Render la revochi con una sola riga
senza toccare il tuo accesso. Token API e chiave vivono solo nelle env vars
di Render e **non vengono mai loggati**; tutte le route richiedono
`X-Auth-Token` (confronto constant-time).

**Nota hook:** `install-aliases` / `chain-hook-install` installa comunque
l'hook bashrc come fallback; se vuoi l'unico driver = Render, rimuovilo con
`chain-hook-uninstall` (il driver esterno continua a funzionare da solo).

---

## 7. Pipeline Completa

### 7.1. Avvia

```bash
# Carica gli alias (una volta per sessione)
source ~/neuro_symbolic_t2g/cluster/aliases.sh

# Avvia pipeline train+eval (default: grpo_optimal)
run-all            # alias per: bash cluster/run_all.sh

# Oppure con un config specifico / ablation:
run-all grpo_qwen05
run-all --ablation
```

`run-all` costruisce `job_chain`, poi avvia il **watcher** come fallback
automatico (su gcluster `at` NON esiste) e stampa in evidenza il comando per
installare l'HOME hook (`chain-hook-install`), la resilienza raccomandata. Se
un giorno `at` comparisse sul login node, verrebbe rilevato e usato
automaticamente dal tick (`--schedule`, dedup ≤1 pending).

### 7.2. Comandi rapidi (con alias caricati)

| Comando | Cosa fa |
| ------- | ------- |
| `run-all [config] [--ablation\|--eval-only\|--train-only\|--resume\|--append\|--force]` | avvia/riprende la pipeline |
| `chain-show` | stato pipeline + job in coda |
| `chain-resume` | **riparte da una catena interrotta** (daemon ucciso, senza `--force`) |
| `chain-stop` / `chain-start` | ferma (preserva stato, config dallo stato) / riprende |
| `chain-add` / `chain-remove` | aggiungi / svuota coda |
| `chain-hook-install` / `chain-hook-uninstall` | hook bashrc che avanza la catena al prompt |
| `watcher-status` / `watcher-kill` | stato watcher+at-tick / uccidi watcher e tick |
| `monitor` (o `t2g-monitor`) | monitor live della pipeline |
| `train` / `run-eval` | singolo training / eval |
| `clean` / `clean-model <TAG>` | pulizia workspace / modello |

Alias **`t2g-*`** equivalenti (allineati a questa guida): `t2g-train`,
`t2g-eval`, `t2g-run-all`, `t2g-monitor`, `t2g-chain-show`, `t2g-chain-stop`,
`t2g-chain-start`, `t2g-chain-resume`, `t2g-watcher-status`,
`t2g-watcher-kill`, `t2g-clean`, `t2g-gpu`, `t2g-help`.

### 7.3. Hook bashrc (PRIMARIO — raccomandato)

```bash
chain-hook-install    # appende il blocco t2g-chain-hook a ~/.bashrc
source ~/.bashrc      # attivalo subito (o ri-loggati)
```

Il hook è un `PROMPT_COMMAND` **silenzioso e throttled**: se `job_chain` non è
vuota e `tick_stamp` ha più di 300s, lancia `chain_tick.sh --quiet`. È sicuro
se il progetto non esiste. **Va ri-eseguito dopo un wipe della home.**

### 7.4. Fallback esterno (ultima ratio)

Se nemmeno l'hook bashrc è utilizzabile (o vuoi una rete di sicurezza
indipendente dal login node), un cron **esterno** può chiamare il tick via ssh:

```bash
# su una macchina esterna (cron ogni 5 min)
*/5 * * * *  bash ~/neuro_symbolic_t2g/cluster/remote_tick.sh >> ~/t2g_tick.log 2>&1
```

Vedi `cluster/remote_tick.sh` (template completo).

---

## 8. Monitorare

### Job SLURM

```bash
squeue -u $USER
myjobs            # (con alias)

scontrol show job <JOB_ID>

tail -f logs/slurm-train-<JOB_ID>.log
t2g-trainlog <JOB_ID>   # (con alias)

scancel <JOB_ID>
```

### Pipeline

```bash
# Monitor live (con alias)
monitor               # vista compatta
monitor --tab         # tabella completa
monitor --all         # tutto: tabella + metriche + samples

# Stato pipeline
chain-show
watcher-status
tail -f logs/chain_watcher.log
```

### GPU

```bash
t2g-gpu   # nvidia-smi sul nodo del job attivo
```

---

## 9. Checkpoint e Resume

### Dove vengono salvati

Layout **flat** (i config v2+ scrivono `training.output_dir` direttamente
sotto `experiments/checkpoints/`):

```
~/neuro_symbolic_t2g/
├── experiments/checkpoints/
│   ├── qwen25-05b/
│   │   ├── run_20260403_120000/
│   │   │   ├── checkpoint-100/
│   │   │   ├── checkpoint-200/
│   │   │   └── final/
│   │   └── latest -> run_20260403_120000
│   └── qwen25-05b-optimal/
│       └── ...
├── experiments/results/<model>/<run_id>/    (eval JSON)
├── experiments/figures/<model>/<run_id>/    (plot)
└── logs/
    ├── slurm-train-<JOB_ID>.log
    ├── slurm-eval-<JOB_ID>.log
    └── chain_watcher.log
```

`ckpts` (alias) mostra questo layout flat.

### Resume automatico dopo TIMEOUT (12h)

La catena reinserisce automaticamente il training con `EXTRA_ARGS="--resume"`
(max 2 tentativi). Manualmente:

```bash
CONFIG=experiments/configs/t2g/grpo_qwen05.yaml EXTRA_ARGS="--resume" sbatch cluster/train.sh
```

### Resume di una catena interrotta

```bash
chain-resume    # riparte dalla coda esistente (NON serve --force)
```

`run-all --resume` funziona **anche senza `.chain_failed`**: basta che
`job_chain` sia non vuota (il caso reale del daemon ucciso dal reaper).

---

## 10. Scaricare i Risultati

### Da Windows PowerShell

```powershell
.\neuro_symbolic_t2g\sync_cluster.ps1 -Action download

# Solo log / checkpoint / risultati / figure
.\sync_cluster.ps1 -Action download-logs
.\sync_cluster.ps1 -Action download-checkpoints
.\sync_cluster.ps1 -Action download-results
.\sync_cluster.ps1 -Action download-figures

# File singolo
.\sync_cluster.ps1 -Action pull -Path "logs/slurm-train-12345.log"
```

### Da Linux/macOS

```bash
rsync -avz <utente>@gcluster.dmi.unict.it:~/neuro_symbolic_t2g/experiments/ ./experiments/
```

---

## 11. Troubleshooting

### "Catena interrotta con N job rimanenti" (run-all si rifiuta)

È il **guard anti-rm-rf**: esistono job pendenti senza driver attivo. Non
azzerare mai con `rm -rf`:

```bash
chain-resume        # riparte dalla coda esistente
# oppure, SOLO se vuoi davvero ricominciare:
run-all --force
```

### Il watcher muore / la pipeline si ferma

```bash
# 1. Diagnosi reaper (vedi sezione 6.3)
loginctl show-user $USER --property=Linger
loginctl enable-linger $USER
command -v at        # su gcluster: atteso assente

# 2. Riprendi e attiva la resilienza (hook PRIMARIO)
chain-resume
chain-hook-install && source ~/.bashrc
```

### "ModuleNotFoundError: No module named 'unsloth'"

La GPU non supporta Unsloth (CC < 7.0). Modifica il config:

```yaml
model:
  use_unsloth: false
  fast_inference: false
```

### "ModuleNotFoundError: sklearn"

Retrieval tfidf attivo ma ambiente non allineato (scikit-learn è una
dipendenza core di `pyproject.toml`):

```bash
pip-setup               # reinstalla tutto da pyproject.toml (core + retrieval)
# oppure da zero:       pip-reset
```

### "CUDA out of memory"

Riduci le risorse:

```yaml
grpo:
  num_generations: 2
  max_completion_length: 128
training:
  per_device_train_batch_size: 1
  gradient_accumulation_steps: 4
```

Oppure riduci `gpu_memory_utilization` nel config.

### Job non parte (PENDING)

- Controlla con `squeue -u $USER`
- Prova un QoS più piccolo (`gpu-large` invece di `gpu-xlarge`)
- Verifica che non hai già un job attivo (limite: **1**, e **0 pending** —
  la catena non sottomette mai mentre c'è un job)

### Eval "CHECKPOINT NON TROVATO — eval RIFIUTATO"

La config dichiara `training.output_dir` ma non esiste alcun checkpoint:
il modello non è stato addestrato (oppure `output_dir` non è quello atteso).
Addestra prima, oppure passa `CHECKPOINT=<path>` esplicito. I config eval-only
(`zero_shot*`) non hanno `output_dir` e girano in zero-shot senza questo errore.

### Dataset non trovato

```bash
# Scarica manualmente in sessione interattiva:
srun --account <queue> --partition <queue> --qos gpu-xlarge \
     --gres=gpu:1 --gres=shard:5000 --mem=8G --pty bash
cd ~/neuro_symbolic_t2g
bash -c 'source cluster/_lib.sh && cd "$PROJ_DIR" && prepare_data'
exit
```

### Training troppo lento

- Su K80: riduci `max_steps` a 500, `max_samples` a 1000
- Su V100/L40S: attiva `use_unsloth: true` e `fast_inference: true`
- Aumenta `gpu_memory_utilization` a 0.90

### "Unsloth cannot find any torch accelerator"

Hai PyTorch compilato per CUDA 13.x ma il cluster ha driver CUDA 12.x.
**Prima** ricarica i file fixati sul cluster, poi resetta l'ambiente:

```bash
# Sul tuo PC:
.\neuro_symbolic_t2g\sync_cluster.ps1 -Action upload

# Sul cluster:
pip-reset
```

### "pip WARNING: not on PATH" (centinaia di warning)

Esegui `source ~/neuro_symbolic_t2g/cluster/aliases.sh` e poi:

```bash
install-aliases
```

Questo aggiunge `~/.local/bin` al PATH in modo persistente.

### "Rows must sum to 1" — errore matrice bigram

La matrice di transizione è corrotta. Ricalcola:

```bash
rm data/bigram_transition.npy data/bigram_transition.npy.meta.json
# Rilancia il training — verrà ricalcolata automaticamente
```

### Pulizia

```bash
clean                     # dry-run
clean --force             # cancella (PRESERVA data/retriever_index* e *.meta.json)
clean --force --all-cache # cancella anche le cache costose (rebuild TF-IDF ~minuti)
clean-model grpo-optimal  # dry-run per un modello (accetta tag o nome cartella)
clean-model grpo-optimal --all   # cancella davvero
```

---

## Riepilogo rapido

```bash
# === PRIMO AVVIO (una volta sola) ===
.\sync_cluster.ps1 -Action upload                       # Windows
ssh utente@gcluster.dmi.unict.it
srun --account <queue> --partition <queue> --qos gpu-xlarge --gres=gpu:1 --pty bash
cd ~/neuro_symbolic_t2g && bash cluster/setup.sh        # core + retrieval da pyproject.toml
exit
source ~/neuro_symbolic_t2g/cluster/aliases.sh
install-aliases
chain-hook-install && source ~/.bashrc                  # PRIMARIO: resilienza della catena

# === OGNI VOLTA ===
run-all                  # oppure: run-all grpo_qwen05 / run-all --ablation
monitor                  # monitor live
# se la catena si ferma: chain-resume
.\sync_cluster.ps1 -Action download                     # scarica i risultati
```
