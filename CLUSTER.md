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
6. [Pipeline Completa](#6-pipeline-completa)
7. [Monitorare](#7-monitorare)
8. [Checkpoint e Resume](#8-checkpoint-e-resume)
9. [Scaricare i Risultati](#9-scaricare-i-risultati)
10. [Troubleshooting](#10-troubleshooting)

---

## 1. Panoramica

### Cosa fa il progetto

Traduzione English → ASL Glosses (T2G) con:

- **Modello**: Qwen2.5-0.5B-Instruct (~1 GB)
- **Constrained Decoding**: LogitsProcessor che forza l'output a sole glosse ASL
- **GRPO Training**: RLHF con 6 reward (translation quality, gold-structure, viterbi, format, repetition, structural dense)
- **LoRA/QLoRA**: Training iper-efficiente via Unsloth (o PEFT standard)

### Cosa serve

| Risorsa  | Minimo                     | Consigliato                  |
| -------- | -------------------------- | ---------------------------- |
| GPU VRAM | 4 GB (fp16, senza Unsloth) | 11 GB (4bit QLoRA + Unsloth) |
| RAM      | 8 GB                       | 48 GB                        |
| Disco    | ~3 GB (modello + dataset)  | 5 GB                         |
| Tempo    | ~2h (500 step)             | ~6h (1500 step)              |

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

### 2.3. Verifica

```bash
ssh <utente>@gcluster.dmi.unict.it
ls ~/neuro_symbolic_t2g/
# Dovresti vedere: src/  experiments/  grammarllm/  main.py  pyproject.toml  ...
```

---

## 3. Setup Iniziale

### 3.1. Apri una sessione interattiva

```bash
# Sostituisci dl-course-q2 con la tua queue
srun --account dl-course-q2 --partition dl-course-q2 --qos gpu-medium \
     --gres=gpu:1 --gres=shard:5000 --mem=8G --pty bash
```

### 3.2. Esegui lo script di setup

```bash
cd ~/neuro_symbolic_t2g
bash cluster/setup.sh
```

Lo script:

- Rileva la GPU e installa le dipendenze appropriate (Unsloth solo se CC >= 7.0)
- Scarica il dataset ASLG-PC12 (~50 MB)
- Estrae il vocabolario gloss (15K token unici)
- Calcola le matrici di transizione bigram
- Verifica che tutto sia installato

### 3.3. Esci dalla sessione

```bash
exit
```

---

## 4. Configurazione

### 4.1. Modifica lo script SLURM

Apri `cluster/train.sh` e imposta i tuoi parametri:

```bash
nano ~/neuro_symbolic_t2g/cluster/train.sh
```

Modifica queste righe:

```bash
#SBATCH --account=dl-course-q2        # ← la tua queue
#SBATCH --partition=dl-course-q2      # ← idem
#SBATCH --qos=gpu-xlarge              # ← il tuo QoS (xlarge = 22 GB VRAM)
#SBATCH --mail-user=tua@email.com     # ← la tua email (opzionale)
#SBATCH --gres=gpu:1 --gres=shard:22528  # ← VRAM in MB
```

### 4.2. Adatta il config YAML alla GPU

Il config base è `experiments/configs/t2g/grpo_qwen05.yaml`. Per GPU diverse da L40S:

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

### 5.2. Evaluation su checkpoint

```bash
CONFIG=experiments/configs/t2g/grpo_qwen05.yaml CHECKPOINT=experiments/checkpoints/grpo/t2g/qwen05/final sbatch cluster/eval.sh
```

### 5.3. Riprendere da un checkpoint

```bash
CONFIG=experiments/configs/t2g/grpo_qwen05.yaml EXTRA_ARGS="--resume" sbatch cluster/train.sh
```

---

## 6. Pipeline Completa

La pipeline orchestrata esegue **train → eval** in sequenza automatica:

```bash
# Carica gli alias (una volta per sessione)
source ~/neuro_symbolic_t2g/cluster/aliases.sh

# Avvia pipeline train+eval
t2g-run-all

# Monitora in tempo reale
t2g-monitor
```

### Comandi rapidi (con alias caricati)

| Comando              | Cosa fa                          |
| -------------------- | -------------------------------- |
| `t2g-train`          | Lancia solo training             |
| `t2g-eval`           | Lancia solo evaluation           |
| `t2g-run-all`        | Train + eval in pipeline         |
| `t2g-monitor`        | Monitor live della pipeline      |
| `t2g-chain-show`     | Mostra stato pipeline            |
| `t2g-chain-stop`     | Ferma pipeline (preserva stato)  |
| `t2g-chain-start`    | Riprendi pipeline dopo stop      |
| `t2g-watcher-status` | Controlla se il watcher è attivo |
| `t2g-watcher-kill`   | Uccidi il watcher                |
| `t2g-clean`          | Pulizia workspace                |
| `t2g-help`           | Lista completa comandi           |

### Pipeline manuale (senza watcher)

```bash
# 1. Training
CONFIG=experiments/configs/t2g/grpo_qwen05.yaml sbatch cluster/train.sh
# Aspetta che finisca (controlla con: squeue -u $USER)

# 2. Evaluation
CONFIG=experiments/configs/t2g/grpo_qwen05.yaml CHECKPOINT=experiments/checkpoints/grpo/t2g/qwen05/final sbatch cluster/eval.sh
```

---

## 7. Monitorare

### Job SLURM

```bash
# Stato dei tuoi job
squeue -u $USER
myjobs   # (con alias caricati)

# Dettagli job
scontrol show job <JOB_ID>

# Output in tempo reale
tail -f logs/slurm-train-<JOB_ID>.log
t2g-trainlog <JOB_ID>   # (con alias)

# Cancella job
scancel <JOB_ID>
```

### Pipeline

```bash
# Monitor live (con alias)
t2g-monitor              # vista compatta
t2g-monitor --tab        # tabella completa
t2g-monitor --all        # tutto: tabella + metriche + samples

# Stato pipeline
t2g-chain-show
t2g-watcher-status
tail -f logs/chain_watcher.log
```

### GPU

```bash
t2g-gpu   # nvidia-smi sul nodo del job attivo
```

---

## 8. Checkpoint e Resume

### Dove vengono salvati

```
~/neuro_symbolic_t2g/
├── experiments/checkpoints/grpo/t2g/qwen05/
│   ├── checkpoint-100/
│   ├── checkpoint-200/
│   └── final/
└── logs/
    ├── slurm-train-<JOB_ID>.log
    └── slurm-eval-<JOB_ID>.log
```

### Resume automatico dopo timeout (12h)

```bash
CONFIG=experiments/configs/t2g/grpo_qwen05.yaml EXTRA_ARGS="--resume" sbatch cluster/train.sh
```

### Resume via pipeline (da job fallito)

```bash
t2g-run-all --resume
```

---

## 9. Scaricare i Risultati

### Da Windows PowerShell

```powershell
# Scarica tutto (logs + checkpoints)
.\neuro_symbolic_t2g\sync_cluster.ps1 -Action download

# Solo log
.\neuro_symbolic_t2g\sync_cluster.ps1 -Action download-logs

# Solo checkpoint
.\neuro_symbolic_t2g\sync_cluster.ps1 -Action download-checkpoints

# File singolo
.\neuro_symbolic_t2g\sync_cluster.ps1 -Action pull -Path "logs/slurm-train-12345.log"
```

### Da Linux/macOS

```bash
rsync -avz <utente>@gcluster.dmi.unict.it:~/neuro_symbolic_t2g/logs/ ./logs/
rsync -avz <utente>@gcluster.dmi.unict.it:~/neuro_symbolic_t2g/checkpoints/ ./checkpoints/
```

### Sincronizzare wandb offline

```bash
# Sul tuo PC, dopo aver scaricato i log con download-wandb:
wandb sync logs/wandb/offline-run-*
```

---

## 10. Troubleshooting

### "ModuleNotFoundError: No module named 'unsloth'"

La GPU non supporta Unsloth (CC < 7.0). Modifica il config:

```yaml
model:
  use_unsloth: false
  fast_inference: false
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
- Verifica che non hai già un job attivo (limite: 1)

### Dataset non trovato

```bash
# Scarica manualmente in sessione interattiva:
srun --account dl-course-q2 --partition dl-course-q2 --qos gpu-medium \
     --gres=gpu:1 --gres=shard:5000 --mem=8G --pty bash
cd ~/neuro_symbolic_t2g
python -c "
from src.datasetsaslg_dataset import download_aslg_dataset, extract_gloss_vocabulary, build_t2g_dataset

dataset = download_aslg_dataset()
vocab = extract_gloss_vocabulary(dataset, split='train')
train_ds = build_t2g_dataset(dataset, split='train')
train_ds.save_to_disk('data/aslg_pc12_train')
"
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
t2g-pip-reset
```

### "pip WARNING: not on PATH" (centinaia di warning)

Esegui `source ~/neuro_symbolic_t2g/cluster/aliases.sh` e poi:

```bash
t2g-install-aliases
```

Questo aggiunge `~/.local/bin` al PATH in modo persistente.

### "Rows must sum to 1" — errore matrice bigram

La matrice di transizione è corrotta. Ricalcola:

```bash
rm data/bigram_transition.npy
# Rilancia il training — verrà ricalcolata automaticamente
```

---

## Riepilogo rapido

```bash
# === PRIMO AVVIO (una volta sola) ===
# 1. Carica il progetto
.\sync_cluster.ps1 -Action upload                    # Windows
rsync -avz neuro_symbolic_t2g/ utente@gcluster:~/neuro_symbolic_t2g/  # Linux

# 2. Setup
ssh utente@gcluster.dmi.unict.it
srun --account <queue> --partition <queue> --qos gpu-medium --gres=gpu:1 --pty bash
cd ~/neuro_symbolic_t2g && bash cluster/setup.sh
exit

# === OGNI VOLTA ===
# 3. Carica alias
source ~/neuro_symbolic_t2g/cluster/aliases.sh

# 4. Lancia pipeline
t2g-run-all

# 5. Monitora
t2g-monitor

# 6. Scarica risultati
.\sync_cluster.ps1 -Action download                  # Windows
```
