# Neuro-Symbolic T2G — Guida all'Addestramento

## Cosa fa questo training

Il progetto addestra **Qwen2.5-0.5B-Instruct** a tradurre frasi inglesi in **glosse ASL**
(American Sign Language) usando **GRPO** (Group Relative Policy Optimization) con
**constrained decoding**.

### La pipeline in 7 step

```
┌─────────────┐    ┌──────────────┐    ┌──────────────────────┐
│ 1. Dataset   │ →  │ 2. Modello    │ →  │ 3. Constrained       │
│ ASLG-PC12    │    │ Qwen 0.5B     │    │ Decoding (vocab mask) │
│ (87K coppie) │    │ + LoRA + 4bit │    │ solo glosse ASL      │
└─────────────┘    └──────────────┘    └──────────────────────┘
                                             ↓
┌─────────────┐    ┌──────────────┐    ┌──────────────────────┐
│ 6. GRPO      │ ←  │ 5. Reward     │ ←  │ 4. T2G Dataset       │
│ Training     │    │ Functions (4) │    │ prompt→completion    │
│ (trl.GRPOTrainer)│                │    │ (chat template)      │
└─────────────┘    └──────────────┘    └──────────────────────┘
```

1. **Dataset**: ASLG-PC12 (87K frasi inglesi → glosse ASL) da HuggingFace
2. **Modello**: Qwen2.5-0.5B-Instruct con LoRA (r=16) e quantizzazione 4-bit (QLoRA)
3. **Constrained Decoding**: un `LogitsProcessor` forza ogni token generato a
   appartenere al vocabolario gloss ASL (15K token). Il modello NON può generare
   parole inglesi.
4. **T2G Dataset**: ogni sample ha `prompt` (frase inglese) e `completion` (glosse gold)
5. **9 Reward Functions**: guidano l'apprendimento senza supervisione umana
6. **GRPO Training**: il modello genera G=8 completions per prompt, riceve reward,
   e aggiorna i pesi LoRA per massimizzare la reward attesa
7. **Salvataggio**: checkpoint ogni 100 step, modello finale in `experiments/checkpoints/grpo/t2g/qwen05/final/`

### Le reward function

| Reward                                | Peso (optimal v2.1) | Cosa misura                                       |
| ------------------------------------- | ------------------- | ------------------------------------------------- |
| **Translation quality** (ROUGE-L)    | 0.20                | Similarità con le glosse gold                     |
| **BLEU-4** (T2G-Reasoner 2025) ⭐     | 0.20                | N-gram precision con effective_order + smoothing   |
| **Gold-structure** (Gold Baseline) ⭐ | 0.20                | Confronto bigram vs gold reference                |
| **Gloss-order** (Edit-distance)       | 0.10                | Levenshtein normalizzato vs gold                  |
| **Verifier-scaled** (RECIPE)          | 0.10                | ROUGE × structural — confidence multiplier        |
| **Format**                            | 0.10                | Assicura output di sole glosse (no free text)     |
| **Repetition**                        | 0.10                | Penalizza sequenze ripetitive                     |

Tutte le reward sono mappate su range simmetrico [-1, 1]. Somma pesi = 1.0.
Vedi `docs/REWARDS.md` per dettagli completi.

### Cosa aspettarsi

**Fase iniziale (step 0-200)**:

- Il modello base produce output casuali/non sense
- Translation reward ~0.0-0.1
- Le glosse generate sono valide (constrained decoding) ma scorrette

**Fase intermedia (step 200-800)**:

- Il modello inizia a produrre glosse correlate all'input
- Translation reward sale a ~0.2-0.4
- Struttura bigram migliora (reward structure ~0.5-0.7)

**Fase avanzata (step 800-1500)**:

- Traduzioni ragionevolmente accurate
- Translation reward ~0.5-0.7
- Il modello impara pattern gloss tipici dell'ASL

**Durata**: ~2-3 ore per 2000 step su L40S con batch_size=1, grad_accum=8, G=8, gradient_checkpointing=true.

### Cosa NON aspettarsi

- **Non è un traduttore perfetto**: Qwen 0.5B è un modello piccolo. La qualità sarà
  sufficiente per dimostrare la metodologia neuro-simbolica, non per uso in produzione.
- **Il constrained decoding garantisce output validi, non corretti**: le glosse generate
  appartengono sempre al vocabolario ASL, ma possono essere sequenze senza senso.
- **vLLM non è usato durante il training**: il `LogitsProcessor` di HuggingFace non è
  compatibile con vLLM. vLLM serve solo per inferenza veloce post-training.
- **gradient_checkpointing**: attivo in tutti i config — ricomputa le attivazioni
  del forward nel backward pass, riducendo peak VRAM del ~30% a costo di ~20% più lento.
  Essenziale per G=8 su GPU 22GB (cluster).
- **Curriculum learning**: 3-stage (simple→medium→hard) abilitato in `grpo_optimal`
  e `grpo_pda_lookahead`. Calibrato sulla distribuzione reale di ASLG-PC12.

### Monitorare il training

```bash
# Tabella live (job, reward, metriche)
t2g-monitor

# Log del watcher (catena job)
tail -f logs/chain_watcher.log

# Log SLURM del job corrente
tail -f logs/slurm-train-<JOB_ID>.log
t2g-trainlog <JOB_ID>

# Stato GPU sul nodo
t2g-gpu
```

### Output attesi

```
experiments/checkpoints/grpo/t2g/qwen05/
├── checkpoint-100/      # Dopo 100 step
├── checkpoint-200/      # Dopo 200 step
├── ...                  # Ogni 100 step
└── final/               # Modello finale (step 2000)

logs/
├── slurm-train-<ID>.log # Log completo training
├── slurm-eval-<ID>.log  # Log evaluation
└── chain_watcher.log    # Log della pipeline
```

### Resume dopo interruzione

```bash
# Training ha crashato? Riprendi dall'ultimo checkpoint
run-all --resume

# Oppure manualmente
CONFIG=experiments/configs/t2g/grpo_optimal.yaml EXTRA_ARGS="--resume" sbatch cluster/train.sh
```

# Ablation study completa (12 config)

```bash
source cluster/aliases.sh
run-all --ablation         # 12 config train+eval (~24h)
monitor --all               # live dashboard
ablation-summary            # tabella + grafico cross-config post-pipeline
```

### Configurazione

Modifica `experiments/configs/t2g/grpo_optimal.yaml` per:

- **Durata**: `training.max_steps` (default 2000)
- **Velocità**: `grpo.num_generations` (default 8, riduci a 4 per GPU piccole)
- **GPU piccole (K80)**: `model.quantization: null`, `model.use_unsloth: false`
- **Quality/speed tradeoff**: `grpo.temperature` (default 0.9, più alto = più esplorazione)
- **OOM**: `training.gradient_checkpointing: true` (già attivo di default)
- **Reward struttura**: `grammar.viterbi_diversity.*` per iperparametri Viterbi
- **Ablation**: `grammar.enabled: false` per GRPO senza constrained decoding
- **PDA**: `grammar.use_grammarllm_pda: true` per LL(1) completo
- **Token lookahead**: `grammar.token_lookahead: true` (solo con PDA, grammarllm v0.5.0)
- **Curriculum**: `curriculum.enabled: true/false`
