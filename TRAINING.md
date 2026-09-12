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
│ Training     │    │ Functions (8) │    │ prompt→completion    │
│ (trl.GRPOTrainer)│                │    │ (chat template)      │
└─────────────┘    └──────────────┘    └──────────────────────┘
```

1. **Dataset**: ASLG-PC12 (87K frasi inglesi → glosse ASL) da HuggingFace
2. **Modello**: Qwen2.5-0.5B-Instruct con LoRA (r=32) e quantizzazione 4-bit (QLoRA)
3. **Constrained Decoding**: un `LogitsProcessor` forza ogni token generato a
   appartenere al vocabolario gloss ASL (15K token). Il modello NON può generare
   parole inglesi.
4. **T2G Dataset**: ogni sample ha `prompt` (frase inglese) e `completion` (glosse gold)
5. **8 Reward Functions**: guidano l'apprendimento senza supervisione umana
6. **GRPO Training**: il modello genera G=8 completions per prompt, riceve reward,
   e aggiorna i pesi LoRA per massimizzare la reward attesa
7. **Salvataggio**: checkpoint ogni `training.save_steps` (500 in base.yaml), modello finale in `experiments/checkpoints/qwen25-05b/sft-grpo/few-shot/run_<timestamp>/final/`

### Le reward function

| Reward                                          | Peso (stack base) | Cosa misura                                      |
| ----------------------------------------------- | ----------------- | ------------------------------------------------ |
| **Translation quality** (ROUGE-L)               | 0.20              | Similarità con le glosse gold                    |
| **BLEU-4**                                      | 0.20              | N-gram precision con effective_order + smoothing  |
| **Gold-structure**                              | 0.20              | Confronto bigram vs gold reference               |
| **Gloss-order** (edit-distance)                 | 0.10              | Levenshtein normalizzato vs gold                 |
| **Verifier-scaled**                             | 0.10              | ROUGE × structural — confidence multiplier       |
| **Gloss-format**                                | 0.10              | Assicura output di sole glosse (no free text)    |
| **Gloss-repetition**                            | 0.10              | Penalizza sequenze ripetitive                    |
| **Edit-validity**                               | 0 (ablation)      | Similarità di edit word-level; fuori dallo stack storico |

Lo stack storico di default somma 1.0 sulle prime 7 componenti (chiavi
`weight_translation`, `weight_bleu`, `weight_gold_structure`,
`weight_gloss_order`, `weight_verifier_scaled`, `weight_format`,
`weight_repetition`). `edit_validity` (`weight_edit_validity`,
`edit_validity_oov_weight` default 0.5) è opt-in: vive solo in
`ablations/rewards/edit-validity.yaml`. Tutte le reward sono mappate su range
simmetrico [-1, 1]. Vedi `docs/REWARDS.md` per dettagli completi.

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

**Durata**: ~8 ore per 5000 step (il `training.max_steps` di base.yaml) su L40S con batch_size=1, grad_accum=8, G=8, gradient_checkpointing=true (~5,8 s/step misurati). NB: 1 PROMPT per passo, non 8 — in TRL 0.24 `steps_per_generation = grad_accum`, quindi l'accumulo genera rollout dello stesso prompt.

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

### Monitorare il training

```bash
# Tabella live (job, reward, metriche)
t2g-monitor

# Log della catena (tick)
tail -f logs/chain.log

# Log SLURM del job corrente
tail -f logs/slurm-train-<JOB_ID>.log
t2g-trainlog <JOB_ID>

# Stato GPU sul nodo
t2g-gpu
```

### Output attesi

```
experiments/checkpoints/qwen25-05b/sft-grpo/few-shot/run_<timestamp>/
├── checkpoint-500/        # Dopo 500 step (training.save_steps)
├── checkpoint-1000/       # Dopo 1000 step
├── ...                    # Ogni save_steps
└── final/                 # Modello finale (step 5000)

logs/
├── slurm-train-<ID>.log # Log completo training
├── slurm-eval-<ID>.log  # Log evaluation
└── chain.log            # Log della pipeline
```

### Resume dopo interruzione

```bash
# Training ha crashato? Riprendi dall'ultimo checkpoint
run-all --resume

# Oppure manualmente
CONFIG=experiments/configs/qwen25-05b/sft-grpo/few-shot.yaml EXTRA_ARGS="--resume" sbatch cluster/train.sh
```

# Ablation study completa (15 celle / 27 entry)

```bash
source cluster/aliases.sh
run-all --ablation         # 15 celle: 3 baseline eval-only + 12 train+eval
monitor --all               # live dashboard
ablation-summary            # tabella + grafico cross-config post-pipeline
```

### Configurazione

Modifica `experiments/configs/qwen25-05b/sft-grpo/few-shot.yaml` per:

- **Durata**: `training.max_steps` (default 5000 in base.yaml; governa SOLO il
  GRPO — l'SFT è governato da `num_train_epochs` e ignora `max_steps`).
  NB: 1 prompt per passo, non 8.
- **Velocità**: `grpo.num_generations` (default 8, riduci a 4 per GPU piccole)
- **GPU piccole (K80)**: `model.quantization: null`, `model.use_unsloth: false`
- **Quality/speed tradeoff**: `grpo.temperature` (default 0.7 nella base, più alto = più esplorazione)
- **OOM**: `training.gradient_checkpointing: true` (già attivo di default)
- **Ablation**: `grammar.enabled: false` per GRPO senza constrained decoding
- **Obiettivo RL** (`grpo:`): `loss_type` (`grpo`|`bnpo`|`dr_grpo`|`dapo`),
  `scale_rewards` (`group`|`batch`|`none`), `mask_truncated_completions`,
  `epsilon`, `epsilon_high`. Se assenti valgono i default di TRL 0.24.0
  (`dapo` / `group` / `False`), gli stessi sotto cui sono stati prodotti tutti
  i risultati storici; vedi `ablations/loss/dr-grpo.yaml`.
- **Obiettivi ausiliari SFT** (`auxiliary_objective:`): `allowed_mass` (peso +
  `warmup_steps`) e `structured` (peso, `warmup_steps`, `top_k`, `alpha`,
  `shuffled_control`). Con peso 0 o sezione assente l'SFT è bit-identico a
  quello standard; vedi `ablations/objectives/{sft-allowed-mass,sft-structured}.yaml`.
  Il braccio di controllo GATE 2 (transizioni permutate) è un config
  separato, `ablations/objectives/sft-structured-shuffled.yaml` (estende
  `sft-structured.yaml`, cambia solo `shuffled_control: true`).
