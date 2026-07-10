# Neuro-Symbolic T2G — Report di Ricerca

**Progetto**: Neuro-Symbolic Text-to-Gloss Translation via Constrained Decoding + GRPO
**Data**: Luglio 2026
**Modello**: Qwen2.5-0.5B-Instruct (4-bit QLoRA, LoRA)
**Dataset**: ASLG-PC12 (87K coppie English→ASL Gloss)
**Codice**: `neuro_symbolic_t2g/`

---

## 1. Obiettivo del Progetto

Il progetto mira a dimostrare che un **LLM di piccola dimensione** (0.5B parametri)
può essere fine-tunato per tradurre testo inglese in **gloss ASL** (American Sign Language)
combinando tre paradigmi:

1. **Reinforcement Learning (GRPO)** — Group Relative Policy Optimization, che
   genera multiple completions per prompt e le confronta relazionalmente.
2. **Constrained Decoding Neurale** — un `LogitsProcessor` maschera a ogni step
   di generazione tutti i token al di fuori del vocabolario ASL (~15K token),
   garantendo output sintatticamente valido.
3. **Reward Symboliche Deterministiche** — 9 funzioni di reward basate su regole
   (ROUGE-L, bigrammi, Viterbi, edit-distance, formato, ripetizione) che
   sostituiscono un reward model neurale, eliminando overhead e bias.

L'architettura **neuro-symbolica** combina l'apprendimento distribuito del modello
neurale con la garanzia formale del decoding vincolato e le reward simboliche.

---

## 2. Architettura del Sistema

### 2.1 Pipeline End-to-End

```text
┌──────────────┐    ┌──────────────────┐    ┌──────────────────────┐
│ English text │ →  │ Qwen2.5-0.5B     │ →  │ Constrained Decoder  │
│ "The man     │    │ + LoRA (QLoRA)   │    │ (vocabulary mask)    │
│  walks home" │    │ + GRPO training  │    │ only ASL gloss tokens │
└──────────────┘    └──────────────────┘    └──────────────────────┘
                                                      ↓
                                            ┌──────────────────────┐
                                            │ IX MAN WALK HOUSE    │
                                            │ (ASL gloss sequence) │
                                            └──────────────────────┘
```

### 2.2 7 Step della Pipeline

| Step | Cosa                                                           | File                             |
| ---- | -------------------------------------------------------------- | -------------------------------- |
| 1    | **Data**: Download ASLG-PC12 (87K coppie) da HuggingFace       | `src/datasets/aslg_dataset.py`   |
| 2    | **Model**: Load Qwen2.5-0.5B + LoRA + 4-bit QLoRA via Unsloth  | `src/models/model_loader.py`     |
| 3    | **Constrained Decoding**: `GlossVocabularyMask` o PDA LL(1)    | `src/grammar/gloss_grammar.py`   |
| 4    | **Dataset**: Formattazione prompt-completion con chat template | `src/datasets/aslg_dataset.py`   |
| 5    | **Reward**: 9 reward deterministiche                           | `src/rewards/t2g_rewards.py`     |
| 6    | **GRPO Training**: TRL `GRPOTrainer`, G=4 completions/prompt   | `src/training/grpo_t2g_train.py` |
| 7    | **Save**: Checkpoint ogni 100 step + modello finale            | Auto                             |

### 2.3 Struttura del Codice

```text
neuro_symbolic_t2g/
├── src/
│   ├── datasets/          # ASLG-PC12 loader, bigram transition matrix
│   ├── grammar/           # GlossVocabularyMask, LogitsProcessor, PDA
│   ├── models/            # Model loader (Unsloth + LoRA)
│   ├── rewards/           # 9 reward functions
│   ├── training/          # GRPO trainer, SFT trainer, eval, callbacks
│   └── utils/             # Metrics, prompting, visualization, monitor
├── experiments/configs/   # 8+ YAML configs (optimal, experimental, ablation)
├── tests/                 # 37 pytest tests (conftest.py fixtures)
├── grammarllm/            # Vendored PDA library
└── cluster/               # SLURM scripts
```

---

## 3. Constrained Decoding

### 3.1 Vocabulary Mask (Default)

Il `GlossVocabularyLogitsProcessor` maschera a ogni step di generazione tutti i
token al di fuori del vocabolario ASL. Il modello può produrre **solo** token
del glossario (più EOS).

```python
# A ogni step:
logits[token_id] = -inf  # per tutti i token non nel vocabolario ASL
```

**Diagnostica W&B**: `MaskedMassTracker` traccia:

- `grammar/masked_mass_avg` — probabilità massicciata mascherata
- `grammar/full_entropy_avg` — entropia della distribuzione completa
- `grammar/allowed_entropy_avg` — entropia dei soli token permessi

### 3.2 PDA LL(1) (Sperimentale)

Il Pushdown Automaton mantiene uno **stack** che traccia lo stato corrente
nella grammatica LL(1). A ogni token:

1. Verifica se il token è valido nello stato corrente
2. Aggiorna lo stack (push/pop)
3. Restituisce i token validi per il prossimo step

```text
Vocab Mask                    PDA LL(1)
─────────────────────          ────────────────────────
Blocca TUTTI i token           Blocca SOLO token che violano
non nel glossario ASL          la grammatica LL(1) corrente

"WALK HOUSE BOOK"  → ✅        "WALK HOUSE BOOK"  → ✅
"the cat sleeps"   → ❌        "the cat sleeps"   → ❌
"IX IX IX IX"      → ✅ (!)    "IX IX IX IX"      → ❌ (rileva loop)
```

---

## 4. Sistema di Reward (9 Moduli)

Tutte le reward sono **deterministiche e rule-based** — nessun reward model neurale.

| #   | Reward                  | Formula                      | Range   | Peso (optimal) |
| --- | ----------------------- | ---------------------------- | ------- | -------------- |
| 1   | **Translation Quality** | ROUGE-L F1 vs gold gloss     | [0, 1]  | 0.30           |
| 2   | **Gold-Structure** ⭐   | exp(llm_avg - gold_avg)      | (0, 1]  | 0.20           |
| 3   | **Structural Dense**    | softmax(avg_log_prob / T)    | (0, 1)  | 0.10           |
| 4   | **Gloss Order**         | 1 - Levenshtein/max_len      | [0, 1]  | 0.10           |
| 5   | **Verifier-Scaled**     | ROUGE × gold_structure       | [0, 1]  | 0.10           |
| 6   | **Soft-Viterbi**        | exp(llm - soft_viterbi)      | (0, 1]  | 0.05           |
| 7   | **Viterbi**             | exp(llm - viterbi)           | (0, 1]  | 0.05           |
| 8   | **Format**              | vocab membership ratio       | [0, 1]  | 0.05           |
| 9   | **Repetition**          | unique_ratio / trigram_ratio | [-1, 1] | 0.05           |

### 4.1 Dettagli delle Reward

**Translation Quality (ROUGE-L)**: Misura la similarità lessicale F1 tra la gloss
generata e la gold reference. È il segnale semantico primario.

**Gold-Structure**: Confronta il bigram log-probability della sequenza generata
con quello della gold reference. Reward ≈ 1.0 → strutturalmente buono quanto
l'umano. Include OOV penalty per token fuori vocabolario.

**Structural Dense**: Bigram log-probability assoluto, normalizzato via softmax
con temperatura. Usa `normalize="softmax"` per evitare il collasso a 0 per
sequenze a bassa probabilità (problema della vecchia `exp` normalization).

**Gloss Order**: Distanza di Levenshtein word-level normalizzata. Complementa
ROUGE-L (che è un proxy di overlap lessicale) con un segnale sensibile
all'**ordine** dei token.

**Verifier-Scaled (RECIPE)**: Usa la plausibilità strutturale come moltiplicatore
di confidenza per la qualità di traduzione:

- Alto ROUGE + alta struttura → alto reward (match confidente)
- Alto ROUGE + bassa struttura → reward ridotto (match sospetto)
- Bassa struttura → reward basso (implausibile)

**Aggiornamento**: Per risolvere il tetto massimo di `~0.22` causato dal log-prob assoluto e dal sigmoide, la reward è stata riscritta per usare direttamente il valore di `gold_structure_reward(completion, gold_gloss, normalize=True)` come moltiplicatore di confidenza. Questo scala correttamente il moltiplicatore a `1.0` quando la struttura è perfetta rispetto alla reference, sbloccando l'intero intervallo di reward `[0, 1]`.

**Soft-Viterbi**: Viterbi differentiable via forward-backward (log-partition).
Ispirato a ViterbiPlanNet's DVL. Più smooth e tight del Viterbi hard.

**Viterbi (Hard)**: Confronta il path LLM con il Viterbi optimum (con diversity
constraints: self-loop penalty + iterative token ban per evitare loop degenerativi).

**Format**: Verifica che ogni token sia nel vocabolario ASL. Penalizza free text,
JSON, token concatenati >25 caratteri.

**Repetition**: Penalizza loop degenerativi (unique_ratio < 0.3 → -1.0).

### 4.2 Combinazione

$$R_{total} = \sum_{i=1}^{9} w_i \cdot R_i$$

I pesi sono configurabili via YAML. La config `grpo_experimental_all.yaml`
attiva tutti i 9 moduli con pesi bilanciati (somma = 1.0).

---

## 5. Training

### 5.1 GRPO

- **Algoritmo**: Group Relative Policy Optimization (TRL 0.24.0)
- **Generations**: G=4 completions per prompt
- **KL penalty**: beta=0.04
- **Temperature**: 0.7 (exploration)
- **LoRA**: r=32 (optimal) o r=16 (default), alpha=64/32
- **Quantization**: 4-bit QLoRA
- **Batch**: 1 × grad_accum=8 (effective batch=8)
- **Learning rate**: 5e-6
- **Steps**: 1500 (~2-3 ore su L40S)

### 5.2 SFT (Baseline)

- **Algoritmo**: Supervised Fine-Tuning (teacher forcing)
- **Epoche**: 3
- **Learning rate**: 2e-5
- **Batch**: 4 × grad_accum=4
- **Scopo**: Baseline supervisionata per misurare il guadagno del RL

### 5.3 W&B Integration

- **Modalità**: Offline (cluster senza internet)
- **console_multipart**: True (log output.log come artifact)
- **Crash safety**: try/finally around training
- **Tags**: per run (es. `grpo`, `optimal`, `ablation`)
- **Comparison plots**: baseline vs GRPO
- **JSON artifacts**: `comparison.json` con metriche comparative

---

## 6. Evaluation

### 6.1 Metriche

- **ROUGE-L F1**: Similarità lessicale con gold gloss
- **BLEU**: N-gram precision
- **Pass@1**: Rate di match esatto
- **Pass@K**: Rate di match con K campioni
- **Validity**: Percentuale di output con solo token validi
- **Reward Breakdown**: Score per ogni componente di reward

### 6.2 Best-of-N Selection

```bash
uv run python -m src.training.eval_t2g \
    --config experiments/configs/t2g/grpo_optimal.yaml \
    --checkpoint experiments/checkpoints/grpo/t2g/qwen05/final \
    --best-of-n --num-samples 5
```

Genera N campioni per prompt e seleziona il migliore per reward.

### 6.3 Comparison Mode

```bash
uv run python -m src.training.eval_t2g \
    --config experiments/configs/t2g/grpo_optimal.yaml \
    --checkpoint experiments/checkpoints/grpo/t2g/qwen05/final \
    --compare
```

Valuta automaticamente baseline (modello base) e GRPO, genera plot comparativi
e JSON report.

---

## 7. Configurazioni Sperimentali

### 7.1 Ablation Matrix

| #   | Config                  | Training | Grammar    | Reward             | Scopo                           |
| --- | ----------------------- | -------- | ---------- | ------------------ | ------------------------------- |
| 1   | `grpo_optimal`          | GRPO     | Vocab Mask | 9 reward           | Config ottimale                 |
| 2   | `grpo_experimental_all` | GRPO     | Vocab Mask | 9 reward (uniform) | Full reward ablation            |
| 3   | `grpo_qwen05`           | GRPO     | Vocab Mask | 4 reward           | Main training                   |
| 4   | `sft`                   | SFT      | Vocab Mask | —                  | Baseline supervisionata         |
| 5   | `zero_shot`             | ❌       | ❌         | —                  | Lower bound                     |
| 6   | `zero_shot_grammar`     | ❌       | Vocab Mask | —                  | Grammar senza training          |
| 7   | `grpo_no_grammar`       | GRPO     | ❌         | 4 reward           | GRPO senza constrained decoding |
| 8   | `grpo_pda`              | GRPO     | PDA LL(1)  | 4 reward           | GRPO + PDA                      |
| 9   | `grpo_soft_viterbi`     | GRPO     | Vocab Mask | +soft_viterbi      | Soft Viterbi ablation           |
| 10  | `grpo_verifier_scaled`  | GRPO     | Vocab Mask | +verifier_scaled   | Verifier ablation               |
| 11  | `grpo_no_sft`           | GRPO     | Vocab Mask | 6 reward           | SFT ablation                    |

### 7.2 Config Optimal (`grpo_optimal.yaml`)

- LoRA r=32 (doppio del default)
- 9 reward weights bilanciati
- `evaluation.max_samples: 500`
- `evaluation.num_samples: 5`
- `verifier_temperature: 5.0`

### 7.3 Config Experimental (`grpo_experimental_all.yaml`)

- Tutti i 9 moduli reward attivi
- Pesi uniformi (~0.10-0.15)
- Scopo: verificare se combinare tutti i segnali migliora o confonde il training

---

## 8. Test Suite

### 8.1 Risultati

| Test File             | Tests  | Status   |
| --------------------- | ------ | -------- |
| `test_data.py`        | 4      | ✅       |
| `test_grammar.py`     | 7      | ✅       |
| `test_rewards.py`     | 9      | ✅       |
| `test_metrics.py`     | 6      | ✅       |
| `test_monitor.py`     | 6      | ✅       |
| `test_integration.py` | 5      | ✅       |
| **TOTAL**             | **37** | **100%** |

### 8.2 Infrastruttura

- **Framework**: pytest con `conftest.py` fixtures condivise
- **Fixtures**: `reward_setup` (mini vocab+bigram), `dataset` (ASLG-PC12),
  `tokenizer` (Qwen o gpt2 fallback)
- **Runner**: `uv run python -m pytest tests/ -v`
- **Skip offline**: `bash tests/run_all_tests.sh --skip-data`

---

## 9. Risultati Attesi

### 9.1 Training Progress

| Fase       | Steps    | ROUGE-L | Comportamento                                                 |
| ---------- | -------- | ------- | ------------------------------------------------------------- |
| Iniziale   | 0–200    | 0.0–0.1 | Random/copying — decoder vincola i token ma output è nonsense |
| Intermedio | 200–800  | 0.2–0.4 | Associa gloss token a significato. Struttura bigram migliora. |
| Avanzato   | 800–1500 | 0.5–0.7 | Traduzioni ragionevolmente accurate. Pattern ASL appresi.     |

### 9.2 Ablation Hypotheses

| Config                  | Ipotesi                                                                        |
| ----------------------- | ------------------------------------------------------------------------------ |
| `zero_shot`             | Lower bound — modello base senza vincoli produce free text                     |
| `zero_shot_grammar`     | Il solo constrained decoding migliora validità ma non ROUGE-L                  |
| `grpo_no_grammar`       | GRPO senza vincoli: il modello può divergere o produrre output non strutturati |
| `grpo_pda`              | PDA produce sequenze più strutturate della vocab mask, meno ripetizioni        |
| `sft`                   | Baseline supervisionata — teacher forcing impara a replicare gold              |
| `grpo_optimal`          | Config ottimale con tutti i segnali reward bilanciati                          |
| `grpo_experimental_all` | Full reward ablation — verifica se 9 segnali aiutano o confondono              |

### 9.3 Risultati Ottenuti

I risultati completi saranno disponibili dopo l'esecuzione sul cluster.
Il sistema è pronto per il deployment:

- ✅ Tutti i 9 moduli reward verificati funzionanti
- ✅ 37/37 test passano
- ✅ 8+ config YAML pronte
- ✅ Pipeline cluster (SLURM) configurata
- ✅ W&B integration con crash safety
- ✅ Evaluation con best-of-N e comparison mode

---

## 10. Innovazioni Chiave

1. **Neuro-Symbolic Architecture**: Constrained decoding + reward simboliche
   eliminano la necessità di un reward model neurale.

2. **9 Reward Modules**: Diversi segnali strutturali (bigram, Viterbi, edit-distance,
   verifier-scaled) che coprono aspetti complementari della qualità della gloss.

3. **Verifier-Scaled Reward (RECIPE-inspired)**: Usa la plausibilità strutturale
   come moltiplicatore di confidenza per la qualità di traduzione. Aggiornato per
   usare `gold_structure_reward` come moltiplicatore relativo per evitare la compressione
   del punteggio (risolto il cap a 0.22, permettendo al segnale di salire fino a 1.0).

4. **Soft-Viterbi (Differentiable)**: Forward-backward log-partition come
   upper bound differentiable, ispirato a ViterbiPlanNet's DVL.

5. **Viterbi Diversity**: Self-loop penalty + iterative token ban per evitare
   che il Viterbi optimum degeneri in loop ripetitivi.

6. **Robust Gold Gloss Lookup**: SHA256 hashing delle user instructions per
   lookup format-agnostic, indipendente dal formato del prompt di TRL.

7. **Centralized Prompting**: `build_t2g_prompt()` garantisce prompt
   byte-identici tra training, eval e test.

8. **Best-of-N + Comparison Eval**: Evaluation mode che genera N campioni
   e confronta baseline vs GRPO con plot e JSON report.

---

## 11. Dipendenze e Ambiente

### 11.1 Core

- Python 3.10+
- PyTorch (CUDA 12.1+)
- Transformers + TRL 0.24.0
- Unsloth (GPU acceleration)
- rouge-score, numpy, pandas, plotnine

### 11.2 GPU

- L40S (8.9): ✅ Ideal
- V100 (7.0): ✅ No bf16
- K80 (3.7): ❌ fp16 only

### 11.3 Package Manager

- `uv` per gestione dipendenze e virtual environment
- `pyproject.toml` con optional GPU extras (`unsloth`, `vllm`)

---

## 12. Stato del Progetto

| Componente           | Stato       | Note                                                         |
| -------------------- | ----------- | ------------------------------------------------------------ |
| Data pipeline        | ✅ Completo | ASLG-PC12 loader, bigram matrix, vocab extraction            |
| Constrained decoding | ✅ Completo | Vocab mask + PDA (sperimentale)                              |
| Reward system        | ✅ Completo | 9 moduli, tutti verificati                                   |
| GRPO training        | ✅ Completo | TRL integration, crash-safe W&B                              |
| SFT baseline         | ✅ Completo | Teacher forcing baseline                                     |
| Evaluation           | ✅ Completo | ROUGE-L, BLEU, Pass@K, best-of-N, --compare                  |
| Test suite           | ✅ Completo | 37/37 pytest pass                                            |
| Config system        | ✅ Completo | 8+ YAML, ablation matrix                                     |
| Cluster pipeline     | ✅ Completo | SLURM scripts, monitor, aliases                              |
| Documentation        | ✅ Completo | README, REWARDS.md, METRICS.md, CONFIGS.md, CONFIGS_GUIDE.md |

### 12.1 Pronto per Cluster

Il sistema è pronto per l'esecuzione sul cluster:

```bash
# Upload
.\sync_cluster.ps1 -Action upload

# Setup
ssh user@gcluster.dmi.unict.it
cd ~/neuro_symbolic_t2g && bash cluster/setup.sh

# Run
source cluster/aliases.sh
t2g-run-all    # Train → Eval pipeline
t2g-monitor    # Live dashboard
```

---

## 13. Conclusioni

Il progetto **neuro_symbolic_t2g** implementa un'architettura neuro-symbolica
completa per la traduzione Text-to-Gloss ASL, combinando:

- **GRPO** per reinforcement learning on-policy
- **Constrained decoding** per garantire output sintatticamente validi
- **9 reward simboliche deterministiche** che coprono qualità di traduzione,
  struttura, ordine, formato, e ripetizione

Tutti i componenti sono implementati, testati (37/37 pytest pass), e pronti
per l'esecuzione su cluster. Il sistema supporta ablation study completa con
8+ configurazioni, evaluation con best-of-N e comparison mode, e logging
W&B crash-safe.

Il prossimo step è l'esecuzione sul cluster per ottenere risultati quantitativi
e confrontare le configurazioni ablation.

---

## Riferimenti

- **ASLG-PC12**: Othman & Jemni (2012), English-ASL Gloss Parallel Corpus
- **GRPO**: TRL GRPOTrainer (HuggingFace)
- **Unsloth**: FastLanguageModel per QLoRA acceleration
- **RECIPE**: arXiv:2605.19976 — Verifier-scaled reward
- **ViterbiPlanNet DVL**: arXiv:2603.04265 — Differentiable Viterbi
- **grammarllm**: Vendored PDA-based constrained decoding library
