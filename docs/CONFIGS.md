# T2G Config Matrix — Ablation Study

8 config YAML che coprono il piano di ablation **Training × Grammar** per Text-to-Gloss (T2G),
più una config **optimal** e una **experimental** con tutti i 9 moduli reward attivi.
Tutti usano **Qwen2.5-0.5B-Instruct** (4-bit, LoRA, 1500 step / 3 epoche).

**Quick reference:**

| #   | Config                            | Training | Grammar       | Eval | Cosa testa                                           |
| --- | --------------------------------- | -------- | ------------- | ---- | ---------------------------------------------------- |
| 1   | `t2g/grpo_optimal.yaml`           | ✅ GRPO  | ✅ Vocab Mask | ✅   | **Optimal** — LoRA r=32, 9 reward weights bilanciati |
| 2   | `t2g/grpo_experimental_all.yaml`  | ✅ GRPO  | ✅ Vocab Mask | ✅   | **Experimental** — tutti i 9 moduli reward attivi    |
| 3   | `t2g/grpo_qwen05.yaml`            | ✅ GRPO  | ✅ Vocab Mask | ✅   | **Main training** — GRPO + gloss vocabulary mask     |
| 4   | `t2g/sft.yaml`                    | ✅ SFT   | ✅ Vocab Mask | ✅   | Baseline supervisionata via teacher forcing          |
| 5   | `ablation/zero_shot.yaml`         | ❌       | ❌            | ✅   | Modello base grezzo (nessun vincolo)                 |
| 6   | `ablation/zero_shot_grammar.yaml` | ❌       | ✅ Vocab Mask | ✅   | Modello base + vincoli (no training)                 |
| 7   | `ablation/grpo_no_grammar.yaml`   | ✅ GRPO  | ❌            | ✅   | GRPO senza constrained decoding                      |
| 8   | `ablation/grpo_pda.yaml`          | ✅ GRPO  | ✅ PDA LL(1)  | ✅   | GRPO + Pushdown Automaton (grammatica completa)      |

## Ablation Matrix

```
                 No Grammar        Vocab Mask         PDA (LL1)
                 ───────────       ───────────        ──────────
No Training      zero_shot         zero_shot_grammar       —
GRPO Training    grpo_no_grammar   grpo_qwen05         grpo_pda
SFT Baseline                       sft
```

## Sezioni condivise da tutti

| Sezione      | Contenuto                                                                                                                                | Note                                                               |
| ------------ | ---------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------ |
| `model`      | `name`, `num_gpus`, `quantization`, `dtype`, `use_unsloth`, `fast_inference`, `max_seq_length`, `gpu_memory_utilization`, `vllm_standby` | `fast_inference: false` sempre (incompatibile con LogitsProcessor) |
| `dataset`    | `dataset_name`, `dataset_cache`, `vocab_path`, `bigram_matrix_path`, `split`, `max_samples`, `seed`, `thinking: false`                   | T2G non usa `<think>` blocks                                       |
| `wandb`      | `project`, `run_name`, `tags`                                                                                                            | `neuro-symbolic-t2g`                                               |
| `lora`       | `r: 16`, `lora_alpha: 32`, `lora_dropout: 0`, `target_modules`, `task_type`, `random_state: 3407`                                        | Solo training configs; eval-only ne sono privi                     |
| `curriculum` | `enabled: false`                                                                                                                         | Disabilitato per Phase 1                                           |
| `evaluation` | `batch_size: 8`                                                                                                                          | Solo training configs                                              |

---

## Differenze chiave

### 1. `t2g/grpo_qwen05.yaml` — **Main Training**

```
training:     GRPO, 1500 step, lr=5e-6, batch=1, grad_accum=8
grammar:      enabled: true, use_grammarllm_pda: false
              → Vocab Mask: blocca tutti i token non nel glossario ASL
reward:       weight_translation=0.40, weight_gold_structure=0.40,
              weight_format=0.10, weight_repetition=0.10
grpo:         num_generations=4, beta=0.04, temperature=0.7
```

**Cosa fa:** Training GRPO standard con constrained decoding via vocabulary mask.
A ogni step di generazione, il modello può produrre SOLO token del glossario ASL
(più EOS). Le reward misurano ROUGE-L rispetto alla gold gloss, plausibilità
strutturale via bigram, formato gloss-only, e penalità di ripetizione.

**Quando usarlo:** Training principale, confronto con ablation.

---

### 2. `t2g/sft.yaml` — **SFT Baseline**

```
training:     SFT, 3 epoche, lr=2e-5, batch=4, grad_accum=4, trainer: sft
grammar:      enabled: true, use_grammarllm_pda: false
              → Vocab Mask (eval-only, non durante training)
generation:   max_completion_length=256, temperature=0.7
              → Usato da eval_t2g.py (non da SFTTrainer)
reward:       weight_translation=0.40, weight_gold_structure=0.40,
              weight_format=0.10, weight_repetition=0.10
```

**Differenze da grpo_qwen05:**

- `training.trainer: sft` → il bootstrap (`__main__.py`) carica `sft_train.py` invece di `grpo_t2g_train.py`
- `generation` invece di `grpo` — parametri di eval, non di training
- `num_train_epochs: 3` invece di `max_steps: 1500`
- Batch più grande (4×4 vs 1×8), lr più alta (2e-5 vs 5e-6)
- Nessun `num_generations` o `beta` (non sono parametri SFT)

**Cosa fa:** Teacher forcing sulle gold gloss. Il modello impara a replicare
le sequenze di riferimento senza reward shaping. Serve come baseline
supervisionata per misurare il guadagno del RL.

---

### 3. `ablation/zero_shot.yaml` — **Base Model, No Grammar**

```
training:     ❌ assente (eval-only)
grammar:      enabled: false
              → Generazione completamente libera (zero vincoli)
grpo:         max_completion_length=256, temperature=0.7
              → Solo parametri di generazione per eval
reward:       weight_translation=1.0
              → Solo ROUGE-L (non serve struttura senza training)
```

**Cosa fa:** Valuta il modello base Qwen2.5-0.5B-Instruct **senza alcun training**
e **senza constrained decoding**. Il modello genera testo libero — ci si aspetta
output in linguaggio naturale, non gloss ASL.

**Ipotesi ablation:** Se le reward/grammatica non servono, il modello base
dovrebbe già performare bene. (Spoiler: non lo farà — testa il lower bound.)

---

### 4. `ablation/zero_shot_grammar.yaml` — **Base Model + Grammar**

```
training:     ❌ assente (eval-only)
grammar:      enabled: true, use_grammarllm_pda: false
              → Vocab Mask ATTIVA durante eval
grpo:         max_completion_length=256, temperature=0.7
reward:       weight_translation=1.0
```

**Differenze da zero_shot.yaml:** Unica differenza: `grammar.enabled: true`.

**Cosa fa:** Stesso modello base, ma con vocabulary mask durante la generazione.
Misura quanto il solo constrained decoding (senza training) migliora la qualità
delle gloss generate rispetto al modello completamente libero.

**Ipotesi ablation:** Il constrained decoding da solo dovrebbe eliminare
output nonsense (free text, JSON, reasoning) e forzare token del glossario,
aumentando ROUGE-L e validità rispetto a zero_shot puro.

---

### 5. `ablation/grpo_no_grammar.yaml` — **GRPO Without Grammar**

```
training:     GRPO, 1500 step (identico a grpo_qwen05)
grammar:      enabled: false
              → NESSUN constrained decoding durante i rollout
reward:       weight_translation=0.40, weight_gold_structure=0.40,
              weight_format=0.10, weight_repetition=0.10
grpo:         num_generations=4, beta=0.04, temperature=0.7
```

**Differenze da grpo_qwen05:** Unica differenza: `grammar.enabled: false`.

**Cosa fa:** Training GRPO standard ma **senza** vocabulary mask durante
i rollout. Il modello esplora liberamente lo spazio dei token, guidato
solo dalle reward. Le reward di formato e struttura dovrebbero comunque
spingerlo verso output simili a gloss.

**Ipotesi ablation:** Senza constrained decoding, il modello potrebbe
divergere o produrre output meno strutturati. Misura l'effetto incrementale
della vocabulary mask sul training GRPO.

---

### 6. `ablation/grpo_pda.yaml` — **GRPO + Pushdown Automaton (LL(1))**

```
training:     GRPO, 1500 step (identico a grpo_qwen05)
grammar:      enabled: true, use_grammarllm_pda: true, pda_temperature: 1.0
              → PDA LL(1) COMPLETO via grammarllm
reward:       weight_translation=0.40, weight_gold_structure=0.40,
              weight_format=0.10, weight_repetition=0.10
grpo:         num_generations=4, beta=0.04, temperature=0.7
```

**Differenze da grpo_qwen05:**

- `use_grammarllm_pda: true` → attiva il Pushdown Automaton
- `pda_temperature: 1.0` → temperatura per lo scaling dei logit nel PDA

**Cosa fa — il PDA spiegato:**

```
Vocab Mask (default)           PDA LL(1) (grpo_pda)
─────────────────────          ────────────────────────
Blocca TUTTI i token           Blocca SOLO i token che violano
non nel glossario ASL          la grammatica LL(1) corrente

"WALK HOUSE BOOK"  → ✅        "WALK HOUSE BOOK"  → ✅
"the cat sleeps"   → ❌        "the cat sleeps"   → ❌
"IX IX IX IX"      → ✅ (!)    "IX IX IX IX"      → ❌ (rileva loop)
"WALK WALK WALK"   → ✅ (!)    "WALK WALK WALK"   → ❌ (rileva loop)

Semplice: set statico         Complesso: automa a stati che sa
di token allowed              "dove si trova" nella grammatica
```

Il PDA (Pushdown Automaton) mantiene uno **stack** che traccia lo stato
corrente nella grammatica LL(1). A ogni token generato:

1. Il PDA verifica se il token è valido nello stato corrente
2. Aggiorna lo stack (push/pop in base alle produzioni grammaticali)
3. Restituisce la lista dei token validi per il prossimo step

La grammatica è costruita da `gloss_grammar.py`:

```
S* → BOS S EOS
S  → GLOSS S*         (continua con un altro gloss)
S  → ε                (termina)
```

Dove `GLOSS` è un non-terminale che espande in TUTTI i token del glossario
(tranne BOS e UNK).

**Vantaggi del PDA:**

- Rileva sequenze degenerate (loop, ripetizioni infinite)
- Può estendersi a vincoli più complessi (es. ordering, agreement)
- Fornisce una baseline superiore per il constrained decoding "forte"

**Svantaggi:**

- Più lento (overhead del PDA a ogni step)
- Richiede `grammarllm` come dipendenza
- Overkill per la maggior parte dei task di gloss (basta la vocab mask)

**Ipotesi ablation:** Il PDA dovrebbe produrre sequenze più strutturate
della vocab mask, con meno ripetizioni. La differenza dovrebbe essere
visibile nelle metriche di repetition e validità.

---

## Cosa manca / Prossimi passi

| Variante                 | Stato           | Note                                                                |
| ------------------------ | --------------- | ------------------------------------------------------------------- |
| `notthink` vs `think`    | Non applicabile | T2G non usa `<think>` blocks                                        |
| Curriculum learning      | Rimandato       | Tutti `curriculum.enabled: false`                                   |
| Multi-modello            | Da aggiungere   | Altri modelli oltre Qwen2.5-0.5B                                    |
| Viterbi diversity tuning | Esplorato       | `verifier_temperature` in `grammar.viterbi_diversity` (default 5.0) |

---

## Config Aggiuntive

### `t2g/grpo_optimal.yaml` — **Optimal Config**

```
training:     GRPO, 1500 step, lr=5e-6, batch=1, grad_accum=8
lora:         r=32, lora_alpha=64 (vs r=16 default)
grammar:      enabled: true, use_grammarllm_pda: false
reward:       weight_translation=0.30, weight_gold_structure=0.20,
              weight_structure=0.10, weight_gloss_order=0.10,
              weight_verifier_scaled=0.10, weight_soft_viterbi=0.05,
              weight_viterbi=0.05, weight_format=0.05, weight_repetition=0.05
evaluation:   max_samples=500, num_samples=5, best_of_n=false
grpo:         num_generations=4, beta=0.04, temperature=0.7
```

**Cosa fa:** Config "ottimale" con LoRA r=32 (doppio del default) e tutti i 9
moduli reward attivi con pesi bilanciati. Il peso maggiore va a translation
(ROUGE-L) e gold-structure, con contributi minori dai moduli strutturali
e di alignment.

---

### `t2g/grpo_experimental_all.yaml` — **Experimental Config (All 9 Modules)**

```
training:     GRPO, 1500 step, lr=5e-6, batch=1, grad_accum=8
lora:         r=32, lora_alpha=64
grammar:      enabled: true, use_grammarllm_pda: false
reward:       weight_translation=0.15, weight_gold_structure=0.15,
              weight_gloss_order=0.10, weight_verifier_scaled=0.10,
              weight_soft_viterbi=0.10, weight_viterbi=0.05,
              weight_structure=0.05, weight_format=0.10, weight_repetition=0.10
              (sum=1.0)
evaluation:   max_samples=500, num_samples=5, best_of_n=false
grpo:         num_generations=4, beta=0.04, temperature=0.7
```

**Cosa fa:** Config sperimentale che attiva TUTTI i 9 moduli reward con pesi
uniformi (~0.10-0.15 ciascuno). Serve per ablation del full reward space:
verifica se combinare tutti i segnali strutturali migliora o confonde
il training rispetto alla config optimal (dove solo 4 moduli sono attivi).

**Verifica moduli:** Tutti i 9 moduli sono stati verificati funzionanti
via test suite (37/37 pytest pass).
