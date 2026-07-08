# Config YAML — Guida ai Config di Training

**Ultimo aggiornamento**: 8 luglio 2026

Questa guida spiega le differenze tra i 9 config YAML disponibili in
`experiments/configs/t2g/`, quando usarli e in che ordine.

---

## 1. Panoramica dei Config

| #   | Config                               | Tipo      | SFT Pre-train |   Grammar   | Reward Attive                                                                      | Somma Pesi |
| --- | ------------------------------------ | --------- | :-----------: | :---------: | ---------------------------------------------------------------------------------- | :--------: |
| 1   | `grpo_optimal.yaml`                  | GRPO      |      ✅       |    Trie     | translation + gold_structure + gloss_order + verifier_scaled + format + repetition |    1.00    |
| 2   | `grpo_qwen05.yaml`                   | GRPO      |      ✅       |    Trie     | translation + gold_structure + gloss_order + format + repetition                   |    1.00    |
| 3   | `sft.yaml`                           | SFT       |       —       | Trie (eval) | translation + gold_structure + gloss_order + format + repetition                   |    1.00    |
| 4   | `ablation/grpo_no_grammar.yaml`      | GRPO      |      ❌       |     ❌      | translation + gold_structure + format + repetition                                 |    1.00    |
| 5   | `ablation/grpo_pda.yaml`             | GRPO      |      ❌       |     PDA     | translation + gold_structure + format + repetition                                 |    1.00    |
| 6   | `ablation/grpo_soft_viterbi.yaml`    | GRPO      |      ✅       |    Trie     | translation + soft_viterbi + gloss_order + format + repetition                     |    1.00    |
| 7   | `ablation/grpo_verifier_scaled.yaml` | GRPO      |      ✅       |    Trie     | verifier_scaled + gloss_order + format + repetition                                |    1.00    |
| 8   | `ablation/zero_shot.yaml`            | Eval-only |      ❌       |     ❌      | translation (1.0)                                                                  |    1.00    |
| 9   | `ablation/zero_shot_grammar.yaml`    | Eval-only |      ❌       |    Trie     | translation (1.0)                                                                  |    1.00    |

---

## 2. Differenze Dettagliate

### 2.1 Config Principali (Produzione)

#### `grpo_optimal.yaml` — ⭐ Config Ottimale (consigliato)

Il config più completo e bilanciato. Combina tutte le migliorie introdotte:

- **LoRA r=32** (doppio del default r=16) per maggiore capacità
- **SFT pre-training** abilitato (1 epoch, impara il formato gloss)
- **GRPO 2000 step** (più lungo del default 1500)
- **num_generations=8** (più alto del default 4, migliore stima del vantaggio)
- **beta=0.02** (KL penalty basso, più esplorazione)
- **temperature=0.8** (più alta, più diversità nei rollout)
- **6 reward attive**: translation + gold_structure + gloss_order + verifier_scaled + format + repetition
- **verifier_gamma=1.5** (bilanciato tra strict e permissivo)
- **Evaluation**: 500 campioni, 5 completions/prompt per Pass@k

#### `grpo_qwen05.yaml` — Config Base (default)

Config di riferimento per GRPO. Più leggero e veloce di `grpo_optimal`:

- **LoRA r=16** (standard)
- **SFT pre-training** abilitato (1 epoch)
- **GRPO 1500 step** (standard)
- **num_generations=4** (standard)
- **beta=0.04** (KL penalty standard)
- **temperature=0.7** (standard)
- **5 reward attive**: translation + gold_structure + gloss_order + format + repetition
  (manca verifier_scaled rispetto all'ottimale)
- **verifier_gamma=1.0** (lineare)
- **max_samples=20000** (sottoinsieme del dataset, non tutto)
- **Evaluation**: solo batch_size=8 (nessun max_samples specificato)

#### `sft.yaml` — SFT Baseline (per confronto)

Supervised Fine-Tuning puro, senza GRPO. Utile come baseline per il paper:

- **3 epoche** (invece di max_steps, tipico per SFT)
- **batch_size=4, grad_accum=4** (effective batch=16)
- **lr=2e-5** (più alto del GRPO, tipico per SFT)
- **max_seq_length=768** (più corto del GRPO)
- **5 reward attive** (solo per eval, non usate in training)
- **gradient_checkpointing** non abilitato (non serve con batch_size=4)
- **use_unsloth non specificato** (default false)

---

### 2.2 Config di Ablation (Studio Sperimentale)

Questi config servono a isolare l'impatto di singole componenti. Ognuno
cambia **una sola variabile** rispetto al config base `grpo_qwen05.yaml`.

#### `ablation/grpo_no_grammar.yaml` — Senza Constrained Decoding

- **grammar.enabled=false** (nessun vocabulary mask)
- **Niente SFT pre-training** (per isolare l'effetto del grammar)
- **4 reward** (senza gloss_order — config più vecchio)
- **max_completion_length=256** (più lungo, il modello può generare free text)
- **max_samples=null** (tutto il dataset)

**Scopo**: dimostrare che il constrained decoding è essenziale per evitare
garbage tokens. Confrontare con `grpo_qwen05.yaml`.

#### `ablation/grpo_pda.yaml` — Constrained Decoding via PDA

- **grammar.use_grammarllm_pda=true** (PDA completo invece del Trie)
- **pda_temperature=1.0** (scaling dei logit del PDA)
- **Niente SFT pre-training**
- **4 reward** (senza gloss_order)
- **max_completion_length=256**

**Scopo**: confrontare Trie dual-root (veloce) vs PDA LL(1) (più espressivo
ma più lento). Confrontare con `grpo_qwen05.yaml`.

#### `ablation/grpo_soft_viterbi.yaml` — Soft Viterbi Reward

- **weight_soft_viterbi=0.35** (sostituisce gold_structure)
- **weight_translation=0.30** (ridotto da 0.40)
- **SFT pre-training** abilitato
- **verifier_gamma=1.0**
- **max_completion_length=128** (più corto, come l'ottimale)

**Scopo**: testare la reward soft Viterbi (differentiable, forward-backward)
ispirata a ViterbiPlanNet DVL (arXiv:2603.04265) al posto di gold_structure.

#### `ablation/grpo_verifier_scaled.yaml` — Verifier-Scaled Reward

- **weight_verifier_scaled=0.65** (sostituisce translation + gold_structure)
- **Niente weight_translation** (azzerata)
- **Niente weight_gold_structure** (azzerata)
- **SFT pre-training** abilitato
- **verifier_gamma=2.0** (quadratico, più stricto)

**Scopo**: testare la reward verifier-scaled (RECIPE, arXiv:2605.19976)
che usa la plausibilità strutturale come moltiplicatore di confidenza
per la qualità di traduzione.

#### `ablation/zero_shot.yaml` — Zero-Shot Base Model

- **Niente training** (eval-only)
- **grammar.enabled=false**
- **weight_translation=1.0** (unica reward)
- **Config minimale**: niente lora, niente sft_pretrain, niente curriculum

**Scopo**: baseline assoluta. Mostra le prestazioni del modello base
Qwen2.5-0.5B senza alcun addestramento né constrained decoding.

#### `ablation/zero_shot_grammar.yaml` — Zero-Shot + Grammar

- **Niente training** (eval-only)
- **grammar.enabled=true** (Trie dual-root)
- **weight_translation=1.0** (unica reward)

**Scopo**: isolare l'impatto del constrained decoding sul modello base.
Confrontare con `zero_shot.yaml` per misurare il delta del grammar.

---

## 3. Tabella Comparativa dei Parametri Chiave

| Parametro              | `grpo_optimal` | `grpo_qwen05` | `sft` | `no_grammar` | `pda` | `soft_viterbi` | `verifier_scaled` | `zero_shot` | `zero_shot_grammar` |
| ---------------------- | :------------: | :-----------: | :---: | :----------: | :---: | :------------: | :---------------: | :---------: | :-----------------: |
| **LoRA r**             |       32       |      16       |  16   |      16      |  16   |       16       |        16         |      —      |          —          |
| **SFT pre-train**      |       ✅       |      ✅       |   —   |      ❌      |  ❌   |       ✅       |        ✅         |     ❌      |         ❌          |
| **max_steps**          |      2000      |     1500      |   —   |     1500     | 1500  |      1500      |       1500        |      —      |          —          |
| **num_generations**    |       8        |       4       |   —   |      4       |   4   |       4        |         4         |      —      |          —          |
| **beta (KL)**          |      0.02      |     0.04      |   —   |     0.04     | 0.04  |      0.04      |       0.04        |      —      |          —          |
| **temperature**        |      0.8       |      0.7      |   —   |     0.7      |  0.7  |      0.7       |        0.7        |     0.7     |         0.7         |
| **max_completion**     |      128       |      128      |  256  |     256      |  256  |      128       |        128        |     256     |         256         |
| **lr**                 |      3e-6      |     5e-6      | 2e-5  |     5e-6     | 5e-6  |      5e-6      |       5e-6        |      —      |          —          |
| **max_grad_norm**      |      0.05      |      0.1      |  1.0  |     0.1      |  0.1  |      0.1       |        0.1        |      —      |          —          |
| **warmup_steps**       |      200       |      150      |  100  |     150      |  150  |      150       |        150        |      —      |          —          |
| **grammar.enabled**    |       ✅       |      ✅       |  ✅   |      ❌      |  ✅   |       ✅       |        ✅         |     ❌      |         ✅          |
| **use_grammarllm_pda** |       ❌       |      ❌       |  ❌   |      ❌      |  ✅   |       ❌       |        ❌         |     ❌      |         ❌          |
| **verifier_gamma**     |      1.5       |      1.0      |   —   |      —       |   —   |      1.0       |        2.0        |      —      |          —          |
| **max_samples**        |      null      |     20000     | null  |     null     | null  |     20000      |       20000       |      —      |          —          |
| **use_unsloth**        |       ✅       |      ✅       |  ❌   |      ❌      |  ❌   |       ✅       |        ✅         |     ❌      |         ❌          |

---

## 4. Quale Config Usare e in Quale Ordine

### 4.1 Pipeline di Produzione (per ottenere i migliori risultati)

```
1. grpo_optimal.yaml   →   Training completo + Evaluation
```

Questo è l'unico config che serve per ottenere i risultati migliori.
Esegue automaticamente: SFT pre-training → GRPO → Evaluation.

```bash
CONFIG=experiments/configs/t2g/grpo_optimal.yaml sbatch cluster/run_all.sh
```

### 4.2 Pipeline di Sviluppo (iterazione veloce)

Se vuoi iterare velocemente senza usare tutto il dataset:

```
1. grpo_qwen05.yaml    →   Training veloce (20000 samples, 1500 step)
```

```bash
CONFIG=experiments/configs/t2g/grpo_qwen05.yaml sbatch cluster/run_all.sh
```

### 4.3 Pipeline di Ricerca (per il paper)

Per un ablation study completo, esegui nell'ordine:

```
 1. zero_shot.yaml              →   Baseline: modello base, niente grammar
 2. zero_shot_grammar.yaml      →   Baseline + grammar (delta del constrained decoding)
 3. sft.yaml                    →   SFT baseline (delta del supervised vs RL)
 4. grpo_no_grammar.yaml        →   GRPO senza grammar (delta del grammar nel RL)
 5. grpo_qwen05.yaml            →   GRPO + grammar (config base, riferimento)
 6. grpo_pda.yaml               →   GRPO + PDA (Trie vs PDA)
 7. grpo_soft_viterbi.yaml      →   GRPO + soft Viterbi (gold_structure vs soft_viterbi)
 8. grpo_verifier_scaled.yaml   →   GRPO + verifier-scaled (RECIPE ablation)
 9. grpo_optimal.yaml           →   Config ottimale (best result)
```

**Ordine logico**: dai config più semplici (zero-shot) a quelli più complessi
(ottimale), così ogni step aggiunge una componente e puoi misurare il delta.

**Comandi**:

```bash
# Zero-shot (eval-only, veloce)
python -m src.training.eval_t2g --config experiments/configs/t2g/ablation/zero_shot.yaml
python -m src.training.eval_t2g --config experiments/configs/t2g/ablation/zero_shot_grammar.yaml

# Training completo (catena train + eval)
for cfg in sft grpo_no_grammar grpo_qwen05 grpo_pda grpo_soft_viterbi grpo_verifier_scaled grpo_optimal; do
    CONFIG=experiments/configs/t2g/${cfg}.yaml sbatch cluster/run_all.sh
done
# Nota: per ablation/, usa il path completo:
# CONFIG=experiments/configs/t2g/ablation/grpo_no_grammar.yaml sbatch cluster/run_all.sh
```

### 4.4 Solo Evaluation (modello già addestrato)

Se hai già un checkpoint e vuoi solo valutare:

```bash
python -m src.training.eval_t2g \
    --config experiments/configs/t2g/grpo_optimal.yaml \
    --checkpoint experiments/checkpoints/grpo/t2g/qwen25-05b-optimal/final \
    --plot
```

---

## 5. Verifica Completezza e Coerenza

### 5.1 `use_unsloth` — ✅ Allineato

| Config                 | `use_unsloth` | Note                                    |
| ---------------------- | :-----------: | --------------------------------------- |
| `grpo_optimal`         |    ✅ true    | Consigliato (più veloce, meno memoria)  |
| `grpo_qwen05`          |    ✅ true    | Consigliato                             |
| `sft`                  |    ✅ true    | Allineato (era mancante, default false) |
| `grpo_no_grammar`      |    ✅ true    | Allineato (era mancante)                |
| `grpo_pda`             |    ✅ true    | Allineato (era mancante)                |
| `grpo_soft_viterbi`    |    ✅ true    | OK                                      |
| `grpo_verifier_scaled` |    ✅ true    | OK                                      |
| `zero_shot`            |  — mancante   | OK (eval-only, non serve)               |
| `zero_shot_grammar`    |  — mancante   | OK (eval-only, non serve)               |

### 5.2 `weight_gloss_order` — ✅ Allineato

| Config                 | `weight_gloss_order` | Note                        |
| ---------------------- | :------------------: | --------------------------- |
| `grpo_optimal`         |         0.15         | ✅                          |
| `grpo_qwen05`          |         0.15         | ✅                          |
| `sft`                  |         0.15         | ✅                          |
| `grpo_no_grammar`      |         0.15         | ✅ Allineato (era mancante) |
| `grpo_pda`             |         0.15         | ✅ Allineato (era mancante) |
| `grpo_soft_viterbi`    |         0.15         | ✅                          |
| `grpo_verifier_scaled` |         0.15         | ✅                          |
| `zero_shot`            |        — n/a         | OK (eval-only, 1 reward)    |
| `zero_shot_grammar`    |        — n/a         | OK (eval-only, 1 reward)    |

### 5.3 `verifier_gamma` — ✅ Allineato

| Config                 | `verifier_gamma` | Note                        |
| ---------------------- | :--------------: | --------------------------- |
| `grpo_optimal`         |       1.5        | Bilanciato                  |
| `grpo_qwen05`          |       1.0        | Lineare (default)           |
| `sft`                  |       1.0        | ✅ Allineato (era mancante) |
| `grpo_no_grammar`      |       1.0        | ✅ Allineato (era mancante) |
| `grpo_pda`             |       1.0        | ✅ Allineato (era mancante) |
| `grpo_soft_viterbi`    |       1.0        | ✅                          |
| `grpo_verifier_scaled` |       2.0        | Quadratico (più stricto)    |
| `zero_shot`            |      — n/a       | OK (eval-only)              |
| `zero_shot_grammar`    |      — n/a       | OK (eval-only)              |

### 5.4 `max_completion_length` — Differenze intenzionali

| Config                 | Valore | Note                                      |
| ---------------------- | :----: | ----------------------------------------- |
| `grpo_optimal`         |  128   | ✅ Ottimale per gloss ASL (20-40 token)   |
| `grpo_qwen05`          |  128   | ✅                                        |
| `sft`                  |  256   | OK (SFT eval, più spazio per generazione) |
| `grpo_no_grammar`      |  256   | OK (senza grammar serve più spazio)       |
| `grpo_pda`             |  256   | OK (PDA può generare sequenze più lunghe) |
| `grpo_soft_viterbi`    |  128   | ✅                                        |
| `grpo_verifier_scaled` |  128   | ✅                                        |

### 5.5 `gradient_checkpointing` e `max_seq_length`

- **`gradient_checkpointing`**: usato solo in `sft_train.py` (non in GRPO).
  Tutti i 4 config con `sft_pretrain.enabled=true` lo hanno a `true`.
  I config GRPO non lo specificano (corretto, non è usato da `GRPOConfig`).
- **`max_seq_length`**: presente in `model` (tutti i config GRPO/SFT: 1024)
  e in `training` (solo `sft.yaml`: 768). Il codice legge `model.max_seq_length`
  per Unsloth e `training.max_seq_length` per SFTConfig. Coerente.

### 5.6 Somma dei pesi reward — ✅ Tutti a 1.00

| Config                 | Somma | Note |
| ---------------------- | :---: | ---- |
| `grpo_optimal`         | 1.00  | ✅   |
| `grpo_qwen05`          | 1.00  | ✅   |
| `sft`                  | 1.00  | ✅   |
| `grpo_no_grammar`      | 1.00  | ✅   |
| `grpo_pda`             | 1.00  | ✅   |
| `grpo_soft_viterbi`    | 1.00  | ✅   |
| `grpo_verifier_scaled` | 1.00  | ✅   |
| `zero_shot`            | 1.00  | ✅   |
| `zero_shot_grammar`    | 1.00  | ✅   |

---

## 6. Raccomandazioni

1. **Per produzione**: usa `grpo_optimal.yaml` — è il config più completo e
   bilanciato.

2. **Per sviluppo veloce**: usa `grpo_qwen05.yaml` — 20000 samples invece
   di 78939, 1500 step invece di 2000.

3. **Per il paper**: esegui tutti i 9 config nell'ordine della §4.3 per
   un ablation study completo.

> **✅ Tutti i config sono ora allineati e coerenti.** Le 3 incongruenze
> precedenti (`use_unsloth`, `weight_gloss_order`, `verifier_gamma`) sono
> state corrette. Tutti i 9 config passano la validazione e le somme dei
> pesi reward sono a 1.00.
