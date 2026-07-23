# Neuro-Symbolic T2G — Report di Ricerca

**Progetto**: Neuro-Symbolic Text-to-Gloss Translation via Constrained Decoding + GRPO
**Data**: Luglio 2026
**Modello**: Qwen2.5-0.5B-Instruct (4-bit QLoRA, LoRA r=32)
**Dataset**: ASLG-PC12 (87K coppie English→ASL Gloss)
**Codice**: `neuro_symbolic_t2g/`

---

## 1. Obiettivo del Progetto

Il progetto dimostra che un **LLM di piccola dimensione** (0.5B parametri)
può essere fine-tunato per tradurre testo inglese in **gloss ASL** (American Sign
Language) combinando tre paradigmi:

1. **Reinforcement Learning (GRPO)** — Group Relative Policy Optimization, che
   genera multiple completions per prompt e le confronta relazionalmente,
   eliminando la necessità di un critic separato (a differenza di PPO).
2. **Constrained Decoding Neurale** — un `LogitsProcessor` maschera a ogni step
   di generazione tutti i token al di fuori del vocabolario ASL (~15K token),
   garantendo output sintatticamente valido. Due strategie: **Trie dual-root**
   (default, leggero) o **PDA LL(1)** con token-boundary lookahead (grammarllm v0.5.0).
3. **Reward Symboliche Deterministiche** — 8 funzioni di reward basate su regole
   (ROUGE-L, BLEU-4, bigrammi, edit-distance, verifier-scaled, formato,
   ripetizione) mappate su range simmetrico [-1, 1], che sostituiscono un reward
   model neurale eliminando overhead e bias.

L'architettura **neuro-symbolica** combina l'apprendimento distribuito del modello
neurale con la garanzia formale del decoding vincolato e le reward simboliche.

---

## 2. Architettura del Sistema

### 2.1 Pipeline End-to-End

```text
┌──────────────┐    ┌──────────────────┮    ┌──────────────────────┐
│ English text │ →  │ Qwen2.5-0.5B     │ →  │ Constrained Decoder  │
│ "The man     │    │ + LoRA r=32      │    │ (Trie dual-root or  │
│  walks home" │    │ + GRPO training  │    │  PDA + lookahead)   │
└──────────────┘    └──────────────────┘    └──────────────────────┘
                                                       ↓
                                             ┌──────────────────────┐
                                             │ IX MAN WALK HOUSE    │
                                             │ (ASL gloss sequence) │
                                             └──────────────────────┘
```

### 2.2 Pipeline in 7 Step

| Step | Cosa                                                           | File                             |
| ---- | -------------------------------------------------------------- | -------------------------------- |
| 1    | **Data**: Download ASLG-PC12 (87K coppie) da HuggingFace       | `src/datasets/aslg_dataset.py`   |
| 2    | **Model**: Load Qwen2.5-0.5B + LoRA r=32 + 4-bit QLoRA via Unsloth | `src/models/model_loader.py`  |
| 3    | **Constrained Decoding**: Trie dual-root o PDA LL(1) + lookahead | `src/grammar/gloss_grammar.py`  |
| 4    | **Dataset**: Formattazione prompt-completion con chat template | `src/datasets/aslg_dataset.py`   |
| 5    | **Reward**: 8 reward simmetriche [-1, 1]                       | `src/rewards/t2g_rewards.py`      |
| 6    | **GRPO Training**: TRL `GRPOTrainer`, G=8 completions/prompt   | `src/training/grpo_t2g_train.py` |
| 7    | **Save**: Checkpoint ogni 100 step + modello finale            | Auto                             |

### 2.3 Struttura del Codice

```text
neuro_symbolic_t2g/
├── src/
│   ├── datasets/          # ASLG-PC12 loader, bigram transition matrix
│   ├── grammar/           # Trie mask, PDA logits processor, MaskedMassTracker
│   ├── models/            # Model loader (Unsloth + LoRA)
│   ├── rewards/           # 8 reward functions [-1, 1] symmetric
│   ├── training/          # GRPO trainer, SFT trainer, eval, callbacks
│   └── utils/             # Metrics, prompting, visualization, chain_monitor,
│       │                  #   ablation_summary (cross-config table + chart)
│       └── ...
├── experiments/configs/   # 12 YAML configs (optimal, experimental, 10 ablation)
├── tests/                 # 21 pytest tests
├── grammarllm/            # grammarllm v0.5.0 (PDA + lookahead + beam search)
├── cluster/               # SLURM scripts, chain watcher, aliases, monitor
└── docs/                  # Documentation
```

---

## 3. Constrained Decoding

### 3.1 Trie Dual-Root (Default)

Il `GlossVocabularyLogitsProcessor` usa un **Trie a due radici**:
- **Root**: token senza leading-space (primo token della generazione)
- **Space-root**: token con leading-space (inizio di una nuova gloss dopo uno spazio)

Questo enforce i boundary di whitespace tra le glosse, prevenendo la
concatenazione arbitraria di single-BPE-token (es. "DE"+"B"+"RE"+"CH"+"T" →
"DEBUTRECHT" invece di "DEBUTRECHT" come singolo token).

**Diagnostica W&B**: `MaskedMassTracker` traccia probabilità massicciata
mascherata ed entropia a ogni step.

### 3.2 PDA LL(1) + Token-Boundary Lookahead (grammarllm v0.5.0)

Il Pushdown Automaton mantiene uno **stack** che traccia lo stato corrente
nella grammatica LL(1). A ogni token:

1. Verifica se il token è valido nello stato corrente
2. Aggiorna lo stack (push/pop)
3. Restituisce i token validi per il prossimo step

**Token-boundary lookahead** (NUOVO in grammarllm v0.5.0): un `VocabTrie` sul
vocabolario del tokenizer permette al modello di emettere **token BPE nativi**
che attraversano i boundary grammaticali. Senza lookahead, il PDA forzava
emissioni single-BPE-token, frammentando gloss come `"DESC-NUMEROUS"` in
`["DESC", "-", "NUMEROUS"]`. Con lookahead, Qwen può emettere `["DESC-NUMEROUS"]`
come singolo BPE token, allineandosi alla tokenizzazione di pre-training.

**StatelessLogitsProcessor**: deriva lo stato del PDA dalla history dei token
(`input_ids`) ad ogni step (con cache LRU per O(1) amortized). Beam-search safe.

```text
Trie Dual-Root (default)        PDA + Lookahead (grammarllm v0.5.0)
─────────────────────          ─────────────────────────────────────
Blocca token non nel vocab      Blocca token che violano la grammatica
Whitespace boundary enforced    Native BPE token emission
Veloce, sufficiente per gloss   Più espressivo, beam search supportato
```

---

## 4. Sistema di Reward (8 Moduli, range [-1, 1])

Tutte le reward sono **deterministiche e rule-based** e mappate su range
**simmetrico [-1, 1]** (dove -1 = completamente sbagliato, 0 = neutro,
1 = perfetto). Nessun reward model neurale.

| #   | Reward                  | Formula                      | Peso (optimal v2.1) |
| --- | ----------------------- | ---------------------------- | ------------------- |
| 1   | **Translation Quality** | ROUGE-L F1 vs gold gloss     | 0.20                |
| 2   | **BLEU-4** ⭐ (NUOVO)   | sacrebleu sentence BLEU (effective_order + smoothing) | 0.20 |
| 3   | **Gold-Structure**      | exp(llm_avg - gold_avg)      | 0.20                |
| 4   | **Gloss Order**         | 1 - Levenshtein/max_len      | 0.10                |
| 5   | **Verifier-Scaled**     | ROUGE × gold_structure       | 0.10                |
| 6   | **Format**              | vocab membership ratio       | 0.10                |
| 7   | **Repetition**          | unique_ratio / trigram_ratio | 0.10                |
| 8   | (Ablation modules)      | soft_viterbi, viterbi, structure | 0.0 (commented) |

### 4.1 BLEU-4 Reward (T2G-Reasoner 2025)

Ispirata a T2G-Reasoner (2025) che mostra BLEU-4 superiore a ROUGE-L come segnale
reward per T2G GRPO. Usa `sacrebleu.BLEU` con:
- **`effective_order=True`**: le sequenze corte (1-3 token, comuni in ASL)
  vengono valutate sugli n-grammi disponibili invece di richiedere 4-grammi
  (che restituirebbero 0 → mapped a -1.0)
- **`smooth_method="floor"`**: previene il collasso del geometric mean a 0
  quando un ordine di n-grammi ha zero match

**Bug fix**: la versione precedente cachiava `_SACREBLEU_AVAILABLE=False` se
l'import falliva una volta (es. container Apptainer senza sacrebleu), uccidendo
silenziosamente il 20% del reward signal per l'intero run. Ora:
- Check eager in `build_t2g_reward_functions`: crash al config time con
  messaggio actionable se sacrebleu manca
- Import con try/except fallback in `eval_t2g.py`, `aslg_dataset.py`,
  `transition_matrix.py` per tolleranza container

### 4.2 Combinazione

$$R_{total} = \sum_{i=1}^{8} w_i \cdot R_i, \quad \sum w_i = 1.0$$

I pesi sono configurabili via YAML. La somma deve essere 1.0 (validato
automaticamente).

---

## 5. Training

### 5.1 GRPO (Config Optimal v2.1)

| Parametro | Valore | Note |
| --- | --- | --- |
| **Algoritmo** | GRPO (TRL 0.24.0) | Group Relative Policy Optimization |
| **Generations** | G=8 completions/prompt | Post-fix OOM (era 16, causava OOM su 22GB GPU) |
| **KL penalty** | beta=0.0 | DAPO-style: no KL al reference, solo PPO clip |
| **Temperature** | 0.9 | Più esplorazione per compensare beta=0 |
| **LoRA** | r=32, alpha=64 | Doppio del default r=16 |
| **Quantization** | 4-bit QLoRA | Via Unsloth |
| **Batch** | 1 × grad_accum=8 | Effective batch=8 |
| **Learning rate** | 3e-6 | Più basso per stabilità con LoRA r=32 |
| **Steps** | 2000 | Più lungo del default 1500 |
| **gradient_checkpointing** | true | OOM mitigation: ricomputa forward nel backward (~20% più lento) |
| **Curriculum** | 3-stage (G²RPO-A 2026) | Simple→medium→hard difficoltà progressiva |

### 5.2 OOM Fix

Il config optimal v2 raddoppiava `num_generations` da 8 a 16, causando OOM sul
cluster (GPU 22GB). Root cause: GRPO trattiene G completions in VRAM attraverso
generazione → recomputazione logprob → backward. Raddoppiare G raddoppia ~peak VRAM.

Fix (v2.1):
- `num_generations: 16 → 8` (valore provato sul cluster)
- `gradient_checkpointing: true` (extra safety margin nel backward)
- Tutte le altre migliorie v2 mantenute (curriculum, BLEU-4, beta=0, temp=0.9)

### 5.3 SFT Pre-training (Phase 0)

- **Algoritmo**: Supervised Fine-Tuning (teacher forcing)
- **Epoche**: 1 (sufficiente per imparare il formato gloss)
- **Learning rate**: 2e-5
- **Scopo**: Insegnare al modello il formato gloss prima del GRPO

### 5.4 Curriculum Learning (G²RPO-A 2026)

3-stage progressive difficulty basata sulla distribuzione reale di ASLG-PC12:
- **Stage 1** (0-33%): 10% simple, 65% medium, 25% hard
- **Stage 2** (33-66%): 5% simple, 40% medium, 55% hard
- **Stage 3** (66-100%): 3% simple, 30% medium, 67% hard

Implementato come `CurriculumFilteredDataset` (wrapper che reshuffle gli indici
senza copiare i dati) + `CurriculumCallback` (transizione stage ogni max_steps/3).

### 5.5 W&B Integration

- **Modalità**: Offline (cluster senza internet)
- **console_multipart**: True (log output.log in chunk, crash-safe)
- **Crash safety**: try/finally around training (wandb.finish sempre chiamato)
- **Comparison plots**: baseline vs GRPO per ogni config
- **JSON artifacts**: `comparison.json` con delta metriche

---

## 6. Evaluation

### 6.1 Metriche

- **ROUGE-L F1**: Similarità lessicale con gold gloss
- **BLEU-4**: N-gram precision (con effective_order + smoothing)
- **Pass@1**: Rate di match esatto
- **Pass@K**: Rate di match con K campioni
- **Validity**: Percentuale di output con solo token validi
- **Reward Breakdown**: Score per ogni componente di reward

### 6.2 Best-of-N Selection

Genera N campioni per prompt e seleziona il migliore (ROUGE-L più alto tra i validi).
Trasforma Pass@N in un Pass@1 più forte senza ulteriore training.

### 6.3 Comparison Mode

Valuta automaticamente baseline (zero-shot) e GRPO checkpoint, genera:
- `baseline_vs_grpo_comparison.png` — bar chart per config
- `comparison.json` — delta ROUGE-L, Pass@1, Exact Match, Validity

### 6.4 Ablation Summary (NUOVO)

Dopo l'ablation study completa, `ablation_summary.py` aggrega tutti i risultati:
- `ablation_summary.csv` — tabella machine-readable
- `ablation_summary.md` — tabella Markdown
- `ablation_comparison.png` — grafico a barre raggruppato cross-config

```bash
ablation-summary  # alias dopo source cluster/aliases.sh
```

---

## 7. Configurazioni Sperimentali (12 Config)

### 7.1 Ablation Matrix

| #   | Config                    | Training | Grammar          | Reward                          | Scopo                         |
| --- | ------------------------- | -------- | ---------------- | ------------------------------- | ----------------------------- |
| 1   | `grpo_optimal`            | GRPO+SFT | Trie             | 7 reward [-1,1] + BLEU-4       | Config ottimale v2.1          |
| 2   | `grpo_experimental_all`   | GRPO+SFT | Trie             | 10 reward (all modules)        | Full reward ablation          |
| 3   | `grpo_qwen05`             | GRPO+SFT | Trie             | 4 reward                       | Config base                   |
| 4   | `sft`                     | SFT      | Trie (eval)      | —                               | Baseline supervisionata       |
| 5   | `zero_shot`               | ❌       | ❌               | translation (1.0)              | Lower bound                   |
| 6   | `zero_shot_grammar`       | ❌       | Trie             | translation (1.0)              | Grammar senza training        |
| 7   | `grpo_no_grammar`         | GRPO     | ❌               | 4 reward                       | GRPO senza constrained dec.   |
| 8   | `grpo_no_sft`             | GRPO     | Trie             | 6 reward                       | SFT ablation                  |
| 9   | `grpo_pda`                | GRPO     | PDA LL(1)        | 4 reward                       | GRPO + PDA baseline           |
| 10  | `grpo_pda_lookahead` ⭐   | GRPO+SFT | PDA + lookahead  | 7 reward + BLEU-4              | GRPO + native BPE emission    |
| 11  | `grpo_soft_viterbi`       | GRPO+SFT | Trie             | +soft_viterbi                   | Soft Viterbi ablation         |
| 12  | `grpo_verifier_scaled`    | GRPO+SFT | Trie             | +verifier_scaled               | Verifier ablation             |

### 7.2 Config Optimal v2.1 (`grpo_optimal.yaml`)

- LoRA r=32, alpha=64
- 7 reward simmetriche [-1, 1] + BLEU-4 (somma=1.0)
- beta=0.0 (DAPO-style), temperature=0.9
- Curriculum learning 3-stage enabled
- gradient_checkpointing=true (OOM mitigation)
- num_generations=8 (post-OOM-fix)

### 7.3 Config PDA + Lookahead (`grpo_pda_lookahead.yaml`) ⭐ NUOVO

- use_grammarllm_pda=true + token_lookahead=true
- Sfrutta grammarllm v0.5.0: StatelessLogitsProcessor + VocabTrie lookahead
- Il modello emette token BPE nativi invece di spelling
- Confrontare con grpo_optimal (Trie) per misurare il delta del lookahead

---

## 8. grammarllm v0.5.0 (Migrazione)

Il progetto usa la libreria `grammarllm` per il constrained decoding via PDA.
Versione precedente: snapshot pre-release vendored (v0.4.x). Versione attuale:
**v0.5.0** con:

- **StatelessLogitsProcessor** (784 righe): cache LRU + re-simulation, beam-search safe
- **Token-boundary lookahead** (`lookahead.py`): VocabTrie per native BPE token emission
- **Beam search support**: `num_beams > 1` con re-simulation
- **5 bug fix** (BUG-13 start_symbol configurable, BUG-17 cross-row conflict check,
  BUG-20 clone() invece di deepcopy, BUG-4/19 EOS validation, BUG-16 duplicate-key assert)
- **6 file di test** (16+ test di regressione)
- **Bound check** per token IDs fuori range (Qwen eos_token_id=151643 ≥ vocab_size=151643)

Documentazione completa in `docs/GRAMMARLLM_CONFRONTO.md` e
`docs/GRAMMARLLM_MIGRAZIONE.md`.

---

## 9. Cluster Pipeline

### 9.1 Architettura

```text
run_all.sh --ablation
    ↓
chain_next.sh (watcher, login node, setsid+disown)
    ↓ (sottomette un job alla volta)
train.sh / eval.sh (SLURM, Apptainer)
    ↓
chain_monitor (login node, monitor live)
    ↓
ablation_summary (post-pipeline, tabella + grafico cross-config)
```

### 9.2 Resilienza

- **Watcher**: `setsid nohup ... & disown` — sopravvive a SIGHUP (disconnect SSH)
  e SIGTERM (process reaper del login node)
- **Signal trap**: logga la causa della morte prima di uscire
- **Chain failure**: se un config fallisce, il suo eval viene saltato ma i
  restanti **continuano** (non stoppia la pipeline)
- **OOM/TIMEOUT/CUDA retry**: auto-resume con --resume (max 2 retry ciascuno)
- **Monitor auto-restart**: se il watcher muore, il monitor lo riavvia
- **Stale RUNNING check**: il monitor fa `sacct` per i job cached come RUNNING

### 9.3 Comandi

```bash
source cluster/aliases.sh
run-all --ablation          # 12 config train+eval (~24h)
monitor --all               # live dashboard
ablation-summary            # tabella + grafico cross-config
```

---

## 10. Test Suite

| Test File             | Tests | Status |
| --------------------- | ----- | ------ |
| `test_grammar.py`     | 7     | ✅     |
| `test_rewards.py`     | 9     | ✅     |
| `test_integration.py` | 5     | ✅     |
| **TOTAL**             | **21** | **100%** |

---

## 11. Innovazioni Chiave

1. **Neuro-Symbolic Architecture**: Constrained decoding + reward simboliche
   eliminano la necessità di un reward model neurale.

2. **8 Reward Modules [-1, 1]**: Range simmetrico per gradiente più forte e
   meno reward hacking. BLEU-4 aggiunto (T2G-Reasoner 2025) complementare a ROUGE-L.

3. **BLEU-4 con effective_order + smoothing**: Sequenze corte (1-3 token, comuni
   in ASL) non collassano a -1.0. Bug del caching silenzioso fixato con check eager.

4. **Curriculum Learning (G²RPO-A 2026)**: Difficoltà progressiva 3-stage calibrata
   sulla distribuzione reale di ASLG-PC12 (9.3% simple, 68.4% medium, 22.2% hard).

5. **Token-Boundary Lookahead (grammarllm v0.5.0)**: Il modello emette token BPE
   nativi invece di essere forzato a spelling — alignment alla tokenizzazione di
   pre-training.

6. **StatelessLogitsProcessor**: Cache LRU + re-simulation dallo stato PDA —
   beam-search safe, O(1) amortized.

7. **OOM Mitigation**: gradient_checkpointing + G=8 (dal OOM-causante G=16) su
   GPU 22GB, mantenendo tutte le migliorie v2 (curriculum, BLEU, beta=0).

8. **Ablation Summary**: Script che aggrega tutti i risultati eval in tabella
   CSV + Markdown + grafico a barre cross-config.

9. **Cluster Resilienza**: Watcher con setsid+disown sopravvive a disconnect SSH,
   chain continua su failure, monitor auto-restart, sacct stale check.

10. **Robust Gold Gloss Lookup**: SHA256 hashing delle user instructions per
    lookup format-agnostic, indipendente dal formato del prompt di TRL.

---

## 12. Stato del Progetto

| Componente           | Stato       | Note                                                         |
| -------------------- | ----------- | ------------------------------------------------------------ |
| Data pipeline        | ✅ Completo | ASLG-PC12 loader, bigram matrix, vocab extraction            |
| Constrained decoding | ✅ Completo | Trie dual-root + PDA v0.5.0 + lookahead                     |
| Reward system        | ✅ Completo | 8 moduli [-1, 1], BLEU-4 con effective_order                 |
| GRPO training        | ✅ Completo | G=8, beta=0, curriculum, gradient_checkpointing, OOM-safe    |
| SFT baseline         | ✅ Completo | Teacher forcing, 1 epoch                                     |
| Evaluation           | ✅ Completo | ROUGE-L, BLEU, Pass@K, best-of-N, --compare, ablation_summary |
| Test suite           | ✅ Completo | 21/21 pytest pass                                            |
| Config system        | ✅ Completo | 12 YAML, 12-config ablation matrix                           |
| Cluster pipeline     | ✅ Completo | SLURM, setsid+disown watcher, auto-restart, chain-failure-safe |
| Documentation        | ✅ Completo | README, REWARDS, METRICS, CONFIGS, GRAMMARLLM_CONFRONTO/MIGRAZIONE |

### 12.1 Pronto per Cluster

```bash
.\sync_cluster.ps1 -Action upload
ssh user@gcluster.dmi.unict.it
cd ~/neuro_symbolic_t2g && bash cluster/setup.sh
source cluster/aliases.sh
run-all --ablation          # 12 config
monitor --all               # live dashboard
ablation-summary            # tabella + grafico
```

---

## 13. Conclusioni

Il progetto **neuro_symbolic_t2g** implementa un'architettura neuro-symbolica
completa per la traduzione Text-to-Gloss ASL, combinando:

- **GRPO** con G=8, beta=0 (DAPO-style), temperature=0.9, curriculum 3-stage
- **Constrained decoding** con Trie dual-root (default) o PDA + lookahead (grammarllm v0.5.0)
- **8 reward simboliche** [-1, 1] incl. BLEU-4 con effective_order + smoothing
- **gradient_checkpointing** per OOM-safe training su GPU 22GB

Tutti i componenti sono implementati, testati (21/21 pytest pass), e pronti
per l'esecuzione su cluster con 12 configurazioni ablation. Il cluster pipeline
include watcher resiliente (setsid+disown), chain-failure-safe (continua su
failure), monitor con auto-restart, e ablation summary per analisi cross-config.

---

## Riferimenti

- **ASLG-PC12**: Othman & Jemni (2012), English-ASL Gloss Parallel Corpus
- **GRPO**: TRL GRPOTrainer 0.24.0 (HuggingFace)
- **Unsloth**: FastLanguageModel per QLoRA acceleration
- **T2G-Reasoner 2025**: BLEU-4 outperforms ROUGE-L as reward signal for T2G
- **G²RPO-A 2026**: Curriculum learning for GRPO on small models
- **RECIPE**: arXiv:2605.19976 — Verifier-scaled reward
- **ViterbiPlanNet DVL**: arXiv:2603.04265 — Differentiable Viterbi
- **grammarllm v0.5.0**: PDA-based constrained decoding with token-boundary lookahead
