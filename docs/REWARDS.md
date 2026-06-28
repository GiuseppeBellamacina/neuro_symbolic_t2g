# T2G GRPO Reward Functions — Documentazione Dettagliata

Questo documento descrive tutte le funzioni di reward utilizzate nel training GRPO per Text-to-Gloss (T2G). Ogni reward è spiegata con la sua formulazione matematica, lo scopo, e le considerazioni pratiche.

**File di riferimento**: `src/rewards/t2g_rewards.py`

---

## Indice

1. [Architettura Generale](#architettura-generale)
2. [Reward 1: Translation Quality (ROUGE-L)](#reward-1-translation-quality-rouge-l)
3. [Reward 2: Structural Dense (Bigram Assoluto)](#reward-2-structural-dense-bigram-assoluto)
4. [Reward 3: Gold-Structure (Gold Baseline)](#reward-3-gold-structure-gold-baseline) ⭐ **Raccomandata**
5. [Reward 4: Viterbi Distance (Upper Bound Teorico)](#reward-4-viterbi-distance-upper-bound-teorico) 🧪 **Sperimentale**
6. [Reward 5: Format](#reward-5-format)
7. [Reward 6: Repetition](#reward-6-repetition)
8. [Confronto tra le Reward Strutturali](#confronto-tra-le-reward-strutturali)
9. [Configurazione](#configurazione)
10. [Implementazione Algoritmo di Viterbi](#implementazione-algoritmo-di-viterbi)

---

## Architettura Generale

Le reward sono combinate come somma pesata:

```
R_total = w_translation × R_translation
        + w_structure  × R_structure         (oppure w_gold_structure, w_viterbi)
        + w_format     × R_format
        + w_repetition × R_repetition
```

Ogni componente restituisce un punteggio che viene poi pesato e sommato. I pesi sono configurabili via YAML (vedi [Configurazione](#configurazione)).

**Stato globale condiviso**:
- `_bigram_matrix`: matrice di transizione bigram `(V × V)`, precomputata sul training set
- `_gloss_vocab`: vocabolario dei gloss ordinato (include `<BOS>`, `<EOS>`, `<UNK>`)
- `_rouge_scorer`: scorer ROUGE-L inizializzato lazy
- `_gold_gloss_registry`: mappa `sample_id → gold_gloss` per lookup veloce durante GRPO

---

## Reward 1: Translation Quality (ROUGE-L)

**Funzione**: `translation_quality_reward(completion, gold_gloss) → float`

**Peso default**: `weight_translation = 0.40`

### Scopo
Misurare la similarità lessicale tra la sequenza di gloss generata dal modello e la sequenza gold di riferimento. Questa è il segnale semantico primario per GRPO.

### Formulazione

```
R_translation = ROUGE-L_F1(generated_gloss, gold_gloss)
```

dove **ROUGE-L** calcola l'F1-score basato sulla Longest Common Subsequence (LCS) tra la sequenza generata e quella gold.

- **Recall**: `LCS_length / |gold_tokens|`
- **Precision**: `LCS_length / |generated_tokens|`
- **F1**: `2 × Recall × Precision / (Recall + Precision)`

### Range
`[0, 1]` — 1.0 indica match perfetto, 0.0 indica nessuna sovrapposizione.

### Implementazione
- Usa `rouge_score.RougeScorer` con stemmer disabilitato (i gloss ASL non vanno stemmati)
- Prima di calcolare ROUGE, estrae il testo pulito dei gloss rimuovendo tag `<think>`, blocchi di codice, e whitespace extra
- Il gold gloss viene recuperato dal `_gold_gloss_registry` usando un sample ID stabile (SHA256 del prompt), garantendo lookup deterministico anche dopo riformattazione del prompt da parte di TRL

### Esempio
```
Gold:     "IX MAN WALK HOUSE"
Generated: "IX MAN GO HOUSE"
→ ROUGE-L F1 ≈ 0.67 (3 token su 4 matchano in LCS: IX, MAN, HOUSE / GO è diverso)
```

---

## Reward 2: Structural Dense (Bigram Assoluto)

**Funzione**: `structural_dense_reward(completion, normalize=True) → float`

**Peso config**: `weight_structure` (default: 0.0 — **deprecato in favore di `weight_gold_structure`**)

### Scopo
Misurare la plausibilità strutturale **assoluta** della sequenza generata sotto il modello bigram. Sequenze con transizioni ad alta probabilità (osservate frequentemente nei dati reali ASL) ricevono punteggi più alti.

### Formulazione

```
R_structure = exp( (1/(L-1)) × Σ_{i=1}^{L-1} log P(gloss_i | gloss_{i-1}) )
```

dove `L` è la lunghezza della sequenza con marcatori BOS/EOS, e `P` proviene dalla matrice bigram con Laplace smoothing (α=1).

- **Con `normalize=True`**: esponenzia la media dei log-prob → `[0, 1]`
- **Con `normalize=False`**: restituisce la media dei log-prob grezzi (tipicamente negativa)

### Range
`[0, 1]` con normalizzazione. 0.0 = sequenza strutturalmente implausibile o troppo corta.

### Limitazioni
- **Nessuna baseline**: misura solo la qualità assoluta, non relativa. Una sequenza con `R=0.3` potrebbe essere eccellente o mediocre a seconda della difficoltà del prompt.
- **Non confronta con il gold**: non sa se la sequenza generata è strutturalmente migliore o peggiore del riferimento umano.
- **Favorisce sequenze più corte**: sequenze lunghe tendono ad accumulare più transizioni a bassa probabilità.

### Esempio
```
"IX MAN WALK HOUSE"    → R ≈ 0.85 (transizioni plausibili)
"DOG CAT BIRD FISH"   → R ≈ 0.12 (transizioni improbabili nei dati ASL)
"IX"                   → R = 0.0  (< 2 token, nessun bigram da valutare)
```

> ⚠️ **Deprecato in produzione**. Usare invece `gold_structure_reward` (Reward 3).

---

## Reward 3: Gold-Structure (Gold Baseline)

**Funzione**: `gold_structure_reward(completion, gold_gloss, normalize=True) → float`

**Peso default**: `weight_gold_structure = 0.40` ⭐

### Scopo
Misurare quanto la plausibilità strutturale della sequenza generata si avvicina a quella della sequenza **gold di riferimento** (scritta da un umano). Questo risolve il problema della Reward 2 (nessuna baseline) confrontando l'LLM contro un ground truth semanticamente significativo.

### Formulazione

```
llm_avg  = (1/L_llm)  × Σ log P(token_i | token_{i-1})   [path LLM]
gold_avg = (1/L_gold) × Σ log P(token_i | token_{i-1})   [path gold]

R_gold_structure = min( exp(llm_avg - gold_avg), 1.0 )
```

### Interpretazione
- **`≈ 1.0`**: la sequenza LLM è strutturalmente all'altezza (o superiore) del gold umano
- **`≪ 1.0`**: la sequenza LLM ha transizioni bigram molto peggiori del gold
- **Cap a 1.0**: se l'LLM supera il gold in termini strutturali, non viene premiato oltre (evita overfitting su pattern statistici)

### Vantaggi rispetto a `structural_dense_reward`
1. **Baseline semanticamente significativa**: confronta contro gloss reali, non contro un optimum degenere
2. **Normalizzato per difficoltà**: prompt difficili hanno gold con transizioni più rare → baseline adeguata
3. **Robusto a lunghezze diverse**: normalizza per numero di transizioni (`L_llm` e `L_gold` possono differire)

### Esempio
```
Gold:          "IX MAN WALK HOUSE"    (gold_avg = -0.5)
LLM generated: "IX MAN GO HOUSE"      (llm_avg  = -1.2)
→ R = min(exp(-1.2 - (-0.5)), 1.0) = min(exp(-0.7), 1.0) ≈ 0.50
```

### Nota implementativa
- Richiede `needs_gold_gloss=True` nel wrapper GRPO
- Il gold gloss viene recuperato dal `_gold_gloss_registry` usando lo stesso sample ID stabile della Reward 1
- Entrambe le sequenze (LLM e gold) sono wrappate con BOS/EOS prima del calcolo

---

## Reward 4: Viterbi Distance (Upper Bound Teorico)

**Funzione**: `viterbi_distance_reward(completion, normalize=True) → float`

**Peso config**: `weight_viterbi` (default: 0.0) 🧪

### Scopo
Misurare quanto il path dell'LLM si avvicina al **cammino ottimo globale** calcolato offline dall'algoritmo di Viterbi sulla matrice di transizione bigram. L'upper bound di Viterbi rappresenta la sequenza di gloss di pari lunghezza che **massimizza** il prodotto delle probabilità di transizione sotto il modello di Markov.

### Formulazione

```
Sia L = lunghezza della sequenza LLM (inclusi BOS e EOS)

viterbi_opt = max_{path di lunghezza L} Σ log P(token_i | token_{i-1})
              soggetto a: path[0] = BOS, path[L-1] = EOS

R_viterbi = exp( (llm_log_prob - viterbi_opt) / (L-1) )
```

### Algoritmo di Viterbi Diversificato

L'algoritmo (vedi [§Implementazione](#implementazione-algoritmo-di-viterbi)) usa **due meccanismi anti-degenerazione**:

1. **Self-loop penalty** (`λ = 0.5`): penalità sottratta al log-prob quando il path resta nello stesso stato:

```
dp[t][s] = max_k ( dp[t-1][k] + log P(s|k) - λ × I[s == k] )
```

2. **Ban iterativo dei token over-rappresentati**: dopo il primo path candidato, i token che compaiono più di `max_occurrences=2` volte vedono la loro self-transition ridotta del 70%. L'algoritmo viene rieseguito fino a 3 volte finché la diversità (unique ratio) ≥ 0.3.

Complessità: `O(V² × L × max_iters)` con max_iters ≤ 3.

### Range
`(0, 1]` con normalizzazione.
- `1.0` = path LLM coincide con l'ottimo Viterbi diversificato
- `≈ 0.0` = path LLM molto distante dall'ottimo

### ⚠️ Caveat (risolto con diversity constraint)

~~Senza probabilità di emissione, l'ottimo Viterbi degenera in loop (es. `BOS → IX → IX → IX → EOS`).~~

**Risolto**: self-loop penalty + ban iterativo producono un path diversificato e plausibile. Il `viterbi_distance_reward` è ora utilizzabile in produzione (anche se `gold_structure_reward` rimane la scelta raccomandata).

### Quando usarla
- **Training GRPO** con `weight_viterbi > 0` per un segnale strutturale aggiuntivo
- **Ablation study** per confrontare il gap tra path LLM e ottimo teorico diversificato

### Raccomandazione
Per training GRPO produttivo, usare **`gold_structure_reward`** (Reward 3). `viterbi_distance_reward` è ora un'alternativa valida.

---

## Reward 5: Format

**Funzione**: `gloss_format_reward(completion) → float`

**Peso default**: `weight_format = 0.10`

### Scopo
Penalizzare output che contengono testo libero, codice, JSON, o punteggiatura invece di puri token di gloss ASL.

### Range
- `1.0` = output pulito, solo gloss token (es. `"IX MAN WALK HOUSE"`)
- `0.5` = contenuto misto (gloss + free text)
- `0.0` = chiaramente non-gloss (es. `"The man walks to the house."`, `{"gloss": "..."}`, vuoto)

### Pattern rilevati
- Articoli e preposizioni inglesi: `the, a, an, is, are, in, on, at, by, for, with...`
- Punteggiatura: `.,!?;:`
- Blocchi di codice: ` ``` `
- JSON: `{`, `}`

---

## Reward 6: Repetition

**Funzione**: `gloss_repetition_reward(completion) → float`

**Peso default**: `weight_repetition = 0.10`

### Scopo
Penalizzare sequenze degenerate con token ripetuti (loop). Durante GRPO, il modello può collassare su strategie che massimizzano altre reward producendo lo stesso token molte volte.

### Range
- `1.0` = output normale, buona diversità lessicale
- `0.0` = ripetizione moderata (uniqueness ratio tra 0.3 e 0.5)
- `-1.0` = loop severo (uniqueness ratio ≤ 0.3)
- `1.0` = sequenze corte (< 4 token, non valutabili)

### Metriche
- **Token uniqueness ratio**: `|unique_tokens| / |total_tokens|`
- **Trigram uniqueness ratio**: `|unique_trigrams| / |total_trigrams|`
- Usa il minimo delle due per penalizzare sia ripetizioni di token che di pattern

### Esempio
```
"IX MAN WALK HOUSE BOOK CAN NOT WANT GO COME"  → R = 1.0 (tutti unici)
"IX IX MAN WALK IX IX MAN WALK"                → R = 0.0 (ripetizione moderata)
"IX IX IX IX IX IX IX IX IX IX"                → R = -1.0 (loop severo)
"IX MAN"                                        → R = 1.0 (< 4 token)
```

---

## Confronto tra le Reward Strutturali

| Reward | Baseline | Range | Significato | Raccomandazione |
|--------|----------|-------|-------------|-----------------|
| **structural_dense** | Nessuna | [0, 1] | Qualità bigram assoluta | ⚠️ Deprecata |
| **gold_structure** | Gold gloss umano | [0, 1] | Quanto l'LLM è strutturalmente vicino al gold | ⭐ **Produzione** |
| **viterbi_distance** | Ottimo Viterbi | (0, 1] | Distanza dal cammino ottimo teorico | 🧪 **Sperimentale** |

### Perché `gold_structure` è superiore

1. **Baseline reale**: il gold gloss è scritto da un umano → confronto semanticamente significativo
2. **Nessuna degenerazione**: non soffre del problema dei path Viterbi degeneri
3. **Adattivo alla difficoltà**: prompt difficili hanno gold con transizioni più rare → la baseline si adatta naturalmente
4. **Interpretabile**: `R ≈ 1.0` significa "strutturalmente all'altezza dell'umano"

### Perché `viterbi_distance` è sperimentale

1. **Upper bound troppo alto**: senza emission probabilities, l'ottimo Viterbi è irraggiungibile per sequenze sensate
2. **Segnale schiacciato**: `exp(negativo_grande) ≈ 0` per quasi tutte le sequenze non degeneri
3. **Potenziale cleanup**: si potrebbe aggiungere un penalty di diversità sul path Viterbi stesso, ma questo richiederebbe un iperparametro aggiuntivo

---

## Configurazione

### YAML Config (`experiments/configs/t2g/grpo_qwen05.yaml`)

```yaml
reward:
  # Translation quality (ROUGE-L) — sempre attiva
  weight_translation: 0.40

  # Gold-structure: bigram vs gold baseline ⭐ RACCOMANDATA
  weight_gold_structure: 0.40

  # Oppure, per la vecchia structural_dense (senza baseline):
  # weight_structure: 0.40

  # Oppure, per Viterbi distance (sperimentale):
  # weight_viterbi: 0.40

  # Format: solo gloss, no free text
  weight_format: 0.10

  # Repetition: penalizza loop
  weight_repetition: 0.10
```

### Combinazioni possibili

```yaml
# Config 1: Produzione (default)
reward:
  weight_translation: 0.40
  weight_gold_structure: 0.40
  weight_format: 0.10
  weight_repetition: 0.10

# Config 2: Solo semantica (baseline semplice)
reward:
  weight_translation: 0.65
  weight_format: 0.20
  weight_repetition: 0.15

# Config 3: Esperimento Viterbi
reward:
  weight_translation: 0.30
  weight_gold_structure: 0.30
  weight_viterbi: 0.20
  weight_format: 0.10
  weight_repetition: 0.10
```

### Pesare a zero
Qualsiasi peso impostato a `0.0` (o assente) disabilita la reward corrispondente. Esempio:

```yaml
reward:
  weight_translation: 0.7
  weight_gold_structure: 0.3
  # format e repetition disabilitate (peso 0 implicito)
```

---

## Implementazione Algoritmo di Viterbi

L'algoritmo è implementato in `src/datasets/transition_matrix.py`:

### `compute_viterbi_path(transition_matrix, start_idx, end_idx, length)`

Calcola il path ottimo **puro** (senza diversity constraint) tramite DP con backtracking.

### `compute_diverse_viterbi_path(...)`

Versione **diversificata** con self-loop penalty e ban iterativo. Accetta:
- `self_loop_penalty=0.5` — penalità log-prob per self-transition
- `max_occurrences=2` — max ripetizioni consentite per token prima del ban
- `diversity_threshold=0.3` — unique-token ratio minimo accettabile
- `max_iters=3` — iterazioni massime di re-ottimizzazione

### `viterbi_optimal_score_diverse(...)`

Versione ottimizzata del Viterbi diversificato che restituisce **solo il punteggio** (usa internamente `compute_diverse_viterbi_path`).

### Complessità
- **Tempo**: `O(V² × L × max_iters)` — con V ≈ 2000, L ≈ 12, max_iters=3 → ~144M operazioni, eseguibile in < 1s su CPU
- **Spazio**: `O(V × L)` per il DP con backtracking

---

## Riferimenti

- **Codice**: `src/rewards/t2g_rewards.py` — tutte le funzioni di reward
- **Matrice di transizione**: `src/datasets/transition_matrix.py` — bigram matrix + Viterbi
- **Test**: `tests/test_rewards.py` — test per tutte le reward
- **Config**: `experiments/configs/t2g/grpo_qwen05.yaml` — pesi configurabili
- **Dataset**: `src/datasets/aslg_dataset.py` — ASLG-PC12, vocabolario gloss
