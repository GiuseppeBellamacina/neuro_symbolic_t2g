# Grammar Diagnostics — Metriche W&B

Questo documento descrive le metriche W&B tracciate dal `CompletionSampleCallback`
per monitorare come il modello internalizza i vincoli grammaticali ASL durante
il training GRPO.

**File di riferimento**: `src/training/callbacks.py`, `src/grammar/grammar_logits_processor.py`

---

## Indice

1. [Metriche Disponibili](#metriche-disponibili)
2. [Interpretazione Congiunta](#interpretazione-congiunta)
3. [Diagnostica della Convergenza](#diagnostica-della-convergenza)
4. [Pattern Tipici di Training](#pattern-tipici-di-training)
5. [Formulazione Matematica](#formulazione-matematica)
6. [FAQ](#faq)

---

## Metriche Disponibili

Tutte le metriche sono loggate a ogni `logging_steps` sotto il prefisso `grammar/` su W&B.
Il `CompletionSampleCallback` chiama `get_masked_mass_stats(reset_after=True)` per ottenere
medie per-intervallo (non cumulative).

### `grammar/masked_mass_avg`

**Range**: `[0, 1]`

**Definizione**: Frazione della distribuzione di probabilità softmax che viene
assegnata a token **non consentiti** dalla grammatica ASL, **prima** che la
maschera venga applicata.

```
masked_mass = Σ_{i ∉ Allowed} p_i
```

**Interpretazione**:
- **Alto** (`> 0.5`): il modello "vuole" generare token non-gloss (testo libero,
  punteggiatura, parole inglesi). Sta combattendo contro i vincoli grammaticali.
- **Basso** (`< 0.2`): il modello ha internalizzato il vocabolario ASL e alloca
  la quasi totalità della massa sui token consentiti.
- **In calo durante il training**: convergenza positiva — il constrained
  decoding sta funzionando e il modello sta imparando.

---

### `grammar/masked_entropy_avg`

**Range**: `[0, log(V)]` dove `V` è la dimensione del vocabolario
(tipicamente `log(32000) ≈ 10.4` per Qwen, `log(50000) ≈ 10.8` per altri modelli).

**Definizione**: Entropia di Shannon della distribuzione softmax **completa**
(prima del mascheramento).

```
H_full = -Σ_{i=0}^{V-1} p_i · log(p_i + ε)
```

**Interpretazione**:
- **Alta** (`> 8`): distribuzione piatta — il modello è **incerto**, assegna
  probabilità simile a molti token diversi.
- **Bassa** (`< 4`): distribuzione concentrata — il modello è **confidente**
  su pochi token.
- **In calo durante il training**: il modello sta diventando più deciso nelle
  sue predizioni.

---

### `grammar/masked_entropy_allowed_avg`

**Range**: `[0, log(|Allowed|)]` dove `|Allowed|` è il numero di token
consentiti (tipicamente poche centinaia, `log(500) ≈ 6.2`).

**Definizione**: Entropia della distribuzione **ri-normalizzata** sui soli
token consentiti dalla grammatica.

```
p̃_i = p_i / Σ_{j ∈ Allowed} p_j    per i ∈ Allowed
H_allowed = -Σ_{i ∈ Allowed} p̃_i · log(p̃_i + ε)
```

**Interpretazione**:
- **Alta** (vicina a `log(|Allowed|)`): il modello è incerto **anche tra i token
  validi** — sta esplorando lo spazio delle glosse ma non sa quale scegliere.
- **Bassa** (`< 2`): il modello è confidente sia nel vocabolario che nella
  scelta specifica — sa esattamente quale glossa generare.
- **In calo** (insieme a `masked_mass_avg` in calo): convergenza ideale — il
  modello smette di combattere i vincoli E diventa preciso sulle scelte.

---

### `grammar/masked_mass_steps`

**Range**: intero positivo

**Definizione**: Numero di step di generazione (token) registrati nell'intervallo
corrente. Usato per verificare che le metriche abbiano campioni sufficienti.

---

## Interpretazione Congiunta

La combinazione delle tre metriche di entropia/massa fornisce un quadro
diagnostico completo dello stato del modello:

| masked_mass | full_entropy | allowed_entropy | Diagnosi |
|-------------|-------------|-----------------|----------|
| Alto (>0.5) | Alta (>8) | Qualsiasi | Il modello non ha ancora internalizzato il vocabolario ASL. Esplora tutto lo spazio, incluso testo libero. **Fase iniziale attesa.** |
| Alto (>0.5) | Bassa (<4) | N/A | Il modello è **confidentemente sbagliato**: assegna alta probabilità a token non-gloss specifici (es. parole inglesi comuni). **Segnale preoccupante** — il modello potrebbe aver memorizzato pattern errati. |
| Basso (<0.2) | Alta (>7) | Alta (>4) | Il modello rispetta il vocabolario ASL ma è **incerto su quale glossa scegliere**. Sta esplorando lo spazio delle glosse valide. **Fase intermedia sana.** |
| Basso (<0.2) | Bassa (<4) | Bassa (<2) | **Convergenza ideale**: il modello è confidente, rispetta la grammatica, e sa quale token generare. |
| Basso (<0.2) | Bassa (<4) | Alta (>4) | Il modello è confidente globalmente (picca su pochi token) ma quei token potrebbero essere sia gloss che non-gloss. Se `masked_mass` è basso, le picche sono su token gloss. **Segnale misto** — controllare la qualità delle generazioni. |

---

## Diagnostica della Convergenza

### Cosa aspettarsi durante il training

1. **Inizio training** (step 0-200):
   - `masked_mass_avg` alto (0.6-0.9) — il modello genera testo libero
   - `masked_entropy_avg` alto (8-10) — distribuzione piatta
   - `masked_entropy_allowed_avg` variabile (dipende da quanti token allowed ci sono)

2. **Metà training** (step 200-800):
   - `masked_mass_avg` in calo costante — il constrained decoding sta funzionando
   - `masked_entropy_avg` in calo — il modello diventa più deciso
   - `masked_entropy_allowed_avg` potrebbe salire temporaneamente (esplorazione dello spazio gloss) prima di scendere

3. **Fine training** (step 800+):
   - `masked_mass_avg` stabilizzato sotto 0.15
   - `masked_entropy_avg` stabilizzato sotto 5
   - `masked_entropy_allowed_avg` stabilizzato sotto 3

### Red flags

- **`masked_mass_avg` non scende mai sotto 0.4**: il modello non sta imparando
  il vocabolario ASL. Possibili cause: learning rate troppo basso, reward mal
  calibrata, dataset troppo piccolo.
- **`masked_entropy_avg` crolla a 0 ma `masked_mass_avg` resta alto**: il modello
  collassa su un singolo token (spesso non-gloss). **Mode collapse** — aumentare
  `weight_repetition` o ridurre `beta`.
- **`masked_entropy_allowed_avg` sale mentre le altre scendono**: il modello
  impara a rispettare la grammatica ma diventa **più incerto** su quale glossa
  scegliere. Possibile che il segnale di reward strutturale (`weight_gold_structure`)
  sia troppo forte rispetto a `weight_translation`.

---

## Pattern Tipici di Training

### Pattern 1: Convergenza Ideale
```
Step    mass    H_full  H_allowed
  0     0.72    9.8     4.1
 50     0.65    9.2     4.8
100     0.48    8.1     5.2    ← inizia a rispettare la grammatica
200     0.28    6.5     4.1    ← esplora meno, più confidente
400     0.15    5.1     3.0
800     0.10    4.2     2.1    ← convergenza
```

### Pattern 2: Confidentemente Sbagliato (problematico)
```
Step    mass    H_full  H_allowed
  0     0.68    9.5     5.2
 50     0.62    6.1     3.8    ← entropia scende ma massa resta alta
100     0.58    4.2     2.9
200     0.55    3.1     2.5    ← stallo: il modello è confidente sui token sbagliati
400     0.52    2.8     2.3
```

### Pattern 3: Mode Collapse
```
Step    mass    H_full  H_allowed
  0     0.71    9.7     5.0
 50     0.35    5.2     3.1
100     0.12    1.5     0.8    ← entropia crolla troppo velocemente
150     0.08    0.3     0.1    ← collapse su un solo token
200     0.05    0.1     0.0
```

---

## Formulazione Matematica

### Softmax e Mascheramento

Dati i logit `z ∈ R^V` dal modello, la distribuzione softmax è:

```
p_i = exp(z_i) / Σ_j exp(z_j)
```

L'insieme `A` (Allowed) contiene gli indici dei token consentiti dalla grammatica.
La masked mass è:

```
M = Σ_{i ∉ A} p_i
```

### Entropia Full

```
H_full = -Σ_{i=0}^{V-1} p_i · log(p_i + ε)
```

dove `ε = 1e-12` evita `log(0)`. Poiché `lim_{x→0} x·log(x) = 0`, il contributo
dei token a probabilità zero è correttamente nullo.

### Entropia Allowed (Ri-normalizzata)

```
Z_allowed = Σ_{i ∈ A} p_i = 1 - M

p̃_i = p_i / Z_allowed    per i ∈ A
H_allowed = -Σ_{i ∈ A} p̃_i · log(p̃_i + ε)
```

Se `M = 1` (nessun token consentito), `Z_allowed = 0` e `H_allowed = 0`
per convenzione (non ci sono token validi su cui essere incerti).

---

## FAQ

### Perché tracciare l'entropia sui soli token permessi?

L'entropia globale (`H_full`) è dominata dalla massa sui token non consentiti
quando `masked_mass` è alta. L'entropia ri-normalizzata (`H_allowed`) isola
il segnale di incertezza **tra i token validi**, che è ciò che interessa per
capire se il modello sta imparando a scegliere le glosse giuste.

### Quale processore genera queste metriche?

Entrambi i processor (`GlossVocabularyLogitsProcessor` e `GrammarPDALogitsProcessor`)
implementano la stessa interfaccia `get_masked_mass_stats()`. Il callback W&B
funziona indipendentemente da quale processor è attivo.

### Con che frequenza vengono loggate?

A ogni `logging_steps` (default: ogni 5 step), il callback chiama
`get_masked_mass_stats(reset_after=True)` e logga le medie dell'intervallo.
Le metriche sono quindi **medie mobili per-intervallo**, non cumulative.

### Posso monitorare queste metriche in tempo reale?

Sì, appaiono su W&B sotto il gruppo `grammar/`. Puoi anche monitorarle
localmente con `tail -f logs/chain_watcher.log` durante il training su cluster.

---

## Riferimenti

- **Codice processor**: `src/grammar/grammar_logits_processor.py`
- **Callback**: `src/training/callbacks.py`
- **Documentazione reward**: `docs/REWARDS.md`
- **Test**: `tests/test_grammar.py`
