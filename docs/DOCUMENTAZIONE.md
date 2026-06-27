# 📘 Documentazione Completa di GrammarLLM

> **Autore:** Analisi automatica della codebase  
> **Versione progetto:** 0.3.3  
> **Data analisi:** 25 Giugno 2026  

---

## Indice

1. [Panoramica Generale](#1-panoramica-generale)
2. [Architettura del Progetto](#2-architettura-del-progetto)
3. [Pipeline di Generazione Vincolata](#3-pipeline-di-generazione-vincolata)
4. [Descrizione Dettagliata dei File](#4-descrizione-dettagliata-dei-file)
   - [4.1 `pyproject.toml` e `requirements.txt`](#41-pyprojecttoml-e-requirementstxt)
   - [4.2 `grammarllm/__init__.py`](#42-grammarllm__init__py)
   - [4.3 `grammarllm/generate_with_constraints.py`](#43-grammarllmgenerate_with_constraintspy)
   - [4.4 `grammarllm/utils/toolbox.py`](#44-grammarllmutilstoolboxpy)
   - [4.5 `grammarllm/utils/common_regex.py`](#45-grammarllmutilscommon_regexpy)
   - [4.6 `grammarllm/scripts/grammar_generation.py`](#46-grammarllmscriptsgrammar_generationpy)
   - [4.7 `grammarllm/scripts/generate_LL1_parsing_table.py`](#47-grammarllmscriptsgenerate_ll1_parsing_tablepy)
   - [4.8 `grammarllm/scripts/map_terminal_tokens.py`](#48-grammarllmscriptsmap_terminal_tokenspy)
   - [4.9 `grammarllm/modules/PushdownAutomaton.py`](#49-grammarllmmodulespushdownautomatonpy)
   - [4.10 `grammarllm/modules/BaseStreamer.py`](#410-grammarllmmodulesbasestreamerpy)
   - [4.11 `grammarllm/modules/SimpleLogitProcessor.py`](#411-grammarllmmodulessimplelogitprocessorpy)
   - [4.12 `grammarllm/modules/SimpleLogitProcessor_.py`](#412-grammarllmmodulessimplelogitprocessor_py)
   - [4.13 `main.py`](#413-mainpy)
   - [4.14 `QuickStart.ipynb`](#414-quickstartipynb)
5. [La Grammatica LL(prefix)](#5-la-grammatica-llprefix)
6. [Esempi di Utilizzo](#6-esempi-di-utilizzo)
   - [6.1 Classificazione Gerarchica](#61-classificazione-gerarchica)
   - [6.2 Restrizione del Vocabolario](#62-restrizione-del-vocabolario)
   - [6.3 Generazione Strutturata (RDF)](#63-generazione-strutturata-rdf)
7. [Flusso dei Dati](#7-flusso-dei-dati)

---

## 1. Panoramica Generale

**GrammarLLM** è una libreria Python open-source (licenza MIT) per la **generazione di testo vincolata da grammatica** basata su modelli Transformer pre-addestrati (Hugging Face 🤗).

In pratica, permette di **forzare un LLM** (come LLaMA, GPT-2, ecc.) a generare testo che rispetti una **grammatica formale** definita dall'utente, garantendo che ogni token prodotto sia sintatticamente corretto secondo le regole specificate.

Il progetto è stato pubblicato come paper accademico a **ACL 2025** (*Findings of ACL 2025*) con il titolo *"GRAMMAR-LLM: Grammar-Constrained Natural Language Generation"*.

### Casi d'uso principali:
- **Classificazione gerarchica** – l'LLM classifica un input scegliendo solo tra categorie predefinite
- **Restrizione del vocabolario** – l'LLM può usare solo parole autorizzate
- **Generazione strutturata** – l'LLM produce output formattati (es. triple RDF, JSON, codice)

### Caratteristiche tecniche:
- ✅ Decodifica in tempo lineare tramite **automa a pila deterministico (PDA)**
- ✅ Compatibile con tutti i modelli Hugging Face (causali)
- ✅ Supporta **LL(prefix)** — generalizzazione di LL(1) per la tokenizzazione subword
- ❌ **NON** supporta beam search (solo greedy decoding o sampling)
- ❌ Non si possono definire più `<<exact_string>>` nella stessa regola

---

## 2. Architettura del Progetto

```
grammarllm/
├── __init__.py                          # API pubblica del package
├── generate_with_constraints.py         # Pipeline principale
├── modules/
│   ├── __init__.py
│   ├── BaseStreamer.py                  # Streamer per aggiornare PDA durante la generazione
│   ├── PushdownAutomaton.py             # Automa a pila deterministico (PDA)
│   ├── SimpleLogitProcessor.py          # Processore logits (versione base)
│   └── SimpleLogitProcessor_.py         # Processore logits (versione avanzata con metriche)
├── scripts/
│   ├── __init__.py
│   ├── grammar_generation.py            # Conversione grammatica utente → LL(1)
│   ├── generate_LL1_parsing_table.py    # Generazione tabella di parsing LL(1)
│   └── map_terminal_tokens.py           # Mappatura terminali → token ID
├── utils/
│   ├── __init__.py
│   ├── toolbox.py                       # Utility: create_prompt, chat_template
│   └── common_regex.py                  # Regex predefinite per terminali
└── README_PACKAGE.md                    # Documentazione per PyPI
```

Con file esterni:
```
main.py              # Script demo completo
QuickStart.ipynb     # Notebook Jupyter con esempi
pyproject.toml       # Configurazione del package Python
requirements.txt     # Dipendenze
README.md            # Documentazione principale
LICENSE              # Licenza MIT
```

---

## 3. Pipeline di Generazione Vincolata

Il flusso completo di GrammarLLM si articola in **5 fasi**:

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                         PIPELINE DI GRAMMARLLM                               │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  FASE 1: PREPROCESSING DELLA GRAMMATICA                                      │
│  ┌─────────────────────────────────────────┐                                │
│  │ ProductionRuleProcessor                 │                                │
│  │                                         │                                │
│  │  Input:  { 'S*': ["<<positive>> A"],    │                                │
│  │            'A':  ["<<happy>>"] }         │                                │
│  │                                         │                                │
│  │  Output: final_grammar (LL(1))          │                                │
│  │          tag_mapping                    │                                │
│  └──────────────┬──────────────────────────┘                                │
│                 │                                                            │
│                 ▼                                                            │
│  FASE 2: TABELLA DI PARSING LL(1)                                           │
│  ┌─────────────────────────────────────────┐                                │
│  │ parsing_table()                         │                                │
│  │                                         │                                │
│  │  - Calcola FIRST e FOLLOW               │                                │
│  │  - Costruisce tabella di parsing        │                                │
│  │  - Rileva conflitti LL(1)               │                                │
│  │                                         │                                │
│  │  Output: pars_tab                       │                                │
│  └──────────────┬──────────────────────────┘                                │
│                 │                                                            │
│                 ▼                                                            │
│  FASE 3: MAPPATURA TERMINALI → TOKEN                                        │
│  ┌─────────────────────────────────────────┐                                │
│  │ generate_token_maps()                   │                                │
│  │                                         │                                │
│  │  - Usa regex per associare terminali    │                                │
│  │    a token ID                            │                                │
│  │  - Verifica assenza di conflitti        │                                │
│  │                                         │                                │
│  │  Output: map_terminal_tokens            │                                │
│  └──────────────┬──────────────────────────┘                                │
│                 │                                                            │
│                 ▼                                                            │
│  FASE 4: CREAZIONE PDA E PROCESSORI                                          │
│  ┌─────────────────────────────────────────┐                                │
│  │ PushdownAutomaton + MaskLogitsProcessor │                                │
│  │ + BaseStreamer                          │                                │
│  │                                         │                                │
│  │  - PDA inizializzato con tabella        │                                │
│  │  - LogitsProcessor maschera token       │                                │
│  │    non validi                           │                                │
│  │  - Streamer aggiorna PDA a ogni token   │                                │
│  └──────────────┬──────────────────────────┘                                │
│                 │                                                            │
│                 ▼                                                            │
│  FASE 5: GENERAZIONE VINCOLATA                                               │
│  ┌─────────────────────────────────────────┐                                │
│  │ model.generate() con vincoli            │                                │
│  │                                         │                                │
│  │  - A ogni passo, LogitsProcessor        │                                │
│  │    azzera i logit dei token non validi  │                                │
│  │  - Streamer aggiorna lo stack del PDA   │                                │
│  │  - Output: testo vincolato              │                                │
│  └─────────────────────────────────────────┘                                │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## 4. Descrizione Dettagliata dei File

### 4.1 `pyproject.toml` e `requirements.txt`

**Configurazione del package Python.** Definisce:

- **Nome:** `grammarllm`
- **Versione:** `0.3.3`
- **Autori:** Gabriele Tuccio, Luana Bulla, Maria Madonia, Aldo Gangemi, Misael Mongiovì
- **Dipendenze:** `torch`, `transformers>=4.30.0`, `accelerate>=0.26.0`, `tqdm`, `regex`, `setuptools`
- **Python minimo:** 3.10
- **Licenza:** MIT
- **Build system:** setuptools

Il package è pubblicato su **Test PyPI** e installabile con:
```bash
pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple grammarllm
```

---

### 4.2 `grammarllm/__init__.py`

**API pubblica del package.** Espone le funzioni principali per l'utente:

| Funzione | Descrizione |
|----------|-------------|
| `get_parsing_table_and_map_tt()` | Genera tabella di parsing e mappa terminali-token |
| `generate_grammar_parameters()` | Crea `LogitProcessor` e `Streamer` |
| `generate_text()` | Esegue la generazione vincolata |
| `setup_logging()` | Configura il logging |
| `create_prompt()` | Costruisce prompt strutturati per chat model |
| `chat_template` | Template Jinja2 per modelli chat |
| `regex_dict` | Dizionario di regex predefinite per terminali |

**Esempio di import tipico:**
```python
from grammarllm import (
    get_parsing_table_and_map_tt,
    generate_grammar_parameters,
    generate_text,
    setup_logging,
    create_prompt,
    chat_template,
)
```

---

### 4.3 `grammarllm/generate_with_constraints.py`

**File centrale della pipeline.** Contiene le funzioni di orchestrazione:

#### `get_parsing_table_and_map_tt(tokenizer, productions, regex_dict=None)`

Orchestra le prime 3 fasi della pipeline:

1. **Crea** un `ProductionRuleProcessor` e processa la grammatica utente → `final_grammar`
2. **Aggiunge** il token EOS alla regola iniziale `S*`
3. **Genera** la tabella di parsing LL(1) con `parsing_table(final_grammar)`
4. **Genera** la mappa terminali→token con `generate_token_maps(tokenizer, pars_tab, regex_dict)`

```python
# Esempio di utilizzo
pars_table, map_terminal_tokens = get_parsing_table_and_map_tt(
    tokenizer,
    productions=productions,
    regex_dict=regex_dict,  # Opzionale
)
```

**Dettaglio:** L'aggiunta di EOS alla regola `S*` è cruciale: permette al modello di terminare la generazione quando la grammatica è stata completamente consumata.

#### `generate_grammar_parameters(tokenizer, pars_tab, map_terminal_tokens)`

Crea i due oggetti chiave per la generazione:

- **`PushdownAutomaton`**: automa a pila inizializzato con la tabella di parsing
- **`MaskLogitsProcessor`**: processore che maschera i logit dei token non validi
- **`BaseStreamer`**: streamer che aggiorna lo stato del PDA durante la generazione

```python
LogitProcessor, Streamer = generate_grammar_parameters(
    tokenizer, pars_table, map_terminal_tokens
)
```

#### `generate_text(model, tokenizer, text, logit_processor, streamer, ...)`

Esegue la generazione vera e propria. Parametri:

| Parametro | Descrizione |
|-----------|-------------|
| `model` | Modello causale Hugging Face |
| `tokenizer` | Tokenizer associato |
| `text` | Prompt di input (stringa o lista di messaggi) |
| `logit_processor` | Istanza di `MaskLogitsProcessor` |
| `streamer` | Istanza di `BaseStreamer` |
| `chat_template` | Template Jinja2 per chat model (opzionale) |
| `max_new_tokens` | Massimo numero di token da generare (default: 400) |
| `do_sample` | Abilita sampling invece di greedy decoding |
| `top_p` | Parametro nucleus sampling |

**Dettagli implementativi:**

- Se `text` è una lista (messaggi), applica il `chat_template` tramite `tokenizer.apply_chat_template()`
- **Forza `num_beams=1`** (beam search non è compatibile con la generazione vincolata)
- Sposta automaticamente input_ids e attention_mask sul device del modello
- Se `do_sample=False`, rimuove i parametri `temperature` e `top_p`

#### `setup_logging()`

Configura il logging su file (`grammarllm/temp/GRAM-GEN.log`), utile per debug.

---

### 4.4 `grammarllm/utils/toolbox.py`

**Funzioni di utilità per la costruzione dei prompt.**

#### `chat_template`

Template Jinja2 per formattare messaggi in stile chat:

```
<|system|> ... <|user|> ... <|assistant|> ...
```

#### `create_prompt(prompt_input, system_prompt, examples)`

Costruisce una lista di messaggi strutturati:

```python
messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": example_user_1},
    {"role": "assistant", "content": example_assistant_1},
    # ... altri esempi ...
    {"role": "user", "content": prompt_input},
]
```

**Esempio:**
```python
prompt = create_prompt(
    prompt_input="It's raining and I feel a bit down.",
    system_prompt="You are a classification assistant...",
    examples=[
        {"role": "user", "content": "I just got a promotion!"},
        {"role": "assistant", "content": "positive joyful"},
    ]
)
```

---

### 4.5 `grammarllm/utils/common_regex.py`

**Raccolta di espressioni regolari predefinite** per identificare pattern comuni nel vocabolario del tokenizer:

| Chiave | Regex | Corrispondenza |
|--------|-------|----------------|
| `regex_alfanum` | `[a-zA-Z0-9]+` | Stringhe alfanumeriche |
| `regex_letters` | `[a-zA-Z]+` | Solo lettere |
| `regex_number` | `\d+` | Numeri interi |
| `regex_decimal` | `\d+([.,]\d+)?` | Numeri decimali |
| `regex_var` | `[a-zA-Z_][a-zA-Z0-9_]*` | Nomi di variabili |

Queste regex vengono usate nella mappatura terminali→token per associare simboli terminali a specifici token ID nel vocabolario.

---

### 4.6 `grammarllm/scripts/grammar_generation.py`

**Il file più complesso del progetto (~350 righe).** Contiene la classe `ProductionRuleProcessor` che converte la grammatica definita dall'utente in una grammatica LL(1) pronta per il parsing.

#### Classe `ProductionRuleProcessor`

**Scopo:** Trasformare una grammatica con tag `<<...>>` in una grammatica equivalente con token subword espliciti e non-terminali fattorizzati.

**Attributi principali:**
- `tokenizer`: tokenizer Hugging Face per tokenizzare i tag
- `nt_counter`: contatore per generare nomi univoci di non-terminali
- `tag_to_nt_mapping`: mappa tag originali → sequenze di token/non-terminali
- `non_terminals`: insieme di tutti i non-terminali
- `rule_specific_grammars`: grammatiche specifiche per ogni regola

#### Metodo: `extract_tags_and_others(rhs_list)`

Analizza le produzioni per separare **tag** `<<...>>` da **testo normale** (other).

**Esempio:**
```python
# Input:  "<<positive >> A"
# Output: [[("tag", "positive "), ("other", "A")]]
```

#### Metodo: `tokenize_tag(tag)`

Converte un tag in una lista di token usando il tokenizer. Esempio con LLaMA:

```python
# Input:  "positive "  (notare lo spazio)
# Output: ["positive", " "]   oppure  ["pos", "itive", " "]
```

#### Metodo: `get_prefix_groups_for_rule(tags, rule_name)`

**Fase chiave dell'algoritmo LL(prefix).** Raggruppa i tag che condividono lo stesso primo token (prefisso comune).

**Esempio concreto:**
```python
tags = ["happy", "peaceful", "joyful"]
# Con LLaMA: i token potrebbero essere diversi
# Se "happy" = ["happy"], "peaceful" = ["peaceful"], "joyful" = ["joyful"]
# Nessun prefisso condiviso → tutti ungrouped
# 
# Se invece: "happiness" = ["happ", "iness"], "happy" = ["happ", "y"]
# Prefisso condiviso "happ" → raggruppati!
```

#### Metodo: `create_initial_grammar_for_rule(prefix_groups, ungrouped_tags, rule_name)`

Crea la grammatica iniziale con non-terminali per i gruppi con prefisso condiviso.

Per ogni gruppo con prefisso `p`, crea un non-terminale `NT_i` e produzione: `p NT_i`.

**Esempio:**
```python
# Input: prefix_groups = {"happ": [("happy", ["y"]), ("happiness", ["iness"])]}
#        ungrouped_tags = ["sad"]
# Output grammar:
#   ("A_TAG_NT1", "happ") → [["y"], ["iness"]]
# Con tag_mapping:
#   "A::happy"     → "happ A_TAG_NT1"
#   "A::happiness" → "happ A_TAG_NT1"
#   "A::sad"       → "sad"
```

#### Metodo: `process_grammar_iteration(grammar)`

Esegue **iterazioni di raffinamento** sulla grammatica. Per ogni non-terminale con più produzioni, cerca ulteriori prefissi comuni e li fattorizza ricorsivamente. Continua finché non ci sono più modifiche (`changed = False`).

#### Metodo: `find_common_prefixes_in_productions(productions)`

**Fattorizzazione dei prefissi comuni nelle produzioni.** Simile all'algoritmo standard di left-factoring per grammatiche LL(1).

**Esempio:**
```python
# Input:  [["positive", "A"], ["positive", "B"], ["negative", "C"]]
# Output: new_productions = [["negative", "C"]]
#         factorization_info = {
#             'common_prefix': ["positive"],
#             'suffixes': [["A"], ["B"]]
#         }
```

#### Metodo: `process_full_grammar(grammar_dict)` ⭐

**Il metodo principale.** Esegue l'intera pipeline di trasformazione:

1. Per ogni regola `lhs → rhs_list`:
   - Estrae tag e testo con `extract_tags_and_others()`
   - Costruisce la grammatica dei tag con `build_tag_grammar_for_rule()`
   - Crea le produzioni finali con `create_final_productions_for_rule()`
   - Applica fattorizzazione con `find_common_prefixes_in_productions()`
2. Salva la grammatica risultante

**Struttura della grammatica finale:**

```python
final_grammar = {
    # Regole principali: (non_terminale, "RULE") → lista di produzioni
    ("S*", "RULE"):      [["positive", " ", "A"]],
    ("A", "RULE"):       [["happy"], ["peaceful"], ["joyful"]],
    # Regole per tag fattorizzati: (NT_fattorizzato, prefisso) → lista di suffissi
    ("A_TAG_NT1", "happ"): [["y"], ["iness"]],
}
```

---

### 4.7 `grammarllm/scripts/generate_LL1_parsing_table.py`

**Genera la tabella di parsing LL(1)** a partire dalla grammatica processata.

#### Funzioni principali:

#### `find_first(symbol, productions, first_sets)`

Calcola il **FIRST set** per un simbolo (terminale o non-terminale). Il FIRST set è l'insieme dei terminali che possono apparire all'inizio di una stringa derivata dal simbolo.

**Algoritmo:**
1. Se il simbolo è un terminale → FIRST = `{simbolo}`
2. Se il simbolo è un non-terminale, per ogni sua produzione:
   - Se la produzione è `ε` → aggiungi `ε` al FIRST
   - Altrimenti, calcola FIRST della sequenza

#### `compute_first_of_string(symbols, first_sets)`

Calcola il FIRST per una sequenza di simboli (es. corpo di una produzione).

#### `follow(productions, first_sets, start_symbol)`

Calcola i **FOLLOW set** per ogni non-terminale. Il FOLLOW set è l'insieme dei terminali che possono apparire immediatamente dopo un non-terminale in una derivazione.

**Algoritmo iterativo:**
1. Inizializza: `FOLLOW(S*) = {$}`
2. Ripeti finché ci sono cambiamenti:
   - Per ogni produzione `A → αBβ`:
     - Aggiungi `FIRST(β) - {ε}` a `FOLLOW(B)`
     - Se `ε ∈ FIRST(β)`, aggiungi `FOLLOW(A)` a `FOLLOW(B)`

#### `compute_parsing_table(productions, first_sets, follow_sets)`

Costruisce la tabella di parsing LL(1):

```python
parsing_table[non_terminal][terminal] = produzione_da_usare
```

**Rilevamento conflitti:** Se due produzioni dello stesso non-terminale condividono un terminale nel loro FIRST, la grammatica **non è LL(1)** e viene sollevata un'eccezione.

#### `parsing_table(final_rules)` ⭐

La funzione principale esportata. Prende le regole nel formato prodotto da `ProductionRuleProcessor` e restituisce la tabella di parsing.

**Esempio di output:**
```python
{
    "S*": {
        "positive": ["positive", " ", "A"],
        "negative": ["negative", " ", "B"],
        "neutral":  ["neutral", " ", "C"]
    },
    "A": {
        "happy":     ["happy"],
        "peaceful":  ["peaceful"],
        "joyful":    ["joyful"]
    }
}
```

---

### 4.8 `grammarllm/scripts/map_terminal_tokens.py`

**Mappa i simboli terminali della grammatica ai corrispondenti token ID nel vocabolario del tokenizer.**

#### `generate_token_maps(tokenizer, table_parsing, regex_dict=None)`

Per ogni terminale nella tabella di parsing:
1. Crea una regex dal terminale: `^terminale_escapato$` (match esatto)
2. Cerca nel vocabolario tutti i token il cui testo matcha la regex
3. Associa al terminale la lista di token ID corrispondenti

Se viene fornito `regex_dict`, i terminali che corrispondono a chiavi `regex_*` vengono mappati usando la regex fornita invece del match esatto.

**Esempio:**
```python
# Terminale "positive" → regex "^positive$"
# Tokenizer LLaMA:
#   "positive" → token_id 1234
#   " positive" → token_id 5678 (con spazio!)
# map_terminal_tokens = {"positive": [1234, 5678]}
```

#### `check_tokens_conflicts(table_parsing, map_terminal_tokens)`

**Verifica che non ci siano conflitti:** per ogni coppia di terminali nella stessa regola, i loro set di token ID devono essere disgiunti. Se c'è intersezione, il PDA non potrebbe decidere deterministicamente quale terminale è stato generato.

---

### 4.9 `grammarllm/modules/PushdownAutomaton.py`

**Il cuore del sistema: l'automa a pila deterministico (DPDA).** Implementa un parser predittivo LL(1) che guida la generazione token per token.

#### `__init__(self, grammar, startSymbol, map)`

Inizializza:
- `self.stack`: pila inizializzata con il simbolo iniziale `S*`
- `self.grammar`: la tabella di parsing
- `self.map_terminals_tokens`: mappa terminali → lista di token ID
- `self.map_tokens_terminals`: mappa **inversa** token ID → lista di terminali (costruita automaticamente)

#### `get_tokens(self)` ⭐

**Il metodo più importante.** Determina quali token sono validi per il prossimo passo di generazione.

1. Chiama `recursive_get_tokens()` per esplorare lo stack e trovare tutti i terminali "raggiungibili"
2. Per ogni terminale raggiungibile, recupera i token ID associati
3. **Verifica** che i set di token siano disgiunti (assert)
4. Salva i terminali correnti in `self.current_terminals`
5. Restituisce la lista di token ID validi

**Esempio:**
```python
# Stack: ["S*"]
# S* ha produzioni: "positive" | "negative" | "neutral"
# Terminali raggiungibili: ["positive", "negative", "neutral"]
# Token validi: [1234, 5678, 9012, ...]  # tutti i token che matchano quei terminali
```

#### `recursive_get_tokens(stack, visited=None)`

Esplora ricorsivamente lo stack per trovare tutti i terminali raggiungibili. Gestisce la ricorsione infinita con un set `visited`.

#### `next_state(self, token_gen)` ⭐

**Transizione di stato.** Dato un token generato:

1. Usa `map_tokens_terminals` per risalire al terminale corrispondente
2. **Verifica** che il token corrisponda esattamente a un solo terminale (assert)
3. Chiama `next_state_terminal()` per aggiornare lo stack

#### `next_state_terminal(self, terminal)`

Aggiorna lo stack in base al terminale generato:

1. **Pop** il top dello stack
2. Se è un **non-terminale** → cerca nella tabella `grammar[top][terminal]` e **pusha** i simboli della produzione in ordine inverso, poi richiama ricorsivamente
3. Se è un **terminale** → verifica che corrisponda al token generato (assert)

**Esempio di esecuzione:**
```
Stack iniziale: ["S*"]
Token generato: "positive" (id 1234)

Step 1: pop "S*", è non-terminale
        grammar["S*"]["positive"] = ["positive", " ", "A"]
        push inverso: ["positive", " ", "A"]
        Stack: ["positive", " ", "A"]
        next_state_terminal("positive")

Step 2: pop "positive", è terminale
        "positive" == "positive" ✓
        Stack: [" ", "A"]
```

#### `eos(self)`

Restituisce `True` se lo stack è vuoto (end of string), ovvero la grammatica è stata completamente consumata.

#### `reset(self)`

Riporta l'automa allo stato iniziale (stack = `[start_symbol]`).

---

### 4.10 `grammarllm/modules/BaseStreamer.py`

**Streamer Hugging Face personalizzato** che viene chiamato a ogni token generato da `model.generate()`.

#### `put(self, value)`

Chiamato da `model.generate()` per ogni token:

1. **Prima chiamata:** contiene i token del prompt iniziale → li ignora (serve solo per inizializzare)
2. **Chiamate successive:** ogni token generato singolarmente → aggiorna il PDA con `pda.next_state(token_id)`
3. Se il PDA è già in stato EOS, termina

#### `end(self)`

Chiamato alla fine della generazione:
- Verifica che lo stack sia vuoto (altrimenti warning)
- Resetta PDA e streamer per una nuova generazione

---

### 4.11 `grammarllm/modules/SimpleLogitProcessor.py`

**Versione base del processore di logit.** Implementa `LogitsProcessor` di Hugging Face.

#### `__call__(self, input_ids, scores)`

Chiamato a ogni passo di generazione:

1. **Applica** la temperatura ai logit
2. **Ottiene** i token validi dal PDA via `pda.get_tokens()`
3. **Se ci sono token validi:** azzera i logit di tutti gli altri token (`-inf`), preservando solo i validi
4. **Se non ci sono token validi e lo stack è vuoto:** forza la generazione del token EOS
5. **Altrimenti:** passa i logit invariati (caso di errore silenzioso)

---

### 4.12 `grammarllm/modules/SimpleLogitProcessor_.py` ⭐

**Versione avanzata del processore di logit.** Aggiunge:

#### Metriche di tracciamento:

- **`points`**: lista di tuple `(entropia_normalizzata, massa_invalida)` raccolte a ogni passo
- **`preserved_mass`**: lista della massa cumulativa di probabilità preservata
- **`temperature`**: attributo per la temperatura (default 1.0)

#### Nuove funzionalità:

- **`log_valid_tokens_prob_mass()`**: calcola la massa di probabilità dei token validi e invalidi
- **`log_invalid_tokens_entropy()`**: calcola l'entropia di Shannon normalizzata della distribuzione dei token invalidi usando `scipy.stats.entropy`
- **`reset()`**: resetta le metriche per una nuova generazione
- **`generation_ended`**: flag per gestire la terminazione

#### Differenze chiave con la versione base:

| Aspetto | Versione base | Versione avanzata |
|---------|---------------|-------------------|
| Metriche | Nessuna | Entropia, massa preservata |
| Gestione errori | Continua silenziosamente | Logga errori e forza EOS |
| Temperatura | Non impostabile | Impostabile via attributo |
| Reset | Non supportato | `reset()` disponibile |

**La versione avanzata è quella effettivamente usata** dal package (importata in `generate_with_constraints.py`).

#### Visualizzazione delle metriche:

Il `main.py` include la funzione `plot_invalid_trajectory()` che visualizza i punti raccolti `(entropy, invalid_mass)` su un grafico 2D con traiettoria, centroide e distanza euclidea dall'origine.

---

### 4.13 `main.py`

**Script demo autonomo** che mostra l'uso completo della libreria.

Contiene:

1. **`plot_invalid_trajectory(points)`**: funzione per visualizzare la traiettoria delle metriche
2. **Reimplementazione** delle funzioni `get_parsing_table_and_map_tt`, `generate_grammar_parameters`, `setup_logging`, `generate_text` (duplicate da `generate_with_constraints.py`)
3. **`main()`**: esempio di classificazione gerarchica con LLaMA 3.2 1B

**Nota:** `main.py` non fa parte del package installabile, è un esempio standalone e contiene codice duplicato rispetto al modulo `grammarllm`.

---

### 4.14 `QuickStart.ipynb`

**Notebook Jupyter** con 3 esempi pronti all'uso:

1. **Classificazione gerarchica** (positive/negative/neutral → sottocategorie)
2. **Restrizione del vocabolario** (solo parole autorizzate)
3. **Generazione strutturata RDF** (conversione NL → triple RDF)

---

## 5. La Grammatica LL(prefix)

GrammarLLM introduce una notazione chiamata **LL(prefix)**, una generalizzazione di LL(1) per gestire la tokenizzazione subword.

### Notazione

| Sintassi | Significato |
|----------|-------------|
| `<<testo>>` | Stringa esatta — il sistema gestisce la tokenizzazione automaticamente |
| `simbolo` (senza `<<>>`) | Terminale — viene mappato a un set di token tramite regex |
| `A, B, C` (maiuscole) | Non-terminali |
| `S*` | Simbolo iniziale (start symbol) |
| `ε` | Transizione epsilon (stringa vuota) |

### Perché LL(prefix)?

In una grammatica LL(1) tradizionale, il parser decide quale produzione usare guardando **1 token** di lookahead. Ma con i tokenizer subword, un tag come `<<happy>>` potrebbe essere tokenizzato come `["happ", "y"]` o `["happy"]` a seconda del modello.

LL(prefix) risolve questo problema:
- Raggruppa i tag che condividono il primo token (prefisso comune)
- Crea non-terminali intermedi per disambiguare i suffissi
- Il PDA può quindi decidere deterministicamente al primo token, poi raffinare la scelta con i token successivi

---

## 6. Esempi di Utilizzo

### 6.1 Classificazione Gerarchica

**Obiettivo:** Forzare l'LLM a classificare un input in una gerarchia predefinita di categorie.

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from grammarllm import (
    generate_grammar_parameters, generate_text,
    get_parsing_table_and_map_tt, create_prompt, chat_template,
)

# Definizione della grammatica gerarchica
productions = {
    'S*': ["<<positive >> A", "<<negative >> B", "<<neutral >> C"],
    'A':  ["<<happy>>", "<<peaceful>>", "<<joyful>>"],
    'B':  ['<<sad>>', '<<angry>>', '<<frustrated>>'],
    'C':  ['<<calm>>', '<<indifferent>>', '<<unemotional>>']
}

# Prompt con esempi
system_prompt = "Sei un classificatore gerarchico..."
examples = [
    {"role": "user", "content": "I just got a promotion!"},
    {"role": "assistant", "content": "positive joyful"},
    # ... altri esempi ...
]

prompt = create_prompt(
    prompt_input="It's raining and I feel a bit down.",
    system_prompt=system_prompt,
    examples=examples
)

# Carica il modello
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")

# Prepara i vincoli grammaticali
pars_table, map_tt = get_parsing_table_and_map_tt(tokenizer, productions)
LogitProcessor, Streamer = generate_grammar_parameters(tokenizer, pars_table, map_tt)

# Genera!
output = generate_text(model, tokenizer, prompt, LogitProcessor, Streamer, chat_template)
print(output)  # → "negative sad"
```

**Cosa succede internamente:**
1. La grammatica viene convertita in LL(1): `S* → positive A | negative B | neutral C`, ecc.
2. Il PDA inizia con stack `["S*"]`
3. A ogni passo, solo i token validi (es. `"positive"`, `"negative"`, `"neutral"`) hanno logit > -inf
4. Il modello sceglie il token più probabile tra quelli validi
5. Il PDA si aggiorna e restringe ulteriormente i token validi

### 6.2 Restrizione del Vocabolario

**Obiettivo:** L'LLM può usare solo parole di un insieme predefinito.

```python
productions = {
    'S*': [
        "<< Yes>> S*", "<< I'm>> S*", "<< very>> S*",
        "<< happy>> S*", "<< !>> S*", "<< so>> S*",
        "<< thanks>> S*", "<< great>> S*", # ...
    ]
}
```

In questo caso, ogni produzione di `S*` è ricorsiva (`... S*`), permettendo all'LLM di generare sequenze di lunghezza arbitraria usando solo le parole autorizzate.

**Output esempio:** `"I'm very happy !"`

### 6.3 Generazione Strutturata (RDF)

**Obiettivo:** Generare triple RDF sintatticamente corrette.

```python
productions = {
    'S*': ["SUBJ PRED OBJ . S*"],
    'SUBJ': ["IRI", "BLANKNODE"],
    'PRED': ["IRI"],
    'OBJ': ["IRI", "BLANKNODE", "LITERAL"],
    'IRI': ["< URI >"],
    'LITERAL': ['" STRING " DESCRIPTION_LANG'],
    # ...
}
```

Qui i terminali come `<`, `>`, `"` sono mappati a token ID tramite regex esatte (es. `regex_<` = `^<$`), mentre `URI` e `STRING` sono non-terminali con le loro regole.

**Output esempio:**
```
<http://example.org/people/GiovanniBianchi> <http://example.org/properties/hasAge> "30" ^^<http://www.w3.org/2001/XMLSchema#integer> .
```

---

## 7. Flusso dei Dati

Ecco il percorso completo dei dati attraverso il sistema:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                            FLUSSO DEI DATI                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  INPUT UTENTE                                                               │
│  ┌──────────────────────────────────┐                                      │
│  │ productions = {                  │                                      │
│  │   'S*': ["<<positive>> A", ...], │                                      │
│  │   'A':  ["<<happy>>", ...]       │                                      │
│  │ }                                │                                      │
│  └──────────────┬───────────────────┘                                      │
│                 │                                                            │
│     ┌───────────▼───────────┐                                              │
│     │ ProductionRuleProcessor│                                             │
│     │ process_full_grammar() │                                             │
│     └───────────┬───────────┘                                              │
│                 │                                                            │
│                 ▼                                                            │
│     ┌───────────────────────────┐                                          │
│     │ final_grammar (LL(1))     │                                          │
│     │ {                         │                                          │
│     │   ("S*","RULE"): [[...]], │                                          │
│     │   ("A","RULE"):  [[...]], │                                          │
│     │ }                         │                                          │
│     └───────────┬───────────────┘                                          │
│                 │                                                            │
│     ┌───────────▼───────────┐     ┌──────────────────┐                     │
│     │ parsing_table()       │     │ tokenizer        │                     │
│     │ (FIRST, FOLLOW, LL(1))│     │ (vocab + regex)  │                     │
│     └───────────┬───────────┘     └────────┬─────────┘                     │
│                 │                           │                               │
│                 ▼                           ▼                               │
│     ┌─────────────────────┐     ┌──────────────────────┐                   │
│     │ pars_tab            │     │ map_terminal_tokens  │                   │
│     │ {                   │     │ {"positive": [1234,  │                   │
│     │   "S*": {"positive":│     │              5678],  │                   │
│     │     ["pos","A"],...}│     │  "happy": [9012],...}│                   │
│     │ }                   │     │ }                    │                   │
│     └──────────┬──────────┘     └──────────┬───────────┘                   │
│                │                            │                               │
│                └──────────┬─────────────────┘                               │
│                           │                                                 │
│                           ▼                                                 │
│     ┌─────────────────────────────────────────┐                            │
│     │         PushdownAutomaton               │                            │
│     │  stack: ["S*"]                          │                            │
│     │  grammar: pars_tab                      │                            │
│     │  map_terminals_tokens: map_terminal...  │                            │
│     │  map_tokens_terminals: inversa          │                            │
│     └──────────────────┬──────────────────────┘                            │
│                        │                                                    │
│          ┌─────────────┴─────────────┐                                     │
│          ▼                           ▼                                     │
│  ┌───────────────┐           ┌───────────────┐                             │
│  │ LogitProcessor│           │   Streamer    │                             │
│  │ maschera      │           │   aggiorna    │                             │
│  │ token non     │◄──────────│   stato PDA   │                             │
│  │ validi (-inf) │           │               │                             │
│  └───────┬───────┘           └───────┬───────┘                             │
│          │                           │                                      │
│          │     ┌─────────────────────┘                                     │
│          │     │                                                            │
│          ▼     ▼                                                            │
│  ┌───────────────────────────────────────┐                                │
│  │          model.generate()             │                                │
│  │  Ad ogni passo:                       │                                │
│  │    1. LogitProcessor maschera logit   │                                │
│  │    2. Softmax → probabilità           │                                │
│  │    3. Campiona/argmax token           │                                │
│  │    4. Streamer aggiorna PDA           │                                │
│  │    5. Ripeti fino a EOS               │                                │
│  └───────────────────┬───────────────────┘                                │
│                      │                                                      │
│                      ▼                                                      │
│  ┌───────────────────────────────────────┐                                │
│  │  OUTPUT: "positive happy"             │                                │
│  │  (testo vincolato dalla grammatica)   │                                │
│  └───────────────────────────────────────┘                                │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Il Ciclo di Generazione nel Dettaglio

Per ogni passo di generazione:

1. **`MaskLogitsProcessor.__call__(input_ids, scores)`**
   - `scores` = logit grezzi dal modello (dimensione: `[1, vocab_size]`)
   - Applica temperatura: `scores = scores / temperature`
   - `pda.get_tokens()` → lista di token ID validi
   - Crea `filtered_scores` con `-inf` ovunque tranne che per i token validi
   - Restituisce `filtered_scores`

2. **`model.generate()`**
   - Applica softmax ai logit filtrati
   - Sceglie il token (greedy: argmax, sampling: multinomial)
   - Chiama `streamer.put(token_id)`

3. **`BaseStreamer.put(value)`**
   - Decodifica il token per logging
   - Chiama `pda.next_state(token_id)`

4. **`PushdownAutomaton.next_state(token_id)`**
   - Converte token_id → terminale (via `map_tokens_terminals`)
   - Aggiorna lo stack con `next_state_terminal(terminale)`

5. **Ripeti** dal passo 1 finché:
   - Lo stack è vuoto (EOS) → forza generazione del token EOS
   - O si raggiunge `max_new_tokens`

---

## Riepilogo

GrammarLLM è una libreria che combina **teoria dei linguaggi formali** (grammatiche LL(1), automi a pila) con **modelli linguistici moderni** (Transformer, Hugging Face) per garantire che l'output di un LLM rispetti una sintassi predefinita.

Il sistema funziona mascherando dinamicamente i logit del modello a ogni passo di generazione, permettendo solo i token che sono sintatticamente validi secondo un parsing predittivo LL(1). Questo garantisce **correttezza sintattica** senza modificare il modello sottostante.

La novità principale è l'algoritmo **LL(prefix)** che gestisce automaticamente la tokenizzazione subword, permettendo all'utente di definire grammatiche con stringhe esatte (`<<...>>`) senza preoccuparsi di come il tokenizer le scompone.
