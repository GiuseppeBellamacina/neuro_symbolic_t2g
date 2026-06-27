# 🔍 Errori e Migliorie di GrammarLLM

> **ATTENZIONE:** Questo documento ha scopo **analitico**. Non contiene modifiche al codice, solo osservazioni e suggerimenti.

---

## Indice

1. [Errori e Bug](#1-errori-e-bug)
2. [Problemi di Design e Architettura](#2-problemi-di-design-e-architettura)
3. [Migliorie Proposte](#3-migliorie-proposte)
4. [Suggerimenti per il Futuro](#4-suggerimenti-per-il-futuro)

---

## 1. Errori e Bug

### 1.1 `raise` con stringa invece di Exception

**File:** `grammarllm/modules/BaseStreamer.py`, riga 19

```python
raise "ERROR ON PDA RESET"
```

**Problema:** In Python, `raise "stringa"` solleva un'eccezione di tipo `str`, che è deprecato e non funziona in Python 3. Il codice non va in esecuzione — si ottiene `TypeError: exceptions must derive from BaseException`.

**Correzione suggerita:**
```python
raise RuntimeError("ERROR ON PDA RESET: PDA già in stato finale prima della generazione. "
                    "Chiamare pda.reset() prima di iniziare una nuova generazione.")
```

**Impatto:** **Bloccante.** Se il PDA è già in EOS prima della generazione, il codice crasha con un TypeError invece di dare un messaggio d'errore utile.

---

### 1.2 `AttributeError` potenziale in `next_state()`

**File:** `grammarllm/modules/PushdownAutomaton.py`, metodo `next_state()`

```python
def next_state(self, token_gen):
    logging.info(f"current terminals is:{self.current_terminals}")
    check_terminals = set(self.map_tokens_terminals[token_gen]).intersection(
        set(self.current_terminals))
```

**Problema:** `self.current_terminals` viene impostato solo dentro `get_tokens()`. Se `next_state()` viene chiamato prima di `get_tokens()`, o se `get_tokens()` non è mai stato chiamato, si ottiene un `AttributeError` perché l'attributo non esiste.

**Correzione suggerita:**
```python
def __init__(self, ...):
    # ...
    self.current_terminals = []  # Inizializzazione nel costruttore
```

**Impatto:** **Medio.** Normalmente il flusso LogitProcessor → Streamer evita questo problema, ma in scenari non standard può causare crash.

---

### 1.3 Streamer non gestisce correttamente `token_id = value[0]` nella prima chiamata

**File:** `grammarllm/modules/BaseStreamer.py`, metodo `put()`

```python
if self.is_first_call:
    generated_token_id = value[0]
    # ...
    return  # non processa i token del prompt
```

**Problema:** `value[0]` prende solo il primo token del prompt. Se il prompt contiene più token, tutti tranne il primo vengono ignorati silenziosamente nella prima chiamata. Inoltre i token rimanenti del prompt verranno processati come token generati nelle chiamate successive, causando potenziali errori di parsing.

**Correzione suggerita:** La prima chiamata di `put` dovrebbe consumare TUTTI i token del prompt senza processarli, oppure il tokenizer dovrebbe separare nettamente prompt da generazione.

**Impatto:** **Basso-Medio.** Funziona correttamente solo perché Hugging Face tipicamente invia l'intero prompt in una singola chiamata `put()`, ma il codice è fragile.

---

### 1.4 `next_state_terminal()` usa `print()` invece di `logging`

**File:** `grammarllm/modules/PushdownAutomaton.py`, metodo `next_state_terminal()`

```python
print("Parser Stack:", stack)
print("Comparing:", top, "vs", token)
print(top == token, f"Errore: trovato '{top}', atteso '{token}'")
```

**Problema:** Output di debug mescolato tra `print` e `logging`. In un ambiente di produzione, i `print` potrebbero non essere catturati dal sistema di logging.

**Correzione suggerita:**
```python
logging.error(f"Parser Stack: {stack}")
logging.error(f"Comparing: {top} vs {token}")
```

**Impatto:** **Basso.** Solo inconsistenza di stile.

---

### 1.5 Possibile ricorsione infinita in `recursive_get_tokens()`

**File:** `grammarllm/modules/PushdownAutomaton.py`, metodo `recursive_get_tokens()`

```python
def recursive_get_tokens(self, stack, visited=None):
    if visited is None:
        visited = set()
    if not stack:
        return []
    top = stack.pop()
    if top in visited:
        return []
    visited.add(top)
    if top not in self.grammar:
        return [top]
    tokens = []
    for symbol in self.grammar[top]:
        if symbol not in visited:
            stack.extend(reversed([symbol]))
            tokens += self.recursive_get_tokens(stack, visited)
    return tokens
```

**Problema:** Il controllo `if top in visited` previene la ricorsione infinita sullo stesso non-terminale. Tuttavia, con grammatiche che hanno ricorsione indiretta (es. `A → B`, `B → A`), la prima occorrenza di `A` verrà aggiunta a `visited`, e quando si incontra di nuovo restituirà `[]`, perdendo produzioni valide.

**Correzione suggerita:** L'approccio attuale è un'approssimazione. Per grammatiche con ricorsione sinistra o cicli, sarebbe meglio usare un algoritmo come l'espansione con profondità limitata, oppure vietare esplicitamente la ricorsione nei non-terminali.

**Impatto:** **Medio.** Non si manifesta con grammatiche "ben formate" senza ricorsione, ma può causare comportamenti errati con grammatiche più complesse.

---

### 1.6 Il metodo `eos()` nel BaseStreamer è checkato due volte

**File:** `grammarllm/modules/BaseStreamer.py`, metodo `put()`

```python
def put(self, value):
    if self.pda.eos() and self.is_first_call:
        # ... raise
    if self.pda.eos():
        self.is_first_call = True
        return
```

**Problema:** Il secondo controllo `if self.pda.eos():` imposta `self.is_first_call = True` ma **non chiama `pda.reset()`**. Questo significa che:
1. Se la generazione finisce normalmente (stack vuoto), `self.is_first_call` viene resettato a `True`
2. Ma il PDA rimane con stack vuoto
3. Alla prossima generazione, il primo controllo scatterà e solleverà l'errore

**Fortunatamente**, `end()` viene sempre chiamato dopo la generazione e fa il reset completo. Ma se per qualche motivo `end()` non viene chiamato, il sistema rimane in uno stato inconsistente.

**Correzione suggerita:** Unire i due controlli o aggiungere un flag esplicito `generation_completed`.

**Impatto:** **Basso.** `end()` è chiamato da Hugging Face alla fine della generazione, quindi il reset avviene comunque.

---

### 1.7 Percorso di salvataggio inconsistente

**File:** `grammarllm/scripts/grammar_generation.py`, metodo `save_final_grammar()`

```python
output_filename = os.path.join("output/temp", filename)
```

**File:** `grammarllm/generate_with_constraints.py`, `setup_logging()`

```python
log_dir = 'grammarllm/temp'
```

**File:** `grammarllm/scripts/generate_LL1_parsing_table.py`, `save_table_parsing_as_txt()`

```python
output_grammar_file = os.path.join('grammarllm/temp', 'table_parsing.json')
```

**Problema:** Tre diverse funzioni usano percorsi diversi per salvare file temporanei: `output/temp`, `grammarllm/temp`. Questo crea confusione e potenziali problemi di permessi.

**Correzione suggerita:** Centralizzare la configurazione dei percorsi in una costante o variabile d'ambiente.

**Impatto:** **Basso.** Funziona comunque, ma è disordinato.

---

### 1.8 `setdefault` non esiste nei dizionari Python

**File:** `grammarllm/scripts/grammar_generation.py`, metodo `get_prefix_groups_for_rule()`

```python
prefix_counts.setdefault(prefix, []).append((tag, tokens[1:]))
```

**Problema:** Il metodo corretto è `setdefault()` (con `t`), non `setdefault()`. Questo causa un `AttributeError` a runtime.

**Correzione suggerita:**
```python
if prefix not in prefix_counts:
    prefix_counts[prefix] = []
prefix_counts[prefix].append((tag, tokens[1:]))
# oppure:
prefix_counts.setdefault(prefix, []).append((tag, tokens[1:]))
```

**Impatto:** **CRITICO.** Questa funzione viene chiamata durante l'elaborazione della grammatica. Se ci sono tag con prefissi comuni (cioè token condivisi), il codice crasha.

---

### 1.9 `check_tokens_conflicts` ha logica potenzialmente incompleta

**File:** `grammarllm/scripts/map_terminal_tokens.py`

```python
for lhs, rhs_list in table_parsing.items():
    for a, b in itertools.combinations(rhs_list.keys(), 2):
        intersection = set(map_terminal_tokens[a]) & set(map_terminal_tokens[b])
```

**Problema:** Controlla i conflitti solo per **coppie** di terminali nella stessa regola. Tuttavia, se tre o più terminali hanno intersezione, il problema viene comunque rilevato dalle coppie. Ma il messaggio di errore potrebbe non essere chiarissimo.

Inoltre, non controlla se un singolo terminale ha conflitti con se stesso (es. regex sovrapposte nello stesso terminale), anche se questo è improbabile dato che ogni terminale ha una regex dedicata.

**Impatto:** **Molto basso.** La logica è sostanzialmente corretta.

---

## 2. Problemi di Design e Architettura

### 2.1 Due versioni del LogitProcessor

**File:** `SimpleLogitProcessor.py` e `SimpleLogitProcessor_.py`

Il progetto mantiene **due implementazioni** del `MaskLogitsProcessor`. La versione `_` (con underscore) è quella effettivamente usata e più completa. Mantenere entrambe crea confusione e potenziali incoerenze.

**Suggerimento:** Rinominare `SimpleLogitProcessor_.py` in `MaskLogitProcessor.py` e rimuovere la versione obsoleta.

---

### 2.2 Duplicazione di codice tra `main.py` e `generate_with_constraints.py`

Le funzioni `get_parsing_table_and_map_tt`, `generate_grammar_parameters`, `setup_logging`, `generate_text` sono definite due volte. `main.py` dovrebbe importare da `grammarllm` come fa il `QuickStart.ipynb` invece di ridefinire tutto.

---

### 2.3 `PushdownAutomaton` è accoppiato alla struttura specifica della grammatica

Il PDA assume che la grammatica abbia chiavi nella forma `(non_terminale, "RULE")` o `(non_terminale, prefisso)`. Questa struttura è creata da `ProductionRuleProcessor` e `parsing_table()`, ma non è documentata come contratto. Un utente che volesse creare un PDA manualmente faticherebbe a capire il formato atteso.

---

### 2.4 Assenza di interfacce formali

Le classi non implementano interfacce o classi astratte (tranne `MaskLogitProcessor` che eredita da `LogitsProcessor` di Hugging Face). Sarebbe utile avere:
- Una classe astratta per il PDA
- Una classe astratta per il GrammarProcessor
- Protocolli per la tabella di parsing

---

### 2.5 Gestione della temperatura

Nel `SimpleLogitProcessor_.py`, la temperatura è un attributo d'istanza (`self.temperature = 1.0`), e l'utente la imposta con `LogitProcessor.temperature = 1.0`. Questo è fragile: se l'utente dimentica di impostarla, usa 1.0 (nessun effetto), ma potrebbe volere un valore diverso. Sarebbe meglio accettarla come parametro nel costruttore.

---

### 2.6 `generate_text()` modifica gli argomenti per riferimento

```python
kwargs.setdefault("num_beams", 1)
kwargs.setdefault("pad_token_id", tokenizer.eos_token_id)
```

Questo modifica il dizionario `kwargs` passato dal chiamante. Se il chiamante riusa lo stesso dizionario per più chiamate, potrebbe avere effetti collaterali inaspettati.

---

## 3. Migliorie Proposte

### 3.1 Aggiungere Type Hints

Quasi tutto il codice è privo di type hints. Esempio di come potrebbe essere:

```python
def get_parsing_table_and_map_tt(
    tokenizer: PreTrainedTokenizer,
    productions: dict[str, list[str]],
    regex_dict: dict[str, re.Pattern] | None = None
) -> tuple[dict, dict]:
    ...
```

**Beneficio:** Migliore esperienza IDE, documentazione automatica, prevenzione bug.

---

### 3.2 Aggiungere Unit Test

Non ci sono test nel progetto. Sarebbe fondamentale aggiungere test per:
- `ProductionRuleProcessor`: test con vari tipi di grammatiche
- `parsing_table()`: verifica FIRST, FOLLOW, tabella di parsing
- `PushdownAutomaton`: test di transizioni di stato
- `MaskLogitsProcessor`: test di mascheramento logit
- `BaseStreamer`: test del flusso di token
- `generate_token_maps()`: test con e senza regex_dict

---

### 3.3 Migliorare la Documentazione delle Funzioni

Molte funzioni hanno docstring assenti o incomplete. Esempio:

```python
def get_tokens(self):
    """Restituisce la lista di token ID validi per il prossimo passo di generazione.
    
    Esplora ricorsivamente lo stack del PDA per trovare tutti i terminali raggiungibili,
    poi li mappa ai rispettivi token ID. Verifica che i set di token siano disgiunti.
    
    Returns:
        list[int]: Lista di token ID validi per il prossimo passo.
        
    Raises:
        AssertionError: Se i token associati a due terminali non sono disgiunti.
    """
```

---

### 3.4 Centralizzare i Percorsi

Creare un modulo di configurazione:

```python
# grammarllm/config.py
import os

TEMP_DIR = os.path.join(os.path.dirname(__file__), 'temp')
LOG_FILE = os.path.join(TEMP_DIR, 'GRAM-GEN.log')
PARSING_TABLE_FILE = os.path.join(TEMP_DIR, 'table_parsing.json')
FINAL_GRAMMAR_FILE = os.path.join(TEMP_DIR, 'final_grammar.txt')
```

---

### 3.5 Gestione Errori Robusta

- Sostituire tutti gli `assert` con eccezioni tipizzate (`ValueError`, `RuntimeError`)
- Gli assert vengono disabilitati con `python -O` (ottimizzazione)
- Esempio: `assert top == token` → `if top != token: raise ParsingError(...)`

---

### 3.6 Validazione dell'Input

`get_parsing_table_and_map_tt()` e `generate_grammar_parameters()` non validano i parametri in ingresso. Sarebbe utile aggiungere controlli:

```python
def get_parsing_table_and_map_tt(tokenizer, productions, regex_dict=None):
    if not productions:
        raise ValueError("productions cannot be empty")
    if 'S*' not in productions:
        raise ValueError("Start symbol 'S*' must be defined in productions")
    # ...
```

---

### 3.7 Migliorare `recursive_get_tokens` con caching

Il metodo esplora ricorsivamente lo stack. Con grammatiche grandi, potrebbe esplorare gli stessi non-terminali molte volte. Aggiungere memoizzazione:

```python
from functools import lru_cache

@lru_cache(maxsize=128)
def _get_terminals_for_nt(self, nt):
    """Restituisce i terminali raggiungibili da un non-terminale."""
    ...
```

---

### 3.8 Rendere `MaskLogitsProcessor` configurabile

Parametri come `temperature` dovrebbero essere passati al costruttore, non impostati come attributi dopo la creazione:

```python
class MaskLogitsProcessor(LogitsProcessor):
    def __init__(self, tokenizer, pda, temperature=1.0, track_metrics=True):
        self.temperature = temperature
        self.track_metrics = track_metrics
        # ...
```

---

### 3.9 Documentare la struttura della grammatica finale

Aggiungere un modulo o un file che descriva il formato esatto di `final_grammar` e `pars_tab`:

```python
"""
Formato di final_grammar:
    dict con chiavi:
    - (non_terminal, "RULE"): list[list[str]]
      Produzioni principali. Ogni produzione è una lista di token stringa.
    - (non_terminal, prefix): list[list[str]]
      Regole per tag fattorizzati. Il prefisso è il primo token condiviso.

Formato di pars_tab:
    dict[non_terminal, dict[terminal, list[str]]]
    Tabella di parsing LL(1) standard.
"""
```

---

### 3.10 Aggiungere supporto per `ruff`/`mypy`

Configurare strumenti di linting e type-checking nel `pyproject.toml`:

```toml
[tool.ruff]
line-length = 100
target-version = "py310"

[tool.mypy]
python_version = "3.10"
strict = false
```

---

## 4. Suggerimenti per il Futuro

### 4.1 Supporto per grammatiche pesate

Permettere di associare pesi alle produzioni per influenzare la probabilità di scelta tra alternative valide (oltre il semplice mascheramento).

### 4.2 Supporto per beam search condizionato

Anche se il beam search pieno non è compatibile con il PDA deterministico, si potrebbe implementare una forma limitata di beam search che esplori solo rami validi.

### 4.3 Streaming dell'output

Implementare un generatore Python che yielda token uno alla volta invece di restituire l'intera stringa alla fine.

### 4.4 Visualizzazione interattiva

Creare un'interfaccia che mostri in tempo reale lo stack del PDA, i token validi, e le probabilità durante la generazione.

### 4.5 Supporto per più LLM

Testare e documentare la compatibilità con altri modelli (Mistral, Falcon, Gemma, ecc.).

### 4.6 Parallelizzazione

Permettere di generare più output vincolati in parallelo con batch processing.

### 4.7 Integrazione con instructor/outlines

Valutare l'integrazione con librerie simili come `instructor` o `outlines` per offrire una gamma più ampia di strategie di generazione vincolata.

---

## Riepilogo delle Priorità

| Priorità | Problema | Impatto |
|----------|----------|---------|
| 🔴 CRITICA | `setdefault` invece di `setdefault` (1.8) | Crash nella generazione della grammatica |
| 🔴 CRITICA | `raise "stringa"` invece di `raise Exception` (1.1) | Crash nel reset del PDA |
| 🟡 MEDIA | `current_terminals` non inizializzato (1.2) | Crash in scenari atipici |
| 🟡 MEDIA | Ricorsione indiretta in `recursive_get_tokens` (1.5) | Comportamento errato con grammatiche cicliche |
| 🟢 BASSA | `print` invece di `logging` (1.4) | Inconsistenza |
| 🟢 BASSA | Percorsi di salvataggio inconsistenti (1.7) | Disorganizzazione |
| 🔵 MIGLIORIA | Mancanza di type hints (3.1) | Manutenibilità |
| 🔵 MIGLIORIA | Mancanza di unit test (3.2) | Qualità |
| 🔵 MIGLIORIA | Duplicazione codice (2.2) | Manutenibilità |
