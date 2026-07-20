# Confronto grammarllm — Interna (vendored) vs Esterna (v0.5.0)

> **Data**: 20 luglio 2026
> **Scope**: Confronto read-only tra `neuro_symbolic_t2g/grammarllm/` (interna, usata dal
> progetto) e `grammarllm/` (esterna, aggiornata a v0.5.0, da NON modificare).
> Obiettivo: verificare compatibilità, identificare bug-fix utili, raccomandare
> cosa adottare e perché, senza alterare la versione esterna.

---

## TL;DR

| Aspetto | Stato |
| --- | --- |
| **Esterna è un miglioramento stretto?** | ✅ Sì — tutti i bug di `docs/ERRORI_E_MIGLIORIE.md` sono fixati o riscritti |
| **Swap diretto drop-in?** | ❌ No — 3 breaking change nei nomi modulo + API del logit processor |
| **Path PDA usato di default?** | ❌ No — `grpo_optimal.yaml` ha `use_grammarllm_pda: false` (usa `GlossVocabularyMask`) |
| **Costo migrare tutto?** | Alto (2 file da riscrivere, API completamente diversa, 784 righe di nuovo processor) |
| **Costo adottare solo i bug-fix?** | Basso (5 fix isolati, drop-in, low-risk) |
| **Raccomandazione** | **Adottare i 5 bug-fix nella copia interna** (vedi §6). Non migrare alla API esterna ora — il path PDA è raramente esercitato e il `GlossVocabularyMask` (Trie dual-root) è sufficiente. |

---

## 1. Struttura

### Interna (`neuro_symbolic_t2g/grammarllm/`)
```
grammarllm/
  __init__.py
  config.py                          ← assente in ESTERNA
  generate_with_constraints.py
  README_PACKAGE.md
  modules/
    BaseStreamer.py                  → rinominato streamer.py
    PushdownAutomaton.py             → rinominato automaton.py
    SimpleLogitProcessor_.py         → rinominato logits_processor.py
  scripts/
    __init__.py
    generate_LL1_parsing_table.py
    grammar_generation.py
    map_terminal_tokens.py
  utils/
    __init__.py
    common_regex.py
    toolbox.py
```

### Esterna (`grammarllm/` → package nidificato v0.5.0)
```
grammarllm/                          ← root repo (pyproject.toml, main.py, docs/, examples/, benchmark_tests/)
  grammarllm/
    __init__.py
    bug_analysis.md                  ← NUOVO (registro bug, 410 righe)
    generate_with_constraints.py
    README_PACKAGE.md
    modules/
      automaton.py                   ← ex PushdownAutomaton (riscritto)
      logits_processor.py            ← ex SimpleLogitProcessor_ (riscritto, 784 righe)
      lookahead.py                   ← NUOVO (token-boundary lookahead, 138 righe)
      streamer.py                    ← ex BaseStreamer (riscritto)
    scripts/
      generate_LL1_parsing_table.py
      grammar_generation.py
      map_terminal_tokens.py
    tests/                           ← NUOVO (6 file test pytest)
      test_chat_template.py
      test_decode_fidelity.py
      test_dev_review_fixes.py
      test_logit_processor_beam.py
      test_lookahead.py
      test_multi_tag.py
    utils/
      common_regex.py
      generation_analysis.py         ← NUOVO (680 righe, analisi constraint-impact)
      pydantic_to_grammar.py        ← NUOVO (695 righe, Pydantic→Grammar)
      score_utils.py                ← NUOVO (64 righe, debug helper)
      toolbox.py
```

### Delta riepilogativo

| Stato | File |
| --- | --- |
| **NUOVI in esterna** | `modules/lookahead.py`, `tests/` (6 file), `utils/generation_analysis.py`, `utils/pydantic_to_grammar.py`, `utils/score_utils.py`, `bug_analysis.md` |
| **RIMOSSI in esterna** | `config.py` |
| **RINOMINATI** | `PushdownAutomaton.py`→`automaton.py`, `BaseStreamer.py`→`streamer.py`, `SimpleLogitProcessor_.py`→`logits_processor.py` |
| **IN ENTRAMBI** (modificati) | `generate_with_constraints.py`, `generate_LL1_parsing_table.py`, `grammar_generation.py`, `map_terminal_tokens.py`, `toolbox.py` |
| **INALTERATI** | `common_regex.py`, `__init__.py` (public API surface) |

---

## 2. Versioni e metadati

| Attributo | Interna | Esterna |
| --- | --- | --- |
| **Versione** | Nessuna (snapshot pre-release, no `pyproject.toml`) | **0.5.0** (`pyproject.toml:3`) |
| **Python** | Nessun vincolo | `>=3.10` |
| **Dipendenze** | Non dichiarate | `regex`, `torch`, `transformers>=4.30.0`, `rich`, `pydantic>=2.12.5` |
| **Build system** | Nessuno | `setuptools>=77` |
| **Test** | Nessuno | 6 file `pytest` |
| **Docs** | Solo `README_PACKAGE.md` | Sito completo: `docs/architecture.md`, `docs/usage.md`, `docs/superpowers/`, `docs/token-boundary-lookahead.md` |
| **Change history** | Nessuna | `bug_analysis.md` documenta 20+ bug con stato del fix |
| **Licenza** | MIT (menzione in README) | MIT (file `LICENSE`) |

**Nota**: l'interna è uno snapshot pre-release senza versione. L'esterna è v0.5.0 con packaging professionale, suite di test e documentazione estesa.

---

## 3. Diff dei moduli chiave

### 3.1 PushdownAutomaton (`PushdownAutomaton.py` → `automaton.py`)

| Bug da `ERRORI_E_MIGLIORIE.md` | Interna | Esterna |
| --- | --- | --- |
| **1.2** `current_terminals` non inizializzato | ✅ FIXATO (linea 68: `self.current_terminals: list[str] = []`) | ✅ FIXATO (linea 131: chiama `self.get_tokens()` in `__init__`) |
| **1.4** `print()` invece di `logging` | ✅ FIXATO (linea 237: `logger.error(...)`) | ✅ FIXATO (linea 433: `raise ValueError(...)`) |
| **1.5** Ricorsione/visited-set fragile in `recursive_get_tokens` | ⚠️ ANCORA FRAGILE (visited-set, linee 93–135) | ✅ RISCRITTO (scan iterativo FIRST-of-stack, linee 204–278, O(stack)) |

Nuove capacità in `automaton.py` (esterna):
- **`clone()`** (linee 133–180) — copia leggera che condivide strutture read-only; critico per beam search corretto
- **`residue`/`lookahead`/`regex_terminals`** — stato per il motore token-boundary lookahead
- **`apply_lookahead_path()`** (linee 438–466) — consuma token merged che attraversano boundary grammaticali
- **`eos()`** controlla `not self.stack and not self.residue` (linea 491) — detection EOS residue-aware
- **`get_tokens()`** (linee 280–330) — check disgiunzione via `isdisjoint` invece di `IntersectionError`
- Type hints completi

### 3.2 BaseStreamer (`BaseStreamer.py` → `streamer.py`)

| Bug da `ERRORI_E_MIGLIORIE.md` | Interna | Esterna |
| --- | --- | --- |
| **1.1** `raise "string"` | ✅ FIXATO (linea 25: `raise RuntimeError(...)`) | ✅ FIXATO (no raise; log warning e return, linee 143–156) |
| **1.3** Handling fragile dei token del prompt | ⚠️ PARZIALMENTE (normalizza value a list, linee 36–43) | ✅ FIXATO (stessa logica, codice più chiaro, linee 159–163) |
| **1.6** `eos()` checkato due volte, reset inconsistente | ✅ FIXATO (`end()` resetta streamer+PDA, linee 68–89) | ✅ FIXATO (`end()` resetta tutti i PDA in `self.pdas`, linee 190–234) |

Differenze architetturali:
- Esterna accetta **lista di PDA** (`self.pdas`), non PDA singolo → supporta multi-prompt batching
- Esterna **disabilita gli update di stato PDA** in `put()` — lo stato è gestito dal `StatelessLogitsProcessor`; lo streamer è solo logging durante la generazione (linee 181–188)
- Interna ancora chiama `pda.next_state()` in `put()` (linea 61)

### 3.3 Logit Processor (`SimpleLogitProcessor_.py` → `logits_processor.py`)

**La rewrite più sostanziosa.** Interna: `MaskLogitsProcessor` (266 righe). Esterna: `StatelessLogitsProcessor` (784 righe).

| Feature | Interna | Esterna |
| --- | --- | --- |
| **Beam search** | ❌ Non supportato | ✅ Completo via re-simulation |
| **Gestione stato** | Stateful (un PDA, update in-place) | Stateless (cache LRU + re-simulation) |
| **Multi-PDA** | PDA singolo | `PdaSet` — multi-stato per lookahead ambiguity (linee 92–156) |
| **Token-boundary lookahead** | ❌ Nessuno | ✅ Trie-guided DFS (`lookahead.py`) |
| **Cache eviction** | ❌ Nessuna cache | ✅ LRU con `_MAX_CACHE_SIZE = 2048` (linea 61) |
| **Beam retirement** | ❌ Crash su token invalidi | ✅ `_retire()` gestisce HF `-inf` beam fill (linee 678–704) |
| **Temperature** | Applicata nel processor (rischio double-scaling) | ❌ Rimossa — gestita solo da HF `generate()` (linee 405–409) |
| **Score history** | Sempre tracciata (rischio OOM) | ✅ Gated dietro `track_score_history` (linee 561–568) |
| **Logging dettagli** | `print` + `logging.info` | Rich Table con guard DEBUG (linee 301–353) |

La classe `MaskLogitsProcessor` **non esiste più** in esterna → **BREAKING CHANGE**.

### 3.4 `generate_with_constraints.py` (entry point)

| Aspetto | Interna | Esterna |
| --- | --- | --- |
| **`generate_grammar_parameters()` ritorna** | `(MaskLogitsProcessor, BaseStreamer)` | `(list[PushdownAutomaton], BaseStreamer)` ← **BREAKING** |
| **`generate_text()` ritorna** | `str` | `dict`/`list[dict]` con `text`, `probability`, `log_prob`, `pda_history`, `pda_stack`, `scores` |
| **Beam search** | ❌ `num_beams=1` hardcoded (linea 109) | ✅ Completo con re-simulation |
| **`num_return_sequences`** | ❌ Non supportato | ✅ Multiple sequences, sorted per probabilità |
| **Batch prompts** | ❌ Singolo prompt | ✅ `list[str]` o `list[list[dict]]` |
| **Chat template** | ❌ Forza legacy o nessuno | ✅ Template nativo tokenizer di default; fallback documentato |
| **Padding** | ❌ Nessuno | ✅ Left padding enforced per decoder-only |
| **Error re-raise** | ❌ `raise RuntimeError(e)` nudo | ✅ `raise RuntimeError(...) from e` (linea 576) |

### 3.5 `scripts/grammar_generation.py`

| Bug da `ERRORI_E_MIGLIORIE.md` | Interna | Esterna |
| --- | --- | --- |
| **1.8** typo `setdefault` | ❌ **NON È UN BUG** — interna usa già `setdefault` (corretto, linea 71) | ✅ Stesso (linea 245) |
| **1.7** Path inconsistente | Usa `config.TEMP_DIR` (linea 401) | Usa path ancorato a `package_dir` (linea 934) |

Aggiunte esterne:
- **Multi-tag position grouping** (linee 788–865) — fixa corruzione silenziosa della gerarchia quando una produzione ha tag multipli a posizioni diverse con continuazioni diverse
- **Ordinamento deterministico** — `sorted()` su `prefix_groups` e `ungrouped_tags` per riproducibilità (linee 255–261)
- **Validazione empty-token** — raise `ValueError` su tag che tokenizzano a `[]` (linee 231–239)
- **Deep copy** nell'iterazione invece di shallow (linee 422–428)
- **Warning duplicate-tag** — warna quando due tag tokenizzano identicamente (linee 297–306)
- Docstring completi (500+ righe di docs)

### 3.6 `scripts/generate_LL1_parsing_table.py`

- **Interna**: `find_first()` usa DFS ricorsivo con sentinella — **rotto per mutua ricorsione** (BUG-12)
- **Esterna**: nuovo `compute_all_first_sets()` con fixed-point iteration — **corretto** per mutua ricorsione (linee 132–209)
- Esterna aggiunge docstring completi per tutte le funzioni
- Esterna usa `package_dir` invece di `config.TEMP_DIR`

### 3.7 `scripts/map_terminal_tokens.py`

- **Esterna**: indurito `check_tokens_conflicts` con `.get()` — no crash su lookahead terminals mancanti (linee 138–142)
- **Esterna**: emette warning quando un terminale non ha token matching (linee 183–187)
- **Esterna**: aggiunge canale metadata `REGEX_TERMINALS_KEY` per il motore lookahead (linee 197–199)

---

## 4. Nuove capacità in ESTERNA

### 4.1 `modules/lookahead.py` — Token-boundary lookahead (138 righe)
**Cosa fa**: Costruisce un `VocabTrie` sul vocabolario del tokenizer e fa DFS sulle transizioni PDA per trovare TUTTI i token realizzabili dallo stato corrente — inclusi token che **fondono attraverso boundary grammaticali** o finiscono mid-terminal. Permette al modello di emettere i suoi **token merged nativi** invece di essere forzato a spelling character-by-character.

**Benefit T2G**: **ALTO**. Qwen2.5 usa BPE subword tokenization. Senza lookahead, il PDA forza emissioni single-BPE-token, frammentando gloss come `"IX-me"` in `["IX", "-", "me"]`. Lookahead permetterebbe la tokenizzazione nativa.

### 4.2 `utils/generation_analysis.py` — Analisi constraint-impact (680 righe)
**Cosa fa**: Calcola per-step preserved probability mass, entropia, e probabilità token per ogni step di generazione. Fornisce `plot_generation_analysis()` (4-panel matplotlib) e `compare_analyses()` per A/B testing.

**Benefit T2G**: **MEDIO**. Potrebbe analizzare se il constraint grammaticale distorce la distribuzione naturale durante la generazione gloss. Richiede `output_scores=True` (costo memoria) — principalmente tool di debugging/ricerca.

### 4.3 `utils/pydantic_to_grammar.py` — Pydantic → Grammar (695 righe)
**Cosa fa**: Converte `pydantic.BaseModel` in `productions` dict + `regex_dict`, generando grammatica JSON strict.

**Benefit T2G**: **BASSO**. T2G genera sequenze gloss, non JSON. Irrilevante salvo futuro lavoro con output strutturato (es. gloss + alignment metadata).

### 4.4 `utils/score_utils.py` — Helper score (64 righe)
**Benefit T2G**: **BASSO**. Piccola utility, scrivibile inline.

### 4.5 `tests/` — Suite di test (6 file)
**Cosa fa**: Test di regressione per FIRST-of-stack scan, reset(), score history gating, beam search, lookahead, multi-tag, chat template, decode fidelity.

**Benefit T2G**: **ALTO** (confidenza). Una suite di test significa che la versione esterna è validata contro failure mode note. La versione interna ha zero test.

### 4.6 `bug_analysis.md` — Registro bug (410 righe)
Documenta 20+ bug con stato di risoluzione.

**Benefit T2G**: **MEDIO** (documentazione). Fornisce tracciabilità per tutti i fix.

### 4.7 Beam search support
**Benefit T2G**: **MEDIO**. GRPO training usa sampling, non beam search. Ma beam search durante evaluation potrebbe migliorare la qualità gloss a costo di velocità inference.

### 4.8 Type hints e docstring
**Benefit T2G**: **MEDIO**. Ogni modulo esterno ha type hints e docstring completi (IT+EN). Interna ne ha minimi.

---

## 5. Assessment compatibilità

### Import sites in neuro_symbolic_t2g

**File 1: `src/grammar/gloss_grammar.py` (linee 19–24)**
```python
from grammarllm import (
    generate_grammar_parameters,
    get_parsing_table_and_map_tt,
    setup_logging,
)
from grammarllm.modules.PushdownAutomaton import PushdownAutomaton
```

**File 2: `src/grammar/grammar_logits_processor.py` (linee 28–31)**
```python
from grammarllm.modules.PushdownAutomaton import PushdownAutomaton
from grammarllm.modules.SimpleLogitProcessor_ import (
    MaskLogitsProcessor as GrammarLLMMaskProcessor,
)
```

### Breaking changes per swap diretto

| Cambiamento | Severità | Dettagli |
| --- | --- | --- |
| **Rename modulo** `PushdownAutomaton`→`automaton` | 🔴 BREAKING | `from grammarllm.modules.PushdownAutomaton` fallirà |
| **Rename modulo** `SimpleLogitProcessor_`→`logits_processor` | 🔴 BREAKING | Classe `MaskLogitsProcessor` non esiste → `StatelessLogitsProcessor` con API diversa |
| **`config.py` rimosso** | 🟡 BREAKING | `setup_logging()` interna usa `config.LOG_FILE`; esterna prende `log_dir` parameter |
| **`generate_grammar_parameters()` return type** | 🔴 BREAKING | Ritorna `(list[PDA], BaseStreamer)` invece di `(MaskLogitsProcessor, BaseStreamer)` |
| **`generate_text()` return type** | 🔴 BREAKING | Ritorna `dict` con keys `text`/`probability`/`pda_stack`, non `str` |

### Impatto sugli integration point specifici

**`create_grammarllm_pipeline()` (`gloss_grammar.py:95–163`)**:
```python
logit_processor, streamer = generate_grammar_parameters(...)
pda: PushdownAutomaton = logit_processor.pda   # ← linea 152: accede .pda sul vecchio MaskLogitsProcessor
logit_processor.temperature = temperature      # ← linea 155: setta attributo temperature
```
In esterna, `generate_grammar_parameters` ritorna `(pdas, streamer)` — **lista di PDA**, non un logit processor. Il codice `logit_processor.pda` fallirebbe. Richiede rewrite per istanziare `StatelessLogitsProcessor` separatamente.

**`GrammarPDALogitsProcessor` (`grammar_logits_processor.py:262–373`)**:
Wrappa `MaskLogitsProcessor` da `grammarllm.modules.SimpleLogitProcessor_`. **Questa classe non esiste più** in esterna. Richiederebbe wrap di `StatelessLogitsProcessor` — constructor signature completamente diversa (`tokenizer, base_pdas, sequences_per_prompt, prompt_len, temperature`).

---

## 6. Raccomandazioni di adozione

### Top 5 miglioramenti da adottare (drop-in, basso rischio)

| # | Miglioramento | Tipo | Drop-in? | Rationale |
| --- | --- | --- | --- | --- |
| **1** | **Fixed-point FIRST computation** (`compute_all_first_sets`) | 🐛 Bug fix | ✅ Sì | Il vecchio DFS ricorsivo produce FIRST sets sbagliati per grammatiche con mutua ricorsione (BUG-12). La grammatica gloss T2G è semplice (singolo NT flat), ma future grammatiche con NT annidati potrebbero produrre risultati silenziosamente errati. Fix isolato a `generate_LL1_parsing_table.py`. |
| **2** | **Scan iterativo FIRST-of-stack** (rewrite `recursive_get_tokens`) | ⚡ Performance + correttezza | ✅ Sì | Scan O(stack) invece di ricorsione worst-case esponenziale. L'euristica visited-set interna può anche prunare silenziosamente continuazioni valide per catene nullable-NT (BUG-3). |
| **3** | **Generazione grammatica deterministica** | 🐛 Correttezza | ✅ Sì | `sorted()` su `prefix_groups` e `ungrouped_tags` assicura grammatiche identiche → tabelle di parsing identiche indipendentemente dall'ordine di inserimento dict — critico per riproducibilità in ricerca. |
| **4** | **Deep copy in iterazione grammatica** | 🐛 Bug fix | ✅ Sì | Interna usa `grammar.copy()` shallow — sub-list sono aliased e possono essere corrotte silenziosamente (FIX-C in esterna). |
| **5** | **Validazione empty-token in grammar generation** | 🐛 Bug fix | ✅ Sì | Tag che tokenizzano a `[]` sono droppati silenziosamente in interna, producendo `KeyError` oscuro a generation time. Esterna raise immediatamente con messaggio actionable. |

### Verdict

**Esterna è un miglioramento stretto di interna?** **SÌ**, per il codice che esiste in entrambe. Fixa bug di correttezza (FIRST computation, aliasing, empty tokens), migliora performance (FIRST-of-stack scan), aggiunge riproducibilità (ordinamento deterministico), e fornisce test.

**T2G può usarla come drop-in replacement?** **NO** — i rename dei moduli e i cambi API (`generate_grammar_parameters` return type, `MaskLogitsProcessor`→`StatelessLogitsProcessor`, `generate_text` return format) richiedono **modifiche al codice** in `src/grammar/gloss_grammar.py` e `src/grammar/grammar_logits_processor.py`.

### Strategia raccomandata

1. **Adottare i 5 fix individuali** (#1–#5 sopra) dalla copia esterna in quella interna — sono isolati, low-risk, e fixano bug reali.
2. **Mantenere la API surface attuale** — il progetto T2G non ha bisogno di beam search, lookahead, o multi-sequence generation. L'API interna (`MaskLogitsProcessor` + `BaseStreamer`) è più semplice e sufficiente.
3. **Migrazione completa solo se** il progetto transiziona al path PDA come default (attualmente usa `GlossVocabularyMask`) E ha bisogno di beam search. In quel caso, investire nella migrazione API.

### Rischi di swap completo
- **Sforzo migrazione**: 2 import site, `create_grammarllm_pipeline()` rewrite, `GrammarPDALogitsProcessor` rewrite
- **Dipendenza `config.py`**: adattare le chiamate `setup_logging()`
- **Nuova complessità**: `StatelessLogitsProcessor` è 784 righe con cache LRU, beam retirement, lookahead — più surface area per bug
- **Impatto path corrente**: il default `grpo_optimal.yaml` usa `use_grammarllm_pda: false`, quindi il path PDA è raramente esercitato. Uno swap completo sarebbe alto sforzo per basso payoff immediato.

### Conclusioni
Per il progetto T2G così com'è oggi (path PDA disabilitato di default), **portare i 5 bug-fix nella copia interna** è la strategia ottimale: basso rischio, alto valore (correttezza + riproducibilità), zero migrazione API. Riservare la migrazione completa a quando il path PDA diventerà il default o si vorrà sperimentare con beam search/lookahead.

---

## 7. Riferimenti
- `docs/ERRORI_E_MIGLIORIE.md` — analisi bug originale della copia interna
- `grammarllm/bug_analysis.md` (esterna) — registro di 20+ bug con stato risoluzione
- `src/grammar/gloss_grammar.py` — integration point `create_grammarllm_pipeline()`
- `src/grammar/grammar_logits_processor.py` — wrapper `GrammarPDALogitsProcessor`
- `experiments/configs/t2g/grpo_optimal.yaml` — `grammar.use_grammarllm_pda: false` (default)
