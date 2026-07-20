# Report Migrazione grammarllm v0.4.x → v0.5.0

> **Data**: 20 luglio 2026
> **Scope**: Sostituzione della copia vendored `grammarllm` (pre-release, v0.4.x)
> con la versione esterna aggiornata v0.5.0, adattamento del codice dipendente,
> aggiornamento dei test, verifica.
> **Vincoli rispettati**: la cartella esterna `D:\Codici\Workspace\_Ricerca\grammarllm`
> non è stata modificata. La vecchia copia interna è preservata come `grammarllm_old`.

---

## TL;DR

| Aspetto | Stato |
| --- | --- |
| **Sostituzione completata** | ✅ `grammarllm_old` (backup) + nuova `grammarllm` v0.5.0 |
| **Codice adattato** | ✅ 4 file: `gloss_grammar.py`, `grammar_logits_processor.py`, `grpo_t2g_train.py`, `eval_t2g.py` |
| **Test aggiornati** | ✅ `test_grammar.py` — 7/7 passano |
| **Suite completa** | ✅ 21/21 test passano (grammar + rewards + integration) |
| **Esterna modificata?** | ❌ No — read-only, come richiesto |
| **Bug fix aggiunto** | ✅ Bound check per token IDs fuori range (Qwen eos_token_id) |
| **Migliorie attese** | Bug fix (FIRST/FOLLOW, aliasing, empty-token), token-boundary lookahead, beam search, suite di test, riproducibilità |

---

## 1. Modifiche ai file

### 1.1 Sostituzione del pacchetto `grammarllm/`

| Operazione | Path |
| --- | --- |
| **Rinomina** (backup) | `neuro_symbolic_t2g/grammarllm/` → `neuro_symbolic_t2g/grammarllm_old/` |
| **Copia** (nuova versione) | `grammarllm/grammarllm/` (esterna v0.5.0) → `neuro_symbolic_t2g/grammarllm/` |

La vecchia copia è preservata come `grammarllm_old` per riferimento e rollback.

### 1.2 `src/grammar/gloss_grammar.py` — adattamento API

**Import (riga 24):**
```diff
-from grammarllm.modules.PushdownAutomaton import PushdownAutomaton
+from grammarllm.modules.automaton import PushdownAutomaton
```

**`create_grammarllm_pipeline()` (riga 95):**
- Return type: `tuple[Any, Any, PushdownAutomaton]` → `tuple[list, Any, PushdownAutomaton]`
- `generate_grammar_parameters` ora ritorna `(pdas: list, streamer)` invece di `(logit_processor, streamer)`
- Rimosso `logit_processor.pda` → ora `pdas[0]` è il PDA primario
- Rimosso `logit_processor.temperature = temperature` → la temperatura è gestita da HF `generate()`
- Docstring aggiornato con note di migrazione

**Return value:**
```diff
-logit_processor, streamer = generate_grammar_parameters(...)
-pda = logit_processor.pda
-logit_processor.temperature = temperature
-return logit_processor, streamer, pda
+pdas, streamer = generate_grammar_parameters(...)
+pda = pdas[0]
+return pdas, streamer, pda
```

### 1.3 `src/grammar/grammar_logits_processor.py` — adattamento wrapper

**Import (riga 28-31):**
```diff
-from grammarllm.modules.PushdownAutomaton import PushdownAutomaton
-from grammarllm.modules.SimpleLogitProcessor_ import (
-    MaskLogitsProcessor as GrammarLLMMaskProcessor,
-)
+from grammarllm.modules.automaton import PushdownAutomaton
+from grammarllm.modules.logits_processor import (
+    StatelessLogitsProcessor as GrammarLLMStatelessProcessor,
+)
```

**`GrammarPDALogitsProcessor.__init__`:**
```diff
-self._grammar_processor = GrammarLLMMaskProcessor(
-    tokenizer, pda, temperature=temperature
-)
+base_pdas = [pda] if not isinstance(pda, list) else pda
+self._grammar_processor = GrammarLLMStatelessProcessor(
+    tokenizer=tokenizer,
+    base_pdas=base_pdas,
+    sequences_per_prompt=1,
+    prompt_len=0,
+    temperature=temperature,
+    track_score_history=False,
+)
```

**`__call__`:**
- Aggiunto bound check per token IDs fuori range (Qwen: `eos_token_id=151643` ≥ `vocab_size=151643`)
- Aggiornato `prompt_len` sul processor wrappato al primo step
- Rimossi `points`/`preserved_mass` (non esposti dal nuovo `StatelessLogitsProcessor`)

### 1.4 `src/training/grpo_t2g_train.py` — call site

**Riga 636:**
```diff
-logit_processor, streamer, pda = create_grammarllm_pipeline(...)
+pdas, streamer, pda = create_grammarllm_pipeline(...)
```

### 1.5 `src/training/eval_t2g.py` — call site

**Riga 278:**
```diff
-_, _, pda = create_grammarllm_pipeline(...)
+pdas, streamer, pda = create_grammarllm_pipeline(...)
```

### 1.6 `tests/test_grammar.py` — test adattato

**`test_grammar_build`:**
```diff
-logit_processor, streamer, pda = create_grammarllm_pipeline(test_vocab, tokenizer)
-assert logit_processor is not None, "LogitsProcessor created"
+pdas, streamer, pda = create_grammarllm_pipeline(test_vocab, tokenizer)
+assert isinstance(pdas, list), f"pdas is a list, got {type(pdas)}"
+assert len(pdas) > 0, "pdas list non-empty"
+assert pdas[0] is pda, "pda is pdas[0] (primary PDA)"
```

### 1.7 `grammarllm/modules/logits_processor.py` — bug fix (copia interna)

**Riga 522-549:** Aggiunto bound check difensivo per token IDs fuori range. Il `StatelessLogitsProcessor` indicizzava `scores[i, eos_token_id]` senza verificare che `eos_token_id < vocab_size`. Per Qwen2.5 (`vocab_size=151643`, `eos_token_id=151643`), questo causava `IndexError`. Fix:
- `eos_id` capped a `vocab_size - 1` se out of range
- `valid_ids` filtrati con `[t for t in valid_tokens if 0 <= t < vocab_size]`
- `mask[valid_ids]` eseguito solo se `valid_ids` non è vuoto

> **Nota**: questa modifica è sulla copia **interna** (il file copiato da esterna, ora parte del progetto). L'esterna originale a `D:\Codici\Workspace\_Ricerca\grammarllm` è intatta.

---

## 2. Verifica

### 2.1 Import check
Tutti i 9 import risolti con successo:
- `grammarllm` (public API: `generate_grammar_parameters`, `get_parsing_table_and_map_tt`, `setup_logging`, `generate_text`)
- `grammarllm.modules.automaton.PushdownAutomaton`
- `grammarllm.modules.logits_processor.StatelessLogitsProcessor`
- `grammarllm.modules.streamer.BaseStreamer`
- `src.grammar.gloss_grammar` (`create_grammarllm_pipeline`, `GlossVocabularyMask`)
- `src.grammar.grammar_logits_processor` (`GrammarPDALogitsProcessor`, `GlossVocabularyLogitsProcessor`)
- `src.training.grpo_t2g_train`
- `src.training.eval_t2g`

### 2.2 Test suite
```
tests/test_grammar.py      7/7  passed (4.84s)
tests/test_rewards.py      9/9  passed
tests/test_integration.py  5/5  passed
─────────────────────────────────
Total: 21/21 passed (62.88s)
```

Nessun regresso. I warning sono pre-existing (NumPy 1.25 deprecation in `transition_matrix.py:732`, unrelated).

---

## 3. Migliorie attese con la nuova versione

### 3.1 Bug fix (correttezza)

| # | Bug | Impatto |
| --- | --- | --- |
| 1 | **FIRST computation con mutua ricorsione** — il vecchio DFS ricorsivo con sentinella produceva FIRST sets sbagliati per grammatiche con NT che si referenziano circolarmente. Ora usa fixed-point iteration. | La grammatica gloss T2G è semplice (singolo NT flat), ma future grammatiche con NT annidati sarebbero state silenziosamente errate. |
| 2 | **Scan FIRST-of-stack iterativo** — `recursive_get_tokens` usava visited-set ricorsivo (worst-case esponenziale, poteva prunare continuazioni valide per catene nullable-NT). Ora scan iterativo O(stack). | Performance + correttezza. Meno rischio di prunare token validi. |
| 3 | **Deep copy in iterazione grammatica** — il vecchio `grammar.copy()` era shallow (sub-list aliased → corruzione silenziosa). | Correttezza nella costruzione della grammatica. |
| 4 | **Validazione empty-token** — i tag che tokenizzano a `[]` erano droppati silenziosamente → `KeyError` oscuro a runtime. Ora raise `ValueError` con messaggio actionable. | Errori più facili da diagnosticare. |
| 5 | **`raise "string"` → `raise RuntimeError(...)`** | Crash con messaggio utile invece di `TypeError`. |
| 6 | **`current_terminals` non inizializzato** | Niente più `AttributeError` in `next_state()` se chiamato prima di `get_tokens()`. |
| 7 | **`eos()` checked due volte** | Reset consistente, no loop di discard silenzioso. |

### 3.2 Token-boundary lookahead (nuova capacità, alto beneficio T2G)

`modules/lookahead.py` (138 righe) costruisce un `VocabTrie` sul vocabolario del tokenizer e fa DFS sulle transizioni PDA per trovare TUTTI i token realizzabili — inclusi token che **fondono attraverso boundary grammaticali**.

**Beneficio T2G**: Qwen2.5 usa BPE subword tokenization. Senza lookahead, il PDA forzava emissioni single-BPE-token, frammentando gloss come `"IX-me"` in `["IX", "-", "me"]`. Con lookahead, il modello può emettere i suoi token BPE nativi (es. `["IX", "-me"]`), allineandosi alla tokenizzazione su cui è stato pre-addestrato.

> **Nota**: il lookahead è ON di default in `generate_grammar_parameters` (`token_lookahead=True`). Si attiva automaticamente quando si usa il path PDA.

### 3.3 Beam search support (nuova capacità)

`StatelessLogitsProcessor` (784 righe) usa cache LRU + re-simulation che permette a HF `num_beams > 1` di funzionare correttamente con vincoli grammaticali. Il vecchio `MaskLogitsProcessor` hardcodava `num_beams=1`.

**Beneficio T2G**: utile per evaluation con beam search (migliore qualità gloss a costo di inference speed). Non impatta il training GRPO (usa sampling).

### 3.4 Architettura stateless (migliore del vecchio stateful)

Il nuovo processor è **stateless**: deriva lo stato del PDA dalla history dei token (`input_ids`) ad ogni step, invece di aggiornare un PDA stateful in-place.

**Vantaggi**:
- **Beam-search safe**: HF può scartare/rimpiazzare beam tra step — un PDA stateful si corromperebbe
- **Cache LRU**: O(1) amortized invece di O(L) per step (L = lunghezza history)
- **Cache eviction**: capped a 2048 entry (evita OOM su generazioni lunghe)
- **Beam retirement**: gestisce gracefully i beam riempiti con token mascherati a `-inf` (artefatto strutturale di HF)

### 3.5 Suite di test (confidenza)

La nuova versione include 6 file pytest (16+ test) che validano: FIRST-of-stack scan, reset(), score history gating, beam search, lookahead, multi-tag, chat template, decode fidelity. La vecchia versione aveva **zero test**.

### 3.6 Altre migliorie

- **Determinismo**: `sorted()` su `prefix_groups`/`ungrouped_tags` → grammatiche identiche producono tabelle di parsing identiche (riproducibilità in ricerca)
- **Type hints** completi su tutti i moduli
- **Docstring** completi (IT+EN, 500+ righe di docs)
- **`bug_analysis.md`**: registro di 20+ bug con stato risoluzione
- **Multi-prompt batching**: `BaseStreamer` accetta lista di PDA, non singolo
- **`clone()` method** sul PDA: copia leggera che condivide strutture read-only

---

## 4. File modificati (riepilogo)

| File | Tipo modifica |
| --- | --- |
| `neuro_symbolic_t2g/grammarllm_old/` | **Rinomina** (backup della vecchia copia) |
| `neuro_symbolic_t2g/grammarllm/` | **Copia** (nuova v0.5.0 dall'esterna) |
| `src/grammar/gloss_grammar.py` | Import + `create_grammarllm_pipeline` adattati |
| `src/grammar/grammar_logits_processor.py` | Import + `GrammarPDALogitsProcessor` riscritto |
| `src/training/grpo_t2g_train.py` | Call site (1 riga: `logit_processor` → `pdas`) |
| `src/training/eval_t2g.py` | Call site (1 riga: `_` → `pdas, streamer`) |
| `tests/test_grammar.py` | `test_grammar_build` adattato al nuovo return type |
| `grammarllm/modules/logits_processor.py` | Bound check per token IDs fuori range (copia interna) |

**Esterna non modificata**: `D:\Codici\Workspace\_Ricerca\grammarllm\` è intatta.

---

## 5. Note di compatibilità

- **Path PDA di default**: `grpo_optimal.yaml` ha `grammar.use_grammarllm_pda: false` → usa `GlossVocabularyMask` (Trie dual-root). Il path PDA completo è attivabile con `use_grammarllm_pda: true` e ora beneficia di tutti i fix sopra.
- **API pubblica preservata**: `generate_grammar_parameters`, `get_parsing_table_and_map_tt`, `setup_logging`, `generate_text` mantengono la stessa signature pubblica (ma `generate_grammar_parameters` ritorna `list[PDA]` invece di `MaskLogitsProcessor`).
- **`generate_text()` return format**: ora ritorna `dict`/`list[dict]` con `text`, `probability`, `log_prob`, `pda_history`, `pda_stack` invece di `str`. Non usato direttamente dal progetto T2G (che usa `model.generate()` con logits_processor), ma rilevante se si volesse usare l'API high-level.
- **`config.py` rimosso**: la vecchia `setup_logging()` usava `config.LOG_FILE`; la nuova prende `log_dir` parameter. `gloss_grammar.py` chiama `setup_logging()` senza argomenti (default `'grammarllm/temp'`), invariato.

---

## 6. Come fare rollback (se necessario)

```bash
# Rimuovi la nuova copia
rm -rf neuro_symbolic_t2g/grammarllm
# Ripristina la vecchia
mv neuro_symbolic_t2g/grammarllm_old neuro_symbolic_t2g/grammarllm
# Revert delle modifiche al codice (git checkout sui 5 file adattati)
git checkout -- src/grammar/gloss_grammar.py src/grammar/grammar_logits_processor.py \
                 src/training/grpo_t2g_train.py src/training/eval_t2g.py tests/test_grammar.py
```

`grammarllm_old` è preservato come safety net.
