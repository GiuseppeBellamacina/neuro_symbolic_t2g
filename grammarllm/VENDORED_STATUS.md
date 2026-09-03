# Vendored grammarllm — Status vs upstream (`_Ricerca/grammarllm`)

Data verifica: 2026-09-03 (backport PR #3 upstream)

## Esito

Le due copie **non sono identiche**. La copia vendored (questa cartella) è
**funzionalmente più avanti** dell'upstream locale, che non può essere
modificato da questo progetto (vincolo di scope). L'upstream ha ricevuto
due PR dopo `ac1af8d`: PR #1 (docs/bench) e PR #3 (`2a4f55c`, SMILES
benchmark + **fix PDA critici — backportati**, vedi sotto).

## Backport da PR #3 upstream (2026-09-03)

- **fix(pda) `9ba32d2`** (BACKPORTATO in `modules/logits_processor.py`):
  `_advance_token()` testava BOS/PAD/UNK **prima** del ramo EOS. Il nostro
  `model_loader` imposta `pad_token = eos_token` (Qwen2.5-Instruct), quindi
  `pad_token_id == eos_token_id` e l'EOS veniva scartato come "token
  speciale" SENZA consumare la produzione `S* → eos_token`: il
  re-simulation finiva con stack non vuota su OGNI generazione valida,
  falsando ogni metrica su `pda_stack`. Fix: ramo EOS spostato prima del
  check PAD.
- **fix(streamer) `9ba32d2`** (BACKPORTATO in `modules/streamer.py`): il
  warning di consistenza in `BaseStreamer.end()` ispezionava i PDA
  "template" che `put()` non avanza mai (STATE UPDATE DISABLED) → warning
  su OGNI generazione, anche valide. Rimosso; il warning ora vive in
  `generate_text()` dopo `pda_stack` dove lo stato reale è ricostruibile.
- **fix(warning move) `9ba32d2`** (BACKPORTATO in
  `generate_with_constraints.py`): il consistency warning ora scatta SOLO
  se `result_item["pda_stack"]` non è vuota (troncamento max_new_tokens),
  non su ogni riga.
- **fix(grammar) `e40fd28` + `e32a9b8`** (NON backportati): LL(1)-ification
  della table construction — script usato SOLO per la grammatica SMILES del
  benchmark upstream; il nostro gloss pipeline non usa parsing tables
  generate da quegli script. Nessun impatto.
- **smiles_qed** (`2d72951`, `e12c4cf`): benchmark SMILES su GDB-17 — non
  rilevante per ASL gloss. Nessun backport.

## Divergenze note (verificate con diff riga per riga)

La copia vendored contiene fix **non presenti** nell'upstream locale
(ultima commit upstream: `ac1af8d`, 2026-07-17 — solo docs/benchmark):

- `modules/logits_processor.py`: bound-check `eos_token_id >= vocab_size`
  (necessario per Qwen2.5, dove `eos_token_id == vocab_size` causava
  `IndexError` nel path PDA).
- `generate_with_constraints.py`: BUG-13 (`start_symbol` configurabile),
  BUG-4/19 (validazione difensiva mapping `eos_token` → token ID).
- Formattazione black + fix minori su `automaton.py`, `streamer.py`,
  `scripts/*`, `utils/*` (strategia documentata in
  `docs/GRAMMARLLM_CONFRONTO.md`: backport dei bug-fix nella copia interna).

L'upstream locale contiene solo materiale non funzionale per questo
progetto (docs, benchmark tests in `benchmark_tests/`, notebook, CI).

## Decisione

- La copia vendored è la **versione canonical** usata da neuro_symbolic_t2g.
- Non sovrascrivere questa cartella con l'upstream senza prima ri-applicare
  i fix sopra (si perderebbero il bound-check EOS e i fix BUG-*).
- `temp/` e `__pycache__/` qui dentro sono residui ignorabili (in .gitignore).

## Impatto sul progetto

Nessun config principale usa il path PDA (`use_grammarllm_pda: false`
ovunque tranne gli ablation `grpo_pda*`), quindi queste divergenze non
influenzano i run "optimal"/"sft".
