# Vendored grammarllm — Status vs upstream (`_Ricerca/grammarllm`)

Data verifica: 2026-08-25

## Esito

Le due copie **non sono identiche**. La copia vendored (questa cartella) è
**funzionalmente più avanti** dell'upstream locale, che non può essere
modificato da questo progetto (vincolo di scope).

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
