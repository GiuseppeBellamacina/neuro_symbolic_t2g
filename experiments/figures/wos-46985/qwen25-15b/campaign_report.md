# Campaign Report — wos-46985 / qwen25-15b

Generated: 2026-11-09T18:02:36+00:00

> Complementare ad `ablation-summary`: qui i run sono appaiati
> per fattore sperimentale. **Legge i JSON di eval: non
> ricalcola nessuna metrica.**

## ⚠️ Mandatory caveats — read before any conclusion

1. **Interpretability threshold 0.02** (metrics on the [0,1] scale): training is bit-deterministic at a fixed seed — two executions of the same cell produce byte-identical generations — so there is no run-to-run noise. But no cell was replicated across seeds, so **deltas below 0.02 are reproducible yet not generalizable**: they may invert under another seed. A delta of 0.003 is not a result.
2. **Prompt counts** (`num_samples_evaluated`) are shown for every run; a pair with differing counts is explicitly flagged (metrics on different samples are not comparable). Historically this number moved from 2000 to 3000 prompts.
3. **`metrics_version`** is shown for every run and flagged when it differs across a pair.
4. **Run dates** are shown for every run: older runs come from different configurations (e.g. `max_steps` moved 2000→5000).
5. **Saturated overlap metrics — do not rank on these**: `rouge_l_mean`, `rouge_l_median`, `valid_rouge_l_mean`, `gloss_f1_micro`, `gloss_f1_sentence_mean`, `chrf_corpus`, `chrf_sentence_mean` are all within noise of a context-free rule baseline on ASLG-PC12 (ROUGE-L ~0.97, EM 0.59, see `src/analysis/rule_baseline.py`). Use the reward-independent primary metrics instead — `exact_match`, `non_copy_token_accuracy` — for any claim that a cell translates better, not merely that it scores higher on the same metric family the reward optimizes.

## How factors are deduced

| Factor | Rule | If not deducible |
|---|---|---|
| `method` | token del nome cella nel percorso (sft-grpo / sft-only / grpo-only / t2g-*); eval_baseline.json => baseline (nessun checkpoint) | unknown (never guessed) |
| `variant` | remaining cell-name tokens after model tag, method and known markers (e.g. structure, viterbi, all-rewards); no-grammar is a grammar marker, not a variant | empty string (plain cell) |
| `prompting` | 1) 'prompting.mode' stamp in the eval JSON; 2) filename suffix __zero-shot / __few-shot (dual/override pass); 3) t2g- zero-shot naming convention (baseline cells only) | unknown — the config-implied mode is NEVER assumed, it can differ from the effective one under dual prompting |
| `grammar` | 'no-grammar' in cell name => off; 'grammar' in t2g-* cell name => on (that family names the toggle explicitly) | unknown (never guessed) |
| `reward_stack` | NOT deducible from path / JSON stamp / payload fields (it lives in the resolved config, which eval JSONs do not embed) | always unknown, declared in every pairing caveat |
| `rl_objective` | NOT deducible from path / JSON stamp / payload fields | always unknown, declared in every pairing caveat |

## Selected runs (4 typologies, latest per typology)

| Run | method | variant | prompting | grammar | Date | Prompts | metrics_version |
|---|---|---|---|---|---|---|---|
| `wos-46985/qwen25-15b/baseline/zero-shot/run_20261107_224538/eval_zero_shot.json` | unknown | wos-46985-qwen25-15b-baseline-zero-shot | zero-shot | unknown | 2026-11-07 | 2000 | 2 |
| `wos-46985/qwen25-15b/sft-grpo/zero-shot/run_20261108_184716/eval_final.json` | sft-grpo | zero-shot | zero-shot | unknown | 2026-11-08 | 2000 | 2 |
| `wos-46985/qwen25-15b/sft/zero-shot/run_20261108_162337/eval_baseline.json` | baseline | wos-46985-qwen25-15b-sft-zero-shot | zero-shot | unknown | 2026-11-08 | 2000 | 2 |
| `wos-46985/qwen25-15b/sft/zero-shot/run_20261108_162337/eval_final.json` | unknown | wos-46985-qwen25-15b-sft-zero-shot | zero-shot | unknown | 2026-11-08 | 2000 | 2 |

## Ablation matrix overview — ROUGE-L

One row per cell (its path under `results/`), one column per prompting mode that has at least one value. `—` = the cell was not evaluated in that mode. `*` = multiple runs collapsed, latest shown. Each cell's own `eval_baseline` (the base model in that cell's context) is not a row here: it repeats the `baseline/` cells and stays in the paired comparisons below.

| cell | zero-shot |
|---|---|
| wos-46985/qwen25-15b/baseline/zero-shot | 0.2203 |
| wos-46985/qwen25-15b/sft/zero-shot | 0.7911 |
| wos-46985/qwen25-15b/sft-grpo/zero-shot | 0.8124 |

## wos-46985 — all metrics

Same rows as the matrix above, restricted to this dataset, one column per metric. On CoNLL-2003 and WOS-46985 read Exact Match and Gloss F1 (micro = entity-level F1 on CoNLL): ROUGE-L and BLEU split `PER:Mary` into `per` + `mary` and give partial credit to a wrong label of the same type.

| cell | prompting | ROUGE-L | Valid ROUGE-L | Exact Match | Non-copy Tok Acc | Pass@1 | Validity | BLEU (corpus) | BLEU (sent) | chrF2 (corpus) | chrF2 (sent) | Gloss F1 (mic) | Gloss F1 (sent) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| wos-46985/qwen25-15b/baseline/zero-shot | zero-shot | 0.2203 | 0.2203 | 0.0998 | 0.2608 | 0.3845 | 1.0000 | 0.1229 | 0.1523 | 25.0838 | 24.6317 | 0.2604 | 0.2604 |
| wos-46985/qwen25-15b/sft/zero-shot | zero-shot | 0.7911 | 0.7911 | 0.7339 | 0.8081 | 0.8655 | 1.0000 | 0.7808 | 0.7527 | 79.2804 | 78.8320 | 0.8075 | 0.8075 |
| wos-46985/qwen25-15b/sft-grpo/zero-shot | zero-shot | 0.8124 | 0.8124 | 0.7607 | 0.8260 | 0.8855 | 1.0000 | 0.7967 | 0.7781 | 81.3892 | 80.9882 | 0.8260 | 0.8260 |

## Paired comparisons — one table per factor

### Factor: `method`

**method effect (cross-run)** — `wos-46985/qwen25-15b/sft/zero-shot/run_20261108_162337/eval_baseline.json` (2026-11-08, zero-shot, unknown) → `wos-46985/qwen25-15b/sft-grpo/zero-shot/run_20261108_184716/eval_final.json` (2026-11-08, zero-shot, unknown)

| Metric | A | B | Δ (B−A) | Interpretation |
|---|---|---|---|---|
| ROUGE-L | 0.2203 | 0.8124 | +0.5921 | above noise threshold |
| Valid ROUGE-L | 0.2203 | 0.8124 | +0.5921 | above noise threshold |
| Exact Match | 0.0998 | 0.7607 | +0.6609 | above noise threshold |
| Non-copy Tok Acc | 0.2608 | 0.8260 | +0.5651 | above noise threshold |
| Pass@1 | 0.3845 | 0.8855 | +0.5010 | above noise threshold |
| Validity | 1.0000 | 1.0000 | +0.0000 | NOT GENERALIZABLE (|delta| < 0.02): reproducible (training is bit-deterministic at fixed seed) but single-seed, so it may invert under another seed — do NOT interpret |
| BLEU (corpus) | 0.1229 | 0.7967 | +0.6738 | above noise threshold |
| BLEU (sent) | 0.1523 | 0.7781 | +0.6258 | above noise threshold |
| chrF2 (corpus) | 25.0838 | 81.3892 | +56.3054 | scale 0-100: 0.02 noise threshold not applicable |
| chrF2 (sent) | 24.6317 | 80.9882 | +56.3565 | scale 0-100: 0.02 noise threshold not applicable |
| Gloss F1 (mic) | 0.2604 | 0.8260 | +0.5656 | above noise threshold |
| Gloss F1 (sent) | 0.2604 | 0.8260 | +0.5656 | above noise threshold |
- ⚠️ factors not deducible on at least one side, equality NOT verified: reward_stack, rl_objective

### Factor: `prompting`

**No paired comparison available.** 1 group(s) with 2+ runs but no pair with a known, differing value of this factor

### Factor: `grammar`

**No paired comparison available.** 1 group(s) with 2+ runs but no pair with a known, differing value of this factor


## Missing paired comparisons (declared, not fabricated)

- **`prompting`**: 1 group(s) with 2+ runs but no pair with a known, differing value of this factor.
  Available typologies: `zero-shot`
- **`grammar`**: 1 group(s) with 2+ runs but no pair with a known, differing value of this factor.
  Available typologies: `grammar=unknown`

## Figures

- `campaign_pairwise_deltas.png` — one panel per factor: paired deltas as diverging bars with the ±0.02 noise band shaded. Diverging bars make the SIGN of each effect immediately visible, and the shaded band makes it impossible to mistake sub-noise deltas for results.
- `campaign_matrix.png` — the ablation matrix as a heatmap (rows: method/variant, columns: prompting, annotated with ROUGE-L); empty typologies stay visibly grey so gaps in the grid are impossible to miss.

