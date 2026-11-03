# Campaign Report — phoenix-2014t / qwen25-15b

Generated: 2026-11-03T15:28:39+00:00

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

## Selected runs (14 typologies, latest per typology)

| Run | method | variant | prompting | grammar | Date | Prompts | metrics_version |
|---|---|---|---|---|---|---|---|
| `phoenix-2014t/qwen25-15b/baseline/few-shot/run_20261021_124104/eval_zero_shot.json` | unknown | phoenix-2014t-qwen25-15b-baseline-few-shot | few-shot | unknown | 2026-10-21 | 642 | 2 |
| `phoenix-2014t/qwen25-15b/baseline/zero-shot/run_20261021_121559/eval_zero_shot.json` | unknown | phoenix-2014t-qwen25-15b-baseline-zero-shot | zero-shot | unknown | 2026-10-21 | 642 | 2 |
| `phoenix-2014t/qwen25-15b/grpo/few-shot/run_20261021_175106/eval_baseline.json` | baseline | phoenix-2014t-qwen25-15b-grpo-few-shot | few-shot | unknown | 2026-10-21 | 642 | 2 |
| `phoenix-2014t/qwen25-15b/grpo/few-shot/run_20261021_175106/eval_baseline__zero-shot.json` | baseline | phoenix-2014t-qwen25-15b-grpo-few-shot | zero-shot | unknown | 2026-10-21 | 642 | 2 |
| `phoenix-2014t/qwen25-15b/grpo/few-shot/run_20261021_175106/eval_final.json` | unknown | phoenix-2014t-qwen25-15b-grpo-few-shot | few-shot | unknown | 2026-10-21 | 642 | 2 |
| `phoenix-2014t/qwen25-15b/grpo/few-shot/run_20261021_175106/eval_final__zero-shot.json` | unknown | phoenix-2014t-qwen25-15b-grpo-few-shot | zero-shot | unknown | 2026-10-21 | 642 | 2 |
| `phoenix-2014t/qwen25-15b/sft-grpo/zero-shot/run_20261021_140143/eval_baseline.json` | baseline | zero-shot | zero-shot | unknown | 2026-10-21 | 642 | 2 |
| `phoenix-2014t/qwen25-15b/sft-grpo/zero-shot/run_20261021_140143/eval_baseline__few-shot.json` | baseline | zero-shot | few-shot | unknown | 2026-10-21 | 642 | 2 |
| `phoenix-2014t/qwen25-15b/sft-grpo/zero-shot/run_20261021_140143/eval_final.json` | sft-grpo | zero-shot | zero-shot | unknown | 2026-10-21 | 642 | 2 |
| `phoenix-2014t/qwen25-15b/sft-grpo/zero-shot/run_20261021_140143/eval_final__few-shot.json` | sft-grpo | zero-shot | few-shot | unknown | 2026-10-21 | 642 | 2 |
| `phoenix-2014t/qwen25-15b/sft/zero-shot/run_20261021_125621/eval_baseline.json` | baseline | phoenix-2014t-qwen25-15b-sft-zero-shot | zero-shot | unknown | 2026-10-21 | 642 | 2 |
| `phoenix-2014t/qwen25-15b/sft/zero-shot/run_20261021_125621/eval_baseline__few-shot.json` | baseline | phoenix-2014t-qwen25-15b-sft-zero-shot | few-shot | unknown | 2026-10-21 | 642 | 2 |
| `phoenix-2014t/qwen25-15b/sft/zero-shot/run_20261021_125621/eval_final.json` | unknown | phoenix-2014t-qwen25-15b-sft-zero-shot | zero-shot | unknown | 2026-10-21 | 642 | 2 |
| `phoenix-2014t/qwen25-15b/sft/zero-shot/run_20261021_125621/eval_final__few-shot.json` | unknown | phoenix-2014t-qwen25-15b-sft-zero-shot | few-shot | unknown | 2026-10-21 | 642 | 2 |

## Ablation matrix overview — ROUGE-L

One row per cell (its path under `results/`), one column per prompting mode that has at least one value. `—` = the cell was not evaluated in that mode. `*` = multiple runs collapsed, latest shown. Each cell's own `eval_baseline` (the base model in that cell's context) is not a row here: it repeats the `baseline/` cells and stays in the paired comparisons below.

| cell | zero-shot | few-shot |
|---|---|---|
| phoenix-2014t/qwen25-15b/baseline/few-shot | — | 0.3519 |
| phoenix-2014t/qwen25-15b/baseline/zero-shot | 0.1199 | — |
| phoenix-2014t/qwen25-15b/sft/zero-shot | 0.5015 | 0.4638 |
| phoenix-2014t/qwen25-15b/sft-grpo/zero-shot | 0.5421 | 0.4928 |
| phoenix-2014t/qwen25-15b/grpo/few-shot | 0.1857 | 0.4337 |

## phoenix-2014t — all metrics

Same rows as the matrix above, restricted to this dataset, one column per metric. On CoNLL-2003 and WOS-46985 read Exact Match and Gloss F1 (micro = entity-level F1 on CoNLL): ROUGE-L and BLEU split `PER:Mary` into `per` + `mary` and give partial credit to a wrong label of the same type.

| cell | prompting | ROUGE-L | Valid ROUGE-L | Exact Match | Non-copy Tok Acc | Pass@1 | Validity | BLEU (corpus) | BLEU (sent) | chrF2 (corpus) | chrF2 (sent) | Gloss F1 (mic) | Gloss F1 (sent) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| phoenix-2014t/qwen25-15b/baseline/few-shot | few-shot | 0.3519 | 0.3339 | 0.0131 | 0.3461 | 0.5545 | 0.9489 | 0.0463 | 0.0939 | 34.1582 | 36.8815 | 0.3037 | 0.3860 |
| phoenix-2014t/qwen25-15b/baseline/zero-shot | zero-shot | 0.1199 | 0.0956 | 0.0000 | 0.1557 | 0.1106 | 0.7969 | 0.0024 | 0.0159 | 17.5347 | 19.5092 | 0.0816 | 0.1266 |
| phoenix-2014t/qwen25-15b/sft/zero-shot | zero-shot | 0.5015 | 0.5015 | 0.0171 | 0.4323 | 0.8380 | 1.0000 | 0.1377 | 0.1408 | 45.3214 | 45.4482 | 0.5433 | 0.5425 |
| phoenix-2014t/qwen25-15b/sft/zero-shot | few-shot | 0.4638 | 0.4638 | 0.0165 | 0.3987 | 0.7804 | 1.0000 | 0.1287 | 0.1336 | 42.0816 | 42.1402 | 0.4995 | 0.4999 |
| phoenix-2014t/qwen25-15b/sft-grpo/zero-shot | zero-shot | 0.5421 | 0.5421 | 0.0318 | 0.4826 | 0.8723 | 1.0000 | 0.1866 | 0.1718 | 49.2819 | 48.9480 | 0.5883 | 0.5837 |
| phoenix-2014t/qwen25-15b/sft-grpo/zero-shot | few-shot | 0.4928 | 0.4928 | 0.0162 | 0.4361 | 0.8178 | 1.0000 | 0.1482 | 0.1428 | 44.9916 | 44.8196 | 0.5366 | 0.5338 |
| phoenix-2014t/qwen25-15b/grpo/few-shot | zero-shot | 0.1857 | 0.1771 | 0.0000 | 0.1749 | 0.2072 | 0.9536 | 0.0048 | 0.0281 | 23.6606 | 24.4664 | 0.1675 | 0.1997 |
| phoenix-2014t/qwen25-15b/grpo/few-shot | few-shot | 0.4337 | 0.4321 | 0.0221 | 0.3901 | 0.7212 | 0.9963 | 0.1059 | 0.1262 | 40.8182 | 41.4991 | 0.4582 | 0.4756 |

## Paired comparisons — one table per factor

### Factor: `method`

**training effect (same config, baseline vs checkpoint)** — `phoenix-2014t/qwen25-15b/sft-grpo/zero-shot/run_20261021_140143/eval_baseline__few-shot.json` (2026-10-21, few-shot, unknown) → `phoenix-2014t/qwen25-15b/sft-grpo/zero-shot/run_20261021_140143/eval_final__few-shot.json` (2026-10-21, few-shot, unknown)

| Metric | A | B | Δ (B−A) | Interpretation |
|---|---|---|---|---|
| ROUGE-L | 0.3519 | 0.4928 | +0.1410 | above noise threshold |
| Valid ROUGE-L | 0.3339 | 0.4928 | +0.1590 | above noise threshold |
| Exact Match | 0.0131 | 0.0162 | +0.0031 | NOT GENERALIZABLE (|delta| < 0.02): reproducible (training is bit-deterministic at fixed seed) but single-seed, so it may invert under another seed — do NOT interpret |
| Non-copy Tok Acc | 0.3461 | 0.4361 | +0.0900 | above noise threshold |
| Pass@1 | 0.5545 | 0.8178 | +0.2632 | above noise threshold |
| Validity | 0.9489 | 1.0000 | +0.0511 | above noise threshold |
| BLEU (corpus) | 0.0463 | 0.1482 | +0.1020 | above noise threshold |
| BLEU (sent) | 0.0939 | 0.1428 | +0.0490 | above noise threshold |
| chrF2 (corpus) | 34.1582 | 44.9916 | +10.8335 | scale 0-100: 0.02 noise threshold not applicable |
| chrF2 (sent) | 36.8815 | 44.8196 | +7.9381 | scale 0-100: 0.02 noise threshold not applicable |
| Gloss F1 (mic) | 0.3037 | 0.5366 | +0.2328 | above noise threshold |
| Gloss F1 (sent) | 0.3860 | 0.5338 | +0.1478 | above noise threshold |
- ⚠️ factors not deducible on at least one side, equality NOT verified: reward_stack, rl_objective

**training effect (same config, baseline vs checkpoint)** — `phoenix-2014t/qwen25-15b/sft-grpo/zero-shot/run_20261021_140143/eval_baseline.json` (2026-10-21, zero-shot, unknown) → `phoenix-2014t/qwen25-15b/sft-grpo/zero-shot/run_20261021_140143/eval_final.json` (2026-10-21, zero-shot, unknown)

| Metric | A | B | Δ (B−A) | Interpretation |
|---|---|---|---|---|
| ROUGE-L | 0.1199 | 0.5421 | +0.4222 | above noise threshold |
| Valid ROUGE-L | 0.0956 | 0.5421 | +0.4466 | above noise threshold |
| Exact Match | 0.0000 | 0.0318 | +0.0318 | above noise threshold |
| Non-copy Tok Acc | 0.1557 | 0.4826 | +0.3269 | above noise threshold |
| Pass@1 | 0.1106 | 0.8723 | +0.7617 | above noise threshold |
| Validity | 0.7969 | 1.0000 | +0.2031 | above noise threshold |
| BLEU (corpus) | 0.0024 | 0.1866 | +0.1842 | above noise threshold |
| BLEU (sent) | 0.0159 | 0.1718 | +0.1558 | above noise threshold |
| chrF2 (corpus) | 17.5347 | 49.2819 | +31.7471 | scale 0-100: 0.02 noise threshold not applicable |
| chrF2 (sent) | 19.5092 | 48.9480 | +29.4388 | scale 0-100: 0.02 noise threshold not applicable |
| Gloss F1 (mic) | 0.0816 | 0.5883 | +0.5067 | above noise threshold |
| Gloss F1 (sent) | 0.1266 | 0.5837 | +0.4571 | above noise threshold |
- ⚠️ factors not deducible on at least one side, equality NOT verified: reward_stack, rl_objective

### Factor: `prompting`

🔴 **prompt-dependence (dual eval, same checkpoint)** — `phoenix-2014t/qwen25-15b/grpo/few-shot/run_20261021_175106/eval_baseline.json` (2026-10-21, few-shot, unknown) → `phoenix-2014t/qwen25-15b/grpo/few-shot/run_20261021_175106/eval_baseline__zero-shot.json` (2026-10-21, zero-shot, unknown)

| Metric | A | B | Δ (B−A) | Interpretation |
|---|---|---|---|---|
| ROUGE-L | 0.3519 | 0.1199 | -0.2319 | above noise threshold |
| Valid ROUGE-L | 0.3339 | 0.0956 | -0.2383 | above noise threshold |
| Exact Match | 0.0131 | 0.0000 | -0.0131 | NOT GENERALIZABLE (|delta| < 0.02): reproducible (training is bit-deterministic at fixed seed) but single-seed, so it may invert under another seed — do NOT interpret |
| Non-copy Tok Acc | 0.3461 | 0.1557 | -0.1904 | above noise threshold |
| Pass@1 | 0.5545 | 0.1106 | -0.4439 | above noise threshold |
| Validity | 0.9489 | 0.7969 | -0.1520 | above noise threshold |
| BLEU (corpus) | 0.0463 | 0.0024 | -0.0438 | above noise threshold |
| BLEU (sent) | 0.0939 | 0.0159 | -0.0779 | above noise threshold |
| chrF2 (corpus) | 34.1582 | 17.5347 | -16.6234 | scale 0-100: 0.02 noise threshold not applicable |
| chrF2 (sent) | 36.8815 | 19.5092 | -17.3723 | scale 0-100: 0.02 noise threshold not applicable |
| Gloss F1 (mic) | 0.3037 | 0.0816 | -0.2222 | above noise threshold |
| Gloss F1 (sent) | 0.3860 | 0.1266 | -0.2593 | above noise threshold |
- ⚠️ HIGHLIGHT: same model evaluated in both prompting modes — how much does it depend on the prompt as a crutch?
- ⚠️ factors not deducible on at least one side, equality NOT verified: reward_stack, rl_objective

🔴 **prompt-dependence (dual eval, same checkpoint)** — `phoenix-2014t/qwen25-15b/sft-grpo/zero-shot/run_20261021_140143/eval_baseline__few-shot.json` (2026-10-21, few-shot, unknown) → `phoenix-2014t/qwen25-15b/sft-grpo/zero-shot/run_20261021_140143/eval_baseline.json` (2026-10-21, zero-shot, unknown)

| Metric | A | B | Δ (B−A) | Interpretation |
|---|---|---|---|---|
| ROUGE-L | 0.3519 | 0.1199 | -0.2319 | above noise threshold |
| Valid ROUGE-L | 0.3339 | 0.0956 | -0.2383 | above noise threshold |
| Exact Match | 0.0131 | 0.0000 | -0.0131 | NOT GENERALIZABLE (|delta| < 0.02): reproducible (training is bit-deterministic at fixed seed) but single-seed, so it may invert under another seed — do NOT interpret |
| Non-copy Tok Acc | 0.3461 | 0.1557 | -0.1904 | above noise threshold |
| Pass@1 | 0.5545 | 0.1106 | -0.4439 | above noise threshold |
| Validity | 0.9489 | 0.7969 | -0.1520 | above noise threshold |
| BLEU (corpus) | 0.0463 | 0.0024 | -0.0438 | above noise threshold |
| BLEU (sent) | 0.0939 | 0.0159 | -0.0779 | above noise threshold |
| chrF2 (corpus) | 34.1582 | 17.5347 | -16.6234 | scale 0-100: 0.02 noise threshold not applicable |
| chrF2 (sent) | 36.8815 | 19.5092 | -17.3723 | scale 0-100: 0.02 noise threshold not applicable |
| Gloss F1 (mic) | 0.3037 | 0.0816 | -0.2222 | above noise threshold |
| Gloss F1 (sent) | 0.3860 | 0.1266 | -0.2593 | above noise threshold |
- ⚠️ HIGHLIGHT: same model evaluated in both prompting modes — how much does it depend on the prompt as a crutch?
- ⚠️ factors not deducible on at least one side, equality NOT verified: reward_stack, rl_objective

🔴 **prompt-dependence (dual eval, same checkpoint)** — `phoenix-2014t/qwen25-15b/sft/zero-shot/run_20261021_125621/eval_baseline__few-shot.json` (2026-10-21, few-shot, unknown) → `phoenix-2014t/qwen25-15b/sft/zero-shot/run_20261021_125621/eval_baseline.json` (2026-10-21, zero-shot, unknown)

| Metric | A | B | Δ (B−A) | Interpretation |
|---|---|---|---|---|
| ROUGE-L | 0.3519 | 0.1199 | -0.2319 | above noise threshold |
| Valid ROUGE-L | 0.3339 | 0.0956 | -0.2383 | above noise threshold |
| Exact Match | 0.0131 | 0.0000 | -0.0131 | NOT GENERALIZABLE (|delta| < 0.02): reproducible (training is bit-deterministic at fixed seed) but single-seed, so it may invert under another seed — do NOT interpret |
| Non-copy Tok Acc | 0.3461 | 0.1557 | -0.1904 | above noise threshold |
| Pass@1 | 0.5545 | 0.1106 | -0.4439 | above noise threshold |
| Validity | 0.9489 | 0.7969 | -0.1520 | above noise threshold |
| BLEU (corpus) | 0.0463 | 0.0024 | -0.0438 | above noise threshold |
| BLEU (sent) | 0.0939 | 0.0159 | -0.0779 | above noise threshold |
| chrF2 (corpus) | 34.1582 | 17.5347 | -16.6234 | scale 0-100: 0.02 noise threshold not applicable |
| chrF2 (sent) | 36.8815 | 19.5092 | -17.3723 | scale 0-100: 0.02 noise threshold not applicable |
| Gloss F1 (mic) | 0.3037 | 0.0816 | -0.2222 | above noise threshold |
| Gloss F1 (sent) | 0.3860 | 0.1266 | -0.2593 | above noise threshold |
- ⚠️ HIGHLIGHT: same model evaluated in both prompting modes — how much does it depend on the prompt as a crutch?
- ⚠️ factors not deducible on at least one side, equality NOT verified: reward_stack, rl_objective

🔴 **prompt-dependence (dual eval, same checkpoint)** — `phoenix-2014t/qwen25-15b/sft-grpo/zero-shot/run_20261021_140143/eval_final__few-shot.json` (2026-10-21, few-shot, unknown) → `phoenix-2014t/qwen25-15b/sft-grpo/zero-shot/run_20261021_140143/eval_final.json` (2026-10-21, zero-shot, unknown)

| Metric | A | B | Δ (B−A) | Interpretation |
|---|---|---|---|---|
| ROUGE-L | 0.4928 | 0.5421 | +0.0493 | above noise threshold |
| Valid ROUGE-L | 0.4928 | 0.5421 | +0.0493 | above noise threshold |
| Exact Match | 0.0162 | 0.0318 | +0.0156 | NOT GENERALIZABLE (|delta| < 0.02): reproducible (training is bit-deterministic at fixed seed) but single-seed, so it may invert under another seed — do NOT interpret |
| Non-copy Tok Acc | 0.4361 | 0.4826 | +0.0466 | above noise threshold |
| Pass@1 | 0.8178 | 0.8723 | +0.0545 | above noise threshold |
| Validity | 1.0000 | 1.0000 | +0.0000 | NOT GENERALIZABLE (|delta| < 0.02): reproducible (training is bit-deterministic at fixed seed) but single-seed, so it may invert under another seed — do NOT interpret |
| BLEU (corpus) | 0.1482 | 0.1866 | +0.0384 | above noise threshold |
| BLEU (sent) | 0.1428 | 0.1718 | +0.0289 | above noise threshold |
| chrF2 (corpus) | 44.9916 | 49.2819 | +4.2903 | scale 0-100: 0.02 noise threshold not applicable |
| chrF2 (sent) | 44.8196 | 48.9480 | +4.1285 | scale 0-100: 0.02 noise threshold not applicable |
| Gloss F1 (mic) | 0.5366 | 0.5883 | +0.0517 | above noise threshold |
| Gloss F1 (sent) | 0.5338 | 0.5837 | +0.0499 | above noise threshold |
- ⚠️ HIGHLIGHT: same model evaluated in both prompting modes — how much does it depend on the prompt as a crutch?
- ⚠️ factors not deducible on at least one side, equality NOT verified: reward_stack, rl_objective

🔴 **prompt-dependence (dual eval, same checkpoint)** — `phoenix-2014t/qwen25-15b/grpo/few-shot/run_20261021_175106/eval_final.json` (2026-10-21, few-shot, unknown) → `phoenix-2014t/qwen25-15b/grpo/few-shot/run_20261021_175106/eval_final__zero-shot.json` (2026-10-21, zero-shot, unknown)

| Metric | A | B | Δ (B−A) | Interpretation |
|---|---|---|---|---|
| ROUGE-L | 0.4337 | 0.1857 | -0.2481 | above noise threshold |
| Valid ROUGE-L | 0.4321 | 0.1771 | -0.2551 | above noise threshold |
| Exact Match | 0.0221 | 0.0000 | -0.0221 | above noise threshold |
| Non-copy Tok Acc | 0.3901 | 0.1749 | -0.2152 | above noise threshold |
| Pass@1 | 0.7212 | 0.2072 | -0.5140 | above noise threshold |
| Validity | 0.9963 | 0.9536 | -0.0427 | above noise threshold |
| BLEU (corpus) | 0.1059 | 0.0048 | -0.1012 | above noise threshold |
| BLEU (sent) | 0.1262 | 0.0281 | -0.0981 | above noise threshold |
| chrF2 (corpus) | 40.8182 | 23.6606 | -17.1576 | scale 0-100: 0.02 noise threshold not applicable |
| chrF2 (sent) | 41.4991 | 24.4664 | -17.0327 | scale 0-100: 0.02 noise threshold not applicable |
| Gloss F1 (mic) | 0.4582 | 0.1675 | -0.2908 | above noise threshold |
| Gloss F1 (sent) | 0.4756 | 0.1997 | -0.2758 | above noise threshold |
- ⚠️ HIGHLIGHT: same model evaluated in both prompting modes — how much does it depend on the prompt as a crutch?
- ⚠️ factors not deducible on at least one side, equality NOT verified: reward_stack, rl_objective

🔴 **prompt-dependence (dual eval, same checkpoint)** — `phoenix-2014t/qwen25-15b/sft/zero-shot/run_20261021_125621/eval_final__few-shot.json` (2026-10-21, few-shot, unknown) → `phoenix-2014t/qwen25-15b/sft/zero-shot/run_20261021_125621/eval_final.json` (2026-10-21, zero-shot, unknown)

| Metric | A | B | Δ (B−A) | Interpretation |
|---|---|---|---|---|
| ROUGE-L | 0.4638 | 0.5015 | +0.0377 | above noise threshold |
| Valid ROUGE-L | 0.4638 | 0.5015 | +0.0377 | above noise threshold |
| Exact Match | 0.0165 | 0.0171 | +0.0006 | NOT GENERALIZABLE (|delta| < 0.02): reproducible (training is bit-deterministic at fixed seed) but single-seed, so it may invert under another seed — do NOT interpret |
| Non-copy Tok Acc | 0.3987 | 0.4323 | +0.0337 | above noise threshold |
| Pass@1 | 0.7804 | 0.8380 | +0.0576 | above noise threshold |
| Validity | 1.0000 | 1.0000 | +0.0000 | NOT GENERALIZABLE (|delta| < 0.02): reproducible (training is bit-deterministic at fixed seed) but single-seed, so it may invert under another seed — do NOT interpret |
| BLEU (corpus) | 0.1287 | 0.1377 | +0.0091 | NOT GENERALIZABLE (|delta| < 0.02): reproducible (training is bit-deterministic at fixed seed) but single-seed, so it may invert under another seed — do NOT interpret |
| BLEU (sent) | 0.1336 | 0.1408 | +0.0072 | NOT GENERALIZABLE (|delta| < 0.02): reproducible (training is bit-deterministic at fixed seed) but single-seed, so it may invert under another seed — do NOT interpret |
| chrF2 (corpus) | 42.0816 | 45.3214 | +3.2398 | scale 0-100: 0.02 noise threshold not applicable |
| chrF2 (sent) | 42.1402 | 45.4482 | +3.3080 | scale 0-100: 0.02 noise threshold not applicable |
| Gloss F1 (mic) | 0.4995 | 0.5433 | +0.0438 | above noise threshold |
| Gloss F1 (sent) | 0.4999 | 0.5425 | +0.0426 | above noise threshold |
- ⚠️ HIGHLIGHT: same model evaluated in both prompting modes — how much does it depend on the prompt as a crutch?
- ⚠️ factors not deducible on at least one side, equality NOT verified: reward_stack, rl_objective

**prompting (cross-run)** — `phoenix-2014t/qwen25-15b/baseline/few-shot/run_20261021_124104/eval_zero_shot.json` (2026-10-21, few-shot, unknown) → `phoenix-2014t/qwen25-15b/baseline/zero-shot/run_20261021_121559/eval_zero_shot.json` (2026-10-21, zero-shot, unknown)

| Metric | A | B | Δ (B−A) | Interpretation |
|---|---|---|---|---|
| ROUGE-L | 0.3519 | 0.1199 | -0.2319 | above noise threshold |
| Valid ROUGE-L | 0.3339 | 0.0956 | -0.2383 | above noise threshold |
| Exact Match | 0.0131 | 0.0000 | -0.0131 | NOT GENERALIZABLE (|delta| < 0.02): reproducible (training is bit-deterministic at fixed seed) but single-seed, so it may invert under another seed — do NOT interpret |
| Non-copy Tok Acc | 0.3461 | 0.1557 | -0.1904 | above noise threshold |
| Pass@1 | 0.5545 | 0.1106 | -0.4439 | above noise threshold |
| Validity | 0.9489 | 0.7969 | -0.1520 | above noise threshold |
| BLEU (corpus) | 0.0463 | 0.0024 | -0.0438 | above noise threshold |
| BLEU (sent) | 0.0939 | 0.0159 | -0.0779 | above noise threshold |
| chrF2 (corpus) | 34.1582 | 17.5347 | -16.6234 | scale 0-100: 0.02 noise threshold not applicable |
| chrF2 (sent) | 36.8815 | 19.5092 | -17.3723 | scale 0-100: 0.02 noise threshold not applicable |
| Gloss F1 (mic) | 0.3037 | 0.0816 | -0.2222 | above noise threshold |
| Gloss F1 (sent) | 0.3860 | 0.1266 | -0.2593 | above noise threshold |
- ⚠️ cross-run comparison: checkpoint identity NOT guaranteed
- ⚠️ factors not deducible on at least one side, equality NOT verified: reward_stack, rl_objective

### Factor: `grammar`

**No paired comparison available.** 4 group(s) with 2+ runs but no pair with a known, differing value of this factor


## Missing paired comparisons (declared, not fabricated)

- **`grammar`**: 4 group(s) with 2+ runs but no pair with a known, differing value of this factor.
  Available typologies: `grammar=unknown`

## Figures

- `campaign_pairwise_deltas.png` — one panel per factor: paired deltas as diverging bars with the ±0.02 noise band shaded. Diverging bars make the SIGN of each effect immediately visible, and the shaded band makes it impossible to mistake sub-noise deltas for results.
- `campaign_matrix.png` — the ablation matrix as a heatmap (rows: method/variant, columns: prompting, annotated with ROUGE-L); empty typologies stay visibly grey so gaps in the grid are impossible to miss.

