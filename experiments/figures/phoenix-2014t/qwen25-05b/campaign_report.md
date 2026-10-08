# Campaign Report — phoenix-2014t / qwen25-05b

Generated: 2026-10-08T13:52:59+00:00

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

## Selected runs (20 typologies, latest per typology)

| Run | method | variant | prompting | grammar | Date | Prompts | metrics_version |
|---|---|---|---|---|---|---|---|
| `phoenix-2014t/qwen25-05b/grpo/few-shot/run_20261004_043119/eval_baseline.json` | baseline | phoenix-2014t-qwen25-05b-grpo-few-shot | few-shot | unknown | 2026-10-04 | 642 | 2 |
| `phoenix-2014t/qwen25-05b/grpo/few-shot/run_20261004_043119/eval_baseline__zero-shot.json` | baseline | phoenix-2014t-qwen25-05b-grpo-few-shot | zero-shot | unknown | 2026-10-04 | 642 | 2 |
| `phoenix-2014t/qwen25-05b/grpo/few-shot/run_20261004_043119/eval_final.json` | unknown | phoenix-2014t-qwen25-05b-grpo-few-shot | few-shot | unknown | 2026-10-04 | 642 | 2 |
| `phoenix-2014t/qwen25-05b/grpo/few-shot/run_20261004_043119/eval_final__zero-shot.json` | unknown | phoenix-2014t-qwen25-05b-grpo-few-shot | zero-shot | unknown | 2026-10-04 | 642 | 2 |
| `phoenix-2014t/qwen25-05b/grpo/zero-shot/run_20261004_001607/eval_baseline.json` | baseline | phoenix-2014t-qwen25-05b-grpo-zero-shot | zero-shot | unknown | 2026-10-04 | 642 | 2 |
| `phoenix-2014t/qwen25-05b/grpo/zero-shot/run_20261004_001607/eval_baseline__few-shot.json` | baseline | phoenix-2014t-qwen25-05b-grpo-zero-shot | few-shot | unknown | 2026-10-04 | 642 | 2 |
| `phoenix-2014t/qwen25-05b/grpo/zero-shot/run_20261004_001607/eval_final.json` | unknown | phoenix-2014t-qwen25-05b-grpo-zero-shot | zero-shot | unknown | 2026-10-04 | 642 | 2 |
| `phoenix-2014t/qwen25-05b/grpo/zero-shot/run_20261004_001607/eval_final__few-shot.json` | unknown | phoenix-2014t-qwen25-05b-grpo-zero-shot | few-shot | unknown | 2026-10-04 | 642 | 2 |
| `phoenix-2014t/qwen25-05b/sft-grpo/few-shot/run_20261004_081604/eval_baseline.json` | baseline | few-shot | few-shot | unknown | 2026-10-04 | 642 | 2 |
| `phoenix-2014t/qwen25-05b/sft-grpo/few-shot/run_20261004_081604/eval_baseline__zero-shot.json` | baseline | few-shot | zero-shot | unknown | 2026-10-04 | 642 | 2 |
| `phoenix-2014t/qwen25-05b/sft-grpo/few-shot/run_20261004_081604/eval_final.json` | sft-grpo | few-shot | few-shot | unknown | 2026-10-04 | 642 | 2 |
| `phoenix-2014t/qwen25-05b/sft-grpo/few-shot/run_20261004_081604/eval_final__zero-shot.json` | sft-grpo | few-shot | zero-shot | unknown | 2026-10-04 | 642 | 2 |
| `phoenix-2014t/qwen25-05b/sft-grpo/zero-shot/run_20261004_112601/eval_baseline.json` | baseline | zero-shot | zero-shot | unknown | 2026-10-04 | 642 | 2 |
| `phoenix-2014t/qwen25-05b/sft-grpo/zero-shot/run_20261004_112601/eval_baseline__few-shot.json` | baseline | zero-shot | few-shot | unknown | 2026-10-04 | 642 | 2 |
| `phoenix-2014t/qwen25-05b/sft-grpo/zero-shot/run_20261004_112601/eval_final.json` | sft-grpo | zero-shot | zero-shot | unknown | 2026-10-04 | 642 | 2 |
| `phoenix-2014t/qwen25-05b/sft-grpo/zero-shot/run_20261004_112601/eval_final__few-shot.json` | sft-grpo | zero-shot | few-shot | unknown | 2026-10-04 | 642 | 2 |
| `phoenix-2014t/qwen25-05b/sft/zero-shot/run_20261003_232114/eval_baseline.json` | baseline | phoenix-2014t-qwen25-05b-sft-zero-shot | zero-shot | unknown | 2026-10-03 | 642 | 2 |
| `phoenix-2014t/qwen25-05b/sft/zero-shot/run_20261003_232114/eval_baseline__few-shot.json` | baseline | phoenix-2014t-qwen25-05b-sft-zero-shot | few-shot | unknown | 2026-10-03 | 642 | 2 |
| `phoenix-2014t/qwen25-05b/sft/zero-shot/run_20261003_232114/eval_final.json` | unknown | phoenix-2014t-qwen25-05b-sft-zero-shot | zero-shot | unknown | 2026-10-03 | 642 | 2 |
| `phoenix-2014t/qwen25-05b/sft/zero-shot/run_20261003_232114/eval_final__few-shot.json` | unknown | phoenix-2014t-qwen25-05b-sft-zero-shot | few-shot | unknown | 2026-10-03 | 642 | 2 |

## Ablation matrix overview — ROUGE-L

One row per cell (its path under `results/`), one column per prompting mode that has at least one value. `—` = the cell was not evaluated in that mode. `*` = multiple runs collapsed, latest shown. Each cell's own `eval_baseline` (the base model in that cell's context) is not a row here: it repeats the `baseline/` cells and stays in the paired comparisons below.

| cell | zero-shot | few-shot |
|---|---|---|
| phoenix-2014t/qwen25-05b/sft/zero-shot | 0.4854 | 0.4121 |
| phoenix-2014t/qwen25-05b/sft-grpo/few-shot | 0.5106 | 0.5039 |
| phoenix-2014t/qwen25-05b/sft-grpo/zero-shot | 0.5274 | 0.4460 |
| phoenix-2014t/qwen25-05b/grpo/few-shot | 0.1337 | 0.3601 |
| phoenix-2014t/qwen25-05b/grpo/zero-shot | 0.2030 | 0.3312 |

## phoenix-2014t — all metrics

Same rows as the matrix above, restricted to this dataset, one column per metric. On CoNLL-2003 and WOS-46985 read Exact Match and Gloss F1 (micro = entity-level F1 on CoNLL): ROUGE-L and BLEU split `PER:Mary` into `per` + `mary` and give partial credit to a wrong label of the same type.

| cell | prompting | ROUGE-L | Valid ROUGE-L | Exact Match | Non-copy Tok Acc | Pass@1 | Validity | BLEU (corpus) | BLEU (sent) | chrF2 (corpus) | chrF2 (sent) | Gloss F1 (mic) | Gloss F1 (sent) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| phoenix-2014t/qwen25-05b/sft/zero-shot | zero-shot | 0.4854 | 0.4839 | 0.0184 | 0.3812 | 0.7913 | 0.9969 | 0.1349 | 0.1418 | 42.6928 | 42.5937 | 0.5175 | 0.5195 |
| phoenix-2014t/qwen25-05b/sft/zero-shot | few-shot | 0.4121 | 0.4109 | 0.0103 | 0.3861 | 0.6994 | 0.9969 | 0.0936 | 0.1085 | 39.8081 | 39.7372 | 0.4465 | 0.4533 |
| phoenix-2014t/qwen25-05b/sft-grpo/few-shot | zero-shot | 0.5106 | 0.5105 | 0.0227 | 0.4183 | 0.8427 | 0.9997 | 0.1538 | 0.1521 | 45.2330 | 44.9755 | 0.5513 | 0.5486 |
| phoenix-2014t/qwen25-05b/sft-grpo/few-shot | few-shot | 0.5039 | 0.5039 | 0.0240 | 0.4429 | 0.8162 | 1.0000 | 0.1569 | 0.1528 | 45.6568 | 45.5933 | 0.5441 | 0.5438 |
| phoenix-2014t/qwen25-05b/sft-grpo/zero-shot | zero-shot | 0.5274 | 0.5274 | 0.0283 | 0.4599 | 0.8364 | 1.0000 | 0.1799 | 0.1651 | 48.1712 | 47.5429 | 0.5749 | 0.5685 |
| phoenix-2014t/qwen25-05b/sft-grpo/zero-shot | few-shot | 0.4460 | 0.4460 | 0.0106 | 0.4307 | 0.7555 | 1.0000 | 0.1055 | 0.1154 | 43.2557 | 43.0282 | 0.4892 | 0.4905 |
| phoenix-2014t/qwen25-05b/grpo/few-shot | zero-shot | 0.1337 | 0.1305 | 0.0000 | 0.1261 | 0.1262 | 0.9763 | 0.0036 | 0.0208 | 20.3556 | 20.4053 | 0.1290 | 0.1400 |
| phoenix-2014t/qwen25-05b/grpo/few-shot | few-shot | 0.3601 | 0.3598 | 0.0162 | 0.3325 | 0.5654 | 0.9991 | 0.0874 | 0.1023 | 35.1035 | 35.1132 | 0.3901 | 0.3979 |
| phoenix-2014t/qwen25-05b/grpo/zero-shot | zero-shot | 0.2030 | 0.2030 | 0.0003 | 0.1658 | 0.2383 | 0.9997 | 0.0112 | 0.0385 | 22.3499 | 22.4535 | 0.2173 | 0.2250 |
| phoenix-2014t/qwen25-05b/grpo/zero-shot | few-shot | 0.3312 | 0.3312 | 0.0146 | 0.2711 | 0.4938 | 1.0000 | 0.0794 | 0.0963 | 31.2445 | 31.4494 | 0.3564 | 0.3617 |

## Paired comparisons — one table per factor

### Factor: `method`

**training effect (same config, baseline vs checkpoint)** — `phoenix-2014t/qwen25-05b/sft-grpo/zero-shot/run_20261004_112601/eval_baseline__few-shot.json` (2026-10-04, few-shot, unknown) → `phoenix-2014t/qwen25-05b/sft-grpo/zero-shot/run_20261004_112601/eval_final__few-shot.json` (2026-10-04, few-shot, unknown)

| Metric | A | B | Δ (B−A) | Interpretation |
|---|---|---|---|---|
| ROUGE-L | 0.3103 | 0.4460 | +0.1357 | above noise threshold |
| Valid ROUGE-L | 0.3090 | 0.4460 | +0.1370 | above noise threshold |
| Exact Match | 0.0106 | 0.0106 | +0.0000 | NOT GENERALIZABLE (|delta| < 0.02): reproducible (training is bit-deterministic at fixed seed) but single-seed, so it may invert under another seed — do NOT interpret |
| Non-copy Tok Acc | 0.2923 | 0.4307 | +0.1384 | above noise threshold |
| Pass@1 | 0.4424 | 0.7555 | +0.3131 | above noise threshold |
| Validity | 0.9956 | 1.0000 | +0.0044 | NOT GENERALIZABLE (|delta| < 0.02): reproducible (training is bit-deterministic at fixed seed) but single-seed, so it may invert under another seed — do NOT interpret |
| BLEU (corpus) | 0.0599 | 0.1055 | +0.0456 | above noise threshold |
| BLEU (sent) | 0.0841 | 0.1154 | +0.0313 | above noise threshold |
| chrF2 (corpus) | 32.2282 | 43.2557 | +11.0275 | scale 0-100: 0.02 noise threshold not applicable |
| chrF2 (sent) | 32.5102 | 43.0282 | +10.5179 | scale 0-100: 0.02 noise threshold not applicable |
| Gloss F1 (mic) | 0.3194 | 0.4892 | +0.1698 | above noise threshold |
| Gloss F1 (sent) | 0.3410 | 0.4905 | +0.1494 | above noise threshold |
- ⚠️ factors not deducible on at least one side, equality NOT verified: reward_stack, rl_objective

**training effect (same config, baseline vs checkpoint)** — `phoenix-2014t/qwen25-05b/sft-grpo/few-shot/run_20261004_081604/eval_baseline.json` (2026-10-04, few-shot, unknown) → `phoenix-2014t/qwen25-05b/sft-grpo/few-shot/run_20261004_081604/eval_final.json` (2026-10-04, few-shot, unknown)

| Metric | A | B | Δ (B−A) | Interpretation |
|---|---|---|---|---|
| ROUGE-L | 0.3103 | 0.5039 | +0.1936 | above noise threshold |
| Valid ROUGE-L | 0.3090 | 0.5039 | +0.1949 | above noise threshold |
| Exact Match | 0.0106 | 0.0240 | +0.0134 | NOT GENERALIZABLE (|delta| < 0.02): reproducible (training is bit-deterministic at fixed seed) but single-seed, so it may invert under another seed — do NOT interpret |
| Non-copy Tok Acc | 0.2923 | 0.4429 | +0.1506 | above noise threshold |
| Pass@1 | 0.4424 | 0.8162 | +0.3738 | above noise threshold |
| Validity | 0.9956 | 1.0000 | +0.0044 | NOT GENERALIZABLE (|delta| < 0.02): reproducible (training is bit-deterministic at fixed seed) but single-seed, so it may invert under another seed — do NOT interpret |
| BLEU (corpus) | 0.0599 | 0.1569 | +0.0970 | above noise threshold |
| BLEU (sent) | 0.0841 | 0.1528 | +0.0687 | above noise threshold |
| chrF2 (corpus) | 32.2282 | 45.6568 | +13.4286 | scale 0-100: 0.02 noise threshold not applicable |
| chrF2 (sent) | 32.5102 | 45.5933 | +13.0830 | scale 0-100: 0.02 noise threshold not applicable |
| Gloss F1 (mic) | 0.3194 | 0.5441 | +0.2247 | above noise threshold |
| Gloss F1 (sent) | 0.3410 | 0.5438 | +0.2027 | above noise threshold |
- ⚠️ factors not deducible on at least one side, equality NOT verified: reward_stack, rl_objective

**training effect (same config, baseline vs checkpoint)** — `phoenix-2014t/qwen25-05b/sft-grpo/zero-shot/run_20261004_112601/eval_baseline.json` (2026-10-04, zero-shot, unknown) → `phoenix-2014t/qwen25-05b/sft-grpo/zero-shot/run_20261004_112601/eval_final.json` (2026-10-04, zero-shot, unknown)

| Metric | A | B | Δ (B−A) | Interpretation |
|---|---|---|---|---|
| ROUGE-L | 0.0961 | 0.5274 | +0.4313 | above noise threshold |
| Valid ROUGE-L | 0.0921 | 0.5274 | +0.4353 | above noise threshold |
| Exact Match | 0.0000 | 0.0283 | +0.0283 | above noise threshold |
| Non-copy Tok Acc | 0.1083 | 0.4599 | +0.3516 | above noise threshold |
| Pass@1 | 0.0732 | 0.8364 | +0.7632 | above noise threshold |
| Validity | 0.9586 | 1.0000 | +0.0414 | above noise threshold |
| BLEU (corpus) | 0.0023 | 0.1799 | +0.1776 | above noise threshold |
| BLEU (sent) | 0.0150 | 0.1651 | +0.1501 | above noise threshold |
| chrF2 (corpus) | 17.4805 | 48.1712 | +30.6907 | scale 0-100: 0.02 noise threshold not applicable |
| chrF2 (sent) | 17.6241 | 47.5429 | +29.9188 | scale 0-100: 0.02 noise threshold not applicable |
| Gloss F1 (mic) | 0.0856 | 0.5749 | +0.4894 | above noise threshold |
| Gloss F1 (sent) | 0.0999 | 0.5685 | +0.4686 | above noise threshold |
- ⚠️ factors not deducible on at least one side, equality NOT verified: reward_stack, rl_objective

**training effect (same config, baseline vs checkpoint)** — `phoenix-2014t/qwen25-05b/sft-grpo/few-shot/run_20261004_081604/eval_baseline__zero-shot.json` (2026-10-04, zero-shot, unknown) → `phoenix-2014t/qwen25-05b/sft-grpo/few-shot/run_20261004_081604/eval_final__zero-shot.json` (2026-10-04, zero-shot, unknown)

| Metric | A | B | Δ (B−A) | Interpretation |
|---|---|---|---|---|
| ROUGE-L | 0.0961 | 0.5106 | +0.4145 | above noise threshold |
| Valid ROUGE-L | 0.0921 | 0.5105 | +0.4183 | above noise threshold |
| Exact Match | 0.0000 | 0.0227 | +0.0227 | above noise threshold |
| Non-copy Tok Acc | 0.1083 | 0.4183 | +0.3100 | above noise threshold |
| Pass@1 | 0.0732 | 0.8427 | +0.7695 | above noise threshold |
| Validity | 0.9586 | 0.9997 | +0.0411 | above noise threshold |
| BLEU (corpus) | 0.0023 | 0.1538 | +0.1515 | above noise threshold |
| BLEU (sent) | 0.0150 | 0.1521 | +0.1372 | above noise threshold |
| chrF2 (corpus) | 17.4805 | 45.2330 | +27.7525 | scale 0-100: 0.02 noise threshold not applicable |
| chrF2 (sent) | 17.6241 | 44.9755 | +27.3514 | scale 0-100: 0.02 noise threshold not applicable |
| Gloss F1 (mic) | 0.0856 | 0.5513 | +0.4657 | above noise threshold |
| Gloss F1 (sent) | 0.0999 | 0.5486 | +0.4487 | above noise threshold |
- ⚠️ factors not deducible on at least one side, equality NOT verified: reward_stack, rl_objective

### Factor: `prompting`

🔴 **prompt-dependence (dual eval, same checkpoint)** — `phoenix-2014t/qwen25-05b/sft-grpo/zero-shot/run_20261004_112601/eval_baseline__few-shot.json` (2026-10-04, few-shot, unknown) → `phoenix-2014t/qwen25-05b/sft-grpo/zero-shot/run_20261004_112601/eval_baseline.json` (2026-10-04, zero-shot, unknown)

| Metric | A | B | Δ (B−A) | Interpretation |
|---|---|---|---|---|
| ROUGE-L | 0.3103 | 0.0961 | -0.2142 | above noise threshold |
| Valid ROUGE-L | 0.3090 | 0.0921 | -0.2168 | above noise threshold |
| Exact Match | 0.0106 | 0.0000 | -0.0106 | NOT GENERALIZABLE (|delta| < 0.02): reproducible (training is bit-deterministic at fixed seed) but single-seed, so it may invert under another seed — do NOT interpret |
| Non-copy Tok Acc | 0.2923 | 0.1083 | -0.1840 | above noise threshold |
| Pass@1 | 0.4424 | 0.0732 | -0.3692 | above noise threshold |
| Validity | 0.9956 | 0.9586 | -0.0371 | above noise threshold |
| BLEU (corpus) | 0.0599 | 0.0023 | -0.0576 | above noise threshold |
| BLEU (sent) | 0.0841 | 0.0150 | -0.0691 | above noise threshold |
| chrF2 (corpus) | 32.2282 | 17.4805 | -14.7477 | scale 0-100: 0.02 noise threshold not applicable |
| chrF2 (sent) | 32.5102 | 17.6241 | -14.8861 | scale 0-100: 0.02 noise threshold not applicable |
| Gloss F1 (mic) | 0.3194 | 0.0856 | -0.2338 | above noise threshold |
| Gloss F1 (sent) | 0.3410 | 0.0999 | -0.2412 | above noise threshold |
- ⚠️ HIGHLIGHT: same model evaluated in both prompting modes — how much does it depend on the prompt as a crutch?
- ⚠️ factors not deducible on at least one side, equality NOT verified: reward_stack, rl_objective

🔴 **prompt-dependence (dual eval, same checkpoint)** — `phoenix-2014t/qwen25-05b/sft-grpo/few-shot/run_20261004_081604/eval_baseline.json` (2026-10-04, few-shot, unknown) → `phoenix-2014t/qwen25-05b/sft-grpo/few-shot/run_20261004_081604/eval_baseline__zero-shot.json` (2026-10-04, zero-shot, unknown)

| Metric | A | B | Δ (B−A) | Interpretation |
|---|---|---|---|---|
| ROUGE-L | 0.3103 | 0.0961 | -0.2142 | above noise threshold |
| Valid ROUGE-L | 0.3090 | 0.0921 | -0.2168 | above noise threshold |
| Exact Match | 0.0106 | 0.0000 | -0.0106 | NOT GENERALIZABLE (|delta| < 0.02): reproducible (training is bit-deterministic at fixed seed) but single-seed, so it may invert under another seed — do NOT interpret |
| Non-copy Tok Acc | 0.2923 | 0.1083 | -0.1840 | above noise threshold |
| Pass@1 | 0.4424 | 0.0732 | -0.3692 | above noise threshold |
| Validity | 0.9956 | 0.9586 | -0.0371 | above noise threshold |
| BLEU (corpus) | 0.0599 | 0.0023 | -0.0576 | above noise threshold |
| BLEU (sent) | 0.0841 | 0.0150 | -0.0691 | above noise threshold |
| chrF2 (corpus) | 32.2282 | 17.4805 | -14.7477 | scale 0-100: 0.02 noise threshold not applicable |
| chrF2 (sent) | 32.5102 | 17.6241 | -14.8861 | scale 0-100: 0.02 noise threshold not applicable |
| Gloss F1 (mic) | 0.3194 | 0.0856 | -0.2338 | above noise threshold |
| Gloss F1 (sent) | 0.3410 | 0.0999 | -0.2412 | above noise threshold |
- ⚠️ HIGHLIGHT: same model evaluated in both prompting modes — how much does it depend on the prompt as a crutch?
- ⚠️ factors not deducible on at least one side, equality NOT verified: reward_stack, rl_objective

🔴 **prompt-dependence (dual eval, same checkpoint)** — `phoenix-2014t/qwen25-05b/grpo/few-shot/run_20261004_043119/eval_baseline.json` (2026-10-04, few-shot, unknown) → `phoenix-2014t/qwen25-05b/grpo/few-shot/run_20261004_043119/eval_baseline__zero-shot.json` (2026-10-04, zero-shot, unknown)

| Metric | A | B | Δ (B−A) | Interpretation |
|---|---|---|---|---|
| ROUGE-L | 0.3103 | 0.0961 | -0.2142 | above noise threshold |
| Valid ROUGE-L | 0.3090 | 0.0921 | -0.2168 | above noise threshold |
| Exact Match | 0.0106 | 0.0000 | -0.0106 | NOT GENERALIZABLE (|delta| < 0.02): reproducible (training is bit-deterministic at fixed seed) but single-seed, so it may invert under another seed — do NOT interpret |
| Non-copy Tok Acc | 0.2923 | 0.1083 | -0.1840 | above noise threshold |
| Pass@1 | 0.4424 | 0.0732 | -0.3692 | above noise threshold |
| Validity | 0.9956 | 0.9586 | -0.0371 | above noise threshold |
| BLEU (corpus) | 0.0599 | 0.0023 | -0.0576 | above noise threshold |
| BLEU (sent) | 0.0841 | 0.0150 | -0.0691 | above noise threshold |
| chrF2 (corpus) | 32.2282 | 17.4805 | -14.7477 | scale 0-100: 0.02 noise threshold not applicable |
| chrF2 (sent) | 32.5102 | 17.6241 | -14.8861 | scale 0-100: 0.02 noise threshold not applicable |
| Gloss F1 (mic) | 0.3194 | 0.0856 | -0.2338 | above noise threshold |
| Gloss F1 (sent) | 0.3410 | 0.0999 | -0.2412 | above noise threshold |
- ⚠️ HIGHLIGHT: same model evaluated in both prompting modes — how much does it depend on the prompt as a crutch?
- ⚠️ factors not deducible on at least one side, equality NOT verified: reward_stack, rl_objective

🔴 **prompt-dependence (dual eval, same checkpoint)** — `phoenix-2014t/qwen25-05b/grpo/zero-shot/run_20261004_001607/eval_baseline__few-shot.json` (2026-10-04, few-shot, unknown) → `phoenix-2014t/qwen25-05b/grpo/zero-shot/run_20261004_001607/eval_baseline.json` (2026-10-04, zero-shot, unknown)

| Metric | A | B | Δ (B−A) | Interpretation |
|---|---|---|---|---|
| ROUGE-L | 0.3103 | 0.0961 | -0.2142 | above noise threshold |
| Valid ROUGE-L | 0.3090 | 0.0921 | -0.2168 | above noise threshold |
| Exact Match | 0.0106 | 0.0000 | -0.0106 | NOT GENERALIZABLE (|delta| < 0.02): reproducible (training is bit-deterministic at fixed seed) but single-seed, so it may invert under another seed — do NOT interpret |
| Non-copy Tok Acc | 0.2923 | 0.1083 | -0.1840 | above noise threshold |
| Pass@1 | 0.4424 | 0.0732 | -0.3692 | above noise threshold |
| Validity | 0.9956 | 0.9586 | -0.0371 | above noise threshold |
| BLEU (corpus) | 0.0599 | 0.0023 | -0.0576 | above noise threshold |
| BLEU (sent) | 0.0841 | 0.0150 | -0.0691 | above noise threshold |
| chrF2 (corpus) | 32.2282 | 17.4805 | -14.7477 | scale 0-100: 0.02 noise threshold not applicable |
| chrF2 (sent) | 32.5102 | 17.6241 | -14.8861 | scale 0-100: 0.02 noise threshold not applicable |
| Gloss F1 (mic) | 0.3194 | 0.0856 | -0.2338 | above noise threshold |
| Gloss F1 (sent) | 0.3410 | 0.0999 | -0.2412 | above noise threshold |
- ⚠️ HIGHLIGHT: same model evaluated in both prompting modes — how much does it depend on the prompt as a crutch?
- ⚠️ factors not deducible on at least one side, equality NOT verified: reward_stack, rl_objective

🔴 **prompt-dependence (dual eval, same checkpoint)** — `phoenix-2014t/qwen25-05b/sft/zero-shot/run_20261003_232114/eval_baseline__few-shot.json` (2026-10-03, few-shot, unknown) → `phoenix-2014t/qwen25-05b/sft/zero-shot/run_20261003_232114/eval_baseline.json` (2026-10-03, zero-shot, unknown)

| Metric | A | B | Δ (B−A) | Interpretation |
|---|---|---|---|---|
| ROUGE-L | 0.3103 | 0.0961 | -0.2142 | above noise threshold |
| Valid ROUGE-L | 0.3090 | 0.0921 | -0.2168 | above noise threshold |
| Exact Match | 0.0106 | 0.0000 | -0.0106 | NOT GENERALIZABLE (|delta| < 0.02): reproducible (training is bit-deterministic at fixed seed) but single-seed, so it may invert under another seed — do NOT interpret |
| Non-copy Tok Acc | 0.2923 | 0.1083 | -0.1840 | above noise threshold |
| Pass@1 | 0.4424 | 0.0732 | -0.3692 | above noise threshold |
| Validity | 0.9956 | 0.9586 | -0.0371 | above noise threshold |
| BLEU (corpus) | 0.0599 | 0.0023 | -0.0576 | above noise threshold |
| BLEU (sent) | 0.0841 | 0.0150 | -0.0691 | above noise threshold |
| chrF2 (corpus) | 32.2282 | 17.4805 | -14.7477 | scale 0-100: 0.02 noise threshold not applicable |
| chrF2 (sent) | 32.5102 | 17.6241 | -14.8861 | scale 0-100: 0.02 noise threshold not applicable |
| Gloss F1 (mic) | 0.3194 | 0.0856 | -0.2338 | above noise threshold |
| Gloss F1 (sent) | 0.3410 | 0.0999 | -0.2412 | above noise threshold |
- ⚠️ HIGHLIGHT: same model evaluated in both prompting modes — how much does it depend on the prompt as a crutch?
- ⚠️ factors not deducible on at least one side, equality NOT verified: reward_stack, rl_objective

🔴 **prompt-dependence (dual eval, same checkpoint)** — `phoenix-2014t/qwen25-05b/sft-grpo/zero-shot/run_20261004_112601/eval_final__few-shot.json` (2026-10-04, few-shot, unknown) → `phoenix-2014t/qwen25-05b/sft-grpo/zero-shot/run_20261004_112601/eval_final.json` (2026-10-04, zero-shot, unknown)

| Metric | A | B | Δ (B−A) | Interpretation |
|---|---|---|---|---|
| ROUGE-L | 0.4460 | 0.5274 | +0.0814 | above noise threshold |
| Valid ROUGE-L | 0.4460 | 0.5274 | +0.0814 | above noise threshold |
| Exact Match | 0.0106 | 0.0283 | +0.0178 | NOT GENERALIZABLE (|delta| < 0.02): reproducible (training is bit-deterministic at fixed seed) but single-seed, so it may invert under another seed — do NOT interpret |
| Non-copy Tok Acc | 0.4307 | 0.4599 | +0.0291 | above noise threshold |
| Pass@1 | 0.7555 | 0.8364 | +0.0810 | above noise threshold |
| Validity | 1.0000 | 1.0000 | +0.0000 | NOT GENERALIZABLE (|delta| < 0.02): reproducible (training is bit-deterministic at fixed seed) but single-seed, so it may invert under another seed — do NOT interpret |
| BLEU (corpus) | 0.1055 | 0.1799 | +0.0745 | above noise threshold |
| BLEU (sent) | 0.1154 | 0.1651 | +0.0497 | above noise threshold |
| chrF2 (corpus) | 43.2557 | 48.1712 | +4.9155 | scale 0-100: 0.02 noise threshold not applicable |
| chrF2 (sent) | 43.0282 | 47.5429 | +4.5147 | scale 0-100: 0.02 noise threshold not applicable |
| Gloss F1 (mic) | 0.4892 | 0.5749 | +0.0858 | above noise threshold |
| Gloss F1 (sent) | 0.4905 | 0.5685 | +0.0780 | above noise threshold |
- ⚠️ HIGHLIGHT: same model evaluated in both prompting modes — how much does it depend on the prompt as a crutch?
- ⚠️ factors not deducible on at least one side, equality NOT verified: reward_stack, rl_objective

🔴 **prompt-dependence (dual eval, same checkpoint)** — `phoenix-2014t/qwen25-05b/sft-grpo/few-shot/run_20261004_081604/eval_final.json` (2026-10-04, few-shot, unknown) → `phoenix-2014t/qwen25-05b/sft-grpo/few-shot/run_20261004_081604/eval_final__zero-shot.json` (2026-10-04, zero-shot, unknown)

| Metric | A | B | Δ (B−A) | Interpretation |
|---|---|---|---|---|
| ROUGE-L | 0.5039 | 0.5106 | +0.0067 | NOT GENERALIZABLE (|delta| < 0.02): reproducible (training is bit-deterministic at fixed seed) but single-seed, so it may invert under another seed — do NOT interpret |
| Valid ROUGE-L | 0.5039 | 0.5105 | +0.0066 | NOT GENERALIZABLE (|delta| < 0.02): reproducible (training is bit-deterministic at fixed seed) but single-seed, so it may invert under another seed — do NOT interpret |
| Exact Match | 0.0240 | 0.0227 | -0.0012 | NOT GENERALIZABLE (|delta| < 0.02): reproducible (training is bit-deterministic at fixed seed) but single-seed, so it may invert under another seed — do NOT interpret |
| Non-copy Tok Acc | 0.4429 | 0.4183 | -0.0246 | above noise threshold |
| Pass@1 | 0.8162 | 0.8427 | +0.0265 | above noise threshold |
| Validity | 1.0000 | 0.9997 | -0.0003 | NOT GENERALIZABLE (|delta| < 0.02): reproducible (training is bit-deterministic at fixed seed) but single-seed, so it may invert under another seed — do NOT interpret |
| BLEU (corpus) | 0.1569 | 0.1538 | -0.0030 | NOT GENERALIZABLE (|delta| < 0.02): reproducible (training is bit-deterministic at fixed seed) but single-seed, so it may invert under another seed — do NOT interpret |
| BLEU (sent) | 0.1528 | 0.1521 | -0.0006 | NOT GENERALIZABLE (|delta| < 0.02): reproducible (training is bit-deterministic at fixed seed) but single-seed, so it may invert under another seed — do NOT interpret |
| chrF2 (corpus) | 45.6568 | 45.2330 | -0.4239 | scale 0-100: 0.02 noise threshold not applicable |
| chrF2 (sent) | 45.5933 | 44.9755 | -0.6177 | scale 0-100: 0.02 noise threshold not applicable |
| Gloss F1 (mic) | 0.5441 | 0.5513 | +0.0072 | NOT GENERALIZABLE (|delta| < 0.02): reproducible (training is bit-deterministic at fixed seed) but single-seed, so it may invert under another seed — do NOT interpret |
| Gloss F1 (sent) | 0.5438 | 0.5486 | +0.0049 | NOT GENERALIZABLE (|delta| < 0.02): reproducible (training is bit-deterministic at fixed seed) but single-seed, so it may invert under another seed — do NOT interpret |
- ⚠️ HIGHLIGHT: same model evaluated in both prompting modes — how much does it depend on the prompt as a crutch?
- ⚠️ factors not deducible on at least one side, equality NOT verified: reward_stack, rl_objective

🔴 **prompt-dependence (dual eval, same checkpoint)** — `phoenix-2014t/qwen25-05b/grpo/few-shot/run_20261004_043119/eval_final.json` (2026-10-04, few-shot, unknown) → `phoenix-2014t/qwen25-05b/grpo/few-shot/run_20261004_043119/eval_final__zero-shot.json` (2026-10-04, zero-shot, unknown)

| Metric | A | B | Δ (B−A) | Interpretation |
|---|---|---|---|---|
| ROUGE-L | 0.3601 | 0.1337 | -0.2265 | above noise threshold |
| Valid ROUGE-L | 0.3598 | 0.1305 | -0.2293 | above noise threshold |
| Exact Match | 0.0162 | 0.0000 | -0.0162 | NOT GENERALIZABLE (|delta| < 0.02): reproducible (training is bit-deterministic at fixed seed) but single-seed, so it may invert under another seed — do NOT interpret |
| Non-copy Tok Acc | 0.3325 | 0.1261 | -0.2064 | above noise threshold |
| Pass@1 | 0.5654 | 0.1262 | -0.4393 | above noise threshold |
| Validity | 0.9991 | 0.9763 | -0.0227 | above noise threshold |
| BLEU (corpus) | 0.0874 | 0.0036 | -0.0838 | above noise threshold |
| BLEU (sent) | 0.1023 | 0.0208 | -0.0815 | above noise threshold |
| chrF2 (corpus) | 35.1035 | 20.3556 | -14.7479 | scale 0-100: 0.02 noise threshold not applicable |
| chrF2 (sent) | 35.1132 | 20.4053 | -14.7079 | scale 0-100: 0.02 noise threshold not applicable |
| Gloss F1 (mic) | 0.3901 | 0.1290 | -0.2611 | above noise threshold |
| Gloss F1 (sent) | 0.3979 | 0.1400 | -0.2580 | above noise threshold |
- ⚠️ HIGHLIGHT: same model evaluated in both prompting modes — how much does it depend on the prompt as a crutch?
- ⚠️ factors not deducible on at least one side, equality NOT verified: reward_stack, rl_objective

🔴 **prompt-dependence (dual eval, same checkpoint)** — `phoenix-2014t/qwen25-05b/grpo/zero-shot/run_20261004_001607/eval_final__few-shot.json` (2026-10-04, few-shot, unknown) → `phoenix-2014t/qwen25-05b/grpo/zero-shot/run_20261004_001607/eval_final.json` (2026-10-04, zero-shot, unknown)

| Metric | A | B | Δ (B−A) | Interpretation |
|---|---|---|---|---|
| ROUGE-L | 0.3312 | 0.2030 | -0.1282 | above noise threshold |
| Valid ROUGE-L | 0.3312 | 0.2030 | -0.1283 | above noise threshold |
| Exact Match | 0.0146 | 0.0003 | -0.0143 | NOT GENERALIZABLE (|delta| < 0.02): reproducible (training is bit-deterministic at fixed seed) but single-seed, so it may invert under another seed — do NOT interpret |
| Non-copy Tok Acc | 0.2711 | 0.1658 | -0.1054 | above noise threshold |
| Pass@1 | 0.4938 | 0.2383 | -0.2555 | above noise threshold |
| Validity | 1.0000 | 0.9997 | -0.0003 | NOT GENERALIZABLE (|delta| < 0.02): reproducible (training is bit-deterministic at fixed seed) but single-seed, so it may invert under another seed — do NOT interpret |
| BLEU (corpus) | 0.0794 | 0.0112 | -0.0682 | above noise threshold |
| BLEU (sent) | 0.0963 | 0.0385 | -0.0577 | above noise threshold |
| chrF2 (corpus) | 31.2445 | 22.3499 | -8.8946 | scale 0-100: 0.02 noise threshold not applicable |
| chrF2 (sent) | 31.4494 | 22.4535 | -8.9959 | scale 0-100: 0.02 noise threshold not applicable |
| Gloss F1 (mic) | 0.3564 | 0.2173 | -0.1390 | above noise threshold |
| Gloss F1 (sent) | 0.3617 | 0.2250 | -0.1367 | above noise threshold |
- ⚠️ HIGHLIGHT: same model evaluated in both prompting modes — how much does it depend on the prompt as a crutch?
- ⚠️ factors not deducible on at least one side, equality NOT verified: reward_stack, rl_objective

🔴 **prompt-dependence (dual eval, same checkpoint)** — `phoenix-2014t/qwen25-05b/sft/zero-shot/run_20261003_232114/eval_final__few-shot.json` (2026-10-03, few-shot, unknown) → `phoenix-2014t/qwen25-05b/sft/zero-shot/run_20261003_232114/eval_final.json` (2026-10-03, zero-shot, unknown)

| Metric | A | B | Δ (B−A) | Interpretation |
|---|---|---|---|---|
| ROUGE-L | 0.4121 | 0.4854 | +0.0733 | above noise threshold |
| Valid ROUGE-L | 0.4109 | 0.4839 | +0.0731 | above noise threshold |
| Exact Match | 0.0103 | 0.0184 | +0.0081 | NOT GENERALIZABLE (|delta| < 0.02): reproducible (training is bit-deterministic at fixed seed) but single-seed, so it may invert under another seed — do NOT interpret |
| Non-copy Tok Acc | 0.3861 | 0.3812 | -0.0049 | NOT GENERALIZABLE (|delta| < 0.02): reproducible (training is bit-deterministic at fixed seed) but single-seed, so it may invert under another seed — do NOT interpret |
| Pass@1 | 0.6994 | 0.7913 | +0.0919 | above noise threshold |
| Validity | 0.9969 | 0.9969 | +0.0000 | NOT GENERALIZABLE (|delta| < 0.02): reproducible (training is bit-deterministic at fixed seed) but single-seed, so it may invert under another seed — do NOT interpret |
| BLEU (corpus) | 0.0936 | 0.1349 | +0.0413 | above noise threshold |
| BLEU (sent) | 0.1085 | 0.1418 | +0.0333 | above noise threshold |
| chrF2 (corpus) | 39.8081 | 42.6928 | +2.8847 | scale 0-100: 0.02 noise threshold not applicable |
| chrF2 (sent) | 39.7372 | 42.5937 | +2.8565 | scale 0-100: 0.02 noise threshold not applicable |
| Gloss F1 (mic) | 0.4465 | 0.5175 | +0.0710 | above noise threshold |
| Gloss F1 (sent) | 0.4533 | 0.5195 | +0.0662 | above noise threshold |
- ⚠️ HIGHLIGHT: same model evaluated in both prompting modes — how much does it depend on the prompt as a crutch?
- ⚠️ factors not deducible on at least one side, equality NOT verified: reward_stack, rl_objective

### Factor: `grammar`

**No paired comparison available.** 6 group(s) with 2+ runs but no pair with a known, differing value of this factor


## Missing paired comparisons (declared, not fabricated)

- **`grammar`**: 6 group(s) with 2+ runs but no pair with a known, differing value of this factor.
  Available typologies: `g|r|a|m|m|a|r|=|u|n|k|n|o|w|n`

## Figures

- `campaign_pairwise_deltas.png` — one panel per factor: paired deltas as diverging bars with the ±0.02 noise band shaded. Diverging bars make the SIGN of each effect immediately visible, and the shaded band makes it impossible to mistake sub-noise deltas for results.
- `campaign_matrix.png` — the ablation matrix as a heatmap (rows: method/variant, columns: prompting, annotated with ROUGE-L); empty typologies stay visibly grey so gaps in the grid are impossible to miss.

