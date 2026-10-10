# Campaign Report — conll-2003 / qwen25-05b

Generated: 2026-10-10T12:57:03+00:00

> Complementare ad `ablation-summary`: qui i run sono appaiati
> per fattore sperimentale. **Legge i JSON di eval: non
> ricalcola nessuna metrica.**

## ⚠️ Mandatory caveats — read before any conclusion

1. **Interpretability threshold 0.02** (metrics on the [0,1] scale): training is bit-deterministic at a fixed seed — two executions of the same cell produce byte-identical generations — so there is no run-to-run noise. But no cell was replicated across seeds, so **deltas below 0.02 are reproducible yet not generalizable**: they may invert under another seed. A delta of 0.003 is not a result.
2. **Prompt counts** (`num_samples_evaluated`) are shown for every run; a pair with differing counts is explicitly flagged (metrics on different samples are not comparable). Historically this number moved from 2000 to 3000 prompts.
3. **`metrics_version`** is shown for every run and flagged when it differs across a pair.
4. **Run dates** are shown for every run: older runs come from different configurations (e.g. `max_steps` moved 2000→5000).
5. **Saturated overlap metrics — do not rank on these**: `rouge_l_mean`, `rouge_l_median`, `valid_rouge_l_mean`, `gloss_f1_micro`, `gloss_f1_sentence_mean`, `chrf_corpus`, `chrf_sentence_mean` are all within noise of a context-free rule baseline on ASLG-PC12 (ROUGE-L ~0.97, EM 0.59, see `src/analysis/rule_baseline.py`). Use the reward-independent primary metrics instead — `exact_match`, `non_copy_token_accuracy` — for any claim that a cell translates better, not merely that it scores higher on the same metric family the reward optimizes.
6. **Pairing warnings:**
   - [method] conll-2003/qwen25-05b/ablations/decoding/no-grammar/run_20261009_133123/eval_baseline.json vs conll-2003/qwen25-05b/sft-grpo/zero-shot/run_20261009_113122/eval_final__few-shot.json: prompt_context_fingerprint DIFFERS: the eval contexts (prompting/grammar/num_samples/model) are not identical — this pair may differ in more than the method factor
   - [method] conll-2003/qwen25-05b/ablations/decoding/no-grammar-zero-shot/run_20261009_162603/eval_baseline.json vs conll-2003/qwen25-05b/sft-grpo/zero-shot/run_20261009_113122/eval_final.json: prompt_context_fingerprint DIFFERS: the eval contexts (prompting/grammar/num_samples/model) are not identical — this pair may differ in more than the method factor

## How factors are deduced

| Factor | Rule | If not deducible |
|---|---|---|
| `method` | token del nome cella nel percorso (sft-grpo / sft-only / grpo-only / t2g-*); eval_baseline.json => baseline (nessun checkpoint) | unknown (never guessed) |
| `variant` | remaining cell-name tokens after model tag, method and known markers (e.g. structure, viterbi, all-rewards); no-grammar is a grammar marker, not a variant | empty string (plain cell) |
| `prompting` | 1) 'prompting.mode' stamp in the eval JSON; 2) filename suffix __zero-shot / __few-shot (dual/override pass); 3) t2g- zero-shot naming convention (baseline cells only) | unknown — the config-implied mode is NEVER assumed, it can differ from the effective one under dual prompting |
| `grammar` | 'no-grammar' in cell name => off; 'grammar' in t2g-* cell name => on (that family names the toggle explicitly) | unknown (never guessed) |
| `reward_stack` | NOT deducible from path / JSON stamp / payload fields (it lives in the resolved config, which eval JSONs do not embed) | always unknown, declared in every pairing caveat |
| `rl_objective` | NOT deducible from path / JSON stamp / payload fields | always unknown, declared in every pairing caveat |

## Selected runs (22 typologies, latest per typology)

| Run | method | variant | prompting | grammar | Date | Prompts | metrics_version |
|---|---|---|---|---|---|---|---|
| `conll-2003/qwen25-05b/ablations/decoding/no-grammar-zero-shot/run_20261009_162603/eval_baseline.json` | baseline | conll-2003-qwen25-05b-ablations-decoding-no-grammar-zero-shot | zero-shot | unknown | 2026-10-09 | 2000 | 2 |
| `conll-2003/qwen25-05b/ablations/decoding/no-grammar-zero-shot/run_20261009_162603/eval_final.json` | unknown | conll-2003-qwen25-05b-ablations-decoding-no-grammar-zero-shot | zero-shot | unknown | 2026-10-09 | 2000 | 2 |
| `conll-2003/qwen25-05b/ablations/decoding/no-grammar/run_20261009_133123/eval_baseline.json` | baseline | conll-2003-qwen25-05b-ablations-decoding-no-grammar | few-shot | unknown | 2026-10-09 | 2000 | 2 |
| `conll-2003/qwen25-05b/ablations/decoding/no-grammar/run_20261009_133123/eval_final.json` | unknown | conll-2003-qwen25-05b-ablations-decoding-no-grammar | few-shot | unknown | 2026-10-09 | 2000 | 2 |
| `conll-2003/qwen25-05b/ablations/objectives/sft-allowed-mass/run_20261009_190603/eval_final.json` | unknown | conll-2003-qwen25-05b-ablations-objectives-sft-allowed-mass | zero-shot | unknown | 2026-10-09 | 2000 | 2 |
| `conll-2003/qwen25-05b/baseline/few-shot/run_20261008_235550/eval_zero_shot.json` | unknown | conll-2003-qwen25-05b-baseline-few-shot | few-shot | unknown | 2026-10-08 | 2000 | 2 |
| `conll-2003/qwen25-05b/baseline/zero-shot-no-grammar/run_20261009_001553/eval_zero_shot.json` | unknown | conll-2003-qwen25-05b-baseline-zero-shot-no-grammar | zero-shot | unknown | 2026-10-09 | 2000 | 2 |
| `conll-2003/qwen25-05b/baseline/zero-shot/run_20261008_233547/eval_zero_shot.json` | unknown | conll-2003-qwen25-05b-baseline-zero-shot | zero-shot | unknown | 2026-10-08 | 2000 | 2 |
| `conll-2003/qwen25-05b/grpo/few-shot/run_20261009_052104/eval_baseline.json` | baseline | conll-2003-qwen25-05b-grpo-few-shot | few-shot | unknown | 2026-10-09 | 2000 | 2 |
| `conll-2003/qwen25-05b/grpo/few-shot/run_20261009_052104/eval_baseline__zero-shot.json` | baseline | conll-2003-qwen25-05b-grpo-few-shot | zero-shot | unknown | 2026-10-09 | 2000 | 2 |
| `conll-2003/qwen25-05b/grpo/few-shot/run_20261009_052104/eval_final.json` | unknown | conll-2003-qwen25-05b-grpo-few-shot | few-shot | unknown | 2026-10-09 | 2000 | 2 |
| `conll-2003/qwen25-05b/grpo/few-shot/run_20261009_052104/eval_final__zero-shot.json` | unknown | conll-2003-qwen25-05b-grpo-few-shot | zero-shot | unknown | 2026-10-09 | 2000 | 2 |
| `conll-2003/qwen25-05b/grpo/zero-shot/run_20261009_020134/eval_final.json` | unknown | conll-2003-qwen25-05b-grpo-zero-shot | zero-shot | unknown | 2026-10-09 | 2000 | 2 |
| `conll-2003/qwen25-05b/grpo/zero-shot/run_20261009_020134/eval_final__few-shot.json` | unknown | conll-2003-qwen25-05b-grpo-zero-shot | few-shot | unknown | 2026-10-09 | 2000 | 2 |
| `conll-2003/qwen25-05b/sft-grpo/few-shot/run_20261009_090130/eval_final.json` | sft-grpo | few-shot | few-shot | unknown | 2026-10-09 | 2000 | 2 |
| `conll-2003/qwen25-05b/sft-grpo/few-shot/run_20261009_090130/eval_final__zero-shot.json` | sft-grpo | few-shot | zero-shot | unknown | 2026-10-09 | 2000 | 2 |
| `conll-2003/qwen25-05b/sft-grpo/zero-shot/run_20261009_113122/eval_final.json` | sft-grpo | zero-shot | zero-shot | unknown | 2026-10-09 | 2000 | 2 |
| `conll-2003/qwen25-05b/sft-grpo/zero-shot/run_20261009_113122/eval_final__few-shot.json` | sft-grpo | zero-shot | few-shot | unknown | 2026-10-09 | 2000 | 2 |
| `conll-2003/qwen25-05b/sft/zero-shot/run_20261009_004112/eval_baseline.json` | baseline | conll-2003-qwen25-05b-sft-zero-shot | zero-shot | unknown | 2026-10-09 | 2000 | 2 |
| `conll-2003/qwen25-05b/sft/zero-shot/run_20261009_004112/eval_baseline__few-shot.json` | baseline | conll-2003-qwen25-05b-sft-zero-shot | few-shot | unknown | 2026-10-09 | 2000 | 2 |
| `conll-2003/qwen25-05b/sft/zero-shot/run_20261009_004112/eval_final.json` | unknown | conll-2003-qwen25-05b-sft-zero-shot | zero-shot | unknown | 2026-10-09 | 2000 | 2 |
| `conll-2003/qwen25-05b/sft/zero-shot/run_20261009_004112/eval_final__few-shot.json` | unknown | conll-2003-qwen25-05b-sft-zero-shot | few-shot | unknown | 2026-10-09 | 2000 | 2 |

## Ablation matrix overview — ROUGE-L

One row per cell (its path under `results/`), one column per prompting mode that has at least one value. `—` = the cell was not evaluated in that mode. `*` = multiple runs collapsed, latest shown. Each cell's own `eval_baseline` (the base model in that cell's context) is not a row here: it repeats the `baseline/` cells and stays in the paired comparisons below.

| cell | zero-shot | few-shot |
|---|---|---|
| conll-2003/qwen25-05b/baseline/few-shot | — | 0.3805 |
| conll-2003/qwen25-05b/baseline/zero-shot | 0.2708 | — |
| conll-2003/qwen25-05b/baseline/zero-shot-no-grammar | 0.2676 | — |
| conll-2003/qwen25-05b/sft/zero-shot | 0.9003 | 0.8138 |
| conll-2003/qwen25-05b/sft-grpo/few-shot | 0.8993 | 0.8859 |
| conll-2003/qwen25-05b/sft-grpo/zero-shot | 0.9128 | 0.8159 |
| conll-2003/qwen25-05b/grpo/few-shot | 0.3115 | 0.5483 |
| conll-2003/qwen25-05b/grpo/zero-shot | 0.3526 | 0.4025 |
| conll-2003/qwen25-05b/ablations/decoding/no-grammar | — | 0.5906 |
| conll-2003/qwen25-05b/ablations/decoding/no-grammar-zero-shot | 0.4548 | — |
| conll-2003/qwen25-05b/ablations/objectives/sft-allowed-mass | 0.9039 | — |

## conll-2003 — all metrics

Same rows as the matrix above, restricted to this dataset, one column per metric. On CoNLL-2003 and WOS-46985 read Exact Match and Gloss F1 (micro = entity-level F1 on CoNLL): ROUGE-L and BLEU split `PER:Mary` into `per` + `mary` and give partial credit to a wrong label of the same type.

| cell | prompting | ROUGE-L | Valid ROUGE-L | Exact Match | Non-copy Tok Acc | Pass@1 | Validity | BLEU (corpus) | BLEU (sent) | chrF2 (corpus) | chrF2 (sent) | Gloss F1 (mic) | Gloss F1 (sent) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| conll-2003/qwen25-05b/baseline/few-shot | few-shot | 0.3805 | 0.3706 | 0.0662 | 0.3169 | 0.5865 | 0.9739 | 0.1429 | 0.2204 | 39.8261 | 41.4512 | 0.1714 | 0.2366 |
| conll-2003/qwen25-05b/baseline/zero-shot | zero-shot | 0.2708 | 0.2625 | 0.0453 | 0.1458 | 0.4145 | 0.9693 | 0.0893 | 0.1329 | 29.5092 | 29.5764 | 0.0762 | 0.1033 |
| conll-2003/qwen25-05b/baseline/zero-shot-no-grammar | zero-shot | 0.2676 | 0.0021 | 0.0011 | 0.0066 | 0.4260 | 0.0079 | 0.0089 | 0.0219 | 17.9933 | 18.2289 | 0.0022 | 0.0056 |
| conll-2003/qwen25-05b/sft/zero-shot | zero-shot | 0.9003 | 0.9001 | 0.7432 | 0.8268 | 0.9685 | 0.9998 | 0.8330 | 0.8600 | 85.9758 | 89.4084 | 0.8205 | 0.8421 |
| conll-2003/qwen25-05b/sft/zero-shot | few-shot | 0.8138 | 0.8116 | 0.5975 | 0.7356 | 0.9210 | 0.9973 | 0.6567 | 0.7476 | 76.0076 | 81.2920 | 0.6911 | 0.7445 |
| conll-2003/qwen25-05b/sft-grpo/few-shot | zero-shot | 0.8993 | 0.8993 | 0.7498 | 0.8340 | 0.9625 | 1.0000 | 0.8571 | 0.8608 | 86.6384 | 89.2491 | 0.8372 | 0.8442 |
| conll-2003/qwen25-05b/sft-grpo/few-shot | few-shot | 0.8859 | 0.8855 | 0.7289 | 0.8277 | 0.9560 | 0.9995 | 0.8230 | 0.8424 | 85.4522 | 88.1871 | 0.8204 | 0.8310 |
| conll-2003/qwen25-05b/sft-grpo/zero-shot | zero-shot | 0.9128 | 0.9127 | 0.7807 | 0.8606 | 0.9660 | 0.9999 | 0.8494 | 0.8778 | 88.0997 | 90.8031 | 0.8531 | 0.8705 |
| conll-2003/qwen25-05b/sft-grpo/zero-shot | few-shot | 0.8159 | 0.8139 | 0.6249 | 0.7730 | 0.9050 | 0.9976 | 0.6651 | 0.7605 | 78.3079 | 82.0286 | 0.7162 | 0.7615 |
| conll-2003/qwen25-05b/grpo/few-shot | zero-shot | 0.3115 | 0.3092 | 0.0741 | 0.2255 | 0.4805 | 0.9925 | 0.1276 | 0.1705 | 35.6965 | 35.3721 | 0.1341 | 0.1649 |
| conll-2003/qwen25-05b/grpo/few-shot | few-shot | 0.5483 | 0.5472 | 0.3315 | 0.4146 | 0.6895 | 0.9979 | 0.3885 | 0.4583 | 48.0140 | 54.7296 | 0.3859 | 0.4566 |
| conll-2003/qwen25-05b/grpo/zero-shot | zero-shot | 0.3526 | 0.3519 | 0.0228 | 0.3262 | 0.6200 | 0.9979 | 0.1964 | 0.1790 | 44.0094 | 40.0840 | 0.2129 | 0.2163 |
| conll-2003/qwen25-05b/grpo/zero-shot | few-shot | 0.4025 | 0.3995 | 0.0448 | 0.4452 | 0.6695 | 0.9924 | 0.2097 | 0.2279 | 49.8149 | 46.0096 | 0.2653 | 0.2864 |
| conll-2003/qwen25-05b/ablations/decoding/no-grammar | few-shot | 0.5906 | 0.4289 | 0.3473 | 0.4019 | 0.7305 | 0.7261 | 0.4205 | 0.4740 | 48.2561 | 56.4984 | 0.3907 | 0.4673 |
| conll-2003/qwen25-05b/ablations/decoding/no-grammar-zero-shot | zero-shot | 0.4548 | 0.0650 | 0.0611 | 0.1084 | 0.7405 | 0.1430 | 0.0142 | 0.1056 | 26.7111 | 34.0781 | 0.0530 | 0.1111 |
| conll-2003/qwen25-05b/ablations/objectives/sft-allowed-mass | zero-shot | 0.9039 | 0.9039 | 0.7469 | 0.8274 | 0.9695 | 1.0000 | 0.8359 | 0.8632 | 85.7456 | 89.6914 | 0.8252 | 0.8500 |

## Paired comparisons — one table per factor

### Factor: `method`

**method effect (cross-run)** — `conll-2003/qwen25-05b/ablations/decoding/no-grammar/run_20261009_133123/eval_baseline.json` (2026-10-09, few-shot, unknown) → `conll-2003/qwen25-05b/sft-grpo/zero-shot/run_20261009_113122/eval_final__few-shot.json` (2026-10-09, few-shot, unknown)

| Metric | A | B | Δ (B−A) | Interpretation |
|---|---|---|---|---|
| ROUGE-L | 0.3700 | 0.8159 | +0.4458 | above noise threshold |
| Valid ROUGE-L | 0.0635 | 0.8139 | +0.7504 | above noise threshold |
| Exact Match | 0.0344 | 0.6249 | +0.5905 | above noise threshold |
| Non-copy Tok Acc | 0.1922 | 0.7730 | +0.5809 | above noise threshold |
| Pass@1 | 0.5440 | 0.9050 | +0.3610 | above noise threshold |
| Validity | 0.1716 | 0.9976 | +0.8260 | above noise threshold |
| BLEU (corpus) | 0.0961 | 0.6651 | +0.5690 | above noise threshold |
| BLEU (sent) | 0.1556 | 0.7605 | +0.6050 | above noise threshold |
| chrF2 (corpus) | 33.2195 | 78.3079 | +45.0884 | scale 0-100: 0.02 noise threshold not applicable |
| chrF2 (sent) | 35.9071 | 82.0286 | +46.1215 | scale 0-100: 0.02 noise threshold not applicable |
| Gloss F1 (mic) | 0.0839 | 0.7162 | +0.6323 | above noise threshold |
| Gloss F1 (sent) | 0.1436 | 0.7615 | +0.6179 | above noise threshold |
- ⚠️ factors not deducible on at least one side, equality NOT verified: reward_stack, rl_objective
- ⚠️ prompt_context_fingerprint DIFFERS: the eval contexts (prompting/grammar/num_samples/model) are not identical — this pair may differ in more than the method factor

**method effect (cross-run)** — `conll-2003/qwen25-05b/grpo/few-shot/run_20261009_052104/eval_baseline.json` (2026-10-09, few-shot, unknown) → `conll-2003/qwen25-05b/sft-grpo/few-shot/run_20261009_090130/eval_final.json` (2026-10-09, few-shot, unknown)

| Metric | A | B | Δ (B−A) | Interpretation |
|---|---|---|---|---|
| ROUGE-L | 0.3805 | 0.8859 | +0.5054 | above noise threshold |
| Valid ROUGE-L | 0.3706 | 0.8855 | +0.5149 | above noise threshold |
| Exact Match | 0.0662 | 0.7289 | +0.6627 | above noise threshold |
| Non-copy Tok Acc | 0.3169 | 0.8277 | +0.5109 | above noise threshold |
| Pass@1 | 0.5865 | 0.9560 | +0.3695 | above noise threshold |
| Validity | 0.9739 | 0.9995 | +0.0256 | above noise threshold |
| BLEU (corpus) | 0.1429 | 0.8230 | +0.6801 | above noise threshold |
| BLEU (sent) | 0.2204 | 0.8424 | +0.6220 | above noise threshold |
| chrF2 (corpus) | 39.8261 | 85.4522 | +45.6261 | scale 0-100: 0.02 noise threshold not applicable |
| chrF2 (sent) | 41.4512 | 88.1871 | +46.7359 | scale 0-100: 0.02 noise threshold not applicable |
| Gloss F1 (mic) | 0.1714 | 0.8204 | +0.6490 | above noise threshold |
| Gloss F1 (sent) | 0.2366 | 0.8310 | +0.5944 | above noise threshold |
- ⚠️ factors not deducible on at least one side, equality NOT verified: reward_stack, rl_objective

**method effect (cross-run)** — `conll-2003/qwen25-05b/ablations/decoding/no-grammar-zero-shot/run_20261009_162603/eval_baseline.json` (2026-10-09, zero-shot, unknown) → `conll-2003/qwen25-05b/sft-grpo/zero-shot/run_20261009_113122/eval_final.json` (2026-10-09, zero-shot, unknown)

| Metric | A | B | Δ (B−A) | Interpretation |
|---|---|---|---|---|
| ROUGE-L | 0.2676 | 0.9128 | +0.6452 | above noise threshold |
| Valid ROUGE-L | 0.0021 | 0.9127 | +0.9106 | above noise threshold |
| Exact Match | 0.0011 | 0.7807 | +0.7796 | above noise threshold |
| Non-copy Tok Acc | 0.0066 | 0.8606 | +0.8541 | above noise threshold |
| Pass@1 | 0.4260 | 0.9660 | +0.5400 | above noise threshold |
| Validity | 0.0079 | 0.9999 | +0.9920 | above noise threshold |
| BLEU (corpus) | 0.0089 | 0.8494 | +0.8405 | above noise threshold |
| BLEU (sent) | 0.0219 | 0.8778 | +0.8560 | above noise threshold |
| chrF2 (corpus) | 17.9933 | 88.0997 | +70.1064 | scale 0-100: 0.02 noise threshold not applicable |
| chrF2 (sent) | 18.2289 | 90.8031 | +72.5741 | scale 0-100: 0.02 noise threshold not applicable |
| Gloss F1 (mic) | 0.0022 | 0.8531 | +0.8509 | above noise threshold |
| Gloss F1 (sent) | 0.0056 | 0.8705 | +0.8650 | above noise threshold |
- ⚠️ factors not deducible on at least one side, equality NOT verified: reward_stack, rl_objective
- ⚠️ prompt_context_fingerprint DIFFERS: the eval contexts (prompting/grammar/num_samples/model) are not identical — this pair may differ in more than the method factor

**method effect (cross-run)** — `conll-2003/qwen25-05b/grpo/few-shot/run_20261009_052104/eval_baseline__zero-shot.json` (2026-10-09, zero-shot, unknown) → `conll-2003/qwen25-05b/sft-grpo/few-shot/run_20261009_090130/eval_final__zero-shot.json` (2026-10-09, zero-shot, unknown)

| Metric | A | B | Δ (B−A) | Interpretation |
|---|---|---|---|---|
| ROUGE-L | 0.2708 | 0.8993 | +0.6284 | above noise threshold |
| Valid ROUGE-L | 0.2625 | 0.8993 | +0.6367 | above noise threshold |
| Exact Match | 0.0453 | 0.7498 | +0.7045 | above noise threshold |
| Non-copy Tok Acc | 0.1458 | 0.8340 | +0.6882 | above noise threshold |
| Pass@1 | 0.4145 | 0.9625 | +0.5480 | above noise threshold |
| Validity | 0.9693 | 1.0000 | +0.0307 | above noise threshold |
| BLEU (corpus) | 0.0893 | 0.8571 | +0.7678 | above noise threshold |
| BLEU (sent) | 0.1329 | 0.8608 | +0.7279 | above noise threshold |
| chrF2 (corpus) | 29.5092 | 86.6384 | +57.1292 | scale 0-100: 0.02 noise threshold not applicable |
| chrF2 (sent) | 29.5764 | 89.2491 | +59.6727 | scale 0-100: 0.02 noise threshold not applicable |
| Gloss F1 (mic) | 0.0762 | 0.8372 | +0.7610 | above noise threshold |
| Gloss F1 (sent) | 0.1033 | 0.8442 | +0.7409 | above noise threshold |
- ⚠️ factors not deducible on at least one side, equality NOT verified: reward_stack, rl_objective

### Factor: `prompting`

🔴 **prompt-dependence (dual eval, same checkpoint)** — `conll-2003/qwen25-05b/grpo/few-shot/run_20261009_052104/eval_baseline.json` (2026-10-09, few-shot, unknown) → `conll-2003/qwen25-05b/grpo/few-shot/run_20261009_052104/eval_baseline__zero-shot.json` (2026-10-09, zero-shot, unknown)

| Metric | A | B | Δ (B−A) | Interpretation |
|---|---|---|---|---|
| ROUGE-L | 0.3805 | 0.2708 | -0.1097 | above noise threshold |
| Valid ROUGE-L | 0.3706 | 0.2625 | -0.1081 | above noise threshold |
| Exact Match | 0.0662 | 0.0453 | -0.0209 | above noise threshold |
| Non-copy Tok Acc | 0.3169 | 0.1458 | -0.1711 | above noise threshold |
| Pass@1 | 0.5865 | 0.4145 | -0.1720 | above noise threshold |
| Validity | 0.9739 | 0.9693 | -0.0046 | NOT GENERALIZABLE (|delta| < 0.02): reproducible (training is bit-deterministic at fixed seed) but single-seed, so it may invert under another seed — do NOT interpret |
| BLEU (corpus) | 0.1429 | 0.0893 | -0.0536 | above noise threshold |
| BLEU (sent) | 0.2204 | 0.1329 | -0.0875 | above noise threshold |
| chrF2 (corpus) | 39.8261 | 29.5092 | -10.3168 | scale 0-100: 0.02 noise threshold not applicable |
| chrF2 (sent) | 41.4512 | 29.5764 | -11.8749 | scale 0-100: 0.02 noise threshold not applicable |
| Gloss F1 (mic) | 0.1714 | 0.0762 | -0.0952 | above noise threshold |
| Gloss F1 (sent) | 0.2366 | 0.1033 | -0.1333 | above noise threshold |
- ⚠️ HIGHLIGHT: same model evaluated in both prompting modes — how much does it depend on the prompt as a crutch?
- ⚠️ factors not deducible on at least one side, equality NOT verified: reward_stack, rl_objective

🔴 **prompt-dependence (dual eval, same checkpoint)** — `conll-2003/qwen25-05b/sft/zero-shot/run_20261009_004112/eval_baseline__few-shot.json` (2026-10-09, few-shot, unknown) → `conll-2003/qwen25-05b/sft/zero-shot/run_20261009_004112/eval_baseline.json` (2026-10-09, zero-shot, unknown)

| Metric | A | B | Δ (B−A) | Interpretation |
|---|---|---|---|---|
| ROUGE-L | 0.3805 | 0.2708 | -0.1097 | above noise threshold |
| Valid ROUGE-L | 0.3706 | 0.2625 | -0.1081 | above noise threshold |
| Exact Match | 0.0662 | 0.0453 | -0.0209 | above noise threshold |
| Non-copy Tok Acc | 0.3169 | 0.1458 | -0.1711 | above noise threshold |
| Pass@1 | 0.5865 | 0.4145 | -0.1720 | above noise threshold |
| Validity | 0.9739 | 0.9693 | -0.0046 | NOT GENERALIZABLE (|delta| < 0.02): reproducible (training is bit-deterministic at fixed seed) but single-seed, so it may invert under another seed — do NOT interpret |
| BLEU (corpus) | 0.1429 | 0.0893 | -0.0536 | above noise threshold |
| BLEU (sent) | 0.2204 | 0.1329 | -0.0875 | above noise threshold |
| chrF2 (corpus) | 39.8261 | 29.5092 | -10.3168 | scale 0-100: 0.02 noise threshold not applicable |
| chrF2 (sent) | 41.4512 | 29.5764 | -11.8749 | scale 0-100: 0.02 noise threshold not applicable |
| Gloss F1 (mic) | 0.1714 | 0.0762 | -0.0952 | above noise threshold |
| Gloss F1 (sent) | 0.2366 | 0.1033 | -0.1333 | above noise threshold |
- ⚠️ HIGHLIGHT: same model evaluated in both prompting modes — how much does it depend on the prompt as a crutch?
- ⚠️ factors not deducible on at least one side, equality NOT verified: reward_stack, rl_objective

**prompting (cross-run)** — `conll-2003/qwen25-05b/ablations/decoding/no-grammar/run_20261009_133123/eval_baseline.json` (2026-10-09, few-shot, unknown) → `conll-2003/qwen25-05b/ablations/decoding/no-grammar-zero-shot/run_20261009_162603/eval_baseline.json` (2026-10-09, zero-shot, unknown)

| Metric | A | B | Δ (B−A) | Interpretation |
|---|---|---|---|---|
| ROUGE-L | 0.3700 | 0.2676 | -0.1024 | above noise threshold |
| Valid ROUGE-L | 0.0635 | 0.0021 | -0.0614 | above noise threshold |
| Exact Match | 0.0344 | 0.0011 | -0.0333 | above noise threshold |
| Non-copy Tok Acc | 0.1922 | 0.0066 | -0.1856 | above noise threshold |
| Pass@1 | 0.5440 | 0.4260 | -0.1180 | above noise threshold |
| Validity | 0.1716 | 0.0079 | -0.1637 | above noise threshold |
| BLEU (corpus) | 0.0961 | 0.0089 | -0.0872 | above noise threshold |
| BLEU (sent) | 0.1556 | 0.0219 | -0.1337 | above noise threshold |
| chrF2 (corpus) | 33.2195 | 17.9933 | -15.2262 | scale 0-100: 0.02 noise threshold not applicable |
| chrF2 (sent) | 35.9071 | 18.2289 | -17.6781 | scale 0-100: 0.02 noise threshold not applicable |
| Gloss F1 (mic) | 0.0839 | 0.0022 | -0.0818 | above noise threshold |
| Gloss F1 (sent) | 0.1436 | 0.0056 | -0.1380 | above noise threshold |
- ⚠️ cross-run comparison: checkpoint identity NOT guaranteed
- ⚠️ factors not deducible on at least one side, equality NOT verified: reward_stack, rl_objective

🔴 **prompt-dependence (dual eval, same checkpoint)** — `conll-2003/qwen25-05b/sft-grpo/zero-shot/run_20261009_113122/eval_final__few-shot.json` (2026-10-09, few-shot, unknown) → `conll-2003/qwen25-05b/sft-grpo/zero-shot/run_20261009_113122/eval_final.json` (2026-10-09, zero-shot, unknown)

| Metric | A | B | Δ (B−A) | Interpretation |
|---|---|---|---|---|
| ROUGE-L | 0.8159 | 0.9128 | +0.0970 | above noise threshold |
| Valid ROUGE-L | 0.8139 | 0.9127 | +0.0988 | above noise threshold |
| Exact Match | 0.6249 | 0.7807 | +0.1558 | above noise threshold |
| Non-copy Tok Acc | 0.7730 | 0.8606 | +0.0876 | above noise threshold |
| Pass@1 | 0.9050 | 0.9660 | +0.0610 | above noise threshold |
| Validity | 0.9976 | 0.9999 | +0.0023 | NOT GENERALIZABLE (|delta| < 0.02): reproducible (training is bit-deterministic at fixed seed) but single-seed, so it may invert under another seed — do NOT interpret |
| BLEU (corpus) | 0.6651 | 0.8494 | +0.1844 | above noise threshold |
| BLEU (sent) | 0.7605 | 0.8778 | +0.1173 | above noise threshold |
| chrF2 (corpus) | 78.3079 | 88.0997 | +9.7918 | scale 0-100: 0.02 noise threshold not applicable |
| chrF2 (sent) | 82.0286 | 90.8031 | +8.7745 | scale 0-100: 0.02 noise threshold not applicable |
| Gloss F1 (mic) | 0.7162 | 0.8531 | +0.1369 | above noise threshold |
| Gloss F1 (sent) | 0.7615 | 0.8705 | +0.1091 | above noise threshold |
- ⚠️ HIGHLIGHT: same model evaluated in both prompting modes — how much does it depend on the prompt as a crutch?
- ⚠️ factors not deducible on at least one side, equality NOT verified: reward_stack, rl_objective

🔴 **prompt-dependence (dual eval, same checkpoint)** — `conll-2003/qwen25-05b/sft-grpo/few-shot/run_20261009_090130/eval_final.json` (2026-10-09, few-shot, unknown) → `conll-2003/qwen25-05b/sft-grpo/few-shot/run_20261009_090130/eval_final__zero-shot.json` (2026-10-09, zero-shot, unknown)

| Metric | A | B | Δ (B−A) | Interpretation |
|---|---|---|---|---|
| ROUGE-L | 0.8859 | 0.8993 | +0.0134 | NOT GENERALIZABLE (|delta| < 0.02): reproducible (training is bit-deterministic at fixed seed) but single-seed, so it may invert under another seed — do NOT interpret |
| Valid ROUGE-L | 0.8855 | 0.8993 | +0.0138 | NOT GENERALIZABLE (|delta| < 0.02): reproducible (training is bit-deterministic at fixed seed) but single-seed, so it may invert under another seed — do NOT interpret |
| Exact Match | 0.7289 | 0.7498 | +0.0209 | above noise threshold |
| Non-copy Tok Acc | 0.8277 | 0.8340 | +0.0062 | NOT GENERALIZABLE (|delta| < 0.02): reproducible (training is bit-deterministic at fixed seed) but single-seed, so it may invert under another seed — do NOT interpret |
| Pass@1 | 0.9560 | 0.9625 | +0.0065 | NOT GENERALIZABLE (|delta| < 0.02): reproducible (training is bit-deterministic at fixed seed) but single-seed, so it may invert under another seed — do NOT interpret |
| Validity | 0.9995 | 1.0000 | +0.0005 | NOT GENERALIZABLE (|delta| < 0.02): reproducible (training is bit-deterministic at fixed seed) but single-seed, so it may invert under another seed — do NOT interpret |
| BLEU (corpus) | 0.8230 | 0.8571 | +0.0341 | above noise threshold |
| BLEU (sent) | 0.8424 | 0.8608 | +0.0184 | NOT GENERALIZABLE (|delta| < 0.02): reproducible (training is bit-deterministic at fixed seed) but single-seed, so it may invert under another seed — do NOT interpret |
| chrF2 (corpus) | 85.4522 | 86.6384 | +1.1863 | scale 0-100: 0.02 noise threshold not applicable |
| chrF2 (sent) | 88.1871 | 89.2491 | +1.0620 | scale 0-100: 0.02 noise threshold not applicable |
| Gloss F1 (mic) | 0.8204 | 0.8372 | +0.0167 | NOT GENERALIZABLE (|delta| < 0.02): reproducible (training is bit-deterministic at fixed seed) but single-seed, so it may invert under another seed — do NOT interpret |
| Gloss F1 (sent) | 0.8310 | 0.8442 | +0.0133 | NOT GENERALIZABLE (|delta| < 0.02): reproducible (training is bit-deterministic at fixed seed) but single-seed, so it may invert under another seed — do NOT interpret |
- ⚠️ HIGHLIGHT: same model evaluated in both prompting modes — how much does it depend on the prompt as a crutch?
- ⚠️ factors not deducible on at least one side, equality NOT verified: reward_stack, rl_objective

🔴 **prompt-dependence (dual eval, same checkpoint)** — `conll-2003/qwen25-05b/grpo/few-shot/run_20261009_052104/eval_final.json` (2026-10-09, few-shot, unknown) → `conll-2003/qwen25-05b/grpo/few-shot/run_20261009_052104/eval_final__zero-shot.json` (2026-10-09, zero-shot, unknown)

| Metric | A | B | Δ (B−A) | Interpretation |
|---|---|---|---|---|
| ROUGE-L | 0.5483 | 0.3115 | -0.2368 | above noise threshold |
| Valid ROUGE-L | 0.5472 | 0.3092 | -0.2380 | above noise threshold |
| Exact Match | 0.3315 | 0.0741 | -0.2574 | above noise threshold |
| Non-copy Tok Acc | 0.4146 | 0.2255 | -0.1891 | above noise threshold |
| Pass@1 | 0.6895 | 0.4805 | -0.2090 | above noise threshold |
| Validity | 0.9979 | 0.9925 | -0.0054 | NOT GENERALIZABLE (|delta| < 0.02): reproducible (training is bit-deterministic at fixed seed) but single-seed, so it may invert under another seed — do NOT interpret |
| BLEU (corpus) | 0.3885 | 0.1276 | -0.2609 | above noise threshold |
| BLEU (sent) | 0.4583 | 0.1705 | -0.2878 | above noise threshold |
| chrF2 (corpus) | 48.0140 | 35.6965 | -12.3175 | scale 0-100: 0.02 noise threshold not applicable |
| chrF2 (sent) | 54.7296 | 35.3721 | -19.3575 | scale 0-100: 0.02 noise threshold not applicable |
| Gloss F1 (mic) | 0.3859 | 0.1341 | -0.2518 | above noise threshold |
| Gloss F1 (sent) | 0.4566 | 0.1649 | -0.2917 | above noise threshold |
- ⚠️ HIGHLIGHT: same model evaluated in both prompting modes — how much does it depend on the prompt as a crutch?
- ⚠️ factors not deducible on at least one side, equality NOT verified: reward_stack, rl_objective

🔴 **prompt-dependence (dual eval, same checkpoint)** — `conll-2003/qwen25-05b/grpo/zero-shot/run_20261009_020134/eval_final__few-shot.json` (2026-10-09, few-shot, unknown) → `conll-2003/qwen25-05b/grpo/zero-shot/run_20261009_020134/eval_final.json` (2026-10-09, zero-shot, unknown)

| Metric | A | B | Δ (B−A) | Interpretation |
|---|---|---|---|---|
| ROUGE-L | 0.4025 | 0.3526 | -0.0499 | above noise threshold |
| Valid ROUGE-L | 0.3995 | 0.3519 | -0.0476 | above noise threshold |
| Exact Match | 0.0448 | 0.0228 | -0.0220 | above noise threshold |
| Non-copy Tok Acc | 0.4452 | 0.3262 | -0.1190 | above noise threshold |
| Pass@1 | 0.6695 | 0.6200 | -0.0495 | above noise threshold |
| Validity | 0.9924 | 0.9979 | +0.0055 | NOT GENERALIZABLE (|delta| < 0.02): reproducible (training is bit-deterministic at fixed seed) but single-seed, so it may invert under another seed — do NOT interpret |
| BLEU (corpus) | 0.2097 | 0.1964 | -0.0133 | NOT GENERALIZABLE (|delta| < 0.02): reproducible (training is bit-deterministic at fixed seed) but single-seed, so it may invert under another seed — do NOT interpret |
| BLEU (sent) | 0.2279 | 0.1790 | -0.0489 | above noise threshold |
| chrF2 (corpus) | 49.8149 | 44.0094 | -5.8055 | scale 0-100: 0.02 noise threshold not applicable |
| chrF2 (sent) | 46.0096 | 40.0840 | -5.9256 | scale 0-100: 0.02 noise threshold not applicable |
| Gloss F1 (mic) | 0.2653 | 0.2129 | -0.0525 | above noise threshold |
| Gloss F1 (sent) | 0.2864 | 0.2163 | -0.0701 | above noise threshold |
- ⚠️ HIGHLIGHT: same model evaluated in both prompting modes — how much does it depend on the prompt as a crutch?
- ⚠️ factors not deducible on at least one side, equality NOT verified: reward_stack, rl_objective

🔴 **prompt-dependence (dual eval, same checkpoint)** — `conll-2003/qwen25-05b/sft/zero-shot/run_20261009_004112/eval_final__few-shot.json` (2026-10-09, few-shot, unknown) → `conll-2003/qwen25-05b/sft/zero-shot/run_20261009_004112/eval_final.json` (2026-10-09, zero-shot, unknown)

| Metric | A | B | Δ (B−A) | Interpretation |
|---|---|---|---|---|
| ROUGE-L | 0.8138 | 0.9003 | +0.0864 | above noise threshold |
| Valid ROUGE-L | 0.8116 | 0.9001 | +0.0885 | above noise threshold |
| Exact Match | 0.5975 | 0.7432 | +0.1457 | above noise threshold |
| Non-copy Tok Acc | 0.7356 | 0.8268 | +0.0913 | above noise threshold |
| Pass@1 | 0.9210 | 0.9685 | +0.0475 | above noise threshold |
| Validity | 0.9973 | 0.9998 | +0.0025 | NOT GENERALIZABLE (|delta| < 0.02): reproducible (training is bit-deterministic at fixed seed) but single-seed, so it may invert under another seed — do NOT interpret |
| BLEU (corpus) | 0.6567 | 0.8330 | +0.1763 | above noise threshold |
| BLEU (sent) | 0.7476 | 0.8600 | +0.1124 | above noise threshold |
| chrF2 (corpus) | 76.0076 | 85.9758 | +9.9682 | scale 0-100: 0.02 noise threshold not applicable |
| chrF2 (sent) | 81.2920 | 89.4084 | +8.1164 | scale 0-100: 0.02 noise threshold not applicable |
| Gloss F1 (mic) | 0.6911 | 0.8205 | +0.1295 | above noise threshold |
| Gloss F1 (sent) | 0.7445 | 0.8421 | +0.0976 | above noise threshold |
- ⚠️ HIGHLIGHT: same model evaluated in both prompting modes — how much does it depend on the prompt as a crutch?
- ⚠️ factors not deducible on at least one side, equality NOT verified: reward_stack, rl_objective

**prompting (cross-run)** — `conll-2003/qwen25-05b/ablations/decoding/no-grammar/run_20261009_133123/eval_final.json` (2026-10-09, few-shot, unknown) → `conll-2003/qwen25-05b/ablations/decoding/no-grammar-zero-shot/run_20261009_162603/eval_final.json` (2026-10-09, zero-shot, unknown)

| Metric | A | B | Δ (B−A) | Interpretation |
|---|---|---|---|---|
| ROUGE-L | 0.5906 | 0.4548 | -0.1358 | above noise threshold |
| Valid ROUGE-L | 0.4289 | 0.0650 | -0.3638 | above noise threshold |
| Exact Match | 0.3473 | 0.0611 | -0.2862 | above noise threshold |
| Non-copy Tok Acc | 0.4019 | 0.1084 | -0.2935 | above noise threshold |
| Pass@1 | 0.7305 | 0.7405 | +0.0100 | NOT GENERALIZABLE (|delta| < 0.02): reproducible (training is bit-deterministic at fixed seed) but single-seed, so it may invert under another seed — do NOT interpret |
| Validity | 0.7261 | 0.1430 | -0.5831 | above noise threshold |
| BLEU (corpus) | 0.4205 | 0.0142 | -0.4063 | above noise threshold |
| BLEU (sent) | 0.4740 | 0.1056 | -0.3684 | above noise threshold |
| chrF2 (corpus) | 48.2561 | 26.7111 | -21.5450 | scale 0-100: 0.02 noise threshold not applicable |
| chrF2 (sent) | 56.4984 | 34.0781 | -22.4203 | scale 0-100: 0.02 noise threshold not applicable |
| Gloss F1 (mic) | 0.3907 | 0.0530 | -0.3377 | above noise threshold |
| Gloss F1 (sent) | 0.4673 | 0.1111 | -0.3561 | above noise threshold |
- ⚠️ cross-run comparison: checkpoint identity NOT guaranteed
- ⚠️ factors not deducible on at least one side, equality NOT verified: reward_stack, rl_objective

**prompting (cross-run)** — `conll-2003/qwen25-05b/ablations/objectives/sft-allowed-mass/run_20261009_190603/eval_final.json` (2026-10-09, zero-shot, unknown) → `conll-2003/qwen25-05b/baseline/few-shot/run_20261008_235550/eval_zero_shot.json` (2026-10-08, few-shot, unknown)

| Metric | A | B | Δ (B−A) | Interpretation |
|---|---|---|---|---|
| ROUGE-L | 0.9039 | 0.3805 | -0.5234 | above noise threshold |
| Valid ROUGE-L | 0.9039 | 0.3706 | -0.5334 | above noise threshold |
| Exact Match | 0.7469 | 0.0662 | -0.6807 | above noise threshold |
| Non-copy Tok Acc | 0.8274 | 0.3169 | -0.5105 | above noise threshold |
| Pass@1 | 0.9695 | 0.5865 | -0.3830 | above noise threshold |
| Validity | 1.0000 | 0.9739 | -0.0261 | above noise threshold |
| BLEU (corpus) | 0.8359 | 0.1429 | -0.6930 | above noise threshold |
| BLEU (sent) | 0.8632 | 0.2204 | -0.6428 | above noise threshold |
| chrF2 (corpus) | 85.7456 | 39.8261 | -45.9195 | scale 0-100: 0.02 noise threshold not applicable |
| chrF2 (sent) | 89.6914 | 41.4512 | -48.2401 | scale 0-100: 0.02 noise threshold not applicable |
| Gloss F1 (mic) | 0.8252 | 0.1714 | -0.6538 | above noise threshold |
| Gloss F1 (sent) | 0.8500 | 0.2366 | -0.6134 | above noise threshold |
- ⚠️ cross-run comparison: checkpoint identity NOT guaranteed
- ⚠️ factors not deducible on at least one side, equality NOT verified: reward_stack, rl_objective

### Factor: `grammar`

**No paired comparison available.** 6 group(s) with 2+ runs but no pair with a known, differing value of this factor


## Missing paired comparisons (declared, not fabricated)

- **`grammar`**: 6 group(s) with 2+ runs but no pair with a known, differing value of this factor.
  Available typologies: `grammar=unknown`

## Figures

- `campaign_pairwise_deltas.png` — one panel per factor: paired deltas as diverging bars with the ±0.02 noise band shaded. Diverging bars make the SIGN of each effect immediately visible, and the shaded band makes it impossible to mistake sub-noise deltas for results.
- `campaign_matrix.png` — the ablation matrix as a heatmap (rows: method/variant, columns: prompting, annotated with ROUGE-L); empty typologies stay visibly grey so gaps in the grid are impossible to miss.

