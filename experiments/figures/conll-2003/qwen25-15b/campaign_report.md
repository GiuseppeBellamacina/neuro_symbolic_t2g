# Campaign Report — conll-2003 / qwen25-15b

Generated: 2026-11-12T13:18:04+00:00

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

## Selected runs (11 typologies, latest per typology)

| Run | method | variant | prompting | grammar | Date | Prompts | metrics_version |
|---|---|---|---|---|---|---|---|
| `conll-2003/qwen25-15b/baseline/zero-shot/run_20261103_165307/eval_zero_shot.json` | unknown | conll-2003-qwen25-15b-baseline-zero-shot | zero-shot | unknown | 2026-11-03 | 2000 | 2 |
| `conll-2003/qwen25-15b/grpo/zero-shot/run_20261103_165804/eval_baseline.json` | baseline | conll-2003-qwen25-15b-grpo-zero-shot | zero-shot | unknown | 2026-11-03 | 2000 | 2 |
| `conll-2003/qwen25-15b/grpo/zero-shot/run_20261103_165804/eval_baseline__few-shot.json` | baseline | conll-2003-qwen25-15b-grpo-zero-shot | few-shot | unknown | 2026-11-03 | 2000 | 2 |
| `conll-2003/qwen25-15b/grpo/zero-shot/run_20261103_165804/eval_final.json` | unknown | conll-2003-qwen25-15b-grpo-zero-shot | zero-shot | unknown | 2026-11-03 | 2000 | 2 |
| `conll-2003/qwen25-15b/grpo/zero-shot/run_20261103_165804/eval_final__few-shot.json` | unknown | conll-2003-qwen25-15b-grpo-zero-shot | few-shot | unknown | 2026-11-03 | 2000 | 2 |
| `conll-2003/qwen25-15b/seeds/sft-zero-shot-s1/run_20261110_004018/eval_final.json` | unknown | conll-2003-qwen25-15b-seeds-sft-zero-shot-s1 | zero-shot | unknown | 2026-11-10 | 2000 | 2 |
| `conll-2003/qwen25-15b/seeds/sft-zero-shot-s1/run_20261110_004018/eval_final__few-shot.json` | unknown | conll-2003-qwen25-15b-seeds-sft-zero-shot-s1 | few-shot | unknown | 2026-11-10 | 2000 | 2 |
| `conll-2003/qwen25-15b/sft-grpo/zero-shot/run_20261106_234209/eval_final.json` | sft-grpo | zero-shot | zero-shot | unknown | 2026-11-06 | 2000 | 2 |
| `conll-2003/qwen25-15b/sft-grpo/zero-shot/run_20261106_234209/eval_final__few-shot.json` | sft-grpo | zero-shot | few-shot | unknown | 2026-11-06 | 2000 | 2 |
| `conll-2003/qwen25-15b/sft/zero-shot/run_20261106_162828/eval_final.json` | unknown | conll-2003-qwen25-15b-sft-zero-shot | zero-shot | unknown | 2026-11-06 | 2000 | 2 |
| `conll-2003/qwen25-15b/sft/zero-shot/run_20261106_162828/eval_final__few-shot.json` | unknown | conll-2003-qwen25-15b-sft-zero-shot | few-shot | unknown | 2026-11-06 | 2000 | 2 |

## Ablation matrix overview — ROUGE-L

One row per cell (its path under `results/`), one column per prompting mode that has at least one value. `—` = the cell was not evaluated in that mode. `*` = multiple runs collapsed, latest shown. Each cell's own `eval_baseline` (the base model in that cell's context) is not a row here: it repeats the `baseline/` cells and stays in the paired comparisons below.

| cell | zero-shot | few-shot |
|---|---|---|
| conll-2003/qwen25-15b/baseline/zero-shot | 0.2224 | — |
| conll-2003/qwen25-15b/sft/zero-shot | 0.9137 | 0.8383 |
| conll-2003/qwen25-15b/sft-grpo/zero-shot | 0.9290 | 0.8552 |
| conll-2003/qwen25-15b/grpo/zero-shot | 0.4291 | 0.6109 |
| conll-2003/qwen25-15b/seeds/sft-zero-shot-s1 | 0.9146 | 0.8462 |

## conll-2003 — all metrics

Same rows as the matrix above, restricted to this dataset, one column per metric. On CoNLL-2003 and WOS-46985 read Exact Match and Gloss F1 (micro = entity-level F1 on CoNLL): ROUGE-L and BLEU split `PER:Mary` into `per` + `mary` and give partial credit to a wrong label of the same type.

| cell | prompting | ROUGE-L | Valid ROUGE-L | Exact Match | Non-copy Tok Acc | Pass@1 | Validity | BLEU (corpus) | BLEU (sent) | chrF2 (corpus) | chrF2 (sent) | Gloss F1 (mic) | Gloss F1 (sent) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| conll-2003/qwen25-15b/baseline/zero-shot | zero-shot | 0.2224 | 0.2224 | 0.2049 | 0.1150 | 0.2295 | 1.0000 | 0.0039 | 0.2124 | 5.9812 | 23.2225 | 0.1483 | 0.2071 |
| conll-2003/qwen25-15b/sft/zero-shot | zero-shot | 0.9137 | 0.9137 | 0.7782 | 0.8467 | 0.9700 | 1.0000 | 0.8609 | 0.8789 | 87.0673 | 90.4472 | 0.8558 | 0.8689 |
| conll-2003/qwen25-15b/sft/zero-shot | few-shot | 0.8383 | 0.8383 | 0.6852 | 0.7820 | 0.9080 | 1.0000 | 0.7985 | 0.7961 | 81.3769 | 83.0637 | 0.7952 | 0.7931 |
| conll-2003/qwen25-15b/sft-grpo/zero-shot | zero-shot | 0.9290 | 0.9290 | 0.8137 | 0.8894 | 0.9740 | 1.0000 | 0.8805 | 0.8979 | 90.8492 | 92.4884 | 0.8853 | 0.8923 |
| conll-2003/qwen25-15b/sft-grpo/zero-shot | few-shot | 0.8552 | 0.8552 | 0.7155 | 0.8279 | 0.9135 | 1.0000 | 0.8432 | 0.8208 | 85.4075 | 85.3832 | 0.8280 | 0.8188 |
| conll-2003/qwen25-15b/grpo/zero-shot | zero-shot | 0.4291 | 0.4291 | 0.2797 | 0.2853 | 0.4965 | 1.0000 | 0.2585 | 0.3690 | 33.7953 | 41.6493 | 0.3152 | 0.3519 |
| conll-2003/qwen25-15b/grpo/zero-shot | few-shot | 0.6109 | 0.6109 | 0.3886 | 0.4936 | 0.7285 | 0.9999 | 0.4511 | 0.5269 | 57.1586 | 60.6821 | 0.4444 | 0.5124 |
| conll-2003/qwen25-15b/seeds/sft-zero-shot-s1 | zero-shot | 0.9146 | 0.9146 | 0.7763 | 0.8475 | 0.9720 | 1.0000 | 0.8595 | 0.8792 | 86.9419 | 90.5082 | 0.8572 | 0.8707 |
| conll-2003/qwen25-15b/seeds/sft-zero-shot-s1 | few-shot | 0.8462 | 0.8462 | 0.6904 | 0.7874 | 0.9160 | 1.0000 | 0.8037 | 0.8039 | 81.8302 | 83.8626 | 0.7997 | 0.7997 |

## Paired comparisons — one table per factor

### Factor: `method`

**method effect (cross-run)** — `conll-2003/qwen25-15b/grpo/zero-shot/run_20261103_165804/eval_baseline__few-shot.json` (2026-11-03, few-shot, unknown) → `conll-2003/qwen25-15b/sft-grpo/zero-shot/run_20261106_234209/eval_final__few-shot.json` (2026-11-06, few-shot, unknown)

| Metric | A | B | Δ (B−A) | Interpretation |
|---|---|---|---|---|
| ROUGE-L | 0.3992 | 0.8552 | +0.4560 | above noise threshold |
| Valid ROUGE-L | 0.3992 | 0.8552 | +0.4560 | above noise threshold |
| Exact Match | 0.2829 | 0.7155 | +0.4326 | above noise threshold |
| Non-copy Tok Acc | 0.2917 | 0.8279 | +0.5362 | above noise threshold |
| Pass@1 | 0.4640 | 0.9135 | +0.4495 | above noise threshold |
| Validity | 1.0000 | 1.0000 | +0.0000 | NOT GENERALIZABLE (|delta| < 0.02): reproducible (training is bit-deterministic at fixed seed) but single-seed, so it may invert under another seed — do NOT interpret |
| BLEU (corpus) | 0.2547 | 0.8432 | +0.5885 | above noise threshold |
| BLEU (sent) | 0.3532 | 0.8208 | +0.4676 | above noise threshold |
| chrF2 (corpus) | 32.5476 | 85.4075 | +52.8598 | scale 0-100: 0.02 noise threshold not applicable |
| chrF2 (sent) | 39.7802 | 85.3832 | +45.6030 | scale 0-100: 0.02 noise threshold not applicable |
| Gloss F1 (mic) | 0.3194 | 0.8280 | +0.5086 | above noise threshold |
| Gloss F1 (sent) | 0.3507 | 0.8188 | +0.4681 | above noise threshold |
- ⚠️ factors not deducible on at least one side, equality NOT verified: reward_stack, rl_objective

**method effect (cross-run)** — `conll-2003/qwen25-15b/grpo/zero-shot/run_20261103_165804/eval_baseline.json` (2026-11-03, zero-shot, unknown) → `conll-2003/qwen25-15b/sft-grpo/zero-shot/run_20261106_234209/eval_final.json` (2026-11-06, zero-shot, unknown)

| Metric | A | B | Δ (B−A) | Interpretation |
|---|---|---|---|---|
| ROUGE-L | 0.2224 | 0.9290 | +0.7066 | above noise threshold |
| Valid ROUGE-L | 0.2224 | 0.9290 | +0.7066 | above noise threshold |
| Exact Match | 0.2049 | 0.8137 | +0.6088 | above noise threshold |
| Non-copy Tok Acc | 0.1150 | 0.8894 | +0.7744 | above noise threshold |
| Pass@1 | 0.2295 | 0.9740 | +0.7445 | above noise threshold |
| Validity | 1.0000 | 1.0000 | +0.0000 | NOT GENERALIZABLE (|delta| < 0.02): reproducible (training is bit-deterministic at fixed seed) but single-seed, so it may invert under another seed — do NOT interpret |
| BLEU (corpus) | 0.0039 | 0.8805 | +0.8766 | above noise threshold |
| BLEU (sent) | 0.2124 | 0.8979 | +0.6856 | above noise threshold |
| chrF2 (corpus) | 5.9812 | 90.8492 | +84.8680 | scale 0-100: 0.02 noise threshold not applicable |
| chrF2 (sent) | 23.2225 | 92.4884 | +69.2660 | scale 0-100: 0.02 noise threshold not applicable |
| Gloss F1 (mic) | 0.1483 | 0.8853 | +0.7370 | above noise threshold |
| Gloss F1 (sent) | 0.2071 | 0.8923 | +0.6851 | above noise threshold |
- ⚠️ factors not deducible on at least one side, equality NOT verified: reward_stack, rl_objective

### Factor: `prompting`

🔴 **prompt-dependence (dual eval, same checkpoint)** — `conll-2003/qwen25-15b/grpo/zero-shot/run_20261103_165804/eval_baseline__few-shot.json` (2026-11-03, few-shot, unknown) → `conll-2003/qwen25-15b/grpo/zero-shot/run_20261103_165804/eval_baseline.json` (2026-11-03, zero-shot, unknown)

| Metric | A | B | Δ (B−A) | Interpretation |
|---|---|---|---|---|
| ROUGE-L | 0.3992 | 0.2224 | -0.1768 | above noise threshold |
| Valid ROUGE-L | 0.3992 | 0.2224 | -0.1768 | above noise threshold |
| Exact Match | 0.2829 | 0.2049 | -0.0780 | above noise threshold |
| Non-copy Tok Acc | 0.2917 | 0.1150 | -0.1767 | above noise threshold |
| Pass@1 | 0.4640 | 0.2295 | -0.2345 | above noise threshold |
| Validity | 1.0000 | 1.0000 | +0.0000 | NOT GENERALIZABLE (|delta| < 0.02): reproducible (training is bit-deterministic at fixed seed) but single-seed, so it may invert under another seed — do NOT interpret |
| BLEU (corpus) | 0.2547 | 0.0039 | -0.2508 | above noise threshold |
| BLEU (sent) | 0.3532 | 0.2124 | -0.1409 | above noise threshold |
| chrF2 (corpus) | 32.5476 | 5.9812 | -26.5664 | scale 0-100: 0.02 noise threshold not applicable |
| chrF2 (sent) | 39.7802 | 23.2225 | -16.5577 | scale 0-100: 0.02 noise threshold not applicable |
| Gloss F1 (mic) | 0.3194 | 0.1483 | -0.1712 | above noise threshold |
| Gloss F1 (sent) | 0.3507 | 0.2071 | -0.1436 | above noise threshold |
- ⚠️ HIGHLIGHT: same model evaluated in both prompting modes — how much does it depend on the prompt as a crutch?
- ⚠️ factors not deducible on at least one side, equality NOT verified: reward_stack, rl_objective

🔴 **prompt-dependence (dual eval, same checkpoint)** — `conll-2003/qwen25-15b/sft-grpo/zero-shot/run_20261106_234209/eval_final__few-shot.json` (2026-11-06, few-shot, unknown) → `conll-2003/qwen25-15b/sft-grpo/zero-shot/run_20261106_234209/eval_final.json` (2026-11-06, zero-shot, unknown)

| Metric | A | B | Δ (B−A) | Interpretation |
|---|---|---|---|---|
| ROUGE-L | 0.8552 | 0.9290 | +0.0738 | above noise threshold |
| Valid ROUGE-L | 0.8552 | 0.9290 | +0.0738 | above noise threshold |
| Exact Match | 0.7155 | 0.8137 | +0.0982 | above noise threshold |
| Non-copy Tok Acc | 0.8279 | 0.8894 | +0.0615 | above noise threshold |
| Pass@1 | 0.9135 | 0.9740 | +0.0605 | above noise threshold |
| Validity | 1.0000 | 1.0000 | +0.0000 | NOT GENERALIZABLE (|delta| < 0.02): reproducible (training is bit-deterministic at fixed seed) but single-seed, so it may invert under another seed — do NOT interpret |
| BLEU (corpus) | 0.8432 | 0.8805 | +0.0373 | above noise threshold |
| BLEU (sent) | 0.8208 | 0.8979 | +0.0771 | above noise threshold |
| chrF2 (corpus) | 85.4075 | 90.8492 | +5.4417 | scale 0-100: 0.02 noise threshold not applicable |
| chrF2 (sent) | 85.3832 | 92.4884 | +7.1052 | scale 0-100: 0.02 noise threshold not applicable |
| Gloss F1 (mic) | 0.8280 | 0.8853 | +0.0573 | above noise threshold |
| Gloss F1 (sent) | 0.8188 | 0.8923 | +0.0735 | above noise threshold |
- ⚠️ HIGHLIGHT: same model evaluated in both prompting modes — how much does it depend on the prompt as a crutch?
- ⚠️ factors not deducible on at least one side, equality NOT verified: reward_stack, rl_objective

🔴 **prompt-dependence (dual eval, same checkpoint)** — `conll-2003/qwen25-15b/seeds/sft-zero-shot-s1/run_20261110_004018/eval_final__few-shot.json` (2026-11-10, few-shot, unknown) → `conll-2003/qwen25-15b/seeds/sft-zero-shot-s1/run_20261110_004018/eval_final.json` (2026-11-10, zero-shot, unknown)

| Metric | A | B | Δ (B−A) | Interpretation |
|---|---|---|---|---|
| ROUGE-L | 0.8462 | 0.9146 | +0.0684 | above noise threshold |
| Valid ROUGE-L | 0.8462 | 0.9146 | +0.0684 | above noise threshold |
| Exact Match | 0.6904 | 0.7763 | +0.0859 | above noise threshold |
| Non-copy Tok Acc | 0.7874 | 0.8475 | +0.0601 | above noise threshold |
| Pass@1 | 0.9160 | 0.9720 | +0.0560 | above noise threshold |
| Validity | 1.0000 | 1.0000 | +0.0000 | NOT GENERALIZABLE (|delta| < 0.02): reproducible (training is bit-deterministic at fixed seed) but single-seed, so it may invert under another seed — do NOT interpret |
| BLEU (corpus) | 0.8037 | 0.8595 | +0.0558 | above noise threshold |
| BLEU (sent) | 0.8039 | 0.8792 | +0.0753 | above noise threshold |
| chrF2 (corpus) | 81.8302 | 86.9419 | +5.1118 | scale 0-100: 0.02 noise threshold not applicable |
| chrF2 (sent) | 83.8626 | 90.5082 | +6.6455 | scale 0-100: 0.02 noise threshold not applicable |
| Gloss F1 (mic) | 0.7997 | 0.8572 | +0.0575 | above noise threshold |
| Gloss F1 (sent) | 0.7997 | 0.8707 | +0.0710 | above noise threshold |
- ⚠️ HIGHLIGHT: same model evaluated in both prompting modes — how much does it depend on the prompt as a crutch?
- ⚠️ factors not deducible on at least one side, equality NOT verified: reward_stack, rl_objective

🔴 **prompt-dependence (dual eval, same checkpoint)** — `conll-2003/qwen25-15b/sft/zero-shot/run_20261106_162828/eval_final__few-shot.json` (2026-11-06, few-shot, unknown) → `conll-2003/qwen25-15b/sft/zero-shot/run_20261106_162828/eval_final.json` (2026-11-06, zero-shot, unknown)

| Metric | A | B | Δ (B−A) | Interpretation |
|---|---|---|---|---|
| ROUGE-L | 0.8383 | 0.9137 | +0.0754 | above noise threshold |
| Valid ROUGE-L | 0.8383 | 0.9137 | +0.0754 | above noise threshold |
| Exact Match | 0.6852 | 0.7782 | +0.0930 | above noise threshold |
| Non-copy Tok Acc | 0.7820 | 0.8467 | +0.0647 | above noise threshold |
| Pass@1 | 0.9080 | 0.9700 | +0.0620 | above noise threshold |
| Validity | 1.0000 | 1.0000 | +0.0000 | NOT GENERALIZABLE (|delta| < 0.02): reproducible (training is bit-deterministic at fixed seed) but single-seed, so it may invert under another seed — do NOT interpret |
| BLEU (corpus) | 0.7985 | 0.8609 | +0.0624 | above noise threshold |
| BLEU (sent) | 0.7961 | 0.8789 | +0.0828 | above noise threshold |
| chrF2 (corpus) | 81.3769 | 87.0673 | +5.6904 | scale 0-100: 0.02 noise threshold not applicable |
| chrF2 (sent) | 83.0637 | 90.4472 | +7.3835 | scale 0-100: 0.02 noise threshold not applicable |
| Gloss F1 (mic) | 0.7952 | 0.8558 | +0.0607 | above noise threshold |
| Gloss F1 (sent) | 0.7931 | 0.8689 | +0.0758 | above noise threshold |
- ⚠️ HIGHLIGHT: same model evaluated in both prompting modes — how much does it depend on the prompt as a crutch?
- ⚠️ factors not deducible on at least one side, equality NOT verified: reward_stack, rl_objective

🔴 **prompt-dependence (dual eval, same checkpoint)** — `conll-2003/qwen25-15b/grpo/zero-shot/run_20261103_165804/eval_final__few-shot.json` (2026-11-03, few-shot, unknown) → `conll-2003/qwen25-15b/grpo/zero-shot/run_20261103_165804/eval_final.json` (2026-11-03, zero-shot, unknown)

| Metric | A | B | Δ (B−A) | Interpretation |
|---|---|---|---|---|
| ROUGE-L | 0.6109 | 0.4291 | -0.1818 | above noise threshold |
| Valid ROUGE-L | 0.6109 | 0.4291 | -0.1817 | above noise threshold |
| Exact Match | 0.3886 | 0.2797 | -0.1089 | above noise threshold |
| Non-copy Tok Acc | 0.4936 | 0.2853 | -0.2083 | above noise threshold |
| Pass@1 | 0.7285 | 0.4965 | -0.2320 | above noise threshold |
| Validity | 0.9999 | 1.0000 | +0.0001 | NOT GENERALIZABLE (|delta| < 0.02): reproducible (training is bit-deterministic at fixed seed) but single-seed, so it may invert under another seed — do NOT interpret |
| BLEU (corpus) | 0.4511 | 0.2585 | -0.1926 | above noise threshold |
| BLEU (sent) | 0.5269 | 0.3690 | -0.1578 | above noise threshold |
| chrF2 (corpus) | 57.1586 | 33.7953 | -23.3633 | scale 0-100: 0.02 noise threshold not applicable |
| chrF2 (sent) | 60.6821 | 41.6493 | -19.0328 | scale 0-100: 0.02 noise threshold not applicable |
| Gloss F1 (mic) | 0.4444 | 0.3152 | -0.1292 | above noise threshold |
| Gloss F1 (sent) | 0.5124 | 0.3519 | -0.1606 | above noise threshold |
- ⚠️ HIGHLIGHT: same model evaluated in both prompting modes — how much does it depend on the prompt as a crutch?
- ⚠️ factors not deducible on at least one side, equality NOT verified: reward_stack, rl_objective

### Factor: `grammar`

**No paired comparison available.** 2 group(s) with 2+ runs but no pair with a known, differing value of this factor


## Missing paired comparisons (declared, not fabricated)

- **`grammar`**: 2 group(s) with 2+ runs but no pair with a known, differing value of this factor.
  Available typologies: `grammar=unknown`

## Figures

- `campaign_pairwise_deltas.png` — one panel per factor: paired deltas as diverging bars with the ±0.02 noise band shaded. Diverging bars make the SIGN of each effect immediately visible, and the shaded band makes it impossible to mistake sub-noise deltas for results.
- `campaign_matrix.png` — the ablation matrix as a heatmap (rows: method/variant, columns: prompting, annotated with ROUGE-L); empty typologies stay visibly grey so gaps in the grid are impossible to miss.

