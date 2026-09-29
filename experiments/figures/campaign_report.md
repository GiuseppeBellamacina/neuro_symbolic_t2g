# Campaign Report — Cross-Factor Ablation Comparison

Generated: 2026-09-29T14:28:36+00:00

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

## Selected runs (44 typologies, latest per typology)

| Run | method | variant | prompting | grammar | Date | Prompts | metrics_version |
|---|---|---|---|---|---|---|---|
| `qwen25-05b/ablations/decoding/hot-rollout/run_20260915_083119/eval_baseline.json` | baseline | qwen25-05b-ablations-decoding-hot-rollout | few-shot | unknown | 2026-09-15 | 2000 | 2 |
| `qwen25-05b/ablations/decoding/hot-rollout/run_20260915_083119/eval_final.json` | unknown | qwen25-05b-ablations-decoding-hot-rollout | few-shot | unknown | 2026-09-15 | 2000 | 2 |
| `qwen25-05b/ablations/decoding/no-grammar/run_20260915_052556/eval_baseline.json` | baseline | qwen25-05b-ablations-decoding-no-grammar | few-shot | unknown | 2026-09-15 | 2000 | 2 |
| `qwen25-05b/ablations/decoding/no-grammar/run_20260915_052556/eval_final.json` | unknown | qwen25-05b-ablations-decoding-no-grammar | few-shot | unknown | 2026-09-15 | 2000 | 2 |
| `qwen25-05b/ablations/glossary/few-shot/run_20260917_212556/eval_baseline.json` | baseline | qwen25-05b-ablations-glossary-few-shot | few-shot | unknown | 2026-09-17 | 2000 | 2 |
| `qwen25-05b/ablations/glossary/few-shot/run_20260917_212556/eval_final.json` | unknown | qwen25-05b-ablations-glossary-few-shot | few-shot | unknown | 2026-09-17 | 2000 | 2 |
| `qwen25-05b/ablations/glossary/zero-shot/run_20260917_105109/eval_baseline.json` | baseline | qwen25-05b-ablations-glossary-zero-shot | zero-shot | unknown | 2026-09-17 | 2000 | 2 |
| `qwen25-05b/ablations/glossary/zero-shot/run_20260917_105109/eval_final.json` | unknown | qwen25-05b-ablations-glossary-zero-shot | zero-shot | unknown | 2026-09-17 | 2000 | 2 |
| `qwen25-05b/ablations/loss/dr-grpo/run_20260916_183557/eval_baseline.json` | baseline | qwen25-05b-ablations-loss-dr-grpo | few-shot | unknown | 2026-09-16 | 2000 | 2 |
| `qwen25-05b/ablations/loss/dr-grpo/run_20260916_183557/eval_final.json` | unknown | qwen25-05b-ablations-loss-dr-grpo | few-shot | unknown | 2026-09-16 | 2000 | 2 |
| `qwen25-05b/ablations/loss/low-beta/run_20260917_053119/eval_final.json` | unknown | qwen25-05b-ablations-loss-low-beta | few-shot | unknown | 2026-09-17 | 2000 | 2 |
| `qwen25-05b/ablations/objectives/sft-allowed-mass/run_20260920_205638/decoding-greedy/eval_baseline.json` | baseline | qwen25-05b-ablations-objectives-sft-allowed-mass-decoding-greedy | zero-shot | unknown | 2026-09-20 | 2000 | 2 |
| `qwen25-05b/ablations/objectives/sft-allowed-mass/run_20260920_205638/decoding-greedy/eval_final.json` | unknown | qwen25-05b-ablations-objectives-sft-allowed-mass-decoding-greedy | zero-shot | unknown | 2026-09-20 | 2000 | 2 |
| `qwen25-05b/ablations/objectives/sft-allowed-mass/run_20260920_205638/eval_final.json` | unknown | qwen25-05b-ablations-objectives-sft-allowed-mass | zero-shot | unknown | 2026-09-20 | 2000 | 2 |
| `qwen25-05b/ablations/objectives/sft-structured-shuffled/run_20260925_192053/eval_final.json` | unknown | qwen25-05b-ablations-objectives-sft-structured-shuffled | zero-shot | unknown | 2026-09-25 | 2000 | 2 |
| `qwen25-05b/ablations/objectives/sft-structured/run_20260923_101821/eval_final.json` | unknown | qwen25-05b-ablations-objectives-sft-structured | zero-shot | unknown | 2026-09-23 | 2000 | 2 |
| `qwen25-05b/ablations/rewards/edit-validity/run_20260915_160558/eval_baseline.json` | baseline | qwen25-05b-ablations-rewards-edit-validity | few-shot | unknown | 2026-09-15 | 2000 | 2 |
| `qwen25-05b/ablations/rewards/edit-validity/run_20260915_160558/eval_final.json` | unknown | qwen25-05b-ablations-rewards-edit-validity | few-shot | unknown | 2026-09-15 | 2000 | 2 |
| `qwen25-05b/ablations/rewards/lean-stack/run_20260916_132557/eval_final.json` | unknown | qwen25-05b-ablations-rewards-lean-stack | few-shot | unknown | 2026-09-16 | 2000 | 2 |
| `qwen25-05b/baseline/few-shot/run_20260912_171054/eval_zero_shot.json` | unknown | qwen25-05b-baseline-few-shot | few-shot | unknown | 2026-09-12 | 2000 | 2 |
| `qwen25-05b/baseline/zero-shot-no-grammar/run_20260912_165549/eval_zero_shot.json` | unknown | qwen25-05b-baseline-zero-shot-no-grammar | zero-shot | unknown | 2026-09-12 | 2000 | 2 |
| `qwen25-05b/baseline/zero-shot/run_20260912_144136/eval_zero_shot.json` | unknown | qwen25-05b-baseline-zero-shot | zero-shot | unknown | 2026-09-12 | 2000 | 2 |
| `qwen25-05b/grpo/few-shot/run_20260914_070600/eval_baseline.json` | baseline | qwen25-05b-grpo-few-shot | few-shot | unknown | 2026-09-14 | 2000 | 2 |
| `qwen25-05b/grpo/few-shot/run_20260914_070600/eval_baseline__zero-shot.json` | baseline | qwen25-05b-grpo-few-shot | zero-shot | unknown | 2026-09-14 | 2000 | 2 |
| `qwen25-05b/grpo/few-shot/run_20260914_070600/eval_final.json` | unknown | qwen25-05b-grpo-few-shot | few-shot | unknown | 2026-09-14 | 2000 | 2 |
| `qwen25-05b/grpo/few-shot/run_20260914_070600/eval_final__zero-shot.json` | unknown | qwen25-05b-grpo-few-shot | zero-shot | unknown | 2026-09-14 | 2000 | 2 |
| `qwen25-05b/grpo/zero-shot/run_20260913_165601/eval_baseline.json` | baseline | qwen25-05b-grpo-zero-shot | zero-shot | unknown | 2026-09-13 | 2000 | 2 |
| `qwen25-05b/grpo/zero-shot/run_20260913_165601/eval_baseline__few-shot.json` | baseline | qwen25-05b-grpo-zero-shot | few-shot | unknown | 2026-09-13 | 2000 | 2 |
| `qwen25-05b/grpo/zero-shot/run_20260913_165601/eval_final.json` | unknown | qwen25-05b-grpo-zero-shot | zero-shot | unknown | 2026-09-13 | 2000 | 2 |
| `qwen25-05b/grpo/zero-shot/run_20260913_165601/eval_final__few-shot.json` | unknown | qwen25-05b-grpo-zero-shot | few-shot | unknown | 2026-09-13 | 2000 | 2 |
| `qwen25-05b/sft-grpo/few-shot/run_20260912_190132/eval_baseline.json` | baseline | few-shot | few-shot | unknown | 2026-09-12 | 2000 | 2 |
| `qwen25-05b/sft-grpo/few-shot/run_20260912_190132/eval_baseline__zero-shot.json` | baseline | few-shot | zero-shot | unknown | 2026-09-12 | 2000 | 2 |
| `qwen25-05b/sft-grpo/few-shot/run_20260912_190132/eval_final.json` | sft-grpo | few-shot | few-shot | unknown | 2026-09-12 | 2000 | 2 |
| `qwen25-05b/sft-grpo/few-shot/run_20260912_190132/eval_final__zero-shot.json` | sft-grpo | few-shot | zero-shot | unknown | 2026-09-12 | 2000 | 2 |
| `qwen25-05b/sft-grpo/zero-shot/run_20260913_065606/eval_baseline.json` | baseline | zero-shot | zero-shot | unknown | 2026-09-13 | 2000 | 2 |
| `qwen25-05b/sft-grpo/zero-shot/run_20260913_065606/eval_baseline__few-shot.json` | baseline | zero-shot | few-shot | unknown | 2026-09-13 | 2000 | 2 |
| `qwen25-05b/sft-grpo/zero-shot/run_20260913_065606/eval_final.json` | sft-grpo | zero-shot | zero-shot | unknown | 2026-09-13 | 2000 | 2 |
| `qwen25-05b/sft-grpo/zero-shot/run_20260913_065606/eval_final__few-shot.json` | sft-grpo | zero-shot | few-shot | unknown | 2026-09-13 | 2000 | 2 |
| `qwen25-05b/sft/zero-shot/run_20260914_220146/decoding-greedy/eval_baseline.json` | baseline | qwen25-05b-sft-zero-shot-decoding-greedy | zero-shot | unknown | 2026-09-14 | 2000 | 2 |
| `qwen25-05b/sft/zero-shot/run_20260914_220146/decoding-greedy/eval_final.json` | unknown | qwen25-05b-sft-zero-shot-decoding-greedy | zero-shot | unknown | 2026-09-14 | 2000 | 2 |
| `qwen25-05b/sft/zero-shot/run_20260914_220146/eval_baseline.json` | baseline | qwen25-05b-sft-zero-shot | zero-shot | unknown | 2026-09-14 | 2000 | 2 |
| `qwen25-05b/sft/zero-shot/run_20260914_220146/eval_baseline__few-shot.json` | baseline | qwen25-05b-sft-zero-shot | few-shot | unknown | 2026-09-14 | 2000 | 2 |
| `qwen25-05b/sft/zero-shot/run_20260914_220146/eval_final.json` | unknown | qwen25-05b-sft-zero-shot | zero-shot | unknown | 2026-09-14 | 2000 | 2 |
| `qwen25-05b/sft/zero-shot/run_20260914_220146/eval_final__few-shot.json` | unknown | qwen25-05b-sft-zero-shot | few-shot | unknown | 2026-09-14 | 2000 | 2 |

## Ablation matrix overview — ROUGE-L

One row per cell (its path under `results/`), one column per prompting mode that has at least one value. `—` = the cell was not evaluated in that mode. `*` = multiple runs collapsed, latest shown. Each cell's own `eval_baseline` (the base model in that cell's context) is not a row here: it repeats the `baseline/` cells and stays in the paired comparisons below.

| cell | zero-shot | few-shot |
|---|---|---|
| qwen25-05b/baseline/few-shot | — | 0.4664 |
| qwen25-05b/baseline/zero-shot | 0.1373 | — |
| qwen25-05b/baseline/zero-shot-no-grammar | 0.3648 | — |
| qwen25-05b/sft/zero-shot | 0.9681 | 0.9191 |
| qwen25-05b/sft/zero-shot/decoding-greedy | 0.9696 | — |
| qwen25-05b/sft-grpo/few-shot | 0.9720 | 0.9628 |
| qwen25-05b/sft-grpo/zero-shot | 0.9744 | 0.9230 |
| qwen25-05b/grpo/few-shot | 0.1925 | 0.6082 |
| qwen25-05b/grpo/zero-shot | 0.5173 | 0.5354 |
| qwen25-05b/ablations/decoding/hot-rollout | — | 0.9634 |
| qwen25-05b/ablations/decoding/no-grammar | — | 0.7590 |
| qwen25-05b/ablations/glossary/few-shot | — | 0.6094 |
| qwen25-05b/ablations/glossary/zero-shot | 0.5159 | — |
| qwen25-05b/ablations/loss/dr-grpo | — | 0.5373 |
| qwen25-05b/ablations/loss/low-beta | — | 0.9644 |
| qwen25-05b/ablations/objectives/sft-allowed-mass | 0.9769 | — |
| qwen25-05b/ablations/objectives/sft-allowed-mass/decoding-greedy | 0.9774 | — |
| qwen25-05b/ablations/objectives/sft-structured | 0.9430 | — |
| qwen25-05b/ablations/objectives/sft-structured-shuffled | 0.9464 | — |
| qwen25-05b/ablations/rewards/edit-validity | — | 0.6373 |
| qwen25-05b/ablations/rewards/lean-stack | — | 0.9634 |

## Paired comparisons — one table per factor

### Factor: `method`

**training effect (same config, baseline vs checkpoint)** — `qwen25-05b/sft-grpo/zero-shot/run_20260913_065606/eval_baseline__few-shot.json` (2026-09-13, few-shot, unknown) → `qwen25-05b/sft-grpo/zero-shot/run_20260913_065606/eval_final__few-shot.json` (2026-09-13, few-shot, unknown)

| Metric | A | B | Δ (B−A) | Interpretation |
|---|---|---|---|---|
| ROUGE-L | 0.4664 | 0.9230 | +0.4565 | above noise threshold |
| Valid ROUGE-L | 0.4581 | 0.9227 | +0.4646 | above noise threshold |
| Exact Match | 0.0148 | 0.5665 | +0.5517 | above noise threshold |
| Non-copy Tok Acc | 0.3555 | 0.9078 | +0.5523 | above noise threshold |
| Pass@1 | 0.7330 | 0.9950 | +0.2620 | above noise threshold |
| Validity | 0.9821 | 0.9997 | +0.0176 | NOT GENERALIZABLE (|delta| < 0.02): reproducible (training is bit-deterministic at fixed seed) but single-seed, so it may invert under another seed — do NOT interpret |
| BLEU (corpus) | 0.1880 | 0.8687 | +0.6807 | above noise threshold |
| BLEU (sent) | 0.1882 | 0.8535 | +0.6653 | above noise threshold |
| chrF2 (corpus) | 41.8895 | 92.8253 | +50.9359 | scale 0-100: 0.02 noise threshold not applicable |
| chrF2 (sent) | 44.6817 | 92.3412 | +47.6595 | scale 0-100: 0.02 noise threshold not applicable |
| Gloss F1 (mic) | 0.4026 | 0.9302 | +0.5276 | above noise threshold |
| Gloss F1 (sent) | 0.4439 | 0.9267 | +0.4828 | above noise threshold |
- ⚠️ factors not deducible on at least one side, equality NOT verified: reward_stack, rl_objective

**training effect (same config, baseline vs checkpoint)** — `qwen25-05b/sft-grpo/few-shot/run_20260912_190132/eval_baseline.json` (2026-09-12, few-shot, unknown) → `qwen25-05b/sft-grpo/few-shot/run_20260912_190132/eval_final.json` (2026-09-12, few-shot, unknown)

| Metric | A | B | Δ (B−A) | Interpretation |
|---|---|---|---|---|
| ROUGE-L | 0.4664 | 0.9628 | +0.4964 | above noise threshold |
| Valid ROUGE-L | 0.4581 | 0.9628 | +0.5047 | above noise threshold |
| Exact Match | 0.0148 | 0.7296 | +0.7148 | above noise threshold |
| Non-copy Tok Acc | 0.3555 | 0.9471 | +0.5916 | above noise threshold |
| Pass@1 | 0.7330 | 0.9980 | +0.2650 | above noise threshold |
| Validity | 0.9821 | 1.0000 | +0.0179 | NOT GENERALIZABLE (|delta| < 0.02): reproducible (training is bit-deterministic at fixed seed) but single-seed, so it may invert under another seed — do NOT interpret |
| BLEU (corpus) | 0.1880 | 0.9220 | +0.7340 | above noise threshold |
| BLEU (sent) | 0.1882 | 0.9150 | +0.7268 | above noise threshold |
| chrF2 (corpus) | 41.8895 | 96.0171 | +54.1276 | scale 0-100: 0.02 noise threshold not applicable |
| chrF2 (sent) | 44.6817 | 95.9517 | +51.2700 | scale 0-100: 0.02 noise threshold not applicable |
| Gloss F1 (mic) | 0.4026 | 0.9648 | +0.5622 | above noise threshold |
| Gloss F1 (sent) | 0.4439 | 0.9637 | +0.5198 | above noise threshold |
- ⚠️ factors not deducible on at least one side, equality NOT verified: reward_stack, rl_objective

**training effect (same config, baseline vs checkpoint)** — `qwen25-05b/sft-grpo/zero-shot/run_20260913_065606/eval_baseline.json` (2026-09-13, zero-shot, unknown) → `qwen25-05b/sft-grpo/zero-shot/run_20260913_065606/eval_final.json` (2026-09-13, zero-shot, unknown)

| Metric | A | B | Δ (B−A) | Interpretation |
|---|---|---|---|---|
| ROUGE-L | 0.1373 | 0.9744 | +0.8371 | above noise threshold |
| Valid ROUGE-L | 0.1243 | 0.9744 | +0.8501 | above noise threshold |
| Exact Match | 0.0017 | 0.8095 | +0.8078 | above noise threshold |
| Non-copy Tok Acc | 0.0451 | 0.9650 | +0.9199 | above noise threshold |
| Pass@1 | 0.1820 | 0.9980 | +0.8160 | above noise threshold |
| Validity | 0.9055 | 1.0000 | +0.0945 | above noise threshold |
| BLEU (corpus) | 0.0270 | 0.9443 | +0.9173 | above noise threshold |
| BLEU (sent) | 0.0304 | 0.9408 | +0.9104 | above noise threshold |
| chrF2 (corpus) | 13.2069 | 97.2655 | +84.0587 | scale 0-100: 0.02 noise threshold not applicable |
| chrF2 (sent) | 13.1640 | 97.2383 | +84.0742 | scale 0-100: 0.02 noise threshold not applicable |
| Gloss F1 (mic) | 0.1186 | 0.9757 | +0.8571 | above noise threshold |
| Gloss F1 (sent) | 0.1416 | 0.9753 | +0.8337 | above noise threshold |
- ⚠️ factors not deducible on at least one side, equality NOT verified: reward_stack, rl_objective

**training effect (same config, baseline vs checkpoint)** — `qwen25-05b/sft-grpo/few-shot/run_20260912_190132/eval_baseline__zero-shot.json` (2026-09-12, zero-shot, unknown) → `qwen25-05b/sft-grpo/few-shot/run_20260912_190132/eval_final__zero-shot.json` (2026-09-12, zero-shot, unknown)

| Metric | A | B | Δ (B−A) | Interpretation |
|---|---|---|---|---|
| ROUGE-L | 0.1373 | 0.9720 | +0.8347 | above noise threshold |
| Valid ROUGE-L | 0.1243 | 0.9720 | +0.8476 | above noise threshold |
| Exact Match | 0.0017 | 0.7991 | +0.7974 | above noise threshold |
| Non-copy Tok Acc | 0.0451 | 0.9615 | +0.9164 | above noise threshold |
| Pass@1 | 0.1820 | 0.9975 | +0.8155 | above noise threshold |
| Validity | 0.9055 | 1.0000 | +0.0945 | above noise threshold |
| BLEU (corpus) | 0.0270 | 0.9422 | +0.9152 | above noise threshold |
| BLEU (sent) | 0.0304 | 0.9365 | +0.9061 | above noise threshold |
| chrF2 (corpus) | 13.2069 | 96.9614 | +83.7545 | scale 0-100: 0.02 noise threshold not applicable |
| chrF2 (sent) | 13.1640 | 96.9054 | +83.7414 | scale 0-100: 0.02 noise threshold not applicable |
| Gloss F1 (mic) | 0.1186 | 0.9740 | +0.8554 | above noise threshold |
| Gloss F1 (sent) | 0.1416 | 0.9731 | +0.8315 | above noise threshold |
- ⚠️ factors not deducible on at least one side, equality NOT verified: reward_stack, rl_objective

### Factor: `prompting`

🔴 **prompt-dependence (dual eval, same checkpoint)** — `qwen25-05b/sft/zero-shot/run_20260914_220146/eval_baseline__few-shot.json` (2026-09-14, few-shot, unknown) → `qwen25-05b/sft/zero-shot/run_20260914_220146/eval_baseline.json` (2026-09-14, zero-shot, unknown)

| Metric | A | B | Δ (B−A) | Interpretation |
|---|---|---|---|---|
| ROUGE-L | 0.4664 | 0.1373 | -0.3291 | above noise threshold |
| Valid ROUGE-L | 0.4581 | 0.1243 | -0.3338 | above noise threshold |
| Exact Match | 0.0148 | 0.0017 | -0.0131 | NOT GENERALIZABLE (|delta| < 0.02): reproducible (training is bit-deterministic at fixed seed) but single-seed, so it may invert under another seed — do NOT interpret |
| Non-copy Tok Acc | 0.3555 | 0.0451 | -0.3104 | above noise threshold |
| Pass@1 | 0.7330 | 0.1820 | -0.5510 | above noise threshold |
| Validity | 0.9821 | 0.9055 | -0.0766 | above noise threshold |
| BLEU (corpus) | 0.1880 | 0.0270 | -0.1610 | above noise threshold |
| BLEU (sent) | 0.1882 | 0.0304 | -0.1578 | above noise threshold |
| chrF2 (corpus) | 41.8895 | 13.2069 | -28.6826 | scale 0-100: 0.02 noise threshold not applicable |
| chrF2 (sent) | 44.6817 | 13.1640 | -31.5177 | scale 0-100: 0.02 noise threshold not applicable |
| Gloss F1 (mic) | 0.4026 | 0.1186 | -0.2840 | above noise threshold |
| Gloss F1 (sent) | 0.4439 | 0.1416 | -0.3023 | above noise threshold |
- ⚠️ HIGHLIGHT: same model evaluated in both prompting modes — how much does it depend on the prompt as a crutch?
- ⚠️ factors not deducible on at least one side, equality NOT verified: reward_stack, rl_objective

🔴 **prompt-dependence (dual eval, same checkpoint)** — `qwen25-05b/grpo/few-shot/run_20260914_070600/eval_baseline.json` (2026-09-14, few-shot, unknown) → `qwen25-05b/grpo/few-shot/run_20260914_070600/eval_baseline__zero-shot.json` (2026-09-14, zero-shot, unknown)

| Metric | A | B | Δ (B−A) | Interpretation |
|---|---|---|---|---|
| ROUGE-L | 0.4664 | 0.1373 | -0.3291 | above noise threshold |
| Valid ROUGE-L | 0.4581 | 0.1243 | -0.3338 | above noise threshold |
| Exact Match | 0.0148 | 0.0017 | -0.0131 | NOT GENERALIZABLE (|delta| < 0.02): reproducible (training is bit-deterministic at fixed seed) but single-seed, so it may invert under another seed — do NOT interpret |
| Non-copy Tok Acc | 0.3555 | 0.0451 | -0.3104 | above noise threshold |
| Pass@1 | 0.7330 | 0.1820 | -0.5510 | above noise threshold |
| Validity | 0.9821 | 0.9055 | -0.0766 | above noise threshold |
| BLEU (corpus) | 0.1880 | 0.0270 | -0.1610 | above noise threshold |
| BLEU (sent) | 0.1882 | 0.0304 | -0.1578 | above noise threshold |
| chrF2 (corpus) | 41.8895 | 13.2069 | -28.6826 | scale 0-100: 0.02 noise threshold not applicable |
| chrF2 (sent) | 44.6817 | 13.1640 | -31.5177 | scale 0-100: 0.02 noise threshold not applicable |
| Gloss F1 (mic) | 0.4026 | 0.1186 | -0.2840 | above noise threshold |
| Gloss F1 (sent) | 0.4439 | 0.1416 | -0.3023 | above noise threshold |
- ⚠️ HIGHLIGHT: same model evaluated in both prompting modes — how much does it depend on the prompt as a crutch?
- ⚠️ factors not deducible on at least one side, equality NOT verified: reward_stack, rl_objective

🔴 **prompt-dependence (dual eval, same checkpoint)** — `qwen25-05b/grpo/zero-shot/run_20260913_165601/eval_baseline__few-shot.json` (2026-09-13, few-shot, unknown) → `qwen25-05b/grpo/zero-shot/run_20260913_165601/eval_baseline.json` (2026-09-13, zero-shot, unknown)

| Metric | A | B | Δ (B−A) | Interpretation |
|---|---|---|---|---|
| ROUGE-L | 0.4664 | 0.1373 | -0.3291 | above noise threshold |
| Valid ROUGE-L | 0.4581 | 0.1243 | -0.3338 | above noise threshold |
| Exact Match | 0.0148 | 0.0017 | -0.0131 | NOT GENERALIZABLE (|delta| < 0.02): reproducible (training is bit-deterministic at fixed seed) but single-seed, so it may invert under another seed — do NOT interpret |
| Non-copy Tok Acc | 0.3555 | 0.0451 | -0.3104 | above noise threshold |
| Pass@1 | 0.7330 | 0.1820 | -0.5510 | above noise threshold |
| Validity | 0.9821 | 0.9055 | -0.0766 | above noise threshold |
| BLEU (corpus) | 0.1880 | 0.0270 | -0.1610 | above noise threshold |
| BLEU (sent) | 0.1882 | 0.0304 | -0.1578 | above noise threshold |
| chrF2 (corpus) | 41.8895 | 13.2069 | -28.6826 | scale 0-100: 0.02 noise threshold not applicable |
| chrF2 (sent) | 44.6817 | 13.1640 | -31.5177 | scale 0-100: 0.02 noise threshold not applicable |
| Gloss F1 (mic) | 0.4026 | 0.1186 | -0.2840 | above noise threshold |
| Gloss F1 (sent) | 0.4439 | 0.1416 | -0.3023 | above noise threshold |
- ⚠️ HIGHLIGHT: same model evaluated in both prompting modes — how much does it depend on the prompt as a crutch?
- ⚠️ factors not deducible on at least one side, equality NOT verified: reward_stack, rl_objective

🔴 **prompt-dependence (dual eval, same checkpoint)** — `qwen25-05b/sft-grpo/zero-shot/run_20260913_065606/eval_baseline__few-shot.json` (2026-09-13, few-shot, unknown) → `qwen25-05b/sft-grpo/zero-shot/run_20260913_065606/eval_baseline.json` (2026-09-13, zero-shot, unknown)

| Metric | A | B | Δ (B−A) | Interpretation |
|---|---|---|---|---|
| ROUGE-L | 0.4664 | 0.1373 | -0.3291 | above noise threshold |
| Valid ROUGE-L | 0.4581 | 0.1243 | -0.3338 | above noise threshold |
| Exact Match | 0.0148 | 0.0017 | -0.0131 | NOT GENERALIZABLE (|delta| < 0.02): reproducible (training is bit-deterministic at fixed seed) but single-seed, so it may invert under another seed — do NOT interpret |
| Non-copy Tok Acc | 0.3555 | 0.0451 | -0.3104 | above noise threshold |
| Pass@1 | 0.7330 | 0.1820 | -0.5510 | above noise threshold |
| Validity | 0.9821 | 0.9055 | -0.0766 | above noise threshold |
| BLEU (corpus) | 0.1880 | 0.0270 | -0.1610 | above noise threshold |
| BLEU (sent) | 0.1882 | 0.0304 | -0.1578 | above noise threshold |
| chrF2 (corpus) | 41.8895 | 13.2069 | -28.6826 | scale 0-100: 0.02 noise threshold not applicable |
| chrF2 (sent) | 44.6817 | 13.1640 | -31.5177 | scale 0-100: 0.02 noise threshold not applicable |
| Gloss F1 (mic) | 0.4026 | 0.1186 | -0.2840 | above noise threshold |
| Gloss F1 (sent) | 0.4439 | 0.1416 | -0.3023 | above noise threshold |
- ⚠️ HIGHLIGHT: same model evaluated in both prompting modes — how much does it depend on the prompt as a crutch?
- ⚠️ factors not deducible on at least one side, equality NOT verified: reward_stack, rl_objective

🔴 **prompt-dependence (dual eval, same checkpoint)** — `qwen25-05b/sft-grpo/few-shot/run_20260912_190132/eval_baseline.json` (2026-09-12, few-shot, unknown) → `qwen25-05b/sft-grpo/few-shot/run_20260912_190132/eval_baseline__zero-shot.json` (2026-09-12, zero-shot, unknown)

| Metric | A | B | Δ (B−A) | Interpretation |
|---|---|---|---|---|
| ROUGE-L | 0.4664 | 0.1373 | -0.3291 | above noise threshold |
| Valid ROUGE-L | 0.4581 | 0.1243 | -0.3338 | above noise threshold |
| Exact Match | 0.0148 | 0.0017 | -0.0131 | NOT GENERALIZABLE (|delta| < 0.02): reproducible (training is bit-deterministic at fixed seed) but single-seed, so it may invert under another seed — do NOT interpret |
| Non-copy Tok Acc | 0.3555 | 0.0451 | -0.3104 | above noise threshold |
| Pass@1 | 0.7330 | 0.1820 | -0.5510 | above noise threshold |
| Validity | 0.9821 | 0.9055 | -0.0766 | above noise threshold |
| BLEU (corpus) | 0.1880 | 0.0270 | -0.1610 | above noise threshold |
| BLEU (sent) | 0.1882 | 0.0304 | -0.1578 | above noise threshold |
| chrF2 (corpus) | 41.8895 | 13.2069 | -28.6826 | scale 0-100: 0.02 noise threshold not applicable |
| chrF2 (sent) | 44.6817 | 13.1640 | -31.5177 | scale 0-100: 0.02 noise threshold not applicable |
| Gloss F1 (mic) | 0.4026 | 0.1186 | -0.2840 | above noise threshold |
| Gloss F1 (sent) | 0.4439 | 0.1416 | -0.3023 | above noise threshold |
- ⚠️ HIGHLIGHT: same model evaluated in both prompting modes — how much does it depend on the prompt as a crutch?
- ⚠️ factors not deducible on at least one side, equality NOT verified: reward_stack, rl_objective

**prompting (cross-run)** — `qwen25-05b/ablations/glossary/few-shot/run_20260917_212556/eval_baseline.json` (2026-09-17, few-shot, unknown) → `qwen25-05b/ablations/objectives/sft-allowed-mass/run_20260920_205638/decoding-greedy/eval_baseline.json` (2026-09-20, zero-shot, unknown)

| Metric | A | B | Δ (B−A) | Interpretation |
|---|---|---|---|---|
| ROUGE-L | 0.4664 | 0.1446 | -0.3218 | above noise threshold |
| Valid ROUGE-L | 0.4581 | 0.1302 | -0.3279 | above noise threshold |
| Exact Match | 0.0148 | 0.0025 | -0.0123 | NOT GENERALIZABLE (|delta| < 0.02): reproducible (training is bit-deterministic at fixed seed) but single-seed, so it may invert under another seed — do NOT interpret |
| Non-copy Tok Acc | 0.3555 | 0.0468 | -0.3087 | above noise threshold |
| Pass@1 | 0.7330 | 0.1880 | -0.5450 | above noise threshold |
| Validity | 0.9821 | 0.9005 | -0.0816 | above noise threshold |
| BLEU (corpus) | 0.1880 | 0.0254 | -0.1626 | above noise threshold |
| BLEU (sent) | 0.1882 | 0.0351 | -0.1531 | above noise threshold |
| chrF2 (corpus) | 41.8895 | 12.9226 | -28.9668 | scale 0-100: 0.02 noise threshold not applicable |
| chrF2 (sent) | 44.6817 | 13.2644 | -31.4174 | scale 0-100: 0.02 noise threshold not applicable |
| Gloss F1 (mic) | 0.4026 | 0.1108 | -0.2918 | above noise threshold |
| Gloss F1 (sent) | 0.4439 | 0.1493 | -0.2945 | above noise threshold |
- ⚠️ cross-run comparison: checkpoint identity NOT guaranteed
- ⚠️ factors not deducible on at least one side, equality NOT verified: reward_stack, rl_objective

**prompting (cross-run)** — `qwen25-05b/ablations/glossary/zero-shot/run_20260917_105109/eval_baseline.json` (2026-09-17, zero-shot, unknown) → `qwen25-05b/ablations/loss/dr-grpo/run_20260916_183557/eval_baseline.json` (2026-09-16, few-shot, unknown)

| Metric | A | B | Δ (B−A) | Interpretation |
|---|---|---|---|---|
| ROUGE-L | 0.1373 | 0.4664 | +0.3291 | above noise threshold |
| Valid ROUGE-L | 0.1243 | 0.4581 | +0.3338 | above noise threshold |
| Exact Match | 0.0017 | 0.0148 | +0.0131 | NOT GENERALIZABLE (|delta| < 0.02): reproducible (training is bit-deterministic at fixed seed) but single-seed, so it may invert under another seed — do NOT interpret |
| Non-copy Tok Acc | 0.0451 | 0.3555 | +0.3104 | above noise threshold |
| Pass@1 | 0.1820 | 0.7330 | +0.5510 | above noise threshold |
| Validity | 0.9055 | 0.9821 | +0.0766 | above noise threshold |
| BLEU (corpus) | 0.0270 | 0.1880 | +0.1610 | above noise threshold |
| BLEU (sent) | 0.0304 | 0.1882 | +0.1578 | above noise threshold |
| chrF2 (corpus) | 13.2069 | 41.8895 | +28.6826 | scale 0-100: 0.02 noise threshold not applicable |
| chrF2 (sent) | 13.1640 | 44.6817 | +31.5177 | scale 0-100: 0.02 noise threshold not applicable |
| Gloss F1 (mic) | 0.1186 | 0.4026 | +0.2840 | above noise threshold |
| Gloss F1 (sent) | 0.1416 | 0.4439 | +0.3023 | above noise threshold |
- ⚠️ cross-run comparison: checkpoint identity NOT guaranteed
- ⚠️ factors not deducible on at least one side, equality NOT verified: reward_stack, rl_objective

**prompting (cross-run)** — `qwen25-05b/ablations/decoding/hot-rollout/run_20260915_083119/eval_baseline.json` (2026-09-15, few-shot, unknown) → `qwen25-05b/sft/zero-shot/run_20260914_220146/decoding-greedy/eval_baseline.json` (2026-09-14, zero-shot, unknown)

| Metric | A | B | Δ (B−A) | Interpretation |
|---|---|---|---|---|
| ROUGE-L | 0.4664 | 0.1446 | -0.3218 | above noise threshold |
| Valid ROUGE-L | 0.4581 | 0.1302 | -0.3279 | above noise threshold |
| Exact Match | 0.0148 | 0.0025 | -0.0123 | NOT GENERALIZABLE (|delta| < 0.02): reproducible (training is bit-deterministic at fixed seed) but single-seed, so it may invert under another seed — do NOT interpret |
| Non-copy Tok Acc | 0.3555 | 0.0468 | -0.3087 | above noise threshold |
| Pass@1 | 0.7330 | 0.1880 | -0.5450 | above noise threshold |
| Validity | 0.9821 | 0.9005 | -0.0816 | above noise threshold |
| BLEU (corpus) | 0.1880 | 0.0254 | -0.1626 | above noise threshold |
| BLEU (sent) | 0.1882 | 0.0351 | -0.1531 | above noise threshold |
| chrF2 (corpus) | 41.8895 | 12.9226 | -28.9668 | scale 0-100: 0.02 noise threshold not applicable |
| chrF2 (sent) | 44.6817 | 13.2644 | -31.4174 | scale 0-100: 0.02 noise threshold not applicable |
| Gloss F1 (mic) | 0.4026 | 0.1108 | -0.2918 | above noise threshold |
| Gloss F1 (sent) | 0.4439 | 0.1493 | -0.2945 | above noise threshold |
- ⚠️ cross-run comparison: checkpoint identity NOT guaranteed
- ⚠️ factors not deducible on at least one side, equality NOT verified: reward_stack, rl_objective

🔴 **prompt-dependence (dual eval, same checkpoint)** — `qwen25-05b/sft-grpo/zero-shot/run_20260913_065606/eval_final__few-shot.json` (2026-09-13, few-shot, unknown) → `qwen25-05b/sft-grpo/zero-shot/run_20260913_065606/eval_final.json` (2026-09-13, zero-shot, unknown)

| Metric | A | B | Δ (B−A) | Interpretation |
|---|---|---|---|---|
| ROUGE-L | 0.9230 | 0.9744 | +0.0515 | above noise threshold |
| Valid ROUGE-L | 0.9227 | 0.9744 | +0.0517 | above noise threshold |
| Exact Match | 0.5665 | 0.8095 | +0.2430 | above noise threshold |
| Non-copy Tok Acc | 0.9078 | 0.9650 | +0.0572 | above noise threshold |
| Pass@1 | 0.9950 | 0.9980 | +0.0030 | NOT GENERALIZABLE (|delta| < 0.02): reproducible (training is bit-deterministic at fixed seed) but single-seed, so it may invert under another seed — do NOT interpret |
| Validity | 0.9997 | 1.0000 | +0.0003 | NOT GENERALIZABLE (|delta| < 0.02): reproducible (training is bit-deterministic at fixed seed) but single-seed, so it may invert under another seed — do NOT interpret |
| BLEU (corpus) | 0.8687 | 0.9443 | +0.0756 | above noise threshold |
| BLEU (sent) | 0.8535 | 0.9408 | +0.0873 | above noise threshold |
| chrF2 (corpus) | 92.8253 | 97.2655 | +4.4402 | scale 0-100: 0.02 noise threshold not applicable |
| chrF2 (sent) | 92.3412 | 97.2383 | +4.8971 | scale 0-100: 0.02 noise threshold not applicable |
| Gloss F1 (mic) | 0.9302 | 0.9757 | +0.0455 | above noise threshold |
| Gloss F1 (sent) | 0.9267 | 0.9753 | +0.0486 | above noise threshold |
- ⚠️ HIGHLIGHT: same model evaluated in both prompting modes — how much does it depend on the prompt as a crutch?
- ⚠️ factors not deducible on at least one side, equality NOT verified: reward_stack, rl_objective

🔴 **prompt-dependence (dual eval, same checkpoint)** — `qwen25-05b/sft-grpo/few-shot/run_20260912_190132/eval_final.json` (2026-09-12, few-shot, unknown) → `qwen25-05b/sft-grpo/few-shot/run_20260912_190132/eval_final__zero-shot.json` (2026-09-12, zero-shot, unknown)

| Metric | A | B | Δ (B−A) | Interpretation |
|---|---|---|---|---|
| ROUGE-L | 0.9628 | 0.9720 | +0.0091 | NOT GENERALIZABLE (|delta| < 0.02): reproducible (training is bit-deterministic at fixed seed) but single-seed, so it may invert under another seed — do NOT interpret |
| Valid ROUGE-L | 0.9628 | 0.9720 | +0.0091 | NOT GENERALIZABLE (|delta| < 0.02): reproducible (training is bit-deterministic at fixed seed) but single-seed, so it may invert under another seed — do NOT interpret |
| Exact Match | 0.7296 | 0.7991 | +0.0695 | above noise threshold |
| Non-copy Tok Acc | 0.9471 | 0.9615 | +0.0144 | NOT GENERALIZABLE (|delta| < 0.02): reproducible (training is bit-deterministic at fixed seed) but single-seed, so it may invert under another seed — do NOT interpret |
| Pass@1 | 0.9980 | 0.9975 | -0.0005 | NOT GENERALIZABLE (|delta| < 0.02): reproducible (training is bit-deterministic at fixed seed) but single-seed, so it may invert under another seed — do NOT interpret |
| Validity | 1.0000 | 1.0000 | +0.0000 | NOT GENERALIZABLE (|delta| < 0.02): reproducible (training is bit-deterministic at fixed seed) but single-seed, so it may invert under another seed — do NOT interpret |
| BLEU (corpus) | 0.9220 | 0.9422 | +0.0202 | above noise threshold |
| BLEU (sent) | 0.9150 | 0.9365 | +0.0215 | above noise threshold |
| chrF2 (corpus) | 96.0171 | 96.9614 | +0.9443 | scale 0-100: 0.02 noise threshold not applicable |
| chrF2 (sent) | 95.9517 | 96.9054 | +0.9537 | scale 0-100: 0.02 noise threshold not applicable |
| Gloss F1 (mic) | 0.9648 | 0.9740 | +0.0092 | NOT GENERALIZABLE (|delta| < 0.02): reproducible (training is bit-deterministic at fixed seed) but single-seed, so it may invert under another seed — do NOT interpret |
| Gloss F1 (sent) | 0.9637 | 0.9731 | +0.0094 | NOT GENERALIZABLE (|delta| < 0.02): reproducible (training is bit-deterministic at fixed seed) but single-seed, so it may invert under another seed — do NOT interpret |
- ⚠️ HIGHLIGHT: same model evaluated in both prompting modes — how much does it depend on the prompt as a crutch?
- ⚠️ factors not deducible on at least one side, equality NOT verified: reward_stack, rl_objective

🔴 **prompt-dependence (dual eval, same checkpoint)** — `qwen25-05b/sft/zero-shot/run_20260914_220146/eval_final__few-shot.json` (2026-09-14, few-shot, unknown) → `qwen25-05b/sft/zero-shot/run_20260914_220146/eval_final.json` (2026-09-14, zero-shot, unknown)

| Metric | A | B | Δ (B−A) | Interpretation |
|---|---|---|---|---|
| ROUGE-L | 0.9191 | 0.9681 | +0.0490 | above noise threshold |
| Valid ROUGE-L | 0.9185 | 0.9681 | +0.0496 | above noise threshold |
| Exact Match | 0.5380 | 0.7662 | +0.2282 | above noise threshold |
| Non-copy Tok Acc | 0.8844 | 0.9510 | +0.0666 | above noise threshold |
| Pass@1 | 0.9945 | 0.9975 | +0.0030 | NOT GENERALIZABLE (|delta| < 0.02): reproducible (training is bit-deterministic at fixed seed) but single-seed, so it may invert under another seed — do NOT interpret |
| Validity | 0.9994 | 1.0000 | +0.0006 | NOT GENERALIZABLE (|delta| < 0.02): reproducible (training is bit-deterministic at fixed seed) but single-seed, so it may invert under another seed — do NOT interpret |
| BLEU (corpus) | 0.8488 | 0.9316 | +0.0829 | above noise threshold |
| BLEU (sent) | 0.8356 | 0.9280 | +0.0924 | above noise threshold |
| chrF2 (corpus) | 91.4987 | 96.3620 | +4.8633 | scale 0-100: 0.02 noise threshold not applicable |
| chrF2 (sent) | 91.2734 | 96.4463 | +5.1729 | scale 0-100: 0.02 noise threshold not applicable |
| Gloss F1 (mic) | 0.9227 | 0.9698 | +0.0471 | above noise threshold |
| Gloss F1 (sent) | 0.9212 | 0.9698 | +0.0486 | above noise threshold |
- ⚠️ HIGHLIGHT: same model evaluated in both prompting modes — how much does it depend on the prompt as a crutch?
- ⚠️ factors not deducible on at least one side, equality NOT verified: reward_stack, rl_objective

🔴 **prompt-dependence (dual eval, same checkpoint)** — `qwen25-05b/grpo/few-shot/run_20260914_070600/eval_final.json` (2026-09-14, few-shot, unknown) → `qwen25-05b/grpo/few-shot/run_20260914_070600/eval_final__zero-shot.json` (2026-09-14, zero-shot, unknown)

| Metric | A | B | Δ (B−A) | Interpretation |
|---|---|---|---|---|
| ROUGE-L | 0.6082 | 0.1925 | -0.4157 | above noise threshold |
| Valid ROUGE-L | 0.6054 | 0.1668 | -0.4387 | above noise threshold |
| Exact Match | 0.0544 | 0.0029 | -0.0515 | above noise threshold |
| Non-copy Tok Acc | 0.4692 | 0.0668 | -0.4023 | above noise threshold |
| Pass@1 | 0.9315 | 0.3020 | -0.6295 | above noise threshold |
| Validity | 0.9954 | 0.8662 | -0.1292 | above noise threshold |
| BLEU (corpus) | 0.3419 | 0.0350 | -0.3069 | above noise threshold |
| BLEU (sent) | 0.3235 | 0.0584 | -0.2651 | above noise threshold |
| chrF2 (corpus) | 55.6782 | 17.7961 | -37.8821 | scale 0-100: 0.02 noise threshold not applicable |
| chrF2 (sent) | 57.8908 | 17.8803 | -40.0105 | scale 0-100: 0.02 noise threshold not applicable |
| Gloss F1 (mic) | 0.6220 | 0.1600 | -0.4620 | above noise threshold |
| Gloss F1 (sent) | 0.6444 | 0.2272 | -0.4172 | above noise threshold |
- ⚠️ HIGHLIGHT: same model evaluated in both prompting modes — how much does it depend on the prompt as a crutch?
- ⚠️ factors not deducible on at least one side, equality NOT verified: reward_stack, rl_objective

🔴 **prompt-dependence (dual eval, same checkpoint)** — `qwen25-05b/grpo/zero-shot/run_20260913_165601/eval_final__few-shot.json` (2026-09-13, few-shot, unknown) → `qwen25-05b/grpo/zero-shot/run_20260913_165601/eval_final.json` (2026-09-13, zero-shot, unknown)

| Metric | A | B | Δ (B−A) | Interpretation |
|---|---|---|---|---|
| ROUGE-L | 0.5354 | 0.5173 | -0.0181 | NOT GENERALIZABLE (|delta| < 0.02): reproducible (training is bit-deterministic at fixed seed) but single-seed, so it may invert under another seed — do NOT interpret |
| Valid ROUGE-L | 0.5333 | 0.5150 | -0.0184 | NOT GENERALIZABLE (|delta| < 0.02): reproducible (training is bit-deterministic at fixed seed) but single-seed, so it may invert under another seed — do NOT interpret |
| Exact Match | 0.0180 | 0.0078 | -0.0102 | NOT GENERALIZABLE (|delta| < 0.02): reproducible (training is bit-deterministic at fixed seed) but single-seed, so it may invert under another seed — do NOT interpret |
| Non-copy Tok Acc | 0.3333 | 0.1927 | -0.1406 | above noise threshold |
| Pass@1 | 0.8600 | 0.8605 | +0.0005 | NOT GENERALIZABLE (|delta| < 0.02): reproducible (training is bit-deterministic at fixed seed) but single-seed, so it may invert under another seed — do NOT interpret |
| Validity | 0.9961 | 0.9954 | -0.0007 | NOT GENERALIZABLE (|delta| < 0.02): reproducible (training is bit-deterministic at fixed seed) but single-seed, so it may invert under another seed — do NOT interpret |
| BLEU (corpus) | 0.2494 | 0.2322 | -0.0173 | NOT GENERALIZABLE (|delta| < 0.02): reproducible (training is bit-deterministic at fixed seed) but single-seed, so it may invert under another seed — do NOT interpret |
| BLEU (sent) | 0.2302 | 0.2060 | -0.0241 | above noise threshold |
| chrF2 (corpus) | 48.1533 | 47.0547 | -1.0986 | scale 0-100: 0.02 noise threshold not applicable |
| chrF2 (sent) | 49.9417 | 47.3767 | -2.5650 | scale 0-100: 0.02 noise threshold not applicable |
| Gloss F1 (mic) | 0.5160 | 0.5593 | +0.0433 | above noise threshold |
| Gloss F1 (sent) | 0.5322 | 0.5628 | +0.0307 | above noise threshold |
- ⚠️ HIGHLIGHT: same model evaluated in both prompting modes — how much does it depend on the prompt as a crutch?
- ⚠️ factors not deducible on at least one side, equality NOT verified: reward_stack, rl_objective

**prompting (cross-run)** — `qwen25-05b/ablations/glossary/few-shot/run_20260917_212556/eval_final.json` (2026-09-17, few-shot, unknown) → `qwen25-05b/ablations/objectives/sft-allowed-mass/run_20260920_205638/eval_final.json` (2026-09-20, zero-shot, unknown)

| Metric | A | B | Δ (B−A) | Interpretation |
|---|---|---|---|---|
| ROUGE-L | 0.6094 | 0.9769 | +0.3675 | above noise threshold |
| Valid ROUGE-L | 0.6082 | 0.9769 | +0.3687 | above noise threshold |
| Exact Match | 0.0544 | 0.8341 | +0.7797 | above noise threshold |
| Non-copy Tok Acc | 0.4636 | 0.9685 | +0.5049 | above noise threshold |
| Pass@1 | 0.9305 | 0.9985 | +0.0680 | above noise threshold |
| Validity | 0.9980 | 1.0000 | +0.0020 | NOT GENERALIZABLE (|delta| < 0.02): reproducible (training is bit-deterministic at fixed seed) but single-seed, so it may invert under another seed — do NOT interpret |
| BLEU (corpus) | 0.3419 | 0.9518 | +0.6100 | above noise threshold |
| BLEU (sent) | 0.3245 | 0.9491 | +0.6247 | above noise threshold |
| chrF2 (corpus) | 55.5551 | 97.5490 | +41.9939 | scale 0-100: 0.02 noise threshold not applicable |
| chrF2 (sent) | 57.7181 | 97.5462 | +39.8281 | scale 0-100: 0.02 noise threshold not applicable |
| Gloss F1 (mic) | 0.6261 | 0.9783 | +0.3522 | above noise threshold |
| Gloss F1 (sent) | 0.6437 | 0.9780 | +0.3343 | above noise threshold |
- ⚠️ cross-run comparison: checkpoint identity NOT guaranteed
- ⚠️ factors not deducible on at least one side, equality NOT verified: reward_stack, rl_objective

**prompting (cross-run)** — `qwen25-05b/ablations/glossary/zero-shot/run_20260917_105109/eval_final.json` (2026-09-17, zero-shot, unknown) → `qwen25-05b/ablations/loss/low-beta/run_20260917_053119/eval_final.json` (2026-09-17, few-shot, unknown)

| Metric | A | B | Δ (B−A) | Interpretation |
|---|---|---|---|---|
| ROUGE-L | 0.5159 | 0.9644 | +0.4485 | above noise threshold |
| Valid ROUGE-L | 0.5133 | 0.9644 | +0.4511 | above noise threshold |
| Exact Match | 0.0098 | 0.7414 | +0.7316 | above noise threshold |
| Non-copy Tok Acc | 0.1858 | 0.9499 | +0.7641 | above noise threshold |
| Pass@1 | 0.8550 | 0.9975 | +0.1425 | above noise threshold |
| Validity | 0.9949 | 1.0000 | +0.0051 | NOT GENERALIZABLE (|delta| < 0.02): reproducible (training is bit-deterministic at fixed seed) but single-seed, so it may invert under another seed — do NOT interpret |
| BLEU (corpus) | 0.2317 | 0.9244 | +0.6928 | above noise threshold |
| BLEU (sent) | 0.2065 | 0.9188 | +0.7123 | above noise threshold |
| chrF2 (corpus) | 47.1415 | 96.1364 | +48.9949 | scale 0-100: 0.02 noise threshold not applicable |
| chrF2 (sent) | 47.4937 | 96.0904 | +48.5967 | scale 0-100: 0.02 noise threshold not applicable |
| Gloss F1 (mic) | 0.5605 | 0.9661 | +0.4056 | above noise threshold |
| Gloss F1 (sent) | 0.5666 | 0.9653 | +0.3987 | above noise threshold |
- ⚠️ cross-run comparison: checkpoint identity NOT guaranteed
- ⚠️ factors not deducible on at least one side, equality NOT verified: reward_stack, rl_objective

**prompting (cross-run)** — `qwen25-05b/ablations/loss/dr-grpo/run_20260916_183557/eval_final.json` (2026-09-16, few-shot, unknown) → `qwen25-05b/ablations/objectives/sft-allowed-mass/run_20260920_205638/decoding-greedy/eval_final.json` (2026-09-20, zero-shot, unknown)

| Metric | A | B | Δ (B−A) | Interpretation |
|---|---|---|---|---|
| ROUGE-L | 0.5373 | 0.9774 | +0.4401 | above noise threshold |
| Valid ROUGE-L | 0.5346 | 0.9774 | +0.4428 | above noise threshold |
| Exact Match | 0.0360 | 0.8420 | +0.8060 | above noise threshold |
| Non-copy Tok Acc | 0.4043 | 0.9699 | +0.5657 | above noise threshold |
| Pass@1 | 0.8385 | 0.9975 | +0.1590 | above noise threshold |
| Validity | 0.9950 | 1.0000 | +0.0050 | NOT GENERALIZABLE (|delta| < 0.02): reproducible (training is bit-deterministic at fixed seed) but single-seed, so it may invert under another seed — do NOT interpret |
| BLEU (corpus) | 0.2672 | 0.9533 | +0.6860 | above noise threshold |
| BLEU (sent) | 0.2534 | 0.9505 | +0.6971 | above noise threshold |
| chrF2 (corpus) | 48.5883 | 97.6321 | +49.0438 | scale 0-100: 0.02 noise threshold not applicable |
| chrF2 (sent) | 51.0681 | 97.6195 | +46.5514 | scale 0-100: 0.02 noise threshold not applicable |
| Gloss F1 (mic) | 0.5246 | 0.9791 | +0.4546 | above noise threshold |
| Gloss F1 (sent) | 0.5501 | 0.9786 | +0.4285 | above noise threshold |
- ⚠️ cross-run comparison: checkpoint identity NOT guaranteed
- ⚠️ factors not deducible on at least one side, equality NOT verified: reward_stack, rl_objective

**prompting (cross-run)** — `qwen25-05b/ablations/objectives/sft-structured/run_20260923_101821/eval_final.json` (2026-09-23, zero-shot, unknown) → `qwen25-05b/ablations/rewards/lean-stack/run_20260916_132557/eval_final.json` (2026-09-16, few-shot, unknown)

| Metric | A | B | Δ (B−A) | Interpretation |
|---|---|---|---|---|
| ROUGE-L | 0.9430 | 0.9634 | +0.0204 | above noise threshold |
| Valid ROUGE-L | 0.9430 | 0.9634 | +0.0204 | above noise threshold |
| Exact Match | 0.6053 | 0.7346 | +0.1293 | above noise threshold |
| Non-copy Tok Acc | 0.9067 | 0.9488 | +0.0421 | above noise threshold |
| Pass@1 | 0.9985 | 0.9985 | +0.0000 | NOT GENERALIZABLE (|delta| < 0.02): reproducible (training is bit-deterministic at fixed seed) but single-seed, so it may invert under another seed — do NOT interpret |
| Validity | 1.0000 | 1.0000 | +0.0000 | NOT GENERALIZABLE (|delta| < 0.02): reproducible (training is bit-deterministic at fixed seed) but single-seed, so it may invert under another seed — do NOT interpret |
| BLEU (corpus) | 0.8803 | 0.9228 | +0.0425 | above noise threshold |
| BLEU (sent) | 0.8750 | 0.9163 | +0.0413 | above noise threshold |
| chrF2 (corpus) | 93.1646 | 96.0506 | +2.8859 | scale 0-100: 0.02 noise threshold not applicable |
| chrF2 (sent) | 93.4710 | 96.0134 | +2.5424 | scale 0-100: 0.02 noise threshold not applicable |
| Gloss F1 (mic) | 0.9466 | 0.9653 | +0.0187 | NOT GENERALIZABLE (|delta| < 0.02): reproducible (training is bit-deterministic at fixed seed) but single-seed, so it may invert under another seed — do NOT interpret |
| Gloss F1 (sent) | 0.9469 | 0.9643 | +0.0175 | NOT GENERALIZABLE (|delta| < 0.02): reproducible (training is bit-deterministic at fixed seed) but single-seed, so it may invert under another seed — do NOT interpret |
- ⚠️ cross-run comparison: checkpoint identity NOT guaranteed
- ⚠️ factors not deducible on at least one side, equality NOT verified: reward_stack, rl_objective

**prompting (cross-run)** — `qwen25-05b/ablations/objectives/sft-structured-shuffled/run_20260925_192053/eval_final.json` (2026-09-25, zero-shot, unknown) → `qwen25-05b/ablations/rewards/edit-validity/run_20260915_160558/eval_final.json` (2026-09-15, few-shot, unknown)

| Metric | A | B | Δ (B−A) | Interpretation |
|---|---|---|---|---|
| ROUGE-L | 0.9464 | 0.6373 | -0.3091 | above noise threshold |
| Valid ROUGE-L | 0.9464 | 0.6350 | -0.3114 | above noise threshold |
| Exact Match | 0.6078 | 0.0608 | -0.5470 | above noise threshold |
| Non-copy Tok Acc | 0.9089 | 0.4460 | -0.4629 | above noise threshold |
| Pass@1 | 0.9975 | 0.9425 | -0.0550 | above noise threshold |
| Validity | 1.0000 | 0.9964 | -0.0036 | NOT GENERALIZABLE (|delta| < 0.02): reproducible (training is bit-deterministic at fixed seed) but single-seed, so it may invert under another seed — do NOT interpret |
| BLEU (corpus) | 0.8863 | 0.3684 | -0.5179 | above noise threshold |
| BLEU (sent) | 0.8822 | 0.3521 | -0.5301 | above noise threshold |
| chrF2 (corpus) | 93.6728 | 58.3563 | -35.3165 | scale 0-100: 0.02 noise threshold not applicable |
| chrF2 (sent) | 93.8950 | 60.5965 | -33.2985 | scale 0-100: 0.02 noise threshold not applicable |
| Gloss F1 (mic) | 0.9474 | 0.6432 | -0.3043 | above noise threshold |
| Gloss F1 (sent) | 0.9477 | 0.6625 | -0.2852 | above noise threshold |
- ⚠️ cross-run comparison: checkpoint identity NOT guaranteed
- ⚠️ factors not deducible on at least one side, equality NOT verified: reward_stack, rl_objective

**prompting (cross-run)** — `qwen25-05b/ablations/decoding/hot-rollout/run_20260915_083119/eval_final.json` (2026-09-15, few-shot, unknown) → `qwen25-05b/sft/zero-shot/run_20260914_220146/decoding-greedy/eval_final.json` (2026-09-14, zero-shot, unknown)

| Metric | A | B | Δ (B−A) | Interpretation |
|---|---|---|---|---|
| ROUGE-L | 0.9634 | 0.9696 | +0.0062 | NOT GENERALIZABLE (|delta| < 0.02): reproducible (training is bit-deterministic at fixed seed) but single-seed, so it may invert under another seed — do NOT interpret |
| Valid ROUGE-L | 0.9633 | 0.9696 | +0.0063 | NOT GENERALIZABLE (|delta| < 0.02): reproducible (training is bit-deterministic at fixed seed) but single-seed, so it may invert under another seed — do NOT interpret |
| Exact Match | 0.7419 | 0.7775 | +0.0356 | above noise threshold |
| Non-copy Tok Acc | 0.9500 | 0.9547 | +0.0047 | NOT GENERALIZABLE (|delta| < 0.02): reproducible (training is bit-deterministic at fixed seed) but single-seed, so it may invert under another seed — do NOT interpret |
| Pass@1 | 0.9965 | 0.9980 | +0.0015 | NOT GENERALIZABLE (|delta| < 0.02): reproducible (training is bit-deterministic at fixed seed) but single-seed, so it may invert under another seed — do NOT interpret |
| Validity | 0.9999 | 1.0000 | +0.0001 | NOT GENERALIZABLE (|delta| < 0.02): reproducible (training is bit-deterministic at fixed seed) but single-seed, so it may invert under another seed — do NOT interpret |
| BLEU (corpus) | 0.9185 | 0.9362 | +0.0177 | NOT GENERALIZABLE (|delta| < 0.02): reproducible (training is bit-deterministic at fixed seed) but single-seed, so it may invert under another seed — do NOT interpret |
| BLEU (sent) | 0.9173 | 0.9319 | +0.0146 | NOT GENERALIZABLE (|delta| < 0.02): reproducible (training is bit-deterministic at fixed seed) but single-seed, so it may invert under another seed — do NOT interpret |
| chrF2 (corpus) | 96.1205 | 96.6053 | +0.4848 | scale 0-100: 0.02 noise threshold not applicable |
| chrF2 (sent) | 96.1337 | 96.6285 | +0.4948 | scale 0-100: 0.02 noise threshold not applicable |
| Gloss F1 (mic) | 0.9641 | 0.9721 | +0.0080 | NOT GENERALIZABLE (|delta| < 0.02): reproducible (training is bit-deterministic at fixed seed) but single-seed, so it may invert under another seed — do NOT interpret |
| Gloss F1 (sent) | 0.9639 | 0.9715 | +0.0077 | NOT GENERALIZABLE (|delta| < 0.02): reproducible (training is bit-deterministic at fixed seed) but single-seed, so it may invert under another seed — do NOT interpret |
- ⚠️ cross-run comparison: checkpoint identity NOT guaranteed
- ⚠️ factors not deducible on at least one side, equality NOT verified: reward_stack, rl_objective

**prompting (cross-run)** — `qwen25-05b/ablations/decoding/no-grammar/run_20260915_052556/eval_final.json` (2026-09-15, few-shot, unknown) → `qwen25-05b/baseline/zero-shot-no-grammar/run_20260912_165549/eval_zero_shot.json` (2026-09-12, zero-shot, unknown)

| Metric | A | B | Δ (B−A) | Interpretation |
|---|---|---|---|---|
| ROUGE-L | 0.7590 | 0.3648 | -0.3942 | above noise threshold |
| Valid ROUGE-L | 0.7513 | 0.0135 | -0.7378 | above noise threshold |
| Exact Match | 0.0604 | 0.0007 | -0.0597 | above noise threshold |
| Non-copy Tok Acc | 0.4730 | 0.0007 | -0.4723 | above noise threshold |
| Pass@1 | 0.9905 | 0.5845 | -0.4060 | above noise threshold |
| Validity | 0.9899 | 0.0370 | -0.9529 | above noise threshold |
| BLEU (corpus) | 0.4435 | 0.0008 | -0.4427 | above noise threshold |
| BLEU (sent) | 0.4304 | 0.0068 | -0.4236 | above noise threshold |
| chrF2 (corpus) | 69.5087 | 1.1814 | -68.3273 | scale 0-100: 0.02 noise threshold not applicable |
| chrF2 (sent) | 70.9738 | 1.3574 | -69.6163 | scale 0-100: 0.02 noise threshold not applicable |
| Gloss F1 (mic) | 0.7175 | 0.2279 | -0.4896 | above noise threshold |
| Gloss F1 (sent) | 0.7321 | 0.2356 | -0.4965 | above noise threshold |
- ⚠️ cross-run comparison: checkpoint identity NOT guaranteed
- ⚠️ factors not deducible on at least one side, equality NOT verified: reward_stack, rl_objective

**prompting (cross-run)** — `qwen25-05b/baseline/few-shot/run_20260912_171054/eval_zero_shot.json` (2026-09-12, few-shot, unknown) → `qwen25-05b/baseline/zero-shot/run_20260912_144136/eval_zero_shot.json` (2026-09-12, zero-shot, unknown)

| Metric | A | B | Δ (B−A) | Interpretation |
|---|---|---|---|---|
| ROUGE-L | 0.4664 | 0.1373 | -0.3291 | above noise threshold |
| Valid ROUGE-L | 0.4581 | 0.1243 | -0.3338 | above noise threshold |
| Exact Match | 0.0148 | 0.0017 | -0.0131 | NOT GENERALIZABLE (|delta| < 0.02): reproducible (training is bit-deterministic at fixed seed) but single-seed, so it may invert under another seed — do NOT interpret |
| Non-copy Tok Acc | 0.3555 | 0.0451 | -0.3104 | above noise threshold |
| Pass@1 | 0.7330 | 0.1820 | -0.5510 | above noise threshold |
| Validity | 0.9821 | 0.9055 | -0.0766 | above noise threshold |
| BLEU (corpus) | 0.1880 | 0.0270 | -0.1610 | above noise threshold |
| BLEU (sent) | 0.1882 | 0.0304 | -0.1578 | above noise threshold |
| chrF2 (corpus) | 41.8895 | 13.2069 | -28.6826 | scale 0-100: 0.02 noise threshold not applicable |
| chrF2 (sent) | 44.6817 | 13.1640 | -31.5177 | scale 0-100: 0.02 noise threshold not applicable |
| Gloss F1 (mic) | 0.4026 | 0.1186 | -0.2840 | above noise threshold |
| Gloss F1 (sent) | 0.4439 | 0.1416 | -0.3023 | above noise threshold |
- ⚠️ cross-run comparison: checkpoint identity NOT guaranteed
- ⚠️ factors not deducible on at least one side, equality NOT verified: reward_stack, rl_objective

### Factor: `grammar`

**No paired comparison available.** 6 group(s) with 2+ runs but no pair with a known, differing value of this factor


## Missing paired comparisons (declared, not fabricated)

- **`grammar`**: 6 group(s) with 2+ runs but no pair with a known, differing value of this factor.
  Available typologies: `g|r|a|m|m|a|r|=|u|n|k|n|o|w|n`

## Figures

- `campaign_pairwise_deltas.png` — one panel per factor: paired deltas as diverging bars with the ±0.02 noise band shaded. Diverging bars make the SIGN of each effect immediately visible, and the shaded band makes it impossible to mistake sub-noise deltas for results.
- `campaign_matrix.png` — the ablation matrix as a heatmap (rows: method/variant, columns: prompting, annotated with ROUGE-L); empty typologies stay visibly grey so gaps in the grid are impossible to miss.

