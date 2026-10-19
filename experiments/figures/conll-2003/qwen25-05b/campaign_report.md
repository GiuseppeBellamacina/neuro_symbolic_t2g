# Campaign Report — conll-2003 / qwen25-05b

Generated: 2026-10-19T17:43:44+00:00

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
   - [method] conll-2003/qwen25-05b/ablations/decoding/no-grammar/run_20261009_133123/eval_baseline.json vs conll-2003/qwen25-05b/sft-grpo/few-shot/run_20261019_002102/eval_final.json: prompt_context_fingerprint DIFFERS: the eval contexts (prompting/grammar/num_samples/model) are not identical — this pair may differ in more than the method factor
   - [method] conll-2003/qwen25-05b/grpo/few-shot/run_20261009_052104/eval_baseline.json vs conll-2003/qwen25-05b/sft-grpo/zero-shot/run_20261019_022555/eval_final__few-shot.json: prompt_context_fingerprint DIFFERS: the eval contexts (prompting/grammar/num_samples/model) are not identical — this pair may differ in more than the method factor
   - [method] conll-2003/qwen25-05b/ablations/decoding/no-grammar-zero-shot/run_20261009_162603/eval_baseline.json vs conll-2003/qwen25-05b/sft-grpo/few-shot/run_20261019_002102/eval_final__zero-shot.json: prompt_context_fingerprint DIFFERS: the eval contexts (prompting/grammar/num_samples/model) are not identical — this pair may differ in more than the method factor
   - [method] conll-2003/qwen25-05b/grpo/few-shot/run_20261009_052104/eval_baseline__zero-shot.json vs conll-2003/qwen25-05b/sft-grpo/zero-shot/run_20261019_022555/eval_final.json: prompt_context_fingerprint DIFFERS: the eval contexts (prompting/grammar/num_samples/model) are not identical — this pair may differ in more than the method factor

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
| `conll-2003/qwen25-05b/ablations/decoding/no-grammar-zero-shot/run_20261019_071102/eval_final.json` | unknown | conll-2003-qwen25-05b-ablations-decoding-no-grammar-zero-shot | zero-shot | unknown | 2026-10-19 | 2000 | 2 |
| `conll-2003/qwen25-05b/ablations/decoding/no-grammar/run_20261009_133123/eval_baseline.json` | baseline | conll-2003-qwen25-05b-ablations-decoding-no-grammar | few-shot | unknown | 2026-10-09 | 2000 | 2 |
| `conll-2003/qwen25-05b/ablations/decoding/no-grammar/run_20261019_043117/eval_final.json` | unknown | conll-2003-qwen25-05b-ablations-decoding-no-grammar | few-shot | unknown | 2026-10-19 | 2000 | 2 |
| `conll-2003/qwen25-05b/ablations/objectives/sft-allowed-mass/run_20261019_093124/eval_final.json` | unknown | conll-2003-qwen25-05b-ablations-objectives-sft-allowed-mass | zero-shot | unknown | 2026-10-19 | 2000 | 2 |
| `conll-2003/qwen25-05b/baseline/few-shot/run_20261018_161555/eval_zero_shot.json` | unknown | conll-2003-qwen25-05b-baseline-few-shot | few-shot | unknown | 2026-10-18 | 2000 | 2 |
| `conll-2003/qwen25-05b/baseline/zero-shot-no-grammar/run_20261018_163110/eval_zero_shot.json` | unknown | conll-2003-qwen25-05b-baseline-zero-shot-no-grammar | zero-shot | unknown | 2026-10-18 | 2000 | 2 |
| `conll-2003/qwen25-05b/baseline/zero-shot/run_20261018_155358/eval_zero_shot.json` | unknown | conll-2003-qwen25-05b-baseline-zero-shot | zero-shot | unknown | 2026-10-18 | 2000 | 2 |
| `conll-2003/qwen25-05b/grpo/few-shot/run_20261009_052104/eval_baseline.json` | baseline | conll-2003-qwen25-05b-grpo-few-shot | few-shot | unknown | 2026-10-09 | 2000 | 2 |
| `conll-2003/qwen25-05b/grpo/few-shot/run_20261009_052104/eval_baseline__zero-shot.json` | baseline | conll-2003-qwen25-05b-grpo-few-shot | zero-shot | unknown | 2026-10-09 | 2000 | 2 |
| `conll-2003/qwen25-05b/grpo/few-shot/run_20261018_212554/eval_final.json` | unknown | conll-2003-qwen25-05b-grpo-few-shot | few-shot | unknown | 2026-10-18 | 2000 | 2 |
| `conll-2003/qwen25-05b/grpo/few-shot/run_20261018_212554/eval_final__zero-shot.json` | unknown | conll-2003-qwen25-05b-grpo-few-shot | zero-shot | unknown | 2026-10-18 | 2000 | 2 |
| `conll-2003/qwen25-05b/grpo/zero-shot/run_20261018_174556/eval_final.json` | unknown | conll-2003-qwen25-05b-grpo-zero-shot | zero-shot | unknown | 2026-10-18 | 2000 | 2 |
| `conll-2003/qwen25-05b/grpo/zero-shot/run_20261018_174556/eval_final__few-shot.json` | unknown | conll-2003-qwen25-05b-grpo-zero-shot | few-shot | unknown | 2026-10-18 | 2000 | 2 |
| `conll-2003/qwen25-05b/sft-grpo/few-shot/run_20261019_002102/eval_final.json` | sft-grpo | few-shot | few-shot | unknown | 2026-10-19 | 2000 | 2 |
| `conll-2003/qwen25-05b/sft-grpo/few-shot/run_20261019_002102/eval_final__zero-shot.json` | sft-grpo | few-shot | zero-shot | unknown | 2026-10-19 | 2000 | 2 |
| `conll-2003/qwen25-05b/sft-grpo/zero-shot/run_20261019_022555/eval_final.json` | sft-grpo | zero-shot | zero-shot | unknown | 2026-10-19 | 2000 | 2 |
| `conll-2003/qwen25-05b/sft-grpo/zero-shot/run_20261019_022555/eval_final__few-shot.json` | sft-grpo | zero-shot | few-shot | unknown | 2026-10-19 | 2000 | 2 |
| `conll-2003/qwen25-05b/sft/zero-shot/run_20261009_004112/eval_baseline.json` | baseline | conll-2003-qwen25-05b-sft-zero-shot | zero-shot | unknown | 2026-10-09 | 2000 | 2 |
| `conll-2003/qwen25-05b/sft/zero-shot/run_20261009_004112/eval_baseline__few-shot.json` | baseline | conll-2003-qwen25-05b-sft-zero-shot | few-shot | unknown | 2026-10-09 | 2000 | 2 |
| `conll-2003/qwen25-05b/sft/zero-shot/run_20261018_165609/eval_final.json` | unknown | conll-2003-qwen25-05b-sft-zero-shot | zero-shot | unknown | 2026-10-18 | 2000 | 2 |
| `conll-2003/qwen25-05b/sft/zero-shot/run_20261018_165609/eval_final__few-shot.json` | unknown | conll-2003-qwen25-05b-sft-zero-shot | few-shot | unknown | 2026-10-18 | 2000 | 2 |

Superseded (older runs of the same typology): 32 (see JSON for the full list).

## Ablation matrix overview — ROUGE-L

One row per cell (its path under `results/`), one column per prompting mode that has at least one value. `—` = the cell was not evaluated in that mode. `*` = multiple runs collapsed, latest shown. Each cell's own `eval_baseline` (the base model in that cell's context) is not a row here: it repeats the `baseline/` cells and stays in the paired comparisons below.

| cell | zero-shot | few-shot |
|---|---|---|
| conll-2003/qwen25-05b/baseline/few-shot | — | 0.3823 |
| conll-2003/qwen25-05b/baseline/zero-shot | 0.2722 | — |
| conll-2003/qwen25-05b/baseline/zero-shot-no-grammar | 0.2676 | — |
| conll-2003/qwen25-05b/sft/zero-shot | 0.9005 | 0.8189 |
| conll-2003/qwen25-05b/sft-grpo/few-shot | 0.2509 | 0.5282 |
| conll-2003/qwen25-05b/sft-grpo/zero-shot | 0.2836 | 0.4395 |
| conll-2003/qwen25-05b/grpo/few-shot | 0.2522 | 0.5396 |
| conll-2003/qwen25-05b/grpo/zero-shot | 0.3167 | 0.4117 |
| conll-2003/qwen25-05b/ablations/decoding/no-grammar | — | 0.6113 |
| conll-2003/qwen25-05b/ablations/decoding/no-grammar-zero-shot | 0.4027 | — |
| conll-2003/qwen25-05b/ablations/objectives/sft-allowed-mass | 0.9044 | — |

## conll-2003 — all metrics

Same rows as the matrix above, restricted to this dataset, one column per metric. On CoNLL-2003 and WOS-46985 read Exact Match and Gloss F1 (micro = entity-level F1 on CoNLL): ROUGE-L and BLEU split `PER:Mary` into `per` + `mary` and give partial credit to a wrong label of the same type.

| cell | prompting | ROUGE-L | Valid ROUGE-L | Exact Match | Non-copy Tok Acc | Pass@1 | Validity | BLEU (corpus) | BLEU (sent) | chrF2 (corpus) | chrF2 (sent) | Gloss F1 (mic) | Gloss F1 (sent) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| conll-2003/qwen25-05b/baseline/few-shot | few-shot | 0.3823 | 0.3820 | 0.0714 | 0.2828 | 0.5825 | 0.9992 | 0.1929 | 0.2246 | 43.9497 | 42.2711 | 0.1845 | 0.2206 |
| conll-2003/qwen25-05b/baseline/zero-shot | zero-shot | 0.2722 | 0.2718 | 0.0445 | 0.1298 | 0.4115 | 0.9987 | 0.1147 | 0.1320 | 32.3153 | 30.1515 | 0.0816 | 0.0975 |
| conll-2003/qwen25-05b/baseline/zero-shot-no-grammar | zero-shot | 0.2676 | 0.0020 | 0.0011 | 0.0066 | 0.4260 | 0.0075 | 0.0089 | 0.0219 | 17.9933 | 18.2289 | 0.0022 | 0.0056 |
| conll-2003/qwen25-05b/sft/zero-shot | zero-shot | 0.9005 | 0.9005 | 0.7448 | 0.8258 | 0.9670 | 1.0000 | 0.8456 | 0.8604 | 86.1363 | 89.4521 | 0.8255 | 0.8425 |
| conll-2003/qwen25-05b/sft/zero-shot | few-shot | 0.8189 | 0.8189 | 0.6003 | 0.7380 | 0.9290 | 1.0000 | 0.7279 | 0.7530 | 77.9801 | 81.8983 | 0.7230 | 0.7485 |
| conll-2003/qwen25-05b/sft-grpo/few-shot | zero-shot | 0.2509 | 0.2508 | 0.1285 | 0.1138 | 0.3150 | 0.9997 | 0.1437 | 0.1846 | 22.6189 | 26.4272 | 0.1022 | 0.1504 |
| conll-2003/qwen25-05b/sft-grpo/few-shot | few-shot | 0.5282 | 0.5280 | 0.2871 | 0.3845 | 0.6760 | 0.9995 | 0.3326 | 0.4221 | 46.5412 | 53.6112 | 0.3392 | 0.4212 |
| conll-2003/qwen25-05b/sft-grpo/zero-shot | zero-shot | 0.2836 | 0.2832 | 0.1122 | 0.1509 | 0.3840 | 0.9987 | 0.1202 | 0.1852 | 29.3123 | 31.2690 | 0.1026 | 0.1552 |
| conll-2003/qwen25-05b/sft-grpo/zero-shot | few-shot | 0.4395 | 0.4389 | 0.1445 | 0.3858 | 0.6325 | 0.9987 | 0.2053 | 0.2954 | 46.6595 | 49.1477 | 0.2468 | 0.3285 |
| conll-2003/qwen25-05b/grpo/few-shot | zero-shot | 0.2522 | 0.2520 | 0.1324 | 0.1348 | 0.3110 | 0.9995 | 0.1152 | 0.1811 | 21.8568 | 28.4721 | 0.1104 | 0.1648 |
| conll-2003/qwen25-05b/grpo/few-shot | few-shot | 0.5396 | 0.5394 | 0.3247 | 0.4067 | 0.6745 | 0.9997 | 0.3949 | 0.4515 | 48.4686 | 54.3000 | 0.3776 | 0.4457 |
| conll-2003/qwen25-05b/grpo/zero-shot | zero-shot | 0.3167 | 0.3166 | 0.0296 | 0.1963 | 0.5015 | 0.9998 | 0.1818 | 0.1729 | 40.9877 | 36.7997 | 0.1298 | 0.1284 |
| conll-2003/qwen25-05b/grpo/zero-shot | few-shot | 0.4117 | 0.4117 | 0.0819 | 0.3388 | 0.6270 | 0.9999 | 0.2440 | 0.2604 | 49.6694 | 46.2937 | 0.2353 | 0.2576 |
| conll-2003/qwen25-05b/ablations/decoding/no-grammar | few-shot | 0.6113 | 0.3974 | 0.3476 | 0.4229 | 0.7550 | 0.6501 | 0.4346 | 0.4840 | 50.5409 | 58.4925 | 0.3776 | 0.4749 |
| conll-2003/qwen25-05b/ablations/decoding/no-grammar-zero-shot | zero-shot | 0.4027 | 0.1843 | 0.1472 | 0.1104 | 0.5735 | 0.4578 | 0.0083 | 0.1717 | 21.3288 | 31.1467 | 0.0732 | 0.1693 |
| conll-2003/qwen25-05b/ablations/objectives/sft-allowed-mass | zero-shot | 0.9044 | 0.9044 | 0.7505 | 0.8253 | 0.9695 | 1.0000 | 0.8407 | 0.8646 | 85.7212 | 89.7119 | 0.8253 | 0.8514 |

## Paired comparisons — one table per factor

### Factor: `method`

**method effect (cross-run)** — `conll-2003/qwen25-05b/ablations/decoding/no-grammar/run_20261009_133123/eval_baseline.json` (2026-10-09, few-shot, unknown) → `conll-2003/qwen25-05b/sft-grpo/few-shot/run_20261019_002102/eval_final.json` (2026-10-19, few-shot, unknown)

| Metric | A | B | Δ (B−A) | Interpretation |
|---|---|---|---|---|
| ROUGE-L | 0.3700 | 0.5282 | +0.1582 | above noise threshold |
| Valid ROUGE-L | 0.0635 | 0.5280 | +0.4645 | above noise threshold |
| Exact Match | 0.0344 | 0.2871 | +0.2527 | above noise threshold |
| Non-copy Tok Acc | 0.1922 | 0.3845 | +0.1923 | above noise threshold |
| Pass@1 | 0.5440 | 0.6760 | +0.1320 | above noise threshold |
| Validity | 0.1716 | 0.9995 | +0.8279 | above noise threshold |
| BLEU (corpus) | 0.0961 | 0.3326 | +0.2365 | above noise threshold |
| BLEU (sent) | 0.1556 | 0.4221 | +0.2665 | above noise threshold |
| chrF2 (corpus) | 33.2195 | 46.5412 | +13.3217 | scale 0-100: 0.02 noise threshold not applicable |
| chrF2 (sent) | 35.9071 | 53.6112 | +17.7041 | scale 0-100: 0.02 noise threshold not applicable |
| Gloss F1 (mic) | 0.0839 | 0.3392 | +0.2552 | above noise threshold |
| Gloss F1 (sent) | 0.1436 | 0.4212 | +0.2776 | above noise threshold |
- ⚠️ factors not deducible on at least one side, equality NOT verified: reward_stack, rl_objective
- ⚠️ prompt_context_fingerprint DIFFERS: the eval contexts (prompting/grammar/num_samples/model) are not identical — this pair may differ in more than the method factor

**method effect (cross-run)** — `conll-2003/qwen25-05b/grpo/few-shot/run_20261009_052104/eval_baseline.json` (2026-10-09, few-shot, unknown) → `conll-2003/qwen25-05b/sft-grpo/zero-shot/run_20261019_022555/eval_final__few-shot.json` (2026-10-19, few-shot, unknown)

| Metric | A | B | Δ (B−A) | Interpretation |
|---|---|---|---|---|
| ROUGE-L | 0.3805 | 0.4395 | +0.0590 | above noise threshold |
| Valid ROUGE-L | 0.3706 | 0.4389 | +0.0684 | above noise threshold |
| Exact Match | 0.0662 | 0.1445 | +0.0783 | above noise threshold |
| Non-copy Tok Acc | 0.3169 | 0.3858 | +0.0689 | above noise threshold |
| Pass@1 | 0.5865 | 0.6325 | +0.0460 | above noise threshold |
| Validity | 0.9739 | 0.9987 | +0.0248 | above noise threshold |
| BLEU (corpus) | 0.1429 | 0.2053 | +0.0624 | above noise threshold |
| BLEU (sent) | 0.2204 | 0.2954 | +0.0750 | above noise threshold |
| chrF2 (corpus) | 39.8261 | 46.6595 | +6.8335 | scale 0-100: 0.02 noise threshold not applicable |
| chrF2 (sent) | 41.4512 | 49.1477 | +7.6965 | scale 0-100: 0.02 noise threshold not applicable |
| Gloss F1 (mic) | 0.1714 | 0.2468 | +0.0754 | above noise threshold |
| Gloss F1 (sent) | 0.2366 | 0.3285 | +0.0919 | above noise threshold |
- ⚠️ factors not deducible on at least one side, equality NOT verified: reward_stack, rl_objective
- ⚠️ prompt_context_fingerprint DIFFERS: the eval contexts (prompting/grammar/num_samples/model) are not identical — this pair may differ in more than the method factor

**method effect (cross-run)** — `conll-2003/qwen25-05b/ablations/decoding/no-grammar-zero-shot/run_20261009_162603/eval_baseline.json` (2026-10-09, zero-shot, unknown) → `conll-2003/qwen25-05b/sft-grpo/few-shot/run_20261019_002102/eval_final__zero-shot.json` (2026-10-19, zero-shot, unknown)

| Metric | A | B | Δ (B−A) | Interpretation |
|---|---|---|---|---|
| ROUGE-L | 0.2676 | 0.2509 | -0.0167 | NOT GENERALIZABLE (|delta| < 0.02): reproducible (training is bit-deterministic at fixed seed) but single-seed, so it may invert under another seed — do NOT interpret |
| Valid ROUGE-L | 0.0021 | 0.2508 | +0.2487 | above noise threshold |
| Exact Match | 0.0011 | 0.1285 | +0.1274 | above noise threshold |
| Non-copy Tok Acc | 0.0066 | 0.1138 | +0.1073 | above noise threshold |
| Pass@1 | 0.4260 | 0.3150 | -0.1110 | above noise threshold |
| Validity | 0.0079 | 0.9997 | +0.9918 | above noise threshold |
| BLEU (corpus) | 0.0089 | 0.1437 | +0.1348 | above noise threshold |
| BLEU (sent) | 0.0219 | 0.1846 | +0.1627 | above noise threshold |
| chrF2 (corpus) | 17.9933 | 22.6189 | +4.6256 | scale 0-100: 0.02 noise threshold not applicable |
| chrF2 (sent) | 18.2289 | 26.4272 | +8.1983 | scale 0-100: 0.02 noise threshold not applicable |
| Gloss F1 (mic) | 0.0022 | 0.1022 | +0.1001 | above noise threshold |
| Gloss F1 (sent) | 0.0056 | 0.1504 | +0.1449 | above noise threshold |
- ⚠️ factors not deducible on at least one side, equality NOT verified: reward_stack, rl_objective
- ⚠️ prompt_context_fingerprint DIFFERS: the eval contexts (prompting/grammar/num_samples/model) are not identical — this pair may differ in more than the method factor

**method effect (cross-run)** — `conll-2003/qwen25-05b/grpo/few-shot/run_20261009_052104/eval_baseline__zero-shot.json` (2026-10-09, zero-shot, unknown) → `conll-2003/qwen25-05b/sft-grpo/zero-shot/run_20261019_022555/eval_final.json` (2026-10-19, zero-shot, unknown)

| Metric | A | B | Δ (B−A) | Interpretation |
|---|---|---|---|---|
| ROUGE-L | 0.2708 | 0.2836 | +0.0127 | NOT GENERALIZABLE (|delta| < 0.02): reproducible (training is bit-deterministic at fixed seed) but single-seed, so it may invert under another seed — do NOT interpret |
| Valid ROUGE-L | 0.2625 | 0.2832 | +0.0207 | above noise threshold |
| Exact Match | 0.0453 | 0.1122 | +0.0669 | above noise threshold |
| Non-copy Tok Acc | 0.1458 | 0.1509 | +0.0051 | NOT GENERALIZABLE (|delta| < 0.02): reproducible (training is bit-deterministic at fixed seed) but single-seed, so it may invert under another seed — do NOT interpret |
| Pass@1 | 0.4145 | 0.3840 | -0.0305 | above noise threshold |
| Validity | 0.9693 | 0.9987 | +0.0294 | above noise threshold |
| BLEU (corpus) | 0.0893 | 0.1202 | +0.0309 | above noise threshold |
| BLEU (sent) | 0.1329 | 0.1852 | +0.0523 | above noise threshold |
| chrF2 (corpus) | 29.5092 | 29.3123 | -0.1969 | scale 0-100: 0.02 noise threshold not applicable |
| chrF2 (sent) | 29.5764 | 31.2690 | +1.6926 | scale 0-100: 0.02 noise threshold not applicable |
| Gloss F1 (mic) | 0.0762 | 0.1026 | +0.0264 | above noise threshold |
| Gloss F1 (sent) | 0.1033 | 0.1552 | +0.0519 | above noise threshold |
- ⚠️ factors not deducible on at least one side, equality NOT verified: reward_stack, rl_objective
- ⚠️ prompt_context_fingerprint DIFFERS: the eval contexts (prompting/grammar/num_samples/model) are not identical — this pair may differ in more than the method factor

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

🔴 **prompt-dependence (dual eval, same checkpoint)** — `conll-2003/qwen25-05b/sft-grpo/zero-shot/run_20261019_022555/eval_final__few-shot.json` (2026-10-19, few-shot, unknown) → `conll-2003/qwen25-05b/sft-grpo/zero-shot/run_20261019_022555/eval_final.json` (2026-10-19, zero-shot, unknown)

| Metric | A | B | Δ (B−A) | Interpretation |
|---|---|---|---|---|
| ROUGE-L | 0.4395 | 0.2836 | -0.1559 | above noise threshold |
| Valid ROUGE-L | 0.4389 | 0.2832 | -0.1557 | above noise threshold |
| Exact Match | 0.1445 | 0.1122 | -0.0323 | above noise threshold |
| Non-copy Tok Acc | 0.3858 | 0.1509 | -0.2349 | above noise threshold |
| Pass@1 | 0.6325 | 0.3840 | -0.2485 | above noise threshold |
| Validity | 0.9987 | 0.9987 | +0.0000 | NOT GENERALIZABLE (|delta| < 0.02): reproducible (training is bit-deterministic at fixed seed) but single-seed, so it may invert under another seed — do NOT interpret |
| BLEU (corpus) | 0.2053 | 0.1202 | -0.0851 | above noise threshold |
| BLEU (sent) | 0.2954 | 0.1852 | -0.1101 | above noise threshold |
| chrF2 (corpus) | 46.6595 | 29.3123 | -17.3473 | scale 0-100: 0.02 noise threshold not applicable |
| chrF2 (sent) | 49.1477 | 31.2690 | -17.8787 | scale 0-100: 0.02 noise threshold not applicable |
| Gloss F1 (mic) | 0.2468 | 0.1026 | -0.1442 | above noise threshold |
| Gloss F1 (sent) | 0.3285 | 0.1552 | -0.1733 | above noise threshold |
- ⚠️ HIGHLIGHT: same model evaluated in both prompting modes — how much does it depend on the prompt as a crutch?
- ⚠️ factors not deducible on at least one side, equality NOT verified: reward_stack, rl_objective

🔴 **prompt-dependence (dual eval, same checkpoint)** — `conll-2003/qwen25-05b/sft-grpo/few-shot/run_20261019_002102/eval_final.json` (2026-10-19, few-shot, unknown) → `conll-2003/qwen25-05b/sft-grpo/few-shot/run_20261019_002102/eval_final__zero-shot.json` (2026-10-19, zero-shot, unknown)

| Metric | A | B | Δ (B−A) | Interpretation |
|---|---|---|---|---|
| ROUGE-L | 0.5282 | 0.2509 | -0.2773 | above noise threshold |
| Valid ROUGE-L | 0.5280 | 0.2508 | -0.2772 | above noise threshold |
| Exact Match | 0.2871 | 0.1285 | -0.1586 | above noise threshold |
| Non-copy Tok Acc | 0.3845 | 0.1138 | -0.2707 | above noise threshold |
| Pass@1 | 0.6760 | 0.3150 | -0.3610 | above noise threshold |
| Validity | 0.9995 | 0.9997 | +0.0002 | NOT GENERALIZABLE (|delta| < 0.02): reproducible (training is bit-deterministic at fixed seed) but single-seed, so it may invert under another seed — do NOT interpret |
| BLEU (corpus) | 0.3326 | 0.1437 | -0.1889 | above noise threshold |
| BLEU (sent) | 0.4221 | 0.1846 | -0.2376 | above noise threshold |
| chrF2 (corpus) | 46.5412 | 22.6189 | -23.9223 | scale 0-100: 0.02 noise threshold not applicable |
| chrF2 (sent) | 53.6112 | 26.4272 | -27.1840 | scale 0-100: 0.02 noise threshold not applicable |
| Gloss F1 (mic) | 0.3392 | 0.1022 | -0.2369 | above noise threshold |
| Gloss F1 (sent) | 0.4212 | 0.1504 | -0.2708 | above noise threshold |
- ⚠️ HIGHLIGHT: same model evaluated in both prompting modes — how much does it depend on the prompt as a crutch?
- ⚠️ factors not deducible on at least one side, equality NOT verified: reward_stack, rl_objective

🔴 **prompt-dependence (dual eval, same checkpoint)** — `conll-2003/qwen25-05b/grpo/few-shot/run_20261018_212554/eval_final.json` (2026-10-18, few-shot, unknown) → `conll-2003/qwen25-05b/grpo/few-shot/run_20261018_212554/eval_final__zero-shot.json` (2026-10-18, zero-shot, unknown)

| Metric | A | B | Δ (B−A) | Interpretation |
|---|---|---|---|---|
| ROUGE-L | 0.5396 | 0.2522 | -0.2874 | above noise threshold |
| Valid ROUGE-L | 0.5394 | 0.2520 | -0.2874 | above noise threshold |
| Exact Match | 0.3247 | 0.1324 | -0.1923 | above noise threshold |
| Non-copy Tok Acc | 0.4067 | 0.1348 | -0.2719 | above noise threshold |
| Pass@1 | 0.6745 | 0.3110 | -0.3635 | above noise threshold |
| Validity | 0.9997 | 0.9995 | -0.0002 | NOT GENERALIZABLE (|delta| < 0.02): reproducible (training is bit-deterministic at fixed seed) but single-seed, so it may invert under another seed — do NOT interpret |
| BLEU (corpus) | 0.3949 | 0.1152 | -0.2797 | above noise threshold |
| BLEU (sent) | 0.4515 | 0.1811 | -0.2704 | above noise threshold |
| chrF2 (corpus) | 48.4686 | 21.8568 | -26.6118 | scale 0-100: 0.02 noise threshold not applicable |
| chrF2 (sent) | 54.3000 | 28.4721 | -25.8278 | scale 0-100: 0.02 noise threshold not applicable |
| Gloss F1 (mic) | 0.3776 | 0.1104 | -0.2673 | above noise threshold |
| Gloss F1 (sent) | 0.4457 | 0.1648 | -0.2810 | above noise threshold |
- ⚠️ HIGHLIGHT: same model evaluated in both prompting modes — how much does it depend on the prompt as a crutch?
- ⚠️ factors not deducible on at least one side, equality NOT verified: reward_stack, rl_objective

🔴 **prompt-dependence (dual eval, same checkpoint)** — `conll-2003/qwen25-05b/grpo/zero-shot/run_20261018_174556/eval_final__few-shot.json` (2026-10-18, few-shot, unknown) → `conll-2003/qwen25-05b/grpo/zero-shot/run_20261018_174556/eval_final.json` (2026-10-18, zero-shot, unknown)

| Metric | A | B | Δ (B−A) | Interpretation |
|---|---|---|---|---|
| ROUGE-L | 0.4117 | 0.3167 | -0.0950 | above noise threshold |
| Valid ROUGE-L | 0.4117 | 0.3166 | -0.0951 | above noise threshold |
| Exact Match | 0.0819 | 0.0296 | -0.0523 | above noise threshold |
| Non-copy Tok Acc | 0.3388 | 0.1963 | -0.1424 | above noise threshold |
| Pass@1 | 0.6270 | 0.5015 | -0.1255 | above noise threshold |
| Validity | 0.9999 | 0.9998 | -0.0001 | NOT GENERALIZABLE (|delta| < 0.02): reproducible (training is bit-deterministic at fixed seed) but single-seed, so it may invert under another seed — do NOT interpret |
| BLEU (corpus) | 0.2440 | 0.1818 | -0.0622 | above noise threshold |
| BLEU (sent) | 0.2604 | 0.1729 | -0.0874 | above noise threshold |
| chrF2 (corpus) | 49.6694 | 40.9877 | -8.6817 | scale 0-100: 0.02 noise threshold not applicable |
| chrF2 (sent) | 46.2937 | 36.7997 | -9.4940 | scale 0-100: 0.02 noise threshold not applicable |
| Gloss F1 (mic) | 0.2353 | 0.1298 | -0.1056 | above noise threshold |
| Gloss F1 (sent) | 0.2576 | 0.1284 | -0.1292 | above noise threshold |
- ⚠️ HIGHLIGHT: same model evaluated in both prompting modes — how much does it depend on the prompt as a crutch?
- ⚠️ factors not deducible on at least one side, equality NOT verified: reward_stack, rl_objective

🔴 **prompt-dependence (dual eval, same checkpoint)** — `conll-2003/qwen25-05b/sft/zero-shot/run_20261018_165609/eval_final__few-shot.json` (2026-10-18, few-shot, unknown) → `conll-2003/qwen25-05b/sft/zero-shot/run_20261018_165609/eval_final.json` (2026-10-18, zero-shot, unknown)

| Metric | A | B | Δ (B−A) | Interpretation |
|---|---|---|---|---|
| ROUGE-L | 0.8189 | 0.9005 | +0.0817 | above noise threshold |
| Valid ROUGE-L | 0.8189 | 0.9005 | +0.0817 | above noise threshold |
| Exact Match | 0.6003 | 0.7448 | +0.1445 | above noise threshold |
| Non-copy Tok Acc | 0.7380 | 0.8258 | +0.0878 | above noise threshold |
| Pass@1 | 0.9290 | 0.9670 | +0.0380 | above noise threshold |
| Validity | 1.0000 | 1.0000 | +0.0000 | NOT GENERALIZABLE (|delta| < 0.02): reproducible (training is bit-deterministic at fixed seed) but single-seed, so it may invert under another seed — do NOT interpret |
| BLEU (corpus) | 0.7279 | 0.8456 | +0.1176 | above noise threshold |
| BLEU (sent) | 0.7530 | 0.8604 | +0.1075 | above noise threshold |
| chrF2 (corpus) | 77.9801 | 86.1363 | +8.1562 | scale 0-100: 0.02 noise threshold not applicable |
| chrF2 (sent) | 81.8983 | 89.4521 | +7.5537 | scale 0-100: 0.02 noise threshold not applicable |
| Gloss F1 (mic) | 0.7230 | 0.8255 | +0.1025 | above noise threshold |
| Gloss F1 (sent) | 0.7485 | 0.8425 | +0.0940 | above noise threshold |
- ⚠️ HIGHLIGHT: same model evaluated in both prompting modes — how much does it depend on the prompt as a crutch?
- ⚠️ factors not deducible on at least one side, equality NOT verified: reward_stack, rl_objective

**prompting (cross-run)** — `conll-2003/qwen25-05b/ablations/decoding/no-grammar/run_20261019_043117/eval_final.json` (2026-10-19, few-shot, unknown) → `conll-2003/qwen25-05b/ablations/decoding/no-grammar-zero-shot/run_20261019_071102/eval_final.json` (2026-10-19, zero-shot, unknown)

| Metric | A | B | Δ (B−A) | Interpretation |
|---|---|---|---|---|
| ROUGE-L | 0.6113 | 0.4027 | -0.2087 | above noise threshold |
| Valid ROUGE-L | 0.3974 | 0.1843 | -0.2131 | above noise threshold |
| Exact Match | 0.3476 | 0.1472 | -0.2004 | above noise threshold |
| Non-copy Tok Acc | 0.4229 | 0.1104 | -0.3125 | above noise threshold |
| Pass@1 | 0.7550 | 0.5735 | -0.1815 | above noise threshold |
| Validity | 0.6501 | 0.4578 | -0.1923 | above noise threshold |
| BLEU (corpus) | 0.4346 | 0.0083 | -0.4263 | above noise threshold |
| BLEU (sent) | 0.4840 | 0.1717 | -0.3123 | above noise threshold |
| chrF2 (corpus) | 50.5409 | 21.3288 | -29.2121 | scale 0-100: 0.02 noise threshold not applicable |
| chrF2 (sent) | 58.4925 | 31.1467 | -27.3458 | scale 0-100: 0.02 noise threshold not applicable |
| Gloss F1 (mic) | 0.3776 | 0.0732 | -0.3044 | above noise threshold |
| Gloss F1 (sent) | 0.4749 | 0.1693 | -0.3056 | above noise threshold |
- ⚠️ cross-run comparison: checkpoint identity NOT guaranteed
- ⚠️ factors not deducible on at least one side, equality NOT verified: reward_stack, rl_objective

**prompting (cross-run)** — `conll-2003/qwen25-05b/ablations/objectives/sft-allowed-mass/run_20261019_093124/eval_final.json` (2026-10-19, zero-shot, unknown) → `conll-2003/qwen25-05b/baseline/few-shot/run_20261018_161555/eval_zero_shot.json` (2026-10-18, few-shot, unknown)

| Metric | A | B | Δ (B−A) | Interpretation |
|---|---|---|---|---|
| ROUGE-L | 0.9044 | 0.3823 | -0.5222 | above noise threshold |
| Valid ROUGE-L | 0.9044 | 0.3820 | -0.5225 | above noise threshold |
| Exact Match | 0.7505 | 0.0714 | -0.6791 | above noise threshold |
| Non-copy Tok Acc | 0.8253 | 0.2828 | -0.5425 | above noise threshold |
| Pass@1 | 0.9695 | 0.5825 | -0.3870 | above noise threshold |
| Validity | 1.0000 | 0.9992 | -0.0008 | NOT GENERALIZABLE (|delta| < 0.02): reproducible (training is bit-deterministic at fixed seed) but single-seed, so it may invert under another seed — do NOT interpret |
| BLEU (corpus) | 0.8407 | 0.1929 | -0.6478 | above noise threshold |
| BLEU (sent) | 0.8646 | 0.2246 | -0.6399 | above noise threshold |
| chrF2 (corpus) | 85.7212 | 43.9497 | -41.7716 | scale 0-100: 0.02 noise threshold not applicable |
| chrF2 (sent) | 89.7119 | 42.2711 | -47.4408 | scale 0-100: 0.02 noise threshold not applicable |
| Gloss F1 (mic) | 0.8253 | 0.1845 | -0.6409 | above noise threshold |
| Gloss F1 (sent) | 0.8514 | 0.2206 | -0.6307 | above noise threshold |
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

