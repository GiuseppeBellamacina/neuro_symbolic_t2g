# Campaign Report — conll-2003 / qwen25-05b

Generated: 2026-11-14T09:24:30+00:00

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
   - [method] conll-2003/qwen25-05b/ablations/decoding/no-grammar/run_20261009_133123/eval_baseline.json vs conll-2003/qwen25-05b/ablations/sft-amount/sft-grpo-ep005/run_20261113_084157/eval_final__few-shot.json: prompt_context_fingerprint DIFFERS: the eval contexts (prompting/grammar/num_samples/model) are not identical — this pair may differ in more than the method factor
   - [method] conll-2003/qwen25-05b/grpo/few-shot/run_20261009_052104/eval_baseline.json vs conll-2003/qwen25-05b/ablations/sft-amount/sft-grpo-ep015/run_20261113_123213/eval_final__few-shot.json: prompt_context_fingerprint DIFFERS: the eval contexts (prompting/grammar/num_samples/model) are not identical — this pair may differ in more than the method factor
   - [method] conll-2003/qwen25-05b/ablations/decoding/no-grammar-zero-shot/run_20261009_162603/eval_baseline.json vs conll-2003/qwen25-05b/ablations/sft-amount/sft-grpo-ep005/run_20261113_084157/eval_final.json: prompt_context_fingerprint DIFFERS: the eval contexts (prompting/grammar/num_samples/model) are not identical — this pair may differ in more than the method factor
   - [method] conll-2003/qwen25-05b/grpo/few-shot/run_20261009_052104/eval_baseline__zero-shot.json vs conll-2003/qwen25-05b/ablations/sft-amount/sft-grpo-ep015/run_20261113_123213/eval_final.json: prompt_context_fingerprint DIFFERS: the eval contexts (prompting/grammar/num_samples/model) are not identical — this pair may differ in more than the method factor

## How factors are deduced

| Factor | Rule | If not deducible |
|---|---|---|
| `method` | token del nome cella nel percorso (sft-grpo / sft-only / grpo-only / t2g-*); eval_baseline.json => baseline (nessun checkpoint) | unknown (never guessed) |
| `variant` | remaining cell-name tokens after model tag, method and known markers (e.g. structure, viterbi, all-rewards); no-grammar is a grammar marker, not a variant | empty string (plain cell) |
| `prompting` | 1) 'prompting.mode' stamp in the eval JSON; 2) filename suffix __zero-shot / __few-shot (dual/override pass); 3) t2g- zero-shot naming convention (baseline cells only) | unknown — the config-implied mode is NEVER assumed, it can differ from the effective one under dual prompting |
| `grammar` | 'no-grammar' in cell name => off; 'grammar' in t2g-* cell name => on (that family names the toggle explicitly) | unknown (never guessed) |
| `reward_stack` | NOT deducible from path / JSON stamp / payload fields (it lives in the resolved config, which eval JSONs do not embed) | always unknown, declared in every pairing caveat |
| `rl_objective` | NOT deducible from path / JSON stamp / payload fields | always unknown, declared in every pairing caveat |

## Selected runs (48 typologies, latest per typology)

| Run | method | variant | prompting | grammar | Date | Prompts | metrics_version |
|---|---|---|---|---|---|---|---|
| `conll-2003/qwen25-05b/ablations/decoding/hot-rollout-zero-shot/run_20261112_181510/eval_final.json` | unknown | conll-2003-qwen25-05b-ablations-decoding-hot-rollout-zero-shot | zero-shot | unknown | 2026-11-12 | 2000 | 2 |
| `conll-2003/qwen25-05b/ablations/decoding/hot-rollout-zero-shot/run_20261112_181510/eval_final__few-shot.json` | unknown | conll-2003-qwen25-05b-ablations-decoding-hot-rollout-zero-shot | few-shot | unknown | 2026-11-12 | 2000 | 2 |
| `conll-2003/qwen25-05b/ablations/decoding/no-grammar-zero-shot/run_20261009_162603/eval_baseline.json` | baseline | conll-2003-qwen25-05b-ablations-decoding-no-grammar-zero-shot | zero-shot | unknown | 2026-10-09 | 2000 | 2 |
| `conll-2003/qwen25-05b/ablations/decoding/no-grammar-zero-shot/run_20261019_071102/eval_final.json` | unknown | conll-2003-qwen25-05b-ablations-decoding-no-grammar-zero-shot | zero-shot | unknown | 2026-10-19 | 2000 | 2 |
| `conll-2003/qwen25-05b/ablations/decoding/no-grammar/run_20261009_133123/eval_baseline.json` | baseline | conll-2003-qwen25-05b-ablations-decoding-no-grammar | few-shot | unknown | 2026-10-09 | 2000 | 2 |
| `conll-2003/qwen25-05b/ablations/decoding/no-grammar/run_20261019_043117/eval_final.json` | unknown | conll-2003-qwen25-05b-ablations-decoding-no-grammar | few-shot | unknown | 2026-10-19 | 2000 | 2 |
| `conll-2003/qwen25-05b/ablations/objectives/sft-allowed-mass/run_20261019_093124/eval_final.json` | unknown | conll-2003-qwen25-05b-ablations-objectives-sft-allowed-mass | zero-shot | unknown | 2026-10-19 | 2000 | 2 |
| `conll-2003/qwen25-05b/ablations/rewards/entity-f1-zero-shot/run_20261020_133542/eval_final.json` | unknown | conll-2003-qwen25-05b-ablations-rewards-entity-f1-zero-shot | zero-shot | unknown | 2026-10-20 | 2000 | 2 |
| `conll-2003/qwen25-05b/ablations/rewards/entity-f1-zero-shot/run_20261020_133542/eval_final__few-shot.json` | unknown | conll-2003-qwen25-05b-ablations-rewards-entity-f1-zero-shot | few-shot | unknown | 2026-10-20 | 2000 | 2 |
| `conll-2003/qwen25-05b/ablations/rewards/entity-f1/run_20261020_184102/eval_final.json` | unknown | conll-2003-qwen25-05b-ablations-rewards-entity-f1 | few-shot | unknown | 2026-10-20 | 2000 | 2 |
| `conll-2003/qwen25-05b/ablations/rewards/entity-f1/run_20261020_184102/eval_final__zero-shot.json` | unknown | conll-2003-qwen25-05b-ablations-rewards-entity-f1 | zero-shot | unknown | 2026-10-20 | 2000 | 2 |
| `conll-2003/qwen25-05b/ablations/sft-amount/sft-ep005/run_20261113_073503/eval_final.json` | unknown | conll-2003-qwen25-05b-ablations-sft-amount-sft-ep005 | zero-shot | unknown | 2026-11-13 | 2000 | 2 |
| `conll-2003/qwen25-05b/ablations/sft-amount/sft-ep005/run_20261113_073503/eval_final__few-shot.json` | unknown | conll-2003-qwen25-05b-ablations-sft-amount-sft-ep005 | few-shot | unknown | 2026-11-13 | 2000 | 2 |
| `conll-2003/qwen25-05b/ablations/sft-amount/sft-ep015/run_20261113_113133/eval_final.json` | unknown | conll-2003-qwen25-05b-ablations-sft-amount-sft-ep015 | zero-shot | unknown | 2026-11-13 | 2000 | 2 |
| `conll-2003/qwen25-05b/ablations/sft-amount/sft-ep015/run_20261113_113133/eval_final__few-shot.json` | unknown | conll-2003-qwen25-05b-ablations-sft-amount-sft-ep015 | few-shot | unknown | 2026-11-13 | 2000 | 2 |
| `conll-2003/qwen25-05b/ablations/sft-amount/sft-ep05/run_20261113_231624/eval_final.json` | unknown | conll-2003-qwen25-05b-ablations-sft-amount-sft-ep05 | zero-shot | unknown | 2026-11-13 | 2000 | 2 |
| `conll-2003/qwen25-05b/ablations/sft-amount/sft-ep05/run_20261113_231624/eval_final__few-shot.json` | unknown | conll-2003-qwen25-05b-ablations-sft-amount-sft-ep05 | few-shot | unknown | 2026-11-13 | 2000 | 2 |
| `conll-2003/qwen25-05b/ablations/sft-amount/sft-grpo-ep005/run_20261113_084157/eval_final.json` | sft-grpo | ep005 | zero-shot | unknown | 2026-11-13 | 2000 | 2 |
| `conll-2003/qwen25-05b/ablations/sft-amount/sft-grpo-ep005/run_20261113_084157/eval_final__few-shot.json` | sft-grpo | ep005 | few-shot | unknown | 2026-11-13 | 2000 | 2 |
| `conll-2003/qwen25-05b/ablations/sft-amount/sft-grpo-ep015/run_20261113_123213/eval_final.json` | sft-grpo | ep015 | zero-shot | unknown | 2026-11-13 | 2000 | 2 |
| `conll-2003/qwen25-05b/ablations/sft-amount/sft-grpo-ep015/run_20261113_123213/eval_final__few-shot.json` | sft-grpo | ep015 | few-shot | unknown | 2026-11-13 | 2000 | 2 |
| `conll-2003/qwen25-05b/ablations/sft-amount/sft-grpo-ep05/run_20261114_003151/eval_final.json` | sft-grpo | ep05 | zero-shot | unknown | 2026-11-14 | 2000 | 2 |
| `conll-2003/qwen25-05b/ablations/sft-amount/sft-grpo-ep05/run_20261114_003151/eval_final__few-shot.json` | sft-grpo | ep05 | few-shot | unknown | 2026-11-14 | 2000 | 2 |
| `conll-2003/qwen25-05b/baseline/few-shot/run_20261018_161555/eval_zero_shot.json` | unknown | conll-2003-qwen25-05b-baseline-few-shot | few-shot | unknown | 2026-10-18 | 2000 | 2 |
| `conll-2003/qwen25-05b/baseline/zero-shot-no-grammar/run_20261018_163110/eval_zero_shot.json` | unknown | conll-2003-qwen25-05b-baseline-zero-shot-no-grammar | zero-shot | unknown | 2026-10-18 | 2000 | 2 |
| `conll-2003/qwen25-05b/baseline/zero-shot/run_20261018_155358/eval_zero_shot.json` | unknown | conll-2003-qwen25-05b-baseline-zero-shot | zero-shot | unknown | 2026-10-18 | 2000 | 2 |
| `conll-2003/qwen25-05b/grpo/few-shot/run_20261009_052104/eval_baseline.json` | baseline | conll-2003-qwen25-05b-grpo-few-shot | few-shot | unknown | 2026-10-09 | 2000 | 2 |
| `conll-2003/qwen25-05b/grpo/few-shot/run_20261009_052104/eval_baseline__zero-shot.json` | baseline | conll-2003-qwen25-05b-grpo-few-shot | zero-shot | unknown | 2026-10-09 | 2000 | 2 |
| `conll-2003/qwen25-05b/grpo/few-shot/run_20261018_212554/eval_final.json` | unknown | conll-2003-qwen25-05b-grpo-few-shot | few-shot | unknown | 2026-10-18 | 2000 | 2 |
| `conll-2003/qwen25-05b/grpo/few-shot/run_20261018_212554/eval_final__zero-shot.json` | unknown | conll-2003-qwen25-05b-grpo-few-shot | zero-shot | unknown | 2026-10-18 | 2000 | 2 |
| `conll-2003/qwen25-05b/grpo/zero-shot/run_20261018_174556/eval_final.json` | unknown | conll-2003-qwen25-05b-grpo-zero-shot | zero-shot | unknown | 2026-10-18 | 2000 | 2 |
| `conll-2003/qwen25-05b/grpo/zero-shot/run_20261018_174556/eval_final__few-shot.json` | unknown | conll-2003-qwen25-05b-grpo-zero-shot | few-shot | unknown | 2026-10-18 | 2000 | 2 |
| `conll-2003/qwen25-05b/seeds/grpo-zero-shot-s1/run_20261019_202106/eval_final.json` | unknown | conll-2003-qwen25-05b-seeds-grpo-zero-shot-s1 | zero-shot | unknown | 2026-10-19 | 2000 | 2 |
| `conll-2003/qwen25-05b/seeds/grpo-zero-shot-s1/run_20261019_202106/eval_final__few-shot.json` | unknown | conll-2003-qwen25-05b-seeds-grpo-zero-shot-s1 | few-shot | unknown | 2026-10-19 | 2000 | 2 |
| `conll-2003/qwen25-05b/seeds/no-grammar-zero-shot-s1/run_20261019_233119/eval_final.json` | unknown | conll-2003-qwen25-05b-seeds-no-grammar-zero-shot-s1 | zero-shot | unknown | 2026-10-19 | 2000 | 2 |
| `conll-2003/qwen25-05b/seeds/sft-grpo-zero-shot-s1/run_20261020_213557/eval_final.json` | sft-grpo | zero-shot-s1 | zero-shot | unknown | 2026-10-20 | 2000 | 2 |
| `conll-2003/qwen25-05b/seeds/sft-grpo-zero-shot-s1/run_20261020_213557/eval_final__few-shot.json` | sft-grpo | zero-shot-s1 | few-shot | unknown | 2026-10-20 | 2000 | 2 |
| `conll-2003/qwen25-05b/seeds/sft-zero-shot-s1/run_20261109_223700/eval_final.json` | unknown | conll-2003-qwen25-05b-seeds-sft-zero-shot-s1 | zero-shot | unknown | 2026-11-09 | 2000 | 2 |
| `conll-2003/qwen25-05b/seeds/sft-zero-shot-s1/run_20261109_223700/eval_final__few-shot.json` | unknown | conll-2003-qwen25-05b-seeds-sft-zero-shot-s1 | few-shot | unknown | 2026-11-09 | 2000 | 2 |
| `conll-2003/qwen25-05b/sft-grpo/few-shot/run_20261019_002102/eval_final.json` | sft-grpo | few-shot | few-shot | unknown | 2026-10-19 | 2000 | 2 |
| `conll-2003/qwen25-05b/sft-grpo/few-shot/run_20261019_002102/eval_final__zero-shot.json` | sft-grpo | few-shot | zero-shot | unknown | 2026-10-19 | 2000 | 2 |
| `conll-2003/qwen25-05b/sft-grpo/zero-shot/run_20261019_022555/eval_final.json` | sft-grpo | zero-shot | zero-shot | unknown | 2026-10-19 | 2000 | 2 |
| `conll-2003/qwen25-05b/sft-grpo/zero-shot/run_20261019_022555/eval_final__few-shot.json` | sft-grpo | zero-shot | few-shot | unknown | 2026-10-19 | 2000 | 2 |
| `conll-2003/qwen25-05b/sft/zero-shot/run_20261009_004112/eval_baseline.json` | baseline | conll-2003-qwen25-05b-sft-zero-shot | zero-shot | unknown | 2026-10-09 | 2000 | 2 |
| `conll-2003/qwen25-05b/sft/zero-shot/run_20261009_004112/eval_baseline__few-shot.json` | baseline | conll-2003-qwen25-05b-sft-zero-shot | few-shot | unknown | 2026-10-09 | 2000 | 2 |
| `conll-2003/qwen25-05b/sft/zero-shot/run_20261018_165609/eval_final.json` | unknown | conll-2003-qwen25-05b-sft-zero-shot | zero-shot | unknown | 2026-10-18 | 2000 | 2 |
| `conll-2003/qwen25-05b/sft/zero-shot/run_20261018_165609/eval_final__few-shot.json` | unknown | conll-2003-qwen25-05b-sft-zero-shot | few-shot | unknown | 2026-10-18 | 2000 | 2 |
| `conll-2003/qwen25-05b/sft/zero-shot/run_20261018_165609/no-trie/eval_final.json` | unknown | conll-2003-qwen25-05b-sft-zero-shot-no-trie | zero-shot | unknown | 2026-10-18 | 2000 | 2 |

Superseded (older runs of the same typology): 32 (see JSON for the full list).

## Ablation matrix overview — ROUGE-L

One row per cell (its path under `results/`), one column per prompting mode that has at least one value. `—` = the cell was not evaluated in that mode. `*` = multiple runs collapsed, latest shown. Each cell's own `eval_baseline` (the base model in that cell's context) is not a row here: it repeats the `baseline/` cells and stays in the paired comparisons below.

| cell | zero-shot | few-shot |
|---|---|---|
| conll-2003/qwen25-05b/baseline/few-shot | — | 0.3823 |
| conll-2003/qwen25-05b/baseline/zero-shot | 0.2722 | — |
| conll-2003/qwen25-05b/baseline/zero-shot-no-grammar | 0.2676 | — |
| conll-2003/qwen25-05b/sft/zero-shot | 0.9005 | 0.8189 |
| conll-2003/qwen25-05b/sft/zero-shot/no-trie | 0.8985 | — |
| conll-2003/qwen25-05b/sft-grpo/few-shot | 0.9017 | 0.8852 |
| conll-2003/qwen25-05b/sft-grpo/zero-shot | 0.9148 | 0.8260 |
| conll-2003/qwen25-05b/grpo/few-shot | 0.2522 | 0.5396 |
| conll-2003/qwen25-05b/grpo/zero-shot | 0.3167 | 0.4117 |
| conll-2003/qwen25-05b/ablations/decoding/hot-rollout-zero-shot | 0.3554 | 0.4947 |
| conll-2003/qwen25-05b/ablations/decoding/no-grammar | — | 0.6113 |
| conll-2003/qwen25-05b/ablations/decoding/no-grammar-zero-shot | 0.4027 | — |
| conll-2003/qwen25-05b/ablations/objectives/sft-allowed-mass | 0.9044 | — |
| conll-2003/qwen25-05b/ablations/rewards/entity-f1 | 0.2726 | 0.5350 |
| conll-2003/qwen25-05b/ablations/rewards/entity-f1-zero-shot | 0.2831 | 0.3743 |
| conll-2003/qwen25-05b/ablations/sft-amount/sft-ep005 | 0.3448 | 0.4281 |
| conll-2003/qwen25-05b/ablations/sft-amount/sft-ep015 | 0.5237 | 0.3628 |
| conll-2003/qwen25-05b/ablations/sft-amount/sft-ep05 | 0.7843 | 0.5342 |
| conll-2003/qwen25-05b/ablations/sft-amount/sft-grpo-ep005 | 0.6981 | 0.5959 |
| conll-2003/qwen25-05b/ablations/sft-amount/sft-grpo-ep015 | 0.7796 | 0.6287 |
| conll-2003/qwen25-05b/ablations/sft-amount/sft-grpo-ep05 | 0.8423 | 0.5067 |
| conll-2003/qwen25-05b/seeds/grpo-zero-shot-s1 | 0.3231 | 0.4153 |
| conll-2003/qwen25-05b/seeds/no-grammar-zero-shot-s1 | 0.4571 | — |
| conll-2003/qwen25-05b/seeds/sft-grpo-zero-shot-s1 | 0.9176 | 0.8356 |
| conll-2003/qwen25-05b/seeds/sft-zero-shot-s1 | 0.9021 | 0.8122 |

## conll-2003 — all metrics

Same rows as the matrix above, restricted to this dataset, one column per metric. On CoNLL-2003 and WOS-46985 read Exact Match and Gloss F1 (micro = entity-level F1 on CoNLL): ROUGE-L and BLEU split `PER:Mary` into `per` + `mary` and give partial credit to a wrong label of the same type.

| cell | prompting | ROUGE-L | Valid ROUGE-L | Exact Match | Non-copy Tok Acc | Pass@1 | Validity | BLEU (corpus) | BLEU (sent) | chrF2 (corpus) | chrF2 (sent) | Gloss F1 (mic) | Gloss F1 (sent) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| conll-2003/qwen25-05b/baseline/few-shot | few-shot | 0.3823 | 0.3820 | 0.0714 | 0.2828 | 0.5825 | 0.9992 | 0.1929 | 0.2246 | 43.9497 | 42.2711 | 0.1845 | 0.2206 |
| conll-2003/qwen25-05b/baseline/zero-shot | zero-shot | 0.2722 | 0.2718 | 0.0445 | 0.1298 | 0.4115 | 0.9987 | 0.1147 | 0.1320 | 32.3153 | 30.1515 | 0.0816 | 0.0975 |
| conll-2003/qwen25-05b/baseline/zero-shot-no-grammar | zero-shot | 0.2676 | 0.0020 | 0.0011 | 0.0066 | 0.4260 | 0.0075 | 0.0089 | 0.0219 | 17.9933 | 18.2289 | 0.0022 | 0.0056 |
| conll-2003/qwen25-05b/sft/zero-shot | zero-shot | 0.9005 | 0.9005 | 0.7448 | 0.8258 | 0.9670 | 1.0000 | 0.8456 | 0.8604 | 86.1363 | 89.4521 | 0.8255 | 0.8425 |
| conll-2003/qwen25-05b/sft/zero-shot | few-shot | 0.8189 | 0.8189 | 0.6003 | 0.7380 | 0.9290 | 1.0000 | 0.7279 | 0.7530 | 77.9801 | 81.8983 | 0.7230 | 0.7485 |
| conll-2003/qwen25-05b/sft/zero-shot/no-trie | zero-shot | 0.8985 | 0.8427 | 0.7274 | 0.8112 | 0.9690 | 0.9379 | 0.8400 | 0.8534 | 85.4184 | 88.7456 | 0.8167 | 0.8331 |
| conll-2003/qwen25-05b/sft-grpo/few-shot | zero-shot | 0.9017 | 0.9017 | 0.7540 | 0.8377 | 0.9695 | 1.0000 | 0.8523 | 0.8635 | 87.1416 | 89.6895 | 0.8365 | 0.8453 |
| conll-2003/qwen25-05b/sft-grpo/few-shot | few-shot | 0.8852 | 0.8852 | 0.7159 | 0.8260 | 0.9655 | 1.0000 | 0.8110 | 0.8379 | 85.4507 | 88.3649 | 0.8149 | 0.8272 |
| conll-2003/qwen25-05b/sft-grpo/zero-shot | zero-shot | 0.9148 | 0.9148 | 0.7811 | 0.8615 | 0.9710 | 1.0000 | 0.8515 | 0.8803 | 88.4057 | 91.1192 | 0.8546 | 0.8715 |
| conll-2003/qwen25-05b/sft-grpo/zero-shot | few-shot | 0.8260 | 0.8260 | 0.6268 | 0.7736 | 0.9265 | 1.0000 | 0.7305 | 0.7689 | 80.4619 | 83.1901 | 0.7461 | 0.7681 |
| conll-2003/qwen25-05b/grpo/few-shot | zero-shot | 0.2522 | 0.2520 | 0.1324 | 0.1348 | 0.3110 | 0.9995 | 0.1152 | 0.1811 | 21.8568 | 28.4721 | 0.1104 | 0.1648 |
| conll-2003/qwen25-05b/grpo/few-shot | few-shot | 0.5396 | 0.5394 | 0.3247 | 0.4067 | 0.6745 | 0.9997 | 0.3949 | 0.4515 | 48.4686 | 54.3000 | 0.3776 | 0.4457 |
| conll-2003/qwen25-05b/grpo/zero-shot | zero-shot | 0.3167 | 0.3166 | 0.0296 | 0.1963 | 0.5015 | 0.9998 | 0.1818 | 0.1729 | 40.9877 | 36.7997 | 0.1298 | 0.1284 |
| conll-2003/qwen25-05b/grpo/zero-shot | few-shot | 0.4117 | 0.4117 | 0.0819 | 0.3388 | 0.6270 | 0.9999 | 0.2440 | 0.2604 | 49.6694 | 46.2937 | 0.2353 | 0.2576 |
| conll-2003/qwen25-05b/ablations/decoding/hot-rollout-zero-shot | zero-shot | 0.3554 | 0.3554 | 0.1234 | 0.2235 | 0.5095 | 1.0000 | 0.2482 | 0.2648 | 36.2266 | 37.5701 | 0.1905 | 0.2192 |
| conll-2003/qwen25-05b/ablations/decoding/hot-rollout-zero-shot | few-shot | 0.4947 | 0.4947 | 0.2173 | 0.3859 | 0.6645 | 1.0000 | 0.3120 | 0.3827 | 50.8083 | 51.7674 | 0.3071 | 0.3651 |
| conll-2003/qwen25-05b/ablations/decoding/no-grammar | few-shot | 0.6113 | 0.3974 | 0.3476 | 0.4229 | 0.7550 | 0.6501 | 0.4346 | 0.4840 | 50.5409 | 58.4925 | 0.3776 | 0.4749 |
| conll-2003/qwen25-05b/ablations/decoding/no-grammar-zero-shot | zero-shot | 0.4027 | 0.1843 | 0.1472 | 0.1104 | 0.5735 | 0.4578 | 0.0083 | 0.1717 | 21.3288 | 31.1467 | 0.0732 | 0.1693 |
| conll-2003/qwen25-05b/ablations/objectives/sft-allowed-mass | zero-shot | 0.9044 | 0.9044 | 0.7505 | 0.8253 | 0.9695 | 1.0000 | 0.8407 | 0.8646 | 85.7212 | 89.7119 | 0.8253 | 0.8514 |
| conll-2003/qwen25-05b/ablations/rewards/entity-f1 | zero-shot | 0.2726 | 0.2724 | 0.1432 | 0.1441 | 0.3390 | 0.9994 | 0.1300 | 0.1961 | 23.6592 | 30.0173 | 0.1200 | 0.1770 |
| conll-2003/qwen25-05b/ablations/rewards/entity-f1 | few-shot | 0.5350 | 0.5349 | 0.3340 | 0.4007 | 0.6595 | 0.9998 | 0.3959 | 0.4509 | 46.9459 | 53.8466 | 0.3773 | 0.4492 |
| conll-2003/qwen25-05b/ablations/rewards/entity-f1-zero-shot | zero-shot | 0.2831 | 0.2830 | 0.0415 | 0.1732 | 0.4325 | 0.9994 | 0.1333 | 0.1453 | 36.5978 | 33.3169 | 0.1020 | 0.1165 |
| conll-2003/qwen25-05b/ablations/rewards/entity-f1-zero-shot | few-shot | 0.3743 | 0.3741 | 0.0576 | 0.3075 | 0.5700 | 0.9995 | 0.1943 | 0.2163 | 46.0106 | 42.9608 | 0.1915 | 0.2203 |
| conll-2003/qwen25-05b/ablations/sft-amount/sft-ep005 | zero-shot | 0.3448 | 0.3448 | 0.1433 | 0.2171 | 0.4485 | 1.0000 | 0.2770 | 0.2474 | 33.8297 | 35.6142 | 0.1991 | 0.2189 |
| conll-2003/qwen25-05b/ablations/sft-amount/sft-ep005 | few-shot | 0.4281 | 0.4281 | 0.2577 | 0.3063 | 0.5285 | 1.0000 | 0.3143 | 0.3545 | 36.9027 | 42.8579 | 0.3217 | 0.3498 |
| conll-2003/qwen25-05b/ablations/sft-amount/sft-ep015 | zero-shot | 0.5237 | 0.5237 | 0.3236 | 0.4110 | 0.6355 | 1.0000 | 0.4519 | 0.4568 | 50.2511 | 51.8350 | 0.4287 | 0.4287 |
| conll-2003/qwen25-05b/ablations/sft-amount/sft-ep015 | few-shot | 0.3628 | 0.3628 | 0.2410 | 0.2632 | 0.4300 | 0.9999 | 0.2298 | 0.3213 | 30.9730 | 36.9577 | 0.2928 | 0.3075 |
| conll-2003/qwen25-05b/ablations/sft-amount/sft-ep05 | zero-shot | 0.7843 | 0.7843 | 0.5437 | 0.6499 | 0.9150 | 1.0000 | 0.6967 | 0.7090 | 72.3738 | 77.7119 | 0.6497 | 0.6833 |
| conll-2003/qwen25-05b/ablations/sft-amount/sft-ep05 | few-shot | 0.5342 | 0.5342 | 0.3289 | 0.4288 | 0.6435 | 1.0000 | 0.4540 | 0.4641 | 50.8174 | 53.5267 | 0.4439 | 0.4507 |
| conll-2003/qwen25-05b/ablations/sft-amount/sft-grpo-ep005 | zero-shot | 0.6981 | 0.6981 | 0.4407 | 0.5469 | 0.8400 | 1.0000 | 0.5528 | 0.5995 | 63.1447 | 68.2546 | 0.5238 | 0.5825 |
| conll-2003/qwen25-05b/ablations/sft-amount/sft-grpo-ep005 | few-shot | 0.5959 | 0.5959 | 0.3244 | 0.4763 | 0.7595 | 0.9999 | 0.4624 | 0.4876 | 58.7044 | 59.3880 | 0.4362 | 0.4735 |
| conll-2003/qwen25-05b/ablations/sft-amount/sft-grpo-ep015 | zero-shot | 0.7796 | 0.7795 | 0.5530 | 0.6479 | 0.9010 | 0.9999 | 0.6822 | 0.7060 | 71.2416 | 76.8638 | 0.6511 | 0.6907 |
| conll-2003/qwen25-05b/ablations/sft-amount/sft-grpo-ep015 | few-shot | 0.6287 | 0.6286 | 0.4128 | 0.5211 | 0.7540 | 0.9998 | 0.5634 | 0.5550 | 60.1020 | 63.2713 | 0.5136 | 0.5418 |
| conll-2003/qwen25-05b/ablations/sft-amount/sft-grpo-ep05 | zero-shot | 0.8423 | 0.8423 | 0.6501 | 0.7423 | 0.9380 | 1.0000 | 0.7570 | 0.7863 | 78.7923 | 83.8403 | 0.7382 | 0.7744 |
| conll-2003/qwen25-05b/ablations/sft-amount/sft-grpo-ep05 | few-shot | 0.5067 | 0.5067 | 0.3606 | 0.4005 | 0.5830 | 1.0000 | 0.3766 | 0.4605 | 45.2541 | 51.2296 | 0.4363 | 0.4494 |
| conll-2003/qwen25-05b/seeds/grpo-zero-shot-s1 | zero-shot | 0.3231 | 0.3231 | 0.0378 | 0.2016 | 0.5090 | 1.0000 | 0.1924 | 0.1827 | 40.9927 | 37.0738 | 0.1400 | 0.1400 |
| conll-2003/qwen25-05b/seeds/grpo-zero-shot-s1 | few-shot | 0.4153 | 0.4151 | 0.0883 | 0.3205 | 0.6240 | 0.9997 | 0.2544 | 0.2664 | 49.4475 | 46.0264 | 0.2311 | 0.2469 |
| conll-2003/qwen25-05b/seeds/no-grammar-zero-shot-s1 | zero-shot | 0.4571 | 0.0922 | 0.0567 | 0.1088 | 0.7450 | 0.2016 | 0.0138 | 0.1043 | 27.7587 | 34.5595 | 0.0605 | 0.1192 |
| conll-2003/qwen25-05b/seeds/sft-grpo-zero-shot-s1 | zero-shot | 0.9176 | 0.9176 | 0.7873 | 0.8599 | 0.9735 | 1.0000 | 0.8591 | 0.8845 | 88.3527 | 91.4008 | 0.8570 | 0.8758 |
| conll-2003/qwen25-05b/seeds/sft-grpo-zero-shot-s1 | few-shot | 0.8356 | 0.8356 | 0.6352 | 0.7833 | 0.9375 | 1.0000 | 0.7192 | 0.7768 | 80.8448 | 84.2907 | 0.7506 | 0.7784 |
| conll-2003/qwen25-05b/seeds/sft-zero-shot-s1 | zero-shot | 0.9021 | 0.9021 | 0.7472 | 0.8309 | 0.9710 | 1.0000 | 0.8432 | 0.8626 | 86.4902 | 89.6613 | 0.8283 | 0.8452 |
| conll-2003/qwen25-05b/seeds/sft-zero-shot-s1 | few-shot | 0.8122 | 0.8122 | 0.5918 | 0.7415 | 0.9255 | 1.0000 | 0.7006 | 0.7462 | 78.1695 | 81.5882 | 0.7124 | 0.7421 |

## Paired comparisons — one table per factor

### Factor: `method`

**method effect (cross-run)** — `conll-2003/qwen25-05b/ablations/decoding/no-grammar/run_20261009_133123/eval_baseline.json` (2026-10-09, few-shot, unknown) → `conll-2003/qwen25-05b/ablations/sft-amount/sft-grpo-ep005/run_20261113_084157/eval_final__few-shot.json` (2026-11-13, few-shot, unknown)

| Metric | A | B | Δ (B−A) | Interpretation |
|---|---|---|---|---|
| ROUGE-L | 0.3700 | 0.5959 | +0.2259 | above noise threshold |
| Valid ROUGE-L | 0.0635 | 0.5959 | +0.5324 | above noise threshold |
| Exact Match | 0.0344 | 0.3244 | +0.2900 | above noise threshold |
| Non-copy Tok Acc | 0.1922 | 0.4763 | +0.2841 | above noise threshold |
| Pass@1 | 0.5440 | 0.7595 | +0.2155 | above noise threshold |
| Validity | 0.1716 | 0.9999 | +0.8283 | above noise threshold |
| BLEU (corpus) | 0.0961 | 0.4624 | +0.3663 | above noise threshold |
| BLEU (sent) | 0.1556 | 0.4876 | +0.3320 | above noise threshold |
| chrF2 (corpus) | 33.2195 | 58.7044 | +25.4849 | scale 0-100: 0.02 noise threshold not applicable |
| chrF2 (sent) | 35.9071 | 59.3880 | +23.4809 | scale 0-100: 0.02 noise threshold not applicable |
| Gloss F1 (mic) | 0.0839 | 0.4362 | +0.3522 | above noise threshold |
| Gloss F1 (sent) | 0.1436 | 0.4735 | +0.3299 | above noise threshold |
- ⚠️ factors not deducible on at least one side, equality NOT verified: reward_stack, rl_objective
- ⚠️ prompt_context_fingerprint DIFFERS: the eval contexts (prompting/grammar/num_samples/model) are not identical — this pair may differ in more than the method factor

**method effect (cross-run)** — `conll-2003/qwen25-05b/grpo/few-shot/run_20261009_052104/eval_baseline.json` (2026-10-09, few-shot, unknown) → `conll-2003/qwen25-05b/ablations/sft-amount/sft-grpo-ep015/run_20261113_123213/eval_final__few-shot.json` (2026-11-13, few-shot, unknown)

| Metric | A | B | Δ (B−A) | Interpretation |
|---|---|---|---|---|
| ROUGE-L | 0.3805 | 0.6287 | +0.2482 | above noise threshold |
| Valid ROUGE-L | 0.3706 | 0.6286 | +0.2580 | above noise threshold |
| Exact Match | 0.0662 | 0.4128 | +0.3466 | above noise threshold |
| Non-copy Tok Acc | 0.3169 | 0.5211 | +0.2042 | above noise threshold |
| Pass@1 | 0.5865 | 0.7540 | +0.1675 | above noise threshold |
| Validity | 0.9739 | 0.9998 | +0.0259 | above noise threshold |
| BLEU (corpus) | 0.1429 | 0.5634 | +0.4205 | above noise threshold |
| BLEU (sent) | 0.2204 | 0.5550 | +0.3346 | above noise threshold |
| chrF2 (corpus) | 39.8261 | 60.1020 | +20.2760 | scale 0-100: 0.02 noise threshold not applicable |
| chrF2 (sent) | 41.4512 | 63.2713 | +21.8200 | scale 0-100: 0.02 noise threshold not applicable |
| Gloss F1 (mic) | 0.1714 | 0.5136 | +0.3422 | above noise threshold |
| Gloss F1 (sent) | 0.2366 | 0.5418 | +0.3051 | above noise threshold |
- ⚠️ factors not deducible on at least one side, equality NOT verified: reward_stack, rl_objective
- ⚠️ prompt_context_fingerprint DIFFERS: the eval contexts (prompting/grammar/num_samples/model) are not identical — this pair may differ in more than the method factor

**method effect (cross-run)** — `conll-2003/qwen25-05b/sft/zero-shot/run_20261009_004112/eval_baseline__few-shot.json` (2026-10-09, few-shot, unknown) → `conll-2003/qwen25-05b/ablations/sft-amount/sft-grpo-ep05/run_20261114_003151/eval_final__few-shot.json` (2026-11-14, few-shot, unknown)

| Metric | A | B | Δ (B−A) | Interpretation |
|---|---|---|---|---|
| ROUGE-L | 0.3805 | 0.5067 | +0.1262 | above noise threshold |
| Valid ROUGE-L | 0.3706 | 0.5067 | +0.1362 | above noise threshold |
| Exact Match | 0.0662 | 0.3606 | +0.2944 | above noise threshold |
| Non-copy Tok Acc | 0.3169 | 0.4005 | +0.0836 | above noise threshold |
| Pass@1 | 0.5865 | 0.5830 | -0.0035 | NOT GENERALIZABLE (|delta| < 0.02): reproducible (training is bit-deterministic at fixed seed) but single-seed, so it may invert under another seed — do NOT interpret |
| Validity | 0.9739 | 1.0000 | +0.0261 | above noise threshold |
| BLEU (corpus) | 0.1429 | 0.3766 | +0.2337 | above noise threshold |
| BLEU (sent) | 0.2204 | 0.4605 | +0.2400 | above noise threshold |
| chrF2 (corpus) | 39.8261 | 45.2541 | +5.4281 | scale 0-100: 0.02 noise threshold not applicable |
| chrF2 (sent) | 41.4512 | 51.2296 | +9.7784 | scale 0-100: 0.02 noise threshold not applicable |
| Gloss F1 (mic) | 0.1714 | 0.4363 | +0.2649 | above noise threshold |
| Gloss F1 (sent) | 0.2366 | 0.4494 | +0.2128 | above noise threshold |
- ⚠️ factors not deducible on at least one side, equality NOT verified: reward_stack, rl_objective

**method effect (cross-run)** — `conll-2003/qwen25-05b/ablations/decoding/no-grammar-zero-shot/run_20261009_162603/eval_baseline.json` (2026-10-09, zero-shot, unknown) → `conll-2003/qwen25-05b/ablations/sft-amount/sft-grpo-ep005/run_20261113_084157/eval_final.json` (2026-11-13, zero-shot, unknown)

| Metric | A | B | Δ (B−A) | Interpretation |
|---|---|---|---|---|
| ROUGE-L | 0.2676 | 0.6981 | +0.4305 | above noise threshold |
| Valid ROUGE-L | 0.0021 | 0.6981 | +0.6960 | above noise threshold |
| Exact Match | 0.0011 | 0.4407 | +0.4396 | above noise threshold |
| Non-copy Tok Acc | 0.0066 | 0.5469 | +0.5404 | above noise threshold |
| Pass@1 | 0.4260 | 0.8400 | +0.4140 | above noise threshold |
| Validity | 0.0079 | 1.0000 | +0.9921 | above noise threshold |
| BLEU (corpus) | 0.0089 | 0.5528 | +0.5439 | above noise threshold |
| BLEU (sent) | 0.0219 | 0.5995 | +0.5777 | above noise threshold |
| chrF2 (corpus) | 17.9933 | 63.1447 | +45.1514 | scale 0-100: 0.02 noise threshold not applicable |
| chrF2 (sent) | 18.2289 | 68.2546 | +50.0257 | scale 0-100: 0.02 noise threshold not applicable |
| Gloss F1 (mic) | 0.0022 | 0.5238 | +0.5216 | above noise threshold |
| Gloss F1 (sent) | 0.0056 | 0.5825 | +0.5769 | above noise threshold |
- ⚠️ factors not deducible on at least one side, equality NOT verified: reward_stack, rl_objective
- ⚠️ prompt_context_fingerprint DIFFERS: the eval contexts (prompting/grammar/num_samples/model) are not identical — this pair may differ in more than the method factor

**method effect (cross-run)** — `conll-2003/qwen25-05b/grpo/few-shot/run_20261009_052104/eval_baseline__zero-shot.json` (2026-10-09, zero-shot, unknown) → `conll-2003/qwen25-05b/ablations/sft-amount/sft-grpo-ep015/run_20261113_123213/eval_final.json` (2026-11-13, zero-shot, unknown)

| Metric | A | B | Δ (B−A) | Interpretation |
|---|---|---|---|---|
| ROUGE-L | 0.2708 | 0.7796 | +0.5087 | above noise threshold |
| Valid ROUGE-L | 0.2625 | 0.7795 | +0.5170 | above noise threshold |
| Exact Match | 0.0453 | 0.5530 | +0.5077 | above noise threshold |
| Non-copy Tok Acc | 0.1458 | 0.6479 | +0.5021 | above noise threshold |
| Pass@1 | 0.4145 | 0.9010 | +0.4865 | above noise threshold |
| Validity | 0.9693 | 0.9999 | +0.0306 | above noise threshold |
| BLEU (corpus) | 0.0893 | 0.6822 | +0.5930 | above noise threshold |
| BLEU (sent) | 0.1329 | 0.7060 | +0.5731 | above noise threshold |
| chrF2 (corpus) | 29.5092 | 71.2416 | +41.7324 | scale 0-100: 0.02 noise threshold not applicable |
| chrF2 (sent) | 29.5764 | 76.8638 | +47.2875 | scale 0-100: 0.02 noise threshold not applicable |
| Gloss F1 (mic) | 0.0762 | 0.6511 | +0.5749 | above noise threshold |
| Gloss F1 (sent) | 0.1033 | 0.6907 | +0.5874 | above noise threshold |
- ⚠️ factors not deducible on at least one side, equality NOT verified: reward_stack, rl_objective
- ⚠️ prompt_context_fingerprint DIFFERS: the eval contexts (prompting/grammar/num_samples/model) are not identical — this pair may differ in more than the method factor

**method effect (cross-run)** — `conll-2003/qwen25-05b/sft/zero-shot/run_20261009_004112/eval_baseline.json` (2026-10-09, zero-shot, unknown) → `conll-2003/qwen25-05b/ablations/sft-amount/sft-grpo-ep05/run_20261114_003151/eval_final.json` (2026-11-14, zero-shot, unknown)

| Metric | A | B | Δ (B−A) | Interpretation |
|---|---|---|---|---|
| ROUGE-L | 0.2708 | 0.8423 | +0.5714 | above noise threshold |
| Valid ROUGE-L | 0.2625 | 0.8423 | +0.5798 | above noise threshold |
| Exact Match | 0.0453 | 0.6501 | +0.6048 | above noise threshold |
| Non-copy Tok Acc | 0.1458 | 0.7423 | +0.5965 | above noise threshold |
| Pass@1 | 0.4145 | 0.9380 | +0.5235 | above noise threshold |
| Validity | 0.9693 | 1.0000 | +0.0307 | above noise threshold |
| BLEU (corpus) | 0.0893 | 0.7570 | +0.6677 | above noise threshold |
| BLEU (sent) | 0.1329 | 0.7863 | +0.6534 | above noise threshold |
| chrF2 (corpus) | 29.5092 | 78.7923 | +49.2831 | scale 0-100: 0.02 noise threshold not applicable |
| chrF2 (sent) | 29.5764 | 83.8403 | +54.2640 | scale 0-100: 0.02 noise threshold not applicable |
| Gloss F1 (mic) | 0.0762 | 0.7382 | +0.6620 | above noise threshold |
| Gloss F1 (sent) | 0.1033 | 0.7744 | +0.6711 | above noise threshold |
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

🔴 **prompt-dependence (dual eval, same checkpoint)** — `conll-2003/qwen25-05b/ablations/sft-amount/sft-grpo-ep05/run_20261114_003151/eval_final__few-shot.json` (2026-11-14, few-shot, unknown) → `conll-2003/qwen25-05b/ablations/sft-amount/sft-grpo-ep05/run_20261114_003151/eval_final.json` (2026-11-14, zero-shot, unknown)

| Metric | A | B | Δ (B−A) | Interpretation |
|---|---|---|---|---|
| ROUGE-L | 0.5067 | 0.8423 | +0.3355 | above noise threshold |
| Valid ROUGE-L | 0.5067 | 0.8423 | +0.3355 | above noise threshold |
| Exact Match | 0.3606 | 0.6501 | +0.2895 | above noise threshold |
| Non-copy Tok Acc | 0.4005 | 0.7423 | +0.3418 | above noise threshold |
| Pass@1 | 0.5830 | 0.9380 | +0.3550 | above noise threshold |
| Validity | 1.0000 | 1.0000 | +0.0000 | NOT GENERALIZABLE (|delta| < 0.02): reproducible (training is bit-deterministic at fixed seed) but single-seed, so it may invert under another seed — do NOT interpret |
| BLEU (corpus) | 0.3766 | 0.7570 | +0.3804 | above noise threshold |
| BLEU (sent) | 0.4605 | 0.7863 | +0.3258 | above noise threshold |
| chrF2 (corpus) | 45.2541 | 78.7923 | +33.5382 | scale 0-100: 0.02 noise threshold not applicable |
| chrF2 (sent) | 51.2296 | 83.8403 | +32.6107 | scale 0-100: 0.02 noise threshold not applicable |
| Gloss F1 (mic) | 0.4363 | 0.7382 | +0.3019 | above noise threshold |
| Gloss F1 (sent) | 0.4494 | 0.7744 | +0.3250 | above noise threshold |
- ⚠️ HIGHLIGHT: same model evaluated in both prompting modes — how much does it depend on the prompt as a crutch?
- ⚠️ factors not deducible on at least one side, equality NOT verified: reward_stack, rl_objective

🔴 **prompt-dependence (dual eval, same checkpoint)** — `conll-2003/qwen25-05b/ablations/sft-amount/sft-grpo-ep015/run_20261113_123213/eval_final__few-shot.json` (2026-11-13, few-shot, unknown) → `conll-2003/qwen25-05b/ablations/sft-amount/sft-grpo-ep015/run_20261113_123213/eval_final.json` (2026-11-13, zero-shot, unknown)

| Metric | A | B | Δ (B−A) | Interpretation |
|---|---|---|---|---|
| ROUGE-L | 0.6287 | 0.7796 | +0.1508 | above noise threshold |
| Valid ROUGE-L | 0.6286 | 0.7795 | +0.1509 | above noise threshold |
| Exact Match | 0.4128 | 0.5530 | +0.1402 | above noise threshold |
| Non-copy Tok Acc | 0.5211 | 0.6479 | +0.1268 | above noise threshold |
| Pass@1 | 0.7540 | 0.9010 | +0.1470 | above noise threshold |
| Validity | 0.9998 | 0.9999 | +0.0001 | NOT GENERALIZABLE (|delta| < 0.02): reproducible (training is bit-deterministic at fixed seed) but single-seed, so it may invert under another seed — do NOT interpret |
| BLEU (corpus) | 0.5634 | 0.6822 | +0.1188 | above noise threshold |
| BLEU (sent) | 0.5550 | 0.7060 | +0.1510 | above noise threshold |
| chrF2 (corpus) | 60.1020 | 71.2416 | +11.1396 | scale 0-100: 0.02 noise threshold not applicable |
| chrF2 (sent) | 63.2713 | 76.8638 | +13.5926 | scale 0-100: 0.02 noise threshold not applicable |
| Gloss F1 (mic) | 0.5136 | 0.6511 | +0.1375 | above noise threshold |
| Gloss F1 (sent) | 0.5418 | 0.6907 | +0.1490 | above noise threshold |
- ⚠️ HIGHLIGHT: same model evaluated in both prompting modes — how much does it depend on the prompt as a crutch?
- ⚠️ factors not deducible on at least one side, equality NOT verified: reward_stack, rl_objective

🔴 **prompt-dependence (dual eval, same checkpoint)** — `conll-2003/qwen25-05b/ablations/sft-amount/sft-grpo-ep005/run_20261113_084157/eval_final__few-shot.json` (2026-11-13, few-shot, unknown) → `conll-2003/qwen25-05b/ablations/sft-amount/sft-grpo-ep005/run_20261113_084157/eval_final.json` (2026-11-13, zero-shot, unknown)

| Metric | A | B | Δ (B−A) | Interpretation |
|---|---|---|---|---|
| ROUGE-L | 0.5959 | 0.6981 | +0.1022 | above noise threshold |
| Valid ROUGE-L | 0.5959 | 0.6981 | +0.1023 | above noise threshold |
| Exact Match | 0.3244 | 0.4407 | +0.1163 | above noise threshold |
| Non-copy Tok Acc | 0.4763 | 0.5469 | +0.0706 | above noise threshold |
| Pass@1 | 0.7595 | 0.8400 | +0.0805 | above noise threshold |
| Validity | 0.9999 | 1.0000 | +0.0001 | NOT GENERALIZABLE (|delta| < 0.02): reproducible (training is bit-deterministic at fixed seed) but single-seed, so it may invert under another seed — do NOT interpret |
| BLEU (corpus) | 0.4624 | 0.5528 | +0.0903 | above noise threshold |
| BLEU (sent) | 0.4876 | 0.5995 | +0.1119 | above noise threshold |
| chrF2 (corpus) | 58.7044 | 63.1447 | +4.4403 | scale 0-100: 0.02 noise threshold not applicable |
| chrF2 (sent) | 59.3880 | 68.2546 | +8.8666 | scale 0-100: 0.02 noise threshold not applicable |
| Gloss F1 (mic) | 0.4362 | 0.5238 | +0.0876 | above noise threshold |
| Gloss F1 (sent) | 0.4735 | 0.5825 | +0.1090 | above noise threshold |
- ⚠️ HIGHLIGHT: same model evaluated in both prompting modes — how much does it depend on the prompt as a crutch?
- ⚠️ factors not deducible on at least one side, equality NOT verified: reward_stack, rl_objective

🔴 **prompt-dependence (dual eval, same checkpoint)** — `conll-2003/qwen25-05b/seeds/sft-grpo-zero-shot-s1/run_20261020_213557/eval_final__few-shot.json` (2026-10-20, few-shot, unknown) → `conll-2003/qwen25-05b/seeds/sft-grpo-zero-shot-s1/run_20261020_213557/eval_final.json` (2026-10-20, zero-shot, unknown)

| Metric | A | B | Δ (B−A) | Interpretation |
|---|---|---|---|---|
| ROUGE-L | 0.8356 | 0.9176 | +0.0821 | above noise threshold |
| Valid ROUGE-L | 0.8356 | 0.9176 | +0.0821 | above noise threshold |
| Exact Match | 0.6352 | 0.7873 | +0.1521 | above noise threshold |
| Non-copy Tok Acc | 0.7833 | 0.8599 | +0.0766 | above noise threshold |
| Pass@1 | 0.9375 | 0.9735 | +0.0360 | above noise threshold |
| Validity | 1.0000 | 1.0000 | +0.0000 | NOT GENERALIZABLE (|delta| < 0.02): reproducible (training is bit-deterministic at fixed seed) but single-seed, so it may invert under another seed — do NOT interpret |
| BLEU (corpus) | 0.7192 | 0.8591 | +0.1399 | above noise threshold |
| BLEU (sent) | 0.7768 | 0.8845 | +0.1078 | above noise threshold |
| chrF2 (corpus) | 80.8448 | 88.3527 | +7.5078 | scale 0-100: 0.02 noise threshold not applicable |
| chrF2 (sent) | 84.2907 | 91.4008 | +7.1101 | scale 0-100: 0.02 noise threshold not applicable |
| Gloss F1 (mic) | 0.7506 | 0.8570 | +0.1064 | above noise threshold |
| Gloss F1 (sent) | 0.7784 | 0.8758 | +0.0974 | above noise threshold |
- ⚠️ HIGHLIGHT: same model evaluated in both prompting modes — how much does it depend on the prompt as a crutch?
- ⚠️ factors not deducible on at least one side, equality NOT verified: reward_stack, rl_objective

🔴 **prompt-dependence (dual eval, same checkpoint)** — `conll-2003/qwen25-05b/sft-grpo/zero-shot/run_20261019_022555/eval_final__few-shot.json` (2026-10-19, few-shot, unknown) → `conll-2003/qwen25-05b/sft-grpo/zero-shot/run_20261019_022555/eval_final.json` (2026-10-19, zero-shot, unknown)

| Metric | A | B | Δ (B−A) | Interpretation |
|---|---|---|---|---|
| ROUGE-L | 0.8260 | 0.9148 | +0.0888 | above noise threshold |
| Valid ROUGE-L | 0.8260 | 0.9148 | +0.0888 | above noise threshold |
| Exact Match | 0.6268 | 0.7811 | +0.1543 | above noise threshold |
| Non-copy Tok Acc | 0.7736 | 0.8615 | +0.0878 | above noise threshold |
| Pass@1 | 0.9265 | 0.9710 | +0.0445 | above noise threshold |
| Validity | 1.0000 | 1.0000 | +0.0000 | NOT GENERALIZABLE (|delta| < 0.02): reproducible (training is bit-deterministic at fixed seed) but single-seed, so it may invert under another seed — do NOT interpret |
| BLEU (corpus) | 0.7305 | 0.8515 | +0.1210 | above noise threshold |
| BLEU (sent) | 0.7689 | 0.8803 | +0.1114 | above noise threshold |
| chrF2 (corpus) | 80.4619 | 88.4057 | +7.9438 | scale 0-100: 0.02 noise threshold not applicable |
| chrF2 (sent) | 83.1901 | 91.1192 | +7.9291 | scale 0-100: 0.02 noise threshold not applicable |
| Gloss F1 (mic) | 0.7461 | 0.8546 | +0.1085 | above noise threshold |
| Gloss F1 (sent) | 0.7681 | 0.8715 | +0.1033 | above noise threshold |
- ⚠️ HIGHLIGHT: same model evaluated in both prompting modes — how much does it depend on the prompt as a crutch?
- ⚠️ factors not deducible on at least one side, equality NOT verified: reward_stack, rl_objective

🔴 **prompt-dependence (dual eval, same checkpoint)** — `conll-2003/qwen25-05b/sft-grpo/few-shot/run_20261019_002102/eval_final.json` (2026-10-19, few-shot, unknown) → `conll-2003/qwen25-05b/sft-grpo/few-shot/run_20261019_002102/eval_final__zero-shot.json` (2026-10-19, zero-shot, unknown)

| Metric | A | B | Δ (B−A) | Interpretation |
|---|---|---|---|---|
| ROUGE-L | 0.8852 | 0.9017 | +0.0165 | NOT GENERALIZABLE (|delta| < 0.02): reproducible (training is bit-deterministic at fixed seed) but single-seed, so it may invert under another seed — do NOT interpret |
| Valid ROUGE-L | 0.8852 | 0.9017 | +0.0165 | NOT GENERALIZABLE (|delta| < 0.02): reproducible (training is bit-deterministic at fixed seed) but single-seed, so it may invert under another seed — do NOT interpret |
| Exact Match | 0.7159 | 0.7540 | +0.0381 | above noise threshold |
| Non-copy Tok Acc | 0.8260 | 0.8377 | +0.0117 | NOT GENERALIZABLE (|delta| < 0.02): reproducible (training is bit-deterministic at fixed seed) but single-seed, so it may invert under another seed — do NOT interpret |
| Pass@1 | 0.9655 | 0.9695 | +0.0040 | NOT GENERALIZABLE (|delta| < 0.02): reproducible (training is bit-deterministic at fixed seed) but single-seed, so it may invert under another seed — do NOT interpret |
| Validity | 1.0000 | 1.0000 | +0.0000 | NOT GENERALIZABLE (|delta| < 0.02): reproducible (training is bit-deterministic at fixed seed) but single-seed, so it may invert under another seed — do NOT interpret |
| BLEU (corpus) | 0.8110 | 0.8523 | +0.0413 | above noise threshold |
| BLEU (sent) | 0.8379 | 0.8635 | +0.0256 | above noise threshold |
| chrF2 (corpus) | 85.4507 | 87.1416 | +1.6909 | scale 0-100: 0.02 noise threshold not applicable |
| chrF2 (sent) | 88.3649 | 89.6895 | +1.3245 | scale 0-100: 0.02 noise threshold not applicable |
| Gloss F1 (mic) | 0.8149 | 0.8365 | +0.0217 | above noise threshold |
| Gloss F1 (sent) | 0.8272 | 0.8453 | +0.0180 | NOT GENERALIZABLE (|delta| < 0.02): reproducible (training is bit-deterministic at fixed seed) but single-seed, so it may invert under another seed — do NOT interpret |
- ⚠️ HIGHLIGHT: same model evaluated in both prompting modes — how much does it depend on the prompt as a crutch?
- ⚠️ factors not deducible on at least one side, equality NOT verified: reward_stack, rl_objective

🔴 **prompt-dependence (dual eval, same checkpoint)** — `conll-2003/qwen25-05b/ablations/sft-amount/sft-ep05/run_20261113_231624/eval_final__few-shot.json` (2026-11-13, few-shot, unknown) → `conll-2003/qwen25-05b/ablations/sft-amount/sft-ep05/run_20261113_231624/eval_final.json` (2026-11-13, zero-shot, unknown)

| Metric | A | B | Δ (B−A) | Interpretation |
|---|---|---|---|---|
| ROUGE-L | 0.5342 | 0.7843 | +0.2501 | above noise threshold |
| Valid ROUGE-L | 0.5342 | 0.7843 | +0.2501 | above noise threshold |
| Exact Match | 0.3289 | 0.5437 | +0.2148 | above noise threshold |
| Non-copy Tok Acc | 0.4288 | 0.6499 | +0.2211 | above noise threshold |
| Pass@1 | 0.6435 | 0.9150 | +0.2715 | above noise threshold |
| Validity | 1.0000 | 1.0000 | +0.0000 | NOT GENERALIZABLE (|delta| < 0.02): reproducible (training is bit-deterministic at fixed seed) but single-seed, so it may invert under another seed — do NOT interpret |
| BLEU (corpus) | 0.4540 | 0.6967 | +0.2427 | above noise threshold |
| BLEU (sent) | 0.4641 | 0.7090 | +0.2449 | above noise threshold |
| chrF2 (corpus) | 50.8174 | 72.3738 | +21.5564 | scale 0-100: 0.02 noise threshold not applicable |
| chrF2 (sent) | 53.5267 | 77.7119 | +24.1853 | scale 0-100: 0.02 noise threshold not applicable |
| Gloss F1 (mic) | 0.4439 | 0.6497 | +0.2058 | above noise threshold |
| Gloss F1 (sent) | 0.4507 | 0.6833 | +0.2325 | above noise threshold |
- ⚠️ HIGHLIGHT: same model evaluated in both prompting modes — how much does it depend on the prompt as a crutch?
- ⚠️ factors not deducible on at least one side, equality NOT verified: reward_stack, rl_objective

🔴 **prompt-dependence (dual eval, same checkpoint)** — `conll-2003/qwen25-05b/ablations/sft-amount/sft-ep015/run_20261113_113133/eval_final__few-shot.json` (2026-11-13, few-shot, unknown) → `conll-2003/qwen25-05b/ablations/sft-amount/sft-ep015/run_20261113_113133/eval_final.json` (2026-11-13, zero-shot, unknown)

| Metric | A | B | Δ (B−A) | Interpretation |
|---|---|---|---|---|
| ROUGE-L | 0.3628 | 0.5237 | +0.1609 | above noise threshold |
| Valid ROUGE-L | 0.3628 | 0.5237 | +0.1609 | above noise threshold |
| Exact Match | 0.2410 | 0.3236 | +0.0826 | above noise threshold |
| Non-copy Tok Acc | 0.2632 | 0.4110 | +0.1478 | above noise threshold |
| Pass@1 | 0.4300 | 0.6355 | +0.2055 | above noise threshold |
| Validity | 0.9999 | 1.0000 | +0.0001 | NOT GENERALIZABLE (|delta| < 0.02): reproducible (training is bit-deterministic at fixed seed) but single-seed, so it may invert under another seed — do NOT interpret |
| BLEU (corpus) | 0.2298 | 0.4519 | +0.2220 | above noise threshold |
| BLEU (sent) | 0.3213 | 0.4568 | +0.1355 | above noise threshold |
| chrF2 (corpus) | 30.9730 | 50.2511 | +19.2782 | scale 0-100: 0.02 noise threshold not applicable |
| chrF2 (sent) | 36.9577 | 51.8350 | +14.8772 | scale 0-100: 0.02 noise threshold not applicable |
| Gloss F1 (mic) | 0.2928 | 0.4287 | +0.1359 | above noise threshold |
| Gloss F1 (sent) | 0.3075 | 0.4287 | +0.1212 | above noise threshold |
- ⚠️ HIGHLIGHT: same model evaluated in both prompting modes — how much does it depend on the prompt as a crutch?
- ⚠️ factors not deducible on at least one side, equality NOT verified: reward_stack, rl_objective

🔴 **prompt-dependence (dual eval, same checkpoint)** — `conll-2003/qwen25-05b/ablations/sft-amount/sft-ep005/run_20261113_073503/eval_final__few-shot.json` (2026-11-13, few-shot, unknown) → `conll-2003/qwen25-05b/ablations/sft-amount/sft-ep005/run_20261113_073503/eval_final.json` (2026-11-13, zero-shot, unknown)

| Metric | A | B | Δ (B−A) | Interpretation |
|---|---|---|---|---|
| ROUGE-L | 0.4281 | 0.3448 | -0.0833 | above noise threshold |
| Valid ROUGE-L | 0.4281 | 0.3448 | -0.0833 | above noise threshold |
| Exact Match | 0.2577 | 0.1433 | -0.1144 | above noise threshold |
| Non-copy Tok Acc | 0.3063 | 0.2171 | -0.0892 | above noise threshold |
| Pass@1 | 0.5285 | 0.4485 | -0.0800 | above noise threshold |
| Validity | 1.0000 | 1.0000 | +0.0000 | NOT GENERALIZABLE (|delta| < 0.02): reproducible (training is bit-deterministic at fixed seed) but single-seed, so it may invert under another seed — do NOT interpret |
| BLEU (corpus) | 0.3143 | 0.2770 | -0.0373 | above noise threshold |
| BLEU (sent) | 0.3545 | 0.2474 | -0.1070 | above noise threshold |
| chrF2 (corpus) | 36.9027 | 33.8297 | -3.0730 | scale 0-100: 0.02 noise threshold not applicable |
| chrF2 (sent) | 42.8579 | 35.6142 | -7.2438 | scale 0-100: 0.02 noise threshold not applicable |
| Gloss F1 (mic) | 0.3217 | 0.1991 | -0.1226 | above noise threshold |
| Gloss F1 (sent) | 0.3498 | 0.2189 | -0.1309 | above noise threshold |
- ⚠️ HIGHLIGHT: same model evaluated in both prompting modes — how much does it depend on the prompt as a crutch?
- ⚠️ factors not deducible on at least one side, equality NOT verified: reward_stack, rl_objective

🔴 **prompt-dependence (dual eval, same checkpoint)** — `conll-2003/qwen25-05b/ablations/decoding/hot-rollout-zero-shot/run_20261112_181510/eval_final__few-shot.json` (2026-11-12, few-shot, unknown) → `conll-2003/qwen25-05b/ablations/decoding/hot-rollout-zero-shot/run_20261112_181510/eval_final.json` (2026-11-12, zero-shot, unknown)

| Metric | A | B | Δ (B−A) | Interpretation |
|---|---|---|---|---|
| ROUGE-L | 0.4947 | 0.3554 | -0.1393 | above noise threshold |
| Valid ROUGE-L | 0.4947 | 0.3554 | -0.1393 | above noise threshold |
| Exact Match | 0.2173 | 0.1234 | -0.0939 | above noise threshold |
| Non-copy Tok Acc | 0.3859 | 0.2235 | -0.1623 | above noise threshold |
| Pass@1 | 0.6645 | 0.5095 | -0.1550 | above noise threshold |
| Validity | 1.0000 | 1.0000 | +0.0000 | NOT GENERALIZABLE (|delta| < 0.02): reproducible (training is bit-deterministic at fixed seed) but single-seed, so it may invert under another seed — do NOT interpret |
| BLEU (corpus) | 0.3120 | 0.2482 | -0.0638 | above noise threshold |
| BLEU (sent) | 0.3827 | 0.2648 | -0.1178 | above noise threshold |
| chrF2 (corpus) | 50.8083 | 36.2266 | -14.5817 | scale 0-100: 0.02 noise threshold not applicable |
| chrF2 (sent) | 51.7674 | 37.5701 | -14.1972 | scale 0-100: 0.02 noise threshold not applicable |
| Gloss F1 (mic) | 0.3071 | 0.1905 | -0.1165 | above noise threshold |
| Gloss F1 (sent) | 0.3651 | 0.2192 | -0.1459 | above noise threshold |
- ⚠️ HIGHLIGHT: same model evaluated in both prompting modes — how much does it depend on the prompt as a crutch?
- ⚠️ factors not deducible on at least one side, equality NOT verified: reward_stack, rl_objective

🔴 **prompt-dependence (dual eval, same checkpoint)** — `conll-2003/qwen25-05b/seeds/sft-zero-shot-s1/run_20261109_223700/eval_final__few-shot.json` (2026-11-09, few-shot, unknown) → `conll-2003/qwen25-05b/seeds/sft-zero-shot-s1/run_20261109_223700/eval_final.json` (2026-11-09, zero-shot, unknown)

| Metric | A | B | Δ (B−A) | Interpretation |
|---|---|---|---|---|
| ROUGE-L | 0.8122 | 0.9021 | +0.0899 | above noise threshold |
| Valid ROUGE-L | 0.8122 | 0.9021 | +0.0899 | above noise threshold |
| Exact Match | 0.5918 | 0.7472 | +0.1554 | above noise threshold |
| Non-copy Tok Acc | 0.7415 | 0.8309 | +0.0894 | above noise threshold |
| Pass@1 | 0.9255 | 0.9710 | +0.0455 | above noise threshold |
| Validity | 1.0000 | 1.0000 | +0.0000 | NOT GENERALIZABLE (|delta| < 0.02): reproducible (training is bit-deterministic at fixed seed) but single-seed, so it may invert under another seed — do NOT interpret |
| BLEU (corpus) | 0.7006 | 0.8432 | +0.1425 | above noise threshold |
| BLEU (sent) | 0.7462 | 0.8626 | +0.1164 | above noise threshold |
| chrF2 (corpus) | 78.1695 | 86.4902 | +8.3207 | scale 0-100: 0.02 noise threshold not applicable |
| chrF2 (sent) | 81.5882 | 89.6613 | +8.0730 | scale 0-100: 0.02 noise threshold not applicable |
| Gloss F1 (mic) | 0.7124 | 0.8283 | +0.1158 | above noise threshold |
| Gloss F1 (sent) | 0.7421 | 0.8452 | +0.1032 | above noise threshold |
- ⚠️ HIGHLIGHT: same model evaluated in both prompting modes — how much does it depend on the prompt as a crutch?
- ⚠️ factors not deducible on at least one side, equality NOT verified: reward_stack, rl_objective

🔴 **prompt-dependence (dual eval, same checkpoint)** — `conll-2003/qwen25-05b/ablations/rewards/entity-f1/run_20261020_184102/eval_final.json` (2026-10-20, few-shot, unknown) → `conll-2003/qwen25-05b/ablations/rewards/entity-f1/run_20261020_184102/eval_final__zero-shot.json` (2026-10-20, zero-shot, unknown)

| Metric | A | B | Δ (B−A) | Interpretation |
|---|---|---|---|---|
| ROUGE-L | 0.5350 | 0.2726 | -0.2625 | above noise threshold |
| Valid ROUGE-L | 0.5349 | 0.2724 | -0.2625 | above noise threshold |
| Exact Match | 0.3340 | 0.1432 | -0.1908 | above noise threshold |
| Non-copy Tok Acc | 0.4007 | 0.1441 | -0.2566 | above noise threshold |
| Pass@1 | 0.6595 | 0.3390 | -0.3205 | above noise threshold |
| Validity | 0.9998 | 0.9994 | -0.0004 | NOT GENERALIZABLE (|delta| < 0.02): reproducible (training is bit-deterministic at fixed seed) but single-seed, so it may invert under another seed — do NOT interpret |
| BLEU (corpus) | 0.3959 | 0.1300 | -0.2659 | above noise threshold |
| BLEU (sent) | 0.4509 | 0.1961 | -0.2548 | above noise threshold |
| chrF2 (corpus) | 46.9459 | 23.6592 | -23.2868 | scale 0-100: 0.02 noise threshold not applicable |
| chrF2 (sent) | 53.8466 | 30.0173 | -23.8293 | scale 0-100: 0.02 noise threshold not applicable |
| Gloss F1 (mic) | 0.3773 | 0.1200 | -0.2573 | above noise threshold |
| Gloss F1 (sent) | 0.4492 | 0.1770 | -0.2722 | above noise threshold |
- ⚠️ HIGHLIGHT: same model evaluated in both prompting modes — how much does it depend on the prompt as a crutch?
- ⚠️ factors not deducible on at least one side, equality NOT verified: reward_stack, rl_objective

🔴 **prompt-dependence (dual eval, same checkpoint)** — `conll-2003/qwen25-05b/ablations/rewards/entity-f1-zero-shot/run_20261020_133542/eval_final__few-shot.json` (2026-10-20, few-shot, unknown) → `conll-2003/qwen25-05b/ablations/rewards/entity-f1-zero-shot/run_20261020_133542/eval_final.json` (2026-10-20, zero-shot, unknown)

| Metric | A | B | Δ (B−A) | Interpretation |
|---|---|---|---|---|
| ROUGE-L | 0.3743 | 0.2831 | -0.0912 | above noise threshold |
| Valid ROUGE-L | 0.3741 | 0.2830 | -0.0911 | above noise threshold |
| Exact Match | 0.0576 | 0.0415 | -0.0161 | NOT GENERALIZABLE (|delta| < 0.02): reproducible (training is bit-deterministic at fixed seed) but single-seed, so it may invert under another seed — do NOT interpret |
| Non-copy Tok Acc | 0.3075 | 0.1732 | -0.1343 | above noise threshold |
| Pass@1 | 0.5700 | 0.4325 | -0.1375 | above noise threshold |
| Validity | 0.9995 | 0.9994 | -0.0001 | NOT GENERALIZABLE (|delta| < 0.02): reproducible (training is bit-deterministic at fixed seed) but single-seed, so it may invert under another seed — do NOT interpret |
| BLEU (corpus) | 0.1943 | 0.1333 | -0.0610 | above noise threshold |
| BLEU (sent) | 0.2163 | 0.1453 | -0.0711 | above noise threshold |
| chrF2 (corpus) | 46.0106 | 36.5978 | -9.4128 | scale 0-100: 0.02 noise threshold not applicable |
| chrF2 (sent) | 42.9608 | 33.3169 | -9.6439 | scale 0-100: 0.02 noise threshold not applicable |
| Gloss F1 (mic) | 0.1915 | 0.1020 | -0.0895 | above noise threshold |
| Gloss F1 (sent) | 0.2203 | 0.1165 | -0.1039 | above noise threshold |
- ⚠️ HIGHLIGHT: same model evaluated in both prompting modes — how much does it depend on the prompt as a crutch?
- ⚠️ factors not deducible on at least one side, equality NOT verified: reward_stack, rl_objective

🔴 **prompt-dependence (dual eval, same checkpoint)** — `conll-2003/qwen25-05b/seeds/grpo-zero-shot-s1/run_20261019_202106/eval_final__few-shot.json` (2026-10-19, few-shot, unknown) → `conll-2003/qwen25-05b/seeds/grpo-zero-shot-s1/run_20261019_202106/eval_final.json` (2026-10-19, zero-shot, unknown)

| Metric | A | B | Δ (B−A) | Interpretation |
|---|---|---|---|---|
| ROUGE-L | 0.4153 | 0.3231 | -0.0921 | above noise threshold |
| Valid ROUGE-L | 0.4151 | 0.3231 | -0.0920 | above noise threshold |
| Exact Match | 0.0883 | 0.0378 | -0.0505 | above noise threshold |
| Non-copy Tok Acc | 0.3205 | 0.2016 | -0.1189 | above noise threshold |
| Pass@1 | 0.6240 | 0.5090 | -0.1150 | above noise threshold |
| Validity | 0.9997 | 1.0000 | +0.0003 | NOT GENERALIZABLE (|delta| < 0.02): reproducible (training is bit-deterministic at fixed seed) but single-seed, so it may invert under another seed — do NOT interpret |
| BLEU (corpus) | 0.2544 | 0.1924 | -0.0619 | above noise threshold |
| BLEU (sent) | 0.2664 | 0.1827 | -0.0837 | above noise threshold |
| chrF2 (corpus) | 49.4475 | 40.9927 | -8.4549 | scale 0-100: 0.02 noise threshold not applicable |
| chrF2 (sent) | 46.0264 | 37.0738 | -8.9526 | scale 0-100: 0.02 noise threshold not applicable |
| Gloss F1 (mic) | 0.2311 | 0.1400 | -0.0911 | above noise threshold |
| Gloss F1 (sent) | 0.2469 | 0.1400 | -0.1069 | above noise threshold |
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

