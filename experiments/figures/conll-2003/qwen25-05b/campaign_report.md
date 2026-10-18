# Campaign Report — conll-2003 / qwen25-05b

Generated: 2026-10-18T13:39:37+00:00

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
   - [method] conll-2003/qwen25-05b/ablations/decoding/no-grammar/run_20261009_133123/eval_baseline.json vs conll-2003/qwen25-05b/sft-grpo/few-shot/run_20261011_000134/eval_final.json: prompt_context_fingerprint DIFFERS: the eval contexts (prompting/grammar/num_samples/model) are not identical — this pair may differ in more than the method factor
   - [method] conll-2003/qwen25-05b/grpo/few-shot/run_20261009_052104/eval_baseline.json vs conll-2003/qwen25-05b/sft-grpo/zero-shot/run_20261011_020559/eval_final__few-shot.json: prompt_context_fingerprint DIFFERS: the eval contexts (prompting/grammar/num_samples/model) are not identical — this pair may differ in more than the method factor
   - [method] conll-2003/qwen25-05b/ablations/decoding/no-grammar-zero-shot/run_20261009_162603/eval_baseline.json vs conll-2003/qwen25-05b/sft-grpo/few-shot/run_20261011_000134/eval_final__zero-shot.json: prompt_context_fingerprint DIFFERS: the eval contexts (prompting/grammar/num_samples/model) are not identical — this pair may differ in more than the method factor
   - [method] conll-2003/qwen25-05b/grpo/few-shot/run_20261009_052104/eval_baseline__zero-shot.json vs conll-2003/qwen25-05b/sft-grpo/zero-shot/run_20261011_020559/eval_final.json: prompt_context_fingerprint DIFFERS: the eval contexts (prompting/grammar/num_samples/model) are not identical — this pair may differ in more than the method factor

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
| `conll-2003/qwen25-05b/ablations/decoding/no-grammar-zero-shot/run_20261011_065106/eval_final.json` | unknown | conll-2003-qwen25-05b-ablations-decoding-no-grammar-zero-shot | zero-shot | unknown | 2026-10-11 | 2000 | 2 |
| `conll-2003/qwen25-05b/ablations/decoding/no-grammar/run_20261009_133123/eval_baseline.json` | baseline | conll-2003-qwen25-05b-ablations-decoding-no-grammar | few-shot | unknown | 2026-10-09 | 2000 | 2 |
| `conll-2003/qwen25-05b/ablations/decoding/no-grammar/run_20261011_042103/eval_final.json` | unknown | conll-2003-qwen25-05b-ablations-decoding-no-grammar | few-shot | unknown | 2026-10-11 | 2000 | 2 |
| `conll-2003/qwen25-05b/ablations/objectives/sft-allowed-mass/run_20261011_091612/eval_final.json` | unknown | conll-2003-qwen25-05b-ablations-objectives-sft-allowed-mass | zero-shot | unknown | 2026-10-11 | 2000 | 2 |
| `conll-2003/qwen25-05b/baseline/few-shot/run_20261010_153552/eval_zero_shot.json` | unknown | conll-2003-qwen25-05b-baseline-few-shot | few-shot | unknown | 2026-10-10 | 2000 | 2 |
| `conll-2003/qwen25-05b/baseline/zero-shot-no-grammar/run_20261010_155550/eval_zero_shot.json` | unknown | conll-2003-qwen25-05b-baseline-zero-shot-no-grammar | zero-shot | unknown | 2026-10-10 | 2000 | 2 |
| `conll-2003/qwen25-05b/baseline/zero-shot/run_20261010_151445/eval_zero_shot.json` | unknown | conll-2003-qwen25-05b-baseline-zero-shot | zero-shot | unknown | 2026-10-10 | 2000 | 2 |
| `conll-2003/qwen25-05b/grpo/few-shot/run_20261009_052104/eval_baseline.json` | baseline | conll-2003-qwen25-05b-grpo-few-shot | few-shot | unknown | 2026-10-09 | 2000 | 2 |
| `conll-2003/qwen25-05b/grpo/few-shot/run_20261009_052104/eval_baseline__zero-shot.json` | baseline | conll-2003-qwen25-05b-grpo-few-shot | zero-shot | unknown | 2026-10-09 | 2000 | 2 |
| `conll-2003/qwen25-05b/grpo/few-shot/run_20261010_204103/eval_final.json` | unknown | conll-2003-qwen25-05b-grpo-few-shot | few-shot | unknown | 2026-10-10 | 2000 | 2 |
| `conll-2003/qwen25-05b/grpo/few-shot/run_20261010_204103/eval_final__zero-shot.json` | unknown | conll-2003-qwen25-05b-grpo-few-shot | zero-shot | unknown | 2026-10-10 | 2000 | 2 |
| `conll-2003/qwen25-05b/grpo/zero-shot/run_20261010_170559/eval_final.json` | unknown | conll-2003-qwen25-05b-grpo-zero-shot | zero-shot | unknown | 2026-10-10 | 2000 | 2 |
| `conll-2003/qwen25-05b/grpo/zero-shot/run_20261010_170559/eval_final__few-shot.json` | unknown | conll-2003-qwen25-05b-grpo-zero-shot | few-shot | unknown | 2026-10-10 | 2000 | 2 |
| `conll-2003/qwen25-05b/sft-grpo/few-shot/run_20261011_000134/eval_final.json` | sft-grpo | few-shot | few-shot | unknown | 2026-10-11 | 2000 | 2 |
| `conll-2003/qwen25-05b/sft-grpo/few-shot/run_20261011_000134/eval_final__zero-shot.json` | sft-grpo | few-shot | zero-shot | unknown | 2026-10-11 | 2000 | 2 |
| `conll-2003/qwen25-05b/sft-grpo/zero-shot/run_20261011_020559/eval_final.json` | sft-grpo | zero-shot | zero-shot | unknown | 2026-10-11 | 2000 | 2 |
| `conll-2003/qwen25-05b/sft-grpo/zero-shot/run_20261011_020559/eval_final__few-shot.json` | sft-grpo | zero-shot | few-shot | unknown | 2026-10-11 | 2000 | 2 |
| `conll-2003/qwen25-05b/sft/zero-shot/run_20261009_004112/eval_baseline.json` | baseline | conll-2003-qwen25-05b-sft-zero-shot | zero-shot | unknown | 2026-10-09 | 2000 | 2 |
| `conll-2003/qwen25-05b/sft/zero-shot/run_20261009_004112/eval_baseline__few-shot.json` | baseline | conll-2003-qwen25-05b-sft-zero-shot | few-shot | unknown | 2026-10-09 | 2000 | 2 |
| `conll-2003/qwen25-05b/sft/zero-shot/run_20261010_162118/eval_final.json` | unknown | conll-2003-qwen25-05b-sft-zero-shot | zero-shot | unknown | 2026-10-10 | 2000 | 2 |
| `conll-2003/qwen25-05b/sft/zero-shot/run_20261010_162118/eval_final__few-shot.json` | unknown | conll-2003-qwen25-05b-sft-zero-shot | few-shot | unknown | 2026-10-10 | 2000 | 2 |

Superseded (older runs of the same typology): 16 (see JSON for the full list).

## Ablation matrix overview — ROUGE-L

One row per cell (its path under `results/`), one column per prompting mode that has at least one value. `—` = the cell was not evaluated in that mode. `*` = multiple runs collapsed, latest shown. Each cell's own `eval_baseline` (the base model in that cell's context) is not a row here: it repeats the `baseline/` cells and stays in the paired comparisons below.

| cell | zero-shot | few-shot |
|---|---|---|
| conll-2003/qwen25-05b/baseline/few-shot | — | 0.3757 |
| conll-2003/qwen25-05b/baseline/zero-shot | 0.2712 | — |
| conll-2003/qwen25-05b/baseline/zero-shot-no-grammar | 0.2676 | — |
| conll-2003/qwen25-05b/sft/zero-shot | 0.9002 | 0.8140 |
| conll-2003/qwen25-05b/sft-grpo/few-shot | 0.2403 | 0.5090 |
| conll-2003/qwen25-05b/sft-grpo/zero-shot | 0.2836 | 0.4302 |
| conll-2003/qwen25-05b/grpo/few-shot | 0.2649 | 0.5470 |
| conll-2003/qwen25-05b/grpo/zero-shot | 0.3382 | 0.4385 |
| conll-2003/qwen25-05b/ablations/decoding/no-grammar | — | 0.6222 |
| conll-2003/qwen25-05b/ablations/decoding/no-grammar-zero-shot | 0.4640 | — |
| conll-2003/qwen25-05b/ablations/objectives/sft-allowed-mass | 0.8992 | — |

## conll-2003 — all metrics

Same rows as the matrix above, restricted to this dataset, one column per metric. On CoNLL-2003 and WOS-46985 read Exact Match and Gloss F1 (micro = entity-level F1 on CoNLL): ROUGE-L and BLEU split `PER:Mary` into `per` + `mary` and give partial credit to a wrong label of the same type.

| cell | prompting | ROUGE-L | Valid ROUGE-L | Exact Match | Non-copy Tok Acc | Pass@1 | Validity | BLEU (corpus) | BLEU (sent) | chrF2 (corpus) | chrF2 (sent) | Gloss F1 (mic) | Gloss F1 (sent) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| conll-2003/qwen25-05b/baseline/few-shot | few-shot | 0.3757 | 0.3684 | 0.0708 | 0.2897 | 0.5805 | 0.9806 | 0.1269 | 0.2191 | 39.2988 | 40.8535 | 0.1780 | 0.2248 |
| conll-2003/qwen25-05b/baseline/zero-shot | zero-shot | 0.2712 | 0.2648 | 0.0470 | 0.1351 | 0.4070 | 0.9766 | 0.0859 | 0.1329 | 30.0026 | 29.5740 | 0.0809 | 0.1015 |
| conll-2003/qwen25-05b/baseline/zero-shot-no-grammar | zero-shot | 0.2676 | 0.0017 | 0.0011 | 0.0066 | 0.4260 | 0.0064 | 0.0089 | 0.0219 | 17.9933 | 18.2289 | 0.0022 | 0.0056 |
| conll-2003/qwen25-05b/sft/zero-shot | zero-shot | 0.9002 | 0.9002 | 0.7434 | 0.8225 | 0.9670 | 1.0000 | 0.8321 | 0.8590 | 85.7425 | 89.3654 | 0.8227 | 0.8410 |
| conll-2003/qwen25-05b/sft/zero-shot | few-shot | 0.8140 | 0.8140 | 0.5980 | 0.7374 | 0.9230 | 1.0000 | 0.6708 | 0.7449 | 77.0962 | 81.4134 | 0.7206 | 0.7462 |
| conll-2003/qwen25-05b/sft-grpo/few-shot | zero-shot | 0.2403 | 0.2393 | 0.0936 | 0.1028 | 0.3450 | 0.9957 | 0.1201 | 0.1577 | 22.3706 | 25.2648 | 0.0920 | 0.1250 |
| conll-2003/qwen25-05b/sft-grpo/few-shot | few-shot | 0.5090 | 0.5068 | 0.2487 | 0.3864 | 0.6685 | 0.9956 | 0.2493 | 0.3896 | 45.5260 | 52.4923 | 0.3223 | 0.3995 |
| conll-2003/qwen25-05b/sft-grpo/zero-shot | zero-shot | 0.2836 | 0.2786 | 0.0902 | 0.1512 | 0.4045 | 0.9825 | 0.1013 | 0.1727 | 28.8423 | 30.1000 | 0.1051 | 0.1431 |
| conll-2003/qwen25-05b/sft-grpo/zero-shot | few-shot | 0.4302 | 0.4204 | 0.1297 | 0.3647 | 0.6230 | 0.9774 | 0.1445 | 0.2825 | 42.6459 | 46.8101 | 0.2286 | 0.3060 |
| conll-2003/qwen25-05b/grpo/few-shot | zero-shot | 0.2649 | 0.2624 | 0.1627 | 0.1462 | 0.3310 | 0.9907 | 0.0960 | 0.2031 | 20.3531 | 28.6782 | 0.1236 | 0.1900 |
| conll-2003/qwen25-05b/grpo/few-shot | few-shot | 0.5470 | 0.5451 | 0.3243 | 0.4167 | 0.6895 | 0.9966 | 0.3074 | 0.4530 | 47.9039 | 54.9969 | 0.3700 | 0.4508 |
| conll-2003/qwen25-05b/grpo/zero-shot | zero-shot | 0.3382 | 0.3375 | 0.0382 | 0.2150 | 0.5295 | 0.9981 | 0.1801 | 0.1959 | 41.0512 | 37.7640 | 0.1481 | 0.1523 |
| conll-2003/qwen25-05b/grpo/zero-shot | few-shot | 0.4385 | 0.4370 | 0.1118 | 0.3686 | 0.6410 | 0.9967 | 0.2103 | 0.2901 | 47.9816 | 47.8211 | 0.2611 | 0.3041 |
| conll-2003/qwen25-05b/ablations/decoding/no-grammar | few-shot | 0.6222 | 0.3699 | 0.3426 | 0.4172 | 0.7790 | 0.5945 | 0.3733 | 0.4786 | 49.6825 | 59.1854 | 0.3224 | 0.4695 |
| conll-2003/qwen25-05b/ablations/decoding/no-grammar-zero-shot | zero-shot | 0.4640 | 0.0005 | 0.0004 | 0.1079 | 0.8320 | 0.0011 | 0.0205 | 0.0564 | 29.3498 | 34.5916 | 0.0514 | 0.0764 |
| conll-2003/qwen25-05b/ablations/objectives/sft-allowed-mass | zero-shot | 0.8992 | 0.8992 | 0.7384 | 0.8240 | 0.9670 | 1.0000 | 0.8241 | 0.8569 | 85.4486 | 89.2719 | 0.8228 | 0.8432 |

## Paired comparisons — one table per factor

### Factor: `method`

**method effect (cross-run)** — `conll-2003/qwen25-05b/ablations/decoding/no-grammar/run_20261009_133123/eval_baseline.json` (2026-10-09, few-shot, unknown) → `conll-2003/qwen25-05b/sft-grpo/few-shot/run_20261011_000134/eval_final.json` (2026-10-11, few-shot, unknown)

| Metric | A | B | Δ (B−A) | Interpretation |
|---|---|---|---|---|
| ROUGE-L | 0.3700 | 0.5090 | +0.1390 | above noise threshold |
| Valid ROUGE-L | 0.0635 | 0.5068 | +0.4433 | above noise threshold |
| Exact Match | 0.0344 | 0.2487 | +0.2143 | above noise threshold |
| Non-copy Tok Acc | 0.1922 | 0.3864 | +0.1942 | above noise threshold |
| Pass@1 | 0.5440 | 0.6685 | +0.1245 | above noise threshold |
| Validity | 0.1716 | 0.9956 | +0.8240 | above noise threshold |
| BLEU (corpus) | 0.0961 | 0.2493 | +0.1532 | above noise threshold |
| BLEU (sent) | 0.1556 | 0.3896 | +0.2340 | above noise threshold |
| chrF2 (corpus) | 33.2195 | 45.5260 | +12.3065 | scale 0-100: 0.02 noise threshold not applicable |
| chrF2 (sent) | 35.9071 | 52.4923 | +16.5853 | scale 0-100: 0.02 noise threshold not applicable |
| Gloss F1 (mic) | 0.0839 | 0.3223 | +0.2384 | above noise threshold |
| Gloss F1 (sent) | 0.1436 | 0.3995 | +0.2560 | above noise threshold |
- ⚠️ factors not deducible on at least one side, equality NOT verified: reward_stack, rl_objective
- ⚠️ prompt_context_fingerprint DIFFERS: the eval contexts (prompting/grammar/num_samples/model) are not identical — this pair may differ in more than the method factor

**method effect (cross-run)** — `conll-2003/qwen25-05b/grpo/few-shot/run_20261009_052104/eval_baseline.json` (2026-10-09, few-shot, unknown) → `conll-2003/qwen25-05b/sft-grpo/zero-shot/run_20261011_020559/eval_final__few-shot.json` (2026-10-11, few-shot, unknown)

| Metric | A | B | Δ (B−A) | Interpretation |
|---|---|---|---|---|
| ROUGE-L | 0.3805 | 0.4302 | +0.0496 | above noise threshold |
| Valid ROUGE-L | 0.3706 | 0.4204 | +0.0498 | above noise threshold |
| Exact Match | 0.0662 | 0.1297 | +0.0635 | above noise threshold |
| Non-copy Tok Acc | 0.3169 | 0.3647 | +0.0478 | above noise threshold |
| Pass@1 | 0.5865 | 0.6230 | +0.0365 | above noise threshold |
| Validity | 0.9739 | 0.9774 | +0.0035 | NOT GENERALIZABLE (|delta| < 0.02): reproducible (training is bit-deterministic at fixed seed) but single-seed, so it may invert under another seed — do NOT interpret |
| BLEU (corpus) | 0.1429 | 0.1445 | +0.0016 | NOT GENERALIZABLE (|delta| < 0.02): reproducible (training is bit-deterministic at fixed seed) but single-seed, so it may invert under another seed — do NOT interpret |
| BLEU (sent) | 0.2204 | 0.2825 | +0.0621 | above noise threshold |
| chrF2 (corpus) | 39.8261 | 42.6459 | +2.8198 | scale 0-100: 0.02 noise threshold not applicable |
| chrF2 (sent) | 41.4512 | 46.8101 | +5.3588 | scale 0-100: 0.02 noise threshold not applicable |
| Gloss F1 (mic) | 0.1714 | 0.2286 | +0.0572 | above noise threshold |
| Gloss F1 (sent) | 0.2366 | 0.3060 | +0.0694 | above noise threshold |
- ⚠️ factors not deducible on at least one side, equality NOT verified: reward_stack, rl_objective
- ⚠️ prompt_context_fingerprint DIFFERS: the eval contexts (prompting/grammar/num_samples/model) are not identical — this pair may differ in more than the method factor

**method effect (cross-run)** — `conll-2003/qwen25-05b/ablations/decoding/no-grammar-zero-shot/run_20261009_162603/eval_baseline.json` (2026-10-09, zero-shot, unknown) → `conll-2003/qwen25-05b/sft-grpo/few-shot/run_20261011_000134/eval_final__zero-shot.json` (2026-10-11, zero-shot, unknown)

| Metric | A | B | Δ (B−A) | Interpretation |
|---|---|---|---|---|
| ROUGE-L | 0.2676 | 0.2403 | -0.0273 | above noise threshold |
| Valid ROUGE-L | 0.0021 | 0.2393 | +0.2372 | above noise threshold |
| Exact Match | 0.0011 | 0.0936 | +0.0925 | above noise threshold |
| Non-copy Tok Acc | 0.0066 | 0.1028 | +0.0962 | above noise threshold |
| Pass@1 | 0.4260 | 0.3450 | -0.0810 | above noise threshold |
| Validity | 0.0079 | 0.9957 | +0.9878 | above noise threshold |
| BLEU (corpus) | 0.0089 | 0.1201 | +0.1112 | above noise threshold |
| BLEU (sent) | 0.0219 | 0.1577 | +0.1359 | above noise threshold |
| chrF2 (corpus) | 17.9933 | 22.3706 | +4.3772 | scale 0-100: 0.02 noise threshold not applicable |
| chrF2 (sent) | 18.2289 | 25.2648 | +7.0359 | scale 0-100: 0.02 noise threshold not applicable |
| Gloss F1 (mic) | 0.0022 | 0.0920 | +0.0898 | above noise threshold |
| Gloss F1 (sent) | 0.0056 | 0.1250 | +0.1194 | above noise threshold |
- ⚠️ factors not deducible on at least one side, equality NOT verified: reward_stack, rl_objective
- ⚠️ prompt_context_fingerprint DIFFERS: the eval contexts (prompting/grammar/num_samples/model) are not identical — this pair may differ in more than the method factor

**method effect (cross-run)** — `conll-2003/qwen25-05b/grpo/few-shot/run_20261009_052104/eval_baseline__zero-shot.json` (2026-10-09, zero-shot, unknown) → `conll-2003/qwen25-05b/sft-grpo/zero-shot/run_20261011_020559/eval_final.json` (2026-10-11, zero-shot, unknown)

| Metric | A | B | Δ (B−A) | Interpretation |
|---|---|---|---|---|
| ROUGE-L | 0.2708 | 0.2836 | +0.0128 | NOT GENERALIZABLE (|delta| < 0.02): reproducible (training is bit-deterministic at fixed seed) but single-seed, so it may invert under another seed — do NOT interpret |
| Valid ROUGE-L | 0.2625 | 0.2786 | +0.0161 | NOT GENERALIZABLE (|delta| < 0.02): reproducible (training is bit-deterministic at fixed seed) but single-seed, so it may invert under another seed — do NOT interpret |
| Exact Match | 0.0453 | 0.0902 | +0.0449 | above noise threshold |
| Non-copy Tok Acc | 0.1458 | 0.1512 | +0.0055 | NOT GENERALIZABLE (|delta| < 0.02): reproducible (training is bit-deterministic at fixed seed) but single-seed, so it may invert under another seed — do NOT interpret |
| Pass@1 | 0.4145 | 0.4045 | -0.0100 | NOT GENERALIZABLE (|delta| < 0.02): reproducible (training is bit-deterministic at fixed seed) but single-seed, so it may invert under another seed — do NOT interpret |
| Validity | 0.9693 | 0.9825 | +0.0132 | NOT GENERALIZABLE (|delta| < 0.02): reproducible (training is bit-deterministic at fixed seed) but single-seed, so it may invert under another seed — do NOT interpret |
| BLEU (corpus) | 0.0893 | 0.1013 | +0.0120 | NOT GENERALIZABLE (|delta| < 0.02): reproducible (training is bit-deterministic at fixed seed) but single-seed, so it may invert under another seed — do NOT interpret |
| BLEU (sent) | 0.1329 | 0.1727 | +0.0398 | above noise threshold |
| chrF2 (corpus) | 29.5092 | 28.8423 | -0.6669 | scale 0-100: 0.02 noise threshold not applicable |
| chrF2 (sent) | 29.5764 | 30.1000 | +0.5237 | scale 0-100: 0.02 noise threshold not applicable |
| Gloss F1 (mic) | 0.0762 | 0.1051 | +0.0289 | above noise threshold |
| Gloss F1 (sent) | 0.1033 | 0.1431 | +0.0398 | above noise threshold |
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

🔴 **prompt-dependence (dual eval, same checkpoint)** — `conll-2003/qwen25-05b/sft-grpo/zero-shot/run_20261011_020559/eval_final__few-shot.json` (2026-10-11, few-shot, unknown) → `conll-2003/qwen25-05b/sft-grpo/zero-shot/run_20261011_020559/eval_final.json` (2026-10-11, zero-shot, unknown)

| Metric | A | B | Δ (B−A) | Interpretation |
|---|---|---|---|---|
| ROUGE-L | 0.4302 | 0.2836 | -0.1466 | above noise threshold |
| Valid ROUGE-L | 0.4204 | 0.2786 | -0.1418 | above noise threshold |
| Exact Match | 0.1297 | 0.0902 | -0.0395 | above noise threshold |
| Non-copy Tok Acc | 0.3647 | 0.1512 | -0.2135 | above noise threshold |
| Pass@1 | 0.6230 | 0.4045 | -0.2185 | above noise threshold |
| Validity | 0.9774 | 0.9825 | +0.0051 | NOT GENERALIZABLE (|delta| < 0.02): reproducible (training is bit-deterministic at fixed seed) but single-seed, so it may invert under another seed — do NOT interpret |
| BLEU (corpus) | 0.1445 | 0.1013 | -0.0432 | above noise threshold |
| BLEU (sent) | 0.2825 | 0.1727 | -0.1098 | above noise threshold |
| chrF2 (corpus) | 42.6459 | 28.8423 | -13.8036 | scale 0-100: 0.02 noise threshold not applicable |
| chrF2 (sent) | 46.8101 | 30.1000 | -16.7100 | scale 0-100: 0.02 noise threshold not applicable |
| Gloss F1 (mic) | 0.2286 | 0.1051 | -0.1235 | above noise threshold |
| Gloss F1 (sent) | 0.3060 | 0.1431 | -0.1629 | above noise threshold |
- ⚠️ HIGHLIGHT: same model evaluated in both prompting modes — how much does it depend on the prompt as a crutch?
- ⚠️ factors not deducible on at least one side, equality NOT verified: reward_stack, rl_objective

🔴 **prompt-dependence (dual eval, same checkpoint)** — `conll-2003/qwen25-05b/sft-grpo/few-shot/run_20261011_000134/eval_final.json` (2026-10-11, few-shot, unknown) → `conll-2003/qwen25-05b/sft-grpo/few-shot/run_20261011_000134/eval_final__zero-shot.json` (2026-10-11, zero-shot, unknown)

| Metric | A | B | Δ (B−A) | Interpretation |
|---|---|---|---|---|
| ROUGE-L | 0.5090 | 0.2403 | -0.2687 | above noise threshold |
| Valid ROUGE-L | 0.5068 | 0.2393 | -0.2675 | above noise threshold |
| Exact Match | 0.2487 | 0.0936 | -0.1551 | above noise threshold |
| Non-copy Tok Acc | 0.3864 | 0.1028 | -0.2836 | above noise threshold |
| Pass@1 | 0.6685 | 0.3450 | -0.3235 | above noise threshold |
| Validity | 0.9956 | 0.9957 | +0.0001 | NOT GENERALIZABLE (|delta| < 0.02): reproducible (training is bit-deterministic at fixed seed) but single-seed, so it may invert under another seed — do NOT interpret |
| BLEU (corpus) | 0.2493 | 0.1201 | -0.1292 | above noise threshold |
| BLEU (sent) | 0.3896 | 0.1577 | -0.2319 | above noise threshold |
| chrF2 (corpus) | 45.5260 | 22.3706 | -23.1555 | scale 0-100: 0.02 noise threshold not applicable |
| chrF2 (sent) | 52.4923 | 25.2648 | -27.2275 | scale 0-100: 0.02 noise threshold not applicable |
| Gloss F1 (mic) | 0.3223 | 0.0920 | -0.2304 | above noise threshold |
| Gloss F1 (sent) | 0.3995 | 0.1250 | -0.2745 | above noise threshold |
- ⚠️ HIGHLIGHT: same model evaluated in both prompting modes — how much does it depend on the prompt as a crutch?
- ⚠️ factors not deducible on at least one side, equality NOT verified: reward_stack, rl_objective

🔴 **prompt-dependence (dual eval, same checkpoint)** — `conll-2003/qwen25-05b/grpo/few-shot/run_20261010_204103/eval_final.json` (2026-10-10, few-shot, unknown) → `conll-2003/qwen25-05b/grpo/few-shot/run_20261010_204103/eval_final__zero-shot.json` (2026-10-10, zero-shot, unknown)

| Metric | A | B | Δ (B−A) | Interpretation |
|---|---|---|---|---|
| ROUGE-L | 0.5470 | 0.2649 | -0.2821 | above noise threshold |
| Valid ROUGE-L | 0.5451 | 0.2624 | -0.2827 | above noise threshold |
| Exact Match | 0.3243 | 0.1627 | -0.1616 | above noise threshold |
| Non-copy Tok Acc | 0.4167 | 0.1462 | -0.2705 | above noise threshold |
| Pass@1 | 0.6895 | 0.3310 | -0.3585 | above noise threshold |
| Validity | 0.9966 | 0.9907 | -0.0059 | NOT GENERALIZABLE (|delta| < 0.02): reproducible (training is bit-deterministic at fixed seed) but single-seed, so it may invert under another seed — do NOT interpret |
| BLEU (corpus) | 0.3074 | 0.0960 | -0.2114 | above noise threshold |
| BLEU (sent) | 0.4530 | 0.2031 | -0.2499 | above noise threshold |
| chrF2 (corpus) | 47.9039 | 20.3531 | -27.5508 | scale 0-100: 0.02 noise threshold not applicable |
| chrF2 (sent) | 54.9969 | 28.6782 | -26.3187 | scale 0-100: 0.02 noise threshold not applicable |
| Gloss F1 (mic) | 0.3700 | 0.1236 | -0.2464 | above noise threshold |
| Gloss F1 (sent) | 0.4508 | 0.1900 | -0.2608 | above noise threshold |
- ⚠️ HIGHLIGHT: same model evaluated in both prompting modes — how much does it depend on the prompt as a crutch?
- ⚠️ factors not deducible on at least one side, equality NOT verified: reward_stack, rl_objective

🔴 **prompt-dependence (dual eval, same checkpoint)** — `conll-2003/qwen25-05b/grpo/zero-shot/run_20261010_170559/eval_final__few-shot.json` (2026-10-10, few-shot, unknown) → `conll-2003/qwen25-05b/grpo/zero-shot/run_20261010_170559/eval_final.json` (2026-10-10, zero-shot, unknown)

| Metric | A | B | Δ (B−A) | Interpretation |
|---|---|---|---|---|
| ROUGE-L | 0.4385 | 0.3382 | -0.1003 | above noise threshold |
| Valid ROUGE-L | 0.4370 | 0.3375 | -0.0995 | above noise threshold |
| Exact Match | 0.1118 | 0.0382 | -0.0736 | above noise threshold |
| Non-copy Tok Acc | 0.3686 | 0.2150 | -0.1536 | above noise threshold |
| Pass@1 | 0.6410 | 0.5295 | -0.1115 | above noise threshold |
| Validity | 0.9967 | 0.9981 | +0.0014 | NOT GENERALIZABLE (|delta| < 0.02): reproducible (training is bit-deterministic at fixed seed) but single-seed, so it may invert under another seed — do NOT interpret |
| BLEU (corpus) | 0.2103 | 0.1801 | -0.0302 | above noise threshold |
| BLEU (sent) | 0.2901 | 0.1959 | -0.0942 | above noise threshold |
| chrF2 (corpus) | 47.9816 | 41.0512 | -6.9304 | scale 0-100: 0.02 noise threshold not applicable |
| chrF2 (sent) | 47.8211 | 37.7640 | -10.0571 | scale 0-100: 0.02 noise threshold not applicable |
| Gloss F1 (mic) | 0.2611 | 0.1481 | -0.1130 | above noise threshold |
| Gloss F1 (sent) | 0.3041 | 0.1523 | -0.1517 | above noise threshold |
- ⚠️ HIGHLIGHT: same model evaluated in both prompting modes — how much does it depend on the prompt as a crutch?
- ⚠️ factors not deducible on at least one side, equality NOT verified: reward_stack, rl_objective

🔴 **prompt-dependence (dual eval, same checkpoint)** — `conll-2003/qwen25-05b/sft/zero-shot/run_20261010_162118/eval_final__few-shot.json` (2026-10-10, few-shot, unknown) → `conll-2003/qwen25-05b/sft/zero-shot/run_20261010_162118/eval_final.json` (2026-10-10, zero-shot, unknown)

| Metric | A | B | Δ (B−A) | Interpretation |
|---|---|---|---|---|
| ROUGE-L | 0.8140 | 0.9002 | +0.0862 | above noise threshold |
| Valid ROUGE-L | 0.8140 | 0.9002 | +0.0862 | above noise threshold |
| Exact Match | 0.5980 | 0.7434 | +0.1454 | above noise threshold |
| Non-copy Tok Acc | 0.7374 | 0.8225 | +0.0851 | above noise threshold |
| Pass@1 | 0.9230 | 0.9670 | +0.0440 | above noise threshold |
| Validity | 1.0000 | 1.0000 | +0.0000 | NOT GENERALIZABLE (|delta| < 0.02): reproducible (training is bit-deterministic at fixed seed) but single-seed, so it may invert under another seed — do NOT interpret |
| BLEU (corpus) | 0.6708 | 0.8321 | +0.1613 | above noise threshold |
| BLEU (sent) | 0.7449 | 0.8590 | +0.1141 | above noise threshold |
| chrF2 (corpus) | 77.0962 | 85.7425 | +8.6463 | scale 0-100: 0.02 noise threshold not applicable |
| chrF2 (sent) | 81.4134 | 89.3654 | +7.9520 | scale 0-100: 0.02 noise threshold not applicable |
| Gloss F1 (mic) | 0.7206 | 0.8227 | +0.1021 | above noise threshold |
| Gloss F1 (sent) | 0.7462 | 0.8410 | +0.0949 | above noise threshold |
- ⚠️ HIGHLIGHT: same model evaluated in both prompting modes — how much does it depend on the prompt as a crutch?
- ⚠️ factors not deducible on at least one side, equality NOT verified: reward_stack, rl_objective

**prompting (cross-run)** — `conll-2003/qwen25-05b/ablations/decoding/no-grammar/run_20261011_042103/eval_final.json` (2026-10-11, few-shot, unknown) → `conll-2003/qwen25-05b/ablations/decoding/no-grammar-zero-shot/run_20261011_065106/eval_final.json` (2026-10-11, zero-shot, unknown)

| Metric | A | B | Δ (B−A) | Interpretation |
|---|---|---|---|---|
| ROUGE-L | 0.6222 | 0.4640 | -0.1582 | above noise threshold |
| Valid ROUGE-L | 0.3699 | 0.0005 | -0.3694 | above noise threshold |
| Exact Match | 0.3426 | 0.0004 | -0.3422 | above noise threshold |
| Non-copy Tok Acc | 0.4172 | 0.1079 | -0.3093 | above noise threshold |
| Pass@1 | 0.7790 | 0.8320 | +0.0530 | above noise threshold |
| Validity | 0.5945 | 0.0011 | -0.5934 | above noise threshold |
| BLEU (corpus) | 0.3733 | 0.0205 | -0.3528 | above noise threshold |
| BLEU (sent) | 0.4786 | 0.0564 | -0.4223 | above noise threshold |
| chrF2 (corpus) | 49.6825 | 29.3498 | -20.3327 | scale 0-100: 0.02 noise threshold not applicable |
| chrF2 (sent) | 59.1854 | 34.5916 | -24.5938 | scale 0-100: 0.02 noise threshold not applicable |
| Gloss F1 (mic) | 0.3224 | 0.0514 | -0.2710 | above noise threshold |
| Gloss F1 (sent) | 0.4695 | 0.0764 | -0.3931 | above noise threshold |
- ⚠️ cross-run comparison: checkpoint identity NOT guaranteed
- ⚠️ factors not deducible on at least one side, equality NOT verified: reward_stack, rl_objective

**prompting (cross-run)** — `conll-2003/qwen25-05b/ablations/objectives/sft-allowed-mass/run_20261011_091612/eval_final.json` (2026-10-11, zero-shot, unknown) → `conll-2003/qwen25-05b/baseline/few-shot/run_20261010_153552/eval_zero_shot.json` (2026-10-10, few-shot, unknown)

| Metric | A | B | Δ (B−A) | Interpretation |
|---|---|---|---|---|
| ROUGE-L | 0.8992 | 0.3757 | -0.5234 | above noise threshold |
| Valid ROUGE-L | 0.8992 | 0.3684 | -0.5307 | above noise threshold |
| Exact Match | 0.7384 | 0.0708 | -0.6676 | above noise threshold |
| Non-copy Tok Acc | 0.8240 | 0.2897 | -0.5343 | above noise threshold |
| Pass@1 | 0.9670 | 0.5805 | -0.3865 | above noise threshold |
| Validity | 1.0000 | 0.9806 | -0.0194 | NOT GENERALIZABLE (|delta| < 0.02): reproducible (training is bit-deterministic at fixed seed) but single-seed, so it may invert under another seed — do NOT interpret |
| BLEU (corpus) | 0.8241 | 0.1269 | -0.6973 | above noise threshold |
| BLEU (sent) | 0.8569 | 0.2191 | -0.6378 | above noise threshold |
| chrF2 (corpus) | 85.4486 | 39.2988 | -46.1498 | scale 0-100: 0.02 noise threshold not applicable |
| chrF2 (sent) | 89.2719 | 40.8535 | -48.4184 | scale 0-100: 0.02 noise threshold not applicable |
| Gloss F1 (mic) | 0.8228 | 0.1780 | -0.6448 | above noise threshold |
| Gloss F1 (sent) | 0.8432 | 0.2248 | -0.6184 | above noise threshold |
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

