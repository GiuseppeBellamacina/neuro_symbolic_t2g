# Campaign Report — wos-46985 / qwen25-05b

Generated: 2026-10-10T12:57:04+00:00

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
   - [method] wos-46985/qwen25-05b/ablations/decoding/no-grammar/run_20261010_030132/eval_baseline.json vs wos-46985/qwen25-05b/sft-grpo/zero-shot/run_20261010_000134/eval_final.json: prompt_context_fingerprint DIFFERS: the eval contexts (prompting/grammar/num_samples/model) are not identical — this pair may differ in more than the method factor

## How factors are deduced

| Factor | Rule | If not deducible |
|---|---|---|
| `method` | token del nome cella nel percorso (sft-grpo / sft-only / grpo-only / t2g-*); eval_baseline.json => baseline (nessun checkpoint) | unknown (never guessed) |
| `variant` | remaining cell-name tokens after model tag, method and known markers (e.g. structure, viterbi, all-rewards); no-grammar is a grammar marker, not a variant | empty string (plain cell) |
| `prompting` | 1) 'prompting.mode' stamp in the eval JSON; 2) filename suffix __zero-shot / __few-shot (dual/override pass); 3) t2g- zero-shot naming convention (baseline cells only) | unknown — the config-implied mode is NEVER assumed, it can differ from the effective one under dual prompting |
| `grammar` | 'no-grammar' in cell name => off; 'grammar' in t2g-* cell name => on (that family names the toggle explicitly) | unknown (never guessed) |
| `reward_stack` | NOT deducible from path / JSON stamp / payload fields (it lives in the resolved config, which eval JSONs do not embed) | always unknown, declared in every pairing caveat |
| `rl_objective` | NOT deducible from path / JSON stamp / payload fields | always unknown, declared in every pairing caveat |

## Selected runs (9 typologies, latest per typology)

| Run | method | variant | prompting | grammar | Date | Prompts | metrics_version |
|---|---|---|---|---|---|---|---|
| `wos-46985/qwen25-05b/ablations/decoding/no-grammar/run_20261010_030132/eval_baseline.json` | baseline | wos-46985-qwen25-05b-ablations-decoding-no-grammar | zero-shot | unknown | 2026-10-10 | 2000 | 2 |
| `wos-46985/qwen25-05b/ablations/decoding/no-grammar/run_20261010_030132/eval_final.json` | unknown | wos-46985-qwen25-05b-ablations-decoding-no-grammar | zero-shot | unknown | 2026-10-10 | 2000 | 2 |
| `wos-46985/qwen25-05b/ablations/objectives/sft-allowed-mass/run_20261010_052117/eval_final.json` | unknown | wos-46985-qwen25-05b-ablations-objectives-sft-allowed-mass | zero-shot | unknown | 2026-10-10 | 2000 | 2 |
| `wos-46985/qwen25-05b/baseline/zero-shot-no-grammar/run_20261009_202056/eval_zero_shot.json` | unknown | wos-46985-qwen25-05b-baseline-zero-shot-no-grammar | zero-shot | unknown | 2026-10-09 | 2000 | 2 |
| `wos-46985/qwen25-05b/baseline/zero-shot/run_20261009_201559/eval_zero_shot.json` | unknown | wos-46985-qwen25-05b-baseline-zero-shot | zero-shot | unknown | 2026-10-09 | 2000 | 2 |
| `wos-46985/qwen25-05b/grpo/zero-shot/run_20261009_220558/eval_final.json` | unknown | wos-46985-qwen25-05b-grpo-zero-shot | zero-shot | unknown | 2026-10-09 | 2000 | 2 |
| `wos-46985/qwen25-05b/sft-grpo/zero-shot/run_20261010_000134/eval_final.json` | sft-grpo | zero-shot | zero-shot | unknown | 2026-10-10 | 2000 | 2 |
| `wos-46985/qwen25-05b/sft/zero-shot/run_20261009_203139/eval_baseline.json` | baseline | wos-46985-qwen25-05b-sft-zero-shot | zero-shot | unknown | 2026-10-09 | 2000 | 2 |
| `wos-46985/qwen25-05b/sft/zero-shot/run_20261009_203139/eval_final.json` | unknown | wos-46985-qwen25-05b-sft-zero-shot | zero-shot | unknown | 2026-10-09 | 2000 | 2 |

## Ablation matrix overview — ROUGE-L

One row per cell (its path under `results/`), one column per prompting mode that has at least one value. `—` = the cell was not evaluated in that mode. `*` = multiple runs collapsed, latest shown. Each cell's own `eval_baseline` (the base model in that cell's context) is not a row here: it repeats the `baseline/` cells and stays in the paired comparisons below.

| cell | zero-shot |
|---|---|
| wos-46985/qwen25-05b/baseline/zero-shot | 0.2394 |
| wos-46985/qwen25-05b/baseline/zero-shot-no-grammar | 0.1142 |
| wos-46985/qwen25-05b/sft/zero-shot | 0.8125 |
| wos-46985/qwen25-05b/sft-grpo/zero-shot | 0.8340 |
| wos-46985/qwen25-05b/grpo/zero-shot | 0.3194 |
| wos-46985/qwen25-05b/ablations/decoding/no-grammar | 0.2045 |
| wos-46985/qwen25-05b/ablations/objectives/sft-allowed-mass | 0.8103 |

## wos-46985 — all metrics

Same rows as the matrix above, restricted to this dataset, one column per metric. On CoNLL-2003 and WOS-46985 read Exact Match and Gloss F1 (micro = entity-level F1 on CoNLL): ROUGE-L and BLEU split `PER:Mary` into `per` + `mary` and give partial credit to a wrong label of the same type.

| cell | prompting | ROUGE-L | Valid ROUGE-L | Exact Match | Non-copy Tok Acc | Pass@1 | Validity | BLEU (corpus) | BLEU (sent) | chrF2 (corpus) | chrF2 (sent) | Gloss F1 (mic) | Gloss F1 (sent) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| wos-46985/qwen25-05b/baseline/zero-shot | zero-shot | 0.2394 | 0.2394 | 0.1039 | 0.2879 | 0.4435 | 1.0000 | 0.1241 | 0.1627 | 25.8721 | 25.6578 | 0.2876 | 0.2876 |
| wos-46985/qwen25-05b/baseline/zero-shot-no-grammar | zero-shot | 0.1142 | 0.0000 | 0.0000 | 0.0175 | 0.2035 | 0.0000 | 0.0003 | 0.0132 | 15.4631 | 15.6771 | 0.0146 | 0.0173 |
| wos-46985/qwen25-05b/sft/zero-shot | zero-shot | 0.8125 | 0.8125 | 0.7622 | 0.8268 | 0.8740 | 1.0000 | 0.7960 | 0.7787 | 81.0950 | 81.0292 | 0.8269 | 0.8269 |
| wos-46985/qwen25-05b/sft-grpo/zero-shot | zero-shot | 0.8340 | 0.8340 | 0.7869 | 0.8472 | 0.8935 | 1.0000 | 0.8146 | 0.8020 | 83.1798 | 83.1297 | 0.8475 | 0.8475 |
| wos-46985/qwen25-05b/grpo/zero-shot | zero-shot | 0.3194 | 0.3194 | 0.1688 | 0.3700 | 0.5435 | 1.0000 | 0.2014 | 0.2282 | 32.1067 | 32.2584 | 0.3702 | 0.3702 |
| wos-46985/qwen25-05b/ablations/decoding/no-grammar | zero-shot | 0.2045 | 0.0000 | 0.0000 | 0.2157 | 0.3030 | 0.0001 | 0.0001 | 0.0363 | 21.1028 | 21.4371 | 0.1467 | 0.1615 |
| wos-46985/qwen25-05b/ablations/objectives/sft-allowed-mass | zero-shot | 0.8103 | 0.8103 | 0.7572 | 0.8256 | 0.8775 | 1.0000 | 0.7916 | 0.7743 | 80.9192 | 80.7761 | 0.8255 | 0.8255 |

## Paired comparisons — one table per factor

### Factor: `method`

**method effect (cross-run)** — `wos-46985/qwen25-05b/ablations/decoding/no-grammar/run_20261010_030132/eval_baseline.json` (2026-10-10, zero-shot, unknown) → `wos-46985/qwen25-05b/sft-grpo/zero-shot/run_20261010_000134/eval_final.json` (2026-10-10, zero-shot, unknown)

| Metric | A | B | Δ (B−A) | Interpretation |
|---|---|---|---|---|
| ROUGE-L | 0.1142 | 0.8340 | +0.7198 | above noise threshold |
| Valid ROUGE-L | 0.0000 | 0.8340 | +0.8340 | above noise threshold |
| Exact Match | 0.0000 | 0.7869 | +0.7869 | above noise threshold |
| Non-copy Tok Acc | 0.0175 | 0.8472 | +0.8297 | above noise threshold |
| Pass@1 | 0.2035 | 0.8935 | +0.6900 | above noise threshold |
| Validity | 0.0000 | 1.0000 | +1.0000 | above noise threshold |
| BLEU (corpus) | 0.0003 | 0.8146 | +0.8143 | above noise threshold |
| BLEU (sent) | 0.0132 | 0.8020 | +0.7889 | above noise threshold |
| chrF2 (corpus) | 15.4631 | 83.1798 | +67.7168 | scale 0-100: 0.02 noise threshold not applicable |
| chrF2 (sent) | 15.6771 | 83.1297 | +67.4526 | scale 0-100: 0.02 noise threshold not applicable |
| Gloss F1 (mic) | 0.0146 | 0.8475 | +0.8328 | above noise threshold |
| Gloss F1 (sent) | 0.0173 | 0.8475 | +0.8301 | above noise threshold |
- ⚠️ factors not deducible on at least one side, equality NOT verified: reward_stack, rl_objective
- ⚠️ prompt_context_fingerprint DIFFERS: the eval contexts (prompting/grammar/num_samples/model) are not identical — this pair may differ in more than the method factor

### Factor: `prompting`

**No paired comparison available.** 2 group(s) with 2+ runs but no pair with a known, differing value of this factor

### Factor: `grammar`

**No paired comparison available.** 2 group(s) with 2+ runs but no pair with a known, differing value of this factor


## Missing paired comparisons (declared, not fabricated)

- **`prompting`**: 2 group(s) with 2+ runs but no pair with a known, differing value of this factor.
  Available typologies: `zero-shot`
- **`grammar`**: 2 group(s) with 2+ runs but no pair with a known, differing value of this factor.
  Available typologies: `grammar=unknown`

## Figures

- `campaign_pairwise_deltas.png` — one panel per factor: paired deltas as diverging bars with the ±0.02 noise band shaded. Diverging bars make the SIGN of each effect immediately visible, and the shaded band makes it impossible to mistake sub-noise deltas for results.
- `campaign_matrix.png` — the ablation matrix as a heatmap (rows: method/variant, columns: prompting, annotated with ROUGE-L); empty typologies stay visibly grey so gaps in the grid are impossible to miss.

