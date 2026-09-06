# Single-Reward Ablations

These are manual scientific qualification experiments, not additions to the primary campaign. The primary campaign remains seven cells and twelve queue entries.

## Candidates

All candidates use the same case-folded gloss tokens, vocabulary/empty/reference validity gate, finite bounded invalid score, and `2 * similarity - 1` transform. Completion text is first extracted from the model wrapper; references are already plain gloss text. Similarities are clamped to `[0, 1]` before the transform, so every returned reward is strictly in `[-1, 1]`. Exactly one reward is selected by `reward.name`:

- `edit-validity`: normalized token Levenshtein similarity (control).
- `token-f1-validity`: clipped multiset token-overlap F1.
- `chrfpp-validity`: SacreBLEU chrF++ (`char_order=6`, `word_order=2`, `beta=2`, `whitespace=false`).
- `rouge-l-validity`: token LCS F1 without stemming.
- `sbleu2-exp-validity`: sentence BLEU-2 with effective order, exponential smoothing, and no tokenizer.

The five configs under `ablations/rewards/` extend `grpo/few-shot.yaml` and alter only ablation identity, reward name, and W&B run name. Artifacts are stored under `grpo/few-shot/ablations/reward-*`. They are selectable manually through the remote driver/TUI or `cluster/run_all.sh grpo-few-reward-...`; they are never part of `DEFAULT_CAMPAIGN`.

## Mandatory qualification

Do not launch a reward training config before scoring all five candidates on the same frozen generation artifact:

```bash
python -m src.analysis rewards \
  --config experiments/configs/qwen25-05b/probes/rewards.yaml \
  --input experiments/results/.../generations_name.json
```

The authorization artifact must contain exactly eight `decoding_mode=sampling` rows per prompt, matching GRPO few-shot (`G=8`, temperature 0.7, retrieval prompting, Trie constraint, max completion length 128). A normal eval artifact has one deployment row plus five sampling rows per prompt: the probe filters deployment correctly and may analyze those five, but marks the report ineligible for training. Generate a bounded frozen artifact with the evaluator's `--num-samples 8 --prompt-mode retrieval --max-new-tokens 128` options before qualification. Existing reports are never overwritten unless `--force` (or `FORCE=1` through `cluster/probe.sh`) is explicit.

The report includes per-reward mean, standard deviation, quantiles, zero-variance/all-min/all-max groups, unique scores, group best-minus-mean, pairwise Spearman correlations, gold-length buckets, and deterministic reference perturbations. Project gates are reported separately for every reward:

- zero-variance group fraction <= 0.50;
- all-min group fraction <= 0.50;
- mean unique scores per group >= 1.50;
- mean best-minus-mean >= 0.05.

Thresholds are screening criteria, not evidence of superiority. Candidate pass status includes variance, all-min, unique-score, best-minus-mean, rankable-group, length-bucket, and perturbation diagnostics. Perturbation expectations are metric-specific: token F1 is deliberately order-insensitive, so reversal is recorded as expected invariance and is not a degradation gate. Palindromic reversals and other no-op or structurally unevaluable perturbations are counted explicitly rather than treated as failures. Deletion, duplication, substitution, and the order-sensitive checks that are evaluable still require non-negative mean degradation and positive-drop evidence. A candidate must pass its own gates and be inspected for character-near, short-sequence, order, and length biases before manual launch. The report stamps input/vocabulary hashes, protocol and reward settings, implementation-source digest, SacreBLEU 2.6.0 signatures, decoding identity, and artifact metadata.

Training has an in-process hard gate before data/model loading:

```bash
EXTRA_ARGS="--reward-qualification-report experiments/analysis/qwen25-05b/rewards/report.json" \
  bash cluster/run_all.sh grpo-few-reward-token-f1
```

Missing, failed, stale, hash-mismatched, protocol-mismatched, or training-identity-mismatched reports abort even when Python is invoked directly. Preflight accepts the same path through `REWARD_QUALIFICATION_REPORT`.

The remote TUI recognizes every registry ID beginning with `grpo-few-reward-`. For a reward **train** action it requires a repository-relative report path below `experiments/analysis/`; the field is hidden for normal configs and eval-only actions. Train+eval sends the qualification option only on the train entry. Reward configs remain manual-only and are not added to the primary campaign.

Custom queue replacement accepts an explicit fourth field only in this unambiguous form:

```text
train:grpo-few-reward-token-f1:my-tag:--reward-qualification-report=experiments/analysis/qwen25-05b/rewards/report.json
```

The TUI converts it to the exact backend mode `--reward-qualification-report <path>`. A reward train row without that fourth field is rejected.
