# Evaluation Protocol

Evaluation is prompt-controlled and uses the same metric protocol across baselines and checkpoints. No run results are asserted here.

## Dual prompt modes

The evaluator provides two modes:

- `--prompt-mode zero-shot`: task instruction without retrieved demonstrations.
- `--prompt-mode retrieval`: retrieval-backed few-shot prompt; artifacts are labelled `few-shot`.

`--prompt-mode auto` follows `retrieval.enabled` for ad-hoc use. The 12-entry default campaign passes explicit modes: each trained eval job runs `zero-shot` and then `retrieval`, producing `eval-zero-shot` and `eval-few-shot` directories. Prompt mode changes conditioning, not method identity.

Baseline configs are untrained base-method evaluations and must match their own prompt identity:

```bash
CONFIG=experiments/configs/qwen25-05b/baseline/zero-shot.yaml sbatch cluster/eval.sh
CONFIG=experiments/configs/qwen25-05b/baseline/few-shot.yaml sbatch cluster/eval.sh
```

For explicit checkpoint evaluation in either mode, invoke the module with the selected checkpoint and prompt mode:

```bash
python -m src.training.eval_t2g --config experiments/configs/qwen25-05b/sft-grpo/zero-shot.yaml --checkpoint <checkpoint> --prompt-mode zero-shot --plot
python -m src.training.eval_t2g --config experiments/configs/qwen25-05b/sft-grpo/zero-shot.yaml --checkpoint <checkpoint> --prompt-mode retrieval --plot
```

## Shared decoding protocol

The shared config defaults to 2,000 seeded test examples. Every prompt has two explicit result paths: `deployment` is exactly one greedy completion (`do_sample=false`), while `sampling` is N=5 independent draws (`do_sample=true`, temperature **0.7**). Headline/top-level metrics mirror deployment only. Sampling reports expected draw quality, diversity/headroom, and Pass@k; any gold-selected oracle remains separate and non-deployable.

Sampling seeds are derived from protocol seed, stable source+gold sample ID, decoding mode, and completion index. Generation is therefore invariant to batching, cache hits, evaluation order, and interruption. Artifacts retain `decoding_mode` and `completion_index`.

Gold references must never select a primary completion. Oracle best-of-N may be reported only as a clearly labelled, non-deployable diagnostic.

## Metrics

Report at least:

1. corpus BLEU-4 (headline literature-facing metric);
2. corpus chrF2;
3. mean ROUGE-L;
4. micro gloss-token F1;
5. normalized exact match (gloss extraction, Unicode casefold, whitespace collapse) and normalized token edit similarity;
6. deployment success and standard Pass@k estimator `1-C(n-c,k)/C(n,k)`, where success is ROUGE-L >= 0.3 **and** lexical validity;
7. vocabulary validity.

Length diagnostics include signed and absolute token-length error, mean generated/reference lengths, corpus length ratio, empty rate, and truncation-hit rate. `valid_rouge_l_mean` is `mean(rouge_i if valid_i else 0)`, not a product of marginal means.

Confidence intervals use prompt-clustered deterministic bootstrap. Sampling sentence metrics first aggregate within prompt; corpus BLEU/chrF are recomputed on each prompt-resampled corpus. Paired comparisons require identical ordered unique sample IDs and recompute corpus deltas on shared bootstrap indices. Cache and explicit baseline artifacts must match the complete protocol identity (metrics version, ordered-ID hash, dataset/model/tokenizer/vocabulary, retrieval, grammar, decoding, seed, mode, N, budget, and temperature); mismatches fail loudly.

BLEU/chrF validate aligned input lengths and store SacreBLEU signatures/settings. Bigram scoring is optional: availability/reason is recorded, and systems are compared only when both have the diagnostic.

Include confidence intervals and difficulty breakdowns where emitted. Pass@k is a project diagnostic adapted from code generation, not a standard T2G metric. Do not compare English-to-gloss numbers with published gloss-to-English scores.

Constrained decoding can improve validity without improving content. Interpret BLEU together with chrF, ROUGE-L, gloss F1, and validity; do not call validity a translation-quality result.

## Artifact paths

Baseline result outputs are:

```text
experiments/results/qwen25-05b/baseline/{zero-shot,few-shot}/run_<timestamp>/
```

Trained outputs append the evaluation prompt mode:

```text
experiments/results/qwen25-05b/<method>/<train-prompt>/
  eval-zero-shot/run_<timestamp>/
  eval-few-shot/run_<timestamp>/
```

Ablations insert `ablations/<variant>/` before `eval-*`. Keep raw generations, metric JSON, config identity, checkpoint source, and W&B offline metadata together when reporting a run.

Each evaluation leg also has an isolated canonical log/W&B directory, so the two offline W&B runs cannot collide:

```text
experiments/logs/qwen25-05b/<method>/<train-prompt>/run_<timestamp>/
  eval-zero-shot/
  eval-few-shot/
```

For ablations, `ablations/<variant>/` appears before `run_<timestamp>`. Plot labels and W&B names use the stable `<method>/<train-prompt>[/ablations/<variant>]/eval-<mode>` identity.

## Literature anchors

- Abdullah et al., Bangla T2G benchmark, arXiv:2504.02293.
- Walsh, Saunders, and Bowden, *Select and Reorder*, arXiv:2404.11532.
- Rao et al., RVLF, arXiv:2512.07273.
- Chen et al., HumanEval/Pass@k, arXiv:2107.03374 (provenance only).

See `SOURCES.md` for the accepted bibliography.
