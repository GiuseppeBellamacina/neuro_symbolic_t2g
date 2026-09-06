# Markov Diagnostics

Bigram, hard-path, and soft-path scores are **diagnostic only**. They are not active members of the runtime reward stack and are not training cells.

## Scope

The Markov probe consumes frozen generations and compares edit/bigram sequence evidence, a hard Viterbi-style path score, a soft log-partition/forward score, and rank association with within-group pairwise discrimination.

Run it offline without training:

```bash
INPUT=/path/to/existing/generations_baseline.json \
  sbatch cluster/probe.sh markov experiments/configs/qwen25-05b/probes/markov.yaml
```

`INPUT` is required and must point to an existing `generations_*.json` emitted by
the current evaluator. The launcher does not create or search for generations.
It runs with network access disabled and writes the default reports under
`experiments/analysis/qwen25-05b/{rollouts,markov}/report.json`. Override the
repository-local destination with `OUTPUT=experiments/analysis/.../report.json`.
From a login shell, the same launcher can allocate and relaunch itself with:

```bash
INPUT=/path/to/existing/generations_baseline.json \
  bash cluster/probe.sh markov experiments/configs/qwen25-05b/probes/markov.yaml
```

The O(L) bigram diagnostic may score the complete vocabulary artifact. Hard and
soft Viterbi helpers are separate, optional small-graph diagnostics: every call
enforces `max_states` (256 in the Markov probe config) before allocating dense
dynamic-programming state. Probe thresholds are acceptance diagnostics, not
claims of policy improvement.

## Attribution boundary

These project diagnostics are inspired by sequence modeling but do **not** implement the Differentiable Viterbi Layer in ViterbiPlanNet ([arXiv:2603.04265](https://arxiv.org/abs/2603.04265)). They have no learned emissions, differentiable backpointers, PKG coupling, or gradient path through Viterbi inference. Calling them “DVL,” “ViterbiPlanNet,” or a differentiable planner would be incorrect.

## Decision rule

Use the probe to decide whether Markov evidence adds ranking information beyond simple edit/bigram similarity on the same frozen samples. Do not add a Markov term to training merely because its scalar correlates with gold quality. Require preregistered thresholds, paired analysis, and a subsequent isolated training ablation before making a causal claim.

## Future DVL work

A genuine DVL experiment is future work. It would require a separately specified differentiable module, learned emissions/transitions, an explicit training objective, gradient-flow tests, and attribution to ViterbiPlanNet. That work must not reuse the current diagnostic name as evidence of an existing implementation.
