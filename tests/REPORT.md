# Test and Consistency Report

This file records the current validation scope without embedding campaign narratives or stale test counts.

## Current invariants

The test suite covers:

- config resolution and experiment identity under `experiments/configs/qwen25-05b/`;
- the two baseline prompt modes and five training cells;
- zero-shot SFT training with dual-mode checkpoint evaluation;
- retrieval-backed few-shot prompting for GRPO and SFT-GRPO;
- `edit-validity` as the production GRPO reward and Dr-GRPO configuration;
- canonical checkpoint, log, and result paths;
- offline Hugging Face and W&B hardening;
- Trie decoding, the manual PDA gate, and the hot-temperature ablation;
- rollout and Markov/Viterbi diagnostics as non-training probes.

## Authoritative config inventory

```text
experiments/configs/qwen25-05b/
  base.yaml
  baseline/{zero-shot,few-shot}.yaml
  sft/zero-shot.yaml
  grpo/{zero-shot,few-shot}.yaml
  sft-grpo/{zero-shot,few-shot}.yaml
  ablations/{sft-grpo-zero-pda,sft-grpo-zero-hot}.yaml
  probes/{rollouts,markov}.yaml
```

The default queue arithmetic is **2 eval-only baselines + 5 training jobs + 5 evaluations = 12 entries**. PDA and hot are manual; probes do not train.

## Run validation

```bash
uv run python -m pytest tests/ -v
python tests/validate_configs.py
```

Offline or GPU-dependent checks must be reported as skipped unless their required artifacts and hardware are available. Do not infer a passing count from this document; use the command output from the revision being validated.

## Documentation consistency

Documentation should contain no obsolete config root, extra method hierarchy, retired reward-stack config names, or wording that treats zero-shot as a method. The canonical path shapes are:

```text
experiments/{checkpoints,logs}/qwen25-05b/<method>/<train-prompt>/run_<timestamp>/
experiments/{checkpoints,logs}/qwen25-05b/<method>/<train-prompt>/ablations/<pda|hot>/run_<timestamp>/
experiments/results/qwen25-05b/baseline/<zero-shot|few-shot>/run_<timestamp>/
experiments/results/qwen25-05b/<method>/<train-prompt>/eval-<zero-shot|few-shot>/run_<timestamp>/
```
