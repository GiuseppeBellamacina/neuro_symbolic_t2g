# Experiment Design

This is the authoritative campaign definition. It describes intended comparisons, not completed results.

## Terms

- **Method** is the learned system: `base`, `sft`, `grpo`, or `sft-grpo`.
- **Prompt mode** is input conditioning: `zero-shot` or retrieval-backed `few-shot`. Zero-shot is prompting, **not a training method**.
- **Baseline** means the untrained `base` method evaluated with the named prompt mode. It is not synonymous with zero-shot.
- **Variant** is a controlled ablation. Current variants are `pda` and `hot`.

## Primary 2x2

The main factorial comparison crosses GRPO initialization with train prompt mode:

| Method | Zero-shot train prompt | Few-shot train prompt |
|---|---|---|
| GRPO from base | `grpo/zero-shot.yaml` | `grpo/few-shot.yaml` |
| GRPO after SFT | `sft-grpo/zero-shot.yaml` | `sft-grpo/few-shot.yaml` |

Evaluation supports zero-shot and few-shot conditioning; the evaluator's internal name for retrieval-backed few-shot mode is `retrieval`. Every trained eval entry runs zero-shot first and retrieval second. Evaluation conditioning does not change method identity.

## Baselines, SFT, and default campaign

The additional reference cells are:

1. `baseline/zero-shot.yaml`: untrained base method, zero-shot evaluation.
2. `baseline/few-shot.yaml`: untrained base method, few-shot evaluation.
3. `sft/zero-shot.yaml`: SFT trained with zero-shot prompts; its checkpoint is evaluated in both prompt modes.

The default campaign is **2 eval-only baselines + 5 train/eval cells = 12 queue entries**: 2 baseline evaluations, plus SFT and the four primary 2x2 cells, each with one training and one dual-prompt evaluation entry.

`bash cluster/run_all.sh` enqueues the default campaign. The two staged ablations are not included.

## Staged ablations

Run only after the primary campaign passes preflight and produces interpretable paired evaluations:

1. `ablations/sft-grpo-zero-pda.yaml` — decoding formalism only: PDA versus the primary Trie path. Manual; see `PDA.md`.
2. `ablations/sft-grpo-zero-hot.yaml` — rollout temperature only: 1.3 versus the inherited 0.7. Manual stability probe.

Markov probes are analyses over frozen generations. They never train a policy and are not campaign cells; see `MARKOV_DIAGNOSTICS.md`.

## Exact config tree

All paths are under `experiments/configs/qwen25-05b/`:

```text
base.yaml
baseline/
  zero-shot.yaml
  few-shot.yaml
sft/
  train.yaml
grpo/
  zero-shot.yaml
  few-shot.yaml
sft-grpo/
  zero-shot.yaml
  few-shot.yaml
ablations/
  sft-grpo-zero-pda.yaml
  sft-grpo-zero-hot.yaml
probes/
  rollouts.yaml
  markov.yaml
```

`base.yaml` is the shared recipe, not an additional campaign cell. Runnable files declare `experiment.model_tag`, `method`, `train_prompt_mode`, `variant`, and `kind`; those fields determine artifact paths. `kind` is lifecycle: `train` produces a checkpoint, while `baseline`, `ablation`, and `probe` identify their respective job classes. Prompt conditioning remains in `train_prompt_mode`.

## Canonical artifact hierarchy

```text
experiments/{checkpoints,logs}/qwen25-05b/
  sft/zero-shot/run_<timestamp>/
  grpo/{zero-shot,few-shot}/run_<timestamp>/
  sft-grpo/{zero-shot,few-shot}/run_<timestamp>/
  sft-grpo/zero-shot/ablations/{pda,hot}/run_<timestamp>/

experiments/results/qwen25-05b/
  baseline/{zero-shot,few-shot}/run_<timestamp>/
  sft/zero-shot/eval-{zero-shot,few-shot}/run_<timestamp>/
  grpo/{zero-shot,few-shot}/eval-{zero-shot,few-shot}/run_<timestamp>/
  sft-grpo/{zero-shot,few-shot}/eval-{zero-shot,few-shot}/run_<timestamp>/
  sft-grpo/zero-shot/ablations/{pda,hot}/eval-{zero-shot,few-shot}/run_<timestamp>/
```

Do not report outcomes until artifacts from these paths have been checked under the protocol in `EVALUATION.md`.
