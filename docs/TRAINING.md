# Training and Offline Cluster Operation

This is the current runtime guide. Config identity and campaign composition are defined in `EXPERIMENT_DESIGN.md`.

## Offline contract

Compute jobs must not download packages, datasets, models, or telemetry. Prepare and synchronize artifacts in an environment with network access first. Runtime exports must include:

```text
HF_HUB_OFFLINE=1
TRANSFORMERS_OFFLINE=1
HF_DATASETS_OFFLINE=1
WANDB_MODE=offline
WANDB_DISABLE_WEAVE=true
WANDB_SILENT=true
```

Hugging Face must resolve the configured model and ASLG-PC12 dataset from local caches. W&B writes offline runs only; synchronize them later from a networked machine if required. Never log or commit HF/W&B credentials.

## Setup and preflight

From the cluster repository root:

```bash
bash cluster/setup.sh
CONFIG=experiments/configs/qwen25-05b/sft-grpo/zero-shot.yaml sbatch cluster/preflight.sh
```

`setup.sh` is online and runs on the login node. It launches `/shared/sifs/latest.sif` directly, never requests SLURM/GPU resources, and never executes host Python. It installs core + retrieval into shared `~/.local`, constrains all preinstalled critical ML/CUDA packages, and hard-fails if pip changes any captured version or the torch/CUDA identity. It also enforces the tested Transformers/TRL/PEFT/Unsloth/torchao versions before caching Qwen and ASLG-PC12 and writing the deterministic train vocabulary plus sidecar. Set `BUILD_BIGRAM=1` to opt into the potentially large Markov matrix.

Preflight is offline compute-only. It verifies the offline environment, dataset/vocabulary/bigram artifacts when needed, model snapshot, imports, offline dataset loading, and W&B mode. It performs no training. A failure is a preparation error; do not let a compute job fall back to network access.

For the PDA ablation, the full-vocabulary gate is mandatory:

```bash
PDA=1 CONFIG=experiments/configs/qwen25-05b/ablations/sft-grpo-zero-pda.yaml sbatch cluster/preflight.sh
```

## Commands

Default campaign (12 entries):

```bash
bash cluster/run_all.sh
```

Single current configs:

```bash
CONFIG=experiments/configs/qwen25-05b/sft/zero-shot.yaml sbatch cluster/train.sh
CONFIG=experiments/configs/qwen25-05b/grpo/zero-shot.yaml sbatch cluster/train.sh
CONFIG=experiments/configs/qwen25-05b/grpo/few-shot.yaml sbatch cluster/train.sh
CONFIG=experiments/configs/qwen25-05b/sft-grpo/zero-shot.yaml sbatch cluster/train.sh
CONFIG=experiments/configs/qwen25-05b/sft-grpo/few-shot.yaml sbatch cluster/train.sh
```

Manual ablations:

```bash
bash cluster/run_all.sh sft-grpo-zero-pda
bash cluster/run_all.sh sft-grpo-zero-hot
# Only after the reward qualification report passes:
bash cluster/run_all.sh grpo-few-reward-token-f1
```

Frozen-generation probes are submitted separately and never train:

```bash
INPUT=experiments/results/.../generations_name.json \
  sbatch cluster/probe.sh rollouts experiments/configs/qwen25-05b/probes/rollouts.yaml
INPUT=experiments/results/.../generations_name.json \
  sbatch cluster/probe.sh markov experiments/configs/qwen25-05b/probes/markov.yaml
INPUT=experiments/results/.../generations_name.json \
  sbatch cluster/probe.sh rewards experiments/configs/qwen25-05b/probes/rewards.yaml
```

Resume a training run with the same config identity:

```bash
CONFIG=experiments/configs/qwen25-05b/sft-grpo/zero-shot.yaml EXTRA_ARGS="--resume" sbatch cluster/train.sh
```

The chain is sequential because the cluster permits one submitted job per user. Use `chain-show` or the monitor aliases for status.

## What each train config means

- `sft/zero-shot.yaml`: SFT only; `kind: train` describes the checkpoint-producing lifecycle, while `train_prompt_mode: zero-shot` describes conditioning.
- `grpo/{zero-shot,few-shot}.yaml`: GRPO initialized from the untrained base method, with the named train prompt mode.
- `sft-grpo/{zero-shot,few-shot}.yaml`: SFT initialization followed by GRPO, with the named GRPO train prompt mode. Adapter reuse is enabled.

Do not copy config files into alternate runtime trees. Extend the shared `base.yaml` and preserve semantic identity fields so checkpoints, logs, results, figures, W&B names, and resume lookup agree.

## Outputs

Training writes timestamped runs under `experiments/checkpoints/` and `experiments/logs/`; evaluation writes corresponding prompt-specific runs under `experiments/results/`. The exact hierarchy is in `EXPERIMENT_DESIGN.md`. Paths are derived from experiment identity at runtime.

## Pre-run checklist

1. Config resolves and has the intended method/train-prompt identity.
2. Model, dataset, vocabulary, bigram matrix, and retriever cache when needed are local.
3. The relevant preflight passes; PDA also passes its full-vocabulary gate. Reward ablations additionally require the frozen-artifact qualification in `REWARD_ABLATIONS.md`.
4. W&B and HF are offline before Python starts.
5. No conflicting active queue exists.
6. Evaluation prompt modes and sample budget are recorded before launch.
