# Training Guide

The authoritative runtime configs are under `experiments/configs/qwen25-05b/`. Zero-shot and few-shot identify prompt conditioning, not training methods.

## Training cells

| Config | Initialization | Training prompts | Evaluation |
|---|---|---|---|
| `sft/zero-shot.yaml` | base model | zero-shot | checkpoint supports zero-shot and few-shot evaluation |
| `grpo/zero-shot.yaml` | base model | zero-shot | either prompt mode |
| `grpo/few-shot.yaml` | base model | retrieval-backed few-shot | either prompt mode |
| `sft-grpo/zero-shot.yaml` | SFT checkpoint | zero-shot | either prompt mode |
| `sft-grpo/few-shot.yaml` | SFT checkpoint | retrieval-backed few-shot | either prompt mode |

Retrieval is the internal implementation of few-shot prompting and applies to GRPO-only and SFT-GRPO few-shot cells. SFT itself is trained zero-shot.

## Objective and decoding

Production GRPO uses:

- `reward.name: edit-validity` only;
- `grpo.loss_type: dr_grpo`;
- Trie vocabulary-constrained decoding as the primary decoding path.

Markov, hard-Viterbi, and soft-path scores are diagnostic only and do not update a policy. PDA changes decoding and is a manual ablation. The `hot` ablation changes rollout temperature from the inherited value to 1.3.

## Launch

First run the online bootstrap on the login node, then run preflight on offline compute before training:

```bash
bash cluster/setup.sh
CONFIG=experiments/configs/qwen25-05b/sft-grpo/zero-shot.yaml sbatch cluster/preflight.sh
```

Setup runs Python/pip only inside Apptainer without SLURM or GPU, preserves the container torch/CUDA stack, and prepares the Qwen, ASLG-PC12, and vocabulary caches. The bigram is optional during setup (`BUILD_BIGRAM=1 bash cluster/setup.sh`) and required only for workflows such as the Markov probe that consume it.

Launch one cell:

```bash
CONFIG=experiments/configs/qwen25-05b/sft/zero-shot.yaml sbatch cluster/train.sh
CONFIG=experiments/configs/qwen25-05b/grpo/zero-shot.yaml sbatch cluster/train.sh
CONFIG=experiments/configs/qwen25-05b/grpo/few-shot.yaml sbatch cluster/train.sh
CONFIG=experiments/configs/qwen25-05b/sft-grpo/zero-shot.yaml sbatch cluster/train.sh
CONFIG=experiments/configs/qwen25-05b/sft-grpo/few-shot.yaml sbatch cluster/train.sh
```

Resume with the same config identity:

```bash
CONFIG=experiments/configs/qwen25-05b/sft-grpo/zero-shot.yaml EXTRA_ARGS="--resume" sbatch cluster/train.sh
```

The default campaign is **2 eval-only baselines + 5 train/eval configs = 12 queue entries**:

```bash
bash cluster/run_all.sh
```

Run PDA and hot-temperature ablations manually; do not count them in the default campaign. Probes are analysis jobs, not training jobs.

## Outputs

```text
experiments/{checkpoints,logs}/qwen25-05b/<method>/<train-prompt>/run_<timestamp>/
experiments/{checkpoints,logs}/qwen25-05b/<method>/<train-prompt>/ablations/<pda|hot>/run_<timestamp>/
```

For SFT, `<train-prompt>` is `zero-shot`. Resume lookup, W&B run identity, and evaluation must retain the same method and training-prompt identity.

## Offline requirements

Compute jobs must use local artifacts and set:

```text
HF_HUB_OFFLINE=1
TRANSFORMERS_OFFLINE=1
HF_DATASETS_OFFLINE=1
WANDB_MODE=offline
WANDB_DISABLE_WEAVE=true
WANDB_SILENT=true
```

Do not permit an online fallback. Prepare the Qwen snapshot, ASLG-PC12 cache, gloss vocabulary, transition data, and retrieval index where required before submission. Never place HF or W&B credentials in configs or logs.

See [docs/TRAINING.md](docs/TRAINING.md) for the operational checklist and [docs/EXPERIMENT_DESIGN.md](docs/EXPERIMENT_DESIGN.md) for campaign identity.
