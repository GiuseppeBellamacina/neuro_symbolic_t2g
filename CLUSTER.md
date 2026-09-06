# Cluster Operation

This guide covers current SLURM operation for the Qwen2.5-0.5B campaign. Config and artifact identity are defined in [docs/EXPERIMENT_DESIGN.md](docs/EXPERIMENT_DESIGN.md).

## Setup

Upload the repository, request an interactive GPU session, and prepare all runtime artifacts:

```bash
ssh <user>@gcluster.dmi.unict.it
srun --account <queue> --partition <queue> --qos gpu-xlarge --gres=gpu:1 --pty bash
cd ~/neuro_symbolic_t2g
bash cluster/setup.sh
```

Setup must populate local Hugging Face model and dataset caches, the gloss vocabulary, transition data, and retrieval cache used by few-shot configs. Compute jobs must not download artifacts.

## Offline contract

Training, evaluation, preflight, and probes run with Hugging Face offline and W&B offline:

```text
HF_HUB_OFFLINE=1
TRANSFORMERS_OFFLINE=1
HF_DATASETS_OFFLINE=1
WANDB_MODE=offline
WANDB_DISABLE_WEAVE=true
WANDB_SILENT=true
```

W&B stores local run data. Synchronize it only later from a networked environment. Keep HF and W&B credentials out of the repository and job logs.

## Preflight

```bash
CONFIG=experiments/configs/qwen25-05b/sft-grpo/zero-shot.yaml sbatch cluster/preflight.sh
```

Preflight checks offline mode, config resolution, imports, model and dataset caches, vocabulary, and transition artifacts. Few-shot jobs also require the retrieval artifacts.

PDA has an additional full-vocabulary gate:

```bash
PDA=1 CONFIG=experiments/configs/qwen25-05b/ablations/sft-grpo-zero-pda.yaml sbatch cluster/preflight.sh
```

## Submit jobs

The default campaign has **12 queue entries**: two eval-only baselines plus training and matching evaluation for SFT, two GRPO cells, and two SFT-GRPO cells. Each trained evaluation entry performs two sequential legs (zero-shot, then retrieval/few-shot) in the same job.

```bash
source cluster/aliases.sh
bash cluster/run_all.sh
chain-show
t2g-monitor
```

The cluster admits one submitted job per user, so the chain advances sequentially. Resume interrupted training with the original config:

```bash
CONFIG=experiments/configs/qwen25-05b/sft-grpo/zero-shot.yaml EXTRA_ARGS="--resume" sbatch cluster/train.sh
```

Examples of direct submissions:

```bash
CONFIG=experiments/configs/qwen25-05b/grpo/few-shot.yaml sbatch cluster/train.sh
CONFIG=experiments/configs/qwen25-05b/baseline/zero-shot.yaml sbatch cluster/eval.sh
```

## Manual ablations and probes

PDA decoding and hot rollout temperature are manual ablations and are not part of the 12-entry campaign:

```bash
bash cluster/run_all.sh sft-grpo-zero-pda
bash cluster/run_all.sh sft-grpo-zero-hot
```

Rollout and Markov probes operate on frozen generations and never train a model:

```bash
INPUT=experiments/results/.../generations_name.json \
  sbatch cluster/probe.sh rollouts experiments/configs/qwen25-05b/probes/rollouts.yaml
INPUT=experiments/results/.../generations_name.json \
  sbatch cluster/probe.sh markov experiments/configs/qwen25-05b/probes/markov.yaml
```

## Artifact locations

```text
experiments/{checkpoints,logs}/qwen25-05b/<method>/<train-prompt>/run_<timestamp>/
experiments/{checkpoints,logs}/qwen25-05b/<method>/<train-prompt>/ablations/<pda|hot>/run_<timestamp>/

experiments/results/qwen25-05b/baseline/<zero-shot|few-shot>/run_<timestamp>/
experiments/results/qwen25-05b/<method>/<train-prompt>/eval-<zero-shot|few-shot>/run_<timestamp>/
```

SFT artifacts use `sft/zero-shot`. A missing checkpoint for a trained method is an error; it must not silently become a base-model evaluation.

## Monitoring

```bash
squeue -u "$USER"
chain-show
t2g-monitor --all
tail -f logs/chain.log
tail -f logs/slurm-train-<JOB_ID>.log
```

The chain can retry resumable training failures. If it stops, inspect the active job, chain state, and `logs/chain.log` before restarting it.

## Download

From Windows PowerShell:

```powershell
.\sync_cluster.ps1 -Action download-logs
.\sync_cluster.ps1 -Action download-checkpoints
.\sync_cluster.ps1 -Action download-results
```

Preserve run directories and their resolved config identity when moving artifacts.

## Troubleshooting

- **Offline cache failure:** rerun setup in a network-enabled preparation session; do not disable offline mode in compute jobs.
- **Missing retrieval index:** prepare the index before launching a few-shot GRPO or SFT-GRPO config.
- **Missing checkpoint:** train the declared method first or pass the intended checkpoint explicitly.
- **CUDA OOM:** reduce the configured generation or batch budget while retaining the config identity and recording the override.
- **Pending job:** verify the one-job account limit with `squeue -u "$USER"`.
