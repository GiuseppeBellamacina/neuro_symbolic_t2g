# Cluster Operation

This guide covers current SLURM operation for the Qwen2.5-0.5B campaign. Config and artifact identity are defined in [docs/EXPERIMENT_DESIGN.md](docs/EXPERIMENT_DESIGN.md).

## Setup

Upload the repository and run setup directly on the internet-enabled login node:

```bash
ssh <user>@gcluster.dmi.unict.it
cd ~/neuro_symbolic_t2g
bash cluster/setup.sh
```

Setup always uses the canonical gcluster path: login node → `srun` → compute node → `apptainer run --nv` with `/shared/sifs/latest.sif` → setup inside the container. The historical allocation defaults are account/partition `thesis-course`, QoS `gpu-xlarge`, one GPU, shard 22000, 48G memory, and 8 CPUs; sensible resource environment overrides are documented in `cluster/setup.sh`. The login node does not need or probe for an Apptainer binary. Python and pip never run on the bare host; they run only after the container relaunch, while shared `$HOME`, `~/.local`, `HF_HOME`, and `HF_HUB_CACHE` remain visible. Setup installs core + retrieval into the user site under constraints generated from every preinstalled critical ML/CUDA package. The optional MiniLM backend is pinned to `sentence-transformers==5.2.3`, matching the distribution preinstalled in `/shared/sifs/latest.sif`; requiring a newer release conflicts with the generated setup constraint and makes pip report `ResolutionImpossible`. The post-install gate rejects any change to that captured stack and enforces sentence-transformers 5.2.3 alongside the tested Transformers 5.3.0, TRL 0.24.0, PEFT 0.19.1, Unsloth 2026.7.1, and torchao 0.16–0.17 combination plus the original torch/CUDA identity. It then caches Qwen with `snapshot_download`, caches ASLG-PC12, and creates the deterministic 90/10 train-derived vocabulary and sidecar. Use `BUILD_BIGRAM=1 bash cluster/setup.sh` to build the optional Markov bigram.

`pip-reset` removes user packages first and then runs this complete online setup. `pip-setup` runs setup without first clearing `~/.local`. The user's QoS permits only one submitted/active job, so run `pip-reset`/setup only when no training, evaluation, preflight, probe, or other allocation is active. The setup allocation requires network access; if cluster network policy changes, setup will fail rather than silently run Python on the host.

If `pip-reset` already cleared `~/.local` and then failed with the sentence-transformers resolver conflict, upload the corrected repository and run `pip-setup`. Do **not** run another `pip-reset`: the user site is already empty, and setup can install the corrected exact package set directly.

## Offline contract

Training, evaluation, preflight, and probes run on compute with Hugging Face offline and W&B offline. Setup is the only online acquisition step:

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
