# Neuro-Symbolic Text-to-Gloss

This repository studies English-to-ASL-gloss generation with Qwen2.5-0.5B-Instruct, supervised fine-tuning (SFT), Group Relative Policy Optimization (GRPO), retrieval-backed prompting, and constrained decoding.

## Experiment model

The learned methods are `base`, `sft`, `grpo`, and `sft-grpo`. **Zero-shot** and **few-shot** are prompt modes, not methods. Few-shot prompting is implemented internally through retrieval.

- Baselines evaluate the untrained base model in zero-shot and few-shot modes.
- SFT trains with zero-shot prompts. Its checkpoint can be evaluated in both prompt modes.
- GRPO and SFT-GRPO each train in zero-shot and retrieval-backed few-shot modes.
- GRPO uses the production `edit-validity` reward and Dr-GRPO loss.
- Trie-constrained vocabulary decoding is the primary path. PDA decoding is a manual ablation.
- The `hot` ablation changes rollout temperature only.
- Markov and Viterbi scoring are frozen-generation diagnostics, never production rewards.

## Authoritative configurations

All current configs live under `experiments/configs/qwen25-05b/`:

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

`base.yaml` is shared configuration, not a runnable campaign cell. See [Experiment Design](docs/EXPERIMENT_DESIGN.md) for config identities and comparisons.

## Default campaign

The default campaign contains **12 queue entries**:

- 2 eval-only baselines;
- 5 training entries: SFT, two GRPO cells, and two SFT-GRPO cells;
- 5 matching evaluation entries.

PDA and hot-temperature ablations are launched manually. Rollout and Markov probes do not train a policy.

```bash
bash cluster/run_all.sh
```

Examples for individual jobs:

```bash
CONFIG=experiments/configs/qwen25-05b/sft/zero-shot.yaml sbatch cluster/train.sh
CONFIG=experiments/configs/qwen25-05b/grpo/few-shot.yaml sbatch cluster/train.sh
CONFIG=experiments/configs/qwen25-05b/sft-grpo/zero-shot.yaml sbatch cluster/train.sh
CONFIG=experiments/configs/qwen25-05b/baseline/few-shot.yaml sbatch cluster/eval.sh
```

## Artifact hierarchy

```text
experiments/{checkpoints,logs}/qwen25-05b/<method>/<train-prompt>/run_<timestamp>/
experiments/{checkpoints,logs}/qwen25-05b/<method>/<train-prompt>/ablations/<pda|hot>/run_<timestamp>/

experiments/results/qwen25-05b/baseline/<zero-shot|few-shot>/run_<timestamp>/
experiments/results/qwen25-05b/<method>/<train-prompt>/eval-<zero-shot|few-shot>/run_<timestamp>/
experiments/results/qwen25-05b/<method>/<train-prompt>/ablations/<pda|hot>/eval-<zero-shot|few-shot>/run_<timestamp>/
```

SFT uses `sft/zero-shot` as its training identity. There is no extra hierarchy level between the model tag and method.

## Offline operation

Run `bash cluster/setup.sh` on the login node. Setup always follows login → `srun` → compute → `apptainer run --nv` with `/shared/sifs/latest.sif`; it never probes for or runs Apptainer on login. Inside the container it installs into shared `~/.local` and prepares model, dataset, and vocabulary caches without running host Python. Because the QoS allows one submitted/active job, run `pip-reset`/setup only when no other allocation is active; the setup allocation must have network access. Cluster train/eval/preflight/probe jobs remain offline; run `sbatch cluster/preflight.sh` after setup. Hugging Face offline flags and `WANDB_MODE=offline` must be set before compute Python starts; credentials must not be committed.

See [Training](TRAINING.md), [Cluster Operation](CLUSTER.md), and [Evaluation](docs/EVALUATION.md).

## Local checks

```bash
pip install -e ".[dev]"
uv run python -m pytest tests/ -v
```

## References

- Othman and Jemni, ASLG-PC12: <https://arxiv.org/abs/1112.0168>
- Shao et al., GRPO: <https://arxiv.org/abs/2402.03300>
- Liu et al., Dr-GRPO: <https://arxiv.org/abs/2503.20783>
- TRL GRPO trainer: <https://huggingface.co/docs/trl/main/en/grpo_trainer>
- Full bibliography: [docs/SOURCES.md](docs/SOURCES.md)

## License

[MIT](LICENSE)
