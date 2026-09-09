# Neuro-Symbolic T2G — Constrained Decoding + GRPO for ASL Gloss Generation

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![TRL](https://img.shields.io/badge/TRL-GRPO-red.svg)](https://huggingface.co/docs/trl/)
[![Tests](https://img.shields.io/badge/Tests-96%2F96%20pytest-green.svg)](tests/)
[![Docs](https://img.shields.io/badge/Docs-REWARDS%20%7C%20METRICS-purple.svg)](docs/)
[![Ablation](https://img.shields.io/badge/Ablation-8%2B%20variants-orange.svg)](experiments/configs/qwen25-05b/)

## Overview

**neuro_symbolic_t2g** applies **Group Relative Policy Optimization (GRPO)** to fine-tune
a small LLM (Qwen2.5-0.5B-Instruct) for **Text-to-Gloss (T2G)** translation — converting
English sentences into **ASL (American Sign Language) gloss sequences**.

The key innovation is the **neuro-symbolic architecture**: a **constrained decoder** forces
every generated token to belong to the ASL gloss vocabulary (~15K tokens), while **GRPO**
optimizes the model through reinforcement learning with **7 rule-based reward functions**
active in the optimal config (plus 3 ablation-only modules) and **10 in total**.

> **No neural reward model needed** — the reward is purely deterministic, computed from
> ROUGE-L similarity, bigram transition probabilities (softmax-normalized),
> RECIPE-inspired verifier scaling, edit-distance ordering, format checks, and repetition penalties.

```text
┌──────────────┐    ┌──────────────────┐    ┌──────────────────────┐
│ English text │ →  │ Qwen2.5-0.5B     │ →  │ Constrained Decoder  │
│ "The man     │    │ + LoRA (QLoRA)   │    │ (vocabulary mask)    │
│  walks home" │    │ + GRPO training  │    │ only ASL gloss tokens │
└──────────────┘    └──────────────────┘    └──────────────────────┘
                                                      ↓
                                            ┌──────────────────────┐
                                            │ IX MAN WALK HOUSE    │
                                            │ (ASL gloss sequence) │
                                            └──────────────────────┘
```

### Key Features

- **Constrained Decoding**: `LogitsProcessor` masks all non-gloss tokens at each generation
  step — the model can only produce valid ASL glosses. A **dual-root token Trie** is the
  only constrained-decoding path; its coverage is effectively complete (4 gloss types out
  of 15472 blocked). Optional diagnostics (`MaskedMassTracker`, off by default) monitor
  masked probability mass, full-distribution entropy, and allowed-token entropy.
- **7 Active Rewards**: Translation quality (ROUGE-L), BLEU-4, gold-structure,
  gloss-order (edit-distance), verifier-scaled (RECIPE-inspired), format, and
  repetition penalty — plus the optional `edit_validity` reward (edit similarity
  with a graded in-vocabulary term). No neural reward model overhead.
- **Best-of-N Selection**: Evaluation supports `best_of_n` mode — generates N samples
  per prompt and selects the best (oracle). Baseline-vs-checkpoint comparison
  (`evaluation.compare`) and every other eval knob live in the `evaluation:`
  section of the config — the eval CLI is only `--config` + `--checkpoint`.
- **W&B Integration**: Offline mode with `console_multipart=True`, crash-safe try/finally,
  tagged runs, comparison plots, and JSON artifact logging.
- **Robust Gold Gloss Lookup**: Uses deterministic SHA256 hashing of user instructions
  to reliably match gold glosses regardless of prompt formatting — eliminates silent
  ROUGE-L=0 failures during training.
- **Centralized Prompting**: Single `build_t2g_prompt()` in `src/utils/prompting.py`
  ensures identical byte streams across training, evaluation, and ad-hoc generation.
- **GRPO Training**: On-policy reinforcement learning with G=8 completions
  per prompt, LoRA (r=32), and 4-bit QLoRA
  quantization — fits in ~11 GB VRAM.
- **Full Cluster Pipeline**: SLURM scripts, tick-based chain, live monitoring dashboard
  (`t2g-monitor`), wandb logging, checkpoint management, and evaluation suite.
- **Ablation Study Ready**: 15 config cells / 27 queue entries (baselines,
  SFT, GRPO and SFT→GRPO in both prompt modes, plus reward / loss / decoding /
  objective ablations) launchable via the `--ablation` flag in
  `cluster/run_all.sh` — all inheriting from `qwen25-05b/base.yaml`.
- **All params configurable via YAML**: reward weights, grammar toggle, RL
  objective knobs (`loss_type`, `scale_rewards`, `mask_truncated_completions`),
  and opt-in auxiliary SFT objectives — no hardcoded values.
- **Efficient**: ~8 hours for 5000 steps (`training.max_steps` in base.yaml) on a single NVIDIA L40S.
- **Comprehensive Test Suite**: 96/96 pytest tests passing (data, grammar,
  rewards, metrics, monitor, config-inheritance, integration) with shared
  `conftest.py` fixtures.

---

## Project Structure

```text
neuro_symbolic_t2g/
├── experiments/configs/qwen25-05b/     # 15 celle / 27 entry di campagna
│   ├── base.yaml                       # Template ereditato via `extends`
│   ├── baseline/                       # Solo eval, nessun training
│   │   ├── zero-shot.yaml              #   base + Trie
│   │   ├── zero-shot-no-grammar.yaml   #   base senza vincolo (lower bound)
│   │   └── few-shot.yaml               #   base + retrieval k=3 + Trie
│   ├── sft/zero-shot.yaml              # Controllo supervisionato
│   ├── grpo/{zero-shot,few-shot}.yaml  # RL dal base model
│   ├── sft-grpo/{zero-shot,few-shot}.yaml  # Pipeline completa SFT→GRPO
│   └── ablations/
│       ├── rewards/{edit-validity,historical-stack}.yaml
│       ├── loss/dr-grpo.yaml           # Dr-GRPO vs default DAPO
│       ├── decoding/{no-grammar,hot-rollout}.yaml
│       └── objectives/{sft-allowed-mass,sft-structured}.yaml
├── src/
│   ├── cluster/                       # SLURM scripts and cluster orchestration
│   │   ├── setup.sh                   # One-shot environment setup
│   │   ├── train.sh / eval.sh         # Job scripts
│   │   ├── run_all.sh                 # Pipeline launcher (train → eval)
│   │   ├── aliases.sh                 # Convenience aliases (t2g-train, t2g-monitor, …)
│   │   └── clean.sh / clean_model.sh  # Cleanup utilities
│   ├── data/
│   │   ├── aslg_dataset.py            # ASLG-PC12 loader, vocab extraction, T2G dataset builder
│   │   └── transition_matrix.py       # Bigram transition matrix computation
│   ├── grammar/
│   │   ├── gloss_grammar.py           # GlossVocabularyMask (vocabolario glossa)
│   │   └── grammar_logits_processor.py # HF LogitsProcessor: Trie dual-root
│   ├── rewards/
│   │   └── t2g_rewards.py             # 8 reward functions (7 attive + edit-validity)
│   ├── training/
│   │   ├── grpo_t2g_train.py          # Main GRPO training loop (7-step pipeline)
│   │   ├── eval_t2g.py                # Checkpoint eval (ROUGE-L, BLEU, best-of-N, compare via config)
│   │   └── callbacks.py               # CompletionSampleLogger + Callback for live monitoring
│   └── utils/
│       ├── chain_monitor.py           # Live pipeline dashboard (t2g-monitor)
│       ├── live_training_table.py     # Real-time metric table via `tail -f`
│       ├── metrics.py                 # ROUGE-L Pass@1/Pass@k, reward breakdown
│       ├── prompting.py               # Centralized T2G prompt builder
│       ├── show_training_log.py       # Post-hoc log viewer + training curve plots
│       └── visualization.py           # Reward breakdown plots, baseline comparison
├── tests/                             # Test suite (96/96 pytest pass)
│   ├── conftest.py                    # Shared fixtures (reward_setup, dataset, tokenizer)
│   ├── test_config.py                 # Config inheritance (extends) resolution
│   ├── test_data.py                   # Dataset loader + transition matrix
│   ├── test_grammar.py                # Vocabulary mask + logits processor
│   ├── test_rewards.py                # All 9 reward functions
│   ├── test_metrics.py                # ROUGE-L + reward breakdown
│   ├── test_monitor.py                # Chain monitor + live table
│   ├── test_integration.py            # End-to-end pipeline
│   └── run_all_tests.sh               # Batch runner (pytest)
├── main.py                            # Component testing (data, grammar, rewards, generation)
├── pyproject.toml                     # Core deps + optional GPU extras (unsloth, vllm)
├── sync_cluster.ps1                   # Upload/download to cluster (PowerShell)
├── TRAINING.md                        # Detailed training guide (what to expect, monitor, resume)
├── CLUSTER.md                         # Complete cluster setup and operations guide
├── docs/
│   ├── RECOVERY_REPORT.md             # What changed vs main/edit-rewards, and the evidence
│   ├── NEW_OBJECTIVES_SPEC.md         # Allowed-mass and structured loss: design + promotion gates
│   ├── HISTORICAL_RECORD.md           # Provenance index for the retained historical docs
│   ├── EVALUATION.md                  # Evaluation protocol (metrics, splits, honest reporting)
│   ├── SOURCES.md                     # Verified bibliography: how each source is used
│   ├── FINDINGS.md                    # Historical results table with run IDs
│   ├── REWARDS.md                     # HISTORICAL: the 10 removed reward functions
│   ├── METRICS.md                     # HISTORICAL: masked-mass / entropy diagnostics
│   ├── RESEARCH_REPORT.md             # HISTORICAL: original project overview
│   └── T2G_PIPELINE_REVIEW.md         # HISTORICAL: root-cause forensics of 3 training bugs
└── README.md                          # This file
```

---

## The Pipeline in 7 Steps

| Step | What                                                                                                                                                                                                                   | Where                            |
| ---- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------- |
| 1    | **Data**: Download ASLG-PC12 (87K English→Gloss pairs) from Hugging Face                                                                                                                                               | `src/data/aslg_dataset.py`       |
| 2    | **Model**: Load Qwen2.5-0.5B-Instruct with LoRA (r=32) + 4-bit QLoRA via Unsloth                                                                                                                                       | `src/training/grpo_t2g_train.py` |
| 3    | **Constrained Decoding**: Build `GlossVocabularyMask` + dual-root token Trie — model can only output ASL gloss tokens                                                                                                | `src/grammar/gloss_grammar.py`   |
| 4    | **Dataset**: Format prompt-completion pairs with chat template                                                                                                                                                         | `src/data/aslg_dataset.py`       |
| 5    | **Reward Functions**: 8 deterministic rewards — translation quality (ROUGE-L), BLEU-4, gold-structure, gloss-order (edit-distance), verifier-scaled (RECIPE), format, repetition (7 attive di default) più edit-validity (opt-in) | `src/rewards/t2g_rewards.py`     |
| 6    | **GRPO Training**: `trl.GRPOTrainer` generates G=8 completions per prompt, computes rewards, updates LoRA weights                                                                                                      | `src/training/grpo_t2g_train.py` |
| 7    | **Save**: Checkpoint every `training.save_steps` (500 in base.yaml) + final model in `experiments/checkpoints/qwen25-05b/<method>/<prompt-mode>/run_<timestamp>/final/`                                                                   | Auto                             |

---

## Reward Functions

| Component                             | Weight (optimal v2.1) | What it measures                                                             |
| ------------------------------------- | --------------------- | ---------------------------------------------------------------------------- |
| **Translation Quality** (ROUGE-L)     | 0.20                  | Lexical similarity between generated and gold gloss sequence                 |
| **BLEU-4**                            | 0.20                  | BLEU-4 precision vs gold gloss (RVLF 2025)                                     |
| **Gold-Structure** (Gold Baseline) ⭐ | 0.20                  | Bigram score vs the gold reference gloss — "as good as the human?"           |
| **Gloss Order** (Edit-Distance)       | 0.10                  | Normalized Levenshtein distance between generated and gold token order       |
| **Verifier-Scaled** (RECIPE)          | 0.10                  | log1p(structural) used as confidence multiplier for translation quality      |
| **Format**                            | 0.10                  | Ensures output is only gloss tokens (penalizes free text, punctuation, JSON) |
| **Repetition**                        | 0.05                  | Penalizes degenerate loops (token/trigram repetition > 50%)                  |
| — *opt-in:* Edit-Validity | 0 (off) | Similarità di edit con termine di validità graduato; attivata da `ablations/rewards/edit-validity.yaml` |

All rewards are **deterministic and rule-based** — no neural reward model, no
human feedback required.

See [docs/REWARDS.md](docs/REWARDS.md) for full details.

---

## Setup

### Local (for development and testing)

**Prerequisites**: Python 3.10+ and [uv](https://docs.astral.sh/uv/) or pip.

```bash
git clone <repo-url>
cd neuro_symbolic_t2g

# Core install — all training deps (incl. scikit-learn for tfidf retrieval)
pip install -e .

# Dev tools (pytest, ruff, black, isort)
pip install -e ".[dev]"

# Optional: minilm retrieval backend (heavy — downloads HF models)
pip install -e ".[retrieval]"

# For cluster: torch built for your CUDA version
# Example — CUDA 12.1:
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install -e .
```

### Test components locally

```bash
# Run all tests with pytest
uv run python -m pytest tests/ -v

# Skip data/integration tests (require dataset download)
bash tests/run_all_tests.sh --skip-data

# Run a single test file
uv run python -m pytest tests/test_rewards.py -v

# Component testing (data → grammar → rewards → generation)
uv run python main.py
```

### Cluster Setup

See the complete [**CLUSTER.md**](CLUSTER.md) guide. Quick start:

```bash
# 1. Upload project to cluster
.\neuro_symbolic_t2g\sync_cluster.ps1 -Action upload    # Windows PowerShell
# OR: rsync -avz neuro_symbolic_t2g/ user@gcluster:~/neuro_symbolic_t2g/

# 2. SSH into cluster
ssh <user>@gcluster.dmi.unict.it

# 3. Setup (downloads dataset, installs deps, computes transitions)
srun --account <queue> --partition <queue> --qos gpu-xlarge --gres=gpu:1 --pty bash
cd ~/neuro_symbolic_t2g && bash cluster/setup.sh

# 4. Load aliases and launch pipeline
source ~/neuro_symbolic_t2g/cluster/aliases.sh
t2g-run-all
t2g-monitor
```

---

## Usage

### Training (via SLURM)

```bash
# Single-model training
CONFIG=experiments/configs/qwen25-05b/sft-grpo/few-shot.yaml sbatch cluster/train.sh

# Resume from checkpoint
CONFIG=experiments/configs/qwen25-05b/sft-grpo/few-shot.yaml EXTRA_ARGS="--resume" sbatch cluster/train.sh
```

### Pipeline (train → eval, automatic)

```bash
source ~/neuro_symbolic_t2g/cluster/aliases.sh

t2g-run-all          # Full pipeline (tick-based chain)
t2g-monitor          # Live dashboard
t2g-monitor --all    # Full: table + metrics + completion samples
```

### Quick Alias Reference

| Command             | What it does                    |
| ------------------- | ------------------------------- |
| `t2g-train`         | Submit training job             |
| `t2g-eval`          | Submit evaluation job           |
| `t2g-run-all`       | Launch full train→eval pipeline |
| `t2g-monitor`       | Live pipeline dashboard         |
| `t2g-trainlog <ID>` | Tail training log               |
| `t2g-gpu`           | Show GPU usage on active node   |
| `t2g-chain-show`    | Show pipeline status            |
| `t2g-chain-stop`    | Stop pipeline (preserves state) |
| `t2g-clean`         | Clean workspace                 |
| `t2g-pip-reset`     | Reset pip environment           |
| `t2g-help`          | Show all aliases                |

### Evaluation

L'invocazione è **solo** `--config` (+ `--checkpoint` se presente): tutti i
knob comportamentali (max_samples, num_samples, best_of_n, prompting,
dual_prompting, compare, plot, ...) vivono nella sezione `evaluation:` del
config. Riferimento chiave per chiave: [docs/CONFIG_REFERENCE.md](docs/CONFIG_REFERENCE.md).

```bash
# Eval di un checkpoint (compare/best_of_n/prompting/… decisi dalla sezione evaluation)
uv run python -m src.training.eval_t2g \
    --config experiments/configs/qwen25-05b/sft-grpo/few-shot.yaml \
    --checkpoint experiments/checkpoints/qwen25-05b/sft-grpo/few-shot/run_<timestamp>/final

# Baseline del base model (celle baseline/*, senza --checkpoint):
uv run python -m src.training.eval_t2g \
    --config experiments/configs/qwen25-05b/baseline/few-shot.yaml
```

### Monitoring & Visualization

```bash
# Live metric table (pipe from SLURM log)
tail -f logs/slurm-train-<ID>.log | python -u -m src.utils.live_training_table

# Post-hoc: training log table
python -m src.utils.show_training_log experiments/checkpoints/grpo/t2g/qwen05/ --last

# Training curve plots (PNG with polynomial regression)
python -m src.utils.show_training_log experiments/checkpoints/grpo/t2g/qwen05/ --plot

# Weights & Biases (offline mode on cluster)
wandb sync logs/wandb/offline-run-*
```

---

## What to Expect

### Training Progress

| Phase    | Steps    | Translation ROUGE-L | Model Behavior                                                                         |
| -------- | -------- | ------------------- | -------------------------------------------------------------------------------------- |
| Initial  | 0–200    | 0.0–0.1             | Random/copying — constrained decoder ensures valid gloss tokens but output is nonsense |
| Mid      | 200–800  | 0.2–0.4             | Learns to associate gloss tokens with input meaning. Bigram structure improves.        |
| Advanced | 800–1500 | 0.5–0.7             | Reasonably accurate gloss translations. Learns typical ASL gloss patterns.             |

**Total time**: ~8 hours for 5000 steps on L40S (batch_size=1, grad_accum=8, ~5,8 s/step misurati).

### What NOT to expect

- **Not a production translator**: Qwen 0.5B is a small model. Quality is sufficient for
  demonstrating the neuro-symbolic methodology, not for deployment.
- **Constrained ≠ Correct**: The decoder guarantees valid gloss tokens, not correct
  translations. The model can still produce grammatically valid but semantically wrong
  gloss sequences.
- **vLLM not used during training**: The HF `LogitsProcessor` is incompatible with
  vLLM's sampling engine. vLLM is available for fast inference post-training.

---

## GPU Compatibility

| GPU  | Compute Cap. | Unsloth | 4-bit QLoRA | Notes           |
| ---- | ------------ | ------- | ----------- | --------------- |
| L40S | 8.9          | ✅      | ✅          | Ideal           |
| V100 | 7.0          | ✅      | ✅          | No bf16 support |
| K80  | 3.7          | ❌      | ❌          | fp16 only, slow |

For K80 or CPU-only, set `use_unsloth: false` and `quantization: null` in the config.

---

## Configuration

I config YAML in `experiments/configs/qwen25-05b/` usano **ereditarietà**: le parti
comuni (modello, LoRA, dataset, training, GRPO, reward, grammar, evaluation,
wandb) vivono in `base.yaml` e ogni config specifico la estende con `extends`
sovrascrivendo **solo le proprie differenze**. La resolution (deep merge
ricorsivo: dict fusi, liste/scalari sostituiti) avviene in
`src/utils/config.py::resolve_config` — il dict risultante è identico a prima
per i trainer, che non vedono mai la chiave `extends`.

```yaml
# experiments/configs/qwen25-05b/sft-grpo/few-shot.yaml
extends: ../base.yaml                # eredita modello/LoRA/dataset/reward/grammar/evaluation…

training:
  learning_rate: 3.0e-6              # sovrascrive SOLO ciò che cambia
  warmup_steps: 200
  output_dir: "experiments/checkpoints/qwen25-05b/sft-grpo/few-shot"
  log_dir: "experiments/logs/qwen25-05b/sft-grpo/few-shot"

retrieval:
  enabled: true                      # attiva il few-shot (k esempi nel prompt)
  top_k: 3
```

Le chiavi ereditate (da `base.yaml`):

```yaml
model:
  name: "Qwen/Qwen2.5-0.5B-Instruct"
  quantization: "4bit" # 4bit / 8bit / null
  use_unsloth: true # Optimized training

training:
  max_steps: 5000 # governa il GRPO (1 prompt per passo, non 8)
  per_device_train_batch_size: 1
  gradient_accumulation_steps: 8

grpo:
  num_generations: 8 # G = completions per prompt
  beta: 0.04 # KL penalty
  temperature: 0.7 # Exploration temperature

reward:
  weight_translation: 0.20 # ROUGE-L similarity
  weight_bleu: 0.20 # BLEU-4 (RVLF 2025)
  weight_gold_structure: 0.20 # Gold baseline (⭐ recommended)
  weight_gloss_order: 0.10 # Edit-distance ordering
  weight_verifier_scaled: 0.10 # RECIPE-inspired
  weight_format: 0.10 # Gloss-only check
  weight_repetition: 0.10 # Repetition penalty

evaluation:
  max_samples: 3000 # Prompt di eval sottocampionati (seeded)
  num_samples: 5 # Generazioni per prompt
  best_of_n: true # Best-of-N oracolo (blocco separato)

grammar:
  enabled: true # Trie dual-root sul vocabolario glossa
```

> Nota: `grammar.track_diagnostics` esiste nei config ma **non è letta da
> nessuno** (la telemetria non si attiva via config), e `evaluation.batch_size`
> e `training.warmup_ratio` sono anch'esse chiavi non consumate. Dettagli in
> [docs/CONFIG_REFERENCE.md](docs/CONFIG_REFERENCE.md).

> **Riallineamento iperparametri (base.yaml)**: i valori di riferimento GRPO
> sono stati allineati al config che converge (beta=0.04, temperature=0.7,
> max_grad_norm=0.1). Nei config che non li sovrascrivono esplicitamente
> (es. `sft-grpo`) questo è un cambio intenzionale rispetto a beta=0.0 /
> temperature=0.9 / max_grad_norm=0.05 — beta=0 causava kl=0 e drift del
> policy senza ancora. Vedi il commento in `base.yaml`.

---

## Output

```text
experiments/checkpoints/qwen25-05b/<method>/<prompt-mode>/run_<timestamp>/
├── checkpoint-500/                # Every training.save_steps (500 in base.yaml)
├── checkpoint-1000/               # …
└── final/                         # Final model

logs/
├── slurm-train-<JOB_ID>.log     # Full training log
├── slurm-eval-<JOB_ID>.log      # Evaluation log
├── chain.log                    # Pipeline orchestrator log
└── wandb/                       # Weights & Biases offline logs
```

---

## License

[MIT](LICENSE)

---

## References

- **Othman, A. & Jemni, M.** (2012). English-ASL Gloss Parallel Corpus 2012. [Hugging Face](https://huggingface.co/datasets/achrafothman/aslg_pc12)
- **TRL — Transformer Reinforcement Learning**: [GRPOTrainer](https://huggingface.co/docs/trl/grpo_trainer)
- **Unsloth** _(optional GPU extra)_: [FastLanguageModel](https://docs.unsloth.ai/)
- **vLLM** _(optional GPU extra)_: [Inference engine](https://docs.vllm.ai/)
- **Constrained decoding**: dual-root token Trie su vocabolario glossa chiuso
- **Test Suite**: 400 pytest tests - `python -m pytest tests -q` per l'inventario completo
