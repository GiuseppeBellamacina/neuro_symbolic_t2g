# Neuro-Symbolic T2G — Constrained Decoding + GRPO for ASL Gloss Generation

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![TRL](https://img.shields.io/badge/TRL-GRPO-red.svg)](https://huggingface.co/docs/trl/)
[![Tests](https://img.shields.io/badge/Tests-37%2F37%20pytest-green.svg)](tests/)
[![Docs](https://img.shields.io/badge/Docs-REWARDS%20%7C%20METRICS-purple.svg)](docs/)
[![Ablation](https://img.shields.io/badge/Ablation-8%2B%20variants-orange.svg)](experiments/configs/t2g/)

## Overview

**neuro_symbolic_t2g** applies **Group Relative Policy Optimization (GRPO)** to fine-tune
a small LLM (Qwen2.5-0.5B-Instruct) for **Text-to-Gloss (T2G)** translation — converting
English sentences into **ASL (American Sign Language) gloss sequences**.

The key innovation is the **neuro-symbolic architecture**: a **constrained decoder** forces
every generated token to belong to the ASL gloss vocabulary (~15K tokens), while **GRPO**
optimizes the model through reinforcement learning with **9 rule-based reward functions**.

> **No neural reward model needed** — the reward is purely deterministic, computed from
> ROUGE-L similarity, bigram transition probabilities (softmax-normalized), Viterbi alignment,
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
  step — the model can only produce valid ASL glosses. Supports both a lightweight
  vocabulary mask and a full grammarllm LL(1) PDA pipeline (_experimental_).
  W&B-tracked diagnostics (`MaskedMassTracker` mixin) monitor masked probability mass,
  full-distribution entropy, and allowed-token entropy.
- **9 Deterministic Rewards**: Translation quality (ROUGE-L), gold-structure (⭐
  recommended), structural dense (softmax+temperature), gloss-order (edit-distance),
  verifier-scaled (RECIPE-inspired, log1p+softmax), soft-Viterbi (differentiable),
  Viterbi (hard alignment), format, and repetition penalty — no neural reward model overhead.
- **Best-of-N Selection**: Evaluation supports `best_of_n` mode — generates N samples
  per prompt and selects the best by reward, with `--compare` flag for automatic
  baseline-vs-GRPO comparison plots and JSON reports.
- **W&B Integration**: Offline mode with `console_multipart=True`, crash-safe try/finally,
  tagged runs, comparison plots, and JSON artifact logging.
- **Robust Gold Gloss Lookup**: Uses deterministic SHA256 hashing of user instructions
  to reliably match gold glosses regardless of prompt formatting — eliminates silent
  ROUGE-L=0 failures during training.
- **Centralized Prompting**: Single `build_t2g_prompt()` in `src/utils/prompting.py`
  ensures identical byte streams across training, evaluation, and ad-hoc generation.
- **GRPO Training**: On-policy reinforcement learning with G=4 completions per prompt,
  LoRA (r=16), and 4-bit QLoRA quantization — fits in ~11 GB VRAM.
- **Full Cluster Pipeline**: SLURM scripts, watcher daemon, live monitoring dashboard
  (`t2g-monitor`), wandb logging, checkpoint management, and evaluation suite.
- **Ablation Study Ready**: 6 config variants (zero-shot, grammar-only, GRPO variants,
  SFT, PDA) launchable via `--ablation` flag in `cluster/run_all.sh`.
- **All params configurable via YAML**: Viterbi diversity penalties, PDA temperature,
  reward weights, grammar toggle — no hardcoded values.
- **Efficient**: ~2-3 hours for 1500 steps on a single NVIDIA L40S.
- **Comprehensive Test Suite**: 37/37 pytest tests passing (data, grammar, rewards,
  metrics, monitor, integration) with shared `conftest.py` fixtures.
- **Experimental Config**: `grpo_experimental_all.yaml` activates all 9 reward modules
  simultaneously for ablation of the full reward space.

---

## Project Structure

```text
neuro_symbolic_t2g/
├── experiments/configs/t2g/
│   ├── grpo_optimal.yaml              # Optimal config (LoRA r=32, all key rewards)
│   ├── grpo_experimental_all.yaml     # Experimental: all 9 reward modules active
│   ├── grpo_qwen05.yaml               # Main training config
│   ├── sft.yaml                       # SFT baseline config
│   └── ablation/                      # Ablation study variants
│       ├── zero_shot.yaml
│       ├── zero_shot_grammar.yaml
│       ├── grpo_no_grammar.yaml
│       └── grpo_pda.yaml
├── src/
│   ├── cluster/                       # SLURM scripts and cluster orchestration
│   │   ├── setup.sh                   # One-shot environment setup
│   │   ├── train.sh / eval.sh         # Job scripts
│   │   ├── run_all.sh                 # Pipeline launcher (train → eval)
│   │   ├── chain_next.sh              # Watcher daemon (sequential job chain)
│   │   ├── aliases.sh                 # Convenience aliases (t2g-train, t2g-monitor, …)
│   │   └── clean.sh / clean_model.sh  # Cleanup utilities
│   ├── data/
│   │   ├── aslg_dataset.py            # ASLG-PC12 loader, vocab extraction, T2G dataset builder
│   │   └── transition_matrix.py       # Bigram transition matrix computation
│   ├── grammar/
│   │   ├── gloss_grammar.py           # GlossVocabularyMask + grammarllm pipeline factory
│   │   └── grammar_logits_processor.py # HF LogitsProcessor (vocab mask + PDA variants)
│   ├── rewards/
│   │   └── t2g_rewards.py             # 9 reward functions for GRPO
│   ├── training/
│   │   ├── grpo_t2g_train.py          # Main GRPO training loop (7-step pipeline)
│   │   ├── eval_t2g.py                # Checkpoint eval (ROUGE-L, BLEU, best-of-N, --compare)
│   │   └── callbacks.py               # CompletionSampleLogger + Callback for live monitoring
│   └── utils/
│       ├── chain_monitor.py           # Live pipeline dashboard (t2g-monitor)
│       ├── live_training_table.py     # Real-time metric table via `tail -f`
│       ├── metrics.py                 # ROUGE-L Pass@1/Pass@k, reward breakdown
│       ├── prompting.py               # Centralized T2G prompt builder
│       ├── show_training_log.py       # Post-hoc log viewer + training curve plots
│       └── visualization.py           # Reward breakdown plots, baseline comparison
├── tests/                             # Test suite (37/37 pytest pass)
│   ├── conftest.py                    # Shared fixtures (reward_setup, dataset, tokenizer)
│   ├── test_data.py                   # Dataset loader + transition matrix
│   ├── test_grammar.py                # Vocabulary mask + logits processor
│   ├── test_rewards.py                # All 9 reward functions
│   ├── test_metrics.py                # ROUGE-L + reward breakdown
│   ├── test_monitor.py                # Chain monitor + live table
│   ├── test_integration.py            # End-to-end pipeline
│   └── run_all_tests.sh               # Batch runner (pytest)
├── grammarllm/                        # Vendored grammarllm library (PDA-based constrained decoding)
├── main.py                            # Component testing (data, grammar, rewards, generation)
├── pyproject.toml                     # Core deps + optional GPU extras (unsloth, vllm)
├── sync_cluster.ps1                   # Upload/download to cluster (PowerShell)
├── TRAINING.md                        # Detailed training guide (what to expect, monitor, resume)
├── CLUSTER.md                         # Complete cluster setup and operations guide
├── docs/
│   ├── REWARDS.md                     # Detailed reward function documentation (9 rewards)
│   ├── METRICS.md                     # W&B grammar metric documentation (masked mass, entropy)
│   ├── CONFIGS.md                     # Config matrix and ablation study documentation
│   ├── CONFIGS_GUIDE.md               # Detailed config field reference
│   ├── DOCUMENTAZIONE.md              # grammarllm library documentation (Italian)
│   └── ERRORI_E_MIGLIORIE.md          # Known issues and improvements (Italian)
└── README.md                          # This file
```

---

## The Pipeline in 7 Steps

| Step | What                                                                                                                                                                                                                   | Where                            |
| ---- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------- |
| 1    | **Data**: Download ASLG-PC12 (87K English→Gloss pairs) from Hugging Face                                                                                                                                               | `src/data/aslg_dataset.py`       |
| 2    | **Model**: Load Qwen2.5-0.5B-Instruct with LoRA (r=16) + 4-bit QLoRA via Unsloth                                                                                                                                       | `src/training/grpo_t2g_train.py` |
| 3    | **Constrained Decoding**: Build `GlossVocabularyMask` (or full grammarllm PDA) — model can only output ASL gloss tokens                                                                                                | `src/grammar/gloss_grammar.py`   |
| 4    | **Dataset**: Format prompt-completion pairs with chat template                                                                                                                                                         | `src/data/aslg_dataset.py`       |
| 5    | **Reward Functions**: 9 deterministic rewards — translation quality, gold-structure (⭐), structural dense (softmax), gloss-order (edit-distance), verifier-scaled (RECIPE), soft-Viterbi, Viterbi, format, repetition | `src/rewards/t2g_rewards.py`     |
| 6    | **GRPO Training**: `trl.GRPOTrainer` generates G=4 completions per prompt, computes rewards, updates LoRA weights                                                                                                      | `src/training/grpo_t2g_train.py` |
| 7    | **Save**: Checkpoint every 100 steps + final model in `experiments/checkpoints/grpo/t2g/qwen05/final/`                                                                                                                 | Auto                             |

---

## Reward Functions

| Component                             | Weight (optimal) | What it measures                                                             |
| ------------------------------------- | ---------------- | ---------------------------------------------------------------------------- |
| **Translation Quality** (ROUGE-L)     | 0.30             | Lexical similarity between generated and gold gloss sequence                 |
| **Gold-Structure** (Gold Baseline) ⭐ | 0.20             | Bigram score vs the gold reference gloss — "as good as the human?"           |
| **Structural Dense** (Softmax)        | 0.10             | Bigram probability with softmax normalization + temperature scaling          |
| **Gloss Order** (Edit-Distance)       | 0.10             | Normalized Levenshtein distance between generated and gold token order       |
| **Verifier-Scaled** (RECIPE)          | 0.10             | log1p(structural) with softmax + verifier_temperature decoupled from gamma   |
| **Soft-Viterbi** (Differentiable)     | 0.05             | Differentiable Viterbi alignment score                                       |
| **Viterbi** (Hard)                    | 0.05             | Hard Viterbi alignment upper bound                                           |
| **Format**                            | 0.05             | Ensures output is only gloss tokens (penalizes free text, punctuation, JSON) |
| **Repetition**                        | 0.05             | Penalizes degenerate loops (token/trigram repetition > 50%)                  |

All rewards are **deterministic and rule-based** — no neural reward model, no
human feedback required. The `grpo_experimental_all.yaml` config activates all 9
modules simultaneously for full reward-space ablation.

See [docs/REWARDS.md](docs/REWARDS.md) for full details.

---

## Setup

### Local (for development and testing)

**Prerequisites**: Python 3.10+ and [uv](https://docs.astral.sh/uv/) or pip.

```bash
git clone <repo-url>
cd neuro_symbolic_t2g

# CPU / development — core dependencies only
pip install -e .

# GPU training — includes Unsloth and vLLM accelerators
pip install -e ".[gpu]"

# For cluster: torch built for your CUDA version
# Example — CUDA 12.1:
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install -e ".[gpu]"
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
srun --account <queue> --partition <queue> --qos gpu-medium --gres=gpu:1 --pty bash
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
CONFIG=experiments/configs/t2g/grpo_qwen05.yaml sbatch cluster/train.sh

# Resume from checkpoint
CONFIG=experiments/configs/t2g/grpo_qwen05.yaml EXTRA_ARGS="--resume" sbatch cluster/train.sh
```

### Pipeline (train → eval, automatic)

```bash
source ~/neuro_symbolic_t2g/cluster/aliases.sh

t2g-run-all          # Full pipeline with watcher
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

```bash
# Evaluate a specific checkpoint
uv run python -m src.training.eval_t2g \
    --config experiments/configs/t2g/grpo_optimal.yaml \
    --checkpoint experiments/checkpoints/grpo/t2g/qwen05/final \
    --max_samples 500

# Best-of-N evaluation (generate N samples, select best by reward)
uv run python -m src.training.eval_t2g \
    --config experiments/configs/t2g/grpo_optimal.yaml \
    --checkpoint experiments/checkpoints/grpo/t2g/qwen05/final \
    --best-of-n --num-samples 5

# Compare baseline vs GRPO (auto-eval both, generate comparison plots + JSON)
uv run python -m src.training.eval_t2g \
    --config experiments/configs/t2g/grpo_optimal.yaml \
    --checkpoint experiments/checkpoints/grpo/t2g/qwen05/final \
    --compare
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

**Total time**: ~2–3 hours for 1500 steps on L40S (batch_size=1, grad_accum=8).

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

Key parameters in `experiments/configs/t2g/grpo_qwen05.yaml`:

```yaml
model:
  name: "Qwen/Qwen2.5-0.5B-Instruct"
  quantization: "4bit" # 4bit / 8bit / null
  use_unsloth: true # Optimized training
  fast_inference: false # Incompatible with constrained decoding

training:
  max_steps: 1500
  per_device_train_batch_size: 1
  gradient_accumulation_steps: 8

grpo:
  num_generations: 4 # G = completions per prompt
  beta: 0.04 # KL penalty
  temperature: 0.7 # Exploration temperature

reward:
  weight_translation: 0.30 # ROUGE-L similarity
  weight_gold_structure: 0.20 # Gold baseline (⭐ recommended)
  weight_structure: 0.10 # Softmax bigram (temperature-scaled)
  weight_gloss_order: 0.10 # Edit-distance ordering
  weight_verifier_scaled: 0.10 # RECIPE-inspired (log1p + softmax)
  weight_soft_viterbi: 0.05 # Differentiable Viterbi
  weight_viterbi: 0.05 # Hard Viterbi alignment
  weight_format: 0.05 # Gloss-only check
  weight_repetition: 0.05 # Repetition penalty

evaluation:
  max_samples: 500 # Eval subset size
  num_samples: 5 # Samples per prompt (for best-of-N)
  best_of_n: false # Enable best-of-N selection

grammar:
  enabled: true
  use_grammarllm_pda: false # Set true for LL(1) PDA path
  viterbi_diversity: # Configurable Viterbi penalties
    self_loop_penalty: 0.5
    max_occurrences: 2
    diversity_threshold: 0.3
    max_iters: 3
    verifier_temperature: 5.0 # Decoupled from verifier_gamma
```

---

## Output

```text
experiments/checkpoints/grpo/t2g/qwen05/
├── checkpoint-100/              # After 100 steps
├── checkpoint-200/              # …
└── final/                       # Final model

logs/
├── slurm-train-<JOB_ID>.log     # Full training log
├── slurm-eval-<JOB_ID>.log      # Evaluation log
├── chain_watcher.log            # Pipeline orchestrator log
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
- **grammarllm**: Vendored constrained decoding library (PDA + LogitsProcessor)
- **Test Suite**: 37/37 pytest tests — see [tests/REPORT.md](tests/REPORT.md) for full test inventory
