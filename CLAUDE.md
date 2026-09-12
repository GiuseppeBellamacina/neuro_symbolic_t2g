# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

GRPO (Group Relative Policy Optimization) fine-tuning of Qwen2.5-0.5B-Instruct to
translate English sentences into ASL gloss sequences (Text-to-Gloss). A constrained
decoder (`LogitsProcessor` + dual-root token Trie) restricts generation to a closed
~15K-token gloss vocabulary; GRPO optimizes the policy against 7 deterministic,
rule-based reward functions (no neural reward model). Training runs on a SLURM GPU
cluster (DMI UniCT); this workstation is used for development, local component
testing, and driving the cluster remotely.

Read [README.md](README.md) for the full pipeline/config/reward description,
[TRAINING.md](TRAINING.md) for what a training run looks like phase-by-phase, and
[CLUSTER.md](CLUSTER.md) for cluster access, SLURM constraints, and the chain/tick
orchestration model. `docs/CONFIG_REFERENCE.md` is the key-by-key config reference;
`docs/REWARDS.md`/`docs/METRICS.md` cover historical reward/diagnostics detail.

**Note**: README.md's "Project Structure" tree (`src/data/`, `src/cluster/`) is
stale — the actual package layout is `src/datasets/`, `src/models/`, `src/retrieval/`,
`src/analysis/`, and cluster scripts live at top-level `cluster/`, not `src/cluster/`.
Trust the structure below and the filesystem over that tree.

## Commands

```bash
# Install (uv preferred; pip -e . works too)
uv sync                      # core deps
uv sync --extra dev          # + pytest, ruff, black, isort, fastapi/uvicorn/httpx
uv sync --extra tui          # + textual/httpx for remote/tui.py (self-contained, no `dev` needed)
uv sync --extra retrieval    # + sentence-transformers (minilm retrieval backend; tfidf is default and needs no extra)

# Tests
uv run python -m pytest tests/ -v
uv run python -m pytest tests/test_rewards.py -v      # single file
uv run python -m pytest tests/test_rewards.py -k name # single test
bash tests/run_all_tests.sh --skip-data               # skip tests needing dataset download (test_data.py, test_integration.py)
uv run python tests/validate_configs.py               # sanity-check every experiment YAML resolves (extends chains, required keys)

# Component smoke-test (data → grammar → rewards → generation), no training
uv run python main.py                                 # all stages
uv run python main.py --task grammar                  # one stage: data|grammar|rewards|single_generation

# Format/lint (also runs automatically pre-commit via .githooks/pre-commit)
bash format.sh          # Linux/macOS: isort . && black . && ruff check --fix .
powershell -File format.ps1   # Windows equivalent

# Training / eval entrypoint (real training needs a GPU + Unsloth/QLoRA — normally run on the cluster, not here)
uv run python -m src.training --config experiments/configs/qwen25-05b/<cell>.yaml [--resume] [--prepare-data]
uv run python -m src.training.eval_t2g --config experiments/configs/qwen25-05b/<cell>.yaml [--checkpoint <path>]
```

One-time repo setup: `git config core.hooksPath .githooks` — this repo's git hooks live
in `.githooks/`, not `.git/hooks/` (see `.githooks/README.md`). `pre-commit` runs the
formatter and re-stages changed files; `pre-push` best-effort `scp`s `src/`, `cluster/`,
`experiments/configs/`, and docs to the cluster (never blocks the push if unreachable).

## Architecture

### The pipeline (config-driven, all knobs in YAML)

`src/training/__main__.py` is the bootstrap: it lightweight-peeks the resolved config
(without importing torch) to decide whether to import Unsloth *before* torch/transformers/trl
(required import order), applies an eval-only guard (baseline cells have no
`output_dir`/`log_dir` and must go through `eval_t2g`, not the trainer), then routes to
`src.training.sft_train.main` or `src.training.grpo_t2g_train.main` based on
`training.trainer` (default `grpo`).

1. **Data** (`src/datasets/aslg_dataset.py`) — downloads ASLG-PC12 (87K pairs) from HF,
   extracts the gloss vocabulary, builds prompt/completion pairs.
2. **Transitions** (`src/datasets/transition_matrix.py`, `structured_transitions.py`) —
   bigram transition matrix over the gloss vocab, used by the gold-structure/verifier-scaled
   rewards.
3. **Grammar** (`src/grammar/`) — `gloss_grammar.py` builds `GlossVocabularyMask` (gloss
   string ↔ token id mapping); `grammar_logits_processor.py` is the HF `LogitsProcessor`
   (dual-root token Trie) that masks every non-gloss token at each generation step;
   `masked_mass_tracker.py` is an optional, off-by-default diagnostic.
4. **Retrieval** (`src/retrieval/example_retriever.py`) — optional few-shot example
   retrieval (tfidf default / sentence-transformers "minilm" backend), wired in via
   `retrieval.enabled` in config and `src/training/retrieval_setup.py`.
5. **Rewards** (`src/rewards/t2g_rewards.py`) — 7 active + 1 opt-in deterministic reward
   functions, all mapped to `[-1, 1]`, combined via `reward.weight_*` config keys.
6. **Training** — `src/training/grpo_t2g_train.py` (`trl.GRPOTrainer`, G completions/prompt,
   LoRA/QLoRA via Unsloth) or `src/training/sft_train.py` (auxiliary/plain SFT, incl.
   `auxiliary_sft_trainer.py` and `allowed_mass_loss.py` for the SFT→GRPO ablation
   objectives). `src/training/callbacks.py` drives live monitoring output.
7. **Eval** — `src/training/eval_t2g.py`: single entrypoint (`--config` [+ `--checkpoint`]),
   every behavioral knob (best_of_n, compare, dual prompting, plotting, sampling) lives
   under the config's `evaluation:` section, never as CLI flags.

### Config inheritance

All experiment YAMLs live under `experiments/configs/qwen25-05b/` and use an `extends:`
key resolved by `src/utils/config.py::resolve_config` — recursive deep-merge (child wins,
dicts merge, lists/scalars replace), cycle-checked, parent paths relative to the child file.
The merged dict never contains `extends`; trainers/eval never see it. `base.yaml` holds all
shared model/LoRA/dataset/GRPO/reward/grammar/evaluation defaults; each cell
(`baseline/`, `sft/`, `grpo/`, `sft-grpo/`, `ablations/{rewards,loss,decoding,objectives}/`)
overrides only its deltas. When adding a new experiment cell, extend `base.yaml` (or a
sibling) rather than duplicating the full config.

### Centralized prompting

`src/utils/prompting.py::build_t2g_prompt()` is the single source of the prompt template —
used identically by training, eval, retrieval, and `main.py`'s ad-hoc generation test. Gold
gloss lookup during training/eval matches on a SHA256 hash of the user instruction (not
literal prompt string) specifically to stay robust to prompt-formatting differences and
avoid silent ROUGE-L=0 mismatches.

### Cluster orchestration (this workstation drives it remotely)

- `cluster/*.sh` — SLURM job scripts (`train.sh`, `eval.sh`), `run_all.sh` (queues a full
  train→eval pipeline, or a full ablation campaign with `--ablation`), `chain_tick.sh`
  (idempotent one-shot tick that advances the queue — the cluster has no cron/`at`/long-lived
  daemons, so the whole chain is tick-driven), `aliases.sh` (cluster-side bash aliases:
  `t2g-train`, `t2g-monitor`, etc. — source it on the cluster, they are not local commands).
- Cluster constraints (see CLUSTER.md §1): max 1 active + 0 pending job per user (QoS), no
  python/pip/cron/`at` on the login node, long-lived processes get reaped — this is *why*
  everything is a stateless one-shot tick against `.chain_state/` on disk.
- `sync_cluster.ps1` — PowerShell upload/download to the cluster (also auto-triggered
  best-effort by the `pre-push` hook).
- `remote/` — a separate FastAPI microservice (`app.py`) + Textual TUI (`tui.py`) that
  drives the same tick/chain mechanism over SSH + a `cluster_helper.sh` remote helper,
  meant for deployment on Render with cronjob.org pinging `/tick` every 5 min (so the
  chain advances without you being at a terminal). See `remote/README.md` for the full
  protocol, env vars, and API. `.env` (gitignored; copy from `.env.example`) configures
  local cluster SSH access and, for the TUI, the driver's URL/token.
- `src/analysis/campaign_report.py` — aggregates results across an ablation campaign run.

### Tests

`tests/conftest.py` provides shared fixtures (reward setup, dataset, tokenizer).
`tests/test_data.py` and `tests/test_integration.py` require downloading the dataset —
skip them with `--skip-data`/`--ignore` when offline. `tests/validate_configs.py` is a
script (not a pytest module) that resolves every experiment config end-to-end.
