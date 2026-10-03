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
uv run python -m src.training --config experiments/configs/<dataset>/qwen25-05b/<cell>.yaml [--resume] [--prepare-data]
uv run python -m src.training.eval_t2g --config experiments/configs/<dataset>/qwen25-05b/<cell>.yaml [--checkpoint <path>]
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

1. **Data** (`src/datasets/registry.py` → `aslg_dataset.py` / `phoenix_dataset.py`) —
   `dataset.dataset_name` picks the corpus: ASLG-PC12 (87K pairs, HF cache, 90/10 split)
   or PHOENIX-2014T (German→DGS, official CSVs in `data/phoenix-2014t/`, official
   splits), plus the two non-gloss tasks of GrammarRL (arXiv:2609.39869) linearized as
   closed-vocabulary token sequences: WOS-46985 (hierarchical classification, target
   `DOMAIN AREA`, `data/wos-46985/Data.xlsx`, zero-shot only) and CoNLL-2003 (NER, target
   `TYPE:Entity ...` or `NONE`, `data/conll-2003/`); the three are downloaded on first use with network
   (`src/datasets/download.py`, sha256-pinned; on the cluster `setup.sh`, compute
   nodes are offline). WOS labels are canonicalized per official code `Y` (134
   classes). The registry also builds/caches the closed gloss vocabulary + bigram matrix
   (`dataset.vocab_source`: `train` default, `all` = deliberate train+test leak used only
   by `ablations/decoding/full-vocab-trie.yaml`). `dataset.prompt_profile` (`en-asl` /
   `de-dgs`) selects the language pair in `src/utils/prompting.py`.
2. **Transitions** (`src/datasets/transition_matrix.py`, `structured_transitions.py`) —
   bigram transition matrix over the gloss vocab, used by the gold-structure/verifier-scaled
   rewards.
3. **Grammar** (`src/grammar/`) — `gloss_grammar.py` builds `GlossVocabularyMask` (gloss
   string ↔ token id mapping); `grammar_logits_processor.py` is the HF `LogitsProcessor`
   (dual-root token Trie) that masks every non-gloss token at each generation step;
   `output_grammar.py` defines WHICH language the Trie admits (`grammar.mode`:
   `vocab` closed train vocabulary, default; `source_spans` one Trie per prompt
   built from the prompt's own sentence, CoNLL; `sequences` whole allowed label
   sequences, WOS) and the same grammar drives validity and the format reward;
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

All experiment YAMLs live under `experiments/configs/<dataset>/qwen25-05b/` (`aslg-pc12/`,
`phoenix-2014t/`, `wos-46985/`, `conll-2003/`; every non-ASLG base extends the ASLG-PC12
base and overrides only the `dataset` section (name, `prompt_profile`, vocab/bigram
paths, for WOS `max_source_words`), `retrieval.cache_path`, `wandb` and, for WOS,
`evaluation.dual_prompting`) and use an `extends:`
key resolved by `src/utils/config.py::resolve_config` — recursive deep-merge (child wins,
dicts merge, lists/scalars replace), cycle-checked, parent paths relative to the child file.
The merged dict never contains `extends`; trainers/eval never see it. `base.yaml` holds all
shared model/LoRA/dataset/GRPO/reward/grammar/evaluation defaults; each cell
(`baseline/`, `sft/`, `grpo/`, `sft-grpo/`, `ablations/{rewards,loss,decoding,objectives}/`)
overrides only its deltas. When adding a new experiment cell, extend `base.yaml` (or a
sibling) rather than duplicating the full config.

### Output layout: dataset first

Outputs live at `experiments/{checkpoints,logs,results,figures}/<dataset>/<model>/<cell>/run_<ts>/`
— the cell mirrors the config path under `experiments/configs/` (`src/utils/run_paths.py`:
`split_cell`, `cell_tag`, `cell_sort_key`; legacy `<model>/...` paths still parse as
ASLG-PC12). Job tags are the path below `<dataset>/<model>/` with `/`→`-`, always prefixed
with the dataset (`aslg-pc12-grpo-few-shot`, `phoenix-2014t-grpo-few-shot`), and the wandb
`run_name` is `<dataset>-<model>-<cell path>` (enforced by a test); the bash mirrors are
`cluster/_lib.sh::t2g_tag_from_config` and `remote/cluster_helper.sh::_cell_key`.
`cluster/migrate_dataset_layout.sh` moves a pre-existing cluster tree to this layout.

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
