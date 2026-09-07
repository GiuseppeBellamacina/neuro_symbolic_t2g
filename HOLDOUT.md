# Project handoff / holdout

Last updated: 2026-09-06. This file is the recovery point for continuing work.

## 1. What the project is

`neuro_symbolic_t2g` studies English-to-ASL-gloss generation with
Qwen2.5-0.5B-Instruct. Its main scientific comparison separates:

- method: base, SFT, GRPO-from-base, SFT-to-GRPO;
- prompt conditioning: zero-shot versus retrieval few-shot;
- constrained decoding: token Trie by default, PDA only as a manual ablation;
- training objective: completion-only SFT NLL or GRPO/Dr-GRPO;
- diagnostics and pilots: reward qualification, masked probability mass,
  Markov/Viterbi diagnostics, and reduced-state structured loss.

The intended claim is conditional: determine when GRPO helps or damages T2G as
a function of initialization, rollout support, prompting and objective design.
It is not a general claim that GRPO beats SFT.

## 2. Important environment facts

- Cluster alias: `gcluster`; checkout: `~/neuro_symbolic_t2g`.
- Login node has shell/SLURM and `/usr/bin/python3`, but no project Python and no
  Apptainer command suitable for the runtime workflow.
- Canonical setup path is always:
  `login -> srun -> compute -> apptainer run -> setup`.
- Project Python on compute uses `apptainer exec`, not `run`, because the SIF
  runscript was proven on the real cluster to corrupt quoted `python -c` argv.
- Compute training/evaluation/probe jobs are offline and fail if caches are
  missing. Setup is the only online acquisition path.
- QoS allows one active/submitted allocation for the user.

## 3. State before the recent work

The repository already had the primary experiment matrix, SFT/GRPO training,
dual-leg evaluation, Trie/PDA decoding, reward ablations, frozen probes,
structured benchmark primitives, remote API/TUI, W&B offline logging and tests.
However:

- `setup.sh` incorrectly tried to run Apptainer directly on the login node;
- pip constraints conflicted with `sentence-transformers>=5.7` versus the
  container's protected 5.2.3;
- SLURM spool copies could resolve `_lib.sh` from `/var/lib/slurm/...`;
- `run_py` used `apptainer run`, corrupting multiline Python argv;
- train/eval/preflight contained inconsistent bare-Python/container fallbacks;
- the rich Python chain monitor had temporarily been replaced by a reduced
  shell monitor;
- reward qualification treated token-F1 reversal as a required degradation,
  making the order-invariant reward impossible to qualify;
- reward values were not strictly protected against nonfinite/out-of-range
  invalid scores and SacreBLEU roundoff;
- configs repeated many values already inherited from parents;
- structured NLL and allowed mass existed only as benchmark/diagnostic ideas,
  not integrated SFT objectives.

## 4. Current cluster state

### Verified on the real cluster

- `pip-setup` completed successfully.
- Protected stack includes torch 2.7.1+cu118, CUDA 11.8, Transformers 5.3.0,
  TRL 0.24.0, PEFT 0.19.1, torchao 0.17.0, SacreBLEU 2.6.0 and
  sentence-transformers 5.2.3.
- Qwen and ASLG-PC12 are cached; deterministic gloss vocabulary exists.
- Normal preflight completed successfully offline.
- SFT run `run_20260906_140438` stopped normally at step 4800 by early stopping.
  Best validation loss was 0.02618797 at checkpoint 4200; `final/` contains the
  restored best adapter and `sft_fingerprint.json`.
- The last observed active GRPO job was 7241 (`train-grpo-few`). Do not assume it
  is still active: check `squeue --me` and `.chain_state` before acting.
- The rich monitor is restored. After sourcing aliases, use `monitor --all`.

### Files intentionally uploaded in the last targeted updates

- `cluster/train.sh`
- `cluster/eval.sh`
- `cluster/diagnose.sh`
- `cluster/aliases.sh`

Earlier broad uploads included the runtime fixes. Later config and auxiliary
loss development described below was explicitly local-only and must not be
assumed present on the cluster.

## 5. Current local repository state

The working tree is large and uncommitted. Do not run a blind `git add .`.
Inspect status and stage coherent groups only. No commit/push was requested or
performed by the assistant.

### Cluster/runtime fixes now present locally

- `setup.sh` always enters through `srun` then Apptainer.
- All sbatch-capable scripts resolve `_lib.sh` safely from script directory,
  `SLURM_SUBMIT_DIR/cluster`, then `$HOME/neuro_symbolic_t2g/cluster`.
- `_lib.sh::run_py` uses `apptainer exec --nv` and fails rather than using bare
  compute Python.
- train/eval use argv arrays and do not interpolate config paths into Python.
- remote/helper status handling is resilient to `squeue` blips.
- monitor is again `python3 -m src.utils.chain_monitor`, with chain, metrics and
  completion samples.

### Reward work

- Five formulas remain in the reward probe registry: edit similarity, multiset
  token F1, chrF++, ROUGE-L and smoothed sentence BLEU-2.
- Primary `grpo/few-shot` is the edit-validity control.
- Duplicate `ablations/rewards/edit.yaml` was removed locally.
- Four true alternative reward training configs remain.
- Rewards now validate finite bounded invalid scores, clamp similarity to [0,1],
  produce strict [-1,1] outputs, robustly parse TRL completions and stamp source
  implementation provenance.
- Qualification perturbations are metric-specific: token-F1 reversal is an
  expected invariant, not a hard degradation gate.

### Config cleanup

- Before auxiliary-loss additions the cleaned tree had 18 configs.
- Inherited YAMLs contain only actual differences from their direct resolved
  parent; `tests/validate_configs.py` checks this recursively.
- Adding `sft-mass`, `sft-structured`, and `sft-mass-structured` brought the
  current local total to 21 configs.
- These auxiliary configs are manual pilots and are not in the default campaign
  or TUI registry.

### Allowed-mass SFT: Phase A complete and accepted

Implemented locally:

- shared `DualRootGlossTrie` state API used by generation and teacher-forced loss;
- `src/training/allowed_mass_loss.py`;
- strict metadata collator and auxiliary SFT trainer;
- training-only mass objective, with normal LM-only evaluation/early stopping;
- unique `sft-mass` config (`lambda=0.1`, warmup 200);
- strict WORLD_SIZE=1 guard;
- compiled Trie/vocabulary/tokenizer fingerprint;
- no automatic reuse of auxiliary pilot adapters.

Gate A received Oracle `ACCEPT`. Before any cluster launch, it still requires a
one-batch smoke with the exact Qwen/PEFT/Unsloth stack.

### Structured SFT: Phase B implemented but NOT accepted yet

Implemented locally:

- cache-safe sparse `StructuredGraphLoss` and vectorized sparse gold scoring;
- deterministic train-only graph manifest and anti-leakage helpers;
- `AuxiliarySFTModel` wrapper registering backbone and structured head;
- training integration for structured-only and mass+structured objectives;
- manual `sft-structured` and `sft-mass-structured` configs;
- checkpoint hooks intended to save/restore head and graph manifest alongside
  normal adapter checkpoints.

This phase has not passed Gate B. Do not upload or launch these configs yet.

## 6. Dataset encoding/noise finding

`Là?VAI` is not terminal mojibake. The raw HF Arrow cache already contains:

- source text: `report katalin lévai` (valid UTF-8);
- gold gloss: `REPORT KATALIN Là?VAI` (valid UTF-8 plus literal `?`).

Measured on the deterministic split:

- train: 324/72,979 suspicious rows (~0.44%);
- test: 30/8,109 (~0.37%);
- SFT holdout: 5/1,460 (~0.34%);
- train vocabulary: 153 suspicious tokens under the conservative detector.

The issue is upstream gloss corruption/transliteration, especially accented
proper names. Current runs remain reproducible and must not be silently cleaned.
A future cleaned corpus requires a new data protocol/version, content hashes in
fingerprints, new vocabulary and new runs. Prefer deterministic row filtering
over speculative character repair, and retain an as-is control.

Safe encoding hardening still worth doing later: explicit `encoding="utf-8"` on
training log tees and structured benchmark text writes; align non-ASCII behavior
between `GlossVocabularyMask.token_ids` diagnostics and Trie compilation.

## 7. Current failing/unfinished checks

At the last Phase B validation:

1. `tests/test_auxiliary_sft_model.py::test_optimizer_sees_every_trainable_parameter_once`
   used tensor equality through `parameter not in list`, causing the PyTorch
   ambiguous-Boolean exception. It has now been changed to identity comparison
   (`parameter is not ...`) but must be rerun.
2. `test_curriculum_filtered_dataset` sampled only the first 100 shuffled rows
   and sometimes observed 29% hard against a `>30%` threshold. It has now been
   changed to compare full resampled stage distributions and assert stage 3 has
   more hard examples than stage 2. This must be rerun and reviewed.
3. Ruff reported import ordering in `auxiliary_sft_trainer.py`,
   `test_allowed_mass_loss.py`, and `test_sft.py`. Run Ruff with `--fix` only on
   those files, inspect, then rerun Ruff globally.
4. Phase B still needs a full local test run after the fixes above.
5. Phase B needs Oracle Gate B review. Expect scrutiny of:
   - wrapper compatibility with TRL/PEFT;
   - checkpoint `_save`, `_load_from_checkpoint`, `_load_best_model` lifecycle;
   - graph train-only construction and eligibility policy;
   - same-forward hidden capture and gradients;
   - standard SFT/mass-only regression safety.
6. Real stack smoke remains mandatory before any auxiliary training upload:
   actual Qwen tokenizer/chat template, PEFT/Unsloth, BF16, gradient
   checkpointing, one backward/optimizer step, checkpoint/reload/best restore.

## 8. Immediate continuation commands

Run locally from repository root:

```powershell
python -m ruff check src/training/auxiliary_sft_trainer.py tests/test_allowed_mass_loss.py tests/test_sft.py --fix
python -m pytest tests/test_auxiliary_sft_model.py tests/test_curriculum.py -q
python -m pytest tests/test_structured_loss.py tests/test_structured_transitions.py tests/test_structured_alignment.py tests/test_auxiliary_sft_model.py tests/test_allowed_mass_loss.py tests/test_sft.py tests/test_existing_config_identities.py tests/test_campaign_screen.py -q
python tests/validate_configs.py
python -m ruff check .
python -m pytest tests/ -q
git diff --check -- . ":(exclude)docs/report/main.pdf"
```

If all pass, update `.slim/deepwork/auxiliary-sft-losses.md` with evidence and
request Gate B Oracle review attempt 1/3. Do not start Phase C first.

## 9. Remaining planned work

### Phase B completion

- resolve Gate B findings, with at most two planned re-reviews;
- do not claim production readiness before exact-stack smoke;
- update the LaTeX chapters for integrated mass/structured pilots after the
  implementation is accepted.

### Phase C: GRPO versus Dr-GRPO

- add a manual one-factor config changing only identity and
  `grpo.loss_type: grpo` relative to current Dr-GRPO control;
- keep `scale_rewards: none` identical;
- test resolved-config equality except intended fields;
- document that original GRPO normalizes by realized completion length while
  Dr-GRPO uses the fixed maximum length;
- require completion-length, truncation and clipping diagnostics in analysis;
- do not combine this first ablation with auxiliary-SFT checkpoints.

### Documentation/final verification

- update `docs/report` and rebuild `main.pdf` only after Gates B/C;
- run full tests/config validation/Ruff/Bash syntax;
- inspect the very large working tree before staging;
- commit/upload only when explicitly requested;
- use targeted `sync_cluster.ps1 -Action push -Path ...`, not broad upload, for
  isolated runtime fixes.

## 10. Default campaign and scientific boundaries

- Current primary campaign remains seven cells / twelve queue entries.
- Auxiliary SFT, reward alternatives, PDA/hot and GRPO-loss-type comparisons are
  manual experiments and must not silently enter the primary campaign.
- Trie is primary decoding; PDA is an ablation.
- Masked mass and Markov/Viterbi values are diagnostics unless an explicitly
  versioned differentiable objective is enabled.
- Historical metrics are not comparable with the current protocol.
- Formula/unit-test correctness does not establish empirical usefulness; use
  controlled evaluation, paired uncertainty and multiple seeds before claims.
