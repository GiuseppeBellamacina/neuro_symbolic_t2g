# Research Recovery Report — `improvement` vs `edit-rewards`

Status: Phases 0-8 complete. Phase 9 (implementation) not started except one
verified low-risk fix (R1, §12). Oracle Gate 1 returned REVISE and its
corrections are incorporated. Every claim below was independently verified by
the orchestrator; specialist claims that failed verification are marked.

Full working evidence log: `.slim/deepwork/recovery-improvement.md`.

---

## 1. Current repository state

| | `improvement` | `edit-rewards` |
|---|---|---|
| commit | `a8bb684` (== `main`) | `68bc12a` |
| relationship | baseline | `improvement` + 10 commits, **strictly linear** |
| delta | — | 181 files, +24500 / -14260 |
| test suite | 7 failed / 253 passed (**now 260 passed / 0 failed** after R1) | 651 passed / 6 skipped |
| ruff check | clean | clean |
| `experiments/results/` | **18 run dirs, 27 `generations_*.json`, 115 MB** | absent |
| historical `FINDINGS.md` | intact | rewritten to an 8-line disclaimer |

History is linear, so `git diff a8bb684..68bc12a` is the complete and
authoritative delta. There is no merge-conflict risk and no hidden divergence.
Most content arrives in one mega-commit (`c1e182d`); the following 9 commits are
cluster fixes, docs, the report PDF, and a final structured-SFT/config commit.

An inspection worktree of `edit-rewards` is checked out at `.wt/edit-rewards`
(untracked, disposable).

### Context correction

`.slim/deepwork/` contained two prior-agent logs (947 + 121 lines). They show the
reward reduction, config rebuild and Viterbi demotion were **explicitly
user-authorized** at the time ("Superseding direction — fresh-start code and
experiments"), not rogue agent behaviour. Authorization does not establish
correctness, but the framing "previous agents damaged the project" is not
supported for those specific changes.

---

## 2. Branch delta (material items only)

| Area | `improvement` | `edit-rewards` | Class |
|---|---|---|---|
| rewards | 10 fns, 7-component production set | 5 fns, single-select, weight 1.0 | REJECT-IN-PART |
| GRPO loss params | **none set** -> TRL defaults `dapo`/`group`/`False` | explicit `dr_grpo`/`none`/`True` | INVESTIGATE |
| config tree | `configs/t2g/*.yaml` (13) | `configs/qwen25-05b/**` (23+) | INVESTIGATE |
| curriculum | ON in every GRPO cell | **no config enables it** | RECOVER |
| docs | 10 docs present (~4600 lines) | deleted | **RECOVER** |
| Viterbi rewards | active ablation cells | removed | KEEP REMOVED (see §5) |
| Viterbi diagnostics | absent | `path_log_energy`, `hard/soft_viterbi_diagnostic` + brute-force tests | **RECOVER** |
| structured/mass SFT | absent | implemented (Phase A accepted, Phase B not) | INVESTIGATE |
| `apptainer` verb | `run` for python (corrupts argv) | `exec --nv` + `--env` | **RECOVER** |
| dependency pins | `transformers` **unpinned** | pinned + post-verified | **RECOVER** |
| weave shim | 2 flags (broken) | 3 flags (correct) | **RECOVERED (R1 done)** |
| Trie allowed set | 3995 tokens | **3995, bit-identical** | KEEP either |
| eval protocol | sampled N=5 headline | greedy N=1 headline + sampled leg | INVESTIGATE |
| `valid_rouge_l_mean` | `mean(rouge) * validity` | `mean(rouge if valid else 0)` | NEW is correct |

---

## 3. Risk assessment

**CRITICAL**
- `experiments/results/` (115 MB, the only copy of all trusted baselines) is
  **gitignored with 0 tracked files**, on one disk. Mitigated locally in §12;
  off-machine backup still required.
- Reproducibility break: OLD configs lack an `experiment:` block and are
  **unrunnable** on `edit-rewards` path code. Historical cells cannot be re-run
  from their own configs on that branch.

**HIGH**
- Irreplaceable doc deletion (`REWARDS.md` is the only record of the deleted
  rewards' mathematics — the implementing code is deleted too).
- Default-path SFT changed on `edit-rewards` (fingerprint v1->v4, bigram artifact
  no longer produced, wandb identity, adapter reuse) => cross-branch SFT runs are
  not comparable.
- All 17 stored `eval_final.json` contain `bigram_log_prob_mean`, but
  `edit-rewards` no longer produces the bigram artifact.

**MEDIUM**
- Cluster: eval 2->5 and diagnose 1->12 serial `apptainer exec` startups; loss of
  bare-python fallback (jobs now hard-fail where they used to proceed).
- `ablation-summary` uses the only remaining `apptainer run` + python, under
  `--cleanenv` with no `--env` passthrough (drops all offline vars).
- Pre-existing (BOTH branches): gloss vocab is built from the full train split
  *before* the SFT holdout is carved — a vocabulary-level leakage channel
  affecting SFT early stopping only, not the test split.

**LOW**
- Tautological test (`test_optimizer_sees_every_trainable_parameter_once`).
- Vocab cache gate hashes only `{seed, train_size}`.
- Sparse `log_partition` lacks the dense path's all-`-inf` guard.
- Dataset encoding noise: 147/87710 rows (0.168%) non-ASCII; 72 suspicious vocab
  tokens. Real, upstream, too small to affect conclusions. Leave as-is.

---

## 4. Scientific protocol reconstruction

Verified from `experiments/results/*/eval_final.json` + resolved YAML.

- Methods: `base` (eval-only), `SFT`, `GRPO`, `SFT-GRPO`.
- Prompt modes: zero-shot, retrieval few-shot (k=3, `max_prompt_length` 768).
- Decoding: unconstrained, **Trie (primary)**, PDA (ablation).
- **All historical GRPO runs were few-shot + curriculum ON.** There was no
  historical zero-shot GRPO cell. This is the single most important protocol fact
  and it is what makes the mission's headline comparison invalid.
- Historical RL objective: **DAPO + `scale_rewards='group'` +
  `mask_truncated_completions=False`** (TRL 0.24.0 defaults; verified by
  introspection — no OLD yaml or code set these).
- Eval: test split, 2000 prompts of 8109, sampled T=0.7 N=5, `metrics_version 2`,
  oracle best-of-N reported separately.
- Matched across old/new GRPO: model, tokenizer, dataset+split+seed 42, lr 3e-6,
  cosine + warmup 200, batch 1, accum 8, G=8, T=0.7, beta 0.04,
  `max_completion_length` 128, 2000 steps (~0.22 epochs), LoRA 32/64/0.05.

---

## 5. Reward inventory

OLD production set (`configs/t2g/base.yaml:102-116`), weights summing to 1.0:
translation .20, bleu .20, gold_structure .20, gloss_order .10,
verifier_scaled .10, format .10, repetition .10.
OLD ablation-only: `structural_dense`, `viterbi_distance`, `soft_viterbi_distance`.

NEW: `edit-validity` (default), `rouge-l`, `sbleu2`, `token-f1`, `chrf++` —
all validity-gated, one at a time, weight hard-coded 1.0. `weight_edit_validity`
**does not exist** (the mission brief's reference to it is incorrect).

Measured reward telemetry from the stored GRPO-only run:
`format 0.998`, `repetition 0.966`, `gloss_order 0.085`, `verifier -0.291`,
`bleu -0.359`.

Two consequences:
- format and repetition were **saturated near 1.0**, i.e. already inert as
  gradient signal. This is evidence *against* restoring them as-is.
- `gloss_order` (the direct ancestor of edit-validity) sat at only 0.085 — the
  edit signal was weak even historically.

The structural family (`structural_dense` / hard / soft Viterbi) was proven by a
prior audit to **algebraically collapse under equal-length completions** (the
all-rewards cell triple-counted one signal) and was empirically null (all cells
within ±0.006 of the 0.5114 control). **Recommendation: keep them removed as
training objectives; recover their corrected diagnostic APIs instead.**

---

## 6. Configuration inventory

Lost experimental cells with no successor: `structure`, `viterbi`,
`soft-viterbi`, `all-rewards`, `sft-grpo-no-grammar` (constrained-decoding-OFF
GRPO), `zero-shot` (the unconstrained lower bound), and **curriculum entirely**.

Of these, the scientifically valuable losses are **no-grammar** and the
**unconstrained zero-shot lower bound** — because §9 shows the Trie is the
dominant factor. The Viterbi cells are correctly retired.

---

## 7. Cluster infrastructure assessment

**The premise is partly refuted.** Verified:
- The rich Python monitor is **byte-identical** on both branches (1741 lines;
  2-line comment diff). `monitor --all` works on both.
- The warning-spam/polling shell monitor existed only in an intermediate commit
  and was **already reverted** by `1d78432`. It is absent from `edit-rewards`
  HEAD. Your cluster checkout is therefore most likely **stale**.
- `chain_failed` has **no writer in either branch** — dead state in OLD too, so
  not a regression.

`edit-rewards` fixed genuine OLD defects: `apptainer run`->`exec` for python
(the mission's own rule says `run` corrupts argv), **unpinned `transformers`**
(OLD could resolve a stack different from the verified one), compute-node
downloads, and a reversed chain resume order (`[eval, train]` -> `[train, eval]`).

Real remaining regressions: serial container startups (eval 2->5, diagnose 1->12),
loss of bare-python fallback, and the `ablation-summary` verb/env bug.

Verdict: **KEEP NEW + targeted de-serialization**, not "recover from OLD".

---

## 8. Structured-loss assessment

Verified CORRECT: graph built strictly post-split from train rows only;
train-only top-512+OTHER state space; genuinely sparse (no `[B,T,V]`/`[B,T,V,V]`
in the trainer path); FP32 logsumexp with max subtraction; **single forward** via
a forward-pre-hook on the output embeddings (no second forward) with an
exactly-once guard; no `.detach()` on the gradient path; `lambda=0` reproduces
stock loss; structured head restored `strict=True` with hard manifest-mismatch
failures at build, artifact load and checkpoint load.

Verified UNSAFE / incorrect:
- **Default-path SFT is not identical to `improvement`** (§3 HIGH). This alone
  justifies Phase B remaining UNACCEPTED.
- `test_optimizer_sees_every_trainable_parameter_once` is tautological: it builds
  the optimizer from `parameters()` itself and never touches TRL's real
  `create_optimizer`. Mission "Test 1" is green but proves nothing.
- Mission "Test 2" (`test_curriculum_filtered_dataset`) is more stable than
  before but weak: it asserts `stage3_hard > stage2_hard` against hard-coded
  schedule constants and would pass if within-stage difficulty selection broke.
- Structured `final/` is not a loadable adapter (no `adapter_config.json`).

---

## 9. `edit-validity` assessment — the central finding

**No mathematical defect.** `R = 2*sim - 1`, `sim = 1 - lev/max(len)`, casefolded,
always finite, `[-1,1]`, invalid gate at `-1`.

I ran an offline discrimination study using a verbatim reimplementation
(self-validated against `edit-rewards`' own test expectations) on stored rollouts:

| base rollout regime | mean R | frac at -1 | frac_zero_std | all-invalid groups |
|---|---|---|---|---|
| **zero-shot + Trie** (the new run's regime) | -0.8034 | 0.5387 | **0.2145** | **0.1990** |
| zero-shot, no grammar | -0.9579 | 0.9054 | 0.7460 | 0.7390 |
| **few-shot + Trie** (GRPO-only's start) | -0.3233 | 0.0982 | **0.0780** | 0.0085 |

**Retraction:** I initially claimed edit-validity rewards verbosity (from the
`R = 1-2k/(G+k)` algebra giving +0.333 at 1.5x length). A simulated padding
attack refuted this: edit-validity penalizes ~**-0.13 per appended token**,
monotonically. What matters for a policy gradient is within-group ranking, and
that ranking opposes length. The claim is withdrawn.

**The comparison in the mission brief is invalid.** The new run is zero-shot;
the 0.608 run is few-shot.

All 27 stored `generations_*.json` were re-scored under **v3** metrics (R3).
Validation: v3 reproduces the v2 stored headline **exactly** for all six cells
(and GRPO-only's `gloss_f1_micro` 0.61468 -> 0.6147), so the harness is sound and
the version confound is eliminated, not assumed away.

| cell | v3 ROUGE-L | v3 validity | v3 EM | v3 glossF1 | v3 absLenErr |
|---|---|---|---|---|---|
| SFT-only | 0.9749 | 0.9998 | 0.8117 | 0.9760 | 0.09 |
| GRPO-only (**few-shot**) | 0.6076 | 0.9915 | 0.0528 | 0.6147 | 2.81 |
| SFT-GRPO (few-shot) | 0.5114 | 0.9880 | 0.0203 | 0.4702 | 3.59 |
| few-shot base | 0.4664 | 0.9676 | 0.0148 | 0.4026 | 5.12 |
| **zero-shot + Trie base** | **0.1373** | **0.9006** | 0.0017 | **0.1186** | **11.24** |
| zero-shot no-grammar base | 0.3648 | 0.1043 | 0.0012 | 0.2279 | 6.98 |
| **NEW edit-validity GRPO** | **0.1893** | **0.8925** | 0.0065 | **0.1537** | **14.54** |

Context-matched against zero-shot + Trie, the new run **improves 4 of 5 metrics**:

| metric | baseline | new run | delta |
|---|---|---|---|
| ROUGE-L | 0.1373 | 0.1893 | **+0.052** |
| gloss F1 micro | 0.1186 | 0.1537 | **+0.035** |
| exact match | 0.0017 | 0.0065 | **+0.0048 (~4x)** |
| validity | 0.9006 | 0.8925 | -0.008 (flat) |
| abs length error | 11.24 | 14.54 | +3.30 (worse) |

**Two premises of the mission brief are refuted:**

1. *"Validity fell 99.2% -> 89.25%"* — false comparison. 99.2% is a **few-shot**
   number; the context-matched zero-shot+Trie baseline is **90.06%**, effectively
   identical to 89.25%. **There is no validity collapse.**
2. *"ROUGE-L collapsed 0.608 -> 0.189"* — false comparison. Context-matched,
   ROUGE-L **rose** 0.137 -> 0.189.

Only length dispersion genuinely worsened, and it starts from an already-poor
11.24 in the *untrained* baseline (vs 2.81 few-shot) — consistent with
heavy-tailed runaway generation being a pre-existing property of zero-shot+Trie.
Mean reward also moved -0.803 -> -0.673, independently indicating real learning.
Stated conservatively: *consistent with a small real improvement; not yet fully
protocol-matched* (greedy-vs-sampled remains open).

Supporting verifications that make this comparison usable:
- `rouge_l_mean` is **definitionally unchanged** v2 vs v3 (both plain means over
  all completions) — so the numbers are comparable.
- The Trie allowed set is **bit-identical** across branches (3995 tokens, tested
  with the real Qwen tokenizer).
- The validity-definition change costs at most **0.45 pp** on real completions,
  so the 99.2% -> 89.25% validity drop is **real degradation**, not an artifact.
- Sampled-N=5 vs single-draw differs by **0.0016** on a trained policy, so that
  protocol change is negligible for ROUGE-L mean.

**The dominant factor is the Trie at initialization, not the reward.**
Zero-shot no-grammar base scores 0.365; adding the Trie drops it to 0.137 and
raises repetition errors from 32 to 945. The Trie costs an untrained model
**0.23 ROUGE-L** and induces degenerate loops. Yet the Trie is also what makes
the validity gate satisfiable at all (74% dead groups without it vs 20% with it).
So the Trie simultaneously **hurts raw quality and enables the reward** — a
genuine neuro-symbolic result, and a direct answer to the project's actual
research question.

Unexplained: length ratio worsened 1.139 -> 1.508 despite a length-penalizing
reward. Leading hypothesis (Oracle): a **greedy-decoding artifact**, since the
0.137 baseline is sampled T=0.7 while the 0.189 headline is greedy N=1, and
greedy decoding of a weak Trie-constrained policy is the canonical recipe for
loops. Resolvable with the run's sampled leg. **UNPROVEN.**

---

## 9b. Dead-code / necessity audit (mission §25)

Verified by execution, not inspection.

- **`grammarllm` must be KEPT.** The PDA *branch* is flag-gated off by default
  (`use_grammarllm_pda: false`), but the *package* is an **import-time hard
  dependency of the default Trie path**: blocking the module makes both
  `src.grammar.grammar_logits_processor` and `src.grammar.gloss_grammar` raise
  `ModuleNotFoundError`. Removing it requires first converting 4 top-level
  imports to lazy/optional. Additionally `grammarllm/VENDORED_STATUS.md` records
  that the vendored copy is **ahead of upstream** (backported EOS-before-PAD fix
  plus local-only EOS bound-check and BUG-13/4/19 fixes) — swapping in the
  upstream package would lose them. The PDA cell also **never ran** (no `*pda*`
  results dir; `docs/FINDINGS.md:214` marks it queued), so it has no
  stored-result dependency, but it is a documented future Trie-vs-PDA experiment
  and is tested.
- **ViterbiPlanNet is CC BY-NC 4.0** (verified in `reference/ViterbiPlanNet/LICENSE`)
  while this repo is MIT. **No code was transplanted** — the implementations
  differ structurally (reference: HMM Viterbi with emissions / batched
  differentiable torch DVL; ours: log-space max-plus over a pure Markov chain /
  numpy non-differentiable forward-backward). No violation exists today, but the
  read-only boundary on `reference/` must be preserved to keep it that way.
- **CRITICAL reproducibility dependency — the bigram path.** All 17 stored
  `eval_final.json` carry `bigram_log_prob_mean`/`_std`, produced by eval loading
  `data/bigram_transition.npy` and scoring via `sequence_score_bigram`. OLD
  produces that artifact automatically during training and **hard-fails** without
  it; NEW has **no training producer** (`sft_train.py:747`) and silently omits the
  metrics. `transition_matrix.py` shrank 914 → 95 lines on `edit-rewards`.
  **Removing `sequence_score_bigram` or its producer makes those stored columns
  unreproducible.** Strongest "do not delete" finding in the audit.
- **Viterbi rewards: demote as objectives, retain as code.** Stored dirs exist for
  `sft-grpo-viterbi` and `sft-grpo-soft-viterbi`, so the machinery is needed to
  re-score those cells even though the rewards are scientifically redundant
  (equal-length collapse). This resolves the §25 tension: retire the *cells*,
  keep the *code*.
- **Genuinely dead (safe to remove, zero call sites — verified):**
  `compute_viterbi_path`, `viterbi_optimal_score`, `compute_trigram_transitions`,
  `sequence_score_trigram`, `soft_viterbi_marginals`
  (`src/datasets/transition_matrix.py:337,432,130,301,868`). Every reference is a
  `def` line or docstring mention. This is the *only* safely deletable code found.

## 9c. The two proposed neuro-symbolic objectives

Full spec: `docs/NEW_OBJECTIVES_SPEC.md`. Both were qualified before spending
GPU time; measurements are offline, on stored artifacts and the dataset cache.

### Idea 1 — the mass removed by constrained decoding

**As a differentiable auxiliary loss (1a): PURSUE AS DIAGNOSTIC.**
The objective `-log P(allowed set)` is not new — it is the candidate-set
log-marginal known as **partial-label learning** (Cour et al., JMLR 2011;
Seo & Huh, arXiv:2010.11600) and **maximum marginal likelihood** (Guu et al.,
arXiv:1704.07926); the token-level variant is **unlikelihood training**
(Welleck et al., arXiv:1908.04319). Present it as an application.
Its motivation is sound: **Grammar-Aligned Decoding** (Park et al.,
arXiv:2405.21047, NeurIPS 2024) proves hard masking produces grammatical output
whose likelihoods are not proportional to the model's own, and internalizing the
mass shrinks that gap.
The decisive limitation, from both the literature and our measurements: MML
maximizes the mass *of the set* without indicating which allowed token is
correct, so it **cannot correct ranking inside the allowed set**. Expect
calibration gains, not exact-match gains.
Correct form: teacher-forced on gold prefixes, `lambda ≈ 0.1`, with
`lambda = 0` reproducing stock SFT bit-for-bit.
Gate: one inference-only cluster probe of `removed_mass` mean/variance under the
frozen SFT policy, reusing the existing `masked_mass_tracker`. If the mass is
already tiny, drop it.

**As a GRPO reward (1b): DROP.**
The decisive argument is structural, not statistical: allowed mass is maximized
by repeating the most probable in-vocabulary token, entirely decoupled from
translation correctness — the Goodhart configuration (Amodei et al.,
arXiv:1606.06565; Gao et al., arXiv:2210.10760; Catastrophic Goodhart,
arXiv:2407.14503, which shows KL regularization does not rescue a misspecified
reward). Self-referential confidence/entropy rewards are documented as
unreliable for long training (PRISM, arXiv:2601.04700), and "confidence reward
hacking" is a named failure mode (arXiv:2607.04332).

A secondary observation — the *output* in-vocabulary fraction is 1.0000 with std
0.0009, so it carries no within-group variance — is **not** load-bearing and is
recorded only for completeness: that quantity is near-tautological under the
Trie, and the raw (unmasked) removed mass could still vary across rollouts. It
was never measured, because `track_diagnostics` defaults to `False`
(`grammar_logits_processor.py:72`) and no training or evaluation script enables
it, so **no historical run recorded this quantity at all**.

Keep the mass as telemetry, which is its current role.

### Idea 2 — source-conditioned sparse structured gloss loss

**DROP as a primary objective on ASLG-PC12; keep as a validated opt-in module.**

Information qualification (held-out, 21917 aligned tokens, proper hierarchical
backoff `(prev,src) → src → unigram`):

| model | held-out cross-entropy |
|---|---|
| `H(gloss)` | 8.578 bits |
| `H(gloss \| source)` | 0.856 bits |
| `H(gloss \| source, prev_gloss)` | 0.812 bits |
| **incremental value of `prev_gloss`** | **+0.044 bits (+5.2% of residual)** |

The source word alone removes ~90% of the uncertainty; a first-order structural
prior can address at most ~5% of what remains, and an autoregressive decoder
already models `gloss_{t-1}` through its own context.

The literature agrees independently: global normalization is theoretically and
practically ≈ local normalization with strong encoders (Goyal, Dyer &
Berg-Kirkpatrick, arXiv:1904.06834), and structured normalization helps on
*small* datasets but not large ones (Huang et al., arXiv:2106.03376) — we have
73K rows and are at the published ceiling. Two further design objections:
top-512 + `OTHER` compresses the 16K-type tail into one state, removing exactly
the rare-gloss gradient the closed vocabulary needs (unlike sampled softmax/NCE,
this is not an unbiased approximation); and exact-length-conditioned partition is
not a standard CRF formulation (linear-chain CRFs are indexed by input length),
which abandons length modelling and comparability with the LM's EOS. No prior
work applies CRF/structured losses to gloss generation (NOT FOUND).

Revisit on PHOENIX-2014T, where gloss/text token overlap is 15.6% rather than
98.2% and the structural residual genuinely exists.

## 10. Controlled experiment plan

Ordered; each step gated on the previous.

1. **Offline, no cluster.** Re-score all 27 stored `generations_*.json` under
   v3 metrics; report paired deltas bootstrapped clustered by prompt. Restores
   full comparability of the historical table at zero GPU cost.
2. **Offline.** Rollout-support probe already run (§9). Extend to the SFT
   checkpoint's rollouts to quantify saturation for edit-validity.
3. **Primary experiment — prompting x constrained decoding at initialization.**
   This is where the measured effects are large (0.33 of the 0.42 headline gap is
   pre-RL). Cells: {zero-shot, few-shot} x {Trie, no-Trie}, base init, eval only
   + short GRPO. Isolates the factor that actually dominates.
4. **Reward experiment — second, not first.** 7-component vs edit-validity, on a
   substrate *with* rollout support (few-shot base or SFT init). A reward
   comparison on zero-shot+Trie is uninterpretable because ~20% of groups are
   dead.
5. **Only then** revisit Dr-GRPO vs DAPO and curriculum as separate one-factor
   arms.

Do not launch any cluster job before steps 1-2 and the §11 recoveries land.

---

## 11. Recovery plan (what to take from `edit-rewards`)

**RECOVER (low risk, high value)**
- weave shim (**done**, §12).
- Viterbi diagnostic APIs + exhaustive brute-force tests.
- `apptainer exec` + `--env` offline passthrough; centralized `export_offline_env`.
- Dependency pinning to the verified stack + post-install verification.
- `require_cluster_artifacts` verify-only; no compute-node downloads.
- Chain resume order fix; `remote/app.py` dict guards; `cluster_helper` squeue 3->1.
- `valid_rouge_l_mean` corrected definition (with a metrics-version note).

**KEEP `improvement`'s version**
- The 10 docs and historical `FINDINGS.md` (already present — do not delete).
- `experiments/results/` (already present — archived in §12).
- The 7-component reward *code* (retain for the §10.4 comparison even though it
  is not the default).

**REJECT / do not adopt wholesale**
- Deletion of the reward code and docs.
- The fresh-start config tree as a drop-in replacement (it makes OLD configs
  unrunnable). Adopt its *identity* concept without breaking OLD config loading.

**INVESTIGATE before adopting**
- Phase B structured SFT: sound mathematically, but must not change default-path
  SFT behaviour. Needs the default path restored to `improvement`-equivalent.
- Dr-GRPO/`scale_rewards=none`/`mask_truncated=True` as a deliberate one-factor
  arm, not a silent default change.

**Pending** `exp-5` (grammarllm / ViterbiPlanNet necessity + bigram producer).

---

## 12. Implementation plan and progress

| # | Change | Test gate | Status |
|---|---|---|---|
| R1 | 3-flag TRL weave shim | full suite + ruff | **DONE** — 260 passed / 0 failed (was 7 failed); ruff clean |
| R2 | Archive `experiments/results/` with hashes | manifest verifies | **DONE** — 13.7 MB zip + 71-file SHA-256 manifest |
| R3 | Re-score stored generations under v3 | reproduce stored v2 values first | **DONE** — v3 reproduces all six v2 headlines exactly (§9) |
| R4 | Recover Viterbi diagnostic APIs + tests | focused tests | **DONE** — `src/analysis/markov_diagnostics.py` + 32 brute-force tests pass; suite 292 passed / 0 failed |
| R6 | Provenance index for historical docs | none | **DONE** — `docs/HISTORICAL_RECORD.md` |
| R5 | Cluster: de-serialize eval/diagnose, fix `ablation-summary` verb/env | bash syntax + dry-run | **BLOCKED** — needs deployed commit hash |
| R7 | Primary prompting x decoding experiment | configs validate | **BLOCKED** — needs the 0.189 run artifacts |

R4 detail: extracted the 4 public diagnostics (`bigram_sequence_mean`,
`path_log_energy`, `hard_viterbi_diagnostic`, `soft_viterbi_diagnostic`) plus 6
private helpers **verbatim** from `edit-rewards`, into a standalone
`src/analysis/markov_diagnostics.py` (192 lines, numpy-only). Deliberately
excluded the probe runner and its CLI/file plumbing, so this does **not** pull in
`edit-rewards`' rewritten `transition_matrix.py` or its path/config system — no
comparability risk. The recovered module keeps the small-state guard
(`DEFAULT_MAX_STATES = 256`), so the corrected DP can never be run over the full
~16K gloss vocabulary. One extraction defect found and fixed by lint (missing
`Iterable` import); ruff check and format clean.

Rationale for recovering these while *not* restoring the Viterbi rewards: the
corrected dynamic program is genuinely better science (validated against
brute-force path enumeration), whereas the rewards built on it were proven
redundant. Diagnostics in, objectives out.

R1 detail: `src/training/grpo_t2g_train.py:55-70`, +13/-4. `improvement` shimmed
only `_mergekit_available` and `_llm_blender_available`, missing
`_weave_available`. transformers 5.3.0 returns `(bool, version)` from
`_is_package_available`; TRL 0.24.0 assigns it directly; `bool((False, None))` is
truthy; `trl/trainer/callbacks.py:57` then runs `import weave` and
`trl.trainer.grpo_trainer` becomes unimportable. Same pins run on the cluster, so
this may have been affecting live GRPO jobs.

---

## Blocking questions for the user

1. Cluster deployed commit hash (`git -C ~/neuro_symbolic_t2g rev-parse HEAD`).
2. The 0.189 run's `generations_*.json` (**both** greedy and sampled legs) and
   training curves. Without these, all causal statements about that run —
   including mine — are inference from one pasted table.
3. Was `probes/rollouts.yaml` run before that training? A prior Oracle gate
   required it.
4. Off-machine backup of `experiments/results/`.
