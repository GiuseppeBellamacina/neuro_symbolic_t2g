# Design spec — two proposed neuro-symbolic objectives

Scope: evaluate and specify (1) using the probability mass removed by
constrained decoding as a training signal, and (2) a source-conditioned sparse
structured gloss loss. Both were requested by the user.

Status: **each is gated on a qualification test.** One of the two gates is
already answered by measurement (Idea 2); the other needs a single
inference-only cluster probe (Idea 1a) and is recommended against in its RL form
(Idea 1b). No training has been run for either.

All measurements below were produced offline on stored artifacts and the HF
dataset cache. No local training or GPU work was performed (the training system
is the Linux/SLURM cluster).

---

## Shared context (measured, see `docs/RECOVERY_REPORT.md`)

- The corpus is a rule-derivable transduction: a word-level rule reaches
  ROUGE-L 0.9697 / EM 0.5912 / non-copy-token accuracy 0.9428 on the full
  official test (8109 rows).
- `H(gloss) = 8.578` bits; `H(gloss | source word) = 0.856` bits held-out.
  **The source word alone removes ~90% of the uncertainty.**
- Overlap metrics are saturated and defective on this corpus (ROUGE-L is
  case-insensitive and splits on non-alphanumerics, so `DESC-GOOD` vs
  `DESC-BAD` scores 0.5). Primary metrics must be **exact match** and
  **non-copy-token accuracy** (`src/analysis/rule_baseline.py`).
- The Trie leaves ~3967 allowed token ids and blocks the first token of only
  4/15472 gloss types, so constraint coverage is effectively complete.

---

## Idea 1 — internalize the constrained-decoding mass

### 1a. As a differentiable auxiliary loss

Proposed term: maximize `log P(allowed set)` on teacher-forced gold prefixes,
i.e. minimize `-log(sum of allowed-token probability)`, added to the SFT NLL
with weight `lambda` and a warmup.

**This objective already has a name, and it is not new.** It is the
candidate-set log-marginal used in two established literatures:

- **partial-label learning** (Cour et al., JMLR 2011; Seo & Huh,
  arXiv:2010.11600 — the "naive" `-log Σ_{c∈S} P(c)` objective already performs
  well), and
- **maximum marginal likelihood** for weakly-supervised semantic parsing
  (Guu et al., arXiv:1704.07926), where the allowed set plays the role of the
  set of consistent derivations.

The token-level variant that pushes disallowed tokens down instead is
**unlikelihood training** (Welleck et al., arXiv:1908.04319). Present this as an
application of MML/PLL, not as an invention.

The strongest *motivating* result is **Grammar-Aligned Decoding** (Park et al.,
arXiv:2405.21047, NeurIPS 2024), which proves hard masking yields outputs that
are grammatical but whose likelihoods are *not* proportional to the model's own —
the "renormalization gap". Internalizing the mass shrinks that gap, which is
consistent with our measured 70x dependence on the mask.

**The logical difficulty, stated up front.** The Trie already guarantees
in-vocabulary output at decode time, with complete coverage. So this term cannot
buy validity — validity is already 1.0 by construction. Any benefit must come
from one of:

- *distribution sharpening*: concentrating raw probability inside the allowed
  set, which reduces the mismatch between the trained distribution and the
  masked distribution actually used at inference (the "projection tax",
  Reddy et al., arXiv:2603.03305);
- *reduced mask dependence*: better behaviour if the constraint is ever relaxed.

Both are plausible; neither is established for this setting. Note the term is a
pure regularizer on gold prefixes, because the gold token is in the allowed set
by construction — it cannot change the argmax ordering *among* allowed tokens,
only the mass allocated outside. That is a real but narrow effect.

**Measured headroom on the validity axis: none.**

| run | mean in-vocab token fraction | std | completions already 100% in-vocab |
|---|---|---|---|
| GRPO-only | 1.0000 | 0.0009 | 99.82% |
| zero-shot + Trie | 0.9996 | 0.0033 | 98.69% |

Correlation between in-vocab fraction and ROUGE-L is +0.07 to +0.10, i.e.
essentially none.

**Honest limitation.** The table measures the *output* token fraction, which the
Trie forces and is therefore near-tautological. The quantity Idea 1a actually
targets is the *raw* (unmasked) probability mass falling outside the allowed set
at each step. That requires a forward pass of the real model and is **not
measurable offline**.

**GATE 1a (must pass before any implementation).** One inference-only cluster
job, reusing the telemetry that already exists in
`src/grammar/masked_mass_tracker.py`. Record, under the frozen SFT policy on
train prompts: mean and standard deviation of `removed_mass`, `log_allowed_mass`,
and `allowed_entropy` vs `raw_entropy`.

- If mean `removed_mass` is small (say <0.05) and its variance is low, the term
  has nothing to optimize → **DROP**.
- If `removed_mass` is substantial, implement as an opt-in auxiliary loss with
  `lambda = 0` default, and require that `lambda = 0` reproduce stock SFT
  bit-for-bit before any comparison.

Cost: one job, no training, no gradient steps.

### 1b. As a GRPO reward — **RECOMMENDED AGAINST**

Two independent, sufficient reasons:

1. **No signal.** The quantity is near-constant across rollouts (std 0.0009), so
   within-group variance is ~0. GRPO advantages are group-relative, so a
   constant reward yields exactly zero gradient — the zero-variance-group
   pathology already documented for this project (and the motivation for DAPO's
   dynamic sampling, which TRL 0.24.0 does **not** implement; it only logs
   `frac_reward_zero_std`).
2. **Structural reward hacking.** Allowed mass is maximized by emitting the most
   probable in-vocabulary token repeatedly, which is entirely decoupled from
   translation correctness. This is a self-referential likelihood/fluency reward,
   the classic Goodhart configuration (Gao et al., arXiv:2210.10760).

Keep the mass as **diagnostics only**, which is its current status.

---

## Idea 2 — source-conditioned sparse structured gloss loss

An existing prototype (on `edit-rewards`) implements a 513-state graph
(top-512 frequent glosses + `OTHER`), transitions estimated from post-split
train rows only, a structured head over the final pre-completion hidden state,
and an exact-length sparse CRF log-partition via `scatter_add` in FP32. It is
mathematically validated against brute-force enumeration, leakage-audited, and
never trained.

### Qualification test (run, held-out, cost zero)

A first-order structural prior can only help if `gloss_{t-1}` carries
information about `gloss_t` **beyond** the aligned source word. Measured on a
90/10 train-internal split (seed 7), 21917 held-out aligned tokens, with proper
hierarchical backoff `(prev, src) -> src -> unigram`:

| model | held-out cross-entropy |
|---|---|
| `H(gloss \| source)` | 0.8560 bits |
| `H(gloss \| source, prev_gloss)` | 0.8116 bits |
| **incremental value of `prev_gloss`** | **+0.0444 bits (+5.19%)** |

Two earlier versions of this measurement were wrong and were corrected: an
in-sample plug-in MLE estimate overstated the gain (0.1534 bits) because
conditional entropy always decreases in-sample, and a first held-out attempt
understated it (−2.58 bits) because the `(prev, src)` model backed off to the
unigram instead of to the source-only distribution. Only the table above is
valid.

**Population caveat — the number is measured on the easy subset.** The
estimator uses position-wise alignment, so it can only consume pairs where
`len(text) == len(gloss)`: **27.5% of the corpus**. That is the same restriction
`rule_baseline.fit` uses. But the structural residual most likely to matter —
contextual copula (`BE`) deletion, which accounts for **65%** of SFT's exact-match
advantage over the rule — lives precisely in the *length-mismatched* pairs this
estimator excludes. So `+0.044 bits / 5.2%` is a lower bound **on the aligned
subpopulation**, not a ceiling over the corpus; insertion/deletion structure is
not measured here.

The DROP verdict does not rest on this number alone. It rests on it *plus* two
independent arguments: an autoregressive decoder already conditions on
`gloss_{t-1}` through its own context, and the literature finds global
normalization ≈ local normalization with strong encoders (Goyal, Dyer &
Berg-Kirkpatrick, arXiv:1904.06834) with structured gains concentrated on small
datasets (Huang et al., arXiv:2106.03376). Quantifying the unaligned residual
would require an alignment model (IBM-1 or similar) and is the correct follow-up
if anyone wants to reopen the question.

### Verdict

The theoretical ceiling is **~5% of the residual uncertainty**, and a
first-order CRF sees nothing the autoregressive decoder does not already model
through its own context. Expected value on this corpus is therefore **low**.
This is not a defect of the implementation — the implementation is correct — it
is a property of a near-deterministic, near-monotone corpus.

**Recommendation: keep as a validated, opt-in research module; do not promote it
to the primary campaign on ASLG-PC12.** It becomes genuinely interesting on a
corpus where gloss is *not* a shallow function of the source (e.g.
PHOENIX-2014T, where published gloss/text token overlap is 15.6% vs 98.2% here),
because there the structural prior has real residual to explain.

**GATE 2 (if ever promoted).** Head-only benchmark with a shuffled-transition
control: if the true graph does not beat the shuffled graph on non-copy-token
accuracy by more than the paired bootstrap CI, the structural prior is not
contributing and the arm is dropped.

---

## Summary

| idea | status | decision |
|---|---|---|
| 1a — mass as auxiliary differentiable loss | headroom unmeasurable offline | **GATE**: one inference-only probe, then decide |
| 1b — mass as GRPO reward | no variance + hackable | **DROP** (keep as diagnostics) |
| 2 — source-conditioned structured loss | +5.2% of residual, corpus-limited | **KEEP opt-in**, do not promote here; revisit on PHOENIX-2014T |

Ordering rationale: both ideas are downstream of a corpus whose overlap metrics
are saturated. Neither can produce a headline result on ASLG-PC12. Spend the
cluster budget on the rollout-support probe and the metric correction first; the
structured objective is scientifically interesting only where the transduction
is not already solved by a lexicon.
