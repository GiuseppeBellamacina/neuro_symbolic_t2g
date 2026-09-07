# Historical record — provenance index

Purpose: some documents in `docs/` describe systems that no longer match the
active code. They are retained deliberately, because they are the only surviving
record of decisions, mathematics and failure diagnoses the code itself no longer
contains — and every number in `FINDINGS.md` was produced by that code.

Do not "modernize" them into agreement with current code: that destroys the
record. Read them as dated artifacts.

## Documents removed in the legacy cleanup

Six documents were **deleted** because the code they documented no longer exists
in the repository at all, so they described nothing and could only mislead:

| Removed document | What it documented |
|---|---|
| `DOCUMENTAZIONE.md` | Full API docs of the vendored `grammarllm` package |
| `GRAMMARLLM_MIGRAZIONE.md` | `grammarllm` v0.4.x -> v0.5.0 migration record |
| `GRAMMARLLM_CONFRONTO.md` | Vendored-vs-upstream `grammarllm` comparison |
| `ERRORI_E_MIGLIORIE.md` | Upstream `grammarllm` bug catalogue |
| `CONFIGS.md` | Index of the deleted `configs/t2g/*.yaml` cells |
| `CONFIGS_GUIDE.md` | `extends` chain of the deleted `configs/t2g/` tree |

The vendored `grammarllm/` package and the `configs/t2g/` tree were removed
because the PDA decoding path never ran (no `*pda*` result directory exists), it
failed with an LL(1) conflict on digit-initial glosses, and a PDA is oversized
for a flat `gloss*` language. The Trie is the only constrained-decoding path and
its coverage is effectively complete (4 gloss types out of 15472 blocked).

Git history retains all six at commit `741cfc6^`, so nothing is lost — the
rationale for the removals lives in `docs/RECOVERY_REPORT.md` §9b.

## Status legend

- **HISTORICAL — code deleted**: describes code that no longer exists.
  Irreplaceable: the doc *is* the specification.
- **HISTORICAL — superseded**: the subject still exists but the doc's numbers or
  API descriptions are stale.
- **CURRENT**: still accurate for `improvement`.

| Document | Status | Why it must be kept |
|---|---|---|
| `REWARDS.md` | HISTORICAL — code deleted | The only record of the 10 historical reward functions: formulas, ranges, OOV/length guards, and the v2 gold-anchored recalibration rationale. `edit-rewards` deleted both this doc **and** the implementing code, so nothing else describes them. |
| `T2G_PIPELINE_REVIEW.md` | HISTORICAL — superseded | Root-cause forensics for three real training bugs: the unsloth `position_ids` RoPE broadcast crash, the `grpo_accumulated_loss` Half/Float dtype bug, and the `prompt_len`-not-reset bug that produced garbage output. Survives elsewhere only as a one-line code comment. |
| `METRICS.md` | HISTORICAL — code deleted | The only definition of the masked-mass / masked-entropy grammar diagnostics. `grammar.track_diagnostics` still emits these, so removing this doc leaves an undocumented live metric. |
| `RESEARCH_REPORT.md` | HISTORICAL — superseded | Project overview and the symbolic-reward design rationale. Partially survives via `docs/report/chapters/*.tex`. |
| `FINDINGS.md` | **CURRENT on `improvement`** | Holds the historical results table with run IDs. On `edit-rewards` this was **rewritten** down to an 8-line disclaimer, so the numeric record survives only here. |

## Companion artifacts

- `experiments/results/` — 18 run dirs, 27 `generations_*.json`, 115 MB. **Not
  tracked by git** (`.gitignore:36`). This is the primary evidence for every
  number in `FINDINGS.md`, and it is recomputable: re-scoring the stored
  completions reproduces the published headlines exactly.
- `experiments/results-archive/` — local hashed snapshot (zip + per-file
  SHA-256 manifest) created during the recovery audit. **A same-disk snapshot is
  not a backup**; an off-machine copy is still required.
- `docs/RECOVERY_REPORT.md` — the audit that established the above, including
  which historical rewards are scientifically redundant versus which are
  required to reproduce stored results.

## Important caveat on the reward documents

`REWARDS.md` must be read together with §5 and §9b of `docs/RECOVERY_REPORT.md`.
A later audit proved that the structural / hard-Viterbi / soft-Viterbi reward
family **collapses algebraically for equal-length completions** (so the
`all-rewards` cell triple-counted one signal), and that those cells were
empirically null. Measured telemetry also shows the format and repetition
rewards were saturated near 1.0, i.e. already inert as gradient signal.

So `REWARDS.md` is an accurate record of what was *implemented and run*, not an
endorsement of that design. The reward *code* is retained because stored
`sft-grpo-viterbi` and `sft-grpo-soft-viterbi` results cannot otherwise be
re-scored — not because the objectives should be revived.
