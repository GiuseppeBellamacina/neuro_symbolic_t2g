# Historical record — provenance index

Purpose: several documents in `docs/` describe systems that no longer match the
active code, and were **deleted wholesale** on the `edit-rewards` branch. They
are retained here deliberately. They are the only surviving record of decisions,
mathematics and failure diagnoses that the code itself no longer contains.

Do not delete these. Do not "modernize" them into agreement with current code —
that destroys the record. Read them as dated artifacts.

Verified: every file listed below exists on `improvement` (`a8bb684`) and is
absent from `edit-rewards` (`68bc12a`), with no surviving copy of its content on
that branch (checked against the new `docs/` tree and `docs/report/chapters/*.tex`).

## Status legend

- **HISTORICAL — code deleted**: describes code that no longer exists on
  `edit-rewards`. Irreplaceable: the doc *is* the specification.
- **HISTORICAL — superseded**: the subject still exists but the doc's numbers or
  API descriptions are stale.
- **CURRENT**: still accurate for `improvement`.

| Document | Status | Why it must be kept |
|---|---|---|
| `REWARDS.md` | HISTORICAL — code deleted | The only record of the 10 historical reward functions: formulas, ranges, OOV/length guards, and the v2 gold-anchored recalibration rationale. `edit-rewards` deleted both this doc **and** the implementing code, so nothing else describes them. |
| `T2G_PIPELINE_REVIEW.md` | HISTORICAL — superseded | Root-cause forensics for three real training bugs: the unsloth `position_ids` RoPE broadcast crash, the `grpo_accumulated_loss` Half/Float dtype bug, and the `prompt_len`-not-reset bug that produced garbage output. Survives elsewhere only as a one-line code comment. |
| `METRICS.md` | HISTORICAL — code deleted | The only definition of the masked-mass / masked-entropy grammar diagnostics. `grammar.track_diagnostics` still emits these, so removing this doc leaves an undocumented live metric. |
| `CONFIGS_GUIDE.md` | HISTORICAL — code deleted | The historical config matrix and `extends` chain (base -> sft-grpo -> 9 children), plus the reward-dilution design and the record of pre-v2 configs. Needed to interpret the stored run directories. |
| `CONFIGS.md` | HISTORICAL — code deleted | Index of the 13 `configs/t2g/*.yaml` cells that produced every stored result. |
| `GRAMMARLLM_MIGRAZIONE.md` | HISTORICAL — superseded | Per-file API migration record v0.4.x -> v0.5.0. The code encodes the outcome; only this doc records what changed where, and why. |
| `GRAMMARLLM_CONFRONTO.md` | HISTORICAL — superseded | Vendored-vs-upstream comparison and the decision to keep the vendored copy. Directly supports `grammarllm/VENDORED_STATUS.md`. |
| `ERRORI_E_MIGLIORIE.md` | HISTORICAL — superseded | Upstream grammarllm bug catalogue with file:line and fixes. Explains why the vendored copy is ahead of upstream. |
| `DOCUMENTAZIONE.md` | HISTORICAL — superseded | Full docs of the pre-migration vendored grammarllm. Least-critical (upstream library docs) but not recoverable from this repo. |
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
