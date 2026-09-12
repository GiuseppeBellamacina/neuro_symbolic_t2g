"""Wiring of symbolic rule-repair (``src/analysis/rule_repair.py``) into eval.

``evaluation.rule_repair`` is an opt-in, additive knob: off by default, and
when on it must never touch the primary metrics — only add a separate
``results["rule_repair"]`` block. These tests cover the two pieces that are
actually new here (``_repair_completions`` and its composition with the
already-tested ``_compute_primary_metrics``); the repair logic itself is
covered by ``tests/test_rule_repair.py`` and the metrics block by
``tests/test_eval_resume.py``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.analysis.rule_baseline import RuleBaseline  # noqa: E402
from src.analysis.rule_repair import RuleRepairer  # noqa: E402
from src.training.eval_t2g import (  # noqa: E402
    _compute_primary_metrics,
    _repair_completions,
)

_MINI_VOCAB = ["<BOS>", "<EOS>", "GLOSS", "GOOD", "BAD", "BLINDNESS"]
_MINI_TOKEN_TO_IDX = {t: i for i, t in enumerate(_MINI_VOCAB)}
_MINI_BIGRAM = np.full((len(_MINI_VOCAB), len(_MINI_VOCAB)), -1.0)


def _repairer(morphology: dict[str, str] | None = None) -> RuleRepairer:
    return RuleRepairer(
        baseline=RuleBaseline(lexicon={}, deletions=frozenset()),
        morphology=morphology or {},
    )


# ---------------------------------------------------------------------------
# _repair_completions
# ---------------------------------------------------------------------------


def test_repair_completions_preserves_nesting_per_prompt():
    """Two prompts, two completions each: shape and per-prompt text pairing
    must be preserved — completion i of prompt p is repaired against text p,
    never against another prompt's source."""
    repairer = _repairer({"blindness": "BLINDNESS"})
    # Same token count as the rule transduction so the replace opcode aligns
    # 1:1 (see rule_repair.py's own tests for why length must match here).
    all_completions = [["MEAN BLIGHT", "MEAN BLIGHT"], ["GOOD", "BAD"]]
    all_texts = ["mean blindness", "unrelated text with no hallucination"]

    out = _repair_completions(all_completions, all_texts, repairer)

    assert len(out) == 2
    assert out[0] == ["MEAN BLINDNESS", "MEAN BLINDNESS"]
    # Second prompt's completions aren't touched by the first prompt's text
    # (different length vs. that prompt's rule transduction, so the model's
    # tokens survive as-is): confirms per-prompt pairing, not a global one.
    assert out[1] == ["GOOD", "BAD"]


def test_repair_completions_empty_completion_list_for_a_prompt():
    """A prompt with zero sampled completions must not crash the mapping."""
    repairer = _repairer()
    out = _repair_completions([[], ["GOOD"]], ["text a", "text b"], repairer)
    assert out == [[], ["GOOD"]]


def test_repair_completions_is_a_pure_reshape_no_shared_mutation():
    """Repairing must not mutate the input lists in place (evaluate_checkpoint
    reuses ``all_completions`` afterwards for the generations log)."""
    repairer = _repairer({"blindness": "BLINDNESS"})
    all_completions = [["MEAN BLIGHT"]]
    all_texts = ["mean blindness"]
    original = [list(c) for c in all_completions]

    _repair_completions(all_completions, all_texts, repairer)

    assert all_completions == original


# ---------------------------------------------------------------------------
# Composition with _compute_primary_metrics (the same helper oracle_best_of_n
# and the resume path already exercise — here just fed repaired completions)
# ---------------------------------------------------------------------------


def test_repaired_completions_improve_exact_match_through_the_real_metrics_path():
    """End-to-end at the unit level: fit a tiny repairer, repair a batch of
    completions containing one hallucination each, and confirm the SAME
    ``_compute_primary_metrics`` used for the primary block reports a higher
    exact_match on the repaired batch — this is the property the
    ``rule_repair`` results block exists to surface."""
    repairer = _repairer({"blindness": "BLINDNESS", "house": "HOUSE"})
    all_completions = [["MEAN BLIGHT"], ["THE BLIGHT"]]
    all_texts = ["mean blindness", "the house"]
    all_references = ["MEAN BLINDNESS", "THE HOUSE"]

    def _score(completions: list[list[str]]) -> dict:
        flat = [c for comps in completions for c in comps]
        flat_refs = [r for comps, r in zip(completions, all_references) for _ in comps]
        flat_sources = [t for comps, t in zip(completions, all_texts) for _ in comps]
        results, *_ = _compute_primary_metrics(
            flat,
            flat_refs,
            completions,
            all_references,
            token_to_idx=_MINI_TOKEN_TO_IDX,
            bigram=_MINI_BIGRAM,
            reward_weights={},
            flat_sources=flat_sources,
            n_bootstrap=20,
        )
        return results

    before = _score(all_completions)
    repaired = _repair_completions(all_completions, all_texts, repairer)
    after = _score(repaired)

    assert before["exact_match"] == 0.0
    assert after["exact_match"] == 1.0


# ---------------------------------------------------------------------------
# Config default
# ---------------------------------------------------------------------------


def test_rule_repair_defaults_to_false_in_base_config():
    """The flag must default off: existing eval runs are unaffected unless a
    cell explicitly opts in."""
    base = yaml.safe_load(
        Path("experiments/configs/qwen25-05b/base.yaml").read_text(encoding="utf-8")
    )
    assert base["evaluation"]["rule_repair"] is False
