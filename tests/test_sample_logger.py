"""Regression: CompletionSampleLogger breakdown passes gold to gold-anchored components.

Bug (found on the real all-rewards run 20260903): the per-sample breakdown
in CompletionSampleLogger._capture stored ``{"normalize": True}`` (no
"gold_gloss" key) for the gold-anchored structural components,
so the gold substitution `if "gold_gloss" in kwargs_call` never fired →
components called without gold → returned neutral 0.0 → the sample
display showed "+0.00" for PERFECT completions (while the trainer metrics
were correctly ~0.87 — the training signal was intact, display-only bug).

Same class of bug, found again on the real ``ablations/rewards/edit-validity``
run 20260912 (job 7374's training log): ``edit_validity_reward`` was never
added to ``_component_fns``/``_REWARD_COMPONENTS`` when it was introduced, so
every printed sample showed ``REWARDS: `` (empty) and ``TOTAL: +0.0000`` for
every completion — including clearly wrong ones — while the trainer's own
logged metric (``rewards/_edit_validity/mean=0.287...`` in the same run) shows
the real training signal was fine all along. Display-only, again.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.training.callbacks import CompletionSampleLogger  # noqa: E402


def test_sample_logger_passes_gold_to_gold_anchored_components():
    """Every gold-anchored component MUST carry a gold_gloss kwarg so
    _capture substitutes the per-sample gold."""
    logger = CompletionSampleLogger(reward_fns=[], reward_weights=[], n_samples=1)
    gold_needing = {
        "translation_quality_reward",
        "bleu_reward",
        "gold_structure_reward",
        "verifier_scaled_reward",
        "gloss_order_reward",
        "edit_validity_reward",
    }
    for name, _fn, kwargs in logger._component_fns:
        if name in gold_needing:
            assert "gold_gloss" in kwargs, (
                f"{name} must carry a gold_gloss kwarg so _capture "
                "substitutes the per-sample gold (v2 components return "
                "neutral 0.0 without it)"
            )


def _make_logger_with_all_weights():
    """Logger con TUTTI i componenti a peso 1 (per il breakdown end-to-end).

    _capture skips i componenti con peso <= 0, quindi serve un
    _weight_map completo con i nomi reali delle funzioni.
    """
    from src.rewards.t2g_rewards import (
        gloss_format_reward,
        gloss_order_reward,
        gloss_repetition_reward,
        gold_structure_reward,
        translation_quality_reward,
        verifier_scaled_reward,
    )

    fns = [
        translation_quality_reward,
        gold_structure_reward,
        verifier_scaled_reward,
        gloss_order_reward,
        gloss_format_reward,
        gloss_repetition_reward,
    ]
    return CompletionSampleLogger(
        reward_fns=fns, reward_weights=[1.0] * len(fns), n_samples=1
    )


def test_sample_logger_breakdown_gold_anchored_perfect_completion(reward_setup):
    """End-to-end: _capture on a perfect completion gives ~+1 for the
    gold-anchored structural components (not 0.0)."""
    logger = _make_logger_with_all_weights()
    perfect = "IX MAN WALK HOUSE"
    logger._capture([perfect], prompts=None, gold_gloss=[perfect])
    sample = logger._buffer[0]
    bd = sample["breakdown"]
    # e i componenti gold-dependent classici restano +1
    assert bd["translation_quality_reward"] > 0.99
    assert bd["gold_structure_reward"] > 0.99


def test_sample_logger_breakdown_without_gold_returns_floor(reward_setup):
    """Senza il kwarg gold_gloss (TRL non lo passa), i componenti
    gold-dependent cadono sulla floor -1.0 (gold vuoto), senza crash."""
    logger = _make_logger_with_all_weights()
    logger._capture(["IX MAN WALK HOUSE"], prompts=None, gold_gloss=None)
    sample = logger._buffer[0]
    bd = sample["breakdown"]
    assert bd["translation_quality_reward"] == -1.0
    assert bd["gold_structure_reward"] == -1.0


def test_sample_logger_breakdown_includes_edit_validity_when_it_is_the_only_weight(
    reward_setup,
):
    """The exact ablations/rewards/edit-validity.yaml scenario: ONLY
    edit_validity_reward has weight > 0. The breakdown (and therefore the
    printed REWARDS/TOTAL line) must reflect it instead of being empty/0.0
    for a completion that is clearly wrong."""
    from src.rewards.t2g_rewards import edit_validity_reward

    logger = CompletionSampleLogger(
        reward_fns=[edit_validity_reward], reward_weights=[1.0], n_samples=1
    )
    logger._capture(
        ["X-WE NEED COOPERATION NOTIFICATION"],
        prompts=None,
        gold_gloss=["X-WE NEED COOPERATION , DESC-NOT CONFRONTATION ."],
    )
    sample = logger._buffer[0]
    bd = sample["breakdown"]
    assert "edit_validity_reward" in bd
    # Real value from the actual mismatched pair above: not the 0.0 the
    # missing-component bug produced for every sample in the real run.
    assert bd["edit_validity_reward"] != 0.0
