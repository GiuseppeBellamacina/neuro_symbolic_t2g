"""Regression: CompletionSampleLogger breakdown passes gold to v2 components.

Bug (found on the real all-rewards run 20260903): the per-sample breakdown
in CompletionSampleLogger._capture stores {"normalize": True} (no
"gold_gloss" key) for the three v2 gold-anchored structural components,
so the gold substitution `if "gold_gloss" in kwargs_call` never fires →
components called without gold → v2 returns neutral 0.0 → the sample
display showed "+0.00" for PERFECT completions (while the trainer metrics
were correctly ~0.87 — the training signal was intact, display-only bug).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.training.callbacks import CompletionSampleLogger  # noqa: E402


def test_sample_logger_passes_gold_to_gold_anchored_components():
    """Every v2 gold-anchored component MUST carry a gold_gloss kwarg so
    _capture substitutes the per-sample gold."""
    logger = CompletionSampleLogger(reward_fns=[], reward_weights=[], n_samples=1)
    gold_needing = {
        "translation_quality_reward",
        "bleu_reward",
        "gold_structure_reward",
        "structural_dense_reward",
        "viterbi_distance_reward",
        "soft_viterbi_distance_reward",
        "verifier_scaled_reward",
        "gloss_order_reward",
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

    _capture salta i componenti con peso <= 0, quindi serve un
    _weight_map completo con i nomi reali delle funzioni.
    """
    from src.rewards.t2g_rewards import (
        gloss_format_reward,
        gloss_order_reward,
        gloss_repetition_reward,
        gold_structure_reward,
        soft_viterbi_distance_reward,
        structural_dense_reward,
        translation_quality_reward,
        viterbi_distance_reward,
    )

    fns = [
        translation_quality_reward,
        gold_structure_reward,
        structural_dense_reward,
        viterbi_distance_reward,
        soft_viterbi_distance_reward,
        gloss_order_reward,
        gloss_format_reward,
        gloss_repetition_reward,
    ]
    return CompletionSampleLogger(
        reward_fns=fns, reward_weights=[1.0] * len(fns), n_samples=1
    )


def test_sample_logger_breakdown_gold_anchored_perfect_completion(reward_setup):
    """End-to-end: _capture on a perfect completion gives ~+1 for the v2
    structural components (not 0.0)."""
    logger = _make_logger_with_all_weights()
    perfect = "IX MAN WALK HOUSE"
    logger._capture([perfect], prompts=None, gold_gloss=[perfect])
    sample = logger._buffer[0]
    bd = sample["breakdown"]
    # v2 gold-anchored: perfect completion → delta 0 → +1 (≈, non 0.0)
    assert bd["structural_dense_reward"] > 0.9, (
        f"structural_dense deve essere ~+1 su completion perfetta, "
        f"ottenuto {bd['structural_dense_reward']}"
    )
    assert bd["viterbi_distance_reward"] > 0.9, bd["viterbi_distance_reward"]
    assert bd["soft_viterbi_distance_reward"] > 0.9, bd["soft_viterbi_distance_reward"]
    # e i componenti gold-dependent classici restano +1
    assert bd["translation_quality_reward"] > 0.99
    assert bd["gold_structure_reward"] > 0.99


def test_sample_logger_breakdown_without_gold_returns_neutral(reward_setup):
    """Senza il kwarg gold_gloss (TRL non lo passa), i v2 restano neutri
    0.0 — comportamento DOCUMENTATO, non bug: il warn-once scatta."""
    import src.rewards.t2g_rewards as R

    R._warned_missing_gold = False  # reset del warn-once

    logger = _make_logger_with_all_weights()
    logger._capture(["IX MAN WALK HOUSE"], prompts=None, gold_gloss=None)
    sample = logger._buffer[0]
    bd = sample["breakdown"]
    assert bd["structural_dense_reward"] == 0.0
    assert bd["viterbi_distance_reward"] == 0.0
    assert bd["soft_viterbi_distance_reward"] == 0.0
