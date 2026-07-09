#!/usr/bin/env python3
"""Test reward functions for T2G GRPO training.

Validates:
  1. Translation quality (ROUGE-L): perfect match=1.0, bad match<perfect
  2. Structural dense: range [0,1], plausible>implausible
  3. Format: clean gloss=1.0, free text<1.0
  4. Repetition: normal=1.0, repetitive<1.0, severe=-1.0
  5. Gold-structure: perfect=1.0, partial<perfect, implausible<partial
  6. Viterbi distance: range [0,1], plausible>bad
  7. build_t2g_reward_functions: correct count, weights sum to 1.0
  8. Soft Viterbi: range [0,1], plausible>bad
  9. Verifier-scaled: perfect>bad, empty=0.0

All tests use the ``reward_setup`` fixture from conftest.py.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# 1. Translation quality (ROUGE-L)
# ---------------------------------------------------------------------------


def test_translation_quality(reward_setup):
    from src.rewards.t2g_rewards import translation_quality_reward

    gold = "IX MAN WALK HOUSE"
    perfect = "IX MAN WALK HOUSE"
    score_perfect = translation_quality_reward(perfect, gold)
    assert score_perfect > 0.8, f"Perfect match > 0.8, got {score_perfect:.4f}"
    assert (
        abs(score_perfect - 1.0) < 0.01
    ), f"Perfect match == 1.0, got {score_perfect:.4f}"

    partial = "IX MAN GO HOUSE"
    score_partial = translation_quality_reward(partial, gold)
    assert (
        score_partial < score_perfect
    ), f"Partial < perfect: {score_partial:.4f} vs {score_perfect:.4f}"
    assert score_partial > 0.0

    bad = "DOG CAT BIRD FISH"
    score_bad = translation_quality_reward(bad, gold)
    assert (
        score_bad < score_partial
    ), f"Bad < partial: {score_bad:.4f} < {score_partial:.4f}"
    assert score_bad >= 0.0

    assert translation_quality_reward("", gold) == 0.0, "Empty completion = 0.0"
    assert translation_quality_reward(perfect, "") == 0.0, "Empty gold = 0.0"


# ---------------------------------------------------------------------------
# 2. Structural dense (bigram)
# ---------------------------------------------------------------------------


def test_structural_dense(reward_setup):
    from src.rewards.t2g_rewards import structural_dense_reward

    plausible = "IX MAN WALK HOUSE"
    score_plausible = structural_dense_reward(plausible, normalize=True)
    assert (
        0.0 <= score_plausible <= 1.0
    ), f"Plausible in [0,1], got {score_plausible:.4f}"
    assert score_plausible > 0.0

    implausible = "DOG fs-JOHN BOOK CAN"
    score_implausible = structural_dense_reward(implausible, normalize=True)
    assert (
        0.0 <= score_implausible <= 1.0
    ), f"Implausible in [0,1], got {score_implausible:.4f}"
    assert (
        score_plausible > score_implausible
    ), f"Plausible > implausible: {score_plausible:.4f} vs {score_implausible:.4f}"

    assert structural_dense_reward("IX", normalize=True) == 0.0, "Single token = 0.0"
    assert structural_dense_reward("", normalize=True) == 0.0, "Empty = 0.0"

    raw = structural_dense_reward(plausible, normalize=False)
    assert raw < 0.0, f"Raw score < 0 (log-prob), got {raw:.4f}"


# ---------------------------------------------------------------------------
# 3. Format reward
# ---------------------------------------------------------------------------


def test_format_reward(reward_setup):
    from src.rewards.t2g_rewards import gloss_format_reward

    assert gloss_format_reward("IX MAN WALK HOUSE") == 1.0, "Clean gloss = 1.0"

    mixed = "Here is: IX MAN WALK"
    assert gloss_format_reward(mixed) < 1.0, f"Mixed < 1.0, got {mixed}"

    free_text = "The man walks to the house."
    assert gloss_format_reward(free_text) < 1.0, "Free text < 1.0"

    assert gloss_format_reward("") == 0.0, "Empty = 0.0"

    json_like = '{"gloss": "IX MAN"}'
    assert gloss_format_reward(json_like) < 1.0, "JSON-like < 1.0"


# ---------------------------------------------------------------------------
# 4. Repetition reward
# ---------------------------------------------------------------------------


def test_repetition_reward(reward_setup):
    from src.rewards.t2g_rewards import gloss_repetition_reward

    normal = "IX MAN WALK HOUSE BOOK CAN NOT WANT GO COME"
    assert gloss_repetition_reward(normal) == 1.0, "Normal = 1.0"

    moderate = "IX IX MAN WALK IX IX MAN WALK"
    score_moderate = gloss_repetition_reward(moderate)
    assert score_moderate <= 1.0, f"Moderate <= 1.0, got {score_moderate}"
    assert score_moderate < 1.0, f"Moderate < 1.0 (penalized), got {score_moderate}"

    severe = "IX IX IX IX IX IX IX IX IX IX"
    assert gloss_repetition_reward(severe) == -1.0, "Severe = -1.0"

    assert gloss_repetition_reward("IX MAN") == 1.0, "Short (<4 tokens) = 1.0"
    assert gloss_repetition_reward("") == 1.0, "Empty = 1.0"


# ---------------------------------------------------------------------------
# 5. Gold-structure reward
# ---------------------------------------------------------------------------


def test_gold_structure_reward(reward_setup):
    from src.rewards.t2g_rewards import gold_structure_reward

    gold = "IX MAN WALK HOUSE"
    perfect = "IX MAN WALK HOUSE"
    score_perfect = gold_structure_reward(perfect, gold, normalize=True)
    assert (
        abs(score_perfect - 1.0) < 0.05
    ), f"Perfect match ~= 1.0, got {score_perfect:.4f}"

    partial = "IX MAN GO HOUSE"
    score_partial = gold_structure_reward(partial, gold, normalize=True)
    assert 0.0 <= score_partial <= 1.0, f"Partial in [0,1], got {score_partial:.4f}"
    assert (
        score_partial < score_perfect
    ), f"Partial < perfect: {score_partial:.4f} < {score_perfect:.4f}"

    implausible = "DOG fs-JOHN BOOK CAN NOT"
    score_implausible = gold_structure_reward(implausible, gold, normalize=True)
    assert 0.0 <= score_implausible <= 1.0
    assert (
        score_implausible < score_partial
    ), f"Implausible < partial: {score_implausible:.4f} < {score_partial:.4f}"

    assert gold_structure_reward("", gold, normalize=True) == 0.0, "Empty = 0.0"
    assert gold_structure_reward(perfect, "", normalize=True) == 0.0, "Empty gold = 0.0"

    raw = gold_structure_reward(perfect, gold, normalize=False)
    assert abs(raw) < 0.5, f"Raw perfect ~= 0.0, got {raw:.4f}"


# ---------------------------------------------------------------------------
# 6. Viterbi distance reward
# ---------------------------------------------------------------------------


def test_viterbi_distance_reward(reward_setup):
    from src.rewards.t2g_rewards import viterbi_distance_reward

    plausible = "IX MAN WALK HOUSE"
    score = viterbi_distance_reward(plausible, normalize=True)
    assert 0.0 <= score <= 1.0, f"Viterbi distance in [0,1], got {score:.4f}"
    assert score > 0.0

    assert (
        viterbi_distance_reward("IX", normalize=True) == 0.0
    ), "Short (<2 tokens) = 0.0"
    assert viterbi_distance_reward("", normalize=True) == 0.0, "Empty = 0.0"

    bad = "DOG fs-JOHN BOOK CAN NOT WANT"
    score_bad = viterbi_distance_reward(bad, normalize=True)
    assert score_bad < score, f"Bad < plausible: {score_bad:.4f} < {score:.4f}"
    assert 0.0 <= score_bad <= 1.0
    assert score > 0.05, f"Plausible > 0.05 (diverse baseline), got {score:.4f}"

    raw = viterbi_distance_reward(plausible, normalize=False)
    assert raw <= 0.0, f"Raw <= 0.0, got {raw:.4f}"


# ---------------------------------------------------------------------------
# 7. build_t2g_reward_functions
# ---------------------------------------------------------------------------


def test_build_reward_functions(reward_setup):
    from src.rewards.t2g_rewards import build_t2g_reward_functions

    # Default config
    funcs, weights = build_t2g_reward_functions()
    assert len(funcs) == 4, f"4 default functions, got {len(funcs)}"
    assert len(weights) == 4
    assert len(funcs) == len(weights)
    assert all(w > 0 for w in weights), f"All weights > 0: {weights}"
    assert abs(sum(weights) - 1.0) < 0.01, f"Weights sum to 1.0, got {sum(weights):.4f}"

    # Each function should be callable
    completions = ["IX MAN WALK", "DOG CAT", "NOT CAN WANT"]
    for fn in funcs:
        result = fn(completions)
        assert isinstance(result, list), f"{fn.__name__} returns list"
        assert len(result) == len(completions), f"{fn.__name__} returns correct length"
        assert all(
            isinstance(v, float) for v in result
        ), f"{fn.__name__} values are floats"

    # Custom with gold_structure
    custom = {
        "weight_translation": 0.4,
        "weight_gold_structure": 0.4,
        "weight_format": 0.1,
        "weight_repetition": 0.1,
    }
    funcs2, weights2 = build_t2g_reward_functions(custom)
    assert len(funcs2) == 4, f"Custom (gold-structure): 4 functions, got {len(funcs2)}"
    assert abs(sum(weights2) - 1.0) < 0.01

    # Custom with viterbi
    custom_vit = {
        "weight_translation": 0.3,
        "weight_viterbi": 0.3,
        "weight_gold_structure": 0.3,
        "weight_format": 0.05,
        "weight_repetition": 0.05,
    }
    funcs3, weights3 = build_t2g_reward_functions(custom_vit)
    assert len(funcs3) == 5, f"Custom (viterbi): 5 functions, got {len(funcs3)}"
    assert abs(sum(weights3) - 1.0) < 0.01

    # Old-style structural_dense
    custom_old = {"weight_translation": 0.5, "weight_structure": 0.5}
    funcs4, weights4 = build_t2g_reward_functions(custom_old)
    assert len(funcs4) == 2, f"Old-style (structure): 2 functions, got {len(funcs4)}"
    assert abs(sum(weights4) - 1.0) < 0.01

    # Soft Viterbi
    custom_soft = {
        "weight_translation": 0.3,
        "weight_soft_viterbi": 0.3,
        "weight_gold_structure": 0.3,
        "weight_format": 0.05,
        "weight_repetition": 0.05,
    }
    funcs5, weights5 = build_t2g_reward_functions(custom_soft)
    assert len(funcs5) == 5, f"Custom (soft-viterbi): 5 functions, got {len(funcs5)}"
    assert abs(sum(weights5) - 1.0) < 0.01

    # Verifier-scaled
    custom_ver = {
        "weight_verifier_scaled": 0.65,
        "weight_gloss_order": 0.15,
        "weight_format": 0.10,
        "weight_repetition": 0.10,
    }
    funcs6, weights6 = build_t2g_reward_functions(custom_ver)
    assert len(funcs6) == 4, f"Custom (verifier-scaled): 4 functions, got {len(funcs6)}"
    assert abs(sum(weights6) - 1.0) < 0.01


# ---------------------------------------------------------------------------
# 8. Soft Viterbi distance reward
# ---------------------------------------------------------------------------


def test_soft_viterbi_distance_reward(reward_setup):
    from src.rewards.t2g_rewards import soft_viterbi_distance_reward

    plausible = "IX MAN WALK HOUSE"
    score = soft_viterbi_distance_reward(plausible, normalize=True)
    assert 0.0 <= score <= 1.0, f"Soft Viterbi in [0,1], got {score:.4f}"
    assert score > 0.0

    assert (
        soft_viterbi_distance_reward("IX", normalize=True) == 0.0
    ), "Short (<2 tokens) = 0.0"
    assert soft_viterbi_distance_reward("", normalize=True) == 0.0, "Empty = 0.0"

    bad = "DOG fs-JOHN BOOK CAN NOT WANT"
    score_bad = soft_viterbi_distance_reward(bad, normalize=True)
    assert score_bad < score, f"Bad < plausible: {score_bad:.4f} < {score:.4f}"
    assert 0.0 <= score_bad <= 1.0

    raw = soft_viterbi_distance_reward(plausible, normalize=False)
    assert raw <= 0.0, f"Soft Viterbi raw <= 0.0, got {raw:.4f}"


# ---------------------------------------------------------------------------
# 9. Verifier-scaled reward
# ---------------------------------------------------------------------------


def test_verifier_scaled_reward(reward_setup):
    from src.rewards.t2g_rewards import verifier_scaled_reward

    plausible = "IX MAN WALK HOUSE"
    gold = "IX MAN WALK HOUSE"
    score = verifier_scaled_reward(plausible, gold)
    assert 0.0 <= score <= 1.0, f"Verifier-scaled perfect in [0,1], got {score:.4f}"
    # With log1p(structural) scaling, perfect match gives ~0.40 (not >0.5
    # as in the old structural^gamma formula). The key property is that
    # it's positive and significantly higher than a bad match.
    assert score > 0.1, f"Verifier-scaled perfect > 0.1, got {score:.4f}"

    bad = "DOG fs-JOHN BOOK CAN NOT WANT"
    score_bad = verifier_scaled_reward(bad, gold)
    assert score_bad < score, f"Bad < perfect: {score_bad:.4f} < {score:.4f}"
    assert 0.0 <= score_bad <= 1.0

    assert verifier_scaled_reward("", gold) == 0.0, "Empty = 0.0"
    assert verifier_scaled_reward(plausible, "") == 0.0, "Empty gold = 0.0"
