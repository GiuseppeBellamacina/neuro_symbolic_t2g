#!/usr/bin/env python3
"""Test reward functions for T2G GRPO training.

Validates:
  1. Translation quality (ROUGE-L): perfect match=1.0, bad match<perfect
  2. Structural dense: range [-1,1], plausible>implausible
  3. Format: clean gloss=1.0, free text<1.0
  4. Repetition: normal=1.0, repetitive<1.0, severe=-1.0
  5. Gold-structure: perfect=1.0, partial<perfect, implausible<partial
  6. Viterbi distance: range [-1,1], plausible>bad
  7. build_t2g_reward_functions: correct count, weights sum to 1.0
  8. Soft Viterbi: range [-1,1], plausible>bad
  9. Verifier-scaled: perfect>bad, empty=-1.0

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
    assert score_bad >= -1.0

    assert translation_quality_reward("", gold) == -1.0, "Empty completion = -1.0"
    assert translation_quality_reward(perfect, "") == -1.0, "Empty gold = -1.0"


# ---------------------------------------------------------------------------
# 2. Structural dense (bigram)
# ---------------------------------------------------------------------------


def test_structural_dense(reward_setup):
    """v2 gold-anchored: gold → exactly +1, corrupted < gold, guards hold."""
    from src.rewards.t2g_rewards import structural_dense_reward

    gold = "IX MAN WALK HOUSE"

    # Perfect completion == gold → calibrated +1 (was ≈ −0.99 pre-fix)
    assert structural_dense_reward(gold, gold) == 1.0, "gold-on-gold must be +1"

    # Corrupted chain (breaks MAN→WALK, WALK→HOUSE) clearly below gold
    swap = "IX MAN HOUSE WALK"
    s_swap = structural_dense_reward(swap, gold)
    assert -1.0 <= s_swap < 1.0, f"swap in [-1,1), got {s_swap:.4f}"
    assert s_swap < 1.0

    # Off-chain garbage also below gold
    garbage = "DOG fs-JOHN BOOK CAN"
    s_garbage = structural_dense_reward(garbage, gold)
    assert s_garbage < 1.0
    assert -1.0 <= s_garbage <= 1.0

    # Guards: single in-vocab token / empty → hard -1
    assert structural_dense_reward("IX", gold) == -1.0, "Single token = -1.0"
    assert structural_dense_reward("", gold) == -1.0, "Empty = -1.0"

    # Missing gold → neutral 0.0 (cannot calibrate without the anchor)
    assert structural_dense_reward(gold, "") == 0.0, "Missing gold = 0.0"

    # Raw mode: per-transition delta in nats (worse than gold → negative)
    raw = structural_dense_reward(swap, gold, normalize=False)
    assert raw < 0.0, f"Raw delta < 0 (worse than gold), got {raw:.4f}"


def test_structural_dense_variance_signal(reward_setup):
    """The run-7078 bug: std ≈ 0 across completions → zero GRPO advantage.

    A mixed group (gold, corrupted, garbage) must now spread widely and
    the gold must top the group."""
    import numpy as np

    from src.rewards.t2g_rewards import structural_dense_reward

    gold = "IX MAN WALK HOUSE"
    group = [gold, "IX MAN HOUSE WALK", "DOG fs-JOHN BOOK CAN", "WANT GO COME NOT"]
    vals = [structural_dense_reward(c, gold) for c in group]
    assert np.std(vals) > 0.1, f"Group std must be >> 0 (was ~0.001 pre-fix): {vals}"
    assert vals[0] == max(vals), f"Gold must top the group: {vals}"


def test_structural_dense_missing_gold_neutral(reward_setup, caplog):
    """No gold → 0.0 for every sample (advantage-neutral in the GRPO group),
    with a single warning — never a constant −1 that poisons rollouts."""
    from src.rewards.t2g_rewards import structural_dense_reward

    for completion in ("IX MAN WALK HOUSE", "DOG CAT BIRD"):
        assert structural_dense_reward(completion, "") == 0.0


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

    assert gloss_format_reward("") == -1.0, "Empty = -1.0"

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

    # Anti-reward-hacking: sequences < 4 tokens (and empty outputs) return a
    # NEUTRAL 0.0 — the old +1.0 rewarded short outputs unconditionally.
    assert gloss_repetition_reward("IX MAN") == 0.0, "Short (<4 tokens) = 0.0"
    assert gloss_repetition_reward("") == 0.0, "Empty = 0.0"


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
    assert -1.0 <= score_partial <= 1.0, f"Partial in [-1,1], got {score_partial:.4f}"
    assert (
        score_partial < score_perfect
    ), f"Partial < perfect: {score_partial:.4f} < {score_perfect:.4f}"

    implausible = "DOG fs-JOHN BOOK CAN NOT"
    score_implausible = gold_structure_reward(implausible, gold, normalize=True)
    assert -1.0 <= score_implausible <= 1.0
    assert (
        score_implausible < score_partial
    ), f"Implausible < partial: {score_implausible:.4f} < {score_partial:.4f}"

    assert gold_structure_reward("", gold, normalize=True) == -1.0, "Empty = -1.0"
    assert (
        gold_structure_reward(perfect, "", normalize=True) == -1.0
    ), "Empty gold = -1.0"

    raw = gold_structure_reward(perfect, gold, normalize=False)
    assert abs(raw) < 0.5, f"Raw perfect ~= 0.0, got {raw:.4f}"


def test_gold_structure_reward_anti_hacking(reward_setup):
    """Anti reward-hacking guards in gold_structure_reward."""
    from src.rewards.t2g_rewards import gold_structure_reward

    gold = "IX MAN WALK HOUSE"

    # < 2 in-vocab tokens → hard -1.0 (was: gliding on BOS→EOS uniform edge).
    assert (
        gold_structure_reward("IX", gold, normalize=True) == -1.0
    ), "Single in-vocab token = -1.0"
    # All-OOV garbage → 0 in-vocab tokens → -1.0.
    assert (
        gold_structure_reward("ZZZ QQQ RRR", gold, normalize=True) == -1.0
    ), "All-OOV completion = -1.0"

    # Length-mismatch penalty: a 2-token completion against a 4-token gold
    # gets factor min(2,4)/max(2,4)=0.5, so its score cannot reach the
    # full-length perfect match and is pushed to <= 0.
    short = "IX MAN"
    full = "IX MAN WALK HOUSE"
    score_short = gold_structure_reward(short, gold, normalize=True)
    score_full = gold_structure_reward(full, gold, normalize=True)
    assert score_full > 0.9, f"Full perfect ~= 1.0, got {score_full:.4f}"
    assert score_short < score_full, "Short path < full-length match"
    assert score_short <= 0.0, f"Short path penalized to <= 0, got {score_short:.4f}"


# ---------------------------------------------------------------------------
# 5b. GRPOTrainer wrapper: gold gloss via kwargs (no global registry)
# ---------------------------------------------------------------------------


def test_reward_wrapper_receives_gold_via_kwargs(reward_setup):
    """Reward wrapper reads gold_gloss from kwargs, aligned with completions."""
    from src.rewards.t2g_rewards import (
        _make_gloss_reward_fn,
        translation_quality_reward,
    )

    fn = _make_gloss_reward_fn(translation_quality_reward, needs_gold_gloss=True)

    completions = ["IX MAN WALK HOUSE", "DOG CAT BIRD FISH"]
    gold_gloss = ["IX MAN WALK HOUSE", "IX MAN WALK HOUSE"]
    scores = fn(
        completions,
        prompts=["p1", "p2"],
        gold_gloss=gold_gloss,
    )

    assert len(scores) == 2
    assert scores[0] > 0.8, f"Perfect match via kwargs > 0.8, got {scores[0]:.4f}"
    assert scores[1] < scores[0], "Bad match via kwargs < perfect"


def test_reward_wrapper_missing_gold_is_neutral(reward_setup, caplog):
    """Missing/None gold_gloss → neutral 0.0 (never -1.0), warn once."""
    import src.rewards.t2g_rewards as _R
    from src.rewards.t2g_rewards import (
        _make_gloss_reward_fn,
        translation_quality_reward,
    )

    # The warn-once flag is session-global and earlier tests may have
    # consumed the warning (missing-gold paths of the structural rewards).
    # This test asserts the warn-ONCE semantics: reset for determinism.
    _R._warned_missing_gold = False

    fn = _make_gloss_reward_fn(translation_quality_reward, needs_gold_gloss=True)

    # gold_gloss kwarg absent entirely.
    assert fn(["IX MAN WALK HOUSE"], prompts=["p1"]) == [0.0]

    # gold_gloss present but shorter than completions → missing for the tail.
    with caplog.at_level("WARNING", logger="src.rewards.t2g_rewards"):
        assert fn(
            ["IX MAN WALK HOUSE", "DOG CAT"],
            prompts=["p1", "p2"],
            gold_gloss=["IX MAN WALK HOUSE"],
        ) == [1.0, 0.0]

    # Warning is logged at most once across all calls.
    with caplog.at_level("WARNING", logger="src.rewards.t2g_rewards"):
        fn(["A B C D"], prompts=["p"], gold_gloss=[])
    warn_count = sum(1 for r in caplog.records if "gold_gloss" in r.message)
    assert warn_count == 1, f"Warned exactly once, got {warn_count}"


# ---------------------------------------------------------------------------
# 6. Viterbi distance reward
# ---------------------------------------------------------------------------


def test_viterbi_distance_reward(reward_setup):
    """v2 gold-anchored: gold → +1, off-chain < gold, guards + raw mode."""
    from src.rewards.t2g_rewards import viterbi_distance_reward

    gold = "IX MAN WALK HOUSE"

    # Perfect completion == gold → +1 (was ≈ −0.91 pre-fix on the real matrix)
    assert viterbi_distance_reward(gold, gold) == 1.0, "gold-on-gold must be +1"

    swap = "IX MAN HOUSE WALK"
    s_swap = viterbi_distance_reward(swap, gold)
    assert -1.0 <= s_swap < 1.0, f"swap in [-1,1), got {s_swap:.4f}"

    garbage = "DOG fs-JOHN BOOK CAN NOT WANT"
    s_garbage = viterbi_distance_reward(garbage, gold)
    assert s_garbage < 1.0, f"garbage < gold: {s_garbage:.4f}"
    assert -1.0 <= s_garbage <= 1.0

    assert viterbi_distance_reward("IX", gold) == -1.0, "Short (<2 tokens) = -1.0"
    assert viterbi_distance_reward("", gold) == -1.0, "Empty = -1.0"
    assert viterbi_distance_reward(gold, "") == 0.0, "Missing gold = 0.0"

    # Raw mode: per-transition gap delta (worse than gold → negative)
    raw = viterbi_distance_reward(swap, gold, normalize=False)
    assert raw < 0.0, f"Raw gap delta < 0, got {raw:.4f}"


def test_viterbi_bound_cached_by_length(reward_setup, monkeypatch):
    """The bound is a property of automaton+length only: one decode per
    length (the per-completion decodes were the 391 s/it killer in 7078)."""
    import src.datasets.transition_matrix as tm
    from src.rewards.t2g_rewards import (
        initialize_rewards,
        viterbi_distance_reward,
    )

    vocab, bigram, _ = reward_setup
    # Fresh caches for this test
    initialize_rewards(bigram, vocab)

    calls = {"n": 0}
    orig = tm.viterbi_optimal_score_diverse

    def counting(*args, **kwargs):
        calls["n"] += 1
        return orig(*args, **kwargs)

    monkeypatch.setattr(tm, "viterbi_optimal_score_diverse", counting)

    gold = "IX MAN WALK HOUSE"
    for _ in range(5):
        viterbi_distance_reward(gold, gold)
    assert calls["n"] == 1, f"same length must decode ONCE, got {calls['n']}"

    # A different completion length → exactly one more decode
    viterbi_distance_reward("IX MAN WALK", gold)
    assert calls["n"] == 2, f"new length = one more decode, got {calls['n']}"

    # ...and the gold itself is scored once, not once per call
    # (gold stats cached by text)
    from src.rewards import t2g_rewards as R

    assert gold in R._gold_stats_cache


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
    """v2 gold-anchored: gold → +1, off-chain < gold, guards + raw mode."""
    from src.rewards.t2g_rewards import soft_viterbi_distance_reward

    gold = "IX MAN WALK HOUSE"

    assert soft_viterbi_distance_reward(gold, gold) == 1.0, "gold-on-gold must be +1"

    swap = "IX MAN HOUSE WALK"
    s_swap = soft_viterbi_distance_reward(swap, gold)
    assert -1.0 <= s_swap < 1.0, f"swap in [-1,1), got {s_swap:.4f}"

    bad = "DOG fs-JOHN BOOK CAN NOT WANT"
    score_bad = soft_viterbi_distance_reward(bad, gold)
    assert score_bad < 1.0, f"Bad < gold: {score_bad:.4f}"
    assert -1.0 <= score_bad <= 1.0

    assert soft_viterbi_distance_reward("IX", gold) == -1.0, "Short (<2 tokens) = -1.0"
    assert soft_viterbi_distance_reward("", gold) == -1.0, "Empty = -1.0"
    assert soft_viterbi_distance_reward(gold, "") == 0.0, "Missing gold = 0.0"

    raw = soft_viterbi_distance_reward(swap, gold, normalize=False)
    assert raw < 0.0, f"Raw gap delta < 0, got {raw:.4f}"


def test_soft_bound_cached_by_length(reward_setup, monkeypatch):
    """The log-partition is automaton+length only → one forward pass per
    length, cached forever."""
    import src.datasets.transition_matrix as tm
    from src.rewards.t2g_rewards import (
        initialize_rewards,
        soft_viterbi_distance_reward,
    )

    vocab, bigram, _ = reward_setup
    initialize_rewards(bigram, vocab)

    calls = {"n": 0}
    orig = tm.soft_viterbi_score

    def counting(*args, **kwargs):
        calls["n"] += 1
        return orig(*args, **kwargs)

    monkeypatch.setattr(tm, "soft_viterbi_score", counting)

    gold = "IX MAN WALK HOUSE"
    for _ in range(4):
        soft_viterbi_distance_reward(gold, gold)
    assert calls["n"] == 1, f"same length = one forward pass, got {calls['n']}"


# ---------------------------------------------------------------------------
# 9. Verifier-scaled reward
# ---------------------------------------------------------------------------


def test_verifier_scaled_reward(reward_setup):
    from src.rewards.t2g_rewards import verifier_scaled_reward

    plausible = "IX MAN WALK HOUSE"
    gold = "IX MAN WALK HOUSE"
    score = verifier_scaled_reward(plausible, gold)
    assert -1.0 <= score <= 1.0, f"Verifier-scaled perfect in [-1,1], got {score:.4f}"
    # With log1p(structural) scaling, perfect match gives ~0.40 (not >0.5
    # as in the old structural^gamma formula). The key property is that
    # it's positive and significantly higher than a bad match.
    assert score > 0.1, f"Verifier-scaled perfect > 0.1, got {score:.4f}"

    bad = "DOG fs-JOHN BOOK CAN NOT WANT"
    score_bad = verifier_scaled_reward(bad, gold)
    assert score_bad < score, f"Bad < perfect: {score_bad:.4f} < {score:.4f}"
    assert -1.0 <= score_bad <= 1.0

    assert verifier_scaled_reward("", gold) == -1.0, "Empty = -1.0"
    assert verifier_scaled_reward(plausible, "") == -1.0, "Empty gold = -1.0"
