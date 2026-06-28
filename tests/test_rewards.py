#!/usr/bin/env python3
"""Verify Reward Functions ? Test Script.

Validates:
  1. Translation quality (ROUGE-L): perfect match=1.0, bad match<perfect
  2. Structural dense: range [0,1], plausible>implausible
  3. Format: clean gloss=1.0, free text<1.0
  4. Repetition: normal=1.0, repetitive<1.0, severe=-1.0

Usage:
    python tests/test_rewards.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

PASS = 0
FAIL = 0


def check(name: str, condition: bool, detail: str = "") -> None:
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  [PASS] {name}" + (f" -- {detail}" if detail else ""))
    else:
        FAIL += 1
        print(f"  [FAIL] {name}" + (f" -- {detail}" if detail else ""))


def setup_rewards():
    """Setup mini vocabulary and bigram matrix for testing."""
    import tempfile

    import numpy as np

    from src.datasets.transition_matrix import save_transition_matrix
    from src.rewards.t2g_rewards import initialize_rewards

    vocab = [
        "<BOS>",
        "<EOS>",
        "<UNK>",
        "IX",
        "MAN",
        "WALK",
        "HOUSE",
        "BOOK",
        "DOG",
        "CAT",
        "NOT",
        "CAN",
        "WANT",
        "GO",
        "COME",
        "fs-JOHN",
    ]
    V = len(vocab)
    # Create a plausible bigram matrix with Laplace smoothing
    # Make "IX ? MAN" and "MAN ? WALK" high probability
    bigram = np.ones((V, V), dtype=np.float32) * (1.0 / V)
    token_to_idx = {t: i for i, t in enumerate(vocab)}
    # Set some plausible transitions
    pairs = [
        ("<BOS>", "IX"),
        ("IX", "MAN"),
        ("MAN", "WALK"),
        ("WALK", "HOUSE"),
        ("HOUSE", "<EOS>"),
    ]
    for a, b in pairs:
        if a in token_to_idx and b in token_to_idx:
            bigram[token_to_idx[a], :] = 0.001
            bigram[token_to_idx[a], token_to_idx[b]] = 0.95
            row = bigram[token_to_idx[a]]
            bigram[token_to_idx[a]] = row / row.sum()

    # Save and reload to test persistence
    with tempfile.TemporaryDirectory() as tmp:
        mpath = str(Path(tmp) / "test_bigram.npy")
        save_transition_matrix(bigram, mpath)

    initialize_rewards(bigram, vocab)
    return vocab, bigram


def test_translation_quality() -> None:
    print("\n-- 1. Translation Quality (ROUGE-L) --")
    from src.rewards.t2g_rewards import translation_quality_reward

    gold = "IX MAN WALK HOUSE"
    perfect = "IX MAN WALK HOUSE"
    score_perfect = translation_quality_reward(perfect, gold)
    check("Perfect match > 0.8", score_perfect > 0.8, f"{score_perfect:.4f}")
    check(
        "Perfect match == 1.0", abs(score_perfect - 1.0) < 0.01, f"{score_perfect:.4f}"
    )

    partial = "IX MAN GO HOUSE"
    score_partial = translation_quality_reward(partial, gold)
    check(
        "Partial match < perfect",
        score_partial < score_perfect,
        f"{score_partial:.4f} vs {score_perfect:.4f}",
    )
    check("Partial match > 0", score_partial > 0.0)

    bad = "DOG CAT BIRD FISH"
    score_bad = translation_quality_reward(bad, gold)
    check("Bad match < partial", score_bad < score_partial, f"{score_bad:.4f}")
    check("Bad match >= 0", score_bad >= 0.0)

    empty = translation_quality_reward("", gold)
    check("Empty completion = 0.0", empty == 0.0)

    empty_gold = translation_quality_reward(perfect, "")
    check("Empty gold = 0.0", empty_gold == 0.0)


def test_structural_dense() -> None:
    print("\n-- 2. Structural Dense (Bigram) --")
    from src.rewards.t2g_rewards import structural_dense_reward

    plausible = "IX MAN WALK HOUSE"
    score_plausible = structural_dense_reward(plausible, normalize=True)
    check("Plausible in [0,1]", 0.0 <= score_plausible <= 1.0, f"{score_plausible:.4f}")
    check("Plausible > 0", score_plausible > 0.0)

    implausible = "DOG fs-JOHN BOOK CAN"
    score_implausible = structural_dense_reward(implausible, normalize=True)
    check(
        "Implausible in [0,1]",
        0.0 <= score_implausible <= 1.0,
        f"{score_implausible:.4f}",
    )
    check(
        "Plausible > implausible",
        score_plausible > score_implausible,
        f"{score_plausible:.4f} vs {score_implausible:.4f}",
    )

    single = "IX"
    score_single = structural_dense_reward(single, normalize=True)
    check("Single token (no bigram) = 0.0", score_single == 0.0)

    empty = structural_dense_reward("", normalize=True)
    check("Empty = 0.0", empty == 0.0)

    # Raw (non-normalized) should be negative
    raw = structural_dense_reward(plausible, normalize=False)
    check("Raw score < 0 (log-prob)", raw < 0.0, f"{raw:.4f}")


def test_format_reward() -> None:
    print("\n-- 3. Format Reward --")
    from src.rewards.t2g_rewards import gloss_format_reward

    clean = "IX MAN WALK HOUSE"
    score_clean = gloss_format_reward(clean)
    check("Clean gloss = 1.0", score_clean == 1.0)

    mixed = "Here is: IX MAN WALK"
    score_mixed = gloss_format_reward(mixed)
    check("Mixed (free text + gloss) < 1.0", score_mixed < 1.0, f"{score_mixed}")

    free_text = "The man walks to the house."
    score_free = gloss_format_reward(free_text)
    check("Pure free text < 1.0", score_free < 1.0, f"{score_free}")

    empty = gloss_format_reward("")
    check("Empty = 0.0", empty == 0.0)

    json_like = '{"gloss": "IX MAN"}'
    score_json = gloss_format_reward(json_like)
    check("JSON-like < 1.0", score_json < 1.0, f"{score_json}")


def test_repetition_reward() -> None:
    print("\n-- 4. Repetition Reward --")
    from src.rewards.t2g_rewards import gloss_repetition_reward

    normal = "IX MAN WALK HOUSE BOOK CAN NOT WANT GO COME"
    score_normal = gloss_repetition_reward(normal)
    check("Normal = 1.0", score_normal == 1.0)

    moderate = "IX IX MAN WALK IX IX MAN WALK"
    score_moderate = gloss_repetition_reward(moderate)
    check("Moderate repetition <= 1.0", score_moderate <= 1.0, f"{score_moderate}")
    check("Moderate < 1.0 (penalized)", score_moderate < 1.0, f"{score_moderate}")

    severe = "IX IX IX IX IX IX IX IX IX IX"
    score_severe = gloss_repetition_reward(severe)
    check("Severe repetition == -1.0", score_severe == -1.0)

    short = "IX MAN"
    score_short = gloss_repetition_reward(short)
    check("Short sequence (<4 tokens) = 1.0", score_short == 1.0)

    empty = gloss_repetition_reward("")
    check("Empty = 1.0", empty == 1.0)


def test_gold_structure_reward() -> None:
    print("\n-- 5. Gold-Structure Reward (Gold Baseline) --")
    from src.rewards.t2g_rewards import gold_structure_reward

    gold = "IX MAN WALK HOUSE"

    # Perfect match with gold (same sequence)
    perfect = "IX MAN WALK HOUSE"
    score_perfect = gold_structure_reward(perfect, gold, normalize=True)
    check(
        "Perfect match = 1.0", abs(score_perfect - 1.0) < 0.05, f"{score_perfect:.4f}"
    )

    # Slightly different from gold
    partial = "IX MAN GO HOUSE"
    score_partial = gold_structure_reward(partial, gold, normalize=True)
    check("Partial in [0, 1]", 0.0 <= score_partial <= 1.0, f"{score_partial:.4f}")
    check(
        "Partial < perfect",
        score_partial < score_perfect,
        f"{score_partial:.4f} < {score_perfect:.4f}",
    )

    # Implausible (bad bigram transitions)
    implausible = "DOG fs-JOHN BOOK CAN NOT"
    score_implausible = gold_structure_reward(implausible, gold, normalize=True)
    check(
        "Implausible in [0, 1]",
        0.0 <= score_implausible <= 1.0,
        f"{score_implausible:.4f}",
    )
    check(
        "Implausible < partial",
        score_implausible < score_partial,
        f"{score_implausible:.4f} < {score_partial:.4f}",
    )

    # Empty
    empty = gold_structure_reward("", gold, normalize=True)
    check("Empty = 0.0", empty == 0.0)

    empty_gold = gold_structure_reward(perfect, "", normalize=True)
    check("Empty gold = 0.0", empty_gold == 0.0)

    # Raw
    raw = gold_structure_reward(perfect, gold, normalize=False)
    check("Raw perfect ~= 0.0", abs(raw) < 0.5, f"{raw:.4f}")


def test_viterbi_distance_reward() -> None:
    print("\n-- 6. Viterbi Distance Reward --")
    from src.rewards.t2g_rewards import viterbi_distance_reward

    plausible = "IX MAN WALK HOUSE"
    score = viterbi_distance_reward(plausible, normalize=True)
    check("Viterbi distance in [0, 1]", 0.0 <= score <= 1.0, f"{score:.4f}")
    check("Viterbi distance > 0", score > 0.0)

    short = "IX"
    score_short = viterbi_distance_reward(short, normalize=True)
    check("Short (<2 tokens) = 0.0", score_short == 0.0)

    empty = viterbi_distance_reward("", normalize=True)
    check("Empty = 0.0", empty == 0.0)

    # Bad sequence — should get lower Viterbi distance score
    bad = "DOG fs-JOHN BOOK CAN NOT WANT"
    score_bad = viterbi_distance_reward(bad, normalize=True)
    check("Bad < plausible", score_bad < score, f"{score_bad:.4f} < {score:.4f}")
    check("Bad in [0, 1]", 0.0 <= score_bad <= 1.0, f"{score_bad:.4f}")

    # Bonus check: the Viterbi distance should NOT be extreme (not ~0.0)
    # because the diverse Viterbi uses a realistic baseline
    check(
        "Plausible Viterbi distance > 0.05 (diverse baseline)",
        score > 0.05,
        f"{score:.4f}",
    )

    # Raw (should be negative)
    raw = viterbi_distance_reward(plausible, normalize=False)
    check("Raw < 0.0", raw < 0.0, f"{raw:.4f}")


def test_build_reward_functions() -> None:
    print("\n-- 7. build_t2g_reward_functions --")
    from src.rewards.t2g_rewards import build_t2g_reward_functions

    funcs, weights = build_t2g_reward_functions()
    check("4 reward functions (default)", len(funcs) == 4, f"got {len(funcs)}")
    check("4 weights", len(weights) == 4, f"got {len(weights)}")
    check("Funcs and weights same length", len(funcs) == len(weights))
    check("All weights > 0", all(w > 0 for w in weights), f"{weights}")
    check(
        "Weights sum ? 1.0", abs(sum(weights) - 1.0) < 0.01, f"sum={sum(weights):.4f}"
    )

    # Check that each function is callable with completions
    completions = ["IX MAN WALK", "DOG CAT", "NOT CAN WANT"]
    for fn in funcs:
        try:
            result = fn(completions)
            check(
                f"  {fn.__name__} returns list of floats",
                isinstance(result, list) and len(result) == len(completions),
                f"{result}",
            )
            check(
                f"  {fn.__name__} values are floats",
                all(isinstance(v, float) for v in result),
            )
        except Exception as e:
            check(f"  {fn.__name__} callable", False, f"Exception: {e}")

    # Check with custom weights including gold_structure
    custom = {
        "weight_translation": 0.4,
        "weight_gold_structure": 0.4,
        "weight_format": 0.1,
        "weight_repetition": 0.1,
    }
    funcs2, weights2 = build_t2g_reward_functions(custom)
    check(
        "Custom (gold-structure): 4 functions", len(funcs2) == 4, f"got {len(funcs2)}"
    )
    check("Custom: weights sum to 1.0", abs(sum(weights2) - 1.0) < 0.01)

    # Check with viterbi weight
    custom_vit = {
        "weight_translation": 0.3,
        "weight_viterbi": 0.3,
        "weight_gold_structure": 0.3,
        "weight_format": 0.05,
        "weight_repetition": 0.05,
    }
    funcs3, weights3 = build_t2g_reward_functions(custom_vit)
    check("Custom (viterbi): 5 functions", len(funcs3) == 5, f"got {len(funcs3)}")
    check("Custom (viterbi): weights sum to 1.0", abs(sum(weights3) - 1.0) < 0.01)

    # Check old-style structural_dense still works
    custom_old = {
        "weight_translation": 0.5,
        "weight_structure": 0.5,
    }
    funcs4, weights4 = build_t2g_reward_functions(custom_old)
    check("Old-style (structure): 2 functions", len(funcs4) == 2, f"got {len(funcs4)}")
    check("Old-style: weights sum to 1.0", abs(sum(weights4) - 1.0) < 0.01)


def main() -> None:
    global PASS, FAIL
    print("=" * 60)
    print("TEST: Reward Functions")
    print("=" * 60)

    try:
        setup_rewards()
        test_translation_quality()
        test_structural_dense()
        test_format_reward()
        test_repetition_reward()
        test_gold_structure_reward()
        test_viterbi_distance_reward()
        test_build_reward_functions()
    except Exception as e:
        print(f"\n  !! CRASH: {e}")
        import traceback

        traceback.print_exc()
        FAIL += 1

    print(f"\n{'=' * 60}")
    print(f"RESULTS: {PASS} passed, {FAIL} failed, {PASS + FAIL} total")
    print(f"{'=' * 60}")
    sys.exit(0 if FAIL == 0 else 1)


if __name__ == "__main__":
    main()
