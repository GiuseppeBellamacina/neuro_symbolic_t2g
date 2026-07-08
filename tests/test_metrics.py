#!/usr/bin/env python3
"""Verify Metrics/Utils ? Test Script.

Validates:
  1. gloss validity checker (free text, repetition detection)
  2. ROUGE-L scoring
  3. Pass@1 and Pass@k computation
  4. Detailed metrics (dict structure, pass rate bounds)
  5. Compute reward breakdown from completions

Usage:
    python tests/test_metrics.py
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
    """Minimal setup required for reward-based metrics."""
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
        "NOT",
        "CAN",
        "WANT",
        "GO",
        "COME",
        "fs-JOHN",
    ]
    V = len(vocab)
    bigram = np.ones((V, V), dtype=np.float32) / V
    with tempfile.TemporaryDirectory() as tmp:
        mpath = str(Path(tmp) / "test_bigram.npy")
        save_transition_matrix(bigram, mpath)
    initialize_rewards(bigram, vocab)
    return vocab


def test_gloss_validity() -> None:
    print("\n-- 1. Gloss Validity --")
    from src.utils.metrics import check_gloss_validity

    # Valid gloss
    is_valid, err = check_gloss_validity("IX MAN WALK HOUSE")
    check("Clean gloss = valid", is_valid)
    check("  Error message empty", err == "", f"'{err}'")

    # Free text — should be invalid because tokens are not in gloss vocab
    is_valid2, err2 = check_gloss_validity("The man walks to the house")
    check("Free text = invalid", not is_valid2)
    check(
        "  Error type = out_of_vocab or free_text",
        "out_of_vocab" in err2 or "free_text" in err2,
        err2,
    )

    # Empty
    is_valid3, err3 = check_gloss_validity("")
    check("Empty = invalid", not is_valid3)
    check("  Error type = empty_output", err3 == "empty_output")

    # Repetitive
    is_valid4, err4 = check_gloss_validity("IX IX IX IX IX IX IX IX IX IX")
    check("Highly repetitive = invalid", not is_valid4)
    check("  Error type = excessive_repetition", "repetition" in err4.lower(), err4)

    # Fenced output (should still be valid after stripping)
    is_valid5, _ = check_gloss_validity("```gloss\nIX MAN WALK\n```")
    check("Fenced gloss = valid (after stripping)", is_valid5)


def test_rouge_l_score() -> None:
    print("\n-- 2. ROUGE-L Score --")
    from src.utils.metrics import rouge_l_score

    score = rouge_l_score("IX MAN WALK HOUSE", "IX MAN WALK HOUSE")
    check("Perfect match = 1.0", abs(score - 1.0) < 0.01, f"{score:.4f}")

    score2 = rouge_l_score("IX MAN GO HOUSE", "IX MAN WALK HOUSE")
    check("Partial match < 1.0", score2 < 1.0, f"{score2:.4f}")
    check("Partial match > 0.0", score2 > 0.0)

    score3 = rouge_l_score("DOG CAT BIRD", "IX MAN WALK")
    check("No overlap = 0.0", abs(score3 - 0.0) < 0.01, f"{score3:.4f}")

    score4 = rouge_l_score("", "IX MAN")
    check("Empty generated = 0.0", score4 == 0.0)

    score5 = rouge_l_score("IX MAN", "")
    check("Empty reference = 0.0", score5 == 0.0)


def test_pass_at_1() -> None:
    print("\n-- 3. Pass@1 --")
    from src.utils.metrics import compute_pass_at_1

    completions = ["IX MAN WALK HOUSE", "DOG CAT BIRD", "NOT CAN WANT"]
    references = ["IX MAN WALK HOUSE", "IX MAN WALK", "NOT CAN WANT"]
    rate = compute_pass_at_1(completions, references, threshold=0.3)
    check("Pass@1 rate in [0,1]", 0.0 <= rate <= 1.0, f"{rate:.4f}")
    check("Pass@1 rate > 0", rate > 0.0)

    # All perfect
    perfect = compute_pass_at_1(references, references, threshold=0.3)
    check("All-perfect Pass@1 = 1.0", abs(perfect - 1.0) < 0.01, f"{perfect:.4f}")

    # All bad
    bad = compute_pass_at_1(completions, ["A B C"] * 3, threshold=0.3)
    check("All-bad Pass@1 near 0", bad < 0.5, f"{bad:.4f}")

    # Edge: empty list
    empty = compute_pass_at_1([], [], threshold=0.3)
    check("Empty list = 0.0", empty == 0.0)


def test_pass_at_k() -> None:
    print("\n-- 4. Pass@k --")
    from src.utils.metrics import compute_pass_at_k

    # 2 prompts, each with 5 completions
    completions_per_prompt = [
        ["IX MAN WALK", "DOG CAT", "IX MAN WALK HOUSE", "NOT CAN", "GO COME"],
        ["DOG CAT BIRD", "DOG CAT", "IX MAN WALK", "NOT CAN", "GO COME"],
    ]
    references = ["IX MAN WALK HOUSE", "IX MAN WALK"]

    result = compute_pass_at_k(completions_per_prompt, references, k_values=(1, 3, 5))
    check("Returns dict with pass@1", "pass@1" in result, f"{result}")
    check("Returns dict with pass@3", "pass@3" in result)
    check("Returns dict with pass@5", "pass@5" in result)
    check(
        "pass@5 >= pass@1",
        result["pass@5"] >= result["pass@1"],
        f"pass@5={result['pass@5']:.4f}, pass@1={result['pass@1']:.4f}",
    )
    check("pass@1 in [0,1]", 0.0 <= result["pass@1"] <= 1.0)
    check("pass@5 in [0,1]", 0.0 <= result["pass@5"] <= 1.0)


def test_detailed_metrics() -> None:
    print("\n-- 5. Detailed Metrics --")
    from src.utils.metrics import compute_detailed_metrics

    completions = [
        "IX MAN WALK HOUSE",  # good match
        "DOG CAT BIRD FISH",  # bad match
        "NOT CAN WANT GO COME",  # some overlap
        "IX IX IX IX IX IX",  # repetitive (invalid)
        "The man walks home",  # free text (invalid)
    ]
    references = [
        "IX MAN WALK HOUSE",
        "IX MAN WALK",
        "NOT CAN WANT",
        "IX MAN GO",
        "IX MAN WALK",
    ]
    result = compute_detailed_metrics(completions, references)
    check("Has 'overall_pass_rate'", "overall_pass_rate" in result)
    check("Has 'overall_rouge_l'", "overall_rouge_l" in result)
    check("Has 'total_samples'", "total_samples" in result)
    check("Has 'valid_samples'", "valid_samples" in result)
    check("Has 'rouge_l_percentiles'", "rouge_l_percentiles" in result)
    check("Has 'error_distribution'", "error_distribution" in result)
    check("Total samples = 5", result["total_samples"] == 5)
    check(
        "Overall ROUGE-L in [0,1]",
        0.0 <= result["overall_rouge_l"] <= 1.0,
        f"{result['overall_rouge_l']:.4f}",
    )
    check("Pass rate in [0,1]", 0.0 <= result["overall_pass_rate"] <= 1.0)
    p = result["rouge_l_percentiles"]
    check("Percentiles sorted", p["25%"] <= p["50%"] <= p["75%"] <= p["90%"])


def test_reward_breakdown() -> None:
    print("\n-- 6. Reward Breakdown --")
    from src.utils.metrics import compute_reward_breakdown

    # Test completion-based version
    completions = ["IX MAN WALK HOUSE", "DOG CAT", "NOT CAN WANT"]
    result2 = compute_reward_breakdown(completions)
    check("Completion-based: has 9 keys", len(result2) >= 9, f"{result2.keys()}")
    check(
        "Completion-based: translation_quality_reward exists",
        "translation_quality_reward" in result2,
    )
    check(
        "Completion-based: structural_dense_reward exists",
        "structural_dense_reward" in result2,
    )
    check(
        "Completion-based: gloss_format_reward exists", "gloss_format_reward" in result2
    )
    check(
        "Completion-based: gloss_repetition_reward exists",
        "gloss_repetition_reward" in result2,
    )
    for k, v in result2.items():
        check(f"  {k} is float", isinstance(v, float), f"{v:.4f}")

    # Test filtering: only active components (weight > 0) should be returned
    filtered = compute_reward_breakdown(
        completions,
        reward_weights={
            "translation_quality_reward": 0.3,
            "gloss_format_reward": 0.1,
            # All others weight 0 → should be skipped
        },
    )
    check("Filtered: has 2 keys", len(filtered) == 2, f"{filtered.keys()}")
    check(
        "Filtered: translation_quality_reward exists",
        "translation_quality_reward" in filtered,
    )
    check("Filtered: gloss_format_reward exists", "gloss_format_reward" in filtered)
    check(
        "Filtered: gold_structure_reward NOT present",
        "gold_structure_reward" not in filtered,
    )


def main() -> None:
    global PASS, FAIL
    print("=" * 60)
    print("TEST: Metrics & Utils")
    print("=" * 60)

    try:
        setup_rewards()
        test_gloss_validity()
        test_rouge_l_score()
        test_pass_at_1()
        test_pass_at_k()
        test_detailed_metrics()
        test_reward_breakdown()
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
