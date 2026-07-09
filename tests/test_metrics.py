#!/usr/bin/env python3
"""Test metrics and utility functions for T2G evaluation.

Validates:
  1. Gloss validity checker (free text, repetition detection)
  2. ROUGE-L scoring
  3. Pass@1 and Pass@k computation
  4. Detailed metrics (dict structure, pass rate bounds)
  5. Compute reward breakdown from completions

All tests use the ``reward_setup`` fixture from conftest.py.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# 1. Gloss validity
# ---------------------------------------------------------------------------


def test_gloss_validity(reward_setup):
    from src.utils.metrics import check_gloss_validity

    is_valid, err = check_gloss_validity("IX MAN WALK HOUSE")
    assert is_valid, "Clean gloss = valid"
    assert err == "", f"Error message empty, got '{err}'"

    is_valid2, err2 = check_gloss_validity("The man walks to the house")
    assert not is_valid2, "Free text = invalid"
    assert "out_of_vocab" in err2 or "free_text" in err2, f"Error type: {err2}"

    is_valid3, err3 = check_gloss_validity("")
    assert not is_valid3, "Empty = invalid"
    assert err3 == "empty_output", f"Error type = empty_output, got {err3}"

    is_valid4, err4 = check_gloss_validity("IX IX IX IX IX IX IX IX IX IX")
    assert not is_valid4, "Highly repetitive = invalid"
    assert "repetition" in err4.lower(), f"Error type = repetition, got {err4}"

    is_valid5, _ = check_gloss_validity("```gloss\nIX MAN WALK\n```")
    assert is_valid5, "Fenced gloss = valid (after stripping)"


# ---------------------------------------------------------------------------
# 2. ROUGE-L score
# ---------------------------------------------------------------------------


def test_rouge_l_score(reward_setup):
    from src.utils.metrics import rouge_l_score

    score = rouge_l_score("IX MAN WALK HOUSE", "IX MAN WALK HOUSE")
    assert abs(score - 1.0) < 0.01, f"Perfect match = 1.0, got {score:.4f}"

    score2 = rouge_l_score("IX MAN GO HOUSE", "IX MAN WALK HOUSE")
    assert score2 < 1.0, f"Partial match < 1.0, got {score2:.4f}"
    assert score2 > 0.0

    score3 = rouge_l_score("DOG CAT BIRD", "IX MAN WALK")
    assert abs(score3 - 0.0) < 0.01, f"No overlap = 0.0, got {score3:.4f}"

    assert rouge_l_score("", "IX MAN") == 0.0, "Empty generated = 0.0"
    assert rouge_l_score("IX MAN", "") == 0.0, "Empty reference = 0.0"


# ---------------------------------------------------------------------------
# 3. Pass@1
# ---------------------------------------------------------------------------


def test_pass_at_1(reward_setup):
    from src.utils.metrics import compute_pass_at_1

    completions = ["IX MAN WALK HOUSE", "DOG CAT BIRD", "NOT CAN WANT"]
    references = ["IX MAN WALK HOUSE", "IX MAN WALK", "NOT CAN WANT"]
    rate = compute_pass_at_1(completions, references, threshold=0.3)
    assert 0.0 <= rate <= 1.0, f"Pass@1 rate in [0,1], got {rate:.4f}"
    assert rate > 0.0

    perfect = compute_pass_at_1(references, references, threshold=0.3)
    assert abs(perfect - 1.0) < 0.01, f"All-perfect Pass@1 = 1.0, got {perfect:.4f}"

    bad = compute_pass_at_1(completions, ["A B C"] * 3, threshold=0.3)
    assert bad < 0.5, f"All-bad Pass@1 near 0, got {bad:.4f}"

    assert compute_pass_at_1([], [], threshold=0.3) == 0.0, "Empty list = 0.0"


# ---------------------------------------------------------------------------
# 4. Pass@k
# ---------------------------------------------------------------------------


def test_pass_at_k(reward_setup):
    from src.utils.metrics import compute_pass_at_k

    completions_per_prompt = [
        ["IX MAN WALK", "DOG CAT", "IX MAN WALK HOUSE", "NOT CAN", "GO COME"],
        ["DOG CAT BIRD", "DOG CAT", "IX MAN WALK", "NOT CAN", "GO COME"],
    ]
    references = ["IX MAN WALK HOUSE", "IX MAN WALK"]

    result = compute_pass_at_k(completions_per_prompt, references, k_values=(1, 3, 5))
    assert "pass@1" in result, f"Returns dict with pass@1: {result}"
    assert "pass@3" in result
    assert "pass@5" in result
    assert (
        result["pass@5"] >= result["pass@1"]
    ), f"pass@5 >= pass@1: {result['pass@5']:.4f} vs {result['pass@1']:.4f}"
    assert 0.0 <= result["pass@1"] <= 1.0
    assert 0.0 <= result["pass@5"] <= 1.0


# ---------------------------------------------------------------------------
# 5. Detailed metrics
# ---------------------------------------------------------------------------


def test_detailed_metrics(reward_setup):
    from src.utils.metrics import compute_detailed_metrics

    completions = [
        "IX MAN WALK HOUSE",
        "DOG CAT BIRD FISH",
        "NOT CAN WANT GO COME",
        "IX IX IX IX IX IX",
        "The man walks home",
    ]
    references = [
        "IX MAN WALK HOUSE",
        "IX MAN WALK",
        "NOT CAN WANT",
        "IX MAN GO",
        "IX MAN WALK",
    ]
    result = compute_detailed_metrics(completions, references)
    assert "overall_pass_rate" in result
    assert "overall_rouge_l" in result
    assert "total_samples" in result
    assert "valid_samples" in result
    assert "rouge_l_percentiles" in result
    assert "error_distribution" in result
    assert result["total_samples"] == 5
    assert 0.0 <= result["overall_rouge_l"] <= 1.0
    assert 0.0 <= result["overall_pass_rate"] <= 1.0
    p = result["rouge_l_percentiles"]
    assert p["25%"] <= p["50%"] <= p["75%"] <= p["90%"], "Percentiles sorted"


# ---------------------------------------------------------------------------
# 6. Reward breakdown
# ---------------------------------------------------------------------------


def test_reward_breakdown(reward_setup):
    from src.utils.metrics import compute_reward_breakdown

    completions = ["IX MAN WALK HOUSE", "DOG CAT", "NOT CAN WANT"]
    result = compute_reward_breakdown(completions)
    assert (
        len(result) >= 9
    ), f"Completion-based: has >=9 keys, got {list(result.keys())}"
    assert "translation_quality_reward" in result
    assert "structural_dense_reward" in result
    assert "gloss_format_reward" in result
    assert "gloss_repetition_reward" in result
    for k, v in result.items():
        assert isinstance(v, float), f"{k} is float, got {type(v)}"

    # Test filtering: only active components (weight > 0)
    filtered = compute_reward_breakdown(
        completions,
        reward_weights={
            "translation_quality_reward": 0.3,
            "gloss_format_reward": 0.1,
        },
    )
    assert len(filtered) == 2, f"Filtered: has 2 keys, got {list(filtered.keys())}"
    assert "translation_quality_reward" in filtered
    assert "gloss_format_reward" in filtered
    assert "gold_structure_reward" not in filtered
