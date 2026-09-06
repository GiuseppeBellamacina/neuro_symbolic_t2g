#!/usr/bin/env python3
"""Test metrics and utility functions for T2G evaluation.

Validates:
  1. Gloss validity checker (free text, repetition detection)
  2. ROUGE-L scoring
  3. Pass@1 and Pass@k computation
  4. Detailed metrics (dict structure, pass rate bounds)
  5. chrF2 via sacrebleu (identical → 100, empty → 0)
  6. Token-level gloss F1 (identical, disjoint, partial, case-insensitive)
  7. BLEU via sacrebleu (short non-zero, identical → 1.0)
  8. Corpus-level metrics (BLEU / chrF / gloss F1)
  9. Seeded sampling function (None → all, seeded reproducible)
  10. Reward breakdown with direct references (no gold-gloss registry)

Tests 1-4 and 10 use the ``reward_setup`` fixture from conftest.py.
All other tests are pure functions — no network, no dataset.
"""

from __future__ import annotations

import pytest

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
    from src.utils.metrics import compute_pass_at_k

    completions = ["IX MAN WALK HOUSE", "DOG CAT BIRD", "NOT CAN WANT"]
    references = ["IX MAN WALK HOUSE", "IX MAN WALK", "NOT CAN WANT"]
    nested = [[c] for c in completions]

    rate = compute_pass_at_k(nested, references, k_values=(1,), threshold=0.3)["pass@1"]
    assert 0.0 <= rate <= 1.0, f"Pass@1 rate in [0,1], got {rate:.4f}"
    assert rate > 0.0

    valid_references = ["IX MAN WALK HOUSE", "IX MAN WALK", "IX MAN WALK"]
    perfect = compute_pass_at_k(
        [[c] for c in valid_references], valid_references, k_values=(1,), threshold=0.3
    )["pass@1"]
    assert abs(perfect - 1.0) < 0.01, f"All-perfect Pass@1 = 1.0, got {perfect:.4f}"

    bad = compute_pass_at_k(
        [[c] for c in completions], ["A B C"] * 3, k_values=(1,), threshold=0.3
    )["pass@1"]
    assert bad < 0.5, f"All-bad Pass@1 near 0, got {bad:.4f}"

    assert (
        compute_pass_at_k([], [], k_values=(1,), threshold=0.3)["pass@1"] == 0.0
    ), "Empty list = 0.0"


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


def test_pass_at_k_uses_standard_estimator_and_validity(reward_setup):
    from src.utils.metrics import compute_pass_at_k

    # Two successes among five gives 1-C(3,2)/C(5,2) = 0.7.
    comps = [["IX MAN WALK", "IX MAN WALK", "DOG CAT", "DOG CAT", "DOG CAT"]]
    assert compute_pass_at_k(comps, ["IX MAN WALK"], (2,))["pass@2"] == pytest.approx(
        0.7
    )
    with pytest.raises(ValueError, match="aligned"):
        compute_pass_at_k(comps, [], (1,))


def test_normalized_exact_and_edit_similarity():
    from src.utils.metrics import normalized_edit_similarity, normalized_exact_match

    assert normalized_exact_match("```gloss\nIX   MAN\n```", "ix man") == 1.0
    assert normalized_edit_similarity("IX WALK", "IX MAN WALK") == pytest.approx(2 / 3)
    assert 0.0 <= normalized_edit_similarity("A", "B") <= 1.0


def test_pass_at_k_matches_success_criterion_at_1(reward_setup):
    """Pass@1 uses ROUGE-L >= 0.3 AND lexical validity."""
    from src.utils.metrics import check_gloss_validity, compute_pass_at_k, rouge_l_score

    completions = ["IX MAN WALK HOUSE", "DOG CAT BIRD", "NOT CAN WANT"]
    references = ["IX MAN WALK HOUSE", "IX MAN WALK", "NOT CAN WANT"]
    nested = [[c] for c in completions]

    p1 = compute_pass_at_k(nested, references, k_values=(1,), threshold=0.3)["pass@1"]
    manual = sum(
        1
        for c, r in zip(completions, references)
        if rouge_l_score(c, r) >= 0.3 and check_gloss_validity(c)[0]
    ) / len(completions)
    assert abs(p1 - manual) < 1e-9, f"pass@1 equal: {p1} vs {manual}"


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
# 6. chrF2 via sacrebleu
# ---------------------------------------------------------------------------


def test_chrf_identical_is_100():
    from src.utils.metrics import chrf_score

    score = chrf_score("IX MAN WALK HOUSE", "IX MAN WALK HOUSE")
    assert abs(score - 100.0) < 1e-6, f"Identical = 100, got {score:.4f}"


def test_chrf_known_partial_and_empty():
    from src.utils.metrics import chrf_score

    # Same tokens in different order: char-level score is high but < 100.
    score = chrf_score("MAN WALK IX", "IX MAN WALK")
    assert 0.0 < score < 100.0, f"Reordered gloss in (0, 100), got {score:.4f}"

    assert chrf_score("", "IX MAN") == 0.0, "Empty hypothesis = 0"
    assert chrf_score("IX MAN", "") == 0.0, "Empty reference = 0"


def test_corpus_chrf_identical_is_100():
    from src.utils.metrics import corpus_chrf

    hyps = ["IX MAN WALK", "DOG CAT"]
    refs = ["IX MAN WALK", "DOG CAT"]
    assert abs(corpus_chrf(hyps, refs) - 100.0) < 1e-6
    assert corpus_chrf([], []) == 0.0, "Empty corpus = 0"


# ---------------------------------------------------------------------------
# 7. Token-level gloss F1
# ---------------------------------------------------------------------------


def test_gloss_f1_known_cases():
    from src.utils.metrics import gloss_f1

    # Identical → 1.0
    assert abs(gloss_f1("IX MAN WALK HOUSE", "IX MAN WALK HOUSE") - 1.0) < 1e-9

    # Disjoint → 0.0
    assert gloss_f1("DOG CAT BIRD", "IX MAN WALK") == 0.0

    # Partial: gen={IX,MAN,WALK,HOUSE}, ref={IX,MAN,WALK}
    # precision=3/4, recall=3/3 → F1 = 2·0.75·1.0/(0.75+1.0) = 6/7 ≈ 0.8571
    f = gloss_f1("IX MAN WALK HOUSE", "IX MAN WALK")
    assert abs(f - 6.0 / 7.0) < 1e-9, f"Partial F1 = 6/7, got {f:.6f}"

    # Case-insensitive comparison
    assert abs(gloss_f1("ix man walk", "IX MAN WALK") - 1.0) < 1e-9

    # Empty inputs → 0.0
    assert gloss_f1("", "IX MAN") == 0.0
    assert gloss_f1("IX MAN", "") == 0.0


def test_corpus_gloss_f1():
    from src.utils.metrics import corpus_gloss_f1

    hyps = ["IX MAN WALK", "MAN WALK"]
    refs = ["IX MAN WALK", "IX MAN WALK"]
    result = corpus_gloss_f1(hyps, refs)

    # Sentence scores: 1.0 and 2·(2/2)·(2/3)/(1+2/3) = 0.8 → mean = 0.9
    assert abs(result["sentence_mean"] - 0.9) < 1e-9
    # Micro: tokens gen={IX,MAN,WALK,MAN,WALK}, ref={IX,MAN,WALK,IX,MAN,WALK}
    # overlap=5, precision=5/5, recall=5/6 → F1 = 10/11 ≈ 0.9091
    assert abs(result["micro"] - 10.0 / 11.0) < 1e-9

    assert corpus_gloss_f1([], []) == {"micro": 0.0, "sentence_mean": 0.0}


# ---------------------------------------------------------------------------
# 8. BLEU via sacrebleu
# ---------------------------------------------------------------------------


def test_bleu_sentence_short_nonzero():
    from src.utils.metrics import bleu_sentence

    # Identical → 1.0 (sacrebleu 100/100)
    assert abs(bleu_sentence("IX MAN WALK HOUSE", "IX MAN WALK HOUSE") - 1.0) < 1e-6

    # Short hypothesis: unigram precision=1, brevity penalty exp(1-2/1) ≈ 0.3679
    score = bleu_sentence("WALK", "WALK HOUSE")
    assert score > 0.0, f"Short partial BLEU non-zero, got {score:.4f}"
    assert score < 1.0, f"Short partial BLEU < 1.0, got {score:.4f}"

    # Disjoint → 0.0
    assert bleu_sentence("DOG CAT", "IX MAN") == 0.0

    # Empty inputs → 0.0
    assert bleu_sentence("", "IX MAN") == 0.0
    assert bleu_sentence("IX MAN", "") == 0.0


def test_bleu_corpus():
    from src.utils.metrics import bleu_corpus

    hyps = ["IX MAN WALK", "DOG CAT"]
    refs = ["IX MAN WALK", "DOG CAT"]
    assert abs(bleu_corpus(hyps, refs) - 1.0) < 1e-6

    assert bleu_corpus([], []) == 0.0, "Empty corpus = 0"
    with pytest.raises(ValueError, match="equal"):
        bleu_corpus(["A"], [])


# ---------------------------------------------------------------------------
# 9. Seeded sampling (eval max_samples)
# ---------------------------------------------------------------------------


def test_seeded_sample_indices_all():
    from src.utils.metrics import seeded_sample_indices

    assert seeded_sample_indices(10, None) == list(range(10))
    assert seeded_sample_indices(10, 15) == list(range(10))
    assert seeded_sample_indices(0, None) == []


def test_seeded_sample_indices_reproducible():
    from src.utils.metrics import seeded_sample_indices

    total, n = 100, 20
    s1 = seeded_sample_indices(total, n, seed=42)
    s2 = seeded_sample_indices(total, n, seed=42)
    s3 = seeded_sample_indices(total, n, seed=1)

    assert s1 == s2, "Same seed → same sample"
    assert len(s1) == n, f"Sample size {n}, got {len(s1)}"
    assert len(set(s1)) == n, "Indices are distinct"
    assert all(0 <= i < total for i in s1), "Indices within range"
    assert s1 == sorted(s1), "Indices are sorted"
    assert s1 != s3, "Different seed → different sample"


# ---------------------------------------------------------------------------
# 10. Production reward breakdown
# ---------------------------------------------------------------------------


def test_reward_breakdown(reward_setup):
    from src.rewards.t2g_rewards import initialize_rewards
    from src.utils.metrics import compute_reward_breakdown

    vocab, _, _ = reward_setup
    initialize_rewards(vocab)
    completions = ["IX MAN WALK HOUSE", "DOG CAT", "NOT CAN WANT"]
    references = ["IX MAN WALK HOUSE", "IX MAN WALK", "NOT CAN WANT"]
    result = compute_reward_breakdown(completions, references=references)
    assert set(result) == {"edit_validity_reward", "edit_validity_diagnostic"}
    assert result["edit_validity_reward"] == pytest.approx(1 / 3)
    assert compute_reward_breakdown(completions) == {}

    disabled = compute_reward_breakdown(
        completions,
        references=references,
        reward_weights={"edit_validity_reward": 0.0},
    )
    assert disabled == {}
