"""Tests for the ASLG-PC12 context-free rule baseline.

The baseline exists to make model numbers interpretable on a corpus whose gloss
side is rule-derivable, so these tests pin the two properties that matter
scientifically: (a) it is estimated from train data only, and (b) it stays
context-free.
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from src.analysis.rule_baseline import (
    DEFAULT_DELETION_THRESHOLD,
    DEFAULT_MIN_COUNT,
    RuleBaseline,
    fit,
    fit_from_split,
    non_copy_token_accuracy,
)

TRAIN = [
    {"text": "the cat sat", "gloss": "CAT SIT"},
    {"text": "the dog sat", "gloss": "DOG SIT"},
    {"text": "the cat ran", "gloss": "CAT RUN"},
]


def _fit_tiny(rows=TRAIN, **kwargs):
    kwargs.setdefault("min_count", 1)
    return fit(rows, **kwargs)


def test_lexicon_uses_only_length_aligned_pairs():
    # "sat"->"SIT" and "ran"->"RUN" are only learnable from aligned pairs; the
    # 3-vs-2 token rows above are not aligned, so nothing should be learned.
    fitted = _fit_tiny()
    assert "sat" not in fitted.lexicon
    aligned = fit([{"text": "cat sat", "gloss": "CAT SIT"}], min_count=1)
    assert aligned.lexicon["sat"] == "SIT"


def test_unknown_words_fall_back_to_uppercase():
    fitted = fit([{"text": "cat sat", "gloss": "CAT SIT"}], min_count=1)
    assert fitted.apply("cat flew") == "CAT FLEW"


def test_deletion_set_learns_systematically_dropped_words():
    fitted = _fit_tiny()
    # "the" never survives into the gloss in any training row.
    assert "the" in fitted.deletions
    assert fitted.apply("the cat") == "CAT"


def test_min_count_suppresses_rare_deletions():
    # With a high min_count no word clears the frequency bar, so nothing is dropped.
    fitted = fit(TRAIN, min_count=99)
    assert fitted.deletions == frozenset()
    assert fitted.apply("the cat") == "THE CAT"


def test_apply_is_context_free_and_position_preserving():
    # The same word maps identically regardless of neighbours, and no reordering
    # happens. This is the property the model is expected to beat.
    fitted = fit([{"text": "cat sat", "gloss": "CAT SIT"}], min_count=1)
    assert fitted.apply("sat cat") == "SIT CAT"
    assert fitted.apply("cat cat") == "CAT CAT"


def test_callable_matches_apply():
    fitted = _fit_tiny()
    assert fitted("the cat") == fitted.apply("the cat")


def test_empty_and_whitespace_input():
    fitted = _fit_tiny()
    assert fitted.apply("") == ""
    assert fitted.apply("   ") == ""


def test_fit_from_split_refuses_non_train_splits():
    dataset = {"train": TRAIN, "test": TRAIN}
    with pytest.raises(ValueError, match="training data only"):
        fit_from_split(dataset, split="test")
    assert isinstance(fit_from_split(dataset, min_count=1), RuleBaseline)


@pytest.mark.parametrize("threshold", [0.0, -0.1, 1.5])
def test_invalid_deletion_threshold_rejected(threshold):
    with pytest.raises(ValueError, match="deletion_threshold"):
        fit(TRAIN, deletion_threshold=threshold)


def test_invalid_min_count_rejected():
    with pytest.raises(ValueError, match="min_count"):
        fit(TRAIN, min_count=0)


def test_defaults_are_the_audited_configuration():
    # These defaults reproduce the numbers quoted in docs/RECOVERY_REPORT.md.
    assert DEFAULT_MIN_COUNT == 30
    assert DEFAULT_DELETION_THRESHOLD == 0.85


def test_baseline_is_frozen():
    # A fitted baseline is a protocol artifact: it must not be mutated after fit.
    fitted = _fit_tiny()
    with pytest.raises(FrozenInstanceError):
        fitted.deletions = frozenset()  # type: ignore[misc]


# --- non_copy_token_accuracy -------------------------------------------------


def test_non_copy_ignores_tokens_obtainable_by_uppercasing_the_source():
    # Every reference token is a source copy, so there is nothing to score.
    acc, hits, total = non_copy_token_accuracy(["CAT SAT"], ["cat sat"], ["CAT SAT"])
    assert (hits, total) == (0, 0)
    assert acc == 0.0


def test_non_copy_scores_only_the_non_copy_tokens():
    # "BE" cannot be obtained from the source, so it is the only scored token.
    acc, hits, total = non_copy_token_accuracy(["CAT BE"], ["cat is"], ["CAT BE"])
    assert (hits, total) == (1, 1)
    assert acc == 1.0
    acc, hits, total = non_copy_token_accuracy(["CAT IS"], ["cat is"], ["CAT BE"])
    assert (hits, total) == (0, 1)
    assert acc == 0.0


def test_non_copy_is_case_sensitive():
    # Lowercase English echoing must NOT earn credit; this is the whole point.
    acc, _, total = non_copy_token_accuracy(["cat be"], ["cat is"], ["CAT BE"])
    assert total == 1
    assert acc == 0.0


def test_non_copy_is_order_insensitive_but_multiset_exact():
    # Order is covered by exact match, so this metric only checks availability...
    acc, _, _ = non_copy_token_accuracy(["BE CAT"], ["cat is"], ["CAT BE"])
    assert acc == 1.0
    # ...but a token required twice must be produced twice.
    acc, hits, total = non_copy_token_accuracy(["BE"], ["is is"], ["BE BE"])
    assert (hits, total) == (1, 2)
    assert acc == 0.5


def test_non_copy_aggregates_across_the_corpus():
    acc, hits, total = non_copy_token_accuracy(
        ["CAT BE", "DOG IS"], ["cat is", "dog is"], ["CAT BE", "DOG BE"]
    )
    assert (hits, total) == (1, 2)
    assert acc == 0.5


def test_non_copy_rejects_length_mismatch():
    with pytest.raises(ValueError, match="length mismatch"):
        non_copy_token_accuracy(["A"], ["a", "b"], ["A"])
