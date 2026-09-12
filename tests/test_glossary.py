"""Train-time rare-word glossary (src/utils/glossary.py).

Covers frequency counting, per-example glossary construction (rare + gold-
aligned words only), block rendering, and the deterministic dropout draw.
The alignment itself (``align_source_to_gold``) is tested in
``tests/test_rule_repair.py``; this file tests the layer built on top of it.
"""

from __future__ import annotations

from collections import Counter

from src.utils.glossary import (
    build_example_glossary,
    compute_word_frequencies,
    format_glossary_block,
    should_include_glossary,
)

# ---------------------------------------------------------------------------
# compute_word_frequencies
# ---------------------------------------------------------------------------


def test_compute_word_frequencies_counts_across_the_corpus():
    freqs = compute_word_frequencies(["the cat sleeps", "the dog runs", "the cat runs"])
    assert freqs["the"] == 3
    assert freqs["cat"] == 2
    assert freqs["runs"] == 2
    assert freqs["sleeps"] == 1


def test_compute_word_frequencies_is_lowercased():
    freqs = compute_word_frequencies(["Blindness blindness BLINDNESS"])
    assert freqs["blindness"] == 3


# ---------------------------------------------------------------------------
# build_example_glossary
# ---------------------------------------------------------------------------


def test_build_example_glossary_keeps_only_rare_words():
    text = "the factories are outsourcing work"
    gold = "FACTORY BE OUTSOURCE WORK"
    # "work" is common in this corpus, "outsourcing"/"factories" are rare.
    frequencies = Counter(
        {"the": 500, "are": 500, "work": 200, "factories": 2, "outsourcing": 1}
    )
    glossary = build_example_glossary(text, gold, frequencies, max_freq=3)
    assert glossary == {"factories": "FACTORY", "outsourcing": "OUTSOURCE"}


def test_build_example_glossary_uses_this_row_own_gold_not_a_fitted_table():
    """Same source word, different gold answer in each of two calls: the
    glossary must reflect THIS example, not a corpus-averaged form."""
    frequencies = Counter({"blindness": 1})
    a = build_example_glossary(
        "only blindness remained", "ONLY BLINDNESS REMAIN", frequencies
    )
    b = build_example_glossary(
        "blindness struck again", "DESC-BLINDNESS STRIKE AGAIN", frequencies
    )
    assert a["blindness"] == "BLINDNESS"
    assert b["blindness"] == "DESC-BLINDNESS"


def test_build_example_glossary_word_absent_from_frequencies_is_treated_as_rarest():
    """A word never seen in the frequency table (freq 0) still counts as
    rare — ``.get(word, 0)`` must not accidentally exclude it."""
    glossary = build_example_glossary(
        "blindness", "BLINDNESS", frequencies={}, max_freq=3
    )
    assert glossary == {"blindness": "BLINDNESS"}


def test_build_example_glossary_empty_when_nothing_is_both_rare_and_aligned():
    # Common word: filtered by frequency even though it aligns.
    common_only = build_example_glossary(
        "the house",
        "THE HOUSE",
        frequencies=Counter({"the": 999, "house": 999}),
        max_freq=3,
    )
    assert common_only == {}

    # Rare but unalignable (hallucination-style mismatch): nothing to hint.
    unalignable = build_example_glossary(
        "hello there", "COMPLETELY UNRELATED", frequencies=Counter(), max_freq=3
    )
    assert unalignable == {}


# ---------------------------------------------------------------------------
# format_glossary_block
# ---------------------------------------------------------------------------


def test_format_glossary_block_empty_dict_is_empty_string():
    assert format_glossary_block({}) == ""


def test_format_glossary_block_renders_one_line_per_entry():
    block = format_glossary_block({"blindness": "BLINDNESS", "house": "HOUSE"})
    assert block == "Glossary:\nblindness -> BLINDNESS\nhouse -> HOUSE"


# ---------------------------------------------------------------------------
# should_include_glossary
# ---------------------------------------------------------------------------


def test_should_include_glossary_dropout_zero_always_true():
    assert should_include_glossary("any-id", dropout=0.0) is True


def test_should_include_glossary_dropout_one_always_false():
    assert should_include_glossary("any-id", dropout=1.0) is False


def test_should_include_glossary_is_deterministic_per_sample_id():
    first = should_include_glossary("sample-42", dropout=0.5)
    second = should_include_glossary("sample-42", dropout=0.5)
    assert first == second


def test_should_include_glossary_varies_across_sample_ids():
    """Not every id gets the same verdict at a mid-range dropout — otherwise
    dropout would be a no-op (all-or-nothing across the whole dataset)."""
    verdicts = {should_include_glossary(f"sample-{i}", dropout=0.5) for i in range(50)}
    assert verdicts == {True, False}
