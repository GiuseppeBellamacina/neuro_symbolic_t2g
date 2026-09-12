"""Symbolic repair of hallucinated lexical items in neural gloss output."""

from __future__ import annotations

from src.analysis.rule_baseline import RuleBaseline
from src.analysis.rule_repair import (
    RuleRepairer,
    align_source_to_gold,
    fit_morphology,
    is_derivable,
)


def _repairer(lexicon=None, deletions=(), morphology=None) -> RuleRepairer:
    return RuleRepairer(
        baseline=RuleBaseline(lexicon=lexicon or {}, deletions=frozenset(deletions)),
        morphology=morphology or {},
    )


# --- is_derivable -----------------------------------------------------------


def test_token_sharing_a_stem_with_a_source_word_is_derivable() -> None:
    assert is_derivable("OUTSOURCE", ["the", "factories", "are", "outsourcing"])


def test_marker_prefix_is_stripped_before_matching() -> None:
    """DESC-/X-/fs- markers must not block the stem comparison."""
    assert is_derivable("DESC-DONATION", ["living", "donations"])


def test_stem_shorter_than_the_threshold_is_not_matched() -> None:
    """'live' vs 'living' shares only 3 characters, below the 4-char threshold.

    Such cases are recovered through the fitted lexicon instead (which maps
    'living' -> 'DESC-LIVE' directly), not through the stem heuristic. The
    threshold is 4 because it was measured best: see the module docstring.
    """
    assert not is_derivable("DESC-LIVE", ["living", "donations"])


def test_hallucinated_token_is_not_derivable() -> None:
    """BLIGHT cannot come from any word of 'friendship does not mean blindness'."""
    assert not is_derivable(
        "BLIGHT", ["friendship", "does", "not", "mean", "blindness"]
    )


def test_punctuation_is_always_derivable() -> None:
    """The rule system has no opinion on punctuation worth enforcing."""
    assert is_derivable(",", ["anything"])
    assert is_derivable(".", [])


def test_short_stems_require_an_exact_source_word() -> None:
    """Stems shorter than the 4-char threshold only match a source word exactly.

    Documented limitation: an abbreviation whose expansion appears in the source
    (EU <- 'european union') is NOT considered derivable, so the repairer would
    defer to the transducer there. This is one of the few ways the step can
    regress a prompt; it is why insert/delete opcodes are left alone and why the
    measured break count stays low (5 of 2000 on the SFT pass, 0 elsewhere).
    """
    assert is_derivable("EU", ["eu", "body"])
    assert not is_derivable("EU", ["european", "union"])


# --- align_source_to_gold (single-row alignment, used by the glossary) ------


def test_align_source_to_gold_matches_stems_within_one_row() -> None:
    aligned = align_source_to_gold(
        "the factories are outsourcing work", "FACTORY BE OUTSOURCE WORK"
    )
    assert aligned == {
        "factories": "FACTORY",
        "outsourcing": "OUTSOURCE",
        "work": "WORK",
    }


def test_align_source_to_gold_ignores_unalignable_tokens() -> None:
    """Same as fit_morphology: a gold token sharing no stem with any source
    word contributes nothing (it cannot be a glossary hint either)."""
    assert align_source_to_gold("hello there", "COMPLETELY UNRELATED") == {}


def test_align_source_to_gold_is_per_row_not_fitted() -> None:
    """No aggregation across rows: this is exactly what fit_morphology's inner
    loop computes for a single row, exposed directly for the glossary, which
    needs THIS example's own gold — never a corpus-fitted table.

    "were" -> "BE" is NOT in the result: they share no 4-char stem (the
    4-char threshold is measured, see the module docstring), so a short
    function-word substitution like this is simply not something the
    stem heuristic can align — same behaviour as ``fit_morphology``.
    """
    row = {
        "text": "the statements were long",
        "gloss": "STATEMENT BE DESC-LONG",
    }
    assert align_source_to_gold(row["text"], row["gloss"]) == {
        "statements": "STATEMENT",
        "long": "DESC-LONG",
    }


# --- morphology fitting -----------------------------------------------------


def test_fit_morphology_learns_corpus_morphology_from_train() -> None:
    rows = [
        {
            "text": "the factories are outsourcing work",
            "gloss": "FACTORY BE OUTSOURCE WORK",
        },
        {"text": "outsourcing continues", "gloss": "OUTSOURCE CONTINUE"},
    ]
    morph = fit_morphology(rows)
    assert morph["outsourcing"] == "OUTSOURCE"
    assert morph["factories"] == "FACTORY"


def test_fit_morphology_picks_the_most_frequent_form() -> None:
    rows = [
        {"text": "the statements were long", "gloss": "STATEMENT BE DESC-LONG"},
        {"text": "statements again", "gloss": "STATEMENT DESC-AGAIN"},
        {"text": "statements vary", "gloss": "STATEMENTS VARY"},
    ]
    assert fit_morphology(rows)["statements"] == "STATEMENT"


def test_fit_morphology_ignores_unalignable_tokens() -> None:
    """A gloss token sharing no stem with any source word contributes nothing."""
    morph = fit_morphology([{"text": "hello there", "gloss": "COMPLETELY UNRELATED"}])
    assert morph == {}


# --- transduction -----------------------------------------------------------


def test_transduce_prefers_lexicon_then_morphology_then_uppercase() -> None:
    r = _repairer(
        lexicon={"cats": "CAT"},
        morphology={"running": "RUN"},
    )
    assert r.transduce("cats running fast") == "CAT RUN FAST"


def test_transduce_drops_deletion_set_words() -> None:
    r = _repairer(lexicon={}, deletions={"the"})
    assert r.transduce("the house") == "HOUSE"


# --- repair -----------------------------------------------------------------


def test_repair_replaces_a_hallucinated_token() -> None:
    """The measured headline case: the model invents a lexical item."""
    r = _repairer(morphology={"blindness": "BLINDNESS"})
    out = r.repair("MEAN BLIGHT BUT", "mean blindness but")
    assert out == "MEAN BLINDNESS BUT"


def test_repair_keeps_a_derivable_model_token() -> None:
    """When the model's token does come from the source, the model wins."""
    r = _repairer(morphology={"statements": "STATEMENT"})
    out = r.repair("COUNCIL STATEMENT ON", "council statements on")
    assert out == "COUNCIL STATEMENT ON"


def test_repair_does_not_touch_insertions_or_deletions() -> None:
    """Only one-for-one substitutions are repaired; length changes are the
    model's business (it handles function words far better than the rules)."""
    r = _repairer(morphology={"the": "THE", "house": "HOUSE"})
    # The model dropped a token relative to the rule output: keep the model's.
    assert r.repair("HOUSE", "the house") == "HOUSE"


def test_repair_is_identity_on_empty_completion() -> None:
    assert _repairer().repair("", "some text") == ""


def test_repair_is_identity_when_model_matches_the_rules() -> None:
    r = _repairer(morphology={"house": "HOUSE"})
    assert r.repair("HOUSE", "house") == "HOUSE"


def test_repairer_is_callable() -> None:
    r = _repairer(morphology={"blindness": "BLINDNESS"})
    assert r("MEAN BLIGHT", "mean blindness") == "MEAN BLINDNESS"
