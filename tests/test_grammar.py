#!/usr/bin/env python3
"""Test grammar and constrained decoding components.

Validates:
  1. GlossVocabularyMask maps gloss tokens to tokenizer IDs
  2. GlossVocabularyLogitsProcessor correctly masks non-gloss tokens
  3. decode_to_glosses (method on GlossVocabularyMask) works correctly
  4. Grammar build via create_grammarllm_pipeline (PDA)
  5. Masked mass tracking (with track_diagnostics=True)
  6. PDA logits processor mass tracking
  7. _build_allowed_mask edge cases

Uses the ``tokenizer`` fixture from conftest.py.
"""

from __future__ import annotations

import pytest
import torch


def test_gloss_vocabulary_mask(tokenizer):
    """GlossVocabularyMask maps gloss tokens to tokenizer IDs."""
    from src.grammar.gloss_grammar import GlossVocabularyMask

    test_vocab = [
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
    ]
    mask = GlossVocabularyMask(test_vocab, tokenizer)
    assert len(mask.token_ids) > 0, f"Token IDs non-empty: {len(mask.token_ids)}"
    assert mask.is_allowed(mask.eos_token_id), "EOS allowed in mask"
    allowed = mask.get_allowed_token_ids()
    assert len(allowed) > 0, "Allowed IDs non-empty"


def test_logits_processor(tokenizer):
    """GlossVocabularyLogitsProcessor masks non-gloss tokens correctly."""
    from src.grammar.gloss_grammar import GlossVocabularyMask
    from src.grammar.grammar_logits_processor import GlossVocabularyLogitsProcessor

    test_vocab = [
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
    ]
    mask = GlossVocabularyMask(test_vocab, tokenizer)
    processor = GlossVocabularyLogitsProcessor(mask, device="cpu")

    vocab_size = tokenizer.vocab_size
    scores = torch.randn(1, vocab_size) * 0.5
    dummy_input_ids = torch.zeros(1, 5, dtype=torch.long)
    result = processor(dummy_input_ids, scores)

    assert (
        result.shape == scores.shape
    ), f"Shape preserved: {result.shape} vs {scores.shape}"
    disallowed = result[0] < -1e10
    assert disallowed.sum() > 0, "Some tokens are masked (-inf)"
    allowed = result[0] > -1e10
    assert allowed.sum() > 0, "Some tokens are allowed (not -inf)"


def test_decode_to_glosses(tokenizer):
    """GlossVocabularyMask.decode_to_glosses converts token IDs to gloss strings."""
    from src.grammar.gloss_grammar import GlossVocabularyMask

    test_vocab = [
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
    ]
    mask = GlossVocabularyMask(test_vocab, tokenizer)

    # Encode a simple gloss sequence
    text = "IX MAN WALK"
    ids = tokenizer.encode(text, add_special_tokens=False)
    result = mask.decode_to_glosses(ids)
    assert isinstance(result, list), f"Returns list, got {type(result)}"
    assert len(result) > 0, f"Non-empty result: {len(result)}"


def test_grammar_build(tokenizer):
    """Build LL(1) grammar and PDA via create_grammarllm_pipeline.

    ``create_grammarllm_pipeline`` returns
    ``(pdas, streamer, pda)`` where ``pdas`` is a list of base PDA templates
    (was ``(logit_processor, streamer, pda)`` in v0.4.x).
    """
    from src.grammar.gloss_grammar import create_grammarllm_pipeline

    test_vocab = [
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
    ]
    pdas, streamer, pda = create_grammarllm_pipeline(test_vocab, tokenizer)

    assert pda is not None, "PDA created"
    assert isinstance(pdas, list), f"pdas is a list, got {type(pdas)}"
    assert len(pdas) > 0, "pdas list non-empty"
    assert pdas[0] is pda, "pda is pdas[0] (primary PDA)"
    assert streamer is not None, "Streamer created"


def test_exact_per_row_diagnostics_and_invariants():
    from src.grammar.masked_mass_tracker import MaskedMassTracker

    tracker = MaskedMassTracker()
    tracker._init_masked_stats()
    logits = torch.log(torch.tensor([[0.1, 0.2, 0.7], [0.6, 0.3, 0.1]]))
    allowed = torch.tensor([[True, True, False], [False, True, True]])
    tracker._track_masked_stats(logits, allowed)
    stats = tracker.get_diagnostics()
    assert stats["allowed_mass_mean"] == pytest.approx(0.35)
    assert stats["removed_mass_mean"] == pytest.approx(0.65)
    assert stats["allowed_mass_min"] == pytest.approx(0.3)
    assert stats["allowed_mass_mean"] + stats["removed_mass_mean"] == pytest.approx(1)
    assert (
        torch.exp(torch.tensor(stats["log_allowed_mass_mean"]))
        <= stats["allowed_mass_mean"]
    )
    assert stats["active_rows"] == 2
    assert stats["steps"] == 1


def test_b1_post_eos_exclusion_and_interval_reset():
    from src.grammar.masked_mass_tracker import MaskedMassTracker

    tracker = MaskedMassTracker()
    tracker._init_masked_stats()
    scores = torch.zeros(1, 3)
    allowed = torch.tensor([[True, False, False]])
    tracker._track_masked_stats(scores, allowed, torch.tensor([True]))
    tracker._track_masked_stats(scores, allowed, torch.tensor([False]))
    tracker._track_masked_stats(scores, allowed, torch.tensor([True]))
    stats = tracker.get_diagnostics(reset_after=True)
    assert stats["active_rows"] == 2
    assert stats["steps"] == 2
    assert tracker.get_diagnostics()["steps"] == 0
