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

    grammarllm v0.5.0 migration: ``create_grammarllm_pipeline`` now returns
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


def test_masked_mass_tracking(tokenizer):
    """MaskedMassTracker tracks entropy and mass statistics (with track_diagnostics=True)."""
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
    # track_diagnostics=True is required for mass tracking
    processor = GlossVocabularyLogitsProcessor(
        mask, device="cpu", track_diagnostics=True
    )

    vocab_size = tokenizer.vocab_size
    for _ in range(5):
        scores = torch.randn(1, vocab_size) * 0.5
        dummy_input_ids = torch.zeros(1, 5, dtype=torch.long)
        _ = processor(dummy_input_ids, scores)

    stats = processor.get_masked_mass_stats()
    assert stats["total_steps"] == 5, f"Total steps = 5, got {stats['total_steps']}"
    assert (
        0.0 <= stats["avg_masked_mass"] <= 1.0
    ), f"Mass in [0,1]: {stats['avg_masked_mass']}"
    assert stats["avg_masked_mass"] > 0.0, "Mass > 0"
    assert (
        0.0 <= stats["avg_masked_entropy"] <= 12.0
    ), f"Entropy in [0, log(V)]: {stats['avg_masked_entropy']}"
    assert stats["avg_masked_entropy"] > 0.0, "Entropy > 0"

    processor.reset()
    stats_reset = processor.get_masked_mass_stats()
    assert stats_reset["total_steps"] == 0, "Reset clears total_steps"
    assert stats_reset["avg_masked_mass"] == 0.0, "Reset clears mass"


def test_pda_logits_processor_mass_tracking(tokenizer):
    """PDA logits processor mass tracking."""
    from src.grammar.gloss_grammar import create_grammarllm_pipeline
    from src.grammar.grammar_logits_processor import GrammarPDALogitsProcessor

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
    _, _, pda = create_grammarllm_pipeline(test_vocab, tokenizer)
    processor = GrammarPDALogitsProcessor(tokenizer, pda)

    vocab_size = tokenizer.vocab_size
    valid_init = processor.get_valid_tokens()
    assert len(valid_init) > 0, f"PDA initial valid tokens non-empty: {len(valid_init)}"

    for _ in range(5):
        scores = torch.randn(1, vocab_size) * 0.5
        dummy_input_ids = torch.zeros(1, 5, dtype=torch.long)
        _ = processor(dummy_input_ids, scores)

    stats = processor.get_masked_mass_stats()
    assert stats["total_steps"] == 5, f"PDA total steps = 5, got {stats['total_steps']}"
    assert 0.0 <= stats["avg_masked_mass"] <= 1.0
    assert stats["avg_masked_mass"] > 0.0, "PDA mass > 0"
    assert 0.0 <= stats["avg_masked_entropy"] <= 12.0
    assert stats["avg_masked_entropy"] > 0.0, "PDA entropy > 0"

    processor.reset()
    stats_reset = processor.get_masked_mass_stats()
    assert stats_reset["total_steps"] == 0, "PDA reset clears total_steps"

    # Accumulate again
    for _ in range(3):
        scores = torch.randn(1, vocab_size) * 0.5
        dummy_input_ids = torch.zeros(1, 5, dtype=torch.long)
        _ = processor(dummy_input_ids, scores)

    stats2 = processor.get_masked_mass_stats()
    assert (
        stats2["total_steps"] == 3
    ), f"PDA after reset + 3 steps: {stats2['total_steps']}"

    # reset_after=True path
    stats3 = processor.get_masked_mass_stats(reset_after=True)
    assert stats3["total_steps"] == 3, "reset_after returns correct total_steps"
    stats_after = processor.get_masked_mass_stats()
    assert stats_after["total_steps"] == 0, "reset_after clears counters"


def test_build_allowed_mask():
    """_build_allowed_mask edge cases."""
    from src.grammar.masked_mass_tracker import MaskedMassTracker

    tracker = MaskedMassTracker()
    tracker._init_masked_stats()

    # Empty set
    mask = tracker._build_allowed_mask(set(), vocab_size=10, device="cpu")
    assert mask.shape == (10,), "Empty set: shape correct"
    assert mask.sum().item() == 0, "Empty set: all False"
    assert mask.dtype == torch.bool, "Empty set: dtype bool"

    # Normal usage
    mask = tracker._build_allowed_mask({0, 5, 9}, vocab_size=10, device="cpu")
    assert mask[0].item() and mask[5].item() and mask[9].item(), "Positions 0,5,9 True"
    assert not mask[1].item(), "Position 1 False"
    assert mask.sum().item() == 3, "Sum=3"

    # Out of range
    mask = tracker._build_allowed_mask({-5, -1, 100, 200}, vocab_size=10, device="cpu")
    assert mask.sum().item() == 0, "Out-of-range: all False"

    # Mixed
    mask = tracker._build_allowed_mask({-1, 0, 5, 100}, vocab_size=10, device="cpu")
    assert mask[0].item() and mask[5].item(), "Mixed: only 0,5 True"
    assert mask.sum().item() == 2, "Mixed: sum=2"

    # vocab_size=0
    mask = tracker._build_allowed_mask({0, 1, 2}, vocab_size=0, device="cpu")
    assert mask.shape == (0,), "vocab_size=0: shape (0,)"
    assert mask.sum().item() == 0, "vocab_size=0: sum=0"

    # All valid
    mask = tracker._build_allowed_mask({0, 1, 2, 3, 4}, vocab_size=5, device="cpu")
    assert mask.sum().item() == 5, "All valid: sum=5"
