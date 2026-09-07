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


class _TrieTokenizer:
    """Small deterministic tokenizer exposing prefix and multi-BPE cases."""

    eos_token_id = 9
    pad_token_id = 9
    vocab_size = 12

    _encodings = {
        "A": [1],
        "AB": [1, 2],
        "LONG": [3, 4],
        "B": [7],
        " A": [5],
        " AB": [5, 2],
        " LONG": [6, 4],
        " B": [8],
    }

    def encode(self, text, add_special_tokens=False):
        assert not add_special_tokens
        return list(self._encodings[text])


class _TrieMask:
    tokenizer = _TrieTokenizer()
    vocab = ["<BOS>", "<EOS>", "<UNK>", "A", "AB", "LONG", "B"]
    token_ids = {1, 2, 3, 4, 5, 6, 7, 8, 9}


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


def test_dual_root_trie_state_transitions_and_token_roles():
    """Pure API covers starts, continuation, boundaries, EOS, and invalid IDs."""
    from src.grammar.grammar_logits_processor import (
        DualRootGlossTrie,
        GlossTrieStatus,
        GlossTrieTokenRole,
    )

    trie = DualRootGlossTrie.from_vocabulary(_TrieMask.vocab, _TrieMask.tokenizer)
    state = trie.initial_state()
    assert hash(state)
    assert not trie.is_invalid(state)
    assert trie.allowed_token_ids(state) == (1, 3, 7, 9)

    # A is terminal and a prefix of AB: both continuation and boundaries survive.
    state = trie.advance(state, 1)
    assert trie.allowed_token_ids(state) == (2, 5, 6, 8, 9)
    assert trie.advance(state, 2).node.is_terminal

    # A second, space-prefixed, multi-BPE gloss follows a terminal first gloss.
    state = trie.advance(state, 6)
    assert trie.allowed_token_ids(state) == (4,)
    state = trie.advance(state, 4)
    assert state.node.is_terminal

    complete = trie.advance(state, 9, GlossTrieTokenRole.EOS)
    assert complete.status is GlossTrieStatus.COMPLETE
    assert trie.allowed_token_ids(complete) == ()

    # EOS and PAD remain semantically distinct even though their IDs are equal.
    padded = trie.advance(state, 9, GlossTrieTokenRole.PAD)
    assert padded.status is GlossTrieStatus.INVALID
    invalid = trie.advance(trie.initial_state(), 11)
    assert invalid.status is GlossTrieStatus.INVALID
    assert trie.is_invalid(invalid)
    assert trie.allowed_token_ids(invalid) == (1, 3, 7)


@pytest.mark.parametrize(
    "history",
    [
        [],  # first token
        [3],  # multi-BPE continuation
        [3, 4, 6],  # second space-prefixed gloss
        [1],  # terminal-prefix ambiguity
        [1, 9],  # historical post-EOS replay behavior
        [11],  # invalid prefix/root recovery
    ],
)
def test_processor_mask_matches_dual_root_api_state_by_state(history):
    """The production processor delegates every row mask to the pure API."""
    from src.grammar.grammar_logits_processor import GlossVocabularyLogitsProcessor

    processor = GlossVocabularyLogitsProcessor(_TrieMask(), device="cpu")
    prompt = [10]
    processor(torch.tensor([prompt]), torch.zeros(1, 12))
    input_ids = torch.tensor([prompt + history])
    output = processor(input_ids, torch.zeros(1, 12))
    actual = tuple(output[0].isfinite().nonzero(as_tuple=True)[0].tolist())
    state = processor.trie.state_for_tokens(history)
    assert actual == processor.trie.allowed_token_ids(state)


def test_processor_api_equivalence_for_row_specific_histories():
    """Batched rows are replayed independently from their generated histories."""
    from src.grammar.grammar_logits_processor import GlossVocabularyLogitsProcessor

    processor = GlossVocabularyLogitsProcessor(_TrieMask(), device="cpu")
    prompt = torch.tensor([[10], [10]])
    processor(prompt, torch.zeros(2, 12))
    histories = [[3], [1]]
    output = processor(
        torch.tensor([[10, *history] for history in histories]),
        torch.zeros(2, 12),
    )
    for row, history in enumerate(histories):
        actual = tuple(output[row].isfinite().nonzero(as_tuple=True)[0].tolist())
        expected = processor.trie.allowed_token_ids(
            processor.trie.state_for_tokens(history)
        )
        assert actual == expected


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
