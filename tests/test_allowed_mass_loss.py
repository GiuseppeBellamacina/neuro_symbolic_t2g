"""Tests for the allowed-mass auxiliary loss (candidate-set log-marginal).

The properties pinned here are the ones that make the term safe to add to a
research pipeline: it is exactly inert at weight 0 (so standard SFT is
unchanged), it is mathematically the log-marginal of the allowed set, it is
numerically stable in the log domain, and it is flat over the allowed set — the
last one is a *limitation* and is asserted deliberately, because it is the
reason the term cannot be expected to improve exact match.
"""

from __future__ import annotations

import math

import pytest
import torch

from src.training.allowed_mass_loss import allowed_mass_loss

VOCAB = 6


def _logits(*rows: list[float]) -> torch.Tensor:
    return torch.tensor(list(rows), dtype=torch.float32)


def _mask(*rows: list[int]) -> torch.Tensor:
    return torch.tensor(list(rows), dtype=torch.bool)


def test_weight_zero_is_exactly_inert():
    """The contract that protects standard SFT: lambda=0 contributes nothing."""
    logits = torch.randn(4, VOCAB)
    mask = torch.zeros(4, VOCAB, dtype=torch.bool)
    mask[:, :2] = True
    loss, diag = allowed_mass_loss(logits, mask, weight=0.0)
    assert loss.item() == 0.0
    assert diag.effective_weight == 0.0


def test_matches_closed_form_log_marginal():
    # Uniform logits over 6 tokens, 2 allowed -> P(allowed) = 2/6.
    logits = torch.zeros(1, VOCAB)
    mask = _mask([1, 1, 0, 0, 0, 0])
    loss, diag = allowed_mass_loss(logits, mask, weight=1.0)
    expected = -math.log(2.0 / VOCAB)
    assert loss.item() == pytest.approx(expected, rel=1e-5)
    assert diag.mean_allowed_mass == pytest.approx(2.0 / VOCAB, rel=1e-5)


def test_loss_is_zero_when_all_mass_is_already_allowed():
    """Nothing to optimize once the model is inside the constraint."""
    logits = _logits([10.0, -10.0, -10.0, -10.0, -10.0, -10.0])
    mask = _mask([1, 0, 0, 0, 0, 0])
    loss, diag = allowed_mass_loss(logits, mask, weight=1.0)
    assert loss.item() == pytest.approx(0.0, abs=1e-3)
    assert diag.mean_allowed_mass == pytest.approx(1.0, abs=1e-3)


def test_loss_decreases_as_mass_moves_into_the_allowed_set():
    mask = _mask([1, 0, 0, 0, 0, 0])
    outside = allowed_mass_loss(
        _logits([0.0, 5.0, 0.0, 0.0, 0.0, 0.0]), mask, weight=1.0
    )[0]
    inside = allowed_mass_loss(
        _logits([5.0, 0.0, 0.0, 0.0, 0.0, 0.0]), mask, weight=1.0
    )[0]
    assert inside.item() < outside.item()


def test_is_flat_over_the_allowed_set():
    """Documented LIMITATION: the term cannot reorder allowed candidates.

    Two distributions with identical total allowed mass but opposite preference
    *within* the set receive the same loss. This is why the objective cannot be
    expected to improve exact match.
    """
    mask = _mask([1, 1, 0, 0, 0, 0])
    a = allowed_mass_loss(_logits([3.0, 1.0, 0.0, 0.0, 0.0, 0.0]), mask, weight=1.0)[0]
    b = allowed_mass_loss(_logits([1.0, 3.0, 0.0, 0.0, 0.0, 0.0]), mask, weight=1.0)[0]
    assert a.item() == pytest.approx(b.item(), rel=1e-6)


def test_gradient_flows_and_is_finite():
    logits = torch.randn(3, VOCAB, requires_grad=True)
    mask = torch.zeros(3, VOCAB, dtype=torch.bool)
    mask[:, :2] = True
    loss, _ = allowed_mass_loss(logits, mask, weight=0.1)
    loss.backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


def test_numerically_stable_with_extreme_logits():
    logits = _logits([1e4, -1e4, 1e4, -1e4, 0.0, 0.0])
    mask = _mask([1, 0, 0, 0, 0, 0])
    loss, diag = allowed_mass_loss(logits, mask, weight=1.0)
    assert math.isfinite(loss.item())
    assert math.isfinite(diag.mean_neg_log_mass)


def test_rows_without_an_allowed_token_are_skipped_not_fabricated():
    logits = torch.zeros(2, VOCAB)
    mask = _mask([1, 1, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0])
    _, diag = allowed_mass_loss(logits, mask, weight=1.0)
    assert diag.scored_positions == 1
    assert diag.skipped_positions == 1


def test_position_mask_selects_contributing_positions():
    logits = torch.zeros(3, VOCAB)
    mask = torch.zeros(3, VOCAB, dtype=torch.bool)
    mask[:, :2] = True
    positions = torch.tensor([True, False, False])
    _, diag = allowed_mass_loss(logits, mask, positions, weight=1.0)
    assert diag.scored_positions == 1
    assert diag.skipped_positions == 0


def test_warmup_ramps_the_effective_weight():
    logits = torch.zeros(1, VOCAB)
    mask = _mask([1, 1, 0, 0, 0, 0])
    weights = [
        allowed_mass_loss(logits, mask, weight=1.0, warmup_steps=10, step=s)[
            1
        ].effective_weight
        for s in (0, 5, 10, 50)
    ]
    assert weights == [0.0, 0.5, 1.0, 1.0]


def test_warmup_zero_or_missing_step_applies_full_weight():
    logits = torch.zeros(1, VOCAB)
    mask = _mask([1, 1, 0, 0, 0, 0])
    _, diag = allowed_mass_loss(logits, mask, weight=0.3, warmup_steps=0, step=None)
    assert diag.effective_weight == pytest.approx(0.3)


def test_mask_is_obtainable_from_the_real_trie():
    """End-to-end: the loss must have a real producer for its input mask.

    Without this, ``allowed_mass_loss`` would be a function nothing can call:
    the hard part is walking the dual-root Trie to get the allowed set for a
    prefix. ``GlossVocabularyLogitsProcessor.allowed_mask_for_prefixes`` is that
    producer, and it reuses the exact walk the decoder applies, so the loss and
    the decoder cannot drift apart.
    """
    from src.grammar.gloss_grammar import GlossVocabularyMask
    from src.grammar.grammar_logits_processor import GlossVocabularyLogitsProcessor

    class TinyTokenizer:
        eos_token_id = 0

        def __init__(self):
            self._vocab = {"IX": 1, "MAN": 2, " IX": 3, " MAN": 4}
            self.vocab_size = 8

        def get_vocab(self):
            return dict(self._vocab)

        def encode(self, text, add_special_tokens=False):
            return [self._vocab[text]] if text in self._vocab else []

    tokenizer = TinyTokenizer()
    mask_obj = GlossVocabularyMask.__new__(GlossVocabularyMask)
    mask_obj.tokenizer = tokenizer
    mask_obj.vocab = ["IX", "MAN"]
    processor = GlossVocabularyLogitsProcessor(mask_obj, device="cpu")

    allowed = processor.allowed_mask_for_prefixes([[], [1]], vocab_size=8)
    assert allowed.shape == (2, 8)
    assert allowed.dtype == torch.bool
    assert allowed.any()

    # And the mask plugs straight into the loss.
    logits = torch.zeros(2, 8, requires_grad=True)
    loss, diag = allowed_mass_loss(logits, allowed, weight=0.1)
    assert diag.scored_positions == 2
    loss.backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


def test_invalid_shapes_and_parameters_are_rejected():
    logits = torch.zeros(2, VOCAB)
    good = torch.ones(2, VOCAB, dtype=torch.bool)
    with pytest.raises(ValueError, match="2-D"):
        allowed_mass_loss(
            torch.zeros(2, 3, VOCAB), torch.ones(2, 3, VOCAB, dtype=torch.bool)
        )
    with pytest.raises(ValueError, match="allowed_mask shape"):
        allowed_mass_loss(logits, torch.ones(3, VOCAB, dtype=torch.bool))
    with pytest.raises(ValueError, match="weight must be finite"):
        allowed_mass_loss(logits, good, weight=float("nan"))
    with pytest.raises(ValueError, match="warmup_steps"):
        allowed_mass_loss(logits, good, weight=1.0, warmup_steps=-1)
    with pytest.raises(ValueError, match="position_mask shape"):
        allowed_mass_loss(logits, good, torch.ones(5, dtype=torch.bool), weight=1.0)
