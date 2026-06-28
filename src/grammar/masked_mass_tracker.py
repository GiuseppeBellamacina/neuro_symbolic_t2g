"""
MaskedMassTracker — shared diagnostics mixin for logits processors.

Provides masked probability mass, entropy, and allowed-token entropy tracking
for both ``GlossVocabularyLogitsProcessor`` and ``GrammarPDALogitsProcessor``.

Usage in subclasses::

    class MyProcessor(LogitsProcessor, MaskedMassTracker):
        def __call__(self, input_ids, scores):
            probs = self._pre_process(scores)
            allowed_mask = self._build_allowed_mask(token_ids, ...)
            self._track_masked_stats(probs, allowed_mask)
            # ... apply mask to scores ...

        def reset(self):
            self.step_count = 0
            self._reset_masked_stats()
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F


class MaskedMassTracker:
    """Mixin that tracks masked probability mass, entropy, and allowed-token
    entropy at each generation step.

    Subclasses must call ``_pre_process(scores)`` at the start of
    ``__call__``, ``_track_masked_stats(probs, allowed_mask)`` before
    applying the mask, and ``_reset_masked_stats()`` inside ``reset()``.

    The ``get_masked_mass_stats(reset_after=False)`` method provides
    per-interval averages for W&B logging.
    """

    # ── Pre-processing (softmax + step counter) ──────────────────────

    def _pre_process(
        self,
        scores: torch.FloatTensor,
    ) -> torch.Tensor:
        """Increment step counter and compute softmax probabilities.

        Must be called at the start of ``__call__`` before any masking.
        Guards against zero ``vocab_size`` for processors that lazily
        detect the vocabulary dimension (e.g. ``GlossVocabularyLogitsProcessor``).

        Returns:
            Softmax probability tensor with same shape as ``scores``.
        """
        self.step_count += 1

        # Lazy vocab_size detection (only GlossVocabularyLogitsProcessor uses this)
        vs: Any = getattr(self, "vocab_size", -1)
        if vs == 0:
            self.vocab_size: int = scores.shape[-1]  # type: ignore[attr-defined]

        with torch.no_grad():
            return F.softmax(scores, dim=-1)

    # ── Allowed mask builder ──────────────────────────────────────────

    def _build_allowed_mask(
        self,
        token_ids: set[int],
        vocab_size: int,
        device: torch.device | str,
    ) -> torch.Tensor:
        """Build a boolean mask marking allowed token positions.

        Args:
            token_ids: Set of allowed integer token IDs.
            vocab_size: Total vocabulary size.
            device: Torch device for the output tensor.

        Returns:
            1-D ``torch.BoolTensor`` of shape ``(vocab_size,)`` where
            ``True`` marks positions in ``token_ids``.
        """
        mask = torch.zeros(vocab_size, dtype=torch.bool, device=device)
        for tid in token_ids:
            if 0 <= tid < vocab_size:
                mask[tid] = True
        return mask

    # ── Stats initialisation / reset ─────────────────────────────────

    def _init_masked_stats(self) -> None:
        """Initialise (or re-initialise) the tracked accumulators."""
        self._masked_mass_sum: float = 0.0
        self._masked_mass_count: int = 0
        self._masked_entropy_sum: float = 0.0
        self._entropy_allowed_sum: float = 0.0
        # Ensure step_count exists (normally set by subclasses, but
        # initialised here as a safety net).
        if not hasattr(self, "step_count"):
            self.step_count: int = 0  # type: ignore[annotation-unchecked]

    def _reset_masked_stats(self) -> None:
        """Reset accumulators for a new generation."""
        self._masked_mass_sum = 0.0
        self._masked_mass_count = 0
        self._masked_entropy_sum = 0.0
        self._entropy_allowed_sum = 0.0

    # ── Per-step tracking ─────────────────────────────────────────────

    def _track_masked_stats(
        self,
        probs: torch.Tensor,
        allowed_mask: torch.Tensor,
    ) -> None:
        """Accumulate masked mass, full entropy, and allowed-token entropy.

        ``probs`` is the full softmax distribution (batch, vocab_size).
        ``allowed_mask`` is a boolean tensor marking allowed token positions.
        Must be called inside ``torch.no_grad()`` and BEFORE applying the
        mask to scores.
        """
        eps = 1e-12

        # ── Masked probability mass ──────────────────────────────────
        masked_mass = probs[:, ~allowed_mask].sum(dim=-1).mean().item()
        self._masked_mass_sum += masked_mass
        self._masked_mass_count += 1

        # ── Full-distribution entropy ─────────────────────────────────
        entropy = -(probs * torch.log(probs + eps)).sum(dim=-1).mean().item()
        self._masked_entropy_sum += entropy

        # ── Allowed-token entropy (re-normalized) ─────────────────────
        with torch.no_grad():
            allowed_sum = probs[:, allowed_mask].sum(dim=-1)
            safe_mask = allowed_sum > eps
        if safe_mask.any():
            probs_allowed = probs[:, allowed_mask]
            probs_allowed = probs_allowed / probs_allowed.sum(dim=-1, keepdim=True)
            entropy_allowed = (
                -(probs_allowed * torch.log(probs_allowed + eps))
                .sum(dim=-1)[safe_mask]
                .mean()
                .item()
            )
        else:
            entropy_allowed = 0.0
        self._entropy_allowed_sum += entropy_allowed

    # ── Public stats interface ────────────────────────────────────────

    def get_masked_mass_stats(self, reset_after: bool = False) -> dict[str, float]:
        """Return average masked probability mass, entropy, and allowed-token
        entropy since last reset.

        The three W&B metrics:

        * ``avg_masked_mass`` — fraction of softmax mass on disallowed tokens.
        * ``avg_masked_entropy`` — full-distribution Shannon entropy.
        * ``avg_masked_entropy_allowed`` — entropy over re-normalized
          allowed-token distribution.

        Args:
            reset_after: If ``True``, reset accumulators after returning
                (for per-interval W&B logging).

        Returns:
            Dict with keys ``avg_masked_mass``, ``avg_masked_entropy``,
            ``avg_masked_entropy_allowed``, and ``total_steps``.
        """
        if self._masked_mass_count == 0:
            return {
                "avg_masked_mass": 0.0,
                "avg_masked_entropy": 0.0,
                "avg_masked_entropy_allowed": 0.0,
                "total_steps": 0,
            }
        stats = {
            "avg_masked_mass": self._masked_mass_sum / self._masked_mass_count,
            "avg_masked_entropy": self._masked_entropy_sum / self._masked_mass_count,
            "avg_masked_entropy_allowed": self._entropy_allowed_sum
            / self._masked_mass_count,
            "total_steps": self._masked_mass_count,
        }
        if reset_after:
            self._reset_masked_stats()
        return stats
