"""Allowed-mass auxiliary loss (candidate-set log-marginal).

What this is
------------
Given the set of token ids the gloss Trie would allow at a position, this adds

.. math::

    -\\log \\sum_{t \\in \\text{Allowed}} p(t \\mid \\text{prefix})

to the standard supervised cross-entropy. The objective is **not novel**: it is
the candidate-set log-marginal known as *partial-label learning* (Cour et al.,
JMLR 2011; Seo & Huh, arXiv:2010.11600, whose "naive" ``-log Σ P(c)`` already
performs well) and as *maximum marginal likelihood* in weakly-supervised
semantic parsing (Guu et al., arXiv:1704.07926). The token-level variant that
pushes disallowed tokens down instead is *unlikelihood training* (Welleck et
al., arXiv:1908.04319). It must be described as an application, not an
invention.

Why it might help
-----------------
Hard masking distorts the model's distribution: *Grammar-Aligned Decoding*
(Park et al., arXiv:2405.21047, NeurIPS 2024) shows constrained output is
grammatical but its likelihoods are not proportional to the model's own. Moving
raw mass inside the allowed set shrinks that renormalization gap. On this
project the mask is load-bearing: without it, copy-insensitive gloss accuracy
collapses from 0.0432 to 0.0006.

Why to keep expectations low
----------------------------
The term maximizes the mass **of the set**, not the mass of the *correct*
member, so it is flat over the allowed set and cannot reorder candidates within
it. On teacher-forced gold prefixes the gold token is in the allowed set by
construction, which makes this a pure sharpening regularizer: expect calibration
effects, not exact-match gains. Over-sharpening can itself hurt calibration
(Müller et al., arXiv:1906.02629), hence the small default weight and the
warmup.

Usage contract
--------------
``weight=0.0`` returns exactly zero and leaves standard SFT bit-for-bit
unchanged. The loss never fabricates a value for a row it cannot score: rows
with no allowed set are skipped and reported in the diagnostics.

Obtaining ``allowed_mask``: use
:meth:`src.grammar.grammar_logits_processor.GlossVocabularyLogitsProcessor.allowed_mask_for_prefixes`,
which exposes the same dual-root Trie walk the decoder applies, so the loss and
the decoder cannot drift apart. That producer is the reason this module is
callable at all.

STATUS ON THIS BRANCH: **no training client.** The mask producer and the loss
both exist and are tested end-to-end, but nothing in ``sft_train.py`` calls
them — wiring an auxiliary objective into the trainer is exactly the change that
was rejected from the ``edit-rewards`` branch for altering default-path SFT.
Before using this in a real run, satisfy GATE 1a in
``docs/NEW_OBJECTIVES_SPEC.md``: one inference-only cluster probe showing that
``removed_mass`` is large enough to be worth optimizing. Note that
``track_diagnostics`` defaults to ``False`` and no training script enables it,
so no historical run recorded this quantity.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class AllowedMassDiagnostics:
    """Per-call telemetry, so a run can be audited without re-deriving it."""

    #: Mean ``-log P(allowed)`` over scored positions (0.0 when none).
    mean_neg_log_mass: float
    #: Mean ``P(allowed)`` over scored positions (0.0 when none).
    mean_allowed_mass: float
    #: Positions that contributed to the loss.
    scored_positions: int
    #: Positions skipped because the allowed set was empty.
    skipped_positions: int
    #: Effective weight applied after warmup.
    effective_weight: float


def _warmup_scale(step: int | None, warmup_steps: int) -> float:
    if warmup_steps <= 0 or step is None:
        return 1.0
    if step >= warmup_steps:
        return 1.0
    return max(0.0, float(step) / float(warmup_steps))


def allowed_mass_loss(
    logits: torch.Tensor,
    allowed_mask: torch.Tensor,
    position_mask: torch.Tensor | None = None,
    *,
    weight: float = 0.0,
    warmup_steps: int = 0,
    step: int | None = None,
) -> tuple[torch.Tensor, AllowedMassDiagnostics]:
    """Compute the weighted candidate-set log-marginal penalty.

    Args:
        logits: Raw (unmasked) logits, shape ``[N, V]``, where ``N`` is the
            number of candidate positions. Must be the *unmasked* logits: the
            whole point is to measure mass falling outside the allowed set.
        allowed_mask: Boolean tensor, shape ``[N, V]``, ``True`` where the
            grammar/Trie permits the token.
        position_mask: Optional boolean tensor ``[N]`` selecting positions that
            should contribute (e.g. completion tokens only, EOS excluded).
        weight: Auxiliary weight. ``0.0`` short-circuits to an exact zero.
        warmup_steps: Linearly ramp the weight over this many steps.
        step: Current global step, used only for the warmup ramp.

    Returns:
        ``(loss, diagnostics)``. ``loss`` is a scalar tensor on ``logits``'
        device/dtype-promoted-to-float32 graph; it is exactly zero (and detached)
        when ``weight == 0`` or when no position can be scored.

    Raises:
        ValueError: On shape mismatch, non-finite weight, or negative warmup.
    """
    if logits.ndim != 2:
        raise ValueError(f"logits must be 2-D [N, V], got shape {tuple(logits.shape)}")
    if allowed_mask.shape != logits.shape:
        raise ValueError(
            f"allowed_mask shape {tuple(allowed_mask.shape)} must match "
            f"logits shape {tuple(logits.shape)}"
        )
    if not math.isfinite(weight):
        raise ValueError(f"weight must be finite, got {weight!r}")
    if warmup_steps < 0:
        raise ValueError(f"warmup_steps must be >= 0, got {warmup_steps!r}")
    if position_mask is not None and position_mask.shape != logits.shape[:1]:
        raise ValueError(
            f"position_mask shape {tuple(position_mask.shape)} must be "
            f"[{logits.shape[0]}]"
        )

    zero = logits.sum() * 0.0  # keeps device/graph without contributing gradient

    scale = _warmup_scale(step, warmup_steps)
    effective = float(weight) * scale

    selected = allowed_mask
    if position_mask is not None:
        selected = selected & position_mask.unsqueeze(1)

    has_allowed = selected.any(dim=1)
    if position_mask is not None:
        has_allowed = has_allowed & position_mask
    scored = int(has_allowed.sum().item())
    total = (
        int(position_mask.sum().item())
        if position_mask is not None
        else logits.shape[0]
    )
    skipped = max(0, total - scored)

    if effective == 0.0 or scored == 0:
        return zero, AllowedMassDiagnostics(0.0, 0.0, scored, skipped, effective)

    # FP32 for numerical stability; log-domain throughout so no exp() overflows.
    rows = logits[has_allowed].float()
    mask_rows = selected[has_allowed]
    log_norm = torch.logsumexp(rows, dim=-1)
    masked = rows.masked_fill(~mask_rows, float("-inf"))
    log_allowed = torch.logsumexp(masked, dim=-1)
    # log P(allowed) = logsumexp(allowed logits) - logsumexp(all logits) <= 0
    neg_log_mass = -(log_allowed - log_norm)

    loss = effective * neg_log_mass.mean()
    diagnostics = AllowedMassDiagnostics(
        mean_neg_log_mass=float(neg_log_mass.mean().detach().item()),
        mean_allowed_mass=float(neg_log_mass.mul(-1).exp().mean().detach().item()),
        scored_positions=scored,
        skipped_positions=skipped,
        effective_weight=effective,
    )
    return loss, diagnostics
