"""Teacher-forced probability-mass loss for token grammar states.

This module deliberately depends on a small state-machine interface rather than
on a concrete Trie implementation.  The same state transitions used during
constrained decoding should be supplied here; this module does not reconstruct
or approximate Trie semantics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Hashable, Protocol, TypeVar

import torch
from torch import Tensor

StateT = TypeVar("StateT", bound=Hashable)


class AllowedTokenStateMachine(Protocol[StateT]):
    """Minimal shared-Trie adapter required by :func:`allowed_mass_loss`.

    ``advance`` always returns a state and ``is_invalid`` reports whether the
    transition failed.  EOS is semantically identified by ``eos_token_id``,
    checked against ``allowed_token_ids``, and is not passed to ``advance``.
    This avoids conflating EOS with PAD when those IDs are equal.
    """

    @property
    def eos_token_id(self) -> int: ...

    def initial_state(self) -> StateT: ...

    def allowed_token_ids(self, state: StateT) -> tuple[int, ...] | list[int]: ...

    def advance(self, state: StateT, token_id: int) -> StateT: ...

    def is_invalid(self, state: StateT) -> bool: ...


@dataclass(frozen=True)
class AllowedMassDiagnostics:
    """Detached diagnostics for one loss invocation."""

    scored_tokens: int
    scored_samples: int
    skipped_truncated: int
    invalid_prefix: int
    mean_allowed_mass: float
    mean_log_allowed_mass: float
    loss_sum: float
    allowed_mass_sum: float
    log_allowed_mass_sum: float


def allowed_mass_loss(
    logits: Tensor,
    labels: Tensor,
    completion_start: Tensor,
    state_machine: AllowedTokenStateMachine[StateT],
    *,
    completion_mask: Tensor | None = None,
    truncation_eligible: Tensor | None = None,
) -> tuple[Tensor, AllowedMassDiagnostics]:
    """Compute negative log probability assigned to teacher-forced allowed sets.

    ``logits[:, q - 1]`` predicts ``labels[:, q]``. The returned loss is the
    mean over scored tokens in this microbatch (not a corpus-level mean). The
    detached sums and count in diagnostics permit exact token-weighted offline
    aggregation across unequal microbatches. The full target trace is
    validated before losses are accumulated.  Samples with an invalid prefix
    or without an eligible EOS are therefore skipped in full; ``scored_samples``
    counts only traces valid through EOS.  Labels after the first EOS are
    ignored. ``truncation_eligible=False`` also skips a sample in full.

    Args:
        logits: Raw model logits with shape ``[B, S, V]``.
        labels: Token targets with shape ``[B, S]``; prompt/padding is ``-100``.
        completion_start: Absolute first completion target index, shape ``[B]``.
        state_machine: Shared Trie/state-machine adapter.
        completion_mask: Optional boolean eligibility mask, shape ``[B, S]``.
        truncation_eligible: Optional per-sample boolean, shape ``[B]``.
    """
    _validate_inputs(
        logits, labels, completion_start, completion_mask, truncation_eligible
    )

    batch_size, sequence_length, vocab_size = logits.shape
    eos_token_id = int(state_machine.eos_token_id)
    fp32_logits = logits.float()
    losses: list[Tensor] = []
    log_masses: list[Tensor] = []
    scored_samples = 0
    skipped_truncated = 0
    invalid_prefix = 0
    allowed_cache: dict[tuple[StateT, torch.device], Tensor] = {}

    for batch_index in range(batch_size):
        start = int(completion_start[batch_index].item())
        if truncation_eligible is not None and not bool(
            truncation_eligible[batch_index].item()
        ):
            skipped_truncated += 1
            continue

        eligible_positions = _eligible_positions(
            labels[batch_index], start, sequence_length, completion_mask, batch_index
        )
        eos_offset = next(
            (
                offset
                for offset, position in enumerate(eligible_positions)
                if int(labels[batch_index, position].item()) == eos_token_id
            ),
            None,
        )
        if eos_offset is None:
            skipped_truncated += 1
            continue

        trace: list[tuple[int, tuple[int, ...], StateT]] = []
        state = state_machine.initial_state()
        trace_is_valid = True
        for position in eligible_positions[: eos_offset + 1]:
            token_id = int(labels[batch_index, position].item())
            allowed_ids = tuple(
                int(token) for token in state_machine.allowed_token_ids(state)
            )
            if (
                not allowed_ids
                or any(token < 0 or token >= vocab_size for token in allowed_ids)
                or token_id not in allowed_ids
            ):
                trace_is_valid = False
                break
            trace.append((position, allowed_ids, state))

            if token_id == eos_token_id:
                break

            next_state = state_machine.advance(state, token_id)
            if state_machine.is_invalid(next_state):
                trace_is_valid = False
                break
            state = next_state

        if not trace_is_valid:
            invalid_prefix += 1
            continue

        for position, allowed_ids, trace_state in trace:
            cache_key = (trace_state, logits.device)
            allowed = allowed_cache.get(cache_key)
            if allowed is None:
                allowed = torch.tensor(
                    allowed_ids, dtype=torch.long, device=logits.device
                )
                allowed_cache[cache_key] = allowed
            prediction_logits = fp32_logits[batch_index, position - 1]
            log_mass = torch.logsumexp(
                prediction_logits.index_select(0, allowed), dim=0
            ) - torch.logsumexp(prediction_logits, dim=0)
            losses.append(-log_mass)
            log_masses.append(log_mass)
        scored_samples += 1

    if losses:
        stacked_losses = torch.stack(losses)
        loss = stacked_losses.mean()
        detached_log_masses = torch.stack(log_masses).detach()
        loss_sum = float(stacked_losses.detach().sum().item())
        log_mass_sum = float(detached_log_masses.sum().item())
        mass_sum = float(detached_log_masses.exp().sum().item())
        mean_log_mass = float(detached_log_masses.mean().item())
        mean_mass = float(detached_log_masses.exp().mean().item())
    else:
        loss = logits.sum() * 0.0
        loss_sum = 0.0
        log_mass_sum = 0.0
        mass_sum = 0.0
        mean_log_mass = 0.0
        mean_mass = 0.0

    diagnostics = AllowedMassDiagnostics(
        scored_tokens=len(losses),
        scored_samples=scored_samples,
        skipped_truncated=skipped_truncated,
        invalid_prefix=invalid_prefix,
        mean_allowed_mass=mean_mass,
        mean_log_allowed_mass=mean_log_mass,
        loss_sum=loss_sum,
        allowed_mass_sum=mass_sum,
        log_allowed_mass_sum=log_mass_sum,
    )
    return loss, diagnostics


def _eligible_positions(
    sample_labels: Tensor,
    start: int,
    sequence_length: int,
    completion_mask: Tensor | None,
    batch_index: int,
) -> list[int]:
    return [
        position
        for position in range(start, sequence_length)
        if int(sample_labels[position].item()) != -100
        and (
            completion_mask is None
            or bool(completion_mask[batch_index, position].item())
        )
    ]


def _validate_inputs(
    logits: Tensor,
    labels: Tensor,
    completion_start: Tensor,
    completion_mask: Tensor | None,
    truncation_eligible: Tensor | None,
) -> None:
    if logits.ndim != 3:
        raise ValueError("logits must have shape [B, S, V]")
    batch_size, sequence_length, _ = logits.shape
    if labels.shape != (batch_size, sequence_length):
        raise ValueError("labels must have shape [B, S] matching logits")
    if completion_start.shape != (batch_size,):
        raise ValueError("completion_start must have shape [B]")
    if completion_mask is not None:
        if completion_mask.shape != labels.shape or completion_mask.dtype != torch.bool:
            raise ValueError("completion_mask must be boolean with shape [B, S]")
    if truncation_eligible is not None:
        if (
            truncation_eligible.shape != (batch_size,)
            or truncation_eligible.dtype != torch.bool
        ):
            raise ValueError("truncation_eligible must be boolean with shape [B]")

    for batch_index in range(batch_size):
        start = int(completion_start[batch_index].item())
        if start < 1 or start >= sequence_length:
            raise ValueError("each completion_start must satisfy 1 <= start < S")
        if torch.any(labels[batch_index, :start] != -100):
            raise ValueError("labels before completion_start must be -100")
        if completion_mask is not None:
            if torch.any(completion_mask[batch_index, :start]):
                raise ValueError("completion_mask cannot select prompt positions")
            selected_ignored = completion_mask[batch_index] & (
                labels[batch_index] == -100
            )
            if torch.any(selected_ignored):
                raise ValueError("completion_mask cannot select -100 labels")
