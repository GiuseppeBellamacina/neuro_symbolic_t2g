"""Exact constrained-decoding diagnostics shared by logits processors."""

from __future__ import annotations

import torch


class MaskedMassTracker:
    """Accumulate exact, equal-weight statistics over active row/steps."""

    _STAT_KEYS = (
        "allowed_mass_mean",
        "removed_mass_mean",
        "log_allowed_mass_mean",
        "allowed_mass_min",
        "entropy_raw_mean",
        "entropy_allowed_mean",
    )

    def _init_masked_stats(self) -> None:
        self._reset_diagnostics()
        if not hasattr(self, "step_count"):
            self.step_count = 0

    def _reset_diagnostics(self) -> None:
        self._diag_sums = {
            "allowed_mass": 0.0,
            "removed_mass": 0.0,
            "log_allowed_mass": 0.0,
            "entropy_raw": 0.0,
            "entropy_allowed": 0.0,
        }
        self._diag_allowed_min = float("inf")
        self._diag_rows = 0
        self._diag_steps = 0

    @staticmethod
    def _active_rows(
        input_ids: torch.LongTensor,
        prompt_len: int,
        eos_token_id: int | None,
        pad_token_id: int | None,
    ) -> torch.Tensor:
        """Rows become inactive on calls after their first generated EOS/pad."""
        history = input_ids[:, max(prompt_len, 0) :]
        active = torch.ones(
            input_ids.shape[0], dtype=torch.bool, device=input_ids.device
        )
        if history.numel() == 0:
            return active
        stop_ids = {
            token for token in (eos_token_id, pad_token_id) if token is not None
        }
        for token_id in stop_ids:
            active &= ~(history == token_id).any(dim=1)
        return active

    def _track_masked_stats(
        self,
        raw_scores: torch.Tensor,
        allowed_mask: torch.Tensor,
        active_rows: torch.Tensor | None = None,
    ) -> None:
        """Track exact per-row values from raw logits and a 2-D applied mask."""
        if raw_scores.ndim != 2 or allowed_mask.shape != raw_scores.shape:
            raise ValueError(
                "raw_scores and allowed_mask must have identical 2-D shapes"
            )
        if allowed_mask.dtype != torch.bool:
            raise TypeError("allowed_mask must be boolean")
        if active_rows is None:
            active_rows = torch.ones(
                raw_scores.shape[0], dtype=torch.bool, device=raw_scores.device
            )
        if active_rows.shape != (raw_scores.shape[0],):
            raise ValueError("active_rows must have shape (batch_size,)")

        active_rows = active_rows.to(device=raw_scores.device, dtype=torch.bool)
        valid = active_rows & allowed_mask.any(dim=1)
        if not bool(valid.any()):
            return

        with torch.no_grad():
            logits = raw_scores[valid].float()
            allowed = allowed_mask[valid]
            log_all = torch.logsumexp(logits, dim=-1)
            log_allowed = torch.logsumexp(
                logits.masked_fill(~allowed, -torch.inf), dim=-1
            )
            log_mass = log_allowed - log_all
            allowed_mass = log_mass.exp()
            removed_mass = 1.0 - allowed_mass

            log_probs = logits - log_all[:, None]
            probs = log_probs.exp()
            entropy_raw = -(probs * log_probs).sum(dim=-1)

            allowed_log_probs = logits - log_allowed[:, None]
            allowed_probs = allowed_log_probs.exp().masked_fill(~allowed, 0.0)
            entropy_allowed = -(
                allowed_probs * allowed_log_probs.masked_fill(~allowed, 0.0)
            ).sum(dim=-1)

            values = {
                "allowed_mass": allowed_mass,
                "removed_mass": removed_mass,
                "log_allowed_mass": log_mass,
                "entropy_raw": entropy_raw,
                "entropy_allowed": entropy_allowed,
            }
            for key, value in values.items():
                self._diag_sums[key] += float(value.sum().cpu())
            self._diag_allowed_min = min(
                self._diag_allowed_min, float(allowed_mass.min().cpu())
            )
            self._diag_rows += int(valid.sum().cpu())
            self._diag_steps += 1

    def get_diagnostics(self, reset_after: bool = False) -> dict[str, float | int]:
        """Return interval aggregates, optionally atomically clearing the interval."""
        rows = self._diag_rows
        stats: dict[str, float | int] = {
            f"{key}_mean": self._diag_sums[key] / rows if rows else 0.0
            for key in self._diag_sums
        }
        stats["allowed_mass_min"] = self._diag_allowed_min if rows else 0.0
        stats["active_rows"] = rows
        stats["steps"] = self._diag_steps
        if reset_after:
            self._reset_diagnostics()
        return stats
