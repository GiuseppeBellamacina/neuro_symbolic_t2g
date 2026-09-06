#!/usr/bin/env python3
"""Parse T2G trainer log lines from stdin and display as a live table.

Supports both metric line formats:
  * TRL dict-style: ``{'step': 5, 'loss': ..., 'reward': ...}``
  * KV-style: ``  step=5  loss=1.23456789  reward=0.50258335  ...``
    (printed by ``HighPrecisionLogCallback`` in ``src.training.callbacks``,
    used by both GRPO and SFT training)

Completion sample blocks (``COMPLETION SAMPLES`` and
``SFT SAMPLE PREDICTIONS``) are skipped so they never corrupt the table.

Usage:
    tail -f logs/slurm-train-1234.log | python -u -m src.utils.live_training_table
    tail -f logs/slurm-train-1234.log | python -u -m src.utils.live_training_table --cols step,reward,loss
    tail -f logs/slurm-train-1234.log | python -u -m src.utils.live_training_table --rows 30
"""

from __future__ import annotations

import ast
import os
import re
import sys
from collections import deque
from typing import Any

_DEFAULT_COLS = [
    "step",
    "loss",
    "reward",
    "reward_std",
    "rewards/edit_validity_reward/mean",
    "rewards/edit_validity_reward/std",
    "completions/mean_length",
    "completions/clipped_ratio",
    "completions/mean_terminated_length",
    "learning_rate",
    "kl",
    "eval_loss",
    "grad_norm",
]

_SHORT_NAMES = {
    "rewards/edit_validity_reward/mean": "edit_validity",
    "rewards/edit_validity_reward/std": "edit_validity_std",
    "completions/mean_length": "comp_len",
    "completions/clipped_ratio": "clipped",
    "completions/mean_terminated_length": "term_len",
    "learning_rate": "lr",
    "kl": "kl",
    "eval_loss": "eval_loss",
}


def _reward_columns(keys: set[str]) -> list[str]:
    """Return reward metrics in stable order without knowing component names."""
    return sorted(key for key in keys if key.startswith("rewards/"))


_DICT_PATTERN = re.compile(r"\{.*\}")
_KV_PAIRS = re.compile(r"(?:\s|^)([A-Za-z0-9_][A-Za-z0-9_/.]*)=([^\s]+)")


def _parse_kv_line(line: str) -> dict[str, Any] | None:
    """Parse a HighPrecisionLogCallback KV line into a dict.

    Format (src/training/callbacks.py)::

         "step=5  loss=1.23456789  reward=0.50258335  completions/mean_length=10"

    Only lines that start with ``step=`` are treated as metric lines; this
    excludes arbitrary log lines that merely contain ``key=value`` tokens.
    """
    if not line.startswith("step="):
        return None
    entry: dict[str, Any] = {}
    for m in _KV_PAIRS.finditer(line):
        key, raw = m.group(1), m.group(2)
        if key == "step":
            try:
                entry[key] = int(float(raw))
            except ValueError:
                return None
        else:
            try:
                entry[key] = float(raw)
            except ValueError:
                entry[key] = raw
    return entry or None


def _format_val(key: str, val: object) -> str:
    if val is None:
        return "-"
    if isinstance(val, float):
        if key == "step":
            return str(int(val))
        if abs(val) < 0.001 and val != 0:
            return f"{val:.2e}"
        return f"{val:.4f}"
    return str(val)


def _clear() -> None:
    """Clear terminal screen."""
    os.system("cls" if os.name == "nt" else "clear")


def _redraw(header: str, separator: str, rows: deque[str]) -> None:
    """Clear and redraw the full display."""
    _clear()
    print(f" {header}")
    print(f" {separator}")
    for row in rows:
        print(f" {row}")
    sys.stdout.flush()


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Live training metrics table for T2G")
    parser.add_argument(
        "--cols", type=str, default=None, help="Comma-separated column names"
    )
    parser.add_argument(
        "--rows",
        type=int,
        default=20,
        help="Number of metric rows to keep visible (default: 20)",
    )
    args = parser.parse_args()

    requested_cols = args.cols.split(",") if args.cols else None
    max_rows = args.rows

    header = ""
    separator = ""
    metric_rows: deque[str] = deque(maxlen=max_rows)
    widths: list[int] = []
    active_cols: list[str] = []
    header_ready = False

    # Skip completion sample blocks
    in_sample_block = False
    pending_separator = False

    try:
        for line in sys.stdin:
            line = line.rstrip("\n\r")
            stripped = line.strip()

            # Handle completion sample blocks (GRPO + SFT variants)
            is_separator = stripped.startswith("═" * 10)
            if in_sample_block:
                if is_separator:
                    in_sample_block = False
                    pending_separator = False
                continue
            if is_separator and not in_sample_block:
                pending_separator = True
                continue
            if pending_separator and (
                "COMPLETION SAMPLES" in stripped or "SFT SAMPLE PREDICTIONS" in stripped
            ):
                in_sample_block = True
                pending_separator = False
                continue
            pending_separator = False

            # Parse metric lines — dict-style TRL or KV-style
            # (HighPrecisionLogCallback used by GRPO + SFT trainers).
            entry = None
            m = _DICT_PATTERN.search(stripped)
            if m:
                try:
                    entry = ast.literal_eval(m.group(0))
                except (ValueError, SyntaxError):
                    entry = None
            if entry is None:
                entry = _parse_kv_line(stripped)

            if not entry or "step" not in entry:
                continue

            # Filter to available columns
            cols = requested_cols or list(
                dict.fromkeys([*_DEFAULT_COLS, *_reward_columns(set(entry))])
            )
            current_active = [c for c in cols if c in entry]
            if not current_active:
                continue

            # Build header on first metric line
            if not header_ready or current_active != active_cols:
                active_cols = current_active
                short_names = [_SHORT_NAMES.get(c, c) for c in active_cols]
                widths = [max(8, len(s)) for s in short_names]
                header = " │ ".join(s.rjust(w) for s, w in zip(short_names, widths))
                separator = "─┼─".join("─" * w for w in widths)
                header_ready = True

            vals = [_format_val(c, entry.get(c)) for c in active_cols]
            for i, v in enumerate(vals):
                if len(v) > widths[i]:
                    widths[i] = len(v)
            row = " │ ".join(v.rjust(w) for v, w in zip(vals, widths))
            metric_rows.append(row)

            _redraw(header, separator, metric_rows)

    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
