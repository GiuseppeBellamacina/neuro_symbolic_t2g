#!/usr/bin/env python3
"""Parse T2G trainer log lines from stdin and display as a live table.

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

_DEFAULT_COLS = [
    "step",
    "loss",
    "reward",
    "reward_std",
    "rewards/translation_quality_reward/mean",
    "rewards/structural_dense_reward/mean",
    "rewards/gloss_format_reward/mean",
    "rewards/gloss_repetition_reward/mean",
    "completion_length",
    "learning_rate",
    "grad_norm",
]

_SHORT_NAMES = {
    "rewards/translation_quality_reward/mean": "translation",
    "rewards/structural_dense_reward/mean": "structure",
    "rewards/gloss_format_reward/mean": "format",
    "rewards/gloss_repetition_reward/mean": "repetition",
    "completion_length": "comp_len",
    "learning_rate": "lr",
}

_DICT_PATTERN = re.compile(r"\{.*\}")


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
    parser.add_argument("--cols", type=str, default=None, help="Comma-separated column names")
    parser.add_argument("--rows", type=int, default=20, help="Number of metric rows to keep visible (default: 20)")
    args = parser.parse_args()

    cols = args.cols.split(",") if args.cols else _DEFAULT_COLS
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

            # Handle completion sample blocks
            is_separator = stripped.startswith("═" * 10)
            if in_sample_block:
                if is_separator:
                    in_sample_block = False
                    pending_separator = False
                continue
            if is_separator and not in_sample_block:
                pending_separator = True
                continue
            if pending_separator and "COMPLETION SAMPLES" in stripped:
                in_sample_block = True
                pending_separator = False
                continue
            pending_separator = False

            # Parse metric lines (TRL dict-format logs only)
            entry = None
            m = _DICT_PATTERN.search(stripped)
            if m:
                try:
                    entry = ast.literal_eval(m.group(0))
                except (ValueError, SyntaxError):
                    pass

            if not entry or "step" not in entry:
                continue

            # Filter to available columns
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
