"""Display training log from trainer_state.json as a formatted table or plot.

Usage:
    python -m src.utils.show_training_log experiments/checkpoints/grpo/t2g/qwen05/checkpoint-500
    python -m src.utils.show_training_log experiments/checkpoints/grpo/t2g/qwen05/ --last
    python -m src.utils.show_training_log experiments/checkpoints/grpo/t2g/qwen05/ --plot
    python -m src.utils.show_training_log experiments/checkpoints/grpo/t2g/qwen05/ --plot --deg 5
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

# Default columns to show for T2G training
_DEFAULT_COLS = [
    "step",
    "loss",
    "reward",
    "reward_std",
    "rewards/translation_quality_reward/mean",
    "rewards/gold_structure_reward/mean",
    "rewards/structural_dense_reward/mean",
    "rewards/viterbi_distance_reward/mean",
    "rewards/soft_viterbi_distance_reward/mean",
    "rewards/verifier_scaled_reward/mean",
    "rewards/gloss_order_reward/mean",
    "rewards/gloss_format_reward/mean",
    "rewards/gloss_repetition_reward/mean",
    "completion_length",
    "learning_rate",
    "grad_norm",
]

_SHORT_NAMES = {
    "rewards/translation_quality_reward/mean": "translation",
    "rewards/gold_structure_reward/mean": "gold_struct",
    "rewards/structural_dense_reward/mean": "structure",
    "rewards/viterbi_distance_reward/mean": "viterbi",
    "rewards/soft_viterbi_distance_reward/mean": "soft_viterbi",
    "rewards/verifier_scaled_reward/mean": "verifier",
    "rewards/gloss_order_reward/mean": "order",
    "rewards/gloss_format_reward/mean": "format",
    "rewards/gloss_repetition_reward/mean": "repetition",
    "completion_length": "comp_len",
    "learning_rate": "lr",
}


def _find_trainer_state(path: str) -> Path | None:
    """Find trainer_state.json from a checkpoint or output dir path."""
    p = Path(path)

    if p.name == "trainer_state.json" and p.exists():
        return p

    # Inside a checkpoint dir
    ts = p / "trainer_state.json"
    if ts.exists():
        return ts

    # --last: find the latest checkpoint-* in a directory
    ckpts = sorted(p.glob("checkpoint-*"))
    if ckpts:
        ts = ckpts[-1] / "trainer_state.json"
        if ts.exists():
            return ts

    return None


def _format_value(val: object) -> str:
    """Format a value for display."""
    if val is None:
        return "-"
    if isinstance(val, float):
        if abs(val) < 0.001 and val != 0:
            return f"{val:.2e}"
        return f"{val:.4f}"
    return str(val)


def show_log(
    path: str,
    columns: list[str] | None = None,
    tail: int | None = None,
) -> None:
    """Display the training log from a checkpoint as a formatted table."""
    ts_path = _find_trainer_state(path)
    if ts_path is None:
        print(f"No trainer_state.json found in {path}")
        return

    print(f"Source: {ts_path.parent.name}/{ts_path.name}")

    data = json.loads(ts_path.read_text(encoding="utf-8"))
    log_history = data.get("log_history", [])
    if not log_history:
        print("No log entries found.")
        return

    # Filter to training logs only (skip eval entries)
    train_logs = [e for e in log_history if "loss" in e or "reward" in e]
    if not train_logs:
        print("No training log entries found.")
        return

    if tail:
        train_logs = train_logs[-tail:]

    cols = columns or _DEFAULT_COLS
    # Filter columns to those that actually exist
    available = set()
    for entry in train_logs:
        available.update(entry.keys())
    cols = [c for c in cols if c in available]

    if not cols:
        print("No matching columns found. Available:")
        for k in sorted(available):
            print(f"  {k}")
        return

    # Shorten column headers
    short_names = [_SHORT_NAMES.get(c, c) for c in cols]

    # Compute column widths
    rows = [[_format_value(entry.get(c)) for c in cols] for entry in train_logs]
    widths = [
        max(len(h), max(len(r[i]) for r in rows)) for i, h in enumerate(short_names)
    ]

    # Print header
    header = " │ ".join(h.rjust(w) for h, w in zip(short_names, widths))
    sep = "─┼─".join("─" * w for w in widths)
    print(f" {header}")
    print(f" {sep}")

    # Print rows
    for row in rows:
        line = " │ ".join(v.rjust(w) for v, w in zip(row, widths))
        print(f" {line}")

    print(f"\n{len(rows)} entries, global_step={data.get('global_step', '?')}")


def plot_from_checkpoint(
    path: str,
    degree: int = 4,
    output_dir: str | None = None,
) -> None:
    """Generate training curve plots from a checkpoint's trainer_state.json."""
    ts_path = _find_trainer_state(path)
    if ts_path is None:
        print(f"No trainer_state.json found in {path}")
        return

    data = json.loads(ts_path.read_text(encoding="utf-8"))

    if output_dir is None:
        output_dir = str(ts_path.parent.parent / "figures")

    from .visualization import plot_training_curves

    model_name = (
        ts_path.parent.parent.name
        if ts_path.parent.name.startswith("checkpoint")
        else ts_path.parent.name
    )
    plot_training_curves(
        data,
        model_name=model_name,
        output_path=f"{output_dir}/training_curves.png",
        degree=degree,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Show T2G training log as table or plot"
    )
    parser.add_argument("path", help="Path to checkpoint dir or output dir")
    parser.add_argument(
        "--cols", type=str, default=None, help="Comma-separated column names"
    )
    parser.add_argument(
        "--tail", type=int, default=None, help="Show only last N entries"
    )
    parser.add_argument(
        "--all-cols", action="store_true", help="List all available columns"
    )
    parser.add_argument(
        "--plot", action="store_true", help="Generate training curve plots"
    )
    parser.add_argument(
        "--deg", type=int, default=4, help="Polynomial regression degree (default: 4)"
    )
    parser.add_argument(
        "--output-dir", type=str, default=None, help="Output directory for plots"
    )
    args = parser.parse_args()

    if args.all_cols:
        ts_path = _find_trainer_state(args.path)
        if ts_path:
            data = json.loads(ts_path.read_text(encoding="utf-8"))
            cols = set()
            for e in data.get("log_history", []):
                cols.update(e.keys())
            print("Available columns:")
            for c in sorted(cols):
                print(f"  {c}")
        return

    if args.plot:
        plot_from_checkpoint(args.path, degree=args.deg, output_dir=args.output_dir)
        return

    columns = args.cols.split(",") if args.cols else None
    show_log(args.path, columns=columns, tail=args.tail)


if __name__ == "__main__":
    main()
