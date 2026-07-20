#!/usr/bin/env python3
"""
Ablation Summary — Aggregate eval results across all configs into a
comparison table (CSV + Markdown) and a cross-config bar chart.

Usage:
    python -m src.utils.ablation_summary
    python -m src.utils.ablation_summary --results-dir experiments/results
    python -m src.utils.ablation_summary --output-dir experiments/figures

Scans ``experiments/results/*/`` for ``eval_*.json`` and
``comparison.json`` files, extracts metrics, and produces:
    - ``ablation_summary.csv`` — machine-readable table
    - ``ablation_summary.md`` — human-readable Markdown table
    - ``ablation_comparison.png`` — grouped bar chart (ROUGE-L, Pass@1, etc.)
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)

# Metrics to extract (key in eval JSON → display label)
METRICS = [
    ("rouge_l_mean", "ROUGE-L"),
    ("pass_at_1", "Pass@1"),
    ("exact_match", "Exact Match"),
    ("validity_rate", "Validity"),
    ("bigram_log_prob_mean", "Bigram LP"),
]

# Also extract delta metrics from comparison.json
DELTA_METRICS = [
    ("rouge_l_mean", "Δ ROUGE-L"),
    ("pass_at_1", "Δ Pass@1"),
    ("exact_match", "Δ Exact Match"),
    ("validity_rate", "Δ Validity"),
]


def find_eval_results(results_dir: Path) -> list[dict]:
    """Scan results_dir for all eval_*.json files (excluding baseline).

    Returns a list of dicts with: config_name, run_id, path, metrics.
    """
    entries = []

    if not results_dir.exists():
        logger.warning("Results directory not found: %s", results_dir)
        return entries

    for config_dir in sorted(results_dir.iterdir()):
        if not config_dir.is_dir():
            continue

        config_name = config_dir.name

        # Each config may have multiple run_* subdirectories.
        # Take the latest one (sorted = chronological).
        run_dirs = sorted(
            [
                d
                for d in config_dir.iterdir()
                if d.is_dir() and d.name.startswith("run_")
            ]
        )
        if not run_dirs:
            # Maybe results are directly in the config dir (no run_ subdirs)
            run_dirs = [config_dir]

        latest_run = run_dirs[-1]

        # Find eval_*.json (skip eval_baseline.json — that's the zero-shot ref)
        eval_files = sorted(latest_run.glob("eval_*.json"))
        eval_files = [f for f in eval_files if f.name != "eval_baseline.json"]

        if not eval_files:
            logger.debug("No eval_*.json in %s", latest_run)
            continue

        # Take the latest eval file
        eval_path = eval_files[-1]
        try:
            with open(eval_path, encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            logger.warning("Failed to read %s: %s", eval_path, e)
            continue

        entry = {
            "config_name": config_name,
            "run_id": latest_run.name,
            "eval_path": str(eval_path),
            "metrics": {},
        }

        # Extract metrics from eval JSON
        for key, label in METRICS:
            val = data.get(key)
            if val is not None:
                entry["metrics"][label] = float(val)

        # Extract delta metrics from comparison.json (if exists)
        comp_path = latest_run / "comparison.json"
        if comp_path.exists():
            try:
                with open(comp_path, encoding="utf-8") as f:
                    comp = json.load(f)
                delta = comp.get("delta", {})
                for key, label in DELTA_METRICS:
                    val = delta.get(key)
                    if val is not None:
                        entry["metrics"][label] = float(val)
            except Exception:
                pass

        entries.append(entry)

    return entries


def build_summary_table(entries: list[dict]) -> str:
    """Build a Markdown table from the entries."""
    if not entries:
        return "No eval results found."

    # Collect all metric labels
    all_labels = []
    for _, label in METRICS:
        all_labels.append(label)
    for _, label in DELTA_METRICS:
        all_labels.append(label)

    # Build table
    header = "| Config | " + " | ".join(all_labels) + " |"
    separator = "|---|" + "|".join(["---"] * len(all_labels)) + "|"
    rows = [header, separator]

    for entry in entries:
        name = entry["config_name"]
        values = []
        for label in all_labels:
            v = entry["metrics"].get(label)
            if v is not None:
                if label.startswith("Δ"):
                    values.append(f"{v:+.4f}")
                else:
                    values.append(f"{v:.4f}")
            else:
                values.append("—")
        rows.append(f"| {name} | " + " | ".join(values) + " |")

    return "\n".join(rows)


def build_csv(entries: list[dict]) -> str:
    """Build a CSV string from the entries."""
    if not entries:
        return "config_name,run_id\n"

    all_labels = [label for _, label in METRICS] + [label for _, label in DELTA_METRICS]
    header = "config_name,run_id," + ",".join(all_labels)
    rows = [header]

    for entry in entries:
        name = entry["config_name"]
        run_id = entry["run_id"]
        values = [name, run_id]
        for label in all_labels:
            v = entry["metrics"].get(label)
            values.append(f"{v:.6f}" if v is not None else "")
        rows.append(",".join(values))

    return "\n".join(rows)


def plot_ablation_comparison(entries: list[dict], output_path: Path) -> None:
    """Generate a grouped bar chart comparing metrics across configs."""
    if not entries:
        logger.warning("No entries to plot")
        return

    # Use the 4 primary metrics (not deltas) for the chart
    chart_labels = [label for _, label in METRICS if label != "Bigram LP"]
    n_metrics = len(chart_labels)
    n_configs = len(entries)
    config_names = [e["config_name"] for e in entries]

    x = np.arange(n_configs)
    width = 0.8 / n_metrics

    fig, ax = plt.subplots(figsize=(max(14, n_configs * 1.5), 7))

    for i, label in enumerate(chart_labels):
        values = [e["metrics"].get(label, 0.0) for e in entries]
        bars = ax.bar(x + i * width - 0.4 + width / 2, values, width, label=label)
        # Add value labels on top of bars
        for bar, val in zip(bars, values):
            if val > 0.001:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.005,
                    f"{val:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                    rotation=45,
                )

    ax.set_xlabel("Config")
    ax.set_ylabel("Score")
    ax.set_title("Ablation Study — Cross-Config Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels(config_names, rotation=45, ha="right")
    ax.legend(loc="upper right")
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Bar chart saved to %s", output_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate eval results into ablation summary table + chart"
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="experiments/results",
        help="Directory containing eval results (default: experiments/results)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="experiments/figures",
        help="Output directory for summary files (default: experiments/figures)",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    entries = find_eval_results(results_dir)

    if not entries:
        print(f"\n❌ No eval results found in {results_dir}/")
        print("   Run an ablation study first: bash cluster/run_all.sh --ablation")
        return

    print(f"\n{'=' * 60}")
    print(f"  Ablation Summary — {len(entries)} configs found")
    print(f"{'=' * 60}\n")

    # Markdown table
    md_table = build_summary_table(entries)
    md_path = output_dir / "ablation_summary.md"
    md_path.write_text(f"# Ablation Summary\n\n{md_table}\n", encoding="utf-8")
    print(md_table)
    print(f"\n  Markdown: {md_path}")

    # CSV
    csv_str = build_csv(entries)
    csv_path = output_dir / "ablation_summary.csv"
    csv_path.write_text(csv_str, encoding="utf-8")
    print(f"  CSV:      {csv_path}")

    # Bar chart
    chart_path = output_dir / "ablation_comparison.png"
    plot_ablation_comparison(entries, chart_path)
    print(f"  Chart:    {chart_path}")

    print(f"\n{'=' * 60}")
    print(f"  Summary complete! {len(entries)} configs compared.")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
