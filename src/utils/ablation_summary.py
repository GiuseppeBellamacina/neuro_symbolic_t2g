#!/usr/bin/env python3
"""
Ablation Summary — Aggregate eval results across all configs into a
comparison table (CSV + Markdown) and a cross-config bar chart.

Usage:
    python -m src.utils.ablation_summary
    python -m src.utils.ablation_summary --results-dir experiments/results
    python -m src.utils.ablation_summary --output-dir experiments/figures

Scans the canonical ``experiments/results/<model>/<method>/<variant>/<prompt>/`` tree for ``eval_*.json`` and
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

from src.utils.paths import is_run_id

logger = logging.getLogger(__name__)


def _result_label(results_dir: Path, run_dir: Path) -> str:
    """Parse a stable method/train-prompt[/ablation]/eval-mode identity."""
    relative = run_dir.relative_to(results_dir)
    parts = relative.parts
    if not parts or not is_run_id(parts[-1]):
        raise ValueError(f"non-canonical result run: {run_dir}")
    identity = parts[:-1]
    if (
        len(identity) == 3
        and identity[1] == "baseline"
        and identity[2] in {"zero-shot", "few-shot"}
    ):
        return f"base/{identity[2]}/eval-{identity[2]}"
    if len(identity) == 4 and identity[3] in {"eval-zero-shot", "eval-few-shot"}:
        return "/".join(identity[1:])
    if (
        len(identity) == 6
        and identity[3] == "ablations"
        and identity[5] in {"eval-zero-shot", "eval-few-shot"}
    ):
        return "/".join((identity[1], identity[2], identity[4], identity[5]))
    raise ValueError(f"non-canonical result run: {run_dir}")


def _is_completed(run_dir: Path) -> bool:
    """Return whether a canonical run carries an explicit completion marker."""
    return (run_dir / "COMPLETED").exists()


def _metric_source(data: dict, block: str) -> dict:
    value = data.get(block)
    return value if isinstance(value, dict) else data


def _read_eval(run_dir: Path) -> tuple[Path, dict] | None:
    eval_files = [
        path
        for path in run_dir.glob("eval_*.json")
        if path.name != "eval_baseline.json"
    ]
    eval_files.sort(
        key=lambda path: (path.name == "eval_final.json", path.stat().st_mtime),
        reverse=True,
    )
    for eval_path in eval_files:
        try:
            data = json.loads(eval_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(data, dict) and any(
            key in _metric_source(data, "deployment") for key, _ in METRICS
        ):
            return eval_path, data
    return None


# Metrics to extract (key in eval JSON → display label)
METRICS = [
    ("rouge_l_mean", "ROUGE-L"),
    ("valid_rouge_l_mean", "Valid ROUGE-L ⭐"),
    ("pass_at_1", "Pass@1"),
    ("exact_match", "Exact Match"),
    ("validity_rate", "Validity"),
    ("bleu_sentence_mean", "BLEU (sent)"),
    ("bleu_corpus", "BLEU (corpus)"),
    ("chrf_sentence_mean", "chrF2 (sent)"),
    ("chrf_corpus", "chrF2 (corpus)"),
    ("gloss_f1_sentence_mean", "Gloss F1 (sent)"),
    ("gloss_f1_micro", "Gloss F1 (micro)"),
    ("bigram_log_prob_mean", "Bigram LP"),
]

# Also extract delta metrics from comparison.json
DELTA_METRICS = [
    ("rouge_l_mean", "Δ ROUGE-L"),
    ("valid_rouge_l_mean", "Δ Valid ROUGE-L"),
    ("pass_at_1", "Δ Pass@1"),
    ("exact_match", "Δ Exact Match"),
    ("validity_rate", "Δ Validity"),
    ("bleu_sentence_mean", "Δ BLEU (sent)"),
    ("bleu_corpus", "Δ BLEU (corpus)"),
    ("chrf_sentence_mean", "Δ chrF2 (sent)"),
    ("chrf_corpus", "Δ chrF2 (corpus)"),
    ("gloss_f1_sentence_mean", "Δ Gloss F1 (sent)"),
    ("gloss_f1_micro", "Δ Gloss F1 (micro)"),
]


def find_eval_results(results_dir: Path) -> list[dict]:
    """Scan results_dir for all eval_*.json files (excluding baseline).

    Returns a list of dicts with: config_name, run_id, path, metrics.
    """
    entries = []

    if not results_dir.exists():
        logger.warning("Results directory not found: %s", results_dir)
        return entries

    run_dirs = sorted(
        path
        for path in results_dir.rglob("run_*")
        if path.is_dir() and is_run_id(path.name)
    )
    runs_by_label: dict[str, list[Path]] = {}
    for run_dir in run_dirs:
        try:
            label = _result_label(results_dir, run_dir)
        except ValueError:
            continue
        runs_by_label.setdefault(label, []).append(run_dir)

    for config_name, candidates in sorted(runs_by_label.items()):
        selected = next(
            (
                (run, result)
                for run in sorted(candidates, reverse=True)
                if _is_completed(run) and (result := _read_eval(run)) is not None
            ),
            None,
        )
        if selected is None:
            continue
        latest_run, (eval_path, data) = selected

        entry = {
            "config_name": config_name,
            "run_id": latest_run.name,
            "eval_path": str(eval_path),
            "metrics": {},
        }

        # Extract metrics from eval JSON
        for block_name, suffix in (
            ("deployment", ""),
            ("sampling", " (sampling diagnostic)"),
        ):
            source = _metric_source(data, block_name)
            if block_name == "sampling" and source is data:
                continue
            for key, label in METRICS:
                val = source.get(key)
                if val is not None:
                    entry["metrics"][label + suffix] = float(val)

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
    all_labels.extend(
        sorted(
            {label for entry in entries for label in entry["metrics"]}.difference(
                all_labels
            )
        )
    )

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
    all_labels.extend(
        sorted(
            {label for entry in entries for label in entry["metrics"]}.difference(
                all_labels
            )
        )
    )
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

    # Use the primary metrics (not deltas) for the chart. Only plot metrics
    # that are actually present in at least one entry, so older eval JSONs
    # (without BLEU/chrF/gloss-F1) render gracefully instead of showing a
    # wall of zero bars.
    chart_labels = [
        label
        for _, label in METRICS
        if label != "Bigram LP"
        and any(e["metrics"].get(label) is not None for e in entries)
    ]
    if not chart_labels:
        logger.warning("No plottable metrics found")
        return

    bounded_labels = [label for label in chart_labels if not label.startswith("chrF2")]
    chrf_labels = [label for label in chart_labels if label.startswith("chrF2")]
    groups = [("Bounded metrics [0,1]", bounded_labels), ("chrF2 [0,100]", chrf_labels)]
    groups = [(title, labels) for title, labels in groups if labels]
    n_configs = len(entries)
    config_names = [e["config_name"] for e in entries]

    x = np.arange(n_configs)
    fig, axes = plt.subplots(
        len(groups),
        1,
        figsize=(max(14, n_configs * 1.5), 6 * len(groups)),
        squeeze=False,
    )

    for ax, (group_title, labels) in zip(axes[:, 0], groups):
        width = 0.8 / len(labels)
        for i, label in enumerate(labels):
            values = [e["metrics"].get(label, np.nan) for e in entries]
            bars = ax.bar(x + i * width - 0.4 + width / 2, values, width, label=label)
            # Add value labels on top of bars
            for bar, val in zip(bars, values):
                if np.isfinite(val) and abs(val) > 0.001:
                    offset = max(abs(val) * 0.01, 0.005)
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + offset,
                        f"{val:.3f}",
                        ha="center",
                        va="bottom",
                        fontsize=7,
                        rotation=45,
                    )
        ax.set_xlabel("Config")
        ax.set_ylabel(group_title)
        ax.set_title(group_title)
        ax.set_xticks(x)
        ax.set_xticklabels(config_names, rotation=45, ha="right")
        ax.legend(loc="upper right")
        if group_title.startswith("Bounded"):
            ax.set_ylim(0, 1.05)
        else:
            ax.set_ylim(bottom=0)
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle("Ablation Study — Cross-Config Comparison")
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
