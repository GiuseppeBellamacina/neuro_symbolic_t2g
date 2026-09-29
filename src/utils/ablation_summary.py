#!/usr/bin/env python3
"""
Ablation Summary — Aggregate eval results across all configs into a
comparison table (CSV + Markdown) and a cross-config bar chart.

Usage:
    python -m src.utils.ablation_summary
    python -m src.utils.ablation_summary --results-dir experiments/results
    python -m src.utils.ablation_summary --output-dir experiments/figures

Scans ``experiments/results/<cella>/run_<ts>/`` (at any nesting depth, baselines
included: ``qwen25-05b/baseline/zero-shot/run_<ts>/``) for ``eval_*.json`` and
``comparison.json`` files, extracts metrics, and produces:
    - ``ablation_summary.csv`` — machine-readable table
    - ``ablation_summary.md`` — human-readable Markdown table
    - ``ablation_comparison.png`` — grouped bar chart (ROUGE-L, Pass@1, etc.)
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.utils import chart_style
from src.utils.run_paths import cell_sort_key

logger = logging.getLogger(__name__)

# Metrics to extract (key in eval JSON → display label)
METRICS = [
    ("rouge_l_mean", "ROUGE-L"),
    ("valid_rouge_l_mean", "Valid ROUGE-L ⭐"),
    ("pass_at_1", "Pass@1"),
    ("exact_match", "Exact Match"),
    # Su questo corpus le metriche di sovrapposizione sono sature: una regola
    # a lessico non addestrata arriva a ROUGE-L 0.9685. La non-copy-token
    # accuracy guarda solo le posizioni in cui il gloss NON copia la sorgente,
    # quindi e' l'unica colonna che non si satura, e la tabella non puo'
    # ometterla.
    ("non_copy_token_accuracy", "Non-copy"),
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


def _run_segment(eval_dir: Path) -> str:
    """Il segmento ``run_*`` più interno sopra ``eval_dir`` (lui stesso incluso)."""
    return next(
        p for p in (eval_dir, *eval_dir.parents) if p.name.startswith("run_")
    ).name


def _discover_cells(results_dir: Path) -> dict[str, list[Path]]:
    """Map each cell to the directories holding its evals, one per run, oldest first.

    Every eval lives in ``<cella>/run_<ts>/`` or in an eval-only sub-directory of
    it (``run_<ts>/<results_subdir>/``, e.g. ``decoding-greedy``), at whatever
    depth the config nests the cell. The anchor is the LAST ``run_*`` segment
    above the file: what precedes it is the cell, what follows it is a variant of
    the same checkpoint with metrics of its own, reported as its own row
    (``qwen25-05b/sft/zero-shot/decoding-greedy``).

    An eval file with no ``run_*`` ancestor is an orphan — no run id, no link to
    the config that produced it — and is ignored: the pre-``run_*`` layout left
    such files in group directories (``qwen25-05b/sft/eval_final.json``) carrying
    numbers that belonged to no cell.
    """
    cells: dict[str, set[Path]] = {}
    for eval_file in results_dir.rglob("eval_*.json"):
        rel = eval_file.parent.relative_to(results_dir).parts
        runs = [i for i, seg in enumerate(rel) if seg.startswith("run_")]
        if not runs or runs[-1] == 0:
            logger.debug("Skipping orphan eval outside any run_*: %s", eval_file)
            continue
        i = runs[-1]
        key = "/".join(rel[:i] + rel[i + 1 :])
        cells.setdefault(key, set()).add(eval_file.parent)
    return {k: sorted(v, key=_run_segment) for k, v in cells.items()}


def find_eval_results(results_dir: Path) -> list[dict]:
    """Scan results_dir for all eval_*.json files (excluding baseline).

    Returns a list of dicts with: config_name, run_id, path, metrics.
    """
    entries = []

    if not results_dir.exists():
        logger.warning("Results directory not found: %s", results_dir)
        return entries

    for config_name, eval_dirs in sorted(
        _discover_cells(results_dir).items(), key=lambda kv: cell_sort_key(kv[0])
    ):
        # One entry per run, oldest first (run_<timestamp> sorts
        # chronologically): the latest run is the one reported.
        latest_run = eval_dirs[-1]

        # Find eval_*.json (skip eval_baseline.json — that's the zero-shot ref)
        eval_files = [
            f for f in latest_run.glob("eval_*.json") if f.name != "eval_baseline.json"
        ]

        if not eval_files:
            logger.debug("No eval_*.json in %s", latest_run)
            continue

        # Prefer eval_final.json (final metrics) — plain alphabetical sorting
        # could pick another eval file when several exist
        # "zero_shot"). Otherwise take the most recently modified eval file.
        final_candidates = [f for f in eval_files if f.name == "eval_final.json"]
        if final_candidates:
            eval_path = final_candidates[0]
        else:
            eval_path = max(eval_files, key=lambda p: p.stat().st_mtime)
        try:
            with open(eval_path, encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            logger.warning("Failed to read %s: %s", eval_path, e)
            continue

        entry = {
            "config_name": config_name,
            "run_id": _run_segment(latest_run),
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

    duplicates = find_duplicate_cells(entries)
    if duplicates:
        rows.append("")
        rows.append(
            "**Celle duplicate** — identiche su ogni metrica, quindi lo stesso "
            "esperimento sotto due nomi: non sono ablazioni indipendenti e non "
            "vanno contate due volte."
        )
        for group in duplicates:
            rows.append("- " + " ≡ ".join(f"`{n}`" for n in group))

    return "\n".join(rows)


def find_duplicate_cells(entries: list[dict]) -> list[list[str]]:
    """Group cells whose absolute metrics coincide to the printed precision.

    Two configs that differ only by inert keys (a weight re-stated at its
    default, a renamed output dir) train the same policy from the same seed and
    produce the same numbers. Reported side by side they read as independent
    evidence, so the table has to say they are not.
    """
    groups: dict[tuple, list[str]] = {}
    for entry in entries:
        key = tuple(
            round(entry["metrics"][label], 4) if label in entry["metrics"] else None
            for _, label in METRICS
        )
        groups.setdefault(key, []).append(entry["config_name"])
    return [sorted(names) for names in groups.values() if len(names) > 1]


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
    """Small multiples: one horizontal-bar panel per key metric, cells as rows.

    Three panels, all on the same 0-1 scale: ROUGE-L (saturated on this corpus:
    an untrained lexical rule reaches 0.9685), exact match and non-copy-token
    accuracy (the two that still discriminate). Metrics on another scale — chrF
    runs 0-100 — are never drawn on the same axis: plotted against a 0-1 axis,
    their labels landed ~90x above it and ``bbox_inches="tight"`` grew the
    figure to 3795 x 75823 px to include them. The exact values live in
    ``ablation_summary.md`` / ``.csv``; the chart is for the shape.
    """
    if not entries:
        logger.warning("No entries to plot")
        return

    panels = [
        label
        for label in ("ROUGE-L", "Exact Match", "Non-copy")
        if any(e["metrics"].get(label) is not None for e in entries)
    ]
    if not panels:
        logger.warning("No 0-1 metric to plot")
        return

    ordered = sorted(entries, key=lambda e: cell_sort_key(e["config_name"]))
    names = [e["config_name"] for e in ordered]
    y = np.arange(len(ordered))

    row_in = 0.32  # altezza di riga in pollici: 48 px a 150 dpi
    fig, axes = plt.subplots(
        1,
        len(panels),
        sharey=True,
        figsize=(3.2 * len(panels) + 3.6, 1.2 + row_in * len(ordered)),
        facecolor=chart_style.SURFACE,
    )
    axes = np.atleast_1d(axes)

    for ax, label in zip(axes, panels):
        ax.set_facecolor(chart_style.SURFACE)
        values = [e["metrics"].get(label) for e in ordered]
        present = [i for i, v in enumerate(values) if v is not None]
        # Barra al 50% della riga: <= 24 px, il resto è aria fra righe vicine.
        ax.barh(
            [y[i] for i in present],
            [values[i] for i in present],
            height=0.5,
            color=chart_style.SERIES_1,
            zorder=2,
        )
        for i, v in enumerate(values):
            if v is None:
                ax.text(
                    0.01, i, "n/d", va="center", fontsize=8, color=chart_style.INK_MUTED
                )
        ax.set_xlim(0, 1)
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
        ax.set_xticklabels(["0", ".25", ".50", ".75", "1"])
        ax.grid(axis="x", color=chart_style.GRID, linewidth=0.8, zorder=0)
        ax.set_title(label, loc="left", fontsize=11, color=chart_style.INK, pad=8)
        for side in ("top", "right", "bottom"):
            ax.spines[side].set_visible(False)
        ax.spines["left"].set_color(chart_style.BASELINE)
        ax.tick_params(colors=chart_style.INK_MUTED, labelsize=8, length=0)

    axes[0].set_yticks(y)
    axes[0].set_yticklabels(names, fontsize=8.5, color=chart_style.INK_SECONDARY)
    axes[0].invert_yaxis()

    fig.suptitle(
        "Latest run of each cell",
        x=0.01,
        ha="left",
        fontsize=12,
        color=chart_style.INK,
    )
    fig.text(
        0.01,
        0.005,
        "Values in ablation_summary.md. All panels share the 0-1 scale.",
        fontsize=8,
        color=chart_style.INK_MUTED,
    )
    fig.tight_layout(rect=(0, 0.02, 1, 0.97), w_pad=3)
    fig.savefig(output_path, dpi=150, facecolor=chart_style.SURFACE)
    plt.close(fig)
    logger.info("Bar chart saved to %s", output_path)


def main() -> None:
    # The table header carries a non-ASCII marker; a cp1252 console (Windows
    # default) would raise on print and lose the whole run before the files
    # are written.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

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
