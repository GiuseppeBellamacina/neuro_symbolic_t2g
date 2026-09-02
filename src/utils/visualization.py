"""Visualization utilities for T2G training curves, reward breakdown, and evaluation plots.

Uses ``plotnine`` (ggplot2 grammar of graphics for Python) for polished,
publication-quality figures.  Falls back to matplotlib backend setup for
headless cluster environments.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import matplotlib

# Use non-interactive backend on headless systems (cluster).
# Interactive users can override this before importing visualization.
if matplotlib.get_backend().lower() == "module://matplotlib_inline.backend_inline":
    pass  # keep inline backend for notebooks
elif "DISPLAY" not in os.environ and "WAYLAND_DISPLAY" not in os.environ:
    matplotlib.use("Agg")

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Default theme — clean, modern look
# ---------------------------------------------------------------------------

_THEME = None  # cached theme instance


def _get_theme():
    """Return a plotnine theme with clean styling."""
    global _THEME
    if _THEME is None:
        from plotnine import element_blank, element_rect, element_text, theme

        _THEME = theme(figure_size=(8, 5)) + theme(
            plot_title=element_text(size=14, weight="bold", ha="center"),
            plot_subtitle=element_text(size=10, ha="center", color="#555555"),
            axis_title=element_text(size=11),
            axis_text=element_text(size=9),
            legend_title=element_text(size=10),
            legend_text=element_text(size=9),
            legend_background=element_rect(fill="white", alpha=0.85),
            legend_position="bottom",
            panel_grid_major=element_blank(),
            panel_grid_minor=element_blank(),
            panel_background=element_rect(fill="#FAFAFA"),
            plot_background=element_rect(fill="white"),
        )
    return _THEME


# ---------------------------------------------------------------------------
# Training curve plots
# ---------------------------------------------------------------------------

_PLOT_METRICS = [
    ("reward", "Mean Reward"),
    ("loss", "Loss"),
    ("eval_loss", "Eval Loss"),
    ("rewards/translation_quality_reward/mean", "Translation Quality (ROUGE-L)"),
    ("rewards/bleu_reward/mean", "BLEU-4"),
    ("rewards/gold_structure_reward/mean", "Gold Structure (Bigram vs Gold)"),
    ("rewards/structural_dense_reward/mean", "Structural (Bigram Proxy)"),
    ("rewards/viterbi_distance_reward/mean", "Viterbi Distance"),
    ("rewards/soft_viterbi_distance_reward/mean", "Soft Viterbi (DVL)"),
    ("rewards/verifier_scaled_reward/mean", "Verifier-Scaled (RECIPE)"),
    ("rewards/gloss_order_reward/mean", "Gloss Order (Edit Dist)"),
    ("rewards/gloss_format_reward/mean", "Format Reward"),
    ("rewards/gloss_repetition_reward/mean", "Repetition Penalty"),
    ("completion_length", "Completion Length"),
]


def plot_training_curves(
    trainer_state: dict[str, Any],
    model_name: str = "",
    output_path: str = "experiments/logs/figures/training_curves.png",
    degree: int = 4,
) -> None:
    """Generate training curve plots with polynomial regression overlay.

    Uses plotnine faceted layout — each metric in its own panel with
    raw data points (faded) and a polynomial trend line.

    Args:
        trainer_state: The parsed ``trainer_state.json`` dict.
        model_name: Short model name for the figure title.
        output_path: Where to save the figure.
        degree: Polynomial regression degree for trend lines.
    """
    log_history = trainer_state.get("log_history", [])
    train_logs = [
        e for e in log_history if "loss" in e or "eval_loss" in e or "reward" in e
    ]
    if not train_logs:
        print("No training log entries found.")
        return

    available_metrics = [
        (key, label)
        for key, label in _PLOT_METRICS
        if any(key in e for e in train_logs)
    ]
    if not available_metrics:
        print("No plottable metrics found.")
        return

    # Build a long-format DataFrame for plotnine
    rows: list[dict[str, Any]] = []
    for key, label in available_metrics:
        for entry in train_logs:
            if key in entry:
                rows.append(
                    {
                        "step": entry.get("step", 0),
                        "value": entry[key],
                        "metric": label,
                    }
                )

    df = pd.DataFrame(rows)
    if df.empty:
        print("No data points to plot.")
        return

    # Build polynomial trend lines per metric
    trend_rows: list[dict[str, Any]] = []
    for label in df["metric"].unique():
        sub = df[df["metric"] == label]
        x = sub["step"].to_numpy(dtype=float)
        y = sub["value"].to_numpy(dtype=float)
        if len(x) <= degree + 1:
            continue
        coeffs = np.polyfit(x, y, degree)
        x_smooth = np.linspace(x.min(), x.max(), 200)
        y_smooth = np.polyval(coeffs, x_smooth)
        for sx, sy in zip(x_smooth, y_smooth):
            trend_rows.append({"step": sx, "value": sy, "metric": label})

    df_trend = pd.DataFrame(trend_rows)

    from plotnine import (
        aes,
        element_text,
        facet_wrap,
        geom_line,
        geom_point,
        ggplot,
        ggtitle,
        labs,
        scale_y_continuous,
        theme,
    )

    n_metrics = len(available_metrics)
    n_cols = min(3, n_metrics)

    p = (
        ggplot(df, aes(x="step", y="value"))
        + geom_point(alpha=0.12, size=0.8, color="#1f77b4", na_rm=True)
        + geom_line(
            aes(x="step", y="value"),
            data=df_trend,
            color="#d62728",
            size=1.0,
            na_rm=True,
        )
        + facet_wrap("~metric", scales="free_y", ncol=n_cols)
        + scale_y_continuous(expand=(0.05, 0.1))
        + labs(x="Step", y="")
        + _get_theme()
        + theme(
            figure_size=(5.5 * n_cols, 4.2 * ((n_metrics + n_cols - 1) // n_cols)),
            strip_text=element_text(size=10, weight="bold"),
        )
    )

    title = "Training Curves"
    if model_name:
        title += f" — {model_name}"
    p += ggtitle(title, subtitle=f"Polynomial regression degree={degree}")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    p.save(
        output_path,
        dpi=150,
        width=5.5 * n_cols,
        height=4.2 * ((n_metrics + n_cols - 1) // n_cols),
        limitsize=False,
        verbose=False,
    )
    print(f"Saved: {output_path}")


# ---------------------------------------------------------------------------
# Reward breakdown plot
# ---------------------------------------------------------------------------

_COMPONENT_ORDER = [
    "translation_quality_reward",
    "bleu_reward",
    "gold_structure_reward",
    "structural_dense_reward",
    "viterbi_distance_reward",
    "soft_viterbi_distance_reward",
    "verifier_scaled_reward",
    "gloss_order_reward",
    "gloss_format_reward",
    "gloss_repetition_reward",
]

_COMPONENT_COLORS = {
    "translation_quality_reward": "#4C72B0",
    "bleu_reward": "#3498DB",
    "gold_structure_reward": "#55A868",
    "structural_dense_reward": "#8172B3",
    "viterbi_distance_reward": "#937860",
    "soft_viterbi_distance_reward": "#DA8BC3",
    "verifier_scaled_reward": "#8C8C8C",
    "gloss_order_reward": "#CCB974",
    "gloss_format_reward": "#DD8452",
    "gloss_repetition_reward": "#C44E52",
}

_COMPONENT_LABELS = {
    "translation_quality_reward": "Translation (ROUGE-L)",
    "bleu_reward": "BLEU-4",
    "gold_structure_reward": "Gold Structure",
    "structural_dense_reward": "Structure (Bigram)",
    "viterbi_distance_reward": "Viterbi",
    "soft_viterbi_distance_reward": "Soft Viterbi (DVL)",
    "verifier_scaled_reward": "Verifier (RECIPE)",
    "gloss_order_reward": "Gloss Order",
    "gloss_format_reward": "Format",
    "gloss_repetition_reward": "Repetition",
}


def plot_reward_breakdown(
    stage_breakdowns: list[dict[str, Any]],
    reward_weights: dict[str, float] | None = None,
    model_name: str = "",
    output_path: str = "experiments/logs/figures/reward_breakdown.png",
) -> None:
    """Stacked bar chart showing weighted reward contributions per stage.

    Uses plotnine for a polished grouped bar chart with component labels
    shown directly on each segment.

    Args:
        stage_breakdowns: List of dicts with ``label`` (str) and ``scores``
            (dict mapping component name → average score).
        reward_weights: Optional dict of component name → weight.
        model_name: Short model name for the figure title.
        output_path: Where to save the figure.
    """
    if not stage_breakdowns:
        print("No reward breakdown data to plot.")
        return

    all_components: set[str] = set()
    for sb in stage_breakdowns:
        all_components.update(sb["scores"].keys())
    # Only plot components with weight > 0 (skip inactive ones)
    if reward_weights is not None:
        all_components = {c for c in all_components if reward_weights.get(c, 0.0) > 0}
    components = [c for c in _COMPONENT_ORDER if c in all_components]

    if reward_weights is None:
        reward_weights = {c: 1.0 for c in components}

    # Build DataFrame
    rows: list[dict[str, Any]] = []
    for sb in stage_breakdowns:
        stage_label = sb["label"]
        cumulative = 0.0
        for c in components:
            w = reward_weights.get(c, 0.0)
            val = sb["scores"].get(c, 0.0) * w
            rows.append(
                {
                    "stage": stage_label,
                    "component": _COMPONENT_LABELS.get(c, c),
                    "value": val,
                    "cumulative": cumulative,
                }
            )
            cumulative += val

    df = pd.DataFrame(rows)
    # Sort components by order for consistent stacking
    df["component"] = pd.Categorical(
        df["component"],
        categories=[_COMPONENT_LABELS.get(c, c) for c in components],
        ordered=True,
    )
    df["stage"] = pd.Categorical(
        df["stage"],
        categories=[sb["label"] for sb in stage_breakdowns],
        ordered=True,
    )

    from plotnine import (
        aes,
        element_text,
        geom_col,
        geom_text,
        ggplot,
        ggtitle,
        labs,
        scale_fill_manual,
        scale_y_continuous,
    )
    from plotnine import theme as pn_theme

    color_map = {
        _COMPONENT_LABELS.get(c, c): _COMPONENT_COLORS.get(c, "#999999")
        for c in components
    }

    p = (
        ggplot(df, aes(x="stage", y="value", fill="component"))
        + geom_col(position="stack", width=0.55, alpha=0.88, na_rm=True)
        + geom_text(
            aes(y="cumulative + value + 0.02", label="round(value, 3)"),
            data=df[df["value"].abs() > 0.001],
            ha="center",
            size=8,
            na_rm=True,
        )
        + scale_fill_manual(values=color_map, name="")
        + scale_y_continuous(expand=(0, 0.15))
        + labs(x="", y="Weighted Reward Contribution")
        + _get_theme()
        + pn_theme(
            figure_size=(max(7, len(stage_breakdowns) * 2.5), 5.5),
            axis_text_x=element_text(angle=0, ha="center"),
        )
    )

    title = "Reward Component Breakdown"
    if model_name:
        title += f" — {model_name}"
    p += ggtitle(title)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    p.save(
        output_path,
        dpi=150,
        width=max(7, len(stage_breakdowns) * 2.5),
        height=5.5,
        limitsize=False,
        verbose=False,
    )
    print(f"Saved: {output_path}")


# ---------------------------------------------------------------------------
# Baseline vs GRPO comparison
# ---------------------------------------------------------------------------


def plot_baseline_vs_grpo(
    baseline_pass1: float,
    grpo_pass1: float,
    model_name: str = "",
    output_path: str = "experiments/logs/figures/baseline_vs_grpo.png",
) -> None:
    """Clean comparison bar chart: baseline vs post-GRPO Pass@1.

    Uses plotnine with direct value labels and delta annotation.
    """
    df = pd.DataFrame(
        {
            "Model": ["Baseline", "Post-GRPO"],
            "Pass@1": [baseline_pass1, grpo_pass1],
        }
    )
    df["Model"] = pd.Categorical(
        df["Model"], categories=["Baseline", "Post-GRPO"], ordered=True
    )

    from plotnine import (
        aes,
        geom_col,
        geom_text,
        ggplot,
        ggtitle,
        labs,
        scale_fill_manual,
        scale_y_continuous,
    )

    color_map = {"Baseline": "#4C72B0", "Post-GRPO": "#DD8452"}

    p = (
        ggplot(df, aes(x="Model", y="Pass@1", fill="Model"))
        + geom_col(width=0.35, alpha=0.88, na_rm=True)
        + geom_text(
            aes(label="round(Pass@1, 4)"),
            va="bottom",
            nudge_y=0.008,
            size=12,
            weight="bold",
            na_rm=True,
        )
        + scale_fill_manual(values=color_map, guide=False)
        + scale_y_continuous(
            expand=(0, 0.12), limits=(0, max(baseline_pass1, grpo_pass1) * 1.25 or 1.0)
        )
        + labs(x="", y="Pass@1 (ROUGE-L >= 0.3)")
        + _get_theme()
        + ggtitle(
            (
                f"Baseline vs Post-GRPO — {model_name}"
                if model_name
                else "Baseline vs Post-GRPO"
            ),
            subtitle=f"Delta = {grpo_pass1 - baseline_pass1:+.4f}",
        )
    )

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    p.save(output_path, dpi=150, width=6, height=5, limitsize=False, verbose=False)
    print(f"Saved: {output_path}")


# ---------------------------------------------------------------------------
# Completion length distribution
# ---------------------------------------------------------------------------


def plot_completion_length_distribution(
    completions: list[str],
    valid_mask: list[bool] | None = None,
    title: str = "Gloss Sequence Length Distribution",
    output_path: str = "experiments/logs/figures/completion_lengths.png",
) -> None:
    """Histogram of gloss sequence lengths split by valid vs invalid.

    Uses plotnine with semi-transparent overlapping histograms.
    """
    lengths = [len(c.split()) for c in completions]
    if valid_mask is None:
        valid_mask = [True] * len(completions)

    rows: list[dict[str, Any]] = []
    for length, v in zip(lengths, valid_mask):
        rows.append({"length": length, "status": "Valid" if v else "Invalid"})

    df = pd.DataFrame(rows)

    from plotnine import (
        aes,
        geom_histogram,
        ggplot,
        ggtitle,
        labs,
        scale_fill_manual,
    )

    color_map = {"Valid": "#2ca02c", "Invalid": "#d62728"}

    binwidth = max(1, int(max(lengths or [1]) / 25))
    p = (
        ggplot(df, aes(x="length", fill="status"))
        + geom_histogram(
            binwidth=binwidth,
            alpha=0.78,
            position="dodge",
            na_rm=True,
            color="white",
            size=0.15,
        )
        + scale_fill_manual(values=color_map, name="")
        + labs(x="Gloss Sequence Length (tokens)", y="Count")
        + ggtitle(title)
        + _get_theme()
    )

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    p.save(output_path, dpi=150, width=9, height=5, limitsize=False, verbose=False)
    print(f"Saved: {output_path}")


# ---------------------------------------------------------------------------
# Evaluation plots (matplotlib-based for more flexibility)
# ---------------------------------------------------------------------------


def plot_rouge_distribution(
    rouge_scores: list[float],
    model_name: str = "",
    output_path: str = "experiments/logs/figures/rouge_distribution.png",
) -> None:
    """Histogram of ROUGE-L scores across all completions.

    Shows the distribution of translation quality, with vertical lines
    for mean and median.

    Args:
        rouge_scores: List of ROUGE-L F1 scores.
        model_name: Short model name for the title.
        output_path: Where to save the figure.
    """
    import matplotlib.pyplot as plt

    if not rouge_scores:
        print("No ROUGE-L scores to plot.")
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(rouge_scores, bins=50, edgecolor="black", color="#4C72B0", alpha=0.7)
    mean_val = np.mean(rouge_scores)
    median_val = np.median(rouge_scores)
    ax.axvline(
        mean_val,
        color="#C44E52",
        linestyle="--",
        linewidth=2,
        label=f"Mean={mean_val:.4f}",
    )
    ax.axvline(
        median_val,
        color="#55A868",
        linestyle="--",
        linewidth=2,
        label=f"Median={median_val:.4f}",
    )
    ax.set_xlabel("ROUGE-L F1 Score")
    ax.set_ylabel("Count")
    title = "ROUGE-L Score Distribution"
    if model_name:
        title += f" — {model_name}"
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend()
    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_score_distribution(
    scores: list[float],
    metric_name: str,
    xlabel: str,
    model_name: str = "",
    output_path: str = "",
    valid_mask: list[bool] | None = None,
) -> None:
    """Histogram of per-completion scores for ANY metric (BLEU, chrF, …).

    Mirrors ``plot_rouge_distribution`` (mean/median lines) with an
    optional valid/invalid overlay (green/red) when ``valid_mask`` is
    provided — the same visual language as the completion length plot.

    Args:
        scores: Per-completion scores.
        metric_name: Metric display name for the title.
        xlabel: X-axis label.
        model_name: Short model name for the title.
        output_path: Where to save the figure.
        valid_mask: Optional per-completion validity flags — when given,
            valid scores are drawn in green, invalid in red.
    """
    import matplotlib.pyplot as plt

    if not scores:
        print(f"No {metric_name} scores to plot.")
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    if valid_mask is not None and len(valid_mask) == len(scores):
        valid_scores = [s for s, v in zip(scores, valid_mask) if v]
        invalid_scores = [s for s, v in zip(scores, valid_mask) if not v]
        ax.hist(
            valid_scores,
            bins=50,
            edgecolor="black",
            color="#55A868",
            alpha=0.7,
            label=f"Valid (n={len(valid_scores)})",
        )
        if invalid_scores:
            ax.hist(
                invalid_scores,
                bins=50,
                edgecolor="black",
                color="#C44E52",
                alpha=0.7,
                label=f"Invalid (n={len(invalid_scores)})",
            )
    else:
        ax.hist(scores, bins=50, edgecolor="black", color="#4C72B0", alpha=0.7)

    mean_val = float(np.mean(scores))
    median_val = float(np.median(scores))
    ax.axvline(
        mean_val,
        color="#C44E52",
        linestyle="--",
        linewidth=2,
        label=f"Mean={mean_val:.4f}",
    )
    ax.axvline(
        median_val,
        color="#55A868",
        linestyle="--",
        linewidth=2,
        label=f"Median={median_val:.4f}",
    )
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Count")
    title = f"{metric_name} Distribution"
    if model_name:
        title += f" — {model_name}"
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)
    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_metrics_dashboard(
    baseline_metrics: dict[str, Any] | None,
    final_metrics: dict[str, Any],
    model_name: str = "",
    label: str = "Checkpoint",
    output_path: str = "",
) -> None:
    """THE comparison figure: headline metrics, baseline vs checkpoint.

    One row of panels — ordered by the literature-based metric hierarchy
    (BLEU-4 corpus first, then chrF corpus, ROUGE-L, Pass@1, Gloss F1,
    validity) — each panel a two-bar chart with the delta annotated.
    Designed as the single "thesis headline" figure summarizing a run.

    Args:
        baseline_metrics: Baseline eval results (None → final-only panels).
        final_metrics: Checkpoint eval results.
        model_name: Short model name for the suptitle.
        label: Display name for the checkpoint bars.
        output_path: Where to save the figure.
    """
    import matplotlib.pyplot as plt

    # (key, display, fmt) — ordered by relevance (docs/EVALUATION.md):
    # BLEU-4 corpus (literature standard) → chrF → ROUGE-L → Pass@1 →
    # Gloss F1 → validity.
    panels: list[tuple[str, str, str]] = [
        ("bleu_corpus", "BLEU-4 (corpus)", ".4f"),
        ("chrf_corpus", "chrF2 (corpus)", ".2f"),
        ("rouge_l_mean", "ROUGE-L (mean)", ".4f"),
        ("pass_at_1", "Pass@1", ".4f"),
        ("gloss_f1_micro", "Gloss F1 (micro)", ".4f"),
        ("validity_rate", "Validity", ".4f"),
    ]
    present = [(k, d, f) for k, d, f in panels if k in final_metrics]
    if not present:
        print("No dashboard metrics found — skipping plot.")
        return
    if baseline_metrics is not None:
        present = [(k, d, f) for k, d, f in present if k in baseline_metrics]
    if not present:
        print("No overlapping dashboard metrics — skipping plot.")
        return

    n = len(present)
    fig, axes = plt.subplots(1, n, figsize=(2.8 * n, 4.6))
    if n == 1:
        axes = [axes]
    colors = {"baseline": "#8172B2", "final": "#4C72B0"}

    for ax, (key, display, fmt) in zip(axes, present):
        final_val = float(final_metrics[key])
        if baseline_metrics is not None:
            base_val = float(baseline_metrics[key])
            bars = ax.bar(
                ["Baseline", label],
                [base_val, final_val],
                color=[colors["baseline"], colors["final"]],
                alpha=0.85,
                width=0.6,
            )
            delta = final_val - base_val
            sign = "+" if delta >= 0 else ""
            # relative delta (guard div-by-zero)
            rel = delta / base_val if base_val != 0 else float("inf")
            if abs(rel) != float("inf"):
                delta_txt = f"{sign}{delta:{fmt}}\n({sign}{100 * rel:.1f}%)"
            else:
                delta_txt = f"{sign}{delta:{fmt}}"
        else:
            bars = ax.bar(
                [label], [final_val], color=colors["final"], alpha=0.85, width=0.6
            )
            delta_txt = ""
        ax.set_title(display, fontsize=11, fontweight="bold")
        ax.set_ylim(0, max(final_val, max(b.get_height() for b in bars)) * 1.25)
        for bar in bars:
            h = bar.get_height()
            ax.annotate(
                f"{h:{fmt}}",
                xy=(bar.get_x() + bar.get_width() / 2, h),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                fontsize=9,
                fontweight="bold",
            )
        if delta_txt:
            ax.annotate(
                delta_txt,
                xy=(0.5, 0.92),
                xycoords="axes fraction",
                ha="center",
                fontsize=8,
                color="#1a7a1a" if delta_txt.startswith("+") else "#C44E52",
            )

    suptitle = "Metrics Dashboard"
    if model_name:
        suptitle += f" — {model_name}"
    fig.suptitle(suptitle, fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_difficulty_breakdown(
    difficulty_breakdown: dict[str, dict[str, float]],
    model_name: str = "",
    output_path: str = "",
) -> None:
    """Grouped bars: key metrics per gold-difficulty level (simple/medium/hard).

    Answers "where does the model do well/badly?" — the per-difficulty
    companion of the metrics dashboard. Difficulty follows the training
    heuristic (gold gloss token count: ≤5 simple, ≤15 medium, >15 hard).
    """
    import matplotlib.pyplot as plt

    if not difficulty_breakdown:
        print("No difficulty breakdown to plot.")
        return

    # Preserve the canonical order; unknown levels appended alphabetically
    levels = [lv for lv in ("simple", "medium", "hard") if lv in difficulty_breakdown]
    levels += sorted(lv for lv in difficulty_breakdown if lv not in levels)
    metrics: list[tuple[str, str]] = [
        ("rouge_l_mean", "ROUGE-L"),
        ("bleu_sentence_mean", "BLEU (sent)"),
        ("pass_at_1", "Pass@1"),
        ("validity_rate", "Validity"),
    ]
    present = [
        (k, d)
        for k, d in metrics
        if all(k in difficulty_breakdown[lv] for lv in levels)
    ]
    if not present or not levels:
        print("No difficulty metrics found — skipping plot.")
        return

    x = np.arange(len(levels))
    width = 0.8 / len(present)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    palette = ["#4C72B0", "#55A868", "#C44E52", "#8172B2"]
    for i, (key, display) in enumerate(present):
        vals = [float(difficulty_breakdown[lv].get(key, 0.0)) for lv in levels]
        offset = (i - (len(present) - 1) / 2) * width
        bars = ax.bar(
            x + offset,
            vals,
            width * 0.92,
            label=display,
            color=palette[i % len(palette)],
            alpha=0.85,
        )
        for bar in bars:
            h = bar.get_height()
            ax.annotate(
                f"{h:.3f}",
                xy=(bar.get_x() + bar.get_width() / 2, h),
                xytext=(0, 2),
                textcoords="offset points",
                ha="center",
                fontsize=8,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(
        [
            f"{lv}\n(n={difficulty_breakdown[lv].get('n_prompts', '?')} prompts)"
            for lv in levels
        ],
        fontsize=10,
    )
    ax.set_ylabel("Score")
    title = "Metrics by Gold Difficulty"
    if model_name:
        title += f" — {model_name}"
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)
    ax.set_ylim(0, 1.05)  # all metrics here are in [0,1]
    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_pass_at_k_curve(
    pass_at_k: dict[str, float],
    model_name: str = "",
    output_path: str = "experiments/logs/figures/pass_at_k.png",
) -> None:
    """Line chart showing Pass@k for k=1,2,...,N.

    Args:
        pass_at_k: Dict mapping "pass@k" → float (e.g. {"pass@1": 0.27, ...}).
        model_name: Short model name for the title.
        output_path: Where to save the figure.
    """
    import matplotlib.pyplot as plt

    if not pass_at_k:
        print("No Pass@k data to plot.")
        return

    ks = sorted(int(k.replace("pass@", "")) for k in pass_at_k.keys())
    vals = [pass_at_k[f"pass@{k}"] for k in ks]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(ks, vals, "o-", color="#4C72B0", linewidth=2, markersize=8)
    for k, v in zip(ks, vals):
        ax.annotate(
            f"{v:.3f}",
            (k, v),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
            fontsize=9,
        )
    ax.set_xlabel("k (number of completions)")
    ax.set_ylabel("Pass@k Rate")
    ax.set_ylim(-0.05, 1.05)
    ax.set_xticks(ks)
    title = "Pass@k Curve"
    if model_name:
        title += f" — {model_name}"
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_error_breakdown(
    error_distribution: dict[str, int],
    model_name: str = "",
    output_path: str = "experiments/logs/figures/error_breakdown.png",
) -> None:
    """Pie chart of error types.

    Args:
        error_distribution: Dict mapping error type → count.
        model_name: Short model name for the title.
        output_path: Where to save the figure.
    """
    import matplotlib.pyplot as plt

    if not error_distribution:
        print("No error data to plot.")
        return

    labels = list(error_distribution.keys())
    sizes = list(error_distribution.values())
    colors = plt.cm.Set3(np.linspace(0, 1, len(labels)))

    fig, ax = plt.subplots(figsize=(8, 6))
    wedges, texts, autotexts = ax.pie(
        sizes, labels=labels, autopct="%1.1f%%", colors=colors, startangle=90
    )
    for t in texts:
        t.set_fontsize(10)
    for t in autotexts:
        t.set_fontsize(9)
        t.set_fontweight("bold")
    title = "Error Distribution"
    if model_name:
        title += f" — {model_name}"
    ax.set_title(title, fontsize=13, fontweight="bold")
    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_validity_pie(
    valid_count: int,
    invalid_count: int,
    model_name: str = "",
    output_path: str = "experiments/logs/figures/validity_pie.png",
) -> None:
    """Pie chart of valid vs invalid completions.

    Args:
        valid_count: Number of valid completions.
        invalid_count: Number of invalid completions.
        model_name: Short model name for the title.
        output_path: Where to save the figure.
    """
    import matplotlib.pyplot as plt

    total = valid_count + invalid_count
    if total == 0:
        print("No validity data to plot.")
        return

    fig, ax = plt.subplots(figsize=(6, 6))
    sizes = [valid_count, invalid_count]
    labels = [f"Valid ({valid_count})", f"Invalid ({invalid_count})"]
    colors = ["#55A868", "#C44E52"]
    ax.pie(
        sizes,
        labels=labels,
        autopct="%1.1f%%",
        colors=colors,
        startangle=90,
        textprops={"fontsize": 11},
    )
    title = "Gloss Validity Rate"
    if model_name:
        title += f" — {model_name}"
    ax.set_title(title, fontsize=13, fontweight="bold")
    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_reward_radar(
    reward_breakdown: dict[str, float],
    reward_weights: dict[str, float] | None = None,
    model_name: str = "",
    output_path: str = "experiments/logs/figures/reward_radar.png",
) -> None:
    """Radar chart of reward component scores.

    Args:
        reward_breakdown: Dict mapping component name → average score.
        reward_weights: Optional dict of component name → weight.
        model_name: Short model name for the title.
        output_path: Where to save the figure.
    """
    import matplotlib.pyplot as plt

    if not reward_breakdown:
        print("No reward breakdown data to plot.")
        return

    # Short labels
    label_map = {
        "translation_quality_reward": "Translation",
        "bleu_reward": "BLEU-4",
        "gold_structure_reward": "Gold Struct",
        "structural_dense_reward": "Struct Dense",
        "viterbi_distance_reward": "Viterbi",
        "soft_viterbi_distance_reward": "Soft Viterbi",
        "verifier_scaled_reward": "Verifier",
        "gloss_order_reward": "Gloss Order",
        "gloss_format_reward": "Format",
        "gloss_repetition_reward": "Repetition",
    }

    components = list(reward_breakdown.keys())
    labels = [label_map.get(c, c) for c in components]
    values = [reward_breakdown[c] for c in components]
    n = len(components)

    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    values_closed = values + values[:1]
    angles_closed = angles + angles[:1]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw={"polar": True})
    ax.fill(angles_closed, values_closed, alpha=0.25, color="#4C72B0")
    ax.plot(
        angles_closed, values_closed, "o-", color="#4C72B0", linewidth=2, markersize=6
    )
    ax.set_xticks(angles)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylim(-1.05, 1.05)
    title = "Reward Component Radar"
    if model_name:
        title += f" — {model_name}"
    ax.set_title(title, fontsize=13, fontweight="bold", pad=20)
    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {output_path}")


def dump_completion_examples(
    completions: list[str],
    references: list[str],
    rouge_scores: list[float],
    prompts: list[str] | None = None,
    n_examples: int = 10,
    model_name: str = "",
    output_dir: str = "experiments/figures",
) -> str:
    """Dump best/worst completion examples as JSON + self-contained HTML.

    Produces a readable HTML table with sortable, searchable, color-coded
    rows and a companion JSON file for programmatic consumption.

    Args:
        completions: Generated gloss sequences (flat: one per completion).
        references: Gold reference glosses, ONE PER COMPLETION (aligned
            with ``completions``; in multi-sample evals a prompt's gold
            repeats ``num_samples`` times). Passing a per-PROMPT reference
            list here misaligns gold vs prompt whenever ``num_samples > 1``.
        rouge_scores: ROUGE-L scores per completion.
        prompts: English prompts (optional). If provided, shown in the
            table and used to group completions by prompt.
        n_examples: Total examples (half best, half worst).
        model_name: Short model name for the title.
        output_dir: Directory for ``completion_examples.json`` and ``.html``.

    Returns:
        Path to the HTML file.
    """
    if not completions:
        print("No completion examples to dump.")
        return ""

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Validate alignment ──────────────────────────────────────────────
    # completions/references/rouge_scores must be FLAT and aligned (one
    # entry per completion). A per-PROMPT reference list (the classic
    # multi-sample mistake) silently misaligns gold vs prompt — fail fast
    # instead of rendering a wrong table.
    if not (len(completions) == len(references) == len(rouge_scores)):
        raise ValueError(
            "dump_completion_examples: misaligned inputs — completions="
            f"{len(completions)}, references={len(references)}, "
            f"rouge_scores={len(rouge_scores)}. All three must be FLAT and "
            "aligned (one entry per completion, gold repeated per sample)."
        )
    if prompts is not None and len(prompts) != len(completions):
        raise ValueError(
            "dump_completion_examples: prompts length "
            f"({len(prompts)}) != completions length ({len(completions)}). "
            "Pass one prompt per completion (repeated per sample) or None."
        )

    # ── Select best + worst ─────────────────────────────────────────────
    # Multi-sample evals (num_samples > 1) produce several completions per
    # prompt: completions/references/rouge_scores/prompts are all FLAT and
    # aligned (one entry per completion). Group them by prompt so each
    # best/worst slot shows a DISTINCT prompt — the prompt's best (resp.
    # worst) completion. With num_samples == 1 every group holds a single
    # entry and selection degrades to plain per-completion ranking.
    groups: dict[str, list[tuple[int, str, str, float]]] = {}
    for i, (comp, ref, rl) in enumerate(
        zip(completions, references, rouge_scores, strict=True)
    ):
        key = prompts[i] if prompts else ref
        groups.setdefault(key, []).append((i, comp, ref, rl))

    n_half = n_examples // 2
    # Best: highest-ROUGE completion of the top prompts (max first)
    best = sorted(
        (max(entries, key=lambda e: e[3]) for entries in groups.values()),
        key=lambda e: e[3],
        reverse=True,
    )[:n_half]
    # Worst: lowest-ROUGE completion of the bottom prompts (min first)
    worst = sorted(
        (min(entries, key=lambda e: e[3]) for entries in groups.values()),
        key=lambda e: e[3],
    )[:n_half]

    # ── Build entries ───────────────────────────────────────────────────
    def _build_entries(group: list, label: str) -> list[dict]:
        entries: list[dict] = []
        for orig_idx, comp, ref, rl in group:
            trunc_comp = comp[:120] + "..." if len(comp) > 120 else comp
            trunc_ref = ref[:120] + "..." if len(ref) > 120 else ref
            entry: dict[str, Any] = {
                "group": label,
                "index": orig_idx,
                "rouge_l": round(rl, 4),
                "gold": trunc_ref,
                "prediction": trunc_comp,
            }
            if prompts and orig_idx < len(prompts):
                prompt = prompts[orig_idx]
                entry["prompt"] = prompt[:200] + "..." if len(prompt) > 200 else prompt
            entries.append(entry)
        return entries

    examples: list[dict[str, Any]] = []
    examples.extend(_build_entries(best, "best"))
    examples.extend(_build_entries(worst, "worst"))

    # ── Write JSON ──────────────────────────────────────────────────────
    json_path = out_dir / "completion_examples.json"
    json_path.write_text(
        json.dumps(
            {"model": model_name, "n_examples": n_examples, "examples": examples},
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"Saved: {json_path}")

    # ── Write self-contained HTML ───────────────────────────────────────
    html_path = out_dir / "completion_examples.html"
    html = _render_completion_html(model_name, examples)
    html_path.write_text(html, encoding="utf-8")
    print(f"Saved: {html_path}")

    return str(html_path)


def _render_completion_html(model_name: str, examples: list[dict[str, Any]]) -> str:
    """Render best/worst completions as a self-contained HTML table."""

    def _color_for(rl: float) -> str:
        if rl >= 0.8:
            return "#1a7a1a"
        if rl >= 0.5:
            return "#55A868"
        if rl >= 0.3:
            return "#DD8452"
        return "#C44E52"

    def _badge_for(group: str) -> str:
        if group == "best":
            return '<span style="background:#d4edda;color:#155724;padding:2px 8px;border-radius:4px;font-weight:bold">BEST</span>'
        return '<span style="background:#f8d7da;color:#721c24;padding:2px 8px;border-radius:4px;font-weight:bold">WORST</span>'

    rows_html: list[str] = []
    for ex in examples:
        rl = ex["rouge_l"]
        color = _color_for(rl)
        badge = _badge_for(ex["group"])
        gold = ex["gold"]
        pred = ex["prediction"]
        prompt = ex.get("prompt", "")

        # Highlight diffs: wrap mismatched words in spans
        # Simple character-level diff for visual alignment
        pred_highlighted = _highlight_diff(gold, pred)

        prompt_cell = f'<td class="prompt">{prompt}</td>' if prompt else ""
        rows_html.append(f"""<tr class="{ex['group']}">
    {prompt_cell}
    <td style="color:{color};font-weight:bold;text-align:center">{rl:.4f}</td>
    <td>{badge}</td>
    <td class="mono">{gold}</td>
    <td class="mono">{pred_highlighted}</td>
</tr>""")

    has_prompt = any(ex.get("prompt") for ex in examples)
    prompt_col = "<th>Prompt</th>" if has_prompt else ""

    title = (
        f"Completion Examples — {model_name}" if model_name else "Completion Examples"
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{title}</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; margin: 20px; background: #f5f5f5; }}
  h1 {{ color: #333; font-size: 1.4em; }}
  table {{ width: 100%; border-collapse: collapse; background: white; border-radius: 8px; overflow: hidden; box-shadow: 0 1px 4px rgba(0,0,0,.08); }}
  th {{ background: #f0f0f0; padding: 10px 12px; text-align: left; font-size: 0.85em; text-transform: uppercase; letter-spacing: .03em; color: #555; }}
  td {{ padding: 10px 12px; border-bottom: 1px solid #eee; vertical-align: top; font-size: 0.9em; }}
  tr.best {{ background: #f9fdf9; }}
  tr.worst {{ background: #fef9f9; }}
  tr:hover {{ background: #eef6ff; }}
  .mono {{ font-family: 'SF Mono', 'Fira Code', 'Consolas', monospace; font-size: 0.85em; white-space: pre-wrap; word-break: break-all; max-width: 400px; }}
  .prompt {{ max-width: 300px; color: #555; font-style: italic; }}
  .diff-del {{ color: #C44E52; text-decoration: line-through; }}
  .diff-add {{ color: #1a7a1a; font-weight: bold; }}
  .summary {{ color: #888; font-size: 0.85em; margin-bottom: 12px; }}
</style>
</head>
<body>
<h1>{title}</h1>
<p class="summary">{len(examples)} examples ({len(examples)//2} best, {len(examples)//2} worst) sorted by ROUGE-L</p>
<table>
<thead>
<tr>
  {prompt_col}
  <th>ROUGE-L</th>
  <th>Group</th>
  <th>Gold</th>
  <th>Prediction</th>
</tr>
</thead>
<tbody>
{"".join(rows_html)}
</tbody>
</table>
</body>
</html>"""


def _highlight_diff(gold: str, pred: str) -> str:
    """Simple word-level diff: mark different words in prediction."""
    g_words = gold.split()
    p_words = pred.split()

    if g_words == p_words:
        return pred

    result: list[str] = []
    # Align by position; for now a simple zip comparison
    max_len = max(len(g_words), len(p_words))
    for i in range(max_len):
        gw = g_words[i] if i < len(g_words) else ""
        pw = p_words[i] if i < len(p_words) else ""
        if gw == pw:
            result.append(pw)
        else:
            if gw:
                result.append(f'<span class="diff-del">{gw}</span>')
            if pw:
                result.append(f'<span class="diff-add">{pw}</span>')
    return " ".join(result)


def plot_baseline_vs_grpo_comparison(
    baseline_metrics: dict[str, float],
    grpo_metrics: dict[str, float],
    model_name: str = "",
    label: str = "GRPO",
    output_path: str = "experiments/logs/figures/baseline_vs_grpo.png",
) -> None:
    """Grouped bar chart comparing baseline vs a checkpoint on key metrics.

    Only metrics present in BOTH dicts are plotted, so older eval JSONs
    that lack the newer keys (BLEU/chrF/gloss F1) render gracefully.

    Args:
        baseline_metrics: Dict with keys like rouge_l_mean, pass_at_1,
            validity_rate, bleu_sentence_mean, chrf_sentence_mean, etc.
        grpo_metrics: Same keys for the checkpoint model.
        model_name: Short model name for the title.
        label: Display name for the checkpoint series (e.g. the eval file
            stem — "GRPO" for GRPO checkpoints, the checkpoint name for SFT).
        output_path: Where to save the figure.
    """
    import matplotlib.pyplot as plt

    metrics_to_compare = [
        ("rouge_l_mean", "ROUGE-L"),
        ("valid_rouge_l_mean", "Valid ROUGE-L"),
        ("pass_at_1", "Pass@1"),
        ("exact_match", "Exact Match"),
        ("validity_rate", "Validity Rate"),
        ("bleu_sentence_mean", "BLEU (sent)"),
        ("bleu_corpus", "BLEU (corpus)"),
        ("chrf_sentence_mean", "chrF2 (sent)"),
        ("chrf_corpus", "chrF2 (corpus)"),
        ("gloss_f1_sentence_mean", "Gloss F1 (sent)"),
        ("gloss_f1_micro", "Gloss F1 (micro)"),
        ("gloss_validity_rate", "Gloss Validity"),
    ]

    present = [
        m
        for m in metrics_to_compare
        if m[0] in baseline_metrics and m[0] in grpo_metrics
    ]
    labels = [m[1] for m in present]
    baseline_vals = [baseline_metrics[m[0]] for m in present]
    grpo_vals = [grpo_metrics[m[0]] for m in present]

    if not labels:
        print("No overlapping metrics to compare.")
        return

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(max(10, len(labels) * 1.3), 6))
    bars1 = ax.bar(
        x - width / 2,
        baseline_vals,
        width,
        label="Baseline",
        color="#8172B2",
        alpha=0.85,
    )
    bars2 = ax.bar(
        x + width / 2, grpo_vals, width, label=label, color="#4C72B0", alpha=0.85
    )

    for bars in [bars1, bars2]:
        for bar in bars:
            h = bar.get_height()
            ax.annotate(
                f"{h:.3f}",
                xy=(bar.get_x() + bar.get_width() / 2, h),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                fontsize=8,
            )

    ax.set_ylabel("Score")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylim(0, max(max(baseline_vals), max(grpo_vals)) * 1.2)
    title = f"Baseline vs {label} Comparison"
    if model_name:
        title += f" — {model_name}"
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {output_path}")
