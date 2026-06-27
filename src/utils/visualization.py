"""Visualization utilities for T2G training curves, reward breakdown, and evaluation plots.

Uses ``plotnine`` (ggplot2 grammar of graphics for Python) for polished,
publication-quality figures.  Falls back to matplotlib backend setup for
headless cluster environments.
"""

from __future__ import annotations

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
    ("rewards/translation_quality_reward/mean", "Translation Quality (ROUGE-L)"),
    ("rewards/structural_dense_reward/mean", "Structural (Bigram Proxy)"),
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
    train_logs = [e for e in log_history if "loss" in e or "reward" in e]
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
                        "step": entry["step"],
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
    "structural_dense_reward",
    "gloss_format_reward",
    "gloss_repetition_reward",
]

_COMPONENT_COLORS = {
    "translation_quality_reward": "#4C72B0",
    "structural_dense_reward": "#55A868",
    "gloss_format_reward": "#DD8452",
    "gloss_repetition_reward": "#C44E52",
}

_COMPONENT_LABELS = {
    "translation_quality_reward": "Translation (ROUGE-L)",
    "structural_dense_reward": "Structure (Bigram)",
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
