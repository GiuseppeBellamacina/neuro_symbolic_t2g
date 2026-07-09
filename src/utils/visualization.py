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
    ax.set_ylim(0, 1.05)
    title = "Reward Component Radar"
    if model_name:
        title += f" — {model_name}"
    ax.set_title(title, fontsize=13, fontweight="bold", pad=20)
    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_completion_examples(
    completions: list[str],
    references: list[str],
    rouge_scores: list[float],
    n_examples: int = 10,
    model_name: str = "",
    output_path: str = "experiments/logs/figures/completion_examples.png",
) -> None:
    """Table-like figure showing example completions vs gold references.

    Shows the top N examples sorted by ROUGE-L (best and worst).

    Args:
        completions: Generated gloss sequences.
        references: Gold reference glosses.
        rouge_scores: ROUGE-L scores per completion.
        n_examples: Number of examples to show (half best, half worst).
        model_name: Short model name for the title.
        output_path: Where to save the figure.
    """
    import matplotlib.pyplot as plt

    if not completions:
        print("No completion examples to plot.")
        return

    # Sort by ROUGE-L and pick best/worst
    sorted_idx = sorted(range(len(rouge_scores)), key=lambda i: rouge_scores[i])
    n_half = n_examples // 2
    selected = sorted_idx[:n_half] + sorted_idx[-n_half:]
    # Reverse so best is at top
    selected = list(reversed(selected))

    fig, axes = plt.subplots(
        len(selected), 1, figsize=(12, max(6, len(selected) * 1.5))
    )
    if len(selected) == 1:
        axes = [axes]

    for ax, idx in zip(axes, selected):
        comp = (
            completions[idx][:80] + "..."
            if len(completions[idx]) > 80
            else completions[idx]
        )
        ref = (
            references[idx][:80] + "..."
            if len(references[idx]) > 80
            else references[idx]
        )
        rl = rouge_scores[idx]
        color = "#55A868" if rl >= 0.5 else "#DD8452" if rl >= 0.2 else "#C44E52"
        ax.text(
            0.01,
            0.7,
            f"ROUGE-L: {rl:.3f}",
            fontsize=9,
            color=color,
            fontweight="bold",
            transform=ax.transAxes,
        )
        ax.text(
            0.01,
            0.4,
            f"GOLD:  {ref}",
            fontsize=8,
            fontfamily="monospace",
            transform=ax.transAxes,
        )
        ax.text(
            0.01,
            0.1,
            f"PRED:  {comp}",
            fontsize=8,
            fontfamily="monospace",
            transform=ax.transAxes,
        )
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
        ax.axhline(y=0.0, color="#CCCCCC", linewidth=0.5)

    title = "Completion Examples (Best & Worst)"
    if model_name:
        title += f" — {model_name}"
    fig.suptitle(title, fontsize=13, fontweight="bold")
    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_baseline_vs_grpo_comparison(
    baseline_metrics: dict[str, float],
    grpo_metrics: dict[str, float],
    model_name: str = "",
    output_path: str = "experiments/logs/figures/baseline_vs_grpo.png",
) -> None:
    """Grouped bar chart comparing baseline vs GRPO on key metrics.

    Args:
        baseline_metrics: Dict with keys like rouge_l, pass_at_1, validity_rate, etc.
        grpo_metrics: Same keys for the GRPO model.
        model_name: Short model name for the title.
        output_path: Where to save the figure.
    """
    import matplotlib.pyplot as plt

    metrics_to_compare = [
        ("rouge_l", "ROUGE-L"),
        ("pass_at_1", "Pass@1"),
        ("exact_match", "Exact Match"),
        ("validity_rate", "Validity Rate"),
        ("gloss_validity_rate", "Gloss Validity"),
    ]

    labels = [
        m[1]
        for m in metrics_to_compare
        if m[0] in baseline_metrics and m[0] in grpo_metrics
    ]
    baseline_vals = [
        baseline_metrics[m[0]]
        for m in metrics_to_compare
        if m[0] in baseline_metrics and m[0] in grpo_metrics
    ]
    grpo_vals = [
        grpo_metrics[m[0]]
        for m in metrics_to_compare
        if m[0] in baseline_metrics and m[0] in grpo_metrics
    ]

    if not labels:
        print("No overlapping metrics to compare.")
        return

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(
        x - width / 2,
        baseline_vals,
        width,
        label="Baseline",
        color="#8172B2",
        alpha=0.85,
    )
    bars2 = ax.bar(
        x + width / 2, grpo_vals, width, label="GRPO", color="#4C72B0", alpha=0.85
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
    title = "Baseline vs GRPO Comparison"
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
