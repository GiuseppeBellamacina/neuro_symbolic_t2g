"""Tests for the new evaluation plots (metrics dashboard, score distribution,
difficulty breakdown) — added with the literature-based metric reordering."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")  # headless

from src.utils.visualization import (  # noqa: E402
    plot_baseline_vs_grpo_comparison,
    plot_difficulty_breakdown,
    plot_metrics_dashboard,
    plot_score_distribution,
)

FINAL = {
    "bleu_corpus": 0.3353,
    "chrf_corpus": 55.71,
    "rouge_l_mean": 0.6076,
    "pass_at_1": 0.9285,
    "gloss_f1_micro": 0.6147,
    "validity_rate": 0.992,
}

BASELINE = {
    "bleu_corpus": 0.188,
    "chrf_corpus": 41.89,
    "rouge_l_mean": 0.4664,
    "pass_at_1": 0.733,
    "gloss_f1_micro": 0.4026,
    "validity_rate": 0.9821,
}

BREAKDOWN = {
    "simple": {
        "n_prompts": 1200,
        "rouge_l_mean": 0.72,
        "bleu_sentence_mean": 0.45,
        "pass_at_1": 0.97,
        "validity_rate": 0.99,
    },
    "medium": {
        "n_prompts": 700,
        "rouge_l_mean": 0.55,
        "bleu_sentence_mean": 0.28,
        "pass_at_1": 0.90,
        "validity_rate": 0.99,
    },
    "hard": {
        "n_prompts": 100,
        "rouge_l_mean": 0.41,
        "bleu_sentence_mean": 0.18,
        "pass_at_1": 0.78,
        "validity_rate": 0.98,
    },
}


def test_metrics_dashboard_renders(tmp_path):
    out = tmp_path / "metrics_dashboard.png"
    plot_metrics_dashboard(
        BASELINE, FINAL, model_name="t", label="ck", output_path=str(out)
    )
    assert out.exists() and out.stat().st_size > 10_000


def test_metrics_dashboard_final_only(tmp_path):
    out = tmp_path / "dashboard_final_only.png"
    plot_metrics_dashboard(
        None, FINAL, model_name="t", label="ck", output_path=str(out)
    )
    assert out.exists() and out.stat().st_size > 10_000


def test_metrics_dashboard_missing_metrics(tmp_path, capsys):
    plot_metrics_dashboard(
        None, {"rouge_l_mean": 0.5}, output_path=str(tmp_path / "x.png")
    )
    out = tmp_path / "x.png"
    assert not out.exists() or out.stat().st_size < 20_000  # renders 1 panel or skips
    # at least it must not raise


def test_metrics_dashboard_accepts_deployment_block(tmp_path):
    out = tmp_path / "deployment.png"
    plot_metrics_dashboard(
        None,
        {"deployment": FINAL, "sampling": {"rouge_l_mean": 0.1}},
        output_path=str(out),
    )
    assert out.exists()


def test_comparison_separates_chrf_and_omits_missing(tmp_path, monkeypatch):
    import matplotlib.axes

    limits = []
    labels = []
    original_ylim = matplotlib.axes.Axes.set_ylim
    original_bar = matplotlib.axes.Axes.bar

    def capture_ylim(self, *args, **kwargs):
        limits.append((args, kwargs))
        return original_ylim(self, *args, **kwargs)

    def capture_bar(self, x, height, *args, **kwargs):
        labels.append((kwargs.get("label"), list(height)))
        return original_bar(self, x, height, *args, **kwargs)

    monkeypatch.setattr(matplotlib.axes.Axes, "set_ylim", capture_ylim)
    monkeypatch.setattr(matplotlib.axes.Axes, "bar", capture_bar)
    out = tmp_path / "comparison.png"
    plot_baseline_vs_grpo_comparison(
        {"rouge_l_mean": 0.4, "chrf_corpus": 40.0},
        {"rouge_l_mean": 0.6, "chrf_corpus": 55.0},
        label="SFT",
        output_path=str(out),
    )
    assert out.exists()
    assert any(values == [55.0] for _, values in labels)
    assert any(kwargs.get("bottom") == 0 and not args for args, kwargs in limits)
    assert all(name != "GRPO" for name, _ in labels)


def test_comparison_labels_sampling_as_diagnostic(tmp_path, monkeypatch):
    import matplotlib.axes

    titles = []
    original = matplotlib.axes.Axes.set_title

    def capture(self, label, *args, **kwargs):
        titles.append(label)
        return original(self, label, *args, **kwargs)

    monkeypatch.setattr(matplotlib.axes.Axes, "set_title", capture)
    out = tmp_path / "blocks.png"
    plot_baseline_vs_grpo_comparison(
        {"deployment": {"rouge_l_mean": 0.4}, "sampling": {"rouge_l_mean": 0.5}},
        {"deployment": {"rouge_l_mean": 0.6}, "sampling": {"rouge_l_mean": 0.7}},
        label="SFT",
        output_path=str(out),
    )
    assert any("Sampling (diagnostic)" in title for title in titles)


def test_score_distribution_renders(tmp_path):
    out = tmp_path / "bleu_dist.png"
    scores = [0.1, 0.2, 0.3, 0.5, 0.8, 0.9, 0.0]
    plot_score_distribution(
        scores,
        "BLEU-4 (sentence)",
        "BLEU-4 Score",
        model_name="t",
        output_path=str(out),
        valid_mask=[True] * 5 + [False, False],
    )
    assert out.exists() and out.stat().st_size > 10_000


def test_score_distribution_empty(capsys):
    plot_score_distribution([], "BLEU", "BLEU", output_path="nowhere/x.png")
    assert "No BLEU scores" in capsys.readouterr().out


def test_difficulty_breakdown_renders(tmp_path):
    out = tmp_path / "difficulty_breakdown.png"
    plot_difficulty_breakdown(BREAKDOWN, model_name="t", output_path=str(out))
    assert out.exists() and out.stat().st_size > 10_000


def test_difficulty_breakdown_empty(capsys):
    plot_difficulty_breakdown({}, output_path="nowhere/x.png")
    assert "No difficulty" in capsys.readouterr().out
