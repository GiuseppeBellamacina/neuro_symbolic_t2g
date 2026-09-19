"""Tests for src/utils/ablation_summary.py's result discovery.

``find_eval_results`` used a fixed-depth ``iterdir()``/``iterdir()`` scan that
only ever found the legacy flat layout (``results/<tag>/run_*/``). The current
nested layout (``results/qwen25-05b/grpo/zero-shot/run_*/``, depth varies per
cell) was invisible to it — a live run against the real post-campaign results
tree confirmed this: campaign_report.py found 39 runs, ablation_summary.py
found 0. Same bug class as src/utils/run_paths.py's fix, applied here to the
read side.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.ablation_summary import _discover_cells, find_eval_results


def _write_eval(path: Path, **metrics) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics), encoding="utf-8")


def test_discover_cells_finds_nested_layout_at_varying_depth(tmp_path):
    """qwen25-05b/grpo/zero-shot (2 levels) and qwen25-05b/ablations/loss/
    dr-grpo (3 levels) must both be discovered as distinct cells."""
    _write_eval(
        tmp_path / "qwen25-05b/grpo/zero-shot/run_1/eval_final.json",
        rouge_l_mean=0.5,
    )
    _write_eval(
        tmp_path / "qwen25-05b/ablations/loss/dr-grpo/run_1/eval_final.json",
        rouge_l_mean=0.6,
    )
    cells = _discover_cells(tmp_path)
    assert set(cells) == {
        "qwen25-05b/grpo/zero-shot",
        "qwen25-05b/ablations/loss/dr-grpo",
    }


def test_discover_cells_finds_legacy_flat_layout(tmp_path):
    """The pre-nesting layout (tag directly under results_dir) still works."""
    _write_eval(
        tmp_path / "qwen25-05b-baseline-few-shot/run_1/eval_final.json",
        rouge_l_mean=0.4,
    )
    cells = _discover_cells(tmp_path)
    assert set(cells) == {"qwen25-05b-baseline-few-shot"}


def test_discover_cells_finds_oldest_layout_with_no_run_dir(tmp_path):
    """Eval files directly inside the cell dir, no run_* subdir at all."""
    _write_eval(tmp_path / "qwen25-05b-optimal/eval_final.json", rouge_l_mean=0.3)
    cells = _discover_cells(tmp_path)
    assert cells == {"qwen25-05b-optimal": []}


def test_discover_cells_picks_no_duplicate_across_mixed_layouts(tmp_path):
    """A results tree mixing nested and legacy cells (the real post-campaign
    state on disk) discovers every distinct cell exactly once."""
    _write_eval(
        tmp_path / "qwen25-05b/sft-grpo/few-shot/run_1/eval_final.json",
        rouge_l_mean=0.9,
    )
    _write_eval(
        tmp_path / "qwen25-05b-baseline-zero-shot/run_1/eval_final.json",
        rouge_l_mean=0.1,
    )
    cells = _discover_cells(tmp_path)
    assert set(cells) == {
        "qwen25-05b/sft-grpo/few-shot",
        "qwen25-05b-baseline-zero-shot",
    }


def test_find_eval_results_picks_latest_run_and_reads_metrics(tmp_path):
    _write_eval(
        tmp_path / "qwen25-05b/grpo/zero-shot/run_1/eval_final.json",
        rouge_l_mean=0.11,
    )
    _write_eval(
        tmp_path / "qwen25-05b/grpo/zero-shot/run_2/eval_final.json",
        rouge_l_mean=0.99,
    )
    entries = find_eval_results(tmp_path)
    assert len(entries) == 1
    entry = entries[0]
    assert entry["config_name"] == "qwen25-05b/grpo/zero-shot"
    assert entry["run_id"] == "run_2"
    assert entry["metrics"]["ROUGE-L"] == 0.99


def test_find_eval_results_skips_baseline_json(tmp_path):
    """eval_baseline.json is a reference, never the cell's own result."""
    _write_eval(
        tmp_path / "qwen25-05b/grpo/zero-shot/run_1/eval_baseline.json",
        rouge_l_mean=0.42,
    )
    assert find_eval_results(tmp_path) == []


def test_find_eval_results_empty_dir_returns_empty_list(tmp_path):
    assert find_eval_results(tmp_path) == []


def test_find_eval_results_missing_dir_returns_empty_list(tmp_path):
    assert find_eval_results(tmp_path / "does-not-exist") == []
