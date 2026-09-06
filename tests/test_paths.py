from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytest

from src.utils.paths import (
    Cell,
    RunPath,
    cell_base_dir,
    cell_from_config,
    cell_run_from_checkpoint,
    evaluation_identifier,
    evaluation_log_dir,
    iter_runs,
    new_run_id,
    newest_checkpoint,
    training_run_paths,
    wandb_name,
    wandb_tags,
)


def _config(method: str = "sft-grpo", prompt: str = "few-shot") -> dict:
    return {
        "experiment": {
            "model_tag": "qwen25-05b",
            "method": method,
            "train_prompt_mode": prompt,
            "variant": "none",
            "kind": "train",
        },
        "model": {"name": "Qwen/Qwen2.5-0.5B-Instruct"},
    }


def test_identity_axes_are_separate() -> None:
    assert cell_from_config(_config()) == Cell(
        "qwen25-05b", "sft-grpo", "few-shot", "none", "train"
    )
    with pytest.raises(ValueError, match="experiment identity"):
        cell_from_config({})
    with pytest.raises(ValueError):
        Cell("qwen25-05b", "grpo", "none", "none", "train")
    with pytest.raises(ValueError):
        Cell("qwen25-05b", "grpo", "zero-shot", "pda", "train")


def test_exact_primary_ablation_and_result_hierarchies(tmp_path: Path) -> None:
    primary = cell_from_config(_config())
    sft = cell_from_config(_config("sft", "zero-shot"))
    ablation = Cell("qwen25-05b", "sft-grpo", "zero-shot", "pda", "ablation")
    baseline = Cell("qwen25-05b", "base", "few-shot", "none", "baseline")
    assert cell_base_dir(tmp_path, "checkpoints", primary) == (
        tmp_path / "checkpoints/qwen25-05b/sft-grpo/few-shot"
    )
    assert (
        cell_base_dir(tmp_path, "logs", sft)
        == tmp_path / "logs/qwen25-05b/sft/zero-shot"
    )
    assert cell_base_dir(tmp_path, "checkpoints", ablation) == (
        tmp_path / "checkpoints/qwen25-05b/sft-grpo/zero-shot/ablations/pda"
    )
    assert cell_base_dir(tmp_path, "results", primary, "zero-shot") == (
        tmp_path / "results/qwen25-05b/sft-grpo/few-shot/eval-zero-shot"
    )
    assert cell_base_dir(tmp_path, "figures", primary, "few-shot") == (
        tmp_path / "figures/qwen25-05b/sft-grpo/few-shot/eval-few-shot"
    )
    assert cell_base_dir(tmp_path, "results", baseline, "few-shot") == (
        tmp_path / "results/qwen25-05b/baseline/few-shot"
    )


def test_training_resume_checkpoint_parse_and_newest(tmp_path: Path) -> None:
    output, logs, run_id, cell = training_run_paths(
        _config(), root=tmp_path, now=datetime(2026, 9, 6, 1, 2, 3)
    )
    assert (
        output
        == tmp_path / "checkpoints/qwen25-05b/sft-grpo/few-shot/run_20260906_010203"
    )
    assert logs == tmp_path / "logs/qwen25-05b/sft-grpo/few-shot/run_20260906_010203"
    (output / "checkpoint-10").mkdir(parents=True)
    assert cell_run_from_checkpoint(output / "checkpoint-10") == RunPath(cell, run_id)
    resumed, _, resumed_id, _ = training_run_paths(
        _config(), root=tmp_path, resume=True
    )
    assert (resumed, resumed_id) == (output, run_id)
    assert list(iter_runs(tmp_path, "checkpoints", cell)) == [output]
    assert newest_checkpoint(tmp_path, cell) == output / "checkpoint-10"


def test_ablation_checkpoint_parse() -> None:
    path = Path(
        "experiments/checkpoints/qwen25-05b/sft-grpo/zero-shot/ablations/hot/"
        "run_20260906_010203/final"
    )
    assert cell_run_from_checkpoint(path).cell == Cell(
        "qwen25-05b", "sft-grpo", "zero-shot", "hot", "ablation"
    )


def test_wandb_identity_has_no_redundant_variant() -> None:
    run_id = new_run_id(datetime(2026, 9, 6, 12, 34, 56))
    run = RunPath(cell_from_config(_config()), run_id, "retrieval")
    assert (
        wandb_name(run) == "qwen25-05b/sft-grpo/few-shot/few-shot/run_20260906_123456"
    )
    assert wandb_tags(run) == ("qwen25-05b", "sft-grpo", "few-shot", "few-shot")


def test_eval_logs_and_labels_are_prompt_specific(tmp_path: Path) -> None:
    run_id = "run_20260906_123456"
    cell = cell_from_config(_config())
    zero = RunPath(cell, run_id, "zero-shot")
    few = RunPath(cell, run_id, "retrieval")

    assert evaluation_log_dir(tmp_path, zero) == (
        tmp_path
        / "logs/qwen25-05b/sft-grpo/few-shot/run_20260906_123456/eval-zero-shot"
    )
    assert evaluation_log_dir(tmp_path, few) == (
        tmp_path / "logs/qwen25-05b/sft-grpo/few-shot/run_20260906_123456/eval-few-shot"
    )
    assert evaluation_log_dir(tmp_path, zero) != evaluation_log_dir(tmp_path, few)
    assert evaluation_identifier(zero) == "sft-grpo/few-shot/eval-zero-shot"
    assert evaluation_identifier(few) == "sft-grpo/few-shot/eval-few-shot"


def test_ablation_eval_identifier_and_log_path(tmp_path: Path) -> None:
    run = RunPath(
        Cell("qwen25-05b", "sft-grpo", "zero-shot", "hot", "ablation"),
        "run_20260906_123456",
        "retrieval",
    )
    assert evaluation_identifier(run) == (
        "sft-grpo/zero-shot/ablations/hot/eval-few-shot"
    )
    assert evaluation_log_dir(tmp_path, run) == (
        tmp_path
        / "logs/qwen25-05b/sft-grpo/zero-shot/ablations/hot/run_20260906_123456/eval-few-shot"
    )
