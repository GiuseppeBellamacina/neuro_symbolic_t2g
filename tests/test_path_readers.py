from __future__ import annotations

import json
from pathlib import Path

from src.utils.ablation_summary import find_eval_results
from src.utils.show_training_log import _find_trainer_state


def test_ablation_summary_discovers_only_canonical_runs(tmp_path: Path) -> None:
    canonical = (
        tmp_path / "qwen25-05b/sft-grpo/few-shot/eval-zero-shot/run_20260905_010203"
    )
    noncanonical = tmp_path / "qwen25-05b/sft-grpo/few-shot/run_20260904_010203"
    canonical.mkdir(parents=True)
    noncanonical.mkdir(parents=True)
    (canonical / "eval_final.json").write_text(
        json.dumps({"rouge_l_mean": 0.8}), encoding="utf-8"
    )
    (canonical / "COMPLETED").touch()
    (noncanonical / "eval_final.json").write_text(
        json.dumps({"rouge_l_mean": 0.9}), encoding="utf-8"
    )
    entries = find_eval_results(tmp_path)
    assert {entry["config_name"] for entry in entries} == {
        "sft-grpo/few-shot/eval-zero-shot",
    }


def test_ablation_summary_discovers_all_canonical_result_trees(tmp_path: Path) -> None:
    paths = {
        "base/zero-shot/eval-zero-shot": tmp_path
        / "qwen25-05b/baseline/zero-shot/run_20260905_010203",
        "base/few-shot/eval-few-shot": tmp_path
        / "qwen25-05b/baseline/few-shot/run_20260905_010204",
        "sft/zero-shot/eval-few-shot": tmp_path
        / "qwen25-05b/sft/zero-shot/eval-few-shot/run_20260905_010205",
        "sft-grpo/zero-shot/pda/eval-zero-shot": tmp_path
        / "qwen25-05b/sft-grpo/zero-shot/ablations/pda/eval-zero-shot/run_20260905_010206",
        "sft-grpo/zero-shot/hot/eval-few-shot": tmp_path
        / "qwen25-05b/sft-grpo/zero-shot/ablations/hot/eval-few-shot/run_20260905_010207",
    }
    for run in paths.values():
        run.mkdir(parents=True)
        (run / "COMPLETED").touch()
        (run / "eval_final.json").write_text(
            json.dumps({"deployment": {"rouge_l_mean": 0.8}}), encoding="utf-8"
        )
    malformed = tmp_path / "qwen25-05b/sft/zero-shot/not-an-eval/run_20260905_010208"
    malformed.mkdir(parents=True)
    (malformed / "COMPLETED").touch()
    (malformed / "eval_final.json").write_text("{}", encoding="utf-8")
    entries = find_eval_results(tmp_path)
    assert {entry["config_name"] for entry in entries} == set(paths)


def test_ablation_summary_falls_back_to_older_completed_valid_run(
    tmp_path: Path,
) -> None:
    parent = tmp_path / "qwen25-05b/grpo/few-shot/eval-zero-shot"
    older = parent / "run_20260905_010203"
    incomplete = parent / "run_20260905_010204"
    malformed = parent / "run_20260905_010205"
    for run in (older, incomplete, malformed):
        run.mkdir(parents=True)
    (older / "COMPLETED").touch()
    (older / "eval_final.json").write_text(
        json.dumps({"rouge_l_mean": 0.7}), encoding="utf-8"
    )
    (incomplete / "eval_final.json").write_text(
        json.dumps({"rouge_l_mean": 0.9}), encoding="utf-8"
    )
    (malformed / "COMPLETED").touch()
    (malformed / "eval_final.json").write_text("not json", encoding="utf-8")
    entries = find_eval_results(tmp_path)
    assert len(entries) == 1
    assert entries[0]["run_id"] == older.name


def test_show_log_finds_state_at_arbitrary_nested_depth(tmp_path: Path) -> None:
    checkpoint = (
        tmp_path / "experiments/checkpoints/qwen25-05b/sft-grpo/few-shot/"
        "run_20260905_010203/checkpoint-10"
    )
    checkpoint.mkdir(parents=True)
    state = checkpoint / "trainer_state.json"
    state.write_text("{}", encoding="utf-8")
    assert _find_trainer_state(str(checkpoint)) == state
