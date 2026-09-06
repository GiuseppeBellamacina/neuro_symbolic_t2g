from __future__ import annotations

import json

import numpy as np
import pytest

from src.analysis.rollout_probe import (
    load_grouped_generations,
    qualify_grouped_outputs,
    run_rollout_probe,
    write_json_report,
)


def test_probe_scores_frozen_outputs_and_within_group_ranks():
    vocab = ["<BOS>", "<EOS>", "A", "B", "C"]
    matrix = np.full((5, 5), 0.01)
    matrix[0, 2] = matrix[2, 3] = matrix[3, 1] = 0.9
    matrix[2, 1] = matrix[3, 2] = matrix[2, 4] = matrix[4, 1] = 0.2
    outputs, groups = qualify_grouped_outputs(
        {"item": ["A B", "A", "B A"]},
        {"item": "A B"},
        vocab,
        matrix,
    )

    assert [item.length for item in outputs] == [2, 1, 2]
    assert [item.edit_score for item in outputs] == [1.0, 0.0, -1.0]
    assert outputs[0].bigram_score > outputs[1].bigram_score
    assert outputs[0].bigram_score > outputs[2].bigram_score
    assert groups[0].count == 3
    assert groups[0].edit_bigram_rank_correlation > 0
    assert groups[0].edit_length_rank_correlation == 0.0


def test_probe_returns_zero_correlation_for_ties():
    vocab = ["<BOS>", "<EOS>", "A"]
    matrix = np.full((3, 3), 1 / 3)
    _, groups = qualify_grouped_outputs(
        {"item": ["A", "A"]}, {"item": "A"}, vocab, matrix
    )
    assert groups[0].edit_bigram_rank_correlation == 0.0
    assert groups[0].edit_length_rank_correlation == 0.0


def test_frozen_eval_generations_report_and_stable_json(tmp_path):
    source = tmp_path / "generations.json"
    source.write_text(
        json.dumps(
            [
                {"sample_id": "b", "completion": "A B", "gold_gloss": "A B"},
                {"sample_id": "b", "completion": "A", "gold_gloss": "A B"},
                {"sample_id": "a", "completion": "X", "gold_gloss": "C"},
                {"sample_id": "a", "completion": "C", "gold_gloss": "C"},
            ]
        ),
        encoding="utf-8",
    )
    rows = load_grouped_generations(source, group_size=2)
    report = run_rollout_probe(rows, ["A", "B", "C"])

    assert report["num_groups"] == 2
    assert report["exact_match_rate"] == 0.5
    assert report["validity_rate"] == 0.75
    assert report["oracle_at_g_exact_match"] == 1.0
    assert [group["group_id"] for group in report["groups"]] == ["a", "b"]

    first, second = tmp_path / "new" / "first.json", tmp_path / "new" / "second.json"
    write_json_report(report, first)
    write_json_report(report, second)
    assert first.read_bytes() == second.read_bytes()


def test_positional_grouping_and_malformed_inputs_fail(tmp_path):
    source = tmp_path / "generations.json"
    source.write_text(
        json.dumps(
            [
                {"completion": "A", "gold_gloss": "A"},
                {"completion": "B", "gold_gloss": "A"},
            ]
        ),
        encoding="utf-8",
    )
    assert [row["group_id"] for row in load_grouped_generations(source, 2)] == [
        "0",
        "0",
    ]

    source.write_text("not json", encoding="utf-8")
    with pytest.raises(ValueError, match="malformed"):
        load_grouped_generations(source, 2)
    with pytest.raises(FileNotFoundError):
        load_grouped_generations(tmp_path / "missing.json", 2)


def test_cli_requires_explicit_generation_input(tmp_path):
    from src.analysis.__main__ import main

    config = tmp_path / "rollouts.yaml"
    config.write_text(
        "probe:\n  input_path: null\n  output_path: report.json\n  group_size: 2\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match=r"--input.*generations_\*\.json"):
        main(["rollouts", "--config", str(config)])
