from __future__ import annotations

import json

import numpy as np
import pytest

import src.analysis.rollout_probe as rollout_probe
from src.analysis.rollout_probe import (
    load_generation_artifact,
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
    qualification = report["reward_qualification"]
    assert set(qualification["rewards"]) == {
        "edit-validity",
        "token-f1-validity",
        "chrfpp-validity",
        "rouge-l-validity",
        "sbleu2-exp-validity",
    }
    assert qualification["pairwise_spearman"]["edit-validity"]["edit-validity"] == 1.0
    assert qualification["perturbations"]["edit-validity"]["delete"]["mean_drop"] > 0
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


def test_reward_probe_is_deterministic_and_reports_per_reward_status():
    rows = [
        {"group_id": "x", "completion": "A B", "gold_gloss": "A B"},
        {"group_id": "x", "completion": "A", "gold_gloss": "A B"},
    ]
    first = run_rollout_probe(rows, ["A", "B"])
    second = run_rollout_probe(rows, ["A", "B"])
    assert first == second
    assert all(
        isinstance(item["passed"], bool)
        for item in first["reward_qualification"]["rewards"].values()
    )


def test_token_f1_non_palindromic_reversal_is_diagnostic_and_eligible():
    rows = [
        {"group_id": "x", "completion": completion, "gold_gloss": "A B C"}
        for completion in ("A B C", "A B", "B C", "A C", "A", "B", "C", "C B A")
    ]
    report = run_rollout_probe(
        rows,
        ["A", "B", "C", "X"],
        thresholds={
            "max_zero_variance_fraction": 1,
            "max_all_min_fraction": 1,
            "min_unique_scores_mean": 1,
            "min_best_minus_mean": 0,
            "min_rankable_group_fraction": 0,
            "min_bucket_count": 1,
        },
        qualification_identity={
            "group_size": 8,
            "decoding_population": "sampling",
            "temperature": 0.7,
            "required_temperature": 0.7,
            "prompt_mode": "retrieval",
            "retrieval_enabled": True,
            "trie_enabled": True,
            "max_completion_length": 128,
            "required_max_completion_length": 128,
        },
    )

    reversal = report["reward_qualification"]["perturbations"]["token-f1-validity"][
        "reverse"
    ]
    assert reversal["expected_behavior"] == "invariant"
    assert reversal["evaluable_count"] == 1
    assert reversal["mean_drop"] == pytest.approx(0.0)
    assert reversal["required_for_qualification"] is False
    assert (
        report["reward_qualification"]["rewards"]["token-f1-validity"][
            "eligible_to_authorize_training"
        ]
        is True
    )


def test_palindromic_and_structurally_unevaluable_perturbations_are_reported():
    report = rollout_probe._perturbation_report(
        [{"gold_gloss": "A B A"}, {"gold_gloss": "A"}], {"a", "b"}
    )
    assert report is not None
    reversal = report["edit-validity"]["reverse"]
    assert reversal["attempted_count"] == 2
    assert reversal["evaluable_count"] == 0
    assert reversal["no_op_count"] == 1
    assert reversal["structurally_unevaluable_count"] == 1
    assert reversal["passed"] is None


def test_expected_degradation_without_positive_drop_fails(monkeypatch):
    monkeypatch.setattr(rollout_probe, "score_validity_reward", lambda *args: 1.0)
    report = rollout_probe._perturbation_report(
        [{"gold_gloss": "A B"}], {"a", "b", "x"}
    )
    assert report is not None
    deletion = report["edit-validity"]["delete"]
    assert deletion["expected_behavior"] == "degradation"
    assert deletion["evaluable_count"] == 1
    assert deletion["mean_drop"] == 0.0
    assert deletion["positive_drop_rate"] == 0.0
    assert deletion["passed"] is False


def test_reward_loader_filters_mixed_six_row_eval_sampling(tmp_path):
    source = tmp_path / "mixed.json"
    rows = [
        {
            "sample_id": "x",
            "decoding_mode": "deployment",
            "completion": "A",
            "gold_gloss": "A",
        },
        *[
            {
                "sample_id": "x",
                "decoding_mode": "sampling",
                "completion": "A",
                "gold_gloss": "A",
            }
            for _ in range(5)
        ],
    ]
    source.write_text(json.dumps(rows), encoding="utf-8")
    selected, _, provenance = load_generation_artifact(
        source, 5, decoding_mode="sampling"
    )
    assert len(selected) == 5
    assert provenance["artifact_rows"] == 6
    assert provenance["selected_rows"] == 5
    report = run_rollout_probe(
        selected,
        ["A"],
        qualification_identity={
            "group_size": 5,
            "decoding_population": "sampling",
            "temperature": 0.7,
            "required_temperature": 0.7,
            "prompt_mode": "retrieval",
            "retrieval_enabled": True,
            "trie_enabled": True,
            "max_completion_length": 128,
            "required_max_completion_length": 128,
        },
    )
    assert report["reward_qualification"]["eligible_to_authorize_training"] is False


def test_probe_cli_refuses_overwrite_without_force(tmp_path):
    from src.analysis.__main__ import main

    source = tmp_path / "rows.json"
    source.write_text(
        json.dumps(
            [
                {
                    "sample_id": "x",
                    "decoding_mode": "sampling",
                    "completion": "A",
                    "gold_gloss": "A",
                }
                for _ in range(8)
            ]
        ),
        encoding="utf-8",
    )
    from src.analysis import __main__ as cli

    output = tmp_path / "report.json"
    output.write_text("{}", encoding="utf-8")
    config = tmp_path / "rewards.yaml"
    config.write_text(
        f"probe:\n  output_path: {output.as_posix()}\n  group_size: 8\n",
        encoding="utf-8",
    )
    with pytest.raises(FileExistsError, match="--force"):
        original = cli.PROJECT_ROOT
        cli.PROJECT_ROOT = tmp_path
        try:
            main(["rewards", "--config", str(config), "--input", str(source)])
        finally:
            cli.PROJECT_ROOT = original
