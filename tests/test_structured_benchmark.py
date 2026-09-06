from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from src.analysis.structured_benchmark import (
    ARMS,
    filter_gloss_lengths,
    finish_device_measurement,
    reset_device_measurement,
    run_benchmark,
    shuffled_weight_control,
    train_head_arm,
    transition_control_diagnostics,
    uniform_support_control,
)
from src.datasets.structured_transitions import build_structured_transition_graph
from src.training.structured_sft import (
    FrozenFeatureArtifact,
    feature_manifest,
    load_feature_artifact,
    save_feature_artifact,
)


def artifact() -> FrozenFeatureArtifact:
    train = (
        {"id": "t0", "text": "x", "gloss": "A B"},
        {"id": "t1", "text": "y", "gloss": "B A"},
        {"id": "t2", "text": "z", "gloss": "A B"},
    )
    dev = ({"id": "d0", "text": "q", "gloss": "A B"},)
    train_x = np.asarray([[1, 0, 0], [0, 1, 0], [1, 0, 0]], dtype=np.float32)
    dev_x = np.asarray([[1, 0, 0]], dtype=np.float32)
    manifest = feature_manifest(train, dev, model_name="frozen-test", hidden_size=3)
    return FrozenFeatureArtifact(train_x, dev_x, train, dev, manifest)


def config() -> dict:
    return {
        "seed": 2,
        "device": "cpu",
        "top_k": 2,
        "alpha": 0.1,
        "transition_scale": 0.25,
        "max_gloss_length": 2,
        "head_lr": 0.05,
        "weight_decay": 0.0,
        "steps": 150,
        "max_peak_memory_bytes": 1_000_000,
        "max_arm_runtime_seconds": 60.0,
    }


def test_feature_manifest_save_reload_is_deterministic(tmp_path):
    value = artifact()
    first, second = tmp_path / "one.npz", tmp_path / "two.npz"
    save_feature_artifact(value, first)
    save_feature_artifact(value, second)
    one, two = load_feature_artifact(first), load_feature_artifact(second)
    assert one.manifest == two.manifest == value.manifest
    np.testing.assert_array_equal(one.train_features, value.train_features)
    assert one.train_rows == value.train_rows


def test_all_arms_share_split_features_and_head_only_overfits():
    value = artifact()
    graph = build_structured_transition_graph(value.train_rows, top_k=2)
    reports = {}
    for arm in ARMS:
        head, reports[arm] = train_head_arm(arm, value, graph, config())
        assert all(parameter.requires_grad for parameter in head.parameters())
        assert reports[arm]["feature_sample_hash"] == value.manifest["sample_hash"]
        assert reports[arm]["train_independent_ce"] < 0.8
    assert (
        reports["true-weights"]["active_graph_hash"]
        == reports["true-weights"]["graph_hash"]
    )
    assert (
        reports["shuffled-weights"]["active_graph_hash"]
        != reports["shuffled-weights"]["graph_hash"]
    )
    assert (
        reports["true-weights"]["dev_structured_nll"]
        != reports["shuffled-weights"]["dev_structured_nll"]
    )


def test_shuffled_control_is_deterministic_finite_and_keeps_support():
    graph = build_structured_transition_graph(artifact().train_rows, top_k=2)
    one = shuffled_weight_control(graph, 3)
    two = shuffled_weight_control(graph, 3)
    np.testing.assert_array_equal(one.edge_src, graph.edge_src)
    np.testing.assert_array_equal(one.edge_dst, graph.edge_dst)
    np.testing.assert_array_equal(one.edge_log_prob, two.edge_log_prob)
    assert np.isfinite(one.edge_log_prob).all()
    diagnostics = transition_control_diagnostics(graph, one)
    assert diagnostics["fraction_log_weights_changed"] > 0
    uniform = uniform_support_control(graph)
    assert uniform.manifest()["graph_hash"] != graph.manifest()["graph_hash"]


def test_unchanged_shuffle_is_rejected():
    import pytest

    graph = build_structured_transition_graph([{"gloss": "A"}], top_k=1)
    with pytest.raises(ValueError, match="did not change"):
        shuffled_weight_control(graph, 1)


def test_cuda_measurement_resets_and_synchronizes(monkeypatch):
    calls = []
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: calls.append("empty"))
    monkeypatch.setattr(
        torch.cuda, "reset_peak_memory_stats", lambda device: calls.append("reset")
    )
    monkeypatch.setattr(torch.cuda, "synchronize", lambda device: calls.append("sync"))
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda device: 123)
    device = torch.device("cuda")
    started = reset_device_measurement(device)
    runtime, peak = finish_device_measurement(device, started)
    assert calls == ["empty", "reset", "sync", "sync"]
    assert runtime >= 0 and peak == 123


def test_length_filter_excludes_before_cap_and_manifest_records_distribution():
    rows = [
        {"id": "ok", "gloss": "A B"},
        {"id": "long", "gloss": "A B C"},
    ]
    retained, report = filter_gloss_lengths(rows, 2)
    assert [row["id"] for row in retained] == ["ok"]
    assert report["excluded_sample_ids"] == ["long"]
    assert report["excluded_fraction"] == 0.5
    assert report["max_length_observed"] == 3


def test_empty_and_batch_overflow_hard_fail():
    import pytest

    with pytest.raises(ValueError, match="empty"):
        filter_gloss_lengths([{"id": "bad", "gloss": "  "}], 2)


def test_benchmark_writes_reloadable_heads_without_control_level(tmp_path):
    features = tmp_path / "features.npz"
    save_feature_artifact(artifact(), features)
    reports, comparison = run_benchmark(config(), features, tmp_path / "structured")
    assert set(reports) == set(ARMS)
    assert comparison["promotion_gate"]["auto_promote"] is False
    assert not (tmp_path / "structured" / "control").exists()
    for arm in ARMS:
        runs = list((tmp_path / "structured" / arm).glob("run_*"))
        checkpoint = torch.load(runs[0] / "head.pt", weights_only=True)
        assert "state_dict" in checkpoint
        rebuilt = __import__(
            "src.models.structured_gloss_head", fromlist=["StructuredGlossHead"]
        ).StructuredGlossHead(**checkpoint["head"])
        rebuilt.load_state_dict(checkpoint["state_dict"])
        original, _ = train_head_arm(
            arm,
            artifact(),
            build_structured_transition_graph(artifact().train_rows, top_k=2),
            config(),
        )
        probe = torch.from_numpy(artifact().dev_features)
        torch.testing.assert_close(rebuilt(probe, 2), original.cpu()(probe, 2))
        report = json.loads((runs[0] / "report.json").read_text())
        assert report["feature_sample_hash"] == artifact().manifest["sample_hash"]


def test_module_does_not_import_policy_trainers():
    source = Path("src/analysis/structured_benchmark.py").read_text()
    assert "sft_train" not in source
    assert "grpo" not in source.lower()
