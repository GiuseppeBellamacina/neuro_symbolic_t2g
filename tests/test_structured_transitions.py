from __future__ import annotations

import json

import numpy as np
import pytest

from src.datasets.structured_transitions import (
    OTHER,
    assert_gloss_paths_supported,
    build_structured_transition_graph,
    load_structured_transition_graph,
    save_structured_transition_graph,
    shuffled_transition_control,
)

ROWS = [
    {"id": "b", "gloss": "B A RARE"},
    {"id": "a", "gloss": "A B TAIL"},
    {"id": "c", "gloss": "B A"},
]


def test_vocab_other_graph_invariants_and_train_coverage():
    graph = build_structured_transition_graph(ROWS, top_k=2)
    assert graph.states == ("A", "B", OTHER)  # frequency ties break lexically
    assert graph.map_glosses("A RARE UNKNOWN") == [0, 2, 2]
    assert not np.any(graph.edge_dst == graph.bos_index)
    assert not np.any(graph.edge_src == graph.eos_index)
    for row in ROWS:
        path = graph.map_glosses(row["gloss"])
        nodes = [graph.bos_index, *path, graph.eos_index]
        assert all(graph.has_edge(a, b) for a, b in zip(nodes, nodes[1:]))
    for source in np.unique(graph.edge_src):
        probabilities = np.exp(graph.edge_log_prob[graph.edge_src == source])
        assert np.isclose(probabilities.sum(), 1.0)


def test_build_and_serialization_are_deterministic(tmp_path):
    first = build_structured_transition_graph(ROWS, top_k=2)
    second = build_structured_transition_graph(ROWS, top_k=2)
    assert first.manifest() == second.manifest()
    paths = []
    for suffix, graph in (("one", first), ("two", second)):
        npz = tmp_path / f"{suffix}.npz"
        manifest = tmp_path / f"{suffix}.json"
        save_structured_transition_graph(graph, npz, manifest)
        paths.append((npz, manifest))
    assert paths[0][0].read_bytes() == paths[1][0].read_bytes()
    assert paths[0][1].read_bytes() == paths[1][1].read_bytes()
    loaded = load_structured_transition_graph(*paths[0])
    assert loaded.manifest() == first.manifest()
    assert json.loads(paths[0][1].read_text())["train_sample_ids"] == ["b", "a", "c"]
    saved_manifest = first.manifest()
    assert saved_manifest["protocol_version"] == "structured-transition-graph-v2"
    assert saved_manifest["top_k"] == 2
    assert saved_manifest["other_token"] == OTHER
    assert saved_manifest["state_digest"] and saved_manifest["edge_digest"]


def test_digest_sensitive_to_edges_counts_and_settings():
    baseline = build_structured_transition_graph(ROWS, top_k=2, alpha=0.1)
    more_count = build_structured_transition_graph([*ROWS, ROWS[0]], top_k=2, alpha=0.1)
    other_alpha = build_structured_transition_graph(ROWS, top_k=2, alpha=0.2)
    other_k = build_structured_transition_graph(ROWS, top_k=1, alpha=0.1)
    digests = {
        graph.manifest()["graph_hash"]
        for graph in (baseline, more_count, other_alpha, other_k)
    }
    assert len(digests) == 4


def test_manifest_mismatch_is_rejected(tmp_path):
    graph = build_structured_transition_graph(ROWS, top_k=2)
    npz, manifest = tmp_path / "graph.npz", tmp_path / "graph.json"
    save_structured_transition_graph(graph, npz, manifest)
    payload = json.loads(manifest.read_text())
    payload["alpha"] = 7.0
    manifest.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="mismatch"):
        load_structured_transition_graph(npz, manifest)


def test_other_repetition_and_no_holdout_leakage():
    train = [{"id": "train", "gloss": "A RARE RARER"}]
    holdout = [{"id": "eval", "gloss": "EVAL_ONLY A"}]
    graph = build_structured_transition_graph(train, top_k=1)
    assert "EVAL_ONLY" not in graph.states
    assert "eval" not in graph.train_sample_ids
    assert graph.map_glosses(holdout[0]["gloss"])[0] == graph.token_to_index[OTHER]
    mapped = graph.map_glosses(train[0]["gloss"])
    assert mapped == [0, 1, 1]
    assert graph.has_edge(1, 1)
    assert_gloss_paths_supported([row["gloss"] for row in train], graph)
    with pytest.raises(ValueError, match="unsupported"):
        assert_gloss_paths_supported([holdout[0]["gloss"]], graph)


def test_shuffled_control_is_deterministic_and_preserves_edge_payloads():
    graph = build_structured_transition_graph(ROWS, top_k=2)
    one = shuffled_transition_control(graph, seed=7)
    two = shuffled_transition_control(graph, seed=7)
    np.testing.assert_array_equal(one.edge_dst, two.edge_dst)
    assert sorted(one.edge_count.tolist()) == sorted(graph.edge_count.tolist())
    assert sorted(one.edge_log_prob.tolist()) == sorted(graph.edge_log_prob.tolist())
    assert not np.array_equal(one.edge_dst, graph.edge_dst)


def test_state_size_guard():
    rows = [{"gloss": " ".join(f"G{i}" for i in range(600))}]
    graph = build_structured_transition_graph(rows)
    assert graph.num_states == 513
