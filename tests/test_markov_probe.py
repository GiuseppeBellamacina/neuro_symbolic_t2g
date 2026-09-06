from __future__ import annotations

import numpy as np

from src.analysis.markov_diagnostics import run_markov_probe


def test_markov_probe_scores_groups_lengths_and_adversarial_rows():
    vocab = ["<BOS>", "<EOS>", "A", "B", "C"]
    matrix = np.full((5, 5), 0.01)
    matrix[0, 2] = matrix[2, 3] = matrix[3, 1] = 0.9
    matrix[2, 1] = matrix[0, 4] = matrix[4, 1] = 0.2
    rows = [
        {"group_id": "0", "completion": "A B", "gold_gloss": "A B"},
        {"group_id": "0", "completion": "A", "gold_gloss": "A B"},
        {"group_id": "1", "completion": "C", "gold_gloss": "C", "adversarial": True},
        {"group_id": "1", "completion": "Z", "gold_gloss": "C", "adversarial": True},
    ]

    report = run_markov_probe(
        rows,
        vocab,
        matrix,
        thresholds={"within_group_pairwise_accuracy": 0.5},
        max_states=3,
    )

    assert report["num_groups"] == 2
    assert report["within_group_comparable_pairs"] == 2
    assert report["within_group_pairwise_accuracy"] >= 0.5
    assert report["adversarial"]["count"] == 2
    assert report["oov_token_count"] == 1
    assert report["qualified"] is True
    assert report["max_states_for_viterbi"] == 3
