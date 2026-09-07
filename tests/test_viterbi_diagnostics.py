from __future__ import annotations

import itertools

import numpy as np
import pytest

from src.analysis.markov_diagnostics import (
    bigram_sequence_mean,
    hard_viterbi_diagnostic,
    path_log_energy,
    soft_viterbi_diagnostic,
)


def _paths(size: int, start: int, end: int, length: int, forbidden: set[int]):
    allowed = [state for state in range(size) if state not in forbidden]
    return (
        [start, *interior, end]
        for interior in itertools.product(allowed, repeat=length - 2)
    )


def _soft(values: list[float], tau: float) -> float:
    maximum = max(values)
    return maximum + tau * np.log(
        sum(np.exp((value - maximum) / tau) for value in values)
    )


@pytest.mark.parametrize("size", [3, 4, 5])
@pytest.mark.parametrize("length", [2, 3, 4, 5])
@pytest.mark.parametrize("penalty", [0.0, 0.7])
def test_hard_and_soft_match_brute_force(size, length, penalty):
    rng = np.random.default_rng(size * 100 + length)
    matrix = rng.random((size, size))
    start, end = 0, size - 1
    paths = list(_paths(size, start, end, length, {start, end}))
    energies = [path_log_energy(matrix, path, penalty) for path in paths]

    path, hard = hard_viterbi_diagnostic(matrix, start, end, length, penalty)
    assert path in paths
    assert hard == pytest.approx(max(energies), abs=1e-10)
    assert path_log_energy(matrix, path, penalty) == pytest.approx(hard, abs=1e-10)

    for tau in (0.1, 0.7, 2.0):
        soft = soft_viterbi_diagnostic(matrix, start, end, length, tau, penalty)
        assert soft == pytest.approx(_soft(energies, tau), abs=1e-10)
        assert soft >= hard - 1e-10


def test_bigram_mean_matches_direct_linear_path_score():
    matrix = np.array([[0.2, 0.8, 0.0], [0.1, 0.3, 0.6], [0.5, 0.5, 0.0]])
    path = [0, 1, 2]
    expected = (np.log(0.8) + np.log(0.6)) / 2
    assert bigram_sequence_mean(matrix, path) == pytest.approx(expected)


def test_endpoints_and_exclusions_are_enforced():
    matrix = np.full((5, 5), 0.2)
    path, _ = hard_viterbi_diagnostic(matrix, 0, 4, 4, excluded_interior=2)
    assert path[0] == 0 and path[-1] == 4
    assert not {0, 2, 4} & set(path[1:-1])


@pytest.mark.parametrize(
    "call",
    [
        lambda: hard_viterbi_diagnostic(np.ones((2, 3)), 0, 1, 3),
        lambda: hard_viterbi_diagnostic(np.ones((2, 2)), 0, 1, 3),
        lambda: soft_viterbi_diagnostic(np.ones((3, 3)), 0, 2, 3, tau=0),
        lambda: path_log_energy(np.ones((3, 3)), [0]),
        lambda: bigram_sequence_mean(np.ones((3, 3)), [0, -1]),
    ],
)
def test_invalid_diagnostic_inputs_fail(call):
    with pytest.raises(ValueError):
        call()


def test_viterbi_dense_state_guard_is_hard_failure():
    matrix = np.ones((5, 5))
    with pytest.raises(ValueError, match="Dense full-vocabulary Viterbi is forbidden"):
        hard_viterbi_diagnostic(matrix, 0, 4, 3, max_states=4)
    with pytest.raises(ValueError, match="Dense full-vocabulary Viterbi is forbidden"):
        soft_viterbi_diagnostic(matrix, 0, 4, 3, max_states=4)
