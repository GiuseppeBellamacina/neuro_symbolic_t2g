"""Markov / Viterbi diagnostic APIs (recovered from the edit-rewards branch).

These are DIAGNOSTIC-ONLY scoring utilities over a bigram transition matrix.
They are deliberately NOT rewards and NOT training objectives: a prior audit
established that the historical hard/soft Viterbi *rewards* collapse
algebraically for equal-length completions and were empirically null. What is
scientifically valuable is the corrected dynamic program, kept here behind a
small-state guard so it can never be run over the full ~16K gloss vocabulary.

Provenance: extracted verbatim from ``src/analysis/markov_diagnostics.py`` on
branch ``edit-rewards`` (commit 68bc12a), where it was validated against
brute-force path enumeration. Only pure functions are recovered; the probe
runner and its file/CLI plumbing are intentionally left out.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np

EPSILON = 1e-10
DEFAULT_MAX_STATES = 256


def bigram_sequence_mean(matrix: np.ndarray, path: Sequence[int]) -> float:
    """Return mean log transition weight for a fixed path in O(L)."""
    validated = _matrix(matrix)
    indices = [_index(item, len(validated), "path") for item in path]
    if len(indices) < 2:
        raise ValueError("path must contain at least two states")
    values = [
        np.log(max(float(validated[source, target]), EPSILON))
        for source, target in zip(indices, indices[1:])
    ]
    return float(np.mean(values))


def path_log_energy(
    matrix: np.ndarray,
    path: Sequence[int],
    self_loop_penalty: float = 0.0,
) -> float:
    """Return fixed path energy with penalties on interior self-loops."""
    validated, penalty = _common(matrix, self_loop_penalty)
    indices = [_index(item, len(validated), "path") for item in path]
    if len(indices) < 2:
        raise ValueError("path must contain at least two states")
    energy = 0.0
    for position, (source, target) in enumerate(zip(indices, indices[1:]), start=1):
        energy += float(np.log(max(float(validated[source, target]), EPSILON)))
        if position < len(indices) - 1 and source == target:
            energy -= penalty
    return energy


def hard_viterbi_diagnostic(
    matrix: np.ndarray,
    start_idx: int,
    end_idx: int,
    length: int,
    self_loop_penalty: float = 0.0,
    excluded_interior: int | Iterable[int] | None = None,
    max_states: int = DEFAULT_MAX_STATES,
) -> tuple[list[int], float]:
    """Return the maximum-energy valid path under the fixed diagnostic energy."""
    validated, start, end, steps, penalty, forbidden = _diagnostic_inputs(
        matrix, start_idx, end_idx, length, self_loop_penalty, excluded_interior
    )
    _state_guard(len(validated), max_states)
    size = len(validated)
    log_matrix = np.log(np.maximum(validated, EPSILON))
    scores = np.full(size, -np.inf)
    scores[start] = 0.0
    backtracks: list[np.ndarray] = []

    for _ in range(1, steps - 1):
        candidates = scores[:, None] + log_matrix
        candidates[np.arange(size), np.arange(size)] -= penalty
        candidates[:, list(forbidden)] = -np.inf
        previous = candidates.argmax(axis=0)
        scores = candidates[previous, np.arange(size)]
        backtracks.append(previous)

    previous = int(np.argmax(scores + log_matrix[:, end]))
    score = float(scores[previous] + log_matrix[previous, end])
    path = [previous, end]
    for backtrack in reversed(backtracks):
        path.insert(0, int(backtrack[path[0]]))
    if steps == 2:
        path = [start, end]
    return path, score


def soft_viterbi_diagnostic(
    matrix: np.ndarray,
    start_idx: int,
    end_idx: int,
    length: int,
    tau: float = 1.0,
    self_loop_penalty: float = 0.0,
    excluded_interior: int | Iterable[int] | None = None,
    max_states: int = DEFAULT_MAX_STATES,
) -> float:
    """Return temperature log-sum-exp over the same valid path energies."""
    validated, start, end, steps, penalty, forbidden = _diagnostic_inputs(
        matrix, start_idx, end_idx, length, self_loop_penalty, excluded_interior
    )
    _state_guard(len(validated), max_states)
    temperature = float(tau)
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("tau must be finite and positive")
    size = len(validated)
    scaled = np.log(np.maximum(validated, EPSILON)) / temperature
    alpha = np.full(size, -np.inf)
    alpha[start] = 0.0

    for _ in range(1, steps - 1):
        candidates = alpha[:, None] + scaled
        candidates[np.arange(size), np.arange(size)] -= penalty / temperature
        alpha = _logsumexp(candidates, axis=0)
        alpha[list(forbidden)] = -np.inf
    return float(temperature * _logsumexp(alpha + scaled[:, end]))


def _matrix(matrix: np.ndarray) -> np.ndarray:
    value = np.asarray(matrix)
    if value.ndim != 2 or value.shape[0] == 0 or value.shape[0] != value.shape[1]:
        raise ValueError("matrix must be nonempty and square")
    if not np.issubdtype(value.dtype, np.number) or np.issubdtype(
        value.dtype, np.complexfloating
    ):
        raise ValueError("matrix must be real numeric")
    if not np.all(np.isfinite(value)) or np.any(value < 0):
        raise ValueError("matrix must contain finite nonnegative weights")
    return value


def _index(value: object, size: int, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} indices must be integers")
    result = int(value)
    if not 0 <= result < size:
        raise ValueError(f"{name} index out of range: {result}")
    return result


def _common(matrix: np.ndarray, penalty: float) -> tuple[np.ndarray, float]:
    validated = _matrix(matrix)
    value = float(penalty)
    if not np.isfinite(value) or value < 0:
        raise ValueError("self_loop_penalty must be finite and nonnegative")
    return validated, value


def _state_guard(size: int, max_states: int) -> None:
    if (
        isinstance(max_states, bool)
        or not isinstance(max_states, (int, np.integer))
        or max_states < 1
    ):
        raise ValueError("max_states must be a positive integer")
    if size > max_states:
        raise ValueError(
            f"Viterbi diagnostic has {size} states; max_states={max_states}. "
            "Dense full-vocabulary Viterbi is forbidden."
        )


def _diagnostic_inputs(matrix, start, end, length, penalty, excluded):
    validated, penalty_value = _common(matrix, penalty)
    size = len(validated)
    start_value = _index(start, size, "start")
    end_value = _index(end, size, "end")
    if (
        isinstance(length, bool)
        or not isinstance(length, (int, np.integer))
        or length < 2
    ):
        raise ValueError("length must be an integer of at least two")
    items = (
        []
        if excluded is None
        else ([excluded] if isinstance(excluded, (int, np.integer)) else list(excluded))
    )
    forbidden = {
        start_value,
        end_value,
        *(_index(item, size, "excluded") for item in items),
    }
    if length > 2 and len(forbidden) == size:
        raise ValueError("no state is available for interior positions")
    return validated, start_value, end_value, int(length), penalty_value, forbidden


def _logsumexp(values: np.ndarray, axis: int | None = None) -> np.ndarray:
    maximum = np.max(values, axis=axis, keepdims=True)
    result = maximum + np.log(
        np.sum(np.exp(values - maximum), axis=axis, keepdims=True)
    )
    return np.squeeze(result, axis=axis) if axis is not None else result.squeeze()
