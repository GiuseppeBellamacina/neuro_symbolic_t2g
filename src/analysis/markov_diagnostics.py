"""Small-graph Markov diagnostics for offline analysis."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

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


def load_markov_artifacts(
    vocab_path: str | Path, matrix_path: str | Path
) -> tuple[list[str], np.ndarray]:
    """Load local text vocabulary and NPY transition matrix, never downloading."""
    vocab_source, matrix_source = Path(vocab_path), Path(matrix_path)
    for source in (vocab_source, matrix_source):
        if not source.is_file():
            raise FileNotFoundError(f"Markov artifact not found: {source}")
    vocab = [
        line.strip()
        for line in vocab_source.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not vocab:
        raise ValueError("vocabulary artifact is empty")
    try:
        matrix = np.load(matrix_source, allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise ValueError(f"invalid matrix artifact: {matrix_source}") from exc
    validated = _matrix(matrix)
    if len(validated) != len(vocab):
        raise ValueError(
            f"vocabulary/matrix size mismatch: {len(vocab)} != {len(validated)}"
        )
    return vocab, validated


def run_markov_probe(
    rows: Sequence[Mapping[str, Any]],
    vocab: Sequence[str],
    matrix: np.ndarray,
    *,
    thresholds: Mapping[str, float] | None = None,
    max_states: int = DEFAULT_MAX_STATES,
) -> dict[str, Any]:
    """Measure whether O(L) bigram scores rank frozen outputs by edit quality."""
    from src.analysis.rollout_probe import edit_similarity, normalized_tokens

    validated = _matrix(matrix)
    if (
        isinstance(max_states, bool)
        or not isinstance(max_states, (int, np.integer))
        or max_states < 1
    ):
        raise ValueError("max_states must be a positive integer")
    if len(validated) != len(vocab):
        raise ValueError("vocabulary and matrix sizes differ")
    token_to_index = {token.casefold(): index for index, token in enumerate(vocab)}
    bos, eos = token_to_index.get("<bos>"), token_to_index.get("<eos>")
    scored: list[dict[str, Any]] = []
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for position, row in enumerate(rows):
        for key in ("group_id", "completion", "gold_gloss"):
            if key not in row:
                raise ValueError(f"generation row {position} missing {key!r}")
        tokens = normalized_tokens(str(row["completion"]))
        oov = [token for token in tokens if token not in token_to_index]
        path = (
            ([bos] if bos is not None else [])
            + [token_to_index[token] for token in tokens if token in token_to_index]
            + ([eos] if eos is not None else [])
        )
        score = (
            bigram_sequence_mean(validated, path)
            if len(path) >= 2
            else float(np.log(EPSILON))
        )
        item = {
            "group_id": str(row["group_id"]),
            "bigram_score": score,
            "edit_similarity": edit_similarity(
                str(row["completion"]), str(row["gold_gloss"])
            ),
            "length": len(tokens),
            "gold_length": len(normalized_tokens(str(row["gold_gloss"]))),
            "oov_count": len(oov),
            "adversarial": bool(row.get("adversarial", False)),
        }
        scored.append(item)
        groups[item["group_id"]].append(item)

    edits = [item["edit_similarity"] for item in scored]
    bigrams = [item["bigram_score"] for item in scored]
    lengths = [float(item["length"]) for item in scored]
    pair_correct = pair_total = 0
    group_correlations = []
    for group_id in sorted(groups):
        items = groups[group_id]
        group_correlations.append(
            _rank_correlation(
                [item["edit_similarity"] for item in items],
                [item["bigram_score"] for item in items],
            )
        )
        for left in range(len(items)):
            for right in range(left + 1, len(items)):
                edit_delta = (
                    items[left]["edit_similarity"] - items[right]["edit_similarity"]
                )
                if edit_delta == 0:
                    continue
                pair_total += 1
                bigram_delta = (
                    items[left]["bigram_score"] - items[right]["bigram_score"]
                )
                pair_correct += (edit_delta > 0) == (bigram_delta > 0)

    residual_edit = _residuals(edits, lengths)
    residual_bigram = _residuals(bigrams, lengths)
    stratified: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in scored:
        length = item["gold_length"]
        stratified["1-3" if length <= 3 else "4-7" if length <= 7 else "8+"].append(
            item
        )
    strata = {
        name: {
            "count": len(items),
            "edit_bigram_spearman": _rank_correlation(
                [item["edit_similarity"] for item in items],
                [item["bigram_score"] for item in items],
            ),
        }
        for name, items in sorted(stratified.items())
    }
    adversarial = [item for item in scored if item["adversarial"]]
    report: dict[str, Any] = {
        "schema_version": 1,
        "probe": "markov_qualification",
        "max_states_for_viterbi": int(max_states),
        "num_outputs": len(scored),
        "num_groups": len(groups),
        "edit_bigram_spearman": _rank_correlation(edits, bigrams),
        "edit_length_spearman": _rank_correlation(edits, lengths),
        "partial_edit_bigram_controlling_length": _pearson(
            residual_edit, residual_bigram
        ),
        "within_group_spearman_mean": float(np.mean(group_correlations)),
        "within_group_pairwise_accuracy": (
            pair_correct / pair_total if pair_total else 0.0
        ),
        "within_group_comparable_pairs": pair_total,
        "gold_length_strata": strata,
        "oov_token_count": sum(item["oov_count"] for item in scored),
        "adversarial": {
            "count": len(adversarial),
            "edit_bigram_spearman": _rank_correlation(
                [item["edit_similarity"] for item in adversarial],
                [item["bigram_score"] for item in adversarial],
            ),
        },
    }
    limits = dict(thresholds or {})
    checks = {
        key: report[key] >= value
        for key, value in sorted(limits.items())
        if key in report and isinstance(report[key], (int, float))
    }
    report["thresholds"] = limits
    report["threshold_checks"] = checks
    report["qualified"] = all(checks.values()) if checks else None
    return report


def _rank_correlation(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) < 2:
        return 0.0
    return _pearson(_rank(left), _rank(right))


def _rank(values: Sequence[float]) -> np.ndarray:
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values), dtype=float)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2
        start = end
    return ranks


def _pearson(
    left: Sequence[float] | np.ndarray, right: Sequence[float] | np.ndarray
) -> float:
    if len(left) < 2 or np.std(left) == 0 or np.std(right) == 0:
        return 0.0
    return float(np.corrcoef(left, right)[0, 1])


def _residuals(values: Sequence[float], lengths: Sequence[float]) -> np.ndarray:
    dependent, control = np.asarray(values, dtype=float), np.asarray(
        lengths, dtype=float
    )
    if len(dependent) < 2 or np.std(control) == 0:
        return dependent - np.mean(dependent)
    design = np.column_stack((np.ones(len(control)), control))
    return dependent - design @ np.linalg.lstsq(design, dependent, rcond=None)[0]


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
