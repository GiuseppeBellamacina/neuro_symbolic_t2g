"""Bigram transition artifact construction and loading."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

import numpy as np

from datasets import DatasetDict

from .aslg_dataset import BOS_GLOSS, EOS_GLOSS, UNK_GLOSS

logger = logging.getLogger(__name__)


def compute_bigram_transitions(
    dataset: DatasetDict,
    vocab: list[str],
    split: str = "train",
    smoothing: float = 1.0,
) -> np.ndarray:
    """Estimate a dense bigram artifact from one dataset split."""
    if smoothing < 0:
        raise ValueError("smoothing must be nonnegative")

    token_to_index = {token: index for index, token in enumerate(vocab)}
    vocab_size = len(vocab)
    counts = np.zeros((vocab_size, vocab_size), dtype=np.float64)
    unknown = token_to_index.get(UNK_GLOSS, 0)

    for raw_sample in dataset[split]:
        sample = cast(Mapping[str, Any], raw_sample)
        tokens = str(sample.get("gloss", "")).split()
        if not tokens:
            continue
        if BOS_GLOSS in token_to_index:
            tokens.insert(0, BOS_GLOSS)
        if EOS_GLOSS in token_to_index:
            tokens.append(EOS_GLOSS)
        for source, target in zip(tokens, tokens[1:]):
            counts[
                token_to_index.get(source, unknown), token_to_index.get(target, unknown)
            ] += 1

    denominator = counts.sum(axis=1, keepdims=True) + smoothing * vocab_size
    if smoothing == 0 and np.any(denominator == 0):
        raise ValueError(
            "unsmoothed transition rows with no observations are undefined"
        )
    return ((counts + smoothing) / denominator).astype(np.float32)


def save_transition_matrix(matrix: np.ndarray, path: str | Path) -> None:
    """Save a dense transition artifact."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.save(destination, matrix)
    logger.info("Transition matrix saved to %s", destination)


def load_transition_matrix(
    path: str | Path,
    expected_size: int | None = None,
) -> np.ndarray:
    """Load a square transition artifact, optionally validating vocab size."""
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"Transition matrix not found: {source}")
    matrix = np.load(source)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(
            f"Transition matrix must be square, got shape {matrix.shape} from {source}"
        )
    if expected_size is not None and matrix.shape != (expected_size, expected_size):
        raise ValueError(
            "Transition matrix/vocabulary size mismatch: "
            f"expected {(expected_size, expected_size)}, got {matrix.shape} from {source}"
        )
    return matrix


def transition_score(matrix: np.ndarray, source: int, target: int) -> float:
    """Return one transition weight."""
    return float(matrix[source, target])


def sequence_score_bigram(matrix: np.ndarray, token_indices: list[int]) -> float:
    """Return cumulative log weight for a fixed sequence in O(L)."""
    return sum(
        float(np.log(max(float(matrix[source, target]), 1e-10)))
        for source, target in zip(token_indices, token_indices[1:])
    )
