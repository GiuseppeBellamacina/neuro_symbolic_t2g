#!/usr/bin/env python3
"""Test data ingestion and transition matrix computation.

Validates:
  1. ASLG-PC12 dataset downloads correctly
  2. Vocabulary has expected size and structure
  3. Bigram transition matrix is row-normalized
  4. Save/load round-trip works
  5. T2G dataset format is correct

Requires internet to download the dataset — tests are skipped if offline.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np


def test_dataset_download(dataset):
    """Dataset downloads and has train/test splits."""
    assert hasattr(dataset, "keys"), "Dataset is DatasetDict"
    assert "train" in dataset, "Has 'train' split"
    assert "test" in dataset, "Has 'test' split"
    assert len(dataset["train"]) > 0, "Train split not empty"
    assert len(dataset["test"]) > 0, "Test split not empty"


def test_vocabulary(dataset):
    """Vocabulary extraction, sorting, and save/load round-trip."""
    from src.datasets.aslg_dataset import (
        BOS_GLOSS,
        EOS_GLOSS,
        UNK_GLOSS,
        extract_gloss_vocabulary,
        load_vocabulary,
        save_vocabulary,
    )

    vocab = extract_gloss_vocabulary(dataset, split="train")
    assert len(vocab) > 0, "Vocab non-empty"
    assert len(vocab) > 100, f"Vocab has > 100 tokens, got {len(vocab)}"
    assert vocab[0] == BOS_GLOSS, f"Starts with {BOS_GLOSS}"
    assert EOS_GLOSS in vocab, f"Contains {EOS_GLOSS}"
    assert UNK_GLOSS in vocab, f"Contains {UNK_GLOSS}"
    assert all(vocab[i] <= vocab[i + 1] for i in range(3, len(vocab) - 1)), "Sorted"

    with tempfile.TemporaryDirectory() as tmp:
        vpath = Path(tmp) / "test_vocab.txt"
        save_vocabulary(vocab, str(vpath))
        reloaded = load_vocabulary(str(vpath))
        assert reloaded == vocab, "Vocab save/load round-trip"


def test_transition_matrix(dataset):
    """Bigram transition matrix shape, normalization, and scoring."""
    from src.datasets.aslg_dataset import extract_gloss_vocabulary
    from src.datasets.transition_matrix import (
        compute_bigram_transitions,
        load_transition_matrix,
        save_transition_matrix,
        sequence_score_bigram,
        transition_score,
    )

    vocab = extract_gloss_vocabulary(dataset, split="train")
    bigram = compute_bigram_transitions(dataset, vocab, split="train", smoothing=1.0)
    V = len(vocab)
    assert bigram.shape == (V, V), f"Matrix shape is (V, V), got {bigram.shape}"
    assert bigram.dtype == np.float32, "Matrix is float32"
    assert np.allclose(bigram.sum(axis=1), 1.0, atol=1e-5), "Rows sum to 1.0"

    row_mins = bigram.min(axis=1)
    assert np.all(row_mins > 0), "All rows have non-zero minimum (smoothing active)"

    score = transition_score(bigram, 0, 1)
    assert 0.0 <= score <= 1.0, f"Transition score in [0,1], got {score:.6f}"
    assert score > 0.0, "Transition score > 0 (smoothed)"

    with tempfile.TemporaryDirectory() as tmp:
        mpath = str(Path(tmp) / "test_bigram.npy")
        save_transition_matrix(bigram, mpath)
        reloaded = load_transition_matrix(mpath)
        assert np.allclose(bigram, reloaded), "Bigram save/load round-trip"

    indices = [0, 1, 2, 3, 4]
    log_prob = sequence_score_bigram(bigram, indices)
    assert log_prob < 0.0, f"Sequence log-prob is negative, got {log_prob:.4f}"
    assert np.isfinite(log_prob), "Sequence log-prob is finite"


def test_t2g_dataset(dataset):
    """T2G dataset format has correct columns and content."""
    from src.datasets.aslg_dataset import build_t2g_dataset

    t2g = build_t2g_dataset(dataset, split="train", max_samples=50)
    assert len(t2g) == 50, f"T2G dataset has correct size, got {len(t2g)}"
    assert "prompt" in t2g.column_names, "Has 'prompt' column"
    assert "completion" in t2g.column_names, "Has 'completion' column"
    assert "difficulty" in t2g.column_names, "Has 'difficulty' column"

    sample = t2g[0]
    assert isinstance(sample["prompt"], str) and len(sample["prompt"]) > 0
    assert isinstance(sample["completion"], str) and len(sample["completion"]) > 0
    assert sample["difficulty"] in ("simple", "medium", "hard")
