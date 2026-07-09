"""Shared pytest fixtures for neuro_symbolic_t2g test suite.

This conftest provides fixtures that were previously handled by manual
``main()`` chaining in standalone test scripts. Each fixture is session-scoped
where possible to avoid redundant expensive setup (e.g. dataset download,
tokenizer loading).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

# Ensure project root is on sys.path (same as the old sys.path.insert in each test)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ---------------------------------------------------------------------------
# Reward setup fixture
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def reward_setup():
    """Initialize rewards with a mini vocabulary and bigram matrix.

    Returns a tuple of (vocab, bigram, token_to_idx).
    """
    from src.rewards.t2g_rewards import initialize_rewards

    vocab = [
        "<BOS>",
        "<EOS>",
        "<UNK>",
        "IX",
        "MAN",
        "WALK",
        "HOUSE",
        "BOOK",
        "DOG",
        "CAT",
        "NOT",
        "CAN",
        "WANT",
        "GO",
        "COME",
        "fs-JOHN",
    ]
    V = len(vocab)
    # Create a plausible bigram matrix with Laplace smoothing
    bigram = np.ones((V, V), dtype=np.float32) * (1.0 / V)
    token_to_idx = {t: i for i, t in enumerate(vocab)}

    # Set some plausible transitions
    pairs = [
        ("<BOS>", "IX"),
        ("IX", "MAN"),
        ("MAN", "WALK"),
        ("WALK", "HOUSE"),
        ("HOUSE", "<EOS>"),
    ]
    for a, b in pairs:
        if a in token_to_idx and b in token_to_idx:
            bigram[token_to_idx[a], :] = 0.001
            bigram[token_to_idx[a], token_to_idx[b]] = 0.95
            row = bigram[token_to_idx[a]]
            bigram[token_to_idx[a]] = row / row.sum()

    initialize_rewards(
        bigram,
        vocab,
        viterbi_diversity={
            "self_loop_penalty": 0.5,
            "max_occurrences": 2,
            "diversity_threshold": 0.3,
            "max_iters": 3,
            "verifier_gamma": 1.5,
            "verifier_temperature": 5.0,
        },
    )
    return vocab, bigram, token_to_idx


# ---------------------------------------------------------------------------
# Dataset fixture (requires internet — skipped if offline)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def dataset():
    """Download the ASLG-PC12 dataset (session-scoped).

    Skipped if the dataset cannot be downloaded (no internet).
    """
    pytest.importorskip("datasets")
    from src.datasets.aslg_dataset import download_aslg_dataset

    try:
        ds = download_aslg_dataset(cache_dir="data/test_cache")
    except Exception as e:
        pytest.skip(f"Cannot download dataset: {e}")
    return ds


# ---------------------------------------------------------------------------
# Tokenizer fixture (tries Qwen first, falls back to gpt2)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def tokenizer():
    """Load a tokenizer for grammar tests (session-scoped).

    Tries Qwen2.5-0.5B-Instruct first, falls back to gpt2.
    Skipped if no tokenizer can be loaded.
    """
    from transformers import AutoTokenizer

    for name in ("Qwen/Qwen2.5-0.5B-Instruct", "gpt2"):
        try:
            tok = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
            if tok.pad_token is None:
                tok.pad_token = tok.eos_token
            return tok
        except Exception:
            continue
    pytest.skip("No tokenizer available for testing")
