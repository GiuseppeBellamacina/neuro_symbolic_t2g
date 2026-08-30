#!/usr/bin/env python3
"""Test data ingestion and transition matrix computation.

Validates:
  1. ASLG-PC12 dataset downloads correctly
  2. Vocabulary has expected size and structure
  3. Bigram transition matrix is row-normalized
  4. Save/load round-trip works
  5. T2G dataset format is correct
  6. Dedup by normalized text removes duplicates before the split
  7. Sample IDs are stable and gloss-sensitive (no hash collisions)
  8. Vocab/bigram cache sidecar invalidation (seed / train_size)

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
    assert "text" in t2g.column_names, "Has 'text' column"
    assert "completion" in t2g.column_names, "Has 'completion' column"
    assert "gold_gloss" in t2g.column_names, "Has 'gold_gloss' column"
    assert "sample_id" in t2g.column_names, "Has 'sample_id' column"
    assert "difficulty" in t2g.column_names, "Has 'difficulty' column"

    sample = t2g[0]
    assert isinstance(sample["prompt"], str) and len(sample["prompt"]) > 0
    assert isinstance(sample["completion"], str) and len(sample["completion"]) > 0
    assert sample["difficulty"] in ("simple", "medium", "hard")
    # Contract: gold_gloss is identical to completion.
    assert sample["gold_gloss"] == sample["completion"], "gold_gloss == completion"
    # Contract: sample_id is a SHA256 hex digest (64 chars).
    assert len(sample["sample_id"]) == 64, "sample_id is sha256 hex digest"


# ---------------------------------------------------------------------------
# Synthetic (offline) tests — no dataset download
# ---------------------------------------------------------------------------


def test_dedup_removes_normalized_duplicates():
    """Dedup by normalized text removes case/whitespace duplicates, keeps first."""
    from datasets import Dataset
    from src.datasets.aslg_dataset import deduplicate_by_text

    rows = [
        {"text": "The man walks.", "gloss": "MAN WALK"},
        {"text": "the   man walks.", "gloss": "MAN WALK"},  # whitespace dup
        {"text": "THE MAN WALKS.", "gloss": "MAN WALK"},  # case dup
        {"text": "A dog barks.", "gloss": "DOG BARK"},
    ]
    deduped, removed = deduplicate_by_text(Dataset.from_list(rows))
    assert removed == 2, f"Removed 2 duplicates, got {removed}"
    assert len(deduped) == 2
    # First occurrence is kept.
    assert deduped[0]["text"] == "The man walks."


def test_dedup_prevents_cross_split_leakage(monkeypatch):
    """After dedup + 90/10 split, no normalized text appears in both splits.

    This replicates the exact flow of ``download_aslg_dataset`` with
    ``load_dataset`` monkeypatched to return a synthetic HF dataset, so no
    internet access is required.
    """
    import src.datasets.aslg_dataset as ds_module
    from datasets import Dataset, DatasetDict

    rows = [
        {
            "text": "In the beginning God created the heaven.",
            "gloss": "BEGIN GOD CREATE HEAVEN",
        },
        {
            "text": "in the beginning god created the heaven.",
            "gloss": "BEGIN GOD CREATE HEAVEN",
        },  # dup
        {"text": "And God said let there be light.", "gloss": "GOD SAY LIGHT EXIST"},
        {
            "text": "AND GOD SAID LET THERE BE LIGHT.",
            "gloss": "GOD SAY LIGHT EXIST",
        },  # dup
        {"text": "And the earth was without form.", "gloss": "EARTH FORM NONE"},
        {"text": "The man walks into the house.", "gloss": "MAN WALK ENTER HOUSE"},
        {"text": "The dog chases the cat.", "gloss": "DOG CHASE CAT"},
        {"text": "The boy reads the book.", "gloss": "BOY READ BOOK"},
    ]
    fake = DatasetDict({"train": Dataset.from_list(rows)})
    monkeypatch.setattr(ds_module, "load_dataset", lambda *args, **kwargs: fake)

    result = ds_module.download_aslg_dataset(cache_dir="unused", seed=42)

    train_norm = {ds_module.normalize_text(t) for t in result["train"]["text"]}
    test_norm = {ds_module.normalize_text(t) for t in result["test"]["text"]}
    assert train_norm.isdisjoint(
        test_norm
    ), "No normalized sentence may leak into both train and test"


def test_sample_id_stable_and_gloss_sensitive():
    """sample_id differs per gold gloss; identical normalized text+gloss collide."""
    from datasets import Dataset, DatasetDict
    from src.datasets.aslg_dataset import build_t2g_dataset

    rows = [
        {"text": "The man walks.", "gloss": "MAN WALK"},
        # Same text, different gloss order → different sample_id.
        {"text": "The man walks.", "gloss": "WALK MAN"},
        # Same normalized text + same gloss → identical sample_id.
        {"text": "the   man walks.", "gloss": "MAN WALK"},
    ]
    ds = DatasetDict({"train": Dataset.from_list(rows)})
    out = build_t2g_dataset(ds, split="train")

    ids = list(out["sample_id"])
    assert len(ids) == 3
    assert ids[0] != ids[1], "Different gold gloss → different sample_id"
    assert ids[0] == ids[2], "Same normalized text + same gloss → same sample_id"
    assert all(len(i) == 64 for i in ids), "sample_id is sha256 hex digest"


def test_cache_meta_invalidation(tmp_path):
    """Vocab/bigram cache sidecar invalidates on seed or train_size change."""
    from src.training.grpo_t2g_train import _cache_is_current, _write_cache_meta

    path = tmp_path / "gloss_vocab.txt"
    path.write_text("IX\nMAN\n", encoding="utf-8")

    # Legacy cache without sidecar → never trusted.
    assert not _cache_is_current(path, seed=42, train_size=100)

    _write_cache_meta(path, seed=42, train_size=100)
    assert _cache_is_current(path, seed=42, train_size=100), "Fresh sidecar is valid"

    assert not _cache_is_current(path, seed=43, train_size=100), "Seed change → invalid"
    assert not _cache_is_current(path, seed=42, train_size=101), "Size change → invalid"

    # Missing artifact → invalid.
    assert not _cache_is_current(tmp_path / "missing.npy", seed=42, train_size=100)


# ---------------------------------------------------------------------------
# 9. Vectorized diverse-Viterbi equivalence (perf rewrite regression)
# ---------------------------------------------------------------------------


def test_diverse_viterbi_matches_reference():
    """The vectorized max-plus DP must match the original per-state DP.

    compute_diverse_viterbi_path was rewritten for performance (the old
    per-state loop with an inner np.log over a V-length column cost
    O(L·V²) log evaluations per decode — the 391 s/it GRPO steps of run
    7078).  This test compares it against a reference implementation of
    the ORIGINAL algorithm on small random matrices: scores must match
    and paths may differ only on exact ties (float32 vs float64
    argmax), which never happens when the scores differ.
    """
    from src.datasets.transition_matrix import (
        _find_overrepresented,
        _path_diversity,
        compute_diverse_viterbi_path,
    )

    def reference(
        matrix,
        start,
        end,
        length,
        self_loop_penalty=0.5,
        max_occurrences=2,
        diversity_threshold=0.3,
        max_iters=3,
    ):
        """The original per-state DP (pre-rewrite), verbatim semantics."""
        V = matrix.shape[0]
        eps = 1e-10
        penalty_matrix = matrix.copy()
        special = {start, end}
        path = [start, end]
        vlp = float("-inf")
        for iteration in range(max_iters + 1):
            dp = np.full((length, V), -np.inf, dtype=np.float64)
            backtrack = np.zeros((length, V), dtype=np.int32)
            dp[0, start] = 0.0
            for t in range(1, length - 1):
                for s in range(V):
                    trans_log = np.log(np.maximum(penalty_matrix[:, s], eps))
                    trans_log[s] -= self_loop_penalty
                    scores = dp[t - 1, :] + trans_log
                    best = int(np.argmax(scores))
                    dp[t, s] = scores[best]
                    backtrack[t, s] = best
            t_final = length - 1
            trans_log = np.log(np.maximum(penalty_matrix[:, end], eps))
            scores = dp[t_final - 1, :] + trans_log
            best = int(np.argmax(scores))
            dp[t_final, end] = scores[best]
            backtrack[t_final, end] = best
            vlp = float(dp[t_final, end])
            path = [end]
            for t in range(t_final, 0, -1):
                path.append(int(backtrack[t, path[-1]]))
            path.reverse()
            div = _path_diversity(path, exclude_tokens=special)
            if div >= diversity_threshold or iteration >= max_iters:
                return path, vlp
            overrep = _find_overrepresented(path, max_occurrences)
            if not overrep:
                return path, vlp
            for tok in overrep:
                penalty_matrix[tok, tok] *= 0.3
        return path, vlp

    rng = np.random.default_rng(123)
    n_checked = 0
    for _ in range(20):
        V = int(rng.integers(6, 25))
        m = (
            rng.random((V, V)) ** 3
        )  # skewed: forces degenerate loops → diversity iterations
        m /= m.sum(axis=1, keepdims=True)
        for length in (2, 3, 5, 8, 11):
            p_new, s_new = compute_diverse_viterbi_path(m, 0, V - 1, length)
            p_ref, s_ref = reference(m, 0, V - 1, length)
            n_checked += 1
            assert np.isclose(
                s_new, s_ref, rtol=1e-4, atol=1e-5
            ), f"score mismatch V={V} L={length}: {s_new:.6f} vs {s_ref:.6f}"
            if p_new != p_ref:
                # Path differences allowed ONLY on exact score ties
                # (float32 argmax vs float64 argmax).
                assert (
                    abs(s_new - s_ref) < 1e-4
                ), f"path differs with different scores V={V} L={length}"
    assert n_checked == 100
