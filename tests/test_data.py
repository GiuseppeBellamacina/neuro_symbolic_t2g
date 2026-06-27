#!/usr/bin/env python3
"""Verify Data Ingestion & Transition Matrix ? Test Script.

Validates:
  1. ASLG-PC12 dataset downloads correctly
  2. Vocabulary has expected size and structure
  3. Bigram transition matrix is row-normalized
  4. Save/load round-trip works
  5. T2G dataset format is correct

Usage:
    python tests/test_data.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

PASS = 0
FAIL = 0


def check(name: str, condition: bool, detail: str = "") -> None:
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  [PASS] {name}" + (f" -- {detail}" if detail else ""))
    else:
        FAIL += 1
        print(f"  [FAIL] {name}" + (f" -- {detail}" if detail else ""))


def test_dataset_download() -> None:
    print("\n-- 1. Dataset Download --")
    from src.data.aslg_dataset import download_aslg_dataset

    dataset = download_aslg_dataset(cache_dir="data/test_cache")
    check("Dataset is DatasetDict", hasattr(dataset, "keys"))
    check("Has 'train' split", "train" in dataset)
    check("Has 'test' split", "test" in dataset)
    check("Train split not empty", len(dataset["train"]) > 0)
    check("Test split not empty", len(dataset["test"]) > 0)
    print(f"     Train: {len(dataset['train'])} samples, Test: {len(dataset['test'])} samples")
    return dataset


def test_vocabulary(dataset) -> list[str]:
    print("\n-- 2. Vocabulary --")
    from src.data.aslg_dataset import (
        BOS_GLOSS,
        EOS_GLOSS,
        UNK_GLOSS,
        extract_gloss_vocabulary,
        load_vocabulary,
        save_vocabulary,
    )

    vocab = extract_gloss_vocabulary(dataset, split="train")
    check("Vocab non-empty", len(vocab) > 0)
    check("Vocab has > 100 tokens", len(vocab) > 100, f"{len(vocab)} tokens")
    check(f"Starts with {BOS_GLOSS}", vocab[0] == BOS_GLOSS)
    check(f"Contains {EOS_GLOSS}", EOS_GLOSS in vocab)
    check(f"Contains {UNK_GLOSS}", UNK_GLOSS in vocab)
    check("Vocab is sorted (after special tokens)", all(vocab[i] <= vocab[i + 1] for i in range(3, len(vocab) - 1)))

    # Save/load round-trip
    with tempfile.TemporaryDirectory() as tmp:
        vpath = Path(tmp) / "test_vocab.txt"
        save_vocabulary(vocab, str(vpath))
        reloaded = load_vocabulary(str(vpath))
        check("Vocab save/load round-trip", reloaded == vocab)

    return vocab


def test_transition_matrix(dataset, vocab) -> None:
    print("\n-- 3. Transition Matrix --")
    from src.data.transition_matrix import (
        compute_bigram_transitions,
        load_transition_matrix,
        save_transition_matrix,
        sequence_score_bigram,
        transition_score,
    )

    bigram = compute_bigram_transitions(dataset, vocab, split="train", smoothing=1.0)
    V = len(vocab)
    check("Matrix shape is (V, V)", bigram.shape == (V, V), f"{bigram.shape}")
    check("Matrix is float32", bigram.dtype == np.float32)
    check("Rows sum to 1.0", np.allclose(bigram.sum(axis=1), 1.0, atol=1e-5))

    # No zero rows (Laplace smoothing guarantees this)
    row_mins = bigram.min(axis=1)
    check("All rows have non-zero minimum (smoothing active)", np.all(row_mins > 0))

    # Transition score
    score = transition_score(bigram, 0, 1)
    check("Transition score in [0, 1]", 0.0 <= score <= 1.0, f"{score:.6f}")
    check("Transition score > 0 (smoothed)", score > 0.0)

    # Save/load round-trip
    with tempfile.TemporaryDirectory() as tmp:
        mpath = str(Path(tmp) / "test_bigram.npy")
        save_transition_matrix(bigram, mpath)
        reloaded = load_transition_matrix(mpath)
        check("Bigram save/load round-trip", np.allclose(bigram, reloaded))

    # Sequence scoring
    indices = [0, 1, 2, 3, 4]  # BOS, 3 glosses, EOS
    log_prob = sequence_score_bigram(bigram, indices)
    check("Sequence log-prob is negative (valid)", log_prob < 0.0, f"{log_prob:.4f}")
    check("Sequence log-prob is finite", np.isfinite(log_prob))


def test_t2g_dataset(dataset) -> None:
    print("\n-- 4. T2G Dataset --")
    from src.data.aslg_dataset import build_t2g_dataset

    t2g = build_t2g_dataset(dataset, split="train", max_samples=50)
    check("T2G dataset has correct size", len(t2g) == 50)
    check("Has 'prompt' column", "prompt" in t2g.column_names)
    check("Has 'completion' column", "completion" in t2g.column_names)
    check("Has 'difficulty' column", "difficulty" in t2g.column_names)

    sample = t2g[0]
    check("Prompt is non-empty string", isinstance(sample["prompt"], str) and len(sample["prompt"]) > 0)
    check("Completion is non-empty string", isinstance(sample["completion"], str) and len(sample["completion"]) > 0)
    check("Difficulty is non-empty", sample["difficulty"] in ("simple", "medium", "hard"))

    print(f"     Sample prompt: {sample['prompt'][:60]}...")
    print(f"     Sample completion: {sample['completion'][:60]}...")


def main() -> None:
    global PASS, FAIL
    print("=" * 60)
    print("TEST: Data Ingestion & Transition Matrix")
    print("=" * 60)

    try:
        ds = test_dataset_download()
        vocab = test_vocabulary(ds)
        test_transition_matrix(ds, vocab)
        test_t2g_dataset(ds)
    except Exception as e:
        print(f"\n  !! CRASH: {e}")
        import traceback
        traceback.print_exc()
        FAIL += 1

    print(f"\n{'=' * 60}")
    print(f"RESULTS: {PASS} passed, {FAIL} failed, {PASS + FAIL} total")
    print(f"{'=' * 60}")
    sys.exit(0 if FAIL == 0 else 1)


if __name__ == "__main__":
    main()
