"""Prepare network-acquired artifacts with lightweight dependencies."""

from __future__ import annotations

import argparse
import os


def prepare(args: argparse.Namespace) -> None:
    """Cache the model and dataset, then derive deterministic data artifacts."""
    if os.environ.get("HF_HUB_OFFLINE") == "1":
        raise RuntimeError("setup artifact preparation requires online mode")

    from huggingface_hub import snapshot_download

    from src.datasets.aslg_dataset import (
        download_aslg_dataset,
        extract_gloss_vocabulary,
        save_vocabulary,
    )
    from src.utils.cache_meta import write_cache_meta

    snapshot = snapshot_download(repo_id=args.model_id)
    print(f"Model snapshot cached at {snapshot}")

    dataset = download_aslg_dataset(
        cache_dir=args.dataset_cache,
        seed=args.seed,
        online=True,
    )
    vocabulary = extract_gloss_vocabulary(dataset, split="train")
    save_vocabulary(vocabulary, args.vocab_path)
    write_cache_meta(args.vocab_path, args.seed, len(dataset["train"]))

    if args.build_bigram:
        from src.datasets.transition_matrix import (
            compute_bigram_transitions,
            save_transition_matrix,
        )

        matrix = compute_bigram_transitions(dataset, vocabulary, split="train")
        save_transition_matrix(matrix, args.bigram_path)
        write_cache_meta(args.bigram_path, args.seed, len(dataset["train"]))
    else:
        print("Skipping optional bigram (set BUILD_BIGRAM=1 to build it)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--dataset-cache", default="data/aslg_pc12")
    parser.add_argument("--vocab-path", default="data/gloss_vocab.txt")
    parser.add_argument("--bigram-path", default="data/bigram_transition.npy")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--build-bigram",
        type=lambda value: value == "1",
        default=False,
        metavar="0|1",
    )
    return parser.parse_args()


if __name__ == "__main__":
    prepare(parse_args())
