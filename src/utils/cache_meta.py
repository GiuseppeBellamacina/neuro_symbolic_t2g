"""Neutral cache sidecar helpers shared by training entry points."""

from __future__ import annotations

import json
from pathlib import Path

SUPPORTED_DATASET_NAME = "achrafothman/aslg_pc12"


def cache_meta_path(cache_path: str | Path) -> Path:
    """Return the ``.meta.json`` sidecar path for an artifact."""
    return Path(cache_path).with_suffix(".meta.json")


def write_cache_meta(cache_path: str | Path, seed: int, train_size: int) -> None:
    """Record the dataset seed and train size used to build an artifact."""
    cache_meta_path(cache_path).write_text(
        json.dumps({"seed": seed, "train_size": train_size}, sort_keys=True),
        encoding="utf-8",
    )


def cache_is_current(cache_path: str | Path, seed: int, train_size: int) -> bool:
    """Return whether an artifact and matching sidecar both exist."""
    path = Path(cache_path)
    meta_path = cache_meta_path(cache_path)
    if not path.exists() or not meta_path.exists():
        return False
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return meta.get("seed") == seed and meta.get("train_size") == train_size


def validate_dataset_name(dataset_name: str | None) -> str:
    """Reject dataset alternatives until the shared loader accepts a name."""
    name = dataset_name or SUPPORTED_DATASET_NAME
    if name != SUPPORTED_DATASET_NAME:
        raise ValueError(
            f"Unsupported dataset.name {name!r}; only {SUPPORTED_DATASET_NAME!r} "
            "is supported by download_aslg_dataset"
        )
    return name
