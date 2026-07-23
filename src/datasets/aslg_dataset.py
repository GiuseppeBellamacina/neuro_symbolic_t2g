"""
ASLG-PC12 Dataset Ingestion.

Downloads and processes the ``achrafothman/aslg_pc12`` dataset (Hugging Face).
Extracts the unique ASL gloss vocabulary from the training set and provides
utilities for building prompt-completion pairs for GRPO training.

Task: Text-to-Gloss (T2G) — English sentences → ASL gloss sequences.

Reference:
    Othman, A. & Jemni, M. (2012). English-ASL Gloss Parallel Corpus 2012.
"""

from __future__ import annotations

import logging
from pathlib import Path

# tqdm fallback for Apptainer containers without tqdm installed
try:
    from tqdm import tqdm
except ImportError:

    def tqdm(iterable=None, **kwargs):
        return iterable if iterable is not None else iter(())


from datasets import Dataset, DatasetDict, load_dataset

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DATASET_NAME: str = "achrafothman/aslg_pc12"
DEFAULT_CACHE_DIR: str = "data/aslg_pc12"

# Special tokens and boundaries for gloss sequences
BOS_GLOSS: str = "<BOS>"
EOS_GLOSS: str = "<EOS>"
UNK_GLOSS: str = "<UNK>"


def download_aslg_dataset(
    cache_dir: str | None = None,
    seed: int = 42,
) -> DatasetDict:
    """Download (or load from cache) the ASLG-PC12 dataset.

    Args:
        cache_dir: Directory to cache the downloaded dataset.
            Defaults to ``data/aslg_pc12``.
        seed: Random seed for reproducible splits.

    Returns:
        A Hugging Face ``DatasetDict`` with ``"train"`` and ``"test"`` splits.

    Raises:
        RuntimeError: If the dataset cannot be downloaded or loaded.
    """
    cache = cache_dir or DEFAULT_CACHE_DIR
    logger.info(f"Loading dataset '{DATASET_NAME}' (cache: {cache})...")

    try:
        ds: DatasetDict = load_dataset(  # type: ignore[assignment]
            DATASET_NAME,
            cache_dir=cache,
        )
    except Exception as e:
        raise RuntimeError(f"Failed to load dataset '{DATASET_NAME}': {e}") from e

    # Always create a reproducible 90/10 train/test split from the full data.
    # This ensures consistency across runs, models, and HF dataset versions —
    # even if the raw dataset changes, the seed guarantees identical splits.
    if "train" in ds:
        logger.info(f"Creating reproducible 90/10 train/test split (seed={seed})...")
        if "test" in ds:
            logger.warning(
                "Raw dataset already has a 'test' split — ignoring it in favor of "
                "reproducible split."
            )
        split_ds = ds["train"].train_test_split(test_size=0.1, seed=seed)
        ds = DatasetDict({"train": split_ds["train"], "test": split_ds["test"]})
    else:
        raise RuntimeError(
            f"Dataset '{DATASET_NAME}' has no 'train' split. "
            f"Available: {list(ds.keys())}"
        )

    for split_name, split_ds in ds.items():
        logger.info(f"  {split_name}: {len(split_ds)} samples")

    # Validate expected columns
    expected_cols = {"text", "gloss"}
    for split_name, split_ds in ds.items():
        missing = expected_cols - set(split_ds.column_names)
        if missing:
            logger.warning(
                f"Split '{split_name}' missing columns: {missing}. "
                f"Available: {split_ds.column_names}"
            )

    return ds


def extract_gloss_vocabulary(
    dataset: DatasetDict,
    split: str = "train",
    include_special_tokens: bool = True,
) -> list[str]:
    """Extract the unique ASL gloss vocabulary from the dataset.

    Glosses are space-separated tokens in the ``"gloss"`` column.
    Each gloss typically represents one ASL sign (e.g., ``"IX"``, ``"MAN"``,
    ``"WALK"``, ``"fs-JOHN"`` for fingerspelling).

    Args:
        dataset: The ASLG-PC12 ``DatasetDict``.
        split: Which split to extract from (``"train"`` or ``"test"``).
        include_special_tokens: If ``True``, prepend ``<BOS>``, ``<EOS>``,
            and ``<UNK>`` to the vocabulary.

    Returns:
        Sorted list of unique gloss tokens.
    """
    logger.info(f"Extracting gloss vocabulary from '{split}' split...")
    glosses: set[str] = set()

    for sample in tqdm(dataset[split], desc="Extracting gloss tokens"):
        gloss_seq: str = sample.get("gloss", "")
        tokens = gloss_seq.split()
        glosses.update(tokens)

    vocab = sorted(glosses)
    logger.info(f"  Raw unique glosses: {len(vocab)}")

    if include_special_tokens:
        vocab = [BOS_GLOSS, EOS_GLOSS, UNK_GLOSS] + vocab
        logger.info(
            f"  With special tokens: {len(vocab)} "
            f"(+ {BOS_GLOSS}, {EOS_GLOSS}, {UNK_GLOSS})"
        )

    return vocab


def build_t2g_dataset(
    dataset: DatasetDict,
    split: str = "train",
    max_samples: int | None = None,
) -> Dataset:
    """Build a prompt-completion dataset formatted for GRPO training.

    Each sample contains:
        - ``"prompt"``: The English source sentence.
        - ``"completion"``: The gold ASL gloss sequence.
        - ``"difficulty"``: Heuristic based on gloss token count:
          ``"simple"`` (1-5 tokens), ``"medium"`` (6-15), ``"hard"`` (16+).

    Args:
        dataset: The ASLG-PC12 ``DatasetDict``.
        split: Which split to use.
        max_samples: Maximum number of samples (for debugging).  ``None``
            means use all samples.

    Returns:
        A Hugging Face ``Dataset`` with columns ``["prompt", "completion", "difficulty"]``.
    """
    logger.info(f"Building T2G dataset from '{split}' split...")
    split_ds = dataset[split]

    rows: list[dict[str, str]] = []
    for sample in tqdm(split_ds, desc="Building prompt-completion pairs"):
        text: str = sample.get("text", "")
        gloss: str = sample.get("gloss", "")

        if not text.strip() or not gloss.strip():
            continue

        gloss_tokens = gloss.strip().split()
        if len(gloss_tokens) <= 5:
            difficulty = "simple"
        elif len(gloss_tokens) <= 15:
            difficulty = "medium"
        else:
            difficulty = "hard"

        rows.append(
            {
                "prompt": text.strip(),
                "completion": gloss.strip(),
                "difficulty": difficulty,
            }
        )

    if max_samples is not None:
        rows = rows[:max_samples]
        logger.info(f"  Truncated to {max_samples} samples")

    logger.info(f"  Total samples: {len(rows)}")
    return Dataset.from_list(rows)


def save_vocabulary(vocab: list[str], path: str | Path) -> None:
    """Save the gloss vocabulary to disk (one token per line).

    Args:
        vocab: The list of unique gloss tokens.
        path: File path to save to.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(vocab), encoding="utf-8")
    logger.info(f"Vocabulary saved to {path} ({len(vocab)} tokens)")


def load_vocabulary(path: str | Path) -> list[str]:
    """Load gloss vocabulary from disk.

    Args:
        path: File path to load from.

    Returns:
        List of gloss tokens (one per line).
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Vocabulary file not found: {path}")
    vocab = path.read_text(encoding="utf-8").strip().split("\n")
    logger.info(f"Vocabulary loaded from {path} ({len(vocab)} tokens)")
    return vocab
