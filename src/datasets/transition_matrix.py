"""
Transition Matrix Computation (Viterbi Proxy).

Computes N‑gram transition probability matrices for ASL gloss sequences
on the training set.  These matrices serve as the *Procedural Knowledge Graph*
for the Viterbi proxy in T2G — telling us how likely gloss ``B`` follows gloss ``A``.

Supports:
    - Bigram transition matrix:  ``P(gloss_j | gloss_i)``
    - Trigram transition matrix: ``P(gloss_k | gloss_i, gloss_j)``

Matrices are saved as NumPy ``.npy`` files for fast loading during GRPO training.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from tqdm import tqdm

from datasets import DatasetDict

from .aslg_dataset import BOS_GLOSS, EOS_GLOSS, UNK_GLOSS

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

TransitionDict = dict[str, dict[str, float]]  # {token_i: {token_j: prob}}


# ---------------------------------------------------------------------------
# Transition matrix computation
# ---------------------------------------------------------------------------


def compute_bigram_transitions(
    dataset: DatasetDict,
    vocab: list[str],
    split: str = "train",
    smoothing: float = 1.0,
) -> np.ndarray:
    """Compute a bigram transition probability matrix on the training glosses.

    .. math::

        P(\\text{gloss}_j \\mid \\text{gloss}_i) =
        \\frac{
            \\text{count}(\\text{gloss}_i, \\text{gloss}_j) + \\alpha
        }{
            \\text{count}(\\text{gloss}_i) + \\alpha \\cdot |V|
        }

    where :math:`\\alpha` is the Laplace (add‑:math:`\\alpha`) smoothing factor
    and :math:`|V|` is the vocabulary size.

    Args:
        dataset: The ASLG-PC12 ``DatasetDict``.
        vocab: The sorted gloss vocabulary (must include ``<BOS>``, ``<EOS>``,
            ``<UNK>`` as the first three entries if desired).
        split: Which split to compute from.
        smoothing: Laplace additive smoothing factor (``1.0`` = classical
            Laplace smoothing).  Set to ``0.0`` for raw MLE.

    Returns:
        A ``(V, V)`` float32 NumPy array where ``matrix[i, j]`` is
        ``P(gloss_j | gloss_i)``.  Each row sums to 1.0.
    """
    logger.info(
        f"Computing bigram transitions on '{split}' split "
        f"(|V|={len(vocab)}, smoothing={smoothing})..."
    )

    # Build token → index mapping
    token_to_idx: dict[str, int] = {t: i for i, t in enumerate(vocab)}
    V = len(vocab)

    # Count matrix (V × V)
    counts = np.zeros((V, V), dtype=np.float64)

    bos_idx = token_to_idx.get(BOS_GLOSS, -1)
    eos_idx = token_to_idx.get(EOS_GLOSS, -1)
    unk_idx = token_to_idx.get(UNK_GLOSS, 0)

    for sample in tqdm(dataset[split], desc="Counting bigrams"):
        gloss_seq: str = sample.get("gloss", "")
        tokens = gloss_seq.split()
        if not tokens:
            continue

        # Wrap with BOS and EOS
        wrapped = tokens
        if bos_idx >= 0:
            wrapped = [BOS_GLOSS] + tokens
        if eos_idx >= 0:
            wrapped = wrapped + [EOS_GLOSS]

        for i in range(len(wrapped) - 1):
            curr = token_to_idx.get(wrapped[i], unk_idx)
            nxt = token_to_idx.get(wrapped[i + 1], unk_idx)
            counts[curr, nxt] += 1.0

    # Apply smoothing and normalize
    row_sums = counts.sum(axis=1, keepdims=True)
    transition = (counts + smoothing) / (row_sums + smoothing * V)

    # Ensure rows sum to 1 (numerical stability)
    transition = transition.astype(np.float32)

    logger.info(
        f"  Bigram transition matrix: shape={transition.shape}, "
        f"density={np.count_nonzero(transition) / transition.size:.2%}"
    )
    return transition


def compute_trigram_transitions(
    dataset: DatasetDict,
    vocab: list[str],
    split: str = "train",
    smoothing: float = 1.0,
) -> np.ndarray:
    """Compute a trigram transition probability matrix.

    .. math::

        P(\\text{gloss}_k \\mid \\text{gloss}_i, \\text{gloss}_j) =
        \\frac{
            \\text{count}(\\text{gloss}_i, \\text{gloss}_j, \\text{gloss}_k) + \\alpha
        }{
            \\text{count}(\\text{gloss}_i, \\text{gloss}_j) + \\alpha \\cdot |V|
        }

    Args:
        dataset: The ASLG-PC12 ``DatasetDict``.
        vocab: The sorted gloss vocabulary.
        split: Which split to compute from.
        smoothing: Laplace additive smoothing factor.

    Returns:
        A ``(V, V, V)`` float32 NumPy array where ``matrix[i, j, k]`` is
        ``P(gloss_k | gloss_i, gloss_j)``.  Each ``(i, j)`` slice sums to 1.0.

    Note:
        Trigram matrices are much larger (``V³``).  For vocabularies larger
        than ~2000 tokens, consider using sparse representations or
        lower-order fallback (backoff to bigram).
    """
    logger.info(
        f"Computing trigram transitions on '{split}' split "
        f"(|V|={len(vocab)}, smoothing={smoothing})..."
    )

    token_to_idx: dict[str, int] = {t: i for i, t in enumerate(vocab)}
    V = len(vocab)

    if V > 1500:
        logger.warning(
            f"Vocabulary size {V} is >1500. Trigram matrix would be "
            f"{V}³ = {V**3:,} entries (~{V**3 * 4 / 1e9:.1f} GB). "
            f"Consider using bigrams only."
        )

    counts = np.zeros((V, V, V), dtype=np.float64)
    bos_idx = token_to_idx.get(BOS_GLOSS, -1)
    eos_idx = token_to_idx.get(EOS_GLOSS, -1)
    unk_idx = token_to_idx.get(UNK_GLOSS, 0)

    for sample in tqdm(dataset[split], desc="Counting trigrams"):
        gloss_seq: str = sample.get("gloss", "")
        tokens = gloss_seq.split()
        if not tokens:
            continue

        # Wrap with BOS (×2) and EOS
        wrapped: list[str] = []
        if bos_idx >= 0:
            wrapped = [BOS_GLOSS, BOS_GLOSS] + tokens
        else:
            wrapped = list(tokens)
        if eos_idx >= 0:
            wrapped = wrapped + [EOS_GLOSS]

        for i in range(len(wrapped) - 2):
            c0 = token_to_idx.get(wrapped[i], unk_idx)
            c1 = token_to_idx.get(wrapped[i + 1], unk_idx)
            c2 = token_to_idx.get(wrapped[i + 2], unk_idx)
            counts[c0, c1, c2] += 1.0

    # Smooth and normalize
    bigram_counts = counts.sum(axis=2)  # (V, V)
    # For rows where bigram_count == 0, we use smoothing only
    denom = bigram_counts[:, :, np.newaxis] + smoothing * V
    transition = (counts + smoothing) / denom

    transition = transition.astype(np.float32)
    logger.info(
        f"  Trigram transition matrix: shape={transition.shape}, "
        f"density={np.count_nonzero(transition) / transition.size:.4%}"
    )
    return transition


def save_transition_matrix(matrix: np.ndarray, path: str | Path) -> None:
    """Save a transition matrix to disk in ``.npy`` format.

    Args:
        matrix: The transition probability matrix (NumPy array).
        path: Output file path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(path), matrix)
    logger.info(
        f"Transition matrix saved to {path} "
        f"(shape={matrix.shape}, dtype={matrix.dtype})"
    )


def load_transition_matrix(path: str | Path) -> np.ndarray:
    """Load a transition matrix from disk.

    Args:
        path: File path to ``.npy`` file.

    Returns:
        The transition probability matrix.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Transition matrix not found: {path}")
    matrix = np.load(str(path))
    logger.info(f"Transition matrix loaded from {path} (shape={matrix.shape})")
    return matrix


def transition_score(
    transition_matrix: np.ndarray,
    token_i: int,
    token_j: int,
) -> float:
    """Look up a single transition probability from the matrix.

    Args:
        transition_matrix: The ``(V, V)`` bigram transition matrix.
        token_i: Index of the source gloss.
        token_j: Index of the target gloss.

    Returns:
        ``P(gloss_j | gloss_i)``, a float in ``[0, 1]``.
    """
    return float(transition_matrix[token_i, token_j])


def sequence_score_bigram(
    transition_matrix: np.ndarray,
    token_indices: list[int],
) -> float:
    """Compute the cumulative bigram log-probability for a gloss sequence.

    Args:
        transition_matrix: The ``(V, V)`` bigram transition matrix.
        token_indices: List of token indices representing the gloss sequence
            (should include BOS and EOS markers).

    Returns:
        Sum of log-probabilities (NaN-safe: replaces ``log(0)`` with a large
        negative value).
    """
    if len(token_indices) < 2:
        return 0.0

    log_prob: float = 0.0
    small_eps = 1e-10  # for numerical stability

    for idx in range(len(token_indices) - 1):
        p = max(
            transition_matrix[token_indices[idx], token_indices[idx + 1]], small_eps
        )
        log_prob += np.log(p)

    return log_prob


def sequence_score_trigram(
    transition_matrix: np.ndarray,
    token_indices: list[int],
) -> float:
    """Compute the cumulative trigram log-probability for a gloss sequence.

    Args:
        transition_matrix: The ``(V, V, V)`` trigram transition matrix.
        token_indices: List of token indices (with doubled BOS and EOS).

    Returns:
        Sum of log-probabilities.
    """
    if len(token_indices) < 3:
        return 0.0

    log_prob: float = 0.0
    small_eps = 1e-10

    for idx in range(len(token_indices) - 2):
        p = max(
            transition_matrix[
                token_indices[idx], token_indices[idx + 1], token_indices[idx + 2]
            ],
            small_eps,
        )
        log_prob += np.log(p)

    return log_prob
