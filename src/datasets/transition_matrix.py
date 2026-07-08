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
from collections import Counter
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


# ---------------------------------------------------------------------------
# Viterbi Optimal Path
# ---------------------------------------------------------------------------


def compute_viterbi_path(
    transition_matrix: np.ndarray,
    start_idx: int,
    end_idx: int,
    length: int,
) -> tuple[list[int], float]:
    """Compute the globally most probable path of a given length through
    the bigram transition matrix using the Viterbi algorithm.

    The path is constrained to start at ``start_idx`` and end at ``end_idx``.
    This finds the sequence of ``length`` token indices that maximizes the
    cumulative bigram log-probability under the Markov model.

    .. note::

       Without emission probabilities (i.e., without conditioning on the
       source English text), the Viterbi optimum for a pure Markov chain
       tends to degenerate into repetitive loops of the single
       highest-probability transition (e.g., ``IX → IX → IX → …``).

       Use this as a *theoretical upper bound* for the structural score,
       not as a linguistically meaningful gloss sequence.  For a
       semantically grounded baseline, consider comparing against the
       gold reference gloss instead (see ``gold_structure_reward`` in
       ``src/rewards/t2g_rewards.py``).

    Args:
        transition_matrix: The ``(V, V)`` bigram transition matrix.
        start_idx: Index of the start token (typically ``<BOS>``).
        end_idx: Index of the end token (typically ``<EOS>``).
        length: Desired path length (number of tokens, **including** BOS
            and EOS).  Must be ``>= 2``.

    Returns:
        A tuple ``(path, viterbi_log_prob)`` where:
            - ``path`` is a list of token indices of length ``length``.
            - ``viterbi_log_prob`` is the (maximized) cumulative
              log-probability of the optimal path.

    Raises:
        ValueError: If ``length < 2``.

    Example:
        >>> import numpy as np
        >>> # 3x3 matrix: BOS=0, A=1, EOS=2
        >>> T = np.array([[0, 1, 0], [0, 0.2, 0.8], [0, 0, 0]], dtype=np.float32)
        >>> path, score = compute_viterbi_path(T, 0, 2, 4)
        >>> len(path)
        4
        >>> path[0], path[-1]
        (0, 2)
    """
    if length < 2:
        raise ValueError(f"Path length must be >= 2, got {length}")

    V = transition_matrix.shape[0]
    small_eps = 1e-10

    # dp[t][s] = maximum log-prob of being in state s at step t
    dp = np.full((length, V), -np.inf, dtype=np.float64)
    backtrack = np.zeros((length, V), dtype=np.int32)

    # Step 0: start from start_idx
    dp[0, start_idx] = 0.0

    # Steps 1 to length-2: free transitions among all states
    for t in range(1, length - 1):
        for s in range(V):
            # Score for transitioning from any prev state to s
            trans_log_probs = np.log(np.maximum(transition_matrix[:, s], small_eps))
            scores = dp[t - 1, :] + trans_log_probs
            best_prev = int(np.argmax(scores))
            dp[t, s] = scores[best_prev]
            backtrack[t, s] = best_prev

    # Final step (length-1): must transition into end_idx
    t_final = length - 1
    trans_log_probs = np.log(np.maximum(transition_matrix[:, end_idx], small_eps))
    scores = dp[t_final - 1, :] + trans_log_probs
    best_prev = int(np.argmax(scores))
    dp[t_final, end_idx] = scores[best_prev]
    backtrack[t_final, end_idx] = best_prev

    viterbi_log_prob = float(dp[t_final, end_idx])

    # Backtrack to recover the optimal path
    path: list[int] = [end_idx]
    for t in range(t_final, 0, -1):
        prev = backtrack[t, path[-1]]
        path.append(int(prev))
    path.reverse()

    return path, viterbi_log_prob


def viterbi_optimal_score(
    transition_matrix: np.ndarray,
    start_idx: int,
    end_idx: int,
    length: int,
) -> float:
    """Compute only the Viterbi optimal log-probability (no path backtracking).

    Faster than ``compute_viterbi_path`` when you only need the score,
    not the actual path.

    Args:
        transition_matrix: The ``(V, V)`` bigram transition matrix.
        start_idx: Index of the start token.
        end_idx: Index of the end token.
        length: Desired path length.

    Returns:
        The Viterbi optimal cumulative log-probability.
    """
    if length < 2:
        raise ValueError(f"Path length must be >= 2, got {length}")

    V = transition_matrix.shape[0]
    small_eps = 1e-10

    dp = np.full((length, V), -np.inf, dtype=np.float64)
    dp[0, start_idx] = 0.0

    for t in range(1, length - 1):
        for s in range(V):
            trans_log_probs = np.log(np.maximum(transition_matrix[:, s], small_eps))
            dp[t, s] = np.max(dp[t - 1, :] + trans_log_probs)

    t_final = length - 1
    trans_log_probs = np.log(np.maximum(transition_matrix[:, end_idx], small_eps))
    dp[t_final, end_idx] = np.max(dp[t_final - 1, :] + trans_log_probs)

    return float(dp[t_final, end_idx])


# ---------------------------------------------------------------------------
# Diverse Viterbi (with diversity constraint to prevent degeneracy)
# ---------------------------------------------------------------------------


def _path_diversity(path: list[int], exclude_tokens: set[int] | None = None) -> float:
    """Compute the unique-token ratio of a path (excluding special tokens).

    Returns:
        Float in ``[0, 1]`` where 1.0 = all tokens unique, 0.0 = all same.
    """
    if len(path) < 2:
        return 0.0
    excluded = exclude_tokens or set()
    tokens = [t for t in path[1:-1] if t not in excluded]  # skip BOS/EOS
    if not tokens:
        return 0.0
    return len(set(tokens)) / len(tokens)


def _find_overrepresented(path: list[int], max_occurrences: int) -> set[int]:
    """Find token indices that appear more than ``max_occurrences`` times."""
    counts = Counter(path[1:-1])  # skip BOS/EOS
    return {t for t, c in counts.items() if c > max_occurrences}


def compute_diverse_viterbi_path(
    transition_matrix: np.ndarray,
    start_idx: int,
    end_idx: int,
    length: int,
    self_loop_penalty: float = 0.5,
    max_occurrences: int = 2,
    diversity_threshold: float = 0.3,
    max_iters: int = 3,
) -> tuple[list[int], float]:
    """Compute the Viterbi-optimal path with diversity constraints.

    Extends the standard Viterbi algorithm with two anti-degeneracy mechanisms:

    1. **Self-loop penalty**: A log-probability penalty ``self_loop_penalty``
       is subtracted whenever the path stays in the same state (``s → s``).
       This discourages paths like ``IX → IX → IX → …``.

    2. **Iterative token ban**: After computing a candidate path, tokens that
       appear more than ``max_occurrences`` times are partially banned
       (their self-transition probabilities are lowered) and the Viterbi
       DP is re-run, up to ``max_iters`` times, until the path passes the
       ``diversity_threshold`` (unique-token ratio).

    .. note::

       This is designed to produce a **linguistically meaningful** Viterbi
       baseline for the ``viterbi_distance_reward``.  Without these
       constraints, the pure Markov-chain Viterbi degenerates into
       repetitive loops (e.g., ``IX → IX → IX``).

    Args:
        transition_matrix: The ``(V, V)`` bigram transition matrix.
        start_idx: Index of the start token (typically ``<BOS>``).
        end_idx: Index of the end token (typically ``<EOS>``).
        length: Desired path length (including BOS and EOS).
        self_loop_penalty: Log-prob penalty for self-transitions.
        max_occurrences: Maximum allowed occurrences per token before
            iterative re-optimization.
        diversity_threshold: Minimum unique-token ratio to accept a path.
        max_iters: Maximum number of iterative re-optimizations.

    Returns:
        A tuple ``(path, viterbi_log_prob)`` where ``path`` is a reasonably
        diverse token-index list of length ``length``.

    Raises:
        ValueError: If ``length < 2``.
    """
    if length < 2:
        raise ValueError(f"Path length must be >= 2, got {length}")

    V = transition_matrix.shape[0]
    small_eps = 1e-10
    penalty_matrix = transition_matrix.copy()

    # Exclude special tokens (BOS, EOS) from diversity checks
    special_tokens = {start_idx, end_idx}

    for iteration in range(max_iters + 1):
        # ── DP with self-loop penalty ──────────────────────────────────
        dp = np.full((length, V), -np.inf, dtype=np.float64)
        backtrack = np.zeros((length, V), dtype=np.int32)
        dp[0, start_idx] = 0.0

        for t in range(1, length - 1):
            for s in range(V):
                trans_log = np.log(np.maximum(penalty_matrix[:, s], small_eps))
                # Apply self-loop penalty
                trans_log[s] -= self_loop_penalty
                scores = dp[t - 1, :] + trans_log
                best_prev = int(np.argmax(scores))
                dp[t, s] = scores[best_prev]
                backtrack[t, s] = best_prev

        # Final step: into end_idx
        t_final = length - 1
        trans_log = np.log(np.maximum(penalty_matrix[:, end_idx], small_eps))
        scores = dp[t_final - 1, :] + trans_log
        best_prev = int(np.argmax(scores))
        dp[t_final, end_idx] = scores[best_prev]
        backtrack[t_final, end_idx] = best_prev

        viterbi_log_prob = float(dp[t_final, end_idx])

        # ── Backtrack ─────────────────────────────────────────────────
        path: list[int] = [end_idx]
        for t in range(t_final, 0, -1):
            prev = backtrack[t, path[-1]]
            path.append(int(prev))
        path.reverse()

        # ── Diversity check ───────────────────────────────────────────
        diversity = _path_diversity(path, exclude_tokens=special_tokens)
        if diversity >= diversity_threshold:
            logger.debug(
                "Diverse Viterbi: diversity=%.3f ≥ threshold=%.3f " "(iteration %d)",
                diversity,
                diversity_threshold,
                iteration,
            )
            return path, viterbi_log_prob

        # Only try to improve if we have iterations left
        if iteration >= max_iters:
            logger.debug(
                "Diverse Viterbi: diversity=%.3f < threshold=%.3f "
                "after %d iterations — returning best effort",
                diversity,
                diversity_threshold,
                iteration,
            )
            return path, viterbi_log_prob

        # ── Apply stronger penalty for over-represented tokens ────────
        overrep = _find_overrepresented(path, max_occurrences)
        if not overrep:
            # No token exceeds max_occurrences but diversity is still
            # low — this only happens for very short paths (L ≤ 5)
            # where the Viterbi reward already returns 0.0.
            # Just return best effort.
            logger.debug(
                "Diverse Viterbi: no over-represented tokens but diversity "
                "=%.3f < %.3f — returning best effort (iteration %d)",
                diversity,
                diversity_threshold,
                iteration,
            )
            return path, viterbi_log_prob

        # Reduce self-transition probabilities for over-represented
        # tokens to discourage them from being chosen again.
        for token_idx in overrep:
            penalty_matrix[token_idx, token_idx] *= 0.3
        logger.debug(
            "Diverse Viterbi iteration %d: penalizing %d over-represented "
            "tokens (diversity=%.3f)",
            iteration + 1,
            len(overrep),
            diversity,
        )

    return path, viterbi_log_prob


def viterbi_optimal_score_diverse(
    transition_matrix: np.ndarray,
    start_idx: int,
    end_idx: int,
    length: int,
    self_loop_penalty: float = 0.5,
    max_occurrences: int = 2,
    diversity_threshold: float = 0.3,
    max_iters: int = 3,
) -> float:
    """Compute the diverse Viterbi optimal log-probability (score only).

    Equivalent to ``compute_diverse_viterbi_path(…)[1]``.  Uses the
    full path for diversity checking but only returns the score.

    Args:
        transition_matrix: The ``(V, V)`` bigram transition matrix.
        start_idx: Index of the start token.
        end_idx: Index of the end token.
        length: Desired path length.
        self_loop_penalty: Log-prob penalty for self-transitions.
        max_occurrences: Maximum allowed occurrences per token.
        diversity_threshold: Minimum unique-token ratio.
        max_iters: Maximum iterative re-optimizations.

    Returns:
        The diverse Viterbi optimal cumulative log-probability.
    """
    _path, score = compute_diverse_viterbi_path(
        transition_matrix,
        start_idx,
        end_idx,
        length,
        self_loop_penalty=self_loop_penalty,
        max_occurrences=max_occurrences,
        diversity_threshold=diversity_threshold,
        max_iters=max_iters,
    )
    return score


# ---------------------------------------------------------------------------
# Soft Viterbi (Forward-Backward) — Differentiable Viterbi relaxation
# ---------------------------------------------------------------------------


def forward_log_probs(
    transition_matrix: np.ndarray,
    start_idx: int,
    end_idx: int,
    length: int,
) -> np.ndarray:
    """Compute forward log-probabilities (alpha) for all states at each step.

    This is the forward pass of the forward-backward algorithm on the
    bigram Markov chain, computed in log-space for numerical stability.

    .. math::

        \\alpha_t(s) = \\log P(o_1, \\ldots, o_t, q_t = s)

    where :math:`q_t` is the state at step :math:`t`.

    Inspired by the Differentiable Viterbi Layer (DVL) in ViterbiPlanNet
    (arXiv:2603.04265), which replaces the non-differentiable argmax
    Viterbi with smooth forward-backward computations to allow gradient
    flow through the structural reward.

    Args:
        transition_matrix: ``(V, V)`` bigram transition probability matrix.
        start_idx: Index of the start token (typically ``<BOS>``).
        end_idx: Index of the end token (typically ``<EOS>``).
        length: Path length (including BOS and EOS).

    Returns:
        ``alpha`` array of shape ``(length, V)`` where
        ``alpha[t, s] = log P(prefix_1..t, state_t=s)``.
    """
    if length < 2:
        raise ValueError(f"Path length must be >= 2, got {length}")

    V = transition_matrix.shape[0]
    small_eps = 1e-10

    log_trans = np.log(np.maximum(transition_matrix, small_eps))

    alpha = np.full((length, V), -np.inf, dtype=np.float64)
    alpha[0, start_idx] = 0.0

    for t in range(1, length - 1):
        # alpha[t, s] = logsumexp(alpha[t-1, :] + log_trans[:, s])
        scores = alpha[t - 1][:, np.newaxis] + log_trans  # (V, V)
        alpha[t, :] = logsumexp(scores, axis=0)

    # Final step: into end_idx
    t_final = length - 1
    alpha[t_final, end_idx] = logsumexp(alpha[t_final - 1] + log_trans[:, end_idx])

    return alpha


def backward_log_probs(
    transition_matrix: np.ndarray,
    start_idx: int,
    end_idx: int,
    length: int,
) -> np.ndarray:
    """Compute backward log-probabilities (beta) for all states at each step.

    This is the backward pass of the forward-backward algorithm,
    computed in log-space.

    .. math::

        \\beta_t(s) = \\log P(o_{t+1}, \\ldots, o_T \\mid q_t = s)

    Args:
        transition_matrix: ``(V, V)`` bigram transition probability matrix.
        start_idx: Index of the start token.
        end_idx: Index of the end token.
        length: Path length (including BOS and EOS).

    Returns:
        ``beta`` array of shape ``(length, V)``.
    """
    if length < 2:
        raise ValueError(f"Path length must be >= 2, got {length}")

    V = transition_matrix.shape[0]
    small_eps = 1e-10

    log_trans = np.log(np.maximum(transition_matrix, small_eps))

    beta = np.full((length, V), -np.inf, dtype=np.float64)
    beta[length - 1, end_idx] = 0.0

    for t in range(length - 2, 0, -1):
        # beta[t, s] = logsumexp(log_trans[s, :] + beta[t+1, :])
        scores = log_trans + beta[t + 1][np.newaxis, :]  # (V, V)
        beta[t, :] = logsumexp(scores, axis=1)

    # Step 0: from start_idx
    beta[0, start_idx] = logsumexp(log_trans[start_idx, :] + beta[1, :])

    return beta


def logsumexp(x: np.ndarray, axis: int | None = None) -> np.ndarray:
    """Numerically stable log-sum-exp.

    Args:
        x: Input array.
        axis: Axis along which to compute logsumexp.

    Returns:
        Log-sum-exp of ``x`` along ``axis``.
    """
    x_max = np.max(x, axis=axis, keepdims=True)
    # Handle -inf max (all elements are -inf)
    x_max = np.where(np.isinf(x_max), 0.0, x_max)
    result = np.log(np.sum(np.exp(x - x_max), axis=axis, keepdims=True)) + x_max
    if axis is not None:
        result = result.squeeze(axis=axis)
    return result


def soft_viterbi_score(
    transition_matrix: np.ndarray,
    start_idx: int,
    end_idx: int,
    length: int,
) -> float:
    """Compute the soft Viterbi (forward-backward) log-probability.

    This is the **differentiable** relaxation of the Viterbi optimal
    score, inspired by ViterbiPlanNet's Differentiable Viterbi Layer
    (arXiv:2603.04265).  Instead of taking the max (hard Viterbi), it
    uses the log-sum-exp (soft Viterbi) which provides a smooth
    upper bound that allows gradient flow.

    .. math::

        \\text{soft\\_viterbi} = \\text{logsumexp}_{\\text{all paths}}
        \\left( \\sum_{t=1}^{L-1} \\log P(q_t \\mid q_{t-1}) \\right)

    This equals the log-partition function ``log Z`` of the Markov chain
    over all paths of the given length from ``start_idx`` to ``end_idx``.

    The soft Viterbi score is always >= the hard Viterbi score (since
    logsumexp >= max), providing a tighter and smoother upper bound for
    the ``soft_viterbi_distance_reward``.

    Args:
        transition_matrix: ``(V, V)`` bigram transition probability matrix.
        start_idx: Index of the start token (typically ``<BOS>``).
        end_idx: Index of the end token (typically ``<EOS>``).
        length: Path length (including BOS and EOS).

    Returns:
        The soft Viterbi log-probability (log-partition function).
    """
    alpha = forward_log_probs(transition_matrix, start_idx, end_idx, length)
    return float(alpha[length - 1, end_idx])


def soft_viterbi_marginals(
    transition_matrix: np.ndarray,
    start_idx: int,
    end_idx: int,
    length: int,
) -> np.ndarray:
    """Compute edge marginals (posterior probabilities) via forward-backward.

    Returns the posterior probability ``P(q_t = s, q_{t+1} = s' | path)``
    for each step ``t`` and each pair of states ``(s, s')``.  This is the
    "soft" version of the Viterbi path — instead of a single best path,
    it gives a distribution over all paths weighted by their probability.

    Used by the differentiable Viterbi reward to provide per-edge
    gradient signals (inspired by ViterbiPlanNet's DVL).

    Args:
        transition_matrix: ``(V, V)`` bigram transition probability matrix.
        start_idx: Index of the start token.
        end_idx: Index of the end token.
        length: Path length (including BOS and EOS).

    Returns:
        ``marginals`` array of shape ``(length-1, V, V)`` where
        ``marginals[t, s, s']`` is the posterior probability of
        transitioning from state ``s`` to ``s'`` at step ``t``.
    """
    V = transition_matrix.shape[0]
    small_eps = 1e-10

    alpha = forward_log_probs(transition_matrix, start_idx, end_idx, length)
    beta = backward_log_probs(transition_matrix, start_idx, end_idx, length)
    log_trans = np.log(np.maximum(transition_matrix, small_eps))

    # Log-partition function (normalizer)
    log_z = alpha[length - 1, end_idx]

    marginals = np.zeros((length - 1, V, V), dtype=np.float64)

    for t in range(length - 1):
        # log P(q_t=s, q_{t+1}=s' | path) = alpha[t, s] + log_trans[s, s'] + beta[t+1, s'] - log_z
        log_edge = (
            alpha[t][:, np.newaxis] + log_trans + beta[t + 1][np.newaxis, :] - log_z
        )
        marginals[t] = np.exp(log_edge)

    return marginals
