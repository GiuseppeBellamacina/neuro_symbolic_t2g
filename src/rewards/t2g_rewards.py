"""
Reward Functions for T2G GRPO Training.

Seven reward components:

1. **Translation Quality Reward** (ROUGE-L):
   Lexical similarity between generated gloss and gold reference.

2. **Structural Dense Reward** (Bigram Log-Probability):
   Average log-probability of bigram transitions (absolute score).

3. **Gold-Structure Reward** (Gold-Baseline Structural) ⭐:
   Compares LLM bigram score against the gold reference gloss.

4. **Viterbi Distance Reward** (Viterbi-Upper-Bound) 🧪:
   Compares LLM path against the diverse Viterbi optimum.

5. **Gloss-Order Reward** (Word-Level Edit-Distance):
   Normalized Levenshtein distance against the gold gloss sequence —
   complements ROUGE-L with a signal sensitive to gloss ordering.

6. **Format Reward**: Penalizes free text / non-gloss outputs.

7. **Repetition Reward**: Penalizes degenerate token repetition.

8. **Verifier-Scaled Reward** (RECIPE-inspired):
   Uses structural plausibility as a confidence multiplier for translation quality.

Rewards are combined via weighted sum and wrapped to match the signature
expected by TRL's ``GRPOTrainer``:
``fn(completions, prompts, **kwargs) -> list[float]``.
"""

from __future__ import annotations

import hashlib
import logging
import re
from typing import Any, Callable

import numpy as np
from rouge_score import rouge_scorer

from src.utils.text_utils import extract_gloss_text, extract_user_text

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Global state (populated at dataset load time)
# ---------------------------------------------------------------------------

#: Bigram transition matrix (V × V), loaded once at training start.
_bigram_matrix: np.ndarray | None = None

#: Gloss vocabulary (sorted list), used for token→index mapping.
_gloss_vocab: list[str] = []

#: Token→index mapping for fast lookups.
_token_to_idx: dict[str, int] = {}

#: ROUGE-L scorer instance (initialized in ``initialize_rewards``).
_ROUGE_SCORER: rouge_scorer.RougeScorer | None = None

#: Viterbi diversity parameters loaded from config YAML.
#  Configured via ``grammar.viterbi_diversity`` section.
_viterbi_diversity_params: dict[str, float | int] = {
    "self_loop_penalty": 0.5,
    "max_occurrences": 2,
    "diversity_threshold": 0.3,
    "max_iters": 3,
    "verifier_gamma": 1.0,
    # Decoupled from verifier_gamma (see verifier_scaled_reward docstring):
    # controls the softmax temperature used to rescale structural_dense_reward
    # before log1p-scaling it as the verifier confidence multiplier. Default
    # 5.0 gives a gentle curve; configs can override via
    # grammar.viterbi_diversity.verifier_temperature.
    "verifier_temperature": 5.0,
}

#: Gold gloss registry: maps sample_id (SHA256 of user instruction) → gold gloss.
#  Populated at dataset load time via ``register_gold_glosses()``.
_gold_gloss_registry: dict[str, str] = {}


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------


def initialize_rewards(
    bigram_matrix: np.ndarray,
    vocab: list[str],
    viterbi_diversity: dict[str, float | int] | None = None,
) -> None:
    """Initialize global state for reward functions.

    Must be called once before training starts.

    Args:
        bigram_matrix: The ``(V, V)`` bigram transition probability matrix.
        vocab: The sorted gloss vocabulary.
    """
    global _bigram_matrix, _gloss_vocab, _token_to_idx, _ROUGE_SCORER
    global _viterbi_diversity_params
    _bigram_matrix = bigram_matrix
    _gloss_vocab = vocab
    _token_to_idx = {t: i for i, t in enumerate(vocab)}

    # Use ROUGE-L F1 as the primary quality metric
    _ROUGE_SCORER = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)

    # Set Viterbi diversity params from config
    diversity_cfg = viterbi_diversity or {}

    _viterbi_diversity_params = {
        "self_loop_penalty": diversity_cfg.get("self_loop_penalty", 0.5),
        "max_occurrences": diversity_cfg.get("max_occurrences", 2),
        "diversity_threshold": diversity_cfg.get("diversity_threshold", 0.3),
        "max_iters": diversity_cfg.get("max_iters", 3),
        "verifier_gamma": diversity_cfg.get("verifier_gamma", 1.0),
        "verifier_temperature": diversity_cfg.get("verifier_temperature", 5.0),
    }
    logger.info("Viterbi diversity params: %s", _viterbi_diversity_params)


def _extract_sample_id(prompt: Any) -> str:
    """Extract a stable sample ID from a prompt in any format.

    Uses the shared ``extract_user_text`` to get the user instruction,
    then returns its SHA256 hash as a deterministic lookup key.

    Args:
        prompt: The prompt in whatever format GRPOTrainer provides.

    Returns:
        SHA256 hex digest of the user instruction, or ``""`` if no
        user content could be extracted.
    """
    user_text = extract_user_text(prompt)
    if not user_text:
        return ""
    return hashlib.sha256(user_text.encode("utf-8", errors="replace")).hexdigest()


def register_gold_glosses(
    sample_ids: list[str],
    gold_glosses: list[str],
) -> None:
    """Populate the gold gloss registry from the training dataset.

    Called once after dataset preparation, before training starts.
    Maps each sample ID (SHA256 of user instruction) to its
    corresponding gold gloss sequence.

    This registry is used by ``translation_quality_reward`` to look up
    the gold reference for each rollout prompt during GRPO.

    Args:
        sample_ids: Stable sample IDs (hashes of user instructions).
        gold_glosses: The gold gloss completion strings (same order).
    """
    global _gold_gloss_registry
    _gold_gloss_registry = dict(zip(sample_ids, gold_glosses))
    logger.info(f"Gold gloss registry: {len(_gold_gloss_registry)} entries registered")


def _lookup_gold_gloss(prompt: Any) -> str:
    """Look up the gold gloss for a prompt in the registry.

    Extracts a stable sample ID from the prompt (handling any format
    that GRPOTrainer may provide), then looks up the gold gloss.
    Falls back to empty string if not found.

    Args:
        prompt: The prompt from GRPOTrainer (any format).

    Returns:
        The gold gloss string, or ``""`` if not found.
    """
    sample_id = _extract_sample_id(prompt)
    if not sample_id:
        return ""
    return _gold_gloss_registry.get(sample_id, "")


def _lookup_gold_gloss_by_id(sample_id: str) -> str:
    """Look up the gold gloss directly by its stable sample ID.

    Unlike ``_lookup_gold_gloss``, this does NOT re-hash the input —
    it performs a direct dictionary lookup.  Use this when you already
    have a pre-computed sample ID (SHA256 hex string).

    Args:
        sample_id: The stable sample ID (SHA256 hex digest).

    Returns:
        The gold gloss string, or ``""`` if not found.
    """
    return _gold_gloss_registry.get(sample_id, "")


# ---------------------------------------------------------------------------
# Reward range helpers
# ---------------------------------------------------------------------------


def _to_symmetric(score: float) -> float:
    """Map a score from ``[0, 1]`` to ``[-1, 1]``.

    ``0 → -1``, ``0.5 → 0``, ``1 → 1``.
    """
    return 2.0 * score - 1.0


def _clamp_symmetric(score: float) -> float:
    """Clamp a raw (possibly unbounded) score to ``[-1, 1]``."""
    return max(-1.0, min(1.0, score))


# ---------------------------------------------------------------------------
# Reward Component 1: Translation Quality (ROUGE-L)
# ---------------------------------------------------------------------------


def translation_quality_reward(
    completion: str,
    gold_gloss: str,
) -> float:
    """Evaluate translation quality via ROUGE-L F1 score.

    Measures how similar the generated gloss sequence is to the gold
    reference.  This is the primary semantic signal for GRPO.

    Args:
        completion: Generated gloss sequence (model output).
        gold_gloss: Ground-truth gloss sequence.

    Returns:
        ROUGE-L F1 score mapped to ``[-1, 1]`` (symmetric range).
        ``-1`` = no overlap, ``1`` = perfect match.
    """
    if _ROUGE_SCORER is None:
        logger.warning("ROUGE scorer not initialized; returning -1.0")
        return -1.0

    generated = extract_gloss_text(completion)
    gold = gold_gloss.strip()

    if not generated:
        return -1.0
    if not gold:
        return -1.0

    scores = _ROUGE_SCORER.score(gold, generated)
    return _to_symmetric(scores["rougeL"].fmeasure)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Reward Component 2: Structural Dense Reward (Viterbi Proxy)
# ---------------------------------------------------------------------------


def structural_dense_reward(
    completion: str,
    normalize: bool | str = True,
    temperature: float = 1.0,
) -> float:
    """Score a gloss sequence using the precomputed bigram transition matrix.

    This is the **Viterbi proxy**: a dense reward that measures how
    "structurally plausible" a generated gloss sequence is, based on
    N‑gram probabilities observed in real ASL data.

    The reward is the average log‑probability of bigram transitions
    in the sequence, mapped to ``[-1, 1]`` (symmetric range).

    Three normalization modes are supported:

    - ``normalize="exp"`` (default, backward-compatible with ``True``):
      :math:`\\exp(\\text{avg\\_log\\_prob} / T)`, then mapped to ``[-1, 1]``.
      With temperature :math:`T=1` this is the classic formulation.
      **Problem**: for sequences with low bigram probability,
      :math:`\\text{avg\\_log\\_prob} \\approx -15`, so
      :math:`e^{-15} \\approx 0`, making the reward vanish.

    - ``normalize="softmax"``:
      :math:`\\frac{\\exp(\\text{avg\\_log\\_prob} / T)}
                 {1 + \\exp(\\text{avg\\_log\\_prob} / T)}`,
      then mapped to ``[-1, 1]``.
      This is a sigmoid-softened version that maps any log-prob to
      :math:`(0, 1)` without collapsing to 0 for low-probability sequences.

    - ``normalize=False``:
      Return the raw average log‑probability (can be negative),
      clamped to ``[-1, 1]``.

    .. math::

        \\text{reward} = \\exp\\left(
            \\frac{1}{L-1} \\sum_{i=1}^{L-1}
            \\log P(\\text{gloss}_i \\mid \\text{gloss}_{i-1})
        \\right)

    where :math:`L` is the sequence length and :math:`P` comes from
    the Laplace‑smoothed bigram matrix.

    Args:
        completion: Generated gloss sequence.
        normalize: Normalization mode. ``True`` or ``"exp"`` for classic
            exponentiation, ``"softmax"`` for sigmoid-softened, ``False``
            for raw log-prob.
        temperature: Temperature for the exponent. Higher values (e.g. 5)
            make the reward less aggressive: :math:`e^{-15/5} = e^{-3} \\approx 0.05`
            instead of :math:`e^{-15} \\approx 0`.

    Returns:
        Structural plausibility score in ``[-1, 1]`` (symmetric).
        ``-1`` = worst structural quality, ``1`` = best structural quality.
    """
    if _bigram_matrix is None or not _gloss_vocab:
        logger.warning("Transition matrix not initialized; returning -1.0")
        return -1.0

    text = extract_gloss_text(completion)
    tokens = text.strip().split()

    if len(tokens) < 2:
        return -1.0

    # Wrap with BOS and EOS if they exist in the vocab
    bos_idx = _token_to_idx.get("<BOS>", -1)
    eos_idx = _token_to_idx.get("<EOS>", -1)

    token_indices: list[int] = []
    if bos_idx >= 0:
        token_indices.append(bos_idx)

    for token in tokens:
        idx = _token_to_idx.get(token, _token_to_idx.get("<UNK>", 0))
        token_indices.append(idx)

    if eos_idx >= 0:
        token_indices.append(eos_idx)

    # Compute average log-probability
    log_sum: float = 0.0
    count: int = 0
    small_eps = 1e-10

    for i in range(len(token_indices) - 1):
        p = max(_bigram_matrix[token_indices[i], token_indices[i + 1]], small_eps)
        log_sum += np.log(p)
        count += 1

    if count == 0:
        return -1.0

    avg_log_prob = log_sum / count

    # Normalize and map to [-1, 1]
    if normalize is False or normalize is None:
        return _clamp_symmetric(avg_log_prob)

    # Temperature-scaled exponentiation
    scaled = avg_log_prob / max(temperature, 1e-8)

    if normalize == "softmax":
        # Sigmoid: maps any log-prob to (0, 1) without collapsing to 0
        # sigmoid(x) = exp(x) / (1 + exp(x))
        # For x = -15/5 = -3 → sigmoid(-3) ≈ 0.047 (not 0!)
        # For x = 0 → sigmoid(0) = 0.5
        # For x = 2 → sigmoid(2) ≈ 0.88
        return _to_symmetric(float(1.0 / (1.0 + np.exp(-scaled))))

    # Default: exp (backward-compatible with normalize=True)
    # e^0 = 1.0 (max), e^{-10} ≈ 0 (min)
    return _to_symmetric(float(np.exp(scaled)))


# ---------------------------------------------------------------------------
# Reward Component 3: Gold-Structure Reward (Gold-Baseline)
# ---------------------------------------------------------------------------


def gold_structure_reward(
    completion: str,
    gold_gloss: str,
    normalize: bool = True,
) -> float:
    """Structural reward using the gold reference gloss as baseline.

    Compares the generated gloss sequence's bigram log-probability against
    the gold reference's bigram log-probability.  This rewards the LLM for
    producing sequences whose structural plausibility (under the bigram
    model) is at least as good as the human-authored gold gloss.

    .. math::

        \\text{reward} = \\exp\\left(
            \\frac{\\text{llm_log_prob} - \\text{gold_log_prob}}{L}
        \\right)

    where :math:`L` is the number of bigram transitions.

    - ``≈ 1.0`` → LLM sequence is structurally as good as (or better than)
      the gold reference (mapped to ``≈ 1.0`` in ``[-1, 1]``).
    - ``≪ 1.0`` → LLM sequence has much worse bigram transitions than the
      gold reference (mapped toward ``-1``).

    .. note::
       This is the **recommended** structural reward for T2G GRPO.  It
       uses a semantically meaningful baseline (the gold gloss) rather
       than the degenerate Viterbi optimum or an absolute score.

    Args:
        completion: Generated gloss sequence.
        gold_gloss: Ground-truth gold gloss sequence.
        normalize: If ``True``, exponentiate and cap at ``1.0``, then map
            to ``[-1, 1]``.  If ``False``, return raw log-prob difference
            clamped to ``[-1, 1]``.

    Returns:
        Structural proximity reward in ``[-1, 1]`` (symmetric).
    """
    if _bigram_matrix is None or not _gloss_vocab:
        logger.warning("Transition matrix not initialized; returning -1.0")
        return -1.0

    llm_text = extract_gloss_text(completion)
    gold_text = gold_gloss.strip()

    if not llm_text or not gold_text:
        return -1.0

    # Map tokens to indices for both sequences
    bos_idx = _token_to_idx.get("<BOS>", -1)
    eos_idx = _token_to_idx.get("<EOS>", -1)

    def _indices(tokens: list[str]) -> tuple[list[int], int]:
        """Map tokens to indices. Returns (indices, oov_count).

        OOV tokens are skipped (not mapped to <UNK>) so that garbage
        tokens don't get partial credit via <UNK> bigram probabilities.
        The oov_count is used to penalize the reward proportionally.
        """
        indices: list[int] = []
        oov_count = 0
        if bos_idx >= 0:
            indices.append(bos_idx)
        for t in tokens:
            idx = _token_to_idx.get(t, -1)
            if idx >= 0:
                indices.append(idx)
            else:
                oov_count += 1
        if eos_idx >= 0:
            indices.append(eos_idx)
        return indices, oov_count

    llm_indices, llm_oov = _indices(llm_text.split())
    gold_indices, gold_oov = _indices(gold_text.split())

    # Compute log-probabilities
    from src.datasets.transition_matrix import sequence_score_bigram

    llm_log_prob = sequence_score_bigram(_bigram_matrix, llm_indices)
    gold_log_prob = sequence_score_bigram(_bigram_matrix, gold_indices)

    # Number of transitions in the LLM path
    n_trans = len(llm_indices) - 1
    if n_trans <= 0:
        return -1.0
    n_gold_trans = len(gold_indices) - 1
    if n_gold_trans <= 0:
        return -1.0

    if normalize:
        # Compare average log-probs
        llm_avg = llm_log_prob / n_trans
        gold_avg = gold_log_prob / n_gold_trans
        reward = float(np.exp(llm_avg - gold_avg))
        # Cap at 1.0 (at or above gold structural quality)
        reward = min(reward, 1.0)
        # Penalize OOV tokens: each OOV token reduces the reward
        # proportionally, so garbage tokens don't get free credit.
        total_tokens = len(llm_text.split())
        if total_tokens > 0:
            oov_penalty = llm_oov / total_tokens
            reward *= 1.0 - oov_penalty
        return _to_symmetric(reward)

    return _clamp_symmetric(llm_log_prob - gold_log_prob)


# ---------------------------------------------------------------------------
# Reward Component: Gloss-Order Edit-Distance Reward
# ---------------------------------------------------------------------------


def _word_level_levenshtein(a: list[str], b: list[str]) -> int:
    """Compute word-level Levenshtein (edit) distance between two token lists.

    Standard O(len(a) * len(b)) dynamic-programming implementation,
    operating on whole gloss tokens rather than characters — appropriate
    since ASL gloss order is a sequence-of-symbols problem, not a
    character-similarity problem.

    Args:
        a: First token sequence.
        b: Second token sequence.

    Returns:
        The minimum number of token insertions/deletions/substitutions
        needed to transform ``a`` into ``b``.
    """
    n, m = len(a), len(b)
    if n == 0:
        return m
    if m == 0:
        return n

    # Single-row DP to keep this cheap (glosses are short sequences).
    prev_row = list(range(m + 1))
    for i in range(1, n + 1):
        curr_row = [i] + [0] * m
        for j in range(1, m + 1):
            cost_sub = prev_row[j - 1] + (0 if a[i - 1] == b[j - 1] else 1)
            cost_del = prev_row[j] + 1
            cost_ins = curr_row[j - 1] + 1
            curr_row[j] = min(cost_sub, cost_del, cost_ins)
        prev_row = curr_row

    return prev_row[m]


def gloss_order_reward(
    completion: str,
    gold_gloss: str,
) -> float:
    """Reward the correct **ordering** of glosses via normalized edit-distance.

    ``translation_quality_reward`` (ROUGE-L) is a lexical-overlap proxy
    designed for natural-language summarization and is comparatively weak
    at penalizing wrong ordering of a short, highly-structured symbol
    sequence like ASL gloss (see docs/T2G_PIPELINE_REVIEW.md §5.3).  This
    reward instead computes the **word-level Levenshtein distance**
    between the generated and gold gloss sequences, normalized by the
    length of the longer sequence, so that gloss transpositions/insertions/
    deletions are penalized in a way that is sensitive to sequence order —
    independent from (and complementary to) the bigram-based structural
    rewards, which only look at local transition plausibility, not
    similarity to the actual gold ordering.

    .. math::

        \\text{reward} = 1 - \\frac{\\text{edit\\_distance}(a, b)}{\\max(|a|, |b|)}

    - ``1.0`` → identical gloss sequence (order and content match exactly).
    - ``-1.0`` → completely different sequence (no overlap after edits).

    Args:
        completion: Generated gloss sequence (model output).
        gold_gloss: Ground-truth gloss sequence.

    Returns:
        Normalized similarity in ``[-1, 1]`` (symmetric); ``-1.0`` if
        either sequence is empty.
    """
    generated = extract_gloss_text(completion).strip()
    gold = gold_gloss.strip()

    if not generated or not gold:
        return -1.0

    gen_tokens = generated.split()
    gold_tokens = gold.split()

    if not gen_tokens or not gold_tokens:
        return -1.0

    distance = _word_level_levenshtein(gen_tokens, gold_tokens)
    max_len = max(len(gen_tokens), len(gold_tokens))

    return _to_symmetric(float(max(0.0, 1.0 - distance / max_len)))


# ---------------------------------------------------------------------------
# Reward Component 4: Viterbi Distance Reward
# ---------------------------------------------------------------------------


def viterbi_distance_reward(
    completion: str,
    normalize: bool = True,
) -> float:
    """Reward based on distance from the Viterbi-optimal path.

    Computes the globally most-probable (Viterbi) path of the **same length**
    as the LLM-generated sequence through the bigram transition matrix,
    constrained to start at ``<BOS>`` and end at ``<EOS>``.  Returns a
    normalized score indicating how close the LLM's path is to this
    theoretical upper bound.

    .. math::

        \\text{reward} = \\exp\\left(
            \\frac{\\text{llm_log_prob} - \\text{viterbi_log_prob}}{L}
        \\right)

    where :math:`L` is the number of bigram transitions.

    - ``1.0`` → LLM path matches the Viterbi optimum exactly.
    - ``-1.0`` → LLM path is far from the Viterbi optimum.

    .. warning::
       **Path diversity**: This function now uses ``viterbi_optimal_score_diverse``
       (with self-loop penalty and iterative token banning) to compute
       the Viterbi baseline.  The pure Markov-chain Viterbi would
       degenerate into repetitive loops (e.g., ``IX → IX → …``).
       The diversity-constrained path provides a more realistic upper
       bound for ASL gloss sequences.

    Args:
        completion: Generated gloss sequence.
        normalize: If ``True``, exponentiate to ``(0, 1]`` range then map
            to ``[-1, 1]``.  If ``False``, return raw average-log-prob
            difference clamped to ``[-1, 1]``.

    Returns:
        Viterbi proximity reward in ``[-1, 1]`` (symmetric).
    """
    if _bigram_matrix is None or not _gloss_vocab:
        logger.warning("Transition matrix not initialized; returning -1.0")
        return -1.0

    text = extract_gloss_text(completion)
    tokens = text.strip().split()

    if len(tokens) < 2:
        return -1.0

    bos_idx = _token_to_idx.get("<BOS>", -1)
    eos_idx = _token_to_idx.get("<EOS>", -1)

    if bos_idx < 0 or eos_idx < 0:
        logger.warning("BOS/EOS not in vocabulary; returning -1.0")
        return -1.0

    # Build LLM path indices (BOS + tokens + EOS)
    llm_indices: list[int] = [bos_idx]
    for token in tokens:
        idx = _token_to_idx.get(token, _token_to_idx.get("<UNK>", 0))
        llm_indices.append(idx)
    llm_indices.append(eos_idx)

    path_length = len(llm_indices)  # includes BOS and EOS

    # Compute Viterbi optimal log-probability for the same length
    # with diversity constraints (self-loop penalty + iterative token ban)
    from src.datasets.transition_matrix import (
        sequence_score_bigram,
        viterbi_optimal_score_diverse,
    )

    llm_log_prob = sequence_score_bigram(_bigram_matrix, llm_indices)

    viterbi_log_prob = viterbi_optimal_score_diverse(
        _bigram_matrix,
        bos_idx,
        eos_idx,
        path_length,
        self_loop_penalty=float(
            _viterbi_diversity_params.get("self_loop_penalty", 0.5)
        ),
        max_occurrences=int(_viterbi_diversity_params.get("max_occurrences", 2)),
        diversity_threshold=float(
            _viterbi_diversity_params.get("diversity_threshold", 0.3)
        ),
        max_iters=int(_viterbi_diversity_params.get("max_iters", 3)),
    )

    n_trans = path_length - 1  # number of transitions
    if n_trans <= 0:
        return -1.0

    if normalize:
        llm_avg = llm_log_prob / n_trans
        viterbi_avg = viterbi_log_prob / n_trans
        return _to_symmetric(float(np.exp(llm_avg - viterbi_avg)))

    return _clamp_symmetric((llm_log_prob - viterbi_log_prob) / n_trans)


# ---------------------------------------------------------------------------
# Reward Component 4b: Soft Viterbi Distance Reward (Differentiable)
# ---------------------------------------------------------------------------


def soft_viterbi_distance_reward(
    completion: str,
    normalize: bool = True,
) -> float:
    """Differentiable reward based on soft Viterbi (forward-backward) distance.

    Inspired by ViterbiPlanNet's Differentiable Viterbi Layer (DVL)
    (arXiv:2603.04265), this replaces the non-differentiable argmax
    Viterbi with a smooth log-sum-exp relaxation (forward-backward).

    The soft Viterbi score is the **log-partition function** — the
    log-probability of *all* paths of the given length, weighted by
    their probability.  This provides a smoother and tighter upper bound
    than the hard Viterbi (which only considers the single best path),
    and allows gradient flow through the structural reward.

    .. math::

        \\text{reward} = \\exp\\left(
            \\frac{\\text{llm\\_log\\_prob} - \\text{soft\\_viterbi\\_log\\_prob}}{L}
        \\right)

    where :math:`\\text{soft\\_viterbi\\_log\\_prob} = \\log Z` is the
    log-partition function computed via forward-backward.

    - ``1.0`` → LLM path matches the soft Viterbi optimum.
    - ``-1.0`` → LLM path is far from the soft optimum.

    .. note::
       The soft Viterbi score is always >= the hard Viterbi score
       (logsumexp >= max), so this reward is generally lower than
       ``viterbi_distance_reward`` for the same sequence.  This is
       expected — the soft bound is tighter.

    Args:
        completion: Generated gloss sequence.
        normalize: If ``True``, exponentiate to ``(0, 1]`` range then map
            to ``[-1, 1]``.  If ``False``, return raw average-log-prob
            difference clamped to ``[-1, 1]``.

    Returns:
        Soft Viterbi proximity reward in ``[-1, 1]`` (symmetric).
    """
    if _bigram_matrix is None or not _gloss_vocab:
        logger.warning("Transition matrix not initialized; returning -1.0")
        return -1.0

    text = extract_gloss_text(completion)
    tokens = text.strip().split()

    if len(tokens) < 2:
        return -1.0

    bos_idx = _token_to_idx.get("<BOS>", -1)
    eos_idx = _token_to_idx.get("<EOS>", -1)

    if bos_idx < 0 or eos_idx < 0:
        logger.warning("BOS/EOS not in vocabulary; returning -1.0")
        return -1.0

    # Build LLM path indices (BOS + tokens + EOS)
    llm_indices: list[int] = [bos_idx]
    for token in tokens:
        idx = _token_to_idx.get(token, _token_to_idx.get("<UNK>", 0))
        llm_indices.append(idx)
    llm_indices.append(eos_idx)

    path_length = len(llm_indices)

    from src.datasets.transition_matrix import (
        sequence_score_bigram,
        soft_viterbi_score,
    )

    llm_log_prob = sequence_score_bigram(_bigram_matrix, llm_indices)
    soft_viterbi_log_prob = soft_viterbi_score(
        _bigram_matrix,
        bos_idx,
        eos_idx,
        path_length,
    )

    n_trans = path_length - 1
    if n_trans <= 0:
        return -1.0

    if normalize:
        llm_avg = llm_log_prob / n_trans
        soft_viterbi_avg = soft_viterbi_log_prob / n_trans
        return _to_symmetric(float(np.exp(llm_avg - soft_viterbi_avg)))

    return _clamp_symmetric((llm_log_prob - soft_viterbi_log_prob) / n_trans)


# ---------------------------------------------------------------------------
# Reward Component 8: Verifier-Scaled Reward (RECIPE-inspired)
# ---------------------------------------------------------------------------


def verifier_scaled_reward(
    completion: str,
    gold_gloss: str,
) -> float:
    """RECIPE-inspired verifier-scaled translation reward.

    Inspired by RECIPE (arXiv:2605.19976): *"extracting clean step labels
    from noisy video is hard, but verifying whether a generated step
    sequence is temporally grounded is cheap and scales to millions of
    videos"*.

    This function implements the verifier principle: instead of using
    the structural quality (bigram plausibility) as a standalone reward,
    it uses it as a **confidence multiplier** for the translation quality
    (ROUGE-L).  This means:

    - High ROUGE-L + high structural plausibility → high reward (confident match)
    - High ROUGE-L + low structural plausibility → reduced reward (suspicious match)
    - Low ROUGE-L + high structural plausibility → low reward (wrong but plausible)
    - Low ROUGE-L + low structural plausibility → very low reward (wrong and implausible)

    .. math::

        \\text{reward} = \\text{ROUGE-L} \\times \\text{verifier\\_confidence}

    where :math:`\\text{verifier\\_confidence} \\in [0, 1]` is the
    structural plausibility (normalized bigram score) of the generated
    sequence.  The final reward is mapped to ``[-1, 1]`` (symmetric).

    This is more informative than either reward alone: it penalizes
    sequences that happen to match the gold lexically but are structurally
    implausible (e.g., correct tokens in wrong order with implausible
    transitions), and vice versa.

    Args:
        completion: Generated gloss sequence.
        gold_gloss: Ground-truth gloss sequence.

    Returns:
        Verifier-scaled reward in ``[-1, 1]`` (symmetric).
    """
    rouge = translation_quality_reward(completion, gold_gloss)

    # Use gold_structure_reward (which compares the bigram log-probability
    # of the completion against the gold reference as a baseline and caps at 1.0)
    # as the verifier confidence multiplier.
    verifier_confidence = gold_structure_reward(completion, gold_gloss, normalize=True)

    # Both sub-rewards now return [-1, 1].  Convert back to [0, 1] for the
    # multiplicative verifier formula, then map the product to [-1, 1].
    rouge_01 = (rouge + 1.0) / 2.0
    confidence_01 = (verifier_confidence + 1.0) / 2.0

    return _to_symmetric(rouge_01 * confidence_01)


# ---------------------------------------------------------------------------
# Format reward: ensure gloss-only output
# ---------------------------------------------------------------------------


def gloss_format_reward(completion: str) -> float:
    """Reward for generating only valid gloss tokens from the vocabulary.

    Validates each whitespace-separated token in the completion against the
    actual gloss vocabulary (``_gloss_vocab``), rather than using generic
    regex patterns that conflict with valid ASL gloss tokens (e.g. ``.``,
    ``BE``, ``FOR``, ``TO`` are all legitimate glosses).

    Scoring:
    - ``1.0`` — all tokens are in the vocabulary.
    - ``0.0`` — mixed: some tokens valid, some not.
    - ``-0.5`` — mostly garbage (>50% tokens out-of-vocab).
    - ``-1.0`` — empty output or all tokens out-of-vocab.

    All scores are in the symmetric ``[-1, 1]`` range via
    ``_to_symmetric`` mapping of the original ``[0, 1]`` levels.

    Also penalizes concatenated subword garbage (tokens >25 chars) and
    severe numeric contamination (3+ consecutive digits).

    Args:
        completion: Raw model completion.

    Returns:
        Format reward in ``[-1, 1]`` (symmetric).
    """
    text = extract_gloss_text(completion)
    if not text:
        return -1.0

    # Strip code blocks / JSON-like wrappers (residual from extract_gloss_text)
    if "```" in text or "{" in text or "}" in text:
        text = re.sub(r"```.*?```", "", text, flags=re.DOTALL)
        text = re.sub(r"[{}]", "", text)

    tokens = text.split()
    if not tokens:
        return -1.0

    # ── Vocabulary membership check ───────────────────────────────
    # This is the primary signal: each token must be a valid gloss.
    vocab_set = set(_gloss_vocab) if _gloss_vocab else None

    if vocab_set is not None:
        valid_count = sum(1 for t in tokens if t in vocab_set)
        valid_ratio = valid_count / len(tokens)

        if valid_ratio == 1.0:
            # All tokens are valid glosses — check for garbage concatenation
            long_token_count = sum(1 for t in tokens if len(t) > 25)
            if long_token_count > 0:
                return _to_symmetric(
                    0.5
                )  # Suspicious: valid but abnormally long tokens
            return 1.0
        elif valid_ratio >= 0.5:
            return _to_symmetric(0.5)  # Mixed: some valid, some not
        elif valid_ratio > 0.0:
            return _to_symmetric(0.25)  # Mostly garbage
        else:
            return -1.0  # All out-of-vocab
    else:
        # Fallback: vocabulary not initialized — use heuristic checks
        # (kept for safety, but should not happen in normal training)
        digit_sequences = re.findall(r"\d{3,}", text)
        if digit_sequences:
            total_digit_chars = sum(len(s) for s in digit_sequences)
            if total_digit_chars > 20:
                return -1.0
            return _to_symmetric(0.25)

        long_token_count = sum(1 for t in tokens if len(t) > 25)
        if long_token_count > 0:
            return _to_symmetric(0.5)

        return 1.0


# ---------------------------------------------------------------------------
# Repetition reward: penalize degenerate loops
# ---------------------------------------------------------------------------


def gloss_repetition_reward(completion: str) -> float:
    """Penalize repetitive gloss sequences (degenerate generation).

    Args:
        completion: Raw model completion.

    Returns:
        ``1.0`` for normal output, ``0.0`` for moderate repetition,
        ``-1.0`` for severe loops.
    """
    text = extract_gloss_text(completion)
    if not text:
        return 1.0

    tokens = text.split()
    if len(tokens) < 4:
        return 1.0

    # Check token-level uniqueness
    unique_ratio = len(set(tokens)) / len(tokens)

    # Check trigram uniqueness
    trigrams = [tuple(tokens[i : i + 3]) for i in range(len(tokens) - 2)]
    trigram_unique_ratio = len(set(trigrams)) / max(len(trigrams), 1)

    ratio = min(unique_ratio, trigram_unique_ratio)

    if ratio > 0.5:
        return 1.0
    if ratio > 0.3:
        return -0.3  # moderate repetition → mild negative
    return -1.0  # severe loops → full penalty


# ---------------------------------------------------------------------------
# BLEU-4 Reward: n-gram precision with sacrebleu
# ---------------------------------------------------------------------------

#: Module-level cache for sacrebleu availability check.
#  None  = not yet checked
#  True  = sacrebleu imported successfully
#  False = import failed (do NOT retry — see _get_sacrebleu_metric which
#          raises ImportError loudly instead of silently caching -1.0)
_SACREBLEU_AVAILABLE: bool | None = None

#: Reusable BLEU metric instance (configured once at first use).
#  effective_order=True lets BLEU score sequences shorter than 4 tokens
#  (BLEU-4 normally requires 4-grams → returns 0 → maps to -1.0 for every
#  short sequence, killing the gradient signal on common short glosses).
#  smooth_method="floor" prevents the geometric mean from collapsing to
#  exactly 0 when one n-gram order has zero matches, giving a smoother
#  gradient for near-miss completions.
_SACREBLEU_METRIC: Any = None


def _check_sacrebleu_available() -> None:
    """Verify sacrebleu is importable; raise ImportError with actionable message.

    Called eagerly from ``build_t2g_reward_functions`` when
    ``weight_bleu > 0`` so a missing dependency crashes training at config
    time — before any reward is computed — with a clear message, rather than
    silently returning -1.0 for every sample during the entire run (which
    previously left 20% of the reward signal dead with no visible warning
    in output.log, since the logger.warning went to stderr, not the tee'd
    stdout).

    Raises:
        ImportError: If sacrebleu is not installed.
    """
    try:
        import sacrebleu  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "sacrebleu is not installed but weight_bleu > 0 in the config. "
            "BLEU reward requires sacrebleu. Install it with: pip install sacrebleu "
            "(or uv pip install sacrebleu). On the cluster, ensure the Apptainer "
            "image includes it — see cluster/setup.sh (pip install --user -e . "
            "from pyproject.toml which declares sacrebleu>=2.0.0)."
        ) from e


def _get_sacrebleu_metric() -> Any:
    """Lazily import sacrebleu and build a reusable BLEU metric.

    Called on the first ``bleu_reward`` invocation.  Should never raise
    ImportError in practice because ``_check_sacrebleu_available`` is
    called eagerly at config time when ``weight_bleu > 0``.

    Returns:
        A configured ``sacrebleu.BLEU`` instance.

    Raises:
        ImportError: If sacrebleu is not installed (caller bypassed init check).
    """
    global _SACREBLEU_AVAILABLE, _SACREBLEU_METRIC

    if _SACREBLEU_METRIC is not None:
        return _SACREBLEU_METRIC

    try:
        import sacrebleu
    except ImportError as e:
        _SACREBLEU_AVAILABLE = False
        raise ImportError(
            "sacrebleu is not installed but weight_bleu > 0 in the config. "
            "BLEU reward requires sacrebleu. Install it with: pip install sacrebleu "
            "(or uv pip install sacrebleu). On the cluster, ensure the Apptainer "
            "image includes it — see cluster/setup.sh (pip install --user -e . "
            "from pyproject.toml which declares sacrebleu>=2.0.0)."
        ) from e

    _SACREBLEU_AVAILABLE = True
    _SACREBLEU_METRIC = sacrebleu.BLEU(
        effective_order=True,
        smooth_method="floor",
        smooth_value=0.1,
    )
    return _SACREBLEU_METRIC


def bleu_reward(completion: str, gold_gloss: str) -> float:
    """BLEU-4 reward using sacrebleu sentence BLEU.

    T2G-Reasoner (2025) shows BLEU-4 outperforms ROUGE-L as a reward
    signal for T2G GRPO training.

    Uses ``effective_order=True`` so short gloss sequences (1–3 tokens,
    common in ASL: ``"IX-1p"``, ``"WALK HOUSE"``) are scored against the
    available n-gram orders instead of being forced to BLEU-4 (which
    requires 4-grams and would return 0 → mapped to -1.0 for every short
    sequence, killing the gradient signal).

    A small ``floor`` smoothing (0.1) prevents the geometric mean from
    collapsing to exactly 0 when one n-gram order has zero matches,
    giving a smoother gradient for near-miss completions.

    Args:
        completion: Generated gloss sequence (model output).
        gold_gloss: Ground-truth gold gloss sequence.

    Returns:
        BLEU-4 score mapped to ``[-1, 1]`` (symmetric).
        ``-1`` = no overlap, ``1`` = perfect match.
    """
    generated = extract_gloss_text(completion)
    gold = gold_gloss.strip()

    if not generated or not gold:
        return -1.0

    try:
        metric = _get_sacrebleu_metric()
        # sentence_score returns BLEUScore with .score in [0, 100]
        bleu_score = metric.sentence_score(generated, [gold]).score
        # Normalize to [0, 1] then map to [-1, 1]
        return _to_symmetric(float(bleu_score) / 100.0)
    except ImportError:
        # Should never reach here — _check_sacrebleu_available() is called
        # eagerly in build_t2g_reward_functions() when weight_bleu > 0, so a
        # missing sacrebleu crashes training at config time with a clear
        # message BEFORE any reward is computed.  If we reach here, the caller
        # bypassed the init check — re-raise to surface the misconfiguration.
        raise
    except Exception:
        logger.warning("BLEU computation failed; returning -1.0", exc_info=True)
        return -1.0


# ---------------------------------------------------------------------------
# GRPOTrainer-compatible wrappers
# ---------------------------------------------------------------------------


def _make_gloss_reward_fn(
    component_fn: Callable[..., float],
    needs_gold_gloss: bool = False,
) -> Callable[..., list[float]]:
    """Wrap a single-sample reward component for GRPOTrainer.

    The GRPOTrainer expects:
        ``fn(completions: list[str], prompts: list[str], **kwargs) -> list[float]``

    For ``needs_gold_gloss=True``, the gold gloss is retrieved from the
    global ``_gold_gloss_registry`` by extracting a stable sample ID
    (SHA256 of user instruction) from the prompt, regardless of format.

    Args:
        component_fn: A function taking a single completion (and optionally
            gold gloss text) and returning a float.
        needs_gold_gloss: If ``True``, the function also receives the gold
            gloss target looked up from the registry.

    Returns:
        A callable with the GRPOTrainer-compatible signature.
    """

    def reward_fn(
        completions: list[Any],
        prompts: list[Any] | None = None,
        **kwargs: Any,
    ) -> list[float]:
        results: list[float] = []

        for idx, completion in enumerate(completions):
            # Handle GRPOTrainer's completion format: list of messages
            text: str = (
                completion[0]["content"]
                if isinstance(completion, list)
                else str(completion)
            )

            if needs_gold_gloss:
                # Look up gold gloss via stable sample ID (format-agnostic)
                prompt = prompts[idx] if prompts and idx < len(prompts) else None
                gold = _lookup_gold_gloss(prompt)
                results.append(component_fn(text, gold))
            else:
                results.append(component_fn(text))

        return results

    reward_fn.__name__ = component_fn.__name__  # for wandb metric naming
    return reward_fn


def build_t2g_reward_functions(
    reward_config: dict[str, float] | None = None,
) -> tuple[list[Callable[..., list[float]]], list[float]]:
    """Build the list of reward functions and weights for T2G GRPO.

    Supported weight keys:

    - ``weight_translation``: ROUGE-L similarity with gold gloss.
    - ``weight_bleu``: BLEU-4 score via sacrebleu with gold gloss.
    - ``weight_structure``: Absolute bigram log-prob reward (no baseline).
    - ``weight_gold_structure``: Bigram score vs gold reference baseline
      **(recommended over weight_structure)**.
    - ``weight_viterbi``: Bigram score vs Viterbi theoretical optimum
      **(experimental — see caveat in ``viterbi_distance_reward``)**.
    - ``weight_soft_viterbi``: Bigram score vs **soft** Viterbi (forward-backward)
      optimum — differentiable relaxation inspired by ViterbiPlanNet's DVL
      (arXiv:2603.04265).  Smoother and tighter than ``weight_viterbi``.
    - ``weight_verifier_scaled``: RECIPE-inspired verifier-scaled reward
      (arXiv:2605.19976) — uses structural plausibility as a confidence
      multiplier for translation quality.  More informative than either
      reward alone.
    - ``weight_gloss_order``: Word-level edit-distance similarity with gold
      gloss — complements ``weight_translation`` (ROUGE-L, a lexical-overlap
      proxy borrowed from summarization) with a signal that is sensitive to
      long-range gloss **ordering**, which bigram-based structural rewards
      do not capture (see docs/T2G_PIPELINE_REVIEW.md §5.3).
    - ``weight_format``: Clean gloss-only format reward.
    - ``weight_repetition``: Repetition penalty.

    Args:
        reward_config: Dictionary with weight keys.  If ``None``, uses
            default weights (translation 0.40, gold-structure 0.40,
            format 0.10, repetition 0.10).

    Returns:
        Tuple of ``(reward_funcs, reward_weights)`` compatible with
        ``GRPOTrainer``.
    """
    if reward_config is None:
        reward_config = {
            "weight_translation": 0.40,
            "weight_gold_structure": 0.40,
            "weight_format": 0.10,
            "weight_repetition": 0.10,
        }

    funcs: list[Callable[..., list[float]]] = []
    weights: list[float] = []

    # Translation quality (needs gold gloss)
    w = reward_config.get("weight_translation", 0.0)
    if w > 0:
        funcs.append(
            _make_gloss_reward_fn(translation_quality_reward, needs_gold_gloss=True)
        )
        weights.append(w)

    # BLEU-4 reward (needs gold gloss)
    w = reward_config.get("weight_bleu", 0.0)
    if w > 0:
        # Eagerly verify sacrebleu is importable so a missing dependency
        # crashes here (before training starts) with a clear message,
        # rather than silently returning -1.0 for every sample during the
        # entire run — which previously left 20% of the reward signal dead
        # with no visible warning (the logger.warning went to stderr, not
        # the tee'd output.log, so it was invisible on the cluster).
        _check_sacrebleu_available()
        funcs.append(_make_gloss_reward_fn(bleu_reward, needs_gold_gloss=True))
        weights.append(w)

    # Structural dense reward (absolute bigram score — no baseline)
    w = reward_config.get("weight_structure", 0.0)
    if w > 0:
        funcs.append(_make_gloss_reward_fn(structural_dense_reward))
        weights.append(w)

    # Gold-structure reward (bigram score vs gold reference baseline)
    # *** Recommended over weight_structure for production ***
    w = reward_config.get("weight_gold_structure", 0.0)
    if w > 0:
        funcs.append(
            _make_gloss_reward_fn(gold_structure_reward, needs_gold_gloss=True)
        )
        weights.append(w)

    # Viterbi distance reward (bigram score vs Viterbi theoretical optimum)
    # *** Experimental — see caveat in viterbi_distance_reward docstring ***
    w = reward_config.get("weight_viterbi", 0.0)
    if w > 0:
        funcs.append(_make_gloss_reward_fn(viterbi_distance_reward))
        weights.append(w)

    # Soft Viterbi distance reward (differentiable, forward-backward)
    # *** Inspired by ViterbiPlanNet's DVL (arXiv:2603.04265) ***
    w = reward_config.get("weight_soft_viterbi", 0.0)
    if w > 0:
        funcs.append(_make_gloss_reward_fn(soft_viterbi_distance_reward))
        weights.append(w)

    # Verifier-scaled reward (RECIPE-inspired)
    # *** Uses structural plausibility as confidence multiplier (arXiv:2605.19976) ***
    w = reward_config.get("weight_verifier_scaled", 0.0)
    if w > 0:
        funcs.append(
            _make_gloss_reward_fn(verifier_scaled_reward, needs_gold_gloss=True)
        )
        weights.append(w)

    # Gloss-order edit-distance reward (needs gold gloss) — complements
    # ROUGE-L with an ordering-sensitive signal (see docs §5.3).
    w = reward_config.get("weight_gloss_order", 0.0)
    if w > 0:
        funcs.append(_make_gloss_reward_fn(gloss_order_reward, needs_gold_gloss=True))
        weights.append(w)

    # Format reward
    w = reward_config.get("weight_format", 0.0)
    if w > 0:
        funcs.append(_make_gloss_reward_fn(gloss_format_reward))
        weights.append(w)

    # Repetition penalty
    w = reward_config.get("weight_repetition", 0.0)
    if w > 0:
        funcs.append(_make_gloss_reward_fn(gloss_repetition_reward))
        weights.append(w)

    names = [f.__name__ for f in funcs]
    weight_strs = [f"{w:.2f}" for w in weights]
    logger.info(
        f"T2G Reward functions: {', '.join(f'{n}={w}' for n, w in zip(names, weight_strs))}"
    )

    return funcs, weights
