"""
Reward Functions for T2G GRPO Training.

Six reward components:

1. **Translation Quality Reward** (ROUGE-L):
   Lexical similarity between generated gloss and gold reference.

2. **Structural Dense Reward** (Bigram Log-Probability):
   Average log-probability of bigram transitions (absolute score).

3. **Gold-Structure Reward** (Gold-Baseline Structural) ⭐:
   Compares LLM bigram score against the gold reference gloss.

4. **Viterbi Distance Reward** (Viterbi-Upper-Bound) 🧪:
   Compares LLM path against the diverse Viterbi optimum.

5. **Format Reward**: Penalizes free text / non-gloss outputs.

6. **Repetition Reward**: Penalizes degenerate token repetition.

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

#: Viterbi diversity parameters loaded from config YAML.
#  Configured via ``grammar.viterbi_diversity`` section.
_viterbi_diversity_params: dict[str, float | int] = {
    "self_loop_penalty": 0.5,
    "max_occurrences": 2,
    "diversity_threshold": 0.3,
    "max_iters": 3,
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
        ROUGE-L F1 score in ``[0, 1]``.
    """
    if _ROUGE_SCORER is None:
        logger.warning("ROUGE scorer not initialized; returning 0.0")
        return 0.0

    generated = extract_gloss_text(completion)
    gold = gold_gloss.strip()

    if not generated:
        return 0.0
    if not gold:
        return 0.0

    scores = _ROUGE_SCORER.score(gold, generated)
    return scores["rougeL"].fmeasure  # type: ignore[no-any-return]


# ---------------------------------------------------------------------------
# Reward Component 2: Structural Dense Reward (Viterbi Proxy)
# ---------------------------------------------------------------------------


def structural_dense_reward(
    completion: str,
    normalize: bool = True,
) -> float:
    """Score a gloss sequence using the precomputed bigram transition matrix.

    This is the **Viterbi proxy**: a dense reward that measures how
    "structurally plausible" a generated gloss sequence is, based on
    N‑gram probabilities observed in real ASL data.

    The reward is the average log‑probability of bigram transitions
    in the sequence, normalized to ``[0, 1]`` via exponentiation.

    .. math::

        \\text{reward} = \\exp\\left(
            \\frac{1}{L-1} \\sum_{i=1}^{L-1}
            \\log P(\\text{gloss}_i \\mid \\text{gloss}_{i-1})
        \\right)

    where :math:`L` is the sequence length and :math:`P` comes from
    the Laplace‑smoothed bigram matrix.

    Args:
        completion: Generated gloss sequence.
        normalize: If ``True``, exponentiate the average log‑prob to
            produce a score in ``[0, 1]``.  If ``False``, return the
            raw average log‑probability (can be negative).

    Returns:
        Structural plausibility score.
    """
    if _bigram_matrix is None or not _gloss_vocab:
        logger.warning("Transition matrix not initialized; returning 0.0")
        return 0.0

    text = extract_gloss_text(completion)
    tokens = text.strip().split()

    if len(tokens) < 2:
        return 0.0

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
        return 0.0

    avg_log_prob = log_sum / count

    if normalize:
        # Exponentiate to get a [0, 1] score
        # e^0 = 1.0 (max), e^{-10} ≈ 0 (min)
        return float(np.exp(avg_log_prob))
    return avg_log_prob


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
      the gold reference.
    - ``≪ 1.0`` → LLM sequence has much worse bigram transitions than the
      gold reference.

    .. note::
       This is the **recommended** structural reward for T2G GRPO.  It
       uses a semantically meaningful baseline (the gold gloss) rather
       than the degenerate Viterbi optimum or an absolute score.

    Args:
        completion: Generated gloss sequence.
        gold_gloss: Ground-truth gold gloss sequence.
        normalize: If ``True``, exponentiate to ``(0, ∞)`` range capped
            at ``1.0``.  If ``False``, return raw log-prob difference.

    Returns:
        Structural proximity reward.
    """
    if _bigram_matrix is None or not _gloss_vocab:
        logger.warning("Transition matrix not initialized; returning 0.0")
        return 0.0

    llm_text = extract_gloss_text(completion)
    gold_text = gold_gloss.strip()

    if not llm_text or not gold_text:
        return 0.0

    # Map tokens to indices for both sequences
    bos_idx = _token_to_idx.get("<BOS>", -1)
    eos_idx = _token_to_idx.get("<EOS>", -1)
    unk_idx = _token_to_idx.get("<UNK>", 0)

    def _indices(tokens: list[str]) -> list[int]:
        indices: list[int] = []
        if bos_idx >= 0:
            indices.append(bos_idx)
        for t in tokens:
            indices.append(_token_to_idx.get(t, unk_idx))
        if eos_idx >= 0:
            indices.append(eos_idx)
        return indices

    llm_indices = _indices(llm_text.split())
    gold_indices = _indices(gold_text.split())

    # Compute log-probabilities
    from src.datasets.transition_matrix import sequence_score_bigram

    llm_log_prob = sequence_score_bigram(_bigram_matrix, llm_indices)
    gold_log_prob = sequence_score_bigram(_bigram_matrix, gold_indices)

    # Number of transitions in the LLM path
    n_trans = len(llm_indices) - 1
    if n_trans <= 0:
        return 0.0
    n_gold_trans = len(gold_indices) - 1
    if n_gold_trans <= 0:
        return 0.0

    if normalize:
        # Compare average log-probs
        llm_avg = llm_log_prob / n_trans
        gold_avg = gold_log_prob / n_gold_trans
        reward = float(np.exp(llm_avg - gold_avg))
        # Cap at 1.0 (at or above gold structural quality)
        return min(reward, 1.0)

    return llm_log_prob - gold_log_prob


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
    - ``≈ 0.0`` → LLM path is far from the Viterbi optimum.

    .. warning::
       **Path diversity**: This function now uses ``viterbi_optimal_score_diverse``
       (with self-loop penalty and iterative token banning) to compute
       the Viterbi baseline.  The pure Markov-chain Viterbi would
       degenerate into repetitive loops (e.g., ``IX → IX → …``).
       The diversity-constrained path provides a more realistic upper
       bound for ASL gloss sequences.

    Args:
        completion: Generated gloss sequence.
        normalize: If ``True``, exponentiate to ``(0, 1]`` range.
            If ``False``, return raw average-log-prob difference.

    Returns:
        Viterbi proximity reward.
    """
    if _bigram_matrix is None or not _gloss_vocab:
        logger.warning("Transition matrix not initialized; returning 0.0")
        return 0.0

    text = extract_gloss_text(completion)
    tokens = text.strip().split()

    if len(tokens) < 2:
        return 0.0

    bos_idx = _token_to_idx.get("<BOS>", -1)
    eos_idx = _token_to_idx.get("<EOS>", -1)

    if bos_idx < 0 or eos_idx < 0:
        logger.warning("BOS/EOS not in vocabulary; returning 0.0")
        return 0.0

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
        return 0.0

    if normalize:
        llm_avg = llm_log_prob / n_trans
        viterbi_avg = viterbi_log_prob / n_trans
        return float(np.exp(llm_avg - viterbi_avg))

    return (llm_log_prob - viterbi_log_prob) / n_trans


# ---------------------------------------------------------------------------
# Format reward: ensure gloss-only output
# ---------------------------------------------------------------------------


def gloss_format_reward(completion: str) -> float:
    """Reward for generating only gloss tokens (no free text, no JSON).

    Punishes outputs that contain natural language, code blocks,
    thinking tags, digit sequences, or mixed alphanumeric garbage.

    Args:
        completion: Raw model completion.

    Returns:
        ``1.0`` if output looks like clean glosses, ``0.5`` if mixed,
        ``0.0`` if clearly non-gloss text or garbage.
    """
    text = extract_gloss_text(completion)
    if not text:
        return 0.0

    # Check for obvious non-gloss patterns
    # Free text typically has lowercase articles, prepositions, punctuation
    free_text_patterns = [
        r"\b(the|a|an|is|are|was|were|will|would|should|can|could)\b",
        r"\b(in|on|at|by|for|with|from|to|of|and|or|but)\b",
        r"[.,!?;:]",  # punctuation
        r"```",  # code blocks
        r"\{|\}",  # JSON-like
    ]

    for pattern in free_text_patterns:
        if re.search(pattern, text, re.IGNORECASE):
            return 0.5  # mixed content

    # ── Digit / numeric garbage detection ─────────────────────────
    # Penalize sequences of 3+ consecutive digits (e.g. "13079117...")
    digit_sequences = re.findall(r"\d{3,}", text)
    if digit_sequences:
        # Proportional penalty: more digit sequences → lower reward
        total_digit_chars = sum(len(s) for s in digit_sequences)
        if total_digit_chars > 20:
            return 0.0  # severe numeric garbage
        return 0.25  # moderate numeric contamination

    # ── Token-level sanity checks ─────────────────────────────────
    tokens = text.split()
    if len(tokens) == 0:
        return 0.0

    # Check for excessively long tokens (concatenated subword garbage)
    # Real ASL glosses rarely exceed 20 characters
    long_token_count = sum(1 for t in tokens if len(t) > 25)
    if long_token_count > 0:
        return 0.5

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
        return 0.0
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
    - ``weight_structure``: Absolute bigram log-prob reward (no baseline).
    - ``weight_gold_structure``: Bigram score vs gold reference baseline
      **(recommended over weight_structure)**.
    - ``weight_viterbi``: Bigram score vs Viterbi theoretical optimum
      **(experimental — see caveat in ``viterbi_distance_reward``)**.
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
