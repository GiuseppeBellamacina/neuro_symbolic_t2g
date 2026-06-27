"""
Reward Functions for T2G GRPO Training.

Two reward components:

1. **Translation Quality Reward** (ROUGE-L):
   Measures lexical similarity between the generated gloss sequence
   and the ground-truth gloss target.  Returns a score in ``[0, 1]``.

2. **Structural Dense Reward** (Viterbi Proxy):
   Uses the precomputed bigram transition matrix to score the
   generated gloss sequence.  Sequences with high-probability
   gloss transitions (as observed in the training data) receive
   higher rewards.  This acts as a "procedural knowledge" signal.

Both rewards are combined via weighted sum and wrapped to match
the signature expected by TRL's ``GRPOTrainer``:
``fn(completions, prompts, **kwargs) -> list[float]``.
"""

from __future__ import annotations

import hashlib
import logging
import re
from typing import Any, Callable

import numpy as np
from rouge_score import rouge_scorer

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

#: ROUGE scorer (lazy-initialized).
_rouge_scorer: rouge_scorer.RougeScorer | None = None

#: Gold gloss registry: maps sample_id (SHA256 of user instruction) → gold gloss.
#  Populated at dataset load time via ``register_gold_glosses()``.
_gold_gloss_registry: dict[str, str] = {}


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------


def initialize_rewards(
    bigram_matrix: np.ndarray,
    vocab: list[str],
) -> None:
    """Initialize global state for reward functions.

    Must be called once before training starts.

    Args:
        bigram_matrix: The ``(V, V)`` bigram transition probability matrix.
        vocab: The sorted gloss vocabulary.
    """
    global _bigram_matrix, _gloss_vocab, _token_to_idx, _rouge_scorer
    _bigram_matrix = bigram_matrix
    _gloss_vocab = vocab
    _token_to_idx = {t: i for i, t in enumerate(vocab)}

    # Use ROUGE-L F1 as the primary quality metric
    _rouge_scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)

    logger.info(
        f"Rewards initialized: |V|={len(vocab)}, "
        f"bigram_matrix shape={bigram_matrix.shape}, "
        f"ROUGE-L scorer ready"
    )


def _extract_sample_id(prompt: Any) -> str:
    """Extract a stable sample ID from a prompt in any format.

    TRL's GRPOTrainer may pass prompts as formatted chat strings,
    stringified lists, or raw strings depending on the backend.
    This function extracts the user instruction (the English sentence
    to translate) regardless of format, then returns its SHA256 hash
    as a deterministic lookup key.

    Args:
        prompt: The prompt in whatever format GRPOTrainer provides.

    Returns:
        SHA256 hex digest of the user instruction, or ``""`` if no
        user content could be extracted.
    """
    if prompt is None:
        return ""

    # Format 1: list of chat messages (most common from TRL)
    if isinstance(prompt, list):
        for msg in reversed(prompt):
            if isinstance(msg, dict) and msg.get("role") == "user":
                return hashlib.sha256(
                    str(msg.get("content", "")).encode("utf-8", errors="replace")
                ).hexdigest()

    # Format 2: plain string — try to extract user instruction from
    # common chat formats before falling back to hashing the whole string.
    text = str(prompt)
    if not text:
        return ""

    # Try Qwen/ChatML format: <|im_start|>user\nTEXT<|im_end|>
    m = re.search(r"<\|im_start\|>user\s*\n(.*?)<\|im_end\|>", text, re.DOTALL)
    if m:
        return hashlib.sha256(
            m.group(1).strip().encode("utf-8", errors="replace")
        ).hexdigest()

    # Try "user: TEXT" or "user\nTEXT" pattern
    m = re.search(r"(?:^|\n)user[:\s]\n?(.*?)(?:\n|$)", text, re.IGNORECASE)
    if m:
        return hashlib.sha256(
            m.group(1).strip().encode("utf-8", errors="replace")
        ).hexdigest()

    # Fallback: hash the entire prompt string
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()


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
# Helper: extract gloss text from completion
# ---------------------------------------------------------------------------


def _extract_gloss_text(completion: str) -> str:
    """Extract clean gloss text from a model completion.

    Strips thinking tags, code fences, and extra whitespace.

    Args:
        completion: Raw model completion string.

    Returns:
        Cleaned gloss token string.
    """
    # Strip <think>...</think> blocks (if thinking mode was on)
    text = re.sub(r"<think>.*?</think>", "", completion, flags=re.DOTALL).strip()

    # Strip fenced code blocks
    m = re.search(r"```(?:gloss)?\s*(.*?)```", text, re.DOTALL)
    if m:
        text = m.group(1).strip()

    return text


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
    if _rouge_scorer is None:
        logger.warning("ROUGE scorer not initialized; returning 0.0")
        return 0.0

    generated = _extract_gloss_text(completion)
    gold = gold_gloss.strip()

    if not generated:
        return 0.0
    if not gold:
        return 0.0

    scores = _rouge_scorer.score(gold, generated)
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

    text = _extract_gloss_text(completion)
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
# Format reward: ensure gloss-only output
# ---------------------------------------------------------------------------


def gloss_format_reward(completion: str) -> float:
    """Reward for generating only gloss tokens (no free text, no JSON).

    Punishes outputs that contain natural language, code blocks, or
    thinking tags not properly handled.

    Args:
        completion: Raw model completion.

    Returns:
        ``1.0`` if output looks like clean glosses, ``0.5`` if mixed,
        ``0.0`` if clearly non-gloss text.
    """
    text = _extract_gloss_text(completion)
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

    # If text is just space-separated tokens (likely glosses), it's fine
    tokens = text.split()
    if len(tokens) > 0:
        return 1.0

    return 0.0


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
    text = _extract_gloss_text(completion)
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

    Args:
        reward_config: Dictionary with keys like ``weight_translation``,
            ``weight_structure``, ``weight_format``, ``weight_repetition``.
            If ``None``, uses default weights.

    Returns:
        Tuple of ``(reward_funcs, reward_weights)`` compatible with
        ``GRPOTrainer``.
    """
    if reward_config is None:
        reward_config = {
            "weight_translation": 0.40,
            "weight_structure": 0.40,
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

    # Structural dense reward (Viterbi proxy)
    w = reward_config.get("weight_structure", 0.0)
    if w > 0:
        funcs.append(_make_gloss_reward_fn(structural_dense_reward))
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
