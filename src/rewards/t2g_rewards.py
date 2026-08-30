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

import logging
import re
from typing import Any, Callable

import numpy as np
from rouge_score import rouge_scorer

from src.utils.text_utils import extract_gloss_text

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
}

#: Guard flag: when True, the missing-gold warning has already been logged.
#  Reset in ``initialize_rewards`` so the warning fires at most once per run
#  (and once per test setup).
_warned_missing_gold: bool = False

# ── Structural-reward caches ─────────────────────────────────────────────────
# The Viterbi/soft-Viterbi bounds are properties of the AUTOMATON and the
# path length ONLY — never of the completion being scored.  Caching them by
# length turns the per-completion O(L·V²) decodes (the 391 s/it GRPO steps
# of run 7078: ~64 completions/step × 2 decodes each) into ~a dozen decodes
# per run, amortized to zero.  Gold stats depend only on the gold text (the
# same gold is scored once per completion in a group — 8× reuse per prompt).
_viterbi_bound_cache: dict[int, float] = (
    {}
)  # path_length -> per-transition diverse-Viterbi bound
_soft_bound_cache: dict[int, float] = {}  # path_length -> per-transition log-partition
_gold_stats_cache: dict[str, tuple[float, int, int]] = (
    {}
)  # gold text -> (s_avg, path_length, in_vocab)

#: Cap for the gold-stats cache (cleared when exceeded — a fresh epoch just
#: recomputes; entries are ~100 bytes so this is a few MB at most).
_GOLD_CACHE_MAX = 65536


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
    global _viterbi_diversity_params, _warned_missing_gold
    global _viterbi_bound_cache, _soft_bound_cache, _gold_stats_cache
    _bigram_matrix = bigram_matrix
    _gloss_vocab = vocab
    _token_to_idx = {t: i for i, t in enumerate(vocab)}

    # Structural-reward caches: the Viterbi/soft-Viterbi bounds depend ONLY
    # on (matrix, BOS/EOS, path length, diversity params) and the gold stats
    # only on the gold text — both are fixed for the lifetime of one
    # initialize_rewards() call, so clear them here (a new matrix/config
    # must never see stale bounds).
    _viterbi_bound_cache.clear()
    _soft_bound_cache.clear()
    _gold_stats_cache.clear()

    # Use ROUGE-L F1 as the primary quality metric
    _ROUGE_SCORER = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)

    # Set Viterbi diversity params from config (only the params actually used
    # by the Viterbi/soft-Viterbi rewards; verifier_gamma and
    # verifier_temperature are dead config and no longer loaded).
    diversity_cfg = viterbi_diversity or {}

    _viterbi_diversity_params = {
        "self_loop_penalty": diversity_cfg.get("self_loop_penalty", 0.5),
        "max_occurrences": diversity_cfg.get("max_occurrences", 2),
        "diversity_threshold": diversity_cfg.get("diversity_threshold", 0.3),
        "max_iters": diversity_cfg.get("max_iters", 3),
    }
    logger.info("Viterbi diversity params: %s", _viterbi_diversity_params)

    # Reset the one-time "gold_gloss missing" warning flag so it can fire
    # again after a fresh initialization (e.g. new training run / test setup).
    _warned_missing_gold = False


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
# Reward Component 2: Structural Dense Reward (Viterbi proxy)
# ---------------------------------------------------------------------------


def _indices_skip_oov(tokens: list[str]) -> tuple[list[int], int]:
    """BOS/EOS-wrapped in-vocab indices + OOV count.

    OOV tokens are skipped (NOT mapped to ``<UNK>``) so garbage tokens
    cannot get partial credit via ``<UNK>`` bigram probabilities — the
    same anti-hacking pattern as ``gold_structure_reward``.
    """
    indices: list[int] = []
    oov = 0
    bos = _token_to_idx.get("<BOS>", -1)
    eos = _token_to_idx.get("<EOS>", -1)
    if bos >= 0:
        indices.append(bos)
    for t in tokens:
        idx = _token_to_idx.get(t, -1)
        if idx >= 0:
            indices.append(idx)
        else:
            oov += 1
    if eos >= 0:
        indices.append(eos)
    return indices, oov


def _gold_stats(gold_text: str) -> tuple[float, int, int] | None:
    """Per-transition bigram stats of the gold reference, cached by text.

    Returns ``(s_avg, path_length, in_vocab_tokens)`` or ``None`` for a
    degenerate gold (< 2 in-vocab tokens).  The same gold is scored once
    per completion in a GRPO group (num_generations × reuse) — the cache
    turns that into a single computation.
    """
    cached = _gold_stats_cache.get(gold_text)
    if cached is not None:
        return cached
    matrix = _bigram_matrix
    assert matrix is not None  # callers check; initialize_rewards contract
    tokens = gold_text.strip().split()
    indices, oov = _indices_skip_oov(tokens)
    in_vocab = len(tokens) - oov
    if len(indices) < 3 or in_vocab < 2:
        return None
    from src.datasets.transition_matrix import sequence_score_bigram

    lp = sequence_score_bigram(matrix, indices)
    stats = (lp / (len(indices) - 1), len(indices), in_vocab)
    if len(_gold_stats_cache) >= _GOLD_CACHE_MAX:
        _gold_stats_cache.clear()
    _gold_stats_cache[gold_text] = stats
    return stats


def _viterbi_bound_avg(path_length: int) -> float:
    """Per-transition diverse-Viterbi bound for *path_length* (cached).

    The bound is a property of the automaton + length only — see the
    module-level cache docs.  Raises if path_length < 2.
    """
    if path_length not in _viterbi_bound_cache:
        from src.datasets.transition_matrix import viterbi_optimal_score_diverse

        matrix = _bigram_matrix
        assert matrix is not None  # callers check; initialize_rewards contract
        bound = viterbi_optimal_score_diverse(
            matrix,
            _token_to_idx["<BOS>"],
            _token_to_idx["<EOS>"],
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
        _viterbi_bound_cache[path_length] = bound / (path_length - 1)
    return _viterbi_bound_cache[path_length]


def _soft_bound_avg(path_length: int) -> float:
    """Per-transition log-partition (soft Viterbi) for *path_length* (cached)."""
    if path_length not in _soft_bound_cache:
        from src.datasets.transition_matrix import soft_viterbi_score

        matrix = _bigram_matrix
        assert matrix is not None  # callers check; initialize_rewards contract
        bound = soft_viterbi_score(
            matrix,
            _token_to_idx["<BOS>"],
            _token_to_idx["<EOS>"],
            path_length,
        )
        _soft_bound_cache[path_length] = bound / (path_length - 1)
    return _soft_bound_cache[path_length]


def _gold_anchored_structural_reward(
    completion: str,
    gold_gloss: str,
    anchor: str,
    temperature: float = 1.5,
    normalize: bool = True,
) -> float:
    """Core of the three structural modules (gold-anchored calibration).

    **Why gold-anchored.**  The pre-fix versions anchored ``exp()`` at
    unattainable optima — absolute 0 (perfect certainty), the
    diverse-Viterbi bound (the theoretical best path) or the
    log-partition (soft bound).  Natural glosses sit 3–6 nats below those
    anchors per transition, so ``2·exp(gap) − 1`` saturated: on run 7078
    even PERFECT completions scored −0.79…−1.00 with std ≈ 0.0001 (zero
    GRPO advantage signal).  Anchoring at the GOLD's own value instead:

    * completion == gold → ``+1`` (calibrated);
    * delta in per-transition nats divided by ``temperature`` τ:
      with τ=1.5 a single-word error ≈ neutral, a shuffled gloss ≈ −0.4,
      random tokens ≈ −0.7 (measured on the real 15518² matrix).

    ``anchor`` selects the geometry:

    * ``"absolute"``: delta = s(completion) − s(gold) — relative bigram
      plausibility (structural_dense / ViterbiPlanNet-DVL flavour);
    * ``"viterbi"``: delta = gap(gold) − gap(completion) where
      gap(x) = diverseViterbi_bound(L_x) − s(x) — how much of the
      achievable headroom the completion uses, relative to the gold;
    * ``"soft_viterbi"``: same with the log-partition (smooth) bound.

    Guards (shared with ``gold_structure_reward``): < 2 in-vocab tokens →
    hard −1 (short/garbage cannot carry a structural comparison); OOV
    ratio and in-vocab length-mismatch penalties applied before the
    symmetric mapping (shorter-than-gold outputs stop getting free
    average-log-prob credit).

    Missing/empty gold → neutral ``0.0`` (warned once): without the
    anchor the reward cannot be calibrated, and a constant −1 would
    silently poison every rollout.
    """
    global _warned_missing_gold

    if _bigram_matrix is None or not _gloss_vocab:
        logger.warning("Transition matrix not initialized; returning -1.0")
        return -1.0

    gold_text = (gold_gloss or "").strip()
    if not gold_text:
        if not _warned_missing_gold:
            _warned_missing_gold = True
            logger.warning(
                "gold_gloss missing/empty for '%s'; returning neutral 0.0 "
                "(gold-anchored structural rewards need the gold reference).",
                anchor,
            )
        return 0.0

    gold = _gold_stats(gold_text)
    if gold is None:
        if not _warned_missing_gold:
            _warned_missing_gold = True
            logger.warning(
                "Degenerate gold (< 2 in-vocab tokens) for '%s': %r — "
                "returning neutral 0.0.",
                anchor,
                gold_text[:60],
            )
        return 0.0
    s_gold, gold_path_length, gold_in_vocab = gold

    text = extract_gloss_text(completion)
    tokens = text.strip().split() if text else []
    indices, oov = _indices_skip_oov(tokens)
    in_vocab = len(tokens) - oov
    if in_vocab < 2:
        return -1.0  # hard fail — anti reward-hacking (cf. gold_structure)

    from src.datasets.transition_matrix import sequence_score_bigram

    n_trans = len(indices) - 1
    if n_trans <= 0:
        return -1.0
    s_comp = sequence_score_bigram(_bigram_matrix, indices) / n_trans  # type: ignore[arg-type]

    if anchor == "absolute":
        delta = s_comp - s_gold
    else:
        bound = _viterbi_bound_avg if anchor == "viterbi" else _soft_bound_avg
        gap_c = bound(len(indices)) - s_comp
        gap_g = bound(gold_path_length) - s_gold
        delta = gap_g - gap_c  # > 0 → completion uses headroom better than gold

    if not normalize:
        return _clamp_symmetric(delta)

    tau = max(float(temperature), 1e-8)
    reward = min(float(np.exp(delta / tau)), 1.0)  # ≥ gold capped at +1
    # OOV penalty: garbage tokens lose credit proportionally.
    if tokens:
        reward *= 1.0 - (oov / len(tokens))
    # Length-mismatch penalty: shorter-than-gold outputs stop receiving
    # free average-log-prob credit (fewer transitions ≈ higher average).
    reward *= min(1.0, min(in_vocab, gold_in_vocab) / max(in_vocab, gold_in_vocab, 1))
    return _to_symmetric(reward)


def structural_dense_reward(
    completion: str,
    gold_gloss: str = "",
    temperature: float = 1.5,
    normalize: bool = True,
) -> float:
    """Gold-anchored relative bigram plausibility of the completion.

    Measures whether the generated gloss is as structurally plausible
    (under the corpus bigram model) as the gold reference:

    .. math::

        \\text{reward} = 2\\exp\\left(
            \\frac{s_{\\text{completion}} - s_{\\text{gold}}}{\\tau}
        \\right) - 1

    where ``s`` is the average per-transition bigram log-probability and
    ``τ`` (``temperature``) controls the sharpness.  At the default
    τ=1.5, measured on the real ASLG-PC12 bigram matrix: gold → +1,
    single-word corruption ≈ +0.05, shuffled ≈ −0.4, random tokens ≈ −0.75.

    .. warning::
       **v2 — gold-anchored.**  The previous absolute formulation
       (``2·exp(s) − 1``) saturated at ≈ −1 for EVERY natural sequence
       (gold included: real bigram averages are −5…−8 nats): miscalibrated
       AND with std ≈ 0 across completions, i.e. zero GRPO signal (run
       7078).  At ``temperature=1`` this component coincides with
       ``gold_structure_reward``; keep it for the tunable-sharpness (DVL)
       variant in ablations.

    Args:
        completion: Generated gloss sequence.
        gold_gloss: Ground-truth gloss sequence (the calibration anchor).
        temperature: τ — nats of per-transition difference per unit of
            reward.  Smaller = sharper discrimination.
        normalize: If ``True`` return the calibrated ``[-1, 1]`` reward;
            if ``False`` the raw per-transition delta (nats), clamped.

    Returns:
        Structural plausibility reward in ``[-1, 1]``: ``+1`` when the
        completion is as plausible as (or more than) the gold.
    """
    return _gold_anchored_structural_reward(
        completion,
        gold_gloss,
        anchor="absolute",
        temperature=temperature,
        normalize=normalize,
    )


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

    **Anti reward-hacking safeguards**:

    - A completion with **fewer than 2 in-vocabulary tokens** (e.g. ``"IX"``
      or all-OOV garbage) scores a hard ``-1.0``.  Without this guard, such
      sequences would glide on the near-uniform ``<BOS> → <EOS>`` transition
      instead of being penalized.
    - A **length-mismatch penalty** ``min(1, min(a, b) / max(a, b))`` (where
      ``a``/``b`` are the in-vocab token counts of completion and gold) is
      applied to the normalized reward BEFORE the symmetric ``[-1, 1]``
      mapping.  Rewarding average bigram log-probability alone favors SHORT
      paths — fewer transitions means fewer chances to hit a low-probability
      edge and thus a higher average.  The factor down-weights any length
      asymmetry (e.g. a 2-token completion against a 6-token gold → factor
      ``1/3``), so degenerate short outputs stop receiving free structural
      credit.

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

    # Effective structural length = number of in-vocabulary tokens (OOV
    # tokens are skipped by _indices and must not count toward length).
    llm_vocab_len = len(llm_text.split()) - llm_oov
    gold_vocab_len = len(gold_text.split()) - gold_oov

    # Anti-hacking guard: fewer than 2 in-vocab tokens → hard failure.
    # Such a sequence cannot carry a meaningful structural comparison; with
    # 0 or 1 in-vocab tokens the path degenerates to BOS→EOS / BOS→tok→EOS
    # whose near-uniform probabilities would give garbage free credit.
    if llm_vocab_len < 2:
        return -1.0

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
        # Length-mismatch penalty (see docstring): down-weights completions
        # whose in-vocab length diverges from the gold's, applied BEFORE the
        # symmetric mapping so asymmetry pushes the score below 0.
        length_factor = min(
            1.0,
            min(llm_vocab_len, gold_vocab_len) / max(llm_vocab_len, gold_vocab_len, 1),
        )
        reward *= length_factor
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
    gold_gloss: str = "",
    temperature: float = 1.5,
    normalize: bool = True,
) -> float:
    """Gold-anchored distance from the Viterbi-optimal path.

    For a sequence ``x`` of length ``L_x``, the *headroom gap* is

    .. math::

        \\text{gap}(x) = \\text{viterbi\\_bound}(L_x)/L_x - s(x)

    i.e. how far below the (diversity-constrained) theoretical optimum
    ``x`` sits, per transition.  The reward anchors at the GOLD's own gap:

    .. math::

        \\text{reward} = 2\\exp\\left(
            \\frac{\\text{gap}(\\text{gold}) - \\text{gap}(\\text{completion})}{\\tau}
        \\right) - 1

    * completion == gold → ``+1`` (calibrated);
    * completion exploits the headroom worse than the gold → toward ``-1``;
    * better than the gold (rare — closer to the optimum than the
      human reference) → capped at ``+1``.

    .. warning::
       **v2 — gold-anchored.**  The previous formulation
       (``2·exp(s − viterbi_bound) − 1``) compared every completion
       against the unattainable theoretical optimum: real glosses sit
       3–4.5 nats/transition below it, so the reward saturated at
       ≈ −0.9 for everything INCLUDING the gold (run 7078: mean −0.909,
       std 0.0009 → zero GRPO signal).  Anchoring at the gold's own gap
       restores calibration and discrimination.  The length-keyed bound
       cache (``_viterbi_bound_cache``) makes the expensive decode a
       once-per-length cost instead of once-per-completion — the main
       fix for the 391 s/it GRPO steps of run 7078.

    Args:
        completion: Generated gloss sequence.
        gold_gloss: Ground-truth gloss sequence (the calibration anchor).
        temperature: τ — nats of gap difference per unit of reward.
        normalize: If ``False`` return the raw per-transition gap delta
            (nats, clamped) instead of the calibrated reward.

    Returns:
        Viterbi-proximity reward in ``[-1, 1]`` (symmetric).
    """
    return _gold_anchored_structural_reward(
        completion,
        gold_gloss,
        anchor="viterbi",
        temperature=temperature,
        normalize=normalize,
    )


# ---------------------------------------------------------------------------
# Reward Component 4b: Soft Viterbi Distance Reward (Differentiable)
# ---------------------------------------------------------------------------


def soft_viterbi_distance_reward(
    completion: str,
    gold_gloss: str = "",
    temperature: float = 1.5,
    normalize: bool = True,
) -> float:
    """Gold-anchored soft-Viterbi (log-partition) distance reward.

    Differentiable-flavoured variant of :func:`viterbi_distance_reward`
    (inspired by ViterbiPlanNet's Differentiable Viterbi Layer,
    arXiv:2603.04265): the headroom gap is measured against the
    **log-partition function** (forward pass over all paths of the same
    length) instead of the single best path:

    .. math::

        \\text{gap}(x) = \\log Z(L_x)/L_x - s(x)

    and the reward is anchored at the GOLD's own gap (see
    :func:`viterbi_distance_reward` for the v2 rationale — the previous
    absolute formulation saturated at ≈ −0.99 with std ≈ 0 on run 7078
    because every natural sequence sits ~5 nats/transition below the
    partition function).

    The log-partition is a property of automaton+length only, so it is
    cached per path length (``_soft_bound_cache``) — one forward pass
    per length per run instead of one per completion.

    Args:
        completion: Generated gloss sequence.
        gold_gloss: Ground-truth gloss sequence (the calibration anchor).
        temperature: τ — nats of gap difference per unit of reward.
        normalize: If ``False`` return the raw per-transition gap delta
            (nats, clamped) instead of the calibrated reward.

    Returns:
        Soft Viterbi proximity reward in ``[-1, 1]`` (symmetric).
    """
    return _gold_anchored_structural_reward(
        completion,
        gold_gloss,
        anchor="soft_viterbi",
        temperature=temperature,
        normalize=normalize,
    )


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

    Sequences shorter than 4 tokens cannot exhibit meaningful repetition
    (no trigram is even available), so they are scored as NEUTRAL ``0.0``.
    The previous behavior returned ``+1.0`` unconditionally for short
    outputs, which incentivized reward hacking: the model could emit 1–3
    tokens and collect a free +1.0.  Empty outputs are treated the same way.

    Args:
        completion: Raw model completion.

    Returns:
        ``1.0`` for normal output, ``0.0`` for short (<4 tokens) or empty
        output, ``-0.3`` for moderate repetition, ``-1.0`` for severe loops.
    """
    text = extract_gloss_text(completion)
    if not text:
        return 0.0

    tokens = text.split()
    if len(tokens) < 4:
        return 0.0

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
            "sacrebleu is a core dependency of this project — your "
            "environment is out of sync with pyproject.toml. Reinstall "
            "with: pip install -e . (or: uv sync)."
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
            "sacrebleu is a core dependency of this project — your "
            "environment is out of sync with pyproject.toml. Reinstall "
            "with: pip install -e . (or: uv sync)."
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

    BLEU as a GRPO reward signal for translation on small models is
    validated by RVLF (Rao et al., 2025, arXiv:2512.07273 — BLEU+ROUGE
    rewards for sign language translation) and by Mosquera et al., 2025
    (arXiv:2508.19481 — GRPO with BLEU similarity reward on
    Qwen2.5-0.5B). See docs/SOURCES.md for the full bibliography.

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
        ``fn(completions, prompts, **kwargs) -> list[float]``

    TRL 0.24 forwards every extra dataset column to the reward function as a
    keyword argument (see ``GRPOTrainer._calculate_rewards``: it builds
    ``reward_kwargs`` from all input columns except ``prompt``/``completion``/
    ``completion_ids`` and calls ``reward_func(prompts=…, completions=…,
    completion_ids=…, **reward_kwargs)``).  Therefore, for
    ``needs_gold_gloss=True``, the gold reference is read directly from the
    ``gold_gloss`` kwarg (a list aligned with ``completions``): ``gold_gloss[idx]``.
    The dataset must retain a ``gold_gloss`` column (see ``build_t2g_dataset``).

    If ``gold_gloss`` is missing, ``None``, or empty for a sample, the wrapper
    logs a warning ONCE per run and returns a neutral ``0.0`` — never ``-1.0``,
    so a misconfigured pipeline degrades the reward signal instead of
    silently punishing every rollout.

    Args:
        component_fn: A function taking a single completion (and optionally
            gold gloss text) and returning a float.
        needs_gold_gloss: If ``True``, the function also receives the gold
            gloss target provided via the ``gold_gloss`` kwargs list.

    Returns:
        A callable with the GRPOTrainer-compatible signature.
    """

    def reward_fn(
        completions: list[Any],
        prompts: list[Any] | None = None,
        *,
        gold_gloss: list[str] | None = None,
        **kwargs: Any,
    ) -> list[float]:
        global _warned_missing_gold

        results: list[float] = []
        for idx, completion in enumerate(completions):
            # Handle GRPOTrainer's completion format: list of messages
            text: str = (
                completion[0]["content"]
                if isinstance(completion, list)
                else str(completion)
            )

            if needs_gold_gloss:
                gold = (
                    gold_gloss[idx]
                    if gold_gloss is not None and idx < len(gold_gloss)
                    else None
                )
                if gold is None or not str(gold).strip():
                    if not _warned_missing_gold:
                        _warned_missing_gold = True
                        logger.warning(
                            "gold_gloss missing/empty for reward '%s' (sample %d); "
                            "returning neutral 0.0. Ensure the training dataset "
                            "retains a 'gold_gloss' column — TRL forwards extra "
                            "dataset columns to reward functions as kwargs.",
                            component_fn.__name__,
                            idx,
                        )
                    results.append(0.0)
                else:
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

    # Structural dense reward (gold-anchored relative bigram plausibility)
    w = reward_config.get("weight_structure", 0.0)
    if w > 0:
        funcs.append(
            _make_gloss_reward_fn(structural_dense_reward, needs_gold_gloss=True)
        )
        weights.append(w)

    # Gold-structure reward (bigram score vs gold reference baseline)
    # *** Recommended over weight_structure for production ***
    w = reward_config.get("weight_gold_structure", 0.0)
    if w > 0:
        funcs.append(
            _make_gloss_reward_fn(gold_structure_reward, needs_gold_gloss=True)
        )
        weights.append(w)

    # Viterbi distance reward (gold-anchored headroom-gap vs the
    # diversity-Viterbi bound — v2, calibrated; see its docstring)
    w = reward_config.get("weight_viterbi", 0.0)
    if w > 0:
        funcs.append(
            _make_gloss_reward_fn(viterbi_distance_reward, needs_gold_gloss=True)
        )
        weights.append(w)

    # Soft Viterbi distance reward (gold-anchored, log-partition bound;
    # ViterbiPlanNet DVL-inspired — v2, calibrated)
    w = reward_config.get("weight_soft_viterbi", 0.0)
    if w > 0:
        funcs.append(
            _make_gloss_reward_fn(soft_viterbi_distance_reward, needs_gold_gloss=True)
        )
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
