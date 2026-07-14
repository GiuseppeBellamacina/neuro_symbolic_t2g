"""Evaluation metrics for T2G gloss generation.

Computes ROUGE-L Pass@1, per-component reward breakdowns, and
completion validity statistics for ASL gloss sequences.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Any

import numpy as np
from rouge_score import rouge_scorer

from src.utils.text_utils import extract_gloss_text


def check_gloss_validity(completion: str) -> tuple[bool, str]:
    """Check if a completion is a valid gloss sequence.

    Uses vocabulary membership (when available via the rewards module's
    ``_gloss_vocab``) instead of regex patterns that produce false
    positives on legitimate ASL glosses like ``CAN``, ``BE``, ``FOR``,
    ``TO``, and ``.`` (which are all valid glosses in ASLG-PC12).

    Returns:
        (is_valid, error_message) — error_message is "" if valid.
    """
    text = extract_gloss_text(completion)
    if not text:
        return False, "empty_output"

    tokens = text.split()

    # Check for code blocks / JSON wrappers (residual)
    if "```" in text or "{" in text or "}" in text:
        return False, "code_block_detected"

    # Try vocabulary-based validation (preferred — no false positives)
    try:
        from src.rewards.t2g_rewards import _gloss_vocab

        if _gloss_vocab:
            vocab_set = set(_gloss_vocab)
            valid_count = sum(1 for t in tokens if t in vocab_set)
            valid_ratio = valid_count / len(tokens) if tokens else 0.0
            if valid_ratio < 0.5:
                return False, "out_of_vocab_tokens"
            # Even if tokens are in vocab, check for excessive repetition
            # (e.g., "IX IX IX IX IX" is all valid glosses but degenerate)
            if len(tokens) > 4:
                unique_ratio = len(set(tokens)) / len(tokens)
                if unique_ratio < 0.3:
                    return False, "excessive_repetition"
            return True, ""
    except ImportError:
        pass

    # Fallback: heuristic checks (only if vocab not available)
    # NOTE: these patterns produce false positives on valid ASL glosses
    # like CAN, BE, FOR, TO — use only as last resort.
    free_text_patterns = [
        r"```",
        r"\{|\}",
    ]
    for pattern in free_text_patterns:
        if re.search(pattern, text):
            return False, "free_text_detected"

    # Check for excessive repetition (>50% same token)
    if len(tokens) > 4:
        unique_ratio = len(set(tokens)) / len(tokens)
        if unique_ratio < 0.3:
            return False, "excessive_repetition"

    return True, ""


# ---------------------------------------------------------------------------
# ROUGE-L Pass@1
# ---------------------------------------------------------------------------

_ROUGE_SCORER = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)


def rouge_l_score(generated: str, reference: str) -> float:
    """Compute ROUGE-L F1 score between generated and reference glosses.

    Args:
        generated: Generated gloss sequence.
        reference: Gold reference gloss sequence.

    Returns:
        ROUGE-L F1 score in [0, 1].
    """
    gen = extract_gloss_text(generated)
    ref = reference.strip()
    if not gen or not ref:
        return 0.0
    scores = _ROUGE_SCORER.score(ref, gen)
    return scores["rougeL"].fmeasure


def compute_pass_at_1(
    completions: list[str],
    references: list[str],
    threshold: float = 0.3,
) -> float:
    """Compute Pass@1: fraction of completions with ROUGE-L ≥ threshold.

    Args:
        completions: Generated gloss sequences.
        references: Gold reference gloss sequences (same order).
        threshold: ROUGE-L threshold for considering a pass.

    Returns:
        Pass@1 rate in [0, 1].
    """
    passes = sum(
        1 for c, r in zip(completions, references) if rouge_l_score(c, r) >= threshold
    )
    return passes / max(len(completions), 1)


def compute_pass_at_k(
    completions_per_prompt: list[list[str]],
    references: list[str],
    k_values: list[int] | tuple[int, ...] = (1, 5, 10),
    threshold: float = 0.3,
) -> dict[str, float]:
    """Compute Pass@k: fraction of prompts where at least 1 of k
    completions reaches ROUGE-L ≥ threshold.

    Args:
        completions_per_prompt: For each prompt, a list of k completions.
        references: Gold reference glosses (one per prompt).
        k_values: Which k values to compute.
        threshold: ROUGE-L pass threshold.

    Returns:
        Dict like {"pass@1": 0.72, "pass@5": 0.88, "pass@10": 0.93}.
    """
    n_prompts = len(completions_per_prompt)
    results: dict[str, float] = {}

    for k in k_values:
        passes = 0
        for comps, ref in zip(completions_per_prompt, references):
            subset = comps[:k]
            if any(rouge_l_score(c, ref) >= threshold for c in subset):
                passes += 1
        results[f"pass@{k}"] = passes / max(n_prompts, 1)

    return results


# ---------------------------------------------------------------------------
# Detailed metrics
# ---------------------------------------------------------------------------


def compute_detailed_metrics(
    completions: list[str],
    references: list[str],
) -> dict[str, Any]:
    """Compute detailed T2G evaluation metrics.

    Args:
        completions: Generated gloss sequences.
        references: Gold reference glosses.

    Returns:
        Dict with: overall_pass_rate, overall_rouge_l, per_category breakdown,
        error distribution.
    """
    total = len(completions)
    valid_count = 0
    rouge_scores: list[float] = []
    error_types: Counter = Counter()

    for comp, ref in zip(completions, references):
        is_valid, error_msg = check_gloss_validity(comp)
        rl = rouge_l_score(comp, ref) if is_valid else 0.0
        rouge_scores.append(rl)

        if is_valid and rl >= 0.3:
            valid_count += 1
        else:
            error_types[error_msg or "low_rouge_l"] += 1

    return {
        "overall_pass_rate": valid_count / max(total, 1),
        "overall_rouge_l": float(np.mean(rouge_scores)),
        "total_samples": total,
        "valid_samples": valid_count,
        "rouge_l_percentiles": {
            "25%": float(np.percentile(rouge_scores, 25)),
            "50%": float(np.percentile(rouge_scores, 50)),
            "75%": float(np.percentile(rouge_scores, 75)),
            "90%": float(np.percentile(rouge_scores, 90)),
        },
        "error_distribution": dict(error_types.most_common(20)),
    }


# ---------------------------------------------------------------------------
# Per-component reward breakdown
# ---------------------------------------------------------------------------


def compute_reward_breakdown(
    completions: list[str],
    prompts: list[str] | None = None,
    sample_ids: list[str] | None = None,
    reward_weights: dict[str, float] | None = None,
) -> dict[str, float]:
    """Compute average score for each T2G reward component directly.

    Calls each reward function (translation_quality, structural_dense,
    gloss_format, gloss_repetition) on every completion and returns
    the mean per component.

    Args:
        completions: Generated gloss sequences.
        prompts: Optional prompts, used as fallback to look up gold glosses
            via ``_extract_sample_id`` if ``sample_ids`` is not provided.
        sample_ids: Optional stable sample IDs (SHA256 of user text) for
            reliable gold gloss lookup.  Preferred over ``prompts``.
        reward_weights: Optional dict mapping component name → weight.
            If provided, only components with weight > 0 are computed
            (others are skipped to save computation).

    Returns:
        Dict mapping component name → average score.
    """
    from src.rewards.t2g_rewards import (
        _lookup_gold_gloss,
        _lookup_gold_gloss_by_id,
        bleu_reward,
        gloss_format_reward,
        gloss_order_reward,
        gloss_repetition_reward,
        gold_structure_reward,
        soft_viterbi_distance_reward,
        structural_dense_reward,
        translation_quality_reward,
        verifier_scaled_reward,
        viterbi_distance_reward,
    )

    n = len(completions)
    # Build set of active components (weight > 0) to skip computation of others
    _ACTIVE: set[str] | None = None
    if reward_weights is not None:
        _ACTIVE = {k for k, v in reward_weights.items() if v > 0}

    def _is_active(name: str) -> bool:
        """Return True if this component should be computed."""
        return _ACTIVE is None or name in _ACTIVE

    sums = {
        "translation_quality_reward": 0.0,
        "bleu_reward": 0.0,
        "structural_dense_reward": 0.0,
        "gold_structure_reward": 0.0,
        "viterbi_distance_reward": 0.0,
        "soft_viterbi_distance_reward": 0.0,
        "verifier_scaled_reward": 0.0,
        "gloss_order_reward": 0.0,
        "gloss_format_reward": 0.0,
        "gloss_repetition_reward": 0.0,
    }

    for i, comp in enumerate(completions):
        gold = ""
        if sample_ids and i < len(sample_ids) and sample_ids[i]:
            gold = _lookup_gold_gloss_by_id(sample_ids[i])
        elif prompts and i < len(prompts):
            gold = _lookup_gold_gloss(prompts[i])
        if _is_active("translation_quality_reward"):
            sums["translation_quality_reward"] += translation_quality_reward(comp, gold)
        if _is_active("bleu_reward"):
            sums["bleu_reward"] += bleu_reward(comp, gold)
        if _is_active("structural_dense_reward"):
            sums["structural_dense_reward"] += structural_dense_reward(
                comp, normalize=True
            )
        if _is_active("gold_structure_reward"):
            sums["gold_structure_reward"] += gold_structure_reward(
                comp, gold, normalize=True
            )
        if _is_active("viterbi_distance_reward"):
            sums["viterbi_distance_reward"] += viterbi_distance_reward(
                comp, normalize=True
            )
        if _is_active("soft_viterbi_distance_reward"):
            sums["soft_viterbi_distance_reward"] += soft_viterbi_distance_reward(
                comp, normalize=True
            )
        if _is_active("verifier_scaled_reward"):
            sums["verifier_scaled_reward"] += verifier_scaled_reward(comp, gold)
        if _is_active("gloss_order_reward"):
            sums["gloss_order_reward"] += gloss_order_reward(comp, gold)
        if _is_active("gloss_format_reward"):
            sums["gloss_format_reward"] += gloss_format_reward(comp)
        if _is_active("gloss_repetition_reward"):
            sums["gloss_repetition_reward"] += gloss_repetition_reward(comp)

    # Only return components that were actually computed (active ones)
    result = {k: v / max(n, 1) for k, v in sums.items()}
    if _ACTIVE is not None:
        result = {k: v for k, v in result.items() if k in _ACTIVE}
    return result


# ---------------------------------------------------------------------------
# BLEU Score (corpus-level and sentence-level)
# ---------------------------------------------------------------------------


def _ngram_counts(tokens: list[str], n: int) -> Counter:
    """Count n-grams in a token list."""
    return Counter(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))


def sentence_bleu(reference: str, hypothesis: str, max_n: int = 4) -> float:
    """Compute sentence-level BLEU score (geometric mean of n-gram precision).

    Uses uniform weights (1/max_n) for n-gram precisions and applies
    brevity penalty.  Suitable for short gloss sequences.

    Args:
        reference: Gold reference gloss sequence.
        hypothesis: Generated gloss sequence.
        max_n: Maximum n-gram order (default 4 = BLEU-4).

    Returns:
        BLEU score in [0, 1].
    """
    ref_tokens = reference.strip().split()
    hyp_tokens = hypothesis.strip().split()

    if not hyp_tokens or not ref_tokens:
        return 0.0

    # Brevity penalty
    bp = (
        1.0
        if len(hyp_tokens) >= len(ref_tokens)
        else np.exp(1 - len(ref_tokens) / len(hyp_tokens))
    )

    # Geometric mean of n-gram precisions
    log_precisions = []
    for n in range(1, max_n + 1):
        ref_counts = _ngram_counts(ref_tokens, n)
        hyp_counts = _ngram_counts(hyp_tokens, n)

        total = sum(hyp_counts.values())
        if total == 0:
            log_precisions.append(-np.inf)
            continue

        matches = sum(min(hyp_counts[ng], ref_counts.get(ng, 0)) for ng in hyp_counts)
        precision = matches / total
        if precision == 0:
            log_precisions.append(-np.inf)
        else:
            log_precisions.append(np.log(precision))

    if any(np.isneginf(lp) for lp in log_precisions):
        return 0.0

    geo_mean = np.exp(np.mean(log_precisions))
    return float(bp * geo_mean)


def corpus_bleu(references: list[str], hypotheses: list[str], max_n: int = 4) -> float:
    """Compute corpus-level BLEU score.

    Aggregates n-gram matches across all sentences before computing
    precision, which is more accurate than averaging sentence-level BLEU.

    Args:
        references: List of gold reference gloss sequences.
        hypotheses: List of generated gloss sequences.
        max_n: Maximum n-gram order.

    Returns:
        Corpus BLEU score in [0, 1].
    """
    if not references or not hypotheses:
        return 0.0

    total_matches = [0] * max_n
    total_counts = [0] * max_n
    ref_len_total = 0
    hyp_len_total = 0

    for ref, hyp in zip(references, hypotheses):
        ref_tokens = ref.strip().split()
        hyp_tokens = hyp.strip().split()
        ref_len_total += len(ref_tokens)
        hyp_len_total += len(hyp_tokens)

        for n in range(1, max_n + 1):
            ref_counts = _ngram_counts(ref_tokens, n)
            hyp_counts = _ngram_counts(hyp_tokens, n)
            total_counts[n - 1] += sum(hyp_counts.values())
            total_matches[n - 1] += sum(
                min(hyp_counts[ng], ref_counts.get(ng, 0)) for ng in hyp_counts
            )

    # Brevity penalty (corpus-level)
    bp = (
        1.0
        if hyp_len_total >= ref_len_total
        else np.exp(1 - ref_len_total / max(hyp_len_total, 1))
    )

    log_precisions = []
    for n in range(max_n):
        if total_counts[n] == 0 or total_matches[n] == 0:
            return 0.0
        log_precisions.append(np.log(total_matches[n] / total_counts[n]))

    geo_mean = np.exp(np.mean(log_precisions))
    return float(bp * geo_mean)


# ---------------------------------------------------------------------------
# Bootstrap confidence intervals
# ---------------------------------------------------------------------------


def bootstrap_confidence_interval(
    values: list[float],
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Compute bootstrap confidence interval for the mean of a list of values.

    Implements the bootstrap resampling method described in
    Koehn (2004) and commonly used in MT evaluation.  This is the
    standard method for reporting statistical significance in
    machine translation papers.

    Args:
        values: List of per-sample metric values (e.g. ROUGE-L scores).
        n_bootstrap: Number of bootstrap resamples (default 1000).
        confidence: Confidence level (default 0.95 = 95% CI).
        seed: Random seed for reproducibility.

    Returns:
        (mean, lower_bound, upper_bound) — the mean and the
        confidence interval bounds.
    """
    if not values:
        return 0.0, 0.0, 0.0

    rng = np.random.RandomState(seed)
    values_arr = np.array(values)
    n = len(values_arr)
    alpha = 1 - confidence

    bootstrap_means = np.array(
        [values_arr[rng.randint(0, n, n)].mean() for _ in range(n_bootstrap)]
    )

    lower = float(np.percentile(bootstrap_means, 100 * alpha / 2))
    upper = float(np.percentile(bootstrap_means, 100 * (1 - alpha / 2)))
    mean = float(values_arr.mean())

    return mean, lower, upper


def compute_evaluation_report(
    completions: list[str],
    references: list[str],
    n_bootstrap: int = 1000,
) -> dict[str, Any]:
    """Compute a comprehensive evaluation report with confidence intervals.

    This is the main entry point for professional evaluation of T2G
    models.  It computes:

    - ROUGE-L (mean, 95% CI)
    - BLEU (corpus-level + sentence-level mean with 95% CI)
    - Pass@1 (with 95% CI)
    - Gloss validity rate
    - Error distribution

    Inspired by the evaluation protocol in RECIPE (arXiv:2605.19976),
    which uses reference-based evaluation with bootstrap confidence
    intervals for statistical significance.

    Args:
        completions: Generated gloss sequences.
        references: Gold reference gloss sequences.
        n_bootstrap: Number of bootstrap resamples for CIs.

    Returns:
        Dict with all metrics and confidence intervals.
    """
    total = len(completions)

    # Per-sample metrics
    rouge_scores = [rouge_l_score(c, r) for c, r in zip(completions, references)]
    bleu_scores = [sentence_bleu(r, c) for c, r in zip(completions, references)]
    pass_scores = [1.0 if s >= 0.3 else 0.0 for s in rouge_scores]

    # Validity
    valid_results = [check_gloss_validity(c) for c in completions]
    valid_count = sum(1 for is_valid, _ in valid_results if is_valid)
    error_types = Counter(msg for _, msg in valid_results if msg)

    # Bootstrap CIs
    rouge_mean, rouge_lo, rouge_hi = bootstrap_confidence_interval(
        rouge_scores, n_bootstrap
    )
    bleu_mean, bleu_lo, bleu_hi = bootstrap_confidence_interval(
        bleu_scores, n_bootstrap
    )
    pass_mean, pass_lo, pass_hi = bootstrap_confidence_interval(
        pass_scores, n_bootstrap
    )

    # Corpus BLEU
    corpus_bleu_score = corpus_bleu(references, completions)

    return {
        "total_samples": total,
        "rouge_l": {
            "mean": rouge_mean,
            "ci_95": [rouge_lo, rouge_hi],
            "percentiles": {
                "25%": float(np.percentile(rouge_scores, 25)),
                "50%": float(np.percentile(rouge_scores, 50)),
                "75%": float(np.percentile(rouge_scores, 75)),
                "90%": float(np.percentile(rouge_scores, 90)),
            },
        },
        "bleu": {
            "corpus": corpus_bleu_score,
            "sentence_mean": bleu_mean,
            "ci_95": [bleu_lo, bleu_hi],
        },
        "pass_at_1": {
            "mean": pass_mean,
            "ci_95": [pass_lo, pass_hi],
        },
        "gloss_validity_rate": valid_count / max(total, 1),
        "error_distribution": dict(error_types.most_common(20)),
    }
