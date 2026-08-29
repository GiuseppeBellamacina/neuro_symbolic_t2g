"""Evaluation metrics for T2G gloss generation.

Computes ROUGE-L, sacreBLEU BLEU (sentence/corpus), chrF2, token-level
gloss F1, Pass@k, per-component reward breakdowns (direct calls — no
gold-gloss registry), seeded evaluation sampling, and completion
validity statistics for ASL gloss sequences.
"""

from __future__ import annotations

import random
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
# ROUGE-L Pass@k
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

#: Reward components that need a gold reference gloss (completion, gold).
_GOLD_REWARD_COMPONENTS: tuple[str, ...] = (
    "translation_quality_reward",
    "bleu_reward",
    "gold_structure_reward",
    "gloss_order_reward",
    "verifier_scaled_reward",
)

#: Reward components that score the completion alone (no gold needed).
_FREE_REWARD_COMPONENTS: tuple[str, ...] = (
    "gloss_format_reward",
    "gloss_repetition_reward",
)

#: Optional structural components that may be removed by refactors of
#: ``src.rewards`` — looked up defensively so a missing function is
#: skipped instead of crashing the eval.
_OPTIONAL_FREE_REWARD_COMPONENTS: tuple[str, ...] = (
    "structural_dense_reward",
    "viterbi_distance_reward",
    "soft_viterbi_distance_reward",
)


def compute_reward_breakdown(
    completions: list[str],
    references: list[str] | None = None,
    reward_weights: dict[str, float] | None = None,
) -> dict[str, float]:
    """Compute average score for each T2G reward component directly.

    Calls the component reward functions in ``src.rewards.t2g_rewards``
    with plain strings — no global gold-gloss registry is used.  Gold
    references are passed in explicitly via ``references`` (same order
    as ``completions``).

    Gold-dependent components (translation quality, BLEU, gold
    structure, gloss order, verifier-scaled) are only computed when
    ``references`` is provided; otherwise they are skipped.  Structural
    components that no longer exist in the rewards module (e.g. the
    dense Viterbi proxies) are skipped gracefully via attribute lookup.

    Args:
        completions: Generated gloss sequences.
        references: Gold reference glosses (same order as ``completions``).
            ``None`` skips gold-dependent components.
        reward_weights: Optional dict mapping component name → weight.
            If provided, only components with weight > 0 are computed
            (others are skipped to save computation).

    Returns:
        Dict mapping component name → average score.
    """
    import src.rewards.t2g_rewards as rewards_mod

    gold_components: dict[str, Any] = {
        name: getattr(rewards_mod, name) for name in _GOLD_REWARD_COMPONENTS
    }
    free_components: dict[str, Any] = {
        name: getattr(rewards_mod, name) for name in _FREE_REWARD_COMPONENTS
    }
    for name in _OPTIONAL_FREE_REWARD_COMPONENTS:
        fn = getattr(rewards_mod, name, None)
        if fn is not None:
            free_components[name] = fn

    has_refs = bool(references) and len(references) == len(completions)

    # Only compute components with weight > 0 to save computation.
    active: set[str] | None = None
    if reward_weights is not None:
        active = {k for k, v in reward_weights.items() if v > 0}

    def _is_active(name: str) -> bool:
        """Return True if this component should be computed."""
        return active is None or name in active

    sums: dict[str, float] = {}
    counts: dict[str, int] = {}

    def _add(name: str, value: float) -> None:
        sums[name] = sums.get(name, 0.0) + value
        counts[name] = counts.get(name, 0) + 1

    for i, comp in enumerate(completions):
        gold = references[i] if references is not None and has_refs else ""
        for name, fn in gold_components.items():
            if _is_active(name) and has_refs:
                _add(name, fn(comp, gold))
        for name, fn in free_components.items():
            if _is_active(name):
                _add(name, fn(comp))

    return {name: sums[name] / counts[name] for name in counts if _is_active(name)}


# ---------------------------------------------------------------------------
# SacreBLEU-based metrics (BLEU, chrF2) and token-level gloss F1
# ---------------------------------------------------------------------------

_SACREBLEU_BLEU_METRIC: Any = None
_SACREBLEU_CHRF_METRIC: Any = None


def _get_sacrebleu_bleu() -> Any:
    """Lazily build the shared sacrebleu BLEU metric instance.

    Uses ``effective_order=True`` (so short gloss sequences are scored
    against the n-gram orders actually present instead of collapsing to
    0) and ``floor`` smoothing, matching the configuration used by
    ``bleu_reward`` in ``src.rewards.t2g_rewards``.
    """
    global _SACREBLEU_BLEU_METRIC
    if _SACREBLEU_BLEU_METRIC is None:
        import sacrebleu

        _SACREBLEU_BLEU_METRIC = sacrebleu.BLEU(
            effective_order=True,
            smooth_method="floor",
            smooth_value=0.1,
        )
    return _SACREBLEU_BLEU_METRIC


def _get_sacrebleu_chrf() -> Any:
    """Lazily build the shared sacrebleu CHRF metric instance (chrF2)."""
    global _SACREBLEU_CHRF_METRIC
    if _SACREBLEU_CHRF_METRIC is None:
        import sacrebleu

        _SACREBLEU_CHRF_METRIC = sacrebleu.CHRF(char_order=6, word_order=2)
    return _SACREBLEU_CHRF_METRIC


def bleu_sentence(generated: str, reference: str) -> float:
    """Sentence-level BLEU via sacrebleu, normalized to [0, 1].

    Args:
        generated: Generated gloss sequence (hypothesis).
        reference: Gold reference gloss sequence.

    Returns:
        BLEU score in [0, 1] (sacrebleu's 0-100 scale divided by 100).
    """
    gen = extract_gloss_text(generated)
    ref = reference.strip()
    if not gen or not ref:
        return 0.0
    return float(_get_sacrebleu_bleu().sentence_score(gen, [ref]).score) / 100.0


def bleu_corpus(generated: list[str], references: list[str]) -> float:
    """Corpus-level BLEU via sacrebleu, normalized to [0, 1].

    Aggregates n-gram matches across all sentence pairs before computing
    precision, which is more accurate than averaging sentence-level BLEU.

    Args:
        generated: List of generated gloss sequences.
        references: List of gold reference glosses (same order).

    Returns:
        Corpus BLEU score in [0, 1].
    """
    if not generated or not references:
        return 0.0
    hyps = [extract_gloss_text(g) for g in generated]
    # sacrebleu corpus_score expects a list of reference STREAMS, where each
    # stream is the full corpus translated once: refs = [[r1, r2, ..., rN]].
    # The previous format [[r1], [r2], ...] (one stream per sentence) made
    # sacrebleu treat the corpus as N "parallel references" of a single
    # sentence, degenerating to a near-sentence-level score on a tiny slice
    # (sys_len/ref_len ≈ one sentence) — e.g. corpus BLEU 0.87 with sentence
    # mean 0.18. Fixed 2026-08-29.
    refs = [[r.strip() for r in references]]
    return float(_get_sacrebleu_bleu().corpus_score(hyps, refs).score) / 100.0


def chrf_score(generated: str, reference: str) -> float:
    """Sentence-level chrF2 via sacrebleu (0-100 scale).

    chrF (character n-gram F-score, beta=2) is a standard MT metric
    robust to word-order errors and out-of-vocabulary tokens — a good
    complement to ROUGE-L and BLEU for short ASL gloss sequences.

    Args:
        generated: Generated gloss sequence (hypothesis).
        reference: Gold reference gloss sequence.

    Returns:
        chrF2 score in [0, 100] (sacrebleu convention).
    """
    gen = extract_gloss_text(generated)
    ref = reference.strip()
    if not gen or not ref:
        return 0.0
    return float(_get_sacrebleu_chrf().sentence_score(gen, [ref]).score)


def corpus_chrf(generated: list[str], references: list[str]) -> float:
    """Corpus-level chrF2 via sacrebleu (0-100 scale).

    Args:
        generated: List of generated gloss sequences.
        references: List of gold reference glosses (same order).

    Returns:
        Corpus chrF2 score in [0, 100].
    """
    if not generated or not references:
        return 0.0
    hyps = [extract_gloss_text(g) for g in generated]
    # Same fix as bleu_corpus: single reference STREAM (see comment there).
    # The previous [[r1],[r2],...] format degenerated to a tiny slice of
    # the corpus (chrF 96 with sentence mean 44 — impossible for a real
    # corpus). Fixed 2026-08-29.
    refs = [[r.strip() for r in references]]
    return float(_get_sacrebleu_chrf().corpus_score(hyps, refs).score)


def gloss_f1(generated: str, reference: str) -> float:
    """Compute token-level F1 between generated and gold gloss sequences.

    Precision/recall are computed over space-separated tokens with a
    case-insensitive (lowercased) comparison, so ``"WALK"`` and
    ``"walk"`` match.

    Args:
        generated: Generated gloss sequence.
        reference: Gold reference gloss sequence.

    Returns:
        F1 score in [0, 1] (0 = no overlap, 1 = identical).
    """
    gen_tokens = extract_gloss_text(generated).lower().split()
    ref_tokens = reference.strip().lower().split()
    if not gen_tokens or not ref_tokens:
        return 0.0
    gen_counts = Counter(gen_tokens)
    ref_counts = Counter(ref_tokens)
    overlap = sum((gen_counts & ref_counts).values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(gen_tokens)
    recall = overlap / len(ref_tokens)
    return float(2.0 * precision * recall / (precision + recall))


def corpus_gloss_f1(
    generated: list[str],
    references: list[str],
) -> dict[str, float]:
    """Compute corpus-level token F1: micro-averaged and sentence-mean.

    - ``"micro"`` aggregates token counts across all sentence pairs
      before computing precision/recall (one global F1).
    - ``"sentence_mean"`` averages the per-sentence F1 scores.

    Args:
        generated: List of generated gloss sequences.
        references: List of gold reference glosses (same order).

    Returns:
        Dict with keys ``"micro"`` and ``"sentence_mean"`` in [0, 1].
    """
    if not generated or not references:
        return {"micro": 0.0, "sentence_mean": 0.0}

    gen_counts_total: Counter[str] = Counter()
    ref_counts_total: Counter[str] = Counter()
    overlap_total = 0
    sentence_scores: list[float] = []

    for gen, ref in zip(generated, references):
        gen_counts = Counter(extract_gloss_text(gen).lower().split())
        ref_counts = Counter(ref.strip().lower().split())
        gen_counts_total.update(gen_counts)
        ref_counts_total.update(ref_counts)
        overlap = sum((gen_counts & ref_counts).values())
        overlap_total += overlap
        gen_len = sum(gen_counts.values())
        ref_len = sum(ref_counts.values())
        if gen_len and ref_len and overlap:
            precision = overlap / gen_len
            recall = overlap / ref_len
            sentence_scores.append(2.0 * precision * recall / (precision + recall))
        else:
            sentence_scores.append(0.0)

    total_gen = sum(gen_counts_total.values())
    total_ref = sum(ref_counts_total.values())
    if total_gen and total_ref and overlap_total:
        precision = overlap_total / total_gen
        recall = overlap_total / total_ref
        micro = 2.0 * precision * recall / (precision + recall)
    else:
        micro = 0.0

    return {
        "micro": float(micro),
        "sentence_mean": float(np.mean(sentence_scores)),
    }


def seeded_sample_indices(
    total: int,
    n: int | None,
    seed: int = 42,
) -> list[int]:
    """Return a reproducible random subset of indices for evaluation.

    When ``n`` is ``None`` (or ``n >= total``), returns all indices
    ``[0, total)`` in order.  Otherwise samples ``n`` distinct indices
    with a seeded RNG — NOT the first ``n`` of the dataset — so partial
    evaluations are unbiased and reproducible.

    Args:
        total: Total number of items (e.g. test-set size).
        n: Number of items to sample, or ``None`` for all.
        seed: Random seed for reproducible sampling.

    Returns:
        Sorted list of distinct indices in ``[0, total)``.
    """
    if n is None or n >= total:
        return list(range(total))
    rng = random.Random(seed)
    return sorted(rng.sample(range(total), n))


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
    - BLEU via sacrebleu (corpus + sentence mean with 95% CI, [0, 1])
    - chrF2 via sacrebleu (corpus + sentence mean with 95% CI, 0-100 scale)
    - Token-level gloss F1 (micro + sentence mean with 95% CI, [0, 1])
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
    bleu_scores = [bleu_sentence(c, r) for c, r in zip(completions, references)]
    chrf_scores = [chrf_score(c, r) for c, r in zip(completions, references)]
    gloss_f1_scores = [gloss_f1(c, r) for c, r in zip(completions, references)]
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
    chrf_mean, chrf_lo, chrf_hi = bootstrap_confidence_interval(
        chrf_scores, n_bootstrap
    )
    gloss_f1_mean, gloss_f1_lo, gloss_f1_hi = bootstrap_confidence_interval(
        gloss_f1_scores, n_bootstrap
    )
    pass_mean, pass_lo, pass_hi = bootstrap_confidence_interval(
        pass_scores, n_bootstrap
    )

    corpus_f1 = corpus_gloss_f1(completions, references)

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
            "corpus": bleu_corpus(completions, references),
            "sentence_mean": bleu_mean,
            "ci_95": [bleu_lo, bleu_hi],
        },
        "chrf": {
            "corpus": corpus_chrf(completions, references),
            "sentence_mean": chrf_mean,
            "ci_95": [chrf_lo, chrf_hi],
        },
        "gloss_f1": {
            "micro": corpus_f1["micro"],
            "sentence_mean": gloss_f1_mean,
            "ci_95": [gloss_f1_lo, gloss_f1_hi],
        },
        "pass_at_1": {
            "mean": pass_mean,
            "ci_95": [pass_lo, pass_hi],
        },
        "gloss_validity_rate": valid_count / max(total, 1),
        "error_distribution": dict(error_types.most_common(20)),
    }
