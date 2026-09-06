"""Evaluation metrics for T2G gloss generation.

Computes text-quality metrics, production reward summaries, seeded
evaluation sampling, and completion validity statistics.
"""

from __future__ import annotations

import random
import re
from collections import Counter
from math import comb
from typing import Any

import numpy as np
from rouge_score import rouge_scorer

from src.utils.text_utils import extract_gloss_text

# Version of the metric definitions. Bump whenever a metric formula changes
# (e.g. v2 = corpus BLEU/chrF reference-format fix). Eval results carry this
# stamp so cached baselines computed with older definitions are detected and
# recomputed instead of silently compared against new ones.
METRICS_VERSION = 3


def normalize_gloss(text: Any) -> str:
    """Extract gloss text, casefold it, and collapse whitespace."""
    return " ".join(extract_gloss_text(str(text or "")).casefold().split())


def normalized_exact_match(generated: Any, reference: Any) -> float:
    """Case-insensitive exact match after gloss extraction/space collapse."""
    return float(normalize_gloss(generated) == normalize_gloss(reference))


def normalized_edit_similarity(generated: Any, reference: Any) -> float:
    """Normalized token Levenshtein similarity in [0, 1]."""
    left, right = normalize_gloss(generated).split(), normalize_gloss(reference).split()
    if not left and not right:
        return 1.0
    previous = list(range(len(right) + 1))
    for i, lhs in enumerate(left, 1):
        current = [i]
        for j, rhs in enumerate(right, 1):
            current.append(
                min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (lhs != rhs))
            )
        previous = current
    return 1.0 - previous[-1] / max(len(left), len(right), 1)


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

    from src.rewards.t2g_rewards import _gloss_vocab

    normalized_tokens = [token.casefold() for token in tokens]
    if _gloss_vocab and any(token not in _gloss_vocab for token in normalized_tokens):
        return False, "out_of_vocab_tokens"

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
    if len(normalized_tokens) > 4:
        unique_ratio = len(set(normalized_tokens)) / len(normalized_tokens)
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
    """Compute the standard Pass@k estimator averaged over prompts.

    Success is explicitly ROUGE-L >= ``threshold`` AND lexical validity.
    For each prompt with ``n`` samples and ``c`` successes, the estimate is
    ``1 - C(n-c,k) / C(n,k)``. All available samples determine ``c``.

    Args:
        completions_per_prompt: For each prompt, a list of k completions.
        references: Gold reference glosses (one per prompt).
        k_values: Which k values to compute.
        threshold: ROUGE-L pass threshold.

    Returns:
        Dict like {"pass@1": 0.72, "pass@5": 0.88, "pass@10": 0.93}.
    """
    if len(completions_per_prompt) != len(references):
        raise ValueError("compute_pass_at_k requires aligned prompt groups/references")
    n_prompts = len(completions_per_prompt)
    results: dict[str, float] = {}

    for k in k_values:
        estimates: list[float] = []
        for comps, ref in zip(completions_per_prompt, references):
            n = len(comps)
            if k <= 0 or k > n:
                raise ValueError(f"pass@k requires 1 <= k <= n, got k={k}, n={n}")
            c = sum(
                check_gloss_validity(comp)[0] and rouge_l_score(comp, ref) >= threshold
                for comp in comps
            )
            estimates.append(1.0 if n - c < k else 1.0 - comb(n - c, k) / comb(n, k))
        results[f"pass@{k}"] = float(np.mean(estimates)) if n_prompts else 0.0

    return results


# ---------------------------------------------------------------------------
# Group diagnostics (preregistered instrumentation over G completions/prompt)
# ---------------------------------------------------------------------------


def normalize_gloss_diagnostics_text(text: Any) -> str:
    """Normalize a completion/reference for group diagnostics.

    Canonical pipeline (preregistered): ``extract_gloss_text`` (strips
    think tags / code fences) → ``casefold`` → whitespace collapse.

    Args:
        text: Raw completion or reference text (coerced to ``str``).

    Returns:
        The normalized gloss string (``""`` for empty).
    """
    return normalize_gloss(text)


def gloss_word_count(text: Any) -> int:
    """Number of word tokens after :func:`normalize_gloss_diagnostics_text`.

    Args:
        text: Raw completion or reference text.

    Returns:
        Token count (0 for an empty normalization).
    """
    normalized = normalize_gloss_diagnostics_text(text)
    return len(normalized.split()) if normalized else 0


def abs_length_error(completion: Any, reference: Any) -> float:
    """Absolute word-length error of one completion vs its gold reference.

    Both sides are normalized with the shared diagnostics pipeline before
    counting words.

    Args:
        completion: Raw model completion.
        reference: Raw gold reference gloss.

    Returns:
        ``|word_count(completion) - word_count(reference)|`` as a float.
    """
    return float(abs(gloss_word_count(completion) - gloss_word_count(reference)))


def mean_abs_length_error(
    completions: list[Any],
    references: list[Any],
) -> float:
    """Mean absolute word-length error over aligned completion/reference pairs.

    The canonical single-number companion of the per-group training
    diagnostic: averaged over the SAME flat completion-reference pairs as
    the other primary eval metrics.

    Args:
        completions: Generated gloss sequences (flat, one per completion).
        references: Gold reference glosses (same order and length).

    Returns:
        Mean ``abs_length_error`` over all pairs.

    Raises:
        ValueError: If the lists are empty or misaligned (loud failure —
            never silently averaged over a subset).
    """
    if len(completions) != len(references):
        raise ValueError(
            "mean_abs_length_error: misaligned inputs — "
            f"{len(completions)} completions vs {len(references)} references"
        )
    if not completions:
        raise ValueError("mean_abs_length_error: empty completion list")
    errors = [abs_length_error(comp, ref) for comp, ref in zip(completions, references)]
    return float(np.mean(errors))


def compute_group_diagnostics(
    completions: list[Any],
    references: list[Any] | None,
    group_size: int,
) -> dict[str, float]:
    """Compute per-group diagnostics over G completions aligned per prompt.

    The input must be the flat GRPO rollout batch: ``len(completions)``
    completions where every consecutive block of ``group_size`` entries
    belongs to ONE prompt.  Over each block:

    - unique normalized outputs (``extract_gloss_text`` → casefold →
      whitespace collapse), averaged over groups → ``unique_outputs_mean``
    - absolute word-length error vs gold (when ``references`` given),
      averaged over completions → ``abs_length_error_mean``

    Malformed input fails LOUDLY (ValueError) rather than inferring wrong
    groups: a non-multiple-of-G batch, a misaligned reference list, an
    empty batch or a non-positive ``group_size`` is a caller bug (e.g. a
    TRL grouping change) and must be surfaced, not silently aggregated.
    Callers that prefer to skip may catch the error and log the reason.

    Args:
        completions: Flat completion list, G per prompt, contiguous.
        references: Gold glosses aligned per completion (same length), or
            ``None`` to compute only ``unique_outputs_mean``.
        group_size: G — completions per prompt.

    Returns:
        ``{"unique_outputs_mean": float}`` plus
        ``{"abs_length_error_mean": float}`` when ``references`` is given.

    Raises:
        ValueError: On empty/misaligned input or a batch whose length is
            not a multiple of ``group_size``.
    """
    g = int(group_size)
    if g <= 0:
        raise ValueError(
            f"compute_group_diagnostics: invalid group_size={group_size!r}"
        )
    n = len(completions)
    if n == 0:
        raise ValueError("compute_group_diagnostics: empty completion batch")
    if references is not None and len(references) != n:
        raise ValueError(
            "compute_group_diagnostics: misaligned inputs — "
            f"{n} completions vs {len(references)} references"
        )
    if n % g != 0:
        raise ValueError(
            "compute_group_diagnostics: batch length "
            f"{n} is not a multiple of group_size={g} — refusing to "
            "infer wrong groups"
        )

    unique_counts: list[int] = []
    length_errors: list[float] = []
    for start in range(0, n, g):
        group = completions[start : start + g]
        normalized = {normalize_gloss_diagnostics_text(c) for c in group}
        unique_counts.append(len(normalized))
        if references is not None:
            group_refs = references[start : start + g]
            length_errors.extend(
                abs_length_error(c, r) for c, r in zip(group, group_refs)
            )

    diagnostics = {
        "unique_outputs_mean": float(np.mean(unique_counts)),
    }
    if references is not None:
        diagnostics["abs_length_error_mean"] = float(np.mean(length_errors))
    return diagnostics


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
# Production reward breakdown
# ---------------------------------------------------------------------------


def compute_reward_breakdown(
    completions: list[str],
    references: list[str] | None = None,
    reward_weights: dict[str, float] | None = None,
    reward_config: dict[str, Any] | None = None,
) -> dict[str, float]:
    """Average the configured optimized reward plus common edit diagnostic."""
    from src.rewards.t2g_rewards import build_t2g_reward_functions, edit_validity_reward

    if references is None:
        return {}
    if len(references) != len(completions):
        raise ValueError("compute_reward_breakdown requires aligned references")
    functions, _ = build_t2g_reward_functions(reward_config)
    optimized = functions[0]
    name = optimized.__name__
    if reward_weights is not None and reward_weights.get(name, 1.0) <= 0.0:
        return {}
    if not completions:
        return {name: 0.0, "edit_validity_diagnostic": 0.0}
    optimized_scores = optimized(completions, gold_gloss=references)
    edit_scores = [
        edit_validity_reward(comp, gold) for comp, gold in zip(completions, references)
    ]
    return {
        name: float(np.mean(optimized_scores)),
        "edit_validity_diagnostic": float(np.mean(edit_scores)),
    }


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
    the evaluation configuration used throughout this module.
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


def sacrebleu_signatures() -> dict[str, str]:
    """Return reproducibility signatures for the configured corpus metrics."""
    return {
        "bleu": str(_get_sacrebleu_bleu().get_signature()),
        "chrf": str(_get_sacrebleu_chrf().get_signature()),
    }


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
    if len(generated) != len(references):
        raise ValueError("bleu_corpus requires equal hypothesis/reference lengths")
    if not generated:
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
    if len(generated) != len(references):
        raise ValueError("corpus_chrf requires equal hypothesis/reference lengths")
    if not generated:
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
    if len(generated) != len(references):
        raise ValueError("corpus_gloss_f1 requires equal hypothesis/reference lengths")
    if not generated:
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
