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

# ---------------------------------------------------------------------------
# Gloss validity check
# ---------------------------------------------------------------------------


def _extract_gloss_text(completion: str) -> str:
    """Extract clean gloss tokens from a model completion."""
    text = re.sub(r"<think>.*?</think>", "", completion, flags=re.DOTALL).strip()
    m = re.search(r"```(?:gloss)?\s*(.*?)```", text, re.DOTALL)
    if m:
        text = m.group(1).strip()
    return text


def check_gloss_validity(completion: str) -> tuple[bool, str]:
    """Check if a completion is a valid gloss sequence.

    Returns:
        (is_valid, error_message) — error_message is "" if valid.
    """
    text = _extract_gloss_text(completion)
    if not text:
        return False, "empty_output"

    tokens = text.split()

    # Check for free-text patterns (English words)
    free_text_patterns = [
        r"\b(the|a|an|is|are|was|were|will|would|should|can|could)\b",
        r"\b(in|on|at|by|for|with|from|to|of|and|or|but)\b",
        r"[.,!?;:]",
        r"```",
        r"\{|\}",
    ]
    for pattern in free_text_patterns:
        if re.search(pattern, text, re.IGNORECASE):
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
    gen = _extract_gloss_text(generated)
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
        1 for c, r in zip(completions, references)
        if rouge_l_score(c, r) >= threshold
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

    Returns:
        Dict mapping component name → average score.
    """
    from src.rewards.t2g_rewards import (
        _lookup_gold_gloss,
        _lookup_gold_gloss_by_id,
        gloss_format_reward,
        gloss_repetition_reward,
        structural_dense_reward,
        translation_quality_reward,
    )

    n = len(completions)
    sums = {
        "translation_quality_reward": 0.0,
        "structural_dense_reward": 0.0,
        "gloss_format_reward": 0.0,
        "gloss_repetition_reward": 0.0,
    }

    for i, comp in enumerate(completions):
        gold = ""
        if sample_ids and i < len(sample_ids) and sample_ids[i]:
            gold = _lookup_gold_gloss_by_id(sample_ids[i])
        elif prompts and i < len(prompts):
            gold = _lookup_gold_gloss(prompts[i])
        sums["translation_quality_reward"] += translation_quality_reward(comp, gold)
        sums["structural_dense_reward"] += structural_dense_reward(comp, normalize=True)
        sums["gloss_format_reward"] += gloss_format_reward(comp)
        sums["gloss_repetition_reward"] += gloss_repetition_reward(comp)

    return {k: v / max(n, 1) for k, v in sums.items()}
