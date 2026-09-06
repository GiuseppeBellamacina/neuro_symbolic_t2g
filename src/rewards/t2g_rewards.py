"""Production reward for text-to-gloss training."""

from __future__ import annotations

import logging
from collections.abc import Collection
from typing import Any, Callable

from src.utils.text_utils import extract_gloss_text

logger = logging.getLogger(__name__)

_gloss_vocab: frozenset[str] = frozenset()
_warned_missing_gold = False


def initialize_rewards(vocab: list[str]) -> None:
    """Initialize the case-insensitive production vocabulary."""
    global _gloss_vocab, _warned_missing_gold
    if not isinstance(vocab, list):
        raise TypeError("initialize_rewards requires a gloss vocabulary list")
    _gloss_vocab = frozenset(token.casefold() for token in vocab)
    _warned_missing_gold = False


def _word_level_levenshtein(a: list[str], b: list[str]) -> int:
    """Return word-level Levenshtein distance using one DP row."""
    if not a:
        return len(b)
    if not b:
        return len(a)

    previous = list(range(len(b) + 1))
    for row, left in enumerate(a, start=1):
        current = [row]
        for column, right in enumerate(b, start=1):
            current.append(
                min(
                    previous[column] + 1,
                    current[column - 1] + 1,
                    previous[column - 1] + (left != right),
                )
            )
        previous = current
    return previous[-1]


def _normalize_gloss_tokens(text: str) -> list[str]:
    """Casefold and split gloss text."""
    return text.casefold().split()


def _normalized_edit_similarity(a: list[str], b: list[str]) -> float:
    """Return word-level edit similarity in ``[0, 1]``."""
    longest = max(len(a), len(b))
    if longest == 0:
        return 1.0
    return 1.0 - _word_level_levenshtein(a, b) / longest


def _edit_validity_score(
    completion: str,
    gold_gloss: str,
    vocab: Collection[str],
    invalid_score: float = -1.0,
) -> float:
    """Return validity-gated edit similarity in ``[-1, 1]``."""
    generated = _normalize_gloss_tokens(extract_gloss_text(completion))
    gold = _normalize_gloss_tokens(gold_gloss)
    normalized_vocab = {token.casefold() for token in vocab}
    if (
        not generated
        or not gold
        or any(token not in normalized_vocab for token in generated)
    ):
        return invalid_score
    return 2.0 * _normalized_edit_similarity(generated, gold) - 1.0


def edit_validity_reward(
    completion: str,
    gold_gloss: str,
    invalid_score: float = -1.0,
) -> float:
    """Score a completion against the initialized vocabulary and reference."""
    if not _gloss_vocab:
        return invalid_score
    return _edit_validity_score(completion, gold_gloss, _gloss_vocab, invalid_score)


def _make_gloss_reward_fn(
    component_fn: Callable[[str, str], float],
) -> Callable[..., list[float]]:
    """Adapt a single-output reward to the GRPO reward signature."""

    def reward_fn(
        completions: list[Any],
        prompts: list[Any] | None = None,
        *,
        gold_gloss: list[str] | None = None,
        **kwargs: Any,
    ) -> list[float]:
        del prompts, kwargs
        global _warned_missing_gold

        scores: list[float] = []
        for index, completion in enumerate(completions):
            text = (
                completion[0]["content"]
                if isinstance(completion, list)
                else str(completion)
            )
            gold = (
                gold_gloss[index]
                if gold_gloss is not None and index < len(gold_gloss)
                else ""
            )
            if not str(gold).strip():
                if not _warned_missing_gold:
                    logger.warning("gold_gloss is missing; returning neutral reward")
                    _warned_missing_gold = True
                scores.append(0.0)
            else:
                scores.append(component_fn(text, str(gold)))
        return scores

    reward_fn.__name__ = component_fn.__name__
    return reward_fn


def build_t2g_reward_functions(
    reward_config: dict[str, Any] | None = None,
) -> tuple[list[Callable[..., list[float]]], list[float]]:
    """Build the sole production reward and its fixed weight."""
    config = {"name": "edit-validity"} if reward_config is None else reward_config
    unsupported = set(config) - {"name", "invalid_score"}
    if unsupported:
        raise ValueError(f"Unsupported production reward fields: {sorted(unsupported)}")
    if config.get("name") != "edit-validity":
        raise ValueError(f"Unknown production reward: {config.get('name')!r}")

    invalid_score = float(config.get("invalid_score", -1.0))

    def configured_reward(completion: str, gold_gloss: str) -> float:
        return edit_validity_reward(completion, gold_gloss, invalid_score)

    configured_reward.__name__ = "edit_validity_reward"
    return [_make_gloss_reward_fn(configured_reward)], [1.0]
