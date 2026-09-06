"""Validity-gated single rewards for text-to-gloss GRPO training."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import logging
import math
from collections import Counter
from collections.abc import Collection, Mapping, Sequence
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable

from src.utils.text_utils import extract_gloss_text

logger = logging.getLogger(__name__)

REWARD_NAMES = (
    "edit-validity",
    "token-f1-validity",
    "chrfpp-validity",
    "rouge-l-validity",
    "sbleu2-exp-validity",
)
REWARD_PROTOCOL_VERSION = "reward-ablation-v2"

_gloss_vocab: frozenset[str] = frozenset()
_warned_missing_gold = False


def initialize_rewards(vocab: list[str]) -> None:
    """Initialize the case-insensitive production vocabulary."""
    global _gloss_vocab, _warned_missing_gold
    if not isinstance(vocab, list):
        raise TypeError("initialize_rewards requires a gloss vocabulary list")
    _gloss_vocab = frozenset(token.casefold() for token in vocab)
    _warned_missing_gold = False


def _word_level_levenshtein(a: Sequence[str], b: Sequence[str]) -> int:
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


def _normalize_gloss_tokens(text: str, *, completion: bool = False) -> list[str]:
    value = extract_gloss_text(text) if completion else text
    return value.casefold().split()


def _normalized_edit_similarity(a: Sequence[str], b: Sequence[str]) -> float:
    longest = max(len(a), len(b))
    return 1.0 if longest == 0 else 1.0 - _word_level_levenshtein(a, b) / longest


def _token_f1(a: Sequence[str], b: Sequence[str]) -> float:
    overlap = sum((Counter(a) & Counter(b)).values())
    if not overlap:
        return 0.0
    precision, recall = overlap / len(a), overlap / len(b)
    return 2.0 * precision * recall / (precision + recall)


def _lcs_length(a: Sequence[str], b: Sequence[str]) -> int:
    previous = [0] * (len(b) + 1)
    for left in a:
        current = [0]
        for column, right in enumerate(b, start=1):
            current.append(
                previous[column - 1] + 1
                if left == right
                else max(previous[column], current[-1])
            )
        previous = current
    return previous[-1]


def _rouge_l_f1(a: Sequence[str], b: Sequence[str]) -> float:
    overlap = _lcs_length(a, b)
    return 0.0 if not overlap else 2.0 * overlap / (len(a) + len(b))


@lru_cache(maxsize=1)
def _chrfpp_metric() -> Any:
    import sacrebleu

    # whitespace=False is explicit and frozen for reproducibility.
    return sacrebleu.CHRF(
        char_order=6, word_order=2, beta=2, whitespace=False, eps_smoothing=False
    )


@lru_cache(maxsize=1)
def _sbleu2_metric() -> Any:
    import sacrebleu

    return sacrebleu.BLEU(
        max_ngram_order=2,
        effective_order=True,
        smooth_method="exp",
        tokenize="none",
    )


def reward_protocol() -> dict[str, Any]:
    """Return the exact scientific reward protocol used by probe and trainer."""
    settings = {
        "common": {
            "completion_normalization": "extract-gloss+casefold+split",
            "reference_normalization": "casefold+split",
            "validity_gate": "nonempty-hypothesis+nonempty-reference+all-hypothesis-tokens-in-vocab",
            "transform": "2*similarity-1",
            "invalid_score_default": -1.0,
        },
        "edit-validity": {"metric": "token-levenshtein/max-length"},
        "token-f1-validity": {
            "metric": "clipped-multiset-f1",
            "order_sensitive": False,
        },
        "chrfpp-validity": {
            "metric": "sacrebleu-chrf",
            "char_order": 6,
            "word_order": 2,
            "beta": 2,
            "whitespace": False,
            "eps_smoothing": False,
        },
        "rouge-l-validity": {"metric": "token-lcs-f1", "use_stemmer": False},
        "sbleu2-exp-validity": {
            "metric": "sacrebleu-sentence-bleu",
            "max_ngram_order": 2,
            "effective_order": True,
            "smooth_method": "exp",
            "tokenize": "none",
        },
    }
    canonical = json.dumps(settings, sort_keys=True, separators=(",", ":"))
    source_root = Path(__file__).parents[2]
    implementation_files = (
        "src/rewards/t2g_rewards.py",
        "src/utils/text_utils.py",
    )
    implementation_hasher = hashlib.sha256()
    for relative_path in implementation_files:
        implementation_hasher.update(relative_path.encode("utf-8"))
        implementation_hasher.update(b"\0")
        implementation_hasher.update((source_root / relative_path).read_bytes())
        implementation_hasher.update(b"\0")
    chrfpp, sbleu2 = _chrfpp_metric(), _sbleu2_metric()
    chrfpp.sentence_score("A", ["A"])
    sbleu2.sentence_score("A", ["A"])
    return {
        "version": REWARD_PROTOCOL_VERSION,
        "settings": settings,
        "settings_sha256": hashlib.sha256(canonical.encode()).hexdigest(),
        "implementation_provenance": {
            "files": list(implementation_files),
            "sha256": implementation_hasher.hexdigest(),
        },
        "sacrebleu_version": importlib.metadata.version("sacrebleu"),
        "sacrebleu_signatures": {
            "chrfpp": str(chrfpp.get_signature()),
            "sbleu2_exp": str(sbleu2.get_signature()),
        },
    }


def _similarity(name: str, generated: Sequence[str], gold: Sequence[str]) -> float:
    if name == "edit-validity":
        return _normalized_edit_similarity(generated, gold)
    if name == "token-f1-validity":
        return _token_f1(generated, gold)
    if name == "rouge-l-validity":
        return _rouge_l_f1(generated, gold)
    hypothesis, reference = " ".join(generated), " ".join(gold)
    if name == "chrfpp-validity":
        return (
            float(_chrfpp_metric().sentence_score(hypothesis, [reference]).score) / 100
        )
    if name == "sbleu2-exp-validity":
        return (
            float(_sbleu2_metric().sentence_score(hypothesis, [reference]).score) / 100
        )
    raise ValueError(f"Unknown production reward: {name!r}")


def score_validity_reward(
    name: str,
    completion: str,
    gold_gloss: str,
    vocab: Collection[str],
    invalid_score: float = -1.0,
) -> float:
    """Apply the shared validity gate, then map similarity from [0, 1] to [-1, 1]."""
    if name not in REWARD_NAMES:
        raise ValueError(f"Unknown production reward: {name!r}")
    generated = _normalize_gloss_tokens(completion, completion=True)
    gold = _normalize_gloss_tokens(gold_gloss)
    if not generated or not gold or any(token not in vocab for token in generated):
        return invalid_score
    similarity = min(1.0, max(0.0, _similarity(name, generated, gold)))
    return 2.0 * similarity - 1.0


def _edit_validity_score(
    completion: str,
    gold_gloss: str,
    vocab: Collection[str],
    invalid_score: float = -1.0,
) -> float:
    normalized_vocab = frozenset(token.casefold() for token in vocab)
    return score_validity_reward(
        "edit-validity", completion, gold_gloss, normalized_vocab, invalid_score
    )


def _initialized_reward(
    name: str, completion: str, gold_gloss: str, invalid_score: float
) -> float:
    if not _gloss_vocab:
        return invalid_score
    return score_validity_reward(
        name, completion, gold_gloss, _gloss_vocab, invalid_score
    )


def edit_validity_reward(
    completion: str, gold_gloss: str, invalid_score: float = -1.0
) -> float:
    return _initialized_reward("edit-validity", completion, gold_gloss, invalid_score)


def token_f1_validity_reward(
    completion: str, gold_gloss: str, invalid_score: float = -1.0
) -> float:
    return _initialized_reward(
        "token-f1-validity", completion, gold_gloss, invalid_score
    )


def chrfpp_validity_reward(
    completion: str, gold_gloss: str, invalid_score: float = -1.0
) -> float:
    return _initialized_reward("chrfpp-validity", completion, gold_gloss, invalid_score)


def rouge_l_validity_reward(
    completion: str, gold_gloss: str, invalid_score: float = -1.0
) -> float:
    return _initialized_reward(
        "rouge-l-validity", completion, gold_gloss, invalid_score
    )


def sbleu2_exp_validity_reward(
    completion: str, gold_gloss: str, invalid_score: float = -1.0
) -> float:
    return _initialized_reward(
        "sbleu2-exp-validity", completion, gold_gloss, invalid_score
    )


_REWARDS: dict[str, Callable[[str, str, float], float]] = {
    "edit-validity": edit_validity_reward,
    "token-f1-validity": token_f1_validity_reward,
    "chrfpp-validity": chrfpp_validity_reward,
    "rouge-l-validity": rouge_l_validity_reward,
    "sbleu2-exp-validity": sbleu2_exp_validity_reward,
}


def _make_gloss_reward_fn(
    component_fn: Callable[[str, str], float],
) -> Callable[..., list[float]]:
    """Adapt a scalar reward to TRL strings or conversational completions.

    Conversational completions use the final message's string ``content``.
    Malformed conversational values receive the configured invalid score via
    an empty completion rather than raising inside the training loop.
    """

    def completion_text(completion: Any) -> str:
        if isinstance(completion, str):
            return completion
        if not isinstance(completion, list) or not completion:
            return ""
        final_message = completion[-1]
        if not isinstance(final_message, Mapping):
            return ""
        content = final_message.get("content")
        return content if isinstance(content, str) else ""

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
            text = completion_text(completion)
            gold = (
                gold_gloss[index]
                if gold_gloss is not None and index < len(gold_gloss)
                else ""
            )
            if not str(gold).strip() and not _warned_missing_gold:
                logger.warning("gold_gloss is missing; returning invalid reward")
                _warned_missing_gold = True
            scores.append(component_fn(text, str(gold)))
        return scores

    reward_fn.__name__ = component_fn.__name__
    return reward_fn


def build_t2g_reward_functions(
    reward_config: dict[str, Any] | None = None,
) -> tuple[list[Callable[..., list[float]]], list[float]]:
    """Build exactly one configured production reward."""
    if reward_config is not None and not isinstance(reward_config, Mapping):
        raise TypeError("reward_config must be a mapping or None")
    config = {"name": "edit-validity"} if reward_config is None else reward_config
    unsupported = set(config) - {"name", "invalid_score"}
    if unsupported:
        raise ValueError(f"Unsupported production reward fields: {sorted(unsupported)}")
    name = config.get("name")
    if name not in _REWARDS:
        raise ValueError(f"Unknown production reward: {name!r}")
    try:
        invalid_score = float(config.get("invalid_score", -1.0))
    except (TypeError, ValueError) as error:
        raise ValueError(
            "invalid_score must be a finite number within [-1, 1]"
        ) from error
    if not math.isfinite(invalid_score) or not -1.0 <= invalid_score <= 1.0:
        raise ValueError("invalid_score must be a finite number within [-1, 1]")
    implementation = _REWARDS[name]

    def configured_reward(completion: str, gold_gloss: str) -> float:
        return implementation(completion, gold_gloss, invalid_score)

    configured_reward.__name__ = implementation.__name__
    return [_make_gloss_reward_fn(configured_reward)], [1.0]
