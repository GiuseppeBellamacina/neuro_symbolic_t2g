"""Completion sample logging for the sole production reward."""

from __future__ import annotations

import pytest

from src.rewards.t2g_rewards import build_t2g_reward_functions
from src.training.callbacks import CompletionSampleLogger


def _edit_validity_logger(n_samples: int = 3) -> CompletionSampleLogger:
    """Build a logger around the production reward function."""
    reward_fns, reward_weights = build_t2g_reward_functions()
    return CompletionSampleLogger(reward_fns, reward_weights, n_samples=n_samples)


def test_sample_logger_preserves_gold_alignment_and_breakdown(reward_setup):
    logger = _edit_validity_logger()
    completions = ["IX MAN WALK HOUSE", "DOG CAT"]
    gold_glosses = ["IX MAN WALK HOUSE", "DOG CAT HOUSE"]

    logger._capture(completions, prompts=None, gold_gloss=gold_glosses)

    assert [sample["gold"] for sample in logger._buffer] == gold_glosses
    assert [set(sample["breakdown"]) for sample in logger._buffer] == [
        {"edit_validity_reward"},
        {"edit_validity_reward"},
    ]
    assert logger._buffer[0]["breakdown"]["edit_validity_reward"] == 1.0
    assert logger._buffer[1]["breakdown"]["edit_validity_reward"] == pytest.approx(
        1 / 3
    )

    formatted = logger.format_samples()
    assert "COMPLETION SAMPLES" in formatted
    assert "GOLD:" in formatted
    assert "IX MAN WALK HOUSE" in formatted
    assert "edit_validity_reward" in formatted


def test_sample_logger_without_gold_uses_invalid_score(reward_setup):
    logger = _edit_validity_logger(n_samples=1)
    logger._capture(["IX MAN WALK HOUSE"], prompts=None, gold_gloss=None)

    sample = logger._buffer[0]
    assert sample["gold"] == ""
    assert sample["breakdown"] == {"edit_validity_reward": -1.0}
    assert "gold non disponibile" in logger.format_samples()


def test_difficulty_roundtrips_from_producer_to_monitor(reward_setup):
    from src.utils.chain_monitor import _extract_completion_samples

    logger = _edit_validity_logger(n_samples=1)
    prompt = [{"role": "user", "content": "The man walks."}]
    logger.set_difficulty_map([{"prompt": prompt, "difficulty": "hard"}])
    logger._capture(["IX MAN WALK"], prompts=[prompt], gold_gloss=["IX MAN WALK"])

    produced = logger.format_samples()
    assert "Sample 1  [difficulty=hard] [✓]" in produced
    parsed = "\n".join(_extract_completion_samples(produced.splitlines()))
    assert "[hard]" in parsed
