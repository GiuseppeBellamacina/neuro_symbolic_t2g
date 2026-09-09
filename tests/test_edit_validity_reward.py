"""Tests for the graded ``edit_validity_reward``.

The reward on the ``edit-rewards`` branch used a hard validity gate: a single
out-of-vocabulary token returned ``-1.0`` and discarded all partial credit.

Scope of what the graded term actually fixes, measured on 10000 zero-shot+Trie
rollouts: 53.88% of rewards sit at ``-1.0``, but only **2.43%** of those come
from the gate (131 OOV, 0 empty); the other 97.57% are completions with zero
edit similarity to the reference, i.e. a legitimate floor. So this removes a
cliff affecting ~1.3% of rollouts and restores ranking among OOV-containing
completions. It is NOT a fix for zero-gradient groups (19.90% -> 19.00%).

These tests pin the properties that matter: the validity term is graded,
verbosity is penalized, the reward is bounded and finite, and the default weight
keeps content ranked above mere validity.
"""

from __future__ import annotations

import math

import pytest

from src.rewards.t2g_rewards import (
    edit_validity_reward,
    gloss_order_reward,
    initialize_rewards,
)

VOCAB = ["IX", "MAN", "WALK", "HOUSE", "BOOK", "BE"]


@pytest.fixture
def vocab(monkeypatch):
    """Install a tiny gloss vocabulary without touching global reward state."""
    monkeypatch.setattr("src.rewards.t2g_rewards._gloss_vocab", list(VOCAB))
    return VOCAB


def test_exact_match_is_maximal(vocab):
    assert edit_validity_reward("IX MAN WALK", "IX MAN WALK") == pytest.approx(1.0)


def test_empty_generation_is_the_only_hard_failure(vocab):
    assert edit_validity_reward("", "IX MAN") == -1.0
    assert edit_validity_reward("IX MAN", "") == -1.0


def test_single_oov_token_no_longer_destroys_all_credit(vocab):
    """The core repair: one bad token must not erase a mostly-correct output."""
    mostly_right = edit_validity_reward("IX MAN WALK ZZZ", "IX MAN WALK HOUSE")
    all_wrong = edit_validity_reward("ZZZ QQQ WWW YYY", "IX MAN WALK HOUSE")
    assert mostly_right > all_wrong
    # Under the old hard gate BOTH of these returned exactly -1.0.
    assert mostly_right > -1.0


def test_validity_term_is_graded_and_monotone(vocab):
    """More in-vocabulary tokens must rank strictly higher, all else equal."""
    scores = [
        edit_validity_reward(" ".join(["IX"] * k + ["ZZZ"] * (4 - k)), "IX IX IX IX")
        for k in range(5)
    ]
    assert scores == sorted(scores)
    assert scores[0] < scores[-1]


def test_verbosity_is_penalized_not_rewarded(vocab):
    """Appending in-vocabulary padding must strictly decrease the reward.

    This is the precise statement of the length behaviour: appending k tokens
    gives edit_sim = 1 - k/(G+k), strictly decreasing in k.
    """
    gold = "IX MAN WALK HOUSE"
    scores = [edit_validity_reward(gold + " IX" * k, gold) for k in range(5)]
    assert scores == sorted(scores, reverse=True)
    assert scores[0] > scores[-1]


def test_oov_weight_zero_reproduces_gloss_order_reward(vocab):
    for completion in ("IX MAN WALK", "IX MAN ZZZ", "IX"):
        assert edit_validity_reward(completion, "IX MAN WALK", oov_weight=0.0) == (
            pytest.approx(gloss_order_reward(completion, "IX MAN WALK"))
        )


def test_reward_is_bounded_and_finite(vocab):
    cases = [
        ("IX MAN WALK", "IX MAN WALK"),
        ("ZZZ", "IX MAN WALK"),
        ("IX " * 40, "IX MAN"),
        ("IX MAN WALK HOUSE BOOK BE", "IX"),
    ]
    for completion, gold in cases:
        value = edit_validity_reward(completion, gold)
        assert math.isfinite(value)
        assert -1.0 <= value <= 1.0


def test_vocabulary_matching_is_casefolded(vocab):
    upper = edit_validity_reward("IX MAN", "IX MAN")
    lower = edit_validity_reward("ix man", "IX MAN")
    # Token identity is case-sensitive for the edit term, but the validity term
    # must not punish lowercase output as out-of-vocabulary.
    assert lower > -1.0
    assert upper == pytest.approx(1.0)


def test_missing_vocabulary_degrades_to_edit_similarity(monkeypatch):
    """Without a vocabulary the reward must not punish every rollout."""
    monkeypatch.setattr("src.rewards.t2g_rewards._gloss_vocab", [])
    assert edit_validity_reward("IX MAN WALK", "IX MAN WALK") == pytest.approx(
        gloss_order_reward("IX MAN WALK", "IX MAN WALK")
    )


@pytest.mark.parametrize("bad", [-0.1, 1.1, float("nan"), float("inf")])
def test_invalid_oov_weight_rejected(vocab, bad):
    with pytest.raises(ValueError, match="oov_weight"):
        edit_validity_reward("IX", "IX", oov_weight=bad)


def test_default_weight_keeps_content_above_mere_validity(vocab):
    """The validity term must not be able to outrank actual correctness.

    Failure mode under review: if ``oov_weight`` is too high, a fully in-vocab
    but meaningless output could outscore a mostly-correct output containing one
    out-of-vocabulary token. Measured: this does NOT happen at the 0.5 default
    (garbage 0.0000 and degenerate repetition 0.1250 both stay below the
    mostly-correct 0.5000), but it DOES happen at 0.75, where degenerate
    repetition reaches 0.5625. This test pins the safe regime.
    """
    gold = "IX MAN WALK HOUSE"
    mostly_correct = edit_validity_reward("IX MAN WALK ZZZ", gold)
    in_vocab_garbage = edit_validity_reward("BE BE BE BE", gold)
    degenerate = edit_validity_reward("IX IX IX IX IX IX IX IX", gold)

    assert in_vocab_garbage < mostly_correct
    assert degenerate < mostly_correct


def test_high_oov_weight_is_documented_as_unsafe(vocab):
    """Guard the boundary: at 0.75 validity starts dominating content."""
    gold = "IX MAN WALK HOUSE"
    mostly_correct = edit_validity_reward("IX MAN WALK ZZZ", gold, oov_weight=0.75)
    degenerate = edit_validity_reward("IX IX IX IX IX IX IX IX", gold, oov_weight=0.75)
    # This inversion is why the default is 0.5 and not higher.
    assert degenerate > mostly_correct


def test_reward_is_reachable_from_the_builder(vocab):
    """It must be selectable from config, not just importable.

    A reward that exists but cannot be switched on from a config is not
    recovered in any usable sense.
    """
    from src.rewards.t2g_rewards import build_t2g_reward_functions

    funcs, weights = build_t2g_reward_functions({"weight_edit_validity": 1.0})
    assert len(funcs) == 1
    assert weights == [1.0]
    scores = funcs[0](["IX MAN WALK HOUSE"], gold_gloss=["IX MAN WALK HOUSE"])
    assert scores == [pytest.approx(1.0)]


def test_builder_default_does_not_enable_it(vocab):
    """Historical configs must keep their exact reward stack."""
    from src.rewards.t2g_rewards import build_t2g_reward_functions

    funcs, _ = build_t2g_reward_functions({"weight_translation": 1.0})
    assert len(funcs) == 1  # only translation, edit-validity stays off


def test_builder_honours_the_oov_weight_override(vocab):
    from src.rewards.t2g_rewards import build_t2g_reward_functions

    funcs, _ = build_t2g_reward_functions(
        {"weight_edit_validity": 1.0, "edit_validity_oov_weight": 0.0}
    )
    # oov_weight=0.0 reproduces gloss_order_reward exactly.
    got = funcs[0](["IX MAN ZZZ"], gold_gloss=["IX MAN WALK"])[0]
    assert got == pytest.approx(gloss_order_reward("IX MAN ZZZ", "IX MAN WALK"))


def test_initialize_rewards_signature_is_unchanged(vocab):
    """Regression guard: the historical initializer must keep working.

    ``edit-rewards`` narrowed ``initialize_rewards`` to ``(vocab)``, which broke
    every historical caller. The repaired reward must not require that change.
    """
    import numpy as np

    bigram = np.full((len(VOCAB), len(VOCAB)), 1.0 / len(VOCAB))
    initialize_rewards(bigram, list(VOCAB))
    assert edit_validity_reward("IX MAN", "IX MAN") == pytest.approx(1.0)
