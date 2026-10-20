"""entity_f1_reward: F1 fra token identici, tipo compreso (CoNLL-2003)."""

from __future__ import annotations

import pytest

from src.rewards.t2g_rewards import build_t2g_reward_functions, entity_f1_reward

GOLD = "LOC:Japan LOC:Syria MISC:Asian_Cup"


def test_exact_and_order_free() -> None:
    assert entity_f1_reward(GOLD, GOLD) == 1.0
    assert entity_f1_reward("MISC:Asian_Cup LOC:Syria LOC:Japan", GOLD) == 1.0


def test_right_span_wrong_type_is_worth_nothing() -> None:
    # ROUGE-L dava fra +0,14 e +0,43 a queste uscite.
    assert entity_f1_reward("ORG:Japan ORG:Syria ORG:Asian_Cup", GOLD) == -1.0
    assert entity_f1_reward(
        "MISC:Japan MISC:Syria MISC:Asian_Cup", GOLD
    ) == pytest.approx(2 * (1 / 3) - 1)


def test_partial_credit_and_none() -> None:
    # 1 entita' giusta su 3 gold, 1 emessa: P=1, R=1/3, F1=0,5 -> 0.
    assert entity_f1_reward("LOC:Japan", GOLD) == pytest.approx(0.0)
    assert entity_f1_reward("NONE", "NONE") == 1.0
    assert entity_f1_reward("NONE", GOLD) == -1.0
    assert entity_f1_reward("", GOLD) == -1.0


def test_weight_key_wires_the_component() -> None:
    funcs, weights = build_t2g_reward_functions({"weight_entity_f1": 0.4})
    assert [f.__name__ for f in funcs] == ["entity_f1_reward"]
    assert weights == [0.4]
    assert funcs[0](["LOC:Japan"], gold_gloss=["LOC:Japan"]) == [1.0]
