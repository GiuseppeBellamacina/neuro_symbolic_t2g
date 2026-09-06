from __future__ import annotations

import ast
from pathlib import Path

import pytest

from src.rewards.t2g_rewards import (
    _edit_validity_score,
    build_t2g_reward_functions,
    edit_validity_reward,
)


def test_edit_validity_normalizes_and_scores_edits(reward_setup):
    gold = "IX MAN WALK HOUSE"
    assert edit_validity_reward("ix  man\twalk house", gold) == 1.0
    assert edit_validity_reward("IX MAN WALK", gold) == pytest.approx(0.5)
    assert edit_validity_reward("IX MAN GO HOUSE", gold) == pytest.approx(0.5)
    assert edit_validity_reward("IX MAN WALK HOUSE BOOK", gold) == pytest.approx(0.6)


def test_edit_validity_hard_invalid_gate(reward_setup):
    gold = "IX MAN WALK HOUSE"
    assert edit_validity_reward("", gold) == -1.0
    assert edit_validity_reward("IX MAN ZZZ HOUSE", gold) == -1.0
    assert edit_validity_reward(gold, "") == -1.0
    assert _edit_validity_score("ZZZ", gold, {"IX"}, invalid_score=-0.25) == -0.25


def test_builder_returns_exact_single_production_reward(reward_setup):
    funcs, weights = build_t2g_reward_functions(
        {"name": "edit-validity", "invalid_score": -0.5}
    )
    assert [fn.__name__ for fn in funcs] == ["edit_validity_reward"]
    assert weights == [1.0]
    assert funcs[0](["IX ZZZ"], gold_gloss=["IX MAN"]) == [-0.5]


def test_builder_defaults_to_edit_validity(reward_setup):
    funcs, weights = build_t2g_reward_functions()
    assert [fn.__name__ for fn in funcs] == ["edit_validity_reward"]
    assert weights == [1.0]


@pytest.mark.parametrize(
    "config",
    [
        {"name": "viterbi"},
        {"weight_translation": 1.0},
        {"name": "structural_dense", "weight_structure": 1.0},
    ],
)
def test_builder_rejects_unknown_and_historical_configs(config):
    with pytest.raises(ValueError, match="production reward"):
        build_t2g_reward_functions(config)


def test_production_module_exports_no_historical_rewards_or_weight_keys():
    module_path = Path(__file__).parents[1] / "src" / "rewards" / "t2g_rewards.py"
    source = module_path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    defined = {node.name for node in tree.body if isinstance(node, ast.FunctionDef)}
    historical = {
        "translation_quality_reward",
        "bleu_reward",
        "structural_dense_reward",
        "gold_structure_reward",
        "verifier_scaled_reward",
        "gloss_format_reward",
        "gloss_repetition_reward",
        "viterbi_distance_reward",
        "soft_viterbi_distance_reward",
    }
    assert defined.isdisjoint(historical)
    assert all(
        key not in source
        for key in ("weight_translation", "weight_bleu", "weight_structure")
    )
    assert "numpy" not in source
    assert "transition_matrix" not in source
