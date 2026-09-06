from __future__ import annotations

import ast
from pathlib import Path

import pytest

from src.rewards.t2g_rewards import (
    REWARD_NAMES,
    _edit_validity_score,
    build_t2g_reward_functions,
    chrfpp_validity_reward,
    edit_validity_reward,
    initialize_rewards,
    reward_protocol,
    rouge_l_validity_reward,
    sbleu2_exp_validity_reward,
    score_validity_reward,
    token_f1_validity_reward,
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


def test_token_f1_uses_clipped_multiset_overlap(reward_setup):
    assert token_f1_validity_reward("IX IX MAN", "IX MAN MAN") == pytest.approx(1 / 3)
    assert token_f1_validity_reward("IX MAN", "IX MAN") == 1.0
    assert token_f1_validity_reward("MAN IX", "IX MAN") == 1.0
    assert (
        reward_protocol()["settings"]["token-f1-validity"]["order_sensitive"] is False
    )


def test_rouge_l_is_normalized_token_lcs_f1(reward_setup):
    assert rouge_l_validity_reward("IX WALK MAN", "IX MAN WALK") == pytest.approx(1 / 3)
    assert rouge_l_validity_reward("ix man", "IX MAN") == 1.0


def test_sacrebleu_rewards_known_and_short_cases(reward_setup):
    assert chrfpp_validity_reward("IX MAN", "IX MAN") == pytest.approx(1.0)
    assert sbleu2_exp_validity_reward("IX", "IX") == pytest.approx(1.0)
    assert sbleu2_exp_validity_reward("IX", "MAN") == -1.0
    assert chrfpp_validity_reward("HOUSE", "HORSE") < 1.0


def test_empty_vocabulary_returns_invalid_and_restores_fixture(reward_setup):
    vocab, _, _ = reward_setup
    try:
        initialize_rewards([])
        assert edit_validity_reward("IX", "IX") == -1.0
        assert token_f1_validity_reward("IX", "IX", invalid_score=-0.25) == -0.25
    finally:
        initialize_rewards(vocab)

    assert edit_validity_reward("IX", "IX") == 1.0


@pytest.mark.parametrize(
    "reward",
    [
        edit_validity_reward,
        token_f1_validity_reward,
        chrfpp_validity_reward,
        rouge_l_validity_reward,
        sbleu2_exp_validity_reward,
    ],
)
def test_all_rewards_reach_lower_bound_for_disjoint_valid_tokens(reward, reward_setup):
    assert reward("BOOK DOG", "IX MAN") == -1.0


@pytest.mark.parametrize("name", REWARD_NAMES)
@pytest.mark.parametrize(
    ("completion", "gold"),
    [("IX", "IX"), ("IX MAN", "MAN IX"), ("IX IX", "IX MAN")],
)
def test_valid_scores_are_bounded(name, completion, gold, reward_setup):
    score = score_validity_reward(name, completion, gold, {"ix", "man", "book", "dog"})
    assert -1.0 <= score <= 1.0


@pytest.mark.parametrize("name", REWARD_NAMES)
def test_all_reward_identities_are_exactly_one(name, reward_setup):
    assert score_validity_reward(name, "IX MAN", "IX MAN", {"ix", "man"}) == 1.0


def test_wrapper_handles_trl_chat_completions_and_gloss_extraction(reward_setup):
    (reward_fn,), _ = build_t2g_reward_functions()
    completions = [
        [{"role": "assistant", "content": "<think>draft</think>  ix\tMAN  "}],
        [{"role": "assistant", "content": "```gloss\nIX MAN\n```"}],
    ]
    assert reward_fn(completions, gold_gloss=["IX MAN", "IX MAN"]) == [1.0, 1.0]


def test_wrapper_uses_final_conversational_message(reward_setup):
    (reward_fn,), _ = build_t2g_reward_functions()
    completion = [
        {"role": "assistant", "content": "IX"},
        {"role": "assistant", "content": "IX MAN"},
    ]
    assert reward_fn([completion], gold_gloss=["IX MAN"]) == [1.0]


@pytest.mark.parametrize(
    "completion",
    [[], [123], [{"role": "assistant"}], [{"content": None}], [{"content": 123}]],
)
def test_wrapper_returns_invalid_for_malformed_conversational_completion(
    completion, reward_setup
):
    (reward_fn,), _ = build_t2g_reward_functions(
        {"name": "edit-validity", "invalid_score": -0.25}
    )
    assert reward_fn([completion], gold_gloss=["IX"]) == [-0.25]


def test_wrapper_uses_custom_invalid_score_for_missing_or_short_gold(reward_setup):
    (reward_fn,), _ = build_t2g_reward_functions(
        {"name": "token-f1-validity", "invalid_score": -0.25}
    )
    assert reward_fn(["IX", "MAN"], gold_gloss=["IX"]) == [1.0, -0.25]
    assert reward_fn(["IX"], gold_gloss=None) == [-0.25]


def test_case_and_whitespace_normalize_but_punctuation_is_a_token(reward_setup):
    assert edit_validity_reward("  ix\tman\n", "IX MAN") == 1.0
    assert edit_validity_reward("IX, MAN", "IX MAN", invalid_score=-0.5) == -0.5


def test_repetition_and_order_have_metric_specific_effects(reward_setup):
    assert token_f1_validity_reward("MAN IX", "IX MAN") == 1.0
    assert rouge_l_validity_reward("MAN IX", "IX MAN") < 1.0
    assert edit_validity_reward("MAN IX", "IX MAN") < 1.0
    assert sbleu2_exp_validity_reward("MAN IX", "IX MAN") < 1.0
    assert token_f1_validity_reward("IX IX", "IX MAN") < 1.0


@pytest.mark.parametrize("name", REWARD_NAMES)
def test_all_rewards_share_gate_scale_and_builder(name, reward_setup):
    funcs, weights = build_t2g_reward_functions({"name": name, "invalid_score": -0.4})
    assert len(funcs) == 1 and weights == [1.0]
    assert funcs[0](["IX MAN"], gold_gloss=["IX MAN"]) == pytest.approx([1.0])
    assert funcs[0](["IX ZZZ"], gold_gloss=["IX MAN"]) == [-0.4]
    assert funcs[0](["IX"], gold_gloss=[""]) == [-0.4]


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


def test_builder_and_initializer_reject_invalid_types(reward_setup):
    with pytest.raises(TypeError, match="vocabulary list"):
        initialize_rewards(("IX", "MAN"))  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="mapping"):
        build_t2g_reward_functions([])  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        build_t2g_reward_functions({"invalid_score": "not-a-number"})


@pytest.mark.parametrize(
    "invalid_score", [float("nan"), float("inf"), float("-inf"), -1.01, 1.01]
)
def test_builder_rejects_nonfinite_or_out_of_range_invalid_score(invalid_score):
    with pytest.raises(ValueError, match=r"finite number within \[-1, 1\]"):
        build_t2g_reward_functions(
            {"name": "edit-validity", "invalid_score": invalid_score}
        )


def test_unicode_casefold_applies_to_vocabulary_and_text(reward_setup):
    vocab, _, _ = reward_setup
    try:
        initialize_rewards(["STRASSE"])
        assert edit_validity_reward("Straße", "STRASSE") == 1.0
    finally:
        initialize_rewards(vocab)


def test_protocol_documents_asymmetric_normalization_and_source_digest():
    protocol = reward_protocol()
    common = protocol["settings"]["common"]
    assert common["completion_normalization"] == "extract-gloss+casefold+split"
    assert common["reference_normalization"] == "casefold+split"

    provenance = protocol["implementation_provenance"]
    assert provenance["files"] == [
        "src/rewards/t2g_rewards.py",
        "src/utils/text_utils.py",
    ]
    assert len(provenance["sha256"]) == 64
    int(provenance["sha256"], 16)


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
