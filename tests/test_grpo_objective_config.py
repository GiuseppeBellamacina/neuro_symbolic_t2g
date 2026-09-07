"""Tests for the GRPO objective knobs exposed to config.

These knobs were previously never passed to TRL, so TRL 0.24.0 defaults applied
(`loss_type='dapo'`, `scale_rewards='group'`, `mask_truncated_completions=False`)
and every stored result was produced under them. The critical property is
therefore *backward compatibility*: an absent key must forward nothing, so
historical configs keep their exact objective.
"""

from __future__ import annotations

import pytest

from src.training.grpo_t2g_train import _grpo_objective_kwargs


def test_absent_keys_forward_nothing():
    """The compatibility guarantee: no key -> no override -> TRL defaults."""
    assert _grpo_objective_kwargs({}) == {}
    assert _grpo_objective_kwargs({"num_generations": 8, "beta": 0.04}) == {}


def test_dr_grpo_arm_is_expressible():
    got = _grpo_objective_kwargs(
        {
            "loss_type": "dr_grpo",
            "scale_rewards": "none",
            "mask_truncated_completions": True,
        }
    )
    assert got == {
        "loss_type": "dr_grpo",
        "scale_rewards": "none",
        "mask_truncated_completions": True,
    }


@pytest.mark.parametrize("loss_type", ["grpo", "bnpo", "dr_grpo", "dapo"])
def test_all_trl_loss_types_accepted(loss_type):
    assert _grpo_objective_kwargs({"loss_type": loss_type})["loss_type"] == loss_type


def test_legacy_boolean_scale_rewards_is_mapped():
    """TRL accepted booleans historically; map them instead of crashing."""
    assert _grpo_objective_kwargs({"scale_rewards": True})["scale_rewards"] == "group"
    assert _grpo_objective_kwargs({"scale_rewards": False})["scale_rewards"] == "none"


def test_epsilon_high_must_not_be_below_epsilon():
    ok = _grpo_objective_kwargs({"epsilon": 0.2, "epsilon_high": 0.28})
    assert ok == {"epsilon": 0.2, "epsilon_high": 0.28}
    with pytest.raises(ValueError, match="epsilon_high"):
        _grpo_objective_kwargs({"epsilon": 0.3, "epsilon_high": 0.1})


@pytest.mark.parametrize(
    "cfg,match",
    [
        ({"loss_type": "nope"}, "loss_type"),
        ({"scale_rewards": "weird"}, "scale_rewards"),
        ({"mask_truncated_completions": "yes"}, "mask_truncated_completions"),
        ({"epsilon": 0.0}, "epsilon"),
        ({"epsilon": -1.0}, "epsilon"),
    ],
)
def test_invalid_values_fail_before_training_starts(cfg, match):
    """A typo must not silently train a different objective for hours."""
    with pytest.raises(ValueError, match=match):
        _grpo_objective_kwargs(cfg)
