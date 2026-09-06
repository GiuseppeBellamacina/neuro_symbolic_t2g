from __future__ import annotations

from types import SimpleNamespace

import pytest

import src.training.grpo_t2g_train as train


def _config(tmp_path, monkeypatch, **grpo):
    captured = {}

    def fake_grpo_config(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(**kwargs)

    monkeypatch.setattr(train, "GRPOConfig", fake_grpo_config)
    training = {"output_dir": str(tmp_path / "out"), "log_dir": str(tmp_path / "log")}
    result = train._build_grpo_config(training, grpo)
    return result, captured


def test_grpo_config_sets_current_trl_fields(tmp_path, monkeypatch):
    result, fields = _config(tmp_path, monkeypatch, beta=0.03, epsilon_high=0.3)
    assert result.loss_type == "dr_grpo"
    assert fields["scale_rewards"] == "none"
    assert fields["mask_truncated_completions"] is True
    assert fields["importance_sampling_level"] == "token"
    assert fields["epsilon"] == 0.2
    assert fields["epsilon_high"] == 0.3
    assert fields["beta"] == 0.03
    assert fields["report_to"] == "none"


def test_num_generations_has_single_config_source(tmp_path, monkeypatch):
    result, fields = _config(tmp_path, monkeypatch, num_generations=6)
    assert result.num_generations == fields["num_generations"] == 6
    source = train.Path(train.__file__).read_text(encoding="utf-8")
    assert "group_size=int(grpo_config.num_generations)" in source
    assert 'grpo_cfg.get("num_generations", 8)' not in source
    assert "WandbCallback" not in source
    assert "TENSORBOARD_LOGGING_DIR" not in source


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("loss_type", "bad"),
        ("scale_rewards", "bad"),
        ("importance_sampling_level", "bad"),
        ("mask_truncated_completions", "yes"),
        ("epsilon", -0.1),
        ("epsilon_high", 0.1),
        ("beta", -0.1),
    ],
)
def test_grpo_config_rejects_invalid_values(tmp_path, monkeypatch, field, value):
    config = {field: value}
    if field == "epsilon_high":
        config["epsilon"] = 0.2
    with pytest.raises(ValueError):
        _config(tmp_path, monkeypatch, **config)


def test_production_trainer_has_no_markov_reward_loading():
    source = train.Path(train.__file__).read_text(encoding="utf-8")
    assert "compute_bigram_transitions" not in source
    assert "load_transition_matrix" not in source
    assert "viterbi_distance_reward" not in source
    assert "initialize_rewards(vocab)" in source


def test_tuple_optional_dependency_flags_are_normalized_before_trl_import():
    source = train.Path(train.__file__).read_text(encoding="utf-8")
    assert '"_weave_available"' in source
    assert "isinstance(getattr(_trl_iu, _optional_flag, False), tuple)" in source


def test_base_enables_and_training_passes_exact_diagnostics():
    base = (
        train.Path(train.__file__).parents[2]
        / "experiments/configs/qwen25-05b/base.yaml"
    ).read_text(encoding="utf-8")
    source = train.Path(train.__file__).read_text(encoding="utf-8")
    assert "track_diagnostics: true" in base
    assert 'track_diagnostics=grammar_cfg.get("track_diagnostics", False)' in source
    assert "reset_generation_state()" in source


def test_reward_ablation_python_gate_cannot_be_bypassed():
    config = {
        "experiment": {"variant": "reward-edit"},
        "reward": {"name": "edit-validity"},
    }
    with pytest.raises(ValueError, match="reward-qualification-report"):
        train.validate_reward_qualification(config, None)


def test_primary_config_does_not_require_qualification():
    assert (
        train.validate_reward_qualification({"experiment": {"variant": "none"}}, None)
        is None
    )
