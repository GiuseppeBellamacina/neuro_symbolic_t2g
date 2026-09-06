from pathlib import Path

import yaml

from remote import app, tui
from src.rewards.t2g_rewards import REWARD_NAMES
from src.utils.config import resolve_config

ROOT = Path(__file__).parents[1]
CONFIG_DIR = ROOT / "experiments/configs/qwen25-05b/ablations/rewards"


def test_reward_ablations_are_one_factor_manual_configs():
    parent = resolve_config(ROOT / "experiments/configs/qwen25-05b/grpo/few-shot.yaml")
    configs = sorted(CONFIG_DIR.glob("*.yaml"))
    assert len(configs) == 5
    names = set()
    for path in configs:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        assert raw["extends"] == "../../grpo/few-shot.yaml"
        assert set(raw) == {"extends", "experiment", "reward", "wandb"}
        assert set(raw["experiment"]) == {"kind", "variant"}
        assert set(raw["reward"]) == {"name"}
        assert set(raw["wandb"]) == {"run_name"}
        resolved = resolve_config(path)
        names.add(resolved["reward"]["name"])
        comparable = dict(resolved)
        comparable["experiment"] = dict(comparable["experiment"])
        comparable["experiment"].update(kind="train", variant="none")
        comparable["reward"] = parent["reward"]
        comparable["wandb"] = parent["wandb"]
        assert comparable == parent
    assert names == set(REWARD_NAMES)


def test_reward_configs_are_manual_only_wired():
    ids = {name for name in app.CONFIG_MAP if name.startswith("grpo-few-reward-")}
    assert len(ids) == 5
    assert ids <= set(tui.CONFIG_NAMES)
    assert ids.isdisjoint({tag for tag, _, _ in app.DEFAULT_CAMPAIGN})


def test_reward_source_has_no_mass_or_markov_reward():
    source = (ROOT / "src/rewards/t2g_rewards.py").read_text(encoding="utf-8").lower()
    assert "transition_matrix" not in source
    assert "markov" not in source
    assert "mass" not in source
