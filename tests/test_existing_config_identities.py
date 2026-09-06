from __future__ import annotations

from pathlib import Path

from src.utils.config import resolve_config
from src.utils.paths import Cell, cell_from_config
from tests.validate_configs import validate_config

ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = ROOT / "experiments/configs/qwen25-05b"

EXPECTED = {
    "baseline/zero-shot.yaml": Cell(
        "qwen25-05b", "base", "zero-shot", "none", "baseline"
    ),
    "baseline/few-shot.yaml": Cell(
        "qwen25-05b", "base", "few-shot", "none", "baseline"
    ),
    "sft/zero-shot.yaml": Cell("qwen25-05b", "sft", "zero-shot", "none", "train"),
    "grpo/zero-shot.yaml": Cell("qwen25-05b", "grpo", "zero-shot", "none", "train"),
    "grpo/few-shot.yaml": Cell("qwen25-05b", "grpo", "few-shot", "none", "train"),
    "sft-grpo/zero-shot.yaml": Cell(
        "qwen25-05b", "sft-grpo", "zero-shot", "none", "train"
    ),
    "sft-grpo/few-shot.yaml": Cell(
        "qwen25-05b", "sft-grpo", "few-shot", "none", "train"
    ),
    "ablations/sft-grpo-zero-pda.yaml": Cell(
        "qwen25-05b", "sft-grpo", "zero-shot", "pda", "ablation"
    ),
    "ablations/sft-grpo-zero-hot.yaml": Cell(
        "qwen25-05b", "sft-grpo", "zero-shot", "hot", "ablation"
    ),
}


def test_exact_tree_identities_and_validator() -> None:
    assert {
        p.relative_to(CONFIG_DIR).as_posix() for p in CONFIG_DIR.rglob("*.yaml")
    } == {"base.yaml", "probes/rollouts.yaml", "probes/markov.yaml", *EXPECTED}
    for name, cell in EXPECTED.items():
        path = CONFIG_DIR / name
        assert cell_from_config(resolve_config(path)) == cell
        assert validate_config(path) == []


def test_primary_single_factor_cells_and_recipes() -> None:
    configs = {name: resolve_config(CONFIG_DIR / name) for name in EXPECTED}
    for prefix in ("grpo", "sft-grpo"):
        zero = configs[f"{prefix}/zero-shot.yaml"]
        few = configs[f"{prefix}/few-shot.yaml"]
        assert zero["retrieval"]["enabled"] is False
        assert few["retrieval"]["enabled"] is True
        assert zero["grpo"]["max_prompt_length"] == 256
        assert few["grpo"]["max_prompt_length"] == 768
        for cfg in (zero, few):
            assert cfg["reward"] == {"name": "edit-validity"}
            assert cfg["grpo"]["loss_type"] == "dr_grpo"
            assert cfg["grpo"]["scale_rewards"] == "none"
            assert cfg["grpo"]["mask_truncated_completions"] is True
            assert cfg["grpo"]["importance_sampling_level"] == "token"
            assert cfg["grpo"]["epsilon"] == 0.2
            assert cfg["grammar"]["enabled"] is True
            assert cfg["grammar"]["use_grammarllm_pda"] is False
            assert cfg["curriculum"]["enabled"] is False

    sft = configs["sft/zero-shot.yaml"]["training"]
    assert (
        sft["num_train_epochs"],
        sft["eval_fraction"],
        sft["early_stopping_patience"],
    ) == (3, 0.02, 3)
    assert all("output_dir" not in cfg.get("training", {}) for cfg in configs.values())
    assert all("log_dir" not in cfg.get("training", {}) for cfg in configs.values())


def test_ablations_change_only_requested_factor() -> None:
    base = resolve_config(CONFIG_DIR / "sft-grpo/zero-shot.yaml")
    pda = resolve_config(CONFIG_DIR / "ablations/sft-grpo-zero-pda.yaml")
    hot = resolve_config(CONFIG_DIR / "ablations/sft-grpo-zero-hot.yaml")
    assert pda["experiment"]["variant"] == "pda"
    assert pda["grammar"]["use_grammarllm_pda"] is True
    assert pda["grammar"]["pda_temperature"] == 0.7
    assert hot["experiment"]["variant"] == "hot"
    assert hot["grpo"]["temperature"] == 1.3
    assert hot["sft_pretrain"] == base["sft_pretrain"]
