from __future__ import annotations

import re
from pathlib import Path

import yaml

from remote import app, tui

ROOT = Path(__file__).resolve().parent.parent


def _run_all_default_models() -> list[tuple[str, str, str]]:
    script = (ROOT / "cluster/run_all.sh").read_text(encoding="utf-8")
    match = re.search(
        r'if \[ "\$ABLATION".*?MODELS=\(\s*(.*?)\s*\)\s*else',
        script,
        re.DOTALL,
    )
    assert match is not None, "default MODELS array not found in cluster/run_all.sh"
    entries = re.findall(r'^\s*"([^"\n]+)"\s*$', match.group(1), re.MULTILINE)
    return [tuple(entry.split(":")) for entry in entries]  # type: ignore[misc]


def test_registry_and_manual_selection_are_exact() -> None:
    expected = (
        "baseline-zero",
        "baseline-few",
        "sft",
        "grpo-zero",
        "grpo-few",
        "sft-grpo-zero",
        "sft-grpo-few",
        "sft-grpo-zero-pda",
        "sft-grpo-zero-hot",
        "grpo-few-reward-edit",
        "grpo-few-reward-token-f1",
        "grpo-few-reward-chrfpp",
        "grpo-few-reward-rouge-l",
        "grpo-few-reward-sbleu2",
    )
    assert tuple(app.CONFIG_MAP) == expected
    assert set(tui.CONFIG_NAMES) == set(expected)
    assert all("probes/" not in path for path in app.CONFIG_MAP.values())


def test_default_campaign_order_count_and_modes() -> None:
    expected = [
        (
            "baseline-zero",
            "experiments/configs/qwen25-05b/baseline/zero-shot.yaml",
            "e",
        ),
        ("baseline-few", "experiments/configs/qwen25-05b/baseline/few-shot.yaml", "e"),
        ("sft", "experiments/configs/qwen25-05b/sft/zero-shot.yaml", "te"),
        ("grpo-zero", "experiments/configs/qwen25-05b/grpo/zero-shot.yaml", "te"),
        ("grpo-few", "experiments/configs/qwen25-05b/grpo/few-shot.yaml", "te"),
        (
            "sft-grpo-zero",
            "experiments/configs/qwen25-05b/sft-grpo/zero-shot.yaml",
            "te",
        ),
        ("sft-grpo-few", "experiments/configs/qwen25-05b/sft-grpo/few-shot.yaml", "te"),
    ]
    run_all_models = _run_all_default_models()
    assert run_all_models == expected
    assert app.DEFAULT_CAMPAIGN == expected
    assert len(run_all_models) == 7
    assert [mode for _, _, mode in run_all_models] == [
        "e",
        "e",
        "te",
        "te",
        "te",
        "te",
        "te",
    ]
    assert [path for _, path, _ in run_all_models] == [path for _, path, _ in expected]
    lines = app.build_queue_lines(app.QueueIn(ablation=True))
    assert len(lines) == 12
    assert [line.split(":", 1)[0] for line in lines[:2]] == ["eval", "eval"]
    tags = {line.split(":")[2] for line in lines}
    assert tags.isdisjoint(
        set(
            expected_name
            for expected_name in app.CONFIG_MAP
            if "reward-" in expected_name
        )
        | {"sft-grpo-zero-pda", "sft-grpo-zero-hot"}
    )


def test_campaign_copy_has_exact_entry_count() -> None:
    assert len(tui._CAMPAIGN_LINES) == 7
    assert all("12" not in line for line in tui._CAMPAIGN_LINES)
    assert all("dual prompt eval" in line for line in tui._CAMPAIGN_LINES[2:])


def test_eval_script_runs_dual_trained_modes_and_one_baseline_mode() -> None:
    script = (ROOT / "cluster/eval.sh").read_text(encoding="utf-8")
    dispatch = """if [ "$EXPERIMENT_KIND" = "baseline" ]; then
    if [ "$TRAIN_PROMPT_MODE" = "few-shot" ]; then
        run_evaluation retrieval
    else
        run_evaluation zero-shot
    fi
else
    run_evaluation zero-shot
    run_evaluation retrieval
fi"""
    assert dispatch in script
    assert "--prompt-mode ${prompt_mode}" in script


def test_yaml_top_level_sections_have_one_blank_line() -> None:
    config_root = ROOT / "experiments/configs/qwen25-05b"
    for path in config_root.rglob("*.yaml"):
        text = path.read_text(encoding="utf-8")
        raw = yaml.safe_load(text)
        top_level_keys = list(raw)
        key_lines = {
            match.group(1): index
            for index, line in enumerate(text.splitlines())
            if (match := re.fullmatch(r"([A-Za-z_][\w-]*):(?:\s+.*)?", line))
        }
        for key in top_level_keys[1:]:
            line_index = key_lines[key]
            assert text.splitlines()[line_index - 1] == "", f"{path}: before {key}"
