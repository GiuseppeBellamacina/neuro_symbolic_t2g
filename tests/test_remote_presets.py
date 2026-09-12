"""Tests for remote/presets.py — the TUI-local, never-sent-to-Render preset
loader and resolver used by the Presets screen (remote/tui.py).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from remote.presets import (
    DEFAULT_PRESETS_PATH,
    JobDef,
    PresetDef,
    PresetsError,
    load_presets,
    resolve_jobs,
)


def _write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "presets.yaml"
    path.write_text(text, encoding="utf-8")
    return path


# ── load_presets ──────────────────────────────────────────────────────────


def test_load_presets_missing_file_returns_empty(tmp_path):
    assert load_presets(tmp_path / "does-not-exist.yaml") == []


def test_load_presets_empty_file_returns_empty(tmp_path):
    assert load_presets(_write(tmp_path, "")) == []


def test_load_presets_parses_basic_shape(tmp_path):
    path = _write(
        tmp_path,
        """
presets:
  - id: t0
    label: "Tier 0"
    description: "canary"
    jobs:
      - {type: eval, config: baseline-zero-shot}
      - {type: train, config: grpo-few-shot, tag: my-tag, mode: "--resume"}
""",
    )
    presets = load_presets(path)
    assert len(presets) == 1
    preset = presets[0]
    assert preset.id == "t0"
    assert preset.label == "Tier 0"
    assert preset.description == "canary"
    assert preset.job_count == 2
    assert preset.jobs[0] == JobDef(type="eval", config="baseline-zero-shot")
    assert preset.jobs[1] == JobDef(
        type="train", config="grpo-few-shot", tag="my-tag", mode="--resume"
    )


def test_load_presets_defaults_label_to_id(tmp_path):
    path = _write(
        tmp_path,
        """
presets:
  - id: bare
    jobs:
      - {type: eval, config: x}
""",
    )
    presets = load_presets(path)
    assert presets[0].label == "bare"
    assert presets[0].description == ""


def test_load_presets_rejects_invalid_yaml(tmp_path):
    path = _write(tmp_path, "presets: [this is not: valid: yaml: at all")
    with pytest.raises(PresetsError):
        load_presets(path)


def test_load_presets_rejects_missing_top_level_key(tmp_path):
    path = _write(tmp_path, "not_presets: []")
    with pytest.raises(PresetsError):
        load_presets(path)


def test_load_presets_rejects_missing_jobs(tmp_path):
    path = _write(tmp_path, "presets:\n  - id: t0\n    label: Tier 0\n")
    with pytest.raises(PresetsError):
        load_presets(path)


def test_load_presets_rejects_empty_jobs_list(tmp_path):
    path = _write(tmp_path, "presets:\n  - id: t0\n    jobs: []\n")
    with pytest.raises(PresetsError):
        load_presets(path)


def test_load_presets_rejects_bad_job_type(tmp_path):
    path = _write(
        tmp_path,
        "presets:\n  - id: t0\n    jobs:\n      - {type: bogus, config: x}\n",
    )
    with pytest.raises(PresetsError):
        load_presets(path)


def test_load_presets_rejects_missing_config(tmp_path):
    path = _write(
        tmp_path,
        "presets:\n  - id: t0\n    jobs:\n      - {type: eval}\n",
    )
    with pytest.raises(PresetsError):
        load_presets(path)


def test_load_presets_rejects_duplicate_ids(tmp_path):
    path = _write(
        tmp_path,
        """
presets:
  - id: dup
    jobs: [{type: eval, config: x}]
  - id: dup
    jobs: [{type: eval, config: y}]
""",
    )
    with pytest.raises(PresetsError):
        load_presets(path)


def test_load_presets_rejects_missing_id(tmp_path):
    path = _write(tmp_path, "presets:\n  - jobs: [{type: eval, config: x}]\n")
    with pytest.raises(PresetsError):
        load_presets(path)


def test_default_presets_path_resolves_and_loads(tmp_path):
    """The real remote/presets.yaml shipped in the repo must itself parse
    cleanly and cover every tier described to the user."""
    presets = load_presets(DEFAULT_PRESETS_PATH)
    ids = [p.id for p in presets]
    assert ids == [
        "tier0-canary",
        "tier1-core",
        "tier2-highvalue",
        "tier3-rewards-loss",
        "tier4-glossary",
    ]
    assert all(p.job_count > 0 for p in presets)


# ── resolve_jobs ──────────────────────────────────────────────────────────


def _presets() -> list[PresetDef]:
    return [
        PresetDef(
            id="a",
            label="A",
            description="",
            jobs=(JobDef(type="eval", config="cfg-a", tag="cfg-a"),),
        ),
        PresetDef(
            id="b",
            label="B",
            description="",
            jobs=(
                JobDef(type="train", config="cfg-b", tag="cfg-b"),
                JobDef(type="eval", config="cfg-b", tag="cfg-b"),
            ),
        ),
    ]


def test_resolve_jobs_single_preset():
    jobs = resolve_jobs(_presets(), ["a"])
    assert jobs == [{"type": "eval", "config": "cfg-a", "tag": "cfg-a"}]


def test_resolve_jobs_concatenates_in_given_order():
    jobs = resolve_jobs(_presets(), ["b", "a"])
    assert jobs == [
        {"type": "train", "config": "cfg-b", "tag": "cfg-b"},
        {"type": "eval", "config": "cfg-b", "tag": "cfg-b"},
        {"type": "eval", "config": "cfg-a", "tag": "cfg-a"},
    ]


def test_resolve_jobs_ignores_unknown_id():
    jobs = resolve_jobs(_presets(), ["a", "ghost"])
    assert jobs == [{"type": "eval", "config": "cfg-a", "tag": "cfg-a"}]


def test_resolve_jobs_empty_selection():
    assert resolve_jobs(_presets(), []) == []


def test_job_def_as_dict_omits_empty_optional_fields():
    assert JobDef(type="eval", config="x").as_dict() == {
        "type": "eval",
        "config": "x",
    }
