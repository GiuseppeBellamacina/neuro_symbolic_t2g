"""Unit tests for config inheritance (``extends`` resolution).

Synthetic tests (no real config files, no network): cover the deep-merge
semantics and the ``extends`` chain resolution, cycle detection and
missing-parent handling implemented in ``src/utils/config.py``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.utils.config import _deep_merge, resolve_config


def _write(tmp_path: Path, name: str, content: str) -> Path:
    p = tmp_path / name
    p.write_text(content, encoding="utf-8")
    return p


# ---------------------------------------------------------------------------
# _deep_merge
# ---------------------------------------------------------------------------


def test_deep_merge_override_wins() -> None:
    merged = _deep_merge({"a": 1, "b": 2}, {"b": 3})
    assert merged == {"a": 1, "b": 3}


def test_deep_merge_nested_dicts_merged_recursively() -> None:
    merged = _deep_merge(
        {"train": {"lr": 1e-5, "batch": 1, "nested": {"x": 1, "y": 2}}},
        {"train": {"batch": 8, "nested": {"y": 9}}},
    )
    assert merged == {
        "train": {"lr": 1e-5, "batch": 8, "nested": {"x": 1, "y": 9}}
    }


def test_deep_merge_lists_replaced_not_concatenated() -> None:
    merged = _deep_merge({"tags": ["a", "b"]}, {"tags": ["c"]})
    assert merged["tags"] == ["c"]


def test_deep_merge_scalar_replaces_dict() -> None:
    merged = _deep_merge({"grpo": {"beta": 0.04}}, {"grpo": 5})
    assert merged == {"grpo": 5}


def test_deep_merge_does_not_mutate_base() -> None:
    base = {"a": {"b": 1}}
    _deep_merge(base, {"a": {"b": 2}, "c": 3})
    assert base == {"a": {"b": 1}}


# ---------------------------------------------------------------------------
# resolve_config
# ---------------------------------------------------------------------------


def test_resolve_without_extends_is_unchanged(tmp_path) -> None:
    p = _write(tmp_path, "a.yaml", "model:\n  name: qwen\nnum_gpus: 1\n")
    assert resolve_config(p) == {"model": {"name": "qwen"}, "num_gpus": 1}


def test_resolve_single_extends_chain(tmp_path) -> None:
    """C → B → A: each level merges over the previous one."""
    _write(tmp_path, "a.yaml", "model:\n  name: qwen\n  quant: 4bit\nlr: 1.0e-5\n")
    _write(tmp_path, "b.yaml", "extends: a.yaml\nmodel:\n  quant: 8bit\nbatch: 2\n")
    _write(tmp_path, "c.yaml", "extends: b.yaml\nlr: 3.0e-6\n")
    cfg = resolve_config(tmp_path / "c.yaml")
    assert cfg == {
        "model": {"name": "qwen", "quant": "8bit"},
        "lr": 3e-6,
        "batch": 2,
    }
    assert "extends" not in cfg


def test_resolve_multiple_extends_list(tmp_path) -> None:
    _write(tmp_path, "common.yaml", "model:\n  name: qwen\nlr: 1.0e-5\n")
    _write(tmp_path, "extra.yaml", "batch: 8\ntags:\n  - a\n")
    _write(
        tmp_path,
        "child.yaml",
        "extends:\n  - common.yaml\n  - extra.yaml\nlr: 5.0e-6\n",
    )
    cfg = resolve_config(tmp_path / "child.yaml")
    assert cfg == {
        "model": {"name": "qwen"},
        "lr": 5e-6,
        "batch": 8,
        "tags": ["a"],
    }


def test_resolve_later_parent_wins_on_conflict(tmp_path) -> None:
    """With a list of parents, later entries override earlier ones."""
    _write(tmp_path, "p1.yaml", "x: 1\n")
    _write(tmp_path, "p2.yaml", "x: 2\n")
    _write(tmp_path, "child.yaml", "extends: [p1.yaml, p2.yaml]\n")
    assert resolve_config(tmp_path / "child.yaml")["x"] == 2


def test_resolve_relative_paths_from_declaring_file(tmp_path) -> None:
    """Parent paths are resolved relative to the declaring file's directory."""
    sub = tmp_path / "nested"
    sub.mkdir()
    _write(tmp_path, "base.yaml", "model:\n  name: qwen\n")
    _write(sub, "child.yaml", "extends: ../base.yaml\nlr: 1.0e-5\n")
    cfg = resolve_config(sub / "child.yaml")
    assert cfg == {"model": {"name": "qwen"}, "lr": 1e-5}


def test_resolve_cycle_raises_value_error(tmp_path) -> None:
    _write(tmp_path, "a.yaml", "extends: b.yaml\n")
    _write(tmp_path, "b.yaml", "extends: a.yaml\n")
    with pytest.raises(ValueError, match="cycle"):
        resolve_config(tmp_path / "a.yaml")


def test_resolve_self_cycle_raises_value_error(tmp_path) -> None:
    _write(tmp_path, "a.yaml", "extends: a.yaml\n")
    with pytest.raises(ValueError, match="cycle"):
        resolve_config(tmp_path / "a.yaml")


def test_resolve_missing_parent_raises_with_message(tmp_path) -> None:
    _write(tmp_path, "child.yaml", "extends: nonexistent.yaml\n")
    with pytest.raises(FileNotFoundError, match="nonexistent.yaml"):
        resolve_config(tmp_path / "child.yaml")


def test_resolve_invalid_extends_type_raises(tmp_path) -> None:
    _write(tmp_path, "child.yaml", "extends: [1, 2]\n")
    with pytest.raises(ValueError, match="extends"):
        resolve_config(tmp_path / "child.yaml")


def test_resolve_missing_file_raises(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        resolve_config(tmp_path / "ghost.yaml")
