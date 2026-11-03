"""cluster/presets.yaml: every preset job must name an existing experiment config cell."""

from __future__ import annotations

import re
from pathlib import Path

import yaml

from src.utils.run_paths import cell_tag

ROOT = Path(__file__).resolve().parent.parent
CONFIGS = ROOT / "experiments" / "configs"
TAG_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}")


def _cells() -> set[str]:
    """Cell tags of every runnable config (the shared base.yaml files are not cells)."""
    return {
        cell_tag(p.relative_to(CONFIGS).with_suffix("").as_posix())
        for p in CONFIGS.rglob("*.yaml")
        if not p.name.startswith("base")
    }


def _presets() -> list[dict]:
    text = (ROOT / "cluster" / "presets.yaml").read_text(encoding="utf-8")
    return yaml.safe_load(text)["presets"]


def test_presets_exist_and_have_unique_ids():
    presets = _presets()
    assert presets
    ids = [p["id"] for p in presets]
    assert len(ids) == len(set(ids))


def test_every_job_names_an_existing_config():
    cells = _cells()
    missing = [
        (p["id"], j["config"])
        for p in _presets()
        for j in p["jobs"]
        if j["config"] not in cells
    ]
    assert not missing, f"presets reference unknown configs: {missing}"


def test_job_types_and_tags_are_valid():
    for p in _presets():
        assert p["jobs"], f"preset {p['id']} has no jobs"
        for j in p["jobs"]:
            assert j["type"] in ("train", "eval"), (p["id"], j)
            assert TAG_RE.fullmatch(j.get("tag") or j["config"]), (p["id"], j)
