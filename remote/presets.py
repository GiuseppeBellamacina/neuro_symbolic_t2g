"""Local-only job presets ("tiers") for the TUI (remote/tui.py).

This module and ``remote/presets.yaml`` are **never imported by remote/app.py**
and never leave this machine: the driver service only ever receives a plain,
already-resolved job list (``{type, config, tag, mode}``, the same shape
``POST /jobs/batch``/``POST /queue`` already accept). A preset is just a
named, ordered group of such jobs kept in a local YAML file, so adding or
reordering one requires no redeploy on Render.

Multiple presets can be combined: :func:`resolve_jobs` concatenates the
``jobs`` of every preset in ``selected_ids``, in the order given — so the
caller (``PresetsScreen``) decides which presets run and in what order by
building that list, and this module stays a pure, easily-testable resolver.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

DEFAULT_PRESETS_PATH = Path(__file__).resolve().parent / "presets.yaml"


class PresetsError(Exception):
    """Raised when ``presets.yaml`` is missing or malformed."""


@dataclass(frozen=True)
class JobDef:
    """One queue entry as a preset declares it (mirrors the API's JobIn)."""

    type: str
    config: str
    tag: str | None = None
    mode: str | None = None

    def as_dict(self) -> dict[str, str]:
        d: dict[str, str] = {"type": self.type, "config": self.config}
        if self.tag:
            d["tag"] = self.tag
        if self.mode:
            d["mode"] = self.mode
        return d


@dataclass(frozen=True)
class PresetDef:
    """A named, ordered group of jobs ("tier")."""

    id: str
    label: str
    description: str
    jobs: tuple[JobDef, ...]

    @property
    def job_count(self) -> int:
        return len(self.jobs)


def _parse_job(raw: Any, *, preset_id: str, index: int) -> JobDef:
    if not isinstance(raw, dict):
        raise PresetsError(
            f"preset {preset_id!r}, job #{index}: atteso un mapping, trovato {raw!r}"
        )
    job_type = raw.get("type")
    config = raw.get("config")
    if job_type not in ("train", "eval"):
        raise PresetsError(
            f"preset {preset_id!r}, job #{index}: type non valido {job_type!r} "
            "(atteso 'train' o 'eval')"
        )
    if not config or not isinstance(config, str):
        raise PresetsError(
            f"preset {preset_id!r}, job #{index}: 'config' mancante o non stringa"
        )
    tag = raw.get("tag")
    mode = raw.get("mode")
    return JobDef(
        type=job_type,
        config=config,
        tag=str(tag) if tag else None,
        mode=str(mode) if mode else None,
    )


def _parse_preset(raw: Any, *, index: int) -> PresetDef:
    if not isinstance(raw, dict):
        raise PresetsError(f"preset #{index}: atteso un mapping, trovato {raw!r}")
    preset_id = raw.get("id")
    if not preset_id or not isinstance(preset_id, str):
        raise PresetsError(f"preset #{index}: 'id' mancante o non stringa")
    label = str(raw.get("label") or preset_id)
    description = str(raw.get("description") or "")
    raw_jobs = raw.get("jobs")
    if not isinstance(raw_jobs, list) or not raw_jobs:
        raise PresetsError(f"preset {preset_id!r}: 'jobs' mancante o vuoto")
    jobs = tuple(
        _parse_job(j, preset_id=preset_id, index=i) for i, j in enumerate(raw_jobs)
    )
    return PresetDef(id=preset_id, label=label, description=description, jobs=jobs)


def load_presets(path: str | Path = DEFAULT_PRESETS_PATH) -> list[PresetDef]:
    """Load and validate every preset from *path* (default ``remote/presets.yaml``).

    Returns an empty list (never raises) when the file simply does not exist —
    the preset screen degrades to "nessun preset trovato", never a crash, same
    convention as every other optional-data panel in the TUI. Malformed YAML or
    a schema violation DOES raise :class:`PresetsError`, since that is a local
    editing mistake worth surfacing loudly rather than silently ignoring.
    """
    file_path = Path(path)
    if not file_path.exists():
        return []
    try:
        data = yaml.safe_load(file_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise PresetsError(f"{file_path}: YAML non valido: {exc}") from exc
    if data is None:
        return []
    if not isinstance(data, dict) or "presets" not in data:
        raise PresetsError(f"{file_path}: attesa una chiave top-level 'presets'")
    raw_presets = data["presets"]
    if not isinstance(raw_presets, list):
        raise PresetsError(f"{file_path}: 'presets' deve essere una lista")
    presets = [_parse_preset(p, index=i) for i, p in enumerate(raw_presets)]
    seen: set[str] = set()
    for preset in presets:
        if preset.id in seen:
            raise PresetsError(f"{file_path}: id duplicato {preset.id!r}")
        seen.add(preset.id)
    return presets


def resolve_jobs(
    presets: list[PresetDef], selected_ids: list[str]
) -> list[dict[str, str]]:
    """Concatenate the jobs of the presets in ``selected_ids``, in that order.

    Each preset may be listed only once in *selected_ids*; a preset id not
    found in *presets* is skipped (defensive: the caller's own picker only
    ever offers known ids, but a stale/edited ``presets.yaml`` between the
    picker being built and this call should not crash the launch).
    """
    by_id = {p.id: p for p in presets}
    jobs: list[dict[str, str]] = []
    for preset_id in selected_ids:
        preset = by_id.get(preset_id)
        if preset is None:
            continue
        jobs.extend(job.as_dict() for job in preset.jobs)
    return jobs
