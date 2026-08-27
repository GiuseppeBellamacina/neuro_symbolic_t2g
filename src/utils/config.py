"""Configuration loading utilities.

Config files support inheritance via an ``extends`` key: a config may
reference one or more parent YAML files whose values are deep-merged under
it (child wins, nested dicts are merged recursively). See
:func:`resolve_config`.
"""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

# NOTE: ``src.utils.distributed`` only imports ``os`` (no torch), so this
# module stays importable without pulling in torch/transformers — required by
# the lightweight bootstrap in ``src/training/__main__.py``.
from src.utils.distributed import is_main_process


def load_config(config_path: str | Path) -> dict[str, Any]:
    """Load a YAML config file and return it as a dictionary.

    The YAML is resolved through :func:`resolve_config`, so ``extends``
    inheritance chains are merged automatically (a file without ``extends``
    behaves exactly as a plain ``yaml.safe_load``).

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        Parsed configuration dictionary (never contains the ``extends`` key).

    Raises:
        FileNotFoundError: If the config file does not exist.
        ValueError: If an ``extends`` cycle is detected or the YAML is not a
            mapping.
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    cfg, parents = _resolve_config(path, [])
    if is_main_process():
        if parents:
            # Display the extends chain relative to the config file (e.g.
            # "base.yaml" or "../base.yaml"), matching how it was declared.
            rel = ", ".join(os.path.relpath(str(p), str(path.parent)) for p in parents)
            print(f"[config] Loaded {path} (extends: {rel})")
        else:
            print(f"[config] Loaded {path}")
    return cfg


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge ``override`` into a copy of ``base``.

    - ``override`` wins on key conflicts.
    - Nested dicts are merged recursively.
    - Lists and scalars are **replaced** (never concatenated).

    ``base`` is not mutated.
    """
    result = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _load_yaml(path: Path) -> dict[str, Any]:
    """Parse a YAML file into a mapping (empty file → ``{}``)."""
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(
            f"Config YAML must be a mapping, got {type(data).__name__}: {path}"
        )
    return data


def resolve_config(config_path: str | Path) -> dict[str, Any]:
    """Load a YAML config, resolving an optional ``extends`` inheritance chain.

    If the YAML contains an ``extends`` key (a string or a list of strings),
    the referenced parent files are resolved recursively first — paths are
    relative to the directory of the file that declares them — and merged
    base→override so the child overrides its parents. The returned dict never
    contains the ``extends`` key.

    A file without ``extends`` is returned as-is (identical to a plain
    ``yaml.safe_load``).

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        The fully resolved configuration dictionary.

    Raises:
        FileNotFoundError: If the config or one of its ``extends`` parents
            does not exist (message names the missing parent and the child
            that referenced it).
        ValueError: If an ``extends`` cycle is detected or ``extends`` is not
            a string/list of strings.
    """
    cfg, _ = _resolve_config(Path(config_path), [])
    return cfg


def _resolve_config(path: Path, chain: list[Path]) -> tuple[dict[str, Any], list[Path]]:
    """Recursively resolve one config file.

    Returns ``(merged_dict, parent_paths)`` where *parent_paths* are the
    resolved parent files directly extended by *path* (used for logging).
    """
    if path in chain:
        cycle = " -> ".join(str(p) for p in [*chain, path])
        raise ValueError(f"Config extends cycle detected: {cycle}")

    chain.append(path)
    data = _load_yaml(path)
    parents: list[str] | None = data.pop("extends", None)

    result: dict[str, Any] = {}
    resolved_parents: list[Path] = []
    if parents is not None:
        if isinstance(parents, str):
            parents = [parents]
        if not isinstance(parents, list) or not all(
            isinstance(p, str) for p in parents
        ):
            raise ValueError(
                f"{path}: 'extends' must be a string or a list of strings, "
                f"got {parents!r}"
            )
        for parent in parents:
            parent_path = (path.parent / parent).resolve()
            if not parent_path.exists():
                raise FileNotFoundError(
                    f"{path}: extends parent not found: {parent_path}"
                )
            resolved_parents.append(parent_path)
            parent_cfg, _ = _resolve_config(parent_path, chain)
            result = _deep_merge(result, parent_cfg)

    result = _deep_merge(result, data)
    chain.pop()
    return result, resolved_parents


def resolve_run_dir(base_dir: str, prefix: str = "run") -> tuple[Path, str]:
    """Create a timestamped run subdirectory and a ``latest`` symlink.

    Structure::

        base_dir/
            train_20260403_120000/   <-- returned
            latest -> train_20260403_120000

    Args:
        base_dir: Parent directory (e.g. ``experiments/checkpoints/grpo/t2g/qwen05``).
        prefix: Name prefix for the subdirectory (``train``, ``eval``, …).

    Returns:
        ``(run_dir, run_id)`` where *run_id* is the subdirectory name.
    """
    run_id = f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    base = Path(base_dir)
    run_dir = base / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    _update_latest_symlink(base, run_id)
    return run_dir, run_id


def resolve_latest_run(base_dir: str) -> Path:
    """Resolve the most recent run directory under *base_dir*.

    Resolution order:
    1. ``base_dir/latest`` symlink (if present).
    2. Most recent timestamped subdirectory (lexicographic sort).
    3. *base_dir* itself (backward-compat: no versioned runs yet).
    """
    base = Path(base_dir)
    latest = base / "latest"
    if latest.exists():
        return latest.resolve()

    if base.exists():
        import re

        _RUN_RE = re.compile(r"^\w+_\d{8}_\d{6}$")
        subdirs = sorted(
            [
                d
                for d in base.iterdir()
                if d.is_dir() and d.name != "latest" and _RUN_RE.match(d.name)
            ],
            key=lambda d: d.name,
        )
        if subdirs:
            return subdirs[-1]

    return base


def _update_latest_symlink(base: Path, target_name: str) -> None:
    """Create or update ``base/latest`` → *target_name* (relative symlink)."""
    latest = base / "latest"
    try:
        if latest.is_symlink() or latest.exists():
            latest.unlink()
        latest.symlink_to(target_name)
    except OSError:
        pass
