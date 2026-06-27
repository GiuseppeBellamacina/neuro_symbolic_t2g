"""Configuration loading utilities."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from src.utils.distributed import is_main_process


def load_config(config_path: str | Path) -> dict[str, Any]:
    """Load a YAML config file and return it as a dictionary.

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        Parsed configuration dictionary.

    Raises:
        FileNotFoundError: If the config file does not exist.
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if is_main_process():
        print(f"[config] Loaded {path}")
    return cfg


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
