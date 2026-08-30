"""Live status file — real-time training state for the external monitor.

The training process (GRPO/SFT/eval) writes a small JSON status file
(``logs/live_status.json`` at the repository root) that the external
cluster driver reads via ``cluster_helper.sh monitor`` (LIVE_STATUS key).
This replaces fragile log parsing: the producers write structured data,
the consumers read it directly.

Design:
- Atomic writes (tmp file + ``os.replace``) — readers never see a torn file.
- Throttled metric updates (max one write per ``_THROTTLE_SECONDS``) with
  un-throttled fast paths for phase changes and samples.
- Absolutely fail-safe: a monitoring bug must NEVER crash training — every
  public call swallows exceptions (first error is logged once).

Path note: the file lives at ``<repo_root>/logs/live_status.json`` where
repo_root is resolved from this module's location (src/utils/ → up two
levels), NOT from cwd, so it is stable regardless of the working directory.
"""

from __future__ import annotations

import json
import logging
import os
import socket
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[2]
STATUS_PATH = _REPO_ROOT / "logs" / "live_status.json"

# Metric fields are throttled to at most one write every 2 seconds; phase
# changes and samples bypass the throttle (they are rare and important).
_THROTTLE_SECONDS = 2.0

# Keep the last N samples in the file (formatted, ready for display).
_MAX_SAMPLES = 6

# Baseline payload — every field the monitor may expect. All nullable.
_BASE_PAYLOAD: dict[str, Any] = {
    "phase": None,  # sft | sft_eval | grpo | grpo_eval | eval | null
    "step": None,
    "total_steps": None,
    "loss": None,
    "reward": None,
    "reward_avg": None,  # running mean of logged GRPO rewards (callback-side)
    "lr": None,
    "eval_loss": None,
    "eval_loss_best": None,
    "epoch": None,
    "eval_active": False,
    "eval_progress": None,  # e.g. "356/365"
    "samples": [],
    "samples_kind": None,  # "sft" | "grpo" | "eval" | null
    "note": None,
}

_state: dict[str, Any] = dict(_BASE_PAYLOAD)
_last_write = 0.0
_last_write_attempt = 0.0
_error_logged = False


def _log_once(message: str) -> None:
    """Log the first live-status error only — no log spam on repeated failures."""
    global _error_logged
    if not _error_logged:
        _error_logged = True
        logger.warning("live_status disabled (first error): %s", message)


def _write(force: bool) -> None:
    """Write ``_state`` atomically to STATUS_PATH (throttled unless forced)."""
    global _last_write, _last_write_attempt
    now = time.monotonic()
    _last_write_attempt = now
    if not force and (now - _last_write) < _THROTTLE_SECONDS:
        return
    _last_write = now
    payload = {
        **_state,
        "ts": datetime.now().isoformat(timespec="seconds"),
        "hostname": socket.gethostname(),
        "pid": os.getpid(),
    }
    try:
        STATUS_PATH.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(
            dir=str(STATUS_PATH.parent), prefix=".live_status.", suffix=".tmp"
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False)
            os.replace(tmp_name, STATUS_PATH)
        except BaseException:
            try:
                os.unlink(tmp_name)
            except OSError:
                pass
            raise
    except Exception as exc:  # monitoring must never crash training
        _log_once(f"{exc.__class__.__name__}: {exc}")


def live_status_set(phase: str | None = None, **fields: Any) -> None:
    """Merge fields into the live status and (re)write the file.

    Args:
        phase: Current pipeline phase ("sft", "sft_eval", "grpo", "grpo_eval",
            "eval" or None when idle). Phase changes are written un-throttled.
        **fields: Any payload fields to update (step, loss, reward, lr,
            eval_loss, eval_active, eval_progress, note, ...). ``None`` values
            overwrite existing ones (explicit reset) — pass only what you own.
    """
    try:
        phase_changed = phase is not None and phase != _state.get("phase")
        if phase is not None:
            _state["phase"] = phase
        for key, value in fields.items():
            _state[key] = value
        _write(force=phase_changed)
    except Exception as exc:
        _log_once(f"live_status_set: {exc}")


def live_status_add_samples(samples: list[str], kind: str) -> None:
    """Replace the sample list with the latest ones (keep last N).

    Un-throttled: samples arrive at most every few hundred steps and are the
    most interesting thing to see live.
    """
    try:
        _state["samples"] = list(samples[-_MAX_SAMPLES:])
        _state["samples_kind"] = kind
        _write(force=True)
    except Exception as exc:
        _log_once(f"live_status_add_samples: {exc}")


def live_status_reset(note: str | None = None) -> None:
    """Reset to the idle baseline (job finished) — keeps nothing but a note."""
    global _state
    try:
        _state = {**_BASE_PAYLOAD, "note": note}
        _write(force=True)
    except Exception as exc:
        _log_once(f"live_status_reset: {exc}")


def live_status_get() -> dict[str, Any] | None:
    """Read the current status file (mainly for tests). None if absent/broken."""
    try:
        return json.loads(STATUS_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def live_status_path() -> Path:
    """Expose the resolved status path (for tests / diagnostics)."""
    return STATUS_PATH
