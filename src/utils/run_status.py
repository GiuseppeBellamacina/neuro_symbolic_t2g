"""Monitoring status for the external monitor (ClusterControl), via the ``runstatus`` library.

The training writes ``logs/run_status-<SLURM_JOB_ID>.json`` (+ a ``.series.json`` sidecar with the chart
series) as it goes; ``cluster/cluster_helper.sh`` hands them to the app, so nothing parses the log.

This module is a thin, process-wide adapter: one shared ``RunStatus`` per process, and every call is a
no-op if ``runstatus`` is not installed or fails, because monitoring must NEVER break a training.
"""

from __future__ import annotations

import atexit
import logging
import os
import sys
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_LOGS_DIR = Path(__file__).resolve().parents[2] / "logs"
_status: Any = None
_tried = False


def get_status() -> Any:
    """The shared RunStatus, or None when runstatus is unavailable."""
    global _status, _tried
    if _tried:
        return _status
    _tried = True
    try:
        from runstatus import RunStatus
        from runstatus.core import default_path

        job = os.environ.get("SLURM_JOB_NAME") or None
        # Repo-root based like the old live status file: stable whatever the cwd is.
        path = (
            default_path()
            if os.environ.get("RUNSTATUS_PATH")
            else _LOGS_DIR / default_path().name
        )
        _status = RunStatus(
            path,
            kind="eval" if (job or "").startswith("eval-") else "train",
            name=job,
            rename={"learning_rate": "lr"},
            best={"eval_loss": "min"},
        )
        atexit.register(_close)
        _chain_excepthook()
    except Exception as exc:  # ImportError or anything else
        logger.warning("run status disabled: %s: %s", exc.__class__.__name__, exc)
    return _status


def _close() -> None:
    try:
        if _status is not None and _status.d["run"]["state"] == "running":
            _status.finish()
    except Exception:
        pass


def _chain_excepthook() -> None:
    previous = sys.excepthook

    def hook(etype, value, tb):
        try:
            if _status is not None:
                _status.fail(f"{etype.__name__}: {value}"[:300])
        except Exception:
            pass
        previous(etype, value, tb)

    sys.excepthook = hook


def run_status_set(
    phase: str | None = None,
    *,
    step: int | None = None,
    total_steps: int | None = None,
    note: str | None = None,
    eval_progress: str | None = None,
    eval_active: bool | None = None,
) -> None:
    """Update phase/progress. A phase change restarts the charts (step counters restart)."""
    try:
        s = get_status()
        if s is None:
            return
        change = phase is not None and s.d["run"]["phase"] not in (None, phase)
        s.set(
            phase=phase,
            total_steps=total_steps,
            note=note,
            eval_progress=eval_progress,
            eval_active=eval_active,
            reset_series=change,
        )
        if step is not None:
            s.log(step)
    except Exception:
        pass


def run_status_log(step: int, epoch: float | None = None, **metrics: Any) -> None:
    """Record the metrics of one log event (numeric values only; None/NaN are ignored)."""
    try:
        s = get_status()
        if s is None:
            return
        if epoch is not None:
            s.set(epoch=epoch)
        s.log(step, **metrics)
    except Exception:
        pass


def run_status_samples(lines: list[str], kind: str) -> None:
    """Latest completion samples, ready for display (last 6 are kept)."""
    try:
        s = get_status()
        if s is not None:
            s.set(samples=lines, extra={"samples_kind": kind})
    except Exception:
        pass


def run_status_stage_done(note: str) -> None:
    """End of a stage (SFT, eval): note only. The run is finished at process exit."""
    run_status_set(note=note, eval_active=False)


def _reset_for_tests() -> None:
    global _status, _tried
    _status, _tried = None, False
