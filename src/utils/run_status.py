"""Monitoring status for the external monitor (ClusterControl), via the ``runstatus`` library.

The training writes ``run_status/<SLURM_JOB_ID>.json`` (+ ``<SLURM_JOB_ID>.series.json`` with the chart
series) as it goes; ``cluster/cluster_helper.sh`` hands them to the app, so nothing parses the log.
The folder is separate from ``logs/`` (SLURM logs) on purpose.

The library is vendored in ``src/_vendor/runstatus`` (see ``vendor_runstatus.ps1``): the cluster has no
access to GitHub, so it travels with the code through the usual sync, never through pip.

This module is a thin, process-wide adapter: one shared ``RunStatus`` per process, and every call is a
no-op if the library fails to load or to write, because monitoring must NEVER break a training.
"""

from __future__ import annotations

import atexit
import logging
import os
import re
import sys
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_ROOT = Path(__file__).resolve().parents[2]
_status: Any = None
_tried = False

# Metrics that get a chart series (the sidecar travels over a slow link: keep it to what is worth plotting).
SERIES = [
    "loss",
    "eval_loss",
    "lr",
    "grad_norm",
    "reward*",
    "rewards/*",
    "kl",
    "entropy",
    "completion*",
    "grammar/*",
    "*accuracy*",
]


def _status_path() -> Path:
    """RUNSTATUS_PATH, else <RUNSTATUS_DIR or <repo>/run_status>/<jobid>.json (stable whatever the cwd is)."""
    if os.environ.get("RUNSTATUS_PATH"):
        return Path(os.environ["RUNSTATUS_PATH"])
    folder = Path(os.environ.get("RUNSTATUS_DIR") or _ROOT / "run_status")
    job = os.environ.get("SLURM_JOB_ID")
    return folder / (f"{job}.json" if job else "run_status.json")


def get_status() -> Any:
    """The shared RunStatus, or None when it cannot be created."""
    global _status, _tried
    if _tried:
        return _status
    _tried = True
    try:
        from src._vendor.runstatus import RunStatus

        job = os.environ.get("SLURM_JOB_NAME") or None
        _status = RunStatus(
            _status_path(),
            kind="eval" if (job or "").startswith("eval-") else "train",
            name=job,
            rename={"learning_rate": "lr"},
            best={"eval_loss": "min"},
            series=SERIES,
        )
        atexit.register(_close)
        _chain_excepthook()
    except Exception as exc:
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
    """Update stage/progress. Entering a new phase closes the previous stage and restarts counters and charts."""
    try:
        s = get_status()
        if s is None:
            return
        if phase is not None:
            s.stage(phase)
        s.set(
            total_steps=total_steps,
            note=note,
            eval_progress=eval_progress,
            eval_active=eval_active,
        )
        if step is not None:
            s.log(step)
    except Exception:
        pass


_STAGE_LABELS = {"sft": "SFT", "grpo": "GRPO", "eval": "Eval"}
_PARAMS = re.compile(r"(\d+(?:\.\d+)?)\s*[Bb](?![a-z])")


def run_status_describe(
    config: dict[str, Any],
    *,
    kind: str,
    method: str | None,
    stages: list[str],
) -> None:
    """Tell the monitor what this run is: kind, algorithm, model, planned stages and a few chips.

    Everything comes from the job config, so a new experiment is described without touching this code.
    """
    try:
        s = get_status()
        if s is None:
            return
        model_cfg = config.get("model") or {}
        name = str(model_cfg.get("name") or "")
        params = _PARAMS.search(name.split("/")[-1])
        model = {"family": "llm", "name": name.split("/")[-1] or None}
        if params:
            model["params"] = params.group(1) + "B"
        badges: list[dict[str, str]] = []
        if config.get("lora"):
            badges.append(
                {"label": f"LoRA r={config['lora'].get('r', '?')}", "icon": "finetune"}
            )
        if model_cfg.get("quantization"):
            badges.append({"label": str(model_cfg["quantization"]), "icon": "gpu"})
        if (config.get("grammar") or {}).get("enabled"):
            badges.append({"label": "grammar", "icon": "tree"})
        if (config.get("retrieval") or {}).get("enabled"):
            badges.append({"label": "retrieval", "icon": "data"})
        dataset = (config.get("dataset") or {}).get("dataset_name")
        if dataset:
            badges.append({"label": str(dataset), "icon": "data"})
        s.describe(
            kind=kind,
            method=method,
            model=model,
            badges=badges,
            stages=[
                {"id": x, "label": _STAGE_LABELS.get(x, x), "icon": x} for x in stages
            ],
        )
    except Exception:
        pass


def run_status_log(
    step: int,
    epoch: float | None = None,
    total_steps: int | None = None,
    **metrics: Any,
) -> None:
    """Record the metrics of one log event (numeric values only; None/NaN are ignored)."""
    try:
        s = get_status()
        if s is None:
            return
        if epoch is not None or total_steps:
            s.set(epoch=epoch, total_steps=total_steps or None)
        s.log(step, **metrics)
    except Exception:
        pass


def run_status_samples(samples: list[Any], kind: str) -> None:
    """Latest completion samples for display (last 6 are kept).

    Items may be plain strings or dicts such as
    ``{"prompt", "output", "gold", "match", "rewards": {component: value}, "total"}``.
    """
    try:
        s = get_status()
        if s is not None:
            s.set(samples=samples, extra={"samples_kind": kind})
    except Exception:
        pass


def run_status_results(results: dict[str, Any], prefix: str = "eval/") -> None:
    """Publish the scalar metrics of a finished evaluation (one nesting level is flattened with ``/``)."""
    try:
        s = get_status()
        if s is None:
            return
        flat: dict[str, float] = {}
        for key, value in results.items():
            if isinstance(value, dict):
                for sub, v in value.items():
                    if isinstance(v, (int, float)) and not isinstance(v, bool):
                        flat[f"{prefix}{key}/{sub}"] = v
            elif isinstance(value, (int, float)) and not isinstance(value, bool):
                flat[f"{prefix}{key}"] = value
        s.log(None, **flat)
        s.event("metriche di valutazione calcolate")
    except Exception:
        pass


def run_status_stage_done(note: str) -> None:
    """End of a stage (SFT, eval): note only. The run is finished at process exit."""
    run_status_set(note=note, eval_active=False)


def _reset_for_tests() -> None:
    global _status, _tried
    _status, _tried = None, False
