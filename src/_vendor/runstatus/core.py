"""Generic, fail-safe run status file: a training writes it, an external monitor reads it.

Atomic writes (tmp + os.replace), throttled, never raises into the caller.
Works for supervised training and RL (reward statistics, running averages, best values).
"""
from __future__ import annotations

import json
import os
import re
import socket
import sys
import tempfile
import time
import uuid
from collections import deque
from datetime import datetime
from fnmatch import fnmatch
from pathlib import Path
from typing import Any, Iterable

SCHEMA = 3
SERIES_EVERY = 15.0  # seconds between sidecar (series file) writes


def default_path() -> Path:
    """$RUNSTATUS_PATH, else <$RUNSTATUS_DIR or run_status>/<SLURM_JOB_ID>.json, else .../run_status.json.

    A dedicated folder keeps these files out of the scheduler's log directory.
    """
    env = os.environ.get("RUNSTATUS_PATH")
    if env:
        return Path(env)
    base = Path(os.environ.get("RUNSTATUS_DIR") or "run_status")
    job = os.environ.get("SLURM_JOB_ID")
    return base / (f"{job}.json" if job else "run_status.json")


_OWN_FILE = re.compile(r"^(\d+|run_status)(\.series)?\.json$")


def _prune(folder: Path, keep_days: float) -> None:
    """Delete our own files (<jobid>.json, <jobid>.series.json, run_status.json) older than ``keep_days``."""
    limit = time.time() - keep_days * 86400
    for f in folder.iterdir():
        try:
            if _OWN_FILE.match(f.name) and f.is_file() and f.stat().st_mtime < limit:
                f.unlink()
        except OSError:
            pass

# direction used to track the best value when `best` does not say otherwise
_AUTO_BEST = (("*loss*", "min"), ("*error*", "min"), ("*perplexity*", "min"), ("*kl*", None),
              ("*reward*", "max"), ("*acc*", "max"), ("*f1*", "max"), ("*score*", "max"),
              ("*rouge*", "max"), ("*bleu*", "max"), ("*success*", "max"))


def _stage(x: Any) -> dict[str, Any]:
    d = {"id": x} if isinstance(x, str) else dict(x)
    d.setdefault("label", d["id"])
    d.setdefault("icon", d["id"])
    d["state"] = "pending"
    return d


def _badge(x: Any) -> dict[str, Any]:
    d = {"label": x} if isinstance(x, str) else dict(x)
    d.setdefault("icon", None)
    return d


def _match(key: str, patterns: Iterable[str] | None) -> bool:
    return bool(patterns) and any(fnmatch(key, p) for p in patterns)  # type: ignore[union-attr]


class RunStatus:
    """
    Args:
        path: status file (default: see default_path). Series go to the sidecar ``<path without .json>.series.json``
            so the main file stays small enough to be read on every poll.
        keep_days: on start, delete our own files older than this in the same folder (None: keep everything).
        kind: free label, e.g. "train", "eval", "rl".
        throttle: min seconds between non-forced writes.
        max_points: max points per series (decimated beyond that).
        include / exclude: glob patterns on metric names (applied before rename).
        rename: {"learning_rate": "lr", ...} applied to metric names.
        best: {"eval_loss": "min", "reward": "max"}; unlisted names use name-based defaults.
        average: glob patterns of metrics that get a running mean and a moving average
            (default: reward metrics, handy for RL).
        window: size of the moving average window.
        series: glob patterns of metrics kept as series (default: all).
        extra: free dict stored under "extra" (hyperparameters, git sha, anything).
        events_max: how many recent events to keep.
        method: the algorithm in use, free text: "sft", "grpo", "ppo", "dpo", "supervised", "pretraining"...
        model: what is being trained, free keys; the usual ones are ``family`` ("llm", "nn", "ml", "cnn",
            "transformer", ...), ``name`` and ``params``.
        stages: the planned stages of the run in order (["sft", "grpo", "eval"], or dicts with ``id``,
            ``label``, ``icon``); the app shows them as a stepper. Move between them with :meth:`stage`.
        badges: free chips for the app, strings or ``{"label", "icon"}`` dicts (e.g. "LoRA", "4-bit").
    Icons are plain ids ("sft", "grpo", "llm", "gpu", ...) that the monitor maps to a picture (ClusterControl draws SVG line icons); an unknown id shows no icon.
    """

    def __init__(self, path: str | os.PathLike | None = None, *, kind: str = "train",
                 name: str | None = None, total_steps: int | None = None,
                 throttle: float = 2.0, max_points: int = 200, keep_days: float | None = 30,
                 include: list[str] | None = None, exclude: list[str] | None = None,
                 rename: dict[str, str] | None = None, best: dict[str, str] | None = None,
                 average: list[str] | None = None, window: int = 50,
                 series: list[str] | None = None, extra: dict[str, Any] | None = None,
                 events_max: int = 20, method: str | None = None, model: dict[str, Any] | None = None,
                 stages: list[Any] | None = None, badges: list[Any] | None = None):
        self.path = Path(path) if path else default_path()
        self.series_path = self.path.with_suffix(".series.json")
        self._series_at = 0.0
        self.throttle, self.max_points = throttle, max_points
        self.include, self.exclude, self.rename = include, exclude or [], rename or {}
        self.best_rules = best or {}
        self.average = ["*reward*"] if average is None else average
        self.window, self.series_patterns = window, series
        self._t0 = time.time()
        self._first: tuple[float, int] | None = None  # (time, step) of the first logged step: rate ignores resume offset
        self._last_write = 0.0
        self._warned = False
        if keep_days is not None and self.path.parent.is_dir():
            _prune(self.path.parent, keep_days)
        self._stride = 1
        self._seen: dict[str, int] = {}
        self._sum: dict[str, tuple[float, int]] = {}
        self._win: dict[str, deque] = {}
        self._events: deque = deque(maxlen=events_max)
        self.d: dict[str, Any] = {
            "schema": SCHEMA,
            "run": {"id": uuid.uuid4().hex[:8], "name": name, "kind": kind, "phase": None,
                    "state": "running", "started_at": self._iso(self._t0),
                    "host": socket.gethostname(), "pid": os.getpid(),
                    "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
                    "method": method, "model": dict(model or {}), "stage": None,
                    "stages": [_stage(x) for x in stages or []], "badges": [_badge(x) for x in badges or []]},
            "progress": {"step": None, "total_steps": total_steps, "epoch": None, "pct": None,
                         "eta_s": None, "steps_per_s": None},
            "metrics": {"latest": {}, "best": {}, "mean": {}, "moving": {}, "stats": {}},
            "series": {},
            "eval": {"active": False, "progress": None},
            "wandb": {},
            "samples": [],
            "events": [],
            "extra": dict(extra or {}),
            "note": None,
        }
        self._write(True)

    @staticmethod
    def _iso(t: float) -> str:
        return datetime.fromtimestamp(t).isoformat(timespec="seconds")

    # ---- generic state -------------------------------------------------
    def set(self, *, phase: str | None = None, note: str | None = None, total_steps: int | None = None,
            epoch: float | None = None, wandb: dict | None = None, samples: list | None = None,
            eval_active: bool | None = None, eval_progress: str | None = None,
            extra: dict[str, Any] | None = None, reset_series: bool = False,
            force: bool = False) -> None:
        """Update non-metric state. ``reset_series`` clears the charts (use when the step counter restarts). ``extra`` is merged into the free-form "extra" dict (None value deletes a key)."""
        try:
            if reset_series:
                self._reset_series()
                force = True
            if phase is not None and phase != self.d["run"]["phase"]:
                self.d["run"]["phase"] = phase
                self.event(f"phase: {phase}", write=False)
                force = True
            if note is not None:
                self.d["note"] = note
            if total_steps is not None:
                self.d["progress"]["total_steps"] = int(total_steps) or None  # 0 means unknown
                self._pct()
            if epoch is not None:
                self.d["progress"]["epoch"] = round(float(epoch), 4)
            if wandb:
                self.d["wandb"].update(wandb)
                force = True
            if samples is not None:
                self.d["samples"] = list(samples[-6:])
                force = True
            if eval_active is not None:
                self.d["eval"]["active"] = eval_active
                force = True
            if eval_progress is not None:
                self.d["eval"]["progress"] = eval_progress
            if extra:
                for k, v in extra.items():
                    if v is None:
                        self.d["extra"].pop(k, None)
                    else:
                        self.d["extra"][k] = v
                force = True
            self._write(force)
        except Exception as e:  # never break the training
            self._warn(e)

    def describe(self, *, kind: str | None = None, method: str | None = None, model: dict[str, Any] | None = None,
                 badges: list[Any] | None = None, stages: list[Any] | None = None) -> None:
        """Set or change what the run is: kind, algorithm, model, chips and the planned stages (replaced when given)."""
        try:
            r = self.d["run"]
            if kind is not None:
                r["kind"] = kind
            if method is not None:
                r["method"] = method
            if model is not None:
                r["model"] = dict(model)
            if badges is not None:
                r["badges"] = [_badge(x) for x in badges]
            if stages is not None:
                r["stages"] = [_stage(x) for x in stages]
            self._write(True)
        except Exception as e:
            self._warn(e)

    def stage(self, id: str, *, label: str | None = None, icon: str | None = None, method: str | None = None,
              total_steps: int | None = None, reset: bool = True) -> None:
        """Enter a stage of the run (e.g. "sft", then "grpo", then "eval").

        The previous stage is closed with the step it reached; a stage not in the plan is appended. The step
        counter, the totals and the charts restart (``reset=False`` keeps them). ``phase`` follows the stage id.
        """
        try:
            r = self.d["run"]
            changed = r["stage"] != id
            if changed:
                self._close_stage("done")
            ent = next((x for x in r["stages"] if x["id"] == id), None)
            if ent is None:
                ent = _stage(id)
                r["stages"].append(ent)
            if label:
                ent["label"] = label
            if icon:
                ent["icon"] = icon
            ent["state"] = "running"
            if changed:
                ent.pop("step", None)
                ent.pop("total", None)
                if reset:
                    self._reset_series()
                self.event(f"stage: {ent['label']}", write=False)
            r["stage"] = r["phase"] = id
            if method is not None:
                r["method"] = method
            if total_steps is not None:
                self.d["progress"]["total_steps"] = int(total_steps) or None
                self._pct()
            self._write(True)
        except Exception as e:
            self._warn(e)

    def _close_stage(self, state: str) -> None:
        r, p = self.d["run"], self.d["progress"]
        for x in r["stages"]:
            if x["id"] == r["stage"] and x["state"] == "running":
                x["state"] = state
                if p["step"] is not None:
                    x["step"] = p["step"]
                if p["total_steps"]:
                    x["total"] = p["total_steps"]

    def _reset_series(self) -> None:
        """New stage: the step counter restarts, so charts, progress and total all start over."""
        self.d["series"], self._seen, self._stride, self._first = {}, {}, 1, None
        self.d["progress"].update(step=None, total_steps=None, pct=None, eta_s=None, steps_per_s=None)

    def event(self, message: str, *, write: bool = True) -> None:
        """Append a short event (checkpoint saved, eval done, ...) to the recent-events list."""
        try:
            self._events.append({"t": self._iso(time.time()), "step": self.d["progress"]["step"], "msg": message})
            self.d["events"] = list(self._events)
            if write:
                self._write(True)
        except Exception as e:
            self._warn(e)

    # ---- metrics -------------------------------------------------------
    def _name(self, key: str) -> str | None:
        if self.include is not None and not _match(key, self.include):
            return None
        if _match(key, self.exclude):
            return None
        return self.rename.get(key, key)

    def _direction(self, name: str) -> str | None:
        if name in self.best_rules:
            return self.best_rules[name]
        for pat, direction in _AUTO_BEST:
            if fnmatch(name.lower(), pat):
                return direction
        return None

    def log(self, step: int | None = None, **metrics: Any) -> None:
        """Record numeric metrics at ``step`` (non-numbers, NaN and None are ignored)."""
        try:
            self._progress(step)
            m = self.d["metrics"]
            for raw, v in metrics.items():
                if not isinstance(v, (int, float)) or isinstance(v, bool) or v != v:
                    continue
                k = self._name(raw)
                if k is None:
                    continue
                m["latest"][k] = v
                direction = self._direction(k)
                if direction in ("min", "max"):
                    cur = m["best"].get(k)
                    m["best"][k] = v if cur is None else (min(cur, v) if direction == "min" else max(cur, v))
                if _match(k, self.average):
                    s, n = self._sum.get(k, (0.0, 0))
                    s, n = s + v, n + 1
                    self._sum[k] = (s, n)
                    m["mean"][k] = round(s / n, 6)
                    w = self._win.setdefault(k, deque(maxlen=self.window))
                    w.append(v)
                    m["moving"][k] = round(sum(w) / len(w), 6)
                    st = m["stats"].setdefault(k, {"min": v, "max": v})
                    st["min"], st["max"] = min(st["min"], v), max(st["max"], v)
                if step is not None and (self.series_patterns is None or _match(k, self.series_patterns)):
                    self._add_point(k, int(step), v)
            self._write(False)
        except Exception as e:
            self._warn(e)

    def _progress(self, step: int | None) -> None:
        if step is None:
            return
        p, now = self.d["progress"], time.time()
        p["step"] = int(step)
        if self._first is None:
            self._first = (now, int(step))
        t0, s0 = self._first
        if step > s0 and now > t0:
            rate = (step - s0) / (now - t0)
            p["steps_per_s"] = round(rate, 4)
            tot = p["total_steps"]
            if tot:
                p["eta_s"] = max(0, int((tot - step) / rate))
        self._pct()

    def _pct(self) -> None:
        """Never report more than 100%: a step past the declared total means the total was wrong, so raise it."""
        p = self.d["progress"]
        if p["total_steps"] and p["step"] is not None:
            p["total_steps"] = max(p["total_steps"], p["step"])
            p["pct"] = round(100 * p["step"] / p["total_steps"], 2)

    def _add_point(self, key: str, step: int, value: float) -> None:
        # keep one point every `stride`; when full, drop every other point and double the stride
        n = self._seen.get(key, 0)
        self._seen[key] = n + 1
        if n % self._stride:
            return
        s = self.d["series"].setdefault(key, [])
        s.append([step, float(f"{value:.6g}")])  # 6 significant digits: the sidecar travels over slow links
        if len(s) > self.max_points:
            self._stride *= 2
            for k in self.d["series"]:
                self.d["series"][k] = self.d["series"][k][::2]

    # ---- lifecycle -----------------------------------------------------
    def finish(self, note: str | None = None) -> None:
        self._end("finished", note)

    def fail(self, note: str | None = None) -> None:
        self._end("failed", note)

    def _end(self, state: str, note: str | None) -> None:
        try:
            self.d["run"]["state"] = state
            self.d["run"]["ended_at"] = self._iso(time.time())
            self.d["eval"]["active"] = False
            self._close_stage("done" if state == "finished" else "failed")
            if note is not None:
                self.d["note"] = note
            if state == "finished" and self.d["progress"]["total_steps"]:
                self.d["progress"].update(pct=100.0, eta_s=0)
            self.event(state, write=False)
            self._write(True)
        except Exception as e:
            self._warn(e)

    def __enter__(self) -> "RunStatus":
        return self

    def __exit__(self, et, ev, tb) -> bool:
        if self.d["run"]["state"] == "running":
            self.fail(f"{et.__name__}: {ev}") if et else self.finish()
        return False

    # ---- output --------------------------------------------------------
    def _gpu(self) -> dict | None:
        torch = sys.modules.get("torch")  # only if the training already imported it
        try:
            if torch is not None and torch.cuda.is_available():
                return {"mem_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2)}
        except Exception:
            pass
        return None

    def _write(self, force: bool) -> None:
        now = time.monotonic()
        if not force and now - self._last_write < self.throttle:
            return
        self._last_write = now
        self.d["run"]["updated_at"] = self._iso(time.time())
        out = {k: v for k, v in self.d.items() if k != "series"}
        out["gpu"] = self._gpu()
        out["series_file"] = self.series_path.name
        self._dump(self.path, out)
        if force or time.monotonic() - self._series_at >= SERIES_EVERY:
            self._series_at = time.monotonic()
            self._dump(self.series_path, {"schema": SCHEMA, "run_id": self.d["run"]["id"],
                                          "total_steps": self.d["progress"]["total_steps"],
                                          "phase": self.d["run"]["phase"], "series": self.d["series"]})

    def _dump(self, target: Path, payload: dict) -> None:
        tmp = None
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            fd, tmp = tempfile.mkstemp(dir=target.parent, prefix=".run_status.", suffix=".tmp")
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, separators=(",", ":"))
            os.replace(tmp, target)
        except Exception as e:
            self._warn(e)
            if tmp:
                try:
                    os.unlink(tmp)
                except OSError:
                    pass

    def _warn(self, e: Exception) -> None:
        if not self._warned:
            self._warned = True
            print(f"[runstatus] disabled after first error: {e!r}")
