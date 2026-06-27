#!/usr/bin/env python3
"""Live monitor for the neuro-symbolic T2G training pipeline.

Shows the status of every job in the pipeline (completed, failed, running,
waiting) and, for the active training job, displays the current curriculum
stage and training step in real time.

When the active job finishes, the monitor automatically picks up the next
job's log.

Usage:
    python3 -m src.utils.chain_monitor              # default, auto-detect
    python3 -m src.utils.chain_monitor --poll 30    # poll every 30s (default 15)

Designed to run on the cluster login node (same node as the watcher).
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# ── Config ────────────────────────────────────────────────────────────────────
PROJ_DIR = Path(os.environ.get("HOME", "~")) / "neuro_symbolic_t2g"
STATE_DIR = PROJ_DIR / ".chain_state"
CHAIN_FILE = STATE_DIR / "job_chain"
CHAIN_PID_FILE = STATE_DIR / "chain_pid"
CACHE_FILE = STATE_DIR / "monitor_cache"
LOGS_DIR = PROJ_DIR / "logs"
CHAIN_LOG = LOGS_DIR / "chain_watcher.log"
ERRORS_FILE = STATE_DIR / "chain_errors"

# Module-level setting for sample display (set by main() from --samples arg)
_SAMPLE_MAX_LINES: int = 0  # 0 = no limit

# Regex patterns for training log parsing
_KV_STEP = re.compile(r"^\s+step=(\d+)\s+loss=")
_KV_REWARD = re.compile(r"reward=([+-]?\d+\.\d+)")
# TRL dict-style log: 'reward': 0.5025833547115326
_DICT_REWARD = re.compile(r"'reward':\s*([+-]?\d+\.\d+)")
_STAGE_START = re.compile(r"\[stage (\d+)\] steps=(\d+)")
_STAGE_DONE = re.compile(r"\[stage (\d+)\] (\S+) completed")
# tqdm progress bar: " 47%|████▋     | 420/900 [29:23<25:49"
_TQDM_PROGRESS = re.compile(r"^\s*\d+%\|.*\|\s*(\d+)/(\d+)\s*\[", re.MULTILINE)
# Eval generation bar: "Generating:  45%|████▍| 17/38 ["
_TQDM_GENERATING = re.compile(r"Generating.*\|\s*(\d+)/(\d+)\s*\[")
# tqdm time info: "[04:25<37:02, 33.17s/it]" or "[1:23:45<2:03:04"
_TQDM_TIME = re.compile(r"\[([\d:]+)<([\d:]+)")
_EVAL_CHECKPOINT = re.compile(r"Evaluating: (.+)")
_EVAL_STAGE_NUM = re.compile(r"Stage (\d+)")
_EVAL_PASS = re.compile(r"(.+?)\s+Pass@1:\s+([\d.]+)")
_EVAL_COMPLETE = re.compile(r"Evaluation complete")

# ── ANSI color helpers ────────────────────────────────────────────────────────
_RST = "\033[0m"
_BOLD = "\033[1m"
_DIM = "\033[2m"
_GREEN = "\033[32m"
_RED = "\033[31m"
_YELLOW = "\033[33m"
_CYAN = "\033[36m"
_BLUE = "\033[34m"
_MAGENTA = "\033[35m"
_WHITE = "\033[97m"
_GRAY = "\033[90m"


# ── Data structures ───────────────────────────────────────────────────────────
@dataclass
class JobInfo:
    """Info about a single job in the chain."""

    job_type: str  # "train" or "eval"
    config: str  # config path
    tag: str  # model tag (e.g. "qwen05")
    slurm_id: str | None = None
    state: str = "PENDING"  # PENDING, STARTING, RUNNING, COMPLETED, FAILED
    step: int = 0  # current training step
    stage_total: int = 0  # total steps for current stage
    eval_label: str = ""  # current eval label
    eval_pass: str = ""  # last eval pass@1
    eval_stages: dict[str, str] = field(default_factory=dict)  # per-stage pass@1
    eval_step_total: int = 0  # total generation batches for eval
    exit_code: str = ""
    elapsed: str = ""  # elapsed time from squeue (e.g. "12:34")
    error_type: str = ""  # error classification (OOM, CUDA_ERROR, TIMEOUT, etc.)
    error_snippet: str = ""  # short error description from log
    tqdm_elapsed: str = ""  # elapsed time from tqdm bar
    tqdm_eta: str = ""  # remaining time from tqdm bar
    last_reward: str = ""  # last logged mean reward
    completion_samples: list[str] = field(default_factory=list)

    @property
    def label(self) -> str:
        return f"{self.job_type}-{self.tag}"


def _run(cmd: str) -> str:
    """Run a shell command and return stdout."""
    try:
        r = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
        return r.stdout.strip()
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return ""


# ── Monitor cache ─────────────────────────────────────────────────────────────
def _load_cache() -> dict:
    """Load the monitor cache from disk."""
    if not CACHE_FILE.exists():
        return {"jobs": {}, "pipeline_jobs": []}
    try:
        cache = json.loads(CACHE_FILE.read_text(encoding="utf-8"))
        cache.setdefault("pipeline_jobs", [])
        return cache
    except (ValueError, OSError):
        return {"jobs": {}, "pipeline_jobs": []}


def _save_cache(cache: dict) -> None:
    """Write the monitor cache to disk."""
    import tempfile

    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            delete=False,
            dir=str(CACHE_FILE.parent),
        ) as f:
            f.write(json.dumps(cache, indent=2, ensure_ascii=False))
            tempname = f.name
        os.replace(tempname, CACHE_FILE)
    except OSError:
        pass


def _cache_update_job(job: JobInfo) -> None:
    """Update the cache with the latest info from a job."""
    cache = _load_cache()
    key = f"{job.job_type}-{job.tag}"

    if key not in cache.get("pipeline_jobs", []):
        return

    if job.state != "PENDING":
        entry = cache["jobs"].get(key, {})
        entry["state"] = job.state
        if job.slurm_id:
            entry["slurm_id"] = job.slurm_id
        if job.exit_code:
            entry["exit_code"] = job.exit_code
        if job.job_type == "eval" and job.eval_pass:
            entry["eval_pass"] = job.eval_pass
        if job.job_type == "eval" and job.eval_stages:
            merged = entry.get("eval_stages", {})
            merged.update(job.eval_stages)
            entry["eval_stages"] = merged
        if job.last_reward:
            entry["last_reward"] = job.last_reward
        if job.error_type:
            entry["error_type"] = job.error_type
        if job.error_snippet:
            entry["error_snippet"] = job.error_snippet
        entry["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        cache["jobs"][key] = entry

    _save_cache(cache)


def _cache_enrich_job(job: JobInfo) -> None:
    """Fill in missing fields on a job from cached data."""
    cache = _load_cache()
    key = f"{job.job_type}-{job.tag}"
    entry = cache["jobs"].get(key)
    if not entry:
        return

    if job.state == "PENDING" and entry.get("state") in ("COMPLETED", "FAILED"):
        job.state = entry["state"]
    if not job.slurm_id and entry.get("slurm_id"):
        job.slurm_id = entry["slurm_id"]
    if not job.eval_pass and entry.get("eval_pass"):
        job.eval_pass = entry["eval_pass"]
    if entry.get("eval_stages"):
        merged = dict(entry["eval_stages"])
        merged.update(job.eval_stages)
        job.eval_stages = merged
    if not job.exit_code and entry.get("exit_code"):
        job.exit_code = entry["exit_code"]
    if not job.last_reward and entry.get("last_reward"):
        job.last_reward = entry["last_reward"]
    if not job.error_type and entry.get("error_type"):
        job.error_type = entry["error_type"]
    if not job.error_snippet and entry.get("error_snippet"):
        job.error_snippet = entry["error_snippet"]


# ── Error log (.chain_errors) ─────────────────────────────────────────────────
def _load_errors() -> dict[str, list[dict]]:
    """Load errors from .chain_errors (JSONL format)."""
    if not ERRORS_FILE.exists():
        return {}
    errors: dict[str, list[dict]] = {}
    try:
        for line in ERRORS_FILE.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except (ValueError, TypeError):
                continue
            key = f"{entry.get('job_type', 'train')}-{entry.get('tag', '')}"
            errors.setdefault(key, []).append(entry)
    except OSError:
        pass
    return errors


# ── SLURM queries ─────────────────────────────────────────────────────────────
def _get_slurm_jobs() -> dict[str, tuple[str, str, str]]:
    """Get recent SLURM jobs. Returns {job_name: (job_id, state, exit_code)}."""
    out = _run(
        "sacct --me --starttime=$(date -d '7 days ago' +%Y-%m-%d) "
        "--format=JobID%20,JobName%30,State%15,ExitCode%10 --noheader --parsable2"
    )
    jobs: dict[str, tuple[str, str, str]] = {}
    for line in out.splitlines():
        parts = line.split("|")
        if len(parts) < 4:
            continue
        job_id, name, state, exit_code = parts[0], parts[1], parts[2], parts[3]
        if "." in job_id:
            continue
        jobs[name] = (job_id, state, exit_code)
    return jobs


def _get_active_job() -> tuple[str, str, str] | None:
    """Return (job_id, job_name, elapsed) of the currently running job, or None."""
    out = _run('squeue --me --noheader --format="%i %j %T %M"')
    for line in out.splitlines():
        parts = line.split()
        if len(parts) >= 4 and parts[2] in ("RUNNING", "PENDING"):
            return parts[0], parts[1], parts[3]
        elif len(parts) >= 3 and parts[2] in ("RUNNING", "PENDING"):
            return parts[0], parts[1], ""
    return None


# ── Chain file parsing ────────────────────────────────────────────────────────
def _parse_chain_log() -> list[tuple[str, str | None]]:
    """Parse chain_watcher.log to find submitted job names and SLURM IDs."""
    if not CHAIN_LOG.exists():
        return []
    submitted: list[tuple[str, str | None]] = []
    pattern = re.compile(r"\[chain\] Sottometto: (\w+) (\S+)")
    job_id_pattern = re.compile(r"\[chain\] Job ID: (\d+)")
    pending_name: str | None = None
    for line in CHAIN_LOG.read_text(errors="replace").splitlines():
        m = pattern.search(line)
        if m:
            if pending_name is not None:
                submitted.append((pending_name, None))
            pending_name = f"{m.group(1)}-{m.group(2)}"
            continue
        m = job_id_pattern.search(line)
        if m and pending_name is not None:
            submitted.append((pending_name, m.group(1)))
            pending_name = None
    if pending_name is not None:
        submitted.append((pending_name, None))
    return submitted


def _read_pending_chain() -> list[tuple[str, str, str]]:
    """Read remaining entries from .job_chain file."""
    if not CHAIN_FILE.exists():
        return []
    entries = []
    for line in CHAIN_FILE.read_text().strip().splitlines():
        parts = line.strip().split(":")
        if len(parts) >= 3:
            entries.append((parts[0], parts[1], parts[2]))
    return entries


def _find_log_file(job_type: str, slurm_id: str) -> Path | None:
    """Find the SLURM log file for a job."""
    log = LOGS_DIR / f"slurm-{job_type}-{slurm_id}.log"
    if log.exists():
        return log
    return None


# ── Log parsing ───────────────────────────────────────────────────────────────
def _tail_lines(log_path: Path, n: int = 500) -> list[str]:
    """Read the last N lines of a file efficiently using tail."""
    out = _run(f"tail -n {n} '{log_path}'")
    if out:
        return out.splitlines()
    try:
        return log_path.read_text(errors="replace").splitlines()[-n:]
    except OSError:
        return []


def _grep_lines(log_path: Path, pattern: str, max_count: int = 20) -> list[str]:
    """Grep a file for a pattern, returning matching lines."""
    out = _run(f"grep -E '{pattern}' '{log_path}' | tail -n {max_count}")
    return out.splitlines() if out else []


def _extract_completion_samples(
    lines: list[str],
    max_lines: int = 0,
) -> list[str]:
    """Extract a compact view of the last sample from the log."""
    block: list[str] = []
    in_block = False
    for line in lines:
        stripped = line.strip()
        if "COMPLETION SAMPLES" in stripped:
            in_block = True
            block = []
            continue
        if in_block:
            block.append(stripped)
            if stripped.startswith("\u2550" * 10) and len(block) > 3:
                in_block = False

    if not block:
        return []

    prompt_text = ""
    think_lines: list[str] = []
    output_lines: list[str] = []
    rewards_line = ""
    total_line = ""
    difficulty = ""
    section = ""
    found_first = False
    for line in block:
        if line.startswith("Sample ") and found_first:
            break
        if line.startswith("Sample "):
            found_first = True
            dm = re.search(r"\[difficulty=(\w+)\]", line)
            if dm:
                difficulty = dm.group(1)
            continue
        if line.startswith("PROMPT:"):
            prompt_text = line[len("PROMPT:") :].strip()
            section = "prompt"
            continue
        if line == "THINK:":
            section = "think"
            continue
        if line == "OUTPUT:":
            section = "output"
            continue
        if line.startswith("REWARDS:"):
            rewards_line = line
            section = "rewards"
            continue
        if line.startswith("TOTAL:"):
            total_line = line
            section = ""
            continue
        if section == "rewards":
            if re.search(r"\w+=[\+\-]?\d+\.\d+", line):
                rewards_line += "  " + line
                continue
            section = ""
        if section == "prompt":
            prompt_text += " " + line
        elif section == "think":
            think_lines.append(line)
        elif section == "output":
            output_lines.append(line)

    if not output_lines and not rewards_line:
        return []

    diff_colors = {"simple": _GREEN, "medium": _YELLOW, "hard": _RED}
    diff_color = diff_colors.get(difficulty, _DIM)
    diff_badge = f" {diff_color}[{difficulty}]{_RST}" if difficulty else ""

    result = [f"{_DIM}─── Last completion{_RST}{diff_badge} {_DIM}───{_RST}"]

    if prompt_text:
        result.append(f"  {_CYAN}PROMPT:{_RST} {prompt_text.strip()}")

    if think_lines:
        think_text = " ".join(tl.strip() for tl in think_lines).strip()
        if max_lines > 0 and len(think_text) > 80:
            think_text = think_text[:80] + "..."
        result.append(f"  {_MAGENTA}<think>{_RST} {_MAGENTA}{think_text}{_RST}")

    if output_lines:
        limit = max_lines if max_lines > 0 else len(output_lines)
        display = output_lines[:limit]
        if len(output_lines) > limit:
            display.append("[...]")
        for dl in display:
            result.append(f"  {_DIM}{dl}{_RST}")

    if rewards_line:
        parts = re.findall(r"(\w+)=([+-]?\d+\.\d+)", rewards_line)
        if parts:
            col_w = 17
            colored_parts: list[str] = []
            for name, val_str in parts:
                v = float(val_str)
                raw = f"{name}={val_str}"
                pad = " " * max(0, col_w - len(raw))
                if v > 0:
                    colored_parts.append(f"{_GREEN}{raw}{_RST}{pad}")
                elif v < 0:
                    colored_parts.append(f"{_RED}{raw}{_RST}{pad}")
                else:
                    colored_parts.append(f"{_GRAY}{raw}{_RST}{pad}")
            has_reasoning = any(n == "reasoning" for n, _ in parts)
            split = 4 if has_reasoning else 3
            row1 = colored_parts[:split]
            row2 = colored_parts[split:]
            result.append(f"  REWARDS: {''.join(row1)}")
            if row2:
                result.append(f"           {''.join(row2)}")
        else:
            result.append(f"  {_CYAN}{rewards_line}{_RST}")

    if total_line:
        tm = re.search(r"([+-]?\d+\.\d+)", total_line)
        if tm:
            tv = float(tm.group(1))
            tc = _GREEN if tv > 0 else (_RED if tv < 0 else _GRAY)
            result.append(f"  {tc}{total_line.strip()}{_RST}")

    return result


def _parse_training_log(log_path: Path, job: JobInfo) -> None:
    """Parse a training log file and update job state."""
    stage_lines = _grep_lines(log_path, r"\[stage [0-9]+\]")
    for line in stage_lines:
        m = _STAGE_START.search(line)
        if m:
            job.stage_total = int(m.group(2))

        m = _STAGE_DONE.search(line)
        if m:
            pass  # stage completion handled elsewhere

    tail = _tail_lines(log_path, n=1500)

    if job.stage_total == 0:
        max_steps_lines = _grep_lines(log_path, r"max_steps=", max_count=1)
        for line in max_steps_lines:
            ms = re.search(r"max_steps=(\d+)", line)
            if ms:
                job.stage_total = int(ms.group(1))
                break

    sample_tail = _tail_lines(log_path, n=1500)
    samples = _extract_completion_samples(sample_tail, max_lines=_SAMPLE_MAX_LINES)
    if samples:
        job.completion_samples = samples

    for line in reversed(tail):
        m = _KV_STEP.search(line)
        if m:
            job.step = int(m.group(1))
            mr = _KV_REWARD.search(line)
            if mr:
                job.last_reward = mr.group(1)
            break
        m = _TQDM_PROGRESS.search(line)
        if m:
            job.step = int(m.group(1))
            if job.stage_total == 0:
                job.stage_total = int(m.group(2))
            mt = _TQDM_TIME.search(line)
            if mt:
                job.tqdm_elapsed = mt.group(1)
                job.tqdm_eta = mt.group(2)
            break

    if not job.last_reward:
        for line in reversed(tail):
            mr = _DICT_REWARD.search(line)
            if mr:
                job.last_reward = mr.group(1)
                break
            if "step=" in line:
                mr = _KV_REWARD.search(line)
                if mr:
                    job.last_reward = mr.group(1)
                    break


def _parse_eval_log(log_path: Path, job: JobInfo) -> None:
    """Parse an eval log file and update job state."""
    tail = _tail_lines(log_path, n=500)

    for line in tail:
        m = _EVAL_CHECKPOINT.search(line)
        if m:
            job.eval_label = m.group(1)
        m = _EVAL_PASS.search(line)
        if m:
            job.eval_label = m.group(1)
            job.eval_pass = m.group(2)
            label = m.group(1).strip()
            if "baseline" in label.lower():
                job.eval_stages["baseline"] = m.group(2)
            elif "grpo" in label.lower():
                job.eval_stages["stage_1"] = m.group(2)
        if _EVAL_COMPLETE.search(line):
            job.eval_label = "COMPLETE"

    pass_lines = _grep_lines(log_path, r"Pass@1:", max_count=10)
    for pl in pass_lines:
        mp = _EVAL_PASS.search(pl)
        if mp:
            label = mp.group(1).strip()
            if "baseline" in label.lower():
                job.eval_stages["baseline"] = mp.group(2)
            elif "grpo" in label.lower():
                job.eval_stages["stage_1"] = mp.group(2)

    for line in reversed(tail):
        m = _TQDM_GENERATING.search(line)
        if m:
            job.step = int(m.group(1))
            job.eval_step_total = int(m.group(2))
            mt = _TQDM_TIME.search(line)
            if mt:
                job.tqdm_elapsed = mt.group(1)
                job.tqdm_eta = mt.group(2)
            break


# ── Build full pipeline view ──────────────────────────────────────────────────
def _build_pipeline() -> list[JobInfo]:
    """Reconstruct the full pipeline from chain log + chain file + sacct."""
    slurm_jobs = _get_slurm_jobs()
    active = _get_active_job()
    pending = _read_pending_chain()

    has_pipeline = CHAIN_PID_FILE.exists() or (CHAIN_FILE.exists() and pending)
    submitted_names = _parse_chain_log() if has_pipeline else []

    jobs: list[JobInfo] = []
    seen_names: set[str] = set()
    chain_log_ids: dict[str, str] = {}

    # 1. Already submitted jobs (from chain log)
    for name, chain_slurm_id in submitted_names:
        if name in seen_names:
            continue
        seen_names.add(name)

        parts = name.split("-", 1)
        job_type = parts[0] if parts else "train"
        tag = parts[1] if len(parts) > 1 else name

        job = JobInfo(job_type=job_type, config="", tag=tag)

        if chain_slurm_id:
            chain_log_ids[name] = chain_slurm_id

        if name in slurm_jobs:
            sid, state, exit_code = slurm_jobs[name]
            job.slurm_id = sid
            job.exit_code = exit_code
            if state == "RUNNING":
                job.state = "RUNNING"
            elif state == "PENDING":
                job.state = "PENDING"
            elif state == "COMPLETED":
                job.state = "COMPLETED" if exit_code == "0:0" else "FAILED"
            elif state in (
                "FAILED",
                "NODE_FAIL",
                "OUT_OF_MEMORY",
                "TIMEOUT",
                "CANCELLED",
            ):
                job.state = "FAILED"
            else:
                job.state = state

            log_file = _find_log_file(job_type, sid)
            if log_file:
                if job_type == "train":
                    _parse_training_log(log_file, job)
                else:
                    _parse_eval_log(log_file, job)
        elif active and active[1] == name:
            job.slurm_id = active[0]
            job.state = "RUNNING"
            job.elapsed = active[2]
            log_file = _find_log_file(job_type, active[0])
            if log_file:
                if job_type == "train":
                    _parse_training_log(log_file, job)
                else:
                    _parse_eval_log(log_file, job)

        jobs.append(job)

    last_active_idx = -1
    for i, job in enumerate(jobs):
        if job.state != "PENDING":
            last_active_idx = i
    for i, job in enumerate(jobs):
        if job.state == "PENDING" and i < last_active_idx:
            job.state = "COMPLETED"
            slurm_id = job.slurm_id or chain_log_ids.get(f"{job.job_type}-{job.tag}")
            if slurm_id:
                log_file = _find_log_file(job.job_type, slurm_id)
                if log_file:
                    if job.job_type == "train":
                        _parse_training_log(log_file, job)
                    else:
                        _parse_eval_log(log_file, job)

    for job in jobs:
        if job.state == "RUNNING":
            has_progress = job.step > 0 or job.eval_label
            if not has_progress:
                job.state = "STARTING"

    # 2. Pending jobs (still in .job_chain)
    for job_type, cfg, tag in pending:
        name = f"{job_type}-{tag}"
        if name in seen_names:
            continue
        seen_names.add(name)

        job = JobInfo(job_type=job_type, config=cfg, tag=tag, state="PENDING")

        if name in slurm_jobs:
            sid, state, exit_code = slurm_jobs[name]
            job.slurm_id = sid
            job.exit_code = exit_code
            if state == "COMPLETED":
                job.state = "COMPLETED" if exit_code == "0:0" else "FAILED"
            elif state == "RUNNING":
                job.state = "RUNNING"
            elif state in (
                "FAILED",
                "NODE_FAIL",
                "OUT_OF_MEMORY",
                "TIMEOUT",
                "CANCELLED",
            ):
                job.state = "FAILED"

            log_file = _find_log_file(job_type, sid)
            if log_file:
                if job_type == "train":
                    _parse_training_log(log_file, job)
                else:
                    _parse_eval_log(log_file, job)
        elif active and active[1] == name:
            job.slurm_id = active[0]
            job.state = "RUNNING"
            job.elapsed = active[2]
            log_file = _find_log_file(job_type, active[0])
            if log_file:
                if job_type == "train":
                    _parse_training_log(log_file, job)
                else:
                    _parse_eval_log(log_file, job)

        jobs.append(job)

    # 3. Standalone mode — discover jobs from squeue/sacct
    if not jobs:

        def _parse_job_name(name: str) -> tuple[str, str] | None:
            parts = name.split("-", 1)
            if parts[0] in ("train", "eval"):
                tag = parts[1] if len(parts) == 2 else ""
                return parts[0], tag
            return None

        for name, (sid, state, exit_code) in sorted(
            slurm_jobs.items(), key=lambda x: x[1][0]
        ):
            parsed = _parse_job_name(name)
            if not parsed:
                continue
            job_type, tag = parsed
            job = JobInfo(job_type=job_type, config="", tag=tag)
            job.slurm_id = sid
            job.exit_code = exit_code
            if state == "RUNNING":
                job.state = "RUNNING"
            elif state == "PENDING":
                job.state = "PENDING"
            elif state == "COMPLETED":
                job.state = "COMPLETED" if exit_code == "0:0" else "FAILED"
            elif state in (
                "FAILED",
                "NODE_FAIL",
                "OUT_OF_MEMORY",
                "TIMEOUT",
                "CANCELLED",
            ):
                job.state = "FAILED"
            else:
                job.state = state

            log_file = _find_log_file(job_type, sid)
            if log_file:
                if job_type == "train":
                    _parse_training_log(log_file, job)
                else:
                    _parse_eval_log(log_file, job)
            jobs.append(job)

        if active:
            a_name = active[1]
            existing_names = set()
            for j in jobs:
                existing_names.add(f"{j.job_type}-{j.tag}" if j.tag else j.job_type)
            if a_name not in existing_names:
                parsed = _parse_job_name(a_name)
                if parsed:
                    job_type, tag = parsed
                    job = JobInfo(
                        job_type=job_type,
                        config="",
                        tag=tag,
                        state="RUNNING",
                        slurm_id=active[0],
                        elapsed=active[2],
                    )
                    log_file = _find_log_file(job.job_type, active[0])
                    if log_file:
                        if job.job_type == "train":
                            _parse_training_log(log_file, job)
                        else:
                            _parse_eval_log(log_file, job)
                    jobs.append(job)

    # ── Cache integration ─────────────────────────────────────────────────
    for job in jobs:
        _cache_enrich_job(job)

    for job in jobs:
        key = f"{job.job_type}-{job.tag}"
        if not job.slurm_id and key in chain_log_ids:
            job.slurm_id = chain_log_ids[key]

    for job in jobs:
        _cache_update_job(job)

    cache = _load_cache()
    pipeline_keys = set(cache.get("pipeline_jobs", []))
    seen_keys = {f"{j.job_type}-{j.tag}" for j in jobs}
    for key in cache.get("pipeline_jobs", []):
        if key in seen_keys:
            continue
        entry = cache["jobs"].get(key, {})
        parts = key.split("-", 1)
        if len(parts) != 2:
            continue
        job_type, tag = parts[0], parts[1]
        job = JobInfo(
            job_type=job_type,
            config="",
            tag=tag,
            state=entry.get("state", "COMPLETED"),
            eval_pass=entry.get("eval_pass", ""),
            slurm_id=entry.get("slurm_id"),
            exit_code=entry.get("exit_code", ""),
        )
        if entry.get("last_reward"):
            job.last_reward = entry["last_reward"]
        if entry.get("eval_stages"):
            job.eval_stages = entry["eval_stages"]
        jobs.append(job)

    if pipeline_keys:
        jobs = [j for j in jobs if f"{j.job_type}-{j.tag}" in pipeline_keys]

    pipeline_order = cache.get("pipeline_jobs", [])
    if pipeline_order:
        order_map = {k: i for i, k in enumerate(pipeline_order)}
        jobs.sort(
            key=lambda j: order_map.get(f"{j.job_type}-{j.tag}", len(pipeline_order))
        )

    # ── Attach error info from .chain_errors ──────────────────────────
    all_errors = _load_errors()
    for job in jobs:
        if job.state == "FAILED" and not job.error_type:
            key = f"{job.job_type}-{job.tag}"
            errs = all_errors.get(key, [])
            if errs:
                last_unresolved = [e for e in errs if not e.get("resolved", False)]
                err = last_unresolved[-1] if last_unresolved else errs[-1]
                job.error_type = err.get("error_type", "")
                job.error_snippet = err.get("error_snippet", "")

    return jobs


# ── Time helpers ──────────────────────────────────────────────────────────────
def _parse_elapsed_seconds(elapsed: str) -> int | None:
    """Parse squeue elapsed time to seconds."""
    if not elapsed:
        return None
    try:
        parts = elapsed.split("-")
        if len(parts) == 2:
            days = int(parts[0])
            rest = parts[1]
        else:
            days = 0
            rest = parts[0]
        t = rest.split(":")
        if len(t) == 3:
            return days * 86400 + int(t[0]) * 3600 + int(t[1]) * 60 + int(t[2])
        elif len(t) == 2:
            return days * 86400 + int(t[0]) * 60 + int(t[1])
        return None
    except (ValueError, IndexError):
        return None


def _format_duration(seconds: int) -> str:
    """Format seconds as human-readable duration."""
    if seconds < 60:
        return f"{seconds}s"
    m, s = divmod(seconds, 60)
    if m < 60:
        return f"{m}m{s:02d}s"
    h, m = divmod(m, 60)
    return f"{h}h{m:02d}m"


def _estimate_eta(job: JobInfo) -> str:
    """Return tqdm ETA if available, otherwise estimate from elapsed."""
    if job.tqdm_eta:
        return job.tqdm_eta

    elapsed_s = _parse_elapsed_seconds(job.elapsed)
    if not elapsed_s or elapsed_s < 30:
        return ""

    if job.job_type == "train" and job.step > 0 and job.stage_total > 0:
        remaining_steps = job.stage_total - job.step
        if remaining_steps <= 0:
            return ""
        secs_per_step = elapsed_s / job.step
        eta = int(secs_per_step * remaining_steps)
        return _format_duration(eta)
    elif job.job_type == "eval" and job.step > 0 and job.eval_step_total > 0:
        remaining = job.eval_step_total - job.step
        if remaining <= 0:
            return ""
        secs_per_step = elapsed_s / job.step
        eta = int(secs_per_step * remaining)
        return _format_duration(eta)
    return ""


def _estimate_total_eta(job: JobInfo) -> str:
    """Estimate total remaining time including future stages.

    For standard (non-curriculum) T2G training this gives the same
    result as ``_estimate_eta`` — the ``(job ~...h)`` suffix only
    appears when curriculum multi-stage training is active.

    For train: uses step speed × remaining steps.
    For eval: uses batch speed × remaining batches.
    """
    speed_elapsed_s = _parse_elapsed_seconds(job.tqdm_elapsed)
    speed_steps = job.step
    if not speed_elapsed_s or speed_elapsed_s < 10:
        speed_elapsed_s = _parse_elapsed_seconds(job.elapsed)
        if not speed_elapsed_s or speed_elapsed_s < 30:
            return ""

    if job.job_type == "train" and speed_steps > 0:
        secs_per_step = speed_elapsed_s / speed_steps
        remaining_current = max(0, job.stage_total - job.step)
        if remaining_current <= 0:
            return ""
        eta = int(secs_per_step * remaining_current)
        return _format_duration(eta)

    elif job.job_type == "eval" and speed_steps > 0 and job.eval_step_total > 0:
        secs_per_batch = speed_elapsed_s / speed_steps
        remaining_current = max(0, job.eval_step_total - job.step)
        if remaining_current <= 0:
            return ""
        eta = int(secs_per_batch * remaining_current)
        return _format_duration(eta)

    return ""


# ── Display ───────────────────────────────────────────────────────────────────
_STATE_ICONS = {
    "COMPLETED": f"{_GREEN}✓{_RST}",
    "FAILED": f"{_RED}✗{_RST}",
    "RUNNING": f"{_CYAN}▶{_RST}",
    "STARTING": f"{_YELLOW}»{_RST}",
    "PENDING": f"{_GRAY}·{_RST}",
}

_STATE_COLORS = {
    "COMPLETED": _GREEN,
    "FAILED": _RED,
    "RUNNING": _CYAN,
    "STARTING": _YELLOW,
    "PENDING": _GRAY,
}

_TYPE_COLORS = {
    "train": _BLUE,
    "eval": _MAGENTA,
}


def _format_status(job: JobInfo) -> str:
    """Format a single line for a job."""
    icon = _STATE_ICONS.get(job.state, "?")
    sc = _STATE_COLORS.get(job.state, "")
    tc = _TYPE_COLORS.get(job.job_type, "")
    slurm = f"{_DIM}[{job.slurm_id}]{_RST}" if job.slurm_id else ""

    detail = ""
    if job.state == "FAILED":
        parts = []
        if job.error_type:
            parts.append(f"{_RED}{_BOLD}{job.error_type}{_RST}")
        if job.exit_code:
            parts.append(f"{_RED}exit={job.exit_code}{_RST}")
        detail = " " + " ".join(parts) if parts else ""

    def _vpad(s: str, width: int) -> str:
        visible = len(re.sub(r"\033\[[0-9;]*m", "", s))
        return s + " " * max(0, width - visible)

    label = (
        f"{tc}{job.job_type}{_RST}-{_BOLD}{job.tag}{_RST}"
        if job.tag
        else f"{tc}{job.job_type}{_RST}"
    )
    state_str = f"{sc}{job.state}{_RST}"

    return (
        f" {icon}  {_vpad(label, 30)} {_vpad(slurm, 12)} {_vpad(state_str, 12)}{detail}"
    )


def _watcher_status() -> str:
    """Check if the watcher process is alive."""
    if not CHAIN_PID_FILE.exists():
        return f"{_RED}Watcher: OFF{_RST}"
    try:
        pid = CHAIN_PID_FILE.read_text().strip()
        result = _run(f"ps -p {pid} -o pid= 2>/dev/null")
        if result:
            return f"{_GREEN}Watcher: ON{_RST} {_DIM}(PID {pid}){_RST}"
        else:
            return f"{_RED}Watcher: DEAD{_RST} {_DIM}(PID {pid}){_RST}"
    except Exception:
        return f"{_RED}Watcher: UNKNOWN{_RST}"


def _display(
    jobs: list[JobInfo],
    show_table: bool = True,
    show_samples: bool = False,
    show_metrics: bool = False,
) -> None:
    """Print the full pipeline status."""
    completed = sum(1 for j in jobs if j.state == "COMPLETED")
    failed = sum(1 for j in jobs if j.state == "FAILED")
    total = len(jobs)

    is_pipeline = CHAIN_PID_FILE.exists() or (
        CHAIN_FILE.exists() and CHAIN_FILE.stat().st_size > 0
    )

    done_badge = f"{_GREEN}{completed}{_RST}/{total} done"
    fail_badge = f"  {_RED}{failed} failed{_RST}" if failed else ""

    os.system("clear")
    print(f"{_CYAN}{'═' * 65}{_RST}")
    if is_pipeline:
        print(
            f"  {_BOLD}{_CYAN}Neuro-Symbolic T2G Monitor{_RST} — {done_badge}{fail_badge}"
        )
        print(f"  {_watcher_status()}")
    elif total > 0:
        print(f"  {_BOLD}{_CYAN}T2G Job Monitor{_RST} — {done_badge}{fail_badge}")
        print(f"  {_DIM}Standalone mode (no pipeline){_RST}")
    else:
        print(f"  {_BOLD}{_CYAN}T2G Job Monitor{_RST} — no jobs found")
        print(f"  {_DIM}Waiting for jobs matching train-*/eval-*...{_RST}")
    print(f"  {_DIM}{time.strftime('%Y-%m-%d %H:%M:%S')}{_RST}")
    print(f"{_CYAN}{'═' * 65}{_RST}")

    if show_table:
        print()
        current_model = ""
        for job in jobs:
            if job.tag != current_model:
                current_model = job.tag
                print(f"  {_BOLD}{_YELLOW}▸ {current_model}{_RST}")
            print(_format_status(job))
        print()
        print(f"{_DIM}{'─' * 65}{_RST}")

    remaining = sum(1 for j in jobs if j.state == "PENDING")
    running = [j for j in jobs if j.state in ("RUNNING", "STARTING")]
    print()
    if running:
        j = running[0]
        tc = _TYPE_COLORS.get(j.job_type, "")
        bar_color = _CYAN if j.job_type == "train" else _MAGENTA

        if j.job_type == "train" and j.step > 0:
            tot = j.stage_total if j.stage_total > 0 else "?"
            desc = f"step {_WHITE}{j.step}{_RST}/{tot}"
        elif j.job_type == "eval":
            if j.eval_label:
                desc = j.eval_label
            else:
                desc = ""
            if j.step > 0 and j.eval_step_total > 0:
                desc += f", batch {_WHITE}{j.step}{_RST}/{j.eval_step_total}"
        else:
            desc = ""

        job_label = (
            f"{tc}{j.job_type}{_RST}-{_BOLD}{j.tag}{_RST}"
            if j.tag
            else f"{tc}{j.job_type}{_RST}"
        )
        print(
            f"  {_CYAN}▶ Active:{_RST} {job_label}"
            + (f" {_DIM}[{j.slurm_id}]{_RST}" if j.slurm_id else "")
            + (f" — {desc}" if desc else "")
        )

        cur = j.step
        tot = j.stage_total if j.job_type == "train" else j.eval_step_total
        if cur > 0 and tot > 0:
            pct = int(cur / tot * 100)
            bar_w = 20
            filled = int(bar_w * pct / 100)
            bar = f"{bar_color}{'█' * filled}{_GRAY}{'░' * (bar_w - filled)}{_RST}"
            eta = _estimate_eta(j)
            total_eta = _estimate_total_eta(j)
            time_parts = ""
            if j.elapsed:
                time_parts += f" ⏰ {_DIM}{j.elapsed}{_RST}"
            if eta:
                time_parts += f" ⏳ {_DIM}~{eta}{_RST}"
            if total_eta and total_eta != eta:
                time_parts += f" {_DIM}(job ~{total_eta}){_RST}"
            print(f"  {bar} {_WHITE}{pct}%{_RST}{time_parts}")
    elif remaining > 0:
        watcher_alive = False
        if CHAIN_PID_FILE.exists():
            try:
                pid = CHAIN_PID_FILE.read_text().strip()
                result = _run(f"ps -p {pid} -o pid= 2>/dev/null")
                watcher_alive = bool(result)
            except Exception:
                pass
        if watcher_alive:
            print(
                f"  {_YELLOW}⏳ Waiting for next job... ({remaining} remaining){_RST}"
            )
        else:
            print(
                f"  {_RED}⚠ Pipeline stalled{_RST} — {remaining} jobs pending but watcher is dead"
            )
            print(
                f"  {_DIM}Restart: bash cluster/run_all.sh   |   Clean: rm -rf .chain_state{_RST}"
            )
    elif not jobs:
        print(f"  {_DIM}No jobs found.{_RST}")
    else:
        print(f"  {_GREEN}{_BOLD}✓ Pipeline finished!{_RST}")

    if show_metrics:
        metrics_data: dict[str, dict[str, Any]] = {}
        tag_order: list[str] = []

        for j in jobs:
            if not j.tag:
                continue
            if j.tag not in metrics_data:
                metrics_data[j.tag] = {"train_rw": "", "eval_stages": {}}
                tag_order.append(j.tag)
            if j.job_type == "train" and j.last_reward:
                metrics_data[j.tag]["train_rw"] = j.last_reward
            if j.job_type == "eval" and j.eval_stages:
                metrics_data[j.tag]["eval_stages"] = j.eval_stages

        cache = _load_cache()
        for key in cache.get("pipeline_jobs", []):
            parts = key.split("-", 1)
            if len(parts) != 2:
                continue
            tag = parts[1]
            if tag not in metrics_data:
                metrics_data[tag] = {"train_rw": "", "eval_stages": {}}
                tag_order.append(tag)
            entry = cache["jobs"].get(key, {})
            if (
                parts[0] == "train"
                and entry.get("last_reward")
                and not metrics_data[tag]["train_rw"]
            ):
                metrics_data[tag]["train_rw"] = entry["last_reward"]
            if (
                parts[0] == "eval"
                and entry.get("eval_stages")
                and not metrics_data[tag]["eval_stages"]
            ):
                metrics_data[tag]["eval_stages"] = entry["eval_stages"]

        all_stage_keys: list[str] = []
        stage_set: set[str] = set()
        for tag in tag_order:
            for sk in metrics_data[tag]["eval_stages"]:
                if sk not in stage_set:
                    stage_set.add(sk)
                    all_stage_keys.append(sk)
        all_stage_keys.sort(
            key=lambda k: (
                -1
                if k == "baseline"
                else (
                    int(k.split("_")[1]) if k.startswith("stage_") and "_" in k else 99
                )
            )
        )

        def _col_label(k: str) -> str:
            if k == "baseline":
                return "Baseline"
            if k.startswith("stage_"):
                return f"Stage {k.split('_')[1]}"
            return k

        rows: list[tuple[str, str, dict[str, str]]] = []
        for tag in tag_order:
            md = metrics_data[tag]
            train_rw = md["train_rw"]
            stages = md["eval_stages"]
            if train_rw or stages:
                rows.append((tag, train_rw, stages))

        if rows:
            col_w = 10
            stage_hdr = "".join(f"{_col_label(k):<{col_w}s}" for k in all_stage_keys)
            print()
            print(f"  {_BOLD}{'Model':<24s} {'Reward':<10s} {stage_hdr}{_RST}")
            print(f"  {'─' * (24 + 10 + col_w * len(all_stage_keys))}")
            for tag, rw, stages in rows:
                if rw:
                    try:
                        rw_fmt = f"{float(rw):.4f}"
                    except ValueError:
                        rw_fmt = rw
                    rw_str = f"{_CYAN}{rw_fmt:<10s}{_RST}"
                else:
                    rw_str = f"{_DIM}{'-':<10s}{_RST}"
                stage_strs = []
                for sk in all_stage_keys:
                    val = stages.get(sk, "")
                    if val:
                        try:
                            val = f"{float(val):.4f}"
                        except ValueError:
                            pass
                        stage_strs.append(f"{_GREEN}{val:<{col_w}s}{_RST}")
                    else:
                        stage_strs.append(f"{_DIM}{'-':<{col_w}s}{_RST}")
                stage_cells = "".join(stage_strs)
                print(f"  {tag:<24s} {rw_str} {stage_cells}")

    if show_samples and running and running[0].completion_samples:
        print()
        for sl in running[0].completion_samples:
            print(f"  {sl}")

    failed_with_errors = [j for j in jobs if j.state == "FAILED" and j.error_type]
    if failed_with_errors:
        print()
        print(f"  {_RED}{_BOLD}⚠ ERRORS ({len(failed_with_errors)}){_RST}")
        print(f"  {_RED}{'─' * 60}{_RST}")
        for j in failed_with_errors:
            job_label = f"{j.job_type}-{_BOLD}{j.tag}{_RST}" if j.tag else j.job_type
            slurm_id = f" {_DIM}[{j.slurm_id}]{_RST}" if j.slurm_id else ""
            print(
                f"  {_RED}✗{_RST} {job_label}{slurm_id}  {_RED}{_BOLD}{j.error_type}{_RST}"
            )
            if j.error_snippet:
                parts = j.error_snippet.split(" | ")
                shown = 0
                for part in parts:
                    part = part.strip()
                    if not part:
                        continue
                    if len(part) > 100:
                        part = part[:100] + "..."
                    print(f"    {_DIM}{part}{_RST}")
                    shown += 1
                    if shown >= 2:
                        break
        print(f"  {_DIM}Dettagli: cat .chain_errors | python3 -m json.tool{_RST}")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Live monitor for the T2G job chain")
    parser.add_argument(
        "--poll", type=int, default=15, help="Seconds between refresh (default: 15)"
    )
    parser.add_argument("--tab", action="store_true", help="Show the full job table")
    parser.add_argument(
        "--samples",
        nargs="?",
        const=0,
        type=int,
        default=None,
        help="Show completion samples. Optional: max output lines",
    )
    parser.add_argument("--metrics", action="store_true", help="Show metrics table")
    parser.add_argument(
        "--all",
        nargs="?",
        const=0,
        type=int,
        default=None,
        dest="all_mode",
        help="Show everything: table + metrics + samples",
    )
    args = parser.parse_args()

    if args.all_mode is not None:
        args.tab = True
        args.metrics = True
        if args.samples is None:
            args.samples = args.all_mode

    show_samples = args.samples is not None
    max_sample_lines = args.samples if args.samples else 0

    global _SAMPLE_MAX_LINES
    _SAMPLE_MAX_LINES = max_sample_lines

    print("T2G Monitor — Ctrl+C to exit")
    print(f"Polling every {args.poll}s...")
    print()

    try:
        while True:
            jobs = _build_pipeline()
            _display(
                jobs,
                show_table=args.tab,
                show_samples=show_samples,
                show_metrics=args.metrics,
            )

            is_pipeline = CHAIN_PID_FILE.exists() or (
                CHAIN_FILE.exists() and CHAIN_FILE.stat().st_size > 0
            )
            all_done = jobs and all(j.state in ("COMPLETED", "FAILED") for j in jobs)
            watcher_alive = CHAIN_PID_FILE.exists()
            no_pending = not CHAIN_FILE.exists() or not _read_pending_chain()

            if is_pipeline and all_done and not watcher_alive and no_pending:
                print("Pipeline complete. Exiting.")
                break
            elif not is_pipeline and all_done:
                print("Job complete. Exiting.")
                break

            time.sleep(args.poll)
    except KeyboardInterrupt:
        print("\n💀 Monitor stopped.")


if __name__ == "__main__":
    main()
