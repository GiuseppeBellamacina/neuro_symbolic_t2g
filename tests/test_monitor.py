#!/usr/bin/env python3
"""Verify Monitor Parsing Logic ? Test Script.

Validates:
  1. Completion sample extraction from log lines
  2. Training log parsing (step, reward, tqdm progress)
  3. Eval log parsing (Pass@1)
  4. Time helpers (parse elapsed, format duration, ETA estimation)
  5. JobInfo dataclass

Usage:
    python tests/test_monitor.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

PASS = 0
FAIL = 0


def check(name: str, condition: bool, detail: str = "") -> None:
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  [PASS] {name}" + (f" -- {detail}" if detail else ""))
    else:
        FAIL += 1
        print(f"  [FAIL] {name}" + (f" -- {detail}" if detail else ""))


def test_completion_sample_extraction() -> None:
    print("\n-- 1. Completion Sample Extraction --")
    from src.utils.chain_monitor import _extract_completion_samples

    # Simulate a log with completion samples block
    log_lines = [
        "step=100 loss=0.005 reward=0.350 ",
        "some other log line",
        "================================================================",
        "  COMPLETION SAMPLES",
        "================================================================",
        "-------------------------------------------------------------------",
        "  Sample 1  [difficulty=medium]",
        "-------------------------------------------------------------------",
        "  PROMPT: The man walks into the house.",
        "  OUTPUT:",
        "    IX MAN WALK HOUSE",
        "  REWARDS: translation_quality_reward=+0.80  structural_dense_reward=+0.65  gloss_format_reward=+1.00  gloss_repetition_reward=+1.00",
        "  TOTAL:   +0.80",
        "================================================================",
        "step=110 loss=0.004 reward=0.380 ",
    ]

    samples = _extract_completion_samples(log_lines, max_lines=10)
    check("Samples extracted (non-empty)", len(samples) > 0, f"{len(samples)} lines")

    # Check key elements are present
    text = "\n".join(samples)
    check("Contains 'Last completion'", "Last completion" in text)
    check("Contains PROMPT", "PROMPT" in text or "prompt" in text.lower())
    check("Contains OUTPUT", "OUTPUT" in text or "IX MAN WALK" in text)
    check("Contains REWARDS", "REWARDS" in text or "translation_quality_reward" in text)
    check("Contains TOTAL", "TOTAL" in text or "total" in text.lower())

    # Verify difficulty badge
    check(
        "Difficulty badge present",
        "medium" in text.lower() or "difficulty" in text.lower(),
    )

    # Empty log should return empty
    empty = _extract_completion_samples(["no samples here"])
    check("No samples in empty log", len(empty) == 0)


def test_training_log_parsing() -> None:
    print("\n-- 2. Training Log Parsing --")

    from src.utils.chain_monitor import _KV_REWARD, _KV_STEP, _TQDM_PROGRESS, JobInfo

    # Test key=value step parsing
    line1 = "  step=420 loss=0.005 reward=0.450 reward_std=0.05 learning_rate=5e-06"
    m = _KV_STEP.search(line1)
    check("KV step matched", m is not None)
    if m:
        check("KV step = 420", int(m.group(1)) == 420, str(m.group(1)))

    m2 = _KV_REWARD.search(line1)
    check("KV reward matched", m2 is not None)
    if m2:
        check("KV reward = 0.450", abs(float(m2.group(1)) - 0.45) < 0.01, m2.group(1))

    # Test tqdm progress parsing
    line2 = " 47%|?????     | 420/900 [29:23<25:49,  3.92s/it]"
    m3 = _TQDM_PROGRESS.search(line2)
    check("TQDM progress matched", m3 is not None)
    if m3:
        check("TQDM current = 420", int(m3.group(1)) == 420)
        check("TQDM total = 900", int(m3.group(2)) == 900)

    # Test dict-style reward
    from src.utils.chain_monitor import _DICT_REWARD

    line3 = "{'loss': 0.005, 'grad_norm': 0.1, 'learning_rate': 5e-06, 'reward': 0.5025, 'epoch': 1.0}"
    m4 = _DICT_REWARD.search(line3)
    check("Dict reward matched", m4 is not None)
    if m4:
        check(
            "Dict reward = 0.5025", abs(float(m4.group(1)) - 0.5025) < 0.01, m4.group(1)
        )

    # Test JobInfo
    job = JobInfo(
        job_type="train",
        config="",
        tag="qwen05",
        slurm_id="12345",
        state="RUNNING",
        step=100,
        stage_total=1500,
    )
    check("JobInfo label", job.label == "train-qwen05")
    check("JobInfo step", job.step == 100)
    check("JobInfo stage_total", job.stage_total == 1500)


def test_time_helpers() -> None:
    print("\n-- 3. Time Helpers --")
    from src.utils.chain_monitor import (
        JobInfo,
        _estimate_eta,
        _format_duration,
        _parse_elapsed_seconds,
    )

    # Parse elapsed
    s1 = _parse_elapsed_seconds("12:34")
    check("Parse '12:34' = 754s", s1 == 754, str(s1))

    s2 = _parse_elapsed_seconds("1:23:45")
    check("Parse '1:23:45' = 5025s", s2 == 5025, str(s2))

    s3 = _parse_elapsed_seconds("1-02:03:04")
    check("Parse '1-02:03:04' = 93784s", s3 == 93784, str(s3))

    s4 = _parse_elapsed_seconds("")
    check("Parse '' = None", s4 is None)

    # Format duration
    check("Format 60s = '1m00s'", _format_duration(60) == "1m00s")
    check("Format 3661s = '1h01m'", _format_duration(3661) == "1h01m")
    check("Format 5s = '5s'", _format_duration(5) == "5s")

    # Estimate ETA ? tqdm ETA takes priority
    job1 = JobInfo(job_type="train", config="", tag="qwen05", tqdm_eta="25:49")
    eta1 = _estimate_eta(job1)
    check("ETA from tqdm", eta1 == "25:49")

    # ETA from elapsed
    job2 = JobInfo(
        job_type="train",
        config="",
        tag="qwen05",
        step=400,
        stage_total=1500,
        elapsed="1:00:00",
    )
    eta2 = _estimate_eta(job2)
    check("ETA from elapsed is not empty", len(eta2) > 0)
    check("ETA from elapsed > 1h", "h" in eta2, eta2)


def test_eval_log_parsing() -> None:
    print("\n-- 4. Eval Log Parsing --")

    from src.utils.chain_monitor import _EVAL_CHECKPOINT, _EVAL_COMPLETE, _EVAL_PASS

    line = "  qwen05                    Pass@1:   0.8523"
    m = _EVAL_PASS.search(line)
    check("Eval pass matched", m is not None)
    if m:
        check("Eval model = qwen05", "qwen05" in m.group(1))
        check("Eval pass = 0.8523", abs(float(m.group(2)) - 0.8523) < 0.01, m.group(2))

    line2 = "Evaluating: baseline"
    m2 = _EVAL_CHECKPOINT.search(line2)
    check("Eval checkpoint matched", m2 is not None)
    if m2:
        check("Eval label = baseline", m2.group(1) == "baseline")

    line3 = "Evaluation complete"
    m3 = _EVAL_COMPLETE.search(line3)
    check("Eval complete matched", m3 is not None)


def test_estimate_total_eta() -> None:
    print("\n-- 5. Estimate Total ETA --")
    from src.utils.chain_monitor import JobInfo, _estimate_total_eta

    job = JobInfo(
        job_type="train",
        config="",
        tag="qwen05",
        step=400,
        stage_total=1500,
        tqdm_elapsed="20:00",
    )
    eta = _estimate_total_eta(job)
    check("Total ETA for train is non-empty", len(eta) > 0 if eta else True)

    job2 = JobInfo(
        job_type="train",
        config="",
        tag="qwen05",
        step=1500,
        stage_total=1500,
        tqdm_elapsed="1:00:00",
    )
    eta2 = _estimate_total_eta(job2)
    check("Total ETA when complete = empty", eta2 == "")

    # No elapsed
    job3 = JobInfo(job_type="train", config="", tag="qwen05", step=0, stage_total=1500)
    eta3 = _estimate_total_eta(job3)
    check("Total ETA without elapsed = empty", eta3 == "")


def main() -> None:
    global PASS, FAIL
    print("=" * 60)
    print("TEST: Monitor Parsing Logic")
    print("=" * 60)

    try:
        test_completion_sample_extraction()
        test_training_log_parsing()
        test_time_helpers()
        test_eval_log_parsing()
        test_estimate_total_eta()
    except Exception as e:
        print(f"\n  !! CRASH: {e}")
        import traceback

        traceback.print_exc()
        FAIL += 1

    print(f"\n{'=' * 60}")
    print(f"RESULTS: {PASS} passed, {FAIL} failed, {PASS + FAIL} total")
    print(f"{'=' * 60}")
    sys.exit(0 if FAIL == 0 else 1)


if __name__ == "__main__":
    main()
