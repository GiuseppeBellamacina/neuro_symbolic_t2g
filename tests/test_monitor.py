#!/usr/bin/env python3
"""Test monitor parsing logic (chain_monitor).

Validates:
  1. Completion sample extraction from log lines
  2. Training log parsing (step, reward, tqdm progress)
  3. SFT log parsing
  4. Time helpers (parse elapsed, format duration, ETA estimation)
  5. Eval log parsing (Pass@1)
  6. Total ETA estimation
  7. JobInfo dataclass
"""

from __future__ import annotations


def test_completion_sample_extraction():
    """Completion samples are extracted from log lines correctly."""
    from src.utils.chain_monitor import _extract_completion_samples

    log_lines = [
        "step=100 loss=0.005 reward=0.350 ",
        "some other log line",
        "================================================================",
        "  COMPLETION SAMPLES",
        "================================================================",
        "-------------------------------------------------------------------",
        "  Sample 1  [difficulty=medium] [✗]",
        "-------------------------------------------------------------------",
        "  PROMPT: The man walks into the house.",
        "  OUTPUT:",
        "    IX MAN WALK HOUSE",
        "  GOLD:",
        "    IX MAN WALK ENTER HOUSE",
        "  REWARDS: translation_quality_reward=+0.80  structural_dense_reward=+0.65  gloss_format_reward=+1.00  gloss_repetition_reward=+1.00",
        "  TOTAL:   +0.80",
        "================================================================",
        "step=110 loss=0.004 reward=0.380 ",
    ]
    samples = _extract_completion_samples(log_lines, max_lines=10)
    assert len(samples) > 0, f"Samples extracted: {len(samples)}"

    text = "\n".join(samples)
    assert "Last completion" in text or "COMPLETION" in text, "Contains header"
    assert "PROMPT" in text or "prompt" in text.lower(), "Contains PROMPT"
    assert "OUTPUT" in text or "IX MAN WALK" in text, "Contains OUTPUT"
    assert "GOLD" in text or "ENTER HOUSE" in text, "Contains GOLD"
    assert "REWARDS" in text or "translation_quality_reward" in text, "Contains REWARDS"
    assert "medium" in text.lower() or "difficulty" in text.lower(), "Difficulty badge"
    assert "mismatch" in text.lower(), "Match indicator present"

    empty = _extract_completion_samples(["no samples here"])
    assert len(empty) == 0, "No samples in empty log"


def test_training_log_parsing():
    """Training log key=value and tqdm parsing."""
    from src.utils.chain_monitor import _KV_REWARD, _KV_STEP, _TQDM_PROGRESS, JobInfo

    line1 = "  step=420 loss=0.005 reward=0.450 reward_std=0.05 learning_rate=5e-06"
    m = _KV_STEP.search(line1)
    assert m is not None, "KV step matched"
    assert int(m.group(1)) == 420, f"KV step = 420, got {m.group(1)}"

    m2 = _KV_REWARD.search(line1)
    assert m2 is not None, "KV reward matched"
    assert (
        abs(float(m2.group(1)) - 0.45) < 0.01
    ), f"KV reward = 0.450, got {m2.group(1)}"

    line2 = " 47%|?????     | 420/900 [29:23<25:49,  3.92s/it]"
    m3 = _TQDM_PROGRESS.search(line2)
    assert m3 is not None, "TQDM progress matched"
    assert int(m3.group(1)) == 420, "TQDM current = 420"
    assert int(m3.group(2)) == 900, "TQDM total = 900"

    from src.utils.chain_monitor import _DICT_REWARD

    line3 = "{'loss': 0.005, 'grad_norm': 0.1, 'learning_rate': 5e-06, 'reward': 0.5025, 'epoch': 1.0}"
    m4 = _DICT_REWARD.search(line3)
    assert m4 is not None, "Dict reward matched"
    assert (
        abs(float(m4.group(1)) - 0.5025) < 0.01
    ), f"Dict reward = 0.5025, got {m4.group(1)}"

    job = JobInfo(
        job_type="train",
        config="",
        tag="qwen05",
        slurm_id="12345",
        state="RUNNING",
        step=100,
        stage_total=1500,
    )
    assert job.label == "train-qwen05", f"JobInfo label: {job.label}"
    assert job.step == 100
    assert job.stage_total == 1500


def test_sft_log_parsing():
    """SFT progress regex and sample extraction."""
    from src.utils.chain_monitor import _SFT_PROGRESS, _extract_sft_samples

    line = "  [sft] step=50/200 (25.0%)  loss=2.345678  avg=2.5  min=2.1  lr=1.5e-05  epoch=0.5"
    m = _SFT_PROGRESS.search(line)
    assert m is not None, "SFT progress matched"
    assert int(m.group(1)) == 50, f"SFT step = 50, got {m.group(1)}"
    assert int(m.group(2)) == 200, f"SFT total = 200, got {m.group(2)}"
    assert (
        abs(float(m.group(3)) - 2.345678) < 0.001
    ), f"SFT loss = 2.345678, got {m.group(3)}"

    sft_log_lines = [
        "some log line",
        "======================================================================",
        "  SFT SAMPLE PREDICTIONS (step 100)",
        "======================================================================",
        "-------------------------------------------------------------------",
        "  PROMPT: The man walks into the house.",
        "  GOLD:   IX MAN WALK ENTER HOUSE",
        "  PRED:   IX MAN WALK HOUSE",
        "-------------------------------------------------------------------",
        "  PROMPT: The woman reads a book.",
        "  GOLD:   IX WOMAN READ BOOK",
        "  PRED:   IX WOMAN READ BOOK",
        "======================================================================",
    ]
    samples = _extract_sft_samples(sft_log_lines)
    assert len(samples) == 2, f"SFT samples extracted: {len(samples)}"
    text = "\n".join(samples)
    assert "GOLD" in text, "SFT sample has GOLD"
    assert "PRED" in text, "SFT sample has PRED"
    assert "ENTER HOUSE" in text, "SFT sample has correct gold"

    empty = _extract_sft_samples(["no sft samples here"])
    assert len(empty) == 0, "No SFT samples in empty log"


def test_time_helpers():
    """Time parsing, formatting, and ETA estimation."""
    from src.utils.chain_monitor import (
        JobInfo,
        _estimate_eta,
        _format_duration,
        _parse_elapsed_seconds,
    )

    assert (
        _parse_elapsed_seconds("12:34") == 754
    ), f"Parse '12:34' = 754s, got {_parse_elapsed_seconds('12:34')}"
    assert _parse_elapsed_seconds("1:23:45") == 5025
    assert _parse_elapsed_seconds("1-02:03:04") == 93784
    assert _parse_elapsed_seconds("") is None

    assert _format_duration(60) == "1m00s", f"Format 60s: {_format_duration(60)}"
    assert _format_duration(3661) == "1h01m"
    assert _format_duration(5) == "5s"

    job1 = JobInfo(job_type="train", config="", tag="qwen05", tqdm_eta="25:49")
    assert _estimate_eta(job1) == "25:49", "ETA from tqdm"

    job2 = JobInfo(
        job_type="train",
        config="",
        tag="qwen05",
        step=400,
        stage_total=1500,
        elapsed="1:00:00",
    )
    eta2 = _estimate_eta(job2)
    assert len(eta2) > 0, "ETA from elapsed non-empty"
    assert "h" in eta2, f"ETA from elapsed > 1h: {eta2}"


def test_eval_log_parsing():
    """Eval log Pass@1 and checkpoint parsing."""
    from src.utils.chain_monitor import _EVAL_CHECKPOINT, _EVAL_COMPLETE, _EVAL_PASS

    line = "  qwen05                    Pass@1:   0.8523"
    m = _EVAL_PASS.search(line)
    assert m is not None, "Eval pass matched"
    assert "qwen05" in m.group(1), f"Eval model: {m.group(1)}"
    assert (
        abs(float(m.group(2)) - 0.8523) < 0.01
    ), f"Eval pass = 0.8523, got {m.group(2)}"

    line2 = "Evaluating: baseline"
    m2 = _EVAL_CHECKPOINT.search(line2)
    assert m2 is not None, "Eval checkpoint matched"
    assert m2.group(1) == "baseline", f"Eval label: {m2.group(1)}"

    line3 = "Evaluation complete"
    m3 = _EVAL_COMPLETE.search(line3)
    assert m3 is not None, "Eval complete matched"


def test_estimate_total_eta():
    """Total ETA estimation for different job states."""
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
    assert eta is None or len(eta) > 0, "Total ETA for train is non-empty or None"

    job2 = JobInfo(
        job_type="train",
        config="",
        tag="qwen05",
        step=1500,
        stage_total=1500,
        tqdm_elapsed="1:00:00",
    )
    eta2 = _estimate_total_eta(job2)
    assert eta2 == "", "Total ETA when complete = empty"

    job3 = JobInfo(job_type="train", config="", tag="qwen05", step=0, stage_total=1500)
    eta3 = _estimate_total_eta(job3)
    assert eta3 == "", "Total ETA without elapsed = empty"
