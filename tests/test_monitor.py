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
  8. Eval "Evaluating" tqdm bar (C1), label-less Pass@1 (C2),
     seeded-sample progress line (C3), last-marker SFT phase detection (C4),
     SFT eval_loss KV + best (C5), eval_*.json loading (C6),
     long-prompt truncation (C7)
  9. live_training_table KV parsing + sample-block skipping (L1/L2)
"""

from __future__ import annotations

import re


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
        "  REWARDS: edit_validity_reward=+0.80",
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
    assert "REWARDS" in text or "edit_validity_reward" in text, "Contains REWARDS"
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


# ---------------------------------------------------------------------------
# C1–C7: audit fixes for chain_monitor
# ---------------------------------------------------------------------------


def test_eval_generating_bar_c1(tmp_path):
    """C1: the 'Evaluating' tqdm bar is recognized (was: only 'Generating')."""
    from src.utils import chain_monitor as cm

    log = tmp_path / "eval.log"
    log.write_text(
        "some line\n"
        "Evaluating:  45%|████▍| 17/38 [00:30<00:40,  0.75s/it]\n"
        "Evaluation complete\n",
        encoding="utf-8",
    )
    job = cm.JobInfo(job_type="eval", config="", tag="qwen05")
    cm._parse_eval_log(log, job)
    assert job.step == 17, f"step from Evaluating bar: {job.step}"
    assert (
        job.eval_step_total == 38
    ), f"total from Evaluating bar: {job.eval_step_total}"
    assert job.tqdm_elapsed == "00:30", f"elapsed: {job.tqdm_elapsed}"
    assert job.tqdm_eta == "00:40", f"eta: {job.tqdm_eta}"


def test_eval_pass_no_label_c2(tmp_path):
    """C2: label-less '  Pass@1: 0.1234' populates stage 'latest' + metrics."""
    from src.utils import chain_monitor as cm

    log = tmp_path / "eval.log"
    log.write_text(
        "Evaluating 17/38 samples (seeded sample)\n"
        "  Pass@1: 0.1234\n"
        "  ROUGE-L mean: 0.4321 ± 0.0567\n"
        "  BLEU (sentence mean / corpus): 0.1234 / 0.1111\n"
        "  chrF2 (sentence mean / corpus): 23.45 / 22.22\n"
        "  Gloss F1 (sentence mean / micro): 0.3456 / 0.3333\n"
        "  Validity rate: 0.9000\n"
        "Evaluation complete\n",
        encoding="utf-8",
    )
    job = cm.JobInfo(job_type="eval", config="", tag="qwen05")
    cm._parse_eval_log(log, job)
    assert job.eval_stages.get("latest") == "0.1234", f"latest stage: {job.eval_stages}"
    assert job.eval_metrics.get("pass_at_1") == "0.1234"
    assert job.eval_metrics.get("rouge_l_mean") == "0.4321"
    assert job.eval_metrics.get("bleu_sentence_mean") == "0.1234"
    assert job.eval_metrics.get("bleu_corpus") == "0.1111"
    assert job.eval_metrics.get("chrf_sentence_mean") == "23.45"
    assert job.eval_metrics.get("gloss_f1_micro") == "0.3333"
    assert job.eval_metrics.get("validity_rate") == "0.9000"
    assert job.eval_label == "COMPLETE", f"label: {job.eval_label}"


def test_eval_checkpoint_seeded_sample_c3():
    """C3: 'Evaluating 17/38 samples (seeded sample)' sets the eval label."""
    from src.utils.chain_monitor import _EVAL_PROGRESS_LINE

    line = "Evaluating 17/38 samples (seeded sample)"
    m = _EVAL_PROGRESS_LINE.search(line)
    assert m is not None, "seeded-sample progress line matched"
    assert int(m.group(1)) == 17
    assert int(m.group(2)) == 38


def test_sft_phase_last_marker_c4(tmp_path):
    """C4: the LAST phase marker in the tail decides sft_active.

    Regression: with both 'STEP 1.5: SFT Pre-training' and
    'STEP 7: GRPO Training' in the window, GRPO must win (the old code
    broke on the FIRST match and left sft_active=True during GRPO).
    """
    from src.utils import chain_monitor as cm

    log = tmp_path / "train.log"
    log.write_text(
        "STEP 1.5: SFT Pre-training\n"
        "  [sft] step=50/200 (25.0%)  loss=2.345678  avg=2.5  min=2.1  lr=1.5e-05  epoch=0.5\n"
        "STEP 7: GRPO Training\n"
        "  step=100  loss=0.005  reward=0.350\n",
        encoding="utf-8",
    )
    job = cm.JobInfo(job_type="train", config="", tag="qwen05")
    cm._parse_training_log(log, job)
    assert job.sft_active is False, "GRPO marker after SFT marker -> not SFT"
    assert job.step == 100, f"GRPO step parsed: {job.step}"
    assert job.last_reward == "0.350", f"GRPO reward: {job.last_reward}"

    log2 = tmp_path / "sft_only.log"
    log2.write_text(
        "STEP 1.5: SFT Pre-training\n"
        "  [sft] step=50/200 (25.0%)  loss=2.345678  avg=2.5  min=2.1  lr=1.5e-05  epoch=0.5\n",
        encoding="utf-8",
    )
    job2 = cm.JobInfo(job_type="train", config="", tag="qwen05")
    cm._parse_training_log(log2, job2)
    assert job2.sft_active is True, "SFT marker last -> SFT active"
    assert job2.sft_step == 50, f"sft step: {job2.sft_step}"
    assert job2.sft_loss == "2.345678", f"sft loss: {job2.sft_loss}"


def test_sft_eval_loss_c5(tmp_path):
    """C5: SFT eval_loss KV lines and '[sft] Best eval_loss=' are parsed."""
    from src.utils import chain_monitor as cm

    log = tmp_path / "train.log"
    log.write_text(
        "STEP 1.5: SFT Pre-training\n"
        "  [sft] step=50/200 (25.0%)  loss=2.345678  avg=2.5  min=2.1  lr=1.5e-05  epoch=0.5\n"
        "  step=100  eval_loss=1.23456789  epoch=0.50000000\n"
        "  step=101  eval_loss=1.11111111  epoch=0.51000000\n"
        "[sft] Best eval_loss=0.987654 (best checkpoint=/tmp/out/checkpoint-500)\n",
        encoding="utf-8",
    )
    job = cm.JobInfo(job_type="train", config="", tag="qwen05")
    cm._parse_training_log(log, job)
    assert job.sft_eval_loss == "1.11111111", f"last eval_loss: {job.sft_eval_loss}"
    assert (
        job.sft_eval_loss_best == "0.987654"
    ), f"best eval_loss: {job.sft_eval_loss_best}"


def test_eval_results_json_c6(tmp_path):
    """C6: after completion, metrics are loaded from the eval_*.json."""
    import json

    from src.utils import chain_monitor as cm

    json_path = tmp_path / "eval_final.json"
    json_path.write_text(
        json.dumps(
            {
                "rouge_l_mean": 0.4321,
                "pass_at_1": 0.1234,
                "bleu_sentence_mean": 0.2222,
                "chrf_sentence_mean": 20.5,
                "gloss_f1_micro": 0.3333,
                "validity_rate": 0.9,
                "exact_match": 0.05,
            }
        ),
        encoding="utf-8",
    )
    log = tmp_path / "eval.log"
    log.write_text(
        "Evaluating 17/38 samples (seeded sample)\n"
        "  Pass@1: 0.1234\n"
        "Evaluation complete\n"
        f"Results saved to {json_path}\n",
        encoding="utf-8",
    )
    job = cm.JobInfo(job_type="eval", config="", tag="qwen05")
    cm._parse_eval_log(log, job)
    assert job.eval_metrics.get("rouge_l_mean") == "0.4321", job.eval_metrics
    assert job.eval_metrics.get("bleu_sentence_mean") == "0.2222"
    assert job.eval_metrics.get("exact_match") == "0.0500"
    assert job.eval_stages.get("latest") == "0.1234", job.eval_stages


def test_completion_prompt_truncation_c7():
    """C7: long few-shot prompts are truncated to ~200 chars in the panel."""
    from src.utils.chain_monitor import _extract_completion_samples

    long_prompt = "The cat sleeps on the sofa. " * 20  # ~480 chars
    log_lines = [
        "================================================================",
        "  COMPLETION SAMPLES",
        "================================================================",
        "-------------------------------------------------------------------",
        "  Sample 1",
        "-------------------------------------------------------------------",
        f"  PROMPT: {long_prompt}",
        "  OUTPUT:",
        "    IX MAN WALK HOUSE",
        "  GOLD:",
        "    IX MAN WALK ENTER HOUSE",
        "  REWARDS: edit_validity_reward=+0.80",
        "  TOTAL:   +0.80",
        "================================================================",
    ]
    samples = _extract_completion_samples(log_lines)
    prompt_line = [line for line in samples if "PROMPT:" in line]
    assert prompt_line, "prompt shown in panel"
    visible = re.sub(r"\033\[[0-9;]*m", "", prompt_line[0])
    assert visible.endswith("..."), "truncation marker present"
    # 2 leading spaces + "PROMPT:" + 2 spaces + <=200 chars + "..."
    assert len(visible) <= 214, f"prompt truncated to ~200: {len(visible)}"


# ---------------------------------------------------------------------------
# live_training_table (L1/L2)
# ---------------------------------------------------------------------------


def test_live_table_kv_parser():
    """L1: live_training_table parses HighPrecisionLogCallback KV lines."""
    from src.utils.live_training_table import _parse_kv_line

    entry = _parse_kv_line(
        "step=5  loss=1.23456789  reward=0.50258335  completions/mean_length=10"
        "  learning_rate=0.00000100  kl=0.12345678"
    )
    assert entry is not None
    assert entry["step"] == 5, f"step: {entry['step']}"
    assert abs(float(entry["loss"]) - 1.23456789) < 1e-8
    assert abs(float(entry["reward"]) - 0.50258335) < 1e-8
    assert float(entry["completions/mean_length"]) == 10.0
    assert abs(float(entry["learning_rate"]) - 1e-6) < 1e-12
    assert abs(float(entry["kl"]) - 0.12345678) < 1e-8

    # Lines that don't start with step= are not metric lines
    assert _parse_kv_line("some log line step=5 loss=1.0") is None
    assert _parse_kv_line("") is None


def test_live_table_skips_sample_blocks(monkeypatch, capsys):
    """L2: SFT SAMPLE PREDICTIONS and COMPLETION SAMPLES blocks are skipped."""
    import io
    import sys

    from src.utils import live_training_table as ltt

    sep = "═" * 70
    stream = io.StringIO(
        f"{sep}\n"
        "  SFT SAMPLE PREDICTIONS (step 100)\n"
        f"{sep}\n"
        "  PROMPT: foo\n"
        "  GOLD:   BAR\n"
        "  PRED:   BAZ\n"
        f"{sep}\n"
        "  step=5  loss=1.23456789  reward=0.50258335\n"
        f"{sep}\n"
        "  COMPLETION SAMPLES\n"
        f"{sep}\n"
        "  Sample 1\n"
        "  OUTPUT:\n"
        "    IX MAN WALK\n"
        f"{sep}\n"
        "  step=6  loss=1.11111111  reward=0.60258335\n"
    )
    monkeypatch.setattr(sys, "stdin", stream)
    monkeypatch.setattr(
        sys, "argv", ["live_training_table", "--cols", "step,loss,reward"]
    )
    monkeypatch.setattr(ltt.os, "system", lambda *a, **k: None)
    ltt.main()
    out = capsys.readouterr().out
    assert "SFT SAMPLE PREDICTIONS" not in out, "SFT block skipped"
    assert "COMPLETION SAMPLES" not in out, "COMPLETION block skipped"
    assert "1.2346" in out, "first KV row rendered"
    assert "1.1111" in out, "second KV row rendered"


# ---------------------------------------------------------------------------
# HighPrecisionLogCallback: live-status fields (reward_avg, None filtering)
# ---------------------------------------------------------------------------


class _FakeState:
    def __init__(self, step: int = 5):
        self.global_step = step
        self.is_local_process_zero = True


def test_high_precision_callback_reward_running_avg(monkeypatch, capsys):
    """on_log accumulates the running mean of logged GRPO rewards; partial
    logs (routine eval, no reward) never reset or overwrite it."""
    from src.training import callbacks as cb

    captured: list[dict] = []
    monkeypatch.setattr(cb, "live_status_set", lambda **kw: captured.append(kw))

    c = cb.HighPrecisionLogCallback()
    c.on_log(
        None,
        _FakeState(5),
        None,
        logs={"loss": 1.0, "reward": 0.5, "learning_rate": 1e-6},
    )
    c.on_log(
        None,
        _FakeState(10),
        None,
        logs={"loss": 0.9, "reward": 0.7, "learning_rate": 1e-6},
    )
    # Routine-eval log: NO reward/loss/lr — must not touch reward_avg/loss/lr
    c.on_log(None, _FakeState(10), None, logs={"eval_loss": 0.3})

    with_reward = [k for k in captured if "reward_avg" in k]
    assert with_reward[-1]["reward_avg"] == 0.6, "mean(0.5, 0.7) = 0.6"
    last = captured[-1]
    assert "reward_avg" not in last, "partial log must not touch reward_avg"
    assert "loss" not in last and "lr" not in last, "partial log must not reset loss/lr"
    assert last["eval_loss"] == 0.3 and last["step"] == 10

    # Third reward event continues the SAME running average
    c.on_log(None, _FakeState(15), None, logs={"loss": 0.8, "reward": 0.9})
    with_reward = [k for k in captured if "reward_avg" in k]
    assert with_reward[-1]["reward_avg"] == 0.7, "mean(0.5, 0.7, 0.9) = 0.7"


def test_high_precision_callback_sft_logs_have_no_reward_avg(monkeypatch, capsys):
    """SFT logs (no 'reward' key) never produce reward_avg — the avg stays
    absent so the TUI shows nothing extra during SFT."""
    from src.training import callbacks as cb

    captured: list[dict] = []
    monkeypatch.setattr(cb, "live_status_set", lambda **kw: captured.append(kw))

    c = cb.HighPrecisionLogCallback()
    for step, loss in ((10, 3.4), (20, 3.0)):
        c.on_log(
            None,
            _FakeState(step),
            None,
            logs={"loss": loss, "learning_rate": 2e-5, "epoch": 0.01},
        )

    assert all("reward_avg" not in k for k in captured)
    assert all("reward" not in k for k in captured)
    assert captured[-1]["loss"] == 3.0
