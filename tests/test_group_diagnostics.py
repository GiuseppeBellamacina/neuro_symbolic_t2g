"""Preregistered instrumentation: group diagnostics over GRPO G-completions.

Covers:
- ``compute_group_diagnostics`` exact values (mean unique normalized outputs
  per group of G completions, mean absolute word-length error vs gold)
- canonical normalization: extract_gloss_text → casefold → whitespace
  collapse (think tags / fences / casing / spacing must not create
  phantom uniqueness or phantom length errors)
- loud failure on malformed / non-multiple-of-G input (no wrong groups)
- CompletionSampleLogger wiring: full-batch diagnostics, strict inference
  from consecutive identical prompts, explicit-skip with logged reason,
  and ZERO disruption of the existing sample buffer / interceptor
- CompletionSampleCallback: stable ``groups/*`` keys logged to wandb
- TRL-native metrics (entropy/KL/clip/reward_std/frac_reward_zero_std)
  pass through HighPrecisionLogCallback untouched, and None filtering in
  the live-status publication remains intact
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.training.callbacks import (  # noqa: E402
    CompletionSampleCallback,
    CompletionSampleLogger,
    HighPrecisionLogCallback,
    _infer_group_size,
)
from src.utils.metrics import (  # noqa: E402
    abs_length_error,
    compute_group_diagnostics,
    gloss_word_count,
    mean_abs_length_error,
    normalize_gloss_diagnostics_text,
)

# ---------------------------------------------------------------------------
# Pure diagnostics — exact values
# ---------------------------------------------------------------------------


class TestGroupDiagnosticsValues:
    def test_exact_values_two_groups(self):
        """G=2, two prompts: unique counts 1 and 2 → mean 1.5; length
        errors [0, 0, 1, 0] → mean 0.25."""
        completions = [
            "IX MAN WALK",  # group 1
            "  ix  man  walk ",  # group 1 (same after normalization)
            "DOG RUN",  # group 2
            "CAT RUN",  # group 2
        ]
        references = [
            "IX MAN WALK",  # 3 words vs 3 → 0
            "IX MAN WALK",  # 3 vs 3 → 0
            "DOG RUN PARK",  # 2 vs 3 → 1
            "CAT RUN",  # 2 vs 2 → 0
        ]
        diag = compute_group_diagnostics(completions, references, group_size=2)
        assert diag["unique_outputs_mean"] == pytest.approx(1.5)
        assert diag["abs_length_error_mean"] == pytest.approx(0.25)

    def test_exact_values_single_group_all_identical(self):
        """One prompt, G=4, identical completions → 1 unique; gold equal →
        0 length error."""
        comps = ["BOOK READ", "BOOK READ", "BOOK READ", "BOOK READ"]
        diag = compute_group_diagnostics(comps, ["BOOK READ"] * 4, group_size=4)
        assert diag == {
            "unique_outputs_mean": 1.0,
            "abs_length_error_mean": 0.0,
        }

    def test_normalization_think_tags_fences_casefold_whitespace(self):
        """Think tags, code fences, case and whitespace collapse to the
        same canonical string — they must not inflate uniqueness."""
        a = "<think>reasoning here</think>```gloss\nIX  MAN\n```"
        b = "ix man"
        assert normalize_gloss_diagnostics_text(a) == "ix man"
        assert normalize_gloss_diagnostics_text(a) == normalize_gloss_diagnostics_text(
            b
        )
        diag = compute_group_diagnostics([a, b], None, group_size=2)
        assert diag["unique_outputs_mean"] == 1.0

    def test_think_tag_words_not_counted_in_length(self):
        """Length counts post-extraction tokens: reasoning inside
        <think> must not contribute to the word count."""
        assert gloss_word_count("<think>let me think about it</think>DOG RUN") == 2
        assert abs_length_error(
            "<think>reasoning</think>DOG RUN", "DOG RUN PARK"
        ) == pytest.approx(1.0)

    def test_empty_completion_counts_zero_words(self):
        """An empty/whitespace-only completion is 0 words — the error vs
        gold is the gold length, never a silent skip."""
        assert gloss_word_count("") == 0
        assert gloss_word_count("   ") == 0
        assert abs_length_error("", "IX MAN WALK") == pytest.approx(3.0)

    def test_references_optional_unique_only(self):
        """references=None → only unique_outputs_mean is reported (no
        length error without gold)."""
        diag = compute_group_diagnostics(["A B", "A B", "C", "C D"], None, 2)
        assert diag == {"unique_outputs_mean": 1.5}
        assert "abs_length_error_mean" not in diag

    def test_mean_abs_length_error_exact(self):
        errs = mean_abs_length_error(
            ["IX MAN WALK", "DOG RUN"], ["IX MAN WALK HOUSE", "CAT RUN"]
        )
        # |3-4| = 1 and |2-2| = 0 → mean 0.5
        assert errs == pytest.approx(0.5)


class TestGroupDiagnosticsMalformed:
    """Malformed input must fail LOUDLY — never infer wrong groups."""

    def test_rejects_non_multiple_of_group_size(self):
        with pytest.raises(ValueError, match="not a multiple"):
            compute_group_diagnostics(
                ["A", "A", "A", "B", "C"], ["A"] * 5, group_size=2
            )

    def test_rejects_empty_batch(self):
        with pytest.raises(ValueError, match="empty"):
            compute_group_diagnostics([], [], group_size=2)

    def test_rejects_misaligned_references(self):
        with pytest.raises(ValueError, match="misaligned"):
            compute_group_diagnostics(["A", "B"], ["A"], group_size=1)

    def test_rejects_non_positive_group_size(self):
        with pytest.raises(ValueError, match="group_size"):
            compute_group_diagnostics(["A", "B"], ["A", "B"], group_size=0)
        with pytest.raises(ValueError, match="group_size"):
            compute_group_diagnostics(["A", "B"], ["A", "B"], group_size=-3)

    def test_mean_abs_length_error_rejects_misaligned_and_empty(self):
        with pytest.raises(ValueError, match="misaligned"):
            mean_abs_length_error(["A"], ["A", "B"])
        with pytest.raises(ValueError, match="empty"):
            mean_abs_length_error([], [])

    def test_partial_batch_never_silently_truncated(self):
        """A 5-completion batch with G=2 raises — it must NOT compute over
        the first 4 (2 valid groups) and drop the tail."""
        with pytest.raises(ValueError):
            compute_group_diagnostics(
                ["A", "A", "A", "A", "B"], ["A"] * 5, group_size=2
            )


# ---------------------------------------------------------------------------
# CompletionSampleLogger wiring
# ---------------------------------------------------------------------------


def _dummy_reward(completions, prompts=None, **kwargs):
    return [0.0] * len(completions)


_dummy_reward.__name__ = "dummy_reward"


def _make_logger(n_samples=3, group_size=None):
    return CompletionSampleLogger(
        reward_fns=[_dummy_reward],
        reward_weights=[0.0],
        n_samples=n_samples,
        group_size=group_size,
    )


class TestCaptureGroupWiring:
    def test_diagnostics_over_full_batch_inferred_groups(self):
        """group_size=None → G inferred from consecutive identical
        prompts; diagnostics cover the FULL batch (not just n_samples)."""
        logger = _make_logger(n_samples=2)  # display buffer ≠ batch size
        prompts = ["<|im_start|>user\np1<|im_end|>"] * 2 + [
            "<|im_start|>user\np2<|im_end|>"
        ] * 2
        logger._capture(
            ["IX MAN", "ix man", "DOG RUN", "CAT SLEEP"],
            prompts=prompts,
            gold_gloss=["IX MAN", "IX MAN", "DOG RUN", "DOG SLEEP"],
        )
        diag = logger.last_group_diagnostics
        assert diag is not None
        # group1: {ix man} → 1; group2: {dog run, cat sleep} → 2 → 1.5
        assert diag["unique_outputs_mean"] == pytest.approx(1.5)
        # errors: 0, 0, 0, |2-2|... "CAT SLEEP" vs "DOG SLEEP" → 0 → mean 0
        assert diag["abs_length_error_mean"] == pytest.approx(0.0)

    def test_explicit_group_size_wins_over_inference(self):
        logger = _make_logger(group_size=2)
        # Prompts NOT repeated (would infer nothing): explicit G still used
        prompts = ["u1", "u2", "u3", "u4"]
        logger._capture(["A B", "A B", "C", "C D"], prompts=prompts, gold_gloss=None)
        assert logger.last_group_diagnostics == {"unique_outputs_mean": 1.5}

    def test_malformed_batch_skipped_with_logged_reason(self, caplog):
        """4 completions with explicit G=3 → no diagnostics + a warning
        naming the reason; NEVER 3+1 wrong grouping."""
        import logging

        logger = _make_logger(group_size=3)
        with caplog.at_level(logging.WARNING, logger="src.training.callbacks"):
            logger._capture(["A", "A", "A", "B"], prompts=None, gold_gloss=["A"] * 4)
        assert logger.last_group_diagnostics is None
        assert any("not a multiple" in r.message for r in caplog.records)

    def test_misaligned_gold_skipped_with_logged_reason(self, caplog):
        import logging

        logger = _make_logger(group_size=2)
        with caplog.at_level(logging.WARNING, logger="src.training.callbacks"):
            logger._capture(
                ["A", "A", "B", "B"],
                prompts=None,
                gold_gloss=["A", "A"],  # 2 golds for 4 completions
            )
        assert logger.last_group_diagnostics is None
        assert any("misaligned" in r.message for r in caplog.records)

    def test_no_prompts_and_no_group_size_skips_quietly(self):
        logger = _make_logger()
        logger._capture(["A", "B"], prompts=None, gold_gloss=None)
        assert logger.last_group_diagnostics is None

    def test_infer_group_size_ragged_returns_none(self):
        assert _infer_group_size(["p1", "p1", "p2"]) is None
        assert _infer_group_size(["p1", "p2", "p2", "p1"]) is None
        assert _infer_group_size([]) is None
        assert _infer_group_size(None) is None
        # No repetition at all (G=1) → no meaningful group structure
        assert _infer_group_size(["p1", "p2", "p3", "p4"]) is None
        assert _infer_group_size(["p1", "p1", "p2", "p2"]) == 2

    def test_diagnostics_reset_between_batches(self):
        logger = _make_logger(group_size=2)
        logger._capture(["A", "A"], prompts=None, gold_gloss=None)
        assert logger.last_group_diagnostics is not None
        logger._capture(["A", "A", "B"], prompts=None, gold_gloss=None)
        # second batch malformed → diagnostics reset to None, not stale
        assert logger.last_group_diagnostics is None


class TestExistingSampleBehaviorPreserved:
    def test_buffer_still_first_n_samples(self):
        logger = _make_logger(n_samples=2)
        logger._capture(
            ["IX MAN", "DOG RUN", "CAT SLEEP", "BOOK READ"],
            prompts=None,
            gold_gloss=["IX MAN", "DOG RUN", "CAT SLEEP", "BOOK READ"],
        )
        assert len(logger._buffer) == 2
        assert logger._buffer[0]["completion"] == "IX MAN"
        assert logger._buffer[1]["completion"] == "DOG RUN"
        assert logger._buffer[0]["gold"] == "IX MAN"

    def test_interceptor_still_wraps_and_forwards(self):
        calls = []

        def reward(completions, prompts=None, **kwargs):
            calls.append(kwargs)
            return [1.0] * len(completions)

        reward.__name__ = "reward"
        logger = CompletionSampleLogger(
            reward_fns=[reward], reward_weights=[1.0], n_samples=1
        )
        out = logger.wrapped_reward_fns[0](
            ["A"], prompts=["u"], gold_gloss=["A"], extra_col="x"
        )
        assert out == [1.0]
        assert calls and calls[0]["gold_gloss"] == ["A"]
        assert len(logger._buffer) == 1

    def test_conversational_completions_still_extracted(self):
        logger = _make_logger(n_samples=1)
        conv = [{"role": "assistant", "content": "IX MAN WALK"}]
        logger._capture([conv], prompts=None, gold_gloss=["IX MAN WALK"])
        assert logger._buffer[0]["completion"] == "IX MAN WALK"
        # same extraction drives the group diagnostics
        logger2 = _make_logger(group_size=2)
        logger2._capture(
            [conv, "ix man walk"],
            prompts=None,
            gold_gloss=None,
        )
        assert logger2.last_group_diagnostics == {"unique_outputs_mean": 1.0}


# ---------------------------------------------------------------------------
# CompletionSampleCallback → wandb groups/* keys
# ---------------------------------------------------------------------------


class _FakeWandbRun:
    def __init__(self):
        self.logged: list[dict] = []
        self.defined: list[str] = []
        self.log_kwargs: list[dict] = []


def _install_fake_wandb(monkeypatch) -> _FakeWandbRun:
    """CompletionSampleCallback imports wandb lazily inside on_log —
    swap a stub into sys.modules so no network/SDK is needed. The stub
    carries a __spec__ so importlib.util.find_spec("wandb") (called by
    transformers when constructing TrainingArguments) stays functional."""
    import importlib.machinery

    fake_run = _FakeWandbRun()
    fake = types.ModuleType("wandb")

    def define_metric(name, **kwargs):  # noqa: ANN001
        fake_run.defined.append(name)

    def log(payload, **kwargs):  # noqa: ANN001
        fake_run.logged.append(payload)
        fake_run.log_kwargs.append(kwargs)

    setattr(fake, "__spec__", importlib.machinery.ModuleSpec("wandb", None))
    setattr(fake, "run", fake_run)  # truthy → wandb.run is active
    setattr(fake, "define_metric", define_metric)
    setattr(fake, "log", log)
    monkeypatch.setitem(sys.modules, "wandb", fake)
    return fake_run


class TestCallbackWandbGroups:
    def test_groups_keys_logged(self, monkeypatch):
        fake_run = _install_fake_wandb(monkeypatch)
        logger = _make_logger(group_size=2)
        logger._capture(["A B", "A B", "C", "C D"], prompts=None, gold_gloss=None)
        callback = CompletionSampleCallback(logger, every_n_steps=5)

        from transformers import TrainerControl, TrainerState, TrainingArguments

        state = TrainerState()
        state.global_step = 5
        callback.on_log(
            TrainingArguments(output_dir="x"),
            state,
            control=TrainerControl(),
            logs={"loss": 0.5},
        )
        group_logs = [
            p for p in fake_run.logged if any(k.startswith("groups/") for k in p)
        ]
        assert group_logs, "groups/* keys must be logged to wandb"
        merged = {k: v for p in group_logs for k, v in p.items()}
        assert merged["groups/unique_outputs_mean"] == pytest.approx(1.5)
        assert "groups/abs_length_error_mean" not in merged  # no gold → skipped
        assert "groups/unique_outputs_mean" in fake_run.defined

    def test_groups_reach_wandb_and_live_status(self, monkeypatch, reward_setup):
        fake_run = _install_fake_wandb(monkeypatch)
        from src.rewards.t2g_rewards import build_t2g_reward_functions

        reward_fns, reward_weights = build_t2g_reward_functions()
        logger = CompletionSampleLogger(
            reward_fns=reward_fns,
            reward_weights=reward_weights,
            n_samples=1,
            group_size=2,
        )
        logger._capture(
            ["IX MAN", "IX MAN WALK"],
            prompts=None,
            gold_gloss=["IX MAN", "IX MAN WALK"],
        )
        callback = CompletionSampleCallback(logger, every_n_steps=5)
        live_updates = []
        monkeypatch.setattr(
            "src.training.callbacks.live_status_set",
            lambda *a, **kw: live_updates.append(kw),
        )

        from transformers import TrainerControl, TrainerState, TrainingArguments

        state = TrainerState()
        state.global_step = 10
        callback.on_log(
            TrainingArguments(output_dir="x"),
            state,
            control=TrainerControl(),
            logs={"loss": 0.1},
        )
        all_keys = {k for p in fake_run.logged for k in p}
        assert "groups/unique_outputs_mean" in all_keys
        assert "rewards/edit_validity_reward" not in all_keys
        assert live_updates[-1]["step"] == 10
        assert "groups/unique_outputs_mean" in live_updates[-1]
        group_index = next(
            i
            for i, payload in enumerate(fake_run.logged)
            if "groups/unique_outputs_mean" in payload
        )
        assert fake_run.log_kwargs[group_index] == {"step": 10, "commit": False}


# ---------------------------------------------------------------------------
# TRL-native metric passthrough (requirement: verify, do not duplicate)
# ---------------------------------------------------------------------------

_TRL_NATIVE_LOGS = {
    "loss": 0.0123,
    "reward": 0.87,
    "reward_std": 0.21,
    "frac_reward_zero_std": 0.05,
    "entropy": 2.71,
    "kl": 0.004,
    "clip_ratio/region_mean": 0.33,
    "learning_rate": 5e-6,
    "epoch": 1.5,
}


class TestTrlNativePassthrough:
    def test_curated_metrics_one_explicit_step_log_and_filtering(self, monkeypatch):
        fake_run = _install_fake_wandb(monkeypatch)
        cb = HighPrecisionLogCallback()
        from transformers import TrainerControl, TrainerState, TrainingArguments

        state = TrainerState()
        state.global_step = 7
        logs: dict[str, object] = {
            key: i + 0.25 for i, key in enumerate(cb._WANDB_KEYS)
        }
        logs["reward_std"] = None
        logs["entropy"] = float("nan")
        logs["unknown"] = 9.0
        cb.on_log(TrainingArguments(output_dir="x"), state, TrainerControl(), logs=logs)
        assert len(fake_run.logged) == 1
        assert fake_run.log_kwargs == [{"step": 7}]
        assert set(fake_run.logged[0]) == {f"train/{key}" for key in cb._WANDB_KEYS} - {
            "train/reward_std",
            "train/entropy",
        }
        assert set(fake_run.defined) == {f"train/{key}" for key in cb._WANDB_KEYS}

        state.global_step = 8
        cb.on_log(
            TrainingArguments(output_dir="x"),
            state,
            TrainerControl(),
            logs={"loss": 1.0},
        )
        assert [kwargs["step"] for kwargs in fake_run.log_kwargs] == [7, 8]
        assert len(fake_run.defined) == len(cb._WANDB_KEYS)

    def test_trl_metrics_printed_undropped(self, capsys):
        """Every TRL-native key must reach the printed line (the
        HighPrecisionLogCallback reprints ALL log keys — it drops none and
        duplicates none)."""
        cb = HighPrecisionLogCallback()
        from transformers import TrainerControl, TrainerState, TrainingArguments

        state = TrainerState()
        state.global_step = 7
        cb.on_log(
            TrainingArguments(output_dir="x"),
            state,
            control=TrainerControl(),
            logs=dict(_TRL_NATIVE_LOGS),
        )
        printed = capsys.readouterr().out
        for key in (
            "entropy",
            "kl",
            "reward_std",
            "frac_reward_zero_std",
            "clip_ratio/region_mean",
        ):
            assert key in printed, f"TRL-native metric {key} was dropped"

    def test_none_filtering_live_status_intact(self, monkeypatch):
        """Partial log events (e.g. holdout eval with ONLY eval_loss) must
        not push None/missing fields into the live status — explicit None
        filtering remains."""
        received: list[dict] = []
        monkeypatch.setattr(
            "src.training.callbacks.live_status_set", lambda **kw: received.append(kw)
        )
        cb = HighPrecisionLogCallback()
        from transformers import TrainerControl, TrainerState, TrainingArguments

        state = TrainerState()
        state.global_step = 3

        # Full GRPO event: only present, non-None fields mapped
        cb.on_log(
            TrainingArguments(output_dir="x"),
            state,
            control=TrainerControl(),
            logs={"loss": 0.1, "learning_rate": None, "epoch": 2.0},
        )
        assert received[-1] == {"step": 3, "loss": 0.1, "epoch": 2.0}
        assert "lr" not in received[-1]  # None was filtered

        # Eval-only event: must NOT carry train metrics
        cb.on_log(
            TrainingArguments(output_dir="x"),
            state,
            control=TrainerControl(),
            logs={"eval_loss": 0.25},
        )
        assert received[-1] == {"step": 3, "eval_loss": 0.25}
