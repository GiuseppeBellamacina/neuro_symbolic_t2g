"""Custom TrainerCallbacks for T2G GRPO and SFT training."""

from __future__ import annotations

import logging
import math
import numbers
import re
from collections import deque
from typing import Any, Callable

from transformers import (
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
)
from transformers.trainer_callback import ProgressCallback

from src.utils.live_status import live_status_add_samples, live_status_set
from src.utils.metrics import compute_group_diagnostics
from src.utils.text_utils import extract_user_text

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Progress and log formatting
# ---------------------------------------------------------------------------


class TqdmOnlyProgressCallback(ProgressCallback):
    """ProgressCallback that keeps the tqdm bar but suppresses the
    duplicate dict-style log line printed by the default ``on_log``.
    """

    def on_log(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        logs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        pass


class HighPrecisionLogCallback(TrainerCallback):
    """Print metrics and replace the deliberately disabled W&B callback."""

    _WANDB_KEYS = (
        "reward",
        "reward_std",
        "frac_reward_zero_std",
        "entropy",
        "kl",
        "clip_ratio/low_mean",
        "clip_ratio/high_mean",
        "clip_ratio/region_mean",
        "importance_sampling/ratio_min",
        "importance_sampling/ratio_mean",
        "importance_sampling/ratio_max",
        "completions/mean_length",
        "completions/clipped_ratio",
        "completions/mean_terminated_length",
        "loss",
        "grad_norm",
        "learning_rate",
    )

    def __init__(self, reward_function_name: str = "edit_validity_reward") -> None:
        self._reward_sum: float = 0.0
        self._reward_count: int = 0
        self._wandb_metrics_defined = False
        self._reward_function_name = reward_function_name

    def on_train_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: Any,
    ) -> None:
        self._reward_sum = 0.0
        self._reward_count = 0

    @staticmethod
    def _finite_scalar(value: Any) -> float | int | None:
        if not isinstance(value, numbers.Real) or isinstance(value, bool):
            return None
        return float(value) if math.isfinite(float(value)) else None

    def _log_wandb(self, step: int, logs: dict[str, Any]) -> None:
        try:
            import wandb

            if not wandb.run:
                return
            if not self._wandb_metrics_defined:
                for key in (*self._WANDB_KEYS, *self._reward_keys()):
                    wandb.define_metric(f"train/{key}", summary="last")
                self._wandb_metrics_defined = True
            payload = {
                f"train/{key}": value
                for key in (*self._WANDB_KEYS, *self._reward_keys())
                if (value := self._finite_scalar(logs.get(key))) is not None
            }
            if payload:
                wandb.log(payload, step=step)
        except Exception:
            logger.debug(
                "Failed to log curated training metrics to wandb", exc_info=True
            )

    def _reward_keys(self) -> tuple[str, str]:
        prefix = f"rewards/{self._reward_function_name}"
        return f"{prefix}/mean", f"{prefix}/std"

    def on_log(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        logs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        if not state.is_local_process_zero or not logs:
            return
        logs = {key: value for key, value in logs.items() if key != "total_flos"}
        parts = [f"step={state.global_step}"]
        for k, v in logs.items():
            parts.append(f"{k}={v:.8f}" if isinstance(v, float) else f"{k}={v}")
        print("  " + "  ".join(parts))
        # Only present fields are published; None explicitly resets status fields.
        fields: dict[str, Any] = {"step": state.global_step}
        for log_key, status_key in (
            ("loss", "loss"),
            ("reward", "reward"),
            ("learning_rate", "lr"),
            ("eval_loss", "eval_loss"),
        ):
            value = logs.get(log_key)
            if value is not None:
                fields[status_key] = value
        if logs.get("epoch") is not None:
            fields["epoch"] = round(float(logs["epoch"]), 4)
        reward = logs.get("reward")
        if reward is not None:
            self._reward_sum += float(reward)
            self._reward_count += 1
            fields["reward_avg"] = round(self._reward_sum / self._reward_count, 6)
        live_status_set(**fields)
        self._log_wandb(state.global_step, logs)

    def on_prediction_step(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: Any,
    ) -> None:
        """Mark routine evaluation active without replacing train metrics."""
        try:
            if state.is_local_process_zero:
                live_status_set(eval_active=True)
        except Exception:
            pass

    def on_evaluate(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        metrics: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Publish routine evaluation loss and clear its active flag."""
        try:
            metrics = metrics or {}
            eval_loss = metrics.get("eval_loss")
            if eval_loss is not None:
                print(f"  step={state.global_step}  eval_loss={float(eval_loss):.8f}")
            if eval_loss is None:
                live_status_set(eval_active=False)
            else:
                live_status_set(eval_active=False, eval_loss=float(eval_loss))
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Completion sample logging
# ---------------------------------------------------------------------------

_SEPARATOR = "─" * 70
_THINK_RE = re.compile(r"grounded(.*?)grounded", re.DOTALL)


def _split_think(text: str) -> tuple[str, str]:
    """Split completion into (think_content, output_content)."""
    m = _THINK_RE.search(text)
    if m:
        think = m.group(1).strip()
        output = text[m.end() :].strip()
        return think, output
    return "", text.strip()


def _first_assistant_content(completion: Any) -> str | None:
    """Extract the first assistant message or a plain completion string."""
    if isinstance(completion, list):
        for msg in completion:
            if isinstance(msg, dict) and msg.get("role") == "assistant":
                content = str(msg.get("content", "")).strip()
                return content or None
    if isinstance(completion, str):
        content = completion.strip()
        return content or None
    return None


def _completion_text(completion: Any) -> str:
    """Extract plain text from conversational or string completions."""
    if isinstance(completion, list):
        for msg in completion:
            if isinstance(msg, dict) and msg.get("role") == "assistant":
                return str(msg.get("content", ""))
        return ""
    return str(completion or "")


def _infer_group_size(prompts: list[Any] | None) -> int | None:
    """Infer a uniform GRPO group size from consecutive identical prompts."""
    if not prompts:
        return None
    keys = [extract_user_text(p) for p in prompts]
    runs: list[int] = []
    prev = keys[0]
    count = 1
    for key in keys[1:]:
        if key == prev:
            count += 1
        else:
            runs.append(count)
            prev = key
            count = 1
    runs.append(count)
    if len(set(runs)) != 1:
        return None  # ragged runs — refuse to guess
    g = runs[0]
    if g < 2 or len(keys) % g != 0:
        return None
    return g


def _render_generation_prompt(tokenizer: Any, prompt: Any) -> str:
    """Render conversational prompts, falling back to simple ChatML text."""
    if isinstance(prompt, list):
        if (
            hasattr(tokenizer, "apply_chat_template")
            and getattr(tokenizer, "chat_template", None) is not None
        ):
            try:
                return tokenizer.apply_chat_template(
                    prompt,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            except Exception:
                pass
        return "\n".join(
            f"<|im_start|>{msg.get('role', 'user')}\n{msg.get('content', '')}"
            f"<|im_end|>"
            for msg in prompt
            if isinstance(msg, dict)
        )
    return str(prompt or "")


class CompletionSampleLogger:
    """Capture reward samples and full-batch GRPO group diagnostics."""

    def __init__(
        self,
        reward_fns: list[Callable[..., list[float]]],
        reward_weights: list[float],
        n_samples: int = 3,
        group_size: int | None = None,
    ) -> None:
        self._reward_fns = list(reward_fns)
        self._reward_weights = list(reward_weights)
        self._n_samples = n_samples
        self._group_size = group_size
        self._buffer: deque[dict[str, Any]] = deque(maxlen=n_samples)
        self._difficulty_map: dict[str, str] = {}
        self.last_group_diagnostics: dict[str, float] | None = None

        self._weight_map = {
            fn.__name__: weight for fn, weight in zip(reward_fns, reward_weights)
        }
        self._component_fns = list(reward_fns)
        # Guard: no reward functions to wrap
        if not self._reward_fns:
            logger.error(
                "CompletionSampleLogger: reward_fns is empty; "
                "no completion samples will be captured."
            )
            return

        original_fn = self._reward_fns[0]

        def _interceptor(
            completions: list[Any],
            prompts: list[Any] | None = None,
            **kwargs: Any,
        ) -> list[float]:
            self._capture(
                completions,
                prompts,
                gold_gloss=kwargs.get("gold_gloss"),
            )
            return original_fn(completions, prompts=prompts, **kwargs)

        _interceptor.__name__ = original_fn.__name__
        self._reward_fns[0] = _interceptor

    def set_difficulty_map(self, dataset: Any) -> None:
        """Build a normalized prompt-to-difficulty lookup."""
        for row in dataset:
            if not isinstance(row, dict):
                continue
            user_text = extract_user_text(row.get("prompt", ""))
            diff = row.get("difficulty", "")
            if user_text and diff:
                self._difficulty_map[user_text] = str(diff)

    def _capture(
        self,
        completions: list[Any],
        prompts: list[Any] | None,
        gold_gloss: list[str] | None = None,
    ) -> None:
        """Store display samples and compute full-batch diagnostics."""
        if not self._reward_fns:
            return
        self._buffer.clear()
        n = min(self._n_samples, len(completions))
        for i in range(n):
            comp = completions[i]
            text: str = _completion_text(comp)
            prompt = prompts[i] if prompts else None
            instruction = extract_user_text(prompt)

            difficulty = self._difficulty_map.get(instruction, "?")

            # Per-sample gold reference from the current batch's kwargs
            gold: str = ""
            if gold_gloss is not None and i < len(gold_gloss):
                gold = str(gold_gloss[i] or "")

            breakdown: dict[str, float] = {}
            for fn in self._component_fns:
                name = fn.__name__
                if self._weight_map.get(name, 0.0) <= 0.0:
                    continue
                try:
                    scores = fn([text], prompts=[prompt], gold_gloss=[gold])
                    breakdown[name] = float(scores[0])
                except Exception:
                    breakdown[name] = 0.0

            self._buffer.append(
                {
                    "instruction": instruction,
                    "completion": text,
                    "difficulty": difficulty,
                    "breakdown": breakdown,
                    "gold": gold,
                }
            )

        self.last_group_diagnostics = None
        effective_group_size = self._group_size
        if effective_group_size is None and prompts is not None:
            effective_group_size = _infer_group_size(prompts)
        if effective_group_size is not None:
            try:
                self.last_group_diagnostics = compute_group_diagnostics(
                    [_completion_text(c) for c in completions],
                    (
                        [str(g or "") for g in gold_gloss]
                        if gold_gloss is not None
                        else None
                    ),
                    effective_group_size,
                )
            except ValueError as exc:
                logger.warning("Group diagnostics skipped for this batch: %s", exc)

    @property
    def wrapped_reward_fns(self) -> list[Callable[..., list[float]]]:
        return self._reward_fns

    def format_samples(self) -> str:
        """Format buffered samples as a readable string for logging.

        Also pushes the formatted samples to the live status file (one
        string per sample, PROMPT/OUTPUT/GOLD compact) so the external
        monitor shows them without parsing the log.
        """
        if not self._buffer:
            return ""
        lines = [
            f"\n{'═' * 70}",
            "  COMPLETION SAMPLES",
            f"{'═' * 70}",
        ]
        live_lines: list[str] = []
        for idx, sample in enumerate(self._buffer, 1):
            instr = sample["instruction"]
            comp = sample["completion"]
            bd = sample["breakdown"]

            # Only display rewards that are active (weight > 0.0 in self._weight_map)
            active_bd = {
                k: v for k, v in bd.items() if self._weight_map.get(k, 0.0) > 0.0
            }
            row1 = "  ".join(f"{k}={v:+.2f}" for k, v in active_bd.items())

            lines.append(f"\n{_SEPARATOR}")
            # Show match indicator (✓/✗) when gold is available
            gold = sample.get("gold", "")
            think, output = _split_think(comp)
            difficulty = sample.get("difficulty", "?")
            difficulty_badge = f"[difficulty={difficulty}]"
            if gold:
                match = output.strip().upper() == gold.strip().upper()
                indicator = "✓" if match else "✗"
                lines.append(f"  Sample {idx}  {difficulty_badge} [{indicator}]")
                live_head = f"[{indicator}]"
            else:
                lines.append(f"  Sample {idx}  {difficulty_badge}")
                live_head = ""
            lines.append(_SEPARATOR)
            lines.append(f"  PROMPT: {instr}")
            if think:
                lines.append("  THINK:")
                for cl in think.splitlines():
                    lines.append(f"    {cl}")
            lines.append("  OUTPUT:")
            for cl in output.splitlines():
                lines.append(f"    {cl}")
            # Gold reference gloss (correct answer) for quick comparison.
            # Graceful marker when the gold_gloss kwarg was unavailable.
            if gold:
                lines.append("  GOLD:")
                for cl in gold.splitlines():
                    lines.append(f"    {cl}")
                gold_txt = gold
            else:
                lines.append("  GOLD: (gold non disponibile)")
                gold_txt = ""
            lines.append(f"  REWARDS: {row1}")
            total = sum(self._weight_map.get(k, 0.0) * v for k, v in bd.items())
            lines.append(f"  TOTAL:   {total:+.4f}")
            live_lines.append(
                f"{live_head} {instr[:100]}\n  → {output[:100]}"
                + (f"\n  gold: {gold_txt[:100]}" if gold_txt else "")
            )
        lines.append(f"{'═' * 70}\n")
        if live_lines:
            live_status_add_samples(live_lines, kind="grpo")
        return "\n".join(lines)


class CompletionSampleCallback(TrainerCallback):
    """Print samples and log grammar, reward, and group diagnostics."""

    def __init__(
        self,
        logger: CompletionSampleLogger,
        every_n_steps: int = 5,
        logits_processor: Any = None,
        plot_every_n: int = 25,
    ) -> None:
        self._logger = logger
        self._every_n_steps = every_n_steps
        self._last_printed_step = -1
        self._logits_processor = logits_processor
        self._plot_every_n = plot_every_n
        # Buffer per il pannello diagnostico convergenza
        self._diag_buffer: deque[dict[str, float]] = deque(maxlen=500)
        self._diag_defined = False
        # Pannello group diagnostics (groups/unique_outputs_mean,
        # groups/abs_length_error_mean) — metric layout defined once.
        self._groups_defined = False

    def on_log(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        logs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        if not state.is_local_process_zero:
            return
        step = state.global_step
        if (
            step > 0
            and step % self._every_n_steps == 0
            and step != self._last_printed_step
        ):
            output = self._logger.format_samples()
            if output:
                print(output)
            self._last_printed_step = step

            group_diag = self._logger.last_group_diagnostics
            if group_diag:
                live_status_set(
                    None,
                    step=step,
                    **{
                        f"groups/{key}": float(value)
                        for key, value in group_diag.items()
                    },
                )

            # ── W&B import once for both panels ─────────────────────────
            try:
                import wandb
            except ImportError:
                wandb = None

            # Log exact constrained-decoding diagnostics to W&B and live status.
            if self._logits_processor is not None and hasattr(
                self._logits_processor, "get_diagnostics"
            ):
                try:
                    metric_keys = (
                        "allowed_mass_mean",
                        "removed_mass_mean",
                        "log_allowed_mass_mean",
                        "allowed_mass_min",
                        "entropy_raw_mean",
                        "entropy_allowed_mean",
                        "active_rows",
                        "steps",
                    )
                    if not self._diag_defined and wandb is not None and wandb.run:
                        for key in metric_keys:
                            wandb.define_metric(f"grammar/{key}", summary="last")
                        self._diag_defined = True

                    stats = self._logits_processor.get_diagnostics(reset_after=True)
                    if stats["steps"] > 0:
                        payload = {
                            f"grammar/{key}": float(stats[key]) for key in metric_keys
                        }
                        live_status_set(None, step=step, **payload)
                        if wandb is not None and wandb.run:
                            wandb.log(payload, step=step, commit=False)

                        # ── Buffer & plot convergence diagnostics ────────
                        self._diag_buffer.append(
                            {
                                "Step": step,
                                "removed_mass": float(stats["removed_mass_mean"]),
                                "full_entropy": float(stats["entropy_raw_mean"]),
                                "allowed_entropy": float(stats["entropy_allowed_mean"]),
                            }
                        )

                        if (
                            wandb is not None
                            and wandb.run
                            and step % self._plot_every_n == 0
                            and len(self._diag_buffer) >= 2
                        ):
                            xs = [d["Step"] for d in self._diag_buffer]
                            ys_mass = [d["removed_mass"] for d in self._diag_buffer]
                            ys_ent = [d["full_entropy"] for d in self._diag_buffer]
                            ys_ent_a = [d["allowed_entropy"] for d in self._diag_buffer]

                            wandb.log(
                                {
                                    "grammar/convergence_diagnostics": wandb.plot.line_series(
                                        xs=xs,
                                        ys=[ys_mass, ys_ent, ys_ent_a],
                                        keys=[
                                            "removed_mass",
                                            "full_entropy",
                                            "allowed_entropy",
                                        ],
                                        title="Grammar Convergence Diagnostics",
                                        xname="Step",
                                    )
                                },
                                step=step,
                                commit=False,
                            )
                except Exception:
                    logger.debug("Failed to log grammar diagnostics", exc_info=True)

            # ── Group diagnostics logging (preregistered instrumentation) ──
            # Per-group diversity (unique normalized outputs per group of G
            # completions) and mean absolute word-length error vs gold,
            # computed over the FULL rollout batch in
            # CompletionSampleLogger._capture. Separate try/except so a
            # failure here never disturbs the sample/reward panels above.
            if group_diag:
                try:
                    group_payload = {
                        f"groups/{key}": float(value)
                        for key, value in group_diag.items()
                    }
                    if not self._groups_defined and wandb is not None and wandb.run:
                        for metric_key in group_payload:
                            wandb.define_metric(metric_key, summary="last")
                        self._groups_defined = True
                    if wandb is not None and wandb.run:
                        wandb.log(group_payload, step=step, commit=False)
                except Exception:
                    logger.debug(
                        "Failed to log group diagnostics to wandb", exc_info=True
                    )


# ---------------------------------------------------------------------------
# SFT-specific callbacks
# ---------------------------------------------------------------------------


class SFTSampleCallback(TrainerCallback):
    """Log SFT loss progress and periodic sample predictions."""

    def __init__(
        self,
        tokenizer: Any | None = None,
        model: Any | None = None,
        dataset: Any | None = None,
        every_n_steps: int = 25,
        sample_every_n_steps: int = 100,
        n_samples: int = 2,
    ) -> None:
        self._tokenizer = tokenizer
        self._model = model
        self._dataset = dataset
        self._every_n_steps = every_n_steps
        self._sample_every_n = sample_every_n_steps
        self._n_samples = n_samples
        self._last_printed_step = -1
        self._last_sample_step = -1
        self._loss_history: deque[dict[str, float]] = deque(maxlen=200)

    def on_log(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        logs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        if not state.is_local_process_zero or not logs:
            return
        step = state.global_step

        # Track loss history
        if "loss" in logs:
            self._loss_history.append(
                {
                    "step": step,
                    "loss": float(logs["loss"]),
                    "lr": float(logs.get("learning_rate", 0.0)),
                }
            )

        # Periodic progress summary
        if (
            step > 0
            and step % self._every_n_steps == 0
            and step != self._last_printed_step
        ):
            self._print_progress(state, args)
            self._last_printed_step = step

        # Periodic sample prediction
        if (
            step > 0
            and step % self._sample_every_n == 0
            and step != self._last_sample_step
        ):
            self._print_sample_prediction(state)
            self._last_sample_step = step

    def on_train_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: Any,
    ) -> None:
        if not state.is_local_process_zero:
            return
        self._print_final_summary(state, args)

    def _print_progress(self, state: TrainerState, args: TrainingArguments) -> None:
        """Print a compact SFT progress line with loss trend."""
        if not self._loss_history:
            return
        recent = list(self._loss_history)
        last = recent[-1]
        avg_loss = sum(r["loss"] for r in recent) / len(recent)
        min_loss = min(r["loss"] for r in recent)
        max_steps = args.max_steps if args.max_steps > 0 else "?"
        pct = (
            f"{state.global_step / args.max_steps * 100:.1f}%"
            if args.max_steps > 0
            else "?"
        )
        print(
            f"  [sft] step={state.global_step}/{max_steps} ({pct})  "
            f"loss={last['loss']:.6f}  avg={avg_loss:.6f}  "
            f"min={min_loss:.6f}  lr={last['lr']:.2e}  "
            f"epoch={state.epoch:.2f}"
        )

    def _print_sample_prediction(self, state: TrainerState) -> None:
        """Generate and print a sample prediction from the current model."""
        if self._model is None or self._tokenizer is None or self._dataset is None:
            return
        try:
            import random

            import torch

            n = min(self._n_samples, len(self._dataset))
            indices = random.sample(range(len(self._dataset)), n)

            print(f"\n{'═' * 70}")
            print(f"  SFT SAMPLE PREDICTIONS (step {state.global_step})")
            print(f"{'═' * 70}")

            live_lines: list[str] = []
            for idx in indices:
                sample = self._dataset[idx]
                if not isinstance(sample, dict):
                    sample = {"prompt": str(sample), "completion": ""}

                # trl 0.24 conversational format: prompt = [system, user]
                # message list, completion = [assistant] gold-gloss message.
                # Graceful fallbacks keep the display working for datasets
                # still in transition to the new schema.
                prompt = sample.get("prompt", "")
                completion = sample.get("completion", "")
                user_text = extract_user_text(prompt)
                gold_part = _first_assistant_content(completion)
                if gold_part is None:
                    gold_part = "(unknown)"

                # Render the generation prompt via the chat template (or
                # pass preformatted strings through untouched)
                prompt_text = _render_generation_prompt(self._tokenizer, prompt)
                if not prompt_text.strip():
                    continue

                inputs = self._tokenizer(
                    prompt_text, return_tensors="pt", truncation=True, max_length=512
                )
                device = next(self._model.parameters()).device
                inputs = {k: v.to(device) for k, v in inputs.items()}

                with torch.no_grad():
                    out = self._model.generate(
                        **inputs,
                        max_new_tokens=64,
                        do_sample=False,
                        temperature=1.0,
                        pad_token_id=self._tokenizer.eos_token_id,
                    )
                generated = self._tokenizer.decode(
                    out[0][inputs["input_ids"].shape[1] :],
                    skip_special_tokens=True,
                ).strip()

                print(f"\n{_SEPARATOR}")
                print(f"  PROMPT: {user_text[:120]}")
                print(f"  GOLD:   {gold_part[:120]}")
                print(f"  PRED:   {generated[:120]}")
                match = (
                    "✓"
                    if generated.strip().upper() == gold_part.strip().upper()
                    else "✗"
                )
                live_lines.append(
                    f"{user_text[:100]}\n  [{match}] pred: {generated[:100]}"
                    f"\n  gold: {gold_part[:100]}"
                )
            print(f"{'═' * 70}\n")
            if live_lines:
                live_status_add_samples(live_lines, kind="sft")
        except Exception:
            logger.debug("Failed to generate SFT sample prediction", exc_info=True)

    def _print_final_summary(
        self, state: TrainerState, args: TrainingArguments
    ) -> None:
        """Print a final SFT training summary."""
        if not self._loss_history:
            return
        all_losses = [r["loss"] for r in self._loss_history]
        first_loss = all_losses[0]
        last_loss = all_losses[-1]
        min_loss = min(all_losses)

        print(f"\n{'═' * 70}")
        print("  SFT TRAINING SUMMARY")
        print(f"{'═' * 70}")
        print(f"  Total steps:      {state.global_step}")
        print(f"  Epochs completed: {state.epoch:.2f}")
        print(f"  Initial loss:     {first_loss:.6f}")
        print(f"  Final loss:       {last_loss:.6f}")
        print(f"  Min loss:         {min_loss:.6f}")
        print(
            f"  Loss reduction:   {first_loss - last_loss:.6f} "
            f"({(first_loss - last_loss) / max(first_loss, 1e-8) * 100:.1f}%)"
        )
        print(f"{'═' * 70}\n")
