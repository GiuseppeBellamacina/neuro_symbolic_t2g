"""Custom TrainerCallbacks for T2G GRPO training."""

from __future__ import annotations

import hashlib
import json as _json
import logging
import re
from collections import deque
from typing import Any, Callable

from transformers import (
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Completion sample logging
# ---------------------------------------------------------------------------

_SEPARATOR = "─" * 70
_THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)


def _split_think(text: str) -> tuple[str, str]:
    """Split completion into (think_content, output_content)."""
    m = _THINK_RE.search(text)
    if m:
        think = m.group(1).strip()
        output = text[m.end():].strip()
        return think, output
    return "", text.strip()


def _extract_user_instruction(prompt: Any) -> str:
    """Extract the user message from a prompt (chat messages or string)."""
    if isinstance(prompt, list):
        for msg in reversed(prompt):
            if isinstance(msg, dict) and msg.get("role") == "user":
                return msg.get("content", "")
    return str(prompt) if prompt is not None else ""


class CompletionSampleLogger:
    """Wraps reward functions to capture (prompt, completion, rewards) samples.

    The first reward function is wrapped with an interceptor that stores
    the last batch of completions and prompts.  The callback reads from
    this buffer and prints periodically to the training log so
    ``chain_monitor.py`` can display them in real time.

    Usage::

        logger = CompletionSampleLogger(reward_fns, reward_weights, n_samples=3)
        trainer = GRPOTrainer(
            ...,
            reward_funcs=logger.wrapped_reward_fns,
            callbacks=[CompletionSampleCallback(logger, every_n_steps=5)],
        )
    """

    def __init__(
        self,
        reward_fns: list[Callable[..., list[float]]],
        reward_weights: list[float],
        n_samples: int = 3,
    ) -> None:
        self._reward_fns = list(reward_fns)
        self._reward_weights = list(reward_weights)
        self._n_samples = n_samples
        self._buffer: deque[dict[str, Any]] = deque(maxlen=n_samples)
        self._difficulty_map: dict[str, str] = {}

        # Build component_name → weight mapping
        self._weight_map: dict[str, float] = {}
        for fn, w in zip(reward_fns, reward_weights):
            self._weight_map[fn.__name__] = w

        # Component functions for per-sample breakdown (from t2g_rewards)
        from src.rewards.t2g_rewards import (
            _extract_sample_id,
            _lookup_gold_gloss,
            gloss_format_reward,
            gloss_repetition_reward,
            structural_dense_reward,
            translation_quality_reward,
        )

        self._component_fns: list[tuple[str, Callable[..., float], dict[str, Any]]] = [
            ("translation_quality_reward", translation_quality_reward, {"gold_gloss": ""}),
            ("structural_dense_reward", structural_dense_reward, {"normalize": True}),
            ("gloss_format_reward", gloss_format_reward, {}),
            ("gloss_repetition_reward", gloss_repetition_reward, {}),
        ]
        self._extract_sample_id = _extract_sample_id
        self._lookup_gold_gloss = _lookup_gold_gloss

        # Guard: no reward functions to wrap
        if not self._reward_fns:
            logger.error(
                "CompletionSampleLogger: reward_fns is empty; "
                "no completion samples will be captured."
            )
            return

        # Wrap the first reward function to intercept
        original_fn = self._reward_fns[0]

        def _interceptor(
            completions: list[Any],
            prompts: list[Any] | None = None,
            **kwargs: Any,
        ) -> list[float]:
            self._capture(completions, prompts)
            return original_fn(completions, prompts=prompts, **kwargs)

        _interceptor.__name__ = original_fn.__name__
        self._reward_fns[0] = _interceptor

    def set_difficulty_map(self, dataset: Any) -> None:
        """Build a prompt→difficulty lookup from the training dataset.

        Uses the stable sample ID (SHA256 of user instruction) as key
        for format-agnostic matching, same as the gold gloss registry.
        """
        for row in dataset:
            if not isinstance(row, dict):
                continue
            user_text = row.get("prompt", "")
            diff = row.get("difficulty", "")
            if user_text and diff:
                sample_id = hashlib.sha256(
                    str(user_text).encode("utf-8", errors="replace")
                ).hexdigest()
                self._difficulty_map[sample_id] = diff

    def _capture(self, completions: list[Any], prompts: list[Any] | None) -> None:
        """Store the first N samples from this batch."""
        if not self._reward_fns:
            return
        self._buffer.clear()
        n = min(self._n_samples, len(completions))
        for i in range(n):
            comp = completions[i]
            text: str = comp[0]["content"] if isinstance(comp, list) else comp
            prompt = prompts[i] if prompts else None
            instruction = _extract_user_instruction(prompt)

            # Use stable sample ID for format-agnostic lookup
            sample_id = self._extract_sample_id(prompt) if prompt is not None else ""
            difficulty = self._difficulty_map.get(sample_id, "?")

            breakdown: dict[str, float] = {}
            for name, fn, kwargs in self._component_fns:
                try:
                    kwargs_call = dict(kwargs)
                    # Dynamically look up the actual gold gloss
                    if name == "translation_quality_reward":
                        kwargs_call["gold_gloss"] = (
                            self._lookup_gold_gloss(prompt)
                            if prompt is not None else ""
                        )
                    breakdown[name] = fn(text, **kwargs_call)
                except Exception:
                    breakdown[name] = 0.0

            self._buffer.append({
                "instruction": instruction,
                "completion": text,
                "difficulty": difficulty,
                "breakdown": breakdown,
            })

    @property
    def wrapped_reward_fns(self) -> list[Callable[..., list[float]]]:
        return self._reward_fns

    def format_samples(self) -> str:
        """Format buffered samples as a readable string for logging."""
        if not self._buffer:
            return ""
        lines = [
            f"\n{'═' * 70}",
            "  COMPLETION SAMPLES",
            f"{'═' * 70}",
        ]
        for idx, sample in enumerate(self._buffer, 1):
            instr = sample["instruction"]
            comp = sample["completion"]
            diff = sample.get("difficulty", "?")
            bd = sample["breakdown"]
            bd_items = list(bd.items())
            row1 = "  ".join(f"{k}={v:+.2f}" for k, v in bd_items)

            lines.append(f"\n{_SEPARATOR}")
            lines.append(f"  Sample {idx}  [difficulty={diff}]")
            lines.append(f"{_SEPARATOR}")
            lines.append(f"  PROMPT: {instr}")
            think, output = _split_think(comp)
            if think:
                lines.append("  THINK:")
                for cl in think.splitlines():
                    lines.append(f"    {cl}")
            lines.append("  OUTPUT:")
            for cl in output.splitlines():
                lines.append(f"    {cl}")
            lines.append(f"  REWARDS: {row1}")
            total = sum(
                self._weight_map.get(k, 0.0) * v for k, v in bd.items()
            )
            lines.append(f"  TOTAL:   {total:+.4f}")
        lines.append(f"{'═' * 70}\n")
        return "\n".join(lines)


class CompletionSampleCallback(TrainerCallback):
    """Print completion samples every ``every_n_steps`` steps.

    These samples are parsed by ``chain_monitor.py`` for live display.
    """

    def __init__(self, logger: CompletionSampleLogger, every_n_steps: int = 5) -> None:
        self._logger = logger
        self._every_n_steps = every_n_steps
        self._last_printed_step = -1

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
        if step > 0 and step % self._every_n_steps == 0 and step != self._last_printed_step:
            output = self._logger.format_samples()
            if output:
                print(output)
            self._last_printed_step = step
