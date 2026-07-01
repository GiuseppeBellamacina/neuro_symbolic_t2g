"""Custom TrainerCallbacks for T2G GRPO training."""

from __future__ import annotations

import hashlib
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
from transformers.trainer_callback import ProgressCallback

from src.utils.text_utils import extract_user_text

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Progress + log formatting (ported from grpo-strict-generation)
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
    """Print training metrics with higher float precision (8 decimal places).

    The default HuggingFace Trainer formats floats to 6 decimal places, which
    causes very small loss values (e.g. GRPO policy gradient loss) to appear
    as ``-0.000000``.  This callback reprints every ``on_log`` event to stdout
    with enough precision to see the actual values.
    """

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
        logs.pop("total_flos", None)
        parts = [f"step={state.global_step}"]
        for k, v in logs.items():
            parts.append(f"{k}={v:.8f}" if isinstance(v, float) else f"{k}={v}")
        print("  " + "  ".join(parts))


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
        output = text[m.end() :].strip()
        return think, output
    return "", text.strip()


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
            gold_structure_reward,
            structural_dense_reward,
            translation_quality_reward,
            viterbi_distance_reward,
        )

        self._component_fns: list[tuple[str, Callable[..., float], dict[str, Any]]] = [
            (
                "translation_quality_reward",
                translation_quality_reward,
                {"gold_gloss": ""},
            ),
            (
                "gold_structure_reward",
                gold_structure_reward,
                {"gold_gloss": "", "normalize": True},
            ),
            ("structural_dense_reward", structural_dense_reward, {"normalize": True}),
            ("viterbi_distance_reward", viterbi_distance_reward, {"normalize": True}),
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
            instruction = extract_user_text(prompt)

            # Use stable sample ID for format-agnostic lookup
            sample_id = self._extract_sample_id(prompt) if prompt is not None else ""
            difficulty = self._difficulty_map.get(sample_id, "?")

            breakdown: dict[str, float] = {}
            for name, fn, kwargs in self._component_fns:
                try:
                    kwargs_call = dict(kwargs)
                    # Dynamically look up the actual gold gloss
                    if name in (
                        "translation_quality_reward",
                        "gold_structure_reward",
                    ):
                        kwargs_call["gold_gloss"] = (
                            self._lookup_gold_gloss(prompt)
                            if prompt is not None
                            else ""
                        )
                    breakdown[name] = fn(text, **kwargs_call)
                except Exception:
                    breakdown[name] = 0.0

            self._buffer.append(
                {
                    "instruction": instruction,
                    "completion": text,
                    "difficulty": difficulty,
                    "breakdown": breakdown,
                }
            )

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
            bd = sample["breakdown"]

            # Only display rewards that are active (weight > 0.0 in self._weight_map)
            active_bd = {
                k: v for k, v in bd.items() if self._weight_map.get(k, 0.0) > 0.0
            }
            row1 = "  ".join(f"{k}={v:+.2f}" for k, v in active_bd.items())

            lines.append(f"\n{_SEPARATOR}")
            lines.append(f"  Sample {idx}")
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
            total = sum(self._weight_map.get(k, 0.0) * v for k, v in bd.items())
            lines.append(f"  TOTAL:   {total:+.4f}")
        lines.append(f"{'═' * 70}\n")
        return "\n".join(lines)


class CompletionSampleCallback(TrainerCallback):
    """Print completion samples and log grammar + reward metrics every ``every_n_steps``.

    These samples are parsed by ``chain_monitor.py`` for live display.
    Grammar metrics (masked probability mass) are logged to wandb
    to track how the model internalizes the ASL vocabulary constraints.

    Custom W&B chart panels:
    * ``grammar/convergence_diagnostics`` — masked_mass, full_entropy, allowed_entropy
    * ``rewards/breakdown_diagnostics`` — all 6 reward components together
    """

    # Reward component names (order determines legend order in W&B plot)
    _REWARD_COMPONENTS: tuple[str, ...] = (
        "translation_quality_reward",
        "gold_structure_reward",
        "structural_dense_reward",
        "viterbi_distance_reward",
        "gloss_format_reward",
        "gloss_repetition_reward",
    )

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
        # Buffer per il pannello reward breakdown
        self._reward_buffer: deque[dict[str, float]] = deque(maxlen=500)
        self._reward_defined = False

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

            # ── W&B import once for both panels ─────────────────────────
            try:
                import wandb
            except ImportError:
                return  # wandb not installed, skip both panels

            # Log masked probability mass / entropy to wandb
            if self._logits_processor is not None and hasattr(
                self._logits_processor, "get_masked_mass_stats"
            ):
                try:

                    # ── Define W&B metric layout once ───────────────────
                    if not self._diag_defined and wandb.run:
                        wandb.define_metric(
                            "grammar/masked_mass_avg",
                            summary="last",
                        )
                        wandb.define_metric(
                            "grammar/masked_entropy_avg",
                            summary="last",
                        )
                        wandb.define_metric(
                            "grammar/masked_entropy_allowed_avg",
                            summary="last",
                        )
                        self._diag_defined = True

                    # Use reset_after=True for per-interval metrics
                    stats = self._logits_processor.get_masked_mass_stats(
                        reset_after=True
                    )
                    if stats["total_steps"] > 0 and wandb.run:
                        mass = stats["avg_masked_mass"]
                        ent = stats.get("avg_masked_entropy", 0.0)
                        ent_allowed = stats.get("avg_masked_entropy_allowed", 0.0)

                        wandb.log(
                            {
                                "grammar/masked_mass_avg": mass,
                                "grammar/masked_entropy_avg": ent,
                                "grammar/masked_entropy_allowed_avg": ent_allowed,
                                "grammar/masked_mass_steps": stats["total_steps"],
                            },
                            step=step,
                        )

                        # ── Buffer & plot convergence diagnostics ────────
                        self._diag_buffer.append(
                            {
                                "Step": step,
                                "masked_mass": mass,
                                "full_entropy": ent,
                                "allowed_entropy": ent_allowed,
                            }
                        )

                        if (
                            step % self._plot_every_n == 0
                            and len(self._diag_buffer) >= 2
                        ):
                            xs = [d["Step"] for d in self._diag_buffer]
                            ys_mass = [d["masked_mass"] for d in self._diag_buffer]
                            ys_ent = [d["full_entropy"] for d in self._diag_buffer]
                            ys_ent_a = [d["allowed_entropy"] for d in self._diag_buffer]

                            wandb.log(
                                {
                                    "grammar/convergence_diagnostics": wandb.plot.line_series(
                                        xs=xs,
                                        ys=[ys_mass, ys_ent, ys_ent_a],
                                        keys=[
                                            "masked_mass",
                                            "full_entropy",
                                            "allowed_entropy",
                                        ],
                                        title="Grammar Convergence Diagnostics",
                                        xname="Step",
                                    )
                                },
                                step=step,
                            )
                except Exception:
                    logger.debug("Failed to log masked mass to wandb", exc_info=True)

            # ── Reward breakdown logging ───────────────────────────────
            if self._logger._buffer:
                try:
                    # Only log and plot components that are active (weight > 0)
                    active_components = [
                        c
                        for c in self._REWARD_COMPONENTS
                        if self._logger._weight_map.get(c, 0.0) > 0.0
                    ]

                    # Define reward metrics once
                    if not self._reward_defined and wandb.run:
                        for comp in active_components:
                            wandb.define_metric(
                                f"rewards/{comp}",
                                summary="last",
                            )
                        self._reward_defined = True

                    # Compute per-interval averages from buffered samples
                    reward_sums: dict[str, float] = {c: 0.0 for c in active_components}
                    n_samples = 0
                    for sample in self._logger._buffer:
                        bd = sample.get("breakdown", {})
                        for comp in active_components:
                            reward_sums[comp] += bd.get(comp, 0.0)
                        n_samples += 1

                    if n_samples > 0 and wandb.run:
                        reward_avgs = {
                            c: reward_sums[c] / n_samples for c in active_components
                        }

                        # Log individual scalars
                        wandb.log(
                            {
                                f"rewards/{comp}": reward_avgs[comp]
                                for comp in active_components
                            },
                            step=step,
                        )

                        # Buffer & plot reward breakdown panel
                        self._reward_buffer.append({"Step": step, **reward_avgs})

                        if (
                            step % self._plot_every_n == 0
                            and len(self._reward_buffer) >= 2
                        ):
                            xs = [d["Step"] for d in self._reward_buffer]
                            ys_list = [
                                [d[comp] for d in self._reward_buffer]
                                for comp in active_components
                            ]
                            # Derive short labels from component names
                            labels = [
                                c.replace("_reward", "") for c in active_components
                            ]

                            wandb.log(
                                {
                                    "rewards/breakdown_diagnostics": wandb.plot.line_series(
                                        xs=xs,
                                        ys=ys_list,
                                        keys=labels,
                                        title="Reward Component Convergence",
                                        xname="Step",
                                    )
                                },
                                step=step,
                            )
                except Exception:
                    logger.debug(
                        "Failed to log reward breakdown to wandb", exc_info=True
                    )
