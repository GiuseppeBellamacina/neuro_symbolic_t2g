"""HuggingFace Trainer adapter (also fits TRL trainers, e.g. GRPO/PPO, which log rewards through on_log)."""
from __future__ import annotations

from typing import Any, Callable

from .core import RunStatus

try:
    from transformers import TrainerCallback
except ImportError:  # keep the package importable without transformers
    TrainerCallback = object  # type: ignore[assignment,misc]

_SKIP = ("total_flos", "epoch")


class RunStatusCallback(TrainerCallback):
    """
    Args:
        status: the RunStatus to feed.
        phase: stage entered at train begin (e.g. "sft", "grpo"): see RunStatus.stage; may be left out when the
            script moves between stages itself.
        transform: optional ``f(logs: dict, state) -> dict`` to derive/clean metrics before they are recorded
            (e.g. compute reward std, drop keys, flatten nested values).
        log_eval: record the metrics of on_evaluate.
        track_eval: flag eval_active while the Trainer evaluates (on_prediction_step).
        on_save_event: add a "checkpoint saved" event on every save.
        extra: free dict merged into the status "extra" at train begin.
    """

    def __init__(self, status: RunStatus, *, phase: str | None = None,
                 transform: Callable[[dict, Any], dict] | None = None, log_eval: bool = True,
                 track_eval: bool = True, on_save_event: bool = True, extra: dict | None = None):
        self.status = status
        self.phase, self.transform, self.log_eval = phase, transform, log_eval
        self.track_eval, self.on_save_event, self.extra = track_eval, on_save_event, extra

    def on_train_begin(self, args, state, control, **kw):
        if state.is_local_process_zero:
            if self.phase:
                self.status.stage(self.phase, total_steps=state.max_steps or None)
            else:
                self.status.set(total_steps=state.max_steps or None)
            if self.extra:
                self.status.set(extra=self.extra)

    def on_log(self, args, state, control, logs: dict[str, Any] | None = None, **kw):
        if state.is_local_process_zero and logs:
            logs = {k: v for k, v in logs.items() if k not in _SKIP}
            if self.transform:
                logs = self.transform(logs, state) or {}
            self.status.set(epoch=state.epoch, total_steps=state.max_steps or None)  # the trainer knows the real total
            self.status.log(state.global_step, **logs)

    def on_prediction_step(self, args, state, control, **kw):
        if self.track_eval and state.is_local_process_zero:
            self.status.set(eval_active=True)

    def on_evaluate(self, args, state, control, metrics=None, **kw):
        if state.is_local_process_zero:
            self.status.set(eval_active=False)
            if self.log_eval and metrics:
                self.status.log(state.global_step, **metrics)
                self.status.event("eval done")

    def on_save(self, args, state, control, **kw):
        if self.on_save_event and state.is_local_process_zero:
            self.status.event(f"checkpoint saved at step {state.global_step}")

    def on_train_end(self, args, state, control, **kw):
        if state.is_local_process_zero:
            self.status.finish()
