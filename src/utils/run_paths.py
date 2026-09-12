"""Map a checkpoint path to the ``<model_name>/<run_id>`` pair used for outputs.

Checkpoints live at ``experiments/checkpoints/<model>/<method>/<variant>/run_<ts>/final``
— the depth follows whatever ``training.output_dir`` the cell's config declares, so it is
NOT fixed: ``qwen25-05b/grpo/zero-shot`` is three levels, ``qwen25-05b/ablations/loss/dr-grpo``
is four, and the historical flat layout ``qwen25-05b-sft-grpo`` is one. Eval results and
figures mirror that layout under ``experiments/results`` / ``experiments/figures``.

Taking a FIXED number of segments after ``checkpoints/`` (the pre-nesting assumption) made
every cell sharing the first two segments resolve to the SAME output directory: both
``grpo/zero-shot`` and ``grpo/few-shot`` wrote to ``experiments/results/qwen25-05b/grpo/``,
and all seven ``ablations/*`` cells wrote to ``experiments/results/qwen25-05b/ablations/``,
each eval silently overwriting the previous cell's JSON and figures. Anchoring on the
``run_*`` segment instead keeps every cell separate at any nesting depth, and still returns
exactly the historical pair for the old flat layout.
"""

from __future__ import annotations

from pathlib import Path

__all__ = ["split_checkpoint_path"]


def split_checkpoint_path(path: str | Path) -> tuple[str, str] | None:
    """Split a checkpoint (or in-checkpoint file) path into ``(model_name, run_id)``.

    ``model_name`` is every segment between ``checkpoints/`` and the ``run_*`` directory,
    joined with ``/`` so callers can use it directly as a nested output path.
    ``run_id`` is the ``run_*`` directory itself.

    Args:
        path: Any path inside ``experiments/checkpoints/`` — an adapter dir, a ``final/``
            dir, or a file such as ``trainer_state.json``.

    Returns:
        ``(model_name, run_id)``, or ``None`` when *path* has no ``checkpoints`` segment
        (callers keep their own fallback for that case).

    Examples:
        >>> split_checkpoint_path("experiments/checkpoints/qwen25-05b/grpo/zero-shot/run_1/final")
        ('qwen25-05b/grpo/zero-shot', 'run_1')
        >>> split_checkpoint_path("experiments/checkpoints/qwen25-05b-sft-grpo/run_1/final")
        ('qwen25-05b-sft-grpo', 'run_1')
    """
    parts = Path(path).resolve().parts
    if "checkpoints" not in parts:
        return None

    after = parts[parts.index("checkpoints") + 1 :]
    if not after:
        return None

    # The LAST run_* wins: the SFT sub-phase nests under the GRPO run dir
    # (run_<ts>/sft_pretrain/final), never the other way around.
    run_positions = [i for i, seg in enumerate(after) if seg.startswith("run_")]
    if run_positions:
        pos = run_positions[-1]
        return "/".join(after[:pos]) or after[0], after[pos]

    # No run_* segment (non-standard path): keep the historical 2-level behaviour.
    if len(after) > 1:
        return after[0], after[1]
    return after[0], "default_run"
