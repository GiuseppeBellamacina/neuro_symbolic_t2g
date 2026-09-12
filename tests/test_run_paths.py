"""Checkpoint path -> (model_name, run_id) resolution.

Regression tests for the collision that silently overwrote eval results: every
cell sharing the first two path segments after ``checkpoints/`` resolved to the
same ``experiments/results/<model_name>/<run_id>`` directory.
"""

from __future__ import annotations

from src.utils.run_paths import split_checkpoint_path


def test_nested_variant_layout_keeps_full_path() -> None:
    """The current layout nests method/variant: all of it must survive."""
    assert split_checkpoint_path(
        "experiments/checkpoints/qwen25-05b/grpo/zero-shot/run_20260910_012602/final"
    ) == ("qwen25-05b/grpo/zero-shot", "run_20260910_012602")


def test_deeply_nested_ablation_layout() -> None:
    """Ablations nest one level deeper (family/cell) — still fully preserved."""
    assert split_checkpoint_path(
        "experiments/checkpoints/qwen25-05b/ablations/loss/dr-grpo/run_20260911_054107/final"
    ) == ("qwen25-05b/ablations/loss/dr-grpo", "run_20260911_054107")


def test_variants_of_same_method_do_not_collide() -> None:
    """THE regression: grpo/zero-shot and grpo/few-shot both resolved to
    ('qwen25-05b', 'grpo') and overwrote each other's eval JSON."""
    zero = split_checkpoint_path(
        "experiments/checkpoints/qwen25-05b/grpo/zero-shot/run_20260910_012602/final"
    )
    few = split_checkpoint_path(
        "experiments/checkpoints/qwen25-05b/grpo/few-shot/run_20260910_152603/final"
    )
    assert zero != few
    assert zero[0] != few[0]


def test_every_ablation_cell_is_distinct() -> None:
    """All seven ablation cells previously collapsed to ('qwen25-05b', 'ablations')."""
    cells = [
        "ablations/decoding/no-grammar",
        "ablations/decoding/hot-rollout",
        "ablations/rewards/edit-validity",
        "ablations/rewards/historical-stack",
        "ablations/rewards/lean-stack",
        "ablations/loss/dr-grpo",
        "ablations/loss/low-beta",
        "ablations/objectives/sft-allowed-mass",
        "ablations/objectives/sft-structured",
    ]
    resolved = {
        split_checkpoint_path(
            f"experiments/checkpoints/qwen25-05b/{cell}/run_20260911_000000/final"
        )
        for cell in cells
    }
    assert len(resolved) == len(cells)


def test_legacy_flat_layout_unchanged() -> None:
    """Historical results used one flat dashed name — behaviour must not change."""
    assert split_checkpoint_path(
        "experiments/checkpoints/qwen25-05b-sft-grpo/run_20260904_062558/final"
    ) == ("qwen25-05b-sft-grpo", "run_20260904_062558")


def test_intermediate_checkpoint_resolves_to_its_run() -> None:
    """A checkpoint-N dir belongs to the same run as its final/."""
    assert split_checkpoint_path(
        "experiments/checkpoints/qwen25-05b/grpo/few-shot/run_1/checkpoint-4500"
    ) == ("qwen25-05b/grpo/few-shot", "run_1")


def test_sft_subphase_nested_under_run_dir() -> None:
    """The SFT sub-phase adapter lives INSIDE the run dir; the run still wins."""
    assert split_checkpoint_path(
        "experiments/checkpoints/qwen25-05b/sft-grpo/few-shot/run_1/sft_pretrain/final"
    ) == ("qwen25-05b/sft-grpo/few-shot", "run_1")


def test_trainer_state_file_path() -> None:
    """show_training_log passes a file path, not a directory."""
    assert split_checkpoint_path(
        "experiments/checkpoints/qwen25-05b/sft/zero-shot/run_1/checkpoint-200/trainer_state.json"
    ) == ("qwen25-05b/sft/zero-shot", "run_1")


def test_path_without_checkpoints_segment_returns_none() -> None:
    """Callers keep their own fallback for unparseable paths."""
    assert split_checkpoint_path("/tmp/some/random/model/final") is None


def test_no_run_segment_falls_back_to_two_levels() -> None:
    """Non-standard layout without run_*: historical 2-level behaviour."""
    assert split_checkpoint_path("experiments/checkpoints/mymodel/final") == (
        "mymodel",
        "final",
    )
