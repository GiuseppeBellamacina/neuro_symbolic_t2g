"""Checkpoint path -> (model_name, run_id) resolution.

Regression tests for the collision that silently overwrote eval results: every
cell sharing the first two path segments after ``checkpoints/`` resolved to the
same ``experiments/results/<model_name>/<run_id>`` directory.
"""

from __future__ import annotations

from pathlib import Path

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


def test_no_run_segment_is_not_guessed() -> None:
    """Without a run_* segment there is no run to attach outputs to.

    The old fallback returned the first two segments, so
    ``checkpoints/qwen25-05b/grpo/final`` became ``('qwen25-05b', 'grpo')`` and
    the eval wrote ``results/qwen25-05b/grpo/eval_final.json``: an orphan in a
    group directory, attributable to no cell. The caller now opens a fresh run
    in the config's cell instead.
    """
    assert (
        split_checkpoint_path("experiments/checkpoints/qwen25-05b/grpo/final") is None
    )
    assert split_checkpoint_path("experiments/checkpoints/run_1/final") is None


def test_cell_from_config_mirrors_the_config_tree() -> None:
    """Baselines have no checkpoint: their cell is the config's own path, the
    same convention every trained cell follows through its output_dir."""
    from src.utils.run_paths import cell_from_config

    for config, cell in [
        (
            "experiments/configs/qwen25-05b/baseline/zero-shot.yaml",
            "qwen25-05b/baseline/zero-shot",
        ),
        (
            "experiments/configs/qwen25-05b/baseline/zero-shot-no-grammar.yaml",
            "qwen25-05b/baseline/zero-shot-no-grammar",
        ),
        (
            "experiments/configs/qwen25-05b/ablations/loss/dr-grpo.yaml",
            "qwen25-05b/ablations/loss/dr-grpo",
        ),
    ]:
        assert cell_from_config(config) == cell
    assert cell_from_config("/tmp/scratch.yaml") is None


_CFG = {"model": {"name": "Qwen/Qwen2.5-0.5B-Instruct"}, "wandb": {"run_name": "x"}}


def test_baseline_evals_go_inside_the_model_tree_in_a_run_dir() -> None:
    """The three baselines have no checkpoint: each writes to
    <model>/baseline/<variant>/run_<ts>/, like every trained cell."""
    from src.utils.run_paths import eval_output_location

    for variant in ("zero-shot", "few-shot", "zero-shot-no-grammar"):
        cell, run_id, label = eval_output_location(
            f"experiments/configs/qwen25-05b/baseline/{variant}.yaml",
            _CFG,
            None,
            "20260929_120000",
        )
        assert (cell, run_id) == (
            f"qwen25-05b/baseline/{variant}",
            "run_20260929_120000",
        )
        assert label == "zero-shot"


def test_checkpoint_eval_joins_its_training_run() -> None:
    """An eval of a trained checkpoint lands in the SAME cell and run as the
    training, whatever the config path says."""
    from src.utils.run_paths import eval_output_location

    assert eval_output_location(
        "experiments/configs/qwen25-05b/grpo/few-shot.yaml",
        _CFG,
        "experiments/checkpoints/qwen25-05b/grpo/few-shot/run_20260914_070600/final",
        "20260929_120000",
    ) == ("qwen25-05b/grpo/few-shot", "run_20260914_070600", "run_20260914_070600")


def test_checkpoint_without_a_run_opens_a_fresh_run_in_the_config_cell() -> None:
    """This is the path that used to write orphans straight into a group
    directory (results/qwen25-05b/grpo/eval_final.json)."""
    from src.utils.run_paths import eval_output_location

    cell, run_id, label = eval_output_location(
        "experiments/configs/qwen25-05b/grpo/few-shot.yaml",
        _CFG,
        "experiments/checkpoints/qwen25-05b/grpo/final",
        "20260929_120000",
    )
    assert (cell, run_id) == ("qwen25-05b/grpo/few-shot", "run_20260929_120000")
    assert label == "grpo"


def test_every_eval_location_is_inside_a_run() -> None:
    """The invariant every reader relies on, over every config in the repo."""
    from src.utils.config import resolve_config
    from src.utils.run_paths import DATASET_KEYS, eval_output_location

    configs = sorted(Path("experiments/configs").rglob("*.yaml"))
    assert configs
    for path in configs:
        cell, run_id, _ = eval_output_location(
            path, resolve_config(str(path)), None, "20260929_120000"
        )
        assert run_id.startswith("run_"), path
        dataset = path.relative_to("experiments/configs").parts[0]
        assert dataset in DATASET_KEYS, path
        assert cell.startswith(f"{dataset}/qwen25-05b/"), (path, cell)


def test_split_cell_and_tag_with_dataset_segment() -> None:
    from src.utils.run_paths import cell_tag, split_cell

    assert split_cell("aslg-pc12/qwen25-05b/ablations/loss/dr-grpo") == (
        "aslg-pc12",
        "qwen25-05b",
        "ablations/loss/dr-grpo",
    )
    # Legacy (senza dataset) = ASLG-PC12, stesso tag con il prefisso.
    assert split_cell("qwen25-05b/grpo/few-shot") == (
        "aslg-pc12",
        "qwen25-05b",
        "grpo/few-shot",
    )
    assert cell_tag("qwen25-05b/grpo/few-shot") == "aslg-pc12-grpo-few-shot"
    assert cell_tag("aslg-pc12/qwen25-05b/grpo/few-shot") == "aslg-pc12-grpo-few-shot"
    assert cell_tag("phoenix-2014t/qwen25-05b/grpo/few-shot") == (
        "phoenix-2014t-grpo-few-shot"
    )


def test_cell_sort_key_groups_by_dataset_then_model_then_family() -> None:
    from src.utils.run_paths import cell_sort_key

    cells = [
        "phoenix-2014t/qwen25-05b/baseline/zero-shot",
        "aslg-pc12/qwen25-05b/ablations/loss/dr-grpo",
        "aslg-pc12/qwen25-05b/grpo/few-shot",
        "aslg-pc12/qwen25-05b/baseline/zero-shot",
    ]
    assert sorted(cells, key=cell_sort_key) == [
        "aslg-pc12/qwen25-05b/baseline/zero-shot",
        "aslg-pc12/qwen25-05b/grpo/few-shot",
        "aslg-pc12/qwen25-05b/ablations/loss/dr-grpo",
        "phoenix-2014t/qwen25-05b/baseline/zero-shot",
    ]


def test_phoenix_baseline_eval_lands_in_phoenix_tree() -> None:
    from src.utils.config import resolve_config
    from src.utils.run_paths import eval_output_location

    path = Path("experiments/configs/phoenix-2014t/qwen25-05b/baseline/zero-shot.yaml")
    cell, run_id, _ = eval_output_location(
        path, resolve_config(str(path)), None, "20261002_000000"
    )
    assert (cell, run_id) == (
        "phoenix-2014t/qwen25-05b/baseline/zero-shot",
        "run_20261002_000000",
    )


def test_job_tag_and_wandb_name_always_start_with_the_dataset() -> None:
    """Tag del job, nome della run wandb e primi tag wandb seguono UNA regola
    per tutti i dataset, ASLG-PC12 compreso: nella TUI e su wandb ogni job
    dice a quale dataset appartiene."""
    from src.utils.config import resolve_config
    from src.utils.run_paths import cell_tag, split_cell

    display = {
        "aslg-pc12": "ASLG-PC12",
        "phoenix-2014t": "PHOENIX-2014T",
        "wos-46985": "WOS-46985",
        "conll-2003": "CoNLL-2003",
    }
    configs = sorted(Path("experiments/configs").rglob("*.yaml"))
    assert configs
    for path in configs:
        cell = cell_from_config(path)
        dataset, model, rest = split_cell(cell)
        wandb_cfg = resolve_config(str(path))["wandb"]
        expected = f"{dataset}-{model}" + (
            "" if rest == "base" else "-" + rest.replace("/", "-")
        )
        assert wandb_cfg["run_name"] == expected, path
        assert wandb_cfg["tags"][:3] == ["T2G", display[dataset], model], path
        if rest != "base":
            assert cell_tag(cell).startswith(f"{dataset}-"), path


def cell_from_config(path):  # noqa: D103 - alias locale per il test sopra
    from src.utils.run_paths import cell_from_config as _cfc

    return _cfc(path)
