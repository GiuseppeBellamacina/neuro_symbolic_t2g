"""Regression: eval must merge the SFT-phase adapter for sft-grpo checkpoints.

``grpo_t2g_train.py`` merges the SFT adapter into the base model IN-MEMORY
ONLY before attaching a fresh LoRA for GRPO (``model_loader.py::_load_with_transformers``
/ ``_load_with_unsloth``); the merge is never written to disk. ``trainer.save_model``
on the resulting PEFT model therefore writes only the GRPO adapter's delta,
computed relative to a base+SFT it never records on its own.

``load_model_for_eval`` used to reconstruct the model as
``raw_base + GRPO_adapter``, silently dropping the entire SFT contribution —
the checkpoint that consumed most of the training wall-clock ended up not
represented in the evaluated model at all, for every ``sft_pretrain``-enabled
cell (``sft-grpo/*``, ``ablations/decoding/hot-rollout``, and any future cell
built on ``sft-grpo/few-shot.yaml``).

The SFT adapter survives on disk at ``<run_dir>/sft_pretrain/final/`` — a
sibling of the GRPO ``final/`` directory that ``grpo_t2g_train.py`` always
trains-or-copies there so the run stays self-contained. This file tests the
detection helper that locates it; the merge itself (``PeftModel`` + a real
base model) needs a GPU/model to exercise and is out of scope here.
"""

from __future__ import annotations

import json

from src.training.eval_t2g import _sft_pretrain_adapter_sibling


def _make_adapter_dir(path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / "adapter_config.json").write_text("{}", encoding="utf-8")
    (path / "adapter_model.safetensors").write_bytes(b"weights")


def test_finds_sft_pretrain_sibling_next_to_final(tmp_path):
    """The layout grpo_t2g_train.py actually produces for sft-grpo cells."""
    run_dir = tmp_path / "run_20260911_054107"
    grpo_final = run_dir / "final"
    _make_adapter_dir(grpo_final)
    _make_adapter_dir(run_dir / "sft_pretrain" / "final")

    found = _sft_pretrain_adapter_sibling(grpo_final)
    assert found == run_dir / "sft_pretrain" / "final"


def test_none_for_a_grpo_only_run_with_no_sft_phase(tmp_path):
    """grpo/*.yaml and ablations/decoding/no-grammar.yaml have no SFT phase —
    must not go looking for a merge that never happened."""
    run_dir = tmp_path / "run_1"
    grpo_final = run_dir / "final"
    _make_adapter_dir(grpo_final)

    assert _sft_pretrain_adapter_sibling(grpo_final) is None


def test_none_when_sft_pretrain_dir_exists_but_is_incomplete(tmp_path):
    """A dir present without adapter_config.json is not a usable adapter —
    e.g. an interrupted SFT phase. Must not be silently skipped as 'merged'."""
    run_dir = tmp_path / "run_1"
    grpo_final = run_dir / "final"
    _make_adapter_dir(grpo_final)
    (run_dir / "sft_pretrain" / "final").mkdir(parents=True)

    assert _sft_pretrain_adapter_sibling(grpo_final) is None


def test_none_for_a_full_merged_checkpoint(tmp_path):
    """Non-adapter checkpoints (a full saved model) never reach this helper
    in practice (load_model_for_eval only calls it when is_peft), but the
    helper itself must not assume a sibling exists."""
    run_dir = tmp_path / "run_1"
    grpo_final = run_dir / "final"
    grpo_final.mkdir(parents=True)
    (grpo_final / "config.json").write_text("{}", encoding="utf-8")

    assert _sft_pretrain_adapter_sibling(grpo_final) is None


def test_does_not_match_a_checkpoint_step_dir_as_the_run_root(tmp_path):
    """checkpoint-N is a sibling of final under the SAME run dir — the
    sft_pretrain lookup must key off the run dir, not off 'final' by name,
    so intermediate-checkpoint eval (--checkpoint .../checkpoint-4500) also
    finds the SFT phase."""
    run_dir = tmp_path / "run_1"
    ckpt = run_dir / "checkpoint-4500"
    _make_adapter_dir(ckpt)
    _make_adapter_dir(run_dir / "sft_pretrain" / "final")

    assert _sft_pretrain_adapter_sibling(ckpt) == run_dir / "sft_pretrain" / "final"


def test_real_cluster_layout_snapshot(tmp_path):
    """Encodes the actual on-disk layout observed on the cluster for an
    sft-grpo run, so a future refactor of grpo_t2g_train.py's save path
    trips this test rather than silently reintroducing the bug."""
    run_dir = tmp_path / "qwen25-05b" / "sft-grpo" / "few-shot" / "run_20260911_000000"
    _make_adapter_dir(run_dir / "final")
    _make_adapter_dir(run_dir / "sft_pretrain" / "final")
    (run_dir / "checkpoint-4500").mkdir()

    found = _sft_pretrain_adapter_sibling(run_dir / "final")
    assert found is not None
    assert json.loads((found / "adapter_config.json").read_text(encoding="utf-8")) == {}
