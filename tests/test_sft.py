"""Unit tests for SFT prompt-completion dataset construction and eval holdout.

Validates — without training, models, or network access:
  1. The prompt-completion builder emits the expected conversational columns
     and that trl loss-masking is active for prompt-completion data.
  2. The seeded eval holdout split is reproducible and disjoint.
  3. The standalone SFT YAML config exposes the new eval/early-stopping keys.

All tests are synthetic (no model, no tokenizer, no dataset download).
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import yaml

from datasets import Dataset
from src.training.sft_train import (
    _build_prompt_completion_example,
    compute_sft_fingerprint,
    find_reusable_sft_adapter_cross_method,
    is_complete_adapter_dir,
    resolve_sft_run_paths,
    split_eval_holdout,
    write_sft_fingerprint,
)
from src.utils.prompting import SYSTEM_PROMPT

SFT_CONFIG_PATH = (
    Path(__file__).resolve().parent.parent
    / "experiments"
    / "configs"
    / "qwen25-05b"
    / "sft"
    / "zero-shot.yaml"
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fake_t2g_row(
    prompt: str = "The man walks the dog.",
    completion: str = "IX MAN WALK DOG",
) -> dict:
    """A minimal row as returned by ``build_t2g_dataset`` today."""
    return {"prompt": prompt, "completion": completion, "difficulty": "simple"}


def _make_sft_dataset(n: int = 100) -> Dataset:
    """Build a synthetic prompt-completion SFT ``Dataset`` (no network)."""
    rows = [
        _build_prompt_completion_example(
            _fake_t2g_row(prompt=f"Sentence number {i} for the split test.")
        )
        for i in range(n)
    ]
    return Dataset.from_list(rows)


# ---------------------------------------------------------------------------
# 1. Prompt-completion builder + trl loss masking
# ---------------------------------------------------------------------------


def test_prompt_completion_builder_columns() -> None:
    """Builder produces the expected conversational prompt/completion columns."""
    ex = _build_prompt_completion_example(_fake_t2g_row())

    assert set(ex) == {
        "prompt",
        "completion",
        "gold_gloss",
        "difficulty",
        "sample_id",
    }
    assert ex["prompt"] == [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "The man walks the dog."},
    ]
    assert ex["completion"] == [{"role": "assistant", "content": "IX MAN WALK DOG"}]
    assert ex["gold_gloss"] == "IX MAN WALK DOG"
    assert ex["difficulty"] == "simple"
    assert (
        ex["sample_id"]
        == hashlib.sha256("The man walks the dog.".encode("utf-8")).hexdigest()
    )


def test_prompt_completion_builder_prefers_gold_gloss() -> None:
    """Builder prefers ``gold_gloss`` over ``completion`` when both exist."""
    row = {
        "prompt": "A sentence.",
        "completion": "OLD",
        "gold_gloss": "NEW",
        "difficulty": "medium",
        "sample_id": "abc123",
    }
    ex = _build_prompt_completion_example(row)

    assert ex["completion"] == [{"role": "assistant", "content": "NEW"}]
    assert ex["gold_gloss"] == "NEW"
    assert ex["difficulty"] == "medium"
    assert ex["sample_id"] == "abc123"


def test_prompt_completion_example_is_conversational() -> None:
    """trl recognizes the built example as conversational prompt-completion."""
    from trl.data_utils import is_conversational

    example = _build_prompt_completion_example(_fake_t2g_row())
    # Conversational prompt-completion → trl takes the chat-template branch
    # (trl/trainer/sft_trainer.py:952) and applies the tokenizer template
    # identically to the GRPO rollout prompts.
    assert is_conversational(example)


def test_completion_only_loss_default_and_masking(tmp_path) -> None:
    """trl masks prompt tokens for prompt-completion data (loss on gloss only)."""
    from trl import SFTConfig
    from trl.trainer.sft_trainer import DataCollatorForLanguageModeling

    # SFTConfig.completion_only_loss defaults to None; SFTTrainer auto-enables
    # completion-only loss when the dataset has prompt+completion columns
    # (trl/trainer/sft_trainer.py:733-739).  ``bf16=False`` keeps the
    # dataclass instantiable on machines without an Ampere+ GPU.
    cfg = SFTConfig(output_dir=str(tmp_path), report_to="none", bf16=False)
    assert cfg.completion_only_loss is None

    # The collator used by SFTTrainer turns every non-completion token into
    # -100 (trl/trainer/sft_trainer.py:211-215): the prompt is never counted.
    collator = DataCollatorForLanguageModeling(
        pad_token_id=0, completion_only_loss=True
    )
    out = collator([{"input_ids": [1, 2, 3, 4, 5], "completion_mask": [0, 0, 1, 1, 1]}])
    assert out["labels"].tolist() == [[-100, -100, 3, 4, 5]]


def test_prepare_sft_dataset_uses_eval_fraction(monkeypatch) -> None:
    """``_prepare_sft_dataset`` honors ``training.eval_fraction`` from config."""
    from src.training import sft_train

    fake_t2g = Dataset.from_list(
        [_fake_t2g_row(prompt=f"Sentence {i}.") for i in range(20)]
    )
    monkeypatch.setattr(sft_train, "build_t2g_dataset", lambda *a, **k: fake_t2g)
    monkeypatch.setattr(sft_train, "download_aslg_dataset", lambda *a, **k: None)

    config = {
        "dataset": {"seed": 42},
        "training": {"eval_fraction": 0.5},
    }
    train_ds, eval_ds = sft_train._prepare_sft_dataset(config)

    assert len(train_ds) == 10
    assert len(eval_ds) == 10


def test_sft_source_uses_vocab_sidecar_and_not_bigram() -> None:
    """SFT follows GRPO vocab metadata discipline without loading bigrams."""
    import inspect

    from src.training import sft_train

    source = inspect.getsource(sft_train.run_sft)
    assert "cache_is_current" in source
    assert "write_cache_meta" in source
    assert "load_transition_matrix" not in source
    assert "compute_bigram_transitions" not in source


# ---------------------------------------------------------------------------
# 2. Seeded eval holdout split
# ---------------------------------------------------------------------------


def test_eval_holdout_reproducible_and_disjoint() -> None:
    """Same seed → identical split; train and eval subsets never overlap."""
    ds = _make_sft_dataset(n=100)

    tr1, ev1 = split_eval_holdout(ds, eval_fraction=0.1, seed=42)
    tr2, ev2 = split_eval_holdout(ds, eval_fraction=0.1, seed=42)

    assert len(tr1) == 90
    assert len(ev1) == 10
    assert [r["sample_id"] for r in tr1] == [r["sample_id"] for r in tr2]
    assert [r["sample_id"] for r in ev1] == [r["sample_id"] for r in ev2]

    tr_ids = {r["sample_id"] for r in tr1}
    ev_ids = {r["sample_id"] for r in ev1}
    all_ids = {
        hashlib.sha256(
            f"Sentence number {i} for the split test.".encode("utf-8")
        ).hexdigest()
        for i in range(100)
    }
    assert tr_ids.isdisjoint(ev_ids)
    assert tr_ids | ev_ids == all_ids


def test_eval_holdout_differs_across_seeds() -> None:
    """A different seed yields a different (valid) partition."""
    ds = _make_sft_dataset(n=100)

    _, ev1 = split_eval_holdout(ds, eval_fraction=0.1, seed=42)
    _, ev2 = split_eval_holdout(ds, eval_fraction=0.1, seed=7)

    assert [r["sample_id"] for r in ev1] != [r["sample_id"] for r in ev2]
    assert len(ev1) == len(ev2) == 10


def test_eval_holdout_zero_fraction() -> None:
    """``eval_fraction <= 0`` disables the holdout (empty eval set)."""
    ds = _make_sft_dataset(n=5)

    tr, ev = split_eval_holdout(ds, eval_fraction=0.0, seed=42)

    assert len(tr) == 5
    assert len(ev) == 0


# ---------------------------------------------------------------------------
# 3. SFT YAML config fields
# ---------------------------------------------------------------------------


def test_sft_yaml_exposes_eval_keys() -> None:
    """The standalone SFT config carries the new eval/early-stopping keys."""
    cfg = yaml.safe_load(SFT_CONFIG_PATH.read_text(encoding="utf-8"))
    training = cfg["training"]

    assert training["eval_fraction"] == 0.02
    assert training["eval_steps"] == 200
    assert training["early_stopping_patience"] == 3
    assert training["per_device_eval_batch_size"] == 8
    assert training["save_total_limit"] == 1
    assert training["num_train_epochs"] == 3


# ---------------------------------------------------------------------------
# 4. SFT adapter fingerprint + reuse
# ---------------------------------------------------------------------------


def _sft_fingerprint_config(**training_overrides: object) -> dict:
    """A minimal SFT config dict (GRPO-flow shape) for fingerprint tests."""
    cfg = {
        "model": {
            "name": "Qwen/Qwen2.5-0.5B-Instruct",
            "quantization": "4bit",
            "dtype": "bfloat16",
            "use_unsloth": True,
        },
        "lora": {
            "r": 32,
            "lora_alpha": 64,
            "lora_dropout": 0.05,
            "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
            "random_state": 3407,
        },
        "dataset": {
            "dataset_name": "achrafothman/aslg_pc12",
            "seed": 42,
            "split": "train",
            "max_samples": None,
            "thinking": False,
        },
        "sft_pretrain": {
            "enabled": True,
            "training": {
                "num_train_epochs": 1,
                "per_device_train_batch_size": 4,
                "gradient_accumulation_steps": 4,
                "learning_rate": 2.0e-5,
                "lr_scheduler_type": "cosine",
                "warmup_steps": 50,
                "optim": "paged_adamw_8bit",
                "weight_decay": 0.1,
                "max_grad_norm": 1.0,
                "bf16": True,
                "max_seq_length": 768,
                "gradient_checkpointing": True,
                "eval_fraction": 0.02,
                "eval_steps": 200,
                "early_stopping_patience": 3,
            },
        },
        "training": {
            "output_dir": "experiments/checkpoints/qwen25-05b/sft-grpo/zero-shot",
            "log_dir": "experiments/logs/qwen25-05b/sft-grpo/zero-shot",
            "run_timestamp": "20260714_233901",
            "max_steps": 2000,
        },
    }
    cfg["sft_pretrain"]["training"].update(training_overrides)
    return cfg


def test_compute_sft_fingerprint_deterministic() -> None:
    """Same config → same fingerprint (canonical JSON, sort_keys)."""
    cfg = _sft_fingerprint_config()
    assert compute_sft_fingerprint(cfg) == compute_sft_fingerprint(cfg)


def test_compute_sft_fingerprint_ignores_paths_and_timestamps() -> None:
    """output_dir/log_dir/run_timestamp never affect the adapter weights."""
    cfg = _sft_fingerprint_config()
    other = _sft_fingerprint_config()
    other["training"]["output_dir"] = "experiments/checkpoints/other-model"
    other["training"]["log_dir"] = "experiments/logs/other-model"
    other["training"]["run_timestamp"] = "20260720_120000"
    assert compute_sft_fingerprint(cfg) == compute_sft_fingerprint(other)


def test_compute_sft_fingerprint_ignores_grpo_only_training_keys() -> None:
    """GRPO hyperparams (e.g. max_steps) must NOT invalidate the SFT adapter."""
    cfg = _sft_fingerprint_config()
    other = _sft_fingerprint_config()
    other["training"]["max_steps"] = 3000
    assert compute_sft_fingerprint(cfg) == compute_sft_fingerprint(other)


def test_compute_sft_fingerprint_sensitive_to_sft_hyperparams() -> None:
    """Changing any SFT hyperparameter changes the fingerprint."""
    for key, value in [
        ("learning_rate", 3.0e-5),
        ("num_train_epochs", 2),
        ("warmup_steps", 100),
        ("max_seq_length", 1024),
        ("per_device_train_batch_size", 8),
    ]:
        cfg = _sft_fingerprint_config()
        other = _sft_fingerprint_config(**{key: value})
        assert compute_sft_fingerprint(cfg) != compute_sft_fingerprint(other), key


def test_compute_sft_fingerprint_sensitive_to_lora() -> None:
    """LoRA shape (r) changes the fingerprint."""
    cfg = _sft_fingerprint_config()
    other = _sft_fingerprint_config()
    other["lora"]["r"] = 16
    assert compute_sft_fingerprint(cfg) != compute_sft_fingerprint(other)


def test_compute_sft_fingerprint_sensitive_to_dataset_and_model() -> None:
    """Dataset seed and model name changes invalidate the fingerprint."""
    cfg = _sft_fingerprint_config()

    other = _sft_fingerprint_config()
    other["dataset"]["seed"] = 7
    assert compute_sft_fingerprint(cfg) != compute_sft_fingerprint(other)

    other = _sft_fingerprint_config()
    other["model"]["name"] = "Qwen/Qwen2.5-1.5B-Instruct"
    assert compute_sft_fingerprint(cfg) != compute_sft_fingerprint(other)


def test_compute_sft_fingerprint_sensitive_to_system_prompt(monkeypatch) -> None:
    """A changed SYSTEM_PROMPT invalidates the adapter."""
    from src.training import sft_train

    cfg = _sft_fingerprint_config()
    fp_before = compute_sft_fingerprint(cfg)
    monkeypatch.setattr(sft_train, "SYSTEM_PROMPT", "A completely different prompt.")
    assert compute_sft_fingerprint(cfg) != fp_before


def test_compute_sft_fingerprint_standalone_sft_flow() -> None:
    """Without a sft_pretrain section the effective training is fingerprinted."""

    def standalone(output_dir: str) -> dict:
        return {
            "model": {
                "name": "Qwen/Qwen2.5-0.5B-Instruct",
                "quantization": "4bit",
                "dtype": "bfloat16",
            },
            "lora": {"r": 16},
            "dataset": {"dataset_name": "achrafothman/aslg_pc12", "seed": 42},
            "training": {
                "output_dir": output_dir,
                "log_dir": output_dir.replace("checkpoints", "logs"),
                "num_train_epochs": 3,
                "learning_rate": 2.0e-5,
            },
        }

    a = standalone("experiments/checkpoints/x")
    b = standalone("experiments/checkpoints/y")
    assert compute_sft_fingerprint(a) == compute_sft_fingerprint(b)
    b["training"]["learning_rate"] = 1.0e-5
    assert compute_sft_fingerprint(a) != compute_sft_fingerprint(b)


def test_write_sft_fingerprint(tmp_path) -> None:
    """``write_sft_fingerprint`` writes the expected JSON document."""
    cfg = _sft_fingerprint_config()
    final_path = tmp_path / "final"
    out = write_sft_fingerprint(final_path, cfg)

    assert out == final_path / "sft_fingerprint.json"
    assert out.is_file()
    doc = json.loads(out.read_text(encoding="utf-8"))
    assert doc["fingerprint"] == compute_sft_fingerprint(cfg)
    assert doc["config"]["version"] == 1
    assert doc["config"]["model"]["name"] == cfg["model"]["name"]
    assert (
        doc["config"]["sft_training"]["learning_rate"]
        == cfg["sft_pretrain"]["training"]["learning_rate"]
    )
    assert "output_dir" not in doc["config"]["sft_training"]


def test_is_complete_adapter_dir(tmp_path) -> None:
    """Adapter dirs need a config file AND weights to be reusable."""
    good = tmp_path / "good"
    good.mkdir()
    (good / "adapter_config.json").write_text("{}", encoding="utf-8")
    (good / "adapter_model.safetensors").write_bytes(b"weights")
    assert is_complete_adapter_dir(good)

    empty = tmp_path / "empty"
    empty.mkdir()
    assert not is_complete_adapter_dir(empty)

    meta_only = tmp_path / "meta_only"
    meta_only.mkdir()
    (meta_only / "sft_fingerprint.json").write_text("{}", encoding="utf-8")
    assert not is_complete_adapter_dir(meta_only)

    assert not is_complete_adapter_dir(tmp_path / "missing")


def _make_adapter_run(
    root: Path, name: str, fingerprint: str, *, complete: bool = True, mtime=None
) -> Path:
    """Create ``<root>/<name>/sft_pretrain/final`` with fingerprint + adapter."""
    final = root / name / "sft_pretrain" / "final"
    final.mkdir(parents=True, exist_ok=True)
    (final / "sft_fingerprint.json").write_text(
        json.dumps({"fingerprint": fingerprint, "created_at": "2026-01-01T00:00:00"}),
        encoding="utf-8",
    )
    if complete:
        (final / "adapter_config.json").write_text("{}", encoding="utf-8")
        (final / "adapter_model.safetensors").write_bytes(b"weights")
    if mtime is not None:
        stamp = mtime.timestamp()
        os.utime(final / "sft_fingerprint.json", (stamp, stamp))
    return final


def _make_standalone_adapter_run(
    model_root: Path, name: str, fingerprint: str, *, complete: bool = True, mtime=None
) -> Path:
    """Create canonical ``sft/zero-shot/<run>/final`` adapter output."""
    final = model_root / "sft" / "zero-shot" / name / "final"
    final.mkdir(parents=True, exist_ok=True)
    metadata = final / "sft_fingerprint.json"
    metadata.write_text(json.dumps({"fingerprint": fingerprint}), encoding="utf-8")
    if complete:
        (final / "adapter_config.json").write_text("{}", encoding="utf-8")
        (final / "adapter_model.safetensors").write_bytes(b"standalone-weights")
    if mtime is not None:
        stamp = mtime.timestamp()
        os.utime(metadata, (stamp, stamp))
    return final


# --- Cross-cell reuse ------------------------------------------------------


def test_find_reusable_sft_adapter_cross_method_finds_standalone(tmp_path) -> None:
    """Canonical standalone SFT output is reusable by an SFT-GRPO run."""
    model_root = tmp_path / "checkpoints" / "qwen25-05b"
    fp = compute_sft_fingerprint(_sft_fingerprint_config())
    adapter = _make_standalone_adapter_run(model_root, "run_1", fp)
    current = model_root / "sft-grpo" / "few-shot" / "run_2"
    current.mkdir(parents=True)

    found = find_reusable_sft_adapter_cross_method(model_root, current, fp)
    assert found == (adapter, "sft/zero-shot")


def test_find_reusable_sft_adapter_cross_method_finds_subphase(tmp_path) -> None:
    """Canonical SFT-GRPO subphase output is also reusable."""
    model_root = tmp_path / "checkpoints" / "qwen25-05b"
    fp = compute_sft_fingerprint(_sft_fingerprint_config())
    source = model_root / "sft-grpo" / "zero-shot"
    adapter = _make_adapter_run(source, "run_1", fp)
    current = model_root / "sft-grpo" / "few-shot" / "run_2"
    current.mkdir(parents=True)

    found = find_reusable_sft_adapter_cross_method(model_root, current, fp)
    assert found == (adapter, "sft-grpo/zero-shot")


def test_find_reusable_sft_adapter_cross_method_excludes_current(tmp_path) -> None:
    """The current run is excluded from canonical cross-method discovery."""
    model_root = tmp_path / "checkpoints" / "qwen25-05b"
    fp = compute_sft_fingerprint(_sft_fingerprint_config())
    current = model_root / "sft-grpo" / "zero-shot" / "run_1"
    _make_adapter_run(current.parent, current.name, fp)

    assert find_reusable_sft_adapter_cross_method(model_root, current, fp) is None


def test_find_reusable_sft_adapter_cross_method_no_match(tmp_path) -> None:
    """Mismatched and incomplete canonical adapters are ignored."""
    model_root = tmp_path / "checkpoints" / "qwen25-05b"
    fp = compute_sft_fingerprint(_sft_fingerprint_config())
    _make_standalone_adapter_run(model_root, "run_1", "deadbeef")
    _make_adapter_run(
        model_root / "sft-grpo" / "zero-shot", "run_2", fp, complete=False
    )
    current = model_root / "sft-grpo" / "few-shot" / "run_3"
    current.mkdir(parents=True)
    assert find_reusable_sft_adapter_cross_method(model_root, current, fp) is None
    assert (
        find_reusable_sft_adapter_cross_method(model_root / "missing", current, fp)
        is None
    )


def test_find_reusable_sft_adapter_cross_method_prefers_newest(tmp_path) -> None:
    """Newest canonical match wins across standalone and subphase layouts."""
    from datetime import datetime

    model_root = tmp_path / "checkpoints" / "qwen25-05b"
    fp = compute_sft_fingerprint(_sft_fingerprint_config())
    old = _make_standalone_adapter_run(
        model_root, "run_1", fp, mtime=datetime(2024, 1, 1)
    )
    new = _make_adapter_run(
        model_root / "sft-grpo" / "zero-shot",
        "run_2",
        fp,
        mtime=datetime(2026, 6, 1),
    )
    current = model_root / "sft-grpo" / "few-shot" / "run_3"
    current.mkdir(parents=True)

    found = find_reusable_sft_adapter_cross_method(model_root, current, fp)
    assert found == (new, "sft-grpo/zero-shot")
    assert found != (old, "sft/zero-shot")


def test_find_reusable_sft_adapter_cross_method_stable_path_tiebreak(tmp_path) -> None:
    """Equal mtimes are resolved deterministically by canonical path."""
    from datetime import datetime

    model_root = tmp_path / "checkpoints" / "qwen25-05b"
    fp = compute_sft_fingerprint(_sft_fingerprint_config())
    stamp = datetime(2026, 6, 1)
    standalone = _make_standalone_adapter_run(model_root, "run_1", fp, mtime=stamp)
    _make_adapter_run(model_root / "sft-grpo" / "zero-shot", "run_1", fp, mtime=stamp)
    current = model_root / "sft-grpo" / "few-shot" / "run_2"
    current.mkdir(parents=True)

    expected = max(
        (standalone, "sft/zero-shot"),
        (
            model_root / "sft-grpo" / "zero-shot" / "run_1" / "sft_pretrain" / "final",
            "sft-grpo/zero-shot",
        ),
        key=lambda item: (item[0] / "sft_fingerprint.json").as_posix(),
    )
    assert find_reusable_sft_adapter_cross_method(model_root, current, fp) == expected


def test_cross_method_search_ignores_noncanonical_flat_paths(tmp_path) -> None:
    """Old flat or arbitrary recursive layouts are never considered."""
    model_root = tmp_path / "checkpoints" / "qwen25-05b"
    fp = compute_sft_fingerprint(_sft_fingerprint_config())
    _make_adapter_run(model_root / "legacy-flat-cell", "run_1", fp)
    current = model_root / "sft-grpo" / "few-shot" / "run_2"
    current.mkdir(parents=True)

    assert find_reusable_sft_adapter_cross_method(model_root, current, fp) is None


def test_resolve_sft_run_paths_standalone_canonical(monkeypatch) -> None:
    """Standalone config without path keys resolves through canonical paths."""
    cfg = yaml.safe_load(SFT_CONFIG_PATH.read_text(encoding="utf-8"))
    cfg["model"] = {"name": "Qwen/Qwen2.5-0.5B-Instruct"}
    expected_output = Path("experiments/checkpoints/qwen25-05b/sft/zero-shot/run_fixed")
    expected_log = Path("experiments/logs/qwen25-05b/sft/zero-shot/run_fixed")

    def fake_training_run_paths(config, *, resume=False):
        from src.utils.paths import cell_from_config

        assert "output_dir" not in config["training"]
        assert "log_dir" not in config["training"]
        assert resume is False
        return expected_output, expected_log, "run_fixed", cell_from_config(config)

    monkeypatch.setattr(
        "src.training.sft_train.training_run_paths", fake_training_run_paths
    )
    output, log, timestamp, cell = resolve_sft_run_paths(cfg)

    assert (output, log, timestamp) == (expected_output, expected_log, "fixed")
    assert (cell.method, cell.train_prompt_mode) == ("sft", "zero-shot")


def test_resolve_sft_run_paths_preserves_grpo_subphase(monkeypatch) -> None:
    """Explicit SFT-GRPO subphase output and log directories are unchanged."""
    cfg = _sft_fingerprint_config()
    output = Path(
        "experiments/checkpoints/qwen25-05b/sft-grpo/few-shot/"
        "run_20260906_130000/sft_pretrain"
    )
    log = Path(
        "experiments/logs/qwen25-05b/sft-grpo/few-shot/"
        "run_20260906_130000/sft_pretrain"
    )
    cfg["experiment"] = {
        "model_tag": "qwen25-05b",
        "method": "sft-grpo",
        "train_prompt_mode": "few-shot",
        "variant": "none",
        "kind": "train",
    }
    cfg["training"].update(output_dir=str(output), log_dir=str(log))
    monkeypatch.setattr(
        "src.training.sft_train.training_run_paths",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("canonical standalone resolver must not run")
        ),
    )

    resolved_output, resolved_log, timestamp, cell = resolve_sft_run_paths(cfg)
    assert (resolved_output, resolved_log) == (output, log)
    assert timestamp == "20260906_130000"
    assert (cell.method, cell.train_prompt_mode) == ("sft-grpo", "few-shot")


def test_grpo_cli_force_sft_flag() -> None:
    """``--force-sft`` parses; other flags stay independent."""
    from src.training.grpo_t2g_train import build_arg_parser

    args = build_arg_parser().parse_args(["--config", "x.yaml", "--force-sft"])
    assert args.force_sft is True
    assert args.resume is False
    assert args.prepare_data is False

    args = build_arg_parser().parse_args(["--config", "x.yaml", "--resume"])
    assert args.force_sft is False
    assert args.resume is True
