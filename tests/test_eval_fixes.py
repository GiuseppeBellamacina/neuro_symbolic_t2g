"""Tests for the eval-fix package (slurm-eval-7077 crash + friends).

Covers:
- resolve_model_source: offline-first hub-id → local-snapshot resolution
  (transformers 5.3 tokenizer init calls model_info() for non-local ids —
  on DNS-less nodes this crashed eval job 7077).
- _load_cached_baseline: compare-mode baseline reuse compatibility checks
  (metrics_version, decoding, sample count).
- dedupe_library_loggers: HF libraries double-print warnings via their own
  handler + root propagation (slurm-eval-7077/7078).
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.model_loader import resolve_model_source  # noqa: E402
from src.utils.log_dedup import dedupe_library_loggers  # noqa: E402

# ---------------------------------------------------------------------------
# resolve_model_source
# ---------------------------------------------------------------------------


def test_resolve_local_path_passthrough(tmp_path):
    """An existing local path/_dir is returned unchanged (no hub lookup)."""
    local_model = tmp_path / "my_model"
    local_model.mkdir()
    assert resolve_model_source(str(local_model)) == str(local_model)


def test_resolve_hub_id_cached(monkeypatch):
    """A cached hub id resolves to the local snapshot path."""

    def fake_snapshot(repo_id, local_files_only):  # noqa: ARG001
        assert local_files_only is True
        return "/hf/cache/models--Qwen--Qwen2.5-0.5B-Instruct/snapshot/abc"

    import src.models.model_loader as ml

    monkeypatch.setattr(
        "huggingface_hub.snapshot_download", fake_snapshot, raising=False
    )
    # The helper imports snapshot_download inside the function from
    # huggingface_hub, so patching the module attr is enough.
    out = resolve_model_source("Qwen/Qwen2.5-0.5B-Instruct")
    assert out == "/hf/cache/models--Qwen--Qwen2.5-0.5B-Instruct/snapshot/abc"
    assert ml is not None  # silence linters


def test_resolve_hub_id_not_cached(monkeypatch):
    """An uncached hub id falls back to the original id (network allowed)."""

    def fake_snapshot(repo_id, local_files_only):  # noqa: ARG001
        raise FileNotFoundError("not in local cache")

    monkeypatch.setattr(
        "huggingface_hub.snapshot_download", fake_snapshot, raising=False
    )
    assert resolve_model_source("Qwen/Qwen2.5-0.5B-Instruct") == (
        "Qwen/Qwen2.5-0.5B-Instruct"
    )


# ---------------------------------------------------------------------------
# _load_cached_baseline
# ---------------------------------------------------------------------------


def _write_baseline(
    results_dir: Path,
    *,
    metrics_version: int | None = 2,
    num_completions: int = 5,
    n_eval: int = 500,
    test_set_size: int = 8109,
    fingerprint: str | None = None,
    with_generations: bool = True,
) -> None:
    baseline: dict[str, object] = {
        "num_completions_per_prompt": num_completions,
        "num_samples_evaluated": n_eval,
        "test_set_size": test_set_size,
    }
    if metrics_version is not None:
        baseline["metrics_version"] = metrics_version
    if fingerprint is not None:
        baseline["prompt_context_fingerprint"] = fingerprint
    (results_dir / "eval_baseline.json").write_text(
        json.dumps(baseline), encoding="utf-8"
    )
    if with_generations:
        (results_dir / "generations_baseline.json").write_text(
            '[{"index": 0, "text": "t", "gold_gloss": "G", "completion": "C"}]',
            encoding="utf-8",
        )


def _import_helpers():
    from src.training.eval_t2g import _load_cached_baseline, _prompt_context_fingerprint

    return _load_cached_baseline, _prompt_context_fingerprint


_MIN_CFG = {
    "model": {"name": "Qwen/Qwen2.5-0.5B-Instruct"},
    "dataset": {"dataset_name": "achrafothman/aslg_pc12", "seed": 42},
}


def _fingerprint(cfg: dict | None = None) -> str:
    _, fp_fn = _import_helpers()
    return fp_fn(cfg or _MIN_CFG, num_samples=5)


def test_cached_baseline_compatible(tmp_path):
    _write_baseline(tmp_path, fingerprint=_fingerprint())
    load, _ = _import_helpers()
    out = load(tmp_path, num_samples=5, max_samples=500, fingerprint=_fingerprint())
    assert out is not None
    baseline, generations, source = out
    assert baseline["num_samples_evaluated"] == 500
    assert generations is not None and generations[0]["text"] == "t"
    assert source == tmp_path


def test_cached_baseline_rejects_old_metrics(tmp_path):
    """Baseline computed with older metric definitions must be recomputed."""
    _write_baseline(tmp_path, metrics_version=1, fingerprint=_fingerprint())
    load, _ = _import_helpers()
    assert (
        load(tmp_path, num_samples=5, max_samples=500, fingerprint=_fingerprint())
        is None
    )


def test_cached_baseline_rejects_missing_version(tmp_path):
    """Pre-corpus-fix eval_baseline.json has no stamp → recompute."""
    _write_baseline(tmp_path, metrics_version=None, fingerprint=_fingerprint())
    load, _ = _import_helpers()
    assert (
        load(tmp_path, num_samples=5, max_samples=500, fingerprint=_fingerprint())
        is None
    )


def test_cached_baseline_rejects_prompt_context_change(tmp_path):
    """A baseline evaluated with different retrieval/grammar/system prompt
    (different fingerprint) must NOT be reused — generations would differ."""
    _write_baseline(tmp_path, fingerprint="old-fingerprint-not-matching")
    load, _ = _import_helpers()
    assert (
        load(tmp_path, num_samples=5, max_samples=500, fingerprint=_fingerprint())
        is None
    )


def test_cached_baseline_rejects_decoding_change(tmp_path):
    _write_baseline(tmp_path, num_completions=1, fingerprint=_fingerprint())
    load, _ = _import_helpers()
    assert (
        load(tmp_path, num_samples=5, max_samples=500, fingerprint=_fingerprint())
        is None
    )


def test_cached_baseline_rejects_sample_count_change(tmp_path):
    _write_baseline(tmp_path, n_eval=500, fingerprint=_fingerprint())
    load, _ = _import_helpers()
    assert (
        load(tmp_path, num_samples=5, max_samples=100, fingerprint=_fingerprint())
        is None
    )


def test_cached_baseline_full_set_ok(tmp_path):
    """max_samples=None (full test set) reuses a cached FULL-set baseline."""
    _write_baseline(
        tmp_path, n_eval=8109, test_set_size=8109, fingerprint=_fingerprint()
    )
    load, _ = _import_helpers()
    out = load(tmp_path, num_samples=5, max_samples=None, fingerprint=_fingerprint())
    assert out is not None


def test_cached_baseline_full_set_rejects_subset(tmp_path):
    """max_samples=None must NOT reuse a 500-sample cached baseline."""
    _write_baseline(
        tmp_path, n_eval=500, test_set_size=8109, fingerprint=_fingerprint()
    )
    load, _ = _import_helpers()
    assert (
        load(tmp_path, num_samples=5, max_samples=None, fingerprint=_fingerprint())
        is None
    )


def test_cached_baseline_absent(tmp_path):
    load, _ = _import_helpers()
    assert (
        load(tmp_path, num_samples=5, max_samples=500, fingerprint=_fingerprint())
        is None
    )


def test_cached_baseline_sibling_run_reuse(tmp_path):
    """A NEW run dir with no baseline reuses a sibling run's baseline
    (same model tag): round-2 GRPO evals stop re-paying ~28 GPU-min."""
    tag_dir = tmp_path / "qwen25-05b-sft-grpo"
    run1 = tag_dir / "run_20260829_120124"
    run2 = tag_dir / "run_20260830_100000"
    run1.mkdir(parents=True)
    run2.mkdir(parents=True)
    _write_baseline(run1, fingerprint=_fingerprint())

    load, _ = _import_helpers()
    out = load(run2, num_samples=5, max_samples=500, fingerprint=_fingerprint())
    assert out is not None
    baseline, generations, source = out
    assert source == run1  # found in the sibling, not the (empty) current dir
    assert generations is not None


def test_cached_baseline_cross_tag_reuse(tmp_path):
    """A NEW model tag (e.g. sft-grpo-*) reuses the baseline cached by
    a DIFFERENT tag (sft-grpo): ablation cells share the same zero-shot
    base model + prompt context — ~28 GPU-min saved per tag."""
    results = tmp_path / "experiments" / "results"
    optimal_run = results / "qwen25-05b-sft-grpo" / "run_20260829_120124"
    ablation_run = results / "qwen25-05b-sft-grpo-structure" / "run_20260830_120000"
    optimal_run.mkdir(parents=True)
    ablation_run.mkdir(parents=True)
    _write_baseline(optimal_run, fingerprint=_fingerprint())

    load, _ = _import_helpers()
    out = load(ablation_run, num_samples=5, max_samples=500, fingerprint=_fingerprint())
    assert out is not None
    baseline, generations, source = out
    assert source == optimal_run


def test_cached_baseline_cross_tag_rejects_stale_context(tmp_path):
    """Cross-tag reuse must NOT fire when the prompt context differs
    (e.g. a tag trained with retrieval off vs baseline with retrieval on)."""
    results = tmp_path / "experiments" / "results"
    optimal_run = results / "qwen25-05b-sft-grpo" / "run_20260829_120124"
    other_run = results / "qwen25-05b-other" / "run_20260830_120000"
    optimal_run.mkdir(parents=True)
    other_run.mkdir(parents=True)
    _write_baseline(optimal_run, fingerprint="different-prompt-context")

    load, _ = _import_helpers()
    assert (
        load(other_run, num_samples=5, max_samples=500, fingerprint=_fingerprint())
        is None
    )


def test_cached_baseline_sibling_rejects_stale_context(tmp_path):
    """A sibling baseline evaluated with a DIFFERENT prompt context
    (e.g. retrieval toggled between rounds) must not be reused."""
    tag_dir = tmp_path / "qwen25-05b-sft-grpo"
    run1 = tag_dir / "run_20260829_120124"
    run2 = tag_dir / "run_20260830_100000"
    run1.mkdir(parents=True)
    run2.mkdir(parents=True)
    _write_baseline(run1, fingerprint="different-context")

    load, _ = _import_helpers()
    assert (
        load(run2, num_samples=5, max_samples=500, fingerprint=_fingerprint()) is None
    )


def test_prompt_context_fingerprint_stable_and_sensitive():
    """Same context → same fingerprint; retrieval/grammar/model/decoding
    changes → different fingerprint."""
    _, fp_fn = _import_helpers()
    assert fp_fn(_MIN_CFG, 5) == fp_fn(_MIN_CFG, 5)

    changed = {**_MIN_CFG, "retrieval": {"enabled": True, "top_k": 3}}
    assert fp_fn(changed, 5) != fp_fn(_MIN_CFG, 5)

    grammar_off = {**_MIN_CFG, "grammar": {"enabled": False}}
    assert fp_fn(grammar_off, 5) != fp_fn(_MIN_CFG, 5)

    other_model = {
        **_MIN_CFG,
        "model": {"name": "Qwen/Qwen2.5-1.5B-Instruct"},
    }
    assert fp_fn(other_model, 5) != fp_fn(_MIN_CFG, 5)

    assert fp_fn(_MIN_CFG, num_samples=1) != fp_fn(_MIN_CFG, num_samples=5)


# ---------------------------------------------------------------------------
# dedupe_library_loggers
# ---------------------------------------------------------------------------


def test_dedupe_clears_library_handlers():
    """Library-owned handlers are stripped; records then print once (root)."""
    lg = logging.getLogger("datasets")
    handler = logging.StreamHandler()
    lg.addHandler(handler)
    try:
        assert lg.handlers  # precondition: duplicate emission
        dedupe_library_loggers()
        assert not lg.handlers
    finally:
        # cleanup (loggers are global)
        if handler in lg.handlers:
            lg.removeHandler(handler)


def test_dedupe_idempotent_no_handlers():
    """Loggers without handlers are untouched (no crash, no-op)."""
    before = logging.getLogger("huggingface_hub").handlers[:]
    dedupe_library_loggers()
    assert logging.getLogger("huggingface_hub").handlers == before
