"""Tests for the two Pass@1 estimators printed by eval_t2g and for the
prompting-mode override plumbing (Task: duplicate "Pass@1" label fix + dual
eval cache safety).

Covers:
- ``compute_pass_at_k`` (k=1): fraction of prompts whose FIRST completion
  reaches the ROUGE-L threshold (single honest draw).
- ``compute_evaluation_report`` ``pass_at_1``: empirical pass rate over ALL
  completions (every sampled completion of every prompt) with bootstrap CI.
- The two estimators MUST coincide when there is 1 completion per prompt
  (or when all completions of each prompt share the same pass status), and
  diverge deterministically otherwise.
- ``_compute_primary_metrics`` end-to-end: the JSON blocks carry the two
  distinct estimators (labels in the printed output are presentation-only).
- ``_prompt_context_fingerprint``: a ``--prompting`` override enters the
  payload ONLY when it changes the effective mode — default runs keep
  byte-identical fingerprints so existing cached baselines stay valid.
- ``_try_cached_baseline``: mode-suffixed baseline filenames are read/written
  separately, so a config-mode baseline and an override-mode baseline do not
  collide.

The pass/fail fixtures use single-token glosses ("PASS"/"FAIL" vs the same
reference) so ROUGE-L is exactly 1.0/0.0 and single-token completions skip
the bigram scoring path (needs >= 2 indices).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.metrics import (  # noqa: E402
    bootstrap_confidence_interval,
    compute_evaluation_report,
    compute_pass_at_k,
)

_REF = "PASS"  # single token: rouge_l("PASS", "PASS") == 1.0, other == 0.0


def _import_eval_helpers():
    from src.training.eval_t2g import (
        _compute_primary_metrics,
        _prompt_context_fingerprint,
        _try_cached_baseline,
    )

    return _compute_primary_metrics, _prompt_context_fingerprint, _try_cached_baseline


# ---------------------------------------------------------------------------
# Estimator semantics (pure metrics.py — no GPU, no heavy imports)
# ---------------------------------------------------------------------------


def test_pass_at_1_uses_first_completion_only():
    """pass@1 = fraction of prompts whose FIRST completion passes (k=1
    anchor of the Pass@k curve, which uses the first k completions)."""
    nested = [
        [_REF, "FAIL"],  # first FAILS
        ["FAIL", _REF],  # first PASSES
        ["FAIL", _REF],  # first FAILS
    ]
    refs = [_REF] * 3
    out = compute_pass_at_k(nested, refs, k_values=(1, 2), threshold=0.3)
    # Solo la prima completion di ciascun prompt conta per k=1.
    assert out["pass@1"] == 1 / 3
    # Con k=2 tutte le prompt hanno almeno un successo ("FAIL" vs "PASS"
    # ha rouge 0, ma la seconda completion è identica al reference).
    assert out["pass@2"] == 1.0


def test_report_pass_mean_averages_all_completions():
    """The CI-block pass@1 is the empirical mean over ALL completions,
    NOT the first-draw estimator."""
    completions = [_REF, "FAIL", "FAIL", _REF, "FAIL", _REF]
    refs = [_REF] * len(completions)
    report = compute_evaluation_report(completions, refs, n_bootstrap=20)
    assert report["pass_at_1"]["mean"] == 3 / 6
    # La CI bootstrap copre questa media: bounds attorno al valore medio.
    lo, hi = report["pass_at_1"]["ci_95"]
    assert 0.0 <= lo <= report["pass_at_1"]["mean"] <= hi <= 1.0


def test_bootstrap_ci_mean_is_plain_mean():
    """bootstrap_confidence_interval returns the plain empirical mean —
    this is what makes the report block an all-completions estimator."""
    values = [1.0, 0.0, 1.0, 1.0]
    mean, lo, hi = bootstrap_confidence_interval(values, n_bootstrap=10)
    assert mean == 0.75
    assert lo <= mean <= hi


# ---------------------------------------------------------------------------
# Coincidence limit cases + deterministic divergence (end-to-end, the pair
# actually printed by eval_t2g)
# ---------------------------------------------------------------------------


def _run_primary_metrics(nested, n_bootstrap=20):
    primary = _import_eval_helpers()[0]
    flat = [c for comps in nested for c in comps]
    refs = [_REF] * len(nested)
    flat_refs = [r for _ in nested for r in refs]
    results, *_rest = primary(
        flat,
        flat_refs,
        nested,
        refs,
        token_to_idx={},
        bigram=None,
        reward_weights={},
        n_bootstrap=n_bootstrap,
    )
    return results


def test_two_estimators_coincide_with_one_completion_per_prompt():
    """Case limite in cui DEVONO coincidere: num_samples = 1 — la "prima"
    completion e "tutte le completions" sono lo stesso insieme."""
    nested = [[_REF], ["FAIL"], [_REF], [_REF]]
    results = _run_primary_metrics(nested)
    assert results["pass_at_1"] == 3 / 4
    assert results["evaluation_report"]["pass_at_1"]["mean"] == results["pass_at_1"]


def test_two_estimators_coincide_with_homogeneous_prompts():
    """Secondo caso limite: quando tutte le completions di ogni prompt
    condividono lo stesso esito, i due stimatori coincidono anche con
    num_samples > 1."""
    nested = [
        [_REF, _REF, _REF],  # tutto passa
        ["FAIL", "FAIL", "FAIL"],  # tutto fallisce
        [_REF, _REF, _REF],
    ]
    results = _run_primary_metrics(nested)
    assert results["pass_at_1"] == 2 / 3
    assert results["evaluation_report"]["pass_at_1"]["mean"] == results["pass_at_1"]


def test_two_estimators_diverge_as_expected():
    """Caso generico: il primo draw per prompt e la media su tutti i draw
    sono stimatori diversi — qui la divergenza è deterministica e calcolata
    a mano (NOTA: nel repo NESSUNO dei due è lo stimatore combinatorio
    1 - C(n-c,k)/C(n,k); sono entrambe medie empiriche su insiemi diversi,
    docs/EVALUATION.md §2a)."""
    # 3 prompt x 2 completions: la PRIMA completion fallisce sempre, la
    # seconda passa sempre.
    nested = [["FAIL", _REF], ["FAIL", _REF], ["FAIL", _REF]]
    results = _run_primary_metrics(nested)
    # first-draw: 0 prompt su 3 hanno la prima completion che passa.
    assert results["pass_at_1"] == 0.0
    # all-completions: 3 successi su 6 completions.
    assert results["evaluation_report"]["pass_at_1"]["mean"] == 3 / 6

    # Specchio: la prima passa sempre, le altre falliscono.
    nested = [[_REF, "FAIL"], [_REF, "FAIL"], [_REF, "FAIL"]]
    results = _run_primary_metrics(nested)
    assert results["pass_at_1"] == 1.0
    assert results["evaluation_report"]["pass_at_1"]["mean"] == 3 / 6


# ---------------------------------------------------------------------------
# Prompting override: fingerprint + baseline cache filenames
# ---------------------------------------------------------------------------

_CFG_ZERO_SHOT = {
    "model": {"name": "Qwen/Qwen2.5-0.5B-Instruct"},
    "dataset": {"dataset_name": "achrafothman/aslg_pc12", "seed": 42},
    "retrieval": {"enabled": False},
}

_CFG_FEWSHOT = {
    "model": {"name": "Qwen/Qwen2.5-0.5B-Instruct"},
    "dataset": {"dataset_name": "achrafothman/aslg_pc12", "seed": 42},
    "retrieval": {"enabled": True, "top_k": 3},
}


def test_fingerprint_unchanged_for_default_and_redundant_override():
    """Default run e override ridondante (few-shot su config few-shot)
    generano le stesse completions: fingerprint identico, la cache esistente
    resta valida."""
    _, fp, _ = _import_eval_helpers()
    base = fp(_CFG_FEWSHOT, num_samples=5)
    assert fp(_CFG_FEWSHOT, num_samples=5, prompting=None) == base
    assert fp(_CFG_FEWSHOT, num_samples=5, prompting="config") == base
    assert fp(_CFG_FEWSHOT, num_samples=5, prompting="few-shot") == base

    base_zs = fp(_CFG_ZERO_SHOT, num_samples=5)
    assert fp(_CFG_ZERO_SHOT, num_samples=5, prompting="zero-shot") == base_zs


def test_fingerprint_changes_when_override_changes_mode():
    """Un override che CAMBIA la modalità deve invalidare la cache: è il bug
    latente reso raggiungibile dal flag --prompting (una baseline calcolata
    in un'altra modalità non è riutilizzabile)."""
    _, fp, _ = _import_eval_helpers()
    # few-shot forzato su una config zero-shot
    assert fp(_CFG_ZERO_SHOT, num_samples=5, prompting="few-shot") != fp(
        _CFG_ZERO_SHOT, num_samples=5
    )
    # zero-shot forzato su una config few-shot
    assert fp(_CFG_FEWSHOT, num_samples=5, prompting="zero-shot") != fp(
        _CFG_FEWSHOT, num_samples=5
    )


def test_cached_baseline_mode_suffixed_filename_isolated(tmp_path):
    """Baseline di default (eval_baseline.json) e baseline con override
    (eval_baseline__<mode>.json) non si sovrascrivono né si mescolano."""
    _, fp, try_cached = _import_eval_helpers()
    suffix_fp = fp(_CFG_ZERO_SHOT, num_samples=5, prompting="few-shot")

    # Solo la baseline di default esiste: il lookup con filename suffissato
    # non deve trovarla (il fingerprint inoltre non combacerebbe).
    (tmp_path / "eval_baseline.json").write_text(
        json.dumps(
            {
                "metrics_version": 2,
                "num_completions_per_prompt": 5,
                "num_samples_evaluated": 100,
                "test_set_size": 8109,
                "prompt_context_fingerprint": suffix_fp,
            }
        ),
        encoding="utf-8",
    )
    assert (
        try_cached(
            tmp_path,
            5,
            100,
            suffix_fp,
            filename="eval_baseline__few-shot.json",
        )
        is None
    )

    # Scritta la baseline suffissata, il lookup la trova con quel nome.
    (tmp_path / "eval_baseline__few-shot.json").write_text(
        json.dumps(
            {
                "metrics_version": 2,
                "num_completions_per_prompt": 5,
                "num_samples_evaluated": 100,
                "test_set_size": 8109,
                "prompt_context_fingerprint": suffix_fp,
            }
        ),
        encoding="utf-8",
    )
    out = try_cached(
        tmp_path, 5, 100, suffix_fp, filename="eval_baseline__few-shot.json"
    )
    assert out is not None
    baseline, _generations = out
    assert baseline["num_samples_evaluated"] == 100
