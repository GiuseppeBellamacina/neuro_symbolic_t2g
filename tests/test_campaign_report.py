"""Test per src/analysis.campaign_report.

Coprono i casi limite dichiarati dal modulo: directory vuota, JSON
malformati/parziali, campioni discordanti, fattore senza coppia appaiata,
checkpoint incompleti, selezione del run più recente per tipologia e
flag di rumore sui delta. Il modulo LEGGE i JSON di eval: i test usano
fixture minime con le stesse chiavi, nessuna metrica viene ricalcolata.
"""

from __future__ import annotations

import json
from pathlib import Path

from src.analysis.campaign_report import (
    NOISE_THRESHOLD,
    build_comparisons,
    build_json_report,
    build_markdown,
    deduce_cell_factors,
    discover_runs,
    select_latest,
)


def make_eval(
    root: Path,
    cell: str,
    run_id: str,
    filename: str,
    payload: dict | None = None,
    raw: str | None = None,
) -> Path:
    """Crea un eval_*.json fittizio con il layout reale <cella>/<run>/."""
    d = root / cell / run_id
    d.mkdir(parents=True, exist_ok=True)
    p = d / filename
    if raw is not None:
        p.write_text(raw, encoding="utf-8")
    else:
        base = {
            "rouge_l_mean": 0.5,
            "exact_match": 0.1,
            "pass_at_1": 0.3,
            "validity_rate": 0.9,
            "num_samples_evaluated": 2000,
            "metrics_version": 2,
        }
        base.update(payload or {})
        p.write_text(json.dumps(base), encoding="utf-8")
    return p


# ---------------------------------------------------------------------------
# Deduzione dei fattori
# ---------------------------------------------------------------------------


def test_deduce_cell_factors_sft_grpo_variant():
    factors, _ = deduce_cell_factors("qwen25-05b-sft-grpo-soft-viterbi")
    assert factors["method"] == "sft-grpo"
    assert factors["variant"] == "soft-viterbi"
    # Non deducibili: dichiarati, mai inventati.
    assert factors["prompting"] == "unknown"
    assert factors["grammar"] == "unknown"
    assert factors["reward_stack"] == "unknown"
    assert factors["rl_objective"] == "unknown"


def test_deduce_cell_factors_no_grammar_is_grammar_marker_not_variant():
    factors, sources = deduce_cell_factors("qwen25-05b-sft-grpo-no-grammar")
    assert factors["method"] == "sft-grpo"
    assert factors["grammar"] == "off"
    assert factors["variant"] == ""


def test_deduce_cell_factors_baseline_family():
    f1, _ = deduce_cell_factors("t2g-zero-shot")
    assert f1["method"] == "baseline"
    assert f1["prompting"] == "zero-shot"
    assert f1["grammar"] == "off"
    f2, _ = deduce_cell_factors("t2g-zero-shot-grammar")
    assert f2["grammar"] == "on"
    assert f2["prompting"] == "zero-shot"


def test_deduce_cell_factors_unknown_method_keeps_full_cell_name():
    # Due celle senza marcatore noto non devono collassare nella stessa
    # tipologia: il nome intero entra nella variante.
    f1, _ = deduce_cell_factors("mystery-cell-a")
    f2, _ = deduce_cell_factors("mystery-cell-b")
    assert f1["method"] == "unknown"
    assert f1["variant"] != f2["variant"]


# ---------------------------------------------------------------------------
# Scoperta run
# ---------------------------------------------------------------------------


def test_empty_results_dir(tmp_path):
    result = discover_runs(tmp_path)
    assert result == {"runs": [], "excluded": [], "malformed": []}
    selected, superseded = select_latest(result["runs"])
    assert selected == [] and superseded == []
    comparisons, missing, _ = build_comparisons(selected)
    assert comparisons == []
    # Nessuna coppia appaiata: dichiarato per OGNI fattore, mai fabbricato.
    assert {m["factor"] for m in missing} == {"method", "prompting", "grammar"}
    md = build_markdown(selected, [], [], comparisons, missing, [], [])
    assert "No eval runs found" in md


def test_missing_results_dir(tmp_path):
    result = discover_runs(tmp_path / "does-not-exist")
    assert result["runs"] == []


def test_malformed_json_reported_not_raised(tmp_path):
    make_eval(
        tmp_path,
        "t2g-zero-shot",
        "zero_shot_20260101_000000",
        "eval_zero_shot.json",
        raw="{not valid json,,,",
    )
    make_eval(
        tmp_path,
        "qwen25-05b-sft-only",
        "run_20260102_000000",
        "eval_final.json",
        {"prompting": {"mode": "zero-shot", "source": "config"}},
    )
    result = discover_runs(tmp_path)
    assert len(result["runs"]) == 1
    assert len(result["malformed"]) == 1
    assert "eval_zero_shot" in result["malformed"][0]["path"]
    # Il run interrotto è segnalato anche nel Markdown, non ignorato.
    md = build_markdown(
        select_latest(result["runs"])[0], [], result["malformed"], [], [], [], []
    )
    assert "malformed/partial JSON" in md


def test_checkpoint_incomplete_excluded(tmp_path):
    make_eval(
        tmp_path,
        "qwen25-05b-sft-only",
        "run_20260101_000000",
        "eval_final.json",
        {"checkpoint_incomplete": True, "checkpoint_step": 500},
    )
    make_eval(tmp_path, "qwen25-05b-sft-only", "run_20260102_000000", "eval_final.json")
    result = discover_runs(tmp_path)
    assert len(result["runs"]) == 1
    assert len(result["excluded"]) == 1
    selected, _ = select_latest(result["runs"])
    comparisons, missing, _ = build_comparisons(selected)
    # Un solo run per fattore: nessuna coppia appaiata, dichiarato.
    assert comparisons == []
    assert {m["factor"] for m in missing} == {"method", "prompting", "grammar"}


def test_latest_per_typology_selected(tmp_path):
    make_eval(
        tmp_path,
        "qwen25-05b-sft-only",
        "run_20260101_000000",
        "eval_final.json",
        {"rouge_l_mean": 0.1},
    )
    make_eval(
        tmp_path,
        "qwen25-05b-sft-only",
        "run_20260301_000000",
        "eval_final.json",
        {"rouge_l_mean": 0.9},
    )
    result = discover_runs(tmp_path)
    selected, superseded = select_latest(result["runs"])
    assert len(selected) == 1
    assert selected[0]["path"].endswith("run_20260301_000000/eval_final.json")
    assert selected[0]["metrics"]["rouge_l_mean"] == 0.9
    assert len(superseded) == 1


def test_prompting_from_stamp_and_suffix(tmp_path):
    make_eval(
        tmp_path,
        "qwen25-05b-sft-grpo",
        "run_20260101_000000",
        "eval_final.json",
        {"prompting": {"mode": "few-shot", "source": "config"}},
    )
    make_eval(
        tmp_path,
        "qwen25-05b-sft-grpo",
        "run_20260102_000000",
        "eval_final__zero-shot.json",
        {"prompting": {"mode": "zero-shot", "source": "config-dual"}},
    )
    result = discover_runs(tmp_path)
    stamp_run, suffix_run = sorted(result["runs"], key=lambda r: r["timestamp"])
    assert stamp_run["factors"]["prompting"] == "few-shot"
    assert suffix_run["factors"]["prompting"] == "zero-shot"


# ---------------------------------------------------------------------------
# Confronti appaiati
# ---------------------------------------------------------------------------


def test_method_pair_same_run_baseline_vs_final(tmp_path):
    make_eval(
        tmp_path,
        "qwen25-05b-sft-only",
        "run_20260101_000000",
        "eval_baseline.json",
        {
            "prompting": {"mode": "zero-shot", "source": "config"},
            "prompt_context_fingerprint": "a" * 64,
            "rouge_l_mean": 0.40,
        },
    )
    make_eval(
        tmp_path,
        "qwen25-05b-sft-only",
        "run_20260101_000000",
        "eval_final.json",
        {
            "prompting": {"mode": "zero-shot", "source": "config"},
            "prompt_context_fingerprint": "a" * 64,
            "rouge_l_mean": 0.50,
        },
    )
    result = discover_runs(tmp_path)
    selected, _ = select_latest(result["runs"])
    comparisons, missing, _ = build_comparisons(selected)
    method_pairs = [c for c in comparisons if c["factor"] == "method"]
    assert len(method_pairs) == 1
    pair = method_pairs[0]
    assert pair["a"]["factors"]["method"] == "baseline"
    assert (
        pair["b"]["factors"]["method"] == "sft-only"
        or pair["b"]["factors"]["method"] == "sft"
    )
    # Delta letto, non ricalcolato: 0.50 - 0.40 = 0.10, sopra la soglia.
    d = pair["deltas"]["rouge_l_mean"]
    assert d["delta"] == 0.10 and "NOISE" not in d["interpretation"]


def test_dual_prompting_pair_highlighted(tmp_path):
    make_eval(
        tmp_path,
        "qwen25-05b-sft-grpo",
        "run_20260101_000000",
        "eval_final.json",
        {"prompting": {"mode": "few-shot", "source": "config"}},
    )
    make_eval(
        tmp_path,
        "qwen25-05b-sft-grpo",
        "run_20260101_000000",
        "eval_final__zero-shot.json",
        {"prompting": {"mode": "zero-shot", "source": "config-dual"}},
    )
    result = discover_runs(tmp_path)
    selected, _ = select_latest(result["runs"])
    comparisons, _, _ = build_comparisons(selected)
    prompting_pairs = [c for c in comparisons if c["factor"] == "prompting"]
    assert len(prompting_pairs) == 1
    pair = prompting_pairs[0]
    # Stessa cella, stessa run dir: il confronto chiave del dual prompting.
    assert "same checkpoint" in pair["kind"]
    assert any("HIGHLIGHT" in cv for cv in pair["caveats"])


def test_sample_size_mismatch_warning(tmp_path):
    make_eval(
        tmp_path,
        "qwen25-05b-sft-only",
        "run_20260101_000000",
        "eval_baseline.json",
        {
            "prompting": {"mode": "zero-shot", "source": "config"},
            "num_samples_evaluated": 2000,
        },
    )
    make_eval(
        tmp_path,
        "qwen25-05b-sft-only",
        "run_20260101_000000",
        "eval_final.json",
        {
            "prompting": {"mode": "zero-shot", "source": "config"},
            "num_samples_evaluated": 3000,
        },
    )
    result = discover_runs(tmp_path)
    selected, _ = select_latest(result["runs"])
    comparisons, _, warnings = build_comparisons(selected)
    assert comparisons, "pair must still be reported, with the warning"
    pair = comparisons[0]
    assert any("num_samples_evaluated DIFFERS" in cv for cv in pair["caveats"])
    assert any("NOT comparable" in w for w in warnings)
    md = build_markdown(selected, [], [], comparisons, [], warnings, [])
    assert "num_samples_evaluated DIFFERS" in md


def test_metrics_version_mismatch_flagged(tmp_path):
    make_eval(
        tmp_path,
        "qwen25-05b-sft-only",
        "run_20260101_000000",
        "eval_baseline.json",
        {"metrics_version": 1},
    )
    make_eval(
        tmp_path,
        "qwen25-05b-sft-only",
        "run_20260101_000000",
        "eval_final.json",
        {"metrics_version": 2},
    )
    result = discover_runs(tmp_path)
    selected, _ = select_latest(result["runs"])
    comparisons, _, warnings = build_comparisons(selected)
    assert any(
        "metrics_version DIFFERS" in cv for c in comparisons for cv in c["caveats"]
    )
    assert warnings


def test_noise_delta_flagged_not_presented_as_result(tmp_path):
    # Delta 0.003 < soglia 0.02: deve essere marcato rumore, non risultato.
    make_eval(
        tmp_path,
        "qwen25-05b-sft-only",
        "run_20260101_000000",
        "eval_baseline.json",
        {"rouge_l_mean": 0.500},
    )
    make_eval(
        tmp_path,
        "qwen25-05b-sft-only",
        "run_20260101_000000",
        "eval_final.json",
        {"rouge_l_mean": 0.503},
    )
    result = discover_runs(tmp_path)
    selected, _ = select_latest(result["runs"])
    comparisons, _, _ = build_comparisons(selected)
    d = comparisons[0]["deltas"]["rouge_l_mean"]
    assert abs(d["delta"]) == 0.003
    assert d["interpretation"].startswith("NOISE")
    assert str(NOISE_THRESHOLD) in d["interpretation"]


def test_missing_metric_yields_no_delta(tmp_path):
    # La metrica assente da un lato NON viene ricostruita con euristiche.
    make_eval(
        tmp_path,
        "qwen25-05b-sft-only",
        "run_20260101_000000",
        "eval_baseline.json",
        {"rouge_l_mean": 0.4},
    )
    make_eval(
        tmp_path,
        "qwen25-05b-sft-only",
        "run_20260101_000000",
        "eval_final.json",
        {"rouge_l_mean": 0.6},
    )
    result = discover_runs(tmp_path)
    selected, _ = select_latest(result["runs"])
    # exact_match manca nel baseline → nessun delta per exact_match.
    # (make_eval lo mette di default: rimuoviamolo da UNA parte.)
    baseline_run = next(r for r in result["runs"] if r["kind"] == "baseline")
    baseline_run["metrics"].pop("exact_match")
    comparisons, _, _ = build_comparisons(selected)
    method_pair = comparisons[0]
    assert "rouge_l_mean" in method_pair["deltas"]
    # ricostruiamo il confronto con i metrics modificati:
    from src.analysis.campaign_report import _make_comparison

    other = next(r for r in result["runs"] if r is not baseline_run)
    pair = _make_comparison("method", baseline_run, other, [])
    assert "exact_match" not in pair["deltas"]


def test_single_run_factor_declares_missing_pair(tmp_path):
    make_eval(tmp_path, "qwen25-05b-sft-only", "run_20260101_000000", "eval_final.json")
    result = discover_runs(tmp_path)
    selected, _ = select_latest(result["runs"])
    comparisons, missing, _ = build_comparisons(selected)
    assert comparisons == []
    for m in missing:
        # Dichiara il perché, con le tipologie disponibili.
        assert m["reason"]
        assert m["available_typologies"]


def test_json_report_shape(tmp_path):
    make_eval(tmp_path, "qwen25-05b-sft-only", "run_20260101_000000", "eval_final.json")
    result = discover_runs(tmp_path)
    selected, superseded = select_latest(result["runs"])
    comparisons, missing, warnings = build_comparisons(selected)
    report = build_json_report(
        selected, superseded, [], [], comparisons, missing, warnings
    )
    for key in (
        "factor_deduction",
        "noise_threshold",
        "runs",
        "comparisons",
        "missing_pairs",
        "warnings",
        "malformed_files",
        "excluded_runs",
        "superseded_runs",
    ):
        assert key in report
    assert report["noise_threshold"]["value"] == NOISE_THRESHOLD
    # La dichiarazione della deduzione è nel JSON: niente deduzione implicita.
    assert {d["factor"] for d in report["factor_deduction"]} >= {
        "method",
        "prompting",
        "grammar",
        "reward_stack",
        "rl_objective",
    }
