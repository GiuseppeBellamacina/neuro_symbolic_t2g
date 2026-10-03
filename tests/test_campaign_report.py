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
        "run_20260101_000000",
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
    assert d["interpretation"].startswith("NOT GENERALIZABLE")
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


def test_nested_cells_do_not_collapse_into_one_typology(tmp_path):
    """Cells nest as deep as their config path. Taking the first path segment
    as the cell name collapsed every nested cell onto the model tag: one
    typology, everything else marked superseded, and a matrix that put
    different cells on the same row."""
    from src.analysis.campaign_report import _split_cell_and_run, discover_runs

    assert _split_cell_and_run(
        ("qwen25-05b", "sft", "zero-shot", "run_1", "eval_final.json")
    ) == ("qwen25-05b-sft-zero-shot", "run_1")
    # an eval-only subdir of a run is a distinct typology: own metrics
    assert _split_cell_and_run(
        ("qwen25-05b", "sft", "zero-shot", "run_1", "decoding-greedy", "e.json")
    ) == ("qwen25-05b-sft-zero-shot-decoding-greedy", "run_1")
    # baselines live inside the model tree like every other cell; the joined
    # name is the one the factor deduction already understood
    assert _split_cell_and_run(
        ("qwen25-05b", "baseline", "zero-shot", "run_1", "eval_zero_shot.json")
    ) == ("qwen25-05b-baseline-zero-shot", "run_1")
    # the old baseline run dir (zero_shot_<ts>) is not a run: orphan, skipped
    assert (
        _split_cell_and_run(
            ("qwen25-05b-baseline-zero-shot", "zero_shot_1", "eval_zero_shot.json")
        )[0]
        is None
    )
    # loose evals with no run segment are the pre-run_* layout: skipped
    assert _split_cell_and_run(("qwen25-05b", "sft", "eval_final.json"))[0] is None

    for cell in ("sft/zero-shot", "grpo/zero-shot", "sft-grpo/few-shot"):
        path = tmp_path / "qwen25-05b" / cell / "run_1" / "eval_final.json"
        path.parent.mkdir(parents=True)
        path.write_text(
            json.dumps({"rouge_l_mean": 0.9, "num_samples_evaluated": 2000}),
            encoding="utf-8",
        )
    discovered = discover_runs(tmp_path)
    assert len({r["cell"] for r in discovered["runs"]}) == 3


def test_two_models_with_the_same_cell_are_both_kept(tmp_path):
    """The model is part of the typology. Without it, qwen25-15b/sft-grpo/
    few-shot and qwen25-05b/sft-grpo/few-shot deduce identical factors and
    one of the two was marked superseded and vanished from the report."""
    from src.analysis.campaign_report import _overview

    for model, rouge in (("qwen25-05b", 0.96), ("qwen25-15b", 0.98)):
        make_eval(
            tmp_path,
            f"{model}/sft-grpo/few-shot",
            "run_20260101_000000",
            "eval_final.json",
            {"rouge_l_mean": rouge, "prompting": {"mode": "few-shot"}},
        )
    selected, superseded = select_latest(discover_runs(tmp_path)["runs"])
    assert superseded == []
    assert {r["factors"]["model_tag"] for r in selected} == {"qwen25-05b", "qwen25-15b"}
    cells, _cols, values, _ann = _overview(selected, "rouge_l_mean")
    assert cells == ["qwen25-05b/sft-grpo/few-shot", "qwen25-15b/sft-grpo/few-shot"]
    assert values == [[0.96], [0.98]]


def test_pairs_never_cross_models(tmp_path):
    """A pair across two models would attribute the model's effect to the
    factor under comparison."""
    make_eval(
        tmp_path,
        "qwen25-05b/sft/zero-shot",
        "run_20260101_000000",
        "eval_final.json",
        {"prompting": {"mode": "zero-shot"}},
    )
    make_eval(
        tmp_path,
        "qwen25-15b/sft/zero-shot",
        "run_20260101_000000",
        "eval_final.json",
        {"prompting": {"mode": "few-shot"}},
    )
    selected, _ = select_latest(discover_runs(tmp_path)["runs"])
    comparisons, _missing, _warn = build_comparisons(selected)
    for c in comparisons:
        assert c["a"]["factors"]["model_tag"] == c["b"]["factors"]["model_tag"]


def test_overview_drops_empty_columns_and_per_cell_baselines(tmp_path):
    """The overview shows cells. A column with no value at all (the old
    always-empty prompting=unknown) is not drawn, and each cell's
    eval_baseline — the base model again, once per cell — is not a row."""
    from src.analysis.campaign_report import _overview

    make_eval(
        tmp_path,
        "qwen25-05b/grpo/few-shot",
        "run_20260101_000000",
        "eval_final.json",
        {"prompting": {"mode": "few-shot"}},
    )
    make_eval(
        tmp_path,
        "qwen25-05b/grpo/few-shot",
        "run_20260101_000000",
        "eval_baseline.json",
        {"prompting": {"mode": "few-shot"}},
    )
    selected, _ = select_latest(discover_runs(tmp_path)["runs"])
    cells, cols, _values, _ann = _overview(selected, "rouge_l_mean")
    assert cells == ["qwen25-05b/grpo/few-shot"]
    assert cols == ["few-shot"]


def test_matrix_png_size_follows_the_row_count(tmp_path):
    """One row per cell at a fixed height: 21 cells must not make a poster."""
    from PIL import Image

    from src.analysis.campaign_report import plot_matrix

    for i in range(21):
        make_eval(
            tmp_path,
            f"qwen25-05b/ablations/g/c{i:02d}",
            "run_20260101_000000",
            "eval_final.json",
            {"prompting": {"mode": "few-shot"}},
        )
    selected, _ = select_latest(discover_runs(tmp_path)["runs"])
    out = tmp_path / "m.png"
    assert plot_matrix(selected, out)
    width, height = Image.open(out).size
    assert width < 1600 and height < 1600, (width, height)


def test_same_cell_on_two_datasets_stays_two_typologies(tmp_path):
    """Layout <dataset>/<model>/...: the dataset is part of model_tag, so the
    same cell on ASLG-PC12 and PHOENIX-2014T is never superseded nor paired
    across corpora (different test sets are not comparable)."""
    for dataset, rouge in (("aslg-pc12", 0.96), ("phoenix-2014t", 0.40)):
        make_eval(
            tmp_path,
            f"{dataset}/qwen25-05b/sft/zero-shot",
            "run_20260101_000000",
            "eval_final.json",
            {"rouge_l_mean": rouge, "prompting": {"mode": "zero-shot"}},
        )
    selected, superseded = select_latest(discover_runs(tmp_path)["runs"])
    assert superseded == []
    assert {r["factors"]["model_tag"] for r in selected} == {
        "aslg-pc12/qwen25-05b",
        "phoenix-2014t/qwen25-05b",
    }
    assert {r["factors"]["dataset"] for r in selected} == {"aslg-pc12", "phoenix-2014t"}
    comparisons, _missing, _warn = build_comparisons(selected)
    for c in comparisons:
        assert c["a"]["factors"]["dataset"] == c["b"]["factors"]["dataset"]
