"""Isolamento degli artefatti fra le due passate del dual prompting.

La passata dual (modalità complementare, ``prompting_changed=True``) DEVE
essere distinguibile dalla passata primaria su TUTTI gli artefatti scritti su
disco, non solo sui file eval_/generations_/eval_baseline_ che portavano già
il suffisso ``__<mode>``:

- ``comparison.json`` — letto da ``src/utils/ablation_summary.py`` per i delta
  della tabella riassuntiva: senza suffisso, su ogni cella addestrabile i
  delta pubblicati sarebbero quelli del CROSS-prompting (es. zero-shot su una
  cella few-shot) invece di quelli della modalità della cella;
- le figure — ``metrics_dashboard.png``, ``baseline_vs_grpo_comparison.png``,
  ``completion_examples.{json,html}`` e le altre: ``figures_dir`` è condivisa
  fra le passate e la seconda riscriveva la prima;
- ``evaluation.output`` esplicito — un solo path dichiarato nel config,
  condiviso fra le passate.

Il test esegue DUE passate reali di ``_run_eval_pass`` su ``tmp_path`` (con
``evaluate_checkpoint`` e i renderer plotnine stubbati: si verificano i PATH
e i CONTENUTI degli artefatti, non il rendering) e prova che nessun file
prodotto dalla prima venga riscritto dalla seconda: ogni file porta un
marcatore della modalità che l'ha prodotto e le asserzioni leggonoi
contenuti.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.training.eval_t2g import _run_eval_pass  # noqa: E402

# Modalità della passata primaria (implicita nella config: retrieval attivo)
# e modalità complementare della passata dual.
PRIMARY_MODE = "few-shot"
DUAL_MODE = "zero-shot"

_CFG: dict[str, Any] = {
    "model": {"name": "Qwen/Qwen2.5-0.5B-Instruct"},
    "dataset": {"dataset_name": "achrafothman/aslg_pc12", "seed": 42},
    "retrieval": {"enabled": True, "top_k": 3},
}

#: Figure/artefatti grafici prodotti da una passata con plot=True.
FIGURE_NAMES = (
    "metrics_dashboard.png",
    "completion_lengths.png",
    "bleu_distribution.png",
    "chrf_distribution.png",
    "rouge_distribution.png",
    "error_breakdown.png",
    "validity_pie.png",
    "reward_breakdown.png",
    "reward_radar.png",
    "baseline_vs_grpo_comparison.png",
    "completion_examples.json",
    "completion_examples.html",
)

#: Funzioni di rendering stubbate (firme: output_path=... o output_dir=...).
_STUBBED_PLOTTERS = (
    "plot_metrics_dashboard",
    "plot_difficulty_breakdown",
    "plot_completion_length_distribution",
    "plot_score_distribution",
    "plot_rouge_distribution",
    "plot_pass_at_k_curve",
    "plot_error_breakdown",
    "plot_validity_pie",
    "plot_reward_breakdown",
    "plot_reward_radar",
    "plot_baseline_vs_grpo",
    "plot_baseline_vs_grpo_comparison",
    "dump_completion_examples",
)


class _WandbDisabled:
    """Stub di wandb: qualunque accesso fallisce (la chiamata è non-fatale)."""

    def __getattr__(self, name: str) -> Any:
        raise RuntimeError("wandb disabled in test")


def _canned_results(mode: str) -> dict[str, Any]:
    """Risultato minimo con le chiavi lette da _run_eval_pass + marcatore.

    ``rouge_l_mean`` è diverso per modalità: è il marcatore di contenuto con
    cui si prova che comparison.json non viene riscritto dalla seconda
    passata. Include anche le chiavi non_copy_* richieste dal blocco di
    stampa delle metriche primarie.
    """
    rouge = 0.5 if mode == PRIMARY_MODE else 0.6
    return {
        "rouge_l_mean": rouge,
        "rouge_l_std": 0.1,
        "rouge_l_median": rouge,
        "valid_rouge_l_mean": rouge * 0.9,
        "bleu_sentence_mean": 0.4,
        "bleu_corpus": 0.42,
        "chrf_sentence_mean": 41.0,
        "chrf_corpus": 43.0,
        "gloss_f1_sentence_mean": 0.6,
        "gloss_f1_micro": 0.61,
        "exact_match": 0.3,
        "non_copy_token_accuracy": rouge,
        "non_copy_token_hits": 2,
        "non_copy_token_total": 4,
        "bigram_log_prob_mean": -5.0,
        "bigram_log_prob_std": 1.0,
        "validity_rate": 0.9,
        "valid_count": 9,
        "invalid_count": 1,
        "pass_at_1": rouge,
        "reward_breakdown": {"gloss_format_reward": 0.5},
        "error_distribution": {"empty_output": 1},
        "total_completions": 10,
        "num_samples_evaluated": 2,
        "test_set_size": 100,
        "num_completions_per_prompt": 1,
        "metrics_version": 2,
        "prompting": {"mode": mode, "source": "test"},
    }


@pytest.fixture()
def eval_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Stuba evaluate_checkpoint (risultati canned per modalità), i renderer
    plotnine (scrivono un file marcatore al posto del rendering) e wandb;
    ritorna le directory condivise fra le passate + il marker mutabile.
    """
    import src.training.eval_t2g as eval_mod
    import src.utils.visualization as vis

    checkpoint_dir = tmp_path / "ckpts" / "final"
    checkpoint_dir.mkdir(parents=True)
    results_dir = tmp_path / "results" / "tag" / "run_1"
    figures_dir = tmp_path / "figures" / "tag" / "run_1"
    logs_dir = tmp_path / "logs"
    for d in (results_dir, figures_dir, logs_dir):
        d.mkdir(parents=True)

    # Marker mutabile letto dagli stub a ogni chiamata: il test lo aggiorna
    # prima di ogni passata, così ogni file creato porta la modalità che
    # l'ha prodotto.
    marker = {"mode": PRIMARY_MODE}

    # ── evaluate_checkpoint: ritorna i risultati canned della SUA modalità ──
    def fake_evaluate_checkpoint(config, checkpoint_path, **kwargs):
        mode = kwargs["prompting_mode"]
        results = _canned_results(mode)
        completions = [f"GLOSS {mode}"]
        validity = [(True, "")]
        references = ["GLOSS GOLD"]
        rouge_scores = [results["rouge_l_mean"]]
        generations = [
            {
                "index": 0,
                "text": "some text",
                "gold_gloss": "GLOSS GOLD",
                "completion": completions[0],
                "valid": True,
                "rouge_l": results["rouge_l_mean"],
            }
        ]
        return (
            results,
            completions,
            validity,
            references,
            rouge_scores,
            generations,
            [0.4],
            [41.0],
        )

    monkeypatch.setattr(eval_mod, "evaluate_checkpoint", fake_evaluate_checkpoint)

    # ── Renderer plotnine: scrivono il marcatore nei path reali ──────────
    def make_stub():
        def stub(*args: Any, **kwargs: Any) -> None:
            mode = marker["mode"]
            if "output_path" in kwargs:
                Path(kwargs["output_path"]).write_text(mode, encoding="utf-8")
            elif "output_dir" in kwargs:
                out_dir = Path(kwargs["output_dir"])
                out_dir.mkdir(parents=True, exist_ok=True)
                (out_dir / "completion_examples.json").write_text(
                    mode, encoding="utf-8"
                )
                (out_dir / "completion_examples.html").write_text(
                    mode, encoding="utf-8"
                )

        return stub

    for name in _STUBBED_PLOTTERS:
        monkeypatch.setattr(vis, name, make_stub())

    # ── wandb: qualunque uso solleva, catturato dal try/except non-fatale ──
    monkeypatch.setitem(sys.modules, "wandb", _WandbDisabled())

    return {
        "checkpoint": checkpoint_dir,
        "results_dir": results_dir,
        "figures_dir": figures_dir,
        "logs_dir": logs_dir,
        "marker": marker,
    }


def _run_pass(env: dict[str, Any], mode: str, changed: bool, **overrides: Any):
    """Una passata di eval in do_compare (baseline + checkpoint, plot on)."""
    common: dict[str, Any] = {
        "config": _CFG,
        "checkpoint_arg": str(env["checkpoint"]),
        "plot": True,
        "best_of_n": False,
        "max_samples": None,
        "num_samples": 1,
        "force_baseline_eval": False,
        "output_override": None,
        "baseline_pass_at1": None,
        "baseline_json": None,
        "model_tag_default": "test-tag",
        "results_dir": env["results_dir"],
        "figures_dir": env["figures_dir"],
        "logs_dir": env["logs_dir"],
    }
    common.update(overrides)
    _run_eval_pass(
        eval_baseline_only=False,
        do_compare=True,
        prompting_mode=mode,
        prompting_source="config" if not changed else "config-dual",
        prompting_changed=changed,
        **common,
    )


def test_dual_pass_never_overwrites_primary_artifacts(eval_env):
    """PROVA DIRETTA: due passate (config-mode + dual) su tmp_path; nessun
    file prodotto dalla prima viene riscritto dalla seconda."""
    env = eval_env
    results_dir: Path = env["results_dir"]
    figures_dir: Path = env["figures_dir"]

    # ── Passata primaria: modalità della config, nessun suffisso atteso ──
    env["marker"]["mode"] = PRIMARY_MODE
    _run_pass(env, PRIMARY_MODE, changed=False)

    # I file della primaria esistono con i NOMI storici (non suffissati).
    primary_results = (
        "eval_final.json",
        "generations_final.json",
        "eval_baseline.json",
        "generations_baseline.json",
        "comparison.json",
    )
    for name in primary_results:
        assert (results_dir / name).exists(), f"missing primary artifact: {name}"
    for name in FIGURE_NAMES:
        assert (figures_dir / name).exists(), f"missing primary figure: {name}"

    # ── Passata dual (modalità complementare, prompting_changed=True) ──
    env["marker"]["mode"] = DUAL_MODE
    _run_pass(env, DUAL_MODE, changed=True)

    # 1. comparison.json: la primaria è INTATTA e la dual è separata.
    #    ablation_summary legge comparison.json a nome fisso: senza isolamento
    #    pubblicherebbe i delta del CROSS-prompting.
    comp_primary = json.loads((results_dir / "comparison.json").read_text("utf-8"))
    assert comp_primary["checkpoint"]["rouge_l_mean"] == 0.5, (
        "comparison.json rewritten by the dual pass — ablation_summary would "
        "publish cross-prompting deltas"
    )
    comp_dual_path = results_dir / f"comparison__{DUAL_MODE}.json"
    assert comp_dual_path.exists(), "dual pass must write its own comparison.json"
    comp_dual = json.loads(comp_dual_path.read_text("utf-8"))
    assert comp_dual["checkpoint"]["rouge_l_mean"] == 0.6

    # 2. eval/generations/baseline JSON: la primaria è intatta.
    for name in ("eval_final.json", "eval_baseline.json", "generations_final.json"):
        data = json.loads((results_dir / name).read_text("utf-8"))
        if isinstance(data, dict):
            assert data.get("prompting", {}).get("mode") == PRIMARY_MODE, name
        else:
            assert f"GLOSS {PRIMARY_MODE}" in json.dumps(data), name
    eval_dual = json.loads(
        (results_dir / f"eval_final__{DUAL_MODE}.json").read_text("utf-8")
    )
    assert eval_dual["prompting"]["mode"] == DUAL_MODE

    # 3. Figure: la primaria (root) è intatta, la dual è in sottodirectory.
    for name in FIGURE_NAMES:
        path = figures_dir / name
        assert path.exists(), f"primary figure clobbered: {name}"
        assert (
            path.read_text("utf-8") == PRIMARY_MODE
        ), f"primary figure {name} was rewritten by the dual pass"
    dual_fig_dir = figures_dir / DUAL_MODE
    assert dual_fig_dir.is_dir(), "dual pass must render in its own subdirectory"
    for name in FIGURE_NAMES:
        dual_file = dual_fig_dir / name
        assert dual_file.exists(), f"missing dual figure: {dual_file}"
        assert (
            dual_file.read_text("utf-8") == DUAL_MODE
        ), f"dual figure {dual_file} does not carry the dual marker"


def test_dual_pass_never_overwrites_explicit_output(eval_env):
    """evaluation.output esplicito: la passata dual NON sovrascrive l'output
    della primaria — il suffisso __<mode> è applicato anche al path custom."""
    env = eval_env
    custom_output = env["results_dir"] / "custom" / "out.json"
    custom_output.parent.mkdir(parents=True, exist_ok=True)

    env["marker"]["mode"] = PRIMARY_MODE
    _run_pass(env, PRIMARY_MODE, changed=False, output_override=str(custom_output))
    assert custom_output.exists()
    first = json.loads(custom_output.read_text("utf-8"))
    assert first["prompting"]["mode"] == PRIMARY_MODE

    env["marker"]["mode"] = DUAL_MODE
    _run_pass(env, DUAL_MODE, changed=True, output_override=str(custom_output))

    # L'output della primaria è INTATTO; la dual ha il suo file suffissato
    # (e il generations_ derivato eredita il suffisso).
    still_primary = json.loads(custom_output.read_text("utf-8"))
    assert (
        still_primary["prompting"]["mode"] == PRIMARY_MODE
    ), "evaluation.output was rewritten by the dual pass"
    dual_out = custom_output.with_stem(f"{custom_output.stem}__{DUAL_MODE}")
    assert dual_out.exists()
    dual_data = json.loads(dual_out.read_text("utf-8"))
    assert dual_data["prompting"]["mode"] == DUAL_MODE
    dual_gens = custom_output.parent / f"generations_{dual_out.stem}.json"
    assert dual_gens.exists(), "generations must inherit the __<mode> suffix"


def test_primary_pass_filenames_unchanged(eval_env):
    """Regressione: la passata PRIMARIA (prompting_changed=False) non cambia
    alcun nome file — comparison.json resta leggibile a nome fisso da
    src/utils/ablation_summary.py."""
    env = eval_env
    _run_pass(env, PRIMARY_MODE, changed=False)
    results_dir: Path = env["results_dir"]
    assert (results_dir / "comparison.json").exists()
    assert not (results_dir / f"comparison__{PRIMARY_MODE}.json").exists()
    # Nessuna sottodirectory per modalità sulla passata primaria.
    assert not (env["figures_dir"] / PRIMARY_MODE).exists()
