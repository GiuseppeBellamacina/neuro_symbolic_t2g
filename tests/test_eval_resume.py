"""Resume dello stato parziale dell'eval (walltime-safe).

Il job di valutazione muore per TIMEOUT in un istante arbitrario e il JSON
dei risultati esiste solo a fine passata: senza checkpoint parziale si perde
TUTTA la generazione (accaduto su sft/zero-shot: kill a 1964/3000 dopo 6,5 h).
Questi test coprono il contratto del meccanismo:

- **equivalenza**: una passata interrotta e ripresa produce accumulatori e
  metriche IDENTICI a una completata in un colpo solo (stesso campione,
  stesse completions deterministiche, stessa matematica);
- **scarto sicuro**: uno stato con fingerprint/max_samples/sample_id non
  corrispondenti, o corrotto, è IGNORATO (con motivo in log) e la passata
  riparte da zero — mai metriche su un insieme misto di prompt;
- **atomicità**: un .tmp scritto a metà (rinomina mai avvenuta) non viene
  mai caricato;
- **pulizia**: a passata completata lo stato è rimosso (la sua presenza deve
  significare solo "passata incompleta");
- **non raccolta**: il nome resume_state_*.json non matcha i pattern
  ``eval_*.json`` di ``src/utils/ablation_summary.py`` e
  ``src/analysis/campaign_report.py`` — uno stato incompleto non è mai
  raccolto come risultato.

Tutto gira su ``tmp_path`` con generazione stubbata: nessun modello, nessuna
rete, nessuna GPU.
"""

from __future__ import annotations

import hashlib
import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.training.eval_t2g import (  # noqa: E402
    _collect_completions_with_resume,
    _compute_primary_metrics,
    _load_resume_state,
    _remove_resume_state,
    _run_eval_pass,
    _select_best_of_n,
)
from src.utils.metrics import METRICS_VERSION  # noqa: E402

N_PROMPTS = 37  # 7 cadenze da 5 + 2: la coda non allineata alla cadenza è coperta
NUM_SAMPLES = 3
RESUME_EVERY = 5
KILL_AFTER = 13  # stato salvato a 10 prompt (ultima cadenza prima del kill)

# Accumulatori = (completions, references, sample_ids, texts, difficulties).
Accs = tuple[list[list[str]], list[str], list[str], list[str], list[str]]

# Mini-vocabolario + matrice bigram per _compute_primary_metrics: le completions
# finte restano dentro questo vocabolario, così il path bigram reale gira su
# indici validi (token_to_idx.get("<BOS>", 0) / "<EOS>", 1).
_MINI_VOCAB = ["<BOS>", "<EOS>", "GLOSS", "GOOD", "BAD"]
_MINI_TOKEN_TO_IDX = {t: i for i, t in enumerate(_MINI_VOCAB)}
_MINI_BIGRAM = np.full((len(_MINI_VOCAB), len(_MINI_VOCAB)), -1.0)


class SimulatedTimeout(RuntimeError):
    """Kill del job per walltime, simulato."""


class DeterministicGenerator:
    """Generatore stub: completions deterministiche dal testo del prompt.

    Il kill avviene DOPO ``kill_after`` chiamate riuscite: la chiamata
    successiva solleva a metà lavoro (il prompt in corso non è mai nello
    stato — il salvataggio segue l'append, mai il contrario).
    """

    def __init__(self, num_samples: int, kill_after: int | None = None) -> None:
        self.num_samples = num_samples
        self.kill_after = kill_after
        self.calls = 0
        self.prompts: list[str] = []

    def __call__(self, prompt: str) -> list[str]:
        self.calls += 1
        self.prompts.append(prompt)
        if self.kill_after is not None and self.calls > self.kill_after:
            raise SimulatedTimeout(
                f"simulated SLURM walltime kill after {self.kill_after} prompts"
            )
        digest = hashlib.md5(prompt.encode("utf-8")).hexdigest()
        completions = []
        for j in range(self.num_samples):
            pick = "GOOD" if (int(digest[:8], 16) + j) % 2 == 0 else "BAD"
            completions.append(f"GLOSS {pick}")
        return completions


def _make_dataset(n: int = N_PROMPTS) -> list[dict[str, str]]:
    """Dataset finto: testi/gold deterministici fra esecuzioni (la stessa
    precondizione che il campione seeded di seeded_sample_indices dà al
    resume per indice sul dataset reale)."""
    return [
        {
            "text": f"Sample sentence {i} for the deterministic eval run.",
            "gloss": f"GLOSS GOLD {i}",
        }
        for i in range(n)
    ]


def _run_collection(
    state_path: Path | None,
    gen: DeterministicGenerator,
    *,
    max_samples: int | None = N_PROMPTS,
    resume_every: int = RESUME_EVERY,
    fingerprint: str = "stable-fingerprint",
) -> tuple[Accs, dict[str, Any]]:
    """Una chiamata al loop di generazione con i parametri di test fissi."""
    out = _collect_completions_with_resume(
        test_ds=_make_dataset(),
        tokenizer=object(),  # build_t2g_prompt: niente chat template → fallback manuale
        examples_batch=None,
        generate_fn=gen,
        num_samples=NUM_SAMPLES,
        resume_state_path=state_path,
        resume_every=resume_every,
        fingerprint=fingerprint,
        metrics_version=METRICS_VERSION,
        max_samples=max_samples,
        prompting_mode="zero-shot",
        checkpoint_key="ckpts/final",
        test_set_size=9999,
    )
    return out[:5], out[5]


def _primary_metrics(accs: Accs) -> dict[str, Any]:
    """Il blocco di metriche primarie esattamente come lo calcola l'eval.

    bootstrap_confidence_interval è seedato internamente: a parità di input
    il dict results è deterministico → confrontabile con ==.
    """
    completions, refs, _sids, texts, _diffs = accs
    flat = [c for comps in completions for c in comps]
    flat_refs = [r for comps, r in zip(completions, refs) for _ in comps]
    flat_sources = [t for comps, t in zip(completions, texts) for _ in comps]
    results, *_rest = _compute_primary_metrics(
        flat,
        flat_refs,
        completions,
        refs,
        token_to_idx=_MINI_TOKEN_TO_IDX,
        bigram=_MINI_BIGRAM,
        reward_weights={},
        flat_sources=flat_sources,
        n_bootstrap=20,
    )
    return results


def _make_partial_state(tmp_path: Path) -> Path:
    """Una passata interrotta: crea e restituisce il path dello stato parziale
    su cui i test di scarto vanno a mutare un campo alla volta."""
    state = tmp_path / "resume_state_final.json"
    gen = DeterministicGenerator(NUM_SAMPLES, kill_after=KILL_AFTER)
    with pytest.raises(SimulatedTimeout):
        _run_collection(state, gen)
    return state


def _load_state(state: Path) -> dict[str, Any]:
    return json.loads(state.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Equivalenza (il test più importante) + best-of-N oracolo
# ---------------------------------------------------------------------------


def test_interrupted_and_resumed_pass_produces_identical_metrics(tmp_path):
    """Una passata completata in un colpo solo e una interrotta a metà e
    ripresa producono accumulatori, metriche primarie e selezione best-of-N
    IDENTICI. Differisce solo il metadata di provenienza (resumed_from)."""
    state = tmp_path / "resume_state_final.json"

    # Passata A: completa, senza stato (comportamento storico).
    gen_a = DeterministicGenerator(NUM_SAMPLES)
    acc_a, info_a = _run_collection(None, gen_a)
    results_a = _primary_metrics(acc_a)
    assert info_a == {"resumed": False, "resume_count": 0, "recovered_prompts": 0}
    assert gen_a.calls == N_PROMPTS

    # Passata B: interrotta per walltime dopo 13 prompt generati.
    gen_b = DeterministicGenerator(NUM_SAMPLES, kill_after=KILL_AFTER)
    with pytest.raises(SimulatedTimeout):
        _run_collection(state, gen_b)
    assert gen_b.calls == KILL_AFTER + 1
    saved = _load_state(state)
    # Ultima cadenza raggiunta prima del kill: multiplo di resume_every ≤ 13.
    last_cadence = (KILL_AFTER // RESUME_EVERY) * RESUME_EVERY
    assert saved["completed_prompts"] == last_cadence
    assert saved["metrics_version"] == METRICS_VERSION

    # Passata C: ripresa dallo stato — rigenera SOLO i prompt mancanti.
    gen_c = DeterministicGenerator(NUM_SAMPLES)
    acc_c, info_c = _run_collection(state, gen_c)
    recovered = last_cadence
    assert info_c == {
        "resumed": True,
        "resume_count": 1,
        "recovered_prompts": recovered,
    }
    assert (
        gen_c.calls == N_PROMPTS - recovered
    ), "il resume deve saltare i prompt già nello stato, non rigenerarli"
    assert gen_c.prompts == gen_a.prompts[recovered:], (
        "i prompt rigenerati sono esattamente quelli che la passata "
        "interrotta non ha fatto"
    )

    # Gli accumulatori sono identici per indice: la matematica non cambia.
    assert acc_c == acc_a

    # Metriche primarie IDENTICHE (dict equality; bootstrap seedato interno).
    results_c = _primary_metrics(acc_c)
    assert results_c == results_a

    # Il best-of-N oracolo usa gli stessi accumulatori: anche la sua
    # selezione è identica dopo una ripresa.
    assert _select_best_of_n(acc_c[0], acc_c[1]) == _select_best_of_n(
        acc_a[0], acc_a[1]
    )

    # Il metadata di provenienza distingue le due passate senza toccare le
    # metriche (results["resumed_from"] viene aggiunto da evaluate_checkpoint
    # a partire da questo dict).
    assert info_c["resumed"] is True and info_a["resumed"] is False


def test_double_resume_increments_resume_count(tmp_path):
    """Due interruzioni consecutive: il contatore di riprese cresce, il
    resume riparte dal SECONDO stato parziale e il risultato resta
    equivalente a quello della passata intera."""
    state = tmp_path / "resume_state_final.json"

    gen_ref = DeterministicGenerator(NUM_SAMPLES)
    acc_ref, _ = _run_collection(None, gen_ref)

    gen_kill_1 = DeterministicGenerator(NUM_SAMPLES, kill_after=7)
    with pytest.raises(SimulatedTimeout):
        _run_collection(state, gen_kill_1)
    gen_kill_2 = DeterministicGenerator(NUM_SAMPLES, kill_after=11)
    with pytest.raises(SimulatedTimeout):
        _run_collection(state, gen_kill_2)
    # La seconda ripresa è partita dallo stato del primo kill e lo ha fatto
    # avanzare: il recovered del terzo avvio è il completed dello stato attuale.
    expected_recovered = _load_state(state)["completed_prompts"]
    assert expected_recovered > RESUME_EVERY

    gen_last = DeterministicGenerator(NUM_SAMPLES)
    acc_last, info_last = _run_collection(state, gen_last)
    assert info_last["resumed"] is True
    assert info_last["resume_count"] == 2
    assert info_last["recovered_prompts"] == expected_recovered
    assert acc_last == acc_ref


# ---------------------------------------------------------------------------
# Scarto sicuro dello stato (ogni scarto: ripartenza da zero + motivo in log)
# ---------------------------------------------------------------------------


def test_discard_on_fingerprint_mismatch(tmp_path, caplog):
    state = _make_partial_state(tmp_path)
    data = _load_state(state)
    data["prompt_context_fingerprint"] = "a-different-context"
    state.write_text(json.dumps(data), encoding="utf-8")

    gen = DeterministicGenerator(NUM_SAMPLES)
    with caplog.at_level(logging.WARNING, logger="t2g-eval"):
        _accs, info = _run_collection(state, gen)
    assert info["resumed"] is False
    assert gen.calls == N_PROMPTS, "stato con fingerprint diverso: ripartenza da zero"
    assert "Discarding partial eval state" in caplog.text
    assert "prompt_context_fingerprint" in caplog.text
    # Lo stato scartato è rimosso e la passata da zero ne riscrive uno con il
    # fingerprint CORRETTO (la rimozione al momento dello scarto è provata da
    # test_discard_removes_state_and_reports_field).
    assert _load_state(state)["prompt_context_fingerprint"] == "stable-fingerprint"


def test_discard_on_max_samples_mismatch(tmp_path, caplog):
    state = _make_partial_state(tmp_path)
    # La stessa passata rilanciata con un max_samples diverso (config cambiata
    # fra le esecuzioni): il campione non è più lo stesso.
    gen = DeterministicGenerator(NUM_SAMPLES)
    with caplog.at_level(logging.WARNING, logger="t2g-eval"):
        _accs, info = _run_collection(state, gen, max_samples=N_PROMPTS - 1)
    assert info["resumed"] is False
    assert gen.calls == N_PROMPTS
    assert "max_samples" in caplog.text


def test_discard_on_sample_id_mismatch(tmp_path, caplog):
    state = _make_partial_state(tmp_path)
    data = _load_state(state)
    # Lo sample_id alla posizione 3 non corrisponde a quello che il campione
    # deterministico produce per quella posizione: stato di un altro campione.
    data["accumulators"]["sample_ids"][3] = "0" * 64
    state.write_text(json.dumps(data), encoding="utf-8")

    gen = DeterministicGenerator(NUM_SAMPLES)
    with caplog.at_level(logging.WARNING, logger="t2g-eval"):
        _accs, info = _run_collection(state, gen)
    assert info["resumed"] is False
    assert gen.calls == N_PROMPTS
    assert "sample_id mismatch at position 3" in caplog.text


def test_discard_on_corrupt_file(tmp_path, caplog):
    state = tmp_path / "resume_state_final.json"
    state.write_bytes(b'{"kind": "eval_partial_stat')  # JSON troncato

    gen = DeterministicGenerator(NUM_SAMPLES)
    with caplog.at_level(logging.WARNING, logger="t2g-eval"):
        _accs, info = _run_collection(state, gen)
    assert info["resumed"] is False
    assert gen.calls == N_PROMPTS
    assert "unreadable or corrupt file" in caplog.text


def test_discard_removes_state_and_reports_field(tmp_path, caplog):
    """Contratto diretto di _load_resume_state: scartato ⇒ rimosso, con il
    campo che non corrispondeva esplicitamente nel log."""
    state = _make_partial_state(tmp_path)
    data = _load_state(state)
    data["num_samples"] = NUM_SAMPLES + 1
    state.write_text(json.dumps(data), encoding="utf-8")
    with caplog.at_level(logging.WARNING, logger="t2g-eval"):
        loaded = _load_resume_state(
            state,
            fingerprint="stable-fingerprint",
            metrics_version=METRICS_VERSION,
            max_samples=N_PROMPTS,
            num_samples=NUM_SAMPLES,
            prompting_mode="zero-shot",
            checkpoint_key="ckpts/final",
            test_set_size=9999,
            sampled_count=N_PROMPTS,
            test_ds=_make_dataset(),
        )
    assert loaded is None
    assert not state.exists()
    assert "num_samples" in caplog.text


# ---------------------------------------------------------------------------
# Atomicità della scrittura
# ---------------------------------------------------------------------------


def test_truncated_tmp_file_is_never_loaded(tmp_path, caplog):
    """Kill DURANTE la scrittura: il .tmp esiste, la rinomina non è avvenuta,
    lo stato vero non c'è. Il .tmp non deve MAI essere caricato."""
    state = tmp_path / "resume_state_final.json"
    tmp_file = state.with_name(state.name + ".tmp")
    tmp_file.write_text(
        '{"kind": "eval_partial_state", "completed_prompts": 3', encoding="utf-8"
    )

    gen = DeterministicGenerator(NUM_SAMPLES)
    with caplog.at_level(logging.WARNING, logger="t2g-eval"):
        _accs, info = _run_collection(state, gen)
    assert info["resumed"] is False
    assert gen.calls == N_PROMPTS, "il .tmp parziale non è uno stato riprendibile"
    # Assenza dello stato vero è il caso normale: nessun avviso di scarto.
    assert "Discarding" not in caplog.text


def test_valid_state_survives_a_stale_tmp(tmp_path):
    """Il .tmp orfano di una scrittura interrotta non invalida lo stato vero
    (già rinominato da una scrittura precedente completata)."""
    state = _make_partial_state(tmp_path)
    recovered = _load_state(state)["completed_prompts"]
    state.with_name(state.name + ".tmp").write_bytes(b"garbage-from-a-killed-write")

    gen = DeterministicGenerator(NUM_SAMPLES)
    accs, info = _run_collection(state, gen)
    assert info["resumed"] is True
    assert info["recovered_prompts"] == recovered
    assert gen.calls == N_PROMPTS - recovered
    assert len(accs[0]) == N_PROMPTS


# ---------------------------------------------------------------------------
# Pulizia e disattivazione
# ---------------------------------------------------------------------------


def test_remove_resume_state_deletes_state_and_tmp(tmp_path):
    state = tmp_path / "resume_state_final.json"
    state.write_text("{}", encoding="utf-8")
    state.with_name(state.name + ".tmp").write_text("{}", encoding="utf-8")
    _remove_resume_state(state)
    assert not state.exists()
    assert not state.with_name(state.name + ".tmp").exists()
    # Idempotente e tollerante su None (path mai usato dalla passata).
    _remove_resume_state(state)
    _remove_resume_state(None)


def test_resume_every_zero_disables_save_and_load(tmp_path):
    """evaluation.resume_every <= 0 = comportamento storico: né salvataggio
    né ripresa, anche con uno stato valido sul disco."""
    state = tmp_path / "resume_state_final.json"
    state.write_text("{}", encoding="utf-8")  # verrebbe caricato se attivo

    gen = DeterministicGenerator(NUM_SAMPLES)
    accs, info = _run_collection(state, gen, resume_every=0)
    assert len(accs[0]) == N_PROMPTS
    assert gen.calls == N_PROMPTS, "resume_every=0: nessuna ripresa"
    assert info == {"resumed": False, "resume_count": 0, "recovered_prompts": 0}
    # Né salvataggio né rimozione: col meccanismo spento lo stato preesistente
    # resta intoccato (non è compito dell'eval cancellarlo).
    assert state.read_text(encoding="utf-8") == "{}"


# ---------------------------------------------------------------------------
# Interazione con _run_eval_pass (pulizia a fine passata, stati per soggetto)
# ---------------------------------------------------------------------------


class _WandbDisabled:
    """Stub di wandb: qualunque accesso fallisce (la chiamata è non-fatale)."""

    def __getattr__(self, name: str) -> Any:
        raise RuntimeError("wandb disabled in test")


def _canned_results(mode: str) -> dict[str, Any]:
    """Risultato minimo con le chiavi lette da _run_eval_pass."""
    return {
        "rouge_l_mean": 0.5,
        "rouge_l_std": 0.1,
        "rouge_l_median": 0.5,
        "valid_rouge_l_mean": 0.45,
        "bleu_sentence_mean": 0.4,
        "bleu_corpus": 0.42,
        "chrf_sentence_mean": 41.0,
        "chrf_corpus": 43.0,
        "gloss_f1_sentence_mean": 0.6,
        "gloss_f1_micro": 0.61,
        "exact_match": 0.3,
        "non_copy_token_accuracy": 0.5,
        "non_copy_token_hits": 2,
        "non_copy_token_total": 4,
        "bigram_log_prob_mean": -5.0,
        "bigram_log_prob_std": 1.0,
        "validity_rate": 0.9,
        "valid_count": 9,
        "invalid_count": 1,
        "pass_at_1": 0.5,
        "reward_breakdown": {"gloss_format_reward": 0.5},
        "error_distribution": {"empty_output": 1},
        "total_completions": 10,
        "num_samples_evaluated": 2,
        "test_set_size": 100,
        "num_completions_per_prompt": 1,
        "metrics_version": 2,
        "prompting": {"mode": mode, "source": "test"},
    }


def test_run_eval_pass_removes_partial_states_and_uses_one_per_subject(
    tmp_path, monkeypatch
):
    """A passata completata i file di stato sono RIMOSSI; baseline e
    checkpoint ricevono percorsi DISTINTI (mai uno stato condiviso), e la
    passata dual usa il suffisso __<mode> anche sullo stato."""
    import src.training.eval_t2g as eval_mod

    results_dir = tmp_path / "results" / "tag" / "run_1"
    figures_dir = tmp_path / "figures" / "tag" / "run_1"
    logs_dir = tmp_path / "logs"
    for d in (results_dir, figures_dir, logs_dir):
        d.mkdir(parents=True)

    received: list[str] = []

    def fake_evaluate_checkpoint(config, checkpoint_path, **kwargs):
        # Il vero evaluate_checkpoint scrive lo stato lungo la generazione:
        # lo stub replica il contratto (stato presente a fine passata).
        state_path = Path(kwargs["resume_state_path"])
        state_path.write_text("{}", encoding="utf-8")
        received.append(state_path.name)
        generations = [
            {
                "index": 0,
                "text": "t",
                "gold_gloss": "GLOSS GOLD",
                "completion": "GLOSS",
                "valid": True,
                "rouge_l": 0.5,
            }
        ]
        return (
            _canned_results(kwargs["prompting_mode"]),
            ["GLOSS"],
            [(True, "")],
            ["GLOSS GOLD"],
            [0.5],
            generations,
            [0.4],
            [41.0],
        )

    monkeypatch.setattr(eval_mod, "evaluate_checkpoint", fake_evaluate_checkpoint)
    monkeypatch.setitem(sys.modules, "wandb", _WandbDisabled())

    common: dict[str, Any] = {
        "config": {
            "model": {"name": "Qwen/Qwen2.5-0.5B-Instruct"},
            "dataset": {"dataset_name": "achrafothman/aslg_pc12", "seed": 42},
            "retrieval": {"enabled": False},
        },
        "checkpoint_arg": str(tmp_path / "ckpts" / "final"),
        "plot": False,
        "best_of_n": False,
        "max_samples": None,
        "num_samples": 1,
        "force_baseline_eval": False,
        "output_override": None,
        "baseline_pass_at1": None,
        "baseline_json": None,
        "model_tag_default": "test-tag",
        "results_dir": results_dir,
        "figures_dir": figures_dir,
        "logs_dir": logs_dir,
    }

    for suffix, mode in (("", "zero-shot"), ("__few-shot", "few-shot")):
        _run_eval_pass(
            eval_baseline_only=False,
            do_compare=True,
            prompting_mode=mode,
            prompting_source="config-dual" if suffix else "config",
            prompting_changed=bool(suffix),
            **common,
        )
        # Un percorso per soggetto (baseline + checkpoint), suffisso per modalità.
        assert f"resume_state_baseline{suffix}.json" in received
        assert f"resume_state_final{suffix}.json" in received
        # Pulizia a passata completata: NESSUN file di stato residuo accanto
        # agli artefatti finiti.
        assert not list(
            results_dir.glob("resume_state_*")
        ), "lo stato parziale deve essere rimosso a passata completata"


# ---------------------------------------------------------------------------
# Non raccolta da ablation_summary e campaign_report
# ---------------------------------------------------------------------------


def test_state_files_are_not_collected_as_results(tmp_path):
    """I pattern dei raccoglitori sono ``latest_run.glob("eval_*.json")``
    (ablation_summary.py:99) e ``results_dir.rglob("eval_*.json")``
    (campaign_report.py:303). Un file che non inizia con ``eval_`` non vi
    ricade: qui si prova sui MODULI REALI, non solo sul glob."""
    from src.analysis.campaign_report import discover_runs
    from src.utils.ablation_summary import find_eval_results

    run_dir = tmp_path / "cell" / "run_1"
    run_dir.mkdir(parents=True)
    (run_dir / "eval_final.json").write_text(
        json.dumps({"rouge_l_mean": 0.5}), encoding="utf-8"
    )
    # Tutti i nomi di stato che il meccanismo può produrre (+ tmp).
    for name in (
        "resume_state_final.json",
        "resume_state_baseline.json",
        "resume_state_final__zero-shot.json",
        "resume_state_baseline__few-shot.json",
        "resume_state_final.json.tmp",
    ):
        (run_dir / name).write_text("{}", encoding="utf-8")

    entries = find_eval_results(tmp_path)
    collected = [Path(e["eval_path"]).name for e in entries]
    assert collected == [
        "eval_final.json"
    ], f"ablation_summary ha raccolto file di stato: {collected}"

    runs = discover_runs(tmp_path)
    run_names = [Path(r["path"]).name for r in runs.get("runs", [])]
    assert run_names == [
        "eval_final.json"
    ], f"campaign_report ha raccolto file di stato: {run_names}"
