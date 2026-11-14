"""src/utils/run_status.py: the adapter between t2g and the runstatus library."""

import json

from src.utils import run_status as rs


def _fresh(monkeypatch, tmp_path, job_name=None):
    monkeypatch.setenv("RUNSTATUS_PATH", str(tmp_path / "rs.json"))
    if job_name:
        monkeypatch.setenv("SLURM_JOB_NAME", job_name)
    else:
        monkeypatch.delenv("SLURM_JOB_NAME", raising=False)
    monkeypatch.setattr(rs.atexit, "register", lambda *a, **k: None)
    monkeypatch.setattr(rs, "_chain_excepthook", lambda: None)
    rs._reset_for_tests()


def _read(tmp_path, name="rs.json"):
    return json.loads((tmp_path / name).read_text())


def test_kind_follows_the_slurm_job_name(monkeypatch, tmp_path):
    _fresh(monkeypatch, tmp_path, "eval-aslg-x")
    rs.run_status_set(phase="eval", total_steps=10)
    assert _read(tmp_path)["run"]["kind"] == "eval"
    _fresh(monkeypatch, tmp_path, "train-aslg-x")
    rs.run_status_set(phase="sft")
    assert _read(tmp_path)["run"]["kind"] == "train"


def test_phase_change_restarts_the_charts(monkeypatch, tmp_path):
    _fresh(monkeypatch, tmp_path)
    rs.run_status_set(phase="sft", total_steps=100)
    rs.run_status_log(10, loss=1.0)
    rs.run_status_set(phase="grpo", total_steps=50)
    rs.run_status_log(5, loss=0.5)
    rs.get_status().finish()
    series = _read(tmp_path, "rs.series.json")["series"]["loss"]
    assert series == [[5, 0.5]]


def test_eval_progress_samples_and_stage_done(monkeypatch, tmp_path):
    _fresh(monkeypatch, tmp_path)
    rs.run_status_set(phase="eval", step=0, total_steps=4)
    rs.run_status_set(step=2, total_steps=4, eval_progress="2/4")
    rs.run_status_samples(["a", "b"], kind="sft")
    rs.run_status_stage_done("fatto")
    d = _read(tmp_path)
    assert d["progress"]["step"] == 2 and d["progress"]["pct"] == 50.0
    assert d["eval"]["progress"] == "2/4" and d["samples"] == ["a", "b"]
    assert (
        d["note"] == "fatto" and d["run"]["state"] == "running"
    ), "finished only at process exit"


def test_calls_are_noops_without_the_library(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def no_runstatus(name, *a, **k):
        if "runstatus" in name:
            raise ImportError(name)
        return real_import(name, *a, **k)

    rs._reset_for_tests()
    monkeypatch.setattr(builtins, "__import__", no_runstatus)
    rs.run_status_set(phase="sft")
    rs.run_status_log(1, loss=1.0)
    rs.run_status_samples(["x"], kind="sft")
    assert rs.get_status() is None
    rs._reset_for_tests()


def test_describe_on_a_real_config_and_full_flow(monkeypatch, tmp_path):
    """Prova a secco dell'intero giro con un config vero: describe, fasi, log,
    campioni, risultati. Il chip del dataset viene da dataset.dataset_name."""
    from src.utils.config import resolve_config

    _fresh(monkeypatch, tmp_path, "train-conll-2003-sft-grpo-zero-shot")
    cfg = resolve_config(
        "experiments/configs/conll-2003/qwen25-05b/sft-grpo/zero-shot.yaml"
    )
    rs.run_status_describe(cfg, kind="train", method="grpo", stages=["sft", "grpo"])
    rs.run_status_set(phase="sft", note="SFT")
    rs.run_status_log(10, epoch=0.1, total_steps=100, loss=1.5, learning_rate=2e-5)
    rs.run_status_set(phase="grpo")
    rs.run_status_log(5, total_steps=50, reward=-0.2, **{"rewards/x": 0.5})
    rs.run_status_samples(
        [{"prompt": "p", "output": "o", "gold": "g", "match": False}], kind="grpo"
    )
    rs.run_status_results({"exact_match": 0.5, "nested": {"a": 1.0}, "skip": "text"})
    rs.get_status().finish()
    d = _read(tmp_path)
    assert d["run"]["state"] == "finished" and d["run"]["method"] == "grpo"
    assert d["run"]["model"] == {
        "family": "llm",
        "name": "Qwen2.5-0.5B-Instruct",
        "params": "0.5B",
    }
    assert "conll-2003" in [b["label"] for b in d["run"]["badges"]]
    assert [s["state"] for s in d["run"]["stages"]] == ["done", "done"]
    assert d["metrics"]["latest"]["eval/exact_match"] == 0.5
    assert d["metrics"]["latest"]["eval/nested/a"] == 1.0
    assert d["samples"][0]["output"] == "o"
    assert _read(tmp_path, "rs.series.json")["series"]["reward"] == [[5, -0.2]]
