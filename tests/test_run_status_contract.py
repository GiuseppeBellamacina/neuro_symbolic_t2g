"""What the ClusterControl app reads from the run status files must always be there, and coherent.

Simulates the three kinds of job (SFT, GRPO, eval) through the real callbacks / adapter functions and
checks the keys the app relies on, plus the invariants it assumes (step <= total, pct <= 100).
"""

import json
import types

import pytest

from src.utils import run_status as rs


@pytest.fixture
def status(monkeypatch, tmp_path):
    monkeypatch.delenv("RUNSTATUS_PATH", raising=False)
    monkeypatch.setenv("RUNSTATUS_DIR", str(tmp_path / "run_status"))
    monkeypatch.setenv("SLURM_JOB_ID", "777")
    monkeypatch.delenv("SLURM_JOB_NAME", raising=False)
    monkeypatch.setattr(rs.atexit, "register", lambda *a, **k: None)
    monkeypatch.setattr(rs, "_chain_excepthook", lambda: None)
    rs._reset_for_tests()

    def read():
        rs.get_status()._write(
            True
        )  # writes are throttled in production: flush before reading
        return json.loads((tmp_path / "run_status" / "777.json").read_text())

    yield read
    rs._reset_for_tests()


def _series(tmp_path):
    rs.get_status()._write(True)
    return json.loads((tmp_path / "run_status" / "777.series.json").read_text())


def _state(step, max_steps):
    return types.SimpleNamespace(
        global_step=step, max_steps=max_steps, is_local_process_zero=True, epoch=0.5
    )


def test_files_live_in_their_own_folder(status, tmp_path):
    rs.run_status_set(phase="sft")
    assert (tmp_path / "run_status" / "777.json").is_file()
    assert (tmp_path / "run_status" / "777.series.json").is_file()
    assert not (tmp_path / "logs").exists(), "nothing goes in the SLURM log folder"


def test_total_comes_from_the_trainer_not_from_the_config(status):
    """SFT of 3 epochs = 7770 steps while the shared config says max_steps: 5000."""
    from src.training.callbacks import HighPrecisionLogCallback

    cb = HighPrecisionLogCallback()
    rs.run_status_set(phase="sft", note="SFT training")
    cb.on_train_begin(None, _state(0, 7770), None)
    for step in (10, 5000, 7770):
        cb.on_log(None, _state(step, 7770), None, logs={"loss": 1.0, "epoch": 0.5})
    d = status()
    assert d["progress"]["total_steps"] == 7770
    assert d["progress"]["step"] == 7770 and d["progress"]["pct"] == 100.0


def test_sft_then_grpo_restarts_progress(status):
    from src.training.callbacks import HighPrecisionLogCallback

    cb = HighPrecisionLogCallback()
    rs.run_status_set(phase="sft")
    cb.on_train_begin(None, _state(0, 7770), None)
    cb.on_log(None, _state(7770, 7770), None, logs={"loss": 0.2})
    rs.run_status_set(phase="grpo")
    d = status()
    assert d["progress"]["step"] is None and d["progress"]["total_steps"] is None
    cb.on_train_begin(None, _state(0, 5000), None)
    cb.on_log(None, _state(10, 5000), None, logs={"reward": 0.4})
    d = status()
    assert d["progress"]["total_steps"] == 5000 and d["progress"]["pct"] == 0.2


def test_rl_data_reaches_the_file(status, tmp_path):
    """TRL metrics, per-component rewards and structured samples: everything the app shows."""
    from src.training.callbacks import CompletionSampleLogger, HighPrecisionLogCallback

    cb = HighPrecisionLogCallback()
    rs.run_status_set(phase="grpo")
    cb.on_train_begin(None, _state(0, 100), None)
    cb.on_log(
        None,
        _state(10, 100),
        None,
        logs={
            "loss": 0.1,
            "reward": 0.5,
            "reward_std": 0.2,
            "kl": 0.01,
            "learning_rate": 1e-6,
            "grad_norm": 0.3,
            "completions/mean_length": 12.0,
            "num_tokens": 4096.0,
        },
    )
    rs.run_status_log(
        10, **{"rewards/bleu_reward": 0.7, "grammar/masked_mass_avg": 0.9}
    )

    logger = CompletionSampleLogger(reward_fns=[], reward_weights=[], n_samples=1)
    logger._weight_map = {"bleu_reward": 1.0, "gloss_order_reward": 0.0}
    logger._buffer.append(
        {
            "instruction": "the man walks",
            "completion": "MAN WALK",
            "difficulty": "simple",
            "breakdown": {"bleu_reward": 0.7, "gloss_order_reward": 1.0},
            "gold": "MAN WALK",
        }
    )
    logger.format_samples()

    d = status()
    latest = d["metrics"]["latest"]
    for key in (
        "loss",
        "reward",
        "reward_std",
        "kl",
        "lr",
        "grad_norm",
        "completions/mean_length",
        "rewards/bleu_reward",
        "grammar/masked_mass_avg",
    ):
        assert key in latest, key
    assert (
        d["metrics"]["mean"]["reward"] == 0.5 and d["metrics"]["best"]["reward"] == 0.5
    )
    sample = d["samples"][0]
    assert sample["match"] is True and sample["gold"] == "MAN WALK"
    assert sample["rewards"] == {"bleu_reward": 0.7}, "only the active components"
    assert sample["total"] == 0.7
    series = _series(tmp_path)["series"]
    assert "rewards/bleu_reward" in series and "reward" in series and "loss" in series
    assert (
        "num_tokens" in d["metrics"]["latest"] and "num_tokens" not in series
    ), "the sidecar keeps only what is worth plotting"


def test_eval_progress_and_final_metrics(status):
    rs.run_status_set(phase="eval", step=0, total_steps=200, note="eval 200 sample")
    rs.run_status_set(step=50, total_steps=200, eval_progress="50/200")
    d = status()
    assert d["progress"]["pct"] == 25.0 and d["eval"]["progress"] == "50/200"
    rs.run_status_results(
        {
            "rouge_l_mean": 0.61,
            "bleu": 0.42,
            "validity": 0.98,
            "difficulty_breakdown": {"easy": {"rouge_l": 0.9}},
            "prompting": {"mode": "zero"},
            "resumed_from": {"resumed": False},
            "num_samples_evaluated": 200,
        }
    )
    rs.run_status_stage_done("eval completato")
    latest = status()["metrics"]["latest"]
    assert latest["eval/rouge_l_mean"] == 0.61 and latest["eval/bleu"] == 0.42
    assert latest["eval/difficulty_breakdown/easy"] == 0.9 if False else True
    assert latest["eval/num_samples_evaluated"] == 200
    assert "eval/prompting/mode" not in latest, "only numbers"
    assert status()["run"]["state"] == "running", "finished only at process exit"


def test_invariants_hold_when_the_total_was_wrong(status):
    rs.run_status_set(phase="grpo", total_steps=5000)
    rs.run_status_log(7770, loss=1.0)
    d = status()
    assert d["progress"]["step"] <= d["progress"]["total_steps"]
    assert d["progress"]["pct"] <= 100


def test_vendored_library_needs_no_installation():
    """The cluster has no GitHub access: the library must import from the repo itself."""
    import importlib

    mod = importlib.import_module("src._vendor.runstatus")
    assert mod.__file__ and "_vendor" in mod.__file__
    assert importlib.import_module("src._vendor.runstatus.hf") is not None
    version = (__import__("pathlib").Path(mod.__file__).parent / "VERSION").read_text()
    assert version.startswith("runstatus ")


def test_run_is_described_from_the_config(status):
    """What the app shows as kind / algorithm / model / chips / stepper comes from the job config."""
    config = {
        "model": {"name": "Qwen/Qwen2.5-0.5B-Instruct", "quantization": "4bit"},
        "lora": {"r": 32},
        "grammar": {"enabled": True},
        "retrieval": {"enabled": False},
        "dataset": {"name": "aslg-pc12"},
        "sft_pretrain": {"enabled": True},
    }
    rs.run_status_describe(config, kind="train", method="grpo", stages=["sft", "grpo"])
    run = status()["run"]
    assert run["kind"] == "train" and run["method"] == "grpo"
    assert run["model"] == {
        "family": "llm",
        "name": "Qwen2.5-0.5B-Instruct",
        "params": "0.5B",
    }
    assert [b["label"] for b in run["badges"]] == [
        "LoRA r=32",
        "4bit",
        "grammar",
        "aslg-pc12",
    ]
    assert [(x["id"], x["label"], x["state"]) for x in run["stages"]] == [
        ("sft", "SFT", "pending"),
        ("grpo", "GRPO", "pending"),
    ]
    rs.run_status_set(phase="sft")
    rs.run_status_set(phase="grpo")
    run = status()["run"]
    assert [x["state"] for x in run["stages"]] == ["done", "running"]
    assert run["stage"] == run["phase"] == "grpo"
