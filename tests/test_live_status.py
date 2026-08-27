"""Test del modulo live status (src/utils/live_status.py).

Il modulo scrive logs/live_status.json ATOMICAMENTE con throttling delle
metriche. Qui testiamo: scrittura/lettura, throttle, keep-6 dei samples,
reset e fail-safety (mai eccezioni verso il chiamante).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.utils import live_status as ls


@pytest.fixture(autouse=True)
def _isolated_status_file(tmp_path, monkeypatch):
    """Punta STATUS_PATH su una tmp dir e resetta lo stato del modulo."""
    target = tmp_path / "live_status.json"
    monkeypatch.setattr(ls, "STATUS_PATH", target)
    # Reset module state (fresh payload, no throttle residue, errors re-arm).
    monkeypatch.setattr(ls, "_state", dict(ls._BASE_PAYLOAD))
    monkeypatch.setattr(ls, "_last_write", 0.0)
    monkeypatch.setattr(ls, "_error_logged", False)
    yield target


def test_set_writes_atomic_json(_isolated_status_file):
    ls.live_status_set(phase="sft", step=10, loss=0.5)
    data = ls.live_status_get()
    assert data is not None
    assert data["phase"] == "sft"
    assert data["step"] == 10
    assert data["loss"] == 0.5
    assert data["ts"]
    assert data["pid"]
    # Single-line JSON (KEY=VALUE protocol safe).
    assert "\n" not in _isolated_status_file.read_text(encoding="utf-8")


def test_merge_and_overwrite(_isolated_status_file):
    ls.live_status_set(phase="grpo", step=1, loss=1.0)
    ls.live_status_add_samples(["s"], kind="grpo")  # scrittura non throttled
    ls.live_status_set(step=2, reward=0.3)  # merge: loss resta, step cambia
    # La seconda set può essere THROTTLED: ri-leggere dopo una scrittura
    # forzata (add_samples bypassa il throttle e preserva lo stato merged).
    ls.live_status_add_samples(["s"], kind="grpo")
    data = ls.live_status_get()
    assert data["phase"] == "grpo"
    assert data["step"] == 2
    assert data["loss"] == 1.0
    assert data["reward"] == 0.3


def test_add_samples_keeps_last_six(_isolated_status_file):
    ls.live_status_add_samples([f"sample-{i}" for i in range(10)], kind="sft")
    data = ls.live_status_get()
    assert data["samples"] == [f"sample-{i}" for i in range(4, 10)]
    assert data["samples_kind"] == "sft"


def test_reset_returns_to_idle(_isolated_status_file):
    ls.live_status_set(phase="grpo", step=100, loss=0.1)
    ls.live_status_reset(note="finito")
    data = ls.live_status_get()
    assert data["phase"] is None
    assert data["step"] is None
    assert data["note"] == "finito"


def test_failsafe_never_raises(tmp_path, monkeypatch):
    """Path non scrivibile → nessuna eccezione, il training continua."""

    class _Boom:
        def mkdir(self, *a, **k):
            raise OSError("read-only")

    monkeypatch.setattr(ls, "STATUS_PATH", tmp_path / "nope" / "x" / "live.json")
    monkeypatch.setattr(ls, "STATUS_PATH", tmp_path / "nope" / "x" / "live.json")
    # Patch Path.mkdir per far fallire la creazione della directory
    real_mkdir = Path.mkdir

    def _fail_mkdir(self, *args, **kwargs):
        if self.name in ("nope", "x"):
            raise OSError("simulated read-only filesystem")
        return real_mkdir(self, *args, **kwargs)

    monkeypatch.setattr(Path, "mkdir", _fail_mkdir)
    ls.live_status_set(phase="sft")  # must NOT raise
    ls.live_status_add_samples(["a"], kind="sft")  # must NOT raise
    ls.live_status_reset()  # must NOT raise
