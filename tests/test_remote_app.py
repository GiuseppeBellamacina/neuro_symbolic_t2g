"""Test del driver esterno (remote/app.py) — nessuna rete reale.

ClusterSSH è sostituito da un doppio in memoria che simula lo stato del
cluster (.chain_state/) ed esegue le stesse mutazioni del helper lato
cluster (enqueue/rewrite_queue/pause/resume). Nessun subprocess ssh/sbatch.

Skip automatico se fastapi/httpx non sono installati (vedi pyproject dev
extras): serve solo a questo test, non alle dipendenze core del progetto.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import remote.app as app_module

AUTH = {"X-Auth-Token": "test-token"}

# Ordine ESATTO di cluster/run_all.sh:134-147 (TAG:CONFIG:MODE) — se cambia,
# aggiornare sia app.ABLATION_MODELS sia questa lista.
EXPECTED_ABLATION_MODELS: list[tuple[str, str, str]] = [
    ("zero-shot", "experiments/configs/t2g/ablation/zero_shot.yaml", "e"),
    ("zero-shot-gram", "experiments/configs/t2g/ablation/zero_shot_grammar.yaml", "e"),
    ("grpo-no-grammar", "experiments/configs/t2g/ablation/grpo_no_grammar.yaml", "te"),
    ("grpo-no-sft", "experiments/configs/t2g/ablation/grpo_no_sft.yaml", "te"),
    ("grpo-grammar", "experiments/configs/t2g/grpo_qwen05.yaml", "te"),
    ("sft", "experiments/configs/t2g/sft.yaml", "te"),
    ("grpo-pda", "experiments/configs/t2g/ablation/grpo_pda.yaml", "te"),
    (
        "grpo-pda-lookahead",
        "experiments/configs/t2g/ablation/grpo_pda_lookahead.yaml",
        "te",
    ),
    (
        "grpo-soft-viterbi",
        "experiments/configs/t2g/ablation/grpo_soft_viterbi.yaml",
        "te",
    ),
    (
        "grpo-verifier",
        "experiments/configs/t2g/ablation/grpo_verifier_scaled.yaml",
        "te",
    ),
    (
        "grpo-experimental-all",
        "experiments/configs/t2g/grpo_experimental_all.yaml",
        "te",
    ),
    ("grpo-optimal", "experiments/configs/t2g/grpo_optimal.yaml", "te"),
]


def _ablation_queue() -> list[str]:
    lines: list[str] = []
    for tag, cfg, mode in EXPECTED_ABLATION_MODELS:
        if mode == "e":
            lines.append(f"eval:{cfg}:{tag}")
        else:
            lines.append(f"train:{cfg}:{tag}")
            lines.append(f"eval:{cfg}:{tag}")
    return lines


class FakeClusterSSH:
    """Doppio di ClusterSSH: stato in memoria + log dei comandi remoti.

    Simula fedelmente il helper lato cluster: subcomandi enqueue /
    rewrite_queue / pause / resume modificano la coda e lo stato; ogni
    `run` ritorna lo snapshot KEY=VALUE aggiornato. Supporta anche i
    subcomandi v2: `monitor` (snapshot + LOG_TAIL_B64), `scancel` (kill).
    """

    def __init__(self, settings: app_module.Settings) -> None:
        self.settings = settings
        self.commands: list[str] = []
        self.queue: list[str] = []
        self.active_job: str = ""
        self.last_job: str = ""
        self.stopped = False
        self.watcher_alive = False
        self.errors: list[str] = []
        self.rc = 0
        self.stderr = ""
        self.status_text: str | None = None  # override totale dello stdout
        self.helper_missing_calls = 0  # invocazioni da simulare come "mancante"
        # v2: contenuto del log del job attivo (per `monitor`); se None,
        # `monitor` risponde senza LOG_TAIL_B64 (nessun log disponibile).
        self.log_lines: list[str] | None = None
        self.scancel_calls: list[str] = []

    def _snapshot(self, with_log: bool = False) -> str:
        if self.status_text is not None:
            return self.status_text
        queue = "\x1f".join(self.queue)
        tail = json.dumps(self.errors[-5:])
        out = (
            "STATUS_OK=1\n"
            f"ACTIVE_JOB={self.active_job}\n"
            f"QUEUE={queue}\n"
            f"QUEUE_COUNT={len(self.queue)}\n"
            f"LAST_JOB={self.last_job}\n"
            f"STOPPED={1 if self.stopped else 0}\n"
            f"WATCHER_ALIVE={1 if self.watcher_alive else 0}\n"
            f"ERRORS_COUNT={len(self.errors)}\n"
            f"ERRORS_TAIL={tail}\n"
        )
        if with_log:
            import base64

            log_path = ""
            b64 = ""
            if self.active_job and self.log_lines is not None:
                prefix = (
                    "eval"
                    if self.active_job.split("|")[1].startswith("eval-")
                    else "train"
                )
                log_path = f"~/neuro_symbolic_t2g/logs/slurm-{prefix}-{self.active_job.split('|')[0]}.log"
                b64 = base64.b64encode("\n".join(self.log_lines).encode()).decode()
            out += f"LOG_PATH={log_path}\nLOG_TAIL_B64={b64}\n"
        return out

    def run(self, remote_cmd: str, timeout: int | None = None) -> app_module.SSHResult:
        self.commands.append(remote_cmd)
        if self.helper_missing_calls > 0:
            self.helper_missing_calls -= 1
            return app_module.SSHResult(0, "HELPER_MISSING=1\n", "")
        sub = self._subcommand(remote_cmd)
        if sub == "enqueue":
            self.queue.append(self._arg(remote_cmd))
        elif sub == "rewrite_queue":
            content = self._arg(remote_cmd)
            self.queue = [e for e in content.split("\x1f") if e]
        elif sub == "pause":
            self.stopped = True
        elif sub == "resume":
            self.stopped = False
        elif sub == "monitor":
            return app_module.SSHResult(
                self.rc, self._snapshot(with_log=True), self.stderr
            )
        elif sub == "scancel":
            if not self.active_job:
                return app_module.SSHResult(1, "", "ERR_NO_ACTIVE_JOB=1")
            job_id = self.active_job.split("|")[0]
            self.scancel_calls.append(job_id)
            self.active_job = ""
            return app_module.SSHResult(0, f"OK_SCANCEL={job_id}", "")
        # "status" e "tick" non mutano lo stato simulato
        return app_module.SSHResult(self.rc, self._snapshot(), self.stderr)

    def upload(self, local_path: str, remote_path: str, timeout: int | None = None):
        self.commands.append(f"UPLOAD:{Path(local_path).name}")
        return app_module.SSHResult(0, "uploaded", "")

    @staticmethod
    def _subcommand(remote_cmd: str) -> str:
        """Subcomando = token dopo `bash .../cluster_helper.sh`."""
        tokens = remote_cmd.split()
        for i, tok in enumerate(tokens):
            if (
                tok == "bash"
                and i + 2 < len(tokens)
                and tokens[i + 1].endswith("cluster_helper.sh")
            ):
                return tokens[i + 2].rstrip(";")
        return ""

    @staticmethod
    def _arg(remote_cmd: str) -> str:
        # l'argomento è la PRIMA stringa tra apici singoli del comando remoto
        parts = remote_cmd.split("'")
        return parts[1] if len(parts) > 1 else ""

    def __enter__(self) -> "FakeClusterSSH":
        return self

    def __exit__(self, *exc) -> None:
        return None


@pytest.fixture()
def client(monkeypatch, tmp_path):
    """TestClient con ClusterSSH sostituito dal doppio e DB su tmp_path."""
    from fastapi.testclient import TestClient

    monkeypatch.setenv("T2G_AUTH_TOKEN", "test-token")
    monkeypatch.setenv("T2G_DB_PATH", str(tmp_path / "t2g_driver.db"))
    monkeypatch.setenv("T2G_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("T2G_SSH_KEY_FILE", str(tmp_path / "ssh_key"))
    monkeypatch.setenv("T2G_SSH_KEY_CONTENT", "")
    monkeypatch.setenv("T2G_SSH_HOST", "unit.test")
    monkeypatch.setenv("T2G_SSH_USER", "tester")
    monkeypatch.setenv("T2G_SSH_PORT", "22")
    monkeypatch.setenv("T2G_HELPER_AUTO_INSTALL", "0")
    (tmp_path / "ssh_key").write_text("chiave-di-test", encoding="utf-8")

    app_module.settings = app_module.Settings.from_env()
    fake = FakeClusterSSH(app_module.settings)
    monkeypatch.setattr(app_module, "ClusterSSH", lambda _settings: fake)

    with TestClient(app_module.app) as test_client:
        yield test_client, fake


# ── Auth ──────────────────────────────────────────────────────────────────────


def test_auth_required_401(client):
    test_client, _ = client
    assert test_client.get("/status").status_code == 401
    assert test_client.post("/tick").status_code == 401
    assert (
        test_client.post("/jobs", json={"type": "train", "config": "sft"}).status_code
        == 401
    )
    assert test_client.post("/queue", json={"ablation": True}).status_code == 401
    assert test_client.delete("/jobs/foo").status_code == 401
    assert test_client.post("/pause").status_code == 401
    assert test_client.post("/resume").status_code == 401
    assert (
        test_client.get("/status", headers={"X-Auth-Token": "sbagliato"}).status_code
        == 401
    )


# ── Chiave SSH opzionale (deploy locale) ──────────────────────────────────────


def _keyless_settings(monkeypatch, tmp_path) -> None:
    """Env del deploy locale: NESSUNA env di chiave (ssh-agent/config default)."""
    monkeypatch.delenv("T2G_SSH_KEY_FILE", raising=False)
    monkeypatch.delenv("T2G_SSH_KEY_CONTENT", raising=False)
    monkeypatch.setenv("T2G_AUTH_TOKEN", "test-token")
    monkeypatch.setenv("T2G_DB_PATH", str(tmp_path / "t2g_driver.db"))
    monkeypatch.setenv("T2G_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("T2G_SSH_HOST", "unit.test")
    monkeypatch.setenv("T2G_SSH_USER", "tester")
    monkeypatch.setenv("T2G_HELPER_AUTO_INSTALL", "0")


def test_ssh_key_optional_local_deploy(monkeypatch, tmp_path):
    """Senza T2G_SSH_KEY_FILE/T2G_SSH_KEY_CONTENT: chiave opzionale → nessun
    `-i` (ssh-agent / ~/.ssh/config) e NESSUN 502 per chiave mancante."""
    from fastapi.testclient import TestClient

    _keyless_settings(monkeypatch, tmp_path)
    app_module.settings = app_module.Settings.from_env()
    assert app_module.settings.ssh_key_file is None

    fake = FakeClusterSSH(app_module.settings)
    monkeypatch.setattr(app_module, "ClusterSSH", lambda _settings: fake)
    with TestClient(app_module.app) as test_client:
        resp = test_client.post("/tick", headers=AUTH)
        assert resp.status_code == 200
        # il comando remoto NON deve contenere `-i` (la chiave è opzionale) e
        # il cluster risponde senza il 502 "Chiave SSH non trovata"
        assert "-i" not in fake.commands[-1]
        assert "cluster_helper.sh tick" in fake.commands[-1]


def test_ssh_key_explicit_but_missing_file_502(client, monkeypatch):
    """Se T2G_SSH_KEY_FILE è impostato esplicitamente ma il file non esiste,
    resta un errore chiaro (502) — il comportamento non cambia con la chiave
    opzionale."""
    test_client, _ = client
    monkeypatch.setattr(
        app_module.settings, "ssh_key_file", str(Path("C:/chiave_inesistente"))
    )
    resp = test_client.post("/tick", headers=AUTH)
    assert resp.status_code == 502
    assert "Chiave SSH" in resp.json()["detail"]


def test_shq_preserves_unit_separator():
    """_shq deve lasciare intatto il separatore \x1f della coda (viaggia dentro
    le virgolette singole del comando remoto, sicuro su commandline ssh)."""
    entry = "train:experiments/configs/t2g/sft.yaml:a\x1feval:experiments/configs/t2g/sft.yaml:a"
    q = app_module._shq(entry)
    assert "\x1f" in q
    assert q.count("'") % 2 == 0  # quoting bilanciato: apri-chiudi
    assert q.startswith("'") and q.endswith("'")


# ── /status ────────────────────────────────────────────────────────────────────


def test_status_format_after_tick(client):
    test_client, fake = client
    fake.active_job = "12345|train-foo|RUNNING"
    fake.queue = ["train:experiments/configs/t2g/grpo_optimal.yaml:run1"]
    fake.last_job = "12345:train:experiments/configs/t2g/grpo_optimal.yaml:run1:0"

    resp = test_client.post("/tick", headers=AUTH)
    assert resp.status_code == 200
    body = resp.json()
    for key in (
        "active_job",
        "queue",
        "last_job",
        "stopped",
        "watcher_alive",
        "errors_recent",
        "last_tick_at",
        "cluster_reachable",
        "events",
    ):
        assert key in body
    assert body["active_job"] == {
        "id": "12345",
        "name": "train-foo",
        "state": "RUNNING",
    }
    assert body["queue"] == ["train:experiments/configs/t2g/grpo_optimal.yaml:run1"]
    assert body["stopped"] is False
    assert body["cluster_reachable"] is True
    assert any(e["type"] == "tick" for e in body["events"])

    st = test_client.get("/status", headers=AUTH).json()
    assert st["queue"] == ["train:experiments/configs/t2g/grpo_optimal.yaml:run1"]
    assert st["cluster_reachable"] is True
    assert st["last_tick_at"]


def test_status_reads_cache_when_cluster_down(client):
    test_client, fake = client
    fake.rc = 255
    fake.stderr = "ssh: connect to host unit.test port 22: Connection refused"

    resp = test_client.post("/tick", headers=AUTH)
    assert resp.status_code == 502
    assert "Cluster" in resp.json()["detail"]

    # GET /status NON fa ssh: risponde con last_known + reachable=false
    st = test_client.get("/status", headers=AUTH)
    assert st.status_code == 200
    assert st.json()["cluster_reachable"] is False


def test_protocol_garbage_is_502(client):
    test_client, fake = client
    fake.status_text = "BOH=1\nnon-so-cosa-sia\n"
    resp = test_client.post("/tick", headers=AUTH)
    assert resp.status_code == 502
    assert "STATUS_OK" in resp.json()["detail"]


def test_helper_missing_no_autoinstall_502(client):
    test_client, fake = client
    fake.helper_missing_calls = 1
    resp = test_client.post("/tick", headers=AUTH)
    assert resp.status_code == 502
    assert "cluster_helper.sh" in resp.json()["detail"]


def test_helper_missing_autoinstall_via_scp(client):
    test_client, fake = client
    app_module.settings.helper_auto_install = True
    fake.helper_missing_calls = 1

    resp = test_client.post("/tick", headers=AUTH)
    assert resp.status_code == 200
    assert any(cmd.startswith("UPLOAD:") for cmd in fake.commands)
    assert resp.json()["cluster_reachable"] is True


# ── /jobs ──────────────────────────────────────────────────────────────────────


def test_jobs_add_and_list(client):
    test_client, fake = client
    resp = test_client.post(
        "/jobs",
        headers=AUTH,
        json={"type": "train", "config": "grpo_optimal", "tag": "run1"},
    )
    assert resp.status_code == 201
    assert (
        resp.json()["added"] == "train:experiments/configs/t2g/grpo_optimal.yaml:run1"
    )
    assert fake.queue == ["train:experiments/configs/t2g/grpo_optimal.yaml:run1"]
    assert " enqueue " in fake.commands[-1]

    jobs = test_client.get("/jobs", headers=AUTH).json()
    assert jobs == [
        {
            "entry": "train:experiments/configs/t2g/grpo_optimal.yaml:run1",
            "type": "train",
            "config": "experiments/configs/t2g/grpo_optimal.yaml",
            "tag": "run1",
            "extra": None,
        }
    ]


def test_jobs_tag_derived_and_mode(client):
    test_client, _ = client
    resp = test_client.post(
        "/jobs", headers=AUTH, json={"type": "eval", "config": "zero_shot"}
    )
    assert resp.status_code == 201
    assert (
        resp.json()["added"]
        == "eval:experiments/configs/t2g/ablation/zero_shot.yaml:zero-shot"
    )

    resp = test_client.post(
        "/jobs",
        headers=AUTH,
        json={"type": "train", "config": "grpo_qwen05", "tag": "x", "mode": "--resume"},
    )
    assert resp.status_code == 201
    assert (
        resp.json()["added"]
        == "train:experiments/configs/t2g/grpo_qwen05.yaml:x:--resume"
    )


def test_jobs_validation(client):
    test_client, fake = client
    assert (
        test_client.post(
            "/jobs", headers=AUTH, json={"type": "train", "config": "non_existente"}
        ).status_code
        == 422
    )
    assert (
        test_client.post(
            "/jobs", headers=AUTH, json={"type": "run", "config": "sft"}
        ).status_code
        == 422
    )
    assert not fake.commands  # nessuna ssh se la validazione fallisce


# ── /queue ────────────────────────────────────────────────────────────────────


def test_ablation_order_matches_run_all(client):
    # Pin dell'ordine: app.ABLATION_MODELS deve coincidere con run_all.sh:134-147
    assert app_module.ABLATION_MODELS == EXPECTED_ABLATION_MODELS


def test_queue_replace_ablation_shortcut(client):
    test_client, fake = client
    resp = test_client.post("/queue", headers=AUTH, json={"ablation": True})
    assert resp.status_code == 200
    body = resp.json()
    assert body["count"] == 22  # 12 config → 2 eval-only + 10 train+eval
    expected = _ablation_queue()
    assert body["queue"] == expected
    assert fake.queue == expected
    assert " rewrite_queue " in fake.commands[-1]


def test_queue_replace_explicit_jobs(client):
    test_client, _ = client
    resp = test_client.post(
        "/queue",
        headers=AUTH,
        json={
            "jobs": [
                {"type": "eval", "config": "zero_shot"},
                {"type": "train", "config": "grpo_qwen05", "tag": "exp1"},
            ]
        },
    )
    assert resp.status_code == 200
    assert resp.json()["queue"] == [
        "eval:experiments/configs/t2g/ablation/zero_shot.yaml:zero-shot",
        "train:experiments/configs/t2g/grpo_qwen05.yaml:exp1",
    ]


def test_queue_validation(client):
    test_client, fake = client
    assert test_client.post("/queue", headers=AUTH, json={}).status_code == 422
    assert (
        test_client.post(
            "/queue", headers=AUTH, json={"ablation": True, "jobs": []}
        ).status_code
        == 422
    )
    assert not fake.commands


# ── DELETE /jobs/{tag} ─────────────────────────────────────────────────────────


def test_delete_jobs_by_tag(client):
    test_client, fake = client
    test_client.post(
        "/queue",
        headers=AUTH,
        json={
            "jobs": [
                {"type": "train", "config": "grpo_optimal", "tag": "t1"},
                {"type": "eval", "config": "grpo_optimal", "tag": "t1"},
                {"type": "eval", "config": "zero_shot", "tag": "t2"},
            ]
        },
    )
    resp = test_client.delete("/jobs/t1", headers=AUTH)
    assert resp.status_code == 200
    assert resp.json()["removed"] == 2
    assert resp.json()["status"]["queue"] == [
        "eval:experiments/configs/t2g/ablation/zero_shot.yaml:t2"
    ]
    assert fake.queue == ["eval:experiments/configs/t2g/ablation/zero_shot.yaml:t2"]


def test_delete_jobs_unknown_tag_no_rewrite(client):
    test_client, fake = client
    test_client.post("/queue", headers=AUTH, json={"ablation": True})
    n_before = len(fake.commands)
    resp = test_client.delete("/jobs/tag-inesistente", headers=AUTH)
    assert resp.status_code == 200
    assert resp.json()["removed"] == 0
    assert len(fake.commands) == n_before + 1  # solo lo status, niente rewrite
    assert len(fake.queue) == 22


# ── /pause / /resume ───────────────────────────────────────────────────────────


def test_pause_creates_chain_stopped(client):
    test_client, fake = client
    resp = test_client.post("/pause", headers=AUTH)
    assert resp.status_code == 200
    assert fake.stopped is True
    assert resp.json()["status"]["stopped"] is True
    assert "cluster_helper.sh pause;" in fake.commands[-1]

    st = test_client.get("/status", headers=AUTH).json()
    assert st["stopped"] is True


def test_resume_removes_stopped_and_ticks(client):
    test_client, fake = client
    test_client.post("/pause", headers=AUTH)
    assert fake.stopped is True

    resp = test_client.post("/resume", headers=AUTH)
    assert resp.status_code == 200
    assert fake.stopped is False
    assert resp.json()["status"]["stopped"] is False
    joined = " ".join(fake.commands)
    assert " resume" in joined
    assert " tick" in joined
    assert any(e["type"] == "resume" for e in resp.json()["status"]["events"])


# ── Diario errori (puntatore errors_offset) ────────────────────────────────────


def test_errors_offset_pointer_dedup(client):
    test_client, fake = client
    fake.errors = [
        '{"tag":"x","error_type":"TIMEOUT","timestamp":"2026-01-01 00:00:00"}',
        '{"tag":"y","error_type":"OOM","timestamp":"2026-01-01 00:01:00"}',
    ]

    test_client.post("/tick", headers=AUTH)
    st = test_client.get("/status", headers=AUTH).json()
    assert len(st["errors_recent"]) == 2
    assert len([e for e in st["events"] if e["type"] == "error"]) == 2

    # stesso stato → nessun nuovo evento (puntatore)
    test_client.post("/tick", headers=AUTH)
    st = test_client.get("/status", headers=AUTH).json()
    assert len([e for e in st["events"] if e["type"] == "error"]) == 2

    # nuovo errore → loggato solo quello nuovo
    fake.errors.append(
        '{"tag":"z","error_type":"CUDA_ERROR","timestamp":"2026-01-01 00:02:00"}'
    )
    test_client.post("/tick", headers=AUTH)
    st = test_client.get("/status", headers=AUTH).json()
    assert len([e for e in st["events"] if e["type"] == "error"]) == 3


# ── / (root, senza auth) ───────────────────────────────────────────────────────


def test_root_no_auth(client):
    test_client, _ = client
    resp = test_client.get("/")
    assert resp.status_code == 200
    assert resp.json()["service"] == "t2g-cluster-driver"
    assert "docs" in resp.json()


# ── API v2: /monitor /jobs/start /kill /logs ──────────────────────────────────


TRAIN_LOG_LINES = [
    "STEP 7: GRPO Training",
    "  step=100  loss=0.5432  reward=0.3500  learning_rate=0.000003  epoch=0.05",
    "  step=105  loss=0.5311  reward=0.3620  learning_rate=0.000003  epoch=0.06",
]


def test_monitor_without_active_job(client):
    """job_detail null e log_tail vuoti quando nessun job è attivo."""
    test_client, fake = client
    fake.active_job = ""
    fake.log_lines = None

    resp = test_client.get("/monitor", headers=AUTH)
    assert resp.status_code == 200
    body = resp.json()
    assert body["job_detail"] is None
    assert body["log_tail"] == []
    assert body["samples"] == []
    assert "ts" in body
    assert "cluster_reachable" in body


def test_monitor_parses_job_detail_from_log(client):
    """Con un job attivo: job_detail popolato dai parser di chain_monitor."""
    test_client, fake = client
    fake.active_job = "12345|train-grpo-optimal|RUNNING"
    fake.log_lines = TRAIN_LOG_LINES

    resp = test_client.get("/monitor", headers=AUTH)
    assert resp.status_code == 200
    body = resp.json()
    detail = body["job_detail"]
    assert detail is not None
    assert detail["id"] == "12345"
    assert detail["name"] == "train-grpo-optimal"
    assert detail["state"] == "RUNNING"
    assert detail["step"] == 105  # ultima riga KV step
    assert detail["loss"] == "0.5311"
    assert detail["reward"] == "0.3620"  # reward dell'ULTIMA riga KV step
    assert detail["lr"] == "0.000003"
    assert detail["sft_active"] is False
    assert "slurm-train-12345.log" in (detail["log_path"] or "")
    assert body["log_tail"], "log tail vuoto"
    assert "step=105" in body["log_tail"][-1]


def test_monitor_eval_job_uses_eval_parser(client):
    """Job eval-*: riconosciuto dal nome (prefisso eval) senza crash."""
    test_client, fake = client
    fake.active_job = "999|eval-grpo-optimal|RUNNING"
    fake.log_lines = ["Evaluating 100/8771 samples (seeded sample)", "  Pass@1: 0.1234"]

    resp = test_client.get("/monitor", headers=AUTH)
    assert resp.status_code == 200
    body = resp.json()
    assert body["job_detail"]["name"] == "eval-grpo-optimal"


def test_start_job_enqueues_and_ticks(client):
    """POST /jobs/start: enqueue + tick; started_now se il job è attivo."""
    test_client, fake = client

    # tick sottomette il job → diventa attivo con nome train-<tag>
    def _tick_side_effect(remote_cmd, timeout=None):
        fake.commands.append(remote_cmd)
        sub = fake._subcommand(remote_cmd)
        if sub == "enqueue":
            fake.queue.append(fake._arg(remote_cmd))
        elif sub == "tick":
            if fake.queue and not fake.active_job:
                entry = fake.queue.pop(0)
                parts = entry.split(":")
                fake.active_job = f"777|{parts[0]}-{parts[2]}|RUNNING"
                fake.last_job = f"777:{entry}:0"
        return app_module.SSHResult(0, fake._snapshot(), "")

    fake.run = _tick_side_effect  # type: ignore[method-assign]

    resp = test_client.post(
        "/jobs/start", headers=AUTH, json={"type": "train", "config": "grpo_optimal"}
    )
    assert resp.status_code == 201
    body = resp.json()
    assert body["started_now"] is True
    assert body["active_job"]["name"] == "train-grpo-optimal"
    joined = " ".join(fake.commands)
    assert "enqueue" in joined
    assert " tick" in joined


def test_start_job_enqueued_when_busy(client):
    """Job attivo diverso → started_now False (resta in coda)."""
    test_client, fake = client
    fake.active_job = "111|train-altro|RUNNING"

    resp = test_client.post(
        "/jobs/start", headers=AUTH, json={"type": "train", "config": "grpo_optimal"}
    )
    assert resp.status_code == 201
    assert resp.json()["started_now"] is False
    assert fake.queue == [
        "train:experiments/configs/t2g/grpo_optimal.yaml:grpo-optimal"
    ]


def test_kill_cancels_active_job(client):
    """POST /kill: scancel dell'id attivo; snapshot di ritorno."""
    test_client, fake = client
    fake.active_job = "12345|train-grpo-optimal|RUNNING"

    resp = test_client.post("/kill", headers=AUTH)
    assert resp.status_code == 200
    assert fake.scancel_calls == ["12345"]
    # il comando scancel è tra quelli inviati (seguito dal monitor di risposta)
    assert any("cluster_helper.sh scancel" in c for c in fake.commands)


def test_kill_without_active_job_409(client):
    test_client, fake = client
    fake.active_job = ""
    resp = test_client.post("/kill", headers=AUTH)
    assert resp.status_code == 409
    assert "Nessun job attivo" in resp.json()["detail"]


def test_logs_endpoint_returns_tail(client):
    test_client, fake = client
    fake.active_job = "12345|train-grpo-optimal|RUNNING"
    fake.log_lines = [f"linea-{i}" for i in range(300)]

    resp = test_client.get("/logs", headers=AUTH, params={"lines": 50})
    assert resp.status_code == 200
    body = resp.json()
    assert "slurm-train-12345.log" in body["log_path"]
    assert len(body["lines"]) == 50
    assert body["lines"][-1] == "linea-299"

    # clamp al massimo 500
    resp_max = test_client.get("/logs", headers=AUTH, params={"lines": 9999})
    assert resp_max.status_code == 200


def test_logs_without_active_job_404(client):
    test_client, fake = client
    fake.active_job = ""
    resp = test_client.get("/logs", headers=AUTH)
    assert resp.status_code == 404


# ── Auth opzionale (deploy locale) ────────────────────────────────────────────


def test_auth_disabled_without_token(monkeypatch, tmp_path):
    """Senza T2G_AUTH_TOKEN le route rispondono SENZA header (deploy locale)."""
    from fastapi.testclient import TestClient

    monkeypatch.delenv("T2G_AUTH_TOKEN", raising=False)
    monkeypatch.setenv("T2G_DB_PATH", str(tmp_path / "t2g_driver.db"))
    monkeypatch.setenv("T2G_DATA_DIR", str(tmp_path))
    monkeypatch.delenv("T2G_SSH_KEY_FILE", raising=False)
    monkeypatch.setenv("T2G_SSH_KEY_CONTENT", "")
    monkeypatch.setenv("T2G_SSH_HOST", "unit.test")
    monkeypatch.setenv("T2G_SSH_USER", "tester")
    monkeypatch.setenv("T2G_HELPER_AUTO_INSTALL", "0")

    app_module.settings = app_module.Settings.from_env()
    assert app_module.settings.auth_token == ""

    fake = FakeClusterSSH(app_module.settings)
    monkeypatch.setattr(app_module, "ClusterSSH", lambda _settings: fake)
    with TestClient(app_module.app) as test_client:
        resp = test_client.get("/status")
        assert resp.status_code == 200
        resp = test_client.get("/monitor")
        assert resp.status_code == 200


# ── SSH alias (~/.ssh/config, senza user@ né -p) ──────────────────────────────


def test_ssh_alias_no_user_no_port(monkeypatch, tmp_path):
    """T2G_SSH_HOST=gcluster senza user/porta: target = solo host, niente -p."""
    from fastapi.testclient import TestClient

    monkeypatch.setenv("T2G_AUTH_TOKEN", "test-token")
    monkeypatch.setenv("T2G_DB_PATH", str(tmp_path / "t2g_driver.db"))
    monkeypatch.setenv("T2G_DATA_DIR", str(tmp_path))
    monkeypatch.delenv("T2G_SSH_KEY_FILE", raising=False)
    monkeypatch.setenv("T2G_SSH_KEY_CONTENT", "")
    monkeypatch.setenv("T2G_SSH_HOST", "gcluster")
    monkeypatch.delenv("T2G_SSH_USER", raising=False)
    monkeypatch.delenv("T2G_SSH_PORT", raising=False)
    monkeypatch.setenv("T2G_HELPER_AUTO_INSTALL", "0")

    app_module.settings = app_module.Settings.from_env()
    assert app_module.settings.ssh_user == ""
    assert app_module.settings.ssh_port == 0

    # Cattura il comando ssh costruito da ClusterSSH reale
    built_cmds: list[list[str]] = []
    real_cls = app_module.ClusterSSH

    class SpySSH(real_cls):
        def __init__(self, settings):
            super().__init__(settings)
            built_cmds.append(list(self.base))

    monkeypatch.setattr(app_module, "ClusterSSH", lambda settings: SpySSH(settings))
    with TestClient(app_module.app) as test_client:
        resp = test_client.post("/tick", headers=AUTH)
        assert resp.status_code == 200
    cmd = built_cmds[0]
    assert cmd[-1] == "gcluster"  # target = solo alias
    assert not any(t == "-p" for t in cmd)  # nessuna porta esplicita
    assert not any("@" in t for t in cmd)  # nessun user@
