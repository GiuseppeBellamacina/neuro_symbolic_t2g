"""Test del driver esterno (remote/app.py) — nessuna rete reale.

ClusterSSH è sostituito da un doppio in memoria che simula lo stato del
cluster (.chain_state/) ed esegue le stesse mutazioni del helper lato
cluster (enqueue/rewrite_queue/pause/resume). Nessun subprocess ssh/sbatch.

Skip automatico se fastapi/httpx non sono installati (vedi pyproject dev
extras): serve solo a questo test, non alle dipendenze core del progetto.
"""

from __future__ import annotations

import base64
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import remote.app as app_module

AUTH = {"X-Auth-Token": "test-token"}

# Ordine ESATTO di remote/app.py:ABLATION_MODELS (= cluster/run_all.sh) -
# se cambia, aggiornare sia app.ABLATION_MODELS sia questa lista.
EXPECTED_ABLATION_MODELS: list[tuple[str, str, str]] = [
    (
        "baseline-zero-shot",
        "experiments/configs/qwen25-05b/baseline/zero-shot.yaml",
        "e",
    ),
    (
        "baseline-zero-shot-no-grammar",
        "experiments/configs/qwen25-05b/baseline/zero-shot-no-grammar.yaml",
        "e",
    ),
    ("baseline-few-shot", "experiments/configs/qwen25-05b/baseline/few-shot.yaml", "e"),
    ("sft-zero-shot", "experiments/configs/qwen25-05b/sft/zero-shot.yaml", "te"),
    ("grpo-zero-shot", "experiments/configs/qwen25-05b/grpo/zero-shot.yaml", "te"),
    ("grpo-few-shot", "experiments/configs/qwen25-05b/grpo/few-shot.yaml", "te"),
    (
        "sft-grpo-zero-shot",
        "experiments/configs/qwen25-05b/sft-grpo/zero-shot.yaml",
        "te",
    ),
    (
        "sft-grpo-few-shot",
        "experiments/configs/qwen25-05b/sft-grpo/few-shot.yaml",
        "te",
    ),
    (
        "ablations-decoding-no-grammar",
        "experiments/configs/qwen25-05b/ablations/decoding/no-grammar.yaml",
        "te",
    ),
    (
        "ablations-decoding-hot-rollout",
        "experiments/configs/qwen25-05b/ablations/decoding/hot-rollout.yaml",
        "te",
    ),
    (
        "ablations-rewards-edit-validity",
        "experiments/configs/qwen25-05b/ablations/rewards/edit-validity.yaml",
        "te",
    ),
    (
        "ablations-rewards-historical-stack",
        "experiments/configs/qwen25-05b/ablations/rewards/historical-stack.yaml",
        "te",
    ),
    (
        "ablations-loss-dr-grpo",
        "experiments/configs/qwen25-05b/ablations/loss/dr-grpo.yaml",
        "te",
    ),
    (
        "ablations-objectives-sft-allowed-mass",
        "experiments/configs/qwen25-05b/ablations/objectives/sft-allowed-mass.yaml",
        "te",
    ),
    (
        "ablations-objectives-sft-structured",
        "experiments/configs/qwen25-05b/ablations/objectives/sft-structured.yaml",
        "te",
    ),
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
    enqueue_batch / start_batch / rewrite_queue / pause / resume modificano
    la coda e lo stato; ogni `run` ritorna lo snapshot KEY=VALUE aggiornato.
    Supporta anche i subcomandi v2/v4: `monitor` (snapshot + LOG_TAIL_B64),
    `scancel` (kill + snapshot), `timeseries` e `results` (dati grafici).
    """

    def __init__(self, settings: app_module.Settings) -> None:
        self.settings = settings
        self.commands: list[str] = []
        self.queue: list[str] = []
        self.active_job: str = ""
        self.last_job: str = ""
        self.stopped = False
        self.errors: list[str] = []
        self.rc = 0
        self.stderr = ""
        self.status_text: str | None = None  # override totale dello stdout
        self.helper_missing_calls = 0  # invocazioni da simulare come "mancante"
        # v2: contenuto del log del job attivo (per `monitor`); se None,
        # `monitor` risponde senza LOG_TAIL_B64 (nessun log disponibile).
        self.log_lines: list[str] | None = None
        self.scancel_calls: list[str] = []
        # v4 (grafici): serie timeseries finta e risultati eval finti
        self.ts_lines: list[str] | None = None
        self.ts_job_id = "777"
        self.ts_job_type = "train"
        self.ts_total_steps = "900"
        self.results_dirs: list[str] = []
        self.results_dir_name = "qwen25-05b-sft-grpo"
        self.results_runs: list[tuple[str, dict]] = []

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
            f"ERRORS_COUNT={len(self.errors)}\n"
            f"ERRORS_TAIL={tail}\n"
        )
        if with_log:
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

    def _timeseries_result(self, _tag: str) -> app_module.SSHResult:
        if self.rc != 0:  # ssh fallita: nessun output utile
            return app_module.SSHResult(self.rc, "", self.stderr)
        if self.ts_lines is None:
            return app_module.SSHResult(
                0,
                "TS_MATCH=0\nTS_JOB_ID=\nTS_JOB_TYPE=\nTS_LOG_PATH=\n"
                "TS_TOTAL_STEPS=\nTS_LOG_B64=\n",
                "",
            )
        b64 = base64.b64encode("\n".join(self.ts_lines).encode()).decode()
        out = (
            "TS_MATCH=1\n"
            f"TS_JOB_ID={self.ts_job_id}\n"
            f"TS_JOB_TYPE={self.ts_job_type}\n"
            "TS_LOG_PATH=~/neuro_symbolic_t2g/logs/slurm-"
            f"{self.ts_job_type}-{self.ts_job_id}.log\n"
            f"TS_TOTAL_STEPS={self.ts_total_steps}\n"
            f"TS_LOG_B64={b64}\n"
        )
        return app_module.SSHResult(0, out, "")

    def _results_result(self, token: str) -> app_module.SSHResult:
        if self.rc != 0:  # ssh fallita: nessun output utile
            return app_module.SSHResult(self.rc, "", self.stderr)
        if not token:
            return app_module.SSHResult(
                0, "RESULTS_DIRS=" + "\x1f".join(self.results_dirs) + "\n", ""
            )
        if not self.results_runs:
            return app_module.SSHResult(0, "RESULTS_DIR=\nRESULTS_COUNT=0\n", "")
        lines = [f"RESULTS_DIR=experiments/results/{self.results_dir_name}"]
        for i, (run_id, payload) in enumerate(self.results_runs, 1):
            lines.append(f"RUN_ID_{i}={run_id}")
            lines.append(
                f"RUN_B64_{i}="
                + base64.b64encode(json.dumps(payload).encode()).decode()
            )
        lines.append(f"RESULTS_COUNT={len(self.results_runs)}")
        return app_module.SSHResult(0, "\n".join(lines) + "\n", "")

    def run(self, remote_cmd: str, timeout: int | None = None) -> app_module.SSHResult:
        self.commands.append(remote_cmd)
        if self.helper_missing_calls > 0:
            self.helper_missing_calls -= 1
            return app_module.SSHResult(0, "HELPER_MISSING=1\n", "")
        sub = self._subcommand(remote_cmd)
        if sub == "enqueue":
            self.queue.append(self._arg(remote_cmd))
        elif sub == "enqueue_batch":
            for entry in [e for e in self._arg(remote_cmd).split("\x1f") if e]:
                self.queue.append(entry)
        elif sub in ("start_batch", "start"):
            for entry in [e for e in self._arg(remote_cmd).split("\x1f") if e]:
                self.queue.append(entry)
            # il tick sottomette il primo job se la coda è libera (chain_tick)
            if self.queue and not self.active_job:
                entry = self.queue.pop(0)
                parts = entry.split(":")
                self.active_job = f"777|{parts[0]}-{parts[2]}|RUNNING"
                self.last_job = f"777:{entry}:0"
            return app_module.SSHResult(
                self.rc, self._snapshot(with_log=True), self.stderr
            )
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
        elif sub == "timeseries":
            return self._timeseries_result(self._arg(remote_cmd))
        elif sub == "results":
            return self._results_result(self._arg(remote_cmd))
        elif sub == "scancel":
            if not self.active_job:
                return app_module.SSHResult(1, "", "ERR_NO_ACTIVE_JOB=1")
            job_id = self.active_job.split("|")[0]
            self.scancel_calls.append(job_id)
            # il job resta RUNNING qualche secondo dopo lo scancel (come
            # squeue reale): lo snapshot viene stampato DOPO il kill
            return app_module.SSHResult(
                0, f"OK_SCANCEL={job_id}\n{self._snapshot(with_log=True)}", ""
            )
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
    assert test_client.get("/monitor").status_code == 401
    assert test_client.get("/logs").status_code == 401
    # endpoint v4 (lettura): stessa convenzione auth delle route esistenti
    assert test_client.get("/timeseries").status_code == 401
    assert test_client.get("/results").status_code == 401
    assert test_client.get("/health").status_code == 401
    assert test_client.get("/configs").status_code == 401
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
    entry = "train:experiments/configs/qwen25-05b/sft/zero-shot.yaml:a\x1feval:experiments/configs/qwen25-05b/sft/zero-shot.yaml:a"
    q = app_module._shq(entry)
    assert "\x1f" in q
    assert q.count("'") % 2 == 0  # quoting bilanciato: apri-chiudi
    assert q.startswith("'") and q.endswith("'")


# ── /status ────────────────────────────────────────────────────────────────────


def test_status_format_after_tick(client):
    test_client, fake = client
    fake.active_job = "12345|train-foo|RUNNING"
    fake.queue = ["train:experiments/configs/qwen25-05b/sft-grpo/few-shot.yaml:run1"]
    fake.last_job = (
        "12345:train:experiments/configs/qwen25-05b/sft-grpo/few-shot.yaml:run1:0"
    )

    resp = test_client.post("/tick", headers=AUTH)
    assert resp.status_code == 200
    body = resp.json()
    for key in (
        "active_job",
        "queue",
        "last_job",
        "stopped",
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
    assert body["queue"] == [
        "train:experiments/configs/qwen25-05b/sft-grpo/few-shot.yaml:run1"
    ]
    assert body["stopped"] is False
    assert body["cluster_reachable"] is True
    assert any(e["type"] == "tick" for e in body["events"])

    st = test_client.get("/status", headers=AUTH).json()
    assert st["queue"] == [
        "train:experiments/configs/qwen25-05b/sft-grpo/few-shot.yaml:run1"
    ]
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
        json={"type": "train", "config": "sft-grpo-few-shot", "tag": "run1"},
    )
    assert resp.status_code == 201
    assert (
        resp.json()["added"]
        == "train:experiments/configs/qwen25-05b/sft-grpo/few-shot.yaml:run1"
    )
    assert fake.queue == [
        "train:experiments/configs/qwen25-05b/sft-grpo/few-shot.yaml:run1"
    ]
    assert " enqueue " in fake.commands[-1]

    jobs = test_client.get("/jobs", headers=AUTH).json()
    assert jobs == [
        {
            "entry": "train:experiments/configs/qwen25-05b/sft-grpo/few-shot.yaml:run1",
            "type": "train",
            "config": "experiments/configs/qwen25-05b/sft-grpo/few-shot.yaml",
            "tag": "run1",
            "extra": None,
        }
    ]


def test_jobs_tag_derived_and_mode(client):
    test_client, _ = client
    resp = test_client.post(
        "/jobs", headers=AUTH, json={"type": "eval", "config": "sft-zero-shot"}
    )
    assert resp.status_code == 201
    assert (
        resp.json()["added"]
        == "eval:experiments/configs/qwen25-05b/sft/zero-shot.yaml:zero-shot"
    )

    resp = test_client.post(
        "/jobs",
        headers=AUTH,
        json={
            "type": "train",
            "config": "grpo-few-shot",
            "tag": "x",
            "mode": "--resume",
        },
    )
    assert resp.status_code == 201
    assert (
        resp.json()["added"]
        == "train:experiments/configs/qwen25-05b/grpo/few-shot.yaml:x:--resume"
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
    # Pin dell'ordine: app.ABLATION_MODELS deve coincidere con run_all.sh (MODELS)
    assert app_module.ABLATION_MODELS == EXPECTED_ABLATION_MODELS


def test_queue_replace_ablation_shortcut(client):
    test_client, fake = client
    resp = test_client.post("/queue", headers=AUTH, json={"ablation": True})
    assert resp.status_code == 200
    body = resp.json()
    assert body["count"] == 27  # 15 celle: 3 eval-only + 12 train+eval
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
                {"type": "eval", "config": "sft-zero-shot"},
                {"type": "train", "config": "grpo-few-shot", "tag": "exp1"},
            ]
        },
    )
    assert resp.status_code == 200
    assert resp.json()["queue"] == [
        "eval:experiments/configs/qwen25-05b/sft/zero-shot.yaml:zero-shot",
        "train:experiments/configs/qwen25-05b/grpo/few-shot.yaml:exp1",
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
                {"type": "train", "config": "sft-grpo-few-shot", "tag": "t1"},
                {"type": "eval", "config": "sft-grpo-few-shot", "tag": "t1"},
                {"type": "eval", "config": "sft-zero-shot", "tag": "t2"},
            ]
        },
    )
    resp = test_client.delete("/jobs/t1", headers=AUTH)
    assert resp.status_code == 200
    assert resp.json()["removed"] == 2
    assert resp.json()["status"]["queue"] == [
        "eval:experiments/configs/qwen25-05b/sft/zero-shot.yaml:t2"
    ]
    assert fake.queue == ["eval:experiments/configs/qwen25-05b/sft/zero-shot.yaml:t2"]


def test_delete_jobs_unknown_tag_no_rewrite(client):
    test_client, fake = client
    test_client.post("/queue", headers=AUTH, json={"ablation": True})
    n_before = len(fake.commands)
    resp = test_client.delete("/jobs/tag-inesistente", headers=AUTH)
    assert resp.status_code == 200
    assert resp.json()["removed"] == 0
    assert len(fake.commands) == n_before + 1  # solo lo status, niente rewrite
    assert len(fake.queue) == 27


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


# ── API v3: LIVE_STATUS + /jobs/batch ────────────────────────────────────────

LIVE_STATUS_BODY = {
    "phase": "sft",
    "step": 1400,
    "total_steps": 4934,
    "loss": 0.49,
    "reward": None,
    "lr": 1.5e-08,
    "eval_loss": 0.51,
    "eval_loss_best": 0.46,
    "epoch": 0.28,
    "eval_active": False,
    "eval_progress": None,
    "samples": [
        "it is simply what the transitional period entails .\n  [✓] pred: X-IT BE DESC-SIMPLY …"
    ],
    "samples_kind": "sft",
    "note": "SFT training",
    "ts": "2026-08-27T14:30:00",
    "hostname": "gnode10",
    "pid": 12345,
}


def _fake_with_live(fake, live: dict | None) -> None:
    """Configura il fake per includere LIVE_STATUS nel subcomando monitor."""
    fake.live_status = live

    orig_run = fake.run

    def _run(remote_cmd, timeout=None):
        result = orig_run(remote_cmd, timeout)
        sub = fake._subcommand(remote_cmd)
        if sub in ("monitor", "start_batch", "start") and live is not None:
            line = f"LIVE_STATUS={json.dumps(live, ensure_ascii=False)}\n"
            result = app_module.SSHResult(
                result.rc, result.stdout + line, result.stderr
            )
        return result

    fake.run = _run


def test_monitor_uses_live_status_when_present(client):
    """LIVE_STATUS valido → job_detail da live (source=live) + samples dal live."""
    test_client, fake = client
    fake.active_job = "12345|train-sft-grpo|RUNNING"
    fake.log_lines = TRAIN_LOG_LINES
    _fake_with_live(fake, LIVE_STATUS_BODY)

    resp = test_client.get("/monitor", headers=AUTH)
    assert resp.status_code == 200
    body = resp.json()
    detail = body["job_detail"]
    assert detail["source"] == "live"
    assert detail["phase"] == "sft"
    assert detail["sft_active"] is True
    assert detail["step"] == 1400
    assert detail["total_steps"] == 4934
    assert detail["sft_eval_loss_best"] == 0.46
    # samples arrivano dal live status (formattati dal produttore)
    assert any("transitional" in s for s in body["samples"])


def test_monitor_falls_back_to_log_without_live_status(client):
    """Senza LIVE_STATUS → job_detail dal parsing del log (source=log)."""
    test_client, fake = client
    fake.active_job = "12345|train-sft-grpo|RUNNING"
    fake.log_lines = TRAIN_LOG_LINES
    _fake_with_live(fake, None)  # nessun live status

    resp = test_client.get("/monitor", headers=AUTH)
    assert resp.status_code == 200
    detail = resp.json()["job_detail"]
    assert detail is not None
    assert detail.get("source") in (None, "log")


def test_monitor_live_status_with_broken_json_falls_back(client):
    """LIVE_STATUS non-JSON → ignorato, fallback al log senza crash."""
    test_client, fake = client
    fake.active_job = "12345|train-sft-grpo|RUNNING"
    fake.log_lines = TRAIN_LOG_LINES

    orig_run = fake.run

    def _run(remote_cmd, timeout=None):
        result = orig_run(remote_cmd, timeout)
        if fake._subcommand(remote_cmd) == "monitor":
            result = app_module.SSHResult(
                result.rc, result.stdout + "LIVE_STATUS={not json}\n", result.stderr
            )
        return result

    fake.run = _run
    resp = test_client.get("/monitor", headers=AUTH)
    assert resp.status_code == 200
    assert resp.json()["job_detail"] is not None


def test_jobs_batch_enqueues_in_order_and_ticks(client):
    """POST /jobs/batch: train+eval accodati in ordine + tick; started_now."""
    test_client, fake = client

    resp = test_client.post(
        "/jobs/batch",
        headers=AUTH,
        json={
            "jobs": [
                {"type": "train", "config": "sft-grpo-few-shot"},
                {"type": "eval", "config": "sft-grpo-few-shot"},
            ],
            "start_now": True,
        },
    )
    assert resp.status_code == 201
    body = resp.json()
    assert body["started_now"] is True
    assert body["active_job"]["name"] == "train-few-shot"
    # train consumato dal tick, eval in coda
    assert fake.queue == [
        "eval:experiments/configs/qwen25-05b/sft-grpo/few-shot.yaml:few-shot"
    ]
    assert len(body["queued"]) == 2
    # PROVA riduzione round-trip: erano N+2 ssh seriali → ora 1 sola
    # connessione; le entry viaggiano separate da \x1f dentro il comando
    assert len(fake.commands) == 1
    assert "cluster_helper.sh start_batch" in fake.commands[0]
    assert fake.commands[0].count("\x1f") == 1


def test_jobs_batch_atomic_validation(client):
    """Config invalido → 422 e NESSUN job accodato (atomicità)."""
    test_client, fake = client

    resp = test_client.post(
        "/jobs/batch",
        headers=AUTH,
        json={
            "jobs": [
                {"type": "train", "config": "sft-grpo-few-shot"},
                {"type": "train", "config": "config_inesistente"},
            ]
        },
    )
    assert resp.status_code == 422
    assert fake.queue == []  # niente scritture sul cluster


def test_jobs_batch_empty_rejected(client):
    test_client, _ = client
    resp = test_client.post("/jobs/batch", headers=AUTH, json={"jobs": []})
    assert resp.status_code == 422


def test_jobs_batch_without_start_now_only_enqueues(client):
    """start_now=False → solo enqueue, nessun tick."""
    test_client, fake = client

    resp = test_client.post(
        "/jobs/batch",
        headers=AUTH,
        json={
            "jobs": [{"type": "eval", "config": "sft-zero-shot"}],
            "start_now": False,
        },
    )
    assert resp.status_code == 201
    assert resp.json()["started_now"] is False
    assert len(fake.queue) == 1  # solo il job richiesto, nessuna espansione
    assert any("enqueue_batch" in c for c in fake.commands)
    assert not any("start_batch" in c for c in fake.commands)
    assert len(fake.commands) == 1  # 1 sola ssh anche senza tick


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
    fake.active_job = "12345|train-sft-grpo|RUNNING"
    fake.log_lines = TRAIN_LOG_LINES

    resp = test_client.get("/monitor", headers=AUTH)
    assert resp.status_code == 200
    body = resp.json()
    detail = body["job_detail"]
    assert detail is not None
    assert detail["id"] == "12345"
    assert detail["name"] == "train-sft-grpo"
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
    fake.active_job = "999|eval-sft-grpo|RUNNING"
    fake.log_lines = ["Evaluating 100/8771 samples (seeded sample)", "  Pass@1: 0.1234"]

    resp = test_client.get("/monitor", headers=AUTH)
    assert resp.status_code == 200
    body = resp.json()
    assert body["job_detail"]["name"] == "eval-sft-grpo"


def test_start_job_enqueues_and_ticks(client):
    """POST /jobs/start: enqueue + tick; started_now se il job è attivo."""
    test_client, fake = client

    resp = test_client.post(
        "/jobs/start",
        headers=AUTH,
        json={"type": "train", "config": "sft-grpo-few-shot"},
    )
    assert resp.status_code == 201
    body = resp.json()
    assert body["started_now"] is True
    assert body["active_job"]["name"] == "train-few-shot"
    # PROVA riduzione round-trip: erano 3 ssh seriali (enqueue, tick, monitor)
    # → ora è UNA sola connessione con il subcomando combinato start_batch
    assert len(fake.commands) == 1
    assert "cluster_helper.sh start_batch" in fake.commands[0]


def test_start_job_enqueued_when_busy(client):
    """Job attivo diverso → started_now False (resta in coda)."""
    test_client, fake = client
    fake.active_job = "111|train-altro|RUNNING"

    resp = test_client.post(
        "/jobs/start",
        headers=AUTH,
        json={"type": "train", "config": "sft-grpo-few-shot"},
    )
    assert resp.status_code == 201
    assert resp.json()["started_now"] is False
    assert fake.queue == [
        "train:experiments/configs/qwen25-05b/sft-grpo/few-shot.yaml:few-shot"
    ]
    assert len(fake.commands) == 1  # anche a coda occupata: 1 sola ssh


def test_kill_cancels_active_job(client):
    """POST /kill: scancel dell'id attivo; snapshot di ritorno."""
    test_client, fake = client
    fake.active_job = "12345|train-sft-grpo|RUNNING"

    resp = test_client.post("/kill", headers=AUTH)
    assert resp.status_code == 200
    assert fake.scancel_calls == ["12345"]
    # PROVA riduzione round-trip: erano 2 ssh (scancel + monitor) → ora 1 sola
    assert len(fake.commands) == 1
    assert "cluster_helper.sh scancel" in fake.commands[0]


def test_kill_without_active_job_409(client):
    test_client, fake = client
    fake.active_job = ""
    resp = test_client.post("/kill", headers=AUTH)
    assert resp.status_code == 409
    assert "Nessun job attivo" in resp.json()["detail"]


def test_logs_endpoint_returns_tail(client):
    test_client, fake = client
    fake.active_job = "12345|train-sft-grpo|RUNNING"
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


# ── Cache con TTL di /monitor ──────────────────────────────────────────────────


def test_monitor_cache_hit_avoids_ssh(client):
    """2ª GET /monitor entro il TTL: risposta dalla cache, ZERO ssh nuove."""
    test_client, fake = client
    fake.active_job = "12345|train-sft-grpo|RUNNING"
    fake.log_lines = TRAIN_LOG_LINES

    r1 = test_client.get("/monitor", headers=AUTH)
    assert r1.status_code == 200
    assert r1.json()["source"] == "live"
    assert r1.json()["age_seconds"] == 0.0
    first_ts = r1.json()["ts"]
    n_after_first = len(fake.commands)

    r2 = test_client.get("/monitor", headers=AUTH)
    assert r2.status_code == 200
    assert len(fake.commands) == n_after_first  # nessuna ssh nuova: cache hit
    body = r2.json()
    assert body["source"] == "cache"
    assert body["age_seconds"] >= 0.0
    assert body["snapshot_ts"] == first_ts  # stesso snapshot, provenienza onesta
    # le chiavi esistenti non cambiano forma (compatibilità client)
    assert body["active_job"]["name"] == "train-sft-grpo"
    assert body["job_detail"]["step"] == 105


def test_monitor_cache_stale_refreshes_via_ssh(client):
    """Snapshot stantio (età > TTL) → nuova ssh, risposta live."""
    test_client, fake = client
    test_client.get("/monitor", headers=AUTH)
    n = len(fake.commands)
    # invecchia artificialmente lo snapshot oltre il TTL (niente sleep nei test)
    app_module._MONITOR_CACHE["ts"] -= app_module.settings.monitor_cache_ttl + 1
    r = test_client.get("/monitor", headers=AUTH)
    assert r.status_code == 200
    assert len(fake.commands) == n + 1  # refresh via ssh
    assert r.json()["source"] == "live"


def test_monitor_refresh_param_forces_ssh(client):
    """?refresh=1 → ssh esplicita anche con cache fresca (tasto 'aggiorna')."""
    test_client, fake = client
    test_client.get("/monitor", headers=AUTH)
    n = len(fake.commands)
    r = test_client.get("/monitor", headers=AUTH, params={"refresh": "1"})
    assert r.status_code == 200
    assert len(fake.commands) == n + 1
    assert r.json()["source"] == "live"


def test_monitor_ttl_configurable_zero_disables_cache(client, monkeypatch):
    """TTL 0 → ogni GET fa ssh (comportamento pre-cache ottenibile via env)."""
    test_client, fake = client
    monkeypatch.setattr(app_module.settings, "monitor_cache_ttl", 0)
    test_client.get("/monitor", headers=AUTH)
    n = len(fake.commands)
    test_client.get("/monitor", headers=AUTH)
    assert len(fake.commands) == n + 1


def test_monitor_stale_cache_served_when_cluster_down(client):
    """Cluster giù + cache stantia → 200 cache con fetch_error esplicito.

    Se esistono dati validi (seppur vecchi) non c'è motivo di dare 502; e i
    dati vecchi NON sono mai spacciati per freschi (source/age/fetch_error).
    """
    test_client, fake = client
    test_client.get("/monitor", headers=AUTH)
    app_module._MONITOR_CACHE["ts"] -= app_module.settings.monitor_cache_ttl + 1
    fake.rc = 255
    fake.stderr = "ssh: connect to host unit.test port 22: Connection refused"
    r = test_client.get("/monitor", headers=AUTH)
    assert r.status_code == 200
    body = r.json()
    assert body["source"] == "cache"
    assert "fetch_error" in body
    assert "Connection refused" in body["fetch_error"]


# ── Conteggio SSH via subprocess.run mockato (ClusterSSH REALE) ───────────────


def test_ssh_roundtrips_counted_via_subprocess_mock(monkeypatch, tmp_path):
    """Conteggio CHIAMATE subprocess.run con ClusterSSH reale (ssh mockata):
    prova diretta della riduzione dei round-trip per operazione.

    Prima: /monitor 1 ssh per poll · /jobs/start 3 · /jobs/batch N+2 · /kill 2.
    Dopo: 1 per tutte (0 per /monitor servito dalla cache).
    """
    from fastapi.testclient import TestClient

    monkeypatch.setenv("T2G_AUTH_TOKEN", "test-token")
    monkeypatch.setenv("T2G_DB_PATH", str(tmp_path / "t2g_driver.db"))
    monkeypatch.setenv("T2G_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("T2G_SSH_KEY_FILE", str(tmp_path / "ssh_key"))
    monkeypatch.setenv("T2G_SSH_KEY_CONTENT", "")
    monkeypatch.setenv("T2G_SSH_HOST", "unit.test")
    monkeypatch.setenv("T2G_SSH_USER", "tester")
    monkeypatch.setenv("T2G_HELPER_AUTO_INSTALL", "0")
    (tmp_path / "ssh_key").write_text("chiave-di-test", encoding="utf-8")
    app_module.settings = app_module.Settings.from_env()

    status_out = (
        "STATUS_OK=1\n"
        "ACTIVE_JOB=777|train-few-shot|RUNNING\n"
        "QUEUE=\nQUEUE_COUNT=0\n"
        "LAST_JOB=777:train:cfg:few-shot:0\n"
        "STOPPED=0\nERRORS_COUNT=0\nERRORS_TAIL=[]\n"
        "LOG_PATH=\nLOG_TAIL_B64=\n"
    )
    ssh_calls: list[list[str]] = []

    def fake_subprocess_run(args, **kwargs):
        ssh_calls.append(list(args))
        cmd = " ".join(args)
        if "scancel" in cmd:
            return SimpleNamespace(
                returncode=0, stdout=f"OK_SCANCEL=777\n{status_out}", stderr=""
            )
        return SimpleNamespace(returncode=0, stdout=status_out, stderr="")

    monkeypatch.setattr(app_module.subprocess, "run", fake_subprocess_run)

    with TestClient(app_module.app) as test_client:
        # /jobs/start: 1 sola invocazione ssh (prima: 3)
        ssh_calls.clear()
        r = test_client.post(
            "/jobs/start",
            headers=AUTH,
            json={"type": "train", "config": "sft-grpo-few-shot"},
        )
        assert r.status_code == 201
        assert r.json()["started_now"] is True
        assert len(ssh_calls) == 1
        assert "start_batch" in " ".join(ssh_calls[0])

        # /jobs/batch (3 job): 1 sola invocazione (prima: N+2 = 5)
        ssh_calls.clear()
        r = test_client.post(
            "/jobs/batch",
            headers=AUTH,
            json={
                "jobs": [
                    {"type": "train", "config": "sft-grpo-few-shot"},
                    {"type": "eval", "config": "sft-grpo-few-shot"},
                    {"type": "eval", "config": "sft-zero-shot"},
                ],
                "start_now": True,
            },
        )
        assert r.status_code == 201
        assert len(ssh_calls) == 1

        # /kill: 1 sola invocazione (prima: 2)
        ssh_calls.clear()
        r = test_client.post("/kill", headers=AUTH)
        assert r.status_code == 200
        assert len(ssh_calls) == 1

        # /monitor con refresh esplicito: 1 sola invocazione
        ssh_calls.clear()
        r = test_client.get("/monitor", headers=AUTH, params={"refresh": "1"})
        assert r.status_code == 200
        assert len(ssh_calls) == 1

        # /monitor senza refresh a cache fresca: ZERO invocazioni
        ssh_calls.clear()
        r = test_client.get("/monitor", headers=AUTH)
        assert r.status_code == 200
        assert len(ssh_calls) == 0


# ── Connessione SQLite unica ───────────────────────────────────────────────────


def test_single_sqlite_connection_reused(client, monkeypatch):
    """Una sola connessione SQLite riusata: prima ogni _kv_get/_kv_set apriva
    e chiudeva una connessione nuova (decine per tick)."""
    test_client, _ = client
    opens: list = []
    real_connect = app_module.sqlite3.connect

    def counting_connect(*args, **kwargs):
        opens.append(args)
        return real_connect(*args, **kwargs)

    monkeypatch.setattr(app_module.sqlite3, "connect", counting_connect)

    test_client.post("/tick", headers=AUTH)  # ~10 scritture kv + eventi
    test_client.get("/status", headers=AUTH)
    test_client.get("/health", headers=AUTH)
    test_client.get("/configs", headers=AUTH)
    # la connessione è già aperta dal lifespan → 0 nuove aperture
    assert len(opens) == 0


# ── API v4: /timeseries ────────────────────────────────────────────────────────


def _ts_lines() -> list[str]:
    """40 righe KV come quelle stampate da src/training/callbacks.py."""
    return [
        f"  step={i * 10}  loss={1.0 - i * 0.01:.4f}  reward={0.1 * i:.4f}  "
        f"learning_rate=0.00003  kl=0.00{i % 10}"
        for i in range(1, 41)
    ]


def _prime_known_tags(test_client, fake, tag: str) -> None:
    """Popola la cache DB (active_job) così il tag è 'noto' a /timeseries."""
    fake.active_job = f"777|train-{tag}|RUNNING"
    r = test_client.post("/tick", headers=AUTH)
    assert r.status_code == 200


def test_timeseries_parses_points(client):
    """Punti (step, value) dalle righe KV; total_steps dal helper; 1 sola ssh."""
    test_client, fake = client
    fake.ts_lines = _ts_lines()
    _prime_known_tags(test_client, fake, "sft-grpo")

    r = test_client.get(
        "/timeseries",
        headers=AUTH,
        params={"tag": "sft-grpo", "metric": "loss"},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["tag"] == "sft-grpo"
    assert body["metric"] == "loss"
    assert body["points"][0] == {"step": 10, "value": 0.99}
    assert body["points"][-1]["step"] == 400
    assert body["current_step"] == 400
    assert body["total_steps"] == 900
    assert body["source"] == "live"
    # il tick iniziale + 1 sola ssh timeseries per costruire l'intera serie
    assert len([c for c in fake.commands if "timeseries" in c]) == 1


def test_timeseries_all_metrics_from_same_cache(client):
    """Le 4 metriche arrivano dalla STESSA serie cached: una ssh sola."""
    test_client, fake = client
    fake.ts_lines = _ts_lines()
    _prime_known_tags(test_client, fake, "sft-grpo")

    first = test_client.get(
        "/timeseries", headers=AUTH, params={"tag": "sft-grpo", "metric": "loss"}
    )
    assert first.status_code == 200
    n = len([c for c in fake.commands if "timeseries" in c])

    for metric, value in (("reward", 0.1), ("lr", 0.00003), ("kl", 0.001)):
        r = test_client.get(
            "/timeseries", headers=AUTH, params={"tag": "sft-grpo", "metric": metric}
        )
        assert r.status_code == 200
        body = r.json()
        assert body["metric"] == metric
        assert body["source"] == "cache"
        assert body["points"][0]["value"] == value
    assert len([c for c in fake.commands if "timeseries" in c]) == n


def test_timeseries_limit_subsample_keeps_latest(client):
    """limit sottocampiona uniformemente: estremi e ultimo punto preservati."""
    test_client, fake = client
    fake.ts_lines = _ts_lines()  # 40 punti
    _prime_known_tags(test_client, fake, "sft-grpo")

    r = test_client.get(
        "/timeseries", headers=AUTH, params={"tag": "sft-grpo", "limit": 10}
    )
    assert r.status_code == 200
    pts = r.json()["points"]
    assert len(pts) == 10
    assert pts[0]["step"] == 10  # primo preservato
    assert pts[-1]["step"] == 400  # ultimo (più recente) SEMPRE incluso
    steps = [p["step"] for p in pts]
    assert steps == sorted(steps)
    assert len(set(steps)) == len(steps)


def test_timeseries_unknown_tag_404_without_ssh(client):
    """Tag non noto e cache vuota → 404 SENZA sprecare una connessione ssh."""
    test_client, fake = client
    r = test_client.get("/timeseries", headers=AUTH, params={"tag": "mai-esistito"})
    assert r.status_code == 404
    assert not fake.commands  # il tag non è tra i noti: zero ssh


def test_timeseries_finished_job_cache_survives_cluster_down(client):
    """Serie cached di un job finito + cluster giù → 200 cache (dati immutabili)."""
    test_client, fake = client
    fake.ts_lines = _ts_lines()
    _prime_known_tags(test_client, fake, "sft-grpo")
    test_client.get("/timeseries", headers=AUTH, params={"tag": "sft-grpo"})

    # cache stantia + cluster irraggiungibile: la serie finita non cambia mai
    app_module._TS_CACHE["sft-grpo"]["ts"] -= (
        app_module.settings.timeseries_cache_ttl + 1
    )
    fake.rc = 255
    fake.stderr = "ssh: connect refused"
    r = test_client.get("/timeseries", headers=AUTH, params={"tag": "sft-grpo"})
    assert r.status_code == 200
    body = r.json()
    assert body["source"] == "cache"
    assert body["points"], "la serie cached non deve svuotarsi"
    assert "fetch_error" in body  # la provenienza resta onesta


def test_timeseries_metric_and_tag_validation(client):
    test_client, _ = client
    assert (
        test_client.get(
            "/timeseries", headers=AUTH, params={"tag": "x", "metric": "speed"}
        ).status_code
        == 422
    )
    assert (
        test_client.get(
            "/timeseries", headers=AUTH, params={"tag": "tag con spazi"}
        ).status_code
        == 422
    )


# ── API v4: /results ───────────────────────────────────────────────────────────

RESULTS_METRICS = {
    "rouge_l_mean": 0.42,
    "exact_match": 0.11,
    "validity_rate": 0.97,
    "pass_at_1": 0.35,
    "gloss_f1_micro": 0.78,
    "bleu_corpus": 0.31,
    "chrf_corpus": 44.2,
    "reward_breakdown": {
        "grammar": 0.9,
        "edit": 0.5,
        "historical": 0.4,
        "validity": 0.8,
        "gloss": 0.6,
        "length": 0.7,
        "format": 0.95,
    },
    "pass_at_k": {
        "pass@1": 0.35,
        "pass@2": 0.4,
        "pass@3": 0.44,
        "pass@4": 0.47,
        "pass@5": 0.5,
    },
    "error_distribution": {"TIMEOUT": 3, "OOM": 1},
    "detailed_metrics": {"rouge_l_percentiles": {"p50": 0.41, "p90": 0.55}},
    "difficulty_breakdown": {
        "simple": {"rouge_l_mean": 0.5},
        "medium": {"rouge_l_mean": 0.4},
        "hard": {"rouge_l_mean": 0.3},
    },
    "prompting": {"mode": "few-shot", "source": "config"},
}


def test_results_returns_runs_with_full_metrics(client):
    """GET /results?config=...: run + metriche complete (contratto client)."""
    test_client, fake = client
    fake.results_runs = [
        ("run_20260904_000559", RESULTS_METRICS),
        ("run_20260901_233552", {"rouge_l_mean": 0.39}),
    ]

    r = test_client.get(
        "/results", headers=AUTH, params={"config": "qwen25-05b-sft-grpo"}
    )
    assert r.status_code == 200
    body = r.json()
    assert body["config"] == "qwen25-05b-sft-grpo"
    assert body["results_dir"] == "experiments/results/qwen25-05b-sft-grpo"
    assert body["source"] == "live"
    assert len(body["runs"]) == 2
    assert body["runs"][0]["run_id"] == "run_20260904_000559"
    m = body["runs"][0]["metrics"]
    for key in (
        "rouge_l_mean",
        "exact_match",
        "validity_rate",
        "pass_at_1",
        "gloss_f1_micro",
        "bleu_corpus",
        "chrf_corpus",
        "reward_breakdown",
        "pass_at_k",
        "error_distribution",
        "detailed_metrics",
        "difficulty_breakdown",
        "prompting",
    ):
        assert key in m, f"manca {key} nel contratto /results"
    assert m["detailed_metrics"]["rouge_l_percentiles"]["p50"] == 0.41
    assert m["pass_at_k"]["pass@5"] == 0.5
    assert m["prompting"] == {"mode": "few-shot", "source": "config"}
    assert len([c for c in fake.commands if "results" in c]) == 1


def test_results_cached_second_call_no_ssh(client):
    """Seconda chiamata entro TTL: cache SQLite, zero ssh (file immutabili)."""
    test_client, fake = client
    fake.results_runs = [("run_1", RESULTS_METRICS)]
    test_client.get("/results", headers=AUTH, params={"config": "qwen25-05b-sft-grpo"})
    n = len(fake.commands)
    r = test_client.get(
        "/results", headers=AUTH, params={"config": "qwen25-05b-sft-grpo"}
    )
    assert r.status_code == 200
    assert len(fake.commands) == n  # cache hit
    body = r.json()
    assert body["source"] == "cache"
    assert body["runs"][0]["metrics"]["rouge_l_mean"] == 0.42


def test_results_stale_cache_served_when_cluster_down(client):
    """Cluster giù ma risultati già in cache → 200 cache (non 502)."""
    test_client, fake = client
    fake.results_runs = [("run_1", RESULTS_METRICS)]
    test_client.get("/results", headers=AUTH, params={"config": "qwen25-05b-sft-grpo"})
    # invalida la cache + cluster giù
    fake.rc = 255
    fake.stderr = "ssh: connect refused"
    conn = app_module._db_conn()
    conn.execute(
        "UPDATE kv SET value = ? WHERE key = ?",
        (
            json.dumps(
                {
                    "ts": 0,
                    "dir": "experiments/results/qwen25-05b-sft-grpo",
                    "runs": [{"run_id": "run_1", "metrics": RESULTS_METRICS}],
                }
            ),
            "results:qwen25-05b-sft-grpo",
        ),
    )
    conn.commit()
    r = test_client.get(
        "/results", headers=AUTH, params={"config": "qwen25-05b-sft-grpo"}
    )
    assert r.status_code == 200
    body = r.json()
    assert body["source"] == "cache"
    assert body["runs"][0]["metrics"]["rouge_l_mean"] == 0.42
    assert "fetch_error" in body


def test_results_unknown_config_404(client):
    test_client, _ = client
    r = test_client.get("/results", headers=AUTH, params={"config": "no-such-config"})
    assert r.status_code == 404
    assert "GET /results" in r.json()["detail"]


def test_results_discovery_lists_dirs(client):
    """GET /results senza config: elenco delle dir disponibili sul cluster."""
    test_client, fake = client
    fake.results_dirs = ["t2g-zero-shot", "qwen25-05b-sft-grpo"]
    r = test_client.get("/results", headers=AUTH)
    assert r.status_code == 200
    assert r.json()["results_dirs"] == ["qwen25-05b-sft-grpo", "t2g-zero-shot"]


# ── API v4: /health e /configs ─────────────────────────────────────────────────


def test_health_reports_state_without_ssh(client):
    """GET /health: DB, pausa, età snapshot, ultimo errore — ZERO ssh."""
    test_client, fake = client
    _prime_known_tags(test_client, fake, "sft-grpo")
    n = len(fake.commands)

    r = test_client.get("/health", headers=AUTH)
    assert r.status_code == 200
    assert len(fake.commands) == n  # nessuna ssh: solo DB
    body = r.json()
    assert body["ok"] is True
    assert body["db_ok"] is True
    assert body["cluster_reachable"] is True
    assert body["paused"] is False
    assert body["last_tick_at"]
    assert body["snapshot_age_seconds"] is not None
    assert body["last_error"] is None


def test_health_reports_pause_and_last_error(client):
    """Catena in pausa + cluster giù: /health mostra lo stato ONESTO."""
    test_client, fake = client
    test_client.post("/pause", headers=AUTH)
    fake.rc = 255
    fake.stderr = "ssh: connect to host unit.test port 22: Connection refused"
    test_client.post("/tick", headers=AUTH)  # fallisce → evento error

    r = test_client.get("/health", headers=AUTH)
    assert r.status_code == 200
    body = r.json()
    assert body["paused"] is True
    assert body["cluster_reachable"] is False
    assert body["last_error"] is not None
    assert body["ok"] is True  # il DB è sano anche a cluster giù


def test_configs_exposes_known_config_map(client):
    """GET /configs: la mappa nome→path — il client può rimuovere la sua copia."""
    test_client, _ = client
    r = test_client.get("/configs", headers=AUTH)
    assert r.status_code == 200
    body = r.json()
    names = [c["name"] for c in body["configs"]]
    assert len(names) == len(app_module.CONFIG_MAP) == 15
    assert "sft-grpo-zero-shot" in names
    assert all(c["path"] for c in body["configs"])
