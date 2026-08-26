"""Driver esterno della catena T2G su gcluster (FastAPI + SQLite).

Deployato su Render (free tier), tickato da cronjob.org (POST /tick ogni 5
min). Ogni tick: (1) ssh sul login node con la chiave da env; (2) esegue il
helper lato cluster `cluster_helper.sh tick`, che chiama il tick one-shot
idempotente `chain_tick.sh --quiet`; (3) sincronizza lo snapshot KEY=VALUE
machine-readable in un DB SQLite locale (cache + diario eventi).

FONTE DI VERITÀ = stato sul cluster (`.chain_state/`): il filesystem di Render
è effimero, quindi il DB è solo cache/diario riletto a ogni tick. GET /status
risponde anche a cluster irraggiungibile (ultimo stato noto +
`cluster_reachable: false`). Sicurezza: mai loggare token/chiave; ogni route
richiede X-Auth-Token; chiave da T2G_SSH_KEY_CONTENT scritta su file 0600.
La chiave è OPZIONALE: se non sono impostate né T2G_SSH_KEY_CONTENT né
T2G_SSH_KEY_FILE, ssh/scp usano l'autenticazione di default dell'utente
(ssh-agent / identità default / ~/.ssh/config) — il caso del deploy LOCALE
di test, esattamente come fa sync_cluster.ps1.
"""

from __future__ import annotations

import json
import logging
import os
import re
import secrets
import sqlite3
import subprocess
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterator, Literal

from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel

_log = logging.getLogger("uvicorn.error")

# ── Configurazione (env vars; obbligatorie: T2G_AUTH_TOKEN, T2G_SSH_USER) ──


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except (TypeError, ValueError):
        return default


@dataclass
class Settings:
    ssh_host: str
    ssh_user: str
    ssh_port: int
    ssh_key_file: str | None
    ssh_key_content: str
    ssh_known_hosts: str
    ssh_timeout: int
    auth_token: str
    db_path: str
    data_dir: str
    helper_auto_install: bool

    @classmethod
    def from_env(cls) -> "Settings":
        data_dir = os.environ.get("T2G_DATA_DIR", str(Path.cwd() / "data"))
        key_content = os.environ.get("T2G_SSH_KEY_CONTENT", "")
        key_file = os.environ.get("T2G_SSH_KEY_FILE")
        if not key_file and key_content:
            # Contenuto incollato in env → scritto su file all'avvio.
            key_file = str(Path(data_dir) / "ssh_key")
        return cls(
            ssh_host=os.environ.get("T2G_SSH_HOST", "gcluster.dmi.unict.it"),
            ssh_user=os.environ.get("T2G_SSH_USER", ""),
            ssh_port=_env_int("T2G_SSH_PORT", 22),
            # None = nessun `-i`: ssh-agent / identità default / ~/.ssh/config
            ssh_key_file=key_file,
            ssh_key_content=key_content,
            ssh_known_hosts=os.environ.get("T2G_SSH_KNOWN_HOSTS", str(Path(data_dir) / "known_hosts")),
            ssh_timeout=_env_int("T2G_SSH_TIMEOUT", 30),
            auth_token=os.environ.get("T2G_AUTH_TOKEN", ""),
            db_path=os.environ.get("T2G_DB_PATH", str(Path(data_dir) / "t2g_driver.db")),
            data_dir=data_dir,
            helper_auto_install=os.environ.get("T2G_HELPER_AUTO_INSTALL", "1")
            not in ("", "0", "false", "False"),
        )


settings = Settings.from_env()

# ── Config noti (nome → path). Stessi nomi validi della coda di run_all.sh ──

CONFIG_MAP: dict[str, str] = {
    "grpo_optimal": "experiments/configs/t2g/grpo_optimal.yaml",
    "grpo_qwen05": "experiments/configs/t2g/grpo_qwen05.yaml",
    "sft": "experiments/configs/t2g/sft.yaml",
    "grpo_experimental_all": "experiments/configs/t2g/grpo_experimental_all.yaml",
    "zero_shot": "experiments/configs/t2g/ablation/zero_shot.yaml",
    "zero_shot_grammar": "experiments/configs/t2g/ablation/zero_shot_grammar.yaml",
    "grpo_no_grammar": "experiments/configs/t2g/ablation/grpo_no_grammar.yaml",
    "grpo_no_sft": "experiments/configs/t2g/ablation/grpo_no_sft.yaml",
    "grpo_pda": "experiments/configs/t2g/ablation/grpo_pda.yaml",
    "grpo_pda_lookahead": "experiments/configs/t2g/ablation/grpo_pda_lookahead.yaml",
    "grpo_soft_viterbi": "experiments/configs/t2g/ablation/grpo_soft_viterbi.yaml",
    "grpo_verifier_scaled": "experiments/configs/t2g/ablation/grpo_verifier_scaled.yaml",
}

CONFIG_PATHS: set[str] = set(CONFIG_MAP.values())

# Ablation study: ordine ESATTO di run_all.sh:134-147 (TAG:CONFIG:MODE).
# MODE: e = eval-only · te = train+eval → 12 righe diventano 22 entry di coda.
ABLATION_MODELS: list[tuple[str, str, str]] = [
    ("zero-shot", "experiments/configs/t2g/ablation/zero_shot.yaml", "e"),
    ("zero-shot-gram", "experiments/configs/t2g/ablation/zero_shot_grammar.yaml", "e"),
    ("grpo-no-grammar", "experiments/configs/t2g/ablation/grpo_no_grammar.yaml", "te"),
    ("grpo-no-sft", "experiments/configs/t2g/ablation/grpo_no_sft.yaml", "te"),
    ("grpo-grammar", "experiments/configs/t2g/grpo_qwen05.yaml", "te"),
    ("sft", "experiments/configs/t2g/sft.yaml", "te"),
    ("grpo-pda", "experiments/configs/t2g/ablation/grpo_pda.yaml", "te"),
    ("grpo-pda-lookahead", "experiments/configs/t2g/ablation/grpo_pda_lookahead.yaml", "te"),
    ("grpo-soft-viterbi", "experiments/configs/t2g/ablation/grpo_soft_viterbi.yaml", "te"),
    ("grpo-verifier", "experiments/configs/t2g/ablation/grpo_verifier_scaled.yaml", "te"),
    ("grpo-experimental-all", "experiments/configs/t2g/grpo_experimental_all.yaml", "te"),
    ("grpo-optimal", "experiments/configs/t2g/grpo_optimal.yaml", "te"),
]

HELPER_NAME = "cluster_helper.sh"  # file locale in remote/ (per auto-install scp)
HELPER_REMOTE = "~/neuro_symbolic_t2g/cluster/cluster_helper.sh"  # path sul cluster

_TAG_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}\Z")
_EXTRA_RE = re.compile(r"[-A-Za-z0-9][A-Za-z0-9 ._=/:,-]{0,127}\Z")

# ── DB SQLite locale (cache + diario eventi) ─────────────────────────────────


def _db_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(settings.db_path, timeout=10)
    conn.row_factory = sqlite3.Row
    return conn


def _init_db() -> None:
    Path(settings.db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = _db_conn()
    try:
        conn.executescript(
            "CREATE TABLE IF NOT EXISTS kv (key TEXT PRIMARY KEY, value TEXT NOT NULL);\n"
            "CREATE TABLE IF NOT EXISTS events (id INTEGER PRIMARY KEY AUTOINCREMENT,"
            " ts TEXT NOT NULL, type TEXT NOT NULL, detail TEXT NOT NULL);"
        )
        conn.commit()
    finally:
        conn.close()


def _kv_get(key: str, default: str | None = None) -> str | None:
    conn = _db_conn()
    try:
        row = conn.execute("SELECT value FROM kv WHERE key = ?", (key,)).fetchone()
        return str(row["value"]) if row else default
    finally:
        conn.close()


def _kv_set(key: str, value: str) -> None:
    conn = _db_conn()
    try:
        conn.execute(
            "INSERT INTO kv (key, value) VALUES (?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (key, value),
        )
        conn.commit()
    finally:
        conn.close()


def _kv_json(key: str, default=None):
    value = _kv_get(key)
    if not value:
        return default
    try:
        return json.loads(value)
    except ValueError:
        return default


def _add_event(event_type: str, detail: str) -> None:
    conn = _db_conn()
    try:
        conn.execute(
            "INSERT INTO events (ts, type, detail) VALUES (?, ?, ?)",
            (datetime.now().isoformat(timespec="seconds"), event_type, str(detail)[:500]),
        )
        conn.commit()
    finally:
        conn.close()


def _recent_events(limit: int = 10) -> list[dict]:
    conn = _db_conn()
    try:
        rows = conn.execute(
            "SELECT ts, type, detail FROM events ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
    finally:
        conn.close()
    return [{"ts": r["ts"], "type": r["type"], "detail": r["detail"]} for r in rows][::-1]

# ── SSH verso il login node (subprocess + ssh nativo, zero lib esotiche) ─────


class ClusterError(Exception):
    """Errore base del driver verso il cluster."""


class ClusterUnreachable(ClusterError):
    """Il cluster non risponde (timeout ssh, chiave assente, ...)."""


class ClusterProtocolError(ClusterError):
    """Il cluster risponde ma l'output del helper non è uno snapshot valido."""


@dataclass
class SSHResult:
    rc: int
    stdout: str
    stderr: str
    timed_out: bool = False


class ClusterSSH:
    """Esegue comandi sul login node: `ssh [-i <key>] -p <port> -o ...`.

    BatchMode=yes (niente prompt); StrictHostKeyChecking=accept-new (host
    registrato al primo contatto in un known_hosts sotto data_dir, scrivibile
    e indipendente dall'HOME di Render).

    La chiave è OPZIONALE: se `settings.ssh_key_file` è None (nessuna env
    T2G_SSH_KEY_FILE/T2G_SSH_KEY_CONTENT) NON passa `-i` e lascia che ssh
    usi l'autenticazione di default (ssh-agent / identità default /
    ~/.ssh/config) — richiesto per il deploy locale di test su Windows.
    """

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.base = ["ssh"]
        if settings.ssh_key_file:
            self.base += ["-i", settings.ssh_key_file]
        self.base += [
            "-p", str(settings.ssh_port),
            "-o", "BatchMode=yes",
            "-o", "ConnectTimeout=15",
            "-o", "StrictHostKeyChecking=accept-new",
            "-o", f"UserKnownHostsFile={settings.ssh_known_hosts}",
            "-o", "LogLevel=ERROR",
            f"{settings.ssh_user}@{settings.ssh_host}",
        ]

    @staticmethod
    def _run(args: list[str], timeout: int) -> SSHResult:
        try:
            proc = subprocess.run(args, capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            return SSHResult(-1, "", "timeout", timed_out=True)
        except FileNotFoundError:
            return SSHResult(-2, "", "binario ssh/scp non trovato")
        except OSError as exc:  # pragma: no cover - dipende dall'ambiente
            return SSHResult(-3, "", f"errore OS: {exc}")
        return SSHResult(proc.returncode, proc.stdout or "", proc.stderr or "")

    def run(self, remote_cmd: str, timeout: int | None = None) -> SSHResult:
        return self._run([*self.base, remote_cmd], timeout or self.settings.ssh_timeout)

    def upload(self, local_path: str, remote_path: str, timeout: int = 30) -> SSHResult:
        args: list[str] = ["scp"]
        if self.settings.ssh_key_file:
            args += ["-i", self.settings.ssh_key_file]
        args += [
            "-P", str(self.settings.ssh_port),
            "-o", "BatchMode=yes",
            "-o", "ConnectTimeout=15",
            "-o", "StrictHostKeyChecking=accept-new",
            "-o", f"UserKnownHostsFile={self.settings.ssh_known_hosts}",
            "-o", "LogLevel=ERROR",
            local_path,
            f"{self.settings.ssh_user}@{self.settings.ssh_host}:{remote_path}",
        ]
        return self._run(args, timeout)

    def __enter__(self) -> "ClusterSSH":
        return self

    def __exit__(self, *exc) -> None:
        return None


def _shq(value: str) -> str:
    """Quota un valore per la shell remota (le entry non contengono apostrofi)."""
    return "'" + value.replace("'", "'\\''") + "'"


def _helper_cmd(subcommand: str, arg: str | None = None) -> str:
    """Comando remoto con probe di esistenza: helper assente → HELPER_MISSING=1."""
    cmd = f"if [ -f {HELPER_REMOTE} ]; then bash {HELPER_REMOTE} {subcommand}"
    if arg is not None:
        cmd += f" {_shq(arg)}"
    cmd += "; else echo 'HELPER_MISSING=1'; fi"
    return cmd


def _install_helper(ssh: ClusterSSH) -> None:
    """Auto-install del helper sul cluster (scp di remote/cluster_helper.sh)."""
    local = Path(__file__).resolve().parent / HELPER_NAME
    if not local.is_file():
        raise ClusterProtocolError(
            "cluster_helper.sh locale mancante in remote/ — non posso auto-installarlo"
        )
    res = ssh.upload(str(local), HELPER_REMOTE)
    if res.rc != 0:
        raise ClusterUnreachable(
            f"scp del helper fallito: {res.stderr.strip()[:200] or res.stdout.strip()[:200]}"
        )
    _log.info("cluster_helper.sh copiato sul cluster")
    _add_event("info", "cluster_helper.sh installato sul cluster")

# ── Parsing dello snapshot del helper (KEY=VALUE; coda separata da \x1f) ─────


def _parse_active_job(value: str) -> dict | None:
    parts = value.split("|", 2)
    return {"id": parts[0] or None, "name": parts[1] or None, "state": parts[2] or None}


def parse_status(text: str) -> dict:
    """Parsa l'output KEY=VALUE del helper; righe sconosciute ignorate."""
    state = {
        "status_ok": 0,
        "active_job": None,
        "queue": [],
        "last_job": None,
        "stopped": False,
        "watcher_alive": False,
        "errors_count": 0,
        "errors_tail": [],
    }
    for line in text.splitlines():
        line = line.strip()
        if not line or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip()
        if key == "STATUS_OK":
            state["status_ok"] = 1 if value == "1" else 0
        elif key == "ACTIVE_JOB":
            state["active_job"] = _parse_active_job(value) if value else None
        elif key == "QUEUE":
            state["queue"] = [e for e in value.split("\x1f") if e] if value else []
        elif key == "LAST_JOB":
            state["last_job"] = value or None
        elif key == "STOPPED":
            state["stopped"] = value == "1"
        elif key == "WATCHER_ALIVE":
            state["watcher_alive"] = value == "1"
        elif key == "ERRORS_COUNT":
            state["errors_count"] = int(value) if value.isdigit() else 0
        elif key == "ERRORS_TAIL":
            try:
                state["errors_tail"] = json.loads(value) if value else []
            except (ValueError, TypeError):
                state["errors_tail"] = []
    return state


def _parse_entry(entry: str) -> dict:
    parts = entry.split(":")
    return {
        "type": parts[0] if len(parts) > 0 else "",
        "config": parts[1] if len(parts) > 1 else "",
        "tag": parts[2] if len(parts) > 2 else "",
        "extra": ":".join(parts[3:]) or None,
    }

# ── Sincronizzazione stato sul DB ────────────────────────────────────────────


def _store_snapshot(state: dict) -> None:
    """Salva lo snapshot (cache) e logga i NUOVI errori (puntatore errors_offset).

    L'helper torna solo le ultime 5 righe di chain_errors: delta ≤ 5 → nuove
    dedotte con precisione; delta maggiore → ultime 5; count calato (file
    resettato) → riparte dalla coda nota.
    """
    now = datetime.now().isoformat(timespec="seconds")
    for key, value in {
        "cluster_reachable": "1",
        "active_job": json.dumps(state["active_job"]) if state.get("active_job") else "",
        "queue": json.dumps(state["queue"]),
        "last_job": state.get("last_job") or "",
        "stopped": "1" if state.get("stopped") else "0",
        "watcher_alive": "1" if state.get("watcher_alive") else "0",
        "errors_recent": json.dumps(state.get("errors_tail", [])),
        "last_tick_at": now,
    }.items():
        _kv_set(key, value)

    total = state.get("errors_count", 0)
    seen = _kv_get("errors_offset")
    seen = int(seen) if seen and seen.isdigit() else 0
    tail = state.get("errors_tail", [])
    new_errors = tail if total < seen else tail[max(0, len(tail) - (total - seen)) :]
    for line in new_errors:
        _add_event("error", f"job fallito sul cluster: {line}")
    _kv_set("errors_offset", str(total))


def _helper_do(ssh: ClusterSSH, subcommand: str, arg: str | None = None) -> dict:
    """Esegue un subcomando del helper e sincronizza il DB col nuovo snapshot.

    Gestisce timeout ssh, helper non installato (auto-install via scp, una
    volta sola) e output non parsabile; ritorna lo snapshot aggiornato.
    """
    res = ssh.run(_helper_cmd(subcommand, arg))
    if res.timed_out:
        raise ClusterUnreachable(f"ssh timeout dopo {settings.ssh_timeout}s")
    if res.rc == -2:
        raise ClusterUnreachable("binario ssh non trovato sul server")
    text = res.stdout or ""
    if "HELPER_MISSING=1" in text:
        if settings.helper_auto_install:
            _install_helper(ssh)
            res = ssh.run(_helper_cmd(subcommand, arg))
            text = res.stdout or ""
        else:
            raise ClusterProtocolError(
                "cluster_helper.sh non presente sul cluster — copialo con: "
                "scp remote/cluster_helper.sh "
                f"{settings.ssh_user}@{settings.ssh_host}:{HELPER_REMOTE}"
            )
    if res.rc != 0:
        raise ClusterUnreachable(
            f"ssh rc={res.rc}: {res.stderr.strip()[:200] or res.stdout.strip()[:200]}"
        )
    state = parse_status(text)
    if state.get("status_ok") != 1:
        raise ClusterProtocolError(
            f"helper '{subcommand}' rc={res.rc} senza STATUS_OK=1: {text[:200]!r}"
        )
    _store_snapshot(state)
    return state


@contextmanager
def _cluster() -> Iterator[ClusterSSH]:
    """Apre ClusterSSH traducendo i fallimenti in HTTP 502 (mai crash)."""
    if not settings.ssh_host or not settings.ssh_user:
        raise HTTPException(502, "T2G_SSH_HOST / T2G_SSH_USER non configurati")
    if settings.ssh_key_file and not Path(settings.ssh_key_file).is_file():
        raise HTTPException(
            502, "Chiave SSH non trovata: imposta T2G_SSH_KEY_CONTENT o T2G_SSH_KEY_FILE"
        )
    try:
        with ClusterSSH(settings) as ssh:
            yield ssh
    except HTTPException:
        raise
    except ClusterUnreachable as exc:
        _add_event("error", f"cluster irraggiungibile: {str(exc)[:200]}")
        _kv_set("cluster_reachable", "0")
        raise HTTPException(502, f"Cluster irraggiungibile: {exc}")
    except ClusterProtocolError as exc:
        _add_event("error", f"protocollo helper: {str(exc)[:200]}")
        raise HTTPException(502, f"Protocollo cluster_helper fallito: {exc}")
    except Exception as exc:  # ultima rete di sicurezza: mai crashare il server
        _log.exception("errore inatteso nell'accesso al cluster")
        _kv_set("cluster_reachable", "0")
        raise HTTPException(502, f"Errore interno nell'accesso al cluster: {exc.__class__.__name__}")

# ── Cache per GET /status (solo DB, mai ssh) ─────────────────────────────────


def _cached_status() -> dict:
    """Stato dal DB cache + ultimi 10 eventi; funziona a cluster irraggiungibile.

    Usato sia da GET /status sia come risposta dei POST: `_helper_do` →
    `_store_snapshot` ha appena scritto lo snapshot fresco, quindi i valori
    letti sono quelli reali dell'ultimo tick.
    """
    return {
        "active_job": _kv_json("active_job"),
        "queue": _kv_json("queue", default=[]),
        "last_job": _kv_get("last_job") or None,
        "stopped": _kv_get("stopped", "0") == "1",
        "watcher_alive": _kv_get("watcher_alive", "0") == "1",
        "errors_recent": _kv_json("errors_recent", default=[]),
        "last_tick_at": _kv_get("last_tick_at") or None,
        "cluster_reachable": _kv_get("cluster_reachable", "0") == "1",
        "events": _recent_events(10),
    }

# ── Validazione config / costruzione entry di coda ───────────────────────────


def resolve_config(config: str) -> str:
    """Risolve un config (nome noto, path o nome file) nel path canonico."""
    if config in CONFIG_MAP:
        return CONFIG_MAP[config]
    if config in CONFIG_PATHS:
        return config
    wanted = Path(config).name  # tollera "zero_shot.yaml" o "ablation/grpo_pda.yaml"
    for path in CONFIG_PATHS:
        if Path(path).name == wanted:
            return path
    raise HTTPException(
        422,
        f"config non valido: {config!r}. Nomi noti: {', '.join(sorted(CONFIG_MAP))}",
    )


def build_entry(job: "JobIn") -> str:
    """Costruisce la riga di coda `type:cfg:tag[:extra]` con validazione."""
    cfg = resolve_config(job.config)
    tag = job.tag if job.tag else Path(cfg).stem.replace("_", "-")
    if not _TAG_RE.fullmatch(tag):
        raise HTTPException(422, f"tag non valido: {tag!r} (consentito [A-Za-z0-9._-])")
    entry = f"{job.type}:{cfg}:{tag}"
    if job.mode:
        if not _EXTRA_RE.fullmatch(job.mode):
            raise HTTPException(422, f"mode non valido: {job.mode!r}")
        entry += f":{job.mode}"
    return entry


def build_queue_lines(payload: "QueueIn") -> list[str]:
    """Espande {ablation: true} o {jobs: [...]} nelle entry di coda."""
    if payload.ablation and payload.jobs is not None:
        raise HTTPException(422, "indicare 'ablation' OPPURE 'jobs', non entrambi")
    if payload.ablation:
        lines: list[str] = []
        for tag, cfg, mode in ABLATION_MODELS:
            if mode == "e":  # eval-only (zero_shot*)
                lines.append(f"eval:{cfg}:{tag}")
            else:  # "te" → train + eval
                lines.append(f"train:{cfg}:{tag}")
                lines.append(f"eval:{cfg}:{tag}")
        return lines
    if payload.jobs is None:
        raise HTTPException(
            422, "corpo richiesto: {'jobs': [...]} oppure {'ablation': true}"
        )
    return [build_entry(j) for j in payload.jobs]

# ── Avvio: chiave ssh da env + init DB ───────────────────────────────────────


def _setup_key() -> None:
    if not settings.ssh_key_content or not settings.ssh_key_file:
        return
    key_path = Path(settings.ssh_key_file)
    key_path.parent.mkdir(parents=True, exist_ok=True)
    # Render serializza le env multilinea con "\n" letterale: normalizza.
    content = settings.ssh_key_content.replace("\\n", "\n").strip() + "\n"
    key_path.write_text(content, encoding="utf-8")
    try:
        os.chmod(key_path, 0o600)  # OpenSSH rifiuta chiavi con permessi aperti
    except OSError:
        pass
    _log.info("Chiave SSH scritta da T2G_SSH_KEY_CONTENT in %s", key_path)
    os.environ.pop("T2G_SSH_KEY_CONTENT", None)


@asynccontextmanager
async def lifespan(app: FastAPI):
    Path(settings.data_dir).mkdir(parents=True, exist_ok=True)
    _setup_key()
    _init_db()
    if not settings.auth_token:
        _log.warning("T2G_AUTH_TOKEN non configurato — le route risponderanno 503")
    _add_event("startup", f"driver avviato ({settings.ssh_user}@{settings.ssh_host})")
    yield
    _add_event("shutdown", "driver fermato")


app = FastAPI(
    title="T2G Cluster Driver",
    description=(
        "Driver esterno della catena T2G su gcluster: avanza la coda "
        "(chain_tick.sh) e ne espone lo stato via API. Tick esterno: "
        "POST /tick ogni 5 min da cronjob.org."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


def require_auth(x_auth_token: str | None = Header(default=None)) -> None:
    """Dependency: tutte le route richiedono X-Auth-Token valido."""
    if not settings.auth_token:
        raise HTTPException(503, "T2G_AUTH_TOKEN non configurato sul server")
    if x_auth_token is None or not secrets.compare_digest(x_auth_token, settings.auth_token):
        raise HTTPException(401, "X-Auth-Token mancante o non valido")

# ── API REST (tutte autenticate) ─────────────────────────────────────────────


class JobIn(BaseModel):
    type: Literal["train", "eval"]
    config: str
    tag: str | None = None
    mode: str | None = None  # extra args del job sulla coda (es. "--resume")


class QueueIn(BaseModel):
    jobs: list[JobIn] | None = None
    ablation: bool = False


@app.get("/")
def root() -> dict:
    return {
        "service": "t2g-cluster-driver",
        "docs": "/docs",
        "cluster": f"{settings.ssh_user}@{settings.ssh_host}",
    }


@app.get("/status", dependencies=[Depends(require_auth)])
def get_status() -> dict:
    """Stato dalla cache DB (mai ssh): funziona anche a cluster giù."""
    return _cached_status()


@app.get("/jobs", dependencies=[Depends(require_auth)])
def list_jobs() -> list[dict]:
    """Job in coda (dal DB sincronizzato all'ultimo tick)."""
    queue = _cached_status()["queue"]
    return [{"entry": e, **_parse_entry(e)} for e in queue]


@app.post("/jobs", status_code=201, dependencies=[Depends(require_auth)])
def add_job(payload: JobIn) -> dict:
    """Accoda un job: {type: train|eval, config: nome|path, tag?, mode?}."""
    entry = build_entry(payload)
    with _cluster() as ssh:
        _helper_do(ssh, "enqueue", entry)
    _add_event("enqueue", entry)
    return {"added": entry, "status": _cached_status()}


@app.post("/queue", dependencies=[Depends(require_auth)])
def replace_queue(payload: QueueIn) -> dict:
    """Rimpiazza l'intera coda: {jobs: [...]} oppure {ablation: true}."""
    lines = build_queue_lines(payload)
    with _cluster() as ssh:
        state = _helper_do(ssh, "rewrite_queue", "\x1f".join(lines))
    _add_event("queue_replace", f"{len(lines)} entry (ablation={payload.ablation})")
    return {"queue": state["queue"], "count": len(state["queue"]), "status": _cached_status()}


@app.delete("/jobs/{tag}", dependencies=[Depends(require_auth)])
def delete_jobs(tag: str) -> dict:
    """Rimuove dalla coda tutti i job col tag dato (riscrive job_chain filtrato)."""
    with _cluster() as ssh:
        state = _helper_do(ssh, "status")
    kept = [e for e in state["queue"] if _parse_entry(e)["tag"] != tag]
    removed = len(state["queue"]) - len(kept)
    if removed == 0:
        _add_event("dequeue", f"nessun job con tag '{tag}' in coda")
        return {"removed": 0, "status": _cached_status()}
    with _cluster() as ssh:
        state = _helper_do(ssh, "rewrite_queue", "\x1f".join(kept))
    _add_event("dequeue", f"rimossi {removed} job con tag '{tag}'")
    return {"removed": removed, "status": _cached_status()}


@app.post("/pause", dependencies=[Depends(require_auth)])
def pause() -> dict:
    """Crea chain_stopped sul cluster (soft stop: nessuna nuova sottomissione)."""
    with _cluster() as ssh:
        _helper_do(ssh, "pause")
    _add_event("pause", "chain_stopped creato sul cluster")
    return {"status": _cached_status()}


@app.post("/resume", dependencies=[Depends(require_auth)])
def resume() -> dict:
    """Rimuove chain_stopped e fa un tick immediato."""
    with _cluster() as ssh:
        _helper_do(ssh, "resume")
        _helper_do(ssh, "tick")
    _add_event("resume", "chain_stopped rimosso + tick immediato")
    return {"status": _cached_status()}


@app.post("/tick", dependencies=[Depends(require_auth)])
def tick() -> dict:
    """Tick manuale: esegue chain_tick.sh sul cluster e sincronizza lo stato."""
    with _cluster() as ssh:
        _helper_do(ssh, "tick")
    _add_event("tick", "tick eseguito (chain_tick.sh --quiet)")
    return _cached_status()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=_env_int("PORT", 8000))
