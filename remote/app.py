"""FastAPI/SQLite driver for the remote T2G cluster chain.

Cluster state remains authoritative; SQLite caches snapshots and events so
status stays available during SSH outages. Authentication and SSH credentials
come from environment variables.
"""

from __future__ import annotations

import base64
import json
import logging
import os
import re
import secrets
import sqlite3
import subprocess
import tempfile
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterator, Literal

from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel

_log = logging.getLogger("uvicorn.error")

_ANSI_ESCAPE_RE = re.compile(r"\x1b(?:\[[0-?]*[ -/]*[@-~]|[@-_])")

# ── dotenv: carica .env dalla repo root e dalla cwd (deploy locale) ──────────
# python-dotenv è dipendenza core del progetto. override=False: le env vars
# reali vincono sul file (comodo per override puntuali senza editare).
try:
    from dotenv import load_dotenv

    for _candidate in (
        Path(__file__).resolve().parent.parent / ".env",
        Path.cwd() / ".env",
    ):
        if _candidate.is_file():
            load_dotenv(_candidate, override=False)
except ImportError:  # pragma: no cover - dotenv è core, fallback silenzioso
    pass

# ── Configurazione (env vars; NESSUNA credenziale del cluster: l'SSH usa ─────
#    l'alias ~/.ssh/config, es. T2G_SSH_HOST=gcluster) ─────────────────────────


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
            # Default "gcluster": l'alias ~/.ssh/config dell'utente (senza
            # user@ né -p: li risolve l'alias stesso). T2G_SSH_USER/T2G_SSH_PORT
            # servono SOLO se non si usa un alias.
            ssh_host=os.environ.get("T2G_SSH_HOST", "gcluster"),
            ssh_user=os.environ.get("T2G_SSH_USER", ""),
            ssh_port=_env_int("T2G_SSH_PORT", 0),
            # None = nessun `-i`: ssh-agent / identità default / ~/.ssh/config
            ssh_key_file=key_file,
            ssh_key_content=key_content,
            ssh_known_hosts=os.environ.get(
                "T2G_SSH_KNOWN_HOSTS", str(Path(data_dir) / "known_hosts")
            ),
            ssh_timeout=_env_int("T2G_SSH_TIMEOUT", 30),
            # OPZIONALE in locale (bind 127.0.0.1): chiave API servizio↔TUI.
            # OBBLIGATORIA su Render (0.0.0.0) — vedi warning in lifespan.
            auth_token=os.environ.get("T2G_AUTH_TOKEN", ""),
            db_path=os.environ.get(
                "T2G_DB_PATH", str(Path(data_dir) / "t2g_driver.db")
            ),
            data_dir=data_dir,
            helper_auto_install=os.environ.get("T2G_HELPER_AUTO_INSTALL", "1")
            not in ("", "0", "false", "False"),
        )


settings = Settings.from_env()

# ── Config noti (nome → path). Nomi cella = schema pipeline-first. ──────────

CONFIG_MAP: dict[str, str] = {
    "baseline-zero": "experiments/configs/qwen25-05b/baseline/zero-shot.yaml",
    "baseline-few": "experiments/configs/qwen25-05b/baseline/few-shot.yaml",
    "sft": "experiments/configs/qwen25-05b/sft/zero-shot.yaml",
    "grpo-zero": "experiments/configs/qwen25-05b/grpo/zero-shot.yaml",
    "grpo-few": "experiments/configs/qwen25-05b/grpo/few-shot.yaml",
    "sft-grpo-zero": "experiments/configs/qwen25-05b/sft-grpo/zero-shot.yaml",
    "sft-grpo-few": "experiments/configs/qwen25-05b/sft-grpo/few-shot.yaml",
    "sft-grpo-zero-pda": "experiments/configs/qwen25-05b/ablations/sft-grpo-zero-pda.yaml",
    "sft-grpo-zero-hot": "experiments/configs/qwen25-05b/ablations/sft-grpo-zero-hot.yaml",
}

CONFIG_PATHS: set[str] = set(CONFIG_MAP.values())
EVAL_ONLY_CONFIGS = frozenset({"baseline-zero", "baseline-few"})

# Default campaign: 2 eval-only baselines + 5 train/eval cells = 12 entries.
# Each trained eval entry runs both prompt modes inside one cluster job.
# Ablations remain manual-only.
DEFAULT_CAMPAIGN: list[tuple[str, str, str]] = [
    ("baseline-zero", CONFIG_MAP["baseline-zero"], "e"),
    ("baseline-few", CONFIG_MAP["baseline-few"], "e"),
    ("sft", CONFIG_MAP["sft"], "te"),
    ("grpo-zero", CONFIG_MAP["grpo-zero"], "te"),
    ("grpo-few", CONFIG_MAP["grpo-few"], "te"),
    ("sft-grpo-zero", CONFIG_MAP["sft-grpo-zero"], "te"),
    ("sft-grpo-few", CONFIG_MAP["sft-grpo-few"], "te"),
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
            (
                datetime.now().isoformat(timespec="seconds"),
                event_type,
                str(detail)[:500],
            ),
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
    return [{"ts": r["ts"], "type": r["type"], "detail": r["detail"]} for r in rows][
        ::-1
    ]


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
    """Run non-interactive SSH/SCP commands with optional explicit identity."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.base = ["ssh"]
        if settings.ssh_key_file:
            self.base += ["-i", settings.ssh_key_file]
        # Alias ~/.ssh/config (default "gcluster"): niente user@ né -p — li
        # risolve l'alias. user/porta espliciti SOLO se configurati via env.
        self.base += [
            "-o",
            "BatchMode=yes",
            "-o",
            "ConnectTimeout=15",
            "-o",
            "StrictHostKeyChecking=accept-new",
            "-o",
            f"UserKnownHostsFile={settings.ssh_known_hosts}",
            "-o",
            "LogLevel=ERROR",
        ]
        if settings.ssh_port:
            self.base += ["-p", str(settings.ssh_port)]
        target = (
            f"{settings.ssh_user}@{settings.ssh_host}"
            if settings.ssh_user
            else settings.ssh_host
        )
        self.base.append(target)

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
            "-o",
            "BatchMode=yes",
            "-o",
            "ConnectTimeout=15",
            "-o",
            "StrictHostKeyChecking=accept-new",
            "-o",
            f"UserKnownHostsFile={self.settings.ssh_known_hosts}",
            "-o",
            "LogLevel=ERROR",
        ]
        if self.settings.ssh_port:
            args += ["-P", str(self.settings.ssh_port)]
        target = (
            f"{self.settings.ssh_user}@{self.settings.ssh_host}"
            if self.settings.ssh_user
            else self.settings.ssh_host
        )
        args += [local_path, f"{target}:{remote_path}"]
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
    """Cache a snapshot and append errors not seen in earlier snapshots."""
    now = datetime.now().isoformat(timespec="seconds")
    for key, value in {
        "cluster_reachable": "1",
        "active_job": (
            json.dumps(state["active_job"]) if state.get("active_job") else ""
        ),
        "queue": json.dumps(state["queue"]),
        "last_job": state.get("last_job") or "",
        "stopped": "1" if state.get("stopped") else "0",
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
    if not settings.ssh_host:
        raise HTTPException(502, "T2G_SSH_HOST non configurato")
    if settings.ssh_key_file and not Path(settings.ssh_key_file).is_file():
        raise HTTPException(
            502,
            "Chiave SSH non trovata: imposta T2G_SSH_KEY_CONTENT o T2G_SSH_KEY_FILE",
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
        raise HTTPException(
            502, f"Errore interno nell'accesso al cluster: {exc.__class__.__name__}"
        )


# ── Cache per GET /status (solo DB, mai ssh) ─────────────────────────────────


def _cached_status() -> dict:
    """Return cached cluster state and recent events."""
    return {
        "active_job": _kv_json("active_job"),
        "queue": _kv_json("queue", default=[]),
        "last_job": _kv_get("last_job") or None,
        "stopped": _kv_get("stopped", "0") == "1",
        "errors_recent": _kv_json("errors_recent", default=[]),
        "last_tick_at": _kv_get("last_tick_at") or None,
        "cluster_reachable": _kv_get("cluster_reachable", "0") == "1",
        "events": _recent_events(10),
    }


# ── Validazione config / costruzione entry di coda ───────────────────────────


def resolve_config(config: str) -> str:
    """Resolve a semantic ID or an exact canonical path."""
    if config in CONFIG_MAP:
        return CONFIG_MAP[config]
    if config in CONFIG_PATHS:
        return config
    raise HTTPException(
        422,
        f"config non valido: {config!r}. Nomi noti: {', '.join(sorted(CONFIG_MAP))}",
    )


def build_entry(job: "JobIn") -> str:
    """Costruisce la riga di coda `type:cfg:tag[:extra]` con validazione."""
    cfg = resolve_config(job.config)
    config_id = next(name for name, path in CONFIG_MAP.items() if path == cfg)
    if job.type == "train" and config_id in EVAL_ONLY_CONFIGS:
        raise HTTPException(422, f"config eval-only non addestrabile: {config_id}")
    tag = job.tag if job.tag else config_id
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
        for tag, cfg, mode in DEFAULT_CAMPAIGN:
            if mode == "e":  # eval-only
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
    if settings.auth_token:
        _log.info("Auth ATTIVA (X-Auth-Token richiesto)")
    else:
        _log.warning(
            "T2G_AUTH_TOKEN non configurato — auth DISABILITATA. Ok SOLO per "
            "il deploy locale (bind 127.0.0.1). OBBLIGATORIA quando il "
            "servizio viene esposto (Render)."
        )
    target = (
        f"{settings.ssh_user}@{settings.ssh_host}"
        if settings.ssh_user
        else settings.ssh_host
    )
    _add_event("startup", f"driver avviato ({target})")
    yield
    _add_event("shutdown", "driver fermato")


app = FastAPI(
    title="T2G Cluster Driver",
    description=(
        "Driver esterno della catena T2G su gcluster: avanza la coda "
        "(chain_tick.sh), ne espone lo stato via API e fornisce il monitor "
        "live (metriche training, completion samples, log tail). "
        "Tick esterno: POST /tick ogni 5 min da cronjob.org."
    ),
    version="2.0.0",
    lifespan=lifespan,
)


def require_auth(x_auth_token: str | None = Header(default=None)) -> None:
    """Dependency: X-Auth-Token valido — oppure nessuna auth se il token non
    è configurato (deploy locale su 127.0.0.1)."""
    if not settings.auth_token:
        return  # auth disabilitata: solo deploy locale
    if x_auth_token is None or not secrets.compare_digest(
        x_auth_token, settings.auth_token
    ):
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


class BatchStartIn(BaseModel):
    """POST /jobs/batch: enqueue di più job (+ tick immediato opzionale)."""

    jobs: list[JobIn]
    start_now: bool = True


@app.get("/")
def root() -> dict:
    target = (
        f"{settings.ssh_user}@{settings.ssh_host}"
        if settings.ssh_user
        else settings.ssh_host
    )
    return {
        "service": "t2g-cluster-driver",
        "docs": "/docs",
        "cluster": target,
        "auth": bool(settings.auth_token),
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
    return {
        "queue": state["queue"],
        "count": len(state["queue"]),
        "status": _cached_status(),
    }


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


# ── API v2: monitor live + controllo job (la TUI è costruita su queste) ──────


def _import_chain_monitor():
    """Importa i parser del monitor (torch-free) — il servizio gira dalla
    repo root, quindi `src.utils.chain_monitor` è importabile direttamente."""
    import sys

    repo_root = str(Path(__file__).resolve().parent.parent)
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    from src.utils import chain_monitor

    return chain_monitor


def _job_detail_from_log(
    active_job: dict | None,
    log_tail_lines: list[str],
    log_path: str | None,
) -> dict | None:
    """Costruisce job_detail riusando i parser di chain_monitor.

    I parser leggono da FILE: il tail (già arrivato via LOG_TAIL_B64) viene
    scritto su un file temporaneo nella data_dir e i parser `_parse_training_log`
    / `_parse_eval_log` (intatti) lo analizzano. I loro `_tail_lines`/`_grep_lines`
    su file così piccolo cadono nel fallback read_text — nessun subprocess.
    """
    if not active_job or not active_job.get("id"):
        return None
    name = active_job.get("name") or ""
    job_type = "eval" if name.startswith("eval-") else "train"
    tag = name.split("-", 1)[1] if "-" in name else name

    detail: dict = {
        "id": active_job.get("id"),
        "name": name,
        "state": active_job.get("state"),
        "elapsed_human": None,
        "log_path": log_path or None,
        "step": None,
        "total_steps": None,
        "loss": None,
        "reward": None,
        "lr": None,
        "sft_active": False,
        "sft_step": None,
        "sft_total": None,
        "sft_loss": None,
        "sft_eval_loss": None,
        "sft_eval_loss_best": None,
        "eval_label": None,
        "eval_progress": None,
        "eval_metrics": {},
    }
    if not log_tail_lines:
        return detail

    try:
        cm = _import_chain_monitor()
        with tempfile.TemporaryDirectory(
            prefix="t2g-monitor-", dir=settings.data_dir
        ) as tmp_dir:
            tmp = Path(tmp_dir) / "tail.log"
            tmp.write_text(
                "\n".join(str(line) for line in log_tail_lines) + "\n",
                encoding="utf-8",
                errors="replace",
            )
            job = cm.JobInfo(job_type=job_type, config="", tag=tag)
            if job_type == "eval":
                cm._parse_eval_log(tmp, job)
            else:
                cm._parse_training_log(tmp, job)
    except Exception as exc:  # parser robusto: mai fallire il monitor
        _log.warning("chain_monitor parse fallito: %s", exc)
        return detail

    detail.update(
        {
            "step": job.step or None,
            "total_steps": job.stage_total or None,
            "reward": job.last_reward or None,
            "sft_active": job.sft_active,
            "sft_step": job.sft_step or None,
            "sft_total": job.sft_total or None,
            "sft_loss": job.sft_loss or None,
            "sft_eval_loss": job.sft_eval_loss or None,
            "sft_eval_loss_best": job.sft_eval_loss_best or None,
            "eval_label": job.eval_label or None,
            "eval_progress": (
                f"{job.eval_label}"
                if job.eval_label and "samples" in job.eval_label
                else None
            ),
            "eval_metrics": job.eval_metrics or {},
        }
    )
    # loss/lr: ultima riga KV che li contiene (il parser non li espone come
    # campi dedicati — loss sta nel KV step, lr nella stessa riga HighPrecision).
    for line in reversed(log_tail_lines):
        m = re.search(r"\bloss=([\d.eE+-]+)", line)
        if m and not job.sft_active:
            detail["loss"] = m.group(1)
            break
    for line in reversed(log_tail_lines):
        m = re.search(r"\b(?:lr|learning_rate)=([\d.eE+-]+)", line)
        if m:
            detail["lr"] = m.group(1)
            break
    return detail


def _decode_log_tail(state: dict) -> tuple[list[str], str | None]:
    """Estrae e decodifica LOG_TAIL_B64 dallo snapshot del subcomando monitor."""
    b64 = state.get("log_tail_b64") or ""
    log_path = state.get("log_path") or None
    if not b64:
        return [], log_path
    try:
        text = base64.b64decode(b64).decode("utf-8", errors="replace")
    except (ValueError, TypeError):
        return [], log_path
    return text.splitlines(), log_path


def _parse_monitor_status(text: str) -> dict:
    """parse_status esteso: cattura LOG_PATH, LOG_TAIL_B64 e LIVE_STATUS."""
    state = parse_status(text)
    for line in text.splitlines():
        key, _, value = line.partition("=")
        if key == "LOG_PATH":
            state["log_path"] = value.strip()
        elif key == "LOG_TAIL_B64":
            state["log_tail_b64"] = value.strip()
        elif key == "LIVE_STATUS":
            # One-line JSON written by src/utils/live_status.py — if it does
            # not parse, simply ignore it (fallback to log parsing).
            try:
                state["live_status"] = json.loads(value.strip())
            except (ValueError, TypeError):
                state["live_status"] = None
    return state


def _helper_monitor(ssh: ClusterSSH, nlines: int = 200) -> dict:
    """Subcomando `monitor`: snapshot + log tail; sincronizza il DB."""
    res = ssh.run(_helper_cmd("monitor", str(nlines)))
    if res.timed_out:
        raise ClusterUnreachable(f"ssh timeout dopo {settings.ssh_timeout}s")
    text = res.stdout or ""
    if "HELPER_MISSING=1" in text and settings.helper_auto_install:
        _install_helper(ssh)
        res = ssh.run(_helper_cmd("monitor", str(nlines)))
        text = res.stdout or ""
    if res.rc != 0:
        raise ClusterUnreachable(
            f"ssh rc={res.rc}: {res.stderr.strip()[:200] or res.stdout.strip()[:200]}"
        )
    state = _parse_monitor_status(text)
    if state.get("status_ok") != 1:
        raise ClusterProtocolError(
            f"helper 'monitor' senza STATUS_OK=1: {text[:200]!r}"
        )
    _store_snapshot(state)
    return state


def _job_detail_from_live(
    active_job: dict | None,
    live: dict | None,
    log_path: str | None,
) -> dict | None:
    """Costruisce job_detail dal live status file (fonte primaria).

    Il training scrive logs/live_status.json via src/utils/live_status.py —
    molto più robusto del log parsing. Ritorna None se il live status non è
    disponibile (fallback al parsing del log).
    """
    if not active_job or not active_job.get("id"):
        return None
    if not isinstance(live, dict) or not live.get("phase"):
        return None
    name = active_job.get("name") or ""
    phase = str(live.get("phase"))
    sft_active = phase in ("sft", "sft_eval")
    return {
        "id": active_job.get("id"),
        "name": name,
        "state": active_job.get("state"),
        "elapsed_human": None,
        "log_path": log_path or None,
        "phase": phase,
        "eval_active": bool(live.get("eval_active"))
        or phase in ("sft_eval", "grpo_eval", "eval"),
        "step": live.get("step"),
        "total_steps": live.get("total_steps"),
        "loss": (
            f"{live['loss']:.6f}"
            if isinstance(live.get("loss"), (int, float))
            else live.get("loss")
        ),
        "reward": live.get("reward"),
        "reward_avg": live.get("reward_avg"),
        "lr": live.get("lr"),
        "sft_active": sft_active,
        "sft_step": live.get("step") if sft_active else None,
        "sft_total": live.get("total_steps") if sft_active else None,
        "sft_loss": live.get("loss") if sft_active else None,
        "sft_eval_loss": live.get("eval_loss"),
        "sft_eval_loss_best": live.get("eval_loss_best"),
        "eval_label": live.get("note") or phase,
        "eval_progress": live.get("eval_progress"),
        "eval_metrics": {},
        "source": "live",
    }


def _monitor_snapshot(ssh: ClusterSSH) -> dict:
    """Snapshot completo per /monitor: stato + job_detail + samples + log tail.

    job_detail: dal live status file (fonte primaria, ``source: "live"``)
    con fallback al parsing del log SLURM via chain_monitor (``source:
    "log"``). samples: dal live status (formattati dal produttore) se
    presenti, altrimenti estratti dal log tail.
    """
    state = _helper_monitor(ssh)
    tail_lines, log_path = _decode_log_tail(state)
    live_value = state.get("live_status")
    live = live_value if isinstance(live_value, dict) else None
    job_detail = _job_detail_from_live(state.get("active_job"), live, log_path)
    if job_detail is None:
        job_detail = _job_detail_from_log(state.get("active_job"), tail_lines, log_path)
    snapshot = _cached_status()
    snapshot.update(
        {
            "job_detail": job_detail,
            "samples": [],
            "log_tail": tail_lines[-40:],
            "ts": datetime.now().isoformat(timespec="seconds"),
        }
    )
    # Samples: prima dal live status (già formattati), poi dal log tail.
    live_samples = live.get("samples") if live else None
    if isinstance(live_samples, list) and live_samples:
        snapshot["samples"] = [str(sample) for sample in live_samples[-8:]]
    elif tail_lines:
        try:
            cm = _import_chain_monitor()
            samples = cm._extract_completion_samples(
                tail_lines, max_lines=cm._SAMPLE_MAX_LINES
            )
            snapshot["samples"] = (
                [_ANSI_ESCAPE_RE.sub("", str(sample)) for sample in samples[-8:]]
                if samples
                else []
            )
        except Exception as exc:
            _log.warning("extract_completion_samples fallito: %s", exc)
    return snapshot


@app.get("/monitor", dependencies=[Depends(require_auth)])
def monitor() -> dict:
    """Snapshot live: stato catena + metriche job attivo + samples + log tail.

    Riusa i parser di src/utils/chain_monitor.py sul log del job attivo
    (trasportato via base64 dal helper — una sola connessione ssh).
    """
    with _cluster() as ssh:
        return _monitor_snapshot(ssh)


@app.post("/jobs/start", status_code=201, dependencies=[Depends(require_auth)])
def start_job(payload: JobIn) -> dict:
    """Accoda un job e fa un tick immediato: parte SUBITO se la coda è libera.

    Response: snapshot /monitor + `started_now` (True se il tick ha sottomesso
    proprio questo job — nessun altro job era attivo).
    """
    entry = build_entry(payload)
    # tag derivato dal payload (stessa regola di build_entry) per verificare
    # che il job attivo dopo il tick sia PROPRIO quello appena accodato —
    # non basta il prefisso del tipo (un "train-altro" già attivo non è il nostro).
    entry_tag = entry.split(":")[2] if entry.count(":") >= 2 else ""
    expected_name = f"{payload.type}-{entry_tag}"
    with _cluster() as ssh:
        _helper_do(ssh, "enqueue", entry)
        state = _helper_do(ssh, "tick")
    active = state.get("active_job")
    started_now = bool(active and active.get("name") == expected_name)
    _add_event("enqueue+tick" if started_now else "enqueue", entry)
    with _cluster() as ssh:
        snapshot = _monitor_snapshot(ssh)
    snapshot["started_now"] = started_now
    return snapshot


@app.post("/jobs/batch", status_code=201, dependencies=[Depends(require_auth)])
def start_batch(payload: BatchStartIn) -> dict:
    """Accoda più job in ordine (+ tick immediato se start_now).

    Atomico: TUTTE le entry vengono validate PRIMA di toccare il cluster —
    un config invalido → 422 e niente viene accodato. Il tick (se start_now)
    sottomette il primo job quando la coda è libera; gli altri restano in
    coda e avanzano col chain tick successivo.

    Response: snapshot /monitor + `started_now` (True se il tick ha sottomesso
    il primo job della lista) + `queued` (le entry accodate).
    """
    if not payload.jobs:
        raise HTTPException(422, "jobs vuoto: almeno un job richiesto")
    # Validazione atomica: build_entry alza 422 al primo invalido — nessuna
    # scrittura sul cluster avviene prima che TUTTE le entry siano valide.
    entries = [build_entry(job) for job in payload.jobs]

    with _cluster() as ssh:
        for entry in entries:
            _helper_do(ssh, "enqueue", entry)
        state = (
            _helper_do(ssh, "tick") if payload.start_now else _helper_do(ssh, "status")
        )

    started_now = False
    if payload.start_now and entries:
        first = entries[0]
        parts = first.split(":")
        expected_name = f"{parts[0]}-{parts[2]}" if len(parts) >= 3 else ""
        active = state.get("active_job")
        started_now = bool(active and active.get("name") == expected_name)

    _add_event(
        "enqueue+tick" if started_now else "enqueue",
        f"batch: {len(entries)} job ({', '.join(_parse_entry(e)['tag'] for e in entries[:5])}"
        f"{'…' if len(entries) > 5 else ''})",
    )
    with _cluster() as ssh:
        snapshot = _monitor_snapshot(ssh)
    snapshot["started_now"] = started_now
    snapshot["queued"] = entries
    return snapshot


@app.post("/kill", dependencies=[Depends(require_auth)])
def kill_active() -> dict:
    """scancel del job SLURM attivo (409 se nessun job attivo).

    Semantica: il job killato appare CANCELLED e al prossimo tick la catena
    CONTINUA col job successivo (continue-on-failure). Per fermare TUTTO:
    POST /pause prima (o subito dopo) del kill.
    """
    with _cluster() as ssh:
        res = ssh.run(_helper_cmd("scancel"))
    if res.rc != 0:
        raise HTTPException(409, "Nessun job attivo da cancellare (o scancel fallito)")
    _add_event("kill", f"scancel del job attivo: {res.stdout.strip()[:100]}")
    with _cluster() as ssh:
        return _monitor_snapshot(ssh)


@app.get("/logs", dependencies=[Depends(require_auth)])
def get_logs(lines: int = 50) -> dict:
    """Ultime `lines` righe del log del job attivo (404 se nessun job attivo)."""
    lines = max(1, min(lines, 500))
    with _cluster() as ssh:
        state = _helper_monitor(ssh, nlines=lines)
    tail_lines, log_path = _decode_log_tail(state)
    if not state.get("active_job"):
        raise HTTPException(404, "Nessun job attivo: nessun log da leggere")
    return {"log_path": log_path, "lines": tail_lines[-lines:]}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=_env_int("PORT", 8000))
