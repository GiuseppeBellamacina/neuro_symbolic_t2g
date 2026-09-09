"""Client TUI locale per il driver T2G (remote/app.py).

Pannello di controllo COMPLETO del cluster: monitor live (metriche training,
completion samples, log tail) + gestione coda + start/pause/kill dei job.
Sostituisce il vecchio monitor testuale (chain_monitor.py resta come
libreria ausiliaria lato servizio).

La dashboard include grafici e segnali di freschezza: sparkline loss/reward
per step (GET /timeseries), barra di avanzamento con ETA dal ritmo osservato,
coda/eventi come DataTable, log tail incrementale colorato per livello,
provenienza dei dati (live/cache + età). Gli endpoint più recenti
(`/timeseries`, `/results`, `/configs`) possono NON esistere sul servizio:
ogni pannello degrada a un messaggio esplicito, mai a un crash.

Avvio:

    uv run --extra tui python remote/tui.py [--url URL] [--token TOKEN]

Configurazione (in ordine di precedenza): flag CLI → env vars
``T2G_SERVICE_URL`` / ``T2G_AUTH_TOKEN`` → file ``.env`` (cwd o repo root).
Il token è OPZIONALE (servizio locale senza auth): basta l'URL. Se manca
pure l'URL, l'app parte sulla schermata di configurazione che salva i
valori in ``.env`` (sezione marcata ``# >>> t2g-tui >>>``). Il token non
viene mai loggato né stampato.

Dipendenze: textual + httpx (extra ``tui`` di pyproject.toml); python-dotenv
è già una dipendenza core del progetto.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import httpx
from dotenv import dotenv_values
from rich.markup import escape
from rich.text import Text
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import Screen
from textual.widgets import (
    Button,
    Checkbox,
    DataTable,
    Footer,
    Header,
    Input,
    LoadingIndicator,
    ProgressBar,
    RichLog,
    Select,
    Sparkline,
    Static,
    TextArea,
)

# ── Config noti al driver (stessi nomi di remote/app.py:CONFIG_MAP) ──────────

CONFIG_NAMES: tuple[str, ...] = (
    "sft-grpo-few-shot",
    "sft-grpo-zero-shot",
    "sft-zero-shot",
    "grpo-few-shot",
    "grpo-zero-shot",
    "baseline-zero-shot",
    "baseline-zero-shot-no-grammar",
    "baseline-few-shot",
    "ablations-decoding-no-grammar",
    "ablations-decoding-hot-rollout",
    "ablations-rewards-edit-validity",
    "ablations-rewards-historical-stack",
    "ablations-loss-dr-grpo",
    "ablations-objectives-sft-allowed-mass",
    "ablations-objectives-sft-structured",
)
CONFIG_NAME_SET: frozenset[str] = frozenset(CONFIG_NAMES)

# Riga che delimita la sezione gestita dal TUI dentro .env (idempotente).
ENV_MARKER = "# >>> t2g-tui >>>"


@dataclass(frozen=True)
class T2GConfig:
    """Configurazione del servizio remoto (URL normalizzato + token).

    `token` è escluso dal `repr` generato: nessun rischio di leak nei log.
    """

    url: str
    token: str = field(repr=False)


# ── Eccezioni tipizzate del client ───────────────────────────────────────────


class RemoteServiceError(Exception):
    """Errore base del client verso il servizio remoto."""


class ConnectionError(RemoteServiceError):
    """Connessione rifiutata / timeout: servizio giù o in cold start."""


class AuthError(RemoteServiceError):
    """401: X-Auth-Token mancante o non valido."""


class ApiError(RemoteServiceError):
    """Errore HTTP con status + detail (4xx/5xx non gestiti sopra)."""

    def __init__(self, status_code: int, detail: str) -> None:
        super().__init__(f"HTTP {status_code}: {detail}")
        self.status_code = status_code
        self.detail = detail


# ── Lettura/scrittura .env (sezione marcata) ─────────────────────────────────


def env_file_candidates() -> list[Path]:
    """.env in cwd, poi nella repo root (dove vive il progetto)."""
    repo_root = Path(__file__).resolve().parent.parent
    return [Path.cwd() / ".env", repo_root / ".env"]


def read_env_file(path: Path) -> dict[str, str]:
    """Legge un .env con python-dotenv, senza toccare os.environ."""
    if not path.is_file():
        return {}
    return {
        key: value
        for key, value in dotenv_values(str(path)).items()
        if value is not None
    }


def _format_env_lines(url: str, token: str) -> list[str]:
    """Righe T2G_* da scrivere nel .env (valori quotati; mai loggati)."""
    return [f'T2G_SERVICE_URL="{url}"', f'T2G_AUTH_TOKEN="{token}"']


def save_env_config(url: str, token: str, path: Path | None = None) -> Path:
    """Salva URL+token nel .env scelto, dentro la sezione marcata.

    La sezione va dalla riga ``# >>> t2g-tui >>>`` fino alla prima riga non
    ``T2G_*`` successiva: viene sostituita per intero (idempotente). Il resto
    del file resta intatto.
    """
    target = path or env_file_candidates()[0]
    lines = target.read_text(encoding="utf-8").splitlines() if target.is_file() else []
    marker_idx = next(
        (i for i, ln in enumerate(lines) if ln.strip() == ENV_MARKER), None
    )
    section = [ENV_MARKER, *_format_env_lines(url, token)]
    if marker_idx is not None:
        end = marker_idx + 1
        while end < len(lines) and lines[end].startswith("T2G_"):
            end += 1
        lines = lines[:marker_idx] + section + lines[end:]
    else:
        if lines and lines[-1].strip():
            lines.append("")
        lines += section + [""]
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return target


def resolve_config(
    cli_url: str | None = None,
    cli_token: str | None = None,
    dotenv_paths: Iterable[Path] | None = None,
) -> T2GConfig | None:
    """Risolve URL+token: CLI args → env vars → .env (cwd, poi repo root).

    Il token è OPZIONALE (il servizio locale gira anche senza auth): basta
    l'URL. Ritorna None solo se manca l'URL (l'app mostrerà la schermata di
    configurazione).
    """
    paths = list(dotenv_paths) if dotenv_paths is not None else env_file_candidates()
    file_values: dict[str, str] = {}
    for path in paths:
        file_values.update(read_env_file(path))
    url = (
        cli_url
        or os.environ.get("T2G_SERVICE_URL")
        or file_values.get("T2G_SERVICE_URL", "")
    )
    token = (
        cli_token
        or os.environ.get("T2G_AUTH_TOKEN")
        or file_values.get("T2G_AUTH_TOKEN", "")
    ) or ""
    if url:
        return T2GConfig(url=url.strip().rstrip("/"), token=token.strip())
    return None


# ── Client API del driver (sincrono, testabile senza TUI) ────────────────────


def _decode(response: httpx.Response) -> Any:
    """Decodifica il body JSON; fallback sul testo grezzo se non è JSON."""
    try:
        return response.json()
    except ValueError:
        return response.text


def _extract_detail(response: httpx.Response) -> str:
    """Estrae il messaggio ``detail`` (stringa o lista di errori FastAPI)."""
    try:
        body = response.json()
    except ValueError:
        return response.text[:300] or f"HTTP {response.status_code}"
    if isinstance(body, dict):
        detail = body.get("detail")
        if isinstance(detail, list):
            return "; ".join(
                str(item.get("msg", item)) for item in detail if isinstance(item, dict)
            )
        if detail:
            return str(detail)
    return f"HTTP {response.status_code}"


class RemoteServiceClient:
    """Client HTTP del driver T2G (httpx.Client + header X-Auth-Token).

    Base URL normalizzata (strip del trailing ``/``), timeout di default 30s
    (90s su POST /tick: può toccare il cluster via ssh dopo lo sleep di
    Render free tier). Errori tipizzati: `ConnectionError` (rete/timeout),
    `AuthError` (401), `ApiError` (status+detail).
    """

    def __init__(
        self,
        url: str,
        token: str,
        timeout: float = 30.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.base_url = url.rstrip("/")
        self._token = token
        self._timeout = timeout
        self._transport = transport
        self._client: httpx.Client | None = None

    @property
    def client(self) -> httpx.Client:
        """httpx.Client condiviso, creato lazy (supporta MockTransport nei test).

        Header X-Auth-Token inviato SOLO se un token è configurato: il
        servizio locale può girare con auth disabilitata (token vuoto).
        """
        if self._client is None or self._client.is_closed:
            headers = {"X-Auth-Token": self._token} if self._token else {}
            self._client = httpx.Client(
                base_url=self.base_url,
                headers=headers,
                timeout=self._timeout,
                transport=self._transport,
            )
        return self._client

    def close(self) -> None:
        """Chiude il client se aperto (niente lock, chiamabile sempre)."""
        if self._client is not None and not self._client.is_closed:
            self._client.close()

    # ── Endpoint ──

    def get_status(self) -> dict[str, Any]:
        """GET /status → stato (cache del servizio, funziona a cluster giù)."""
        return self._request("GET", "/status")

    def get_jobs(self) -> list[dict[str, Any]]:
        """GET /jobs → job in coda (entry + type/config/tag parsati)."""
        return self._request("GET", "/jobs")

    def add_job(
        self,
        job_type: str,
        config: str,
        tag: str | None = None,
        mode: str | None = None,
    ) -> dict[str, Any]:
        """POST /jobs → accoda ``{type, config, tag?, mode?}``."""
        payload: dict[str, Any] = {"type": job_type, "config": config}
        if tag:
            payload["tag"] = tag
        if mode:
            payload["mode"] = mode
        return self._request("POST", "/jobs", json=payload)

    def replace_queue(
        self,
        jobs: list[dict[str, str]] | None = None,
        ablation: bool = False,
    ) -> dict[str, Any]:
        """POST /queue → rimpiazza la coda (``{jobs:[...]}`` o ``{ablation:true}``)."""
        payload: dict[str, Any] = (
            {"ablation": ablation} if jobs is None else {"jobs": jobs}
        )
        return self._request("POST", "/queue", json=payload)

    def delete_job(self, tag: str) -> dict[str, Any]:
        """DELETE /jobs/{tag} → rimuove tutti i job col tag dato."""
        return self._request("DELETE", f"/jobs/{tag}")

    def pause(self) -> dict[str, Any]:
        """POST /pause → crea chain_stopped sul cluster (soft stop)."""
        return self._request("POST", "/pause")

    def resume(self) -> dict[str, Any]:
        """POST /resume → rimuove chain_stopped + tick immediato."""
        return self._request("POST", "/resume")

    def tick(self) -> dict[str, Any]:
        """POST /tick → tick manuale; timeout generoso (ssh + cold start)."""
        return self._request("POST", "/tick", timeout=90.0)

    # ── Endpoint v2 (monitor live + controllo job) ──

    def get_monitor(self) -> dict[str, Any]:
        """GET /monitor → snapshot live: stato + job_detail + samples + log tail.

        Timeout generoso: il servizio legge il log del job via ssh.
        """
        return self._request("GET", "/monitor", timeout=60.0)

    def start_job(
        self,
        job_type: str,
        config: str,
        tag: str | None = None,
    ) -> dict[str, Any]:
        """POST /jobs/start → accoda + tick immediato (parte subito se libero).

        Response: snapshot monitor + ``started_now``.
        """
        payload: dict[str, Any] = {"type": job_type, "config": config}
        if tag:
            payload["tag"] = tag
        return self._request("POST", "/jobs/start", json=payload, timeout=90.0)

    def start_batch(
        self, jobs: list[dict[str, Any]], start_now: bool = True
    ) -> dict[str, Any]:
        """POST /jobs/batch → accoda più job in ordine (+ tick se start_now).

        Response: snapshot monitor + ``started_now`` + ``queued`` (entry).
        Timeout generoso: enqueue multipli via ssh + tick.
        """
        payload: dict[str, Any] = {"jobs": jobs, "start_now": start_now}
        return self._request("POST", "/jobs/batch", json=payload, timeout=120.0)

    def kill_active(self) -> dict[str, Any]:
        """POST /kill → scancel del job attivo (409 → ApiError se nessuno)."""
        return self._request("POST", "/kill", timeout=60.0)

    def get_logs(self, lines: int = 50) -> dict[str, Any]:
        """GET /logs?lines=N → ultime N righe del log del job attivo."""
        return self._request("GET", "/logs", params={"lines": lines}, timeout=60.0)

    # ── Endpoint opzionali (possono NON esistere: il chiamatore degrada) ──

    def get_timeseries(self, tag: str, metric: str, limit: int = 200) -> dict[str, Any]:
        """GET /timeseries → serie per-step di una metrica del job col tag.

        Args:
            tag: tag del job (attivo, altrimenti ultimo sottomesso).
            metric: ``loss`` | ``reward`` | ``lr`` | ``kl``.
            limit: numero massimo di punti (il servizio sottocampiona).

        Raises:
            ApiError: 404 se l'endpoint non esiste o il tag non è noto.
        """
        return self._request(
            "GET",
            "/timeseries",
            params={"tag": tag, "metric": metric, "limit": limit},
            timeout=30.0,
        )

    def get_results(self, config: str | None = None) -> dict[str, Any]:
        """GET /results → run e metriche delle eval per config.

        Senza ``config`` (discovery): ``{"results_dirs": [...]}``.
        Args:
            config: nome della dir risultati o del config (risoluzione
                tollerante lato servizio).

        Raises:
            ApiError: 404 se l'endpoint non esiste o nessuna dir matcha.
        """
        params = {"config": config} if config else None
        return self._request("GET", "/results", params=params, timeout=60.0)

    def get_configs(self) -> Any:
        """GET /configs → config noti al servizio (lista o ``{"configs":[]}``).

        Raises:
            ApiError: 404 se l'endpoint non esiste (fallback alla copia locale).
        """
        return self._request("GET", "/configs", timeout=15.0)

    # ── Interno ──

    def _request(self, method: str, path: str, **kwargs: Any) -> Any:
        try:
            response = self.client.request(method, path, **kwargs)
        except httpx.TimeoutException as exc:
            raise ConnectionError(
                "Il servizio non risponde entro il timeout. Se è la prima "
                "chiamata dopo lo sleep, Render free tier può metterci 30-50s "
                "(cold start)."
            ) from exc
        except httpx.TransportError as exc:
            raise ConnectionError(
                "Impossibile connettersi al servizio: verifica l'URL "
                "(T2G_SERVICE_URL) e che Render sia attivo."
            ) from exc
        except httpx.HTTPError as exc:
            raise ConnectionError(
                f"Errore di trasporto HTTP: {exc.__class__.__name__}"
            ) from exc
        if response.status_code == 401:
            raise AuthError(
                "Token non valido (401): aggiorna T2G_AUTH_TOKEN (env, .env o "
                "schermata di configurazione)."
            )
        if response.status_code == 503:
            raise ApiError(503, "T2G_AUTH_TOKEN non configurato sul server.")
        if response.is_error:
            raise ApiError(response.status_code, _extract_detail(response))
        return _decode(response)


# ── App Textual ───────────────────────────────────────────────────────────────
# Gli stili vivono in remote/tui.tcss (CSS_PATH sulla App): niente CSS inline.
# Qui sotto solo helper di presentazione puri (testabili senza TUI).


# Soglia oltre la quale uno snapshot servito dalla cache merita il banner
# giallo in cima alla dashboard (5 minuti di dati fermi non sono "live").
_STALE_AFTER_SECONDS: float = 300.0

# Punti richiesti a /timeseries per le sparkline (larghezza tipica ~100 col).
_TIMESERIES_LIMIT: int = 200

# Esteri del log: parole chiave (minuscole) per livello di colorazione.
_LOG_ERROR_HINTS: tuple[str, ...] = (
    "error",
    "traceback",
    "exception",
    "failed",
    "cuda out of memory",
    "oom",
    "killed",
)
_LOG_WARN_HINTS: tuple[str, ...] = ("warning", "warn")
_LOG_PHASE_HINTS: tuple[str, ...] = ("step ", "stage", "===", "---", "phase", "epoch")


def _style_log_line(line: str) -> str:
    """Colora una riga di log per livello (errore/warning/fase/metrica).

    Le righe di metrica (``step=… loss=…``) restano a pieno colore: sono il
    contenuto principale; il resto è dim, così il rumore non compete con
    il segnale.

    Args:
        line: riga grezza del log SLURM (mai markup: viene escapata).

    Returns:
        Righe markup Textual con il colore del livello.
    """
    low = line.lower()
    escaped = escape(line)
    if any(hint in low for hint in _LOG_ERROR_HINTS):
        return f"[red]{escaped}[/red]"
    if any(hint in low for hint in _LOG_WARN_HINTS):
        return f"[yellow]{escaped}[/yellow]"
    if "step=" in low:
        return escaped
    if any(hint in low for hint in _LOG_PHASE_HINTS):
        return f"[cyan]{escaped}[/cyan]"
    return f"[dim]{escaped}[/dim]"


def _unseen_lines(shown: list[str], incoming: list[str]) -> list[str]:
    """Righe di ``incoming`` non ancora mostrate (append, non ricostruzione).

    Il log arriva come tail (finestra scorrevole): cerca il più lungo
    suffisso di ``shown`` che è prefisso di ``incoming`` e restituisce il
    resto. Senza overlap (rotazione/nuovo file) restituisce tutto.

    Args:
        shown: righe già scritte nel RichLog (finestra recente).
        incoming: nuovo tail completo dal servizio.

    Returns:
        Solo le righe da appendere (vuoto se identico).
    """
    if not shown or not incoming:
        return list(incoming)
    max_overlap = min(len(shown), len(incoming))
    for size in range(max_overlap, 0, -1):
        if shown[-size:] == incoming[:size]:
            return incoming[size:]
    return list(incoming)


def _human_age(seconds: float) -> str:
    """Età in forma compatta: 58s · 12 min · 1.5 h."""
    if seconds < 90.0:
        return f"{round(seconds)}s"
    minutes = seconds / 60.0
    if minutes < 90.0:
        return f"{round(minutes)} min"
    return f"{minutes / 60.0:.1f} h"


def _human_minutes(minutes: float) -> str:
    """Durata stimata in forma compatta: meno di 1 min · 40 min · 2.3 h."""
    if minutes < 1.0:
        return "meno di 1 min"
    if minutes < 90.0:
        return f"{round(minutes)} min"
    return f"{minutes / 60.0:.1f} h"


def _freshness_markup(snap: dict[str, Any]) -> str:
    """Chip di provenienza dati per l'header del job (source/age_seconds).

    Un pannello che presenta numeri di 5 minuti fa come live è peggio di uno
    che dichiara l'età: il chip dice sempre da dove arrivano i numeri.
    """
    source = snap.get("source")
    age = snap.get("age_seconds")
    if source is None and age is None:
        return ""
    if source == "live" or (isinstance(age, (int, float)) and age <= 0.0):
        return " · dati [green]live[/green]"
    if not isinstance(age, (int, float)):
        return " · dati cache (età n/d)"
    return f" · dati [yellow]cache · {_human_age(float(age))} fa[/yellow]"


def _active_tag(snap: dict[str, Any]) -> str | None:
    """Tag del job attivo per GET /timeseries (best effort, mai critico).

    Prova il campo esplicito di ``job_detail`` (se il servizio lo espone),
    poi parsa ``last_job`` (``id:tipo:config:tag:extra``).
    """
    detail = snap.get("job_detail")
    if isinstance(detail, dict) and detail.get("tag"):
        return str(detail["tag"])
    parts = str(snap.get("last_job") or "").split(":")
    if len(parts) >= 4 and parts[3]:
        return parts[3]
    return None


def _parse_entry(entry: str) -> tuple[str, str, str]:
    """Entry di coda ``tipo:config:tag`` → (tipo, basename config, tag)."""
    parts = entry.split(":")
    jtype = parts[0] if parts else "?"
    config = Path(parts[1]).name if len(parts) > 1 else "?"
    tag = parts[2] if len(parts) > 2 and parts[2] else "—"
    return jtype, config, tag


def _fmt_metric(value: Any) -> str:
    """Metrica → stringa compatta (— se assente, 4 cifre significative)."""
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, (int, float)):
        return f"{float(value):.4g}"
    return "—" if value is None else str(value)


# Reward breakdown: 7 componenti (src/utils/metrics.py), etichette corte.
_REWARD_ORDER: tuple[str, ...] = (
    "translation_quality_reward",
    "bleu_reward",
    "gold_structure_reward",
    "gloss_order_reward",
    "verifier_scaled_reward",
    "gloss_format_reward",
    "gloss_repetition_reward",
)
_REWARD_SHORT: dict[str, str] = {
    "translation_quality_reward": "translation",
    "bleu_reward": "bleu",
    "gold_structure_reward": "gold structure",
    "gloss_order_reward": "gloss order",
    "verifier_scaled_reward": "verifier",
    "gloss_format_reward": "format",
    "gloss_repetition_reward": "repetition",
}


def _reward_bars_markup(breakdown: dict[str, Any], width: int = 20) -> str:
    """Reward breakdown → righe ``nome ███████░░░ 0.42`` (barre orizzontali).

    Le componenti sature (~0.99, es. format/repetition nei run reali) si
    vedono a colpo d'occhio: barra piena e valore verde; quelle che portano
    segnale restano parziali. La scala è fissa 0-1 (le componenti sono
    medie di reward in [0,1]).

    Args:
        breakdown: mapping componente → punteggio medio.
        width: larghezza della barra in caratteri.

    Returns:
        Righe markup Textual (una per componente, ordine canonico).
    """
    ordered = [name for name in _REWARD_ORDER if name in breakdown]
    ordered += sorted(name for name in breakdown if name not in _REWARD_ORDER)
    rows: list[str] = []
    for name in ordered:
        try:
            value = float(breakdown[name])
        except (TypeError, ValueError):
            continue
        filled = round(min(1.0, max(0.0, value)) * width)
        bar = "█" * filled + "░" * (width - filled)
        style = "green" if value >= 0.8 else ("yellow" if value >= 0.5 else "red")
        label = escape(_REWARD_SHORT.get(name, name)[:16])
        rows.append(f"[dim]{label:<16}[/dim] {bar} [{style}]{value:.3f}[/{style}]")
    return "\n".join(rows)


class T2GScreen(Screen[None]):
    """Base delle schermate: espone l'App tipizzata con i metodi custom."""

    @property
    def t2g_app(self) -> "T2GDashApp":
        """Riferimento all'app (Textual tipizza ``app`` come generico)."""
        return self.app  # type: ignore[return-value]


class DashboardScreen(T2GScreen):
    """Monitor principale (ex-dashboard): stato + metriche live del job attivo.

    Auto-refresh ogni ``refresh_interval`` secondi (GET /monitor); refresh
    manuale con ``r``. Pannelli: banner di degrado (cluster giù OPPURE dati
    della cache più vecchi di 5 min), job attivo (riepilogo + ProgressBar
    step/total con ETA dal ritmo osservato + sparkline loss/reward per-step
    da GET /timeseries), completion samples, log tail incrementale colorato
    per livello, coda + eventi/errori come DataTable affiancate. Se
    ``/timeseries`` non esiste, le sparkline degradano a un placeholder e
    tutto il resto resta pienamente utilizzabile.

    Binding: r refresh · g queue · a add · s job singolo · S batch · k kill ·
    w replace · p pause · C campaign · R resume · t tick · L log fullscreen ·
    v risultati.
    """

    BINDINGS = [
        Binding("r", "refresh", "Refresh"),
        Binding("g", "queue", "Queue"),
        Binding("a", "add_job", "Add job"),
        Binding("s", "start_job", "Job"),
        Binding("S", "start_batch", "Batch"),
        Binding("k", "kill_job", "KILL job"),
        Binding("w", "replace_queue", "Replace queue"),
        Binding("p", "pause", "Pause"),
        Binding("C", "campaign", "Campaign"),
        Binding("R", "resume", "Resume"),
        Binding("t", "tick", "Tick"),
        Binding("L", "log_full", "Log"),
        Binding("v", "results", "Risultati"),
    ]

    def compose(self) -> ComposeResult:
        yield Header()
        yield Static(id="banner")
        yield LoadingIndicator(id="loading")
        with Vertical(id="job-box"):
            yield Static(id="job-panel")
            yield ProgressBar(id="job-progress", show_percentage=False, show_eta=False)
            yield Static(id="eta-hint")
            with Horizontal(id="spark-row"):
                with Vertical(id="loss-box"):
                    yield Static("", id="loss-stats", classes="spark-stats")
                    yield Sparkline(id="loss-spark", min_color="green", max_color="red")
                with Vertical(id="reward-box"):
                    yield Static("", id="reward-stats", classes="spark-stats")
                    yield Sparkline(
                        id="reward-spark", min_color="red", max_color="green"
                    )
        with Vertical(id="samples-panel"):
            yield RichLog(id="samples-log", highlight=False, markup=True)
        with Vertical(id="tail-panel"):
            yield RichLog(id="tail-log", highlight=False, markup=True, max_lines=240)
        with Horizontal(id="bottom-row"):
            with Vertical(id="queue-panel"):
                yield DataTable(
                    id="queue-table", cursor_type="none", zebra_stripes=True
                )
            with Vertical(id="events-panel"):
                yield DataTable(
                    id="events-table", cursor_type="none", zebra_stripes=True
                )
        yield Footer()

    def on_mount(self) -> None:
        # Stato per l'ETA (osservazioni (monotonic, step) del job attivo) e
        # per il log tail incrementale (righe già mostrate + job corrente).
        self._step_obs: list[tuple[float, float]] = []
        self._step_job_key: str | None = None
        self._tail_shown: list[str] = []
        self._tail_job_key: str | None = None
        self._tail_placeholder = False
        self.query_one("#samples-panel").border_title = "Completion samples (ultime 8)"
        self.query_one("#tail-panel").border_title = "Log tail del job attivo"
        self.query_one("#queue-panel").border_title = "Coda"
        self.query_one("#events-panel").border_title = "Eventi & errori recenti"
        self.query_one("#queue-table", DataTable).add_columns(
            "#", "tipo", "config", "tag"
        )
        self.query_one("#events-table", DataTable).add_columns(
            "ora", "tipo", "dettaglio"
        )
        self.set_interval(self.t2g_app.refresh_interval, self.t2g_app.refresh_monitor)
        self.refresh_view()
        self.t2g_app.run_worker(self.t2g_app.refresh_monitor())

    # ── Azioni (binding) ──

    def action_refresh(self) -> None:
        self.t2g_app.run_worker(self.t2g_app.refresh_monitor())

    def action_queue(self) -> None:
        self.t2g_app.switch_screen("queue")

    def action_add_job(self) -> None:
        self.t2g_app.switch_screen("add_job")

    def action_start_job(self) -> None:
        self.t2g_app.switch_screen("start_job")

    def action_start_batch(self) -> None:
        self.t2g_app.switch_screen("batch_start")

    def action_campaign(self) -> None:
        self.t2g_app.switch_screen("campaign")

    def action_results(self) -> None:
        self.t2g_app.switch_screen("results")

    def action_kill_job(self) -> None:
        self.t2g_app.confirm_kill()

    def action_replace_queue(self) -> None:
        self.t2g_app.switch_screen("replace")

    def action_pause(self) -> None:
        self.t2g_app.run_worker(self.t2g_app.pause())

    def action_resume(self) -> None:
        self.t2g_app.run_worker(self.t2g_app.resume())

    def action_tick(self) -> None:
        self.t2g_app.run_worker(self.t2g_app.tick())

    def action_log_full(self) -> None:
        self.t2g_app.switch_screen("biglog")

    # ── Rendering ──

    def set_busy(self, busy: bool) -> None:
        """Mostra/nasconde lo spinner (usato dalle operazioni lente)."""
        self.query_one("#loading", LoadingIndicator).display = busy

    def refresh_view(self) -> None:
        """Ridisegna i pannelli a partire da ``app.monitor_snapshot``."""
        if not self.is_mounted:
            return
        snap = self.t2g_app.monitor_snapshot or self.t2g_app.status
        if snap is None:
            self.query_one("#job-panel", Static).update("Caricamento dal servizio…")
            return

        banner = self.query_one("#banner", Static)
        job_box = self.query_one("#job-panel", Static)
        samples_log = self.query_one("#samples-log", RichLog)
        tail_log = self.query_one("#tail-log", RichLog)

        reachable = bool(snap.get("cluster_reachable", False))
        self.t2g_app.sub_title = self.t2g_app.config.url if self.t2g_app.config else ""
        age = snap.get("age_seconds")
        age_s = float(age) if isinstance(age, (int, float)) else None
        stale = (
            snap.get("source") != "live"
            and age_s is not None
            and age_s > _STALE_AFTER_SECONDS
        )
        if not reachable:
            banner.display = True
            banner.update(
                "[yellow]⚠ CLUSTER IRRAGGIUNGIBILE — mostrato l'ultimo stato "
                "noto dalla cache del servizio[/yellow]"
            )
        elif stale and age_s is not None:
            banner.display = True
            banner.update(
                "[yellow]⚠ DATI NON FRECHI — ultimo aggiornamento "
                f"{_human_age(age_s)} fa (cache del servizio)[/yellow]"
            )
        else:
            banner.display = False

        job_box.update(self._job_text(snap))
        self._update_progress_and_eta(snap)
        self._update_sparklines()
        samples_log.clear()
        for line in (snap.get("samples") or [])[-8:]:
            samples_log.write(line)
        if not snap.get("samples"):
            samples_log.write("[dim]— nessun sample disponibile —[/dim]")
        self._append_tail(snap, tail_log)
        self._render_queue(snap)
        self._render_events(snap)

    def _job_text(self, snap: dict[str, Any]) -> str:
        active = snap.get("active_job")
        detail = snap.get("job_detail")
        stopped = bool(snap.get("stopped"))
        last_tick = escape(str(snap.get("last_tick_at") or "mai"))
        reach = (
            "[green]ok[/green]" if snap.get("cluster_reachable") else "[red]giù[/red]"
        )
        stop_txt = "[red]PAUSA[/red]" if stopped else "[green]attivo[/green]"

        header = (
            f"cluster {reach} · catena {stop_txt} · tick "
            f"{last_tick}{_freshness_markup(snap)}"
        )
        if not isinstance(active, dict) or not active:
            queue = snap.get("queue") or []
            nxt = "\n".join(f"  [dim]{escape(str(e))}[/dim]" for e in queue[:3])
            hint = (
                f"Prossimi {min(3, len(queue))} in coda:\n{nxt}"
                if queue
                else "Coda vuota."
            )
            return "\n".join(
                [
                    header,
                    "[bold]Nessun job attivo[/bold] — premi [b]s[/b] per un job, "
                    f"[b]S[/b] per un batch, [b]g[/b] per la coda ({len(queue)} job)",
                    hint,
                ]
            )

        lines = [
            header,
            f"Job: [bold]{escape(str(active.get('name') or '?'))}[/bold] "
            f"[dim](id {escape(str(active.get('id') or '?'))} · "
            f"{escape(str(active.get('state') or '?'))})[/dim]",
        ]
        if detail:
            # Phase badge (live status): sft=giallo, grpo=verde, eval=ciano.
            phase = detail.get("phase")
            if phase:
                badge = {
                    "sft": "[yellow]SFT[/yellow]",
                    "sft_eval": "[yellow]SFT·eval[/yellow]",
                    "grpo": "[green]GRPO[/green]",
                    "grpo_eval": "[green]GRPO·eval[/green]",
                    "eval": "[cyan]EVAL[/cyan]",
                }.get(str(phase), escape(str(phase)))
                src = " [dim]live[/dim]" if detail.get("source") == "live" else ""
                lines.append(f"Fase: {badge}{src}")
            step = detail.get("step")
            total = detail.get("total_steps")
            if step is not None and total:
                pct = min(100.0, 100.0 * float(step) / max(1, float(total)))
                lines.append(f"step [bold]{step}/{total}[/bold] ({pct:.1f}%)")
            metrics = []
            if detail.get("loss") is not None:
                metrics.append(f"loss [bold]{escape(str(detail['loss']))}[/bold]")
            if detail.get("reward") is not None:
                metrics.append(f"reward [bold]{escape(str(detail['reward']))}[/bold]")
            if detail.get("reward_avg") is not None:
                metrics.append(
                    f"avg reward [dim]{escape(str(detail['reward_avg']))}[/dim]"
                )
            if detail.get("lr") is not None:
                metrics.append(f"lr [dim]{escape(str(detail['lr']))}[/dim]")
            if detail.get("eval_progress"):
                metrics.append(
                    f"eval [cyan]{escape(str(detail['eval_progress']))}[/cyan]"
                )
            if metrics:
                lines.append(" | ".join(metrics))
            # Routine in-train eval (SFT holdout): ADDITIONAL line — the train
            # step counter above is NOT overwritten (the eval loop never
            # touches step/loss/lr in the live status, see HighPrecisionLogCallback).
            if detail.get("eval_active") and str(phase) in ("sft", "grpo"):
                lines.append(
                    "[cyan]⏳ eval di routine in corso (il contatore di training resta attivo)[/cyan]"
                )
            if detail.get("sft_active"):
                sft = [
                    f"SFT: step {detail.get('sft_step') or '?'}/{detail.get('sft_total') or '?'}"
                ]
                if detail.get("sft_loss"):
                    sft.append(f"loss {escape(str(detail['sft_loss']))}")
                if detail.get("sft_eval_loss"):
                    sft.append(f"eval {escape(str(detail['sft_eval_loss']))}")
                if detail.get("sft_eval_loss_best"):
                    sft.append(f"best {escape(str(detail['sft_eval_loss_best']))}")
                lines.append("[yellow]" + " · ".join(sft) + "[/yellow]")
            if detail.get("eval_label"):
                lines.append(f"[cyan]eval: {escape(str(detail['eval_label']))}[/cyan]")
            for key, value in (detail.get("eval_metrics") or {}).items():
                lines.append(f"[cyan]  {escape(str(key))}: {escape(str(value))}[/cyan]")
        else:
            lines.append(
                "[dim](dettagli non disponibili — log non ancora prodotto)[/dim]"
            )
        return "\n".join(lines)

    def _update_progress_and_eta(self, snap: dict[str, Any]) -> None:
        """ProgressBar step/total + ETA dal ritmo osservato tra i poll.

        I training sono da ~5000 passi (≈6h): sapere quanto resta è il dato
        più utile del pannello. Il ritmo si stima dalle osservazioni locali
        (step, monotonic) del job attivo; a cambio job la stima si azzera.
        """
        detail = snap.get("job_detail")
        active = snap.get("active_job")
        step = detail.get("step") if isinstance(detail, dict) else None
        total = detail.get("total_steps") if isinstance(detail, dict) else None
        bar = self.query_one("#job-progress", ProgressBar)
        eta_box = self.query_one("#eta-hint", Static)
        if not isinstance(step, (int, float)) or not total:
            bar.update(total=100, progress=0.0)
            eta_box.update("")
            return
        step_f = float(step)
        total_f = float(total)
        pct = min(100.0, 100.0 * step_f / max(1.0, total_f))
        bar.update(total=100, progress=pct)

        job_key = (
            f"{(active or {}).get('id')}:{(active or {}).get('name')}"
            if isinstance(active, dict)
            else "??"
        )
        now = time.monotonic()
        if job_key != self._step_job_key:
            self._step_obs = []
            self._step_job_key = job_key
        if not self._step_obs or self._step_obs[-1][1] != step_f:
            self._step_obs.append((now, step_f))
            self._step_obs = self._step_obs[-32:]

        rate = self._step_rate()
        if rate is not None and rate > 0.0:
            remaining = (total_f - step_f) / rate  # minuti
            if remaining > 0.0:
                eta_box.update(
                    f"ritmo ~{rate:.1f} step/min · stimati "
                    f"~{_human_minutes(remaining)} alla fine"
                )
            else:
                eta_box.update(f"ritmo ~{rate:.1f} step/min · step finale raggiunto")
        else:
            eta_box.update(
                "[dim]ritmo non ancora stimabile (attesi due poll con step "
                "diverso)[/dim]"
            )

    def _step_rate(self) -> float | None:
        """Step/min dalle osservazioni locali; None se la stima è prematura."""
        obs = self._step_obs
        if len(obs) < 2:
            return None
        (t0, s0), (t1, s1) = obs[0], obs[-1]
        elapsed = t1 - t0
        if elapsed < 1.0:  # troppo poco tempo: stima inaffidabile
            return None
        return max(0.0, (s1 - s0) / elapsed * 60.0)

    def _update_sparklines(self) -> None:
        """Sparkline loss/reward per-step da ``app.timeseries`` (o fallback)."""
        series = self.t2g_app.timeseries
        for metric, spark_id, stats_id in (
            ("loss", "#loss-spark", "#loss-stats"),
            ("reward", "#reward-spark", "#reward-stats"),
        ):
            spark = self.query_one(spark_id, Sparkline)
            stats = self.query_one(stats_id, Static)
            payload = series.get(metric)
            points = (
                [p for p in (payload or {}).get("points") or [] if isinstance(p, dict)]
                if isinstance(payload, dict)
                else []
            )
            values = [
                float(p["value"])
                for p in points
                if isinstance(p.get("value"), (int, float))
            ]
            if not values:
                spark.display = False
                if not self.t2g_app.timeseries_available:
                    stats.update(
                        f"[dim]{metric}: serie per-step non disponibile "
                        "(il servizio non espone /timeseries)[/dim]"
                    )
                else:
                    stats.update(f"[dim]{metric}: serie per-step non disponibile[/dim]")
                continue
            spark.display = True
            spark.data = values
            stats.update(
                f"{metric} attuale [b]{values[-1]:.4g}[/b] · min "
                f"{min(values):.4g} · max {max(values):.4g} · {len(values)} punti"
            )

    def _append_tail(self, snap: dict[str, Any], tail_log: RichLog) -> None:
        """Scrive SOLO le righe nuove del log tail (niente ricostruzione).

        Il clear+rewrite a ogni poll sfarfalla e azzera lo scroll
        dell'utente: si tiene traccia delle righe già mostrate e si appende
        il delta, con reset solo a cambio job. Ogni riga è colorata per
        livello (``_style_log_line``).
        """
        active = snap.get("active_job")
        job_key = (
            str((active or {}).get("id") or "") if isinstance(active, dict) else ""
        )
        lines = [str(line) for line in snap.get("log_tail") or []]
        if job_key != self._tail_job_key:
            tail_log.clear()
            self._tail_shown = []
            self._tail_placeholder = False
            self._tail_job_key = job_key
        if not lines:
            if not self._tail_shown:
                tail_log.write("[dim]— log vuoto —[/dim]")
                self._tail_placeholder = True
            return
        if self._tail_placeholder:
            tail_log.clear()
            self._tail_placeholder = False
        new_lines = _unseen_lines(self._tail_shown, lines)
        for line in new_lines:
            tail_log.write(_style_log_line(line))
        self._tail_shown = (self._tail_shown + new_lines)[-64:]

    def _render_queue(self, snap: dict[str, Any]) -> None:
        """Coda come DataTable (niente cap a 5 entry: si vede tutto)."""
        queue = [str(entry) for entry in snap.get("queue") or []]
        self.query_one("#queue-panel", Vertical).border_title = (
            f"Coda — {len(queue)} job · g: lista completa"
        )
        table = self.query_one("#queue-table", DataTable)
        table.clear()
        if not queue:
            table.add_row(
                Text("—", style="dim"), Text("coda vuota", style="dim"), "", ""
            )
            return
        for pos, entry in enumerate(queue, start=1):
            jtype, config, tag = _parse_entry(entry)
            table.add_row(str(pos), jtype, config, tag)

    def _render_events(self, snap: dict[str, Any]) -> None:
        """Eventi recenti + errori in una DataTable (errori in rosso)."""
        errors = [str(e) for e in snap.get("errors_recent") or []][-3:]
        events = [e for e in snap.get("events") or [] if isinstance(e, dict)][-8:]
        title = "Eventi & errori recenti"
        if errors:
            title += f" — {len(errors)} errori"
        self.query_one("#events-panel", Vertical).border_title = title
        table = self.query_one("#events-table", DataTable)
        table.clear()
        for err in errors:
            table.add_row(
                Text("—", style="dim"),
                Text("errore", style="red"),
                Text(err[:100], style="red"),
            )
        for event in events:
            ts = str(event.get("ts", ""))[:19].replace("T", " ")
            table.add_row(
                Text(ts, style="dim"),
                str(event.get("type", "")),
                str(event.get("detail", ""))[:80],
            )
        if not errors and not events:
            table.add_row(
                Text("—", style="dim"), Text("nessun evento", style="dim"), ""
            )


class QueueScreen(T2GScreen):
    """Coda dei job (GET /jobs): posizione, tipo, config e tag.

    Da qui: ``d`` cancella per tag (con conferma), ``a`` apre il form di
    accodamento, ``r`` aggiorna, ``Esc`` torna alla dashboard.
    """

    BINDINGS = [
        Binding("a", "add_job", "Add job"),
        Binding("d", "delete_job", "Delete tag"),
        Binding("r", "refresh", "Refresh"),
        Binding("escape", "go_back", "Back"),
    ]

    def compose(self) -> ComposeResult:
        yield Header()
        yield Static(
            "Coda — [b]d[/b] cancella per tag · [b]a[/b] accoda · [b]r[/b] aggiorna",
            classes="hint",
        )
        yield DataTable(id="queue", cursor_type="row", zebra_stripes=True)
        yield Footer()

    def on_mount(self) -> None:
        self.query_one("#queue", DataTable).add_columns("Pos", "Type", "Config", "Tag")
        self.t2g_app.run_worker(self.t2g_app.refresh_jobs())

    def reload(self) -> None:
        """Ricompone la tabella da ``app.jobs``.

        NOTA: la chiave di riga è l'entry COMPLETA (type:config:tag), mai il
        solo tag — train e eval della stessa cella condividono il tag e la
        DataTable di Textual solleva DuplicateKey con chiavi duplicate
        (bug: la coda appariva vuota dopo l'accodamento train+eval).
        """
        if not self.is_mounted:
            return
        table = self.query_one("#queue", DataTable)
        table.clear()
        for pos, job in enumerate(self.t2g_app.jobs, start=1):
            entry = str(
                job.get(
                    "entry", f"{job.get('type')}:{job.get('config')}:{job.get('tag')}"
                )
            )
            table.add_row(
                str(pos),
                str(job.get("type", "")),
                Path(str(job.get("config", ""))).name,
                str(job.get("tag", "")),
                key=entry,
            )

    # ── Azioni (binding) ──

    def action_add_job(self) -> None:
        self.t2g_app.switch_screen("add_job")

    def action_refresh(self) -> None:
        self.t2g_app.run_worker(self.t2g_app.refresh_jobs())

    def action_go_back(self) -> None:
        self.t2g_app.switch_screen("dashboard")

    def action_delete_job(self) -> None:
        table = self.query_one("#queue", DataTable)
        if table.row_count == 0:
            self.t2g_app.notify(
                "[yellow]Coda vuota[/yellow]", severity="warning", timeout=4
            )
            return
        row_index = table.cursor_row
        if row_index is None:
            self.t2g_app.notify(
                "[yellow]Nessuna riga selezionata[/yellow]",
                severity="warning",
                timeout=4,
            )
            return
        row = table.get_row_at(row_index)
        tag = str(row[3]) if row else ""
        if not tag:
            self.t2g_app.notify(
                "[yellow]Riga senza tag[/yellow]", severity="warning", timeout=4
            )
            return
        self.t2g_app.push_screen(
            ConfirmScreen(f"Rimuovere TUTTI i job con tag '{tag}' dalla coda?"),
            lambda ok: self._confirmed_delete(bool(ok), tag),
        )

    def _confirmed_delete(self, ok: bool, tag: str) -> None:
        if ok:
            self.t2g_app.run_worker(self.t2g_app.delete_job(tag))


class AddJobScreen(T2GScreen):
    """Form per accodare (POST /jobs) o AVVIARE (POST /jobs/start) un job.

    ``start_mode=True`` (binding ``s`` dal monitor): il submit chiama
    ``/jobs/start`` (enqueue + tick immediato) — e se il tipo è ``train``
    con la checkbox "Accoda anche la eval" attiva (default), usa
    ``/jobs/batch`` per accodare train+eval insieme. Modalità normale
    (binding ``a``): semplice enqueue.
    """

    def __init__(self, start_mode: bool = False) -> None:
        super().__init__()
        self.start_mode = start_mode

    BINDINGS = [Binding("escape", "go_back", "Back")]

    def compose(self) -> ComposeResult:
        title = (
            "AVVIA un job (accoda + tick immediato - parte subito se libero)"
            if self.start_mode
            else "Aggiungi un job alla coda"
        )
        # Config noti: GET /configs se il servizio lo espone, altrimenti la
        # copia locale CONFIG_NAMES (aggiornata a mount dell'app).
        configs = self.t2g_app.available_configs or list(CONFIG_NAMES)
        yield Header()
        yield Static(title, classes="title")
        yield Select(
            [("train", "train"), ("eval", "eval")],
            prompt="Tipo",
            value="train",
            id="type",
        )
        yield Select(
            [(name, name) for name in configs],
            prompt="Config",
            value=configs[0],
            id="config",
        )
        yield Input(
            placeholder="Tag (opzionale - di default derivato dal config)", id="tag"
        )
        # Accodare anche l'eval dopo il train (train + checkbox eval): il
        # batch endpoint enqueue train+eval. Disponibile sia in start_mode
        # ('s') sia in normal mode ('a').
        yield Checkbox(
            "Accoda anche la eval dopo il train (train+eval insieme)",
            value=True,
            id="also-eval",
        )
        submit_label = "Avvia ora" if self.start_mode else "Accoda"
        yield Button(submit_label, variant="primary", id="submit")
        yield Footer()

    def on_mount(self) -> None:
        self._last_derived_tag: str | None = None
        self._prefill_tag()
        self._sync_eval_checkbox()
        self.query_one("#type", Select).focus()

    # ── Azioni (binding) ──

    def action_go_back(self) -> None:
        self.t2g_app.switch_screen("dashboard")

    # ── Eventi widget ──

    def on_select_changed(self, event: Select.Changed) -> None:
        if event.select.id == "config":
            self._prefill_tag()
        elif event.select.id == "type":
            self._sync_eval_checkbox()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "submit":
            self._submit()

    # ── Interno ──

    def _config_value(self) -> str:
        return str(self.query_one("#config", Select).value)

    def _type_value(self) -> str:
        return str(self.query_one("#type", Select).value)

    def _sync_eval_checkbox(self) -> None:
        """La checkbox train+eval ha senso solo per type=train."""
        checkbox = self.query_one("#also-eval", Checkbox) if self.start_mode else None
        if checkbox is not None:
            checkbox.display = self._type_value() == "train"

    def _prefill_tag(self) -> None:
        """Suggerisce il tag derivato dal config (stessa regola del driver)."""
        config = self._config_value()
        derived = config.replace("_", "-")
        tag = self.query_one("#tag", Input)
        if not tag.value or tag.value == self._last_derived_tag:
            tag.value = derived
            self._last_derived_tag = derived

    def _submit(self) -> None:
        job_type = self._type_value()
        config = self._config_value()
        tag = self.query_one("#tag", Input).value.strip() or None
        checkbox = self.query_one("#also-eval", Checkbox)
        also_eval = bool(checkbox.value) and job_type == "train"
        if also_eval:
            # Batch train+eval: un solo endpoint, la eval resta in coda dopo.
            # Disponibile sia in start_mode ('s') che in normal mode ('a').
            if self.start_mode:
                self.t2g_app.run_worker(
                    self.t2g_app.start_batch_job(job_type, config, tag)
                )
            else:
                self.t2g_app.run_worker(self._do_submit_batch(job_type, config, tag))
        elif self.start_mode:
            self.t2g_app.run_worker(self.t2g_app.start_job(job_type, config, tag))
        else:
            self.t2g_app.run_worker(self._do_submit(job_type, config, tag))

    async def _do_submit(self, job_type: str, config: str, tag: str | None) -> None:
        if await self.t2g_app.add_job(job_type, config, tag):
            self.t2g_app.switch_screen("dashboard")

    async def _do_submit_batch(
        self, job_type: str, config: str, tag: str | None
    ) -> None:
        """Accoda train+eval insieme (POST /jobs/batch senza tick).

        Modalità 'a': entrambi i job finiscono in CODA (nessun tick
        immediato — parte il primo solo quando la QoS libera o al prossimo
        tick).
        """
        jobs: list[dict[str, Any]] = [{"type": job_type, "config": config}]
        if tag:
            jobs[0]["tag"] = tag
        eval_job: dict[str, Any] = {"type": "eval", "config": config}
        if tag:
            eval_job["tag"] = tag
        jobs.append(eval_job)
        await self.t2g_app.start_batch(jobs, start_now=False)
        self.t2g_app.switch_screen("dashboard")


class BatchStartScreen(T2GScreen):
    """Avvio batch multi-config con checkbox (binding ``S`` dal monitor).

    Una checkbox per ogni config noto; per ognuno selezionato si crea
    ``train+eval`` (default), ``train`` o ``eval`` in base alla Select.
    Submit → POST /jobs/batch (enqueue in ordine + tick): i job vengono
    AGGIUNTI in coda a quella esistente (append, non replace).
    """

    BINDINGS = [Binding("escape", "go_back", "Back")]

    def compose(self) -> ComposeResult:
        yield Header()
        yield Static("Avvio batch — seleziona i config", classes="title")
        yield Static(
            "I job vengono AGGIUNTI in coda (append): il primo parte subito "
            "se la coda è libera, gli altri avanzano coi tick.",
            classes="hint",
        )
        with VerticalScroll(id="config-list"):
            for name in self.t2g_app.available_configs or CONFIG_NAMES:
                yield Checkbox(name, id=f"cfg-{name}")
        yield Static("Per ogni config selezionato accoda:", classes="hint")
        yield Select(
            [
                ("train + eval", "train+eval"),
                ("solo train", "train"),
                ("solo eval", "eval"),
            ],
            value="train+eval",
            id="mode",
        )
        yield Checkbox("Avvia subito (tick immediato)", value=True, id="start-now")
        yield Button("Accoda selezione", variant="primary", id="submit")
        yield Footer()

    # ── Azioni (binding) ──

    def action_go_back(self) -> None:
        self.t2g_app.switch_screen("dashboard")

    # ── Eventi widget ──

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "submit":
            self._submit()

    # ── Interno ──

    def _selected_configs(self) -> list[str]:
        selected: list[str] = []
        for name in self.t2g_app.available_configs or CONFIG_NAMES:
            try:
                checkbox = self.query_one(f"#cfg-{name}", Checkbox)
            except Exception:
                continue
            if checkbox.value:
                selected.append(name)
        return selected

    def _submit(self) -> None:
        selected = self._selected_configs()
        if not selected:
            self.t2g_app.notify(
                "[yellow]Nessun config selezionato — spunta almeno una checkbox[/yellow]",
                severity="warning",
                timeout=5,
            )
            return
        mode = str(self.query_one("#mode", Select).value)
        start_now = bool(self.query_one("#start-now", Checkbox).value)

        jobs: list[dict[str, str]] = []
        for name in selected:
            if mode in ("train+eval", "train"):
                jobs.append({"type": "train", "config": name})
            if mode in ("train+eval", "eval"):
                jobs.append({"type": "eval", "config": name})

        label_mode = {
            "train+eval": "train+eval",
            "train": "solo train",
            "eval": "solo eval",
        }
        self.t2g_app.push_screen(
            ConfirmScreen(
                f"Accodare {len(jobs)} job ({len(selected)} config × "
                f"{label_mode.get(mode, mode)})?\n"
                "La coda esistente riceve i job IN CODA (append)."
            ),
            lambda ok: self._confirmed(bool(ok), jobs, start_now),
        )

    def _confirmed(self, ok: bool, jobs: list[dict[str, str]], start_now: bool) -> None:
        if ok:
            self.t2g_app.run_worker(self.t2g_app.start_batch(jobs, start_now))


class LogScreen(T2GScreen):
    """Log del job attivo a schermo intero (GET /logs, auto-refresh 10s).

    Aperta con ``L`` dal monitor; ``Esc`` torna al monitor. Come il tail
    della dashboard, le righe sono incrementalmente appese (niente clear a
    ogni poll) e colorate per livello; il reset avviene solo a cambio file
    di log (job diverso).
    """

    BINDINGS = [
        Binding("r", "refresh", "Refresh"),
        Binding("escape", "go_back", "Back"),
    ]

    def compose(self) -> ComposeResult:
        yield Header()
        yield Static(id="log-path", classes="hint")
        with Vertical(id="big-log"):
            yield RichLog(
                id="big-richtext", highlight=False, markup=True, max_lines=400
            )
        yield Footer()

    def on_mount(self) -> None:
        self._shown: list[str] = []
        self._log_key: str | None = None
        self._placeholder = False
        self.set_interval(self.t2g_app.refresh_interval, self.refresh_logs)
        self.refresh_logs()

    # ── Azioni (binding) ──

    def action_refresh(self) -> None:
        self.refresh_logs()

    def action_go_back(self) -> None:
        self.t2g_app.switch_screen("dashboard")

    # ── Interno ──

    def refresh_logs(self) -> None:
        self.t2g_app.run_worker(self._do_refresh())

    async def _do_refresh(self) -> None:
        if self.t2g_app.client is None:
            return
        try:
            result = await asyncio.to_thread(self.t2g_app.client.get_logs, 200)
        except RemoteServiceError as exc:
            self.t2g_app.notify(
                f"[red]{escape(str(exc))}[/red]", severity="error", timeout=8
            )
            return
        if not self.is_mounted:
            return
        path_box = self.query_one("#log-path", Static)
        rich = self.query_one("#big-richtext", RichLog)
        path = str(result.get("log_path") or "?")
        path_box.update(f"Log: {escape(path)}")
        lines = [str(line) for line in result.get("lines") or []]
        if path != self._log_key:
            rich.clear()
            self._shown = []
            self._placeholder = False
            self._log_key = path
        if not lines:
            if not self._shown:
                rich.write("[dim]— log vuoto o nessun job attivo —[/dim]")
                self._placeholder = True
            return
        if self._placeholder:
            rich.clear()
            self._placeholder = False
        new_lines = _unseen_lines(self._shown, lines)
        for line in new_lines:
            rich.write(_style_log_line(line))
        self._shown = (self._shown + new_lines)[-120:]


class ResultsScreen(T2GScreen):
    """Risultati delle eval per config (binding ``v``; GET /results).

    DataTable con le run del config selezionato (metriche chiave:
    rouge_l, exact_match, validity, pass@1) + reward breakdown a 7
    componenti come barre orizzontali dell'ultima run. Se ``/results``
    non esiste (servizio precedente all'endpoint) o il config non ha run,
    degrada a un messaggio esplicito: mai un crash.
    """

    BINDINGS = [
        Binding("r", "refresh", "Refresh"),
        Binding("escape", "go_back", "Back"),
    ]

    def compose(self) -> ComposeResult:
        configs = self.t2g_app.available_configs or list(CONFIG_NAMES)
        yield Header()
        yield Static("Risultati delle eval per config", classes="title")
        yield Select(
            [(name, name) for name in configs],
            prompt="Config",
            value=configs[0],
            id="results-config",
        )
        yield Static("Caricamento…", id="results-summary", classes="hint")
        with VerticalScroll(id="results-scroll"):
            yield DataTable(id="runs-table", cursor_type="none", zebra_stripes=True)
            with Vertical(id="reward-panel"):
                yield Static(id="reward-bars")
        yield Footer()

    def on_mount(self) -> None:
        table = self.query_one("#runs-table", DataTable)
        table.add_columns("run", "rouge_l", "exact", "validity", "pass@1")
        self.query_one("#reward-panel").border_title = "Reward breakdown — ultima run"
        self.t2g_app.run_worker(self._discover_and_load())

    # ── Azioni (binding) ──

    def action_refresh(self) -> None:
        self.t2g_app.run_worker(self._load_current())

    def action_go_back(self) -> None:
        self.t2g_app.switch_screen("dashboard")

    # ── Eventi widget ──

    def on_select_changed(self, event: Select.Changed) -> None:
        if event.select.id == "results-config":
            self.t2g_app.run_worker(self._load_current())

    # ── Interno ──

    async def _discover_and_load(self) -> None:
        """Discovery (GET /results senza config) → opzioni reali, poi load.

        La discovery elenca le dir con run effettive; se fallisce (endpoint
        assente o errore) si usa la lista dei config noti.
        """
        client = self.t2g_app.client
        dirs: list[str] = []
        if client is not None:
            try:
                discovery = await asyncio.to_thread(client.get_results)
            except RemoteServiceError:
                dirs = []
            else:
                if isinstance(discovery, dict):
                    dirs = [str(d) for d in discovery.get("results_dirs") or []]
        if not self.is_mounted:
            return
        options = dirs or list(self.t2g_app.available_configs or CONFIG_NAMES)
        select = self.query_one("#results-config", Select)
        select.set_options([(name, name) for name in options])
        if select.value != options[0]:
            select.value = options[0]  # genera Select.Changed → _load_current
        await self._load(options[0])

    async def _load_current(self) -> None:
        value = self.query_one("#results-config", Select).value
        if value in (None, ""):
            return
        await self._load(str(value))

    async def _load(self, config: str) -> None:
        """GET /results?config=… → tabella run + barre reward (o fallback)."""
        client = self.t2g_app.client
        if client is None:
            return
        try:
            payload = await asyncio.to_thread(client.get_results, config)
        except ApiError as exc:
            if not self.is_mounted:
                return
            self.query_one("#results-summary", Static).update(
                f"[yellow]Nessun risultato per '{escape(config)}' "
                f"(HTTP {exc.status_code}: endpoint assente o config "
                "senza run)[/yellow]"
            )
            self._clear_outputs()
            return
        except RemoteServiceError as exc:
            if not self.is_mounted:
                return
            self.query_one("#results-summary", Static).update(
                f"[red]{escape(str(exc))}[/red]"
            )
            self._clear_outputs()
            return
        if not self.is_mounted or not isinstance(payload, dict):
            return
        self._render_payload(config, payload)

    def _render_payload(self, config: str, payload: dict[str, Any]) -> None:
        """Popola summary, tabella run e barre reward dal payload /results."""
        runs = [r for r in payload.get("runs") or [] if isinstance(r, dict)]
        source = payload.get("source")
        age = payload.get("age_seconds")
        if source == "live" or (isinstance(age, (int, float)) and float(age) <= 0.0):
            fresh = "live"
        elif isinstance(age, (int, float)):
            fresh = f"cache · {_human_age(float(age))} fa"
        else:
            fresh = "cache"
        self.query_one("#results-summary", Static).update(
            f"{escape(str(payload.get('config') or config))} · "
            f"{len(runs)} run · dati {fresh}"
        )
        table = self.query_one("#runs-table", DataTable)
        table.clear()
        for run in runs:
            metrics = run.get("metrics") or {}
            table.add_row(
                str(run.get("run_id", "?")),
                _fmt_metric(metrics.get("rouge_l_mean")),
                _fmt_metric(metrics.get("exact_match")),
                _fmt_metric(metrics.get("validity_rate")),
                _fmt_metric(metrics.get("pass_at_1")),
            )
        if not runs:
            table.add_row("—", "nessuna run registrata", "", "", "")
        breakdown: dict[str, Any] = {}
        if runs:
            metrics = runs[-1].get("metrics") or {}
            if isinstance(metrics.get("reward_breakdown"), dict):
                breakdown = metrics["reward_breakdown"]
        self.query_one("#reward-bars", Static).update(
            _reward_bars_markup(breakdown)
            if breakdown
            else "[dim]reward breakdown non presente nelle metriche "
            "dell'ultima run[/dim]"
        )

    def _clear_outputs(self) -> None:
        self.query_one("#runs-table", DataTable).clear()
        self.query_one("#reward-bars", Static).update("")


# ── Campaign summary lines (reusable: shown in the CampaignScreen and in
# the confirmation) — order = app.py ABLATION_MODELS (ordine di riuso). ───

_CAMPAIGN_LINES: list[str] = [
    "1. baseline-zero-shot               (eval-only) — base + Trie, CACHEA la baseline --compare",
    "2. baseline-zero-shot-no-grammar    (eval-only) — lower bound senza vincolo",
    "3. baseline-few-shot                (eval-only) — base + few-shot retrieval",
    "4. sft-zero-shot                    (train+eval) — addestra l'adapter SFT",
    "5. grpo-zero-shot                   (train+eval) — GRPO dal base, zero-shot",
    "6. grpo-few-shot                    (train+eval) — GRPO dal base, few-shot",
    "7. sft-grpo-zero-shot               (train+eval) — SFT→GRPO zero-shot",
    "8. sft-grpo-few-shot                (train+eval) — SFT→GRPO few-shot",
    "9. ablations-decoding-no-grammar    (train+eval) — GRPO senza vincolo simbolico",
    "10. ablations-decoding-hot-rollout  (train+eval) — rollout sampler T=1.3",
    "11. ablations-rewards-edit-validity (train+eval) — reward edit-validity singola",
    "12. ablations-rewards-historical-stack (train+eval) — stack storico su init zero-shot",
    "13. ablations-loss-dr-grpo          (train+eval) — obiettivo Dr-GRPO",
    "14. ablations-objectives-sft-allowed-mass (train+eval) — SFT + massa ammessa",
    "15. ablations-objectives-sft-structured   (train+eval) — SFT + loss strutturata",
]


class CampaignScreen(T2GScreen):
    """Riepilogo della campagna completa in ordine di riuso (binding ``C``).

    Mostra l'ordine di esecuzione con le note sul riuso (SFT adapter +
    baseline cached), poi conferma prima di POST /queue {ablation: true} +
    tick immediato.
    """

    BINDINGS = [Binding("escape", "go_back", "Back")]

    def compose(self) -> ComposeResult:
        yield Header()
        yield Static("Campagna completa — ordine di riuso", classes="title")
        yield Static(
            "15 celle, 27 entry (3 eval-only + 24 train/eval).\n"
            "L'ordine massimizza il riuso: la coda esistente viene SOSTITUITA.",
            classes="hint",
        )
        yield Static(
            "\n".join(f"  [dim]{line}[/dim]" for line in _CAMPAIGN_LINES),
            classes="hint",
        )
        yield Button(
            "Avvia campagna completa (SOSTITUISCE la coda + tick)",
            variant="primary",
            id="submit",
        )
        yield Footer()

    def action_go_back(self) -> None:
        self.t2g_app.switch_screen("dashboard")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "submit":
            self.t2g_app.push_screen(
                ConfirmScreen(
                    "Avviare la CAMPAGNA COMPLETA?\n"
                    "15 celle in ordine di riuso (27 entry).\n"
                    "La coda esistente viene SOSTITUITA.\n"
                    "Il primo job parte subito (tick immediato)."
                ),
                self._confirmed,
            )

    def _confirmed(self, ok: bool | None) -> None:
        if ok:
            self.t2g_app.run_worker(self.t2g_app.run_campaign())


class ReplaceQueueScreen(T2GScreen):
    """Rimpiazza l'intera coda: ablation completa o lista custom.

    Due modalità (entrambe con conferma, avvisano che la coda esistente viene
    SOSTITUITA): ``Ablation completa`` (15 config → 27 entry, stesso ordine di
    ``run_all.sh``) oppure coda custom, una ``tipo:config[:tag]`` per riga.
    """

    BINDINGS = [Binding("escape", "go_back", "Back")]

    def compose(self) -> ComposeResult:
        yield Header()
        yield Static("Rimpiazza la coda esistente", classes="title")
        yield Static(
            "ATTENZIONE: la coda attuale viene SOSTITUITA dall'operazione.",
            classes="hint",
        )
        yield Button(
            "Ablation completa (15 config → 27 job)", variant="primary", id="ablation"
        )
        yield Static(
            "…oppure definisci una coda custom (una entry per riga):", classes="hint"
        )
        yield Static(
            "Formato [b]tipo:config[:tag][/b] — es. [b]train:sft-grpo-few-shot[/b] "
            "o [b]train:sft-grpo-few-shot:my-run[/b]",
            classes="hint",
        )
        yield TextArea(
            "train:sft-grpo-few-shot\n# le righe che iniziano con # sono ignorate\neval:grpo-few-shot",
            id="custom",
        )
        yield Button("Invia coda custom", variant="error", id="submit")
        yield Footer()

    # ── Azioni (binding) ──

    def action_go_back(self) -> None:
        self.t2g_app.switch_screen("dashboard")

    # ── Eventi widget ──

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "ablation":
            self.t2g_app.push_screen(
                ConfirmScreen(
                    "Avviare l'ABLATION COMPLETA?\n15 config → 27 entry. "
                    "La coda esistente viene SOSTITUITA."
                ),
                self._confirmed_ablation,
            )
        elif event.button.id == "submit":
            self._submit_custom()

    # ── Interno ──

    def _confirmed_ablation(self, ok: bool | None) -> None:
        if ok:
            self.t2g_app.run_worker(self.t2g_app.replace_queue(ablation=True))
            # Torna SUBITO alla dashboard: il worker notifica l'esito, mentre
            # restando sulla CampaignScreen i binding (g/r/a...) resterebbero
            # inutilizzabili finche' il POST /queue non risponde (o per sempre).
            self.t2g_app.switch_screen("dashboard")

    def _submit_custom(self) -> None:
        text = self.query_one("#custom", TextArea).text
        try:
            jobs = self._parse_custom(text)
        except ValueError as exc:
            self.t2g_app.notify(
                f"[red]{escape(str(exc))}[/red]", severity="error", timeout=8
            )
            return
        if not jobs:
            self.t2g_app.notify(
                "[yellow]Nessuna entry valida: scrivi almeno una riga tipo:config[:tag][/yellow]",
                severity="warning",
                timeout=6,
            )
            return
        self.t2g_app.push_screen(
            ConfirmScreen(
                f"Invio {len(jobs)} job custom? La coda esistente viene SOSTITUITA."
            ),
            lambda ok: self._confirmed_custom(bool(ok), jobs),
        )

    def _confirmed_custom(self, ok: bool, jobs: list[dict[str, str]]) -> None:
        if ok:
            self.t2g_app.run_worker(self.t2g_app.replace_queue(jobs=jobs))

    def _parse_custom(self, text: str) -> list[dict[str, str]]:
        """Parsa le righe ``tipo:config[:tag]`` in job per POST /queue.

        I config accettati sono quelli noti al servizio (GET /configs, se
        disponibile) oltre alla copia locale, più i path ``.yaml``.
        """
        known = set(self.t2g_app.available_configs or CONFIG_NAMES) | CONFIG_NAME_SET
        jobs: list[dict[str, str]] = []
        for raw in text.splitlines():
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(":")
            if len(parts) not in (2, 3):
                raise ValueError(
                    f"Formato non valido: {line!r} (atteso tipo:config[:tag])"
                )
            job_type, config = parts[0].strip(), parts[1].strip()
            if job_type not in ("train", "eval"):
                raise ValueError(f"Tipo non valido: {job_type!r} (usare train o eval)")
            if config not in known and not config.endswith(".yaml"):
                raise ValueError(
                    f"Config non valido: {config!r} (nome noto o path .yaml)"
                )
            job: dict[str, str] = {"type": job_type, "config": config}
            if len(parts) == 3 and parts[2].strip():
                job["tag"] = parts[2].strip()
            jobs.append(job)
        return jobs


class ConfirmScreen(Screen[bool]):
    """Dialogo di conferma Sì/No riusabile (dismiss → True/False).

    Nota: ``ConfirmModal`` non esiste più in Textual ≥ 8, quindi è definito
    un dialoghetto minimale con i widget standard.
    """

    BINDINGS = [Binding("escape", "cancel", "Annulla")]

    def __init__(
        self,
        prompt: str,
        *,
        confirm_label: str = "Conferma",
        cancel_label: str = "Annulla",
    ) -> None:
        super().__init__()
        self._prompt = escape(prompt)
        self._confirm_label = confirm_label
        self._cancel_label = cancel_label

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Static(self._prompt, classes="dialog-prompt")
            with Horizontal(classes="dialog-actions"):
                yield Button(self._cancel_label, id="cancel")
                yield Button(self._confirm_label, id="confirm", variant="error")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "confirm":
            self.dismiss(True)
        elif event.button.id == "cancel":
            self.dismiss(False)

    def action_cancel(self) -> None:
        self.dismiss(False)


class ConfigScreen(T2GScreen):
    """Prima configurazione: nessun URL trovato (env o .env).

    L'URL è obbligatorio; il token è OPZIONALE (il servizio locale può girare
    senza auth — il campo restà vuoto). I valori vengono salvati nel .env
    locale (sezione marcata). Il token viene inserito in un Input mascherato
    e non è mai loggato.
    """

    def compose(self) -> ComposeResult:
        yield Header()
        yield Static("Configura il servizio", classes="title")
        yield Static(
            "Nessun T2G_SERVICE_URL trovato in env o .env.\n"
            "I valori verranno salvati nel file .env locale (sezione marcata).\n"
            "Il token è opzionale per il servizio locale (http://127.0.0.1:8000).",
            classes="hint",
        )
        yield Input(
            placeholder="URL del servizio (es. http://127.0.0.1:8000)",
            id="url",
        )
        yield Input(
            placeholder="X-Auth-Token — opzionale in locale (vuoto = auth off)",
            id="token",
            password=True,
        )
        yield Button("Salva e connetti", variant="primary", id="save")
        yield Footer()

    def on_mount(self) -> None:
        self.query_one("#url", Input).focus()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id != "save":
            return
        url = self.query_one("#url", Input).value.strip()
        token = self.query_one("#token", Input).value.strip()
        if not url:
            self.t2g_app.notify(
                "[red]L'URL è obbligatorio[/red]", severity="error", timeout=6
            )
            return
        if not url.startswith(("http://", "https://")):
            self.t2g_app.notify(
                "[red]URL non valido: deve iniziare con http(s)://[/red]",
                severity="error",
                timeout=6,
            )
            return
        path = save_env_config(url, token)
        self.t2g_app.config = T2GConfig(url=url.rstrip("/"), token=token)
        self.t2g_app.client = RemoteServiceClient(
            self.t2g_app.config.url, self.t2g_app.config.token
        )
        self.t2g_app.notify(
            f"[green]Configurazione salvata in {escape(str(path))}[/green]",
            severity="information",
            timeout=6,
        )
        self.t2g_app.switch_screen("dashboard")


class T2GDashApp(App[None]):
    """App Textual: monitor live + coda + form per pilotare il driver remoto."""

    TITLE = "T2G Cluster Driver"
    CSS_PATH = "tui.tcss"
    SCREENS = {
        "dashboard": DashboardScreen,
        "queue": QueueScreen,
        "add_job": AddJobScreen,
        "start_job": lambda: AddJobScreen(start_mode=True),
        "batch_start": BatchStartScreen,
        "campaign": CampaignScreen,
        "replace": ReplaceQueueScreen,
        "biglog": LogScreen,
        "results": ResultsScreen,
        "config": ConfigScreen,
    }
    BINDINGS = [Binding("q", "quit", "Quit")]

    def __init__(
        self,
        config: T2GConfig | None,
        client: RemoteServiceClient | None = None,
        refresh_interval: float = 10.0,
    ) -> None:
        super().__init__()
        self.config = config
        self.client = client or (
            RemoteServiceClient(config.url, config.token) if config else None
        )
        self.refresh_interval = refresh_interval
        self.status: dict[str, Any] | None = None
        self.monitor_snapshot: dict[str, Any] | None = None
        self.jobs: list[dict[str, Any]] = []
        self._refreshing = False
        # Config noti al servizio (GET /configs) con fallback alla copia
        # locale: usati dai form AddJob/BatchStart/Results.
        self.available_configs: list[str] = list(CONFIG_NAMES)
        # Serie per-step loss/reward (GET /timeseries) per le sparkline.
        # `timeseries_available` diventa False al primo 404: il pannello
        # degrada e non si riprova per tutta la sessione.
        self.timeseries: dict[str, dict[str, Any]] = {}
        self.timeseries_available: bool = True

    def on_mount(self) -> None:
        if self.config is None:
            self.push_screen("config")
        else:
            self.push_screen("dashboard")
            self.run_worker(self._load_configs())

    # ── Letture ──

    async def _load_configs(self) -> None:
        """GET /configs: config noti dal servizio; fallback alla copia locale.

        Silenzioso su errore (404 o endpoint assente): la copia locale di
        CONFIG_NAMES è già sufficiente per i form. La risposta può essere
        una lista o un dict con chiave ``configs``/``names``.
        """
        if self.client is None:
            return
        try:
            payload = await asyncio.to_thread(self.client.get_configs)
        except RemoteServiceError:
            return
        if isinstance(payload, dict):
            payload = payload.get("configs") or payload.get("names") or []
        if isinstance(payload, list):
            # Il servizio restituisce dict ``{"name": ..., "path": ...}``; una
            # versione precedente stringhe nude: accettiamo entrambe. I nomi
            # finiscono negli id dei widget (``Checkbox(id=f"cfg-{name}")``):
            # graffe o apici sollevano BadIdentifier e rompono la schermata
            # batch — senza alcun errore visibile qui.
            names = [
                (
                    str(item["name"])
                    if isinstance(item, dict) and "name" in item
                    else str(item)
                )
                for item in payload
            ]
            names = [name for name in names if name]
            if names:
                self.available_configs = names

    async def refresh_status(self) -> None:
        """Ricarica GET /status in un thread separato (UI mai bloccata)."""
        if self._refreshing or self.client is None:
            return
        self._refreshing = True
        try:
            try:
                status = await asyncio.to_thread(self.client.get_status)
            except RemoteServiceError as exc:
                self.notify(
                    f"[red]{escape(str(exc))}[/red]", severity="error", timeout=8
                )
            else:
                self._set_status(status)
        except Exception as exc:  # rete di sicurezza: mai crashare la UI
            self.notify(
                f"[red]Errore inatteso: {exc.__class__.__name__}[/red]",
                severity="error",
                timeout=8,
            )
        finally:
            self._refreshing = False

    async def refresh_monitor(self) -> None:
        """Ricarica GET /monitor (+ serie loss/reward se /timeseries esiste)."""
        if self._refreshing or self.client is None:
            return
        self._refreshing = True
        try:
            try:
                snapshot = await asyncio.to_thread(self.client.get_monitor)
            except RemoteServiceError as exc:
                self.notify(
                    f"[red]{escape(str(exc))}[/red]", severity="error", timeout=8
                )
            else:
                self.monitor_snapshot = snapshot
                self.status = snapshot  # campi status condivisi
                await self._refresh_timeseries(snapshot)
                screen = self.screen
                if isinstance(screen, DashboardScreen):
                    screen.refresh_view()
        except Exception as exc:  # rete di sicurezza: mai crashare la UI
            self.notify(
                f"[red]Errore inatteso: {exc.__class__.__name__}[/red]",
                severity="error",
                timeout=8,
            )
        finally:
            self._refreshing = False

    async def _refresh_timeseries(self, snapshot: dict[str, Any]) -> None:
        """Serie per-step loss/reward (GET /timeseries) — degrada in silenzio.

        L'endpoint può non esistere (404): lo si disabilita per la sessione
        e la dashboard mostra il placeholder. Su errore di rete si tengono
        le serie dell'ultimo poll riuscito. Mai un'eccezione verso il
        chiamante: i grafici sono un'aggiunta, non un requisito.
        """
        if self.client is None or not self.timeseries_available:
            return
        tag = _active_tag(snapshot)
        if not tag:
            return
        for metric in ("loss", "reward"):
            try:
                payload = await asyncio.to_thread(
                    self.client.get_timeseries, tag, metric, _TIMESERIES_LIMIT
                )
            except ApiError as exc:
                if exc.status_code == 404:
                    self.timeseries_available = False
                    self.timeseries = {}
                return
            except RemoteServiceError:
                return  # transitorio: si riprova al prossimo poll
            if isinstance(payload, dict):
                self.timeseries[metric] = payload

    async def refresh_jobs(self) -> None:
        """Ricarica GET /jobs e aggiorna la schermata Queue se attiva."""
        if self.client is None:
            return
        try:
            jobs = await asyncio.to_thread(self.client.get_jobs)
        except RemoteServiceError as exc:
            self.notify(f"[red]{escape(str(exc))}[/red]", severity="error", timeout=8)
            return
        self.jobs = jobs
        screen = self.screen
        if isinstance(screen, QueueScreen):
            screen.reload()

    # ── Mutazioni ──

    async def start_batch_job(
        self, job_type: str, config: str, tag: str | None
    ) -> None:
        """POST /jobs/batch per un singolo config: train+eval insieme.

        Usata dal form AddJobScreen (start_mode, train + checkbox eval): il
        train parte subito (tick) e la sua eval resta in coda.
        """
        jobs: list[dict[str, Any]] = [{"type": job_type, "config": config}]
        if tag:
            jobs[0]["tag"] = tag
        if job_type == "train":
            eval_job: dict[str, Any] = {"type": "eval", "config": config}
            if tag:
                eval_job["tag"] = tag
            jobs.append(eval_job)
        await self._run_batch(jobs, start_now=True)

    async def start_batch(
        self, jobs: list[dict[str, Any]], start_now: bool = True
    ) -> None:
        """POST /jobs/batch: enqueue multipli (+ tick se start_now)."""
        await self._run_batch(jobs, start_now=start_now)

    async def _run_batch(self, jobs: list[dict[str, Any]], start_now: bool) -> None:
        if self.client is None:
            return
        try:
            result = await asyncio.to_thread(self.client.start_batch, jobs, start_now)
        except RemoteServiceError as exc:
            self.notify(f"[red]{escape(str(exc))}[/red]", severity="error", timeout=10)
            return
        started_now = bool(result.get("started_now"))
        n = len(jobs)
        if started_now:
            self.notify(
                f"[green]Batch: {n} job accodati, il primo è PARTITO[/green]",
                severity="information",
                timeout=6,
            )
        else:
            self.notify(
                f"[yellow]Batch: {n} job accodati (job attivo — partiranno coi "
                "tick)[/yellow]",
                severity="warning",
                timeout=6,
            )
        self.monitor_snapshot = result
        self.status = result
        self.switch_screen("dashboard")
        await self.refresh_monitor()

    async def start_job(self, job_type: str, config: str, tag: str | None) -> None:
        """POST /jobs/start: accoda + tick immediato (parte subito se libero)."""
        if self.client is None:
            return
        try:
            result = await asyncio.to_thread(
                self.client.start_job, job_type, config, tag
            )
        except RemoteServiceError as exc:
            self.notify(f"[red]{escape(str(exc))}[/red]", severity="error", timeout=10)
            return
        started_now = bool(result.get("started_now"))
        if started_now:
            self.notify(
                f"[green]Job AVVIATO: {escape(job_type)} {escape(config)}[/green]",
                severity="information",
                timeout=6,
            )
        else:
            self.notify(
                f"[yellow]Aggiunto alla coda (job attivo): {escape(job_type)} "
                f"{escape(config)}[/yellow]",
                severity="warning",
                timeout=6,
            )
        self.monitor_snapshot = result
        self.status = result
        self.switch_screen("dashboard")
        await self.refresh_monitor()

    def confirm_kill(self) -> None:
        """Chiede conferma (dialogo rosso) e poi esegue POST /kill."""
        self.push_screen(
            ConfirmScreen(
                "TERMINARE il job attivo?\n\n"
                "Il job verrà cancellato con scancel. La coda CONTINUA col "
                "prossimo job al prossimo tick.\n"
                "Per fermare TUTTO: premi Annulla e usa pause (p)."
            ),
            self._confirmed_kill,
        )

    def _confirmed_kill(self, ok: bool | None) -> None:
        if ok:
            self.run_worker(self.kill_active())

    async def kill_active(self) -> None:
        """POST /kill: scancel del job attivo (409 → toast giallo)."""
        if self.client is None:
            return
        try:
            await asyncio.to_thread(self.client.kill_active)
        except ApiError as exc:
            if exc.status_code == 409:
                self.notify(
                    "[yellow]Nessun job attivo da terminare[/yellow]",
                    severity="warning",
                    timeout=6,
                )
            else:
                self.notify(
                    f"[red]{escape(str(exc))}[/red]", severity="error", timeout=10
                )
            return
        except RemoteServiceError as exc:
            self.notify(f"[red]{escape(str(exc))}[/red]", severity="error", timeout=10)
            return
        self.notify(
            "[green]Job terminato (scancel) — la coda continua al prossimo tick[/green]",
            severity="information",
            timeout=8,
        )
        await self.refresh_monitor()

    async def add_job(self, job_type: str, config: str, tag: str | None) -> bool:
        """POST /jobs: accoda un job; True se riuscito."""
        if self.client is None:
            return False
        try:
            result = await asyncio.to_thread(self.client.add_job, job_type, config, tag)
        except RemoteServiceError as exc:
            self.notify(f"[red]{escape(str(exc))}[/red]", severity="error", timeout=8)
            return False
        self._set_status(result.get("status"))
        added = str(result.get("added", ""))
        self.notify(
            f"[green]Job accodato: {escape(added)}[/green]",
            severity="information",
            timeout=6,
        )
        return True

    async def delete_job(self, tag: str) -> None:
        """DELETE /jobs/{tag}: rimuove dalla coda tutti i job col tag dato."""
        if self.client is None:
            return
        try:
            result = await asyncio.to_thread(self.client.delete_job, tag)
        except RemoteServiceError as exc:
            self.notify(f"[red]{escape(str(exc))}[/red]", severity="error", timeout=8)
            return
        self._set_status(result.get("status"))
        removed = int(result.get("removed", 0) or 0)
        if removed:
            self.notify(
                f"[green]Rimossi {removed} job con tag '{escape(tag)}'[/green]",
                severity="information",
                timeout=5,
            )
        else:
            self.notify(
                f"[yellow]Nessun job con tag '{escape(tag)}' in coda[/yellow]",
                severity="warning",
                timeout=5,
            )
        await self.refresh_jobs()

    async def run_campaign(self) -> None:
        """POST /queue {ablation: true} + tick immediato (Campagna ``C``).

        Sostituisce l'intera coda con la campagna in ordine di riuso
        (app.py ABLATION_MODELS) e fa subito un tick così il primo job
        parte senza attendere l'hook/server.
        """
        if self.client is None:
            return
        try:
            result = await asyncio.to_thread(self.client.replace_queue, None, True)
        except RemoteServiceError as exc:
            self.notify(f"[red]{escape(str(exc))}[/red]", severity="error", timeout=10)
            return
        self._set_status(result.get("status"))
        count = int(result.get("count", 0) or 0)
        self.notify(
            f"[green]Campagna avviata: {count} job in coda[/green]",
            severity="information",
            timeout=6,
        )
        # Tick immediato: il primo job parte ora (se QoS libera)
        try:
            await asyncio.to_thread(self.client.tick)
            self.notify(
                "[green]Tick eseguito — primo job partito/in partenza[/green]",
                severity="information",
                timeout=6,
            )
        except RemoteServiceError:
            pass  # il tick fallirà se QoS occupata: la catena avanza dopo
        await self.refresh_jobs()
        self.switch_screen("dashboard")

    async def replace_queue(
        self,
        *,
        jobs: list[dict[str, str]] | None = None,
        ablation: bool = False,
    ) -> None:
        """POST /queue: rimpiazza l'intera coda (custom o ablation)."""
        if self.client is None:
            return
        try:
            result = await asyncio.to_thread(self.client.replace_queue, jobs, ablation)
        except RemoteServiceError as exc:
            self.notify(f"[red]{escape(str(exc))}[/red]", severity="error", timeout=10)
            return
        self._set_status(result.get("status"))
        count = int(result.get("count", 0) or 0)
        self.notify(
            f"[green]Coda rimpiazzata: {count} job in coda[/green]",
            severity="information",
            timeout=6,
        )
        self.switch_screen("dashboard")

    async def pause(self) -> None:
        """POST /pause: soft-stop (chain_stopped creato sul cluster)."""
        await self._simple_action("pause", "Catena in pausa (chain_stopped creato)")

    async def resume(self) -> None:
        """POST /resume: rimuove chain_stopped e fa un tick immediato."""
        await self._simple_action(
            "resume", "Catena ripresa (chain_stopped rimosso + tick)"
        )

    async def tick(self) -> None:
        """POST /tick: tick manuale (ssh sul cluster — può durare secondi)."""
        if self.client is None:
            return
        self.set_busy(True)
        result: dict[str, Any] | None = None
        try:
            try:
                result = await asyncio.to_thread(self.client.tick)
            except RemoteServiceError as exc:
                self.notify(
                    f"[red]{escape(str(exc))}[/red]", severity="error", timeout=12
                )
        finally:
            self.set_busy(False)
        if result is not None:
            self._set_status(result)
            self.notify(
                "[green]Tick manuale eseguito[/green]",
                severity="information",
                timeout=4,
            )

    # ── Interno ──

    async def _simple_action(self, action: str, ok_message: str) -> None:
        """Esegue pause/resume e aggiorna lo stato dalla risposta."""
        if self.client is None:
            return
        try:
            result = await asyncio.to_thread(getattr(self.client, action))
        except RemoteServiceError as exc:
            self.notify(f"[red]{escape(str(exc))}[/red]", severity="error", timeout=8)
            return
        if isinstance(result, dict):
            self._set_status(result.get("status"))
        self.notify(
            f"[green]{escape(ok_message)}[/green]", severity="information", timeout=4
        )

    def _set_status(self, status: Any) -> None:
        """Aggiorna ``self.status`` e ridisegna la dashboard se visibile."""
        if isinstance(status, dict):
            self.status = status
            screen = self.screen
            if isinstance(screen, DashboardScreen):
                screen.refresh_view()

    def set_busy(self, busy: bool) -> None:
        """Mostra/nasconde lo spinner sulla dashboard (operazioni lente)."""
        screen = self.screen
        if isinstance(screen, DashboardScreen):
            screen.set_busy(busy)


# ── Entry point ───────────────────────────────────────────────────────────────


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Argomenti CLI: --url/--token (override di env/.env)."""
    parser = argparse.ArgumentParser(
        prog="remote/tui.py",
        description="Client TUI per il driver T2G (orchestrazione cluster su Render).",
    )
    parser.add_argument("--url", help="URL del servizio (override di env/.env)")
    parser.add_argument("--token", help="X-Auth-Token (override di env/.env)")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Avvia la TUI con la configurazione risolta (CLI > env > .env)."""
    args = parse_args(argv)
    config = resolve_config(cli_url=args.url, cli_token=args.token)
    T2GDashApp(config=config).run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
