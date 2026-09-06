"""Client TUI locale per il driver T2G (remote/app.py).

Monitor live e controllo di coda e job del cluster.

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
    Static,
    TextArea,
)

# Config noti al driver (stessi nomi di remote/app.py:CONFIG_MAP).

CONFIG_NAMES: tuple[str, ...] = (
    "sft",
    "grpo-zero",
    "grpo-few",
    "sft-grpo-zero",
    "sft-grpo-few",
    "baseline-zero",
    "baseline-few",
    "sft-grpo-zero-pda",
    "sft-grpo-zero-hot",
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

_CSS = """
Screen {
    background: $surface;
}

.panel {
    height: auto;
    border: round $primary;
    padding: 0 1;
    margin: 0 1 1 1;
}

.hint {
    height: auto;
    padding: 0 1;
    color: $text-muted;
}

.title {
    height: auto;
    padding: 0 1;
    text-style: bold;
}

#banner {
    display: none;
    height: auto;
    background: $warning;
    color: $text;
    text-style: bold;
    padding: 0 1;
    margin: 0 1 1 1;
}

#loading {
    display: none;
    height: 3;
}

#job-panel {
    height: auto;
    border: round $primary;
    padding: 0 1;
    margin: 0 1 1 1;
}

#job-panel #job-progress {
    height: 1;
    margin: 0;
}

#samples-panel {
    height: 1fr;
    border: round $primary;
    padding: 0 1;
    margin: 0 1 1 1;
}

#tail-panel {
    height: 1fr;
    border: round $panel;
    padding: 0 1;
    margin: 0 1 1 1;
}

#samples-panel RichLog, #tail-panel RichLog {
    height: 1fr;
}

#queue-panel {
    height: auto;
    border: round $panel;
    padding: 0 1;
    margin: 0 1 1 1;
}

#big-log {
    height: 1fr;
    border: round $primary;
    padding: 0 1;
    margin: 0 1 1 1;
}

#log-path {
    height: auto;
    padding: 0 1;
    color: $text-muted;
}

#status {
    height: auto;
    border: round $primary;
    padding: 0 1;
    margin: 0 1 1 1;
}

#errors {
    height: auto;
    border: round $error;
    padding: 0 1;
    margin: 0 1 1 1;
}

#events {
    height: auto;
    border: round $panel;
    padding: 0 1;
    margin: 0 1 1 1;
}

DataTable {
    height: 1fr;
    margin: 0 1;
}

Select, Input, TextArea, Button {
    margin: 0 1 0 1;
}

/* Dialogo di conferma Sì/No */
ConfirmScreen {
    align: center middle;
}

ConfirmScreen > Vertical {
    width: 76;
    height: auto;
    border: round $error;
    background: $surface;
    padding: 1 2;
}

ConfirmScreen .dialog-prompt {
    padding-bottom: 1;
}

ConfirmScreen .dialog-actions {
    height: auto;
    align-horizontal: right;
}

ConfirmScreen .dialog-actions Button {
    width: auto;
    margin-left: 1;
}
"""


class T2GScreen(Screen[None]):
    """Base delle schermate: espone l'App tipizzata con i metodi custom."""

    @property
    def t2g_app(self) -> "T2GDashApp":
        """Riferimento all'app (Textual tipizza ``app`` come generico)."""
        return self.app  # type: ignore[return-value]


class DashboardScreen(T2GScreen):
    """Monitor principale (ex-dashboard): stato + metriche live del job attivo.

    Auto-refresh ogni ``refresh_interval`` secondi (GET /monitor); refresh
    manuale con ``r``. Pannelli: banner raggiungibilità, job attivo (con
    ProgressBar step/total e metriche loss/reward/lr + sezioni SFT/eval),
    completion samples, log tail, coda/errori/eventi. A cluster
    irraggiungibile il servizio risponde con la cache dell'ultimo tick e
    ``cluster_reachable: false`` → banner giallo.

    Binding: r refresh · g queue · a add · s job singolo · S batch · k kill ·
    w replace · p pause · R resume · t tick · L log fullscreen.
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
    ]

    def compose(self) -> ComposeResult:
        yield Header()
        yield Static(id="banner")
        yield LoadingIndicator(id="loading")
        with Vertical(id="job-panel"):
            yield Static(id="job-summary")
            yield ProgressBar(total=100, show_eta=False, id="job-progress")
        yield Static("", classes="panel-title", id="samples-title")
        with Vertical(id="samples-panel"):
            yield RichLog(id="samples-log", highlight=False, markup=True)
        with Vertical(id="tail-panel"):
            yield RichLog(id="tail-log", highlight=False, markup=True)
        yield Static(id="queue-panel")
        yield Footer()

    def on_mount(self) -> None:
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
        snap = (
            self.t2g_app.monitor_snapshot
            if self.t2g_app.monitor_snapshot is not None
            else self.t2g_app.status
        )
        job_box = self.query_one("#job-summary", Static)
        progress = self.query_one("#job-progress", ProgressBar)
        if not isinstance(snap, dict):
            job_box.update("Caricamento dal servizio…")
            progress.display = False
            return

        banner = self.query_one("#banner", Static)
        samples_log = self.query_one("#samples-log", RichLog)
        tail_log = self.query_one("#tail-log", RichLog)
        queue_box = self.query_one("#queue-panel", Static)

        reachable = bool(snap.get("cluster_reachable", False))
        self.t2g_app.sub_title = self.t2g_app.config.url if self.t2g_app.config else ""
        banner.display = not reachable
        banner.update(
            ""
            if reachable
            else "[yellow]⚠ CLUSTER IRRAGGIUNGIBILE — mostrato l'ultimo stato "
            "noto dalla cache del servizio[/yellow]"
        )

        job_box.update(self._job_text(snap))
        detail = snap.get("job_detail")
        step = detail.get("step") if isinstance(detail, dict) else None
        total = detail.get("total_steps") if isinstance(detail, dict) else None
        try:
            progress_value = max(0.0, min(float(str(step)), float(str(total))))
            progress_total = float(str(total))
        except (TypeError, ValueError):
            progress.display = False
        else:
            progress.display = progress_total > 0
            if progress_total > 0:
                progress.update(total=progress_total, progress=progress_value)
        samples_log.clear()
        samples_value = snap.get("samples")
        samples = samples_value if isinstance(samples_value, list) else []
        for line in samples[-8:]:
            samples_log.write(Text.from_markup(escape(str(line))))
        if not samples:
            samples_log.write("[dim]— nessun sample disponibile —[/dim]")
        tail_log.clear()
        tail_value = snap.get("log_tail")
        log_tail = tail_value if isinstance(tail_value, list) else []
        for line in log_tail:
            tail_log.write(Text.from_markup(f"[dim]{escape(str(line))}[/dim]"))
        if not log_tail:
            tail_log.write("[dim]— log vuoto —[/dim]")
        queue_box.update(
            "\n\n".join(
                part
                for part in [
                    self._queue_text(snap),
                    self._errors_text(snap),
                    self._events_text(snap),
                ]
                if part
            )
        )

    def _job_text(self, snap: dict[str, Any]) -> str:
        active = snap.get("active_job")
        detail = snap.get("job_detail")
        detail = detail if isinstance(detail, dict) else None
        stopped = bool(snap.get("stopped"))
        last_tick = escape(str(snap.get("last_tick_at") or "mai"))
        reach = (
            "[green]ok[/green]" if snap.get("cluster_reachable") else "[red]giù[/red]"
        )
        stop_txt = "[red]PAUSA[/red]" if stopped else "[green]attivo[/green]"

        header = f"cluster {reach} · catena {stop_txt} · tick {last_tick}"
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

    def _queue_text(self, snap: dict[str, Any]) -> str:
        queue_value = snap.get("queue")
        queue = queue_value if isinstance(queue_value, list) else []
        if not queue:
            return "Coda: [bold]vuota[/bold]"
        rows = []
        for entry in queue[:5]:
            parts = str(entry).split(":")
            tag = parts[2] if len(parts) > 2 else "?"
            jtype = parts[0] if parts else "?"
            rows.append(f"  [dim]{jtype:>5} · {escape(tag)}[/dim]")
        more = f"\n  [dim]… e altri {len(queue) - 5}[/dim]" if len(queue) > 5 else ""
        return (
            f"Coda: [bold]{len(queue)}[/bold] job · [b]g[/b] lista completa\n"
            + "\n".join(rows)
            + more
        )

    def _errors_text(self, snap: dict[str, Any]) -> str:
        errors_value = snap.get("errors_recent")
        errors = errors_value[-3:] if isinstance(errors_value, list) else []
        if not errors:
            return ""
        rows = "\n".join(f"[red]✖ {escape(str(e))[:100]}[/red]" for e in errors)
        return rows

    def _events_text(self, snap: dict[str, Any]) -> str:
        events_value = snap.get("events")
        events = events_value[-5:] if isinstance(events_value, list) else []
        if not events:
            return ""
        rows = []
        for event in events:
            if not isinstance(event, dict):
                continue
            # Full date+time (was time-only: events across days were
            # indistinguishable). ts is ISO "YYYY-MM-DDTHH:MM:SS".
            ts = str(event.get("ts", ""))[:19].replace("T", " ")
            etype = str(event.get("type", ""))
            detail = escape(str(event.get("detail", "")))[:80]
            rows.append(f"[dim]{ts} {etype:<16} {detail}[/dim]")
        return "\n".join(rows)


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
        yield Header()
        yield Static(title, classes="title")
        yield Select(
            [("train", "train"), ("dual prompt eval", "eval")],
            prompt="Tipo",
            value="train",
            id="type",
        )
        yield Select(
            [(name, name) for name in CONFIG_NAMES],
            prompt="Config",
            value=CONFIG_NAMES[0],
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
            for name in CONFIG_NAMES:
                yield Checkbox(name, id=f"cfg-{name}")
        yield Static("Per ogni config selezionato accoda:", classes="hint")
        yield Select(
            [
                ("train + dual prompt eval", "train+eval"),
                ("solo train", "train"),
                ("solo dual prompt eval", "eval"),
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
        for name in CONFIG_NAMES:
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

    Aperta con ``L`` dal monitor; ``Esc`` torna al monitor.
    """

    BINDINGS = [
        Binding("r", "refresh", "Refresh"),
        Binding("escape", "go_back", "Back"),
    ]

    def compose(self) -> ComposeResult:
        yield Header()
        yield Static(id="log-path", classes="hint")
        with Vertical(id="big-log"):
            yield RichLog(id="big-richtext", highlight=False, markup=True)
        yield Footer()

    def on_mount(self) -> None:
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
        path_box.update(f"Log: {escape(str(result.get('log_path') or '?'))}")
        rich.clear()
        for line in result.get("lines") or []:
            rich.write(escape(line))
        if not result.get("lines"):
            rich.write("[dim]— log vuoto o nessun job attivo —[/dim]")


# Campaign summary; order matches remote.app.DEFAULT_CAMPAIGN.

_CAMPAIGN_LINES: list[str] = [
    "1. baseline-zero          (eval-only)",
    "2. baseline-few           (eval-only)",
    "3. sft                    (train + dual prompt eval)",
    "4. grpo-zero              (train + dual prompt eval)",
    "5. grpo-few               (train + dual prompt eval)",
    "6. sft-grpo-zero          (train + dual prompt eval, reuse SFT)",
    "7. sft-grpo-few           (train + dual prompt eval, reuse SFT)",
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
            "7 celle, 12 entry (2 eval-only + 5 train/dual-prompt-eval).\n"
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
                    "7 celle in ordine di riuso (12 entry).\n"
                    "La coda esistente viene SOSTITUITA.\n"
                    "Il primo job parte subito (tick immediato)."
                ),
                self._confirmed,
            )

    def _confirmed(self, ok: bool | None) -> None:
        if ok:
            self.t2g_app.run_worker(self.t2g_app.run_campaign())
            # Return before the potentially slow queue rewrite/tick. The
            # worker keeps running and app-level notifications remain visible.
            self.t2g_app.switch_screen("dashboard")


class ReplaceQueueScreen(T2GScreen):
    """Rimpiazza l'intera coda: ablation completa o lista custom.

    Due modalità (entrambe con conferma, avvisano che la coda esistente viene
    SOSTITUITA): ``Ablation completa`` (7 config → 12 entry, stesso ordine di
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
            "Campagna default (7 config → 12 job)", variant="primary", id="ablation"
        )
        yield Static(
            "…oppure definisci una coda custom (una entry per riga):", classes="hint"
        )
        yield Static(
            "Formato [b]tipo:config[:tag][/b] — es. [b]train:sft-grpo-zero[/b] "
            "o [b]train:sft-grpo-zero:my-run[/b]",
            classes="hint",
        )
        yield TextArea(
            "train:sft-grpo-zero\n# le righe che iniziano con # sono ignorate\neval:baseline-zero",
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
                    "Avviare la CAMPAGNA DEFAULT?\n7 config → 12 entry. "
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
            # Torna SUBITO alla dashboard: il worker continua in background
            # e notifica l'esito. Restare sulla CampaignScreen lascerebbe
            # inutilizzabili i binding del dashboard (g/r/a...) finché il
            # POST /queue non risponde (o per sempre, se fallisce).
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

    @staticmethod
    def _parse_custom(text: str) -> list[dict[str, str]]:
        """Parsa le righe ``tipo:config[:tag]`` in job per POST /queue."""
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
            if config not in CONFIG_NAME_SET and not config.endswith(".yaml"):
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
    CSS = _CSS
    SCREENS = {
        "dashboard": DashboardScreen,
        "queue": QueueScreen,
        "add_job": AddJobScreen,
        "start_job": lambda: AddJobScreen(start_mode=True),
        "batch_start": BatchStartScreen,
        "campaign": CampaignScreen,
        "replace": ReplaceQueueScreen,
        "biglog": LogScreen,
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

    def on_mount(self) -> None:
        if self.config is None:
            self.push_screen("config")
        else:
            self.push_screen("dashboard")

    # ── Letture ──

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
        """Ricarica GET /monitor (stato + job_detail + samples + log tail)."""
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
                snapshot = snapshot if isinstance(snapshot, dict) else {}
                self.monitor_snapshot = snapshot
                self.status = snapshot  # campi status condivisi
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
        (app.py DEFAULT_CAMPAIGN) e fa subito un tick così il primo job
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
