"""Client TUI locale per il driver T2G su Render (remote/app.py).

Pilota il micro-servizio di orchestrazione del cluster da terminale (Windows
pwsh incluso): dashboard con stato, coda job, form per accodare e per
rimpiazzare l'intera coda, pause/resume/tick manuale.

Avvio:

    uv run --extra tui python remote/tui.py [--url URL] [--token TOKEN]

Configurazione (in ordine di precedenza): flag CLI → env vars
``T2G_SERVICE_URL`` / ``T2G_AUTH_TOKEN`` → file ``.env`` (cwd o repo root).
Se mancano, l'app parte sulla schermata di configurazione che salva i valori
in ``.env`` (sezione marcata ``# >>> t2g-tui >>>``). Il token non viene mai
loggato né stampato.

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
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.screen import Screen
from textual.widgets import (
    Button,
    DataTable,
    Footer,
    Header,
    Input,
    LoadingIndicator,
    Select,
    Static,
    TextArea,
)

# ── Config noti al driver (stessi 12 nomi di remote/app.py:CONFIG_MAP) ───────

CONFIG_NAMES: tuple[str, ...] = (
    "grpo_optimal",
    "grpo_qwen05",
    "sft",
    "grpo_experimental_all",
    "zero_shot",
    "zero_shot_grammar",
    "grpo_no_grammar",
    "grpo_no_sft",
    "grpo_pda",
    "grpo_pda_lookahead",
    "grpo_soft_viterbi",
    "grpo_verifier_scaled",
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
    return {key: value for key, value in dotenv_values(str(path)).items() if value is not None}


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
    marker_idx = next((i for i, ln in enumerate(lines) if ln.strip() == ENV_MARKER), None)
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

    Ritorna None se anche solo uno dei due manca (l'app mostrerà la schermata
    di configurazione).
    """
    paths = list(dotenv_paths) if dotenv_paths is not None else env_file_candidates()
    file_values: dict[str, str] = {}
    for path in paths:
        file_values.update(read_env_file(path))
    url = cli_url or os.environ.get("T2G_SERVICE_URL") or file_values.get("T2G_SERVICE_URL", "")
    token = cli_token or os.environ.get("T2G_AUTH_TOKEN") or file_values.get("T2G_AUTH_TOKEN", "")
    if url and token:
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
        """httpx.Client condiviso, creato lazy (supporta MockTransport nei test)."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.Client(
                base_url=self.base_url,
                headers={"X-Auth-Token": self._token},
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
        payload: dict[str, Any] = {"ablation": ablation} if jobs is None else {"jobs": jobs}
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
            raise ConnectionError(f"Errore di trasporto HTTP: {exc.__class__.__name__}") from exc
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
    """Dashboard principale: stato del cluster, errori ed eventi recenti.

    Auto-refresh ogni ``refresh_interval`` secondi; refresh manuale con ``r``.
    A cluster irraggiungibile il servizio risponde comunque con la cache
    dell'ultimo tick e ``cluster_reachable: false``: la dashboard la mostra
    con un banner giallo.
    """

    BINDINGS = [
        Binding("r", "refresh", "Refresh"),
        Binding("g", "queue", "Queue"),
        Binding("a", "add_job", "Add job"),
        Binding("w", "replace_queue", "Replace queue"),
        Binding("p", "pause", "Pause"),
        Binding("R", "resume", "Resume"),
        Binding("t", "tick", "Tick"),
    ]

    def compose(self) -> ComposeResult:
        yield Header()
        yield Static(id="banner")
        yield LoadingIndicator(id="loading")
        yield Static(id="status", classes="panel")
        yield Static(id="errors", classes="panel")
        yield Static(id="events", classes="panel")
        yield Footer()

    def on_mount(self) -> None:
        self.set_interval(self.t2g_app.refresh_interval, self.t2g_app.refresh_status)
        self.refresh_view()
        self.t2g_app.run_worker(self.t2g_app.refresh_status())

    # ── Azioni (binding) ──

    def action_refresh(self) -> None:
        self.t2g_app.run_worker(self.t2g_app.refresh_status())

    def action_queue(self) -> None:
        self.t2g_app.switch_screen("queue")

    def action_add_job(self) -> None:
        self.t2g_app.switch_screen("add_job")

    def action_replace_queue(self) -> None:
        self.t2g_app.switch_screen("replace")

    def action_pause(self) -> None:
        self.t2g_app.run_worker(self.t2g_app.pause())

    def action_resume(self) -> None:
        self.t2g_app.run_worker(self.t2g_app.resume())

    def action_tick(self) -> None:
        self.t2g_app.run_worker(self.t2g_app.tick())

    # ── Rendering ──

    def set_busy(self, busy: bool) -> None:
        """Mostra/nasconde lo spinner (usato dalle operazioni lente)."""
        self.query_one("#loading", LoadingIndicator).display = busy

    def refresh_view(self) -> None:
        """Ridisegna i pannelli a partire da ``app.status``."""
        if not self.is_mounted:
            return
        banner = self.query_one("#banner", Static)
        status_box = self.query_one("#status", Static)
        errors_box = self.query_one("#errors", Static)
        events_box = self.query_one("#events", Static)
        status = self.t2g_app.status
        if status is None:
            banner.display = False
            status_box.update("Caricamento stato dal servizio…")
            errors_box.update("")
            events_box.update("")
            return
        reachable = bool(status.get("cluster_reachable", False))
        self.t2g_app.sub_title = self.t2g_app.config.url if self.t2g_app.config else ""
        banner.display = not reachable
        if reachable:
            banner.update("")
        else:
            banner.update(
                "[yellow]⚠ CLUSTER IRRAGGIUNGIBILE — mostrato l'ultimo stato "
                "noto dalla cache del servizio[/yellow]"
            )
        status_box.update(self._status_text(status, reachable))
        errors_box.update(self._errors_text(status))
        events_box.update(self._events_text(status))

    def _status_text(self, status: dict[str, Any], reachable: bool) -> str:
        active = status.get("active_job")
        queue_n = len(status.get("queue") or [])
        stopped = bool(status.get("stopped"))
        watcher = bool(status.get("watcher_alive"))
        last_tick = escape(str(status.get("last_tick_at") or "mai"))
        last_job = escape(str(status.get("last_job") or "—"))[:80]

        reach_txt = "[green]cluster raggiungibile[/green]" if reachable else "[red]cluster IRRAGGIUNGIBILE[/red]"
        stop_txt = "[red]in PAUSA[/red]" if stopped else "[green]attivo[/green]"
        watch_txt = "[green]vivo[/green]" if watcher else "[yellow]spento[/yellow]"
        if isinstance(active, dict) and active:
            active_txt = (
                f"[bold]{escape(str(active.get('name') or '?'))}[/bold] "
                f"[dim](id {escape(str(active.get('id') or '?'))} · "
                f"{escape(str(active.get('state') or '?'))})[/dim]"
            )
        else:
            active_txt = "[dim]nessuno[/dim]"

        return "\n".join(
            [
                f"{reach_txt} · catena {stop_txt} · watcher {watch_txt}",
                f"Ultimo tick: {last_tick} · Ultimo job: {last_job}",
                f"Job attivo: {active_txt}",
                f"Job in coda: [bold]{queue_n}[/bold]  (apri la lista con [b]g[/b])",
            ]
        )

    def _errors_text(self, status: dict[str, Any]) -> str:
        errors = (status.get("errors_recent") or [])[-5:]
        if not errors:
            return ""
        rows = "\n".join(f"[red]✖ {escape(str(e))[:100]}[/red]" for e in errors)
        return f"[bold]Errori recenti ({len(errors)})[/bold]\n{rows}"

    def _events_text(self, status: dict[str, Any]) -> str:
        events = (status.get("events") or [])[-8:]
        if not events:
            return ""
        color_of = {
            "error": "red",
            "enqueue": "green",
            "dequeue": "green",
            "queue_replace": "green",
            "pause": "yellow",
            "resume": "yellow",
            "tick": "dim",
        }
        rows: list[str] = []
        for event in events:
            ts = str(event.get("ts", ""))[11:19]
            etype = str(event.get("type", ""))
            detail = escape(str(event.get("detail", "")))[:80]
            color = color_of.get(etype)
            if color:
                rows.append(f"[{color}]{ts} {etype:<16}[/{color}] {detail}")
            else:
                rows.append(f"{ts} {etype:<16} {detail}")
        return "[bold]Eventi recenti[/bold]\n" + "\n".join(rows)


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
        """Ricompone la tabella da ``app.jobs``."""
        if not self.is_mounted:
            return
        table = self.query_one("#queue", DataTable)
        table.clear()
        for pos, job in enumerate(self.t2g_app.jobs, start=1):
            table.add_row(
                str(pos),
                str(job.get("type", "")),
                Path(str(job.get("config", ""))).name,
                str(job.get("tag", "")),
                key=str(job.get("tag", "")),
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
            self.t2g_app.notify("[yellow]Coda vuota[/yellow]", severity="warning", timeout=4)
            return
        row_index = table.cursor_row
        if row_index is None:
            self.t2g_app.notify("[yellow]Nessuna riga selezionata[/yellow]", severity="warning", timeout=4)
            return
        row = table.get_row_at(row_index)
        tag = str(row[3]) if row else ""
        if not tag:
            self.t2g_app.notify("[yellow]Riga senza tag[/yellow]", severity="warning", timeout=4)
            return
        self.t2g_app.push_screen(
            ConfirmScreen(f"Rimuovere TUTTI i job con tag '{tag}' dalla coda?"),
            lambda ok: self._confirmed_delete(bool(ok), tag),
        )

    def _confirmed_delete(self, ok: bool, tag: str) -> None:
        if ok:
            self.t2g_app.run_worker(self.t2g_app.delete_job(tag))


class AddJobScreen(T2GScreen):
    """Form per accodare un singolo job (POST /jobs) e tornare alla dashboard."""

    BINDINGS = [Binding("escape", "go_back", "Back")]

    def compose(self) -> ComposeResult:
        yield Header()
        yield Static("Aggiungi un job alla coda", classes="title")
        yield Select(
            [("train", "train"), ("eval", "eval")],
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
        yield Input(placeholder="Tag (opzionale — di default derivato dal config)", id="tag")
        yield Button("Accoda", variant="primary", id="submit")
        yield Footer()

    def on_mount(self) -> None:
        self._last_derived_tag: str | None = None
        self._prefill_tag()
        self.query_one("#type", Select).focus()

    # ── Azioni (binding) ──

    def action_go_back(self) -> None:
        self.t2g_app.switch_screen("dashboard")

    # ── Eventi widget ──

    def on_select_changed(self, event: Select.Changed) -> None:
        if event.select.id == "config":
            self._prefill_tag()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "submit":
            self._submit()

    # ── Interno ──

    def _config_value(self) -> str:
        return str(self.query_one("#config", Select).value)

    def _prefill_tag(self) -> None:
        """Suggerisce il tag derivato dal config (stessa regola del driver)."""
        config = self._config_value()
        derived = config.replace("_", "-")
        tag = self.query_one("#tag", Input)
        if not tag.value or tag.value == self._last_derived_tag:
            tag.value = derived
            self._last_derived_tag = derived

    def _submit(self) -> None:
        job_type = str(self.query_one("#type", Select).value)
        config = self._config_value()
        tag = self.query_one("#tag", Input).value.strip() or None
        self.t2g_app.run_worker(self._do_submit(job_type, config, tag))

    async def _do_submit(self, job_type: str, config: str, tag: str | None) -> None:
        if await self.t2g_app.add_job(job_type, config, tag):
            self.t2g_app.switch_screen("dashboard")


class ReplaceQueueScreen(T2GScreen):
    """Rimpiazza l'intera coda: ablation completa o lista custom.

    Due modalità (entrambe con conferma, avvisano che la coda esistente viene
    SOSTITUITA): ``Ablation completa`` (12 config → 22 entry, stesso ordine di
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
        yield Button("Ablation completa (12 config → 22 job)", variant="primary", id="ablation")
        yield Static("…oppure definisci una coda custom (una entry per riga):", classes="hint")
        yield Static(
            "Formato [b]tipo:config[:tag][/b] — es. [b]train:grpo_optimal[/b] "
            "o [b]train:grpo_optimal:my-run[/b]",
            classes="hint",
        )
        yield TextArea(
            "train:grpo_optimal\n# le righe che iniziano con # sono ignorate\neval:zero_shot",
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
                    "Avviare l'ABLATION COMPLETA?\n12 config → 22 entry. "
                    "La coda esistente viene SOSTITUITA."
                ),
                self._confirmed_ablation,
            )
        elif event.button.id == "submit":
            self._submit_custom()

    # ── Interno ──

    def _confirmed_ablation(self, ok: bool) -> None:
        if ok:
            self.t2g_app.run_worker(self.t2g_app.replace_queue(ablation=True))

    def _submit_custom(self) -> None:
        text = self.query_one("#custom", TextArea).text
        try:
            jobs = self._parse_custom(text)
        except ValueError as exc:
            self.t2g_app.notify(f"[red]{escape(str(exc))}[/red]", severity="error", timeout=8)
            return
        if not jobs:
            self.t2g_app.notify(
                "[yellow]Nessuna entry valida: scrivi almeno una riga tipo:config[:tag][/yellow]",
                severity="warning",
                timeout=6,
            )
            return
        self.t2g_app.push_screen(
            ConfirmScreen(f"Invio {len(jobs)} job custom? La coda esistente viene SOSTITUITA."),
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
                raise ValueError(f"Formato non valido: {line!r} (atteso tipo:config[:tag])")
            job_type, config = parts[0].strip(), parts[1].strip()
            if job_type not in ("train", "eval"):
                raise ValueError(f"Tipo non valido: {job_type!r} (usare train o eval)")
            if config not in CONFIG_NAME_SET and not config.endswith(".yaml"):
                raise ValueError(f"Config non valido: {config!r} (nome noto o path .yaml)")
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
    """Prima configurazione: nessun URL/token trovato (env o .env).

    Salva i valori nel .env locale (sezione marcata) e crea il client.
    Il token viene inserito in un Input mascherato e non è mai loggato.
    """

    def compose(self) -> ComposeResult:
        yield Header()
        yield Static("Configura il servizio remoto", classes="title")
        yield Static(
            "Nessun T2G_SERVICE_URL / T2G_AUTH_TOKEN trovato in env o .env.\n"
            "I valori verranno salvati nel file .env locale (sezione marcata).",
            classes="hint",
        )
        yield Input(
            placeholder="URL del servizio (es. https://t2g-cluster-driver.onrender.com)",
            id="url",
        )
        yield Input(
            placeholder="X-Auth-Token (header di autenticazione)",
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
        if not url or not token:
            self.t2g_app.notify("[red]URL e token sono obbligatori[/red]", severity="error", timeout=6)
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
        self.t2g_app.client = RemoteServiceClient(self.t2g_app.config.url, self.t2g_app.config.token)
        self.t2g_app.notify(
            f"[green]Configurazione salvata in {escape(str(path))}[/green]",
            severity="information",
            timeout=6,
        )
        self.t2g_app.switch_screen("dashboard")


class T2GDashApp(App[None]):
    """App Textual: dashboard + coda + form per pilotare il driver remoto."""

    TITLE = "T2G Cluster Driver"
    CSS = _CSS
    SCREENS = {
        "dashboard": DashboardScreen,
        "queue": QueueScreen,
        "add_job": AddJobScreen,
        "replace": ReplaceQueueScreen,
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
                self.notify(f"[red]{escape(str(exc))}[/red]", severity="error", timeout=8)
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
        self.notify(f"[green]Job accodato: {escape(added)}[/green]", severity="information", timeout=6)
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
        await self._simple_action("resume", "Catena ripresa (chain_stopped rimosso + tick)")

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
                self.notify(f"[red]{escape(str(exc))}[/red]", severity="error", timeout=12)
        finally:
            self.set_busy(False)
        if result is not None:
            self._set_status(result)
            self.notify("[green]Tick manuale eseguito[/green]", severity="information", timeout=4)

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
        self.notify(f"[green]{escape(ok_message)}[/green]", severity="information", timeout=4)

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
