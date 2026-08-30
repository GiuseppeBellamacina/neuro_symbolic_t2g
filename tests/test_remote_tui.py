"""Test del client TUI remoto (remote/tui.py) — nessuna rete reale.

Il client è testato con ``httpx.MockTransport`` (handler in memoria); la TUI
con ``App.run_test()`` di Textual (headless). Le chiamate di rete non esistono
in questi test: MockTransport intercetta tutto.
"""

from __future__ import annotations

import asyncio
import json

import httpx
import pytest

pytest.importorskip("textual")
pytest.importorskip("httpx")

from remote import tui

# ── Corpi di risposta campione (formato esatto di remote/app.py) ─────────────

STATUS_BODY = {
    "active_job": {"id": "12345", "name": "train-grpo", "state": "RUNNING"},
    "queue": ["train:experiments/configs/t2g/sft-grpo.yaml:run1"],
    "last_job": "12345:train:experiments/configs/t2g/sft-grpo.yaml:run1:0",
    "stopped": False,
    "watcher_alive": True,
    "errors_recent": [],
    "last_tick_at": "2026-08-26T10:00:00",
    "cluster_reachable": True,
    "events": [
        {"ts": "2026-08-26T09:59:00", "type": "tick", "detail": "tick eseguito"}
    ],
}

MONITOR_BODY = {
    **STATUS_BODY,
    "job_detail": {
        "id": "12345",
        "name": "train-grpo",
        "state": "RUNNING",
        "elapsed_human": "01:23",
        "log_path": "logs/slurm-train-12345.log",
        "step": 100,
        "total_steps": 200,
        "loss": "0.5432",
        "reward": "0.3500",
        "lr": "3e-06",
        "sft_active": False,
        "sft_step": None,
        "sft_total": None,
        "sft_loss": None,
        "sft_eval_loss": None,
        "sft_eval_loss_best": None,
        "eval_label": None,
        "eval_progress": None,
        "eval_metrics": {},
    },
    "samples": ["Sample 1 PROMPT: ... OUTPUT: ... GOLD: ..."],
    "log_tail": ["  step=100  loss=0.5432  reward=0.3500", "STEP 7: GRPO Training"],
    "ts": "2026-08-26T10:01:00",
}

LOGS_BODY = {
    "log_path": "logs/slurm-train-12345.log",
    "lines": ["line1", "  step=100  loss=0.5432", "line3"],
}

JOBS_BODY = [
    {
        "entry": "train:experiments/configs/t2g/sft-grpo.yaml:run1",
        "type": "train",
        "config": "experiments/configs/t2g/sft-grpo.yaml",
        "tag": "run1",
        "extra": None,
    }
]


class _Recorder:
    """Adattatore MockTransport: registra le richieste per le asserzioni."""

    def __init__(self, handler) -> None:
        self.requests: list[httpx.Request] = []
        self._handler = handler

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        return self._handler(request)


def _default_handler(request: httpx.Request) -> httpx.Response:
    path = request.url.path
    if request.method == "GET" and path == "/status":
        return httpx.Response(200, json=STATUS_BODY)
    if request.method == "GET" and path == "/monitor":
        return httpx.Response(200, json=MONITOR_BODY)
    if request.method == "GET" and path == "/logs":
        return httpx.Response(200, json=LOGS_BODY)
    if request.method == "GET" and path == "/jobs":
        return httpx.Response(200, json=JOBS_BODY)
    if request.method == "POST" and path == "/jobs":
        return httpx.Response(
            201,
            json={
                "added": "train:experiments/configs/t2g/sft.yaml:run1",
                "status": STATUS_BODY,
            },
        )
    if request.method == "POST" and path == "/jobs/start":
        return httpx.Response(201, json={**MONITOR_BODY, "started_now": True})
    if request.method == "POST" and path == "/jobs/batch":
        return httpx.Response(
            201,
            json={
                **MONITOR_BODY,
                "started_now": True,
                "queued": [
                    "train:experiments/configs/t2g/sft-grpo.yaml:sft-grpo",
                    "eval:experiments/configs/t2g/sft-grpo.yaml:sft-grpo",
                ],
            },
        )
    if request.method == "POST" and path == "/kill":
        return httpx.Response(200, json=MONITOR_BODY)
    if request.method == "POST" and path == "/queue":
        return httpx.Response(
            200,
            json={
                "queue": ["train:experiments/configs/t2g/sft.yaml:x"],
                "count": 1,
                "status": STATUS_BODY,
            },
        )
    if request.method == "DELETE" and path.startswith("/jobs/"):
        return httpx.Response(200, json={"removed": 1, "status": STATUS_BODY})
    if request.method == "POST" and path in ("/pause", "/resume", "/tick"):
        return httpx.Response(200, json=STATUS_BODY)
    return httpx.Response(404, json={"detail": "not found"})


def _client(handler=None, *, url="https://t2g.example.com", token="test-token"):
    """Costruisce un client con MockTransport (nessuna rete)."""
    if handler is None:
        handler = _default_handler
    recorder = _Recorder(handler)
    client = tui.RemoteServiceClient(
        url=url, token=token, transport=httpx.MockTransport(recorder)
    )
    return client, recorder


# ── RemoteServiceClient ───────────────────────────────────────────────────────


def test_client_normalizes_base_url():
    client = tui.RemoteServiceClient("https://t2g.example.com///", "tok")
    assert client.base_url == "https://t2g.example.com"


def test_get_status_parses_fields():
    client, _ = _client()
    status = client.get_status()
    assert status["cluster_reachable"] is True
    assert status["active_job"] == {
        "id": "12345",
        "name": "train-grpo",
        "state": "RUNNING",
    }
    assert status["queue"] == ["train:experiments/configs/t2g/sft-grpo.yaml:run1"]
    assert status["watcher_alive"] is True
    assert status["stopped"] is False
    assert status["events"][0]["type"] == "tick"


def test_get_jobs_parses_entries():
    client, _ = _client()
    jobs = client.get_jobs()
    assert jobs[0]["type"] == "train"
    assert jobs[0]["tag"] == "run1"
    assert jobs[0]["extra"] is None


def test_add_job_payload_and_auth_header():
    client, recorder = _client()
    client.add_job("train", "sft-grpo", tag="run1")
    request = recorder.requests[-1]
    assert request.method == "POST"
    assert request.url.path == "/jobs"
    assert request.headers["X-Auth-Token"] == "test-token"
    assert json.loads(request.content) == {
        "type": "train",
        "config": "sft-grpo",
        "tag": "run1",
    }


def test_add_job_omits_optional_fields():
    client, recorder = _client()
    client.add_job("eval", "sft-only")
    assert json.loads(recorder.requests[-1].content) == {
        "type": "eval",
        "config": "sft-only",
    }


def test_add_job_with_mode():
    client, recorder = _client()
    client.add_job("train", "grpo-only", tag="x", mode="--resume")
    body = json.loads(recorder.requests[-1].content)
    assert body == {
        "type": "train",
        "config": "grpo-only",
        "tag": "x",
        "mode": "--resume",
    }


def test_replace_queue_ablation_payload():
    client, recorder = _client()
    client.replace_queue(ablation=True)
    request = recorder.requests[-1]
    assert request.method == "POST"
    assert request.url.path == "/queue"
    assert json.loads(request.content) == {"ablation": True}


def test_replace_queue_jobs_payload():
    client, recorder = _client()
    jobs = [
        {"type": "train", "config": "sft-grpo"},
        {"type": "eval", "config": "sft-only", "tag": "zs"},
    ]
    client.replace_queue(jobs=jobs)
    assert json.loads(recorder.requests[-1].content) == {"jobs": jobs}


def test_delete_job_url():
    client, recorder = _client()
    client.delete_job("run1")
    request = recorder.requests[-1]
    assert request.method == "DELETE"
    assert request.url.path == "/jobs/run1"


def test_auth_header_on_every_request():
    client, recorder = _client()
    for call in (
        client.get_status,
        client.get_jobs,
        client.pause,
        client.resume,
        client.tick,
    ):
        call()
    assert len(recorder.requests) == 5
    assert all(
        request.headers["X-Auth-Token"] == "test-token" for request in recorder.requests
    )


def test_401_raises_auth_error():
    def handler(request):
        return httpx.Response(
            401, json={"detail": "X-Auth-Token mancante o non valido"}
        )

    client, _ = _client(handler=handler)
    with pytest.raises(tui.AuthError, match="401"):
        client.get_status()


def test_connection_error_is_friendly():
    def handler(request):
        raise httpx.ConnectError("connection refused", request=request)

    client, _ = _client(handler=handler)
    with pytest.raises(tui.ConnectionError, match="connettersi"):
        client.get_status()


def test_timeout_error_is_friendly():
    def handler(request):
        raise httpx.ConnectTimeout("timed out", request=request)

    client, _ = _client(handler=handler)
    with pytest.raises(tui.ConnectionError, match="timeout"):
        client.get_status()


def test_api_error_502_includes_detail():
    def handler(request):
        return httpx.Response(
            502, json={"detail": "Cluster irraggiungibile: ssh timeout"}
        )

    client, _ = _client(handler=handler)
    with pytest.raises(tui.ApiError) as excinfo:
        client.tick()
    assert excinfo.value.status_code == 502
    assert "Cluster irraggiungibile" in excinfo.value.detail


# ── Configurazione (.env con marker) ──────────────────────────────────────────


def test_save_env_config_writes_marker_section(tmp_path, monkeypatch):
    monkeypatch.delenv("T2G_SERVICE_URL", raising=False)
    monkeypatch.delenv("T2G_AUTH_TOKEN", raising=False)
    env = tmp_path / ".env"
    tui.save_env_config("https://t2g.example.com", "tok123", path=env)
    content = env.read_text(encoding="utf-8")
    assert tui.ENV_MARKER in content
    assert "T2G_SERVICE_URL" in content
    assert "T2G_AUTH_TOKEN" in content
    # round-trip: il file scritto viene letto correttamente
    cfg = tui.resolve_config(dotenv_paths=[env])
    assert cfg is not None
    assert cfg.url == "https://t2g.example.com"
    assert cfg.token == "tok123"


def test_save_env_config_updates_existing_section(tmp_path):
    env = tmp_path / ".env"
    env.write_text(
        'OTHER=keep\n# >>> t2g-tui >>>\nT2G_SERVICE_URL="https://old.example.com"\n'
        'T2G_AUTH_TOKEN="old"\n',
        encoding="utf-8",
    )
    tui.save_env_config("https://new.example.com", "newtok", path=env)
    content = env.read_text(encoding="utf-8")
    assert "OTHER=keep" in content  # righe fuori sezione preservate
    assert "old" not in content
    assert "https://new.example.com" in content
    assert "newtok" in content
    assert content.count(tui.ENV_MARKER) == 1  # sezione unica, idempotente


def test_resolve_priority_env_over_file(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text(
        'T2G_SERVICE_URL="https://file.example.com"\nT2G_AUTH_TOKEN="file-token"\n',
        encoding="utf-8",
    )
    monkeypatch.setenv("T2G_SERVICE_URL", "https://env.example.com")
    monkeypatch.setenv("T2G_AUTH_TOKEN", "env-token")
    cfg = tui.resolve_config(dotenv_paths=[env])
    assert cfg is not None
    assert cfg.url == "https://env.example.com"
    assert cfg.token == "env-token"


def test_resolve_priority_cli_over_env(tmp_path, monkeypatch):
    monkeypatch.setenv("T2G_SERVICE_URL", "https://env.example.com")
    monkeypatch.setenv("T2G_AUTH_TOKEN", "env-token")
    cfg = tui.resolve_config(
        cli_url="https://cli.example.com",
        cli_token="cli-token",
        dotenv_paths=[tmp_path / ".env"],  # file assente → nessun valore
    )
    assert cfg is not None
    assert cfg.url == "https://cli.example.com"
    assert cfg.token == "cli-token"


def test_resolve_missing_returns_none(tmp_path, monkeypatch):
    monkeypatch.delenv("T2G_SERVICE_URL", raising=False)
    monkeypatch.delenv("T2G_AUTH_TOKEN", raising=False)
    assert tui.resolve_config(dotenv_paths=[tmp_path / ".env"]) is None


# ── TUI (Textual run_test headless, senza pytest-asyncio) ────────────────────
#
# run_test() è un context manager asincrono: lo guidiamo con asyncio.run()
# dentro test sincroni, così non serve pytest-asyncio nel dev extras.


def _make_app(handler=None):
    client, _ = _client(handler=handler)
    return tui.T2GDashApp(
        config=tui.T2GConfig(url="https://t2g.example.com", token="test-token"),
        client=client,
    )


def _richlog_text(widget) -> str:
    """Testo completo di un RichLog (Textual 8: render() torna un Panel)."""
    return "\n".join(strip.text for strip in widget.lines)


def test_screens_are_registered():
    app = _make_app()
    assert {
        "dashboard",
        "queue",
        "add_job",
        "start_job",
        "replace",
        "biglog",
        "config",
    } <= set(app.SCREENS)


def test_app_mounts_monitor_and_shows_job_detail():
    """Smoke: il monitor monta e mostra job_detail (step/loss/reward)."""

    async def _run() -> None:
        app = _make_app()
        async with app.run_test() as pilot:
            await pilot.pause()
            assert isinstance(app.screen, tui.DashboardScreen)
            await app.refresh_monitor()
            for _ in range(20):
                if app.monitor_snapshot is not None:
                    break
                await pilot.pause()
            assert app.monitor_snapshot is not None
            assert app.monitor_snapshot["job_detail"]["step"] == 100
            job_box = app.screen.query_one("#job-panel", tui.Static)
            text = job_box.render().plain
            assert "train-grpo" in text
            assert "100/200" in text
            assert "0.5432" in text
            assert "0.3500" in text
            # Samples e log tail nei RichLog
            samples_log = app.screen.query_one("#samples-log", tui.RichLog)
            assert "Sample 1" in _richlog_text(samples_log)
            tail_log = app.screen.query_one("#tail-log", tui.RichLog)
            assert "step=100" in _richlog_text(tail_log)

    asyncio.run(_run())


def test_monitor_shows_placeholder_without_active_job():
    """Nessun job attivo → placeholder + prossime entry della coda."""

    async def _run() -> None:
        body = {**MONITOR_BODY, "active_job": None, "job_detail": None}

        def handler(request):
            return httpx.Response(200, json=body)

        client, _ = _client(handler=handler)
        app = tui.T2GDashApp(
            config=tui.T2GConfig(url="https://t2g.example.com", token="test-token"),
            client=client,
        )
        async with app.run_test() as pilot:
            await pilot.pause()
            await app.refresh_monitor()
            for _ in range(20):
                if app.monitor_snapshot is not None:
                    break
                await pilot.pause()
            job_box = app.screen.query_one("#job-panel", tui.Static)
            text = job_box.render().plain
            assert "Nessun job attivo" in text
            assert "sft-grpo" in text  # prossima entry in coda

    asyncio.run(_run())


def test_dashboard_shows_unreachable_banner():
    """A cluster irraggiungibile il monitor mostra il banner giallo con la
    cache del servizio (cluster_reachable: false)."""

    async def _run() -> None:
        body = {**MONITOR_BODY, "cluster_reachable": False}

        def handler(request):
            return httpx.Response(200, json=body)

        client, _ = _client(handler=handler)
        app = tui.T2GDashApp(
            config=tui.T2GConfig(url="https://t2g.example.com", token="test-token"),
            client=client,
        )
        async with app.run_test() as pilot:
            await pilot.pause()
            await app.refresh_monitor()
            for _ in range(20):
                if app.monitor_snapshot is not None:
                    break
                await pilot.pause()
            assert app.monitor_snapshot is not None
            assert app.monitor_snapshot["cluster_reachable"] is False
            banner = app.screen.query_one("#banner", tui.Static)
            assert banner.display is True
            assert "IRRAGGIUNGIBILE" in banner.render().plain

    asyncio.run(_run())


def test_kill_binding_calls_confirm_and_api():
    """`k` → ConfirmScreen → conferma → POST /kill chiamato."""

    async def _run() -> None:
        client, recorder = _client()
        app = tui.T2GDashApp(
            config=tui.T2GConfig(url="https://t2g.example.com", token="test-token"),
            client=client,
        )
        async with app.run_test() as pilot:
            await pilot.pause()
            await pilot.press("k")
            await pilot.pause()
            assert isinstance(app.screen, tui.ConfirmScreen)
            # Conferma (bottone id=confirm)
            confirm_btn = app.screen.query_one("#confirm", tui.Button)
            confirm_btn.press()
            await pilot.pause()
            for _ in range(20):
                if any(r.url.path == "/kill" for r in recorder.requests):
                    break
                await pilot.pause()
            assert any(r.url.path == "/kill" for r in recorder.requests)

    asyncio.run(_run())


def test_start_job_screen_submits_to_start_endpoint():
    """`s` → AddJobScreen(start_mode) → submit (checkbox eval ON di default)
    → POST /jobs/batch con train+eval. Con checkbox OFF → /jobs/start."""

    async def _run() -> None:
        client, recorder = _client()
        app = tui.T2GDashApp(
            config=tui.T2GConfig(url="https://t2g.example.com", token="test-token"),
            client=client,
        )
        async with app.run_test() as pilot:
            await pilot.pause()
            await pilot.press("s")
            await pilot.pause()
            assert isinstance(app.screen, tui.AddJobScreen)
            assert app.screen.start_mode is True
            # Default: "Accoda anche la eval" ON → batch train+eval
            checkbox = app.screen.query_one("#also-eval", tui.Checkbox)
            assert checkbox.value is True
            submit = app.screen.query_one("#submit", tui.Button)
            submit.press()
            await pilot.pause()
            for _ in range(20):
                if any(r.url.path == "/jobs/batch" for r in recorder.requests):
                    break
                await pilot.pause()
            batch_calls = [r for r in recorder.requests if r.url.path == "/jobs/batch"]
            assert batch_calls, "POST /jobs/batch non chiamato"
            payload = json.loads(batch_calls[0].read())
            assert payload["start_now"] is True
            assert [j["type"] for j in payload["jobs"]] == ["train", "eval"]
            assert payload["jobs"][0]["config"] == "sft-grpo"

    asyncio.run(_run())


def test_start_job_screen_without_eval_uses_start_endpoint():
    """Checkbox eval OFF → solo POST /jobs/start (job singolo)."""

    async def _run() -> None:
        client, recorder = _client()
        app = tui.T2GDashApp(
            config=tui.T2GConfig(url="https://t2g.example.com", token="test-token"),
            client=client,
        )
        async with app.run_test() as pilot:
            await pilot.pause()
            await pilot.press("s")
            await pilot.pause()
            checkbox = app.screen.query_one("#also-eval", tui.Checkbox)
            checkbox.value = False  # niente eval accodata
            await pilot.pause()
            submit = app.screen.query_one("#submit", tui.Button)
            submit.press()
            await pilot.pause()
            for _ in range(20):
                if any(r.url.path == "/jobs/start" for r in recorder.requests):
                    break
                await pilot.pause()
            start_calls = [r for r in recorder.requests if r.url.path == "/jobs/start"]
            assert start_calls, "POST /jobs/start non chiamato"
            payload = json.loads(start_calls[0].read())
            assert payload["type"] == "train"
            assert payload["config"] == "sft-grpo"
            # nessun batch in questo flusso
            assert not any(r.url.path == "/jobs/batch" for r in recorder.requests)

    asyncio.run(_run())


def test_batch_start_screen_submits_selected_configs():
    """`S` → BatchStartScreen: 2 checkbox → train+eval → POST /jobs/batch (4 job)."""

    async def _run() -> None:
        client, recorder = _client()
        app = tui.T2GDashApp(
            config=tui.T2GConfig(url="https://t2g.example.com", token="test-token"),
            client=client,
        )
        async with app.run_test() as pilot:
            await pilot.pause()
            await pilot.press("S")
            await pilot.pause()
            assert isinstance(app.screen, tui.BatchStartScreen)
            # Seleziona due config
            app.screen.query_one("#cfg-sft-grpo", tui.Checkbox).value = True
            app.screen.query_one("#cfg-grpo-only", tui.Checkbox).value = True
            await pilot.pause()
            submit = app.screen.query_one("#submit", tui.Button)
            submit.press()
            await pilot.pause()
            await pilot.pause()
            # ConfirmScreen di riepilogo → conferma (query_one come nel test
            # del kill: il dialogo è lo screen attivo)
            assert isinstance(app.screen, tui.ConfirmScreen)
            confirm_btn = app.screen.query_one("#confirm", tui.Button)
            confirm_btn.press()
            await pilot.pause()
            await pilot.pause()
            for _ in range(20):
                if any(r.url.path == "/jobs/batch" for r in recorder.requests):
                    break
                await pilot.pause()
            batch_calls = [r for r in recorder.requests if r.url.path == "/jobs/batch"]
            assert batch_calls, "POST /jobs/batch non chiamato"
            payload = json.loads(batch_calls[0].read())
            assert payload["start_now"] is True
            assert [j["type"] for j in payload["jobs"]] == [
                "train",
                "eval",
                "train",
                "eval",
            ]
            assert [j["config"] for j in payload["jobs"]] == [
                "sft-grpo",
                "sft-grpo",
                "grpo-only",
                "grpo-only",
            ]

    asyncio.run(_run())


def test_monitor_shows_phase_and_queue():
    """Monitor: job_detail con phase → badge fase; coda con 5+ entry visibili."""

    async def _run() -> None:
        body = dict(MONITOR_BODY)
        body["job_detail"] = {
            **MONITOR_BODY["job_detail"],
            "phase": "sft_eval",
            "eval_active": True,
            "source": "live",
        }
        body["queue"] = [
            f"eval:cfg:{tag}" for tag in ("a", "b", "c", "d", "e", "f", "g")
        ]

        def handler(request):
            return httpx.Response(200, json=body)

        client, _ = _client(handler=handler)
        app = tui.T2GDashApp(
            config=tui.T2GConfig(url="https://t2g.example.com", token="test-token"),
            client=client,
        )
        async with app.run_test() as pilot:
            await pilot.pause()
            await app.refresh_monitor()
            for _ in range(20):
                if app.monitor_snapshot is not None:
                    break
                await pilot.pause()
            job_text = app.screen.query_one("#job-panel", tui.Static).render().plain
            assert "SFT" in job_text  # badge fase (sft_eval)
            queue_text = app.screen.query_one("#queue-panel", tui.Static).render().plain
            assert "7 job" in queue_text
            assert "altri 2" in queue_text  # 5 mostrati + 2 extra

    asyncio.run(_run())


def test_log_screen_shows_lines():
    """`L` → LogScreen con le righe di GET /logs."""

    async def _run() -> None:
        client, recorder = _client()
        app = tui.T2GDashApp(
            config=tui.T2GConfig(url="https://t2g.example.com", token="test-token"),
            client=client,
        )
        async with app.run_test() as pilot:
            await pilot.pause()
            await pilot.press("L")
            await pilot.pause()
            assert isinstance(app.screen, tui.LogScreen)
            for _ in range(20):
                if any(r.url.path == "/logs" for r in recorder.requests):
                    break
                await pilot.pause()
            log_widget = app.screen.query_one("#big-richtext", tui.RichLog)
            text = _richlog_text(log_widget)
            assert "step=100" in text
            # Esc torna al monitor
            await pilot.press("escape")
            await pilot.pause()
            assert isinstance(app.screen, tui.DashboardScreen)

    asyncio.run(_run())


def test_app_starts_in_config_screen_without_config():
    """Senza URL/token l'app parte sulla schermata di configurazione."""

    async def _run() -> None:
        client, _ = _client()
        app = tui.T2GDashApp(config=None, client=client)
        async with app.run_test() as pilot:
            await pilot.pause()
            assert isinstance(app.screen, tui.ConfigScreen)

    asyncio.run(_run())


def test_q_binding_quits():
    """Il binding `q` chiude l'app."""

    async def _run() -> None:
        app = _make_app()
        async with app.run_test() as pilot:
            await pilot.pause()
            await pilot.press("q")
            await pilot.pause()
            assert not app.is_running

    asyncio.run(_run())
