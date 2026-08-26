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
    "queue": ["train:experiments/configs/t2g/grpo_optimal.yaml:run1"],
    "last_job": "12345:train:experiments/configs/t2g/grpo_optimal.yaml:run1:0",
    "stopped": False,
    "watcher_alive": True,
    "errors_recent": [],
    "last_tick_at": "2026-08-26T10:00:00",
    "cluster_reachable": True,
    "events": [{"ts": "2026-08-26T09:59:00", "type": "tick", "detail": "tick eseguito"}],
}

JOBS_BODY = [
    {
        "entry": "train:experiments/configs/t2g/grpo_optimal.yaml:run1",
        "type": "train",
        "config": "experiments/configs/t2g/grpo_optimal.yaml",
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
    if request.method == "GET" and path == "/jobs":
        return httpx.Response(200, json=JOBS_BODY)
    if request.method == "POST" and path == "/jobs":
        return httpx.Response(
            201,
            json={"added": "train:experiments/configs/t2g/sft.yaml:run1", "status": STATUS_BODY},
        )
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
    assert status["active_job"] == {"id": "12345", "name": "train-grpo", "state": "RUNNING"}
    assert status["queue"] == ["train:experiments/configs/t2g/grpo_optimal.yaml:run1"]
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
    client.add_job("train", "grpo_optimal", tag="run1")
    request = recorder.requests[-1]
    assert request.method == "POST"
    assert request.url.path == "/jobs"
    assert request.headers["X-Auth-Token"] == "test-token"
    assert json.loads(request.content) == {"type": "train", "config": "grpo_optimal", "tag": "run1"}


def test_add_job_omits_optional_fields():
    client, recorder = _client()
    client.add_job("eval", "zero_shot")
    assert json.loads(recorder.requests[-1].content) == {"type": "eval", "config": "zero_shot"}


def test_add_job_with_mode():
    client, recorder = _client()
    client.add_job("train", "grpo_qwen05", tag="x", mode="--resume")
    body = json.loads(recorder.requests[-1].content)
    assert body == {"type": "train", "config": "grpo_qwen05", "tag": "x", "mode": "--resume"}


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
        {"type": "train", "config": "grpo_optimal"},
        {"type": "eval", "config": "zero_shot", "tag": "zs"},
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
    for call in (client.get_status, client.get_jobs, client.pause, client.resume, client.tick):
        call()
    assert len(recorder.requests) == 5
    assert all(request.headers["X-Auth-Token"] == "test-token" for request in recorder.requests)


def test_401_raises_auth_error():
    def handler(request):
        return httpx.Response(401, json={"detail": "X-Auth-Token mancante o non valido"})

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
        return httpx.Response(502, json={"detail": "Cluster irraggiungibile: ssh timeout"})

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


def test_screens_are_registered():
    app = _make_app()
    assert {"dashboard", "queue", "add_job", "replace", "config"} <= set(app.SCREENS)


def test_app_mounts_dashboard_and_shows_status():
    """Smoke: la dashboard monta e mostra lo stato dopo un refresh."""

    async def _run() -> None:
        app = _make_app()
        async with app.run_test() as pilot:
            await pilot.pause()
            assert isinstance(app.screen, tui.DashboardScreen)
            await app.refresh_status()
            for _ in range(20):
                if app.status is not None:
                    break
                await pilot.pause()
            assert app.status is not None
            assert app.status["cluster_reachable"] is True
            status_box = app.screen.query_one("#status", tui.Static)
            assert "cluster raggiungibile" in status_box.render().plain

    asyncio.run(_run())


def test_dashboard_shows_unreachable_banner():
    """A cluster irraggiungibile la dashboard mostra il banner giallo con la
    cache del servizio (cluster_reachable: false)."""

    async def _run() -> None:
        body = dict(STATUS_BODY)
        body["cluster_reachable"] = False

        def handler(request):
            return httpx.Response(200, json=body)

        client, _ = _client(handler=handler)
        app = tui.T2GDashApp(
            config=tui.T2GConfig(url="https://t2g.example.com", token="test-token"),
            client=client,
        )
        async with app.run_test() as pilot:
            await pilot.pause()
            await app.refresh_status()
            for _ in range(20):
                if app.status is not None:
                    break
                await pilot.pause()
            assert app.status is not None
            assert app.status["cluster_reachable"] is False
            banner = app.screen.query_one("#banner", tui.Static)
            assert banner.display is True
            assert "IRRAGGIUNGIBILE" in banner.render().plain

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
