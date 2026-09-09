"""Tests for the CampaignScreen (binding ``C``) + PDA config presence."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from remote.tui import CampaignScreen  # noqa: E402
from tests.test_remote_tui import _client, _make_app  # noqa: E402


def test_campaign_screen_registered():
    """La schermata 'campaign' è registrata nell'app."""
    app = _make_app()
    assert "campaign" in app.SCREENS
    assert app.SCREENS["campaign"] is CampaignScreen


def test_campaign_binding_on_dashboard():
    """Il dashboard ha il binding 'C' per la campagna."""
    app = _make_app()
    dash_cls = app.SCREENS["dashboard"]
    if isinstance(dash_cls, type):
        keys = {b.key for b in dash_cls.BINDINGS}
        assert "C" in keys


def test_campaign_screen_shows_summary_and_confirm():
    """'C' → CampaignScreen: mostra le 15 celle e il confirmation flow
    porta a POST /queue {ablation: true} + tick."""
    import remote.tui as tui

    client, recorder = _client()
    app = tui.T2GDashApp(
        config=tui.T2GConfig(url="https://t2g.example.com", token="test-token"),
        client=client,
    )

    async def _run() -> None:
        async with app.run_test() as pilot:
            await pilot.pause()
            await pilot.press("C")
            await pilot.pause()
            assert isinstance(app.screen, CampaignScreen)
            statics = [
                s.render() if hasattr(s, "render") else ""
                for s in app.screen.query("Static")
            ]
            all_text = "\n".join(str(s) for s in statics)
            assert "ablations-decoding-no-grammar" in all_text, "no-grammar nella lista"
            assert (
                "ablations-decoding-hot-rollout" in all_text
            ), "hot-rollout nella lista"
            assert "15 celle" in all_text, "conteggio celle aggiornato"
            assert "baseline-zero-shot" in all_text, "baseline zero-shot nella lista"
            assert "sft-zero-shot" in all_text, "sft nella lista"
            assert "SOSTITUITA" in all_text.upper()
            # submit → ConfirmScreen
            import textual.widgets as tw  # noqa: E402

            submit = app.screen.query_one("#submit", tw.Button)
            submit.press()
            await pilot.pause()
            await pilot.pause()
            # Conferma (ConfirmScreen è il dialogo attivo)
            confirm = app.screen.query_one("#confirm", tw.Button)
            confirm.press()
            await pilot.pause()
            await pilot.pause()
            # attende il worker async (POST /queue + tick)
            for _ in range(20):
                if any(r.url.path == "/queue" for r in recorder.requests):
                    break
                await pilot.pause()
            # dopo la conferma la TUI torna SUBITO alla dashboard: i binding
            # del dashboard (g/r/a...) restano disponibili mentre il worker
            # lavora in background
            # dopo la conferma la TUI torna SUBITO alla dashboard: i binding
            # del dashboard (g/r/a...) restano disponibili mentre il worker
            # lavora in background
            assert not isinstance(
                app.screen, CampaignScreen
            ), "torna alla dashboard dopo conferma"
            calls = [r.url.path for r in recorder.requests]
            assert "/queue" in calls
            assert "/tick" in calls
            # POST /queue body: ablation=True
            q = [r for r in recorder.requests if r.url.path == "/queue"][0]
            body = json.loads(q.read())
            assert body.get("ablation") is True

    import json  # noqa: E402

    asyncio.run(_run())


def test_zero_shot_no_grammar_config_exists():
    """baseline/zero-shot-no-grammar.yaml: lower bound non vincolato (grammar OFF)."""
    from src.utils.config import resolve_config

    cfg = resolve_config(
        "experiments/configs/qwen25-05b/baseline/zero-shot-no-grammar.yaml"
    )
    assert cfg["grammar"]["enabled"] is False
    assert cfg["wandb"]["run_name"] == "qwen25-05b-baseline-zero-shot-no-grammar"
    # eval-only: nessun output_dir (eredita una sezione training parziale da base)
    assert "output_dir" not in cfg["training"]


def test_hotrollout_config_exists():
    """ablations/decoding/hot-rollout.yaml: controllo Finding 1 (T=1.3, riusa SFT).

    Differenza a fattore unico vs sft-grpo/few-shot: solo la rollout temperature.
    La sezione sft_pretrain NON viene toccata (fingerprint SFT invariata
    → riuso dell'adapter di sft/zero-shot).
    """
    from src.utils.config import resolve_config

    base = resolve_config("experiments/configs/qwen25-05b/sft-grpo/few-shot.yaml")
    cfg = resolve_config(
        "experiments/configs/qwen25-05b/ablations/decoding/hot-rollout.yaml"
    )
    assert cfg["grpo"]["temperature"] == 1.3
    assert cfg["grpo"]["num_generations"] == base["grpo"]["num_generations"]
    assert cfg["sft_pretrain"] == base["sft_pretrain"], "fingerprint SFT invariata"
    assert cfg["training"]["output_dir"] == (
        "experiments/checkpoints/qwen25-05b/ablations/decoding/hot-rollout"
    )
