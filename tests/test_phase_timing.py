"""Test del context manager ``phase()`` e di ``format_duration()``.

Il punto essenziale del contratto: la riga di APERTURA esce prima del
lavoro (un messaggio a posteriori non dice nulla mentre il processo e'
fermo) e la riga di chiusura esce anche quando la fase solleva
un'eccezione.
"""

from __future__ import annotations

import re

import pytest

from src.utils.phase_timing import format_duration, phase


class TestFormatDuration:
    def test_secondi(self):
        assert format_duration(0.0) == "0.0s"
        assert format_duration(4.23) == "4.2s"
        assert format_duration(59.4) == "59.4s"

    def test_minuti(self):
        assert format_duration(60) == "1m00s"
        assert format_duration(372) == "6m12s"
        assert format_duration(3599) == "59m59s"

    def test_ore(self):
        assert format_duration(3600) == "1h00m"
        assert format_duration(3720) == "1h02m"
        assert format_duration(7322) == "2h02m"


class TestPhase:
    def test_apertura_prima_del_lavoro_e_chiusura_dopo(self, capsys):
        with phase("Fase di prova", detail="3 elementi"):
            print("LAVORO")
        out = capsys.readouterr().out
        lines = out.splitlines()
        # Ordine: apertura, lavoro, chiusura con la durata.
        assert lines[0] == "  Fase di prova (3 elementi)..."
        assert lines[1] == "LAVORO"
        assert re.match(r"^  Fase di prova: \d+\.\ds$", lines[2])

    def test_senza_detail_nessun_suffisso(self, capsys):
        with phase("Fase nuda"):
            pass
        out = capsys.readouterr().out
        assert "  Fase nuda...\n" in out
        assert re.search(r"^  Fase nuda: \d+\.\ds$", out, re.MULTILINE)

    def test_propaga_eccezione_emettendo_chiusura(self, capsys):
        with pytest.raises(RuntimeError, match="boom"):
            with phase("Fase che fallisce"):
                raise RuntimeError("boom")
        out = capsys.readouterr().out
        # La riga di apertura esce comunque, e il finally emette la chiusura.
        assert "  Fase che fallisce...\n" in out
        assert re.search(r"^  Fase che fallisce: \d+\.\ds$", out, re.MULTILINE)

    def test_threshold_sopprime_la_chiusura(self, capsys):
        with phase("Fase rapida", threshold=float("inf")):
            pass
        out = capsys.readouterr().out
        assert "  Fase rapida...\n" in out
        assert "Fase rapida:" not in out
