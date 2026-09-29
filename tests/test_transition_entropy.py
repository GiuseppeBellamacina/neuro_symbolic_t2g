"""Entropia condizionale delle transizioni fra gloss.

Il numero che regge l'argomento della relazione e' il GUADAGNO: quanto il gloss
precedente riduce l'incertezza gia' condizionata sul token sorgente. I test
fissano le proprieta' che rendono quel numero interpretabile.
"""

from __future__ import annotations

import math

import pytest

from src.analysis.transition_entropy import aligned_pairs, estimate


def _rows(pairs):
    return [{"text": t, "gloss": g} for t, g in pairs]


def test_aligned_pairs_drops_misaligned_rows():
    """Una riga con lunghezze diverse non ha corrispondenza posizionale."""
    rows = _rows([("a b", "A B"), ("a b c", "A B"), ("x", "X")])
    assert [s for s, _ in aligned_pairs(rows)] == [["a", "b"], ["x"]]


def test_deterministic_mapping_has_zero_entropy_and_zero_gain():
    """Se il gloss e' funzione del solo token sorgente, non resta incertezza
    e il gloss precedente non puo' aggiungere nulla."""
    rows = _rows([("a b", "A B"), ("b a", "B A"), ("a a", "A A")])
    r = estimate(rows, rows)
    assert r.h_given_source == pytest.approx(0.0, abs=1e-12)
    assert r.gain == pytest.approx(0.0, abs=1e-12)


def test_previous_gloss_that_disambiguates_yields_a_positive_gain():
    """Sorgente ambigua (``x`` -> ``P`` o ``Q``) ma risolta dal gloss
    precedente. Le entropie sono medie su TUTTE le posizioni: qui una su due e'
    deterministica, quindi 1 bit di ambiguita' su ``x`` fa 0,5 bit di media, e
    il gloss precedente lo azzera del tutto."""
    rows = _rows([("a x", "A P"), ("b x", "B Q")] * 8)
    r = estimate(rows, rows)
    assert r.h_given_source == pytest.approx(0.5, abs=1e-9)
    assert r.h_given_source_and_prev == pytest.approx(0.0, abs=1e-9)
    assert r.gain == pytest.approx(0.5, abs=1e-9)
    assert r.gain_fraction == pytest.approx(1.0, abs=1e-9)


def test_gain_is_never_negative_under_backoff():
    """Il condizionamento aggiuntivo fa backoff sulla stima meno informata,
    quindi non puo' peggiorarla: e' cio' che rende il divario leggibile."""
    train = _rows([("a x", "A P"), ("b x", "B Q"), ("c y", "C R")] * 5)
    held_out = _rows([("a x", "A P"), ("c y", "C R"), ("d z", "D S")])
    r = estimate(train, held_out)
    assert r.gain >= -1e-12


def test_unseen_gloss_does_not_produce_infinite_entropy():
    """Un gloss mai visto nel train non deve mandare la stima a infinito."""
    train = _rows([("a b", "A B")] * 4)
    held_out = _rows([("q r", "Q R")])
    r = estimate(train, held_out)
    assert math.isfinite(r.h_given_source)
    assert math.isfinite(r.h_given_source_and_prev)


def test_empty_training_counts_are_refused():
    """Nessuna coppia allineata: meglio fallire che restituire zero bit."""
    with pytest.raises(ValueError):
        estimate(_rows([("a b c", "A B")]), _rows([("a", "A")]))
