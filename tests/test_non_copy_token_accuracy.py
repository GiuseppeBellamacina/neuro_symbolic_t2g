"""non_copy_token_accuracy: metrica primaria copy-insensitive.

Tre garanzie verificate qui:

1. **Equivalenza col comportamento storico**: la funzione è stata spostata da
   ``src/analysis/rule_baseline.py`` a ``src/utils/metrics.py``; i valori
   storici citati nella relazione (rule 0.9424, SFT 0.9660, …) sono stati
   prodotti con la definizione originale, quindi la semantica NON deve
   cambiare. La prova è doppia: (a) l'implementazione originale è congelata
   verbatim qui sotto e confrontata su casi costruiti a mano e su un corpus
   random seedato; (b) ``src.analysis.rule_baseline.non_copy_token_accuracy``
   è lo STESSO oggetto della funzione spostata (re-export, non copia).
2. **Caso degenere definito**: insieme non banale vuoto (gloss identico al
   source maiuscolizzato) → ``(0.0, 0, 0)``. La scelta è documentata nel
   docstring della metrica: accuracy 0.0 ma denominatore 0 rende esplicito al
   chiamante che NON ci sono posizioni valutabili.
3. **Cablaggio nella pipeline di eval**: ``_compute_primary_metrics`` produce
   le chiavi ``non_copy_token_accuracy`` / ``non_copy_token_hits`` /
   ``non_copy_token_total`` nel dict serializzato nel JSON dei risultati, con
   il denominatore coerente col numero di posizioni non banali calcolato a
   mano.
"""

from __future__ import annotations

import random
from collections import Counter

import numpy as np
import pytest

import src.analysis.rule_baseline as rule_baseline_mod
import src.utils.metrics as metrics_mod
from src.training.eval_t2g import _compute_primary_metrics
from src.utils.metrics import non_copy_token_accuracy

# ---------------------------------------------------------------------------
# Implementazione originale congelata (equivalenza)
# ---------------------------------------------------------------------------


def _reference_non_copy_token_accuracy(
    predictions, sources, references
) -> tuple[float, int, int]:
    """Implementazione originale (pre-spostamento), congelata VERBATIM.

    Sorgente: src/analysis/rule_baseline.py prima dello spostamento in
    src/utils/metrics.py. NON modificare: è il riferimento di equivalenza per
    i valori storici pubblicati nella relazione.
    """
    if not (len(predictions) == len(sources) == len(references)):
        raise ValueError(
            f"length mismatch: predictions={len(predictions)}, "
            f"sources={len(sources)}, references={len(references)}"
        )
    hits = 0
    total = 0
    for prediction, source, reference in zip(predictions, sources, references):
        copyable = {word.upper() for word in str(source).split()}
        available = Counter(str(prediction).split())
        for token in str(reference).split():
            if token in copyable:
                continue
            total += 1
            if available[token] > 0:
                available[token] -= 1
                hits += 1
    return (hits / total if total else 0.0), hits, total


# ---------------------------------------------------------------------------
# Equivalenza: funzione spostata vs originale congelata
# ---------------------------------------------------------------------------

# Caso degenere incluso: gloss identico al source maiuscolizzato → insieme non
# banale vuoto. Atteso per entrambe: (0.0, 0, 0) — scelta documentata.
HAND_CASES = [
    # (predictions, sources, references, expected)
    # gloss identico al source maiuscolizzato: nulla da valutare
    (["CAT SAT"], ["cat sat"], ["CAT SAT"], (0.0, 0, 0)),
    # gloss completamente diverso dal source
    (["BE DOG"], ["cat is"], ["BE DOG"], (1.0, 2, 2)),
    # un solo token non banale, colpito
    (["CAT BE"], ["cat is"], ["CAT BE"], (1.0, 1, 1)),
    # un solo token non banale, mancato (echo dell'inglese)
    (["CAT IS"], ["cat is"], ["CAT BE"], (0.0, 0, 1)),
    # echo lowercase: la metrica è case-sensitive, nessun credito
    (["cat be"], ["cat is"], ["CAT BE"], (0.0, 0, 1)),
    # predizione più lunga del reference: i token extra non aiutano né danneggiano
    (["CAT BE DOG"], ["cat is"], ["CAT BE"], (1.0, 1, 1)),
    # predizione più corta del reference: la posizione esiste e fallisce
    (["CAT"], ["cat is"], ["CAT BE"], (0.0, 0, 1)),
    # matching a multinsieme: BE richiesto due volte, prodotto una
    (["BE"], ["is is"], ["BE BE"], (0.5, 1, 2)),
    # aggregazione su più esempi
    (
        ["CAT BE", "DOG IS"],
        ["cat is", "dog is"],
        ["CAT BE", "DOG BE"],
        (0.5, 1, 2),
    ),
]


@pytest.mark.parametrize("case", HAND_CASES, ids=lambda c: f"{c[0]}|{c[2]}")
def test_equivalence_with_frozen_reference_hand_cases(case):
    """Casi costruiti a mano: funzione spostata == originale congelata."""
    predictions, sources, references, expected = case
    got = non_copy_token_accuracy(predictions, sources, references)
    assert got == expected
    assert got == _reference_non_copy_token_accuracy(predictions, sources, references)


def test_equivalence_with_frozen_reference_random_corpus():
    """Corpus random seedato: le due implementazioni coincidono su 300 triple.

    Un test di equivalenza randomizzato copre le interazioni (token misti
    copiabili/non copiabili, ripetizioni, fuori vocabolario) che i casi a mano
    non esauriscono.
    """
    rng = random.Random(42)
    words = [w.upper() for w in ("cat", "is", "be", "dog", "the", "sit", "good")]
    for _ in range(300):
        sources = [
            " ".join(rng.choice(words).lower() for _ in range(rng.randint(1, 6)))
        ]
        references = [
            " ".join(rng.choice(words + ["ZZZ"]) for _ in range(rng.randint(1, 6)))
        ]
        predictions = [
            " ".join(rng.choice(words + ["QQQ"]) for _ in range(rng.randint(0, 6)))
        ]
        assert non_copy_token_accuracy(
            predictions, sources, references
        ) == _reference_non_copy_token_accuracy(predictions, sources, references)


def test_reexport_is_the_moved_function():
    """rule_baseline re-esporta lo STESSO oggetto: una sola definizione viva.

    Garantisce che il path storico ``src.analysis.rule_baseline`` e la
    pipeline di eval usino la STESSA implementazione (nessuna copia che può
    divergere).
    """
    assert (
        rule_baseline_mod.non_copy_token_accuracy is metrics_mod.non_copy_token_accuracy
    )


def test_length_mismatch_rejected():
    """Triplette disallineate rifiutate (fail loud, mai numero falso)."""
    with pytest.raises(ValueError, match="length mismatch"):
        non_copy_token_accuracy(["A"], ["a", "b"], ["A"])


# ---------------------------------------------------------------------------
# Caso degenere: insieme non banale vuoto
# ---------------------------------------------------------------------------


def test_degenerate_empty_non_trivial_set():
    """Insieme non banale vuoto → (0.0, 0, 0), scelta definita e documentata.

    accuracy 0.0 con denominatore 0: il chiamante che legge solo l'accuracy
    vede il valore neutro; chi valuta la metrica deve leggere il denominatore
    (total) per distinguere "nessuna posizione valutabile" da "nessuna
    posizione corretta" — per questo l'eval serializza anche hits/total.
    """
    assert non_copy_token_accuracy(["CAT SAT"], ["cat sat"], ["CAT SAT"]) == (0.0, 0, 0)
    # Anche con predizioni arbitrarie: nessun token di reference è non banale.
    assert non_copy_token_accuracy(["XXX"], ["cat sat"], ["CAT SAT"]) == (0.0, 0, 0)
    # Stringhe vuote: nessun token ovunque, stesso caso degenere.
    assert non_copy_token_accuracy([""], [""], [""]) == (0.0, 0, 0)


# ---------------------------------------------------------------------------
# Cablaggio nella pipeline di eval (_compute_primary_metrics)
# ---------------------------------------------------------------------------

_VOCAB = ["<BOS>", "<EOS>", "<UNK>", "CAT", "BE", "IS", "DOG"]
_TOKEN_TO_IDX = {t: i for i, t in enumerate(_VOCAB)}
_BIGRAM = np.ones((len(_VOCAB), len(_VOCAB)), dtype=np.float32)
_ZERO_REWARD_WEIGHTS = {
    "translation_quality_reward": 0.0,
    "bleu_reward": 0.0,
    "gold_structure_reward": 0.0,
    "verifier_scaled_reward": 0.0,
    "gloss_order_reward": 0.0,
    "gloss_format_reward": 0.0,
    "gloss_repetition_reward": 0.0,
}


def _run_primary_metrics():
    """2 prompt × 2 completions con risultato non-copy noto a mano.

    Ogni reference contiene UN solo token non banale ("BE"; "CAT"/"DOG"/"IS"
    sono copie uppercaseabili del source): 4 completions → denominatore 4.
    Colpiscono "BE" solo le completion 1 e 3 → hits 2 → accuracy 0.5.
    """
    all_completions = [["CAT BE", "CAT IS"], ["DOG BE", "DOG IS"]]
    all_references = ["CAT BE", "DOG BE"]
    all_texts = ["cat is", "dog is"]
    flat_completions = [c for comps in all_completions for c in comps]
    flat_references = [r for r in all_references for _ in range(2)]
    flat_sources = [s for s in all_texts for _ in range(2)]
    metrics, _, _, _, _ = _compute_primary_metrics(
        flat_completions,
        flat_references,
        all_completions,
        all_references,
        token_to_idx=_TOKEN_TO_IDX,
        bigram=_BIGRAM,
        reward_weights=_ZERO_REWARD_WEIGHTS,
        flat_sources=flat_sources,
        n_bootstrap=5,
    )
    return metrics


def test_eval_primary_block_contains_non_copy_keys():
    """La metrica appare nel blocco primario serializzato nel JSON dei risultati."""
    metrics = _run_primary_metrics()
    assert "non_copy_token_accuracy" in metrics
    assert "non_copy_token_hits" in metrics
    assert "non_copy_token_total" in metrics
    # Il valore deve essere JSON-serializzabile così com'è (float/int nativi).
    assert isinstance(metrics["non_copy_token_accuracy"], float)
    assert isinstance(metrics["non_copy_token_hits"], int)
    assert isinstance(metrics["non_copy_token_total"], int)


def test_eval_non_copy_value_and_denominator_coherent():
    """Valore e denominatore coincidono col calcolo fatto a mano (4 posizioni)."""
    metrics = _run_primary_metrics()
    # A mano: 2 reference × 2 completions, un solo token non banale ("BE") per
    # reference → total = 4; colpiscono solo le completion con "BE" → hits = 2.
    assert metrics["non_copy_token_total"] == 4
    assert metrics["non_copy_token_hits"] == 2
    assert metrics["non_copy_token_accuracy"] == pytest.approx(2 / 4)
    # Coerenza interna: hits ≤ total e accuracy = hits / total.
    assert metrics["non_copy_token_hits"] <= metrics["non_copy_token_total"]
    assert (
        metrics["non_copy_token_accuracy"]
        == metrics["non_copy_token_hits"] / metrics["non_copy_token_total"]
    )


def test_eval_non_copy_propagates_to_oracle_shape():
    """Il contratto flat_sources/allineamento: lunghezze diverse → ValueError.

    La funzione di metrica fallisce loud su disallineamenti: l'eval non può
    produrre un numero calcolato su dati sbagliati (nessun fallback silenzioso).
    """
    with pytest.raises(ValueError, match="length mismatch"):
        _compute_primary_metrics(
            ["CAT BE"],
            ["CAT BE"],
            [["CAT BE"]],
            ["CAT BE"],
            token_to_idx=_TOKEN_TO_IDX,
            bigram=_BIGRAM,
            reward_weights=_ZERO_REWARD_WEIGHTS,
            flat_sources=["cat is", "extra-source"],
            n_bootstrap=2,
        )
