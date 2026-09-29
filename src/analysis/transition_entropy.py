"""Quanto informa il gloss precedente, dato che si conosce gia' il token sorgente?

Perche' esiste
--------------
E' la qualificazione a priori dell'obiettivo strutturato: prima di spendere GPU
su un termine CRF sulle transizioni fra gloss, conviene misurare quanta
informazione quelle transizioni portano davvero. La domanda si formula in modo
esatto con l'entropia condizionale:

    H(gloss | src)         incertezza residua dopo aver visto il token sorgente
    H(gloss | src, prev)   la stessa, sapendo anche il gloss precedente

La differenza e' il limite superiore a cio' che un termine di transizione puo'
aggiungere. Su ASLG-PC12 vale circa 0,046 bit, cioe' qualche punto percentuale
dell'incertezza residua: non zero, ma piccolo. L'esperimento (si veda il
capitolo sui risultati) ha poi confermato la previsione, e l'obiettivo
strutturato e' stato abbandonato perche' non supera il proprio controllo
negativo.

Cosa e' robusto e cosa no
-------------------------
Il **guadagno** e' la quantita' affidabile: e' una differenza fra due stime che
condividono lo stesso backoff, quindi gran parte dell'errore di stima si
cancella. I **livelli assoluti** dipendono dalla politica di smoothing scelta
per i contesti non osservati e non vanno citati come costanti del corpus.

Protocollo
----------
Le statistiche sono stimate SOLO dal train; la valutazione e' sul test. Si
usano solo le coppie allineate in lunghezza, dove la corrispondenza posizionale
fra parola sorgente e gloss e' non ambigua (stessa convenzione di
``src/analysis/rule_baseline.py``). I contesti non osservati fanno backoff a
tre livelli: (prev, src) -> src -> unigramma.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass

__all__ = ["TransitionEntropy", "aligned_pairs", "estimate"]

_START = "<s>"


@dataclass(frozen=True)
class TransitionEntropy:
    """Le due entropie condizionali e il loro divario, in bit."""

    h_given_source: float
    h_given_source_and_prev: float
    tokens_evaluated: int

    @property
    def gain(self) -> float:
        """Bit rimossi dal gloss precedente. La quantita' di interesse."""
        return self.h_given_source - self.h_given_source_and_prev

    @property
    def gain_fraction(self) -> float:
        """Frazione dell'incertezza residua che il gloss precedente rimuove."""
        if self.h_given_source <= 0:
            return 0.0
        return self.gain / self.h_given_source


def aligned_pairs(
    rows: Iterable[Mapping[str, str]],
) -> Iterator[tuple[list[str], list[str]]]:
    """Coppie (sorgente, gloss) con lunghezze uguali, tokenizzate su spazi."""
    for row in rows:
        source = str(row["text"]).split()
        gloss = str(row["gloss"]).split()
        if len(source) == len(gloss):
            yield source, gloss


def estimate(
    train_rows: Iterable[Mapping[str, str]],
    eval_rows: Iterable[Mapping[str, str]],
) -> TransitionEntropy:
    """Stima le due entropie: conteggi da ``train_rows``, media su ``eval_rows``.

    ``train_rows`` devono essere righe di TRAIN: stimare i conteggi sulle righe
    di valutazione renderebbe entrambe le entropie artificialmente basse e il
    divario privo di significato.
    """
    joint: dict[tuple[str, str], Counter] = defaultdict(Counter)
    by_source: dict[str, Counter] = defaultdict(Counter)
    marginal: Counter = Counter()

    for source, gloss in aligned_pairs(train_rows):
        previous = _START
        for word, target in zip(source, gloss):
            key = word.lower()
            joint[(previous, key)][target] += 1
            by_source[key][target] += 1
            marginal[target] += 1
            previous = target

    total = sum(marginal.values())
    if total == 0:
        raise ValueError("nessuna coppia allineata nel train: conteggi vuoti")

    def unigram_logp(target: str) -> float:
        # Mezzo conteggio per i gloss mai visti: evita -inf senza spostare
        # sensibilmente i gloss osservati.
        return math.log2(max(marginal.get(target, 0), 0.5) / total)

    def logp(counts: Counter, target: str, fallback) -> float:
        observed = counts.get(target, 0)
        if observed == 0:
            return fallback()
        return math.log2(observed / sum(counts.values()))

    h_source = 0.0
    h_both = 0.0
    evaluated = 0
    empty: Counter = Counter()

    for source, gloss in aligned_pairs(eval_rows):
        previous = _START
        for word, target in zip(source, gloss):
            key = word.lower()
            source_counts = by_source.get(key, empty)
            h_source -= logp(source_counts, target, lambda: unigram_logp(target))
            h_both -= logp(
                joint.get((previous, key), empty),
                target,
                lambda: logp(source_counts, target, lambda: unigram_logp(target)),
            )
            evaluated += 1
            previous = target

    if evaluated == 0:
        raise ValueError("nessuna coppia allineata nelle righe di valutazione")
    return TransitionEntropy(h_source / evaluated, h_both / evaluated, evaluated)


def main() -> None:
    """Stima da train, valuta su test, stampa i valori citati nella relazione."""
    from src.datasets.aslg_dataset import download_aslg_dataset

    dataset = download_aslg_dataset(seed=42)
    result = estimate(dataset["train"], dataset["test"])

    print(f"token di test valutati:   {result.tokens_evaluated}")
    print(f"H(gloss | src):           {result.h_given_source:.4f} bit")
    print(f"H(gloss | src, prev):     {result.h_given_source_and_prev:.4f} bit")
    print(
        f"guadagno:                 {result.gain:+.4f} bit "
        f"({100 * result.gain_fraction:.2f}% dell'incertezza residua)"
    )


if __name__ == "__main__":
    main()
