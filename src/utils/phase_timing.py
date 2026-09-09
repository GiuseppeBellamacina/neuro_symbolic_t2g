"""Annuncia e cronometra le fasi lunghe di train ed eval.

Motivazione. I job su cluster passano minuti interi senza emettere una riga:
non si distingue un processo che lavora da uno bloccato, e in coda a
un'allocazione singola quel dubbio costa tempo reale. I casi osservati:

- retrieval few-shot su 72.979 prompt (nessuna barra, nessun messaggio);
- costruzione della maschera di vocabolario su 15.472 gloss;
- caricamento del modello quantizzato a 4 bit;
- calcolo delle metriche con bootstrap in eval;
- salvataggio dei checkpoint e generazione delle figure.

Lo stile replica quello gia' presente nei punti di ingresso (``print`` con
indentazione a due spazi sotto i banner ``STEP N``), per non introdurre un
secondo formato di output. ``PYTHONUNBUFFERED=1`` e' esportato da
``cluster/_lib.sh``, quindi non serve ``flush=True``; viene comunque passato
per rendere il modulo utilizzabile anche fuori dal container.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Generator
from contextlib import contextmanager

__all__ = ["phase", "format_duration", "log_step"]


def format_duration(seconds: float) -> str:
    """Formatta una durata in modo leggibile a colpo d'occhio.

    Args:
        seconds: Durata in secondi.

    Returns:
        ``"4.2s"`` sotto il minuto, ``"6m12s"`` sotto l'ora, ``"1h03m"`` oltre.
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    if seconds < 3600:
        minutes, secs = divmod(int(seconds), 60)
        return f"{minutes}m{secs:02d}s"
    hours, rem = divmod(int(seconds), 3600)
    return f"{hours}h{rem // 60:02d}m"


def _print_sink(message: str) -> None:
    """Sink predefinito: ``print`` con flush.

    ``PYTHONUNBUFFERED=1`` e' esportato da ``cluster/_lib.sh``, quindi il
    flush e' ridondante nel container; viene passato per rendere il modulo
    usabile anche fuori.
    """
    print(message, flush=True)


@contextmanager
def phase(
    label: str,
    *,
    detail: str = "",
    indent: str = "  ",
    threshold: float = 0.0,
    sink: Callable[[str], None] | None = None,
    enabled: bool = True,
) -> Generator[None]:
    """Annuncia una fase prima di eseguirla e ne riporta la durata dopo.

    Il messaggio di apertura e' emesso *prima* del lavoro: e' il punto
    essenziale, perche' un messaggio a posteriori non dice nulla mentre il
    processo e' fermo.

    Args:
        label: Nome della fase, all'imperativo o al gerundio.
        detail: Informazione dimensionale opzionale (per esempio il numero di
            elementi), utile per stimare a occhio se la durata e' plausibile.
        indent: Indentazione, per coerenza con i banner ``STEP N``. Ignorata
            quando si passa un ``sink`` che formatta da se' (per esempio un
            logger, che ha il proprio prefisso).
        threshold: Se maggiore di zero, la riga di chiusura viene emessa solo
            quando la durata la supera. Serve per le fasi che di norma sono
            istantanee e vanno segnalate solo quando degenerano.
        sink: Funzione che riceve il messaggio. Se omessa si usa ``print``.
            Serve ai moduli che emettono via ``logger.info``: passare
            ``logger.info`` evita di introdurre un secondo canale di output
            nello stesso percorso, che altererebbe il formato dei log.
        enabled: Se falso il contesto non emette nulla ed esegue solo il
            blocco. Serve nei percorsi distribuiti, dove solo il processo
            principale deve scrivere.

    Yields:
        Nulla; il contesto esiste per delimitare la fase.

    Example:
        >>> with phase("Retrieving few-shot examples", detail="72979 prompt"):
        ...     examples = retrieve_few_shot_batch(...)
        Retrieving few-shot examples (72979 prompt)...
        Retrieving few-shot examples: 6m12s

        Con un logger come sink, senza indentazione propria:

        >>> with phase("Loading model", sink=logger.info, indent=""):
        ...     model = load()
    """
    emit = (sink or _print_sink) if enabled else None
    suffix = f" ({detail})" if detail else ""
    if emit is not None:
        emit(f"{indent}{label}{suffix}...")
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        if emit is not None and elapsed >= threshold:
            emit(f"{indent}{label}: {format_duration(elapsed)}")


def log_step(number: int, title: str, *, width: int = 60) -> None:
    """Stampa un banner di fase nel formato gia' usato dai punti di ingresso.

    Centralizzare il formato evita che i banner divergano fra
    ``grpo_t2g_train``, ``sft_train`` ed ``eval_t2g``, che oggi lo
    replicano a mano.

    Args:
        number: Numero progressivo della fase.
        title: Titolo della fase.
        width: Larghezza della riga di separazione.
    """
    print(f"\n{'=' * width}", flush=True)
    print(f"STEP {number}: {title}", flush=True)
