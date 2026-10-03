"""Il linguaggio di uscita ammesso: UNA definizione per decoder, validity e reward.

Tre modalità (``grammar.mode`` nel config, assente = ``vocab``):

* ``vocab`` — qualunque sequenza di token del vocabolario chiuso estratto dal
  train (ASLG-PC12, PHOENIX-2014T). È il Trie dual-root storico, invariato.
* ``source_spans`` — per il NER (CoNLL-2003): ogni token è ``TIPO:pezzo``,
  con ``TIPO`` in ``grammar.span_types`` e ``pezzo`` una sequenza contigua di
  al massimo ``grammar.max_span_words`` parole DELLA FRASE DEL PROMPT (parole
  unite da ``_``), oppure il solo ``grammar.empty_token`` (``NONE``). Il
  vocabolario non viene dal train ma dall'input di ogni esempio: nessun leak,
  e ogni entità gold è raggiungibile (nel corpus ogni entità è un pezzo della
  frase), mentre con il vocabolario di train il 48% delle entità di test era
  impossibile da generare.
* ``sequences`` — classificazione gerarchica (WOS-46985): l'uscita intera
  deve essere una delle sequenze viste negli split del vocabolario (con
  ``vocab_source: train`` solo il train): dominio, poi un'area DI QUEL
  dominio, poi fine. Il Trie a token non vincolava l'ordine (``Medical
  Medical`` era generabile).

:class:`OutputGrammar` è la definizione a livello di parole. Il Trie a token
(``grammar_logits_processor.py``) e i controlli di validity/formato
(``t2g_rewards``/``metrics``) la leggono entrambi, così ciò che il decoder
impone e ciò che le metriche chiamano "valido" non possono divergere.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

GRAMMAR_MODES: tuple[str, ...] = ("vocab", "source_spans", "sequences")
DEFAULT_GRAMMAR_MODE = "vocab"


def resolve_grammar_mode(config: Mapping[str, Any] | None) -> str:
    """``grammar.mode`` del config (default ``vocab``), validato."""
    mode = str(
        ((config or {}).get("grammar") or {}).get("mode") or DEFAULT_GRAMMAR_MODE
    )
    if mode not in GRAMMAR_MODES:
        raise ValueError(
            f"grammar.mode={mode!r} non valido: atteso uno di {GRAMMAR_MODES}"
        )
    return mode


@dataclass(frozen=True)
class OutputGrammar:
    """Definizione a parole del linguaggio di uscita ammesso.

    Attributes:
        mode: Una di :data:`GRAMMAR_MODES`.
        vocab: Vocabolario chiuso (modalità ``vocab``; nelle altre resta per
            le reward che lo usano, es. i bigrammi).
        sequences: Uscite ammesse intere, come tuple di parole (``sequences``).
        span_types: Tipi ammessi davanti ai pezzi (``source_spans``).
        empty_token: Uscita "nessuna entità" (``source_spans``), solo da sola.
        max_span_words: Parole massime di un pezzo (``source_spans``).
    """

    mode: str
    vocab: frozenset[str] = frozenset()
    sequences: frozenset[tuple[str, ...]] = frozenset()
    span_types: tuple[str, ...] = ()
    empty_token: str | None = None
    max_span_words: int = 10
    _span_re: Any = field(default=None, compare=False, repr=False)

    def __post_init__(self) -> None:
        if self.mode not in GRAMMAR_MODES:
            raise ValueError(f"grammar.mode={self.mode!r} non valido")
        if self.mode == "source_spans" and not self.span_types:
            raise ValueError("grammar.mode=source_spans richiede grammar.span_types")
        if self.mode == "sequences" and not self.sequences:
            raise ValueError("grammar.mode=sequences senza sequenze ammesse")
        types = "|".join(re.escape(t) for t in self.span_types)
        object.__setattr__(self, "_span_re", re.compile(rf"(?:{types}):\S+"))

    # ── source_spans ────────────────────────────────────────────────────
    def span_items(self, source: str) -> list[str]:
        """Token ``TIPO:pezzo`` ammessi per la frase ``source`` (senza duplicati)."""
        words = source.split()
        pieces = {
            "_".join(words[i:j])
            for i in range(len(words))
            for j in range(i + 1, min(i + self.max_span_words, len(words)) + 1)
        }
        return [f"{t}:{p}" for t in self.span_types for p in sorted(pieces)]

    def _span_word_ok(self, word: str, items: set[str] | None) -> bool:
        if items is not None:
            return word in items
        # Senza la frase (diagnostica di training) si controlla solo la forma.
        return bool(self._span_re.fullmatch(word))

    # ── controlli a parole ──────────────────────────────────────────────
    def word_ok(self, word: str, source: str | None = None) -> bool:
        """La parola può comparire in un'uscita ammessa (per il format parziale)."""
        if self.mode == "vocab":
            return word in self.vocab
        if self.mode == "sequences":
            return any(word in seq for seq in self.sequences)
        if word == self.empty_token:
            return True
        items = set(self.span_items(source)) if source is not None else None
        return self._span_word_ok(word, items)

    def accepts(self, text: str, source: str | None = None) -> bool:
        """``text`` è un'uscita completa ammessa dalla grammatica.

        In ``source_spans`` senza ``source`` si verifica solo la forma
        (``TIPO:qualcosa``): serve alle diagnostiche che non hanno la frase,
        mai alle metriche di valutazione, che la passano sempre.
        """
        words = text.split()
        if not words:
            return False
        if self.mode == "vocab":
            return all(w in self.vocab for w in words)
        if self.mode == "sequences":
            return tuple(words) in self.sequences
        if words == [self.empty_token]:
            return True
        items = set(self.span_items(source)) if source is not None else None
        return all(
            w != self.empty_token and self._span_word_ok(w, items) for w in words
        )


def build_output_grammar(
    config: Mapping[str, Any],
    vocab: Iterable[str],
    sequences: Iterable[str] = (),
) -> OutputGrammar:
    """Grammatica di uscita del config.

    Args:
        config: Config risolto (legge la sezione ``grammar``).
        vocab: Vocabolario chiuso (``prepare_vocab_and_bigram``).
        sequences: Gloss gold degli split del vocabolario (stesso
            ``vocab_source``), usate solo dalla modalità ``sequences``.
    """
    grammar_cfg = config.get("grammar") or {}
    mode = resolve_grammar_mode(config)
    return OutputGrammar(
        mode=mode,
        vocab=frozenset(vocab),
        sequences=(
            frozenset(tuple(s.split()) for s in sequences if s.split())
            if mode == "sequences"
            else frozenset()
        ),
        span_types=tuple(grammar_cfg.get("span_types") or ()),
        empty_token=grammar_cfg.get("empty_token"),
        max_span_words=int(grammar_cfg.get("max_span_words", 10)),
    )


def build_run_grammar(
    config: Mapping[str, Any], vocab: Iterable[str], dataset: Mapping[str, Any]
) -> OutputGrammar:
    """Grammatica del run: le sequenze di ``sequences`` vengono dagli STESSI
    split del vocabolario (``dataset.vocab_source``: solo il train di default,
    tutti gli split nell'ablazione ``full-vocab-trie``)."""
    from src.datasets.registry import resolve_vocab_source, vocab_splits

    sequences: list[str] = []
    if resolve_grammar_mode(config) == "sequences":
        source = resolve_vocab_source(config.get("dataset", {}))
        sequences = [
            g
            for split in vocab_splits(dataset, source)
            for g in dataset[split]["gloss"]
        ]
    return build_output_grammar(config, vocab, sequences)
