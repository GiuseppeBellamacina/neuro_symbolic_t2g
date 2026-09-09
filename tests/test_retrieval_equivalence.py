"""Test di equivalenza fra la selezione a finestra (argpartition + fallback)
e lo scan completo su argsort stabile, che e' l'implementazione di
riferimento pre-ottimizzazione.

Contratto: la nuova ``retrieve``/``retrieve_batch``/``retrieve_few_shot_batch``
deve restituire risultati IDENTICI alla referenza — stesso ordine, stessi
indici, stessi valori di score, compreso il tie-breaking a parita' di score
(prima vince l'indice piu' basso, come ``np.argsort(..., kind="stable")``).
Sul backend tfidf la bit-parita' con le chiamate singole e' vera; i confronti
negli score usano comunque ``pytest.approx`` per robustezza.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.retrieval import ExampleRetriever, RetrievedExample, normalize_text
from src.training.retrieval_setup import retrieve_few_shot_batch

# ---------------------------------------------------------------------------
# Referenza: l'algoritmo pre-ottimizzazione (argsort completo + scan)
# ---------------------------------------------------------------------------


def _reference_retrieve(
    retriever: ExampleRetriever,
    query: str,
    k: int,
    *,
    exclude: set[str] | None = None,
    max_self_similarity: float = 0.98,
) -> list[tuple[str, str, float, int]]:
    """Copia fedele della vecchia ``retrieve``: argsort completo + scan."""
    scores = retriever._query_scores(query)
    order = np.argsort(-scores, kind="stable")
    excluded = {normalize_text(t) for t in (exclude or ())}
    out: list[tuple[str, str, float, int]] = []
    for idx in order:
        score = float(scores[idx])
        if score > max_self_similarity:
            continue
        if retriever._normalized_texts[idx] in excluded:
            continue
        out.append((retriever._texts[idx], retriever._glosses[idx], score, int(idx)))
        if len(out) == k:
            break
    return out


def _as_tuples(results: list[RetrievedExample]) -> list[tuple[str, str, float, int]]:
    return [(r.text, r.gloss, r.score, r.index) for r in results]


def _assert_same_as_reference(
    actual: list[RetrievedExample],
    expected: list[tuple[str, str, float, int]],
) -> None:
    """Testo, gloss e indice devono coincidere ESATTAMENTE e nell'ordine;
    lo score e' confrontato con approx."""
    assert len(actual) == len(expected)
    for got, (text, gloss, score, index) in zip(actual, expected):
        assert (got.text, got.gloss, got.index) == (text, gloss, index)
        assert got.score == pytest.approx(score, rel=1e-12, abs=1e-12)


# ---------------------------------------------------------------------------
# Corpora sintetici
# ---------------------------------------------------------------------------


def _equivalence_corpus() -> tuple[list[str], list[str], dict[str, str]]:
    """Corpus con duplicati esatti e near-duplicati deliberati.

    I duplicati esatti (stesso testo) generano similarita' 1.0, quindi
    candidati sopra ``max_self_similarity``; i near-duplicati (una parola
    sostituita) popolano la banda sotto la soglia di default.  Ritorna anche
    query di vario tipo per coprire i casi di filtro.
    """
    rng = np.random.default_rng(20260908)
    pool = [
        "cat",
        "dog",
        "fish",
        "bird",
        "sofa",
        "park",
        "budget",
        "minister",
        "agreement",
        "garden",
        "market",
        "morning",
        "evening",
        "house",
        "street",
        "child",
        "teacher",
        "coffee",
        "letter",
        "winter",
        "mountain",
        "river",
        "clock",
        "paper",
    ]

    def sentence(lo: int, hi: int) -> str:
        n = int(rng.integers(lo, hi))
        return " ".join(pool[int(rng.integers(0, len(pool)))] for _ in range(n))

    texts: list[str] = []
    glosses: list[str] = []
    base: list[str] = []
    for _ in range(40):
        sent = sentence(6, 12)
        base.append(sent)
        texts.append(sent)
        glosses.append(" ".join(w.upper() for w in sent.split()))
    # 20 perturbazioni alternate: duplicato esatto e near-duplicato.
    for j in range(20):
        src = base[(j * 2) % len(base)]
        if j % 2 == 0:
            texts.append(src)
            glosses.append("COPY " + " ".join(w.upper() for w in src.split()))
        else:
            words = src.split()
            # La sostituzione potrebbe in teoria rigenerare la frase di
            # partenza: con seed fisso non accade, e comunque non cambierebbe
            # l'equivalenza da verificare.
            words[int(rng.integers(0, len(words)))] = pool[
                int(rng.integers(0, len(pool)))
            ]
            texts.append(" ".join(words))
            glosses.append("NEAR " + " ".join(w.upper() for w in words))
    for _ in range(140):
        sent = sentence(4, 10)
        texts.append(sent)
        glosses.append(" ".join(w.upper() for w in sent.split()))

    near = base[3].split()
    near[1] = pool[(pool.index(near[1]) + 7) % len(pool)]
    queries = {
        "in_corpus": base[0],
        "near_query": " ".join(near),
        "oov": "qqq zzz wxyz",
        "short": " ".join(pool[:2]),
    }
    return texts, glosses, queries


def _dup_corpus(n_copies: int, n_fill: int) -> tuple[list[str], list[str], str]:
    """``n_copies`` copie esatte di un testo + riempimento a vocabolario
    disgiunto (score esattamente 0.0 contro il target)."""
    rng = np.random.default_rng(7)
    greek = [
        "alpha",
        "beta",
        "gamma",
        "delta",
        "epsilon",
        "zeta",
        "eta",
        "theta",
        "iota",
        "kappa",
    ]
    target = "the minister signed the agreement yesterday morning"
    fillers = []
    for _ in range(n_fill):
        n = int(rng.integers(4, 8))
        fillers.append(
            " ".join(greek[int(rng.integers(0, len(greek)))] for _ in range(n))
        )
    texts = [target] * n_copies + fillers
    target_gloss = "MINISTER SIGN AGREEMENT YESTERDAY MORNING"
    glosses = [target_gloss] * n_copies + [
        " ".join(w.upper() for w in f.split()) for f in fillers
    ]
    return texts, glosses, target


@pytest.fixture(scope="module")
def equiv_retriever() -> tuple[ExampleRetriever, dict[str, str]]:
    texts, glosses, queries = _equivalence_corpus()
    return ExampleRetriever.build(texts, glosses), queries


def _exclude_for(mode: str, query: str, retriever: ExampleRetriever):
    if mode == "none":
        return None
    if mode == "self":
        return {normalize_text(query)}
    # Misto: la query, una frase del corpus e una stringa inesistente.
    return {
        normalize_text(query),
        normalize_text(retriever._texts[5]),
        "not in corpus",
    }


# ---------------------------------------------------------------------------
# Equivalenza di retrieve() contro la referenza
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "query_name,k,exclude_mode,threshold",
    [
        ("in_corpus", 3, "none", 0.98),
        ("in_corpus", 1, "none", 0.98),
        ("in_corpus", 2, "self", 0.98),
        ("in_corpus", 10, "self", 0.98),
        ("in_corpus", 8, "none", 1.0),  # i duplicati a 1.0 sopravvivono
        ("in_corpus", 5, "none", 0.5),  # banda sotto soglia
        ("near_query", 3, "none", 0.98),
        ("near_query", 4, "self", 0.99),
        ("oov", 3, "none", 0.98),  # tutti 0.0: pareggio su tutta la finestra
        ("short", 6, "none", 0.98),
        ("short", 3, "mixed", 0.98),
    ],
)
def test_retrieve_matches_reference(
    equiv_retriever, query_name, k, exclude_mode, threshold
):
    retriever, queries = equiv_retriever
    query = queries[query_name]
    exclude = _exclude_for(exclude_mode, query, retriever)
    expected = _reference_retrieve(
        retriever, query, k, exclude=exclude, max_self_similarity=threshold
    )
    got = retriever.retrieve(query, k, exclude=exclude, max_self_similarity=threshold)
    _assert_same_as_reference(got, expected)

    if query_name == "oov":
        # Pareggio esatto a 0.0 su tutto il corpus: a parita' di score
        # vincono gli indici piu' bassi (tie-breaking dell'argsort stable).
        assert [r.index for r in got] == [0, 1, 2]


def test_k_larger_than_surviving_candidates():
    """k maggiore del numero di candidati sopravvissuti ai filtri: la lista
    e' piu' corta di k (qui 200 richiesti, 60 i riempimenti sotto soglia)
    ma identica alla referenza."""
    texts, glosses, target = _dup_corpus(80, 60)
    retriever = ExampleRetriever.build(texts, glosses)

    got = retriever.retrieve(target, 200)
    expected = _reference_retrieve(retriever, target, 200)

    _assert_same_as_reference(got, expected)
    assert len(got) == 60


def test_all_candidates_above_threshold_returns_empty():
    # Con la soglia di default tutti e tre i candidati (score 1.0) vengono
    # scartati: il risultato e' una lista vuota, come nella referenza.
    base = "the minister signed the agreement yesterday morning"
    texts = [base, base, base]
    glosses = ["MINISTER SIGN AGREEMENT YESTERDAY MORNING"] * 3
    retriever = ExampleRetriever.build(texts, glosses)

    got = retriever.retrieve(base, 3)
    expected = _reference_retrieve(retriever, base, 3)

    assert got == []
    _assert_same_as_reference(got, expected)


def test_exact_ties_break_toward_lower_index():
    # Le tre copie della query hanno lo stesso score esatto (vettori
    # tf-idf identici): il tie-breaking deve preferire l'indice piu' basso.
    texts = [
        "alpha beta gamma",
        "the minister signed the agreement",
        "alpha beta gamma",
        "delta epsilon zeta",
        "alpha beta gamma",
        "the minister signed the agreement",
    ]
    glosses = [
        "A B G",
        "T M S T A",
        "A B G COPY",
        "D E Z",
        "A B G COPY2",
        "T M S T A COPY",
    ]
    retriever = ExampleRetriever.build(texts, glosses)

    got = retriever.retrieve("alpha beta gamma", 3, max_self_similarity=1.0)
    expected = _reference_retrieve(
        retriever, "alpha beta gamma", 3, max_self_similarity=1.0
    )

    _assert_same_as_reference(got, expected)
    # score identici e indici in ordine crescente fra le copie
    assert got[0].score == got[1].score == got[2].score
    assert [x.index for x in got] == [0, 2, 4]


# ---------------------------------------------------------------------------
# Fallback della finestra
# ---------------------------------------------------------------------------


def test_fallback_when_threshold_filters_whole_window():
    # Le prime 80 righe hanno score 1.0 > 0.98: la finestra iniziale M=64
    # contiene solo candidati scartati, quindi serve il fallback.
    texts, glosses, target = _dup_corpus(80, 60)
    retriever = ExampleRetriever.build(texts, glosses)

    got = retriever.retrieve(target, 3)
    expected = _reference_retrieve(retriever, target, 3)

    _assert_same_as_reference(got, expected)
    assert len(got) == 3
    assert all(x.text != target for x in got)


def test_fallback_widens_window(monkeypatch):
    """La finestra parte da M=64 e viene allargata quando i sopravvissuti
    non bastano; quando M supera la dimensione del corpus si ricade
    sull'argsort completo."""
    texts, glosses, target = _dup_corpus(80, 60)
    retriever = ExampleRetriever.build(texts, glosses)
    calls: list[int] = []
    orig = ExampleRetriever._stable_top_window

    def spy(self, scores, m):
        calls.append(int(m))
        return orig(scores, m)

    monkeypatch.setattr(ExampleRetriever, "_stable_top_window", spy)

    got = retriever.retrieve(target, 3)
    expected = _reference_retrieve(retriever, target, 3)

    _assert_same_as_reference(got, expected)
    # n = 140: parte da 64 e l'allargamento a 256 copre tutto il corpus
    # (argsort completo come ultimo tentativo).
    assert calls == [64, 256]


def test_fallback_widens_twice(monkeypatch):
    """Con 300 copie scartate dalla soglia servono DUE allargamenti prima
    dell'argsort completo."""
    texts, glosses, target = _dup_corpus(300, 100)
    retriever = ExampleRetriever.build(texts, glosses)
    calls: list[int] = []
    orig = ExampleRetriever._stable_top_window

    def spy(self, scores, m):
        calls.append(int(m))
        return orig(scores, m)

    monkeypatch.setattr(ExampleRetriever, "_stable_top_window", spy)

    got = retriever.retrieve(target, 3)
    expected = _reference_retrieve(retriever, target, 3)

    _assert_same_as_reference(got, expected)
    # n = 400: 64 -> 256 (ancora solo copie scartate) -> 1024 >= 400.
    assert calls == [64, 256, 1024]


def test_fallback_when_exclude_removes_top_candidates():
    """Con soglia 1.0 gli 80 duplicati NON scattano il filtro di similarita'
    (1.0 non e' > 1.0): e' l'exclude a rimuovere i primi M candidati,
    forzando il fallback."""
    texts, glosses, target = _dup_corpus(80, 60)
    retriever = ExampleRetriever.build(texts, glosses)
    exclude = {normalize_text(target)}

    got = retriever.retrieve(target, 3, exclude=exclude, max_self_similarity=1.0)
    expected = _reference_retrieve(
        retriever, target, 3, exclude=exclude, max_self_similarity=1.0
    )

    _assert_same_as_reference(got, expected)
    assert len(got) == 3
    assert all(normalize_text(x.text) != normalize_text(target) for x in got)


# ---------------------------------------------------------------------------
# Equivalenza dei percorsi batch
# ---------------------------------------------------------------------------


def test_retrieve_batch_matches_reference(equiv_retriever):
    retriever, queries = equiv_retriever
    batch_queries = [
        queries["in_corpus"],
        queries["near_query"],
        queries["oov"],
        queries["short"],
        queries["in_corpus"],  # duplicata: stessa query due volte nel batch
    ]
    for threshold in (0.98, 1.0):
        expected = [
            _reference_retrieve(retriever, q, 3, max_self_similarity=threshold)
            for q in batch_queries
        ]
        got = retriever.retrieve_batch(batch_queries, 3, max_self_similarity=threshold)
        for g, e in zip(got, expected):
            _assert_same_as_reference(g, e)
        # Bit-parita' con le chiamate singole (backend tfidf).
        singles = [
            retriever.retrieve(q, 3, max_self_similarity=threshold)
            for q in batch_queries
        ]
        assert [_as_tuples(g) for g in got] == [_as_tuples(s) for s in singles]


def test_retrieve_batch_per_query_exclude_matches_reference(equiv_retriever):
    """Il caso "exclude specifico per query" (uno set per query, unito a
    un eventuale exclude condiviso) deve coincidere con la referenza."""
    retriever, queries = equiv_retriever
    qs = [
        queries["in_corpus"],
        queries["near_query"],
        queries["short"],
        queries["oov"],
    ]
    per_query = [{normalize_text(q)} for q in qs]

    got = retriever.retrieve_batch(qs, 3, per_query_exclude=per_query)
    expected = [
        _reference_retrieve(retriever, q, 3, exclude={normalize_text(q)}) for q in qs
    ]
    for g, e in zip(got, expected):
        _assert_same_as_reference(g, e)

    # Union con un exclude condiviso.
    shared = {normalize_text(retriever._texts[7])}
    got2 = retriever.retrieve_batch(qs, 3, exclude=shared, per_query_exclude=per_query)
    expected2 = [
        _reference_retrieve(retriever, q, 3, exclude=shared | {normalize_text(q)})
        for q in qs
    ]
    for g, e in zip(got2, expected2):
        _assert_same_as_reference(g, e)


def test_few_shot_batch_matches_reference(equiv_retriever, capsys):
    """``retrieve_few_shot_batch`` (per-query exclude + soglia) deve dare
    gli stessi risultati del vecchio percorso a query singola; la fase
    viene annunciata prima del lavoro."""
    retriever, queries = equiv_retriever
    qs = [
        queries["in_corpus"],
        queries["near_query"],
        queries["short"],
        queries["oov"],
    ]

    got = retrieve_few_shot_batch(retriever, qs, 3, 0.98)
    expected = [
        _reference_retrieve(
            retriever, q, 3, exclude={normalize_text(q)}, max_self_similarity=0.98
        )
        for q in qs
    ]
    for g, e in zip(got, expected):
        _assert_same_as_reference(g, e)

    out = capsys.readouterr().out
    assert "Retrieving few-shot examples" in out
    assert "Retrieving few-shot examples (4 prompt)..." in out


def test_retrieve_batch_edges(equiv_retriever):
    retriever, queries = equiv_retriever
    assert retriever.retrieve_batch([], 3) == []
    assert retriever.retrieve_batch([queries["in_corpus"]], 0) == [[]]
    with pytest.raises(ValueError, match="one entry per query"):
        retriever.retrieve_batch(["a", "b"], 3, per_query_exclude=[{"x"}])


def test_few_shot_batch_matches_reference_on_dup_corpus():
    """Percorso few-shot su un corpus con 80 copie esatte della query:
    pareggi a 1.0 (soglia), pareggi a 0.0 (riempimento) e exclude per query
    che toglie tutti i duplicati."""
    texts, glosses, target = _dup_corpus(80, 60)
    retriever = ExampleRetriever.build(texts, glosses)
    queries = [target, "alpha beta gamma delta", "alpha"]

    got = retrieve_few_shot_batch(retriever, queries, 3, 0.98)
    expected = [
        _reference_retrieve(
            retriever, q, 3, exclude={normalize_text(q)}, max_self_similarity=0.98
        )
        for q in queries
    ]
    for g, e in zip(got, expected):
        _assert_same_as_reference(g, e)
    # La query stessa non compare mai fra le proprie few-shot examples.
    for q, results in zip(queries, got):
        assert all(normalize_text(x.text) != normalize_text(q) for x in results)
