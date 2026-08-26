#!/usr/bin/env python3
"""Tests for the few-shot retrieval module (src/retrieval).

All tests are offline and synthetic: they build tiny corpora ad hoc and
never download models.  The MiniLM test skips unless
``sentence_transformers`` is installed AND the model is already present in
the local HuggingFace cache (``HF_HUB_OFFLINE=1`` prevents any download
attempt).
"""

from __future__ import annotations

import json

import pytest

from src.retrieval import ExampleRetriever, RetrievedExample, normalize_text


def _animal_corpus() -> tuple[list[str], list[str]]:
    """A tiny two-topic corpus: pets (cat/dog) vs politics (budget)."""
    texts = [
        "The cat sleeps on the sofa",
        "A dog runs in the park",
        "My cat loves fish",
        "The parliament approved the budget",
        "The government debates the budget",
        "Birds fly over the city",
    ]
    glosses = [
        "CAT SLEEP SOFA",
        "DOG RUN PARK",
        "MY CAT LOVE FISH",
        "PARLIAMENT APPROVE BUDGET",
        "GOVERNMENT DEBATE BUDGET",
        "BIRD FLY CITY",
    ]
    return texts, glosses


def _as_tuples(results: list[RetrievedExample]) -> list[tuple[str, str, float, int]]:
    """Reduce results to comparable tuples with rounded scores."""
    return [(r.text, r.gloss, round(r.score, 6), r.index) for r in results]


def test_tfidf_retrieves_expected_neighbors():
    texts, glosses = _animal_corpus()
    retriever = ExampleRetriever.build(texts, glosses, backend="tfidf")

    results = retriever.retrieve("My cat naps on the sofa", k=2)

    assert len(results) == 2
    # Most similar: same topic, heavy n-gram overlap.
    assert results[0].text == texts[0]
    # Second neighbor is the other pet sentence, not a politics one.
    assert results[1].text == texts[2]
    # Scores sorted descending and within [0, 1].
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)
    for r in results:
        assert 0.0 <= r.score <= 1.0
    # Both results come from the pets topic, not from politics.
    assert all(("cat" in r.text or "dog" in r.text) for r in results)


def test_exclude_never_returns_query():
    texts, glosses = _animal_corpus()
    retriever = ExampleRetriever.build(texts, glosses, backend="tfidf")
    query = "The cat sleeps on the sofa"

    # Sanity: with no filters the exact in-corpus query is returned.
    plain = retriever.retrieve(query, k=3, max_self_similarity=1.0)
    assert any(r.text == query for r in plain)

    # Exact exclusion.
    filtered = retriever.retrieve(
        query, k=3, exclude={query}, max_self_similarity=1.0
    )
    assert all(r.text != query for r in filtered)

    # Exclusion via normalized form: differently cased/spaced query text
    # normalizes to the same sentence and is still excluded.
    quirky = "  the CAT  sleeps on the   sofa  "
    assert normalize_text(quirky) == normalize_text(query)
    filtered2 = retriever.retrieve(
        quirky, k=3, exclude={normalize_text(query)}, max_self_similarity=1.0
    )
    assert all(r.text != query for r in filtered2)


def test_max_self_similarity_filters_near_duplicates():
    texts = [
        "The government plans to raise taxes next year",
        "The government plans to raise taxes next decade",
        "The cat sits on the windowsill",
    ]
    glosses = [
        "GOVERNMENT PLAN RAISE TAX NEXT YEAR",
        "GOVERNMENT PLAN RAISE TAX NEXT DECADE",
        "CAT SIT WINDOWSILL",
    ]
    retriever = ExampleRetriever.build(texts, glosses, backend="tfidf")
    query = "The government plans to raise taxes next year"

    # Permissive threshold: the one-word near-duplicate is retrieved.
    loose = retriever.retrieve(query, k=2, max_self_similarity=1.0)
    assert any("decade" in r.text for r in loose)

    # Strict threshold: both the exact query and the near-duplicate are
    # dropped (scores above 0.5), leaving only the unrelated sentence.
    strict = retriever.retrieve(query, k=2, max_self_similarity=0.5)
    assert all("year" not in r.text and "decade" not in r.text for r in strict)
    assert {r.text for r in strict} == {texts[2]}


def test_save_load_roundtrip(tmp_path):
    texts, glosses = _animal_corpus()
    retriever = ExampleRetriever.build(texts, glosses, backend="tfidf", seed=7)
    out_dir = tmp_path / "retriever"
    retriever.save(out_dir)

    loaded = ExampleRetriever.load(out_dir)
    assert loaded.backend == "tfidf"
    assert loaded.seed == 7

    query = "My cat naps on the sofa"
    assert _as_tuples(loaded.retrieve(query, k=2, max_self_similarity=1.0)) == _as_tuples(
        retriever.retrieve(query, k=2, max_self_similarity=1.0)
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("seed", 999),  # seed forged to a different value
        ("backend", "minilm"),
        ("model_name", "some/other-model"),
        ("n_examples", 0),
        ("version", 99),
    ],
)
def test_load_meta_mismatch_raises(tmp_path, field, value):
    texts, glosses = _animal_corpus()
    retriever = ExampleRetriever.build(texts, glosses, backend="tfidf", seed=42)
    out_dir = tmp_path / "retriever"
    retriever.save(out_dir)

    meta_path = out_dir / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta[field] = value
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    with pytest.raises(ValueError, match="mismatch"):
        ExampleRetriever.load(out_dir)


def test_retrieve_batch_matches_single_retrieval():
    texts, glosses = _animal_corpus()
    retriever = ExampleRetriever.build(texts, glosses, backend="tfidf")
    queries = ["My cat naps on the sofa", "The parliament debates the budget"]

    batch = retriever.retrieve_batch(queries, k=2)
    singles = [retriever.retrieve(q, k=2) for q in queries]

    assert _as_tuples(batch[0]) == _as_tuples(singles[0])
    assert _as_tuples(batch[1]) == _as_tuples(singles[1])


def test_retrieve_k_zero_returns_empty():
    texts, glosses = _animal_corpus()
    retriever = ExampleRetriever.build(texts, glosses, backend="tfidf")
    assert retriever.retrieve("anything", k=0) == []


def test_minilm_backend(tmp_path, monkeypatch):
    """MiniLM backend — skipped unless installed AND cached locally."""
    pytest.importorskip("sentence_transformers")
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")  # never attempt a download
    texts, glosses = _animal_corpus()
    try:
        retriever = ExampleRetriever.build(
            texts,
            glosses,
            backend="minilm",
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            device="cpu",
        )
    except Exception as e:
        pytest.skip(f"MiniLM model not in local cache: {e}")
    else:
        query = "A dog runs in the park"
        results = retriever.retrieve(query, k=2)
        assert len(results) == 2
        assert all(0.0 <= r.score <= 1.0 for r in results)

        # save/load roundtrip for the minilm backend.
        out_dir = tmp_path / "minilm_retriever"
        retriever.save(out_dir)
        loaded = ExampleRetriever.load(out_dir, device="cpu")
        assert _as_tuples(loaded.retrieve(query, k=2)) == _as_tuples(results)
