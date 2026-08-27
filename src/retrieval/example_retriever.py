"""Few-shot example retrieval for T2G (English text → ASL gloss).

Retrieves the ``k`` most similar ``(text, gloss)`` training examples for a
query so that GRPO/eval prompts can be augmented with few-shot
demonstrations drawn from the TRAIN split.

This module is the standalone retrieval base; wiring the examples into
actual prompts happens at the call site and is out of scope here.

Anti-leakage
------------
Two mechanisms prevent the gold translation of the query itself (or of a
near-duplicate sentence) from leaking into the few-shot examples:

* ``exclude`` — a set of normalized texts that must never be returned.
  Use it to exclude the query itself and/or its gold ``(text, gloss)`` pair.
* ``max_self_similarity`` — candidates whose similarity to the query is
  above this threshold are dropped.  The default ``0.98`` catches
  near-duplicates whose normalized text differs only slightly (e.g. one
  word changed or extra whitespace): if such a sentence leaked, the model
  could simply copy its gold gloss instead of translating the query.

Backends
--------
* ``tfidf`` (default): ``TfidfVectorizer`` from scikit-learn with
  ``sublinear_tf=True`` and 1-2 word n-grams.  Similarity is cosine
  similarity, which is always in ``[0, 1]`` for TF-IDF vectors.  This
  backend involves no randomness, so it is deterministic; ``seed`` is
  stored only for provenance and for the meta-consistency check on load.
* ``minilm``: sentence-transformers with the default model
  ``sentence-transformers/all-MiniLM-L6-v2``.  The model is loaded lazily
  (nothing is downloaded at import time) and ``sentence_transformers`` is
  imported on demand, raising an actionable ``ImportError`` if the package
  is missing.
"""

from __future__ import annotations

import json
import logging
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

_META_VERSION = 1
_META_FILENAME = "meta.json"
_INDEX_FILENAME = "index.pkl"
_SUPPORTED_BACKENDS = ("tfidf", "minilm")
_DEFAULT_MINILM_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
_MINILM_INSTALL_HINT = (
    "Backend 'minilm' requires the optional 'retrieval' extra. "
    "Install it with: pip install -e '.[retrieval]' "
    "(or: uv sync --extra retrieval)."
)
_TFIDF_INSTALL_HINT = (
    "Backend 'tfidf' requires 'scikit-learn', a core dependency of this "
    "project. Your environment is out of sync with pyproject.toml — "
    "reinstall with: pip install -e . (or: uv sync)."
)


def normalize_text(text: str) -> str:
    """Normalize text for matching: lowercase, collapse whitespace, strip.

    Args:
        text: Raw text.

    Returns:
        The normalized text ("" for empty or whitespace-only input).
    """
    return " ".join(text.lower().split())


def _resolve_device(device: str | None) -> str:
    """Resolve a device string for the minilm backend.

    Args:
        device: "cpu", "cuda", "auto" or None (the latter means auto-detect).

    Returns:
        The resolved device name.

    Raises:
        ValueError: If ``device`` is not one of the supported values.
    """
    if device in (None, "auto"):
        try:
            import torch
        except ImportError:
            return "cpu"
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device not in ("cpu", "cuda"):
        raise ValueError(f"device must be 'cpu', 'cuda' or 'auto', got {device!r}")
    return device


@dataclass(frozen=True)
class RetrievedExample:
    """A single retrieved few-shot example.

    Attributes:
        text: The source English sentence.
        gloss: The gold ASL gloss translation.
        score: Similarity of this example to the query, in ``[0, 1]``.
        index: Position of the example in the corpus passed to ``build``.
    """

    text: str
    gloss: str
    score: float
    index: int


@dataclass
class _RetrieverState:
    """Picklable on-disk state for an :class:`ExampleRetriever`."""

    backend: str
    model_name: str | None
    seed: int
    texts: list[str]
    glosses: list[str]
    normalized_texts: list[str]
    payload: Any


class ExampleRetriever:
    """Retrieve similar ``(text, gloss)`` training examples for a query.

    Attributes:
        backend: Retrieval backend name (``"tfidf"`` or ``"minilm"``).
        model_name: Embedding model name for ``"minilm"``, else ``None``.
        seed: Random seed used when building the index.
    """

    def __init__(
        self,
        *,
        backend: str,
        model_name: str | None,
        seed: int,
        device: str | None = None,
    ) -> None:
        if backend not in _SUPPORTED_BACKENDS:
            raise ValueError(
                f"Unknown backend {backend!r}; expected one of {_SUPPORTED_BACKENDS}"
            )
        self.backend = backend
        self.model_name = model_name
        self.seed = seed
        self._device = _resolve_device(device) if backend == "minilm" else None
        self._texts: list[str] = []
        self._glosses: list[str] = []
        self._normalized_texts: list[str] = []
        self._vectorizer: Any = None
        self._matrix: Any = None
        self._embeddings: np.ndarray | None = None
        self._encoder: Any = None

    # ------------------------------------------------------------------
    # Factories
    # ------------------------------------------------------------------

    @classmethod
    def build(
        cls,
        texts: list[str],
        glosses: list[str],
        *,
        backend: str = "tfidf",
        model_name: str | None = None,
        seed: int = 42,
        device: str | None = None,
    ) -> "ExampleRetriever":
        """Build and index a retriever over the given corpus.

        Args:
            texts: Source English sentences.
            glosses: Gold ASL glosses, one per ``text``.
            backend: ``"tfidf"`` (default; deterministic, no model
                download) or ``"minilm"`` (sentence-transformers
                embeddings; requires the model to be available).
            model_name: Embedding model name for ``"minilm"`` (default:
                ``sentence-transformers/all-MiniLM-L6-v2``).  Ignored for
                ``"tfidf"``.
            seed: Random seed.  The ``"tfidf"`` backend is fully
                deterministic (it uses no randomness), so ``seed`` is
                stored for provenance and meta-consistency only.
            device: Device for ``"minilm"``: ``"cpu"``, ``"cuda"`` or
                ``"auto"`` (default: auto-detect).

        Returns:
            A ready-to-use :class:`ExampleRetriever`.

        Raises:
            ValueError: If ``texts`` and ``glosses`` differ in length, the
                corpus is empty, or ``backend``/``device`` are invalid.
        """
        if len(texts) != len(glosses):
            raise ValueError(
                f"texts and glosses must have the same length "
                f"(got {len(texts)} and {len(glosses)})"
            )
        if not texts:
            raise ValueError("Cannot build a retriever from an empty corpus")
        if backend not in _SUPPORTED_BACKENDS:
            raise ValueError(
                f"Unknown backend {backend!r}; expected one of {_SUPPORTED_BACKENDS}"
            )
        if backend == "minilm" and model_name is None:
            model_name = _DEFAULT_MINILM_MODEL
        self = cls(backend=backend, model_name=model_name, seed=seed, device=device)
        self._index(texts, glosses)
        logger.info(
            "Built %s retriever on %d examples (seed=%d)", backend, len(texts), seed
        )
        return self

    @classmethod
    def load(cls, dir: str | Path, *, device: str | None = None) -> "ExampleRetriever":
        """Load a retriever previously saved with :meth:`save`.

        The sidecar ``meta.json`` is validated against the pickled index
        (version, backend, model_name, seed, n_examples); any mismatch
        raises a ``ValueError`` with a clear message.

        Warning:
            The index pickle is deserialized with ``pickle.load``; only
            load directories you trust.

        Args:
            dir: Directory written by :meth:`save`.
            device: Device override for the ``"minilm"`` backend (default:
                auto-detect).

        Returns:
            The loaded retriever, reproducing the exact retrieval results
            of the original instance.

        Raises:
            FileNotFoundError: If ``dir`` does not contain a saved retriever.
            ValueError: If the meta sidecar is inconsistent with the index.
        """
        dir = Path(dir)
        meta_path = dir / _META_FILENAME
        index_path = dir / _INDEX_FILENAME
        if not meta_path.exists() or not index_path.exists():
            raise FileNotFoundError(
                f"No saved retriever found in {dir}: expected "
                f"{_META_FILENAME} and {_INDEX_FILENAME}"
            )
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("version") != _META_VERSION:
            raise ValueError(
                f"Retriever meta version mismatch: found "
                f"{meta.get('version')!r}, expected {_META_VERSION!r}. "
                "Rebuild the index with ExampleRetriever.build(...) and "
                "save() it again."
            )
        with open(index_path, "rb") as f:
            state = pickle.load(f)
        expected = {
            "backend": state.backend,
            "model_name": state.model_name,
            "seed": state.seed,
            "n_examples": len(state.texts),
        }
        for field, state_value in expected.items():
            meta_value = meta.get(field)
            if meta_value != state_value:
                raise ValueError(
                    f"Retriever meta mismatch for '{field}': meta.json "
                    f"says {meta_value!r} but the index was built with "
                    f"{state_value!r}. Rebuild the index with "
                    "ExampleRetriever.build(...) and save() it again."
                )
        self = cls(
            backend=state.backend,
            model_name=state.model_name,
            seed=state.seed,
            device=device,
        )
        self._texts = state.texts
        self._glosses = state.glosses
        self._normalized_texts = state.normalized_texts
        if state.backend == "tfidf":
            self._vectorizer, self._matrix = state.payload
        else:
            self._embeddings = np.asarray(state.payload, dtype=np.float32)
        logger.info(
            "Loaded %s retriever with %d examples from %s",
            state.backend,
            len(state.texts),
            dir,
        )
        return self

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------

    def _index(self, texts: list[str], glosses: list[str]) -> None:
        """Embed/index the corpus for the configured backend."""
        self._texts = list(texts)
        self._glosses = list(glosses)
        self._normalized_texts = [normalize_text(t) for t in self._texts]
        if self.backend == "tfidf":
            self._build_tfidf()
        else:
            self._build_minilm()

    def _build_tfidf(self) -> None:
        """Fit a TF-IDF vectorizer on the corpus texts."""
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
        except ImportError as e:
            raise ImportError(_TFIDF_INSTALL_HINT) from e
        self._vectorizer = TfidfVectorizer(sublinear_tf=True, ngram_range=(1, 2))
        self._matrix = self._vectorizer.fit_transform(self._texts)

    def _build_minilm(self) -> None:
        """Encode the corpus texts with the MiniLM sentence transformer."""
        model = self._get_encoder()
        vectors = model.encode(
            self._texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        self._embeddings = np.asarray(vectors, dtype=np.float32)

    def _get_encoder(self) -> Any:
        """Load (once) and return the sentence-transformer encoder."""
        if self._encoder is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as e:
                raise ImportError(_MINILM_INSTALL_HINT) from e
            logger.info(
                "Loading sentence-transformer model %s on device %s",
                self.model_name,
                self._device,
            )
            self._encoder = SentenceTransformer(self.model_name, device=self._device)
        return self._encoder

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def retrieve(
        self,
        query: str,
        k: int,
        *,
        exclude: set[str] | None = None,
        max_self_similarity: float = 0.98,
    ) -> list[RetrievedExample]:
        """Return the ``k`` most similar examples for ``query``.

        Scores are cosine similarities in ``[0, 1]``, sorted descending.

        Anti-leakage: candidates whose normalized text is in ``exclude``
        are never returned, and candidates with similarity greater than
        ``max_self_similarity`` are dropped.  With the default threshold
        of ``0.98`` this also removes the query itself when it is part of
        the indexed corpus, plus near-duplicates whose gold gloss would
        let the model copy the answer.

        Args:
            query: Source English sentence to find examples for.
            k: Number of examples to return (fewer if not enough
                candidates survive the filters).
            exclude: Set of normalized texts to never return (e.g. the
                query itself and/or its gold translation pair).
            max_self_similarity: Drop candidates with similarity above
                this threshold.

        Returns:
            List of :class:`RetrievedExample`, best first.
        """
        if k <= 0:
            return []
        if not self._texts:
            raise RuntimeError("Retriever has no indexed corpus; call build() first")
        scores = self._query_scores(query)
        order = np.argsort(-scores, kind="stable")
        excluded = {normalize_text(t) for t in (exclude or ())}
        results: list[RetrievedExample] = []
        for idx in order:
            score = float(scores[idx])
            if score > max_self_similarity:
                continue
            if self._normalized_texts[idx] in excluded:
                continue
            results.append(
                RetrievedExample(
                    text=self._texts[idx],
                    gloss=self._glosses[idx],
                    score=score,
                    index=int(idx),
                )
            )
            if len(results) == k:
                break
        return results

    def retrieve_batch(
        self,
        queries: list[str],
        k: int,
        *,
        exclude: set[str] | None = None,
        max_self_similarity: float = 0.98,
    ) -> list[list[RetrievedExample]]:
        """Apply :meth:`retrieve` to every query.

        Args:
            queries: List of queries.
            k: Number of examples per query.
            exclude: See :meth:`retrieve`.
            max_self_similarity: See :meth:`retrieve`.

        Returns:
            One list of examples per query.
        """
        return [
            self.retrieve(
                q, k, exclude=exclude, max_self_similarity=max_self_similarity
            )
            for q in queries
        ]

    def _query_scores(self, query: str) -> np.ndarray:
        """Compute the similarity of ``query`` against every corpus item."""
        if self.backend == "tfidf":
            q_vec = self._vectorizer.transform([query]).toarray().ravel()
            scores = self._matrix.dot(q_vec)
        else:
            if self._embeddings is None:
                raise RuntimeError(
                    "Retriever has no indexed corpus; call build() first"
                )
            model = self._get_encoder()
            q_vec = model.encode(
                [query], convert_to_numpy=True, normalize_embeddings=True
            )[0]
            scores = self._embeddings @ q_vec
        return np.clip(np.asarray(scores, dtype=np.float64), 0.0, 1.0)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, dir: str | Path) -> None:
        """Persist the retriever to ``dir``.

        Writes ``index.pkl`` (the pickled index state) plus a human-
        readable ``meta.json`` sidecar with ``backend``, ``model_name``,
        ``seed``, ``n_examples`` and ``version``.  The sidecar is
        validated against the pickled index on :meth:`load`, so a
        tampered or stale meta file raises a clear error.

        Args:
            dir: Directory to write into (created if needed).
        """
        out_dir = Path(dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        payload: Any = (
            (self._vectorizer, self._matrix)
            if self.backend == "tfidf"
            else self._embeddings
        )
        state = _RetrieverState(
            backend=self.backend,
            model_name=self.model_name,
            seed=self.seed,
            texts=self._texts,
            glosses=self._glosses,
            normalized_texts=self._normalized_texts,
            payload=payload,
        )
        with open(out_dir / _INDEX_FILENAME, "wb") as f:
            pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)
        meta = {
            "version": _META_VERSION,
            "backend": self.backend,
            "model_name": self.model_name,
            "seed": self.seed,
            "n_examples": len(self._texts),
            "device": self._device,
        }
        (out_dir / _META_FILENAME).write_text(
            json.dumps(meta, indent=2) + "\n", encoding="utf-8"
        )
        logger.info("Saved %s retriever to %s", self.backend, out_dir)
