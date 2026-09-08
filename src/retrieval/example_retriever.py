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

# tqdm fallback for Apptainer containers without tqdm installed
try:
    from tqdm import tqdm
except ImportError:

    def tqdm(iterable=None, **kwargs):
        return iterable if iterable is not None else iter(())


from src.utils.phase_timing import phase

logger = logging.getLogger(__name__)

_META_VERSION = 1
_META_FILENAME = "meta.json"
_INDEX_FILENAME = "index.pkl"
_SUPPORTED_BACKENDS = ("tfidf", "minilm")
# Query per chunk nel percorso batch: gli score densi sono C x N float64,
# quindi con N = 72979 e C = 256 si arriva a ~150 MB per chunk (72979 * 256
# * 8 byte).  C = 512 raddoppierebbe a ~300 MB; float32 dimezzerebbe ma
# cambiare i valori romperebbe la bit-parita' con il percorso a query
# singola, che e' un requisito.
_SCORE_CHUNK = 256
# Finestra top-M iniziale della selezione: see _select_results.
_TOP_M_BASE = 64
_TOP_M_K_FACTOR = 8
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
        # Il fit su ~73k documenti costa 30-120s in silenzio: la fase viene
        # annunciata PRIMA di iniziare, perche' un messaggio a posteriori non
        # serve a nulla mentre il processo e' fermo.
        with phase("Building TF-IDF index", detail=f"{len(self._texts)} documents"):
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
        excluded = {normalize_text(t) for t in (exclude or ())}
        return self._select_results(
            scores, k, max_self_similarity=max_self_similarity, excluded=excluded
        )

    def retrieve_batch(
        self,
        queries: list[str],
        k: int,
        *,
        exclude: set[str] | None = None,
        max_self_similarity: float = 0.98,
        per_query_exclude: list[set[str]] | None = None,
    ) -> list[list[RetrievedExample]]:
        """Apply :meth:`retrieve` to every query with one vectorized pass.

        Le query vengono trasformate in un colpo solo e gli score calcolati
        a chunk con un prodotto matriciale per blocco, invece che con una
        ``transform`` e un prodotto per query; la selezione per riga usa la
        stessa logica di :meth:`retrieve` (finestra top-M con fallback), quindi
        sul backend ``tfidf`` i risultati sono bit-identici alle chiamate
        singole.  Sul backend ``minilm`` gli score passano da BLAS in batch e
        possono differire di qualche ulp dal percorso a query singola.

        Args:
            queries: Lista di query.
            k: Number of examples per query.
            exclude: Set di testi da non restituire mai (normalizzati
                internamente), valido per tutte le query.
            max_self_similarity: See :meth:`retrieve`.
            per_query_exclude: Un set per query, unito a ``exclude``; serve
                al caso "ogni query esclude se stessa" di
                :func:`src.training.retrieval_setup.retrieve_few_shot_batch`.
                Deve avere una voce per ogni query.

        Returns:
            One list of examples per query.

        Raises:
            ValueError: Se ``per_query_exclude`` non ha una voce per query.
        """
        if not queries:
            return []
        if not self._texts:
            raise RuntimeError("Retriever has no indexed corpus; call build() first")
        if k <= 0:
            return [[] for _ in queries]
        if per_query_exclude is not None and len(per_query_exclude) != len(queries):
            raise ValueError(
                "per_query_exclude must have one entry per query "
                f"(got {len(per_query_exclude)} for {len(queries)} queries)"
            )
        shared = {normalize_text(t) for t in (exclude or ())}
        # Set per riga precalcolati: nel loop caldo non si rifanno unioni ne'
        # normalizzazioni a ogni query.
        if per_query_exclude is None:
            row_excludes: list[set[str] | None] = [None] * len(queries)
        else:
            row_excludes = [
                shared | {normalize_text(t) for t in entry}
                for entry in per_query_exclude
            ]

        n_chunks = -(-len(queries) // _SCORE_CHUNK)
        chunks: Any = self._iter_score_chunks(queries)
        # La barra ha senso solo su batch grandi (piu' di un chunk): su
        # batch piccoli e' solo rumore e inquina l'output catturato dai test.
        if len(queries) > _SCORE_CHUNK:
            chunks = tqdm(
                chunks, total=n_chunks, desc="Retrieving examples", unit="chunk"
            )
        results: list[list[RetrievedExample]] = []
        offset = 0
        for scores_chunk in chunks:
            for row in range(scores_chunk.shape[0]):
                row_exclude = row_excludes[offset + row]
                excluded = shared if row_exclude is None else row_exclude
                results.append(
                    self._select_results(
                        scores_chunk[row],
                        k,
                        max_self_similarity=max_self_similarity,
                        excluded=excluded,
                    )
                )
            offset += scores_chunk.shape[0]
        return results

    def _iter_score_chunks(self, queries: list[str]) -> Any:
        """Genera gli score a chunk: un ndarray ``(C, N)``, una query per riga.

        La memoria del chunk e' il compromesso chiave: gli score densi sono
        ``C x N`` float64, quindi con N = 72979 e C = 256 si tengono ~150 MB
        per chunk.  ``float32`` dimezzerebbe il picco ma cambiare i valori
        romperebbe la bit-parita' con :meth:`retrieve`.
        """
        n = len(queries)
        if self.backend == "tfidf":
            # Una sola transform per tutte le query: nel percorso a query
            # singola il costo dominante era la ripetizione di
            # transform([q]) per ogni query.
            q_matrix = self._vectorizer.transform(queries)
            for start in range(0, n, _SCORE_CHUNK):
                q_chunk = q_matrix[start : start + _SCORE_CHUNK]
                # La matrice del corpus resta a SINISTRA: l'accumulo del
                # prodotto sparso segue le righe del corpus, lo stesso ordine
                # di self._matrix.dot(q_vec), quindi gli score sono
                # bit-identici al percorso a query singola (con le query a
                # sinistra l'ordine di accumulo cambia e gli score differiscono
                # negli ultimi bit).  toarray(order="F") + trasposizione da
                # una vista (C, N) a righe contigue, senza copia aggiuntiva.
                chunk = (self._matrix @ q_chunk.T).toarray(order="F").T
                yield np.clip(chunk, 0.0, 1.0)
        else:
            if self._embeddings is None:
                raise RuntimeError(
                    "Retriever has no indexed corpus; call build() first"
                )
            model = self._get_encoder()
            # Un'unica encode per tutte le query al posto di N encode([q]).
            emb = model.encode(
                queries,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            for start in range(0, n, _SCORE_CHUNK):
                emb_chunk = emb[start : start + _SCORE_CHUNK]
                # BLAS su (C, D) @ (D, N): rispetto al prodotto a query
                # singola (gemv) il blocking puo' differire di qualche ulp.
                chunk = emb_chunk @ self._embeddings.T
                yield np.clip(np.asarray(chunk, dtype=np.float64), 0.0, 1.0)

    def _select_results(
        self,
        scores: np.ndarray,
        k: int,
        *,
        max_self_similarity: float,
        excluded: set[str] | None,
    ) -> list[RetrievedExample]:
        """Top-k con i filtri anti-leakage, equivalente allo scan completo.

        Invece di ordinare tutto il corpus con ``argsort`` (O(N log N) per
        query: ~5.5 min su 72979 query) si ordina solo una finestra top-M.
        La selezione e' corretta finché nella finestra sopravvivono ai
        filtri almeno ``k`` candidati; altrimenti la finestra si allarga
        (M * 4) e in ultima istanza si ricade sull'argsort completo.  Il
        fallback e' obbligatorio: su questo corpus la query e' sempre
        presente nel corpus con similarita' 1.0 e ha near-duplicati sopra
        0.98, quindi il numero di candidati scartati non e' limitato a
        priori.
        """
        if k <= 0:
            return []
        n = scores.shape[0]
        # M = max(64, 8k): con k = 3 (il caso few-shot) una finestra da 64
        # assorbe la query se stessa e le decine di near-duplicati scartati
        # dalla soglia; il fattore 8 copia k grandi.  Da benchmark locale
        # (N = 72979): argpartition su M = 64 costa ~0.35 ms/query contro
        # ~4.5 ms dell'argsort completo.
        m = max(_TOP_M_BASE, _TOP_M_K_FACTOR * k)
        excluded = excluded or set()
        while True:
            order = self._stable_top_window(scores, m)
            results = self._scan_order(
                order,
                scores,
                k,
                max_self_similarity=max_self_similarity,
                excluded=excluded,
            )
            if len(results) == k or m >= n:
                # Con la finestra completa l'ordine e' quello dell'argsort
                # globale: il risultato (anche piu' corto di k, quando i
                # candidati superanti i filtri non bastano) e' definitivo.
                return results
            m *= 4

    @staticmethod
    def _stable_top_window(scores: np.ndarray, m: int) -> np.ndarray:
        """Indici del top-``m`` in ordine stabile (score desc, indice asc).

        ``np.argpartition`` non e' stabile e a parita' di score sceglie un
        sottoinsieme arbitrario degli ex aequo; la finestra deve invece
        contenere ESATTAMENTE il prefisso stabile del ranking completo,
        altrimenti a parita' di score emergerebbero indici diversi dallo
        scan di riferimento (non e' un caso teorico: la maggior parte del
        corpus ha score 0.0 e il pareggio a cavallo della finestra e' la
        norma).  Per questo si ricava la soglia ``tau`` (l'm-esimo score
        maggiore), si prendono tutti gli strettamente maggiori di ``tau``
        piu' i pareggi a ``tau`` con indice piu' basso finché non si arriva
        a ``m``, e infine si ordina la finestra con ``lexsort`` su
        (indice, -score), che replica il tie-breaking di
        ``np.argsort(-scores, kind="stable")``.
        """
        n = scores.shape[0]
        if m >= n:
            return np.argsort(-scores, kind="stable")
        # Il pivot alla posizione m-1 dell'argpartition e' l'm-esimo score
        # maggiore, indipendentemente da quale pareggio finisce nel pivot.
        tau = scores[np.argpartition(-scores, m - 1)[m - 1]]
        greater = np.flatnonzero(scores > tau)
        ties = np.flatnonzero(scores == tau)[: m - greater.size]
        window = np.concatenate((greater, ties))
        # lexsort usa l'ULTIMO key come primario: score decrescente, poi
        # indice crescente per i pareggi.
        return window[np.lexsort((window, -scores[window]))]

    def _scan_order(
        self,
        order: np.ndarray,
        scores: np.ndarray,
        k: int,
        *,
        max_self_similarity: float,
        excluded: set[str],
    ) -> list[RetrievedExample]:
        """Scorre ``order`` e raccoglie fino a ``k`` candidati superanti i filtri.

        Identico allo scan dell'implementazione originale sull'argsort
        completo: e' il punto in cui la semantica (scarti e ordine) resta
        definita.
        """
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
