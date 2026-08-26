"""Few-shot retrieval setup shared by GRPO training and evaluation.

Builds (and caches on disk) an :class:`~src.retrieval.ExampleRetriever`
over the deduplicated TRAIN split, then retrieves ``top_k`` similar
``(text, gloss)`` examples per query so GRPO/eval prompts can be augmented
with few-shot demonstrations.

Anti-leakage
------------
The retriever corpus is the TRAIN split only — eval/test queries are never
indexed.  Each query additionally excludes its own normalized text
(``exclude={normalize_text(query)}``) and drops candidates whose similarity
exceeds ``max_self_similarity`` (default ``0.98``), so neither the query
itself nor a near-duplicate whose gold gloss could be copied can leak into
the few-shot demonstrations.

The same code path is used by GRPO training and evaluation
(``grpo_t2g_train.py`` / ``eval_t2g.py``) so train/inference prompts stay
consistent.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from src.retrieval import ExampleRetriever, RetrievedExample, normalize_text

logger = logging.getLogger(__name__)

_DEFAULT_CACHE_PATH = "data/retriever_index"
_DEFAULT_TOP_K = 3
_DEFAULT_MAX_SELF_SIMILARITY = 0.98


def _resolve_cache_path(
    retriever_cfg: dict[str, Any],
    cache_dir: str | Path | None,
) -> Path:
    """Resolve the on-disk retriever index path.

    ``retriever_cfg["cache_path"]`` wins when set; otherwise the index lives
    in ``<cache_dir>/retriever_index`` when a base ``cache_dir`` is given,
    falling back to ``data/retriever_index``.
    """
    configured = retriever_cfg.get("cache_path")
    if configured:
        return Path(configured)
    if cache_dir is not None:
        return Path(cache_dir) / "retriever_index"
    return Path(_DEFAULT_CACHE_PATH)


def build_train_retriever(
    dataset_dict: Any,
    retriever_cfg: dict[str, Any] | None,
    cache_dir: str | Path | None = None,
    *,
    seed: int = 42,
    device: str | None = None,
) -> ExampleRetriever | None:
    """Build (or load from cache) the few-shot retriever over the TRAIN split.

    The corpus is the already-deduplicated ``dataset_dict["train"]`` split
    (columns ``text``/``gloss``).  When ``cache_path`` points to an existing
    index whose sidecar meta is consistent (version, backend, model_name,
    seed, n_examples) the index is loaded instead of rebuilt; any mismatch
    (e.g. a different seed or corpus size) triggers a rebuild.  ``save``/``load``
    live on the retriever itself, so no extra cache bookkeeping is needed.

    Args:
        dataset_dict: A Hugging Face ``DatasetDict`` as returned by
            ``download_aslg_dataset`` (only ``"train"`` is indexed).
        retriever_cfg: The resolved ``retrieval`` config section
            (``enabled``, ``backend``, ``model_name``, ``top_k``,
            ``max_self_similarity``, ``cache_path``).
        cache_dir: Base directory used for the index cache when
            ``retriever_cfg["cache_path"]`` is unset.
        seed: Random seed for the index (should match the dataset seed; used
            for the meta consistency check on load).
        device: Device for the ``"minilm"`` backend (default: auto-detect).

    Returns:
        A ready-to-use :class:`ExampleRetriever`, or ``None`` when retrieval
        is disabled in the config (⇒ zero-shot prompts).

    Raises:
        ValueError: If ``backend``/``device`` are invalid or the train split
            is empty.
    """
    cfg = retriever_cfg or {}
    if not cfg.get("enabled", False):
        logger.info("Few-shot retrieval disabled in config — zero-shot prompts")
        return None

    backend = cfg.get("backend", "tfidf")
    model_name = cfg.get("model_name")
    top_k = int(cfg.get("top_k", _DEFAULT_TOP_K))
    max_self_similarity = float(
        cfg.get("max_self_similarity", _DEFAULT_MAX_SELF_SIMILARITY)
    )
    cache_path = _resolve_cache_path(cfg, cache_dir)

    # ── Load from cache when present and meta-consistent ────────────────
    # ExampleRetriever.load validates the meta sidecar against the pickled
    # index (version/backend/model_name/seed/n_examples); any mismatch
    # raises and we fall back to rebuilding the index.
    if cache_path.exists():
        try:
            retriever = ExampleRetriever.load(cache_path, device=device)
            logger.info(
                "Loaded few-shot retriever from %s (backend=%s, n_examples=%d, "
                "top_k=%d, max_self_similarity=%.2f)",
                cache_path,
                retriever.backend,
                len(retriever._texts),  # noqa: SLF001 — corpus size for logging
                top_k,
                max_self_similarity,
            )
            return retriever
        except (ValueError, OSError) as exc:
            logger.warning(
                "Retriever cache %s is stale or invalid (%s); rebuilding",
                cache_path,
                exc,
            )

    # ── Build a fresh index over the deduplicated TRAIN split ───────────
    train_ds = dataset_dict["train"]
    texts = [str(sample["text"]) for sample in train_ds]
    glosses = [str(sample["gloss"]) for sample in train_ds]
    retriever = ExampleRetriever.build(
        texts,
        glosses,
        backend=backend,
        model_name=model_name,
        seed=seed,
        device=device,
    )
    retriever.save(cache_path)
    logger.info(
        "Built few-shot retriever (backend=%s, n_examples=%d, top_k=%d, "
        "max_self_similarity=%.2f) and cached it to %s",
        backend,
        len(texts),
        top_k,
        max_self_similarity,
        cache_path,
    )
    return retriever


def retrieve_few_shot_batch(
    retriever: ExampleRetriever,
    queries: list[str],
    top_k: int,
    max_self_similarity: float = _DEFAULT_MAX_SELF_SIMILARITY,
) -> list[list[RetrievedExample]]:
    """Retrieve few-shot examples per query with per-query anti-leakage.

    ``ExampleRetriever.retrieve_batch`` applies a single ``exclude`` set to
    every query; here each query excludes its OWN normalized text, so the
    query itself (or any exact in-corpus duplicate) can never appear among
    its own few-shot demonstrations.

    Args:
        retriever: The built few-shot retriever (see
            :func:`build_train_retriever`).
        queries: Source English sentences.
        top_k: Number of examples per query (fewer if not enough candidates
            survive the anti-leakage filters).
        max_self_similarity: Drop candidates whose similarity to the query
            is above this threshold (near-duplicate anti-leakage).

    Returns:
        One list of :class:`RetrievedExample` per query, best first.
    """
    return [
        retriever.retrieve(
            query,
            top_k,
            exclude={normalize_text(query)},
            max_self_similarity=max_self_similarity,
        )
        for query in queries
    ]
