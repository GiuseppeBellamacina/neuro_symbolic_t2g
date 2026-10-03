"""Dataset registry: which corpus a config uses and how its artifacts are built.

Ogni config sceglie il corpus con ``dataset.dataset_name``; questo modulo è
l'unico punto che traduce quel nome in:

* il loader (:func:`load_t2g_dataset`) — ASLG-PC12 da Hugging Face (cache
  offline), PHOENIX-2014T dai CSV ufficiali in locale, e i due dataset
  non-gloss di GrammarRL (arXiv:2609.39869) da file locali: WOS-46985
  (classificazione gerarchica, target ``DOMINIO AREA``) e CoNLL-2003 (NER,
  target ``TIPO:Entità ...``). Tutti restituiscono un ``DatasetDict`` con
  ``train``/``test`` (+ ``validation`` dove esiste uno split ufficiale) e
  colonne ``text``/``gloss``, quindi il resto della pipeline non distingue;
* la chiave di layout (:attr:`DatasetSpec.key`, es. ``aslg-pc12``), cioè il
  primo segmento sotto ``experiments/{configs,checkpoints,logs,results,
  figures}/<dataset>/<modello>/...`` (vedi ``src/utils/run_paths.py``);
* il vocabolario glossa CHIUSO (Trie, reward di formato, validity) e la
  matrice di transizione, con la loro cache su disco
  (:func:`prepare_vocab_and_bigram`).

``dataset.vocab_source``
------------------------
``train`` (default, storico): il vocabolario è estratto dal SOLO train
split. ``all``: è l'UNIONE di tutti gli split (train + test, + dev per
PHOENIX). È un DATA LEAK DELIBERATO, solo per l'ablazione che misura quanto
pesa il tetto imposto dal Trie (le glosse che compaiono solo nel test sono
irraggiungibili con il vocabolario di train). Le statistiche di transizione
restano contate sul solo train anche con ``all``: cambia QUALI glosse sono
ammesse, non quanto spesso il modello le ha viste.

Con ``all`` i file di cache ricevono il suffisso ``_all`` (es.
``gloss_vocab_all.txt``) senza che il config debba ridichiarare i path: le
celle train-only e quella con leak non possono sovrascriversi a vicenda.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from datasets import DatasetDict

from .aslg_dataset import (
    download_aslg_dataset,
    extract_gloss_vocabulary,
    load_vocabulary,
    save_vocabulary,
)
from .conll_dataset import DEFAULT_CONLL_DIR, load_conll_dataset
from .phoenix_dataset import DEFAULT_PHOENIX_DIR, load_phoenix_dataset
from .transition_matrix import (
    compute_bigram_transitions,
    load_transition_matrix,
    save_transition_matrix,
)
from .wos_dataset import DEFAULT_WOS_DIR, load_wos_dataset

logger = logging.getLogger(__name__)

__all__ = [
    "ASLG_PC12",
    "CONLL_2003",
    "DATASETS",
    "DatasetSpec",
    "PHOENIX_2014T",
    "VOCAB_SOURCES",
    "WOS_46985",
    "artifact_paths",
    "cache_is_current",
    "get_dataset_spec",
    "load_t2g_dataset",
    "prepare_vocab_and_bigram",
    "resolve_vocab_source",
    "vocab_splits",
    "write_cache_meta",
]


@dataclass(frozen=True)
class DatasetSpec:
    """Static description of a supported T2G corpus.

    Attributes:
        key: Layout segment (``experiments/<kind>/<key>/<model>/...``).
        names: Accepted values of ``dataset.dataset_name`` (case-insensitive).
        display_name: Human-readable name for logs.
        prompt_profile: Default ``dataset.prompt_profile`` (see
            ``src/utils/prompting.py``).
        default_cache_dir: Default ``dataset.dataset_cache``.
        default_vocab_path: Default ``dataset.vocab_path``.
        default_bigram_path: Default ``dataset.bigram_matrix_path``.
    """

    key: str
    names: tuple[str, ...]
    display_name: str
    prompt_profile: str
    default_cache_dir: str
    default_vocab_path: str
    default_bigram_path: str


# ASLG-PC12 conserva i path storici (data/gloss_vocab.txt, ...): la cache HF in
# data/aslg_pc12 è l'unica copia offline sui nodi del cluster, e spostarla
# romperebbe ogni job (HF_HUB_OFFLINE=1, nessuna rete sui nodi di calcolo).
ASLG_PC12 = DatasetSpec(
    key="aslg-pc12",
    names=("achrafothman/aslg_pc12", "aslg-pc12", "aslg_pc12"),
    display_name="ASLG-PC12",
    prompt_profile="en-asl",
    default_cache_dir="data/aslg_pc12",
    default_vocab_path="data/gloss_vocab.txt",
    default_bigram_path="data/bigram_transition.npy",
)

PHOENIX_2014T = DatasetSpec(
    key="phoenix-2014t",
    names=("phoenix-2014t", "phoenix2014t", "phoenix_2014t", "rwth-phoenix-2014t"),
    display_name="RWTH-PHOENIX-Weather 2014T",
    prompt_profile="de-dgs",
    default_cache_dir=DEFAULT_PHOENIX_DIR,
    default_vocab_path=f"{DEFAULT_PHOENIX_DIR}/gloss_vocab.txt",
    default_bigram_path=f"{DEFAULT_PHOENIX_DIR}/bigram_transition.npy",
)

# I due dataset non-gloss di GrammarRL (Tuccio et al., arXiv:2609.39869),
# linearizzati come sequenze di token di un vocabolario chiuso: vedi
# wos_dataset.py e conll_dataset.py per il formato del target.
WOS_46985 = DatasetSpec(
    key="wos-46985",
    names=("wos-46985", "wos46985", "wos", "web-of-science", "hdltex/web_of_science"),
    display_name="Web of Science (WOS-46985)",
    prompt_profile="en-wos",
    default_cache_dir=DEFAULT_WOS_DIR,
    default_vocab_path=f"{DEFAULT_WOS_DIR}/gloss_vocab.txt",
    default_bigram_path=f"{DEFAULT_WOS_DIR}/bigram_transition.npy",
)

CONLL_2003 = DatasetSpec(
    key="conll-2003",
    names=("conll-2003", "conll2003", "conll_2003", "eriktks/conll2003"),
    display_name="CoNLL-2003 (NER)",
    prompt_profile="en-conll",
    default_cache_dir=DEFAULT_CONLL_DIR,
    default_vocab_path=f"{DEFAULT_CONLL_DIR}/gloss_vocab.txt",
    default_bigram_path=f"{DEFAULT_CONLL_DIR}/bigram_transition.npy",
)

#: Registry by layout key.
DATASETS: dict[str, DatasetSpec] = {
    s.key: s for s in (ASLG_PC12, PHOENIX_2014T, WOS_46985, CONLL_2003)
}

#: Accepted values of ``dataset.vocab_source``.
VOCAB_SOURCES: tuple[str, ...] = ("train", "all")


def get_dataset_spec(ds_cfg: Mapping[str, Any] | None) -> DatasetSpec:
    """Resolve the ``dataset`` config section to its :class:`DatasetSpec`.

    A missing ``dataset_name`` means ASLG-PC12 (historical default).

    Raises:
        ValueError: for an unknown ``dataset_name``.
    """
    name = str((ds_cfg or {}).get("dataset_name") or ASLG_PC12.names[0])
    wanted = name.strip().lower()
    for spec in DATASETS.values():
        if wanted in {n.lower() for n in spec.names}:
            return spec
    known = sorted(n for spec in DATASETS.values() for n in spec.names)
    raise ValueError(f"dataset.dataset_name sconosciuto: {name!r}. Noti: {known}")


def load_t2g_dataset(ds_cfg: Mapping[str, Any]) -> DatasetDict:
    """Load the corpus selected by ``ds_cfg`` (``train``/``test`` splits).

    Args:
        ds_cfg: The resolved ``dataset`` config section.

    Returns:
        A ``DatasetDict`` with at least ``"train"`` and ``"test"`` and columns
        ``text``/``gloss``.
    """
    spec = get_dataset_spec(ds_cfg)
    cache_dir = ds_cfg.get("dataset_cache") or spec.default_cache_dir
    if spec is PHOENIX_2014T:
        return load_phoenix_dataset(cache_dir)
    if spec is WOS_46985:
        return load_wos_dataset(
            cache_dir,
            seed=ds_cfg.get("seed", 42),
            max_source_words=ds_cfg.get("max_source_words"),
        )
    if spec is CONLL_2003:
        return load_conll_dataset(cache_dir)
    return download_aslg_dataset(cache_dir=cache_dir, seed=ds_cfg.get("seed", 42))


def resolve_vocab_source(ds_cfg: Mapping[str, Any] | None) -> str:
    """Return ``dataset.vocab_source`` (default ``"train"``), validated.

    Raises:
        ValueError: for a value outside :data:`VOCAB_SOURCES`.
    """
    source = str((ds_cfg or {}).get("vocab_source") or "train").strip().lower()
    if source not in VOCAB_SOURCES:
        raise ValueError(
            f"dataset.vocab_source={source!r} non valido: atteso uno di "
            f"{list(VOCAB_SOURCES)}"
        )
    return source


def vocab_splits(dataset: Mapping[str, Any], source: str) -> list[str]:
    """Splits whose glosses form the closed vocabulary for ``source``.

    ``train`` → ``["train"]``; ``all`` → every split of the corpus, in a fixed
    order (train, validation, test, then anything else alphabetically).
    """
    if source == "train":
        return ["train"]
    order = {"train": 0, "validation": 1, "test": 2}
    return sorted(dataset.keys(), key=lambda s: (order.get(s, 3), s))


def _with_source_suffix(path: str | Path, source: str) -> Path:
    """``data/gloss_vocab.txt`` → ``data/gloss_vocab_all.txt`` for non-train."""
    path = Path(path)
    if source == "train":
        return path
    return path.with_name(f"{path.stem}_{source}{path.suffix}")


def artifact_paths(ds_cfg: Mapping[str, Any]) -> tuple[Path, Path]:
    """``(vocab_path, bigram_path)`` effective for this config.

    The configured (or per-dataset default) paths, suffixed with
    ``_<vocab_source>`` when the source is not ``train``.
    """
    spec = get_dataset_spec(ds_cfg)
    source = resolve_vocab_source(ds_cfg)
    vocab_path = ds_cfg.get("vocab_path") or spec.default_vocab_path
    bigram_path = ds_cfg.get("bigram_matrix_path") or spec.default_bigram_path
    return (
        _with_source_suffix(vocab_path, source),
        _with_source_suffix(bigram_path, source),
    )


# ---------------------------------------------------------------------------
# Cache invalidation for vocab / bigram artifacts
# ---------------------------------------------------------------------------


def cache_meta_path(cache_path: str | Path) -> Path:
    """Sidecar JSON path for a cache file (``<stem>.meta.json``)."""
    return Path(cache_path).with_suffix(".meta.json")


def _cache_identity(ds_cfg: Mapping[str, Any] | None) -> dict[str, str]:
    """Extra sidecar keys beyond ``{seed, train_size}``.

    Only NON-default values are recorded, so the sidecars that ASLG-PC12 runs
    already wrote on the cluster (``{seed, train_size}`` only) stay valid.
    """
    if ds_cfg is None:
        return {}
    identity: dict[str, str] = {}
    spec = get_dataset_spec(ds_cfg)
    if spec is not ASLG_PC12:
        identity["dataset"] = spec.key
    source = resolve_vocab_source(ds_cfg)
    if source != "train":
        identity["vocab_source"] = source
    return identity


def write_cache_meta(
    cache_path: str | Path,
    seed: int,
    train_size: int,
    ds_cfg: Mapping[str, Any] | None = None,
) -> None:
    """Write the sidecar JSON recording what a cache file was built from."""
    meta = {"seed": seed, "train_size": train_size, **_cache_identity(ds_cfg)}
    cache_meta_path(cache_path).write_text(
        json.dumps(meta, sort_keys=True), encoding="utf-8"
    )


def cache_is_current(
    cache_path: str | Path,
    seed: int,
    train_size: int,
    ds_cfg: Mapping[str, Any] | None = None,
) -> bool:
    """Whether a cached artifact matches the current run.

    Valid only if the file exists AND its sidecar matches ``(seed,
    train_size)`` plus the dataset/vocab-source identity. Legacy caches
    WITHOUT a sidecar are never trusted and are regenerated.
    """
    path = Path(cache_path)
    meta_path = cache_meta_path(cache_path)
    if not path.exists() or not meta_path.exists():
        return False
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    if meta.get("seed") != seed or meta.get("train_size") != train_size:
        return False
    identity = _cache_identity(ds_cfg)
    for key in ("dataset", "vocab_source"):
        if meta.get(key) != identity.get(key):
            return False
    return True


def prepare_vocab_and_bigram(
    ds_cfg: Mapping[str, Any],
    dataset: DatasetDict,
    seed: int,
    *,
    use_cache_meta: bool = True,
) -> tuple[list[str], np.ndarray]:
    """Load (or build and cache) the closed gloss vocabulary and bigram matrix.

    The vocabulary comes from :func:`vocab_splits` (train only, or every
    split with ``vocab_source: all``); the bigram counts ALWAYS come from the
    train split.

    Args:
        ds_cfg: Resolved ``dataset`` config section.
        dataset: The loaded corpus (see :func:`load_t2g_dataset`).
        seed: Run seed (part of the cache identity).
        use_cache_meta: ``True`` (GRPO/eval) trusts a cache only when its
            sidecar matches; ``False`` (historical SFT behavior) trusts any
            existing file. Non-default datasets/sources always check the
            sidecar.

    Returns:
        ``(vocab, bigram_matrix)``.
    """
    vocab_path, bigram_path = artifact_paths(ds_cfg)
    source = resolve_vocab_source(ds_cfg)
    train_size = len(dataset["train"])
    check_meta = use_cache_meta or bool(_cache_identity(ds_cfg))

    def _current(path: Path) -> bool:
        if check_meta:
            return cache_is_current(path, seed, train_size, ds_cfg)
        return path.exists()

    if _current(vocab_path):
        vocab = load_vocabulary(vocab_path)
    else:
        splits: Sequence[str] = vocab_splits(dataset, source)
        if source != "train":
            logger.warning(
                "dataset.vocab_source=%s: vocabolario glossa costruito su %s "
                "— DATA LEAK deliberato (ablazione), non usare come risultato "
                "principale.",
                source,
                list(splits),
            )
        vocab = extract_gloss_vocabulary(dataset, split=splits)
        save_vocabulary(vocab, vocab_path)
        write_cache_meta(vocab_path, seed, train_size, ds_cfg)

    if _current(bigram_path):
        bigram_matrix = load_transition_matrix(bigram_path)
    else:
        bigram_matrix = compute_bigram_transitions(
            dataset, vocab, split="train", smoothing=1.0
        )
        save_transition_matrix(bigram_matrix, bigram_path)
        write_cache_meta(bigram_path, seed, train_size, ds_cfg)

    return vocab, bigram_matrix
