"""
CoNLL-2003 (inglese) — named entity recognition come sequenza vincolata.

Uno dei tre dataset di GrammarRL (Tuccio et al., arXiv:2609.39869), dove il
modello genera un oggetto JSON con i campi person / organization /
location / miscellaneous, vincolati a vocabolari specifici del task. Qui il
target è LINEARIZZATO nella stessa forma delle glosse::

    text  = "EU rejects German call to boycott British lamb ."
    gloss = "ORG:EU MISC:German MISC:British"

Ogni entità è UN token ``TIPO:Parole_unite_da_underscore``; i token seguono
l'ordine dei campi del JSON (PER, ORG, LOC, MISC) e, dentro un campo, l'ordine
di apparizione, senza duplicati (il campo è un insieme di entità). Una frase
senza entità ha target ``NONE`` (una riga vuota verrebbe scartata dalla
pipeline). Così il vocabolario chiuso del Trie è l'insieme delle entità
tipizzate viste nel train (+ ``NONE``), l'exact match è "tutte le entità
giuste" e il gloss F1 a token coincide con l'F1 a livello di entità.

Con ``dataset.vocab_source: all`` il Trie ammette anche le entità che
compaiono solo nel test: è qui che il leak pesa di più, perché in NER la
maggior parte delle entità di test non è mai vista in train.

Dove trovare i dati
-------------------
I file nel formato a colonne originale (``parola POS chunk NER``, frasi
separate da righe vuote, righe ``-DOCSTART-`` ignorate), con uno di questi
nomi, a qualunque profondità sotto ``dataset.dataset_cache`` (default
``data/conll-2003/``)::

    train      : eng.train  | train.txt
    validation : eng.testa  | valid.txt | dev.txt
    test       : eng.testb  | test.txt

Se mancano vengono scaricati (:data:`CONLL_URL`, lo stesso archivio usato dal
loader Hugging Face ``eriktks/conll2003``: ``train.txt``/``valid.txt``/
``test.txt``, IOB2) e verificati con sha256; con ``HF_HUB_OFFLINE=1`` niente
download (sul cluster li scarica ``cluster/setup.sh``).

Il tagging può essere IOB1 (originale) o IOB2: un'entità comincia a ``B-``,
a un cambio di tipo o dopo ``O``. Split UFFICIALI (14.041 / 3.250 / 3.453
frasi), nessun dedup; il validation entra solo nel vocabolario ``all``.
"""

from __future__ import annotations

import logging
from pathlib import Path

from datasets import Dataset, DatasetDict

from .download import fetch_zip_members, is_offline

logger = logging.getLogger(__name__)

CONLL_DATASET_NAME: str = "conll-2003"
DEFAULT_CONLL_DIR: str = "data/conll-2003"

#: Split della pipeline → nomi di file accettati (il primo trovato vince).
CONLL_SPLIT_FILES: dict[str, tuple[str, ...]] = {
    "train": ("eng.train", "train.txt"),
    "validation": ("eng.testa", "valid.txt", "dev.txt"),
    "test": ("eng.testb", "test.txt"),
}

#: Archivio CoNLL-2003 del loader HF ``eriktks/conll2003`` e suo sha256.
CONLL_URL: str = "https://data.deepai.org/conll2003.zip"
CONLL_SHA256: str = "96a104d174ddae7558bab603f19382c5fe02ff1da5c077a7f3ce2ced1578a2c3"
_CONLL_ZIP_MEMBERS: dict[str, str] = {
    "train.txt": "train.txt",
    "valid.txt": "valid.txt",
    "test.txt": "test.txt",
}

#: Ordine dei campi del JSON di GrammarRL (person, organization, location, misc).
ENTITY_TYPES: tuple[str, ...] = ("PER", "ORG", "LOC", "MISC")

#: Target di una frase senza entità.
NO_ENTITY_TOKEN: str = "NONE"


def extract_entities(tokens: list[str], tags: list[str]) -> list[tuple[str, str]]:
    """Entità ``(tipo, testo)`` da una frase taggata IOB1 o IOB2."""
    entities: list[tuple[str, str]] = []
    current_type: str | None = None
    current: list[str] = []

    def flush() -> None:
        nonlocal current_type, current
        if current_type is not None and current:
            entities.append((current_type, " ".join(current)))
        current_type, current = None, []

    for token, tag in zip(tokens, tags):
        if tag == "O" or "-" not in tag:
            flush()
            continue
        prefix, entity_type = tag.split("-", 1)
        if prefix == "B" or entity_type != current_type:
            flush()
            current_type = entity_type
        current.append(token)
    flush()
    return entities


def linearize_entities(entities: list[tuple[str, str]]) -> str:
    """``[("ORG", "EU"), ("MISC", "German")]`` → ``"ORG:EU MISC:German"``."""
    ordered: list[str] = []
    known = set(ENTITY_TYPES)
    types = list(ENTITY_TYPES) + sorted({t for t, _ in entities} - known)
    for entity_type in types:
        for t, text in entities:
            if t != entity_type:
                continue
            token = f"{entity_type}:{'_'.join(text.split())}"
            if token not in ordered:
                ordered.append(token)
    return " ".join(ordered) if ordered else NO_ENTITY_TOKEN


def read_conll_file(path: str | Path) -> list[dict[str, str]]:
    """Read one CoNLL-2003 file into ``text``/``gloss`` rows (one per sentence)."""
    rows: list[dict[str, str]] = []
    tokens: list[str] = []
    tags: list[str] = []

    def emit() -> None:
        if tokens:
            rows.append(
                {
                    "text": " ".join(tokens),
                    "gloss": linearize_entities(extract_entities(tokens, tags)),
                }
            )
        tokens.clear()
        tags.clear()

    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            parts = line.split()
            if not parts:
                emit()
                continue
            if parts[0] == "-DOCSTART-":
                emit()
                continue
            tokens.append(parts[0])
            tags.append(parts[-1])
    emit()
    return rows


def _find_split_file(data_dir: Path, split: str) -> Path | None:
    for filename in CONLL_SPLIT_FILES[split]:
        direct = data_dir / filename
        if direct.is_file():
            return direct
        matches = sorted(data_dir.rglob(filename)) if data_dir.is_dir() else []
        if matches:
            return matches[0]
    return None


def ensure_conll_files(data_dir: str | Path) -> dict[str, Path]:
    """Path dei file per split, scaricandoli se ne manca uno (non offline).

    Raises:
        FileNotFoundError: se mancano e il nodo è offline.
    """
    root = Path(data_dir)
    if any(_find_split_file(root, split) is None for split in CONLL_SPLIT_FILES):
        if is_offline():
            raise FileNotFoundError(
                f"CoNLL-2003: file mancanti sotto '{root}' e nodo offline "
                "(HF_HUB_OFFLINE=1). Eseguire cluster/setup.sh (scarica i file) "
                "o copiare eng.train/eng.testa/eng.testb (o train/valid/test.txt)."
            )
        root.mkdir(parents=True, exist_ok=True)
        fetch_zip_members(CONLL_URL, CONLL_SHA256, _CONLL_ZIP_MEMBERS, root)
    return {split: _find_split_file(root, split) for split in CONLL_SPLIT_FILES}  # type: ignore[misc]


def load_conll_dataset(data_dir: str | Path | None = None) -> DatasetDict:
    """Load CoNLL-2003 (official splits) as ``text``/``gloss``.

    Returns:
        ``DatasetDict`` with ``"train"``, ``"validation"``, ``"test"``.
    """
    root = Path(data_dir or DEFAULT_CONLL_DIR)
    logger.info(f"Loading CoNLL-2003 from '{root}' (official splits)")
    splits: dict[str, Dataset] = {}
    for split, path in ensure_conll_files(root).items():
        rows = read_conll_file(path)
        splits[split] = Dataset.from_list(rows)
        logger.info(f"  {split}: {len(rows)} sentences")
    return DatasetDict(splits)
