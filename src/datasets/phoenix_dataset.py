"""
RWTH-PHOENIX-Weather 2014T Dataset Ingestion (text-only).

Loads the German ↔ DGS gloss annotations of PHOENIX-2014T from the official
corpus CSV files and exposes them with the SAME columns as ASLG-PC12
(``text`` = German sentence, ``gloss`` = DGS gloss sequence), so the rest of
the pipeline (vocabulary, Trie, transitions, rewards, retrieval) is reused
unchanged.

Task: Text-to-Gloss (T2G) — German weather-forecast sentences → DGS glosses.

Dove trovare i dati
-------------------
Il corpus NON è su Hugging Face Hub in forma ufficiale e la licenza
(CC BY-NC-SA 3.0) ne vieta la redistribuzione: va scaricato da RWTH
(https://www-i6.informatik.rwth-aachen.de/~koller/RWTH-PHOENIX-2014-T/).
Servono SOLO le tre annotazioni testuali (pochi MB, non i ~40 GB di video)::

    PHOENIX-2014-T.train.corpus.csv
    PHOENIX-2014-T.dev.corpus.csv
    PHOENIX-2014-T.test.corpus.csv

che nell'archivio ufficiale stanno in
``PHOENIX-2014-T/annotations/manual/``. Vanno copiate (a qualunque
profondità) sotto ``dataset.dataset_cache`` (default
``data/phoenix-2014t/``): il loader le cerca ricorsivamente per nome, quindi
si può copiare anche l'intera cartella ``annotations/``. Nessun accesso di
rete: funziona identico sui nodi di calcolo offline del cluster.

Formato: CSV separato da ``|`` con header
``name|video|start|end|speaker|orth|translation``; ``orth`` è la glossa DGS
(maiuscola, es. ``JETZT WETTER MORGEN DONNERSTAG``), ``translation`` la frase
tedesca già tokenizzata e minuscola (punteggiatura separata da spazi).

Split
-----
Si usano gli split UFFICIALI (7096 train / 519 dev / 642 test), senza dedup
né re-split: è il protocollo con cui la letteratura riporta i numeri su
PHOENIX-2014T, e uno split diverso renderebbe i risultati non confrontabili.
Il dev è esposto come ``"validation"`` ma la pipeline non lo usa (l'SFT
ritaglia il proprio holdout dal train, come su ASLG-PC12); entra solo nel
vocabolario ``dataset.vocab_source: all``.

Reference:
    Camgoz, N. C., Hadfield, S., Koller, O., Ney, H. & Bowden, R. (2018).
    Neural Sign Language Translation. CVPR 2018.
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path

from datasets import Dataset, DatasetDict

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PHOENIX_DATASET_NAME: str = "phoenix-2014t"
DEFAULT_PHOENIX_DIR: str = "data/phoenix-2014t"

#: Split della pipeline → file ufficiale del corpus.
PHOENIX_SPLIT_FILES: dict[str, str] = {
    "train": "PHOENIX-2014-T.train.corpus.csv",
    "validation": "PHOENIX-2014-T.dev.corpus.csv",
    "test": "PHOENIX-2014-T.test.corpus.csv",
}

_GLOSS_COLUMN = "orth"
_TEXT_COLUMN = "translation"


def _find_split_file(data_dir: Path, filename: str) -> Path:
    """Locate ``filename`` anywhere under ``data_dir`` (first match, sorted).

    Raises:
        FileNotFoundError: with the download/copy instructions when missing.
    """
    direct = data_dir / filename
    if direct.is_file():
        return direct
    matches = sorted(data_dir.rglob(filename)) if data_dir.is_dir() else []
    if matches:
        return matches[0]
    raise FileNotFoundError(
        f"PHOENIX-2014T: file '{filename}' non trovato sotto '{data_dir}'.\n"
        "  Scarica le annotazioni ufficiali da RWTH "
        "(https://www-i6.informatik.rwth-aachen.de/~koller/RWTH-PHOENIX-2014-T/)\n"
        "  e copia PHOENIX-2014-T.{train,dev,test}.corpus.csv "
        "(da PHOENIX-2014-T/annotations/manual/) sotto "
        f"'{data_dir}' (anche in sottocartelle)."
    )


def read_phoenix_split(path: str | Path) -> list[dict[str, str]]:
    """Read one official PHOENIX-2014T corpus CSV into ``text``/``gloss`` rows.

    Whitespace inside both fields is collapsed to single spaces; rows with an
    empty sentence or an empty gloss are dropped (logged), never repaired.

    Args:
        path: Path to a ``PHOENIX-2014-T.<split>.corpus.csv`` file.

    Returns:
        Rows with keys ``text``, ``gloss``, ``name``, ``speaker``.

    Raises:
        ValueError: if the header lacks the ``orth``/``translation`` columns.
    """
    path = Path(path)
    rows: list[dict[str, str]] = []
    dropped = 0
    with path.open(encoding="utf-8", newline="") as handle:
        # QUOTE_NONE: il corpus non usa virgolette, e le frasi possono
        # contenere `"` letterali che il dialetto di default interpreterebbe.
        reader = csv.DictReader(handle, delimiter="|", quoting=csv.QUOTE_NONE)
        fields = set(reader.fieldnames or [])
        missing = {_GLOSS_COLUMN, _TEXT_COLUMN} - fields
        if missing:
            raise ValueError(
                f"{path}: colonne mancanti {sorted(missing)} "
                f"(trovate: {sorted(fields)})"
            )
        for record in reader:
            text = " ".join(str(record.get(_TEXT_COLUMN) or "").split())
            gloss = " ".join(str(record.get(_GLOSS_COLUMN) or "").split())
            if not text or not gloss:
                dropped += 1
                continue
            rows.append(
                {
                    "text": text,
                    "gloss": gloss,
                    "name": str(record.get("name") or ""),
                    "speaker": str(record.get("speaker") or ""),
                }
            )
    if dropped:
        logger.warning(f"  {path.name}: dropped {dropped} rows with empty fields")
    return rows


def load_phoenix_dataset(data_dir: str | Path | None = None) -> DatasetDict:
    """Load PHOENIX-2014T (official splits) as a ``DatasetDict``.

    Args:
        data_dir: Directory containing (at any depth) the three official
            corpus CSV files. Defaults to ``data/phoenix-2014t``.

    Returns:
        ``DatasetDict`` with ``"train"``, ``"validation"`` and ``"test"``
        splits and columns ``text``, ``gloss``, ``name``, ``speaker``.

    Raises:
        FileNotFoundError: if any of the three CSV files is missing.
    """
    root = Path(data_dir or DEFAULT_PHOENIX_DIR)
    logger.info(f"Loading PHOENIX-2014T annotations from '{root}' (official splits)")
    splits: dict[str, Dataset] = {}
    for split_name, filename in PHOENIX_SPLIT_FILES.items():
        rows = read_phoenix_split(_find_split_file(root, filename))
        splits[split_name] = Dataset.from_list(rows)
        logger.info(f"  {split_name}: {len(rows)} samples")
    return DatasetDict(splits)
