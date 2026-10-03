"""
Web of Science (WOS-46985) — classificazione gerarchica come sequenza vincolata.

Uno dei tre dataset di GrammarRL (Tuccio et al., arXiv:2609.39869), dove il
modello genera un oggetto JSON ``{domain, area}`` con entrambi i campi
vincolati al proprio vocabolario chiuso. Qui il target è LINEARIZZATO nella
stessa forma delle glosse, così il Trie dual-root, le reward e le metriche
della pipeline si applicano senza modifiche::

    text  = abstract dell'articolo (eventualmente troncato, vedi sotto)
    gloss = "<DOMINIO> <AREA>"       es. "CS Machine_learning"

Le etichette sono rese token singoli (spazi interni → ``_``, spazi ai bordi
rimossi: nel file originale ``"CS "`` e ``" Machine learning"``).

Etichette canoniche
-------------------
La classe ufficiale del benchmark è la colonna ``Y`` (134 classi), ma le
colonne testuali ``Domain``/``area`` non le corrispondono sempre: lo stesso
codice compare con nomi d'area diversi (``Electric motor`` / ``Satellite
radio``) e la stessa area con maiuscole diverse in domini diversi
(``Medical Depression`` contro ``Psychology depression``, entrambe ``Y=40``).
Presi alla lettera, i nomi darebbero 145 classi, alcune con 1-14 esempi, e
token che differiscono solo per le maiuscole. Ogni riga riceve quindi la
coppia (dominio, area) PIÙ FREQUENTE del suo ``Y`` (a parità, la prima in
ordine alfabetico): 134 classi come il benchmark; 447 righe (1%) cambiano nome.
Senza colonna ``Y`` (un ``Data.csv`` ridotto) restano i nomi del file.
I nomi canonici dipendono dai nomi presenti nella sorgente: con la copia
Hugging Face scaricata in automatico (vedi sotto) la classe ``Y=25`` si chiama
``ECE Electrical_generator``, con il ``Data.xlsx`` di Mendeley
``ECE Analog_signal_processing``. Stessi testi, stessi codici e stesse
classi: cambia solo la stringa di quel token. L'ordine
dominio → area codifica la gerarchia; il Trie vincola i token al vocabolario
ma NON la struttura (due aree di fila restano generabili): exact match = la
coppia (dominio, area) è corretta.

Dove trovare i dati
-------------------
Kowsari et al. (2017), "HDLTex: Hierarchical Deep Learning for Text
Classification", Mendeley Data (doi:10.17632/9rw3vkcfy4.6). Dall'archivio
``WebOfScience.zip`` serve ``Meta-data/Data.xlsx`` (46.985 righe, colonne
``Y1, Y2, Y, Domain, area, keywords, Abstract``), cercato sotto
``dataset.dataset_cache`` (default ``data/wos-46985/``, a qualunque
profondità; in alternativa un ``Data.csv`` con le colonne ``Y, Domain,
area, Abstract``). Mendeley non è raggiungibile dal cluster, quindi se manca
viene scaricata la copia parquet su Hugging Face :data:`WOS_HF` (revisione
fissata, sha256) e riscritta come ``Data.csv``. Verificata contro ``Data.xlsx``:
stessi 46.985 abstract nello stesso ordine e stesso codice ``Y`` in ogni riga
(nel parquet ``label`` è un vettore a 141 posizioni: dominio ``Y1`` in 0-6,
area in ``7 + Y``). Con ``HF_HUB_OFFLINE=1`` niente download (sul cluster lo
scarica ``cluster/setup.sh``). L'xlsx è
letto con la sola libreria standard (niente ``openpyxl``, assente nel
container del cluster).

Split
-----
Il dataset non ha uno split ufficiale e quello di GrammarRL non è pubblico:
si applica la STESSA logica di ASLG-PC12 (dedup per testo normalizzato, poi
90/10 deterministico con ``seed``). I numeri non sono quindi confrontabili
uno-a-uno con quelli del paper.

Lunghezza del source
--------------------
Gli abstract sono lunghi (centinaia di parole): ``max_source_words`` tronca
il testo alle prime N parole DOPO lo split (il dedup usa il testo intero),
così il prompt zero-shot resta sotto ``grpo.max_prompt_length``. Con il
few-shot tre abstract non ci starebbero comunque: le celle WoS sono solo
zero-shot.
"""

from __future__ import annotations

import csv
import logging
import re
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from xml.etree import ElementTree as ET

from datasets import Dataset, DatasetDict

from .aslg_dataset import deduplicate_by_text
from .download import fetch_parquet_rows, is_offline

logger = logging.getLogger(__name__)

WOS_DATASET_NAME: str = "wos-46985"
DEFAULT_WOS_DIR: str = "data/wos-46985"

_DATA_FILES: tuple[str, ...] = ("Data.xlsx", "Data.csv")

#: Copia parquet di WOS-46985 su Hugging Face, a una revisione fissata, e sha256.
WOS_HF: str = (
    "https://huggingface.co/datasets/jesse-tong/wos46985/resolve/"
    "30fd4f04782ba43cab5d3e8cf6a84b23358481cf/wos46895.parquet"
)
WOS_HF_SHA256: str = "b515250891046916253e9f144d4c391bf7951e2baaabb82c522fca048752a30e"

#: Posizioni del vettore ``label`` del parquet prima delle aree (i 7 domini).
_WOS_N_DOMAINS = 7

_XLSX_NS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
_CELL_REF_RE = re.compile(r"([A-Z]+)")


def label_token(label: str) -> str:
    """Rende un'etichetta WoS un token singolo (``" Machine learning"`` → ``Machine_learning``)."""
    return "_".join(str(label).split())


def _column_index(ref: str) -> int:
    """``"C12"`` → 2 (indice 0-based della colonna)."""
    letters = _CELL_REF_RE.match(ref)
    if not letters:
        raise ValueError(f"riferimento di cella non valido: {ref!r}")
    index = 0
    for ch in letters.group(1):
        index = index * 26 + (ord(ch) - ord("A") + 1)
    return index - 1


def read_xlsx_rows(path: str | Path) -> list[dict[str, str]]:
    """Legge il PRIMO foglio di un .xlsx come lista di dict (header = riga 1).

    Lettore minimale con la sola stdlib (zip + XML): stringhe condivise,
    stringhe inline e valori semplici; nessuna formula viene valutata (si
    legge il valore in cache, come fa qualunque lettore).
    """
    path = Path(path)
    with zipfile.ZipFile(path) as zf:
        names = set(zf.namelist())
        shared: list[str] = []
        if "xl/sharedStrings.xml" in names:
            root = ET.fromstring(zf.read("xl/sharedStrings.xml"))
            for si in root.iter(f"{_XLSX_NS}si"):
                # Solo i run di testo (t diretti o dentro r), non la fonetica (rPh).
                parts = [t.text or "" for t in si.findall(f"{_XLSX_NS}t")]
                parts += [
                    t.text or ""
                    for r in si.findall(f"{_XLSX_NS}r")
                    for t in r.findall(f"{_XLSX_NS}t")
                ]
                shared.append("".join(parts))
        sheets = sorted(
            n
            for n in names
            if n.startswith("xl/worksheets/sheet") and n.endswith(".xml")
        )
        if not sheets:
            raise ValueError(f"{path}: nessun foglio di lavoro trovato")
        sheet = (
            "xl/worksheets/sheet1.xml"
            if "xl/worksheets/sheet1.xml" in names
            else sheets[0]
        )

        raw_rows: list[dict[int, str]] = []
        with zf.open(sheet) as handle:
            for _event, elem in ET.iterparse(handle):
                if elem.tag != f"{_XLSX_NS}row":
                    continue
                values: dict[int, str] = {}
                for cell in elem.findall(f"{_XLSX_NS}c"):
                    kind = cell.get("t")
                    if kind == "inlineStr":
                        text = "".join(t.text or "" for t in cell.iter(f"{_XLSX_NS}t"))
                    else:
                        v = cell.find(f"{_XLSX_NS}v")
                        text = (v.text or "") if v is not None else ""
                        if kind == "s" and text:
                            text = shared[int(text)]
                    values[_column_index(cell.get("r", "A1"))] = text
                raw_rows.append(values)
                elem.clear()

    if not raw_rows:
        return []
    header = {i: name.strip() for i, name in raw_rows[0].items()}
    return [
        {name: row.get(i, "") for i, name in header.items()} for row in raw_rows[1:]
    ]


def _read_rows(path: Path) -> list[dict[str, str]]:
    if path.suffix.lower() == ".xlsx":
        return read_xlsx_rows(path)
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _find_data_file(data_dir: Path) -> Path | None:
    for filename in _DATA_FILES:
        direct = data_dir / filename
        if direct.is_file():
            return direct
        matches = sorted(data_dir.rglob(filename)) if data_dir.is_dir() else []
        if matches:
            return matches[0]
    return None


def ensure_wos_files(data_dir: str | Path) -> Path:
    """Path di ``Data.xlsx``/``Data.csv``, scaricandolo se manca (non offline).

    Raises:
        FileNotFoundError: se manca e il nodo è offline.
    """
    root = Path(data_dir)
    path = _find_data_file(root)
    if path is not None:
        return path
    if is_offline():
        raise FileNotFoundError(
            f"WOS-46985: né Data.xlsx né Data.csv sotto '{root}' e nodo offline "
            "(HF_HUB_OFFLINE=1). Eseguire cluster/setup.sh (scarica il file) o "
            "copiare Meta-data/Data.xlsx di WebOfScience.zip (Mendeley)."
        )
    root.mkdir(parents=True, exist_ok=True)
    rows = fetch_parquet_rows(WOS_HF, WOS_HF_SHA256, root)
    path = root / "Data.csv"
    # .part + rename: un Data.csv a metà verrebbe preso per buono al job dopo.
    part = root / "Data.csv.part"
    with part.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["Y", "Domain", "area", "Abstract"])
        writer.writeheader()
        for row in rows:
            ones = [i for i, v in enumerate(row["label"]) if v == 1.0]
            domain, area = row["label_description"]
            writer.writerow(
                {
                    "Y": ones[-1] - _WOS_N_DOMAINS,
                    "Domain": domain,
                    "area": area,
                    "Abstract": row["text"],
                }
            )
    part.replace(path)
    return path


def canonical_labels(records: list[dict[str, str]]) -> dict[str, tuple[str, str]]:
    """Codice ``Y`` → coppia (dominio, area) più frequente fra le sue righe.

    A parità di frequenza vince la coppia prima in ordine alfabetico, così il
    risultato non dipende dall'ordine delle righe.
    """
    counts: dict[str, Counter[tuple[str, str]]] = defaultdict(Counter)
    for record in records:
        code = str(record.get("Y") or "").strip()
        if code:
            pair = (
                label_token(record.get("Domain") or ""),
                label_token(record.get("area") or ""),
            )
            counts[code][pair] += 1
    return {
        code: min(c.items(), key=lambda kv: (-kv[1], kv[0]))[0]
        for code, c in counts.items()
    }


def load_wos_dataset(
    data_dir: str | Path | None = None,
    seed: int = 42,
    max_source_words: int | None = None,
) -> DatasetDict:
    """Load WOS-46985 as ``text`` (abstract) / ``gloss`` (``DOMINIO AREA``).

    Args:
        data_dir: Directory containing ``Data.xlsx`` (or ``Data.csv``),
            downloaded there if missing.
        seed: Seed of the deterministic 90/10 split (same logic as ASLG-PC12).
        max_source_words: Keep only the first N words of each abstract
            (``None`` = full text). Applied AFTER dedup and split.

    Returns:
        ``DatasetDict`` with ``"train"``/``"test"`` and columns ``text``,
        ``gloss``, ``domain``, ``area``.
    """
    root = Path(data_dir or DEFAULT_WOS_DIR)
    path = ensure_wos_files(root)
    logger.info(f"Loading WOS-46985 from '{path}'")

    records = _read_rows(path)
    canonical = canonical_labels(records)
    rows: list[dict[str, str]] = []
    dropped = 0
    for record in records:
        text = " ".join(str(record.get("Abstract") or "").split())
        code = str(record.get("Y") or "").strip()
        domain, area = canonical.get(
            code,
            (
                label_token(record.get("Domain") or ""),
                label_token(record.get("area") or ""),
            ),
        )
        if not text or not domain or not area:
            dropped += 1
            continue
        rows.append(
            {"text": text, "gloss": f"{domain} {area}", "domain": domain, "area": area}
        )
    if dropped:
        logger.warning(f"  dropped {dropped} rows with empty abstract/domain/area")
    if not rows:
        raise ValueError(f"{path}: nessuna riga valida (colonne Domain/area/Abstract?)")

    dedup, _ = deduplicate_by_text(Dataset.from_list(rows))
    split = dedup.train_test_split(test_size=0.1, seed=seed)
    ds = DatasetDict({"train": split["train"], "test": split["test"]})

    if max_source_words:
        limit = int(max_source_words)

        def _truncate(sample: dict[str, str]) -> dict[str, str]:
            return {"text": " ".join(sample["text"].split()[:limit])}

        ds = DatasetDict({name: d.map(_truncate) for name, d in ds.items()})

    for split_name, split_ds in ds.items():
        logger.info(f"  {split_name}: {len(split_ds)} samples")
    return ds
