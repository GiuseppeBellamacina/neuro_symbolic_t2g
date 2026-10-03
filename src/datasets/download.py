"""Download automatico dei corpus che non stanno nella cache HF (PHOENIX, WOS, CoNLL).

Stessa logica di ASLG-PC12: il primo uso con rete scarica i file grezzi in
``dataset.dataset_cache`` (default ``data/<dataset>/``), gli usi successivi li
leggono da disco. Con ``HF_HUB_OFFLINE=1`` (nodi di calcolo del cluster, vedi
``cluster/_lib.sh::export_offline_env``) non si tenta nessun download: i file
devono esserci già, messi da ``cluster/setup.sh`` o da ``prepare_data``.

Le sorgenti sono tutte su Hugging Face (il proxy del cluster lascia passare
solo Hugging Face, PyPI e Kaggle), a revisioni fissate. Ogni file scaricato
è verificato contro uno sha256 fissato nel loader del suo dataset: se la
sorgente cambia contenuto il download fallisce invece di produrre in silenzio
numeri non confrontabili.

Uso da riga di comando (setup.sh)::

    python -m src.datasets.download                 # tutti
    python -m src.datasets.download conll-2003      # solo alcuni
"""

from __future__ import annotations

import hashlib
import logging
import os
import shutil
import sys
import urllib.request
from pathlib import Path

logger = logging.getLogger(__name__)

# User-Agent esplicito: alcuni server rifiutano quello di default di urllib.
_USER_AGENT = "neuro-symbolic-t2g/1.0"


def is_offline() -> bool:
    """True sui nodi senza rete (``HF_HUB_OFFLINE=1``, come per ASLG-PC12)."""
    return os.environ.get("HF_HUB_OFFLINE", "0") == "1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fetch(url: str, dest: str | Path, sha256: str) -> Path:
    """Scarica ``url`` in ``dest`` verificandone lo sha256.

    Scrive su ``<dest>.part`` e rinomina solo a verifica riuscita: un
    download interrotto o alterato non lascia mai un file valido a metà.

    Raises:
        RuntimeError: se il contenuto non corrisponde a ``sha256``.
    """
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    logger.info(f"  download {url} -> {dest}")
    request = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    with urllib.request.urlopen(request, timeout=120) as response:
        with part.open("wb") as handle:
            shutil.copyfileobj(response, handle)
    actual = _sha256(part)
    if actual != sha256:
        part.unlink()
        raise RuntimeError(
            f"{url}: sha256 {actual} diverso da quello atteso {sha256}. "
            "La sorgente è cambiata: verificare a mano prima di aggiornare l'hash."
        )
    part.replace(dest)
    return dest


def fetch_parquet_rows(url: str, sha256: str, dest_dir: str | Path) -> list[dict]:
    """Scarica un parquet, ne restituisce le righe e cancella il file.

    Il parquet è solo il formato di trasporto della copia su Hugging Face: il
    chiamante lo riscrive nel formato originale del corpus, così il loader
    legge un solo formato qualunque sia la provenienza.
    """
    import pyarrow.parquet as pq  # dipendenza di `datasets`, già installata

    path = fetch(url, Path(dest_dir) / "_download.parquet", sha256)
    try:
        return pq.read_table(path).to_pylist()
    finally:
        path.unlink()


def main(argv: list[str]) -> int:
    """Scarica i dataset richiesti (default: tutti quelli con un downloader)."""
    from .conll_dataset import DEFAULT_CONLL_DIR, ensure_conll_files
    from .phoenix_dataset import DEFAULT_PHOENIX_DIR, ensure_phoenix_files
    from .wos_dataset import DEFAULT_WOS_DIR, ensure_wos_files

    downloaders = {
        "phoenix-2014t": lambda: ensure_phoenix_files(DEFAULT_PHOENIX_DIR),
        "wos-46985": lambda: ensure_wos_files(DEFAULT_WOS_DIR),
        "conll-2003": lambda: ensure_conll_files(DEFAULT_CONLL_DIR),
    }
    unknown = [k for k in argv if k not in downloaders]
    if unknown:
        print(f"dataset sconosciuti: {unknown}. Noti: {sorted(downloaders)}")
        return 2
    failed = []
    # Un dataset che fallisce non ferma gli altri: setup.sh li vuole tutti.
    for key in argv or list(downloaders):
        try:
            downloaders[key]()
        except Exception as exc:  # noqa: BLE001 - riportato e conteggiato
            failed.append(key)
            print(f"ERRORE {key}: {exc}")
            continue
        print(f"OK {key}: file pronti")
    return 1 if failed else 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    raise SystemExit(main(sys.argv[1:]))
