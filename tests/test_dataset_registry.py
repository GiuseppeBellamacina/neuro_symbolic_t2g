"""Multi-dataset support: registry, PHOENIX-2014T loader, ``vocab_source``.

No network: PHOENIX-2014T is read from tiny CSV fixtures in the official
format, and ASLG-PC12 is replaced by an in-memory ``DatasetDict``.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from datasets import Dataset, DatasetDict
from src.datasets.registry import (
    ASLG_PC12,
    DATASETS,
    PHOENIX_2014T,
    artifact_paths,
    cache_is_current,
    get_dataset_spec,
    load_t2g_dataset,
    prepare_vocab_and_bigram,
    resolve_vocab_source,
    vocab_splits,
)
from src.utils import run_paths
from src.utils.config import resolve_config

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "experiments" / "configs"

_HEADER = "name|video|start|end|speaker|orth|translation"


def _write_phoenix(root: Path, rows: dict[str, list[tuple[str, str]]]) -> None:
    """Official-format CSVs, nested like the RWTH archive."""
    ann = root / "PHOENIX-2014-T" / "annotations" / "manual"
    ann.mkdir(parents=True)
    for split, file_split in (("train", "train"), ("dev", "dev"), ("test", "test")):
        lines = [_HEADER]
        for i, (gloss, text) in enumerate(rows[split]):
            lines.append(f"s{i}|s{i}/1/*.png|-1|-1|Signer01|{gloss}|{text}")
        (ann / f"PHOENIX-2014-T.{file_split}.corpus.csv").write_text(
            "\n".join(lines) + "\n", encoding="utf-8"
        )


@pytest.fixture
def phoenix_dir(tmp_path: Path) -> Path:
    root = tmp_path / "phoenix-2014t"
    _write_phoenix(
        root,
        {
            "train": [
                ("JETZT WETTER MORGEN", "und nun die wettervorhersage für morgen ."),
                ("REGEN  NORD", 'im norden regen "stark" .'),
                ("", "riga senza glossa"),
            ],
            "dev": [("SONNE SUED", "im süden sonne .")],
            "test": [("SCHNEE BERG", "in den bergen schnee .")],
        },
    )
    return root


# ── Registry ────────────────────────────────────────────────────────────────


def test_dataset_keys_mirror_run_paths() -> None:
    """run_paths duplica le chiavi per restare import-light: devono coincidere."""
    assert tuple(DATASETS) == run_paths.DATASET_KEYS
    assert run_paths.DEFAULT_DATASET_KEY == ASLG_PC12.key


def test_get_dataset_spec_by_name() -> None:
    assert get_dataset_spec({}) is ASLG_PC12
    assert get_dataset_spec({"dataset_name": "achrafothman/aslg_pc12"}) is ASLG_PC12
    assert get_dataset_spec({"dataset_name": "PHOENIX-2014T"}) is PHOENIX_2014T
    with pytest.raises(ValueError, match="sconosciuto"):
        get_dataset_spec({"dataset_name": "how2sign"})


def test_vocab_source_validation() -> None:
    assert resolve_vocab_source({}) == "train"
    assert resolve_vocab_source({"vocab_source": "ALL"}) == "all"
    with pytest.raises(ValueError):
        resolve_vocab_source({"vocab_source": "test"})


def test_vocab_splits_order() -> None:
    ds = {"test": 0, "validation": 0, "train": 0}
    assert vocab_splits(ds, "train") == ["train"]
    assert vocab_splits(ds, "all") == ["train", "validation", "test"]


def test_artifact_paths_suffix_only_for_non_train() -> None:
    """Le celle con leak non possono sovrascrivere la cache train-only."""
    base = {
        "vocab_path": "data/x/gloss_vocab.txt",
        "bigram_matrix_path": "data/x/b.npy",
    }
    assert artifact_paths(base) == (
        Path("data/x/gloss_vocab.txt"),
        Path("data/x/b.npy"),
    )
    assert artifact_paths({**base, "vocab_source": "all"}) == (
        Path("data/x/gloss_vocab_all.txt"),
        Path("data/x/b_all.npy"),
    )
    # Default per dataset quando il config non dichiara i path.
    assert artifact_paths({}) == (
        Path("data/gloss_vocab.txt"),
        Path("data/bigram_transition.npy"),
    )
    assert artifact_paths({"dataset_name": "phoenix-2014t"})[0] == Path(
        "data/phoenix-2014t/gloss_vocab.txt"
    )


# ── PHOENIX-2014T loader ────────────────────────────────────────────────────


def test_phoenix_loader_official_splits(phoenix_dir: Path) -> None:
    ds = load_t2g_dataset(
        {"dataset_name": "phoenix-2014t", "dataset_cache": phoenix_dir}
    )
    assert set(ds) == {"train", "validation", "test"}
    # La riga con glossa vuota e' scartata, mai riparata.
    assert len(ds["train"]) == 2
    first = ds["train"][0]
    assert first["text"] == "und nun die wettervorhersage für morgen ."
    assert first["gloss"] == "JETZT WETTER MORGEN"
    # Spazi multipli collassati; virgolette letterali preservate (QUOTE_NONE).
    assert ds["train"][1]["gloss"] == "REGEN NORD"
    assert ds["train"][1]["text"] == 'im norden regen "stark" .'


def test_phoenix_loader_missing_files_explains(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")  # offline: errore, niente download
    with pytest.raises(FileNotFoundError, match="PHOENIX-2014-T.train.corpus.csv"):
        load_t2g_dataset({"dataset_name": "phoenix-2014t", "dataset_cache": tmp_path})


# ── Vocabulary / bigram cache ───────────────────────────────────────────────


def _toy_aslg() -> DatasetDict:
    return DatasetDict(
        {
            "train": Dataset.from_list(
                [{"text": "a b", "gloss": "A B"}, {"text": "b c", "gloss": "B C"}]
            ),
            "test": Dataset.from_list([{"text": "d", "gloss": "D A"}]),
        }
    )


def test_prepare_vocab_train_vs_all(tmp_path: Path) -> None:
    ds = _toy_aslg()
    cfg = {
        "vocab_path": str(tmp_path / "gloss_vocab.txt"),
        "bigram_matrix_path": str(tmp_path / "bigram.npy"),
    }
    vocab_train, bigram_train = prepare_vocab_and_bigram(cfg, ds, seed=42)
    assert "D" not in vocab_train

    cfg_all = {**cfg, "vocab_source": "all"}
    vocab_all, bigram_all = prepare_vocab_and_bigram(cfg_all, ds, seed=42)
    assert set(vocab_all) == set(vocab_train) | {"D"}
    assert bigram_all.shape == (len(vocab_all), len(vocab_all))
    # Le transizioni restano contate sul SOLO train: D (solo nel test) ha la
    # riga uniforme del solo smoothing.
    row = bigram_all[vocab_all.index("D")]
    assert np.allclose(row, row[0])

    # File separati: la cache train-only resta intatta e valida.
    assert (tmp_path / "gloss_vocab_all.txt").exists()
    assert (tmp_path / "bigram_all.npy").exists()
    assert cache_is_current(tmp_path / "gloss_vocab.txt", 42, 2, cfg)
    assert not cache_is_current(tmp_path / "gloss_vocab_all.txt", 42, 2, cfg)
    assert cache_is_current(tmp_path / "gloss_vocab_all.txt", 42, 2, cfg_all)


def test_legacy_aslg_sidecar_stays_valid(tmp_path: Path) -> None:
    """I sidecar gia' sul cluster ({seed, train_size}) non vanno invalidati."""
    vocab = tmp_path / "gloss_vocab.txt"
    vocab.write_text("A", encoding="utf-8")
    vocab.with_suffix(".meta.json").write_text(
        json.dumps({"seed": 42, "train_size": 10}), encoding="utf-8"
    )
    assert cache_is_current(vocab, 42, 10, {"dataset_name": "achrafothman/aslg_pc12"})
    # ...ma non per un altro dataset con la stessa dimensione del train.
    assert not cache_is_current(vocab, 42, 10, {"dataset_name": "phoenix-2014t"})


# ── Config trees ────────────────────────────────────────────────────────────


#: Dataset non-default → (profilo di prompt atteso).
_OTHER_DATASETS = {
    "phoenix-2014t": "de-dgs",
    "wos-46985": "en-wos",
    "conll-2003": "en-conll",
}


@pytest.mark.parametrize(
    "dataset,rel",
    sorted(
        (dataset, str(p.relative_to(CONFIGS / dataset)))
        for dataset in _OTHER_DATASETS
        for p in (CONFIGS / dataset).rglob("*.yaml")
    ),
)
def test_cells_resolve_to_their_dataset_everywhere(dataset: str, rel: str) -> None:
    """Ogni cella usa dataset, prompt, cache e output del PROPRIO dataset."""
    cfg = resolve_config(str(CONFIGS / dataset / rel))
    assert get_dataset_spec(cfg["dataset"]) is DATASETS[dataset]
    assert cfg["dataset"]["prompt_profile"] == _OTHER_DATASETS[dataset]
    for key in ("dataset_cache", "vocab_path", "bigram_matrix_path"):
        assert cfg["dataset"][key].startswith(f"data/{dataset}"), key
    assert cfg["retrieval"]["cache_path"].startswith(f"data/{dataset}")
    for key in ("output_dir", "log_dir"):
        if key in cfg.get("training", {}):
            assert f"/{dataset}/qwen25-05b/" in cfg["training"][key], key
    assert cfg["wandb"]["run_name"].startswith(f"{dataset}-")


@pytest.mark.parametrize("dataset", ["aslg-pc12", *_OTHER_DATASETS])
def test_full_vocab_trie_is_single_factor_ablation(dataset: str) -> None:
    """full-vocab-trie = la sua cella grpo genitrice + vocab_source: all."""
    import yaml

    root = CONFIGS / dataset / "qwen25-05b"
    leak_path = root / "ablations/decoding/full-vocab-trie.yaml"
    parent = yaml.safe_load(leak_path.read_text(encoding="utf-8"))["extends"]
    assert parent in ("../../grpo/few-shot.yaml", "../../grpo/zero-shot.yaml")
    leak = resolve_config(str(leak_path))
    ref = resolve_config(str((leak_path.parent / parent).resolve()))
    assert leak["dataset"].pop("vocab_source") == "all"
    assert "vocab_source" not in ref["dataset"]
    for section in (
        "model",
        "lora",
        "dataset",
        "grpo",
        "reward",
        "grammar",
        "retrieval",
    ):
        assert leak[section] == ref[section], section
    assert leak["training"]["output_dir"].endswith("ablations/decoding/full-vocab-trie")


def test_cells_do_not_declare_vocab_source_except_leak() -> None:
    """La chiave entra nel fingerprint SFT: solo le celle con leak la dichiarano."""
    for path in CONFIGS.rglob("*.yaml"):
        cfg = resolve_config(str(path))
        if path.name != "full-vocab-trie.yaml":
            assert "vocab_source" not in cfg["dataset"], path


# ── Tag: Python, _lib.sh e cluster_helper.sh concordano ─────────────────────


@pytest.mark.skipif(shutil.which("bash") is None, reason="bash non disponibile")
@pytest.mark.skipif(shutil.which("bash") is None, reason="bash non disponibile")
def test_bash_tag_matches_python_cell_tag_for_every_config() -> None:
    # Percorso completo: su Windows un "bash" nudo fa trovare a CreateProcess
    # prima il bash di WSL in System32, anche quando nel PATH c'e' Git Bash.
    configs = sorted(
        p.relative_to(ROOT).as_posix()
        for p in CONFIGS.rglob("*.yaml")
        if p.name != "base.yaml"
    )
    script = f"source '{(ROOT / 'cluster' / '_lib.sh').as_posix()}'\n" + "".join(
        f"t2g_tag_from_config '{c}'\n" for c in configs
    )
    out = subprocess.run(
        [shutil.which("bash"), "-c", script], capture_output=True, text=True, check=True
    ).stdout.split()
    expected = [
        run_paths.cell_tag(run_paths.cell_from_config(ROOT / c)) for c in configs
    ]
    assert out == expected
    assert len(set(out)) == len(out), "due celle con lo stesso tag"


# ── WOS-46985 ───────────────────────────────────────────────────────────────


def _write_xlsx(path: Path, rows: list[list[str]]) -> None:
    """Minimal .xlsx (shared strings + one sheet), as Excel writes it."""
    import zipfile

    strings: list[str] = []
    index: dict[str, int] = {}

    def sid(value: str) -> int:
        if value not in index:
            index[value] = len(strings)
            strings.append(value)
        return index[value]

    def col(i: int) -> str:
        name = ""
        i += 1
        while i:
            i, rem = divmod(i - 1, 26)
            name = chr(65 + rem) + name
        return name

    ns = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
    sheet_rows = []
    for r, row in enumerate(rows, start=1):
        cells = "".join(
            f'<c r="{col(c)}{r}" t="s"><v>{sid(v)}</v></c>' for c, v in enumerate(row)
        )
        sheet_rows.append(f'<row r="{r}">{cells}</row>')
    sst = "".join(f'<si><t xml:space="preserve">{s}</t></si>' for s in strings)
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("xl/sharedStrings.xml", f'<sst xmlns="{ns}">{sst}</sst>')
        zf.writestr(
            "xl/worksheets/sheet1.xml",
            f'<worksheet xmlns="{ns}"><sheetData>{"".join(sheet_rows)}</sheetData></worksheet>',
        )


_WOS_HEADER = ["Y1", "Y2", "Y", "Domain", "area", "keywords", "Abstract"]


def _wos_rows(n: int) -> list[list[str]]:
    domains = [("CS ", " Machine learning"), ("Medical  ", "Alzheimer's Disease")]
    return [
        ["0", "0", "0", *domains[i % 2], "kw", f"abstract number {i} " + "word " * 10]
        for i in range(n)
    ]


def test_wos_xlsx_loader(tmp_path: Path) -> None:
    from src.datasets.wos_dataset import label_token, read_xlsx_rows

    root = tmp_path / "wos" / "Meta-data"
    root.mkdir(parents=True)
    rows = _wos_rows(20)
    # Un duplicato esatto e una riga senza abstract: entrambi scartati.
    rows.append(list(rows[0]))
    rows.append(["0", "0", "0", "CS", "Machine learning", "kw", ""])
    _write_xlsx(root / "Data.xlsx", [_WOS_HEADER, *rows])

    assert read_xlsx_rows(root / "Data.xlsx")[0]["Domain"] == "CS "
    assert label_token(" Machine   learning ") == "Machine_learning"

    ds = load_t2g_dataset(
        {
            "dataset_name": "wos-46985",
            "dataset_cache": str(tmp_path / "wos"),
            "seed": 42,
        }
    )
    assert set(ds) == {"train", "test"}
    assert len(ds["train"]) + len(ds["test"]) == 20
    assert len(ds["test"]) == 2
    glosses = {r["gloss"] for r in ds["train"]}
    assert glosses <= {"CS Machine_learning", "Medical Alzheimer's_Disease"}


def test_wos_csv_and_truncation(tmp_path: Path) -> None:
    import csv

    root = tmp_path / "wos"
    root.mkdir()
    with (root / "Data.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(_WOS_HEADER)
        writer.writerows(_wos_rows(10))
    cfg = {"dataset_name": "wos", "dataset_cache": str(root), "max_source_words": 3}
    ds = load_t2g_dataset(cfg)
    assert all(len(r["text"].split()) == 3 for r in ds["train"])
    # Il dedup/split usa il testo INTERO: con il troncamento a 3 parole le
    # righe sarebbero tutte uguali, ma restano 10.
    assert len(ds["train"]) + len(ds["test"]) == 10


def test_wos_missing_file_explains(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    with pytest.raises(FileNotFoundError, match="Data.xlsx"):
        load_t2g_dataset({"dataset_name": "wos-46985", "dataset_cache": str(tmp_path)})


# ── CoNLL-2003 ──────────────────────────────────────────────────────────────


def test_conll_entity_extraction_iob1_and_iob2() -> None:
    from src.datasets.conll_dataset import extract_entities, linearize_entities

    tokens = ["EU", "rejects", "German", "call", "to", "boycott", "British", "lamb"]
    iob1 = ["I-ORG", "O", "I-MISC", "O", "O", "O", "I-MISC", "O"]
    iob2 = ["B-ORG", "O", "B-MISC", "O", "O", "O", "B-MISC", "O"]
    assert extract_entities(tokens, iob1) == extract_entities(tokens, iob2)
    assert linearize_entities(extract_entities(tokens, iob1)) == (
        "ORG:EU MISC:German MISC:British"
    )
    # Due entità adiacenti dello stesso tipo: separate solo da B- (IOB1).
    assert extract_entities(
        ["Peter", "Blackburn", "Paul"], ["I-PER", "I-PER", "B-PER"]
    ) == [
        ("PER", "Peter Blackburn"),
        ("PER", "Paul"),
    ]
    # Ordine dei campi del JSON (PER, ORG, LOC, MISC) e niente duplicati.
    ents = [("LOC", "New York"), ("PER", "Bob"), ("LOC", "New York")]
    assert linearize_entities(ents) == "PER:Bob LOC:New_York"
    assert linearize_entities([]) == "NONE"


def test_conll_loader_official_splits(tmp_path: Path) -> None:
    root = tmp_path / "conll" / "raw"
    root.mkdir(parents=True)
    doc = (
        "-DOCSTART- -X- -X- O\n\n"
        "EU NNP B-NP I-ORG\nrejects VBZ B-VP O\nGerman JJ B-NP I-MISC\n. . O O\n\n"
        "Peter NNP B-NP I-PER\nBlackburn NNP I-NP I-PER\n\n"
        "1996-08-22 CD B-NP O\n\n"
    )
    for name in ("eng.train", "eng.testa", "eng.testb"):
        (root / name).write_text(doc, encoding="utf-8")
    ds = load_t2g_dataset(
        {"dataset_name": "conll-2003", "dataset_cache": str(tmp_path / "conll")}
    )
    assert set(ds) == {"train", "validation", "test"}
    rows = list(ds["train"])
    assert [r["gloss"] for r in rows] == [
        "ORG:EU MISC:German",
        "PER:Peter_Blackburn",
        "NONE",
    ]
    assert rows[0]["text"] == "EU rejects German ."


def test_conll_missing_split_explains(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    (tmp_path / "train.txt").write_text("EU NNP B-NP B-ORG\n", encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="setup.sh"):
        load_t2g_dataset({"dataset_name": "conll-2003", "dataset_cache": str(tmp_path)})


# ── Download automatico ─────────────────────────────────────────────────────


def test_fetch_verifies_sha256_and_leaves_no_partial_file(tmp_path: Path) -> None:
    import hashlib

    from src.datasets.download import fetch

    src = tmp_path / "src.txt"
    src.write_bytes(b"ciao")
    good = hashlib.sha256(b"ciao").hexdigest()
    dest = fetch(src.as_uri(), tmp_path / "out" / "a.txt", good)
    assert dest.read_bytes() == b"ciao"
    with pytest.raises(RuntimeError, match="sha256"):
        fetch(src.as_uri(), tmp_path / "out" / "b.txt", "0" * 64)
    assert sorted(p.name for p in (tmp_path / "out").iterdir()) == ["a.txt"]


def test_missing_conll_files_are_downloaded_when_online(
    tmp_path: Path, monkeypatch
) -> None:
    """Online e file assenti: il loader scarica i parquet della copia HF, li
    riscrive nel formato a colonne e carica."""
    import hashlib

    import pyarrow as pa
    import pyarrow.parquet as pq

    from src.datasets import conll_dataset

    src = tmp_path / "hf"
    src.mkdir()
    files = {}
    for split, (_name, _sha, target) in conll_dataset.CONLL_HF_FILES.items():
        # ner_tags interi come nella copia HF: 3 = B-ORG, 7 = B-MISC.
        table = pa.table(
            {"tokens": [["EU", "rejects", "German"]], "ner_tags": [[3, 0, 7]]}
        )
        pq.write_table(table, src / f"{split}.parquet")
        sha = hashlib.sha256((src / f"{split}.parquet").read_bytes()).hexdigest()
        files[split] = (f"{split}.parquet", sha, target)
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    monkeypatch.setattr(conll_dataset, "CONLL_HF", src.as_uri() + "/")
    monkeypatch.setattr(conll_dataset, "CONLL_HF_FILES", files)
    data_dir = tmp_path / "data"
    ds = load_t2g_dataset(
        {"dataset_name": "conll-2003", "dataset_cache": str(data_dir)}
    )
    assert ds["test"][0]["gloss"] == "ORG:EU MISC:German"
    assert ds["test"][0]["text"] == "EU rejects German"
    # I parquet vengono cancellati dopo la conversione.
    assert sorted(p.name for p in data_dir.iterdir()) == [
        "test.txt",
        "train.txt",
        "valid.txt",
    ]


def test_missing_wos_file_is_rebuilt_from_the_hf_parquet(
    tmp_path: Path, monkeypatch
) -> None:
    """Parquet HF → Data.csv con il codice Y ricavato dal vettore label."""
    import hashlib

    import pyarrow as pa
    import pyarrow.parquet as pq

    from src.datasets import wos_dataset

    def label(domain: int, y: int) -> list[float]:
        v = [0.0] * 141
        v[domain] = v[7 + y] = 1.0
        return v

    texts = [f"abstract number {i} about things" for i in range(10)]
    table = pa.table(
        {
            "text": texts,
            # Y=40 con due nomi: il canonico è il più frequente (Psychology).
            "label": [label(2, 40)] * 6 + [label(5, 40)] * 4,
            "label_description": [["Psychology  ", "depression"]] * 6
            + [["Medical ", "Depression"]] * 4,
        }
    )
    parquet = tmp_path / "wos.parquet"
    pq.write_table(table, parquet)
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    monkeypatch.setattr(wos_dataset, "WOS_HF", parquet.as_uri())
    monkeypatch.setattr(
        wos_dataset, "WOS_HF_SHA256", hashlib.sha256(parquet.read_bytes()).hexdigest()
    )
    data_dir = tmp_path / "data"
    ds = load_t2g_dataset({"dataset_name": "wos", "dataset_cache": str(data_dir)})
    glosses = set(ds["train"]["gloss"]) | set(ds["test"]["gloss"])
    assert glosses == {"Psychology depression"}
    assert sorted(p.name for p in data_dir.iterdir()) == ["Data.csv"]


def test_wos_labels_follow_the_official_code() -> None:
    """Nomi incoerenti con lo stesso Y → la coppia più frequente del codice."""
    from src.datasets.wos_dataset import canonical_labels

    rows = [
        {"Y": "40", "Domain": "Psychology  ", "area": " depression "},
        {"Y": "40", "Domain": "Psychology  ", "area": " depression "},
        {"Y": "40", "Domain": "Medical ", "area": " Depression "},
        {"Y": "7", "Domain": "ECE ", "area": " Satellite radio "},
        {"Y": "7", "Domain": "ECE ", "area": " Electric motor "},
    ]
    assert canonical_labels(rows) == {
        "40": ("Psychology", "depression"),
        # Parità 1-1: vince l'ordine alfabetico, non l'ordine delle righe.
        "7": ("ECE", "Electric_motor"),
    }
