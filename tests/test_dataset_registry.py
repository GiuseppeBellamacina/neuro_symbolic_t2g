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


def test_phoenix_loader_missing_files_explains(tmp_path: Path) -> None:
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


@pytest.mark.parametrize(
    "rel",
    sorted(
        str(p.relative_to(CONFIGS / "phoenix-2014t"))
        for p in (CONFIGS / "phoenix-2014t").rglob("*.yaml")
    ),
)
def test_phoenix_cells_resolve_to_phoenix_everywhere(rel: str) -> None:
    """Ogni cella PHOENIX usa dataset, prompt, cache e output di PHOENIX."""
    cfg = resolve_config(str(CONFIGS / "phoenix-2014t" / rel))
    assert get_dataset_spec(cfg["dataset"]) is PHOENIX_2014T
    assert cfg["dataset"]["prompt_profile"] == "de-dgs"
    for key in ("dataset_cache", "vocab_path", "bigram_matrix_path"):
        assert cfg["dataset"][key].startswith("data/phoenix-2014t"), key
    assert cfg["retrieval"]["cache_path"].startswith("data/phoenix-2014t")
    for key in ("output_dir", "log_dir"):
        if key in cfg.get("training", {}):
            assert "/phoenix-2014t/qwen25-05b/" in cfg["training"][key], key
    assert cfg["wandb"]["run_name"].startswith("phoenix-2014t-")


@pytest.mark.parametrize("dataset", ["aslg-pc12", "phoenix-2014t"])
def test_full_vocab_trie_is_single_factor_ablation(dataset: str) -> None:
    """full-vocab-trie = grpo/few-shot + vocab_source: all (e nient'altro di rilevante)."""
    root = CONFIGS / dataset / "qwen25-05b"
    leak = resolve_config(str(root / "ablations/decoding/full-vocab-trie.yaml"))
    ref = resolve_config(str(root / "grpo/few-shot.yaml"))
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


def test_aslg_cells_do_not_declare_vocab_source() -> None:
    """La chiave entra nel fingerprint SFT: solo le celle con leak la dichiarano."""
    for path in (CONFIGS / "aslg-pc12").rglob("*.yaml"):
        cfg = resolve_config(str(path))
        if path.name != "full-vocab-trie.yaml":
            assert "vocab_source" not in cfg["dataset"], path


# ── Tag: Python, _lib.sh e cluster_helper.sh concordano ─────────────────────


@pytest.mark.skipif(shutil.which("bash") is None, reason="bash non disponibile")
def test_bash_tag_matches_python_cell_tag_for_every_config() -> None:
    configs = sorted(
        str(p.relative_to(ROOT))
        for p in CONFIGS.rglob("*.yaml")
        if p.name != "base.yaml"
    )
    script = f"source '{(ROOT / 'cluster' / '_lib.sh').as_posix()}'\n" + "".join(
        f"t2g_tag_from_config '{c}'\n" for c in configs
    )
    out = subprocess.run(
        ["bash", "-c", script], capture_output=True, text=True, check=True
    ).stdout.split()
    expected = [
        run_paths.cell_tag(run_paths.cell_from_config(ROOT / c)) for c in configs
    ]
    assert out == expected
    assert len(set(out)) == len(out), "due celle con lo stesso tag"
