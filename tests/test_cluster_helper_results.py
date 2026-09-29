"""Risoluzione delle celle di risultati in ``remote/cluster_helper.sh``.

Il client (TUI e app remota) chiede i risultati o con il percorso esatto di una
cella, dalla discovery, o con la chiave del config (``grpo-few-shot``,
``baseline-zero-shot``) quando la discovery non risponde. Le celle sono
annidate come i config, quindi la chiave non è una sottostringa del percorso
(``baseline-zero-shot`` contro ``qwen25-05b/baseline/zero-shot``) e una
ricerca per sottostringa cadeva sulla cella sbagliata.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

HELPER = Path(__file__).resolve().parents[1] / "remote" / "cluster_helper.sh"
BASH = shutil.which("bash")

CELLS = [
    "qwen25-05b/baseline/zero-shot",
    "qwen25-05b/baseline/zero-shot-no-grammar",
    "qwen25-05b/baseline/few-shot",
    "qwen25-05b/grpo/few-shot",
    "qwen25-05b/sft-grpo/few-shot",
    "qwen25-05b/ablations/loss/dr-grpo",
    "qwen25-05b/sft/zero-shot",
]

pytestmark = pytest.mark.skipif(BASH is None, reason="bash non disponibile")


def _call(project: Path, fn: str, arg: str = "") -> str:
    script = (
        f'PROJ_DIR="{project.as_posix()}"\n'
        f"eval \"$(sed -n '/^_results_cells() {{/,/^}}/p' '{HELPER.as_posix()}')\"\n"
        f"eval \"$(sed -n '/^results() {{/,/^}}/p' '{HELPER.as_posix()}')\"\n"
        "_emit_run() { :; }\n"
        f'{fn} "{arg}"\n'
    )
    out = subprocess.run(
        [BASH, "-c", script], capture_output=True, text=True, errors="replace"
    )
    assert out.returncode == 0, out.stderr
    return out.stdout


@pytest.fixture
def project(tmp_path: Path) -> Path:
    results = tmp_path / "experiments" / "results"
    for cell in CELLS:
        run = results / cell / "run_20260101_000000"
        run.mkdir(parents=True)
        (run / "eval_final.json").write_text("{}", encoding="utf-8")
    greedy = results / "qwen25-05b/sft/zero-shot/run_20260101_000000/decoding-greedy"
    greedy.mkdir()
    (greedy / "eval_final.json").write_text("{}", encoding="utf-8")
    # orfano del vecchio layout: niente segmento run_*
    (results / "qwen25-05b/sft/eval_final.json").write_text("{}", encoding="utf-8")
    return tmp_path


def _resolved(project: Path, token: str) -> str:
    for line in _call(project, "results", token).splitlines():
        if line.startswith("RESULTS_DIR="):
            return line.split("experiments/results/", 1)[-1]
    raise AssertionError("RESULTS_DIR mancante")


def test_discovery_lists_each_cell_once_and_skips_orphans(project: Path) -> None:
    listed = [c for c in _call(project, "_results_cells").splitlines() if c]
    assert sorted(listed) == sorted(CELLS)


@pytest.mark.parametrize("cell", CELLS)
def test_config_key_resolves_to_its_own_cell(project: Path, cell: str) -> None:
    key = cell.split("/", 1)[1].replace("/", "-")
    assert _resolved(project, key) == cell


def test_exact_cell_path_still_resolves(project: Path) -> None:
    assert _resolved(project, "qwen25-05b/baseline/few-shot") == (
        "qwen25-05b/baseline/few-shot"
    )
