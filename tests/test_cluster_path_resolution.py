"""Regression tests for SLURM-spool-safe cluster library resolution."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
CLUSTER_DIR = ROOT / "cluster"
SCRIPTS = (
    "setup.sh",
    "preflight.sh",
    "train.sh",
    "eval.sh",
    "probe.sh",
    "structured_probe.sh",
    "diagnose.sh",
    "run_all.sh",
    "chain_tick.sh",
)
BEGIN = "# BEGIN T2G_LIB_RESOLVER"
END = "# END T2G_LIB_RESOLVER"


def resolver_block(name: str = "preflight.sh") -> str:
    source = (CLUSTER_DIR / name).read_text(encoding="utf-8")
    return source.split(BEGIN, 1)[1].split(END, 1)[0].strip()


def write_runner(path: Path) -> None:
    path.write_text(
        "#!/bin/bash\nset -euo pipefail\n"
        f"{resolver_block()}\n"
        "printf '%s\\n' \"$SCRIPT_DIR\"\n",
        encoding="utf-8",
    )


def write_lib(cluster: Path) -> None:
    cluster.mkdir(parents=True)
    (cluster / "_lib.sh").write_text(":\n", encoding="utf-8")


def run_runner(
    runner: Path, *, home: Path, submit_dir: Path | None
) -> subprocess.CompletedProcess[str]:
    bash_probe = subprocess.run(
        ["bash", "--version"], check=False, capture_output=True, text=True
    )
    if bash_probe.returncode != 0:
        pytest.skip(f"bash runtime unavailable: {bash_probe.stderr.strip()}")
    env = os.environ.copy()
    env["HOME"] = str(home)
    if submit_dir is None:
        env.pop("SLURM_SUBMIT_DIR", None)
    else:
        env["SLURM_SUBMIT_DIR"] = str(submit_dir)
    return subprocess.run(
        ["bash", str(runner)],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )


def test_repo_invocation_resolves_script_local_cluster(tmp_path: Path):
    cluster = tmp_path / "repo" / "cluster"
    write_lib(cluster)
    runner = cluster / "preflight.sh"
    write_runner(runner)

    result = run_runner(runner, home=tmp_path / "empty-home", submit_dir=None)

    assert result.returncode == 0, result.stderr
    assert Path(result.stdout.strip()) == cluster.resolve()


def test_slurm_spool_copy_resolves_submit_directory(tmp_path: Path):
    cluster = tmp_path / "repo" / "cluster"
    write_lib(cluster)
    spool = tmp_path / "var" / "lib" / "slurm" / "job1"
    spool.mkdir(parents=True)
    runner = spool / "slurm_script"
    write_runner(runner)

    result = run_runner(runner, home=tmp_path / "empty-home", submit_dir=cluster.parent)

    assert result.returncode == 0, result.stderr
    assert Path(result.stdout.strip()) == cluster.resolve()


def test_cleanenv_style_home_fallback_resolves_canonical_checkout(tmp_path: Path):
    cluster = tmp_path / "home" / "neuro_symbolic_t2g" / "cluster"
    write_lib(cluster)
    spool = tmp_path / "spool"
    spool.mkdir()
    runner = spool / "slurm_script"
    write_runner(runner)

    result = run_runner(runner, home=tmp_path / "home", submit_dir=None)

    assert result.returncode == 0, result.stderr
    assert Path(result.stdout.strip()) == cluster.resolve()


def test_missing_candidates_fail_with_actionable_diagnostics(tmp_path: Path):
    spool = tmp_path / "spool"
    spool.mkdir()
    runner = spool / "slurm_script"
    write_runner(runner)
    submit_dir = tmp_path / "missing-repo"

    result = run_runner(runner, home=tmp_path / "empty-home", submit_dir=submit_dir)

    assert result.returncode != 0
    assert "cannot locate cluster/_lib.sh" in result.stderr
    assert f"BASH_SOURCE={runner}" in result.stderr
    assert f"SLURM_SUBMIT_DIR={submit_dir}" in result.stderr


@pytest.mark.parametrize("name", SCRIPTS)
def test_every_in_scope_lib_source_uses_identical_hardened_resolver(name: str):
    source = (CLUSTER_DIR / name).read_text(encoding="utf-8")
    assert source.count(BEGIN) == source.count(END) == 1
    assert resolver_block(name) == resolver_block()
    assert source.count('source "$SCRIPT_DIR/_lib.sh"') == 1
    assert '[ -f "$_lib_candidate/_lib.sh" ]' in source
    assert '[ -f "${SLURM_SUBMIT_DIR}/cluster/_lib.sh" ]' in source
    assert '[ -f "${HOME}/neuro_symbolic_t2g/cluster/_lib.sh" ]' in source
    assert "BASH_SOURCE=%s, SLURM_SUBMIT_DIR=%s" in source
