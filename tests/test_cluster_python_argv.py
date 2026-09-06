"""Executable regressions for shell-to-Python argument boundaries."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


def test_eval_identity_keeps_adversarial_config_in_one_argv_element():
    candidates = [
        shutil.which("bash"),
        r"C:\Program Files\Git\bin\bash.exe",
    ]
    bash = next(
        (
            candidate
            for candidate in candidates
            if candidate
            and Path(candidate).is_file()
            and subprocess.run(
                [candidate, "-c", "exit 0"],
                check=False,
                capture_output=True,
            ).returncode
            == 0
        ),
        None,
    )
    if bash is None:
        pytest.skip("bash is unavailable")

    source = (ROOT / "cluster" / "eval.sh").read_text(encoding="utf-8")
    match = re.search(
        r"^resolve_experiment_identity\(\) \{.*?^\}",
        source,
        re.MULTILINE | re.DOTALL,
    )
    assert match is not None

    config = "configs/quote'and\"double;$(printf injected).yaml"
    command = "\n".join(
        (
            'run_py() { printf "%s\\0" "$@"; }',
            match.group(0),
            "resolve_experiment_identity",
        )
    )
    env = os.environ.copy()
    env["CONFIG"] = config
    result = subprocess.run(
        [bash, "-c", command],
        check=True,
        capture_output=True,
        env=env,
    )

    argv = result.stdout.rstrip(b"\0").split(b"\0")
    assert argv[0] == b"-c"
    assert b"resolve_config(sys.argv[1])" in argv[1]
    assert argv[2:] == [config.encode()]
