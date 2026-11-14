"""chain_prereq_train (cluster/_lib.sh): an eval whose checkpoint is missing gets its train queued first."""

import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
BASH = shutil.which("bash")
pytestmark = pytest.mark.skipif(BASH is None, reason="bash not available")

BASE = "conll-2003/qwen25-05b"


def _project(tmp_path):
    cfg = tmp_path / "experiments/configs" / BASE
    (cfg / "sft").mkdir(parents=True)
    (cfg / "ablations").mkdir()
    (cfg / "base.yaml").write_text("model:\n  name: x\n")
    (cfg / "sft/zero-shot.yaml").write_text(
        'extends: ../base.yaml\ntraining:\n  output_dir: "experiments/checkpoints/ck/sft"\n'
    )
    (cfg / "ablations/greedy.yaml").write_text(
        "extends: ../sft/zero-shot.yaml\nevaluation:\n  eval_only: true\n"
    )
    (cfg / "baseline.yaml").write_text(
        "extends: base.yaml\nevaluation:\n  eval_only: true\n"
    )
    return tmp_path


def _prereq(proj, cell):
    script = (
        f'source "{ROOT.as_posix()}/cluster/_lib.sh"; PROJ_DIR="$(cd "{proj.as_posix()}" && pwd)"; '
        f'STATE_DIR="$PROJ_DIR/.state"; '
        f'chain_prereq_train "experiments/configs/{BASE}/{cell}.yaml"'
    )
    r = subprocess.run([BASH, "-c", script], capture_output=True, text=True, check=True)
    return r.stdout.strip()


def test_missing_checkpoint_queues_parent_train(tmp_path):
    proj = _project(tmp_path)
    out = _prereq(proj, "ablations/greedy")
    assert (
        out == f"experiments/configs/{BASE}/sft/zero-shot.yaml|conll-2003-sft-zero-shot"
    )


def test_only_one_attempt_per_cell(tmp_path):
    proj = _project(tmp_path)
    assert _prereq(proj, "ablations/greedy")
    assert _prereq(proj, "ablations/greedy") == ""  # marker: no loop if the train fails


def test_existing_checkpoint_needs_nothing(tmp_path):
    proj = _project(tmp_path)
    (proj / "experiments/checkpoints/ck/sft/run_1/final").mkdir(parents=True)
    assert _prereq(proj, "ablations/greedy") == ""


def test_base_model_eval_needs_nothing(tmp_path):
    proj = _project(tmp_path)
    assert _prereq(proj, "baseline") == ""
