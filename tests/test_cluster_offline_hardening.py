#!/usr/bin/env python3
"""CI tests for the cluster offline hardening (shell-text based, no network).

The cluster invariant: the login node may have network but never runs project
python; compute jobs have python/GPU but NO internet. These tests read the
cluster shell scripts as TEXT (no execution, no network) and assert:

  1. export_offline_env defines every required offline variable and defaults
     HF_HOME to the shared project cache while preserving an explicit value;
  2. online setup is separated from offline train/eval/preflight/probe;
  3. prepare_data is gone — no compute-side download/regeneration path;
  4. require_cluster_artifacts targets the real loader artifacts
     (dataset cache dir, gloss_vocab.txt, bigram_transition.npy, sidecars);
  5. W&B offline vars are exported and forwarded to apptainer (train/eval
     parity);
  6. setup.sh is an online, login-node Apptainer bootstrap;
  7. preflight.sh validates offline env/artifacts/dataset/model/W&B without
     training, with the optional T2G_PDA_FULL_VOCAB=1 gate under PDA=1;
  8. probe.sh follows the same relaunch/offline/artifact conventions;
  9. bash-4 compatibility (no bash-5-only parameter expansions).
"""

from __future__ import annotations

import ast
import re
import tomllib
from pathlib import Path

CLUSTER_DIR = Path(__file__).resolve().parent.parent / "cluster"
EXPECTED_SENTENCE_TRANSFORMERS_VERSION = "5.2.3"

REQUIRED_OFFLINE_VARS = (
    "HF_HUB_OFFLINE",
    "TRANSFORMERS_OFFLINE",
    "HF_DATASETS_OFFLINE",
    "WANDB_MODE",
    "WANDB_DISABLE_WEAVE",
    "WANDB_SILENT",
    "PYTHONUNBUFFERED",
)

# Forbidden on compute nodes: nothing may install or fetch anything.
DOWNLOAD_FORBIDDEN_RE = re.compile(
    r"\b(pip3?\s+install|wget|curl|git\s+clone|hf_hub_download)\b"
)
PYTHON_OR_APPTAINER_RE = re.compile(r"\b(run_py|python3?|apptainer)\b")
BASH5_PARAM_EXPANSION_RE = re.compile(r"\$\{[A-Za-z_][A-Za-z0-9_]*@[A-Za-z]+\}")


def read_script(name: str) -> str:
    return (CLUSTER_DIR / name).read_text(encoding="utf-8")


def code_lines(name: str) -> list[str]:
    """Non-comment, non-blank lines (drops `#SBATCH` headers and comments)."""
    out: list[str] = []
    for line in read_script(name).splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        out.append(line)
    return out


def code_text(name: str) -> str:
    return "\n".join(code_lines(name))


_QUOTED_SPAN_RE = re.compile(r'"[^"]*"|\'[^\']*\'')


def code_text_unquoted(name: str) -> str:
    """Code lines with quoted spans blanked.

    Echo messages (e.g. ``echo "NESSUN pip install"``) are documentation, not
    invocations: forbidden-token scans must only see real command text.
    Multi-line quoted bodies (heredoc-style python) survive line-wise
    stripping, which is intentional: their content is still executed.
    """
    return "\n".join(_QUOTED_SPAN_RE.sub('""', ln) for ln in code_lines(name))


def first_index(lines: list[str], pattern: re.Pattern[str]) -> int | None:
    for i, line in enumerate(lines):
        if pattern.search(line):
            return i
    return None


# ---------------------------------------------------------------------------
# 1. export_offline_env in _lib.sh
# ---------------------------------------------------------------------------


def test_export_offline_env_defines_all_required_vars():
    """export_offline_env exports HF/transformers/datasets/W&B/PYTHON flags."""
    src = read_script("_lib.sh")
    body = src.split("export_offline_env() {", 1)[1].split("\n}", 1)[0]
    for var in REQUIRED_OFFLINE_VARS:
        assert re.search(
            rf"^\s*export {var}=", body, re.M
        ), f"export_offline_env must export {var}"
    # W&B must be offline with weave disabled and silent output.
    assert re.search(r"export WANDB_MODE=offline\b", body)
    assert re.search(r"export WANDB_DISABLE_WEAVE=true\b", body)
    assert re.search(r"export WANDB_SILENT=true\b", body)


def test_export_offline_env_defaults_hf_home_but_preserves_explicit():
    """HF_HOME defaults to the established shared cache; explicit value wins."""
    src = read_script("_lib.sh")
    body = src.split("export_offline_env() {", 1)[1].split("\n}", 1)[0]
    assert 'T2G_HF_HOME_DEFAULT="$HOME/.cache/huggingface"' in src, (
        "offline jobs must preserve the historical NFS-shared HF default; "
        "changing to a fresh cache root would hide existing snapshots"
    )
    assert re.search(r'if \[ -z "\$\{HF_HOME:-\}" \]; then', body), (
        "HF_HOME must only be defaulted when not already set "
        "(explicitly set HF_HOME must be preserved)"
    )
    assert "export HF_HOME=" in body


def test_artifact_probe_does_not_wrap_shell_function_with_timeout():
    """`timeout run_py` is invalid because run_py is a shell function."""
    src = read_script("_lib.sh")
    assert "timeout 120 run_py" not in src
    assert 'probe_out=$(run_py -c "$_T2G_ARTIFACT_PROBE"' in src
    assert ") || return 1" in src, "container/runtime failures must propagate"


def test_run_py_has_no_implicit_bare_python_fallback():
    body = read_script("_lib.sh").split("run_py() {", 1)[1].split("\n}", 1)[0]
    assert "APPTAINER_CONTAINER:-" in body
    assert "RUN_PY_FORCE_BARE:-0" in body
    assert "command -v apptainer" in body
    assert "[ ! -f /shared/sifs/latest.sif ]" in body
    assert "return 127" in body
    assert body.count('python3 "$@"') == 2  # safe direct branch + container command
    assert re.search(
        r'if \[ -n "\$\{APPTAINER_CONTAINER:-\}" \] \|\| '
        r'\[ "\$\{RUN_PY_FORCE_BARE:-0\}" = "1" \]; then\s*python3 "\$@"',
        body,
    )


def test_run_py_uses_exec_and_preserves_quoted_python_argv():
    """Bypass the SIF runscript, which rewrites quoted ``python -c`` argv."""
    body = read_script("_lib.sh").split("run_py() {", 1)[1].split("\n}", 1)[0]
    assert re.search(r"^\s*apptainer exec --nv \\$", body, re.M)
    assert not re.search(r"^\s*apptainer run\b", body, re.M)
    assert re.search(
        r"/shared/sifs/latest\.sif \\\n\s*python3 \"\$@\"\s*$",
        body,
    ), "run_py must pass each original argument directly after the SIF and python3"
    assert "eval " not in body


def test_remote_queue_rewrite_clears_last_job_only_without_active_slurm_job():
    """Fresh replacement drops stale completion state, not live tracking."""
    helper = (
        Path(__file__).resolve().parent.parent / "remote" / "cluster_helper.sh"
    ).read_text(encoding="utf-8")
    body = helper.split("rewrite_queue() {", 1)[1].split("\n}", 1)[0]
    assert "active_id=$(active_job_id)" in body
    branches = re.search(
        r'if \[ -z "\$active_id" \]; then(.*?)else(.*?)fi', body, re.DOTALL
    )
    assert branches is not None
    assert 'rm -f "$LAST_JOB_FILE"' in branches.group(1)
    assert "last_job stale rimosso" in branches.group(1)
    assert 'rm -f "$LAST_JOB_FILE"' not in branches.group(2)
    assert "last_job preservato" in branches.group(2)


# ---------------------------------------------------------------------------
# 2. Offline exports precede the first python/apptainer invocation
# ---------------------------------------------------------------------------


def _assert_offline_exports_first(name: str, code: list[str]) -> None:
    export_idx = first_index(code, re.compile(r"\bexport_offline_env\b"))
    python_idx = first_index(code, PYTHON_OR_APPTAINER_RE)
    assert export_idx is not None, f"{name}: export_offline_env must be called"
    assert python_idx is not None, f"{name}: expected a python/apptainer usage"
    assert export_idx < python_idx, (
        f"{name}: export_offline_env (line {export_idx}) must precede the "
        f"first python/apptainer invocation (line {python_idx})"
    )


def test_offline_exports_precede_first_python_in_train():
    _assert_offline_exports_first("train.sh", code_lines("train.sh"))


def test_offline_exports_precede_first_python_in_eval():
    _assert_offline_exports_first("eval.sh", code_lines("eval.sh"))


def test_offline_exports_precede_execution_in_other_compute_entrypoints():
    for name in ("preflight.sh", "diagnose.sh", "structured_probe.sh"):
        _assert_offline_exports_first(name, code_lines(name))


def test_setup_relaunches_before_any_python_on_login_node():
    """The sole outer relaunch path may not execute host Python or pip."""
    src = read_script("setup.sh")
    outer = src.split(
        '[ "${T2G_SETUP_CONTAINER:-0}" = "1" ] || [ -n "${APPTAINER_CONTAINER:-}" ]',
        1,
    )[0]
    assert "exec srun" in outer
    assert not re.search(r"^\s*(?:\"?\$?PYTHON[^ ]*\"?|python3?|pip3?)\s", outer, re.M)
    assert re.search(r"\bpython3?|\bpip\b", src[len(outer) :])


# ---------------------------------------------------------------------------
# 3. prepare_data removed — no download/regeneration path on compute nodes
# ---------------------------------------------------------------------------


def test_prepare_data_is_gone_everywhere():
    for script in sorted(CLUSTER_DIR.glob("*.sh")):
        code = "\n".join(
            ln
            for ln in script.read_text(encoding="utf-8").splitlines()
            if ln.strip() and not ln.strip().startswith("#")
        )
        assert not re.search(r"\bprepare_data\b", code), (
            f"{script.name}: prepare_data (compute-node download/regeneration) "
            "must not exist or be referenced in code"
        )


def test_lib_defines_require_cluster_artifacts():
    src = read_script("_lib.sh")
    assert re.search(
        r"^require_cluster_artifacts\(\) \{", src, re.M
    ), "_lib.sh must define require_cluster_artifacts"


def test_no_download_or_install_commands_in_compute_scripts():
    """No pip install / wget / curl / git clone / hf_hub_download anywhere."""
    for name in ("_lib.sh", "train.sh", "eval.sh", "preflight.sh", "probe.sh"):
        assert not DOWNLOAD_FORBIDDEN_RE.search(
            code_text_unquoted(name)
        ), f"{name}: compute scripts must never install or download"


def test_no_dataset_regeneration_calls_in_cluster_scripts():
    """The old prepare_data python (download/extract/save) is gone."""
    forbidden = (
        "download_aslg_dataset",
        "extract_gloss_vocabulary",
        "build_t2g_dataset",
        "save_to_disk",
        "compute_bigram_transitions",
        "save_transition_matrix",
    )
    for name in ("_lib.sh", "train.sh", "eval.sh", "preflight.sh", "probe.sh"):
        code = code_text(name)
        for token in forbidden:
            assert token not in code, f"{name}: must not call {token}()"


def test_train_and_eval_call_require_cluster_artifacts_with_config():
    for name in ("train.sh", "eval.sh"):
        code = code_text(name)
        assert (
            'require_cluster_artifacts "$CONFIG"' in code
        ), f"{name}: must fail fast on missing offline artifacts"


def test_no_fallback_download_claims_in_comments():
    """Old comments claimed a first-run download fallback was possible."""
    for name in ("train.sh", "eval.sh", "setup.sh"):
        assert "conserva la rete" not in read_script(
            name
        ), f"{name}: stale fallback-download comment must be removed"


# ---------------------------------------------------------------------------
# 4. require_cluster_artifacts targets the REAL loader artifacts
# ---------------------------------------------------------------------------


def test_artifact_check_targets_real_paths():
    src = read_script("_lib.sh")
    assert "data/aslg_pc12" in src, "HF dataset cache dir must be validated"
    assert "data/gloss_vocab.txt" in src, "gloss vocab must be validated"
    assert "data/bigram_transition.npy" in src, "optional bigram path must be resolved"
    assert ".meta.json" in src, "vocab sidecar must be required"
    # Real checks: nonempty file (-s) / nonempty dir, not just existence.
    assert re.search(r"\[ ! -s \"\$vocab_path\" \]", src)
    assert re.search(r"\[ ! -s \"\$bigram_path\" \]", src)
    assert "diagnostica bigram omessa" in src
    assert "_dir_nonempty" in src


def test_artifact_failure_message_tells_user_to_prepare_offline():
    src = read_script("_lib.sh")
    body = src.split("_artifact_fail() {", 1)[1].split("\n}", 1)[0]
    assert (
        "NON hanno internet" in body
    ), "message must state compute nodes have no internet"
    assert "ambiente" in body and "rete" in body, (
        "message must instruct preparing artifacts from a network-enabled "
        "environment"
    )
    assert (
        "data/gloss_vocab.txt" in body and "sidecar .meta.json" in body
    ), "message must name only the required artifacts to prepare/upload"
    assert "matrice bigram è opzionale" in body


def test_model_snapshot_checked_offline_via_local_files_only():
    """The probe mirrors resolve_model_source: local_files_only, no network."""
    src = read_script("_lib.sh")
    assert (
        "snapshot_download(mid, local_files_only=True)" in src
    ), "model probe must resolve the snapshot with local_files_only=True"
    assert (
        "MISSING" in src and "UNKNOWN" in src
    ), "probe must distinguish not-cached from unknown (shell fallback)"
    assert (
        "_hf_model_cached_shell" in src
    ), "shell-only HF cache fallback check must exist"


def test_tfidf_index_not_required_but_minilm_hard_fails():
    """tfidf may rebuild its index offline; minilm must never download."""
    code = code_text_unquoted("_lib.sh")
    assert (
        "data/retriever_index" not in code
    ), "retriever index must not be a hard requirement (rebuilds offline)"
    body = read_script("_lib.sh").split("require_cluster_artifacts() {", 1)[1]
    assert 'ret_backend" = "minilm"' in body, "minilm branch must exist"
    assert (
        "Modello SentenceTransformer" in body
    ), "minilm without a cached snapshot must hard fail"
    assert (
        "ricostruzione offline consentita" in body
    ), "tfidf/minilm index rebuild offline must be explicitly allowed"
    assert 'T2G_MINILM_DEFAULT="sentence-transformers/all-MiniLM-L6-v2"' in body or (
        'T2G_MINILM_DEFAULT="sentence-transformers/all-MiniLM-L6-v2"'
        in read_script("_lib.sh")
    ), "default MiniLM id must match example_retriever._DEFAULT_MINILM_MODEL"


# ---------------------------------------------------------------------------
# 5. W&B parity + apptainer env forwarding in train/eval
# ---------------------------------------------------------------------------


def test_train_and_eval_export_offline_env_shell_level():
    """W&B parity: both scripts export the offline env at shell level."""
    for name in ("train.sh", "eval.sh"):
        code = code_text(name)
        assert "export_offline_env" in code, f"{name}: shell-level offline env"


def test_train_and_eval_use_run_py_without_hand_rolled_container_or_python():
    """One shared execution model prevents train/eval container drift."""
    for name in ("train.sh", "eval.sh"):
        code = code_text(name)
        assert "run_py" in code
        assert not re.search(r"^\s*apptainer\s+(?:run|exec)\b", code, re.M)
        assert not re.search(r"^\s*python3?\s", code, re.M)
    assert "run_py -m src.training --config" in code_text("train.sh")
    assert "run_py -m src.training.eval_t2g" in code_text("eval.sh")
    assert "PYTORCH_ALLOC_CONF" in code_text("train.sh")


def test_run_py_forwards_offline_env_to_apptainer():
    body = read_script("_lib.sh").split("run_py() {", 1)[1].split("\n}", 1)[0]
    for var in REQUIRED_OFFLINE_VARS:
        assert f'--env "{var}=' in body, f"run_py must forward {var} to apptainer"
    assert '--env "HF_HUB_CACHE=' in body
    assert '--env "PYTORCH_ALLOC_CONF=' in body


# ---------------------------------------------------------------------------
# 6. setup.sh is online acquisition
# ---------------------------------------------------------------------------


def test_setup_is_online_login_container_bootstrap():
    src = read_script("setup.sh")
    code = code_text("setup.sh")
    assert "export_offline_env" not in code
    assert "HF_HUB_OFFLINE" not in code
    assert "TRANSFORMERS_OFFLINE" not in code
    assert "HF_DATASETS_OFFLINE" not in code
    assert "apptainer run --nv" in code
    assert "--cleanenv" in code
    assert '--home "$HOME:$HOME"' in code and '--bind "$HOME:$HOME"' in code
    assert 'HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"' in code
    assert '--env "HF_HOME=$HF_HOME"' in code
    assert "-m pip install --user" in code
    assert '".[retrieval]"' in code
    assert "snapshot_download" in src
    assert "src.utils.setup_artifacts" in code
    assert "load_dataset" in src
    assert "preflight.sh" in src


def test_setup_uses_historical_srun_as_only_outer_path():
    src = read_script("setup.sh")
    assert "command -v srun >/dev/null 2>&1" in src
    assert 'ACCOUNT="${SLURM_ACCOUNT:-thesis-course}"' in src
    assert 'PARTITION="${T2G_SLURM_PARTITION:-${SLURM_PARTITION:-$ACCOUNT}}"' in src
    assert 'QOS="${T2G_SLURM_QOS:-${SLURM_QOS:-gpu-xlarge}}"' in src
    assert 'GPU_GRES="${T2G_SETUP_GPU_GRES:-gpu:1}"' in src
    assert 'SHARD_GRES="${T2G_SETUP_SHARD_GRES:-shard:22000}"' in src
    assert 'MEM="${T2G_SETUP_MEM:-48G}"' in src
    assert 'CPUS="${T2G_SETUP_CPUS:-8}"' in src
    assert re.search(r"exec srun .*?apptainer run --nv", src, re.DOTALL)
    assert src.count("exec srun") == 1
    for forbidden in (
        "T2G_APPTAINER",
        "SETUP_RUNTIME",
        "command -v apptainer",
        "command -v singularity",
        "/usr/bin/apptainer",
        "/usr/local/bin/apptainer",
        "apptainer exec",
        "singularity exec",
    ):
        assert forbidden not in src


def test_setup_relaunch_marker_and_loop_guard_prevent_bare_execution():
    src = read_script("setup.sh")
    assert src.count('--env "T2G_SETUP_CONTAINER=1"') == 1
    assert re.search(
        r'\[ "\$\{T2G_SETUP_CONTAINER:-0\}" = "1" \] \|\| '
        r'\[ -n "\$\{APPTAINER_CONTAINER:-\}" \] \|\| \{',
        src,
    )
    assert "refusing bare host execution" in src
    assert "setup must launch Apptainer on a compute node" in src


def test_setup_is_the_only_cluster_online_acquisition_script():
    for script in sorted(CLUSTER_DIR.glob("*.sh")):
        if script.name == "setup.sh":
            continue
        code = "\n".join(
            _QUOTED_SPAN_RE.sub('""', line)
            for line in script.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.strip().startswith("#")
        )
        assert not DOWNLOAD_FORBIDDEN_RE.search(
            code
        ), f"{script.name}: setup.sh must remain the only online acquisition script"


def test_setup_protects_and_verifies_complete_critical_stack():
    src = read_script("setup.sh")
    assert "importlib.metadata" in src
    for package in (
        "torch",
        "torchao",
        "triton",
        "xformers",
        "bitsandbytes",
        "nvidia-",
        "transformers",
        "accelerate",
        "trl",
        "peft",
        "unsloth",
        "unsloth-zoo",
        "sentence-transformers",
    ):
        assert package in src
    assert 're.sub(r"[-_.]+", "-", name).lower()' in src
    assert '--constraint "$CONSTRAINTS"' in src
    assert '"packages": before' in src
    assert "pip changed protected packages" in src
    assert "tested package versions not active" in src
    assert 'Version("0.16")' in src and 'Version("0.18")' in src
    assert "pip uninstall" not in src


def test_pyproject_pins_tested_transformers_without_changing_stack_pins():
    project = (CLUSTER_DIR.parent / "pyproject.toml").read_text(encoding="utf-8")
    for requirement in (
        '"transformers==5.3.0"',
        '"trl==0.24.0"',
        '"peft==0.19.1"',
        '"unsloth==2026.7.1"',
        '"unsloth_zoo==2026.7.1"',
        '"torchao>=0.16.0,<0.18"',
    ):
        assert requirement in project


def test_retrieval_extra_matches_protected_sentence_transformers_version():
    project_path = CLUSTER_DIR.parent / "pyproject.toml"
    project = tomllib.loads(project_path.read_text(encoding="utf-8"))
    retrieval = project["project"]["optional-dependencies"]["retrieval"]
    declared = next(
        requirement
        for requirement in retrieval
        if requirement.startswith("sentence-transformers")
    )
    assert declared == (
        f"sentence-transformers=={EXPECTED_SENTENCE_TRANSFORMERS_VERSION}"
    )


def test_setup_enforces_retrieval_version_as_part_of_exact_tested_stack():
    tree = ast.parse(read_script("setup.sh").split("<<'PY'", 2)[2].split("\nPY", 1)[0])
    validate = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "validate_versions"
    )
    exact_assignment = next(
        node
        for node in ast.walk(validate)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "exact"
            for target in node.targets
        )
    )
    exact = ast.literal_eval(exact_assignment.value)
    assert exact["sentence-transformers"] == EXPECTED_SENTENCE_TRANSFORMERS_VERSION


def _load_setup_version_validator():
    """Compile only setup's pure version-comparison function."""
    tree = ast.parse(read_script("setup.sh").split("<<'PY'", 2)[2].split("\nPY", 1)[0])
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "validate_versions"
    )
    module = ast.Module(body=[function], type_ignores=[])
    namespace = {
        "Version": __import__("packaging.version", fromlist=["Version"]).Version
    }
    exec(
        compile(ast.fix_missing_locations(module), "setup-validator", "exec"), namespace
    )
    return namespace["validate_versions"]


def test_setup_version_validator_accepts_tested_stack():
    validate = _load_setup_version_validator()
    packages = {
        "torch": "2.7.1",
        "transformers": "5.3.0",
        "trl": "0.24.0",
        "peft": "0.19.1",
        "unsloth": "2026.7.1",
        "unsloth-zoo": "2026.7.1",
        "torchao": "0.17.0",
        "sacrebleu": "2.6.0",
        "sentence-transformers": EXPECTED_SENTENCE_TRANSFORMERS_VERSION,
    }
    identity = {"torch": "2.7.1+cu118", "cuda": "11.8"}
    validate(packages, packages.copy(), identity, identity.copy())


def test_setup_version_validator_rejects_changes_and_untested_versions():
    validate = _load_setup_version_validator()
    packages = {
        "transformers": "5.3.0",
        "trl": "0.24.0",
        "peft": "0.19.1",
        "unsloth": "2026.7.1",
        "unsloth-zoo": "2026.7.1",
        "torchao": "0.17.0",
        "sentence-transformers": EXPECTED_SENTENCE_TRANSFORMERS_VERSION,
    }
    identity = {"torch": "2.7.1+cu118", "cuda": "11.8"}
    changed = packages | {"trl": "0.25.0"}
    try:
        validate(packages, changed, identity, identity)
    except RuntimeError as error:
        assert "changed protected packages" in str(error)
    else:
        raise AssertionError("a protected-package change must fail")

    absent_before = {
        name: version for name, version in packages.items() if name != "torchao"
    }
    try:
        validate(absent_before, packages | {"torchao": "0.18.0"}, identity, identity)
    except RuntimeError as error:
        assert "tested package versions" in str(error)
    else:
        raise AssertionError("an unsupported torchao version must fail")


def test_artifact_cli_is_lightweight_and_explicitly_online():
    cli = (CLUSTER_DIR.parent / "src" / "utils" / "setup_artifacts.py").read_text(
        encoding="utf-8"
    )
    assert "snapshot_download" in cli
    assert "download_aslg_dataset" in cli and "online=True" in cli
    assert "extract_gloss_vocabulary" in cli and "write_cache_meta" in cli
    assert "BUILD_BIGRAM=1" in cli
    assert "import torch" not in cli and "unsloth" not in cli.lower()


# ---------------------------------------------------------------------------
# 7. preflight.sh: validates everything, trains nothing
# ---------------------------------------------------------------------------


def test_preflight_exists_and_uses_slurm_conventions():
    src = read_script("preflight.sh")
    assert "#SBATCH" in src, "preflight must be sbatch-able"
    assert "SLURM_JOB_ID" in src and "exec srun" in src
    assert "run_py" in src, "preflight must use the centralized container runner"
    assert "APPTAINER_CONTAINER" in read_script("_lib.sh")
    assert "srun" in src, "preflight must support direct login-node srun"
    assert "_T2G_PREFLIGHT_BARE" not in src
    assert "--gres=gpu:1 --gres=shard:22528" in src
    assert not re.search(
        r"^\s*apptainer\s+(?:run|exec)\b", code_text("preflight.sh"), re.M
    )


def test_preflight_validates_offline_env_artifacts_dataset_wandb():
    code = code_text("preflight.sh")
    assert "export_offline_env" in code
    assert "require_cluster_artifacts" in code
    assert "load_dataset" in code, "preflight must test the cached dataset load"
    assert (
        "resolve_model_source" in code
    ), "preflight must test offline model source resolution"
    assert "wandb" in code, "preflight must validate the W&B offline mode"


def test_preflight_does_not_train():
    code = code_text("preflight.sh")
    assert "-m src.training" not in code, "preflight must never launch training"
    assert (
        "python -m src" not in code
    ), "preflight must not invoke any src.training entrypoint"


def test_preflight_pda_gate_is_focused_and_opt_in():
    code = code_text("preflight.sh")
    assert "PDA:-0" in code or "PDA:-0}" in code, "PDA gate must be opt-in"
    assert (
        "T2G_PDA_FULL_VOCAB=1" in code
    ), "PDA=1 must run the T2G_PDA_FULL_VOCAB=1 full-vocabulary gate"
    assert "-k full_vocabulary" in code, "only the focused gate may run"
    assert "test_pda_grammar.py" in code


# ---------------------------------------------------------------------------
# 8. probe launcher
# ---------------------------------------------------------------------------


def test_probe_is_sbatchable_and_relaunches_login_node_first():
    src = read_script("probe.sh")
    assert "#SBATCH --output=logs/slurm-probe-%j.log" in src
    assert "#SBATCH --mem=" in src and "#SBATCH --cpus-per-task=" in src
    code = code_lines("probe.sh")
    relaunch_idx = first_index(code, re.compile(r"^\s*exec srun\b"))
    assert relaunch_idx is not None
    before = "\n".join(code[:relaunch_idx])
    assert not re.search(
        r"\b(run_py|python3?|apptainer\s+(run|exec))\b", before
    ), "login-node code must not execute Python/Apptainer before exec srun"
    assert "apptainer run --nv /shared/sifs/latest.sif" in src


def test_probe_initializes_compute_environment_before_python():
    code = code_lines("probe.sh")
    assert 'source "$SCRIPT_DIR/_lib.sh"' in code_text("probe.sh")
    assert 'cd "$PROJ_DIR"' in code_text("probe.sh")
    export_idx = first_index(code, re.compile(r"^export_offline_env$"))
    run_idx = first_index(code, re.compile(r"^run_py -m src\.analysis\b"))
    assert export_idx is not None and run_idx is not None
    assert export_idx < run_idx
    assert "RUN_PY_FORCE_BARE=1" in code_text("probe.sh")


def test_probe_checks_input_and_mode_artifacts_before_analysis():
    code = code_lines("probe.sh")
    run_idx = first_index(code, re.compile(r"^run_py -m src\.analysis\b"))
    assert run_idx is not None
    before = "\n".join(code[:run_idx])
    assert '[ -f "$INPUT" ]' in before
    assert '[ -s "$VOCAB" ]' in before
    assert 'if [ "$COMMAND" = "markov" ]; then' in before
    assert '[ -s "$BIGRAM" ]' in before


def test_probe_runs_only_analysis_for_both_supported_modes():
    code = code_text("probe.sh")
    assert "rollouts|rewards|markov" in code
    assert 'ARGS=("$COMMAND" --config "$CONFIG" --input "$INPUT")' in code
    assert 'ARGS+=(--output "$OUTPUT")' in code
    assert code.count("run_py -m src.analysis") == 1
    assert "-m src.training" not in code
    assert not DOWNLOAD_FORBIDDEN_RE.search(code_text_unquoted("probe.sh"))


def test_diagnose_avoids_nested_srun_and_uses_shared_python_runner():
    src = read_script("diagnose.sh")
    code = code_text("diagnose.sh")
    assert '[ -z "${SLURM_JOB_ID:-}" ]' in src
    assert src.count("exec srun") == 1
    assert "--gres=gpu:1 --gres=shard:22528" in src
    assert "export_offline_env" in code
    assert "run_py" in code
    assert not re.search(r"^\s*\$PY\b|^\s*python3?\s", code, re.M)
    assert not re.search(r"^\s*apptainer\s+(?:run|exec)\b", code, re.M)


def test_structured_probe_uses_run_py_and_requires_config_artifacts():
    src = read_script("structured_probe.sh")
    code = code_text("structured_probe.sh")
    assert "--gres=gpu:1 --gres=shard:22528" in src
    assert 'require_cluster_artifacts "$CONFIG"' in code
    assert code.count("run_py -m src.analysis.structured_benchmark") == 2
    assert not re.search(r"^\s*apptainer\s+(?:run|exec)\b", code, re.M)


# ---------------------------------------------------------------------------
# 9. bash 4 compatibility
# ---------------------------------------------------------------------------


def test_no_bash5_only_parameter_expansions():
    for name in (
        "_lib.sh",
        "train.sh",
        "eval.sh",
        "setup.sh",
        "preflight.sh",
        "probe.sh",
        "diagnose.sh",
        "structured_probe.sh",
    ):
        src = read_script(name)
        match = BASH5_PARAM_EXPANSION_RE.search(src)
        assert match is None, f"{name}: bash-5-only expansion {match.group(0)!r}"


def test_scripts_are_quoted_paths_only():
    """Basic quoting hygiene: cd/source/require use quoted variables."""
    for name in (
        "train.sh",
        "eval.sh",
        "setup.sh",
        "preflight.sh",
        "probe.sh",
        "diagnose.sh",
        "structured_probe.sh",
    ):
        code = code_text(name)
        assert 'source "$SCRIPT_DIR/_lib.sh"' in code, f"{name}: quoted source"
        assert 'cd "$PROJ_DIR"' in code, f"{name}: quoted cd"
