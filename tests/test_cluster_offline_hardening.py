#!/usr/bin/env python3
"""CI tests for the cluster offline hardening (shell-text based, no network).

The cluster invariant: the login node may have network but never runs project
python; compute jobs have python/GPU but NO internet. These tests read the
cluster shell scripts as TEXT (no execution, no network) and assert:

  1. export_offline_env defines every required offline variable and defaults
     HF_HOME to the shared project cache while preserving an explicit value;
  2. offline exports precede the first python/apptainer invocation in
     train.sh / eval.sh / setup.sh (for setup.sh: after the login relaunch);
  3. prepare_data is gone — no compute-side download/regeneration path;
  4. require_cluster_artifacts targets the real loader artifacts
     (dataset cache dir, gloss_vocab.txt, bigram_transition.npy, sidecars);
  5. W&B offline vars are exported and forwarded to apptainer (train/eval
     parity);
  6. setup.sh is verify-only (no pip install / no download of any kind);
  7. preflight.sh validates offline env/artifacts/dataset/model/W&B without
     training, with the optional T2G_PDA_FULL_VOCAB=1 gate under PDA=1;
  8. probe.sh follows the same relaunch/offline/artifact conventions;
  9. bash-4 compatibility (no bash-5-only parameter expansions).
"""

from __future__ import annotations

import re
from pathlib import Path

CLUSTER_DIR = Path(__file__).resolve().parent.parent / "cluster"

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
PYTHON_OR_APPTAINER_RE = re.compile(r"\b(python3?|apptainer)\b")
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


def test_offline_exports_precede_first_python_in_setup_after_relaunch():
    """setup.sh: scan starts AFTER the login-node srun relaunch (`set -e`)."""
    code = code_lines("setup.sh")
    relaunch = [ln for ln in code if "APPTAINER_CONTAINER" in ln]
    assert relaunch, "setup.sh must keep the login→srun+Apptainer relaunch"
    set_e_idx = first_index(code, re.compile(r"^set -e$"))
    assert set_e_idx is not None, "setup.sh must `set -e` after the relaunch"
    _assert_offline_exports_first("setup.sh", code[set_e_idx:])


def test_setup_relaunches_before_any_python_on_login_node():
    """The srun relaunch block must appear before any python usage: the login
    node has no python, so the only python/apptainer use is the relaunch."""
    code = code_lines("setup.sh")
    relaunch_idx = first_index(code, re.compile(r"APPTAINER_CONTAINER"))
    assert relaunch_idx is not None, "setup.sh must keep the relaunch block"
    assert (
        first_index(code[: relaunch_idx + 1], PYTHON_OR_APPTAINER_RE) is None
    ), "no python/apptainer may run before the login-node relaunch"
    assert (
        first_index(code[relaunch_idx + 1 :], PYTHON_OR_APPTAINER_RE) is not None
    ), "setup.sh verify step must reach python/apptainer after the relaunch"


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


def test_no_download_or_install_commands_in_cluster_scripts():
    """No pip install / wget / curl / git clone / hf_hub_download anywhere."""
    for name in ("_lib.sh", "train.sh", "eval.sh", "setup.sh", "preflight.sh"):
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
    for name in ("_lib.sh", "train.sh", "eval.sh", "setup.sh", "preflight.sh"):
        code = code_text(name)
        for token in forbidden:
            assert token not in code, f"{name}: must not call {token}()"


def test_train_and_eval_call_require_cluster_artifacts_with_config():
    for name in ("train.sh", "eval.sh"):
        code = code_text(name)
        assert (
            'require_cluster_artifacts "$CONFIG"' in code
        ), f"{name}: must fail fast on missing offline artifacts"
        assert "require_cluster_artifacts" in code_text(
            "setup.sh"
        ), "setup.sh verify step must call require_cluster_artifacts"


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


def test_train_and_eval_forward_offline_env_to_apptainer():
    """Apptainer must receive the offline env via explicit --env args."""
    for name in ("train.sh", "eval.sh"):
        code = code_text(name)
        for var in REQUIRED_OFFLINE_VARS:
            assert (
                f'--env "{var}=' in code
            ), f"{name}: apptainer must receive {var} via --env"
        assert (
            '--env "HF_HOME=${HF_HOME}"' in code
        ), f"{name}: apptainer must receive HF_HOME"
        assert (
            '--env "HF_HUB_CACHE=${HF_HUB_CACHE}"' in code
        ), f"{name}: apptainer must receive an explicit custom HF_HUB_CACHE"


def test_run_py_forwards_offline_env_to_apptainer():
    body = read_script("_lib.sh").split("run_py() {", 1)[1].split("\n}", 1)[0]
    for var in REQUIRED_OFFLINE_VARS:
        assert f'--env "{var}=' in body, f"run_py must forward {var} to apptainer"
    assert '--env "HF_HUB_CACHE=' in body


# ---------------------------------------------------------------------------
# 6. setup.sh is verify-only
# ---------------------------------------------------------------------------


def test_setup_is_verify_only():
    code = code_text_unquoted("setup.sh")
    assert "pip" not in code, "setup.sh must not touch pip"
    assert not re.search(
        r"\bprepare\b", code, re.I
    ), "setup.sh must not regenerate artifacts"
    # Local import/version verification is still there (multi-line quoted
    # python survives line-wise stripping → still visible).
    for module in ("torch", "transformers", "trl", "peft", "datasets", "sklearn"):
        assert module in code_text(
            "setup.sh"
        ), f"setup.sh must verify import of {module}"


def test_setup_prints_network_enabled_acquisition_instructions():
    src = read_script("setup.sh")
    assert "ambiente separato con rete" in src.lower() or (
        "ambiente con rete" in src.lower()
    ), "setup.sh must explain where acquisition happens instead"
    assert (
        "T2G_HF_HOME_DEFAULT" in src
    ), "setup.sh must point at the shared HF cache for uploads"


# ---------------------------------------------------------------------------
# 7. preflight.sh: validates everything, trains nothing
# ---------------------------------------------------------------------------


def test_preflight_exists_and_uses_slurm_conventions():
    src = read_script("preflight.sh")
    assert "#SBATCH" in src, "preflight must be sbatch-able"
    assert (
        "APPTAINER_CONTAINER" in src
    ), "preflight must follow the container relaunch conventions"
    assert (
        "srun" in src and "apptainer run --nv" in src
    ), "preflight must be usable via srun under container conventions"


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
    assert "rollouts|markov" in code
    assert 'ARGS=("$COMMAND" --config "$CONFIG" --input "$INPUT")' in code
    assert 'ARGS+=(--output "$OUTPUT")' in code
    assert code.count("run_py -m src.analysis") == 1
    assert "-m src.training" not in code
    assert not DOWNLOAD_FORBIDDEN_RE.search(code_text_unquoted("probe.sh"))


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
    ):
        src = read_script(name)
        match = BASH5_PARAM_EXPANSION_RE.search(src)
        assert match is None, f"{name}: bash-5-only expansion {match.group(0)!r}"


def test_scripts_are_quoted_paths_only():
    """Basic quoting hygiene: cd/source/require use quoted variables."""
    for name in ("train.sh", "eval.sh", "setup.sh", "preflight.sh", "probe.sh"):
        code = code_text(name)
        assert 'source "$SCRIPT_DIR/_lib.sh"' in code, f"{name}: quoted source"
        assert 'cd "$PROJ_DIR"' in code, f"{name}: quoted cd"
