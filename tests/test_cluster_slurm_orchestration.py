"""Focused guards for login-node-safe shell orchestration."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _text(relative: str) -> str:
    return (ROOT / relative).read_text(encoding="utf-8")


def test_login_aliases_restore_full_python_monitor_and_wrap_summary() -> None:
    aliases = _text("cluster/aliases.sh")
    monitor_body = aliases.split("monitor() {", 1)[1].split("\n}", 1)[0]
    summary_body = aliases.split("ablation-summary() {", 1)[1].split("\n}", 1)[0]

    assert (
        'cd "$PROJ_DIR" && python3 -u -m src.utils.chain_monitor "$@"' in monitor_body
    )
    assert "--samples [N]" in aliases
    assert "--metrics" in aliases
    assert "--all [N]" in aliases
    assert "srun " in summary_body
    assert "apptainer run" in summary_body
    assert summary_body.index("srun ") < summary_body.index("apptainer run")
    assert "squeue --me" in summary_body


def test_run_all_captures_tick_failure_under_errexit() -> None:
    run_all = _text("cluster/run_all.sh")

    assert "bash cluster/chain_tick.sh --quiet || rc=$?" in run_all


def test_resume_rebuild_preserves_train_then_eval_order() -> None:
    aliases = _text("cluster/aliases.sh")
    train_case = aliases.split("            train)\n", 1)[1].split(
        "                ;;", 1
    )[0]

    eval_insert = 'rebuild_chain "eval:${st_cfg}:${st_tag}"'
    train_insert = 'rebuild_chain "train:${st_cfg}:${st_tag}:--resume"'
    assert train_case.index(eval_insert) < train_case.index(train_insert)


def test_remote_monitor_maps_all_slurm_job_log_prefixes() -> None:
    helper = _text("remote/cluster_helper.sh")

    for job_name, log_prefix in (
        ("preflight-*", "preflight"),
        ("probe-*", "probe"),
        ("structured-*", "structured"),
        ("eval-*", "eval"),
        ("train-*", "train"),
    ):
        assert f"{job_name})" in helper
        assert f'prefix="{log_prefix}"' in helper


def test_remote_status_tolerates_squeue_failure() -> None:
    helper = _text("remote/cluster_helper.sh")
    status_body = helper.split("dump_status() {", 1)[1].split("\n}", 1)[0]

    assert "squeue --me -h -o '%A|%j|%T'" in status_body
    assert ") || true" in status_body
