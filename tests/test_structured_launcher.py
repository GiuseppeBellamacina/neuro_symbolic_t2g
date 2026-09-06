from pathlib import Path


def test_launcher_sets_offline_before_python_and_extracts_before_run():
    script = Path("cluster/structured_probe.sh").read_text()
    offline = script.index("export_offline_env")
    extract = script.index("structured_benchmark extract")
    run = script.index("structured_benchmark run")
    assert offline < extract < run
    assert "--gres=gpu:1 --gres=shard:22528" in script
    assert "--mem=48G" in script
    assert "pip install" not in script
