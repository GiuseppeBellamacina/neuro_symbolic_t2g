"""CLI for local frozen-artifact probes."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

import yaml

from src.analysis.markov_diagnostics import load_markov_artifacts, run_markov_probe
from src.analysis.rollout_probe import (
    load_grouped_generations,
    run_rollout_probe,
    write_json_report,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _config(path: str) -> dict[str, Any]:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"probe config not found: {source}")
    value = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or not isinstance(value.get("probe"), dict):
        raise ValueError("config must contain a probe mapping")
    return value["probe"]


def _resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _repo_output(value: str) -> Path:
    output = _resolve(value)
    try:
        output.relative_to(PROJECT_ROOT)
    except ValueError as exc:
        raise ValueError(
            f"output_path must stay under repository root: {output}"
        ) from exc
    return output


def main(argv: list[str] | None = None) -> int:
    os.environ.update(
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        HF_DATASETS_OFFLINE="1",
        WANDB_MODE="disabled",
        WANDB_DISABLED="true",
    )
    parser = argparse.ArgumentParser(description="Offline frozen-generation probes")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("rollouts", "markov"):
        child = subparsers.add_parser(command)
        child.add_argument("--config", required=True)
        child.add_argument(
            "--input",
            help="Existing eval generations_*.json (required unless config sets input_path)",
        )
        child.add_argument("--output", help="Repository-local report path override")
    args = parser.parse_args(argv)
    config = _config(args.config)
    input_value = args.input or config.get("input_path")
    if not input_value:
        raise ValueError(
            "generation input is required: pass --input path/to/generations_*.json "
            "(cluster launcher: INPUT=path/to/generations_*.json)"
        )
    generations = _resolve(str(input_value))
    output_value = args.output or config.get("output_path")
    if not output_value:
        raise ValueError("output_path is required in config or via --output")
    output = _repo_output(str(output_value))
    rows = load_grouped_generations(generations, config.get("group_size"))

    if args.command == "rollouts":
        vocab = None
        if config.get("vocab_path"):
            vocab_path = _resolve(str(config["vocab_path"]))
            if not vocab_path.is_file():
                raise FileNotFoundError(f"vocabulary not found: {vocab_path}")
            vocab = vocab_path.read_text(encoding="utf-8").splitlines()
        report = run_rollout_probe(
            rows,
            vocab,
            low_threshold=float(config.get("low_threshold", 0.0)),
            high_threshold=float(config.get("high_threshold", 1.0)),
        )
    else:
        vocab, matrix = load_markov_artifacts(
            _resolve(str(config["vocab_path"])),
            _resolve(str(config["bigram_path"])),
        )
        report = run_markov_probe(
            rows,
            vocab,
            matrix,
            thresholds=config.get("thresholds"),
            max_states=config.get("max_states", 256),
        )
    write_json_report(report, output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
