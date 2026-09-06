"""Executable research-only frozen-feature structured head benchmark."""

from __future__ import annotations

import argparse
import json
import random
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import transformers
import yaml

from src.datasets.structured_transitions import (
    StructuredTransitionGraph,
    build_structured_transition_graph,
)
from src.models.structured_gloss_head import StructuredGlossHead, structured_nll
from src.training.structured_sft import (
    FrozenFeatureArtifact,
    assistant_boundary_indices,
    canonical_sample_id,
    feature_manifest,
    independent_position_ce,
    load_feature_artifact,
    render_source_prompt,
    save_feature_artifact,
)

ARMS = ("independent", "uniform-support", "true-weights", "shuffled-weights")


def reset_device_measurement(device: torch.device) -> float:
    """Reset per-phase CUDA accounting and return a synchronized start time."""
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    return time.perf_counter()


def finish_device_measurement(
    device: torch.device, started: float
) -> tuple[float, int]:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    runtime = time.perf_counter() - started
    peak = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
    return runtime, peak


def _reweight_graph(
    graph: StructuredTransitionGraph, counts: np.ndarray
) -> StructuredTransitionGraph:
    smoothed = counts.astype(np.float64) + graph.alpha
    denominators = {
        source: smoothed[graph.edge_src == source].sum()
        for source in np.unique(graph.edge_src)
    }
    log_prob = np.asarray(
        [
            np.log(value / denominators[source])
            for source, value in zip(graph.edge_src, smoothed)
        ],
        dtype=np.float32,
    )
    return StructuredTransitionGraph(
        graph.states,
        graph.edge_src.copy(),
        graph.edge_dst.copy(),
        counts,
        log_prob,
        graph.alpha,
        graph.train_sample_ids,
        graph.train_hash,
    )


def uniform_support_control(
    graph: StructuredTransitionGraph,
) -> StructuredTransitionGraph:
    """Uniform outgoing probability over the observed support."""
    control = _reweight_graph(graph, np.ones_like(graph.edge_count))
    if np.array_equal(control.edge_log_prob, graph.edge_log_prob):
        raise ValueError("uniform-support control did not change transition weights")
    return control


def shuffled_weight_control(
    graph: StructuredTransitionGraph, seed: int
) -> StructuredTransitionGraph:
    """Shuffle edge counts while retaining support; reject a null control."""
    counts = np.random.default_rng(seed).permutation(graph.edge_count)
    if np.array_equal(counts, graph.edge_count) and len(counts) > 1:
        for shift in range(1, len(counts)):
            candidate = np.roll(graph.edge_count, shift)
            if not np.array_equal(candidate, graph.edge_count):
                counts = candidate
                break
    control = _reweight_graph(graph, counts)
    if np.array_equal(control.edge_log_prob, graph.edge_log_prob):
        raise ValueError("shuffled-weights control did not change transition weights")
    return control


def transition_control_diagnostics(
    graph: StructuredTransitionGraph, control: StructuredTransitionGraph
) -> dict[str, float]:
    delta = control.edge_log_prob - graph.edge_log_prob
    return {
        "fraction_log_weights_changed": float(np.mean(delta != 0)),
        "mean_absolute_log_weight_delta": float(np.mean(np.abs(delta))),
        "max_absolute_log_weight_delta": float(np.max(np.abs(delta))),
    }


def filter_gloss_lengths(rows, max_length):
    """Reject empty rows and deterministically exclude overflow before capping."""
    rows = [dict(row) for row in rows]
    lengths = np.asarray([len(str(row.get("gloss", "")).split()) for row in rows])
    if np.any(lengths == 0):
        raise ValueError("empty gloss paths are unsupported")
    excluded = [row for row, length in zip(rows, lengths) if length > max_length]
    retained = [row for row, length in zip(rows, lengths) if length <= max_length]
    percentiles = np.percentile(lengths, [50, 90, 95, 99, 100]).tolist()
    return retained, {
        "total_before_filter": len(rows),
        "excluded_count": len(excluded),
        "excluded_fraction": len(excluded) / len(rows) if rows else 0.0,
        "excluded_sample_ids": [canonical_sample_id(row) for row in excluded],
        "max_length_observed": int(lengths.max()) if len(lengths) else 0,
        "length_percentiles": dict(
            zip(("p50", "p90", "p95", "p99", "p100"), percentiles)
        ),
        "max_gloss_length": max_length,
    }


def _load_config(path: str | Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as handle:
        return yaml.safe_load(handle)["probe"]


def _rows_from_cached_dataset(config: dict[str, Any]):
    from src.datasets.aslg_dataset import download_aslg_dataset

    dataset = download_aslg_dataset(
        cache_dir=config["dataset_cache"], seed=config["seed"], online=False
    )
    eligible, length_filter = filter_gloss_lengths(
        dataset["train"], config["max_gloss_length"]
    )
    from datasets import Dataset

    split = Dataset.from_list(eligible).train_test_split(
        test_size=config["dev_fraction"], seed=config["seed"]
    )
    train = [
        dict(row)
        for row in split["train"].select(
            range(min(config["train_samples"], len(split["train"])))
        )
    ]
    dev = [
        dict(row)
        for row in split["test"].select(
            range(min(config["dev_samples"], len(split["test"])))
        )
    ]
    return train, dev, length_filter


def extract_frozen_features(
    config: dict[str, Any], output: str | Path
) -> FrozenFeatureArtifact:
    """Extract only final-layer boundary vectors with a frozen standard HF model."""
    from transformers import AutoModel, AutoTokenizer

    train_rows, dev_rows, length_filter = _rows_from_cached_dataset(config)
    tokenizer = AutoTokenizer.from_pretrained(
        config["model_name"], local_files_only=True
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    model = AutoModel.from_pretrained(
        config["model_name"], local_files_only=True, torch_dtype=dtype
    )
    model.requires_grad_(False).eval()
    model.to(device)
    started = reset_device_measurement(device)

    def encode(rows):
        batches = []
        for start in range(0, len(rows), config["feature_batch_size"]):
            prompts = [
                render_source_prompt(row["text"], tokenizer)
                for row in rows[start : start + config["feature_batch_size"]]
            ]
            tokens = tokenizer(
                prompts, padding=True, truncation=True, return_tensors="pt"
            )
            tokens = {key: value.to(device) for key, value in tokens.items()}
            with torch.inference_mode():
                output = model(**tokens, output_hidden_states=False, return_dict=True)
                boundary = output.last_hidden_state[
                    torch.arange(len(prompts), device=device),
                    assistant_boundary_indices(tokens["attention_mask"]),
                ]
            batches.append(boundary.float().cpu().numpy())
        return np.concatenate(batches)

    train_features, dev_features = encode(train_rows), encode(dev_rows)
    runtime, peak_memory = finish_device_measurement(device, started)
    extraction = {
        "runtime_seconds": runtime,
        "peak_memory_bytes": peak_memory,
        "device": str(device),
        "dtype": str(dtype),
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "device_total_memory_bytes": (
            torch.cuda.get_device_properties(device).total_memory
            if device.type == "cuda"
            else 0
        ),
    }
    manifest = feature_manifest(
        train_rows,
        dev_rows,
        model_name=config["model_name"],
        hidden_size=train_features.shape[1],
        length_filter=length_filter,
        extraction=extraction,
    )
    artifact = FrozenFeatureArtifact(
        train_features, dev_features, tuple(train_rows), tuple(dev_rows), manifest
    )
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    save_feature_artifact(artifact, output)
    return artifact


def _batch_targets(rows, graph, max_length):
    paths = [graph.map_glosses(row["gloss"]) for row in rows]
    lengths = torch.tensor([len(path) for path in paths], dtype=torch.long)
    if torch.any(lengths < 1) or torch.any(lengths > max_length):
        raise ValueError("all gloss paths must lie in [1, max_gloss_length]")
    targets = torch.zeros((len(paths), max_length), dtype=torch.long)
    for index, path in enumerate(paths):
        targets[index, : len(path)] = torch.tensor(path)
    return targets, lengths


def _edit_distance(left: list[int], right: list[int]) -> int:
    previous = list(range(len(right) + 1))
    for i, a in enumerate(left, 1):
        current = [i]
        for j, b in enumerate(right, 1):
            current.append(
                min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (a != b))
            )
        previous = current
    return previous[-1]


def _metrics(emissions, targets, lengths, graph):
    predictions = emissions.argmax(dim=-1)
    mask = torch.arange(emissions.shape[1])[None, :] < lengths[:, None]
    accuracy = ((predictions == targets) & mask).sum().item() / mask.sum().item()
    edits = []
    exact = []
    for index, length in enumerate(lengths.tolist()):
        gold = targets[index, :length].tolist()
        predicted = predictions[index, :length].tolist()
        edits.append(_edit_distance(predicted, gold) / max(len(gold), 1))
        exact.append(predicted == gold)
    return {
        "emission_accuracy": accuracy,
        "sequence_exact": float(np.mean(exact)),
        "normalized_edit_distance": float(np.mean(edits)),
        "path_coverage": float(
            np.mean(
                [
                    all(
                        graph.has_edge(a, b)
                        for a, b in zip(
                            [graph.bos_index, *targets[i, :n].tolist()],
                            [*targets[i, :n].tolist(), graph.eos_index],
                        )
                    )
                    for i, n in enumerate(lengths.tolist())
                ]
            )
        ),
    }


def train_head_arm(
    arm: str,
    artifact: FrozenFeatureArtifact,
    graph: StructuredTransitionGraph,
    config: dict[str, Any],
) -> tuple[StructuredGlossHead, dict[str, Any]]:
    """Train one head; features are immutable and shared across all arms."""
    if arm not in ARMS:
        raise ValueError(f"unknown arm: {arm}")
    torch.manual_seed(config["seed"])
    device = torch.device(
        config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    )
    if arm == "shuffled-weights":
        active_graph = shuffled_weight_control(graph, config["seed"])
    elif arm == "uniform-support":
        active_graph = uniform_support_control(graph)
    else:
        active_graph = graph
    train_x = torch.from_numpy(artifact.train_features).float().to(device)
    dev_x = torch.from_numpy(artifact.dev_features).float().to(device)
    train_y, train_lengths = _batch_targets(
        artifact.train_rows, graph, config["max_gloss_length"]
    )
    dev_y, dev_lengths = _batch_targets(
        artifact.dev_rows, graph, config["max_gloss_length"]
    )
    train_y, train_lengths = train_y.to(device), train_lengths.to(device)
    dev_y, dev_lengths = dev_y.to(device), dev_lengths.to(device)
    head = StructuredGlossHead(
        train_x.shape[1], graph.num_states, config["max_gloss_length"]
    ).to(device)
    optimizer = torch.optim.AdamW(
        head.parameters(), lr=config["head_lr"], weight_decay=config["weight_decay"]
    )
    started = reset_device_measurement(device)
    head.train()
    for _ in range(config["steps"]):
        emissions = head(train_x, config["max_gloss_length"])
        loss = (
            independent_position_ce(emissions, train_y, train_lengths).mean()
            if arm == "independent"
            else structured_nll(
                emissions,
                train_y,
                train_lengths,
                active_graph,
                config["transition_scale"],
            ).mean()
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    runtime, peak_memory = finish_device_measurement(device, started)
    head.eval()
    with torch.no_grad():
        train_emissions = head(train_x, config["max_gloss_length"])
        dev_emissions = head(dev_x, config["max_gloss_length"])
        report = {
            "arm": arm,
            "feature_sample_hash": artifact.manifest["sample_hash"],
            "graph_hash": graph.manifest()["graph_hash"],
            "active_graph_hash": active_graph.manifest()["graph_hash"],
            "train_structured_nll": structured_nll(
                train_emissions,
                train_y,
                train_lengths,
                active_graph,
                config["transition_scale"],
            )
            .mean()
            .item(),
            "dev_structured_nll": structured_nll(
                dev_emissions,
                dev_y,
                dev_lengths,
                active_graph,
                config["transition_scale"],
            )
            .mean()
            .item(),
            "train_independent_ce": independent_position_ce(
                train_emissions, train_y, train_lengths
            )
            .mean()
            .item(),
            "dev_independent_ce": independent_position_ce(
                dev_emissions, dev_y, dev_lengths
            )
            .mean()
            .item(),
            "runtime_seconds": runtime,
            "peak_memory_bytes": peak_memory,
            "device": str(device),
            "device_total_memory_bytes": (
                torch.cuda.get_device_properties(device).total_memory
                if device.type == "cuda"
                else 0
            ),
            "torch_version": torch.__version__,
            "transformers_version": transformers.__version__,
            "length_conditioned": True,
            "control_diagnostics": transition_control_diagnostics(graph, active_graph),
            **_metrics(
                dev_emissions.cpu(), dev_y.cpu(), dev_lengths.cpu(), active_graph
            ),
        }
    return head, report


def run_benchmark(
    config: dict[str, Any], features: str | Path, output_root: str | Path | None = None
):
    artifact = load_feature_artifact(features)
    random.seed(config["seed"])
    graph = build_structured_transition_graph(
        artifact.train_rows, top_k=config["top_k"], alpha=config["alpha"]
    )
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    root = Path(output_root or config["output_root"])
    reports = {}
    for arm in ARMS:
        head, report = train_head_arm(arm, artifact, graph, config)
        run_dir = root / arm / f"run_{timestamp}"
        run_dir.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": head.state_dict(),
                "head": {
                    "hidden_size": artifact.train_features.shape[1],
                    "num_states": graph.num_states,
                    "max_length": config["max_gloss_length"],
                },
                "config": config,
            },
            run_dir / "head.pt",
        )
        (run_dir / "report.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n"
        )
        reports[arm] = report
    comparison = {
        "feature_sample_hash": artifact.manifest["sample_hash"],
        "graph_hash": graph.manifest()["graph_hash"],
        "true_vs_shuffled_dev_nll_delta": reports["shuffled-weights"][
            "dev_structured_nll"
        ]
        - reports["true-weights"]["dev_structured_nll"],
        "pilot_only": True,
        "seeds_completed": [config["seed"]],
        "promotion_gate": {
            "passed": False,
            "reason": "single-seed pilot; promotion requires true-weights to beat independent, uniform-support, and shuffled-weights across configured seeds",
            "auto_promote": False,
        },
        "resource_gate": {
            "max_arm_peak_memory_bytes": max(
                report["peak_memory_bytes"] for report in reports.values()
            ),
            "max_arm_runtime_seconds": max(
                report["runtime_seconds"] for report in reports.values()
            ),
            "passed": max(report["peak_memory_bytes"] for report in reports.values())
            <= config["max_peak_memory_bytes"]
            and max(report["runtime_seconds"] for report in reports.values())
            <= config["max_arm_runtime_seconds"],
            "auto_promote": False,
        },
    }
    (root / f"comparison_{timestamp}.json").write_text(
        json.dumps(comparison, indent=2, sort_keys=True) + "\n"
    )
    return reports, comparison


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("extract", "run"))
    parser.add_argument("--config", required=True)
    parser.add_argument("--features", required=True)
    parser.add_argument("--output")
    args = parser.parse_args(argv)
    config = _load_config(args.config)
    if args.command == "extract":
        extract_frozen_features(config, args.features)
    else:
        run_benchmark(config, args.features, args.output)


if __name__ == "__main__":
    main()
