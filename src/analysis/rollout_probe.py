"""Deterministic, non-training diagnostics for frozen eval generations."""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from src.analysis.markov_diagnostics import bigram_sequence_mean
from src.utils.text_utils import extract_gloss_text


@dataclass(frozen=True)
class OutputProbe:
    group: str
    output: str
    edit_score: float
    length: int
    bigram_score: float


@dataclass(frozen=True)
class GroupProbe:
    group: str
    count: int
    edit_bigram_rank_correlation: float
    edit_length_rank_correlation: float


def normalized_tokens(text: str) -> list[str]:
    return extract_gloss_text(text).casefold().split()


def edit_similarity(output: str, gold: str) -> float:
    left, right = normalized_tokens(output), normalized_tokens(gold)
    longest = max(len(left), len(right))
    if not longest:
        return 1.0
    previous = list(range(len(right) + 1))
    for row, token in enumerate(left, 1):
        current = [row]
        for column, target in enumerate(right, 1):
            current.append(
                min(
                    previous[column] + 1,
                    current[column - 1] + 1,
                    previous[column - 1] + (token != target),
                )
            )
        previous = current
    return 1.0 - previous[-1] / longest


def load_grouped_generations(
    path: str | Path, group_size: int | None = None
) -> list[dict[str, Any]]:
    """Load current eval JSON and assign a stable ``group_id`` to every row."""
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"generations file not found: {source}")
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"malformed generations JSON: {source}: {exc.msg}") from exc
    if not isinstance(payload, list) or not payload:
        raise ValueError("generations JSON must be a nonempty list")
    if group_size is not None and (
        isinstance(group_size, bool)
        or not isinstance(group_size, int)
        or group_size < 1
    ):
        raise ValueError("group_size must be a positive integer")

    rows: list[dict[str, Any]] = []
    for position, raw in enumerate(payload):
        if not isinstance(raw, dict):
            raise ValueError(f"generation row {position} must be an object")
        for key in ("completion", "gold_gloss"):
            if not isinstance(raw.get(key), str):
                raise ValueError(f"generation row {position} has no string {key!r}")
        explicit = raw.get("group_id", raw.get("sample_id"))
        if explicit is None:
            if group_size is None:
                raise ValueError(
                    f"generation row {position} has no sample_id/group_id and no group_size"
                )
            explicit = position // group_size
        row = dict(raw)
        row["group_id"] = str(explicit)
        rows.append(row)

    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[row["group_id"]].append(row)
    for group_id, members in groups.items():
        golds = {member["gold_gloss"] for member in members}
        if len(golds) != 1:
            raise ValueError(
                f"group {group_id!r} contains inconsistent gold_gloss values"
            )
        if group_size is not None and len(members) != group_size:
            raise ValueError(
                f"group {group_id!r} has {len(members)} rows; expected {group_size}"
            )
    return rows


def run_rollout_probe(
    rows: Sequence[Mapping[str, Any]],
    vocab: Collection[str] | None = None,
    *,
    low_threshold: float = 0.0,
    high_threshold: float = 1.0,
) -> dict[str, Any]:
    """Compute rollout-support metrics without importing training rewards."""
    if not rows:
        raise ValueError("at least one generation is required")
    vocabulary = {token.casefold() for token in vocab} if vocab is not None else None
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    all_scores: list[float] = []
    all_exact: list[bool] = []
    all_valid: list[bool] = []
    signed_errors: list[int] = []

    for position, row in enumerate(rows):
        try:
            group_id = str(row["group_id"])
            completion, gold = str(row["completion"]), str(row["gold_gloss"])
        except KeyError as exc:
            raise ValueError(
                f"generation row {position} missing {exc.args[0]!r}"
            ) from exc
        output_tokens, gold_tokens = normalized_tokens(completion), normalized_tokens(
            gold
        )
        valid = bool(output_tokens) and (
            vocabulary is None or all(token in vocabulary for token in output_tokens)
        )
        score = edit_similarity(completion, gold) if valid else 0.0
        item = {
            "score": score,
            "exact": output_tokens == gold_tokens,
            "valid": valid,
            "normalized": " ".join(output_tokens),
            "length_error": len(output_tokens) - len(gold_tokens),
            "gold_length": len(gold_tokens),
        }
        groups[group_id].append(item)
        all_scores.append(score)
        all_exact.append(item["exact"])
        all_valid.append(valid)
        signed_errors.append(item["length_error"])

    summaries = []
    best_gains = []
    oracle_hits = []
    for group_id in sorted(groups):
        items = groups[group_id]
        scores = [item["score"] for item in items]
        mean = float(np.mean(scores))
        best = max(scores)
        best_gains.append(best - mean)
        oracle_hits.append(any(item["exact"] for item in items))
        summaries.append(
            {
                "group_id": group_id,
                "size": len(items),
                "reward_mean": mean,
                "reward_std": float(np.std(scores)),
                "all_low": all(score <= low_threshold for score in scores),
                "all_high": all(score >= high_threshold for score in scores),
                "unique_normalized_outputs": len(
                    {item["normalized"] for item in items}
                ),
                "oracle_reward": best,
                "best_minus_mean": best - mean,
            }
        )

    def bucket(length: int) -> str:
        if length <= 3:
            return "1-3"
        if length <= 7:
            return "4-7"
        return "8+"

    bucket_values: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for items in groups.values():
        bucket_values[bucket(items[0]["gold_length"])].extend(items)
    buckets = {
        name: {
            "count": len(items),
            "reward_mean": float(np.mean([item["score"] for item in items])),
            "exact_match_rate": float(np.mean([item["exact"] for item in items])),
            "mean_absolute_length_error": float(
                np.mean([abs(item["length_error"]) for item in items])
            ),
        }
        for name, items in sorted(bucket_values.items())
    }
    group_stds = [summary["reward_std"] for summary in summaries]
    return {
        "schema_version": 1,
        "probe": "rollout_support",
        "num_groups": len(groups),
        "num_outputs": len(rows),
        "group_reward_mean": float(np.mean(all_scores)),
        "group_reward_std": float(np.std(all_scores)),
        "frac_zero_variance": float(np.mean([value == 0.0 for value in group_stds])),
        "frac_all_low": float(np.mean([item["all_low"] for item in summaries])),
        "frac_all_high": float(np.mean([item["all_high"] for item in summaries])),
        "unique_normalized_outputs_mean": float(
            np.mean([item["unique_normalized_outputs"] for item in summaries])
        ),
        "exact_match_rate": float(np.mean(all_exact)),
        "validity_rate": float(np.mean(all_valid)),
        "oov_rate": float(np.mean([not value for value in all_valid])),
        "signed_length_error_mean": float(np.mean(signed_errors)),
        "absolute_length_error_mean": float(np.mean(np.abs(signed_errors))),
        "reward_quantiles": {
            key: float(value)
            for key, value in zip(
                ("p00", "p25", "p50", "p75", "p100"),
                np.quantile(all_scores, [0, 0.25, 0.5, 0.75, 1]),
            )
        },
        "best_minus_mean": float(np.mean(best_gains)),
        "oracle_at_g_exact_match": float(np.mean(oracle_hits)),
        "gold_length_buckets": buckets,
        "groups": summaries,
    }


def write_json_report(report: Mapping[str, Any], path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(
            report, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False
        )
        + "\n",
        encoding="utf-8",
    )


def qualify_grouped_outputs(
    generations: Mapping[str, Sequence[str]],
    references: Mapping[str, str],
    vocab: Collection[str],
    transition_matrix: np.ndarray,
) -> tuple[list[OutputProbe], list[GroupProbe]]:
    """Compatibility API for fixed-output Markov qualification."""
    token_to_index = {token.casefold(): index for index, token in enumerate(vocab)}
    bos, eos = token_to_index.get("<bos>"), token_to_index.get("<eos>")
    outputs: list[OutputProbe] = []
    groups: list[GroupProbe] = []
    for group, values in generations.items():
        if group not in references:
            raise ValueError(f"missing reference for group {group!r}")
        current = []
        for output in values:
            tokens = normalized_tokens(output)
            path = (
                ([bos] if bos is not None else [])
                + [token_to_index[token] for token in tokens if token in token_to_index]
                + ([eos] if eos is not None else [])
            )
            bigram = (
                bigram_sequence_mean(transition_matrix, path)
                if len(path) >= 2
                else -100.0
            )
            valid = bool(tokens) and all(token in token_to_index for token in tokens)
            probe = OutputProbe(
                group,
                output,
                2 * edit_similarity(output, references[group]) - 1 if valid else -1.0,
                len(tokens),
                bigram,
            )
            outputs.append(probe)
            current.append(probe)
        edits = [item.edit_score for item in current]
        groups.append(
            GroupProbe(
                group,
                len(current),
                _spearman(edits, [item.bigram_score for item in current]),
                _spearman(edits, [float(item.length) for item in current]),
            )
        )
    return outputs, groups


def _spearman(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) < 2 or any(not np.isfinite(value) for value in (*left, *right)):
        return 0.0
    left_ranks, right_ranks = _ranks(left), _ranks(right)
    if np.std(left_ranks) == 0 or np.std(right_ranks) == 0:
        return 0.0
    return float(np.corrcoef(left_ranks, right_ranks)[0, 1])


def _ranks(values: Sequence[float]) -> np.ndarray:
    result = np.empty(len(values), dtype=float)
    order = np.argsort(values, kind="stable")
    position = 0
    while position < len(order):
        end = position + 1
        while end < len(order) and values[order[end]] == values[order[position]]:
            end += 1
        result[order[position:end]] = (position + end - 1) / 2
        position = end
    return result
