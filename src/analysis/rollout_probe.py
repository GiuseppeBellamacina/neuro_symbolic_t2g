"""Deterministic, non-training diagnostics for frozen eval generations."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from src.analysis.markov_diagnostics import bigram_sequence_mean
from src.rewards.t2g_rewards import (
    REWARD_NAMES,
    reward_protocol,
    score_validity_reward,
)
from src.utils.text_utils import extract_gloss_text

_PERTURBATION_KINDS = ("delete", "duplicate", "reverse", "substitute")
_INVARIANT_PERTURBATIONS = {("token-f1-validity", "reverse")}
_DROP_EPSILON = 1e-12


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


def load_generation_artifact(
    path: str | Path,
    group_size: int | None = None,
    *,
    decoding_mode: str | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    """Load current eval JSON and assign a stable ``group_id`` to every row."""
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"generations file not found: {source}")
    content = source.read_bytes()
    try:
        payload = json.loads(content.decode("utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"malformed generations JSON: {source}: {exc.msg}") from exc
    metadata: dict[str, Any] = {}
    if isinstance(payload, dict):
        metadata = dict(payload.get("metadata") or {})
        payload = payload.get("generations", payload.get("rows"))
    if not isinstance(payload, list) or not payload:
        raise ValueError("generations JSON must be a nonempty list")
    if isinstance(payload[0], dict) and isinstance(
        payload[0].get("artifact_metadata"), dict
    ):
        metadata = dict(payload[0]["artifact_metadata"])
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
        if decoding_mode is not None and raw.get("decoding_mode") != decoding_mode:
            continue
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

    if not rows:
        raise ValueError(f"no generation rows match decoding_mode={decoding_mode!r}")

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
    provenance = {
        "path": str(source.resolve()),
        "sha256": hashlib.sha256(content).hexdigest(),
        "bytes": len(content),
        "artifact_rows": len(payload),
        "selected_rows": len(rows),
        "selected_groups": len(groups),
    }
    return rows, metadata, provenance


def load_grouped_generations(
    path: str | Path, group_size: int | None = None
) -> list[dict[str, Any]]:
    """Compatibility loader without decoding-mode filtering."""
    rows, _, _ = load_generation_artifact(path, group_size)
    return rows


def run_rollout_probe(
    rows: Sequence[Mapping[str, Any]],
    vocab: Collection[str] | None = None,
    *,
    low_threshold: float = 0.0,
    high_threshold: float = 1.0,
    thresholds: Mapping[str, float] | None = None,
    include_perturbations: bool = True,
    qualification_identity: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Score every single-reward candidate on one frozen rollout artifact."""
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
        score = edit_similarity(completion, gold) if valid and gold_tokens else 0.0
        reward_scores = {
            name: score_validity_reward(
                name,
                completion,
                gold,
                vocabulary if vocabulary is not None else set(output_tokens),
            )
            for name in REWARD_NAMES
        }
        item = {
            "score": score,
            "exact": output_tokens == gold_tokens,
            "valid": valid,
            "normalized": " ".join(output_tokens),
            "length_error": len(output_tokens) - len(gold_tokens),
            "gold_length": len(gold_tokens),
            "reward_scores": reward_scores,
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
    report = {
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
    report["reward_qualification"] = _reward_qualification(
        groups,
        thresholds or {},
        rows,
        vocabulary,
        include_perturbations,
        qualification_identity or {},
    )
    return report


def _reward_qualification(
    groups: Mapping[str, Sequence[Mapping[str, Any]]],
    thresholds: Mapping[str, float],
    rows: Sequence[Mapping[str, Any]],
    vocabulary: set[str] | None,
    include_perturbations: bool,
    identity: Mapping[str, Any],
) -> dict[str, Any]:
    defaults = {
        "max_zero_variance_fraction": 0.5,
        "max_all_min_fraction": 0.5,
        "min_unique_scores_mean": 1.5,
        "min_best_minus_mean": 0.05,
        "min_rankable_group_fraction": 0.5,
        "min_bucket_count": 1,
    }
    limits = {**defaults, **{key: float(value) for key, value in thresholds.items()}}
    rewards: dict[str, Any] = {}
    for name in REWARD_NAMES:
        values = [
            float(item["reward_scores"][name])
            for items in groups.values()
            for item in items
        ]
        group_values = [
            [float(item["reward_scores"][name]) for item in items]
            for items in groups.values()
        ]
        group_stds = [float(np.std(value)) for value in group_values]
        unique_mean = float(np.mean([len(set(value)) for value in group_values]))
        best_gain = float(
            np.mean([max(value) - float(np.mean(value)) for value in group_values])
        )
        all_min = float(
            np.mean([all(value == -1.0 for value in item) for item in group_values])
        )
        all_max = float(
            np.mean([all(value == 1.0 for value in item) for item in group_values])
        )
        checks = {
            "zero_variance": float(np.mean([value == 0.0 for value in group_stds]))
            <= limits["max_zero_variance_fraction"],
            "all_min": all_min <= limits["max_all_min_fraction"],
            "unique_scores": unique_mean >= limits["min_unique_scores_mean"],
            "best_minus_mean": best_gain >= limits["min_best_minus_mean"],
            "rankable_groups": float(
                np.mean([len(set(value)) > 1 for value in group_values])
            )
            >= limits["min_rankable_group_fraction"],
            "length_buckets": all(
                bucket["count"] >= limits["min_bucket_count"]
                for bucket in _reward_length_buckets(groups, name).values()
            ),
        }
        rewards[name] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "quantiles": {
                key: float(value)
                for key, value in zip(
                    ("p00", "p25", "p50", "p75", "p100"),
                    np.quantile(values, [0, 0.25, 0.5, 0.75, 1]),
                )
            },
            "zero_variance_groups": int(sum(value == 0.0 for value in group_stds)),
            "frac_zero_variance": float(
                np.mean([value == 0.0 for value in group_stds])
            ),
            "frac_all_min": all_min,
            "frac_all_max": all_max,
            "unique_scores": len(set(values)),
            "unique_scores_mean_per_group": unique_mean,
            "best_minus_mean": best_gain,
            "rankable_group_fraction": float(
                np.mean([len(set(value)) > 1 for value in group_values])
            ),
            "length_buckets": _reward_length_buckets(groups, name),
            "checks": checks,
            "passed": all(checks.values()),
        }

    correlations = {
        left: {
            right: _spearman(
                [
                    item["reward_scores"][left]
                    for items in groups.values()
                    for item in items
                ],
                [
                    item["reward_scores"][right]
                    for items in groups.values()
                    for item in items
                ],
            )
            for right in REWARD_NAMES
        }
        for left in REWARD_NAMES
    }
    perturbations = (
        _perturbation_report(rows, vocabulary) if include_perturbations else None
    )
    if perturbations is not None:
        for name, item in rewards.items():
            candidate = perturbations[name]
            required = [
                result
                for result in candidate.values()
                if result["required_for_qualification"]
                and result["evaluable_count"] > 0
            ]
            # At least one degradation probe must be evaluable, while
            # structurally impossible/no-op probes are explicitly non-gating.
            item["checks"]["perturbations"] = bool(required) and all(
                result["passed"] for result in required
            )
            item["passed"] = all(item["checks"].values())
    population_size = int(identity.get("group_size", 0))
    training_matches = {
        "group_size": population_size == 8,
        "decoding_mode": identity.get("decoding_population") == "sampling",
        "temperature": identity.get("temperature")
        == identity.get("required_temperature"),
        "prompt_mode": identity.get("prompt_mode") == "retrieval",
        "retrieval": identity.get("retrieval_enabled") is True,
        "trie": identity.get("trie_enabled") is True,
        "max_length": identity.get("max_completion_length")
        == identity.get("required_max_completion_length"),
    }
    eligible = all(training_matches.values())
    for item in rewards.values():
        item["eligible_to_authorize_training"] = eligible and item["passed"]
    return {
        "protocol": reward_protocol(),
        "identity": dict(identity),
        "training_match_checks": training_matches,
        "eligible_to_authorize_training": eligible,
        "thresholds": limits,
        "rewards": rewards,
        "pairwise_spearman": correlations,
        "perturbations": perturbations,
        "all_passed": all(item["passed"] for item in rewards.values()),
    }


def _length_bucket(length: int) -> str:
    if length <= 3:
        return "1-3"
    if length <= 7:
        return "4-7"
    return "8+"


def _reward_length_buckets(
    groups: Mapping[str, Sequence[Mapping[str, Any]]], name: str
) -> dict[str, Any]:
    buckets: dict[str, list[float]] = defaultdict(list)
    for items in groups.values():
        for item in items:
            buckets[_length_bucket(int(item["gold_length"]))].append(
                float(item["reward_scores"][name])
            )
    return {
        bucket: {
            "count": len(values),
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
        }
        for bucket, values in sorted(buckets.items())
    }


def _perturbation_report(
    rows: Sequence[Mapping[str, Any]], vocabulary: set[str] | None
) -> dict[str, Any] | None:
    references = {
        str(row.get("gold_gloss", ""))
        for row in rows
        if str(row.get("gold_gloss", "")).strip()
    }
    if not references:
        return None
    vocab = vocabulary or {
        token for reference in references for token in normalized_tokens(reference)
    }
    deltas: dict[str, dict[str, list[float]]] = {
        name: {kind: [] for kind in _PERTURBATION_KINDS} for name in REWARD_NAMES
    }
    attempted = {kind: 0 for kind in _PERTURBATION_KINDS}
    structurally_unevaluable = {kind: 0 for kind in _PERTURBATION_KINDS}
    no_ops = {kind: 0 for kind in _PERTURBATION_KINDS}
    for reference in sorted(references):
        tokens = normalized_tokens(reference)
        replacement = (
            next((token for token in sorted(vocab) if token != tokens[0]), None)
            if tokens
            else None
        )
        candidates: dict[str, list[str] | None] = {
            "delete": tokens[:-1] if len(tokens) > 1 else None,
            "duplicate": [*tokens, tokens[-1]] if tokens else None,
            "reverse": list(reversed(tokens)) if len(tokens) > 1 else None,
            "substitute": (
                [replacement, *tokens[1:]] if replacement is not None else None
            ),
        }
        for kind, candidate in candidates.items():
            attempted[kind] += 1
            if candidate is None:
                structurally_unevaluable[kind] += 1
            elif candidate == tokens:
                no_ops[kind] += 1
        for name in REWARD_NAMES:
            clean = score_validity_reward(name, reference, reference, vocab)
            for kind, candidate in candidates.items():
                if candidate is None or candidate == tokens:
                    continue
                changed = score_validity_reward(
                    name, " ".join(candidate), reference, vocab
                )
                deltas[name][kind].append(clean - changed)

    report: dict[str, Any] = {}
    for name, kinds in deltas.items():
        report[name] = {}
        for kind, values in sorted(kinds.items()):
            expected = (
                "invariant"
                if (name, kind) in _INVARIANT_PERTURBATIONS
                else "degradation"
            )
            mean_drop = float(np.mean(values)) if values else None
            positive_drop_rate = (
                float(np.mean([value > _DROP_EPSILON for value in values]))
                if values
                else None
            )
            required = expected == "degradation"
            passed = None
            if values and required:
                passed = bool(
                    mean_drop is not None
                    and mean_drop >= 0.0
                    and positive_drop_rate is not None
                    and positive_drop_rate > 0.0
                )
            report[name][kind] = {
                # ``count``, ``mean_drop``, and ``positive_drop_rate`` retain
                # their old meanings for consumers of v2 qualification JSON.
                "count": len(values),
                "mean_drop": mean_drop,
                "positive_drop_rate": positive_drop_rate,
                "expected_behavior": expected,
                "required_for_qualification": required,
                "attempted_count": attempted[kind],
                "evaluable_count": len(values),
                "structurally_unevaluable_count": structurally_unevaluable[kind],
                "no_op_count": no_ops[kind],
                "passed": passed,
            }
    return report


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
