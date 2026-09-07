"""Pure preprocessing and loss-composition helpers for structured SFT experiments."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor

from src.datasets.structured_transitions import StructuredTransitionGraph
from src.utils.prompting import build_t2g_prompt


def render_source_prompt(text: str, tokenizer: Any) -> str:
    """Render through the canonical generation-boundary prompt contract."""
    return build_t2g_prompt(text, tokenizer)


def assistant_boundary_indices(attention_mask: Tensor) -> Tensor:
    """Return the final non-padding prompt token, supporting left/right padding."""
    if attention_mask.ndim != 2:
        raise ValueError("attention_mask must have shape [B,S]")
    positions = torch.arange(attention_mask.shape[1], device=attention_mask.device)
    masked = positions[None, :].masked_fill(attention_mask == 0, -1)
    indices = masked.max(dim=1).values
    if torch.any(indices < 0):
        raise ValueError("each prompt must contain at least one token")
    return indices


def map_whitespace_glosses(
    glosses: str | Sequence[str], graph: StructuredTransitionGraph
) -> list[int]:
    """Whitespace-tokenize and map known long-tail/unseen tokens to OTHER."""
    return graph.map_glosses(glosses)


@dataclass(frozen=True)
class MappedGlossBatch:
    """Complete, untruncated mapped paths and their source-row alignment."""

    states: Tensor
    lengths: Tensor
    kept_indices: tuple[int, ...]
    excluded_indices: tuple[int, ...]


def map_complete_gloss_sequences(
    glosses: Sequence[str | Sequence[str]],
    graph: StructuredTransitionGraph,
    *,
    max_length: int,
    overlength: str = "error",
) -> MappedGlossBatch:
    """Map whole glosses, either failing or excluding overlength examples.

    Truncation is deliberately unsupported because it changes the gold path and
    its EOS transition. Empty glosses are rejected by the structured objective.
    """
    if max_length < 1:
        raise ValueError("max_length must be positive")
    if overlength not in {"error", "exclude"}:
        raise ValueError("overlength must be 'error' or 'exclude'")
    mapped: list[list[int]] = []
    kept: list[int] = []
    excluded: list[int] = []
    for index, gloss in enumerate(glosses):
        path = graph.map_glosses(gloss)
        if not path:
            raise ValueError(f"gloss {index} is empty")
        if len(path) > max_length:
            if overlength == "error":
                raise ValueError(
                    f"gloss {index} length {len(path)} exceeds max_length {max_length}"
                )
            excluded.append(index)
            continue
        mapped.append(path)
        kept.append(index)
    states = torch.zeros((len(mapped), max_length), dtype=torch.long)
    lengths = torch.tensor([len(path) for path in mapped], dtype=torch.long)
    for row, path in enumerate(mapped):
        states[row, : len(path)] = torch.tensor(path, dtype=torch.long)
    return MappedGlossBatch(states, lengths, tuple(kept), tuple(excluded))


def independent_position_ce(
    emissions: Tensor, gold_states: Tensor, lengths: Tensor
) -> Tensor:
    """Independent-position source-only control, normalized per gloss."""
    losses = torch.nn.functional.cross_entropy(
        emissions.transpose(1, 2), gold_states, reduction="none"
    )
    mask = (
        torch.arange(emissions.shape[1], device=emissions.device)[None, :]
        < lengths[:, None]
    )
    return (losses * mask).sum(dim=1) / lengths.to(losses.dtype)


def structured_weight(step: int, target: float, warmup_steps: int) -> float:
    """Linear structured-loss warmup as a side-effect-free scalar function."""
    if step < 0 or target < 0 or warmup_steps < 0:
        raise ValueError("step, target, and warmup_steps must be nonnegative")
    return target if warmup_steps == 0 else target * min(step / warmup_steps, 1.0)


def combine_lm_structured_loss(
    lm_loss: Tensor,
    structured_loss: Tensor,
    *,
    step: int,
    structured_lambda: float,
    warmup_steps: int,
) -> Tensor:
    weight = structured_weight(step, structured_lambda, warmup_steps)
    return lm_loss + weight * structured_loss


def canonical_sample_id(row: dict[str, Any]) -> str:
    """Stable feature/sample identity independent of row order metadata."""
    explicit = row.get("sample_id", row.get("id"))
    if explicit is not None:
        return str(explicit)
    payload = f"{row.get('text', '')}||{row.get('gloss', '')}"
    return hashlib.sha256(payload.encode()).hexdigest()


@dataclass(frozen=True)
class FrozenFeatureArtifact:
    """Frozen assistant-boundary features and aligned raw examples."""

    train_features: np.ndarray
    dev_features: np.ndarray
    train_rows: tuple[dict[str, Any], ...]
    dev_rows: tuple[dict[str, Any], ...]
    manifest: dict[str, Any]


def feature_manifest(
    train_rows: Sequence[dict[str, Any]],
    dev_rows: Sequence[dict[str, Any]],
    *,
    model_name: str,
    hidden_size: int,
    length_filter: dict[str, Any] | None = None,
    extraction: dict[str, Any] | None = None,
) -> dict[str, Any]:
    train_ids = [canonical_sample_id(row) for row in train_rows]
    dev_ids = [canonical_sample_id(row) for row in dev_rows]
    payload = {"train_sample_ids": train_ids, "dev_sample_ids": dev_ids}
    return {
        "format": 1,
        "model_name": model_name,
        "hidden_size": hidden_size,
        **payload,
        "sample_hash": hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "backbone_frozen": True,
        "output_hidden_states": False,
        "length_filter": length_filter or {},
        "extraction": extraction or {},
    }


def save_feature_artifact(artifact: FrozenFeatureArtifact, path: str | Path) -> None:
    """Save explicit features; rows and manifest contain no model objects."""
    np.savez_compressed(
        path,
        train_features=artifact.train_features,
        dev_features=artifact.dev_features,
        train_rows=np.asarray(
            [json.dumps(row, sort_keys=True) for row in artifact.train_rows]
        ),
        dev_rows=np.asarray(
            [json.dumps(row, sort_keys=True) for row in artifact.dev_rows]
        ),
        manifest=np.asarray(json.dumps(artifact.manifest, sort_keys=True)),
    )


def load_feature_artifact(path: str | Path) -> FrozenFeatureArtifact:
    with np.load(path, allow_pickle=False) as data:
        artifact = FrozenFeatureArtifact(
            data["train_features"].copy(),
            data["dev_features"].copy(),
            tuple(json.loads(str(row)) for row in data["train_rows"]),
            tuple(json.loads(str(row)) for row in data["dev_rows"]),
            json.loads(str(data["manifest"])),
        )
    expected = feature_manifest(
        artifact.train_rows,
        artifact.dev_rows,
        model_name=artifact.manifest["model_name"],
        hidden_size=artifact.train_features.shape[1],
        length_filter=artifact.manifest.get("length_filter", {}),
        extraction=artifact.manifest.get("extraction", {}),
    )
    if expected != artifact.manifest:
        raise ValueError("feature artifact manifest mismatch")
    if artifact.dev_features.shape[1] != artifact.train_features.shape[1]:
        raise ValueError("train/dev feature dimensions differ")
    return artifact
