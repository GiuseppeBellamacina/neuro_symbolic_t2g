"""Reduced-state transition artifacts for the structured gloss prototype."""

from __future__ import annotations

import hashlib
import io
import json
import zipfile
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

OTHER = "<OTHER>"
BOS = "<BOS>"
EOS = "<EOS>"


def _canonical_hash(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class StructuredTransitionGraph:
    """Sparse graph. Emission states precede graph-only BOS and EOS nodes."""

    states: tuple[str, ...]
    edge_src: np.ndarray
    edge_dst: np.ndarray
    edge_count: np.ndarray
    edge_log_prob: np.ndarray
    alpha: float
    train_sample_ids: tuple[str, ...]
    train_hash: str

    @property
    def num_states(self) -> int:
        return len(self.states)

    @property
    def bos_index(self) -> int:
        return self.num_states

    @property
    def eos_index(self) -> int:
        return self.num_states + 1

    @property
    def token_to_index(self) -> dict[str, int]:
        return {token: index for index, token in enumerate(self.states)}

    def map_glosses(self, glosses: str | Sequence[str]) -> list[int]:
        tokens = glosses.split() if isinstance(glosses, str) else list(glosses)
        lookup = self.token_to_index
        other = lookup[OTHER]
        return [lookup.get(token, other) for token in tokens]

    def has_edge(self, source: int, destination: int) -> bool:
        return bool(np.any((self.edge_src == source) & (self.edge_dst == destination)))

    def manifest(self) -> dict[str, Any]:
        graph_hash = _canonical_hash(
            {
                "states": self.states,
                "src": self.edge_src.tolist(),
                "dst": self.edge_dst.tolist(),
                "count": self.edge_count.tolist(),
                "log_prob": [float(x).hex() for x in self.edge_log_prob],
                "alpha": self.alpha,
                "train_hash": self.train_hash,
            }
        )
        return {
            "format": 1,
            "states": list(self.states),
            "other_index": self.token_to_index[OTHER],
            "bos_index": self.bos_index,
            "eos_index": self.eos_index,
            "alpha": self.alpha,
            "train_sample_ids": list(self.train_sample_ids),
            "train_hash": self.train_hash,
            "graph_hash": graph_hash,
        }


def _row_id(row: Mapping[str, Any]) -> str:
    explicit = row.get("id", row.get("sample_id"))
    return str(explicit) if explicit is not None else _canonical_hash(dict(row))


def build_structured_transition_graph(
    train_rows: Sequence[Mapping[str, Any]],
    *,
    top_k: int = 512,
    alpha: float = 0.1,
    gloss_key: str = "gloss",
) -> StructuredTransitionGraph:
    """Build solely from explicitly supplied post-holdout training rows."""
    if top_k < 0:
        raise ValueError("top_k must be nonnegative")
    if alpha < 0:
        raise ValueError("alpha must be nonnegative")
    rows = list(train_rows)
    paths = [str(row.get(gloss_key, "")).split() for row in rows]
    frequencies = Counter(token for path in paths for token in path)
    ordinary = sorted(frequencies, key=lambda token: (-frequencies[token], token))[
        :top_k
    ]
    states = tuple(ordinary + [OTHER])
    lookup = {token: index for index, token in enumerate(states)}
    other = lookup[OTHER]
    bos, eos = len(states), len(states) + 1
    counts: Counter[tuple[int, int]] = Counter()
    for path in paths:
        mapped = [lookup.get(token, other) for token in path]
        nodes = [bos, *mapped, eos]
        counts.update(zip(nodes, nodes[1:]))

    edges = sorted(counts)
    src = np.asarray([edge[0] for edge in edges], dtype=np.int64)
    dst = np.asarray([edge[1] for edge in edges], dtype=np.int64)
    count = np.asarray([counts[edge] for edge in edges], dtype=np.int64)
    smoothed = count.astype(np.float64) + alpha
    denominators: Counter[int] = Counter()
    for source, weight in zip(src.tolist(), smoothed.tolist()):
        denominators[source] += weight
    log_prob = np.asarray(
        [
            np.log(weight / denominators[source])
            for source, weight in zip(src, smoothed)
        ],
        dtype=np.float32,
    )
    sample_ids = tuple(_row_id(row) for row in rows)
    train_hash = _canonical_hash(
        [{"id": sample_id, "gloss": path} for sample_id, path in zip(sample_ids, paths)]
    )
    return StructuredTransitionGraph(
        states, src, dst, count, log_prob, alpha, sample_ids, train_hash
    )


def shuffled_transition_control(
    graph: StructuredTransitionGraph, *, seed: int = 0
) -> StructuredTransitionGraph:
    """Permute ordinary destination labels, preserving edge/count degree multisets."""
    size = graph.num_states
    permutation = np.random.default_rng(seed).permutation(size)
    if size > 1 and np.array_equal(permutation, np.arange(size)):
        permutation = np.roll(permutation, 1)
    dst = graph.edge_dst.copy()
    ordinary = dst < size
    dst[ordinary] = permutation[dst[ordinary]]
    order = np.lexsort((dst, graph.edge_src))
    return StructuredTransitionGraph(
        graph.states,
        graph.edge_src[order].copy(),
        dst[order],
        graph.edge_count[order].copy(),
        graph.edge_log_prob[order].copy(),
        graph.alpha,
        graph.train_sample_ids,
        graph.train_hash,
    )


def save_structured_transition_graph(
    graph: StructuredTransitionGraph, npz_path: str | Path, manifest_path: str | Path
) -> None:
    """Write canonical JSON and a byte-deterministic NPZ archive."""
    arrays = {
        "edge_src": graph.edge_src,
        "edge_dst": graph.edge_dst,
        "edge_count": graph.edge_count,
        "edge_log_prob": graph.edge_log_prob,
    }
    destination = Path(npz_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, array in arrays.items():
            buffer = io.BytesIO()
            np.lib.format.write_array(buffer, array, allow_pickle=False)
            info = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o600 << 16
            archive.writestr(info, buffer.getvalue())
    manifest = Path(manifest_path)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(
        json.dumps(graph.manifest(), sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )


def load_structured_transition_graph(
    npz_path: str | Path, manifest_path: str | Path
) -> StructuredTransitionGraph:
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    with np.load(npz_path, allow_pickle=False) as arrays:
        graph = StructuredTransitionGraph(
            tuple(manifest["states"]),
            arrays["edge_src"].copy(),
            arrays["edge_dst"].copy(),
            arrays["edge_count"].copy(),
            arrays["edge_log_prob"].copy(),
            float(manifest["alpha"]),
            tuple(manifest["train_sample_ids"]),
            str(manifest["train_hash"]),
        )
    if graph.manifest()["graph_hash"] != manifest["graph_hash"]:
        raise ValueError("structured transition artifact hash mismatch")
    return graph
