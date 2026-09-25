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
STRUCTURED_GRAPH_PROTOCOL = "structured-transition-graph-v2"


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
    top_k: int = 512
    protocol_version: str = STRUCTURED_GRAPH_PROTOCOL

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
        payload = {
            "protocol_version": self.protocol_version,
            "states": self.states,
            "src": self.edge_src.tolist(),
            "dst": self.edge_dst.tolist(),
            "count": self.edge_count.tolist(),
            "log_prob": [float(x).hex() for x in self.edge_log_prob],
            "alpha": float(self.alpha).hex(),
            "top_k": self.top_k,
            "other": OTHER,
            "train_hash": self.train_hash,
        }
        graph_hash = _canonical_hash(payload)
        return {
            "format": 2,
            "protocol_version": self.protocol_version,
            "states": list(self.states),
            "top_k": self.top_k,
            "other_token": OTHER,
            "other_index": self.token_to_index[OTHER],
            "bos_index": self.bos_index,
            "eos_index": self.eos_index,
            "alpha": self.alpha,
            "train_sample_ids": list(self.train_sample_ids),
            "train_hash": self.train_hash,
            "state_digest": _canonical_hash(list(self.states)),
            "edge_digest": _canonical_hash(
                {
                    "src": self.edge_src.tolist(),
                    "dst": self.edge_dst.tolist(),
                    "count": self.edge_count.tolist(),
                    "log_prob": [float(x).hex() for x in self.edge_log_prob],
                }
            ),
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
    """Build solely from explicitly supplied finalized post-holdout train rows.

    Smoothing adds ``alpha`` only to transitions observed in these rows. It does
    not create unseen edges, so every supported path has explicit provenance.
    """
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
    graph = StructuredTransitionGraph(
        states, src, dst, count, log_prob, alpha, sample_ids, train_hash, top_k
    )
    assert_gloss_paths_supported(paths, graph)
    return graph


def assert_gloss_paths_supported(
    glosses: Sequence[str | Sequence[str]], graph: StructuredTransitionGraph
) -> None:
    """Raise if any complete mapped gloss path is absent from ``graph``."""
    edge_keys = set(zip(graph.edge_src.tolist(), graph.edge_dst.tolist()))
    for index, gloss in enumerate(glosses):
        mapped = graph.map_glosses(gloss)
        nodes = [graph.bos_index, *mapped, graph.eos_index]
        missing = [edge for edge in zip(nodes, nodes[1:]) if edge not in edge_keys]
        if missing:
            raise ValueError(f"gloss path {index} has unsupported edges: {missing}")


def shuffled_transition_control(
    graph: StructuredTransitionGraph, *, seed: int = 0
) -> StructuredTransitionGraph:
    """GATE 2 negative control: same support, scrambled transition preferences.

    Every ``(src, dst)`` edge is kept; only the (count, log_prob) payloads are
    permuted *within each source's outgoing edges*. Each state therefore keeps
    its exact set of allowed successors, its row normalization and its
    outgoing-probability multiset (entropy); the only information destroyed
    is which successor is likely. Real gold paths stay scorable by
    construction, so the structured term stays active exactly as often as
    with the real graph.

    The previous version permuted destination labels (``a->b`` became
    ``a->pi(b)``). That removed the edges real gold paths need: in the
    structured-shuffled run 98% of logged steps (2012/2051) had every row
    unsupported, so the term was zero and the "control" trained as plain SFT
    (it matched the plain-SFT cell within 0.8pp EM). It compared
    "structured term vs none", not "real vs scrambled transitions".

    Caveat, inherent to any scorable control: the support itself (which
    transitions exist at all) still comes from real data, so this isolates
    the value of the transition *weights*, not of the support.
    """
    rng = np.random.default_rng(seed)
    count = graph.edge_count.copy()
    log_prob = graph.edge_log_prob.copy()
    for source in np.unique(graph.edge_src):
        indices = np.flatnonzero(graph.edge_src == source)
        if indices.size < 2:
            continue
        permutation = rng.permutation(indices.size)
        if np.array_equal(permutation, np.arange(indices.size)):
            permutation = np.roll(permutation, 1)
        count[indices] = graph.edge_count[indices[permutation]]
        log_prob[indices] = graph.edge_log_prob[indices[permutation]]
    return StructuredTransitionGraph(
        graph.states,
        graph.edge_src.copy(),
        graph.edge_dst.copy(),
        count,
        log_prob,
        graph.alpha,
        graph.train_sample_ids,
        graph.train_hash,
        graph.top_k,
        graph.protocol_version,
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
    if manifest.get("protocol_version") != STRUCTURED_GRAPH_PROTOCOL:
        raise ValueError("unsupported structured transition protocol")
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
            int(manifest["top_k"]),
            str(manifest["protocol_version"]),
        )
    if graph.manifest() != manifest:
        raise ValueError("structured transition artifact hash mismatch")
    return graph
