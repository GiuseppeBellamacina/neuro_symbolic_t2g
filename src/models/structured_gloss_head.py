"""Source-conditioned emission head and sparse structured NLL primitives."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from src.datasets.structured_transitions import StructuredTransitionGraph


class StructuredGlossHead(nn.Module):
    """Produce all emissions from one source-side assistant-boundary vector."""

    def __init__(self, hidden_size: int, num_states: int, max_length: int = 64):
        super().__init__()
        self.max_length = max_length
        self.position = nn.Embedding(max_length, hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.output = nn.Linear(hidden_size, num_states)

    def forward(self, boundary_hidden: Tensor, length: int | None = None) -> Tensor:
        if boundary_hidden.ndim != 2:
            raise ValueError("boundary_hidden must have shape [B, H]")
        steps = self.max_length if length is None else length
        if not 0 < steps <= self.max_length:
            raise ValueError("length must be between 1 and max_length")
        positions = torch.arange(steps, device=boundary_hidden.device)
        hidden = boundary_hidden[:, None, :] + self.position(positions)[None, :, :]
        return self.output(self.layer_norm(hidden))


def gather_assistant_boundary(
    hidden_states: Tensor, boundary_indices: Tensor
) -> Tensor:
    """Gather one source-only hidden vector per batch item."""
    if hidden_states.ndim != 3 or boundary_indices.ndim != 1:
        raise ValueError("expected hidden_states [B,S,H] and boundary_indices [B]")
    batch = torch.arange(hidden_states.shape[0], device=hidden_states.device)
    return hidden_states[batch, boundary_indices]


def _graph_tensors(graph: StructuredTransitionGraph, device: torch.device):
    src = torch.as_tensor(graph.edge_src, device=device, dtype=torch.long)
    dst = torch.as_tensor(graph.edge_dst, device=device, dtype=torch.long)
    weight = torch.as_tensor(graph.edge_log_prob, device=device, dtype=torch.float32)
    return src, dst, weight


def dense_transition_scores(graph: StructuredTransitionGraph, device=None) -> Tensor:
    """Dense oracle matrix; intended only for tests and tiny graphs."""
    matrix = torch.full(
        (graph.num_states + 2, graph.num_states + 2),
        -torch.inf,
        dtype=torch.float32,
        device=device,
    )
    src, dst, weight = _graph_tensors(graph, matrix.device)
    matrix[src, dst] = weight
    return matrix


def _length_conditioned_logz(history: list[Tensor], lengths: Tensor) -> Tensor:
    """Select EOS termination exactly at each observed length."""
    values = torch.stack(history, dim=1)
    return values.gather(1, (lengths.to(values.device) - 1)[:, None]).squeeze(1)


def _safe_logsumexp(values: Tensor, dim: int) -> Tensor:
    """Avoid undefined gradients for slices containing only ``-inf``."""
    reachable = torch.isfinite(values).any(dim=dim, keepdim=True)
    safe_values = torch.where(reachable, values, torch.zeros_like(values))
    result = torch.logsumexp(safe_values, dim=dim)
    return torch.where(reachable.squeeze(dim), result, -torch.inf)


def dense_log_partition(
    emissions: Tensor,
    lengths: Tensor,
    graph: StructuredTransitionGraph,
    transition_scale: float = 0.25,
) -> Tensor:
    """FP32 dense reference partition conditioned on the observed length."""
    scores = emissions.float()
    transition = dense_transition_scores(graph, scores.device) * transition_scale
    g = graph.num_states
    alpha = scores[:, 0] + transition[graph.bos_index, :g]
    endings = [torch.logsumexp(alpha + transition[:g, graph.eos_index], dim=1)]
    for step in range(1, scores.shape[1]):
        alpha = scores[:, step] + _safe_logsumexp(
            alpha[:, :, None] + transition[:g, :g], dim=1
        )
        endings.append(torch.logsumexp(alpha + transition[:g, graph.eos_index], dim=1))
    return _length_conditioned_logz(endings, lengths)


def sparse_log_partition(
    emissions: Tensor,
    lengths: Tensor,
    graph: StructuredTransitionGraph,
    transition_scale: float = 0.25,
) -> Tensor:
    """FP32 exact-length edge/scatter recurrence without [B,T,V,V]."""
    scores = emissions.float()
    batch, _, g = scores.shape
    src, dst, weight = _graph_tensors(graph, scores.device)
    bos_mask = src == graph.bos_index
    eos_mask = dst == graph.eos_index
    internal = (src < g) & (dst < g)
    start = torch.full((g,), -torch.inf, device=scores.device)
    start = start.scatter(0, dst[bos_mask], weight[bos_mask] * transition_scale)
    end = torch.full((g,), -torch.inf, device=scores.device)
    end = end.scatter(0, src[eos_mask], weight[eos_mask] * transition_scale)
    alpha = scores[:, 0] + start
    endings = [torch.logsumexp(alpha + end, dim=1)]
    edge_src, edge_dst = src[internal], dst[internal]
    edge_weight = weight[internal] * transition_scale
    for step in range(1, scores.shape[1]):
        candidates = alpha[:, edge_src] + edge_weight
        incoming = torch.full((batch, g), -torch.inf, device=scores.device)
        incoming = incoming.scatter_reduce(
            1,
            edge_dst.expand(batch, -1),
            candidates,
            reduce="amax",
            include_self=True,
        )
        stable = torch.where(
            torch.isfinite(incoming[:, edge_dst]),
            torch.exp(candidates - incoming[:, edge_dst]),
            torch.zeros_like(candidates),
        )
        totals = torch.zeros((batch, g), device=scores.device).scatter_add(
            1, edge_dst.expand(batch, -1), stable
        )
        incoming = incoming + torch.log(totals)
        alpha = scores[:, step] + incoming
        endings.append(torch.logsumexp(alpha + end, dim=1))
    return _length_conditioned_logz(endings, lengths)


class StructuredGraphLoss(nn.Module):
    """Cache-safe sparse exact-length CRF loss for one immutable graph.

    Graph tensors and edge lookup keys are registered buffers, so repeated
    calls allocate no graph representation and normal ``to``/state-dict
    semantics apply. Transition smoothing in the artifact covers observed
    support only; it never introduces dense or unseen transitions.
    """

    def __init__(
        self, graph: StructuredTransitionGraph, transition_scale: float = 0.25
    ) -> None:
        super().__init__()
        self.num_states = graph.num_states
        self.bos_index = graph.bos_index
        self.eos_index = graph.eos_index
        self.transition_scale = float(transition_scale)
        src = torch.as_tensor(graph.edge_src, dtype=torch.long)
        dst = torch.as_tensor(graph.edge_dst, dtype=torch.long)
        weight = torch.as_tensor(graph.edge_log_prob, dtype=torch.float32)
        internal = (src < self.num_states) & (dst < self.num_states)
        start = torch.full((self.num_states,), -torch.inf)
        start[dst[src == self.bos_index]] = weight[src == self.bos_index]
        end = torch.full((self.num_states,), -torch.inf)
        end[src[dst == self.eos_index]] = weight[dst == self.eos_index]
        keys = src * (self.num_states + 2) + dst
        order = torch.argsort(keys)
        self.register_buffer("internal_src", src[internal])
        self.register_buffer("internal_dst", dst[internal])
        self.register_buffer("internal_weight", weight[internal])
        self.register_buffer("start_weight", start)
        self.register_buffer("end_weight", end)
        self.register_buffer("edge_keys", keys[order])
        self.register_buffer("edge_weights", weight[order])

    def log_partition(self, emissions: Tensor, lengths: Tensor) -> Tensor:
        """FP32 sparse partition, conditioned on each exact gold length."""
        scores = emissions.float()
        batch, _, states = scores.shape
        if states != self.num_states:
            raise ValueError("emissions state dimension does not match graph")
        scale = self.transition_scale
        alpha = scores[:, 0] + self.start_weight * scale
        endings = [torch.logsumexp(alpha + self.end_weight * scale, dim=1)]
        for step in range(1, scores.shape[1]):
            candidates = alpha[:, self.internal_src] + self.internal_weight * scale
            incoming = torch.full((batch, states), -torch.inf, device=scores.device)
            destinations = self.internal_dst.expand(batch, -1)
            incoming.scatter_reduce_(
                1, destinations, candidates, reduce="amax", include_self=True
            )
            maxima = incoming[:, self.internal_dst]
            stable = torch.where(
                torch.isfinite(maxima), torch.exp(candidates - maxima), 0.0
            )
            totals = torch.zeros_like(incoming).scatter_add(1, destinations, stable)
            incoming = incoming + torch.log(totals)
            alpha = scores[:, step] + incoming
            endings.append(torch.logsumexp(alpha + self.end_weight * scale, dim=1))
        return _length_conditioned_logz(endings, lengths)

    def gold_score(
        self, emissions: Tensor, gold_states: Tensor, lengths: Tensor
    ) -> Tensor:
        """Vectorized sparse gold score; unsupported active edges yield ``-inf``."""
        scores = emissions.float()
        batch, steps, _ = scores.shape
        positions = torch.arange(steps, device=scores.device)[None, :]
        active_states = positions < lengths[:, None]
        selected = scores.gather(2, gold_states.to(scores.device)[:, :, None]).squeeze(
            2
        )
        emission_score = torch.where(active_states, selected, 0.0).sum(dim=1)

        nodes = torch.full(
            (batch, steps + 2), self.eos_index, dtype=torch.long, device=scores.device
        )
        nodes[:, 0] = self.bos_index
        nodes[:, 1 : steps + 1] = gold_states.to(scores.device)
        edge_positions = torch.arange(steps + 1, device=scores.device)[None, :]
        active_edges = edge_positions <= lengths[:, None]
        destinations = torch.where(
            edge_positions == lengths[:, None], self.eos_index, nodes[:, 1:]
        )
        keys = nodes[:, :-1] * (self.num_states + 2) + destinations
        indices = torch.searchsorted(self.edge_keys, keys)
        safe_indices = indices.clamp_max(max(self.edge_keys.numel() - 1, 0))
        found = (indices < self.edge_keys.numel()) & (
            self.edge_keys[safe_indices] == keys
        )
        weights = self.edge_weights[safe_indices]
        edge_scores = torch.where(
            active_edges & found, weights * self.transition_scale, 0.0
        )
        supported = (~active_edges | found).all(dim=1)
        transition_score = edge_scores.sum(dim=1)
        return torch.where(supported, emission_score + transition_score, -torch.inf)

    def forward(
        self, emissions: Tensor, gold_states: Tensor, lengths: Tensor
    ) -> Tensor:
        _validate_inputs(emissions, gold_states, lengths, self.num_states)
        log_z = self.log_partition(emissions, lengths)
        gold = self.gold_score(emissions, gold_states, lengths)
        return (log_z - gold) / lengths.to(device=log_z.device, dtype=log_z.dtype)


def gold_path_score(
    emissions: Tensor,
    gold_states: Tensor,
    lengths: Tensor,
    graph: StructuredTransitionGraph,
    transition_scale: float = 0.25,
) -> Tensor:
    """Exact score of padded mapped gold paths; absent edges score ``-inf``."""
    return (
        StructuredGraphLoss(graph, transition_scale)
        .to(emissions.device)
        .gold_score(emissions, gold_states, lengths)
    )


def _validate_inputs(
    emissions: Tensor, gold_states: Tensor, lengths: Tensor, num_states: int
) -> None:
    if emissions.ndim != 3 or emissions.shape[2] != num_states:
        raise ValueError("emissions must have shape [B,T,G]")
    if gold_states.shape != emissions.shape[:2] or lengths.shape != emissions.shape[:1]:
        raise ValueError("gold_states must be [B,T] and lengths must be [B]")
    if torch.any(lengths < 1) or torch.any(lengths > emissions.shape[1]):
        raise ValueError("lengths must lie in [1,T]")
    if torch.any(gold_states < 0) or torch.any(gold_states >= num_states):
        raise ValueError("gold state index out of range")


def structured_nll(
    emissions: Tensor,
    gold_states: Tensor,
    lengths: Tensor,
    graph: StructuredTransitionGraph,
    transition_scale: float = 0.25,
    *,
    dense: bool = False,
) -> Tensor:
    """Per-example, per-gloss length-conditioned negative log likelihood."""
    _validate_inputs(emissions, gold_states, lengths, graph.num_states)
    if not dense:
        return StructuredGraphLoss(graph, transition_scale).to(emissions.device)(
            emissions, gold_states, lengths
        )
    partition_fn = dense_log_partition if dense else sparse_log_partition
    log_z = partition_fn(emissions, lengths, graph, transition_scale)
    gold = gold_path_score(emissions, gold_states, lengths, graph, transition_scale)
    return (log_z - gold) / lengths.to(log_z.dtype)
