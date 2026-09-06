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


def gold_path_score(
    emissions: Tensor,
    gold_states: Tensor,
    lengths: Tensor,
    graph: StructuredTransitionGraph,
    transition_scale: float = 0.25,
) -> Tensor:
    """Exact score of padded mapped gold paths; absent edges score ``-inf``."""
    scores = emissions.float()
    transition = dense_transition_scores(graph, scores.device) * transition_scale
    result = []
    for batch_index in range(scores.shape[0]):
        length = int(lengths[batch_index].item())
        path = gold_states[batch_index, :length].to(scores.device)
        emission_score = scores[batch_index, :length].gather(1, path[:, None]).sum()
        nodes = torch.cat(
            (
                torch.tensor([graph.bos_index], device=scores.device),
                path,
                torch.tensor([graph.eos_index], device=scores.device),
            )
        )
        result.append(emission_score + transition[nodes[:-1], nodes[1:]].sum())
    return torch.stack(result)


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
    if emissions.ndim != 3 or emissions.shape[2] != graph.num_states:
        raise ValueError("emissions must have shape [B,T,G]")
    if torch.any(lengths < 1) or torch.any(lengths > emissions.shape[1]):
        raise ValueError("lengths must lie in [1,T]")
    partition_fn = dense_log_partition if dense else sparse_log_partition
    log_z = partition_fn(emissions, lengths, graph, transition_scale)
    gold = gold_path_score(emissions, gold_states, lengths, graph, transition_scale)
    return (log_z - gold) / lengths.to(log_z.dtype)
