from __future__ import annotations

import itertools

import pytest
import torch

from src.datasets.structured_transitions import (
    build_structured_transition_graph,
    shuffled_transition_control,
)
from src.models.structured_gloss_head import (
    dense_log_partition,
    gold_path_score,
    sparse_log_partition,
    structured_nll,
)
from src.training.structured_sft import (
    combine_lm_structured_loss,
    independent_position_ce,
    structured_weight,
)


def tiny_graph():
    paths = ["A", "B", "A A", "A B", "B A", "B B"]
    return build_structured_transition_graph(
        [{"id": str(i), "gloss": path} for i, path in enumerate(paths)], top_k=2
    )


def exhaustive_logz(emissions, lengths, graph, scale=0.25):
    values = []
    for batch in range(emissions.shape[0]):
        scores = []
        length = int(lengths[batch])
        for path in itertools.product(range(graph.num_states), repeat=length):
            gold = torch.tensor([path])
            scores.append(
                gold_path_score(
                    emissions[batch : batch + 1],
                    gold,
                    torch.tensor([length]),
                    graph,
                    scale,
                )[0]
            )
        values.append(torch.logsumexp(torch.stack(scores), dim=0))
    return torch.stack(values)


def test_dense_sparse_and_exhaustive_partition_agree_with_variable_lengths():
    torch.manual_seed(1)
    graph = tiny_graph()
    emissions = torch.randn(2, 3, graph.num_states, dtype=torch.float64)
    lengths = torch.tensor([1, 3])
    dense = dense_log_partition(emissions, lengths, graph)
    sparse = sparse_log_partition(emissions, lengths, graph)
    brute = exhaustive_logz(emissions.float(), lengths, graph)
    torch.testing.assert_close(sparse, dense, atol=2e-6, rtol=2e-6)
    torch.testing.assert_close(dense, brute, atol=2e-6, rtol=2e-6)


def test_gold_nll_and_disconnected_path():
    graph = build_structured_transition_graph([{"gloss": "A B"}], top_k=2)
    emissions = torch.zeros(1, 2, graph.num_states)
    lengths = torch.tensor([2])
    valid = torch.tensor([[0, 1]])
    invalid = torch.tensor([[1, 0]])
    dense = structured_nll(emissions, valid, lengths, graph, dense=True)
    sparse = structured_nll(emissions, valid, lengths, graph)
    torch.testing.assert_close(dense, sparse)
    assert torch.isneginf(gold_path_score(emissions, invalid, lengths, graph)).all()
    assert torch.isposinf(structured_nll(emissions, invalid, lengths, graph)).all()


def test_gradcheck_and_finite_gradients():
    graph = tiny_graph()
    lengths = torch.tensor([2])
    gold = torch.tensor([[0, 1]])

    def objective(value):
        return structured_nll(value, gold, lengths, graph, dense=True).double()

    emissions = torch.randn(
        1, 2, graph.num_states, dtype=torch.float64, requires_grad=True
    )
    # DP is intentionally FP32; use relaxed finite-difference tolerances.
    assert torch.autograd.gradcheck(
        objective, (emissions,), eps=1e-3, atol=3e-3, rtol=3e-2
    )
    loss = structured_nll(emissions, gold, lengths, graph).sum()
    loss.backward()
    assert emissions.grad is not None and torch.isfinite(emissions.grad).all()


def test_controls_and_loss_combination():
    graph = build_structured_transition_graph(
        [{"gloss": "A B"}, {"gloss": "A B"}, {"gloss": "B A"}], top_k=2
    )
    shuffled = shuffled_transition_control(graph, seed=3)
    emissions = torch.tensor([[[2.0, -1.0, -2.0], [-1.0, 2.0, -2.0]]])
    gold = torch.tensor([[0, 1]])
    lengths = torch.tensor([2])
    assert not torch.allclose(
        structured_nll(emissions, gold, lengths, graph),
        structured_nll(emissions, gold, lengths, shuffled),
    )
    ce = independent_position_ce(emissions, gold, lengths)
    assert ce.shape == (1,) and torch.isfinite(ce).all()
    assert structured_weight(5, 0.4, 10) == pytest.approx(0.2)
    combined = combine_lm_structured_loss(
        torch.tensor(1.0), ce.mean(), step=5, structured_lambda=0.4, warmup_steps=10
    )
    assert combined == pytest.approx(1.0 + 0.2 * ce.item())
