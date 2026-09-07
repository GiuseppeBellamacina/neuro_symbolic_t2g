from __future__ import annotations

import pytest
import torch

from src.datasets.structured_transitions import build_structured_transition_graph
from src.models.structured_gloss_head import (
    StructuredGlossHead,
    gather_assistant_boundary,
)
from src.training.structured_sft import (
    assistant_boundary_indices,
    map_complete_gloss_sequences,
    map_whitespace_glosses,
    render_source_prompt,
)


class FakeTokenizer:
    chat_template = None


def test_prompt_boundary_contract_and_whitespace_mapping():
    rendered = render_source_prompt("A sentence", FakeTokenizer())
    assert rendered.endswith("<|im_start|>assistant\n")
    assert "A sentence" in rendered
    graph = build_structured_transition_graph([{"gloss": "A B"}], top_k=1)
    assert map_whitespace_glosses(" A\nB  ", graph) == [0, 1]


def test_boundary_indices_support_padding_and_gather_without_backbone_grad():
    mask = torch.tensor([[1, 1, 0, 0], [0, 1, 1, 1]])
    indices = assistant_boundary_indices(mask)
    assert indices.tolist() == [1, 3]
    hidden = torch.randn(2, 4, 5)  # represents a frozen/detached backbone output
    gathered = gather_assistant_boundary(hidden, indices)
    assert torch.equal(gathered[0], hidden[0, 1])
    assert not gathered.requires_grad


def test_source_only_head_shapes_and_gradients():
    head = StructuredGlossHead(hidden_size=6, num_states=4, max_length=64)
    boundary = torch.randn(3, 6, requires_grad=True)
    emissions = head(boundary, length=7)
    assert emissions.shape == (3, 7, 4)
    emissions.sum().backward()
    assert boundary.grad is not None and torch.isfinite(boundary.grad).all()
    assert head.position.weight.grad is not None


def test_complete_mapping_overlength_fail_or_exclude_without_truncation():
    graph = build_structured_transition_graph(
        [{"gloss": "A B C"}, {"gloss": "A"}], top_k=2
    )
    with pytest.raises(ValueError, match="exceeds max_length"):
        map_complete_gloss_sequences(["A B C"], graph, max_length=2)
    batch = map_complete_gloss_sequences(
        ["A B C", "A UNKNOWN"], graph, max_length=2, overlength="exclude"
    )
    assert batch.kept_indices == (1,)
    assert batch.excluded_indices == (0,)
    assert batch.lengths.tolist() == [2]
    assert batch.states.tolist() == [[0, graph.token_to_index["<OTHER>"]]]
