from __future__ import annotations

import torch

from src.datasets.structured_transitions import build_structured_transition_graph
from src.models.structured_gloss_head import (
    StructuredGlossHead,
    gather_assistant_boundary,
)
from src.training.structured_sft import (
    assistant_boundary_indices,
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
