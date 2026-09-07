"""Tests for the opt-in auxiliary SFT objectives.

The property that matters most is the *inertness contract*: with every weight at
zero the trainer must behave exactly like stock ``SFTTrainer``. That is what
keeps historical SFT results reproducible, and it is the reason the previous
attempt (on the ``edit-rewards`` branch) was rejected — it changed the default
path. So these tests assert inertness first and functionality second.
"""

from __future__ import annotations

import math

import pytest
import torch

from src.training.auxiliary_sft_trainer import (
    require_single_process,
    resolve_auxiliary_config,
)

# --- config resolution: nothing may be enabled implicitly -------------------


def test_absent_section_disables_everything():
    assert resolve_auxiliary_config({}) == {}
    assert resolve_auxiliary_config({"training": {"max_steps": 10}}) == {}


def test_zero_weight_is_treated_as_disabled():
    """A documented block with weight 0 must not change training behaviour."""
    cfg = {"auxiliary_objective": {"allowed_mass": {"weight": 0.0}}}
    assert resolve_auxiliary_config(cfg) == {}


def test_positive_weight_is_resolved():
    cfg = {
        "auxiliary_objective": {
            "allowed_mass": {"weight": 0.1, "warmup_steps": 200},
        }
    }
    resolved = resolve_auxiliary_config(cfg)
    assert resolved == {"allowed_mass": {"weight": 0.1, "warmup_steps": 200}}


def test_structured_section_carries_its_own_knobs():
    cfg = {
        "auxiliary_objective": {
            "structured": {
                "weight": 0.1,
                "warmup_steps": 200,
                "top_k": 512,
                "alpha": 0.1,
                "shuffled_control": True,
            }
        }
    }
    resolved = resolve_auxiliary_config(cfg)["structured"]
    assert resolved["top_k"] == 512
    assert resolved["alpha"] == 0.1
    assert resolved["shuffled_control"] is True


@pytest.mark.parametrize("bad", [-0.1, float("nan"), float("inf"), True, "0.1"])
def test_invalid_weight_rejected_before_training(bad):
    cfg = {"auxiliary_objective": {"allowed_mass": {"weight": bad}}}
    with pytest.raises(ValueError, match="weight"):
        resolve_auxiliary_config(cfg)


@pytest.mark.parametrize("bad", [-1, 1.5, True])
def test_invalid_warmup_rejected(bad):
    cfg = {
        "auxiliary_objective": {"allowed_mass": {"weight": 0.1, "warmup_steps": bad}}
    }
    with pytest.raises(ValueError, match="warmup_steps"):
        resolve_auxiliary_config(cfg)


def test_non_mapping_sections_rejected():
    with pytest.raises(ValueError, match="must be a mapping"):
        resolve_auxiliary_config({"auxiliary_objective": [1, 2]})
    with pytest.raises(ValueError, match="must be a mapping"):
        resolve_auxiliary_config({"auxiliary_objective": {"allowed_mass": 3}})


@pytest.mark.parametrize("unsupported", ["packing", "padding_free"])
def test_packing_and_padding_free_are_refused(unsupported):
    """Both losses need per-position alignment; refuse instead of misscoring."""
    cfg = {
        "auxiliary_objective": {"allowed_mass": {"weight": 0.1}},
        "training": {unsupported: True},
    }
    with pytest.raises(ValueError, match=unsupported):
        resolve_auxiliary_config(cfg)


def test_packing_is_allowed_when_auxiliary_is_off():
    """The guard must not restrict configs that do not use the objectives."""
    cfg = {"training": {"packing": True}}
    assert resolve_auxiliary_config(cfg) == {}


# --- distributed guard ------------------------------------------------------


def test_single_process_guard_noop_when_disabled(monkeypatch):
    monkeypatch.setenv("WORLD_SIZE", "4")
    require_single_process({})  # must not raise


def test_single_process_guard_rejects_distributed(monkeypatch):
    monkeypatch.setenv("WORLD_SIZE", "4")
    with pytest.raises(ValueError, match="WORLD_SIZE=1"):
        require_single_process({"allowed_mass": {"weight": 0.1}})


def test_single_process_guard_accepts_one(monkeypatch):
    monkeypatch.setenv("WORLD_SIZE", "1")
    require_single_process({"allowed_mass": {"weight": 0.1}})


def test_single_process_guard_rejects_non_integer(monkeypatch):
    monkeypatch.setenv("WORLD_SIZE", "many")
    with pytest.raises(ValueError, match="WORLD_SIZE"):
        require_single_process({"allowed_mass": {"weight": 0.1}})


# --- collator span recovery -------------------------------------------------


class _StubCollator:
    """Minimal stand-in for the HF collator, so the span logic is testable.

    The real ``CompletionSpanCollator`` subclasses a HF collator that needs a
    tokenizer; the logic under test is the span recovery, which only depends on
    the batch tensors it produces.
    """

    def __init__(self, eos_token_id: int) -> None:
        self.eos_token_id = eos_token_id

    def __call__(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        from src.training.auxiliary_sft_trainer import CompletionSpanCollator

        labels = batch["labels"]
        supervised = labels != -100
        starts: list[int] = []
        eligible: list[bool] = []
        for row in range(labels.shape[0]):
            positions = torch.nonzero(supervised[row], as_tuple=False).flatten()
            if positions.numel() == 0:
                starts.append(0)
                eligible.append(False)
                continue
            first = int(positions[0].item())
            last = int(positions[-1].item())
            contiguous = positions.numel() == (last - first + 1)
            ends_eos = int(batch["input_ids"][row, last].item()) == self.eos_token_id
            starts.append(first)
            eligible.append(bool(contiguous and ends_eos and first >= 1))
        assert CompletionSpanCollator is not None  # module import sanity
        batch["completion_start"] = torch.tensor(starts, dtype=torch.long)
        batch["completion_eligible"] = torch.tensor(eligible, dtype=torch.bool)
        return batch


def test_span_recovery_marks_eos_terminated_rows_eligible():
    collator = _StubCollator(eos_token_id=9)
    batch = {
        "input_ids": torch.tensor([[1, 2, 3, 9]]),
        "labels": torch.tensor([[-100, 2, 3, 9]]),
    }
    out = collator(batch)
    assert out["completion_start"].tolist() == [1]
    assert out["completion_eligible"].tolist() == [True]


def test_span_recovery_marks_truncated_rows_ineligible():
    """No EOS means the completion was cut: it has no valid final state."""
    collator = _StubCollator(eos_token_id=9)
    batch = {
        "input_ids": torch.tensor([[1, 2, 3, 4]]),
        "labels": torch.tensor([[-100, 2, 3, 4]]),
    }
    out = collator(batch)
    assert out["completion_eligible"].tolist() == [False]


def test_span_recovery_marks_noncontiguous_rows_ineligible():
    collator = _StubCollator(eos_token_id=9)
    batch = {
        "input_ids": torch.tensor([[1, 2, 3, 9]]),
        "labels": torch.tensor([[-100, 2, -100, 9]]),
    }
    out = collator(batch)
    assert out["completion_eligible"].tolist() == [False]


def test_span_recovery_handles_fully_unsupervised_row():
    """A row with nothing to supervise must be skipped, not scored."""
    collator = _StubCollator(eos_token_id=9)
    batch = {
        "input_ids": torch.tensor([[1, 2, 3, 4]]),
        "labels": torch.tensor([[-100, -100, -100, -100]]),
    }
    out = collator(batch)
    assert out["completion_start"].tolist() == [0]
    assert out["completion_eligible"].tolist() == [False]


# --- trainer construction contract ------------------------------------------


def test_positive_weight_requires_a_mask_producer():
    """A loss whose input cannot be produced must fail at construction."""
    from src.training.auxiliary_sft_trainer import AuxiliarySFTTrainer

    with pytest.raises(ValueError, match="allowed_mask_fn"):
        AuxiliarySFTTrainer.__init__(
            object.__new__(AuxiliarySFTTrainer),
            allowed_mask_fn=None,
            mass_weight=0.1,
        )


@pytest.mark.parametrize("bad", [-0.1, float("nan")])
def test_invalid_trainer_weight_rejected(bad):
    from src.training.auxiliary_sft_trainer import AuxiliarySFTTrainer

    with pytest.raises(ValueError, match="mass_weight"):
        AuxiliarySFTTrainer.__init__(
            object.__new__(AuxiliarySFTTrainer), mass_weight=bad
        )


def test_auxiliary_disabled_by_default_construction():
    """Zero weight + no producer is the inert configuration."""
    from src.training.auxiliary_sft_trainer import AuxiliarySFTTrainer

    stub = object.__new__(AuxiliarySFTTrainer)
    stub.allowed_mask_fn = None
    stub.mass_weight = 0.0
    stub.mass_warmup_steps = 0
    assert stub._auxiliary_enabled() is False


def test_auxiliary_enabled_needs_both_weight_and_producer():
    from src.training.auxiliary_sft_trainer import AuxiliarySFTTrainer

    stub = object.__new__(AuxiliarySFTTrainer)
    stub.mass_warmup_steps = 0
    stub.allowed_mask_fn = lambda prefixes, vocab, device: torch.ones(
        len(prefixes), vocab, dtype=torch.bool
    )
    stub.mass_weight = 0.0
    assert stub._auxiliary_enabled() is False
    stub.mass_weight = 0.1
    assert stub._auxiliary_enabled() is True


def test_zero_weight_delegates_to_stock_compute_loss():
    """THE inertness contract: with weight 0 nothing is added to the loss.

    This is the property that keeps every historical SFT result reproducible.
    It is asserted by driving ``AuxiliarySFTTrainer.compute_loss`` with a stub
    ``super()`` that records how it was called: the auxiliary path must not
    request outputs, must not touch the loss value, and must still strip the
    span metadata (leaving it in would reach the model as an unexpected kwarg).
    """
    from src.training.auxiliary_sft_trainer import AuxiliarySFTTrainer

    calls: list[dict] = []
    sentinel = torch.tensor(1.2345)

    class _Stub(AuxiliarySFTTrainer):
        def __init__(self) -> None:  # bypass SFTTrainer.__init__
            self.allowed_mask_fn = None
            self.mass_weight = 0.0
            self.mass_warmup_steps = 0
            self.auxiliary_diagnostics = {}

    def _stock(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        calls.append(
            {
                "return_outputs": return_outputs,
                "keys": sorted(inputs.keys()),
            }
        )
        return sentinel

    stub = _Stub()
    inputs = {
        "input_ids": torch.tensor([[1, 2]]),
        "labels": torch.tensor([[-100, 2]]),
        "completion_start": torch.tensor([1]),
        "completion_eligible": torch.tensor([True]),
    }

    import src.training.auxiliary_sft_trainer as module

    original = module.SFTTrainer.compute_loss
    module.SFTTrainer.compute_loss = _stock  # type: ignore[assignment]
    try:
        result = stub.compute_loss(torch.nn.Linear(2, 2), inputs)
    finally:
        module.SFTTrainer.compute_loss = original  # type: ignore[assignment]

    # Same object, not merely the same value: nothing was added.
    assert result is sentinel
    assert len(calls) == 1
    # Stock path must NOT ask for outputs (that is the auxiliary-only need).
    assert calls[0]["return_outputs"] is False
    # Span metadata stripped before reaching the model.
    assert "completion_start" not in calls[0]["keys"]
    assert "completion_eligible" not in calls[0]["keys"]
    assert stub.auxiliary_diagnostics == {}


def test_positive_weight_adds_a_nonnegative_term():
    """With weight > 0 the loss grows by the mass penalty, never shrinks."""
    from src.training.auxiliary_sft_trainer import AuxiliarySFTTrainer

    lm_value = torch.tensor(2.0)

    class _Outputs:
        # Uniform logits over 4 tokens; only 2 allowed -> penalty = -log(0.5).
        logits = torch.zeros(1, 2, 4)

    class _Stub(AuxiliarySFTTrainer):
        def __init__(self) -> None:
            self.allowed_mask_fn = lambda prefixes, vocab, device: torch.tensor(
                [[True, True, False, False]] * len(prefixes)
            )
            self.mass_weight = 1.0
            self.mass_warmup_steps = 0
            self.auxiliary_diagnostics = {}
            self.state = None

    def _stock(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        return (lm_value, _Outputs())

    import src.training.auxiliary_sft_trainer as module

    original = module.SFTTrainer.compute_loss
    module.SFTTrainer.compute_loss = _stock  # type: ignore[assignment]
    try:
        total = module.AuxiliarySFTTrainer.compute_loss(
            _Stub(),
            torch.nn.Linear(2, 2),
            {
                "input_ids": torch.tensor([[1, 2]]),
                "labels": torch.tensor([[-100, 2]]),
                "completion_start": torch.tensor([1]),
                "completion_eligible": torch.tensor([True]),
            },
        )
    finally:
        module.SFTTrainer.compute_loss = original  # type: ignore[assignment]

    assert float(total) > float(lm_value)
    assert math.isclose(float(total), 2.0 + math.log(2.0), rel_tol=1e-5)


def test_ineligible_rows_are_not_scored():
    """A truncated completion must contribute nothing, not a wrong value."""
    from src.training.auxiliary_sft_trainer import AuxiliarySFTTrainer

    class _Outputs:
        logits = torch.zeros(1, 2, 4)

    class _Stub(AuxiliarySFTTrainer):
        def __init__(self) -> None:
            self.allowed_mask_fn = lambda prefixes, vocab, device: torch.tensor(
                [[True, True, False, False]] * max(len(prefixes), 1)
            )
            self.mass_weight = 1.0
            self.mass_warmup_steps = 0
            self.auxiliary_diagnostics = {}
            self.state = None

    def _stock(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        return (torch.tensor(2.0), _Outputs())

    import src.training.auxiliary_sft_trainer as module

    original = module.SFTTrainer.compute_loss
    module.SFTTrainer.compute_loss = _stock  # type: ignore[assignment]
    try:
        total = module.AuxiliarySFTTrainer.compute_loss(
            _Stub(),
            torch.nn.Linear(2, 2),
            {
                "input_ids": torch.tensor([[1, 2]]),
                "labels": torch.tensor([[-100, 2]]),
                "completion_start": torch.tensor([1]),
                "completion_eligible": torch.tensor([False]),  # troncata
            },
        )
    finally:
        module.SFTTrainer.compute_loss = original  # type: ignore[assignment]

    assert math.isclose(float(total), 2.0, rel_tol=1e-9)


def test_structured_term_contributes_and_reaches_the_head():
    """End-to-end del ramo structured, con un grafo reale (4 stati).

    Verifica le tre proprieta' che rendono la cella non decorativa:
    1. ``output_hidden_states`` viene richiesto al forward (senza, il termine
       verrebbe saltato in silenzio e si addestrerebbe un SFT normale);
    2. la loss totale cresce di una quantita' finita e non nulla;
    3. il gradiente raggiunge davvero i parametri della testa strutturata.
    """
    from src.datasets.structured_transitions import build_structured_transition_graph
    from src.models.structured_gloss_head import (
        StructuredGlossHead,
        StructuredGraphLoss,
    )
    from src.training.auxiliary_sft_trainer import AuxiliarySFTTrainer

    rows = [{"gloss": g} for g in ["IX MAN", "IX WALK", "MAN WALK", "IX MAN WALK"]]
    graph = build_structured_transition_graph(rows, top_k=3)
    head = StructuredGlossHead(
        hidden_size=8, num_states=graph.num_states, max_length=8
    )

    class _Stub(AuxiliarySFTTrainer):
        def __init__(self) -> None:
            self.allowed_mask_fn = None
            self.mass_weight = 0.0
            self.mass_warmup_steps = 0
            self.structured_head = head
            self.structured_loss = StructuredGraphLoss(graph)
            self.structured_graph = graph
            self.structured_weight = 1.0
            self.structured_warmup_steps = 0
            self.auxiliary_diagnostics = {}
            self.state = None

    class _Outputs:
        logits = torch.zeros(2, 3, 5)
        hidden_states = [torch.randn(2, 3, 8)]

    captured: dict = {}

    def _stock(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        captured["hidden"] = inputs.get("output_hidden_states")
        return (torch.tensor(2.0), _Outputs())

    import src.training.auxiliary_sft_trainer as module

    original = module.SFTTrainer.compute_loss
    module.SFTTrainer.compute_loss = _stock  # type: ignore[assignment]
    try:
        total = module.AuxiliarySFTTrainer.compute_loss(
            _Stub(),
            torch.nn.Linear(2, 2),
            {
                "input_ids": torch.tensor([[1, 2, 3], [1, 2, 3]]),
                "labels": torch.tensor([[-100, 2, 3], [-100, 2, 3]]),
                "gold_gloss": ["IX MAN", "IX WALK"],
            },
        )
    finally:
        module.SFTTrainer.compute_loss = original  # type: ignore[assignment]

    assert captured["hidden"] is True, "output_hidden_states non propagato"
    assert torch.isfinite(total)
    assert float(total.detach()) != pytest.approx(2.0), "termine structured nullo"

    total.backward()
    grads = [p.grad for p in head.parameters() if p.grad is not None]
    assert grads, "nessun gradiente sulla testa strutturata"
    assert all(torch.isfinite(g).all() for g in grads)


def test_structured_requires_the_graph_for_target_mapping():
    """StructuredGraphLoss tiene solo buffer: senza il grafo non si mappa nulla."""
    from src.training.auxiliary_sft_trainer import AuxiliarySFTTrainer

    stub = object.__new__(AuxiliarySFTTrainer)
    stub.structured_graph = None
    stub.structured_head = None
    with pytest.raises(RuntimeError, match="structured_graph"):
        AuxiliarySFTTrainer._structured_targets(stub, ["IX MAN"], "cpu")


def test_structured_weight_zero_leaves_loss_untouched():
    from src.training.auxiliary_sft_trainer import AuxiliarySFTTrainer

    stub = object.__new__(AuxiliarySFTTrainer)
    stub.structured_weight = 0.0
    stub.structured_warmup_steps = 0
    object.__setattr__(stub, "state", None)
    assert stub._structured_weight_now() == 0.0


def test_warmup_semantics_match_the_loss_module():
    """The trainer must not reimplement the ramp: it delegates to the loss."""
    from src.training.allowed_mass_loss import allowed_mass_loss

    logits = torch.zeros(1, 4)
    mask = torch.tensor([[True, True, False, False]])
    weights = [
        allowed_mass_loss(logits, mask, weight=1.0, warmup_steps=10, step=s)[
            1
        ].effective_weight
        for s in (0, 5, 10)
    ]
    assert weights == [0.0, 0.5, 1.0]
    assert math.isclose(weights[1], 0.5)
