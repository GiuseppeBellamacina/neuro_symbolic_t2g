"""Tests for the opt-in auxiliary SFT objectives.

The property that matters most is the *inertness contract*: with every weight at
zero the trainer must behave exactly like stock ``SFTTrainer``. That is what
keeps historical SFT results reproducible, and it is the reason the previous
attempt (on the ``edit-rewards`` branch) was rejected — it changed the default
path. So these tests assert inertness first and functionality second.
"""

from __future__ import annotations

import math
import os
from typing import Any

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
            span_ids = batch["input_ids"][row, first : last + 1]
            has_eos = bool((span_ids == self.eos_token_id).any().item())
            starts.append(first)
            eligible.append(bool(contiguous and has_eos and first >= 1))
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


# --- real CompletionSpanCollator: the actual HF/TRL collation path ----------
#
# The tests above exercise the span-recovery MATH via _StubCollator, which
# never calls the real base class's __call__ — so they could not have caught
# the real bug (jobs 7457/7458/7459, all three auxiliary-objective cells):
# CompletionSpanCollator subclassed transformers.DataCollatorForLanguageModeling
# (the unrelated BERT-style MLM collator) instead of
# trl.trainer.sft_trainer.DataCollatorForLanguageModeling (the SFT-aware one
# that knows how to pad a per-example completion_mask). Two examples with
# different completion lengths in the same batch crashed torch.tensor()
# ("expected sequence of length 101 at dim 1 (got 98)"). These tests
# instantiate the REAL class and drive it through __call__.


def test_completion_span_collator_pads_examples_of_different_completion_length():
    """The actual failure mode of jobs 7457-7459: two examples whose
    completion_mask differs in length must not crash torch.tensor()."""
    from src.training.auxiliary_sft_trainer import CompletionSpanCollator

    collator = CompletionSpanCollator(pad_token_id=0, eos_token_id=9)
    features = [
        {
            "input_ids": [1, 2, 3, 4, 5, 6, 9],
            "completion_mask": [0, 0, 0, 0, 1, 1, 1],
        },
        {
            "input_ids": [1, 2, 9],
            "completion_mask": [0, 0, 1],
        },
    ]

    batch = collator(features)

    assert batch["input_ids"].shape == batch["labels"].shape == (2, 7)
    # Row 0: completion is the last 3 tokens [4, 5, 6? no: positions 4,5,6 -> values 5,6,9]
    assert batch["labels"][0].tolist() == [-100, -100, -100, -100, 5, 6, 9]
    # Row 1: completion is just the EOS token, right-padded with -100 to len 7.
    assert batch["labels"][1].tolist() == [-100, -100, 9, -100, -100, -100, -100]
    assert batch["completion_start"].tolist() == [4, 2]
    assert batch["completion_eligible"].tolist() == [True, True]


def test_completion_span_collator_eligible_with_trailing_token_after_eos():
    """Regression for job 7551: aux/mass_scored_positions was 0 for 100+
    steps, on both the real ASLG dataset and synthetic data with no chat
    template involved. Root cause: Qwen's chat template emits
    "<|im_end|>\n" for the assistant turn, and that trailing "\n" is itself
    part of the completion span (labels != -100), landing at the actual
    last position instead of the EOS token one before it. The old
    "input_ids[last] == eos_token_id" check made every row ineligible
    regardless of data; checking for EOS anywhere in the contiguous
    supervised span (verified against the real tokenizer: pc_ids tail
    [..., 151645, 198], i.e. [<|im_end|>, "\\n"]) fixes it while still
    rejecting a genuinely truncated completion (no EOS at all)."""
    from src.training.auxiliary_sft_trainer import CompletionSpanCollator

    collator = CompletionSpanCollator(pad_token_id=0, eos_token_id=9)
    features = [
        {
            # completion = positions [1,2,3,4]: gloss tokens, EOS, then a
            # trailing template token (e.g. "\n") that is still supervised.
            "input_ids": [1, 2, 3, 9, 99],
            "completion_mask": [0, 1, 1, 1, 1],
        }
    ]

    batch = collator(features)

    assert batch["completion_start"].tolist() == [1]
    assert batch["completion_eligible"].tolist() == [True]


def test_completion_span_collator_rejects_transformers_style_kwargs():
    """Constructing it the OLD (wrong-base-class) way must fail loudly: the
    real base class takes pad_token_id, not tokenizer/mlm."""
    from src.training.auxiliary_sft_trainer import CompletionSpanCollator

    with pytest.raises(TypeError):
        CompletionSpanCollator(tokenizer=object(), mlm=False, eos_token_id=9)


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
    head = StructuredGlossHead(hidden_size=8, num_states=graph.num_states, max_length=8)

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


def test_structured_term_never_touches_outputs_logits():
    """Regression for the real crash: under Unsloth, outputs.logits is an
    ``EmptyLogits`` placeholder that raises on ANY access (.sum(), .device,
    __getitem__, ...) unless UNSLOTH_RETURN_LOGITS=1 was set before Unsloth
    was imported (a process-wide, import-time decision this trainer cannot
    control from inside compute_loss). structured_term never uses logit
    *values* (only hidden_states), so it must not touch .logits at all.
    Job 7517-7519 all crashed here (auxiliary_sft_trainer.py:490,
    `outputs.logits.sum()`) even with the env var correctly set at import
    time — this fix sidesteps the whole question by not needing logits.
    """
    from src.datasets.structured_transitions import build_structured_transition_graph
    from src.models.structured_gloss_head import (
        StructuredGlossHead,
        StructuredGraphLoss,
    )
    from src.training.auxiliary_sft_trainer import AuxiliarySFTTrainer

    rows = [{"gloss": g} for g in ["IX MAN", "IX WALK", "MAN WALK", "IX MAN WALK"]]
    graph = build_structured_transition_graph(rows, top_k=3)
    head = StructuredGlossHead(hidden_size=8, num_states=graph.num_states, max_length=8)

    class _RaisingLogits:
        """Mirrors unsloth.models._utils.EmptyLogits: any attribute access
        or subscript raises, exactly like the real thing does."""

        def __getattr__(self, name):
            raise NotImplementedError(f"Unsloth: Logits are empty ({name})")

        def __getitem__(self, item):
            raise NotImplementedError("Unsloth: Logits are empty (getitem)")

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
        logits = _RaisingLogits()
        hidden_states = [torch.randn(2, 3, 8)]

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
                "input_ids": torch.tensor([[1, 2, 3], [1, 2, 3]]),
                "labels": torch.tensor([[-100, 2, 3], [-100, 2, 3]]),
                "gold_gloss": ["IX MAN", "IX WALK"],
            },
        )
    finally:
        module.SFTTrainer.compute_loss = original  # type: ignore[assignment]

    assert torch.isfinite(total)
    assert float(total.detach()) != pytest.approx(2.0), "termine structured nullo"


def test_structured_requires_the_graph_for_target_mapping():
    """StructuredGraphLoss tiene solo buffer: senza il grafo non si mappa nulla."""
    from src.training.auxiliary_sft_trainer import AuxiliarySFTTrainer

    stub = object.__new__(AuxiliarySFTTrainer)
    stub.structured_graph = None
    stub.structured_head = None
    with pytest.raises(RuntimeError, match="structured_graph"):
        AuxiliarySFTTrainer._structured_targets(stub, ["IX MAN"], "cpu")


def test_signature_columns_protect_gold_gloss_when_structured_is_on():
    """Regression for job 7520: structured_scored_rows was 0 for 800+ steps.

    Root cause had nothing to do with Unsloth: Trainer.remove_unused_columns
    defaults to True (never overridden in this project) and strips any
    dataset column outside _signature_columns before the collator ever sees
    it. TRL's SFTTrainer signature list has no "gold_gloss", so it was always
    dropped upstream of compute_loss, silently. This test exercises the real
    mechanism (_signature_columns), not a hand-built inputs dict that already
    contains gold_gloss like the other tests in this file do.
    """
    from src.training.auxiliary_sft_trainer import AuxiliarySFTTrainer

    stub = object.__new__(AuxiliarySFTTrainer)
    stub._signature_columns = None
    stub._is_vision_dataset = False
    stub.structured_weight = 1.0
    stub.structured_head = object()
    stub.structured_loss = object()

    AuxiliarySFTTrainer._set_signature_columns_if_needed(stub)

    assert "gold_gloss" in stub._signature_columns


def test_signature_columns_leave_gold_gloss_out_when_structured_is_off():
    """Inertness contract: weight 0 must not add columns the collator sees."""
    from src.training.auxiliary_sft_trainer import AuxiliarySFTTrainer

    stub = object.__new__(AuxiliarySFTTrainer)
    stub._signature_columns = None
    stub._is_vision_dataset = False
    stub.structured_weight = 0.0
    stub.structured_head = None
    stub.structured_loss = None

    AuxiliarySFTTrainer._set_signature_columns_if_needed(stub)

    assert "gold_gloss" not in stub._signature_columns


def test_decode_gold_gloss_from_labels_recovers_completion_text(tokenizer):
    """Regression for job 7522: gold_gloss was fixed (2 commits) but
    structured_skip_reason stayed 1 (no gold) at step 200+.

    Root cause: Unsloth's compiled _prepare_dataset (replaces TRL's own once
    Unsloth is imported, materialized at unsloth_compiled_cache/UnslothSFTTrainer.py)
    tokenizes via dataset.map(_tokenize_pc, remove_columns=list(column_names)),
    dropping gold_gloss (and everything else) during tokenization — upstream
    of Trainer.remove_unused_columns and the collator, so neither of the
    previous two fixes could reach it. This test verifies the fallback: decode
    the labels!=-100 span back to text instead of depending on the column.
    """
    from src.training.auxiliary_sft_trainer import AuxiliarySFTTrainer

    prompt = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "translate: the man walks"},
    ]
    completion = [{"role": "assistant", "content": "IX MAN WALK"}]
    prompt_ids = tokenizer.apply_chat_template(
        prompt, tokenize=True, add_generation_prompt=True, return_dict=False
    )
    pc_ids = tokenizer.apply_chat_template(
        prompt + completion, tokenize=True, return_dict=True
    )["input_ids"]
    n_prompt = len(prompt_ids)

    labels = [-100] * n_prompt + pc_ids[n_prompt:]

    stub = object.__new__(AuxiliarySFTTrainer)
    stub.processing_class = tokenizer
    decoded = AuxiliarySFTTrainer._decode_gold_gloss_from_labels(
        stub,
        {
            "input_ids": torch.tensor([pc_ids]),
            "labels": torch.tensor([labels]),
        },
    )

    assert decoded == ["IX MAN WALK"]


def test_decode_gold_gloss_returns_none_without_tokenizer_or_labels():
    from src.training.auxiliary_sft_trainer import AuxiliarySFTTrainer

    stub = object.__new__(AuxiliarySFTTrainer)
    stub.processing_class = None
    assert (
        AuxiliarySFTTrainer._decode_gold_gloss_from_labels(
            stub,
            {"input_ids": torch.tensor([[1, 2]]), "labels": torch.tensor([[1, 2]])},
        )
        is None
    )


def test_structured_term_falls_back_to_decoded_labels_when_gold_gloss_is_absent():
    """End-to-end: no gold_gloss key at all in inputs (the real-world case
    under Unsloth), structured_states/lengths absent too — the term must
    still score by decoding labels, not silently skip with reason=1."""
    from src.datasets.structured_transitions import build_structured_transition_graph
    from src.models.structured_gloss_head import (
        StructuredGlossHead,
        StructuredGraphLoss,
    )
    from src.training.auxiliary_sft_trainer import AuxiliarySFTTrainer

    rows = [{"gloss": g} for g in ["IX MAN", "IX WALK", "MAN WALK", "IX MAN WALK"]]
    graph = build_structured_transition_graph(rows, top_k=3)
    head = StructuredGlossHead(hidden_size=8, num_states=graph.num_states, max_length=8)

    class _FakeTokenizer:
        def decode(self, ids, skip_special_tokens=True):
            # A trivial reversible "tokenizer": token id N -> chr(N).
            return "".join(chr(i) for i in ids)

    gloss_a = "IX MAN"
    gloss_b = "IX WALK"

    def _ids_for(text):
        return [ord(c) for c in text]

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
            self.processing_class = _FakeTokenizer()

    class _Outputs:
        logits = torch.zeros(2, 3, 5)
        hidden_states = [torch.randn(2, 3, 8)]

    def _stock(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        return (torch.tensor(2.0), _Outputs())

    import src.training.auxiliary_sft_trainer as module

    max_len = max(len(gloss_a), len(gloss_b))
    ids_a = _ids_for(gloss_a) + [0] * (max_len - len(gloss_a))
    ids_b = _ids_for(gloss_b) + [0] * (max_len - len(gloss_b))
    labels_a = _ids_for(gloss_a) + [-100] * (max_len - len(gloss_a))
    labels_b = _ids_for(gloss_b) + [-100] * (max_len - len(gloss_b))

    original = module.SFTTrainer.compute_loss
    module.SFTTrainer.compute_loss = _stock  # type: ignore[assignment]
    try:
        total = module.AuxiliarySFTTrainer.compute_loss(
            _Stub(),
            torch.nn.Linear(2, 2),
            {
                "input_ids": torch.tensor([ids_a, ids_b]),
                "labels": torch.tensor([labels_a, labels_b]),
                # deliberately no "gold_gloss" key: this is the real-world
                # shape of inputs once Unsloth has stripped it.
            },
        )
    finally:
        module.SFTTrainer.compute_loss = original  # type: ignore[assignment]

    assert torch.isfinite(total)
    assert float(total.detach()) != pytest.approx(2.0), "termine structured nullo"


def test_move_structured_modules_moves_both_head_and_loss():
    """Regression for job 7530: only structured_head was moved to the
    backbone's device; structured_loss's registered buffers (transition
    scores) stayed on CPU and the first real forward crashed inside
    log_partition with a cuda/cpu tensor mismatch. CPU-only tests can't
    reproduce a device mismatch directly, so this spies on .to() calls
    instead of asserting actual tensor placement."""
    from src.training.auxiliary_sft_trainer import AuxiliarySFTTrainer

    calls: dict[str, object] = {}

    class _Spy:
        def to(self, device):
            calls[type(self).__name__] = device
            return self

    class _Head(_Spy):
        pass

    class _Loss(_Spy):
        pass

    class _Model:
        device = "meta"

    stub = object.__new__(AuxiliarySFTTrainer)
    stub.model = _Model()
    stub.structured_head = _Head()
    stub.structured_loss = _Loss()

    AuxiliarySFTTrainer._move_structured_modules_to_device(stub)

    assert calls == {"_Head": "meta", "_Loss": "meta"}


def test_move_structured_modules_noop_when_either_is_absent():
    from src.training.auxiliary_sft_trainer import AuxiliarySFTTrainer

    class _Model:
        device = "meta"

    stub = object.__new__(AuxiliarySFTTrainer)
    stub.model = _Model()
    stub.structured_head = None
    stub.structured_loss = None

    AuxiliarySFTTrainer._move_structured_modules_to_device(stub)  # must not raise


def test_force_unsloth_return_logits_sets_env_var(monkeypatch):
    """Direct test of the callback class: both hooks must set the var."""
    from src.training.auxiliary_sft_trainer import _ForceUnslothReturnLogits

    monkeypatch.delenv("UNSLOTH_RETURN_LOGITS", raising=False)
    cb = _ForceUnslothReturnLogits()

    cb.on_train_begin(None, None, None)
    assert os.environ["UNSLOTH_RETURN_LOGITS"] == "1"

    monkeypatch.setenv("UNSLOTH_RETURN_LOGITS", "0")
    cb.on_step_begin(None, None, None)
    assert os.environ["UNSLOTH_RETURN_LOGITS"] == "1"


def test_mass_weight_positive_registers_force_return_logits_callback():
    """Regression for job 7536+: setting UNSLOTH_RETURN_LOGITS=1 before
    importing Unsloth is not sufficient — something inside Trainer.train()'s
    own setup resets it to "0" before the first forward pass (confirmed live
    on the cluster: a direct compute_loss() call returns real logits, but
    the same call through trainer.train() gets EmptyLogits, with the env var
    reading "0" at that exact point despite being set to "1" moments
    earlier). AuxiliarySFTTrainer must register a callback that re-asserts
    it every step, and only when mass_weight > 0 — structured_term never
    needs real logit values, so it must not pay this cost."""
    import src.training.auxiliary_sft_trainer as module

    original_init = module.SFTTrainer.__init__
    registered: list[Any] = []

    def _stub_init(self, *args: Any, **kwargs: Any) -> None:
        self.model = None
        self.add_callback = registered.append

    module.SFTTrainer.__init__ = _stub_init  # type: ignore[assignment]
    try:
        module.AuxiliarySFTTrainer(
            allowed_mask_fn=lambda prefixes, vocab, device: None,
            mass_weight=0.1,
        )
    finally:
        module.SFTTrainer.__init__ = original_init  # type: ignore[assignment]

    assert any(isinstance(cb, module._ForceUnslothReturnLogits) for cb in registered)


def test_mass_weight_zero_does_not_register_force_return_logits_callback():
    import src.training.auxiliary_sft_trainer as module

    original_init = module.SFTTrainer.__init__
    registered: list[Any] = []

    def _stub_init(self, *args: Any, **kwargs: Any) -> None:
        self.model = None
        self.add_callback = registered.append

    module.SFTTrainer.__init__ = _stub_init  # type: ignore[assignment]
    try:
        module.AuxiliarySFTTrainer()
    finally:
        module.SFTTrainer.__init__ = original_init  # type: ignore[assignment]

    assert not any(
        isinstance(cb, module._ForceUnslothReturnLogits) for cb in registered
    )


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
