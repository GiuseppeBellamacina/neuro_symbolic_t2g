from copy import deepcopy
from typing import cast

import pytest
import torch


def test_completion_metadata_collator_preserves_trl_labels_and_metadata() -> None:
    from trl.trainer.sft_trainer import DataCollatorForLanguageModeling

    from src.training.auxiliary_sft_trainer import CompletionMetadataCollator

    features = [
        {"input_ids": [1, 2, 3, 9], "completion_mask": [0, 0, 1, 1]},
        {"input_ids": [1, 4, 9], "completion_mask": [0, 1, 1]},
    ]
    stock = DataCollatorForLanguageModeling(pad_token_id=0, completion_only_loss=True)
    wrapped = CompletionMetadataCollator(
        pad_token_id=0, eos_token_id=9, completion_only_loss=True
    )
    stock_batch = stock([dict(row) for row in features])
    batch = wrapped([dict(row) for row in features])

    assert torch.equal(batch["labels"], stock_batch["labels"])
    assert batch["completion_start"].tolist() == [2, 1]
    assert batch["completion_mask"].dtype == torch.bool
    assert batch["completion_mask"].tolist() == [
        [False, False, True, True],
        [False, True, True, False],
    ]
    assert batch["truncation_eligible"].tolist() == [True, True]


def test_completion_metadata_collator_marks_missing_eos_as_ineligible() -> None:
    from src.training.auxiliary_sft_trainer import CompletionMetadataCollator

    collator = CompletionMetadataCollator(
        pad_token_id=0, eos_token_id=9, completion_only_loss=True
    )
    batch = collator(
        [
            {"input_ids": [1, 2, 3], "completion_mask": [0, 1, 1]},
        ]
    )
    assert batch["truncation_eligible"].tolist() == [False]
    assert collator.diagnostics["missing_eos_or_truncated"] == 1


@pytest.mark.parametrize(
    "features,match",
    [
        (
            [{"input_ids": [1], "completion_mask": [1]}, {"input_ids": [1]}],
            "feature 1.*completion_mask",
        ),
        ([{"input_ids": [1], "completion_mask": []}], "nonempty completion_mask"),
        ([{"completion_mask": [1]}], "nonempty input_ids"),
        ([{"input_ids": [1, 2], "completion_mask": [1]}], "length mismatch"),
        ([{"input_ids": [1], "completion_mask": [1, 0]}], "length mismatch"),
        ([{"input_ids": [1], "completion_mask": [2]}], "bool or 0/1"),
        ([{"input_ids": [1], "completion_mask": [0.5]}], "bool or 0/1"),
        ([{"input_ids": [1], "completion_mask": [0]}], "select at least one"),
        (
            [{"input_ids": [1, 9], "completion_mask": [0, 1], "labels": [-100]}],
            "labels length",
        ),
        (
            [{"input_ids": [1, 9], "completion_mask": [0, 1], "labels": [1, 9]}],
            "explicit labels conflict",
        ),
    ],
)
def test_completion_metadata_collator_rejects_malformed(features, match) -> None:
    from src.training.auxiliary_sft_trainer import CompletionMetadataCollator

    collator = CompletionMetadataCollator(
        pad_token_id=0, eos_token_id=9, completion_only_loss=True
    )
    with pytest.raises(ValueError, match=match):
        collator(features)


def test_collator_is_immutable_and_pad_multiple_matches_stock() -> None:
    from trl.trainer.sft_trainer import DataCollatorForLanguageModeling

    from src.training.auxiliary_sft_trainer import CompletionMetadataCollator

    features = [{"input_ids": [1, 2, 9], "completion_mask": [0, 1, 1]}]
    original = deepcopy(features)
    stock = DataCollatorForLanguageModeling(
        pad_token_id=0, completion_only_loss=True, pad_to_multiple_of=8
    )
    wrapped = CompletionMetadataCollator(
        pad_token_id=0,
        eos_token_id=9,
        completion_only_loss=True,
        pad_to_multiple_of=8,
    )
    expected = stock(deepcopy(features))
    actual = wrapped(features)
    assert features == original
    assert actual["input_ids"].shape == (1, 8)
    assert torch.equal(actual["labels"], expected["labels"])
    assert actual["completion_mask"].tolist() == [[False, True, True] + [False] * 5]


def test_auxiliary_trainer_training_eval_warmup_and_metadata(monkeypatch) -> None:
    from collections import defaultdict
    from types import SimpleNamespace

    from trl.trainer.sft_trainer import SFTTrainer

    from src.training.auxiliary_sft_trainer import AuxiliaryMassSFTTrainer

    class Machine:
        eos_token_id = 2

        def initial_state(self):
            return 0

        def allowed_token_ids(self, state):
            return (1,) if state == 0 else (2,)

        def advance(self, state, token_id):
            return state + 1

        def is_invalid(self, state):
            return False

    parameter = torch.tensor(0.25, requires_grad=True)
    seen_inputs = []

    def stock_compute(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        seen_inputs.append(set(inputs))
        logits = torch.zeros(1, 4, 4) + parameter
        logits = (
            logits
            + torch.nn.functional.one_hot(torch.ones(1, 4, dtype=torch.long), 4)
            * parameter
        )
        output = SimpleNamespace(logits=logits)
        loss = parameter.square()
        return (loss, output) if return_outputs else loss

    monkeypatch.setattr(SFTTrainer, "compute_loss", stock_compute)
    trainer = object.__new__(AuxiliaryMassSFTTrainer)
    trainer.mass_state_machine = Machine()
    trainer.mass_lambda = 0.1
    trainer.mass_warmup_steps = 10
    trainer.state = SimpleNamespace(global_step=5)  # type: ignore[assignment]
    trainer._metrics = defaultdict(lambda: defaultdict(list))
    model = torch.nn.Linear(1, 1)
    labels = torch.tensor([[-100, 1, 2, -100]])

    def inputs():
        return {
            "labels": labels,
            "attention_mask": torch.ones_like(labels),
            "completion_start": torch.tensor([1]),
            "completion_mask": torch.tensor([[False, True, True, False]]),
            "truncation_eligible": torch.tensor([True]),
        }

    model.train()
    train_loss = cast(torch.Tensor, trainer.compute_loss(model, inputs()))
    assert train_loss > parameter.square()
    assert trainer.mass_weight() == 0.05
    assert all("completion_start" not in keys for keys in seen_inputs)
    assert trainer._metrics["train"]["mass_scored_samples"] == [1]
    assert trainer._metrics["train"]["mass_loss_sum"]

    model.eval()
    eval_loss = cast(torch.Tensor, trainer.compute_loss(model, inputs()))
    assert torch.equal(eval_loss, parameter.square())

    trainer.mass_lambda = 0.0
    model.train()
    zero_loss = cast(torch.Tensor, trainer.compute_loss(model, inputs()))
    stock_loss = parameter.square()
    zero_grad = torch.autograd.grad(zero_loss, parameter)[0]
    stock_grad = torch.autograd.grad(stock_loss, parameter)[0]
    assert torch.equal(zero_loss, stock_loss)
    assert torch.equal(zero_grad, stock_grad)


def test_structured_trainer_formula_boundary_skips_gradients_and_eval(
    monkeypatch,
) -> None:
    from collections import defaultdict
    from types import SimpleNamespace

    from torch import nn
    from trl.trainer.sft_trainer import SFTTrainer

    from src.datasets.structured_transitions import build_structured_transition_graph
    from src.models.auxiliary_sft_model import AuxiliarySFTModel
    from src.models.structured_gloss_head import (
        StructuredGlossHead,
        StructuredGraphLoss,
    )
    from src.training.auxiliary_sft_trainer import AuxiliarySFTTrainer

    class Backbone(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Embedding(8, 4)
            self.lm_head = nn.Linear(4, 8, bias=False)
            self.config = SimpleNamespace(hidden_size=4)

        def get_output_embeddings(self):
            return self.lm_head

        def forward(self, input_ids=None, labels=None, **kwargs):
            hidden = self.embed(input_ids)
            return SimpleNamespace(logits=self.lm_head(hidden))

    graph = build_structured_transition_graph(
        [{"sample_id": "a", "gloss": "A B"}], top_k=2
    )
    model = AuxiliarySFTModel(
        Backbone(), StructuredGlossHead(4, graph.num_states, max_length=3)
    )
    seen_keys = []

    def stock_compute(
        self, forwarded_model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        seen_keys.append(set(inputs))
        outputs = forwarded_model(**inputs)
        lm = outputs.logits.square().mean()
        return (lm, outputs) if return_outputs else lm

    monkeypatch.setattr(SFTTrainer, "compute_loss", stock_compute)
    trainer = object.__new__(AuxiliarySFTTrainer)
    trainer.mass_lambda = 0.0
    trainer.structured_lambda = 0.4
    trainer.structured_warmup_steps = 10
    trainer.structured_loss = StructuredGraphLoss(graph)
    trainer.state = SimpleNamespace(global_step=5)
    trainer._metrics = defaultdict(lambda: defaultdict(list))
    batch = {
        "input_ids": torch.tensor([[1, 2, 3, 4], [4, 3, 2, 1]]),
        "labels": torch.tensor([[-100, -100, 1, 2], [-100, 1, 2, 3]]),
        "completion_start": torch.tensor([2, 1]),
        "completion_mask": torch.tensor(
            [[False, False, True, True], [False, True, True, True]]
        ),
        "truncation_eligible": torch.tensor([True, True]),
        "structured_gold_states": torch.tensor([[0, 1, 0], [0, 0, 0]]),
        "structured_length": torch.tensor([2, 0]),
        "structured_eligible": torch.tensor([True, False]),
        "sample_id": ["a", "skip"],
    }
    model.train()
    loss = trainer.compute_loss(model, dict(batch))
    lm = trainer._metrics["train"]["structured_lm_loss"][0]
    aux = trainer._metrics["train"]["structured_objective_loss"][0]
    assert loss.item() == pytest.approx(lm + 0.2 * aux)
    assert trainer._metrics["train"]["structured_weight"] == [0.2]
    assert trainer._metrics["train"]["structured_valid_samples"] == [1]
    assert trainer._metrics["train"]["structured_skipped_samples"] == [1]
    assert all(
        "structured_gold_states" not in keys and "completion_start" not in keys
        for keys in seen_keys
    )
    loss.backward()
    assert model.backbone.embed.weight.grad is not None
    assert model.structured_head.output.weight.grad is not None
    # The eligible row starts at 2, hence its source-only boundary is position 1.
    expected_boundary = model.backbone.embed(batch["input_ids"])[0, 1]
    captured_emission = model.structured_head(expected_boundary[None, :], length=2)
    assert captured_emission.shape == (1, 2, graph.num_states)

    model.eval()
    eval_loss = trainer.compute_loss(model, dict(batch))
    assert eval_loss.item() >= 0
    with pytest.raises(RuntimeError, match="no unconsumed"):
        model.consume_output_hidden()


def test_diagnostic_sums_reconstruct_token_weighted_microbatch_mean() -> None:
    machine = FakeTrie({}, {(): (5,)})
    first_logits = torch.zeros(1, 2, 6)
    second_logits = torch.zeros(1, 4, 6)
    first = allowed_mass_loss(
        first_logits, torch.tensor([[-100, 5]]), torch.tensor([1]), machine
    )[1]
    second_machine = FakeTrie(
        {((), 1): (1,), ((1,), 2): (1, 2)},
        {(): (1,), (1,): (2,), (1, 2): (5,)},
    )
    second = allowed_mass_loss(
        second_logits,
        torch.tensor([[-100, 1, 2, 5]]),
        torch.tensor([1]),
        second_machine,
    )[1]
    combined_loss = (first.loss_sum + second.loss_sum) / (
        first.scored_tokens + second.scored_tokens
    )
    combined_log_mass = (first.log_allowed_mass_sum + second.log_allowed_mass_sum) / (
        first.scored_tokens + second.scored_tokens
    )
    assert combined_loss == pytest.approx(-combined_log_mass)
    assert first.allowed_mass_sum == pytest.approx(first.mean_allowed_mass)
    assert second.allowed_mass_sum == pytest.approx(
        second.mean_allowed_mass * second.scored_tokens
    )


from src.training.allowed_mass_loss import allowed_mass_loss


class TinyTokenizer:
    eos_token_id = 9
    pad_token_id = 9
    vocab_size = 12
    _encodings = {
        "A": [1],
        "AB": [1, 2],
        "LONG": [3, 4],
        "B": [7],
        " A": [5],
        " AB": [5, 2],
        " LONG": [6, 4],
        " B": [8],
    }

    def encode(self, text, add_special_tokens=False):
        assert not add_special_tokens
        return list(self._encodings[text])


class TinyMask:
    tokenizer = TinyTokenizer()
    vocab = ["<BOS>", "<EOS>", "<UNK>", "A", "AB", "LONG", "B"]
    token_ids = {1, 2, 3, 4, 5, 6, 7, 8, 9}


def test_eos_only_target_is_valid_with_real_dual_root_trie() -> None:
    from src.grammar.grammar_logits_processor import DualRootGlossTrie

    state_machine = DualRootGlossTrie.from_vocabulary(
        TinyMask.vocab, TinyMask.tokenizer
    )
    logits = torch.zeros(1, 2, TinyMask.tokenizer.vocab_size)
    loss, diagnostics = allowed_mass_loss(
        logits,
        torch.tensor([[-100, TinyMask.tokenizer.eos_token_id]]),
        torch.tensor([1]),
        state_machine,
        completion_mask=torch.tensor([[False, True]]),
        truncation_eligible=torch.tensor([True]),
    )
    assert torch.isfinite(loss)
    assert diagnostics.scored_tokens == 1
    assert diagnostics.scored_samples == 1


class FakeTrie:
    eos_token_id = 5

    def __init__(self, transitions, allowed):
        self.transitions = transitions
        self.allowed = allowed

    def initial_state(self):
        return ()

    def allowed_token_ids(self, state):
        return self.allowed[state]

    def advance(self, state, token_id):
        return self.transitions.get((state, token_id), ("invalid",))

    def is_invalid(self, state):
        return state == ("invalid",)


def trie():
    return FakeTrie(
        {((), 1): (1,), ((1,), 2): (1, 2)},
        {(): (1, 3, 5), (1,): (2, 5), (1, 2): (5,)},
    )


def batch(tokens, start=2, sequence_length=7):
    labels = torch.full((1, sequence_length), -100, dtype=torch.long)
    labels[0, start : start + len(tokens)] = torch.tensor(tokens)
    return labels, torch.tensor([start])


def test_matches_brute_force_softmax_and_uses_strict_causal_alignment():
    labels, starts = batch([1, 2, 5])
    logits = torch.randn(1, 7, 6, dtype=torch.float64)
    logits[0, 0] = 1000  # must not affect the first completion target at q=2
    loss, diagnostics = allowed_mass_loss(logits, labels, starts, trie())

    allowed = [(1, 3, 5), (2, 5), (5,)]
    brute = []
    for position, ids in zip((2, 3, 4), allowed, strict=True):
        probabilities = logits[0, position - 1].float().softmax(dim=-1)
        brute.append(-probabilities[list(ids)].sum().log())
    torch.testing.assert_close(loss, torch.stack(brute).mean())
    assert diagnostics.scored_tokens == 3
    assert diagnostics.scored_samples == 1
    assert diagnostics.mean_allowed_mass == pytest.approx(
        torch.stack([-value for value in brute]).exp().mean().item()
    )


@pytest.mark.parametrize("tokens, expected", [([5, 4, 4], 1), ([1, 5, 4], 2)])
def test_eos_is_scored_and_trailing_tokens_are_ignored(tokens, expected):
    labels, starts = batch(tokens)
    logits = torch.randn(1, 7, 6)
    loss, diagnostics = allowed_mass_loss(logits, labels, starts, trie())
    assert torch.isfinite(loss)
    assert diagnostics.scored_tokens == expected
    assert diagnostics.invalid_prefix == 0


def test_missing_eos_and_ineligible_truncation_skip_whole_samples():
    labels = torch.tensor([[-100, -100, 1, 2], [-100, -100, 1, 5]])
    starts = torch.tensor([2, 2])
    logits = torch.randn(2, 4, 6, requires_grad=True)
    loss, diagnostics = allowed_mass_loss(
        logits,
        labels,
        starts,
        trie(),
        truncation_eligible=torch.tensor([True, False]),
    )
    assert loss.item() == 0
    assert diagnostics.skipped_truncated == 2
    assert diagnostics.scored_tokens == 0
    loss.backward()
    assert logits.grad is not None and torch.count_nonzero(logits.grad) == 0


def test_invalid_prefix_is_counted_without_nan():
    labels, starts = batch([4, 5])
    logits = torch.randn(1, 7, 6)
    loss, diagnostics = allowed_mass_loss(logits, labels, starts, trie())
    assert torch.isfinite(loss)
    assert loss.item() == 0
    assert diagnostics.scored_tokens == 0
    assert diagnostics.scored_samples == 0
    assert diagnostics.invalid_prefix == 1


def test_all_vocab_allowed_is_zero_and_extreme_logits_are_finite():
    all_vocab = FakeTrie({}, {(): tuple(range(6))})
    labels, starts = batch([5])
    logits = torch.tensor([[[0.0] * 6, [1e30, -1e30, 0, 0, 0, 0], [0.0] * 6]])
    short_labels = labels[:, :3]
    loss, diagnostics = allowed_mass_loss(logits, short_labels, starts, all_vocab)
    assert torch.isfinite(loss)
    assert loss.item() == pytest.approx(0.0, abs=1e-6)
    assert diagnostics.mean_allowed_mass == pytest.approx(1.0, abs=1e-6)


def test_explicit_starts_make_left_and_right_padding_invariant():
    base_logits = torch.randn(1, 5, 6)
    right_labels = torch.tensor([[-100, 1, 2, 5, -100]])
    right_loss, _ = allowed_mass_loss(
        base_logits, right_labels, torch.tensor([1]), trie()
    )

    left_logits = torch.cat((torch.randn(1, 2, 6), base_logits), dim=1)
    left_labels = torch.tensor([[-100, -100, -100, 1, 2, 5, -100]])
    left_loss, _ = allowed_mass_loss(
        left_logits, left_labels, torch.tensor([3]), trie()
    )
    torch.testing.assert_close(left_loss, right_loss)


def test_gradients_are_finite_and_step_increases_allowed_mass():
    labels, starts = batch([1, 2, 5])
    logits = torch.randn(1, 7, 6, requires_grad=True)
    loss, before = allowed_mass_loss(logits, labels, starts, trie())
    loss.backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()

    with torch.no_grad():
        updated = logits - logits.grad
    _, after = allowed_mass_loss(updated, labels, starts, trie())
    assert after.mean_allowed_mass > before.mean_allowed_mass


def test_completion_mask_and_metadata_validation():
    labels, starts = batch([1, 2, 5])
    mask = labels != -100
    loss, diagnostics = allowed_mass_loss(
        torch.randn(1, 7, 6), labels, starts, trie(), completion_mask=mask
    )
    assert torch.isfinite(loss) and diagnostics.scored_tokens == 3

    bad_labels = labels.clone()
    bad_labels[0, 0] = 1
    with pytest.raises(ValueError, match="before completion_start"):
        allowed_mass_loss(torch.randn(1, 7, 6), bad_labels, starts, trie())


@pytest.mark.parametrize(
    "tokens, expected_allowed",
    [
        ([1, 2, 9], [(1, 3, 7, 9), (2, 5, 6, 8, 9), (5, 6, 8, 9)]),
        ([3, 4, 6, 4, 9], [(1, 3, 7, 9), (4,), (5, 6, 8, 9), (4,), (5, 6, 8, 9)]),
    ],
)
def test_real_trie_loss_matches_brute_force_and_processor_masks(
    tokens, expected_allowed
):
    from src.grammar.grammar_logits_processor import (
        DualRootGlossTrie,
        GlossVocabularyLogitsProcessor,
    )

    shared_trie = DualRootGlossTrie.from_vocabulary(TinyMask.vocab, TinyMask.tokenizer)
    labels = torch.full((1, len(tokens) + 2), -100, dtype=torch.long)
    labels[0, 1 : len(tokens) + 1] = torch.tensor(tokens)
    logits = torch.randn(1, labels.shape[1], TinyTokenizer.vocab_size)
    loss, diagnostics = allowed_mass_loss(
        logits, labels, torch.tensor([1]), shared_trie
    )

    processor = GlossVocabularyLogitsProcessor(TinyMask(), device="cpu")
    prompt = [10]
    processor(torch.tensor([prompt]), torch.zeros(1, TinyTokenizer.vocab_size))
    brute = []
    history = []
    for offset, allowed in enumerate(expected_allowed):
        output = processor(
            torch.tensor([prompt + history]),
            torch.zeros(1, TinyTokenizer.vocab_size),
        )
        processor_allowed = tuple(
            output[0].isfinite().nonzero(as_tuple=True)[0].tolist()
        )
        assert processor_allowed == allowed
        probabilities = logits[0, offset].softmax(dim=-1)
        brute.append(-probabilities[list(allowed)].sum().log())
        history.append(tokens[offset])

    torch.testing.assert_close(loss, torch.stack(brute).mean())
    assert diagnostics.scored_tokens == len(tokens)
    assert diagnostics.scored_samples == 1
    assert diagnostics.invalid_prefix == 0


def test_real_trie_invalid_target_skips_entire_sample_without_root_recovery():
    from src.grammar.grammar_logits_processor import DualRootGlossTrie

    shared_trie = DualRootGlossTrie.from_vocabulary(TinyMask.vocab, TinyMask.tokenizer)
    labels = torch.tensor([[-100, 11, 1, 9]])
    logits = torch.randn(1, 4, TinyTokenizer.vocab_size, requires_grad=True)
    loss, diagnostics = allowed_mass_loss(
        logits, labels, torch.tensor([1]), shared_trie
    )
    assert loss.item() == 0
    assert diagnostics.invalid_prefix == 1
    assert diagnostics.scored_tokens == 0
    assert diagnostics.scored_samples == 0
    loss.backward()
    assert logits.grad is not None and torch.count_nonzero(logits.grad) == 0


def test_real_trie_shared_eos_pad_id_is_scored_once_as_semantic_eos():
    from src.grammar.grammar_logits_processor import DualRootGlossTrie

    shared_trie = DualRootGlossTrie.from_vocabulary(TinyMask.vocab, TinyMask.tokenizer)
    labels = torch.tensor([[-100, 1, 9, 9]])
    logits = torch.randn(1, 4, TinyTokenizer.vocab_size)
    _, diagnostics = allowed_mass_loss(logits, labels, torch.tensor([1]), shared_trie)
    assert diagnostics.scored_tokens == 2
    assert diagnostics.scored_samples == 1
    assert diagnostics.invalid_prefix == 0
