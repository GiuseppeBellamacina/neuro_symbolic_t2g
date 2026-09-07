from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from src.models.auxiliary_sft_model import AuxiliarySFTModel
from src.models.structured_gloss_head import StructuredGlossHead


class FakeCausalLM(nn.Module):
    def __init__(self, *, output_calls: int = 1) -> None:
        super().__init__()
        self.embed = nn.Embedding(11, 5)
        self.projection = nn.Linear(5, 5)
        self.lm_head = nn.Linear(5, 11, bias=False)
        self.output_calls = output_calls
        self.config = SimpleNamespace(model_type="fake")
        self.generation_config = SimpleNamespace(max_length=9)
        self.checkpointing = False

    @property
    def device(self):
        return self.embed.weight.device

    @property
    def is_gradient_checkpointing(self):
        return self.checkpointing

    def get_input_embeddings(self):
        return self.embed

    def set_input_embeddings(self, value):
        self.embed = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, value):
        self.lm_head = value

    def gradient_checkpointing_enable(self, **_kwargs):
        self.checkpointing = True

    def gradient_checkpointing_disable(self):
        self.checkpointing = False

    def generate(self, input_ids, **_kwargs):
        return input_ids

    def prepare_inputs_for_generation(self, input_ids, **kwargs):
        return {"input_ids": input_ids, **kwargs}

    def forward(self, input_ids=None, inputs_embeds=None, **_kwargs):
        hidden = self.projection(
            self.embed(input_ids) if inputs_embeds is None else inputs_embeds
        )
        logits = None
        for _ in range(self.output_calls):
            logits = self.lm_head(hidden)
        if logits is None:
            logits = hidden.new_empty((*hidden.shape[:2], 11))
        return SimpleNamespace(logits=logits)

    def save_pretrained(self, output_dir):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), output_dir / "adapter_model.bin")
        (output_dir / "adapter_config.json").write_text(
            json.dumps({"fake": True}), encoding="utf-8"
        )


def make_wrapper(*, output_calls: int = 1):
    torch.manual_seed(4)
    backbone = FakeCausalLM(output_calls=output_calls)
    head = StructuredGlossHead(hidden_size=5, num_states=7, max_length=4)
    return AuxiliarySFTModel(backbone, head)


def test_forward_is_unchanged_and_capture_is_same_forward_input():
    wrapper = make_wrapper()
    input_ids = torch.tensor([[1, 2, 3], [4, 5, 6]])

    direct_logits = wrapper.backbone(input_ids=input_ids).logits
    wrapped_logits = wrapper(input_ids=input_ids).logits
    hidden = wrapper.consume_output_hidden()
    expected_hidden = wrapper.backbone.projection(wrapper.backbone.embed(input_ids))

    assert torch.equal(hidden, expected_hidden)
    assert torch.equal(wrapped_logits, direct_logits)
    assert torch.equal(wrapped_logits, wrapper.backbone.lm_head(expected_hidden))
    with pytest.raises(RuntimeError, match="no unconsumed"):
        wrapper.consume_output_hidden()


def test_capture_and_structured_head_preserve_both_gradient_paths():
    wrapper = make_wrapper()
    outputs = wrapper(input_ids=torch.tensor([[1, 2, 3]]))
    hidden = wrapper.consume_output_hidden()
    structured = wrapper.structured_head(hidden[:, 0], length=2)

    (outputs.logits.sum() + structured.sum()).backward()

    assert wrapper.backbone.embed.weight.grad is not None
    assert wrapper.structured_head.output.weight.grad is not None


@pytest.mark.parametrize("output_calls", [0, 2])
def test_output_head_must_be_called_exactly_once_without_stale_capture(output_calls):
    wrapper = make_wrapper(output_calls=output_calls)
    with pytest.raises(RuntimeError, match="exactly once|more than once"):
        wrapper(input_ids=torch.tensor([[1, 2]]))
    with pytest.raises(RuntimeError, match="no unconsumed"):
        wrapper.consume_output_hidden()

    wrapper.backbone.output_calls = 1
    wrapper(input_ids=torch.tensor([[3, 4]]))
    assert wrapper.consume_output_hidden().shape == (1, 2, 5)


def test_optimizer_sees_every_trainable_parameter_once():
    wrapper = make_wrapper()
    wrapper.backbone.projection.bias.requires_grad_(False)
    named = list(wrapper.named_parameters())
    trainable = [
        parameter for parameter in wrapper.parameters() if parameter.requires_grad
    ]
    optimizer = torch.optim.SGD(trainable, lr=0.1)
    optimized = [
        parameter for group in optimizer.param_groups for parameter in group["params"]
    ]

    assert any(name.startswith("backbone.") for name, _ in named)
    assert any(name.startswith("structured_head.") for name, _ in named)
    assert (
        len(optimized)
        == len({id(parameter) for parameter in optimized})
        == len(trainable)
    )
    assert all(
        parameter is not wrapper.backbone.projection.bias for parameter in optimized
    )


def test_structured_artifact_save_does_not_resave_or_replace_adapter(tmp_path):
    wrapper = make_wrapper()
    wrapper.backbone.save_pretrained(tmp_path)
    adapter = tmp_path / "adapter_model.bin"
    before = adapter.read_bytes()
    wrapper.save_structured_artifacts(tmp_path, {"identity": "graph"})
    assert adapter.read_bytes() == before
    assert (tmp_path / "structured_head.pt").is_file()
    assert (tmp_path / "auxiliary_sft_config.json").is_file()
    assert (tmp_path / "graph_manifest.json").is_file()


def test_structured_artifact_best_checkpoint_restores_matching_head(tmp_path):
    wrapper = make_wrapper()
    manifest = {"identity": "same-graph"}
    best = tmp_path / "checkpoint-1"
    last = tmp_path / "checkpoint-2"
    with torch.no_grad():
        wrapper.structured_head.output.weight.fill_(1.0)
    wrapper.save_structured_artifacts(best, manifest)
    with torch.no_grad():
        wrapper.structured_head.output.weight.fill_(2.0)
    wrapper.save_structured_artifacts(last, manifest)
    wrapper.load_auxiliary_components(best, manifest)
    assert torch.all(wrapper.structured_head.output.weight == 1.0)
    with pytest.raises(ValueError, match="manifest mismatch"):
        wrapper.load_auxiliary_components(best, {"identity": "other"})
    (best / "structured_head.pt").unlink()
    with pytest.raises(FileNotFoundError, match="structured_head.pt"):
        wrapper.load_auxiliary_components(best, manifest)


def test_save_load_roundtrip_and_adapter_artifacts(tmp_path):
    wrapper = make_wrapper()
    manifest = {"graph_protocol": 3, "states": ["A", "B"]}
    original = {
        name: value.detach().clone()
        for name, value in wrapper.structured_head.state_dict().items()
    }
    wrapper.save_auxiliary_components(tmp_path, manifest)

    assert (tmp_path / "adapter_model.bin").is_file()
    assert (tmp_path / "adapter_config.json").is_file()
    with torch.no_grad():
        for parameter in wrapper.structured_head.parameters():
            parameter.zero_()
    wrapper.load_auxiliary_components(tmp_path, manifest)

    for name, value in wrapper.structured_head.state_dict().items():
        assert torch.equal(value, original[name])


def test_load_fails_loudly_on_manifest_config_and_missing_artifacts(tmp_path):
    wrapper = make_wrapper()
    manifest = {"identity": "graph-a"}
    wrapper.save_auxiliary_components(tmp_path, manifest)

    with pytest.raises(ValueError, match="graph manifest mismatch"):
        wrapper.load_auxiliary_components(tmp_path, {"identity": "graph-b"})

    config_path = tmp_path / "auxiliary_sft_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["structured_head"]["num_states"] += 1
    config_path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError, match="configuration mismatch"):
        wrapper.load_auxiliary_components(tmp_path, manifest)

    config_path.unlink()
    with pytest.raises(FileNotFoundError, match="auxiliary_sft_config.json"):
        wrapper.load_auxiliary_components(tmp_path, manifest)


def test_optional_head_and_generation_delegation():
    backbone = FakeCausalLM()
    wrapper = AuxiliarySFTModel(backbone)
    input_ids = torch.tensor([[1, 2]])

    assert wrapper.structured_head is None
    assert wrapper.backbone_for_generation is backbone
    assert torch.equal(wrapper.generate(input_ids), input_ids)
    wrapper.gradient_checkpointing_enable()
    assert wrapper.is_gradient_checkpointing
    wrapper.gradient_checkpointing_disable()
    assert not wrapper.is_gradient_checkpointing
