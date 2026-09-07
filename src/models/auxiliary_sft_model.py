"""Causal-LM wrapper for structured auxiliary SFT objectives."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

from src.models.structured_gloss_head import StructuredGlossHead

AUXILIARY_PROTOCOL_VERSION = 1
_CONFIG_FILE = "auxiliary_sft_config.json"
_HEAD_FILE = "structured_head.pt"
_MANIFEST_FILE = "graph_manifest.json"


class AuxiliarySFTModel(nn.Module):
    """Keep a causal LM and optional structured head in one module tree.

    The output-embedding pre-hook captures the exact hidden tensor consumed by
    the LM head. The capture belongs to one successful forward and must be
    consumed before its graph is released by the caller.
    """

    def __init__(
        self, backbone: nn.Module, structured_head: StructuredGlossHead | None = None
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.structured_head = structured_head
        self._output_hidden: Tensor | None = None
        self._capture_active = False
        self._capture_count = 0

        output_embeddings = backbone.get_output_embeddings()
        if not isinstance(output_embeddings, nn.Module):
            raise TypeError("backbone.get_output_embeddings() must return an nn.Module")
        self._output_hook = output_embeddings.register_forward_pre_hook(
            self._capture_output_hidden
        )

    def _capture_output_hidden(
        self, _module: nn.Module, inputs: tuple[Any, ...]
    ) -> None:
        if not self._capture_active:
            return
        self._capture_count += 1
        if self._capture_count > 1:
            raise RuntimeError(
                "output embeddings were called more than once in one forward"
            )
        if len(inputs) != 1 or not isinstance(inputs[0], Tensor):
            raise RuntimeError(
                "output embeddings must receive exactly one positional hidden tensor"
            )
        hidden = inputs[0]
        if hidden.ndim != 3:
            raise RuntimeError("captured output hidden must have shape [B,S,H]")
        self._output_hidden = hidden

    @staticmethod
    def _logits_from(outputs: Any) -> Tensor:
        logits = (
            outputs.get("logits")
            if isinstance(outputs, dict)
            else getattr(outputs, "logits", None)
        )
        if not isinstance(logits, Tensor):
            raise RuntimeError("backbone forward output must expose tensor logits")
        return logits

    def forward(
        self,
        input_ids: Tensor | None = None,
        attention_mask: Tensor | None = None,
        labels: Tensor | None = None,
        **kwargs: Any,
    ) -> Any:
        self._output_hidden = None
        self._capture_count = 0
        self._capture_active = True
        call_kwargs = dict(kwargs)
        if input_ids is not None:
            call_kwargs["input_ids"] = input_ids
        if attention_mask is not None:
            call_kwargs["attention_mask"] = attention_mask
        if labels is not None:
            call_kwargs["labels"] = labels
        try:
            outputs = self.backbone(**call_kwargs)
            if self._capture_count != 1 or self._output_hidden is None:
                raise RuntimeError(
                    "backbone forward must call output embeddings exactly once"
                )
            logits = self._logits_from(outputs)
            if self._output_hidden.shape[:2] != logits.shape[:2]:
                raise RuntimeError(
                    "captured output hidden batch/sequence shape does not match logits"
                )
            return outputs
        except Exception:
            self._output_hidden = None
            self._capture_count = 0
            raise
        finally:
            self._capture_active = False

    def consume_output_hidden(self) -> Tensor:
        """Return and immediately clear the current forward's graph-bearing capture."""
        hidden = self._output_hidden
        self._output_hidden = None
        self._capture_count = 0
        if hidden is None:
            raise RuntimeError("no unconsumed output hidden is available")
        return hidden

    @property
    def backbone_for_generation(self) -> nn.Module:
        return self.backbone

    @property
    def config(self) -> Any:
        return self.backbone.config

    @property
    def generation_config(self) -> Any:
        return getattr(self.backbone, "generation_config", None)

    @property
    def main_input_name(self) -> str:
        return getattr(self.backbone, "main_input_name", "input_ids")

    @property
    def device(self) -> torch.device:
        backbone_device = getattr(self.backbone, "device", None)
        if backbone_device is not None:
            return torch.device(backbone_device)
        try:
            return next(self.backbone.parameters()).device
        except StopIteration:
            return torch.device("cpu")

    @property
    def is_gradient_checkpointing(self) -> bool:
        return bool(getattr(self.backbone, "is_gradient_checkpointing", False))

    def get_input_embeddings(self) -> nn.Module:
        return self.backbone.get_input_embeddings()

    def set_input_embeddings(self, value: nn.Module) -> None:
        self.backbone.set_input_embeddings(value)

    def get_output_embeddings(self) -> nn.Module:
        return self.backbone.get_output_embeddings()

    def set_output_embeddings(self, value: nn.Module) -> None:
        self._output_hook.remove()
        self.backbone.set_output_embeddings(value)
        output_embeddings = self.backbone.get_output_embeddings()
        if not isinstance(output_embeddings, nn.Module):
            raise TypeError("backbone.get_output_embeddings() must return an nn.Module")
        self._output_hook = output_embeddings.register_forward_pre_hook(
            self._capture_output_hidden
        )

    def gradient_checkpointing_enable(self, **kwargs: Any) -> Any:
        return self.backbone.gradient_checkpointing_enable(**kwargs)

    def gradient_checkpointing_disable(self) -> Any:
        return self.backbone.gradient_checkpointing_disable()

    def generate(self, *args: Any, **kwargs: Any) -> Any:
        return self.backbone.generate(*args, **kwargs)

    def prepare_inputs_for_generation(self, *args: Any, **kwargs: Any) -> Any:
        return self.backbone.prepare_inputs_for_generation(*args, **kwargs)

    def can_generate(self) -> bool:
        method = getattr(self.backbone, "can_generate", None)
        return (
            bool(method()) if callable(method) else hasattr(self.backbone, "generate")
        )

    def _head_config(self) -> dict[str, int]:
        head = self.structured_head
        if not isinstance(head, StructuredGlossHead):
            raise RuntimeError(
                "a StructuredGlossHead is required for auxiliary artifacts"
            )
        return {
            "hidden_size": head.position.embedding_dim,
            "num_states": head.output.out_features,
            "max_length": head.max_length,
        }

    def save_auxiliary_components(
        self, output_dir: str | Path, graph_manifest: dict[str, Any]
    ) -> None:
        """Save adapter and structured artifacts under one checkpoint directory."""
        if not isinstance(graph_manifest, dict) or not graph_manifest:
            raise ValueError("graph_manifest must be a nonempty dictionary")
        head_config = self._head_config()
        head = self.structured_head
        if not isinstance(head, StructuredGlossHead):
            raise RuntimeError(
                "a StructuredGlossHead is required for auxiliary artifacts"
            )
        save_pretrained = getattr(self.backbone, "save_pretrained", None)
        if not callable(save_pretrained):
            raise TypeError("backbone must provide save_pretrained")

        destination = Path(output_dir)
        destination.parent.mkdir(parents=True, exist_ok=True)
        staging = Path(
            tempfile.mkdtemp(prefix=f".{destination.name}.aux-", dir=destination.parent)
        )
        try:
            save_pretrained(staging)
            torch.save(head.state_dict(), staging / _HEAD_FILE)
            (staging / _CONFIG_FILE).write_text(
                json.dumps(
                    {
                        "protocol_version": AUXILIARY_PROTOCOL_VERSION,
                        "structured_head": head_config,
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
            (staging / _MANIFEST_FILE).write_text(
                json.dumps(graph_manifest, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            destination.mkdir(parents=True, exist_ok=True)
            for source in staging.rglob("*"):
                relative = source.relative_to(staging)
                target = destination / relative
                if source.is_dir():
                    target.mkdir(parents=True, exist_ok=True)
                else:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    os.replace(source, target)
        finally:
            shutil.rmtree(staging, ignore_errors=True)

    def save_structured_artifacts(
        self, output_dir: str | Path, graph_manifest: dict[str, Any]
    ) -> None:
        """Add only structured files after Trainer has saved the backbone.

        This deliberately does not call ``backbone.save_pretrained``: Trainer's
        save path is authoritative for adapter/model files and trainer state.
        """
        if not isinstance(graph_manifest, dict) or not graph_manifest:
            raise ValueError("graph_manifest must be a nonempty dictionary")
        head = self.structured_head
        if not isinstance(head, StructuredGlossHead):
            raise RuntimeError(
                "a StructuredGlossHead is required for auxiliary artifacts"
            )
        destination = Path(output_dir)
        destination.mkdir(parents=True, exist_ok=True)
        torch.save(head.state_dict(), destination / _HEAD_FILE)
        (destination / _CONFIG_FILE).write_text(
            json.dumps(
                {
                    "protocol_version": AUXILIARY_PROTOCOL_VERSION,
                    "structured_head": self._head_config(),
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        (destination / _MANIFEST_FILE).write_text(
            json.dumps(graph_manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    def load_auxiliary_components(
        self, output_dir: str | Path, graph_manifest: dict[str, Any]
    ) -> None:
        """Validate the checkpoint contract and strictly restore the head."""
        directory = Path(output_dir)
        required = [
            directory / name for name in (_CONFIG_FILE, _HEAD_FILE, _MANIFEST_FILE)
        ]
        missing = [path.name for path in required if not path.is_file()]
        if missing:
            raise FileNotFoundError(
                f"missing auxiliary checkpoint artifacts: {', '.join(missing)}"
            )
        config = json.loads((directory / _CONFIG_FILE).read_text(encoding="utf-8"))
        if config.get("protocol_version") != AUXILIARY_PROTOCOL_VERSION:
            raise ValueError("auxiliary checkpoint protocol version mismatch")
        if config.get("structured_head") != self._head_config():
            raise ValueError("structured head configuration mismatch")
        saved_manifest = json.loads(
            (directory / _MANIFEST_FILE).read_text(encoding="utf-8")
        )
        if saved_manifest != graph_manifest:
            raise ValueError("graph manifest mismatch")
        state = torch.load(
            directory / _HEAD_FILE, map_location=self.device, weights_only=True
        )
        head = self.structured_head
        if not isinstance(head, StructuredGlossHead):
            raise RuntimeError(
                "a StructuredGlossHead is required for auxiliary artifacts"
            )
        head.load_state_dict(state, strict=True)
