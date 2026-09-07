"""Opt-in mass and structured objectives for standalone SFT.

The classes in this module are intentionally absent from the normal SFT path.
They preserve TRL's labels and standard LM loss while carrying the completion
boundary metadata needed by :func:`allowed_mass_loss`.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Any, cast

import torch
from torch import Tensor
from trl import SFTTrainer  # type: ignore[import]
from trl.trainer.sft_trainer import DataCollatorForLanguageModeling

from src.models.auxiliary_sft_model import AuxiliarySFTModel
from src.models.structured_gloss_head import (
    StructuredGraphLoss,
    gather_assistant_boundary,
)
from src.training.allowed_mass_loss import AllowedTokenStateMachine, allowed_mass_loss
from src.training.structured_sft import structured_weight

TRIE_PROTOCOL_VERSION = 1


def canonical_vocabulary(vocab: list[str]) -> tuple[str, ...]:
    """Canonical Trie vocabulary: stripped, nonempty, sorted unique strings."""
    return tuple(sorted({str(item).strip() for item in vocab if str(item).strip()}))


def build_auxiliary_trie_manifest(
    vocab: list[str], vocab_path: str | Path, tokenizer: Any
) -> dict[str, Any]:
    """Build deterministic identity for the exact Trie/tokenizer contract."""
    canonical = canonical_vocabulary(vocab)
    compiled_entries = [
        (
            gloss,
            list(tokenizer.encode(gloss, add_special_tokens=False)),
            list(tokenizer.encode(" " + gloss, add_special_tokens=False)),
        )
        for gloss in canonical
    ]
    compiled_json = json.dumps(
        compiled_entries, ensure_ascii=False, separators=(",", ":")
    )
    identity = str(getattr(tokenizer, "name_or_path", "") or "")
    revision = getattr(tokenizer, "init_kwargs", {}).get("revision")
    return {
        "trie_protocol_version": TRIE_PROTOCOL_VERSION,
        "vocab_path": Path(vocab_path).as_posix(),
        "vocab_set_sha256": hashlib.sha256(
            json.dumps(canonical, ensure_ascii=False, separators=(",", ":")).encode()
        ).hexdigest(),
        "vocab_size": len(canonical),
        "compiled_entry_count": len(compiled_entries),
        "compiled_entries_sha256": hashlib.sha256(compiled_json.encode()).hexdigest(),
        "tokenizer_identity": identity,
        "tokenizer_revision": revision,
        "eos_token_id": int(tokenizer.eos_token_id),
        "pad_token_id": int(tokenizer.pad_token_id),
    }


def validate_mass_config(config: dict[str, Any]) -> dict[str, Any]:
    """Validate and return the optional auxiliary mass section."""
    mass = config.get("auxiliary_loss", {}).get("mass", {})
    if not mass or not mass.get("enabled", False):
        return mass
    coefficient = mass.get("lambda")
    warmup = mass.get("warmup_steps")
    if (
        isinstance(coefficient, bool)
        or not isinstance(coefficient, (int, float))
        or not math.isfinite(coefficient)
        or coefficient <= 0
    ):
        raise ValueError("enabled auxiliary mass lambda must be finite and > 0")
    if isinstance(warmup, bool) or not isinstance(warmup, int) or warmup < 0:
        raise ValueError("auxiliary mass warmup_steps must be a non-negative integer")
    training = config.get("training", {})
    if training.get("packing", False) or training.get("padding_free", False):
        raise ValueError("auxiliary mass loss does not support packing or padding_free")
    return mass


def require_single_process_for_mass(config: dict[str, Any]) -> None:
    """Fail before model construction when enabled mass SFT is distributed."""
    mass = config.get("auxiliary_loss", {}).get("mass", {})
    if not mass.get("enabled", False):
        return
    try:
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
    except ValueError as exc:
        raise ValueError("WORLD_SIZE must be an integer") from exc
    if world_size != 1:
        raise ValueError("auxiliary mass SFT currently requires WORLD_SIZE=1")


def validate_structured_config(config: dict[str, Any]) -> dict[str, Any]:
    """Validate and return the optional structured auxiliary section."""
    structured = config.get("auxiliary_loss", {}).get("structured", {})
    if not structured or not structured.get("enabled", False):
        return structured
    coefficient = structured.get("lambda")
    warmup = structured.get("warmup_steps")
    if (
        isinstance(coefficient, bool)
        or not isinstance(coefficient, (int, float))
        or not math.isfinite(coefficient)
        or coefficient <= 0
    ):
        raise ValueError("enabled auxiliary structured lambda must be finite and > 0")
    if isinstance(warmup, bool) or not isinstance(warmup, int) or warmup < 0:
        raise ValueError(
            "auxiliary structured warmup_steps must be a non-negative integer"
        )
    for key in ("top_k", "max_gloss_length"):
        value = structured.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"auxiliary structured {key} must be a positive integer")
    for key in ("alpha", "transition_scale"):
        value = structured.get(key)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value <= 0
        ):
            raise ValueError(f"auxiliary structured {key} must be finite and > 0")
    training = config.get("training", {})
    if training.get("packing", False) or training.get("padding_free", False):
        raise ValueError(
            "auxiliary structured loss does not support packing or padding_free"
        )
    return structured


def require_single_process_for_auxiliary(config: dict[str, Any]) -> None:
    """Fail before model construction for any distributed auxiliary SFT."""
    auxiliary = config.get("auxiliary_loss", {})
    if not any(
        auxiliary.get(name, {}).get("enabled", False) for name in ("mass", "structured")
    ):
        return
    try:
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
    except ValueError as exc:
        raise ValueError("WORLD_SIZE must be an integer") from exc
    if world_size != 1:
        raise ValueError("auxiliary SFT currently requires WORLD_SIZE=1")


class CompletionMetadataCollator(DataCollatorForLanguageModeling):
    """TRL collator that re-emits consumed completion metadata.

    TRL consumes ``completion_mask`` while constructing completion-only labels.
    This wrapper snapshots the tokenized examples, delegates all ordinary
    padding and label construction to TRL, then right-pads the masks to the
    resulting sequence length. A row is eligible only when its retained
    completion is one contiguous non-empty span containing semantic EOS.
    """

    def __init__(self, *, eos_token_id: int, **kwargs: Any) -> None:
        if kwargs.get("padding_free", False):
            raise ValueError("auxiliary mass loss does not support padding_free")
        super().__init__(**kwargs)
        self.eos_token_id = int(eos_token_id)
        self.diagnostics: dict[str, int] = defaultdict(int)

    def __call__(
        self, features: list[dict[str, Any]], return_tensors: str | None = None
    ) -> dict[str, Tensor]:
        if not features:
            raise ValueError("auxiliary collator requires at least one feature")
        token_ids: list[list[int]] = []
        masks: list[list[int | bool]] = []
        for index, feature in enumerate(features):
            if "input_ids" not in feature or not feature["input_ids"]:
                raise ValueError(f"feature {index} requires nonempty input_ids")
            if "completion_mask" not in feature or not feature["completion_mask"]:
                raise ValueError(f"feature {index} requires nonempty completion_mask")
            ids = list(feature["input_ids"])
            mask = list(feature["completion_mask"])
            if len(ids) != len(mask):
                raise ValueError(
                    f"feature {index} input_ids/completion_mask length mismatch"
                )
            if any(
                not isinstance(value, (bool, int))
                or (isinstance(value, int) and value not in (0, 1))
                for value in mask
            ):
                raise ValueError(
                    f"feature {index} completion_mask must contain bool or 0/1"
                )
            if "labels" in feature:
                explicit = list(feature["labels"])
                if len(explicit) != len(ids):
                    raise ValueError(f"feature {index} labels length mismatch")
                expected = [
                    token if bool(selected) else -100
                    for token, selected in zip(ids, mask, strict=True)
                ]
                if explicit != expected:
                    raise ValueError(
                        f"feature {index} explicit labels conflict with completion_mask"
                    )
            token_ids.append(ids)
            masks.append(mask)

        batch = super().__call__(deepcopy(features), return_tensors=return_tensors)
        if batch["labels"].shape != batch["input_ids"].shape:
            raise RuntimeError(
                "TRL collator returned mismatched labels/input_ids shapes"
            )
        sequence_length = int(batch["labels"].shape[1])
        padded_masks: list[list[bool]] = []
        starts: list[int] = []
        eligible: list[bool] = []
        structured_states: list[list[int]] = []
        structured_lengths: list[int] = []
        structured_eligible: list[bool] = []
        sample_ids: list[str] = []

        for feature, ids, raw_mask in zip(features, token_ids, masks, strict=True):
            retained_length = min(len(ids), sequence_length)
            mask = [bool(value) for value in raw_mask[:retained_length]]
            mask.extend([False] * (sequence_length - len(mask)))
            positions = [index for index, selected in enumerate(mask) if selected]
            if not positions:
                raise ValueError(
                    "completion_mask must select at least one target token"
                )
            contiguous = (
                bool(positions)
                and positions[0] >= 1
                and positions == list(range(positions[0], positions[-1] + 1))
            )
            if not contiguous:
                raise ValueError(
                    "completion_mask must select one contiguous span after index 0"
                )
            supervised = batch["labels"][len(padded_masks)] != -100
            if supervised.tolist() != mask:
                raise RuntimeError(
                    "TRL labels do not exactly match retained completion_mask"
                )
            if "attention_mask" in batch:
                attention = batch["attention_mask"][len(padded_masks)].bool()
                if torch.any(torch.tensor(mask) & ~attention):
                    raise RuntimeError("completion_mask selects padding")
                if torch.any(batch["labels"][len(padded_masks)][~attention] != -100):
                    raise RuntimeError("padding labels must be -100")
            has_eos = any(
                ids[index] == self.eos_token_id
                for index in positions
                if index < retained_length
            )
            eos_is_final = has_eos and ids[positions[-1]] == self.eos_token_id
            if has_eos and not eos_is_final:
                raise ValueError("semantic EOS must terminate the completion span")
            row_eligible = has_eos
            if positions and not has_eos:
                self.diagnostics["missing_eos_or_truncated"] += 1
            padded_masks.append(mask)
            # Ineligible rows are skipped before use; retain a shape-valid start.
            starts.append(positions[0])
            eligible.append(row_eligible)
            states = list(feature.get("structured_gold_states", []))
            length = int(feature.get("structured_length", 0))
            precomputed_eligible = bool(feature.get("structured_eligible", False))
            if length < 0 or length > len(states):
                raise ValueError(
                    "structured_length is inconsistent with structured_gold_states"
                )
            structured_states.append(states)
            structured_lengths.append(length)
            structured_eligible.append(precomputed_eligible and row_eligible)
            sample_ids.append(str(feature.get("sample_id", "")))

        batch["completion_start"] = torch.tensor(starts, dtype=torch.long)
        batch["completion_mask"] = torch.tensor(padded_masks, dtype=torch.bool)
        batch["truncation_eligible"] = torch.tensor(eligible, dtype=torch.bool)
        if any(structured_states):
            width = max(len(states) for states in structured_states)
            batch["structured_gold_states"] = torch.tensor(
                [states + [0] * (width - len(states)) for states in structured_states],
                dtype=torch.long,
            )
            batch["structured_length"] = torch.tensor(
                structured_lengths, dtype=torch.long
            )
            batch["structured_eligible"] = torch.tensor(
                structured_eligible, dtype=torch.bool
            )
            batch["sample_id"] = sample_ids
        return batch


class AuxiliarySFTTrainer(SFTTrainer):
    """SFTTrainer adding mass, structured, or both losses during training only."""

    def __init__(
        self,
        *args: Any,
        mass_state_machine: AllowedTokenStateMachine[Any] | None = None,
        mass_lambda: float = 0.0,
        mass_warmup_steps: int = 0,
        structured_loss: StructuredGraphLoss | None = None,
        structured_lambda: float = 0.0,
        structured_warmup_steps: int = 0,
        graph_manifest: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        if mass_lambda < 0:
            raise ValueError("auxiliary mass lambda must be non-negative")
        if mass_warmup_steps < 0:
            raise ValueError("auxiliary mass warmup_steps must be non-negative")
        self.mass_state_machine = mass_state_machine
        self.mass_lambda = float(mass_lambda)
        self.mass_warmup_steps = int(mass_warmup_steps)
        if structured_lambda < 0 or structured_warmup_steps < 0:
            raise ValueError("structured lambda and warmup_steps must be non-negative")
        self.structured_loss = structured_loss
        self.structured_lambda = float(structured_lambda)
        self.structured_warmup_steps = int(structured_warmup_steps)
        self.graph_manifest = graph_manifest
        if self.structured_lambda and (structured_loss is None or not graph_manifest):
            raise ValueError(
                "structured loss and graph manifest are required when enabled"
            )
        super().__init__(*args, **kwargs)

    def mass_weight(self) -> float:
        """Current resume-safe linear-warmup coefficient."""
        if self.mass_lambda == 0.0:
            return 0.0
        if self.mass_warmup_steps == 0:
            return self.mass_lambda
        progress = min(1.0, self.state.global_step / self.mass_warmup_steps)
        return self.mass_lambda * progress

    def _record_mass_metrics(
        self, lm_loss: Tensor, mass_loss: Tensor, weight: float, diagnostics: Any
    ) -> None:
        metrics = self._metrics["train"]
        metrics["mass_lm_loss"].append(float(lm_loss.detach().float().item()))
        metrics["mass_objective_loss"].append(float(mass_loss.detach().float().item()))
        metrics["mass_weight"].append(weight)
        metrics["mass_allowed_mean"].append(diagnostics.mean_allowed_mass)
        metrics["mass_log_allowed_mean"].append(diagnostics.mean_log_allowed_mass)
        metrics["mass_loss_sum"].append(diagnostics.loss_sum)
        metrics["mass_allowed_sum"].append(diagnostics.allowed_mass_sum)
        metrics["mass_log_allowed_sum"].append(diagnostics.log_allowed_mass_sum)
        metrics["mass_scored_tokens"].append(diagnostics.scored_tokens)
        metrics["mass_scored_samples"].append(diagnostics.scored_samples)
        metrics["mass_skipped_truncated"].append(diagnostics.skipped_truncated)
        metrics["mass_invalid_prefix"].append(diagnostics.invalid_prefix)

    def compute_loss(
        self,
        model: torch.nn.Module,
        inputs: dict[str, torch.Tensor | Any],
        return_outputs: bool = False,
        num_items_in_batch: torch.Tensor | None = None,
    ):
        completion_start = inputs.pop("completion_start")
        completion_mask = inputs.pop("completion_mask")
        truncation_eligible = inputs.pop("truncation_eligible")
        structured_states = inputs.pop("structured_gold_states", None)
        structured_lengths = inputs.pop("structured_length", None)
        structured_eligible = inputs.pop("structured_eligible", None)
        inputs.pop("sample_id", None)
        labels = inputs["labels"]
        lm_loss, outputs = super().compute_loss(
            model,
            inputs,
            return_outputs=True,
            num_items_in_batch=num_items_in_batch,
        )

        captured_hidden = None
        if isinstance(model, AuxiliarySFTModel):
            captured_hidden = model.consume_output_hidden()

        # Keep eval_loss, early stopping, and best-model selection comparable.
        if not model.training:
            return (lm_loss, outputs) if return_outputs else lm_loss
        loss = lm_loss
        mass_lambda = float(getattr(self, "mass_lambda", 0.0))
        structured_lambda = float(getattr(self, "structured_lambda", 0.0))
        if mass_lambda:
            if self.mass_state_machine is None:
                raise RuntimeError("mass state machine is missing")
            mass_loss, diagnostics = allowed_mass_loss(
                cast(Any, outputs).logits,
                labels,
                completion_start,
                self.mass_state_machine,
                completion_mask=completion_mask,
                truncation_eligible=truncation_eligible,
            )
            weight = self.mass_weight()
            self._record_mass_metrics(lm_loss, mass_loss, weight, diagnostics)
            loss = loss + weight * mass_loss
        if structured_lambda:
            if not isinstance(model, AuxiliarySFTModel) or captured_hidden is None:
                raise RuntimeError("structured SFT requires AuxiliarySFTModel")
            if (
                structured_states is None
                or structured_lengths is None
                or structured_eligible is None
            ):
                raise RuntimeError("structured batch metadata is missing")
            valid = structured_eligible.bool()
            valid_count = int(valid.sum().item())
            skipped = int(valid.numel() - valid_count)
            if valid_count:
                boundary = gather_assistant_boundary(
                    captured_hidden[valid], completion_start[valid] - 1
                )
                lengths = structured_lengths[valid]
                steps = int(lengths.max().item())
                head = model.structured_head
                if head is None or self.structured_loss is None:
                    raise RuntimeError("structured head/loss is missing")
                emissions = head(boundary, length=steps)
                loss_device = next(self.structured_loss.buffers()).device
                if loss_device != emissions.device:
                    self.structured_loss.to(emissions.device)
                per_row = self.structured_loss(
                    emissions, structured_states[valid, :steps], lengths
                )
                aux_loss = per_row.mean()
                aux_sum = float(per_row.detach().float().sum().item())
            else:
                aux_loss = captured_hidden.sum() * 0.0
                aux_sum = 0.0
            weight = structured_weight(
                self.state.global_step,
                structured_lambda,
                int(getattr(self, "structured_warmup_steps", 0)),
            )
            metrics = self._metrics["train"]
            metrics["structured_lm_loss"].append(float(lm_loss.detach().float().item()))
            metrics["structured_objective_loss"].append(
                float(aux_loss.detach().float().item())
            )
            metrics["structured_weight"].append(weight)
            metrics["structured_loss_sum"].append(aux_sum)
            metrics["structured_valid_samples"].append(valid_count)
            metrics["structured_skipped_samples"].append(skipped)
            loss = loss + weight * aux_loss
        self._metrics["train"]["auxiliary_combined_loss"].append(
            float(loss.detach().float().item())
        )
        return (loss, outputs) if return_outputs else loss

    def _save(self, output_dir: str | None = None, state_dict=None) -> None:
        super()._save(output_dir, state_dict)
        if float(getattr(self, "structured_lambda", 0.0)):
            if not isinstance(self.model, AuxiliarySFTModel):
                raise RuntimeError("structured checkpoint requires AuxiliarySFTModel")
            if self.graph_manifest is None:
                raise RuntimeError("structured graph manifest is missing")
            destination = output_dir or self.args.output_dir
            if destination is None:
                raise RuntimeError("structured checkpoint output directory is missing")
            self.model.save_structured_artifacts(destination, self.graph_manifest)

    def _load_from_checkpoint(self, resume_from_checkpoint: str, model=None) -> None:
        super()._load_from_checkpoint(resume_from_checkpoint, model=model)
        target = model or self.model
        if float(getattr(self, "structured_lambda", 0.0)):
            if not isinstance(target, AuxiliarySFTModel):
                raise RuntimeError("structured checkpoint requires AuxiliarySFTModel")
            if self.graph_manifest is None:
                raise RuntimeError("structured graph manifest is missing")
            target.load_auxiliary_components(
                resume_from_checkpoint, self.graph_manifest
            )

    def _load_best_model(self) -> None:
        super()._load_best_model()
        if float(getattr(self, "structured_lambda", 0.0)):
            checkpoint = self.state.best_model_checkpoint
            if not checkpoint:
                raise RuntimeError("best structured checkpoint is missing")
            if not isinstance(self.model, AuxiliarySFTModel):
                raise RuntimeError("structured checkpoint requires AuxiliarySFTModel")
            if self.graph_manifest is None:
                raise RuntimeError("structured graph manifest is missing")
            self.model.load_auxiliary_components(checkpoint, self.graph_manifest)


# Backward-compatible public name for the accepted mass-only API.
AuxiliaryMassSFTTrainer = AuxiliarySFTTrainer
