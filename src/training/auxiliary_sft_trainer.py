"""Opt-in auxiliary objectives for SFT (allowed-mass and structured gloss loss).

Design contract, in order of importance
---------------------------------------
1. **Zero weight is bit-identical to stock SFT.** With every auxiliary weight at
   0 the trainer must not perturb the loss, the optimizer, the dataset, or the
   collator. This is what keeps historical SFT results reproducible, and it is
   asserted by tests rather than assumed.
2. **Nothing is enabled implicitly.** The objectives activate only when a config
   sets a positive weight under ``auxiliary_objective``. A malformed section is
   rejected *before* the model is built, so a typo fails in seconds rather than
   after hours of GPU time.
3. **No leakage.** The structured graph is built by the caller from post-split
   train rows only; this module never touches the dataset.

Why the previous attempt was rejected
-------------------------------------
The ``edit-rewards`` branch shipped an equivalent trainer that also changed the
*default* SFT path (fingerprint bumped 1->4, the bigram artifact stopped being
produced, and historical configs without an ``experiment:`` block became
unrunnable). That broke comparability with every stored result, so it was not
recovered. This module keeps the mathematics and drops those side effects: the
stock ``SFTTrainer`` is still constructed when no auxiliary objective is on.

Scientific expectations, measured (see docs/NEW_OBJECTIVES_SPEC.md)
-------------------------------------------------------------------
- Allowed-mass is the candidate-set log-marginal, i.e. partial-label learning /
  maximum marginal likelihood (Cour et al. JMLR 2011; Guu et al.
  arXiv:1704.07926). It is *flat over the allowed set*, so it cannot reorder
  candidates inside it: expect calibration effects, not exact-match gains.
- The structured prior is worth **+0.044 bits (5.2% of the residual)** on
  length-aligned pairs of this corpus, and an autoregressive decoder already
  conditions on ``gloss_{t-1}``. Expected value here is low; the module exists
  so the claim can be tested rather than argued.
"""

from __future__ import annotations

import math
import os
from copy import deepcopy
from typing import Any

import torch
from torch import Tensor
from transformers import DataCollatorForLanguageModeling
from trl import SFTTrainer  # type: ignore[import]

from src.training.allowed_mass_loss import allowed_mass_loss


# ---------------------------------------------------------------------------
# Config validation (runs before the model is constructed)
# ---------------------------------------------------------------------------
def _validate_weight(section: dict[str, Any], name: str) -> float:
    weight = section.get("weight", 0.0)
    if isinstance(weight, bool) or not isinstance(weight, (int, float)):
        raise ValueError(f"auxiliary_objective.{name}.weight must be a number")
    weight = float(weight)
    if not math.isfinite(weight) or weight < 0.0:
        raise ValueError(
            f"auxiliary_objective.{name}.weight must be finite and >= 0, got {weight}"
        )
    return weight


def _validate_warmup(section: dict[str, Any], name: str) -> int:
    warmup = section.get("warmup_steps", 0)
    if isinstance(warmup, bool) or not isinstance(warmup, int) or warmup < 0:
        raise ValueError(
            f"auxiliary_objective.{name}.warmup_steps must be a non-negative int"
        )
    return int(warmup)


def resolve_auxiliary_config(config: dict[str, Any]) -> dict[str, Any]:
    """Return the normalized auxiliary spec, or an empty dict when disabled.

    A section with weight 0 is treated as *disabled*, so a config can keep the
    block for documentation without changing training behaviour.

    Raises:
        ValueError: On a malformed section, or on a training setup the
            objectives cannot support (packing / padding-free destroy the
            per-position alignment both losses rely on).
    """
    auxiliary = config.get("auxiliary_objective") or {}
    if not isinstance(auxiliary, dict):
        raise ValueError("auxiliary_objective must be a mapping")

    resolved: dict[str, Any] = {}
    for name in ("allowed_mass", "structured"):
        section = auxiliary.get(name) or {}
        if not isinstance(section, dict):
            raise ValueError(f"auxiliary_objective.{name} must be a mapping")
        if not section:
            continue
        weight = _validate_weight(section, name)
        if weight == 0.0:
            continue
        resolved[name] = {
            "weight": weight,
            "warmup_steps": _validate_warmup(section, name),
        }
        if name == "structured":
            for key, minimum in (("top_k", 1), ("alpha", 0)):
                value = section.get(key)
                if value is None:
                    continue
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    raise ValueError(
                        f"auxiliary_objective.structured.{key} must be a number"
                    )
                if float(value) < minimum:
                    raise ValueError(
                        f"auxiliary_objective.structured.{key} must be >= {minimum}"
                    )
                resolved[name][key] = value
            resolved[name]["shuffled_control"] = bool(
                section.get("shuffled_control", False)
            )

    if resolved:
        training = config.get("training", {})
        for unsupported in ("packing", "padding_free"):
            if training.get(unsupported, False):
                raise ValueError(
                    f"auxiliary objectives do not support training.{unsupported}: "
                    f"they need per-position alignment between labels and logits"
                )
    return resolved


def build_structured_graph(
    train_rows: Any,
    spec: dict[str, Any],
) -> Any:
    """Build the structured transition graph from POST-SPLIT train rows only.

    The anti-leakage contract lives here and nowhere else: the caller must pass
    the finalized train split, i.e. the rows that remain *after* the evaluation
    holdout has been carved out. This function never reads the dataset, so it
    cannot reach evaluation data on its own — but it also cannot verify what it
    is handed, which is why the call site in ``sft_train.py`` passes the split
    output directly rather than the raw dataset.

    Args:
        train_rows: Sequence of mappings with a ``gloss`` key. Must be
            post-split train rows.
        spec: Resolved ``structured`` section (``top_k``, ``alpha``,
            ``shuffled_control``).

    Returns:
        A ``StructuredTransitionGraph``. When ``shuffled_control`` is set, the
        transitions are permuted: that is the GATE 2 negative control, and a
        run using it must NOT beat the real graph if the structural prior is
        genuinely contributing.
    """
    from src.datasets.structured_transitions import (
        build_structured_transition_graph,
        shuffled_transition_control,
    )

    rows = [{"gloss": str(row["gold_gloss"])} for row in train_rows]
    graph = build_structured_transition_graph(
        rows,
        top_k=int(spec.get("top_k", 512)),
        alpha=float(spec.get("alpha", 0.1)),
    )
    if spec.get("shuffled_control", False):
        graph = shuffled_transition_control(graph)
    return graph


def require_single_process(auxiliary: dict[str, Any]) -> None:
    """Fail fast when auxiliary objectives are combined with distributed training.

    The losses accumulate per-row diagnostics and assume one process owns the
    whole batch; they were never validated under gathering, so refuse instead of
    silently reporting wrong numbers.
    """
    if not auxiliary:
        return
    try:
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
    except ValueError as exc:
        raise ValueError("WORLD_SIZE must be an integer") from exc
    if world_size != 1:
        raise ValueError(f"auxiliary objectives require WORLD_SIZE=1, got {world_size}")


# ---------------------------------------------------------------------------
# Collator: preserves the completion span the losses need
# ---------------------------------------------------------------------------
class CompletionSpanCollator(DataCollatorForLanguageModeling):
    """Adds the completion span metadata that auxiliary losses require.

    TRL's collator produces ``labels`` with ``-100`` on the prompt, which is
    enough for the LM loss but loses *where* the completion starts. Both
    auxiliary losses score only completion positions, so the span is recovered
    here and validated against the labels TRL actually produced — if the two
    disagree we raise instead of scoring the wrong positions.
    """

    def __init__(self, *, eos_token_id: int, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.eos_token_id = int(eos_token_id)

    def __call__(
        self, features: list[dict[str, Any]], return_tensors: str | None = None
    ) -> dict[str, Tensor]:
        if not features:
            raise ValueError("collator requires at least one feature")

        batch = super().__call__(deepcopy(features), return_tensors=return_tensors)
        labels = batch["labels"]
        if labels.shape != batch["input_ids"].shape:
            raise RuntimeError("collator returned mismatched labels/input_ids shapes")

        # The completion is exactly the supervised span: labels != -100.
        supervised = labels != -100
        if not bool(supervised.any()):
            raise ValueError("no supervised token in batch: nothing to score")

        starts: list[int] = []
        eligible: list[bool] = []
        for row in range(labels.shape[0]):
            positions = torch.nonzero(supervised[row], as_tuple=False).flatten()
            if positions.numel() == 0:
                # Keep a shape-valid start; the row is marked ineligible and
                # skipped before use rather than silently scored.
                starts.append(0)
                eligible.append(False)
                continue
            first = int(positions[0].item())
            last = int(positions[-1].item())
            contiguous = positions.numel() == (last - first + 1)
            ends_with_eos = (
                int(batch["input_ids"][row, last].item()) == self.eos_token_id
            )
            starts.append(first)
            # Only score rows whose completion is a contiguous span terminated by
            # EOS: a truncated completion has no valid final state.
            eligible.append(bool(contiguous and ends_with_eos and first >= 1))

        batch["completion_start"] = torch.tensor(starts, dtype=torch.long)
        batch["completion_eligible"] = torch.tensor(eligible, dtype=torch.bool)
        return batch


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------
class AuxiliarySFTTrainer(SFTTrainer):
    """``SFTTrainer`` plus opt-in auxiliary objectives.

    Args:
        allowed_mask_fn: Callable mapping a list of token-id prefixes and a
            vocabulary size to a ``[N, V]`` bool mask of permitted tokens.
            :meth:`src.grammar.grammar_logits_processor.GlossVocabularyLogitsProcessor.allowed_mask_for_prefixes`
            is the intended producer, so the loss and the decoder share one Trie
            walk and cannot drift apart.
        mass_weight: Weight of the allowed-mass term. ``0.0`` disables it.
        mass_warmup_steps: Linear warmup for the mass weight.

    With ``mass_weight == 0`` this class delegates entirely to ``SFTTrainer``:
    ``compute_loss`` returns ``super().compute_loss(...)`` unchanged.
    """

    def __init__(
        self,
        *args: Any,
        allowed_mask_fn: Any = None,
        mass_weight: float = 0.0,
        mass_warmup_steps: int = 0,
        structured_head: Any = None,
        structured_loss: Any = None,
        structured_graph: Any = None,
        structured_weight: float = 0.0,
        structured_warmup_steps: int = 0,
        **kwargs: Any,
    ) -> None:
        if not math.isfinite(mass_weight) or mass_weight < 0.0:
            raise ValueError(f"mass_weight must be finite and >= 0, got {mass_weight}")
        if mass_warmup_steps < 0:
            raise ValueError("mass_warmup_steps must be >= 0")
        if mass_weight > 0.0 and allowed_mask_fn is None:
            raise ValueError(
                "mass_weight > 0 requires allowed_mask_fn: without it the loss has "
                "no way to obtain the permitted-token mask"
            )
        if not math.isfinite(structured_weight) or structured_weight < 0.0:
            raise ValueError(
                f"structured_weight must be finite and >= 0, got {structured_weight}"
            )
        if structured_warmup_steps < 0:
            raise ValueError("structured_warmup_steps must be >= 0")
        if structured_weight > 0.0 and (
            structured_head is None or structured_loss is None
        ):
            raise ValueError(
                "structured_weight > 0 requires both structured_head and "
                "structured_loss: the emission head and the graph partition are "
                "two halves of the same objective"
            )
        self.allowed_mask_fn = allowed_mask_fn
        self.mass_weight = float(mass_weight)
        self.mass_warmup_steps = int(mass_warmup_steps)
        self.structured_head = structured_head
        self.structured_loss = structured_loss
        # StructuredGraphLoss conserva solo buffer tensoriali, non il grafo:
        # per mappare le glosse gold in stati serve il grafo esplicito.
        self.structured_graph = structured_graph
        self.structured_weight = float(structured_weight)
        self.structured_warmup_steps = int(structured_warmup_steps)
        self.auxiliary_diagnostics: dict[str, float] = {}
        super().__init__(*args, **kwargs)
        # The head is a real parameter tree: it must reach the optimizer, and it
        # must live on the same device as the backbone.
        if self.structured_head is not None and self.model is not None:
            device = getattr(self.model, "device", None)
            if device is not None:
                self.structured_head.to(device)

    def _auxiliary_enabled(self) -> bool:
        mass_on = (
            getattr(self, "mass_weight", 0.0) > 0.0
            and getattr(self, "allowed_mask_fn", None) is not None
        )
        structured_on = getattr(self, "structured_weight", 0.0) > 0.0 and (
            getattr(self, "structured_head", None) is not None
            and getattr(self, "structured_loss", None) is not None
        )
        return mass_on or structured_on

    def compute_loss(
        self,
        model: torch.nn.Module,
        inputs: dict[str, Any],
        return_outputs: bool = False,
        num_items_in_batch: Tensor | None = None,
    ) -> Any:
        # Metadata is popped unconditionally: leaving it in `inputs` would reach
        # the model as an unexpected keyword argument.
        completion_start = inputs.pop("completion_start", None)
        completion_eligible = inputs.pop("completion_eligible", None)

        if not self._auxiliary_enabled():
            # Exactly stock SFT. No extra forward, no perturbation.
            return super().compute_loss(
                model,
                inputs,
                return_outputs=return_outputs,
                num_items_in_batch=num_items_in_batch,
            )

        # Il ramo structured legge l'ultimo hidden state dallo STESSO forward
        # della LM loss. Va richiesto esplicitamente: senza questo flag
        # `outputs.hidden_states` è None e il termine verrebbe saltato in
        # silenzio, cioè si addestrerebbe un normale SFT credendo di misurare
        # l'obiettivo strutturato.
        structured_on = (
            getattr(self, "structured_weight", 0.0) > 0.0
            and getattr(self, "structured_head", None) is not None
            and getattr(self, "structured_loss", None) is not None
        )
        gold_glosses = inputs.pop("gold_gloss", None)
        if structured_on:
            inputs["output_hidden_states"] = True

        lm_loss, outputs = super().compute_loss(
            model,
            inputs,
            return_outputs=True,
            num_items_in_batch=num_items_in_batch,
        )
        inputs.pop("output_hidden_states", None)
        if gold_glosses is not None:
            inputs["gold_gloss"] = gold_glosses

        total = lm_loss
        diagnostics_out: dict[str, float] = {
            "lm_loss": float(lm_loss.detach().float().item())
        }

        if (
            getattr(self, "mass_weight", 0.0) > 0.0
            and getattr(self, "allowed_mask_fn", None) is not None
            and completion_start is not None
            and completion_eligible is not None
        ):
            mass_term, mass_diag = self._mass_term(
                inputs, outputs, completion_start, completion_eligible
            )
            total = total + mass_term
            diagnostics_out.update(
                {
                    "mass_weight": mass_diag.effective_weight,
                    "mass_neg_log_allowed": mass_diag.mean_neg_log_mass,
                    "mass_allowed": mass_diag.mean_allowed_mass,
                    "mass_scored_positions": float(mass_diag.scored_positions),
                    "mass_skipped_positions": float(mass_diag.skipped_positions),
                }
            )

        if (
            getattr(self, "structured_weight", 0.0) > 0.0
            and getattr(self, "structured_head", None) is not None
            and getattr(self, "structured_loss", None) is not None
        ):
            structured_term, structured_diag = self._structured_term(inputs, outputs)
            total = total + structured_term
            diagnostics_out.update(structured_diag)

        self.auxiliary_diagnostics = diagnostics_out
        return (total, outputs) if return_outputs else total

    def _structured_weight_now(self) -> float:
        """Linear warmup, resume-safe (reads the trainer's global step)."""
        if self.structured_warmup_steps <= 0:
            return self.structured_weight
        step = int(self.state.global_step) if self.state is not None else 0
        progress = min(1.0, step / self.structured_warmup_steps)
        return self.structured_weight * progress

    def _structured_targets(
        self, gold_glosses: list[str], device: Any
    ) -> tuple[Tensor, Tensor]:
        """Map gold gloss strings to graph state ids, padded to the batch max.

        Targets are derived here rather than in the dataset so the mapping and
        the graph can never disagree: both come from the same
        ``StructuredTransitionGraph`` instance. Sequences longer than the head's
        ``max_length`` are marked ineligible (length 0) instead of being
        truncated, because a truncated path has no valid EOS transition and its
        partition would be wrong.
        """
        graph = self.structured_graph
        if graph is None:
            raise RuntimeError(
                "structured_graph is required to map gold glosses to states; "
                "StructuredGraphLoss keeps only tensor buffers, so the graph "
                "must be passed to the trainer explicitly"
            )
        max_length = int(getattr(self.structured_head, "max_length", 64))
        mapped: list[list[int]] = []
        lengths: list[int] = []
        for gold in gold_glosses:
            states = graph.map_glosses(gold)
            if not states or len(states) > max_length:
                mapped.append([])
                lengths.append(0)
                continue
            mapped.append(states)
            lengths.append(len(states))
        width = max((len(row) for row in mapped), default=0)
        padded = [row + [0] * (width - len(row)) for row in mapped]
        return (
            (
                torch.tensor(padded, dtype=torch.long, device=device)
                if width
                else torch.zeros((len(mapped), 0), dtype=torch.long, device=device)
            ),
            torch.tensor(lengths, dtype=torch.long, device=device),
        )

    def _structured_term(
        self, inputs: dict[str, Any], outputs: Any
    ) -> tuple[Tensor, dict[str, float]]:
        """Structured (CRF) NLL over the reduced gloss state space.

        Targets come either from the batch (``structured_states`` /
        ``structured_length``, if a dataset mapper provided them) or are derived
        on the fly from ``gold_gloss``. Rows that cannot be mapped are skipped
        rather than scored against a fabricated target: a wrong structured
        target is worse than no structured term at all.
        """
        weight = self._structured_weight_now()
        zero = outputs.logits.sum() * 0.0
        skipped = {"structured_weight": weight, "structured_scored_rows": 0.0}
        if weight == 0.0:
            return zero, skipped

        states = inputs.get("structured_states")
        lengths = inputs.get("structured_length")
        if states is None or lengths is None:
            gold = inputs.get("gold_gloss")
            if not gold:
                return zero, skipped
            states, lengths = self._structured_targets(
                [str(g) for g in gold], outputs.logits.device
            )

        hidden = getattr(outputs, "hidden_states", None)
        if hidden is None:
            # Senza hidden states non si forza un SECONDO forward: costerebbe il
            # doppio e cambierebbe la LM loss. Si salta e lo si dichiara.
            return zero, skipped

        valid = lengths > 0
        if not bool(valid.any()) or states.numel() == 0:
            return zero, skipped

        boundary = hidden[-1][:, -1, :]
        steps = int(lengths[valid].max().item())
        emissions = self.structured_head(boundary[valid], length=steps)
        per_row = self.structured_loss(emissions, states[valid, :steps], lengths[valid])
        term = weight * per_row.mean()
        return term, {
            "structured_weight": weight,
            "structured_nll": float(per_row.mean().detach().float().item()),
            "structured_scored_rows": float(int(valid.sum().item())),
        }

    def _mass_term(
        self,
        inputs: dict[str, Any],
        outputs: Any,
        completion_start: Tensor,
        completion_eligible: Tensor,
    ) -> tuple[Tensor, Any]:
        """Score the allowed-mass penalty on eligible completion positions.

        Uses the logits from the SAME forward pass as the LM loss (no second
        forward), and causal alignment ``logits[t-1] -> labels[t]``.
        """
        logits = outputs.logits
        input_ids = inputs["input_ids"]
        labels = inputs["labels"]
        vocab_size = int(logits.shape[-1])

        prefixes: list[list[int]] = []
        rows: list[Tensor] = []
        for row in range(labels.shape[0]):
            if not bool(completion_eligible[row].item()):
                continue
            start = int(completion_start[row].item())
            positions = torch.nonzero(labels[row] != -100, as_tuple=False).flatten()
            if positions.numel() == 0:
                continue
            for position in positions.tolist():
                if position < 1:
                    continue
                # Prefix = generated tokens before this position, i.e. the
                # completion so far. The Trie state depends only on that.
                prefixes.append(input_ids[row, start:position].tolist())
                rows.append(logits[row, position - 1])

        if not rows:
            zero = logits.sum() * 0.0
            from src.training.allowed_mass_loss import AllowedMassDiagnostics

            return zero, AllowedMassDiagnostics(0.0, 0.0, 0, 0, 0.0)

        stacked = torch.stack(rows, dim=0)
        allowed = self.allowed_mask_fn(prefixes, vocab_size, stacked.device)
        return allowed_mass_loss(
            stacked,
            allowed,
            weight=self.mass_weight,
            warmup_steps=self.mass_warmup_steps,
            step=int(self.state.global_step) if self.state is not None else None,
        )

    def log(self, logs: dict[str, float], *args: Any, **kwargs: Any) -> None:
        """Surface auxiliary diagnostics next to the standard training metrics."""
        if self.auxiliary_diagnostics:
            logs = {
                **logs,
                **{f"aux/{k}": v for k, v in self.auxiliary_diagnostics.items()},
            }
        super().log(logs, *args, **kwargs)
