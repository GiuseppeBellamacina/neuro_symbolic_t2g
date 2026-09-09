"""Tests for the config-driven eval surface (knob moved from CLI/env into the
`evaluation:` config section) and the partial-checkpoint completeness stamp.

Covers:
- _resolve_prompting: default (config-derivation), redundant override and
  mode-changing override (provenance + changed flag).
- _deduce_eval_modes: baseline/* cells stay eval-only WITHOUT declaring
  anything (historical cluster/eval.sh deduction); explicit config values win.
- _complement_prompting: the dual-eval complementary mode.
- _checkpoint_completeness_stamp: an eval on a checkpoint-<step> dir (not
  `final`) is stamped checkpoint_incomplete=true — never indistinguishable
  from a full-model eval.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.training.eval_t2g import (  # noqa: E402
    _checkpoint_completeness_stamp,
    _complement_prompting,
    _deduce_eval_modes,
    _resolve_prompting,
)

_FEWSHOT_CFG = {
    "retrieval": {"enabled": True},
    "training": {"output_dir": "experiments/checkpoints/x"},
}
_ZEROSHOT_CFG = {
    "retrieval": {"enabled": False},
    "training": {"output_dir": "experiments/checkpoints/x"},
}


# ---------------------------------------------------------------------------
# _resolve_prompting
# ---------------------------------------------------------------------------


def test_prompting_default_derives_from_retrieval():
    """evaluation.prompting assente/'config' → modalità da retrieval.enabled,
    provenienza 'config' (comportamento storico)."""
    mode, source, changed = _resolve_prompting({}, _FEWSHOT_CFG)
    assert (mode, source, changed) == ("few-shot", "config", False)


def test_prompting_redundant_override_is_not_changed():
    """Un override che coincide con la modalità implicita è ridondante: nessun
    suffisso file, nessuna invalidazione di cache (changed=False)."""
    declared = {"prompting": "few-shot"}
    mode, source, changed = _resolve_prompting(declared, _FEWSHOT_CFG)
    assert (mode, source, changed) == ("few-shot", "config", False)


def test_prompting_override_marks_changed():
    """Un override che CAMBIA modalità è marcato: suffisso file + fingerprint
    diverso (invalidazione cache baseline al cambio di modalità)."""
    declared = {"prompting": "zero-shot"}
    mode, source, changed = _resolve_prompting(declared, _FEWSHOT_CFG)
    assert (mode, source, changed) == ("zero-shot", "config-override", True)


# ---------------------------------------------------------------------------
# _deduce_eval_modes
# ---------------------------------------------------------------------------


def test_eval_only_cell_auto_deduction_no_declaration():
    """Una cella baseline/* (niente output_dir, niente checkpoint) resta
    eval-only SENZA dichiarare nulla — deduzione storica di cluster/eval.sh."""
    baseline_only, do_compare = _deduce_eval_modes(
        {}, has_checkpoint=False, has_output_dir=False
    )
    assert baseline_only is True
    assert do_compare is False


def test_training_cell_auto_deduces_compare():
    """Una cella di training (output_dir) deduce compare con la baseline."""
    baseline_only, do_compare = _deduce_eval_modes(
        {}, has_checkpoint=True, has_output_dir=True
    )
    assert baseline_only is False
    assert do_compare is True


def test_training_config_without_checkpoint_still_compare():
    """Config di training invocata senza checkpoint: compare (la guardia
    loud-fail è responsabilità del chiamante), non eval-baseline-only."""
    baseline_only, do_compare = _deduce_eval_modes(
        {}, has_checkpoint=False, has_output_dir=True
    )
    assert baseline_only is False
    assert do_compare is True


def test_explicit_config_values_win_over_deduction():
    """compare/eval_baseline_only dichiarati vincono sulla deduzione."""
    forced = {"compare": False, "eval_baseline_only": False}
    baseline_only, do_compare = _deduce_eval_modes(
        forced, has_checkpoint=True, has_output_dir=True
    )
    assert baseline_only is False
    assert do_compare is False

    forced_only = {"eval_baseline_only": True}
    baseline_only, do_compare = _deduce_eval_modes(
        forced_only, has_checkpoint=True, has_output_dir=True
    )
    assert baseline_only is True


def test_null_values_fall_back_to_deduction():
    """compare: null / eval_baseline_only: null (default di base.yaml) NON
    vincono: si deduce come se non fossero dichiarati."""
    baseline_only, do_compare = _deduce_eval_modes(
        {"compare": None, "eval_baseline_only": None},
        has_checkpoint=False,
        has_output_dir=False,
    )
    assert baseline_only is True
    assert do_compare is False


# ---------------------------------------------------------------------------
# _complement_prompting
# ---------------------------------------------------------------------------


def test_complement_prompting_swaps_modes():
    assert _complement_prompting("zero-shot") == "few-shot"
    assert _complement_prompting("few-shot") == "zero-shot"


# ---------------------------------------------------------------------------
# _checkpoint_completeness_stamp
# ---------------------------------------------------------------------------


def test_partial_checkpoint_is_stamped():
    """Un checkpoint-<step> (NON final) è un modello parziale: lo stamp lo
    dichiara con lo step raggiunto."""
    stamp = _checkpoint_completeness_stamp(
        "experiments/checkpoints/qwen25-05b/grpo/few-shot/run_1/checkpoint-3500"
    )
    assert stamp == {"checkpoint_incomplete": True, "checkpoint_step": 3500}


def test_final_checkpoint_is_not_stamped():
    """Il path final (scritto solo a training completato) non produce stamp."""
    assert _checkpoint_completeness_stamp("some/run_1/final") == {}


def test_missing_checkpoint_is_not_stamped():
    """Base model (nessun checkpoint): nessuno stamp."""
    assert _checkpoint_completeness_stamp(None) == {}
    assert _checkpoint_completeness_stamp("") == {}
