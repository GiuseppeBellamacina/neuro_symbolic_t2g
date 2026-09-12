"""Regression: ``plot_reward_breakdown`` must not crash on a reward stack
whose only active component isn't in the historical 7-component set.

``ablations/rewards/edit-validity.yaml`` zeroes every historical weight and
sets only ``weight_edit_validity``. Before this fix, ``edit_validity_reward``
was absent from ``_COMPONENT_ORDER``/``_COMPONENT_LABELS``/
``_COMPONENT_COLORS`` in ``src/utils/visualization.py``, so ``components``
ended up empty, the DataFrame built from an empty ``rows`` list had no
"component" column, and ``df["component"] = pd.Categorical(df["component"],
...)`` raised ``KeyError: 'component'`` — which is exactly what killed job
7374 (eval of the edit-validity cell) on the cluster.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.visualization import plot_reward_breakdown  # noqa: E402


def test_plot_reward_breakdown_handles_a_component_outside_the_historical_set(
    tmp_path,
):
    """The exact edit-validity scenario: one active, non-historical component."""
    out = tmp_path / "reward_breakdown.png"
    plot_reward_breakdown(
        [{"label": "edit-validity", "scores": {"edit_validity_reward": 0.62}}],
        reward_weights={"edit_validity_reward": 1.0},
        output_path=str(out),
    )
    assert out.exists()


def test_plot_reward_breakdown_prints_instead_of_raising_when_nothing_is_active(
    tmp_path, capsys
):
    """A reward stack that filters out to nothing must degrade gracefully,
    not raise KeyError on an empty DataFrame."""
    out = tmp_path / "reward_breakdown.png"
    plot_reward_breakdown(
        [{"label": "x", "scores": {"some_unregistered_component": 0.5}}],
        reward_weights={"some_unregistered_component": 1.0},
        output_path=str(out),
    )
    assert not out.exists()
    assert "No active reward components" in capsys.readouterr().out


def test_plot_reward_breakdown_still_handles_the_historical_seven_components(
    tmp_path,
):
    """Non-regression: the original 7-component stack keeps working."""
    out = tmp_path / "reward_breakdown.png"
    plot_reward_breakdown(
        [
            {
                "label": "sft-grpo",
                "scores": {
                    "translation_quality_reward": 0.5,
                    "bleu_reward": 0.4,
                    "gold_structure_reward": 0.3,
                    "gloss_order_reward": 0.2,
                    "gloss_format_reward": 0.9,
                    "gloss_repetition_reward": 0.9,
                },
            }
        ],
        reward_weights={
            "translation_quality_reward": 0.2,
            "bleu_reward": 0.2,
            "gold_structure_reward": 0.2,
            "gloss_order_reward": 0.1,
            "gloss_format_reward": 0.1,
            "gloss_repetition_reward": 0.1,
        },
        output_path=str(out),
    )
    assert out.exists()
