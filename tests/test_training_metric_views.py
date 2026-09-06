from __future__ import annotations

from src.utils import live_training_table, show_training_log, visualization


def test_current_trl_metric_keys_are_used_by_tables_and_plots():
    current = {
        "rewards/edit_validity_reward/mean",
        "rewards/edit_validity_reward/std",
        "completions/mean_length",
        "completions/clipped_ratio",
        "completions/mean_terminated_length",
    }
    stale = {"rewards/edit_validity_reward", "completion_length"}

    assert current <= set(live_training_table._DEFAULT_COLS)
    assert current <= set(show_training_log._DEFAULT_COLS)
    assert current <= {key for key, _ in visualization._PLOT_METRICS}
    assert stale.isdisjoint(live_training_table._DEFAULT_COLS)
    assert stale.isdisjoint(show_training_log._DEFAULT_COLS)
    assert stale.isdisjoint(key for key, _ in visualization._PLOT_METRICS)


def test_dynamic_reward_metric_discovery_is_retained():
    keys = {"loss", "rewards/custom_reward/mean", "rewards/custom_reward/std"}
    assert live_training_table._reward_columns(keys) == [
        "rewards/custom_reward/mean",
        "rewards/custom_reward/std",
    ]
