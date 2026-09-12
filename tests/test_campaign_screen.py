"""Config sanity checks for cells once exercised via the TUI's old, now
retired CampaignScreen (superseded by PresetsScreen — see
tests/test_remote_tui.py's ``test_presets_screen_*`` and
tests/test_remote_presets.py for its coverage).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def test_zero_shot_no_grammar_config_exists():
    """baseline/zero-shot-no-grammar.yaml: lower bound non vincolato (grammar OFF)."""
    from src.utils.config import resolve_config

    cfg = resolve_config(
        "experiments/configs/qwen25-05b/baseline/zero-shot-no-grammar.yaml"
    )
    assert cfg["grammar"]["enabled"] is False
    assert cfg["wandb"]["run_name"] == "qwen25-05b-baseline-zero-shot-no-grammar"
    # eval-only: nessun output_dir (eredita una sezione training parziale da base)
    assert "output_dir" not in cfg["training"]


def test_hotrollout_config_exists():
    """ablations/decoding/hot-rollout.yaml: controllo Finding 1 (T=1.3, riusa SFT).

    Differenza a fattore unico vs sft-grpo/few-shot: solo la rollout temperature.
    La sezione sft_pretrain NON viene toccata (fingerprint SFT invariata
    → riuso dell'adapter di sft/zero-shot).
    """
    from src.utils.config import resolve_config

    base = resolve_config("experiments/configs/qwen25-05b/sft-grpo/few-shot.yaml")
    cfg = resolve_config(
        "experiments/configs/qwen25-05b/ablations/decoding/hot-rollout.yaml"
    )
    assert cfg["grpo"]["temperature"] == 1.3
    assert cfg["grpo"]["num_generations"] == base["grpo"]["num_generations"]
    assert cfg["sft_pretrain"] == base["sft_pretrain"], "fingerprint SFT invariata"
    assert cfg["training"]["output_dir"] == (
        "experiments/checkpoints/qwen25-05b/ablations/decoding/hot-rollout"
    )
