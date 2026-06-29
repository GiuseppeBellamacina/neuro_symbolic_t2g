"""
Bootstrap entry point for T2G training scripts.

Usage:
    python -m src.training --config experiments/configs/t2g/grpo_qwen05.yaml [--resume] [--prepare-data]

Loads the config YAML and routes to the correct trainer (GRPO or SFT).
"""

from __future__ import annotations

import argparse
import sys as _sys

import yaml


def _peek_config(config_path: str) -> dict:
    """Lightweight config read without importing torch."""
    try:
        with open(config_path) as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}


# Parse --config early to decide bootstrap
_parser = argparse.ArgumentParser(add_help=False)
_parser.add_argument("--config", type=str, default=None)
_parser.add_argument("--prepare-data", action="store_true", default=False)
_early_args, _remaining = _parser.parse_known_args()

_cfg = _peek_config(_early_args.config) if _early_args.config else {}

# Add project root to path for imports
from pathlib import Path as _Path

_project_root = _Path(__file__).resolve().parent.parent.parent
if str(_project_root) not in _sys.path:
    _sys.path.insert(0, str(_project_root))

# ── Route to correct trainer based on config ─────────────────────────
print(f"[bootstrap] config={_early_args.config}")

_trainer = _cfg.get("training", {}).get("trainer", "grpo")
if _trainer == "sft":
    print("[bootstrap] trainer=sft → importing sft_train.main")
    from src.training.sft_train import main  # noqa: E402
else:
    print("[bootstrap] trainer=grpo → importing grpo_t2g_train.main")
    from src.training.grpo_t2g_train import main  # noqa: E402

main()
