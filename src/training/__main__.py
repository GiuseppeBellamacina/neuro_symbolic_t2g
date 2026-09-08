"""
Bootstrap entry point for T2G training scripts.

Usage:
    python -m src.training --config experiments/configs/qwen25-05b/sft-grpo/few-shot.yaml [--resume] [--prepare-data]

Loads the config YAML and routes to the correct trainer (GRPO or SFT).
"""

import argparse
import sys as _sys

# ── Workaround for trl 0.24.0 bug ────────────────────────────────────
# trl/extras/vllm_client.py unconditionally imports vllm_ascend (Huawei
# Ascend NPU support). On NVIDIA GPUs this package does not exist and
# the import fails with ModuleNotFoundError, crashing the training.
# We inject a dummy module with a valid ModuleSpec into sys.modules
# before trl is imported to satisfy Python's importlib.util.find_spec.
if "vllm_ascend" not in _sys.modules:
    import importlib.machinery
    import types

    spec = importlib.machinery.ModuleSpec("vllm_ascend", None)
    dummy = types.ModuleType("vllm_ascend")
    dummy.__spec__ = spec
    _sys.modules["vllm_ascend"] = dummy

import yaml


def _peek_config(config_path: str) -> dict:
    """Lightweight config read without importing torch.

    Resolves ``extends`` chains via ``src.utils.config.resolve_config`` so the
    bootstrap sees the same merged values as the trainers (e.g.
    ``model.use_unsloth`` inherited from ``base.yaml``). ``src.utils.config``
    is import-safe here: it only pulls in yaml + os (no torch/transformers).

    On any resolution failure it falls back to a plain YAML read, then to
    ``{}`` (historical behavior — the trainer's own ``load_config`` will
    surface real errors later).
    """
    try:
        from src.utils.config import resolve_config

        return resolve_config(config_path) or {}
    except Exception:
        pass
    try:
        with open(config_path, encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}


# Parse --config early to decide bootstrap
_parser = argparse.ArgumentParser(add_help=False)
_parser.add_argument("--config", type=str, default=None)
_parser.add_argument("--prepare-data", action="store_true", default=False)
_early_args, _remaining = _parser.parse_known_args()

_cfg = _peek_config(_early_args.config) if _early_args.config else {}

# ── Guardia eval-only: PRIMA di caricare Unsloth ────────────────────────────
# Le celle `baseline/*` non addestrano: ereditano una sezione `training`
# parziale da base.yaml e non dichiarano output_dir/log_dir. Lanciarle con
# cluster/train.sh e' un errore d'uso.
#
# La guardia sta QUI e non nel trainer perche' l'import di Unsloth (sotto)
# costa minuti su un nodo GPU allocato: il job 7294 fallira con
# `KeyError: 'output_dir'` solo DOPO aver caricato Unsloth, il modello e il
# dataset. Fallire in un secondo, prima di occupare la GPU, e' il punto.
if _early_args.config and not _early_args.prepare_data:
    _training = _cfg.get("training", {})
    if _training.get("trainer", "grpo") != "sft":
        _missing = [k for k in ("output_dir", "log_dir") if k not in _training]
        if _missing:
            _has_steps = bool({"max_steps", "num_train_epochs"} & set(_training))
            _sys.stderr.write(
                f"\n[bootstrap] Config non addestrabile: {_early_args.config}\n"
                f"            Chiavi mancanti in `training`: "
                f"{', '.join(_missing)}.\n"
                + (
                    "            Questa e' una cella EVAL-ONLY: usa "
                    "cluster/eval.sh, non cluster/train.sh.\n"
                    f"              CONFIG={_early_args.config} "
                    "sbatch cluster/eval.sh\n"
                    if not _has_steps
                    else "            La cella dichiara step di training ma "
                    "non le directory di output: aggiungi training.output_dir "
                    "e training.log_dir.\n"
                )
            )
            raise SystemExit(2)

# Auto-disable Unsloth when using multiple GPUs
_num_gpus = _cfg.get("model", {}).get("num_gpus", 1)
if _num_gpus > 1:
    _cfg.setdefault("model", {})["use_unsloth"] = False
    print(
        f"[bootstrap] num_gpus={_num_gpus} → disabling Unsloth (not compatible with multi-GPU)"
    )

# Unsloth early import — MUST happen before importing torch/transformers/trl
if _cfg.get("model", {}).get("use_unsloth", False):
    print(
        "[bootstrap] use_unsloth=True → importing Unsloth before torch/transformers/trl"
    )
    import unsloth as _unsloth  # noqa: F401

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
