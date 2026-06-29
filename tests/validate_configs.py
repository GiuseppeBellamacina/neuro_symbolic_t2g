"""
Config YAML Validator — Verifica che tutti i config YAML abbiano le sezioni
e chiavi obbligatorie.

Uso:
    python -m tests.validate_configs
    python -m tests.validate_configs --verbose
    python -m tests.validate_configs --config experiments/configs/t2g/grpo_qwen05.yaml

Regole di validazione:
    - Ogni config ha un "tipo" rilevato automaticamente (grpo, sft, eval-only)
    - Sezioni obbligatorie per tipo
    - Chiavi nidificate obbligatorie
    - Vincoli di tipo (bool, int, float, list)
    - Coerenza cross-sezione (es. grammar.use_grammarllm_pda → pda_temperature)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import yaml

# ── Project root ──────────────────────────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_CONFIG_GLOB = "experiments/configs/**/*.yaml"


# ═══════════════════════════════════════════════════════════════════════════════
# Validation rules
# ═══════════════════════════════════════════════════════════════════════════════

# Required top-level sections per config "kind"
REQUIRED_SECTIONS: dict[str, set[str]] = {
    # All configs must have these
    "_all": {"model", "dataset", "wandb"},
    # GRPO training configs
    "grpo": {"training", "reward", "grpo", "lora"},
    # SFT training configs (has training.trainer=sft)
    "sft": {"training", "reward", "generation", "lora"},
    # eval-only: solo _all, nessuna entry qui
}

# Required nested keys per section
REQUIRED_KEYS: dict[str, set[str]] = {
    "model": {"name", "num_gpus"},
    "dataset": {"dataset_name", "vocab_path", "bigram_matrix_path", "seed"},
    "training": {"output_dir", "log_dir"},
    "wandb": {"project", "run_name"},
}

# Exclusive-or: training must have EITHER max_steps OR num_train_epochs
TRAINING_STEPS_KEYS = {"max_steps", "num_train_epochs"}

# Type constraints: section.key → expected type
TYPE_CONSTRAINTS: dict[str, type | tuple[type, ...]] = {
    "model.num_gpus": int,
    "model.use_unsloth": bool,
    "dataset.seed": int,
    "dataset.thinking": bool,
    "training.max_steps": int,
    "training.num_train_epochs": (int, float),
    "training.per_device_train_batch_size": int,
    "training.gradient_accumulation_steps": int,
    "training.learning_rate": float,
    "training.warmup_ratio": float,
    "training.warmup_steps": int,
    "training.weight_decay": float,
    "training.max_grad_norm": float,
    "training.bf16": bool,
    "training.logging_steps": int,
    "training.save_steps": int,
    "training.save_total_limit": int,
    "training.max_seq_length": int,
    "grpo.num_generations": int,
    "grpo.max_completion_length": int,
    "grpo.max_prompt_length": int,
    "grpo.beta": float,
    "grpo.temperature": (int, float),
    "generation.max_completion_length": int,
    "generation.max_prompt_length": int,
    "generation.temperature": (int, float),
    "grammar.enabled": bool,
    "grammar.use_grammarllm_pda": bool,
    "grammar.pda_temperature": (int, float),
    "grammar.viterbi_diversity.self_loop_penalty": float,
    "grammar.viterbi_diversity.max_occurrences": int,
    "grammar.viterbi_diversity.diversity_threshold": float,
    "grammar.viterbi_diversity.max_iters": int,
    "curriculum.enabled": bool,
    "evaluation.batch_size": int,
    "lora.r": int,
    "lora.lora_alpha": int,
    "lora.lora_dropout": (int, float),
    "lora.random_state": int,
    "reward.weight_translation": (int, float),
    "reward.weight_gold_structure": (int, float),
    "reward.weight_format": (int, float),
    "reward.weight_repetition": (int, float),
}


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════


def _detect_kind(cfg: dict[str, Any]) -> str:
    """Detect the config kind: 'grpo', 'sft', or 'eval-only'."""
    trainer = cfg.get("training", {}).get("trainer", "grpo")
    if trainer == "sft":
        return "sft"
    if "grpo" in cfg and "num_generations" in cfg.get("grpo", {}):
        return "grpo"
    if "training" in cfg:
        return "grpo"  # has training section, assume GRPO
    return "eval-only"


def _get_nested(cfg: dict[str, Any], dotted_key: str) -> Any:
    """Get a nested value by dotted key, e.g. 'grammar.viterbi_diversity.self_loop_penalty'.

    Returns a sentinel object if any intermediate key is missing.
    """
    keys = dotted_key.split(".")
    current: Any = cfg
    for k in keys:
        if not isinstance(current, dict) or k not in current:
            return _MISSING
        current = current[k]
    return current


_MISSING = object()


def _validate_type(
    cfg: dict[str, Any],
    dotted_key: str,
    expected: type | tuple[type, ...],
    errors: list[str],
    path: str,
) -> None:
    """Validate that cfg[key] matches the expected type."""
    value = _get_nested(cfg, dotted_key)
    if value is _MISSING:
        return  # missing key is handled by REQUIRED_KEYS
    if not isinstance(value, expected):
        type_name = (
            " | ".join(t.__name__ for t in expected)  # type: ignore[union-attr]
            if isinstance(expected, tuple)
            else expected.__name__
        )
        actual = type(value).__name__
        errors.append(
            f"{path}: {dotted_key} deve essere {type_name}, "
            f"trovato {actual} ({value!r})"
        )


def _validate_reward_weights(cfg: dict[str, Any], errors: list[str], path: str) -> None:
    """Ensure reward weights sum to ~1.0 (warn if not)."""
    reward = cfg.get("reward", {})
    weights = {
        k: v
        for k, v in reward.items()
        if k.startswith("weight_") and isinstance(v, (int, float))
    }
    if not weights:
        return
    total = sum(weights.values())
    if abs(total - 1.0) > 0.15:
        errors.append(
            f"{path}: reward weights sum to {total:.3f} "
            f"(expected ~1.0); weights: {weights}"
        )


def _validate_cross_section(cfg: dict[str, Any], errors: list[str], path: str) -> None:
    """Cross-section consistency checks."""
    grammar = cfg.get("grammar", {})

    # If use_grammarllm_pda is true, pda_temperature should exist
    if grammar.get("use_grammarllm_pda"):
        if "pda_temperature" not in grammar:
            errors.append(
                f"{path}: grammar.use_grammarllm_pda=true "
                f"ma grammar.pda_temperature mancante"
            )

    # If grammar.enabled is true, viterbi_diversity should exist
    if grammar.get("enabled", True):
        if "viterbi_diversity" not in grammar:
            errors.append(
                f"{path}: grammar.enabled=true "
                f"ma grammar.viterbi_diversity mancante"
            )

    # Training configs should have either max_steps or num_train_epochs
    training = cfg.get("training", {})
    if training:
        has_steps = TRAINING_STEPS_KEYS & set(training.keys())
        if not has_steps:
            errors.append(f"{path}: training deve avere max_steps o num_train_epochs")

    # SFT must have generation section (not grpo)
    trainer = cfg.get("training", {}).get("trainer", "grpo")
    if trainer == "sft":
        if "generation" not in cfg and "grpo" not in cfg:
            errors.append(
                f"{path}: SFT config deve avere sezione 'generation' "
                f"(o 'grpo' come fallback)"
            )

    # GRPO configs must have grpo section with num_generations and beta
    if trainer != "sft" and "training" in cfg:
        grpo = cfg.get("grpo", {})
        if "num_generations" not in grpo:
            errors.append(f"{path}: GRPO config deve avere grpo.num_generations")
        if "beta" not in grpo:
            errors.append(f"{path}: GRPO config deve avere grpo.beta")


# ═══════════════════════════════════════════════════════════════════════════════
# Main validator
# ═══════════════════════════════════════════════════════════════════════════════


def validate_config(config_path: Path, verbose: bool = False) -> list[str]:
    """Validate a single config YAML file.

    Returns a list of error messages (empty = valid).
    """
    path = str(config_path.relative_to(_PROJECT_ROOT))
    errors: list[str] = []

    # ── Parse YAML ───────────────────────────────────────────────────────
    try:
        with open(config_path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
    except yaml.YAMLError as e:
        return [f"{path}: errore di parsing YAML: {e}"]
    except Exception as e:
        return [f"{path}: errore lettura file: {e}"]

    if cfg is None:
        return [f"{path}: file YAML vuoto"]

    if not isinstance(cfg, dict):
        return [
            f"{path}: il contenuto YAML non è un dizionario (tipo={type(cfg).__name__})"
        ]

    kind = _detect_kind(cfg)
    if verbose:
        print(f"  [{kind}] {path}")

    # ── Required top-level sections ──────────────────────────────────────
    required = set(REQUIRED_SECTIONS["_all"])
    for extra in (kind,):
        required |= REQUIRED_SECTIONS.get(extra, set())

    for section in sorted(required):
        if section not in cfg:
            errors.append(f"{path}: sezione '{section}' mancante")

    # ── Required nested keys ─────────────────────────────────────────────
    for section, keys in REQUIRED_KEYS.items():
        if section not in cfg:
            continue  # already reported above
        sec = cfg[section]
        if not isinstance(sec, dict):
            errors.append(
                f"{path}: '{section}' deve essere un dizionario, "
                f"trovato {type(sec).__name__}"
            )
            continue
        for key in sorted(keys):
            if key not in sec:
                errors.append(f"{path}: {section}.{key} mancante")

    # ── Type constraints ─────────────────────────────────────────────────
    for dotted_key, expected_type in TYPE_CONSTRAINTS.items():
        _validate_type(cfg, dotted_key, expected_type, errors, path)

    # ── Reward weights consistency ───────────────────────────────────────
    if "reward" in cfg:
        _validate_reward_weights(cfg, errors, path)

    # ── Cross-section consistency ────────────────────────────────────────
    _validate_cross_section(cfg, errors, path)

    return errors


def find_configs(config_root: Path | None = None) -> list[Path]:
    """Find all YAML config files."""
    root = config_root or (_PROJECT_ROOT / "experiments" / "configs")
    if not root.exists():
        print(f"⚠️  Directory config non trovata: {root}")
        return []
    return sorted(root.glob("**/*.yaml"))


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════


def main() -> None:
    parser = argparse.ArgumentParser(description="Validatore YAML per config T2G")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Valida un singolo config (default: tutti i config in experiments/configs/)",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Output dettagliato"
    )
    args = parser.parse_args()

    if args.config:
        config_path = _PROJECT_ROOT / args.config
        if not config_path.exists():
            print(f"[FAIL] File non trovato: {config_path}")
            sys.exit(1)
        configs = [config_path]
    else:
        configs = find_configs()

    if not configs:
        print("[INFO] Nessun config YAML trovato.")
        sys.exit(0)

    print(f"Validazione {len(configs)} config YAML...")
    print()

    total_errors = 0
    for config_path in configs:
        errors = validate_config(config_path, verbose=args.verbose)
        if errors:
            for err in errors:
                print(f"  FAIL  {err}")
            total_errors += len(errors)
        elif args.verbose:
            print(f"  OK    {config_path.relative_to(_PROJECT_ROOT)}")

    print()
    if total_errors == 0:
        print(f"[OK] Tutti i {len(configs)} config sono validi!")
        sys.exit(0)
    else:
        print(f"[FAIL] {total_errors} errori trovati in " f"{len(configs)} config.")
        sys.exit(1)


if __name__ == "__main__":
    main()
