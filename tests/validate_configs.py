"""
Config YAML Validator — Verifica che tutti i config YAML abbiano le sezioni
e chiavi obbligatorie.

Uso:
    python -m tests.validate_configs
    python -m tests.validate_configs --verbose
    python -m tests.validate_configs --config experiments/configs/qwen25-05b/sft-grpo/zero-shot.yaml

I config vengono caricati via ``src.utils.config.resolve_config``, quindi le
catene ``extends`` vengono risolte prima della validazione.

Regole di validazione:
    - Ogni config ha un "tipo" rilevato automaticamente (grpo, sft, eval-only)
    - Sezioni obbligatorie per tipo
    - Chiavi nidificate obbligatorie
    - Vincoli di tipo (bool, int, float, list)
    - Coerenza cross-sezione (es. grammar.use_grammarllm_pda → pda_temperature)
    - Somma dei reward weights = 1.0 (±1e-9)
    - Assenza di chiavi morte (verifier_gamma / verifier_temperature)
    - Assenza di ``extends`` residuo nel dict fuso
    - Ogni config YAML referenziato da cluster/run_all.sh esiste
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Any

import yaml

# Ensure project root is importable (also when run as a plain script).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.config import resolve_config

# ── Project root ──────────────────────────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_CONFIG_GLOB = "experiments/configs/**/*.yaml"
_CLUSTER_RUN_ALL = _PROJECT_ROOT / "cluster" / "run_all.sh"

# Chiavi morte: rimosse dal codice (src/rewards/t2g_rewards.py non le legge
# più — solo i 4 parametri viterbi_diversity reali vengono caricati).
DEAD_KEYS = {"verifier_gamma", "verifier_temperature"}


# ═══════════════════════════════════════════════════════════════════════════════
# Validation rules
# ═══════════════════════════════════════════════════════════════════════════════

# Required top-level sections per config "kind"
REQUIRED_SECTIONS: dict[str, set[str]] = {
    # All configs must have these
    "_all": {"experiment", "model", "dataset", "wandb"},
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
    "training": set(),
    "wandb": {"project", "run_name"},
}

# Exclusive-or: training must have EITHER max_steps OR num_train_epochs
TRAINING_STEPS_KEYS = {"max_steps", "num_train_epochs"}

# Type constraints: section.key → expected type
TYPE_CONSTRAINTS: dict[str, type | tuple[type, ...]] = {
    "model.num_gpus": int,
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
}

_TOKEN_RE = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════


def _detect_kind(cfg: dict[str, Any]) -> str:
    """Detect the config kind: 'grpo', 'sft', or 'eval-only'.

    Un config è ``eval-only`` se non dichiara alcun training attivo:
    niente ``training.trainer`` e nessuna chiave di step (``max_steps`` /
    ``num_train_epochs``) — può comunque ereditare una sezione ``training``
    parziale e un blocco ``grpo`` da ``base.yaml``.
    """
    if cfg.get("experiment", {}).get("kind") == "probe":
        return "probe"
    trainer = cfg.get("training", {}).get("trainer", "grpo")
    if trainer == "sft":
        return "sft"
    training = cfg.get("training", {})
    if training and (TRAINING_STEPS_KEYS & set(training.keys())):
        return "grpo"
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


def _iter_dead_keys(
    obj: Any, prefix: str = "", found: list[str] | None = None
) -> list[str]:
    """Collect any occurrence of a DEAD_KEYS key in a (nested) dict."""
    if found is None:
        found = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            dotted = f"{prefix}.{k}" if prefix else k
            if k in DEAD_KEYS:
                found.append(dotted)
            _iter_dead_keys(v, dotted, found)
    return found


def _validate_reward_weights(cfg: dict[str, Any], errors: list[str], path: str) -> None:
    """Reward weights must sum to 1.0 (±1e-9)."""
    reward = cfg.get("reward", {})
    weights = {
        k: v
        for k, v in reward.items()
        if k.startswith("weight_") and isinstance(v, (int, float))
    }
    if not weights:
        return
    total = sum(weights.values())
    if abs(total - 1.0) > 1e-9:
        errors.append(
            f"{path}: reward weights sum to {total:.6f} "
            f"(expected 1.0 ±1e-9); weights: {weights}"
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

    # Training configs should have either max_steps or num_train_epochs
    # (eval-only configs ereditano una sezione `training` parziale da base.yaml)
    training = cfg.get("training", {})
    if training and _detect_kind(cfg) != "eval-only":
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
    if trainer != "sft" and "training" in cfg and _detect_kind(cfg) != "eval-only":
        grpo = cfg.get("grpo", {})
        if "num_generations" not in grpo:
            errors.append(f"{path}: GRPO config deve avere grpo.num_generations")
        if "beta" not in grpo:
            errors.append(f"{path}: GRPO config deve avere grpo.beta")

    experiment = cfg.get("experiment")
    if experiment is not None:
        if not isinstance(experiment, dict):
            errors.append(f"{path}: experiment deve essere un dizionario")
        else:
            for key in ("model_tag", "method", "train_prompt_mode", "variant", "kind"):
                value = experiment.get(key)
                if not isinstance(value, str) or not _TOKEN_RE.fullmatch(value):
                    errors.append(f"{path}: experiment.{key} token invalido: {value!r}")
            if experiment.get("method") not in {"base", "sft", "grpo", "sft-grpo"}:
                errors.append(f"{path}: experiment.method enum invalido")
            if experiment.get("train_prompt_mode") not in {
                "none",
                "zero-shot",
                "few-shot",
            }:
                errors.append(f"{path}: experiment.train_prompt_mode enum invalido")
            if experiment.get("variant") not in {
                "none",
                "pda",
                "hot",
                "reward-edit",
                "reward-token-f1",
                "reward-chrfpp",
                "reward-rouge-l",
                "reward-sbleu2",
            }:
                errors.append(f"{path}: experiment.variant enum invalido")
            if experiment.get("kind") not in {"baseline", "train", "ablation", "probe"}:
                errors.append(f"{path}: experiment.kind enum invalido")


# ═══════════════════════════════════════════════════════════════════════════════
# Main validator
# ═══════════════════════════════════════════════════════════════════════════════


def validate_config(config_path: Path, verbose: bool = False) -> list[str]:
    """Validate a single config YAML file.

    The file is loaded through ``resolve_config`` (extends chains merged);
    the raw YAML is additionally checked for a stray top-level ``extends``
    surviving in the merged dict.

    Returns a list of error messages (empty = valid).
    """
    path = str(config_path.relative_to(_PROJECT_ROOT))
    errors: list[str] = []

    # ── Parse YAML (raw, per mostrare la catena extends) ────────────────
    try:
        with open(config_path, encoding="utf-8") as f:
            raw = yaml.safe_load(f)
    except yaml.YAMLError as e:
        return [f"{path}: errore di parsing YAML: {e}"]
    except Exception as e:
        return [f"{path}: errore lettura file: {e}"]

    if raw is None:
        return [f"{path}: file YAML vuoto"]

    if not isinstance(raw, dict):
        return [
            f"{path}: il contenuto YAML non è un dizionario (tipo={type(raw).__name__})"
        ]

    # ── Resolve extends chain ───────────────────────────────────────────
    try:
        cfg = resolve_config(config_path)
    except FileNotFoundError as e:
        return [f"{path}: extends non risolvibile — {e}"]
    except ValueError as e:
        return [f"{path}: extends invalido — {e}"]

    if "extends" in cfg:
        errors.append(f"{path}: chiave 'extends' residua nel dict fuso")

    kind = _detect_kind(cfg)
    if verbose:
        parents = raw.get("extends")
        if parents:
            p_str = ", ".join(parents) if isinstance(parents, list) else parents
            print(f"  [{kind}] {path} (extends: {p_str})")
        else:
            print(f"  [{kind}] {path}")

    if kind == "probe":
        if set(cfg) != {"experiment", "probe"}:
            errors.append(f"{path}: probe schema consente solo experiment e probe")
        if not isinstance(cfg.get("probe"), dict) or not cfg["probe"].get("name"):
            errors.append(f"{path}: probe.name mancante")
        if cfg.get("probe", {}).get("name") == "structured":
            required = {
                "model_name",
                "dataset_cache",
                "output_root",
                "feature_artifact",
                "top_k",
                "max_gloss_length",
                "alpha",
                "transition_scale",
                "head_lr",
                "weight_decay",
                "steps",
                "seed",
                "train_samples",
                "dev_samples",
                "dev_fraction",
                "feature_batch_size",
                "seeds",
                "max_peak_memory_bytes",
                "max_arm_runtime_seconds",
            }
            missing = required - set(cfg["probe"])
            if missing:
                errors.append(
                    f"{path}: structured probe chiavi mancanti: {sorted(missing)}"
                )
            expected = {
                "top_k": int,
                "max_gloss_length": int,
                "alpha": float,
                "transition_scale": float,
                "head_lr": float,
                "weight_decay": float,
                "steps": int,
                "seed": int,
                "train_samples": int,
                "dev_samples": int,
                "dev_fraction": float,
                "feature_batch_size": int,
                "seeds": list,
                "max_peak_memory_bytes": int,
                "max_arm_runtime_seconds": float,
            }
            for key, expected_type in expected.items():
                value = cfg["probe"].get(key)
                if value is not None and not isinstance(value, expected_type):
                    errors.append(
                        f"{path}: probe.{key} deve essere {expected_type.__name__}"
                    )
            fixed = {
                "top_k": 512,
                "max_gloss_length": 64,
                "alpha": 0.1,
                "transition_scale": 0.25,
            }
            for key, expected_value in fixed.items():
                if cfg["probe"].get(key) != expected_value:
                    errors.append(f"{path}: probe.{key} deve essere {expected_value}")
        _validate_cross_section(cfg, errors, path)
        return errors

    # ── Chiavi morte (verifier_gamma / verifier_temperature) ───────────
    for dotted in _iter_dead_keys(cfg):
        errors.append(f"{path}: chiave morta '{dotted}' (rimossa dal codice)")

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


def validate_cluster_references() -> list[str]:
    """Verify every config YAML referenced by cluster/run_all.sh exists.

    Parses the ``MODELS=( ... :path:mode ... )`` array lines.
    """
    errors: list[str] = []
    if not _CLUSTER_RUN_ALL.exists():
        errors.append(f"cluster/run_all.sh non trovato: {_CLUSTER_RUN_ALL}")
        return errors

    refs = re.findall(
        r"(?:experiments/configs/[\w/.-]+\.yaml)",
        _CLUSTER_RUN_ALL.read_text(encoding="utf-8"),
    )
    for ref in sorted(set(refs)):
        target = _PROJECT_ROOT / ref
        if not target.exists():
            errors.append(f"cluster/run_all.sh: config referenziato mancante: {ref}")
    return errors


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

    print(f"Validazione {len(configs)} config YAML (con risoluzione extends)...")
    print()

    total_errors = 0
    for config_path in configs:
        # base.yaml è un template di ereditarietà (non eseguibile): viene
        # validato indirettamente da ogni config che lo estende.
        if config_path.name == "base.yaml":
            if args.verbose:
                rel = config_path.relative_to(_PROJECT_ROOT)
                print(f"  [skip] {rel} (template di ereditarietà, non eseguibile)")
            continue
        errors = validate_config(config_path, verbose=args.verbose)
        if errors:
            for err in errors:
                print(f"  FAIL  {err}")
            total_errors += len(errors)
        elif args.verbose:
            print(f"  OK    {config_path.relative_to(_PROJECT_ROOT)}")

    # ── Riferimenti da cluster/run_all.sh ────────────────────────────────
    cluster_errors = validate_cluster_references()
    if cluster_errors:
        for err in cluster_errors:
            print(f"  FAIL  {err}")
        total_errors += len(cluster_errors)

    print()
    if total_errors == 0:
        print(f"[OK] Tutti i {len(configs)} config sono validi!")
        sys.exit(0)
    else:
        print(f"[FAIL] {total_errors} errori trovati in " f"{len(configs)} config.")
        sys.exit(1)


if __name__ == "__main__":
    main()
