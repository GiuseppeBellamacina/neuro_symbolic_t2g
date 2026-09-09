"""
Config YAML Validator — Verifica che tutti i config YAML abbiano le sezioni
e chiavi obbligatorie.

Uso:
    python -m tests.validate_configs
    python -m tests.validate_configs --verbose
    python -m tests.validate_configs --config experiments/configs/qwen25-05b/sft-grpo/few-shot.yaml

I config vengono caricati via ``src.utils.config.resolve_config``, quindi le
catene ``extends`` vengono risolte prima della validazione.

Regole di validazione:
    - Ogni config ha un "tipo" rilevato automaticamente (grpo, sft, eval-only)
    - Sezioni obbligatorie per tipo
    - Chiavi nidificate obbligatorie
    - Vincoli di tipo (bool, int, float, list)
    - Sezione `evaluation`: solo chiavi note (rifiuto dei typo, che altrimenti
      passerebbero silenziosamente), tipi, valori ammessi per `prompting` e
      coerenza `prompting: few-shot` ⇒ max_prompt_length >= 512
    - Coerenza cross-sezione (knob RL validi, peso OOV nel regime sicuro,
      max_prompt_length adeguato col few-shot attivo)
    - Somma dei reward weights = 1.0 (±1e-9)
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

# Chiavi ammesse nella sezione `evaluation`. Una chiave fuori da questo set è
# un typo (o un knob rimosso): eval_t2g la ignorerebbe silenziosamente e la
# cella misurerebbe altro senza alcun errore — il validator la blocca.
ALLOWED_EVALUATION_KEYS = {
    "batch_size",
    "max_samples",
    "num_samples",
    "resume_every",
    "best_of_n",
    "plot",
    "compare",
    "eval_baseline_only",
    "force_baseline_eval",
    "dual_prompting",
    "prompting",
    "output",
    "baseline_pass_at1",
    "baseline_json",
}

# Chiavi (qualunque sezione) che ammettono esplicitamente null: il type check
# standard rifiuterebbe None anche dove il codice lo tratta come "non
# impostato" (es. compare: null = deduzione automatica).
NULLABLE_TYPE_KEYS = {
    "evaluation.max_samples",
    "evaluation.compare",
    "evaluation.eval_baseline_only",
    "evaluation.output",
    "evaluation.baseline_json",
    "evaluation.baseline_pass_at1",
}

# Valori ammessi per evaluation.prompting (stessa scelta di eval_t2g.py):
# "config" deriva la modalità da retrieval.enabled, gli altri due la forzano.
ALLOWED_PROMPTING_MODES = {"config", "zero-shot", "few-shot"}

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
    "evaluation.batch_size": int,
    "evaluation.max_samples": int,
    "evaluation.num_samples": int,
    # Cadenza di salvataggio dello stato parziale (resume da walltime);
    # <= 0 disattiva il meccanismo.
    "evaluation.resume_every": int,
    "evaluation.best_of_n": bool,
    "evaluation.plot": bool,
    "evaluation.compare": bool,
    "evaluation.eval_baseline_only": bool,
    "evaluation.force_baseline_eval": bool,
    "evaluation.dual_prompting": bool,
    "evaluation.prompting": str,
    "evaluation.output": str,
    "evaluation.baseline_json": str,
    "evaluation.baseline_pass_at1": (int, float),
    "lora.r": int,
    "lora.lora_alpha": int,
    "lora.lora_dropout": (int, float),
    "lora.random_state": int,
    "reward.weight_translation": (int, float),
    "reward.weight_gold_structure": (int, float),
    "reward.weight_verifier_scaled": (int, float),
    "reward.weight_gloss_order": (int, float),
    "reward.weight_format": (int, float),
    "reward.weight_repetition": (int, float),
    "reward.weight_bleu": (int, float),
    "reward.weight_edit_validity": (int, float),
    "reward.edit_validity_oov_weight": (int, float),
    "grammar.track_diagnostics": bool,
    "grpo.epsilon": (int, float),
    "grpo.epsilon_high": (int, float),
    "grpo.mask_truncated_completions": bool,
}


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════


def _detect_kind(cfg: dict[str, Any]) -> str:
    """Detect the config kind: 'grpo', 'sft', or 'eval-only'.

    Un config è ``eval-only`` se non dichiara ``training.output_dir``: senza
    una directory di destinazione non c'è nulla da addestrare né da salvare.

    WHY output_dir e non le chiavi di step: ``max_steps`` e
    ``num_train_epochs`` vivono in ``base.yaml`` perché sono comuni a tutte
    le celle addestrabili, quindi vengono EREDITATE anche dalle celle
    eval-only. La loro presenza non distingue più nulla. ``output_dir``
    invece è per-cella per costruzione (ogni cella scrive in una directory
    propria) e la sua assenza è il segnale che il resto del sistema già usa:
    ``src/training/eval_t2g.py`` deduce da lì ``eval_baseline_only`` e
    ``compare``, e la guardia in ``src/training/__main__.py`` rifiuta con
    exit 2 una cella senza ``output_dir`` lanciata come training.

    Allineare il validatore a quel segnale elimina una seconda definizione
    divergente di "cella addestrabile".
    """
    trainer = cfg.get("training", {}).get("trainer", "grpo")
    if trainer == "sft":
        return "sft"
    if not cfg.get("training", {}).get("output_dir"):
        return "eval-only"
    return "grpo"


def _get_nested(cfg: dict[str, Any], dotted_key: str) -> Any:
    """Get a nested value by dotted key, e.g. 'grammar.track_diagnostics'.

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
    if value is None and dotted_key in NULLABLE_TYPE_KEYS:
        return  # null = "non impostato" (es. compare: null = deduzione automatica)
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
    # I knob dell'obiettivo RL devono avere valori che TRL accetta: un typo qui
    # addestrerebbe un obiettivo diverso per ore senza alcun errore.
    grpo_cfg = cfg.get("grpo", {})
    loss_type = grpo_cfg.get("loss_type")
    if loss_type is not None and loss_type not in {"grpo", "bnpo", "dr_grpo", "dapo"}:
        errors.append(
            f"{path}: grpo.loss_type={loss_type!r} non valido "
            f"(attesi: grpo, bnpo, dr_grpo, dapo)"
        )
    scale_rewards = grpo_cfg.get("scale_rewards")
    if (
        scale_rewards is not None
        and not isinstance(scale_rewards, bool)
        and scale_rewards not in {"group", "batch", "none"}
    ):
        errors.append(
            f"{path}: grpo.scale_rewards={scale_rewards!r} non valido "
            f"(attesi: group, batch, none)"
        )

    # Il peso del termine di validita' deve stare nel regime sicuro: a 0.75 la
    # validita' domina il contenuto e la spazzatura in vocabolario supera un
    # quasi-corretto con un OOV (tests/test_edit_validity_reward.py).
    reward = cfg.get("reward", {})
    oov_weight = reward.get("edit_validity_oov_weight")
    if oov_weight is not None and not 0.0 <= float(oov_weight) <= 0.6:
        errors.append(
            f"{path}: reward.edit_validity_oov_weight={oov_weight} fuori dal "
            f"regime sicuro [0.0, 0.6]"
        )

    # Il few-shot allunga il prompt: con retrieval attivo servono piu' token,
    # altrimenti gli esempi vengono troncati e la cella misura altro.
    if cfg.get("retrieval", {}).get("enabled"):
        max_prompt = grpo_cfg.get("max_prompt_length") or cfg.get("generation", {}).get(
            "max_prompt_length"
        )
        if max_prompt is not None and int(max_prompt) < 512:
            errors.append(
                f"{path}: retrieval.enabled=true ma max_prompt_length="
                f"{max_prompt} (<512): gli esempi few-shot verrebbero troncati"
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


def _validate_evaluation_section(
    cfg: dict[str, Any], errors: list[str], path: str
) -> None:
    """Validate the `evaluation` section: chiavi note, valori ammessi,
    coerenza col budget di prompt.

    eval_t2g.py legge i knob comportamentali SOLO da questa sezione (niente
    flag CLI oltre a --config/--checkpoint): una chiave sconosciuta sarebbe
    ignorata silenziosamente dal trainer — qui diventa un errore loud.
    """
    evaluation = cfg.get("evaluation")
    if evaluation is None:
        return  # assente: nessun knob dichiarato (tutti i default)
    if not isinstance(evaluation, dict):
        # il messaggio di tipo lo emette già il REQUIRED_KEYS/type check
        return

    # ── Chiavi sconosciute = typo (intercettate, non ignorate) ──────────
    for key in sorted(set(evaluation) - ALLOWED_EVALUATION_KEYS):
        errors.append(
            f"{path}: evaluation.{key} non riconosciuta (chiavi ammesse: "
            f"{', '.join(sorted(ALLOWED_EVALUATION_KEYS))})"
        )

    # ── Valori ammessi per prompting ────────────────────────────────────
    prompting = evaluation.get("prompting")
    if prompting is not None and prompting not in ALLOWED_PROMPTING_MODES:
        errors.append(
            f"{path}: evaluation.prompting={prompting!r} non valido "
            f"(attesi: {', '.join(sorted(ALLOWED_PROMPTING_MODES))})"
        )

    # ── prompting: few-shot ⇒ budget di prompt adeguato ─────────────────
    # Replica il fail-loud runtime di eval_t2g.py: forzare few-shot con
    # max_prompt_length < 512 troncherebbe gli esempi few-shot rendendo la
    # cella indistinguibile dallo zero-shot.
    if prompting == "few-shot":
        max_prompt = cfg.get("grpo", {}).get("max_prompt_length") or cfg.get(
            "generation", {}
        ).get("max_prompt_length")
        if max_prompt is None or int(max_prompt) < 512:
            errors.append(
                f"{path}: evaluation.prompting='few-shot' ma max_prompt_length="
                f"{max_prompt} (serve >= 512, in grpo o generation): gli esempi "
                f"few-shot verrebbero troncati"
            )


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
        # eval-only configs don't train → output/log dirs non richiesti
        if kind == "eval-only" and section == "training":
            continue
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

    # ── Evaluation section (knob dell'eval, ex flag CLI / env var) ───────
    _validate_evaluation_section(cfg, errors, path)

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
