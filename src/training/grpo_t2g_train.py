"""
GRPO T2G Training Loop — Phase 1.

Integrates Constrained Decoding with Group Relative Policy Optimization (GRPO)
for Text-to-Gloss (T2G) translation using HuggingFace + PEFT and TRL.

Architecture:
    1. Load model + tokenizer via HuggingFace (LoRA + 4-bit quantization).
    2. Load ASLG-PC12 dataset and build prompt-completion pairs.
    3. Compute/load bigram transition matrix (Viterbi proxy).
    4. Build gloss vocabulary mask for constrained decoding.
    5. Define reward functions (translation quality + structural proxy).
    6. Train with ``trl.GRPOTrainer``, constraining generation rollouts
       to ASL gloss tokens only.

Usage:
    python -m src.training --config experiments/configs/qwen25-05b/sft-grpo/few-shot.yaml
    CONFIG=experiments/configs/qwen25-05b/sft-grpo/few-shot.yaml sbatch cluster/train.sh
"""

from __future__ import annotations

import argparse
import gc

# ── Workaround: _is_package_available in transformers 5.3.0 restituisce
# una TUPLA (bool, str) invece di un bool: (False, None) e' truthy → trl
# prova a importare mergekit/llm_blender assenti. Fix piu' sotto.
import importlib
import logging
import os
import random
import sys
import warnings
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch

# tqdm fallback for Apptainer containers without tqdm installed
try:
    from tqdm import tqdm
except ImportError:

    def tqdm(iterable=None, **kwargs):
        return iterable if iterable is not None else iter(())


# ── Silence noisy transformers FutureWarnings ──────────────────────────
# transformers 5.3.0: 5 FutureWarning per generate() su AttentionMaskConverter
# (API interna, fix atteso in v5.10). Soppressi per tenere pulito il log.
warnings.filterwarnings("ignore", category=FutureWarning, module="transformers")
warnings.filterwarnings(
    "ignore",
    message=".*AttentionMaskConverter.*",
    category=FutureWarning,
)

# transformers 5.3.0 changed ``_is_package_available`` to always return a
# ``(bool, version)`` tuple, but TRL 0.24.0 assigns that result directly to its
# ``_<pkg>_available`` module flags. A non-empty tuple is truthy, so
# ``is_weave_available()`` reports True even when weave is absent and
# ``trl/trainer/callbacks.py`` then executes ``import weave``, making
# ``trl.trainer.grpo_trainer`` unimportable. Normalise the flags back to bool.
_trl_iu = importlib.import_module("trl.import_utils")  # noqa: E402
for _optional_flag in (
    "_mergekit_available",
    "_llm_blender_available",
    "_weave_available",
):
    if isinstance(getattr(_trl_iu, _optional_flag, False), tuple):
        setattr(_trl_iu, _optional_flag, False)

import wandb
from dotenv import load_dotenv
from transformers.integrations.integration_utils import WandbCallback
from transformers.trainer_callback import ProgressCallback
from trl import GRPOConfig, GRPOTrainer  # type: ignore[import]

from datasets import Dataset
from src.datasets.aslg_dataset import (
    build_t2g_dataset,
    download_aslg_dataset,
    extract_gloss_vocabulary,
    save_vocabulary,
)
from src.datasets.transition_matrix import (
    compute_bigram_transitions,
    load_transition_matrix,
    save_transition_matrix,
)
from src.grammar.gloss_grammar import GlossVocabularyMask
from src.grammar.grammar_logits_processor import GlossVocabularyLogitsProcessor
from src.models.model_loader import load_model_and_tokenizer
from src.retrieval import ExampleRetriever
from src.rewards.t2g_rewards import (
    build_t2g_reward_functions,
    initialize_rewards,
)
from src.training.retrieval_setup import (
    build_train_retriever,
    retrieve_few_shot_batch,
)
from src.utils.config import load_config
from src.utils.glossary import (
    build_example_glossary,
    compute_word_frequencies,
    format_glossary_block,
    should_include_glossary,
)
from src.utils.live_status import live_status_set
from src.utils.phase_timing import log_step, phase
from src.utils.prompting import build_t2g_prompt

# ───────────────────────────────────────────────────────────────────────────


load_dotenv()

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration helpers
# ---------------------------------------------------------------------------


def _build_grpo_config(
    training_cfg: dict[str, Any],
    grpo_cfg: dict[str, Any],
    full_config: dict[str, Any] | None = None,
    reward_weights: list[float] | None = None,
) -> GRPOConfig:
    """Build a ``GRPOConfig`` from config sections."""
    output_dir = training_cfg["output_dir"]
    log_dir = training_cfg["log_dir"]
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    Path(log_dir).mkdir(parents=True, exist_ok=True)

    warmup_kwargs: dict[str, Any] = {}
    if "warmup_steps" in training_cfg:
        warmup_kwargs["warmup_steps"] = training_cfg["warmup_steps"]
    else:
        warmup_kwargs["warmup_steps"] = 50

    wandb_cfg = (full_config or {}).get("wandb", {})
    from datetime import datetime

    base_name = wandb_cfg.get("run_name", "grpo-t2g")
    run_timestamp = training_cfg.get("run_timestamp") or datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )
    run_name = f"{base_name}-{run_timestamp}"

    # Set tensorboard logging dir via env var (logging_dir kwarg is deprecated
    # since transformers 5.2).
    os.environ.setdefault("TENSORBOARD_LOGGING_DIR", log_dir)

    return GRPOConfig(
        output_dir=output_dir,
        run_name=run_name,
        seed=training_cfg.get(
            "seed", (full_config or {}).get("dataset", {}).get("seed", 42)
        ),
        max_steps=training_cfg.get("max_steps", 1500),
        per_device_train_batch_size=training_cfg.get("per_device_train_batch_size", 1),
        gradient_accumulation_steps=training_cfg.get("gradient_accumulation_steps", 8),
        learning_rate=training_cfg.get("learning_rate", 5e-6),
        lr_scheduler_type=training_cfg.get("lr_scheduler_type", "cosine"),
        **warmup_kwargs,
        optim=training_cfg.get("optim", "paged_adamw_8bit"),
        weight_decay=training_cfg.get("weight_decay", 0.1),
        max_grad_norm=training_cfg.get("max_grad_norm", 0.1),
        bf16=training_cfg.get("bf16", True),
        # Gradient checkpointing: ~20% di compute in cambio di VRAM
        # (necessario per num_generations=8 su GPU 22GB). Default False per
        # non cambiare il comportamento delle config che non lo impostano.
        gradient_checkpointing=training_cfg.get("gradient_checkpointing", False),
        logging_steps=training_cfg.get("logging_steps", 5),
        save_steps=training_cfg.get("save_steps", 100),
        save_total_limit=training_cfg.get("save_total_limit", 3),
        # GRPO-specific
        num_generations=grpo_cfg.get("num_generations", 4),
        max_completion_length=grpo_cfg.get("max_completion_length", 256),
        max_prompt_length=grpo_cfg.get("max_prompt_length", 256),
        beta=grpo_cfg.get("beta", 0.04),
        temperature=grpo_cfg.get("temperature", 0.7),
        reward_weights=reward_weights,
        report_to="wandb",
        # ── RL objective knobs (config pass-through, vedi _grpo_objective_kwargs) ──
        # loss_type='dr_grpo' divide per B * max_completion_length (Dr-GRPO,
        # arXiv:2503.20783); scale_rewards='none' mantiene A = R - mean(R) MA
        # preserva la scala del reward, quindi un reward che e' una contrazione
        # affine di un altro cambia l'effective step size (vedi
        # edit_validity_reward). TRL 0.24.0 NON implementa il dynamic sampling
        # di DAPO e clip-higher richiede epsilon_high esplicito.
        **_grpo_objective_kwargs(grpo_cfg),
    )


def _grpo_objective_kwargs(grpo_cfg: dict[str, Any]) -> dict[str, Any]:
    """Validated pass-through for the RL objective knobs.

    Only keys explicitly present in the config are forwarded, so omitting them
    preserves TRL's defaults exactly and keeps historical runs reproducible.

    Raises:
        ValueError: On an unsupported value, so a typo fails before the job
            starts rather than silently training a different objective.
    """
    kwargs: dict[str, Any] = {}

    if "loss_type" in grpo_cfg:
        loss_type = str(grpo_cfg["loss_type"])
        allowed = {"grpo", "bnpo", "dr_grpo", "dapo"}
        if loss_type not in allowed:
            raise ValueError(
                f"grpo.loss_type must be one of {sorted(allowed)}, got {loss_type!r}"
            )
        kwargs["loss_type"] = loss_type

    if "scale_rewards" in grpo_cfg:
        scale = grpo_cfg["scale_rewards"]
        allowed_scale = {"group", "batch", "none"}
        if isinstance(scale, bool):
            scale = "group" if scale else "none"
        scale = str(scale)
        if scale not in allowed_scale:
            raise ValueError(
                f"grpo.scale_rewards must be one of {sorted(allowed_scale)}, "
                f"got {scale!r}"
            )
        kwargs["scale_rewards"] = scale

    if "mask_truncated_completions" in grpo_cfg:
        value = grpo_cfg["mask_truncated_completions"]
        if not isinstance(value, bool):
            raise ValueError(
                f"grpo.mask_truncated_completions must be a boolean, got {value!r}"
            )
        kwargs["mask_truncated_completions"] = value

    if "epsilon" in grpo_cfg:
        epsilon = float(grpo_cfg["epsilon"])
        if epsilon <= 0:
            raise ValueError(f"grpo.epsilon must be positive, got {epsilon!r}")
        kwargs["epsilon"] = epsilon

    if "epsilon_high" in grpo_cfg:
        epsilon_high = float(grpo_cfg["epsilon_high"])
        low = float(grpo_cfg.get("epsilon", 0.2))
        if epsilon_high < low:
            raise ValueError(
                f"grpo.epsilon_high ({epsilon_high}) must be >= grpo.epsilon ({low})"
            )
        kwargs["epsilon_high"] = epsilon_high

    return kwargs


# ---------------------------------------------------------------------------
# Dataset preparation
# ---------------------------------------------------------------------------


def _prepare_t2g_dataset(
    config: dict[str, Any],
    tokenizer: Any,
    vocab: list[str],
    dataset: Any = None,
    *,
    retriever: ExampleRetriever | None = None,
    retrieval_cfg: dict[str, Any] | None = None,
) -> Dataset:
    """Load ASLG-PC12 and build prompt-completion pairs for GRPO.

    The dataset has columns: ``prompt``, ``text``, ``completion``,
    ``gold_gloss``, ``sample_id``, ``difficulty``.  All extra columns are
    intentionally preserved: TRL 0.24's ``GRPOTrainer`` forwards every
    dataset column (except ``prompt``/``completion``/``completion_ids``) to
    the reward functions as keyword arguments, so the ``gold_gloss`` column
    is what feeds the gold reference to the rewards at rollout time.

    When ``retriever`` is given, every prompt is augmented with ``top_k``
    similar ``(text, gloss)`` examples retrieved from the TRAIN split (same
    strategy as ``eval_t2g.py``, so train/inference prompts stay coherent).
    Retrieval happens ONCE up front — never per training step.  Anti-leakage
    (excluding the query's own normalized text and dropping near-duplicates
    above ``max_self_similarity``) is handled by
    :func:`retrieve_few_shot_batch`.

    Args:
        config: Full config dict.
        tokenizer: Hugging Face tokenizer.
        vocab: Gloss vocabulary (unused here, kept for API compatibility).
        dataset: Optional pre-loaded ``DatasetDict``. If ``None``, downloads it.
        retriever: Optional few-shot retriever built over the train split
            (see ``src/training/retrieval_setup.py``). ``None`` ⇒ zero-shot.
        retrieval_cfg: Resolved ``retrieval`` config section (``top_k``,
            ``max_self_similarity``); ignored when ``retriever`` is ``None``.
    """
    ds_cfg = config["dataset"]
    if dataset is None:
        dataset = download_aslg_dataset(
            cache_dir=ds_cfg.get("dataset_cache"), seed=ds_cfg.get("seed", 42)
        )

    t2g_ds = build_t2g_dataset(
        dataset,
        split=ds_cfg.get("split", "train"),
        max_samples=ds_cfg.get("max_samples"),
    )

    # Retrieve few-shot examples for every sample in one pass (the tfidf
    # backend is deterministic and fast; this never runs during rollout).
    top_k = int((retrieval_cfg or {}).get("top_k", 3))
    max_self_similarity = float((retrieval_cfg or {}).get("max_self_similarity", 0.98))
    # WHY phase: 72.979 accessi singoli ad Arrow (~10-60s), completamente muti.
    with phase("Extracting prompts from dataset", detail=f"{len(t2g_ds)} rows"):
        texts = [t2g_ds[i]["prompt"] for i in range(len(t2g_ds))]
    examples_batch = (
        retrieve_few_shot_batch(retriever, texts, top_k, max_self_similarity)
        if retriever is not None
        else None
    )

    # Train-time rare-word glossary (opt-in, never used at eval — see
    # src/utils/glossary.py for the full design and why the block is dropped
    # on a fraction of examples). Frequencies are computed over this SAME
    # resolved train subset (the ``texts`` just extracted above), never over
    # eval/test data.
    glossary_cfg = config.get("glossary", {})
    glossary_enabled = bool(glossary_cfg.get("enabled", False))
    word_frequencies: Counter[str] | None = None
    glossary_max_freq = int(glossary_cfg.get("max_freq", 3))
    glossary_dropout = float(glossary_cfg.get("dropout", 0.5))
    if glossary_enabled:
        with phase(
            "Computing train word frequencies for glossary",
            detail=f"{len(texts)} rows",
        ):
            word_frequencies = compute_word_frequencies(texts)

    # Format prompts with the centralized T2G prompt builder.
    # This guarantees train/eval/test use identical formatting.
    # WHY phase+barra: 72.979 build_t2g_prompt con apply_chat_template sono
    # la fase di setup più lunga (2-6 min), finora completamente muta; la
    # barra segue il pattern di src/datasets/aslg_dataset.py.
    with phase("Formatting prompts with chat template", detail=f"{len(t2g_ds)} rows"):
        formatted: list[dict[str, str]] = []
        for i in tqdm(range(len(t2g_ds)), desc="Formatting T2G prompts"):
            sample = t2g_ds[i]
            text = sample["prompt"]

            glossary_block = None
            if word_frequencies is not None:
                sample_id = sample.get("sample_id", "")
                if should_include_glossary(sample_id, glossary_dropout):
                    example_glossary = build_example_glossary(
                        text,
                        sample["completion"],
                        word_frequencies,
                        max_freq=glossary_max_freq,
                    )
                    glossary_block = format_glossary_block(example_glossary) or None

            prompt = build_t2g_prompt(
                text,
                tokenizer,
                examples=examples_batch[i] if examples_batch is not None else None,
                glossary_block=glossary_block,
            )

            # Keep every column produced by build_t2g_dataset: ``gold_gloss``
            # deve arrivare al GRPOTrainer perche' TRL la forwarda alle reward
            # fn come kwarg (t2g_rewards._make_gloss_reward_fn); ``sample_id``
            # codifica gia' text+gold gloss, non va ricalcolato.
            formatted.append(
                {
                    "prompt": prompt,
                    "text": sample.get("text", text),
                    "completion": sample["completion"],
                    "gold_gloss": sample.get("gold_gloss", sample["completion"]),
                    "difficulty": sample.get("difficulty", "medium"),
                    "sample_id": sample.get("sample_id", ""),
                }
            )

    # WHY phase: l'encoding Arrow di 72.979 righe impiega 10-60s in silenzio.
    with phase("Encoding dataset to Arrow", detail=f"{len(formatted)} rows"):
        result = Dataset.from_list(formatted)
    logger.info(f"[dataset] T2G training set: {len(result)} prompts")
    return result


# ---------------------------------------------------------------------------
# Cache invalidation for vocab / bigram artifacts
# ---------------------------------------------------------------------------


def _cache_meta_path(cache_path: str | Path) -> Path:
    """Return the sidecar JSON path for a cache file (``<stem>.meta.json``).

    Args:
        cache_path: Path to the cache artifact (e.g. ``data/gloss_vocab.txt``).

    Returns:
        Sidecar path, e.g. ``data/gloss_vocab.meta.json``.
    """
    return Path(cache_path).with_suffix(".meta.json")


def _write_cache_meta(cache_path: str | Path, seed: int, train_size: int) -> None:
    """Write a sidecar JSON recording ``{seed, train_size}`` for a cache file.

    Args:
        cache_path: Path to the cache artifact.
        seed: Random seed used to build the artifact.
        train_size: Size of the training split used to build the artifact.
    """
    import json

    meta_path = _cache_meta_path(cache_path)
    meta_path.write_text(
        json.dumps({"seed": seed, "train_size": train_size}, sort_keys=True),
        encoding="utf-8",
    )


def _cache_is_current(cache_path: str | Path, seed: int, train_size: int) -> bool:
    """Check whether a cached artifact is up to date for the current run.

    A cache is valid only if the file exists AND its sidecar JSON matches the
    current ``(seed, train_size)``.  Legacy cache files WITHOUT a sidecar are
    NEVER trusted and are regenerated — this prevents silently reusing
    vocab/bigram artifacts built under a different seed or before dataset
    dedup changed the training-set composition.

    Args:
        cache_path: Path to the cache artifact.
        seed: Current run seed.
        train_size: Current training split size.

    Returns:
        ``True`` if the cache is current, ``False`` otherwise.
    """
    import json

    path = Path(cache_path)
    meta_path = _cache_meta_path(cache_path)
    if not path.exists() or not meta_path.exists():
        return False
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return meta.get("seed") == seed and meta.get("train_size") == train_size


# ---------------------------------------------------------------------------
# Vocabulary-constrained generation config for GRPO
# ---------------------------------------------------------------------------


def _build_generation_kwargs(
    config: dict[str, Any],
) -> dict[str, Any]:
    """Build generation kwargs for GRPO rollouts.

    Set via ``grpo_config.generation_kwargs`` before creating GRPOTrainer.
    In trl 0.24.0, generation_kwargs lives on GRPOConfig (args), not on
    GRPOTrainer.__init__() directly.

    **IMPORTANT**: ``logits_processor`` is NOT included here.  trl 0.24.0
    passes generation_kwargs to ``GenerationConfig(**kwargs)``, and
    transformers 5.3.0 rejects ``logits_processor`` as a GenerationConfig
    argument.  Instead, the logits processor is injected via a monkey-patch
    of ``model.generate()`` in ``main()``.

    Args:
        config: Full config dict.

    Returns:
        Dict of generation kwargs compatible with ``GenerationConfig()``.
    """
    grpo_cfg = config.get("generation", config.get("grpo", {}))
    kwargs: dict[str, Any] = {
        "max_new_tokens": grpo_cfg.get("max_completion_length", 128),
    }
    return kwargs


# ---------------------------------------------------------------------------
# Main training entry point
# ---------------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser.

    Extracted from :func:`main` so tests can exercise the argument wiring
    (e.g. ``--force-sft``) without launching a training run.
    """
    parser = argparse.ArgumentParser(
        description="GRPO training for Text-to-Gloss (T2G) with constrained decoding"
    )
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    parser.add_argument(
        "--resume", action="store_true", help="Resume from latest checkpoint"
    )
    parser.add_argument(
        "--prepare-data",
        action="store_true",
        help="Only prepare data (download dataset, compute transitions, save vocab)",
    )
    parser.add_argument(
        "--force-sft",
        action="store_true",
        help="Ignore SFT adapter reuse and always retrain SFT from scratch "
        "(bypasses the fingerprint match on previously saved SFT adapters)",
    )
    return parser


def main() -> None:
    """Main entry point for T2G GRPO training."""
    args = build_arg_parser().parse_args()

    config = load_config(args.config)

    # ── Resolve timestamped output/log directories and resume logic ──────
    from datetime import datetime

    training_cfg = config.get("training", {})

    # Le celle eval-only (baseline/*) NON dichiarano output_dir/log_dir:
    # lanciarle con cluster/train.sh e' errore d'uso. Prima qui partiva un
    # KeyError: 'output_dir' dopo il caricamento di Unsloth/modello (minuti).
    missing = [key for key in ("output_dir", "log_dir") if key not in training_cfg]
    if missing:
        has_steps = bool({"max_steps", "num_train_epochs"} & set(training_cfg))
        raise SystemExit(
            f"\n[grpo] Config non addestrabile: {args.config}\n"
            f"       Chiavi mancanti in `training`: {', '.join(missing)}.\n"
            + (
                "       Questa e' una cella EVAL-ONLY (nessun training.max_steps "
                "ne' num_train_epochs).\n"
                "       Usa cluster/eval.sh, non cluster/train.sh:\n"
                f"         CONFIG={args.config} sbatch cluster/eval.sh\n"
                if not has_steps
                else "       La cella dichiara step di training ma non le "
                "directory di output: aggiungi training.output_dir e "
                "training.log_dir al config.\n"
            )
        )

    base_output_dir = Path(training_cfg["output_dir"])
    base_log_dir = Path(training_cfg["log_dir"])

    run_timestamp = None
    if args.resume:
        run_folders = sorted(base_output_dir.glob("run_*"))
        if run_folders:
            output_dir = run_folders[-1]
            run_timestamp = output_dir.name.removeprefix("run_")
            log_dir = base_log_dir / f"run_{run_timestamp}"
            print(f"[grpo] Resuming training in existing directory: {output_dir}")
        else:
            print(
                f"[grpo] Warning: No existing run directory found in {base_output_dir} to resume. Creating a new run."
            )

    if run_timestamp is None:
        run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = base_output_dir / f"run_{run_timestamp}"
        log_dir = base_log_dir / f"run_{run_timestamp}"
        print(f"[grpo] Starting new training run. Output dir: {output_dir}")

    config["training"]["output_dir"] = str(output_dir)
    config["training"]["log_dir"] = str(log_dir)
    config["training"]["run_timestamp"] = run_timestamp

    # Safe config access: support both 'grpo' (GRPO) and 'generation' (SFT) keys
    grpo_cfg = config.get("generation", config.get("grpo", {}))

    # ── Setup logging ────────────────────────────────────────────────────
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
    )
    # Quiet down external libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("datasets").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    # HF libraries attach their own StreamHandler AND propagate to root —
    # every library warning printed twice (slurm-train-7073). Strip the
    # library-owned handlers so each record prints exactly once.
    from src.utils.log_dedup import dedupe_library_loggers

    dedupe_library_loggers()

    # ── Set random seeds for reproducibility ─────────────────────────────
    seed = config["dataset"].get("seed", 42)
    random.seed(seed)
    np.random.seed(seed)  # noqa: NPY002
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    print(f"[grpo] Reproducibility: seed={seed} (random, numpy, torch, cuda)")

    # ── Step 1: Data preparation ─────────────────────────────────────────
    ds_cfg = config["dataset"]
    vocab_path = ds_cfg.get("vocab_path", "data/gloss_vocab.txt")
    bigram_path = ds_cfg.get("bigram_matrix_path", "data/bigram_transition.npy")

    log_step(1, "Data Preparation")

    # Download dataset
    # WHY phase: load_dataset + dedup di ~80k righe + split 90/10: 10-60s in
    # silenzio, al primo avvio il job sembra appeso.
    with phase("Loading ASLG-PC12 dataset", detail="cache + dedup + 90/10 split"):
        dataset = download_aslg_dataset(
            cache_dir=ds_cfg.get("dataset_cache"), seed=ds_cfg.get("seed", 42)
        )

    # Cache keyed by (seed, train_size): se cambia uno dei due (nuovo seed,
    # dedup che cambia la composizione dello split) vocab e bigram vanno
    # rigenerati. Cache legacy senza sidecar mai fidata (vedi _cache_is_current).
    train_size = len(dataset["train"])

    # Extract vocabulary (or load from cache if still current)
    if _cache_is_current(vocab_path, seed, train_size):
        from src.datasets.aslg_dataset import load_vocabulary

        vocab = load_vocabulary(vocab_path)
    else:
        vocab = extract_gloss_vocabulary(dataset, split="train")
        save_vocabulary(vocab, vocab_path)
        _write_cache_meta(vocab_path, seed, train_size)

    # Compute transition matrix (or load from cache if still current)
    if _cache_is_current(bigram_path, seed, train_size):
        bigram_matrix = load_transition_matrix(bigram_path)
    else:
        bigram_matrix = compute_bigram_transitions(
            dataset, vocab, split="train", smoothing=1.0
        )
        save_transition_matrix(bigram_matrix, bigram_path)
        _write_cache_meta(bigram_path, seed, train_size)

    print(f"  Data prepared: |V|={len(vocab)}, bigram shape={bigram_matrix.shape}")

    if args.prepare_data:
        print("Data preparation complete. Exiting.")
        return

    # ── Step 1.5: Optional SFT Pre-training ─────────────────────────────
    sft_adapter_path: str | None = None
    sft_pretrain_cfg = config.get("sft_pretrain", {})
    if sft_pretrain_cfg.get("enabled", False):
        print(f"\n{'=' * 60}")
        print("STEP 1.5: SFT Pre-training")
        # Live status: the SFT phase begins (adapter reuse skips this block).
        live_status_set(phase="sft", note="SFT pre-training")

        from src.training.sft_train import (
            clone_sft_adapter,
            compute_sft_fingerprint,
            find_reusable_sft_adapter,
            find_reusable_sft_adapter_cross_tag,
            is_complete_adapter_dir,
            run_sft,
        )

        # Build a synthetic config for run_sft using sft_pretrain section
        sft_config = {
            **config,
            "training": {
                **config["training"],
                **sft_pretrain_cfg.get("training", {}),
                "output_dir": str(
                    sft_pretrain_cfg.get(
                        "output_dir",
                        Path(config["training"]["output_dir"]) / "sft_pretrain",
                    )
                ),
                "log_dir": str(
                    sft_pretrain_cfg.get(
                        "log_dir",
                        Path(config["training"]["log_dir"]) / "sft_pretrain",
                    )
                ),
                "trainer": "sft",
            },
        }

        # ── SFT adapter reuse ──────────────────────────────────────────
        # Se un run precedente ha gia' prodotto un adapter SFT con lo STESSO
        # fingerprint (model/lora/dataset/system prompt/iperparametri — vedi
        # compute_sft_fingerprint) riaddestrare SFT e' spreco: riusalo. Il
        # riuso salta solo lo Step 1.5; GRPO prosegue dai SUOI checkpoint,
        # quindi `--resume` resta invariato.
        #   - sft_pretrain.reuse_adapter: false → riaddestra sempre
        #   - sft_pretrain.adapter_path: <dir>  → adapter esplicito (no search)
        #   - --force-sft                        → riaddestra sempre (CLI)
        explicit_adapter = sft_pretrain_cfg.get("adapter_path")
        reuse_adapter = sft_pretrain_cfg.get("reuse_adapter", True)
        reused_adapter: str | None = None

        if explicit_adapter is not None:
            if is_complete_adapter_dir(explicit_adapter):
                reused_adapter = str(explicit_adapter)
                print(f"  Reusing SFT adapter (explicit path): {reused_adapter}")
            else:
                print(
                    "  ⚠️  Explicit adapter_path missing or incomplete: "
                    f"{explicit_adapter} — ignoring it"
                )
        elif reuse_adapter and not args.force_sft:
            fingerprint = compute_sft_fingerprint(sft_config)
            # Search order: (1) sibling runs of the SAME tag, e.g.
            # experiments/checkpoints/qwen25-05b-sft-grpo/run_*/sft_pretrain/final;
            # (2) ANY other tag sotto experiments/checkpoints/ con sezione
            # sft_pretrain IDENTICA (il fingerprint e' tag-independent), cosi'
            # un nuovo tag salta il ~1h di retrain SFT. I match cross-tag
            # vengono COPIATI in sft_pretrain/final di questo run perche'
            # resti self-contained.
            model_root = Path(config["training"]["output_dir"]).parent
            found = find_reusable_sft_adapter(model_root, fingerprint)
            if found is not None:
                reused_adapter = str(found)
                print(
                    "  Reusing SFT adapter from "
                    f"{Path(reused_adapter).parent.parent.name} "
                    "(fingerprint match) — skipping SFT training"
                )
            else:
                cross = find_reusable_sft_adapter_cross_tag(
                    model_root.parent, model_root, fingerprint
                )
                if cross is not None:
                    src_adapter, src_tag = cross
                    dest = (
                        Path(config["training"]["output_dir"])
                        / "sft_pretrain"
                        / "final"
                    )
                    clone_sft_adapter(src_adapter, dest)
                    reused_adapter = str(dest)
                    print(
                        f"  Reusing SFT adapter from tag '{src_tag}' "
                        "(identical SFT config — fingerprint match), "
                        f"copied to {dest} — skipping SFT training"
                    )

        if reused_adapter is not None:
            sft_adapter_path = reused_adapter
        else:
            if args.force_sft:
                print(
                    "  --force-sft: retraining SFT from scratch "
                    "(adapter reuse disabled)"
                )
            sft_adapter_path = run_sft(sft_config, resume=args.resume)
            print(f"  SFT adapter saved to: {sft_adapter_path}")

        # Aggressive cleanup between SFT and GRPO
        gc.collect()
        torch.cuda.empty_cache()

    # ── Step 2: Model loading ────────────────────────────────────────────
    log_step(2, "Model Loading")

    # WHY phase: il caricamento 4-bit + tokenizer + LoRA è la parte più lunga
    # del setup (decine di secondi); l'annuncio esce PRIMA, la durata DOPO.
    # Le sotto-fasi dettagliate sono emesse da model_loader stesso.
    with phase(
        "Loading model + tokenizer",
        detail=config["model"]["name"] + (" + SFT adapter" if sft_adapter_path else ""),
    ):
        model, tokenizer = load_model_and_tokenizer(
            config, adapter_path=sft_adapter_path
        )

    # ── Step 3: Constrained decoding setup ────────────────────────────────
    log_step(3, "Constrained Decoding Setup")

    # Grammar toggle: set ``grammar.enabled: false`` to disable constrained
    # decoding (for ablation study — GRPO without grammar).
    grammar_enabled = config.get("grammar", {}).get("enabled", True)
    if not grammar_enabled:
        print(
            "  ⚠️  grammar.enabled=false — GRPO rollouts will use UNCONSTRAINED "
            "generation (no vocabulary mask).  This is intended for ablation "
            "studies only."
        )
        logits_processor_for_gen = None
    else:
        print("  Using lightweight GlossVocabularyMask for constrained decoding")
        gloss_mask = GlossVocabularyMask(vocab, tokenizer)
        logits_processor_for_gen = GlossVocabularyLogitsProcessor(
            gloss_mask, device="cuda" if torch.cuda.is_available() else "cpu"
        )
        print("  Vocabulary mask ready")

    # ── Step 4: Dataset preparation ──────────────────────────────────────
    log_step(4, "Dataset Preparation")

    # ── Optional few-shot retrieval (train-split demonstrations) ─────────
    # Build (or load from cache) the ExampleRetriever over the deduplicated
    # TRAIN split.  When enabled, every GRPO prompt is augmented with top_k
    # similar (text→gloss) examples; the query itself and near-duplicates
    # are always excluded (anti-leakage).  SFT pre-training stays zero-shot.
    retrieval_cfg = config.get("retrieval", {})
    retriever = build_train_retriever(
        dataset,
        retrieval_cfg,
        seed=ds_cfg.get("seed", 42),
    )
    if retriever is not None:
        print(
            f"  Few-shot retrieval ENABLED: backend={retriever.backend}, "
            f"top_k={retrieval_cfg.get('top_k', 3)}, "
            f"max_self_similarity="
            f"{retrieval_cfg.get('max_self_similarity', 0.98)}"
        )
        # Few-shot examples (~40-60 tokens each) inflate prompt length; the
        # default 256 can silently truncate them.  Warn — never force.
        max_prompt_length = grpo_cfg.get("max_prompt_length", 256)
        if max_prompt_length < 768:
            logger.warning(
                "[retrieval] Few-shot retrieval is enabled but "
                "grpo.max_prompt_length=%d < 768: few-shot prompts may be "
                "truncated during rollout. Consider raising it in the config "
                "(e.g. grpo.max_prompt_length: 768). Not forced automatically.",
                max_prompt_length,
            )
    else:
        print("  Few-shot retrieval disabled — zero-shot prompts")

    t2g_dataset = _prepare_t2g_dataset(
        config,
        tokenizer,
        vocab,
        dataset=dataset,
        retriever=retriever,
        retrieval_cfg=retrieval_cfg,
    )

    # NOTE: niente piu' registry gold-gloss: la colonna ``gold_gloss`` e'
    # forwardata da TRL alle reward fn come kwarg (vedi _prepare_t2g_dataset),
    # senza il problema di collisione SHA256 del vecchio registry.

    # ── Step 5: Reward functions ─────────────────────────────────────────
    log_step(5, "Reward Functions")

    initialize_rewards(
        bigram_matrix,
        vocab,
    )
    reward_fns, reward_weights = build_t2g_reward_functions(config.get("reward"))

    # ── Wire completion sample logging (for live chain_monitor display) ─
    from src.training.callbacks import (
        CompletionSampleCallback,
        CompletionSampleLogger,
        HighPrecisionLogCallback,
        TqdmOnlyProgressCallback,
    )

    sample_logger = CompletionSampleLogger(reward_fns, reward_weights, n_samples=3)
    # WHY phase: set_difficulty_map itera TUTTO il dataset (~73k righe) in
    # silenzio per costruire la lookup prompt→difficoltà.
    with phase("Indexing difficulty map", detail=f"{len(t2g_dataset)} samples"):
        sample_logger.set_difficulty_map(t2g_dataset)
    wrapped_reward_fns = sample_logger.wrapped_reward_fns
    sample_callback = CompletionSampleCallback(
        sample_logger,
        every_n_steps=5,
        logits_processor=logits_processor_for_gen,
    )

    # ── Step 6: GRPO configuration ───────────────────────────────────────
    log_step(6, "GRPO Configuration")

    grpo_config = _build_grpo_config(
        config["training"],
        grpo_cfg,
        config,
        reward_weights=reward_weights,
    )

    print(
        f"[grpo] max_steps={grpo_config.max_steps}, "
        f"batch={grpo_config.per_device_train_batch_size}, "
        f"grad_accum={grpo_config.gradient_accumulation_steps}, "
        f"lr={grpo_config.learning_rate}, "
        f"num_gen={grpo_config.num_generations}, "
        f"beta={grpo_config.beta}, "
        f"max_completion={grpo_config.max_completion_length}"
    )

    # ── Workaround: unsloth-zoo autocast dtype defaults to float16 ───────
    # `unsloth_zoo.rl_replacements.grpo_accumulated_loss` inizializza lazy
    # `trainer._autocast_dtype` dal RAW env var ACCELERATE_MIXED_PRECISION,
    # bypassando lo stato di Accelerate. GRPOConfig(bf16=True) NON tocca
    # os.environ (solo `accelerate launch`/DeepSpeed lo fanno; qui si lancia
    # via `python -m src.training`), quindi l'env var resta unset → fp16 →
    # autocast float16 su pesi bfloat16:
    #   RuntimeError: self and mat2 must have the same dtype (Half vs Float).
    # Fix: impostare l'env var da grpo_config.bf16 PRIMA di costruire
    # GRPOTrainer, cosi' il lazy check raccoglie il dtype giusto.
    os.environ["ACCELERATE_MIXED_PRECISION"] = "bf16" if grpo_config.bf16 else "fp16"

    # ── Resume logic ─────────────────────────────────────────────────────
    resume_from: str | None = None
    if args.resume:
        ckpts = sorted(Path(grpo_config.output_dir).glob("checkpoint-*"))
        if ckpts:
            resume_from = str(ckpts[-1])
            print(f"[grpo] Resuming from {resume_from}")

    # ── Wandb setup ──────────────────────────────────────────────────────
    # Modalità offline (cluster senza internet) — come grpo-strict-generation.
    # WANDB_MODE=offline è già esportato da train.sh; lo rinforziamo qui.
    wandb_cfg = config.get("wandb", {})
    log_dir = config["training"]["log_dir"]
    if "WANDB_MODE" not in os.environ:
        os.environ["WANDB_MODE"] = "offline"
    # Disable weave (wandb 0.25.0 tenta il login anche offline).
    os.environ["WANDB_DISABLE_WEAVE"] = "true"
    os.environ["WANDB_PROJECT"] = wandb_cfg.get("project", "neuro-symbolic-t2g")
    os.environ["WANDB_DIR"] = log_dir
    os.environ["WANDB_TAGS"] = ",".join(
        wandb_cfg.get("tags", ["T2G", "grpo", "constrained-decoding"])
    )

    if not wandb.run:
        wandb.init(
            project=wandb_cfg.get("project", "neuro-symbolic-t2g"),
            name=grpo_config.run_name,
            config=config,
            tags=wandb_cfg.get("tags", ["T2G", "grpo"]),
            dir=log_dir,
            mode="offline",
            # ── Fix: output.log missing on Files tab ──────────────────
            # Senza console_multipart W&B bufferizza TUTTO stdout/stderr e
            # scrive output.log solo a wandb.finish() riuscito: su SLURM un
            # job ucciso da OOM/timeout/SIGKILL perde l'intero log. Con
            # multipart scrive chunk incrementali in wandb/run-*/files/logs/,
            # quindi il log parziale sopravvive al crash.
            # https://docs.wandb.ai/models/app/console-logs
            settings=wandb.Settings(
                console_multipart=True,
                console_chunk_max_bytes=1_000_000,
                console_chunk_max_seconds=60,
            ),
        )

    # ── Tee stdout → output.log (sync_cluster download) ─────────────────
    # console_multipart scrive chunk sotto wandb/run-*/files/logs/ ma
    # sync_cluster.ps1 si aspetta un singolo output.log: teniamo entrambi.
    _output_log_path = os.path.join(log_dir, "output.log")
    _sys_stdout = sys.stdout
    _output_log_fh = open(_output_log_path, "a", buffering=1)

    class _Tee:
        def write(self, data):
            _sys_stdout.write(data)
            _output_log_fh.write(data)

        def flush(self):
            _sys_stdout.flush()
            _output_log_fh.flush()

    sys.stdout = _Tee()

    # ── Step 7: Training ─────────────────────────────────────────────────
    log_step(7, "GRPO Training")
    # Live status: the GRPO phase begins (SFT phase, if any, is over).
    live_status_set(
        phase="grpo",
        total_steps=int(config["training"].get("max_steps", 1500)),
        note="GRPO training",
    )

    # ── Workaround: transformers 5.3.0 + peft non espongono  ──────────
    # model.warnings_issued, ma trl 0.24.0 lo usa in GRPOTrainer.__init__.
    if not hasattr(model, "warnings_issued"):
        model.warnings_issued = {}

    # ── Generation kwargs for vocabulary-constrained rollout generation ──
    # trl 0.24.0: generation_kwargs va in GRPOConfig (args), NON in
    # GRPOTrainer.__init__(). NOTE: logits_processor NON puo' stare in
    # generation_kwargs (trl fa GenerationConfig(**generation_kwargs) e
    # transformers 5.3.0 lo rifiuta): serve il monkey-patch di generate().
    gen_kwargs = _build_generation_kwargs(config)
    grpo_config.generation_kwargs = gen_kwargs

    trainer = GRPOTrainer(
        model=model,
        args=grpo_config,
        train_dataset=t2g_dataset,
        reward_funcs=wrapped_reward_fns,
        processing_class=tokenizer,
        callbacks=[sample_callback],
    )

    # ── Defensive: pre-set unsloth-zoo's internal autocast dtype ─────────
    # Belt-and-suspenders accanto al fix env var sopra: rende no-op il lazy
    # init di grpo_accumulated_loss a prescindere da tempi/caching dell'env var.
    trainer._autocast_dtype = torch.bfloat16 if grpo_config.bf16 else torch.float16

    # ── Monkey-patch model.generate() AFTER trainer init ────────────────
    # Patch su trainer.model, non sul modello passato: il trainer puo'
    # wrapparlo diversamente e i rollout TRL usano trainer.model. Due funzioni:
    #   1. AUTOCAST: generate() nei rollout gira FUORI dall'autocast del
    #      trainer. Con 4-bit + LoRA, lm_head resta bfloat16 ma riceve hidden
    #      states float32 → crash "expected scalar type BFloat16 but found
    #      Float"; l'autocast armonizza i dtype.
    #   2. LOGITS PROCESSOR: transformers 5.3.0 GenerationConfig rifiuta
    #      logits_processor come kwarg, ma model.generate() lo accetta:
    #      qui si inietta la maschera vocabolario se la grammar e' attiva.
    _generation_model = trainer.model
    _orig_generate = _generation_model.generate
    _autocast_dtype = torch.bfloat16 if grpo_config.bf16 else torch.float16
    _lp_called = False

    def _patched_generate(*_args: Any, **_kwargs: Any) -> Any:
        nonlocal _lp_called
        if logits_processor_for_gen is not None:
            # CRITICAL: reset the processor's prompt_len cache before each
            # generate(): TRL genera i completions per prompt diversi in
            # sequenza e un prompt_len stantio fa tracciare il Trie
            # dall'offset sbagliato (root cause dei garbage token
            # DEBUTRECHT/HOWEVERY nel run 2026-07-08).
            logits_processor_for_gen.reset()
            _kwargs["logits_processor"] = [logits_processor_for_gen] + _kwargs.get(
                "logits_processor", []
            )
            if not _lp_called:
                _lp_called = True
                print("  [constrained-decoding] logits_processor ACTIVE in generate()")
                allowed_count = 0
                if hasattr(logits_processor_for_gen, "allowed_ids"):
                    allowed_count = len(logits_processor_for_gen.allowed_ids)
                elif hasattr(logits_processor_for_gen, "mask") and hasattr(
                    logits_processor_for_gen.mask, "token_ids"
                ):
                    allowed_count = len(logits_processor_for_gen.mask.token_ids)
                print(f"  [constrained-decoding] allowed tokens: {allowed_count}")
        with torch.autocast(device_type="cuda", dtype=_autocast_dtype):
            return _orig_generate(*_args, **_kwargs)

    _generation_model.generate = _patched_generate  # type: ignore[method-assign]
    print(
        "  model.generate monkey-patched on trainer.model (autocast + logits_processor)"
    )

    # Replace default ProgressCallback with TqdmOnlyProgressCallback
    # (keeps tqdm bar, suppresses duplicate log lines — same as grpo-strict-generation)
    try:
        trainer.remove_callback(ProgressCallback)
        trainer.add_callback(TqdmOnlyProgressCallback)
        trainer.add_callback(HighPrecisionLogCallback())
        trainer.remove_callback(WandbCallback)
    except Exception:
        pass

    # ── Fix: guarantee wandb.finish() even on crash/exception ───────────
    # Senza try/finally un'eccezione in trainer.train() (OOM, CUDA error,
    # kill SLURM) lascia il run wandb "crashed"/unfinished e il log
    # non flushed.
    try:
        print("\n[grpo] Starting GRPO training...")
        trainer.train(resume_from_checkpoint=resume_from)

        # ── Save final model ─────────────────────────────────────────────
        # WHY phase: save_model su NFS scrive GB di pesi in silenzio.
        final_path = Path(grpo_config.output_dir) / "final"
        with phase("Saving final model", detail=str(final_path)):
            trainer.save_model(str(final_path))
            tokenizer.save_pretrained(str(final_path))

        # ── Clean up duplicate final step checkpoint ──────────────────────
        global_step = trainer.state.global_step
        last_ckpt = Path(grpo_config.output_dir) / f"checkpoint-{global_step}"
        if last_ckpt.exists():
            import shutil

            # WHY phase: rmtree di GB di optimizer state su NFS puo'
            # richiedere minuti.
            with phase("Removing duplicate final checkpoint", detail=str(last_ckpt)):
                shutil.rmtree(last_ckpt, ignore_errors=True)
    finally:
        # ── Cleanup ───────────────────────────────────────────────────────
        # WHY phase: wandb.finish() flussha il run offline su NFS (muta con
        # WANDB_SILENT=true); il rilascio puo' bloccarsi su gc/empty_cache.
        if wandb.run:
            with phase("Finalizing wandb run"):
                wandb.finish()

        with phase("Releasing trainer memory"):
            del trainer
            gc.collect()
            torch.cuda.empty_cache()

    print(f"\n{'=' * 60}")
    print("GRPO T2G training complete!")
    print(f"  Model: {final_path}")
    print(f"  Logs:  {config['training']['log_dir']}")


if __name__ == "__main__":
    raise RuntimeError(
        "Do not run this script directly. " "Use 'python -m src.training --config ...'"
    )
