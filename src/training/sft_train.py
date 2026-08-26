"""
SFT T2G Training Script — Text-to-Gloss Supervised Fine-Tuning.

Trains Qwen2.5-0.5B-Instruct via teacher forcing on gold ASL gloss sequences
using ``trl.SFTTrainer``.  No reward shaping, no constrained decoding —
the model simply learns to replicate the gold gloss given the English input.

The dataset uses trl's native prompt-completion conversational format
(``prompt`` = ``[system, user]`` message list, ``completion`` = the gold
gloss) with ``completion_only_loss=True``, so the loss masks the prompt
tokens and only the gold gloss is counted.  A small seeded holdout is carved
from the train split and used with early stopping to guard against
overfitting.  Prompt formatting is identical to the GRPO rollout prompts
(see ``src/utils/prompting.py``).

Usage:
    python -m src.training --config experiments/configs/t2g/sft.yaml
    CONFIG=experiments/configs/t2g/sft.yaml sbatch cluster/train.sh
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import logging
import os
import random
import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import wandb

# Silence noisy transformers FutureWarnings (AttentionMaskConverter deprecation)
warnings.filterwarnings("ignore", category=FutureWarning, module="transformers")
warnings.filterwarnings(
    "ignore",
    message=".*AttentionMaskConverter.*",
    category=FutureWarning,
)
from dotenv import load_dotenv
from trl import SFTConfig, SFTTrainer  # type: ignore[import]

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
from src.models.model_loader import load_model_and_tokenizer
from src.utils.config import load_config
from src.utils.prompting import SYSTEM_PROMPT

load_dotenv()

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# SFT dataset preparation
# ---------------------------------------------------------------------------


def _build_prompt_completion_example(sample: dict[str, Any]) -> dict[str, Any]:
    """Convert a raw T2G row into a prompt-completion SFT example.

    ``prompt`` is the conversational message list ``[system, user]`` and
    ``completion`` the single ``[assistant]`` gold-gloss message.  trl 0.24
    tokenizes the prompt with ``apply_chat_template(prompt,
    add_generation_prompt=True)`` (``trl/trainer/sft_trainer.py:956-962``),
    producing byte-identical prompts to the GRPO rollout path
    (``build_t2g_prompt`` in ``src/utils/prompting.py``).  The full sequence
    is tokenized from ``prompt + completion`` and a ``completion_mask`` marks
    everything after the prompt (``trl/trainer/sft_trainer.py:997-1000``), so
    with ``completion_only_loss=True`` the loss only counts the gold gloss.

    Args:
        sample: Row from ``build_t2g_dataset`` (``prompt``, ``completion``,
            ``difficulty``; optionally ``gold_gloss`` and ``sample_id``).

    Returns:
        Dict with ``prompt`` and ``completion`` message lists plus the
        metadata columns ``gold_gloss``, ``difficulty``, ``sample_id``.
    """
    text = str(sample["prompt"]).strip()
    gold = str(sample.get("gold_gloss") or sample["completion"]).strip()
    return {
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": text},
        ],
        "completion": [{"role": "assistant", "content": gold}],
        "gold_gloss": gold,
        "difficulty": str(sample.get("difficulty", "medium")),
        "sample_id": str(
            sample.get("sample_id")
            or hashlib.sha256(text.encode("utf-8")).hexdigest()
        ),
    }


def split_eval_holdout(
    dataset: Dataset,
    eval_fraction: float = 0.02,
    seed: int = 42,
) -> tuple[Dataset, Dataset]:
    """Split a Hugging Face ``Dataset`` into ``(train, eval)`` holdout subsets.

    Seeded and deterministic (``Dataset.train_test_split`` with a fixed
    ``seed``), so the same inputs always produce the same partition and the
    two subsets are disjoint by construction.  Used to carve a small held-out
    set from the ASLG train split for evaluation and early stopping.

    Args:
        dataset: Source ``Dataset`` (the built SFT prompt-completion set).
        eval_fraction: Fraction of rows held out for evaluation.  Values
            ``<= 0`` return an empty eval set.
        seed: RNG seed for the shuffle (use the dataset seed).

    Returns:
        ``(train_ds, eval_ds)``.
    """
    if eval_fraction <= 0.0:
        return dataset, dataset.select([])
    split = dataset.train_test_split(
        test_size=eval_fraction, seed=seed, shuffle=True
    )
    return split["train"], split["test"]


def _prepare_sft_dataset(
    config: dict[str, Any],
    dataset: Any = None,
) -> tuple[Dataset, Dataset]:
    """Build prompt-completion train/eval datasets for SFT.

    Uses trl 0.24's native prompt-completion conversational format: the
    ``prompt`` column is the message list ``[system, user]`` and the
    ``completion`` column the single ``[assistant]`` gold-gloss message
    (see ``_build_prompt_completion_example``).  ``SFTTrainer`` tokenizes
    these with the tokenizer chat template and builds a ``completion_mask``
    (``trl/trainer/sft_trainer.py:949-1000``), so the loss — with
    ``completion_only_loss=True`` — only covers the gold gloss.

    A small seeded holdout is carved from the train split
    (``config["training"]["eval_fraction"]``, default 0.02) for evaluation
    and early stopping.

    Args:
        config: Full config dict.
        dataset: Optional pre-loaded ``DatasetDict``. If ``None``, downloads it.

    Returns:
        ``(train_ds, eval_ds)`` pair with columns ``prompt``, ``completion``,
        ``gold_gloss``, ``difficulty``, ``sample_id``.
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

    rows = [_build_prompt_completion_example(sample) for sample in t2g_ds]
    sft_ds = Dataset.from_list(rows)
    logger.info(
        "[sft] SFT dataset: %d prompt-completion pairs (columns=%s)",
        len(sft_ds),
        list(sft_ds.column_names),
    )

    eval_fraction = config.get("training", {}).get("eval_fraction", 0.02)
    seed = ds_cfg.get("seed", 42)
    train_ds, eval_ds = split_eval_holdout(
        sft_ds, eval_fraction=eval_fraction, seed=seed
    )
    logger.info(
        "[sft] Eval holdout: eval_fraction=%.3f → train=%d, eval=%d",
        eval_fraction,
        len(train_ds),
        len(eval_ds),
    )
    return train_ds, eval_ds


# ---------------------------------------------------------------------------
# SFT adapter fingerprint & reuse
# ---------------------------------------------------------------------------

#: Version of the fingerprint schema.  Bump when the fingerprinted fields
#: change (e.g. a new field starts affecting the adapter) — the version is
#: part of the hash, so a bump invalidates every previously stored
#: fingerprint automatically.
_SFT_FINGERPRINT_VERSION = 1

#: Training keys that never affect the SFT adapter weights (paths/timestamps).
_NON_DETERMINISTIC_TRAINING_KEYS = ("output_dir", "log_dir", "run_timestamp", "trainer")


def _sft_training_fingerprint_source(config: dict[str, Any]) -> dict[str, Any]:
    """SFT-relevant training hyperparameters (path/timestamp keys excluded).

    In the GRPO flow ``sft_config["sft_pretrain"]["training"]`` carries the
    SFT hyperparameters, while the merged ``training`` section additionally
    holds GRPO-only keys such as ``max_steps`` that must NOT invalidate the
    SFT adapter.  The standalone ``sft.yaml`` flow has no ``sft_pretrain``
    section, so the effective ``training`` section is used instead.
    """
    pretrain_training = config.get("sft_pretrain", {}).get("training", {})
    if isinstance(pretrain_training, dict) and pretrain_training:
        training = dict(pretrain_training)
    else:
        training = dict(config.get("training", {}))
    for key in _NON_DETERMINISTIC_TRAINING_KEYS:
        training.pop(key, None)
    return training


def _sft_fingerprint_payload(config: dict[str, Any]) -> dict[str, Any]:
    """The exact dict hashed to produce the SFT fingerprint.

    Contains every field that determines the SFT adapter (model + loading,
    LoRA shape, dataset selection, SFT hyperparameters, system prompt).
    Output/log paths and run timestamps are deliberately excluded — they
    never affect the adapter weights.
    """
    model = config.get("model", {})
    lora = config.get("lora", {})
    dataset = config.get("dataset", {})
    return {
        "version": _SFT_FINGERPRINT_VERSION,
        "model": {
            key: model.get(key)
            for key in ("name", "quantization", "dtype", "use_unsloth")
            if key in model
        },
        "lora": {
            key: lora.get(key)
            for key in (
                "r",
                "lora_alpha",
                "lora_dropout",
                "target_modules",
                "random_state",
            )
            if key in lora
        },
        "dataset": {
            key: dataset.get(key)
            for key in ("dataset_name", "seed", "split", "max_samples", "thinking")
            if key in dataset
        },
        "sft_training": _sft_training_fingerprint_source(config),
        "system_prompt": SYSTEM_PROMPT,
    }


def compute_sft_fingerprint(sft_config: dict[str, Any]) -> str:
    """SHA-256 fingerprint of everything that determines an SFT adapter.

    Two runs with the same fingerprint are expected to produce equivalent
    adapters, so the SFT phase can be skipped and the previously saved
    adapter reused.  Changing any SFT hyperparameter, the model, the LoRA
    config, the dataset or the system prompt changes the fingerprint and
    forces a retrain.

    Args:
        sft_config: Full config dict as passed to :func:`run_sft`.

    Returns:
        64-char hex SHA-256 of the canonical JSON of the fingerprinted
        fields.
    """
    canonical = json.dumps(
        _sft_fingerprint_payload(sft_config), sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def write_sft_fingerprint(
    final_path: str | Path, sft_config: dict[str, Any]
) -> Path:
    """Write ``sft_fingerprint.json`` next to a freshly-trained SFT adapter.

    The file records the fingerprint plus the fingerprinted config so a
    later GRPO run can decide whether this adapter is reusable.  It is
    written ONLY after training completed (called at the end of
    :func:`run_sft`); a ``final/`` without it is never reused.

    Args:
        final_path: Directory of the saved SFT adapter (``.../final``).
        sft_config: Full config dict used to train the adapter.

    Returns:
        Path of the written ``sft_fingerprint.json`` file.
    """
    document = {
        "fingerprint": compute_sft_fingerprint(sft_config),
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "config": _sft_fingerprint_payload(sft_config),
    }
    final_dir = Path(final_path)
    final_dir.mkdir(parents=True, exist_ok=True)
    out = final_dir / "sft_fingerprint.json"
    out.write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return out


def is_complete_adapter_dir(path: str | Path) -> bool:
    """Whether *path* looks like a loadable (PEFT/merged) adapter directory.

    Accepts a PEFT LoRA adapter (``adapter_config.json`` + weights) or a
    merged model directory (``config.json`` + weights).  Used to guard
    adapter reuse: a directory containing only ``sft_fingerprint.json`` is
    NOT a usable adapter.

    Args:
        path: Candidate adapter directory.

    Returns:
        ``True`` if the directory has a config file AND weight files.
    """
    d = Path(path)
    if not d.is_dir():
        return False
    has_config = (d / "adapter_config.json").is_file() or (d / "config.json").is_file()
    has_weights = any(
        (d / name).is_file()
        for name in (
            "adapter_model.safetensors",
            "adapter_model.bin",
            "model.safetensors",
            "pytorch_model.bin",
        )
    )
    return has_config and has_weights


def _sft_fingerprint_candidates(parent: Path) -> list[Path]:
    """All ``run_*/sft_pretrain/final/sft_fingerprint.json`` under *parent*.

    Sorted by modification time, most recent first.
    """
    return sorted(
        parent.glob("run_*/sft_pretrain/final/sft_fingerprint.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )


def find_reusable_sft_adapter(
    model_ckpt_parent: str | Path, fingerprint: str
) -> Path | None:
    """Find a previously-trained SFT adapter matching *fingerprint*.

    Scans sibling ``run_*/sft_pretrain/final`` directories under
    *model_ckpt_parent* (e.g. ``experiments/checkpoints/qwen25-05b-optimal``)
    and returns the most recently modified adapter whose
    ``sft_fingerprint.json`` matches and whose weight files are intact.  A
    candidate whose fingerprint matches but whose adapter files are missing
    is skipped (logged loudly) in favour of the next candidate.

    Args:
        model_ckpt_parent: Directory containing the ``run_*`` subdirectories
            of the model family.
        fingerprint: Expected SFT fingerprint (see
            :func:`compute_sft_fingerprint`).

    Returns:
        Path to the reusable adapter directory (``.../final``), or ``None``
        if no candidate matches.
    """
    parent = Path(model_ckpt_parent)
    if not parent.is_dir():
        return None
    candidates = _sft_fingerprint_candidates(parent)
    for candidate in candidates:
        try:
            meta = json.loads(candidate.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            logger.warning(
                "[sft-reuse] Unreadable sft_fingerprint.json, skipping: %s", candidate
            )
            continue
        if meta.get("fingerprint") != fingerprint:
            continue
        adapter_dir = candidate.parent
        if not is_complete_adapter_dir(adapter_dir):
            logger.warning(
                "[sft-reuse] Fingerprint match but adapter files missing, "
                "skipping: %s",
                adapter_dir,
            )
            continue
        logger.info("[sft-reuse] Reusable SFT adapter found: %s", adapter_dir)
        return adapter_dir
    if candidates:
        logger.warning(
            "[sft-reuse] No matching SFT adapter found "
            "(checked %d candidate run(s)) — training SFT",
            len(candidates),
        )
    return None


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def run_sft(config: dict[str, Any], resume: bool = False) -> str:
    """Run SFT training and return the path to the saved adapter.

    This function is designed to be called from grpo_t2g_train.py for
    SFT pre-training before GRPO.  It aggressively cleans up GPU memory
    when done so GRPO can use the full VRAM.

    Args:
        config: Full config dict (same format as YAML).

    Returns:
        Path to the saved SFT LoRA adapter directory.
    """

    # ── Setup logging ────────────────────────────────────────────────────
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
    )
    # Quiet down external libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("datasets").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)

    # ── Set random seeds for reproducibility ─────────────────────────────
    seed = config["dataset"].get("seed", 42)
    random.seed(seed)
    np.random.seed(seed)  # noqa: NPY002
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    logger.info("Reproducibility: seed=%d (random, numpy, torch, cuda)", seed)

    # ── Step 1: Data preparation ─────────────────────────────────────────
    ds_cfg = config["dataset"]
    vocab_path = ds_cfg.get("vocab_path", "data/gloss_vocab.txt")
    bigram_path = ds_cfg.get("bigram_matrix_path", "data/bigram_transition.npy")

    logger.info("=" * 60)
    logger.info("STEP 1: Data Preparation")
    logger.info("=" * 60)

    dataset = download_aslg_dataset(
        cache_dir=ds_cfg.get("dataset_cache"), seed=ds_cfg.get("seed", 42)
    )

    # Vocabulary (needed for eval compatibility)
    if Path(vocab_path).exists():
        from src.datasets.aslg_dataset import load_vocabulary

        vocab = load_vocabulary(vocab_path)
    else:
        vocab = extract_gloss_vocabulary(dataset, split="train")
        save_vocabulary(vocab, vocab_path)

    # Bigram matrix (needed for eval compatibility)
    if Path(bigram_path).exists():
        bigram_matrix = load_transition_matrix(bigram_path)
    else:
        bigram_matrix = compute_bigram_transitions(
            dataset, vocab, split="train", smoothing=1.0
        )
        save_transition_matrix(bigram_matrix, bigram_path)

    logger.info(
        "Data prepared: |V|=%d, bigram shape=%s",
        len(vocab),
        bigram_matrix.shape,
    )

    # (prepare-data is handled in main(), not here)

    # ── Step 2: Model loading ────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 2: Model Loading")
    logger.info("=" * 60)

    model, tokenizer = load_model_and_tokenizer(config)

    # ── Step 3: SFT dataset preparation ──────────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 3: SFT Dataset Preparation")
    logger.info("=" * 60)

    sft_train_ds, sft_eval_ds = _prepare_sft_dataset(config, dataset=dataset)

    # Log a few sample pairs for verification
    logger.info("[sft] Sample prompt-completion pairs (first 2):")
    for i in range(min(2, len(sft_train_ds))):
        sample = sft_train_ds[i]
        user_text = sample["prompt"][-1]["content"]
        gold_text = sample["completion"][0]["content"]
        logger.info("[sft]   #%d  EN: %s", i, user_text[:80])
        logger.info("[sft]        GOLD: %s", gold_text[:80])

    # ── Step 4: SFT configuration ────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 4: SFT Configuration")
    logger.info("=" * 60)

    from datetime import datetime

    training_cfg = config["training"]
    base_output_dir = Path(training_cfg["output_dir"])
    base_log_dir = Path(training_cfg["log_dir"])

    # Check if a run_ directory is already in the parent paths (GRPO sub-phase)
    is_subphase = any(part.startswith("run_") for part in base_output_dir.parts)

    if is_subphase:
        output_dir = base_output_dir
        log_dir = base_log_dir
        logger.info("SFT running as GRPO sub-phase. Using path: %s", output_dir)
        run_timestamp = next(
            (
                part.removeprefix("run_")
                for part in reversed(base_output_dir.parts)
                if part.startswith("run_")
            ),
            datetime.now().strftime("%Y%m%d_%H%M%S"),
        )
    else:
        run_timestamp = None
        if resume:
            run_folders = sorted(base_output_dir.glob("run_*"))
            if run_folders:
                output_dir = run_folders[-1]
                run_timestamp = output_dir.name.removeprefix("run_")
                log_dir = base_log_dir / f"run_{run_timestamp}"
                logger.info("Resuming SFT in existing directory: %s", output_dir)
            else:
                logger.warning(
                    "No existing run directory found in %s to resume. Creating a new run.",
                    base_output_dir,
                )

        if run_timestamp is None:
            run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = base_output_dir / f"run_{run_timestamp}"
            log_dir = base_log_dir / f"run_{run_timestamp}"
            logger.info("Starting new SFT run. Output dir: %s", output_dir)

    output_dir = str(output_dir)
    log_dir = str(log_dir)
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    Path(log_dir).mkdir(parents=True, exist_ok=True)

    wandb_cfg = config.get("wandb", {})
    base_name = wandb_cfg.get("run_name", "sft-t2g")
    run_name = f"{base_name}-{run_timestamp}"

    # Set tensorboard logging dir via env var (logging_dir kwarg is deprecated
    # since transformers 5.2).
    os.environ.setdefault("TENSORBOARD_LOGGING_DIR", log_dir)

    eval_enabled = len(sft_eval_ds) > 0
    if not eval_enabled:
        logger.warning(
            "[sft] eval_fraction<=0 → eval_strategy='no', early stopping disabled"
        )

    sft_config = SFTConfig(
        output_dir=output_dir,
        run_name=run_name,
        seed=training_cfg.get("seed", config["dataset"].get("seed", 42)),
        num_train_epochs=training_cfg.get("num_train_epochs", 3),
        per_device_train_batch_size=training_cfg.get("per_device_train_batch_size", 4),
        per_device_eval_batch_size=training_cfg.get("per_device_eval_batch_size", 8),
        gradient_accumulation_steps=training_cfg.get("gradient_accumulation_steps", 4),
        learning_rate=training_cfg.get("learning_rate", 2e-5),
        lr_scheduler_type=training_cfg.get("lr_scheduler_type", "cosine"),
        warmup_steps=training_cfg.get("warmup_steps", 100),
        optim=training_cfg.get("optim", "paged_adamw_8bit"),
        weight_decay=training_cfg.get("weight_decay", 0.1),
        max_grad_norm=training_cfg.get("max_grad_norm", 1.0),
        bf16=training_cfg.get("bf16", True),
        logging_steps=training_cfg.get("logging_steps", 10),
        save_steps=training_cfg.get("save_steps", 200),
        save_total_limit=training_cfg.get("save_total_limit", 2),
        max_length=training_cfg.get(
            "max_seq_length", 768
        ),  # renamed from max_seq_length in TRL 0.20+
        gradient_checkpointing=training_cfg.get("gradient_checkpointing", False),
        # ── Loss masking: only the gold gloss (completion) counts ───────
        # The dataset uses trl's prompt-completion conversational format, so
        # SFTTrainer builds a completion_mask and, when completion_only_loss
        # is None (default), auto-enables completion-only loss for
        # prompt-completion datasets (trl/trainer/sft_trainer.py:733-739).
        # Set explicitly for clarity and forward-compatibility.
        completion_only_loss=True,
        # ── Held-out eval + early stopping (overfitting guard) ──────────
        # ``sft_pretrain.training`` is merged into ``training`` by
        # grpo_t2g_train.py, so eval_fraction/eval_steps/... are read here
        # from the same key regardless of the entry point.
        eval_strategy="steps" if eval_enabled else "no",
        eval_steps=training_cfg.get("eval_steps", 200),
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        load_best_model_at_end=eval_enabled,
        report_to="wandb",
    )

    logger.info(
        "[sft] epochs=%d, batch=%d, grad_accum=%d, lr=%.1e, max_len=%d",
        sft_config.num_train_epochs,
        sft_config.per_device_train_batch_size,
        sft_config.gradient_accumulation_steps,
        sft_config.learning_rate,
        sft_config.max_length,
    )
    logger.info(
        "[sft] warmup=%d, weight_decay=%.3f, scheduler=%s, optim=%s, bf16=%s",
        sft_config.warmup_steps,
        sft_config.weight_decay,
        sft_config.lr_scheduler_type,
        sft_config.optim,
        sft_config.bf16,
    )
    logger.info(
        "[sft] dataset_size=%d, effective_batch=%d, total_optim_steps≈%d",
        len(sft_train_ds),
        sft_config.per_device_train_batch_size * sft_config.gradient_accumulation_steps,
        max(
            1,
            len(sft_train_ds)
            // (
                sft_config.per_device_train_batch_size
                * sft_config.gradient_accumulation_steps
            ),
        )
        * sft_config.num_train_epochs,
    )
    logger.info(
        "[sft] eval: strategy=%s, eval_steps=%d, eval_size=%d, "
        "completion_only_loss=%s",
        sft_config.eval_strategy,
        sft_config.eval_steps,
        len(sft_eval_ds),
        sft_config.completion_only_loss,
    )

    # ── Resume logic ─────────────────────────────────────────────────────
    resume_from: str | None = None
    if resume:
        ckpts = sorted(Path(output_dir).glob("checkpoint-*"))
        if ckpts:
            resume_from = str(ckpts[-1])
            logger.info("Resuming from %s", resume_from)

    # ── Wandb setup ──────────────────────────────────────────────────────
    # Modalità offline — come grpo-strict-generation.
    if "WANDB_MODE" not in os.environ:
        os.environ["WANDB_MODE"] = "offline"
    # Disable weave (wandb 0.25.0 tenta il login anche offline).
    os.environ["WANDB_DISABLE_WEAVE"] = "true"
    os.environ["WANDB_PROJECT"] = wandb_cfg.get("project", "neuro-symbolic-t2g")
    os.environ["WANDB_DIR"] = log_dir
    os.environ["WANDB_TAGS"] = ",".join(
        wandb_cfg.get("tags", ["sft", "t2g", "supervised"])
    )

    if not wandb.run:
        wandb.init(
            project=wandb_cfg.get("project", "neuro-symbolic-t2g"),
            name=run_name,
            config=config,
            tags=wandb_cfg.get("tags", ["sft", "t2g"]),
            dir=log_dir,
            mode="offline",
            # ── Fix: output.log missing on Files tab ──────────────────
            # See grpo_t2g_train.py for full explanation: without
            # console_multipart, W&B only flushes output.log on a clean
            # wandb.finish(). SLURM OOM/timeout kills lose the log entirely.
            settings=wandb.Settings(
                console_multipart=True,
                console_chunk_max_bytes=1_000_000,
                console_chunk_max_seconds=60,
            ),
        )

    # ── Tee stdout → output.log (sync_cluster download) ─────────────────
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

    # ── Step 5: Training ─────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 5: SFT Training")
    logger.info("=" * 60)

    # ── Workaround: transformers 5.3.0 + peft non espongono  ──────────
    # model.warnings_issued.
    if not hasattr(model, "warnings_issued"):
        model.warnings_issued = {}

    from transformers.integrations.integration_utils import WandbCallback
    from transformers.trainer_callback import EarlyStoppingCallback, ProgressCallback

    from src.training.callbacks import (
        HighPrecisionLogCallback,
        SFTSampleCallback,
        TqdmOnlyProgressCallback,
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=sft_train_ds,
        eval_dataset=sft_eval_ds if eval_enabled else None,
        processing_class=tokenizer,
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

    # Early stopping on eval_loss: stop if it does not improve for
    # `early_stopping_patience` evaluations (guards overfitting on the
    # prompt-redundant 78K train samples).
    if eval_enabled:
        trainer.add_callback(
            EarlyStoppingCallback(
                early_stopping_patience=training_cfg.get(
                    "early_stopping_patience", 3
                )
            )
        )

    # SFT sample + loss tracking callback for visibility into pre-training
    sft_sample_cb = SFTSampleCallback(
        tokenizer=tokenizer,
        model=model,
        dataset=sft_train_ds,
        every_n_steps=training_cfg.get("logging_steps", 10) * 5,
        sample_every_n_steps=training_cfg.get("sft_sample_every_n_steps", 100),
        n_samples=2,
    )
    trainer.add_callback(sft_sample_cb)

    # ── Fix: guarantee wandb.finish() even on crash/exception ───────────
    # See grpo_t2g_train.py for full explanation.
    final_path_str: str
    try:
        logger.info("Starting SFT training...")
        trainer.train(resume_from_checkpoint=resume_from)

        # ── Best metric (tracked by load_best_model_at_end) ─────────────
        best_metric = getattr(trainer.state, "best_metric", None)
        if best_metric is not None:
            logger.info(
                "[sft] Best eval_loss=%.6f (best checkpoint=%s)",
                best_metric,
                trainer.state.best_model_checkpoint,
            )
        else:
            logger.info("[sft] No eval metric tracked (evaluation disabled).")

        # ── Save final model ─────────────────────────────────────────────
        # With load_best_model_at_end=True the trainer already re-loaded the
        # best checkpoint weights, so `final` holds the best adapter.
        final_path = Path(output_dir) / "final"
        logger.info("Saving final model to %s...", final_path)
        trainer.save_model(str(final_path))
        tokenizer.save_pretrained(str(final_path))
        final_path_str = str(final_path)

        # ── Record SFT fingerprint (adapter reuse in the GRPO flow) ──────
        # Written ONLY after a completed training: a ``final/`` without
        # sft_fingerprint.json is never reused by grpo_t2g_train.
        try:
            fingerprint_path = write_sft_fingerprint(final_path, config)
            logger.info("SFT fingerprint written to %s", fingerprint_path)
        except Exception as exc:  # metadata only — never fail a completed training
            logger.warning("[sft] Failed to write sft_fingerprint.json: %s", exc)

        # ── Clean up duplicate final step checkpoint ──────────────────────
        global_step = trainer.state.global_step
        last_ckpt = Path(output_dir) / f"checkpoint-{global_step}"
        if last_ckpt.exists():
            import shutil

            logger.info(
                "Cleaning up duplicate final step checkpoint folder: %s", last_ckpt
            )
            shutil.rmtree(last_ckpt, ignore_errors=True)
    finally:
        # ── Cleanup (aggressive: free VRAM for GRPO phase) ───────────────
        if wandb.run:
            wandb.finish()

        del trainer, model
        gc.collect()
        torch.cuda.empty_cache()

    logger.info("=" * 60)
    logger.info("SFT T2G training complete!")
    logger.info("  Model: %s", final_path_str)
    logger.info("  Logs:  %s", log_dir)
    logger.info("=" * 60)

    return final_path_str


def main() -> None:
    """Standalone entry point for SFT training (used by __main__.py)."""
    parser = argparse.ArgumentParser(description="SFT training for Text-to-Gloss (T2G)")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    parser.add_argument(
        "--resume", action="store_true", help="Resume from latest checkpoint"
    )
    parser.add_argument(
        "--prepare-data",
        action="store_true",
        help="Only prepare data (download dataset, compute transitions, save vocab)",
    )
    args = parser.parse_args()

    config = load_config(args.config)

    if args.prepare_data:
        # Handle prepare-data separately
        ds_cfg = config["dataset"]
        from src.datasets.aslg_dataset import download_aslg_dataset

        download_aslg_dataset(
            cache_dir=ds_cfg.get("dataset_cache"), seed=ds_cfg.get("seed", 42)
        )
        print("Data preparation complete.")
        return

    run_sft(config, resume=args.resume)


if __name__ == "__main__":
    raise RuntimeError(
        "Do not run this script directly. "
        "Use 'python -m src.training --config ...' to ensure "
        "Unsloth is imported before trl/transformers/peft for optimizations."
    )
