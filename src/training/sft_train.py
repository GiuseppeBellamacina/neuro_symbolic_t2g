"""
SFT T2G Training Script — Text-to-Gloss Supervised Fine-Tuning.

Trains Qwen2.5-0.5B-Instruct via teacher forcing on gold ASL gloss sequences
using ``trl.SFTTrainer``.  No reward shaping, no constrained decoding —
the model simply learns to replicate the gold gloss given the English input.

Usage:
    python -m src.training --config experiments/configs/t2g/sft.yaml
    CONFIG=experiments/configs/t2g/sft.yaml sbatch cluster/train.sh
"""

from __future__ import annotations

import argparse
import gc
import logging
import os
import random
import sys
import warnings
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


def _prepare_sft_dataset(
    config: dict[str, Any],
    tokenizer: Any,
    dataset: Any = None,
) -> Dataset:
    """Build conversation pairs for SFT: prompt → gold gloss.

    SFTTrainer needs a single ``"text"`` column with the full conversation
    (chat template applied).  We build prompt+gold pairs using the same
    centralized ``build_t2g_prompt`` used in GRPO training and evaluation.

    Args:
        config: Full config dict.
        tokenizer: Hugging Face tokenizer.
        dataset: Optional pre-loaded ``DatasetDict``. If ``None``, downloads it.

    Returns:
        HuggingFace ``Dataset`` with a ``"text"`` column.
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

    conversations: list[dict[str, str]] = []
    for i in range(len(t2g_ds)):
        sample = t2g_ds[i]
        text = sample["prompt"]
        gold = sample["completion"]

        # Build full conversation: system prompt + user text + assistant gold gloss
        # This aligns formatting exactly with the GRPO rollout prompt structure.
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": text},
            {"role": "assistant", "content": gold},
        ]
        try:
            full_text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )
        except Exception:
            # Fallback: concatenate manually with ChatML format
            full_text = (
                f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
                f"<|im_start|>user\n{text}<|im_end|>\n"
                f"<|im_start|>assistant\n{gold}<|im_end|>"
            )

        conversations.append({"text": full_text})

    result = Dataset.from_list(conversations)
    logger.info(
        "[sft] SFT dataset: %d conversation pairs " "(columns=%s)",
        len(result),
        list(result.column_names),
    )
    return result


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

    sft_dataset = _prepare_sft_dataset(config, tokenizer, dataset=dataset)

    # Log a few sample pairs for verification
    logger.info("[sft] Sample conversation pairs (first 2):")
    for i in range(min(2, len(sft_dataset))):
        text = sft_dataset[i]["text"]
        # Extract just the user instruction and gold gloss for compact display
        user_marker = "<|im_start|>user\n"
        asst_marker = "<|im_start|>assistant\n"
        if user_marker in text and asst_marker in text:
            user_text = text.split(user_marker)[1].split("<|im_end|>")[0].strip()
            gold_text = text.split(asst_marker)[1].split("<|im_end|>")[0].strip()
            logger.info("[sft]   #%d  EN: %s", i, user_text[:80])
            logger.info("[sft]        GOLD: %s", gold_text[:80])
        else:
            logger.info("[sft]   #%d  (raw) %s", i, text[:100])

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

    sft_config = SFTConfig(
        output_dir=output_dir,
        run_name=run_name,
        seed=training_cfg.get("seed", config["dataset"].get("seed", 42)),
        num_train_epochs=training_cfg.get("num_train_epochs", 3),
        per_device_train_batch_size=training_cfg.get("per_device_train_batch_size", 4),
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
        save_total_limit=training_cfg.get("save_total_limit", 3),
        max_length=training_cfg.get(
            "max_seq_length", 768
        ),  # renamed from max_seq_length in TRL 0.20+
        gradient_checkpointing=training_cfg.get("gradient_checkpointing", False),
        dataset_text_field="text",
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
        len(sft_dataset),
        sft_config.per_device_train_batch_size * sft_config.gradient_accumulation_steps,
        max(
            1,
            len(sft_dataset)
            // (
                sft_config.per_device_train_batch_size
                * sft_config.gradient_accumulation_steps
            ),
        )
        * sft_config.num_train_epochs,
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
    from transformers.trainer_callback import ProgressCallback

    from src.training.callbacks import (
        HighPrecisionLogCallback,
        SFTSampleCallback,
        TqdmOnlyProgressCallback,
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=sft_dataset,
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

    # SFT sample + loss tracking callback for visibility into pre-training
    sft_sample_cb = SFTSampleCallback(
        tokenizer=tokenizer,
        model=model,
        dataset=sft_dataset,
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

        # ── Save final model ─────────────────────────────────────────────
        final_path = Path(output_dir) / "final"
        logger.info("Saving final model to %s...", final_path)
        trainer.save_model(str(final_path))
        tokenizer.save_pretrained(str(final_path))
        final_path_str = str(final_path)

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
