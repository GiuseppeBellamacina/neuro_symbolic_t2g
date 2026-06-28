"""
SFT T2G Training Script — Text-to-Gloss Supervised Fine-Tuning.

Trains Qwen2.5-0.5B-Instruct via teacher forcing on gold ASL gloss sequences
using ``trl.SFTTrainer``.  No reward shaping, no constrained decoding —
the model simply learns to replicate the gold gloss given the English input.

Usage:
    python -m src.training.sft_train --config experiments/configs/t2g/sft.yaml
    CONFIG=experiments/configs/t2g/sft.yaml sbatch cluster/train.sh
"""

from __future__ import annotations

import argparse
import gc
import logging
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import wandb
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
from src.utils.prompting import build_t2g_prompt

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

        # Build prompt with centralized template (same as training/eval)
        prompt = build_t2g_prompt(text, tokenizer)

        # Build full conversation: user prompt + assistant gold gloss
        # Apply chat template to get a single string for SFTTrainer
        messages = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": gold},
        ]
        try:
            full_text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )
        except Exception:
            # Fallback: concatenate manually
            full_text = f"{prompt}\n{gold}"

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


def main() -> None:
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

    # ── Setup logging ────────────────────────────────────────────────────
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

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

    if args.prepare_data:
        logger.info("Data preparation complete. Exiting.")
        return

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

    # ── Step 4: SFT configuration ────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 4: SFT Configuration")
    logger.info("=" * 60)

    training_cfg = config["training"]
    output_dir = training_cfg["output_dir"]
    log_dir = training_cfg["log_dir"]
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    Path(log_dir).mkdir(parents=True, exist_ok=True)

    wandb_cfg = config.get("wandb", {})
    from datetime import datetime

    base_name = wandb_cfg.get("run_name", "sft-t2g")
    run_name = f"{base_name}-{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    sft_config = SFTConfig(
        output_dir=output_dir,
        run_name=run_name,
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
        logging_dir=log_dir,
        save_steps=training_cfg.get("save_steps", 200),
        save_total_limit=training_cfg.get("save_total_limit", 3),
        max_seq_length=training_cfg.get("max_seq_length", 768),
        dataset_text_field="text",
        report_to="wandb",
    )

    logger.info(
        "[sft] epochs=%d, batch=%d, grad_accum=%d, lr=%.1e, " "max_seq=%d",
        sft_config.num_train_epochs,
        sft_config.per_device_train_batch_size,
        sft_config.gradient_accumulation_steps,
        sft_config.learning_rate,
        sft_config.max_seq_length,
    )

    # ── Resume logic ─────────────────────────────────────────────────────
    resume_from: str | None = None
    if args.resume:
        ckpts = sorted(Path(output_dir).glob("checkpoint-*"))
        if ckpts:
            resume_from = str(ckpts[-1])
            logger.info("Resuming from %s", resume_from)

    # ── Wandb setup ──────────────────────────────────────────────────────
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
        )

    # ── Step 5: Training ─────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 5: SFT Training")
    logger.info("=" * 60)

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=sft_dataset,
        processing_class=tokenizer,
    )

    logger.info("Starting SFT training...")
    trainer.train(resume_from_checkpoint=resume_from)

    # ── Save final model ─────────────────────────────────────────────────
    final_path = Path(output_dir) / "final"
    logger.info("Saving final model to %s...", final_path)
    trainer.save_model(str(final_path))
    tokenizer.save_pretrained(str(final_path))

    # ── Cleanup ──────────────────────────────────────────────────────────
    if wandb.run:
        wandb.finish()

    del trainer
    gc.collect()
    torch.cuda.empty_cache()

    logger.info("=" * 60)
    logger.info("SFT T2G training complete!")
    logger.info("  Model: %s", final_path)
    logger.info("  Logs:  %s", log_dir)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
