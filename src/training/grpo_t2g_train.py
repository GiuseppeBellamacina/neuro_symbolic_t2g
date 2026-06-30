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
    python -m src.training --config experiments/configs/t2g/grpo_qwen05.yaml
    CONFIG=experiments/configs/t2g/grpo_qwen05.yaml sbatch cluster/train.sh
"""

from __future__ import annotations

import argparse
import gc
import hashlib

# ── Workaround: _is_package_available in transformers 5.3.0 restituisce  ─
# una TUPLA (bool, str) invece di un bool.  In Python, una tupla non vuota
# è sempre truthy → (False, None) è True → trl prova a importare mergekit e
# llm_blender anche quando non sono installati.
# Fix: usiamo importlib per caricare trl.import_utils bypassando il pigro
# __getattr__, correggiamo le variabili a False, poi importiamo GRPOTrainer.
import importlib
import logging
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch

_trl_iu = importlib.import_module("trl.import_utils")  # noqa: E402
if isinstance(_trl_iu._mergekit_available, tuple):
    _trl_iu._mergekit_available = False
if isinstance(_trl_iu._llm_blender_available, tuple):
    _trl_iu._llm_blender_available = False

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
from src.grammar.gloss_grammar import GlossVocabularyMask, create_grammarllm_pipeline
from src.grammar.grammar_logits_processor import (
    GlossVocabularyLogitsProcessor,
    GrammarPDALogitsProcessor,
)
from src.models.model_loader import load_model_and_tokenizer
from src.rewards.t2g_rewards import (
    build_t2g_reward_functions,
    initialize_rewards,
    register_gold_glosses,
)
from src.utils.config import load_config
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
    run_name = f"{base_name}-{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # Set tensorboard logging dir via env var (logging_dir kwarg is deprecated
    # since transformers 5.2).
    os.environ.setdefault("TENSORBOARD_LOGGING_DIR", log_dir)

    return GRPOConfig(
        output_dir=output_dir,
        run_name=run_name,
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
    )


# ---------------------------------------------------------------------------
# Dataset preparation
# ---------------------------------------------------------------------------


def _prepare_t2g_dataset(
    config: dict[str, Any],
    tokenizer: Any,
    vocab: list[str],
    dataset: Any = None,
) -> Dataset:
    """Load ASLG-PC12 and build prompt-completion pairs for GRPO.

    The dataset has columns: ``prompt``, ``completion`` (gold gloss), ``difficulty``.

    Args:
        config: Full config dict.
        tokenizer: Hugging Face tokenizer.
        vocab: Gloss vocabulary (unused here, kept for API compatibility).
        dataset: Optional pre-loaded ``DatasetDict``. If ``None``, downloads it.
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

    # Format prompts with the centralized T2G prompt builder.
    # This guarantees train/eval/test use identical formatting.
    formatted: list[dict[str, str]] = []
    for i in range(len(t2g_ds)):
        sample = t2g_ds[i]
        text = sample["prompt"]

        prompt = build_t2g_prompt(text, tokenizer)

        # Stable sample ID: SHA256 of the user instruction (English sentence).
        # This survives any prompt format changes by TRL, enabling reliable
        # gold gloss lookup in reward functions.
        sample_id = hashlib.sha256(
            str(text).encode("utf-8", errors="replace")
        ).hexdigest()

        formatted.append(
            {
                "prompt": prompt,
                "completion": sample["completion"],
                "difficulty": sample.get("difficulty", "medium"),
                "sample_id": sample_id,
            }
        )

    result = Dataset.from_list(formatted)
    logger.info(f"[dataset] T2G training set: {len(result)} prompts")
    return result


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
        "max_new_tokens": grpo_cfg.get("max_completion_length", 256),
    }
    return kwargs


# ---------------------------------------------------------------------------
# Main training entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Main entry point for T2G GRPO training."""
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
    args = parser.parse_args()

    config = load_config(args.config)

    # Safe config access: support both 'grpo' (GRPO) and 'generation' (SFT) keys
    grpo_cfg = config.get("generation", config.get("grpo", {}))

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
    print(f"[grpo] Reproducibility: seed={seed} (random, numpy, torch, cuda)")

    # ── Step 1: Data preparation ─────────────────────────────────────────
    ds_cfg = config["dataset"]
    vocab_path = ds_cfg.get("vocab_path", "data/gloss_vocab.txt")
    bigram_path = ds_cfg.get("bigram_matrix_path", "data/bigram_transition.npy")

    print(f"\n{'=' * 60}")
    print("STEP 1: Data Preparation")

    # Download dataset
    dataset = download_aslg_dataset(
        cache_dir=ds_cfg.get("dataset_cache"), seed=ds_cfg.get("seed", 42)
    )

    # Extract vocabulary (or load from cache)
    if Path(vocab_path).exists():
        from src.datasets.aslg_dataset import load_vocabulary

        vocab = load_vocabulary(vocab_path)
    else:
        vocab = extract_gloss_vocabulary(dataset, split="train")
        save_vocabulary(vocab, vocab_path)

    # Compute transition matrix (or load from cache)
    if Path(bigram_path).exists():
        bigram_matrix = load_transition_matrix(bigram_path)
    else:
        bigram_matrix = compute_bigram_transitions(
            dataset, vocab, split="train", smoothing=1.0
        )
        save_transition_matrix(bigram_matrix, bigram_path)

    print(f"  Data prepared: |V|={len(vocab)}, bigram shape={bigram_matrix.shape}")

    if args.prepare_data:
        print("Data preparation complete. Exiting.")
        return

    # ── Step 2: Model loading ────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print("STEP 2: Model Loading")

    model, tokenizer = load_model_and_tokenizer(config)

    # ── Step 3: Constrained decoding setup ────────────────────────────────
    print(f"\n{'=' * 60}")
    print("STEP 3: Constrained Decoding Setup")

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
        # Determine which constrained decoding strategy to use.
        # Set ``use_grammarllm_pda: true`` in the config to enable the full
        # grammarllm PDA pipeline (LL(1) parsing).  Default is lightweight
        # vocabulary mask (faster, sufficient for most gloss constraints).
        use_pda = config.get("grammar", {}).get("use_grammarllm_pda", False)

        if use_pda:
            print("  Using FULL grammarllm PDA pipeline for constrained decoding")
            logit_processor, streamer, pda = create_grammarllm_pipeline(
                vocab,
                tokenizer,
                temperature=grpo_cfg.get("temperature", 0.7),
            )
            # Wrap in GrammarPDALogitsProcessor for consistent interface
            grammar_lp = GrammarPDALogitsProcessor(
                tokenizer,
                pda,
                temperature=grpo_cfg.get("temperature", 0.7),
            )
            logits_processor_for_gen = grammar_lp
            print("  GrammarLLM PDA pipeline ready")
        else:
            print("  Using lightweight GlossVocabularyMask for constrained decoding")
            gloss_mask = GlossVocabularyMask(vocab, tokenizer)
            logits_processor_for_gen = GlossVocabularyLogitsProcessor(
                gloss_mask, device="cuda" if torch.cuda.is_available() else "cpu"
            )
            print("  Vocabulary mask ready")

    # ── Step 4: Dataset preparation ──────────────────────────────────────
    print(f"\n{'=' * 60}")
    print("STEP 4: Dataset Preparation")

    t2g_dataset = _prepare_t2g_dataset(config, tokenizer, vocab, dataset=dataset)

    # Register gold glosses for the translation quality reward function.
    # Uses stable sample IDs (SHA256 of user instruction) for format-agnostic
    # matching, so TRL prompt reformatting doesn'’t break the lookup.
    register_gold_glosses(
        sample_ids=list(t2g_dataset["sample_id"]),
        gold_glosses=list(t2g_dataset["completion"]),
    )

    # ── Step 5: Reward functions ─────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print("STEP 5: Reward Functions")

    initialize_rewards(
        bigram_matrix,
        vocab,
        viterbi_diversity=config.get("grammar", {}).get("viterbi_diversity"),
    )
    reward_fns, reward_weights = build_t2g_reward_functions(config.get("reward"))

    # ── Wire completion sample logging (for live chain_monitor display) ─
    from src.training.callbacks import (
        CompletionSampleCallback,
        CompletionSampleLogger,
    )

    sample_logger = CompletionSampleLogger(reward_fns, reward_weights, n_samples=3)
    sample_logger.set_difficulty_map(t2g_dataset)
    wrapped_reward_fns = sample_logger.wrapped_reward_fns
    sample_callback = CompletionSampleCallback(
        sample_logger,
        every_n_steps=5,
        logits_processor=logits_processor_for_gen,
    )

    # ── Step 6: GRPO configuration ───────────────────────────────────────
    print(f"\n{'=' * 60}")
    print("STEP 6: GRPO Configuration")

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
        wandb_cfg.get("tags", ["grpo", "t2g", "constrained-decoding"])
    )

    if not wandb.run:
        wandb.init(
            project=wandb_cfg.get("project", "neuro-symbolic-t2g"),
            name=grpo_config.run_name,
            config=config,
            tags=wandb_cfg.get("tags", ["grpo", "t2g"]),
            dir=log_dir,
            mode="offline",
        )

    # ── Step 7: Training ─────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print("STEP 7: GRPO Training")

    # ── Workaround: transformers 5.3.0 + peft non espongono  ──────────
    # model.warnings_issued, ma trl 0.24.0 lo usa in GRPOTrainer.__init__.
    if not hasattr(model, "warnings_issued"):
        model.warnings_issued = {}

    # ── Generation kwargs for vocabulary-constrained rollout generation ──
    # In trl 0.24.0, generation_kwargs goes into GRPOConfig (args), NOT
    # directly into GRPOTrainer.__init__().
    # NOTE: logits_processor CANNOT be in generation_kwargs because trl
    # 0.24.0 does GenerationConfig(**generation_kwargs) and transformers
    # 5.3.0 rejects logits_processor in GenerationConfig.
    # Workaround: monkey-patch model.generate() to inject the processor.
    gen_kwargs = _build_generation_kwargs(config)
    grpo_config.generation_kwargs = gen_kwargs

    # ── Inject logits_processor via monkey-patch ─────────────────────
    # transformers 5.3.0 GenerationConfig rejects logits_processor, but
    # model.generate() accepts it.  Wrap model.generate so the processor
    # is always passed during GRPO rollouts.
    if logits_processor_for_gen is not None:
        _orig_generate = model.generate

        def _patched_generate(*_args: Any, **_kwargs: Any) -> Any:
            # Merge: vocabulary mask FIRST, then any existing processors.
            _kwargs["logits_processor"] = [logits_processor_for_gen] + _kwargs.get(
                "logits_processor", []
            )
            return _orig_generate(*_args, **_kwargs)

        model.generate = _patched_generate  # type: ignore[method-assign]
        print("  logits_processor injected via model.generate monkey-patch")

    trainer = GRPOTrainer(
        model=model,
        args=grpo_config,
        train_dataset=t2g_dataset,
        reward_funcs=wrapped_reward_fns,
        processing_class=tokenizer,
        callbacks=[sample_callback],
    )

    # Remove default callbacks that conflict
    try:
        trainer.remove_callback(ProgressCallback)
        trainer.remove_callback(WandbCallback)
    except Exception:
        pass

    print("\n[grpo] Starting GRPO training...")
    trainer.train(resume_from_checkpoint=resume_from)

    # ── Save final model ─────────────────────────────────────────────────
    final_path = Path(grpo_config.output_dir) / "final"
    print(f"\n[grpo] Saving final model to {final_path}...")
    trainer.save_model(str(final_path))
    tokenizer.save_pretrained(str(final_path))

    # ── Cleanup ──────────────────────────────────────────────────────────
    if wandb.run:
        wandb.finish()

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
