"""
GRPO T2G Training Loop — Phase 1.

Integrates Constrained Decoding with Group Relative Policy Optimization (GRPO)
for Text-to-Gloss (T2G) translation using Unsloth, vLLM, and TRL.

Architecture:
    1. Load model + tokenizer via Unsloth (with optional vLLM fast inference).
    2. Load ASLG-PC12 dataset and build prompt-completion pairs.
    3. Compute/load bigram transition matrix (Viterbi proxy).
    4. Build gloss vocabulary mask for constrained decoding.
    5. Define reward functions (translation quality + structural proxy).
    6. Train with ``trl.GRPOTrainer``, constraining generation rollouts
       to ASL gloss tokens only.

Usage:
    python -m src.training --config config/grpo_t2g_qwen05.yaml
    CONFIG=config/grpo_t2g_qwen05.yaml sbatch src/cluster/train.sh
"""

from __future__ import annotations

import argparse
import gc
import logging
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
import wandb
from dotenv import load_dotenv
from transformers.integrations.integration_utils import WandbCallback
from transformers.trainer_callback import ProgressCallback
from trl import GRPOConfig, GRPOTrainer  # type: ignore[import]

from datasets import Dataset

from src.data.aslg_dataset import (
    build_t2g_dataset,
    download_aslg_dataset,
    extract_gloss_vocabulary,
    save_vocabulary,
)
from src.data.transition_matrix import (
    compute_bigram_transitions,
    load_transition_matrix,
    save_transition_matrix,
)
from src.rewards.t2g_rewards import (
    build_t2g_reward_functions,
    initialize_rewards,
    register_gold_glosses,
)
from src.grammar.gloss_grammar import GlossVocabularyMask, create_grammarllm_pipeline
from src.grammar.grammar_logits_processor import (
    GlossVocabularyLogitsProcessor,
    GrammarPDALogitsProcessor,
)

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
    if "warmup_ratio" in training_cfg:
        warmup_kwargs["warmup_ratio"] = training_cfg["warmup_ratio"]
    else:
        warmup_kwargs["warmup_steps"] = training_cfg.get("warmup_steps", 50)

    wandb_cfg = (full_config or {}).get("wandb", {})
    from datetime import datetime

    base_name = wandb_cfg.get("run_name", "grpo-t2g")
    run_name = f"{base_name}-{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    return GRPOConfig(
        output_dir=output_dir,
        run_name=run_name,
        max_steps=training_cfg.get("max_steps", 1500),
        per_device_train_batch_size=training_cfg.get(
            "per_device_train_batch_size", 1
        ),
        gradient_accumulation_steps=training_cfg.get(
            "gradient_accumulation_steps", 8
        ),
        learning_rate=training_cfg.get("learning_rate", 5e-6),
        lr_scheduler_type=training_cfg.get("lr_scheduler_type", "cosine"),
        **warmup_kwargs,
        optim=training_cfg.get("optim", "paged_adamw_8bit"),
        weight_decay=training_cfg.get("weight_decay", 0.1),
        max_grad_norm=training_cfg.get("max_grad_norm", 0.1),
        bf16=training_cfg.get("bf16", True),
        logging_steps=training_cfg.get("logging_steps", 5),
        logging_dir=log_dir,
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
# Model loading
# ---------------------------------------------------------------------------


def _load_model_and_tokenizer(
    config: dict[str, Any],
) -> tuple[Any, Any]:
    """Load model and tokenizer with Unsloth or standard HF."""
    model_cfg = config["model"]
    use_unsloth = model_cfg.get("use_unsloth", False)

    if use_unsloth:
        return _load_with_unsloth(config)
    return _load_with_transformers(config)


def _load_with_unsloth(
    config: dict[str, Any],
) -> tuple[Any, Any]:
    """Load via Unsloth's FastLanguageModel (optimized training + vLLM)."""
    from unsloth import FastLanguageModel

    model_cfg = config["model"]
    lora_cfg = config.get("lora", {})

    quantization = model_cfg.get("quantization", "4bit")
    load_in_4bit = quantization == "4bit"

    logger.info(
        f"[unsloth] Loading {model_cfg['name']} "
        f"(4bit={load_in_4bit}, max_seq={model_cfg.get('max_seq_length', 1024)})"
    )

    # vLLM fast inference
    fi_kwargs: dict[str, Any] = {}
    use_fast = model_cfg.get("fast_inference", False)
    if use_fast:
        try:
            import vllm  # noqa: F401
        except ImportError:
            logger.warning("vLLM not available; disabling fast_inference")
            use_fast = False

    if use_fast:
        fi_kwargs["fast_inference"] = True
        fi_kwargs["max_lora_rank"] = lora_cfg.get("r", 16)
        fi_kwargs["gpu_memory_utilization"] = model_cfg.get(
            "gpu_memory_utilization", 0.9
        )
        fi_kwargs["unsloth_vllm_standby"] = model_cfg.get(
            "vllm_standby", False
        )
        logger.info("  fast_inference=ON (vLLM backend)")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_cfg["name"],
        max_seq_length=model_cfg.get("max_seq_length", 1024),
        load_in_4bit=load_in_4bit,
        dtype=None,
        **fi_kwargs,
    )

    # Apply LoRA
    if lora_cfg:
        target_modules = lora_cfg.get(
            "target_modules",
            ["q_proj", "k_proj", "v_proj", "o_proj",
             "gate_proj", "up_proj", "down_proj"],
        )
        logger.info(
            f"[unsloth-lora] r={lora_cfg.get('r', 16)}, "
            f"alpha={lora_cfg.get('lora_alpha', 32)}, "
            f"targets={target_modules}"
        )
        model = FastLanguageModel.get_peft_model(
            model,
            r=lora_cfg.get("r", 16),
            lora_alpha=lora_cfg.get("lora_alpha", 32),
            lora_dropout=lora_cfg.get("lora_dropout", 0),
            target_modules=target_modules,
            use_gradient_checkpointing="unsloth",
            random_state=lora_cfg.get("random_state", 3407),
        )

    # Tokenizer setup
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    return model, tokenizer


def _load_with_transformers(
    config: dict[str, Any],
) -> tuple[Any, Any]:
    """Load via standard HuggingFace transformers + PEFT."""
    from peft import LoraConfig, get_peft_model
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
    )

    model_cfg = config["model"]
    lora_cfg = config.get("lora", {})

    # Quantization
    quant_config = None
    if model_cfg.get("quantization") == "4bit":
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    elif model_cfg.get("quantization") == "8bit":
        quant_config = BitsAndBytesConfig(load_in_8bit=True)

    logger.info(
        f"[transformers] Loading {model_cfg['name']} "
        f"(quantization={model_cfg.get('quantization', 'none')})"
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_cfg["name"],
        quantization_config=quant_config,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_cfg["name"], trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    # Apply LoRA via PEFT
    if lora_cfg:
        from peft import prepare_model_for_kbit_training

        if getattr(model, "is_loaded_in_4bit", False) or getattr(
            model, "is_loaded_in_8bit", False
        ):
            model = prepare_model_for_kbit_training(model)

        peft_config = LoraConfig(
            r=lora_cfg.get("r", 16),
            lora_alpha=lora_cfg.get("lora_alpha", 32),
            lora_dropout=lora_cfg.get("lora_dropout", 0.05),
            target_modules=lora_cfg.get(
                "target_modules",
                ["q_proj", "k_proj", "v_proj", "o_proj"],
            ),
            task_type="CAUSAL_LM",
            bias="none",
        )
        model = get_peft_model(model, peft_config)
        logger.info("[peft] LoRA adapters applied")

    return model, tokenizer


# ---------------------------------------------------------------------------
# Dataset preparation
# ---------------------------------------------------------------------------


def _prepare_t2g_dataset(
    config: dict[str, Any],
    tokenizer: Any,
    vocab: list[str],
) -> Dataset:
    """Load ASLG-PC12 and build prompt-completion pairs for GRPO.

    The dataset has columns: ``prompt``, ``completion`` (gold gloss), ``difficulty``.
    """
    ds_cfg = config["dataset"]
    dataset = download_aslg_dataset(cache_dir=ds_cfg.get("dataset_cache"))

    t2g_ds = build_t2g_dataset(
        dataset,
        split=ds_cfg.get("split", "train"),
        max_samples=ds_cfg.get("max_samples"),
    )

    # Format prompts with a system message for T2G
    system_prompt = (
        "You are an English-to-ASL-gloss translator. "
        "Translate the following English sentence into a sequence of "
        "ASL glosses. Output ONLY the gloss tokens separated by spaces. "
        "Do not include explanations or extra text."
    )

    formatted: list[dict[str, str]] = []
    for i in range(len(t2g_ds)):
        sample = t2g_ds[i]
        text = sample["prompt"]

        # Build prompt in chat format
        if hasattr(tokenizer, "apply_chat_template"):
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": text},
            ]
            try:
                prompt = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            except Exception:
                prompt = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n"
        else:
            prompt = f"Translate to ASL glosses: {text}\nGlosses:"

        formatted.append(
            {
                "prompt": prompt,
                "completion": sample["completion"],
                "difficulty": sample.get("difficulty", "medium"),
            }
        )

    result = Dataset.from_list(formatted)
    logger.info(
        f"[dataset] T2G training set: {len(result)} prompts"
    )
    return result


# ---------------------------------------------------------------------------
# Vocabulary-constrained generation config for GRPO
# ---------------------------------------------------------------------------


def _build_generation_kwargs(
    config: dict[str, Any],
    logits_processor: GlossVocabularyLogitsProcessor,
) -> dict[str, Any]:
    """Build generation kwargs for GRPO rollouts.

    Passed to GRPOTrainer via ``generation_kwargs`` parameter.
    Includes the logits processor for vocabulary-constrained generation.

    Args:
        config: Full config dict.
        logits_processor: The ``GlossVocabularyLogitsProcessor`` instance.

    Returns:
        Dict of generation kwargs compatible with ``model.generate()``.
    """
    grpo_cfg = config["grpo"]
    return {
        "max_new_tokens": grpo_cfg.get("max_completion_length", 256),
        # NOTE: do_sample and temperature are controlled by GRPOConfig,
        # not duplicated here to avoid conflicts.
        "logits_processor": [logits_processor],
    }


# ---------------------------------------------------------------------------
# Main training entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Main entry point for T2G GRPO training."""
    parser = argparse.ArgumentParser(
        description="GRPO training for Text-to-Gloss (T2G) with constrained decoding"
    )
    parser.add_argument(
        "--config", type=str, required=True, help="Path to config YAML"
    )
    parser.add_argument(
        "--resume", action="store_true", help="Resume from latest checkpoint"
    )
    parser.add_argument(
        "--prepare-data",
        action="store_true",
        help="Only prepare data (download dataset, compute transitions, save vocab)",
    )
    args = parser.parse_args()

    import yaml

    with open(args.config, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # ── Setup logging ────────────────────────────────────────────────────
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    # ── Step 1: Data preparation ─────────────────────────────────────────
    ds_cfg = config["dataset"]
    vocab_path = ds_cfg.get("vocab_path", "data/gloss_vocab.txt")
    bigram_path = ds_cfg.get("bigram_matrix_path", "data/bigram_transition.npy")

    logger.info("=" * 60)
    logger.info("STEP 1: Data Preparation")
    logger.info("=" * 60)

    # Download dataset
    dataset = download_aslg_dataset(cache_dir=ds_cfg.get("dataset_cache"))

    # Extract vocabulary (or load from cache)
    if Path(vocab_path).exists():
        from src.data.aslg_dataset import load_vocabulary
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

    logger.info(
        f"Data prepared: |V|={len(vocab)}, "
        f"bigram shape={bigram_matrix.shape}"
    )

    if args.prepare_data:
        logger.info("Data preparation complete. Exiting.")
        return

    # ── Step 2: Model loading ────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 2: Model Loading")
    logger.info("=" * 60)

    model, tokenizer = _load_model_and_tokenizer(config)

    # ── Step 3: Constrained decoding setup ────────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 3: Constrained Decoding Setup")
    logger.info("=" * 60)

    # Determine which constrained decoding strategy to use.
    # Set ``use_grammarllm_pda: true`` in the config to enable the full
    # grammarllm PDA pipeline (LL(1) parsing).  Default is the lightweight
    # vocabulary mask (faster, sufficient for most gloss constraints).
    use_pda = config.get("grammar", {}).get("use_grammarllm_pda", False)

    if use_pda:
        logger.info("Using FULL grammarllm PDA pipeline for constrained decoding")
        logit_processor, streamer, pda = create_grammarllm_pipeline(
            vocab, tokenizer,
            temperature=config["grpo"].get("temperature", 0.7),
        )
        # Wrap in GrammarPDALogitsProcessor for consistent interface
        grammar_lp = GrammarPDALogitsProcessor(
            tokenizer, pda,
            temperature=config["grpo"].get("temperature", 0.7),
        )
        logits_processor_for_gen = grammar_lp
        logger.info("  GrammarLLM PDA pipeline ready")
    else:
        logger.info("Using lightweight GlossVocabularyMask for constrained decoding")
        gloss_mask = GlossVocabularyMask(vocab, tokenizer)
        logits_processor_for_gen = GlossVocabularyLogitsProcessor(
            gloss_mask, device="cuda" if torch.cuda.is_available() else "cpu"
        )
        logger.info("  Vocabulary mask ready")

    # ── vLLM compatibility note ─────────────────────────────────────────
    # When fast_inference is enabled, Unsloth uses vLLM for rollouts.
    # vLLM's sampling engine does NOT use HuggingFace LogitsProcessor —
    # it uses SamplingParams.logit_bias.  Since our constrained decoding
    # relies on LogitsProcessor, we auto-disable fast_inference to
    # guarantee vocabulary constraints are enforced.
    #
    # For production vLLM support, implement logit_bias injection via
    # Unsloth's vLLM integration (see FastLanguageModel docs).
    if config.get("model", {}).get("fast_inference", False):
        logger.warning(
            "⚠️  fast_inference=True is INCOMPATIBLE with constrained decoding "
            "via HF LogitsProcessor.  vLLM does not respect the logits_processor "
            "parameter.  Auto-disabling fast_inference to enforce ASL gloss "
            "vocabulary constraints."
        )
        config["model"]["fast_inference"] = False
        logger.info(
            "    → fast_inference=False (constrained decoding will use "
            "standard HF generation with logits_processor)"
        )

    # ── Step 4: Dataset preparation ──────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 4: Dataset Preparation")
    logger.info("=" * 60)

    t2g_dataset = _prepare_t2g_dataset(config, tokenizer, vocab)

    # Register gold glosses for the translation quality reward function.
    # Maps formatted_prompt → gold_gloss so the reward can look up
    # ground-truth targets without embedding them in the prompt.
    register_gold_glosses(
        formatted_prompts=list(t2g_dataset["prompt"]),
        gold_glosses=list(t2g_dataset["completion"]),
    )

    # ── Step 5: Reward functions ─────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 5: Reward Functions")
    logger.info("=" * 60)

    initialize_rewards(bigram_matrix, vocab)
    reward_fns, reward_weights = build_t2g_reward_functions(
        config.get("reward")
    )

    # ── Step 6: GRPO configuration ───────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 6: GRPO Configuration")
    logger.info("=" * 60)

    grpo_config = _build_grpo_config(
        config["training"],
        config["grpo"],
        config,
        reward_weights=reward_weights,
    )

    logger.info(
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
            logger.info(f"Resuming from {resume_from}")

    # ── Wandb setup ──────────────────────────────────────────────────────
    wandb_cfg = config.get("wandb", {})
    os.environ["WANDB_PROJECT"] = wandb_cfg.get("project", "neuro-symbolic-t2g")
    os.environ["WANDB_DIR"] = config["training"]["log_dir"]
    os.environ["WANDB_TAGS"] = ",".join(
        wandb_cfg.get("tags", ["grpo", "t2g", "constrained-decoding"])
    )

    if not wandb.run:
        wandb.init(
            project=wandb_cfg.get("project", "neuro-symbolic-t2g"),
            name=grpo_config.run_name,
            config=config,
            tags=wandb_cfg.get("tags", ["grpo", "t2g"]),
            dir=config["training"]["log_dir"],
        )

    # ── Step 7: Training ─────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 7: GRPO Training")
    logger.info("=" * 60)

    # ── Generation kwargs for vocabulary constraint ──
    # These are passed to GRPOTrainer which forwards them to model.generate()
    # during rollout exploration.  This ensures EVERY generated token is
    # constrained to the ASL gloss vocabulary.
    gen_kwargs = _build_generation_kwargs(config, logits_processor_for_gen)

    trainer = GRPOTrainer(
        model=model,
        args=grpo_config,
        train_dataset=t2g_dataset,
        reward_funcs=reward_fns,
        processing_class=tokenizer,
        generation_kwargs=gen_kwargs,
        callbacks=[],
    )

    # Remove default callbacks that conflict
    try:
        trainer.remove_callback(ProgressCallback)
        trainer.remove_callback(WandbCallback)
    except Exception:
        pass

    logger.info("Starting GRPO training...")
    trainer.train(resume_from_checkpoint=resume_from)

    # ── Save final model ─────────────────────────────────────────────────
    final_path = Path(grpo_config.output_dir) / "final"
    logger.info(f"Saving final model to {final_path}...")
    trainer.save_model(str(final_path))
    tokenizer.save_pretrained(str(final_path))

    # ── Cleanup ──────────────────────────────────────────────────────────
    if wandb.run:
        wandb.finish()

    del trainer
    gc.collect()
    torch.cuda.empty_cache()

    logger.info("=" * 60)
    logger.info("GRPO T2G training complete!")
    logger.info(f"  Model: {final_path}")
    logger.info(f"  Logs:  {config['training']['log_dir']}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
