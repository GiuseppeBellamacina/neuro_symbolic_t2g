"""Model loading utilities: HuggingFace model + tokenizer + LoRA + quantization
for the neuro_symbolic_t2g project.

Uses standard HuggingFace backend (transformers + peft + bitsandbytes).
"""

from __future__ import annotations

import logging
from typing import Any

import torch
from peft import (
    LoraConfig,
    PeftModel,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

from src.utils.distributed import is_main_process

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Quantization
# ---------------------------------------------------------------------------


def get_quantization_config(
    quantization: str, dtype: str = "bfloat16"
) -> BitsAndBytesConfig | None:
    """Return a BitsAndBytesConfig based on the quantization string."""
    if quantization == "4bit":
        compute_dtype = getattr(torch, dtype, torch.bfloat16)
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True,
        )
    if quantization == "8bit":
        return BitsAndBytesConfig(load_in_8bit=True)
    return None


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------


def load_tokenizer(model_name: str) -> Any:
    """Load and configure the tokenizer.

    Sets pad_token to eos_token if missing, and forces padding_side="left"
    (required for batched generation).
    """
    if is_main_process():
        logger.info(f"Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.info(
            "pad_token not set → using eos_token (%r)",
            tokenizer.eos_token,
        )
    tokenizer.padding_side = "left"
    return tokenizer


# ---------------------------------------------------------------------------
# Standard HuggingFace backend
# ---------------------------------------------------------------------------


def load_model(
    model_name: str,
    quantization: str = "4bit",
    dtype: str = "bfloat16",
    device_map: str = "auto",
) -> Any:
    """Load a causal LM with optional quantization."""
    torch_dtype = getattr(torch, dtype, torch.bfloat16)
    quant_config = get_quantization_config(quantization, dtype=dtype)
    if is_main_process():
        logger.info(
            "Loading %s (quantization=%s, dtype=%s, device_map=%s)",
            model_name,
            quantization,
            dtype,
            device_map,
        )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quant_config,
        dtype=torch_dtype,  # transformers 5.3.0: 'torch_dtype' → 'dtype'
        device_map=device_map,
        trust_remote_code=True,
    )
    # NOTE: Do NOT manually cast non-quantized layers (lm_head, embed, norms)
    # to bfloat16 here.  prepare_model_for_kbit_training() (called by
    # apply_lora) may re-cast some of them to float32 for gradient stability,
    # creating a dtype mismatch that crashes lm_head.forward().
    # Mixed precision is handled by the trainer's bf16=True via autocast.
    return model


def apply_lora(
    model: Any,
    r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    target_modules: list[str] | None = None,
    task_type: str = "CAUSAL_LM",
) -> Any:
    """Apply LoRA adapters to the model via PEFT."""
    if target_modules is None:
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]

    if is_main_process():
        logger.info(
            "Applying LoRA: r=%d, alpha=%d, dropout=%s, targets=%s",
            r,
            lora_alpha,
            lora_dropout,
            target_modules,
        )

    if getattr(model, "is_loaded_in_4bit", False) or getattr(
        model, "is_loaded_in_8bit", False
    ):
        model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
        task_type=task_type,
        bias="none",
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model


def _load_with_transformers(
    config: dict[str, Any],
    adapter_path: str | None = None,
) -> tuple[Any, Any]:
    """Load via standard HuggingFace transformers + PEFT.

    Args:
        config: Full config dict.
        adapter_path: Optional path to a saved LoRA adapter (e.g. from SFT
            pre-training).  If provided, loads the adapter on top of the base
            model instead of creating a fresh LoRA config.
    """
    model_cfg = config["model"]
    lora_cfg = config.get("lora", {})

    quantization = model_cfg.get("quantization", "4bit")
    if adapter_path:
        logger.info(
            "Adapter path provided → disabling quantization to allow in-memory merging."
        )
        quantization = "none"

    logger.info(
        "Backend: HuggingFace (transformers + peft) — quantization=%s",
        quantization,
    )

    model = load_model(
        model_name=model_cfg["name"],
        quantization=quantization,
        dtype=model_cfg.get("dtype", "bfloat16"),
    )
    tokenizer = load_tokenizer(model_cfg["name"])

    if adapter_path:
        logger.info("Loading existing SFT adapter from: %s", adapter_path)
        # Load adapter as non-trainable, merge it, and unload it to get a pure base model with SFT weights
        model = PeftModel.from_pretrained(model, adapter_path, is_trainable=False)
        logger.info("Merging SFT adapter in-memory...")
        model = model.merge_and_unload()
        logger.info("SFT adapter successfully merged into base model.")

    if lora_cfg:
        model = apply_lora(
            model,
            r=lora_cfg.get("r", 16),
            lora_alpha=lora_cfg.get("lora_alpha", 32),
            lora_dropout=lora_cfg.get("lora_dropout", 0.05),
            target_modules=lora_cfg.get("target_modules"),
            task_type=lora_cfg.get("task_type", "CAUSAL_LM"),
        )

    return model, tokenizer


def _load_with_unsloth(
    config: dict[str, Any],
    adapter_path: str | None = None,
) -> tuple[Any, Any]:
    """Load model + tokenizer via Unsloth's FastLanguageModel."""
    from unsloth import FastLanguageModel

    model_cfg = config["model"]
    lora_cfg = config.get("lora", {})

    quantization = model_cfg.get("quantization", "4bit")
    load_in_4bit = quantization == "4bit"

    if adapter_path:
        if is_main_process():
            logger.info(
                "[unsloth] Adapter path provided → disabling quantization to allow in-memory merging."
            )
        load_in_4bit = False
        quantization = "none"

    model_to_load = adapter_path if adapter_path else model_cfg["name"]

    if is_main_process():
        logger.info(
            "[unsloth] Loading %s (quantization=%s, max_seq_length=%d)",
            model_to_load,
            quantization,
            model_cfg.get("max_seq_length", 2048),
        )

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_to_load,
        max_seq_length=model_cfg.get("max_seq_length", 2048),
        load_in_4bit=load_in_4bit,
        dtype=None,  # auto-detect
    )

    if adapter_path:
        if is_main_process():
            logger.info("[unsloth] Merging SFT adapter in-memory...")
        model = model.merge_and_unload()
        if is_main_process():
            logger.info("[unsloth] SFT adapter successfully merged into base model.")

    if lora_cfg:
        target_modules = lora_cfg.get(
            "target_modules",
            [
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
        )
        if is_main_process():
            logger.info(
                "[unsloth-lora] r=%d, alpha=%d, targets=%s",
                lora_cfg.get("r", 16),
                lora_cfg.get("lora_alpha", 32),
                target_modules,
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

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    return model, tokenizer


def load_model_and_tokenizer(
    config: dict[str, Any],
    adapter_path: str | None = None,
) -> tuple[Any, Any]:
    """High-level loader: model + tokenizer from a config dict.

    Expected config structure::

        model:
          name: "Qwen/Qwen2.5-0.5B-Instruct"
          quantization: "4bit"
          dtype: "bfloat16"
          use_unsloth: true
        lora:  # optional
          r: 16
          lora_alpha: 32
          ...
    """
    model_cfg = config["model"]
    use_unsloth = model_cfg.get("use_unsloth", False)

    if use_unsloth:
        if is_main_process():
            logger.info("Backend: Unsloth")
        return _load_with_unsloth(config, adapter_path=adapter_path)

    return _load_with_transformers(config, adapter_path=adapter_path)
