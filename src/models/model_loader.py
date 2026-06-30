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
    # Explicitly cast non-quantized layers (e.g. lm_head, embed, norm)
    # to bfloat16 — bitsandbytes 4-bit only handles quantized linear layers.
    if quant_config is not None:
        for name, param in model.named_parameters():
            if param.dtype == torch.float32:
                param.data = param.data.to(torch_dtype)
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
) -> tuple[Any, Any]:
    """Load via standard HuggingFace transformers + PEFT."""
    model_cfg = config["model"]
    lora_cfg = config.get("lora", {})

    logger.info(
        "Backend: HuggingFace (transformers + peft) — quantization=%s",
        model_cfg.get("quantization", "none"),
    )

    model = load_model(
        model_name=model_cfg["name"],
        quantization=model_cfg.get("quantization", "4bit"),
        dtype=model_cfg.get("dtype", "bfloat16"),
    )
    tokenizer = load_tokenizer(model_cfg["name"])

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


def load_model_and_tokenizer(
    config: dict[str, Any],
) -> tuple[Any, Any]:
    """High-level loader: model + tokenizer from a config dict.

    Expected config structure::

        model:
          name: "Qwen/Qwen2.5-0.5B-Instruct"
          quantization: "4bit"
          dtype: "bfloat16"
        lora:  # optional
          r: 16
          lora_alpha: 32
          ...
    """
    return _load_with_transformers(config)
